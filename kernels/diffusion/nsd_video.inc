// nsd_video.inc: the video decoders', the latent and pixel upscalers' and the audio decoder's operations of nextsycl's
// nsd.h (H3's interface), on CUDA. Part of nsd.cu's translation unit (included at its end): Ctx, scratch, get / put,
// gemm_nt and the conv3d_ex kernels are its. The arithmetic is nextsycl's nsd.cpp; its oneDNN convolution (conv3d) is
// here an unfolding (im2col) in tiles and one cuBLAS GEMM a tile, accumulated in float32.

namespace {

// rows of x [M, C] scaled in place, row r by s[r]
__global__ void vid_scale_rows_k(void* x, int dt, int64_t M, int64_t C, const float* s) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * C) return;
    put(x, dt, i, get(x, dt, i) * s[i / C]);
}

// conv3d, stride 1, zero padding k / 2: col [P, k*k*k*Ci] for voxels [p0, p0 + P) of [T, H, W], (kt, ky, kx, ci) order
// (that of reorder_w3d_k's weights)
__global__ void vid_im2col3d_pad_k(const void* x, int dt, int64_t T, int64_t H, int64_t W, int64_t Ci, int64_t k, int64_t p0, int64_t P,
                                   void* col) {
    const int64_t KK = k * k * k * Ci;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P * KK) return;
    const int64_t p = i / KK, j = i % KK;
    const int64_t ci = j % Ci, t = j / Ci, ix = t % k, iy = (t / k) % k, it = t / (k * k);
    const int64_t g = p0 + p, ox = g % W, oy = (g / W) % H, ot = g / (W * H);
    const int64_t tt = ot + it - k / 2, yy = oy + iy - k / 2, xx = ox + ix - k / 2;
    float v = 0.0f;
    if (tt >= 0 && tt < T && yy >= 0 && yy < H && xx >= 0 && xx < W) v = get(x, dt, ((tt * H + yy) * W + xx) * Ci + ci);
    put(col, dt, i, v);
}

// GroupNorm, pass 1: partial sums (s, s^2) of part `part` of the voxels of (frame f, group g); a block of 64 lanes
constexpr int64_t kGnParts = 256;
__global__ void vid_gn_partial_k(const void* x, int dt, int64_t N, int64_t C, int64_t G, int64_t cg, int64_t per, float* gs) {
    const int64_t gp = blockIdx.x, fg = gp / kGnParts, part = gp % kGnParts, f = fg / G, g = fg % G;
    const int64_t lane = threadIdx.x;
    float s = 0.0f, s2 = 0.0f;
    const int64_t n0 = part * per, n1 = min(N, n0 + per);
    for (int64_t n = n0; n < n1; ++n)
        for (int64_t j = lane; j < cg; j += 64) {
            const float v = get(x, dt, (f * N + n) * C + g * cg + j);
            s += v; s2 += v * v;
        }
    s = block_sum(s);
    s2 = block_sum(s2);
    if (lane == 0) { gs[gp * 2] = s; gs[gp * 2 + 1] = s2; }
}

// pass 2: (mean, 1 / std) per (frame, group)
__global__ void vid_gn_stat_k(const float* gs, int64_t FG, int64_t N, int64_t cg, float eps, float* stat) {
    const int64_t fg = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (fg >= FG) return;
    float s = 0.0f, s2 = 0.0f;
    for (int64_t p = 0; p < kGnParts; ++p) { s += gs[(fg * kGnParts + p) * 2]; s2 += gs[(fg * kGnParts + p) * 2 + 1]; }
    const float cnt = (float) N * (float) cg, mean = s / cnt, var = fmaxf(s2 / cnt - mean * mean, 0.0f);
    stat[fg * 2] = mean;
    stat[fg * 2 + 1] = rsqrtf(var + eps);
}

// pass 3: normalize, affine, the optional modulation, SiLU
__global__ void vid_gn_apply_k(const void* x, int dt, int64_t F, int64_t N, int64_t C, int64_t G, int64_t cg, const float* stat,
                               const float* weight, const float* bias, const float* scale, const float* shift, void* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= F * N * C) return;
    const int64_t fn = i / C, ch = i % C;
    const int64_t g = (fn / N) * G + ch / cg;
    float v = (get(x, dt, i) - stat[g * 2]) * stat[g * 2 + 1] * weight[ch] + bias[ch];
    if (scale) v = v * (1.0f + scale[ch]) + shift[ch];
    put(out, dt, i, v / (1.0f + expf(-v)));
}

// depthwise along T (kernel K, zero padding K / 2), x and out [T, P, C], w [C, K]
__global__ void vid_temporal_dwconv_k(const void* x, int dt, int64_t T, int64_t P, int64_t C, const float* w, int64_t K, const float* bias,
                                      void* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * P * C) return;
    const int64_t ch = i % C, p = (i / C) % P, t = i / (C * P);
    float acc = bias ? bias[ch] : 0.0f;
    for (int64_t k = 0; k < K; ++k) {
        const int64_t ts = t + k - K / 2;
        if (ts >= 0 && ts < T) acc += w[ch * K + k] * get(x, dt, (ts * P + p) * C + ch);
    }
    put(out, dt, i, acc);
}

// align_corners=False source coordinate of output index d
__device__ __forceinline__ void vid_src_coord(int64_t d, float scale, int64_t in, int64_t& i0, int64_t& i1, float& l) {
    float s = ((float) d + 0.5f) * scale - 0.5f;
    if (s < 0.0f) s = 0.0f;
    i0 = (int64_t) s;
    if (i0 > in - 1) i0 = in - 1;
    i1 = i0 < in - 1 ? i0 + 1 : i0;
    l = s - (float) i0;
}

__global__ void vid_trilinear_k(const void* x, int dt, int64_t T, int64_t H, int64_t W, int64_t C, int64_t To, int64_t Ho, int64_t Wo,
                                float st, float sh, float sw, void* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= To * Ho * Wo * C) return;
    const int64_t o = i / C, ch = i % C;
    const int64_t ot = o / (Ho * Wo), oh = (o / Wo) % Ho, ow = o % Wo;
    int64_t t0, t1, h0, h1, w0, w1;
    float lt, lh, lw;
    vid_src_coord(ot, st, T, t0, t1, lt);
    vid_src_coord(oh, sh, H, h0, h1, lh);
    vid_src_coord(ow, sw, W, w0, w1, lw);
    auto v = [&](int64_t t, int64_t h, int64_t w) { return get(x, dt, ((t * H + h) * W + w) * C + ch); };
    const float a = (1 - lw) * v(t0, h0, w0) + lw * v(t0, h0, w1), b = (1 - lw) * v(t0, h1, w0) + lw * v(t0, h1, w1);
    const float cc = (1 - lw) * v(t1, h0, w0) + lw * v(t1, h0, w1), d = (1 - lw) * v(t1, h1, w0) + lw * v(t1, h1, w1);
    const float r = (1 - lt) * ((1 - lh) * a + lh * b) + lt * ((1 - lh) * cc + lh * d);
    put(out, dt, i, r);
}

// zeros in front along T, the spatial border reflected (PyTorch's "reflect")
__global__ void vid_pad3d_k(const void* x, int dt, int64_t T, int64_t H, int64_t W, int64_t C, int64_t front, int64_t top, int64_t left,
                            int64_t To, int64_t Ho, int64_t Wo, void* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= To * Ho * Wo * C) return;
    const int64_t o = i / C, ch = i % C;
    const int64_t t = o / (Ho * Wo) - front, h0 = (o / Wo) % Ho - top, w0 = o % Wo - left;
    float v = 0.0f;
    if (t >= 0) {
        const int64_t h = h0 < 0 ? -h0 : (h0 >= H ? 2 * (H - 1) - h0 : h0);
        const int64_t w = w0 < 0 ? -w0 : (w0 >= W ? 2 * (W - 1) - w0 : w0);
        v = get(x, dt, ((t * H + h) * W + w) * C + ch);
    }
    put(out, dt, i, v);
}

__global__ void vid_prelu_k(void* x, int dt, int64_t M, int64_t C, const float* alpha) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * C) return;
    const float v = get(x, dt, i);
    put(x, dt, i, v > 0.0f ? v : alpha[i % C] * v);
}

__global__ void vid_pixel_shuffle_add_k(const void* x, int dt, int64_t N, int64_t H, int64_t W, int64_t C, int64_t r, const void* base,
                                        void* out) {
    const int64_t Ho = H * r, Wo = W * r;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * Ho * Wo * C) return;
    const int64_t ch = i % C, xo = (i / C) % Wo, nyo = i / (C * Wo), n = nyo / Ho, yo = nyo % Ho;
    const int64_t y = yo / r, ii = yo % r, xx = xo / r, j = xo % r;
    const int64_t src = ((n * H + y) * W + xx) * (C * r * r) + (ch * r * r + ii * r + j);
    const int64_t b = ((n * H + y) * W + xx) * C + ch;
    put(out, dt, i, get(x, dt, src) + get(base, dt, b));
}

// x [N, Hi, Wi, C] -> out float32 planar [C, N, Ho, Wo]: adaptive average pooling, clamped to [0, 1]
__global__ void vid_resize_area_k(const void* x, int dt, int64_t N, int64_t Hi, int64_t Wi, int64_t C, int64_t Ho, int64_t Wo, float* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C * N * Ho * Wo) return;
    const int64_t xo = i % Wo, yo = (i / Wo) % Ho, cn = i / (Wo * Ho), ch = cn / N, n = cn % N;
    // [floor(o * In / Out), ceil((o + 1) * In / Out))
    const int64_t y0 = (yo * Hi) / Ho, y1 = ((yo + 1) * Hi + Ho - 1) / Ho;
    const int64_t x0 = (xo * Wi) / Wo, x1 = ((xo + 1) * Wi + Wo - 1) / Wo;
    float s = 0.0f;
    for (int64_t y = y0; y < y1; ++y)
        for (int64_t xx = x0; xx < x1; ++xx) s += get(x, dt, ((n * Hi + y) * Wi + xx) * C + ch);
    const float v = s / (float) ((y1 - y0) * (x1 - x0));
    out[i] = fminf(fmaxf(v, 0.0f), 1.0f);
}

// BigVGAN's anti-aliased SnakeBeta in one pass (see nsd_aa_snake)
__global__ void vid_aa_snake_k(const float* x, int64_t B, int64_t C, int64_t L, const float* log_alpha, const float* log_beta, const float* up,
                               const float* down, float* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * C * L) return;
    const int64_t n = i % L, bc = i / L, ch = bc % C;
    const float* xr = x + bc * L;
    const float alpha = expf(log_alpha[ch]), inv_beta = 1.0f / (expf(log_beta[ch]) + 1e-9f);
    float acc = 0.0f;
    for (int k = 0; k < 12; ++k) {
        // the upsampled signal (length 2L) at j, replicate-padded by 5 on the left
        int64_t j = 2 * n + k - 5;
        j = j < 0 ? 0 : (j > 2 * L - 1 ? 2 * L - 1 : j);
        float u = 0.0f;
        // up[j] = 2 sum_k' f[k'] xp[(j + 15 - k') / 2] over even j + 15 - k'; xp[p] = x[clamp(p - 5)]
        for (int kk = (int) ((j + 15) & 1); kk < 12; kk += 2) {
            int64_t p = (j + 15 - kk) / 2 - 5;
            p = p < 0 ? 0 : (p > L - 1 ? L - 1 : p);
            u += up[kk] * xr[p];
        }
        u *= 2.0f;
        const float s = sinf(alpha * u);
        acc += down[k] * (u + inv_beta * s * s);
    }
    out[i] = acc;
}

}  // namespace

int nsd_scale_rows(void* ctx, void* x, int dt, int64_t M, int64_t C, const float* s) {
    auto* c = (Ctx*) ctx;
    CU(cudaSetDevice(c->dev));
    if (M <= 0 || C <= 0) return 0;
    vid_scale_rows_k<<<nb(M * C), 256, 0, c->s>>>(x, dt, M, C, s);
    LAUNCHED("scale_rows");
    return 0;
}

int nsd_conv3d(void* ctx, const void* x, int dt, int64_t T, int64_t H, int64_t W, int64_t Ci, const void* w, int64_t Co, int64_t k,
               const float* bias, void* out) {
    auto* c = (Ctx*) ctx;
    CU(cudaSetDevice(c->dev));
    if (T <= 0 || H <= 0 || W <= 0) return 0;
    const int64_t px = T * H * W;
    if (k == 1) {
        // w [Co, Ci, 1, 1, 1] is already the GEMM's [N, K]
        if (int rc = gemm_nt(c, x, dt, px, Ci, w, dt, Co, out, dt, 0.0f)) return rc;
        if (bias) { bias_k<<<nb(px * Co), 256, 0, c->s>>>(out, dt, px, Co, bias); LAUNCHED("bias"); }
        return 0;
    }
    const int64_t KK = k * k * k * Ci;
    // a tile of output voxels: its unfolding under ~192 MiB
    const int64_t P = std::max<int64_t>(1, std::min<int64_t>(px, (192ll << 20) / (std::max<int64_t>(KK, 1) * (int64_t) esz(dt))));
    const size_t wbytes = (size_t) Co * KK * esz(dt);
    char* wr = (char*) scratch(c, 2, wbytes + (size_t) P * KK * esz(dt));
    if (!wr) return fail("nsd_conv3d: the im2col buffer did not fit in memory");
    char* col = wr + wbytes;
    reorder_w3d_k<<<nb(Co * KK), 256, 0, c->s>>>(w, dt, Co, Ci, k, k, k, wr);
    LAUNCHED("conv3d weights");
    for (int64_t p0 = 0; p0 < px; p0 += P) {
        const int64_t r = std::min(P, px - p0);
        vid_im2col3d_pad_k<<<nb(r * KK), 256, 0, c->s>>>(x, dt, T, H, W, Ci, k, p0, r, col);
        LAUNCHED("conv3d im2col");
        char* o = (char*) out + (size_t) p0 * Co * esz(dt);
        if (int rc = gemm_nt(c, col, dt, r, KK, wr, dt, Co, o, dt, 0.0f)) return rc;
        if (bias) { bias_k<<<nb(r * Co), 256, 0, c->s>>>(o, dt, r, Co, bias); LAUNCHED("bias"); }
    }
    return 0;
}

int nsd_group_norm_silu(void* ctx, const void* x, int dt, int64_t F, int64_t N, int64_t C, int64_t G, const float* weight, const float* bias,
                        float eps, const float* scale, const float* shift, void* out) {
    auto* c = (Ctx*) ctx;
    CU(cudaSetDevice(c->dev));
    if (F <= 0 || N <= 0 || C <= 0) return 0;
    const int64_t cg = C / G;
    // partial sums, then the statistics: in scratch 0 (the attention's score table; the stream orders the uses)
    float* gs = (float*) scratch(c, 0, (size_t) (F * G * kGnParts * 2 + F * G * 2) * 4);
    if (!gs) return fail("nsd_group_norm_silu: the statistics did not fit in memory");
    float* stat = gs + F * G * kGnParts * 2;
    const int64_t per = (N + kGnParts - 1) / kGnParts;
    vid_gn_partial_k<<<(unsigned) (F * G * kGnParts), 64, 0, c->s>>>(x, dt, N, C, G, cg, per, gs);
    LAUNCHED("group_norm partial sums");
    vid_gn_stat_k<<<nb(F * G), 256, 0, c->s>>>(gs, F * G, N, cg, eps, stat);
    LAUNCHED("group_norm statistics");
    vid_gn_apply_k<<<nb(F * N * C), 256, 0, c->s>>>(x, dt, F, N, C, G, cg, stat, weight, bias, scale, shift, out);
    LAUNCHED("group_norm_silu");
    return 0;
}

int nsd_temporal_dwconv(void* ctx, const void* x, int dt, int64_t T, int64_t P, int64_t C, const float* w, int64_t K,
                        const float* bias, void* out) {
    auto* c = (Ctx*) ctx;
    CU(cudaSetDevice(c->dev));
    if (T <= 0 || P <= 0 || C <= 0) return 0;
    vid_temporal_dwconv_k<<<nb(T * P * C), 256, 0, c->s>>>(x, dt, T, P, C, w, K, bias, out);
    LAUNCHED("temporal_dwconv");
    return 0;
}

int nsd_trilinear(void* ctx, const void* x, int dt, int64_t T, int64_t H, int64_t W, int64_t C, int64_t To, int64_t Ho, int64_t Wo,
                  void* out) {
    auto* c = (Ctx*) ctx;
    CU(cudaSetDevice(c->dev));
    if (To <= 0 || Ho <= 0 || Wo <= 0 || C <= 0) return 0;
    const float st = (float) T / (float) To, sh = (float) H / (float) Ho, sw = (float) W / (float) Wo;
    vid_trilinear_k<<<nb(To * Ho * Wo * C), 256, 0, c->s>>>(x, dt, T, H, W, C, To, Ho, Wo, st, sh, sw, out);
    LAUNCHED("trilinear");
    return 0;
}

int nsd_pad3d(void* ctx, const void* x, int dt, int64_t T, int64_t H, int64_t W, int64_t C, int64_t front, int64_t top, int64_t bottom,
              int64_t left, int64_t right, void* out) {
    auto* c = (Ctx*) ctx;
    CU(cudaSetDevice(c->dev));
    const int64_t To = T + front, Ho = H + top + bottom, Wo = W + left + right;
    if (To <= 0 || Ho <= 0 || Wo <= 0 || C <= 0) return 0;
    vid_pad3d_k<<<nb(To * Ho * Wo * C), 256, 0, c->s>>>(x, dt, T, H, W, C, front, top, left, To, Ho, Wo, out);
    LAUNCHED("pad3d");
    return 0;
}

int nsd_prelu(void* ctx, void* x, int dt, int64_t M, int64_t C, const float* alpha) {
    auto* c = (Ctx*) ctx;
    CU(cudaSetDevice(c->dev));
    if (M <= 0 || C <= 0) return 0;
    vid_prelu_k<<<nb(M * C), 256, 0, c->s>>>(x, dt, M, C, alpha);
    LAUNCHED("prelu");
    return 0;
}

int nsd_pixel_shuffle_add(void* ctx, const void* x, int dt, int64_t N, int64_t H, int64_t W, int64_t C, int64_t r,
                          const void* base, void* out) {
    auto* c = (Ctx*) ctx;
    CU(cudaSetDevice(c->dev));
    if (N <= 0 || H <= 0 || W <= 0) return 0;
    const int64_t n = N * H * r * W * r * C;
    if (n <= 0) return 0;
    vid_pixel_shuffle_add_k<<<nb(n), 256, 0, c->s>>>(x, dt, N, H, W, C, r, base, out);
    LAUNCHED("pixel_shuffle_add");
    return 0;
}

int nsd_resize_area(void* ctx, const void* x, int dt, int64_t N, int64_t Hi, int64_t Wi, int64_t C, int64_t Ho, int64_t Wo,
                    float* out) {
    auto* c = (Ctx*) ctx;
    CU(cudaSetDevice(c->dev));
    if (N <= 0 || Ho <= 0 || Wo <= 0) return 0;
    const int64_t n = C * N * Ho * Wo;
    if (n <= 0) return 0;
    vid_resize_area_k<<<nb(n), 256, 0, c->s>>>(x, dt, N, Hi, Wi, C, Ho, Wo, out);
    LAUNCHED("resize_area");
    return 0;
}

int nsd_aa_snake(void* ctx, const float* x, int64_t B, int64_t C, int64_t L, const float* log_alpha, const float* log_beta,
                 const float* up, const float* down, float* out) {
    auto* c = (Ctx*) ctx;
    CU(cudaSetDevice(c->dev));
    if (B <= 0 || C <= 0 || L <= 0) return 0;
    vid_aa_snake_k<<<nb(B * C * L), 256, 0, c->s>>>(x, B, C, L, log_alpha, log_beta, up, down, out);
    LAUNCHED("aa_snake");
    return 0;
}
