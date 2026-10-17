// uocr.cu: the Unlimited-OCR engine's kernels (uocr.h) - its picture layouts (patches, 3 x 3 columns, windows), SAM's
// attention with decomposed relative positions, Q5_0 weights to half, the decoder's row copies, gathers and weighted
// scatters. nextsycl's uocr.cpp in CUDA, the same names, arguments and arithmetic.
#include "ns.h"
#include "ns_internal.hpp"
#include "uocr.h"

#include <cuda_fp16.h>
#include <cmath>

namespace {

unsigned blocks(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }
#define ID() const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x

__global__ void to_half_k(const float* x, __half* o, int64_t n) { ID(); if (i < n) o[i] = __float2half_rn(x[i]); }

__global__ void quick_gelu_k(float* x, int64_t n) {
    ID();
    if (i < n) { const float v = x[i]; x[i] = v / (1.0f + expf(-1.702f * v)); }
}

__global__ void patchify_k(const float* img, int64_t H, int64_t W, int64_t P, int64_t gh, int64_t gw, int64_t k, float* out, int64_t n) {
    ID();
    if (i >= n) return;
    const int64_t e = i % k, p = i / k;
    const int64_t b = p / (gh * gw), py = (p / gw) % gh, px = p % gw;
    const int64_t c = e / (P * P), y = (e / P) % P, x = e % P;
    out[i] = img[((b * 3 + c) * H + py * P + y) * W + px * P + x];
}

__global__ void im2col3_k(const float* x, int64_t H, int64_t W, int64_t C, int64_t s, int64_t ho, int64_t wo, float* out, int64_t n) {
    ID();
    if (i >= n) return;
    const int64_t k = C * 9, e = i % k, p = i / k;
    const int64_t b = p / (ho * wo), oy = (p / wo) % ho, ox = p % wo;
    const int64_t c = e / 9, ky = (e / 3) % 3, kx = e % 3;
    const int64_t y = oy * s + ky - 1, xx = ox * s + kx - 1;
    out[i] = (y < 0 || y >= H || xx < 0 || xx >= W) ? 0.0f : x[((b * H + y) * W + xx) * C + c];
}

__global__ void window_k(const float* x, int64_t H, int64_t W, int64_t C, int64_t w, int64_t nh, int64_t nw, int back, float* out, int64_t n) {
    ID();
    if (i >= n) return;
    const int64_t c = i % C, r = i / C;
    const int64_t wx = r % w, wy = (r / w) % w, win = r / (w * w);
    const int64_t bx = win % nw, by = (win / nw) % nh, b = win / (nw * nh);
    const int64_t y = by * w + wy, xx = bx * w + wx;
    const bool in = y < H && xx < W;
    if (back == 0) out[i] = in ? x[((b * H + y) * W + xx) * C + c] : 0.0f;
    else if (in) out[((b * H + y) * W + xx) * C + c] = x[i];
}

constexpr int WG = 256, DD = 64;

__device__ float block_max(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    __syncthreads();
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = red[0];
    for (int k = 1; k < WG / 32; ++k) t = fmaxf(t, red[k]);
    return t;
}
__device__ float block_sum(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.0f;
    for (int k = 0; k < WG / 32; ++k) t += red[k];
    return t;
}

// a block a (grid, head, query): the scores in shared memory, their softmax, then the values' sum (64 dims x 4 key
// groups)
__global__ void sam_attn_k(const float* qkv, int64_t N, int64_t Gh, int64_t Gw, int64_t heads, int64_t D, const float* rh, const float* rw, float scale,
                           float* out) {
    extern __shared__ float sc[];
    __shared__ float qv[DD], rel[128], red[WG], wred[WG / 32];
    const int t = threadIdx.x;
    const int64_t gi = blockIdx.x;
    const int64_t qi = gi % N, hd = (gi / N) % heads, b = gi / (N * heads);
    const int64_t qy = qi / Gw, qx = qi % Gw;
    const int64_t C = heads * D, row = 3 * C;
    const float* base = qkv + b * N * row;
    if (t < DD) qv[t] = base[qi * row + hd * D + t];
    __syncthreads();
    if (t < Gh + Gw) {
        const bool hgt = t < Gh;
        const int64_t kk = hgt ? t : t - Gh;
        const float* tab = hgt ? rh + (qy - kk + Gh - 1) * D : rw + (qx - kk + Gw - 1) * D;
        float s = 0.0f;
        for (int d = 0; d < DD; ++d) s += qv[d] * tab[d];
        rel[t] = s;
    }
    __syncthreads();
    float mx = -INFINITY;
    for (int64_t j = t; j < N; j += WG) {
        const float* kr = base + j * row + C + hd * D;
        float s = 0.0f;
        for (int d = 0; d < DD; ++d) s += qv[d] * kr[d];
        s = s * scale + rel[j / Gw] + rel[Gh + j % Gw];
        sc[j] = s;
        mx = fmaxf(mx, s);
    }
    mx = block_max(mx, wred);
    float sum = 0.0f;
    for (int64_t j = t; j < N; j += WG) {
        const float e = expf(sc[j] - mx);
        sc[j] = e;
        sum += e;
    }
    sum = block_sum(sum, wred);
    __syncthreads();
    const int d = t % DD, part = t / DD;
    float acc = 0.0f;
    for (int64_t j = part; j < N; j += WG / DD) acc += sc[j] * base[j * row + 2 * C + hd * D + d];
    red[t] = acc;
    __syncthreads();
    if (t < DD) {
        float v = 0.0f;
        for (int p = 0; p < WG / DD; ++p) v += red[p * DD + t];
        out[(b * N + qi) * C + hd * D + t] = v / sum;
    }
}

// ggml's dequantize_row_q5_0: element j < 16 the low nibble of qs[j] and bit j of qh, j >= 16 the high nibble of
// qs[j - 16] and bit j
__global__ void q5_0_half_k(const uint8_t* s, __half* o, int64_t n) {
    ID();
    if (i >= n) return;
    const int64_t b = i / 32, j = i % 32;
    const uint8_t* blk = s + b * 22;
    const float d = __half2float(__ushort_as_half((unsigned short) (blk[0] | (blk[1] << 8))));
    const uint32_t qh = (uint32_t) blk[2] | ((uint32_t) blk[3] << 8) | ((uint32_t) blk[4] << 16) | ((uint32_t) blk[5] << 24);
    const uint8_t q = blk[6 + (j % 16)];
    const int lo = j < 16 ? (q & 15) : (q >> 4);
    const int hi = (int) ((qh >> j) & 1) << 4;
    o[i] = __float2half_rn((float) ((lo | hi) - 16) * d);
}

__global__ void copy_rows_k(const float* src, int64_t ss, float* dst, int64_t ds, int64_t cols, int64_t n) {
    ID();
    if (i < n) dst[(i / cols) * ds + i % cols] = src[(i / cols) * ss + i % cols];
}

__global__ void gather_k(const float* src, int64_t C, const int32_t* idx, float* out, int64_t n) {
    ID();
    if (i < n) out[i] = src[(int64_t) idx[i / C] * C + i % C];
}

__global__ void scatter_add_k(float* dst, int64_t C, const float* src, const int32_t* idx, const float* w, int64_t n) {
    ID();
    if (i < n) dst[(int64_t) idx[i / C] * C + i % C] += w[i / C] * src[i];
}

}  // namespace

extern "C" {

#define BEGIN(n) if ((n) <= 0) return 0; NS_CU(cudaSetDevice(g->dev))
#define END() NS_LAUNCH_CHECK(); return 0

int ns_image_uocr_to_half(ns_gpu* g, const float* x, void* out, int64_t n) {
    BEGIN(n);
    to_half_k<<<blocks(n), 256, 0, g->q>>>(x, (__half*) out, n);
    END();
}

int ns_image_uocr_quick_gelu(ns_gpu* g, float* x, int64_t n) {
    BEGIN(n);
    quick_gelu_k<<<blocks(n), 256, 0, g->q>>>(x, n);
    END();
}

int ns_image_uocr_patchify(ns_gpu* g, const float* img, int64_t B, int64_t H, int64_t W, int64_t P, float* out) {
    if (H % P || W % P) return ns_fail("ns_image_uocr_patchify: sides multiples of the patch");
    const int64_t gh = H / P, gw = W / P, k = 3 * P * P, n = B * gh * gw * k;
    BEGIN(n);
    patchify_k<<<blocks(n), 256, 0, g->q>>>(img, H, W, P, gh, gw, k, out, n);
    END();
}

int ns_image_uocr_im2col3(ns_gpu* g, const float* x, int64_t B, int64_t H, int64_t W, int64_t C, int64_t s, float* out) {
    const int64_t ho = (H - 1) / s + 1, wo = (W - 1) / s + 1, n = B * ho * wo * C * 9;
    BEGIN(n);
    im2col3_k<<<blocks(n), 256, 0, g->q>>>(x, H, W, C, s, ho, wo, out, n);
    END();
}

int ns_image_uocr_window(ns_gpu* g, const float* x, int64_t B, int64_t H, int64_t W, int64_t C, int64_t w, int back, float* out) {
    const int64_t hp = (H + w - 1) / w * w, wp = (W + w - 1) / w * w, nh = hp / w, nw = wp / w, n = B * hp * wp * C;
    BEGIN(n);
    window_k<<<blocks(n), 256, 0, g->q>>>(x, H, W, C, w, nh, nw, back, out, n);
    END();
}

int ns_image_uocr_sam_attn(ns_gpu* g, const float* qkv, int64_t B, int64_t Gh, int64_t Gw, int64_t heads, int64_t D, const float* rh,
                           const float* rw, float* out) {
    const int64_t N = Gh * Gw;
    if (N > 4096 || D != 64 || Gh > 64 || Gw > 64) return ns_fail("ns_image_uocr_sam_attn: at most 64 x 64 tokens, heads of 64");
    BEGIN(B * heads * N);
    sam_attn_k<<<(unsigned) (B * heads * N), WG, (size_t) N * sizeof(float), g->q>>>(qkv, N, Gh, Gw, heads, D, rh, rw, 1.0f / sqrtf((float) D), out);
    END();
}

int ns_image_uocr_q5_0_half(ns_gpu* g, const void* src, int64_t n, void* out) {
    if (n % 32) return ns_fail("ns_image_uocr_q5_0_half: whole blocks of 32");
    BEGIN(n);
    q5_0_half_k<<<blocks(n), 256, 0, g->q>>>((const uint8_t*) src, (__half*) out, n);
    END();
}

int ns_image_uocr_copy_rows(ns_gpu* g, const float* src, int64_t ss, float* dst, int64_t ds, int64_t rows, int64_t cols) {
    BEGIN(rows * cols);
    copy_rows_k<<<blocks(rows * cols), 256, 0, g->q>>>(src, ss, dst, ds, cols, rows * cols);
    END();
}

int ns_image_uocr_gather(ns_gpu* g, const float* src, int64_t C, const int32_t* idx, int64_t n, float* out) {
    BEGIN(n * C);
    gather_k<<<blocks(n * C), 256, 0, g->q>>>(src, C, idx, out, n * C);
    END();
}

int ns_image_uocr_scatter_add(ns_gpu* g, float* dst, int64_t C, const float* src, const int32_t* idx, const float* w, int64_t n) {
    BEGIN(n * C);
    scatter_add_k<<<blocks(n * C), 256, 0, g->q>>>(dst, C, src, idx, w, n * C);
    END();
}

}  // extern "C"
