// h3d.cu: the Hunyuan3D engine's own small kernels - what the shared diffusion kernels (nsd) do not have: the layer
// norm with its affine and modulation in one pass, the per-head q / k norms in strided rows, the residual adds, the
// sampler's steps, the shape decoder's query embedding and the MoE's routing, and the paint model's group norm, GEGLU,
// padding, concatenation, upsampling and half adds. nextsycl's h3d.cpp in CUDA, the same names, arguments and
// arithmetic: its work-group reductions are block reductions (warp shuffles, then shared memory), on the GPU's stream.
#include "ns.h"
#include "ns_internal.hpp"
#include "h3d.h"

#include <cuda_fp16.h>
#include <cmath>

namespace {

unsigned blocks(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }
#define ID() const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x

__device__ inline float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// the sum over the block (blockDim.x a multiple of 32, at most 1024), every thread gets it; red holds 32 floats
__device__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < (int) (blockDim.x >> 5); ++i) t += red[i];
    __syncthreads();
    return t;
}

constexpr int LN_WG = 256, LN_PER = 32;  // a row of up to 8,192 values held in registers

__global__ void ln_k(const void* x, int x_half, int64_t C, float eps, const float* w, const float* b, const float* scale,
                     const float* shift, void* out, int out_half) {
    __shared__ float red[32];
    const int64_t r = blockIdx.x;
    const int l = threadIdx.x;
    float v[LN_PER];
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < LN_PER; ++j) {
        const int64_t i = (int64_t) j * LN_WG + l;
        float a = 0.0f;
        if (i < C) a = x_half ? __half2float(((const __half*) x)[r * C + i]) : ((const float*) x)[r * C + i];
        v[j] = a;
        s += a;
    }
    const float mean = block_sum(s, red) / (float) C;
    float s2 = 0.0f;
#pragma unroll
    for (int j = 0; j < LN_PER; ++j) {
        const int64_t i = (int64_t) j * LN_WG + l;
        const float d = i < C ? v[j] - mean : 0.0f;
        s2 += d * d;
    }
    const float inv = rsqrtf(block_sum(s2, red) / (float) C + eps);
#pragma unroll
    for (int j = 0; j < LN_PER; ++j) {
        const int64_t i = (int64_t) j * LN_WG + l;
        if (i < C) {
            float y = (v[j] - mean) * inv;
            if (w) y *= w[i];
            if (b) y += b[i];
            if (scale) y *= 1.0f + scale[i];
            if (shift) y += shift[i];
            if (out_half) ((__half*) out)[r * C + i] = __float2half_rn(y);
            else ((float*) out)[r * C + i] = y;
        }
    }
}

constexpr int HN_WG = 32, HN_PER = 8;  // a warp a head, up to 256 features

__global__ void head_norm_k(__half* p, int64_t H, int64_t D, int64_t stride, const float* w, const float* b, float eps, int rms, int per) {
    const int64_t gi = blockIdx.x;
    const int64_t m = gi / H, h = gi % H;
    const int l = threadIdx.x;
    __half* row = p + m * stride + h * D;
    float v[HN_PER];
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < HN_PER; ++j) {
        v[j] = j < per ? __half2float(row[j * HN_WG + l]) : 0.0f;
        s += rms ? v[j] * v[j] : v[j];
    }
    const float tot = warp_sum(s);
    float mean = 0.0f, inv;
    if (rms) {
        inv = rsqrtf(tot / (float) D + eps);
    } else {
        mean = tot / (float) D;
        float s2 = 0.0f;
#pragma unroll
        for (int j = 0; j < HN_PER; ++j) {
            const float d = j < per ? v[j] - mean : 0.0f;
            s2 += d * d;
        }
        inv = rsqrtf(warp_sum(s2) / (float) D + eps);
    }
#pragma unroll
    for (int j = 0; j < HN_PER; ++j) {
        if (j < per) {
            const int i = j * HN_WG + l;
            float y = (v[j] - mean) * inv * w[i];
            if (b) y += b[i];
            row[i] = __float2half_rn(y);
        }
    }
}

__global__ void silu_k(const float* x, __half* o, int64_t n) {
    ID();
    if (i < n) {
        const float a = x[i];
        o[i] = __float2half_rn(a / (1.0f + expf(-a)));
    }
}

__global__ void to_half_k(const float* x, __half* o, int64_t n) { ID(); if (i < n) o[i] = __float2half_rn(x[i]); }

__global__ void to_float_k(const __half* a, float* out, int64_t n) { ID(); if (i < n) out[i] = __half2float(a[i]); }

__global__ void add_k(float* x, const void* y, int y_half, int64_t C, const float* gate, int64_t n) {
    ID();
    if (i >= n) return;
    const float v = y_half ? __half2float(((const __half*) y)[i]) : ((const float*) y)[i];
    x[i] += gate ? v * gate[i % C] : v;
}

__global__ void axpy_k(float* x, const float* y, int64_t n, float a) { ID(); if (i < n) x[i] += a * y[i]; }

__global__ void cfg_k(float* out, const float* c, const float* u, int64_t n, float s) {
    ID();
    if (i < n) out[i] = u[i] + s * (c[i] - u[i]);
}

__global__ void fourier_k(const float* pts, const int32_t* idx, int64_t start, int64_t N, int64_t R1, float lo, float step, int nf, float fpi,
                          int64_t width, __half* o) {
    ID();
    if (i >= N) return;
    float p[3];
    if (pts) {
        p[0] = pts[i * 3];
        p[1] = pts[i * 3 + 1];
        p[2] = pts[i * 3 + 2];
    } else {
        const int64_t q = idx ? (int64_t) idx[i] : start + i;
        const int64_t ix = q / (R1 * R1), iy = (q / R1) % R1, iz = q % R1;
        p[0] = lo + (float) ix * step;
        p[1] = lo + (float) iy * step;
        p[2] = lo + (float) iz * step;
    }
    __half* row = o + i * width;
    for (int d = 0; d < 3; ++d) row[d] = __float2half_rn(p[d]);
    for (int d = 0; d < 3; ++d) {
        float f = fpi;
        for (int j = 0; j < nf; ++j) {
            const float a = p[d] * f;
            row[3 + d * nf + j] = __float2half_rn(sinf(a));
            row[3 + 3 * nf + d * nf + j] = __float2half_rn(cosf(a));
            f *= 2.0f;
        }
    }
}

__global__ void route_k(const float* logits, int64_t M, int64_t E, int64_t k, int32_t* idx, float* w) {
    const int64_t r = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= M) return;
    const float* l = logits + r * E;
    float mx = -INFINITY;
    for (int64_t e = 0; e < E; ++e) mx = fmaxf(mx, l[e]);
    float sum = 0.0f;
    for (int64_t e = 0; e < E; ++e) sum += expf(l[e] - mx);
    uint64_t taken = 0;
    for (int64_t j = 0; j < k; ++j) {
        int64_t best = -1;
        float bv = -INFINITY;
        for (int64_t e = 0; e < E; ++e) {
            if ((taken >> e) & 1) continue;
            if (best < 0 || l[e] > bv) {
                best = e;
                bv = l[e];
            }
        }
        taken |= (uint64_t) 1 << best;
        idx[r * k + j] = (int32_t) best;
        w[r * k + j] = expf(bv - mx) / sum;
    }
}

__global__ void gather_k(const __half* a, const int32_t* idx, int64_t C, __half* o, int64_t n) {
    ID();
    if (i >= n) return;
    const int64_t r = i / C, c = i % C;
    o[i] = a[(int64_t) idx[r] * C + c];
}

__global__ void scatter_add_k(const __half* a, const int32_t* idx, const float* w, int64_t C, float* out, int64_t n) {
    ID();
    if (i >= n) return;
    const int64_t r = i / C, c = i % C;
    out[(int64_t) idx[r] * C + c] += w[r] * __half2float(a[i]);
}

constexpr int GN_WG = 256;

// the statistics: one block a frame and group, the mean first, then the variance about it
__global__ void gn_stats_k(const __half* a, int64_t N, int64_t C, int64_t G, int64_t cg, int64_t per, float eps, float* stats) {
    __shared__ float red[32];
    const int64_t fg = blockIdx.x;
    const int64_t f = fg / G, gr = fg % G;
    const int64_t l = threadIdx.x;
    const __half* base = a + f * N * C + gr * cg;
    float s = 0.0f;
    for (int64_t i = l; i < per; i += GN_WG) s += __half2float(base[(i / cg) * C + i % cg]);
    const float mean = block_sum(s, red) / (float) per;
    float s2 = 0.0f;
    for (int64_t i = l; i < per; i += GN_WG) {
        const float d = __half2float(base[(i / cg) * C + i % cg]) - mean;
        s2 += d * d;
    }
    const float var = block_sum(s2, red) / (float) per;
    if (l == 0) {
        stats[fg * 2] = mean;
        stats[fg * 2 + 1] = rsqrtf(var + eps);
    }
}

__global__ void gn_apply_k(const __half* a, int64_t N, int64_t C, int64_t G, int64_t cg, const float* w, const float* b, int silu,
                           const float* stats, __half* o, int64_t n) {
    ID();
    if (i >= n) return;
    const int64_t c = i % C, f = i / (N * C);
    const int64_t fg = f * G + c / cg;
    float y = (__half2float(a[i]) - stats[fg * 2]) * stats[fg * 2 + 1] * w[c] + b[c];
    if (silu) y = y / (1.0f + expf(-y));
    o[i] = __float2half_rn(y);
}

__global__ void geglu_k(const __half* a, int64_t F, __half* o, int64_t n) {
    ID();
    if (i >= n) return;
    const int64_t r = i / F, c = i % F;
    const float h = __half2float(a[r * 2 * F + c]), gt = __half2float(a[r * 2 * F + F + c]);
    o[i] = __float2half_rn(h * 0.5f * gt * (1.0f + erff(gt * 0.70710678118654752f)));
}

__global__ void pad2d_k(const __half* a, int64_t H, int64_t W, int64_t C, int64_t t, int64_t l, int64_t Ho, int64_t Wo, __half* o, int64_t n) {
    ID();
    if (i >= n) return;
    const int64_t c = i % C, px = (i / C) % Wo, py = (i / C / Wo) % Ho, nn = i / C / Wo / Ho;
    const int64_t sy = py - t, sx = px - l;
    o[i] = (sy >= 0 && sy < H && sx >= 0 && sx < W) ? a[((nn * H + sy) * W + sx) * C + c] : __float2half_rn(0.0f);
}

__global__ void concat_k(const __half* pa, int64_t Ca, const __half* pb, int64_t Cb, __half* o, int64_t n) {
    ID();
    if (i >= n) return;
    const int64_t C = Ca + Cb;
    const int64_t r = i / C, c = i % C;
    o[i] = c < Ca ? pa[r * Ca + c] : pb[r * Cb + c - Ca];
}

__global__ void up2_k(const __half* a, int64_t H, int64_t W, int64_t C, __half* o, int64_t n) {
    ID();
    if (i >= n) return;
    const int64_t Wo = 2 * W, Ho = 2 * H;
    const int64_t c = i % C, px = (i / C) % Wo, py = (i / C / Wo) % Ho, nn = i / C / Wo / Ho;
    o[i] = a[((nn * H + py / 2) * W + px / 2) * C + c];
}

__global__ void add_half_k(__half* a, const __half* b, int64_t n, float s) {
    ID();
    if (i < n) a[i] = __float2half_rn(__half2float(a[i]) + s * __half2float(b[i]));
}

}  // namespace

extern "C" {

int ns_model3d_hunyuan3d_ln(ns_gpu* g, const void* x, int x_half, int64_t M, int64_t C, float eps, const float* w, const float* b,
                            const float* scale, const float* shift, void* out, int out_half) {
    if (C > (int64_t) LN_WG * LN_PER || C <= 0) return ns_fail("ns_model3d_hunyuan3d_ln: rows of 1 to 8192 values");
    if (M <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    ln_k<<<(unsigned) M, LN_WG, 0, g->q>>>(x, x_half, C, eps, w, b, scale, shift, out, out_half);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_hunyuan3d_head_norm(ns_gpu* g, void* x, int64_t M, int64_t H, int64_t D, int64_t stride, const float* w, const float* b,
                                   float eps, int rms) {
    if (D % HN_WG != 0 || D > HN_WG * HN_PER) return ns_fail("ns_model3d_hunyuan3d_head_norm: D a multiple of 32, at most 256");
    if (stride < H * D) return ns_fail("ns_model3d_hunyuan3d_head_norm: rows narrower than their heads");
    if (M <= 0 || H <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    head_norm_k<<<(unsigned) (M * H), HN_WG, 0, g->q>>>((__half*) x, H, D, stride, w, b, eps, rms, (int) (D / HN_WG));
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_hunyuan3d_silu(ns_gpu* g, const float* x, void* out, int64_t n) {
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    silu_k<<<blocks(n), 256, 0, g->q>>>(x, (__half*) out, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_hunyuan3d_to_half(ns_gpu* g, const float* x, void* out, int64_t n) {
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    to_half_k<<<blocks(n), 256, 0, g->q>>>(x, (__half*) out, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_hunyuan3d_to_float(ns_gpu* g, const void* x, float* out, int64_t n) {
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    to_float_k<<<blocks(n), 256, 0, g->q>>>((const __half*) x, out, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_hunyuan3d_add(ns_gpu* g, float* x, const void* y, int y_half, int64_t M, int64_t C, const float* gate) {
    if (M <= 0 || C <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    add_k<<<blocks(M * C), 256, 0, g->q>>>(x, y, y_half, C, gate, M * C);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_hunyuan3d_axpy(ns_gpu* g, float* x, const float* y, int64_t n, float a) {
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    axpy_k<<<blocks(n), 256, 0, g->q>>>(x, y, n, a);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_hunyuan3d_cfg(ns_gpu* g, float* out, const float* c, const float* u, int64_t n, float s) {
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    cfg_k<<<blocks(n), 256, 0, g->q>>>(out, c, u, n, s);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_hunyuan3d_fourier(ns_gpu* g, const float* pts, const int32_t* idx, int64_t start, int64_t N, int64_t R1, float lo,
                                 float step, int nf, int pi, void* out) {
    if (nf < 0 || nf > 16) return ns_fail("ns_model3d_hunyuan3d_fourier: 0 to 16 frequencies");
    if (!pts && R1 <= 0) return ns_fail("ns_model3d_hunyuan3d_fourier: no points and no lattice");
    if (N <= 0) return 0;
    const int64_t width = 3 * (2 * (int64_t) nf + 1);
    const float fpi = pi ? 3.14159265358979323846f : 1.0f;
    NS_CU(cudaSetDevice(g->dev));
    fourier_k<<<blocks(N), 256, 0, g->q>>>(pts, idx, start, N, R1, lo, step, nf, fpi, width, (__half*) out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_hunyuan3d_route(ns_gpu* g, const float* logits, int64_t M, int64_t E, int64_t k, int32_t* idx, float* w) {
    if (E <= 0 || E > 64 || k <= 0 || k > 8 || k > E) return ns_fail("ns_model3d_hunyuan3d_route: 1 to 64 experts, 1 to 8 chosen");
    if (M <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    route_k<<<blocks(M), 256, 0, g->q>>>(logits, M, E, k, idx, w);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_hunyuan3d_gather(ns_gpu* g, const void* x, const int32_t* idx, int64_t n, int64_t C, void* out) {
    if (n <= 0 || C <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    gather_k<<<blocks(n * C), 256, 0, g->q>>>((const __half*) x, idx, C, (__half*) out, n * C);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_hunyuan3d_scatter_add(ns_gpu* g, const void* y, const int32_t* idx, const float* w, int64_t n, int64_t C, float* out) {
    if (n <= 0 || C <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    scatter_add_k<<<blocks(n * C), 256, 0, g->q>>>((const __half*) y, idx, w, C, out, n * C);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_hunyuan3d_group_norm(ns_gpu* g, const void* x, int64_t F, int64_t N, int64_t C, int64_t G, const float* w, const float* b,
                                    float eps, int silu, float* stats, void* out) {
    if (G <= 0 || C % G != 0) return ns_fail("ns_model3d_hunyuan3d_group_norm: channels not a multiple of the groups");
    if (F <= 0 || N <= 0) return 0;
    const __half* a = (const __half*) x;
    __half* o = (__half*) out;
    const int64_t cg = C / G, per = N * cg;
    NS_CU(cudaSetDevice(g->dev));
    gn_stats_k<<<(unsigned) (F * G), GN_WG, 0, g->q>>>(a, N, C, G, cg, per, eps, stats);
    NS_LAUNCH_CHECK();
    const int64_t n = F * N * C;
    if (n <= 0) return 0;
    gn_apply_k<<<blocks(n), 256, 0, g->q>>>(a, N, C, G, cg, w, b, silu, stats, o, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_hunyuan3d_geglu(ns_gpu* g, const void* x, int64_t M, int64_t F, void* out) {
    if (M <= 0 || F <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    geglu_k<<<blocks(M * F), 256, 0, g->q>>>((const __half*) x, F, (__half*) out, M * F);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_hunyuan3d_pad2d(ns_gpu* g, const void* x, int64_t N, int64_t H, int64_t W, int64_t C, int64_t t, int64_t b, int64_t l,
                               int64_t r, void* out) {
    if (t < 0 || b < 0 || l < 0 || r < 0) return ns_fail("ns_model3d_hunyuan3d_pad2d: negative padding");
    const int64_t Ho = H + t + b, Wo = W + l + r;
    if (N <= 0 || Ho <= 0 || Wo <= 0 || C <= 0) return 0;
    const int64_t n = N * Ho * Wo * C;
    NS_CU(cudaSetDevice(g->dev));
    pad2d_k<<<blocks(n), 256, 0, g->q>>>((const __half*) x, H, W, C, t, l, Ho, Wo, (__half*) out, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_hunyuan3d_concat(ns_gpu* g, const void* a, int64_t Ca, const void* b, int64_t Cb, int64_t M, void* out) {
    if (M <= 0) return 0;
    const int64_t n = M * (Ca + Cb);
    if (n <= 0) return 0;  // SYCL's empty range: nothing launched
    NS_CU(cudaSetDevice(g->dev));
    concat_k<<<blocks(n), 256, 0, g->q>>>((const __half*) a, Ca, (const __half*) b, Cb, (__half*) out, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_hunyuan3d_up2(ns_gpu* g, const void* x, int64_t N, int64_t H, int64_t W, int64_t C, void* out) {
    if (N <= 0 || H <= 0 || W <= 0 || C <= 0) return 0;
    const int64_t n = N * 2 * H * 2 * W * C;
    NS_CU(cudaSetDevice(g->dev));
    up2_k<<<blocks(n), 256, 0, g->q>>>((const __half*) x, H, W, C, (__half*) out, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_hunyuan3d_add_half(ns_gpu* g, void* x, const void* y, int64_t n, float s) {
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    add_half_k<<<blocks(n), 256, 0, g->q>>>((__half*) x, (const __half*) y, n, s);
    NS_LAUNCH_CHECK();
    return 0;
}

}  // extern "C"
