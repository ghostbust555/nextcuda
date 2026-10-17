// vcp.cu: the VoxCPM2 engine's own kernels (vcp.h). Everything here is small - a local sequence is 5 or 11 rows, the
// VAE's channels a few hundred thousand samples - so these are plain element-per-thread kernels, as nextsycl's
// vcp.cpp (the same names, arguments and arithmetic).
#include "ns.h"
#include "ns_internal.hpp"
#include "vcp.h"

#include <cmath>

namespace {

unsigned blocks(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }

#define ID() const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x

__global__ void rope_k(float* x, int64_t xs, int64_t S, int64_t H, int64_t D, const float* inv_freq, int64_t p0, int64_t n) {
    ID();
    if (id >= n) return;
    const int64_t half = D / 2;
    const int64_t r = id / (H * half), h = (id / half) % H, i = id % half;
    float* p = x + r * xs + h * D;
    const float a = (float) (p0 + r % S) * inv_freq[i], c = cosf(a), s = sinf(a);
    const float x1 = p[i], x2 = p[i + half];
    p[i] = x1 * c - x2 * s;
    p[i + half] = x2 * c + x1 * s;
}

__global__ void local_attn_k(const float* q, const float* k, const float* v, int64_t S, int64_t Hq, int64_t G, int64_t D, int64_t qs, int64_t kvs,
                             float scale, float* out, int64_t n) {
    ID();
    if (id >= n) return;
    const int64_t r = id / Hq, h = id % Hq, b = r / S, kh = h / G;
    const float* qi = q + r * qs + h * D;
    float sc[64];
    float mx = -INFINITY;
    for (int64_t j = 0; j < S && j < 64; ++j) {
        const float* kj = k + (b * S + j) * kvs + kh * D;
        float s = 0.0f;
        for (int64_t d = 0; d < D; ++d) s += qi[d] * kj[d];
        sc[j] = s * scale;
        mx = fmaxf(mx, sc[j]);
    }
    float acc[128];
    for (int64_t d = 0; d < D; ++d) acc[d] = 0.0f;
    float sum = 0.0f;
    for (int64_t j = 0; j < S && j < 64; ++j) {
        const float p = expf(sc[j] - mx);
        sum += p;
        const float* vj = v + (b * S + j) * kvs + kh * D;
        for (int64_t d = 0; d < D; ++d) acc[d] += p * vj[d];
    }
    float* o = out + r * (Hq * D) + h * D;
    for (int64_t d = 0; d < D; ++d) o[d] = acc[d] / sum;
}

__global__ void snake_k(const float* x, int64_t L, const float* alpha, float* out, int64_t n) {
    ID();
    if (id >= n) return;
    const float a = alpha[id / L], v = x[id], s = sinf(a * v);
    out[id] = v + s * s / (a + 1e-9f);
}

__global__ void dwconv_k(const float* x, int64_t L, const float* w, const float* bias, int64_t K, int64_t dil, float* out, int64_t n) {
    ID();
    if (id >= n) return;
    const int64_t c = id / L, t = id % L;
    float a = bias ? bias[c] : 0.0f;
    for (int64_t k = 0; k < K; ++k) {
        const int64_t s = t - (K - 1 - k) * dil;
        if (s >= 0) a += w[c * K + k] * x[c * L + s];
    }
    out[id] = a;
}

__global__ void affine_k(float* x, int64_t L, const float* scale, const float* bias, int64_t n) {
    ID();
    if (id >= n) return;
    const int64_t c = id / L;
    x[id] = x[id] * scale[c] + bias[c];
}

__global__ void fsq_k(float* x, int64_t n, float s) {
    ID();
    if (id < n) x[id] = rintf(tanhf(x[id]) * s) / s;
}

}  // namespace

extern "C" {

int ns_audio_vcp_rope(ns_gpu* g, float* x, int64_t xs, int64_t rows, int64_t S, int64_t H, int64_t D, const float* inv_freq, int64_t p0) {
    if (rows <= 0 || S <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    const int64_t n = rows * H * (D / 2);
    rope_k<<<blocks(n), 256, 0, g->q>>>(x, xs, S, H, D, inv_freq, p0, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_vcp_local_attn(ns_gpu* g, const float* q, const float* k, const float* v, int64_t B, int64_t S, int64_t Hq, int64_t Hkv, int64_t D,
                            int64_t qs, int64_t kvs, float* out) {
    if (B <= 0 || S <= 0) return 0;
    if (D > 128 || Hq % Hkv != 0) return ns_fail("ns_audio_vcp_local_attn: D <= 128, query heads a multiple of the key heads");
    NS_CU(cudaSetDevice(g->dev));
    const int64_t n = B * S * Hq;
    local_attn_k<<<blocks(n, 64), 64, 0, g->q>>>(q, k, v, S, Hq, Hq / Hkv, D, qs, kvs, 1.0f / sqrtf((float) D), out, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_vcp_snake(ns_gpu* g, const float* x, int64_t C, int64_t L, const float* alpha, float* out) {
    if (C <= 0 || L <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    snake_k<<<blocks(C * L), 256, 0, g->q>>>(x, L, alpha, out, C * L);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_vcp_dwconv(ns_gpu* g, const float* x, int64_t C, int64_t L, const float* w, const float* bias, int64_t K, int64_t dil, float* out) {
    if (C <= 0 || L <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    dwconv_k<<<blocks(C * L), 256, 0, g->q>>>(x, L, w, bias, K, dil, out, C * L);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_vcp_affine(ns_gpu* g, float* x, int64_t C, int64_t L, const float* scale, const float* bias) {
    if (C <= 0 || L <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    affine_k<<<blocks(C * L), 256, 0, g->q>>>(x, L, scale, bias, C * L);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_vcp_fsq(ns_gpu* g, float* x, int64_t n, float s) {
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    fsq_k<<<blocks(n), 256, 0, g->q>>>(x, n, s);
    NS_LAUNCH_CHECK();
    return 0;
}

}  // extern "C"
