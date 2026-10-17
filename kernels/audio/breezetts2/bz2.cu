// bz2.cu: the Breeze TTS 2 engine's own kernels (bz2.h). The text encoder's sequences are a prompt's text segments
// (tens to a few hundred tokens) and the depth decoder's rows two at most, so these are plain thread-per-row kernels,
// as nextsycl's bz2.cpp (the same names, arguments and arithmetic); the cached decode is the MiniMax Music engine's
// mm3 kernels, the linear layers cuBLAS's (nsd).
#include "ns.h"
#include "ns_internal.hpp"
#include "bz2.h"

#include <cmath>

namespace {

unsigned blocks(int64_t n, int t) { return (unsigned) ((n + t - 1) / t); }

// one thread a (row, head): the norm over its D values, then each pair rotated
__global__ void head_norm_rope_k(float* x, int64_t xs, int64_t S, int64_t H, int64_t D, const float* weight, float eps, const float* inv_freq,
                                 int64_t p0, int64_t n) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const int64_t r = id / H, h = id % H, half = D / 2;
    float* v = x + r * xs + h * D;
    if (weight) {
        float ss = 0.0f;
        for (int64_t d = 0; d < D; ++d) ss += v[d] * v[d];
        const float inv = rsqrtf(ss / (float) D + eps);
        for (int64_t d = 0; d < D; ++d) v[d] = v[d] * inv * weight[d];
    }
    const float pos = (float) (p0 + r % S);
    for (int64_t j = 0; j < half; ++j) {
        const float a = pos * inv_freq[j], c = cosf(a), s = sinf(a);
        const float x1 = v[j], x2 = v[j + half];
        v[j] = x1 * c - x2 * s;
        v[j + half] = x2 * c + x1 * s;
    }
}

// one thread a (query, head): the window's scores (max first, then the weighted values)
__global__ void attn_window_k(const float* q, const float* k, const float* v, int64_t L, int64_t Hq, int64_t G, int64_t D, int64_t qs, int64_t kvs,
                              int64_t left, int64_t right, float scale, float* out, int64_t n) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const int64_t i = id / Hq, h = id % Hq, kh = h / G;
    const float* qi = q + i * qs + h * D;
    const int64_t j0 = i - left + 1 > 0 ? i - left + 1 : 0;
    const int64_t j1 = i + right - 1 < L - 1 ? i + right - 1 : L - 1;
    float mx = -INFINITY;
    for (int64_t j = j0; j <= j1; ++j) {
        const float* kj = k + j * kvs + kh * D;
        float s = 0.0f;
        for (int64_t d = 0; d < D; ++d) s += qi[d] * kj[d];
        mx = fmaxf(mx, s * scale);
    }
    float acc[256];
    for (int64_t d = 0; d < D; ++d) acc[d] = 0.0f;
    float sum = 0.0f;
    for (int64_t j = j0; j <= j1; ++j) {
        const float* kj = k + j * kvs + kh * D;
        float s = 0.0f;
        for (int64_t d = 0; d < D; ++d) s += qi[d] * kj[d];
        const float p = expf(s * scale - mx);
        sum += p;
        const float* vj = v + j * kvs + kh * D;
        for (int64_t d = 0; d < D; ++d) acc[d] += p * vj[d];
    }
    float* o = out + i * (Hq * D) + h * D;
    for (int64_t d = 0; d < D; ++d) o[d] = acc[d] / sum;
}

__global__ void gelu_mul_k(const float* gu, int64_t F, float* out, int64_t n) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const int64_t r = id / F, f = id % F;
    const float a = gu[r * 2 * F + f], u = gu[r * 2 * F + F + f];
    // gelu_pytorch_tanh: 0.5 a (1 + tanh(sqrt(2 / pi) (a + 0.044715 a^3)))
    const float t = tanhf(0.7978845608028654f * (a + 0.044715f * a * a * a));
    out[id] = 0.5f * a * (1.0f + t) * u;
}

}  // namespace

extern "C" {

int ns_audio_bz2_head_norm_rope(ns_gpu* g, float* x, int64_t xs, int64_t rows, int64_t S, int64_t H, int64_t D, const float* weight, float eps,
                                const float* inv_freq, int64_t p0) {
    if (D > 256 || D % 2 != 0 || S <= 0) return ns_fail("ns_audio_bz2_head_norm_rope: D even, at most 256");
    if (rows <= 0 || H <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    head_norm_rope_k<<<blocks(rows * H, 128), 128, 0, g->q>>>(x, xs, S, H, D, weight, eps, inv_freq, p0, rows * H);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_bz2_attn_window(ns_gpu* g, const float* q, const float* k, const float* v, int64_t L, int64_t Hq, int64_t Hkv, int64_t D,
                             int64_t qs, int64_t kvs, int64_t left, int64_t right, float scale, float* out) {
    if (D > 256 || Hkv <= 0 || Hq % Hkv != 0 || left < 1 || right < 1)
        return ns_fail("ns_audio_bz2_attn_window: D at most 256, Hq a multiple of Hkv, windows of 1 or more");
    if (L <= 0 || Hq <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    // a thread's 256 accumulators: small blocks
    attn_window_k<<<blocks(L * Hq, 32), 32, 0, g->q>>>(q, k, v, L, Hq, Hq / Hkv, D, qs, kvs, left, right, scale, out, L * Hq);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_bz2_gelu_mul(ns_gpu* g, const float* gu, int64_t R, int64_t F, float* out) {
    if (R <= 0 || F <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    gelu_mul_k<<<blocks(R * F, 256), 256, 0, g->q>>>(gu, F, out, R * F);
    NS_LAUNCH_CHECK();
    return 0;
}

}  // extern "C"
