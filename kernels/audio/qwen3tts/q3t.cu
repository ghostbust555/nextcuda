// q3t.cu: the Qwen3-TTS engine's own kernels (q3t.h): the 12 Hz codec decoder's - its SnakeBeta, depthwise causal
// convolution, sliding-window attention, rotary positions in float32 and layer-scaled residuals - and small element-wise
// helpers. The codec decoder's sequences are short (a chunk: 325 frames at 12.5 Hz, then the upsampled waveform's
// channels), so these are plain element-per-thread kernels, as nextsycl's q3t.cpp (the same names, arguments and
// arithmetic); the language models' decode runs on the MiniMax Music engine's mm3 kernels. Breeze TTS 2's codec is the
// same one.
#include "ns.h"
#include "ns_internal.hpp"
#include "q3t.h"

#include <cmath>

namespace {

unsigned blocks(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }

#define ID() const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x

__global__ void snake_beta_k(const float* x, int64_t L, const float* la, const float* lb, float* out, int64_t n) {
    ID();
    if (id >= n) return;
    const int64_t c = id / L;
    const float v = x[id], s = sinf(v * expf(la[c]));
    out[id] = v + s * s / (expf(lb[c]) + 1e-9f);
}

__global__ void dwconv_k(const float* x, int64_t L, const float* w, const float* bias, int64_t K, float* out, int64_t n) {
    ID();
    if (id >= n) return;
    const int64_t c = id / L, t = id % L;
    float a = bias ? bias[c] : 0.0f;
    for (int64_t k = 0; k < K; ++k) {
        const int64_t s = t - K + 1 + k;
        if (s >= 0) a += w[c * K + k] * x[c * L + s];
    }
    out[id] = a;
}

// one thread a (query, head): the window's scores, then the values (D <= 128)
__global__ void window_attn_k(const float* q, const float* k, const float* v, int64_t H, int64_t D, int64_t qs, int64_t kvs, int64_t W, float scale,
                              float* out, int64_t n) {
    ID();
    if (id >= n) return;
    const int64_t i = id / H, h = id % H;
    const float* qi = q + i * qs + h * D;
    const int64_t j0 = i - W + 1 > 0 ? i - W + 1 : 0;
    float mx = -INFINITY;
    for (int64_t j = j0; j <= i; ++j) {
        const float* kj = k + j * kvs + h * D;
        float s = 0.0f;
        for (int64_t d = 0; d < D; ++d) s += qi[d] * kj[d];
        mx = fmaxf(mx, s * scale);
    }
    float acc[128];
    for (int64_t d = 0; d < D; ++d) acc[d] = 0.0f;
    float sum = 0.0f;
    for (int64_t j = j0; j <= i; ++j) {
        const float* kj = k + j * kvs + h * D;
        float s = 0.0f;
        for (int64_t d = 0; d < D; ++d) s += qi[d] * kj[d];
        const float p = expf(s * scale - mx);
        sum += p;
        const float* vj = v + j * kvs + h * D;
        for (int64_t d = 0; d < D; ++d) acc[d] += p * vj[d];
    }
    float* o = out + i * (H * D) + h * D;
    for (int64_t d = 0; d < D; ++d) o[d] = acc[d] / sum;
}

__global__ void rope_k(float* x, int64_t xs, int64_t H, int64_t D, float theta, int64_t p0, int64_t n) {
    ID();
    if (id >= n) return;
    const int64_t half = D / 2;
    const int64_t i = id / (H * half), h = (id / half) % H, j = id % half;
    const float inv = powf(theta, -2.0f * (float) j / (float) D);
    const float a = (float) (p0 + i) * inv, c = cosf(a), s = sinf(a);
    float* r = x + i * xs + h * D;
    const float x1 = r[j], x2 = r[j + half];
    r[j] = x1 * c - x2 * s;
    r[j + half] = x2 * c + x1 * s;
}

__global__ void scale_add_k(float* x, const float* y, const float* scale, int64_t C, int64_t n) {
    ID();
    if (id < n) x[id] += scale[id % C] * y[id];
}

__global__ void silu_k(float* x, int64_t n) {
    ID();
    if (id < n) {
        const float v = x[id];
        x[id] = v / (1.0f + expf(-v));
    }
}

__global__ void elu_k(const float* x, int64_t n, float* out) {
    ID();
    if (id < n) {
        const float v = x[id];
        out[id] = v > 0.0f ? v : expm1f(v);
    }
}

__global__ void clamp_k(float* x, int64_t n, float lo, float hi) {
    ID();
    if (id < n) x[id] = fminf(fmaxf(x[id], lo), hi);
}

}  // namespace

extern "C" {

int ns_audio_q3t_snake_beta(ns_gpu* g, const float* x, int64_t C, int64_t L, const float* log_alpha, const float* log_beta, float* out) {
    if (C <= 0 || L <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    snake_beta_k<<<blocks(C * L), 256, 0, g->q>>>(x, L, log_alpha, log_beta, out, C * L);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_q3t_dwconv(ns_gpu* g, const float* x, int64_t C, int64_t L, const float* w, const float* bias, int64_t K, float* out) {
    if (C <= 0 || L <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    dwconv_k<<<blocks(C * L), 256, 0, g->q>>>(x, L, w, bias, K, out, C * L);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_q3t_window_attn(ns_gpu* g, const float* q, const float* k, const float* v, int64_t L, int64_t H, int64_t D, int64_t qs, int64_t kvs,
                             int64_t W, float* out) {
    if (L <= 0 || H <= 0) return 0;
    if (D > 128) return ns_fail("ns_audio_q3t_window_attn: D at most 128");
    NS_CU(cudaSetDevice(g->dev));
    // a thread's 128 accumulators: smaller blocks keep them in registers
    window_attn_k<<<blocks(L * H, 64), 64, 0, g->q>>>(q, k, v, H, D, qs, kvs, W, 1.0f / sqrtf((float) D), out, L * H);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_q3t_rope(ns_gpu* g, float* x, int64_t xs, int64_t L, int64_t H, int64_t D, float theta, int64_t p0) {
    if (L <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    const int64_t n = L * H * (D / 2);
    rope_k<<<blocks(n), 256, 0, g->q>>>(x, xs, H, D, theta, p0, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_q3t_scale_add(ns_gpu* g, float* x, const float* y, const float* scale, int64_t R, int64_t C) {
    if (R <= 0 || C <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    scale_add_k<<<blocks(R * C), 256, 0, g->q>>>(x, y, scale, C, R * C);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_q3t_silu(ns_gpu* g, float* x, int64_t n) {
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    silu_k<<<blocks(n), 256, 0, g->q>>>(x, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_q3t_elu(ns_gpu* g, const float* x, int64_t n, float* out) {
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    elu_k<<<blocks(n), 256, 0, g->q>>>(x, n, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_q3t_clamp(ns_gpu* g, float* x, int64_t n, float lo, float hi) {
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    clamp_k<<<blocks(n), 256, 0, g->q>>>(x, n, lo, hi);
    NS_LAUNCH_CHECK();
    return 0;
}

}  // extern "C"
