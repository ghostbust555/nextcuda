// qi21.cu: the Qwen-Image 2.1 engine's own small kernels - what the shared diffusion kernels (nsd) do not have: the
// norm with the modulation's (1 + scale), the activations in place, the sampler's step, the VAE's upsampling and
// shortcut, the picture's bytes. Elementwise or one block a row, on the GPU's stream. (nextsycl's qi21.cpp in CUDA:
// the same names, arguments and arithmetic, so the engine above them is the same.)
#include "ns.h"
#include "ns_internal.hpp"
#include "qi21.h"

#include <cuda_fp16.h>
#include <cfloat>

namespace {

constexpr int WG = 256;

__device__ inline float h2f(__half h) { return __half2float(h); }
__device__ inline __half f2h(float f) { return __float2half_rn(f); }

// a block's sum / max, every thread getting the result (WG threads)
__device__ float block_sum(float v) {
    __shared__ float s[WG / 32];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < WG / 32; ++i) t += s[i];
    __syncthreads();
    return t;
}

__device__ float block_max(float v) {
    __shared__ float s[WG / 32];
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = s[0];
    for (int i = 1; i < WG / 32; ++i) t = fmaxf(t, s[i]);
    __syncthreads();
    return t;
}

constexpr int PER = 32;  // a row of up to 8,192 values held in registers

__global__ void ln_mod_k(const float* x, int64_t C, float eps, const float* scale, __half* o) {
    const size_t r = blockIdx.x;
    const int l = threadIdx.x;
    const float* row = x + r * C;
    float v[PER];
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t i = (int64_t) j * WG + l;
        v[j] = i < C ? row[i] : 0.0f;
        s += v[j];
    }
    const float mean = block_sum(s) / (float) C;
    float s2 = 0.0f;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t i = (int64_t) j * WG + l;
        const float d = i < C ? v[j] - mean : 0.0f;
        s2 += d * d;
    }
    const float inv = rsqrtf(block_sum(s2) / (float) C + eps);
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t i = (int64_t) j * WG + l;
        if (i < C) o[r * C + i] = f2h((v[j] - mean) * inv * (scale ? 1.0f + scale[i] : 1.0f));
    }
}

__global__ void quant_rows_k(const float* w, int64_t K, int8_t* q, float* scale) {
    const size_t r = blockIdx.x;
    const int64_t l = threadIdx.x;
    const float* row = w + r * K;
    float m = 0.0f;
    for (int64_t i = l; i < K; i += WG) m = fmaxf(m, fabsf(row[i]));
    const float s = fmaxf(block_max(m) / 127.0f, 1e-30f);
    for (int64_t i = l; i < K; i += WG) q[r * K + i] = (int8_t) fminf(fmaxf(rintf(row[i] / s), -127.0f), 127.0f);
    if (l == 0) scale[r] = s;
}

__global__ void softmax_rows_k(const float* x, int64_t N, float scale, __half* o) {
    const size_t r = blockIdx.x;
    const int64_t l = threadIdx.x;
    const float* row = x + r * N;
    float m = -INFINITY;
    for (int64_t i = l; i < N; i += WG) m = fmaxf(m, row[i] * scale);
    m = block_max(m);
    float s = 0.0f;
    for (int64_t i = l; i < N; i += WG) s += expf(row[i] * scale - m);
    const float inv = 1.0f / block_sum(s);
    for (int64_t i = l; i < N; i += WG) o[r * N + i] = f2h(expf(row[i] * scale - m) * inv);
}

__global__ void gelu_tanh_k(__half* p, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = h2f(p[i]);
    p[i] = f2h(0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v))));
}

__global__ void silu_k(__half* p, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = h2f(p[i]);
    p[i] = f2h(v / (1.0f + expf(-v)));
}

__global__ void axpy_k(float* x, const float* y, int64_t n, float a) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += a * y[i];
}

__global__ void to_half_k(const float* x, __half* o, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = f2h(x[i]);
}

__global__ void to_float_k(const __half* p, float* o, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = h2f(p[i]);
}

__global__ void up2_k(const __half* s, __half* o, int64_t H, int64_t W, int64_t C) {
    const int64_t n = 2 * H * 2 * W * C;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t c = i % C, xx = (i / C) % (2 * W), y = i / (C * 2 * W);
    o[i] = s[((y / 2) * W + xx / 2) * C + c];
}

__global__ void dupup_add_k(const __half* s, __half* o, int64_t H, int64_t W, int64_t Ci, int64_t Co, int64_t F, int64_t R, int64_t t) {
    const int64_t n = 2 * H * 2 * W * Co;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t oc = i % Co, xx = (i / Co) % (2 * W), y = i / (Co * 2 * W);
    const int64_t ci = (oc * F + t * 4 + (y % 2) * 2 + (xx % 2)) / R;
    o[i] = f2h(h2f(o[i]) + h2f(s[((y / 2) * W + xx / 2) * Ci + ci]));
}

__global__ void to_rgba8_k(const __half* s, uint8_t* rgba, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = fminf(fmaxf(h2f(s[i]), -1.0f), 1.0f) * 0.5f + 0.5f;
    rgba[i] = (uint8_t) fminf(fmaxf(roundf(v * 255.0f), 0.0f), 255.0f);
}

__global__ void pad_br_k(const __half* xi, __half* o, int64_t H, int64_t W, int64_t C) {
    const int64_t Wo = W + 1;
    const int64_t n = (H + 1) * Wo * C;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t c = i % C, xx = (i / C) % Wo, y = i / (C * Wo);
    o[i] = (y < H && xx < W) ? xi[(y * W + xx) * C + c] : f2h(0.0f);
}

__global__ void avg_down_add_k(const __half* xi, __half* o, int64_t H, int64_t W, int64_t Ci, int64_t Co, int64_t ft, int64_t F, int64_t G) {
    const int64_t Ho = H / 2, Wo = W / 2;
    const int64_t n = Ho * Wo * Co;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t oc = i % Co, xx = (i / Co) % Wo, y = i / (Co * Wo);
    float s = 0.0f;
    for (int64_t gi = 0; gi < G; ++gi) {
        const int64_t j = oc * G + gi, c = j / F, r = j % F, t = r / 4, hs = (r % 4) / 2, ws = r % 2;
        if (ft == 2 && t == 0) continue;
        s += h2f(xi[((2 * y + hs) * W + 2 * xx + ws) * Ci + c]);
    }
    o[i] = f2h(h2f(o[i]) + s / (float) G);
}

inline unsigned blocks(int64_t n) { return (unsigned) ((n + 255) / 256); }

}  // namespace

extern "C" {

int ns_image_qi21_ln_mod(ns_gpu* g, const float* x, int64_t M, int64_t C, float eps, const float* scale, void* out) {
    if (C > (int64_t) WG * PER) return ns_fail("ns_image_qi21_ln_mod: rows longer than 8192");
    NS_CU(cudaSetDevice(g->dev));
    ln_mod_k<<<(unsigned) M, WG, 0, g->q>>>(x, C, eps, scale, (__half*) out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_qi21_quant_rows(ns_gpu* g, const float* w, int64_t N, int64_t K, int8_t* q, float* scale) {
    NS_CU(cudaSetDevice(g->dev));
    quant_rows_k<<<(unsigned) N, WG, 0, g->q>>>(w, K, q, scale);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_qi21_softmax_rows(ns_gpu* g, const float* x, int64_t M, int64_t N, float scale, void* out) {
    NS_CU(cudaSetDevice(g->dev));
    softmax_rows_k<<<(unsigned) M, WG, 0, g->q>>>(x, N, scale, (__half*) out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_qi21_gelu_tanh(ns_gpu* g, void* x, int64_t n) {
    NS_CU(cudaSetDevice(g->dev));
    gelu_tanh_k<<<blocks(n), 256, 0, g->q>>>((__half*) x, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_qi21_silu(ns_gpu* g, void* x, int64_t n) {
    NS_CU(cudaSetDevice(g->dev));
    silu_k<<<blocks(n), 256, 0, g->q>>>((__half*) x, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_qi21_axpy(ns_gpu* g, float* x, const float* y, int64_t n, float a) {
    NS_CU(cudaSetDevice(g->dev));
    axpy_k<<<blocks(n), 256, 0, g->q>>>(x, y, n, a);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_qi21_to_half(ns_gpu* g, const float* x, void* out, int64_t n) {
    NS_CU(cudaSetDevice(g->dev));
    to_half_k<<<blocks(n), 256, 0, g->q>>>(x, (__half*) out, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_qi21_to_float(ns_gpu* g, const void* x, float* out, int64_t n) {
    NS_CU(cudaSetDevice(g->dev));
    to_float_k<<<blocks(n), 256, 0, g->q>>>((const __half*) x, out, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_qi21_up2(ns_gpu* g, const void* x, int64_t H, int64_t W, int64_t C, void* out) {
    NS_CU(cudaSetDevice(g->dev));
    up2_k<<<blocks(4 * H * W * C), 256, 0, g->q>>>((const __half*) x, (__half*) out, H, W, C);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_qi21_dupup_add(ns_gpu* g, const void* x, int64_t H, int64_t W, int64_t Ci, int64_t Co, int ft, void* out) {
    if (ft != 1 && ft != 2) return ns_fail("ns_image_qi21_dupup_add: 1 or 2 time duplicates");
    const int64_t F = (int64_t) ft * 4;
    if ((Co * F) % Ci != 0) return ns_fail("ns_image_qi21_dupup_add: out channels x factor not a multiple of in channels");
    const int64_t R = Co * F / Ci;
    NS_CU(cudaSetDevice(g->dev));
    dupup_add_k<<<blocks(4 * H * W * Co), 256, 0, g->q>>>((const __half*) x, (__half*) out, H, W, Ci, Co, F, R, ft - 1);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_qi21_to_rgba8(ns_gpu* g, const void* x, int64_t H, int64_t W, uint8_t* rgba) {
    NS_CU(cudaSetDevice(g->dev));
    to_rgba8_k<<<blocks(H * W * 4), 256, 0, g->q>>>((const __half*) x, rgba, H * W * 4);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_qi21_pad_br(ns_gpu* g, const void* x, int64_t H, int64_t W, int64_t C, void* out) {
    NS_CU(cudaSetDevice(g->dev));
    pad_br_k<<<blocks((H + 1) * (W + 1) * C), 256, 0, g->q>>>((const __half*) x, (__half*) out, H, W, C);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_qi21_avg_down_add(ns_gpu* g, const void* x, int64_t H, int64_t W, int64_t Ci, int64_t Co, int ft, void* out) {
    const int64_t F = (int64_t) ft * 4;
    if (Ci * F % Co != 0) return ns_fail("ns_image_qi21_avg_down_add: Ci * F not a multiple of Co");
    const int64_t G = Ci * F / Co;
    NS_CU(cudaSetDevice(g->dev));
    avg_down_add_k<<<blocks((H / 2) * (W / 2) * Co), 256, 0, g->q>>>((const __half*) x, (__half*) out, H, W, Ci, Co, ft, F, G);
    NS_LAUNCH_CHECK();
    return 0;
}

}  // extern "C"
