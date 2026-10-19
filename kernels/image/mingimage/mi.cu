// mi.cu: the Ming-Image engine's own small kernels - what the shared diffusion kernels (nsd) do not have: the DiT's
// modulated norms and tanh-gated residuals, the text encoder's mixture-of-experts rows (gathered by expert, summed back
// by token), the grouped heads made full, the sampler's step, the VAE's upsampling, softmax and the picture's bytes.
// Elementwise or one block a row, on the GPU's stream.
#include "ns.h"
#include "ns_internal.hpp"
#include "mi.h"

#include <cuda_fp16.h>
#include <cfloat>

namespace {

constexpr int WG = 256;
constexpr int PER = 32;  // a row of up to 8,192 values held in registers

__device__ inline float h2f(__half h) { return __half2float(h); }
__device__ inline __half f2h(float f) { return __float2half_rn(f); }

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

__global__ void rms_mod_k(const float* x, int64_t C, const float* w, float eps, const float* scale, __half* o) {
    const size_t r = blockIdx.x;
    const int l = threadIdx.x;
    const float* row = x + r * C;
    float v[PER];
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t i = (int64_t) j * WG + l;
        v[j] = i < C ? row[i] : 0.0f;
        s += v[j] * v[j];
    }
    const float inv = rsqrtf(block_sum(s) / (float) C + eps);
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t i = (int64_t) j * WG + l;
        if (i < C) o[r * C + i] = f2h(v[j] * inv * (w ? w[i] : 1.0f) * (scale ? 1.0f + scale[i] : 1.0f));
    }
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

__global__ void to_rgba8_k(const __half* s, uint8_t* rgba, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = fminf(fmaxf(h2f(s[i]), -1.0f), 1.0f) * 0.5f + 0.5f;
    rgba[i] = (uint8_t) fminf(fmaxf(roundf(v * 255.0f), 0.0f), 255.0f);
}

__global__ void gather_k(const __half* x, int64_t C, const int32_t* idx, __half* o) {
    const int64_t r = blockIdx.x;
    const int64_t src = idx[r];
    for (int64_t i = threadIdx.x; i < C; i += blockDim.x) o[r * C + i] = x[src * C + i];
}

__global__ void swiglu_scaled_k(const __half* x, int64_t C, int64_t n, float s, __half* o) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t r = i / C, c = i % C;
    const float a = h2f(x[r * 2 * C + c]), b = h2f(x[r * 2 * C + C + c]);
    o[i] = f2h(a / (1.0f + expf(-a)) * b * s);
}

__global__ void to_half_rows_k(const float* x, int64_t C, int64_t n, float s, __half* o, int64_t ld) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    o[(i / C) * ld + i % C] = f2h(x[i] * s);
}

__global__ void combine_k(float* x, int64_t C, const float* y, const int32_t* slot, const float* w, int64_t K) {
    const int64_t t = blockIdx.x;
    for (int64_t i = threadIdx.x; i < C; i += blockDim.x) {
        float s = 0.0f;
        for (int64_t j = 0; j < K; ++j) {
            const int32_t sl = slot[t * K + j];
            if (sl >= 0) s += w[t * K + j] * y[(int64_t) sl * C + i];
        }
        x[t * C + i] += s;
    }
}

__global__ void gate_add_k(float* x, int64_t C, int64_t n, const __half* o, const float* gate) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] += h2f(o[i]) * (gate ? gate[i % C] : 1.0f);
}

__global__ void repeat_heads_k(const __half* x, int64_t stride, int64_t off, int64_t Hkv, int64_t Hq, int64_t D, __half* o) {
    const int64_t r = blockIdx.x;
    const int64_t rep = Hq / Hkv;
    for (int64_t i = threadIdx.x; i < Hq * D; i += blockDim.x) {
        const int64_t h = i / D, d = i % D;
        o[r * Hq * D + i] = x[r * stride + off + (h / rep) * D + d];
    }
}

inline unsigned blocks(int64_t n) { return (unsigned) ((n + 255) / 256); }

}  // namespace

extern "C" {

int ns_image_mi_ln_mod(ns_gpu* g, const float* x, int64_t M, int64_t C, float eps, const float* scale, void* out) {
    if (C > (int64_t) WG * PER) return ns_fail("ns_image_mi_ln_mod: rows longer than 8192");
    NS_CU(cudaSetDevice(g->dev));
    ln_mod_k<<<(unsigned) M, WG, 0, g->q>>>(x, C, eps, scale, (__half*) out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_mi_rms_mod(ns_gpu* g, const float* x, int64_t M, int64_t C, const float* w, float eps, const float* scale, void* out) {
    if (C > (int64_t) WG * PER) return ns_fail("ns_image_mi_rms_mod: rows longer than 8192");
    NS_CU(cudaSetDevice(g->dev));
    rms_mod_k<<<(unsigned) M, WG, 0, g->q>>>(x, C, w, eps, scale, (__half*) out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_mi_softmax_rows(ns_gpu* g, const float* x, int64_t M, int64_t N, float scale, void* out) {
    NS_CU(cudaSetDevice(g->dev));
    softmax_rows_k<<<(unsigned) M, WG, 0, g->q>>>(x, N, scale, (__half*) out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_mi_silu(ns_gpu* g, void* x, int64_t n) {
    NS_CU(cudaSetDevice(g->dev));
    silu_k<<<blocks(n), 256, 0, g->q>>>((__half*) x, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_mi_axpy(ns_gpu* g, float* x, const float* y, int64_t n, float a) {
    NS_CU(cudaSetDevice(g->dev));
    axpy_k<<<blocks(n), 256, 0, g->q>>>(x, y, n, a);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_mi_to_half(ns_gpu* g, const float* x, void* out, int64_t n) {
    NS_CU(cudaSetDevice(g->dev));
    to_half_k<<<blocks(n), 256, 0, g->q>>>(x, (__half*) out, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_mi_to_float(ns_gpu* g, const void* x, float* out, int64_t n) {
    NS_CU(cudaSetDevice(g->dev));
    to_float_k<<<blocks(n), 256, 0, g->q>>>((const __half*) x, out, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_mi_up2(ns_gpu* g, const void* x, int64_t H, int64_t W, int64_t C, void* out) {
    NS_CU(cudaSetDevice(g->dev));
    up2_k<<<blocks(4 * H * W * C), 256, 0, g->q>>>((const __half*) x, (__half*) out, H, W, C);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_mi_to_rgba8(ns_gpu* g, const void* x, int64_t H, int64_t W, uint8_t* rgba) {
    NS_CU(cudaSetDevice(g->dev));
    to_rgba8_k<<<blocks(H * W * 4), 256, 0, g->q>>>((const __half*) x, rgba, H * W * 4);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_mi_gather_rows(ns_gpu* g, const void* x, int64_t C, const int32_t* idx, int64_t n, void* out) {
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    gather_k<<<(unsigned) n, WG, 0, g->q>>>((const __half*) x, C, idx, (__half*) out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_mi_combine(ns_gpu* g, float* x, int64_t M, int64_t C, const void* y, const int32_t* slot, const float* w, int64_t K) {
    if (M <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    combine_k<<<(unsigned) M, WG, 0, g->q>>>(x, C, (const float*) y, slot, w, K);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_mi_swiglu_scaled(ns_gpu* g, const void* x, int64_t M, int64_t C, float s, void* out) {
    NS_CU(cudaSetDevice(g->dev));
    swiglu_scaled_k<<<blocks(M * C), 256, 0, g->q>>>((const __half*) x, C, M * C, s, (__half*) out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_mi_to_half_rows(ns_gpu* g, const float* x, int64_t M, int64_t C, float s, void* out, int64_t ld) {
    NS_CU(cudaSetDevice(g->dev));
    to_half_rows_k<<<blocks(M * C), 256, 0, g->q>>>(x, C, M * C, s, (__half*) out, ld);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_mi_gate_add(ns_gpu* g, float* x, int64_t M, int64_t C, const void* other, const float* gate) {
    NS_CU(cudaSetDevice(g->dev));
    gate_add_k<<<blocks(M * C), 256, 0, g->q>>>(x, C, M * C, (const __half*) other, gate);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_mi_repeat_heads(ns_gpu* g, const void* x, int64_t rows, int64_t stride, int64_t off, int64_t Hkv, int64_t Hq, int64_t D, void* out) {
    if (Hkv <= 0 || Hq % Hkv != 0) return ns_fail("ns_image_mi_repeat_heads: query heads not a multiple of the key / value heads");
    NS_CU(cudaSetDevice(g->dev));
    repeat_heads_k<<<(unsigned) rows, WG, 0, g->q>>>((const __half*) x, stride, off, Hkv, Hq, D, (__half*) out);
    NS_LAUNCH_CHECK();
    return 0;
}

}  // extern "C"
