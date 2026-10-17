// pocr.cu: the PaddleOCR-VL engine's kernels (pocr.h) - the projector's 2 x 2 merge, the decoder's cache copies and
// its one-row attention over grouped key / value heads; the linears, norms, RoPE, GELU and the prompt's attention are
// the shared diffusion kernels' (nsd.h). nextsycl's pocr.cpp in CUDA (the same names, arguments and arithmetic): its
// sub-groups of 16 are here 16 threads reduced with half-warp shuffles.
#include "ns.h"
#include "ns_internal.hpp"
#include "pocr.h"

#include <cuda_fp16.h>
#include <cmath>

namespace {

unsigned blocks(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }

__global__ void to_half_k(const float* x, __half* o, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = __float2half_rn(x[i]);
}

__global__ void copy_rows_k(const float* src, int64_t ss, float* dst, int64_t ds, int64_t cols, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[(i / cols) * ds + i % cols] = src[(i / cols) * ss + i % cols];
}

__global__ void merge2_k(const float* x, int64_t W, int64_t C, float* out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t w2 = W / 2, c = i % (4 * C), r = i / (4 * C);
    const int64_t q = c / C, cc = c % C, y = (r / w2) * 2 + q / 2, xx = (r % w2) * 2 + q % 2;
    out[i] = x[(y * W + xx) * C + cc];
}

constexpr int64_t CHUNK = 256;

// 16 threads a (kv head, key), each D / 16 features, every query head of the group at once
__global__ void scores_k(const float* q, const float* k, int64_t n, int64_t kvs, int64_t rep, int64_t D, float scale, float* sc) {
    const int64_t w = blockIdx.x, gh = w / n, j = w % n;
    const int l = threadIdx.x;
    const int64_t per = D / 16;
    const float* kj = k + j * kvs + gh * D + l * per;
    float acc[16];
    for (int64_t r = 0; r < rep; ++r) acc[r] = 0.0f;
    for (int64_t e = 0; e < per; ++e) {
        const float kv = kj[e];
        for (int64_t r = 0; r < rep; ++r) acc[r] += q[(gh * rep + r) * D + l * per + e] * kv;
    }
    for (int64_t r = 0; r < rep; ++r) {
        float s = acc[r];
        for (int o = 8; o > 0; o >>= 1) s += __shfl_xor_sync(0x0000ffffu, s, o, 16);
        if (l == 0) sc[(gh * rep + r) * n + j] = s * scale;
    }
}

__device__ float block_max(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = red[0];
    for (int i = 1; i < (int) (blockDim.x >> 5); ++i) t = fmaxf(t, red[i]);
    __syncthreads();
    return t;
}

__device__ float block_sum(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < (int) (blockDim.x >> 5); ++i) t += red[i];
    __syncthreads();
    return t;
}

// each (head, chunk): its max, its sum of exp and its exp-weighted values, a thread a feature
__global__ void chunk_k(const float* sc, const float* v, int64_t n, int64_t kvs, int64_t rep, int64_t D, int64_t chunks, float* part) {
    __shared__ float red[8];
    const int64_t w = blockIdx.x, h = w / chunks, c = w % chunks;
    const int l = threadIdx.x, wg = blockDim.x;
    const int64_t j0 = c * CHUNK, j1 = min(n, j0 + CHUNK);
    const float* s = sc + h * n;
    float m = -INFINITY;
    for (int64_t j = j0 + l; j < j1; j += wg) m = fmaxf(m, s[j]);
    m = block_max(m, red);
    float sum = 0.0f;
    for (int64_t j = j0 + l; j < j1; j += wg) sum += expf(s[j] - m);
    sum = block_sum(sum, red);
    float* p = part + (h * chunks + c) * (D + 2);
    if (l < D) {
        const float* vh = v + (h / rep) * D + l;
        float acc = 0.0f;
        for (int64_t j = j0; j < j1; ++j) acc += expf(s[j] - m) * vh[j * kvs];
        p[2 + l] = acc;
    }
    if (l == 0) {
        p[0] = m;
        p[1] = sum;
    }
}

__global__ void combine_k(const float* part, int64_t D, int64_t chunks, float* out) {
    const int64_t h = blockIdx.x;
    const int l = threadIdx.x;
    if (l >= D) return;
    const float* p = part + h * chunks * (D + 2);
    float m = -INFINITY;
    for (int64_t c = 0; c < chunks; ++c) m = fmaxf(m, p[c * (D + 2)]);
    float sum = 0.0f, acc = 0.0f;
    for (int64_t c = 0; c < chunks; ++c) {
        const float f = expf(p[c * (D + 2)] - m);
        sum += p[c * (D + 2) + 1] * f;
        acc += p[c * (D + 2) + 2 + l] * f;
    }
    out[h * D + l] = acc / sum;
}

}  // namespace

extern "C" {

int ns_image_pocr_to_half(ns_gpu* g, const float* x, void* out, int64_t n) {
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    to_half_k<<<blocks(n), 256, 0, g->q>>>(x, (__half*) out, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_pocr_copy_rows(ns_gpu* g, const float* src, int64_t ss, float* dst, int64_t ds, int64_t rows, int64_t cols) {
    if (rows <= 0 || cols <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    copy_rows_k<<<blocks(rows * cols), 256, 0, g->q>>>(src, ss, dst, ds, cols, rows * cols);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_pocr_merge2(ns_gpu* g, const float* x, int64_t H, int64_t W, int64_t C, float* out) {
    if (H % 2 || W % 2) return ns_fail("ns_image_pocr_merge2: even sides");
    const int64_t n = H * W * C;
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    merge2_k<<<blocks(n), 256, 0, g->q>>>(x, W, C, out, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int64_t ns_image_pocr_attn1_scratch(int64_t n, int64_t Hq, int64_t D) {
    const int64_t chunks = (n + CHUNK - 1) / CHUNK;
    return Hq * n + Hq * chunks * (D + 2);
}

int ns_image_pocr_attn1(ns_gpu* g, const float* q, const float* k, const float* v, int64_t n, int64_t kvs, int64_t Hq, int64_t Hkv,
                        int64_t D, float* scratch, float* out) {
    if (n <= 0) return 0;
    if (Hq % Hkv || D % 16 || D > 256 || Hq / Hkv > 16)
        return ns_fail("ns_image_pocr_attn1: Hq a multiple of Hkv (at most 16 a kv head), D a multiple of 16 up to 256");
    NS_CU(cudaSetDevice(g->dev));
    const int64_t rep = Hq / Hkv, chunks = (n + CHUNK - 1) / CHUNK;
    float* sc = scratch;
    float* part = scratch + Hq * n;
    scores_k<<<(unsigned) (Hkv * n), 16, 0, g->q>>>(q, k, n, kvs, rep, D, 1.0f / sqrtf((float) D), sc);
    NS_LAUNCH_CHECK();
    const int wg = D <= 64 ? 64 : (D <= 128 ? 128 : 256);
    chunk_k<<<(unsigned) (Hq * chunks), wg, 0, g->q>>>(sc, v, n, kvs, rep, D, chunks, part);
    NS_LAUNCH_CHECK();
    combine_k<<<(unsigned) Hq, wg, 0, g->q>>>(part, D, chunks, out);
    NS_LAUNCH_CHECK();
    return 0;
}

}  // extern "C"
