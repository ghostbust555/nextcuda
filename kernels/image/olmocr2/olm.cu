// olm.cu: the olmOCR 2 engine's kernels (olm.h) - the vision tower's attention within each picture's segment, the
// decoder's Q8_0 product read as int8 rows with a half scale a block (one row a 16-thread group) and to half for the
// prompt's GEMMs, its one-row cached attention (as PaddleOCR-VL's), row copies and gathers. nextsycl's olm.cpp in CUDA,
// the same names, arguments and arithmetic.
#include "ns.h"
#include "ns_internal.hpp"
#include "olm.h"

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

__global__ void gather_k(const float* src, int64_t C, const int32_t* idx, float* out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = src[(int64_t) idx[i / C] * C + i % C];
}

// 16 threads a (token, head), each D / 16 features; an online softmax over the segment's keys [lo[i], hi[i])
__global__ void seg_attn_k(const float* q, const float* k, const float* v, int64_t stride, int64_t H, int64_t D, const int32_t* lo, const int32_t* hi,
                           float scale, float* out) {
    const int64_t w = blockIdx.x, i = w / H, h = w % H;
    const int l = threadIdx.x;
    const int64_t per = D / 16;
    float qd[16], acc[16];
    for (int64_t e = 0; e < per; ++e) {
        qd[e] = q[i * stride + h * D + l * per + e] * scale;
        acc[e] = 0.0f;
    }
    float m = -INFINITY, s = 0.0f;
    for (int64_t j = lo[i]; j < hi[i]; ++j) {
        const float* kj = k + j * stride + h * D + l * per;
        float p = 0.0f;
        for (int64_t e = 0; e < per; ++e) p += qd[e] * kj[e];
        for (int o = 8; o > 0; o >>= 1) p += __shfl_xor_sync(0x0000ffffu, p, o, 16);
        const float mn = fmaxf(m, p), a = expf(m - mn), b = expf(p - mn);
        const float* vj = v + j * stride + h * D + l * per;
        for (int64_t e = 0; e < per; ++e) acc[e] = acc[e] * a + b * vj[e];
        s = s * a + b;
        m = mn;
    }
    for (int64_t e = 0; e < per; ++e) out[i * H * D + h * D + l * per + e] = acc[e] / s;
}

// 16 threads a row: each step they read 256 values (16 bytes a thread, 8 blocks of 32), coalesced; 4 rows a block
__global__ void q8r_gemv_k(const int8_t* qs, const __half* dh, const float* x, int64_t n, int64_t k, const float* bias, float* y) {
    const int64_t r = (int64_t) blockIdx.x * 4 + threadIdx.x / 16;
    const int l = threadIdx.x % 16;
    const int64_t nb = k / 32;
    float acc = 0.0f;
    if (r < n) {
        const int8_t* row = qs + r * k;
        const __half* rd = dh + r * nb;
        for (int64_t at = l * 16; at < k; at += 256) {
            const int4 qq = *reinterpret_cast<const int4*>(row + at);
            const int8_t* q = reinterpret_cast<const int8_t*>(&qq);
            const float4* xv = reinterpret_cast<const float4*>(x + at);
            float s = 0.0f;
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const float4 xx = xv[v];
                s += (float) q[4 * v] * xx.x + (float) q[4 * v + 1] * xx.y + (float) q[4 * v + 2] * xx.z + (float) q[4 * v + 3] * xx.w;
            }
            acc += s * __half2float(rd[at / 32]);
        }
    }
    for (int o = 8; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o, 16);
    if (r < n && l == 0) y[r] = acc + (bias ? bias[r] : 0.0f);
}

__global__ void q8r_half_k(const int8_t* qs, const __half* dh, __half* o, int64_t total) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) o[i] = __float2half_rn((float) qs[i] * __half2float(dh[i / 32]));
}

}  // namespace

extern "C" {

int ns_image_olm_to_half(ns_gpu* g, const float* x, void* out, int64_t n) {
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    to_half_k<<<blocks(n), 256, 0, g->q>>>(x, (__half*) out, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_olm_copy_rows(ns_gpu* g, const float* src, int64_t ss, float* dst, int64_t ds, int64_t rows, int64_t cols) {
    if (rows <= 0 || cols <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    copy_rows_k<<<blocks(rows * cols), 256, 0, g->q>>>(src, ss, dst, ds, cols, rows * cols);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_olm_gather(ns_gpu* g, const float* src, int64_t C, const int32_t* idx, int64_t n, float* out) {
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    gather_k<<<blocks(n * C), 256, 0, g->q>>>(src, C, idx, out, n * C);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_olm_seg_attn(ns_gpu* g, const float* q, const float* k, const float* v, int64_t stride, int64_t N, int64_t H, int64_t D,
                          const int32_t* lo, const int32_t* hi, float* out) {
    if (N <= 0) return 0;
    if (D % 16 || D > 256) return ns_fail("ns_image_olm_seg_attn: D a multiple of 16, at most 256");
    NS_CU(cudaSetDevice(g->dev));
    seg_attn_k<<<(unsigned) (N * H), 16, 0, g->q>>>(q, k, v, stride, H, D, lo, hi, 1.0f / sqrtf((float) D), out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_olm_q8r_gemv(ns_gpu* g, const int8_t* qs, const void* d, const float* x, int64_t n, int64_t k, const float* bias, float* y) {
    if (n <= 0) return 0;
    if (k % 256) return ns_fail("ns_image_olm_q8r_gemv: k a multiple of 256");
    NS_CU(cudaSetDevice(g->dev));
    q8r_gemv_k<<<(unsigned) ((n + 3) / 4), 64, 0, g->q>>>(qs, (const __half*) d, x, n, k, bias, y);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_olm_q8r_half(ns_gpu* g, const int8_t* qs, const void* d, int64_t n, int64_t k, void* out) {
    const int64_t total = n * k;
    if (total <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    q8r_half_k<<<blocks(total), 256, 0, g->q>>>(qs, (const __half*) d, (__half*) out, total);
    NS_LAUNCH_CHECK();
    return 0;
}

int64_t ns_image_olm_attn1_scratch(int64_t n, int64_t Hq, int64_t D) {
    const int64_t chunks = (n + CHUNK - 1) / CHUNK;
    return Hq * n + Hq * chunks * (D + 2);
}

int ns_image_olm_attn1(ns_gpu* g, const float* q, const float* k, const float* v, int64_t n, int64_t kvs, int64_t Hq, int64_t Hkv,
                        int64_t D, float* scratch, float* out) {
    if (n <= 0) return 0;
    if (Hq % Hkv || D % 16 || D > 256 || Hq / Hkv > 16)
        return ns_fail("ns_image_olm_attn1: Hq a multiple of Hkv (at most 16 a kv head), D a multiple of 16 up to 256");
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
