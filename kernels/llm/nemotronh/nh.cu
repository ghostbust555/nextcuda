// nh.cu: the nemotronh engine's own kernels (nh.h) - what Nemotron-H has that the shared ones do not: the Mamba-2
// mixer's conv (with its bias), its scan and its gated group norm, relu^2 for the experts, attention without positions
// (head 128) against a half cache, and the small passes between the products (RMS norm, residual add, the router's
// float32 product, the MoE combine). After llama.cpp's nemotron-h / mamba-base graphs and ggml's ssm_conv / ssm_scan.
// Plain kernels on purpose (one thread block a unit of work, no tiling): first it has to be right.
#include "ns.h"
#include "ns_internal.hpp"
#include "nh.h"

#include <cuda_fp16.h>
#include <cfloat>
#include <cmath>

namespace {

constexpr int WG = 256;

__device__ inline float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// a block's sum, every thread getting it (blockDim.x a multiple of 32, at most 1024)
__device__ float block_sum(float v) {
    __shared__ float s[32];
    v = warp_sum(v);
    const int w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    __syncthreads();
    if ((threadIdx.x & 31) == 0) s[w] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < nw; ++i) t += s[i];
    return t;
}

__device__ float block_max(float v) {
    __shared__ float s[32];
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    const int w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    __syncthreads();
    if ((threadIdx.x & 31) == 0) s[w] = v;
    __syncthreads();
    float t = s[0];
    for (int i = 1; i < nw; ++i) t = fmaxf(t, s[i]);
    return t;
}

// a thread a channel: its k-1 inputs before, then the rows in order (the state's last k-1 written back)
__global__ void conv_k(const float* x, int64_t xs, float* state, const float* w, const float* b, float* out, int64_t T, int64_t C, int k) {
    const int64_t c = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float win[8];
    for (int j = 0; j < k - 1; ++j) win[j] = state[j * C + c];
    const float bias = b ? b[c] : 0.0f;
    for (int64_t t = 0; t < T; ++t) {
        win[k - 1] = x[t * xs + c];
        float s = bias;
        for (int j = 0; j < k; ++j) s += win[j] * w[c * k + j];
        out[t * C + c] = s / (1.0f + __expf(-s));
        for (int j = 0; j < k - 1; ++j) win[j] = win[j + 1];
    }
    for (int j = 0; j < k - 1; ++j) state[j * C + c] = win[j];
}

// a block a (head, p): its threads the state's N values; the rows in order, C . S reduced over the block
__global__ void ssm_k(const float* xbc, int64_t xs, const float* dt, int64_t dts, const float* dt_bias, const float* A, const float* Dp, float* S, float* y,
                      int64_t T, int64_t H, int64_t P, int64_t N, int64_t G) {
    const int64_t hp = blockIdx.x, h = hp / P, p = hp % P, gr = h / (H / G);
    const int n = threadIdx.x;
    const int64_t xd = H * P;
    float* sp = S + (h * P + p) * N;
    float s = n < N ? sp[n] : 0.0f;
    const float a = A[h], dd = Dp[h], bias = dt_bias[h];
    for (int64_t t = 0; t < T; ++t) {
        const float* row = xbc + t * xs;
        float d = dt[t * dts + h] + bias;
        d = d <= 20.0f ? log1pf(expf(d)) : d;
        const float dA = expf(d * a);
        const float xv = row[h * P + p];
        float part = 0.0f;
        if (n < N) {
            s = s * dA + row[xd + gr * N + n] * (xv * d);
            part = s * row[xd + G * N + gr * N + n];
        }
        const float acc = block_sum(part);
        if (n == 0) y[t * xd + h * P + p] = acc + dd * xv;
    }
    if (n < N) sp[n] = s;
}

// a block a (row, group of `group` channels)
__global__ void gated_norm_k(const float* y, const float* z, int64_t zs, const float* w, float* out, int64_t C, int64_t group, float eps) {
    const int64_t ng = C / group, t = blockIdx.x / ng, gi = blockIdx.x % ng;
    const float* yr = y + t * C + gi * group;
    const float* zr = z + t * zs + gi * group;
    float ss = 0.0f;
    for (int64_t i = threadIdx.x; i < group; i += blockDim.x) {
        const float zv = zr[i];
        const float v = yr[i] * (zv / (1.0f + expf(-zv)));
        ss += v * v;
    }
    ss = block_sum(ss);
    const float inv = rsqrtf(ss / (float) group + eps);
    float* o = out + t * C + gi * group;
    for (int64_t i = threadIdx.x; i < group; i += blockDim.x) {
        const float zv = zr[i];
        o[i] = yr[i] * (zv / (1.0f + expf(-zv))) * inv * w[gi * group + i];
    }
}

__global__ void relu2_k(float* x, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = fmaxf(x[i], 0.0f);
        x[i] = v * v;
    }
}

__global__ void kv_store_k(const float* k, int64_t ks, const float* v, int64_t vs, int64_t T, int64_t Hkv, int64_t D, int64_t cap, int64_t p0, __half* ko, __half* vo) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * Hkv * D) return;
    const int64_t d = i % D, h = (i / D) % Hkv, t = i / (D * Hkv);
    const int64_t at = (h * cap + p0 + t) * D + d;
    ko[at] = __float2half_rn(k[t * ks + h * D + d]);
    vo[at] = __float2half_rn(v[t * vs + h * D + d]);
}

constexpr int ATT_KEYS = 12288;   // 48 KiB of scores

// a block (128 threads) a (row, query head): the scores of the keys it may see in shared memory, softmax, the values
__global__ void attn_k(const float* q, int64_t qs, const __half* kc, const __half* vc, int64_t Hq, int64_t Hkv, int64_t cap, int64_t p0, float* out) {
    constexpr int D = 128;
    extern __shared__ float sc[];
    __shared__ float qv[D];
    const int64_t r = blockIdx.x / Hq, hq = blockIdx.x % Hq, hk = hq / (Hq / Hkv);
    const int64_t kv = p0 + r + 1;
    const float scale = rsqrtf((float) D);
    qv[threadIdx.x] = q[r * qs + hq * D + threadIdx.x] * scale;
    __syncthreads();
    const __half* kb = kc + hk * cap * D;
    const __half* vb = vc + hk * cap * D;
    float mx = -INFINITY;
    for (int64_t j = threadIdx.x; j < kv; j += blockDim.x) {
        float d = 0.0f;
        for (int i = 0; i < D; ++i) d += qv[i] * __half2float(kb[j * D + i]);
        sc[j] = d;
        mx = fmaxf(mx, d);
    }
    mx = block_max(mx);
    float sum = 0.0f;
    for (int64_t j = threadIdx.x; j < kv; j += blockDim.x) {
        const float e = expf(sc[j] - mx);
        sc[j] = e;
        sum += e;
    }
    sum = block_sum(sum);
    __syncthreads();
    float acc = 0.0f;
    for (int64_t j = 0; j < kv; ++j) acc += sc[j] * __half2float(vb[j * D + threadIdx.x]);
    out[(r * Hq + hq) * D + threadIdx.x] = acc / sum;
}

__global__ void rms_k(const float* x, const float* w, float* y, int64_t C, float eps) {
    const float* xr = x + (int64_t) blockIdx.x * C;
    float ss = 0.0f;
    for (int64_t i = threadIdx.x; i < C; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss);
    const float inv = rsqrtf(ss / (float) C + eps);
    float* yr = y + (int64_t) blockIdx.x * C;
    for (int64_t i = threadIdx.x; i < C; i += blockDim.x) yr[i] = xr[i] * inv * w[i];
}

__global__ void add_k(float* y, const float* x, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}

// a warp a (row of W): the tokens' dot products
__global__ void gemv_f32_k(const float* W, const float* x, float* y, int64_t T, int64_t R, int64_t C) {
    const int64_t r = (int64_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= R) return;
    const int lane = threadIdx.x & 31;
    for (int64_t t = 0; t < T; ++t) {
        float s = 0.0f;
        for (int64_t c = lane; c < C; c += 32) s += W[r * C + c] * x[t * C + c];
        s = warp_sum(s);
        if (lane == 0) y[t * R + r] = s;
    }
}

__global__ void combine_k(float* out, const float* dn, const int32_t* tp, const float* w, int64_t L) {
    const int64_t t = blockIdx.x;
    for (int64_t i = threadIdx.x; i < L; i += blockDim.x) {
        float s = 0.0f;
        for (int e = tp[t]; e < tp[t + 1]; ++e) s += w[e] * dn[(int64_t) e * L + i];
        out[t * L + i] = s;
    }
}

// out row j = x row rows[j] (rows of `rb` bytes, copied as 4-byte words)
__global__ void gather_rows_k(const uint32_t* x, const int32_t* rows, uint32_t* out, int64_t rw, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * rw) return;
    out[i] = x[(int64_t) rows[i / rw] * rw + i % rw];
}

// out [t] = sum over the token's entries e in [tp[t], tp[t+1]) of w[ids[e]] * dn[ids[e]] (entries in any order, ids their places)
__global__ void combine_ids_k(float* out, const float* dn, const int32_t* tp, const int32_t* ids, const float* w, int64_t L) {
    const int64_t t = blockIdx.x;
    for (int64_t i = threadIdx.x; i < L; i += blockDim.x) {
        float s = 0.0f;
        for (int e = tp[t]; e < tp[t + 1]; ++e) s += w[ids[e]] * dn[(int64_t) ids[e] * L + i];
        out[t * L + i] = s;
    }
}

}  // namespace

extern "C" {

int ns_nh_gather_rows(ns_gpu* g, const void* x, const int32_t* rows, void* out, int64_t row_bytes, int64_t n) {
    NS_TRY
    if (n <= 0) return 0;
    if (row_bytes % 4 != 0) return ns_fail("ns_nh_gather_rows: rows of whole words");
    const int64_t rw = row_bytes / 4;
    gather_rows_k<<<(unsigned) ((n * rw + 255) / 256), 256, 0, g->q>>>((const uint32_t*) x, rows, (uint32_t*) out, rw, n);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_nh_moe_combine_ids(ns_gpu* g, float* out, const float* dn, const int32_t* tp, const int32_t* ids, const float* w, int64_t T, int64_t L) {
    NS_TRY
    if (T <= 0) return 0;
    combine_ids_k<<<(unsigned) T, 256, 0, g->q>>>(out, dn, tp, ids, w, L);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_nh_conv(ns_gpu* g, const float* x, int64_t xs, float* state, const float* w, const float* b, float* out, int64_t T, int64_t C, int k) {
    NS_TRY
    if (T <= 0) return 0;
    if (k < 2 || k > 8) return ns_fail("ns_nh_conv: a kernel of 2 to 8");
    conv_k<<<(unsigned) ((C + 127) / 128), 128, 0, g->q>>>(x, xs, state, w, b, out, T, C, k);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_nh_ssm(ns_gpu* g, const float* xbc, int64_t xs, const float* dt, int64_t dts, const float* dt_bias, const float* A, const float* Dp, float* S,
              float* y, int64_t T, int64_t H, int64_t P, int64_t N, int64_t G) {
    NS_TRY
    if (T <= 0) return 0;
    if (N % 32 != 0 || N > 256 || H % G != 0) return ns_fail("ns_nh_ssm: d_state a multiple of 32 up to 256, the heads a multiple of the groups");
    ssm_k<<<(unsigned) (H * P), (unsigned) N, 0, g->q>>>(xbc, xs, dt, dts, dt_bias, A, Dp, S, y, T, H, P, N, G);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_nh_gated_norm(ns_gpu* g, const float* y, const float* z, int64_t zs, const float* w, float* out, int64_t T, int64_t C, int64_t group, float eps) {
    NS_TRY
    if (T <= 0) return 0;
    if (C % group != 0) return ns_fail("ns_nh_gated_norm: the channels a multiple of the group");
    gated_norm_k<<<(unsigned) (T * (C / group)), WG, 0, g->q>>>(y, z, zs, w, out, C, group, eps);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_nh_relu2(ns_gpu* g, float* x, int64_t n) {
    NS_TRY
    if (n > 0) relu2_k<<<(unsigned) ((n + 255) / 256), 256, 0, g->q>>>(x, n);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_nh_kv_store(ns_gpu* g, const float* k, int64_t ks, const float* v, int64_t vs, int64_t T, int64_t Hkv, int64_t D, int64_t cap, int64_t p0,
                   void* kc, void* vc) {
    NS_TRY
    if (T <= 0) return 0;
    if (p0 + T > cap) return ns_fail("ns_nh_kv_store: past the cache's capacity");
    const int64_t n = T * Hkv * D;
    kv_store_k<<<(unsigned) ((n + 255) / 256), 256, 0, g->q>>>(k, ks, v, vs, T, Hkv, D, cap, p0, (__half*) kc, (__half*) vc);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int64_t ns_nh_attn_max_keys(void) { return ATT_KEYS; }

int ns_nh_attn(ns_gpu* g, const float* q, int64_t qs, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t D, int64_t cap,
               int64_t p0, float* out) {
    NS_TRY
    if (T <= 0) return 0;
    if (p0 + T > cap) return ns_fail("ns_nh_attn: past the cache's capacity");
    if (p0 + T > ATT_KEYS) return ns_fail("ns_nh_attn: more keys than the plain kernel holds scores for");
    if (D != 128 || Hq % Hkv != 0) return ns_fail("ns_nh_attn: head 128, query heads a multiple of the key heads");
    static bool raised = false;
    if (!raised) {
        NS_CU(cudaFuncSetAttribute(attn_k, cudaFuncAttributeMaxDynamicSharedMemorySize, ATT_KEYS * 4));
        raised = true;
    }
    attn_k<<<(unsigned) (T * Hq), 128, (size_t) (p0 + T) * 4, g->q>>>(q, qs, (const __half*) kc, (const __half*) vc, Hq, Hkv, cap, p0, out);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_nh_rms_norm(ns_gpu* g, const float* x, const float* w, float* y, int64_t T, int64_t C, float eps) {
    NS_TRY
    if (T <= 0) return 0;
    rms_k<<<(unsigned) T, WG, 0, g->q>>>(x, w, y, C, eps);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_nh_add(ns_gpu* g, float* y, const float* x, int64_t n) {
    NS_TRY
    if (n > 0) add_k<<<(unsigned) ((n + 255) / 256), 256, 0, g->q>>>(y, x, n);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_nh_gemv_f32(ns_gpu* g, const float* W, const float* x, float* y, int64_t T, int64_t R, int64_t C) {
    NS_TRY
    if (T <= 0) return 0;
    gemv_f32_k<<<(unsigned) ((R + 7) / 8), 256, 0, g->q>>>(W, x, y, T, R, C);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_nh_moe_combine(ns_gpu* g, float* out, const float* dn, const int32_t* tp, const float* w, int64_t T, int64_t L) {
    NS_TRY
    if (T <= 0) return 0;
    combine_k<<<(unsigned) T, 256, 0, g->q>>>(out, dn, tp, w, L);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

}  // extern "C"
