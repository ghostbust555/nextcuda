// q35.cu: the qwen35 engine's own kernels (q35.h) - what dense Qwen3.5 / Qwen3.8 has that the shared ones do not: the
// gated full attention (head 256, a gate per head in the query projection, partial rotary positions) against a half
// cache, DeltaNet's gates, its conv, its delta-rule scan and its SiLU-gated output norm. After nextsycl's
// kernels/llm/qwen35/q35.cpp and the generic kernels of its glm5next engine (glm.cpp: conv + SiLU, L2 norm, the KDA scan,
// SwiGLU) - the same names (glm's under ns_q35_), arguments and sums; after llama.cpp's qwen35 graph.
#include "ns.h"
#include "ns_internal.hpp"
#include "q35.h"

#include <cuda_fp16.h>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

namespace {

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

// ---- the cache's rows: [Hkv][cap] rows of the head's 256 features. KT 0: half (512 bytes); 1: q8_0 (llama.cpp's
// cache-type q8_0: 256 int8, then the 8 half scales of its blocks of 32 - 272 bytes); 2: q4_0 (ggml's blocks of 32 as
// nibbles: a block's byte j value j low, value j + 16 high - 128 bytes, then the 8 half scales - 144 bytes). `row` = key
// head * cap + position
constexpr int64_t Q8ROW = 256 + 16, Q4ROW = 128 + 16;
template <int KT>
__host__ __device__ constexpr int64_t kv_row() { return KT == 0 ? 512 : KT == 1 ? Q8ROW : Q4ROW; }

// 8 features (c8 * 8 ..) of a quantized row r as floats
template <int KT>
__device__ __forceinline__ void widen8f(const unsigned char* __restrict__ r, int c8, float* f) {
    if constexpr (KT == 1) {
        const int2 q = *reinterpret_cast<const int2*>(r + c8 * 8);
        const float d = __half2float(reinterpret_cast<const __half*>(r + 256)[c8 >> 2]);
        const char4 a = *reinterpret_cast<const char4*>(&q.x), b = *reinterpret_cast<const char4*>(&q.y);
        f[0] = a.x * d; f[1] = a.y * d; f[2] = a.z * d; f[3] = a.w * d;
        f[4] = b.x * d; f[5] = b.y * d; f[6] = b.z * d; f[7] = b.w * d;
    } else {
        // block c8 / 4; its values (c8 % 4) * 8 ..: the low nibbles of bytes 0..15 for the first 16, the high for the rest
        const int blk = c8 >> 2, sub = c8 & 3;
        const int2 q = *reinterpret_cast<const int2*>(r + blk * 16 + (sub & 1) * 8);
        const float d = __half2float(reinterpret_cast<const __half*>(r + 128)[blk]);
        const int sh = sub >= 2 ? 4 : 0;
        const uint32_t w[2] = {(uint32_t) q.x >> sh, (uint32_t) q.y >> sh};
#pragma unroll
        for (int i = 0; i < 8; ++i) f[i] = (float) ((int) ((w[i >> 2] >> (8 * (i & 3))) & 15) - 8) * d;
    }
}

// 8 features (c8 * 8 ..) of a row as halves
template <int KT>
__device__ __forceinline__ uint4 kv_load8(const void* __restrict__ c, int64_t row, int c8) {
    if constexpr (KT == 0) {
        return *reinterpret_cast<const uint4*>(reinterpret_cast<const __half*>(c) + row * 256 + c8 * 8);
    } else {
        float f[8];
        widen8f<KT>(reinterpret_cast<const unsigned char*>(c) + row * kv_row<KT>(), c8, f);
        __half2 h[4] = {__floats2half2_rn(f[0], f[1]), __floats2half2_rn(f[2], f[3]), __floats2half2_rn(f[4], f[5]), __floats2half2_rn(f[6], f[7])};
        return *reinterpret_cast<const uint4*>(h);
    }
}
// the same 8 as floats
template <int KT>
__device__ __forceinline__ void kv_load8f(const void* __restrict__ c, int64_t row, int c8, float* f) {
    if constexpr (KT == 0) {
        const uint4 u = kv_load8<0>(c, row, c8);
        const __half2* h = reinterpret_cast<const __half2*>(&u);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 v = __half22float2(h[i]);
            f[2 * i] = v.x;
            f[2 * i + 1] = v.y;
        }
    } else {
        widen8f<KT>(reinterpret_cast<const unsigned char*>(c) + row * kv_row<KT>(), c8, f);
    }
}
// features 2 * d2, 2 * d2 + 1 of a row
template <int KT>
__device__ __forceinline__ __half2 kv_load2(const void* __restrict__ c, int64_t row, int d2) {
    if constexpr (KT == 0) {
        return reinterpret_cast<const __half2*>(reinterpret_cast<const __half*>(c) + row * 256)[d2];
    } else if constexpr (KT == 1) {
        const unsigned char* r = reinterpret_cast<const unsigned char*>(c) + row * Q8ROW;
        const char2 q = *reinterpret_cast<const char2*>(r + 2 * d2);
        const float d = __half2float(reinterpret_cast<const __half*>(r + 256)[d2 >> 4]);
        return __floats2half2_rn(q.x * d, q.y * d);
    } else {
        const unsigned char* r = reinterpret_cast<const unsigned char*>(c) + row * Q4ROW;
        const int v = 2 * d2, blk = v >> 5, j = v & 31, sh = j >= 16 ? 4 : 0, byte = j & 15;
        const float d = __half2float(reinterpret_cast<const __half*>(r + 128)[blk]);
        const int a = (r[blk * 16 + byte] >> sh) & 15, b2 = (r[blk * 16 + byte + 1] >> sh) & 15;
        return __floats2half2_rn((a - 8) * d, (b2 - 8) * d);
    }
}

// ---- attention: decode rows (and short verify passes) - nextsycl's: a block per (row, key head, split of the keys),
// NW warps taking the split's keys in turn, each lane DL features of the G query heads that share the key head; the
// warps' (max, sum, weighted values) merged in shared memory, the splits by a second kernel
constexpr int NW = 4;
constexpr int64_t DECODE_ROWS = 16;   // rows up to this: the float32 split kernel (when the tensor cores do not take them)
// the pass a call takes: the tensor cores from 3 rows when a key head has at most 6 query heads (a verify pass of
// drafts shares each key tile over its rows, where the split kernel reads the cache once a row: at 128K keys, 0.91 ms a
// row on the 5070 against 2.0 ms for 3..16 rows; 1.77 against 4.85 on the 3060); else the float32 split kernel up to
// 16 rows, float32 tiles past that
__host__ __device__ inline bool use_mma(int64_t T, int64_t G) { return T > 2 && G <= 6; }

// keys a split: enough splits that the card has ~512 blocks whatever the context, at least 32 keys each
__host__ __device__ inline int64_t split_keys(int64_t T, int64_t kv, int64_t Hkv) {
    if (T > DECODE_ROWS) return kv;
    const int64_t want = (kv * Hkv * T + 511) / 512;
    const int64_t kb = (want + 31) / 32 * 32;
    return kb < 32 ? 32 : kb;
}
__host__ __device__ inline int64_t n_splits(int64_t T, int64_t kv, int64_t Hkv) {
    const int64_t kb = split_keys(T, kv, Hkv);
    return kv > kb ? (kv + kb - 1) / kb : 1;
}

template <int G, int KT>
__global__ void __launch_bounds__(NW * 32) attn_decode_k(const float* __restrict__ qp, int64_t qs, const void* __restrict__ kc,
                                                         const void* __restrict__ vc, int64_t Hq, int64_t Hkv, int64_t cap, int64_t p0,
                                                         int64_t ns, int64_t KB, float* __restrict__ out, float* __restrict__ part) {
    constexpr int D = 256, DL = D / 32;
    __shared__ float slm[NW * G * (D + 2)];
    const int lane = threadIdx.x & 31, wi = threadIdx.x >> 5;
    int64_t gi = blockIdx.x;
    const int64_t sp = gi % ns;
    gi /= ns;
    const int64_t hk = gi % Hkv, r = gi / Hkv;
    const int64_t kv_end = p0 + r + 1;
    const int64_t j0 = sp * (ns > 1 ? KB : kv_end);
    const int64_t j1 = min(kv_end, ns > 1 ? (sp + 1) * KB : kv_end);
    const float scale = rsqrtf((float) D);
    float qv[G][DL], acc[G][DL], m[G], l[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const float* qr = qp + r * qs + (hk * G + g) * D + lane * DL;
#pragma unroll
        for (int i = 0; i < DL; ++i) {
            qv[g][i] = qr[i] * scale;
            acc[g][i] = 0.0f;
        }
        m[g] = -INFINITY;
        l[g] = 0.0f;
    }
    static_assert(DL == 8, "a lane's features: one load of 8");
    for (int64_t j = j0 + wi; j < j1; j += NW) {
        float kf[DL], vf[DL];
        kv_load8f<KT>(kc, hk * cap + j, lane, kf);
        kv_load8f<KT>(vc, hk * cap + j, lane, vf);
#pragma unroll
        for (int g = 0; g < G; ++g) {
            float d = 0.0f;
#pragma unroll
            for (int i = 0; i < DL; ++i) d += qv[g][i] * kf[i];
            const float sc = warp_sum(d);
            const float mn = fmaxf(m[g], sc);
            const float c = expf(m[g] - mn), e = expf(sc - mn);
#pragma unroll
            for (int i = 0; i < DL; ++i) acc[g][i] = acc[g][i] * c + e * vf[i];
            l[g] = l[g] * c + e;
            m[g] = mn;
        }
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
        float* o = &slm[(wi * G + g) * (D + 2)];
#pragma unroll
        for (int i = 0; i < DL; ++i) o[lane * DL + i] = acc[g][i];
        if (lane == 0) {
            o[D] = m[g];
            o[D + 1] = l[g];
        }
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < G * D; idx += NW * 32) {
        const int g = idx / D, d = idx % D;
        float mx = -INFINITY;
        for (int k = 0; k < NW; ++k) mx = fmaxf(mx, slm[(k * G + g) * (D + 2) + D]);
        float L = 0.0f, A = 0.0f;
        if (mx != -INFINITY) {
            for (int k = 0; k < NW; ++k) {
                const float* o = &slm[(k * G + g) * (D + 2)];
                const float e = expf(o[D] - mx);
                L += o[D + 1] * e;
                A += o[d] * e;
            }
        }
        const int64_t hq = hk * G + g;
        if (ns == 1) {
            out[r * Hq * D + hq * D + d] = L > 0.0f ? A / L : 0.0f;
        } else {
            float* p = part + ((r * Hq + hq) * ns + sp) * (D + 2);
            p[d] = A;
            if (d == 0) {
                p[D] = mx;
                p[D + 1] = L;
            }
        }
    }
}

// the splits of each (row, query head) merged: a block of D threads
__global__ void attn_merge_k(const float* __restrict__ part, int64_t ns, float* __restrict__ out) {
    constexpr int D = 256;
    const int64_t rh = blockIdx.x;
    const int d = threadIdx.x;
    const float* p = part + rh * ns * (D + 2);
    float mx = -INFINITY;
    for (int64_t k = 0; k < ns; ++k) mx = fmaxf(mx, p[k * (D + 2) + D]);
    float L = 0.0f, A = 0.0f;
    if (mx != -INFINITY) {
        for (int64_t k = 0; k < ns; ++k) {
            const float e = expf(p[k * (D + 2) + D] - mx);
            L += p[k * (D + 2) + D + 1] * e;
            A += p[k * (D + 2) + d] * e;
        }
    }
    out[rh * D + d] = L > 0.0f ? A / L : 0.0f;
}

// ---- attention: prompt chunks - a block per (key head, tile of BR rows): the tile's BR x G query vectors against the
// keys in tiles of BK (both in shared memory, each key tile read once for the BR rows), an online softmax per query;
// thread d of the block owns output feature d of every query vector. The same sums as the decode kernel, in another
// order (float32 throughout, the cache's halves widened).
constexpr int BR = 8, BK = 32, TD = 256;
constexpr int KS2 = TD / 2 + 1;   // a key row's stride in half2: 129 words, so lanes on 32 keys hit 32 banks

template <int G>
constexpr size_t tile_smem() {
    return (size_t) BR * G * (TD / 2) * 4      // Q (half2)
           + (size_t) BK * KS2 * 4             // K (half2, padded)
           + (size_t) BK * TD * 2              // V (half)
           + (size_t) BR * G * BK * 4          // P
           + (size_t) BR * G * 4;              // the rescale factors
}

template <int G, int KT>
__global__ void __launch_bounds__(TD) attn_tile_k(const float* __restrict__ qp, int64_t qs, const void* __restrict__ kc,
                                                  const void* __restrict__ vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t cap,
                                                  int64_t p0, float* __restrict__ out) {
    constexpr int D = TD, NQ = BR * G, D2 = D / 2;
    extern __shared__ __align__(16) unsigned char smem[];
    __half2* Qs = reinterpret_cast<__half2*>(smem);                 // [NQ][D2]
    __half2* Ks = Qs + NQ * D2;                                     // [BK][KS2]
    __half* Vs = reinterpret_cast<__half*>(Ks + BK * KS2);          // [BK][D]
    float* Ps = reinterpret_cast<float*>(Vs + BK * D);              // [NQ][BK]
    float* Cs = Ps + NQ * BK;                                       // [NQ]
    const int tid = threadIdx.x, lane = tid & 31, wi = tid >> 5;    // 8 warps
    const int64_t hk = blockIdx.y, r0 = (int64_t) blockIdx.x * BR;
    const float scale = rsqrtf((float) D);
    // the tile's queries (scaled), as halves; rows past T zero
    for (int idx = tid; idx < NQ * D2; idx += TD) {
        const int q = idx / D2, d2 = idx % D2, rr = q / G, g = q % G;
        const int64_t r = r0 + rr;
        float2 v = make_float2(0.0f, 0.0f);
        if (r < T) {
            const float* src = qp + r * qs + (hk * G + g) * D + 2 * d2;
            v = make_float2(src[0] * scale, src[1] * scale);
        }
        Qs[idx] = __float22half2_rn(v);
    }
    float acc[NQ];
#pragma unroll
    for (int q = 0; q < NQ; ++q) acc[q] = 0.0f;
    // each warp's G queries: their running max and sum, kept by the warp (lane-uniform)
    float m[G], l[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        m[g] = -INFINITY;
        l[g] = 0.0f;
    }
    const int64_t rlast = min(T, r0 + BR) - 1;
    const int64_t kv_end = p0 + rlast + 1;   // the last row's keys: every row's are a prefix of them
    for (int64_t k0 = 0; k0 < kv_end; k0 += BK) {
        __syncthreads();   // the previous tile's P and V are used
        // the key and value tiles
        for (int idx = tid; idx < BK * D2; idx += TD) {
            const int j = idx / D2, d2 = idx % D2;
            const int64_t key = k0 + j;
            __half2 kv2 = __float2half2_rn(0.0f), vv2 = kv2;
            if (key < kv_end) {
                kv2 = kv_load2<KT>(kc, hk * cap + key, d2);
                vv2 = kv_load2<KT>(vc, hk * cap + key, d2);
            }
            Ks[j * KS2 + d2] = kv2;
            reinterpret_cast<__half2*>(Vs)[j * D2 + d2] = vv2;
        }
        __syncthreads();
        // scores: lane j's key against warp wi's G queries (query wi * G + g)
        {
            float s[G];
#pragma unroll
            for (int g = 0; g < G; ++g) s[g] = 0.0f;
            const __half2* kr = Ks + lane * KS2;
            for (int d2 = 0; d2 < D2; ++d2) {
                const float2 kf = __half22float2(kr[d2]);
#pragma unroll
                for (int g = 0; g < G; ++g) {
                    const float2 qf = __half22float2(Qs[(wi * G + g) * D2 + d2]);
                    s[g] += qf.x * kf.x + qf.y * kf.y;
                }
            }
            const int64_t key = k0 + lane;
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const int q = wi * G + g, rr = q / G;
                const bool ok = key <= p0 + r0 + rr && key < kv_end;
                const float sc = ok ? s[g] : -INFINITY;
                const float mx = warp_max(sc);
                const float mn = fmaxf(m[g], mx);
                float p = 0.0f, c = 1.0f;
                if (mn != -INFINITY) {
                    p = ok ? expf(sc - mn) : 0.0f;
                    c = expf(m[g] - mn);   // m[g] -inf: 0 (acc and l are 0 then)
                }
                const float ps = warp_sum(p);
                l[g] = l[g] * c + ps;
                m[g] = mn;
                Ps[q * BK + lane] = p;
                if (lane == 0) Cs[q] = c;
            }
        }
        __syncthreads();
        // the values: thread tid = feature d
        const int d = tid;
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            float a = acc[q] * Cs[q];
            const float* pr = Ps + q * BK;
#pragma unroll 8
            for (int j = 0; j < BK; ++j) a += pr[j] * __half2float(Vs[j * D + d]);
            acc[q] = a;
        }
    }
    // the sums: each warp's queries' l to shared memory, then every thread its features
    __syncthreads();
    if (lane == 0) {
#pragma unroll
        for (int g = 0; g < G; ++g) Cs[wi * G + g] = l[g];
    }
    __syncthreads();
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
        const int rr = q / G, g = q % G;
        const int64_t r = r0 + rr;
        if (r < T) {
            const float L = Cs[q];
            out[r * Hq * D + (hk * G + g) * D + tid] = L > 0.0f ? acc[q] / L : 0.0f;
        }
    }
}


// ---- attention: prompt chunks on the tensor cores (mma.sync m16n8k16, half in, float32 sums; FlashAttention-2's
// scheme) - a block per (16 rows, key head, split of the keys), a warp per query head of the key head's group, so a
// tile of BKT keys and their values is read into shared memory once for the G heads (the values' fragments by
// ldmatrix's transpose). The queries, keys
// and the softmax's weights are halves in the products (the decode kernel stays float32 throughout).
constexpr int MR = 16, BKT = 32, MD = 256;
constexpr int QSTR = MD + 8;     // a query, key or value row's stride in halves (16-byte rows; 8 rows hit 32 banks)
constexpr int64_t MMA_BLOCKS = 96;   // fewer blocks than this a pass: the keys are split

// two stages of key and value tiles (half: the tiles themselves; quantized: their raw rows, widened into one half pair)
template <int G, int KT>
constexpr size_t mma_smem() {
    return KT ? 2 * 2 * (size_t) BKT * kv_row<KT>() + 2 * (size_t) BKT * QSTR * 2 : 2 * 2 * (size_t) BKT * QSTR * 2;
}

// 16 bytes global -> shared, asynchronously (zeros when !pred: nothing read)
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool pred) {
    const unsigned a = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(a), "l"(gmem), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
// keys a split of the tensor-core pass (kv: the last row's keys)
__host__ __device__ inline int64_t mma_split_keys(int64_t T, int64_t kv, int64_t Hkv) {
    const int64_t b = (T + MR - 1) / MR * Hkv;
    if (b >= MMA_BLOCKS) return kv;
    int64_t ns = (MMA_BLOCKS + b - 1) / b;
    const int64_t most = (kv + 255) / 256;   // at least 256 keys a split
    ns = ns < most ? ns : most;
    if (ns <= 1) return kv;
    return ((kv + ns - 1) / ns + BKT - 1) / BKT * BKT;
}
__host__ __device__ inline int64_t mma_splits(int64_t T, int64_t kv, int64_t Hkv) {
    const int64_t kb = mma_split_keys(T, kv, Hkv);
    return kv > kb ? (kv + kb - 1) / kb : 1;
}

__device__ __forceinline__ void mma16816(float& c0, float& c1, float& c2, float& c3, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// the B fragments of two n-tiles from row-major [k][n] halves (the values: keys by features), transposed by ldmatrix
__device__ __forceinline__ void ldsm_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const __half* p) {
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}

template <int G, int KT>
__global__ void __launch_bounds__(G * 32, 1) attn_mma_k(const float* __restrict__ qp, int64_t qs, const void* __restrict__ kc,
                                                        const void* __restrict__ vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t cap,
                                                        int64_t p0, int64_t ns, int64_t KB, float* __restrict__ out, float* __restrict__ part) {
    // the queries in registers (each warp its head's 16 rows as the products' A fragments); the key and value tiles in
    // two stages, the next read (cp.async) while this one is multiplied. q8: the raw rows in the stages, widened to the
    // half tiles once they are in
    extern __shared__ __align__(16) unsigned char smem[];
    constexpr int TB = BKT * QSTR;                    // a half tile's halves
    __half* Kh = reinterpret_cast<__half*>(smem);     // [2][BKT][QSTR] (q8: [1] - the widened tile)
    constexpr bool QZ = KT != 0;                     // a quantized cache
    constexpr int64_t RAW = kv_row<KT>();
    __half* Vh = Kh + (QZ ? 1 : 2) * TB;
    unsigned char* raw = smem + (QZ ? 2 : 4) * TB * 2;   // quantized: [2 stages][K, V][BKT][RAW]
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, nth = G * 32;
    const int gr = lane >> 2, tq = lane & 3;          // the fragment's row group and column pair
    const int64_t r0 = (int64_t) blockIdx.x * MR, hk = blockIdx.y, sp = blockIdx.z;
    const int64_t rows = min((int64_t) MR, T - r0);
    const int64_t kv_end = p0 + r0 + rows;           // the last row's keys
    const int64_t j0 = sp * KB, j1 = min(kv_end, ns > 1 ? (sp + 1) * KB : kv_end);
    const float scale = rsqrtf((float) MD);
    // the tile at k0 into stage st (keys past j1 zero)
    auto load = [&](int st, int64_t k0) {
        if constexpr (!QZ) {
            __half* kd = Kh + st * TB;
            __half* vd = Vh + st * TB;
            for (int idx = tid; idx < BKT * (MD / 8); idx += nth) {
                const int j = idx / (MD / 8), c = idx % (MD / 8);
                const bool ok = k0 + j < j1;
                const int64_t row = hk * cap + (ok ? k0 + j : j0);
                cp_async16(kd + j * QSTR + c * 8, reinterpret_cast<const __half*>(kc) + row * MD + c * 8, ok);
                cp_async16(vd + j * QSTR + c * 8, reinterpret_cast<const __half*>(vc) + row * MD + c * 8, ok);
            }
        } else {
            unsigned char* kd = raw + (size_t) st * 2 * BKT * RAW;
            unsigned char* vd = kd + BKT * RAW;
            constexpr int C16 = (int) (RAW / 16);
            for (int idx = tid; idx < BKT * C16; idx += nth) {
                const int j = idx / C16, c = idx % C16;
                const bool ok = k0 + j < j1;
                const int64_t row = hk * cap + (ok ? k0 + j : j0);
                cp_async16(kd + j * RAW + c * 16, reinterpret_cast<const unsigned char*>(kc) + row * RAW + c * 16, ok);
                cp_async16(vd + j * RAW + c * 16, reinterpret_cast<const unsigned char*>(vc) + row * RAW + c * 16, ok);
            }
        }
        cp_async_commit();
    };
    if (j0 < j1) load(0, j0);
    // the queries: rows gr, gr + 8 of this warp's head, scaled, as halves (rows past T zero)
    uint32_t qa[MD / 16][4];
    {
        const float* qr_a = qp + (r0 + gr) * qs + (hk * G + w) * MD;
        const float* qr_b = qr_a + 8 * qs;
        const bool ra = gr < rows, rb = gr + 8 < rows;
#pragma unroll
        for (int kk = 0; kk < MD / 16; ++kk) {
            const int c0 = kk * 16 + tq * 2, c1 = c0 + 8;
            const float2 x0 = ra ? *reinterpret_cast<const float2*>(qr_a + c0) : make_float2(0.0f, 0.0f);
            const float2 x1 = rb ? *reinterpret_cast<const float2*>(qr_b + c0) : make_float2(0.0f, 0.0f);
            const float2 x2 = ra ? *reinterpret_cast<const float2*>(qr_a + c1) : make_float2(0.0f, 0.0f);
            const float2 x3 = rb ? *reinterpret_cast<const float2*>(qr_b + c1) : make_float2(0.0f, 0.0f);
            qa[kk][0] = pack_h2(x0.x * scale, x0.y * scale);
            qa[kk][1] = pack_h2(x1.x * scale, x1.y * scale);
            qa[kk][2] = pack_h2(x2.x * scale, x2.y * scale);
            qa[kk][3] = pack_h2(x3.x * scale, x3.y * scale);
        }
    }
    float o[MD / 8][4];
#pragma unroll
    for (int n = 0; n < MD / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    float m_a = -INFINITY, m_b = -INFINITY, l_a = 0.0f, l_b = 0.0f;   // rows gr and gr + 8 (l: this thread's part)
    int st = 0;
    for (int64_t k0 = j0; k0 < j1; k0 += BKT, st ^= 1) {
        // the next tile on its way while this one is used
        if (k0 + BKT < j1) {
            load(st ^ 1, k0 + BKT);
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();
        const __half* Ks;
        const __half* Vs;
        if constexpr (QZ) {
            // the raw rows widened: 8 features a thread a step, the scale of their block of 32
            const unsigned char* kr = raw + (size_t) st * 2 * BKT * RAW;
            const unsigned char* vr = kr + BKT * RAW;
            for (int idx = tid; idx < 2 * BKT * (MD / 8); idx += nth) {
                const int which = idx / (BKT * (MD / 8)), rem = idx % (BKT * (MD / 8)), j = rem / (MD / 8), c = rem % (MD / 8);
                float f[8];
                widen8f<KT>((which ? vr : kr) + j * RAW, c, f);
                __half2 h[4] = {__floats2half2_rn(f[0], f[1]), __floats2half2_rn(f[2], f[3]), __floats2half2_rn(f[4], f[5]), __floats2half2_rn(f[6], f[7])};
                *reinterpret_cast<uint4*>((which ? Vh : Kh) + j * QSTR + c * 8) = *reinterpret_cast<const uint4*>(h);
            }
            __syncthreads();
            Ks = Kh;
            Vs = Vh;
        } else {
            Ks = Kh + st * TB;
            Vs = Vh + st * TB;
        }
        // scores S [16][BKT] = Q . K^T
        float s[BKT / 8][4];
#pragma unroll
        for (int n = 0; n < BKT / 8; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
#pragma unroll
        for (int kk = 0; kk < MD / 16; ++kk) {
#pragma unroll
            for (int n = 0; n < BKT / 8; ++n) {
                const __half* kr = Ks + (n * 8 + gr) * QSTR + kk * 16 + tq * 2;
                mma16816(s[n][0], s[n][1], s[n][2], s[n][3], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], *reinterpret_cast<const uint32_t*>(kr),
                         *reinterpret_cast<const uint32_t*>(kr + 8));
            }
        }
        // the causal mask, the online softmax (a row's 4 threads: lanes 4 gr .. 4 gr + 3)
        const int64_t lim_a = p0 + r0 + gr, lim_b = p0 + r0 + gr + 8;
        float mx_a = -INFINITY, mx_b = -INFINITY;
#pragma unroll
        for (int n = 0; n < BKT / 8; ++n)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int64_t key = k0 + n * 8 + tq * 2 + e;
                if (key >= j1 || key > lim_a) s[n][e] = -INFINITY;
                if (key >= j1 || key > lim_b) s[n][2 + e] = -INFINITY;
                mx_a = fmaxf(mx_a, s[n][e]);
                mx_b = fmaxf(mx_b, s[n][2 + e]);
            }
        mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffffu, mx_a, 1));
        mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffffu, mx_a, 2));
        mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffffu, mx_b, 1));
        mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffffu, mx_b, 2));
        const float mn_a = fmaxf(m_a, mx_a), mn_b = fmaxf(m_b, mx_b);
        // (the hardware exponent: the products are halves already; the llm sources are built without fast math)
        const float c_a = mn_a == -INFINITY ? 1.0f : __expf(m_a - mn_a), c_b = mn_b == -INFINITY ? 1.0f : __expf(m_b - mn_b);
        float ps_a = 0.0f, ps_b = 0.0f;
#pragma unroll
        for (int n = 0; n < BKT / 8; ++n)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                s[n][e] = mn_a == -INFINITY ? 0.0f : __expf(s[n][e] - mn_a);
                s[n][2 + e] = mn_b == -INFINITY ? 0.0f : __expf(s[n][2 + e] - mn_b);
                ps_a += s[n][e];
                ps_b += s[n][2 + e];
            }
        l_a = l_a * c_a + ps_a;
        l_b = l_b * c_b + ps_b;
        m_a = mn_a;
        m_b = mn_b;
#pragma unroll
        for (int n = 0; n < MD / 8; ++n) {
            o[n][0] *= c_a; o[n][1] *= c_a;
            o[n][2] *= c_b; o[n][3] *= c_b;
        }
        // O += P . V: the scores' accumulators are the next product's A fragments (two n-tiles a k-step of 16 keys)
#pragma unroll
        for (int ks = 0; ks < BKT / 16; ++ks) {
            const uint32_t a0 = pack_h2(s[2 * ks][0], s[2 * ks][1]);
            const uint32_t a1 = pack_h2(s[2 * ks][2], s[2 * ks][3]);
            const uint32_t a2 = pack_h2(s[2 * ks + 1][0], s[2 * ks + 1][1]);
            const uint32_t a3 = pack_h2(s[2 * ks + 1][2], s[2 * ks + 1][3]);
            // lane l's row for ldmatrix: key ks*16 + (l & 7) (+8 for the second matrix), features of n-tile n (+1 for l >= 16)
            const __half* vrow = Vs + (ks * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * QSTR + (lane >> 4) * 8;
#pragma unroll
            for (int n = 0; n < MD / 8; n += 2) {
                uint32_t b0, b1, b2, b3;
                ldsm_x4_trans(b0, b1, b2, b3, vrow + n * 8);
                mma16816(o[n][0], o[n][1], o[n][2], o[n][3], a0, a1, a2, a3, b0, b1);
                mma16816(o[n + 1][0], o[n + 1][1], o[n + 1][2], o[n + 1][3], a0, a1, a2, a3, b2, b3);
            }
        }
        // this stage (and the widened tile) are read again only after every warp is done with them
        __syncthreads();
    }
    // a row's sum over its 4 threads
    l_a += __shfl_xor_sync(0xffffffffu, l_a, 1);
    l_a += __shfl_xor_sync(0xffffffffu, l_a, 2);
    l_b += __shfl_xor_sync(0xffffffffu, l_b, 1);
    l_b += __shfl_xor_sync(0xffffffffu, l_b, 2);
    const int64_t hq = hk * G + w;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int r = gr + 8 * half;
        if (r >= rows) continue;
        const float L = half ? l_b : l_a, M = half ? m_b : m_a;
        const int64_t row = r0 + r;
        if (ns == 1) {
            float* dst = out + row * Hq * MD + hq * MD;
            const float inv = L > 0.0f ? 1.0f / L : 0.0f;
#pragma unroll
            for (int n = 0; n < MD / 8; ++n)
                *reinterpret_cast<float2*>(dst + n * 8 + tq * 2) = make_float2(o[n][2 * half] * inv, o[n][2 * half + 1] * inv);
        } else {
            float* p = part + ((row * Hq + hq) * ns + sp) * (MD + 2);
#pragma unroll
            for (int n = 0; n < MD / 8; ++n) *reinterpret_cast<float2*>(p + n * 8 + tq * 2) = make_float2(o[n][2 * half], o[n][2 * half + 1]);
            if (tq == 0) {
                p[MD] = M;
                p[MD + 1] = L;
            }
        }
    }
}

// ---- the delta-rule scan (glm.cpp's kda_scan): NC value columns of one head a warp, each lane KPL of its 128 keys
template <int NC>
__global__ void __launch_bounds__(256) kda_scan_k(const float* __restrict__ qv, const float* __restrict__ kv, const float* __restrict__ vv,
                                                  const float* __restrict__ eg, const float* __restrict__ beta, float* __restrict__ S,
                                                  float* __restrict__ o, int64_t T, int64_t H, float* __restrict__ snap) {
    constexpr int KPL = 128 / 32;
    const int64_t c0 = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) / 32 * NC;
    if (c0 >= H * 128) return;
    const int64_t hd = c0 / 128;
    const int j0 = (int) (c0 % 128), lane = threadIdx.x & 31, i0 = lane * KPL;
    float* Sh = S + hd * 128 * 128;
    float col[NC][KPL];
    // the state value-column first (S [H][j][i]): a warp reads a column's 128 keys as one 512-byte sweep
#pragma unroll
    for (int n = 0; n < NC; ++n) {
        const float4 c = *reinterpret_cast<const float4*>(Sh + (j0 + n) * 128 + i0);
        col[n][0] = c.x, col[n][1] = c.y, col[n][2] = c.z, col[n][3] = c.w;
    }
    const float scale = rsqrtf(128.0f);
    for (int64_t t = 0; t < T; ++t) {
        const int64_t base = (t * H + hd) * 128;
        const float4 kq = *reinterpret_cast<const float4*>(kv + base + i0);
        const float4 gq = *reinterpret_cast<const float4*>(eg + base + i0);
        const float4 qq = *reinterpret_cast<const float4*>(qv + base + i0);
        const float kk[KPL] = {kq.x, kq.y, kq.z, kq.w}, gg[KPL] = {gq.x, gq.y, gq.z, gq.w}, qx[KPL] = {qq.x, qq.y, qq.z, qq.w};
        const float bt = beta[t * H + hd];
        float dot[NC], outv[NC];
#pragma unroll
        for (int n = 0; n < NC; ++n) {
            dot[n] = 0.0f;
#pragma unroll
            for (int i = 0; i < KPL; ++i) {
                col[n][i] *= gg[i];
                dot[n] += col[n][i] * kk[i];
            }
        }
#pragma unroll
        for (int n = 0; n < NC; ++n) dot[n] = warp_sum(dot[n]);
#pragma unroll
        for (int n = 0; n < NC; ++n) {
            const float delta = (vv[base + j0 + n] - dot[n]) * bt;
            outv[n] = 0.0f;
#pragma unroll
            for (int i = 0; i < KPL; ++i) {
                col[n][i] += kk[i] * delta;
                outv[n] += col[n][i] * qx[i];
            }
        }
#pragma unroll
        for (int n = 0; n < NC; ++n) outv[n] = warp_sum(outv[n]);
        if (lane < NC) {
            float mine = outv[0];
#pragma unroll
            for (int n = 1; n < NC; ++n) mine = lane == n ? outv[n] : mine;
            o[base + j0 + lane] = mine * scale;
        }
        if (snap && t + 1 < T) {
            float* Sn = snap + (t * H + hd) * 128 * 128;
#pragma unroll
            for (int n = 0; n < NC; ++n) *reinterpret_cast<float4*>(Sn + (j0 + n) * 128 + i0) = make_float4(col[n][0], col[n][1], col[n][2], col[n][3]);
        }
    }
#pragma unroll
    for (int n = 0; n < NC; ++n) *reinterpret_cast<float4*>(Sh + (j0 + n) * 128 + i0) = make_float4(col[n][0], col[n][1], col[n][2], col[n][3]);
}

// ---- the small passes
// pos3 (nullable): each row's (t, h, w) positions [3][T] - Qwen3.8's interleaved M-RoPE (llama.cpp's IMROPE with
// sections 11 / 11 / 10): pair k turns by position section k % 3; without it every pair by p0 + t
__global__ void qk_norm_rope_k(const float* src, int64_t ss, int64_t hs, float* dst, int64_t ds, int64_t H, int64_t D, const float* w,
                               float eps, int64_t n_rot, float theta, int64_t p0, const int32_t* pos3, int64_t T) {
    extern __shared__ float row[];   // the normed row: dst may be src
    const int l = threadIdx.x;
    const int64_t th = blockIdx.x, t = th / H, h = th % H;
    const float* x = src + t * ss + h * hs;
    float s = 0.0f;
    for (int64_t i = l; i < D; i += 32) s += x[i] * x[i];
    const float inv = rsqrtf(warp_sum(s) / (float) D + eps);
    for (int64_t i = l; i < D; i += 32) row[i] = x[i] * inv * w[i];
    __syncwarp();
    const int64_t half_rot = n_rot / 2;
    const float lin = (float) (p0 + t);
    float* y = dst + t * ds + h * D;
    for (int64_t i = l; i < D; i += 32) {
        const float v = row[i];
        if (i < n_rot) {
            // rotate-half over the first n_rot: pair (i, i + n_rot / 2)
            const int64_t k = i < half_rot ? i : i - half_rot;
            const float pos = pos3 ? (float) pos3[(k % 3) * T + t] : lin;
            const float a = pos * powf(theta, -2.0f * (float) k / (float) n_rot);
            const float c = cosf(a), sn = sinf(a);
            const float other = row[i < half_rot ? i + half_rot : i - half_rot];
            y[i] = i < half_rot ? v * c - other * sn : v * c + other * sn;
        } else {
            y[i] = v;
        }
    }
}

__global__ void kv_store_k(const float* k, int64_t ks, const float* v, int64_t vs, int64_t T, int64_t Hkv, int64_t D, int64_t cap, int64_t p0,
                           __half* ko, __half* vo) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * Hkv * D) return;
    const int64_t d = i % D, h = (i / D) % Hkv, t = i / (D * Hkv);
    const int64_t at = (h * cap + p0 + t) * D + d;
    ko[at] = __float2half(k[t * ks + h * D + d]);
    vo[at] = __float2half(v[t * vs + h * D + d]);
}

// q4_0 rows (D 256): a warp a (row, head), each lane 8 features, a block of 32 over 4 lanes - ggml's
// quantize_row_q4_0: the block's value of largest magnitude m (its sign kept), d = m / -8, q = min(15, (int) (x / d + 8.5));
// byte j of a block: value j low, value j + 16 high (lanes 0, 1 of the block write its 16 bytes)
__global__ void kv_store_q4_k(const float* k, int64_t ks, const float* v, int64_t vs, int64_t T, int64_t Hkv, int64_t cap, int64_t p0,
                              unsigned char* ko, unsigned char* vo) {
    const int64_t w = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (w >= T * Hkv) return;
    const int lane = threadIdx.x & 31, blk = lane >> 2, sub = lane & 3;
    const int64_t h = w % Hkv, t = w / Hkv;
    const int64_t row = h * cap + p0 + t;
    for (int which = 0; which < 2; ++which) {
        const float* src = (which ? v + t * vs : k + t * ks) + h * 256 + lane * 8;
        unsigned char* dst = (which ? vo : ko) + row * Q4ROW;
        const float4 a = *reinterpret_cast<const float4*>(src), b = *reinterpret_cast<const float4*>(src + 4);
        const float x[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        float m = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) m = fabsf(x[i]) > fabsf(m) ? x[i] : m;
#pragma unroll
        for (int o = 1; o <= 2; o <<= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, m, o);
            // (the larger magnitude; on a tie the lower lane's - ggml's first)
            const bool take = fabsf(other) > fabsf(m) || (fabsf(other) == fabsf(m) && (lane & o));
            m = take ? other : m;
        }
        const float d = m / -8.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        uint32_t lo = 0, hi = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t q = (uint32_t) min(15, (int) (x[i] * id + 8.5f));
            if (i < 4) lo |= q << (8 * i); else hi |= q << (8 * (i - 4));
        }
        // lanes 2, 3 of a block (values 16..31) hand their nibbles to lanes 0, 1 (values 0..15)
        const uint32_t lo2 = __shfl_down_sync(0xffffffffu, lo, 2), hi2 = __shfl_down_sync(0xffffffffu, hi, 2);
        if (sub < 2) {
            const int2 out = make_int2((int) (lo | (lo2 << 4)), (int) (hi | (hi2 << 4)));
            *reinterpret_cast<int2*>(dst + blk * 16 + sub * 8) = out;
        }
        if (sub == 0) reinterpret_cast<__half*>(dst + 128)[blk] = __float2half(d);
    }
}

// a 32-value Walsh-Hadamard transform, normalized (its own inverse), on every block of 32 of x [rows][n] (rows `stride`
// floats apart) in place - a warp a block, a value a lane
__global__ void fwht32_k(float* x, int64_t rows, int64_t n, int64_t stride) {
    const int64_t w = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int64_t per = n / 32;
    if (w >= rows * per) return;
    const int lane = threadIdx.x & 31;
    float* p = x + (w / per) * stride + (w % per) * 32;
    float v = p[lane];
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const float u = __shfl_xor_sync(0xffffffffu, v, o);
        v = (lane & o) ? u - v : v + u;
    }
    p[lane] = v * 0.17677669529663687f;   // 1 / sqrt(32)
}

// q8_0 rows (D 256): a warp a (row, head), each lane 8 features, a block of 32 over 4 lanes - ggml's
// quantize_f32_q8_0_block: d = amax / 127, q = round(x / d)
__global__ void kv_store_q8_k(const float* k, int64_t ks, const float* v, int64_t vs, int64_t T, int64_t Hkv, int64_t cap, int64_t p0,
                              unsigned char* ko, unsigned char* vo) {
    const int64_t w = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (w >= T * Hkv) return;
    const int lane = threadIdx.x & 31;
    const int64_t h = w % Hkv, t = w / Hkv;
    const int64_t row = h * cap + p0 + t;
    for (int which = 0; which < 2; ++which) {
        const float* src = (which ? v + t * vs : k + t * ks) + h * 256 + lane * 8;
        unsigned char* dst = (which ? vo : ko) + row * Q8ROW;
        const float4 a = *reinterpret_cast<const float4*>(src), b = *reinterpret_cast<const float4*>(src + 4);
        const float x[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(x[i]));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
        const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        char q[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) q[i] = (char) roundf(x[i] * id);
        *reinterpret_cast<int2*>(dst + lane * 8) = *reinterpret_cast<const int2*>(q);
        if ((lane & 3) == 0) reinterpret_cast<__half*>(dst + 256)[lane >> 2] = __float2half(d);
    }
}

__global__ void gate_mul_k(float* out, const float* q_full, int64_t gs, int64_t T, int64_t H, int64_t D) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * H * D) return;
    const int64_t d = i % D, h = (i / D) % H, t = i / (D * H);
    out[i] *= sigm(q_full[t * gs + h * 2 * D + D + d]);
}

__global__ void gdn_gates_k(const float* alpha, const float* dt, const float* a, float* eg, int64_t T, int64_t H, int64_t d) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * H * d) return;
    const int64_t h = (i / d) % H, t = i / (d * H);
    const float x = alpha[t * H + h] + dt[h];
    // ggml's softplus: x past 20 as it is
    const float sp = x > 20.0f ? x : log1pf(expf(x));
    eg[i] = expf(sp * a[h]);
}

__global__ void sigmoid_k(float* x, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = sigm(x[i]);
}

__global__ void expand_k(const float* src, int64_t ss, float* dst, int64_t T, int64_t Hk, int64_t Hv, int64_t d) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * Hv * d) return;
    const int64_t c = i % d, h = (i / d) % Hv, t = i / (d * Hv);
    dst[i] = src[t * ss + (h % Hk) * d + c];
}

__global__ void gdn_out_k(const float* o, const float* z, int64_t zs, const float* w, float* y, int64_t H, int64_t d, float eps) {
    const int l = threadIdx.x;
    const int64_t r = blockIdx.x, t = r / H, h = r % H;
    const float* x = o + r * d;
    float s = 0.0f;
    for (int64_t i = l; i < d; i += 32) s += x[i] * x[i];
    const float inv = rsqrtf(warp_sum(s) / (float) d + eps);
    for (int64_t i = l; i < d; i += 32) {
        const float zz = z[t * zs + h * d + i];
        y[r * d + i] = x[i] * inv * w[i] * (zz * sigm(zz));
    }
}

// a block's (max, lowest index at the max), every thread getting it
__device__ void block_argmax(float& best, int32_t& idx) {
    __shared__ float sv[32];
    __shared__ int32_t si[32];
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, o);
        const int32_t oi = __shfl_xor_sync(0xffffffffu, idx, o);
        if (ov > best || (ov == best && oi < idx)) {
            best = ov;
            idx = oi;
        }
    }
    const int w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    __syncthreads();
    if ((threadIdx.x & 31) == 0) {
        sv[w] = best;
        si[w] = idx;
    }
    __syncthreads();
    best = sv[0];
    idx = si[0];
    for (int i = 1; i < nw; ++i)
        if (sv[i] > best || (sv[i] == best && si[i] < idx)) {
            best = sv[i];
            idx = si[i];
        }
}

// rows of n: block per (row, part); a part's best to bv / bi (parts 1: straight to out)
__global__ void argmax_k(const float* x, int64_t n, int64_t per, float* bv, int32_t* bi, int32_t* out) {
    const int64_t r = blockIdx.y, p = blockIdx.x, lo = p * per, hi = min(n, lo + per);
    float best = -INFINITY;
    int32_t idx = INT32_MAX;
    for (int64_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        const float v = x[r * n + i];
        if (v > best) {   // a thread's indices rise: the first at its max is its lowest
            best = v;
            idx = (int32_t) i;
        }
    }
    block_argmax(best, idx);
    if (threadIdx.x == 0) {
        if (gridDim.x == 1) out[r] = idx == INT32_MAX ? 0 : idx;
        else {
            bv[p] = best;
            bi[p] = idx;
        }
    }
}
__global__ void argmax_final_k(const float* bv, const int32_t* bi, int parts, int32_t* out) {
    float best = -INFINITY;
    int32_t idx = INT32_MAX;
    for (int i = threadIdx.x; i < parts; i += blockDim.x)
        if (bv[i] > best || (bv[i] == best && bi[i] < idx)) {
            best = bv[i];
            idx = bi[i];
        }
    block_argmax(best, idx);
    if (threadIdx.x == 0) out[0] = idx == INT32_MAX ? 0 : idx;
}

// glm.cpp's conv + SiLU: the outputs a thread per (row, channel), then per channel the snapshots and the new state
__global__ void conv_silu_out_k(const float* x, const float* state, const float* w, float* out, int64_t T, int64_t D, int k) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * D) return;
    const int64_t t = i / D, c = i % D;
    float s = 0.0f;
    for (int j = 0; j < k; ++j) {
        const int64_t src = t - (k - 1) + j;   // inputs t-k+1 .. t
        s += (src >= 0 ? x[src * D + c] : state[(k - 1 + src) * D + c]) * w[c * k + j];
    }
    out[t * D + c] = s / (1.0f + expf(-s));
}
__global__ void conv_silu_state_k(const float* x, float* state, int64_t T, int64_t D, int k, float* snap) {
    const int64_t c = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D) return;
    auto in = [&](int64_t src) { return src >= 0 ? x[src * D + c] : state[(k - 1 + src) * D + c]; };
    if (snap)
        for (int64_t r = 0; r + 1 < T; ++r)
            for (int j = 0; j < k - 1; ++j) snap[(r * (k - 1) + j) * D + c] = in(r + 1 - (k - 1) + j);
    float last[8];
    for (int j = 0; j < k - 1; ++j) last[j] = in(T - (k - 1) + j);
    for (int j = 0; j < k - 1; ++j) state[j * D + c] = last[j];
}

__global__ void l2_norm_k(float* x, int64_t n, float eps) {
    __shared__ float part[4];
    float* xr = x + (int64_t) blockIdx.x * n;
    float s = 0.0f;
    for (int64_t c = threadIdx.x; c < n; c += blockDim.x) s += xr[c] * xr[c];
    s = warp_sum(s);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = s;
    __syncthreads();
    const float tot = part[0] + part[1] + part[2] + part[3];
    const float inv = 1.0f / sqrtf(tot + eps);
    for (int64_t c = threadIdx.x; c < n; c += blockDim.x) xr[c] *= inv;
}

__global__ void swiglu_k(const float* gate, const float* up, float* out, int64_t n, float limit) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = fminf(gate[i], limit);
    const float u = fminf(fmaxf(up[i], -limit), limit);
    out[i] = a / (1.0f + expf(-a)) * u;
}

inline unsigned blocks(int64_t n, int b = 256) { return (unsigned) ((n + b - 1) / b); }

template <int G, int KT>
int attn_launch(ns_gpu* g, const float* q, int64_t qs, const void* k, const void* v, int64_t T, int64_t Hq, int64_t Hkv, int64_t cap,
                int64_t p0, float* out, float* scratch) {
    if (use_mma(T, G)) {
        constexpr size_t sm = mma_smem<G <= 6 ? G : 1, KT>();
        static std::mutex mu;
        static bool raised[64] = {};
        {
            std::lock_guard<std::mutex> lk(mu);
            if (g->dev < 0 || g->dev >= 64) return ns_fail("ns_q35_attn: device ordinal");
            if (!raised[g->dev]) {
                NS_CU(cudaFuncSetAttribute(attn_mma_k<G <= 6 ? G : 1, KT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm));
                raised[g->dev] = true;
            }
        }
        const int64_t kv = p0 + T, ns = mma_splits(T, kv, Hkv), KB = mma_split_keys(T, kv, Hkv);
        const dim3 grid((unsigned) ((T + MR - 1) / MR), (unsigned) Hkv, (unsigned) ns);
        attn_mma_k<G <= 6 ? G : 1, KT><<<grid, G * 32, sm, g->q>>>(q, qs, k, v, T, Hq, Hkv, cap, p0, ns, KB, out, scratch);
        NS_LAUNCH_CHECK();
        if (ns > 1) {
            attn_merge_k<<<(unsigned) (T * Hq), 256, 0, g->q>>>(scratch, ns, out);
            NS_LAUNCH_CHECK();
        }
        return 0;
    }
    if (T > DECODE_ROWS) {
        constexpr size_t sm = tile_smem<G>();
        // past the default 48 KiB: raised once a device (the attribute is the current device's)
        static std::mutex mu;
        static bool raised[64] = {};
        {
            std::lock_guard<std::mutex> lk(mu);
            if (g->dev < 0 || g->dev >= 64) return ns_fail("ns_q35_attn: device ordinal");
            if (!raised[g->dev]) {
                NS_CU(cudaFuncSetAttribute(attn_tile_k<G, KT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm));
                raised[g->dev] = true;
            }
        }
        const dim3 grid((unsigned) ((T + BR - 1) / BR), (unsigned) Hkv);
        attn_tile_k<G, KT><<<grid, TD, sm, g->q>>>(q, qs, k, v, T, Hq, Hkv, cap, p0, out);
        NS_LAUNCH_CHECK();
        return 0;
    }
    const int64_t kv = p0 + T, ns = n_splits(T, kv, Hkv), KB = split_keys(T, kv, Hkv);
    attn_decode_k<G, KT><<<(unsigned) (T * Hkv * ns), NW * 32, 0, g->q>>>(q, qs, k, v, Hq, Hkv, cap, p0, ns, KB, out, scratch);
    NS_LAUNCH_CHECK();
    if (ns > 1) {
        attn_merge_k<<<(unsigned) (T * Hq), 256, 0, g->q>>>(scratch, ns, out);
        NS_LAUNCH_CHECK();
    }
    return 0;
}

template <int KT>
int attn_entry(ns_gpu* g, const float* q, int64_t qs, const void* k, const void* v, int64_t T, int64_t Hq, int64_t Hkv, int64_t D,
               int64_t cap, int64_t p0, float* out, float* scratch) {
    NS_CU(cudaSetDevice(g->dev));
    if (T <= 0) return 0;
    if (p0 + T > cap) return ns_fail("ns_q35_attn: past the cache's capacity");
    if (ns_q35_attn_scratch(T, Hq, Hkv, D, p0) > 0 && !scratch) return ns_fail("ns_q35_attn: no scratch");
    if (D != 256 || Hkv <= 0 || Hq % Hkv) return ns_fail("ns_q35_attn: head 256, the query heads a multiple of the key heads");
    switch (Hq / Hkv) {
        case 1: return attn_launch<1, KT>(g, q, qs, k, v, T, Hq, Hkv, cap, p0, out, scratch);
        case 2: return attn_launch<2, KT>(g, q, qs, k, v, T, Hq, Hkv, cap, p0, out, scratch);
        case 4: return attn_launch<4, KT>(g, q, qs, k, v, T, Hq, Hkv, cap, p0, out, scratch);
        case 6: return attn_launch<6, KT>(g, q, qs, k, v, T, Hq, Hkv, cap, p0, out, scratch);
        case 8: return attn_launch<8, KT>(g, q, qs, k, v, T, Hq, Hkv, cap, p0, out, scratch);
        default: return ns_fail("ns_q35_attn: 1, 2, 4, 6 or 8 query heads a key head");
    }
}

}  // namespace

extern "C" {

int ns_q35_qk_norm_rope(ns_gpu* g, const float* src, int64_t ss, int64_t hs, float* dst, int64_t ds, int64_t T, int64_t H, int64_t D,
                        const float* w, float eps, int64_t n_rot, float theta, int64_t p0) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (T <= 0) return 0;
    if (D % 32 != 0 || D > 4096 || n_rot > D || n_rot % 2) return ns_fail("ns_q35_qk_norm_rope: D a multiple of 32 (at most 4096), n_rot even and at most D");
    qk_norm_rope_k<<<(unsigned) (T * H), 32, (size_t) D * 4, g->q>>>(src, ss, hs, dst, ds, H, D, w, eps, n_rot, theta, p0, nullptr, T);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_q35_qk_norm_mrope(ns_gpu* g, const float* src, int64_t ss, int64_t hs, float* dst, int64_t ds, int64_t T, int64_t H, int64_t D,
                         const float* w, float eps, int64_t n_rot, float theta, const int32_t* pos3) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (T <= 0) return 0;
    if (D % 32 != 0 || D > 4096 || n_rot > D || n_rot % 2 || !pos3) return ns_fail("ns_q35_qk_norm_mrope: D a multiple of 32 (at most 4096), n_rot even and at most D, the positions given");
    qk_norm_rope_k<<<(unsigned) (T * H), 32, (size_t) D * 4, g->q>>>(src, ss, hs, dst, ds, H, D, w, eps, n_rot, theta, 0, pos3, T);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_q35_kv_store(ns_gpu* g, const float* k, int64_t ks, const float* v, int64_t vs, int64_t T, int64_t Hkv, int64_t D, int64_t cap, int64_t p0,
                    void* kc, void* vc) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (T <= 0) return 0;
    if (p0 + T > cap) return ns_fail("ns_q35_kv_store: past the cache's capacity");
    kv_store_k<<<blocks(T * Hkv * D), 256, 0, g->q>>>(k, ks, v, vs, T, Hkv, D, cap, p0, (__half*) kc, (__half*) vc);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_q35_kv_store_q8(ns_gpu* g, const float* k, int64_t ks, const float* v, int64_t vs, int64_t T, int64_t Hkv, int64_t D, int64_t cap, int64_t p0,
                       void* kc, void* vc) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (T <= 0) return 0;
    if (D != 256 || ks % 4 || vs % 4) return ns_fail("ns_q35_kv_store_q8: head 256, rows 16-byte aligned");
    if (p0 + T > cap) return ns_fail("ns_q35_kv_store_q8: past the cache's capacity");
    kv_store_q8_k<<<blocks(T * Hkv * 32), 256, 0, g->q>>>(k, ks, v, vs, T, Hkv, cap, p0, (unsigned char*) kc, (unsigned char*) vc);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_q35_kv_store_q4(ns_gpu* g, const float* k, int64_t ks, const float* v, int64_t vs, int64_t T, int64_t Hkv, int64_t D, int64_t cap, int64_t p0,
                       void* kc, void* vc) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (T <= 0) return 0;
    if (D != 256 || ks % 4 || vs % 4) return ns_fail("ns_q35_kv_store_q4: head 256, rows 16-byte aligned");
    if (p0 + T > cap) return ns_fail("ns_q35_kv_store_q4: past the cache's capacity");
    kv_store_q4_k<<<blocks(T * Hkv * 32), 256, 0, g->q>>>(k, ks, v, vs, T, Hkv, cap, p0, (unsigned char*) kc, (unsigned char*) vc);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_q35_fwht32(ns_gpu* g, float* x, int64_t rows, int64_t n, int64_t stride) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (rows <= 0) return 0;
    if (n % 32) return ns_fail("ns_q35_fwht32: n a multiple of 32");
    fwht32_k<<<blocks(rows * (n / 32) * 32), 256, 0, g->q>>>(x, rows, n, stride);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int64_t ns_q35_attn_scratch(int64_t T, int64_t Hq, int64_t Hkv, int64_t D, int64_t p0) {
    if (T <= 0 || Hkv <= 0) return 0;
    int64_t ns = 1;
    if (use_mma(T, Hq / Hkv)) ns = mma_splits(T, p0 + T, Hkv);   // the tensor-core pass splits a small grid's keys
    else if (T <= DECODE_ROWS) ns = n_splits(T, p0 + T, Hkv);
    return ns > 1 ? T * Hq * ns * (D + 2) : 0;
}


int ns_q35_attn(ns_gpu* g, const float* q, int64_t qs, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t D,
                int64_t cap, int64_t p0, float* out, float* scratch) {
    NS_TRY
    return attn_entry<0>(g, q, qs, kc, vc, T, Hq, Hkv, D, cap, p0, out, scratch);
    NS_CATCH
}

int ns_q35_attn_q8(ns_gpu* g, const float* q, int64_t qs, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t D,
                   int64_t cap, int64_t p0, float* out, float* scratch) {
    NS_TRY
    return attn_entry<1>(g, q, qs, kc, vc, T, Hq, Hkv, D, cap, p0, out, scratch);
    NS_CATCH
}

int ns_q35_attn_q4(ns_gpu* g, const float* q, int64_t qs, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t D,
                   int64_t cap, int64_t p0, float* out, float* scratch) {
    NS_TRY
    return attn_entry<2>(g, q, qs, kc, vc, T, Hq, Hkv, D, cap, p0, out, scratch);
    NS_CATCH
}

int ns_q35_gate_mul(ns_gpu* g, float* out, const float* q_full, int64_t gs, int64_t T, int64_t H, int64_t D) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (T <= 0) return 0;
    gate_mul_k<<<blocks(T * H * D), 256, 0, g->q>>>(out, q_full, gs, T, H, D);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_q35_gdn_gates(ns_gpu* g, const float* alpha, float* beta, const float* dt, const float* a, float* eg, int64_t T, int64_t H, int64_t d) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (T <= 0) return 0;
    gdn_gates_k<<<blocks(T * H * d), 256, 0, g->q>>>(alpha, dt, a, eg, T, H, d);
    NS_LAUNCH_CHECK();
    sigmoid_k<<<blocks(T * H), 256, 0, g->q>>>(beta, T * H);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_q35_expand(ns_gpu* g, const float* src, int64_t ss, float* dst, int64_t T, int64_t Hk, int64_t Hv, int64_t d) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (T <= 0) return 0;
    expand_k<<<blocks(T * Hv * d), 256, 0, g->q>>>(src, ss, dst, T, Hk, Hv, d);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_q35_gdn_out(ns_gpu* g, const float* o, const float* z, int64_t zs, const float* w, float* y, int64_t T, int64_t H, int64_t d, float eps) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (T <= 0) return 0;
    gdn_out_k<<<(unsigned) (T * H), 32, 0, g->q>>>(o, z, zs, w, y, H, d, eps);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_q35_argmax(ns_gpu* g, const float* x, int64_t rows, int64_t n, int32_t* out) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (rows <= 0) return 0;
    // one row over many blocks (a 248K vocabulary): each its slice's best, then one block the best of those
    if (rows == 1 && n >= 65536) {
        constexpr int PARTS = 256;
        static std::mutex mu;
        static std::unordered_map<ns_gpu*, void*> scratch;
        void* sp;
        {
            std::lock_guard<std::mutex> lk(mu);
            auto& e = scratch[g];
            if (!e) NS_CU(cudaMalloc(&e, PARTS * 8));
            sp = e;
        }
        float* bv = static_cast<float*>(sp);
        int32_t* bi = reinterpret_cast<int32_t*>(bv + PARTS);
        const int64_t per = (n + PARTS - 1) / PARTS;
        argmax_k<<<dim3(PARTS, 1), 256, 0, g->q>>>(x, n, per, bv, bi, out);
        NS_LAUNCH_CHECK();
        argmax_final_k<<<1, PARTS, 0, g->q>>>(bv, bi, PARTS, out);
        NS_LAUNCH_CHECK();
        return 0;
    }
    argmax_k<<<dim3(1, (unsigned) rows), 1024, 0, g->q>>>(x, n, n, nullptr, nullptr, out);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_q35_conv_silu(ns_gpu* g, const float* x, float* state, const float* w, float* out, int64_t T, int64_t D, int k, float* snap) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (T <= 0) return 0;
    if (k < 2 || k > 9) return ns_fail("ns_q35_conv_silu: a kernel of 2..9");
    // the outputs first (they read the old state), then (the stream is in order) the snapshots and the new state
    conv_silu_out_k<<<blocks(T * D), 256, 0, g->q>>>(x, state, w, out, T, D, k);
    NS_LAUNCH_CHECK();
    conv_silu_state_k<<<blocks(D), 256, 0, g->q>>>(x, state, T, D, k, snap);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_q35_l2_norm(ns_gpu* g, float* x, int64_t rows, int64_t n, float eps) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (rows <= 0) return 0;
    l2_norm_k<<<(unsigned) rows, 128, 0, g->q>>>(x, n, eps);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_q35_kda_scan(ns_gpu* g, const float* q, const float* k, const float* v, const float* gate, const float* beta, float* S, float* o,
                    int64_t T, int64_t H, int64_t d, float* snap) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (T <= 0) return 0;
    if (d != 128) return ns_fail("ns_q35_kda_scan: head size 128 only");
    // columns a warp (NS_KDA_COLS: 1, 2, 4 or 8)
    static const int env = [] { const char* e = std::getenv("NS_KDA_COLS"); return e ? std::atoi(e) : 0; }();
    const int nc = env ? env : 4;
    const int64_t warps = H * 128 / nc;
    const unsigned grid = (unsigned) ((warps * 32 + 255) / 256);
    switch (nc) {
        case 1: kda_scan_k<1><<<grid, 256, 0, g->q>>>(q, k, v, gate, beta, S, o, T, H, snap); break;
        case 2: kda_scan_k<2><<<grid, 256, 0, g->q>>>(q, k, v, gate, beta, S, o, T, H, snap); break;
        case 8: kda_scan_k<8><<<grid, 256, 0, g->q>>>(q, k, v, gate, beta, S, o, T, H, snap); break;
        default: kda_scan_k<4><<<grid, 256, 0, g->q>>>(q, k, v, gate, beta, S, o, T, H, snap); break;
    }
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_q35_swiglu(ns_gpu* g, const float* gate, const float* up, float* out, int64_t n, float limit) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (n <= 0) return 0;
    swiglu_k<<<blocks(n), 256, 0, g->q>>>(gate, up, out, n, limit);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

}  // extern "C"
