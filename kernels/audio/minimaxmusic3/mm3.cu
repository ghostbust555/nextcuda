// mm3.cu: the MiniMax Music 3 engine's own kernels - what the shared diffusion kernels (nsd) do not have: the
// autoregressive stage's small-batch products and cached attention (two rows a frame: the prompt and its
// classifier-free twin), Qwen3's q / k norm with its rotary positions, the flow transformer's partial rotation, the
// codebooks' embeddings, the per-frame layer mix, the sampler's step, and the decoder's 1-D convolutions. nextsycl's
// mm3.cpp and conv.cpp in CUDA: the same names, arguments and arithmetic, so the engines above them are the same. The
// speech engines (Qwen3-TTS, Breeze TTS 2) and their 12 Hz codec run on these too.
//
// What differs is the width: a warp is 32 lanes where Xe2's sub-group is 16, so a lane takes D / 32 features of a head
// (attention) and the product's lanes still read 8 weights each (a warp covers 256 of K a step: K a multiple of 256,
// as before). The convolutions, oneDNN's on nextsycl, are here an unfolding of the signal and one cuBLAS GEMM a tile
// (float32, no TF32: the reference's numbers), the tiles sized so their scratch stays under SCRATCH_FLOATS.
#include "ns.h"
#include "ns_internal.hpp"
#include "mm3.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace {

constexpr int WARP = 32;

__device__ inline float bf16f(uint16_t b) { return __uint_as_float((uint32_t) b << 16); }

__device__ inline float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

unsigned blocks(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }

// ---- the decode's product: each warp R output rows, its lanes 8 features at a time across K
template <typename WT> __device__ inline void load8(const WT* p, float* o);
template <> __device__ inline void load8<__half>(const __half* p, float* o) {
    const uint4 u = *reinterpret_cast<const uint4*>(p);
    const __half2* h = reinterpret_cast<const __half2*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __half22float2(h[i]);
        o[2 * i] = f.x;
        o[2 * i + 1] = f.y;
    }
}
template <> __device__ inline void load8<int8_t>(const int8_t* p, float* o) {
    const uint2 u = *reinterpret_cast<const uint2*>(p);
    const int8_t* b = reinterpret_cast<const int8_t*>(&u);
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = (float) b[i];
}

constexpr int G_NW = 8, G_R = 2, G_MM = 8, G_V = 8;

template <typename WT>
__global__ void gemv_k(const float* x, int M, int64_t K, const WT* w, const float* scale, int64_t N, const float* bias, float* out, int64_t ldo,
                       bool acc) {
    const int lane = threadIdx.x & 31, wi = threadIdx.x >> 5;
    const int64_t n0 = ((int64_t) blockIdx.x * G_NW + wi) * G_R;
    float a[G_MM][G_R];
#pragma unroll
    for (int m = 0; m < G_MM; ++m)
#pragma unroll
        for (int r = 0; r < G_R; ++r) a[m][r] = 0.0f;
    for (int64_t k = (int64_t) lane * G_V; k < K; k += WARP * G_V) {
        float wf[G_R][G_V];
#pragma unroll
        for (int r = 0; r < G_R; ++r) {
            const int64_t n = n0 + r < N ? n0 + r : N - 1;
            load8<WT>(w + n * K + k, wf[r]);
        }
#pragma unroll
        for (int m = 0; m < G_MM; ++m) {
            if (m < M) {
                const float4 x0 = *reinterpret_cast<const float4*>(x + m * K + k);
                const float4 x1 = *reinterpret_cast<const float4*>(x + m * K + k + 4);
#pragma unroll
                for (int r = 0; r < G_R; ++r) {
                    a[m][r] += wf[r][0] * x0.x + wf[r][1] * x0.y + wf[r][2] * x0.z + wf[r][3] * x0.w + wf[r][4] * x1.x + wf[r][5] * x1.y +
                               wf[r][6] * x1.z + wf[r][7] * x1.w;
                }
            }
        }
    }
#pragma unroll
    for (int m = 0; m < G_MM; ++m) {
        if (m < M) {
#pragma unroll
            for (int r = 0; r < G_R; ++r) {
                const float s = warp_sum(a[m][r]);
                const int64_t n = n0 + r;
                if (lane == 0 && n < N) {
                    const float v = s * (scale ? scale[n] : 1.0f) + (bias ? bias[n] : 0.0f);
                    float* o = out + m * ldo + n;
                    *o = acc ? *o + v : v;
                }
            }
        }
    }
}

// ---- cached causal attention: a block per (row, key head, split of the keys), NW_A warps taking the split's keys in
// turn, each lane DL features of the G query heads that share the key head; the warps' running (max, sum, weighted
// values) merged in shared memory, the splits (decode only) by a second kernel
constexpr int64_t KB = 256;  // keys a split (decode)
constexpr int NW_A = 4;

int64_t splits(int64_t S, int64_t kv) { return S == 1 && kv > KB ? (kv + KB - 1) / KB : 1; }

template <int DL, int G>
__global__ void attn_k(const float* qp, int64_t qs, const __half* kc, const __half* vc, int64_t S, int64_t Hq, int64_t Hkv, int64_t T,
                       int64_t p0, int64_t ns, float* out, float* part) {
    constexpr int D = WARP * DL, WG = WARP * NW_A;
    extern __shared__ float slm[];
    const int lane = threadIdx.x & 31, sgi = threadIdx.x >> 5;
    const float scale = rsqrtf((float) D);
    int64_t gi = blockIdx.x;
    const int64_t sp = gi % ns;
    gi /= ns;
    const int64_t hk = gi % Hkv, r = gi / Hkv, b = r / S, s = r % S;
    const int64_t kv_end = p0 + s + 1;
    const int64_t j0 = sp * (ns > 1 ? KB : kv_end), j1 = min(kv_end, ns > 1 ? (sp + 1) * KB : kv_end);
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
    const __half* kb = kc + (b * Hkv + hk) * T * D + lane * DL;
    const __half* vb = vc + (b * Hkv + hk) * T * D + lane * DL;
    for (int64_t j = j0 + sgi; j < j1; j += NW_A) {
        float kv[DL], vv[DL];
#pragma unroll
        for (int i = 0; i < DL; ++i) {
            kv[i] = __half2float(kb[j * D + i]);
            vv[i] = __half2float(vb[j * D + i]);
        }
#pragma unroll
        for (int g = 0; g < G; ++g) {
            float d = 0.0f;
#pragma unroll
            for (int i = 0; i < DL; ++i) d += qv[g][i] * kv[i];
            const float sc = warp_sum(d);
            const float mn = fmaxf(m[g], sc);
            const float c = expf(m[g] - mn), e = expf(sc - mn);
#pragma unroll
            for (int i = 0; i < DL; ++i) acc[g][i] = acc[g][i] * c + e * vv[i];
            l[g] = l[g] * c + e;
            m[g] = mn;
        }
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
        float* o = &slm[(sgi * G + g) * (D + 2)];
#pragma unroll
        for (int i = 0; i < DL; ++i) o[lane * DL + i] = acc[g][i];
        if (lane == 0) {
            o[D] = m[g];
            o[D + 1] = l[g];
        }
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < G * D; idx += WG) {
        const int g = idx / D, d = idx % D;
        float mx = -INFINITY;
        for (int k = 0; k < NW_A; ++k) mx = fmaxf(mx, slm[(k * G + g) * (D + 2) + D]);
        float L = 0.0f, A = 0.0f;
        if (mx != -INFINITY) {
            for (int k = 0; k < NW_A; ++k) {
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

__global__ void attn_merge_k(const float* part, int64_t ns, int64_t D, float* out) {
    const int64_t rh = blockIdx.x, d = threadIdx.x;
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

template <int DL, int G>
int attn(ns_gpu* g, const float* q, int64_t qs, const __half* k, const __half* v, int64_t B, int64_t S, int64_t Hq, int64_t Hkv, int64_t T,
         int64_t p0, float* out, float* part) {
    constexpr int D = WARP * DL;
    const int64_t ns = splits(S, p0 + S);
    const size_t shm = (size_t) NW_A * G * (D + 2) * sizeof(float);
    attn_k<DL, G><<<(unsigned) (B * S * Hkv * ns), WARP * NW_A, shm, g->q>>>(q, qs, k, v, S, Hq, Hkv, T, p0, ns, out, part);
    NS_LAUNCH_CHECK();
    if (ns > 1) {
        attn_merge_k<<<(unsigned) (B * S * Hq), D, 0, g->q>>>(part, ns, D, out);
        NS_LAUNCH_CHECK();
    }
    return 0;
}

__global__ void kv_store_k(const float* k, const float* v, int64_t ks, int64_t S, int64_t Hkv, int64_t D, int64_t T, int64_t p0, __half* ko,
                           __half* vo, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t d = i % D, h = (i / D) % Hkv, r = i / (D * Hkv), b = r / S, s = r % S;
    const int64_t at = ((b * Hkv + h) * T + p0 + s) * D + d;
    ko[at] = __float2half_rn(k[r * ks + h * D + d]);
    vo[at] = __float2half_rn(v[r * ks + h * D + d]);
}

// one warp a (row, head) of 128: lane l holds features l + 32 j (j < 4), the pairs (i, 64 + i) are j and j + 2
__global__ void qk_norm_rope_k(float* x, int64_t xs, int64_t S, int64_t H, const float* weight, float eps, const float* inv_freq, int64_t p0) {
    const int l = threadIdx.x;
    const int64_t rh = blockIdx.x, r = rh / H, h = rh % H;
    float* p = x + r * xs + h * 128;
    float v[4];
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] = p[l + 32 * j];
        ss += v[j] * v[j];
    }
    const float inv = rsqrtf(warp_sum(ss) / 128.0f + eps);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = weight[l + 32 * j] * (v[j] * inv);
    const float pos = (float) (p0 + r % S);
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int i = l + 32 * j;
        const float a = pos * inv_freq[i], c = cosf(a), s = sinf(a);
        p[i] = v[j] * c - v[j + 2] * s;
        p[i + 64] = v[j + 2] * c + v[j] * s;
    }
}

__global__ void rope_partial_k(__half* xp, int64_t xs, int64_t S, int64_t H, int64_t D, int64_t hr, const float* inv_freq, int64_t n) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const int64_t i = id % hr, h = (id / hr) % H, r = id / (hr * H);
    __half* p = xp + r * xs + h * D;
    const float a = (float) (r % S) * inv_freq[i], c = cosf(a), s = sinf(a);
    const float x1 = __half2float(p[i]), x2 = __half2float(p[i + hr]);
    p[i] = __float2half_rn(x1 * c - x2 * s);
    p[i + hr] = __float2half_rn(x2 * c + x1 * s);
}

struct Ix { int32_t v[8]; };
struct Wt { float v[8]; };

__global__ void embed_k(const __half* t, int64_t D, Ix ix, int n, float scale, float* out, int64_t ldo, int64_t total) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= total) return;
    const int64_t j = id % D, c = id / D;
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += __half2float(t[(int64_t) ix.v[i] * D + j]);
    out[c * ldo + j] = s * scale;
}

__global__ void mix_k(const float* h, int n, int64_t D, Wt wt, float scale, float* out) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= D) return;
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += wt.v[i] * h[i * D + id];
    out[id] = scale * s;
}

__global__ void to_half_k(const float* x, __half* o, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = __float2half_rn(x[i]);
}
__global__ void bf16_to_half_k(const uint16_t* x, __half* o, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = __float2half_rn(bf16f(x[i]));
}
__global__ void bf16_to_float_k(const uint16_t* x, float* o, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = bf16f(x[i]);
}

constexpr int QWG = 256;
__global__ void quant_rows_k(const float* w, int64_t K, int8_t* q, float* scale) {
    __shared__ float red[QWG / 32];
    const int64_t r = blockIdx.x;
    const int l = threadIdx.x;
    const float* row = w + r * K;
    float mx = 0.0f;
    for (int64_t i = l; i < K; i += QWG) mx = fmaxf(mx, fabsf(row[i]));
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if ((l & 31) == 0) red[l >> 5] = mx;
    __syncthreads();
    mx = red[0];
    for (int i = 1; i < QWG / 32; ++i) mx = fmaxf(mx, red[i]);
    const float s = fmaxf(mx / 127.0f, 1e-30f);
    for (int64_t i = l; i < K; i += QWG) q[r * K + i] = (int8_t) fminf(fmaxf(rintf(row[i] / s), -127.0f), 127.0f);
    if (l == 0) scale[r] = s;
}

__global__ void dequant_rows_k(const int8_t* q, const float* scale, int64_t K, __half* o, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = __float2half_rn((float) q[i] * scale[i / K]);
}

__global__ void add_k(float* x, const float* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

__global__ void dit_in_k(const float* lat, const float* cond, int64_t L, int64_t C, int64_t Cc, float* out, int64_t n) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const int64_t W = 2 * C + Cc;
    const int64_t c = id % W, l = (id / W) % L, b = id / (W * L);
    float v = 0.0f;
    if (c < C) v = lat[l * C + c];
    else if (c >= 2 * C && b == 0) v = cond[l * Cc + c - 2 * C];
    out[id] = v;
}

__global__ void cfg_step_k(float* lat, const float* vc, const float* vu, int64_t n, float cfg, float dt) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) lat[i] += dt * (vu[i] + cfg * (vc[i] - vu[i]));
}

__global__ void blend_k(float* x, const float* p, const float* q, int64_t n, float a, float b) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = a * p[i] + b * q[i];
}

__global__ void nearest_rows_k(const float* x, int64_t C, int64_t Li, int64_t Lo, float sc, float* out, int64_t n) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const int64_t c = id % C, j = id / C;
    int64_t s;
    if (Lo == Li) s = j;
    else if (Lo == 2 * Li) s = j >> 1;
    else s = min((int64_t) floorf((float) j * sc), Li - 1);
    out[id] = x[c * Li + s];
}

__global__ void transpose_k(const float* x, int64_t R, int64_t C, float* out, int64_t n) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const int64_t r = id % R, c = id / R;
    out[id] = x[r * C + c];
}

__global__ void tanh_k(float* x, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = tanhf(x[i]);
}

// ---- the decoder's convolutions. A forward one: the signal unfolded for a tile of output positions, col
// [Ci * K, Lt] (col[(ci K + k) Lt + t] = x[ci, (t0 + t) stride + k dil - pad_l], zero outside), then
// out[:, t0 : t0 + Lt] = w [Co, Ci K] col - one GEMM. A transposed one: col [Co K, L] = w^T x (col[(co K + k) L + s] =
// sum_ci w[ci, co, k] x[ci, s]), then each output gathers what lands on it: out[co, t] = sum over k with
// (t + pad_l - k) = s stride, 0 <= s < L, of col[(co K + k) L + s]; output channels in tiles. Bias added by the
// fold / the gather.
constexpr int64_t SCRATCH_FLOATS = 32ll << 20;  // 128 MiB of scratch at most

struct Conv {
    cublasHandle_t cb = nullptr;
    float* scratch = nullptr;
    size_t cap = 0;  // floats
};

std::mutex mu;
std::map<ns_gpu*, std::unique_ptr<Conv>> convs;

int conv_ctx(ns_gpu* g, Conv** out) {
    std::lock_guard<std::mutex> l(mu);
    auto& c = convs[g];
    if (!c) {
        c = std::make_unique<Conv>();
        if (cublasCreate(&c->cb) != CUBLAS_STATUS_SUCCESS) return ns_fail("ns_audio_mm3_conv: cublasCreate failed");
        // float32 as the reference: no TF32 for these
        cublasSetMathMode(c->cb, CUBLAS_DEFAULT_MATH);
    }
    if (cublasSetStream(c->cb, g->q) != CUBLAS_STATUS_SUCCESS) return ns_fail("ns_audio_mm3_conv: cublasSetStream failed");
    *out = c.get();
    return 0;
}

int scratch(ns_gpu* g, Conv* c, size_t floats) {
    if (floats <= c->cap) return 0;
    NS_CU(cudaStreamSynchronize(g->q));  // nothing queued may still use the old one
    if (c->scratch) cudaFree(c->scratch);
    c->scratch = nullptr;
    c->cap = 0;
    cudaError_t e = cudaMalloc(&c->scratch, floats * sizeof(float));
    if (e != cudaSuccess) return ns_fail("ns_audio_mm3_conv: the device refused a scratch of " + std::to_string((floats * 4) >> 20) + " MiB");
    c->cap = floats;
    return 0;
}

__global__ void unfold_k(const float* x, int64_t Ci, int64_t L, int64_t K, int64_t stride, int64_t dil, int64_t pl, int64_t t0, int64_t Lt,
                         float* col, int64_t n) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const int64_t t = id % Lt, ck = id / Lt, k = ck % K, ci = ck / K;
    const int64_t s = (t0 + t) * stride + k * dil - pl;
    col[id] = s >= 0 && s < L ? x[ci * L + s] : 0.0f;
}

__global__ void bias_rows_k(float* out, const float* bias, int64_t Lo, int64_t t0, int64_t Lt, int64_t n) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const int64_t co = id / Lt, t = id % Lt;
    out[co * Lo + t0 + t] += bias[co];
}

__global__ void gather_k(const float* col, int64_t L, int64_t K, int64_t stride, int64_t pl, const float* bias, int64_t co0, int64_t Cot, int64_t Lo,
                         float* out, int64_t n) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const int64_t t = id % Lo, c = id / Lo;  // c: the tile's channel
    float a = bias ? bias[co0 + c] : 0.0f;
    for (int64_t k = 0; k < K; ++k) {
        const int64_t u = t + pl - k;
        if (u < 0 || u % stride != 0) continue;
        const int64_t s = u / stride;
        if (s < L) a += col[(c * K + k) * L + s];
    }
    out[(co0 + c) * Lo + t] = a;
}

#define BLAS(x) do { if ((x) != CUBLAS_STATUS_SUCCESS) return ns_fail(std::string("ns_audio_mm3_conv: ") + #x + " failed"); } while (0)

int conv_run(ns_gpu* g, bool tr, const float* x, int64_t B, int64_t Ci, int64_t L, const float* w, int64_t Co, int64_t K, const float* bias,
             int64_t stride, int64_t dil, int64_t pl, int64_t pr, float* out, int64_t Lo) {
    (void) pr;  // the output length says where the right padding ends
    NS_CU(cudaSetDevice(g->dev));
    Conv* c;
    if (conv_ctx(g, &c)) return -1;
    const float one = 1.0f, zero = 0.0f;
    if (!tr) {
        const int64_t CK = Ci * K;
        const int64_t Lt = std::max<int64_t>(1, std::min<int64_t>(Lo, SCRATCH_FLOATS / std::max<int64_t>(CK, 1)));
        if (scratch(g, c, (size_t) (CK * Lt))) return -1;
        for (int64_t b = 0; b < B; ++b) {
            const float* xb = x + b * Ci * L;
            float* ob = out + b * Co * Lo;
            for (int64_t t0 = 0; t0 < Lo; t0 += Lt) {
                const int64_t lt = std::min(Lt, Lo - t0), n = CK * lt;
                unfold_k<<<blocks(n), 256, 0, g->q>>>(xb, Ci, L, K, stride, dil, pl, t0, lt, c->scratch, n);
                NS_LAUNCH_CHECK();
                // column-major: out' (lt x Co, ld Lo) = col' (lt x CK, ld lt) w' (CK x Co, ld CK)
                BLAS(cublasSgemm(c->cb, CUBLAS_OP_N, CUBLAS_OP_N, (int) lt, (int) Co, (int) CK, &one, c->scratch, (int) lt, w, (int) CK, &zero,
                                 ob + t0, (int) Lo));
                if (bias) {
                    bias_rows_k<<<blocks(Co * lt), 256, 0, g->q>>>(ob, bias, Lo, t0, lt, Co * lt);
                    NS_LAUNCH_CHECK();
                }
            }
        }
        return 0;
    }
    // transposed: w [Ci, Co, K]
    const int64_t Cot = std::max<int64_t>(1, std::min<int64_t>(Co, SCRATCH_FLOATS / std::max<int64_t>(K * L, 1)));
    if (scratch(g, c, (size_t) (Cot * K * L))) return -1;
    for (int64_t b = 0; b < B; ++b) {
        const float* xb = x + b * Ci * L;
        float* ob = out + b * Co * Lo;
        for (int64_t co0 = 0; co0 < Co; co0 += Cot) {
            const int64_t ct = std::min(Cot, Co - co0);
            // column-major: col' (L x ct K, ld L) = x' (L x Ci, ld L) w'^T, w' the tile's columns of w ([Ci, Co K] row-major:
            // column-major (Co K x Ci) with ld Co K, from co0 K)
            BLAS(cublasSgemm(c->cb, CUBLAS_OP_N, CUBLAS_OP_T, (int) L, (int) (ct * K), (int) Ci, &one, xb, (int) L, w + co0 * K, (int) (Co * K), &zero,
                             c->scratch, (int) L));
            const int64_t n = ct * Lo;
            gather_k<<<blocks(n), 256, 0, g->q>>>(c->scratch, L, K, stride, pl, bias, co0, ct, Lo, ob, n);
            NS_LAUNCH_CHECK();
        }
    }
    return 0;
}

}  // namespace

extern "C" {

int ns_audio_mm3_gemv(ns_gpu* g, const float* x, int64_t M, int64_t K, const void* w, int wt, const float* scale, int64_t N,
                      const float* bias, float* out, int64_t ldo, int acc) {
    if (M <= 0 || N <= 0) return 0;
    if (M > 8 || K % 256 != 0) return ns_fail("ns_audio_mm3_gemv: at most 8 rows, K a multiple of 256");
    NS_CU(cudaSetDevice(g->dev));
    const unsigned grid = (unsigned) ((N + G_NW * G_R - 1) / (G_NW * G_R));
    if (wt == 0) gemv_k<__half><<<grid, WARP * G_NW, 0, g->q>>>(x, (int) M, K, (const __half*) w, scale, N, bias, out, ldo, acc != 0);
    else if (wt == 1) gemv_k<int8_t><<<grid, WARP * G_NW, 0, g->q>>>(x, (int) M, K, (const int8_t*) w, scale, N, bias, out, ldo, acc != 0);
    else return ns_fail("ns_audio_mm3_gemv: weights half (0) or int8 (1)");
    NS_LAUNCH_CHECK();
    return 0;
}

int64_t ns_audio_mm3_attn_scratch(int64_t B, int64_t S, int64_t Hq, int64_t D, int64_t p0) {
    const int64_t ns = splits(S, p0 + S);
    return ns > 1 ? B * S * Hq * ns * (D + 2) : 0;
}

int ns_audio_mm3_attn(ns_gpu* g, const float* q, int64_t qs, const void* kc, const void* vc, int64_t B, int64_t S, int64_t Hq,
                      int64_t Hkv, int64_t D, int64_t T, int64_t p0, float* out, float* part) {
    if (B <= 0 || S <= 0) return 0;
    if (p0 + S > T) return ns_fail("ns_audio_mm3_attn: past the cache's capacity");
    if (ns_audio_mm3_attn_scratch(B, S, Hq, D, p0) > 0 && !part) return ns_fail("ns_audio_mm3_attn: no scratch");
    NS_CU(cudaSetDevice(g->dev));
    const __half* k = (const __half*) kc;
    const __half* v = (const __half*) vc;
    if (D == 128 && Hq == 4 * Hkv) return attn<4, 4>(g, q, qs, k, v, B, S, Hq, Hkv, T, p0, out, part);
    if (D == 128 && Hq == 2 * Hkv) return attn<4, 2>(g, q, qs, k, v, B, S, Hq, Hkv, T, p0, out, part);
    if (D == 128 && Hq == 8 * Hkv) return attn<4, 8>(g, q, qs, k, v, B, S, Hq, Hkv, T, p0, out, part);
    if (D == 256 && Hq == Hkv) return attn<8, 1>(g, q, qs, k, v, B, S, Hq, Hkv, T, p0, out, part);
    return ns_fail("ns_audio_mm3_attn: D 128 with 8, 4 or 2 query heads a key head, or D 256 with one");
}

int ns_audio_mm3_kv_store(ns_gpu* g, const float* k, const float* v, int64_t ks, int64_t B, int64_t S, int64_t Hkv, int64_t D,
                          int64_t T, int64_t p0, void* kc, void* vc) {
    if (B <= 0 || S <= 0) return 0;
    if (p0 + S > T) return ns_fail("ns_audio_mm3_kv_store: past the cache's capacity");
    NS_CU(cudaSetDevice(g->dev));
    const int64_t n = B * S * Hkv * D;
    kv_store_k<<<blocks(n), 256, 0, g->q>>>(k, v, ks, S, Hkv, D, T, p0, (__half*) kc, (__half*) vc, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_mm3_qk_norm_rope(ns_gpu* g, float* x, int64_t xs, int64_t rows, int64_t S, int64_t H, const float* weight, float eps,
                              const float* inv_freq, int64_t p0) {
    if (rows <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    qk_norm_rope_k<<<(unsigned) (rows * H), WARP, 0, g->q>>>(x, xs, S, H, weight, eps, inv_freq, p0);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_mm3_rope_partial(ns_gpu* g, void* x, int64_t xs, int64_t rows, int64_t S, int64_t H, int64_t D, int64_t rot,
                              const float* inv_freq) {
    if (rows <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    const int64_t hr = rot / 2, n = rows * H * hr;
    rope_partial_k<<<blocks(n), 256, 0, g->q>>>((__half*) x, xs, S, H, D, hr, inv_freq, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_mm3_embed(ns_gpu* g, const void* table, int64_t D, const int32_t* idx, int n, float scale, float* out, int64_t ldo,
                       int copies) {
    if (n < 1 || n > 8) return ns_fail("ns_audio_mm3_embed: 1 to 8 rows");
    NS_CU(cudaSetDevice(g->dev));
    Ix ix{};
    for (int i = 0; i < n; ++i) ix.v[i] = idx[i];
    const int64_t total = D * copies;
    embed_k<<<blocks(total), 256, 0, g->q>>>((const __half*) table, D, ix, n, scale, out, ldo, total);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_mm3_mix(ns_gpu* g, const float* h, int n, int64_t D, const float* w, float scale, float* out) {
    if (n < 1 || n > 8) return ns_fail("ns_audio_mm3_mix: 1 to 8 rows");
    NS_CU(cudaSetDevice(g->dev));
    Wt wt{};
    for (int i = 0; i < n; ++i) wt.v[i] = w[i];
    mix_k<<<blocks(D), 256, 0, g->q>>>(h, n, D, wt, scale, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_mm3_to_half(ns_gpu* g, const float* x, void* out, int64_t n) {
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    to_half_k<<<blocks(n), 256, 0, g->q>>>(x, (__half*) out, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_mm3_bf16_to_half(ns_gpu* g, const uint16_t* x, void* out, int64_t n) {
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    bf16_to_half_k<<<blocks(n), 256, 0, g->q>>>(x, (__half*) out, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_mm3_bf16_to_float(ns_gpu* g, const uint16_t* x, float* out, int64_t n) {
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    bf16_to_float_k<<<blocks(n), 256, 0, g->q>>>(x, out, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_mm3_quant_rows(ns_gpu* g, const float* w, int64_t N, int64_t K, int8_t* q, float* scale) {
    if (N <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    quant_rows_k<<<(unsigned) N, QWG, 0, g->q>>>(w, K, q, scale);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_mm3_dequant_rows(ns_gpu* g, const int8_t* q, const float* scale, int64_t N, int64_t K, void* out) {
    if (N <= 0 || K <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    dequant_rows_k<<<blocks(N * K), 256, 0, g->q>>>(q, scale, K, (__half*) out, N * K);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_mm3_add(ns_gpu* g, float* x, const float* y, int64_t n) {
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    add_k<<<blocks(n), 256, 0, g->q>>>(x, y, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_mm3_dit_in(ns_gpu* g, const float* lat, const float* cond, int64_t L, int64_t C, int64_t Cc, float* out) {
    if (L <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    const int64_t n = 2 * L * (2 * C + Cc);
    dit_in_k<<<blocks(n), 256, 0, g->q>>>(lat, cond, L, C, Cc, out, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_mm3_cfg_step(ns_gpu* g, float* lat, const float* vc, const float* vu, int64_t n, float cfg, float dt) {
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    cfg_step_k<<<blocks(n), 256, 0, g->q>>>(lat, vc, vu, n, cfg, dt);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_mm3_blend(ns_gpu* g, float* x, const float* p, const float* q, int64_t n, float a, float b) {
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    blend_k<<<blocks(n), 256, 0, g->q>>>(x, p, q, n, a, b);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_mm3_nearest_rows(ns_gpu* g, const float* x, int64_t C, int64_t Li, int64_t Lo, float* out) {
    if (Lo <= 0 || Li <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    const int64_t n = Lo * C;
    nearest_rows_k<<<blocks(n), 256, 0, g->q>>>(x, C, Li, Lo, (float) Li / (float) Lo, out, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_mm3_transpose(ns_gpu* g, const float* x, int64_t R, int64_t C, float* out) {
    if (R <= 0 || C <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    transpose_k<<<blocks(R * C), 256, 0, g->q>>>(x, R, C, out, R * C);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_mm3_tanh(ns_gpu* g, float* x, int64_t n) {
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    tanh_k<<<blocks(n), 256, 0, g->q>>>(x, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_mm3_conv1d(ns_gpu* g, const float* x, int64_t B, int64_t Ci, int64_t L, const float* w, int64_t Co, int64_t K, const float* bias,
                        int64_t stride, int64_t dil, int64_t pad, float* out, int64_t Lo) {
    if (B <= 0 || Lo <= 0) return 0;
    return conv_run(g, false, x, B, Ci, L, w, Co, K, bias, stride, dil, pad, pad, out, Lo);
}

int ns_audio_mm3_conv_transpose1d(ns_gpu* g, const float* x, int64_t B, int64_t Ci, int64_t L, const float* w, int64_t Co, int64_t K,
                                  const float* bias, int64_t stride, int64_t pad, float* out, int64_t Lo) {
    if (B <= 0 || Lo <= 0) return 0;
    return conv_run(g, true, x, B, Ci, L, w, Co, K, bias, stride, 1, pad, pad, out, Lo);
}

int ns_audio_mm3_conv1d_lr(ns_gpu* g, int transposed, const float* x, int64_t B, int64_t Ci, int64_t L, const float* w, int64_t Co, int64_t K,
                           const float* bias, int64_t stride, int64_t dil, int64_t pad_l, int64_t pad_r, float* out, int64_t Lo) {
    if (B <= 0 || Lo <= 0) return 0;
    return conv_run(g, transposed != 0, x, B, Ci, L, w, Co, K, bias, stride, transposed ? 1 : dil, pad_l, pad_r, out, Lo);
}

}  // extern "C"
