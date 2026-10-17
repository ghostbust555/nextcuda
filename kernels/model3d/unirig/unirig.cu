// unirig.cu: UniRig's kernels (unirig.h): norms, activations, attention (dense, and a decode step's over a slot pool
// of keys and values shared by the beams), gathers, the sparse convolutions' im2col and the skin head's per-head bone
// attention, all float32 as the reference runs; the linear layers are the shared diffusion kernels'. nextsycl's
// unirig.cpp in CUDA, the same names, arguments and arithmetic: its work-group reductions are block reductions (warp
// shuffles, then shared memory) and its local accessors shared memory, on the GPU's stream.
#include "ns.h"
#include "ns_internal.hpp"
#include "unirig.h"

#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>

namespace {

unsigned blocks(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }
#define ID() const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x

__device__ inline float gelu_erf(float v) {
    return 0.5f * v * (1.0f + erff(v * 0.7071067811865476f));
}

// the sum / max over the block (blockDim.x a multiple of 32, at most 1024), every thread gets it; red holds 32 floats
__device__ float block_sum(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < (int) (blockDim.x >> 5); ++i) t += red[i];
    __syncthreads();
    return t;
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

constexpr int LN_WG = 256, LN_PER = 32;

__global__ void ln_k(const float* x, int64_t x_row, int64_t C, float eps, const float* w, const float* b, float* out) {
    __shared__ float red[32];
    const int64_t r = blockIdx.x;
    const int l = threadIdx.x;
    const float* row = x + r * x_row;
    float v[LN_PER];
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < LN_PER; ++j) {
        const int64_t i = (int64_t) j * LN_WG + l;
        v[j] = i < C ? row[i] : 0.0f;
        s += v[j];
    }
    const float mean = block_sum(s, red) / (float) C;
    float s2 = 0.0f;
#pragma unroll
    for (int j = 0; j < LN_PER; ++j) {
        const int64_t i = (int64_t) j * LN_WG + l;
        const float d = i < C ? v[j] - mean : 0.0f;
        s2 += d * d;
    }
    const float inv = rsqrtf(block_sum(s2, red) / (float) C + eps);
    __syncthreads();
#pragma unroll
    for (int j = 0; j < LN_PER; ++j) {
        const int64_t i = (int64_t) j * LN_WG + l;
        if (i < C) {
            float y = (v[j] - mean) * inv;
            if (w) y = y * w[i];
            if (b) y = y + b[i];
            out[r * C + i] = y;
        }
    }
}

__global__ void head_norm_k(float* x, int64_t M, int64_t row, int64_t col, int64_t n_heads, int64_t D, float eps) {
    const int64_t t = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= M * n_heads) return;
    float* p = x + (t / n_heads) * row + col + (t % n_heads) * D;
    float v[64];
    float s = 0.0f;
    for (int64_t i = 0; i < D; ++i) {
        v[i] = p[i];
        s += v[i];
    }
    const float mean = s / (float) D;
    float s2 = 0.0f;
    for (int64_t i = 0; i < D; ++i) s2 += (v[i] - mean) * (v[i] - mean);
    const float inv = rsqrtf(s2 / (float) D + eps);
    for (int64_t i = 0; i < D; ++i) p[i] = (v[i] - mean) * inv;
}

__global__ void round_half_k(float* x, int64_t n) { ID(); if (i < n) x[i] = __half2float(__float2half_rn(x[i])); }

__global__ void act_k(float* x, int64_t n, int kind) {
    ID();
    if (i >= n) return;
    const float v = x[i];
    x[i] = kind == 0 ? gelu_erf(v) : fmaxf(v, 0.0f);
}

__global__ void add_k(float* x, const float* y, int64_t n) { ID(); if (i < n) x[i] = x[i] + y[i]; }

__global__ void affine_k(float* x, int64_t C, const float* scale, const float* shift, int64_t n) {
    ID();
    if (i >= n) return;
    const int64_t c = i % C;
    x[i] = x[i] * scale[c] + shift[c];
}

constexpr int AT_WG = 256, AT_MAXD = 1024;

// a block a (batch, head, query row) of this pass; the row's scores in the scratch
__global__ void attention_k(const float* q, int64_t q_row, int64_t q_head, const float* k, int64_t k_row, int64_t k_head, const float* v,
                            int64_t v_row, int64_t v_head, int64_t Sq, int64_t Skv, int64_t H, int64_t D, int64_t Dv, float scale, int causal,
                            float* scratch, float* out, int64_t out_row, int64_t g0) {
    __shared__ float qv[AT_MAXD];
    __shared__ float red[32];
    const int64_t gr = blockIdx.x, gi = g0 + gr;
    const int64_t i = gi % Sq, hh = (gi / Sq) % H, b = gi / (Sq * H);
    const int l = threadIdx.x;
    const float* qr = q + (b * Sq + i) * q_row + hh * q_head;
    for (int64_t d = l; d < D; d += AT_WG) qv[d] = qr[d];
    __syncthreads();
    const int64_t nk = causal ? i + 1 : Skv;
    float* sc = scratch + gr * Skv;
    float mx = -INFINITY;
    for (int64_t j = l; j < nk; j += AT_WG) {
        const float* kr = k + (b * Skv + j) * k_row + hh * k_head;
        float s = 0.0f;
        for (int64_t d = 0; d < D; ++d) s += qv[d] * kr[d];
        s *= scale;
        sc[j] = s;
        mx = fmaxf(mx, s);
    }
    mx = block_max(mx, red);
    float sum = 0.0f;
    for (int64_t j = l; j < nk; j += AT_WG) {
        const float e = expf(sc[j] - mx);
        sc[j] = e;
        sum += e;
    }
    sum = block_sum(sum, red);
    __threadfence_block();
    __syncthreads();
    const float inv = 1.0f / sum;
    for (int64_t d = l; d < Dv; d += AT_WG) {
        const float* vc = v + b * Skv * v_row + hh * v_head + d;
        float acc = 0.0f;
        for (int64_t j = 0; j < nk; ++j) acc += sc[j] * vc[j * v_row];
        out[(b * Sq + i) * out_row + hh * Dv + d] = acc * inv;
    }
}

constexpr int SL_WG = 256, SL_MAXL = 8192, SL_MAXD = 128;

__global__ void attn_slots_k(const float* q, const float* kpool, const float* vpool, const int32_t* slots, int64_t stride, int64_t len, int64_t H,
                             int64_t D, float scale, float* out) {
    __shared__ float sc[SL_MAXL];
    __shared__ float qv[SL_MAXD];
    __shared__ float red[32];
    const int64_t row = H * D;
    const int64_t gi = blockIdx.x, b = gi / H, hh = gi % H;
    const int l = threadIdx.x;
    for (int64_t d = l; d < D; d += SL_WG) qv[d] = q[b * row + hh * D + d];
    __syncthreads();
    const int32_t* st = slots + b * stride;
    float mx = -INFINITY;
    for (int64_t t = l; t < len; t += SL_WG) {
        const float* kr = kpool + (int64_t) st[t] * row + hh * D;
        float s = 0.0f;
        for (int64_t d = 0; d < D; ++d) s += qv[d] * kr[d];
        s *= scale;
        sc[t] = s;
        mx = fmaxf(mx, s);
    }
    mx = block_max(mx, red);
    float sum = 0.0f;
    for (int64_t t = l; t < len; t += SL_WG) {
        const float e = expf(sc[t] - mx);
        sc[t] = e;
        sum += e;
    }
    sum = block_sum(sum, red);
    __syncthreads();
    const float inv = 1.0f / sum;
    for (int64_t d = l; d < D; d += SL_WG) {
        float acc = 0.0f;
        for (int64_t t = 0; t < len; ++t) acc += sc[t] * vpool[(int64_t) st[t] * row + hh * D + d];
        out[b * row + hh * D + d] = acc * inv;
    }
}

__global__ void gather_k(const float* x, int64_t x_row, const int32_t* idx, int64_t C, float* out, int64_t n) {
    ID();
    if (i >= n) return;
    const int64_t r = i / C, c = i % C;
    out[i] = x[(int64_t) idx[r] * x_row + c];
}

__global__ void scatter_k(const float* x, const int32_t* idx, int64_t C, float* out, int64_t n) {
    ID();
    if (i >= n) return;
    const int64_t r = i / C, c = i % C;
    out[(int64_t) idx[r] * C + c] = x[i];
}

__global__ void im2col_k(const float* x, int64_t C, const int32_t* nbr, int64_t K, int64_t r0, float* out, int64_t n) {
    ID();
    if (i >= n) return;
    const int64_t c = i % C, kk = (i / C) % K, r = i / C / K;
    const int32_t j = nbr[(r0 + r) * K + kk];
    out[(r * K + kk) * C + c] = j < 0 ? 0.0f : x[(int64_t) j * C + c];
}

constexpr int BA_WG = 128, BA_MAXD = 1024;

__global__ void bone_attn_k(const float* q, const float* k, int64_t J, int64_t H, int64_t D, float scale, float* out) {
    __shared__ float qv[BA_MAXD];
    __shared__ float red[32];
    const int64_t row = H * D;
    const int64_t gi = blockIdx.x, n = gi / H, hh = gi % H;
    const int l = threadIdx.x;
    for (int64_t d = l; d < D; d += BA_WG) qv[d] = q[n * row + hh * D + d];
    __syncthreads();
    float mx = -INFINITY;
    for (int64_t j = l; j < J; j += BA_WG) {
        const float* kr = k + j * row + hh * D;
        float s = 0.0f;
        for (int64_t d = 0; d < D; ++d) s += qv[d] * kr[d];
        mx = fmaxf(mx, s * scale);
    }
    mx = block_max(mx, red);
    float sum = 0.0f;
    for (int64_t j = l; j < J; j += BA_WG) {
        const float* kr = k + j * row + hh * D;
        float s = 0.0f;
        for (int64_t d = 0; d < D; ++d) s += qv[d] * kr[d];
        const float e = expf(s * scale - mx);
        out[(n * J + j) * H + hh] = e;
        sum += e;
    }
    sum = block_sum(sum, red);
    __threadfence_block();
    __syncthreads();
    const float inv = 1.0f / sum;
    for (int64_t j = l; j < J; j += BA_WG) out[(n * J + j) * H + hh] *= inv;
}

__global__ void skin_in_k(const float* a, const float* vox, int64_t RJ, int64_t H, const float* aw, const float* ab, const float* ew,
                          const float* eb, const float* vw, const float* vb, float* out) {
    const int64_t r = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= RJ) return;
    float v[64];
    // the attention's heads, normalized
    float s = 0.0f;
    for (int64_t c = 0; c < H; ++c) {
        v[c] = a[r * H + c];
        s += v[c];
    }
    float mean = s / (float) H, s2 = 0.0f;
    for (int64_t c = 0; c < H; ++c) s2 += (v[c] - mean) * (v[c] - mean);
    float inv = rsqrtf(s2 / (float) H + 1e-5f);
    for (int64_t c = 0; c < H; ++c) out[r * 2 * H + c] = (v[c] - mean) * inv * aw[c] + ab[c];
    // the voxel prior through its embedding, normalized
    const float x = vox[r];
    s = 0.0f;
    for (int64_t c = 0; c < H; ++c) {
        v[c] = x * ew[c] + eb[c];
        s += v[c];
    }
    mean = s / (float) H;
    s2 = 0.0f;
    for (int64_t c = 0; c < H; ++c) s2 += (v[c] - mean) * (v[c] - mean);
    inv = rsqrtf(s2 / (float) H + 1e-5f);
    for (int64_t c = 0; c < H; ++c) out[r * 2 * H + H + c] = (v[c] - mean) * inv * vw[c] + vb[c];
}

}  // namespace

extern "C" {

int ns_model3d_unirig_ln(ns_gpu* g, const float* x, int64_t x_row, int64_t M, int64_t C, float eps, const float* w, const float* b, float* out) {
    if (C > (int64_t) LN_WG * LN_PER) return ns_fail("ns_model3d_unirig_ln: rows longer than 8192");
    if (M <= 0 || C <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    ln_k<<<(unsigned) M, LN_WG, 0, g->q>>>(x, x_row, C, eps, w, b, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_unirig_head_norm(ns_gpu* g, float* x, int64_t M, int64_t row, int64_t col, int64_t n_heads, int64_t D, float eps) {
    if (D <= 0 || D > 64) return ns_fail("ns_model3d_unirig_head_norm: 0 < D <= 64");
    if (M <= 0 || n_heads <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    head_norm_k<<<blocks(M * n_heads), 256, 0, g->q>>>(x, M, row, col, n_heads, D, eps);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_unirig_round_half(ns_gpu* g, float* x, int64_t n) {
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    round_half_k<<<blocks(n), 256, 0, g->q>>>(x, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_unirig_act(ns_gpu* g, float* x, int64_t n, int kind) {
    if (kind < 0 || kind > 1) return ns_fail("ns_model3d_unirig_act: kind 0 (gelu) or 1 (relu)");
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    act_k<<<blocks(n), 256, 0, g->q>>>(x, n, kind);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_unirig_add(ns_gpu* g, float* x, const float* y, int64_t n) {
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    add_k<<<blocks(n), 256, 0, g->q>>>(x, y, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_unirig_affine(ns_gpu* g, float* x, int64_t M, int64_t C, const float* scale, const float* shift) {
    if (M <= 0 || C <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    affine_k<<<blocks(M * C), 256, 0, g->q>>>(x, C, scale, shift, M * C);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_unirig_attention(ns_gpu* g, const float* q, int64_t q_row, int64_t q_head, const float* k, int64_t k_row, int64_t k_head,
                                const float* v, int64_t v_row, int64_t v_head, int64_t B, int64_t Sq, int64_t Skv, int64_t H, int64_t D,
                                int64_t Dv, float scale, int causal, float* scratch, int64_t scratch_n, float* out, int64_t out_row) {
    if (B <= 0 || Sq <= 0 || Skv <= 0 || H <= 0) return 0;
    if (D > AT_MAXD || D <= 0 || Dv <= 0) return ns_fail("ns_model3d_unirig_attention: 0 < D <= 1024, Dv > 0");
    if (scratch_n < Skv) return ns_fail("ns_model3d_unirig_attention: the scratch holds less than one row of scores");
    if (causal && Sq != Skv) return ns_fail("ns_model3d_unirig_attention: a causal mask needs Sq = Skv");
    NS_CU(cudaSetDevice(g->dev));
    const int64_t total = B * H * Sq;
    const int64_t per = std::max<int64_t>(1, std::min<int64_t>(total, scratch_n / Skv));
    for (int64_t g0 = 0; g0 < total; g0 += per) {
        const int64_t n = std::min(per, total - g0);
        attention_k<<<(unsigned) n, AT_WG, 0, g->q>>>(q, q_row, q_head, k, k_row, k_head, v, v_row, v_head, Sq, Skv, H, D, Dv, scale, causal,
                                                     scratch, out, out_row, g0);
        NS_LAUNCH_CHECK();
    }
    return 0;
}

int ns_model3d_unirig_attn_slots(ns_gpu* g, const float* q, const float* kpool, const float* vpool, const int32_t* slots, int64_t stride, int64_t len,
                                 int64_t B, int64_t H, int64_t D, float scale, float* out) {
    if (len <= 0 || B <= 0) return 0;
    if (len > SL_MAXL || D > SL_MAXD || D <= 0) return ns_fail("ns_model3d_unirig_attn_slots: len <= 8192, 0 < D <= 128");
    if (H <= 0) return 0;  // SYCL's empty range: nothing launched
    NS_CU(cudaSetDevice(g->dev));
    attn_slots_k<<<(unsigned) (B * H), SL_WG, 0, g->q>>>(q, kpool, vpool, slots, stride, len, H, D, scale, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_unirig_gather(ns_gpu* g, const float* x, int64_t x_row, const int32_t* idx, int64_t R, int64_t C, float* out) {
    if (R <= 0 || C <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    gather_k<<<blocks(R * C), 256, 0, g->q>>>(x, x_row, idx, C, out, R * C);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_unirig_scatter(ns_gpu* g, const float* x, const int32_t* idx, int64_t R, int64_t C, float* out) {
    if (R <= 0 || C <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    scatter_k<<<blocks(R * C), 256, 0, g->q>>>(x, idx, C, out, R * C);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_unirig_im2col(ns_gpu* g, const float* x, int64_t C, const int32_t* nbr, int64_t K, int64_t r0, int64_t R, float* out) {
    if (R <= 0 || C <= 0 || K <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    im2col_k<<<blocks(R * K * C), 256, 0, g->q>>>(x, C, nbr, K, r0, out, R * K * C);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_unirig_bone_attn(ns_gpu* g, const float* q, const float* k, int64_t R, int64_t J, int64_t H, int64_t D, float scale, float* out) {
    if (R <= 0 || J <= 0 || H <= 0) return 0;
    if (D <= 0 || D > BA_MAXD) return ns_fail("ns_model3d_unirig_bone_attn: 0 < D <= 1024");
    NS_CU(cudaSetDevice(g->dev));
    bone_attn_k<<<(unsigned) (R * H), BA_WG, 0, g->q>>>(q, k, J, H, D, scale, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_unirig_skin_in(ns_gpu* g, const float* a, const float* vox, int64_t RJ, int64_t H, const float* aw, const float* ab,
                              const float* ew, const float* eb, const float* vw, const float* vb, float* out) {
    if (H <= 0 || H > 64) return ns_fail("ns_model3d_unirig_skin_in: 0 < H <= 64");
    if (RJ <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    skin_in_k<<<blocks(RJ), 256, 0, g->q>>>(a, vox, RJ, H, aw, ab, ew, eb, vw, vb, out);
    NS_LAUNCH_CHECK();
    return 0;
}

}  // extern "C"
