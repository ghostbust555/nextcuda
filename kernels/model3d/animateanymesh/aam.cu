// aam.cu: AnimateAnyMesh's kernels (aam.h): the norms, activations, attention (dense and over a mesh's neighbours),
// farthest point sampling and the flow solver's steps, all float32 as the reference runs, on the GPU's own stream;
// the linear layers are the shared diffusion kernels' (nsd.h). nextsycl's aam.cpp in CUDA, the same names, arguments
// and arithmetic: its work-group reductions are shared-memory block reductions, its local accessors shared arrays,
// and where it turned floating-point contraction off the products and sums are the _rn intrinsics (never fused).
#include "ns.h"
#include "ns_internal.hpp"
#include "aam.h"

#include <cuda_fp16.h>
#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>

namespace {

unsigned blocks(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }

__device__ __forceinline__ float gelu_erf(float v) {
    return 0.5f * v * (1.0f + erff(v * 0.7071067811865476f));
}

// reductions over the block, every thread given the result (blockDim.x a multiple of 32, at most 1024); red: 32 slots
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

__device__ int block_min_int(int v, int* red) {
    for (int o = 16; o > 0; o >>= 1) v = min(v, __shfl_xor_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    int t = red[0];
    for (int i = 1; i < (int) (blockDim.x >> 5); ++i) t = min(t, red[i]);
    __syncthreads();
    return t;
}

constexpr int LN_WG = 256, LN_PER = 32;  // a row of up to 8,192 values held in registers

__global__ void __launch_bounds__(LN_WG) ln_k(const float* x, int64_t C, float eps, int64_t S, int64_t T, const float* w0, const float* b0,
                                              const float* scale0, const float* shift0, const float* w1, const float* b1,
                                              const float* scale1, const float* shift1, float* out) {
    __shared__ float red[32];
    const int64_t r = blockIdx.x;
    const int l = threadIdx.x;
    const bool s1 = (r % S) >= T;
    const float* w = s1 ? w1 : w0;
    const float* b = s1 ? b1 : b0;
    const float* sc = s1 ? scale1 : scale0;
    const float* sh = s1 ? shift1 : shift0;
    const float* row = x + r * C;
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
    __syncthreads();  // every lane has read its row before any writes it (out may be x)
#pragma unroll
    for (int j = 0; j < LN_PER; ++j) {
        const int64_t i = (int64_t) j * LN_WG + l;
        if (i < C) {
            float y = (v[j] - mean) * inv;
            if (w) y = y * w[i];
            if (b) y = y + b[i];
            if (sc) y = y * (1.0f + sc[i]);
            if (sh) y = y + sh[i];
            out[r * C + i] = y;
        }
    }
}

__global__ void gated_add_k(float* x, const float* y, int64_t C, int64_t S, int64_t T, const float* gate0, const float* gate1, int64_t n) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const int64_t r = id / C, c = id % C;
    const float* gt = (r % S) >= T ? gate1 : gate0;
    const float add = gt ? gt[c] * y[r * C + c] : y[r * C + c];
    x[r * C + c] = x[r * C + c] + add;
}

__global__ void act_k(float* x, int64_t n, int kind) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    float y;
    if (kind == 0) y = gelu_erf(v);
    else if (kind == 1) y = v / (1.0f + expf(-v));
    else y = v / (1.0f + expf(-1.702f * v));
    x[i] = y;
}

__global__ void geglu_k(const float* x, int64_t C, float* out, int64_t n) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const int64_t r = id / C, c = id % C;
    out[r * C + c] = x[r * 2 * C + c] * gelu_erf(x[r * 2 * C + C + c]);
}

constexpr int ATT_WG = 256, ATT_MAXD = 1024;

__global__ void __launch_bounds__(ATT_WG) attention_k(const float* q, int64_t q_row, int64_t q_head, const float* k, int64_t k_row,
                                                      int64_t k_head, const float* v, int64_t v_row, int64_t v_head, int64_t Sq, int64_t Skv,
                                                      int64_t H, int64_t D, int64_t Dv, float scale, int causal, float* scratch, float* out,
                                                      int64_t out_row, int64_t g0) {
    __shared__ float qv[ATT_MAXD];
    __shared__ float red[32];
    const int64_t gr = blockIdx.x, gi = g0 + gr;
    const int64_t i = gi % Sq, hh = (gi / Sq) % H, b = gi / (Sq * H);
    const int l = threadIdx.x;
    const float* qr = q + (b * Sq + i) * q_row + hh * q_head;
    for (int64_t d = l; d < D; d += ATT_WG) qv[d] = qr[d];
    __syncthreads();
    const int64_t nk = causal ? i + 1 : Skv;
    float* sc = scratch + gr * Skv;
    float mx = -INFINITY;
    for (int64_t j = l; j < nk; j += ATT_WG) {
        const float* kr = k + (b * Skv + j) * k_row + hh * k_head;
        float s = 0.0f;
        for (int64_t d = 0; d < D; ++d) s += qv[d] * kr[d];
        s *= scale;
        sc[j] = s;
        mx = fmaxf(mx, s);
    }
    mx = block_max(mx, red);
    float sum = 0.0f;
    for (int64_t j = l; j < nk; j += ATT_WG) {
        const float e = expf(sc[j] - mx);
        sc[j] = e;
        sum += e;
    }
    sum = block_sum(sum, red);
    // every lane's weights (global memory) seen by every lane
    __threadfence_block();
    __syncthreads();
    const float inv = 1.0f / sum;
    for (int64_t d = l; d < Dv; d += ATT_WG) {
        const float* vc = v + b * Skv * v_row + hh * v_head + d;
        float acc = 0.0f;
        for (int64_t j = 0; j < nk; ++j) acc += sc[j] * vc[j * v_row];
        out[(b * Sq + i) * out_row + hh * Dv + d] = acc * inv;
    }
}

constexpr int CSR_WG = 128, CSR_MAXN = 1024;

__global__ void __launch_bounds__(CSR_WG) csr_attention_k(const float* q, const float* k, const float* v, int64_t D, const int32_t* rowptr,
                                                          const int32_t* cols, float scale, float* out) {
    __shared__ float sc[CSR_MAXN];
    __shared__ float red[32];
    const int64_t i = blockIdx.x;
    const int l = threadIdx.x;
    const int64_t a = rowptr[i];
    const int64_t n = min((int64_t) rowptr[i + 1] - a, (int64_t) CSR_MAXN);
    float mx = -INFINITY;
    for (int64_t e = 0; e < n; ++e) {
        const float* kr = k + (int64_t) cols[a + e] * D;
        float s = 0.0f;
        for (int64_t d = l; d < D; d += CSR_WG) s += q[i * D + d] * kr[d];
        s = block_sum(s, red) * scale;
        if (l == 0) sc[e] = s;
        mx = fmaxf(mx, s);
    }
    __syncthreads();
    float sum = 0.0f;
    for (int64_t e = 0; e < n; ++e) sum += expf(sc[e] - mx);
    for (int64_t d = l; d < D; d += CSR_WG) {
        float acc = 0.0f;
        for (int64_t e = 0; e < n; ++e) acc += expf(sc[e] - mx) * v[(int64_t) cols[a + e] * D + d];
        out[i * D + d] = n > 0 ? acc / sum : 0.0f;
    }
}

constexpr int FPS_WG = 1024, FPS_MAXD = 1024;

// one block; the squared distances summed feature by feature with neither fusion nor reordering (_rn intrinsics)
__global__ void __launch_bounds__(FPS_WG) fps_k(const float* x, int64_t N, int64_t D, int64_t K, float* dist, int32_t* idx) {
    __shared__ float sel[FPS_MAXD];
    __shared__ float redf[32];
    __shared__ int redi[32];
    const int l = threadIdx.x;
    for (int64_t p = l; p < N; p += FPS_WG) dist[p] = FLT_MAX;
    if (l == 0) idx[0] = 0;
    int64_t cur = 0;
    for (int64_t kk = 1; kk < K; ++kk) {
        for (int64_t d = l; d < D; d += FPS_WG) sel[d] = x[cur * D + d];
        __syncthreads();
        float best = -1.0f;
        int bi = INT_MAX;
        for (int64_t p = l; p < N; p += FPS_WG) {
            const float* xp = x + p * D;
            float s = 0.0f;
            for (int64_t d = 0; d < D; ++d) {
                const float diff = __fsub_rn(sel[d], xp[d]);
                const float sq = __fmul_rn(diff, diff);
                s = __fadd_rn(s, sq);
            }
            const float dd = fminf(s, dist[p]);
            dist[p] = dd;
            if (dd > best) {
                best = dd;
                bi = (int) p;
            }
        }
        const float m = block_max(best, redf);
        const int c = block_min_int(best == m ? bi : INT_MAX, redi);
        cur = c;
        if (l == 0) idx[kk] = c;
        __syncthreads();
    }
}

__global__ void gather_k(const float* x, const int32_t* idx, int64_t C, float* out, int64_t n) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const int64_t r = id / C, c = id % C;
    out[r * C + c] = x[(int64_t) idx[r] * C + c];
}

__global__ void flow_in_k(const float* f0, const float* y, int64_t K, int64_t C, int64_t F, float* out, int64_t n) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const int64_t r = id / C, c = id % C, kk = r % K;
    out[r * C + c] = c < F ? f0[kk * F + c] : y[kk * C + c];
}

__global__ void cfg_k(const float* flow, int64_t n, float scale, float* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float c = flow[i], u = flow[n + i];
    const float d = __fsub_rn(c, u);
    const float s = __fmul_rn(scale, d);
    out[i] = __fadd_rn(u, s);
}

__global__ void axpy_k(const float* y, const float* f, float a, int64_t n, float* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float s = __fmul_rn(f[i], a);
    out[i] = __fadd_rn(y[i], s);
}

}  // namespace

extern "C" {

int ns_model3d_aam_ln(ns_gpu* g, const float* x, int64_t M, int64_t C, float eps, int64_t S, int64_t T,
                      const float* w0, const float* b0, const float* scale0, const float* shift0,
                      const float* w1, const float* b1, const float* scale1, const float* shift1, float* out) {
    if (C > (int64_t) LN_WG * LN_PER) return ns_fail("ns_model3d_aam_ln: rows longer than 8192");
    if (S <= 0 || T <= 0 || T > S) return ns_fail("ns_model3d_aam_ln: 0 < T <= S");
    if (M <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    ln_k<<<(unsigned) M, LN_WG, 0, g->q>>>(x, C, eps, S, T, w0, b0, scale0, shift0, w1, b1, scale1, shift1, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_aam_gated_add(ns_gpu* g, float* x, const float* y, int64_t M, int64_t C, int64_t S, int64_t T, const float* gate0,
                             const float* gate1) {
    if (S <= 0 || T <= 0 || T > S) return ns_fail("ns_model3d_aam_gated_add: 0 < T <= S");
    if (M <= 0 || C <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    gated_add_k<<<blocks(M * C), 256, 0, g->q>>>(x, y, C, S, T, gate0, gate1, M * C);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_aam_act(ns_gpu* g, float* x, int64_t n, int kind) {
    if (kind < 0 || kind > 2) return ns_fail("ns_model3d_aam_act: kind 0 (gelu), 1 (silu) or 2 (quick gelu)");
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    act_k<<<blocks(n), 256, 0, g->q>>>(x, n, kind);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_aam_geglu(ns_gpu* g, const float* x, int64_t M, int64_t C, float* out) {
    if (M <= 0 || C <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    geglu_k<<<blocks(M * C), 256, 0, g->q>>>(x, C, out, M * C);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_aam_attention(ns_gpu* g, const float* q, int64_t q_row, int64_t q_head, const float* k, int64_t k_row, int64_t k_head,
                             const float* v, int64_t v_row, int64_t v_head, int64_t B, int64_t Sq, int64_t Skv, int64_t H, int64_t D,
                             int64_t Dv, float scale, int causal, float* scratch, int64_t scratch_n, float* out, int64_t out_row) {
    if (B <= 0 || Sq <= 0 || Skv <= 0 || H <= 0) return 0;
    if (D > ATT_MAXD || D <= 0 || Dv <= 0) return ns_fail("ns_model3d_aam_attention: 0 < D <= 1024, Dv > 0");
    if (scratch_n < Skv) return ns_fail("ns_model3d_aam_attention: the scratch holds less than one row of scores");
    if (causal && Sq != Skv) return ns_fail("ns_model3d_aam_attention: a causal mask needs Sq = Skv");
    NS_CU(cudaSetDevice(g->dev));
    const int64_t total = B * H * Sq;
    const int64_t per = std::max<int64_t>(1, std::min<int64_t>(total, scratch_n / Skv));
    for (int64_t g0 = 0; g0 < total; g0 += per) {
        const int64_t n = std::min(per, total - g0);
        attention_k<<<(unsigned) n, ATT_WG, 0, g->q>>>(q, q_row, q_head, k, k_row, k_head, v, v_row, v_head, Sq, Skv, H, D, Dv, scale, causal,
                                                       scratch, out, out_row, g0);
        NS_LAUNCH_CHECK();
    }
    return 0;
}

int ns_model3d_aam_csr_attention(ns_gpu* g, const float* q, const float* k, const float* v, int64_t N, int64_t D, const int32_t* rowptr,
                                 const int32_t* cols, float scale, float* out) {
    if (D <= 0 || D > 1024) return ns_fail("ns_model3d_aam_csr_attention: 0 < D <= 1024");
    if (N <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    csr_attention_k<<<(unsigned) N, CSR_WG, 0, g->q>>>(q, k, v, D, rowptr, cols, scale, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_aam_fps(ns_gpu* g, const float* x, int64_t N, int64_t D, int64_t K, float* dist, int32_t* idx) {
    if (K <= 0 || N <= 0) return 0;
    if (K > N || D <= 0 || D > FPS_MAXD) return ns_fail("ns_model3d_aam_fps: K <= N, 0 < D <= 1024");
    NS_CU(cudaSetDevice(g->dev));
    fps_k<<<1, FPS_WG, 0, g->q>>>(x, N, D, K, dist, idx);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_aam_gather(ns_gpu* g, const float* x, const int32_t* idx, int64_t R, int64_t C, float* out) {
    if (R <= 0 || C <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    gather_k<<<blocks(R * C), 256, 0, g->q>>>(x, idx, C, out, R * C);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_aam_flow_in(ns_gpu* g, const float* f0, const float* y, int64_t K, int64_t C, int64_t F, float* out) {
    if (K <= 0 || C <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    flow_in_k<<<blocks(2 * K * C), 256, 0, g->q>>>(f0, y, K, C, F, out, 2 * K * C);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_aam_cfg(ns_gpu* g, const float* flow, int64_t n, float scale, float* out) {
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    cfg_k<<<blocks(n), 256, 0, g->q>>>(flow, n, scale, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_aam_axpy(ns_gpu* g, const float* y, const float* f, float a, int64_t n, float* out) {
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    axpy_k<<<blocks(n), 256, 0, g->q>>>(y, f, a, n, out);
    NS_LAUNCH_CHECK();
    return 0;
}

}  // extern "C"
