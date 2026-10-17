// t2.cu: the TRELLIS.2 engine's own kernels - what the shared diffusion kernels (nsd) do not have: the modulated
// layer norm, TRELLIS's per-head RMS norm with its interleaved 3-D rotary embedding, the gated residual adds, the
// sparse convolution's 27-tap gather, the channel-to-space subdivision's gathers, the dense decoder's 3-D pixel
// shuffle, Pixal3D's bilinear feature sampling. nextsycl's t2.cpp in CUDA, the same names, arguments and arithmetic:
// its work-group reductions are shared-memory block reductions, its sub-group of 32 a head is a warp. Every loop is
// bounded by its arguments. (The float32-output products, oneDNN's on nextsycl, are cuBLAS: mm.cu.)
#include "ns.h"
#include "ns_internal.hpp"
#include "t2.h"

#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>

namespace {

unsigned blocks(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }

__device__ __forceinline__ float ld(const void* p, int dt, size_t i) {
    return dt == 0 ? ((const float*) p)[i] : __half2float(((const __half*) p)[i]);
}
__device__ __forceinline__ void st(void* p, int dt, size_t i, float v) {
    if (dt == 0) ((float*) p)[i] = v;
    else ((__half*) p)[i] = __float2half_rn(v);
}
bool dt_ok(int dt) { return dt == 0 || dt == 1; }

// the sum over the block, every thread given it (blockDim.x a multiple of 32, at most 1024)
__device__ float block_sum(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < (int) (blockDim.x >> 5); ++i) t += red[i];
    __syncthreads();
    return t;
}

constexpr int LN_WG = 256, LN_PER = 32;  // a row of up to 8,192 values held in registers

__global__ void __launch_bounds__(LN_WG) ln_mod_k(const void* x, int x_dt, int64_t r0, int64_t C, float eps, const float* weight,
                                                  const float* bias, const float* scale, const float* shift, void* out, int out_dt) {
    __shared__ float red[LN_WG / 32];
    const size_t r = r0 + blockIdx.x;
    const int l = threadIdx.x;
    float v[LN_PER];
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < LN_PER; ++i) {
        const int64_t c = l + (int64_t) i * LN_WG;
        v[i] = c < C ? ld(x, x_dt, r * C + c) : 0.f;
        s += v[i];
    }
    const float mean = block_sum(s, red) / (float) C;
    float q = 0.f;
#pragma unroll
    for (int i = 0; i < LN_PER; ++i) {
        const int64_t c = l + (int64_t) i * LN_WG;
        if (c < C) q += (v[i] - mean) * (v[i] - mean);
    }
    const float var = block_sum(q, red) / (float) C;
    const float rs = rsqrtf(var + eps);
#pragma unroll
    for (int i = 0; i < LN_PER; ++i) {
        const int64_t c = l + (int64_t) i * LN_WG;
        if (c >= C) break;
        float y = (v[i] - mean) * rs;
        if (weight) y = y * weight[c] + (bias ? bias[c] : 0.f);
        if (scale) y = y * (1.f + scale[c]);
        if (shift) y += shift[c];
        st(out, out_dt, r * C + c, y);
    }
}

// a warp a head: lane l holds features 4l .. 4l + 3 (pairs 2l and 2l + 1)
__global__ void qk_norm_rope_k(__half* xh, int64_t H, int64_t D, int64_t stride, const float* gamma, const float* cs) {
    const int64_t grp = blockIdx.x, m = grp / H, h = grp % H;
    const int l = threadIdx.x;
    __half* p = xh + m * stride + h * D + 4 * l;
    float v[4];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i] = __half2float(p[i]);
        ss += v[i] * v[i];
    }
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float k = sqrtf((float) D) / fmaxf(sqrtf(ss), 1e-12f);
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = v[i] * k * gamma[h * D + 4 * l + i];
    if (cs) {
#pragma unroll
        for (int pr = 0; pr < 2; ++pr) {
            const float* c = cs + (m * (D / 2) + 2 * l + pr) * 2;
            const float a = v[2 * pr], b = v[2 * pr + 1];
            v[2 * pr] = a * c[0] - b * c[1];
            v[2 * pr + 1] = a * c[1] + b * c[0];
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) p[i] = __float2half_rn(v[i]);
}

__global__ void gate_add_k(float* x, int64_t C, const void* h, int h_dt, const float* gate, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = ld(h, h_dt, (size_t) i);
    x[i] += gate ? v * gate[i % C] : v;
}

__global__ void add_row_k(float* x, int64_t C, const float* row, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += row[i % C];
}

__global__ void convert_k(const void* src, int src_dt, void* dst, int dst_dt, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) st(dst, dst_dt, (size_t) i, ld(src, src_dt, (size_t) i));
}

__global__ void silu_k(void* x, int dt, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = ld(x, dt, (size_t) i);
    st(x, dt, (size_t) i, v / (1.f + expf(-v)));
}

__global__ void add_k(void* x, int x_dt, const void* y, int y_dt, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) st(x, x_dt, (size_t) i, ld(x, x_dt, (size_t) i) + ld(y, y_dt, (size_t) i));
}

// 8 halves a load
__global__ void gather27_k(const uint4* f, int64_t C8, const int32_t* nmap, int64_t row0, uint4* o, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t c = i % C8, t = (i / C8) % 27, r = i / (27 * C8);
    const int32_t s = nmap[(row0 + r) * 27 + t];
    o[i] = s >= 0 ? f[(int64_t) s * C8 + c] : make_uint4(0u, 0u, 0u, 0u);
}

__global__ void gather_rows_vec_k(const uint4* s, int64_t V, const int32_t* idx, uint4* o, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = s[(int64_t) idx[i / V] * V + i % V];
}

__global__ void gather_rows_k(const void* src, int dt, int64_t C, const int32_t* idx, void* out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) st(out, dt, (size_t) i, ld(src, dt, (size_t) ((int64_t) idx[i / C] * C + i % C)));
}

__global__ void c2s_skip_k(const float* x, int64_t Cin, const int32_t* parent, const int32_t* sub, int64_t Cout, int64_t r, float* out,
                           int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t j = i / Cout, c = i % Cout;
    out[i] = x[(int64_t) parent[j] * Cin + ((int64_t) sub[j] * Cout + c) / r];
}

__global__ void pixel_shuffle3d_k(const __half* xi, int64_t D, int64_t C, __half* o, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t D2 = 2 * D;
    const int64_t c = i % C, v = i / C;
    const int64_t k2 = v % D2, j2 = (v / D2) % D2, i2 = v / (D2 * D2);
    const int64_t s = (i2 % 2) * 4 + (j2 % 2) * 2 + (k2 % 2);
    o[i] = xi[(((i2 / 2) * D + j2 / 2) * D + k2 / 2) * 8 * C + c * 8 + s];
}

__global__ void grid_sample2d_k(const float* feat, int64_t Hf, int64_t Wf, int64_t C, const float* uv, float* out, int64_t stride, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t j = i / C, c = i % C;
    // grid_sample, align_corners=False: pixel = ((u + 1) W - 1) / 2; padding_mode border: clipped to the map
    const float fx = fminf(fmaxf(((uv[2 * j] + 1.f) * (float) Wf - 1.f) * 0.5f, 0.f), (float) (Wf - 1));
    const float fy = fminf(fmaxf(((uv[2 * j + 1] + 1.f) * (float) Hf - 1.f) * 0.5f, 0.f), (float) (Hf - 1));
    const float x0f = floorf(fx), y0f = floorf(fy);
    const int64_t x0 = (int64_t) x0f, y0 = (int64_t) y0f;
    const float lx = fx - x0f, ly = fy - y0f;
    float acc = 0.f;
    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            const int64_t yy = y0 + dy, xx = x0 + dx;
            if (yy < 0 || yy >= Hf || xx < 0 || xx >= Wf) continue;
            const float w = (dy ? ly : 1.f - ly) * (dx ? lx : 1.f - lx);
            acc += w * feat[(yy * Wf + xx) * C + c];
        }
    }
    out[j * stride + c] = acc;
}

}  // namespace

extern "C" {

int ns_model3d_trellis2_ln_mod(ns_gpu* g, const void* x, int x_dt, int64_t M, int64_t C, float eps, const float* weight, const float* bias,
                               const float* scale, const float* shift, void* out, int out_dt) {
    if (C > (int64_t) LN_WG * LN_PER || C <= 0) return ns_fail("ns_model3d_trellis2_ln_mod: rows of 1..8192");
    if (!dt_ok(x_dt) || !dt_ok(out_dt)) return ns_fail("ns_model3d_trellis2_ln_mod: float32 or half");
    if (M <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    // launches of at most 2^22 rows (as nextsycl's)
    constexpr int64_t kRows = int64_t(1) << 22;
    for (int64_t r0 = 0; r0 < M; r0 += kRows) {
        const int64_t rows = std::min(kRows, M - r0);
        ln_mod_k<<<(unsigned) rows, LN_WG, 0, g->q>>>(x, x_dt, r0, C, eps, weight, bias, scale, shift, out, out_dt);
        NS_LAUNCH_CHECK();
    }
    return 0;
}

int ns_model3d_trellis2_qk_norm_rope(ns_gpu* g, void* x, int64_t M, int64_t H, int64_t D, int64_t stride, const float* gamma, const float* cs) {
    if (D != 128) return ns_fail("ns_model3d_trellis2_qk_norm_rope: heads of 128");
    if (stride < H * D) return ns_fail("ns_model3d_trellis2_qk_norm_rope: a stride under a row");
    if (M <= 0 || H <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    qk_norm_rope_k<<<(unsigned) (M * H), 32, 0, g->q>>>((__half*) x, H, D, stride, gamma, cs);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_trellis2_gate_add(ns_gpu* g, float* x, int64_t M, int64_t C, const void* h, int h_dt, const float* gate) {
    if (!dt_ok(h_dt)) return ns_fail("ns_model3d_trellis2_gate_add: float32 or half");
    if (M <= 0 || C <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    gate_add_k<<<blocks(M * C), 256, 0, g->q>>>(x, C, h, h_dt, gate, M * C);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_trellis2_add_row(ns_gpu* g, float* x, int64_t M, int64_t C, const float* row) {
    if (M <= 0 || C <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    add_row_k<<<blocks(M * C), 256, 0, g->q>>>(x, C, row, M * C);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_trellis2_convert(ns_gpu* g, const void* src, int src_dt, int64_t n, void* dst, int dst_dt) {
    if (!dt_ok(src_dt) || !dt_ok(dst_dt)) return ns_fail("ns_model3d_trellis2_convert: float32 or half");
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    convert_k<<<blocks(n), 256, 0, g->q>>>(src, src_dt, dst, dst_dt, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_trellis2_silu(ns_gpu* g, void* x, int dt, int64_t n) {
    if (!dt_ok(dt)) return ns_fail("ns_model3d_trellis2_silu: float32 or half");
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    silu_k<<<blocks(n), 256, 0, g->q>>>(x, dt, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_trellis2_add(ns_gpu* g, void* x, int x_dt, const void* y, int y_dt, int64_t n) {
    if (!dt_ok(x_dt) || !dt_ok(y_dt)) return ns_fail("ns_model3d_trellis2_add: float32 or half");
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    add_k<<<blocks(n), 256, 0, g->q>>>(x, x_dt, y, y_dt, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_trellis2_gather27(ns_gpu* g, const void* feats, int64_t C, const int32_t* nmap, int64_t row0, int64_t rows, void* out) {
    if (rows <= 0 || C <= 0) return 0;
    if (C % 8) return ns_fail("ns_model3d_trellis2_gather27: channels a multiple of 8");
    NS_CU(cudaSetDevice(g->dev));
    const int64_t C8 = C / 8, n = rows * 27 * C8;
    gather27_k<<<blocks(n), 256, 0, g->q>>>((const uint4*) feats, C8, nmap, row0, (uint4*) out, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_trellis2_gather_rows(ns_gpu* g, const void* src, int dt, int64_t C, const int32_t* idx, int64_t n, void* out) {
    if (!dt_ok(dt)) return ns_fail("ns_model3d_trellis2_gather_rows: float32 or half");
    if (n <= 0 || C <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    const int64_t e = dt == 0 ? 4 : 2;
    if ((C * e) % 16 == 0) {
        const int64_t V = C * e / 16;
        gather_rows_vec_k<<<blocks(n * V), 256, 0, g->q>>>((const uint4*) src, V, idx, (uint4*) out, n * V);
    } else {
        gather_rows_k<<<blocks(n * C), 256, 0, g->q>>>(src, dt, C, idx, out, n * C);
    }
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_trellis2_c2s_skip(ns_gpu* g, const float* x, int64_t Cin, const int32_t* parent, const int32_t* sub, int64_t n, int64_t Cout,
                                 int64_t r, float* out) {
    if (n <= 0 || Cout <= 0) return 0;
    if (r <= 0 || 8 * Cout != Cin * r) return ns_fail("ns_model3d_trellis2_c2s_skip: 8 Cout must be Cin r");
    NS_CU(cudaSetDevice(g->dev));
    c2s_skip_k<<<blocks(n * Cout), 256, 0, g->q>>>(x, Cin, parent, sub, Cout, r, out, n * Cout);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_trellis2_pixel_shuffle3d(ns_gpu* g, const void* x, int64_t D, int64_t C, void* out) {
    if (D <= 0 || C <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    const int64_t D2 = 2 * D, n = D2 * D2 * D2 * C;
    pixel_shuffle3d_k<<<blocks(n), 256, 0, g->q>>>((const __half*) x, D, C, (__half*) out, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_trellis2_grid_sample2d(ns_gpu* g, const float* feat, int64_t Hf, int64_t Wf, int64_t C, const float* uv, int64_t n, float* out,
                                      int64_t stride) {
    if (n <= 0 || C <= 0) return 0;
    if (stride < C) return ns_fail("ns_model3d_trellis2_grid_sample2d: a stride under the channels");
    NS_CU(cudaSetDevice(g->dev));
    grid_sample2d_k<<<blocks(n * C), 256, 0, g->q>>>(feat, Hf, Wf, C, uv, out, stride, n * C);
    NS_LAUNCH_CHECK();
    return 0;
}

}  // extern "C"
