// rignet.cu: RigNet's kernels (rignet.h): the MLPs' ReLU + folded BatchNorm, EdgeConv's messages and their max per
// node, concatenation by columns. On the GPU's own stream; the linear layers are the shared diffusion kernels'
// (float32). GPLv3, as RigNet (github.com/zhan-xu/RigNet). nextsycl's rignet.cpp in CUDA, the same names, arguments
// and arithmetic: its 2-d ranges are here one flat index over rows x columns.
#include "ns.h"
#include "ns_internal.hpp"
#include "rignet.h"

#include <cmath>

namespace {

unsigned blocks(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }

__global__ void relu_affine_k(float* x, int64_t C, const float* scale, const float* shift, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t c = i % C;
    float v = fmaxf(x[i], 0.0f);
    if (scale) v = v * scale[c] + shift[c];
    x[i] = v;
}

__global__ void act_k(float* x, int64_t n, int kind) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    x[i] = kind == 0 ? tanhf(v) : 1.0f / (1.0f + expf(-v));
}

__global__ void edge_feat_k(const float* x, int64_t C, const int32_t* src, const int32_t* dst, int64_t n, float* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t e = i / C, c = i % C;
    const float xi = x[(int64_t) dst[e] * C + c];
    out[e * 2 * C + c] = xi;
    out[e * 2 * C + C + c] = x[(int64_t) src[e] * C + c] - xi;
}

__global__ void seg_max_k(const float* x, int64_t C, const int32_t* rowptr, int64_t n, float* out, int64_t out_stride, int64_t out_off) {
    const int64_t t = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const int64_t i = t / C, c = t % C;
    const int64_t a = rowptr[i], b = rowptr[i + 1];
    float m = -INFINITY;
    for (int64_t r = a; r < b; ++r) m = fmaxf(m, x[r * C + c]);
    out[i * out_stride + out_off + c] = b > a ? m : 0.0f;
}

__global__ void copy_cols_k(const float* src, int64_t src_stride, int64_t C, float* dst, int64_t dst_stride, int64_t dst_off, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t r = i / C, c = i % C;
    dst[r * dst_stride + dst_off + c] = src[r * src_stride + c];
}

}  // namespace

extern "C" {

int ns_model3d_rignet_relu_affine(ns_gpu* g, float* x, int64_t M, int64_t C, const float* scale, const float* shift) {
    if (M <= 0 || C <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    relu_affine_k<<<blocks(M * C), 256, 0, g->q>>>(x, C, scale, shift, M * C);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_rignet_act(ns_gpu* g, float* x, int64_t n, int kind) {
    if (kind < 0 || kind > 1) return ns_fail("ns_model3d_rignet_act: kind 0 (tanh) or 1 (sigmoid)");
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    act_k<<<blocks(n), 256, 0, g->q>>>(x, n, kind);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_rignet_edge_feat(ns_gpu* g, const float* x, int64_t C, const int32_t* src, const int32_t* dst, int64_t E, float* out) {
    if (E <= 0 || C <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    edge_feat_k<<<blocks(E * C), 256, 0, g->q>>>(x, C, src, dst, E * C, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_rignet_seg_max(ns_gpu* g, const float* x, int64_t C, const int32_t* rowptr, int64_t N, float* out, int64_t out_stride,
                              int64_t out_off) {
    if (N <= 0 || C <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    seg_max_k<<<blocks(N * C), 256, 0, g->q>>>(x, C, rowptr, N * C, out, out_stride, out_off);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_rignet_copy_cols(ns_gpu* g, const float* src, int64_t src_stride, int64_t M, int64_t C, float* dst, int64_t dst_stride,
                                int64_t dst_off) {
    if (M <= 0 || C <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    copy_cols_k<<<blocks(M * C), 256, 0, g->q>>>(src, src_stride, C, dst, dst_stride, dst_off, M * C);
    NS_LAUNCH_CHECK();
    return 0;
}

}  // extern "C"
