// mm.cu: the TRELLIS.2 DiT's and decoders' products that feed a float32 residual stream, with half inputs and a
// float32 output (accumulated and written in float32; the shared nsd_linear writes its product in the inputs' type).
// nextsycl's mm.cpp in CUDA, the same names, arguments and arithmetic: oneDNN's matmul is here one cuBLAS GemmEx
// (half A / B, float32 C, float32 compute), a handle a GPU, then the bias (float32 [N]) added to each row.
#include "ns.h"
#include "ns_internal.hpp"
#include "t2.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <climits>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace {

struct Blas {
    cublasHandle_t cb = nullptr;
};

std::mutex mu;
std::map<ns_gpu*, std::unique_ptr<Blas>> ctxs;

unsigned blocks(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }

__global__ void add_bias_k(float* out, const float* bias, int64_t N, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += bias[i % N];
}

}  // namespace

extern "C" int ns_model3d_trellis2_linear_f32(ns_gpu* g, const void* x, int64_t M, int64_t K, const void* w, int64_t N, const float* bias, float* out) {
    NS_TRY
    if (M <= 0 || K <= 0 || N <= 0) return 0;
    if (M > INT_MAX || K > INT_MAX || N > INT_MAX) return ns_fail("ns_model3d_trellis2_linear_f32: sides past 2^31");
    NS_CU(cudaSetDevice(g->dev));
    std::lock_guard<std::mutex> l(mu);
    auto& c = ctxs[g];
    if (!c) {
        auto n = std::make_unique<Blas>();
        if (cublasCreate(&n->cb) != CUBLAS_STATUS_SUCCESS) return ns_fail("ns_model3d_trellis2_linear_f32: cublasCreate failed");
        c = std::move(n);
    }
    cublasStatus_t s = cublasSetStream(c->cb, g->q);
    if (s != CUBLAS_STATUS_SUCCESS) return ns_fail("ns_model3d_trellis2_linear_f32: cublasSetStream failed, status " + std::to_string((int) s));
    // row-major out [M, N] = x [M, K] . w^T (w [N, K]) is column-major out^T [N, M] = w^T-of-column-major (K x N, ld K)
    // transposed . x (K x M, ld K)
    const float one = 1.0f, zero = 0.0f;
    s = cublasGemmEx(c->cb, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) M, (int) K, &one, w, CUDA_R_16F, (int) K, x, CUDA_R_16F, (int) K, &zero,
                     out, CUDA_R_32F, (int) N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (s != CUBLAS_STATUS_SUCCESS) return ns_fail("ns_model3d_trellis2_linear_f32: cublasGemmEx failed, status " + std::to_string((int) s));
    if (bias) {
        add_bias_k<<<blocks(M * N), 256, 0, g->q>>>(out, bias, N, M * N);
        NS_LAUNCH_CHECK();
    }
    return 0;
    NS_CATCH
}
