// nh_iq.cu: the products from the stored blocks. Strata's CUDA kernels (third_party/strata/iq_kernels.cu, MIT; the
// block layouts, grids and dot products are llama.cpp's ggml-cuda, third_party/ggml/LICENSE) under nh.h's C ABI.
#include "ns.h"
#include "ns_internal.hpp"
#include "nh.h"

#include "third_party/strata/iq_kernels.cu"

#include <cuda_fp16.h>

// the experts of a layer in one launch: entry j multiplies the weights at wp[j] (rows of `row_bytes`) by the q8_1 row xrow[j]
// of x (rows of n_in values), into row j of y (n_out floats)
namespace strata::kernels {
namespace {
template<int TY>
__global__ void __launch_bounds__(128) mmvq_ptrs_kernel(const unsigned long long* __restrict__ wp, size_t row_bytes, const int32_t* __restrict__ xrow,
                                                        const block_q8_1* __restrict__ x, float* __restrict__ y, int n_in, int n_out) {
    const int row = blockIdx.x * 4 + threadIdx.y;
    if (row >= n_out) return;
    const int j = blockIdx.y;
    const uint8_t* wr = (const uint8_t*) wp[j] + (size_t) row * row_bytes;
    const float s = row_dot<TY>(wr, x + (size_t) xrow[j] * (n_in / 32), n_in / Fmt<TY>::qk, threadIdx.x);
    if (threadIdx.x == 0) y[(size_t) j * n_out + row] = s;
}
}  // namespace

void iq_mmvq_ptrs(int t, const unsigned long long* wp, const int32_t* xrow, const void* x_q8_1, float* y, int n_in, int n_out, int n_entries, void* stream) {
    const size_t rb = iq_row_bytes(t, n_in);
    const dim3 grid((unsigned) ((n_out + 3) / 4), (unsigned) n_entries), block(32, 4);
    cudaStream_t s = (cudaStream_t) stream;
    switch (t) {
#define NH_PTRS(T) case T: mmvq_ptrs_kernel<T><<<grid, block, 0, s>>>(wp, rb, xrow, (const block_q8_1*) x_q8_1, y, n_in, n_out); break;
        STRATA_MMVQ_FMTS(NH_PTRS)
#undef NH_PTRS
        default: std::fprintf(stderr, "iq_mmvq_ptrs: type %d is not supported\n", t); std::exit(1);
    }
    check("iq_mmvq_ptrs");
}
}  // namespace strata::kernels

namespace {

// F32 / F16 / BF16 to float32, one thread a value
__global__ void to_f32_k(int type, const void* src, int64_t n, float* dst) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (type == 0) dst[i] = ((const float*) src)[i];
    else if (type == 1) dst[i] = __half2float(((const __half*) src)[i]);
    else dst[i] = __uint_as_float(((uint32_t) ((const uint16_t*) src)[i]) << 16);   // BF16
}

}  // namespace

extern "C" {

size_t ns_nh_row_bytes(int type, int64_t n) {
    return strata::kernels::iq_row_bytes(type, n);
}

int ns_nh_mmvq_supported(int type) {
    return strata::kernels::iq_supported(type) ? 1 : 0;
}

int ns_nh_dequant_f32(ns_gpu* g, int type, const void* src, int64_t n, float* dst) {
    NS_TRY
    if (n <= 0) return 0;
    if (type == 0 || type == 1 || type == 30) {
        to_f32_k<<<(unsigned) ((n + 255) / 256), 256, 0, g->q>>>(type == 30 ? 2 : type, src, n, dst);
        NS_LAUNCH_CHECK();
        return 0;
    }
    if (type == 19 || type == 14) return ns_fail("ns_nh_dequant_f32: IQ1_S and Q6_K have products here, not a dequantizer");
    if (n % 256 != 0 || !strata::kernels::iq_supported(type)) return ns_fail("ns_nh_dequant_f32: a type the products do not take, or n not a multiple of 256");
    strata::kernels::iq_dequant_f32(type, src, n, dst, g->q);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

size_t ns_nh_q8_1_bytes(int64_t C, int64_t T) {
    return (size_t) (C / 32) * 36 * (size_t) T;
}

int ns_nh_quantize(ns_gpu* g, const float* x, int64_t T, int64_t C, void* out) {
    NS_TRY
    if (C % 32 != 0) return ns_fail("ns_nh_quantize: the columns a multiple of 32");
    strata::kernels::quantize_q8_1_rows(x, T, C, out, g->q);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_nh_mmvq(ns_gpu* g, int type, const void* w, const void* xq, float* y, int64_t n_in, int64_t n_out, int64_t ncols) {
    NS_TRY
    if (!strata::kernels::iq_supported(type)) return ns_fail("ns_nh_mmvq: type " + std::to_string(type) + " is not one the products take");
    strata::kernels::iq_mmvq(type, w, xq, y, (int) n_in, (int) n_out, (int) ncols, g->q);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_nh_mmvq_ptrs(ns_gpu* g, int type, const void* wptrs, const int32_t* xrow, const void* xq, float* y, int64_t n_in, int64_t n_out, int64_t n_entries) {
    NS_TRY
    if (n_entries <= 0) return 0;
    if (!strata::kernels::iq_supported(type)) return ns_fail("ns_nh_mmvq_ptrs: type " + std::to_string(type) + " is not one the products take");
    strata::kernels::iq_mmvq_ptrs(type, (const unsigned long long*) wptrs, xrow, xq, y, (int) n_in, (int) n_out, (int) n_entries, g->q);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_nh_embed(ns_gpu* g, int type, const void* table, int64_t row_bytes, const int32_t* tokens, int64_t n_tok, int64_t n_embd, float* out) {
    NS_TRY
    if (!strata::kernels::embed_type_supported(type)) return ns_fail("ns_nh_embed: type " + std::to_string(type) + " is not one the embedding takes");
    strata::kernels::iq_embed_rows(type, table, (size_t) row_bytes, tokens, n_tok, n_embd, out, g->q);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

}  // extern "C"
