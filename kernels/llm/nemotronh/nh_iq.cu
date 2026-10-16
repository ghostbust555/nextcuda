// nh_iq.cu: the products from the stored blocks. Strata's CUDA kernels (third_party/strata/iq_kernels.cu, MIT; the
// block layouts, grids and dot products are llama.cpp's ggml-cuda, third_party/ggml/LICENSE) under nh.h's C ABI.
#include "ns.h"
#include "ns_internal.hpp"
#include "nh.h"

#include "third_party/strata/iq_kernels.cu"

#include <cuda_fp16.h>

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

int ns_nh_embed(ns_gpu* g, int type, const void* table, int64_t row_bytes, const int32_t* tokens, int64_t n_tok, int64_t n_embd, float* out) {
    NS_TRY
    if (!strata::kernels::embed_type_supported(type)) return ns_fail("ns_nh_embed: type " + std::to_string(type) + " is not one the embedding takes");
    strata::kernels::iq_embed_rows(type, table, (size_t) row_bytes, tokens, n_tok, n_embd, out, g->q);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

}  // extern "C"
