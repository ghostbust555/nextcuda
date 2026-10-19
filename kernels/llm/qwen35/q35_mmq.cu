// q35_mmq.cu: the prompt products from the stored blocks, ggml-cuda's MMQ (third_party/ggml-cuda, MIT): x quantized
// to q8_1 in MMQ's tile layout, each matrix's blocks multiplied on the int8 tensor cores with float32 sums per block
// (no half copy of the weights). The types Qwen3.8-27B GSQ-RCO's file holds; IQ1_M (one tensor) is not one: the
// engine expands it to half as before. Under q35.h's C ABI.
#include "ns.h"
#include "ns_internal.hpp"
#include "q35.h"

#include "third_party/ggml-cuda/common.cuh"

#include <mutex>
#include <vector>

// the pool the launchers take the stream-k fixup buffer from: one cached allocation per device, grown on demand
struct ns_mmq_pool : ggml_cuda_pool {
    int dev = 0;
    void* buf = nullptr;
    size_t cap = 0;
    bool used = false;
    void* alloc(size_t size, size_t* actual) override {
        if (used || size > cap) {
            // a second buffer at once (none here: one launch at a time), or more room
            if (!used && buf) {
                cudaFree(buf);
                buf = nullptr;
                cap = 0;
            }
            void* p = nullptr;
            const size_t want = std::max(size, (size_t) 4 << 20);
            CUDA_CHECK(cudaMalloc(&p, want));
            if (used) {
                *actual = want;
                return p; // freed in free()
            }
            buf = p;
            cap = want;
        }
        used = true;
        *actual = cap;
        return buf;
    }
    void free(void* ptr, size_t) override {
        if (ptr == buf) {
            used = false;
        } else {
            cudaFree(ptr);
        }
    }
};

struct ns_mmq_ctx {
    ggml_cuda_pool& pool(int id) {
        static std::mutex m;
        static std::vector<ns_mmq_pool*> pools;
        std::lock_guard<std::mutex> l(m);
        if ((int) pools.size() <= id) pools.resize(id + 1, nullptr);
        if (!pools[id]) {
            pools[id] = new ns_mmq_pool();
            pools[id]->dev = id;
        }
        return *pools[id];
    }
};

#include "third_party/ggml-cuda/mmq.cuh"
#include "third_party/ggml-cuda/quantize.cuh"
#include "third_party/ggml-cuda/quantize.cu"
#include "third_party/ggml-cuda/mmvq.cu"

// ---- what the headers need from ggml-cuda.cu / ggml.c
const ggml_cuda_device_info& ggml_cuda_info() {
    static ggml_cuda_device_info info = [] {
        ggml_cuda_device_info d = {};
        int n = 0;
        cudaGetDeviceCount(&n);
        d.device_count = d.physical_device_count = std::min(n, GGML_CUDA_MAX_DEVICES);
        for (int i = 0; i < d.device_count; ++i) {
            cudaDeviceProp p;
            cudaGetDeviceProperties(&p, i);
            auto& x = d.devices[i];
            x.cc = 100 * p.major + 10 * p.minor;
            x.nsm = p.multiProcessorCount;
            x.smpb = p.sharedMemPerBlock;
            x.smpbo = p.sharedMemPerBlockOptin;
            x.integrated = p.integrated;
            x.total_vram = p.totalGlobalMem;
            x.warp_size = p.warpSize;
            x.supports_cooperative_launch = p.cooperativeLaunch;
            x.physical_device = i;
            x.physical_share_count = 1;
            x.virtual_index = 0;
        }
        return d;
    }();
    return info;
}

int ggml_cuda_get_device() {
    int id = 0;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

[[noreturn]] void ggml_cuda_error(const char* stmt, const char* func, const char* file, int line, const char* msg) {
    std::fprintf(stderr, "ggml-cuda (MMQ): %s in %s at %s:%d: %s\n", msg, func, file, line, stmt);
    std::abort();
}

extern "C" [[noreturn]] void ggml_abort(const char* file, int line, const char* fmt, ...) {
    std::fprintf(stderr, "ggml-cuda (MMQ): abort at %s:%d: %s\n", file, line, fmt);
    std::abort();
}

// the types this engine multiplies this way
DECL_MMQ_CASE(GGML_TYPE_Q2_K);
DECL_MMQ_CASE(GGML_TYPE_Q4_K);
DECL_MMQ_CASE(GGML_TYPE_Q6_K);
DECL_MMQ_CASE(GGML_TYPE_IQ2_XXS);
DECL_MMQ_CASE(GGML_TYPE_IQ2_XS);
DECL_MMQ_CASE(GGML_TYPE_IQ2_S);
DECL_MMQ_CASE(GGML_TYPE_IQ3_XXS);
DECL_MMQ_CASE(GGML_TYPE_IQ3_S);
DECL_MMQ_CASE(GGML_TYPE_IQ4_XS);

// the types the decode products (ggml-cuda's MMVQ) take here, and their values a block
static int mmvq_block(int t) {
    switch (t) {
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1: case GGML_TYPE_Q8_0: case GGML_TYPE_IQ4_NL:
            return 32;
        case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K: case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S: case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ1_S: case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_XS:
            return 256;
        default:
            return 0;
    }
}

// what mmvq.cu asks of ggml.c (its launch checks): from the table above (an F32 / F16 / BF16 one value a block)
extern "C" int64_t ggml_blck_size(enum ggml_type type) {
    const int b = mmvq_block(type);
    return b > 0 ? b : 1;
}
extern "C" bool ggml_is_quantized(enum ggml_type type) { return mmvq_block(type) > 0; }

static bool mmq_type(int t) {
    switch (t) {
        case GGML_TYPE_Q2_K: case GGML_TYPE_Q4_K: case GGML_TYPE_Q6_K: case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

extern "C" {

// the q8_1 layout x takes for a type's MMQ (ggml's mmq_q8_1_ds_layout; -1: none): matrices of one layout share it
int ns_q35_mmq_layout(int type) { return mmq_type(type) ? (int) mmq_get_q8_1_ds_layout((ggml_type) type) : -1; }

int ns_q35_mmq_supported(int type) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 0;
    // the int8 tensor-core products: Turing and later (both cards here: sm_86, sm_120)
    return mmq_type(type) && ggml_cuda_info().devices[dev].cc >= 750 ? 1 : 0;
}

size_t ns_q35_mmq_q8_bytes(int64_t K, int64_t T) {
    const int64_t kp = GGML_PAD(K, MATRIX_ROW_PADDING);
    // the rows' tiles, and one column tile's worth past them (the launch reads a whole tile)
    return (size_t) (T * kp) * sizeof(block_q8_1_mmq) / QK8_1_MMQ + 128 * sizeof(block_q8_1_mmq);
}

int ns_q35_mmq_quantize(ns_gpu* g, int type, const float* x, int64_t T, int64_t K, void* xq) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (!mmq_type(type) || K % 4 != 0) return ns_fail("ns_q35_mmq_quantize: a type without MMQ, or K not a multiple of 4");
    const int64_t kp = GGML_PAD(K, MATRIX_ROW_PADDING);
    quantize_mmq_q8_1_cuda(x, nullptr, xq, (ggml_type) type, K, K, T * K, T * K, kp, T, 1, 1, g->q);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_q35_mmq(ns_gpu* g, int type, const void* w, int64_t n_out, int64_t K, const void* xq, int64_t T, float* y) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (!mmq_type(type)) return ns_fail("ns_q35_mmq: type " + std::to_string(type) + " has no MMQ here");
    // values a block (all of these: 256)
    const int64_t qk = 256;
    if (K % qk != 0) return ns_fail("ns_q35_mmq: K not a multiple of the type's block");
    const int64_t kp = GGML_PAD(K, MATRIX_ROW_PADDING);
    const int64_t s12 = T * kp * (int64_t) sizeof(block_q8_1) / (QK8_1 * (int64_t) sizeof(int));
    const mmq_args args = {
        (const char*) w, (ggml_type) type, (const int*) xq, nullptr, nullptr, y, nullptr,
        K, n_out, T, K / qk, T, n_out,
        1, 1, 0, s12, 0,
        1, 1, 0, s12, 0,
        T, T};
    ns_mmq_ctx ctx;
    switch (type) {
        case GGML_TYPE_Q2_K:    mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, g->q); break;
        case GGML_TYPE_Q4_K:    mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, g->q); break;
        case GGML_TYPE_Q6_K:    mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, g->q); break;
        case GGML_TYPE_IQ2_XXS: mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, g->q); break;
        case GGML_TYPE_IQ2_XS:  mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, g->q); break;
        case GGML_TYPE_IQ2_S:   mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, g->q); break;
        case GGML_TYPE_IQ3_XXS: mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, g->q); break;
        case GGML_TYPE_IQ3_S:   mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, g->q); break;
        case GGML_TYPE_IQ4_XS:  mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, g->q); break;
        default: return ns_fail("ns_q35_mmq: no MMQ for this type");
    }
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

// the GPU's stream-k fixup buffer at its largest (a block per SM, a tile of at most 128 x 128 floats), so no product
// allocates it later - inside a CUDA graph's capture an allocation fails (a chunk's last, shorter pass is the first
// to need the fixup when the full ones tile the SMs evenly)
int ns_q35_mmq_reserve(ns_gpu* g) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    ns_mmq_ctx ctx;
    ggml_cuda_pool& pool = ctx.pool(g->dev);
    size_t got = 0;
    void* p = pool.alloc((size_t) ggml_cuda_info().devices[g->dev].nsm * 128 * 128 * sizeof(float), &got);
    pool.free(p, got);
    return 0;
    NS_CATCH
}

int ns_q35_mmvq_supported(int type) { return mmvq_block(type) > 0 ? 1 : 0; }

// y [ncols][n_out] = W [n_out][n_in] . xq [ncols] - nh's ns_nh_mmvq's arguments (xq: its Q8_1 rows, ggml's block_q8_1),
// ggml-cuda's MMVQ (llama.cpp's decode products: up to 8 rows)
int ns_q35_mmvq(ns_gpu* g, int type, const void* w, const void* xq, float* y, int64_t n_in, int64_t n_out, int64_t ncols) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    const int qk = mmvq_block(type);
    if (!qk) return ns_fail("ns_q35_mmvq: type " + std::to_string(type) + " has no MMVQ here");
    if (n_in % qk != 0 || n_in % 32 != 0 || ncols < 1 || ncols > MMVQ_MAX_BATCH_SIZE) return ns_fail("ns_q35_mmvq: n_in a multiple of the block, 1..8 rows");
    const int bpr = (int) (n_in / qk), yr = (int) (n_in / 32);
    mul_mat_vec_q_switch_type(w, (ggml_type) type, xq, nullptr, ggml_cuda_mm_fusion_args_device{}, y, (int) n_in, (int) n_out, (int) ncols,
                              bpr, yr, (int) n_out,
                              1, 1, 1, bpr * (int) n_out, yr * (int) ncols, (int) (n_out * ncols),
                              1, 1, bpr * (int) n_out, yr * (int) ncols, (int) (n_out * ncols), 0, g->q);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

}  // extern "C"
