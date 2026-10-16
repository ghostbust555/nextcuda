/* nh.h: the nemotronh engine's kernels (Nemotron-H: Mamba-2 mixers, latent MoE, attention without positions) and its
 * C ABI, bound by its crate by name (llm/nemotronh/src/ffi.rs). Float32 activations, row-major, all queued on the GPU's
 * stream; 0, or -1 with the reason in ns_last_error. After llama.cpp's src/models/nemotron-h.cpp and nextsycl's
 * kernels/llm/nemotronh/nh.cpp (the same names and arguments; plain kernels first, tuned ones when it works). The
 * products from the stored blocks are Strata's (third_party/strata/iq_kernels.cu, MIT) and ggml-cuda's. */
#ifndef NS_NH_H
#define NS_NH_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* The Mamba-2 conv: rows x [T] of C channels (xs floats apart) after the k-1 inputs before them (state [k-1][C],
 * oldest first; updated to the last k-1): out [T][C] = silu(sum_j in[t + j][c] * w[c][j] + b[c]). */
int ns_nh_conv(ns_gpu* g, const float* x, int64_t xs, float* state, const float* w, const float* b, float* out, int64_t T, int64_t C, int k);
/* The Mamba-2 scan (ggml's ssm_scan, d_state N): rows xbc [T] = x [H][P] | B [G][N] | C [G][N] (xs floats apart), dt [T][H]
 * (dts apart, before its bias); per row and head dt' = softplus(dt + dt_bias), the state S [H][P][N] decayed by
 * exp(dt' A) plus dt' B x, y [T][H][P] = C . S + Dp x. A head h reads group h / (H / G). N a multiple of 32, up to 256 */
int ns_nh_ssm(ns_gpu* g, const float* xbc, int64_t xs, const float* dt, int64_t dts, const float* dt_bias, const float* A, const float* Dp, float* S,
              float* y, int64_t T, int64_t H, int64_t P, int64_t N, int64_t G);
/* out [T][C] = rms_norm(y * silu(z)) * w over groups of `group` values (z rows zs apart) */
int ns_nh_gated_norm(ns_gpu* g, const float* y, const float* z, int64_t zs, const float* w, float* out, int64_t T, int64_t C, int64_t group, float eps);
/* x = max(x, 0)^2 */
int ns_nh_relu2(ns_gpu* g, float* x, int64_t n);
/* The cache [Hkv][cap][D] half: rows k [T, Hkv x D] (ks floats apart) and v (vs apart) at positions p0.. */
int ns_nh_kv_store(ns_gpu* g, const float* k, int64_t ks, const float* v, int64_t vs, int64_t T, int64_t Hkv, int64_t D, int64_t cap, int64_t p0,
                   void* kc, void* vc);
/* Causal attention (no positions; scale 1/sqrt(D)) of rows q [T, Hq x D] (qs floats apart) at positions p0.. against the
 * cache: out [T, Hq x D]. D 128; the keys' scores in shared memory: p0 + T up to ns_nh_attn_max_keys() */
int64_t ns_nh_attn_max_keys(void);
int ns_nh_attn(ns_gpu* g, const float* q, int64_t qs, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t D, int64_t cap,
               int64_t p0, float* out);

/* y [T][C] = rms_norm(x [T][C]) * w (float32) */
int ns_nh_rms_norm(ns_gpu* g, const float* x, const float* w, float* y, int64_t T, int64_t C, float eps);
/* y += x */
int ns_nh_add(ns_gpu* g, float* y, const float* x, int64_t n);
/* y [T][R] = x [T][C] . W [R][C]^T with W float32 (the router) */
int ns_nh_gemv_f32(ns_gpu* g, const float* W, const float* x, float* y, int64_t T, int64_t R, int64_t C);
/* out [T][L] = sum over the token's entries e in [tp[t], tp[t+1]) of w[e] * dn[e][L] (entries in token order) */
int ns_nh_moe_combine(ns_gpu* g, float* out, const float* dn, const int32_t* tp, const float* w, int64_t T, int64_t L);

/* a GGUF type's row bytes for n values (0: not one the products take) */
size_t ns_nh_row_bytes(int type, int64_t n);
int ns_nh_mmvq_supported(int type);
/* n values of a tensor of any type the products take (or F32 / F16 / BF16) to float32 */
int ns_nh_dequant_f32(ns_gpu* g, int type, const void* src, int64_t n, float* dst);
/* Q8_1 blocks (36 bytes a 32 values, fp16 scale and sum) of rows x [T][C] (C a multiple of 32) */
size_t ns_nh_q8_1_bytes(int64_t C, int64_t T);
int ns_nh_quantize(ns_gpu* g, const float* x, int64_t T, int64_t C, void* out);
/* y [ncols][n_out] = W [n_out][n_in] . xq [ncols] (W stored blocks of `type`, rows of n_in) */
int ns_nh_mmvq(ns_gpu* g, int type, const void* w, const void* xq, float* y, int64_t n_in, int64_t n_out, int64_t ncols);
/* rows of a token embedding table (device or mapped host memory) as float32 */
int ns_nh_embed(ns_gpu* g, int type, const void* table, int64_t row_bytes, const int32_t* tokens, int64_t n_tok, int64_t n_embd, float* out);

#ifdef __cplusplus
}
#endif
#endif
