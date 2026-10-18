/* q35.h: the qwen35 engine's own kernels (dense Qwen3.5 / Qwen3.8: gated DeltaNet and gated full attention), its C
 * ABI, bound by its crate by name (llm/qwen35/src/ffi.rs). After nextsycl's kernels/llm/qwen35/q35.h and the generic
 * ones of its glm5next engine (glm.h: conv + SiLU, L2 norm, the delta-rule scan, SwiGLU) - the same names and arguments
 * (glm.h's under an ns_q35_ prefix here). The products, dequantizers, embedding, RMS norm, add and GEMM are the
 * nemotronh engine's (nh.h; Q2_K added to them for this model's file). Float32 activations, row-major, all queued on
 * the GPU's stream (each entry selects the GPU's device: an engine over two cards calls them from either); 0, or -1
 * with the reason in ns_last_error. */
#ifndef NS_Q35_H
#define NS_Q35_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* ---- q35.h (nextsycl's) */

/* Per (row t, head h): x = src[t * ss + h * hs .. + D] RMS-normed (weight w [D], eps), then rotate-half rotary positions
 * on its first n_rot features (pairs i, i + n_rot/2; angle (p0 + t) * theta^(-2i / n_rot)); into dst [t * ds + h * D] */
int ns_q35_qk_norm_rope(ns_gpu* g, const float* src, int64_t ss, int64_t hs, float* dst, int64_t ds, int64_t T, int64_t H, int64_t D,
                        const float* w, float eps, int64_t n_rot, float theta, int64_t p0);
/* As ns_q35_qk_norm_rope with each row's own positions: pos3 [3][T] int32 on the GPU, (temporal, row, column) - a
 * picture's tokens. Qwen3.8's interleaved M-RoPE (llama.cpp's IMROPE, sections 11 / 11 / 10): rotary pair k turns by
 * position section k % 3. A text row's three are equal (= ns_q35_qk_norm_rope at that position) */
int ns_q35_qk_norm_mrope(ns_gpu* g, const float* src, int64_t ss, int64_t hs, float* dst, int64_t ds, int64_t T, int64_t H, int64_t D,
                         const float* w, float eps, int64_t n_rot, float theta, const int32_t* pos3);
/* The cache [Hkv][cap][D] half: rows k [T, Hkv x D] (ks floats apart) and v (vs apart) at positions p0.. */
int ns_q35_kv_store(ns_gpu* g, const float* k, int64_t ks, const float* v, int64_t vs, int64_t T, int64_t Hkv, int64_t D, int64_t cap, int64_t p0,
                    void* kc, void* vc);
/* As ns_q35_kv_store into a q8_0 cache (llama.cpp's cache-type q8_0; D 256): a row [Hkv][cap] is the 256 features
 * as int8 then the 8 half scales of its blocks of 32 (272 bytes; d = amax / 127, q = round(x / d)) */
int ns_q35_kv_store_q8(ns_gpu* g, const float* k, int64_t ks, const float* v, int64_t vs, int64_t T, int64_t Hkv, int64_t D, int64_t cap, int64_t p0,
                       void* kc, void* vc);
/* Floats of scratch ns_q35_attn needs for T rows of Hq heads (Hkv key heads) x D at positions p0.. (0: none; only
 * T <= 16 needs any) */
int64_t ns_q35_attn_scratch(int64_t T, int64_t Hq, int64_t Hkv, int64_t D, int64_t p0);
/* Causal attention (scale 1/sqrt(D)) of rows q [T, Hq x D] (qs floats apart) at positions p0.. against the cache (D 256,
 * Hq / Hkv 1, 2, 4, 6 or 8; any length up to cap): out [T, Hq x D] (contiguous). One or two rows (decode): float32, the keys
 * split over blocks. 3 rows and more (verify passes, prompt chunks): the tensor cores (half products, float32 sums; tiles of
 * 16 rows x the group's heads against 32 keys), the keys split when the grid is small - 8 query heads a key head:
 * float32 (splits up to 16 rows, tiles past). Scratch: ns_q35_attn_scratch (0 when the pass has one split) */
int ns_q35_attn(ns_gpu* g, const float* q, int64_t qs, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t D,
                int64_t cap, int64_t p0, float* out, float* scratch);
/* As ns_q35_attn against a q8_0 cache (ns_q35_kv_store_q8's rows; each key and value widened to half as it is read) */
int ns_q35_attn_q8(ns_gpu* g, const float* q, int64_t qs, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t D,
                   int64_t cap, int64_t p0, float* out, float* scratch);
/* The attention's output gate: out [T, H x D] *= sigmoid(gate), the gate at q_full[t * gs + h * 2D + D + d] */
int ns_q35_gate_mul(ns_gpu* g, float* out, const float* q_full, int64_t gs, int64_t T, int64_t H, int64_t D);
/* DeltaNet's per-head gates: eg [T, H, d] = exp(softplus(alpha + dt) * a) (each head's d copies), beta = sigmoid(beta);
 * alpha / beta [T, H] (as rows as apart) */
int ns_q35_gdn_gates(ns_gpu* g, const float* alpha, float* beta, const float* dt, const float* a, float* eg, int64_t T, int64_t H, int64_t d);
/* q / k heads to the value heads: dst [T, Hv, d] = src[t * ss + (h % Hk) * d ..] */
int ns_q35_expand(ns_gpu* g, const float* src, int64_t ss, float* dst, int64_t T, int64_t Hk, int64_t Hv, int64_t d);
/* DeltaNet's output: y [T, H, d] = rms_norm(o) * w * silu(z), per head; z rows zs floats apart */
int ns_q35_gdn_out(ns_gpu* g, const float* o, const float* z, int64_t zs, const float* w, float* y, int64_t T, int64_t H, int64_t d, float eps);
/* The index of each row's largest value (the lowest on a tie): x [rows, n] -> out [rows] */
int ns_q35_argmax(ns_gpu* g, const float* x, int64_t rows, int64_t n, int32_t* out);

/* ---- glm.h's generic ones (nextsycl kernels/llm/glm5next/glm.h), ns_q35_ prefixed */

/* The causal depthwise conv + SiLU: rows x [T][D] after the k-1 rows before them (state [k-1][D], oldest first; updated
 * to the last k-1), w [D][k]: out [T][D] = silu(sum_j in[t + j][c] * w[c][j]). snap (nullable): the state after each
 * row but the last, [T-1][k-1][D] (speculative rollback) */
int ns_q35_conv_silu(ns_gpu* g, const float* x, float* state, const float* w, float* out, int64_t T, int64_t D, int k, float* snap);
/* x [rows][n] /= sqrt(sum x^2 + eps) per row - in place (eps inside the root, as glm.cpp) */
int ns_q35_l2_norm(ns_gpu* g, float* x, int64_t rows, int64_t n, float eps);
/* The gated delta rule (KDA's scan, per-feature decay; d 128 only): per row t and head h, with q, k, v [T][H][d], gate
 * [T][H][d] (the decay factors, multiplied in), beta [T][H]: S = diag(gate) S;  S += k (beta (v - S^T k))^T;
 * o = S^T q / sqrt(d) (the scale is the kernel's).  S [H][d][d] (value index first: S[h][j][i] is key i, value j;
 * glm.cpp's is key first - a warp reads a value column whole), updated; o [T][H][d]. snap
 * (nullable): S after each row but the last, [T-1][H][d][d] */
int ns_q35_kda_scan(ns_gpu* g, const float* q, const float* k, const float* v, const float* gate, const float* beta, float* S, float* o,
                    int64_t T, int64_t H, int64_t d, float* snap);
/* out [n] = silu(min(gate, limit)) * clamp(up, -limit, limit) (always clamped: +inf for none); out may be gate */
int ns_q35_swiglu(ns_gpu* g, const float* gate, const float* up, float* out, int64_t n, float limit);

/* ---- the prompt products, ggml-cuda's MMQ (q35_mmq.cu, third_party/ggml-cuda): x as q8_1 in MMQ's tiles, the
 * matrix's blocks on the int8 tensor cores. Types: Q2_K Q4_K Q6_K IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S IQ4_XS */
int ns_q35_mmq_supported(int type);
/* bytes of x [T][K] quantized for ns_q35_mmq */
size_t ns_q35_mmq_q8_bytes(int64_t K, int64_t T);
/* x [T][K] float32 -> xq, in the layout a matrix of `type` reads */
int ns_q35_mmq_quantize(ns_gpu* g, int type, const float* x, int64_t T, int64_t K, void* xq);
/* y [T][n_out] = x . W^T, W [n_out][K] in `type`'s blocks, x as ns_q35_mmq_quantize made it (for that type) */
int ns_q35_mmq(ns_gpu* g, int type, const void* w, int64_t n_out, int64_t K, const void* xq, int64_t T, float* y);

#ifdef __cplusplus
}
#endif
#endif
