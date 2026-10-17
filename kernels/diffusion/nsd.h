/* nsd.h: the shared diffusion kernels' C ABI (kernels/diffusion/nsd.cu), bound by name by nextcuda-diffusion
 * (crates/diffusion/src/kernels.rs). The names and arguments are nextsycl's (H3's h3sycl.h), so the engines above
 * them are nextsycl's Rust code; what differs is under them: cuBLAS for the matrix products, CUDA kernels for the
 * rest, on the GPU's own stream.
 *
 * Every data pointer is a CUDA device pointer of the context's GPU. Kernels are asynchronous: a call returns once
 * its work is queued, and the stream is in order, so a later call sees an earlier call's result; nsd_wait is the
 * point that waits. Functions return 0, or -1 with the reason in nsd_last_error() (per thread).
 */
#ifndef NSD_H
#define NSD_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* element types of floating-point tensors */
enum { NSD_F32 = 0, NSD_F16 = 1, NSD_BF16 = 2 };

const char* nsd_last_error(void);

/* A context on an opened GPU (`ns_gpu*` of ns.h): its stream, a cuBLAS handle on it, scratch buffers. */
void* nsd_create(void* ns_gpu);
/* Waits for the stream, frees the scratch buffers. */
void nsd_destroy(void* ctx);
/* Waits until everything queued has run; reports an asynchronous error if there was one. */
int nsd_wait(void* ctx);

/* A plain linear layer: out = x . w^T + bias. x [M, K] and w [N, K] in the same type dt (float32, half or
 * bfloat16); bias float32 [N] or NULL; out [M, N] in out_dt (float32 or dt). Accumulated in float32. */
int nsd_linear(void* ctx, const void* x, int dt, int64_t M, int64_t K, const void* w, int64_t N, const float* bias,
               void* out, int out_dt);
/* out [M, N] += x [M, K] . W^T (W [N, K]), all in type dt: a LoRA's second factor, accumulated into the output. `ldo`:
 * the elements between the rows of out (0: N) - a matrix's output beside others' in a wider buffer. */
int nsd_linear_acc(void* ctx, const void* x, int dt, int64_t M, int64_t K, const void* w, int64_t N, const float* bias, void* out, int64_t ldo);

/* dequant: n values of a GGUF type into out_dt: 12 = Q4_K, 14 = Q6_K (multiples of 256), 8 = Q8_0 (of 32), 13 = Q5_K,
 * 30 = BF16, 1 = F16, 0 = F32. */
int nsd_dequant(void* ctx, const void* src, int qtype, int64_t n, void* out, int out_dt);

/* out [M, N] = x [M, K] . W^T + bias with W quantized (a GGUF type, rows of K values, N rows): W is dequantized into
 * a scratch of the context in slices of rows, each multiplied as it is made - the weights stay in their stored
 * size on the card, and only a slice is ever half. x in dt (half or bfloat16), out in out_dt. `ldo`: the elements
 * between the rows of out (0: N) - the parts of a fused matrix, each its own type, written side by side. */
int nsd_linear_q(void* ctx, const void* x, int dt, int64_t M, int64_t K, const void* w, int qtype, int64_t N, const float* bias,
                 void* out, int out_dt, int64_t ldo);

/* A linear layer with int8 weights and activations quantized on the fly - nextsycl's int8_linear (ComfyUI's ConvRot):
 *
 *   x_rot = x . H           per group of `group` features, H the normalized regular Hadamard matrix ("ConvRot")
 *   s_r   = max|x_rot[r]| / 127                          per row, at least 1e-30
 *   q     = clamp(round(x_rot / s_r), -128, 127)         int8
 *   acc   = q . W^T                                      int32 - the card's int8 tensor cores
 *   out   = acc * s_r * wscale + bias                    in out_dt
 *
 * x [M, K] in x_dt (half or float32); w int8 [N, K]; wscale float32 [N]; bias float32 [N] or NULL; out [M, N] in
 * out_dt, its rows `ldo` elements apart (0: N). group = 0: no rotation, else a power of 4, at most 256, dividing K. */
int nsd_int8_linear(void* ctx, const void* x, int x_dt, int64_t M, int64_t K, const int8_t* w, int64_t N, const float* wscale,
                    const float* bias, void* out, int out_dt, int group, int64_t ldo);
/* The weights for it from a matrix of any GGUF type (or half / float32 with qtype 1 / 0): rows [row0, row0 + n) of
 * src [N_total, K] are dequantized, rotated by H (along K, groups of `group`), and quantized per row into
 * q int8 [n, K] and scale float32 [n] (scale[r] = max|row| / 127, q = round(row / scale)). */
int nsd_int8_prepare(void* ctx, const void* src, int qtype, int64_t row0, int64_t n, int64_t K, int group, int8_t* q, float* scale);

/* Row-wise RMS norm, optionally followed by a per-row scale and shift picked from a table:
 *   n   = x[r] / sqrt(mean(x[r]^2) + eps) * weight
 *   out = n * (1 + scale[rows[r]]) + shift[rows[r]]        (or out = n when rows, scale or shift is NULL)
 * x, out [M, C]; weight float32 [C] (NULL: 1); rows int32 [M]; scale, shift float32 [R, C]. out may be x. */
int nsd_rms_norm_mod(void* ctx, const void* x, int x_dt, int64_t M, int64_t C, const float* weight, float eps,
                     const int32_t* rows, const float* scale, const float* shift, void* out, int out_dt);

/* Layer norm of x [M, C] per row: (x - mean) / sqrt(var + eps) * weight + bias (weight, bias float32 [C] or NULL). */
int nsd_layer_norm(void* ctx, const void* x, int x_dt, int64_t M, int64_t C, const float* weight, const float* bias, float eps,
                   void* out, int out_dt);

/* GELU in place over n values: exact (erf, mode 0) or the tanh approximation (mode 1). */
int nsd_gelu(void* ctx, void* x, int x_dt, int64_t n, int mode);

/* RoPE alone (no norm), in place, as nsd_rms_rope without its norm; cs float32 [M, rot_dim/2, 2]. */
int nsd_rope(void* ctx, void* x, int x_dt, int64_t M, int64_t H, int64_t D, int64_t stride, const float* cs, int rot_dim);

/* Per-head RMS norm, then the rotary position rotation, in place. x holds M token rows of H heads x D features; row m
 * starts at element m * stride. Each head's D features are normalized (weight float32 [D], NULL: none); then feature
 * pairs (i, rot_dim/2 + i) are rotated by the token's angle: a' = a cos - b sin, b' = b cos + a sin.
 * cs float32 [M, rot_dim/2, 2] holds (cos, sin) per token and pair. Features from rot_dim on are only normalized. */
int nsd_rms_rope(void* ctx, void* x, int x_dt, int64_t M, int64_t H, int64_t D, int64_t stride, const float* weight,
                 float eps, const float* cs, int rot_dim);

/* The gated activation between an MLP's two linears: out[r, i] = silu(x[r, i]) * x[r, C + i]. x [M, 2C], out [M, C]. */
int nsd_swiglu(void* ctx, const void* x, int x_dt, int64_t M, int64_t C, void* out, int out_dt);

/* x[r] += other[r] * gate[rows[r]], in place. x, other [M, C]; rows int32 [M]; gate float32 [R, C]. gate NULL: add. */
int nsd_gate_add(void* ctx, void* x, int x_dt, int64_t M, int64_t C, const void* other, int other_dt,
                 const int32_t* rows, const float* gate);

/* Attention, per head: out = softmax(q . k^T / sqrt(D)) . v, Hq query heads over Hkv key/value heads (head h reads kv
 * head h / (Hq / Hkv)); q rows of stride qs, k/v rows of stride kvs (elements), out [rows, Hq * D] in dt.
 * attention_causal: L rows over L keys, row i over keys [0, i]; attention_causal_rows: query rows [row0, row0 + rows)
 * of q (q points at row 0), over keys [0, i]; attention_qk: Sq query rows (stride qs) over Skv key / value rows, no
 * mask, out in out_dt. The score table is never held whole: a block of query rows at a time. */
int nsd_attention_causal(void* ctx, const void* q, const void* k, const void* v, int dt, int64_t L, int64_t Hq, int64_t Hkv,
                         int64_t D, int64_t qs, int64_t kvs, void* out);
int nsd_attention_causal_rows(void* ctx, const void* q, const void* k, const void* v, int dt, int64_t row0, int64_t rows,
                              int64_t Hq, int64_t Hkv, int64_t D, int64_t qs, int64_t kvs, void* out);
int nsd_attention_qk(void* ctx, const void* q, int64_t Sq, int64_t qs, const void* k, const void* v, int64_t Skv, int64_t kvs,
                     int dt, int64_t H, int64_t D, void* out, int out_dt);

/* conv2d: a k x k convolution of N frames channels-last [N, H, W, Ci] (k odd, zero padding k / 2), w [Co, Ci, k, k]
 * in dt, bias float32 [Co] or NULL; out [N, H, W, Co].
 * conv3d_ex: x [T, H, W, Ci], w [Co, Ci, kt, kh, kw], strided, no padding (out dims (D - k) / s + 1). */
int nsd_conv2d(void* ctx, const void* x, int dt, int64_t N, int64_t H, int64_t W, int64_t Ci, const void* w, int64_t Co, int64_t k,
               const float* bias, void* out);
int nsd_conv3d_ex(void* ctx, const void* x, int dt, int64_t T, int64_t H, int64_t W, int64_t Ci, const void* w, int64_t Co,
                  int64_t kt, int64_t kh, int64_t kw, int64_t st, int64_t sh, int64_t sw, const float* bias, void* out);

/* Attention without a mask: out = softmax(q . k^T / sqrt(D)) . v, per head. q, k, v: S token rows of H heads x D, row s
 * at element s * stride, in dt; out [S, H * D] in out_dt. attention_batch: B independent sequences of S tokens, sequence
 * b's rows from row b * S of q, k, v and out. (nextsycl's; here the blocked cuBLAS attention above.) */
int nsd_attention(void* ctx, const void* q, const void* k, const void* v, int dt, int64_t S, int64_t H, int64_t D,
                  int64_t stride, void* out, int out_dt);
int nsd_attention_batch(void* ctx, const void* q, const void* k, const void* v, int dt, int64_t B, int64_t S, int64_t H,
                        int64_t D, int64_t stride, void* out, int out_dt);
/* The audio decoders' 1-D operations, float32, signals [B, C, L] row-major (an unfolding and one cuBLAS GEMM a tile,
 * float32 without TF32).
 * conv1d: w [Co, Ci, K], zero padding `pad` both sides; out [B, Co, Lo], Lo = (L + 2 pad - dil (K - 1) - 1) / stride + 1.
 * conv_transpose1d: w [Ci, Co, K]; out [B, Co, Lo], Lo = (L - 1) stride - 2 pad + K.
 * snake: x + sin^2(alpha x) / alpha per channel (alpha [C]); scale: x *= s over n values. */
int nsd_conv1d(void* ctx, const float* x, int64_t B, int64_t Ci, int64_t L, const float* w, int64_t Co, int64_t K,
               const float* bias, int64_t stride, int64_t dil, int64_t pad, float* out, int64_t Lo);
int nsd_conv_transpose1d(void* ctx, const float* x, int64_t B, int64_t Ci, int64_t L, const float* w, int64_t Co, int64_t K,
                         const float* bias, int64_t stride, int64_t pad, float* out, int64_t Lo);
int nsd_snake(void* ctx, const float* x, int64_t B, int64_t C, int64_t L, const float* alpha, float* out);
int nsd_scale(void* ctx, float* x, int64_t n, float s);

#ifdef __cplusplus
}
#endif
#endif
