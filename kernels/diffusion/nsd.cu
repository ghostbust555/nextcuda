// nsd.cu: the shared diffusion kernels on CUDA (nsd.h). Matrix products are cuBLAS (tensor cores); the rest are the
// elementwise and row kernels of nextsycl's nsd.cpp, with its arithmetic.
#include "nsd.h"
#include "ns_internal.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

thread_local std::string g_err;
int fail(const std::string& s) { g_err = s; return -1; }

struct Ctx {
    int dev = 0;
    cudaStream_t s = nullptr;
    cublasHandle_t cb = nullptr;
    // scratch buffers (grown, kept): 0 = score table, 1 = probabilities, 2 = im2col / reordered weights,
    // 3 = a slice of dequantized weights
    void* buf[4] = {nullptr, nullptr, nullptr, nullptr};
    size_t cap[4] = {0, 0, 0, 0};
};

int cuda_fail(cudaError_t e, const char* what) {
    return fail(std::string(what) + ": " + cudaGetErrorName(e) + " (" + cudaGetErrorString(e) + ")");
}
int blas_fail(cublasStatus_t st, const char* what) {
    return fail(std::string(what) + ": cuBLAS status " + std::to_string((int) st));
}
#define CU(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) return cuda_fail(e_, #x); } while (0)
#define BL(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) return blas_fail(s_, #x); } while (0)
#define LAUNCHED(what) do { cudaError_t e_ = cudaGetLastError(); if (e_ != cudaSuccess) return cuda_fail(e_, what); } while (0)

void* scratch(Ctx* c, int i, size_t bytes) {
    if (c->cap[i] < bytes) {
        if (c->buf[i]) { cudaStreamSynchronize(c->s); cudaFree(c->buf[i]); c->buf[i] = nullptr; c->cap[i] = 0; }
        if (cudaMalloc(&c->buf[i], bytes) != cudaSuccess) { cudaGetLastError(); return nullptr; }
        c->cap[i] = bytes;
    }
    return c->buf[i];
}

// ---- element access by type ------------------------------------------------------------------------------------

__device__ __forceinline__ float get(const void* p, int dt, int64_t i) {
    if (dt == NSD_F32) return ((const float*) p)[i];
    if (dt == NSD_F16) return __half2float(((const __half*) p)[i]);
    return __bfloat162float(((const __nv_bfloat16*) p)[i]);
}
__device__ __forceinline__ void put(void* p, int dt, int64_t i, float v) {
    if (dt == NSD_F32) ((float*) p)[i] = v;
    else if (dt == NSD_F16) ((__half*) p)[i] = __float2half_rn(v);
    else ((__nv_bfloat16*) p)[i] = __float2bfloat16_rn(v);
}

inline cudaDataType_t cdt(int dt) { return dt == NSD_F32 ? CUDA_R_32F : dt == NSD_F16 ? CUDA_R_16F : CUDA_R_16BF; }
inline size_t esz(int dt) { return dt == NSD_F32 ? 4 : 2; }
inline unsigned nb(int64_t n) { return (unsigned) ((n + 255) / 256); }

constexpr int WG = 256;

__device__ float block_sum(float v) {
    __shared__ float s[WG / 32];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < (int) (blockDim.x / 32); ++i) t += s[i];
    __syncthreads();
    return t;
}
__device__ float block_max(float v) {
    __shared__ float s[WG / 32];
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = s[0];
    for (int i = 1; i < (int) (blockDim.x / 32); ++i) t = fmaxf(t, s[i]);
    __syncthreads();
    return t;
}

// ---- dequantization (GGUF types) ---------------------------------------------------------------------------------

__device__ inline void scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// element i of a tensor in `qtype`
__device__ float dq_at(const uint8_t* src, int qtype, int64_t i) {
    switch (qtype) {
    case 0: return ((const float*) src)[i];
    case 1: return __half2float(((const __half*) src)[i]);
    case 30: return __uint_as_float(((uint32_t) ((const uint16_t*) src)[i]) << 16);
    case 8: {  // Q8_0: half d, int8 qs[32] (34 bytes)
        const uint8_t* b = src + (i / 32) * 34;
        return __half2float(*(const __half*) b) * (float) ((const int8_t*) (b + 2))[i % 32];
    }
    case 12: {  // Q4_K: half d, half dmin, scales[12], qs[128] (144 bytes)
        const uint8_t* b = src + (i / 256) * 144;
        const float d = __half2float(*(const __half*) b), dmin = __half2float(*(const __half*) (b + 2));
        const int r = (int) (i % 256), grp = r / 64, hlf = (r % 64) / 32, l = r % 32;
        uint8_t s, m;
        scale_min_k4(grp * 2 + hlf, b + 4, &s, &m);
        const uint8_t q = (b + 16)[grp * 32 + l];
        return d * s * (hlf ? (q >> 4) : (q & 0xF)) - dmin * m;
    }
    case 13: {  // Q5_K: half d, half dmin, scales[12], qh[32], qs[128] (176 bytes)
        const uint8_t* b = src + (i / 256) * 176;
        const float d = __half2float(*(const __half*) b), dmin = __half2float(*(const __half*) (b + 2));
        const uint8_t* qh = b + 16;
        const uint8_t* qs = b + 48;
        const int r = (int) (i % 256), grp = r / 64, hlf = (r % 64) / 32, l = r % 32;
        uint8_t s, m;
        scale_min_k4(grp * 2 + hlf, b + 4, &s, &m);
        const uint8_t q = qs[grp * 32 + l];
        const int lo = hlf ? (q >> 4) : (q & 0xF);
        const int hi = (qh[l] >> (grp * 2 + hlf)) & 1;
        return d * s * (lo + (hi ? 16 : 0)) - dmin * m;
    }
    case 14: {  // Q6_K: ql[128], qh[64], int8 scales[16], half d (210 bytes)
        const uint8_t* b = src + (i / 256) * 210;
        const uint8_t* ql = b;
        const uint8_t* qh = b + 128;
        const int8_t* sc = (const int8_t*) (b + 192);
        const float d = __half2float(*(const __half*) (b + 208));
        const int r = (int) (i % 256), n128 = r / 128, rem = r % 128, l = rem % 32, quad = rem / 32;
        const int lo = ql[n128 * 64 + l + (quad & 1) * 32];
        const int nib = quad >= 2 ? (lo >> 4) : (lo & 0xF);
        const int hi = (qh[n128 * 32 + l] >> (quad * 2)) & 3;
        return d * (float) sc[n128 * 8 + l / 16 + quad * 2] * (float) ((nib | (hi << 4)) - 32);
    }
    }
    return 0.0f;
}

__global__ void dequant_k(const uint8_t* src, int qtype, int64_t n, void* out, int out_dt) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) put(out, out_dt, i, dq_at(src, qtype, i));
}

// a slice of rows [r0, r0 + rows) of a quantized [N, K] matrix: element j of the slice is element r0 * K + j of it
__global__ void dequant_slice_k(const uint8_t* src, int qtype, int64_t first, int64_t n, void* out, int out_dt) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) put(out, out_dt, i, dq_at(src, qtype, first + i));
}

// ---- elementwise and row kernels -------------------------------------------------------------------------------------

__global__ void bias_k(void* out, int dt, int64_t M, int64_t N, const float* bias) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * N) return;
    put(out, dt, i, get(out, dt, i) + bias[i % N]);
}

// out [M, N] column block: bias on a slice of columns [c0, c0 + n) of a [M, ldo] matrix
__global__ void bias_cols_k(void* out, int dt, int64_t M, int64_t ldo, int64_t c0, int64_t n, const float* bias) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * n) return;
    const int64_t r = i / n, c = i % n;
    put(out, dt, r * ldo + c0 + c, get(out, dt, r * ldo + c0 + c) + bias[c0 + c]);
}

__global__ void rms_norm_mod_k(const void* x, int xdt, int64_t C, const float* w, float eps, const int32_t* rows, const float* scale,
                               const float* shift, void* out, int odt) {
    const int64_t r = blockIdx.x;
    float s = 0.0f;
    for (int64_t i = threadIdx.x; i < C; i += blockDim.x) { const float v = get(x, xdt, r * C + i); s += v * v; }
    const float inv = rsqrtf(block_sum(s) / (float) C + eps);
    const int64_t mr = (rows && scale && shift) ? rows[r] : -1;
    for (int64_t i = threadIdx.x; i < C; i += blockDim.x) {
        float n = get(x, xdt, r * C + i) * inv * (w ? w[i] : 1.0f);
        if (mr >= 0) n = n * (1.0f + scale[mr * C + i]) + shift[mr * C + i];
        put(out, odt, r * C + i, n);
    }
}

__global__ void rms_rope_k(void* x, int dt, int64_t H, int64_t D, int64_t stride, const float* w, float eps, const float* cs, int rot) {
    extern __shared__ float sh[];  // D floats
    const int64_t m = blockIdx.x, h = blockIdx.y;
    const int64_t base = m * stride + h * D;
    float s = 0.0f;
    for (int64_t i = threadIdx.x; i < D; i += blockDim.x) { const float v = get(x, dt, base + i); sh[i] = v; s += v * v; }
    const float inv = rsqrtf(block_sum(s) / (float) D + eps);
    __syncthreads();
    for (int64_t i = threadIdx.x; i < D; i += blockDim.x) sh[i] = sh[i] * inv * (w ? w[i] : 1.0f);
    __syncthreads();
    const int half = rot / 2;
    for (int64_t i = threadIdx.x; i < D; i += blockDim.x) {
        float v = sh[i];
        if (rot > 0 && i < rot) {
            const int p = (int) (i % half);
            const float c = cs[(m * half + p) * 2], sn = cs[(m * half + p) * 2 + 1];
            const float a = sh[p], b = sh[half + p];
            v = i < half ? a * c - b * sn : b * c + a * sn;
        }
        put(x, dt, base + i, v);
    }
}

__global__ void swiglu_k(const void* x, int xdt, int64_t C, void* out, int odt, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t r = i / C, c = i % C;
    const float g = get(x, xdt, r * 2 * C + c), u = get(x, xdt, r * 2 * C + C + c);
    put(out, odt, i, g / (1.0f + expf(-g)) * u);
}

__global__ void gate_add_k(void* x, int xdt, int64_t C, const void* other, int odt, const int32_t* rows, const float* gate, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t r = i / C, c = i % C;
    const float g = gate ? gate[(rows ? (int64_t) rows[r] : 0) * C + c] : 1.0f;
    put(x, xdt, i, get(x, xdt, i) + get(other, odt, i) * g);
}

// softmax over rows of the score table [Hq, R, Skv] (float32, already scaled), with the causal limit: row (h, r) of the
// chunk starting at query row q0 may read keys j <= row0 + q0 + r; probabilities in dt
__global__ void softmax_k(const float* s, void* p, int dt, int64_t R, int64_t Skv, int64_t row0, int64_t q0, int causal) {
    const int64_t row = blockIdx.x;                 // over Hq * R
    const int64_t r = row % R;
    const int64_t lim = causal ? min(Skv - 1, row0 + q0 + r) : Skv - 1;
    const float* x = s + row * Skv;
    float m = -INFINITY;
    for (int64_t j = threadIdx.x; j <= lim; j += blockDim.x) m = fmaxf(m, x[j]);
    m = block_max(m);
    float sum = 0.0f;
    for (int64_t j = threadIdx.x; j <= lim; j += blockDim.x) sum += expf(x[j] - m);
    const float inv = 1.0f / block_sum(sum);
    for (int64_t j = threadIdx.x; j < Skv; j += blockDim.x) put(p, dt, row * Skv + j, j <= lim ? expf(x[j] - m) * inv : 0.0f);
}

// im2col for conv2d: col [P, k*k*Ci] for pixels [p0, p0 + P) of N*H*W, zero padded (k / 2)
__global__ void im2col2d_k(const void* x, int dt, int64_t H, int64_t W, int64_t Ci, int64_t k, int64_t p0, int64_t P, void* col) {
    const int64_t KK = k * k * Ci;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P * KK) return;
    const int64_t p = i / KK, j = i % KK;
    const int64_t ci = j % Ci, t = j / Ci, kx = t % k, ky = t / k;
    const int64_t g = p0 + p, xx = g % W, y = (g / W) % H, n = g / (W * H);
    const int64_t yy = y + ky - k / 2, x2 = xx + kx - k / 2;
    float v = 0.0f;
    if (yy >= 0 && yy < H && x2 >= 0 && x2 < W) v = get(x, dt, ((n * H + yy) * W + x2) * Ci + ci);
    put(col, dt, i, v);
}

// w [Co, Ci, k, k] (PyTorch) -> [Co, (ky, kx, ci)]
__global__ void reorder_w_k(const void* w, int dt, int64_t Co, int64_t Ci, int64_t k, void* out) {
    const int64_t KK = k * k * Ci;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Co * KK) return;
    const int64_t co = i / KK, j = i % KK;
    const int64_t ci = j % Ci, t = j / Ci, kx = t % k, ky = t / k;
    put(out, dt, i, get(w, dt, ((co * Ci + ci) * k + ky) * k + kx));
}

// conv3d, strided, no padding: col [P, kt*kh*kw*Ci] for output voxels [p0, p0 + P) of [To, Ho, Wo]
__global__ void im2col3d_k(const void* x, int dt, int64_t H, int64_t W, int64_t Ci, int64_t kt, int64_t kh, int64_t kw, int64_t st,
                           int64_t sh_, int64_t sw, int64_t Ho, int64_t Wo, int64_t p0, int64_t P, void* col) {
    const int64_t KK = kt * kh * kw * Ci;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P * KK) return;
    const int64_t p = i / KK, j = i % KK;
    const int64_t ci = j % Ci, t = j / Ci, ix = t % kw, iy = (t / kw) % kh, it = t / (kw * kh);
    const int64_t g = p0 + p, ox = g % Wo, oy = (g / Wo) % Ho, ot = g / (Wo * Ho);
    put(col, dt, i, get(x, dt, (((ot * st + it) * H + oy * sh_ + iy) * W + ox * sw + ix) * Ci + ci));
}

__global__ void reorder_w3d_k(const void* w, int dt, int64_t Co, int64_t Ci, int64_t kt, int64_t kh, int64_t kw, void* out) {
    const int64_t KK = kt * kh * kw * Ci;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Co * KK) return;
    const int64_t co = i / KK, j = i % KK;
    const int64_t ci = j % Ci, t = j / Ci, ix = t % kw, iy = (t / kw) % kh, it = t / (kw * kh);
    put(out, dt, i, get(w, dt, ((((co * Ci + ci) * kt + it) * kh + iy) * kw) + ix));
}

// out [M, N] (+)= x [M, K] . w [N, K]^T, float32 accumulation; `ldo`: the row stride of out (>= N)
int gemm_nt(Ctx* c, const void* x, int xdt, int64_t M, int64_t K, const void* w, int wdt, int64_t N, void* out, int odt, float beta, int64_t ldo = 0) {
    const float alpha = 1.0f;
    BL(cublasGemmEx(c->cb, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) M, (int) K, &alpha, w, cdt(wdt), (int) K, x, cdt(xdt), (int) K,
                    &beta, out, cdt(odt), (int) (ldo ? ldo : N), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    return 0;
}

// the attention core: query rows [0, Sq) of q (already offset by the caller) over Skv keys, in chunks
int attention(Ctx* c, const void* q, int64_t Sq, int64_t qs, const void* k, const void* v, int64_t Skv, int64_t kvs, int dt, int64_t Hq,
              int64_t Hkv, int64_t D, void* out, int odt, int causal, int64_t row0) {
    if (Hq % Hkv) return fail("attention: query heads are not a multiple of the key/value heads");
    const int64_t g = Hq / Hkv;
    const float scale = 1.0f / sqrtf((float) D);
    // a block of query rows: Hq x R x Skv float32 scores under ~256 MiB
    int64_t R = std::max<int64_t>(1, ((256ll << 20) / 4) / (Hq * Skv));
    R = std::min(R, Sq);
    float* S = (float*) scratch(c, 0, (size_t) Hq * R * Skv * 4);
    void* P = scratch(c, 1, (size_t) Hq * R * Skv * esz(dt));
    if (!S || !P) return fail("attention: the score table did not fit in memory");
    const float one = 1.0f, zero = 0.0f;
    const size_t eq = esz(dt);
    for (int64_t q0 = 0; q0 < Sq; q0 += R) {
        const int64_t r = std::min(R, Sq - q0);
        const char* qc = (const char*) q + (size_t) q0 * qs * eq;
        // scores: for each kv head, its g query heads (batched; K shared when g > 1)
        for (int64_t j = 0; j < Hkv; ++j) {
            const char* kj = (const char*) k + (size_t) j * D * eq;
            const char* qj = qc + (size_t) j * g * D * eq;
            float* sj = S + (size_t) j * g * r * Skv;
            BL(cublasGemmStridedBatchedEx(c->cb, CUBLAS_OP_T, CUBLAS_OP_N, (int) Skv, (int) r, (int) D, &scale, kj, cdt(dt), (int) kvs, 0,
                                          qj, cdt(dt), (int) qs, (long long) D, &zero, sj, CUDA_R_32F, (int) Skv, (long long) r * Skv,
                                          (int) g, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        softmax_k<<<(unsigned) (Hq * r), WG, 0, c->s>>>(S, P, dt, r, Skv, row0, q0, causal);
        LAUNCHED("attention softmax");
        for (int64_t j = 0; j < Hkv; ++j) {
            const char* vj = (const char*) v + (size_t) j * D * eq;
            const char* pj = (const char*) P + (size_t) j * g * r * Skv * eq;
            char* oj = (char*) out + ((size_t) q0 * Hq * D + (size_t) j * g * D) * esz(odt);
            // out^T [D, r] = V^T [D, Skv] . P^T [Skv, r], per query head (V shared across the g heads of a kv head)
            BL(cublasGemmStridedBatchedEx(c->cb, CUBLAS_OP_N, CUBLAS_OP_N, (int) D, (int) r, (int) Skv, &one, vj, cdt(dt), (int) kvs, 0,
                                          pj, cdt(dt), (int) Skv, (long long) r * Skv, &zero, oj, cdt(odt), (int) (Hq * D), (long long) D,
                                          (int) g, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
    }
    return 0;
}

}  // namespace

extern "C" {

const char* nsd_last_error(void) { return g_err.c_str(); }

void* nsd_create(void* gpu) {
    auto* g = (ns_gpu*) gpu;
    if (!g) { fail("nsd_create: no GPU"); return nullptr; }
    cudaSetDevice(g->dev);
    auto* c = new Ctx();
    c->dev = g->dev;
    c->s = g->q;
    if (cublasCreate(&c->cb) != CUBLAS_STATUS_SUCCESS) { delete c; fail("nsd_create: cublasCreate failed"); return nullptr; }
    cublasSetStream(c->cb, c->s);
    cublasSetMathMode(c->cb, CUBLAS_DEFAULT_MATH);
    return c;
}

void nsd_destroy(void* ctx) {
    auto* c = (Ctx*) ctx;
    if (!c) return;
    cudaSetDevice(c->dev);
    cudaStreamSynchronize(c->s);
    for (void* b : c->buf) if (b) cudaFree(b);
    cublasDestroy(c->cb);
    delete c;
}

int nsd_wait(void* ctx) {
    auto* c = (Ctx*) ctx;
    CU(cudaSetDevice(c->dev));
    CU(cudaStreamSynchronize(c->s));
    CU(cudaGetLastError());
    return 0;
}

int nsd_linear(void* ctx, const void* x, int dt, int64_t M, int64_t K, const void* w, int64_t N, const float* bias, void* out, int out_dt) {
    auto* c = (Ctx*) ctx;
    CU(cudaSetDevice(c->dev));
    if (int rc = gemm_nt(c, x, dt, M, K, w, dt, N, out, out_dt, 0.0f)) return rc;
    if (bias) { bias_k<<<nb(M * N), 256, 0, c->s>>>(out, out_dt, M, N, bias); LAUNCHED("bias"); }
    return 0;
}

int nsd_linear_acc(void* ctx, const void* x, int dt, int64_t M, int64_t K, const void* w, int64_t N, const float* bias, void* out) {
    auto* c = (Ctx*) ctx;
    CU(cudaSetDevice(c->dev));
    if (int rc = gemm_nt(c, x, dt, M, K, w, dt, N, out, dt, 1.0f)) return rc;
    if (bias) { bias_k<<<nb(M * N), 256, 0, c->s>>>(out, dt, M, N, bias); LAUNCHED("bias"); }
    return 0;
}

int nsd_dequant(void* ctx, const void* src, int qtype, int64_t n, void* out, int out_dt) {
    auto* c = (Ctx*) ctx;
    CU(cudaSetDevice(c->dev));
    if (qtype != 0 && qtype != 1 && qtype != 8 && qtype != 12 && qtype != 13 && qtype != 14 && qtype != 30)
        return fail("nsd_dequant: GGUF type " + std::to_string(qtype) + " is not read");
    if ((qtype == 12 || qtype == 13 || qtype == 14) && n % 256) return fail("nsd_dequant: a K-quant tensor is a multiple of 256 values");
    if (qtype == 8 && n % 32) return fail("nsd_dequant: Q8_0 is a multiple of 32 values");
    dequant_k<<<nb(n), 256, 0, c->s>>>((const uint8_t*) src, qtype, n, out, out_dt);
    LAUNCHED("dequant");
    return 0;
}

// bytes of a row of K values of a GGUF type, and the values a block holds
static bool row_geometry(int qtype, int64_t K, int64_t* row_bytes) {
    switch (qtype) {
    case 0: *row_bytes = K * 4; return true;
    case 1: case 30: *row_bytes = K * 2; return true;
    case 8: if (K % 32) return false; *row_bytes = K / 32 * 34; return true;
    case 12: if (K % 256) return false; *row_bytes = K / 256 * 144; return true;
    case 13: if (K % 256) return false; *row_bytes = K / 256 * 176; return true;
    case 14: if (K % 256) return false; *row_bytes = K / 256 * 210; return true;
    }
    return false;
}

int nsd_linear_q(void* ctx, const void* x, int dt, int64_t M, int64_t K, const void* w, int qtype, int64_t N, const float* bias, void* out,
                 int out_dt, int64_t ldo) {
    auto* c = (Ctx*) ctx;
    CU(cudaSetDevice(c->dev));
    int64_t rb = 0;
    if (!row_geometry(qtype, K, &rb)) return fail("nsd_linear_q: a row of " + std::to_string(K) + " values is not whole blocks of GGUF type " + std::to_string(qtype));
    if (dt == NSD_F32) return fail("nsd_linear_q: activations in half or bfloat16");
    // a slice of rows dequantized to `dt` at a time: about 64 MiB of them
    int64_t rows = std::max<int64_t>(1, (64ll << 20) / (K * (int64_t) esz(dt)));
    rows = std::min(rows, N);
    void* wt = scratch(c, 3, (size_t) rows * K * esz(dt));
    if (!wt) return fail("nsd_linear_q: the weight slice did not fit in memory");
    for (int64_t r0 = 0; r0 < N; r0 += rows) {
        const int64_t n = std::min(rows, N - r0);
        // the slice's first value is element r0 * K of the matrix: whole blocks, because K is a multiple of the block
        dequant_slice_k<<<nb(n * K), 256, 0, c->s>>>((const uint8_t*) w, qtype, r0 * K, n * K, wt, dt);
        LAUNCHED("dequant slice");
        char* o = (char*) out + (size_t) r0 * esz(out_dt);
        if (int rc = gemm_nt(c, x, dt, M, K, wt, dt, n, o, out_dt, 0.0f, ldo ? ldo : N)) return rc;
        if (bias) { bias_cols_k<<<nb(M * n), 256, 0, c->s>>>(out, out_dt, M, ldo ? ldo : N, r0, n, bias); LAUNCHED("bias"); }
    }
    return 0;
}

int nsd_rms_norm_mod(void* ctx, const void* x, int x_dt, int64_t M, int64_t C, const float* weight, float eps, const int32_t* rows, const float* scale,
                     const float* shift, void* out, int out_dt) {
    auto* c = (Ctx*) ctx;
    CU(cudaSetDevice(c->dev));
    rms_norm_mod_k<<<(unsigned) M, WG, 0, c->s>>>(x, x_dt, C, weight, eps, rows, scale, shift, out, out_dt);
    LAUNCHED("rms_norm_mod");
    return 0;
}

int nsd_rms_rope(void* ctx, void* x, int x_dt, int64_t M, int64_t H, int64_t D, int64_t stride, const float* weight, float eps, const float* cs,
                 int rot_dim) {
    auto* c = (Ctx*) ctx;
    CU(cudaSetDevice(c->dev));
    if (rot_dim > D || rot_dim % 2) return fail("nsd_rms_rope: rot_dim must be even and at most D");
    dim3 grid((unsigned) M, (unsigned) H);
    rms_rope_k<<<grid, 128, (size_t) D * 4, c->s>>>(x, x_dt, H, D, stride, weight, eps, cs, rot_dim);
    LAUNCHED("rms_rope");
    return 0;
}

int nsd_swiglu(void* ctx, const void* x, int x_dt, int64_t M, int64_t C, void* out, int out_dt) {
    auto* c = (Ctx*) ctx;
    CU(cudaSetDevice(c->dev));
    swiglu_k<<<nb(M * C), 256, 0, c->s>>>(x, x_dt, C, out, out_dt, M * C);
    LAUNCHED("swiglu");
    return 0;
}

int nsd_gate_add(void* ctx, void* x, int x_dt, int64_t M, int64_t C, const void* other, int other_dt, const int32_t* rows, const float* gate) {
    auto* c = (Ctx*) ctx;
    CU(cudaSetDevice(c->dev));
    gate_add_k<<<nb(M * C), 256, 0, c->s>>>(x, x_dt, C, other, other_dt, rows, gate, M * C);
    LAUNCHED("gate_add");
    return 0;
}

int nsd_attention_causal(void* ctx, const void* q, const void* k, const void* v, int dt, int64_t L, int64_t Hq, int64_t Hkv, int64_t D,
                         int64_t qs, int64_t kvs, void* out) {
    auto* c = (Ctx*) ctx;
    CU(cudaSetDevice(c->dev));
    return attention(c, q, L, qs, k, v, L, kvs, dt, Hq, Hkv, D, out, dt, 1, 0);
}

int nsd_attention_causal_rows(void* ctx, const void* q, const void* k, const void* v, int dt, int64_t row0, int64_t rows, int64_t Hq,
                              int64_t Hkv, int64_t D, int64_t qs, int64_t kvs, void* out) {
    auto* c = (Ctx*) ctx;
    CU(cudaSetDevice(c->dev));
    const char* qr = (const char*) q + (size_t) row0 * qs * esz(dt);
    return attention(c, qr, rows, qs, k, v, row0 + rows, kvs, dt, Hq, Hkv, D, out, dt, 1, row0);
}

int nsd_attention_qk(void* ctx, const void* q, int64_t Sq, int64_t qs, const void* k, const void* v, int64_t Skv, int64_t kvs, int dt,
                     int64_t H, int64_t D, void* out, int out_dt) {
    auto* c = (Ctx*) ctx;
    CU(cudaSetDevice(c->dev));
    return attention(c, q, Sq, qs, k, v, Skv, kvs, dt, H, H, D, out, out_dt, 0, 0);
}

int nsd_conv2d(void* ctx, const void* x, int dt, int64_t N, int64_t H, int64_t W, int64_t Ci, const void* w, int64_t Co, int64_t k, const float* bias,
               void* out) {
    auto* c = (Ctx*) ctx;
    CU(cudaSetDevice(c->dev));
    const int64_t px = N * H * W;
    if (k == 1) {
        if (int rc = gemm_nt(c, x, dt, px, Ci, w, dt, Co, out, dt, 0.0f)) return rc;
        if (bias) { bias_k<<<nb(px * Co), 256, 0, c->s>>>(out, dt, px, Co, bias); LAUNCHED("bias"); }
        return 0;
    }
    const int64_t KK = k * k * Ci;
    const int64_t P = std::max<int64_t>(1, std::min<int64_t>(px, (192ll << 20) / (KK * (int64_t) esz(dt))));
    const size_t wbytes = (size_t) Co * KK * esz(dt);
    char* wr = (char*) scratch(c, 2, wbytes + (size_t) P * KK * esz(dt));
    if (!wr) return fail("nsd_conv2d: the im2col buffer did not fit in memory");
    char* col = wr + wbytes;
    reorder_w_k<<<nb(Co * KK), 256, 0, c->s>>>(w, dt, Co, Ci, k, wr);
    LAUNCHED("conv2d weights");
    for (int64_t p0 = 0; p0 < px; p0 += P) {
        const int64_t r = std::min(P, px - p0);
        im2col2d_k<<<nb(r * KK), 256, 0, c->s>>>(x, dt, H, W, Ci, k, p0, r, col);
        LAUNCHED("conv2d im2col");
        char* o = (char*) out + (size_t) p0 * Co * esz(dt);
        if (int rc = gemm_nt(c, col, dt, r, KK, wr, dt, Co, o, dt, 0.0f)) return rc;
        if (bias) { bias_k<<<nb(r * Co), 256, 0, c->s>>>(o, dt, r, Co, bias); LAUNCHED("bias"); }
    }
    return 0;
}

int nsd_conv3d_ex(void* ctx, const void* x, int dt, int64_t T, int64_t H, int64_t W, int64_t Ci, const void* w, int64_t Co, int64_t kt, int64_t kh,
                  int64_t kw, int64_t st, int64_t sh, int64_t sw, const float* bias, void* out) {
    auto* c = (Ctx*) ctx;
    CU(cudaSetDevice(c->dev));
    const int64_t To = (T - kt) / st + 1, Ho = (H - kh) / sh + 1, Wo = (W - kw) / sw + 1;
    const int64_t px = To * Ho * Wo, KK = kt * kh * kw * Ci;
    const int64_t P = std::max<int64_t>(1, std::min<int64_t>(px, (192ll << 20) / (KK * (int64_t) esz(dt))));
    const size_t wbytes = (size_t) Co * KK * esz(dt);
    char* wr = (char*) scratch(c, 2, wbytes + (size_t) P * KK * esz(dt));
    if (!wr) return fail("nsd_conv3d_ex: the im2col buffer did not fit in memory");
    char* col = wr + wbytes;
    reorder_w3d_k<<<nb(Co * KK), 256, 0, c->s>>>(w, dt, Co, Ci, kt, kh, kw, wr);
    LAUNCHED("conv3d weights");
    for (int64_t p0 = 0; p0 < px; p0 += P) {
        const int64_t r = std::min(P, px - p0);
        im2col3d_k<<<nb(r * KK), 256, 0, c->s>>>(x, dt, H, W, Ci, kt, kh, kw, st, sh, sw, Ho, Wo, p0, r, col);
        LAUNCHED("conv3d im2col");
        char* o = (char*) out + (size_t) p0 * Co * esz(dt);
        if (int rc = gemm_nt(c, col, dt, r, KK, wr, dt, Co, o, dt, 0.0f)) return rc;
        if (bias) { bias_k<<<nb(r * Co), 256, 0, c->s>>>(o, dt, r, Co, bias); LAUNCHED("bias"); }
    }
    return 0;
}

}  // extern "C"
