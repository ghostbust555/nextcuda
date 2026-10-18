// q35_test.cu: the qwen35 kernels (q35.h) and the products' Q2_K / Q6_K additions (nh.h) against plain CPU references
// on random data, then the attention's and the scan's times. Not part of the library: test/run.sh builds it against
// libnextcuda-llm.so. Prints each check's max error and PASS / FAIL; exits 1 on a failure.
#include "ns.h"
#include "nh.h"
#include "q35.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

static ns_gpu* G;
static int fails = 0;
static std::mt19937 rng(1234);

static void ok(int rc, const char* what) {
    if (rc != 0) {
        std::fprintf(stderr, "%s: %s\n", what, ns_last_error());
        std::exit(2);
    }
}
static std::vector<float> rnd(size_t n, float lo = -1.0f, float hi = 1.0f) {
    std::uniform_real_distribution<float> u(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = u(rng);
    return v;
}
template <typename T>
static T* up(const std::vector<T>& h) {
    void* p;
    ok(ns_alloc(G, h.size() * sizeof(T) + 16, &p), "alloc");
    ok(ns_copy_to(G, p, h.data(), h.size() * sizeof(T)), "copy to");
    return (T*) p;
}
static int GPU = 0;
static void* dev(size_t bytes) {
    void* p;
    ok(ns_alloc(G, bytes + 16, &p), "alloc");
    // zeroed before any queued work reads it: the memset on this device, finished here
    cudaSetDevice(GPU);
    cudaMemset(p, 0, bytes + 16);
    cudaDeviceSynchronize();
    return p;
}
template <typename T>
static std::vector<T> down(const void* p, size_t n) {
    std::vector<T> h(n);
    ok(ns_sync(G), "sync");
    ok(ns_copy_from(G, h.data(), p, n * sizeof(T)), "copy from");
    return h;
}
// max |a - b| / (1 + |b|)
static void check(const char* name, const std::vector<float>& a, const std::vector<float>& b, double tol) {
    double m = 0;
    for (size_t i = 0; i < b.size(); ++i) {
        const double e = std::fabs((double) a[i] - b[i]) / (1.0 + std::fabs((double) b[i]));
        if (!(e <= m)) m = std::isnan(e) ? INFINITY : std::max(m, e);
    }
    const bool pass = m <= tol;
    if (!pass) ++fails;
    std::printf("%-44s max err %.3g (tol %.1g)  %s\n", name, m, tol, pass ? "PASS" : "FAIL");
}
static float h2f(uint16_t h) {
    __half x;
    std::memcpy(&x, &h, 2);
    return __half2float(x);
}
static uint16_t f2h(float f) {
    __half x = __float2half(f);
    uint16_t h;
    std::memcpy(&h, &x, 2);
    return h;
}
static float hround(float f) { return h2f(f2h(f)); }

// ---- Q2_K / Q6_K (ggml's dequantize_row_q2_K / _q6_K)
struct BQ2 { uint8_t scales[16]; uint8_t qs[64]; uint16_t d, dmin; };
struct BQ6 { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; };
static_assert(sizeof(BQ2) == 84 && sizeof(BQ6) == 210, "block sizes");

static void deq_q2(const BQ2* b, int64_t nb, float* y) {
    for (int64_t i = 0; i < nb; ++i) {
        const float d = h2f(b[i].d), mn = h2f(b[i].dmin);
        const uint8_t* q = b[i].qs;
        int is = 0;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = b[i].scales[is++];
                float dl = d * (sc & 0xF), ml = mn * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t) ((q[l] >> shift) & 3)) - ml;
                sc = b[i].scales[is++];
                dl = d * (sc & 0xF), ml = mn * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t) ((q[l + 16] >> shift) & 3)) - ml;
                shift += 2;
            }
            q += 32;
        }
    }
}
static void deq_q6(const BQ6* b, int64_t nb, float* y) {
    for (int64_t i = 0; i < nb; ++i) {
        const float d = h2f(b[i].d);
        const uint8_t* ql = b[i].ql;
        const uint8_t* qh = b[i].qh;
        const int8_t* sc = b[i].scales;
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int8_t q1 = (int8_t) ((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t) ((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t) ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}
// x as the products see it: Q8_1 blocks of 32 (d = amax / 127 as half, q = round(x / d))
static std::vector<float> q8_1_view(const std::vector<float>& x) {
    std::vector<float> o(x.size());
    for (size_t b = 0; b < x.size(); b += 32) {
        float am = 0;
        for (int i = 0; i < 32; ++i) am = std::max(am, std::fabs(x[b + i]));
        const float d = am / 127.0f, id = d > 0 ? 1.0f / d : 0.0f;
        for (int i = 0; i < 32; ++i) o[b + i] = std::round(x[b + i] * id) * hround(d);
    }
    return o;
}

static void test_quants() {
    const int64_t n_in = 1024, n_out = 96, nb = n_in / 256;
    std::uniform_int_distribution<int> byte(0, 255);
    // Q2_K
    std::vector<BQ2> w2((size_t) (n_out * nb));
    for (auto& b : w2) {
        for (auto& s : b.scales) s = (uint8_t) byte(rng);
        for (auto& q : b.qs) q = (uint8_t) byte(rng);
        b.d = f2h(rnd(1, 0.001f, 0.02f)[0]);
        b.dmin = f2h(rnd(1, 0.001f, 0.02f)[0]);
    }
    std::vector<float> ref((size_t) (n_out * n_in));
    deq_q2(w2.data(), n_out * nb, ref.data());
    void* W2 = up(std::vector<uint8_t>((uint8_t*) w2.data(), (uint8_t*) (w2.data() + w2.size())));
    std::printf("Q2_K row bytes %zu (want %lld), supported %d\n", ns_nh_row_bytes(10, n_in), (long long) (nb * 84), ns_nh_mmvq_supported(10));
    if (ns_nh_row_bytes(10, n_in) != (size_t) (nb * 84) || !ns_nh_mmvq_supported(10)) ++fails;
    float* y32 = (float*) dev(ref.size() * 4);
    ok(ns_nh_dequant_f32(G, 10, W2, (int64_t) ref.size(), y32), "q2 dequant f32");
    check("Q2_K dequant f32", down<float>(y32, ref.size()), ref, 1e-6);
    void* y16 = dev(ref.size() * 2);
    ok(ns_nh_dequant_f16(G, 10, W2, (int64_t) ref.size(), y16), "q2 dequant f16");
    {
        auto h = down<uint16_t>(y16, ref.size());
        std::vector<float> f(h.size());
        for (size_t i = 0; i < h.size(); ++i) f[i] = h2f(h[i]);
        check("Q2_K dequant f16", f, ref, 2e-3);
    }
    // the product: 3 activation rows
    const int ncols = 3;
    auto x = rnd((size_t) (ncols * n_in));
    float* X = up(x);
    void* XQ = dev(ns_nh_q8_1_bytes(n_in, ncols));
    ok(ns_nh_quantize(G, X, ncols, n_in, XQ), "quantize");
    float* Y = (float*) dev((size_t) (ncols * n_out) * 4);
    ok(ns_nh_mmvq(G, 10, W2, XQ, Y, n_in, n_out, ncols), "q2 mmvq");
    std::vector<float> want((size_t) (ncols * n_out));
    for (int c = 0; c < ncols; ++c) {
        auto xv = q8_1_view(std::vector<float>(x.begin() + c * n_in, x.begin() + (c + 1) * n_in));
        for (int64_t r = 0; r < n_out; ++r) {
            double s = 0;
            for (int64_t i = 0; i < n_in; ++i) s += (double) ref[r * n_in + i] * xv[i];
            want[c * n_out + r] = (float) s;
        }
    }
    check("Q2_K mmvq (vs dequant . q8_1(x))", down<float>(Y, want.size()), want, 2e-4);
    // the experts' form: two entries, both the same matrix, rows 2 and 0 of x
    {
        const unsigned long long ptrs[2] = {(unsigned long long) W2, (unsigned long long) W2};
        void* P = up(std::vector<unsigned long long>(ptrs, ptrs + 2));
        int32_t* rows = up(std::vector<int32_t>{2, 0});
        float* Y2 = (float*) dev(2 * n_out * 4);
        ok(ns_nh_mmvq_ptrs(G, 10, P, rows, XQ, Y2, n_in, n_out, 2), "q2 mmvq ptrs");
        std::vector<float> w2r(want.begin() + 2 * n_out, want.begin() + 3 * n_out);
        w2r.insert(w2r.end(), want.begin(), want.begin() + n_out);
        check("Q2_K mmvq_ptrs", down<float>(Y2, 2 * n_out), w2r, 2e-4);
    }
    // Q6_K's dequantizers (f32 now the dispatch's too; f16 the nh kernel's)
    std::vector<BQ6> w6((size_t) (n_out * nb));
    for (auto& b : w6) {
        for (auto& q : b.ql) q = (uint8_t) byte(rng);
        for (auto& q : b.qh) q = (uint8_t) byte(rng);
        for (auto& s : b.scales) s = (int8_t) (byte(rng) - 128);
        b.d = f2h(rnd(1, 0.0001f, 0.002f)[0]);
    }
    std::vector<float> ref6((size_t) (n_out * n_in));
    deq_q6(w6.data(), n_out * nb, ref6.data());
    void* W6 = up(std::vector<uint8_t>((uint8_t*) w6.data(), (uint8_t*) (w6.data() + w6.size())));
    ok(ns_nh_dequant_f32(G, 14, W6, (int64_t) ref6.size(), y32), "q6 dequant f32");
    check("Q6_K dequant f32", down<float>(y32, ref6.size()), ref6, 1e-6);
    ok(ns_nh_dequant_f16(G, 14, W6, (int64_t) ref6.size(), y16), "q6 dequant f16");
    {
        auto h = down<uint16_t>(y16, ref6.size());
        std::vector<float> f(h.size());
        for (size_t i = 0; i < h.size(); ++i) f[i] = h2f(h[i]);
        check("Q6_K dequant f16", f, ref6, 2e-3);
    }
}

// ---- attention: kv_store + attn against a double-precision softmax (the cache's halves as the reference's keys)
// q8: the q8_0 cache (ns_q35_kv_store_q8 / ns_q35_attn_q8): the store against ggml's quantize_f32_q8_0_block, the
// reference over the cache as stored
static void test_attn(int64_t T, int64_t p0, int64_t Hq, int64_t Hkv, bool q8 = false) {
    const int64_t D = 256, cap = p0 + T + 3, all = p0 + T, RB = q8 ? D + 16 : D * 2;
    auto k = rnd((size_t) (all * Hkv * D)), v = rnd((size_t) (all * Hkv * D));
    // q rows strided: qs = Hq * D + 64 (as the engine's fused projections leave them)
    const int64_t qs = Hq * D + 64;
    auto q = rnd((size_t) (T * qs), -2.0f, 2.0f);
    float* K = up(k);
    float* V = up(v);
    float* Q = up(q);
    void* KC = dev((size_t) (Hkv * cap * RB));
    void* VC = dev((size_t) (Hkv * cap * RB));
    auto store = q8 ? ns_q35_kv_store_q8 : ns_q35_kv_store;
    auto attn = q8 ? ns_q35_attn_q8 : ns_q35_attn;
    // the cache in two stores: the earlier keys, then the rows' own (as decode does)
    if (p0 > 0) ok(store(G, K, Hkv * D, V, Hkv * D, p0, Hkv, D, cap, 0, KC, VC), "kv_store");
    ok(store(G, K + p0 * Hkv * D, Hkv * D, V + p0 * Hkv * D, Hkv * D, T, Hkv, D, cap, p0, KC, VC), "kv_store");
    const int64_t sc = ns_q35_attn_scratch(T, Hq, Hkv, D, p0);
    float* S = sc > 0 ? (float*) dev((size_t) sc * 4) : nullptr;
    float* O = (float*) dev((size_t) (T * Hq * D) * 4);
    ok(attn(G, Q, qs, KC, VC, T, Hq, Hkv, D, cap, p0, O, S), "attn");
    // the cache back as halves [Hkv][cap][D] (a q8 row widened: its int8 times its block's half scale): the store's
    // check, and the reference's keys
    std::vector<uint16_t> kc, vc;
    if (!q8) {
        kc = down<uint16_t>(KC, (size_t) (Hkv * cap * D));
        vc = down<uint16_t>(VC, (size_t) (Hkv * cap * D));
    } else {
        auto kb = down<uint8_t>(KC, (size_t) (Hkv * cap * RB)), vb = down<uint8_t>(VC, (size_t) (Hkv * cap * RB));
        std::vector<float> got, want;
        for (auto* pr : {&kb, &vb}) {
            auto& out = pr == &kb ? kc : vc;
            const auto& src = pr == &kb ? k : v;
            out.assign((size_t) (Hkv * cap * D), 0);
            for (int64_t t = 0; t < all; ++t)
                for (int64_t h = 0; h < Hkv; ++h) {
                    const uint8_t* row = pr->data() + (h * cap + t) * RB;
                    for (int64_t d = 0; d < D; ++d) {
                        uint16_t sh;
                        std::memcpy(&sh, row + D + (d / 32) * 2, 2);
                        const float x = (float) (int8_t) row[d] * h2f(sh);
                        out[(h * cap + t) * D + d] = f2h(x);
                        // ggml's block: d = amax / 127, q = round(x / d), the scale kept as half
                        const float* blk = &src[(t * Hkv + h) * D + d / 32 * 32];
                        float amax = 0;
                        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(blk[i]));
                        const float sd = amax / 127.0f, id = sd != 0 ? 1.0f / sd : 0.0f;
                        got.push_back(x);
                        want.push_back(std::round(src[(t * Hkv + h) * D + d] * id) * hround(sd));
                    }
                }
        }
        if (T == 1 && p0 == 1000) check("kv_store_q8 (ggml's q8_0 blocks)", got, want, 0);
    }
    if (!q8) {
        std::vector<float> got, want;
        for (int64_t t = 0; t < all; ++t)
            for (int64_t h = 0; h < Hkv; ++h)
                for (int64_t d = 0; d < D; ++d) {
                    got.push_back(h2f(kc[(h * cap + t) * D + d]));
                    want.push_back(hround(k[(t * Hkv + h) * D + d]));
                    got.push_back(h2f(vc[(h * cap + t) * D + d]));
                    want.push_back(hround(v[(t * Hkv + h) * D + d]));
                }
        if (T == 1 && p0 == 1000) check("kv_store (the halves)", got, want, 0);
    }
    std::vector<float> want((size_t) (T * Hq * D));
    const int64_t G_ = Hq / Hkv;
    std::vector<double> sco;
    for (int64_t t = 0; t < T; ++t)
        for (int64_t h = 0; h < Hq; ++h) {
            const int64_t hk = h / G_, n = p0 + t + 1;
            sco.assign((size_t) n, 0);
            double mx = -1e300;
            for (int64_t j = 0; j < n; ++j) {
                double s = 0;
                for (int64_t d = 0; d < D; ++d) s += (double) q[t * qs + h * D + d] * h2f(kc[(hk * cap + j) * D + d]);
                s /= std::sqrt((double) D);
                sco[j] = s;
                mx = std::max(mx, s);
            }
            double L = 0;
            for (int64_t j = 0; j < n; ++j) L += (sco[j] = std::exp(sco[j] - mx));
            for (int64_t d = 0; d < D; ++d) {
                double a = 0;
                for (int64_t j = 0; j < n; ++j) a += sco[j] * h2f(vc[(hk * cap + j) * D + d]);
                want[(t * Hq + h) * D + d] = (float) (a / L);
            }
        }
    const bool mma = T > 2 && Hq / Hkv <= 6;
    char name[96];
    std::snprintf(name, sizeof name, "attn%s T=%lld p0=%lld Hq/Hkv=%lld (%s)", q8 ? " q8" : "", (long long) T, (long long) p0, (long long) (Hq / Hkv),
                  mma ? (sc > 0 ? "tensor cores, splits" : "tensor cores") : T > 16 ? "float tiles" : sc > 0 ? "splits" : "one split");
    // the tensor cores' and the float tiles' queries are halves: a looser bound
    // (q8: the reference's values are the widened products rounded to half; the decode kernel's are float32)
    check(name, down<float>(O, want.size()), want, mma || T > 16 ? 3e-3 : q8 ? 1e-3 : 2e-5);
    ns_free(G, K); ns_free(G, V); ns_free(G, Q); ns_free(G, KC); ns_free(G, VC); ns_free(G, O);
    if (S) ns_free(G, S);
}

static void test_kda(int64_t T, int64_t H, bool snap) {
    const int64_t d = 128, n = T * H * d;
    auto q = rnd((size_t) n), k = rnd((size_t) n), v = rnd((size_t) n), g = rnd((size_t) n, 0.8f, 1.0f), beta = rnd((size_t) (T * H), 0.0f, 1.0f);
    // q, k as the engine has them: L2-normed rows
    for (int64_t r = 0; r < T * H; ++r) {
        double sq = 0, sk = 0;
        for (int64_t i = 0; i < d; ++i) sq += q[r * d + i] * q[r * d + i], sk += k[r * d + i] * k[r * d + i];
        for (int64_t i = 0; i < d; ++i) q[r * d + i] /= (float) std::sqrt(sq), k[r * d + i] /= (float) std::sqrt(sk);
    }
    auto s0 = rnd((size_t) (H * d * d), -0.3f, 0.3f);
    float *Q = up(q), *K = up(k), *V = up(v), *Gg = up(g), *B = up(beta), *S = up(s0);
    float* O = (float*) dev((size_t) n * 4);
    float* SN = snap && T > 1 ? (float*) dev((size_t) ((T - 1) * H * d * d) * 4) : nullptr;
    ok(ns_q35_kda_scan(G, Q, K, V, Gg, B, S, O, T, H, d, SN), "kda_scan");
    std::vector<double> Sr(s0.begin(), s0.end());
    std::vector<float> o((size_t) n), snapw;
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t h = 0; h < H; ++h) {
            double* Sh = &Sr[(size_t) (h * d * d)];
            const int64_t base = (t * H + h) * d;
            for (int64_t i = 0; i < d; ++i)
                for (int64_t j = 0; j < d; ++j) Sh[j * d + i] *= g[base + i];
            for (int64_t j = 0; j < d; ++j) {
                double kv = 0;
                for (int64_t i = 0; i < d; ++i) kv += Sh[j * d + i] * k[base + i];
                const double delta = (v[base + j] - kv) * beta[t * H + h];
                for (int64_t i = 0; i < d; ++i) Sh[j * d + i] += k[base + i] * delta;
            }
            for (int64_t j = 0; j < d; ++j) {
                double a = 0;
                for (int64_t i = 0; i < d; ++i) a += Sh[j * d + i] * q[base + i];
                o[base + j] = (float) (a / std::sqrt((double) d));
            }
        }
        if (SN && t + 1 < T) snapw.insert(snapw.end(), Sr.begin(), Sr.end());
    }
    char name[64];
    std::snprintf(name, sizeof name, "kda_scan T=%lld H=%lld: o", (long long) T, (long long) H);
    check(name, down<float>(O, o.size()), o, 1e-4);
    std::snprintf(name, sizeof name, "kda_scan T=%lld H=%lld: S", (long long) T, (long long) H);
    check(name, down<float>(S, Sr.size()), std::vector<float>(Sr.begin(), Sr.end()), 1e-4);
    if (SN) check("kda_scan snapshots", down<float>(SN, snapw.size()), snapw, 1e-4);
}

static void test_small() {
    // conv + SiLU with the state and snapshots
    {
        const int64_t T = 5, D = 96;
        const int k = 4;
        auto x = rnd((size_t) (T * D)), st = rnd((size_t) ((k - 1) * D)), w = rnd((size_t) (D * k));
        float *X = up(x), *ST = up(st), *W = up(w);
        float* O = (float*) dev((size_t) (T * D) * 4);
        float* SN = (float*) dev((size_t) ((T - 1) * (k - 1) * D) * 4);
        ok(ns_q35_conv_silu(G, X, ST, W, O, T, D, k, SN), "conv_silu");
        auto in = [&](int64_t src, int64_t c) { return src >= 0 ? x[src * D + c] : st[(k - 1 + src) * D + c]; };
        std::vector<float> o((size_t) (T * D)), sn, ns((size_t) ((k - 1) * D));
        for (int64_t t = 0; t < T; ++t)
            for (int64_t c = 0; c < D; ++c) {
                double s = 0;
                for (int j = 0; j < k; ++j) s += in(t - (k - 1) + j, c) * w[c * k + j];
                o[t * D + c] = (float) (s / (1 + std::exp(-s)));
            }
        for (int64_t r = 0; r + 1 < T; ++r)
            for (int j = 0; j < k - 1; ++j)
                for (int64_t c = 0; c < D; ++c) sn.push_back(in(r + 1 - (k - 1) + j, c));
        for (int j = 0; j < k - 1; ++j)
            for (int64_t c = 0; c < D; ++c) ns[j * D + c] = in(T - (k - 1) + j, c);
        check("conv_silu out", down<float>(O, o.size()), o, 1e-6);
        check("conv_silu state", down<float>(ST, ns.size()), ns, 0);
        check("conv_silu snapshots", down<float>(SN, sn.size()), sn, 0);
    }
    // L2 norm
    {
        const int64_t rows = 7, n = 128;
        const float eps = 1e-6f;
        auto x = rnd((size_t) (rows * n));
        float* X = up(x);
        ok(ns_q35_l2_norm(G, X, rows, n, eps), "l2");
        std::vector<float> w = x;
        for (int64_t r = 0; r < rows; ++r) {
            double s = 0;
            for (int64_t i = 0; i < n; ++i) s += (double) x[r * n + i] * x[r * n + i];
            for (int64_t i = 0; i < n; ++i) w[r * n + i] = (float) (x[r * n + i] / std::sqrt(s + eps));
        }
        check("l2_norm", down<float>(X, w.size()), w, 1e-6);
    }
    // SwiGLU (no clamp, then a clamp; in place on gate)
    for (float lim : {INFINITY, 0.7f}) {
        const int64_t n = 1000;
        auto a = rnd((size_t) n, -3, 3), b = rnd((size_t) n, -3, 3);
        float *A = up(a), *B = up(b);
        ok(ns_q35_swiglu(G, A, B, A, n, lim), "swiglu");
        std::vector<float> w((size_t) n);
        for (int64_t i = 0; i < n; ++i) {
            const double g = std::min((double) a[i], (double) lim), u = std::min(std::max((double) b[i], (double) -lim), (double) lim);
            w[i] = (float) (g / (1 + std::exp(-g)) * u);
        }
        check(std::isinf(lim) ? "swiglu (no clamp, in place)" : "swiglu (clamp 0.7)", down<float>(A, w.size()), w, 1e-6);
    }
    // DeltaNet's gates
    {
        const int64_t T = 3, H = 48, d = 128;
        auto al = rnd((size_t) (T * H), -30, 30), be = rnd((size_t) (T * H), -4, 4), dt = rnd((size_t) H), a = rnd((size_t) H, -2, -0.01f);
        float *AL = up(al), *BE = up(be), *DT = up(dt), *A = up(a);
        float* EG = (float*) dev((size_t) (T * H * d) * 4);
        ok(ns_q35_gdn_gates(G, AL, BE, DT, A, EG, T, H, d), "gdn_gates");
        std::vector<float> eg((size_t) (T * H * d)), bw((size_t) (T * H));
        for (int64_t t = 0; t < T; ++t)
            for (int64_t h = 0; h < H; ++h) {
                const double x = al[t * H + h] + dt[h], sp = x > 20 ? x : std::log1p(std::exp(x));
                for (int64_t i = 0; i < d; ++i) eg[(t * H + h) * d + i] = (float) std::exp(sp * a[h]);
                bw[t * H + h] = (float) (1 / (1 + std::exp(-(double) be[t * H + h])));
            }
        check("gdn_gates eg", down<float>(EG, eg.size()), eg, 1e-6);
        check("gdn_gates beta", down<float>(BE, bw.size()), bw, 1e-6);
    }
    // expand
    {
        const int64_t T = 3, Hk = 16, Hv = 48, d = 128, ss = Hk * d + 32;
        auto s = rnd((size_t) (T * ss));
        float* S = up(s);
        float* D_ = (float*) dev((size_t) (T * Hv * d) * 4);
        ok(ns_q35_expand(G, S, ss, D_, T, Hk, Hv, d), "expand");
        std::vector<float> w((size_t) (T * Hv * d));
        for (int64_t t = 0; t < T; ++t)
            for (int64_t h = 0; h < Hv; ++h)
                for (int64_t c = 0; c < d; ++c) w[(t * Hv + h) * d + c] = s[t * ss + (h % Hk) * d + c];
        check("expand", down<float>(D_, w.size()), w, 0);
    }
    // DeltaNet's output norm
    {
        const int64_t T = 3, H = 48, d = 128, zs = H * d + 16;
        const float eps = 1e-6f;
        auto o = rnd((size_t) (T * H * d)), z = rnd((size_t) (T * zs), -3, 3), w = rnd((size_t) d);
        float *O = up(o), *Z = up(z), *W = up(w);
        float* Y = (float*) dev((size_t) (T * H * d) * 4);
        ok(ns_q35_gdn_out(G, O, Z, zs, W, Y, T, H, d, eps), "gdn_out");
        std::vector<float> y((size_t) (T * H * d));
        for (int64_t t = 0; t < T; ++t)
            for (int64_t h = 0; h < H; ++h) {
                const float* x = &o[(t * H + h) * d];
                double s = 0;
                for (int64_t i = 0; i < d; ++i) s += (double) x[i] * x[i];
                const double inv = 1 / std::sqrt(s / d + eps);
                for (int64_t i = 0; i < d; ++i) {
                    const double zz = z[t * zs + h * d + i];
                    y[(t * H + h) * d + i] = (float) (x[i] * inv * w[i] * (zz / (1 + std::exp(-zz))));
                }
            }
        check("gdn_out", down<float>(Y, y.size()), y, 1e-5);
    }
    // the attention's gate
    {
        const int64_t T = 3, H = 24, D = 256, gs = H * 2 * D;
        auto out = rnd((size_t) (T * H * D)), qf = rnd((size_t) (T * gs), -4, 4);
        float *O = up(out), *QF = up(qf);
        ok(ns_q35_gate_mul(G, O, QF, gs, T, H, D), "gate_mul");
        std::vector<float> w = out;
        for (int64_t t = 0; t < T; ++t)
            for (int64_t h = 0; h < H; ++h)
                for (int64_t d = 0; d < D; ++d) w[(t * H + h) * D + d] *= (float) (1 / (1 + std::exp(-(double) qf[t * gs + h * 2 * D + D + d])));
        check("gate_mul", down<float>(O, w.size()), w, 1e-6);
    }
    // q / k norm + rotary positions: a strided source (the query's rows carry the gate: hs = 2 D), and in place
    for (int inplace = 0; inplace < 2; ++inplace) {
        const int64_t T = 4, H = 24, D = 256, n_rot = 64, p0 = 37;
        const float eps = 1e-6f, theta = 1e7f;
        const int64_t hs = inplace ? D : 2 * D, ss = H * hs + 8, ds = inplace ? ss : H * D;
        auto src = rnd((size_t) (T * ss), -2, 2), w = rnd((size_t) D, 0.5f, 1.5f);
        float *S = up(src), *W = up(w);
        float* Dd = inplace ? S : (float*) dev((size_t) (T * ds) * 4);
        ok(ns_q35_qk_norm_rope(G, S, ss, hs, Dd, ds, T, H, D, W, eps, n_rot, theta, p0), "qk_norm_rope");
        std::vector<float> want((size_t) (T * H * D)), got;
        auto all = down<float>(Dd, (size_t) (T * ds));
        for (int64_t t = 0; t < T; ++t)
            for (int64_t h = 0; h < H; ++h) {
                const float* x = &src[t * ss + h * hs];
                double s = 0;
                for (int64_t i = 0; i < D; ++i) s += (double) x[i] * x[i];
                const double inv = 1 / std::sqrt(s / D + eps);
                std::vector<double> nv((size_t) D);
                for (int64_t i = 0; i < D; ++i) nv[i] = x[i] * inv * w[i];
                for (int64_t i = 0; i < D; ++i) {
                    double y = nv[i];
                    if (i < n_rot) {
                        const int64_t hr = n_rot / 2, kk = i < hr ? i : i - hr;
                        const double a = (double) (p0 + t) * std::pow((double) theta, -2.0 * kk / n_rot);
                        y = i < hr ? nv[i] * std::cos(a) - nv[i + hr] * std::sin(a) : nv[i] * std::cos(a) + nv[i - hr] * std::sin(a);
                    }
                    want[(t * H + h) * D + i] = (float) y;
                    got.push_back(all[t * ds + h * (inplace ? hs : D) + i]);
                }
            }
        // the angle in float32 (positions times theta's powers): ~1e-5 at position 40
        check(inplace ? "qk_norm_rope (in place)" : "qk_norm_rope (strided)", got, want, 5e-5);
    }
    // argmax: small rows with ties, and one 248,320-wide row (the vocabulary) with a tie past the first slice
    {
        const int64_t rows = 3, n = 1000;
        auto x = rnd((size_t) (rows * n));
        x[0 * n + 17] = 5; x[0 * n + 900] = 5;   // tie: 17
        x[1 * n + 999] = 7;
        x[2 * n + 3] = 2;
        float* X = up(x);
        int32_t* O = (int32_t*) dev(rows * 4);
        ok(ns_q35_argmax(G, X, rows, n, O), "argmax");
        auto o = down<int32_t>(O, rows);
        const bool p = o[0] == 17 && o[1] == 999 && o[2] == 3;
        if (!p) ++fails;
        std::printf("%-44s %d %d %d  %s\n", "argmax rows (want 17 999 3)", o[0], o[1], o[2], p ? "PASS" : "FAIL");
        const int64_t V = 248320;
        auto y = rnd((size_t) V);
        y[200000] = 9; y[150001] = 9;   // tie: 150001
        float* Y = up(y);
        ok(ns_q35_argmax(G, Y, 1, V, O), "argmax wide");
        auto o2 = down<int32_t>(O, 1);
        const bool p2 = o2[0] == 150001;
        if (!p2) ++fails;
        std::printf("%-44s %d  %s\n", "argmax 248320 (want 150001)", o2[0], p2 ? "PASS" : "FAIL");
    }
}

// ---- times
static double ms_of(int reps, const std::function<void()>& f) {
    f();
    ok(ns_sync(G), "sync");
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) f();
    ok(ns_sync(G), "sync");
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
}
static void bench_attn(int64_t T, int64_t keys, bool q8 = false) {
    const int64_t Hq = 24, Hkv = 4, D = 256, p0 = keys - T, cap = keys, RB = q8 ? D + 16 : D * 2;
    void* KC = dev((size_t) (Hkv * cap * RB));
    void* VC = dev((size_t) (Hkv * cap * RB));
    ok(cudaMemset(KC, 0, (size_t) (Hkv * cap * RB)) == cudaSuccess ? 0 : -1, "memset");
    ok(cudaMemset(VC, 0, (size_t) (Hkv * cap * RB)) == cudaSuccess ? 0 : -1, "memset");
    auto attn = q8 ? ns_q35_attn_q8 : ns_q35_attn;
    float* Q = up(rnd((size_t) (T * Hq * D)));
    const int64_t sc = ns_q35_attn_scratch(T, Hq, Hkv, D, p0);
    float* S = sc > 0 ? (float*) dev((size_t) sc * 4) : nullptr;
    float* O = (float*) dev((size_t) (T * Hq * D) * 4);
    const double ms = ms_of(T == 1 ? 50 : 3, [&] { ok(attn(G, Q, Hq * D, KC, VC, T, Hq, Hkv, D, cap, p0, O, S), "attn"); });
    const double gb = (double) Hkv * keys * RB * 2 / 1e9;
    std::printf("attn%s T=%-4lld keys=%-7lld  %8.3f ms  (cache %.2f GB: %.0f GB/s)\n", q8 ? " q8" : "   ", (long long) T, (long long) keys, ms, gb, gb / ms * 1e3);
    ns_free(G, KC); ns_free(G, VC); ns_free(G, Q); ns_free(G, O);
    if (S) ns_free(G, S);
}
static void bench_kda(int64_t T) {
    const int64_t H = 48, d = 128, n = T * H * d;
    float *Q = up(rnd((size_t) n, -0.1f, 0.1f)), *K = up(rnd((size_t) n, -0.1f, 0.1f)), *V = up(rnd((size_t) n)), *Gg = up(rnd((size_t) n, 0.9f, 1.0f));
    float *B = up(rnd((size_t) (T * H), 0, 1)), *S = (float*) dev((size_t) (H * d * d) * 4), *O = (float*) dev((size_t) n * 4);
    const double ms = ms_of(T == 1 ? 200 : 10, [&] { ok(ns_q35_kda_scan(G, Q, K, V, Gg, B, S, O, T, H, d, nullptr), "kda"); });
    std::printf("kda_scan T=%-4lld H=48            %8.3f ms\n", (long long) T, ms);
}

int main(int argc, char** argv) {
    const int gpu = argc > 1 ? std::atoi(argv[1]) : 0;
    GPU = gpu;
    ok(ns_gpu_open(gpu, &G), "gpu open");
    std::printf("== GPU %d\n", gpu);
    test_quants();
    test_small();
    test_kda(1, 4, false);
    test_kda(6, 3, true);
    test_attn(1, 0, 24, 4);
    test_attn(1, 1000, 24, 4);
    test_attn(1, 5000, 24, 4);
    test_attn(4, 300, 24, 4);
    test_attn(16, 777, 24, 4);
    test_attn(17, 0, 24, 4);
    test_attn(37, 100, 24, 4);
    test_attn(64, 1100, 24, 4);
    test_attn(9, 50, 8, 8);
    test_attn(3, 4000, 32, 4);
    test_attn(2, 0, 24, 4);
    test_attn(40, 33, 16, 2);
    test_attn(300, 2000, 24, 4);
    test_attn(130, 0, 24, 4);
    test_attn(33, 900, 4, 4);
    for (auto [T, p0, Hq, Hkv] : {std::array<int64_t, 4>{1, 0, 24, 4}, {1, 1000, 24, 4}, {1, 5000, 24, 4}, {2, 70, 24, 4}, {4, 300, 24, 4},
                                  {16, 777, 24, 4}, {37, 100, 24, 4}, {300, 2000, 24, 4}, {9, 50, 8, 8}, {40, 33, 16, 2}})
        test_attn(T, p0, Hq, Hkv, true);
    std::printf("== times\n");
    bench_attn(1, 32768);
    bench_attn(1, 131072);
    bench_attn(2, 131072);
    bench_attn(4, 131072);
    bench_attn(16, 131072);
    bench_attn(512, 512);
    bench_attn(512, 32768);
    bench_attn(512, 131072);
    bench_attn(64, 131072);
    bench_attn(1, 131072, true);
    bench_attn(4, 131072, true);
    bench_attn(512, 32768, true);
    bench_attn(512, 131072, true);
    bench_kda(1);
    bench_kda(512);
    std::printf("== %s (%d failed)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
