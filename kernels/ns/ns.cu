// ns.cu: the devices, memory and copies of the C ABI (ns.h) on CUDA. Kernels are called through their own wrappers
// (kernels/<kind>/<arch>/*.cu), each taking the ns_gpu's stream.
#include "ns.h"
#include "ns_internal.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {
thread_local std::string g_err;

int count() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess ? n : 0;
}

// make GPU g current for this thread (every call: threads are not tied to a GPU)
inline cudaError_t use(ns_gpu* g) { return cudaSetDevice(g->dev); }
}  // namespace

int ns_fail(const std::string& what) {
    g_err = what;
    return -1;
}

int ns_cuda(cudaError_t e, const char* what) {
    return ns_fail(std::string(what) + ": " + cudaGetErrorName(e) + " (" + cudaGetErrorString(e) + ")");
}

extern "C" {

const char* ns_last_error(void) { return g_err.c_str(); }
const char* ns_version(void) { return "nextcuda 0.1"; }

int ns_gpu_count(void) { return count(); }

int ns_gpu_name(int index, char* buf, size_t len) {
    cudaDeviceProp p;
    NS_CU(cudaGetDeviceProperties(&p, index));
    std::snprintf(buf, len, "%s", p.name);
    return 0;
}

// GPU `index`'s multiprocessors and their clock (MHz): how much it computes
int ns_gpu_units(int index, int* units, int* mhz) {
    int sm = 0, khz = 0;
    NS_CU(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, index));
    NS_CU(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, index));
    *units = sm;
    *mhz = khz / 1000;
    return 0;
}

// the PCI address of GPU `index` ("0000:09:00.0") into buf
int ns_gpu_pci(int index, char* buf, size_t len) {
    char id[32] = {0};
    NS_CU(cudaDeviceGetPCIBusId(id, (int) sizeof id, index));
    std::string a(id);
    for (auto& c : a) c = (char) std::tolower((unsigned char) c);
    // "00000000:09:00.0" -> "0000:09:00.0"
    const auto colon = a.find(':');
    if (colon != std::string::npos && colon > 4) a = a.substr(colon - 4);
    std::snprintf(buf, len, "%s", a.c_str());
    return 0;
}

int ns_gpu_open(int index, ns_gpu** out) {
    NS_TRY
    const int n = count();
    if (index < 0 || index >= n) return ns_fail("no GPU " + std::to_string(index) + " (" + std::to_string(n) + " found)");
    NS_CU(cudaSetDevice(index));
    auto* g = new ns_gpu();
    g->dev = index;
    const char* pf = getenv("NS_PROFILE");
    g->stamps = pf && std::string(pf) == "gpu";
    if (cudaStreamCreateWithFlags(&g->q, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithFlags(&g->cq, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithFlags(&g->cq2, cudaStreamNonBlocking) != cudaSuccess) {
        delete g;
        return ns_fail("creating the GPU's streams failed");
    }
    // tickets are events kept in a ring (the host never waits on one: the device orders by them)
    g->ev.resize(1 << 12);
    for (auto& e : g->ev) cudaEventCreateWithFlags(&e, g->stamps ? cudaEventDefault : cudaEventDisableTiming);
    *out = g;
    return 0;
    NS_CATCH
}

void ns_gpu_close(ns_gpu* g) {
    if (!g) return;
    cudaSetDevice(g->dev);
    cudaStreamSynchronize(g->q);
    cudaStreamSynchronize(g->cq);
    cudaStreamSynchronize(g->cq2);
    for (auto e : g->ev) cudaEventDestroy(e);
    cudaStreamDestroy(g->q);
    cudaStreamDestroy(g->cq);
    cudaStreamDestroy(g->cq2);
    delete g;
}

void* ns_gpu_queue(ns_gpu* g) { return (void*) g->q; }

int ns_gpu_memory(ns_gpu* g, uint64_t* total, int64_t* free_bytes) {
    NS_CU(use(g));
    size_t f = 0, t = 0;
    NS_CU(cudaMemGetInfo(&f, &t));
    *total = t;
    *free_bytes = (int64_t) f;
    return 0;
}

int ns_alloc(ns_gpu* g, size_t bytes, void** out) {
    NS_CU(use(g));
    cudaError_t e = cudaMalloc(out, bytes ? bytes : 1);
    if (e != cudaSuccess) {
        cudaGetLastError();
        return ns_fail("device allocation of " + std::to_string(bytes) + " bytes failed: " + cudaGetErrorString(e));
    }
    return 0;
}

int ns_free(ns_gpu* g, void* p) {
    NS_CU(use(g));
    // the stream is in order: work queued on p runs first
    NS_CU(cudaStreamSynchronize(g->q));
    NS_CU(cudaFree(p));
    return 0;
}

int ns_alloc_host(ns_gpu* g, size_t bytes, void** out) {
    NS_CU(use(g));
    cudaError_t e = cudaMallocHost(out, bytes ? bytes : 1);
    if (e != cudaSuccess) {
        cudaGetLastError();
        return ns_fail("pinned host allocation of " + std::to_string(bytes) + " bytes failed: " + cudaGetErrorString(e));
    }
    return 0;
}

int ns_free_host(ns_gpu* g, void* p) {
    NS_CU(use(g));
    NS_CU(cudaStreamSynchronize(g->q));
    NS_CU(cudaFreeHost(p));
    return 0;
}

// queued on the GPU's stream (in order); a pageable host source is staged by the driver before the call returns
int ns_copy_to(ns_gpu* g, void* dst, const void* src, size_t bytes) {
    NS_CU(use(g));
    NS_CU(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, g->q));
    return 0;
}

int ns_copy_from(ns_gpu* g, void* dst, const void* src, size_t bytes) {
    NS_CU(use(g));
    NS_CU(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, g->q));
    return 0;
}

int ns_copy_dev(ns_gpu* g, void* dst, const void* src, size_t bytes) {
    NS_CU(use(g));
    NS_CU(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, g->q));
    return 0;
}

int ns_copy_2d(ns_gpu* g, void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height) {
    NS_CU(use(g));
    NS_CU(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice, g->q));
    return 0;
}

// Copies beside the GPU's work. The order between the streams is kept on the device (events, cudaStreamWaitEvent):
// the host never waits on an event.
static int64_t keep(ns_gpu* g, cudaStream_t s) {
    const int64_t t = g->next_ev++;
    cudaEventRecord(g->ev[t % g->ev.size()], s);
    return t;
}

// a ticket for everything submitted to the GPU's stream so far
int ns_mark(ns_gpu* g, int64_t* ticket) {
    NS_CU(use(g));
    *ticket = keep(g, g->q);
    return 0;
}

// bytes from src to dst on copy lane `lane` (0 or 1), after the `ndeps` tickets `after`; its own ticket
int ns_stream_copy(ns_gpu* g, void* dst, const void* src, size_t bytes, const int64_t* after, int ndeps, int lane, int64_t* ticket) {
    NS_CU(use(g));
    cudaStream_t s = lane ? g->cq2 : g->cq;
    for (int i = 0; i < ndeps; ++i) NS_CU(cudaStreamWaitEvent(s, g->ev[after[i] % g->ev.size()], 0));
    NS_CU(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, s));
    *ticket = keep(g, s);
    return 0;
}

// a timestamp: an event on the GPU's stream (its time is when the work before it ended); its ticket
int ns_stamp(ns_gpu* g, int64_t* ticket) {
    NS_CU(use(g));
    *ticket = keep(g, g->q);
    return 0;
}

// nanoseconds between tickets t0 and t1 (stamps; waits for them - profiling only, with NS_PROFILE=gpu)
int ns_elapsed(ns_gpu* g, int64_t t0, int64_t t1, double* ns) {
    NS_CU(use(g));
    if (!g->stamps) return ns_fail("ns_elapsed needs NS_PROFILE=gpu");
    float ms = 0;
    NS_CU(cudaEventSynchronize(g->ev[t1 % g->ev.size()]));
    NS_CU(cudaEventElapsedTime(&ms, g->ev[t0 % g->ev.size()], g->ev[t1 % g->ev.size()]));
    *ns = (double) ms * 1e6;
    return 0;
}

// the GPU's stream waits (on the device) for ticket t
int ns_await(ns_gpu* g, int64_t t) {
    NS_CU(use(g));
    NS_CU(cudaStreamWaitEvent(g->q, g->ev[t % g->ev.size()], 0));
    return 0;
}

int ns_fill(ns_gpu* g, void* dst, uint8_t value, size_t bytes) {
    NS_CU(use(g));
    NS_CU(cudaMemsetAsync(dst, value, bytes, g->q));
    return 0;
}

int ns_sync(ns_gpu* g) {
    NS_CU(use(g));
    NS_CU(cudaStreamSynchronize(g->q));
    // a kernel's asynchronous error surfaces here
    NS_CU(cudaGetLastError());
    return 0;
}

int ns_copy_peer(ns_gpu* to, void* dst, ns_gpu* from, const void* src, size_t bytes, void* staging) {
    NS_CU(use(from));
    NS_CU(cudaMemcpyAsync(staging, src, bytes, cudaMemcpyDeviceToHost, from->q));
    NS_CU(cudaStreamSynchronize(from->q));
    NS_CU(use(to));
    NS_CU(cudaMemcpyAsync(dst, staging, bytes, cudaMemcpyHostToDevice, to->q));
    NS_CU(cudaStreamSynchronize(to->q));
    return 0;
}

}  // extern "C"
