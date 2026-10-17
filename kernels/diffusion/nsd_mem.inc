// nsd_mem.inc: H3's way in (nextsycl's nsd.h, "context" and "device memory"): a context of its own on a GPU, and the
// device and pinned host memory it hands out. Included at the end of nsd.cu, inside its extern "C" block: Ctx, fail,
// cuda_fail and the CU macro are the ones above.
//
// The accounting is nextsycl's: every nsd_alloc is counted against Ctx::mem_cap (0.94 of the card; NSD_MEM_FRACTION
// overrides, clamped to [0.05, 0.97]) and one that would pass it is refused here, with nextsycl's message. It works on
// a context made with nsd_create too (the cap is worked out on first use). Allocation, the copies and the uploads are
// thread-safe: the book is under Ctx::mu, the CUDA calls are thread-safe themselves.
//
// Differences from nextsycl: the kernels' scratch buffers (nsd.cu's `scratch`) are not counted against the cap; the
// pinned-source check of nsd_upload asks the CUDA runtime (cudaPointerGetAttributes), so any cudaMallocHost memory
// passes, as any USM host memory of the SYCL context does there.

static size_t nsd_mem_total(int dev) {
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) { cudaGetLastError(); return 0; }
    return p.totalGlobalMem;
}

// the cap and the name, for a context that does not have them yet (nsd_create's); call with c->mu held
static void nsd_mem_init_locked(Ctx* c) {
    if (c->mem_cap == 0) {
        double frac = 0.94;                        // of the card, for nsd_alloc
        if (const char* e = std::getenv("NSD_MEM_FRACTION")) frac = std::min(0.97, std::max(0.05, std::atof(e)));
        c->mem_cap = (size_t) ((double) nsd_mem_total(c->dev) * frac);
    }
    if (!c->name[0]) {
        cudaDeviceProp p;
        if (cudaGetDeviceProperties(&p, c->dev) == cudaSuccess) std::snprintf(c->name, sizeof c->name, "%s", p.name);
        else cudaGetLastError();
    }
}

// the copy stream, made on first use for a context around an ns_gpu (an owned context makes it when opened)
static cudaStream_t nsd_copy_stream(Ctx* c) {
    std::lock_guard<std::mutex> l(c->mu);
    if (!c->cq && cudaStreamCreate(&c->cq) != cudaSuccess) { cudaGetLastError(); c->cq = nullptr; }
    return c->cq;
}

void* nsd_open(void) { return nsd_open_gpu(0); }

int nsd_gpu_count(void) {
    int n = 0;
    const cudaError_t e = cudaGetDeviceCount(&n);
    if (e == cudaErrorNoDevice) { cudaGetLastError(); return 0; }
    if (e != cudaSuccess) return cuda_fail(e, "cudaGetDeviceCount");
    return n;
}

int nsd_gpu_info(int index, char* name, int name_len, uint64_t* mem_bytes, char* pci, int pci_len) {
    const int n = nsd_gpu_count();
    if (n < 0) return -1;
    if (index < 0 || index >= n) return fail("no GPU " + std::to_string(index));
    cudaDeviceProp p;
    CU(cudaGetDeviceProperties(&p, index));
    if (name && name_len > 0) std::snprintf(name, name_len, "%s", p.name);
    if (mem_bytes) *mem_bytes = p.totalGlobalMem;
    if (pci && pci_len > 0) {
        char addr[64] = {0};
        if (cudaDeviceGetPCIBusId(addr, sizeof addr, index) != cudaSuccess) { cudaGetLastError(); addr[0] = 0; }
        // CUDA writes the domain in 8 hex digits ("00000000:0b:00.0"); nextsycl's form has 4
        const char* a = addr;
        if (std::strlen(addr) == 16 && std::strncmp(addr, "0000", 4) == 0) a = addr + 4;
        for (char* q = addr; *q; ++q) if (*q >= 'A' && *q <= 'F') *q = (char) (*q - 'A' + 'a');
        std::snprintf(pci, pci_len, "%s", a);
    }
    return 0;
}

void* nsd_open_gpu(int index) {
    const int n = nsd_gpu_count();
    if (n < 0) return nullptr;
    if (index < 0 || index >= n) {
        fail("no GPU " + std::to_string(index) + " (" + std::to_string(n) + " found)");
        return nullptr;
    }
    cudaError_t e = cudaSetDevice(index);
    if (e != cudaSuccess) { cuda_fail(e, "nsd_open_gpu: cudaSetDevice"); return nullptr; }
    auto* c = new Ctx();
    c->dev = index;
    c->owned = true;
    if ((e = cudaStreamCreate(&c->s)) != cudaSuccess) { cuda_fail(e, "nsd_open_gpu: cudaStreamCreate"); delete c; return nullptr; }
    if ((e = cudaStreamCreate(&c->cq)) != cudaSuccess) {
        cuda_fail(e, "nsd_open_gpu: cudaStreamCreate (copies)");
        cudaStreamDestroy(c->s); delete c; return nullptr;
    }
    if (cublasCreate(&c->cb) != CUBLAS_STATUS_SUCCESS) {
        fail("nsd_open_gpu: cublasCreate failed");
        cudaStreamDestroy(c->cq); cudaStreamDestroy(c->s); delete c; return nullptr;
    }
    cublasSetStream(c->cb, c->s);
    cublasSetMathMode(c->cb, CUBLAS_DEFAULT_MATH);
    if (cublasLtCreate(&c->lt) != CUBLAS_STATUS_SUCCESS) {
        fail("nsd_open_gpu: cublasLtCreate failed");
        cublasDestroy(c->cb); cudaStreamDestroy(c->cq); cudaStreamDestroy(c->s); delete c; return nullptr;
    }
    {
        std::lock_guard<std::mutex> l(c->mu);
        nsd_mem_init_locked(c);
    }
    return c;
}

const char* nsd_device_name(void* ctx) {
    auto* c = (Ctx*) ctx;
    cudaSetDevice(c->dev);
    std::lock_guard<std::mutex> l(c->mu);
    nsd_mem_init_locked(c);
    return c->name;
}

uint64_t nsd_mem_cap(void* ctx) {
    auto* c = (Ctx*) ctx;
    cudaSetDevice(c->dev);
    std::lock_guard<std::mutex> l(c->mu);
    nsd_mem_init_locked(c);
    return c->mem_cap;
}

uint64_t nsd_mem_used(void* ctx) {
    auto* c = (Ctx*) ctx;
    std::lock_guard<std::mutex> l(c->mu);
    return c->used;
}

// what the whole card has free right now, every process counted (0 when the driver cannot tell)
uint64_t nsd_mem_free(void* ctx) {
    auto* c = (Ctx*) ctx;
    if (cudaSetDevice(c->dev) != cudaSuccess) { cudaGetLastError(); return 0; }
    size_t fr = 0, total = 0;
    if (cudaMemGetInfo(&fr, &total) != cudaSuccess) { cudaGetLastError(); return 0; }
    return fr;
}

void* nsd_alloc(void* ctx, uint64_t bytes) {
    auto* c = (Ctx*) ctx;
    cudaError_t e = cudaSetDevice(c->dev);
    if (e != cudaSuccess) { cuda_fail(e, "nsd_alloc: cudaSetDevice"); return nullptr; }
    std::lock_guard<std::mutex> l(c->mu);
    nsd_mem_init_locked(c);
    if (c->used + bytes > c->mem_cap) {
        fail("nsd_alloc: " + std::to_string(bytes >> 20) + " MiB would pass the device memory cap (" +
             std::to_string(c->used >> 20) + " of " + std::to_string(c->mem_cap >> 20) + " MiB in use)");
        return nullptr;
    }
    void* p = nullptr;
    if (cudaMalloc(&p, bytes ? bytes : 1) != cudaSuccess || !p) {
        cudaGetLastError();
        fail("nsd_alloc: the device refused " + std::to_string(bytes >> 20) + " MiB");
        return nullptr;
    }
    // NSD_POISON=1: new buffers filled with NaN bytes (finds reads of unwritten memory)
    if (std::getenv("NSD_POISON")) cudaMemset(p, 0xff, bytes ? bytes : 1);
    c->allocs.emplace(p, bytes);
    c->used += bytes;
    return p;
}

void nsd_free(void* ctx, void* p) {
    auto* c = (Ctx*) ctx;
    if (!p) return;
    cudaSetDevice(c->dev);
    cudaStreamSynchronize(c->s);                   // nothing queued may still read or write it
    if (c->cq) cudaStreamSynchronize(c->cq);       // nor an upload into it
    std::lock_guard<std::mutex> l(c->mu);
    auto it = c->allocs.find(p);
    if (it == c->allocs.end()) return;
    c->used -= it->second;
    c->allocs.erase(it);
    cudaFree(p);
}

// host <-> device: ordered after the stream's queued work, and waited for (the host buffer may be released on return)
int nsd_write(void* ctx, void* dst, const void* src_host, uint64_t bytes) {
    auto* c = (Ctx*) ctx;
    CU(cudaSetDevice(c->dev));
    if (!bytes) return 0;
    CU(cudaMemcpyAsync(dst, src_host, bytes, cudaMemcpyHostToDevice, c->s));
    CU(cudaStreamSynchronize(c->s));
    return 0;
}

int nsd_read(void* ctx, void* dst_host, const void* src, uint64_t bytes) {
    auto* c = (Ctx*) ctx;
    CU(cudaSetDevice(c->dev));
    if (!bytes) return 0;
    CU(cudaMemcpyAsync(dst_host, src, bytes, cudaMemcpyDeviceToHost, c->s));
    CU(cudaStreamSynchronize(c->s));
    return 0;
}

// device to device, queued like a kernel (it does not wait)
int nsd_copy(void* ctx, void* dst, const void* src, uint64_t bytes) {
    auto* c = (Ctx*) ctx;
    CU(cudaSetDevice(c->dev));
    if (!bytes) return 0;
    CU(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, c->s));
    return 0;
}

// pinned host memory, not counted against the device cap
void* nsd_alloc_host(void* ctx, uint64_t bytes) {
    auto* c = (Ctx*) ctx;
    cudaError_t e = cudaSetDevice(c->dev);
    if (e != cudaSuccess) { cuda_fail(e, "nsd_alloc_host: cudaSetDevice"); return nullptr; }
    void* p = nullptr;
    if (cudaMallocHost(&p, bytes ? bytes : 1) != cudaSuccess || !p) {
        cudaGetLastError();
        fail("nsd_alloc_host: the runtime refused " + std::to_string(bytes >> 20) + " MiB of pinned memory");
        return nullptr;
    }
    std::lock_guard<std::mutex> l(c->mu);
    c->host.emplace(p, bytes);
    return p;
}

void nsd_free_host(void* ctx, void* p) {
    auto* c = (Ctx*) ctx;
    if (!p) return;
    cudaSetDevice(c->dev);
    if (c->cq) cudaStreamSynchronize(c->cq);       // no upload may still read it
    std::lock_guard<std::mutex> l(c->mu);
    auto it = c->host.find(p);
    if (it == c->host.end()) return;
    c->host.erase(it);
    cudaFreeHost(p);
}

// pinned host -> device on the copy stream, beside the kernels (it does not wait, and the kernels do not wait for it)
int nsd_upload(void* ctx, void* dst, const void* src_pinned, uint64_t bytes) {
    auto* c = (Ctx*) ctx;
    CU(cudaSetDevice(c->dev));
    cudaPointerAttributes sa, da;
    const bool ok = cudaPointerGetAttributes(&sa, src_pinned) == cudaSuccess && sa.type == cudaMemoryTypeHost &&
                    cudaPointerGetAttributes(&da, dst) == cudaSuccess && da.type == cudaMemoryTypeDevice && da.device == c->dev;
    if (!ok) {
        cudaGetLastError();
        return fail("nsd_upload: the source must be nsd_alloc_host memory and the destination device memory");
    }
    cudaStream_t cq = nsd_copy_stream(c);
    if (!cq) return fail("nsd_upload: cudaStreamCreate (copies) failed");
    if (!bytes) return 0;
    CU(cudaMemcpyAsync(dst, src_pinned, bytes, cudaMemcpyHostToDevice, cq));
    return 0;
}

int nsd_upload_wait(void* ctx) {
    auto* c = (Ctx*) ctx;
    CU(cudaSetDevice(c->dev));
    if (c->cq) CU(cudaStreamSynchronize(c->cq));
    return 0;
}
