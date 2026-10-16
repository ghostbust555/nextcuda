// ns_internal.hpp: what the C ABI's translation units share (not part of the ABI).
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <vector>

struct ns_gpu {
    int dev = 0;               // the CUDA device ordinal (CUDA_DEVICE_ORDER=PCI_BUS_ID numbers them as nvidia-smi)
    cudaStream_t q = nullptr;  // the work stream (in order)
    cudaStream_t cq = nullptr; // copies that overlap q's work (ns_stream_copy); ordered against q on the device only
    cudaStream_t cq2 = nullptr;// a second copy lane: the other PCIe direction runs beside the first
    std::vector<cudaEvent_t> ev;   // tickets (ns_mark, ns_stream_copy, ns_stamp), % size
    int64_t next_ev = 0;
    bool stamps = false;       // NS_PROFILE=gpu: events keep timing
};

// sets the thread's error (ns_last_error) and returns -1
int ns_fail(const std::string& what);

// the CUDA call's error as ns_fail's, with where it was
int ns_cuda(cudaError_t e, const char* what);

#define NS_CU(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) return ns_cuda(e_, #call); } while (0)
#define NS_TRY try {
#define NS_CATCH } catch (const std::exception& e) { return ns_fail(e.what()); }
// a kernel launch's own error (bad configuration), checked right after it
#define NS_LAUNCH_CHECK() do { cudaError_t e_ = cudaGetLastError(); if (e_ != cudaSuccess) return ns_cuda(e_, "kernel launch"); } while (0)
