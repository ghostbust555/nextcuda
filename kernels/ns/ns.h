/* ns.h: libnextcuda's C ABI - what the Rust runtime (crates/sys) calls; the same ABI nextsycl's SYCL library has, so an
 * engine's Rust side does not know which GPU runtime is under it.
 *
 * Every call returns 0 on success, or -1 with the reason in ns_last_error() (per thread). Device memory is CUDA
 * device memory of one GPU (cudaSetDevice is done by each call); a GPU has an in-order stream and two copy streams
 * that overlap it, ordered against it on the device with events. Copies between GPUs go through host memory
 * (ns_copy_peer). */
#ifndef NS_H
#define NS_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

const char* ns_last_error(void);
const char* ns_version(void);

/* the CUDA GPUs, in the order ns_gpu_open takes; name of index i into buf */
int ns_gpu_count(void);
int ns_gpu_name(int index, char* buf, size_t len);
int ns_gpu_pci(int index, char* buf, size_t len);
int ns_gpu_units(int index, int* units, int* mhz);
/* opens GPU `index`: its stream, its two copy streams */
int ns_gpu_open(int index, ns_gpu** out);
void ns_gpu_close(ns_gpu* g);
/* the stream, as a kernel launch takes it (the pointer is a cudaStream_t) */
void* ns_gpu_queue(ns_gpu* g);
/* bytes: total device memory, and free (-1 when the driver does not say) */
int ns_gpu_memory(ns_gpu* g, uint64_t* total, int64_t* free_bytes);

int ns_alloc(ns_gpu* g, size_t bytes, void** out);
int ns_free(ns_gpu* g, void* p);
/* pinned host memory of this GPU's context: the fast source and target of its copies */
int ns_alloc_host(ns_gpu* g, size_t bytes, void** out);
int ns_free_host(ns_gpu* g, void* p);
/* queued on the GPU's queue (in order); ns_sync waits for everything queued */
int ns_copy_to(ns_gpu* g, void* dst, const void* src_host, size_t bytes);
int ns_copy_from(ns_gpu* g, void* dst_host, const void* src, size_t bytes);
int ns_copy_dev(ns_gpu* g, void* dst, const void* src, size_t bytes);
/* `height` rows of `width` bytes, device to device, the rows `dpitch` / `spitch` bytes apart (queued like ns_copy_dev) */
int ns_copy_2d(ns_gpu* g, void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height);
int ns_mark(ns_gpu* g, int64_t* ticket);
int ns_stream_copy(ns_gpu* g, void* dst, const void* src, size_t bytes, const int64_t* after, int ndeps, int lane, int64_t* ticket);
int ns_await(ns_gpu* g, int64_t ticket);
/* the work queued between begin and launch recorded, then launched as one graph (ns.cu); abort: a failed capture ended */
int ns_capture_begin(ns_gpu* g);
int ns_capture_launch(ns_gpu* g);
int ns_capture_abort(ns_gpu* g);
/* the host waits until the work before ticket is done */
int ns_ticket_wait(ns_gpu* g, int64_t ticket);
int ns_stream_copy_batch(ns_gpu* g, void* const* dsts, const void* const* srcs, const size_t* sizes, size_t n, const int64_t* after, int ndeps, int lane, int64_t* ticket);
int ns_stamp(ns_gpu* g, int64_t* ticket);
int ns_elapsed(ns_gpu* g, int64_t t0, int64_t t1, double* ns);
int ns_fill(ns_gpu* g, void* dst, uint8_t value, size_t bytes);
int ns_sync(ns_gpu* g);
/* device memory of one GPU to another's, through `staging`: host memory of `bytes` (pinned memory belongs to one
 * context, so it is pinned for at most one of the two copies; plain memory works for both, slower); waits */
int ns_copy_peer(ns_gpu* to, void* dst, ns_gpu* from, const void* src, size_t bytes, void* staging);
/* As ns_copy_peer through pinned staging (ns_alloc_host), ordered on the devices only: after the work queued on both so
 * far, before the work queued on either later; the host does not wait (the GPUs run side by side) */
int ns_copy_peer_async(ns_gpu* to, void* dst, ns_gpu* from, const void* src, size_t bytes, void* staging);

#ifdef __cplusplus
}
#endif
#endif
