# ggml-cuda's MMQ, vendored for the qwen35 engine's prompt products

From the llama.cpp tree brainiac-llm builds (`~/brainiac_llm/llama.cpp`, commit 4098fdc92, 2026-09-22), MIT
(LICENSE): `ggml/src/ggml-cuda/{mmq.cuh, mmq-config-*.cuh, mmq-load-tiles.cuh, mmq-vec-dot.cuh, mma.cuh, vecdotq.cuh,
common.cuh, quantize.cu, quantize.cuh, vendors/cuda.h}`, `ggml/src/{ggml-common.h, ggml-impl.h}` and
`ggml/include/{ggml.h, gguf.h, ggml-cuda.h, ggml-backend.h, ggml-alloc.h}` (flattened into this folder).

One change: `mmq.cuh`'s launchers take `ns_mmq_ctx & ctx` instead of `ggml_backend_cuda_context & ctx` (they only
use it for `ctx.pool(id)`, the stream-k fixup buffer). `../../q35_mmq.cu` defines `ns_mmq_ctx`, the few ggml-cuda.cu
symbols the headers need (`ggml_cuda_info`, `ggml_cuda_error`, `ggml_abort`), and the C ABI the engine binds
(q35.h: `ns_q35_mmq*`). Nothing here runs Python or ggml itself: it is the int8 tensor-core products only.
