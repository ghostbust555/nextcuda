# nextcuda: phases

nextcuda is nextsycl's control plane and API (front door, jobs, queue, precheck, contracts, CLI, web front ends) with
NVIDIA behind it instead of SYCL. The platform talks to it with the same `/rpc/<kind>.<method>` API.

## Phase 1 (this tree): wired up

- **Foundation:** `crates/core` is NVIDIA-only (`nvidia-smi`; buffers are host memory: engines own their GPU memory).
  `glue/serve/src/gpustat.rs` samples cards with `nvidia-smi` for the front door's placement.
- **Engines:** `image/qwenimage21`: Qwen-Image 2.1 through stable-diffusion.cpp (`sd-cli`, CUDA), driven by the
  platform's own python engine (`image/qwenimage21/python/`, vendored) over JSON lines. Arch name, file roles and
  contract are the SYCL engine's, so the registry, model ids and API are the same.
- **Models:** `model-cards/image/qwen-image-2.1-{q8,q6k}.yml` are the platform's files (leejet GGUF, Qwen3-VL GGUF +
  projector, VAE).
- **Run:** servers are plain processes (`src/native.rs`), not containers; `NS_CONTAINER_ENGINE` brings containers
  back. Build: `./build.sh` (debian:trixie toolchain, glibc matches the GPU host) -> `dist/`.
- **Not here yet:** llm, video (H3), audio (speech, music), OCR, 3D; the turbo preset (a fused LoRA, a local part).
  `image check`, the stage checks and the kernels directory are gone with SYCL.

## Phase 2 progress

- **Done (verified on the 5070 and 3060):** `kernels/ns/ns.cu` (the ns.h ABI on CUDA: devices, memory, streams,
  copies, tickets as events), `kernels/image/qwenimage21/qi21.cu` (the engine's own kernels, ported), `crates/sys`
  (dlopen) and a device-backed `crates/core` (host-memory fallback without the library), `kernels/build.sh`
  (nvcc, sm_86 + sm_120 + PTX), `nextcuda image selftest` (every qi21 kernel against host arithmetic: ok).
- **Next, in order:** `kernels/diffusion/nsd.cu` (H3's shared kernels: int8 ConvRot linear, rms norm + modulation,
  rope, attention, dequant, conv; oneDNN's matmul and fused SDPA become cuBLASLt and a CUDA flash attention);
  `crates/qwen3vl` (the text encoder), the DiT (`dit.rs`), the VAE (`vae.rs`), the sampler; then `check` against
  sd.cpp's pictures (the python engine is the reference); then the silo (`kernels/image/qwenimage21/silo`).
- **Qwen-Image-2512** is a different architecture from 2.1 (a 20B dual-stream MMDiT with a Qwen2.5-VL encoder, not
  the 7B single-stream DiT with Qwen3-VL), so it is its own engine crate on the same foundation; it follows 2.1's
  nsd port, which it shares most kernels with.

## Phase 2 progress (the Rust engine on the shared CUDA kernels)

- **Done, checked on the 5070:** `kernels/diffusion/nsd.cu` (cuBLAS linears, a quantized linear that keeps GGUF
  weights in their stored type and makes a slice of rows half at a time, RMS norm + modulation, RoPE, SwiGLU, gated
  add, attention for grouped heads, im2col convolutions); `nextcuda image selftest` runs each against host
  arithmetic (dequant of Q8_0 / Q4_K / Q5_K / Q6_K exact; the rest within 1e-3).
- **Running, checked on the 5070 (2026-10-16):** nextsycl's engine as `image/qwenimage21/src/rust/` - the DiT with
  its blocks in the type the GGUF stores (mixed-quant files included), the VAE, every ComfyUI sampler and schedule,
  sigma presets, img2img with strength, edits with up to 8 references through the vision tower, LoRAs as side
  paths, RGBA, progress and cancel; a text encoder read from a llama.cpp GGUF a layer at a time with a host cache
  and an embedding cache (`crates/qwen3vl/src/gguf.rs`). Selected with `NC_IMAGE_BACKEND=rust`.
- **nextsycl's documented optimizations, done:** fused flash attention in the silo (18.3 -> 5.5 ms a call at 4k
  tokens), layer norm + modulation fused, ConvRot int8 W8A8 (`NS_QI_INT8=1`; cuBLASLt int8 tensor cores, about 2x a
  step here against 1.4x on the B70), AOT per-card builds, the kernels checked against host arithmetic.
- **Not done:** `image check` stage dumps (no reference dumps exist for CUDA), the Rust engine for Qwen-Image-2512
  (still stable-diffusion.cpp), int8 at the quality nextsycl documents (30.4 dB PSNR against half here, about 39 dB
  there; the picture looks right, the cause is not found), attention at nextsycl's ~110 TFLOPS (ldmatrix and
  cp.async not used yet, about 50 now), the Rust engine as the default.

### Rust engine clocks (RTX 5070, 8 steps guided, `nextcuda image bench`, 3 seeds, warm median)

| model | picture | stable-diffusion.cpp | Rust, half | Rust, int8 |
|---|---|---:|---:|---:|
| qwen-image-2.1 Q8_0 | 768x768 | 13.4 s | 11.2 s | 5.4 s |
| qwen-image-2.1 Q8_0 | 1024x1024 | 26.3 s | 20.2 s | 10.9 s |
| qwen-image-2.1 Q4 (rrq4) | 1024x1024 | - | - | 10.9 s |

Peak GPU memory with int8: 9.5 GB (768) and 10.6-11.1 GB (1024): close to the 12 GB limit. Edit with references:
45 s (sd.cpp 89 s); turbo LoRA at 4 steps 5.4 s. Matrices: `benchy-results/image-v1-5070-rust2` and `-rust-int8`.

## Clocks (RTX 5070 12 GB, in process through stable-diffusion.cpp, 2026-10-15)

| model | picture | text encoder | sampler | VAE | total |
|---|---|---|---|---|---|
| qwen-image-2.1 Q8_0 | 768x768, 12 steps | 0.8-1.2 s | 1.27 s/step (16.5 s) | 2.4 s | 19.6 s warm, 30 s after a cold page-in |
| qwen-image-2.1, DiT resident on the card | same | 0.84 s | 1.27 s/step | 2.6 s | 22 s: the step is compute, not PCIe |
| qwen-image-2512 Q4_K_M, Lightning 4 steps | 1024x1024 | 0.8 s | 2.8 s/step | 4.7 s | 63 s (about 45 s of it the first graph streaming 13 GB in) |
| qwen-image-2512 Q4_K_M, full | 1024x1024, 24 steps | 0.8 s | 4.5 s/step | 4.7 s | 121 s |

The phase 1 worker made the 2.1 picture in 31.8 s; in process it is 19.6 s because the load is paid once. For
reference nextsycl's B70 makes Qwen-Image 2.1 at 1024x1024 in 0.6 s/step (0.41 with int8 weights), with about twice
the tokens of the 768 picture here: the 5070 has room for roughly 3x. Where it goes is the silo's first target
(the quantized matmuls and attention at 2,400-4,400 tokens).

## Phase 2: nextsycl's Rust-only runtime setup

Bring in how nextsycl runs engines without Python (the `engine` crates, `nextcuda-sys` over `dist/libnextcuda-<kind>.so`
built from CUDA kernels), one engine at a time behind the same contracts; the python worker for an engine goes when
its Rust one passes the same checks.

## nextsycl's contracts, declared ahead of their engines (2026-10-17)

Every API contract nextsycl has is in place here, so a client sees one API on both boxes. A method or engine whose
port has not landed answers with what it would take and does, and refuses the call with a reason:

- **Speech engines:** `audio/qwen3tts`, `audio/voxcpm2`, `audio/breezetts2` (`nextcuda audio engines`) carry nextsycl's
  registry entries, file roles, options and abilities (`speech()`), and their cards are in `model-cards/audio/`.
  Loading says "not ported to CUDA yet". The audio contract has nextsycl's `direction` and `events` (Breeze TTS 2's
  voice direction and vocal events), and `audio.speak` its Breeze fields and checks: `cfg`, a clone with instructions
  going to a model that directs, a transcript for a model that clones only with one.
- **3D:** `model3d/contract` and `crates/mesh` (GLB / OBJ and the CPU mesh tools) as nextsycl's. The `model3d.*`
  methods are in the spec with their fields, answers and errors, and answer 501 `not_implemented`.
- **Drives:** `system.drives`, `system.drive.measure`, `system.move` and `system.move.cancel` likewise: declared, 501.
- `system.version` lists the methods of the kinds that are on, so none of these show there until they work.

Then the same day: the three speech engines run on CUDA in Rust (no Python) - Breeze TTS 2, Qwen3-TTS and VoxCPM2,
nextsycl's code over CUDA ports of its audio kernels (`kernels/audio/`), each checked against its upstream package
(README "Speech"). Still declared only: model3d, the drives, music (MiniMax Music 3), OCR, video.
