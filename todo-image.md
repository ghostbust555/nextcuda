# todo: image (Qwen-Image 2.1 and 2512 on the 5070)

Where things stand after the Rust engine and int8 (2026-10-16). Clocks are in `docs/PHASES.md`.

## Open

- [ ] **int8 quality.** 30.4 dB PSNR against half; nextsycl documents about 39 dB. The picture looks right, the cause is
      not found. Look at: which layers are sensitive (first/last blocks, modulation, `out`), rotation precision on the
      weights, per-group scale vs per-row, keeping a few matrices half. Decide default vs opt-in once it is understood
      (`NS_QI_INT8=1` is opt-in today).
- [ ] **Attention speed.** The silo's flash attention is about 50 TFLOPS; nextsycl reaches about 110. Use ldmatrix and
      cp.async, vectorized V loads (D=128 only today).
- [ ] **Qwen-Image-2512 in Rust.** 20B dual-stream MMDiT + Qwen2.5-VL; still stable-diffusion.cpp. nextsycl has no
      engine for it, so this is a new port on the shared `nsd` kernels (text encoder at `~/nextcuda/models/qwen-image-2512/`,
      `qwen_image_vae.safetensors`, Lightning LoRA, effort ladder).
- [ ] **Rust engine as the default backend** (after int8 quality and rrq4/2512 checks); `native` stays as the reference.
- [ ] **`image check` stage checks.** No reference dumps exist for CUDA; make them from sd.cpp for the same seed.
- [ ] **Cold loads.** Host I/O prefetch / `posix_fadvise`; text-encoder layer streaming over the 3060's gen4 x1 link.
- [ ] **Text encoder in int8** (optional).
- [ ] **Re-run the 3060 benches** with the Rust engine and int8; test rrq4 and 2512 on the Rust engine.
- [ ] **Peak memory at 1024²** is 10.6-11.1 GB of 12 GB: no headroom for a second job or a larger picture.
- [ ] **Bench report**: the rrq4 section prints its last size row twice.

## Done

- Rust engine: samplers and schedules, sigma presets, img2img, edits with up to 8 references, LoRAs, RGBA, progress
  and cancel; beats stable-diffusion.cpp (1024x1024 8 steps: 20.2 s vs 26.3 s half, 10.9 s int8).
- nextsycl's documented optimizations: flash attention, fused layer norm + modulation, ConvRot int8, AOT builds.
