---
name: cuda-porting
description: Port or tune a GPU model engine on NVIDIA CUDA behind nextcuda's contracts - safety, checking against a reference, clocking with the kind's bench (stages, peaks, energy), and putting the tuned kernels in the model's silo - with host code in Rust
---

# Porting and tuning an engine on NVIDIA CUDA

The method nextsycl uses for its SYCL engines, kept for CUDA: the order of the sections is the order to work in, and
each step leaves a measurement behind.

## 1. Safety first

- **CUDA numbers GPUs fastest-first.** Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` (the program does) so index 0 is
  nvidia-smi's 0; a process that touches CUDA before it is set gets the other order for its life.
- **Budget memory before allocating.** `cudaMalloc` fails cleanly, but a model that does not fit still takes the
  card from everything beside it: weights + caches + activations + work buffers against `ns_gpu_memory`'s free, and
  refuse with a message.
- **After every GPU test**, look at the kernel log: `journalctl -k -b | grep -ci xid`. A count that moved means stop
  and find out why before the next run.
- **Never kill an engine mid-kernel;** stop it through its own exit path (`nextcuda image stop`, or the service).
- **Test through the real server** after any change to memory: it holds more than a one-shot run.
- **Never write over a library a process has mapped.** Build to a temporary name and rename over it (a rename keeps
  the old file for whoever mapped it); `kernels/build.sh` does, and a copy in place once killed a bench mid-run.
- **Pause the neighbours:** a clock taken while another job uses the card is not a clock. Hold the queues that share
  it (and release them after: a bench script puts the release in a `finally`).

## 2. Exact before fast

1. **Fixed inputs:** the same prompt, seed, size and steps for every comparison.
2. **Kernels against the host:** `nextcuda image selftest --gpu N` runs each of the engine's kernels on small inputs
   and compares with the same arithmetic on the CPU (relative error limits in `selftest.rs`).
3. **An engine against a reference:** the picture of the same seed from the reference implementation (the model's own
   code, or the library the engine is built on) viewed beside the engine's; the same subject, composition and
   detail. Numeric stage checks (`image check`) when the reference can dump its tensors.
4. **A silo against the shared kernels:** run with `NS_SILO=0` and without; the outputs must agree to the
   tolerance the kernel's self-test sets.

## 3. Clock it

`nextcuda image bench` (see `docs/benchy/README.md`): fixed inputs, cold and warm kept apart, every stage timed,
peak memory, power and energy sampled in the run's window, `matrix.md` and `matrix.jsonl` written.

- **Where the time goes before changing anything.** The bench's stage columns say whether it is text encode, the
  sampler's steps or the VAE; per-kernel time comes from `nsys profile` / `ncu` when installed, or from the
  `NS_PROFILE=gpu` stamps (`ns_stamp` / `ns_elapsed`, events on the GPU's stream) around a suspect.
- **Rule out the link before the kernels.** Run once with the weights resident and once offloaded
  (`NC_SD_OFFLOAD=0|1`): the same step time says the step is compute, not PCIe.
- **Compare like with like.** Same card, driver, size, steps and prompt; a different size is a different row.
- **Quote both.** A change is described by the matrix before and the matrix after, each saved under
  `docs/benchy/` with a name that says what changed.

## 4. Tune in the silo

The shared kernels (`kernels/ns`, an engine's own another engine calls) stay as the engines on them were tuned. A
model's faster version of an op goes in `kernels/<kind>/<arch>/silo/`:

1. Copy the shared kernel there, then change the copy (tile sizes, tensor-core MMA, fused epilogues, int8 paths).
2. Give each silo op a `_supported(shape)` check; the engine asks per call and falls back to the shared kernel.
3. Build it into `dist/silo/libnextcuda-<arch>.so` (`kernels/build.sh`); the engine opens it at load.
4. Check it (section 2.4), clock it (section 3), keep the matrices.

## 5. Host code in Rust

- A kernel's launch checks the launch (`NS_LAUNCH_CHECK()`); asynchronous errors surface at `ns_sync`.
- Buffers free themselves (`DevBuf`); a free waits for the stream, so work queued on a buffer runs first.
- Progress and cancel go through the engine's `generate` callbacks; a cancel stops the step in progress.
- Settings are `NC_*` variables with a row each in `docs/environment.md`; the architecture test fails on a missing one.

## 6. Checklist for a change

- [ ] `cargo test --workspace` and `nextcuda image selftest` pass
- [ ] outputs checked against the reference (or the shared kernels, for a silo)
- [ ] `nextcuda image bench` before and after, saved in `docs/benchy/`, the change named in `--note`
- [ ] peak GPU memory and energy in the table did not get worse without a reason given
- [ ] the kernel log has no new Xid
- [ ] settings added are in `docs/environment.md`; speeds in the README's table
