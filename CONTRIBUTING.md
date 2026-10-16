# Contributing

nextcuda welcomes new models - image first, then language, video and audio - and work on the ones it has. Each model
architecture is its own **engine**: a complete runtime for that model, free to be as specialised as it needs to be
fast. This guide is the few rules every change follows; [skills/cuda-porting/SKILL.md](skills/cuda-porting/SKILL.md)
is the method (how to check, clock and tune an engine) and `docs/benchy/README.md` the format of a result.

## The rules

1. **CUDA is the only GPU runtime.** Kernels are CUDA (nvcc), on the GPU's own stream; cuBLAS and ggml-cuda are
   fine. No SYCL, Vulkan, OpenCL, HIP, Metal, wgpu, or a framework that brings another (candle, PyTorch, ONNX
   Runtime, burn...). Host code is Rust. Python is a fallback for an engine whose Rust one is not there yet, never the
   first choice.
2. **Third-party crates are fine** - anything that does not bring another GPU runtime (rule 1). Prefer a few
   well-known ones; say why in the pull request.
3. **Keep the architecture** (`docs/architecture.md`):
   - an engine is its own crate under its kind (`image/<arch>`, ...), with its kernels in `kernels/<kind>/<arch>/`;
   - it depends only on the foundation (`crates/`) and its own kind's contract (`<kind>/contract`) - never on another
     kind, the glue (`glue/`) or the program (`cli/`);
   - what two kinds both need moves down into the foundation, not sideways;
   - its kernels' C functions are named `ns_<kind>_<arch>_*`, declared in its own header, and bound by its own crate
     by name - never added to the shared `kernels/ns/ns.h`.
4. **Exact before fast.** An engine is checked against a reference before it is tuned, and every speed claim comes
   with the measurement that made it: `nextcuda image bench` before and after, both quoted.
5. **Tune in the model's silo, never in the shared kernels.** A kernel tuned for one model goes in that model's
   silo, `kernels/<kind>/<arch>/silo/` - its own library, `dist/silo/libnextcuda-<arch>.so`, which the engine opens at
   load and calls for what it covers, the shared kernel for everything else (`NS_SILO=0` turns it off). Start a silo
   kernel as a copy of the shared one, then change the copy.

Rules 1 and 3 are checked by `cargo test` (`cli/nextcuda/tests/architecture.rs`).

## Building

```sh
./build.sh                        # the program and the web front end, in a debian:trixie container: dist/
kernels/build.sh                  # the CUDA libraries, with nvcc, on a host that has it (OUT=... for where)
cargo test --workspace            # the architecture rules, the model cards, the unit tests
nextcuda image selftest --gpu 0   # the engine's kernels against host arithmetic, on the GPU
```

## Adding a model

1. **A model of an architecture already served needs only its card**: `model-cards/<kind>/<id>.yml` (every file a
   pinned link with its size and SHA-256; `model-cards/README.md` has the format), then `nextcuda models pull <id>`.
2. **A new architecture is an engine crate** registered in the program's list of that kind
   (`cli/nextcuda/src/image.rs`: `engines()`), with a row in `docs/environment.md` for every setting it reads (the
   architecture test fails on one that has none).
3. **Then make it fast**, as its own change: clock it with the kind's bench, find where the time is, put the faster
   kernels in its silo, check the results are the same and clock it again.

## Code style

- Write like the code around you: the same comment density, naming and idioms. Comments say what something is and
  why, in plain words.
- `cargo clippy --all-targets -- -D warnings` is clean; `unsafe` blocks carry a `// SAFETY:` line.
- CUDA kernels: every exported function is `extern "C"`, returns 0 or -1 with `ns_fail(reason)`, and queues its work
  on the GPU's stream; a launch is checked with `NS_LAUNCH_CHECK()`.
