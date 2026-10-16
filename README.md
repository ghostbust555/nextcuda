<p align="center"><img src="docs/logo-card.svg" alt="nextcuda - the swiss-army compute engine: video, chat, image, audio" width="520"></p>

We stand on the shoulders of giants.

# nextcuda

- **nextsycl's control plane, on NVIDIA** - the same front door on :8000, the same `/rpc/<kind>.<method>` API, the same
  jobs, queue and model registry; what runs the models under it is CUDA
- **No Python** - Rust on the control side, CUDA ([stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)'s
  ggml-cuda kernels) on the GPU side; the model stays loaded between pictures
- **Made for the cards people have** - Qwen-Image 2.1 and 2512 run from GGUF on a 12 GB card, weights streamed from
  host memory per graph
- **A single API control plane** - one service, `nextcuda.service`, answers on :8000 and starts a model server when a
  request names its model
- **A Docker-style CLI** - `nextcuda <kind> start | stop | ps | logs | ...`, as a process, not a container
- **Silo architecture** - a model's tuned kernels live in its own library; the shared ones stay as they are
- **Test driven** - the architecture rules, the model cards and the kernels are checked by `cargo test` and
  `nextcuda image selftest`

It is the same program as [nextsycl](https://github.com/) with the SYCL runtime taken out and CUDA put in, so what
you know of one is what you know of the other. This page is only what runs here today.

## RUN

`nextcuda models pull <id>`, then ask the front door (:8000) for a picture - or `nextcuda image gen "<prompt>"`.
Measured on an RTX 5070 (12 GB; the box also has an RTX 3060), in process, weights offloaded to host memory:

| id | model | picture | speed |
|---|---|---|---|
| `qwen-image-2.1-q8` | **Qwen-Image 2.1** Q8_0 (7.3 GB), text to image | 768x768, 12 steps | 1.27 s/step, VAE 2.4 s, 19.6 s in all |
| `qwen-image-2.1-rrq4` | Qwen-Image 2.1 Q4 (realrebelai, 5.7 GB) | 768x768, 12 steps | the same 1.27 s/step, 21 s in all |
| `qwen-image-2512-q4km` | **Qwen-Image 2512** Q4_K_M (13.2 GB), 4 steps with the Lightning LoRA | 1024x1024 | 2.8 s/step, 63 s cold (first graph streams the weights in) |
| `qwen-image-2512-q4km` | the same, a full guided run | 1024x1024, 24 steps | 4.5 s/step, 121 s |

GPU memory (the model's own, peak over a picture; `docs/benchy/image-v1-2026-10-16-first.md`):

| id | 768x768 | 1024x1024 | weights resident on the card |
|---|---:|---:|---:|
| `qwen-image-2.1-q8` | 8.2 GB | 8.6 GB | 10.6 GB (768²) |
| `qwen-image-2.1-rrq4` | 6.4 GB | 6.8 GB | - |
| `qwen-image-2512-q4km` | 11.0 GB | 11.0 GB | - |

2512 leaves under 1 GB of a 12 GB card at 1024²; the same model with a larger canvas needs `NC_SD_MAX_VRAM`.

The text encoder takes 0.8-1.2 s. With the 2.1 transformer resident on the card the step is the same 1.27 s, so the
step is compute and not the PCIe link: the quantized matmuls and attention are what the silo is for
(`docs/PHASES.md` has the clocks and where nextsycl's B70 is for comparison).

## Supported models

| id | model | engine | on disk | status |
|---|---|---|---|---|
| `qwen-image-2.1-q8` | Qwen-Image 2.1 Q8_0 with Qwen3-VL 8B Q4_K_M, its vision projector and the RGBA VAE | `qwen-image-2.1` | 13.6 GiB | served, checked |
| `qwen-image-2.1-q6k` | the same with a Q6_K transformer | `qwen-image-2.1` | 12.0 GiB | listed |
| `qwen-image-2.1-rrq4` | [realrebelai's Q4](https://huggingface.co/realrebelai/Qwen-Image-2.1_GGUFs): a Q8_0 / Q6_K / Q5_K mix | `qwen-image-2.1` | 11.9 GiB | served, checked |
| `qwen-image-2512-q4km` | Qwen-Image 2512 Q4_K_M with Qwen2.5-VL 7B Q4_K_M, its VAE and the Lightning 4-step LoRA | `qwen-image-2512` | 17.7 GiB | served, checked |

The model cards are in `model-cards/` (one YAML a model, every file a pinned link with its size and SHA-256).
`nextcuda models pull <id> [--from DIR]...` downloads what is missing or hard-links a file already on the machine,
checks it and registers the model.

## What it does

- **The front door on :8000:** `/rpc/<kind>.<method>` with a bearer token (`serve.json`, or `NS_API_TOKEN`); every
  answer is the envelope `{ok, result}` or `{ok: false, error: {code, message, retryable, ...}}`. `image.generate`
  and `image.edit` run as jobs (`wait: false`, `label`, `reentrant_id`), asked for with `image.status` /
  `image.jobs`, stopped with `image.cancel`; `image.upload`, `image.queue.*`, `system.busy`, `system.cards`,
  `system.models` and `system.version` (which lists its features) are there as they are on nextsycl.
- **A model server a model, on demand:** the first request for a registered model starts `nextcuda image serve` for
  it on the first card with room, swaps it when a request names another, and lets it go after the idle time.
- **Qwen-Image 2.1:** text to image (RGBA with `rgba`), img2img with a `strength`, edits and compositions from up to
  8 reference pictures with an optional mask, the effort ladder below.
- **Qwen-Image 2512:** text to image and img2img. Under 20 steps the Lightning LoRA runs (4 steps, guidance 1); at 20
  or more it is bypassed for a full guided run - the platform's effort ladder, so an engine behind it behaves as
  the one it replaces.
- **A 12 GB card is enough:** the weights stay in host memory and are staged to the card per graph
  (`NC_SD_OFFLOAD`, `NC_SD_MAX_VRAM`); the text encoder, the transformer and the VAE take turns.
- **Community GGUFs are accepted as they are:** a file that stores `img_in.weight` reshaped (realrebelai's does) gets
  a corrected copy made once beside it; your file is not changed.
- **Progress and cancel:** every step is reported as it is made and a cancel stops the step in progress.
- **Energy and timing in the answer:** `nextsycl.seconds`, `steps`, `seed` come back in each result (the key keeps
  nextsycl's name: the platform reads it).
- **Servers as processes:** `nextcuda image start` is a process with a pid and a log in the socket directory;
  `NS_CONTAINER_ENGINE` brings containers back.

## Quick start

```sh
nextcuda models pull qwen-image-2.1-q8 --from ~/models/qwen21   # link the files you already have
nextcuda image gen "a small red fox in fresh snow, soft morning light" --model qwen-image-2.1-q8 --size 768x768 \
        --steps 12 --seed 7 --out fox.png --local
nextcuda gpus                                   # each GPU: memory, units, copy rates
nextcuda image selftest --gpu 0                 # every kernel of the engine against host arithmetic
```

Through the front door:

```sh
TOKEN=$(python3 -c "import json;print(json.load(open('serve.json'))['token'])")
curl -s -X POST http://127.0.0.1:8000/rpc/image.generate -H "Authorization: Bearer $TOKEN" \
     -d '{"model": "qwen-image-2.1-q8", "prompt": "a lighthouse at dusk", "size": "768x768", "steps": 12,
          "wait": false, "label": "lighthouse", "reentrant_id": "lh-1"}'
curl -s -X POST http://127.0.0.1:8000/rpc/image.status -H "Authorization: Bearer $TOKEN" -d '{"reentrant_id": "lh-1"}'
```

## Run it as a service

`tooling/nextcuda.service` is a systemd user unit (`:8000`, restarted on failure); `tooling/deploy.sh [host]` builds
here, builds the CUDA libraries where `nvcc` is, copies it all over and restarts the unit:

```sh
./tooling/deploy.sh gpu-host            # an ssh host with CUDA; SKIP_BUILD=1, SKIP_KERNELS=1, RESTART=0 to do less
systemctl --user status nextcuda.service
journalctl --user -u nextcuda.service -f
```

Settings are `nextcuda.conf` beside `dist/` (or `~/.config/nextcuda.conf`), then the environment; every variable is
in `docs/environment.md`.

## Engine options: `--opt-NAME`

| option | variable | |
|---|---|---|
| `--opt-offload 0\|1` | `NC_SD_OFFLOAD` | weights in host memory, staged per graph (default 1) |
| `--opt-max-vram GiB` | `NC_SD_MAX_VRAM` | the card memory the weights and buffers may take |
| `--opt-sd-args k=v;...` | `NC_SD_ARGS` | `threads=N`, `flash_attn=0\|1`, `vae_tiling=0\|1` |
| `--opt-backend native\|python` | `NC_IMAGE_BACKEND` | in process (default) or the phase 1 sd-cli worker |

## How it is built

```
kernels/ns            the C ABI (GPUs, memory, streams, copies) on CUDA - nextsycl's ns.h, the same functions
kernels/image/qwenimage21   the engine's own kernels (qi21.cu)
kernels/image/sd      a flat C ABI over stable-diffusion.cpp's ggml-cuda: one loaded model, progress, cancel
crates/core           the safe layer over libnextcuda-<kind>.so (host-memory fallback without it)
image/qwenimage21     the engines: Qwen-Image 2.1 and 2512 behind nextcuda-image's contract
glue/serve, glue/models, cli/nextcuda    the control plane, the registry and the program, as nextsycl's
```

The rules of nextsycl's `docs/architecture.md` hold - dependencies point down, one library a kind, tuning in silos -
and `cli/nextcuda/tests/architecture.rs` checks them on every `cargo test`, with CUDA as the one GPU runtime.

`./build.sh` builds the program in a debian:trixie toolchain image (its glibc is the GPU host's) and the web front
end; `kernels/build.sh` builds the CUDA libraries with `nvcc` for the 3060 (sm_86) and the 5070 (sm_120).

## What is not here yet

llm, video (H3), speech and music, OCR and 3D: their contracts are in the tree and the control plane routes them,
but no engine of ours runs them. The Rust-only runtime nextsycl has (its own kernels, no ggml) comes engine by
engine behind the same contracts (`docs/PHASES.md`); the python worker of phase 1 stays as a fallback until then.

## Clocking it

`nextcuda image bench` is the image kind's benchy: fixed prompt and seeds, cold and warm pictures kept apart, every
stage timed, peak GPU memory, power and energy sampled beside each picture, `matrix.md` and `matrix.jsonl` written.
The method (what to capture, how a result is kept and compared, how an engine is tuned in its silo) is
`docs/benchy/README.md` and `skills/cuda-porting/SKILL.md`.

```sh
nextcuda image bench --models qwen-image-2.1-q8,qwen-image-2512-q4km --sizes 768x768,1024x1024 --steps 8 --runs 3 \
        --note "what is different about this run"
```

## Checking it

```sh
cargo test --workspace            # the architecture rules, the model cards, the samplers' math
nextcuda image selftest --gpu 0   # the CUDA kernels against host arithmetic
```

## Lessons learned

- **CUDA's GPU numbers are not nvidia-smi's.** CUDA puts the fastest card first; `CUDA_DEVICE_ORDER=PCI_BUS_ID` makes
  card 0 the one nvidia-smi calls 0. The program sets it before any CUDA call.
- **The step is not the PCIe link.** Keeping the 7.3 GB transformer resident on the 5070 gave the same 1.27 s/step as
  streaming it: the time is in the matmuls and attention, so that is where the silo goes.
- **A community GGUF is a converter's choice.** One stores `img_in.weight` as [1024, 256] for [4096, 64]; the loader
  takes the model's hidden size from it and refuses the file. A shape fix in a copy is the right size of answer.
- **The first picture pays the page-in.** The first graph after a load streams the weights from disk to host memory
  (30-45 s for a 13 GB model); the second picture does not.

## Standing on

- [nextsycl](https://github.com/) - the control plane, the contracts, the web front end and the rules.
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) and [ggml](https://github.com/ggml-org/ggml)
  - the CUDA kernels the engines run on.
- Qwen-Image 2.1 and 2512 (Qwen), the GGUFs of leejet, unsloth and realrebelai, and Viggle's and lightx2v's LoRAs.

## Docs

- `docs/PHASES.md` - what is done, the clocks, and what comes next
- `docs/architecture.md` - the layers, the contracts, the silos (nextsycl's, which holds)
- `docs/environment.md` - every variable
- `model-cards/README.md` - the model card format
- `CONTRIBUTING.md` - the rules and how a model is ported
