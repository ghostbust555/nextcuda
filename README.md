<p align="center"><img src="docs/logo-card.svg" alt="nextcuda - the swiss-army compute engine: video, chat, image, audio" width="520"></p>

We stand on the shoulders of giants.

# nextcuda

- **nextsycl's control plane, on NVIDIA** - the same front door on :8000, the same `/rpc/<kind>.<method>` API, the same
  jobs, queue and model registry; what runs the models under it is CUDA
- **No Python** - Rust on the control side, CUDA ([stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)'s
  ggml-cuda kernels) on the GPU side; the model stays loaded between pictures
- **Made for the cards people have** - two 12 GB cards (an RTX 5070 and an RTX 3060) run chat, pictures, OCR, speech,
  songs, video with sound and 3D: Qwen3.8-27B reads prompts at ~1,200 tok/s over both cards (llama.cpp: 925),
  MiniMax H3 makes video with sound across both, Nemotron 3.5 Super (120B) chats on one, its experts streamed
- **Every kind behind one door** - 21 models served here, 7 more a click away (`Not downloaded` on the page), each
  started on the card with room when a request names it and let go when idle
- **A single API control plane** - one service, `nextcuda.service`, answers on :8000 and starts a model server when a
  request names its model
- **A Docker-style CLI** - `nextcuda <kind> start | stop | ps | logs | ...`, as a process, not a container
- **Silo architecture** - a model's tuned kernels live in its own library; the shared ones stay as they are
- **Test driven** - the architecture rules, the model cards and the kernels are checked by `cargo test` and
  `nextcuda image selftest`

It is the same program as [nextsycl](https://github.com/) with the SYCL runtime taken out and CUDA put in, so what
you know of one is what you know of the other. This page is only what runs here today.

## The control plane

`http://<host>:8000/` - every card, what each holds and what the enabled models are vouched for there; every model
by kind with its cards (tick which may hold it), enable / disable, start / stop and its own page; the models a card
offers that are not downloaded yet; the front ends (Open WebUI on :3000, the video studio). The API and Config tabs
are the OpenAPI document and `nextcuda.conf`.

<p align="center"><img src="docs/screenshots/control-plane-gpus.png" alt="the GPUs: memory in use by process, and what each enabled model is vouched for" width="900"></p>

<p align="center"><img src="docs/screenshots/control-plane-models.png" alt="the models by kind: chat, images and OCR, audio, video - their cards, enable, start and their pages" width="900"></p>

<p align="center"><img src="docs/screenshots/control-plane-downloads.png" alt="the models not downloaded, a download each, and the front ends" width="900"></p>

- **Downloads on demand:** a request for a model whose files are not here answers `model_not_downloaded` (its size,
  whether it fits, how to pull it); the page's Download (with progress and cancel) or `system.models.pull` fetches it;
  the "download automatically" box (`NS_AUTO_PULL`) lets a request wait for it instead.
- **Cards shared, not fought over:** a model starts on the first of its cards with room; idle picture, sound, video
  and chat servers are stopped for it (lightest first) only when it does not fit beside them. Qwen3.8 keeps ~5.9 GB of
  the 3060 free, so a voice (VoxCPM2 int8, 3.3 GiB) reads its replies aloud without the chat model being swapped out.
- **Maintenance mode** sets the box aside (callers told, queues held) - below.

## RUN

`nextcuda models pull <id>`, then ask the front door (:8000) for a picture - or `nextcuda image gen "<prompt>"`.
Measured by `nextcuda image bench` (8 steps, true guidance: two passes a step; warm pictures, the first after a load in
brackets in the full tables of `docs/benchy/`), weights offloaded to host memory, in process:

**RTX 5070, 12 GB, PCIe gen4 x16**

| id | engine | picture | text encode | per step | VAE | total | peak GPU memory |
|---|---|---|---:|---:|---:|---:|---:|
| `qwen-image-2.1-q8` | stable-diffusion.cpp | 768x768 | 0.7 s | 0.95 s | 1.6 s | 13.4 s | 8.2 GB |
| `qwen-image-2.1-q8` | stable-diffusion.cpp | 1024x1024 | 0.3 s | 1.42 s | 3.5 s | 26.3 s | 8.6 GB |
| `qwen-image-2.1-q8` | **Rust + CUDA** | 768x768 | 2.7 s | 1.62 s | 0.5 s | 16.3 s | 10.6 GB |
| `qwen-image-2.1-q8` | **Rust + CUDA** | 1024x1024 | 2.8 s | 3.26 s | 0.9 s | 29.9 s | 11.6 GB |
| `qwen-image-2.1-rrq4` | stable-diffusion.cpp | 768x768 | 0.4 s | 1.03 s | 1.6 s | 13.8 s | 6.4 GB |
| `qwen-image-2.1-rrq4` | stable-diffusion.cpp | 1024x1024 | 0.3 s | 1.52 s | 3.5 s | 28.0 s | 6.8 GB |
| `qwen-image-2512-q4km` (Lightning, 8 steps) | stable-diffusion.cpp | 768x768 | 0.6 s | 0.42 s | 2.2 s | 16.6 s | 11.0 GB |
| `qwen-image-2512-q4km` (Lightning, 8 steps) | stable-diffusion.cpp | 1024x1024 | 0.3 s | 0.44 s | 4.4 s | 28.5 s | 11.0 GB |

**RTX 3060, 12 GB, PCIe gen4 x1** (the card sits on a x1 link: about 2 GB/s to the host, where the 5070 has about 25)

| id | engine | picture | text encode | per step | VAE | total | peak GPU memory |
|---|---|---|---:|---:|---:|---:|---:|
| `qwen-image-2.1-q8` | stable-diffusion.cpp | 768x768 | 3.0 s | 2.35 s | 4.2 s | 36.9 s | 8.1 GB |
| `qwen-image-2.1-q8` | stable-diffusion.cpp | 1024x1024 | 2.9 s | 3.18 s | 9.0 s | 64.7 s | 8.5 GB |
| `qwen-image-2.1-q8` | **Rust + CUDA** | 768x768 | 7.8 s | 3.81 s | 1.3 s | 39.8 s | 10.6 GB |
| `qwen-image-2.1-rrq4` | stable-diffusion.cpp | 768x768 | 3.0 s | 2.48 s | 4.2 s | 37.5 s | 6.5 GB |
| `qwen-image-2512-q4km` (Lightning, 4 steps) | stable-diffusion.cpp | 768x768 | 4.3 s | 0.75 s | 6.0 s | 37.6 s | 11.3 GB |

The first picture after a load is slower (the weights page in: the 2512 takes 70-85 s on either card). Energy and
average power are in the full tables: the 5070 draws about 220 W while sampling, the 3060 about 140 W.

What the clocks say: the step is compute, not PCIe (a resident transformer gave the same 1.27 s/step); the Rust
engine's VAE is 3x faster than ggml's and its steps are 1.7-2.3x slower, which is where the silo goes (attention at
4,000 tokens is cuBLAS and a score table today, a fused kernel in the silo); its text encoder streams the layers
over PCIe on every prompt, so it costs 2.7 s on the 5070 and 7.8 s on the 3060's x1 link (a host-resident cache of
the encoder is the fix). `docs/PHASES.md` has the history of the numbers.

## Running the LLMs

### Qwen3.8-27B over both cards

**Qwen3.8-27B GSQ-RCO IQ3_S** (`llm/qwen35`; 48 gated DeltaNet + 16 gated attention layers, dense SwiGLU, its MTP
draft head, the vision tower) whole in the two cards' VRAM, nothing streamed: the 5070 does the work (40 of the 64
layers, the head and the draft block), the 3060 supports it (the other 24 layers, the vision tower). 80K context with
pictures (`NS_Q35_VISION=0`: 96K text only), the cache as q8_0. The same file brainiac-llm's llama.cpp serves.

```sh
curl -s http://127.0.0.1:8000/v1/chat/completions -H "authorization: Bearer $TOKEN" -H 'content-type: application/json' \
     -d '{"model": "qwen3.8-27b-gsq-iq3s", "messages": [{"role": "user", "content": "Hello"}],
          "chat_template_kwargs": {"enable_thinking": false}}'
```

Measured on the box against llama.cpp (b11112, its `qwen3.8-vision` / `-text` presets: q8_0 cache, flash attention,
MTP drafts) on the same prompts, greedy:

| | nextcuda | llama.cpp |
|---|---|---|
| a 16.6K-token prompt read | **1,200 tok/s** (CLI), ~1,160 through :8000 | 925 |
| a 3.1K-token prompt read | 1,010 tok/s | - |
| a 256-token story written | **44 tok/s** | 35.4 |
| written after the 16.6K prompt | 44-53 tok/s | 43.6 |

What it took (2026-10-18, from 691 tok/s and 31 tok/s that morning):

- **Both cards at once.** A prompt is pipelined like llama.cpp's ubatches: the 3060's layers of chunk k+1 are queued
  before the 5070's of chunk k, the rows cross by `ns_copy_peer_async` (pinned staging, ordered by events on the
  devices - the host never waits), chunks before the last read nothing back, token rows upload through pinned slots.
- **One CUDA graph a card a chunk.** A chunk is ~1,000 kernels a card; launched one by one they filled the stream's
  queue and the single host thread waited, leaving the 3060 idle ~80 ms a chunk. Captured and launched as one graph
  each, the 3060 is busy 13.2 of 14.3 s.
- **The work on the faster card.** The planner picks the chunk (512 rows) that leaves the 5070 the most layers; the
  cache as q8_0 (272 bytes a head's row, not 512) made room for them.
- **Prompt attention:** the queries in registers, key / value tiles in two `cp.async` stages, the hardware exponent:
  25% faster on the 3060.
- **ggml-cuda's MMQ and MMVQ** for the products (vendored from llama.cpp's tree, `kernels/llm/qwen35/third_party`): the
  shared matrix-vector kernels read the 3060's layers at ~97 GB/s of its 360. With MMVQ two drafts a step is best.

Settings (`docs/environment.md`): `NS_Q35_PIPE`, `NS_Q35_GRAPHS`, `NS_Q35_MMVQ`, `NS_Q35_KV`, `NS_Q35_KV_ROT`, `NS_Q35_DRAFTS` (2),
`NS_Q35_CHUNK`, `NS_Q35_HEAD`, `NS_Q35_VISION`; `NS_PROFILE=gpu NS_Q35_PROFILE=1` times every kernel by card and phase.
`nextcuda llm generate MODEL.gguf --gpu 1,0 --ctx 81920 --prompt-file F --effort none` is the engine without the server.
**The context's format** (`NS_Q35_KV`, `NS_Q35_KV_ROT`): per card, chosen by the plan. Measured after the 16.6K prompt,
256 tokens teacher-forced (`llm generate --score TEXT --ref OUT`) against the half cache, where only changing the chunk
size gives KL 0.00093:

| `NS_Q35_KV` | 3060 / 5070 | KL mean (max) | top-1 same | 80K: 5070 layers, prompt | 192K |
|---|---|---|---|---|---|
| `auto` (default) | half / q8_0 | 0.00054 (0.003) | 97.3% | 40, 1,230 tok/s | q8 / q8: 873 tok/s |
| `q8` | q8_0 / q8_0 | 0.00080 (0.005) | 97.7% | | |
| `q4` | q4_0 / q4_0 | 0.0082 (0.24) | 93.4% | | |
| `q4` + `NS_Q35_KV_ROT=1` | q4_0 rotated | 0.0055 (0.21) | 97.3% | | |
| `auto-q4` + `NS_Q35_KV_ROT=1` | half / q4_0 rotated | 0.0032 (0.017) | 96.5% | 45, 1,493 tok/s | 1,145 tok/s |

q8_0 is within the noise; q4_0 is ~3-9x it, the rotation (a 32-block Hadamard transform of keys, values and queries,
QuaRot's) taking a third off. q4_0 on the 5070 is the fast and long-context choice: five more layers there.

Also done: prompts with pictures are pipelined too (their positions and rows through pinned slots); a layer's
tensors are one allocation (one each had lost ~260 MB on the 5070 to 2 MiB rounding); a shared input is quantized
once for the products that read it; four drafts a step load at 80K.

### Nemotron 3.5 Super

**Nemotron 3.5 Super 120B-A12B UD-IQ1_M** (`llm/nemotronh`; Mamba-2 + latent MoE + attention) on one
12 GB card. The dense weights (5.8 GiB) and the experts wanted most stay in VRAM, a cache of experts in pinned host
memory, the rest is read from the file as decode asks for it; the 3060 on its x1 link is not used.

```sh
nextcuda models pull nemotron-3.5-super-iq1m          # 46 GB, or --from DIR to link files you have
nextcuda llm start nemotron-3.5-super-iq1m            # a process (pid, log in the socket directory), up in ~25 s
nextcuda llm chat "What is the capital of France?"    # one request through the control socket, streamed
nextcuda llm status                                   # the model, the card, the prompt cache
nextcuda llm logs | nextcuda llm stop
```

The server speaks OpenAI's chat API on `127.0.0.1:8085` (`/v1/chat/completions`, `/v1/models`; `/api/chat` for a page);
through the front door, `llm.chat` (or `/v1/chat/completions` on :8000) starts it when a request names the model:

```sh
curl -s http://127.0.0.1:8085/v1/chat/completions -H 'content-type: application/json' \
     -d '{"model": "nemotron-3.5-super-iq1m", "max_tokens": 200, "messages": [{"role": "user", "content": "Hello"}]}'
```

- **Prompt cache:** the server checkpoints each prompt (every attention layer's keys and values, every Mamba layer's
  state: ~175 MB + 8 KB a token) at the start of its first and last user turn and at its end, in host memory
  (`NS_PROMPT_CACHE_MIB`, the card's 2048) and then on disk (`NS_CACHE_DIR`, default `~/.cache/nextcuda/prompts`).
  A request that shares a system prompt or continues a chat reads only what is new: measured, a 1,314-token prompt
  read in 18.1 s the first time took 2.1 s the second.
- **The expert profile** (`expert-profile.txt` beside the model): which experts decode asked for. The next start fills
  VRAM and the pinned cache by it, so the first runs of a new topic are slower than later ones.
- **Memory:** `NS_NH_PINNED_GIB` is the biggest knob (the card sets 22; 25 is faster but leaves the 31 GB box little
  else, 18 slower). The prompt cache comes out of the same RAM.
- **Speed on the 5070** (`nh --bench`, `docs/benchy/nemotron-3.5-super-iq1m.md`): decode 8-10.5 tok/s with 25 GiB
  pinned, 6-8 with the card's 22, 4-6 with 18; prompts 170-190 tok/s (a prompt longer than 128 rows borrows 2.6 GiB of
  the VRAM's hot experts for passes of 2,048 rows, read back after it).
- **Engine options** (`nextcuda llm engines`): `--opt-pinned-gib`, `--opt-chunk` (prompt rows a pass, 2048),
  `--opt-reserve-mib` (VRAM left free, 700). `NS_NH_CPU_FRAC` gives part of each layer's pinned experts to the CPU
  (ggml-cpu, `libggml-base.so` / `libggml-cpu.so` beside the program) - measured slower here than the copies over
  gen4 x16, so it is off.
- **Open WebUI** (`http://<host>:3000`, the page's Chat link): `tooling/open-webui.sh` (re)makes its container (the
  `open-webui` volume kept) with the front door's `/v1` and token as its OpenAI connection, and
  `tooling/open-webui.service` starts it with the box. The chat model loads on the first message and unloads after
  `NS_IDLE_MINUTES`; set the connection's header `{"X-Keep-Up-S": "300"}` to keep it up between messages. Its
  read-aloud speaks through `/v1/audio/speech` (Admin > Settings > Audio: OpenAI, `http://host.containers.internal:8000/v1`,
  model `voxcpm2-int8`, a saved voice's name) - sentence by sentence, faster than it plays, beside the chat model.
  A follow-up on a long pasted document reuses its checkpoint (the prompt cache keeps the start of each reply).
- **`nh`** (`llm/nemotronh/src/bin/nh.rs`): the engine without the server - `nh MODEL --prompt TEXT --n N`, or
  `--bench 20,2000,8000 --new 128` for the numbers.

## Supported models

33 model cards in `model-cards/` (one YAML a model, every file a pinned link with its size and SHA-256);
`nextcuda models pull <id> [--from DIR]` downloads what is missing or hard-links a file already on the machine,
checks it and registers the model. **Served** - running on this box today; **download** - its card is here and it
runs once pulled; **too big** - more than these 12 GB cards hold: the page shows it "won't run here" (why, in GiB)
with its Download off (`system.models.pull` still fetches it, for bigger cards).

| kind | id | model | on disk | here |
|---|---|---|---|---|
| chat | `qwen3.8-27b-gsq-iq3s` | Qwen3.8-27B, ISTA-DASLab's GSQ-RCO IQ3_S with its MTP head and the BF16 vision projector: text and pictures, thinking switch, 80K context (192K with q4_0) | 12.2 GiB | served, both cards whole |
| chat | `qwen3.8-27b-gsq-iq3s-long` | the same files, fast / long: the cache q4_0 rotated, 160K of context (a speech model still fits beside it) | (the above's) | served |
| chat | `nemotron-3.5-super-iq1m` | NVIDIA Nemotron 3.5 Super 120B-A12B, unsloth's UD-IQ1_M (text) | 43.0 GiB | served (one card, experts streamed; off by default) |
| image | `qwen-image-2.1-q8` | Qwen-Image 2.1 Q8_0 with Qwen3-VL 8B: text to image, RGBA, edits from up to 8 pictures | 13.6 GiB | served |
| image | `qwen-image-2.1-rrq4` | Qwen-Image 2.1, [realrebelai's Q4](https://huggingface.co/realrebelai/Qwen-Image-2.1_GGUFs) | 5.5 GiB | served |
| image | `qwen-image-2512-q4km` | Qwen-Image 2512 Q4_K_M with Qwen2.5-VL 7B and the Lightning 4-step LoRA | 17.7 GiB | served |
| image | `qwen-image-2.1-q6k` | Qwen-Image 2.1 with a Q6_K transformer | 12.0 GiB | download |
| OCR | `unlimited-ocr-q8` | Unlimited-OCR Q8_0: pages to Markdown | 3.7 GiB | served |
| OCR | `unlimited-ocr-q4` | Unlimited-OCR Q4_K_M | 2.6 GiB | served |
| OCR | `paddleocr-vl-1.6` | PaddleOCR-VL 1.6: pages, tables, formulas, charts, seals | 1.7 GiB | served |
| OCR | `olmocr-2-7b-q8` | olmOCR-2 7B Q8_0: pages to Markdown | 8.8 GiB | served |
| speech | `breeze-tts-2` | Breeze TTS 2 Q8_0 with Qwen3-TTS's 12 Hz codec: English and Chinese, voices designed, cloned or directed, vocal events (non-commercial) | 4.0 GiB | served |
| speech | `qwen3-tts-custom` | Qwen3-TTS 12 Hz 1.7B CustomVoice: nine built-in voices styled by an instruction, ten languages | 4.2 GiB | served |
| speech | `voxcpm2` | VoxCPM2 2B: 30 languages at 48 kHz, voices designed or cloned | 4.6 GiB | served |
| speech | `voxcpm2-int8` | VoxCPM2, its language models in int8: 3.3 GiB on a card, beside the chat model | (voxcpm2's) | served (Open WebUI's voice) |
| speech | `breeze-tts-2-q4` | Breeze TTS 2 Q4_K, a smaller download | 2.7 GiB | download |
| speech | `qwen3-tts-base` | Qwen3-TTS Base: a voice cloned from a recording | 4.2 GiB | download |
| speech | `qwen3-tts-design` | Qwen3-TTS VoiceDesign: a voice made up from a description | 4.2 GiB | download |
| songs | `minimax-music3-int8` | MiniMax Music 3, its language model in int8: songs from lyrics and a description, up to 6 minutes, 44.1 kHz stereo | (music3's) | served, both cards |
| songs | `minimax-music3` | MiniMax Music 3 in half | 26.6 GiB | too big (23 GB a card) |
| video | `minimax-h3-q4km` | MiniMax H3 Q4_K_M: text, pictures and a voice to video with sound; the denoiser in 10.6 GiB | 10.6 GiB | served, both cards |
| video | `minimax-h3-q6k` | MiniMax H3 Q6_K: the denoiser in 15.4 GiB | 15.4 GiB | download |
| video | `minimax-h3-q8` | MiniMax H3 Q8_0, a reference form | 20.0 GiB | too big |
| video | `minimax-h3-ref2va` | MiniMax H3 Ref2VA (int8 ConvRot): reference pictures, clips and sounds to video | 19.5 GiB | too big |
| video | `minimax-h3` | MiniMax H3, int8 ConvRot denoiser: the fastest | 47.9 GiB | too big |
| 3D | `trellis2-512` | TRELLIS.2 4B Q8_0 at 512: picture to textured mesh, the light path (MIT) | 1.3 GiB | served |
| 3D | `hunyuan3d-2mini-turbo` | Hunyuan3D-2mini-turbo: a mesh from a picture in seconds (shape) | 3.9 GiB | served |
| 3D | `unirig` | UniRig: a skeleton and skin weights for any mesh (MIT) | 5.4 GiB | served |
| 3D | `rignet` | RigNet: a skeleton and skin weights for a character mesh (GPLv3) | 0.2 GiB | served |
| 3D | `animateanymesh` | AnimateAnyMesh: text-driven mesh animation, 16 frames as morph targets (Apache-2.0) | 0.9 GiB | served |
| 3D | `trellis2` | TRELLIS.2 4B Q8_0, the 1024 cascade | 8.8 GiB | download |
| 3D | `pixal3d` | Pixal3D Q8_0: pixel-aligned TRELLIS.2, the 1024 cascade (MIT) | 8.9 GiB | download |
| 3D | `hunyuan3d-2.1` | Hunyuan3D 2.1: a detailed mesh with PBR textures | 15.5 GiB | too big |

Each model's guide (how to ask it, what it takes, what it answers) is in `model-cards/guides/`.

## What it does

- **The front door on :8000:** `/rpc/<kind>.<method>` with a bearer token (`serve.json`, or `NS_API_TOKEN`); every
  answer is the envelope `{ok, result}` or `{ok: false, error: {code, message, retryable, ...}}`. `image.generate`
  and `image.edit` run as jobs (`wait: false`, `label`, `reentrant_id`), asked for with `image.status` /
  `image.jobs`, stopped with `image.cancel`; `image.upload`, `image.queue.*`, `system.busy`, `system.cards`,
  `system.models` and `system.version` (which lists its features) are there as they are on nextsycl.
- **Maintenance mode** (the page's Maintenance button, `system.maintenance {on, message, eta_s, unload}`): the box set
  aside for development while callers are told. What would start work - chat, generation, edits, speech,
  `system.load` / `enable`, the OpenAI POST routes - is refused with error code `maintenance` (503, retryable,
  `retry_after_s`; `/v1` answers HTTP 503 with `Retry-After`); status, listings and cancels answer as always. Queued
  image and audio jobs wait, the video queue pauses, running work finishes; the switcher answers Open WebUI's chats
  with the notice (`maintenance.json` beside the registry). Requests with `X-Maintenance-Bypass: <key>` (the key turning
  it on answers) go through.
- **A model server a model, on demand:** the first request for a registered model starts `nextcuda image serve` for
  it on the first card with room, swaps it when a request names another, and lets it go after the idle time.
- **Qwen-Image 2.1:** text to image (RGBA with `rgba`), img2img with a `strength`, edits and compositions from up to
  8 reference pictures with an optional mask, the effort ladder below. `NC_IMAGE_BACKEND=rust` runs it on the Rust
  engine instead (text to image so far): the pipeline is Rust, the kernels CUDA (cuBLAS and our own), the
  transformer kept on the card in its stored GGUF type, no ggml.
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
| `--opt-backend native\|rust\|python` | `NC_IMAGE_BACKEND` | stable-diffusion.cpp in process (default), the Rust engine on the shared CUDA kernels, or the phase 1 sd-cli worker |

## How it is built

```
kernels/ns            the C ABI (GPUs, memory, streams, copies) on CUDA - nextsycl's ns.h, the same functions
kernels/image/qwenimage21   the engine's own kernels (qi21.cu)
kernels/image/sd      a flat C ABI over stable-diffusion.cpp's ggml-cuda: one loaded model, progress, cancel
crates/core           the safe layer over libnextcuda-<kind>.so (host-memory fallback without it)
image/qwenimage21     the engines: Qwen-Image 2.1 and 2512 behind nextcuda-image's contract
kernels/llm/nemotronh, llm/nemotronh   Nemotron-H: Mamba-2, attention and Strata's i-quant products (CUDA), the
                      engine behind nextcuda-llm's contract (llm/contract), its expert cache and reader threads
glue/serve, glue/models, cli/nextcuda    the control plane, the registry and the program, as nextsycl's
```

The rules of nextsycl's `docs/architecture.md` hold - dependencies point down, one library a kind, tuning in silos -
and `cli/nextcuda/tests/architecture.rs` checks them on every `cargo test`, with CUDA as the one GPU runtime.

`./build.sh` builds the program in a debian:trixie toolchain image (its glibc is the GPU host's) and the web front
end; `kernels/build.sh` builds the CUDA libraries with `nvcc` for the 3060 (sm_86) and the 5070 (sm_120).

## Speech: Breeze TTS 2, Qwen3-TTS, VoxCPM2

nextsycl's engine (`audio/breezetts2`) and the 12 Hz codec (`crates/codec12hz`), unchanged, over the audio kernel
library: the music engine's decode kernels (`kernels/audio/minimaxmusic3/mm3.cu`: the small-batch product, the cached
attention; the codec's convolutions as an unfolding and one cuBLAS GEMM a tile, float32), the codec's
(`kernels/audio/qwen3tts/q3t.cu`) and Breeze's own (`kernels/audio/breezetts2/bz2.cu`) - CUDA ports with nextsycl's
names and arithmetic. No Python.

```sh
nextcuda models pull breeze-tts-2
nextcuda audio gen "(sigh) Welcome aboard." --model breeze-tts-2 --instructions "a calm adult male narrator" --out a.wav
nextcuda audio check <reference dump dir> --model breeze-tts-2        # reference/breezetts2/ref.py makes the dumps
```

- **Checked** (2026-10-17) against upstream's own code in float32 on the CPU (`reference/breezetts2/ref.py`, a designed
  voice, guidance 4): token ids identical; with the Q8_0 file the text encoder 6.6e-3, the prompt rows 9.5e-3, the
  backbone's logits 4.7e-3 and the depth decoder's 9.1e-3 (teacher forced: 1 of 60 first codes picks otherwise), the
  codec 1.4e-6 - nextsycl's numbers for the same file. Takes made here transcribe word for word (Whisper small).
- **Qwen3-TTS and VoxCPM2**: nextsycl's engines too (`audio/qwen3tts`, `audio/voxcpm2`; VoxCPM2's kernels
  `kernels/audio/voxcpm2/vcp.cu`). Checked against their own packages in float32 (`reference/qwen3tts/ref.py`,
  `reference/voxcpm2/ref.py`): Qwen3-TTS's prompt rows 1.9e-5, first logits 5.4e-4, frames forced: no first code
  differs (the other codes 4 of 585, near ties), the codec 1.2e-6; VoxCPM2's prompt tokens identical, both language
  models 3e-4, each patch from the reference's history and noise 5e-3 (cos 0.99999), the VAE 1.1e-6 - run free it
  takes another valid path after a few patches, as nextsycl documents. On the 5070, 5.6 s of Qwen3-TTS speech in 1.0 s,
  4.8 s of VoxCPM2's in 1.0 s; both transcribe word for word.
- **Speed** (RTX 5070): 5.5 GiB loaded in 4 s; 26 frames a second with guidance (2.1x real time; the B70: 22); through
  the front door, 12.7 s of designed speech in 6.5 s.
- **Saved voices** (`audio.voices.add`: a recording, its transcript, a description): spoken by name by any model that
  clones (`voice: "skippy"` on `/v1/audio/speech`). VoxCPM2 takes a saved voice's timbre alone - continuing its
  recording (`NS_VCP_CONTINUE=1`) led the recording's last words into each sentence (Whisper heard them ahead of 4 of
  5 lines; the timbre alone said all the plain ones word for word).
- **Text as it is said:** math and code symbols become words before any speech model sees them (`Q·K^T / √d_k`:
  "Q dot K to the T, divided by square root of d sub k"), markdown's marks go, dashes and brackets are pauses
  (`NS_SPEECH_NORMALIZE=0` or `normalize: false` for the text as sent) - one such line had a voice loop for 18 s.

## What runs where

| kind | engines | on the two cards |
|---|---|---|
| chat | `qwen35` (Qwen3.8: DeltaNet + attention, MTP drafts, vision), `nemotron_h_moe` | Qwen3.8 over both, pipelined; Nemotron on the 5070 |
| image, OCR | stable-diffusion.cpp's ggml-cuda (Qwen-Image), our Rust + CUDA engines (Qwen-Image 2.1 `rust`, Unlimited-OCR, PaddleOCR-VL, olmOCR-2) | one card each; olmOCR-2 the decoder on one, the vision tower on the other |
| speech | `breeze-tts-2`, `qwen3-tts`, `voxcpm2` | one card each; VoxCPM2 int8 fits beside the chat model |
| songs | `minimax-music3` (int8) | the language model on the 5070, the flow stage and decoder on the 3060 |
| video | `minimax-h3` (the native daemon, `--span`) and the video studio (scenes, plans, a film's clips joined) | the denoiser over both cards, nothing streamed: a clip in ~62 s from the SSD |
| 3D | TRELLIS.2, Hunyuan3D-2mini, UniRig, RigNet, AnimateAnyMesh | one card each |

Not here yet: `audio.transcribe` (Whisper - the platform's music and audio makers still listen locally), ACE-Step,
VibeVoice, HunyuanImage 2.1, the video upscalers; the bigger cards' forms above. The answers' `nextsycl` result key keeps
its name (the platform reads it).

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

- **Two cards are a pipeline, not a relay.** Splitting the layers only adds the cards' times; the speed came from the
  first card starting the next chunk while the second finishes this one - and from nothing on the host waiting.
- **A launch queue fills.** One host thread feeding two cards a thousand kernels each blocks in `cudaLaunchKernel`
  (nsys showed 0.17 ms a launch); graphs, one a stage, fixed it.
- **Measure the other runtime, kernel by kernel.** nsys on llama.cpp showed its per-card kernels were not much faster -
  the cards overlapped - and, for decode, that its matrix-vector products were.

- **A read-aloud that repeats is a queue that ran dry.** Open WebUI ends playback when its next sentence is late; the
  speech model was being swapped out for the chat model's title and tag requests. Fit them side by side.
- **Score a voice with ears you trust.** Whisper against the text, the port against its reference package on the
  same lines: the same error rate - the trouble was the mode (continuing a recording) and the symbols, not the port.
- **A cache needs a floor, not just a budget.** Checkpoints of forms the engine no longer makes filled the drive; the
  cache now drops them at start and keeps 8 GiB of its drive free.

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
