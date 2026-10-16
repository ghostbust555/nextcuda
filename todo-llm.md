# todo: llm - Nemotron 3.5 Super UD-IQ1_M (nextsycl's `nemotron-3.5-super-iq1m`) on the 5070 + 3060

Study of nextsycl's `llm/nemotronh` (engine.rs 1,982 lines, ffi.rs, lib.rs) and `kernels/llm/nemotronh` (nh.cpp 598,
silo/{attn,dense,nhs}.cpp ~2,000 lines), 2026-10-16. Nothing is ported yet.

## The model

NVIDIA-Nemotron-3.5-Super-VL-120B-A12B, unsloth UD-IQ1_M, 2 GGUF shards, 43 GiB (46.1 GB). 88 layers: 40 Mamba-2
mixers (conv 4, 128 heads of 64, 8 groups, state 128), 40 latent-MoE layers (float32 sigmoid router + selection bias,
22 of 512 experts, experts in a 1,024-wide latent, ReLU-squared, a shared expert on the full width), 8 attention layers
without positions (32/2 heads of 128), plus an MTP draft block. Quant types: IQ1_S / IQ2_* / IQ3_XXS ups, Q2_0 downs,
Q6_K Mamba projections, K-quant shared experts. Text only (vision tower not ported in nextsycl either).

## What this box has (the constraint)

| | |
|---|---|
| VRAM | 12 GB 5070 (gen4 x16) + 12 GB 3060 (gen4 **x1**, about 1.5 GB/s) = 24 GB, minus KV / Mamba state / scratch |
| host RAM | 31 GB, 8 cores; pinned memory limit to be checked (`ulimit -l` is 8 MB; needs a raised limit or cudaHostAlloc via the driver) |
| disk | 76 GB free of 1.3 TB (94 % full): the 46 GB file fits once |
| model | not on the box; 46 GB to fetch (nextsycl's card has url + sha256 per shard) |

So about 43 - 18 = 25 GiB of experts cannot live on the cards. nextsycl's one-card mode (`NS_NH_HOST`, B70 32 GB
with the rest in pinned RAM) is the nearest case; here it is three tiers: VRAM, pinned RAM (about 18-20 GB), and the
page cache / NVMe for the remainder (nextsycl does this for DeepSeek-R1: "experts in VRAM, pinned RAM and the NVMe").

## Tricks to take from nextsycl, in order of value

1. **Expert store with three tiers and an expert profile.** At load VRAM takes the experts earlier runs asked for most
   (profile saved beside the prompt cache: first answer of a new topic ~10 % fewer swaps); a missed expert is swapped in
   at decode on two copy streams while resident experts compute; the next layer's likely experts are prefetched.
   (`reference/expert-cache` has the trace simulations: pinning hot experts lost - the hot set follows the topic;
   admission/prefetch policies.)
2. **Bound the prize first** (`NS_FREE_MISSES=1`, "a measurement, wrong output"): with every miss free decode was only
   13 % faster in nextsycl, so the copies matter less than the kernels. Do this before building a clever cache.
3. **Never stream onto the x1 card.** The 3060 holds whole layers only (its experts resident); the 5070, on x16, takes
   the layers that stream. Worst case 22 experts x 40 layers is ~1.8 GB a token: 70 ms at 25 GB/s (x16), 1.2 s at x1.
4. **One launch per projection for all chosen experts** at decode (routing on the GPU), downs repacked into aligned rows.
5. **Prompts**: chunks of 2,048 rows, dense matrices expanded to half for the GEMM (cuBLAS here, XMX there), experts
   grouped by expert after a GPU routing pass; layers split over the cards by measured speed (`NS_NH_SPEED`), the cards
   pipelined over chunks.
6. **MTP drafts** (2 a step, Mamba states snapshotted for the verify's rollback): 32 -> 48 tok/s in nextsycl. Last.
7. **Mamba chunked scan** for prompt rows (64-row chunks).
8. Method: `NS_NH_PROFILE=1` times every kernel; check against llama.cpp CPU on the same files (`llm check`,
   cosine 0.99999; `spec-check` for verify rows); microbenchmarks rotate many experts (one reused sits in L2).

## What to expect

nextsycl, UD-IQ1_M: both cards 702 / 41.6 tok/s at 2K (MTP), 31.8 without drafts; one B70 alone with host experts
347 / 21.4. On 24 GB with ~25 GiB off-card and only x1 for the second card, a first honest target is 5-10 tok/s decode
without drafts and a few hundred tok/s prompt; the miss rate decides it, which is why step 2 comes first.

## Decisions (2026-10-16)

- **5070 only.** The 3060 is not used: one card, 12 GB, everything else streamed (pinned RAM, then page cache/NVMe).
  Goal first: make it work at all, then fast.
- **Borrow CUDA kernels, do not port SYCL.** Strata_B70 (`brainiac:~/Dev/Strata_B70/src/kernels/cuda/`) has them for
  NVIDIA already: `iq_kernels.cu` (1,583 lines: IQ1_S/IQ2_XXS/XS/S/IQ3_XXS/Q2_0 dequant + mmvq-style dots, grouped
  experts), `native_mmvq.cu`, `native_moe.cu`, `native_router.cu`, `native_flash_attn.cu`, `quantize_act.cu`. What
  Strata lacks is Mamba-2: take `ssm-conv.cu` / `ssm-scan.cu` (516 lines, MIT) from llama.cpp's ggml-cuda
  (`~/llama.cpp` on brainiac-nvidia, which also knows `NEMOTRON_H_MOE`: the reference for step 4 and a fallback).
  Check each borrowed file's licence before it goes into the repo.

## Plan (each step checked before the next)

- [ ] 0. Fetch the two shards (46 GB) to `~/nextcuda/models/nemotron35-super-iq1m/`, verify sha256. Raise memlock for
      the service (`LimitMEMLOCK=infinity` in the unit). Needs the user's go-ahead (disk to 30 GB free).
- [ ] 1. GGUF reader: multi-shard, the IQ1_S / IQ2_XXS/XS/S / IQ3_XXS / Q2_0 / Q6_K / Q2_K..Q5_K tensor types (the
      grid tables) in `nextcuda_gguf` + dequant on the card in `kernels/llm/nemotronh/nh.cu` (exact against host).
- [ ] 2. Kernels in CUDA (from nh.cpp / silo): Mamba-2 conv + selective scan (decode and chunked), group-norm gate,
      latent MoE (router, top-22, ReLU^2), attention without RoPE, dense mmvq for each quant (decode), expanded GEMM
      (prompt). Selftests against host arithmetic, as the diffusion kernels have.
- [ ] 3. `llm/nemotronh` engine on the `LlmEngine` contract (`llm/contract`): load by tiers, one card first (5070,
      all experts through the store), then the layer split with the 3060.
- [ ] 4. Check against llama.cpp (build it with CUDA on the host, CPU run for reference): same next token, logits cosine.
- [ ] 5. Expert store: profile, miss swap on two streams, prefetch; measure `FREE_MISSES` first.
- [ ] 6. Bench with `nextcuda llm bench` (benchy methodology as the image kind: cold/warm, peak MiB, PCIe, energy).
- [ ] 7. MTP drafts; model card `model-cards/llm/nemotron-3.5-super-iq1m.yml`; serve behind the same RPC.

## Status (2026-10-16): it works

`nh MODEL.gguf --prompt TEXT --n N` (llm/nemotronh, kernels/llm/nemotronh) on the 5070 alone: UD-IQ1_M loads in 35 s
(5.8 GiB dense in VRAM; experts: 5 layers in VRAM, 16 pinned, 19 on disk), answers coherently ("The user is asking a
simple factual question" to "What is the capital of France?", greedy). Both shards sha256-match nextsycl's card.
First clock: 30-token prompt 25.5 s (1.2 tok/s: 8-row passes, 25 s of the 29 s in MoE are reading disk-resident
experts), decode 1.87 tok/s once warm (page cache).
Kernels: Strata's i-quant mmvq (+ IQ1_S and Q6_K added from llama.cpp's vecdotq), plain ones of our own (Mamba-2
conv/scan, gated norm, attention to 12K keys, RMS norm, combine). Not yet checked against llama.cpp's logits.

Next: check against llama.cpp (cosine of logits); the expert cache (hot experts by profile; pinned tier chosen per
expert, not per layer); bigger prompt chunks (group by expert, read each once); `FREE_MISSES` bound; GPU routing;
Mamba chunked scan; MTP.

## Status (2026-10-16, later): 10 tok/s

On the 5070 alone, UD-IQ1_M, greedy, the right answer ("Paris") and fluent 160-token stories:

| step | decode | 30-token prompt |
|---|---|---|
| first working (experts of 5 layers in VRAM, 16 pinned, 19 read from disk per pass) | 1.9 tok/s | 25 s |
| expert cache in pinned memory (LRU, one pool per matrix size, 22 GiB, 16 reader threads) | 4.5-5.3 | 13 s (cold) |
| experts' products in one launch (pointer arrays) | 5.3 | |
| cache preloaded at start (profile saved by the last run: most wanted first) | 7.3 | 3.6 s |
| VRAM = the hot experts of any layer by the profile (4.2 GiB), not the first layers' whole | 10.2 (same topic) | 3.0 s |

A different topic than the profile's: about 5 tok/s until the hot set follows it (profiles merge: this run's counts plus the
old ones halved). What is left per token at 10 tok/s (~100 ms): expert up/down 47 ms (the matrices read over PCIe from
pinned memory, 15-17 GB/s), cache misses read from disk 30 ms, Mamba 10 ms.
Load: 20 s + 15 s preload. Next: promote on a miss into VRAM (dynamic hot set), prefetch the next layers' likely experts
(overlap the disk reads), GPU routing, the Mamba chunked scan for prompts, MTP drafts, and the check against llama.cpp.

## Status (2026-10-16, end of day): numbers in docs/benchy/nemotron-3.5-super-iq1m.md

5070 alone: 2,000-token prompt 19 s (103 tok/s; 121 tok/s at 8,000), decode 8-9 tok/s (10-15 on a matched topic) with 25 GiB pinned;
22 GiB: 88 / 6-7.5. Done since the last status: grouped prompt MoE on tensor cores, dense GEMMs for prompts, tiled Mamba scan, O_DIRECT
reader threads, scratch for a prompt's misses, DMA staging for decode, routing guessed a layer ahead. Left: logit cosine against llama.cpp,
reading further ahead (two MoE layers), dynamic promotion into VRAM, wiring the engine to the `Engine` contract and `nextcuda llm serve`,
a model card, MTP (probably not worth it: a verify pass reads the union of its rows' experts).

## Status (2026-10-16, night): served on :8000, prompt cache, two experiments that lost

- **Served:** `llm/nemotronh` is behind the `Engine` contract (`engine.rs`: sessions, decode step, nextsycl's `NSNMH`
  checkpoints); `nextcuda llm start | serve | stop | logs` run it as a native process; model card
  `model-cards/llm/nemotron-3.5-super-iq1m.yml` (18 GiB pinned, 2 GiB prompt cache). The front door starts it on
  `llm.chat` / `/v1/chat/completions`: checked end to end on brainiac-nvidia.
- **Prompt cache:** checkpoints at each template's user-turn start (nextsycl's `USER_TURNS`; nextcuda only knew GLM's
  `<|user|>`, so ChatML prompts never matched). Measured: a 1,314-token prompt 18.1 s, then 2.1 s with the same system
  prompt; 454 tokens 11.5 s, then 2.8 s.
- **Reading further ahead lost:** guesses for two or three MoE layers and wider (top 32-48) read wrong experts that push
  right ones out of the pools: 2,000-token decode 7.6 -> 6.7 / 6.2 / 5.9 tok/s. Default stays one layer, top 22.
- **The CPU's share lost here:** ggml-cpu (Strata's / nextsycl deepseek2's tier) multiplying part of each layer's pinned
  experts in place: 8.4 tok/s with none, 7.2 / 7.0 / 6.8 at 30 / 50 / 70 % (the VM's 8 vCPUs do ~10 GB/s on a layer's
  few experts; the x16 link copies at ~24). Kept, off (`NS_NH_CPU_FRAC`), for boxes with more cores or a slower link.
- **Where the time is (profiled, 25 GiB):** ~115 ms a token: copies of pinned experts to VRAM ~38 ms (about 1 GB a token,
  the link's limit), misses read and waited for ~42 ms (24 MB a token: latency, not bandwidth), Mamba 10, the rest ~25.
  Next: a larger VRAM hot set (the prompt arena borrowed from it, Strata's trick, frees ~1 GB at decode), bigger prompt
  chunks in that room, MTP.
