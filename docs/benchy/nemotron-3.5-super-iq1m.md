# Nemotron 3.5 Super 120B-A12B UD-IQ1_M on one RTX 5070 (12 GB), 2026-10-16

`nh MODEL.gguf --bench 20,2000,8000 --new 128` (llm/nemotronh): a prompt of the stated token count (paragraphs of prose, repeated),
read, then 128 greedy tokens. Host: 8 cores, 31 GB RAM, virtio NVMe, PCIe gen4 x16; the 3060 is not used. 43 GiB model:
5.8 GiB dense weights in VRAM, the 34 GiB of experts in three tiers - the hot ones in VRAM (about 4 GiB, chosen by a profile),
a pinned-memory cache (LRU), and the file (O_DIRECT reads, 24 threads).

Method: the expert profile (`expert-profile.txt`, which experts earlier runs wanted) is deleted, then the bench runs three times;
each run preloads the cache by the profile the last one saved. The third run is below (the first two: see "Warm-up"). The peak is the
card's own (idle 19 MiB). Energy and watts are the card's `power.draw` over prompt + decode. Cache hit % is of the experts' matrices
a pass wanted (VRAM-hot and pinned hits count, the staged copies not).

| pinned cache | prompt tokens | prompt read | prompt tok/s | decode tok/s | cache hit | peak GPU MiB | avg W | energy J |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 GiB | 20 | 1.2 s | 17.0 | 8.89 | 94.4 % | 10,786 | 68 | 1,041 |
| 25 GiB | 2,000 | 19.3 s | 103.4 | 7.79 | 88.7 % | 10,812 | 71 | 2,509 |
| 25 GiB | 8,000 | 65.8 s | 121.5 | 9.38 | 89.9 % | 10,850 | 87 | 6,905 |
| 22 GiB | 20 | 1.4 s | 14.3 | 7.54 | 91.2 % | 10,780 | 62 | 1,119 |
| 22 GiB | 2,000 | 22.6 s | 88.5 | 6.12 | 82.8 % | 10,806 | 64 | 2,778 |
| 22 GiB | 8,000 | 78.3 s | 102.2 | 7.42 | 82.7 % | 10,844 | 79 | 7,497 |

Warm-up (25 GiB, prompt 2,000 / decode): run 1 97 / 7.28, run 2 101 / 7.89, run 3 103 / 7.79; (22 GiB) 84.5 / 5.69, 88.4 / 6.06,
88.5 / 6.12. The first run has no profile (the cache preloaded in file order). Decode of the 20-token prompt, 25 GiB: 7.7, 8.5, 8.9.
A profile that matches the topic gives more: a repeated chat prompt reached 13-15 tok/s decode.

Pinned memory is the biggest knob (25 GiB: +20 % decode, +15 % prompt): 25 GiB of the host's 31 leaves little for anything else on the
box (the platform's workers), so the service should run 22 or less where they share it. Load: 20-25 s, then 13-15 s preloading.

## What the speed came from (one picture of the same box, decode tok/s, a profiled topic)

| step | decode | 2,000-token prompt |
|---|---:|---:|
| experts of 5 layers in VRAM, 16 pinned, 19 read per pass | 1.9 | 25 s for 30 tokens |
| pinned LRU cache, one pool per matrix size | 4.5-5.3 | |
| cache preloaded by the saved profile | 7.3 | |
| VRAM holds the hot experts of any layer | 10.2 | |
| a prompt pass of 512 rows: experts grouped, dequantized to half, cuBLAS; dense matrices too; tiled Mamba-2 scan; a prompt's misses through a scratch (no eviction) | | 149 s -> 19 s |
| decode: experts DMA-copied to VRAM before the products; routing guessed a layer ahead (80 % right) to start reads and copies | 9-10 (14.7 at 25 GiB, matched topic) | |

Time per decode token at 10 tok/s (~100 ms): cache misses read from disk and waited for ~38 ms, copying the pinned experts to VRAM
~25 ms, Mamba ~10 ms, the rest ~25 ms. Prompt, 2,000 tokens at 19 s: experts' products 9 s, reads 7 s (27 GB at ~4-6 GB/s), the rest ~3 s.

## Check against llama.cpp

llama.cpp (mainline, CPU, the same file, greedy): "The user is asking a simple factual question: What is the capital of France? The
answer is Paris. ..." (prompt 0.85 tok/s, decode 1.46 tok/s on this box). nextcuda: the same first eight tokens, then "...: capital of
France. Answer: Paris. ..." - a different but equally right continuation after a near tie on a 1-bit quant (activations through
Q8_1, as llama.cpp's CUDA path does; its CPU path differs in rounding). Logit-level comparison (cosine) is not done.
