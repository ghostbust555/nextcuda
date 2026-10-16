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

## Later the same night: overlap, fewer guessed copies, prompts that borrow the hot set

Same method, the second of two runs kept, 128 new tokens. Prompts of more than 128 rows take passes of 2,048 rows in VRAM
borrowed from the hot set's tail (its least wanted experts, read back after the prompt: Strata's trick), so decode keeps a
bigger hot set (4.7 GiB) and prompts sweep the experts a quarter as often. Decode multiplies the experts it has while the
misses are read (the missed ones in a second batch), and copies only the top 8 of a guessed next layer up.

| pinned cache | prompt tokens | prompt read | prompt tok/s | decode tok/s | cache hit | peak GPU MiB | avg W | energy J |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 GiB | 20 | 1.2 s | 16.1 | 9.85 | 95.8 % | 10,786 | 71 | 973 |
| 25 GiB | 2,000 | 10.7 s | 186.1 | 8.01 | 90.6 % | 10,812 | 73 | 1,917 |
| 25 GiB | 8,000 | 42.6 s | 187.8 | 10.52 | 89.5 % | 10,850 | 92 | 5,018 |
| 22 GiB | 20 | 1.5 s | 13.5 | 8.04 | 92.9 % | 10,780 | 64 | 1,092 |
| 22 GiB | 2,000 | 11.8 s | 168.8 | 6.12 | 86.2 % | 10,806 | 65 | 2,129 |
| 22 GiB | 8,000 | 45.5 s | 175.8 | 7.78 | 84.2 % | 10,844 | 85 | 5,266 |
| 18 GiB | 20 | 1.9 s | 10.4 | 5.75 | 87.8 % | 10,772 | 54 | 1,299 |
| 18 GiB | 2,000 | 13.6 s | 147.4 | 4.17 | 78.9 % | 10,798 | 55 | 2,413 |
| 18 GiB | 8,000 | 52.3 s | 153.1 | 5.39 | 76.3 % | 10,836 | 73 | 5,544 |

Against the first table (25 GiB): prompts 103 -> 186 tok/s at 2,000 and 121 -> 188 at 8,000; decode 8.9 -> 9.9, 7.8 -> 8.0, 9.4 -> 10.5.
Passes of 1,024 rows borrowed instead: prompts 143 / 161, decode about the same. The model card serves with 22 GiB.

What was tried and lost (2,000-token decode, 25 GiB): guesses two or three layers ahead or wider (7.6 -> 6.7 / 6.2 / 5.9);
the CPU multiplying part of the pinned experts with ggml-cpu (8.4 -> 7.2 / 7.0 / 6.8 at 30 / 50 / 70 %: 8 vCPUs do ~10 GB/s on a
layer's few experts, the link ~24); an LRU VRAM tier (48K copies in, 7 % more VRAM hits, twice the misses: 9.2 -> 7.0).
