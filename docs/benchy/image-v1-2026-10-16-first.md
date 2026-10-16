# nextcuda image bench, 2026-10-16

GPU 0 (NVIDIA GeForce RTX 5070); prompt: "a lighthouse on a rocky coast at dusk, long exposure, dramatic clouds, photorealistic"; 8 steps; seeds 1..3; 3 pictures a size, the first (cold) in brackets beside the warm median.

first capture with the bench: offload on, platform queue paused

Stages in seconds; peaks of GPU memory are the model's own (the idle reading before the load taken off); energy and watts are the card's `power.draw` over the picture.

## qwen-image-2.1-q8 (Qwen-Image 2.1 Q8_0 (text to image, RGBA, edits; stable-diffusion.cpp on CUDA))

load 0.4 s; GPU memory after the load 188 MiB (idle 19); engine: Qwen-Image 2.1 on NVIDIA GeForce RTX 5070 through stable-diffusion.cpp (loaded 1 s ago)

| size | steps | text encode | step 1 | step (mean of 2..n) | sampler | VAE | total | peak GPU MiB | host RSS MiB | energy J | avg W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 768x768 | 8 | 0.50 (2.51) | 2.7 (0.0) | 0.95 (1.25) | 11.1 (17.5) | 1.6 (2.5) | 13.2 (22.6) | 8197 (8085) | 13560 | 2848 (2890) | 227 (128) |
| 1024x1024 | 8 | 0.46 (0.32) | 3.9 (3.6) | 1.42 (1.42) | 22.6 (22.4) | 3.5 (3.5) | 26.6 (26.2) | 8591 (8591) | 13578 | 5790 (5683) | 219 (216) |

## qwen-image-2.1-rrq4 (Qwen-Image 2.1 Q4 (realrebelai GGUF, 6.0 GB; stable-diffusion.cpp on CUDA))

load 0.2 s; GPU memory after the load 206 MiB (idle 206); engine: Qwen-Image 2.1 on NVIDIA GeForce RTX 5070 through stable-diffusion.cpp (loaded 1 s ago)

| size | steps | text encode | step 1 | step (mean of 2..n) | sampler | VAE | total | peak GPU MiB | host RSS MiB | energy J | avg W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 768x768 | 8 | 0.54 (0.77) | 2.4 (0.0) | 1.03 (1.58) | 11.6 (23.8) | 1.6 (2.2) | 13.8 (26.9) | 6358 (6246) | 13578 | 3095 (3439) | 227 (127) |
| 1024x1024 | 8 | 0.33 (0.32) | 3.7 (3.7) | 1.52 (1.52) | 24.1 (24.1) | 3.5 (3.5) | 28.0 (28.0) | 6752 (6752) | 13578 | 6200 (6166) | 221 (220) |

## qwen-image-2512-q4km (Qwen-Image 2512 Q4_K_M (20B MMDiT, Qwen2.5-VL; Lightning 4-step LoRA below 20 steps))

load 0.2 s; GPU memory after the load 206 MiB (idle 206); engine: Qwen-Image 2512 on NVIDIA GeForce RTX 5070 through stable-diffusion.cpp (loaded 1 s ago, distilled LoRA below the ladder)

| size | steps | text encode | step 1 | step (mean of 2..n) | sampler | VAE | total | peak GPU MiB | host RSS MiB | energy J | avg W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 768x768 | 8 | 1.19 (8.23) | 3.7 (0.4) | 0.42 (0.56) | 13.5 (59.8) | 2.2 (2.5) | 16.9 (70.9) | 10994 (10964) | 18772 | 2706 (4310) | 164 (61) |
| 1024x1024 | 8 | 0.29 (0.29) | 3.9 (3.9) | 0.44 (0.43) | 23.6 (23.2) | 4.4 (4.4) | 28.3 (28.0) | 11042 (10994) | 18795 | 5118 (4977) | 181 (179) |
