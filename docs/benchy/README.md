# Benchy: how a run is captured

nextsycl's method, for the kinds nextcuda serves. A result is a matrix of **clocked runs**, written beside what was
changed so a number is never quoted without the run that made it.

## The command

```sh
nextcuda image bench [--models ID,ID] [--sizes 768x768,1024x1024] [--steps 12] [--runs 3] [--gpu 0] \
                     [--prompt TEXT] [--out DIR] [--note "what is different about this run"]
```

- **Fixed inputs:** one prompt (override with `--prompt`), seeds 1..N, a size list and a step count. Nothing else
  changes between runs of one matrix, so two matrices can be compared.
- **Cold and warm:** each model is loaded once (its load time is a row). The first picture of a size is the **cold**
  one - the weights paged in, the graphs built - and is shown in brackets beside the **warm** median of the rest,
  as nextsycl's decode table shows the first pass after a load.
- **Stages, in seconds:** text encode; the first sampling step (it builds the graph and stages the weights) and the
  mean of the others; the sampler as a whole; the VAE decode; the picture's total.
- **Peaks:** `nvidia-smi` is sampled every 250 ms for each picture's window: the highest GPU memory used, less the
  idle reading before the load (the model's own), the card's `power.draw` over the picture as joules and as
  average watts, and the process's peak resident host memory (`VmHWM`).
- **Out:** `matrix.md` (printed too) and `matrix.jsonl` (one JSON object a picture, every field above) in `--out`,
  default `benchy-results/image-v1-<date>`.

## Keeping a result

A result worth keeping is copied here as `docs/benchy/<kind>-v1-<date>-<what changed>.md` with its `.jsonl` beside it,
and quoted with the change that moved it:

```
image-v1-2026-10-15-first.md            the in-process engine, offload on (the baseline)
image-v1-2026-10-16-silo-attention.md   the silo's attention kernel
```

A pull request that claims a speed-up links the matrix before and the matrix after, run on the same card, the same
driver, the same prompt, steps and sizes, with `--note` naming what changed.

## What a result must say

The card (name and memory), the driver, the weights placement (offload on or off), the date, the engine's backend
(`native` or `python`), and anything else running on the box. The bench writes the card and the note itself; the rest
goes in `--note`.
