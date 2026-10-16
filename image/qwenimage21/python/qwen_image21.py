"""A Qwen-Image 2.1 engine over stable-diffusion.cpp's sd-cli (CUDA), vendored from a project that already ran it;
nextcuda's phase 1 drives it from the Rust control plane (worker.py) and the in-process engine replaces it. Changes
from the original: the protocol import, and `init_image` + `strength` (plain img2img: sd-cli's -i / --strength) which the
nextsycl API's image.generate takes. The rest is the platform's, so its fixes can be diffed in.

Concept images: Qwen-Image-2.1 through stable-diffusion.cpp (leejet), the
GGML runtime — one `sd-cli` process per candidate, like trellis.cpp. The
7B single-stream DiT is a GGUF (Q8_0 7.2GB / Q6_K 5.6GB / Q4_K 3.9GB), the
text encoder is Qwen3-VL-8B as a llama.cpp GGUF (+ its mmproj vision
weights, needed whenever a reference image is attached), the VAE is the
2.1 RGBA one (bf16 safetensors). `--offload-to-cpu` keeps the weights in
RAM and stages them into VRAM per graph, so the 12GB cards run Q8 at 1024².

What the model can do, and how this engine exposes it (all optional):
  text-to-image        prompt only
  transparent (RGBA)   `transparent=True` wraps the prompt in the official
                       "This is an RGBA image with transparency. ... The image
                       has alpha channel and the background is transparent."
                       form; the PNG keeps its alpha
  editing              `references=[png, ...]` (up to 10) — the prompt
                       refers to them as image 1, image 2, ...: restyle,
                       relight, change one thing, combine several
                       subjects, extract a subject with transparency
  region editing       `mask` (white = edit here) + `region_mode`:
                         mask     the mask travels as one more reference
                                  image and the prompt names it
                         paint    the UI already painted the region white
                                  onto reference 1 ("white masked region")
                         outline  the UI drew a coloured outline onto
                                  reference 1 ("the red outlined region")
                       `keep_outside=True` composites the original pixels
                       back outside the mask afterwards (pixel-exact
                       inpainting; the model alone only approximates it)

Alpha caveat (2026-09-22): stable-diffusion.cpp decodes the 2.1 VAE's alpha
wrongly (issue leejet/stable-diffusion.cpp#2024 — most of the background
comes back opaque in latent-grid blotches; reproduced here with Q8_0 and
Q6_K on CUDA). Until that is fixed, `settings.qwen21_alpha` = auto (default)
keeps the native alpha only when it looks sane (the outer border band is
transparent) and otherwise cuts the subject out with rembg on the RGB
composited over white — the RGBA prompt still makes the model paint an
isolated subject on a clean background, which is what rembg needs.
"native" / "rembg" force either path.

  turbo                Viggle's Qwen-Image-2.1-viggle-turbo LoRA (DMD, rank 64):
                       4–6 steps, cfg 1, preview quality; ~8× faster than 40 steps

Models (spec["models"]): sdcpp_bin, qwen21_gguf, qwen21_vae, qwen21_llm,
qwen21_llm_vision, qwen21_turbo_lora.
"""

import math
import os
import re
import shutil
import subprocess
import tempfile
import time

from protocol import log

RGBA_PREFIX = "This is an RGBA image with transparency. "
RGBA_SUFFIX = " The image has alpha channel and the background is transparent."

MAX_REFERENCES = 10
# References go in at their own size (--disable-auto-resize-ref-image), and
# Qwen3-VL's vision encoder grows with the pixels: a 2048x2048 reference (an
# upscale fed back in) failed every attempt with "qwen3vl segment 2/30
# (llm.vision.blocks.0) failed during weight preparation" on the 12GB cards,
# while 1600x1600 still ran (2026-10-12). Qwen-Image edits at about a
# megapixel, and that is what it gets.
MAX_REF_PIXELS = 1024 * 1024


def rgba_prompt(prompt: str) -> str:
    """The official transparent-output prompt form (idempotent)."""
    p = prompt.strip().rstrip(".")
    if p.lower().startswith("this is an rgba image"):
        return prompt
    return f"{RGBA_PREFIX}{p}.{RGBA_SUFFIX}"


def region_hint(mode: str, n_refs: int, color: str = "red") -> str:
    """One sentence telling the model where the edit goes and that the
    marking itself must not be rendered — phrased like the official demo
    cases (annotations / separate mask / painted region)."""
    if mode == "mask":
        return (f" Image {n_refs} is a mask: its white area marks the only region of image 1 to change; "
                f"keep everything outside it exactly as it is, and do not render the mask itself.")
    if mode == "paint":
        return (" Apply the change only inside the region painted white on image 1; keep the rest of the "
                "image exactly as it is, and the white paint must not remain in the output.")
    if mode == "outline":
        return (f" Apply the change only inside the {color} outlined region of image 1; keep the rest of the "
                f"image exactly as it is, and the {color} outline must not be rendered in the output.")
    return ""


class QwenImage21Engine:
    def __init__(self):
        self.bin = ""
        self.models = {}
        self.settings = {}
        self.default_steps = 40
        self.default_cfg = 4.0

    # -- lifecycle ---------------------------------------------------------------

    def load(self, models: dict, settings: dict):
        self.models, self.settings = models, settings
        self.bin = models.get("sdcpp_bin") or ""
        if not self.bin or not os.path.exists(self.bin):
            raise FileNotFoundError(f"sd-cli not found at {self.bin!r} (config models.sdcpp_bin)")
        for key in ("qwen21_gguf", "qwen21_vae", "qwen21_llm"):
            p = models.get(key) or ""
            if not p or not os.path.isfile(p):
                raise FileNotFoundError(f"Qwen-Image-2.1: {key} missing ({p!r})")
        vis = models.get("qwen21_llm_vision") or ""
        if vis and not os.path.isfile(vis):
            log(f"qwen-image-2.1: mmproj {vis!r} missing — image editing will fail", "warn")
        log(f"qwen-image-2.1: {self.bin} gguf={os.path.basename(models['qwen21_gguf'])} "
            f"llm={os.path.basename(models['qwen21_llm'])} vision={'yes' if vis else 'no'}")

    def reload(self):
        pass

    def unload(self):
        pass

    def clear_vram(self):
        pass

    # -- generation ----------------------------------------------------------------

    def _args(self, prompt, negative, width, height, steps, cfg, seed, refs, out_png, turbo=False, init_image=None, strength=None):
        m = self.models
        if turbo:
            # sd.cpp applies LoRAs from a prompt tag; the file lives in its own dir
            lora = m.get("qwen21_turbo_lora") or ""
            prompt = f"{prompt}<lora:{os.path.splitext(os.path.basename(lora))[0]}:1>"
        args = [self.bin,
                "--diffusion-model", m["qwen21_gguf"],
                "--vae", m["qwen21_vae"],
                "--llm", m["qwen21_llm"],
                "-p", prompt,
                "-W", str(int(width)), "-H", str(int(height)),
                "--steps", str(int(steps)),
                "--cfg-scale", str(float(cfg)),
                "--sampling-method", "euler",
                "--seed", str(int(seed)),
                "--offload-to-cpu", "--diffusion-fa", "--vae-tiling",
                "--disable-image-metadata",
                "-o", out_png]
        if negative.strip():
            args += ["-n", negative.strip()]
        if init_image:
            args += ["-i", init_image]
            if strength is not None:
                args += ["--strength", str(float(strength))]
        if turbo:
            args += ["--lora-model-dir", os.path.dirname(m.get("qwen21_turbo_lora") or "")]
        if refs:
            vis = m.get("qwen21_llm_vision") or ""
            if vis:
                args += ["--llm_vision", vis]
            # the model handles differently sized references itself; sd-cli
            # would otherwise centre-crop each one to the output size
            args += ["--disable-auto-resize-ref-image"]
            for r in refs:
                args += ["-r", r]
        extra = (self.settings.get("sdcpp_extra_args") or "").split()
        args += extra
        return args

    def _run(self, args, progress, expect_steps):
        log("qwen-image-2.1: " + " ".join(os.path.basename(a) if i == 0 or a.startswith("/") else a
                                          for i, a in enumerate(args)))
        env = dict(os.environ)
        t0 = time.time()
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, env=env, bufsize=1, errors="replace")
        tail = []
        # sd.cpp prints "|====>   | 12/40 - 1.85s/it" while sampling; the
        # loading / encoding phases are named in its INFO lines
        step_re = re.compile(r"\|[=> ]*\|\s*(\d+)/(\d+)")
        phase_re = re.compile(r"(?i)(loading|load model|llm|text encod|vae|encod|decod|sampling|denois)")
        last_step = 0
        denoised = False  # the sampling bar has been seen (bars before it are VAE encodes of references, after it the tiled decode)
        for raw in proc.stdout:
            # progress bars are redrawn with \r
            for line in raw.replace("\r", "\n").split("\n"):
                line = line.rstrip()
                if not line:
                    continue
                m = step_re.search(line)
                if m:
                    step, total = int(m.group(1)), int(m.group(2))
                    if total == expect_steps:
                        phase = "denoise"
                        denoised = denoised or step >= total
                    else:
                        phase = "decode" if denoised else "encode"
                    if (step, total) != (last_step, None) and progress:
                        last_step = step
                        progress(step, total, phase)
                    continue
                if re.search(r"\|\s*\d+/\d+\s*-\s*[\d.]+MB/s", line):
                    continue  # tensor-loading bar, not worth a line of the error tail
                tail.append(line)
                if len(tail) > 60:
                    tail.pop(0)
                if progress and last_step == 0:
                    pm = phase_re.search(line)
                    if pm:
                        progress(0, expect_steps, pm.group(1).lower())
        rc = proc.wait()
        return rc, tail, round(time.time() - t0, 1)

    def generate(self, prompt, negative, width, height, steps=0, cfg=0.0, seed=0, progress=None,
                 references=None, transparent=False, mask=None, region_mode="mask", keep_outside=True,
                 outline_color="red", out_path=None, turbo=False, init_image=None, strength=None):
        from PIL import Image
        refs = [r for r in (references or []) if r and os.path.isfile(r)][:MAX_REFERENCES]
        lora = self.models.get("qwen21_turbo_lora") or ""
        if turbo and not os.path.isfile(lora):
            log("qwen-image-2.1: turbo LoRA not installed (config models.qwen21_turbo_lora) — running the full model", "warn")
            turbo = False
        if turbo:
            # Viggle's DMD student: 4 passes, no classifier-free guidance
            steps = min(max(int(steps or 0) or 4, 4), 8)
            cfg = 1.0
        else:
            steps = int(steps or 0) or self.default_steps
            cfg = float(cfg or 0) or self.default_cfg
        # the DiT wants multiples of 32; the 2.1 VAE is 16× so 32 keeps the
        # latent even
        width = max(256, int(round(int(width) / 32)) * 32)
        height = max(256, int(round(int(height) / 32)) * 32)

        full = prompt.strip()
        mask_img = None
        if refs and mask and os.path.isfile(mask):
            mask_img = Image.open(mask).convert("L")
            if region_mode == "mask":
                if len(refs) < MAX_REFERENCES:
                    refs = refs + [mask]
                    full += region_hint("mask", len(refs))
                else:
                    log("qwen-image-2.1: 10 references already — mask not sent as an image", "warn")
            else:
                full += region_hint(region_mode, len(refs), outline_color)
        if transparent:
            full = rgba_prompt(full)

        out_dir = tempfile.mkdtemp(prefix="qwen21-")
        out_png = os.path.join(out_dir, "out.png")
        try:
            # sd.cpp's own reference resizing is off (it would centre-crop
            # every reference to the output size), so the 32-multiple the
            # 2.1 conditioner insists on is our job: a light resample of each
            # reference (and of the mask sent as one). Compositing below still
            # uses the originals.
            sd_refs = [self._ref_for_sd(r, out_dir, i) for i, r in enumerate(refs)]
            init = self._ref_for_sd(init_image, out_dir, 90) if init_image and os.path.isfile(init_image) else None
            args = self._args(full, negative or "", width, height, steps, cfg, seed, sd_refs, out_png, turbo=turbo,
                              init_image=init, strength=strength)
            rc, tail, secs = self._run(args, progress, steps)
            if rc != 0 or not os.path.exists(out_png):
                errs = [l for l in tail if "[ERROR" in l or "error" in l.lower()]
                raise RuntimeError(f"sd-cli exited with {rc}: " + " | ".join((errs or tail)[-6:]))
            img = Image.open(out_png)
            img.load()
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

        info = {"engine": "qwen_image21", "steps": steps, "cfg": cfg, "secs": secs,
                "prompt_sent": full, "transparent": bool(transparent), "turbo": bool(turbo)}
        if refs:
            info["references"] = len(refs)
            info["edit"] = True
        if mask_img is not None:
            info["region_mode"] = region_mode
            if keep_outside:
                img = self._composite_outside(img, refs[0], mask_img)
                info["keep_outside"] = True
        if transparent:
            img, src = self._transparent_alpha(img)
            info["alpha_source"] = src
        return img, info

    @staticmethod
    def _ref_for_sd(path, out_dir, idx):
        """The reference as sd-cli may take it: at most MAX_REF_PIXELS, every
        side a multiple of 32 (the nearest one, never below 32). Untouched
        files are passed as is."""
        from PIL import Image
        try:
            im = Image.open(path)
            im.load()
        except Exception as e:
            log(f"qwen-image-2.1: reference {os.path.basename(path)} unreadable ({e})", "warn")
            return path
        w, h = im.size
        scale = min(1.0, (MAX_REF_PIXELS / float(w * h)) ** 0.5)
        nw, nh = max(32, int(round(w * scale / 32)) * 32), max(32, int(round(h * scale / 32)) * 32)
        if (nw, nh) == (w, h):
            return path
        fixed = os.path.join(out_dir, f"ref_{idx + 1:02d}.png")
        im.resize((nw, nh), Image.LANCZOS).save(fixed)
        why = "shrunk to about a megapixel for the vision encoder" if scale < 1.0 else "sd.cpp wants multiples of 32"
        log(f"qwen-image-2.1: reference {idx + 1} resampled {w}x{h} → {nw}x{nh} ({why})")
        return fixed

    def _transparent_alpha(self, img):
        """Native alpha when sane, else rembg (see the module docstring)."""
        import numpy as np
        from PIL import Image
        mode = (self.settings.get("qwen21_alpha") or "auto").lower()
        rgba = img.convert("RGBA")
        a = np.asarray(rgba.getchannel("A"))
        h, w = a.shape
        b = max(1, int(0.1 * min(h, w)))
        border = np.concatenate([a[:b].ravel(), a[-b:].ravel(), a[:, :b].ravel(), a[:, -b:].ravel()])
        border_opaque = float((border > 127).mean())
        transparent_share = float((a < 8).mean())
        sane = border_opaque < 0.15 and transparent_share > 0.05
        log(f"qwen-image-2.1: native alpha border-opaque={border_opaque:.2f} transparent={transparent_share:.2f} "
            f"→ {'sane' if sane else 'looks broken (sd.cpp #2024)'}; mode={mode}")
        if mode == "native" or (mode == "auto" and sane):
            return rgba, "native"
        try:
            from ..tools import _remove_background
        except Exception as e:  # pragma: no cover
            log(f"qwen-image-2.1: rembg unavailable ({e}); keeping the native alpha", "warn")
            return rgba, "native"
        # the model painted the subject on a clean background; sd.cpp's
        # bogus transparent pixels carry junk colour, so flatten over white first
        flat = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
        cut = _remove_background(flat, False)
        return cut, "rembg"

    @staticmethod
    def _composite_outside(img, ref_path, mask_img):
        """Pixel-exact inpainting: the model's output inside the (feathered)
        mask, the original reference outside it. Sizes are matched by
        resampling the original + mask to the output."""
        from PIL import Image, ImageFilter
        ref = Image.open(ref_path)
        ref.load()
        mode = "RGBA" if "A" in img.getbands() or "A" in ref.getbands() else "RGB"
        out = img.convert(mode)
        base = ref.convert(mode).resize(out.size, Image.LANCZOS)
        m = mask_img.resize(out.size, Image.LANCZOS).filter(ImageFilter.GaussianBlur(max(1, out.width // 256)))
        return Image.composite(out, base, m)
