//! The Qwen-Image engines on the in-process library: Qwen-Image 2.1 and Qwen-Image 2512 share this file, because
//! stable-diffusion.cpp runs both from a GGUF transformer, a Qwen-VL text encoder (a GGUF too) and a VAE. What
//! differs between them is a `Model`: its architecture name, the effort ladder (steps and guidance), the canvas
//! rounding, and whether it edits (2.1 conditions on reference pictures; 2512 is text and img2img only).
//!
//! The weights stay in host memory and are staged to the card per graph (`offload`), which is how an 8 GB
//! transformer and a 5 GB text encoder take turns on a 12 GB card.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextcuda_image::{Defaults, EngineOption, Error, Gpu, ImageEngine, ImageRequest, LoadOptions, ModelFiles, Picture, Result, Sampler, Schedule, Step};

use crate::sd::{Gen, Raw, Sd};

/// One architecture's differences
#[derive(Clone, Copy)]
pub struct Model {
    pub arch: &'static str,
    pub label: &'static str,
    /// the steps and guidance of a full run (no LoRA)
    pub steps: u32,
    pub cfg: f32,
    /// a distilled LoRA's: steps and guidance (the card's `turbo-lora` role), used under `ladder` steps
    pub lora_steps: u32,
    pub lora_cfg: f32,
    /// asking for at least this many steps bypasses the LoRA for a full run (the platform's effort ladder)
    pub ladder: u32,
    /// canvas sides round to this
    pub round: usize,
    pub edits: bool,
}

pub const QWEN_IMAGE_21: Model = Model {
    arch: "qwen-image-2.1", label: "Qwen-Image 2.1", steps: 40, cfg: 4.0, lora_steps: 4, lora_cfg: 1.0, ladder: 20, round: 32, edits: true,
};
pub const QWEN_IMAGE_2512: Model = Model {
    arch: "qwen-image-2512", label: "Qwen-Image 2512", steps: 30, cfg: 4.0, lora_steps: 4, lora_cfg: 1.0, ladder: 20, round: 16, edits: false,
};

pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "offload", env: "NC_SD_OFFLOAD", value: "0|1", help: "weights in host memory, staged to the card per graph (default 1)", at: nextcuda_image::At::Load },
    EngineOption { name: "max-vram", env: "NC_SD_MAX_VRAM", value: "GiB", help: "the card memory the weights and buffers may take (default: what is free)", at: nextcuda_image::At::Load },
    EngineOption { name: "sd-args", env: "NC_SD_ARGS", value: "k=v;...", help: "the bridge's own: threads=N, flash_attn=0|1, vae_tiling=0|1", at: nextcuda_image::At::Load },
];

pub struct Native {
    m: Model,
    sd: Mutex<Sd>,
    lora: Option<String>,
    load_s: f64,
    gpu: String,
    loaded: Instant,
}

/// The files, loaded on `gpu`: a LoRA found as the `turbo-lora` role (or a registry LoRA merged at load) is the
/// distilled path below the ladder
pub fn load(m: Model, files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Native> {
    let path = |role: &str| -> Result<&Path> { files.get(role).map(|p| p.as_path()).ok_or_else(|| Error(format!("{}: no {role} file", m.arch))) };
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    let t0 = Instant::now();
    let lora = files.get("turbo-lora").cloned().or_else(|| o.merge_loras.first().map(|l| l.path.clone())).map(|p| p.to_string_lossy().into_owned());
    let offload = o.setting("NC_SD_OFFLOAD").is_none_or(|v| v != "0");
    let max_vram = o.setting("NC_SD_MAX_VRAM").and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.0);
    let args = o.setting("NC_SD_ARGS").unwrap_or_default();
    let backend = format!("cuda{}", gpu.index);
    log(format!("{}: loading on {} ({backend}{}{})", m.arch, gpu.name, if offload { ", weights offloaded to host memory" } else { "" },
                nextcuda_core::pcie_link(gpu.index).map_or(String::new(), |l| format!(", PCIe {l}"))));
    let to_s = |p: &Path| p.to_string_lossy().into_owned();
    let transformer = crate::gguffix::qwen_image_transformer(path("transformer")?, log)?;
    let sd = Sd::load(&to_s(&transformer), &to_s(path("text-encoder")?), files.get("llm-vision").map(|p| to_s(p)).as_deref(), &to_s(path("vae")?),
                      &backend, max_vram, offload, &args)?;
    Ok(Native { m, sd: Mutex::new(sd), lora, load_s: t0.elapsed().as_secs_f64(), gpu: gpu.name.clone(), loaded: Instant::now() })
}

fn raw(p: &Picture) -> Raw<'_> {
    Raw { data: &p.data, w: p.width as usize, h: p.height as usize, c: p.channels as usize }
}

/// A reference as sd.cpp may take it: at most a megapixel (the vision encoder grows with the pixels), each side a
/// multiple of 32 - what the platform's engine did, for the same reason
fn fit_ref(p: &Picture) -> Picture {
    const MAX_PIXELS: f64 = 1024.0 * 1024.0;
    let (w, h) = (p.width as f64, p.height as f64);
    let scale = (MAX_PIXELS / (w * h)).sqrt().min(1.0);
    let r = |v: f64| (((v * scale / 32.0).round() as u32).max(1)) * 32;
    let (nw, nh) = (r(w), r(h));
    if (nw, nh) == (p.width, p.height) {
        p.clone()
    } else {
        p.resize_lanczos(nw, nh)
    }
}

const RGBA_PREFIX: &str = "This is an RGBA image with transparency. ";
const RGBA_SUFFIX: &str = " The image has alpha channel and the background is transparent.";

fn rgba_prompt(prompt: &str) -> String {
    let p = prompt.trim().trim_end_matches('.');
    if p.to_lowercase().starts_with("this is an rgba image") {
        prompt.to_string()
    } else {
        format!("{RGBA_PREFIX}{p}.{RGBA_SUFFIX}")
    }
}

/// One sentence telling the model where an edit goes (the platform's "mask" form: the mask rides as the last reference)
fn mask_hint(n_refs: usize) -> String {
    format!(" Image {n_refs} is a mask: its white area marks the only region of image 1 to change; keep everything outside it exactly as it is, and do not render the mask itself.")
}

impl Native {
    fn round(&self, v: usize) -> usize {
        ((v as f64 / self.m.round as f64).round() as usize).max(256 / self.m.round) * self.m.round
    }
}

impl ImageEngine for Native {
    fn arch(&self) -> &'static str {
        self.m.arch
    }

    fn defaults(&self) -> Defaults {
        let (steps, cfg) = if self.lora.is_some() { (self.m.lora_steps, self.m.lora_cfg) } else { (self.m.steps, self.m.cfg) };
        Defaults { width: 1024, height: 1024, steps, cfg, sampler: Sampler::Euler, schedule: Schedule::Shift, shift: 0.0 }
    }

    fn edits(&self) -> bool {
        self.m.edits
    }

    fn samplers(&self) -> Vec<Sampler> {
        vec![Sampler::Euler]
    }

    fn schedules(&self) -> Vec<Schedule> {
        vec![Schedule::Shift]
    }

    fn guidance(&self) -> bool {
        true
    }

    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }

    fn load_seconds(&self) -> f64 {
        self.load_s
    }

    fn generate(&self, req: &ImageRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Vec<Picture>> {
        let d = self.defaults();
        let t0 = Instant::now();
        let ask_steps = req.steps.unwrap_or(0);
        // the effort ladder: a LoRA below `ladder` steps (distilled, no guidance), a full run at or above it
        let distilled = self.lora.is_some() && ask_steps < self.m.ladder;
        let (steps, cfg) = if distilled {
            (if ask_steps > 0 { ask_steps } else { self.m.lora_steps }, req.cfg.filter(|c| *c > 0.0).unwrap_or(self.m.lora_cfg))
        } else {
            (if ask_steps > 0 { ask_steps } else { self.m.steps }, req.cfg.filter(|c| *c > 0.0).unwrap_or(self.m.cfg))
        };
        let (w, h) = (self.round(req.width.unwrap_or(d.width) as usize), self.round(req.height.unwrap_or(d.height) as usize));
        let mut prompt = req.prompt.trim().to_string();
        let (mut init, mut refs, mut strength) = (None, Vec::new(), 1.0f32);
        if let Some(e) = &req.edit {
            if e.condition {
                if !self.m.edits {
                    return Err(Error(format!("{}: reference edits are not part of this model (img2img, with a strength, is)", self.m.arch)));
                }
                refs.push(fit_ref(&e.image));
                refs.extend(e.refs.iter().map(fit_ref));
                if let Some(m) = &e.mask {
                    refs.push(fit_ref(m));
                    prompt.push_str(&mask_hint(refs.len()));
                }
            } else {
                // plain img2img: the canvas takes the picture's place at the output's size
                init = Some(e.image.rgba().resize_lanczos(w as u32, h as u32));
                strength = e.strength;
            }
        }
        if req.rgba {
            prompt = rgba_prompt(&prompt);
        }
        let negative = req.negative.clone().unwrap_or_default();
        let negative = if negative.trim().is_empty() && cfg > 1.0 { " ".to_string() } else { negative };
        let lora = if distilled { self.lora.as_deref().map(|p| (p, 1.0f32)) } else { None };
        let sd = self.sd.lock().unwrap();
        let mut out = Vec::new();
        for i in 0..req.n.max(1) {
            progress(Step { picture: i, at: 0, of: steps, seconds: t0.elapsed().as_secs_f64() })?;
            let mut stop: Option<Error> = None;
            let init_raw = init.as_ref().map(raw);
            let g = Gen {
                prompt: &prompt, negative: &negative, width: w, height: h, steps, cfg, seed: req.seed + i as u64, sampler: "euler", scheduler: "",
                flow_shift: req.shift.unwrap_or(0.0), rgba: req.rgba, strength, init: init_raw, refs: refs.iter().map(raw).collect(), lora,
            };
            let pic = sd.generate(&g, &mut |step, of, _secs| {
                if let Err(e) = progress(Step { picture: i, at: step.max(0) as u32, of: of.max(1) as u32, seconds: t0.elapsed().as_secs_f64() }) {
                    stop = Some(e);
                    return false;
                }
                true
            });
            if let Some(e) = stop {
                return Err(e);
            }
            let p = pic?;
            out.push(Picture { width: p.w as u32, height: p.h as u32, channels: p.c as u8, data: p.data });
        }
        Ok(out)
    }

    fn report(&self) -> Vec<String> {
        vec![format!("{} on {} through stable-diffusion.cpp (loaded {:.0} s ago{})", self.m.label, self.gpu, self.loaded.elapsed().as_secs_f64(),
                     if self.lora.is_some() { ", distilled LoRA below the ladder" } else { "" })]
    }
}
