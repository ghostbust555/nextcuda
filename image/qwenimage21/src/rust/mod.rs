//! Qwen-Image 2.1 in Rust and CUDA, from nextsycl's engine: the pipeline is host code in Rust over the shared
//! diffusion kernels (`kernels/diffusion/nsd.cu`, cuBLAS and CUDA), the transformer kept in the type its GGUF stores
//! it in (7.3 GB at Q8_0: it fits a 12 GB card beside its activations), the text encoder streamed layer by layer
//! from its GGUF, the VAE in half. No Python and no ggml.
//!
//! ```text
//!   prompt -> chat template -> Qwen3-VL 8B text tower (GGUF, a layer on the card at a time) -> last hidden state,
//!     the system turn's tokens dropped
//!   noise [h/16 * w/16, 64] (seeded) -> Euler over the flow-matching sigmas (sched.rs), the DiT's velocity each step
//!     (dit.rs: the text's keys / values computed once, then image rows only)
//!   latents -> VAE decoder (vae.rs) -> RGBA
//! ```

pub mod dit;
pub mod prof;
pub mod sched;
pub mod vae;

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextcuda_core::{DevBuf, Gpu};
use nextcuda_diffusion::kernels::Nsd;
use nextcuda_gguf::Gguf;
use nextcuda_diffusion::lora::{self, Delta};
use nextcuda_image::{At, Defaults, EngineOption, Error, ImageEngine, ImageRequest, LoadOptions, ModelFiles, Picture, Result, Sampler, Schedule, Step};
use nextcuda_qwen3vl::{GgufTextEncoder, Seen, Vision};

use crate::ffi;

pub const ARCH: &str = "qwen-image-2.1";

/// The prompt's chat turn; the system turn's tokens are dropped from the encoding
const SYSTEM: &str = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";

/// The VAE's latent statistics (diffusers' vae/config.json latents_mean / latents_std), per channel
#[allow(clippy::approx_constant)] // measured statistics, not -pi / 6
const LATENT_MEAN: [f32; 64] = [
    0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873, -1.7177, -0.651, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363,
    2.021, 0.0472, 0.932, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464, -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955,
    -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376, -1.0914, -1.9435, -0.2784, -1.368, 0.4809, -0.4433, 0.3764, 0.5729,
    -2.0595, 1.096, -1.326, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505, 0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304,
];
const LATENT_STD: [f32; 64] = [
    3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.791, 3.0785, 3.65, 3.9308, 3.0904, 2.8778, 3.7675, 3.732, 5.0756, 3.2864,
    4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489, 3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.283, 3.5694, 4.2467, 3.9037, 3.2947,
    5.077, 3.5075, 3.27, 3.4767, 2.8063, 5.1125, 3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875, 3.9624, 4.0168, 3.5643,
    4.055, 5.5614, 4.2963, 4.408, 3.4959, 3.8747, 3.7608, 3.5735, 3.149, 3.7662, 3.6746, 3.4563, 3.8161,
];

pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "step-profile", env: "NS_QI_PROFILE", value: "0|1", help: "wait after each named step and report where the time goes", at: At::Load },
    EngineOption { name: "int8", env: "NS_QI_INT8", value: "0|1", help: "the DiT's block matrices in int8 ConvRot: the card's int8 tensor cores (about 3x the half rate on a GeForce)", at: At::Load },
    EngineOption { name: "sigmas", env: "NS_QI_SIGMAS", value: "1,X,...", help: "the noise levels from 1 down (a few-step model's own; its step count)", at: At::Both },
    EngineOption { name: "sigma-shift", env: "NS_QI_SIGMA_SHIFT", value: "dynamic|none", help: "the size's shift on those levels, or as given", at: At::Both },
];

/// A few-step model's (or LoRA's) sigma preset: its nodes from 1 down (0 is appended), and whether the size's
/// exponential shift is put on them (as diffusers' pipeline does to `sigmas=`) or they are used as given
#[derive(Clone, Debug)]
pub struct Preset {
    pub nodes: Vec<f64>,
    pub dynamic: bool,
}

fn preset_of(sigmas: Option<String>, shift: Option<String>) -> Result<Option<Preset>> {
    let Some(v) = sigmas else { return Ok(None) };
    let bad = || Error(format!("sigmas {v}: numbers from 1 down, comma separated"));
    let nodes: Vec<f64> = v.split(',').map(|x| x.trim().parse::<f64>()).collect::<std::result::Result<_, _>>().map_err(|_| bad())?;
    if nodes.is_empty() || nodes.windows(2).any(|w| w[0] <= w[1]) || nodes[0] > 1.0 || *nodes.last().unwrap() <= 0.0 {
        return Err(bad());
    }
    let dynamic = match shift.as_deref() {
        None | Some("dynamic") => true,
        Some("none") => false,
        Some(x) => return Err(Error(format!("sigma-shift {x}: dynamic or none"))),
    };
    Ok(Some(Preset { nodes, dynamic }))
}

/// The pipeline's calculate_dimensions: `area` at aspect `ratio` (width / height), sides rounded to multiples of 32
/// (Python's round: halves to even)
pub fn dims(area: f64, ratio: f64) -> (usize, usize) {
    let w = (area * ratio).sqrt();
    let h = w / ratio;
    let r = |x: f64| -> usize {
        let q = x / 32.0;
        let f = q.floor();
        let n = if (q - f - 0.5).abs() < 1e-12 { if f as i64 % 2 == 0 { f } else { f + 1.0 } } else { q.round() };
        (n as usize).max(1) * 32
    };
    (r(w), r(h))
}

/// A seeded standard-normal draw (splitmix64, Box-Muller): the starting noise
pub fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut out = Vec::with_capacity(n + 1);
    while out.len() < n {
        let (u, v) = (next().max(1e-300), next());
        let r = (-2.0 * u.ln()).sqrt();
        out.push((r * (std::f64::consts::TAU * v).cos()) as f32);
        out.push((r * (std::f64::consts::TAU * v).sin()) as f32);
    }
    out.truncate(n);
    out
}

/// The engine's pieces, on one GPU
pub struct RustQwenImage21 {
    nsd: Nsd,
    te: GgufTextEncoder,
    dit: dit::Dit,
    vae: vae::Vae,
    gpu_name: String,
    loaded: Instant,
    load_s: f64,
    // the one picture at a time the card has room for
    busy: Mutex<()>,
    // the last prompts' embeddings (a negative prompt is the same one every request)
    embeds: Mutex<Vec<(String, Arc<(DevBuf, usize)>)>>,
    /// the model's own sigma preset (a few-step distill), and the LoRAs of its registry entry (kept on every request)
    preset: Option<Preset>,
    base_loras: Vec<Delta>,
    /// the vision tower's file (the `llm-vision` role), opened for an edit and dropped after it
    vision_file: Option<std::path::PathBuf>,
}

fn role<'f>(files: &'f ModelFiles, r: &str) -> Result<&'f Path> {
    files.get(r).map(|p| p.as_path()).ok_or_else(|| Error(format!("{ARCH}: no {r} file")))
}

impl RustQwenImage21 {
    /// The model on `gpu`: the transformer and VAE on the card, the text encoder opened (its layers come per use)
    pub fn load(files: &ModelFiles, gpu: &Arc<Gpu>, o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<RustQwenImage21> {
        let t0 = Instant::now();
        let preset = preset_of(o.setting("NS_QI_SIGMAS"), o.setting("NS_QI_SIGMA_SHIFT"))?;
        if let Some(p) = &preset {
            log(format!("sigmas: a {}-step preset ({})", p.nodes.len(), if p.dynamic { "the size's shift" } else { "as given" }));
        }
        let mut base_loras = Vec::new();
        for l in &o.merge_loras {
            let d = lora::read(&l.path, l.scale).map_err(Error)?;
            log(format!("LoRA {} x {}: {} matrices, rank {}", l.name, l.scale, d.len(), d.first().map_or(0, |x| x.r)));
            base_loras.extend(d);
        }
        nextcuda_core::use_kind("image");
        let f = Gguf::open(role(files, "transformer")?).map_err(|e| Error(e.0))?;
        let (total, free) = gpu.memory()?;
        let dit_b: u64 = f.tensors.iter().map(|t| t.ty.bytes(t.elements()).unwrap_or(0)).sum();
        // the transformer as stored, the VAE, and ~2.5 GiB of activations at 1024 x 1024
        let need = dit_b + (1u64 << 30) + (5u64 << 29);
        let free = free.unwrap_or(total);
        if need > free {
            return Err(Error(format!("{ARCH} needs about {:.1} GiB on {}; {:.1} GiB are free", need as f64 / (1u64 << 30) as f64, gpu.name,
                                      free as f64 / (1u64 << 30) as f64)));
        }
        let nsd = Nsd::new(gpu)?;
        let te = GgufTextEncoder::load(role(files, "text-encoder")?, gpu, log)?;
        let int8 = o.setting("NS_QI_INT8").is_some_and(|v| v == "1");
        let dit = dit::Dit::load(&f, &nsd, int8, log)?;
        let vae = vae::Vae::load(role(files, "vae")?, &nsd, (LATENT_MEAN.to_vec(), LATENT_STD.to_vec()), log)?;
        dit.set_loras(&nsd, &base_loras)?;
        Ok(RustQwenImage21 {
            nsd,
            te,
            dit,
            vae,
            gpu_name: gpu.name.clone(),
            loaded: Instant::now(),
            load_s: t0.elapsed().as_secs_f64(),
            busy: Mutex::new(()),
            embeds: Mutex::new(Vec::new()),
            preset,
            base_loras,
            vision_file: files.get("llm-vision").cloned(),
        })
    }

    /// The prompt's tokens (the whole chat turn) and how many lead the turn (the system's, dropped)
    fn tokens(&self, prompt: &str) -> (Vec<u32>, usize) {
        let ids = self.te.tokenize(&format!("{SYSTEM}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"));
        let drop = self.te.tokenize(SYSTEM).len();
        (ids, drop)
    }

    /// The text's embedding for the DiT: float32 [T, 4096] on the card, T tokens after the system turn
    fn encode(&self, prompt: &str) -> Result<Arc<(DevBuf, usize)>> {
        if let Some((_, e)) = self.embeds.lock().unwrap().iter().find(|(p, _)| p == prompt) {
            return Ok(e.clone());
        }
        let e = Arc::new(self.encode_now(prompt)?);
        let mut c = self.embeds.lock().unwrap();
        c.push((prompt.to_string(), e.clone()));
        if c.len() > 8 {
            c.remove(0);
        }
        Ok(e)
    }

    /// An edit's pictures: each resized to the output area at its own aspect (multiples of 32, Lanczos), seen by the
    /// vision tower over white, and encoded by the VAE (RGBA in [-1, 1]): (what the text encoder takes, the latents
    /// and their size). The tower is opened for this and let go after it.
    #[allow(clippy::type_complexity)]
    fn pictures(&self, pics: &[&Picture], area: usize) -> Result<(Vec<Seen>, Vec<(DevBuf, (usize, usize))>)> {
        let path = self.vision_file.as_ref().ok_or_else(|| Error(format!("{ARCH}: the model has no vision tower file (its llm-vision role): edits need it")))?;
        let vis = Vision::load(path, &self.nsd.gpu, &mut |_| {})?;
        let mut seen = Vec::new();
        let mut lats = Vec::new();
        for p in pics {
            let (w, h) = dims(area as f64, p.width as f64 / p.height as f64);
            let r = p.rgba().resize_lanczos(w as u32, h as u32);
            let rgb: Vec<f32> = r.data.chunks_exact(4).flat_map(|c| {
                let a = c[3] as f32 / 255.0;
                [0, 1, 2].map(|i| (c[i] as f32 * a + 255.0 * (1.0 - a)).round())
            }).collect();
            seen.push(vis.see(&self.nsd, &rgb, h, w)?);
            let px: Vec<f32> = r.data.iter().map(|v| *v as f32 / 255.0 * 2.0 - 1.0).collect();
            let l = self.vae.encode(&self.nsd, &px, h, w)?;
            lats.push((DevBuf::from_f32(&self.nsd.gpu, &l)?, (h / 16, w / 16)));
        }
        Ok((seen, lats))
    }

    /// An edit prompt's ids: the template with a slot a picture, each slot's `<|image_pad|>` expanded to the
    /// picture's `tokens[k]` merged tokens
    fn edit_ids(&self, prompt: &str, tokens: &[usize]) -> Result<Vec<u32>> {
        let slots: Vec<String> = (1..=tokens.len()).map(|i| format!("<image{i}><|vision_start|><|image_pad|><|vision_end|>")).collect();
        let prompt = if prompt.is_empty() { " " } else { prompt };
        let text = format!("{SYSTEM}<|im_start|>user\n{}{prompt}<|im_end|>\n<|im_start|>assistant\n", slots.join(" "));
        let pad = self.te.tok.id("<|image_pad|>").ok_or_else(|| Error("the tokenizer has no <|image_pad|>".into()))?;
        let mut ids = Vec::new();
        let mut k = 0;
        for id in self.te.tokenize(&text) {
            if id == pad {
                let n = *tokens.get(k).ok_or_else(|| Error("more picture slots than pictures".into()))?;
                ids.extend(std::iter::repeat_n(pad, n));
                k += 1;
            } else {
                ids.push(id);
            }
        }
        Ok(ids)
    }

    /// An edit prompt with its pictures: (the text's embedding after the system turn, its tokens, which are slots)
    fn encode_edit(&self, prompt: &str, seen: &[Seen]) -> Result<(DevBuf, usize, Vec<bool>)> {
        let ids = self.edit_ids(prompt, &seen.iter().map(|v| v.n).collect::<Vec<_>>())?;
        let pad = self.te.tok.id("<|image_pad|>").ok_or_else(|| Error("the tokenizer has no <|image_pad|>".into()))?;
        let drop = self.te.tokenize(SYSTEM).len();
        let refs: Vec<&Seen> = seen.iter().collect();
        let all = self.te.encode_seen(&self.nsd, &ids, &refs, usize::MAX)?;
        let t = ids.len() - drop;
        let d = self.te.hidden;
        let out = DevBuf::f32(&self.nsd.gpu, t * d)?;
        out.copy_within(0, &all, drop * d * 4, t * d * 4)?;
        self.nsd.gpu.sync()?;
        Ok((out, t, ids[drop..].iter().map(|i| *i == pad).collect()))
    }

    fn encode_now(&self, prompt: &str) -> Result<(DevBuf, usize)> {
        let (ids, drop) = self.tokens(prompt);
        let all = self.te.encode_ids(&self.nsd, &ids)?;
        let t = ids.len() - drop;
        let d = self.te.hidden;
        let out = DevBuf::f32(&self.nsd.gpu, t * d)?;
        out.copy_within(0, &all, drop * d * 4, t * d * 4)?;
        // `all` is freed on return: the copy first
        self.nsd.gpu.sync()?;
        Ok((out, t))
    }
}

impl ImageEngine for RustQwenImage21 {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        let steps = self.preset.as_ref().map_or(40, |p| p.nodes.len() as u32);
        let cfg = if self.preset.is_some() { 1.0 } else { 4.0 };
        Defaults { width: 1024, height: 1024, steps, cfg, sampler: Sampler::Euler, schedule: Schedule::Shift, shift: 0.0 }
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }
    /// every shared sampler (Euler on the model's schedule stays on the GPU; the others step on the host)
    fn samplers(&self) -> Vec<Sampler> {
        Sampler::ALL.to_vec()
    }
    fn schedules(&self) -> Vec<Schedule> {
        Schedule::ALL.to_vec()
    }
    /// a picture (and references) with instructions: the pictures through the vision tower and the VAE
    fn edits(&self) -> bool {
        true
    }
    fn steps_for(&self, req: &ImageRequest) -> u32 {
        match (req.steps, req.extra.get("sigmas")) {
            (Some(s), _) => s,
            (None, Some(v)) => v.split(',').count() as u32,
            (None, None) => self.defaults().steps,
        }
    }
    fn guidance(&self) -> bool {
        true
    }
    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }

    fn generate(&self, req: &ImageRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Vec<Picture>> {
        let _one = self.busy.lock().unwrap();
        let d = self.defaults();
        let t0 = Instant::now();
        // an edit conditions on its pictures (the first changed, the others references); img2img starts from the
        // picture instead
        let (img2img, pics): (Option<&nextcuda_image::Edit>, Vec<&Picture>) = match &req.edit {
            Some(e) if e.condition => {
                if e.mask.is_some() {
                    return Err(Error(format!("{ARCH}: masks are not part of this model's edits (describe the change instead)")));
                }
                (None, std::iter::once(&e.image).chain(e.refs.iter()).collect())
            }
            Some(e) => (Some(e), Vec::new()),
            None => (None, Vec::new()),
        };
        let area = (d.width * d.height) as usize;
        let (dw, dh) = match pics.last() {
            Some(p) => dims(area as f64, p.width as f64 / p.height as f64),
            None => (d.width as usize, d.height as usize),
        };
        let (w, h) = (req.width.map_or(dw, |v| v as usize), req.height.map_or(dh, |v| v as usize));
        if w % 16 != 0 || h % 16 != 0 {
            return Err(Error(format!("{w}x{h}: the sides are multiples of 16")));
        }
        let cfg = req.cfg.filter(|c| *c > 0.0).unwrap_or(d.cfg);
        let sampler = req.sampler.unwrap_or(Sampler::Euler);
        let schedule = req.schedule.unwrap_or(Schedule::Shift);
        let hw = (h / 16, w / 16);
        // a request's own sigmas win over the model's preset
        let mine = preset_of(req.extra.get("sigmas").cloned(), req.extra.get("sigma-shift").cloned())?;
        let preset = mine.as_ref().or(self.preset.as_ref());
        let steps = match (&mine, req.steps) {
            (Some(p), None) => p.nodes.len(),
            (_, s) => s.unwrap_or(d.steps) as usize,
        }
        .max(1);
        let levels = |steps: usize| -> Result<Vec<f32>> {
            Ok(match preset {
                // another schedule than the model's own: ComfyUI's, at the shift the model's would use for this size
                _ if schedule != Schedule::Shift => {
                    let shift = req.shift.filter(|s| *s > 0.0).unwrap_or_else(|| sched::mu(hw.0 * hw.1).exp() as f32);
                    nextcuda_diffusion::samplers::sigmas_for(sampler, schedule, steps as u32, shift)
                }
                Some(p) if steps == p.nodes.len() => sched::preset(&p.nodes, hw.0 * hw.1, p.dynamic),
                Some(p) => {
                    return Err(Error(format!("this model runs its own {} sigmas (a few-step distill): {steps} steps is not one of its schedules", p.nodes.len())))
                }
                None => sched::sigmas(steps, hw.0 * hw.1),
            })
        };
        let mut sig = levels(steps)?;
        // a strength below 1: start partway down from the picture - the levels a full run of steps / strength would
        // take, the last `steps` of them
        let strength = img2img.map_or(1.0, |e| e.strength);
        if !(0.05..=1.0).contains(&strength) {
            return Err(Error(format!("strength {strength}: from 0.05 to 1")));
        }
        if strength < 1.0 {
            sig = match (preset, schedule == Schedule::Shift) {
                (Some(_), true) => sched::below(&sig, strength),
                _ => sched::tail(&levels(((steps as f32 / strength).round() as usize).max(steps))?, steps),
            };
        }
        let own = sampler == Sampler::Euler && schedule == Schedule::Shift;
        let k = ffi::api()?;
        let gpu = &self.nsd.gpu;
        // the request's LoRAs beside the entry's own, for this picture only
        let mut loras: Vec<Delta> = self.base_loras.clone();
        for l in &req.loras {
            loras.extend(lora::read(&l.path, l.scale).map_err(Error)?);
        }
        if !req.loras.is_empty() {
            self.dit.set_loras(&self.nsd, &loras)?;
        }
        let result = (|| -> Result<Vec<Picture>> {
            progress(Step { picture: 0, at: 0, of: (sig.len() - 1) as u32, seconds: t0.elapsed().as_secs_f64() })?;
            let start = match img2img {
                Some(e) if strength < 1.0 => {
                    let r = e.image.rgba().resize_lanczos(w as u32, h as u32);
                    let px: Vec<f32> = r.data.iter().map(|v| *v as f32 / 255.0 * 2.0 - 1.0).collect();
                    Some(self.vae.encode(&self.nsd, &px, h, w)?)
                }
                _ => None,
            };
            let t_text = Instant::now();
            // an edit's pictures through the vision tower and the VAE; then each prompt's embedding and its prefix
            let (seen, lats) = if pics.is_empty() { (Vec::new(), Vec::new()) } else { self.pictures(&pics, area)? };
            let conds: Vec<(&DevBuf, (usize, usize))> = lats.iter().map(|(b, s)| (b, *s)).collect();
            let prompt = if req.prompt.trim().is_empty() { " " } else { req.prompt.trim() };
            let neg_prompt = req.negative.as_deref().filter(|n| !n.trim().is_empty()).unwrap_or(" ");
            let prefix_of = |p: &str| -> Result<dit::Prefix> {
                if seen.is_empty() {
                    let e = self.encode(p)?;
                    self.dit.prefix(&self.nsd, &e.0, e.1, self.te.hidden)
                } else {
                    let (e, t, pads) = self.encode_edit(p, &seen)?;
                    self.dit.prefix_with(&self.nsd, &e, t, self.te.hidden, &pads, &conds)
                }
            };
            let pre = prefix_of(prompt)?;
            // true guidance: a negative prompt (" " when none was given) and a second pass a step
            let npre = if cfg > 1.0 { Some(prefix_of(neg_prompt)?) } else { None };
            crate::sd::record(format!("get_learned_condition completed, taking {:.2}s", t_text.elapsed().as_secs_f64()));
            let n = hw.0 * hw.1 * dit::LATENT;
            let of = (sig.len() - 1) as u32;
            // the velocity at latents `lat`, sigma `s`: guided when there is a negative prompt (v_neg + cfg (v - v_neg),
            // each token's norm put back to v's, as the pipeline's true guidance)
            let velocity = |lat: &DevBuf, s: f32| -> Result<DevBuf> {
                let mut v = self.dit.velocity(&self.nsd, &pre, lat, hw, s, None)?;
                if let Some(np) = &npre {
                    let vn = self.dit.velocity(&self.nsd, np, lat, hw, s, None)?;
                    // SAFETY: n floats each, rows of LATENT values.
                    ffi::check(unsafe { (k.cfg_rescale)(gpu.raw(), v.fp(), vn.fp(), (n / dit::LATENT) as i64, dit::LATENT as i64, cfg, v.fp()) }, "guidance")?;
                }
                Ok(v)
            };
            let mut out = Vec::new();
            for p in 0..req.n.max(1) {
                let mut nz = noise(req.seed + p as u64, n);
                if let Some(l0) = &start {
                    // x = s0 noise + (1 - s0) picture; the samplers scale their start by s0, the GPU path takes x as is
                    let s0 = sig[0];
                    for (x, l) in nz.iter_mut().zip(l0) {
                        *x = if own { s0 * *x + (1.0 - s0) * l } else { *x + (1.0 - s0) / s0 * l };
                    }
                }
                let t_sample = Instant::now();
                let lat = if own {
                    // the model's own path: Euler on the GPU
                    let lat = DevBuf::from_f32(gpu, &nz)?;
                    for i in 0..sig.len() - 1 {
                        prof::reset_clock(&self.nsd)?;
                        let v = velocity(&lat, sig[i])?;
                        // SAFETY: n floats each.
                        ffi::check(unsafe { (k.axpy)(gpu.raw(), lat.fp(), v.fp(), n as i64, sig[i + 1] - sig[i]) }, "euler step")?;
                        gpu.sync()?;
                        progress(Step { picture: p, at: i as u32 + 1, of, seconds: t0.elapsed().as_secs_f64() })?;
                    }
                    lat
                } else {
                    let lat = DevBuf::f32(gpu, n)?;
                    let mut failed: Option<Error> = None;
                    let mut model = |x: &[f32], sigma: f32| -> std::result::Result<Vec<f32>, String> {
                        let run = || -> Result<Vec<f32>> {
                            lat.write(0, &x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
                            let v = velocity(&lat, sigma)?.to_f32()?;
                            // the x0 prediction
                            Ok(x.iter().zip(&v).map(|(a, b)| a - sigma * b).collect())
                        };
                        run().map_err(|e| e.0)
                    };
                    // the samplers' own draws: a stream a call, from the seed (apart from the starting noise's)
                    let mut draw = 0u64;
                    let seed = req.seed + p as u64;
                    let mut noise_fn = |m: usize| -> Vec<f32> {
                        draw += 1;
                        noise(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(draw), m)
                    };
                    let start_x: Vec<f32> = nz.iter().map(|v| v * sig[0]).collect();
                    let mut step = |i: usize| {
                        if failed.is_none() {
                            if let Err(e) = progress(Step { picture: p, at: i as u32 + 1, of, seconds: t0.elapsed().as_secs_f64() }) {
                                failed = Some(e);
                            }
                        }
                    };
                    let res = nextcuda_diffusion::samplers::sample(sampler, &mut model, start_x, &sig, &mut noise_fn, &mut step).map_err(Error)?;
                    if let Some(e) = failed {
                        return Err(e);
                    }
                    // ComfyUI's inverse noise scaling for a schedule that stops above 0
                    let last = *sig.last().unwrap_or(&0.0);
                    let fin: Vec<f32> = if last > 0.0 { res.iter().map(|v| v / (1.0 - last)).collect() } else { res };
                    DevBuf::from_f32(gpu, &fin)?
                };
                crate::sd::record(format!("generating 1 latent images completed, taking {:.2}s", t_sample.elapsed().as_secs_f64()));
                let t_vae = Instant::now();
                let (rgba, pw, ph) = self.vae.decode(&self.nsd, &lat.to_f32()?, hw.0, hw.1)?;
                crate::sd::record(format!("decode_first_stage completed, taking {:.2}s", t_vae.elapsed().as_secs_f64()));
                if prof::on() {
                    prof::report(&mut |l| eprintln!("{l}"));
                }
                let (data, channels) = if req.rgba { (rgba, 4) } else { (rgba.chunks_exact(4).flat_map(|c| [c[0], c[1], c[2]]).collect(), 3) };
                out.push(Picture { width: pw as u32, height: ph as u32, channels, data });
            }
            Ok(out)
        })();
        // the request's LoRAs end with it
        if !req.loras.is_empty() {
            self.dit.set_loras(&self.nsd, &self.base_loras)?;
        }
        result
    }

    fn report(&self) -> Vec<String> {
        vec![format!("{ARCH} in Rust and CUDA on {} (loaded {:.0} s ago)", self.gpu_name, self.loaded.elapsed().as_secs_f64())]
    }
}
