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
use nextcuda_image::{At, Defaults, EngineOption, Error, ImageEngine, ImageRequest, LoadOptions, ModelFiles, Picture, Result, Sampler, Schedule, Step};
use nextcuda_qwen3vl::GgufTextEncoder;

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
];

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
}

fn role<'f>(files: &'f ModelFiles, r: &str) -> Result<&'f Path> {
    files.get(r).map(|p| p.as_path()).ok_or_else(|| Error(format!("{ARCH}: no {r} file")))
}

impl RustQwenImage21 {
    /// The model on `gpu`: the transformer and VAE on the card, the text encoder opened (its layers come per use)
    pub fn load(files: &ModelFiles, gpu: &Arc<Gpu>, _o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<RustQwenImage21> {
        let t0 = Instant::now();
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
        let dit = dit::Dit::load(&f, &nsd, log)?;
        let vae = vae::Vae::load(role(files, "vae")?, &nsd, (LATENT_MEAN.to_vec(), LATENT_STD.to_vec()), log)?;
        Ok(RustQwenImage21 { nsd, te, dit, vae, gpu_name: gpu.name.clone(), loaded: Instant::now(), load_s: t0.elapsed().as_secs_f64(), busy: Mutex::new(()) })
    }

    /// The prompt's tokens (the whole chat turn) and how many lead the turn (the system's, dropped)
    fn tokens(&self, prompt: &str) -> (Vec<u32>, usize) {
        let ids = self.te.tokenize(&format!("{SYSTEM}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"));
        let drop = self.te.tokenize(SYSTEM).len();
        (ids, drop)
    }

    /// The text's embedding for the DiT: float32 [T, 4096] on the card, T tokens after the system turn
    fn encode(&self, prompt: &str) -> Result<(DevBuf, usize)> {
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
        Defaults { width: 1024, height: 1024, steps: 40, cfg: 4.0, sampler: Sampler::Euler, schedule: Schedule::Shift, shift: 0.0 }
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
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

    fn generate(&self, req: &ImageRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Vec<Picture>> {
        let _one = self.busy.lock().unwrap();
        let d = self.defaults();
        let t0 = Instant::now();
        if req.edit.is_some() {
            return Err(Error(format!("{ARCH}: edits and img2img are not in the Rust engine yet (the stable-diffusion.cpp one takes them)")));
        }
        if !req.loras.is_empty() {
            return Err(Error(format!("{ARCH}: LoRAs are not in the Rust engine yet")));
        }
        let (w, h) = (req.width.unwrap_or(d.width) as usize, req.height.unwrap_or(d.height) as usize);
        if w % 16 != 0 || h % 16 != 0 {
            return Err(Error(format!("{w}x{h}: the sides are multiples of 16")));
        }
        let steps = req.steps.unwrap_or(d.steps).max(1) as usize;
        let cfg = req.cfg.filter(|c| *c > 0.0).unwrap_or(d.cfg);
        let hw = (h / 16, w / 16);
        let sig = sched::sigmas(steps, hw.0 * hw.1);
        let k = ffi::api()?;
        let gpu = &self.nsd.gpu;
        progress(Step { picture: 0, at: 0, of: steps as u32, seconds: t0.elapsed().as_secs_f64() })?;
        let (embeds, tokens) = self.encode(if req.prompt.trim().is_empty() { " " } else { req.prompt.trim() })?;
        // true guidance: a negative prompt (" " when none was given) and a second pass a step
        let negative = if cfg > 1.0 { Some(self.encode(req.negative.as_deref().filter(|n| !n.trim().is_empty()).unwrap_or(" "))?) } else { None };
        let pre = self.dit.prefix(&self.nsd, &embeds, tokens, self.te.hidden)?;
        let npre = negative.as_ref().map(|(e, t)| self.dit.prefix(&self.nsd, e, *t, self.te.hidden)).transpose()?;
        let n = hw.0 * hw.1 * dit::LATENT;
        let mut out = Vec::new();
        for p in 0..req.n.max(1) {
            let nz = noise(req.seed + p as u64, n);
            let lat = DevBuf::from_f32(gpu, &nz)?;
            for i in 0..sig.len() - 1 {
                prof::reset_clock(&self.nsd)?;
                let mut v = self.dit.velocity(&self.nsd, &pre, &lat, hw, sig[i], None)?;
                if let Some(np) = &npre {
                    let vn = self.dit.velocity(&self.nsd, np, &lat, hw, sig[i], None)?;
                    // v_neg + scale (v - v_neg)
                    // SAFETY: n floats each.
                    ffi::check(unsafe { (k.axpy)(gpu.raw(), v.fp(), vn.fp(), n as i64, -1.0) }, "guidance")?;
                    ffi::check(unsafe { (k.axpy)(gpu.raw(), vn.fp(), v.fp(), n as i64, cfg) }, "guidance")?;
                    v = vn;
                }
                // SAFETY: n floats each.
                ffi::check(unsafe { (k.axpy)(gpu.raw(), lat.fp(), v.fp(), n as i64, sig[i + 1] - sig[i]) }, "euler step")?;
                gpu.sync()?;
                progress(Step { picture: p, at: i as u32 + 1, of: steps as u32, seconds: t0.elapsed().as_secs_f64() })?;
            }
            let (rgba, pw, ph) = self.vae.decode(&self.nsd, &lat.to_f32()?, hw.0, hw.1)?;
            if prof::on() {
                prof::report(&mut |l| eprintln!("{l}"));
            }
            let (data, channels) = if req.rgba {
                (rgba, 4)
            } else {
                (rgba.chunks_exact(4).flat_map(|c| [c[0], c[1], c[2]]).collect(), 3)
            };
            out.push(Picture { width: pw as u32, height: ph as u32, channels, data });
        }
        Ok(out)
    }

    fn report(&self) -> Vec<String> {
        vec![format!("{ARCH} in Rust and CUDA on {} (loaded {:.0} s ago)", self.gpu_name, self.loaded.elapsed().as_secs_f64())]
    }
}
