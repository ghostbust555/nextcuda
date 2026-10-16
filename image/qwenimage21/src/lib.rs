//! Qwen-Image 2.1 on NVIDIA GPUs, phase 1: the pipeline is stable-diffusion.cpp (`sd-cli`, built with CUDA) driven by
//! the platform's own python engine (`python/qwen_image21.py`, vendored), spoken to through `python/worker.py` as
//! JSON lines. This crate is only the contract's side of that: it owns the worker process, maps an `ImageRequest` to
//! the worker's request, turns progress lines into `Step`s and the PNG it wrote back into a `Picture`.
//!
//! The files are what stable-diffusion.cpp reads (not the SYCL engine's int8 safetensors): the transformer GGUF, the
//! Qwen3-VL text encoder as a GGUF (+ its vision projector for edits), the 2.1 VAE. Phase 2 replaces this crate's
//! insides with the Rust runtime; the arch name, the files by role and the contract stay.
//!
//! Settings (`LoadOptions::setting`, else the environment): `NC_SDCPP_BIN` (the sd-cli binary), `NC_PYTHON` (python3),
//! `NC_IMAGE_WORKER` (worker.py; default `python/worker.py` beside the program or in the source tree),
//! `NC_SDCPP_EXTRA_ARGS`, `NC_QWEN21_ALPHA` (auto|native|rembg).

pub mod ffi;
pub mod gguffix;
pub mod native;
pub mod rust;
pub mod sd;
pub mod selftest;

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextcuda_image::{At, Defaults, EngineOption, Error, Gpu, ImageEngine, ImageKind, ImageRequest, LoadOptions, ModelFiles, Picture, Result, Sampler, Schedule, Step};
use serde_json::{json, Value};

pub const ARCH: &str = "qwen-image-2.1";

/// The engine's kernels, each against host arithmetic on `gpu` (`nextcuda image selftest`): the worst error of each
pub fn selftest(gpu: &Arc<Gpu>) -> Result<Vec<f32>> {
    let mut v = selftest::run(gpu)?;
    // the shared diffusion kernels under it, when the library has them
    v.extend(nextcuda_diffusion::kernels_check::selftest(gpu)?);
    Ok(v)
}
/// The files it needs; `llm-vision` (the projector an edit's references go through) is read when there is one
pub const ROLES: &[&str] = &["transformer", "text-encoder", "vae"];

pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "backend", env: "NC_IMAGE_BACKEND", value: "native|rust|python", help: "native: in process through stable-diffusion.cpp (default); rust: the Rust engine on the shared CUDA kernels; python: the sd-cli worker", at: At::Load },
    EngineOption { name: "sdcpp-bin", env: "NC_SDCPP_BIN", value: "PATH", help: "the stable-diffusion.cpp sd-cli binary (CUDA build)", at: At::Load },
    EngineOption { name: "sdcpp-extra-args", env: "NC_SDCPP_EXTRA_ARGS", value: "ARGS", help: "more arguments for every sd-cli run", at: At::Load },
    EngineOption { name: "alpha", env: "NC_QWEN21_ALPHA", value: "auto|native|rembg", help: "where a transparent picture's alpha comes from", at: At::Load },
];

pub fn kind() -> ImageKind {
    ImageKind { archs: &[ARCH], name: "Qwen-Image 2.1 (7B single-stream DiT, Qwen3-VL text encoder, RGBA VAE; stable-diffusion.cpp)", roles: ROLES, load, options: OPTIONS }
}

/// Qwen-Image 2512 (the 20B MMDiT with Qwen2.5-VL): the same library, its own architecture and ladder
pub const ARCH_2512: &str = "qwen-image-2512";

pub fn kind_2512() -> ImageKind {
    ImageKind { archs: &[ARCH_2512], name: "Qwen-Image 2512 (20B MMDiT, Qwen2.5-VL text encoder; stable-diffusion.cpp)", roles: ROLES, load: load_2512, options: native::OPTIONS }
}

fn load_2512(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn ImageEngine>> {
    if !sd::available() {
        return Err(Error(format!("{ARCH_2512}: the image kernel library has no stable-diffusion.cpp bridge (kernels/build.sh with SDCPP and SDLIB)")));
    }
    Ok(Box::new(native::load(native::QWEN_IMAGE_2512, files, gpus, o, log)?))
}

/// The worker process: its pipes, kept while it lives
struct Worker {
    child: Child,
    stdin: ChildStdin,
    out: BufReader<ChildStdout>,
}

impl Worker {
    fn send(&mut self, v: &Value) -> Result<()> {
        writeln!(self.stdin, "{v}").and_then(|_| self.stdin.flush()).map_err(|e| Error(format!("the image worker: {e}")))
    }

    /// The next event line; an error when the worker is gone
    fn next(&mut self) -> Result<Value> {
        let mut line = String::new();
        loop {
            line.clear();
            let n = self.out.read_line(&mut line).map_err(|e| Error(format!("the image worker: {e}")))?;
            if n == 0 {
                return Err(Error("the image worker exited".into()));
            }
            if let Ok(v) = serde_json::from_str::<Value>(line.trim()) {
                return Ok(v);
            }
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub struct QwenImage21 {
    models: Value,
    settings: Value,
    python: String,
    script: PathBuf,
    turbo: bool,
    worker: Mutex<Option<Worker>>,
    load_s: f64,
    gpus: Vec<(usize, String)>,
}

static SEQ: AtomicU64 = AtomicU64::new(0);

/// worker.py: `NC_IMAGE_WORKER`, else `python/worker.py` beside the program (a release), else the source tree's
fn find_script(o: &LoadOptions) -> Result<PathBuf> {
    if let Some(p) = o.setting("NC_IMAGE_WORKER") {
        return Ok(PathBuf::from(p));
    }
    let mut tried = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        for rel in ["python/worker.py", "../python/worker.py", "../image/qwenimage21/python/worker.py", "../../image/qwenimage21/python/worker.py"] {
            if let Some(d) = exe.parent() {
                let p = d.join(rel);
                if p.is_file() {
                    return Ok(p);
                }
                tried.push(p);
            }
        }
    }
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("python/worker.py");
    if src.is_file() {
        return Ok(src);
    }
    Err(Error(format!("{ARCH}: worker.py not found (NC_IMAGE_WORKER); looked in {} places", tried.len() + 1)))
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn ImageEngine>> {
    // the Rust engine: the pipeline in Rust over the shared CUDA kernels, no ggml (NC_IMAGE_BACKEND=rust)
    if o.setting("NC_IMAGE_BACKEND").is_some_and(|v| v == "rust") {
        let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
        return Ok(Box::new(rust::RustQwenImage21::load(files, gpu, o, log)?));
    }
    // in process, when the library has stable-diffusion.cpp (the model stays loaded); else the phase 1 worker
    if sd::available() && o.setting("NC_IMAGE_BACKEND").is_none_or(|v| v != "python") {
        return Ok(Box::new(native::load(native::QWEN_IMAGE_21, files, gpus, o, log)?));
    }
    let path = |role: &str| -> Result<String> {
        files.get(role).map(|p| p.to_string_lossy().into_owned()).ok_or_else(|| Error(format!("{ARCH}: no {role} file")))
    };
    let bin = o.setting("NC_SDCPP_BIN").ok_or_else(|| Error(format!("{ARCH}: NC_SDCPP_BIN (the sd-cli binary) is not set")))?;
    let mut models = json!({
        "sdcpp_bin": bin,
        "qwen21_gguf": path("transformer")?,
        "qwen21_llm": path("text-encoder")?,
        "qwen21_vae": path("vae")?,
    });
    if let Some(v) = files.get("llm-vision") {
        models["qwen21_llm_vision"] = json!(v.to_string_lossy());
    }
    // a turbo model is the full one with its LoRA merged at load (a registry entry's lora)
    let turbo = o.merge_loras.first();
    if let Some(l) = turbo {
        models["qwen21_turbo_lora"] = json!(l.path.to_string_lossy());
    }
    let mut settings = json!({});
    if let Some(a) = o.setting("NC_SDCPP_EXTRA_ARGS") {
        settings["sdcpp_extra_args"] = json!(a);
    }
    if let Some(a) = o.setting("NC_QWEN21_ALPHA") {
        settings["qwen21_alpha"] = json!(a);
    }
    let e = QwenImage21 {
        models,
        settings,
        python: o.setting("NC_PYTHON").unwrap_or_else(|| "python3".into()),
        script: find_script(o)?,
        turbo: turbo.is_some(),
        worker: Mutex::new(None),
        load_s: 0.0,
        gpus: gpus.iter().map(|g| (g.index, g.name.clone())).collect(),
    };
    let t = Instant::now();
    log(format!("{ARCH}: starting {} {}", e.python, e.script.display()));
    e.ensure(&mut e.worker.lock().unwrap())?;
    Ok(Box::new(QwenImage21 { load_s: t.elapsed().as_secs_f64(), ..e }))
}

impl QwenImage21 {
    /// The worker, started and loaded when there is none (it is killed on a cancel)
    fn ensure(&self, slot: &mut Option<Worker>) -> Result<()> {
        if slot.is_some() {
            return Ok(());
        }
        let mut cmd = Command::new(&self.python);
        cmd.arg(&self.script).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit());
        // the card the engine was given is the one sd-cli sees, numbered as nvidia-smi numbers them
        cmd.env("CUDA_DEVICE_ORDER", "PCI_BUS_ID");
        if let Some((i, _)) = self.gpus.first() {
            cmd.env("CUDA_VISIBLE_DEVICES", i.to_string());
        }
        let mut child = cmd.spawn().map_err(|e| Error(format!("{ARCH}: starting {}: {e}", self.python)))?;
        let stdin = child.stdin.take().ok_or_else(|| Error("the image worker has no stdin".into()))?;
        let out = BufReader::new(child.stdout.take().ok_or_else(|| Error("the image worker has no stdout".into()))?);
        let mut w = Worker { child, stdin, out };
        w.send(&json!({"op": "load", "models": self.models, "settings": self.settings}))?;
        loop {
            let v = w.next()?;
            match v["event"].as_str() {
                Some("loaded") => break,
                Some("error") => return Err(Error(format!("{ARCH}: {}", v["message"].as_str().unwrap_or("the worker failed to load")))),
                _ => {}
            }
        }
        *slot = Some(w);
        Ok(())
    }
}

/// A scratch directory removed when it goes out of scope
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Result<Scratch> {
        let d = std::env::temp_dir().join(format!("nextcuda-qwen21-{}-{}", std::process::id(), SEQ.fetch_add(1, Ordering::Relaxed)));
        std::fs::create_dir_all(&d).map_err(|e| Error(format!("{}: {e}", d.display())))?;
        Ok(Scratch(d))
    }
    fn put(&self, name: &str, p: &Picture) -> Result<String> {
        let f = self.0.join(name);
        p.write_png(&f, &[]).map_err(Error)?;
        Ok(f.to_string_lossy().into_owned())
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

impl ImageEngine for QwenImage21 {
    fn arch(&self) -> &'static str {
        ARCH
    }

    fn defaults(&self) -> Defaults {
        if self.turbo {
            Defaults { width: 1024, height: 1024, steps: 6, cfg: 1.0, sampler: Sampler::Euler, schedule: Schedule::Shift, shift: 1.0 }
        } else {
            Defaults { width: 1024, height: 1024, steps: 40, cfg: 4.0, sampler: Sampler::Euler, schedule: Schedule::Shift, shift: 1.0 }
        }
    }

    fn edits(&self) -> bool {
        true
    }

    fn samplers(&self) -> Vec<Sampler> {
        vec![Sampler::Euler]
    }

    fn schedules(&self) -> Vec<Schedule> {
        vec![Schedule::Shift]
    }

    fn guidance(&self) -> bool {
        !self.turbo
    }

    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }

    fn steps_for(&self, req: &ImageRequest) -> u32 {
        if self.turbo {
            self.defaults().steps
        } else {
            req.steps.unwrap_or(self.defaults().steps)
        }
    }

    fn load_seconds(&self) -> f64 {
        self.load_s
    }

    fn generate(&self, req: &ImageRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Vec<Picture>> {
        let d = self.defaults();
        let steps = self.steps_for(req);
        let started = Instant::now();
        let scratch = Scratch::new()?;
        let n = req.n.max(1);
        // an edit: the picture to change is reference 1, the others follow; a plain img2img goes as the init image
        let (mut refs, mut mask, mut init, mut strength) = (Vec::new(), None, None, None);
        if let Some(e) = &req.edit {
            if e.condition {
                refs.push(scratch.put("ref1.png", &e.image)?);
                for (i, r) in e.refs.iter().enumerate() {
                    refs.push(scratch.put(&format!("ref{}.png", i + 2), r)?);
                }
                if let Some(m) = &e.mask {
                    mask = Some(scratch.put("mask.png", m)?);
                }
            } else {
                init = Some(scratch.put("init.png", &e.image)?);
                strength = Some(e.strength);
            }
        }
        let mut out = Vec::new();
        for i in 0..n {
            let target = scratch.0.join(format!("out{i}.png"));
            let id = SEQ.fetch_add(1, Ordering::Relaxed);
            let mut last_phase = String::new();
            let mut last_at = u32::MAX;
            let mut guard = self.worker.lock().unwrap();
            self.ensure(&mut guard)?;
            let w = guard.as_mut().unwrap();
            w.send(&json!({
                "op": "generate", "id": id, "prompt": req.prompt, "negative": req.negative.clone().unwrap_or_default(),
                "width": req.width.unwrap_or(d.width), "height": req.height.unwrap_or(d.height), "steps": steps,
                "cfg": req.cfg.unwrap_or(d.cfg), "seed": req.seed + i as u64, "references": refs, "mask": mask,
                "transparent": req.rgba, "turbo": self.turbo, "init_image": init, "strength": strength,
                "out": target.to_string_lossy(),
            }))?;
            loop {
                let v = match w.next() {
                    Ok(v) => v,
                    Err(e) => {
                        *guard = None;
                        return Err(e);
                    }
                };
                match v["event"].as_str() {
                    Some("progress") => {
                        // sd-cli names a phase many times (its encode bars); the server wants a change
                        let phase = v["phase"].as_str().unwrap_or("").to_string();
                        let at = if phase == "denoise" { v["step"].as_u64().unwrap_or(0) as u32 } else { 0 };
                        if phase == last_phase && at == last_at {
                            continue;
                        }
                        last_phase = phase;
                        last_at = at;
                        let step = Step {
                            picture: i,
                            at: if v["phase"] == "denoise" { v["step"].as_u64().unwrap_or(0) as u32 } else { 0 },
                            of: steps,
                            seconds: started.elapsed().as_secs_f64(),
                        };
                        if let Err(e) = progress(step) {
                            // a cancel: sd-cli cannot be told to stop between steps, so the process goes, and the
                            // next request starts a fresh one
                            *guard = None;
                            return Err(e);
                        }
                    }
                    Some("done") => {
                        let p = Picture::read_png(&target).map_err(Error)?;
                        out.push(p);
                        break;
                    }
                    Some("error") => return Err(Error(format!("{ARCH}: {}", v["message"].as_str().unwrap_or("the worker failed")))),
                    _ => {}
                }
            }
        }
        Ok(out)
    }

    fn report(&self) -> Vec<String> {
        let mut v = vec![format!("{ARCH} via stable-diffusion.cpp ({})", if self.turbo { "turbo" } else { "full" })];
        for (i, n) in &self.gpus {
            v.push(format!("GPU {i}: {n}"));
        }
        v
    }
}
