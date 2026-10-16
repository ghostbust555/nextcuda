//! `nextcuda image bench`: the image kind's benchy - the same method as nextsycl's `llm bench`: fixed prompt and
//! seeds, the first picture after a load (cold) kept apart from the warm ones, every run's stages clocked, the card's
//! memory, power and energy sampled beside it, and the matrix written as `matrix.md` and `matrix.jsonl`.
//!
//!     nextcuda image bench [--models ID,ID] [--sizes 768x768,1024x1024] [--steps 12] [--runs 3] [--gpu 0]
//!                          [--prompt TEXT] [--out DIR] [--note TEXT] [--opt-NAME VALUE]...
//!
//! - **Each model** is loaded once (its load time is a row), then each size makes `--runs` pictures: the first is
//!   the cold one (the weights paged in, the graphs built), the rest the warm ones; the table gives the cold run
//!   in brackets beside the warm median, as nextsycl's decode table does.
//! - **Stages** (seconds): text encode, the first sampling step (it builds the graph and stages the weights) and the
//!   mean of the others, the sampler, the VAE decode, the whole picture.
//! - **Peaks:** `nvidia-smi` sampled every 250 ms for the run's window: the highest memory used on each card (less
//!   the idle reading before the load: the model's own), the power drawn, its integral as joules, the average watts;
//!   the process's peak resident host memory (VmHWM).
//! - **Out:** `matrix.md` (printed too) and `matrix.jsonl` in --out (default benchy-results/image-v1-<date>); a
//!   result worth keeping is copied to `docs/benchy/` under a name with the date and what changed.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use nextcuda_image::ImageRequest;
use serde_json::{json, Value};

use crate::image::{load, opt};

const PROMPT: &str = "a lighthouse on a rocky coast at dusk, long exposure, dramatic clouds, photorealistic";

/// (when, gpu, MiB used, watts, PCIe gen, PCIe width)
type Sample = (Instant, usize, f64, f64, f64, f64);

/// `nvidia-smi` streaming every 250 ms into a list, until dropped
struct Sampler {
    samples: Arc<Mutex<Vec<Sample>>>,
    child: std::process::Child,
}

impl Sampler {
    fn start() -> Result<Sampler, String> {
        let mut child = Command::new("nvidia-smi")
            .args(["--query-gpu=index,memory.used,power.draw,pcie.link.gen.current,pcie.link.width.current", "--format=csv,noheader,nounits", "-lms", "250"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("nvidia-smi: {e}"))?;
        let out = child.stdout.take().ok_or("nvidia-smi has no stdout")?;
        let samples = Arc::new(Mutex::new(Vec::new()));
        let s2 = samples.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(out).lines().map_while(Result::ok) {
                let f: Vec<f64> = line.split(',').filter_map(|x| x.trim().parse::<f64>().ok()).collect();
                if f.len() == 5 {
                    s2.lock().unwrap().push((Instant::now(), f[0] as usize, f[1], f[2], f[3], f[4]));
                }
            }
        });
        Ok(Sampler { samples, child })
    }

    fn len(&self) -> usize {
        self.samples.lock().unwrap().len()
    }

    /// The window from sample `from`: per GPU (index, peak MiB, joules, seconds covered, highest PCIe gen, width seen)
    fn window(&self, from: usize) -> Vec<(usize, f64, f64, f64, f64, f64)> {
        let s = self.samples.lock().unwrap();
        let mut out: Vec<(usize, f64, f64, f64, f64, f64)> = Vec::new();
        let mut last: Vec<(usize, Instant, f64)> = Vec::new();
        for (t, g, mem, w, gen, width) in s.iter().skip(from) {
            let i = match out.iter().position(|o| o.0 == *g) {
                Some(i) => i,
                None => {
                    out.push((*g, 0.0, 0.0, 0.0, 0.0, 0.0));
                    out.len() - 1
                }
            };
            out[i].1 = out[i].1.max(*mem);
            out[i].4 = out[i].4.max(*gen);
            out[i].5 = out[i].5.max(*width);
            if let Some(l) = last.iter_mut().find(|l| l.0 == *g) {
                let dt = t.duration_since(l.1).as_secs_f64();
                out[i].2 += (l.2 + w) / 2.0 * dt;
                out[i].3 += dt;
                l.1 = *t;
                l.2 = *w;
            } else {
                last.push((*g, *t, *w));
            }
        }
        out
    }

    fn idle(&self) -> Vec<(usize, f64)> {
        let s = self.samples.lock().unwrap();
        let mut v: Vec<(usize, f64)> = Vec::new();
        for (_, g, m, _, _, _) in s.iter().rev().take(16) {
            if !v.iter().any(|x| x.0 == *g) {
                v.push((*g, *m));
            }
        }
        v
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() { 0.0 } else { v[v.len() / 2] }
}

fn host_peak_mib() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("VmHWM:")).and_then(|l| l.split_whitespace().nth(1).and_then(|k| k.parse::<f64>().ok())))
        .map_or(0.0, |kb| kb / 1024.0)
}

pub fn run(cfg: &nextcuda_models::Config, args: &[String]) -> Result<(), String> {
    let gpu: usize = opt(args, "--gpu").and_then(|v| v.parse().ok()).unwrap_or(0);
    let runs: usize = opt(args, "--runs").and_then(|v| v.parse().ok()).unwrap_or(3).max(1);
    let steps: u32 = opt(args, "--steps").and_then(|v| v.parse().ok()).unwrap_or(12);
    let sizes: Vec<(u32, u32)> = opt(args, "--sizes")
        .unwrap_or("768x768")
        .split(',')
        .map(|s| {
            let (w, h) = s.split_once('x').ok_or("--sizes WxH,WxH")?;
            Ok((w.parse().map_err(|_| "--sizes WxH")?, h.parse().map_err(|_| "--sizes WxH")?))
        })
        .collect::<Result<_, String>>()?;
    let prompt = opt(args, "--prompt").unwrap_or(PROMPT).to_string();
    let models: Vec<String> = match opt(args, "--models") {
        Some(m) => m.split(',').map(str::to_string).collect(),
        None => nextcuda_models::registry::all(cfg)?
            .iter()
            .filter(|m| nextcuda_models::registry::kind_of(m) == "image" && m["enabled"] != false)
            .filter_map(|m| m["id"].as_str().map(str::to_string))
            .collect(),
    };
    if models.is_empty() {
        return Err("no image model is registered (nextcuda models pull <id>)".into());
    }
    let date = {
        let d = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
        // days since the epoch to a calendar date (civil_from_days)
        let z = (d / 86400) as i64 + 719468;
        let era = z.div_euclid(146097);
        let doe = z.rem_euclid(146097);
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let day = doy - (153 * mp + 2) / 5 + 1;
        let month = if mp < 10 { mp + 3 } else { mp - 9 };
        let year = yoe + era * 400 + (month <= 2) as i64;
        format!("{year:04}-{month:02}-{day:02}")
    };
    let out_dir = opt(args, "--out").map(str::to_string).unwrap_or_else(|| format!("benchy-results/image-v1-{date}"));
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("{out_dir}: {e}"))?;
    let sampler = Sampler::start()?;
    std::thread::sleep(std::time::Duration::from_millis(800));
    let mut jsonl = std::fs::File::create(format!("{out_dir}/matrix.jsonl")).map_err(|e| e.to_string())?;
    let mut md = String::new();
    let gpu_name = nextcuda_core::gpus().ok().and_then(|g| g.into_iter().find(|x| x.0 == gpu)).map_or("?".into(), |g| g.1);
    md.push_str(&format!("# nextcuda image bench, {date}\n\nGPU {gpu} ({gpu_name}); prompt: \"{prompt}\"; {steps} steps; seeds 1..{runs}; {runs} pictures a size, the first (cold) in brackets beside the warm median.\n"));
    if let Some(n) = opt(args, "--note") {
        md.push_str(&format!("\n{n}\n"));
    }
    for (i, n) in nextcuda_core::gpus().unwrap_or_default() {
        if let Some(l) = nextcuda_core::pcie_link(i) {
            md.push_str(&format!("\nGPU {i} ({n}): PCIe {l} at the start of the run (the link steps down when a card is idle; each row says what it ran at)\n"));
        }
    }
    md.push_str("\nStages in seconds; peaks of GPU memory are the model's own (the idle reading before the load taken off); energy and watts are the card's `power.draw` over the picture.\n");
    for model in &models {
        let mut a: Vec<String> = args.to_vec();
        a.retain(|x| x != "--models");
        a.extend(["--model".into(), model.clone(), "--gpu".into(), gpu.to_string()]);
        let idle = sampler.idle().iter().find(|x| x.0 == gpu).map_or(0.0, |x| x.1);
        let t0 = Instant::now();
        let (m, engine) = load(cfg, &a)?;
        let load_s = t0.elapsed().as_secs_f64();
        std::thread::sleep(std::time::Duration::from_millis(600));
        let after_load = sampler.idle().iter().find(|x| x.0 == gpu).map_or(0.0, |x| x.1);
        md.push_str(&format!("\n## {} ({})\n\nload {load_s:.1} s; GPU memory after the load {:.0} MiB (idle {idle:.0}); engine: {}\n\n", m["id"].as_str().unwrap_or(model),
                             m["title"].as_str().unwrap_or(""), after_load, engine.report().first().cloned().unwrap_or_default()));
        md.push_str("| size | steps | text encode | step 1 | step (mean of 2..n) | sampler | VAE | total | peak GPU MiB | PCIe | host RSS MiB | energy J | avg W |\n");
        md.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|\n");
        for &(w, h) in &sizes {
            let mut rows: Vec<Value> = Vec::new();
            for r in 0..runs {
                let _ = nextcuda_image_qwenimage21::sd::take_log();
                let from = sampler.len();
                let t = Instant::now();
                let stamps: std::cell::RefCell<Vec<(u32, f64)>> = std::cell::RefCell::new(Vec::new());
                let req = ImageRequest { prompt: prompt.clone(), width: Some(w), height: Some(h), steps: Some(steps), seed: 1 + r as u64, n: 1, ..Default::default() };
                engine
                    .generate(&req, &mut |s| {
                        stamps.borrow_mut().push((s.at, s.seconds));
                        Ok(())
                    })
                    .map_err(|e| e.0)?;
                let total = t.elapsed().as_secs_f64();
                std::thread::sleep(std::time::Duration::from_millis(300));
                let (text, sample, vae) = nextcuda_image_qwenimage21::sd::stage_times(&nextcuda_image_qwenimage21::sd::take_log());
                let st = stamps.borrow();
                // the seconds each denoising step took: the gaps between the progress lines
                let at: Vec<(u32, f64)> = st.iter().filter(|x| x.0 > 0).cloned().collect();
                // the first step: from the end of the text encode (the library's own line) to its progress line
                let first = (at.first().map(|x| x.1).unwrap_or(0.0) - text.unwrap_or(0.0)).max(0.0);
                let rest: Vec<f64> = at.windows(2).map(|p| p[1].1 - p[0].1).collect();
                let step_mean = if rest.is_empty() { 0.0 } else { rest.iter().sum::<f64>() / rest.len() as f64 };
                let win = sampler.window(from);
                let peak = win.iter().find(|x| x.0 == gpu).map_or(0.0, |x| x.1);
                let (link_gen, link_width) = win.iter().find(|x| x.0 == gpu).map_or((0.0, 0.0), |x| (x.4, x.5));
                let (joules, secs): (f64, f64) = win.iter().fold((0.0, 0.0), |a, x| (a.0 + x.2, a.1.max(x.3)));
                let row = json!({
                    "model": m["id"], "size": format!("{w}x{h}"), "steps": steps, "run": r, "cold": r == 0, "load_s": load_s,
                    "text_encode_s": text, "step1_s": first, "step_mean_s": step_mean, "sampler_s": sample, "vae_s": vae, "total_s": total,
                    "peak_gpu_mib": peak, "peak_gpu_model_mib": (peak - idle).max(0.0), "host_rss_peak_mib": host_peak_mib(),
                    "pcie_gen": link_gen, "pcie_width": link_width, "pcie_max": nextcuda_core::pcie_link(gpu),
                    "energy_j": joules, "avg_w": if secs > 0.0 { joules / secs } else { 0.0 }, "idle_gpu_mib": idle, "gpu": gpu, "gpu_name": gpu_name,
                    "note": opt(args, "--note"),
                });
                writeln!(jsonl, "{row}").map_err(|e| e.to_string())?;
                rows.push(row);
            }
            let f = |k: &str, rows: &[&Value]| -> f64 { median(&mut rows.iter().filter_map(|r| r[k].as_f64()).collect::<Vec<_>>()) };
            let cold = &rows[0];
            let warm: Vec<&Value> = if rows.len() > 1 { rows[1..].iter().collect() } else { rows.iter().collect() };
            let cell = |k: &str, p: usize| -> String {
                let c = cold[k].as_f64().unwrap_or(0.0);
                format!("{:.p$} ({:.p$})", f(k, &warm), c, p = p)
            };
            let link = format!("g{:.0} x{:.0}", f("pcie_gen", &warm), f("pcie_width", &warm));
            md.push_str(&format!("| {w}x{h} | {steps} | {} | {} | {} | {} | {} | {} | {} | {link} | {:.0} | {} | {} |\n", cell("text_encode_s", 2), cell("step1_s", 1),
                                 cell("step_mean_s", 2), cell("sampler_s", 1), cell("vae_s", 1), cell("total_s", 1), cell("peak_gpu_model_mib", 0),
                                 host_peak_mib(), cell("energy_j", 0), cell("avg_w", 0)));
        }
        drop(engine);
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
    std::fs::write(format!("{out_dir}/matrix.md"), &md).map_err(|e| e.to_string())?;
    println!("{md}");
    println!("wrote {out_dir}/matrix.md and matrix.jsonl");
    Ok(())
}
