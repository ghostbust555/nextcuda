//! nh MODEL.gguf [--prompt TEXT] [--n 32] [--raw]: a prompt through the chat template, greedy, the answer streamed
use std::io::Write;
use std::time::Instant;

use nextcuda_llm_nemotronh::{NemotronH, Options};
use nextcuda_tok::{nemotron_chat, Effort, Message};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let opt = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned();
    let Some(path) = a.first() else { eprintln!("usage: nh MODEL.gguf [--prompt TEXT] [--n 32] [--raw]"); std::process::exit(2) };
    if a.iter().any(|x| x == "--info") {
        let f = nextcuda_gguf::Gguf::open(std::path::Path::new(path)).unwrap_or_else(|x| {
            eprintln!("{}", x.0);
            std::process::exit(1)
        });
        let mut m: std::collections::BTreeMap<(String, String), (u64, usize, String)> = Default::default();
        for t in &f.tensors {
            let role = match t.name.strip_prefix("blk.") {
                Some(r) => format!("blk.*.{}", r.split_once('.').map_or(r, |x| x.1)),
                None => t.name.clone(),
            };
            let e = m.entry((role, t.ty.name().to_string())).or_insert((0, 0, format!("{} {:?}", t.name, t.shape)));
            e.0 += t.bytes;
            e.1 += 1;
        }
        let mut v: Vec<_> = m.into_iter().collect();
        v.sort_by_key(|x| std::cmp::Reverse(x.1 .0));
        for ((role, ty), (b, n, ex)) in v {
            println!("{:9.1} MiB x{n:<3} {ty:7} {role}   e.g. {ex}", b as f64 / 1048576.0);
        }
        println!("total {:.2} GiB", f.total_bytes() as f64 / (1u64 << 30) as f64);
        return;
    }
    if let Some(sizes) = opt("--bench") {
        bench(path, &sizes, opt("--new").and_then(|v| v.parse().ok()).unwrap_or(64), &a);
        return;
    }
    let prompt = opt("--prompt").unwrap_or_else(|| "What is the capital of France?".into());
    let n: usize = opt("--n").and_then(|v| v.parse().ok()).unwrap_or(32);
    let mut o = Options::default();
    o.profile = Some(std::path::PathBuf::from(path).with_file_name("expert-profile.txt"));
    let m = NemotronH::load(std::path::Path::new(path), &o, &mut |s| eprintln!("{s}")).unwrap_or_else(|x| {
        eprintln!("load: {}", x.0);
        std::process::exit(1)
    });
    let prof = std::path::PathBuf::from(path).with_file_name("expert-profile.txt");
    if std::env::var("NS_NH_PRELOAD").map_or(true, |v| v != "0") {
        if let Err(x) = m.preload(Some(&prof), &mut |s| eprintln!("{s}")) {
            eprintln!("preload: {}", x.0);
        }
    }
    let tok = m.tokenizer().unwrap_or_else(|x| {
        eprintln!("tokenizer: {x}");
        std::process::exit(1)
    });
    let text = if a.iter().any(|x| x == "--raw") {
        prompt.clone()
    } else {
        nemotron_chat(&[Message { role: "user", content: &prompt, reasoning: None, calls: &[] }], Effort::Low, &[])
    };
    let ids = tok.encode(&text);
    eprintln!("prompt: {} tokens", ids.len());
    let mut s = m.session(o.ctx).unwrap_or_else(|x| {
        eprintln!("session: {}", x.0);
        std::process::exit(1)
    });
    let t0 = Instant::now();
    let mut logits = m.forward(&mut s, &ids, 1).unwrap_or_else(|x| {
        eprintln!("prompt: {}", x.0);
        std::process::exit(1)
    });
    eprintln!("prompt read in {:.1} s ({:.1} tok/s)", t0.elapsed().as_secs_f64(), ids.len() as f64 / t0.elapsed().as_secs_f64());
    eprintln!("prompt times: {:?}", m.times.lock().unwrap());
    *m.times.lock().unwrap() = Default::default();
    let t1 = Instant::now();
    let mut out = Vec::new();
    let (mut tl, mut hl) = (Instant::now(), m.cache_stats());
    for i in 0..n {
        if i > 0 && i % 16 == 0 {
            let h = m.cache_stats();
            eprintln!("\n[{i}: {:.2} tok/s, cache hits {} misses {}]", 16.0 / tl.elapsed().as_secs_f64(), h.0 - hl.0, h.1 - hl.1);
            tl = Instant::now();
            hl = h;
        }
        let l = &logits[0];
        let next = l.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).map(|x| x.0 as u32).unwrap();
        if a.iter().any(|x| x == "--top") {
            let mut v: Vec<(usize, f32)> = l.iter().copied().enumerate().collect();
            v.sort_by(|x, y| y.1.total_cmp(&x.1));
            eprintln!("\n[step {i}: {}]", v.iter().take(4).map(|(t, x)| format!("{:?} {x:.2}", tok.decode(&[*t as u32]))).collect::<Vec<_>>().join(", "));
        }
        out.push(next);
        if tok.decode(&[next]).contains("<|im_end|>") && a.iter().all(|x| x != "--no-stop") {
            break;
        }
        print!("{}", tok.decode(&[next]));
        let _ = std::io::stdout().flush();
        logits = m.forward(&mut s, &[next], 1).unwrap_or_else(|x| {
            eprintln!("\ndecode: {}", x.0);
            std::process::exit(1)
        });
    }
    println!();
    if let Err(x) = m.save_profile(&prof) {
        eprintln!("profile: {x}");
    }
    eprintln!("{} tokens in {:.1} s ({:.2} tok/s); {:?}; expert cache (hits, misses) {:?}", n, t1.elapsed().as_secs_f64(), n as f64 / t1.elapsed().as_secs_f64(), m.times.lock().unwrap(), m.cache_stats());
}

const PARAS: &[&str] = &[
    "The harbour town woke slowly. Fishing boats knocked against the pier while gulls argued over scraps, and the baker unlocked her shutters to the smell of salt and yeast. ",
    "In the laboratory the sequencing run had finished overnight; the technician pulled the report, flagged three samples with low coverage, and wrote a note asking whether the library preparation had been repeated. ",
    "A compiler is a program that translates source text into another form, usually machine code, and along the way it parses, checks types, optimizes and allocates registers. ",
    "The council debated the new tram line for six hours, weighing the cost of tunnelling under the old quarter against the noise of running it overhead past the school. ",
    "To make the sauce, soften the onions in butter, add the tomatoes and a pinch of sugar, and let it simmer until the oil separates; season at the end, never at the start. ",
    "Mountaineers speak of the death zone above eight thousand metres, where the body cannot acclimatize and every decision must be made quickly with a clouded mind. ",
    "The treaty, signed after a long winter of negotiation, divided the river valley between the two crowns and obliged each to keep its fortresses a day's march from the border. ",
];

/// (time, MiB used, watts) of one GPU, sampled by nvidia-smi
type Smp = std::sync::Arc<std::sync::Mutex<Vec<(Instant, f64, f64)>>>;

fn sampler(gpu: usize) -> (Smp, std::process::Child) {
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};
    let mut child = Command::new("nvidia-smi")
        .args(["--query-gpu=index,memory.used,power.draw", "--format=csv,noheader,nounits", "-lms", "250"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("nvidia-smi");
    let out = child.stdout.take().unwrap();
    let smp: Smp = Default::default();
    let s2 = smp.clone();
    std::thread::spawn(move || {
        for line in BufReader::new(out).lines().map_while(Result::ok) {
            let f: Vec<f64> = line.split(',').filter_map(|x| x.trim().parse::<f64>().ok()).collect();
            if f.len() == 3 && f[0] as usize == gpu {
                s2.lock().unwrap().push((Instant::now(), f[1], f[2]));
            }
        }
    });
    (smp, child)
}

/// nh MODEL --bench 20,2000,8000 [--new 64]: a prompt of about each size (token counts exact), read, then `new` greedy tokens
fn bench(path: &str, sizes: &str, new: usize, a: &[String]) {
    let mut o = Options::default();
    o.profile = Some(std::path::PathBuf::from(path).with_file_name("expert-profile.txt"));
    let (smp, mut child) = sampler(o.gpu);
    std::thread::sleep(std::time::Duration::from_millis(600));
    let idle = smp.lock().unwrap().last().map_or(0.0, |x| x.1);
    let m = NemotronH::load(std::path::Path::new(path), &o, &mut |s| eprintln!("{s}")).expect("load");
    if std::env::var("NS_NH_PRELOAD").map_or(true, |v| v != "0") {
        m.preload(o.profile.as_deref(), &mut |s| eprintln!("{s}")).expect("preload");
    }
    let tok = m.tokenizer().expect("tokenizer");
    let sizes: Vec<usize> = sizes.split(',').filter_map(|x| x.trim().parse().ok()).collect();
    let note = a.iter().position(|x| x == "--note").and_then(|i| a.get(i + 1)).cloned().unwrap_or_default();
    println!("| prompt tokens | prompt read (s) | prompt tok/s | decode tok/s | cache hit % | peak GPU MiB (idle {idle:.0}) | avg W | energy J | {note} |");
    println!("|---:|---:|---:|---:|---:|---:|---:|---:|---|");
    for &n in &sizes {
        let mut text = String::new();
        let mut i = 0;
        let mut ids = Vec::new();
        while ids.len() < n {
            text += PARAS[i % PARAS.len()];
            i += 1;
            ids = tok.encode(&text);
        }
        ids.truncate(n);
        let mut s = m.session((n + new + 8).min(12000)).expect("session");
        let (from, h0) = (smp.lock().unwrap().len(), m.cache_stats());
        let ts0 = m.tier_stats();
        let t0 = Instant::now();
        let mut logits = m.forward(&mut s, &ids, 1).expect("prompt");
        let pp = t0.elapsed().as_secs_f64();
        eprintln!("[{n}: prompt times {:?}]", m.times.lock().unwrap());
        *m.times.lock().unwrap() = Default::default();
        let t1 = Instant::now();
        for _ in 0..new {
            let l = &logits[0];
            let next = l.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).map(|x| x.0 as u32).unwrap();
            logits = m.forward(&mut s, &[next], 1).expect("decode");
        }
        let dec = t1.elapsed().as_secs_f64();
        let ts1 = m.tier_stats();
        eprintln!("[{n}: tiers (VRAM hits, pinned hits, misses, copied into VRAM) over prompt + decode: {:?}]", (ts1.0 - ts0.0, ts1.1 - ts0.1, ts1.2 - ts0.2, ts1.3 - ts0.3));
        let h1 = m.cache_stats();
        eprintln!("[{n}: decode times {:?}]", m.times.lock().unwrap());
        *m.times.lock().unwrap() = Default::default();
        let (peak, joules, secs) = {
            let g = smp.lock().unwrap();
            let w = &g[from.min(g.len())..];
            let peak = w.iter().map(|x| x.1).fold(0.0, f64::max);
            let mut j = 0.0;
            for k in 1..w.len() {
                j += w[k].2 * (w[k].0 - w[k - 1].0).as_secs_f64();
            }
            (peak, j, w.last().zip(w.first()).map_or(0.0, |(l, f)| (l.0 - f.0).as_secs_f64()))
        };
        let (hits, miss) = (h1.0 - h0.0, h1.1 - h0.1);
        println!("| {n} | {pp:.1} | {:.1} | {:.2} | {:.1} | {peak:.0} | {:.0} | {joules:.0} | |", n as f64 / pp, new as f64 / dec, 100.0 * hits as f64 / (hits + miss).max(1) as f64, joules / secs.max(0.001));
    }
    if let Err(x) = m.save_profile(&std::path::PathBuf::from(path).with_file_name("expert-profile.txt")) {
        eprintln!("profile: {x}");
    }
    let _ = child.kill();
}
