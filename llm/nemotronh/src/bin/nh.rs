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
