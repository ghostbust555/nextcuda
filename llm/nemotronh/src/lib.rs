//! Nemotron-H MoE (`nemotron_h_moe`, llama.cpp's src/models/nemotron-h.cpp: Nemotron 3.5 Super 120B-A12B) on one CUDA
//! card: 88 layers of three kinds, each a pre-norm residual block - Mamba-2 mixers (an input projection to z | x B C |
//! dt, a causal conv + SiLU over x B C, the selective scan, a SiLU(z)-gated group RMS norm, the output projection),
//! latent MoE layers (a sigmoid router with a selection bias picks 22 of 512 experts, weights renormalized and scaled;
//! the token projected to a 1024-wide latent, each expert relu^2(up) then down there, the sum projected back; plus a
//! relu^2 shared expert on the full width) and attention layers (GQA, no positions). After nextsycl's `nemotronh`
//! engine (the same graph and names), on CUDA kernels: Strata's i-quant products and plain ones of its own.
//!
//! ```text
//!   x = embed(tokens)
//!   per layer:  x += mixer(attn_norm(x))      mixer: mamba2 | moe | attention
//!   logits = output(output_norm(x))
//! ```
//!
//! The weights stay in their GGUF blocks; every product goes through the Q8_1 kernels, a pass at most `DEC` rows. What
//! does not change the answer, only where the bytes are: the dense weights and as many layers' experts as fit are in
//! VRAM; the other layers' through a cache of pinned host memory (which the kernels read in place over PCIe) of the
//! experts used lately, the misses read from the file.

pub mod ffi;

use std::ffi::c_void;
use std::sync::Arc;
use std::time::Instant;

use nextcuda_core::{Arena, DevBuf, Error, Gpu, HostBuf, Result};
use nextcuda_gguf::{GType, Gguf, Tensor};

/// Rows a pass takes
pub const DEC: usize = 8;
/// Rows a pass of the grouped MoE path takes (prompts); passes up to SMALL rows run each expert entry on its own
pub const CHUNK: usize = 1024;
const SMALL: usize = 4;

fn e(x: impl std::fmt::Display) -> Error {
    Error(x.to_string())
}

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

/// What a layer is
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Mamba,
    Moe,
    Attn,
}

/// The model's shape, from the file's metadata
#[derive(Clone, Debug)]
pub struct Geometry {
    pub layers: usize,
    pub kinds: Vec<Kind>,
    pub n_embd: usize,
    pub vocab: usize,
    pub eps: f32,
    pub heads: usize,
    pub heads_kv: usize,
    pub head: usize,
    pub conv: usize,
    pub state: usize,
    pub groups: usize,
    pub inner: usize,
    pub ssm_heads: usize,
    pub experts: usize,
    pub used: usize,
    pub latent: usize,
    pub n_ff_exp: usize,
    pub n_ff_shared: usize,
    pub w_scale: f32,
    pub w_norm: bool,
    pub nextn: usize,
}

impl Geometry {
    pub fn read(f: &Gguf) -> std::result::Result<Geometry, String> {
        let arch = f.architecture();
        let u = |k: &str| f.arch_meta(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| format!("no {arch}.{k}"));
        let fl = |k: &str| f.arch_meta(k).and_then(|v| v.as_f64()).map(|v| v as f32).ok_or_else(|| format!("no {arch}.{k}"));
        // a per-layer value: an array, or one value for every layer
        let per = |k: &str, n: usize| -> std::result::Result<Vec<usize>, String> {
            let v = f.arch_meta(k).ok_or_else(|| format!("no {arch}.{k}"))?;
            match v.as_array() {
                Some(a) => Ok(a.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect()),
                None => Ok(vec![v.as_u64().unwrap_or(0) as usize; n]),
            }
        };
        let nextn = u("nextn_predict_layers").unwrap_or(0);
        let blocks = u("block_count")?;
        let layers = blocks - nextn;
        let kv = per("attention.head_count_kv", blocks)?;
        let ff = per("feed_forward_length", blocks)?;
        if kv.len() < layers || ff.len() < layers {
            return Err(format!("{arch}: per-layer head_count_kv / feed_forward_length shorter than the {layers} layers"));
        }
        // a layer is a mixer when it has neither key heads nor a feed-forward (llama.cpp's is_recr)
        let kinds: Vec<Kind> = (0..layers).map(|l| if kv[l] > 0 { Kind::Attn } else if ff[l] > 0 { Kind::Moe } else { Kind::Mamba }).collect();
        let heads_kv = kv.iter().take(layers).copied().find(|v| *v > 0).unwrap_or(0);
        let eps = fl("attention.layer_norm_rms_epsilon").or_else(|_| fl("attention.layer_norm_epsilon"))?;
        let g = Geometry {
            layers,
            kinds,
            n_embd: u("embedding_length")?,
            vocab: f.tensor("token_embd.weight").map(|t| t.shape[0] as usize).ok_or("no token_embd.weight")?,
            eps,
            heads: u("attention.head_count")?,
            heads_kv,
            head: u("attention.key_length")?,
            conv: u("ssm.conv_kernel")?,
            state: u("ssm.state_size")?,
            groups: u("ssm.group_count")?,
            inner: u("ssm.inner_size")?,
            ssm_heads: u("ssm.time_step_rank")?,
            experts: u("expert_count")?,
            used: u("expert_used_count")?,
            latent: u("moe_latent_size").unwrap_or(0),
            n_ff_exp: u("expert_feed_forward_length")?,
            n_ff_shared: u("expert_shared_feed_forward_length")?,
            w_scale: fl("expert_weights_scale").unwrap_or(1.0),
            w_norm: f.arch_meta("expert_weights_norm").and_then(|v| v.as_u64()).is_some_and(|v| v != 0),
            nextn,
        };
        if g.latent == 0 {
            return Err(format!("{arch}: no moe_latent_size (this engine reads latent-MoE files: Nemotron 3.5 Super)"));
        }
        if g.head != 128 || g.heads_kv == 0 || !g.heads.is_multiple_of(g.heads_kv) {
            return Err(format!("attention heads of {} ({} / {}): head 128 with a multiple of the key heads", g.head, g.heads, g.heads_kv));
        }
        if !g.state.is_multiple_of(32) || g.state > 256 || !g.inner.is_multiple_of(g.ssm_heads) || !g.ssm_heads.is_multiple_of(g.groups) {
            return Err(format!("Mamba-2 of state {}, inner {}, {} heads in {} groups: state a multiple of 32 up to 256, inner a multiple of the heads", g.state, g.inner, g.ssm_heads, g.groups));
        }
        if !(2..=8).contains(&g.conv) || g.used == 0 || g.used > g.experts {
            return Err(format!("a conv of {} / {} of {} experts a token", g.conv, g.used, g.experts));
        }
        Ok(g)
    }

    /// x | B | C: the conv's channels
    pub fn conv_dim(&self) -> usize {
        self.inner + 2 * self.groups * self.state
    }

    /// z | x B C | dt: the mixer's input projection
    pub fn in_proj(&self) -> usize {
        2 * self.inner + 2 * self.groups * self.state + self.ssm_heads
    }

    pub fn count(&self, k: Kind) -> usize {
        self.kinds.iter().filter(|x| **x == k).count()
    }
}

/// A matrix [rows (outputs), cols (inputs)]: its stored blocks, or float32 (small or unquantized tensors)
pub struct Mat {
    pub buf: DevBuf,
    pub ty: GType,
    pub rows: usize,
    pub cols: usize,
}

/// Where a layer's experts are
enum Home {
    Vram(DevBuf),
    /// the file, through the expert cache: the experts a pass chose are in pinned host memory (read in place over PCIe)
    Cached,
}

/// A layer's experts [n, rows, cols], each `stride` bytes
struct Experts {
    home: Home,
    name: String,
    ty: GType,
    rows: usize,
    cols: usize,
    stride: usize,
}

impl Experts {
    fn tier(&self) -> &'static str {
        match self.home {
            Home::Vram(_) => "VRAM",
            Home::Cached => "cached",
        }
    }
}

struct Mamba {
    inp: Mat,
    conv_w: DevBuf,
    conv_b: DevBuf,
    dt_b: DevBuf,
    a: DevBuf,
    d: DevBuf,
    norm: DevBuf,
    out: Mat,
}

struct Attn {
    q: Mat,
    k: Mat,
    v: Mat,
    o: Mat,
}

struct Moe {
    layer: usize,
    router: Mat,
    /// the selection bias (exp_probs_b)
    bias: Vec<f32>,
    to_latent: Mat,
    from_latent: Mat,
    up: Experts,
    down: Experts,
    sh_up: Mat,
    sh_down: Mat,
}

enum Mix {
    Mamba(Mamba),
    Moe(Box<Moe>),
    Attn(Attn),
}

struct Layer {
    norm: DevBuf,
    mix: Mix,
}

enum LState {
    /// conv inputs [conv-1][channels], scan state [heads][head dim][state], float32
    Mamba { conv: DevBuf, s: DevBuf },
    /// key / value cache [kv heads][max_ctx][128] half
    Attn { k: DevBuf, v: DevBuf },
    None,
}

/// A conversation
pub struct Session {
    pub pos: usize,
    pub max_ctx: usize,
    layers: Vec<LState>,
}

/// Where the time of a pass went (seconds), by what the layer did
#[derive(Default, Clone, Debug)]
pub struct Times {
    pub mamba: f64,
    pub attn: f64,
    pub moe: f64,
    pub moe_fetch: f64,
    pub head: f64,
    pub passes: usize,
    /// seconds in the reader threads and the bytes they read
    pub io_s: f64,
    pub io_bytes: u64,
    /// NS_NH_PROFILE=1: a MoE layer's parts (seconds, each waited for)
    pub parts: std::collections::BTreeMap<&'static str, f64>,
}

/// A Q8_1 copy of rows
struct Q8 {
    buf: DevBuf,
}

pub struct Options {
    /// the card
    pub gpu: usize,
    /// the longest conversation
    pub ctx: usize,
    /// GiB of pinned host memory for experts (the layers' after those in VRAM)
    pub pinned_gib: f64,
    /// MiB of VRAM kept free beside the weights (the arena, caches, the driver's own)
    pub reserve_mib: u64,
    /// a profile (`save_profile`'s file): the VRAM left after the dense weights then holds the experts it says are wanted
    /// most, of any layer, instead of the first layers' whole
    pub profile: Option<std::path::PathBuf>,
    /// rows of a prompt pass (an expert read once for all of them)
    pub chunk: usize,
    /// MiB of pinned memory for a prompt pass's misses (part of the pinned budget)
    pub scratch_mib: usize,
}

impl Default for Options {
    fn default() -> Options {
        let env = |k: &str| std::env::var(k).ok();
        Options {
            gpu: env("NS_NH_GPU").and_then(|v| v.parse().ok()).unwrap_or(0),
            ctx: env("NS_NH_CTX").and_then(|v| v.parse().ok()).unwrap_or(4096),
            pinned_gib: env("NS_NH_PINNED_GIB").and_then(|v| v.parse().ok()).unwrap_or(14.0),
            reserve_mib: env("NS_NH_RESERVE_MIB").and_then(|v| v.parse().ok()).unwrap_or(1800),
            profile: None,
            scratch_mib: env("NS_NH_SCRATCH_MIB").and_then(|v| v.parse().ok()).unwrap_or(1024),
            chunk: env("NS_NH_CHUNK").and_then(|v| v.parse().ok()).unwrap_or(512).clamp(1, CHUNK),
        }
    }
}

pub struct NemotronH {
    pub g: Geometry,
    file: Gguf,
    gpu: Arc<Gpu>,
    k: &'static ffi::Api,
    layers: Vec<Layer>,
    embd: Tensor,
    output_norm: DevBuf,
    output: Mat,
    arena: Arena,
    /// the experts of the layers not in VRAM: slots of pinned host memory, the recently used kept (LRU)
    cache: Option<std::sync::Mutex<ExpertCache>>,
    direct: Option<Direct>,
    chunk: usize,
    pub times: std::sync::Mutex<Times>,
    pub load_s: f64,
}

type Key = (usize, usize, bool);

/// Slots of one size (the stride of some layers' up or down matrices) of pinned host memory, the kernels reading in place
struct Pool {
    _host: HostBuf,
    view: DevBuf,
    stride: usize,
    owner: Vec<Option<Key>>,
    used: Vec<u64>,
}

/// The experts of the layers not in VRAM, a matrix (up or down) a slot of its size's pool: the used lately kept (LRU)
struct ExpertCache {
    pools: Vec<Pool>,
    map: std::collections::HashMap<Key, (usize, usize)>,
    clock: u64,
    hits: u64,
    misses: u64,
    /// how often each matrix was wanted (the profile saved for the next start)
    wanted: std::collections::HashMap<Key, u32>,
    /// the matrices kept in VRAM (offset in `hot_buf`), chosen by the profile at preload
    hot: std::collections::HashMap<Key, usize>,
    hot_buf: Option<DevBuf>,
    hot_hits: u64,
    /// pinned room for the misses of a prompt pass, used and dropped (a sweep over every layer's experts would only push
    /// out what the profile put in the pools)
    scratch: Option<(HostBuf, DevBuf)>,
    scratch_used: usize,
    scratch_map: std::collections::HashMap<Key, usize>,
}

impl NemotronH {
    /// Fill the cache before the first token: the matrices a saved profile says were wanted most, then the rest in file order,
    /// until the pools are full (reads on a few threads). Returns the slots filled
    pub fn preload(&self, profile: Option<&std::path::Path>, log: &mut dyn FnMut(String)) -> Result<usize> {
        let Some(cache) = &self.cache else { return Ok(0) };
        let t0 = Instant::now();
        let mut want: std::collections::HashMap<Key, u32> = Default::default();
        if let Some(text) = profile.and_then(|p| std::fs::read_to_string(p).ok()) {
            for line in text.lines() {
                let v: Vec<&str> = line.split_whitespace().collect();
                if let [l, x, d, n] = v[..] {
                    if let (Ok(l), Ok(x), Ok(d), Ok(n)) = (l.parse(), x.parse(), d.parse::<u8>(), n.parse()) {
                        want.insert((l, x, d == 1), n);
                    }
                }
            }
        }
        let have_profile = !want.is_empty();
        let mut c = cache.lock().unwrap();
        // (tensor name, expert, down, stride) of every cached matrix, by pool
        let mut jobs: Vec<(usize, usize, usize, bool, usize)> = Vec::new(); // (layer, expert, slot-in-pool-encoded: pool, down, ...)
        let mut cand: Vec<Vec<Key>> = vec![Vec::new(); c.pools.len()];
        for (l, ly) in self.layers.iter().enumerate() {
            if let Mix::Moe(m) = &ly.mix {
                if !matches!(m.up.home, Home::Cached) {
                    continue;
                }
                for (down, ex) in [(false, &m.up), (true, &m.down)] {
                    let p = c.pools.iter().position(|q| q.stride == ex.stride).unwrap();
                    cand[p].extend((0..self.g.experts).map(|x| (l, x, down)));
                }
            }
        }
        let mut filled = 0usize;
        // the VRAM's: the most wanted of every pool's candidates until the hot buffer is full
        let mut hot_jobs: Vec<(Key, usize, usize)> = Vec::new(); // (key, stride, offset)
        if let Some(hb) = &c.hot_buf {
            let cap = hb.len;
            let mut all: Vec<(Key, usize)> = Vec::new();
            for (p, keys) in cand.iter().enumerate() {
                let st = c.pools[p].stride;
                all.extend(keys.iter().map(|k| (*k, st)));
            }
            all.sort_by_key(|(k, _)| std::cmp::Reverse(want.get(k).copied().unwrap_or(0)));
            let mut at = 0usize;
            for (k, st) in all {
                if at + st > cap {
                    continue;
                }
                hot_jobs.push((k, st, at));
                at += st;
            }
            for (k, _, off) in &hot_jobs {
                c.hot.insert(*k, *off);
            }
            for keys in cand.iter_mut() {
                keys.retain(|k| !c.hot.contains_key(k));
            }
        }
        for (p, keys) in cand.iter_mut().enumerate() {
            if have_profile {
                keys.sort_by_key(|k| std::cmp::Reverse(want.get(k).copied().unwrap_or(0)));
            }
            let n = c.pools[p].owner.len().min(keys.len());
            for (sl, key) in keys.iter().take(n).enumerate() {
                c.pools[p].owner[sl] = Some(*key);
                c.map.insert(*key, (p, sl));
                jobs.push((key.0, key.1, p, key.2, sl));
            }
            filled += n;
        }
        let bases: Vec<usize> = c.pools.iter().map(|q| q._host.as_slice().as_ptr() as usize).collect();
        let strides: Vec<usize> = c.pools.iter().map(|q| q.stride).collect();
        drop(c);
        let nt: usize = std::env::var("NS_NH_IO_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
        let err = std::sync::Mutex::new(None);
        let mut by_layer: Vec<Vec<&(usize, usize, usize, bool, usize)>> = vec![Vec::new(); nt];
        for (i, j) in jobs.iter().enumerate() {
            by_layer[i % nt].push(j);
        }
        std::thread::scope(|sc| {
            for part in &by_layer {
                let (file, err, layers, bases, strides, direct) = (&self.file, &err, &self.layers, &bases, &strides, &self.direct);
                sc.spawn(move || {
                    for &&(l, x, p, down, sl) in part {
                        let Mix::Moe(m) = &layers[l].mix else { continue };
                        let ex = if down { &m.down } else { &m.up };
                        let Some(t) = file.tensor(&ex.name) else { continue };
                        // SAFETY: the slot is inside its pool and owned by this job.
                        let dst = unsafe { std::slice::from_raw_parts_mut((bases[p] + sl * strides[p]) as *mut u8, ex.stride) };
                        let r = match direct {
                            Some(d) => d.read(t, (x * ex.stride) as u64, dst).map_err(|y| y.to_string()),
                            None => file.read_into(t, (x * ex.stride) as u64, dst).map_err(|y| y.0),
                        };
                        if let Err(y) = r {
                            *err.lock().unwrap() = Some(y);
                        }
                    }
                });
            }
        });
        if let Some(y) = err.into_inner().unwrap() {
            return Err(e(y));
        }
        // the hot matrices into VRAM, in chunks through a host buffer
        if !hot_jobs.is_empty() {
            let hb = {
                let c = cache.lock().unwrap();
                let b = c.hot_buf.as_ref().unwrap();
                b.view(0, b.len)?
            };
            let chunk_bytes = 192usize << 20;
            let mut stage = vec![0u8; chunk_bytes];
            let mut i = 0;
            while i < hot_jobs.len() {
                let start = hot_jobs[i].2;
                let mut j = i;
                while j < hot_jobs.len() && hot_jobs[j].2 + hot_jobs[j].1 - start <= chunk_bytes {
                    j += 1;
                }
                let part = &hot_jobs[i..j];
                let sbase = stage.as_mut_ptr() as usize;
                let err2 = std::sync::Mutex::new(None);
                std::thread::scope(|sc| {
                    for q in part.chunks(part.len().div_ceil(nt).max(1)) {
                        let (file, layers, err2, direct) = (&self.file, &self.layers, &err2, &self.direct);
                        sc.spawn(move || {
                            for &((l, x, down), st, off) in q {
                                let Mix::Moe(m) = &layers[l].mix else { continue };
                                let ex = if down { &m.down } else { &m.up };
                                let Some(t) = file.tensor(&ex.name) else { continue };
                                // SAFETY: each job owns its [off - start, +st) of the staging buffer.
                                let dst = unsafe { std::slice::from_raw_parts_mut((sbase + off - start) as *mut u8, st) };
                                let r = match direct {
                                    Some(d) => d.read(t, (x * st) as u64, dst).map_err(|y| y.to_string()),
                                    None => file.read_into(t, (x * st) as u64, dst).map_err(|y| y.0),
                                };
                                if let Err(y) = r {
                                    *err2.lock().unwrap() = Some(y);
                                }
                            }
                        });
                    }
                });
                if let Some(y) = err2.into_inner().unwrap() {
                    return Err(e(y));
                }
                let end = part.last().map(|q| q.2 + q.1).unwrap();
                hb.write(start, &stage[..end - start])?;
                i = j;
            }
            filled += hot_jobs.len();
        }
        log(format!("nemotronh: the cache preloaded with {filled} matrices ({}) in {:.1} s", if have_profile { "by the saved profile" } else { "in file order" }, t0.elapsed().as_secs_f64()));
        Ok(filled)
    }

    /// The matrices wanted so far, to a file `preload` reads
    pub fn save_profile(&self, path: &std::path::Path) -> std::io::Result<()> {
        let Some(cache) = &self.cache else { return Ok(()) };
        let c = cache.lock().unwrap();
        // this run's counts on top of the old profile's, halved: the hot set follows the topics of the latest runs but
        // keeps what earlier ones wanted
        let mut merged: std::collections::HashMap<Key, u32> = Default::default();
        if let Ok(text) = std::fs::read_to_string(path) {
            for line in text.lines() {
                let v: Vec<&str> = line.split_whitespace().collect();
                if let [l, x, d, n] = v[..] {
                    if let (Ok(l), Ok(x), Ok(d), Ok(n)) = (l.parse(), x.parse(), d.parse::<u8>(), n.parse::<u32>()) {
                        merged.insert((l, x, d == 1), n / 2);
                    }
                }
            }
        }
        for (k, n) in &c.wanted {
            *merged.entry(*k).or_default() += *n;
        }
        let mut v: Vec<(&Key, &u32)> = merged.iter().collect();
        v.sort_by_key(|x| std::cmp::Reverse(*x.1));
        let mut text = String::new();
        for ((l, x, d), n) in v {
            text += &format!("{l} {x} {} {n}\n", *d as u8);
        }
        std::fs::write(path, text)
    }
}

/// (hits, misses) of the expert cache
impl NemotronH {
    pub fn cache_stats(&self) -> (u64, u64) {
        self.cache.as_ref().map_or((0, 0), |c| {
            let c = c.lock().unwrap();
            (c.hits + c.hot_hits, c.misses)
        })
    }
}

/// The model's files opened for direct reads (O_DIRECT: past the page cache, which holds the same bytes the cache pools do
/// and slows the reads; many in flight, an NVMe's speed)
struct Direct {
    files: Vec<std::fs::File>,
}

impl Direct {
    fn open(f: &Gguf) -> Option<Direct> {
        use std::os::unix::fs::OpenOptionsExt;
        if std::env::var("NS_NH_DIRECT").is_ok_and(|v| v == "0") {
            return None;
        }
        let files: std::io::Result<Vec<_>> = f.paths.iter().map(|p| std::fs::OpenOptions::new().read(true).custom_flags(0x4000).open(p)).collect();
        files.ok().map(|files| Direct { files })
    }

    /// bytes [at, at + dst.len()) of tensor `t`
    fn read(&self, t: &Tensor, at: u64, dst: &mut [u8]) -> std::io::Result<()> {
        use std::os::unix::fs::FileExt;
        let abs = t.offset + at;
        let start = abs & !4095;
        let end = (abs + dst.len() as u64 + 4095) & !4095;
        let n = (end - start) as usize;
        let layout = std::alloc::Layout::from_size_align(n, 4096).unwrap();
        // SAFETY: a non-zero layout; freed below.
        let p = unsafe { std::alloc::alloc(layout) };
        if p.is_null() {
            return Err(std::io::Error::other("no memory for a direct read"));
        }
        // SAFETY: n bytes were allocated.
        let buf = unsafe { std::slice::from_raw_parts_mut(p, n) };
        // the file may end inside the last block: read what is there
        let mut got = 0usize;
        let r = loop {
            match self.files[t.shard].read_at(&mut buf[got..], start + got as u64) {
                Ok(0) => break Ok(()),
                Ok(k) => {
                    got += k;
                    if got >= n {
                        break Ok(());
                    }
                }
                Err(x) if x.kind() == std::io::ErrorKind::Interrupted => {}
                Err(x) => break Err(x),
            }
        };
        let skip = (abs - start) as usize;
        let ok = r.is_ok() && got >= skip + dst.len();
        if ok {
            dst.copy_from_slice(&buf[skip..skip + dst.len()]);
        }
        // SAFETY: allocated above with this layout.
        unsafe { std::alloc::dealloc(p, layout) };
        r?;
        if ok {
            Ok(())
        } else {
            Err(std::io::Error::other("a direct read came up short"))
        }
    }
}

fn plain(ty: GType) -> bool {
    matches!(ty, GType::F32 | GType::F16 | GType::BF16)
}

impl NemotronH {
    pub fn load(path: &std::path::Path, o: &Options, log: &mut dyn FnMut(String)) -> Result<NemotronH> {
        let t0 = Instant::now();
        nextcuda_core::use_kind("llm");
        let k = ffi::api()?;
        let f = Gguf::open(path).map_err(|x| e(x.0))?;
        if f.architecture() != "nemotron_h_moe" {
            return Err(e(format!("{}: {}, not nemotron_h_moe", path.display(), f.architecture())));
        }
        let g = Geometry::read(&f).map_err(e)?;
        let gpu = Gpu::open(o.gpu)?;
        let (total, free) = gpu.memory()?;
        let free = free.unwrap_or(total);
        // the plan: what is not an expert goes to VRAM; then layers' experts into what is left, pinned memory, disk
        let exp_names = |l: usize| [format!("blk.{l}.ffn_up_exps.weight"), format!("blk.{l}.ffn_down_exps.weight")];
        let exp_bytes: Vec<u64> = (0..g.layers).map(|l| if g.kinds[l] == Kind::Moe { exp_names(l).iter().map(|n| f.tensor(n).map_or(0, |t| t.bytes)).sum() } else { 0 }).collect();
        let nextn_prefix = format!("blk.{}.", g.layers);
        let dense: u64 = f.tensors.iter().filter(|t| !t.name.contains("_exps.") && !t.name.starts_with(&nextn_prefix)).map(|t| t.bytes).sum();
        // the embedding stays in the file (rows read as needed)
        let dense = dense - f.tensor("token_embd.weight").map_or(0, |t| t.bytes);
        let reserve = o.reserve_mib << 20;
        let mut vram_left = free.saturating_sub(dense + reserve);
        let mut homes = vec![0u8; g.layers]; // 0 VRAM, 2 the cache
        // with a profile the VRAM is the hot experts' (of any layer), not the first layers' whole
        let hot_budget = if o.profile.as_ref().is_some_and(|p| p.exists()) && std::env::var("NS_NH_HOT").map_or(true, |v| v != "0") { vram_left } else { 0 };
        for l in 0..g.layers {
            if exp_bytes[l] == 0 {
                continue;
            }
            if hot_budget == 0 && exp_bytes[l] <= vram_left {
                vram_left -= exp_bytes[l];
            } else {
                homes[l] = 2;
            }
        }
        let n = |h: u8| (0..g.layers).filter(|l| exp_bytes[*l] > 0 && homes[*l] == h).count();
        log(format!("nemotronh: {:.1} GiB dense + {:.1} GiB of experts in {} MoE layers; {:.1} GiB free on {}: experts of {} layers in VRAM, the other {} through the cache",
                    gib(dense), gib(exp_bytes.iter().sum()), g.count(Kind::Moe), gib(free), gpu.name, n(0), n(2)));

        let mut layers = Vec::with_capacity(g.layers);
        let loaded = std::cell::Cell::new(0u64);
        for l in 0..g.layers {
            let w = |s: &str| format!("blk.{l}.{s}.weight");
            let read = |name: &str| -> Result<Mat> { mat(&f, &gpu, k, name, &loaded) };
            let norm = read(&w("attn_norm"))?.buf;
            let mix = match g.kinds[l] {
                Kind::Mamba => {
                    let m = Mamba {
                        inp: read(&w("ssm_in"))?,
                        conv_w: read(&w("ssm_conv1d"))?.buf,
                        conv_b: read(&format!("blk.{l}.ssm_conv1d.bias"))?.buf,
                        dt_b: read(&format!("blk.{l}.ssm_dt.bias"))?.buf,
                        a: read(&format!("blk.{l}.ssm_a"))?.buf,
                        d: read(&format!("blk.{l}.ssm_d"))?.buf,
                        norm: read(&w("ssm_norm"))?.buf,
                        out: read(&w("ssm_out"))?,
                    };
                    if m.inp.rows != g.in_proj() || m.out.cols != g.inner {
                        return Err(e(format!("layer {l}: ssm_in of {} rows (want {}), ssm_out of {} inputs (want {})", m.inp.rows, g.in_proj(), m.out.cols, g.inner)));
                    }
                    Mix::Mamba(m)
                }
                Kind::Attn => Mix::Attn(Attn { q: read(&w("attn_q"))?, k: read(&w("attn_k"))?, v: read(&w("attn_v"))?, o: read(&w("attn_output"))? }),
                Kind::Moe => {
                    let router = read(&w("ffn_gate_inp"))?;
                    let bias_t = f.tensor(&format!("blk.{l}.exp_probs_b.bias")).ok_or_else(|| e(format!("layer {l}: no exp_probs_b.bias")))?;
                    let raw = f.read(bias_t).map_err(|x| e(x.0))?;
                    let bias: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                    let m = Moe {
                        layer: l,
                        router,
                        bias,
                        to_latent: read(&w("ffn_latent_down"))?,
                        from_latent: read(&w("ffn_latent_up"))?,
                        up: experts(&f, &gpu, &w("ffn_up_exps"), homes[l], &loaded)?,
                        down: experts(&f, &gpu, &w("ffn_down_exps"), homes[l], &loaded)?,
                        sh_up: read(&w("ffn_up_shexp"))?,
                        sh_down: read(&w("ffn_down_shexp"))?,
                    };
                    if m.router.rows != g.experts || m.bias.len() != g.experts || m.up.rows != g.n_ff_exp || m.up.cols != g.latent || m.down.rows != g.latent || m.down.cols != g.n_ff_exp {
                        return Err(e(format!("layer {l}: the experts' shapes ({} x {} up, {} x {} down, a router of {}) are not the metadata's", m.up.rows, m.up.cols, m.down.rows, m.down.cols, m.router.rows)));
                    }
                    for x in [&m.up, &m.down] {
                        // SAFETY: arithmetic only.
                        if unsafe { (k.mmvq_supported)(x.ty.code() as i32) } == 0 {
                            return Err(e(format!("layer {l}: experts in {} (the products take Q8_0, the K and IQ types, IQ4_NL, Q2_0)", x.ty.name())));
                        }
                    }
                    Mix::Moe(Box::new(m))
                }
            };
            layers.push(Layer { norm, mix });
            if l % 11 == 10 {
                log(format!("nemotronh: layer {} of {} ({:.1} GiB, {:.0} s)", l + 1, g.layers, gib(loaded.get()), t0.elapsed().as_secs_f64()));
            }
        }
        let output_norm = mat(&f, &gpu, k, "output_norm.weight", &loaded)?.buf;
        let output = mat(&f, &gpu, k, if f.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" }, &loaded)?;
        let embd = f.tensor("token_embd.weight").ok_or_else(|| e("no token_embd.weight"))?.clone();
        // the cache: a pool for each stride in use (up and down matrices of the layers outside VRAM), the pinned budget
        // shared by the bytes each stride's tensors hold; at least a pass's worth of slots each
        let mut by_stride: std::collections::BTreeMap<usize, u64> = Default::default();
        for ly in &layers {
            if let Mix::Moe(m) = &ly.mix {
                if matches!(m.up.home, Home::Cached) {
                    for x in [&m.up, &m.down] {
                        *by_stride.entry(x.stride).or_default() += (x.stride * g.experts) as u64;
                    }
                }
            }
        }
        let total: u64 = by_stride.values().sum();
        let budget = ((o.pinned_gib * (1u64 << 30) as f64) as u64).saturating_sub((o.scratch_mib as u64) << 20);
        let mut pools = Vec::new();
        for (&stride, &bytes) in &by_stride {
            let n = ((budget as f64 * bytes as f64 / total as f64) as usize / stride).clamp(g.experts.min(o.chunk.min(CHUNK) * g.used), (bytes as usize) / stride);
            let mut hb = HostBuf::new(&gpu, n * stride)?;
            let view = hb.device_view(0, n * stride)?;
            log(format!("nemotronh: expert cache: {n} slots of {:.2} MiB ({:.1} GiB of {:.1} GiB)", stride as f64 / 1048576.0, gib((n * stride) as u64), gib(bytes)));
            pools.push(Pool { _host: hb, view, stride, owner: vec![None; n], used: vec![0; n] });
        }
        let scratch = if o.scratch_mib > 0 && !pools.is_empty() {
            let mut hb = HostBuf::new(&gpu, o.scratch_mib << 20)?;
            let view = hb.device_view(0, o.scratch_mib << 20)?;
            Some((hb, view))
        } else {
            None
        };
        let hot_buf = if hot_budget > 0 && !pools.is_empty() { Some(DevBuf::new(&gpu, hot_budget as usize)?) } else { None };
        let cache = if pools.is_empty() { None } else { Some(std::sync::Mutex::new(ExpertCache { pools, map: Default::default(), clock: 0, hits: 0, misses: 0, wanted: Default::default(), hot: Default::default(), hot_buf, hot_hits: 0, scratch, scratch_used: 0, scratch_map: Default::default() })) };
        let direct = Direct::open(&f);
        let arena = Arena::new(&gpu, (96 << 20) + o.chunk * 700_000 + (o.chunk * g.used * 19_000))?;
        gpu.sync()?;
        let load_s = t0.elapsed().as_secs_f64();
        log(format!("nemotronh: loaded {:.1} GiB onto {} in {load_s:.0} s ({} layers: {} Mamba-2, {} MoE, {} attention)", gib(loaded.get()), gpu.name, g.layers, g.count(Kind::Mamba),
                    g.count(Kind::Moe), g.count(Kind::Attn)));
        let _ = o;
        Ok(NemotronH { g, file: f, gpu, k, layers, embd, output_norm, output, arena, cache, direct, chunk: o.chunk, times: Default::default(), load_s })
    }

    /// A new conversation of up to `max_ctx` tokens
    pub fn session(&self, max_ctx: usize) -> Result<Session> {
        let g = &self.g;
        if max_ctx as i64 > unsafe { (self.k.attn_max_keys)() } {
            return Err(e(format!("a context of {max_ctx}: the plain attention holds scores for {}", unsafe { (self.k.attn_max_keys)() })));
        }
        let mut layers = Vec::with_capacity(g.layers);
        for l in 0..g.layers {
            layers.push(match g.kinds[l] {
                Kind::Mamba => {
                    let (conv, s) = (DevBuf::f32(&self.gpu, (g.conv - 1) * g.conv_dim())?, DevBuf::f32(&self.gpu, g.inner * g.state)?);
                    conv.fill(0)?;
                    s.fill(0)?;
                    LState::Mamba { conv, s }
                }
                Kind::Attn => {
                    let b = g.heads_kv * max_ctx * g.head * 2;
                    LState::Attn { k: DevBuf::new(&self.gpu, b)?, v: DevBuf::new(&self.gpu, b)? }
                }
                Kind::Moe => LState::None,
            });
        }
        Ok(Session { pos: 0, max_ctx, layers })
    }

    fn raw(&self) -> *mut c_void {
        self.gpu.raw()
    }

    fn op(&self, what: &str, r: impl FnOnce() -> i32) -> Result<()> {
        ffi::check(r(), what)
    }

    fn quantize(&self, x: *const f32, cols: usize, t: usize) -> Result<Q8> {
        // SAFETY: arithmetic only.
        let buf = self.arena.bytes(unsafe { (self.k.q8_1_bytes)(cols as i64, t as i64) })?;
        // SAFETY: t rows of cols floats at x, the Q8_1 buffer sized for them.
        self.op("quantize", || unsafe { (self.k.quantize)(self.raw(), x, t as i64, cols as i64, buf.ptr()) })?;
        Ok(Q8 { buf })
    }

    /// y [t, rows] = w . x [t, cols]; `xq`: x's Q8_1 when the caller has it
    fn matmul(&self, w: &Mat, x: &DevBuf, xq: Option<&Q8>, t: usize, y: &DevBuf) -> Result<()> {
        let k = self.k;
        if w.ty == GType::F32 {
            // SAFETY: w float32 [rows, cols], x [t, cols], y [t, rows].
            return self.op("gemv", || unsafe { (k.gemv_f32)(self.raw(), w.buf.fp(), x.fp(), y.fp(), t as i64, w.rows as i64, w.cols as i64) });
        }
        let own;
        let q = match xq {
            Some(q) => q,
            None => {
                own = self.quantize(x.fp(), w.cols, t)?;
                &own
            }
        };
        // SAFETY: w's blocks, x's Q8_1, y [t, rows].
        self.op("mmvq", || unsafe { (k.mmvq)(self.raw(), w.ty.code() as i32, w.buf.ptr(), q.buf.ptr(), y.fp(), w.cols as i64, w.rows as i64, t as i64) })
    }

    fn rms(&self, x: &DevBuf, w: &DevBuf, y: &DevBuf, rows: usize, c: usize) -> Result<()> {
        // SAFETY: rows of c floats.
        self.op("rms", || unsafe { (self.k.rms_norm)(self.raw(), x.fp(), w.fp(), y.fp(), rows as i64, c as i64, self.g.eps) })
    }

    fn add(&self, y: &DevBuf, x: &DevBuf, n: usize) -> Result<()> {
        // SAFETY: n floats each.
        self.op("add", || unsafe { (self.k.add)(self.raw(), y.fp(), x.fp(), n as i64) })
    }

    /// The tokens' embeddings [t, n_embd] (their rows read from the file, expanded on the GPU)
    fn embed(&self, tokens: &[u32]) -> Result<DevBuf> {
        let g = &self.g;
        let row = self.embd.ty.bytes(g.n_embd as u64).ok_or_else(|| e("the embedding's row size"))? as usize;
        let mut raw = vec![0u8; tokens.len() * row];
        for (i, t) in tokens.iter().enumerate() {
            if *t as usize >= g.vocab {
                return Err(e(format!("token {t} past the vocabulary ({})", g.vocab)));
            }
            self.file.read_into(&self.embd, *t as u64 * row as u64, &mut raw[i * row..(i + 1) * row]).map_err(|x| e(x.0))?;
        }
        let src = self.arena.bytes(raw.len())?;
        src.write(0, &raw)?;
        let x = self.arena.f32(tokens.len() * g.n_embd)?;
        // SAFETY: the rows' blocks -> float32.
        self.op("embedding", || unsafe { (self.k.dequant_f32)(self.raw(), self.embd.ty.code() as i32, src.ptr(), (tokens.len() * g.n_embd) as i64, x.fp()) })?;
        Ok(x)
    }

    /// A Mamba-2 mixer on h [t, n_embd]: its output [t, n_embd]
    fn mamba(&self, m: &Mamba, conv: &DevBuf, s: &DevBuf, h: &DevBuf, t: usize) -> Result<DevBuf> {
        let (g, k) = (&self.g, self.k);
        let (ip, cd) = (g.in_proj(), g.conv_dim());
        let zx = self.arena.f32(t * ip)?;
        self.matmul(&m.inp, h, None, t, &zx)?;
        let co = self.arena.f32(t * cd)?;
        let y = self.arena.f32(t * g.inner)?;
        let yn = self.arena.f32(t * g.inner)?;
        let hp = g.inner / g.ssm_heads;
        // SAFETY: the buffers as named in nh.h; z at 0, x B C at inner, dt at inner + conv_dim of each zx row.
        unsafe {
            self.op("conv", || (k.conv)(self.raw(), zx.fp().add(g.inner), ip as i64, conv.fp(), m.conv_w.fp(), m.conv_b.fp(), co.fp(), t as i64, cd as i64, g.conv as i32))?;
            self.op("ssm scan", || (k.ssm)(self.raw(), co.fp(), cd as i64, zx.fp().add(g.inner + cd), ip as i64, m.dt_b.fp(), m.a.fp(), m.d.fp(), s.fp(), y.fp(), t as i64,
                                           g.ssm_heads as i64, hp as i64, g.state as i64, g.groups as i64))?;
            self.op("gated norm", || (k.gated_norm)(self.raw(), y.fp(), zx.fp(), ip as i64, m.norm.fp(), yn.fp(), t as i64, g.inner as i64, (g.inner / g.groups) as i64, g.eps))?;
        }
        let out = self.arena.f32(t * g.n_embd)?;
        self.matmul(&m.out, &yn, None, t, &out)?;
        Ok(out)
    }

    /// An attention layer on h [t, n_embd] (no positions): its output [t, n_embd]
    fn attention(&self, a: &Attn, kc: &DevBuf, vc: &DevBuf, cap: usize, h: &DevBuf, t: usize, p0: usize) -> Result<DevBuf> {
        let (g, k) = (&self.g, self.k);
        let (hq, hk, d) = (g.heads, g.heads_kv, g.head);
        let hq8 = self.quantize(h.fp(), g.n_embd, t)?;
        let q = self.arena.f32(t * hq * d)?;
        let kr = self.arena.f32(t * hk * d)?;
        let vr = self.arena.f32(t * hk * d)?;
        self.matmul(&a.q, h, Some(&hq8), t, &q)?;
        self.matmul(&a.k, h, Some(&hq8), t, &kr)?;
        self.matmul(&a.v, h, Some(&hq8), t, &vr)?;
        let att = self.arena.f32(t * hq * d)?;
        let (ti, hqi, hki, di, ci, pi) = (t as i64, hq as i64, hk as i64, d as i64, cap as i64, p0 as i64);
        // SAFETY: the rows and caches as named in nh.h.
        unsafe {
            self.op("kv store", || (k.kv_store)(self.raw(), kr.fp(), hki * di, vr.fp(), hki * di, ti, hki, di, ci, pi, kc.ptr(), vc.ptr()))?;
            self.op("attention", || (k.attn)(self.raw(), q.fp(), hqi * di, kc.ptr(), vc.ptr(), ti, hqi, hki, di, ci, pi, att.fp()))?;
        }
        let out = self.arena.f32(t * g.n_embd)?;
        self.matmul(&a.o, &att, None, t, &out)?;
        Ok(out)
    }

    /// The device address of expert `x`'s matrix `ex` (of layer `l`; up or down)
    fn expert_at(&self, ex: &Experts, l: usize, down: bool, x: usize) -> *const c_void {
        // SAFETY: inside the tensor / the cache (callers' counts).
        unsafe {
            match &ex.home {
                Home::Vram(b) => b.ptr().cast::<u8>().add(x * ex.stride).cast(),
                Home::Cached => {
                    let c = self.cache.as_ref().unwrap().lock().unwrap();
                    if let Some(&off) = c.hot.get(&(l, x, down)) {
                        return c.hot_buf.as_ref().unwrap().ptr().cast::<u8>().add(off).cast();
                    }
                    if let Some(&off) = c.scratch_map.get(&(l, x, down)) {
                        return c.scratch.as_ref().unwrap().1.ptr().cast::<u8>().add(off).cast();
                    }
                    let (p, sl) = c.map[&(l, x, down)];
                    c.pools[p].view.ptr().cast::<u8>().add(sl * ex.stride).cast()
                }
            }
        }
    }

    /// A latent-MoE layer on h [t, n_embd]: the routed experts in the latent, projected back, plus the shared expert
    fn moe(&self, m: &Moe, h: &DevBuf, t: usize) -> Result<DevBuf> {
        let (g, k) = (&self.g, self.k);
        let prof = std::env::var("NS_NH_PROFILE").is_ok_and(|v| v == "1");
        let mut lap_t = Instant::now();
        let mut lap = |name: &'static str| {
            if prof {
                let _ = self.gpu.sync();
                *self.times.lock().unwrap().parts.entry(name).or_default() += lap_t.elapsed().as_secs_f64();
                lap_t = Instant::now();
            }
        };
        let hq8 = self.quantize(h.fp(), g.n_embd, t)?;
        // the router (float32) and the choice on the host: sigmoid, the bias for the choice only, the chosen ones'
        // probabilities renormalized and scaled
        let logits = self.arena.f32(t * g.experts)?;
        self.matmul(&m.router, h, None, t, &logits)?;
        let lat = self.arena.f32(t * g.latent)?;
        self.matmul(&m.to_latent, h, Some(&hq8), t, &lat)?;
        lap("router+latent");
        let lg = logits.to_f32()?;
        let n_e = t * g.used;
        let mut ent: Vec<(usize, usize, f32)> = Vec::with_capacity(n_e);
        for r in 0..t {
            let row = &lg[r * g.experts..(r + 1) * g.experts];
            let p: Vec<f32> = row.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
            let mut idx: Vec<usize> = (0..g.experts).collect();
            // the choice by probability plus bias, highest first (the lower index on a tie)
            idx.sort_by(|a, b| (p[*b] + m.bias[*b]).total_cmp(&(p[*a] + m.bias[*a])).then(a.cmp(b)));
            idx.truncate(g.used);
            let mut w: Vec<f32> = idx.iter().map(|i| p[*i]).collect();
            if g.w_norm {
                let s: f32 = w.iter().sum();
                if s > 0.0 {
                    w.iter_mut().for_each(|x| *x /= s);
                }
            }
            for (i, x) in idx.into_iter().zip(w) {
                ent.push((r, i, x * g.w_scale));
            }
        }
        // the experts' bytes where the kernels can read them: those of a cached layer into slots (the least recently
        // used given up), the misses read from the file on a few threads
        let tf = Instant::now();
        if matches!(m.up.home, Home::Cached) {
            let mut c = self.cache.as_ref().unwrap().lock().unwrap();
            c.clock += 1;
            let clock = c.clock;
            c.scratch_map.clear();
            c.scratch_used = 0;
            let bypass = t > SMALL && c.scratch.is_some();
            // (matrix is down, expert, the pool's host base + slot)
            let mut jobs: Vec<(bool, usize, usize)> = Vec::new();
            let mut seen = std::collections::HashSet::new();
            for (_, x, _) in &ent {
                if !seen.insert(*x) {
                    continue;
                }
                for (down, ex) in [(false, &m.up), (true, &m.down)] {
                    let key = (m.layer, *x, down);
                    *c.wanted.entry(key).or_default() += 1;
                    if c.hot.contains_key(&key) {
                        c.hot_hits += 1;
                        continue;
                    }
                    if let Some(&(p, sl)) = c.map.get(&key) {
                        c.pools[p].used[sl] = clock;
                        c.hits += 1;
                        continue;
                    }
                    if bypass && c.scratch_used + ex.stride <= c.scratch.as_ref().unwrap().0.len {
                        let off = c.scratch_used;
                        c.scratch_used += (ex.stride + 255) & !255;
                        c.scratch_map.insert(key, off);
                        c.misses += 1;
                        jobs.push((down, *x, c.scratch.as_ref().unwrap().0.as_slice().as_ptr() as usize + off));
                        continue;
                    }
                    let p = c.pools.iter().position(|q| q.stride == ex.stride).unwrap();
                    let sl = (0..c.pools[p].used.len()).min_by_key(|i| c.pools[p].used[*i]).unwrap();
                    if let Some(old) = c.pools[p].owner[sl].take() {
                        c.map.remove(&old);
                    }
                    c.pools[p].owner[sl] = Some(key);
                    c.pools[p].used[sl] = clock;
                    c.map.insert(key, (p, sl));
                    c.misses += 1;
                    jobs.push((down, *x, c.pools[p]._host.as_slice().as_ptr() as usize + sl * ex.stride));
                }
            }
            if !jobs.is_empty() {
                let tio = Instant::now();
                let nbytes: usize = jobs.iter().map(|j| if j.0 { m.down.stride } else { m.up.stride }).sum();
                let (tu, td) = (self.file.tensor(&m.up.name).unwrap(), self.file.tensor(&m.down.name).unwrap());
                let (us, ds) = (m.up.stride, m.down.stride);
                let nt = jobs.len().clamp(1, std::env::var("NS_NH_IO_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(16));
                let err = std::sync::Mutex::new(None);
                std::thread::scope(|sc| {
                    for part in jobs.chunks(jobs.len().div_ceil(nt)) {
                        let (file, err, direct) = (&self.file, &err, &self.direct);
                        sc.spawn(move || {
                            for &(down, x, at) in part {
                                let (t, st) = if down { (td, ds) } else { (tu, us) };
                                // SAFETY: the slot is inside its pool; each job owns its slot while the pass runs.
                                let dst = unsafe { std::slice::from_raw_parts_mut(at as *mut u8, st) };
                                let r = match direct {
                                    Some(d) => d.read(t, (x * st) as u64, dst).map_err(|y| y.to_string()),
                                    None => file.read_into(t, (x * st) as u64, dst).map_err(|y| y.0),
                                };
                                if let Err(y) = r {
                                    *err.lock().unwrap() = Some(y);
                                }
                            }
                        });
                    }
                });
                if let Some(y) = err.into_inner().unwrap() {
                    return Err(e(y));
                }
                let mut tm = self.times.lock().unwrap();
                tm.io_s += tio.elapsed().as_secs_f64();
                tm.io_bytes += nbytes as u64;
            }
        }
        lap("route+fetch");
        self.times.lock().unwrap().moe_fetch += tf.elapsed().as_secs_f64();
        let latq = self.quantize(lat.fp(), g.latent, t)?;
        // SAFETY: arithmetic only.
        let mo = if t <= SMALL { {
        let u = self.arena.f32(n_e * g.n_ff_exp)?;
        // the experts' addresses and the activation rows, in one upload: up's pointers | down's | the tokens' rows | 0..n_e
        let mut ptrs: Vec<u8> = Vec::with_capacity(n_e * 24);
        for (_, x, _) in &ent {
            ptrs.extend_from_slice(&(self.expert_at(&m.up, m.layer, false, *x) as u64).to_le_bytes());
        }
        for (_, x, _) in &ent {
            ptrs.extend_from_slice(&(self.expert_at(&m.down, m.layer, true, *x) as u64).to_le_bytes());
        }
        for (r, _, _) in &ent {
            ptrs.extend_from_slice(&(*r as i32).to_le_bytes());
        }
        for j in 0..n_e {
            ptrs.extend_from_slice(&(j as i32).to_le_bytes());
        }
        let pb = self.arena.bytes(ptrs.len())?;
        pb.write(0, &ptrs)?;
        let (up_p, down_p) = (pb.ptr().cast_const(), unsafe { pb.ptr().cast::<u8>().add(n_e * 8).cast_const() });
        let (rows_p, seq_p) = unsafe { (pb.ptr().cast::<u8>().add(n_e * 16).cast::<i32>().cast_const(), pb.ptr().cast::<u8>().add(n_e * 16 + n_e * 4).cast::<i32>().cast_const()) };
        // SAFETY: n_e experts' up [n_ff, latent] in their blocks, the tokens' latent Q8_1 rows, u [n_e][n_ff].
        self.op("expert up", || unsafe {
            (k.mmvq_ptrs)(self.raw(), m.up.ty.code() as i32, up_p.cast(), rows_p, latq.buf.ptr(), u.fp(), g.latent as i64, g.n_ff_exp as i64, n_e as i64)
        })?;
        lap("expert up");
        // SAFETY: n floats.
        self.op("relu2", || unsafe { (k.relu2)(self.raw(), u.fp(), (n_e * g.n_ff_exp) as i64) })?;
        let uq8 = self.quantize(u.fp(), g.n_ff_exp, n_e)?;
        let dn = self.arena.f32(n_e * g.latent)?;
        // SAFETY: the experts' down [latent, n_ff], entry j's Q8_1 row, dn [n_e][latent].
        self.op("expert down", || unsafe {
            (k.mmvq_ptrs)(self.raw(), m.down.ty.code() as i32, down_p.cast(), seq_p, uq8.buf.ptr(), dn.fp(), g.n_ff_exp as i64, g.latent as i64, n_e as i64)
        })?;
        lap("expert down");
        // the weighted sum per token (entries in token order)
        let mut tp = vec![0i32; t + 1];
        for (r, _, _) in &ent {
            tp[r + 1] += 1;
        }
        for r in 0..t {
            tp[r + 1] += tp[r];
        }
        let ws: Vec<f32> = ent.iter().map(|x| x.2).collect();
        let (tpb, wb) = (self.arena.bytes((t + 1) * 4)?, self.arena.bytes(n_e * 4)?);
        tpb.write(0, &tp.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
        wb.write(0, &ws.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
        let mo = self.arena.f32(t * g.latent)?;
        // SAFETY: mo [t, latent], dn [entries, latent], the token pointers and weights as named in nh.h.
        self.op("moe combine", || unsafe { (k.moe_combine)(self.raw(), mo.fp(), dn.fp(), tpb.ptr().cast(), wb.fp(), t as i64, g.latent as i64) })?;
        mo
        } } else {
            // a prompt pass: the entries by expert; an expert's rows (the tokens that chose it) in one product
            let mut order: Vec<usize> = (0..n_e).collect();
            order.sort_by_key(|&j| (ent[j].1, ent[j].0));
            let (mut rows, mut ws, mut place) = (vec![0i32; n_e], vec![0f32; n_e], vec![0i32; n_e]);
            let mut groups: Vec<(usize, usize, usize)> = Vec::new(); // (expert, first, count)
            for (gi, &j) in order.iter().enumerate() {
                rows[gi] = ent[j].0 as i32;
                ws[gi] = ent[j].2;
                place[j] = gi as i32;
                match groups.last_mut() {
                    Some(l) if l.0 == ent[j].1 => l.2 += 1,
                    _ => groups.push((ent[j].1, gi, 1)),
                }
            }
            // each token's entries' places, in token order
            let mut tp = vec![0i32; t + 1];
            for (r, _, _) in &ent {
                tp[r + 1] += 1;
            }
            for r in 0..t {
                tp[r + 1] += tp[r];
            }
            let mut ids = vec![0i32; n_e];
            let mut fill = tp.clone();
            for (j, (r, _, _)) in ent.iter().enumerate() {
                ids[fill[*r] as usize] = place[j];
                fill[*r] += 1;
            }
            let bytes = |v: &[i32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
            let (rb, tb, ib, wb) = (self.arena.bytes(n_e * 4)?, self.arena.bytes((t + 1) * 4)?, self.arena.bytes(n_e * 4)?, self.arena.bytes(n_e * 4)?);
            rb.write(0, &bytes(&rows))?;
            tb.write(0, &bytes(&tp))?;
            ib.write(0, &bytes(&ids))?;
            wb.write(0, &ws.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
            // SAFETY: arithmetic only.
            let (lq, uq) = unsafe { ((k.q8_1_bytes)(g.latent as i64, 1), (k.q8_1_bytes)(g.n_ff_exp as i64, 1)) };
            let xg = self.arena.bytes(n_e * lq)?;
            // SAFETY: n_e rows of the latent's Q8_1 gathered in expert order.
            self.op("gather", || unsafe { (k.gather_rows)(self.raw(), latq.buf.ptr(), rb.ptr().cast(), xg.ptr(), lq as i64, n_e as i64) })?;
            let u = self.arena.f32(n_e * g.n_ff_exp)?;
            for &(x, first, n) in &groups {
                let w = self.expert_at(&m.up, m.layer, false, x);
                // SAFETY: expert x's up against its n rows of xg, into rows of u.
                self.op("expert up", || unsafe {
                    (k.mmvq)(self.raw(), m.up.ty.code() as i32, w, xg.ptr().cast::<u8>().add(first * lq).cast(), u.fp().add(first * g.n_ff_exp), g.latent as i64, g.n_ff_exp as i64, n as i64)
                })?;
            }
            lap("expert up");
            // SAFETY: n floats.
            self.op("relu2", || unsafe { (k.relu2)(self.raw(), u.fp(), (n_e * g.n_ff_exp) as i64) })?;
            let uq8 = self.quantize(u.fp(), g.n_ff_exp, n_e)?;
            let dn = self.arena.f32(n_e * g.latent)?;
            for &(x, first, n) in &groups {
                let w = self.expert_at(&m.down, m.layer, true, x);
                // SAFETY: expert x's down against its n rows of uq8, into rows of dn.
                self.op("expert down", || unsafe {
                    (k.mmvq)(self.raw(), m.down.ty.code() as i32, w, uq8.buf.ptr().cast::<u8>().add(first * uq).cast(), dn.fp().add(first * g.latent), g.n_ff_exp as i64, g.latent as i64, n as i64)
                })?;
            }
            lap("expert down");
            let mo = self.arena.f32(t * g.latent)?;
            // SAFETY: mo [t, latent], dn [entries, latent], the token pointers, places and weights as named in nh.h.
            self.op("moe combine", || unsafe { (k.moe_combine_ids)(self.raw(), mo.fp(), dn.fp(), tb.ptr().cast(), ib.ptr().cast(), wb.fp(), t as i64, g.latent as i64) })?;
            mo
        };
        let out = self.arena.f32(t * g.n_embd)?;
        self.matmul(&m.from_latent, &mo, None, t, &out)?;
        lap("combine+latent up");
        // the shared expert on the full width
        let su = self.arena.f32(t * g.n_ff_shared)?;
        self.matmul(&m.sh_up, h, Some(&hq8), t, &su)?;
        // SAFETY: n floats.
        self.op("relu2", || unsafe { (k.relu2)(self.raw(), su.fp(), (t * g.n_ff_shared) as i64) })?;
        let sd = self.arena.f32(t * g.n_embd)?;
        self.matmul(&m.sh_down, &su, None, t, &sd)?;
        self.add(&out, &sd, t * g.n_embd)?;
        lap("shared expert");
        Ok(out)
    }

    /// One pass of at most DEC rows: the last `n_out` rows' logits (0: none - a prompt chunk before the last)
    pub fn pass(&self, s: &mut Session, tokens: &[u32], n_out: usize) -> Result<Vec<Vec<f32>>> {
        let t = tokens.len();
        if t == 0 || t > self.chunk {
            return Err(e(format!("a pass of {t} rows; it takes 1 to {}", self.chunk)));
        }
        if s.pos + t > s.max_ctx {
            return Err(e(format!("the context is full ({} of {})", s.pos + t, s.max_ctx)));
        }
        let g = &self.g;
        let p0 = s.pos;
        self.arena.reset();
        let x = self.embed(tokens)?;
        let h = self.arena.f32(t * g.n_embd)?;
        let prof = std::env::var("NS_NH_PROFILE").is_ok_and(|v| v == "1");
        for l in 0..g.layers {
            let (ly, st) = (&self.layers[l], &s.layers[l]);
            let mark = self.arena.mark();
            let t1 = Instant::now();
            self.rms(&x, &ly.norm, &h, t, g.n_embd)?;
            let out = match (&ly.mix, st) {
                (Mix::Mamba(m), LState::Mamba { conv, s: ss }) => self.mamba(m, conv, ss, &h, t)?,
                (Mix::Attn(a), LState::Attn { k, v }) => self.attention(a, k, v, s.max_ctx, &h, t, p0)?,
                (Mix::Moe(m), LState::None) => self.moe(m, &h, t)?,
                _ => return Err(e(format!("layer {l}: its state and its weights disagree"))),
            };
            self.add(&x, &out, t * g.n_embd)?;
            if prof {
                self.gpu.sync()?;
                let d = t1.elapsed().as_secs_f64();
                let mut tm = self.times.lock().unwrap();
                match g.kinds[l] {
                    Kind::Mamba => tm.mamba += d,
                    Kind::Attn => tm.attn += d,
                    Kind::Moe => tm.moe += d,
                }
            }
            self.arena.rewind(mark);
        }
        self.times.lock().unwrap().passes += 1;
        s.pos += t;
        if n_out == 0 {
            self.gpu.sync()?;
            return Ok(Vec::new());
        }
        let n = n_out.clamp(1, t);
        let th = Instant::now();
        // the last n rows through the output norm and the head
        let xl = x.view((t - n) * g.n_embd * 4, n * g.n_embd * 4)?;
        let hn = self.arena.f32(n * g.n_embd)?;
        self.rms(&xl, &self.output_norm, &hn, n, g.n_embd)?;
        let lg = self.arena.f32(n * g.vocab)?;
        self.matmul(&self.output, &hn, None, n, &lg)?;
        let all = lg.to_f32()?;
        self.times.lock().unwrap().head += th.elapsed().as_secs_f64();
        Ok(all.chunks_exact(g.vocab).map(|c| c.to_vec()).collect())
    }

    /// Rows of any length: the passes of DEC rows, the logits of the last `n_out`
    pub fn forward(&self, s: &mut Session, tokens: &[u32], n_out: usize) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::new();
        let chunks: Vec<&[u32]> = tokens.chunks(self.chunk).collect();
        let last = chunks.len() - 1;
        for (i, c) in chunks.iter().enumerate() {
            out = self.pass(s, c, if i == last { n_out } else { 0 })?;
        }
        Ok(out)
    }

    pub fn tokenizer(&self) -> std::result::Result<nextcuda_tok::Tokenizer, String> {
        nextcuda_tok::Tokenizer::from_gguf(&self.file)
    }

    pub fn gpu(&self) -> &Arc<Gpu> {
        &self.gpu
    }
}

/// A matrix (or vector) onto the GPU in its blocks; float32 / half tensors expanded to float32
fn mat(f: &Gguf, gpu: &Arc<Gpu>, k: &ffi::Api, name: &str, loaded: &std::cell::Cell<u64>) -> Result<Mat> {
    let t = f.tensor(name).ok_or_else(|| e(format!("no {name}")))?;
    // outermost first: [rows (outputs), cols (inputs)]; a vector is one row
    let (rows, cols) = match t.shape.len() {
        1 => (1, t.shape[0] as usize),
        _ => (t.shape[0] as usize, t.shape[1..].iter().product::<u64>() as usize),
    };
    let raw = f.read(t).map_err(|x| e(x.0))?;
    loaded.set(loaded.get() + raw.len() as u64);
    if plain(t.ty) {
        let src = DevBuf::new(gpu, raw.len())?;
        src.write(0, &raw)?;
        let out = DevBuf::f32(gpu, rows * cols)?;
        // SAFETY: src holds the tensor, out its elements as float32.
        ffi::check(unsafe { (k.dequant_f32)(gpu.raw(), t.ty.code() as i32, src.ptr(), (rows * cols) as i64, out.fp()) }, name)?;
        gpu.sync()?;
        return Ok(Mat { buf: out, ty: GType::F32, rows, cols });
    }
    // SAFETY: arithmetic only.
    if unsafe { (k.mmvq_supported)(t.ty.code() as i32) } == 0 {
        return Err(e(format!("{name}: {} is not a type the products take", t.ty.name())));
    }
    let buf = DevBuf::new(gpu, raw.len())?;
    buf.write(0, &raw)?;
    Ok(Mat { buf, ty: t.ty, rows, cols })
}

/// A layer's stacked experts [experts, rows, cols] where `home` says (0 VRAM, 2 the file through the cache)
fn experts(f: &Gguf, gpu: &Arc<Gpu>, name: &str, home: u8, loaded: &std::cell::Cell<u64>) -> Result<Experts> {
    let t = f.tensor(name).ok_or_else(|| e(format!("no {name}")))?;
    if t.shape.len() != 3 {
        return Err(e(format!("{name}: [experts, rows, cols], not {:?}", t.shape)));
    }
    let (n, rows, cols) = (t.shape[0] as usize, t.shape[1] as usize, t.shape[2] as usize);
    let stride = t.ty.bytes((rows * cols) as u64).ok_or_else(|| e(format!("{name}: rows of {cols} are not whole {} blocks", t.ty.name())))? as usize;
    if stride * n != t.bytes as usize {
        return Err(e(format!("{name}: {} bytes, not {n} experts of {stride}", t.bytes)));
    }
    let h = match home {
        2 => Home::Cached,
        _ => {
            // in slices: no host copy of the whole tensor
            let buf = DevBuf::new(gpu, t.bytes as usize)?;
            let mut chunk = vec![0u8; 64 << 20];
            let mut at = 0u64;
            while at < t.bytes {
                let n = chunk.len().min((t.bytes - at) as usize);
                f.read_into(t, at, &mut chunk[..n]).map_err(|x| e(x.0))?;
                buf.write(at as usize, &chunk[..n])?;
                at += n as u64;
            }
            loaded.set(loaded.get() + t.bytes);
            Home::Vram(buf)
        }
    };
    Ok(Experts { home: h, name: name.to_string(), ty: t.ty, rows, cols, stride })
}

impl NemotronH {
    /// Which tier each MoE layer's experts are in (for the log)
    pub fn tiers(&self) -> Vec<&'static str> {
        self.layers.iter().filter_map(|l| match &l.mix { Mix::Moe(m) => Some(m.up.tier()), _ => None }).collect()
    }
}
