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

pub mod cpu;
pub mod engine;
pub mod ffi;

use std::ffi::c_void;
use std::sync::Arc;
use std::time::Instant;

use nextcuda_core::{Arena, DevBuf, Error, Gpu, HostBuf, Result};
use nextcuda_gguf::{GType, Gguf, Tensor};

/// Rows a pass takes
pub const DEC: usize = 8;
/// Rows a pass of the grouped MoE path takes (prompts); passes up to SMALL rows run each expert entry on its own
pub const CHUNK: usize = 2048;
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
    /// NS_NH_PREDICT=1: routing of the next MoE layer guessed from the residual stream: the experts guessed right / chosen
    pub pred_wait: f64,
    /// matrices served from the guessed experts' buffer / copied at the layer
    pub pre_hit: u64,
    pub stage_copy: u64,
    pub pred_right: u64,
    pub pred_total: u64,
    /// decode's experts multiplied on the CPU, and the seconds it took (the GPU working meanwhile)
    pub cpu_experts: u64,
    pub cpu_s: f64,
    /// seconds reading the borrowed hot experts back after long prompts
    pub give_back_s: f64,
    /// decode's chosen experts dropped by the tail skip
    pub skipped: u64,
    /// NS_NH_PROFILE=1: a MoE layer's parts (seconds, each waited for)
    pub parts: std::collections::BTreeMap<&'static str, f64>,
}

type CopyBatch = unsafe extern "C" fn(*mut c_void, *const *mut c_void, *const *const c_void, *const usize, usize, *const i64, i32, i32, *mut i64) -> i32;

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
    /// decode passes copy the experts they use from pinned memory into VRAM before the products
    pub stage: bool,
}

impl Default for Options {
    fn default() -> Options {
        let env = |k: &str| std::env::var(k).ok();
        Options {
            gpu: env("NS_NH_GPU").and_then(|v| v.parse().ok()).unwrap_or(0),
            ctx: env("NS_NH_CTX").and_then(|v| v.parse().ok()).unwrap_or(4096),
            pinned_gib: env("NS_NH_PINNED_GIB").and_then(|v| v.parse().ok()).unwrap_or(14.0),
            reserve_mib: env("NS_NH_RESERVE_MIB").and_then(|v| v.parse().ok()).unwrap_or(700),
            profile: None,
            stage: env("NS_NH_STAGE").map_or(true, |v| v != "0"),
            scratch_mib: env("NS_NH_SCRATCH_MIB").and_then(|v| v.parse().ok()).unwrap_or(1024),
            chunk: env("NS_NH_CHUNK").and_then(|v| v.parse().ok()).unwrap_or(2048).clamp(1, CHUNK),
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
    /// the arena's own buffer (decode, short prompts), the rows it takes, and what a long prompt borrows from the hot set's tail
    arena_home: DevBuf,
    resident_rows: usize,
    borrow_bytes: usize,
    /// the hot matrices the borrowed tail held (key, stride, offset), read back after the prompt
    lent: std::sync::Mutex<Vec<(Key, usize, usize)>>,
    /// the experts of the layers not in VRAM: slots of pinned host memory, the recently used kept (LRU)
    cache: Option<std::sync::Mutex<ExpertCache>>,
    io: Io,
    /// reads started for the next MoE layer from a guess at its routing, by layer
    inflight: std::sync::Mutex<std::collections::HashMap<usize, Vec<(usize, Arc<Pending>)>>>,
    chunk: usize,
    predict: bool,
    /// decode multiplies the experts it has while the misses are read, the missed ones in a second batch (NS_NH_OVERLAP=0: all after the reads)
    overlap: bool,
    /// NS_NH_DYN_VRAM=1: decode's copies up go into the VRAM tier's slots (LRU) instead of a per-pass buffer. Off: measured, 48K copies
    /// in gave 7 % more VRAM hits and twice the misses (what it gives up is not in pinned memory), decode 9.2 -> 7.0 tok/s
    dyn_vram: bool,
    /// copies toward VRAM for the first NS_NH_PRE_K (8) of a guess's experts (by score) only: a wrong guess's copy is the link's time
    /// lost (all 22: 2,000-token decode 8.5 tok/s, the top 8: 8.9)
    pre_k: usize,
    /// decode drops a chosen expert ranked after the first NS_NH_TAIL_SKIP (8; 0: off) whose bytes are not where
    /// NS_NH_TAIL_TIER says (pinned, the default: in neither VRAM nor pinned memory, a disk read; vram: not in VRAM), its
    /// weight not given to the others (Strata's route tail skip). 22 GiB, 384 tokens: decode 7.2 -> 9.5 and 5.9 -> 7.8 tok/s
    /// at KL 0.0037 (top token the same 98.8 %); from rank 4 KL 0.0048 and no faster; vram from 12: KL 0.017
    tail_skip: usize,
    tail_vram: bool,
    /// decode's staging copies on copy lane 1 (NS_NH_COPY_LANE=0: on the GPU's stream), so the experts already in VRAM are
    /// multiplied while the others are copied up (three batches: in VRAM, staged, read from disk)
    copy_lane: bool,
    /// the core library's ns_stream_copy_batch, if it has it (NS_NH_COPY_BATCH=0: not used): a layer's staging copies in one
    /// call instead of one each
    copy_batch: Option<CopyBatch>,
    /// small uploads (a batch's expert pointers, the combine's weights, token rows) through a pinned ring, queued without a
    /// wait (DevBuf::write syncs the stream after each: the host could not run ahead of the GPU); a wrap syncs once first
    ring: std::sync::Mutex<(HostBuf, usize)>,
    pred: std::sync::Mutex<std::collections::HashMap<usize, Vec<usize>>>,
    /// MoE layers ahead whose misses a guess reads, and how many of each guess's top experts
    pred_ahead: usize,
    pred_k: usize,
    /// the largest dense matrix as half (the prompt passes' GEMMs), and the rows from which they are used
    w16: Option<DevBuf>,
    dense_gemm_min: usize,
    pub times: std::sync::Mutex<Times>,
    /// decode's CPU share (cpu.rs): ggml's products and the workers, and the share of a layer's pinned experts it takes
    cpu: Option<(&'static cpu::Ggml, cpu::Pool)>,
    cpu_frac: f64,
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
    /// (used, slot), least recently used first: a victim without a scan of every slot
    lru: std::collections::BTreeSet<(u64, usize)>,
}

impl Pool {
    fn touch(&mut self, sl: usize, clock: u64) {
        self.lru.remove(&(self.used[sl], sl));
        self.used[sl] = clock;
        self.lru.insert((clock, sl));
    }
}

/// VRAM slots of one matrix size in the hot buffer (from `base`): who holds each, and when it was last wanted
struct VSlots {
    stride: usize,
    base: usize,
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
    /// the hot buffer as slots of each matrix size, and where each hot matrix is in them
    vslots: Vec<VSlots>,
    hot_at: std::collections::HashMap<Key, (usize, usize)>,
    /// matrices decode copied up into a VRAM slot (another given up)
    pub vram_ins: u64,
    /// pinned room for the misses of a prompt pass, used and dropped (a sweep over every layer's experts would only push
    /// out what the profile put in the pools)
    scratch: Option<(HostBuf, DevBuf)>,
    scratch_used: usize,
    scratch_map: std::collections::HashMap<Key, usize>,
    /// VRAM room for a decode pass's experts copied from pinned memory (a DMA at the link's speed, then products at VRAM's:
    /// better than the kernels reading host memory a few words at a time)
    stage: Option<DevBuf>,
    stage_used: usize,
    stage_map: std::collections::HashMap<Key, usize>,
    /// two VRAM buffers the next MoE layer's guessed experts are copied into (on the copy lane) while this layer computes
    pre: Vec<DevBuf>,
    pre_used: [usize; 2],
    pre_map: std::collections::HashMap<Key, (usize, usize)>,
    /// by layer: the copy lane's ticket after its copies, and the buffer they went to
    pre_ticket: std::collections::HashMap<usize, (i64, usize)>,
    /// by buffer: a ticket after the kernels that last read it (the next copies into it wait for that)
    pre_done: [i64; 2],
    pre_ring: usize,
    /// a ticket after the kernels that last read the staging buffer (copies into it on the copy lane wait for that)
    stage_done: i64,
    /// matrices whose slot a guess's read is still filling (not copied toward VRAM, nor their slot given up, until it is done)
    reading: std::collections::HashSet<Key>,
    /// a guess's reads of experts their layer did not choose, left to finish meanwhile: (layer, expert, the read)
    stale: Vec<(usize, usize, Arc<Pending>)>,
}

impl ExpertCache {
    /// Expert x of layer l's read is done: its slots usable (or, if it failed, empty again)
    fn settle(&mut self, l: usize, x: usize, failed: bool) {
        for down in [false, true] {
            let key = (l, x, down);
            if self.reading.remove(&key) && failed {
                if let Some((p, sl)) = self.map.remove(&key) {
                    self.pools[p].owner[sl] = None;
                    self.pools[p].touch(sl, 0);
                }
            }
        }
    }

    /// The least recently used slot of pool p not being read into
    fn victim(&self, p: usize) -> Option<usize> {
        let q = &self.pools[p];
        q.lru.iter().map(|x| x.1).find(|i| !q.owner[*i].is_some_and(|k| self.reading.contains(&k)))
    }
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
            // the most wanted until the buffer is full, then laid out as slots of each matrix size (decode swaps the experts it
            // copies up into them, the least recently used given up: the VRAM tier follows the conversation)
            let mut at = 0usize;
            let mut chosen: Vec<(Key, usize)> = Vec::new();
            for (k, st) in all {
                if at + st > cap {
                    continue;
                }
                chosen.push((k, st));
                at += st;
            }
            let mut sizes: std::collections::BTreeMap<usize, usize> = Default::default();
            for (_, st) in &chosen {
                *sizes.entry(*st).or_default() += 1;
            }
            let mut base = 0usize;
            let mut vi_of: std::collections::HashMap<usize, usize> = Default::default();
            for (&st, &n) in &sizes {
                vi_of.insert(st, c.vslots.len());
                c.vslots.push(VSlots { stride: st, base, owner: vec![None; n], used: vec![0; n] });
                base += n * st;
            }
            let mut next = vec![0usize; c.vslots.len()];
            let mut rank_at = 0usize;
            for (k, st) in chosen {
                let vi = vi_of[&st];
                let sl = next[vi];
                next[vi] += 1;
                // the LRU tier's slots by size; else in rank order (the least wanted last: what a long prompt borrows)
                let off = if self.dyn_vram {
                    c.vslots[vi].owner[sl] = Some(k);
                    c.hot_at.insert(k, (vi, sl));
                    c.vslots[vi].base + sl * st
                } else {
                    rank_at += st;
                    rank_at - st
                };
                hot_jobs.push((k, st, off));
            }
            hot_jobs.sort_by_key(|j| j.2);
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
        let reqs: Vec<IoReq> = jobs.iter().filter_map(|&(l, x, p, down, sl)| {
            let Mix::Moe(m) = &self.layers[l].mix else { return None };
            let ex = if down { &m.down } else { &m.up };
            Some(self.req(ex, x, bases[p] + sl * strides[p]))
        }).collect();
        self.io.run(&reqs).map_err(e)?;
        // the hot matrices into VRAM, in chunks through a host buffer
        if !hot_jobs.is_empty() {
            self.fill_hot(&hot_jobs)?;
            filled += hot_jobs.len();
        }
        log(format!("nemotronh: the cache preloaded with {filled} matrices ({}) in {:.1} s", if have_profile { "by the saved profile" } else { "in file order" }, t0.elapsed().as_secs_f64()));
        Ok(filled)
    }

    /// While MoE layer `l` runs (decode): a guess at the next MoE layer's routing - its router on the residual stream as it is
    /// before `l` (58 % of the experts it will choose, measured) - and the reads of what that layer will need and the cache
    /// lacks started now, to be waited for when it runs.
    /// The guesses' routers launched (their logits in the returned buffer, read with the layer's own router: one wait on the GPU a
    /// layer, not two)
    fn predict_launch(&self, l: usize, x: &DevBuf) -> Result<Option<(Vec<usize>, DevBuf)>> {
        let g = &self.g;
        // the next MoE layers (NS_NH_PRED_AHEAD, 1 by default: further guesses read wrong experts that push right ones out of the pools, measured slower): each one's router on the residual stream as it is now; the
        // reads of the misses among each guess's top NS_NH_PRED_K (more than the 22 it will choose: a wrong guess costs a read the
        // disk has time for, a right one saves a wait), the copies toward VRAM for the first only
        let targets: Vec<usize> = (l + 1..g.layers).filter(|i| g.kinds[*i] == Kind::Moe).take(self.pred_ahead).collect();
        let targets: Vec<usize> = targets.into_iter().filter(|t| matches!(&self.layers[*t].mix, Mix::Moe(m) if matches!(m.up.home, Home::Cached))).collect();
        if targets.is_empty() {
            return Ok(None);
        }
        let hh = self.arena.f32(g.n_embd)?;
        let lg = self.arena.f32(g.experts * targets.len())?;
        for (i, &lt) in targets.iter().enumerate() {
            let Mix::Moe(mt) = &self.layers[lt].mix else { continue };
            self.rms(x, &self.layers[lt].norm, &hh, 1, g.n_embd)?;
            self.matmul(&mt.router, &hh, None, 1, &lg.view(i * g.experts * 4, g.experts * 4)?)?;
        }
        Ok(Some((targets, lg)))
    }

    /// The guesses' choices from their logits: the reads of the misses started, the first guess's pinned experts copied up
    fn predict_finish(&self, targets: &[usize], all: &[f32]) -> Result<()> {
        let g = &self.g;
        let mut reqs: Vec<(usize, usize, IoReq)> = Vec::new();
        let mut c = self.cache.as_ref().unwrap().lock().unwrap();
        let clock = c.clock;
        for (i, &lt) in targets.iter().enumerate() {
            let Mix::Moe(mt) = &self.layers[lt].mix else { continue };
            let v = &all[i * g.experts..(i + 1) * g.experts];
            let mut idx: Vec<usize> = (0..g.experts).collect();
            let sc = |i: usize| 1.0 / (1.0 + (-v[i]).exp()) + mt.bias[i];
            idx.sort_by(|a, b| sc(*b).total_cmp(&sc(*a)).then(a.cmp(b)));
            idx.truncate(self.pred_k.max(g.used));
            if i == 0 {
                self.pred.lock().unwrap().insert(lt, idx[..g.used].to_vec());
            }
            self.claim_reads(&mut c, mt, lt, &idx, clock, &mut reqs);
        }
        let l2 = targets[0];
        let Mix::Moe(m2) = &self.layers[l2].mix else { return Ok(()) };
        self.copy_guessed(&mut c, m2, l2, &reqs)?;
        drop(c);
        if !reqs.is_empty() {
            // one batch per expert (its matrices), by the layer of its slots: that layer waits for the experts it chose only
            let mut by: std::collections::BTreeMap<(usize, usize), Vec<IoReq>> = Default::default();
            for (l, x, r) in reqs {
                by.entry((l, x)).or_default().push(r);
            }
            let mut fl = self.inflight.lock().unwrap();
            for ((l, x), rs) in by {
                fl.entry(l).or_default().push((x, self.io.submit(&rs)));
            }
        }
        Ok(())
    }

    /// Slots for the experts of `idx` (of MoE layer `lt`) the cache lacks, their reads appended to `reqs`
    fn claim_reads(&self, c: &mut ExpertCache, m2: &Moe, l2: usize, idx: &[usize], clock: u64, reqs: &mut Vec<(usize, usize, IoReq)>) {
        for &x in idx {
            for (down, ex) in [(false, &m2.up), (true, &m2.down)] {
                let key = (l2, x, down);
                if c.hot.contains_key(&key) {
                    continue;
                }
                if let Some(&(p, sl)) = c.map.get(&key) {
                    c.pools[p].touch(sl, clock);
                    continue;
                }
                let p = c.pools.iter().position(|q| q.stride == ex.stride).unwrap();
                let Some(sl) = c.victim(p) else { continue };
                if c.pools[p].used[sl] >= clock {
                    continue; // nothing older to give up
                }
                if let Some(old) = c.pools[p].owner[sl].take() {
                    c.map.remove(&old);
                }
                c.pools[p].owner[sl] = Some(key);
                c.pools[p].touch(sl, clock);
                c.map.insert(key, (p, sl));
                c.misses += 1;
                c.reading.insert(key);
                reqs.push((l2, x, self.req(ex, x, c.pools[p]._host.as_slice().as_ptr() as usize + sl * ex.stride)));
            }
        }
    }

    /// The guessed experts of `l2` already in pinned memory (not those just queued to be read) copied toward VRAM now, on the
    /// copy lane, after the kernels that last read the buffer
    fn copy_guessed(&self, c: &mut ExpertCache, m2: &Moe, l2: usize, reqs: &[(usize, usize, IoReq)]) -> Result<()> {
        if !c.pre.is_empty() {
            let b = c.pre_ring;
            c.pre_ring = 1 - b;
            c.pre_used[b] = 0;
            let cap = c.pre[b].len;
            let fresh: std::collections::HashSet<usize> = reqs.iter().map(|r| r.2 .3).collect();
            let after: Vec<i64> = if c.pre_done[b] >= 0 { vec![c.pre_done[b]] } else { Vec::new() };
            let api = nextcuda_core::api()?;
            let mut last = -1i64;
            for x in self.pred.lock().unwrap().get(&l2).cloned().unwrap_or_default().into_iter().take(self.pre_k) {
                for (down, ex) in [(false, &m2.up), (true, &m2.down)] {
                    let key = (l2, x, down);
                    if c.hot.contains_key(&key) || c.pre_used[b] + ex.stride > cap {
                        continue;
                    }
                    let Some(&(p, sl)) = c.map.get(&key) else { continue };
                    let src = c.pools[p]._host.as_slice().as_ptr() as usize + sl * ex.stride;
                    if fresh.contains(&src) || c.reading.contains(&key) {
                        continue;
                    }
                    let off = c.pre_used[b];
                    c.pre_used[b] += (ex.stride + 255) & !255;
                    let mut ticket = 0i64;
                    // SAFETY: a pinned slot of ex.stride bytes into the buffer's [off, off + stride), on the copy lane after the ticket.
                    let dst = unsafe { c.pre[b].ptr().cast::<u8>().add(off) };
                    ffi::check(unsafe { (api.stream_copy)(self.raw(), dst.cast(), src as *const c_void, ex.stride, after.as_ptr(), after.len() as i32, 0, &mut ticket) }, "copying a guessed expert")?;
                    last = ticket;
                    c.pre_map.insert(key, (b, off));
                }
            }
            if last >= 0 {
                c.pre_ticket.insert(l2, (last, b));
            }
        }
        Ok(())
    }

    /// A decode pass's experts `xs` of layer m (those not in VRAM already) copied from their pinned slots into the
    /// staging buffer, on the GPU's stream or the copy lane (then the ticket the products wait for): they then read VRAM
    fn stage(&self, c: &mut ExpertCache, m: &Moe, xs: &[usize]) -> Result<Option<i64>> {
        let Some(cap) = c.stage.as_ref().map(|b| b.len) else { return Ok(None) };
        let mut last: Option<i64> = None;
        let (mut dsts, mut srcs, mut sizes): (Vec<*mut c_void>, Vec<*const c_void>, Vec<usize>) = Default::default();
        let mut done = std::collections::HashSet::new();
        for x in xs {
            if !done.insert(*x) {
                continue;
            }
            for (down, ex) in [(false, &m.up), (true, &m.down)] {
                let key = (m.layer, *x, down);
                if c.hot.contains_key(&key) || c.stage_map.contains_key(&key) {
                    continue;
                }
                if c.pre_map.contains_key(&key) {
                    self.times.lock().unwrap().pre_hit += 1;
                    continue;
                }
                let Some(&(p, sl)) = c.map.get(&key) else { continue };
                let src = c.pools[p]._host.as_slice().as_ptr() as usize + sl * ex.stride;
                // into the VRAM tier: the least recently wanted slot of its size that this layer does not use (what earlier
                // layers' kernels read from it is queued before this copy, on the same stream)
                if self.dyn_vram {
                    if let Some(vi) = c.vslots.iter().position(|v| v.stride == ex.stride) {
                        let clock = c.clock;
                        let (vs, _) = c.vslots[vi].used.iter().enumerate().fold((usize::MAX, u64::MAX), |b, (i, u)| if *u < b.1 { (i, *u) } else { b });
                        if vs != usize::MAX && c.vslots[vi].used[vs] < clock {
                            if let Some(old) = c.vslots[vi].owner[vs].take() {
                                c.hot.remove(&old);
                                c.hot_at.remove(&old);
                            }
                            let off = c.vslots[vi].base + vs * ex.stride;
                            // SAFETY: a pinned slot of ex.stride bytes into the hot buffer's slot [off, off + stride), on the GPU's stream.
                            let dst = unsafe { c.hot_buf.as_ref().unwrap().ptr().cast::<u8>().add(off) };
                            ffi::check(unsafe { (nextcuda_core::api()?.copy_to)(self.raw(), dst.cast(), src as *const c_void, ex.stride) }, "an expert into VRAM")?;
                            c.vslots[vi].owner[vs] = Some(key);
                            c.vslots[vi].used[vs] = clock;
                            c.hot.insert(key, off);
                            c.hot_at.insert(key, (vi, vs));
                            c.vram_ins += 1;
                            self.times.lock().unwrap().stage_copy += 1;
                            continue;
                        }
                    }
                }
                if c.stage_used + ex.stride > cap {
                    continue;
                }
                self.times.lock().unwrap().stage_copy += 1;
                let off = c.stage_used;
                c.stage_used += (ex.stride + 255) & !255;
                // SAFETY: a pinned slot of ex.stride bytes to the staging buffer's [off, off + stride), on the GPU's stream
                // or on copy lane 1 after the kernels that last read the buffer.
                let dst = unsafe { c.stage.as_ref().unwrap().ptr().cast::<u8>().add(off) };
                if self.copy_lane && self.copy_batch.is_some() {
                    dsts.push(dst.cast());
                    srcs.push(src as *const c_void);
                    sizes.push(ex.stride);
                } else if self.copy_lane {
                    let after: Vec<i64> = if c.stage_done >= 0 { vec![c.stage_done] } else { Vec::new() };
                    let mut tk = 0i64;
                    ffi::check(unsafe { (nextcuda_core::api()?.stream_copy)(self.raw(), dst.cast(), src as *const c_void, ex.stride, after.as_ptr(), after.len() as i32, 1, &mut tk) }, "staging an expert")?;
                    last = Some(tk);
                } else {
                    ffi::check(unsafe { (nextcuda_core::api()?.copy_to)(self.raw(), dst.cast(), src as *const c_void, ex.stride) }, "staging an expert")?;
                }
                c.stage_map.insert(key, off);
            }
        }
        if let (Some(f), false) = (self.copy_batch, dsts.is_empty()) {
            let after: Vec<i64> = if c.stage_done >= 0 { vec![c.stage_done] } else { Vec::new() };
            let mut tk = 0i64;
            // SAFETY: dsts[i] is ex.stride bytes of the staging buffer, srcs[i] a pinned slot of sizes[i] bytes; copy lane 1
            // after the kernels that last read the buffer.
            ffi::check(unsafe { f(self.raw(), dsts.as_ptr(), srcs.as_ptr(), sizes.as_ptr(), dsts.len(), after.as_ptr(), after.len() as i32, 1, &mut tk) }, "staging experts")?;
            last = Some(tk);
        }
        Ok(last)
    }

    /// The read of expert `x`'s matrix of `ex` into `dst`
    fn req(&self, ex: &Experts, x: usize, dst: usize) -> IoReq {
        let t = self.file.tensor(&ex.name).unwrap();
        (t.shard, t.offset + (x * ex.stride) as u64, ex.stride, dst)
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

impl Drop for NemotronH {
    /// reads still filling the pools finish before the pools go
    fn drop(&mut self) {
        for (_, v) in self.inflight.lock().unwrap().drain() {
            for (_, p) in v {
                let _ = p.wait();
            }
        }
        if let Some(c) = &self.cache {
            for (_, _, p) in c.lock().unwrap().stale.drain(..) {
                let _ = p.wait();
            }
        }
    }
}

/// (hits, misses) of the expert cache
impl NemotronH {
    /// (VRAM hits, pinned hits, misses, matrices decode copied into VRAM slots)
    pub fn tier_stats(&self) -> (u64, u64, u64, u64) {
        self.cache.as_ref().map_or((0, 0, 0, 0), |c| {
            let c = c.lock().unwrap();
            (c.hot_hits, c.hits, c.misses, c.vram_ins)
        })
    }

    pub fn cache_stats(&self) -> (u64, u64) {
        self.cache.as_ref().map_or((0, 0), |c| {
            let c = c.lock().unwrap();
            (c.hits + c.hot_hits, c.misses)
        })
    }
}

/// A read of the model's files: (file of the split, absolute offset, bytes, destination address)
type IoReq = (usize, u64, usize, usize);

/// A batch of reads in flight
struct Pending {
    left: std::sync::Mutex<usize>,
    cv: std::sync::Condvar,
    err: std::sync::Mutex<Option<String>>,
}

impl Pending {
    fn ready(&self) -> bool {
        *self.left.lock().unwrap() == 0
    }

    fn wait(&self) -> std::result::Result<(), String> {
        let mut l = self.left.lock().unwrap();
        while *l > 0 {
            l = self.cv.wait(l).unwrap();
        }
        match self.err.lock().unwrap().take() {
            Some(x) => Err(x),
            None => Ok(()),
        }
    }
}

/// Reader threads over the model's files, opened for direct reads (O_DIRECT: past the page cache, which holds the same
/// bytes the cache pools do and slows the reads; many in flight, an NVMe's speed) unless NS_NH_DIRECT=0. A batch is
/// submitted and waited for later, so reads can run while the GPU computes.
struct Io {
    tx: std::sync::mpsc::Sender<(IoReq, Arc<Pending>)>,
}

struct AlignedBuf {
    p: *mut u8,
    cap: usize,
}

impl AlignedBuf {
    fn get(&mut self, n: usize) -> &mut [u8] {
        if self.cap < n {
            if !self.p.is_null() {
                // SAFETY: allocated below with this layout.
                unsafe { std::alloc::dealloc(self.p, std::alloc::Layout::from_size_align(self.cap, 4096).unwrap()) };
            }
            let cap = n.next_multiple_of(1 << 20);
            // SAFETY: a non-zero layout.
            self.p = unsafe { std::alloc::alloc(std::alloc::Layout::from_size_align(cap, 4096).unwrap()) };
            self.cap = cap;
        }
        // SAFETY: cap >= n bytes were allocated.
        unsafe { std::slice::from_raw_parts_mut(self.p, n) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        if !self.p.is_null() {
            // SAFETY: allocated in `get` with this layout.
            unsafe { std::alloc::dealloc(self.p, std::alloc::Layout::from_size_align(self.cap, 4096).unwrap()) };
        }
    }
}

// SAFETY: a plain allocation owned by one thread.
unsafe impl Send for AlignedBuf {}

impl Io {
    fn new(f: &Gguf, nthreads: usize) -> Io {
        use std::os::unix::fs::OpenOptionsExt;
        let direct: Option<Vec<std::fs::File>> = if std::env::var("NS_NH_DIRECT").is_ok_and(|v| v == "0") {
            None
        } else {
            f.paths.iter().map(|p| std::fs::OpenOptions::new().read(true).custom_flags(0x4000).open(p)).collect::<std::io::Result<Vec<_>>>().ok()
        };
        let buffered: Vec<std::fs::File> = f.paths.iter().map(|p| std::fs::File::open(p).expect("the model file")).collect();
        let files = Arc::new((direct, buffered));
        let (tx, rx) = std::sync::mpsc::channel::<(IoReq, Arc<Pending>)>();
        let rx = Arc::new(std::sync::Mutex::new(rx));
        for _ in 0..nthreads {
            let (rx, files) = (rx.clone(), files.clone());
            std::thread::spawn(move || {
                let mut buf = AlignedBuf { p: std::ptr::null_mut(), cap: 0 };
                loop {
                    let job = rx.lock().unwrap().recv();
                    let Ok((req, pend)) = job else { break };
                    if let Err(x) = Io::read(&files, &mut buf, req) {
                        *pend.err.lock().unwrap() = Some(x);
                    }
                    let mut l = pend.left.lock().unwrap();
                    *l -= 1;
                    if *l == 0 {
                        pend.cv.notify_all();
                    }
                }
            });
        }
        Io { tx }
    }

    fn read(files: &(Option<Vec<std::fs::File>>, Vec<std::fs::File>), buf: &mut AlignedBuf, (shard, abs, len, dst): IoReq) -> std::result::Result<(), String> {
        use std::os::unix::fs::FileExt;
        // SAFETY: the destination is a slot or buffer of `len` bytes that the submitter keeps and does not touch until the batch is waited for.
        let dst = unsafe { std::slice::from_raw_parts_mut(dst as *mut u8, len) };
        let Some(direct) = &files.0 else {
            return files.1[shard].read_exact_at(dst, abs).map_err(|x| x.to_string());
        };
        let start = abs & !4095;
        let end = (abs + len as u64 + 4095) & !4095;
        let n = (end - start) as usize;
        let b = buf.get(n);
        let mut got = 0usize;
        while got < n {
            match direct[shard].read_at(&mut b[got..], start + got as u64) {
                Ok(0) => break,
                Ok(k) => got += k,
                Err(x) if x.kind() == std::io::ErrorKind::Interrupted => {}
                Err(x) => return Err(x.to_string()),
            }
        }
        let skip = (abs - start) as usize;
        if got < skip + len {
            return Err("a direct read came up short".into());
        }
        dst.copy_from_slice(&b[skip..skip + len]);
        Ok(())
    }

    /// Queue the reads; wait on the result when the bytes are needed
    fn submit(&self, reqs: &[IoReq]) -> Arc<Pending> {
        let pend = Arc::new(Pending { left: std::sync::Mutex::new(reqs.len()), cv: std::sync::Condvar::new(), err: std::sync::Mutex::new(None) });
        for r in reqs {
            let _ = self.tx.send((*r, pend.clone()));
        }
        pend
    }

    fn run(&self, reqs: &[IoReq]) -> std::result::Result<(), String> {
        self.submit(reqs).wait()
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
        // what the pass buffers take beside the weights: the arena, the biggest dense matrix as half, the decode staging
        let w16_need = f.tensors.iter().filter(|t| !t.name.contains("_exps.") && !t.name.starts_with(&nextn_prefix) && t.shape.len() >= 2 && !plain(t.ty)).map(|t| t.elements() * 2).max().unwrap_or(0);
        let widest_pair = (0..g.layers).filter(|l| exp_bytes[*l] > 0).map(|l| exp_bytes[l] / g.experts as u64).max().unwrap_or(0);
        // a prompt pass's buffers for `rows` rows; with a profile (the VRAM then a hot set) only decode's and short prompts' stay
        // resident (NS_NH_RESIDENT_ROWS, 128) and a long prompt borrows the rest from the hot set's tail (Strata's trick: the
        // least wanted hot experts, read back after it)
        let arena_bytes = |rows: usize| (96u64 << 20) + rows as u64 * 700_000 + (rows * g.used * 28_000) as u64;
        let has_profile = o.profile.as_ref().is_some_and(|p| p.exists()) && std::env::var("NS_NH_HOT").map_or(true, |v| v != "0");
        let resident_rows = if has_profile && std::env::var("NS_NH_BORROW").map_or(true, |v| v != "0") {
            std::env::var("NS_NH_RESIDENT_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(128usize).min(o.chunk)
        } else {
            o.chunk
        };
        let arena_need = arena_bytes(resident_rows) + w16_need + if o.stage { SMALL as u64 * g.used as u64 * widest_pair + 2 * g.used as u64 * widest_pair } else { 0 };
        let mut vram_left = free.saturating_sub(dense + reserve + arena_need);
        let mut homes = vec![0u8; g.layers]; // 0 VRAM, 2 the cache
        // with a profile the VRAM is the hot experts' (of any layer), not the first layers' whole
        let hot_budget = if has_profile { vram_left } else { 0 };
        // what a long prompt borrows (0: none - its passes are the resident rows')
        let borrow_bytes = if resident_rows < o.chunk && hot_budget >= arena_bytes(o.chunk) + (512 << 20) { arena_bytes(o.chunk) as usize } else { 0 };
        let chunk = if resident_rows < o.chunk && borrow_bytes == 0 { resident_rows } else { o.chunk };
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
            pools.push(Pool { _host: hb, view, stride, owner: vec![None; n], used: vec![0; n], lru: (0..n).map(|i| (0, i)).collect() });
        }
        let scratch = if o.scratch_mib > 0 && !pools.is_empty() {
            let mut hb = HostBuf::new(&gpu, o.scratch_mib << 20)?;
            let view = hb.device_view(0, o.scratch_mib << 20)?;
            Some((hb, view))
        } else {
            None
        };
        // the biggest dense matrix as half
        let mut big = 0usize;
        for ly in &layers {
            let mats: Vec<&Mat> = match &ly.mix {
                Mix::Mamba(m) => vec![&m.inp, &m.out],
                Mix::Attn(a) => vec![&a.q, &a.k, &a.v, &a.o],
                Mix::Moe(m) => vec![&m.to_latent, &m.from_latent, &m.sh_up, &m.sh_down],
            };
            for x in mats {
                if x.ty != GType::F32 {
                    big = big.max(x.rows * x.cols * 2);
                }
            }
        }
        let w16 = if big > 0 && o.chunk > 16 { Some(DevBuf::new(&gpu, big)?) } else { None };
        let stage_bytes = if o.stage && !pools.is_empty() {
            pools.iter().map(|q| q.stride).max().unwrap_or(0) * 0 + SMALL * g.used * (layers.iter().filter_map(|ly| match &ly.mix { Mix::Moe(m) if matches!(m.up.home, Home::Cached) => Some(m.up.stride + m.down.stride), _ => None }).max().unwrap_or(0))
        } else {
            0
        };
        let stage = if stage_bytes > 0 { Some(DevBuf::new(&gpu, stage_bytes)?) } else { None };
        // the guessed experts' two buffers: one layer's worth (22 experts) each
        let pre_bytes = if stage_bytes > 0 { stage_bytes / SMALL } else { 0 };
        let pre: Vec<DevBuf> = if pre_bytes > 0 && std::env::var("NS_NH_PREDICT").map_or(true, |v| v != "0") { (0..2).map(|_| DevBuf::new(&gpu, pre_bytes)).collect::<Result<_>>()? } else { Vec::new() };
        let hot_buf = if hot_budget > 0 && !pools.is_empty() { Some(DevBuf::new(&gpu, hot_budget as usize)?) } else { None };
        let cache = if pools.is_empty() { None } else { Some(std::sync::Mutex::new(ExpertCache { pools, map: Default::default(), clock: 0, hits: 0, misses: 0, wanted: Default::default(), hot: Default::default(), hot_buf, hot_hits: 0, vslots: Vec::new(), hot_at: Default::default(), vram_ins: 0, scratch, scratch_used: 0, scratch_map: Default::default(), stage, stage_used: 0, stage_map: Default::default(), pre, pre_used: [0; 2], pre_map: Default::default(), pre_ticket: Default::default(), pre_done: [-1; 2], pre_ring: 0, reading: Default::default(), stale: Vec::new(), stage_done: -1 })) };
        // the CPU's share: when ggml-cpu is there and takes every cached layer's expert types
        let cpu = cpu::Ggml::get().filter(|gg| layers.iter().all(|ly| match &ly.mix {
            Mix::Moe(m) if matches!(m.up.home, Home::Cached) => gg.supports(m.up.ty.code() as i32) && gg.supports(m.down.ty.code() as i32),
            _ => true,
        })).map(|gg| (gg, cpu::Pool::new(std::env::var("NS_NH_CPU_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get())))));
        let cpu_frac: f64 = std::env::var("NS_NH_CPU_FRAC").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0f64).clamp(0.0, 1.0);
        if let (Some((_, p)), true) = (&cpu, cpu_frac > 0.0) {
            log(format!("nemotronh: decode multiplies {:.0} % of a layer's pinned experts on {} CPU threads (ggml-cpu)", cpu_frac * 100.0, p.threads));
        }
        let copy_batch: Option<CopyBatch> = match nextcuda_core::api()?.symbol("ns_stream_copy_batch") {
            p if p.is_null() || std::env::var("NS_NH_COPY_BATCH").is_ok_and(|v| v == "0") => None,
            // SAFETY: ns.h's declaration of ns_stream_copy_batch is CopyBatch.
            p => Some(unsafe { std::mem::transmute::<*mut c_void, CopyBatch>(p) }),
        };
        let ring = std::sync::Mutex::new((HostBuf::new(&gpu, 8 << 20)?, 0usize));
        let io = Io::new(&f, std::env::var("NS_NH_IO_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(24));
        let arena_home = DevBuf::new(&gpu, arena_bytes(resident_rows) as usize)?;
        let arena = Arena::on(arena_home.view(0, arena_home.len)?);
        if borrow_bytes > 0 {
            log(format!("nemotronh: prompts of more than {resident_rows} rows borrow {:.1} GiB of the hot set for passes of {chunk}", gib(borrow_bytes as u64)));
        }
        gpu.sync()?;
        let load_s = t0.elapsed().as_secs_f64();
        log(format!("nemotronh: loaded {:.1} GiB onto {} in {load_s:.0} s ({} layers: {} Mamba-2, {} MoE, {} attention)", gib(loaded.get()), gpu.name, g.layers, g.count(Kind::Mamba),
                    g.count(Kind::Moe), g.count(Kind::Attn)));
        let _ = o;
        Ok(NemotronH { g, file: f, gpu, k, layers, embd, output_norm, output, arena, cache, io, inflight: Default::default(), chunk, resident_rows, borrow_bytes, arena_home, lent: Default::default(), predict: std::env::var("NS_NH_PREDICT").map_or(true, |v| v != "0"), pred: Default::default(), pred_ahead: std::env::var("NS_NH_PRED_AHEAD").ok().and_then(|v| v.parse().ok()).unwrap_or(1), pred_k: std::env::var("NS_NH_PRED_K").ok().and_then(|v| v.parse().ok()).unwrap_or(22), w16, dense_gemm_min: std::env::var("NS_NH_DENSE_GEMM_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(16), times: Default::default(), load_s, cpu, cpu_frac, dyn_vram: std::env::var("NS_NH_DYN_VRAM").is_ok_and(|v| v == "1"), overlap: std::env::var("NS_NH_OVERLAP").map_or(true, |v| v != "0"), pre_k: std::env::var("NS_NH_PRE_K").ok().and_then(|v| v.parse().ok()).unwrap_or(8), tail_skip: std::env::var("NS_NH_TAIL_SKIP").ok().and_then(|v| v.parse().ok()).unwrap_or(8), tail_vram: std::env::var("NS_NH_TAIL_TIER").is_ok_and(|v| v == "vram"), copy_lane: std::env::var("NS_NH_COPY_LANE").map_or(true, |v| v != "0"), copy_batch, ring })
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
        // a pass of many rows: the matrix expanded to half once and multiplied on the tensor cores (the quantized product would read
        // it again for each eight rows)
        if t >= self.dense_gemm_min && self.w16.as_ref().is_some_and(|b| b.len >= w.rows * w.cols * 2) && w.ty != GType::F32 {
            let w16 = self.w16.as_ref().unwrap();
            let xh = self.arena.bytes(t * w.cols * 2)?;
            // SAFETY: w's blocks -> half [rows][cols]; x [t][cols] -> half; y [t][rows] = x . w^T.
            unsafe {
                self.op("dequant", || (k.dequant_f16)(self.raw(), w.ty.code() as i32, w.buf.ptr(), (w.rows * w.cols) as i64, w16.ptr()))?;
                self.op("to half", || (k.to_half)(self.raw(), x.fp(), xh.ptr(), (t * w.cols) as i64))?;
                return self.op("gemm", || (k.gemm_f16)(self.raw(), w16.ptr(), xh.ptr(), y.fp(), t as i64, w.rows as i64, w.cols as i64));
            }
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

    /// `src` into the front of `dst`, queued on the GPU's stream from the pinned ring (no wait)
    fn up(&self, dst: &DevBuf, src: &[u8]) -> Result<()> {
        dst.bounds(0, src.len())?;
        let mut r = self.ring.lock().unwrap();
        if src.len() > r.0.len / 4 || std::env::var("NS_NH_RING").is_ok_and(|v| v == "0") {
            return dst.write(0, src);
        }
        if r.1 + src.len() > r.0.len {
            // everything queued from the ring so far done before it is written again
            self.gpu.sync()?;
            r.1 = 0;
        }
        let off = r.1;
        r.1 = (off + src.len() + 255) & !255;
        r.0.as_mut_slice()[off..off + src.len()].copy_from_slice(src);
        let p = r.0.as_slice()[off..].as_ptr();
        // SAFETY: src.len() pinned bytes of the ring, not written again before a sync, into dst (in bounds, checked).
        ffi::check(unsafe { (nextcuda_core::api()?.copy_to)(self.raw(), dst.ptr(), p.cast(), src.len()) }, "an upload")
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
        self.up(&src, &raw)?;
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

    /// The address of expert `x`'s matrix `ex` (of layer `l`; up or down) and whether it is in host memory (a pool's or the
    /// scratch's slot, which the kernels read over PCIe) rather than VRAM
    fn expert_src(&self, ex: &Experts, l: usize, down: bool, x: usize) -> (*const c_void, bool) {
        // SAFETY: inside the tensor / the cache (callers' counts).
        unsafe {
            match &ex.home {
                Home::Vram(b) => (b.ptr().cast::<u8>().add(x * ex.stride).cast(), false),
                Home::Cached => {
                    let c = self.cache.as_ref().unwrap().lock().unwrap();
                    if let Some(&off) = c.hot.get(&(l, x, down)) {
                        return (c.hot_buf.as_ref().unwrap().ptr().cast::<u8>().add(off).cast(), false);
                    }
                    if let Some(&(b, off)) = c.pre_map.get(&(l, x, down)) {
                        return (c.pre[b].ptr().cast::<u8>().add(off).cast(), false);
                    }
                    if let Some(&off) = c.stage_map.get(&(l, x, down)) {
                        return (c.stage.as_ref().unwrap().ptr().cast::<u8>().add(off).cast(), false);
                    }
                    if let Some(&off) = c.scratch_map.get(&(l, x, down)) {
                        return (c.scratch.as_ref().unwrap().1.ptr().cast::<u8>().add(off).cast(), true);
                    }
                    let (p, sl) = c.map[&(l, x, down)];
                    (c.pools[p].view.ptr().cast::<u8>().add(sl * ex.stride).cast(), true)
                }
            }
        }
    }

    fn expert_at(&self, ex: &Experts, l: usize, down: bool, x: usize) -> *const c_void {
        self.expert_src(ex, l, down, x).0
    }

    /// The matrix where the kernels read it at VRAM's speed: in VRAM already, or copied (DMA) into `tmp` first
    fn expert_dev(&self, ex: &Experts, l: usize, down: bool, x: usize, tmp: &DevBuf) -> Result<*const c_void> {
        let (p, host) = self.expert_src(ex, l, down, x);
        if !host {
            return Ok(p);
        }
        // SAFETY: a host slot of ex.stride bytes to the temporary buffer (as large), on the GPU's stream.
        ffi::check(unsafe { (nextcuda_core::api()?.copy_to)(self.raw(), tmp.ptr(), p, ex.stride) }, "copying an expert")?;
        Ok(tmp.ptr().cast_const())
    }

    /// A latent-MoE layer on h [t, n_embd]: the routed experts in the latent, projected back, plus the shared expert
    /// `x`: the residual stream before the layer, for the next MoE layer's guessed routing (decode)
    fn moe(&self, m: &Moe, h: &DevBuf, t: usize, x: Option<&DevBuf>) -> Result<DevBuf> {
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
        let guess = match x {
            Some(x) => self.predict_launch(m.layer, x)?,
            None => None,
        };
        let lg = logits.to_f32()?;
        if let Some((targets, glg)) = guess {
            let all = glg.to_f32()?;
            self.predict_finish(&targets, &all)?;
        }
        // the CPU's share needs the latent row on the host (the sync above has it ready)
        let cpu_on = t == 1 && self.cpu.is_some() && self.cpu_frac > 0.0 && matches!(m.up.home, Home::Cached);
        let lat_host = if cpu_on { Some(lat.to_f32()?) } else { None };
        let mut cpu_jobs: Vec<cpu::Job> = Vec::new();
        let mut cpu_x: std::collections::HashSet<usize> = Default::default();
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
        lap("cpu route");
        if t == 1 {
            if let Some(pr) = self.pred.lock().unwrap().remove(&m.layer) {
                let mut tm = self.times.lock().unwrap();
                tm.pred_total += ent.len() as u64;
                tm.pred_right += ent.iter().filter(|x| pr.contains(&x.1)).count() as u64;
            }
            if let Ok(path) = std::env::var("NS_NH_TRACE") {
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                    let _ = writeln!(f, "{} {}", m.layer, ent.iter().map(|x| x.1.to_string()).collect::<Vec<_>>().join(","));
                }
            }
        }
        // the tail skip: of a decode token's chosen experts (in rank order), the low-ranked ones whose bytes are not near
        let mut ent = ent;
        if t == 1 && self.tail_skip > 0 && self.tail_skip < ent.len() && matches!(m.up.home, Home::Cached) {
            let c = self.cache.as_ref().unwrap().lock().unwrap();
            let near = |x: usize| {
                [false, true].iter().all(|&down| {
                    let key = (m.layer, x, down);
                    c.hot.contains_key(&key) || c.pre_map.contains_key(&key) || (!self.tail_vram && c.map.contains_key(&key))
                })
            };
            let before = ent.len();
            let mut rank = 0;
            ent.retain(|e| {
                rank += 1;
                rank <= self.tail_skip || near(e.1)
            });
            self.times.lock().unwrap().skipped += (before - ent.len()) as u64;
        }
        // the experts' bytes where the kernels can read them: those of a cached layer into slots (the least recently
        // used given up), the misses read from the file on a few threads
        let tf = Instant::now();
        let mut pre_b: Option<usize> = None;
        // decode's missed experts, read while the first batch runs, and their reads
        let mut late: std::collections::HashSet<usize> = Default::default();
        let mut late_io: Option<(Arc<Pending>, Instant)> = None;
        let mut late_wait: Vec<(usize, Arc<Pending>)> = Vec::new();
        // decode's experts already in VRAM, and the copy lane's ticket after the others' staging
        let mut in_vram: std::collections::HashSet<usize> = Default::default();
        let mut stage_tk: Option<i64> = None;
        if matches!(m.up.home, Home::Cached) {
            // reads started for this layer from a guess at its routing, and those of earlier tokens' guesses not done yet:
            // decode waits only for the experts it chose (those still reading join the second batch when it overlaps), the
            // others left to finish meanwhile, their slots kept from eviction until then; a prompt pass waits for all
            let mut pend = self.inflight.lock().unwrap().remove(&m.layer).unwrap_or_default();
            let mut c = self.cache.as_ref().unwrap().lock().unwrap();
            for (l, x, p) in std::mem::take(&mut c.stale) {
                if l == m.layer {
                    pend.push((x, p));
                } else if p.ready() {
                    let failed = p.wait().is_err();
                    c.settle(l, x, failed);
                } else {
                    c.stale.push((l, x, p));
                }
            }
            let chosen: std::collections::HashSet<usize> = ent.iter().map(|x| x.1).collect();
            let tw = Instant::now();
            for (x, p) in pend {
                if t == 1 && !chosen.contains(&x) {
                    c.stale.push((m.layer, x, p));
                } else if t == 1 && self.overlap && !cpu_on && !p.ready() {
                    late.insert(x);
                    late_wait.push((x, p));
                } else {
                    let r = p.wait();
                    c.settle(m.layer, x, r.is_err());
                    r.map_err(e)?;
                }
            }
            self.times.lock().unwrap().pred_wait += tw.elapsed().as_secs_f64();
            c.clock += 1;
            let clock = c.clock;
            c.scratch_map.clear();
            c.scratch_used = 0;
            c.stage_map.clear();
            c.stage_used = 0;
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
                        if let Some(&(vi, sl)) = c.hot_at.get(&key) {
                            c.vslots[vi].used[sl] = clock;
                        }
                        continue;
                    }
                    if let Some(&(p, sl)) = c.map.get(&key) {
                        c.pools[p].touch(sl, clock);
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
                    let sl = c.victim(p).ok_or_else(|| e("every slot of a pool is being read into"))?;
                    if let Some(old) = c.pools[p].owner[sl].take() {
                        c.map.remove(&old);
                    }
                    c.pools[p].owner[sl] = Some(key);
                    c.pools[p].touch(sl, clock);
                    c.map.insert(key, (p, sl));
                    c.misses += 1;
                    jobs.push((down, *x, c.pools[p]._host.as_slice().as_ptr() as usize + sl * ex.stride));
                }
            }
            if !jobs.is_empty() {
                let tio = Instant::now();
                let nbytes: usize = jobs.iter().map(|j| if j.0 { m.down.stride } else { m.up.stride }).sum();
                let reqs: Vec<IoReq> = jobs.iter().map(|&(down, x, at)| self.req(if down { &m.down } else { &m.up }, x, at)).collect();
                self.times.lock().unwrap().io_bytes += nbytes as u64;
                if t == 1 && self.overlap && !cpu_on {
                    // decode: the reads in flight while the GPU multiplies the experts it has (a second batch takes these)
                    late.extend(jobs.iter().map(|j| j.1));
                    late_io = Some((self.io.submit(&reqs), tio));
                } else {
                    self.io.run(&reqs).map_err(e)?;
                    self.times.lock().unwrap().io_s += tio.elapsed().as_secs_f64();
                }
            }
            lap("fetch misses");
            // the CPU's share: of the experts whose matrices are both in pinned memory only (not hot, not copied up already
            // for a guess), the first part by weight multiplied there by the CPU while the GPU does the rest
            if cpu_on {
                let mut cand: Vec<(usize, f32, usize, usize)> = Vec::new();
                for &(_, x, w) in &ent {
                    let (ku, kd) = ((m.layer, x, false), (m.layer, x, true));
                    if c.hot.contains_key(&ku) || c.hot.contains_key(&kd) || c.pre_map.contains_key(&ku) || c.pre_map.contains_key(&kd) {
                        continue;
                    }
                    let (Some(&(pu, su)), Some(&(pd, sd))) = (c.map.get(&ku), c.map.get(&kd)) else { continue };
                    cand.push((x, w, c.pools[pu]._host.as_slice().as_ptr() as usize + su * m.up.stride, c.pools[pd]._host.as_slice().as_ptr() as usize + sd * m.down.stride));
                }
                let n = (cand.len() as f64 * self.cpu_frac).round() as usize;
                for (x, w, up, down) in cand.into_iter().take(n) {
                    cpu_x.insert(x);
                    cpu_jobs.push(cpu::Job { up, down, w });
                }
                self.times.lock().unwrap().cpu_experts += cpu_jobs.len() as u64;
            }
            // a decode pass: the experts not in VRAM already copied into the staging buffer, then the products read that
            if t <= SMALL {
                let xs: Vec<usize> = ent.iter().map(|x| x.1).filter(|x| !cpu_x.contains(x) && !late.contains(x)).collect();
                in_vram = xs.iter().copied().filter(|&x| [false, true].iter().all(|&d| c.hot.contains_key(&(m.layer, x, d)) || c.pre_map.contains_key(&(m.layer, x, d)))).collect();
                stage_tk = self.stage(&mut c, m, &xs)?;
            }
            // the copies of the guessed experts, if any: the kernels wait for them (on the device)
            if t <= SMALL {
                if let Some((tk, b)) = c.pre_ticket.remove(&m.layer) {
                    ffi::check(unsafe { (nextcuda_core::api()?.await_)(self.raw(), tk) }, "waiting for the guessed experts")?;
                    pre_b = Some(b);
                }
            }
        }
        lap("route+fetch");
        self.times.lock().unwrap().moe_fetch += tf.elapsed().as_secs_f64();
        let latq = self.quantize(lat.fp(), g.latent, t)?;
        // the GPU's entries: those the CPU does not take
        let ent: Vec<(usize, usize, f32)> = if cpu_x.is_empty() { ent } else { ent.into_iter().filter(|x| !cpu_x.contains(&x.1)).collect() };
        let n_e = ent.len();
        // SAFETY: arithmetic only.
        let mo = if t <= SMALL && n_e == 0 {
            let mo = self.arena.f32(t * g.latent)?;
            mo.fill(0)?;
            mo
        } else if t <= SMALL { {
        // the entries the GPU has the bytes of first, the missed ones (reads still in flight) after
        let mut ent = ent;
        let class = |x: usize| if late.contains(&x) { 2 } else if in_vram.contains(&x) { 0 } else { 1 };
        ent.sort_by_key(|x| class(x.1));
        let n_v = ent.iter().filter(|x| class(x.1) == 0).count();
        let n_a = ent.iter().filter(|x| class(x.1) < 2).count();
        let u = self.arena.f32(n_e * g.n_ff_exp)?;
        let dn = self.arena.f32(n_e * g.latent)?;
        let api = nextcuda_core::api()?;
        for (bi, (b0, b1)) in [(0, n_v), (n_v, n_a), (n_a, n_e)].into_iter().enumerate() {
            if b0 == b1 {
                continue;
            }
            if bi == 1 {
                // the staged experts' copies landed
                if let Some(tk) = stage_tk.take() {
                    ffi::check(unsafe { (api.await_)(self.raw(), tk) }, "waiting for the staged experts")?;
                }
            }
            if bi == 2 {
                // the second batch: its reads waited for, then staged as the first was
                if let Some((p, tio)) = late_io.take() {
                    p.wait().map_err(e)?;
                    self.times.lock().unwrap().io_s += tio.elapsed().as_secs_f64();
                }
                let tw = Instant::now();
                let mut c = self.cache.as_ref().unwrap().lock().unwrap();
                for (x, p) in std::mem::take(&mut late_wait) {
                    let r = p.wait();
                    c.settle(m.layer, x, r.is_err());
                    r.map_err(e)?;
                }
                self.times.lock().unwrap().pred_wait += tw.elapsed().as_secs_f64();
                let xs: Vec<usize> = ent[b0..b1].iter().map(|x| x.1).collect();
                if let Some(tk) = self.stage(&mut c, m, &xs)? {
                    ffi::check(unsafe { (api.await_)(self.raw(), tk) }, "waiting for the staged experts")?;
                }
            }
            let bt = &ent[b0..b1];
            let nb = bt.len();
            // the experts' addresses and the activation rows, in one upload: up's pointers | down's | the tokens' rows | 0..nb
            let mut ptrs: Vec<u8> = Vec::with_capacity(nb * 24);
            for (_, x, _) in bt {
                ptrs.extend_from_slice(&(self.expert_at(&m.up, m.layer, false, *x) as u64).to_le_bytes());
            }
            for (_, x, _) in bt {
                ptrs.extend_from_slice(&(self.expert_at(&m.down, m.layer, true, *x) as u64).to_le_bytes());
            }
            for (r, _, _) in bt {
                ptrs.extend_from_slice(&(*r as i32).to_le_bytes());
            }
            for j in 0..nb {
                ptrs.extend_from_slice(&(j as i32).to_le_bytes());
            }
            let pb = self.arena.bytes(ptrs.len())?;
            self.up(&pb, &ptrs)?;
            let (up_p, down_p) = (pb.ptr().cast_const(), unsafe { pb.ptr().cast::<u8>().add(nb * 8).cast_const() });
            let (rows_p, seq_p) = unsafe { (pb.ptr().cast::<u8>().add(nb * 16).cast::<i32>().cast_const(), pb.ptr().cast::<u8>().add(nb * 16 + nb * 4).cast::<i32>().cast_const()) };
            // SAFETY: the batch's rows of u [n_e][n_ff] and dn [n_e][latent], from entry b0.
            let (ub, dnb) = unsafe { (u.fp().add(b0 * g.n_ff_exp), dn.fp().add(b0 * g.latent)) };
            // SAFETY: nb experts' up [n_ff, latent] in their blocks, the tokens' latent Q8_1 rows, ub [nb][n_ff].
            self.op("expert up", || unsafe {
                (k.mmvq_ptrs)(self.raw(), m.up.ty.code() as i32, up_p.cast(), rows_p, latq.buf.ptr(), ub, g.latent as i64, g.n_ff_exp as i64, nb as i64)
            })?;
            // SAFETY: n floats.
            self.op("relu2", || unsafe { (k.relu2)(self.raw(), ub, (nb * g.n_ff_exp) as i64) })?;
            let uq8 = self.quantize(ub, g.n_ff_exp, nb)?;
            // SAFETY: the experts' down [latent, n_ff], entry j's Q8_1 row, dnb [nb][latent].
            self.op("expert down", || unsafe {
                (k.mmvq_ptrs)(self.raw(), m.down.ty.code() as i32, down_p.cast(), seq_p, uq8.buf.ptr(), dnb, g.n_ff_exp as i64, g.latent as i64, nb as i64)
            })?;
        }
        if let (true, Some(c)) = (self.copy_lane && matches!(m.up.home, Home::Cached), &self.cache) {
            // the staging buffer is free for the next layer's copies once these kernels are done with it
            let mut tk = 0i64;
            ffi::check(unsafe { (api.mark)(self.raw(), &mut tk) }, "marking the stream")?;
            c.lock().unwrap().stage_done = tk;
        }
        lap("expert up+down");
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
        self.up(&tpb, &tp.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
        self.up(&wb, &ws.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
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
            self.up(&rb, &bytes(&rows))?;
            self.up(&tb, &bytes(&tp))?;
            self.up(&ib, &bytes(&ids))?;
            self.up(&wb, &ws.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
            // SAFETY: arithmetic only.
            let (lq, uq) = unsafe { ((k.q8_1_bytes)(g.latent as i64, 1), (k.q8_1_bytes)(g.n_ff_exp as i64, 1)) };
            let xg = self.arena.bytes(n_e * lq)?;
            // SAFETY: n_e rows of the latent's Q8_1 gathered in expert order.
            self.op("gather", || unsafe { (k.gather_rows)(self.raw(), latq.buf.ptr(), rb.ptr().cast(), xg.ptr(), lq as i64, n_e as i64) })?;
            let u = self.arena.f32(n_e * g.n_ff_exp)?;
            // groups of many rows through the tensor cores (the expert dequantized to half, one GEMM); the small ones by the
            // quantized product
            let gemm_min: usize = std::env::var("NS_NH_GEMM_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(12);
            let big = groups.iter().any(|gr| gr.2 >= gemm_min);
            let w16 = if big { Some(self.arena.bytes(g.latent * g.n_ff_exp * 2)?) } else { None };
            let xh = if big { Some(self.arena.bytes(n_e * g.latent * 2)?) } else { None };
            if let Some(xh) = &xh {
                // SAFETY: n_e rows of the latent, half, in expert order.
                self.op("gather", || unsafe { (k.gather_half)(self.raw(), lat.fp(), rb.ptr().cast(), xh.ptr(), g.latent as i64, n_e as i64) })?;
            }
            let tmpw = self.arena.bytes(m.up.stride.max(m.down.stride))?;
            for &(x, first, n) in &groups {
                let w = self.expert_dev(&m.up, m.layer, false, x, &tmpw)?;
                if n >= gemm_min {
                    let (w16, xh) = (w16.as_ref().unwrap(), xh.as_ref().unwrap());
                    // SAFETY: expert x's up, dequantized to half [n_ff][latent], against n rows of xh, into rows of u.
                    unsafe {
                        self.op("dequant up", || (k.dequant_f16)(self.raw(), m.up.ty.code() as i32, w, (g.n_ff_exp * g.latent) as i64, w16.ptr()))?;
                        self.op("gemm up", || (k.gemm_f16)(self.raw(), w16.ptr(), xh.ptr().cast::<u8>().add(first * g.latent * 2).cast(), u.fp().add(first * g.n_ff_exp), n as i64, g.n_ff_exp as i64, g.latent as i64))?;
                    }
                } else {
                    // SAFETY: expert x's up against its n rows of xg, into rows of u.
                    self.op("expert up", || unsafe {
                        (k.mmvq)(self.raw(), m.up.ty.code() as i32, w, xg.ptr().cast::<u8>().add(first * lq).cast(), u.fp().add(first * g.n_ff_exp), g.latent as i64, g.n_ff_exp as i64, n as i64)
                    })?;
                }
            }
            lap("expert up");
            let uh = if big { Some(self.arena.bytes(n_e * g.n_ff_exp * 2)?) } else { None };
            if let Some(uh) = &uh {
                // SAFETY: n floats -> half(relu^2).
                self.op("relu2 half", || unsafe { (k.relu2_half)(self.raw(), u.fp(), uh.ptr(), (n_e * g.n_ff_exp) as i64) })?;
            }
            // SAFETY: n floats.
            self.op("relu2", || unsafe { (k.relu2)(self.raw(), u.fp(), (n_e * g.n_ff_exp) as i64) })?;
            let uq8 = self.quantize(u.fp(), g.n_ff_exp, n_e)?;
            let dn = self.arena.f32(n_e * g.latent)?;
            for &(x, first, n) in &groups {
                let w = self.expert_dev(&m.down, m.layer, true, x, &tmpw)?;
                if n >= gemm_min {
                    let (w16, uh) = (w16.as_ref().unwrap(), uh.as_ref().unwrap());
                    // SAFETY: expert x's down dequantized to half [latent][n_ff], against n rows of uh, into rows of dn.
                    unsafe {
                        self.op("dequant down", || (k.dequant_f16)(self.raw(), m.down.ty.code() as i32, w, (g.latent * g.n_ff_exp) as i64, w16.ptr()))?;
                        self.op("gemm down", || (k.gemm_f16)(self.raw(), w16.ptr(), uh.ptr().cast::<u8>().add(first * g.n_ff_exp * 2).cast(), dn.fp().add(first * g.latent), n as i64, g.latent as i64, g.n_ff_exp as i64))?;
                    }
                } else {
                    // SAFETY: expert x's down against its n rows of uq8, into rows of dn.
                    self.op("expert down", || unsafe {
                        (k.mmvq)(self.raw(), m.down.ty.code() as i32, w, uq8.buf.ptr().cast::<u8>().add(first * uq).cast(), dn.fp().add(first * g.latent), g.n_ff_exp as i64, g.latent as i64, n as i64)
                    })?;
                }
            }
            lap("expert down");
            let mo = self.arena.f32(t * g.latent)?;
            // SAFETY: mo [t, latent], dn [entries, latent], the token pointers, places and weights as named in nh.h.
            self.op("moe combine", || unsafe { (k.moe_combine_ids)(self.raw(), mo.fp(), dn.fp(), tb.ptr().cast(), ib.ptr().cast(), wb.fp(), t as i64, g.latent as i64) })?;
            mo
        };
        // the CPU's experts, multiplied while the GPU runs its own, then added to theirs
        if let (Some((gg, pool)), Some(xh)) = (&self.cpu, &lat_host) {
            if !cpu_jobs.is_empty() {
                let tc = Instant::now();
                let v = cpu::experts(gg, pool, &cpu_jobs, m.up.ty.code() as i32, m.down.ty.code() as i32, xh, g.n_ff_exp);
                self.times.lock().unwrap().cpu_s += tc.elapsed().as_secs_f64();
                let cb = self.arena.f32(g.latent)?;
                cb.write(0, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
                self.add(&mo, &cb, g.latent)?;
            }
        }
        lap("cpu experts");
        if matches!(m.up.home, Home::Cached) {
            let mut c = self.cache.as_ref().unwrap().lock().unwrap();
            c.pre_map.retain(|k, _| k.0 != m.layer);
            if let Some(b) = pre_b {
                // the buffer is free for the next copies once the kernels queued so far are done with it
                let mut tk = 0i64;
                ffi::check(unsafe { (nextcuda_core::api()?.mark)(self.raw(), &mut tk) }, "marking the stream")?;
                c.pre_done[b] = tk;
            }
        }
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
            let guess_x = (t == 1 && g.kinds[l] == Kind::Moe && self.predict).then_some(&x);
            let out = match (&ly.mix, st) {
                (Mix::Mamba(m), LState::Mamba { conv, s: ss }) => self.mamba(m, conv, ss, &h, t)?,
                (Mix::Attn(a), LState::Attn { k, v }) => self.attention(a, k, v, s.max_ctx, &h, t, p0)?,
                (Mix::Moe(m), LState::None) => self.moe(m, &h, t, guess_x)?,
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

    /// Rows of any length: the passes, the logits of the last `n_out`
    pub fn forward(&self, s: &mut Session, tokens: &[u32], n_out: usize) -> Result<Vec<Vec<f32>>> {
        Ok(self.feed(s, tokens, n_out, &|_| false)?.1)
    }

    /// As `forward`, `stop(read)` asked between passes: (tokens read, the last pass's logits). A prompt longer than the resident
    /// rows borrows the hot set's tail for passes of `chunk` rows, given back (its experts read again) at the end
    pub fn feed(&self, s: &mut Session, tokens: &[u32], n_out: usize, stop: &dyn Fn(usize) -> bool) -> Result<(usize, Vec<Vec<f32>>)> {
        let big = self.borrow_bytes > 0 && tokens.len() > self.resident_rows;
        let rows = if big || self.borrow_bytes == 0 { self.chunk } else { self.resident_rows };
        if big {
            self.borrow()?;
        }
        let mut run = || -> Result<(usize, Vec<Vec<f32>>)> {
            let mut out = Vec::new();
            let mut done = 0;
            for c in tokens.chunks(rows) {
                if done > 0 && stop(done) {
                    break;
                }
                // every pass's last rows' logits when asked for (the head is a millisecond): a stop may come after any
                out = self.pass(s, c, n_out)?;
                done += c.len();
            }
            Ok((done, out))
        };
        let r = run();
        if big {
            self.give_back()?;
        }
        r
    }

    /// The hot set's tail as the arena: its matrices dropped from the hot set until `give_back`
    fn borrow(&self) -> Result<()> {
        let Some(cache) = &self.cache else { return Ok(()) };
        let mut c = cache.lock().unwrap();
        let Some(hb) = c.hot_buf.as_ref().map(|b| b.view(0, b.len)).transpose()? else { return Ok(()) };
        let start = (hb.len - self.borrow_bytes) & !4095;
        let stride = |k: &Key| match &self.layers[k.0].mix {
            Mix::Moe(m) => if k.2 { m.down.stride } else { m.up.stride },
            _ => 0,
        };
        let mut lent = Vec::new();
        c.hot.retain(|k, off| {
            let st = stride(k);
            if *off + st > start {
                lent.push((*k, st, *off));
                false
            } else {
                true
            }
        });
        drop(c);
        lent.sort_by_key(|j| j.2);
        *self.lent.lock().unwrap() = lent;
        self.arena.set(hb.view(start, hb.len - start)?);
        Ok(())
    }

    /// The arena back on its own buffer, the borrowed tail's matrices read into it again (after the prompt's kernels: the same stream)
    fn give_back(&self) -> Result<()> {
        self.arena.set(self.arena_home.view(0, self.arena_home.len)?);
        let lent = std::mem::take(&mut *self.lent.lock().unwrap());
        if lent.is_empty() {
            return Ok(());
        }
        let t0 = Instant::now();
        self.fill_hot(&lent)?;
        let mut c = self.cache.as_ref().unwrap().lock().unwrap();
        for (k, _, off) in &lent {
            c.hot.insert(*k, *off);
        }
        self.times.lock().unwrap().give_back_s += t0.elapsed().as_secs_f64();
        Ok(())
    }

    /// Matrices (key, stride, offset in the hot buffer; by offset) read from the file into the hot buffer, a few hundred MB at a time
    fn fill_hot(&self, jobs: &[(Key, usize, usize)]) -> Result<()> {
        let hb = {
            let c = self.cache.as_ref().unwrap().lock().unwrap();
            let b = c.hot_buf.as_ref().unwrap();
            b.view(0, b.len)?
        };
        let chunk_bytes = 192usize << 20;
        let mut stage = vec![0u8; chunk_bytes];
        let mut i = 0;
        while i < jobs.len() {
            let start = jobs[i].2;
            let mut j = i;
            while j < jobs.len() && jobs[j].2 + jobs[j].1 - start <= chunk_bytes {
                j += 1;
            }
            let part = &jobs[i..j];
            let sbase = stage.as_mut_ptr() as usize;
            let reqs: Vec<IoReq> = part.iter().filter_map(|&((l, x, down), _, off)| {
                let Mix::Moe(m) = &self.layers[l].mix else { return None };
                let ex = if down { &m.down } else { &m.up };
                Some(self.req(ex, x, sbase + off - start))
            }).collect();
            self.io.run(&reqs).map_err(e)?;
            let end = part.last().map(|q| q.2 + q.1).unwrap();
            hb.write(start, &stage[..end - start])?;
            // the host buffer is reused for the next part: this copy done first
            self.gpu.sync()?;
            i = j;
        }
        Ok(())
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
