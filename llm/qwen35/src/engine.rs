//! Dense Qwen3.5 / Qwen3.8 (`qwen35`, llama.cpp's src/models/qwen35.cpp): 64 layers, three of every four gated
//! DeltaNet (linear attention: a conv + SiLU, the delta-rule recurrence, a SiLU-gated norm), the fourth gated full
//! attention (a sigmoid gate a head in the query projection, q / k norms, rotary positions on the first 64 of 256
//! features), each followed by a dense SwiGLU. The weights stay in their GGUF blocks on the GPU: decode reads them
//! through the Q8_1 products (up to 8 rows: nemotronh's, Strata's and ggml-cuda's), a prompt chunk expands each matrix
//! to half and multiplies (cuBLAS). After nextsycl's qwen35 engine; no silo here.
//!
//! ```text
//!   x = embed(tokens)
//!   per layer:  h = attn_norm(x);  x += linear(h) or full(h);  x += ffn(post_attention_norm(x))
//!   logits = output(output_norm(x))
//! ```
//!
//! MTP (the GGUF's `MTP/` file: its block `layers`): a draft block that reads a position's normed hidden state h with
//! the token that follows it and predicts the one after - eh_proj(enorm(embed(x_p)) | hnorm(h_{p-1})), a gated
//! full-attention layer with its own cache, a SwiGLU, its head norm and head. It runs behind every pass over that
//! pass's rows (so its cache follows the conversation); `step` drafts `NS_Q35_DRAFTS` tokens (the block chained on its
//! own normed output), verifies [token, drafts] in one pass and keeps the longest prefix the sampled tokens match - a
//! rejected row's DeltaNet states come back from the pass's snapshots.

use std::collections::HashMap;
use std::ffi::c_int;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextcuda_core::{Arena, DevBuf, Gpu, HostBuf};
use nextcuda_gguf::{GType, Gguf, Tensor};
use nextcuda_llm::{Error, GpuInfo, Result, Sampler, Tap};

use crate::ffi;

fn e(x: impl std::fmt::Display) -> Error {
    Error(x.to_string())
}

/// The model's shape, from the file's metadata
#[derive(Clone, Debug)]
pub struct Geometry {
    pub layers: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub heads: usize,
    pub heads_kv: usize,
    pub head: usize,
    pub n_rot: usize,
    pub theta: f32,
    pub eps: f32,
    pub conv: usize,
    pub state: usize,
    pub k_heads: usize,
    pub v_heads: usize,
    pub inner: usize,
    pub interval: usize,
    pub vocab: usize,
    pub ctx: usize,
}

impl Geometry {
    pub fn read(f: &Gguf) -> std::result::Result<Geometry, String> {
        let u = |k: &str| f.arch_meta(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| format!("no {}.{k}", f.architecture()));
        let fl = |k: &str| f.arch_meta(k).and_then(|v| v.as_f64()).map(|v| v as f32).ok_or_else(|| format!("no {}.{k}", f.architecture()));
        let nextn = u("nextn_predict_layers").unwrap_or(0);
        let g = Geometry {
            layers: u("block_count")? - nextn,
            n_embd: u("embedding_length")?,
            n_ff: u("feed_forward_length")?,
            heads: u("attention.head_count")?,
            heads_kv: u("attention.head_count_kv")?,
            head: u("attention.key_length")?,
            n_rot: u("rope.dimension_count")?,
            theta: fl("rope.freq_base")?,
            eps: fl("attention.layer_norm_rms_epsilon")?,
            conv: u("ssm.conv_kernel")?,
            state: u("ssm.state_size")?,
            k_heads: u("ssm.group_count")?,
            v_heads: u("ssm.time_step_rank")?,
            inner: u("ssm.inner_size")?,
            interval: u("full_attention_interval").unwrap_or(4),
            // shapes are outermost first: [vocab, n_embd]
            vocab: f.tensor("token_embd.weight").map(|t| t.shape[0] as usize).ok_or("no token_embd.weight")?,
            ctx: u("context_length").unwrap_or(262_144),
        };
        if g.state != 128 || g.inner != g.v_heads * g.state || !g.v_heads.is_multiple_of(g.k_heads) {
            return Err(format!("DeltaNet heads of {} ({} key / {} value heads, inner {}): head 128 and value heads a multiple of the key heads",
                               g.state, g.k_heads, g.v_heads, g.inner));
        }
        if g.head != 256 || !matches!(g.heads / g.heads_kv.max(1), 2 | 4 | 6 | 8) || !g.heads.is_multiple_of(g.heads_kv) {
            return Err(format!("attention heads of {} ({} / {}): head 256 with 2, 4, 6 or 8 query heads a key head", g.head, g.heads, g.heads_kv));
        }
        Ok(g)
    }

    pub fn full(&self, l: usize) -> bool {
        (l + 1).is_multiple_of(self.interval)
    }

    /// conv channels: q, k (key heads) and v (value heads)
    pub fn conv_dim(&self) -> usize {
        2 * self.k_heads * self.state + self.inner
    }
}

/// A decode row set's Q8_1 blocks (the products')
pub struct Q8 {
    std: DevBuf,
}

/// A matrix [rows (outputs), cols (inputs)]: its stored blocks, or float32 (small or unquantized tensors)
pub struct Mat {
    pub buf: DevBuf,
    pub ty: GType,
    pub rows: usize,
    pub cols: usize,
}

struct Full {
    q: Mat,
    k: Mat,
    v: Mat,
    o: Mat,
    q_norm: DevBuf,
    k_norm: DevBuf,
}

struct Linear {
    qkv: Mat,
    z: Mat,
    beta: Mat,
    alpha: Mat,
    /// alpha's rows then beta's in one matrix (both the same block type): one decode product for a row, not two
    ab: Option<Mat>,
    conv: DevBuf,
    dt: DevBuf,
    a: DevBuf,
    norm: DevBuf,
    out: Mat,
}

struct Layer {
    attn_norm: DevBuf,
    post_norm: DevBuf,
    full: Option<Full>,
    linear: Option<Linear>,
    gate: Mat,
    up: Mat,
    down: Mat,
}

/// A conversation: per full-attention layer its key / value cache ([kv heads][max_ctx][256] half), per DeltaNet layer
/// its recurrent state ([48][128][128] float32) and the conv's last inputs ([3][channels])
pub struct Session {
    pub pos: usize,
    pub max_ctx: usize,
    /// a text token's rotary position is its slot less this: the slots pictures took beyond the positions they
    /// advance (a picture of n tokens on an nx x ny grid moves the position by max(nx, ny))
    pub shift: i64,
    layers: Vec<LState>,
    /// rows of the last forward pass
    last_rows: usize,
    /// which session (the engine's snapshots name the one they belong to)
    id: u64,
    /// the draft block's cache and inputs (MTP loaded)
    mtp: Option<MtpState>,
}

/// A session's draft block state: its cache ([kv heads][max_ctx][256] half), the normed hidden state of the last
/// position fed (the next token pairs with it), and the block's own normed output (a chained draft pairs with it)
struct MtpState {
    k: DevBuf,
    v: DevBuf,
    pending: DevBuf,
    chain: DevBuf,
}

enum LState {
    Full { k: DevBuf, v: DevBuf },
    Linear { s: DevBuf, conv: DevBuf },
}

/// A conversation's state in host memory
pub struct Checkpoint {
    pub pos: usize,
    bufs: Vec<Vec<u8>>,
    pub bytes: usize,
}

impl Checkpoint {
    pub fn write_to(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        let u = |w: &mut dyn std::io::Write, x: usize| w.write_all(&(x as u64).to_le_bytes());
        w.write_all(b"NSQ35")?;
        u(w, self.pos)?;
        u(w, self.bufs.len())?;
        for b in &self.bufs {
            u(w, b.len())?;
            w.write_all(b)?;
        }
        Ok(())
    }

    pub fn read_from(r: &mut dyn std::io::Read) -> std::io::Result<Checkpoint> {
        let u = |r: &mut dyn std::io::Read| -> std::io::Result<usize> {
            let mut b = [0u8; 8];
            r.read_exact(&mut b)?;
            Ok(u64::from_le_bytes(b) as usize)
        };
        let mut magic = [0u8; 5];
        r.read_exact(&mut magic)?;
        if &magic != b"NSQ35" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not a qwen35 checkpoint"));
        }
        let pos = u(r)?;
        let n = u(r)?;
        let mut bufs = Vec::with_capacity(n);
        let mut bytes = 0;
        for _ in 0..n {
            let len = u(r)?;
            let mut v = vec![0u8; len];
            r.read_exact(&mut v)?;
            bytes += len;
            bufs.push(v);
        }
        Ok(Checkpoint { pos, bufs, bytes })
    }
}

/// Generation from a fed prompt: the logits to draw the next token from, or the committed token not fed yet
pub struct Decoder {
    logits: Option<Vec<f32>>,
    next: Option<u32>,
    /// draft with the MTP block
    mtp: bool,
    /// the drafts for the positions after `next`
    drafts: Vec<u32>,
    pub drafted: u64,
    pub accepted: u64,
}

impl Decoder {
    pub fn new(logits: Vec<f32>, mtp: bool) -> Decoder {
        Decoder { logits: Some(logits), next: None, mtp, drafts: Vec::new(), drafted: 0, accepted: 0 }
    }
    pub fn after(next: u32, mtp: bool) -> Decoder {
        Decoder { logits: None, next: Some(next), mtp, drafts: Vec::new(), drafted: 0, accepted: 0 }
    }
    pub fn pending(&mut self) -> Option<u32> {
        self.drafts.clear();
        self.next.take()
    }
}

/// The MTP draft block (its own file: block `layers`, its token embedding and head)
struct Mtp {
    file: Box<Gguf>,
    embd: Tensor,
    vocab: usize,
    /// eh_proj's columns for the embedding and for the hidden state (split at load: two products, no concatenation)
    eh_e: Mat,
    eh_h: Mat,
    enorm: DevBuf,
    hnorm: DevBuf,
    head_norm: DevBuf,
    attn_norm: DevBuf,
    post_norm: DevBuf,
    attn: Full,
    gate: Mat,
    up: Mat,
    down: Mat,
    /// its own head (None: the model's - the block in the model's file)
    output: Option<Mat>,
}

/// Each DeltaNet layer's state after each row of a verify pass but the last (its recurrence [rows-1][48][128][128],
/// its conv inputs [rows-1][3][channels]), and whose pass they are (session, first position, rows)
struct Snaps {
    layers: Vec<(DevBuf, DevBuf)>,
    of: Option<(u64, usize, usize)>,
}

/// NS_Q35_DRAFTS: tokens drafted a step (0: none; at most 7 - a verify pass is at most 8 rows)
/// (2: with ggml-cuda's MMVQ a verify row costs more past three - a story's 256 tokens, the 5070 + 3060: drafts 0 30.9
/// tok/s, 1 42.7, 2 44.4, 3 38.2, 5 31.3)
fn draft_count() -> usize {
    std::env::var("NS_Q35_DRAFTS").ok().and_then(|v| v.parse().ok()).unwrap_or(2).min(7)
}

/// NS_Q35_MTP: the draft block's file; else the model's own when it holds the block (blk.<layers>.nextn: ISTA-DASLab's
/// -mtp files), else an `MTP/` folder beside the model's file holding one
fn mtp_path(f: &Gguf, layers: usize) -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("NS_Q35_MTP") {
        return Some(p.into());
    }
    if f.tensor(&format!("blk.{layers}.nextn.eh_proj.weight")).is_some() {
        return f.paths.first().cloned();
    }
    let dir = f.paths.first()?.parent()?.join("MTP");
    let mut v: Vec<_> = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "gguf")).collect();
    v.sort();
    v.into_iter().next()
}

/// One GPU's share: its layers [lb, le), a prompt chunk's work memory and the half buffer its products expand into
struct Dev {
    gpu: Arc<Gpu>,
    arena: Arena,
    /// the largest matrix expanded to half (a prompt chunk's products)
    w16: DevBuf,
    lb: usize,
    le: usize,
    /// pinned rows for the passes' token embeddings, used in turn (each slot's ticket: its last reader), so an upload
    /// never waits for the GPU's queue - the host queues the next chunk while the GPUs are on this one
    uploads: Mutex<Uploads>,
}

/// Slots of pinned host memory a kernel reads in place, with the ticket of the work that last read each
#[derive(Default)]
struct Uploads {
    slots: Vec<(Option<HostBuf>, Option<i64>)>,
    next: usize,
}

/// Upload slots a GPU: as many passes queued ahead of the GPU as there are
const UPLOAD_SLOTS: usize = 4;

/// A picture's tokens in a prompt: ids past the vocabulary, PICTURE | (the picture's key << 12) | its token's index,
/// the key a hash of the picture (so a prefix with another picture is never reused for it). Rows in raster order of
/// the merged patch grid, nx columns by ny rows
pub const PICTURE: u32 = nextcuda_llm::PICTURE_TOKEN;
pub const PICTURE_INDEX_BITS: u32 = nextcuda_llm::PICTURE_INDEX_BITS;

/// What the vision tower made of a picture: its tokens' embeddings [nx * ny, n_embd] (float32, host) and its grid
pub struct Picture {
    pub rows: Vec<f32>,
    pub nx: usize,
    pub ny: usize,
}

/// A picture token's (key, index)
fn picture_of(id: u32) -> Option<(u32, usize)> {
    (id & PICTURE != 0).then(|| (id & !PICTURE & !((1 << PICTURE_INDEX_BITS) - 1), (id & ((1 << PICTURE_INDEX_BITS) - 1)) as usize))
}

/// The rotary positions of a pass's rows: one run from a position (text), or each row's own (temporal, row, column)
/// - pictures' tokens (Qwen3.8's M-RoPE), [3][t] int32 (put on each GPU as it is used)
enum Rope {
    From(i64),
    Rows(Vec<i32>),
}

pub struct Qwen35<'g> {
    pub g: Geometry,
    file: &'g Gguf,
    /// the GPUs in layer order (the first embeds, the last holds the head and the draft block)
    devs: Vec<Dev>,
    /// each layer's GPU (an index into devs)
    layer_dev: Vec<usize>,
    /// the GPU the kernels go to now (set under `work`: a pass walks the GPUs in order)
    cur: std::sync::atomic::AtomicUsize,
    /// host memory a pass's rows cross between GPUs through
    staging: Mutex<Option<HostBuf>>,
    k: &'static ffi::Api,
    layers: Vec<Layer>,
    output_norm: DevBuf,
    output: Mat,
    /// the token embedding's rows stay in the file (read a row a token)
    embd: Tensor,
    /// the rows a forward pass takes at most
    chunk: usize,
    work: Mutex<()>,
    /// the draft block (loaded with drafts on and its file found)
    mtp: Option<Mtp>,
    /// drafts a step
    drafts: usize,
    /// the verify passes' DeltaNet snapshots (under `work`)
    snaps: Mutex<Snaps>,
    /// the normed hidden states of the last pass's rows (up to 8: a rollback's pending state)
    vh: DevBuf,
    next_id: std::sync::atomic::AtomicU64,
    load_s: f64,
    load_bytes: u64,
    /// the pictures prompts name (`add_picture`), by key
    pictures: Mutex<HashMap<u32, Arc<Picture>>>,
    /// the vision tower (a projector beside the model), on the first GPU
    vision: Option<crate::vision::Vision>,
    /// NS_Q35_PROFILE=1: the kernels' times
    prof: Option<Mutex<Profile>>,
    /// the GPU of the head and the draft block (an index into devs)
    head: usize,
    /// the prompt products by MMQ (NS_Q35_MMQ=0: by the half expansion and cuBLAS)
    mmq: bool,
    /// the attention caches in q8_0 (NS_Q35_KV=f16: half)
    kv_q8: bool,
    /// bytes of a head's row in them
    kv_row: usize,
    /// over two GPUs: the two buffers on the last that pipelined chunks' rows cross into (`feed_pipelined`)
    pipe: Vec<DevBuf>,
    /// the decode products by ggml-cuda's MMVQ (NS_Q35_MMVQ=nh: the shared kernels)
    ggml_mmvq: bool,
}

/// The kernels' times by (GPU, phase, kernel): seconds, calls, bytes, arithmetic
#[derive(Default)]
struct Profile {
    phase: &'static str,
    rows: HashMap<(usize, &'static str, &'static str), (f64, u64, f64, f64)>,
}

/// The projector beside the model (`mmproj*.gguf` in its folder), unless NS_Q35_VISION=0
fn vision_path(f: &Gguf) -> Option<std::path::PathBuf> {
    if std::env::var("NS_Q35_VISION").is_ok_and(|v| v == "0") {
        return None;
    }
    let dir = f.paths.first()?.parent()?;
    let mut found: Vec<std::path::PathBuf> = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.contains("mmproj") && n.ends_with(".gguf"))).collect();
    found.sort();
    found.into_iter().next()
}

/// NS_Q35_CHUNK when set; otherwise chosen at load (`Qwen35::load`: 2,048 rows when they fit)
fn chunk_setting() -> Option<usize> {
    std::env::var("NS_Q35_CHUNK").ok().and_then(|v| v.parse().ok()).filter(|v| *v >= 16)
}

/// A prompt chunk's work memory (the arena) for `chunk` rows
fn arena_bytes(g: &Geometry, chunk: usize) -> usize {
    chunk * (4 * g.n_embd + 2 * g.n_ff + g.conv_dim() + g.inner + 4 * g.v_heads * g.state + g.heads * 2 * g.head + 3 * g.heads * g.head) * 4
        + chunk * g.n_ff.max(g.conv_dim()) * 2 + 8 * g.vocab * 4 + (64 << 20)
}

/// The attention cache as llama.cpp's cache-type q8_0 (a head's row 256 int8 and 8 half scales: 272 bytes against
/// 512), unless NS_Q35_KV=f16
fn kv_q8_setting() -> bool {
    !std::env::var("NS_Q35_KV").is_ok_and(|v| v == "f16")
}

/// Bytes of a key (or value) head's row in the cache
fn kv_row_bytes(g: &Geometry, q8: bool) -> usize {
    if q8 { g.head + g.head / 32 * 2 } else { g.head * 2 }
}

/// Room kept for a session when the chunk is chosen (a 32K context's cache on the 27B is 2 GiB)
const SESSION_ROOM: u64 = 5 << 29;

impl<'g> Qwen35<'g> {
    /// Loads the model over `gpus` in order: with one GPU all of it there; with more, the layers split so each GPU keeps
    /// the same share of its memory free for the sessions (NS_SPLIT: the first layer of each GPU after the first), the
    /// head and the draft block on the last. `kv`: the context the sessions reserve (tokens in all; 0: 32K) - a split
    /// plans their caches on the GPUs of their layers.
    pub fn load(f: &'g Gguf, gpus: &[Arc<Gpu>], draft: bool, kv: (usize, usize), log: &mut dyn FnMut(String)) -> Result<Qwen35<'g>> {
        let t0 = Instant::now();
        if gpus.is_empty() {
            return Err(e("no GPU"));
        }
        let g = Geometry::read(f).map_err(e)?;
        let k = ffi::api()?;
        log("qwen35: nextcuda's kernels (no silo)".into());
        // the plan: every tensor but the token embedding, a prompt chunk's work, 768 MiB to spare (cuBLAS's workspace,
        // the context's own)
        let embd = f.tensor("token_embd.weight").ok_or_else(|| e("no token_embd.weight"))?.clone();
        let weights: u64 = f.tensors.iter().filter(|t| t.name != "token_embd.weight" && !t.name.starts_with(&format!("blk.{}.", g.layers))).map(|t| t.bytes).sum();
        let layer_w: Vec<u64> = (0..g.layers).map(|l| {
            let p = format!("blk.{l}.");
            f.tensors.iter().filter(|t| t.name.starts_with(&p)).map(|t| t.bytes).sum()
        }).collect();
        let head_w = weights - layer_w.iter().sum::<u64>();
        // the half buffer a prompt chunk expands a matrix into: with MMQ only the matrices it does not take (the file's
        // IQ1_M, the float32 ones), so each GPU's is its own layers' largest such (without MMQ: the largest matrix)
        let mmq_on = !std::env::var("NS_Q35_MMQ").is_ok_and(|v| v == "0");
        let expands = |t: &Tensor| t.shape.len() == 2 && !(mmq_on && unsafe { (k.mmq_supported)(t.ty.code() as i32) } != 0) && t.shape[0] > 1 && t.shape[1] > 1;
        let layer_x16: Vec<usize> = (0..g.layers).map(|l| {
            let p = format!("blk.{l}.");
            f.tensors.iter().filter(|t| t.name.starts_with(&p) && expands(t)).map(|t| t.elements() as usize * 2).max().unwrap_or(0)
        }).collect();
        // the head's and the draft block's (on the last GPU)
        let tail_x16 = f.tensors.iter().filter(|t| (!t.name.starts_with("blk.") || t.name.starts_with(&format!("blk.{}.", g.layers))) && t.name != "token_embd.weight" && expands(t))
            .map(|t| t.elements() as usize * 2).max().unwrap_or(0);
        let w16_bytes = layer_x16.iter().copied().max().unwrap_or(0).max(tail_x16);
        let mut free = Vec::with_capacity(gpus.len());
        for gpu in gpus {
            let (total, fr) = gpu.memory()?;
            free.push(fr.unwrap_or(total));
        }
        // the prompt chunk: 2,048 rows (each matrix expanded to half once for them: 8K prompts 760 -> 1,100 tok/s against
        // 512) when they leave room for a session, else 1,024, else 512
        // the draft block: its file's tensors but its token embedding (rows read from the file), its snapshots
        let drafts = if draft { draft_count() } else { 0 };
        let mfile = match (drafts > 0).then(|| mtp_path(f, g.layers)).flatten() {
            Some(p) => match Gguf::open(&p) {
                Ok(m) => Some(Box::new(m)),
                Err(x) => {
                    log(format!("qwen35: the MTP file {} does not open ({}); no drafts", p.display(), x.0));
                    None
                }
            },
            None => None,
        };
        let lin_state = (g.v_heads * g.state * g.state + (g.conv - 1) * g.conv_dim()) * 4;
        let snap_bytes = (g.layers - (0..g.layers).filter(|l| g.full(*l)).count()) * drafts * lin_state;
        // in the model's own file: the block's tensors only (its head is the model's)
        let in_file = mfile.as_ref().is_some_and(|m| m.paths.first() == f.paths.first());
        let mtp_w = mfile.as_ref().map_or(0, |m| m.tensors.iter().filter(|t| if in_file { t.name.starts_with(&format!("blk.{}.", g.layers)) } else { t.name != "token_embd.weight" })
            .map(|t| t.bytes).sum::<u64>());
        let mtp_bytes = mtp_w + if mfile.is_some() { snap_bytes as u64 } else { 0 };
        let margin: u64 = 512 << 20;
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        // the vision tower: a Qwen3-VL projector beside the model (NS_Q35_VISION=0: none), its weights on the first GPU
        // (the slower card: a picture is rare, and the faster card's room goes to layers)
        let mmproj = vision_path(f);
        let vision_bytes = match &mmproj {
            Some(p) => Gguf::open(p).map(|m| crate::vision::card_bytes(&m)).unwrap_or(0),
            None => 0,
        };
        // the draft block's pass behind a chunk's: its inputs and outputs beside the main pass's
        let has_mtp = mfile.is_some();
        let arena_of = |c: usize| arena_bytes(&g, c) + if has_mtp { c * 8 * g.n_embd * 4 } else { 0 };
        // the head and the draft block: on the last GPU, or (NS_Q35_HEAD=first) on the first - the slower card when the
        // server orders them weakest first, so the faster one's room goes to layers (a prompt pass runs ~2.7x faster
        // there; decode reads the head and the draft block from the slower card)
        let head_first = gpus.len() > 1 && std::env::var("NS_Q35_HEAD").is_ok_and(|v| v == "first");
        let head_i = if head_first { 0 } else { gpus.len() - 1 };
        let (chunk, bounds) = if gpus.len() == 1 {
            let fixed = weights + mtp_bytes + vision_bytes + w16_bytes as u64 + margin;
            let chunk = chunk_setting().unwrap_or_else(|| {
                [2048, 1024].into_iter().find(|c| fixed + arena_bytes(&g, *c) as u64 + SESSION_ROOM <= free[0]).unwrap_or(512)
            });
            let need = fixed + arena_of(chunk) as u64;
            if need > free[0] {
                return Err(e(format!("{} needs about {:.1} GiB on {} ({:.1} GiB of weights); {:.1} GiB are free", f.architecture(), gib(need),
                                     gpus[0].name, gib(weights), gib(free[0]))));
            }
            (chunk, vec![(0, g.layers)])
        } else {
            // a layer's cost where it lands: its weights, its cache for the planned context (full attention) or its
            // state and verify snapshots (DeltaNet)
            let ctx = if kv.0 > 0 { kv.0 } else { 32768 };
            let mtp_on = has_mtp;
            let row = kv_row_bytes(&g, kv_q8_setting());
            // (a layer with a matrix MMQ does not take: its GPU's half buffer as large as it)
            let cost: Vec<u64> = (0..g.layers).map(|l| layer_w[l] + layer_x16[l] as u64 + if g.full(l) { (2 * g.heads_kv * ctx * row) as u64 } else {
                (lin_state * (1 + if mtp_on { drafts } else { 0 })) as u64
            }).collect();
            let split_env: Option<Vec<usize>> = std::env::var("NS_SPLIT").ok().map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect());
            let plan = |c: usize| -> std::result::Result<Vec<(usize, usize)>, String> {
                let n = gpus.len();
                // what each GPU has for layers: its free memory less the spare, its half buffer and work memory; the
                // last's less the head and the draft block (its weights, its cache)
                let room: Vec<i64> = (0..n).map(|i| {
                    // (and 256 MiB the plan does not see: the allocator's rounding, small buffers)
                    let mut over = margin + (256 << 20) + arena_of(c) as u64;
                    if i == 0 {
                        over += vision_bytes;
                    }
                    if i == head_i {
                        over += head_w + tail_x16 as u64 + if mtp_on { mtp_w + (2 * g.heads_kv * ctx * row) as u64 } else { 0 };
                    }
                    free[i] as i64 - over as i64
                }).collect();
                let mut bounds = Vec::with_capacity(n);
                if let Some(sp) = &split_env {
                    let mut lb = 0;
                    for i in 0..n {
                        let le = if i + 1 == n { g.layers } else { *sp.get(i).ok_or("NS_SPLIT: a first layer for each GPU after the first")? };
                        bounds.push((lb, le.min(g.layers)));
                        lb = le.min(g.layers);
                    }
                } else {
                    // the last GPU first (the one that computes most: the server orders them weakest first), then each
                    // before it: each takes the last layers left while they fit; the first the rest. A GPU the layers
                    // do not reach is not used (a B70 runs a layer ~1.5x faster than a B65: no layer goes to the slower
                    // card while the faster one has room)
                    let mut le = g.layers;
                    let mut rev = Vec::with_capacity(n);
                    for i in (0..n).rev() {
                        let mut lb = le;
                        if i == 0 {
                            lb = 0;
                        } else {
                            let mut have = room[i];
                            while lb > 0 && cost[lb - 1] as i64 <= have {
                                have -= cost[lb - 1] as i64;
                                lb -= 1;
                            }
                        }
                        rev.push((lb, le));
                        le = lb;
                    }
                    rev.reverse();
                    bounds = rev;
                }
                for (i, &(lb, le)) in bounds.iter().enumerate() {
                    let need: u64 = cost[lb..le].iter().sum();
                    if le <= lb {
                        // a GPU before the last with no layers left for it is dropped (`load`)
                        if i + 1 < n && split_env.is_none() {
                            continue;
                        }
                        return Err(format!("{} gets no layers", gpus[i].name));
                    }
                    if need as i64 > room[i] {
                        return Err(format!("layers {lb}-{} need {:.1} GiB on {}; {:.1} GiB are left for them", le - 1, gib(need), gpus[i].name,
                                           gib(room[i].max(0) as u64)));
                    }
                }
                Ok(bounds)
            };
            let tries: Vec<usize> = chunk_setting().map_or_else(|| vec![2048, 1024, 512], |c| vec![c]);
            let mut last_err = String::new();
            // of the chunks that fit, the one that leaves the last GPU (the fastest: the server orders them weakest
            // first) the most layers - the work goes to the faster card, the slower one takes what is left; a larger
            // chunk on a tie (measured, the 5070 + 3060 at 80K: 2,048 rows 35 layers there 707 tok/s, 512 rows 40
            // layers 732)
            let mut found: Option<(usize, Vec<(usize, usize)>)> = None;
            for c in tries {
                match plan(c) {
                    Ok(b) => {
                        let last_layers = |b: &[(usize, usize)]| b.last().map_or(0, |x| x.1 - x.0);
                        if found.as_ref().is_none_or(|(_, fb)| last_layers(&b) > last_layers(fb)) {
                            found = Some((c, b));
                        }
                    }
                    Err(x) => last_err = x,
                }
            }
            found.ok_or_else(|| e(format!("{} over {} GPUs at a {ctx}-token context: {last_err}", f.architecture(), gpus.len())))?
        };
        let given = gpus.len();
        let (gpus, bounds): (Vec<Arc<Gpu>>, Vec<(usize, usize)>) = gpus.iter().cloned().zip(bounds).filter(|(_, b)| b.1 > b.0).unzip();
        if gpus.len() < given {
            log(format!("qwen35: all of it fits {}: {} of the {given} GPUs not used", gpus.iter().map(|g| g.name.clone()).collect::<Vec<_>>().join(" + "),
                        given - gpus.len()));
        }
        let mut layer_dev = vec![0usize; g.layers];
        for (i, &(lb, le)) in bounds.iter().enumerate() {
            for d in &mut layer_dev[lb..le] {
                *d = i;
            }
        }
        // (a GPU dropped by the plan is never the first: the head's index is kept)
        let head = if head_first { 0 } else { gpus.len() - 1 };
        let lastg = gpus[head].clone();
        let mut loaded = 0u64;
        let mut layers = Vec::with_capacity(g.layers);
        for l in 0..g.layers {
            let gpu = &gpus[layer_dev[l]];
            let n = |s: &str| format!("blk.{l}.{s}.weight");
            let mut read = |name: &str| -> Result<Mat> { vec_mat(f, gpu, k, name, &mut loaded) };
            let full = if g.full(l) {
                Some(Full { q: read(&n("attn_q"))?, k: read(&n("attn_k"))?, v: read(&n("attn_v"))?, o: read(&n("attn_output"))?,
                            q_norm: read(&n("attn_q_norm"))?.buf, k_norm: read(&n("attn_k_norm"))?.buf })
            } else {
                None
            };
            let linear = if g.full(l) {
                None
            } else {
                Some(Linear {
                    qkv: read(&n("attn_qkv"))?,
                    z: read(&n("attn_gate"))?,
                    beta: read(&n("ssm_beta"))?,
                    alpha: read(&n("ssm_alpha"))?,
                    ab: None,
                    conv: read(&n("ssm_conv1d"))?.buf,
                    dt: read(&format!("blk.{l}.ssm_dt.bias"))?.buf,
                    a: read(&format!("blk.{l}.ssm_a"))?.buf,
                    norm: read(&n("ssm_norm"))?.buf,
                    out: read(&n("ssm_out"))?,
                })
            };
            layers.push(Layer {
                attn_norm: read(&n("attn_norm"))?.buf,
                post_norm: read(&n("post_attention_norm"))?.buf,
                full,
                linear,
                gate: read(&n("ffn_gate"))?,
                up: read(&n("ffn_up"))?,
                down: read(&n("ffn_down"))?,
            });
            if l % 16 == 15 {
                log(format!("qwen35: layer {} of {} ({:.1} GiB, {:.0} s)", l + 1, g.layers, loaded as f64 / (1u64 << 30) as f64, t0.elapsed().as_secs_f64()));
            }
        }
        // alpha and beta stacked (their bytes one after the other) where they share a block type
        for ly in layers.iter_mut() {
            if let Some(a) = ly.linear.as_mut() {
                if a.alpha.ty == a.beta.ty && a.alpha.cols == a.beta.cols && a.alpha.ty != GType::F32 {
                    let (na, nb) = (a.alpha.buf.len, a.beta.buf.len);
                    let buf = DevBuf::new(&a.alpha.buf.gpu().clone(), na + nb)?;
                    buf.copy_within(0, &a.alpha.buf, 0, na)?;
                    buf.copy_within(na, &a.beta.buf, 0, nb)?;
                    a.ab = Some(Mat { buf, ty: a.alpha.ty, rows: a.alpha.rows + a.beta.rows, cols: a.alpha.cols });
                }
            }
        }
        let output_norm = vec_mat(f, &lastg, k, "output_norm.weight", &mut loaded)?.buf;
        let output = vec_mat(f, &lastg, k, if f.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" }, &mut loaded)?;
        let mtp = match mfile {
            Some(m) => Some(load_mtp(m, &g, &lastg, k, in_file, &mut loaded)?),
            None => None,
        };
        let drafts = if mtp.is_some() { drafts } else { 0 };
        let mut snap_layers = Vec::new();
        if drafts > 0 {
            for l in (0..g.layers).filter(|l| !g.full(*l)) {
                let gpu = &gpus[layer_dev[l]];
                snap_layers.push((DevBuf::f32(gpu, drafts * g.v_heads * g.state * g.state)?, DevBuf::f32(gpu, drafts * (g.conv - 1) * g.conv_dim())?));
            }
        }
        log(match &mtp {
            Some(_) => format!("qwen35: MTP drafts ({drafts} a step)"),
            None if draft => "qwen35: no MTP file (NS_Q35_MTP or an MTP/ folder beside the model): no drafts".into(),
            None => "qwen35: drafts off".into(),
        });
        let vh = DevBuf::f32(&lastg, 8 * g.n_embd)?;
        let mut devs = Vec::with_capacity(gpus.len());
        for (i, (gpu, &(lb, le))) in gpus.iter().zip(&bounds).enumerate() {
            gpu.sync()?;
            // its half buffer: its own layers' largest matrix to expand (the last GPU: the head's and the draft block's too)
            let own = layer_x16[lb..le].iter().copied().max().unwrap_or(0).max(if i == head { tail_x16 } else { 0 }).max(1 << 20);
            devs.push(Dev { gpu: gpu.clone(), w16: DevBuf::new(gpu, own)?, arena: Arena::new(gpu, arena_of(chunk))?, lb, le,
                                uploads: Mutex::new(Uploads { slots: (0..UPLOAD_SLOTS).map(|_| (None, None)).collect(), next: 0 }) });
        }
        let load_s = t0.elapsed().as_secs_f64();
        let wh = devs.iter().map(|d| if d.lb == 0 && d.le == g.layers { d.gpu.name.clone() } else { format!("{} (layers {}-{})", d.gpu.name, d.lb, d.le - 1) })
            .collect::<Vec<_>>().join(", ");
        log(format!("qwen35: {} layers of {} ({} full attention), {:.1} GiB on {wh} in {load_s:.0} s; prompt chunks of {chunk}", g.layers, g.n_embd,
                    (0..g.layers).filter(|l| g.full(*l)).count(), gib(loaded)));
        let vision = match &mmproj {
            Some(p) => match crate::vision::Vision::load(p, &devs[0].gpu, &mut *log) {
                Ok(v) => Some(v),
                Err(x) => {
                    log(format!("qwen35: the vision tower {} does not load ({}); no pictures", p.display(), x.0));
                    None
                }
            },
            None => None,
        };
        let kv_row = kv_row_bytes(&g, kv_q8_setting());
        let pipe = if devs.len() == 2 && head == 1 {
            (0..2).map(|_| DevBuf::new(&devs[1].gpu, chunk * g.n_embd * 4)).collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        Ok(Qwen35 { g, file: f, devs, layer_dev, cur: 0.into(), staging: Mutex::new(None), k, layers, output_norm, output, embd, chunk,
                    work: Mutex::new(()), mtp, drafts, snaps: Mutex::new(Snaps { layers: snap_layers, of: None }), vh, next_id: 0.into(), load_s,
                    load_bytes: loaded, pictures: Mutex::new(HashMap::new()), vision,
                    mmq: !std::env::var("NS_Q35_MMQ").is_ok_and(|v| v == "0"), head, kv_q8: kv_q8_setting(), kv_row, pipe,
                    ggml_mmvq: !std::env::var("NS_Q35_MMVQ").is_ok_and(|v| v == "nh"),
                    prof: ["NS_Q35_PROFILE", "NS_PROFILE"].iter().any(|k| std::env::var(k).is_ok_and(|v| v == "1")).then(|| Mutex::new(Profile::default())) })
    }

    /// A kernel call (its return code checked): timed on the device with NS_Q35_PROFILE=1, with the bytes it moves
    /// and the arithmetic it does
    fn op(&self, name: &'static str, key: impl FnOnce() -> String, bytes: usize, flops: usize, f: impl FnOnce() -> c_int) -> Result<()> {
        let rc = self.op_rc(name, key, bytes, flops, f)?;
        ffi::check(rc, name)
    }

    /// `op`, its return code left to the caller. NS_Q35_PROFILE=1: each kernel timed (the GPU synced after it: the
    /// times are a pass's shares, not its speed), by its phase and name, with the bytes and arithmetic it does
    fn op_rc(&self, name: &'static str, _key: impl FnOnce() -> String, bytes: usize, flops: usize, f: impl FnOnce() -> c_int) -> Result<c_int> {
        let Some(prof) = &self.prof else { return Ok(f()) };
        let t0 = Instant::now();
        let rc = f();
        self.dev().gpu.sync()?;
        let dt = t0.elapsed().as_secs_f64();
        let mut p = prof.lock().unwrap();
        let phase = p.phase;
        let dev = self.cur.load(std::sync::atomic::Ordering::Relaxed);
        let r = p.rows.entry((dev, phase, name)).or_default();
        r.0 += dt;
        r.1 += 1;
        r.2 += bytes as f64;
        r.3 += flops as f64;
        Ok(rc)
    }

    /// What the next kernels are part of (the profile groups by it)
    fn phase(&self, p: &'static str) {
        if let Some(prof) = &self.prof {
            prof.lock().unwrap().phase = p;
        }
    }

    pub fn profile_lines(&self) -> Vec<String> {
        let Some(prof) = &self.prof else { return Vec::new() };
        let mut p = prof.lock().unwrap();
        let total: f64 = p.rows.values().map(|r| r.0).sum();
        let mut rows: Vec<_> = p.rows.iter().map(|(k, v)| (*k, *v)).collect();
        rows.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
        let mut out = vec![format!("qwen35 profile: {total:.3} s in kernels (synced after each)")];
        for d in &self.devs {
            let mib = |b: usize| b as f64 / (1u64 << 20) as f64;
            out.push(format!("  {}: work memory {:.0} of {:.0} MiB used at most, {} spills", d.gpu.name,
                             mib(d.arena.peak.load(std::sync::atomic::Ordering::Relaxed)), mib(d.arena.len()),
                             d.arena.spills.load(std::sync::atomic::Ordering::Relaxed)));
        }
        for ((dev, phase, name), (t, n, b, fl)) in rows.into_iter().take(40) {
            out.push(format!("  gpu{dev} {phase:<12} {name:<14} {:>8.3} s {:>5.1}% {n:>7} calls {:>7.1} GB/s {:>7.1} TFLOP/s", t, 100.0 * t / total.max(1e-12),
                             b / t.max(1e-12) / 1e9, fl / t.max(1e-12) / 1e12));
        }
        p.rows.clear();
        out
    }

    pub fn load_seconds(&self) -> f64 {
        self.load_s
    }
    pub fn load_bytes(&self) -> u64 {
        self.load_bytes
    }
    pub fn prefill_chunk(&self) -> usize {
        self.chunk
    }

    /// The GPU the kernels go to now
    fn dev(&self) -> &Dev {
        &self.devs[self.cur.load(std::sync::atomic::Ordering::Relaxed)]
    }

    fn on(&self, i: usize) {
        self.cur.store(i, std::sync::atomic::Ordering::Relaxed);
    }

    /// the GPU of the head and the draft block
    fn last(&self) -> usize {
        self.head
    }

    fn arena(&self) -> &Arena {
        &self.dev().arena
    }

    fn raw(&self) -> *mut std::ffi::c_void {
        self.dev().gpu.raw()
    }

    /// x (rows of the GPU it is on) onto GPU `to`: a buffer of that GPU's arena
    fn hop(&self, x: &DevBuf, to: usize) -> Result<DevBuf> {
        self.on(to);
        let y = self.arena().bytes(x.len)?;
        let mut st = self.staging.lock().unwrap();
        if st.as_ref().is_none_or(|b| b.len < x.len) {
            // a larger staging area: the copies through the old one finished first
            for d in &self.devs {
                d.gpu.sync()?;
            }
            *st = Some(HostBuf::new(&self.devs[self.layer_dev[0]].gpu, x.len.next_power_of_two())?);
        }
        // ordered on the GPUs only: after the GPU before (its layers) and this one's queue, before what either does
        // next - the host goes on to queue more (the next chunk on the first GPU while this one is on these rows)
        y.copy_from_peer_async(x, st.as_mut().unwrap())?;
        Ok(y)
    }

    /// y [t, w.rows] = x [t, w.cols] . W^T; `role` names it
    /// `xq`: x's Q8_1 blocks when they were made already (decode rows), so the product does not quantize x again
    fn matmul(&self, role: &'static str, w: &Mat, x: &DevBuf, xq: Option<&Q8>, t: usize, y: &DevBuf) -> Result<()> {
        let k = self.k;
        let (n, kk) = (w.rows as i64, w.cols as i64);
        let (rows, cols) = (w.rows, w.cols);
        let key = || format!("{role} {} {rows}x{cols}", w.ty.name());
        let wb = w.ty.bytes((rows * cols) as u64).unwrap_or(0) as usize;
        let flops = 2 * t * rows * cols;
        if w.ty == GType::F32 && t <= 8 {
            // SAFETY: the matrix [rows, cols] float32, x [t, cols], y [t, rows].
            return self.op("gemv f32", key, 4 * (t * cols + rows * cols + t * rows), flops,
                           || unsafe { (k.gemv_f32)(self.raw(), w.buf.fp(), x.fp(), y.fp(), t as i64, n, kk) });
        }
        if w.ty == GType::F32 {
            // a prompt chunk: the float32 matrix (DeltaNet's alpha / beta, BF16 in the file: half holds them exactly) to
            // half, then the half product - the row-at-a-time gemv was most of a prompt pass
            let elems = rows * cols;
            if elems * 2 > self.dev().w16.len {
                return Err(e(format!("a {rows}x{cols} matrix past the half buffer")));
            }
            let xh = self.arena().bytes(t * cols * 2)?;
            // SAFETY: the matrix -> half in w16, x [t, cols] -> half; y [t, rows] = xh . w16^T.
            self.op("to half", key, 6 * elems, 0, || unsafe { (k.to_half)(self.raw(), w.buf.fp(), self.dev().w16.ptr(), elems as i64) })?;
            self.op("x to half", key, 6 * t * cols, 0, || unsafe { (k.to_half)(self.raw(), x.fp(), xh.ptr(), (t * cols) as i64) })?;
            return self.op("gemm f16", key, 2 * t * cols + 2 * elems + 4 * t * rows, flops,
                           || unsafe { (k.gemm_f16)(self.raw(), self.dev().w16.ptr(), xh.ptr(), y.fp(), t as i64, n, kk) });
        }
        // SAFETY: arithmetic only.
        let ggml_mmvq = self.ggml_mmvq && unsafe { (k.q35_mmvq_supported)(w.ty.code() as i32) } != 0;
        let mmvq = ggml_mmvq || unsafe { (k.mmvq_supported)(w.ty.code() as i32) } != 0;
        if t <= 8 && mmvq {
            // SAFETY: arithmetic only.
            let q8 = unsafe { (k.q8_1_bytes)(kk, t as i64) };
            let own;
            let qb = match xq {
                Some(q) => q,
                None => {
                    own = self.quantize(x, cols, t, &key)?;
                    &own
                }
            };
            // SAFETY: the matrix in its blocks, x's Q8_1 blocks, y [t, rows].
            // ggml-cuda's MMVQ where it takes the type (NS_Q35_MMVQ=nh: the shared kernels)
            let f = if ggml_mmvq { k.q35_mmvq } else { k.mmvq };
            return self.op("mmvq", key, wb + q8 + 4 * t * rows, flops,
                           || unsafe { f(self.raw(), w.ty.code() as i32, w.buf.ptr(), qb.std.ptr(), y.fp(), kk, n, t as i64) });
        }
        // a prompt's rows: ggml-cuda's MMQ from the blocks (int8 tensor cores; NS_Q35_MMQ=0: the half product)
        if self.mmq && unsafe { (k.mmq_supported)(w.ty.code() as i32) } != 0 {
            // SAFETY: arithmetic only; then x [t, cols] -> its q8_1 tiles; y [t, rows] = x . W^T.
            let qb = self.arena().bytes(unsafe { (k.mmq_q8_bytes)(kk, t as i64) })?;
            self.op("mmq quantize", key, 4 * t * cols + t * cols, 0,
                    || unsafe { (k.mmq_quantize)(self.raw(), w.ty.code() as i32, x.fp(), t as i64, kk, qb.ptr()) })?;
            return self.op("mmq", key, wb + t * cols + 4 * t * rows, flops,
                           || unsafe { (k.mmq)(self.raw(), w.ty.code() as i32, w.buf.ptr(), n, kk, qb.ptr(), t as i64, y.fp()) });
        }
        // the matrix expanded to half (directly, or through float32), x to half, then the half product
        let elems = rows * cols;
        if elems * 2 > self.dev().w16.len {
            return Err(e(format!("a {rows}x{cols} matrix past the half buffer")));
        }
        // SAFETY: the matrix's blocks -> rows x cols halfs in w16 (non-zero: a type it does not expand directly)
        let rc = self.op_rc("dequant f16", key, wb + 2 * elems, 0,
                            || unsafe { (k.dequant_f16)(self.raw(), w.ty.code() as i32, w.buf.ptr(), elems as i64, self.dev().w16.ptr()) })?;
        if rc != 0 {
            let f = self.arena().f32(elems)?;
            // SAFETY: the blocks -> float32, then -> half.
            self.op("dequant", key, wb + 4 * elems, 0, || unsafe { (k.dequant_f32)(self.raw(), w.ty.code() as i32, w.buf.ptr(), elems as i64, f.fp()) })?;
            self.op("to half", key, 6 * elems, 0, || unsafe { (k.to_half)(self.raw(), f.fp(), self.dev().w16.ptr(), elems as i64) })?;
        }
        let xh = self.arena().bytes(t * cols * 2)?;
        // SAFETY: x [t, cols] -> half; then y [t, rows] = xh . w16^T.
        self.op("x to half", key, 6 * t * cols, 0, || unsafe { (k.to_half)(self.raw(), x.fp(), xh.ptr(), (t * cols) as i64) })?;
        self.op("gemm f16", key, 2 * t * cols + 2 * elems + 4 * t * rows, flops,
                || unsafe { (k.gemm_f16)(self.raw(), self.dev().w16.ptr(), xh.ptr(), y.fp(), t as i64, n, kk) })
    }

    /// The RMS norm into y (the products after it quantize it themselves)
    fn norm(&self, x: &DevBuf, w: &DevBuf, y: &DevBuf, rows: usize, c: usize) -> Result<Option<Q8>> {
        self.rms(x, w, y, rows, c)?;
        // decode rows: quantized once here for the several products that read them
        if rows <= 8 && c.is_multiple_of(32) {
            return Ok(Some(self.quantize(y, c, rows, &String::new)?));
        }
        Ok(None)
    }

    /// x [t, cols] -> its Q8_1 blocks
    fn quantize(&self, x: &DevBuf, cols: usize, t: usize, key: &dyn Fn() -> String) -> Result<Q8> {
        let k = self.k;
        // SAFETY: arithmetic only.
        let q8 = unsafe { (k.q8_1_bytes)(cols as i64, t as i64) };
        let std = self.arena().bytes(q8)?;
        // SAFETY: x [t, cols] float32 -> its Q8_1 blocks.
        self.op("quantize", key, 4 * t * cols + q8, 0, || unsafe { (k.quantize)(self.raw(), x.fp(), t as i64, cols as i64, std.ptr()) })?;
        Ok(Q8 { std })
    }

    fn rms(&self, x: &DevBuf, w: &DevBuf, y: &DevBuf, rows: usize, c: usize) -> Result<()> {
        // SAFETY: x and y [rows, c], w c floats.
        self.op("rms norm", String::new, 8 * rows * c + 4 * c, 4 * rows * c,
                || unsafe { (self.k.rms_norm)(self.raw(), x.fp(), w.fp(), y.fp(), rows as i64, c as i64, self.g.eps) })
    }

    fn add(&self, y: &DevBuf, x: &DevBuf, n: usize) -> Result<()> {
        // SAFETY: n floats each.
        self.op("add", String::new, 12 * n, n, || unsafe { (self.k.add)(self.raw(), y.fp(), x.fp(), n as i64) })
    }

    /// `embed_from` with pictures' tokens: their rows the picture's (the vision tower's), the others the table's
    fn embed_with(&self, file: &Gguf, embd: &Tensor, vocab: usize, tokens: &[u32]) -> Result<DevBuf> {
        if !tokens.iter().any(|t| t & PICTURE != 0) {
            return self.embed_from(file, embd, vocab, tokens);
        }
        let plain: Vec<u32> = tokens.iter().map(|t| if t & PICTURE != 0 { 0 } else { *t }).collect();
        let x = self.embed_from(file, embd, vocab, &plain)?;
        let ne = self.g.n_embd;
        let pics = self.pictures.lock().unwrap();
        // runs of one picture's consecutive tokens, one write each
        let mut j = 0;
        while j < tokens.len() {
            let Some((key, i)) = picture_of(tokens[j]) else {
                j += 1;
                continue;
            };
            let p = pics.get(&key).ok_or_else(|| e(format!("picture {key:#x}: not given to the engine")))?;
            let mut n = 1;
            while j + n < tokens.len() && picture_of(tokens[j + n]) == Some((key, i + n)) {
                n += 1;
            }
            if i + n > p.nx * p.ny {
                return Err(e(format!("picture token {} of {}", i + n - 1, p.nx * p.ny)));
            }
            let bytes: Vec<u8> = p.rows[i * ne..(i + n) * ne].iter().flat_map(|v| v.to_le_bytes()).collect();
            x.write(j * ne * 4, &bytes)?;
            j += n;
        }
        Ok(x)
    }

    /// Rows of a token embedding kept in its file
    fn embed_from(&self, file: &Gguf, embd: &Tensor, vocab: usize, tokens: &[u32]) -> Result<DevBuf> {
        let g = &self.g;
        let row = embd.ty.bytes(g.n_embd as u64).ok_or_else(|| e("the embedding's row size"))? as usize;
        let n = tokens.len() * row;
        // the rows into the GPU's next pinned slot (once the work that read it last is done), read there in place
        let d = self.dev();
        let mut up = d.uploads.lock().unwrap();
        let i = up.next;
        up.next = (i + 1) % up.slots.len();
        let slot = &mut up.slots[i];
        if let Some(t) = slot.1.take() {
            d.gpu.wait_ticket(t)?;
        }
        if slot.0.as_ref().is_none_or(|b| b.len < n) {
            slot.0 = Some(HostBuf::new(&d.gpu, n.next_power_of_two())?);
        }
        let hb = slot.0.as_mut().unwrap();
        {
            let raw = hb.as_mut_slice();
            for (i, t) in tokens.iter().enumerate() {
                if *t as usize >= vocab {
                    return Err(e(format!("token {t} past the vocabulary ({vocab})")));
                }
                file.read_into(embd, *t as u64 * row as u64, &mut raw[i * row..(i + 1) * row]).map_err(|x| e(x.0))?;
            }
        }
        let src = hb.device_view(0, n)?;
        let x = self.arena().f32(tokens.len() * g.n_embd)?;
        // SAFETY: the rows' blocks (pinned host memory, read over PCIe) -> float32.
        self.op("dequant", String::new, n + 4 * tokens.len() * g.n_embd, 0,
                || unsafe { (self.k.dequant_f32)(self.raw(), embd.ty.code() as i32, src.ptr(), (tokens.len() * g.n_embd) as i64, x.fp()) })?;
        slot.1 = d.gpu.ticket()?;
        Ok(x)
    }

    /// A full-attention layer on h [t, n_embd]: its output [t, n_embd]
    #[allow(clippy::too_many_arguments)]
    fn full(&self, a: &Full, kc: &DevBuf, vc: &DevBuf, cap: usize, h: &DevBuf, hq8: Option<&Q8>, t: usize, p0: usize, rope: &Rope) -> Result<DevBuf> {
        let (g, k) = (&self.g, self.k);
        let (hq, hk, d) = (g.heads, g.heads_kv, g.head);
        let qf = self.arena().f32(t * hq * 2 * d)?;
        let kr = self.arena().f32(t * hk * d)?;
        let vr = self.arena().f32(t * hk * d)?;
        self.matmul("q", &a.q, h, hq8, t, &qf)?;
        self.matmul("k", &a.k, h, hq8, t, &kr)?;
        self.matmul("v", &a.v, h, hq8, t, &vr)?;
        let q = self.arena().f32(t * hq * d)?;
        let kn = self.arena().f32(t * hk * d)?;
        let att = self.arena().f32(t * hq * d)?;
        let none = String::new;
        // the keys the rows see in all (causal): what attention reads and multiplies
        let keys = t * p0 + t * (t + 1) / 2;
        // a picture's rows: their positions on this GPU
        let pos3 = match rope {
            Rope::Rows(p) => {
                let b = self.arena().bytes(p.len() * 4)?;
                b.write(0, &p.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
                Some(b)
            }
            Rope::From(_) => None,
        };
        let from = match rope {
            Rope::From(f) => *f,
            Rope::Rows(_) => 0,
        };
        // SAFETY: the rows and caches as named in q35.h.
        unsafe {
            for (src, ss, hs, dst, heads, w, name) in [(&qf, hq * 2 * d, 2 * d, &q, hq, &a.q_norm, "q norm rope"), (&kr, hk * d, d, &kn, hk, &a.k_norm, "k norm rope")] {
                self.op(name, none, 8 * t * heads * d, 10 * t * heads * d, || match &pos3 {
                    Some(p) => (k.qk_norm_mrope)(self.raw(), src.fp(), ss as i64, hs as i64, dst.fp(), (heads * d) as i64, t as i64, heads as i64, d as i64,
                                                 w.fp(), g.eps, g.n_rot as i64, g.theta, p.ptr().cast()),
                    None => (k.qk_norm_rope)(self.raw(), src.fp(), ss as i64, hs as i64, dst.fp(), (heads * d) as i64, t as i64, heads as i64, d as i64,
                                             w.fp(), g.eps, g.n_rot as i64, g.theta, from),
                })?;
            }
            self.op("kv store", none, 12 * t * hk * d, 0,
                    || (if self.kv_q8 { k.kv_store_q8 } else { k.kv_store })(self.raw(), kn.fp(), (hk * d) as i64, vr.fp(), (hk * d) as i64, t as i64, hk as i64, d as i64, cap as i64, p0 as i64,
                                    kc.ptr(), vc.ptr()))?;
            let sc = (k.attn_scratch)(t as i64, hq as i64, hk as i64, d as i64, p0 as i64) as usize;
            let part = if sc > 0 { Some(self.arena().f32(sc)?) } else { None };
            self.op("attention", none, keys * hk * d * 4 + 8 * t * hq * d, 4 * keys * hq * d,
                    || (if self.kv_q8 { k.attn_q8 } else { k.attn })(self.raw(), q.fp(), (hq * d) as i64, kc.ptr(), vc.ptr(), t as i64, hq as i64, hk as i64, d as i64, cap as i64, p0 as i64,
                                att.fp(), part.as_ref().map_or(std::ptr::null_mut(), |p| p.fp())))?;
            self.op("output gate", none, 12 * t * hq * d, 4 * t * hq * d,
                    || (k.gate_mul)(self.raw(), att.fp(), qf.fp(), (hq * 2 * d) as i64, t as i64, hq as i64, d as i64))?;
        }
        let out = self.arena().f32(t * g.n_embd)?;
        self.matmul("o", &a.o, &att, None, t, &out)?;
        Ok(out)
    }

    /// A DeltaNet layer on h [t, n_embd]: its output [t, n_embd]
    /// `snap`: where the state after each row but the last goes (a verify pass's: its recurrence's, its conv's)
    #[allow(clippy::too_many_arguments)]
    fn linear(&self, a: &Linear, s: &DevBuf, conv: &DevBuf, h: &DevBuf, hq8: Option<&Q8>, t: usize, snap: Option<(&DevBuf, &DevBuf)>) -> Result<DevBuf> {
        let (g, k) = (&self.g, self.k);
        let (hv, hk, d, cd) = (g.v_heads, g.k_heads, g.state, g.conv_dim());
        let qkv = self.arena().f32(t * cd)?;
        let z = self.arena().f32(t * g.inner)?;
        self.matmul("qkv", &a.qkv, h, hq8, t, &qkv)?;
        self.matmul("z", &a.z, h, hq8, t, &z)?;
        // one row: alpha and beta from their stacked matrix in one product
        let (alpha, beta) = match &a.ab {
            Some(ab) if t == 1 => {
                let both = self.arena().f32(2 * hv)?;
                self.matmul("alpha beta", ab, h, hq8, t, &both)?;
                (both.view(0, hv * 4)?, both.view(hv * 4, hv * 4)?)
            }
            _ => {
                let beta = self.arena().f32(t * hv)?;
                let alpha = self.arena().f32(t * hv)?;
                self.matmul("beta", &a.beta, h, hq8, t, &beta)?;
                self.matmul("alpha", &a.alpha, h, hq8, t, &alpha)?;
                (alpha, beta)
            }
        };
        let eg = self.arena().f32(t * hv * d)?;
        let co = self.arena().f32(t * cd)?;
        let q = self.arena().f32(t * hv * d)?;
        let kk = self.arena().f32(t * hv * d)?;
        let v = self.arena().f32(t * hv * d)?;
        let o = self.arena().f32(t * hv * d)?;
        let y = self.arena().f32(t * g.inner)?;
        let qd = (hk * d) as i64;
        let none = String::new;
        let thd = t * hv * d;
        // SAFETY: the buffers as named in glm.h / q35.h.
        unsafe {
            self.op("gates", none, 4 * (3 * t * hv + 2 * hv) + 4 * thd, 20 * t * hv,
                    || (k.gdn_gates)(self.raw(), alpha.fp(), beta.fp(), a.dt.fp(), a.a.fp(), eg.fp(), t as i64, hv as i64, d as i64))?;
            self.op("conv silu", none, 4 * (2 * t * cd + 2 * (g.conv - 1) * cd + g.conv * cd), 2 * g.conv * t * cd + 4 * t * cd,
                    || (k.conv_silu)(self.raw(), qkv.fp(), conv.fp(), a.conv.fp(), co.fp(), t as i64, cd as i64, g.conv as i32, snap.map_or(std::ptr::null_mut(), |x| x.1.fp())))?;
            self.op("expand q", none, 4 * t * hk * d + 4 * thd, 0, || (k.expand)(self.raw(), co.fp(), cd as i64, q.fp(), t as i64, hk as i64, hv as i64, d as i64))?;
            self.op("expand k", none, 4 * t * hk * d + 4 * thd, 0,
                    || (k.expand)(self.raw(), co.fp().add(qd as usize), cd as i64, kk.fp(), t as i64, hk as i64, hv as i64, d as i64))?;
            self.op("expand v", none, 8 * thd, 0, || (k.expand)(self.raw(), co.fp().add(2 * qd as usize), cd as i64, v.fp(), t as i64, hv as i64, hv as i64, d as i64))?;
            self.op("l2 q", none, 8 * thd, 3 * thd, || (k.l2_norm)(self.raw(), q.fp(), (t * hv) as i64, d as i64, g.eps))?;
            self.op("l2 k", none, 8 * thd, 3 * thd, || (k.l2_norm)(self.raw(), kk.fp(), (t * hv) as i64, d as i64, g.eps))?;
            // the state [hv][d][d] read and written once, each row's q, k, v, decay in and output out; per row and head
            // the state decayed, read against k, updated, read against q: ~8 d^2
            self.op("delta rule", none, 8 * hv * d * d + 4 * (5 * thd + t * hv), 8 * t * hv * d * d,
                    || (k.kda_scan)(self.raw(), q.fp(), kk.fp(), v.fp(), eg.fp(), beta.fp(), s.fp(), o.fp(), t as i64, hv as i64, d as i64, snap.map_or(std::ptr::null_mut(), |x| x.0.fp())))?;
            self.op("gated norm", none, 4 * (3 * t * g.inner + d), 10 * t * g.inner,
                    || (k.gdn_out)(self.raw(), o.fp(), z.fp(), g.inner as i64, a.norm.fp(), y.fp(), t as i64, hv as i64, d as i64, g.eps))?;
        }
        let out = self.arena().f32(t * g.n_embd)?;
        self.matmul("out", &a.out, &y, None, t, &out)?;
        Ok(out)
    }

    /// One pass of at most a chunk: the last `n_out` rows' logits
    fn pass(&self, s: &mut Session, tokens: &[u32], n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
        let _w = self.work.lock().unwrap();
        self.pass_in(s, tokens, n_out, tap)
    }

    /// `pass` with the work lock held: a verify pass (2 ..= drafts + 1 rows, drafts on) keeps the DeltaNet
    /// snapshots; with MTP the draft block runs over the rows behind it
    fn pass_in(&self, s: &mut Session, tokens: &[u32], n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
        let t = tokens.len();
        self.check_rows(s, s.pos, t)?;
        for d in &self.devs {
            d.arena.reset();
        }
        self.on(0);
        let p0 = s.pos;
        let (rope, shift_after) = self.rope_of(tokens, p0, s.shift)?;
        let mut snaps = self.snaps.lock().unwrap();
        let snapping = (2..=self.drafts + 1).contains(&t) && !snaps.layers.is_empty();
        snaps.of = None;
        self.phase("embed");
        let x = self.embed_with(self.file, &self.embd, self.g.vocab, tokens)?;
        let x = self.run_layers(s, 0..self.g.layers, x, t, p0, &rope, snapping.then_some(&*snaps), &mut *tap)?;
        if snapping {
            snaps.of = Some((s.id, p0, t));
        }
        drop(snaps);
        s.pos += t;
        s.shift = shift_after;
        s.last_rows = t;
        self.finish(s, x, tokens, p0, &rope, n_out, tap)
    }

    /// A pass of t rows at p0 fits the chunk and the context
    fn check_rows(&self, s: &Session, p0: usize, t: usize) -> Result<()> {
        if t == 0 {
            return Err(e("an empty pass"));
        }
        if t > self.chunk {
            return Err(e(format!("a pass of {t} rows; it takes {}", self.chunk)));
        }
        if p0 + t > s.max_ctx {
            return Err(e(format!("the context is full ({} of {})", p0 + t, s.max_ctx)));
        }
        Ok(())
    }

    /// Layers `range` over the rows x [t, n_embd] at p0 (on the current GPU, its arena from where it is; the rows
    /// cross to the next GPU where its layers start): the rows after them. `snaps`: a verify pass's DeltaNet snapshots
    #[allow(clippy::too_many_arguments)]
    fn run_layers(&self, s: &Session, range: std::ops::Range<usize>, mut x: DevBuf, t: usize, p0: usize, rope: &Rope, snaps: Option<&Snaps>,
                  tap: Tap) -> Result<DevBuf> {
        let g = &self.g;
        let mut h = self.arena().f32(t * g.n_embd)?;
        let mut ffn_g = self.arena().f32(t * g.n_ff)?;
        let mut ffn_u = self.arena().f32(t * g.n_ff)?;
        let mut ffn = self.arena().f32(t * g.n_embd)?;
        // the DeltaNet layers before the range (their snapshots' index)
        let mut li = (0..range.start).filter(|l| !g.full(*l)).count();
        // the arena past this point is the layer's own, made again each layer
        for l in range {
            let (ly, st) = (&self.layers[l], &s.layers[l]);
            // the next GPU's layers: the rows cross over, its own buffers for them
            if self.layer_dev[l] != self.cur.load(std::sync::atomic::Ordering::Relaxed) {
                self.phase("GPU to GPU");
                x = self.hop(&x, self.layer_dev[l])?;
                h = self.arena().f32(t * g.n_embd)?;
                ffn_g = self.arena().f32(t * g.n_ff)?;
                ffn_u = self.arena().f32(t * g.n_ff)?;
                ffn = self.arena().f32(t * g.n_embd)?;
            }
            let mark = self.arena().mark();
            self.phase(if ly.full.is_some() { "attention" } else { "deltanet" });
            let hq8 = self.norm(&x, &ly.attn_norm, &h, t, g.n_embd)?;
            let out = match (st, &ly.full, &ly.linear) {
                (LState::Full { k, v }, Some(a), _) => self.full(a, k, v, s.max_ctx, &h, hq8.as_ref(), t, p0, rope)?,
                (LState::Linear { s: rs, conv }, _, Some(a)) => {
                    let sn = snaps.map(|sn| (&sn.layers[li].0, &sn.layers[li].1));
                    li += 1;
                    self.linear(a, rs, conv, &h, hq8.as_ref(), t, sn)?
                }
                _ => return Err(e(format!("layer {l}: its state and its weights disagree"))),
            };
            self.add(&x, &out, t * g.n_embd)?;
            self.phase("ffn");
            let hq8 = self.norm(&x, &ly.post_norm, &h, t, g.n_embd)?;
            self.matmul("gate", &ly.gate, &h, hq8.as_ref(), t, &ffn_g)?;
            self.matmul("up", &ly.up, &h, hq8.as_ref(), t, &ffn_u)?;
            let nf = t * g.n_ff;
            // SAFETY: n floats each.
            self.op("swiglu", String::new, 12 * nf, 6 * nf,
                    || unsafe { (self.k.swiglu)(self.raw(), ffn_g.fp(), ffn_u.fp(), ffn_g.fp(), nf as i64, f32::INFINITY) })?;
            self.matmul("down", &ly.down, &ffn_g, None, t, &ffn)?;
            self.add(&x, &ffn, t * g.n_embd)?;
            tap(&format!("l_out-{l}"), &x)?;
            self.arena().rewind(mark);
        }
        Ok(x)
    }

    /// A pass's end over its rows x after the last layer (the session's position already past them): the draft block
    /// over them (MTP), the head's logits for the last n_out rows (0: no head, nothing read back)
    #[allow(clippy::too_many_arguments)]
    fn finish(&self, s: &Session, mut x: DevBuf, tokens: &[u32], p0: usize, rope: &Rope, n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
        let g = &self.g;
        let t = tokens.len();
        // the output norm: the last rows' for the head; every row's with MTP (the draft block reads them). n_out 0: no
        // head (a prompt's chunk before its last: the draft block's rows only)
        let no_head = n_out == 0;
        let n = n_out.clamp(1, t);
        let ne = g.n_embd;
        // the head's GPU (NS_Q35_HEAD=first: back to the first)
        if self.cur.load(std::sync::atomic::Ordering::Relaxed) != self.head {
            self.phase("GPU to GPU");
            x = self.hop(&x, self.head)?;
        }
        self.phase("head");
        let (normed, nq8) = if let Some(ms) = s.mtp.as_ref().filter(|_| self.mtp.is_some()) {
            let all = self.arena().f32(t * ne)?;
            let q = self.norm(&x, &self.output_norm, &all, t, ne)?;
            tap("result_norm", &all.view((t - n) * ne * 4, n * ne * 4)?)?;
            // the rows kept for a rollback; the draft block over this pass's rows: row i pairs token i with the hidden
            // state before it (the pending one, then this pass's)
            self.vh.copy_within(0, &all, 0, t.min(8) * ne * 4)?;
            self.phase("mtp");
            let hs = self.arena().f32(t * ne)?;
            hs.copy_within(0, &ms.pending, 0, ne * 4)?;
            if t > 1 {
                hs.copy_within(ne * 4, &all, 0, (t - 1) * ne * 4)?;
            }
            self.mtp_rows(ms, tokens, &hs, p0, rope, s.max_ctx, false)?;
            ms.pending.copy_within(0, &all, (t - 1) * ne * 4, ne * 4)?;
            if no_head {
                return Ok(Vec::new());
            }
            self.phase("head");
            // the head reads the rows' Q8_1 forms when it takes them all; fewer rows (a short prompt's last) it quantizes
            // itself (the split layout does not slice by rows)
            (all.view((t - n) * ne * 4, n * ne * 4)?, if n == t { q } else { None })
        } else {
            if no_head {
                return Ok(Vec::new());
            }
            let last = x.view((t - n) * ne * 4, n * ne * 4)?;
            let normed = self.arena().f32(n * ne)?;
            let q = self.norm(&last, &self.output_norm, &normed, n, ne)?;
            tap("result_norm", &normed)?;
            (normed, q)
        };
        let logits = self.arena().f32(n * self.output.rows)?;
        self.matmul("output", &self.output, &normed, nq8.as_ref(), n, &logits)?;
        let all = logits.to_f32()?;
        Ok(all.chunks(self.output.rows).map(|c| c.to_vec()).collect())
    }

    /// A prompt over two GPUs as a pipeline (llama.cpp's over its ubatches): the first GPU's layers of chunk k+1 queued
    /// before the last GPU's layers (and the draft block, the head) of chunk k, so the first always has a chunk queued
    /// while the host waits on the last's full launch queue; the rows cross into one of two buffers on the last (the
    /// one chunk k-2 used). Each layer's state for chunk k+1 needs only its own after chunk k: the same sums as one
    /// pass a chunk. None: not this prompt's (one GPU, the head first, pictures) - `feed_until` reads it a pass a chunk
    fn feed_pipelined(&self, s: &mut Session, tokens: &[u32], stop: &(dyn Fn(usize) -> bool + Sync), tap: Tap) -> Result<Option<(usize, Vec<f32>)>> {
        if self.devs.len() != 2 || self.head != 1 || self.pipe.len() != 2 || tokens.len() <= self.chunk || tokens.iter().any(|t| t & PICTURE != 0) {
            return Ok(None);
        }
        let _w = self.work.lock().unwrap();
        self.snaps.lock().unwrap().of = None;
        let split = self.devs[1].lb;
        let ne = self.g.n_embd;
        // (chunk, its p0, its rows on the last GPU) waiting for the last GPU's part
        let mut pending: Option<(&[u32], usize, DevBuf)> = None;
        let mut queued = 0;
        let mut logits = Vec::new();
        // the next chunk's first position (the session's moves on as the last GPU's parts are queued)
        let mut next = s.pos;
        for (k, c) in tokens.chunks(self.chunk).enumerate() {
            if queued > 0 && stop(queued) {
                // (stopped early: what is queued is read in full, no logits)
                break;
            }
            let (t, p0) = (c.len(), next);
            self.check_rows(s, p0, t)?;
            next += t;
            // the first GPU's part of chunk k
            self.devs[0].arena.reset();
            self.on(0);
            self.phase("embed");
            let x = self.embed_with(self.file, &self.embd, self.g.vocab, c)?;
            let rope = Rope::From(p0 as i64 - s.shift);
            let x = self.stage(s, 0..split, x, t, p0, &rope, k > 0, &mut *tap)?;
            self.phase("GPU to GPU");
            let xb = self.pipe[k % 2].view(0, t * ne * 4)?;
            {
                let mut st = self.staging.lock().unwrap();
                if st.as_ref().is_none_or(|b| b.len < x.len) {
                    for d in &self.devs {
                        d.gpu.sync()?;
                    }
                    *st = Some(HostBuf::new(&self.devs[0].gpu, (self.chunk * ne * 4).next_power_of_two())?);
                }
                xb.copy_from_peer_async(&x, st.as_mut().unwrap())?;
            }
            queued += t;
            // the last GPU's part of chunk k-1
            if let Some((pc, pp0, px)) = pending.take() {
                self.last_part(s, pc, pp0, px, 0, k > 1, &mut *tap)?;
            }
            pending = Some((c, p0, xb));
        }
        if let Some((pc, pp0, px)) = pending.take() {
            let all = queued == tokens.len();
            logits = self.last_part(s, pc, pp0, px, if all { 1 } else { 0 }, true, &mut *tap)?.pop().unwrap_or_default();
        }
        Ok(Some((queued, logits)))
    }

    /// `run_layers` over one GPU's layers, as one graph when `graph` (NS_Q35_GRAPHS=0: never; not while profiling):
    /// the stage's thousand kernels one entry in the GPU's queue. A capture that fails is run again as launches
    #[allow(clippy::too_many_arguments)]
    fn stage(&self, s: &Session, range: std::ops::Range<usize>, x: DevBuf, t: usize, p0: usize, rope: &Rope, graph: bool, tap: Tap) -> Result<DevBuf> {
        let graph = graph && self.prof.is_none() && !std::env::var("NS_Q35_GRAPHS").is_ok_and(|v| v == "0");
        if graph {
            let mark = self.arena().mark();
            let xv = x.view(0, x.len)?;
            match self.dev().gpu.graph(|| self.run_layers(s, range.clone(), xv, t, p0, rope, None, &mut *tap))? {
                Ok(y) => return Ok(y),
                Err(why) => {
                    static SAID: std::sync::Once = std::sync::Once::new();
                    SAID.call_once(|| eprintln!("[qwen35: a prompt stage as a graph failed ({why}); its kernels launched one by one]"));
                    self.arena().rewind(mark);
                }
            }
        }
        self.run_layers(s, range, x, t, p0, rope, None, tap)
    }

    /// The last GPU's part of a pipelined chunk (rows x there, at p0 = the session's position): its layers, the
    /// session past them, the pass's end
    #[allow(clippy::too_many_arguments)]
    fn last_part(&self, s: &mut Session, tokens: &[u32], p0: usize, x: DevBuf, n_out: usize, graph: bool, tap: Tap) -> Result<Vec<Vec<f32>>> {
        let t = tokens.len();
        debug_assert_eq!(p0, s.pos);
        self.devs[1].arena.reset();
        self.on(1);
        let rope = Rope::From(p0 as i64 - s.shift);
        let x = self.stage(s, self.devs[1].lb..self.g.layers, x, t, p0, &rope, graph, &mut *tap)?;
        s.pos += t;
        s.last_rows = t;
        self.finish(s, x, tokens, p0, &rope, n_out, tap)
    }

    /// The draft block over `tokens` at positions p0.. (row i: token i with hidden state hs[i]): its cache written
    /// there; with `head`, the last row's logits (and its normed output kept for a chained draft)
    /// the last row's likeliest token (`head`: its logits' argmax, taken on the GPU - 4 bytes read back, not the logits)
    fn mtp_rows(&self, ms: &MtpState, tokens: &[u32], hs: &DevBuf, p0: usize, rope: &Rope, cap: usize, head: bool) -> Result<Option<u32>> {
        let Some(m) = &self.mtp else { return Ok(None) };
        let g = &self.g;
        let (t, ne) = (tokens.len(), g.n_embd);
        let emb = self.embed_with(&m.file, &m.embd, m.vocab, tokens)?;
        let en = self.arena().f32(t * ne)?;
        self.rms(&emb, &m.enorm, &en, t, ne)?;
        let hn = self.arena().f32(t * ne)?;
        self.rms(hs, &m.hnorm, &hn, t, ne)?;
        // eh_proj . [en | hn] = W_e . en + W_h . hn
        let cur = self.arena().f32(t * ne)?;
        self.matmul("mtp eh e", &m.eh_e, &en, None, t, &cur)?;
        let tmp = self.arena().f32(t * ne)?;
        self.matmul("mtp eh h", &m.eh_h, &hn, None, t, &tmp)?;
        self.add(&cur, &tmp, t * ne)?;
        let h = self.arena().f32(t * ne)?;
        let hq8 = self.norm(&cur, &m.attn_norm, &h, t, ne)?;
        let att = self.full(&m.attn, &ms.k, &ms.v, cap, &h, hq8.as_ref(), t, p0, rope)?;
        self.add(&cur, &att, t * ne)?;
        let hq8 = self.norm(&cur, &m.post_norm, &h, t, ne)?;
        let fg = self.arena().f32(t * g.n_ff)?;
        let fu = self.arena().f32(t * g.n_ff)?;
        self.matmul("mtp gate", &m.gate, &h, hq8.as_ref(), t, &fg)?;
        self.matmul("mtp up", &m.up, &h, hq8.as_ref(), t, &fu)?;
        let nf = t * g.n_ff;
        // SAFETY: n floats each.
        self.op("swiglu", String::new, 12 * nf, 6 * nf, || unsafe { (self.k.swiglu)(self.raw(), fg.fp(), fu.fp(), fg.fp(), nf as i64, f32::INFINITY) })?;
        self.matmul("mtp down", &m.down, &fg, None, t, &tmp)?;
        self.add(&cur, &tmp, t * ne)?;
        if !head {
            return Ok(None);
        }
        let last = cur.view((t - 1) * ne * 4, ne * 4)?;
        let hn2 = self.arena().f32(ne)?;
        let q = self.norm(&last, &m.head_norm, &hn2, 1, ne)?;
        ms.chain.copy_within(0, &hn2, 0, ne * 4)?;
        let head = m.output.as_ref().unwrap_or(&self.output);
        let logits = self.arena().f32(head.rows)?;
        self.matmul("mtp head", head, &hn2, q.as_ref(), 1, &logits)?;
        let best = self.arena().bytes(4)?;
        // SAFETY: logits [1, vocab] float32 -> its argmax (one int32).
        self.op("argmax", String::new, head.rows * 4, 0,
                || unsafe { (self.k.argmax)(self.raw(), logits.fp(), 1, head.rows as i64, best.ptr().cast()) })?;
        let mut b = [0u8; 4];
        best.read(0, &mut b)?;
        Ok(Some(u32::from_le_bytes(b)))
    }

    /// Drafts for the positions after `tail` (committed at s.pos, not fed): the draft block on (the last fed
    /// position's hidden state, tail), then chained on its own normed output and each draft
    fn draft_in(&self, s: &Session, tail: u32) -> Result<Vec<u32>> {
        let Some(ms) = &s.mtp else { return Ok(Vec::new()) };
        let k = self.drafts.min(s.max_ctx.saturating_sub(s.pos + 1));
        let mut out = Vec::with_capacity(k);
        let mut tok = tail;
        // the draft block is on the last GPU
        self.on(self.last());
        for i in 0..k {
            self.arena().reset();
            self.phase("mtp draft");
            let hs = if i == 0 { &ms.pending } else { &ms.chain };
            let Some(t) = self.mtp_rows(ms, &[tok], hs, s.pos + i, &Rope::From((s.pos + i) as i64 - s.shift), s.max_ctx, true)? else { break };
            tok = t;
            out.push(tok);
        }
        Ok(out)
    }

    /// Back to the first `keep` rows of the session's last pass: its DeltaNet states from the pass's snapshots, the
    /// attention caches and the draft block's cache simply rewritten later
    fn rollback_in(&self, s: &mut Session, keep: usize) -> Result<()> {
        if keep == s.last_rows {
            return Ok(());
        }
        let mut snaps = self.snaps.lock().unwrap();
        let Some((id, p0, rows)) = snaps.of else {
            return Err(e("qwen35: a rollback needs the last pass's snapshots (a verify pass of 2 ..= drafts + 1 rows, drafts on)"));
        };
        if id != s.id || p0 + rows != s.pos || keep == 0 || keep >= rows {
            return Err(e(format!("qwen35: a rollback to {keep} of a pass of {rows} rows at {p0} (session {id}); this session is at {}", s.pos)));
        }
        let g = &self.g;
        let (sb, cb) = (g.v_heads * g.state * g.state * 4, (g.conv - 1) * g.conv_dim() * 4);
        let mut li = 0;
        for st in &s.layers {
            if let LState::Linear { s: rs, conv } = st {
                let (ss, cs) = &snaps.layers[li];
                rs.copy_within(0, ss, (keep - 1) * sb, sb)?;
                conv.copy_within(0, cs, (keep - 1) * cb, cb)?;
                li += 1;
            }
        }
        if let Some(ms) = &s.mtp {
            ms.pending.copy_within(0, &self.vh, (keep - 1) * g.n_embd * 4, g.n_embd * 4)?;
        }
        snaps.of = None;
        s.pos = p0 + keep;
        s.last_rows = keep;
        Ok(())
    }

    pub fn session(&self, max_ctx: usize) -> Result<Session> {
        let (g, row) = (&self.g, self.kv_row);
        // the caches and states, checked against what is free first (a clear error before an allocation fails)
        let mtp_kv = if self.mtp.is_some() { 2 * g.heads_kv * max_ctx * row } else { 0 };
        // per GPU: its layers' share (the draft block's cache on the last)
        for (i, d) in self.devs.iter().enumerate() {
            let full = (d.lb..d.le).filter(|l| g.full(*l)).count();
            let kv = if i == self.last() { mtp_kv } else { 0 };
            let need = (full * 2 * g.heads_kv * max_ctx * row + kv + (d.le - d.lb - full) * (g.v_heads * g.state * g.state + (g.conv - 1) * g.conv_dim()) * 4) as u64;
            let (total, free) = d.gpu.memory()?;
            let free = free.unwrap_or(total);
            if need + (512 << 20) > free {
                return Err(e(format!("a session of {max_ctx} tokens needs {:.2} GiB on {}; {:.2} GiB are free (512 MiB kept spare)", need as f64 / (1u64 << 30) as f64,
                                     d.gpu.name, free as f64 / (1u64 << 30) as f64)));
            }
        }
        let mut layers = Vec::with_capacity(g.layers);
        for l in 0..g.layers {
            let gpu = &self.devs[self.layer_dev[l]].gpu;
            layers.push(if g.full(l) {
                let n = g.heads_kv * max_ctx * row;
                LState::Full { k: DevBuf::new(gpu, n)?, v: DevBuf::new(gpu, n)? }
            } else {
                let s = DevBuf::f32(gpu, g.v_heads * g.state * g.state)?;
                let conv = DevBuf::f32(gpu, (g.conv - 1) * g.conv_dim())?;
                s.fill(0)?;
                conv.fill(0)?;
                LState::Linear { s, conv }
            });
        }
        let mtp = match &self.mtp {
            Some(_) => {
                let n = g.heads_kv * max_ctx * row;
                let gpu = &self.devs[self.last()].gpu;
                let ms = MtpState { k: DevBuf::new(gpu, n)?, v: DevBuf::new(gpu, n)?, pending: DevBuf::f32(gpu, g.n_embd)?, chain: DevBuf::f32(gpu, g.n_embd)? };
                ms.pending.fill(0)?;
                Some(ms)
            }
            None => None,
        };
        let id = self.next_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(Session { pos: 0, max_ctx, shift: 0, layers, last_rows: 0, id, mtp })
    }

    pub fn reset_session(&self, s: &mut Session) -> Result<()> {
        for st in &s.layers {
            if let LState::Linear { s, conv } = st {
                s.fill(0)?;
                conv.fill(0)?;
            }
        }
        if let Some(ms) = &s.mtp {
            ms.pending.fill(0)?;
        }
        s.pos = 0;
        s.shift = 0;
        s.last_rows = 0;
        Ok(())
    }

    pub fn copy_session(&self, dst: &mut Session, src: &Session) -> Result<()> {
        if src.pos > dst.max_ctx {
            return Err(e(format!("a session of {} tokens into one of {}", src.pos, dst.max_ctx)));
        }
        let (g, row) = (&self.g, self.kv_row);
        for (d, s) in dst.layers.iter().zip(&src.layers) {
            match (d, s) {
                (LState::Full { k: dk, v: dv }, LState::Full { k: sk, v: sv }) => {
                    for h in 0..g.heads_kv {
                        let n = src.pos * row;
                        dk.copy_within(h * dst.max_ctx * row, sk, h * src.max_ctx * row, n)?;
                        dv.copy_within(h * dst.max_ctx * row, sv, h * src.max_ctx * row, n)?;
                    }
                }
                (LState::Linear { s: ds, conv: dc }, LState::Linear { s: ss, conv: sc }) => {
                    ds.copy_within(0, ss, 0, ss.len)?;
                    dc.copy_within(0, sc, 0, sc.len)?;
                }
                _ => return Err(e("sessions of other shapes")),
            }
        }
        if let (Some(d), Some(sm)) = (&dst.mtp, &src.mtp) {
            for h in 0..g.heads_kv {
                let n = src.pos * row;
                d.k.copy_within(h * dst.max_ctx * row, &sm.k, h * src.max_ctx * row, n)?;
                d.v.copy_within(h * dst.max_ctx * row, &sm.v, h * src.max_ctx * row, n)?;
            }
            d.pending.copy_within(0, &sm.pending, 0, g.n_embd * 4)?;
        }
        dst.pos = src.pos;
        dst.shift = src.shift;
        dst.last_rows = 0;
        Ok(())
    }

    pub fn feed_until(&self, s: &mut Session, tokens: &[u32], stop: &(dyn Fn(usize) -> bool + Sync), tap: Tap) -> Result<(usize, Vec<f32>)> {
        if !std::env::var("NS_Q35_PIPE").is_ok_and(|v| v == "0") {
            if let Some(r) = self.feed_pipelined(s, tokens, stop, &mut *tap)? {
                return Ok(r);
            }
        }
        let mut logits = Vec::new();
        let mut done = 0;
        for c in tokens.chunks(self.chunk) {
            if done > 0 && stop(done) {
                // (stopped early: no logits - the chunk before had no head)
                break;
            }
            // a chunk before the last: no head (nothing read back), so the host queues the next chunk at once and
            // the GPUs run side by side (the first on the next rows while the last is on these)
            let last = done + c.len() == tokens.len();
            logits = self.pass(s, c, if last { 1 } else { 0 }, &mut *tap)?.pop().unwrap_or_default();
            done += c.len();
        }
        Ok((done, logits))
    }

    pub fn forward_rows(&self, s: &mut Session, tokens: &[u32], n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
        self.pass(s, tokens, n_out, tap)
    }

    pub fn rollback(&self, s: &mut Session, keep: usize) -> Result<()> {
        let _w = self.work.lock().unwrap();
        self.rollback_in(s, keep)
    }

    /// The attention caches in q8_0 (else half)
    pub fn kv_q8(&self) -> bool {
        self.kv_q8
    }

    pub fn has_draft(&self) -> bool {
        self.drafts > 0
    }

    pub fn max_verify(&self) -> usize {
        self.drafts + 1
    }

    pub fn forward_batch(&self, sessions: &mut [&mut Session], tokens: &[u32], tap: Tap) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(tokens.len());
        for (s, t) in sessions.iter_mut().zip(tokens) {
            out.push(self.pass(s, &[*t], 1, &mut *tap)?.pop().unwrap_or_default());
        }
        Ok(out)
    }

    /// The next committed token(s). With drafts: [token, drafts] in one pass, each row's token drawn as plain
    /// sampling would draw it, the drafts kept while they match (what is committed is distributed as without drafts)
    pub fn step(&self, s: &mut Session, d: &mut Decoder, smp: &mut dyn Sampler, tap: Tap) -> Result<Vec<u32>> {
        let _w = self.work.lock().unwrap();
        let drafting = d.mtp && self.drafts > 0 && s.mtp.is_some();
        let out = match (d.logits.take(), d.next.take()) {
            (Some(l), _) => vec![smp.sample(&l)],
            (None, Some(x)) => {
                let mut rows = vec![x];
                if drafting {
                    rows.extend(d.drafts.iter().take(s.max_ctx.saturating_sub(s.pos + 1)));
                }
                d.drafts.clear();
                let logits = self.pass_in(s, &rows, rows.len(), tap)?;
                let mut out = Vec::with_capacity(rows.len());
                for (i, l) in logits.iter().enumerate() {
                    let y = smp.sample(l);
                    out.push(y);
                    if i + 1 < rows.len() {
                        d.drafted += 1;
                        if y == rows[i + 1] {
                            d.accepted += 1;
                            continue;
                        }
                    }
                    break;
                }
                if out.len() < rows.len() {
                    self.rollback_in(s, out.len())?;
                }
                out
            }
            (None, None) => return Err(e("a decoder with nothing to draw from")),
        };
        let tail = *out.last().unwrap();
        d.next = Some(tail);
        if drafting {
            d.drafts = self.draft_in(s, tail)?;
        }
        Ok(out)
    }

    pub fn save(&self, s: &Session) -> Result<Checkpoint> {
        let (g, row) = (&self.g, self.kv_row);
        let mut bufs = Vec::new();
        for st in &s.layers {
            match st {
                LState::Full { k, v } => {
                    for c in [k, v] {
                        let mut b = vec![0u8; g.heads_kv * s.pos * row];
                        for h in 0..g.heads_kv {
                            let n = s.pos * row;
                            c.read(h * s.max_ctx * row, &mut b[h * n..(h + 1) * n])?;
                        }
                        bufs.push(b);
                    }
                }
                LState::Linear { s: rs, conv } => {
                    for c in [rs, conv] {
                        let mut b = vec![0u8; c.len];
                        c.read(0, &mut b)?;
                        bufs.push(b);
                    }
                }
            }
        }
        if let Some(ms) = &s.mtp {
            for c in [&ms.k, &ms.v] {
                let mut b = vec![0u8; g.heads_kv * s.pos * row];
                for h in 0..g.heads_kv {
                    let n = s.pos * row;
                    c.read(h * s.max_ctx * row, &mut b[h * n..(h + 1) * n])?;
                }
                bufs.push(b);
            }
            let mut b = vec![0u8; g.n_embd * 4];
            ms.pending.read(0, &mut b)?;
            bufs.push(b);
        }
        // the rotary shift last (a checkpoint without it: none)
        bufs.push(s.shift.to_le_bytes().to_vec());
        let bytes = bufs.iter().map(|b| b.len()).sum();
        Ok(Checkpoint { pos: s.pos, bufs, bytes })
    }

    pub fn restore(&self, s: &mut Session, ck: &Checkpoint) -> Result<()> {
        let (g, row) = (&self.g, self.kv_row);
        if ck.pos > s.max_ctx {
            return Err(e(format!("a checkpoint of {} tokens into a session of {}", ck.pos, s.max_ctx)));
        }
        let mut it = ck.bufs.iter();
        for st in &s.layers {
            match st {
                LState::Full { k, v } => {
                    for c in [k, v] {
                        let b = it.next().ok_or_else(|| e("a short checkpoint"))?;
                        let n = ck.pos * row;
                        if b.len() != g.heads_kv * n {
                            return Err(e("a checkpoint of another shape"));
                        }
                        for h in 0..g.heads_kv {
                            c.write(h * s.max_ctx * row, &b[h * n..(h + 1) * n])?;
                        }
                    }
                }
                LState::Linear { s: rs, conv } => {
                    for c in [rs, conv] {
                        let b = it.next().ok_or_else(|| e("a short checkpoint"))?;
                        if b.len() != c.len {
                            return Err(e("a checkpoint of another shape"));
                        }
                        c.write(0, b)?;
                    }
                }
            }
        }
        if let Some(ms) = &s.mtp {
            for c in [&ms.k, &ms.v] {
                let b = it.next().ok_or_else(|| e("a checkpoint without the draft block's cache"))?;
                let n = ck.pos * row;
                if b.len() != g.heads_kv * n {
                    return Err(e("a checkpoint of another shape"));
                }
                for h in 0..g.heads_kv {
                    c.write(h * s.max_ctx * row, &b[h * n..(h + 1) * n])?;
                }
            }
            let b = it.next().ok_or_else(|| e("a checkpoint without the draft block's state"))?;
            ms.pending.write(0, b)?;
        }
        s.shift = it.next().filter(|b| b.len() == 8).map_or(0, |b| i64::from_le_bytes(b[..8].try_into().unwrap()));
        s.pos = ck.pos;
        s.last_rows = 0;
        Ok(())
    }

    /// A picture prompts may name by its key (its tokens: `PICTURE | key | index`); the same key again replaces it
    pub fn add_picture(&self, key: u32, p: Picture) -> Result<()> {
        if p.nx * p.ny == 0 || p.nx * p.ny > 1 << PICTURE_INDEX_BITS || p.rows.len() != p.nx * p.ny * self.g.n_embd {
            return Err(e(format!("a picture of {} x {} tokens and {} values (at most {} tokens of {})", p.nx, p.ny, p.rows.len(), 1 << PICTURE_INDEX_BITS, self.g.n_embd)));
        }
        let mut m = self.pictures.lock().unwrap();
        // the most recent ones kept (a picture is n x 20 KiB of host memory)
        if m.len() >= 64 {
            m.clear();
        }
        m.insert(key & !PICTURE & !((1 << PICTURE_INDEX_BITS) - 1), Arc::new(p));
        Ok(())
    }

    /// Whether pictures can be given (a vision tower loaded)
    pub fn sees(&self) -> bool {
        self.vision.is_some()
    }

    /// A picture (RGB8 rows, w x h) through the vision tower, kept under `key` for the prompts that name its tokens:
    /// its token grid (columns, rows). Resized as llama.cpp's mtmd resizes it (sides to 32s, 8 ..= 4,096 tokens,
    /// bicubic), fewer tokens when the first GPU's work memory is smaller than a picture that size needs
    pub fn add_picture_rgb(&self, key: u32, rgb: &[u8], w: usize, h: usize) -> Result<(usize, usize)> {
        let v = self.vision.as_ref().ok_or_else(|| e("this model has no vision tower (no mmproj beside it)"))?;
        if rgb.len() != w * h * 3 || w == 0 || h == 0 {
            return Err(e(format!("a {w}x{h} picture of {} bytes (RGB: {})", rgb.len(), w * h * 3)));
        }
        // a picture already seen (its key): as it was
        if let Some(p) = self.pictures.lock().unwrap().get(&(key & !PICTURE & !((1 << PICTURE_INDEX_BITS) - 1))) {
            return Ok((p.nx, p.ny));
        }
        let _w = self.work.lock().unwrap();
        // the tower's GPU: the first (its activations in that GPU's prompt arena)
        let at = 0;
        let arena = &self.devs[at].arena;
        let (mut tw, mut th) = crate::vision::target_size(w, h);
        // the work memory a picture needs: fewer tokens when the arena is smaller (it spills to the card otherwise)
        while v.work_bytes((tw / crate::vision::PATCH) * (th / crate::vision::PATCH)) > arena.len() && tw * th > crate::vision::MIN_TOKENS * 1024 {
            let a = crate::vision::ALIGN;
            (tw, th) = (((tw * 9 / 10) / a).max(1) * a, ((th * 9 / 10) / a).max(1) * a);
        }
        let pic = nextcuda_diffusion::picture::Picture { width: w as u32, height: h as u32, channels: 3, data: rgb.to_vec() };
        let sized = if (tw, th) == (w, h) { pic } else { pic.resize_bicubic(tw as u32, th as u32) };
        let f: Vec<f32> = sized.data.iter().map(|b| *b as f32).collect();
        arena.reset();
        let nsd = nextcuda_diffusion::kernels::Nsd::new(&self.devs[at].gpu)?;
        let alloc = |n: usize| arena.bytes(n);
        let (rows, (nx, ny)) = v.see(&nsd, &f, th, tw, &alloc)?;
        drop(nsd);
        arena.reset();
        if v.out != self.g.n_embd {
            return Err(e(format!("the vision tower gives {} wide tokens; the model takes {}", v.out, self.g.n_embd)));
        }
        self.add_picture(key, Picture { rows, nx, ny })?;
        Ok((nx, ny))
    }

    /// The rows' rotary positions from slot p0 with the session's shift, and the shift after them
    fn rope_of(&self, tokens: &[u32], p0: usize, shift: i64) -> Result<(Rope, i64)> {
        if !tokens.iter().any(|t| t & PICTURE != 0) {
            return Ok((Rope::From(p0 as i64 - shift), shift));
        }
        let t = tokens.len();
        let mut pos = vec![0i32; 3 * t];
        let mut shift = shift;
        let pics = self.pictures.lock().unwrap();
        for (j, tok) in tokens.iter().enumerate() {
            let slot = (p0 + j) as i64;
            let (a, b, c) = match picture_of(*tok) {
                Some((key, i)) => {
                    let p = pics.get(&key).ok_or_else(|| e(format!("picture {key:#x}: not given to the engine")))?;
                    if i >= p.nx * p.ny {
                        return Err(e(format!("picture token {i} of {}", p.nx * p.ny)));
                    }
                    // llama.cpp's M-RoPE for an image: (t, t + row, t + column), t its first token's position
                    let t0 = slot - i as i64 - shift;
                    let r = (t0, t0 + (i / p.nx) as i64, t0 + (i % p.nx) as i64);
                    if i + 1 == p.nx * p.ny {
                        shift += (p.nx * p.ny) as i64 - p.nx.max(p.ny) as i64;
                    }
                    r
                }
                None => {
                    let q = slot - shift;
                    (q, q, q)
                }
            };
            pos[j] = a as i32;
            pos[t + j] = b as i32;
            pos[2 * t + j] = c as i32;
        }
        Ok((Rope::Rows(pos), shift))
    }

    pub fn gpu_info(&self) -> Vec<GpuInfo> {
        self.devs
            .iter()
            .map(|d| {
                let (total, free) = d.gpu.memory().unwrap_or((0, None));
                GpuInfo { index: d.gpu.index, name: d.gpu.name.clone(), pci: d.gpu.pci.clone(), total, free, layers: (d.lb as u64, d.le as u64), expert_slots: 0,
                          host_slots: 0 }
            })
            .collect()
    }
}

/// A tensor read onto the GPU: its blocks, or float32 for F32 / F16 / BF16
fn vec_mat(f: &Gguf, gpu: &Arc<Gpu>, k: &ffi::Api, name: &str, loaded: &mut u64) -> Result<Mat> {
    let t = f.tensor(name).ok_or_else(|| e(format!("no {name}")))?;
    // outermost first: [rows (outputs), cols (inputs)]; a vector is one row
    let (rows, cols) = match t.shape.len() {
        1 => (1, t.shape[0] as usize),
        _ => (t.shape[0] as usize, t.shape[1..].iter().product::<u64>() as usize),
    };
    let raw = f.read(t).map_err(|x| e(x.0))?;
    *loaded += raw.len() as u64;
    if matches!(t.ty, GType::F32 | GType::F16 | GType::BF16) {
        let src = DevBuf::new(gpu, raw.len())?;
        src.write(0, &raw)?;
        let out = DevBuf::f32(gpu, rows * cols)?;
        // SAFETY: src holds the tensor, out its elements as float32.
        ffi::check(unsafe { (k.dequant_f32)(gpu.raw(), t.ty.code() as i32, src.ptr(), (rows * cols) as i64, out.fp()) }, name)?;
        gpu.sync()?;
        return Ok(Mat { buf: out, ty: GType::F32, rows, cols });
    }
    let buf = DevBuf::new(gpu, raw.len())?;
    buf.write(0, &raw)?;
    Ok(Mat { buf, ty: t.ty, rows, cols })
}

/// The draft block from its file: block `layers` (eh_proj split into its embedding and hidden-state columns on the
/// host), its head norm and head; its token embedding stays in the file
/// `shared_head`: the block is in the model's own file - its head is the model's (not loaded twice)
fn load_mtp(m: Box<Gguf>, g: &Geometry, gpu: &Arc<Gpu>, k: &ffi::Api, shared_head: bool, loaded: &mut u64) -> Result<Mtp> {
    let l = g.layers;
    let f: &Gguf = &m;
    let n = |s: &str| format!("blk.{l}.{s}.weight");
    let mut read = |name: &str| -> Result<Mat> { vec_mat(f, gpu, k, name, loaded) };
    let attn = Full { q: read(&n("attn_q"))?, k: read(&n("attn_k"))?, v: read(&n("attn_v"))?, o: read(&n("attn_output"))?,
                      q_norm: read(&n("attn_q_norm"))?.buf, k_norm: read(&n("attn_k_norm"))?.buf };
    let (enorm, hnorm, head_norm) = (read(&n("nextn.enorm"))?.buf, read(&n("nextn.hnorm"))?.buf, read(&n("nextn.shared_head_norm"))?.buf);
    let (attn_norm, post_norm) = (read(&n("attn_norm"))?.buf, read(&n("post_attention_norm"))?.buf);
    let (gate, up, down) = (read(&n("ffn_gate"))?, read(&n("ffn_up"))?, read(&n("ffn_down"))?);
    let output = if shared_head { None } else { Some(read(if f.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" })?) };
    // eh_proj [n_embd, 2 n_embd]: each row's first half of blocks multiplies the embedding, the second the hidden state
    let t = f.tensor(&n("nextn.eh_proj")).ok_or_else(|| e("the MTP file has no nextn.eh_proj"))?;
    let (rows, cols) = (t.shape[0] as usize, t.shape[1..].iter().product::<u64>() as usize);
    if cols != 2 * g.n_embd || matches!(t.ty, GType::F32 | GType::F16 | GType::BF16) {
        return Err(e("nextn.eh_proj: a quantized [n_embd, 2 n_embd] matrix"));
    }
    let raw = f.read(t).map_err(|x| e(x.0))?;
    *loaded += raw.len() as u64;
    let (rb, hb) = (t.ty.bytes(cols as u64).unwrap_or(0) as usize, t.ty.bytes(g.n_embd as u64).unwrap_or(0) as usize);
    let (mut ea, mut ha) = (Vec::with_capacity(rows * hb), Vec::with_capacity(rows * hb));
    for r in 0..rows {
        ea.extend_from_slice(&raw[r * rb..r * rb + hb]);
        ha.extend_from_slice(&raw[r * rb + hb..(r + 1) * rb]);
    }
    let (be, bh) = (DevBuf::new(gpu, ea.len())?, DevBuf::new(gpu, ha.len())?);
    be.write(0, &ea)?;
    bh.write(0, &ha)?;
    let embd = f.tensor("token_embd.weight").ok_or_else(|| e("the MTP file has no token_embd.weight"))?.clone();
    let vocab = embd.shape[0] as usize;
    Ok(Mtp { eh_e: Mat { buf: be, ty: t.ty, rows, cols: g.n_embd }, eh_h: Mat { buf: bh, ty: t.ty, rows, cols: g.n_embd }, enorm, hnorm, head_norm,
             attn_norm, post_norm, attn, gate, up, down, output, embd, vocab, file: m })
}
