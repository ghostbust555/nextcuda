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
//! VRAM; the next layers' in pinned host memory, which the kernels read in place over PCIe; the rest on disk, read
//! into a pinned staging buffer for the experts a pass chose.

pub mod ffi;

use std::ffi::c_void;
use std::sync::Arc;
use std::time::Instant;

use nextcuda_core::{Arena, DevBuf, Error, Gpu, HostBuf, Result};
use nextcuda_gguf::{GType, Gguf, Tensor};

/// Rows a pass takes
pub const DEC: usize = 8;

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
    /// pinned host memory the kernels read in place (`view`)
    Pinned(#[allow(dead_code)] HostBuf, DevBuf),
    /// the file: the chosen experts are read into the staging buffer
    Disk,
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
            Home::Pinned(..) => "pinned",
            Home::Disk => "disk",
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
}

impl Default for Options {
    fn default() -> Options {
        let env = |k: &str| std::env::var(k).ok();
        Options {
            gpu: env("NS_NH_GPU").and_then(|v| v.parse().ok()).unwrap_or(0),
            ctx: env("NS_NH_CTX").and_then(|v| v.parse().ok()).unwrap_or(4096),
            pinned_gib: env("NS_NH_PINNED_GIB").and_then(|v| v.parse().ok()).unwrap_or(14.0),
            reserve_mib: env("NS_NH_RESERVE_MIB").and_then(|v| v.parse().ok()).unwrap_or(1800),
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
    /// the disk-resident experts' staging (pinned), and its device view
    staging: Option<(HostBuf, DevBuf, usize)>,
    pub times: std::sync::Mutex<Times>,
    pub load_s: f64,
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
        let mut pinned_left = (o.pinned_gib * (1u64 << 30) as f64) as u64;
        let mut homes = vec![0u8; g.layers]; // 0 VRAM, 1 pinned, 2 disk
        for l in 0..g.layers {
            if exp_bytes[l] == 0 {
                continue;
            }
            if exp_bytes[l] <= vram_left {
                vram_left -= exp_bytes[l];
            } else if exp_bytes[l] <= pinned_left {
                homes[l] = 1;
                pinned_left -= exp_bytes[l];
            } else {
                homes[l] = 2;
            }
        }
        let n = |h: u8| (0..g.layers).filter(|l| exp_bytes[*l] > 0 && homes[*l] == h).count();
        log(format!("nemotronh: {:.1} GiB dense + {:.1} GiB of experts in {} MoE layers; {:.1} GiB free on {}: experts of {} layers in VRAM, {} pinned, {} on disk",
                    gib(dense), gib(exp_bytes.iter().sum()), g.count(Kind::Moe), gib(free), gpu.name, n(0), n(1), n(2)));

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
        // the disk-resident layers' staging: the most experts a pass can pick, the widest up + down
        let widest = layers.iter().filter_map(|ly| match &ly.mix { Mix::Moe(m) if matches!(m.up.home, Home::Disk) => Some(m.up.stride + m.down.stride), _ => None }).max();
        let staging = match widest {
            Some(w) => {
                let cap = DEC * g.used * w;
                let mut hb = HostBuf::new(&gpu, cap)?;
                let view = hb.device_view(0, cap)?;
                Some((hb, view, w))
            }
            None => None,
        };
        let arena = Arena::new(&gpu, 160 << 20)?;
        gpu.sync()?;
        let load_s = t0.elapsed().as_secs_f64();
        log(format!("nemotronh: loaded {:.1} GiB onto {} in {load_s:.0} s ({} layers: {} Mamba-2, {} MoE, {} attention)", gib(loaded.get()), gpu.name, g.layers, g.count(Kind::Mamba),
                    g.count(Kind::Moe), g.count(Kind::Attn)));
        let _ = o;
        Ok(NemotronH { g, file: f, gpu, k, layers, embd, output_norm, output, arena, staging, times: Default::default(), load_s })
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

    /// The device address of expert `x`'s block of `ex` (a disk-resident one: its slot in the staging buffer)
    fn expert_at(&self, ex: &Experts, x: usize, slot: usize, off: usize) -> *const c_void {
        // SAFETY: inside the tensor / the staging buffer (callers' counts).
        unsafe {
            match &ex.home {
                Home::Vram(b) | Home::Pinned(_, b) => b.ptr().cast::<u8>().add(x * ex.stride).cast(),
                Home::Disk => self.staging.as_ref().unwrap().1.ptr().cast::<u8>().add(slot * self.staging.as_ref().unwrap().2 + off).cast(),
            }
        }
    }

    /// A latent-MoE layer on h [t, n_embd]: the routed experts in the latent, projected back, plus the shared expert
    fn moe(&self, m: &Moe, h: &DevBuf, t: usize) -> Result<DevBuf> {
        let (g, k) = (&self.g, self.k);
        let hq8 = self.quantize(h.fp(), g.n_embd, t)?;
        // the router (float32) and the choice on the host: sigmoid, the bias for the choice only, the chosen ones'
        // probabilities renormalized and scaled
        let logits = self.arena.f32(t * g.experts)?;
        self.matmul(&m.router, h, None, t, &logits)?;
        let lat = self.arena.f32(t * g.latent)?;
        self.matmul(&m.to_latent, h, Some(&hq8), t, &lat)?;
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
        // the experts' bytes where the kernels can read them: disk-resident ones into the staging slots
        let tf = Instant::now();
        let mut slot_of = std::collections::HashMap::new();
        if matches!(m.up.home, Home::Disk) {
            let (hb, _, w) = self.staging.as_ref().unwrap();
            let base = hb.as_slice().as_ptr() as *mut u8;
            let (tu, td) = (self.file.tensor(&m.up.name).unwrap(), self.file.tensor(&m.down.name).unwrap());
            for (_, x, _) in &ent {
                if slot_of.contains_key(x) {
                    continue;
                }
                let s = slot_of.len();
                slot_of.insert(*x, s);
                // SAFETY: the slot is inside the staging buffer (DEC * used slots of w bytes); nothing else touches it now.
                let dst = unsafe { std::slice::from_raw_parts_mut(base.add(s * w), *w) };
                let (a, b) = dst.split_at_mut(m.up.stride);
                self.file.read_into(tu, (*x * m.up.stride) as u64, a).map_err(|y| e(y.0))?;
                self.file.read_into(td, (*x * m.down.stride) as u64, &mut b[..m.down.stride]).map_err(|y| e(y.0))?;
            }
        }
        self.times.lock().unwrap().moe_fetch += tf.elapsed().as_secs_f64();
        let latq = self.quantize(lat.fp(), g.latent, t)?;
        // SAFETY: arithmetic only.
        let (lq, uq) = unsafe { ((k.q8_1_bytes)(g.latent as i64, 1), (k.q8_1_bytes)(g.n_ff_exp as i64, 1)) };
        let u = self.arena.f32(n_e * g.n_ff_exp)?;
        for (j, (r, x, _)) in ent.iter().enumerate() {
            let slot = slot_of.get(x).copied().unwrap_or(0);
            let w = self.expert_at(&m.up, *x, slot, 0);
            // SAFETY: expert x's up [n_ff, latent] in its blocks, token r's latent Q8_1, its row of u.
            self.op("expert up", || unsafe {
                (k.mmvq)(self.raw(), m.up.ty.code() as i32, w, latq.buf.ptr().cast::<u8>().add(r * lq).cast(), u.fp().add(j * g.n_ff_exp), g.latent as i64, g.n_ff_exp as i64, 1)
            })?;
        }
        // SAFETY: n floats.
        self.op("relu2", || unsafe { (k.relu2)(self.raw(), u.fp(), (n_e * g.n_ff_exp) as i64) })?;
        let uq8 = self.quantize(u.fp(), g.n_ff_exp, n_e)?;
        let dn = self.arena.f32(n_e * g.latent)?;
        for (j, (_, x, _)) in ent.iter().enumerate() {
            let slot = slot_of.get(x).copied().unwrap_or(0);
            let w = self.expert_at(&m.down, *x, slot, m.up.stride);
            // SAFETY: expert x's down [latent, n_ff] in its blocks, entry j's Q8_1, its row of dn.
            self.op("expert down", || unsafe {
                (k.mmvq)(self.raw(), m.down.ty.code() as i32, w, uq8.buf.ptr().cast::<u8>().add(j * uq).cast(), dn.fp().add(j * g.latent), g.n_ff_exp as i64, g.latent as i64, 1)
            })?;
        }
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
        let out = self.arena.f32(t * g.n_embd)?;
        self.matmul(&m.from_latent, &mo, None, t, &out)?;
        // the shared expert on the full width
        let su = self.arena.f32(t * g.n_ff_shared)?;
        self.matmul(&m.sh_up, h, Some(&hq8), t, &su)?;
        // SAFETY: n floats.
        self.op("relu2", || unsafe { (k.relu2)(self.raw(), su.fp(), (t * g.n_ff_shared) as i64) })?;
        let sd = self.arena.f32(t * g.n_embd)?;
        self.matmul(&m.sh_down, &su, None, t, &sd)?;
        self.add(&out, &sd, t * g.n_embd)?;
        Ok(out)
    }

    /// One pass of at most DEC rows: the last `n_out` rows' logits (0: none - a prompt chunk before the last)
    pub fn pass(&self, s: &mut Session, tokens: &[u32], n_out: usize) -> Result<Vec<Vec<f32>>> {
        let t = tokens.len();
        if t == 0 || t > DEC {
            return Err(e(format!("a pass of {t} rows; it takes 1 to {DEC}")));
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
        let chunks: Vec<&[u32]> = tokens.chunks(DEC).collect();
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

/// A layer's stacked experts [experts, rows, cols] where `home` says (0 VRAM, 1 pinned host memory, 2 the file)
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
        2 => Home::Disk,
        1 => {
            let mut pin = HostBuf::new(gpu, t.bytes as usize)?;
            f.read_into(t, 0, pin.as_mut_slice()).map_err(|x| e(x.0))?;
            let view = pin.device_view(0, t.bytes as usize)?;
            loaded.set(loaded.get() + t.bytes);
            Home::Pinned(pin, view)
        }
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
