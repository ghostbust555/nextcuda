//! The denoiser: Qwen-Image 2.1's single-stream transformer (diffusers' QwenImage21Transformer2DModel), 32 blocks of
//! 4,096 features (32 heads x 128), its weights from the GGUF (Q8_0 and BF16) dequantized to half on the GPU.
//!
//! The text and the image share one sequence, but attention is block-causal: the text is causal and never sees the
//! image, and text tokens are modulated from t = 0 (`causal_condition`) - so the text's keys and values in every
//! layer depend on the prompt alone. They are computed once (`prefix`) and every step runs only the image's tokens,
//! each attending to [the text's keys | the image's own]. One block, for the rows it runs:
//!
//! ```text
//!   h = layernorm(x) * (1 + scale1)          q k v = h [Wq|Wk|Wv]; q, k: per-head rmsnorm + 3-axis RoPE
//!   x += tanh(gate1) * attention(q, k, v) Wo
//!   h = layernorm(x) * (1 + scale2)
//!   x += tanh(gate2) * (silu(h Wgate) * (h Wproj)) Wout
//! ```
//!
//! scale / gate: one modulation shared by every block, from the timestep (the image's rows) or t = 0 (the text's).
//! The RoPE pairs features (2i, 2i + 1); the shared kernel rotates (i, 64 + i), so q's and k's output rows (and their
//! norms) are permuted at load - attention scores do not change, v is untouched.

use std::sync::Arc;

use nextcuda_core::{DevBuf, Error, Gpu, Result};
use std::sync::Mutex;

use nextcuda_diffusion::kernels::{none, Dt, Nsd};
use nextcuda_diffusion::lora::{self, Delta};
use nextcuda_gguf::{GType, Gguf};

use super::prof;
use crate::ffi;

pub const DIM: usize = 4096;
pub const HEADS: usize = 32;
pub const HEAD: usize = 128;
pub const FFN: usize = 12288;
pub const LATENT: usize = 64;
const EPS: f32 = 1e-6;
/// RoPE: the frame, height and width axes' features (pairs: 8, 28, 28)
const AXES: [usize; 3] = [16, 56, 56];

/// The ConvRot group: int8 matrices are rotated by the normalized 256 x 256 Hadamard matrix along their inputs
const GROUP: usize = 256;

/// One stored matrix of a block, kept in its GGUF type on the card (Q8_0, Q6_K, Q5_K, BF16 ...): the 12 GB cards hold
/// the whole transformer at its stored size (7.3 GB for the Q8_0 file) and a slice of rows is made half at a time
struct Part {
    w: DevBuf,
    qtype: u32,
    rows: usize,
}

/// A module a LoRA may name inside a matrix: its key, its rows and where they start in the matrix's output (a fused
/// gate | up is two), and whether its rows are permuted like q's and k's
struct Slot {
    key: String,
    rows: usize,
    off: usize,
    perm: bool,
}

/// One LoRA's update of a slot, half on the card: A [r, k] and B [rows, r] (B carries the scale)
struct LoraPair {
    a: DevBuf,
    b: DevBuf,
    r: usize,
}

/// A block's matrix [n, k] as the parts the file stores (a fused gate|up is one part; q, k and v are three, and may
/// be of different types in a mixed-quant file). LoRAs ride beside it as low-rank products added to the output: the
/// weights are quantized, so there is nothing to merge into.
struct Mat {
    parts: Vec<Part>,
    /// the matrix as int8 rows (ConvRot: rotated by the Hadamard matrix along the inputs, a scale per row) and the
    /// parts dropped: `NS_QI_INT8=1`, the card's int8 tensor cores
    int8: Option<(DevBuf, DevBuf)>,
    n: usize,
    k: usize,
    slots: Vec<Slot>,
    active: Mutex<Vec<(usize, LoraPair)>>,
}

impl Mat {
    /// out [m, n] = x [m, k] (half) . W^T, each part's rows written beside the others'
    fn apply(&self, nsd: &Nsd, x: *const std::ffi::c_void, m: usize, k: usize, out: *mut std::ffi::c_void, out_dt: Dt) -> Result<()> {
        let esz = if out_dt == Dt::F32 { 4 } else { 2 };
        if let Some((w, s)) = &self.int8 {
            nsd.int8_linear(x, Dt::F16, m, k, w.ptr(), self.n, s.ptr(), none(), out, out_dt, GROUP, self.n)?;
        } else {
            let mut off = 0usize;
            for p in &self.parts {
                let o = out.cast::<u8>().wrapping_add(off * esz).cast();
                nsd.linear_q(x, Dt::F16, m, k, p.w.ptr(), p.qtype, p.rows, none(), o, out_dt, self.n)?;
                off += p.rows;
            }
        }
        let act = self.active.lock().unwrap();
        if !act.is_empty() && out_dt == Dt::F16 {
            for (si, l) in act.iter() {
                let s = &self.slots[*si];
                let tmp = DevBuf::new(&nsd.gpu, m * l.r * 2)?;
                nsd.linear(x, Dt::F16, m, k, l.a.ptr(), l.r, none(), tmp.ptr(), Dt::F16)?;
                let o = out.cast::<u8>().wrapping_add(s.off * esz).cast();
                nsd.linear_acc(tmp.ptr(), Dt::F16, m, l.r, l.b.ptr(), s.rows, o, self.n)?;
                // `tmp` goes at the end of the call: its free waits for the work above
            }
        }
        Ok(())
    }
}

fn f16_bits(f: f32) -> u16 {
    let b = f.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7f_ffff;
    if exp >= 31 {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let m = (man | 0x80_0000) >> (1 - exp);
        return sign | ((m + 0xfff + ((m >> 13) & 1)) >> 13) as u16;
    }
    let r = ((exp as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    sign | (r + (rem > 0x1000 || (rem == 0x1000 && (r & 1) == 1)) as u32) as u16
}

fn half_buf(gpu: &Arc<Gpu>, v: &[f32]) -> Result<DevBuf> {
    let b = DevBuf::new(gpu, v.len().max(1) * 2)?;
    b.write(0, &v.iter().flat_map(|x| f16_bits(*x).to_le_bytes()).collect::<Vec<u8>>())?;
    Ok(b)
}

/// One block's weights
struct Block {
    qkv: Mat,
    out: Mat,
    /// gate | proj rows (SwiGLU reads them as one product)
    gp: Mat,
    down: Mat,
    norm_q: DevBuf,
    norm_k: DevBuf,
}

pub struct Dit {
    gpu: Arc<Gpu>,
    blocks: Vec<Block>,
    // the small linears, float32
    img_in: DevBuf,
    txt_norm: DevBuf,
    txt_in1: DevBuf,
    txt_in2: DevBuf,
    t_lin1: DevBuf,
    t_lin2: DevBuf,
    modulation: DevBuf,
    norm_out: DevBuf,
    proj_out: DevBuf,
    /// the small linears a LoRA was merged into (their float32 weights are changed in place): the originals, by field
    saved: Mutex<Vec<(&'static str, DevBuf)>>,
}

enum Kv {
    Dev(DevBuf),
    Host(Vec<u8>),
}

/// The text's state every step reads: each layer's q | k | v rows of the text (k with its RoPE), half
pub struct Prefix {
    /// rows: the text's, and the condition pictures' latents
    pub tokens: usize,
    /// each layer's k | v rows of the prefix, [tokens, 2 * 4096] half: on the card, or in host memory when the edit's
    /// pictures would not fit beside the weights
    kv: Vec<Kv>,
    /// the frame position the target picture takes (after the text and the pictures)
    pub frame: i64,
}

/// q / k row order: pair (2i, 2i + 1) -> (i, 64 + i)
fn rope_perm(j: usize) -> usize {
    if j < HEAD / 2 {
        2 * j
    } else {
        2 * (j - HEAD / 2) + 1
    }
}

fn ge(e: nextcuda_gguf::Error) -> Error {
    Error(e.0)
}

impl Dit {
    /// The weights from `f`, matrices in the type they are stored in; `int8`: the block matrices as int8 ConvRot rows
    pub fn load(f: &Gguf, nsd: &Nsd, int8: bool, log: &mut dyn FnMut(String)) -> Result<Dit> {
        let gpu = nsd.gpu.clone();
        let t0 = std::time::Instant::now();
        let tensor = |name: &str| f.tensor(name).ok_or_else(|| Error(format!("{}: no tensor {name}", f.paths[0].display())));
        // a matrix as stored (`perm`: new row r reads old row perm(r) within each 128-row head), on the card
        let part = |name: &str, perm: bool| -> Result<Part> {
            let t = tensor(name)?;
            let code = t.ty.code();
            if ![0, 1, 8, 12, 13, 14, 30].contains(&code) {
                return Err(Error(format!("{name}: {} weights are not read yet", t.ty.name())));
            }
            let rows = t.shape[0] as usize;
            let mut bytes = f.read(t).map_err(ge)?;
            if perm {
                let rb = bytes.len() / rows;
                let mut out = Vec::with_capacity(bytes.len());
                for r in 0..rows {
                    let (h, j) = (r / HEAD, r % HEAD);
                    let src = h * HEAD + rope_perm(j);
                    out.extend_from_slice(&bytes[src * rb..(src + 1) * rb]);
                }
                bytes = out;
            }
            let w = DevBuf::new(&gpu, bytes.len())?;
            w.write(0, &bytes)?;
            Ok(Part { w, qtype: code, rows })
        };
        // `slots`: the modules a LoRA may name in it, as (module path, rows); the stored parts' names when None
        let mat = |names: &[&str], perm: &[bool], slots: Option<&[(String, usize)]>, k: usize| -> Result<Mat> {
            let parts = names.iter().zip(perm).map(|(n, p)| part(n, *p)).collect::<Result<Vec<_>>>()?;
            let n = parts.iter().map(|p| p.rows).sum();
            let list: Vec<Slot> = match slots {
                Some(sl) => {
                    let mut off = 0;
                    sl.iter().map(|(nm, rows)| { let s = Slot { key: lora::key(nm), rows: *rows, off, perm: false }; off += rows; s }).collect()
                }
                None => {
                    let mut off = 0;
                    names.iter().zip(&parts).zip(perm).map(|((nm, p), pm)| {
                        let s = Slot { key: lora::key(nm.strip_suffix(".weight").unwrap_or(nm)), rows: p.rows, off, perm: *pm };
                        off += p.rows;
                        s
                    }).collect()
                }
            };
            // int8: the parts' rows rotated and quantized into one matrix [n, k] (one quantization of the activations
            // serves all of its rows), the stored copies let go
            let q8 = if int8 {
                let w = DevBuf::new(&gpu, n * k)?;
                let sc = DevBuf::f32(&gpu, n)?;
                let mut off = 0usize;
                for p in &parts {
                    nsd.int8_prepare(p.w.ptr(), p.qtype, 0, p.rows, k, GROUP, w.ptr().cast::<u8>().wrapping_add(off * k).cast(), sc.fp().wrapping_add(off).cast())?;
                    off += p.rows;
                }
                nsd.wait()?;
                Some((w, sc))
            } else {
                None
            };
            let parts = if q8.is_some() { Vec::new() } else { parts };
            Ok(Mat { parts, int8: q8, n, k, slots: list, active: Mutex::new(Vec::new()) })
        };
        // a vector (norm weights, a small linear) as float32 on the host, whatever its stored type
        let host_vec = |name: &str| -> Result<Vec<f32>> {
            let t = tensor(name)?;
            let b = f.read(t).map_err(ge)?;
            Ok(match t.ty {
                GType::BF16 => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
                GType::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
                GType::F16 => b.chunks_exact(2).map(|c| nextcuda_gguf::safetensors::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
                other => return Err(Error(format!("{name}: a {} vector", other.name()))),
            })
        };
        let single = |name: &str| -> Result<DevBuf> { DevBuf::from_f32(&gpu, &host_vec(name)?) };
        // ... permuted like q / k when asked
        let vector = |name: &str, permute: bool| -> Result<DevBuf> {
            let v = host_vec(name)?;
            let v = if permute { (0..v.len()).map(|j| v[rope_perm(j)]).collect() } else { v };
            DevBuf::from_f32(&gpu, &v)
        };
        let n_blocks = (0..).take_while(|i| f.tensor(&format!("transformer_blocks.{i}.attn.to_q.weight")).is_some()).count();
        let mut blocks = Vec::with_capacity(n_blocks);
        for i in 0..n_blocks {
            let p = format!("transformer_blocks.{i}.");
            blocks.push(Block {
                qkv: mat(&[&format!("{p}attn.to_q.weight"), &format!("{p}attn.to_k.weight"), &format!("{p}attn.to_v.weight")], &[true, true, false], None, DIM)?,
                out: mat(&[&format!("{p}attn.to_out.0.weight")], &[false], None, DIM)?,
                // the MLP's gate and up rows: two matrices, or one fused (`gate_up`)
                gp: if f.tensor(&format!("{p}img_mlp.gate_up.weight")).is_some() {
                    let sl = [(format!("{p}img_mlp.gate_layer"), FFN), (format!("{p}img_mlp.proj"), FFN)];
                    mat(&[&format!("{p}img_mlp.gate_up.weight")], &[false], Some(&sl), DIM)?
                } else {
                    mat(&[&format!("{p}img_mlp.gate_layer.weight"), &format!("{p}img_mlp.proj.weight")], &[false, false], None, DIM)?
                },
                down: mat(&[&format!("{p}img_mlp.out.weight")], &[false], None, FFN)?,
                norm_q: vector(&format!("{p}attn.norm_q.weight"), true)?,
                norm_k: vector(&format!("{p}attn.norm_k.weight"), true)?,
            });
        }
        // the text norm stores (scale - 1): the kernel multiplies by its weight
        let tn = {
            let v: Vec<f32> = host_vec("txt_in.text_norm.weight")?.iter().map(|x| x + 1.0).collect();
            DevBuf::from_f32(&gpu, &v)?
        };
        let dit = Dit {
            gpu: gpu.clone(),
            blocks,
            img_in: single("img_in.weight")?,
            txt_norm: tn,
            txt_in1: single("txt_in.in_layer.weight")?,
            txt_in2: single("txt_in.out_layer.weight")?,
            t_lin1: single("time_text_embed.timestep_embedder.linear_1.weight")?,
            t_lin2: single("time_text_embed.timestep_embedder.linear_2.weight")?,
            modulation: single("modulation.1.weight")?,
            norm_out: single("norm_out.linear.weight")?,
            proj_out: single("proj_out.weight")?,
            saved: Mutex::new(Vec::new()),
        };
        log(format!("qwen-image 2.1: {n_blocks} blocks on {} ({}) in {:.1} s", gpu.name, if int8 { "int8 ConvRot" } else { "as stored" }, t0.elapsed().as_secs_f64()));
        Ok(dit)
    }

    /// The small float32 linear a LoRA module names: (its key, field, rows, inputs)
    fn small_names() -> [(&'static str, &'static str, usize, usize); 8] {
        [
            ("img_in", "img_in", DIM, LATENT),
            ("txt_in.in_layer", "txt_in1", DIM, 0),
            ("txt_in.out_layer", "txt_in2", DIM, DIM),
            ("time_text_embed.timestep_embedder.linear_1", "t_lin1", DIM, 256),
            ("time_text_embed.timestep_embedder.linear_2", "t_lin2", DIM, DIM),
            ("modulation.1", "modulation", 4 * DIM, DIM),
            ("norm_out.linear", "norm_out", DIM, DIM),
            ("proj_out", "proj_out", LATENT, DIM),
        ]
    }

    fn small_buf(&self, field: &str) -> &DevBuf {
        match field {
            "img_in" => &self.img_in,
            "txt_in1" => &self.txt_in1,
            "txt_in2" => &self.txt_in2,
            "t_lin1" => &self.t_lin1,
            "t_lin2" => &self.t_lin2,
            "modulation" => &self.modulation,
            "norm_out" => &self.norm_out,
            _ => &self.proj_out,
        }
    }

    /// Put an end to the LoRAs in use: the block side paths dropped, the small linears' weights put back
    pub fn clear_loras(&self, nsd: &Nsd) -> Result<()> {
        for b in &self.blocks {
            for m in [&b.qkv, &b.out, &b.gp, &b.down] {
                m.active.lock().unwrap().clear();
            }
        }
        let mut saved = self.saved.lock().unwrap();
        for (field, orig) in saved.drain(..) {
            let w = self.small_buf(field);
            w.copy_within(0, &orig, 0, orig.len)?;
        }
        nsd.wait()
    }

    /// Use `deltas` (a LoRA's updates, B scaled): beside the block matrices as low-rank side paths, merged into the
    /// small float32 linears (their originals kept for `clear_loras`). A module this engine has no matrix for is an
    /// error, as in nextsycl's.
    pub fn set_loras(&self, nsd: &Nsd, deltas: &[Delta]) -> Result<()> {
        self.clear_loras(nsd)?;
        if deltas.is_empty() {
            return Ok(());
        }
        let gpu = &self.gpu;
        let mut by_key: std::collections::BTreeMap<String, Vec<&Delta>> = Default::default();
        for d in deltas {
            by_key.entry(lora::key(&d.target)).or_default().push(d);
        }
        let mut used = std::collections::BTreeSet::new();
        for b in &self.blocks {
            for m in [&b.qkv, &b.out, &b.gp, &b.down] {
                for (si, s) in m.slots.iter().enumerate() {
                    for d in by_key.get(&s.key).into_iter().flatten() {
                        if d.n != s.rows || d.k != m.k {
                            return Err(Error(format!("LoRA {}: {}x{}, the matrix is {}x{}", d.target, d.n, d.k, s.rows, m.k)));
                        }
                        // B's rows reordered as the matrix's are
                        let bv: Vec<f32> = if s.perm {
                            (0..d.n).flat_map(|row| {
                                let src = (row / HEAD) * HEAD + rope_perm(row % HEAD);
                                d.b[src * d.r..(src + 1) * d.r].iter().copied()
                            }).collect()
                        } else {
                            d.b.clone()
                        };
                        m.active.lock().unwrap().push((si, LoraPair { a: half_buf(gpu, &d.a)?, b: half_buf(gpu, &bv)?, r: d.r }));
                        used.insert(s.key.clone());
                    }
                }
            }
        }
        for (key, field, n, k) in Self::small_names() {
            let k_key = lora::key(key);
            let Some(list) = by_key.get(&k_key) else { continue };
            let w = self.small_buf(field);
            let orig = DevBuf::new(gpu, w.len)?;
            orig.copy_within(0, w, 0, w.len)?;
            self.saved.lock().unwrap().push((field, orig));
            for d in list {
                let k_in = if k == 0 { d.k } else { k };
                if d.n != n || d.k != k_in {
                    return Err(Error(format!("LoRA {}: {}x{}, the matrix is {n}x{k_in}", d.target, d.n, d.k)));
                }
                let at: Vec<f32> = (0..d.k).flat_map(|c| (0..d.r).map(move |j| (c, j))).map(|(c, j)| d.a[j * d.k + c]).collect();
                let (bd, ad) = (DevBuf::from_f32(gpu, &d.b)?, DevBuf::from_f32(gpu, &at)?);
                nsd.linear_acc(bd.ptr(), Dt::F32, d.n, d.r, ad.ptr(), d.k, w.ptr(), 0)?;
                nsd.wait()?;
                used.insert(k_key.clone());
            }
        }
        let missed: std::collections::BTreeSet<&str> = deltas.iter().filter(|d| !used.contains(&lora::key(&d.target))).map(|d| d.target.as_str()).collect();
        if !missed.is_empty() {
            let few: Vec<&str> = missed.iter().take(4).copied().collect();
            return Err(Error(format!("LoRA modules this engine does not take ({} of them): {}", missed.len(), few.join(", "))));
        }
        Ok(())
    }

    /// Attention of `sq` query rows (q at `q`, `qs`-element rows) over `skv` keys of every head (32 x 128): the silo's
    /// fused kernel when it is built and `NS_SILO` allows, else the shared one. `causal`: row r reads keys 0..=row0 + r.
    #[allow(clippy::too_many_arguments)]
    fn attend(&self, nsd: &Nsd, q: *const std::ffi::c_void, k: *const std::ffi::c_void, v: *const std::ffi::c_void, sq: usize, skv: usize, qs: usize,
              kvs: usize, causal: bool, row0: usize, out: *mut std::ffi::c_void) -> Result<()> {
        if let Some(si) = ffi::silo() {
            // SAFETY: a pure check of the shape.
            if unsafe { (si.attention_supported)(HEAD as i64, qs as i64, kvs as i64) } == 1 {
                // SAFETY: q, k, v, out are the sizes named, on this GPU.
                let rc = unsafe { (si.attention)(self.gpu.raw(), q, k, v, sq as i64, skv as i64, HEADS as i64, HEADS as i64, qs as i64, kvs as i64,
                                                 causal as i32, row0 as i64, out) };
                return ffi::check(rc, "attention (silo)");
            }
        }
        if causal {
            // q addresses row `row0` already: the shared kernel takes the base and the first row
            let base = q.cast::<u16>().wrapping_sub(row0 * qs).cast();
            nsd.attention_causal_rows(base, k, v, Dt::F16, row0, sq, HEADS, HEADS, HEAD, qs, kvs, out)
        } else {
            nsd.attention_qk(q, sq, qs, k, v, skv, kvs, Dt::F16, HEADS, HEAD, out, Dt::F16)
        }
    }

    /// The modulation for timestep `t` (a sigma in [0, 1]) and for t = 0: per row (0: t, 1: zero) scale1, tanh(gate1),
    /// scale2, tanh(gate2), the output norm's scale - each [2, 4096] float32 on the GPU
    fn modulation(&self, nsd: &Nsd, t: f32) -> Result<[DevBuf; 5]> {
        let gpu = &self.gpu;
        // the sinusoid: cos then sin of 1000 t * 10000^(-i / 128)
        let mut emb = vec![0f32; 2 * 256];
        for (row, tt) in [t, 0.0].iter().enumerate() {
            for i in 0..128 {
                let a = 1000.0 * *tt as f64 * (-(10000f64).ln() * i as f64 / 128.0).exp();
                emb[row * 256 + i] = a.cos() as f32;
                emb[row * 256 + 128 + i] = a.sin() as f32;
            }
        }
        let silu = |v: &mut [f32]| v.iter_mut().for_each(|x| *x /= 1.0 + (-*x).exp());
        let e = DevBuf::from_f32(gpu, &emb)?;
        let h = DevBuf::f32(gpu, 2 * DIM)?;
        nsd.linear(e.ptr(), Dt::F32, 2, 256, self.t_lin1.ptr(), DIM, none(), h.ptr(), Dt::F32)?;
        let mut hv = h.to_f32()?;
        silu(&mut hv);
        let h = DevBuf::from_f32(gpu, &hv)?;
        let temb = DevBuf::f32(gpu, 2 * DIM)?;
        nsd.linear(h.ptr(), Dt::F32, 2, DIM, self.t_lin2.ptr(), DIM, none(), temb.ptr(), Dt::F32)?;
        let mut tv = temb.to_f32()?;
        silu(&mut tv);
        let st = DevBuf::from_f32(gpu, &tv)?;
        let m = DevBuf::f32(gpu, 2 * 4 * DIM)?;
        nsd.linear(st.ptr(), Dt::F32, 2, DIM, self.modulation.ptr(), 4 * DIM, none(), m.ptr(), Dt::F32)?;
        let o = DevBuf::f32(gpu, 2 * DIM)?;
        nsd.linear(st.ptr(), Dt::F32, 2, DIM, self.norm_out.ptr(), DIM, none(), o.ptr(), Dt::F32)?;
        let mv = m.to_f32()?;
        // [scale1 | gate1 | scale2 | gate2] per row
        let part = |k: usize, tanh: bool| -> Result<DevBuf> {
            let mut v = vec![0f32; 2 * DIM];
            for row in 0..2 {
                for c in 0..DIM {
                    let x = mv[row * 4 * DIM + k * DIM + c];
                    v[row * DIM + c] = if tanh { x.tanh() } else { x };
                }
            }
            DevBuf::from_f32(gpu, &v)
        };
        Ok([part(0, false)?, part(1, true)?, part(2, false)?, part(3, true)?, DevBuf::from_f32(gpu, &o.to_f32()?)?])
    }

    /// (cos, sin) per token and pair for positions `idx` (frame, height, width per token), in the permuted order
    fn rope_table(&self, idx: &[[i64; 3]]) -> Result<DevBuf> {
        let mut cs = vec![0f32; idx.len() * HEAD];
        for (t, p) in idx.iter().enumerate() {
            let mut pair = 0;
            for (axis, dim) in AXES.iter().enumerate() {
                for i in 0..dim / 2 {
                    let inv = 1.0 / 10000f64.powf(2.0 * i as f64 / *dim as f64);
                    let a = p[axis] as f64 * inv;
                    cs[t * HEAD + 2 * pair] = a.cos() as f32;
                    cs[t * HEAD + 2 * pair + 1] = a.sin() as f32;
                    pair += 1;
                }
            }
        }
        DevBuf::from_f32(&self.gpu, &cs)
    }

    /// One block's rows `x` (float32 [m, 4096], in place): modulation row `row` of `md`; attention through `att`
    #[allow(clippy::too_many_arguments)]
    fn block_ffn(&self, nsd: &Nsd, b: &Block, x: &DevBuf, m: usize, md: &[DevBuf; 5], row: usize, zeros: &DevBuf, h: &DevBuf, gp: &DevBuf,
                 act: &DevBuf, o: &DevBuf) -> Result<()> {
        let k = ffi::api()?;
        let off = |buf: &DevBuf| -> *const f32 { buf.fp().wrapping_add(row * DIM) };
        // SAFETY: x holds m x 4096 floats, h as many halves; the scale row 4096 floats.
        ffi::check(unsafe { (k.ln_mod)(self.gpu.raw(), x.fp(), m as i64, DIM as i64, EPS, off(&md[2]), h.ptr()) }, "norm + modulate")?;
        prof::mark(nsd, "norm + modulate")?;
        b.gp.apply(nsd, h.ptr(), m, DIM, gp.ptr(), Dt::F16)?;
        prof::mark(nsd, "gate|up linear")?;
        nsd.swiglu(gp.ptr(), Dt::F16, m, FFN, act.ptr(), Dt::F16)?;
        prof::mark(nsd, "swiglu")?;
        b.down.apply(nsd, act.ptr(), m, FFN, o.ptr(), Dt::F16)?;
        prof::mark(nsd, "down linear")?;
        nsd.gate_add(x.ptr(), Dt::F32, m, DIM, o.ptr(), Dt::F16, zeros.ptr(), off(&md[3]).cast())?;
        prof::mark(nsd, "gate add")
    }

    /// The text's pass through every block (once a prompt): `embeds` float32 [T, 3584 or 4096] -> each layer's q|k|v
    pub fn prefix(&self, nsd: &Nsd, embeds: &DevBuf, tokens: usize, context_dim: usize) -> Result<Prefix> {
        self.prefix_with(nsd, embeds, tokens, context_dim, &[], &[])
    }

    /// The prefix of an edit: the text (`pads`: which of its tokens are `<|image_pad|>` slots - each stands for 2 x 2
    /// latents) with the condition pictures' latents (`conds`: float32 [h * w, 64] and (h, w), in the order of their
    /// slot runs) put in their slots. Block-causal: the text is causal, each picture's latents attend to everything
    /// before and to each other. Positions: text advances one on all three axes; a picture sits at the next frame
    /// with its rows and columns centred on zero, and the next text starts past its larger side. All at t = 0.
    pub fn prefix_with(&self, nsd: &Nsd, embeds: &DevBuf, tokens: usize, context_dim: usize, pads: &[bool], conds: &[(&DevBuf, (usize, usize))])
                       -> Result<Prefix> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let t = tokens;
        let md = self.modulation(nsd, 0.0)?;
        // txt_in: zero-centred RMS norm, linear, gelu (tanh), linear
        let hn = DevBuf::f32(gpu, t * context_dim)?;
        nsd.rms_norm_mod(embeds.ptr(), Dt::F32, t, context_dim, self.txt_norm.ptr(), EPS, none(), none(), none(), hn.ptr(), Dt::F32)?;
        let h1 = DevBuf::f32(gpu, t * DIM)?;
        nsd.linear(hn.ptr(), Dt::F32, t, context_dim, self.txt_in1.ptr(), DIM, none(), h1.ptr(), Dt::F32)?;
        let h1h = DevBuf::new(gpu, t * DIM * 2)?;
        // SAFETY: t x 4096 values each way.
        ffi::check(unsafe { (k.to_half)(gpu.raw(), h1.fp(), h1h.ptr(), (t * DIM) as i64) }, "to half")?;
        ffi::check(unsafe { (k.gelu_tanh)(gpu.raw(), h1h.ptr(), (t * DIM) as i64) }, "gelu")?;
        ffi::check(unsafe { (k.to_float)(gpu.raw(), h1h.ptr(), h1.fp(), (t * DIM) as i64) }, "to float")?;
        let xt = DevBuf::f32(gpu, t * DIM)?;
        nsd.linear(h1.ptr(), Dt::F32, t, DIM, self.txt_in2.ptr(), DIM, none(), xt.ptr(), Dt::F32)?;
        // the joint rows: (start in x, rows, text?) segments, their positions; the text's rows and the pictures' latents
        let mut segs: Vec<(usize, usize, bool)> = Vec::new();
        let mut idx: Vec<[i64; 3]> = Vec::new();
        let mut pos = 0i64;
        let mut copies: Vec<(usize, usize, Option<usize>, usize)> = Vec::new(); // (dst row, src row, picture, rows)
        let (mut i, mut c) = (0usize, 0usize);
        while i < t {
            let row = idx.len();
            if pads.get(i).copied().unwrap_or(false) {
                let (_, (ch, cw)) = *conds.get(c).ok_or_else(|| Error("more picture slots in the prompt than pictures".into()))?;
                let slots = (ch * cw) / 4;
                if pads.len() < i + slots || !pads[i..i + slots].iter().all(|p| *p) {
                    return Err(Error(format!("picture {c}: {slots} slots expected in the prompt")));
                }
                for n in 0..ch * cw {
                    let (r, q) = ((n / cw) as i64, (n % cw) as i64);
                    idx.push([pos, r - (ch as i64 - ch as i64 / 2), q - (cw as i64 - cw as i64 / 2)]);
                }
                copies.push((row, 0, Some(c), ch * cw));
                segs.push((row, ch * cw, false));
                pos += ch.max(cw) as i64;
                i += slots;
                c += 1;
            } else {
                let st = i;
                while i < t && !pads.get(i).copied().unwrap_or(false) {
                    idx.push([pos, pos, pos]);
                    pos += 1;
                    i += 1;
                }
                copies.push((row, st, None, i - st));
                segs.push((row, i - st, true));
            }
        }
        if c != conds.len() {
            return Err(Error(format!("{} pictures for {c} slot runs in the prompt", conds.len())));
        }
        let p = idx.len();
        let x = DevBuf::f32(gpu, p * DIM)?;
        for (dst, src, pic, rows) in &copies {
            match pic {
                None => x.copy_within(dst * DIM * 4, &xt, src * DIM * 4, rows * DIM * 4)?,
                Some(ci) => nsd.linear(conds[*ci].0.ptr(), Dt::F32, *rows, LATENT, self.img_in.ptr(), DIM, none(), x.fp().wrapping_add(dst * DIM).cast(), Dt::F32)?,
            }
        }
        let zeros = DevBuf::new(gpu, p.max(1) * 4)?;
        zeros.fill(0)?;
        let cs = self.rope_table(&idx)?;
        let (h, qkv, att, o) = (DevBuf::new(gpu, p * DIM * 2)?, DevBuf::new(gpu, p * 3 * DIM * 2)?, DevBuf::new(gpu, p * DIM * 2)?, DevBuf::new(gpu, p * DIM * 2)?);
        let (gp, act) = (DevBuf::new(gpu, p * 2 * FFN * 2)?, DevBuf::new(gpu, p * FFN * 2)?);
        // the prefix kept on the card when its k | v of every layer leaves room for the activations beside the weights
        // (a few GiB: a step's buffers, the attention's score blocks, the VAE); else in host memory, a layer a step
        let kv_bytes = self.blocks.len() * p * 2 * DIM * 2;
        let (_, free) = gpu.memory()?;
        let spill = free.is_some_and(|f| (kv_bytes as u64) + (3u64 << 30) + (p as u64) * 400 * 1024 > f);
        let mut kv = Vec::with_capacity(self.blocks.len());
        let off = |buf: &DevBuf| -> *const f32 { buf.fp().wrapping_add(DIM) }; // the t = 0 row
        let half_at = |buf: &DevBuf, n: usize| -> *mut std::ffi::c_void { buf.ptr().cast::<u16>().wrapping_add(n).cast() };
        let text_only = segs.len() == 1 && segs[0].2;
        for b in &self.blocks {
            // SAFETY: x holds p x 4096 floats, h as many halves; the scale row 4096 floats.
            ffi::check(unsafe { (k.ln_mod)(gpu.raw(), x.fp(), p as i64, DIM as i64, EPS, off(&md[0]), h.ptr()) }, "norm + modulate")?;
            b.qkv.apply(nsd, h.ptr(), p, DIM, qkv.ptr(), Dt::F16)?;
            nsd.rms_rope(qkv.ptr(), Dt::F16, p, HEADS, HEAD, 3 * DIM, b.norm_q.ptr(), EPS, cs.ptr(), HEAD)?;
            nsd.rms_rope(half_at(&qkv, DIM), Dt::F16, p, HEADS, HEAD, 3 * DIM, b.norm_k.ptr(), EPS, cs.ptr(), HEAD)?;
            if text_only {
                self.attend(nsd, qkv.ptr(), half_at(&qkv, DIM), half_at(&qkv, 2 * DIM), p, p, 3 * DIM, 3 * DIM, true, 0, att.ptr())?;
            } else {
                for (st, n, text) in &segs {
                    let out = half_at(&att, st * DIM);
                    if *text {
                        self.attend(nsd, half_at(&qkv, st * 3 * DIM), half_at(&qkv, DIM), half_at(&qkv, 2 * DIM), *n, st + n, 3 * DIM, 3 * DIM, true, *st, out)?;
                    } else {
                        self.attend(nsd, half_at(&qkv, st * 3 * DIM), half_at(&qkv, DIM), half_at(&qkv, 2 * DIM), *n, st + n, 3 * DIM, 3 * DIM, false, 0, out)?;
                    }
                }
            }
            // the layer's prefix q|k|v, kept for every step
            // k | v of the prefix's rows (the queries are not read again)
            let keep = DevBuf::new(gpu, p * 2 * DIM * 2)?;
            keep.copy_2d(0, 2 * DIM * 2, &qkv, DIM * 2, 3 * DIM * 2, 2 * DIM * 2, p)?;
            if spill {
                let mut host = vec![0u8; keep.len];
                keep.read(0, &mut host)?;
                kv.push(Kv::Host(host));
            } else {
                kv.push(Kv::Dev(keep));
            }
            b.out.apply(nsd, att.ptr(), p, DIM, o.ptr(), Dt::F16)?;
            nsd.gate_add(x.ptr(), Dt::F32, p, DIM, o.ptr(), Dt::F16, zeros.ptr(), off(&md[1]).cast())?;
            self.block_ffn(nsd, b, &x, p, &md, 1, &zeros, &h, &gp, &act, &o)?;
        }
        nsd.wait()?;
        Ok(Prefix { tokens: p, kv, frame: pos })
    }

    /// The velocity for latents `lat` (float32 [hw.0 * hw.1, 64]) at sigma `t`, after the text `pre`: float32 [N, 64].
    /// `block0`: block 0's output rows kept there (a check), when given.
    pub fn velocity(&self, nsd: &Nsd, pre: &Prefix, lat: &DevBuf, hw: (usize, usize), t: f32, block0: Option<&DevBuf>) -> Result<DevBuf> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let n = hw.0 * hw.1;
        let tt = pre.tokens;
        let md = self.modulation(nsd, t)?;
        let zeros = DevBuf::new(gpu, n * 4)?;
        zeros.fill(0)?;
        let x = DevBuf::f32(gpu, n * DIM)?;
        nsd.linear(lat.ptr(), Dt::F32, n, LATENT, self.img_in.ptr(), DIM, none(), x.ptr(), Dt::F32)?;
        // positions: the frame after the text; height and width centred on zero, row-major
        let (hh, ww) = (hw.0 as i64, hw.1 as i64);
        let idx: Vec<[i64; 3]> = (0..n as i64).map(|i| [pre.frame, i / ww - (hh - hh / 2), i % ww - (ww - ww / 2)]).collect();
        let cs = self.rope_table(&idx)?;
        let h = DevBuf::new(gpu, n * DIM * 2)?;
        // [the text's q|k|v | the image's], one layer at a time
        let all = DevBuf::new(gpu, (tt + n) * 3 * DIM * 2)?;
        let att = DevBuf::new(gpu, n * DIM * 2)?;
        // the linears' outputs before their gated add: half
        let o = DevBuf::new(gpu, n * DIM * 2)?;
        let (gp, act) = (DevBuf::new(gpu, n * 2 * FFN * 2)?, DevBuf::new(gpu, n * FFN * 2)?);
        let half_at = |buf: &DevBuf, at: usize| -> *mut std::ffi::c_void { buf.ptr().cast::<u16>().wrapping_add(at).cast() };
        let img = tt * 3 * DIM; // the image's first row in `all`, in halves
        let stage = if pre.kv.iter().any(|k| matches!(k, Kv::Host(_))) { Some(DevBuf::new(gpu, tt * 2 * DIM * 2)?) } else { None };
        prof::mark(nsd, "step setup")?;
        for (bi, b) in self.blocks.iter().enumerate() {
            // the prefix's k | v into the first rows of this layer's buffer
            match &pre.kv[bi] {
                Kv::Dev(b) => all.copy_2d(DIM * 2, 3 * DIM * 2, b, 0, 2 * DIM * 2, 2 * DIM * 2, tt)?,
                Kv::Host(v) => {
                    let st = stage.as_ref().expect("a staging buffer for host prefixes");
                    st.write(0, v)?;
                    all.copy_2d(DIM * 2, 3 * DIM * 2, st, 0, 2 * DIM * 2, 2 * DIM * 2, tt)?;
                }
            }
            prof::mark(nsd, "prefix copy")?;
            // SAFETY: x holds n x 4096 floats, h as many halves; the scale row 4096 floats.
            ffi::check(unsafe { (k.ln_mod)(gpu.raw(), x.fp(), n as i64, DIM as i64, EPS, md[0].fp(), h.ptr()) }, "norm + modulate")?;
            prof::mark(nsd, "norm + modulate")?;
            b.qkv.apply(nsd, h.ptr(), n, DIM, half_at(&all, img), Dt::F16)?;
            prof::mark(nsd, "qkv linear")?;
            nsd.rms_rope(half_at(&all, img), Dt::F16, n, HEADS, HEAD, 3 * DIM, b.norm_q.ptr(), EPS, cs.ptr(), HEAD)?;
            nsd.rms_rope(half_at(&all, img + DIM), Dt::F16, n, HEADS, HEAD, 3 * DIM, b.norm_k.ptr(), EPS, cs.ptr(), HEAD)?;
            prof::mark(nsd, "rms + rope")?;
            self.attend(nsd, half_at(&all, img), half_at(&all, DIM), half_at(&all, 2 * DIM), n, tt + n, 3 * DIM, 3 * DIM, false, 0, att.ptr())?;
            prof::mark(nsd, "attention")?;
            b.out.apply(nsd, att.ptr(), n, DIM, o.ptr(), Dt::F16)?;
            prof::mark(nsd, "out linear")?;
            nsd.gate_add(x.ptr(), Dt::F32, n, DIM, o.ptr(), Dt::F16, zeros.ptr(), md[1].ptr())?;
            prof::mark(nsd, "gate add")?;
            self.block_ffn(nsd, b, &x, n, &md, 0, &zeros, &h, &gp, &act, &o)?;
            if bi == 0 {
                if let Some(keep) = block0 {
                    keep.copy_within(0, &x, 0, n * DIM * 4)?;
                }
            }
        }
        // the output norm: layernorm * (1 + scale), then the projection to 64 latent channels
        let xh = DevBuf::new(gpu, n * DIM * 2)?;
        let xn = DevBuf::f32(gpu, n * DIM)?;
        // SAFETY: n x 4096 values each way; the scale row 4096 floats.
        ffi::check(unsafe { (k.ln_mod)(gpu.raw(), x.fp(), n as i64, DIM as i64, EPS, md[4].fp(), xh.ptr()) }, "norm + modulate")?;
        ffi::check(unsafe { (k.to_float)(gpu.raw(), xh.ptr(), xn.fp(), (n * DIM) as i64) }, "to float")?;
        let v = DevBuf::f32(gpu, n * LATENT)?;
        nsd.linear(xn.ptr(), Dt::F32, n, DIM, self.proj_out.ptr(), LATENT, none(), v.ptr(), Dt::F32)?;
        prof::mark(nsd, "output")?;
        nsd.wait()?;
        Ok(v)
    }
}
