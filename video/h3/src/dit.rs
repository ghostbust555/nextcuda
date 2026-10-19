//! The denoiser's block stack: 50 identical blocks over one sequence of tokens (text, conditioning, audio and video
//! tokens together), each block = attention over all tokens, then a two-layer MLP, both added back into the stream.
//!
//! One block, on a stream `x` [S, hidden]:
//!
//! ```text
//!   h   = norm(x) * (1 + scale_a[row]) + shift_a[row]           rows: which timestep/modality table row a token uses
//!   qkv = linear(h)                                             int8
//!   q,k = per-head norm + position rotation (in place in qkv)
//!   a   = attention(q, k, v)
//!   x  += linear(a) * gate_a[row]                               int8
//!   h   = norm(x) * (1 + scale_m[row]) + shift_m[row]
//!   x  += linear(silu(u) * w) * gate_m[row]   with [u | w] = linear(h)      int8, int8
//! ```
//!
//! The six tables come from a small linear on the timestep embedding, computed on the host once per step.
//!
//! The four block matrices come in two forms: int8 (ComfyUI's `int8_convrot` safetensors, int8 GEMMs) or a
//! llama.cpp k-quant (the GGUF denoisers, Q4_K / Q6_K: expanded to bfloat16 one matrix at a time, then a 16-bit
//! GEMM). The k-quants hold less of the card (Q4_K 10.6 GiB, Q6_K 15.4, int8 19.5), which leaves the room for
//! longer sequences; the int8 form computes faster.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::device::{Device, Tensor};
use crate::dtype::{bytes_to_f32, DType};
use crate::gguf::GType;
use crate::ops::{self, Int8Linear, KLinear, Mod, Rows};
use crate::safetensors::Checkpoint;
use crate::{load, Ctx, Error, Result};

/// The sizes of the denoiser, read from a checkpoint's shapes.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    /// The MLP's inner width (its first linear writes twice this).
    pub ffn: usize,
    /// Features of q and k that carry position (rotated); the rest are only normalized.
    pub rot_dim: usize,
    /// Table rows per distinct timestep: one per modality (video, text, audio).
    pub modalities: usize,
    /// Width of the timestep embedding the tables are made from.
    pub t_dim: usize,
    pub norm_eps: f32,
    pub qk_eps: f32,
    pub blocks: usize,
}

impl Config {
    pub fn from_checkpoint(ck: &Checkpoint) -> Result<Config> {
        let hidden = ck.get("blocks.0.norm1.weight")?.shape[0];
        let head_dim = ck.get("blocks.0.attn.q_norm.weight")?.shape[0];
        let heads = ck.get("blocks.0.attn.out_proj.weight")?.shape[1] / head_dim;
        let ffn = ck.get("blocks.0.mlp.fc2.weight")?.shape[1];
        let adaln = &ck.get("blocks.0.adaln_proj.linear.weight")?.shape;
        let inv_freq = ck.get("rope.inv_freq")?.shape[0];
        let blocks = ck
            .entries
            .keys()
            .filter_map(|k| k.strip_prefix("blocks.")?.split('.').next()?.parse::<usize>().ok())
            .max()
            .map_or(0, |m| m + 1);
        Ok(Config {
            hidden,
            heads,
            head_dim,
            ffn,
            rot_dim: inv_freq * 3 * 2, // three axes (time, height, width), a pair of features per frequency
            modalities: adaln[0] / (6 * hidden),
            t_dim: adaln[1],
            norm_eps: 1e-5,
            qk_eps: 1e-5,
            blocks,
        })
    }
}

/// A block matrix in either of the checkpoint forms.
pub enum Weight {
    I8(Int8Linear),
    K(KLinear),
}

impl Weight {
    fn size(&self) -> (usize, usize) {
        match self {
            Weight::I8(l) => (l.outputs(), l.inputs()),
            Weight::K(l) => (l.n, l.k),
        }
    }
}

/// One block's weights, on the device (the big ones) and on the host (the small table projection).
pub struct Block {
    norm1: Tensor,
    norm2: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    qkv: Weight,
    out_proj: Weight,
    fc1: Weight,
    fc2: Weight,
    /// [6 * hidden * modalities, t_dim] and its bias
    adaln_w: Vec<f32>,
    adaln_b: Vec<f32>,
}

const LINEARS: [&str; 4] = ["attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"];

pub(crate) fn small(dev: &Arc<Device>, ck: &Checkpoint, name: &str) -> Result<Tensor> {
    let e = ck.get(name)?;
    let v = bytes_to_f32(&ck.read(name)?, e.dtype)?;
    let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
    Tensor::from_bytes(dev, DType::F32, &[v.len()], &bytes)
}

pub(crate) fn host(ck: &Checkpoint, name: &str) -> Result<Vec<f32>> {
    bytes_to_f32(&ck.read(name)?, ck.get(name)?.dtype)
}

impl Block {
    /// `big`: the int8 weights and their scales, already on the device (see `Blocks::load`).
    fn assemble(dev: &Arc<Device>, ck: &Checkpoint, prefix: &str, index: usize, big: &mut BTreeMap<String, Tensor>) -> Result<Block> {
        let p = format!("{prefix}.{index}");
        let mut linear = |l: &str| -> Result<Weight> {
            let name = format!("{p}.{l}");
            let mut take = |k: String| big.remove(&k).ok_or_else(|| Error(format!("{k} was not loaded")));
            let wn = format!("{name}.weight");
            if let Some(kind) = ck.kquant_of(&wn) {
                let shape = ck.get(&wn)?.shape.clone();
                return Ok(Weight::K(KLinear { blocks: take(wn)?, kind, n: shape[0], k: shape[1] }));
            }
            let q = ck.quant(&name)?.ok_or_else(|| Error(format!("{name} has no quantization record: neither an int8 nor a k-quant checkpoint")))?;
            Ok(Weight::I8(Int8Linear::new(take(wn)?, take(format!("{name}.weight_scale"))?, None, q.convrot.then_some(q.group))?))
        };
        Ok(Block {
            qkv: linear(LINEARS[0])?,
            out_proj: linear(LINEARS[1])?,
            fc1: linear(LINEARS[2])?,
            fc2: linear(LINEARS[3])?,
            norm1: small(dev, ck, &format!("{p}.norm1.weight"))?,
            norm2: small(dev, ck, &format!("{p}.norm2.weight"))?,
            q_norm: small(dev, ck, &format!("{p}.attn.q_norm.weight"))?,
            k_norm: small(dev, ck, &format!("{p}.attn.k_norm.weight"))?,
            adaln_w: host(ck, &format!("{p}.adaln_proj.linear.weight"))?,
            adaln_b: host(ck, &format!("{p}.adaln_proj.linear.bias"))?,
        })
    }

    /// The block's six tables for the timestep embeddings `t_emb` [R, t_dim]: shift, scale, gate for the attention
    /// half, then for the MLP half; each [R * modalities, hidden], row = timestep index * modalities + modality.
    pub fn tables(&self, cfg: &Config, t_emb: &[f32]) -> [Vec<f32>; 6] {
        let (c, m, td) = (cfg.hidden, cfg.modalities, cfg.t_dim);
        let r = t_emb.len() / td;
        let mut out: [Vec<f32>; 6] = std::array::from_fn(|_| vec![0f32; r * m * c]);
        for ri in 0..r {
            let t = &t_emb[ri * td..(ri + 1) * td];
            for mi in 0..m {
                for (ci, table) in out.iter_mut().enumerate() {
                    let o0 = mi * 6 * c + ci * c;
                    let row = &mut table[(ri * m + mi) * c..(ri * m + mi + 1) * c];
                    for (i, v) in row.iter_mut().enumerate() {
                        let o = o0 + i;
                        let w = &self.adaln_w[o * td..(o + 1) * td];
                        *v = self.adaln_b[o] + w.iter().zip(t).map(|(a, b)| a * b).sum::<f32>();
                    }
                }
            }
        }
        out
    }
}

/// What a step fixes for every block: each token's table row, the position rotations, and every block's tables.
pub struct Step {
    pub tokens: usize,
    /// int32 [S], on each of the model's cards
    rows: Vec<Tensor>,
    /// float32 [S, rot_dim / 2, 2]: (cos, sin), on each card
    cs: Vec<Tensor>,
    /// per block: shift_a, scale_a, gate_a, shift_m, scale_m, gate_m
    tables: Vec<[Tensor; 6]>,
}

/// (cos, sin) of every token's rotation angles: positions [S, 3] (time, height, width) times the model's frequencies,
/// the three axes one after the other.
pub fn rotations(positions: &[f64], inv_freq: &[f32]) -> Vec<f32> {
    let s = positions.len() / 3;
    let mut cs = Vec::with_capacity(s * 3 * inv_freq.len() * 2);
    for p in positions.chunks_exact(3) {
        for axis in p {
            for f in inv_freq {
                let a = *axis as f32 * f;
                cs.push(a.cos());
                cs.push(a.sin());
            }
        }
    }
    cs
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

impl Step {
    pub fn new(dev: &Arc<Device>, model: &Blocks, rows: &[i32], positions: &[f64], inv_freq: &[f32], t_emb: &[f32]) -> Result<Step> {
        let cfg = &model.cfg;
        let tokens = rows.len();
        if positions.len() != tokens * 3 || inv_freq.len() * 6 != cfg.rot_dim || !t_emb.len().is_multiple_of(cfg.t_dim) {
            return Err(Error("step: positions, frequencies or timestep embedding do not fit the model".into()));
        }
        let r = t_emb.len() / cfg.t_dim * cfg.modalities;
        if let Some(bad) = rows.iter().find(|x| **x < 0 || **x as usize >= r) {
            return Err(Error(format!("step: table row {bad} of {r}")));
        }
        let cs = rotations(positions, inv_freq);
        let row_bytes: Vec<u8> = rows.iter().flat_map(|x| x.to_le_bytes()).collect();
        // a model on one card is on `dev`; split, each block's tables go to its own card
        let devs: Vec<Arc<Device>> = if model.devs.len() > 1 { model.devs.clone() } else { vec![dev.clone()] };
        let mut tables = Vec::with_capacity(model.blocks.len());
        for (i, b) in model.blocks.iter().enumerate() {
            let on = &devs[model.dev_of.get(i).copied().unwrap_or(0).min(devs.len() - 1)];
            let t = b.tables(cfg, t_emb);
            let mut dev_t = Vec::with_capacity(6);
            for v in &t {
                dev_t.push(Tensor::from_bytes(on, DType::F32, &[r, cfg.hidden], &f32_bytes(v))?);
            }
            tables.push(dev_t.try_into().map_err(|_| Error("six tables per block".into()))?);
        }
        let cs_bytes = f32_bytes(&cs);
        Ok(Step {
            tokens,
            rows: devs.iter().map(|d| Tensor::from_bytes(d, DType::I32, &[tokens], &row_bytes)).collect::<Result<_>>()?,
            cs: devs.iter().map(|d| Tensor::from_bytes(d, DType::F32, &[tokens, cfg.rot_dim / 2, 2], &cs_bytes)).collect::<Result<_>>()?,
            tables,
        })
    }
}

/// The buffers a block writes between its kernels; one set serves every block.
pub struct Scratch {
    pub(crate) h: Tensor,
    pub(crate) qkv: Tensor,
    pub(crate) att: Tensor,
    pub(crate) proj: Tensor,
    pub(crate) fc1: Tensor,
    pub(crate) act: Tensor,
    /// a LoRA's middle product [tokens, rank]
    pub(crate) lora: Tensor,
}

/// The widest LoRA the scratch has room for.
pub const MAX_LORA_RANK: usize = 128;

impl Scratch {
    pub fn new(dev: &Arc<Device>, cfg: &Config, tokens: usize, dtype: DType) -> Result<Scratch> {
        let w = cfg.heads * cfg.head_dim;
        Ok(Scratch {
            h: Tensor::new(dev, dtype, &[tokens, cfg.hidden])?,
            qkv: Tensor::new(dev, dtype, &[tokens, 3 * w])?,
            att: Tensor::new(dev, dtype, &[tokens, w])?,
            proj: Tensor::new(dev, dtype, &[tokens, cfg.hidden])?,
            fc1: Tensor::new(dev, dtype, &[tokens, 2 * cfg.ffn])?,
            act: Tensor::new(dev, dtype, &[tokens, cfg.ffn])?,
            lora: Tensor::new(dev, dtype, &[tokens, MAX_LORA_RANK])?,
        })
    }
}

/// A LoRA's addition to linear `j`'s output, when the block has one there.
pub(crate) fn side(lora: Option<&crate::lora::BlockLora>, j: usize, x: &Tensor, out: &Tensor, tmp: &Tensor) -> Result<()> {
    for l in lora.map(|l| l[j].as_slice()).unwrap_or(&[]) {
        l.apply(x, tmp, out)?;
    }
    Ok(())
}

/// Called after each stage of a block with the stage's name and its result: how a run is checked against a reference.
pub type Tap<'a> = &'a mut dyn FnMut(&str, &Tensor) -> Result<()>;

/// The block stack.
pub struct Blocks {
    pub cfg: Config,
    pub blocks: Vec<Block>,
    pub load_seconds: f64,
    pub load_bytes: u64,
    /// the k-quant form's (`None` for int8): which, and the bfloat16 buffer each matrix is expanded into (one a card)
    pub kquant: Option<GType>,
    wbuf: Vec<Option<Tensor>>,
    /// the cards the blocks are on (one, unless `load_split`), and each block's card (an index into `devs`)
    pub devs: Vec<Arc<Device>>,
    pub dev_of: Vec<usize>,
}

impl Blocks {
    /// Loads the first `count` blocks (all of them when `None`): the int8 weights with `threads` parallel readers.
    pub fn load(dev: &Arc<Device>, ck: &Checkpoint, count: Option<usize>, threads: usize) -> Result<Blocks> {
        let mut cfg = Config::from_checkpoint(ck)?;
        cfg.blocks = count.unwrap_or(cfg.blocks).min(cfg.blocks);
        Self::load_from(dev, ck, "blocks", cfg, threads)
    }

    /// `cfg.blocks` blocks under `prefix` (`blocks`; a ControlNet's `control_blocks`), shaped as `cfg` says
    pub fn load_from(dev: &Arc<Device>, ck: &Checkpoint, prefix: &str, cfg: Config, threads: usize) -> Result<Blocks> {
        let n = cfg.blocks;
        let lead = format!("{prefix}.");
        let wanted = |name: &str| -> bool {
            let Some(rest) = name.strip_prefix(lead.as_str()) else { return false };
            let Some((idx, tail)) = rest.split_once('.') else { return false };
            idx.parse::<usize>().is_ok_and(|i| i < n) && LINEARS.iter().any(|l| tail == format!("{l}.weight") || tail == format!("{l}.weight_scale"))
        };
        let mut loaded = load::load(dev, ck, threads, wanted).ctx("loading the denoiser's weights")?;
        let mut blocks = Vec::with_capacity(n);
        for i in 0..n {
            blocks.push(Block::assemble(dev, ck, prefix, i, &mut loaded.tensors).ctx(format!("block {i}"))?);
        }
        let kquant = ck.kquant_of(&format!("{prefix}.0.mlp.fc1.weight"));
        let wbuf = match kquant {
            Some(_) => {
                let most = blocks.iter().flat_map(|b| [&b.qkv, &b.out_proj, &b.fc1, &b.fc2]).map(|w| { let (n, k) = w.size(); n * k }).max().unwrap_or(0);
                Some(Tensor::new(dev, DType::BF16, &[most]).ctx("the buffer the k-quant matrices expand into")?)
            }
            None => None,
        };
        Ok(Blocks { cfg, blocks, load_seconds: loaded.seconds, load_bytes: loaded.bytes, kquant, wbuf: vec![wbuf], devs: vec![dev.clone()], dev_of: vec![0; n] })
    }

    /// The bytes of block `i`'s matrices in the checkpoint
    fn block_bytes(ck: &Checkpoint, prefix: &str, i: usize) -> u64 {
        LINEARS.iter().flat_map(|l| [format!("{prefix}.{i}.{l}.weight"), format!("{prefix}.{i}.{l}.weight_scale")])
            .filter_map(|n| ck.get(&n).ok().map(|e| e.bytes as u64)).sum()
    }

    /// All the blocks over several cards (nextcuda: a 12 GB pair holds what one 32 GB card did): the first card takes
    /// blocks while it has the room beyond `reserve[d]` bytes kept for the run's buffers, then the next. Fails when
    /// they do not fit - nothing is streamed.
    pub fn load_split(devs: &[Arc<Device>], ck: &Checkpoint, reserve: &[u64], threads: usize) -> Result<Blocks> {
        let cfg = Config::from_checkpoint(ck)?;
        let n = cfg.blocks;
        if devs.len() < 2 {
            return Self::load_from(devs.first().ok_or("no card for the denoiser")?, ck, "blocks", cfg, threads);
        }
        let kquant = ck.kquant_of("blocks.0.mlp.fc1.weight");
        // the expansion buffer each card needs beside its blocks (the widest matrix, bfloat16)
        let widest: u64 = LINEARS.iter().filter_map(|l| ck.get(&format!("blocks.0.{l}.weight")).ok()).map(|e| e.shape.iter().product::<usize>() as u64 * 2).max().unwrap_or(0);
        let mut room: Vec<u64> = devs.iter().enumerate()
            .map(|(d, dev)| dev.mem_cap().saturating_sub(dev.mem_used()).saturating_sub(reserve.get(d).copied().unwrap_or(0)).saturating_sub(if kquant.is_some() { widest } else { 0 }))
            .collect();
        // the cards left the same room beside the blocks (each at least its reserve): filling the first card to its
        // reserve left the 5070 2 GiB for a clip's buffers (a 10 s pass at 384x288, 8,600 tokens, did not get them)
        // while the 3060 kept 6
        let total: u64 = (0..n).map(|j| Self::block_bytes(ck, "blocks", j)).sum();
        let extra = room.iter().sum::<u64>().saturating_sub(total) / devs.len() as u64;
        let last = devs.len() - 1;
        for (di, r) in room.iter_mut().enumerate() {
            if di < last {
                *r = r.saturating_sub(extra);
            }
        }
        let mut dev_of = Vec::with_capacity(n);
        let mut d = 0;
        for i in 0..n {
            let b = Self::block_bytes(ck, "blocks", i);
            while d < devs.len() && b > room[d] {
                d += 1;
            }
            if d == devs.len() {
                let total: u64 = (0..n).map(|j| Self::block_bytes(ck, "blocks", j)).sum();
                return Err(Error(format!("the denoiser's blocks ({:.1} GiB) do not fit on {}: block {i} has no card (a smaller quantization, or fewer seconds)",
                                         total as f64 / 1073741824.0, devs.iter().map(|d| d.name()).collect::<Vec<_>>().join(" + "))));
            }
            room[d] -= b;
            dev_of.push(d);
        }
        let mut blocks: Vec<Option<Block>> = (0..n).map(|_| None).collect();
        let (mut seconds, mut bytes) = (0f64, 0u64);
        for (di, dev) in devs.iter().enumerate() {
            let mine: Vec<usize> = (0..n).filter(|i| dev_of[*i] == di).collect();
            if mine.is_empty() {
                continue;
            }
            let wanted = |name: &str| -> bool {
                let Some(rest) = name.strip_prefix("blocks.") else { return false };
                let Some((idx, tail)) = rest.split_once('.') else { return false };
                idx.parse::<usize>().is_ok_and(|i| mine.contains(&i)) && LINEARS.iter().any(|l| tail == format!("{l}.weight") || tail == format!("{l}.weight_scale"))
            };
            let mut loaded = load::load(dev, ck, threads, wanted).ctx(format!("loading the denoiser's blocks onto {}", dev.name()))?;
            seconds += loaded.seconds;
            bytes += loaded.bytes;
            for i in &mine {
                blocks[*i] = Some(Block::assemble(dev, ck, "blocks", *i, &mut loaded.tensors).ctx(format!("block {i}"))?);
            }
        }
        let blocks: Vec<Block> = blocks.into_iter().map(|b| b.ok_or("a block was not placed")).collect::<std::result::Result<_, _>>()?;
        let wbuf = devs.iter().enumerate().map(|(di, dev)| -> Result<Option<Tensor>> {
            match kquant {
                Some(_) if dev_of.contains(&di) => {
                    let most = blocks.iter().zip(&dev_of).filter(|(_, d)| **d == di).flat_map(|(b, _)| [&b.qkv, &b.out_proj, &b.fc1, &b.fc2])
                        .map(|w| { let (n, k) = w.size(); n * k }).max().unwrap_or(0);
                    Ok(Some(Tensor::new(dev, DType::BF16, &[most]).ctx("the buffer the k-quant matrices expand into")?))
                }
                _ => Ok(None),
            }
        }).collect::<Result<Vec<_>>>()?;
        Ok(Blocks { cfg, blocks, load_seconds: seconds, load_bytes: bytes, kquant, wbuf, devs: devs.to_vec(), dev_of })
    }

    /// Block `i`'s card (an index into `devs`)
    pub fn card_of(&self, i: usize) -> usize {
        self.dev_of.get(i).copied().unwrap_or(0)
    }

    /// `out = linear(x)` with one of the blocks' matrices.
    fn mm(&self, w: &Weight, x: &Tensor, out: &Tensor, card: usize) -> Result<()> {
        match w {
            Weight::I8(l) => l.forward(x, out),
            Weight::K(l) => l.forward(x, out, self.wbuf.get(card).and_then(Option::as_ref).ok_or("a k-quant matrix without its expansion buffer")?),
        }
    }

    /// Runs block `index` on the stream `x` [S, hidden], in place.
    /// `lora`: this block's LoRAs, if any.
    pub fn block(&self, index: usize, x: &Tensor, step: &Step, s: &Scratch, lora: Option<&crate::lora::BlockLora>, mut tap: Option<Tap>) -> Result<()> {
        let cfg = &self.cfg;
        let b = &self.blocks[index];
        let card = self.card_of(index);
        let (rows, cs) = (&step.rows[card.min(step.rows.len() - 1)], &step.cs[card.min(step.cs.len() - 1)]);
        let [shift_a, scale_a, gate_a, shift_m, scale_m, gate_m] = &step.tables[index];
        let mut tap = |name: &str, t: &Tensor| -> Result<()> {
            match tap.as_mut() {
                Some(f) => f(name, t),
                None => Ok(()),
            }
        };
        // attention half
        ops::rms_norm_mod(x, &b.norm1, cfg.norm_eps, Some(&Mod { rows: rows, scale: scale_a, shift: shift_a }), &s.h)?;
        tap("h1", &s.h)?;
        self.mm(&b.qkv, &s.h, &s.qkv, card)?;
        side(lora, 0, &s.h, &s.qkv, &s.lora)?;
        tap("qkv", &s.qkv)?;
        let w = cfg.heads * cfg.head_dim;
        let part = |i: usize| Rows { t: &s.qkv, offset: i * w, stride: 3 * w, tokens: step.tokens, heads: cfg.heads, dim: cfg.head_dim };
        ops::rms_rope(part(0), &b.q_norm, cfg.qk_eps, cs, cfg.rot_dim)?;
        ops::rms_rope(part(1), &b.k_norm, cfg.qk_eps, cs, cfg.rot_dim)?;
        tap("qkv_rotated", &s.qkv)?;
        ops::attention(part(0), part(1), part(2), &s.att)?;
        tap("att", &s.att)?;
        self.mm(&b.out_proj, &s.att, &s.proj, card)?;
        side(lora, 1, &s.att, &s.proj, &s.lora)?;
        tap("attn_out", &s.proj)?;
        ops::gate_add(x, &s.proj, rows, gate_a)?;
        tap("x1", x)?;
        // MLP half
        ops::rms_norm_mod(x, &b.norm2, cfg.norm_eps, Some(&Mod { rows: rows, scale: scale_m, shift: shift_m }), &s.h)?;
        tap("h2", &s.h)?;
        self.mm(&b.fc1, &s.h, &s.fc1, card)?;
        side(lora, 2, &s.h, &s.fc1, &s.lora)?;
        tap("fc1", &s.fc1)?;
        ops::swiglu(&s.fc1, &s.act)?;
        self.mm(&b.fc2, &s.act, &s.proj, card)?;
        side(lora, 3, &s.act, &s.proj, &s.lora)?;
        tap("mlp", &s.proj)?;
        ops::gate_add(x, &s.proj, rows, gate_m)?;
        tap("x2", x)
    }
}
