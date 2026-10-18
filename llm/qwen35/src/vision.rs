//! Qwen3.8's vision tower: pictures to the language model's image tokens, from the llama.cpp projector beside the
//! model (`mmproj-Qwen3.8-27B-BF16.gguf`: `clip.projector_type qwen3vl_merger`, `v.*` and `mm.*`). This engine's own
//! copy (the silo) of nextcuda's crates/qwen3vl/src/vision.rs - the Qwen3-VL tower - for this file's form: BF16
//! matrices (made half at load), the two temporal slices of the patch kernel in float32, no deepstack mergers, a merger
//! out to the 27B's 5,120. Half weights and activations, a float32 residual stream (the diffusion kernels, nsd).
//!
//! ```text
//!   picture -> resized as llama.cpp's mtmd does (sides to multiples of 32, 8 ..= 4,096 tokens of 32 x 32 pixels,
//!     bicubic) -> 16 x 16 patches, (x / 255 - 0.5) / 0.5, in 2 x 2 merge-block order (the two temporal slices of the
//!     patch kernel summed: a picture is its own frame pair)
//!   -> linear 768 -> 1,152 + the learned 48 x 48 position grid resampled (bilinear, align_corners) to the picture's
//!   -> 27 blocks: x += proj(attention(rope2d(qkv(layernorm(x)))));  x += fc2(gelu_tanh(fc1(layernorm(x))))
//!      16 heads of 72, 2-D RoPE (height and width angles side by side)
//!   -> merger: layernorm (1,152), each 2 x 2 block's four tokens as one row of 4,608 -> fc1, gelu, fc2 -> 5,120
//! ```
//! The merged tokens come out in raster order of the merged grid (block row, block column): the order llama.cpp
//! places them in the prompt.

use std::path::Path;
use std::sync::Arc;

use nextcuda_core::{DevBuf, Error, Gpu, Result};
use nextcuda_diffusion::kernels::{none, Dt, Nsd};
use nextcuda_gguf::{GType, Gguf};

pub const PATCH: usize = 16;
pub const MERGE: usize = 2;
const PATCH_IN: usize = 3 * PATCH * PATCH;
const EPS: f32 = 1e-6;
const THETA: f64 = 10000.0;
/// a merged token's side in pixels
pub const ALIGN: usize = PATCH * MERGE;
/// llama.cpp's limits for Qwen-VL pictures (set_limit_image_tokens(8, 4096)): merged tokens
pub const MIN_TOKENS: usize = 8;
pub const MAX_TOKENS: usize = 4096;

struct Block {
    n1w: DevBuf,
    n1b: DevBuf,
    qkv: DevBuf,
    qkv_b: DevBuf,
    proj: DevBuf,
    proj_b: DevBuf,
    n2w: DevBuf,
    n2b: DevBuf,
    fc1: DevBuf,
    fc1_b: DevBuf,
    fc2: DevBuf,
    fc2_b: DevBuf,
}

pub struct Vision {
    gpu: Arc<Gpu>,
    patch: DevBuf,
    patch_b: DevBuf,
    /// the learned position grid, on the host [side * side, hidden]
    pos: Vec<f32>,
    side: usize,
    blocks: Vec<Block>,
    nw: DevBuf,
    nb: DevBuf,
    fc1: DevBuf,
    fc1_b: DevBuf,
    fc2: DevBuf,
    fc2_b: DevBuf,
    pub hidden: usize,
    heads: usize,
    ffn: usize,
    pub out: usize,
    /// its weights' bytes on the card
    pub bytes: u64,
}

fn ge(e: nextcuda_gguf::Error) -> Error {
    Error(e.0)
}

fn f16_to_f32(h: u16) -> f32 {
    nextcuda_gguf::safetensors::f16_to_f32(h)
}

fn f32_to_f16(f: f32) -> u16 {
    let b = f.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7f_ffff;
    if exp >= 31 {
        return sign | 0x7bff; // the largest half (the weights are far from it; nothing turns infinite)
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

/// The bytes the tower's weights take on the card as this engine loads them (half matrices, float32 vectors), from
/// the file's shapes - for the load plan, before anything is read
pub fn card_bytes(f: &Gguf) -> u64 {
    f.tensors.iter().map(|t| {
        let n = t.elements();
        if t.name == "v.position_embd.weight" {
            0 // on the host
        } else if t.name == "v.patch_embd.weight.1" {
            0 // summed into the first
        } else if t.name.ends_with(".weight") && t.shape.len() >= 2 {
            n * 2
        } else {
            n * 4
        }
    }).sum()
}

/// The size llama.cpp's mtmd resizes a w x h picture to (img_tool::calc_size_preserved_ratio with Qwen-VL's limits):
/// each side rounded to a multiple of 32, then scaled down to at most 4,096 or up to at least 8 merged tokens
pub fn target_size(w: usize, h: usize) -> (usize, usize) {
    let a = ALIGN as f32;
    let (wf, hf) = (w as f32, h as f32);
    let round = |x: f32| ((x / a).round() as usize) * ALIGN;
    let ceil = |x: f32| ((x / a).ceil() as usize) * ALIGN;
    let floor = |x: f32| ((x / a).floor() as usize) * ALIGN;
    let (minp, maxp) = (MIN_TOKENS * ALIGN * ALIGN, MAX_TOKENS * ALIGN * ALIGN);
    let mut wb = ALIGN.max(round(wf));
    let mut hb = ALIGN.max(round(hf));
    if hb * wb > maxp {
        let beta = (hf * wf / maxp as f32).sqrt();
        hb = ALIGN.max(floor(hf / beta));
        wb = ALIGN.max(floor(wf / beta));
    } else if hb * wb < minp {
        let beta = (minp as f32 / (hf * wf)).sqrt();
        hb = ceil(hf * beta);
        wb = ceil(wf * beta);
    }
    (wb, hb)
}

/// The processor's patches of a picture: RGB float32 [H, W, 3] in 0..255 (sides multiples of 32) -> [n, 768] in
/// merge-block order (block row, block column, row in block, column in block), each patch (channel, y, x)
pub fn patches(rgb: &[f32], h: usize, w: usize) -> Result<(Vec<f32>, (usize, usize))> {
    if h % ALIGN != 0 || w % ALIGN != 0 || rgb.len() != h * w * 3 {
        return Err(Error(format!("a {w}x{h} picture: the vision tower takes sides that are multiples of {ALIGN}")));
    }
    let (gh, gw) = (h / PATCH, w / PATCH);
    let mut out = Vec::with_capacity(gh * gw * PATCH_IN);
    for br in 0..gh / MERGE {
        for bc in 0..gw / MERGE {
            for ir in 0..MERGE {
                for ic in 0..MERGE {
                    let (py, px) = ((br * MERGE + ir) * PATCH, (bc * MERGE + ic) * PATCH);
                    for c in 0..3 {
                        for y in 0..PATCH {
                            for x in 0..PATCH {
                                out.push((rgb[((py + y) * w + px + x) * 3 + c] / 255.0 - 0.5) / 0.5);
                            }
                        }
                    }
                }
            }
        }
    }
    Ok((out, (gh, gw)))
}

/// Each patch's (row, column), in merge-block order
fn positions(gh: usize, gw: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::with_capacity(gh * gw);
    for br in 0..gh / MERGE {
        for bc in 0..gw / MERGE {
            for ir in 0..MERGE {
                for ic in 0..MERGE {
                    v.push((br * MERGE + ir, bc * MERGE + ic));
                }
            }
        }
    }
    v
}

/// Bilinear taps (align_corners) of `n` output positions over `side` grid points: (low, high, weight of high)
fn taps(i: usize, n: usize, side: usize) -> (usize, usize, f32) {
    let src = if n > 1 { i as f64 * (side - 1) as f64 / (n - 1) as f64 } else { 0.0 };
    let lo = (src.floor() as usize).min(side - 1);
    let hi = (lo + 1).min(side - 1);
    (lo, hi, (src - lo as f64) as f32)
}

/// A tensor's values as half (F16 as stored; BF16 and F32 converted)
fn halves(f: &Gguf, t: &nextcuda_gguf::Tensor) -> Result<Vec<u8>> {
    let b = f.read(t).map_err(ge)?;
    Ok(match t.ty {
        GType::F16 => b,
        GType::BF16 => b.chunks_exact(2).flat_map(|c| f32_to_f16(f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).to_le_bytes()).collect(),
        GType::F32 => b.chunks_exact(4).flat_map(|c| f32_to_f16(f32::from_le_bytes([c[0], c[1], c[2], c[3]])).to_le_bytes()).collect(),
        other => return Err(Error(format!("{}: {} - the vision tower takes F16, BF16 or F32", t.name, other.name()))),
    })
}

/// A float tensor of the file as float32 on the host, whatever its stored type
fn host_f32(f: &Gguf, t: &nextcuda_gguf::Tensor) -> Result<Vec<f32>> {
    let b = f.read(t).map_err(ge)?;
    Ok(match t.ty {
        GType::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        GType::F16 => b.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        GType::BF16 => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
        other => return Err(Error(format!("{}: a {} vector", t.name, other.name()))),
    })
}

impl Vision {
    /// The tower from the projector file, onto `gpu`
    pub fn load(path: &Path, gpu: &Arc<Gpu>, log: &mut dyn FnMut(String)) -> Result<Vision> {
        let t0 = std::time::Instant::now();
        let f = Gguf::open(path).map_err(ge)?;
        if f.meta("clip.projector_type").and_then(|v| v.as_str()) != Some("qwen3vl_merger") {
            return Err(Error(format!("{}: not a Qwen3-VL projector (clip.projector_type qwen3vl_merger)", path.display())));
        }
        let tensor = |n: &str| f.tensor(n).ok_or_else(|| Error(format!("{}: no tensor {n}", path.display())));
        let mat = |n: &str| -> Result<DevBuf> {
            let raw = halves(&f, tensor(n)?)?;
            let b = DevBuf::new(gpu, raw.len())?;
            b.write(0, &raw)?;
            Ok(b)
        };
        let vec = |n: &str| -> Result<DevBuf> { DevBuf::from_f32(gpu, &host_f32(&f, tensor(n)?)?) };
        let hidden = tensor("v.patch_embd.bias")?.elements() as usize;
        let ffn = tensor("v.blk.0.ffn_up.bias")?.elements() as usize;
        let out = tensor("mm.2.bias")?.elements() as usize;
        let heads = f.meta("clip.vision.attention.head_count").and_then(|v| v.as_u64()).map_or(hidden / 72, |v| v as usize);
        let n_blocks = (0..).take_while(|i| f.tensor(&format!("v.blk.{i}.ln1.weight")).is_some()).count();
        if f.tensors.iter().any(|t| t.name.starts_with("v.deepstack")) {
            return Err(Error(format!("{}: deepstack mergers - a Qwen3-VL projector, not Qwen3.8's (this engine's tower has none)", path.display())));
        }
        let mut blocks = Vec::with_capacity(n_blocks);
        for i in 0..n_blocks {
            let p = |s: &str| format!("v.blk.{i}.{s}");
            blocks.push(Block {
                n1w: vec(&p("ln1.weight"))?, n1b: vec(&p("ln1.bias"))?,
                qkv: mat(&p("attn_qkv.weight"))?, qkv_b: vec(&p("attn_qkv.bias"))?,
                proj: mat(&p("attn_out.weight"))?, proj_b: vec(&p("attn_out.bias"))?,
                n2w: vec(&p("ln2.weight"))?, n2b: vec(&p("ln2.bias"))?,
                fc1: mat(&p("ffn_up.weight"))?, fc1_b: vec(&p("ffn_up.bias"))?,
                fc2: mat(&p("ffn_down.weight"))?, fc2_b: vec(&p("ffn_down.bias"))?,
            });
        }
        let pos = host_f32(&f, tensor("v.position_embd.weight")?)?;
        let side = ((pos.len() / hidden) as f64).sqrt() as usize;
        // the patch kernel's two temporal slices, summed (a picture is its own frame pair): [hidden, 3, 16, 16]
        let (w0, w1) = (host_f32(&f, tensor("v.patch_embd.weight")?)?, host_f32(&f, tensor("v.patch_embd.weight.1")?)?);
        if w0.len() != hidden * PATCH_IN || w1.len() != w0.len() {
            return Err(Error(format!("the vision tower's patch kernel holds {} values, not {hidden} x {PATCH_IN}", w0.len())));
        }
        let eff: Vec<u8> = w0.iter().zip(&w1).flat_map(|(a, b)| f32_to_f16(a + b).to_le_bytes()).collect();
        let patch = DevBuf::new(gpu, eff.len())?;
        patch.write(0, &eff)?;
        let bytes = card_bytes(&f);
        let (fc1, fc2) = (mat("mm.0.weight")?, mat("mm.2.weight")?);
        let v = Vision {
            gpu: gpu.clone(),
            patch,
            patch_b: vec("v.patch_embd.bias")?,
            pos,
            side,
            blocks,
            nw: vec("v.post_ln.weight")?,
            nb: vec("v.post_ln.bias")?,
            fc1,
            fc1_b: vec("mm.0.bias")?,
            fc2,
            fc2_b: vec("mm.2.bias")?,
            hidden,
            heads,
            ffn,
            out,
            bytes,
        };
        log(format!("qwen35 vision: {n_blocks} blocks ({hidden} wide, {heads} heads), out {out}, {:.2} GiB on {} in {:.1} s", bytes as f64 / (1u64 << 30) as f64,
                    gpu.name, t0.elapsed().as_secs_f64()));
        Ok(v)
    }

    /// Device memory `see_patches` takes from `alloc` for a picture of n patches (its activations)
    pub fn work_bytes(&self, n: usize) -> usize {
        let d = self.hidden;
        let k = d * MERGE * MERGE;
        n * (4 * d + 2 * d + 2 * 3 * d + 2 * d + 4 * d + 2 * self.ffn + 2 * PATCH_IN) + n * 4 * d + n * 72 * 4
            + (n / 4) * (2 * k + 4 * self.out) + n * d * 2 + (16 << 20)
    }

    /// A picture's merged tokens [n / 4, out] as float32 on the host, from RGB float32 [H, W, 3] in 0..255 (sides
    /// multiples of 32): the activations from `alloc` (the engine's arena), the kernels on their own stream (`nsd`)
    pub fn see(&self, nsd: &Nsd, rgb: &[f32], hh: usize, ww: usize, alloc: &dyn Fn(usize) -> Result<DevBuf>) -> Result<(Vec<f32>, (usize, usize))> {
        let (px, (gh, gw)) = patches(rgb, hh, ww)?;
        let n = gh * gw;
        let d = self.hidden;
        let pos = positions(gh, gw);
        // the learned grid resampled to this one, added to the patches' embedding
        let mut pe = vec![0f32; n * d];
        for (i, (r, c)) in pos.iter().enumerate() {
            let (r0, r1, fr) = taps(*r, gh, self.side);
            let (c0, c1, fc) = taps(*c, gw, self.side);
            for (row, wgt) in [(r0 * self.side + c0, (1.0 - fr) * (1.0 - fc)), (r0 * self.side + c1, (1.0 - fr) * fc), (r1 * self.side + c0, fr * (1.0 - fc)),
                               (r1 * self.side + c1, fr * fc)] {
                for j in 0..d {
                    pe[i * d + j] += wgt * self.pos[row * d + j];
                }
            }
        }
        // 2-D RoPE: angles of the row, then of the column (a quarter of the head each), over the head's two halves
        let hd = d / self.heads;
        let q = hd / 4;
        let inv: Vec<f32> = (0..q).map(|i| (1.0 / THETA.powf((2 * i) as f64 / (hd / 2) as f64)) as f32).collect();
        let mut cs = vec![0f32; n * hd];
        for (t, (r, c)) in pos.iter().enumerate() {
            for i in 0..hd / 2 {
                let a = if i < q { *r as f32 * inv[i] } else { *c as f32 * inv[i - q] };
                cs[t * hd + 2 * i] = a.cos();
                cs[t * hd + 2 * i + 1] = a.sin();
            }
        }
        let up = |v: &[f32]| -> Result<DevBuf> {
            let b = alloc(v.len() * 4)?;
            b.write(0, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
            Ok(b)
        };
        let cs = up(&cs)?;
        let input = alloc(n * PATCH_IN * 2)?;
        input.write(0, &px.iter().flat_map(|v| f32_to_f16(*v).to_le_bytes()).collect::<Vec<u8>>())?;
        let x = alloc(n * d * 4)?;
        nsd.linear(input.ptr(), Dt::F16, n, PATCH_IN, self.patch.ptr(), d, self.patch_b.ptr(), x.ptr(), Dt::F32)?;
        let peb = up(&pe)?;
        nsd.gate_add(x.ptr(), Dt::F32, n, d, peb.ptr(), Dt::F32, none(), none())?;
        // half inside a block, float32 between them
        let (hb, qkv, att) = (alloc(n * d * 2)?, alloc(n * 3 * d * 2)?, alloc(n * d * 2)?);
        let o = alloc(n * d * 4)?;
        let mlp = alloc(n * self.ffn * 2)?;
        let at = |b: &DevBuf, i: usize| -> *mut std::ffi::c_void { b.ptr().cast::<u16>().wrapping_add(i).cast() };
        for b in &self.blocks {
            nsd.layer_norm(x.ptr(), Dt::F32, n, d, b.n1w.ptr(), b.n1b.ptr(), EPS, hb.ptr(), Dt::F16)?;
            nsd.linear(hb.ptr(), Dt::F16, n, d, b.qkv.ptr(), 3 * d, b.qkv_b.ptr(), qkv.ptr(), Dt::F16)?;
            nsd.rope(qkv.ptr(), Dt::F16, n, self.heads, hd, 3 * d, cs.ptr(), hd)?;
            nsd.rope(at(&qkv, d), Dt::F16, n, self.heads, hd, 3 * d, cs.ptr(), hd)?;
            nsd.attention_qk(qkv.ptr(), n, 3 * d, at(&qkv, d), at(&qkv, 2 * d), n, 3 * d, Dt::F16, self.heads, hd, att.ptr(), Dt::F16)?;
            nsd.linear(att.ptr(), Dt::F16, n, d, b.proj.ptr(), d, b.proj_b.ptr(), o.ptr(), Dt::F32)?;
            nsd.gate_add(x.ptr(), Dt::F32, n, d, o.ptr(), Dt::F32, none(), none())?;
            nsd.layer_norm(x.ptr(), Dt::F32, n, d, b.n2w.ptr(), b.n2b.ptr(), EPS, hb.ptr(), Dt::F16)?;
            nsd.linear(hb.ptr(), Dt::F16, n, d, b.fc1.ptr(), self.ffn, b.fc1_b.ptr(), mlp.ptr(), Dt::F16)?;
            nsd.gelu(mlp.ptr(), Dt::F16, n * self.ffn, true)?;
            nsd.linear(mlp.ptr(), Dt::F16, n, self.ffn, b.fc2.ptr(), d, b.fc2_b.ptr(), o.ptr(), Dt::F32)?;
            nsd.gate_add(x.ptr(), Dt::F32, n, d, o.ptr(), Dt::F32, none(), none())?;
        }
        // the merger: each token normed, a 2 x 2 block's four as one row
        let (k, rows) = (d * MERGE * MERGE, n / (MERGE * MERGE));
        let normed = alloc(n * d * 2)?;
        nsd.layer_norm(x.ptr(), Dt::F32, n, d, self.nw.ptr(), self.nb.ptr(), EPS, normed.ptr(), Dt::F16)?;
        let h = alloc(rows * k * 2)?;
        nsd.linear(normed.ptr(), Dt::F16, rows, k, self.fc1.ptr(), k, self.fc1_b.ptr(), h.ptr(), Dt::F16)?;
        nsd.gelu(h.ptr(), Dt::F16, rows * k, false)?;
        let tok = alloc(rows * self.out * 4)?;
        nsd.linear(h.ptr(), Dt::F16, rows, k, self.fc2.ptr(), self.out, self.fc2_b.ptr(), tok.ptr(), Dt::F32)?;
        nsd.wait()?;
        let _ = &self.gpu;
        Ok((tok.to_f32()?, (gw / MERGE, gh / MERGE)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_as_mtmd_makes_them() {
        // rounded to 32s; within 8 ..= 4,096 tokens
        assert_eq!(target_size(640, 480), (640, 480));
        assert_eq!(target_size(1000, 750), (992, 736));
        let (w, h) = target_size(4000, 3000);
        assert!(w * h <= MAX_TOKENS * ALIGN * ALIGN && w % ALIGN == 0 && h % ALIGN == 0);
        let (w, h) = target_size(40, 30);
        assert!(w * h >= MIN_TOKENS * ALIGN * ALIGN);
        assert_eq!(positions(2, 4)[4], (0, 2));
        assert_eq!(taps(0, 32, 48), (0, 1, 0.0));
    }
}
