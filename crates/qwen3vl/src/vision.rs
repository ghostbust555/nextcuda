//! Qwen3-VL's vision tower: pictures to the language model's image tokens (transformers' Qwen3VLVisionModel), from
//! the llama.cpp projector file (`mmproj-Qwen3VL-8B-Instruct-F16.gguf`: `v.*` and `mm.*`), half weights and half
//! activations on the GPU with a float32 residual stream. nextsycl's tower read the bf16 safetensors and ran
//! float32; this is the same computation in the types a 12 GB card can hold beside the denoiser.
//!
//! ```text
//!   picture (RGB, sides multiples of 32) -> 16 x 16 patches, (x / 255 - 0.5) / 0.5, in 2 x 2 merge-block order
//!     (a picture is its own frame pair: the two temporal slices of the patch kernel are summed)
//!   -> linear 768 -> 1,152 + the learned 48 x 48 position grid resampled (bilinear, align_corners) to the picture's
//!   -> 27 blocks: x += proj(attention(rope2d(qkv(layernorm(x)))));  x += fc2(gelu_tanh(fc1(layernorm(x))))
//!      16 heads of 72, 2-D RoPE (height and width angles side by side); after blocks 8, 16, 24 a deepstack merger
//!   -> merger: layernorm (1,152), each 2 x 2 block's four tokens as one row of 4,608 -> fc1, gelu, fc2 -> 4,096
//!   deepstack mergers: the four tokens as one row first, then layernorm (4,608), fc1, gelu, fc2 -> 4,096
//! ```

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
/// the blocks after which the deepstack mergers read (Qwen3-VL 8B's deepstack_visual_indexes)
const DEEP_AT: [usize; 3] = [8, 16, 24];

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

struct Merger {
    nw: DevBuf,
    nb: DevBuf,
    /// the norm is over the four tokens' row (deepstack) rather than each token (the final merger)
    post: bool,
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
    deep_at: Vec<usize>,
    deep: Vec<Merger>,
    merger: Merger,
    pub hidden: usize,
    heads: usize,
    ffn: usize,
    pub out: usize,
}

/// What a picture gives the language model: its tokens [n, out] and the deepstack features (one [n, out] each),
/// float32
pub struct Seen {
    pub tokens: DevBuf,
    pub deep: Vec<DevBuf>,
    pub n: usize,
    /// the patch grid (height, width) in 16-pixel patches
    pub grid: (usize, usize),
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

/// The processor's patches of a picture: RGB float32 [H, W, 3] in 0..255 (sides multiples of 32) -> [n, 768] in
/// merge-block order (block row, block column, row in block, column in block), each patch (channel, y, x)
pub fn patches(rgb: &[f32], h: usize, w: usize) -> Result<(Vec<f32>, (usize, usize))> {
    if h % (PATCH * MERGE) != 0 || w % (PATCH * MERGE) != 0 || rgb.len() != h * w * 3 {
        return Err(Error(format!("a {w}x{h} picture: the vision tower takes sides that are multiples of {}", PATCH * MERGE)));
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

impl Vision {
    /// The tower from the projector file
    pub fn load(path: &Path, gpu: &Arc<Gpu>, log: &mut dyn FnMut(String)) -> Result<Vision> {
        let t0 = std::time::Instant::now();
        let f = Gguf::open(path).map_err(ge)?;
        let tensor = |n: &str| f.tensor(n).ok_or_else(|| Error(format!("{}: no tensor {n}", path.display())));
        // a matrix as stored (F16), on the card
        let mat = |n: &str| -> Result<DevBuf> {
            let t = tensor(n)?;
            if t.ty != GType::F16 {
                return Err(Error(format!("{n}: {} - the vision tower's weights are read as F16", t.ty.name())));
            }
            let raw = f.read(t).map_err(ge)?;
            let b = DevBuf::new(gpu, raw.len())?;
            b.write(0, &raw)?;
            Ok(b)
        };
        // a vector (bias, norm) as float32
        let vec = |n: &str| -> Result<DevBuf> { DevBuf::from_f32(gpu, &host_f32(&f, tensor(n)?)?) };
        let hidden = tensor("v.patch_embd.bias")?.elements() as usize;
        let ffn = tensor("v.blk.0.ffn_up.bias")?.elements() as usize;
        let out = tensor("mm.2.bias")?.elements() as usize;
        let n_blocks = (0..).take_while(|i| f.tensor(&format!("v.blk.{i}.ln1.weight")).is_some()).count();
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
        let deep = DEEP_AT
            .iter()
            .map(|l| {
                let p = format!("v.deepstack.{l}");
                Ok(Merger {
                    nw: vec(&format!("{p}.norm.weight"))?, nb: vec(&format!("{p}.norm.bias"))?, post: true,
                    fc1: mat(&format!("{p}.fc1.weight"))?, fc1_b: vec(&format!("{p}.fc1.bias"))?,
                    fc2: mat(&format!("{p}.fc2.weight"))?, fc2_b: vec(&format!("{p}.fc2.bias"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if n_blocks <= DEEP_AT[DEEP_AT.len() - 1] {
            return Err(Error(format!("the vision tower: {n_blocks} blocks - not Qwen3-VL 8B's 27")));
        }
        let pos = host_f32(&f, tensor("v.position_embd.weight")?)?;
        let side = ((pos.len() / hidden) as f64).sqrt() as usize;
        // the patch kernel's two temporal slices, summed (a picture is its own frame pair): [hidden, 3, 16, 16]
        let (w0, w1) = (f.read(tensor("v.patch_embd.weight")?).map_err(ge)?, f.read(tensor("v.patch_embd.weight.1")?).map_err(ge)?);
        if w0.len() != hidden * PATCH_IN * 2 || w1.len() != w0.len() {
            return Err(Error(format!("the vision tower's patch kernel is {} bytes, not {hidden} x {PATCH_IN} halves", w0.len())));
        }
        let eff: Vec<u8> = w0
            .chunks_exact(2)
            .zip(w1.chunks_exact(2))
            .flat_map(|(a, b)| f32_to_f16(f16_to_f32(u16::from_le_bytes([a[0], a[1]])) + f16_to_f32(u16::from_le_bytes([b[0], b[1]]))).to_le_bytes())
            .collect();
        let patch = DevBuf::new(gpu, eff.len())?;
        patch.write(0, &eff)?;
        let v = Vision {
            gpu: gpu.clone(),
            patch,
            patch_b: vec("v.patch_embd.bias")?,
            pos,
            side,
            blocks,
            deep_at: DEEP_AT.to_vec(),
            deep,
            merger: Merger {
                nw: vec("v.post_ln.weight")?, nb: vec("v.post_ln.bias")?, post: false,
                fc1: mat("mm.0.weight")?, fc1_b: vec("mm.0.bias")?,
                fc2: mat("mm.2.weight")?, fc2_b: vec("mm.2.bias")?,
            },
            hidden,
            heads: hidden / 72,
            ffn,
            out,
        };
        log(format!("qwen3vl vision: {n_blocks} blocks ({hidden} wide), deepstack after {:?}, on {} in {:.1} s", v.deep_at, gpu.name, t0.elapsed().as_secs_f64()));
        Ok(v)
    }

    fn merge(&self, nsd: &Nsd, m: &Merger, x: &DevBuf, n: usize) -> Result<DevBuf> {
        let gpu = &self.gpu;
        let (d, k) = (self.hidden, self.hidden * MERGE * MERGE);
        let rows = n / (MERGE * MERGE);
        let normed = DevBuf::new(gpu, n * d * 2)?;
        if m.post {
            nsd.layer_norm(x.ptr(), Dt::F32, rows, k, m.nw.ptr(), m.nb.ptr(), EPS, normed.ptr(), Dt::F16)?;
        } else {
            nsd.layer_norm(x.ptr(), Dt::F32, n, d, m.nw.ptr(), m.nb.ptr(), EPS, normed.ptr(), Dt::F16)?;
        }
        let h = DevBuf::new(gpu, rows * k * 2)?;
        nsd.linear(normed.ptr(), Dt::F16, rows, k, m.fc1.ptr(), k, m.fc1_b.ptr(), h.ptr(), Dt::F16)?;
        nsd.gelu(h.ptr(), Dt::F16, rows * k, false)?;
        let o = DevBuf::f32(gpu, rows * self.out)?;
        nsd.linear(h.ptr(), Dt::F16, rows, k, m.fc2.ptr(), self.out, m.fc2_b.ptr(), o.ptr(), Dt::F32)?;
        nsd.wait()?;
        Ok(o)
    }

    /// A picture's tokens: RGB float32 [H, W, 3] in 0..255
    pub fn see(&self, nsd: &Nsd, rgb: &[f32], hh: usize, ww: usize) -> Result<Seen> {
        let (px, grid) = patches(rgb, hh, ww)?;
        self.see_patches(nsd, &px, grid)
    }

    /// The tokens from the processor's patches [n, 768] of a (rows, columns) patch grid
    pub fn see_patches(&self, nsd: &Nsd, px: &[f32], (gh, gw): (usize, usize)) -> Result<Seen> {
        let gpu = &self.gpu;
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
        let cs = DevBuf::from_f32(gpu, &cs)?;
        let input = DevBuf::new(gpu, n * PATCH_IN * 2)?;
        input.write(0, &px.iter().flat_map(|v| f32_to_f16(*v).to_le_bytes()).collect::<Vec<u8>>())?;
        let x = DevBuf::f32(gpu, n * d)?;
        nsd.linear(input.ptr(), Dt::F16, n, PATCH_IN, self.patch.ptr(), d, self.patch_b.ptr(), x.ptr(), Dt::F32)?;
        let peb = DevBuf::from_f32(gpu, &pe)?;
        nsd.gate_add(x.ptr(), Dt::F32, n, d, peb.ptr(), Dt::F32, none(), none())?;
        // half inside a block, float32 between them
        let (hb, qkv, att) = (DevBuf::new(gpu, n * d * 2)?, DevBuf::new(gpu, n * 3 * d * 2)?, DevBuf::new(gpu, n * d * 2)?);
        let o = DevBuf::f32(gpu, n * d)?;
        let mlp = DevBuf::new(gpu, n * self.ffn * 2)?;
        let mut deep = Vec::new();
        let at = |b: &DevBuf, i: usize| -> *mut std::ffi::c_void { b.ptr().cast::<u16>().wrapping_add(i).cast() };
        for (bi, b) in self.blocks.iter().enumerate() {
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
            if let Some(k) = self.deep_at.iter().position(|l| *l == bi) {
                deep.push(self.merge(nsd, &self.deep[k], &x, n)?);
            }
        }
        let tokens = self.merge(nsd, &self.merger, &x, n)?;
        nsd.wait()?;
        Ok(Seen { tokens, deep, n: n / (MERGE * MERGE), grid: (gh, gw) })
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patches_come_in_merge_blocks() {
        // a 32 x 64 picture: 2 x 4 patches, two merge blocks side by side
        let (h, w) = (32, 64);
        let rgb: Vec<f32> = (0..h * w * 3).map(|i| ((i / 3) % w) as f32).collect();
        let (px, grid) = patches(&rgb, h, w).unwrap();
        assert_eq!(grid, (2, 4));
        assert_eq!(px.len(), 8 * PATCH_IN);
        // the 2nd patch of the 1st block is column 1 (x from 16); the 5th patch (2nd block) column 2 (x from 32)
        let first = |p: usize| px[p * PATCH_IN] * 0.5 * 255.0 + 127.5;
        assert_eq!(first(1), 16.0);
        assert_eq!(first(4), 32.0);
        assert_eq!(positions(2, 4)[4], (0, 2));
        assert_eq!(taps(0, 32, 48), (0, 1, 0.0));
        assert_eq!(taps(31, 32, 48).1, 47);
    }
}
