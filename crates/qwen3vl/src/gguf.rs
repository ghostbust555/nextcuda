//! The same text encoder from a llama.cpp GGUF (Qwen3-VL's language model as `Qwen3VL-8B-Instruct-Q4_K_M.gguf`
//! holds it): Q4_K / Q6_K matrices, the tokenizer in the file's metadata. A layer's matrices go to the card as they
//! are stored and are made half a slice of rows at a time by the quantized linear, so an 8B encoder takes the
//! card's memory for one layer (about 150 MiB) and not for five gigabytes - the 12 GB cards hold the denoiser beside it.
//!
//! One layer, as in `lib.rs` (the reference's): activations float32 between the layers, half inside them.
//!
//! ```text
//!   h = rmsnorm(x)                  q k v = h [Wq|Wk|Wv]   (32 query heads, 8 key/value heads, 128 features each)
//!   q, k: per-head rmsnorm + rotary positions (theta 5e6, pairs (i, 64 + i))
//!   x += attention(q, k, v) Wo      causal; 4 query heads share each key/value head
//!   h = rmsnorm(x)
//!   x += (silu(h Wgate) * (h Wup)) Wdown
//! ```
//!
//! The result is the last layer's hidden state for every token, before the final norm (as the image pipelines take it).

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use nextcuda_core::{DevBuf, Error, Gpu, Result};
use nextcuda_diffusion::kernels::{none, Dt, Nsd};
use nextcuda_gguf::{GType, Gguf};
use nextcuda_tok::Tokenizer;

use crate::vision::{self, Seen};

const HEAD: usize = 128;

fn ge(e: nextcuda_gguf::Error) -> Error {
    Error(e.0)
}

pub struct GgufTextEncoder {
    file: Gguf,
    pub tok: Tokenizer,
    gpu: Arc<Gpu>,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub ffn: usize,
    layers: usize,
    theta: f64,
    eps: f32,
    /// the layers' bytes once read, in host memory (`NC_TE_HOST_CACHE=0`: read from the file each time): the encoder
    /// is streamed to the card for every prompt, and a read of 5 GB costs more than the copy
    cache: Option<Mutex<HashMap<String, Arc<Vec<u8>>>>>,
}

/// A tensor of a layer on the card, as stored
struct W {
    buf: DevBuf,
    qtype: u32,
    rows: usize,
}

impl GgufTextEncoder {
    /// Open the file: the tokenizer and the geometry from its metadata; the weights are read per layer at encode time
    pub fn load(path: &Path, gpu: &Arc<Gpu>, log: &mut dyn FnMut(String)) -> Result<GgufTextEncoder> {
        let t0 = std::time::Instant::now();
        let file = Gguf::open(path).map_err(ge)?;
        let arch = file.architecture().to_string();
        if arch != "qwen3vl" && arch != "qwen3" {
            return Err(Error(format!("{}: a {arch} model, not Qwen3-VL", path.display())));
        }
        let geti = |k: &str| -> Result<usize> {
            file.arch_meta(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| Error(format!("{}: no {arch}.{k}", path.display())))
        };
        let getf = |k: &str, d: f64| file.arch_meta(k).and_then(|v| v.as_f64()).unwrap_or(d);
        let layers = geti("block_count")?;
        let hidden = geti("embedding_length")?;
        let heads = geti("attention.head_count")?;
        let kv_heads = geti("attention.head_count_kv")?;
        let ffn = geti("feed_forward_length")?;
        let theta = getf("rope.freq_base", 5_000_000.0);
        let eps = getf("attention.layer_norm_rms_epsilon", 1e-6) as f32;
        let tok = Tokenizer::from_gguf(&file).map_err(Error)?;
        log(format!("qwen3vl text encoder (GGUF): {layers} layers, {hidden} wide, {heads} / {kv_heads} heads, {ffn} ffn; {:.1} s", t0.elapsed().as_secs_f64()));
        Ok(GgufTextEncoder {
            file,
            tok,
            gpu: gpu.clone(),
            hidden,
            heads,
            kv_heads,
            ffn,
            layers,
            theta,
            eps,
            cache: (std::env::var("NC_TE_HOST_CACHE").map_or(true, |v| v != "0")).then(|| Mutex::new(HashMap::new())),
        })
    }

    /// `text` tokenized as it is (the caller builds the chat template)
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        self.tok.encode(text)
    }

    fn tensor(&self, name: &str) -> Result<&nextcuda_gguf::Tensor> {
        self.file.tensor(name).ok_or_else(|| Error(format!("{}: no tensor {name}", self.file.paths[0].display())))
    }

    /// A tensor's bytes, from the host cache after the first read
    fn bytes(&self, name: &str, t: &nextcuda_gguf::Tensor) -> Result<Arc<Vec<u8>>> {
        if let Some(c) = &self.cache {
            if let Some(b) = c.lock().unwrap().get(name) {
                return Ok(b.clone());
            }
        }
        let b = Arc::new(self.file.read(t).map_err(ge)?);
        if let Some(c) = &self.cache {
            c.lock().unwrap().insert(name.to_string(), b.clone());
        }
        Ok(b)
    }

    /// A matrix of the file on the card, as stored
    fn matrix(&self, name: &str) -> Result<W> {
        let t = self.tensor(name)?;
        let code = t.ty.code();
        if ![0, 1, 8, 12, 13, 14, 30].contains(&code) {
            return Err(Error(format!("{name}: {} weights are not read yet", t.ty.name())));
        }
        let raw = self.bytes(name, t)?;
        let buf = DevBuf::new(&self.gpu, raw.len())?;
        buf.write(0, &raw)?;
        Ok(W { buf, qtype: code, rows: t.shape[0] as usize })
    }

    /// A float32 vector of the file (a norm's weight) on the card
    fn vector(&self, name: &str) -> Result<DevBuf> {
        let t = self.tensor(name)?;
        let b = self.file.read(t).map_err(ge)?;
        let v: Vec<f32> = match t.ty {
            GType::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            GType::F16 => b.chunks_exact(2).map(|c| nextcuda_gguf::safetensors::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            GType::BF16 => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            other => return Err(Error(format!("{name}: a {} vector", other.name()))),
        };
        DevBuf::from_f32(&self.gpu, &v)
    }

    /// The embedding rows of `ids`, float32 [S, hidden] on the card: the rows' bytes read from the file, made float32
    /// by the dequant kernel
    fn embed(&self, nsd: &Nsd, ids: &[u32]) -> Result<DevBuf> {
        let t = self.tensor("token_embd.weight")?;
        let d = self.hidden;
        let rb = t.ty.bytes(d as u64).ok_or_else(|| Error("token_embd.weight: a type with no row size".into()))? as usize;
        let vocab = t.shape[0] as usize;
        let mut raw = vec![0u8; ids.len() * rb];
        for (i, id) in ids.iter().enumerate() {
            if *id as usize >= vocab {
                return Err(Error(format!("token {id} is outside the {vocab} of the embedding")));
            }
            self.file.read_into(t, *id as u64 * rb as u64, &mut raw[i * rb..(i + 1) * rb]).map_err(ge)?;
        }
        let src = DevBuf::new(&self.gpu, raw.len())?;
        src.write(0, &raw)?;
        let out = DevBuf::f32(&self.gpu, ids.len() * d)?;
        nsd.dequant(src.ptr(), t.ty.code(), ids.len() * d, out.ptr(), Dt::F32)?;
        nsd.wait()?;
        Ok(out)
    }

    /// The hidden state after the last layer for every token of `ids`: float32 [S, hidden] on the card
    pub fn encode_ids(&self, nsd: &Nsd, ids: &[u32]) -> Result<DevBuf> {
        self.encode_layers(nsd, ids, self.layers)
    }

    /// The hidden state after the first `n` layers
    pub fn encode_layers(&self, nsd: &Nsd, ids: &[u32], n: usize) -> Result<DevBuf> {
        self.encode_seen(nsd, ids, &[], n)
    }

    /// Each token's (time, height, width) position (transformers' get_rope_index): text tokens advance one on all
    /// three axes; a picture's run of `<|image_pad|>` tokens (its merged grid, row by row) sits at the run's start
    /// plus its row and column, and the next text starts past the larger side
    pub fn positions(ids: &[u32], pad: u32, grids: &[(usize, usize)]) -> Result<Vec<[i64; 3]>> {
        let mut out = Vec::with_capacity(ids.len());
        let (mut p, mut i, mut g) = (0i64, 0usize, 0usize);
        while i < ids.len() {
            if ids[i] != pad {
                out.push([p, p, p]);
                p += 1;
                i += 1;
                continue;
            }
            let (gh, gw) = *grids.get(g).ok_or_else(|| Error("more pictures in the prompt than were given".into()))?;
            let (lh, lw) = (gh / vision::MERGE, gw / vision::MERGE);
            for r in 0..lh {
                for c in 0..lw {
                    if ids.get(i) != Some(&pad) {
                        return Err(Error(format!("picture {g}: its run of image tokens is shorter than its {lh} x {lw} grid")));
                    }
                    out.push([p, p + r as i64, p + c as i64]);
                    i += 1;
                }
            }
            p += lh.max(lw) as i64;
            g += 1;
        }
        Ok(out)
    }

    /// The hidden state after `n` layers for `ids` with pictures in it: `seen` (the vision tower's, in the order of
    /// their `<|image_pad|>` runs) at their tokens, Qwen3-VL's interleaved M-RoPE (24 / 20 / 20 pairs: t, then h and
    /// w on every third pair from the second), the deepstack features added after the first layers
    pub fn encode_seen(&self, nsd: &Nsd, ids: &[u32], seen: &[&Seen], n: usize) -> Result<DevBuf> {
        let s = ids.len();
        let (d, h, kv, f) = (self.hidden, self.heads, self.kv_heads, self.ffn);
        let qkv_w = (h + 2 * kv) * HEAD;
        let gpu = &self.gpu;
        let x = self.embed(nsd, ids)?;
        let pad = self.tok.id("<|image_pad|>").unwrap_or(u32::MAX);
        // each picture's rows: the start of its run of image tokens
        let mut runs = Vec::new();
        let mut i = 0;
        while i < s {
            if ids[i] == pad {
                let st = i;
                while i < s && ids[i] == pad {
                    i += 1;
                }
                runs.push((st, i - st));
            } else {
                i += 1;
            }
        }
        if runs.len() != seen.len() || runs.iter().zip(seen).any(|((_, len), v)| *len != v.n) {
            return Err(Error(format!("the prompt holds {} picture runs {:?}; {} pictures were seen", runs.len(), runs.iter().map(|r| r.1).collect::<Vec<_>>(),
                                     seen.len())));
        }
        for ((st, len), v) in runs.iter().zip(seen) {
            x.copy_within(st * d * 4, &v.tokens, 0, len * d * 4)?;
        }
        let pos: Vec<[i64; 3]> = if seen.is_empty() {
            (0..s as i64).map(|p| [p, p, p]).collect()
        } else {
            Self::positions(ids, pad, &seen.iter().map(|v| v.grid).collect::<Vec<_>>())?
        };
        let hb = DevBuf::new(gpu, s * d * 2)?;
        let qkv = DevBuf::new(gpu, s * qkv_w * 2)?;
        let att = DevBuf::new(gpu, s * h * HEAD * 2)?;
        let o = DevBuf::f32(gpu, s * d)?;
        let gu = DevBuf::new(gpu, s * 2 * f * 2)?;
        let act = DevBuf::new(gpu, s * f * 2)?;
        // (cos, sin) per token and pair (i, 64 + i); pair i's axis: interleaved M-RoPE (the three agree on text)
        let mut cs = vec![0f32; s * HEAD];
        for (t, p) in pos.iter().enumerate() {
            for i in 0..HEAD / 2 {
                let axis = if i < 60 && i % 3 == 1 { 1 } else if i < 60 && i % 3 == 2 { 2 } else { 0 };
                let a = p[axis] as f64 / self.theta.powf(2.0 * i as f64 / HEAD as f64);
                cs[t * HEAD + 2 * i] = a.cos() as f32;
                cs[t * HEAD + 2 * i + 1] = a.sin() as f32;
            }
        }
        let cs = DevBuf::from_f32(gpu, &cs)?;
        let half_at = |b: &DevBuf, at: usize| -> *mut std::ffi::c_void { b.ptr().cast::<u16>().wrapping_add(at).cast() };
        for l in 0..n.min(self.layers) {
            let p = format!("blk.{l}.");
            let (in_norm, post_norm) = (self.vector(&format!("{p}attn_norm.weight"))?, self.vector(&format!("{p}ffn_norm.weight"))?);
            let (q_norm, k_norm) = (self.vector(&format!("{p}attn_q_norm.weight"))?, self.vector(&format!("{p}attn_k_norm.weight"))?);
            nsd.rms_norm_mod(x.ptr(), Dt::F32, s, d, in_norm.ptr(), self.eps, none(), none(), none(), hb.ptr(), Dt::F16)?;
            let mut off = 0;
            for name in ["attn_q", "attn_k", "attn_v"] {
                let w = self.matrix(&format!("{p}{name}.weight"))?;
                nsd.linear_q(hb.ptr(), Dt::F16, s, d, w.buf.ptr(), w.qtype, w.rows, none(), half_at(&qkv, off), Dt::F16, qkv_w)?;
                off += w.rows;
                // the weights are freed when `w` goes: the work on them first
                nsd.wait()?;
            }
            nsd.rms_rope(qkv.ptr(), Dt::F16, s, h, HEAD, qkv_w, q_norm.ptr(), self.eps, cs.ptr(), HEAD)?;
            nsd.rms_rope(half_at(&qkv, h * HEAD), Dt::F16, s, kv, HEAD, qkv_w, k_norm.ptr(), self.eps, cs.ptr(), HEAD)?;
            nsd.attention_causal(qkv.ptr(), half_at(&qkv, h * HEAD), half_at(&qkv, (h + kv) * HEAD), Dt::F16, s, h, kv, HEAD, qkv_w, qkv_w, att.ptr())?;
            {
                let w = self.matrix(&format!("{p}attn_output.weight"))?;
                nsd.linear_q(att.ptr(), Dt::F16, s, h * HEAD, w.buf.ptr(), w.qtype, w.rows, none(), o.ptr(), Dt::F32, 0)?;
                nsd.wait()?;
            }
            nsd.gate_add(x.ptr(), Dt::F32, s, d, o.ptr(), Dt::F32, none(), none())?;
            nsd.rms_norm_mod(x.ptr(), Dt::F32, s, d, post_norm.ptr(), self.eps, none(), none(), none(), hb.ptr(), Dt::F16)?;
            let mut off = 0;
            for name in ["ffn_gate", "ffn_up"] {
                let w = self.matrix(&format!("{p}{name}.weight"))?;
                nsd.linear_q(hb.ptr(), Dt::F16, s, d, w.buf.ptr(), w.qtype, w.rows, none(), half_at(&gu, off), Dt::F16, 2 * f)?;
                off += w.rows;
                nsd.wait()?;
            }
            nsd.swiglu(gu.ptr(), Dt::F16, s, f, act.ptr(), Dt::F16)?;
            {
                let w = self.matrix(&format!("{p}ffn_down.weight"))?;
                nsd.linear_q(act.ptr(), Dt::F16, s, f, w.buf.ptr(), w.qtype, w.rows, none(), o.ptr(), Dt::F32, 0)?;
                nsd.wait()?;
            }
            nsd.gate_add(x.ptr(), Dt::F32, s, d, o.ptr(), Dt::F32, none(), none())?;
            // deepstack: the pictures' features of this depth added at their tokens
            for ((st, len), v) in runs.iter().zip(seen) {
                if let Some(ds) = v.deep.get(l) {
                    nsd.gate_add(x.fp().wrapping_add(st * d).cast(), Dt::F32, *len, d, ds.ptr(), Dt::F32, none(), none())?;
                }
            }
        }
        nsd.wait()?;
        Ok(x)
    }
}
