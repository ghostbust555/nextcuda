//! The engine behind the runtime's `Engine` contract (`nextcuda-llm`): sessions, the decode step, and checkpoints of a
//! conversation's whole state (the attention layers' key / value rows, each Mamba layer's conv inputs and scan state) for
//! the server's prompt cache, after nextsycl's nemotronh engine (the same checkpoint layout, `NSNMH`). `kind()` is its
//! registry entry. No drafts here (the MTP block is not loaded), so a verify pass is one row and `rollback` keeps all.

use std::any::Any;
use std::sync::{Arc, Mutex};

use nextcuda_core::{At, EngineOption, Error, Gpu, Result};
use nextcuda_gguf::Gguf;
use nextcuda_llm::{Checkpoint, CheckpointState, Decoder, DecoderState, Engine, EngineKind, GpuInfo, LoadOptions, Sampler, Session, SessionState, Tap};

use crate::{Kind, LState, NemotronH, Options};

pub const ARCH: &str = "nemotron_h_moe";

pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "chunk", env: "NS_NH_CHUNK", value: "TOKENS", help: "a long prompt pass's rows (default 2048, in VRAM borrowed from the hot set)", at: At::Load },
    EngineOption { name: "pinned-gib", env: "NS_NH_PINNED_GIB", value: "GIB", help: "pinned host memory for the expert cache (default: the server's mirror share)", at: At::Load },
    EngineOption { name: "reserve-mib", env: "NS_NH_RESERVE_MIB", value: "MIB", help: "VRAM kept free beside the weights (default 700)", at: At::Load },
];

pub fn kind() -> EngineKind {
    EngineKind { archs: &[ARCH], name: "Nemotron-H MoE (Mamba-2 + latent MoE + attention: Nemotron 3.5 Super 120B-A12B), one card, experts streamed", load, info,
                 kernels: None, chat: nextcuda_tok::nemotron_chat, options: OPTIONS, prepare: None }
}

fn e(x: impl std::fmt::Display) -> Error {
    Error(x.to_string())
}

fn info(f: &Gguf) -> std::result::Result<String, String> {
    let g = crate::Geometry::read(f)?;
    Ok(format!("{ARCH}: {} layers of {} ({} Mamba-2: {} heads of {} in {} groups, state {}, conv {}; {} latent MoE: {} of {} experts of {} in a latent of {}, \
                a shared expert of {}; {} attention: {} / {} heads of {}, no positions), vocabulary {}", g.layers, g.n_embd, g.count(Kind::Mamba), g.ssm_heads,
               g.inner / g.ssm_heads, g.groups, g.state, g.conv, g.count(Kind::Moe), g.used, g.experts, g.n_ff_exp, g.latent, g.n_ff_shared, g.count(Kind::Attn),
               g.heads, g.heads_kv, g.head, g.vocab))
}

/// The loaded model as the server drives it: one pass at a time, the expert profile beside the model file
pub struct Served {
    m: NemotronH,
    work: Mutex<()>,
    profile: std::path::PathBuf,
}

fn load<'g>(f: &'g Gguf, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn Engine + 'g>> {
    let path = f.paths.first().ok_or_else(|| e("a model without files"))?.clone();
    let mut opt = Options::default();
    opt.gpu = gpus.first().map_or(opt.gpu, |g| g.index);
    if std::env::var("NS_NH_PINNED_GIB").is_err() {
        if let Some(b) = o.mirror_bytes {
            opt.pinned_gib = (b as f64 / (1u64 << 30) as f64).max(4.0);
        }
    }
    let profile = std::env::var("NS_EXPERT_PROFILE").map(std::path::PathBuf::from).unwrap_or_else(|_| path.with_file_name("expert-profile.txt"));
    opt.profile = Some(profile.clone());
    opt.ctx = o.kv.0.max(256);
    let m = NemotronH::load(&path, &opt, log)?;
    m.preload(Some(&profile), log)?;
    Ok(Box::new(Served { m, work: Mutex::new(()), profile }))
}

impl SessionState for crate::Session {
    fn pos(&self) -> usize {
        self.pos
    }
    fn max_ctx(&self) -> usize {
        self.max_ctx
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Generation from a fed prompt: the logits to draw the next token from, or the committed token not fed yet
pub struct NhDecoder {
    logits: Option<Vec<f32>>,
    next: Option<u32>,
}

impl DecoderState for NhDecoder {
    fn drafts(&self) -> (u64, u64) {
        (0, 0)
    }
    fn pending(&mut self) -> Option<u32> {
        self.next.take()
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// A conversation's state in host memory (nextsycl's layout: attention rows per kv head up to `pos`, Mamba buffers whole)
pub struct NhCheckpoint {
    pos: usize,
    bufs: Vec<Vec<u8>>,
    bytes: usize,
}

impl CheckpointState for NhCheckpoint {
    fn pos(&self) -> usize {
        self.pos
    }
    fn bytes(&self) -> usize {
        self.bytes
    }
    fn write_to(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        let u = |w: &mut dyn std::io::Write, x: usize| w.write_all(&(x as u64).to_le_bytes());
        w.write_all(b"NSNMH")?;
        u(w, self.pos)?;
        u(w, self.bufs.len())?;
        for b in &self.bufs {
            u(w, b.len())?;
            w.write_all(b)?;
        }
        Ok(())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn read_checkpoint(r: &mut dyn std::io::Read) -> std::io::Result<NhCheckpoint> {
    let u = |r: &mut dyn std::io::Read| -> std::io::Result<usize> {
        let mut b = [0u8; 8];
        r.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b) as usize)
    };
    let mut magic = [0u8; 5];
    r.read_exact(&mut magic)?;
    if &magic != b"NSNMH" {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not a nemotronh checkpoint"));
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
    Ok(NhCheckpoint { pos, bufs, bytes })
}

fn sess(s: &Session) -> Result<&crate::Session> {
    s.get().ok_or_else(|| e("a session of another engine"))
}

fn sess_mut(s: &mut Session) -> Result<&mut crate::Session> {
    s.get_mut().ok_or_else(|| e("a session of another engine"))
}

impl NemotronH {
    pub fn reset_session(&self, s: &mut crate::Session) -> Result<()> {
        for st in &s.layers {
            if let LState::Mamba { conv, s: ss } = st {
                conv.fill(0)?;
                ss.fill(0)?;
            }
        }
        s.pos = 0;
        Ok(())
    }

    pub fn copy_session(&self, dst: &mut crate::Session, src: &crate::Session) -> Result<()> {
        if src.pos > dst.max_ctx {
            return Err(e(format!("a session of {} tokens into one of {}", src.pos, dst.max_ctx)));
        }
        let g = &self.g;
        for (d, s) in dst.layers.iter().zip(&src.layers) {
            match (d, s) {
                (LState::Attn { k: dk, v: dv }, LState::Attn { k: sk, v: sv }) => {
                    for h in 0..g.heads_kv {
                        let n = src.pos * g.head * 2;
                        dk.copy_within(h * dst.max_ctx * g.head * 2, sk, h * src.max_ctx * g.head * 2, n)?;
                        dv.copy_within(h * dst.max_ctx * g.head * 2, sv, h * src.max_ctx * g.head * 2, n)?;
                    }
                }
                (LState::Mamba { conv: dc, s: ds }, LState::Mamba { conv: sc, s: ss }) => {
                    dc.copy_within(0, sc, 0, sc.len)?;
                    ds.copy_within(0, ss, 0, ss.len)?;
                }
                (LState::None, LState::None) => {}
                _ => return Err(e("sessions of other shapes")),
            }
        }
        dst.pos = src.pos;
        Ok(())
    }

    pub fn save(&self, s: &crate::Session) -> Result<NhCheckpoint> {
        let g = &self.g;
        self.gpu.sync()?;
        let mut bufs = Vec::new();
        for st in &s.layers {
            match st {
                LState::Attn { k, v } => {
                    for c in [k, v] {
                        let n = s.pos * g.head * 2;
                        let mut b = vec![0u8; g.heads_kv * n];
                        for h in 0..g.heads_kv {
                            c.read(h * s.max_ctx * g.head * 2, &mut b[h * n..(h + 1) * n])?;
                        }
                        bufs.push(b);
                    }
                }
                LState::Mamba { conv, s: ss } => {
                    for c in [conv, ss] {
                        let mut b = vec![0u8; c.len];
                        c.read(0, &mut b)?;
                        bufs.push(b);
                    }
                }
                LState::None => {}
            }
        }
        let bytes = bufs.iter().map(|b| b.len()).sum();
        Ok(NhCheckpoint { pos: s.pos, bufs, bytes })
    }

    pub fn restore(&self, s: &mut crate::Session, ck: &NhCheckpoint) -> Result<()> {
        let g = &self.g;
        if ck.pos > s.max_ctx {
            return Err(e(format!("a checkpoint of {} tokens into a session of {}", ck.pos, s.max_ctx)));
        }
        let mut it = ck.bufs.iter();
        for st in &s.layers {
            match st {
                LState::Attn { k, v } => {
                    for c in [k, v] {
                        let b = it.next().ok_or_else(|| e("a short checkpoint"))?;
                        let n = ck.pos * g.head * 2;
                        if b.len() != g.heads_kv * n {
                            return Err(e("a checkpoint of another shape"));
                        }
                        for h in 0..g.heads_kv {
                            c.write(h * s.max_ctx * g.head * 2, &b[h * n..(h + 1) * n])?;
                        }
                    }
                }
                LState::Mamba { conv, s: ss } => {
                    for c in [conv, ss] {
                        let b = it.next().ok_or_else(|| e("a short checkpoint"))?;
                        if b.len() != c.len {
                            return Err(e("a checkpoint of another shape"));
                        }
                        c.write(0, b)?;
                    }
                }
                LState::None => {}
            }
        }
        if it.next().is_some() {
            return Err(e("a checkpoint with more state than these sessions (a draft block's?)"));
        }
        s.pos = ck.pos;
        Ok(())
    }
}

impl Served {
    fn feed_until(&self, s: &mut crate::Session, tokens: &[u32], stop: &(dyn Fn(usize) -> bool + Sync)) -> Result<(usize, Vec<f32>)> {
        let _w = self.work.lock().unwrap();
        let (done, mut out) = self.m.feed(s, tokens, 1, &|n| stop(n))?;
        Ok((done, out.pop().unwrap_or_default()))
    }
}

impl Engine for Served {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn load_seconds(&self) -> f64 {
        self.m.load_s
    }
    fn has_draft(&self) -> bool {
        false
    }
    fn prefill_chunk(&self) -> usize {
        self.m.chunk
    }
    fn max_verify(&self) -> usize {
        1
    }
    fn cache_fingerprint(&self) -> String {
        "nemotronh f16-kv f32-ssm nsnmh-1".into()
    }
    fn session(&self, max_ctx: usize) -> Result<Session> {
        Ok(Session::new(self.m.session(max_ctx)?))
    }
    fn reset_session(&self, s: &mut Session) -> Result<()> {
        self.m.reset_session(sess_mut(s)?)
    }
    fn copy_session(&self, dst: &mut Session, src: &Session) -> Result<()> {
        let _w = self.work.lock().unwrap();
        let src = sess(src)?;
        self.m.copy_session(sess_mut(dst)?, src)
    }
    fn feed(&self, s: &mut Session, tokens: &[u32], _tap: Tap) -> Result<Vec<f32>> {
        Ok(self.feed_until(sess_mut(s)?, tokens, &|_| false)?.1)
    }
    fn feed_until(&self, s: &mut Session, tokens: &[u32], stop: &(dyn Fn(usize) -> bool + Sync), _tap: Tap) -> Result<(usize, Vec<f32>)> {
        Served::feed_until(self, sess_mut(s)?, tokens, stop)
    }
    fn forward(&self, s: &mut Session, tokens: &[u32], _tap: Tap) -> Result<Vec<f32>> {
        let _w = self.work.lock().unwrap();
        Ok(self.m.forward(sess_mut(s)?, tokens, 1)?.pop().unwrap_or_default())
    }
    fn forward_rows(&self, s: &mut Session, tokens: &[u32], n_out: usize, _tap: Tap) -> Result<Vec<Vec<f32>>> {
        let _w = self.work.lock().unwrap();
        self.m.forward(sess_mut(s)?, tokens, n_out)
    }
    fn rollback(&self, s: &mut Session, keep: usize) -> Result<()> {
        // the Mamba states cannot step back: only a pass's every row kept (no drafts, so a verify pass is one row)
        let _ = (s, keep);
        if keep == 0 {
            return Err(e("nemotronh cannot roll a pass back (its Mamba states have no snapshots here)"));
        }
        Ok(())
    }
    fn forward_batch(&self, sessions: &mut [&mut Session], tokens: &[u32], _tap: Tap) -> Result<Vec<Vec<f32>>> {
        let _w = self.work.lock().unwrap();
        let mut out = Vec::with_capacity(tokens.len());
        for (s, t) in sessions.iter_mut().zip(tokens) {
            out.push(self.m.pass(sess_mut(s)?, &[*t], 1)?.pop().unwrap_or_default());
        }
        Ok(out)
    }
    fn decoder(&self, logits: Vec<f32>, _draft: bool) -> Decoder {
        Decoder::new(NhDecoder { logits: Some(logits), next: None })
    }
    fn decoder_after(&self, next: u32, _draft: bool) -> Decoder {
        Decoder::new(NhDecoder { logits: None, next: Some(next) })
    }
    fn step(&self, s: &mut Session, d: &mut Decoder, smp: &mut dyn Sampler, _tap: Tap) -> Result<Vec<u32>> {
        let d = d.get_mut::<NhDecoder>().ok_or_else(|| e("a decoder of another engine"))?;
        let y = match (d.logits.take(), d.next.take()) {
            (Some(l), _) => smp.sample(&l),
            (None, Some(x)) => {
                let _w = self.work.lock().unwrap();
                let l = self.m.pass(sess_mut(s)?, &[x], 1)?.pop().unwrap_or_default();
                smp.sample(&l)
            }
            (None, None) => return Err(e("a decoder with nothing to draw from")),
        };
        d.next = Some(y);
        Ok(vec![y])
    }
    fn save(&self, s: &Session) -> Result<Checkpoint> {
        let _w = self.work.lock().unwrap();
        Ok(Checkpoint::new(self.m.save(sess(s)?)?))
    }
    fn restore(&self, s: &mut Session, ck: &Checkpoint) -> Result<()> {
        let _w = self.work.lock().unwrap();
        let ck = ck.get::<NhCheckpoint>().ok_or_else(|| e("a checkpoint of another engine"))?;
        self.m.restore(sess_mut(s)?, ck)
    }
    fn read_checkpoint(&self, r: &mut dyn std::io::Read) -> std::io::Result<Checkpoint> {
        Ok(Checkpoint::new(read_checkpoint(r)?))
    }
    fn gpu_info(&self) -> Vec<GpuInfo> {
        let g = self.m.gpu();
        let (total, free) = g.memory().unwrap_or((0, None));
        vec![GpuInfo { index: g.index, name: g.name.clone(), pci: None, total, free, layers: (0, self.m.g.layers as u64), expert_slots: 0, host_slots: 0 }]
    }
    fn save_profile(&self, path: &std::path::Path) -> std::io::Result<usize> {
        self.m.save_profile(path)?;
        Ok(1)
    }
    fn report(&self, tokens: usize) -> Vec<String> {
        let (h, m) = self.m.cache_stats();
        vec![format!("nemotronh: {tokens} tokens; expert cache hits {h}, misses {m}; profile {}", self.profile.display())]
    }
}
