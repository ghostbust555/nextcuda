//! VoxCPM2: tokenizer-free speech in 30 languages at 48 kHz. One model does it all: plain text (the model picks a
//! voice), a voice designed by a description in parentheses ahead of the text, a voice cloned from a recording (its
//! timbre; with the recording's transcript also its delivery, continued).
//!
//! On nextsycl (audio/voxcpm2) this is the base and residual MiniCPM4 language models, the local flow-matching DiT
//! and AudioVAE V2's decoder. Here only nextsycl's contract is in place: the registry entry, the files by role, the
//! options, what it can do (`speech`) - loading says it is not ported yet.

use std::path::Path;
use std::sync::Arc;

use nextcuda_audio::{At, AudioEngine, AudioKind, EngineOption, Error, LoadOptions, ModelFiles, Result, Speech};
use nextcuda_core::Gpu;

pub const ARCH: &str = "voxcpm2";
pub const ROLES: &[&str] = &["model", "config", "tokenizer", "vae"];
/// patches a second (25 latents a second, 4 a patch), and the most made
pub const PPS: f32 = 6.25;
pub const MAX_PATCHES: usize = 2000;

pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "int8", env: "NS_VCP_INT8", value: "0|1", help: "the language models' matrices in int8 (less VRAM, faster patches)", at: At::Load },
    EngineOption { name: "max-ratio", env: "NS_VCP_MAX_RATIO", value: "6", help: "the most patches a text token (a runaway's bound, as the reference's)",
                   at: At::Request },
];

pub fn kind() -> AudioKind {
    AudioKind { archs: &[ARCH], name: "VoxCPM2 2B (speech in 30 languages at 48 kHz: voices designed from a description, cloned from a recording)", roles: ROLES,
                load, options: OPTIONS }
}

/// What it can do, as nextsycl's engine reports it
pub fn speech() -> Speech {
    Speech { voices: Vec::new(), languages: Vec::new(), instructions: true, design: true, clone: true, clone_needs_text: false,
             clone_takes_text: true, direction: false, events: Vec::new(), requires: None }
}

/// Loading answers this until the engine is ported: the registry entry, its files and its options are nextsycl's, so a
/// model card, `models pull` and the front door already know the model; nothing runs it on CUDA yet.
fn load(files: &ModelFiles, _gpus: &[Arc<Gpu>], _o: &LoadOptions, _log: &mut dyn FnMut(String)) -> Result<Box<dyn AudioEngine>> {
    for r in ROLES {
        if !files.contains_key(*r) {
            return Err(Error(format!("{ARCH}: no {r} file")));
        }
    }
    Err(not_ported("loading the model"))
}

pub fn not_ported(what: &str) -> Error {
    Error(format!("{ARCH}: {what} is not ported to CUDA yet (VoxCPM2 runs on nextsycl; docs/PHASES.md)"))
}

/// A checkpoint's directory as downloaded: its files by role
pub fn files_in(dir: &Path) -> Result<ModelFiles> {
    let mut f = ModelFiles::new();
    for (role, name) in [("model", "model.safetensors"), ("config", "config.json"), ("tokenizer", "tokenizer.json"), ("vae", "audiovae.pth")] {
        let p = dir.join(name);
        if !p.exists() {
            return Err(Error(format!("{}: no {name}", dir.display())));
        }
        f.insert(role.into(), p);
    }
    Ok(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_entry_is_nextsycls_and_loading_says_it_is_not_ported() {
        let k = kind();
        assert_eq!(k.archs, &[ARCH]);
        assert_eq!(k.roles, ROLES);
        let mut files = ModelFiles::new();
        for r in ROLES {
            files.insert(r.to_string(), "/nonexistent".into());
        }
        let e = (k.load)(&files, &[], &LoadOptions::default(), &mut |_| {}).err().expect("not ported");
        assert!(e.0.contains("not ported to CUDA"), "{}", e.0);
        files.remove(ROLES[0]);
        let e = (k.load)(&files, &[], &LoadOptions::default(), &mut |_| {}).err().expect("a file missing");
        assert!(e.0.contains("no "), "{}", e.0);
        assert!(files_in(Path::new("/nonexistent")).is_err());
        assert!(speech().design && speech().clone && !speech().direction);
    }
}
