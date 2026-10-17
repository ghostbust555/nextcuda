//! Breeze TTS 2 (BreezeBlue, a CSM fork): English and Chinese speech at 24 kHz - in a voice designed from a
//! description, cloned from a recording and its exact transcript, or cloned and directed by an instruction (tone,
//! emotion, pace); vocal events inline in the text ("(laugh)", "[笑]").
//!
//! On nextsycl (audio/breezetts2) this is the T5Gemma2 text encoder, the Qwen3 backbone (28 layers of 2,048) guided by
//! a branch without the instruction, the depth decoder and the 12 Hz codec's decoder. Here only nextsycl's contract is
//! in place: the registry entry, the files by role, the options, what it can do (`speech`) - loading says it is not
//! ported yet.
//!
//! License: the weights and what they make are BreezeBlue's Research and Non-Commercial License 1.1; the inference
//! code upstream is Apache-2.0.

use std::path::Path;
use std::sync::Arc;

use nextcuda_audio::{At, AudioEngine, AudioKind, EngineOption, Error, LoadOptions, ModelFiles, Result, Speech};
use nextcuda_core::Gpu;

pub const ARCH: &str = "breeze-tts-2";
pub const ROLES: &[&str] = &["model", "tokenizer", "codec"];
pub const FPS: f32 = 12.5;
/// upstream's limit (infer.py): 1,500 frames (2 minutes)
pub const MAX_FRAMES: usize = 1500;
/// the guidance a request with an instruction gets when it names none (upstream's recommendation)
pub const CFG: f32 = 4.0;

pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "greedy", env: "NS_BZ2_GREEDY", value: "0|1", help: "always the most likely code (a deterministic read; flatter, may not end)",
                   at: At::Request },
    EngineOption { name: "temperature", env: "NS_BZ2_TEMPERATURE", value: "0.9", help: "the sampling temperature (backbone and depth decoder)", at: At::Request },
    EngineOption { name: "top-k", env: "NS_BZ2_TOP_K", value: "50", help: "the sampling top-k (backbone and depth decoder)", at: At::Request },
];

/// The vocal events the model knows inline in the text (upstream's examples): English in parentheses, Chinese in
/// square brackets
pub const EVENTS: &[&str] = &["(laugh)", "(sigh)", "(cough)", "(clears throat)", "[笑]", "[叹气]", "[咳嗽]", "[清嗓子]"];

pub fn kind() -> AudioKind {
    AudioKind { archs: &[ARCH], name: "Breeze TTS 2 (speech in English and Chinese: voices designed from a description, cloned from a recording, or cloned and directed; vocal events)",
                roles: ROLES, load, options: OPTIONS }
}

/// What it can do, as nextsycl's engine reports it: it clones only with the recording's exact transcript, and directs
/// a cloned voice by an instruction
pub fn speech() -> Speech {
    Speech {
        voices: Vec::new(),
        languages: vec!["chinese".into(), "english".into()],
        instructions: true,
        design: true,
        clone: true,
        clone_needs_text: true,
        clone_takes_text: true,
        direction: true,
        events: EVENTS.iter().map(|e| e.to_string()).collect(),
        requires: None,
    }
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
    Error(format!("{ARCH}: {what} is not ported to CUDA yet (Breeze TTS 2 runs on nextsycl; docs/PHASES.md)"))
}

/// A directory with the GGUF (q8_0 preferred), tokenizer.json and the codec's model.safetensors (as `models pull` lays
/// them out, or by hand): its files by role
pub fn files_in(dir: &Path) -> Result<ModelFiles> {
    let mut f = ModelFiles::new();
    let gguf = std::fs::read_dir(dir).map_err(|e| Error(format!("{}: {e}", dir.display())))?.flatten().map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "gguf") && p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("breeze-tts-2")))
        // the q8_0 file when there are several (the card's), else the first by name
        .min_by_key(|p| (!p.to_string_lossy().contains("q8_0"), p.clone())).ok_or_else(|| Error(format!("{}: no breeze-tts-2*.gguf", dir.display())))?;
    f.insert("model".into(), gguf);
    for (role, name) in [("tokenizer", "tokenizer.json"), ("codec", "audio_tokenizer/model.safetensors")] {
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
        assert!(speech().direction && speech().clone_needs_text && speech().events.len() == EVENTS.len());
    }
}
