//! Qwen3-TTS (12 Hz, 1.7B): speech from text in ten languages - in one of nine built-in voices styled by an
//! instruction (CustomVoice), in a voice made up from a description (VoiceDesign), or in the voice of a recording
//! (Base: its x-vector, and with its transcript its codec frames as an example to continue).
//!
//! On nextsycl (audio/qwen3tts) this is the talker (a 28-layer Qwen3 of 2,048) drawing a frame's first code, its code
//! predictor the other fifteen, and the 12 Hz codec's decoder. Here only nextsycl's contract is in place: the
//! registry entry, the files by role, the options, what it can do (`speech`) - loading says it is not ported yet.

use std::path::Path;
use std::sync::Arc;

use nextcuda_audio::{At, AudioEngine, AudioKind, EngineOption, Error, LoadOptions, ModelFiles, Result, Speech};
use nextcuda_core::Gpu;

pub const ARCH: &str = "qwen3-tts";
pub const ROLES: &[&str] = &["model", "config", "vocab", "merges", "tokenizer-config", "codec"];
/// frames a second, and the longest speech made (the checkpoint's max_new_tokens)
pub const FPS: f32 = 12.5;
pub const MAX_FRAMES: usize = 8192;

pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "int8", env: "NS_Q3T_INT8", value: "0|1", help: "the talker's and code predictor's matrices in int8 (less VRAM, faster frames)", at: At::Load },
    EngineOption { name: "greedy", env: "NS_Q3T_GREEDY", value: "0|1", help: "always the most likely code, no repetition penalty (a deterministic read; flatter)",
                   at: At::Request },
    EngineOption { name: "temperature", env: "NS_Q3T_TEMPERATURE", value: "0.9", help: "the talker's sampling temperature", at: At::Request },
    EngineOption { name: "top-k", env: "NS_Q3T_TOP_K", value: "50", help: "the talker's top-k", at: At::Request },
    EngineOption { name: "streaming", env: "NS_Q3T_STREAMING", value: "0|1",
                   help: "the text fed a token a frame (a clone's default) rather than all ahead (the built-in and designed voices' default)", at: At::Request },
];

pub fn kind() -> AudioKind {
    AudioKind { archs: &[ARCH], name: "Qwen3-TTS 12 Hz 1.7B (speech: built-in voices, voices designed from a description, voices cloned from a recording)",
                roles: ROLES, load, options: OPTIONS }
}

/// What a checkpoint of this kind ("custom_voice", "voice_design", "base": its config's tts_model_type) can do, as
/// nextsycl's engine reports it once loaded (its voices and languages come from the config)
pub fn speech(kind: &str, voices: Vec<String>, languages: Vec<String>) -> Speech {
    Speech {
        voices,
        languages,
        instructions: kind != "base",
        design: kind == "voice_design",
        clone: kind == "base",
        clone_needs_text: false,
        clone_takes_text: kind == "base",
        direction: false,
        events: Vec::new(),
        requires: Some(match kind {
            "custom_voice" => "voice",
            "voice_design" => "instructions",
            _ => "reference",
        }),
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
    Error(format!("{ARCH}: {what} is not ported to CUDA yet (Qwen3-TTS runs on nextsycl; docs/PHASES.md)"))
}

/// A checkpoint's directory as downloaded: its files by role (the speech tokenizer in `speech_tokenizer/`)
pub fn files_in(dir: &Path) -> Result<ModelFiles> {
    let mut f = ModelFiles::new();
    for (role, name) in [("model", "model.safetensors"), ("config", "config.json"), ("vocab", "vocab.json"), ("merges", "merges.txt"),
                         ("tokenizer-config", "tokenizer_config.json"), ("codec", "speech_tokenizer/model.safetensors")] {
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
        assert_eq!(speech("custom_voice", vec![], vec![]).requires, Some("voice"));
        assert!(speech("base", vec![], vec![]).clone && !speech("base", vec![], vec![]).instructions);
    }
}
