//! nextcuda-qwen3vl: Qwen3-VL's language model as a text encoder - what Qwen-Image 2.1 conditions on. Read from a
//! llama.cpp GGUF (`gguf`), its matrices in their stored types on the card, made half a slice at a time.
//!
//! nextsycl's version of this crate read ComfyUI's int8 ConvRot file and had the vision tower for edits; those are
//! not here yet (an edit runs on the stable-diffusion.cpp engine).

pub mod gguf;

pub use gguf::GgufTextEncoder;
