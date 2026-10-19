//! The kernels this engine runs, bound from the llm kernel library by name: its own (`kernels/llm/qwen35/q35.h`: the
//! attention, the DeltaNet pieces, and glm5next's generic conv + SiLU, L2 norm, delta-rule scan and SwiGLU under an
//! ns_q35_ prefix) and the nemotronh engine's (`kernels/llm/nemotronh/nh.h`: the products from the stored blocks, the
//! dequantizers, the RMS norm, add, the half GEMM).

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

type G = *mut c_void;
type P = *const c_void;
type M = *mut c_void;
type F = *const f32;
type FM = *mut f32;

#[allow(clippy::type_complexity)]
pub struct Api {
    // nh.h
    pub dequant_f32: unsafe extern "C" fn(G, c_int, P, i64, FM) -> c_int,
    pub dequant_f16: unsafe extern "C" fn(G, c_int, P, i64, M) -> c_int,
    pub gemv_f32: unsafe extern "C" fn(G, F, F, FM, i64, i64, i64) -> c_int,
    pub rms_norm: unsafe extern "C" fn(G, F, F, FM, i64, i64, f32) -> c_int,
    pub add: unsafe extern "C" fn(G, FM, F, i64) -> c_int,
    pub to_half: unsafe extern "C" fn(G, F, M, i64) -> c_int,
    pub gemm_f16: unsafe extern "C" fn(G, P, P, FM, i64, i64, i64) -> c_int,
    pub mmvq_supported: unsafe extern "C" fn(c_int) -> c_int,
    pub q8_1_bytes: unsafe extern "C" fn(i64, i64) -> usize,
    pub quantize: unsafe extern "C" fn(G, F, i64, i64, M) -> c_int,
    pub mmvq: unsafe extern "C" fn(G, c_int, P, P, FM, i64, i64, i64) -> c_int,
    // q35.h: glm.h's generic ones
    pub conv_silu: unsafe extern "C" fn(G, F, FM, F, FM, i64, i64, c_int, FM) -> c_int,
    pub l2_norm: unsafe extern "C" fn(G, FM, i64, i64, f32) -> c_int,
    pub kda_scan: unsafe extern "C" fn(G, F, F, F, F, F, FM, FM, i64, i64, i64, FM) -> c_int,
    pub swiglu: unsafe extern "C" fn(G, F, F, FM, i64, f32) -> c_int,
    // q35.h: its own
    pub qk_norm_rope: unsafe extern "C" fn(G, F, i64, i64, FM, i64, i64, i64, i64, F, f32, i64, f32, i64) -> c_int,
    pub mmq_supported: unsafe extern "C" fn(c_int) -> c_int,
    pub mmq_reserve: unsafe extern "C" fn(G) -> c_int,
    pub mmq_layout: unsafe extern "C" fn(c_int) -> c_int,
    /// ggml-cuda's MMVQ (nh's mmvq's arguments)
    pub q35_mmvq_supported: unsafe extern "C" fn(c_int) -> c_int,
    pub q35_mmvq: unsafe extern "C" fn(G, c_int, P, P, FM, i64, i64, i64) -> c_int,
    pub mmq_q8_bytes: unsafe extern "C" fn(i64, i64) -> usize,
    pub mmq_quantize: unsafe extern "C" fn(G, c_int, F, i64, i64, M) -> c_int,
    pub mmq: unsafe extern "C" fn(G, c_int, P, i64, i64, P, i64, FM) -> c_int,
    pub qk_norm_mrope: unsafe extern "C" fn(G, F, i64, i64, FM, i64, i64, i64, i64, F, f32, i64, f32, P) -> c_int,
    pub kv_store: unsafe extern "C" fn(G, F, i64, F, i64, i64, i64, i64, i64, i64, M, M) -> c_int,
    /// the q8_0 cache's (rows of 256 int8 and 8 half scales)
    pub kv_store_q8: unsafe extern "C" fn(G, F, i64, F, i64, i64, i64, i64, i64, i64, M, M) -> c_int,
    pub kv_store_q4: unsafe extern "C" fn(G, F, i64, F, i64, i64, i64, i64, i64, i64, M, M) -> c_int,
    /// the 32-block Hadamard transform in place (a rotated cache)
    pub fwht32: unsafe extern "C" fn(G, FM, i64, i64, i64) -> c_int,
    pub attn_scratch: unsafe extern "C" fn(i64, i64, i64, i64, i64) -> i64,
    pub attn: unsafe extern "C" fn(G, F, i64, P, P, i64, i64, i64, i64, i64, i64, FM, FM) -> c_int,
    pub attn_q8: unsafe extern "C" fn(G, F, i64, P, P, i64, i64, i64, i64, i64, i64, FM, FM) -> c_int,
    pub attn_q4: unsafe extern "C" fn(G, F, i64, P, P, i64, i64, i64, i64, i64, i64, FM, FM) -> c_int,
    pub gate_mul: unsafe extern "C" fn(G, FM, F, i64, i64, i64, i64) -> c_int,
    pub gdn_gates: unsafe extern "C" fn(G, F, FM, F, F, FM, i64, i64, i64) -> c_int,
    pub expand: unsafe extern "C" fn(G, F, i64, FM, i64, i64, i64, i64) -> c_int,
    pub gdn_out: unsafe extern "C" fn(G, F, F, i64, F, FM, i64, i64, i64, f32) -> c_int,
    /// the greedy pick on the GPU (the MTP drafts)
    pub argmax: unsafe extern "C" fn(G, F, i64, i64, *mut i32) -> c_int,
}

// SAFETY: function pointers into the library, which stays loaded for the process.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

pub fn api() -> nextcuda_core::Result<&'static Api> {
    static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| {
        let lib = nextcuda_core::api().map_err(|e| e.0)?;
        macro_rules! sym {
            ($name:literal) => {{
                let p = lib.symbol($name);
                if p.is_null() {
                    return Err(format!("{} is not in the kernel library (the llm library, rebuilt: kernels/build.sh)", $name));
                }
                // SAFETY: the symbol is declared in nh.h / q35.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            dequant_f32: sym!("ns_nh_dequant_f32"),
            dequant_f16: sym!("ns_nh_dequant_f16"),
            gemv_f32: sym!("ns_nh_gemv_f32"),
            rms_norm: sym!("ns_nh_rms_norm"),
            add: sym!("ns_nh_add"),
            to_half: sym!("ns_nh_to_half"),
            gemm_f16: sym!("ns_nh_gemm_f16"),
            mmvq_supported: sym!("ns_nh_mmvq_supported"),
            q8_1_bytes: sym!("ns_nh_q8_1_bytes"),
            quantize: sym!("ns_nh_quantize"),
            mmvq: sym!("ns_nh_mmvq"),
            conv_silu: sym!("ns_q35_conv_silu"),
            l2_norm: sym!("ns_q35_l2_norm"),
            kda_scan: sym!("ns_q35_kda_scan"),
            swiglu: sym!("ns_q35_swiglu"),
            qk_norm_rope: sym!("ns_q35_qk_norm_rope"),
            qk_norm_mrope: sym!("ns_q35_qk_norm_mrope"),
            mmq_supported: sym!("ns_q35_mmq_supported"),
            mmq_reserve: sym!("ns_q35_mmq_reserve"),
            mmq_layout: sym!("ns_q35_mmq_layout"),
            q35_mmvq_supported: sym!("ns_q35_mmvq_supported"),
            q35_mmvq: sym!("ns_q35_mmvq"),
            mmq_q8_bytes: sym!("ns_q35_mmq_q8_bytes"),
            mmq_quantize: sym!("ns_q35_mmq_quantize"),
            mmq: sym!("ns_q35_mmq"),
            kv_store: sym!("ns_q35_kv_store"),
            kv_store_q8: sym!("ns_q35_kv_store_q8"),
            kv_store_q4: sym!("ns_q35_kv_store_q4"),
            fwht32: sym!("ns_q35_fwht32"),
            attn_scratch: sym!("ns_q35_attn_scratch"),
            attn: sym!("ns_q35_attn"),
            attn_q8: sym!("ns_q35_attn_q8"),
            attn_q4: sym!("ns_q35_attn_q4"),
            gate_mul: sym!("ns_q35_gate_mul"),
            gdn_gates: sym!("ns_q35_gdn_gates"),
            expand: sym!("ns_q35_expand"),
            gdn_out: sym!("ns_q35_gdn_out"),
            argmax: sym!("ns_q35_argmax"),
        })
    })
    .as_ref()
    .map_err(|e| nextcuda_core::Error(e.clone()))
}

/// A call's return code as a result (the reason from the library)
pub fn check(rc: c_int, what: &str) -> nextcuda_core::Result<()> {
    if rc == 0 {
        return Ok(());
    }
    let why = nextcuda_core::api().map(|a| a.error()).unwrap_or_default();
    Err(nextcuda_core::Error(format!("{what}: {why}")))
}
