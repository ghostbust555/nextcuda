//! The engine's kernels (`kernels/llm/nemotronh/nh.h`), bound from the llm kernel library by name.

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

type G = *mut c_void;
type P = *const c_void;
type M = *mut c_void;
type F = *const f32;
type FM = *mut f32;

pub struct Api {
    pub conv: unsafe extern "C" fn(G, F, i64, FM, F, F, FM, i64, i64, c_int) -> c_int,
    pub ssm: unsafe extern "C" fn(G, F, i64, F, i64, F, F, F, FM, FM, i64, i64, i64, i64, i64) -> c_int,
    pub gated_norm: unsafe extern "C" fn(G, F, F, i64, F, FM, i64, i64, i64, f32) -> c_int,
    pub relu2: unsafe extern "C" fn(G, FM, i64) -> c_int,
    pub kv_store: unsafe extern "C" fn(G, F, i64, F, i64, i64, i64, i64, i64, i64, M, M) -> c_int,
    pub attn_max_keys: unsafe extern "C" fn() -> i64,
    pub attn: unsafe extern "C" fn(G, F, i64, P, P, i64, i64, i64, i64, i64, i64, FM) -> c_int,
    pub rms_norm: unsafe extern "C" fn(G, F, F, FM, i64, i64, f32) -> c_int,
    pub add: unsafe extern "C" fn(G, FM, F, i64) -> c_int,
    pub gemv_f32: unsafe extern "C" fn(G, F, F, FM, i64, i64, i64) -> c_int,
    pub moe_combine: unsafe extern "C" fn(G, FM, F, *const i32, F, i64, i64) -> c_int,
    pub row_bytes: unsafe extern "C" fn(c_int, i64) -> usize,
    pub mmvq_supported: unsafe extern "C" fn(c_int) -> c_int,
    pub dequant_f32: unsafe extern "C" fn(G, c_int, P, i64, FM) -> c_int,
    pub q8_1_bytes: unsafe extern "C" fn(i64, i64) -> usize,
    pub quantize: unsafe extern "C" fn(G, F, i64, i64, M) -> c_int,
    pub mmvq: unsafe extern "C" fn(G, c_int, P, P, FM, i64, i64, i64) -> c_int,
    pub embed: unsafe extern "C" fn(G, c_int, P, i64, *const i32, i64, i64, FM) -> c_int,
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
                // SAFETY: the symbol is declared in nh.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            conv: sym!("ns_nh_conv"),
            ssm: sym!("ns_nh_ssm"),
            gated_norm: sym!("ns_nh_gated_norm"),
            relu2: sym!("ns_nh_relu2"),
            kv_store: sym!("ns_nh_kv_store"),
            attn_max_keys: sym!("ns_nh_attn_max_keys"),
            attn: sym!("ns_nh_attn"),
            rms_norm: sym!("ns_nh_rms_norm"),
            add: sym!("ns_nh_add"),
            gemv_f32: sym!("ns_nh_gemv_f32"),
            moe_combine: sym!("ns_nh_moe_combine"),
            row_bytes: sym!("ns_nh_row_bytes"),
            mmvq_supported: sym!("ns_nh_mmvq_supported"),
            dequant_f32: sym!("ns_nh_dequant_f32"),
            q8_1_bytes: sym!("ns_nh_q8_1_bytes"),
            quantize: sym!("ns_nh_quantize"),
            mmvq: sym!("ns_nh_mmvq"),
            embed: sym!("ns_nh_embed"),
        })
    })
    .as_ref()
    .map_err(|e| nextcuda_core::Error(e.clone()))
}

/// A kernel's status as an error naming what it was doing
pub fn check(r: c_int, what: &str) -> nextcuda_core::Result<()> {
    if r == 0 {
        return Ok(());
    }
    let msg = nextcuda_core::api().map(|a| a.error()).unwrap_or_default();
    Err(nextcuda_core::Error(format!("{what}: {msg}")))
}
