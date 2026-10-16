//! The in-process engine: stable-diffusion.cpp (ggml-cuda) through the bridge in the image kernel library
//! (`kernels/image/sd/sd.h`). The model stays loaded between pictures - the phase 1 worker paid its load (about half
//! of a 768 x 768 picture's time) every time - and a cancel stops the step in progress.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::OnceLock;

use nextcuda_core::{Error, Result};

type Progress = unsafe extern "C" fn(c_int, c_int, f32, *mut c_void);
type Log = unsafe extern "C" fn(c_int, *const c_char, *mut c_void);

struct Api {
    last_error: unsafe extern "C" fn() -> *const c_char,
    create: unsafe extern "C" fn(*const c_char, *const c_char, *const c_char, *const c_char, *const c_char, f32, c_int, *const c_char, *mut *mut c_void) -> c_int,
    destroy: unsafe extern "C" fn(*mut c_void),
    #[allow(clippy::type_complexity)]
    generate: unsafe extern "C" fn(
        *mut c_void, *const c_char, *const c_char, c_int, c_int, c_int, f32, i64, *const c_char, *const c_char, f32, c_int, f32,
        *const u8, c_int, c_int, c_int, *const *const u8, *const c_int, *const c_int, *const c_int, c_int, *const c_char, f32,
        *mut *mut u8, *mut c_int, *mut c_int, *mut c_int,
    ) -> c_int,
    free_image: unsafe extern "C" fn(*mut u8),
    cancel: unsafe extern "C" fn(*mut c_void),
    set_progress: unsafe extern "C" fn(Option<Progress>, *mut c_void),
    set_log: unsafe extern "C" fn(Option<Log>, *mut c_void),
}

// SAFETY: function pointers into a library that stays loaded for the process.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

/// The bridge, when the image kernel library has it
fn api() -> Result<&'static Api> {
    static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| {
        let lib = nextcuda_core::api().map_err(|e| e.0)?;
        macro_rules! sym {
            ($name:literal) => {{
                let p = lib.symbol($name);
                if p.is_null() {
                    return Err(format!("{} is not in the image kernel library (built without stable-diffusion.cpp)", $name));
                }
                // SAFETY: declared in sd.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            last_error: sym!("ns_sd_last_error"),
            create: sym!("ns_sd_create"),
            destroy: sym!("ns_sd_destroy"),
            generate: sym!("ns_sd_generate"),
            free_image: sym!("ns_sd_free_image"),
            cancel: sym!("ns_sd_cancel"),
            set_progress: sym!("ns_sd_set_progress"),
            set_log: sym!("ns_sd_set_log"),
        })
    })
    .as_ref()
    .map_err(|e| Error(e.clone()))
}

/// Whether the bridge is there (else the python worker is the engine)
pub fn available() -> bool {
    api().is_ok()
}

fn last(k: &Api) -> String {
    // SAFETY: a C string owned by the library, valid until the next call on this thread.
    unsafe { CStr::from_ptr((k.last_error)()) }.to_string_lossy().into_owned()
}

fn cstr(s: &str) -> Result<CString> {
    CString::new(s).map_err(|_| Error(format!("{s:?} holds a NUL")))
}

/// One loaded model
pub struct Sd {
    k: &'static Api,
    ctx: *mut c_void,
}

// SAFETY: the context is used by one thread at a time (the engine holds it in a mutex).
unsafe impl Send for Sd {}

impl Drop for Sd {
    fn drop(&mut self) {
        // SAFETY: made by ns_sd_create, destroyed once.
        unsafe { (self.k.destroy)(self.ctx) };
    }
}

/// A picture as the bridge takes and gives them: 8-bit interleaved
pub struct Raw<'a> {
    pub data: &'a [u8],
    pub w: usize,
    pub h: usize,
    pub c: usize,
}

pub struct Out {
    pub data: Vec<u8>,
    pub w: usize,
    pub h: usize,
    pub c: usize,
}

/// What one picture takes
pub struct Gen<'a> {
    pub prompt: &'a str,
    pub negative: &'a str,
    pub width: usize,
    pub height: usize,
    pub steps: u32,
    pub cfg: f32,
    pub seed: u64,
    pub sampler: &'a str,
    pub scheduler: &'a str,
    pub flow_shift: f32,
    pub rgba: bool,
    pub strength: f32,
    pub init: Option<Raw<'a>>,
    pub refs: Vec<Raw<'a>>,
    pub lora: Option<(&'a str, f32)>,
}

/// The library's log, kept (last 4,000 lines) so a bench can read the stage times it prints
static LOG: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// The log lines since the last call
pub fn take_log() -> Vec<String> {
    std::mem::take(&mut *LOG.lock().unwrap())
}

/// The stage times the library's log gives for one picture: (text encode, sampling, VAE decode), seconds
pub fn stage_times(lines: &[String]) -> (Option<f64>, Option<f64>, Option<f64>) {
    let taking = |needle: &str| -> Option<f64> {
        lines.iter().rev().find(|l| l.contains(needle)).and_then(|l| {
            let at = l.rfind("taking ")? + 7;
            l[at..].trim().trim_end_matches('s').parse::<f64>().ok()
        })
    };
    (taking("get_learned_condition completed"), taking("generating 1 latent images completed").or_else(|| taking("sampling completed")),
     taking("decode_first_stage completed"))
}

unsafe extern "C" fn log_cb(level: c_int, text: *const c_char, _user: *mut c_void) {
    // SAFETY: a NUL-terminated line from the library.
    let t = unsafe { CStr::from_ptr(text) }.to_string_lossy();
    {
        let mut l = LOG.lock().unwrap();
        if l.len() >= 4000 {
            l.drain(..2000);
        }
        l.push(t.trim_end().to_string());
    }
    if level >= 2 {
        eprint!("{t}");
    }
}

impl Sd {
    /// Load the model: `files` are (diffusion, llm, llm vision, vae), `backend` "cuda0", `offload` weights in host memory
    pub fn load(diffusion: &str, llm: &str, llm_vision: Option<&str>, vae: &str, backend: &str, max_vram_gib: f32, offload: bool, args: &str) -> Result<Sd> {
        let k = api()?;
        let (d, l, v) = (cstr(diffusion)?, cstr(llm)?, cstr(vae)?);
        let lv = llm_vision.map(cstr).transpose()?;
        let (b, a) = (cstr(backend)?, cstr(args)?);
        let mut ctx = std::ptr::null_mut();
        // SAFETY: valid C strings and an out-pointer to a local; the callbacks are plain functions.
        unsafe {
            (k.set_log)(Some(log_cb), std::ptr::null_mut());
            let rc = (k.create)(d.as_ptr(), l.as_ptr(), lv.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()), v.as_ptr(), b.as_ptr(), max_vram_gib,
                                offload as c_int, a.as_ptr(), &mut ctx);
            if rc != 0 {
                return Err(Error(format!("loading the model: {}", last(k))));
            }
        }
        Ok(Sd { k, ctx })
    }

    /// Stop the picture being made (callable from the progress callback)
    pub fn cancel(&self) {
        // SAFETY: a live context.
        unsafe { (self.k.cancel)(self.ctx) };
    }

    /// One picture. `progress(step, steps, seconds)` is called as the sampler goes; returning false cancels.
    pub fn generate(&self, g: &Gen, progress: &mut dyn FnMut(i32, i32, f32) -> bool) -> Result<Out> {
        struct Cb<'a> {
            f: &'a mut dyn FnMut(i32, i32, f32) -> bool,
            sd: *mut c_void,
            k: &'static Api,
            cancelled: bool,
        }
        unsafe extern "C" fn tramp(step: c_int, steps: c_int, secs: f32, user: *mut c_void) {
            // SAFETY: `user` is the Cb below, alive for the whole generate call, used on this thread only.
            let cb = unsafe { &mut *(user as *mut Cb) };
            if !(cb.f)(step, steps, secs) && !cb.cancelled {
                cb.cancelled = true;
                // SAFETY: the context made the call that is running this callback.
                unsafe { (cb.k.cancel)(cb.sd) };
            }
        }
        let k = self.k;
        let (p, n, l) = (cstr(g.prompt)?, cstr(g.negative)?, g.lora.map(|(p, _)| cstr(p)).transpose()?);
        let (sa, sc) = (cstr(g.sampler)?, cstr(g.scheduler)?);
        let init = g.init.as_ref();
        let ptrs: Vec<*const u8> = g.refs.iter().map(|r| r.data.as_ptr()).collect();
        let (rw, rh, rc): (Vec<c_int>, Vec<c_int>, Vec<c_int>) =
            (g.refs.iter().map(|r| r.w as c_int).collect(), g.refs.iter().map(|r| r.h as c_int).collect(), g.refs.iter().map(|r| r.c as c_int).collect());
        let mut cb = Cb { f: progress, sd: self.ctx, k, cancelled: false };
        let (mut out, mut ow, mut oh, mut oc) = (std::ptr::null_mut::<u8>(), 0, 0, 0);
        // SAFETY: every pointer is to memory that outlives the call; the progress user pointer is `cb`.
        let rc_ = unsafe {
            (k.set_progress)(Some(tramp), (&mut cb as *mut Cb).cast());
            let rc_ = (k.generate)(
                self.ctx, p.as_ptr(), n.as_ptr(), g.width as c_int, g.height as c_int, g.steps as c_int, g.cfg, g.seed as i64, sa.as_ptr(), sc.as_ptr(),
                g.flow_shift, g.rgba as c_int, g.strength, init.map_or(std::ptr::null(), |r| r.data.as_ptr()), init.map_or(0, |r| r.w as c_int),
                init.map_or(0, |r| r.h as c_int), init.map_or(0, |r| r.c as c_int), ptrs.as_ptr(), rw.as_ptr(), rh.as_ptr(), rc.as_ptr(), ptrs.len() as c_int,
                l.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()), g.lora.map_or(0.0, |(_, s)| s), &mut out, &mut ow, &mut oh, &mut oc,
            );
            (k.set_progress)(None, std::ptr::null_mut());
            rc_
        };
        if rc_ != 0 {
            return Err(Error(if cb.cancelled { "cancelled".into() } else { format!("making the picture: {}", last(k)) }));
        }
        let (w, h, c) = (ow as usize, oh as usize, oc as usize);
        // SAFETY: the bridge made w x h x c bytes at `out`.
        let data = unsafe { std::slice::from_raw_parts(out, w * h * c) }.to_vec();
        // SAFETY: from this library's allocator, freed once.
        unsafe { (k.free_image)(out) };
        Ok(Out { data, w, h, c })
    }
}
