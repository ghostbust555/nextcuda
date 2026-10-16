//! The CPU's share of a decode layer's experts (after nextsycl's deepseek2 `cpu.rs`, and Strata's CPU tier): experts in
//! pinned host memory are multiplied where they are, by the host's cores, instead of being copied over PCIe - the
//! link is decode's bottleneck (about 1 GB of experts a token at ~24 GB/s), the host's memory is not, and the two run
//! at once. The products are ggml-cpu's (llama.cpp's CPU backend: every i-quant and K-quant, AVX-512 / AVX2), loaded
//! at run time from `NS_NH_GGML_DIR` (default: beside the program) as `libggml-base.so` + `libggml-cpu.so`; without
//! them the engine copies every expert to the GPU as before.

use std::ffi::{c_char, c_int, c_void, CString};
use std::sync::{Arc, Condvar, Mutex};

extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}
const RTLD_NOW: c_int = 2;
const RTLD_GLOBAL: c_int = 0x100;

type FromFloat = unsafe extern "C" fn(x: *const f32, y: *mut c_void, k: i64);
type VecDot = unsafe extern "C" fn(n: c_int, s: *mut f32, bs: usize, x: *const c_void, bx: usize, y: *const c_void, by: usize, nrc: c_int);

/// ggml-cpu.h's `struct ggml_type_traits_cpu`
#[repr(C)]
struct Traits {
    from_float: Option<FromFloat>,
    vec_dot: Option<VecDot>,
    vec_dot_type: c_int,
    nrows: i64,
}

pub struct Ggml {
    traits: unsafe extern "C" fn(c_int) -> *const Traits,
    row_size: unsafe extern "C" fn(c_int, i64) -> usize,
}

// SAFETY: function pointers into a library that stays loaded.
unsafe impl Send for Ggml {}
unsafe impl Sync for Ggml {}

impl Ggml {
    /// The library, loaded once: None when it is not there (or NS_NH_CPU=0)
    pub fn get() -> Option<&'static Ggml> {
        static G: std::sync::OnceLock<Option<Ggml>> = std::sync::OnceLock::new();
        G.get_or_init(|| {
            if std::env::var("NS_NH_CPU").is_ok_and(|v| v == "0") {
                return None;
            }
            match Ggml::load() {
                Ok(g) => Some(g),
                Err(x) => {
                    eprintln!("[nemotronh: no CPU experts ({x})]");
                    None
                }
            }
        })
        .as_ref()
    }

    fn load() -> Result<Ggml, String> {
        let dir = std::env::var("NS_NH_GGML_DIR").map(std::path::PathBuf::from).or_else(|_| {
            std::env::current_exe().map_err(|x| x.to_string()).and_then(|p| p.parent().map(|d| d.to_path_buf()).ok_or_else(|| "no directory".into()))
        })?;
        let open = |name: &str| -> Result<*mut c_void, String> {
            let p = dir.join(name);
            let c = CString::new(p.to_string_lossy().as_bytes()).map_err(|x| x.to_string())?;
            // SAFETY: a path; the handle is kept for the process's life.
            let h = unsafe { dlopen(c.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
            if h.is_null() {
                Err(format!("{} not loaded", p.display()))
            } else {
                Ok(h)
            }
        };
        let base = open("libggml-base.so")?;
        let cpu = open("libggml-cpu.so")?;
        let sym = |h: *mut c_void, n: &str| -> Result<*mut c_void, String> {
            let c = CString::new(n).unwrap();
            // SAFETY: a symbol of a loaded library.
            let p = unsafe { dlsym(h, c.as_ptr()) };
            if p.is_null() {
                Err(format!("no {n}"))
            } else {
                Ok(p)
            }
        };
        // SAFETY: the symbols' C signatures (ggml-cpu.h, ggml.h).
        unsafe {
            let init: unsafe extern "C" fn() = std::mem::transmute(sym(cpu, "ggml_cpu_init")?);
            init();
            Ok(Ggml { traits: std::mem::transmute(sym(cpu, "ggml_get_type_traits_cpu")?), row_size: std::mem::transmute(sym(base, "ggml_row_size")?) })
        }
    }

    /// Whether ggml multiplies `ty` (a GGUF type code) against a quantized vector
    pub fn supports(&self, ty: i32) -> bool {
        // SAFETY: a type code below GGML_TYPE_COUNT is asked of; beyond it ggml aborts, so only the known ones.
        if !(0..64).contains(&ty) {
            return false;
        }
        let t = unsafe { &*(self.traits)(ty) };
        t.vec_dot.is_some() && unsafe { (*(self.traits)(t.vec_dot_type)).from_float.is_some() }
    }

    fn row(&self, ty: i32, n: usize) -> usize {
        // SAFETY: arithmetic.
        unsafe { (self.row_size)(ty, n as i64) }
    }

    /// x quantized as `ty`'s products take it
    fn quantize(&self, ty: i32, x: &[f32]) -> Vec<u8> {
        // SAFETY: the traits of a supported type; the buffer holds a row of x.len() values of the vec_dot type.
        unsafe {
            let vt = (*(self.traits)(ty)).vec_dot_type;
            let mut q = vec![0u8; self.row(vt, x.len()) + 64];
            ((*(self.traits)(vt)).from_float.unwrap())(x.as_ptr(), q.as_mut_ptr().cast(), x.len() as i64);
            q
        }
    }

    /// y[r] = row r of w (`rows` rows of `cols`, type `ty`) . xq, for r in `range`
    fn matvec(&self, ty: i32, w: *const u8, cols: usize, xq: &[u8], range: std::ops::Range<usize>, y: &mut [f32]) {
        let rb = self.row(ty, cols);
        // SAFETY: rows of rb bytes from w (the caller's slot), xq a quantized row of cols, y as long as the range.
        unsafe {
            let dot = (*(self.traits)(ty)).vec_dot.unwrap();
            for (i, r) in range.enumerate() {
                dot(cols as c_int, y.as_mut_ptr().add(i), 0, w.add(r * rb).cast(), 0, xq.as_ptr().cast(), 0, 1);
            }
        }
    }
}

type JobFn = Arc<dyn Fn(usize) + Send + Sync>;

/// Workers that take the parts of a job (nextsycl's pool): the caller works too, and `run` returns when all are done
pub struct Pool {
    shared: Arc<Shared>,
    pub threads: usize,
}

struct Shared {
    /// (the job, its generation, its parts, stop)
    m: Mutex<(Option<JobFn>, u64, usize, bool)>,
    work: Condvar,
    /// generation << 32 | the next part
    ctr: std::sync::atomic::AtomicU64,
    left: std::sync::atomic::AtomicUsize,
}

impl Pool {
    pub fn new(threads: usize) -> Pool {
        let shared = Arc::new(Shared { m: Mutex::new((None, 0, 0, false)), work: Condvar::new(), ctr: 0.into(), left: 0.into() });
        for _ in 0..threads.saturating_sub(1) {
            let sh = shared.clone();
            std::thread::spawn(move || {
                let mut seen = 0u64;
                loop {
                    let (job, gen, parts) = {
                        let mut g = sh.m.lock().unwrap();
                        loop {
                            if g.3 {
                                return;
                            }
                            if g.1 != seen {
                                seen = g.1;
                                if let Some(j) = g.0.clone() {
                                    break (j, g.1, g.2);
                                }
                            }
                            g = sh.work.wait(g).unwrap();
                        }
                    };
                    Self::work(&sh, &job, gen, parts);
                }
            });
        }
        Pool { shared, threads }
    }

    fn work(sh: &Shared, job: &JobFn, gen: u64, parts: usize) {
        use std::sync::atomic::Ordering::*;
        loop {
            let cur = sh.ctr.load(Acquire);
            if cur >> 32 != gen & 0xffff_ffff {
                return;
            }
            let i = (cur & 0xffff_ffff) as usize;
            if i >= parts {
                return;
            }
            if sh.ctr.compare_exchange_weak(cur, cur + 1, AcqRel, Acquire).is_err() {
                continue;
            }
            // a part that panics still counts as done (else `run` would wait forever); the panic is reported
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job(i))).is_err() {
                eprintln!("[nemotronh cpu: a part of the experts' products panicked]");
            }
            sh.left.fetch_sub(1, AcqRel);
        }
    }

    /// f(0..parts) over the workers and this thread; returns when every part is done
    pub fn run(&self, parts: usize, f: JobFn) {
        use std::sync::atomic::Ordering::*;
        if parts == 0 {
            return;
        }
        let sh = &*self.shared;
        let gen = {
            let mut g = sh.m.lock().unwrap();
            g.1 += 1;
            g.0 = Some(f.clone());
            g.2 = parts;
            sh.left.store(parts, Release);
            sh.ctr.store((g.1 & 0xffff_ffff) << 32, Release);
            g.1
        };
        sh.work.notify_all();
        Self::work(sh, &f, gen, parts);
        while sh.left.load(Acquire) > 0 {
            std::hint::spin_loop();
        }
        sh.m.lock().unwrap().0 = None;
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.m.lock().unwrap().3 = true;
        self.shared.work.notify_all();
    }
}

/// A raw pointer the pool's parts write their own disjoint ranges through
#[derive(Clone, Copy)]
struct Out(*mut f32);
// SAFETY: every part writes a range no other part writes; the buffer outlives `run` (it waits for all parts).
unsafe impl Send for Out {}
unsafe impl Sync for Out {}
impl Out {
    /// SAFETY: the caller's range is inside the buffer and written by it alone
    #[allow(clippy::mut_from_ref)]
    unsafe fn at(&self, i: usize, n: usize) -> &mut [f32] {
        std::slice::from_raw_parts_mut(self.0.add(i), n)
    }
}

/// One expert the CPU multiplies: its up [n_ff, latent] and down [latent, n_ff] matrices' bytes in pinned memory, and its weight
pub struct Job {
    pub up: usize,
    pub down: usize,
    pub w: f32,
}

/// sum over the jobs of w * down . relu^2(up . x): the latent's [latent] for the GPU to add
pub fn experts(g: &'static Ggml, pool: &Pool, jobs: &[Job], up_ty: i32, down_ty: i32, x: &[f32], n_ff: usize) -> Vec<f32> {
    const ROWS: usize = 64;
    let (n, lat) = (jobs.len(), x.len());
    if n == 0 {
        return vec![0.0; lat];
    }
    let addrs: Arc<Vec<(usize, usize)>> = Arc::new(jobs.iter().map(|j| (j.up, j.down)).collect());
    let xq = Arc::new(g.quantize(up_ty, x));
    let mut u = vec![0f32; n * n_ff];
    let chunks = n_ff.div_ceil(ROWS);
    {
        let (addrs, out, xq) = (addrs.clone(), Out(u.as_mut_ptr()), xq.clone());
        pool.run(n * chunks, Arc::new(move |i| {
            let (e, c) = (i / chunks, i % chunks);
            let r = c * ROWS..((c + 1) * ROWS).min(n_ff);
            // SAFETY: rows [e][r] of u are this part's alone
            let y = unsafe { out.at(e * n_ff + r.start, r.len()) };
            g.matvec(up_ty, addrs[e].0 as *const u8, lat, &xq, r, y);
        }));
    }
    let uq: Arc<Vec<Vec<u8>>> = Arc::new(u.chunks(n_ff).map(|v| {
        let h: Vec<f32> = v.iter().map(|a| {
            let r = a.max(0.0);
            r * r
        }).collect();
        g.quantize(down_ty, &h)
    }).collect());
    let mut d = vec![0f32; n * lat];
    let dchunks = lat.div_ceil(ROWS);
    {
        let (addrs, out, uq) = (addrs.clone(), Out(d.as_mut_ptr()), uq.clone());
        pool.run(n * dchunks, Arc::new(move |i| {
            let (e, c) = (i / dchunks, i % dchunks);
            let r = c * ROWS..((c + 1) * ROWS).min(lat);
            // SAFETY: as above
            let y = unsafe { out.at(e * lat + r.start, r.len()) };
            g.matvec(down_ty, addrs[e].1 as *const u8, n_ff, &uq[e], r, y);
        }));
    }
    let mut out = vec![0f32; lat];
    for (j, row) in jobs.iter().zip(d.chunks(lat)) {
        for (o, v) in out.iter_mut().zip(row) {
            *o += j.w * v;
        }
    }
    out
}
