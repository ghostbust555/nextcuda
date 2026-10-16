//! The safe layer over `libnextcuda-<kind>.so` (kernels/ns: CUDA behind nextsycl's ns.h ABI): GPUs, device and pinned
//! host buffers that free themselves, copies with their sizes checked. Everything the kernels run on comes from here.
//!
//! Without the library (the phase 1 python engine needs no kernels of ours) the GPUs are read with `nvidia-smi` and a
//! `DevBuf` is plain host memory with the same methods: the contracts and the glue behave as they did, and nothing in
//! them needs to know which GPU runtime is behind an engine.

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ffi::{c_char, c_void};
use std::process::Command;
use std::sync::{Arc, OnceLock};

pub mod options;

pub use nextcuda_sys::Api;
pub use options::{At, EngineOption};

#[derive(Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}

impl From<&str> for Error {
    fn from(s: &str) -> Error {
        Error(s.into())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();
static KIND: OnceLock<&'static str> = OnceLock::new();

/// The kind whose kernel library this process uses ("image", "llm", "video", "audio"; `libnextcuda-<kind>.so`), set
/// before the first GPU call - a process runs one kind. The default is "image". False when it was chosen already.
pub fn use_kind(kind: &'static str) -> bool {
    KIND.set(kind).is_ok() && API.get().is_none()
}

/// The kernel library, loaded once: the kind's (`use_kind`), or `NEXTCUDA_LIB`. None when there is none.
fn lib() -> Option<&'static Api> {
    API.get_or_init(|| Api::load(KIND.get().copied().unwrap_or("image"))).as_ref().ok()
}

/// The kernel library, or why it did not load
pub fn api() -> Result<&'static Api> {
    API.get_or_init(|| Api::load(KIND.get().copied().unwrap_or("image"))).as_ref().map_err(|e| Error(e.clone()))
}

/// A model's silo (`kernels/<kind>/<arch>/silo/`: its tuned kernels, `dist/silo/libnextcuda-<arch>.so`), opened on
/// the first ask and kept: None when there is none (or NS_SILO=0) - the engine then uses the shared kernels
pub fn silo(arch: &str) -> Result<Option<&'static nextcuda_sys::Silo>> {
    static SILOS: std::sync::Mutex<Vec<(String, Option<&'static nextcuda_sys::Silo>)>> = std::sync::Mutex::new(Vec::new());
    let mut m = SILOS.lock().unwrap();
    if let Some((_, s)) = m.iter().find(|(a, _)| a == arch) {
        return Ok(*s);
    }
    let s = api()?.silo(arch).map_err(Error)?.map(|s| &*Box::leak(Box::new(s)));
    m.push((arch.to_string(), s));
    Ok(s)
}

fn check(api: &Api, rc: i32, what: &str) -> Result<()> {
    if rc == 0 {
        Ok(())
    } else {
        Err(Error(format!("{what}: {}", api.error())))
    }
}

/// One row of `nvidia-smi`: (index, name, pci bus id, total MiB, free MiB)
type Row = (usize, String, String, u64, u64);

fn smi() -> Result<Vec<Row>> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=index,name,pci.bus_id,memory.total,memory.free", "--format=csv,noheader,nounits"])
        .output()
        .map_err(|e| Error(format!("listing the GPUs: nvidia-smi: {e}")))?;
    if !out.status.success() {
        return Err(Error(format!("listing the GPUs: nvidia-smi: {}", String::from_utf8_lossy(&out.stderr).trim())));
    }
    let mut rows = Vec::new();
    for l in String::from_utf8_lossy(&out.stdout).lines() {
        let f: Vec<&str> = l.split(',').map(str::trim).collect();
        if f.len() < 5 {
            continue;
        }
        let n = |s: &str| s.parse::<u64>().unwrap_or(0);
        // nvidia-smi writes the bus id as 00000000:01:00.0; sysfs wants 0000:01:00.0
        let pci = f[2].to_lowercase();
        let pci = pci.strip_prefix("0000").filter(|_| pci.len() > 12).map(str::to_string).unwrap_or(pci);
        rows.push((n(f[0]) as usize, f[1].to_string(), pci, n(f[3]), n(f[4])));
    }
    Ok(rows)
}

/// The GPUs: (index, name).
pub fn gpus() -> Result<Vec<(usize, String)>> {
    if let Some(a) = lib() {
        // SAFETY: no arguments.
        let n = unsafe { (a.gpu_count)() };
        if n < 0 {
            return Err(Error(format!("listing the GPUs: {}", a.error())));
        }
        return (0..n)
            .map(|i| {
                let mut buf = [0 as c_char; 256];
                // SAFETY: buf is 256 bytes, its length passed.
                check(a, unsafe { (a.gpu_name)(i, buf.as_mut_ptr(), buf.len()) }, "the GPU's name")?;
                // SAFETY: the library wrote a NUL-terminated string into buf.
                Ok((i as usize, unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned()))
            })
            .collect();
    }
    Ok(smi()?.into_iter().map(|r| (r.0, r.1)).collect())
}

/// GPU `index`'s compute units and clock. The driver gives neither through nvidia-smi's plain query; (0, 0) = unknown.
pub fn gpu_units(index: usize) -> Result<(u32, u32)> {
    let Some(a) = lib() else { return Ok((0, 0)) };
    let (mut u, mut m) = (0, 0);
    // SAFETY: out-pointers to locals.
    check(a, unsafe { (a.gpu_units)(index as i32, &mut u, &mut m) }, "the GPU's compute units")?;
    Ok((u as u32, m as u32))
}

/// Every GPU's index, the one with the most memory last; ties keep the driver's order.
pub fn gpus_weakest_first() -> Result<Vec<usize>> {
    let mut v: Vec<(usize, u64)> = smi()?.into_iter().map(|r| (r.0, r.3)).collect();
    v.sort_by_key(|x| x.1);
    Ok(v.into_iter().map(|x| x.0).collect())
}

/// GPU `index`'s PCIe link as nvidia-smi gives it: "gen4 x16 (max gen4 x16)". The link drops a generation when the
/// card is idle (power saving): read it under load for what a copy gets.
pub fn pcie_link(index: usize) -> Option<String> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=index,pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout).lines().find_map(|l| {
        let f: Vec<&str> = l.split(',').map(str::trim).collect();
        (f.len() >= 5 && f[0].parse::<usize>().ok() == Some(index)).then(|| format!("gen{} x{} (max gen{} x{})", f[1], f[2], f[3], f[4]))
    })
}

/// One GPU.
pub struct Gpu {
    /// the kernel library and this GPU's handle in it; None: no library (host memory, nvidia-smi)
    dev: Option<(&'static Api, nextcuda_sys::Gpu)>,
    pub index: usize,
    pub name: String,
    /// its PCI address ("0000:03:00.0"), where its sensors are (sysfs hwmon)
    pub pci: Option<String>,
}

// SAFETY: the library's GPU handle is a context and streams; CUDA's are thread-safe.
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

impl Drop for Gpu {
    fn drop(&mut self) {
        if let Some((api, raw)) = self.dev {
            // SAFETY: the handle came from gpu_open; buffers keep an Arc<Gpu>, so none outlives it.
            unsafe {
                let _ = (api.sync)(raw);
                (api.gpu_close)(raw)
            }
        }
    }
}

impl Gpu {
    pub fn open(index: usize) -> Result<Arc<Gpu>> {
        if let Some(api) = lib() {
            let name = gpus()?.into_iter().find(|g| g.0 == index).map(|g| g.1).ok_or_else(|| Error(format!("no GPU {index}")))?;
            let mut raw: nextcuda_sys::Gpu = std::ptr::null_mut();
            // SAFETY: an out-pointer to a local.
            check(api, unsafe { (api.gpu_open)(index as i32, &mut raw) }, &format!("opening GPU {index}"))?;
            let mut pb = [0 as c_char; 64];
            // SAFETY: pb is 64 bytes, its length passed; the library NUL-terminates.
            let pci = (unsafe { (api.gpu_pci)(index as i32, pb.as_mut_ptr(), pb.len()) } == 0)
                .then(|| unsafe { std::ffi::CStr::from_ptr(pb.as_ptr()) }.to_string_lossy().into_owned())
                .filter(|s| !s.is_empty());
            return Ok(Arc::new(Gpu { dev: Some((api, raw)), index, name, pci }));
        }
        let r = smi()?.into_iter().find(|r| r.0 == index).ok_or_else(|| Error(format!("no GPU {index}")))?;
        Ok(Arc::new(Gpu { dev: None, index, name: r.1, pci: Some(r.2).filter(|s| !s.is_empty()) }))
    }

    /// The library's handle of this GPU (`ns_gpu*`), for an engine's own ABI; null without the library
    pub fn raw(&self) -> *mut c_void {
        self.dev.map_or(std::ptr::null_mut(), |d| d.1)
    }

    /// The stream, for a kernel launch (the pointer is a cudaStream_t)
    pub fn queue(&self) -> *mut c_void {
        // SAFETY: a live handle.
        self.dev.map_or(std::ptr::null_mut(), |(a, r)| unsafe { (a.gpu_queue)(r) })
    }

    /// (total bytes, free bytes)
    pub fn memory(&self) -> Result<(u64, Option<u64>)> {
        if let Some((a, r)) = self.dev {
            let (mut t, mut f) = (0u64, 0i64);
            // SAFETY: out-pointers to locals.
            check(a, unsafe { (a.gpu_memory)(r, &mut t, &mut f) }, "the GPU's memory")?;
            return Ok((t, (f >= 0).then_some(f as u64)));
        }
        let r = smi()?.into_iter().find(|r| r.0 == self.index).ok_or_else(|| Error(format!("no GPU {}", self.index)))?;
        Ok((r.3 << 20, Some(r.4 << 20)))
    }

    pub fn sync(&self) -> Result<()> {
        match self.dev {
            // SAFETY: a live handle.
            Some((a, r)) => check(a, unsafe { (a.sync)(r) }, "waiting for the GPU"),
            None => Ok(()),
        }
    }
}

/// A buffer the contracts pass around. Host memory, 64-byte aligned.
pub struct DevBuf {
    gpu: Arc<Gpu>,
    ptr: *mut c_void,
    pub len: usize,
    /// false for a view: the owner frees the memory
    owned: bool,
}

// SAFETY: plain memory, accessed through &self by callers that keep their own order.
unsafe impl Send for DevBuf {}
unsafe impl Sync for DevBuf {}

fn layout(len: usize) -> Layout {
    Layout::from_size_align(len.max(1), 64).expect("a buffer layout")
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        if self.owned {
            match self.gpu.dev {
                // SAFETY: allocated by this GPU; the stream is in order, so work queued on it runs first.
                Some((a, r)) => unsafe {
                    (a.free)(r, self.ptr);
                },
                // SAFETY: allocated in `new` with this layout.
                None => unsafe { dealloc(self.ptr.cast(), layout(self.len)) },
            }
        }
    }
}

impl DevBuf {
    pub fn new(gpu: &Arc<Gpu>, len: usize) -> Result<DevBuf> {
        if let Some((a, r)) = gpu.dev {
            let mut ptr = std::ptr::null_mut();
            // SAFETY: an out-pointer to a local.
            check(a, unsafe { (a.alloc)(r, len.max(1), &mut ptr) }, &format!("allocating {len} bytes on {}", gpu.name))?;
            return Ok(DevBuf { gpu: gpu.clone(), ptr, len, owned: true });
        }
        // SAFETY: a non-zero layout.
        let ptr = unsafe { alloc_zeroed(layout(len)) };
        if ptr.is_null() {
            return Err(Error(format!("allocating {len} bytes")));
        }
        Ok(DevBuf { gpu: gpu.clone(), ptr: ptr.cast(), len, owned: true })
    }
    pub fn ptr(&self) -> *mut c_void {
        self.ptr
    }
    pub fn gpu(&self) -> &Arc<Gpu> {
        &self.gpu
    }
    pub fn write(&self, at: usize, src: &[u8]) -> Result<()> {
        self.bounds(at, src.len())?;
        if let Some((a, r)) = self.gpu.dev {
            // SAFETY: in bounds (checked); synced before `src` can go.
            check(a, unsafe { (a.copy_to)(r, self.ptr.cast::<u8>().add(at).cast(), src.as_ptr().cast(), src.len()) }, "copying to the GPU")?;
            return self.gpu.sync();
        }
        // SAFETY: in bounds (checked).
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr.cast::<u8>().add(at), src.len()) };
        Ok(())
    }
    pub fn write_async(&self, at: usize, src: &[u8]) -> Result<()> {
        // a pageable source is staged by the driver before the call returns: no ring needed
        self.write(at, src)
    }
    pub fn read(&self, at: usize, dst: &mut [u8]) -> Result<()> {
        self.bounds(at, dst.len())?;
        if let Some((a, r)) = self.gpu.dev {
            // SAFETY: in bounds (checked); synced before returning.
            check(a, unsafe { (a.copy_from)(r, dst.as_mut_ptr().cast(), self.ptr.cast::<u8>().add(at).cast(), dst.len()) }, "copying from the GPU")?;
            return self.gpu.sync();
        }
        // SAFETY: in bounds (checked).
        unsafe { std::ptr::copy_nonoverlapping(self.ptr.cast::<u8>().add(at), dst.as_mut_ptr(), dst.len()) };
        Ok(())
    }
    pub fn fill(&self, v: u8) -> Result<()> {
        if let Some((a, r)) = self.gpu.dev {
            // SAFETY: the whole buffer.
            return check(a, unsafe { (a.fill)(r, self.ptr, v, self.len) }, "filling");
        }
        // SAFETY: the whole buffer.
        unsafe { std::ptr::write_bytes(self.ptr.cast::<u8>(), v, self.len) };
        Ok(())
    }
    /// `[at, at + n)` inside this buffer, or an error
    pub fn bounds(&self, at: usize, n: usize) -> Result<()> {
        if at.checked_add(n).is_none_or(|e| e > self.len) {
            return Err(Error(format!("bytes {at}..{} of a {}-byte buffer", at.saturating_add(n), self.len)));
        }
        Ok(())
    }
    /// All of `src` into the front of this buffer (the staging area is not needed here).
    pub fn copy_from_peer(&self, src: &DevBuf, staging: &mut [u8]) -> Result<()> {
        self.bounds(0, src.len)?;
        if let (Some((a, to)), Some((_, from))) = (self.gpu.dev, src.gpu.dev) {
            if staging.len() < src.len {
                return Err(Error(format!("a staging area of {} bytes for {}", staging.len(), src.len)));
            }
            // SAFETY: both buffers live, sizes checked; the call waits for both copies.
            return check(a, unsafe { (a.copy_peer)(to, self.ptr, from, src.ptr, src.len, staging.as_mut_ptr().cast()) }, "copying between GPUs");
        }
        // SAFETY: both in bounds; distinct allocations.
        unsafe { std::ptr::copy_nonoverlapping(src.ptr.cast::<u8>(), self.ptr.cast::<u8>(), src.len) };
        Ok(())
    }
    pub fn f32(gpu: &Arc<Gpu>, n: usize) -> Result<DevBuf> {
        DevBuf::new(gpu, n * 4)
    }
    pub fn from_f32(gpu: &Arc<Gpu>, v: &[f32]) -> Result<DevBuf> {
        let b = DevBuf::f32(gpu, v.len())?;
        // SAFETY: plain floats as bytes.
        let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), v.len() * 4) };
        b.write(0, bytes)?;
        Ok(b)
    }
    pub fn floats(&self) -> usize {
        self.len / 4
    }
    pub fn fp(&self) -> *mut f32 {
        self.ptr.cast()
    }
    pub fn to_f32(&self) -> Result<Vec<f32>> {
        let mut v = vec![0f32; self.floats()];
        // SAFETY: v is floats() floats, viewed as their bytes.
        let bytes = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), v.len() * 4) };
        self.read(0, bytes)?;
        Ok(v)
    }
    pub fn copy_within(&self, at: usize, src: &DevBuf, src_at: usize, n: usize) -> Result<()> {
        self.bounds(at, n)?;
        src.bounds(src_at, n)?;
        if let Some((a, r)) = self.gpu.dev {
            // SAFETY: both ranges in bounds (checked), same GPU.
            return check(a, unsafe { (a.copy_dev)(r, self.ptr.cast::<u8>().add(at).cast(), src.ptr.cast::<u8>().add(src_at).cast(), n) }, "copying on the GPU");
        }
        // SAFETY: both ranges in bounds; `copy` allows overlap.
        unsafe { std::ptr::copy(src.ptr.cast::<u8>().add(src_at), self.ptr.cast::<u8>().add(at), n) };
        Ok(())
    }
    /// Bytes `[at, at + len)` as a buffer of its own that does not free them.
    pub fn view(&self, at: usize, len: usize) -> Result<DevBuf> {
        self.bounds(at, len)?;
        // SAFETY: in bounds (checked).
        Ok(DevBuf { gpu: self.gpu.clone(), ptr: unsafe { self.ptr.cast::<u8>().add(at).cast() }, len, owned: false })
    }
}

/// Host memory kept for fast copies on a CUDA/SYCL runtime; here an ordinary allocation.
pub struct HostBuf {
    data: Vec<u8>,
    pub len: usize,
    gpu: Arc<Gpu>,
}

impl HostBuf {
    pub fn new(gpu: &Arc<Gpu>, len: usize) -> Result<HostBuf> {
        Ok(HostBuf { data: vec![0; len.max(1)], len, gpu: gpu.clone() })
    }
    pub fn device_view(&mut self, at: usize, len: usize) -> Result<DevBuf> {
        if at.checked_add(len).is_none_or(|e| e > self.len) {
            return Err(Error(format!("device_view: [{at}, {}) outside {} bytes", at.saturating_add(len), self.len)));
        }
        // SAFETY: in bounds (checked); the buffer outlives the view by the caller's contract.
        Ok(DevBuf { gpu: self.gpu.clone(), ptr: unsafe { self.data.as_mut_ptr().add(at).cast() }, len, owned: false })
    }
    pub fn as_slice(&self) -> &[u8] {
        &self.data[..self.len]
    }
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data[..self.len]
    }
}

/// A bump allocator over one allocation, as the engines on a device use it.
pub struct Arena {
    buf: std::sync::Mutex<DevBuf>,
    off: std::sync::Mutex<usize>,
    pub peak: std::sync::atomic::AtomicUsize,
    pub spills: std::sync::atomic::AtomicUsize,
}

impl Arena {
    pub fn new(gpu: &Arc<Gpu>, bytes: usize) -> Result<Arena> {
        Ok(Arena::on(DevBuf::new(gpu, bytes)?))
    }
    pub fn on(buf: DevBuf) -> Arena {
        Arena { buf: std::sync::Mutex::new(buf), off: std::sync::Mutex::new(0), peak: 0.into(), spills: 0.into() }
    }
    pub fn set(&self, buf: DevBuf) {
        *self.buf.lock().unwrap() = buf;
        *self.off.lock().unwrap() = 0;
    }
    pub fn len(&self) -> usize {
        self.buf.lock().unwrap().len
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn bytes(&self, n: usize) -> Result<DevBuf> {
        let mut off = self.off.lock().unwrap();
        let buf = self.buf.lock().unwrap();
        let at = off.next_multiple_of(256);
        if at + n > buf.len {
            self.spills.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return DevBuf::new(&buf.gpu, n);
        }
        *off = at + n;
        self.peak.fetch_max(*off, std::sync::atomic::Ordering::Relaxed);
        buf.view(at, n.max(1).min(buf.len - at))
    }
    pub fn f32(&self, n: usize) -> Result<DevBuf> {
        self.bytes(n * 4)
    }
    pub fn reset(&self) {
        *self.off.lock().unwrap() = 0;
    }
    pub fn mark(&self) -> usize {
        *self.off.lock().unwrap()
    }
    pub fn rewind(&self, mark: usize) {
        let mut off = self.off.lock().unwrap();
        if mark < *off {
            *off = mark;
        }
    }
}
