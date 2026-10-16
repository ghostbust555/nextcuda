//! Community GGUFs that store `img_in.weight` reshaped (realrebelai's Qwen-Image-2.1-Q4 has it as [1024, 256] where
//! the model's is [4096, 64]: the same 262,144 values in the same order, a shape written by a converter). The
//! stable-diffusion.cpp loader derives the hidden size from that tensor's second dimension and refuses the file, so
//! the dimensions are put right - in a copy beside the file, made once, never in the user's own file.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use nextcuda_image::{Error, Result};

fn io(p: &Path, e: std::io::Error) -> Error {
    Error(format!("{}: {e}", p.display()))
}

fn u32le(f: &mut File) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    f.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn u64le(f: &mut File) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    f.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn skip_string(f: &mut File) -> std::io::Result<()> {
    let n = u64le(f)?;
    f.seek(SeekFrom::Current(n as i64)).map(|_| ())
}

fn string(f: &mut File) -> std::io::Result<String> {
    let n = u64le(f)? as usize;
    let mut b = vec![0u8; n];
    f.read_exact(&mut b)?;
    Ok(String::from_utf8_lossy(&b).into_owned())
}

fn skip_value(f: &mut File, t: u32) -> std::io::Result<()> {
    match t {
        8 => skip_string(f),
        9 => {
            let it = u32le(f)?;
            let n = u64le(f)?;
            for _ in 0..n {
                skip_value(f, it)?;
            }
            Ok(())
        }
        0 | 1 | 7 => f.seek(SeekFrom::Current(1)).map(|_| ()),
        2 | 3 => f.seek(SeekFrom::Current(2)).map(|_| ()),
        4..=6 => f.seek(SeekFrom::Current(4)).map(|_| ()),
        _ => f.seek(SeekFrom::Current(8)).map(|_| ()),
    }
}

/// (file offset of the tensor's dims, its dims) for `name`
fn find(path: &Path, name: &str) -> Result<Option<(u64, Vec<u64>)>> {
    let mut f = File::open(path).map_err(|e| io(path, e))?;
    let r = (|| -> std::io::Result<Option<(u64, Vec<u64>)>> {
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"GGUF" {
            return Ok(None);
        }
        let _ver = u32le(&mut f)?;
        let (nt, nkv) = (u64le(&mut f)?, u64le(&mut f)?);
        for _ in 0..nkv {
            skip_string(&mut f)?;
            let t = u32le(&mut f)?;
            skip_value(&mut f, t)?;
        }
        for _ in 0..nt {
            let n = string(&mut f)?;
            let d = u32le(&mut f)? as usize;
            let pos = f.stream_position()?;
            let dims: Vec<u64> = (0..d).map(|_| u64le(&mut f)).collect::<std::io::Result<_>>()?;
            f.seek(SeekFrom::Current(12))?;
            if n == name {
                return Ok(Some((pos, dims)));
            }
        }
        Ok(None)
    })();
    r.map_err(|e| io(path, e))
}

/// `path`, or a copy with `img_in.weight` as [64, 4096] when the file stores it reshaped. The copy is `<stem>.nc.gguf`
/// beside the file (made once; the model's files directory is the user's own to keep)
pub fn qwen_image_transformer(path: &Path, log: &mut dyn FnMut(String)) -> Result<PathBuf> {
    let Some((pos, dims)) = find(path, "img_in.weight")? else { return Ok(path.to_path_buf()) };
    // ggml order: [in, out] = [64, 4096]
    const RIGHT: [u64; 2] = [64, 4096];
    if dims == RIGHT || dims.iter().product::<u64>() != RIGHT[0] * RIGHT[1] {
        return Ok(path.to_path_buf());
    }
    let fixed = path.with_extension("nc.gguf");
    let size_ok = std::fs::metadata(&fixed).ok().zip(std::fs::metadata(path).ok()).is_some_and(|(a, b)| a.len() == b.len());
    if size_ok && find(&fixed, "img_in.weight")?.is_some_and(|(_, d)| d == RIGHT) {
        return Ok(fixed);
    }
    log(format!("{}: img_in.weight is stored as {dims:?}, not {RIGHT:?}: a corrected copy is made once ({})", path.display(), fixed.display()));
    let tmp = path.with_extension("nc.gguf.part");
    std::fs::copy(path, &tmp).map_err(|e| io(&tmp, e))?;
    let mut f = std::fs::OpenOptions::new().write(true).open(&tmp).map_err(|e| io(&tmp, e))?;
    f.seek(SeekFrom::Start(pos)).and_then(|_| {
        let mut b = Vec::new();
        for d in RIGHT {
            b.extend_from_slice(&d.to_le_bytes());
        }
        f.write_all(&b)
    }).map_err(|e| io(&tmp, e))?;
    drop(f);
    std::fs::rename(&tmp, &fixed).map_err(|e| io(&fixed, e))?;
    Ok(fixed)
}
