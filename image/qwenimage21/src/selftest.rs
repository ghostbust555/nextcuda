//! The engine's kernels against host arithmetic on the GPU (`nextcuda image selftest`): each of qi21.h's operations on
//! small inputs, compared with the same computation here. The numbers the kernel library must reproduce for the
//! engine above it to be the one nextsycl's tests were written against.

use std::sync::Arc;

use nextcuda_core::{DevBuf, Error, Gpu, Result};

use crate::ffi::{api, check};

fn half_to_f32(h: u16) -> f32 {
    let (s, e, m) = ((h >> 15) & 1, (h >> 10) & 0x1f, h & 0x3ff);
    let v = match e {
        0 => (m as f32) * 2f32.powi(-24),
        31 => if m == 0 { f32::INFINITY } else { f32::NAN },
        _ => (1.0 + m as f32 / 1024.0) * 2f32.powi(e as i32 - 15),
    };
    if s == 1 { -v } else { v }
}

fn read_half(b: &DevBuf, n: usize) -> Result<Vec<f32>> {
    let mut raw = vec![0u8; n * 2];
    b.read(0, &mut raw)?;
    Ok(raw.chunks_exact(2).map(|c| half_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect())
}

/// A deterministic spread of values in [-2, 2)
fn data(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(12345);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1u32 << 24) as f32) * 4.0 - 2.0
        })
        .collect()
}

fn within(what: &str, got: &[f32], want: &[f32], tol: f32) -> Result<f32> {
    let mut worst = 0.0f32;
    for (g, w) in got.iter().zip(want) {
        worst = worst.max((g - w).abs() / w.abs().max(1.0));
    }
    if got.len() != want.len() || worst > tol {
        return Err(Error(format!("{what}: worst relative error {worst:.2e} over {} values (limit {tol:.0e})", want.len())));
    }
    Ok(worst)
}

/// Every kernel of qi21.h once: the worst relative error of each, in order
pub fn run(gpu: &Arc<Gpu>) -> Result<Vec<f32>> {
    let k = api()?;
    let g = gpu.raw();
    let mut out = Vec::new();

    // axpy: x += a * y
    {
        let (n, a) = (1000usize, 0.75f32);
        let (x, y) = (data(n, 1), data(n, 2));
        let dx = DevBuf::from_f32(gpu, &x)?;
        let dy = DevBuf::from_f32(gpu, &y)?;
        // SAFETY: n floats each, live buffers.
        check(unsafe { (k.axpy)(g, dx.fp(), dy.fp(), n as i64, a) }, "axpy")?;
        let want: Vec<f32> = x.iter().zip(&y).map(|(x, y)| x + a * y).collect();
        out.push(within("axpy", &dx.to_f32()?, &want, 1e-6)?);
    }
    // to_half, to_float: a round trip through half
    {
        let n = 777usize;
        let x = data(n, 3);
        let dx = DevBuf::from_f32(gpu, &x)?;
        let h = DevBuf::new(gpu, n * 2)?;
        let back = DevBuf::f32(gpu, n)?;
        // SAFETY: sized above.
        check(unsafe { (k.to_half)(g, dx.fp(), h.ptr(), n as i64) }, "to_half")?;
        check(unsafe { (k.to_float)(g, h.ptr(), back.fp(), n as i64) }, "to_float")?;
        out.push(within("half round trip", &back.to_f32()?, &x, 1e-3)?);
    }
    // ln_mod: layernorm without affine, times (1 + scale)
    {
        let (m, c, eps) = (5usize, 1536usize, 1e-6f32);
        let x = data(m * c, 4);
        let scale = data(c, 5);
        let dx = DevBuf::from_f32(gpu, &x)?;
        let ds = DevBuf::from_f32(gpu, &scale)?;
        let o = DevBuf::new(gpu, m * c * 2)?;
        // SAFETY: sized above.
        check(unsafe { (k.ln_mod)(g, dx.fp(), m as i64, c as i64, eps, ds.fp(), o.ptr()) }, "ln_mod")?;
        let mut want = Vec::with_capacity(m * c);
        for r in 0..m {
            let row = &x[r * c..(r + 1) * c];
            let mean = row.iter().sum::<f32>() / c as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / c as f32;
            let inv = 1.0 / (var + eps).sqrt();
            want.extend(row.iter().zip(&scale).map(|(v, s)| (v - mean) * inv * (1.0 + s)));
        }
        out.push(within("ln_mod", &read_half(&o, m * c)?, &want, 4e-3)?);
    }
    // softmax_rows
    {
        let (m, n, sc) = (4usize, 300usize, 0.5f32);
        let x = data(m * n, 6);
        let dx = DevBuf::from_f32(gpu, &x)?;
        let o = DevBuf::new(gpu, m * n * 2)?;
        // SAFETY: sized above.
        check(unsafe { (k.softmax_rows)(g, dx.fp(), m as i64, n as i64, sc, o.ptr()) }, "softmax_rows")?;
        let mut want = Vec::new();
        for r in 0..m {
            let row: Vec<f32> = x[r * n..(r + 1) * n].iter().map(|v| v * sc).collect();
            let mx = row.iter().cloned().fold(f32::MIN, f32::max);
            let e: Vec<f32> = row.iter().map(|v| (v - mx).exp()).collect();
            let s: f32 = e.iter().sum();
            want.extend(e.iter().map(|v| v / s));
        }
        out.push(within("softmax_rows", &read_half(&o, m * n)?, &want, 2e-3)?);
    }
    // quant_rows: int8 per row
    {
        let (n, kk) = (3usize, 513usize);
        let w = data(n * kk, 7);
        let dw = DevBuf::from_f32(gpu, &w)?;
        let q = DevBuf::new(gpu, n * kk)?;
        let sc = DevBuf::f32(gpu, n)?;
        // SAFETY: sized above.
        check(unsafe { (k.quant_rows)(g, dw.fp(), n as i64, kk as i64, q.ptr().cast(), sc.fp()) }, "quant_rows")?;
        let mut qh = vec![0u8; n * kk];
        q.read(0, &mut qh)?;
        let scales = sc.to_f32()?;
        let mut worst = 0.0f32;
        for r in 0..n {
            let row = &w[r * kk..(r + 1) * kk];
            let want_s = (row.iter().fold(0.0f32, |a, v| a.max(v.abs())) / 127.0).max(1e-30);
            if (scales[r] - want_s).abs() > 1e-7 {
                return Err(Error(format!("quant_rows: row {r} scale {} != {want_s}", scales[r])));
            }
            for (i, v) in row.iter().enumerate() {
                let want = (v / want_s).round_ties_even().clamp(-127.0, 127.0);
                worst = worst.max((qh[r * kk + i] as i8 as f32 - want).abs());
            }
        }
        if worst > 1.0 {
            return Err(Error(format!("quant_rows: an int8 value is {worst} off")));
        }
        out.push(worst);
    }
    // gelu_tanh and silu, in place
    for (name, silu) in [("gelu_tanh", false), ("silu", true)] {
        let n = 999usize;
        let x = data(n, 8);
        let dx = DevBuf::from_f32(gpu, &x)?;
        let h = DevBuf::new(gpu, n * 2)?;
        // SAFETY: sized above.
        check(unsafe { (k.to_half)(g, dx.fp(), h.ptr(), n as i64) }, "to_half")?;
        if silu {
            check(unsafe { (k.silu)(g, h.ptr(), n as i64) }, name)?;
        } else {
            check(unsafe { (k.gelu_tanh)(g, h.ptr(), n as i64) }, name)?;
        }
        let want: Vec<f32> = x
            .iter()
            .map(|&v| {
                let v = half_to_f32(f32_to_half(v));
                if silu { v / (1.0 + (-v).exp()) } else { 0.5 * v * (1.0 + (0.7978845608f32 * (v + 0.044715 * v * v * v)).tanh()) }
            })
            .collect();
        out.push(within(name, &read_half(&h, n)?, &want, 2e-3)?);
    }
    // up2: nearest 2x
    {
        let (h, w, c) = (3usize, 4usize, 5usize);
        let x = data(h * w * c, 9);
        let dx = DevBuf::from_f32(gpu, &x)?;
        let hx = DevBuf::new(gpu, x.len() * 2)?;
        let o = DevBuf::new(gpu, 4 * x.len() * 2)?;
        // SAFETY: sized above.
        check(unsafe { (k.to_half)(g, dx.fp(), hx.ptr(), x.len() as i64) }, "to_half")?;
        check(unsafe { (k.up2)(g, hx.ptr(), h as i64, w as i64, c as i64, o.ptr()) }, "up2")?;
        let got = read_half(&o, 4 * x.len())?;
        let mut want = vec![0.0f32; 4 * x.len()];
        for y in 0..2 * h {
            for xx in 0..2 * w {
                for ch in 0..c {
                    want[(y * 2 * w + xx) * c + ch] = half_to_f32(f32_to_half(x[((y / 2) * w + xx / 2) * c + ch]));
                }
            }
        }
        out.push(within("up2", &got, &want, 1e-6)?);
    }
    // to_rgba8
    {
        let (h, w) = (2usize, 3usize);
        let x = data(h * w * 4, 10);
        let dx = DevBuf::from_f32(gpu, &x)?;
        let hx = DevBuf::new(gpu, x.len() * 2)?;
        let o = DevBuf::new(gpu, x.len())?;
        // SAFETY: sized above.
        check(unsafe { (k.to_half)(g, dx.fp(), hx.ptr(), x.len() as i64) }, "to_half")?;
        check(unsafe { (k.to_rgba8)(g, hx.ptr(), h as i64, w as i64, o.ptr().cast()) }, "to_rgba8")?;
        let mut got = vec![0u8; x.len()];
        o.read(0, &mut got)?;
        let mut worst = 0i32;
        for (i, v) in x.iter().enumerate() {
            let v = half_to_f32(f32_to_half(*v)).clamp(-1.0, 1.0) * 0.5 + 0.5;
            let want = (v * 255.0).round().clamp(0.0, 255.0) as i32;
            worst = worst.max((got[i] as i32 - want).abs());
        }
        if worst > 1 {
            return Err(Error(format!("to_rgba8: a byte is {worst} off")));
        }
        out.push(worst as f32);
    }
    // pad_br
    {
        let (h, w, c) = (2usize, 3usize, 2usize);
        let x = data(h * w * c, 11);
        let dx = DevBuf::from_f32(gpu, &x)?;
        let hx = DevBuf::new(gpu, x.len() * 2)?;
        let o = DevBuf::new(gpu, (h + 1) * (w + 1) * c * 2)?;
        // SAFETY: sized above.
        check(unsafe { (k.to_half)(g, dx.fp(), hx.ptr(), x.len() as i64) }, "to_half")?;
        check(unsafe { (k.pad_br)(g, hx.ptr(), h as i64, w as i64, c as i64, o.ptr()) }, "pad_br")?;
        let got = read_half(&o, (h + 1) * (w + 1) * c)?;
        let mut want = vec![0.0f32; (h + 1) * (w + 1) * c];
        for y in 0..h {
            for xx in 0..w {
                for ch in 0..c {
                    want[(y * (w + 1) + xx) * c + ch] = half_to_f32(f32_to_half(x[(y * w + xx) * c + ch]));
                }
            }
        }
        out.push(within("pad_br", &got, &want, 1e-6)?);
    }
    Ok(out)
}

/// float32 to the nearest half (ties to even), as the GPU's conversion does
fn f32_to_half(f: f32) -> u16 {
    let b = f.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7f_ffff;
    if exp >= 31 {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let m = (man | 0x80_0000) >> (1 - exp);
        let r = (m + 0xfff + ((m >> 13) & 1)) >> 13;
        return sign | r as u16;
    }
    let r = ((exp as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    let up = rem > 0x1000 || (rem == 0x1000 && (r & 1) == 1);
    sign | (r + up as u32) as u16
}
