//! The shared kernels (`nsd.cu`) against host arithmetic on the GPU: each operation on small inputs, compared with
//! the same computation here (the quantized types' dequantization written out the way ggml's reference does, a block
//! at a time, so a slip in the kernel's per-element indexing shows). `nextcuda image selftest` runs it.

use std::sync::Arc;

use nextcuda_core::{DevBuf, Error, Gpu, Result};

use crate::kernels::{none, Dt, Nsd};

fn f16_to_f32(h: u16) -> f32 {
    let (s, e, m) = ((h >> 15) & 1, (h >> 10) & 0x1f, h & 0x3ff);
    let v = match e {
        0 => (m as f32) * 2f32.powi(-24),
        31 => if m == 0 { f32::INFINITY } else { f32::NAN },
        _ => (1.0 + m as f32 / 1024.0) * 2f32.powi(e as i32 - 15),
    };
    if s == 1 { -v } else { v }
}

fn f32_to_f16(f: f32) -> u16 {
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
        return sign | ((m + 0xfff + ((m >> 13) & 1)) >> 13) as u16;
    }
    let r = ((exp as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    sign | (r + (rem > 0x1000 || (rem == 0x1000 && (r & 1) == 1)) as u32) as u16
}

/// A value as the GPU holds it in half
fn h(v: f32) -> f32 {
    f16_to_f32(f32_to_f16(v))
}

fn data(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(12345);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (((s >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

fn bytes(seed: u32, n: usize) -> Vec<u8> {
    let mut s = seed.wrapping_mul(2246822519).wrapping_add(3266489917);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 16) as u8
        })
        .collect()
}

fn put_f32(g: &Arc<Gpu>, v: &[f32]) -> Result<DevBuf> {
    DevBuf::from_f32(g, v)
}

fn put_half(g: &Arc<Gpu>, v: &[f32]) -> Result<DevBuf> {
    let b = DevBuf::new(g, v.len() * 2)?;
    b.write(0, &v.iter().flat_map(|x| f32_to_f16(*x).to_le_bytes()).collect::<Vec<u8>>())?;
    Ok(b)
}

fn get_half(b: &DevBuf, n: usize) -> Result<Vec<f32>> {
    let mut raw = vec![0u8; n * 2];
    b.read(0, &mut raw)?;
    Ok(raw.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect())
}

fn within(what: &str, got: &[f32], want: &[f32], tol: f32) -> Result<f32> {
    let mut worst = 0.0f32;
    for (g, w) in got.iter().zip(want) {
        worst = worst.max((g - w).abs() / w.abs().max(1.0));
    }
    if got.len() != want.len() || !(worst <= tol) {
        return Err(Error(format!("{what}: worst relative error {worst:.2e} over {} values (limit {tol:.0e})", want.len())));
    }
    Ok(worst)
}

// ---- GGUF blocks, dequantized as ggml's reference does ---------------------------------------------------------------

fn scale_min(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

fn dq_q8_0(b: &[u8]) -> Vec<f32> {
    let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
    (0..32).map(|i| d * b[2 + i] as i8 as f32).collect()
}

fn dq_q4k(b: &[u8]) -> Vec<f32> {
    let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
    let min = f16_to_f32(u16::from_le_bytes([b[2], b[3]]));
    let (sc, qs) = (&b[4..16], &b[16..144]);
    let mut y = Vec::with_capacity(256);
    let (mut is, mut q) = (0usize, 0usize);
    for _ in 0..4 {
        let (s1, m1) = scale_min(is, sc);
        let (s2, m2) = scale_min(is + 1, sc);
        let (d1, mm1, d2, mm2) = (d * s1 as f32, min * m1 as f32, d * s2 as f32, min * m2 as f32);
        for l in 0..32 {
            y.push(d1 * (qs[q + l] & 0xF) as f32 - mm1);
        }
        for l in 0..32 {
            y.push(d2 * (qs[q + l] >> 4) as f32 - mm2);
        }
        q += 32;
        is += 2;
    }
    y
}

fn dq_q5k(b: &[u8]) -> Vec<f32> {
    let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
    let min = f16_to_f32(u16::from_le_bytes([b[2], b[3]]));
    let (sc, qh, qs) = (&b[4..16], &b[16..48], &b[48..176]);
    let mut y = Vec::with_capacity(256);
    let (mut is, mut q) = (0usize, 0usize);
    let (mut u1, mut u2) = (1u8, 2u8);
    for _ in 0..4 {
        let (s1, m1) = scale_min(is, sc);
        let (s2, m2) = scale_min(is + 1, sc);
        let (d1, mm1, d2, mm2) = (d * s1 as f32, min * m1 as f32, d * s2 as f32, min * m2 as f32);
        for l in 0..32 {
            y.push(d1 * ((qs[q + l] & 0xF) + if qh[l] & u1 != 0 { 16 } else { 0 }) as f32 - mm1);
        }
        for l in 0..32 {
            y.push(d2 * ((qs[q + l] >> 4) + if qh[l] & u2 != 0 { 16 } else { 0 }) as f32 - mm2);
        }
        q += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
    y
}

fn dq_q6k(b: &[u8]) -> Vec<f32> {
    let d = f16_to_f32(u16::from_le_bytes([b[208], b[209]]));
    let (mut ql, mut qh, mut sc) = (0usize, 128usize, 192usize);
    let mut y = vec![0f32; 256];
    for n in (0..256).step_by(128) {
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((b[ql + l] & 0xF) | ((b[qh + l] & 3) << 4)) as i32 - 32;
            let q2 = ((b[ql + l + 32] & 0xF) | (((b[qh + l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((b[ql + l] >> 4) | (((b[qh + l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((b[ql + l + 32] >> 4) | (((b[qh + l] >> 6) & 3) << 4)) as i32 - 32;
            let s = |k: usize| b[sc + is + k] as i8 as f32;
            y[n + l] = d * s(0) * q1 as f32;
            y[n + l + 32] = d * s(2) * q2 as f32;
            y[n + l + 64] = d * s(4) * q3 as f32;
            y[n + l + 96] = d * s(6) * q4 as f32;
        }
        ql += 64;
        qh += 32;
        sc += 8;
    }
    y
}

/// `blocks` random blocks of a GGUF type: their bytes (with sane half scales) and their values
fn blocks(qtype: u32, blocks: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
    let (size, f): (usize, fn(&[u8]) -> Vec<f32>) = match qtype {
        8 => (34, dq_q8_0),
        12 => (144, dq_q4k),
        13 => (176, dq_q5k),
        14 => (210, dq_q6k),
        _ => unreachable!(),
    };
    let mut raw = bytes(seed, blocks * size);
    for b in raw.chunks_exact_mut(size) {
        let put = |b: &mut [u8], at: usize, v: f32| b[at..at + 2].copy_from_slice(&f32_to_f16(v).to_le_bytes());
        match qtype {
            8 => put(b, 0, 0.004),
            12 | 13 => {
                put(b, 0, 0.003);
                put(b, 2, 0.002);
            }
            _ => put(b, 208, 0.0015),
        }
    }
    let vals = raw.chunks_exact(size).flat_map(f).collect();
    (raw, vals)
}

/// Every operation of nsd.h once; the worst relative error of each, in order
pub fn selftest(gpu: &Arc<Gpu>) -> Result<Vec<f32>> {
    let nsd = Nsd::new(gpu)?;
    let mut out = Vec::new();
    let g = gpu;

    // linear, float32 and half
    {
        let (m, k, n) = (7usize, 70usize, 19usize);
        let (x, w, bias) = (data(m * k, 1, 1.0), data(n * k, 2, 0.5), data(n, 3, 1.0));
        let mut want = vec![0f32; m * n];
        for r in 0..m {
            for c in 0..n {
                want[r * n + c] = bias[c] + (0..k).map(|i| x[r * k + i] * w[c * k + i]).sum::<f32>();
            }
        }
        let (dx, dw, db, o) = (put_f32(g, &x)?, put_f32(g, &w)?, put_f32(g, &bias)?, DevBuf::f32(g, m * n)?);
        nsd.linear(dx.ptr(), Dt::F32, m, k, dw.ptr(), n, db.ptr(), o.ptr(), Dt::F32)?;
        nsd.wait()?;
        out.push(within("linear f32", &o.to_f32()?, &want, 1e-5)?);
        let (xh, wh) = (put_half(g, &x)?, put_half(g, &w)?);
        let (xq, wq): (Vec<f32>, Vec<f32>) = (x.iter().map(|v| h(*v)).collect(), w.iter().map(|v| h(*v)).collect());
        let mut want2 = vec![0f32; m * n];
        for r in 0..m {
            for c in 0..n {
                want2[r * n + c] = bias[c] + (0..k).map(|i| xq[r * k + i] * wq[c * k + i]).sum::<f32>();
            }
        }
        let o2 = DevBuf::f32(g, m * n)?;
        nsd.linear(xh.ptr(), Dt::F16, m, k, wh.ptr(), n, db.ptr(), o2.ptr(), Dt::F32)?;
        nsd.wait()?;
        out.push(within("linear half", &o2.to_f32()?, &want2, 1e-4)?);
    }
    // int8 ConvRot: weights prepared from a float matrix (rotated, quantized per row), activations rotated and
    // quantized on the fly, against the plain float product (error is the two int8 roundings: about a percent)
    for group in [0usize, 64, 256] {
        let (m, k, n) = (37usize, 512usize, 48usize);
        let (x, w) = (data(m * k, 60, 1.0), data(n * k, 61, 0.5));
        let xq: Vec<f32> = x.iter().map(|v| h(*v)).collect();
        let mut want = vec![0f32; m * n];
        for r in 0..m {
            for c in 0..n {
                want[r * n + c] = (0..k).map(|i| xq[r * k + i] * w[c * k + i]).sum::<f32>();
            }
        }
        let (dw, dx) = (put_f32(g, &w)?, put_half(g, &x)?);
        let (q, sc, o) = (DevBuf::new(g, n * k)?, DevBuf::f32(g, n)?, DevBuf::new(g, m * n * 2)?);
        nsd.int8_prepare(dw.ptr(), 0, 0, n, k, group, q.ptr(), sc.ptr())?;
        nsd.int8_linear(dx.ptr(), Dt::F16, m, k, q.ptr(), n, sc.ptr(), none(), o.ptr(), Dt::F16, group, 0)?;
        nsd.wait()?;
        let got = get_half(&o, m * n)?;
        // relative to the product's size (the values are of order sqrt(k) x the inputs' spread)
        let scale = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1.0);
        let worst = got.iter().zip(&want).fold(0f32, |a, (g, w)| a.max((g - w).abs())) / scale;
        if !(worst < 0.03) {
            return Err(Error(format!("int8 linear (group {group}): worst error {worst:.2e} of the largest value, limit 3e-2")));
        }
        out.push(worst);
    }
    // dequant: each GGUF type
    for (name, qt) in [("Q8_0", 8u32), ("Q4_K", 12), ("Q5_K", 13), ("Q6_K", 14)] {
        let (raw, want) = blocks(qt, 6, 10 + qt);
        let src = DevBuf::new(g, raw.len())?;
        src.write(0, &raw)?;
        let o = DevBuf::f32(g, want.len())?;
        nsd.dequant(src.ptr(), qt, want.len(), o.ptr(), Dt::F32)?;
        nsd.wait()?;
        out.push(within(&format!("dequant {name}"), &o.to_f32()?, &want, 1e-6)?);
    }
    // the quantized linear: W in Q6_K (K a multiple of 256, N rows), half activations, in slices of rows
    for (name, qt, k) in [("Q8_0", 8u32, 64usize), ("Q4_K", 12, 512), ("Q6_K", 14, 256)] {
        let (m, n) = (5usize, 37usize);
        let per = if qt == 8 { 32 } else { 256 };
        let (raw, w) = blocks(qt, n * k / per, 40 + qt);
        let x = data(m * k, 5, 1.0);
        let xq: Vec<f32> = x.iter().map(|v| h(*v)).collect();
        let wq: Vec<f32> = w.iter().map(|v| h(*v)).collect();
        let mut want = vec![0f32; m * n];
        for r in 0..m {
            for c in 0..n {
                want[r * n + c] = (0..k).map(|i| xq[r * k + i] * wq[c * k + i]).sum::<f32>();
            }
        }
        let src = DevBuf::new(g, raw.len())?;
        src.write(0, &raw)?;
        let (dx, o) = (put_half(g, &x)?, DevBuf::f32(g, m * n)?);
        nsd.linear_q(dx.ptr(), Dt::F16, m, k, src.ptr(), qt, n, none(), o.ptr(), Dt::F32, 0)?;
        nsd.wait()?;
        out.push(within(&format!("linear {name}"), &o.to_f32()?, &want, 2e-3)?);
    }
    // rms norm + modulation
    {
        let (m, c, eps) = (4usize, 130usize, 1e-6f32);
        let (x, w) = (data(m * c, 6, 2.0), data(c, 7, 1.0));
        let (rows, scale, shift) = ([1i32, 0, 1, 0], data(2 * c, 8, 0.5), data(2 * c, 9, 0.5));
        let mut want = vec![0f32; m * c];
        for r in 0..m {
            let ms = x[r * c..(r + 1) * c].iter().map(|v| v * v).sum::<f32>() / c as f32;
            let inv = 1.0 / (ms + eps).sqrt();
            for i in 0..c {
                let n = x[r * c + i] * inv * w[i];
                let rr = rows[r] as usize;
                want[r * c + i] = n * (1.0 + scale[rr * c + i]) + shift[rr * c + i];
            }
        }
        let (dx, dw, ds, dh) = (put_f32(g, &x)?, put_f32(g, &w)?, put_f32(g, &scale)?, put_f32(g, &shift)?);
        let drows = DevBuf::new(g, m * 4)?;
        drows.write(0, &rows.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        let o = DevBuf::f32(g, m * c)?;
        nsd.rms_norm_mod(dx.ptr(), Dt::F32, m, c, dw.ptr(), eps, drows.ptr(), ds.ptr(), dh.ptr(), o.ptr(), Dt::F32)?;
        nsd.wait()?;
        out.push(within("rms norm + modulation", &o.to_f32()?, &want, 1e-5)?);
    }
    // per-head rms norm + rope, rows inside a wider buffer
    {
        let (m, hh, d, eps) = (3usize, 2usize, 16usize, 1e-6f32);
        let stride = hh * d + 8;
        let (x, w) = (data(m * stride, 11, 1.5), data(d, 12, 1.0));
        let cs: Vec<f32> = (0..m * d / 2).flat_map(|i| [(0.1 * i as f32).cos(), (0.1 * i as f32).sin()]).collect();
        let mut want = x.clone();
        for r in 0..m {
            for hd in 0..hh {
                let base = r * stride + hd * d;
                let ms = x[base..base + d].iter().map(|v| v * v).sum::<f32>() / d as f32;
                let inv = 1.0 / (ms + eps).sqrt();
                let nrm: Vec<f32> = (0..d).map(|i| x[base + i] * inv * w[i]).collect();
                for i in 0..d / 2 {
                    let (c, s) = (cs[(r * d / 2 + i) * 2], cs[(r * d / 2 + i) * 2 + 1]);
                    want[base + i] = nrm[i] * c - nrm[d / 2 + i] * s;
                    want[base + d / 2 + i] = nrm[d / 2 + i] * c + nrm[i] * s;
                }
            }
        }
        let (dx, dw, dc) = (put_f32(g, &x)?, put_f32(g, &w)?, put_f32(g, &cs)?);
        nsd.rms_rope(dx.ptr(), Dt::F32, m, hh, d, stride, dw.ptr(), eps, dc.ptr(), d)?;
        nsd.wait()?;
        out.push(within("rms norm + rope", &dx.to_f32()?, &want, 1e-5)?);
    }
    // layer norm, gelu (both forms), rope alone
    {
        let (m, c, eps) = (5usize, 70usize, 1e-6f32);
        let (x, w, b) = (data(m * c, 50, 2.0), data(c, 51, 1.0), data(c, 52, 1.0));
        let mut want = vec![0f32; m * c];
        for r in 0..m {
            let row = &x[r * c..(r + 1) * c];
            let mean = row.iter().sum::<f32>() / c as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / c as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for i in 0..c {
                want[r * c + i] = (row[i] - mean) * inv * w[i] + b[i];
            }
        }
        let (dx, dw, db, o) = (put_f32(g, &x)?, put_f32(g, &w)?, put_f32(g, &b)?, DevBuf::f32(g, m * c)?);
        nsd.layer_norm(dx.ptr(), Dt::F32, m, c, dw.ptr(), db.ptr(), eps, o.ptr(), Dt::F32)?;
        nsd.wait()?;
        out.push(within("layer norm", &o.to_f32()?, &want, 1e-5)?);
        let g_in = data(500, 53, 3.0);
        for (name, tanh) in [("gelu (erf)", false), ("gelu (tanh)", true)] {
            let want: Vec<f32> = g_in.iter().map(|&v| if tanh { 0.5 * v * (1.0 + (0.7978845608f32 * (v + 0.044715 * v * v * v)).tanh()) } else {
                // erf via the Abramowitz-Stegun 7.1.26 series is too coarse for 1e-5; use the libm-free A&S 7.1.28 identity through tanh-free f64 series
                let z = v as f64 / std::f64::consts::SQRT_2;
                let t = 1.0 / (1.0 + 0.3275911 * z.abs());
                let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-z * z).exp();
                (0.5 * v as f64 * (1.0 + y.copysign(z))) as f32
            }).collect();
            let dg = put_f32(g, &g_in)?;
            nsd.gelu(dg.ptr(), Dt::F32, g_in.len(), tanh)?;
            nsd.wait()?;
            out.push(within(name, &dg.to_f32()?, &want, if tanh { 1e-5 } else { 2e-3 })?);
        }
        // rope alone
        let (rm, hh, d) = (3usize, 2usize, 16usize);
        let stride = hh * d + 8;
        let xr = data(rm * stride, 54, 1.5);
        let cs: Vec<f32> = (0..rm * d / 2).flat_map(|i| [(0.1 * i as f32).cos(), (0.1 * i as f32).sin()]).collect();
        let mut want = xr.clone();
        for r in 0..rm {
            for hd in 0..hh {
                let base = r * stride + hd * d;
                for i in 0..d / 2 {
                    let (c, s) = (cs[(r * d / 2 + i) * 2], cs[(r * d / 2 + i) * 2 + 1]);
                    want[base + i] = xr[base + i] * c - xr[base + d / 2 + i] * s;
                    want[base + d / 2 + i] = xr[base + d / 2 + i] * c + xr[base + i] * s;
                }
            }
        }
        let (dx, dc) = (put_f32(g, &xr)?, put_f32(g, &cs)?);
        nsd.rope(dx.ptr(), Dt::F32, rm, hh, d, stride, dc.ptr(), d)?;
        nsd.wait()?;
        out.push(within("rope", &dx.to_f32()?, &want, 1e-5)?);
    }
    // swiglu and gated add
    {
        let (m, c) = (3usize, 50usize);
        let x = data(m * 2 * c, 13, 2.0);
        let want: Vec<f32> = (0..m * c).map(|i| {
            let (r, j) = (i / c, i % c);
            let gv = x[r * 2 * c + j];
            gv / (1.0 + (-gv).exp()) * x[r * 2 * c + c + j]
        }).collect();
        let (dx, o) = (put_f32(g, &x)?, DevBuf::f32(g, m * c)?);
        nsd.swiglu(dx.ptr(), Dt::F32, m, c, o.ptr(), Dt::F32)?;
        nsd.wait()?;
        out.push(within("swiglu", &o.to_f32()?, &want, 1e-5)?);
        let (base, other, gate) = (data(m * c, 14, 1.0), data(m * c, 15, 1.0), data(2 * c, 16, 1.0));
        let rows = [1i32, 0, 1];
        let want2: Vec<f32> = (0..m * c).map(|i| base[i] + other[i] * gate[rows[i / c] as usize * c + i % c]).collect();
        let (db, dot, dg) = (put_f32(g, &base)?, put_f32(g, &other)?, put_f32(g, &gate)?);
        let dr = DevBuf::new(g, m * 4)?;
        dr.write(0, &rows.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        nsd.gate_add(db.ptr(), Dt::F32, m, c, dot.ptr(), Dt::F32, dr.ptr(), dg.ptr())?;
        nsd.wait()?;
        out.push(within("gated add", &db.to_f32()?, &want2, 1e-6)?);
    }
    // attention: grouped heads, causal, and rows of a longer sequence, and unmasked q over k/v
    {
        let (l, hq, hkv, d) = (9usize, 4usize, 2usize, 16usize);
        let (qs, kvs) = (hq * d, hkv * d);
        let (q, k, v) = (data(l * qs, 17, 1.0), data(l * kvs, 18, 1.0), data(l * kvs, 19, 1.0));
        let host = |q0: usize, rows: usize, skv: usize, causal: bool| -> Vec<f32> {
            let mut o = vec![0f32; rows * hq * d];
            for r in 0..rows {
                for hd in 0..hq {
                    let kv = hd / (hq / hkv);
                    let lim = if causal { q0 + r } else { skv - 1 };
                    let sc: Vec<f32> = (0..=lim).map(|j| (0..d).map(|i| q[(q0 + r) * qs + hd * d + i] * k[j * kvs + kv * d + i]).sum::<f32>() / (d as f32).sqrt()).collect();
                    let mx = sc.iter().cloned().fold(f32::MIN, f32::max);
                    let e: Vec<f32> = sc.iter().map(|s| (s - mx).exp()).collect();
                    let z: f32 = e.iter().sum();
                    for i in 0..d {
                        o[r * hq * d + hd * d + i] = (0..=lim).map(|j| e[j] / z * v[j * kvs + kv * d + i]).sum();
                    }
                }
            }
            o
        };
        let (dq, dk, dv) = (put_half(g, &q)?, put_half(g, &k)?, put_half(g, &v)?);
        // the host reads the values as the GPU holds them
        let (q, k, v) = (q.iter().map(|x| h(*x)).collect::<Vec<_>>(), k.iter().map(|x| h(*x)).collect::<Vec<_>>(), v.iter().map(|x| h(*x)).collect::<Vec<_>>());
        let _ = (&q, &k, &v);
        let hostq = |q0: usize, rows: usize, skv: usize, causal: bool| -> Vec<f32> {
            let mut o = vec![0f32; rows * hq * d];
            for r in 0..rows {
                for hd in 0..hq {
                    let kv = hd / (hq / hkv);
                    let lim = if causal { q0 + r } else { skv - 1 };
                    let sc: Vec<f32> = (0..=lim).map(|j| (0..d).map(|i| q[(q0 + r) * qs + hd * d + i] * k[j * kvs + kv * d + i]).sum::<f32>() / (d as f32).sqrt()).collect();
                    let mx = sc.iter().cloned().fold(f32::MIN, f32::max);
                    let e: Vec<f32> = sc.iter().map(|s| (s - mx).exp()).collect();
                    let z: f32 = e.iter().sum();
                    for i in 0..d {
                        o[r * hq * d + hd * d + i] = (0..=lim).map(|j| e[j] / z * v[j * kvs + kv * d + i]).sum();
                    }
                }
            }
            o
        };
        let _ = host;
        let o = DevBuf::new(g, l * hq * d * 2)?;
        nsd.attention_causal(dq.ptr(), dk.ptr(), dv.ptr(), Dt::F16, l, hq, hkv, d, qs, kvs, o.ptr())?;
        nsd.wait()?;
        out.push(within("attention (causal, grouped heads)", &get_half(&o, l * hq * d)?, &hostq(0, l, l, true), 2e-3)?);
        let o2 = DevBuf::new(g, 4 * hq * d * 2)?;
        nsd.attention_causal_rows(dq.ptr(), dk.ptr(), dv.ptr(), Dt::F16, 5, 4, hq, hkv, d, qs, kvs, o2.ptr())?;
        nsd.wait()?;
        out.push(within("attention (rows 5..9)", &get_half(&o2, 4 * hq * d)?, &hostq(5, 4, l, true), 2e-3)?);
        // unmasked: equal heads (the DiT's), q rows 0..3 over all 9 keys
        let hh = hkv;
        let o3 = DevBuf::new(g, 3 * hh * d * 2)?;
        let q_eq: Vec<f32> = (0..l * kvs).map(|i| q[(i / kvs) * qs + i % kvs]).collect();
        let dq2 = put_half(g, &q_eq)?;
        nsd.attention_qk(dq2.ptr(), 3, kvs, dk.ptr(), dv.ptr(), l, kvs, Dt::F16, hh, d, o3.ptr(), Dt::F16)?;
        nsd.wait()?;
        let mut want = vec![0f32; 3 * hh * d];
        for r in 0..3 {
            for hd in 0..hh {
                let sc: Vec<f32> = (0..l).map(|j| (0..d).map(|i| q_eq[r * kvs + hd * d + i] * k[j * kvs + hd * d + i]).sum::<f32>() / (d as f32).sqrt()).collect();
                let mx = sc.iter().cloned().fold(f32::MIN, f32::max);
                let e: Vec<f32> = sc.iter().map(|s| (s - mx).exp()).collect();
                let z: f32 = e.iter().sum();
                for i in 0..d {
                    want[r * hh * d + hd * d + i] = (0..l).map(|j| e[j] / z * v[j * kvs + hd * d + i]).sum();
                }
            }
        }
        out.push(within("attention (no mask)", &get_half(&o3, 3 * hh * d)?, &want, 2e-3)?);
    }
    // conv2d 3x3 (zero padded) and 1x1, channels last
    {
        let (hh, ww, ci, co) = (5usize, 6usize, 7usize, 9usize);
        let k = 3usize;
        let (x, w, b) = (data(hh * ww * ci, 20, 1.0), data(co * ci * k * k, 21, 0.4), data(co, 22, 0.3));
        let (xq, wq): (Vec<f32>, Vec<f32>) = (x.iter().map(|v| h(*v)).collect(), w.iter().map(|v| h(*v)).collect());
        let mut want = vec![0f32; hh * ww * co];
        for y in 0..hh {
            for xx in 0..ww {
                for o in 0..co {
                    let mut acc = b[o];
                    for ky in 0..k {
                        for kx in 0..k {
                            let (yy, x2) = (y as i64 + ky as i64 - 1, xx as i64 + kx as i64 - 1);
                            if yy < 0 || x2 < 0 || yy >= hh as i64 || x2 >= ww as i64 {
                                continue;
                            }
                            for c in 0..ci {
                                acc += xq[((yy as usize) * ww + x2 as usize) * ci + c] * wq[((o * ci + c) * k + ky) * k + kx];
                            }
                        }
                    }
                    want[(y * ww + xx) * co + o] = acc;
                }
            }
        }
        let (dx, dw, db, o) = (put_half(g, &x)?, put_half(g, &w)?, put_f32(g, &b)?, DevBuf::new(g, hh * ww * co * 2)?);
        nsd.conv2d(dx.ptr(), Dt::F16, 1, hh, ww, ci, dw.ptr(), co, k, db.ptr(), o.ptr())?;
        nsd.wait()?;
        out.push(within("conv2d 3x3", &get_half(&o, hh * ww * co)?, &want, 3e-3)?);
        // strided, no padding (the encoder's down-convolution): T = 1, kernel 1 x 3 x 3, stride 1 x 2 x 2
        let (h2, w2) = (7usize, 7usize);
        let x3 = data(h2 * w2 * ci, 23, 1.0);
        let x3q: Vec<f32> = x3.iter().map(|v| h(*v)).collect();
        let (ho, wo) = ((h2 - 3) / 2 + 1, (w2 - 3) / 2 + 1);
        let mut want3 = vec![0f32; ho * wo * co];
        for y in 0..ho {
            for xx in 0..wo {
                for o in 0..co {
                    let mut acc = b[o];
                    for ky in 0..3 {
                        for kx in 0..3 {
                            for c in 0..ci {
                                acc += x3q[((y * 2 + ky) * w2 + xx * 2 + kx) * ci + c] * wq[((o * ci + c) * 3 + ky) * 3 + kx];
                            }
                        }
                    }
                    want3[(y * wo + xx) * co + o] = acc;
                }
            }
        }
        let dx3 = put_half(g, &x3)?;
        let o3 = DevBuf::new(g, ho * wo * co * 2)?;
        nsd.conv3d_ex(dx3.ptr(), Dt::F16, (1, h2, w2), ci, dw.ptr(), co, (1, 3, 3), (1, 2, 2), db.ptr(), o3.ptr())?;
        nsd.wait()?;
        out.push(within("conv3d (strided)", &get_half(&o3, ho * wo * co)?, &want3, 3e-3)?);
    }
    // the 1-D operations (float32): a strided, dilated convolution, a transposed one, snake, scale
    {
        let (b, ci, l, co, k) = (2usize, 5usize, 23usize, 4usize, 3usize);
        let (x, w, bias) = (data(b * ci * l, 31, 1.0), data(co * ci * k, 32, 0.5), data(co, 33, 1.0));
        let (stride, dil, pad) = (2usize, 2usize, 2usize);
        let lo = (l + 2 * pad - dil * (k - 1) - 1) / stride + 1;
        let mut want = vec![0f32; b * co * lo];
        for bb in 0..b {
            for o in 0..co {
                for t in 0..lo {
                    let mut a = bias[o];
                    for c in 0..ci {
                        for kk in 0..k {
                            let s = (t * stride + kk * dil) as i64 - pad as i64;
                            if s >= 0 && (s as usize) < l {
                                a += w[(o * ci + c) * k + kk] * x[(bb * ci + c) * l + s as usize];
                            }
                        }
                    }
                    want[(bb * co + o) * lo + t] = a;
                }
            }
        }
        let (dx, dw, db) = (put_f32(g, &x)?, put_f32(g, &w)?, put_f32(g, &bias)?);
        let o = DevBuf::f32(g, b * co * lo)?;
        nsd.conv1d(dx.ptr(), b, ci, l, dw.ptr(), co, k, db.ptr(), stride, dil, pad, o.ptr(), lo)?;
        nsd.wait()?;
        out.push(within("conv1d", &o.to_f32()?, &want, 1e-5)?);
        // transposed: w [Ci, Co, K]
        let (st, pd, kt) = (3usize, 1usize, 5usize);
        let wt = data(ci * co * kt, 34, 0.5);
        let lt = (l - 1) * st + kt - 2 * pd;
        let mut want2 = vec![0f32; b * co * lt];
        for bb in 0..b {
            for o in 0..co {
                for t in 0..lt {
                    want2[(bb * co + o) * lt + t] = bias[o];
                }
            }
            for c in 0..ci {
                for s in 0..l {
                    for o in 0..co {
                        for kk in 0..kt {
                            let t = (s * st + kk) as i64 - pd as i64;
                            if t >= 0 && (t as usize) < lt {
                                want2[(bb * co + o) * lt + t as usize] += x[(bb * ci + c) * l + s] * wt[(c * co + o) * kt + kk];
                            }
                        }
                    }
                }
            }
        }
        let dwt = put_f32(g, &wt)?;
        let o2 = DevBuf::f32(g, b * co * lt)?;
        nsd.conv_transpose1d(dx.ptr(), b, ci, l, dwt.ptr(), co, kt, db.ptr(), st, pd, o2.ptr(), lt)?;
        nsd.wait()?;
        out.push(within("conv_transpose1d", &o2.to_f32()?, &want2, 1e-5)?);
        let alpha: Vec<f32> = data(ci, 35, 1.0).iter().map(|a| a.abs() + 0.5).collect();
        let want3: Vec<f32> = x.iter().enumerate().map(|(i, v)| {
            let a = alpha[(i / l) % ci];
            v + (a * v).sin().powi(2) / (a + 1e-9)
        }).collect();
        let (da, o3) = (put_f32(g, &alpha)?, DevBuf::f32(g, x.len())?);
        nsd.snake(dx.ptr(), b, ci, l, da.ptr(), o3.ptr())?;
        nsd.scale(o3.ptr(), x.len(), 0.5)?;
        nsd.wait()?;
        let want3: Vec<f32> = want3.iter().map(|v| v * 0.5).collect();
        out.push(within("snake, scale", &o3.to_f32()?, &want3, 1e-5)?);
    }
    // unmasked attention, half in, float32 out; and two sequences in one call
    {
        let (bs, s, hh, d) = (2usize, 9usize, 3usize, 16usize);
        let n = bs * s * hh * d;
        let (q, k, v) = (data(n, 41, 1.0), data(n, 42, 1.0), data(n, 43, 1.0));
        let (qq, kq, vq): (Vec<f32>, Vec<f32>, Vec<f32>) = (q.iter().map(|x| h(*x)).collect(), k.iter().map(|x| h(*x)).collect(), v.iter().map(|x| h(*x)).collect());
        let mut want = vec![0f32; n];
        for b in 0..bs {
            for i in 0..s {
                for head in 0..hh {
                    let row = |j: usize| ((b * s + j) * hh + head) * d;
                    let qi = row(i);
                    let sc: Vec<f32> = (0..s).map(|j| (0..d).map(|e| qq[qi + e] * kq[row(j) + e]).sum::<f32>() / (d as f32).sqrt()).collect();
                    let mx = sc.iter().cloned().fold(f32::MIN, f32::max);
                    let p: Vec<f32> = sc.iter().map(|x| (x - mx).exp()).collect();
                    let z: f32 = p.iter().sum();
                    for e in 0..d {
                        want[qi + e] = (0..s).map(|j| p[j] / z * vq[row(j) + e]).sum();
                    }
                }
            }
        }
        let (dq, dk, dv) = (put_half(g, &q)?, put_half(g, &k)?, put_half(g, &v)?);
        let o = DevBuf::f32(g, n)?;
        nsd.attention_batch(dq.ptr(), dk.ptr(), dv.ptr(), Dt::F16, bs, s, hh, d, hh * d, o.ptr(), Dt::F32)?;
        nsd.wait()?;
        out.push(within("attention (batch)", &o.to_f32()?, &want, 3e-3)?);
        let o1 = DevBuf::f32(g, s * hh * d)?;
        nsd.attention(dq.ptr(), dk.ptr(), dv.ptr(), Dt::F16, s, hh, d, hh * d, o1.ptr(), Dt::F32)?;
        nsd.wait()?;
        out.push(within("attention", &o1.to_f32()?, &want[..s * hh * d], 3e-3)?);
    }
    Ok(out)
}
