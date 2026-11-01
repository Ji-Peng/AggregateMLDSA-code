//! Integer CRT finishing, sparse negacyclic products, and H-u digits.
use super::{FinishError, FinishOut, FinishRow, FinishRow3P, FinishRowExact, HuCols};
use crate::ntt::scalar::mont;
use crate::{
    params::*,
    poly::{Challenge, Poly32},
};

pub(crate) fn sparse_ct<const K: usize, const TAU: usize>(
    c: &Challenge<TAU>,
    t1: &[[u16; 256]; K],
    shift: u32,
    out: &mut [[i32; 256]; K],
) {
    sparse_ct_slice(c, t1, shift, out);
}

pub(crate) fn sparse_ct_slice<const TAU: usize>(
    c: &Challenge<TAU>,
    t1: &[[u16; 256]],
    shift: u32,
    out: &mut [[i32; 256]],
) {
    for (t, dest) in t1.iter().zip(out.iter_mut()) {
        let mut window = [0i32; 512];
        for i in 0..256 {
            window[i] = -(t[i] as i32);
            window[i + 256] = t[i] as i32;
        }
        dest.fill(0);
        for (k, &pos) in c.pos.iter().enumerate() {
            let sign = 1 - 2 * ((c.neg >> k) & 1) as i32;
            let src = &window[256 - pos as usize..][..256];
            for (x, &v) in dest.iter_mut().zip(src) {
                *x += sign * v;
            }
        }
        for x in dest {
            *x <<= shift;
        }
    }
}

#[inline]
fn freeze(x: i32) -> i32 {
    // |x| < 2^28.4; rounded reduction leaves (-q,q).
    let t = (x + (1 << 22)) >> 23;
    let r = x - t * Q;
    r + ((r >> 31) & Q)
}

#[inline]
fn use_hint_88(r: i32, h: u8) -> i32 {
    let mut hi = (r + 127) >> 7;
    hi = (hi * 11275 + (1 << 23)) >> 24;
    hi ^= ((43 - hi) >> 31) & hi;
    let alpha = 2 * 95_232;
    let mut lo = r - hi * alpha;
    lo -= (((Q - 1) / 2 - lo) >> 31) & Q;
    if h != 0 {
        if lo > 0 {
            if hi == 43 {
                0
            } else {
                hi + 1
            }
        } else if hi == 0 {
            43
        } else {
            hi - 1
        }
    } else {
        hi
    }
}

#[inline]
fn use_hint_32(r: i32, h: u8) -> i32 {
    let mut hi = (r + 127) >> 7;
    hi = (hi * 1025 + (1 << 21)) >> 22;
    hi &= 15;
    let alpha = 2 * 261_888;
    let mut lo = r - hi * alpha;
    lo -= (((Q - 1) / 2 - lo) >> 31) & Q;
    if h != 0 {
        if lo > 0 {
            (hi + 1) & 15
        } else {
            (hi - 1) & 15
        }
    } else {
        hi
    }
}

#[inline]
fn finish_row_from_k<P: ParamSet>(
    xq: &Poly32,
    k_coeffs: &[i32; 256],
    t: &[i32; 256],
    h: &[u8; 256],
    out: &mut FinishOut,
) -> Result<(), FinishError> {
    let alpha = 2 * P::GAMMA2;
    for i in 0..256 {
        let xqi = xq[i];
        let k = k_coeffs[i];
        let r = freeze(xqi - t[i]);
        let w = if P::GAMMA2 == (Q - 1) / 88 {
            use_hint_88(r, h[i])
        } else {
            use_hint_32(r, h[i])
        };
        let mut u = r - alpha * w;
        if u > (Q - 1) / 2 {
            u -= Q;
        }
        if u < -(Q - 1) / 2 {
            u += Q;
        }
        let e = xqi - t[i] - alpha * w - u;
        let ve = e.wrapping_mul(QINV32);
        if ve.unsigned_abs() > P::C_B_Q {
            return Err(FinishError::Quotient);
        }
        let v = k + ve;
        if u.unsigned_abs() > P::B_U as u32 {
            return Err(FinishError::UBound);
        }
        if v.unsigned_abs() > P::B_V as u32 {
            return Err(FinishError::VBound);
        }
        out.w1[i] = w as u8;
        out.u[i] = u;
        out.v[i] = v;
    }
    Ok(())
}

pub(crate) fn finish_row<P: ParamSet>(row: FinishRow<'_>, out: &mut FinishOut) -> Result<(), FinishError> {
    let mut k_coeffs = [0i32; 256];
    for i in 0..256 {
        let xq = row.xq[i];
        let mut k = mont((row.xp[i] - xq) as i64 * CRT_C as i64, CONSTS_P2);
        k += (k >> 31) & P2;
        if k > (P2 - 1) / 2 || (k == (P2 - 1) / 2 && xq > (Q - 1) / 2) {
            k -= P2;
        }
        k_coeffs[i] = k;
    }
    finish_row_from_k::<P>(row.xq, &k_coeffs, row.t, row.h, out)
}

pub(crate) fn finish_row_3p<P: ParamSet>(
    row: FinishRow3P<'_>,
    out: &mut FinishOut,
) -> Result<(), FinishError> {
    let mut k_coeffs = [0i32; 256];
    let p2 = P2 as i64;
    let p3 = P3 as i64;
    for i in 0..256 {
        let xq = row.xq[i];
        let mut k2 = mont((row.xp2[i] - xq) as i64 * CRT_C as i64, CONSTS_P2);
        k2 += (k2 >> 31) & P2;
        let mut k3 = mont((row.xp3[i] - xq) as i64 * CRT_C_P3 as i64, CONSTS_P3);
        k3 += (k3 >> 31) & P3;

        // Chinese Remainder Theorem combination of k mod P2 and k mod P3:
        let diff = (k3 as i64 - k2 as i64).rem_euclid(p3);
        let factor = (diff * P2_INV_MOD_P3 as i64).rem_euclid(p3);
        let mut k = k2 as i64 + p2 * factor;
        let m = p2 * p3;
        if k > m / 2 {
            k -= m;
        }
        k_coeffs[i] = k as i32;
    }
    finish_row_from_k::<P>(row.xq, &k_coeffs, row.t, row.h, out)
}

pub(crate) fn finish_row_exact<P: ParamSet>(
    row: FinishRowExact<'_>,
    out: &mut FinishOut,
) -> Result<(), FinishError> {
    let mut k_coeffs = [0i32; 256];
    let q = Q as i128;
    for i in 0..256 {
        let k = (row.az[i] - row.xq[i] as i128) / q;
        k_coeffs[i] = k as i32;
    }
    finish_row_from_k::<P>(row.xq, &k_coeffs, row.t, row.h, out)
}

pub(crate) fn hu_columns<P: ParamSet>(z: &Poly32, out: &FinishOut, cols: HuCols<'_>) {
    for col in &cols.wide {
        assert_eq!(col.len(), 256);
    }
    for col in &cols.narrow {
        assert_eq!(col.len(), 256);
    }
    if P::GAMMA2 == (Q - 1) / 88 {
        for i in 0..256 {
            let y = z[i] + GAMMA1;
            cols.wide[0][i].write((y & 65535) - 32768);
            cols.narrow[0][i].write(((y >> 16) - 2) as i8);
            let y = out.u[i] + B_U;
            let h = [63079, 126158, 189237, 252316, 315395]
                .iter()
                .filter(|&&t| y >= t)
                .count() as i32;
            cols.wide[1][i].write(y - 63079 * h - 32768);
            cols.narrow[1][i].write((h - 3) as i8);
            let y = (out.v[i] as u32).wrapping_add(1 << 31);
            cols.wide[2][i].write((y & 65535) as i32 - 32768);
            cols.wide[3][i].write((y >> 16) as i32 - 32768);
            for t in 0..6 {
                cols.narrow[t + 2][i].write(((out.w1[i] >> t) & 1) as i8);
            }
        }
    } else {
        for i in 0..256 {
            // z: relaxed digit-slack form
            let yz = (z[i] as i64) + (1 << 19);
            let a_z = i8::from(yz >= (1 << 19));
            let rest_z = yz & ((1 << 19) - 1);
            let z_mid = ((rest_z >> 16) as i8) - 4;
            let z_lo = ((rest_z & 0xFFFF) as i32) - 32768;
            cols.wide[0][i].write(z_lo);
            cols.narrow[0][i].write(z_mid);
            cols.narrow[1][i].write(a_z);

            // u: exact digit-slack form
            let yu = (out.u[i] as i64) + 523_777;
            let a_u = i8::from(yu >= 524_288);
            let rest_u = yu - 523_267 * i64::from(a_u);
            let u_mid = (((rest_u >> 16) & 7) as i8) - 4;
            let u_lo = ((rest_u & 0xFFFF) as i32) - 32768;
            cols.wide[1][i].write(u_lo);
            cols.narrow[2][i].write(u_mid);
            cols.narrow[3][i].write(a_u);

            // v: wide[2] low 16 bits, wide[3] high 16 bits
            let yv = (out.v[i] as u32).wrapping_add(1 << 31);
            cols.wide[2][i].write((yv & 65535) as i32 - 32768);
            cols.wide[3][i].write((yv >> 16) as i32 - 32768);

            // w1: 4 bits
            for t in 0..4 {
                cols.narrow[t + 4][i].write(((out.w1[i] >> t) & 1) as i8);
            }
        }
    }
}
