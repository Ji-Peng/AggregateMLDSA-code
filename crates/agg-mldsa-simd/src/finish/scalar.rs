//! Integer CRT finishing, sparse negacyclic products, and H-u digits.
use super::{FinishError, FinishOut, FinishRow, HuCols};
use crate::ntt::scalar::mont;
use crate::{
    params::*,
    poly::{Challenge39, Poly32},
};

pub(crate) fn sparse_ct(
    c: &Challenge39,
    t1: &[[u16; 256]; 4],
    shift: u32,
    out: &mut [[i32; 256]; 4],
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
            // At most 39 signed terms of magnitude <= 1023: |sum| <= 39897.
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
fn use_hint(r: i32, h: u8) -> i32 {
    let mut hi = (r + 127) >> 7;
    hi = (hi * 11275 + (1 << 23)) >> 24;
    hi ^= ((43 - hi) >> 31) & hi;
    let mut lo = r - hi * ALPHA;
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
pub(crate) fn finish_row(row: FinishRow<'_>, out: &mut FinishOut) -> Result<(), FinishError> {
    for i in 0..256 {
        let xq = row.xq[i];
        let mut k = mont((row.xp[i] - xq) as i64 * CRT_C as i64, CONSTS_P2);
        k += (k >> 31) & P2;
        if k > (P2 - 1) / 2 || (k == (P2 - 1) / 2 && xq > (Q - 1) / 2) {
            k -= P2;
        }
        let r = freeze(xq - row.t[i]);
        let w = use_hint(r, row.h[i]);
        let mut u = r - ALPHA * w;
        if u > (Q - 1) / 2 {
            u -= Q;
        }
        if u < -(Q - 1) / 2 {
            u += Q;
        }
        let e = xq - row.t[i] - ALPHA * w - u;
        let ve = e.wrapping_mul(QINV32);
        if ve.unsigned_abs() > 41 {
            return Err(FinishError::Quotient);
        }
        let v = k + ve;
        if u.unsigned_abs() > B_U as u32 {
            return Err(FinishError::UBound);
        }
        if v.unsigned_abs() > B_V as u32 {
            return Err(FinishError::VBound);
        }
        out.w1[i] = w as u8;
        out.u[i] = u;
        out.v[i] = v;
    }
    Ok(())
}
pub(crate) fn hu_columns(z: &Poly32, out: &FinishOut, cols: HuCols<'_>) {
    for col in &cols.wide {
        assert_eq!(col.len(), 256);
    }
    for col in &cols.narrow {
        assert_eq!(col.len(), 256);
    }
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
}
