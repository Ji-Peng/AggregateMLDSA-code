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
    // |x| < q + B_T: 2^28.33 (ML-DSA-44), 2^28.65 (65), 2^28.93 (87). The rounded
    // quotient by 2^23 leaves (-q,q) for any |x| < 2^31 - 2^22.
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
    k_coeffs: &[i64; 256],
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
        // The CRT quotient k is wide; v is formed and bounded in i64 and narrowed
        // only after |v| <= B_v < 2^31 holds, so no wrong k can wrap into range.
        let v = k + i64::from(ve);
        if u.unsigned_abs() > P::B_U as u32 {
            return Err(FinishError::UBound);
        }
        if v.unsigned_abs() > P::B_V as u64 {
            return Err(FinishError::VBound);
        }
        out.w1[i] = w as u8;
        out.u[i] = u;
        out.v[i] = v as i32;
    }
    Ok(())
}

pub(crate) fn finish_row<P: ParamSet>(row: FinishRow<'_>, out: &mut FinishOut) -> Result<(), FinishError> {
    let mut k_coeffs = [0i64; 256];
    for i in 0..256 {
        let xq = row.xq[i];
        let mut k = mont((row.xp[i] - xq) as i64 * CRT_C as i64, CONSTS_P2);
        k += (k >> 31) & P2;
        if k > (P2 - 1) / 2 || (k == (P2 - 1) / 2 && xq > (Q - 1) / 2) {
            k -= P2;
        }
        k_coeffs[i] = i64::from(k);
    }
    finish_row_from_k::<P>(row.xq, &k_coeffs, row.t, row.h, out)
}

/// Largest |k| = |(A z - xq) / q| of an honest row: |A z| <= l 256 ((q-1)/2) gamma1
/// and 0 <= xq < q. A larger CRT quotient comes from a wrong residue.
const fn k_bound<P: ParamSet>() -> i64 {
    (P::L as i64 * 256 * ((Q as i64 - 1) / 2) * P::GAMMA1 as i64 + Q as i64) / Q as i64
}

/// `2^32 P2^-1 mod P3`: a Montgomery product by it is a product by `P2^-1 mod P3`.
const P2_INV_MONT_P3: i32 = (((P2_INV_MOD_P3 as i64) << 32) % P3 as i64) as i32;

/// [`finish_row_3p_reference`] in two branch-free passes over the row that the compiler
/// vectorises. A row with any out-of-bound value is finished again by the reference, which
/// returns its error, so the result is the reference's (on an error the outputs are
/// unspecified, as for every finish).
pub(crate) fn finish_row_3p<P: ParamSet>(
    row: FinishRow3P<'_>,
    out: &mut FinishOut,
) -> Result<(), FinishError> {
    let (p2, p3) = (i64::from(P2), i64::from(P3));
    let (m, kb) = (p2 * p3, k_bound::<P>());
    let mut k = [0_i32; 256];
    let mut bad = false;
    for i in 0..256 {
        let xq = row.xq[i];
        // k2 = (xp2 - xq) / q mod P2 and k3 likewise, each in [0, P).
        let k2 = mont(i64::from(row.xp2[i] - xq) * i64::from(CRT_C), CONSTS_P2);
        let k2 = k2 + ((k2 >> 31) & P2);
        let k3 = mont(i64::from(row.xp3[i] - xq) * i64::from(CRT_C_P3), CONSTS_P3);
        let k3 = k3 + ((k3 >> 31) & P3);
        // Garner: diff = (k3 - k2) mod P3 from (-P2, P3) with one correction, the factor
        // diff / P2 mod P3 as a Montgomery product (|diff * c| < 2^58, result in (-P3, P3)).
        let diff = k3 - k2;
        let diff = diff + ((diff >> 31) & P3);
        let factor = mont(i64::from(diff) * i64::from(P2_INV_MONT_P3), CONSTS_P3);
        let factor = factor + ((factor >> 31) & P3);
        let kk = i64::from(k2) + p2 * i64::from(factor);
        let kk = if kk > m / 2 { kk - m } else { kk };
        bad |= kk.abs() > kb;
        // |kk| <= k_bound < 2^29 whenever the row is accepted.
        k[i] = kk as i32;
    }
    if bad {
        return finish_row_3p_reference::<P>(row, out);
    }
    let alpha = 2 * P::GAMMA2;
    let mut w = [0_i32; 256];
    let mut out_of_bounds = 0_u32;
    for i in 0..256 {
        let xqi = row.xq[i];
        let r = freeze(xqi - row.t[i]);
        let wi = if P::GAMMA2 == (Q - 1) / 88 {
            use_hint_88(r, row.h[i])
        } else {
            // use_hint_32 without branches: with a hint, hi + 1 for lo > 0, else hi - 1
            // (`((lo - 1) >> 31) | 1`), modulo 16.
            let mut hi = (r + 127) >> 7;
            hi = (hi * 1025 + (1 << 21)) >> 22;
            hi &= 15;
            let lo = r - hi * alpha;
            let lo = lo - ((((Q - 1) / 2 - lo) >> 31) & Q);
            let step = ((lo - 1) >> 31) | 1;
            let hint = -i32::from(row.h[i] != 0);
            (hi + (step & hint)) & 15
        };
        let u = r - alpha * wi;
        let u = u - (((((Q - 1) / 2) - u) >> 31) & Q);
        let u = u + (((u + (Q - 1) / 2) >> 31) & Q);
        let e = xqi - row.t[i] - alpha * wi - u;
        let ve = e.wrapping_mul(QINV32);
        // With |ve| <= C_B_Q the sum is exact; otherwise the reference runs.
        let v = k[i].wrapping_add(ve);
        out_of_bounds |= u32::from(ve.unsigned_abs() > P::C_B_Q)
            | u32::from(u.unsigned_abs() > P::B_U as u32)
            | u32::from(v.unsigned_abs() > P::B_V as u32);
        w[i] = wi;
        out.u[i] = u;
        out.v[i] = v;
    }
    for (dst, &wi) in out.w1.iter_mut().zip(&w) {
        *dst = wi as u8;
    }
    bad |= out_of_bounds != 0;
    if bad {
        return finish_row_3p_reference::<P>(row, out);
    }
    Ok(())
}

/// The coefficient-by-coefficient three-prime finish: CRT of `k` modulo `P2 P3`, then
/// [`finish_row_from_k`]; the first failing coefficient decides the error.
pub(crate) fn finish_row_3p_reference<P: ParamSet>(
    row: FinishRow3P<'_>,
    out: &mut FinishOut,
) -> Result<(), FinishError> {
    let mut k_coeffs = [0i64; 256];
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
        // Reject before any narrowing: a residue that is wrong modulo P2 or P3
        // gives a k spread over |k| <= P2 P3 / 2 (about 2^54.8), almost always far
        // above the honest bound (below 2^28.81).
        if k.abs() > k_bound::<P>() {
            return Err(FinishError::VBound);
        }
        k_coeffs[i] = k;
    }
    finish_row_from_k::<P>(row.xq, &k_coeffs, row.t, row.h, out)
}

pub(crate) fn finish_row_exact<P: ParamSet>(
    row: FinishRowExact<'_>,
    out: &mut FinishOut,
) -> Result<(), FinishError> {
    let mut k_coeffs = [0i64; 256];
    let q = Q as i128;
    for i in 0..256 {
        let k = (row.az[i] - row.xq[i] as i128) / q;
        if k.abs() > i128::from(k_bound::<P>()) {
            return Err(FinishError::VBound);
        }
        k_coeffs[i] = k as i64;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A small deterministic generator.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 11
        }
        fn range(&mut self, lo: i64, hi: i64) -> i64 {
            lo + (self.next() % (hi - lo + 1) as u64) as i64
        }
    }

    /// The row residues of the integers `az`.
    fn residues(az: &[i64; 256]) -> (Poly32, Poly32, Poly32) {
        let m = |p: i32| Poly32(std::array::from_fn(|i| az[i].rem_euclid(i64::from(p)) as i32));
        (m(Q), m(P2), m(P3))
    }

    /// The two-pass finish agrees with the reference on honest rows and on rows with a
    /// corrupted residue, a large `T` or a flipped hint.
    fn matches_the_reference<P: ParamSet>(seed: u64) {
        let mut rng = Rng(seed);
        let kb = k_bound::<P>() * i64::from(Q);
        for case in 0..400 {
            let scale = [kb, kb / 4, kb / 64, 2 * kb][case % 4];
            let az: [i64; 256] = std::array::from_fn(|_| rng.range(-scale, scale));
            let (mut xq, mut xp2, mut xp3) = residues(&az);
            let tb = if case % 7 == 0 { 1 << 30 } else { 1 << 28 };
            let t: [i32; 256] = std::array::from_fn(|_| rng.range(-tb, tb) as i32);
            let h: [u8; 256] = std::array::from_fn(|_| u8::from(rng.next() % 9 == 0));
            match case % 5 {
                1 => xp2[(rng.next() % 256) as usize] ^= 1 << (rng.next() % 20),
                2 => xp3[(rng.next() % 256) as usize] ^= 1 << (rng.next() % 20),
                3 => xq[(rng.next() % 256) as usize] = (rng.next() % Q as u64) as i32,
                _ => {}
            }
            let row = FinishRow3P { xq: &xq, xp2: &xp2, xp3: &xp3, t: &t, h: &h };
            let (mut got, mut want) = (FinishOut::zero(), FinishOut::zero());
            let (r_got, r_want) = (finish_row_3p::<P>(row, &mut got), finish_row_3p_reference::<P>(row, &mut want));
            assert_eq!(r_got, r_want, "case {case}");
            if r_want.is_ok() {
                assert_eq!(got, want, "case {case}");
            }
        }
    }

    #[test]
    fn two_pass_finish_matches_the_reference() {
        matches_the_reference::<Set65>(65);
        matches_the_reference::<Set87>(87);
    }
}
