//! Integer CRT finishing and sparse negacyclic products.
use super::{FinishError, FinishOut, FinishRow, FinishRow2P, FinishRowExact};
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

#[inline(always)]
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

/// The second pass of the vectorised finish: UseHint on `xq - t`, `u`, `v = k + ve` and
/// `w1'` into `out`; false when a bound fails (`|ve| > C_B_Q`, `|u| > B_u`, `|v| > B_v`).
/// `k` is the CRT quotient `(A z - xq) / q`, `|k| <= k_bound`.
#[inline(always)]
fn finish_from_k<P: ParamSet>(xq: &[i32; 256], k: &[i32; 256], t: &[i32; 256], h: &[u8; 256], out: &mut FinishOut) -> bool {
    let alpha = 2 * P::GAMMA2;
    let mut w = [std::mem::MaybeUninit::<i32>::uninit(); 256];
    let mut out_of_bounds = 0_u32;
    for i in 0..256 {
        let xqi = xq[i];
        let r = freeze(xqi - t[i]);
        let wi = if P::GAMMA2 == (Q - 1) / 88 {
            use_hint_88(r, h[i])
        } else {
            // use_hint_32 without branches: with a hint, hi + 1 for lo > 0, else hi - 1
            // (`((lo - 1) >> 31) | 1`), modulo 16.
            let mut hi = (r + 127) >> 7;
            hi = (hi * 1025 + (1 << 21)) >> 22;
            hi &= 15;
            let lo = r - hi * alpha;
            let lo = lo - ((((Q - 1) / 2 - lo) >> 31) & Q);
            let step = ((lo - 1) >> 31) | 1;
            let hint = -i32::from(h[i] != 0);
            (hi + (step & hint)) & 15
        };
        let u = r - alpha * wi;
        let u = u - (((((Q - 1) / 2) - u) >> 31) & Q);
        let u = u + (((u + (Q - 1) / 2) >> 31) & Q);
        let e = xqi - t[i] - alpha * wi - u;
        let ve = e.wrapping_mul(QINV32);
        // With |ve| <= C_B_Q the sum is exact; otherwise the reference runs.
        let v = k[i].wrapping_add(ve);
        out_of_bounds |= u32::from(ve.unsigned_abs() > P::C_B_Q)
            | u32::from(u.unsigned_abs() > P::B_U as u32)
            | u32::from(v.unsigned_abs() > P::B_V as u32);
        w[i].write(wi);
        out.u[i] = u;
        out.v[i] = v;
    }
    // SAFETY: the loop above initialized all 256 entries of `w`.
    let w = unsafe { &*(w.as_ptr().cast::<[i32; 256]>()) };
    for (dst, &wi) in out.w1.iter_mut().zip(w) {
        *dst = wi as u8;
    }
    out_of_bounds == 0
}

/// Finishes a row from the residues modulo P2 and P3 in two branch-free passes that the
/// compiler vectorises: Garner's CRT gives A z modulo `P2 P3` (centered), the row is
/// accepted only where the quotient `k = floor(A z / q)` is in `[-k_bound, k_bound]` (the
/// three-prime bound), and `xq = A z mod q`, `k` follow from the CRT value (`xq` from its
/// parts modulo q, `k = (A z - xq) q^-1 mod 2^32`, exact with `|k| < 2^31`). A row that
/// fails a second-pass bound is finished again by [`finish_row_from_k`], which returns the
/// first failing coefficient's error.
#[inline(always)]
pub(crate) fn finish_row_2p<P: ParamSet>(
    row: FinishRow2P<'_>,
    out: &mut FinishOut,
) -> Result<(), FinishError> {
    let (p2, p3, q) = (i64::from(P2), i64::from(P3), i64::from(Q));
    let (m, kb) = (p2 * p3, k_bound::<P>());
    let (lo, hi) = (-kb * q, (kb + 1) * q);
    let mut xq = [std::mem::MaybeUninit::<i32>::uninit(); 256];
    let mut k = [std::mem::MaybeUninit::<i32>::uninit(); 256];
    let mut out_of_range = 0_u32;
    for i in 0..256 {
        let (x2, x3) = (row.xp2[i], row.xp3[i]);
        // diff = (x3 - x2) mod P3 from (-P2, P3); factor = diff / P2 mod P3 in [0, P3).
        let diff = x3 - x2;
        let diff = diff + ((diff >> 31) & P3);
        let factor = mont(i64::from(diff) * i64::from(P2_INV_MONT_P3), CONSTS_P3);
        let factor = factor + ((factor >> 31) & P3);
        let raw = i64::from(x2) + p2 * i64::from(factor);
        let wrap = -i64::from(raw > m / 2);
        let az = raw - (m & wrap);
        out_of_range |= u32::from(az < lo) | u32::from(az >= hi);
        // A z mod q = x2 + (P2 factor mod q) - [wrap] (P2 P3 mod q): the sum lies in
        // (-2q, P2 + q), inside freeze's input range.
        let s = x2 + mont(i64::from(factor) * i64::from(P2_MONT_Q), CONSTS_Q) - (M_MOD_Q & wrap as i32);
        let r = freeze(s);
        xq[i].write(r);
        // A z - xq = k q exactly with |k| <= k_bound < 2^31 when accepted.
        k[i].write((az as i32).wrapping_sub(r).wrapping_mul(QINV32));
    }
    if out_of_range != 0 {
        // The three-prime finish checks every k before any other bound.
        return Err(FinishError::VBound);
    }
    // SAFETY: the loop above initialized all 256 entries of `xq` and `k`.
    let xq = unsafe { &*(xq.as_ptr().cast::<[i32; 256]>()) };
    let k = unsafe { &*(k.as_ptr().cast::<[i32; 256]>()) };
    if !finish_from_k::<P>(xq, k, row.t, row.h, out) {
        let k_coeffs = k.map(i64::from);
        return finish_row_from_k::<P>(&Poly32(*xq), &k_coeffs, row.t, row.h, out);
    }
    Ok(())
}

/// `2^32 P2 mod q`: a Montgomery product by it is a product by `P2` modulo q.
const P2_MONT_Q: i32 = (((P2 as i64) << 32) % Q as i64) as i32;
/// `P2 P3 mod q`.
const M_MOD_Q: i32 = ((P2 as i64 * P3 as i64) % Q as i64) as i32;

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

    /// The coefficient-by-coefficient three-prime finish: CRT of `k` modulo `P2 P3`, then
    /// [`finish_row_from_k`]; the first failing coefficient decides the error.
    fn finish_row_3p_reference<P: ParamSet>(
        xq: &Poly32,
        xp2: &Poly32,
        xp3: &Poly32,
        t: &[i32; 256],
        h: &[u8; 256],
        out: &mut FinishOut,
    ) -> Result<(), FinishError> {
        let mut k_coeffs = [0i64; 256];
        let p2 = P2 as i64;
        let p3 = P3 as i64;
        for i in 0..256 {
            let xqi = xq[i];
            let mut k2 = mont((xp2[i] - xqi) as i64 * CRT_C as i64, CONSTS_P2);
            k2 += (k2 >> 31) & P2;
            let mut k3 = mont((xp3[i] - xqi) as i64 * CRT_C_P3 as i64, CONSTS_P3);
            k3 += (k3 >> 31) & P3;
            let diff = (k3 as i64 - k2 as i64).rem_euclid(p3);
            let factor = (diff * P2_INV_MOD_P3 as i64).rem_euclid(p3);
            let mut k = k2 as i64 + p2 * factor;
            let m = p2 * p3;
            if k > m / 2 {
                k -= m;
            }
            if k.abs() > k_bound::<P>() {
                return Err(FinishError::VBound);
            }
            k_coeffs[i] = k;
        }
        finish_row_from_k::<P>(xq, &k_coeffs, t, h, out)
    }

    /// The two-prime finish agrees with the three-prime reference on the residues of the
    /// same A z, inside and outside the accepted range.
    fn two_prime_matches_the_reference<P: ParamSet>(seed: u64) {
        let mut rng = Rng(seed);
        let kb = k_bound::<P>() * i64::from(Q);
        let mut accepted = 0;
        for case in 0..400 {
            let scale = [kb, kb / 4, kb / 64, 2 * kb][case % 4];
            let mut az: [i64; 256] = std::array::from_fn(|_| rng.range(-scale, scale));
            if case % 9 == 0 {
                // The edges of the accepted quotients.
                let edges = [-kb, -kb - 1, kb + i64::from(Q) - 1, kb + i64::from(Q), 0, -1];
                az[(rng.next() % 256) as usize] = edges[(rng.next() % 6) as usize];
            }
            let (xq, xp2, xp3) = residues(&az);
            let tb = if case % 7 == 0 { 1 << 30 } else { 1 << 28 };
            let t: [i32; 256] = std::array::from_fn(|_| rng.range(-tb, tb) as i32);
            let h: [u8; 256] = std::array::from_fn(|_| u8::from(rng.next() % 9 == 0));
            let (mut got, mut want) = (FinishOut::zero(), FinishOut::zero());
            let r_got = finish_row_2p::<P>(FinishRow2P { xp2: &xp2, xp3: &xp3, t: &t, h: &h }, &mut got);
            let r_want = finish_row_3p_reference::<P>(&xq, &xp2, &xp3, &t, &h, &mut want);
            assert_eq!(r_got, r_want, "case {case}");
            accepted += usize::from(r_want.is_ok());
            if r_want.is_ok() {
                assert_eq!(got, want, "case {case}");
            }
        }
        assert!((40..360).contains(&accepted), "{accepted} of 400 rows accepted");
    }

    #[test]
    fn two_prime_finish_matches_the_reference() {
        two_prime_matches_the_reference::<Set65>(650);
        two_prime_matches_the_reference::<Set87>(870);
    }
}
