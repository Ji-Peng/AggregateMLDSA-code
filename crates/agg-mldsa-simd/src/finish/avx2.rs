//! Packed sparse products and CRT finishing.
//!
//! Rounding follows the CC0 pq-crystals Dilithium AVX2 rounding implementation.
//! The sparse signed-window technique also appears in Hybrid-SHA-3's pspm.c.
use super::{FinishError, FinishOut, FinishRow};
use crate::{
    params::*,
    poly::{Challenge, Poly32},
};
use std::mem::MaybeUninit;
use std::arch::x86_64::*;

type V = __m256i;
#[inline]
#[target_feature(enable = "avx2,popcnt")]
fn splat(x: i32) -> V { _mm256_set1_epi32(x) }
#[inline]
#[target_feature(enable = "avx2,popcnt")]
fn add(a: V, b: V) -> V { _mm256_add_epi32(a,b) }
#[inline]
#[target_feature(enable = "avx2,popcnt")]
fn sub(a: V, b: V) -> V { _mm256_sub_epi32(a,b) }
#[inline]
#[target_feature(enable = "avx2,popcnt")]
fn mul(a: V, b: V) -> V { _mm256_mullo_epi32(a,b) }
#[inline]
#[target_feature(enable = "avx2,popcnt")]
fn and(a: V, b: V) -> V { _mm256_and_si256(a,b) }
#[inline]
#[target_feature(enable = "avx2,popcnt")]
fn or(a: V, b: V) -> V { _mm256_or_si256(a,b) }
#[inline]
#[target_feature(enable = "avx2,popcnt")]
fn gt(a: V, b: V) -> V { _mm256_cmpgt_epi32(a,b) }
#[inline]
#[target_feature(enable = "avx2,popcnt")]
fn eq(a: V, b: V) -> V { _mm256_cmpeq_epi32(a,b) }
#[inline]
#[target_feature(enable = "avx2,popcnt")]
fn sar<const K: i32>(a: V) -> V { _mm256_srai_epi32::<K>(a) }
#[inline]
#[target_feature(enable = "avx2,popcnt")]
fn shr<const K: i32>(a: V) -> V { _mm256_srli_epi32::<K>(a) }
#[inline]
#[target_feature(enable = "avx2,popcnt")]
fn andnot(a: V,b: V) -> V { _mm256_andnot_si256(a,b) }
#[inline]
#[target_feature(enable = "avx2,popcnt")]
fn load(a: &[i32; 8]) -> V {
    // SAFETY: the fixed-size reference contains exactly one unaligned vector.
    unsafe { _mm256_loadu_si256(a.as_ptr().cast()) }
}
#[inline]
#[target_feature(enable = "avx2,popcnt")]
fn store(a: &mut [i32; 8], v: V) {
    // SAFETY: the fixed-size reference contains exactly one unaligned vector.
    unsafe { _mm256_storeu_si256(a.as_mut_ptr().cast(), v) }
}
#[inline]
#[target_feature(enable = "avx2,popcnt")]
fn bytes(a: &[u8; 8]) -> V {
    // SAFETY: the load reads exactly the fixed-size byte array.
    unsafe { _mm256_cvtepu8_epi32(_mm_loadl_epi64(a.as_ptr().cast())) }
}


#[target_feature(enable = "avx2,popcnt")]
pub(crate) unsafe fn sparse_ct<const K: usize, const TAU: usize>(
    c: &Challenge<TAU>,
    t1: &[[u16; 256]; K],
    shift: u32,
    out: &mut [[i32; 256]; K],
) {
    // SAFETY: AVX2 and POPCNT target features are enabled on this function.
    unsafe { sparse_ct_slice(c, t1, shift, out) }
}

#[target_feature(enable = "avx2,popcnt")]
pub(crate) unsafe fn sparse_ct_slice<const TAU: usize>(
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
        for base in (0..256).step_by(64) {
            let mut a0 = splat(0);
            let mut a1 = splat(0);
            let mut a2 = splat(0);
            let mut a3 = splat(0);
            let mut a4 = splat(0);
            let mut a5 = splat(0);
            let mut a6 = splat(0);
            let mut a7 = splat(0);
            for (k, &pos) in c.pos.iter().enumerate() {
                let start = 256 - pos as usize + base;
                let sign = splat(-((c.neg >> k) as i32 & 1));
                // Each partial sum has magnitude <= 60*1023 = 61380, safely in i32.
                let v = load(window[start..start + 8].try_into().unwrap());
                a0 = add(a0, sub(_mm256_xor_si256(v, sign), sign));
                let v = load(window[start + 8..start + 16].try_into().unwrap());
                a1 = add(a1, sub(_mm256_xor_si256(v, sign), sign));
                let v = load(window[start + 16..start + 24].try_into().unwrap());
                a2 = add(a2, sub(_mm256_xor_si256(v, sign), sign));
                let v = load(window[start + 24..start + 32].try_into().unwrap());
                a3 = add(a3, sub(_mm256_xor_si256(v, sign), sign));
                let v = load(window[start + 32..start + 40].try_into().unwrap());
                a4 = add(a4, sub(_mm256_xor_si256(v, sign), sign));
                let v = load(window[start + 40..start + 48].try_into().unwrap());
                a5 = add(a5, sub(_mm256_xor_si256(v, sign), sign));
                let v = load(window[start + 48..start + 56].try_into().unwrap());
                a6 = add(a6, sub(_mm256_xor_si256(v, sign), sign));
                let v = load(window[start + 56..start + 64].try_into().unwrap());
                a7 = add(a7, sub(_mm256_xor_si256(v, sign), sign));
            }
            store(
                (&mut dest[base..base + 8]).try_into().unwrap(),
                _mm256_sllv_epi32(a0, splat(shift as i32)),
            );
            store(
                (&mut dest[base + 8..base + 16]).try_into().unwrap(),
                _mm256_sllv_epi32(a1, splat(shift as i32)),
            );
            store(
                (&mut dest[base + 16..base + 24]).try_into().unwrap(),
                _mm256_sllv_epi32(a2, splat(shift as i32)),
            );
            store(
                (&mut dest[base + 24..base + 32]).try_into().unwrap(),
                _mm256_sllv_epi32(a3, splat(shift as i32)),
            );
            store(
                (&mut dest[base + 32..base + 40]).try_into().unwrap(),
                _mm256_sllv_epi32(a4, splat(shift as i32)),
            );
            store(
                (&mut dest[base + 40..base + 48]).try_into().unwrap(),
                _mm256_sllv_epi32(a5, splat(shift as i32)),
            );
            store(
                (&mut dest[base + 48..base + 56]).try_into().unwrap(),
                _mm256_sllv_epi32(a6, splat(shift as i32)),
            );
            store(
                (&mut dest[base + 56..base + 64]).try_into().unwrap(),
                _mm256_sllv_epi32(a7, splat(shift as i32)),
            );
        }
    }
}
#[repr(C, align(32))]
struct AlignedConst<T>(pub T);

macro_rules! c32 {
    ($val:expr) => {
        AlignedConst([$val; 8])
    };
}

static C_CRT_C: AlignedConst<[i32; 8]> = c32!(CRT_C);
static C_CRT_C_PINV: AlignedConst<[i32; 8]> = c32!(CRT_C.wrapping_mul(CONSTS_P2.pinv));
static C_P2: AlignedConst<[i32; 8]> = c32!(P2);
static C_P2_HALF: AlignedConst<[i32; 8]> = c32!((P2 - 1) / 2);
static C_Q_HALF: AlignedConst<[i32; 8]> = c32!((Q - 1) / 2);
static C_NEG_Q_HALF: AlignedConst<[i32; 8]> = c32!(-(Q - 1) / 2);
static C_Q: AlignedConst<[i32; 8]> = c32!(Q);
static C_ROUND_Q: AlignedConst<[i32; 8]> = c32!(1 << 22);
static C_OFF127: AlignedConst<[i32; 8]> = c32!(127);
static C_MUL11275: AlignedConst<[i32; 8]> = c32!(11275);
static C_SHIFT128: AlignedConst<[i32; 8]> = c32!(128);
static C_MAX43: AlignedConst<[i32; 8]> = c32!(43);
static C_ALPHA_93: AlignedConst<[i32; 8]> = c32!(93);
#[inline]
#[target_feature(enable = "avx2,popcnt")]
fn times_alpha_with(a: V, alpha_93: V) -> V {
    _mm256_slli_epi32::<11>(_mm256_mullo_epi16(a, alpha_93))
}

#[inline]
#[target_feature(enable = "avx2,popcnt")]
fn put_bytes(a: &mut [u8; 8], v: V) {
    let words = _mm_packus_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    let b = _mm_packus_epi16(words, words);
    // SAFETY: the store writes exactly eight bytes.
    unsafe { _mm_storel_epi64(a.as_mut_ptr().cast(), b) }
}

static C_ONE: AlignedConst<[i32; 8]> = c32!(1);
static C_MOD44: AlignedConst<[i32; 8]> = c32!(44);
static C_QINV32: AlignedConst<[i32; 8]> = c32!(QINV32);
static C_B_Q: AlignedConst<[i32; 8]> = c32!(41);
static C_B_U: AlignedConst<[i32; 8]> = c32!(B_U);
static C_B_V: AlignedConst<[i32; 8]> = c32!(B_V);

#[inline(always)]
fn c_load(c: &'static AlignedConst<[i32; 8]>) -> V {
    // SAFETY: the static array is aligned to 32 bytes and contains 8 i32 values.
    unsafe { _mm256_load_si256(c.0.as_ptr().cast()) }
}

/// `k = ((xp - xq) q^-1) mod P2` centred as the CRT of `(xq, xp)` requires
/// (see the scalar kernel): a signed Montgomery product by `CRT_C`, one
/// conditional `+ P2`, then `- P2` iff `k > (P2-1)/2` or (`k = (P2-1)/2` and
/// `xq > (q-1)/2`), computed as `k + [xq > (q-1)/2] > (P2-1)/2` (the compare
/// mask is -1 where true, so subtracting it adds the indicator).
#[inline]
#[target_feature(enable = "avx2,popcnt")]
fn crt_k(xq: V, xp: V, c: V, c_pinv: V, p2: V, p2_half: V, q_half: V) -> V {
    let a = sub(xp, xq);
    let m = mul(a, c_pinv);
    // Products of the even and the odd lanes; their high halves are the
    // Montgomery results, |k| < P2 for |xp - xq| < 2^28.
    let even = _mm256_sub_epi64(_mm256_mul_epi32(a, c), _mm256_mul_epi32(m, p2));
    let odd = _mm256_sub_epi64(
        _mm256_mul_epi32(_mm256_srli_epi64::<32>(a), c),
        _mm256_mul_epi32(_mm256_srli_epi64::<32>(m), p2),
    );
    let k = _mm256_blend_epi32::<0xAA>(_mm256_srli_epi64::<32>(even), odd);
    let k = _mm256_min_epu32(k, add(k, p2));
    let center = gt(sub(k, gt(xq, q_half)), p2_half);
    sub(k, and(center, p2))
}

#[target_feature(enable = "avx2,popcnt")]
pub(crate) unsafe fn finish_row<P: ParamSet>(
    row: FinishRow<'_>,
    out: &mut FinishOut,
) -> Result<(), FinishError> {
    if P::GAMMA2 == (Q - 1) / 88 {
        // SAFETY: AVX2 and POPCNT target features are enabled on this function.
        unsafe { finish_row_44(row, out) }
    } else {
        super::scalar::finish_row::<P>(row, out)
    }
}

#[target_feature(enable = "avx2,popcnt")]
pub(crate) unsafe fn finish_row_2p<P: ParamSet>(
    row: super::FinishRow2P<'_>,
    out: &mut FinishOut,
) -> Result<(), FinishError> {
    super::scalar::finish_row_2p::<P>(row, out)
}

#[target_feature(enable = "avx2,popcnt")]
pub(crate) unsafe fn finish_row_exact<P: ParamSet>(
    row: super::FinishRowExact<'_>,
    out: &mut FinishOut,
) -> Result<(), FinishError> {
    super::scalar::finish_row_exact::<P>(row, out)
}

#[target_feature(enable = "avx2,popcnt")]
unsafe fn finish_row_44(row: FinishRow<'_>, out: &mut FinishOut) -> Result<(), FinishError> {
    // Four short passes through L1-resident rows instead of one loop: a fused
    // iteration is one ~75-cycle dependency chain, and the scheduler then holds
    // too few iterations to fill the ports (measured 25% slower). `r` holds
    // r = (xq - t) mod q, later r - u; `w` holds w1 as i32; `out.v` holds kc
    // until the last pass adds ve.
    let mut r = MaybeUninit::<Poly32>::uninit();
    let mut w = MaybeUninit::<Poly32>::uninit();
    {
        // Pass 1: CRT quotient kc and r; two independent short chains.
        let c = c_load(&C_CRT_C);
        let c_pinv = c_load(&C_CRT_C_PINV);
        let p2 = c_load(&C_P2);
        let p2_half = c_load(&C_P2_HALF);
        let q_half = c_load(&C_Q_HALF);
        let qv = c_load(&C_Q);
        let round_q = c_load(&C_ROUND_Q);
        for i in (0..256).step_by(8) {
            let xq = load(row.xq[i..i+8].try_into().unwrap());
            let xp = load(row.xp[i..i+8].try_into().unwrap());
            store((&mut out.v[i..i+8]).try_into().unwrap(), crt_k(xq, xp, c, c_pinv, p2, p2_half, q_half));
            // a = xq - t, |a| < 2^28.4. Subtracting round(a / 2^23) q, with
            // t q = (t << 23) - (t << 13) + t (shorter latency than a
            // multiply; equal modulo 2^32), leaves (-q, q); then add q to
            // negatives (unsigned minimum).
            let a = sub(xq, load(row.t[i..i+8].try_into().unwrap()));
            let tq = sar::<23>(add(a, round_q));
            let x = add(sub(a, _mm256_slli_epi32::<23>(tq)), sub(_mm256_slli_epi32::<13>(tq), tq));
            // SAFETY: i + 8 <= 256 coefficients of the aligned scratch row.
            unsafe {
                _mm256_store_si256(r.as_mut_ptr().cast::<i32>().add(i).cast(), _mm256_min_epu32(x, add(x, qv)))
            };
        }
    }
    // SAFETY: pass 1 wrote all 256 coefficients.
    let r = unsafe { r.assume_init_mut() };
    {
        // Pass 2: Decompose and UseHint (gamma2 = (q-1)/88), w1 in [0, 43].
        let off127 = c_load(&C_OFF127);
        let mul11275 = c_load(&C_MUL11275);
        let shift128 = c_load(&C_SHIFT128);
        let max43 = c_load(&C_MAX43);
        let alpha_93 = c_load(&C_ALPHA_93);
        let one = c_load(&C_ONE);
        let q_half = c_load(&C_Q_HALF);
        let mod44 = c_load(&C_MOD44);
        for i in (0..256).step_by(8) {
            let f = load(r[i..i+8].try_into().unwrap());
            let hint = bytes(row.h[i..i+8].try_into().unwrap());
            let hi = shr::<7>(add(f, off127));
            let hi = _mm256_mulhrs_epi16(_mm256_mulhi_epu16(hi, mul11275), shift128);
            let hi = andnot(gt(hi, max43), hi);
            let lo = sub(f, times_alpha_with(hi, alpha_93));
            let neg = or(gt(one, lo), gt(lo, q_half));
            // hi + 1 or hi - 1 with a hint (in [-1, 44]), wrapped into [0, 43].
            let x = add(hi, sub(_mm256_xor_si256(hint, neg), neg));
            let x = _mm256_min_epu32(x, add(x, mod44));
            let x = _mm256_min_epu32(x, sub(x, mod44));
            // SAFETY: i + 8 <= 256 coefficients of the aligned scratch row.
            unsafe { _mm256_store_si256(w.as_mut_ptr().cast::<i32>().add(i).cast(), x) };
        }
    }
    // SAFETY: pass 2 wrote all 256 coefficients.
    let w = unsafe { w.assume_init_ref() };
    let qinv32 = c_load(&C_QINV32);
    let b_q = c_load(&C_B_Q);
    let b_v = c_load(&C_B_V);
    let mut bad_q = splat(0);
    let mut bad_u = splat(0);
    let mut bad_v = splat(0);
    {
        // Pass 3: u = centred (r - 2 gamma2 w1) from (-q, q); w1 bytes.
        let alpha_93 = c_load(&C_ALPHA_93);
        let qv = c_load(&C_Q);
        let q_half = c_load(&C_Q_HALF);
        let neg_q_half = c_load(&C_NEG_Q_HALF);
        let b_u = c_load(&C_B_U);
        for i in (0..256).step_by(8) {
            let rv = load(r[i..i+8].try_into().unwrap());
            let wv = load(w[i..i+8].try_into().unwrap());
            let wa = times_alpha_with(wv, alpha_93);
            let u = sub(rv, wa);
            let u = sub(u, and(gt(u, q_half), qv));
            let u = add(u, and(gt(neg_q_half, u), qv));
            bad_u = _mm256_max_epu32(bad_u, add(u, b_u));
            store((&mut out.u[i..i+8]).try_into().unwrap(), u);
            store((&mut r[i..i+8]).try_into().unwrap(), add(wa, u));
            put_bytes((&mut out.w1[i..i+8]).try_into().unwrap(), wv);
        }
    }
    // Pass 4: e = (xq - t) - (2 gamma2 w1 + u), |e| < 2^28.4; ve = e / q exact
    // when q divides e (QINV32 = q^-1 mod 2^32); v = kc + ve.
    for i in (0..256).step_by(8) {
        let a = sub(load(row.xq[i..i+8].try_into().unwrap()), load(row.t[i..i+8].try_into().unwrap()));
        let e = sub(a, load(r[i..i+8].try_into().unwrap()));
        let ve = mul(e, qinv32);
        let v = add(load(out.v[i..i+8].try_into().unwrap()), ve);
        bad_q = _mm256_max_epu32(bad_q, add(ve, b_q));
        bad_v = _mm256_max_epu32(bad_v, add(v, b_v));
        store((&mut out.v[i..i+8]).try_into().unwrap(), v);
    }

    let exceeds = |acc: V, bound: i32| {
        let limit = splat(2 * bound);
        _mm256_movemask_epi8(eq(_mm256_max_epu32(acc, limit), limit)) != -1
    };
    if exceeds(bad_q, 41) { Err(FinishError::Quotient) }
    else if exceeds(bad_u, B_U) { Err(FinishError::UBound) }
    else if exceeds(bad_v, B_V) { Err(FinishError::VBound) }
    else { Ok(()) }
}

/// Standalone step for CRT Montgomery reduction and centring (for microbenchmarks).
///
/// # Safety
/// Requires target features `avx2` and `popcnt`.
#[target_feature(enable = "avx2,popcnt")]
pub unsafe fn step_crt(xq: &Poly32, xp: &Poly32, out_kc: &mut Poly32) {
    let c = c_load(&C_CRT_C);
    let c_pinv = c_load(&C_CRT_C_PINV);
    let p2 = c_load(&C_P2);
    let p2_half = c_load(&C_P2_HALF);
    let q_half = c_load(&C_Q_HALF);

    for i in (0..256).step_by(16) {
        let xq0 = load(xq[i..i+8].try_into().unwrap());
        let xp0 = load(xp[i..i+8].try_into().unwrap());
        let xq1 = load(xq[i+8..i+16].try_into().unwrap());
        let xp1 = load(xp[i+8..i+16].try_into().unwrap());

        let a_p0 = sub(xp0, xq0);
        let a_p1 = sub(xp1, xq1);

        let m0 = mul(a_p0, c_pinv);
        let m1 = mul(a_p1, c_pinv);

        let even0 = _mm256_sub_epi64(_mm256_mul_epi32(a_p0, c), _mm256_mul_epi32(m0, p2));
        let even1 = _mm256_sub_epi64(_mm256_mul_epi32(a_p1, c), _mm256_mul_epi32(m1, p2));

        let odd0 = _mm256_sub_epi64(
            _mm256_mul_epi32(_mm256_srli_epi64::<32>(a_p0), c),
            _mm256_mul_epi32(_mm256_srli_epi64::<32>(m0), p2),
        );
        let odd1 = _mm256_sub_epi64(
            _mm256_mul_epi32(_mm256_srli_epi64::<32>(a_p1), c),
            _mm256_mul_epi32(_mm256_srli_epi64::<32>(m1), p2),
        );

        let k0 = _mm256_blend_epi32::<0xAA>(_mm256_srli_epi64::<32>(even0), odd0);
        let k1 = _mm256_blend_epi32::<0xAA>(_mm256_srli_epi64::<32>(even1), odd1);

        let k0 = _mm256_min_epu32(k0, add(k0, p2));
        let k1 = _mm256_min_epu32(k1, add(k1, p2));

        let center0 = gt(sub(k0, gt(xq0, q_half)), p2_half);
        let center1 = gt(sub(k1, gt(xq1, q_half)), p2_half);

        let kc0 = sub(k0, and(center0, p2));
        let kc1 = sub(k1, and(center1, p2));

        store((&mut out_kc[i..i+8]).try_into().unwrap(), kc0);
        store((&mut out_kc[i+8..i+16]).try_into().unwrap(), kc1);
    }
}

/// Standalone step for reduce32 + caddq modulo q (for microbenchmarks).
///
/// # Safety
/// Requires target features `avx2` and `popcnt`.
#[target_feature(enable = "avx2,popcnt")]
pub unsafe fn step_reduce32_caddq(a: &mut Poly32) {
    let qv = c_load(&C_Q);
    let round_q = c_load(&C_ROUND_Q);
    for i in (0..256).step_by(8) {
        let v = load(a[i..i+8].try_into().unwrap());
        let r = sub(v, mul(sar::<23>(add(v, round_q)), qv));
        let r = _mm256_min_epu32(r, add(r, qv));
        store((&mut a[i..i+8]).try_into().unwrap(), r);
    }
}

/// Standalone step for polynomial decomposition (for microbenchmarks).
///
/// # Safety
/// Requires target features `avx2` and `popcnt`.
#[target_feature(enable = "avx2,popcnt")]
pub unsafe fn step_decompose(r: &Poly32, a1: &mut Poly32, a0: &mut Poly32) {
    let off127 = c_load(&C_OFF127);
    let mul11275 = c_load(&C_MUL11275);
    let shift128 = c_load(&C_SHIFT128);
    let max43 = c_load(&C_MAX43);
    let alpha_93 = c_load(&C_ALPHA_93);
    let qv = c_load(&C_Q);
    let q_half = c_load(&C_Q_HALF);

    for i in (0..256).step_by(8) {
        let f = load(r[i..i+8].try_into().unwrap());
        let hi = shr::<7>(add(f, off127));
        let hi = _mm256_mulhrs_epi16(_mm256_mulhi_epu16(hi, mul11275), shift128);
        let hi = andnot(gt(hi, max43), hi);
        let lo = sub(f, times_alpha_with(hi, alpha_93));
        let lo = sub(lo, and(gt(lo, q_half), qv));
        store((&mut a1[i..i+8]).try_into().unwrap(), hi);
        store((&mut a0[i..i+8]).try_into().unwrap(), lo);
    }
}

/// Standalone step for use_hint (for microbenchmarks).
///
/// # Safety
/// Requires target features `avx2` and `popcnt`.
#[target_feature(enable = "avx2,popcnt")]
pub unsafe fn step_use_hint(r: &Poly32, h: &[u8; 256], out: &mut Poly32) {
    let off127 = c_load(&C_OFF127);
    let mul11275 = c_load(&C_MUL11275);
    let shift128 = c_load(&C_SHIFT128);
    let max43 = c_load(&C_MAX43);
    let alpha_93 = c_load(&C_ALPHA_93);
    let one = c_load(&C_ONE);
    let q_half = c_load(&C_Q_HALF);
    let mod44 = c_load(&C_MOD44);

    for i in (0..256).step_by(8) {
        let f = load(r[i..i+8].try_into().unwrap());
        let hint = bytes(h[i..i+8].try_into().unwrap());
        let hi = shr::<7>(add(f, off127));
        let hi = _mm256_mulhrs_epi16(_mm256_mulhi_epu16(hi, mul11275), shift128);
        let hi = andnot(gt(hi, max43), hi);

        let lo = sub(f, times_alpha_with(hi, alpha_93));
        let neg = or(gt(one, lo), gt(lo, q_half));
        let w = add(hi, sub(_mm256_xor_si256(hint, neg), neg));
        let w = _mm256_min_epu32(w, add(w, mod44));
        let w = _mm256_min_epu32(w, sub(w, mod44));
        store((&mut out[i..i+8]).try_into().unwrap(), w);
    }
}

/// Standalone step for u, e, ve, v and bounds check (for microbenchmarks).
///
/// # Safety
/// Requires target features `avx2` and `popcnt`.
#[target_feature(enable = "avx2,popcnt")]
pub unsafe fn step_u_e_v_bounds(
    a: &Poly32,
    r: &Poly32,
    w: &Poly32,
    kc: &Poly32,
    out_u: &mut Poly32,
    out_v: &mut Poly32,
) -> Result<(), FinishError> {
    let alpha_93 = c_load(&C_ALPHA_93);
    let qv = c_load(&C_Q);
    let q_half = c_load(&C_Q_HALF);
    let neg_q_half = c_load(&C_NEG_Q_HALF);
    let qinv32 = c_load(&C_QINV32);
    let b_q = c_load(&C_B_Q);
    let b_u = c_load(&C_B_U);
    let b_v = c_load(&C_B_V);

    let mut bad_q = splat(0);
    let mut bad_u = splat(0);
    let mut bad_v = splat(0);

    for i in (0..256).step_by(8) {
        let a_val = load(a[i..i+8].try_into().unwrap());
        let r_val = load(r[i..i+8].try_into().unwrap());
        let w_val = load(w[i..i+8].try_into().unwrap());
        let kc_val = load(kc[i..i+8].try_into().unwrap());

        let wa = times_alpha_with(w_val, alpha_93);
        let u = sub(r_val, wa);
        let u = sub(u, and(gt(u, q_half), qv));
        let u = add(u, and(gt(neg_q_half, u), qv));

        let e = sub(sub(a_val, wa), u);
        let ve = mul(e, qinv32);
        let v = add(kc_val, ve);

        bad_q = _mm256_max_epu32(bad_q, add(ve, b_q));
        bad_u = _mm256_max_epu32(bad_u, add(u, b_u));
        bad_v = _mm256_max_epu32(bad_v, add(v, b_v));

        store((&mut out_u[i..i+8]).try_into().unwrap(), u);
        store((&mut out_v[i..i+8]).try_into().unwrap(), v);
    }

    let exceeds = |acc: V, bound: i32| {
        let limit = splat(2 * bound);
        _mm256_movemask_epi8(eq(_mm256_max_epu32(acc, limit), limit)) != -1
    };
    if exceeds(bad_q, 41) { Err(FinishError::Quotient) }
    else if exceeds(bad_u, B_U) { Err(FinishError::UBound) }
    else if exceeds(bad_v, B_V) { Err(FinishError::VBound) }
    else { Ok(()) }
}
