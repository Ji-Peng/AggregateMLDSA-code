//! Packed sparse products, CRT finishing, and H-u digits.
//!
//! Rounding follows the CC0 pq-crystals Dilithium AVX2 rounding implementation.
//! The sparse signed-window technique also appears in Hybrid-SHA-3's pspm.c.
use super::{FinishError, FinishOut, FinishRow, HuCols};
use crate::{params::*, poly::{Challenge39, Poly32}};
use std::mem::MaybeUninit;
use std::arch::x86_64::*;
const _: () = assert!(GAMMA1 == 1 << 17);
/// f32 value of (1/63079) * (1 + 2^-20).
const HU_RECIP: f32 = 1.5853151e-5;
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
fn wide(a: &mut [MaybeUninit<i32>], v: V) {
    let a: &mut [MaybeUninit<i32>; 8] = a.try_into().unwrap();
    // SAFETY: all vector bytes lie in the destination chunk; no uninitialized value is read.
    unsafe { _mm256_storeu_si256(a.as_mut_ptr().cast(), v) }
}
#[inline]
#[target_feature(enable = "avx2,popcnt")]
fn bytes(a: &[u8; 8]) -> V {
    // SAFETY: the load reads exactly the fixed-size byte array.
    unsafe { _mm256_cvtepu8_epi32(_mm_loadl_epi64(a.as_ptr().cast())) }
}


#[target_feature(enable = "avx2,popcnt")]
pub(crate) unsafe fn sparse_ct(c: &Challenge39, t1: &[[u16;256];4], shift: u32, out: &mut [[i32;256];4]) {
    for (t,dest) in t1.iter().zip(out) {
        let mut window = [0i32;512];
        for i in 0..256 { window[i] = -(t[i] as i32); window[i+256] = t[i] as i32; }
        for base in (0..256).step_by(64) {
            let mut a0 = splat(0);
            let mut a1 = splat(0);
            let mut a2 = splat(0);
            let mut a3 = splat(0);
            let mut a4 = splat(0);
            let mut a5 = splat(0);
            let mut a6 = splat(0);
            let mut a7 = splat(0);
            for (k,&pos) in c.pos.iter().enumerate() {
                let start = 256 - pos as usize + base;
                let sign = splat(-((c.neg >> k) as i32 & 1));
                // Each partial sum has magnitude <= 39*1023 = 39897, safely in i32.
                let v = load(window[start..start+8].try_into().unwrap());
                a0 = add(a0, sub(_mm256_xor_si256(v,sign),sign));
                let v = load(window[start+8..start+16].try_into().unwrap());
                a1 = add(a1, sub(_mm256_xor_si256(v,sign),sign));
                let v = load(window[start+16..start+24].try_into().unwrap());
                a2 = add(a2, sub(_mm256_xor_si256(v,sign),sign));
                let v = load(window[start+24..start+32].try_into().unwrap());
                a3 = add(a3, sub(_mm256_xor_si256(v,sign),sign));
                let v = load(window[start+32..start+40].try_into().unwrap());
                a4 = add(a4, sub(_mm256_xor_si256(v,sign),sign));
                let v = load(window[start+40..start+48].try_into().unwrap());
                a5 = add(a5, sub(_mm256_xor_si256(v,sign),sign));
                let v = load(window[start+48..start+56].try_into().unwrap());
                a6 = add(a6, sub(_mm256_xor_si256(v,sign),sign));
                let v = load(window[start+56..start+64].try_into().unwrap());
                a7 = add(a7, sub(_mm256_xor_si256(v,sign),sign));
            }
            store((&mut dest[base..base+8]).try_into().unwrap(), _mm256_sllv_epi32(a0,splat(shift as i32)));
            store((&mut dest[base+8..base+16]).try_into().unwrap(), _mm256_sllv_epi32(a1,splat(shift as i32)));
            store((&mut dest[base+16..base+24]).try_into().unwrap(), _mm256_sllv_epi32(a2,splat(shift as i32)));
            store((&mut dest[base+24..base+32]).try_into().unwrap(), _mm256_sllv_epi32(a3,splat(shift as i32)));
            store((&mut dest[base+32..base+40]).try_into().unwrap(), _mm256_sllv_epi32(a4,splat(shift as i32)));
            store((&mut dest[base+40..base+48]).try_into().unwrap(), _mm256_sllv_epi32(a5,splat(shift as i32)));
            store((&mut dest[base+48..base+56]).try_into().unwrap(), _mm256_sllv_epi32(a6,splat(shift as i32)));
            store((&mut dest[base+56..base+64]).try_into().unwrap(), _mm256_sllv_epi32(a7,splat(shift as i32)));
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
fn times_alpha(a: V) -> V {
    _mm256_slli_epi32::<11>(_mm256_mullo_epi16(a, c_load(&C_ALPHA_93)))
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

#[inline]
#[target_feature(enable = "avx2,popcnt")]
fn put_bytes(a: &mut [u8; 8], v: V) {
    let words = _mm_packus_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    let b = _mm_packus_epi16(words, words);
    // SAFETY: the store writes exactly eight bytes.
    unsafe { _mm_storel_epi64(a.as_mut_ptr().cast(), b) }
}

#[target_feature(enable = "avx2,popcnt")]
pub(crate) unsafe fn finish_row(row: FinishRow<'_>, out: &mut FinishOut) -> Result<(), FinishError> {
    let mut bad_q = splat(0);
    let mut bad_u = splat(0);
    let mut bad_v = splat(0);

    let c = c_load(&C_CRT_C);
    let c_pinv = c_load(&C_CRT_C_PINV);
    let p2 = c_load(&C_P2);
    let p2_half = c_load(&C_P2_HALF);
    let q_half = c_load(&C_Q_HALF);
    let qv = c_load(&C_Q);

    for i in (0..256).step_by(8) {
        let xq = load(row.xq[i..i+8].try_into().unwrap());
        let xp = load(row.xp[i..i+8].try_into().unwrap());
        let t = load(row.t[i..i+8].try_into().unwrap());
        let hint = bytes(row.h[i..i+8].try_into().unwrap());

        // CRT Montgomery reduction & centering
        let a_p = sub(xp, xq);
        let m = mul(a_p, c_pinv);
        let even = _mm256_sub_epi64(_mm256_mul_epi32(a_p, c), _mm256_mul_epi32(m, p2));
        let odd = _mm256_sub_epi64(
            _mm256_mul_epi32(_mm256_srli_epi64::<32>(a_p), c),
            _mm256_mul_epi32(_mm256_srli_epi64::<32>(m), p2),
        );
        let k = _mm256_blend_epi32::<0xAA>(_mm256_srli_epi64::<32>(even), odd);
        let k = _mm256_min_epu32(k, add(k, p2));
        let center = gt(sub(k, gt(xq, q_half)), p2_half);
        let kc = sub(k, and(center, p2));

        // Reduction modulo q of (xq - t)
        let a = sub(xq, t);
        let r = sub(a, mul(sar::<23>(add(a, c_load(&C_ROUND_Q))), qv));
        let r = _mm256_min_epu32(r, add(r, qv));

        // Decompose / Hint for gamma2 = (q-1)/88
        let hi = shr::<7>(add(r, c_load(&C_OFF127)));
        let hi = _mm256_mulhrs_epi16(_mm256_mulhi_epu16(hi, c_load(&C_MUL11275)), c_load(&C_SHIFT128));
        let hi = andnot(gt(hi, c_load(&C_MAX43)), hi);

        let lo = sub(r, times_alpha(hi));
        let neg = or(gt(c_load(&C_ONE), lo), gt(lo, q_half));
        let w = add(hi, sub(_mm256_xor_si256(hint, neg), neg));
        let w = _mm256_min_epu32(w, add(w, c_load(&C_MOD44)));
        let w = _mm256_min_epu32(w, sub(w, c_load(&C_MOD44)));

        let wa = times_alpha(w);
        let u = sub(r, wa);
        let u = sub(u, and(gt(u, q_half), qv));
        let u = add(u, and(gt(c_load(&C_NEG_Q_HALF), u), qv));

        let e = sub(sub(a, wa), u);
        let ve = mul(e, c_load(&C_QINV32));
        let v = add(kc, ve);

        bad_q = _mm256_max_epu32(bad_q, add(ve, c_load(&C_B_Q)));
        bad_u = _mm256_max_epu32(bad_u, add(u, c_load(&C_B_U)));
        bad_v = _mm256_max_epu32(bad_v, add(v, c_load(&C_B_V)));

        store((&mut out.u[i..i+8]).try_into().unwrap(), u);
        store((&mut out.v[i..i+8]).try_into().unwrap(), v);
        put_bytes((&mut out.w1[i..i+8]).try_into().unwrap(), w);
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
/// Packs four vectors of small signed values into 32 consecutive bytes (saturating).
#[inline]
#[target_feature(enable = "avx2,popcnt")]
fn pack32(a: V, b: V, c: V, d: V) -> V {
    let ab = _mm256_packs_epi32(a,b);
    let cd = _mm256_packs_epi32(c,d);
    let bytes = _mm256_packs_epi16(ab,cd);
    _mm256_permutevar8x32_epi32(bytes,_mm256_setr_epi32(0,4,1,5,2,6,3,7))
}
#[inline]
#[target_feature(enable = "avx2,popcnt")]
fn narrow32(a: &mut [MaybeUninit<i8>], v: V) {
    let a: &mut [MaybeUninit<i8>; 32] = a.try_into().unwrap();
    // SAFETY: the destination chunk contains exactly 32 writable bytes.
    unsafe { _mm256_storeu_si256(a.as_mut_ptr().cast(), v) }
}
#[target_feature(enable = "avx2,popcnt")]
pub(crate) unsafe fn hu_columns(z: &Poly32, out: &FinishOut, cols: HuCols<'_>) {
    for col in &cols.wide { assert_eq!(col.len(),256); }
    for col in &cols.narrow { assert_eq!(col.len(),256); }
    for i in (0..256).step_by(32) {
        let mut zhi = [splat(0);4];
        let mut uhi = [splat(0);4];
        for j in 0..4 {
            let o = i + 8*j;
            let z = load(z[o..o+8].try_into().unwrap());
            let u = load(out.u[o..o+8].try_into().unwrap());
            let v = load(out.v[o..o+8].try_into().unwrap());
            // z + 2^17 shifts the high digit by exactly 2 and leaves the low 16 bits unchanged.
            wide(&mut cols.wide[0][o..o+8],sub(and(z,splat(65535)),splat(32768)));
            zhi[j] = sar::<16>(z);
            // Valid u gives y in [0,380930]. The digit H counts the thresholds below y and
            // saturates at 5. The float estimate y * (1/63079) * (1 + 2^-20) never rounds below
            // the true quotient and stays under the next integer on this range.
            let y = add(u,splat(B_U));
            let hf = _mm256_cvttps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(y),_mm256_set1_ps(HU_RECIP)));
            let h = _mm256_min_epi32(hf,splat(5));
            wide(&mut cols.wide[1][o..o+8],sub(sub(y,mul(h,splat(63079))),splat(32768)));
            uhi[j] = sub(h,splat(3));
            wide(&mut cols.wide[2][o..o+8],sub(and(v,splat(65535)),splat(32768)));
            wide(&mut cols.wide[3][o..o+8],sar::<16>(v));
        }
        narrow32(&mut cols.narrow[0][i..i+32],pack32(zhi[0],zhi[1],zhi[2],zhi[3]));
        narrow32(&mut cols.narrow[1][i..i+32],pack32(uhi[0],uhi[1],uhi[2],uhi[3]));
        // Bits of the w1 bytes are split in the byte domain: shift 16-bit lanes, keep bit 0.
        // SAFETY: w1 has 256 bytes, so the 32-byte load at i is in bounds.
        let w = unsafe { _mm256_loadu_si256(out.w1.as_ptr().add(i).cast()) };
        let one = _mm256_set1_epi8(1);
        narrow32(&mut cols.narrow[2][i..i+32],and(w,one));
        narrow32(&mut cols.narrow[3][i..i+32],and(_mm256_srli_epi16::<1>(w),one));
        narrow32(&mut cols.narrow[4][i..i+32],and(_mm256_srli_epi16::<2>(w),one));
        narrow32(&mut cols.narrow[5][i..i+32],and(_mm256_srli_epi16::<3>(w),one));
        narrow32(&mut cols.narrow[6][i..i+32],and(_mm256_srli_epi16::<4>(w),one));
        narrow32(&mut cols.narrow[7][i..i+32],and(_mm256_srli_epi16::<5>(w),one));
    }
}
