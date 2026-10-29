//! Packed sparse products, CRT finishing, and H-u digits.
//!
//! Rounding follows the CC0 pq-crystals Dilithium AVX2 rounding implementation.
//! The sparse signed-window technique also appears in Hybrid-SHA-3's pspm.c.
use super::{FinishError, FinishOut, FinishRow, HuCols};
use crate::{params::*, poly::{Challenge39, Poly32}};
use std::mem::MaybeUninit;
use std::arch::x86_64::*;
type V = __m512i;
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
fn splat(x: i32) -> V { _mm512_set1_epi32(x) }
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
fn add(a: V, b: V) -> V { _mm512_add_epi32(a,b) }
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
fn sub(a: V, b: V) -> V { _mm512_sub_epi32(a,b) }
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
fn mul(a: V, b: V) -> V { _mm512_mullo_epi32(a,b) }
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
fn and(a: V, b: V) -> V { _mm512_and_si512(a,b) }
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
fn or(a: V, b: V) -> V { _mm512_or_si512(a,b) }
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
fn gt(a: V, b: V) -> V { _mm512_maskz_set1_epi32(_mm512_cmpgt_epi32_mask(a,b), -1) }
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
fn eq(a: V, b: V) -> V { _mm512_maskz_set1_epi32(_mm512_cmpeq_epi32_mask(a,b), -1) }
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
fn sar<const K: u32>(a: V) -> V { _mm512_srai_epi32::<K>(a) }
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
fn shr<const K: u32>(a: V) -> V { _mm512_srli_epi32::<K>(a) }
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
fn select(mask: V, yes: V, no: V) -> V { or(and(mask,yes), andnot(mask,no)) }
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
fn andnot(a: V,b: V) -> V { _mm512_andnot_si512(a,b) }
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
fn load(a: &[i32; 16]) -> V {
    // SAFETY: the fixed-size reference contains exactly one unaligned vector.
    unsafe { _mm512_loadu_si512(a.as_ptr().cast()) }
}
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
fn store(a: &mut [i32; 16], v: V) {
    // SAFETY: the fixed-size reference contains exactly one unaligned vector.
    unsafe { _mm512_storeu_si512(a.as_mut_ptr().cast(), v) }
}
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
fn wide(a: &mut [MaybeUninit<i32>], v: V) {
    let a: &mut [MaybeUninit<i32>; 16] = a.try_into().unwrap();
    // SAFETY: all vector bytes lie in the destination chunk; no uninitialized value is read.
    unsafe { _mm512_storeu_si512(a.as_mut_ptr().cast(), v) }
}
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
fn narrow(aout: &mut [MaybeUninit<i8>], v: V) {
    let aout: &mut [MaybeUninit<i8>; 16] = aout.try_into().unwrap();
    let bytes = _mm512_cvtepi32_epi8(v);
    // SAFETY: the destination chunk contains exactly sixteen writable bytes.
    unsafe { _mm_storeu_si128(aout.as_mut_ptr().cast(),bytes) }
}
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
fn bytes(a: &[u8; 16]) -> V {
    // SAFETY: the load reads exactly the fixed-size byte array.
    unsafe { _mm512_cvtepu8_epi32(_mm_loadu_si128(a.as_ptr().cast())) }
}
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
fn put_bytes(a: &mut [u8; 16], v: V) {
    let b = _mm512_cvtusepi32_epi8(v);
    // SAFETY: the store writes exactly sixteen bytes.
    unsafe { _mm_storeu_si128(a.as_mut_ptr().cast(), b) }
}
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
fn crt_mont(a: V) -> V {
    let c = splat(CRT_C);
    let t = mul(a, splat(CRT_C.wrapping_mul(CONSTS_P2.pinv)));
    let even = _mm512_sub_epi64(_mm512_mul_epi32(a,c), _mm512_mul_epi32(t,splat(P2)));
    let odd = _mm512_sub_epi64(_mm512_mul_epi32(_mm512_srli_epi64::<32>(a),c),
        _mm512_mul_epi32(_mm512_srli_epi64::<32>(t),splat(P2)));
    // |a| < P2, |CRT_C| < P2/2: Montgomery output lies in (-P2,P2).
    or(_mm512_srli_epi64::<32>(even), _mm512_slli_epi64::<32>(_mm512_srli_epi64::<32>(odd)))
}
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
pub(crate) unsafe fn sparse_ct(c: &Challenge39, t1: &[[u16;256];4], shift: u32, out: &mut [[i32;256];4]) {
    for (t,dest) in t1.iter().zip(out) {
        let mut window = [0i32;512];
        for i in 0..256 { window[i] = -(t[i] as i32); window[i+256] = t[i] as i32; }
        for base in (0..256).step_by(256) {
            let mut a0 = splat(0);
            let mut a1 = splat(0);
            let mut a2 = splat(0);
            let mut a3 = splat(0);
            let mut a4 = splat(0);
            let mut a5 = splat(0);
            let mut a6 = splat(0);
            let mut a7 = splat(0);
            let mut a8 = splat(0);
            let mut a9 = splat(0);
            let mut a10 = splat(0);
            let mut a11 = splat(0);
            let mut a12 = splat(0);
            let mut a13 = splat(0);
            let mut a14 = splat(0);
            let mut a15 = splat(0);
            for (k,&pos) in c.pos.iter().enumerate() {
                let start = 256 - pos as usize + base;
                let sign = splat(-((c.neg >> k) as i32 & 1));
                // Each partial sum has magnitude <= 39*1023 = 39897, safely in i32.
                let v = load(window[start+0..start+16].try_into().unwrap());
                a0 = add(a0, sub(_mm512_xor_si512(v,sign),sign));
                let v = load(window[start+16..start+32].try_into().unwrap());
                a1 = add(a1, sub(_mm512_xor_si512(v,sign),sign));
                let v = load(window[start+32..start+48].try_into().unwrap());
                a2 = add(a2, sub(_mm512_xor_si512(v,sign),sign));
                let v = load(window[start+48..start+64].try_into().unwrap());
                a3 = add(a3, sub(_mm512_xor_si512(v,sign),sign));
                let v = load(window[start+64..start+80].try_into().unwrap());
                a4 = add(a4, sub(_mm512_xor_si512(v,sign),sign));
                let v = load(window[start+80..start+96].try_into().unwrap());
                a5 = add(a5, sub(_mm512_xor_si512(v,sign),sign));
                let v = load(window[start+96..start+112].try_into().unwrap());
                a6 = add(a6, sub(_mm512_xor_si512(v,sign),sign));
                let v = load(window[start+112..start+128].try_into().unwrap());
                a7 = add(a7, sub(_mm512_xor_si512(v,sign),sign));
                let v = load(window[start+128..start+144].try_into().unwrap());
                a8 = add(a8, sub(_mm512_xor_si512(v,sign),sign));
                let v = load(window[start+144..start+160].try_into().unwrap());
                a9 = add(a9, sub(_mm512_xor_si512(v,sign),sign));
                let v = load(window[start+160..start+176].try_into().unwrap());
                a10 = add(a10, sub(_mm512_xor_si512(v,sign),sign));
                let v = load(window[start+176..start+192].try_into().unwrap());
                a11 = add(a11, sub(_mm512_xor_si512(v,sign),sign));
                let v = load(window[start+192..start+208].try_into().unwrap());
                a12 = add(a12, sub(_mm512_xor_si512(v,sign),sign));
                let v = load(window[start+208..start+224].try_into().unwrap());
                a13 = add(a13, sub(_mm512_xor_si512(v,sign),sign));
                let v = load(window[start+224..start+240].try_into().unwrap());
                a14 = add(a14, sub(_mm512_xor_si512(v,sign),sign));
                let v = load(window[start+240..start+256].try_into().unwrap());
                a15 = add(a15, sub(_mm512_xor_si512(v,sign),sign));
            }
            store((&mut dest[base+0..base+16]).try_into().unwrap(), _mm512_sllv_epi32(a0,splat(shift as i32)));
            store((&mut dest[base+16..base+32]).try_into().unwrap(), _mm512_sllv_epi32(a1,splat(shift as i32)));
            store((&mut dest[base+32..base+48]).try_into().unwrap(), _mm512_sllv_epi32(a2,splat(shift as i32)));
            store((&mut dest[base+48..base+64]).try_into().unwrap(), _mm512_sllv_epi32(a3,splat(shift as i32)));
            store((&mut dest[base+64..base+80]).try_into().unwrap(), _mm512_sllv_epi32(a4,splat(shift as i32)));
            store((&mut dest[base+80..base+96]).try_into().unwrap(), _mm512_sllv_epi32(a5,splat(shift as i32)));
            store((&mut dest[base+96..base+112]).try_into().unwrap(), _mm512_sllv_epi32(a6,splat(shift as i32)));
            store((&mut dest[base+112..base+128]).try_into().unwrap(), _mm512_sllv_epi32(a7,splat(shift as i32)));
            store((&mut dest[base+128..base+144]).try_into().unwrap(), _mm512_sllv_epi32(a8,splat(shift as i32)));
            store((&mut dest[base+144..base+160]).try_into().unwrap(), _mm512_sllv_epi32(a9,splat(shift as i32)));
            store((&mut dest[base+160..base+176]).try_into().unwrap(), _mm512_sllv_epi32(a10,splat(shift as i32)));
            store((&mut dest[base+176..base+192]).try_into().unwrap(), _mm512_sllv_epi32(a11,splat(shift as i32)));
            store((&mut dest[base+192..base+208]).try_into().unwrap(), _mm512_sllv_epi32(a12,splat(shift as i32)));
            store((&mut dest[base+208..base+224]).try_into().unwrap(), _mm512_sllv_epi32(a13,splat(shift as i32)));
            store((&mut dest[base+224..base+240]).try_into().unwrap(), _mm512_sllv_epi32(a14,splat(shift as i32)));
            store((&mut dest[base+240..base+256]).try_into().unwrap(), _mm512_sllv_epi32(a15,splat(shift as i32)));
        }
    }
}
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
pub(crate) unsafe fn finish_row(row: FinishRow<'_>, out: &mut FinishOut) -> Result<(), FinishError> {
    let mut bad_q = splat(0);
    let mut bad_u = splat(0);
    let mut bad_v = splat(0);
    for i in (0..256).step_by(16) {
        let xq = load(row.xq[i..i+16].try_into().unwrap());
        let xp = load(row.xp[i..i+16].try_into().unwrap());
        let t = load(row.t[i..i+16].try_into().unwrap());
        let hint = bytes(row.h[i..i+16].try_into().unwrap());
        let k = crt_mont(sub(xp,xq));
        let k = add(k,and(sar::<31>(k),splat(P2)));
        let center = or(gt(k,splat((P2-1)/2)),and(eq(k,splat((P2-1)/2)),gt(xq,splat((Q-1)/2))));
        let kc = sub(k,and(center,splat(P2)));
        // |xq-t| <= q-1+B_T < 2^29; reduce32 leaves (-q,q).
        let a = sub(xq,t);
        let r = sub(a,mul(sar::<23>(add(a,splat(1<<22))),splat(Q)));
        let r = add(r,and(sar::<31>(r),splat(Q)));
        let hi = shr::<7>(add(r,splat(127)));
        let hi = shr::<24>(add(mul(hi,splat(11275)),splat(1<<23)));
        let hi = andnot(gt(hi,splat(43)),hi);
        let lo = sub(r,mul(hi,splat(ALPHA)));
        let lo = sub(lo,and(gt(lo,splat((Q-1)/2)),splat(Q)));
        let step = select(gt(lo,splat(0)),hint,sub(splat(0),hint));
        let w = add(hi,step);
        let w = select(gt(splat(0),w),splat(43),w);
        let w = andnot(gt(w,splat(43)),w);
        // r-ALPHA*w is in (-q,q), so two conditional corrections center it.
        let u = sub(r,mul(w,splat(ALPHA)));
        let u = sub(u,and(gt(u,splat((Q-1)/2)),splat(Q)));
        let u = add(u,and(gt(splat(-(Q-1)/2),u),splat(Q)));
        // |e| <= B_T+2*q < 2^29 and e is divisible by q. Its quotient is in [-41,41].
        let e = sub(sub(a,mul(w,splat(ALPHA))),u);
        let ve = mul(e,splat(QINV32));
        let v = add(kc,ve);
        bad_q = or(bad_q,gt(_mm512_abs_epi32(ve),splat(41)));
        bad_u = or(bad_u,gt(_mm512_abs_epi32(u),splat(B_U)));
        bad_v = or(bad_v,gt(_mm512_abs_epi32(v),splat(B_V)));
        store((&mut out.u[i..i+16]).try_into().unwrap(),u);
        store((&mut out.v[i..i+16]).try_into().unwrap(),v);
        put_bytes((&mut out.w1[i..i+16]).try_into().unwrap(),w);
    }
    if _mm512_test_epi32_mask(bad_q,bad_q) != 0 { Err(FinishError::Quotient) }
    else if _mm512_test_epi32_mask(bad_u,bad_u) != 0 { Err(FinishError::UBound) }
    else if _mm512_test_epi32_mask(bad_v,bad_v) != 0 { Err(FinishError::VBound) }
    else { Ok(()) }
}
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
pub(crate) unsafe fn hu_columns(z: &Poly32, out: &FinishOut, cols: HuCols<'_>) {
    for col in &cols.wide { assert_eq!(col.len(),256); }
    for col in &cols.narrow { assert_eq!(col.len(),256); }
    for i in (0..256).step_by(16) {
        let z = load(z[i..i+16].try_into().unwrap());
        let u = load(out.u[i..i+16].try_into().unwrap());
        let v = load(out.v[i..i+16].try_into().unwrap());
        let w = bytes(out.w1[i..i+16].try_into().unwrap());
        let y = add(z,splat(GAMMA1));
        wide(&mut cols.wide[0][i..i+16],sub(and(y,splat(65535)),splat(32768)));
        narrow(&mut cols.narrow[0][i..i+16],sub(sar::<16>(y),splat(2)));
        // Valid u gives y in [0,380930], H in [0,5], and a low digit in [-32768,30310].
        let y = add(u,splat(B_U));
        let mut h = splat(0);
        for threshold in [63079,126158,189237,252316,315395] { h = sub(h,gt(y,splat(threshold-1))); }
        wide(&mut cols.wide[1][i..i+16],sub(sub(y,mul(h,splat(63079))),splat(32768)));
        narrow(&mut cols.narrow[1][i..i+16],sub(h,splat(3)));
        let y = add(v,splat(i32::MIN));
        wide(&mut cols.wide[2][i..i+16],sub(and(y,splat(65535)),splat(32768)));
        wide(&mut cols.wide[3][i..i+16],sub(shr::<16>(y),splat(32768)));
        narrow(&mut cols.narrow[2][i..i+16],and(shr::<0>(w),splat(1)));
        narrow(&mut cols.narrow[3][i..i+16],and(shr::<1>(w),splat(1)));
        narrow(&mut cols.narrow[4][i..i+16],and(shr::<2>(w),splat(1)));
        narrow(&mut cols.narrow[5][i..i+16],and(shr::<3>(w),splat(1)));
        narrow(&mut cols.narrow[6][i..i+16],and(shr::<4>(w),splat(1)));
        narrow(&mut cols.narrow[7][i..i+16],and(shr::<5>(w),splat(1)));
    }
}
