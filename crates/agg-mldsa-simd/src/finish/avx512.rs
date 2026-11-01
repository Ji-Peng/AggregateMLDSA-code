//! Packed sparse products, CRT finishing, and H-u digits.
//!
//! Rounding follows the CC0 pq-crystals Dilithium AVX2 rounding implementation.
//! The sparse signed-window technique also appears in Hybrid-SHA-3's pspm.c.
use super::{FinishError, FinishOut, FinishRow, HuCols};
use crate::{
    params::*,
    poly::{Challenge, Poly32},
};
use std::mem::MaybeUninit;
use std::arch::x86_64::*;

/// f32 value of (1/63079) * (1 + 2^-20).
const HU_RECIP: f32 = 1.5853151e-5;
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
fn sar<const K: u32>(a: V) -> V { _mm512_srai_epi32::<K>(a) }
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
fn shr<const K: u32>(a: V) -> V { _mm512_srli_epi32::<K>(a) }
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
    // The result is the high half of each 64-bit difference: even lanes from `even`, odd lanes from `odd`.
    _mm512_mask_blend_epi32(0xAAAA, _mm512_srli_epi64::<32>(even), odd)
}
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
pub(crate) unsafe fn sparse_ct<const K: usize, const TAU: usize>(
    c: &Challenge<TAU>,
    t1: &[[u16; 256]; K],
    shift: u32,
    out: &mut [[i32; 256]; K],
) {
    // SAFETY: AVX-512 target features are enabled on this function.
    unsafe { sparse_ct_slice(c, t1, shift, out) }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
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
            for (k, &pos) in c.pos.iter().enumerate() {
                let start = 256 - pos as usize + base;
                let sign = splat(-((c.neg >> k) as i32 & 1));
                // Each partial sum has magnitude <= 60*1023 = 61380, safely in i32.
                let v = load(window[start..start + 16].try_into().unwrap());
                a0 = add(a0, sub(_mm512_xor_si512(v, sign), sign));
                let v = load(window[start + 16..start + 32].try_into().unwrap());
                a1 = add(a1, sub(_mm512_xor_si512(v, sign), sign));
                let v = load(window[start + 32..start + 48].try_into().unwrap());
                a2 = add(a2, sub(_mm512_xor_si512(v, sign), sign));
                let v = load(window[start + 48..start + 64].try_into().unwrap());
                a3 = add(a3, sub(_mm512_xor_si512(v, sign), sign));
                let v = load(window[start + 64..start + 80].try_into().unwrap());
                a4 = add(a4, sub(_mm512_xor_si512(v, sign), sign));
                let v = load(window[start + 80..start + 96].try_into().unwrap());
                a5 = add(a5, sub(_mm512_xor_si512(v, sign), sign));
                let v = load(window[start + 96..start + 112].try_into().unwrap());
                a6 = add(a6, sub(_mm512_xor_si512(v, sign), sign));
                let v = load(window[start + 112..start + 128].try_into().unwrap());
                a7 = add(a7, sub(_mm512_xor_si512(v, sign), sign));
                let v = load(window[start + 128..start + 144].try_into().unwrap());
                a8 = add(a8, sub(_mm512_xor_si512(v, sign), sign));
                let v = load(window[start + 144..start + 160].try_into().unwrap());
                a9 = add(a9, sub(_mm512_xor_si512(v, sign), sign));
                let v = load(window[start + 160..start + 176].try_into().unwrap());
                a10 = add(a10, sub(_mm512_xor_si512(v, sign), sign));
                let v = load(window[start + 176..start + 192].try_into().unwrap());
                a11 = add(a11, sub(_mm512_xor_si512(v, sign), sign));
                let v = load(window[start + 192..start + 208].try_into().unwrap());
                a12 = add(a12, sub(_mm512_xor_si512(v, sign), sign));
                let v = load(window[start + 208..start + 224].try_into().unwrap());
                a13 = add(a13, sub(_mm512_xor_si512(v, sign), sign));
                let v = load(window[start + 224..start + 240].try_into().unwrap());
                a14 = add(a14, sub(_mm512_xor_si512(v, sign), sign));
                let v = load(window[start + 240..start + 256].try_into().unwrap());
                a15 = add(a15, sub(_mm512_xor_si512(v, sign), sign));
            }
            store(
                (&mut dest[base..base + 16]).try_into().unwrap(),
                _mm512_sllv_epi32(a0, splat(shift as i32)),
            );
            store(
                (&mut dest[base + 16..base + 32]).try_into().unwrap(),
                _mm512_sllv_epi32(a1, splat(shift as i32)),
            );
            store(
                (&mut dest[base + 32..base + 48]).try_into().unwrap(),
                _mm512_sllv_epi32(a2, splat(shift as i32)),
            );
            store(
                (&mut dest[base + 48..base + 64]).try_into().unwrap(),
                _mm512_sllv_epi32(a3, splat(shift as i32)),
            );
            store(
                (&mut dest[base + 64..base + 80]).try_into().unwrap(),
                _mm512_sllv_epi32(a4, splat(shift as i32)),
            );
            store(
                (&mut dest[base + 80..base + 96]).try_into().unwrap(),
                _mm512_sllv_epi32(a5, splat(shift as i32)),
            );
            store(
                (&mut dest[base + 96..base + 112]).try_into().unwrap(),
                _mm512_sllv_epi32(a6, splat(shift as i32)),
            );
            store(
                (&mut dest[base + 112..base + 128]).try_into().unwrap(),
                _mm512_sllv_epi32(a7, splat(shift as i32)),
            );
            store(
                (&mut dest[base + 128..base + 144]).try_into().unwrap(),
                _mm512_sllv_epi32(a8, splat(shift as i32)),
            );
            store(
                (&mut dest[base + 144..base + 160]).try_into().unwrap(),
                _mm512_sllv_epi32(a9, splat(shift as i32)),
            );
            store(
                (&mut dest[base + 160..base + 176]).try_into().unwrap(),
                _mm512_sllv_epi32(a10, splat(shift as i32)),
            );
            store(
                (&mut dest[base + 176..base + 192]).try_into().unwrap(),
                _mm512_sllv_epi32(a11, splat(shift as i32)),
            );
            store(
                (&mut dest[base + 192..base + 208]).try_into().unwrap(),
                _mm512_sllv_epi32(a12, splat(shift as i32)),
            );
            store(
                (&mut dest[base + 208..base + 224]).try_into().unwrap(),
                _mm512_sllv_epi32(a13, splat(shift as i32)),
            );
            store(
                (&mut dest[base + 224..base + 240]).try_into().unwrap(),
                _mm512_sllv_epi32(a14, splat(shift as i32)),
            );
            store(
                (&mut dest[base + 240..base + 256]).try_into().unwrap(),
                _mm512_sllv_epi32(a15, splat(shift as i32)),
            );
        }
    }
}
/// Multiplies lanes in [0,44] by ALPHA = 93 * 2^11 with one 16-bit multiply and a shift.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
fn times_alpha(a: V) -> V { _mm512_slli_epi32::<11>(_mm512_mullo_epi16(a,splat(93))) }
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
pub(crate) unsafe fn finish_row<P: ParamSet>(
    row: FinishRow<'_>,
    out: &mut FinishOut,
) -> Result<(), FinishError> {
    if P::GAMMA2 == (Q - 1) / 88 {
        // SAFETY: AVX-512 target features are enabled on this function.
        unsafe { finish_row_44(row, out) }
    } else {
        super::scalar::finish_row::<P>(row, out)
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
pub(crate) unsafe fn finish_row_3p<P: ParamSet>(
    row: super::FinishRow3P<'_>,
    out: &mut FinishOut,
) -> Result<(), FinishError> {
    super::scalar::finish_row_3p::<P>(row, out)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
pub(crate) unsafe fn finish_row_exact<P: ParamSet>(
    row: super::FinishRowExact<'_>,
    out: &mut FinishOut,
) -> Result<(), FinishError> {
    super::scalar::finish_row_exact::<P>(row, out)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
unsafe fn finish_row_44(row: FinishRow<'_>, out: &mut FinishOut) -> Result<(), FinishError> {
    let zero = splat(0);
    let hq = splat((Q-1)/2);
    let qv = splat(Q);
    let p2 = splat(P2);
    let mut bad_q = 0u16;
    let mut bad_u = 0u16;
    let mut bad_v = 0u16;
    // The row step is expanded 2 times per iteration so that independent chains interleave.
    macro_rules! step { ($i:expr) => {{
        let i: usize = $i;
        let xq = load(row.xq[i..i+16].try_into().unwrap());
        let xp = load(row.xp[i..i+16].try_into().unwrap());
        let t = load(row.t[i..i+16].try_into().unwrap());
        let hint = bytes(row.h[i..i+16].try_into().unwrap());
        let k = crt_mont(sub(xp,xq));
        let k = _mm512_mask_add_epi32(k,_mm512_cmplt_epi32_mask(k,zero),k,p2);
        let half = splat((P2-1)/2);
        let center = _mm512_cmpgt_epi32_mask(k,half)
            | _mm512_mask_cmpgt_epi32_mask(_mm512_cmpeq_epi32_mask(k,half),xq,hq);
        let kc = _mm512_mask_sub_epi32(k,center,k,p2);
        // |xq-t| <= q-1+B_T < 2^29; reduce32 leaves (-q,q).
        let a = sub(xq,t);
        let r = sub(a,mul(sar::<23>(add(a,splat(1<<22))),qv));
        let r = _mm512_mask_add_epi32(r,_mm512_cmplt_epi32_mask(r,zero),r,qv);
        // (r+127)>>7 < 2^16, so the 16-bit multiply-high pair computes round(x*11275/2^24) exactly.
        let hi = shr::<7>(add(r,splat(127)));
        let hi = _mm512_mulhrs_epi16(_mm512_mulhi_epu16(hi,splat(11275)),splat(128));
        let hi = _mm512_maskz_mov_epi32(_mm512_cmple_epi32_mask(hi,splat(43)),hi);
        let lo = sub(r,times_alpha(hi));
        // The centred low part is positive exactly when 0 < lo <= (q-1)/2.
        let pos = _mm512_mask_cmpgt_epi32_mask(_mm512_cmple_epi32_mask(lo,hq),lo,zero);
        let w = _mm512_mask_add_epi32(sub(hi,hint),pos,hi,hint);
        let w = _mm512_mask_mov_epi32(w,_mm512_cmplt_epi32_mask(w,zero),splat(43));
        let w = _mm512_maskz_mov_epi32(_mm512_cmple_epi32_mask(w,splat(43)),w);
        // r-ALPHA*w is in (-q,q), so two conditional corrections center it.
        let wa = times_alpha(w);
        let u = sub(r,wa);
        let u = _mm512_mask_sub_epi32(u,_mm512_cmpgt_epi32_mask(u,hq),u,qv);
        let u = _mm512_mask_add_epi32(u,_mm512_cmplt_epi32_mask(u,sub(zero,hq)),u,qv);
        // |e| <= B_T+2*q < 2^29 and e is divisible by q. Its quotient is in [-41,41].
        let e = sub(sub(a,wa),u);
        let ve = mul(e,splat(QINV32));
        let v = add(kc,ve);
        bad_q |= _mm512_cmpgt_epi32_mask(_mm512_abs_epi32(ve),splat(41));
        bad_u |= _mm512_cmpgt_epi32_mask(_mm512_abs_epi32(u),splat(B_U));
        bad_v |= _mm512_cmpgt_epi32_mask(_mm512_abs_epi32(v),splat(B_V));
        store((&mut out.u[i..i+16]).try_into().unwrap(),u);
        store((&mut out.v[i..i+16]).try_into().unwrap(),v);
        put_bytes((&mut out.w1[i..i+16]).try_into().unwrap(),w);
    }}; }
    for base in (0..256).step_by(32) {
        step!(base);
        step!(base+16);
    }
    if bad_q != 0 { Err(FinishError::Quotient) }
    else if bad_u != 0 { Err(FinishError::UBound) }
    else if bad_v != 0 { Err(FinishError::VBound) }
    else { Ok(()) }
}
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
fn narrow64(a: &mut [MaybeUninit<i8>], v: __m512i) {
    let a: &mut [MaybeUninit<i8>; 64] = a.try_into().unwrap();
    // SAFETY: the destination chunk contains exactly 64 writable bytes.
    unsafe { _mm512_storeu_si512(a.as_mut_ptr().cast(), v) }
}
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
pub(crate) unsafe fn hu_columns<P: ParamSet>(z: &Poly32, out: &FinishOut, cols: HuCols<'_>) {
    if P::GAMMA2 == (Q - 1) / 88 {
        // SAFETY: AVX-512 target features are enabled on this function.
        unsafe { hu_columns_44(z, out, cols) };
    } else {
        super::scalar::hu_columns::<P>(z, out, cols);
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
unsafe fn hu_columns_44(z: &Poly32, out: &FinishOut, cols: HuCols<'_>) {
    for col in &cols.wide { assert_eq!(col.len(),256); }
    for col in &cols.narrow { assert_eq!(col.len(),256); }
    for i in (0..256).step_by(16) {
        let z = load(z[i..i+16].try_into().unwrap());
        let u = load(out.u[i..i+16].try_into().unwrap());
        let v = load(out.v[i..i+16].try_into().unwrap());
        // z + 2^17 shifts the high digit by exactly 2 and leaves the low 16 bits unchanged.
        wide(&mut cols.wide[0][i..i+16],sub(and(z,splat(65535)),splat(32768)));
        narrow(&mut cols.narrow[0][i..i+16],sar::<16>(z));
        // Valid u gives y in [0,380930]. The digit H counts the thresholds below y and
        // saturates at 5. The float estimate y * (1/63079) * (1 + 2^-20) never rounds below
        // the true quotient and stays under the next integer on this range.
        let y = add(u,splat(B_U));
        let hf = _mm512_cvttps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(y),_mm512_set1_ps(HU_RECIP)));
        let h = _mm512_min_epi32(hf,splat(5));
        wide(&mut cols.wide[1][i..i+16],sub(sub(y,mul(h,splat(63079))),splat(32768)));
        narrow(&mut cols.narrow[1][i..i+16],sub(h,splat(3)));
        // The low 16 bits minus 2^15 and the arithmetic high half give the two v digits.
        wide(&mut cols.wide[2][i..i+16],sub(and(v,splat(65535)),splat(32768)));
        wide(&mut cols.wide[3][i..i+16],sar::<16>(v));
    }
    // Bits of the w1 bytes are split in the byte domain: shift 16-bit lanes, keep bit 0.
    let one = _mm512_set1_epi8(1);
    for i in (0..256).step_by(64) {
        // SAFETY: w1 has 256 bytes, so the 64-byte load at i is in bounds.
        let w = unsafe { _mm512_loadu_si512(out.w1.as_ptr().add(i).cast()) };
        narrow64(&mut cols.narrow[2][i..i+64],and(w,one));
        narrow64(&mut cols.narrow[3][i..i+64],and(_mm512_srli_epi16::<1>(w),one));
        narrow64(&mut cols.narrow[4][i..i+64],and(_mm512_srli_epi16::<2>(w),one));
        narrow64(&mut cols.narrow[5][i..i+64],and(_mm512_srli_epi16::<3>(w),one));
        narrow64(&mut cols.narrow[6][i..i+64],and(_mm512_srli_epi16::<4>(w),one));
        narrow64(&mut cols.narrow[7][i..i+64],and(_mm512_srli_epi16::<5>(w),one));
    }
}
