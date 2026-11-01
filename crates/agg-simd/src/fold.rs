//! Fused difference, extension product and addition, with canonical output.
use crate::{
    detect::{backend, Backend},
    k::{check, Planes, PlanesMut},
    scalar, K,
};
pub fn k_fold(r: K, a: Planes<'_>, b: Planes<'_>, out: PlanesMut<'_>) {
    check(a, b, &out);
    #[cfg(target_arch = "x86_64")]
    if a.0.len() >= 64 && backend() == Backend::Avx512 {
        // SAFETY: the caller checked the IFMA backend, and every slice or pointer is valid for the length asserted above.
        unsafe {
            return vector(r, a, b, out);
        }
    }
    #[cfg(target_arch = "x86_64")]
    if a.0.len() >= 64 {
        let n = a.0.len();
        let (o0, o1) = (out.0.as_mut_ptr(), out.1.as_mut_ptr());
        if let Some(()) = crate::wide::dispatch!(k_w8(r, a, b, o0, o1, n), k_w4(r, a, b, o0, o1, n)) {
            return;
        }
    }
    scalar::k_fold(r, a, b, out);
}
pub fn int_fold(r: K, lo: &[i64], hi: &[i64], out: PlanesMut<'_>) {
    assert_eq!(lo.len(), hi.len());
    assert_eq!(lo.len(), out.0.len());
    assert_eq!(lo.len(), out.1.len());
    #[cfg(target_arch = "x86_64")]
    if lo.len() >= 64 && backend() == Backend::Avx512 {
        // SAFETY: the caller checked the IFMA backend, and every slice or pointer is valid for the length asserted above.
        unsafe {
            return int_vector(r, lo, hi, out);
        }
    }
    #[cfg(target_arch = "x86_64")]
    if lo.len() >= 64 {
        let (o0, o1) = (out.0.as_mut_ptr(), out.1.as_mut_ptr());
        if let Some(()) = crate::wide::dispatch!(int_w8(r, lo, hi, o0, o1), int_w4(r, lo, hi, o0, o1)) {
            return;
        }
    }
    scalar::int_fold(r, lo, hi, out);
}

/// `a + r (b - a)` lane-wise; `o0`, `o1` address `n` writable words each.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn k_wide<B: crate::wide::Simd>(r: K, a: Planes<'_>, b: Planes<'_>, o0: *mut u64, o1: *mut u64, n: usize) {
    use crate::wide::*;
    let stop = n / B::LANES * B::LANES;
    // SAFETY: the caller guarantees the backend's features; every index is below `stop`.
    unsafe {
        let rs = split_k::<B>(bck::<B>(r));
        for i in (0..stop).step_by(B::LANES) {
            let (re, im) = kfold::<B>(loadk::<B>(a, i), loadk::<B>(b, i), rs);
            B::store(o0.add(i), re);
            B::store(o1.add(i), im);
        }
        scalar::k_fold(
            r,
            (&a.0[stop..], &a.1[stop..]),
            (&b.0[stop..], &b.1[stop..]),
            (std::slice::from_raw_parts_mut(o0.add(stop), n - stop), std::slice::from_raw_parts_mut(o1.add(stop), n - stop)),
        );
    }
}
#[cfg(target_arch = "x86_64")]
crate::wide::entry_points!(k_wide => k_w8, k_w4; (r: K, a: Planes<'_>, b: Planes<'_>, o0: *mut u64, o1: *mut u64, n: usize));

/// `l + r (h - l)` for signed integers read as canonical residues: the real part is
/// `l + r0 d`, the imaginary part `r1 d`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn int_wide<B: crate::wide::Simd>(r: K, lower: &[i64], upper: &[i64], o0: *mut u64, o1: *mut u64) {
    use crate::wide::*;
    let n = lower.len();
    let stop = n / B::LANES * B::LANES;
    // SAFETY: the caller guarantees the backend's features; every index is below `stop`.
    unsafe {
        let (r0, r1) = bck::<B>(r);
        let (r0, r1) = (split::<B>(r0), split::<B>(r1));
        for i in (0..stop).step_by(B::LANES) {
            let l = B::residue(B::load(lower.as_ptr().add(i).cast()));
            let h = B::residue(B::load(upper.as_ptr().add(i).cast()));
            let d = B::sub_mod(h, l);
            B::store(o0.add(i), mul_add::<B>(l, d, r0));
            B::store(o1.add(i), mul_s::<B>(d, r1));
        }
        scalar::int_fold(
            r,
            &lower[stop..],
            &upper[stop..],
            (std::slice::from_raw_parts_mut(o0.add(stop), n - stop), std::slice::from_raw_parts_mut(o1.add(stop), n - stop)),
        );
    }
}
#[cfg(target_arch = "x86_64")]
crate::wide::entry_points!(int_wide => int_w8, int_w4; (r: K, lower: &[i64], upper: &[i64], o0: *mut u64, o1: *mut u64));
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
#[cfg(target_arch = "x86_64")]
unsafe fn vector(r: K, a: Planes<'_>, b: Planes<'_>, out: PlanesMut<'_>) {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
    unsafe {
        use crate::x86::*;
        let r0 = bc(r.c0().to_canonical_u64());
        let r1 = bc(r.c1().to_canonical_u64());
        let n = a.0.len() / 8 * 8;
        for i in (0..n).step_by(8) {
            let a0 = load(a.0.as_ptr().add(i));
            let a1 = load(a.1.as_ptr().add(i));
            let d0 = sub_mod(load(b.0.as_ptr().add(i)), a0);
            let d1 = sub_mod(load(b.1.as_ptr().add(i)), a1);
            let (re, im) = kmul(d0, d1, r0, r1);
            store(out.0.as_mut_ptr().add(i), add_mod(a0, re));
            store(out.1.as_mut_ptr().add(i), add_mod(a1, im));
        }
        scalar::k_fold(
            r,
            (&a.0[n..], &a.1[n..]),
            (&b.0[n..], &b.1[n..]),
            (&mut out.0[n..], &mut out.1[n..]),
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
unsafe fn int_vector(r: K, lower: &[i64], upper: &[i64], out: PlanesMut<'_>) {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
    unsafe {
        use crate::x86::*;
        use std::arch::x86_64::*;
        let n = lower.len() / 8 * 8;
        let r0 = bc(r.c0().to_canonical_u64());
        let r1 = bc(r.c1().to_canonical_u64());
        for i in (0..n).step_by(8) {
            let l = load(lower.as_ptr().add(i).cast());
            let h = load(upper.as_ptr().add(i).cast());
            // Signed inputs lie in [-2^63,2^63). Adding p on negative lanes is
            // wrapping arithmetic and yields the canonical residue without scalar staging.
            let l = _mm512_mask_add_epi64(l, _mm512_cmplt_epi64_mask(l, z()), l, bc(crate::P));
            let h = _mm512_mask_add_epi64(h, _mm512_cmplt_epi64_mask(h, z()), h, bc(crate::P));
            let d = sub_mod(h, l);
            store(out.0.as_mut_ptr().add(i), add_mod(l, mul(r0, d)));
            store(out.1.as_mut_ptr().add(i), mul(r1, d));
        }
        scalar::int_fold(
            r,
            &lower[n..],
            &upper[n..],
            (&mut out.0[n..], &mut out.1[n..]),
        );
    }
}
