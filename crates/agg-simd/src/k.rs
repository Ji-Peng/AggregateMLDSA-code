//! Extension products read and write coordinate planes directly.
use crate::{
    detect::{backend, Backend},
    scalar,
};
pub type Planes<'a> = (&'a [u64], &'a [u64]);
pub type PlanesMut<'a> = (&'a mut [u64], &'a mut [u64]);
pub(crate) fn check(a: Planes<'_>, b: Planes<'_>, out: &PlanesMut<'_>) {
    let n = a.0.len();
    assert_eq!(a.1.len(), n);
    assert_eq!(b.0.len(), n);
    assert_eq!(b.1.len(), n);
    assert_eq!(out.0.len(), n);
    assert_eq!(out.1.len(), n);
}
pub fn k_mul(a: Planes<'_>, b: Planes<'_>, out: PlanesMut<'_>) {
    check(a, b, &out);
    #[cfg(target_arch = "x86_64")]
    if a.0.len() >= 64 && backend() == Backend::Avx512 {
        // SAFETY: the IFMA backend is active (checked here), and the lengths asserted above bound
        // every read and write.
        unsafe {
            return vector(a, b, out);
        }
    }
    #[cfg(target_arch = "x86_64")]
    if a.0.len() >= 64 {
        let n = a.0.len();
        let (o0, o1) = (out.0.as_mut_ptr(), out.1.as_mut_ptr());
        if let Some(()) = crate::wide::dispatch!(mul_w8(a, b, o0, o1, n), mul_w4(a, b, o0, o1, n)) {
            return;
        }
    }
    scalar::k_mul(a, b, out);
}
pub fn k_mul_base(a: Planes<'_>, b: &[u64], out: PlanesMut<'_>) {
    crate::fp::fp_mul(a.0, b, out.0);
    crate::fp::fp_mul(a.1, b, out.1);
}
/// Lane-wise `K` products; `o0`, `o1` address `n` writable words each.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn mul_wide<B: crate::wide::Simd>(a: Planes<'_>, b: Planes<'_>, o0: *mut u64, o1: *mut u64, n: usize) {
    use crate::wide::{kmul, loadk};
    let stop = n / B::LANES * B::LANES;
    // SAFETY: the caller guarantees the backend's features; every index is below `stop`.
    unsafe {
        for i in (0..stop).step_by(B::LANES) {
            let (re, im) = kmul::<B>(loadk::<B>(a, i), loadk::<B>(b, i));
            B::store(o0.add(i), re);
            B::store(o1.add(i), im);
        }
        scalar::k_mul(
            (&a.0[stop..], &a.1[stop..]),
            (&b.0[stop..], &b.1[stop..]),
            (std::slice::from_raw_parts_mut(o0.add(stop), n - stop), std::slice::from_raw_parts_mut(o1.add(stop), n - stop)),
        );
    }
}
#[cfg(target_arch = "x86_64")]
crate::wide::entry_points!(mul_wide => mul_w8, mul_w4; (a: Planes<'_>, b: Planes<'_>, o0: *mut u64, o1: *mut u64, n: usize));
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
#[cfg(target_arch = "x86_64")]
unsafe fn vector(a: Planes<'_>, b: Planes<'_>, out: PlanesMut<'_>) {
    // SAFETY: the target features are enabled here and the caller checked them at run time.
    // `check` asserts that all six planes hold `a.0.len()` words, and every vector access
    // ends at or below `n <= a.0.len()`.
    unsafe {
        use crate::x86::*;
        let n = a.0.len() / 8 * 8;
        for i in (0..n).step_by(8) {
            let (re, im) = kmul(
                load(a.0.as_ptr().add(i)),
                load(a.1.as_ptr().add(i)),
                load(b.0.as_ptr().add(i)),
                load(b.1.as_ptr().add(i)),
            );
            store(out.0.as_mut_ptr().add(i), re);
            store(out.1.as_mut_ptr().add(i), im);
        }
        scalar::k_mul(
            (&a.0[n..], &a.1[n..]),
            (&b.0[n..], &b.1[n..]),
            (&mut out.0[n..], &mut out.1[n..]),
        );
    }
}
