//! Canonical base-field slices.
use crate::{
    detect::{backend, Backend},
    scalar,
};
pub fn fp_mul(a: &[u64], b: &[u64], out: &mut [u64]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    #[cfg(target_arch = "x86_64")]
    if a.len() >= 64 && backend() == Backend::Avx512 {
        // SAFETY: the IFMA backend is active (checked here), and the lengths asserted above bound
        // every read and write.
        unsafe {
            return vector(a, b, out);
        }
    }
    #[cfg(target_arch = "x86_64")]
    if a.len() >= 64 {
        if let Some(()) = crate::wide::dispatch!(mul_w8(a, b, out), mul_w4(a, b, out)) {
            return;
        }
    }
    scalar::fp_mul(a, b, out);
}

/// Two independent vectors per step; the scalar oracle takes the tail.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn mul_wide<B: crate::wide::Simd>(a: &[u64], b: &[u64], out: &mut [u64]) {
    use crate::wide::mul;
    let l = B::LANES;
    let n = a.len() / (2 * l) * (2 * l);
    // SAFETY: the caller guarantees the backend's features; every index is below `n`.
    unsafe {
        for i in (0..n).step_by(2 * l) {
            let x0 = mul::<B>(B::load(a.as_ptr().add(i)), B::load(b.as_ptr().add(i)));
            let x1 = mul::<B>(B::load(a.as_ptr().add(i + l)), B::load(b.as_ptr().add(i + l)));
            B::store(out.as_mut_ptr().add(i), x0);
            B::store(out.as_mut_ptr().add(i + l), x1);
        }
    }
    scalar::fp_mul(&a[n..], &b[n..], &mut out[n..]);
}
#[cfg(target_arch = "x86_64")]
crate::wide::entry_points!(mul_wide => mul_w8, mul_w4; (a: &[u64], b: &[u64], out: &mut [u64]));
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
#[cfg(target_arch = "x86_64")]
unsafe fn vector(a: &[u64], b: &[u64], out: &mut [u64]) {
    // SAFETY: the target features are enabled here and the caller checked them at run time.
    // `fp_mul` asserts that `b` and `out` hold `a.len()` words, and every vector access ends
    // at or below `n <= a.len()`.
    unsafe {
        use crate::x86::*;
        let n = a.len() / 32 * 32;
        for i in (0..n).step_by(32) {
            // Four independent blocks let LLVM schedule across IFMA dependency chains.
            let x0 = mul(load(a.as_ptr().add(i)), load(b.as_ptr().add(i)));
            let x1 = mul(load(a.as_ptr().add(i + 8)), load(b.as_ptr().add(i + 8)));
            let x2 = mul(load(a.as_ptr().add(i + 16)), load(b.as_ptr().add(i + 16)));
            let x3 = mul(load(a.as_ptr().add(i + 24)), load(b.as_ptr().add(i + 24)));
            store(out.as_mut_ptr().add(i), x0);
            store(out.as_mut_ptr().add(i + 8), x1);
            store(out.as_mut_ptr().add(i + 16), x2);
            store(out.as_mut_ptr().add(i + 24), x3);
        }
        scalar::fp_mul(&a[n..], &b[n..], &mut out[n..]);
    }
}
