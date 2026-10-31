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
        // SAFETY: the caller checked the IFMA backend, and every slice or pointer is valid for the length asserted above.
        unsafe {
            return vector(a, b, out);
        }
    }
    scalar::k_mul(a, b, out);
}
pub fn k_mul_base(a: Planes<'_>, b: &[u64], out: PlanesMut<'_>) {
    crate::fp::fp_mul(a.0, b, out.0);
    crate::fp::fp_mul(a.1, b, out.1);
}
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
#[cfg(target_arch = "x86_64")]
unsafe fn vector(a: Planes<'_>, b: Planes<'_>, out: PlanesMut<'_>) {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
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
