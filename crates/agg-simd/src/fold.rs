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
        unsafe {
            return vector(r, a, b, out);
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
        unsafe {
            return int_vector(r, lo, hi, out);
        }
    }
    scalar::int_fold(r, lo, hi, out);
}
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
#[cfg(target_arch = "x86_64")]
unsafe fn vector(r: K, a: Planes<'_>, b: Planes<'_>, out: PlanesMut<'_>) {
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
