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
        unsafe {
            return vector(a, b, out);
        }
    }
    scalar::fp_mul(a, b, out);
}
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
#[cfg(target_arch = "x86_64")]
unsafe fn vector(a: &[u64], b: &[u64], out: &mut [u64]) {
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
