//! Independent Jolt arithmetic oracle for slice kernels. Inputs must be canonical.
use crate::{F, K};
use jolt_field::{Ring, Zero};
pub fn fp_mul(a: &[u64], b: &[u64], out: &mut [u64]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    for ((&a, &b), o) in a.iter().zip(b).zip(out) {
        *o = (F::from_u64(a) * F::from_u64(b)).to_canonical_u64();
    }
}
pub fn fp_dot(a: &[u64], b: &[u64]) -> u64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(F::zero(), |s, (&a, &b)| s + F::from_u64(a) * F::from_u64(b))
        .to_canonical_u64()
}
pub fn k_mul(a: (&[u64], &[u64]), b: (&[u64], &[u64]), out: (&mut [u64], &mut [u64])) {
    let n = a.0.len();
    assert_eq!(a.1.len(), n);
    assert_eq!(b.0.len(), n);
    assert_eq!(b.1.len(), n);
    assert_eq!(out.0.len(), n);
    assert_eq!(out.1.len(), n);
    for i in 0..n {
        let z = get(a, i) * get(b, i);
        out.0[i] = z.c0().to_canonical_u64();
        out.1[i] = z.c1().to_canonical_u64();
    }
}
pub fn get(a: (&[u64], &[u64]), i: usize) -> K {
    K::new(F::from_u64(a.0[i]), F::from_u64(a.1[i]))
}
pub fn k_dot(a: (&[u64], &[u64]), b: (&[u64], &[u64])) -> K {
    let n = a.0.len();
    assert_eq!(a.1.len(), n);
    assert_eq!(b.0.len(), n);
    assert_eq!(b.1.len(), n);
    (0..n).fold(K::zero(), |s, i| s + get(a, i) * get(b, i))
}
pub fn k_base_dot(a: (&[u64], &[u64]), b: &[u64]) -> K {
    assert_eq!(a.0.len(), a.1.len());
    assert_eq!(a.0.len(), b.len());
    (0..b.len()).fold(K::zero(), |s, i| s + get(a, i) * K::from_u64(b[i]))
}
pub fn k_int_dot(a: (&[u64], &[u64]), b: &[i64]) -> K {
    assert_eq!(a.0.len(), a.1.len());
    assert_eq!(a.0.len(), b.len());
    (0..b.len()).fold(K::zero(), |s, i| s + get(a, i) * K::from_i64(b[i]))
}
pub fn k_fold(r: K, a: (&[u64], &[u64]), b: (&[u64], &[u64]), out: (&mut [u64], &mut [u64])) {
    let n = a.0.len();
    assert_eq!(a.1.len(), n);
    assert_eq!(b.0.len(), n);
    assert_eq!(b.1.len(), n);
    assert_eq!(out.0.len(), n);
    assert_eq!(out.1.len(), n);
    for i in 0..n {
        let x = get(a, i);
        let z = x + r * (get(b, i) - x);
        out.0[i] = z.c0().to_canonical_u64();
        out.1[i] = z.c1().to_canonical_u64();
    }
}
pub fn int_fold(r: K, lo: &[i64], hi: &[i64], out: (&mut [u64], &mut [u64])) {
    assert_eq!(lo.len(), hi.len());
    assert_eq!(lo.len(), out.0.len());
    assert_eq!(lo.len(), out.1.len());
    for i in 0..lo.len() {
        let x = K::from_i64(lo[i]);
        let z = x + r * (K::from_i64(hi[i]) - x);
        out.0[i] = z.c0().to_canonical_u64();
        out.1[i] = z.c1().to_canonical_u64();
    }
}

pub fn k_mul_base(a: (&[u64], &[u64]), b: &[u64], out: (&mut [u64], &mut [u64])) {
    fp_mul(a.0, b, out.0);
    fp_mul(a.1, b, out.1);
}
/// Two limbs represent the signed integer lo + 2^52 hi, with lo < 2^52.
pub fn k_int2_dot(a: (&[u64], &[u64]), lo: &[u64], hi: &[i64]) -> K {
    assert_eq!(a.0.len(), a.1.len());
    assert_eq!(a.0.len(), lo.len());
    assert_eq!(lo.len(), hi.len());
    let mut sum = K::zero();
    for i in 0..lo.len() {
        assert!(lo[i] < (1 << 52));
        sum += get(a, i) * (K::from_u64(lo[i]) + K::from_u64(1 << 52) * K::from_i64(hi[i]));
    }
    sum
}
