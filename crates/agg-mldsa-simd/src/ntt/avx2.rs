//! AVX2 implementation of NTT operations.

use super::{Form, Scale};
use crate::params::Prime;
use crate::poly::Poly32;

#[target_feature(enable = "avx2,popcnt")]
pub(crate) unsafe fn ntt(p: Prime, a: &mut Poly32) {
    super::scalar::ntt(p, a)
}

#[target_feature(enable = "avx2,popcnt")]
pub(crate) unsafe fn intt(p: Prime, a: &mut Poly32, s: Scale, f: Form) {
    super::scalar::intt(p, a, s, f)
}

#[target_feature(enable = "avx2,popcnt")]
pub(crate) unsafe fn dot4(p: Prime, a: &[Poly32; 4], b: &[Poly32; 4], out: &mut Poly32) {
    super::scalar::dot4(p, a, b, out)
}

#[target_feature(enable = "avx2,popcnt")]
pub(crate) unsafe fn reduce(p: Prime, a: &mut Poly32, f: Form) {
    super::scalar::reduce(p, a, f)
}

#[target_feature(enable = "avx2,popcnt")]
pub(crate) unsafe fn order_from_fips(a: &mut Poly32) {
    super::scalar::order_from_fips(a)
}

#[target_feature(enable = "avx2,popcnt")]
pub(crate) unsafe fn order_to_fips(a: &mut Poly32) {
    super::scalar::order_to_fips(a)
}
