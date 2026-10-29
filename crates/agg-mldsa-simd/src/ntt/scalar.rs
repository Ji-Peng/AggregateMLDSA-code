//! Scalar implementation of NTT operations.

use super::{Form, Scale};
use crate::params::Prime;
use crate::poly::Poly32;

pub(crate) fn ntt(p: Prime, a: &mut Poly32) {
    let _ = (p, a);
    todo!("T2")
}

pub(crate) fn intt(p: Prime, a: &mut Poly32, s: Scale, f: Form) {
    let _ = (p, a, s, f);
    todo!("T2")
}

pub(crate) fn dot4(p: Prime, a: &[Poly32; 4], b: &[Poly32; 4], out: &mut Poly32) {
    let _ = (p, a, b, out);
    todo!("T2")
}

pub(crate) fn reduce(p: Prime, a: &mut Poly32, f: Form) {
    let _ = (p, a, f);
    todo!("T2")
}

pub(crate) fn order_from_fips(a: &mut Poly32) {
    let _ = a;
    todo!("T2")
}

pub(crate) fn order_to_fips(a: &mut Poly32) {
    let _ = a;
    todo!("T2")
}
