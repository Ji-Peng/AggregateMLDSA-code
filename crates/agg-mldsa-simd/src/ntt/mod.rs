//! Number Theoretic Transform (NTT), inverse NTT, and polynomial arithmetic.

pub(crate) mod avx2;
pub(crate) mod avx512;
pub(crate) mod scalar;

use crate::dispatch::{Backend, Engine};
use crate::params::{ParamSet, Prime};
use crate::poly::Poly32;

/// Scale factor applied during inverse NTT.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scale {
    /// Scale by R / 256 mod p (input of scale 1).
    Plain,
    /// Scale by R^2 / 256 mod p (Montgomery scale).
    Mont,
}

/// Output representative form for NTT operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    /// Canonical representative in [0, p).
    Canonical,
    /// Centered representative in [-(p - 1) / 2, (p - 1) / 2].
    Centered,
}

impl<P: ParamSet> Engine<P> {
    /// Forward NTT: natural order with |a| <= 2^22 to backend order.
    pub fn ntt(&self, p: Prime, a: &mut Poly32) {
        match self.backend() {
            Backend::Scalar => scalar::ntt(p, a),
            Backend::Avx2 => {
                // SAFETY: The engine guarantees AVX2 and POPCNT features are available.
                unsafe { avx2::ntt(p, a) }
            }
            Backend::Avx512 => {
                // SAFETY: The engine guarantees AVX-512 features are available.
                unsafe { avx512::ntt(p, a) }
            }
        }
    }

    /// Inverse NTT: backend order to natural order with scaling and reduction.
    pub fn intt(&self, p: Prime, a: &mut Poly32, s: Scale, f: Form) {
        match self.backend() {
            Backend::Scalar => scalar::intt(p, a, s, f),
            Backend::Avx2 => {
                // SAFETY: The engine guarantees AVX2 and POPCNT features are available.
                unsafe { avx2::intt(p, a, s, f) }
            }
            Backend::Avx512 => {
                // SAFETY: The engine guarantees AVX-512 features are available.
                unsafe { avx512::intt(p, a, s, f) }
            }
        }
    }

    /// L-vector inner product in NTT domain: out = sum(a[i] o b[i]) / R.
    pub fn dot_l<const L: usize>(
        &self,
        p: Prime,
        a: &[Poly32; L],
        b: &[Poly32; L],
        out: &mut Poly32,
    ) {
        match self.backend() {
            Backend::Scalar => scalar::dot_l(p, a, b, out),
            Backend::Avx2 => {
                // SAFETY: The engine guarantees AVX2 and POPCNT features are available.
                unsafe { avx2::dot_l(p, a, b, out) }
            }
            Backend::Avx512 => {
                // SAFETY: The engine guarantees AVX-512 features are available.
                unsafe { avx512::dot_l(p, a, b, out) }
            }
        }
    }

    /// 4-vector inner product in NTT domain: out = sum(a[i] o b[i]) / R.
    pub fn dot4(&self, p: Prime, a: &[Poly32; 4], b: &[Poly32; 4], out: &mut Poly32) {
        self.dot_l(p, a, b, out)
    }

    /// Reduces arbitrary i32 coefficients to canonical [0, p) or centered form.
    pub fn reduce(&self, p: Prime, a: &mut Poly32, f: Form) {
        match self.backend() {
            Backend::Scalar => scalar::reduce(p, a, f),
            Backend::Avx2 => {
                // SAFETY: The engine guarantees AVX2 and POPCNT features are available.
                unsafe { avx2::reduce(p, a, f) }
            }
            Backend::Avx512 => {
                // SAFETY: The engine guarantees AVX-512 features are available.
                unsafe { avx512::reduce(p, a, f) }
            }
        }
    }

    /// Permutes coefficients from FIPS 204 order to backend order.
    pub fn order_from_fips(&self, a: &mut Poly32) {
        match self.backend() {
            Backend::Scalar => scalar::order_from_fips(a),
            Backend::Avx2 => {
                // SAFETY: The engine guarantees AVX2 and POPCNT features are available.
                unsafe { avx2::order_from_fips(a) }
            }
            Backend::Avx512 => {
                // SAFETY: The engine guarantees AVX-512 features are available.
                unsafe { avx512::order_from_fips(a) }
            }
        }
    }

    /// Permutes coefficients from backend order to FIPS 204 order.
    pub fn order_to_fips(&self, a: &mut Poly32) {
        match self.backend() {
            Backend::Scalar => scalar::order_to_fips(a),
            Backend::Avx2 => {
                // SAFETY: The engine guarantees AVX2 and POPCNT features are available.
                unsafe { avx2::order_to_fips(a) }
            }
            Backend::Avx512 => {
                // SAFETY: The engine guarantees AVX-512 features are available.
                unsafe { avx512::order_to_fips(a) }
            }
        }
    }
}
