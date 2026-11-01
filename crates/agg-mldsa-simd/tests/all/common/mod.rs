//! Common test helpers, oracles, and test RNG.

#![allow(dead_code, unused_imports)]

use agg_mldsa_simd::dispatch::{Backend, Engine};
use agg_mldsa_simd::params::{consts, Prime};
use agg_mldsa_simd::poly::Poly32;

/// Deterministic 64-bit pseudo-random number generator.
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    /// Initializes the generator with a 64-bit seed.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Generates the next pseudo-random `u64`.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Fills the destination buffer with pseudo-random bytes.
    pub fn fill_bytes(&mut self, dest: &mut [u8]) {
        for chunk in dest.chunks_mut(8) {
            let val = self.next_u64().to_le_bytes();
            let len = chunk.len();
            chunk.copy_from_slice(&val[..len]);
        }
    }
}

/// Returns an `Engine` for every backend supported on the current host.
pub fn engines() -> Vec<Engine> {
    [Backend::Scalar, Backend::Avx2, Backend::Avx512]
        .into_iter()
        .filter_map(Engine::new)
        .collect()
}

/// Creates an `agg_witness::ntt::NttPrime` instance for prime `p` in FIPS order.
pub fn oracle_ntt_prime(p: Prime) -> agg_witness::ntt::NttPrime {
    let c = consts(p);
    agg_witness::ntt::NttPrime::new(c.p as u64, c.zeta as u64)
}

/// Converts a `Poly32` to an array of canonical `u64` coefficients in `[0, p)`.
pub fn poly32_to_u64(poly: &Poly32, p: i32) -> [u64; 256] {
    let mut out = [0u64; 256];
    for (o, &c) in out.iter_mut().zip(poly.iter()) {
        let m = (c as i64).rem_euclid(p as i64);
        *o = m as u64;
    }
    out
}

/// Asserts that two polynomials are congruent modulo `p`.
pub fn assert_poly_eq_mod_p(a: &Poly32, b: &Poly32, p: i32) {
    for i in 0..256 {
        let va = (a.0[i] as i64).rem_euclid(p as i64);
        let vb = (b.0[i] as i64).rem_euclid(p as i64);
        assert_eq!(
            va, vb,
            "Coefficient mismatch at index {i}: {} vs {} (mod {})",
            a.0[i], b.0[i], p
        );
    }
}

/// Computes polynomial product modulo (X^256 + 1) and modulo `p` using schoolbook multiplication.
pub fn negacyclic_schoolbook_mod(a: &[i32; 256], b: &[i32; 256], p: i32) -> [i32; 256] {
    let p_i128 = p as i128;
    let mut c = [0i128; 512];
    for i in 0..256 {
        for j in 0..256 {
            c[i + j] += (a[i] as i128) * (b[j] as i128);
        }
    }
    let mut out = [0i32; 256];
    for i in 0..256 {
        let val = (c[i] - c[i + 256]).rem_euclid(p_i128);
        out[i] = val as i32;
    }
    out
}

/// Computes SHAKE128 over a list of input byte segments.
pub fn shake128_segments(segments: &[&[u8]], out: &mut [u8]) {
    use sha3::digest::{ExtendableOutput, Update, XofReader};
    let mut hasher = sha3::Shake128::default();
    for seg in segments {
        hasher.update(seg);
    }
    let mut reader = hasher.finalize_xof();
    reader.read(out);
}

/// Computes SHAKE256 over a list of input byte segments.
pub fn shake256_segments(segments: &[&[u8]], out: &mut [u8]) {
    use sha3::digest::{ExtendableOutput, Update, XofReader};
    let mut hasher = sha3::Shake256::default();
    for seg in segments {
        hasher.update(seg);
    }
    let mut reader = hasher.finalize_xof();
    reader.read(out);
}

/// Re-exports of `agg_witness::native` reference algorithms.
pub mod native {
    pub use agg_witness::native::{
        ctr_q, sample_in_ball, sig_decode, use_hint, w1_encode, Challenge, DecodedSig,
    };
}

/// Re-exports of `agg_witness::expand` public-key expansion routines.
pub mod expand {
    pub use agg_witness::expand::{expand_a, public_t, public_t_into};
}
