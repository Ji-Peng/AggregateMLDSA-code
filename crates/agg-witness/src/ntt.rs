//! Negacyclic NTT of FIPS 204 (Algorithms 41 to 43), generic over an NTT-friendly prime.
//!
//! The FIPS instance is `NttPrime::fips()` (q = 8,380,417, zeta = 1753). The same butterfly
//! network works for any prime P < 2^31 with a primitive 512th root of unity; the CRT product
//! of `crate::crt` uses a second such prime. Arithmetic is plain `u64` with `%` reduction; this
//! is reference code, not a constant-time or vectorized implementation.

use crate::params::{N, N_INV_Q, Q, ZETA};
use std::sync::OnceLock;

/// An NTT-friendly prime with its zetas table (zetas[k] = zeta^BitRev8(k) mod P).
#[derive(Clone, Debug)]
pub struct NttPrime {
    /// The prime modulus.
    pub p: u64,
    /// zetas[0..256], zetas[k] = zeta^BitRev8(k) mod p (FIPS 204 Appendix B for q).
    pub zetas: [u64; N],
    /// 256^{-1} mod p.
    pub n_inv: u64,
}

/// BitRev8 (FIPS 204 Algorithm 43).
pub fn bitrev8(m: u8) -> u8 {
    m.reverse_bits()
}

/// x^e mod p.
pub fn pow_mod(mut x: u64, mut e: u64, p: u64) -> u64 {
    let mut r = 1u64;
    x %= p;
    while e > 0 {
        if e & 1 == 1 {
            r = r * x % p;
        }
        x = x * x % p;
        e >>= 1;
    }
    r
}

impl NttPrime {
    /// Builds the table for prime `p` and a primitive 512th root of unity `zeta` (zeta^256 = -1).
    pub fn new(p: u64, zeta: u64) -> Self {
        assert!(p < 1 << 31, "NttPrime requires p < 2^31");
        assert_eq!(
            pow_mod(zeta, 256, p),
            p - 1,
            "zeta must satisfy zeta^256 = -1"
        );
        let mut zetas = [0u64; N];
        for (k, z) in zetas.iter_mut().enumerate() {
            *z = pow_mod(zeta, u64::from(bitrev8(k as u8)), p);
        }
        let n_inv = pow_mod(256, p - 2, p);
        Self { p, zetas, n_inv }
    }

    /// The FIPS 204 instance (q, zeta = 1753).
    pub fn fips() -> &'static NttPrime {
        static FIPS: OnceLock<NttPrime> = OnceLock::new();
        FIPS.get_or_init(|| {
            let t = NttPrime::new(Q as u64, ZETA as u64);
            debug_assert_eq!(t.n_inv, N_INV_Q as u64);
            t
        })
    }

    /// Reduces a signed integer into [0, p).
    #[inline]
    pub fn reduce(&self, x: i64) -> u64 {
        x.rem_euclid(self.p as i64) as u64
    }

    /// NTT (FIPS 204 Algorithm 41); input and output coefficients in [0, p).
    pub fn ntt(&self, w: &[u64; N]) -> [u64; N] {
        let p = self.p;
        let mut a = *w;
        let mut m = 0usize;
        let mut len = 128usize;
        while len >= 1 {
            let mut start = 0usize;
            while start < N {
                m += 1;
                let z = self.zetas[m];
                for j in start..start + len {
                    let t = z * a[j + len] % p;
                    a[j + len] = (a[j] + p - t) % p;
                    a[j] = (a[j] + t) % p;
                }
                start += 2 * len;
            }
            len /= 2;
        }
        a
    }

    /// Inverse NTT (FIPS 204 Algorithm 42); input and output coefficients in [0, p).
    pub fn intt(&self, w_hat: &[u64; N]) -> [u64; N] {
        let p = self.p;
        let mut a = *w_hat;
        let mut m = 256usize;
        let mut len = 1usize;
        while len < N {
            let mut start = 0usize;
            while start < N {
                m -= 1;
                let z = p - self.zetas[m];
                for j in start..start + len {
                    let t = a[j];
                    a[j] = (t + a[j + len]) % p;
                    a[j + len] = (t + p - a[j + len]) % p;
                    a[j + len] = z * a[j + len] % p;
                }
                start += 2 * len;
            }
            len *= 2;
        }
        for x in a.iter_mut() {
            *x = self.n_inv * *x % p;
        }
        a
    }

    /// Pointwise product in the NTT domain (FIPS 204 Algorithm 45).
    pub fn pointwise(&self, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        let mut c = [0u64; N];
        for i in 0..N {
            c[i] = a[i] * b[i] % self.p;
        }
        c
    }

    /// NTT of a signed-coefficient polynomial.
    pub fn ntt_signed(&self, w: &[i64; N]) -> [u64; N] {
        let mut a = [0u64; N];
        for i in 0..N {
            a[i] = self.reduce(w[i]);
        }
        self.ntt(&a)
    }
}

/// Negacyclic product a * b mod (X^256 + 1, p) by schoolbook (reference for tests).
pub fn negacyclic_schoolbook_mod(a: &[u64; N], b: &[u64; N], p: u64) -> [u64; N] {
    let mut c = [0u64; N];
    for i in 0..N {
        for j in 0..N {
            let prod = a[i] * b[j] % p;
            let k = i + j;
            if k < N {
                c[k] = (c[k] + prod) % p;
            } else {
                c[k - N] = (c[k - N] + p - prod) % p;
            }
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gen::SplitMix64;

    #[test]
    fn zetas_match_fips_appendix_b() {
        let t = NttPrime::fips();
        // First row and two later entries of FIPS 204 Appendix B.
        let expected = [
            0u64, 4808194, 3765607, 3761513, 5178923, 5496691, 5234739, 5178987,
        ];
        assert_eq!(&t.zetas[1..8], &expected[1..]);
        assert_eq!(t.zetas[8], 7778734);
        assert_eq!(t.zetas[22], 676590);
        // zetas[0] = zeta^0 = 1 by the formula; FIPS lists 0 there because index 0 is unused.
        assert_eq!(t.zetas[0], 1);
    }

    #[test]
    fn ntt_roundtrip_and_convolution() {
        let t = NttPrime::fips();
        let mut rng = SplitMix64::new(7);
        for _ in 0..4 {
            let mut a = [0u64; N];
            let mut b = [0u64; N];
            for i in 0..N {
                a[i] = rng.next_u64() % t.p;
                b[i] = rng.next_u64() % t.p;
            }
            assert_eq!(t.intt(&t.ntt(&a)), a);
            let c = t.intt(&t.pointwise(&t.ntt(&a), &t.ntt(&b)));
            assert_eq!(c, negacyclic_schoolbook_mod(&a, &b, t.p));
        }
    }

    #[test]
    fn ntt_evaluates_at_odd_powers() {
        // NTT(w)[i] = w(zeta^(2 BitRev8(i) + 1)) (FIPS 204 (7.1)).
        let t = NttPrime::fips();
        let mut rng = SplitMix64::new(11);
        let mut w = [0u64; N];
        for x in w.iter_mut() {
            *x = rng.next_u64() % t.p;
        }
        let w_hat = t.ntt(&w);
        for i in [0usize, 1, 77, 255] {
            let x = pow_mod(ZETA as u64, 2 * u64::from(bitrev8(i as u8)) + 1, t.p);
            let mut acc = 0u64;
            for j in (0..N).rev() {
                acc = (acc * x + w[j]) % t.p;
            }
            assert_eq!(w_hat[i], acc);
        }
    }
}
