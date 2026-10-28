//! Negacyclic NTT of FIPS 204 (Algorithms 41 to 43), generic over an NTT-friendly prime.
//!
//! The FIPS instance is `NttPrime::fips()` (q = 8,380,417, zeta = 1753). The same butterfly
//! network works for any prime P < 2^31 with a primitive 512th root of unity; the CRT product
//! of `crate::crt` uses a second such prime. Arithmetic is `u64` with a Barrett reduction of the
//! products (below `2^62`) and conditional subtractions for the sums (exact; a `%` division
//! makes the NTTs the bulk of the instance and witness time); not constant time, not vectorized.

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
    /// The Barrett constant `floor(2^64 / p)`.
    barrett: u64,
    /// Per butterfly of the forward transform: `(zeta, floor(zeta 2^32 / p))` (Shoup).
    forward: [(u32, u32); N],
    /// Per butterfly of the inverse transform: `(p - zeta, floor((p - zeta) 2^32 / p))`.
    inverse: [(u32, u32); N],
    /// `(256^{-1}, floor(256^{-1} 2^32 / p))`.
    n_inv_shoup: (u32, u32),
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
        assert!(p < 1 << 31, "NttPrime requires p < 2^31 (Shoup remainders below 2^32)");
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
        let barrett = u64::try_from((1_u128 << 64) / u128::from(p)).expect("p > 1");
        let shoup = |z: u64| (z as u32, ((z << 32) / p) as u32);
        let forward = std::array::from_fn(|k| shoup(zetas[k]));
        let inverse = std::array::from_fn(|k| shoup((p - zetas[k]) % p));
        Self {
            p,
            zetas,
            n_inv,
            barrett,
            forward,
            inverse,
            n_inv_shoup: shoup(n_inv),
        }
    }

    /// `z x mod p` for a constant `z < p` with its Shoup quotient `zq = floor(z 2^32 / p)`
    /// and `x < p` (`p < 2^31`): the quotient estimate is short by at most one, so the 32-bit
    /// remainder lies in `[0, 2p)` and one conditional subtraction ends it (no 128-bit
    /// product, no division).
    #[inline(always)]
    fn shoup_mul(&self, x: u32, (z, zq): (u32, u32)) -> u32 {
        let p = self.p as u32;
        let q = ((u64::from(zq) * u64::from(x)) >> 32) as u32;
        let r = z.wrapping_mul(x).wrapping_sub(q.wrapping_mul(p));
        r.min(r.wrapping_sub(p))
    }

    /// `x mod p` for any `x < 2^64` (Barrett: the quotient estimate is short by at most
    /// one, so one conditional subtraction).
    #[inline(always)]
    pub fn reduce_u64(&self, x: u64) -> u64 {
        let quotient = ((u128::from(x) * u128::from(self.barrett)) >> 64) as u64;
        let r = x - quotient * self.p;
        // Branch free: `r - p` wraps above `r` when `r < p`.
        r.min(r.wrapping_sub(self.p))
    }

    /// `a b mod p` for `a, b < p`.
    #[inline(always)]
    pub fn mul_mod(&self, a: u64, b: u64) -> u64 {
        self.reduce_u64(a * b)
    }

    /// `a + b mod p` for `a, b < p`.
    #[inline(always)]
    pub fn add_mod(&self, a: u64, b: u64) -> u64 {
        let s = a + b;
        s.min(s.wrapping_sub(self.p))
    }

    /// `a - b mod p` for `a, b < p`.
    #[inline(always)]
    pub fn sub_mod(&self, a: u64, b: u64) -> u64 {
        let d = a.wrapping_sub(b);
        d.min(d.wrapping_add(self.p))
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
        let r = self.reduce_u64(x.unsigned_abs());
        if x >= 0 || r == 0 {
            r
        } else {
            self.p - r
        }
    }

    /// NTT (FIPS 204 Algorithm 41); input and output coefficients in [0, p).
    pub fn ntt(&self, w: &[u64; N]) -> [u64; N] {
        // 32-bit lanes (p < 2^31) with Shoup products by the constant zetas.
        let p = self.p as u32;
        let mut a: [u32; N] = std::array::from_fn(|i| w[i] as u32);
        let mut m = 0usize;
        let mut len = 128usize;
        while len >= 1 {
            let mut start = 0usize;
            while start < N {
                m += 1;
                let z = self.forward[m];
                let (low, high) = a[start..start + 2 * len].split_at_mut(len);
                for (x, y) in low.iter_mut().zip(high.iter_mut()) {
                    let t = self.shoup_mul(*y, z);
                    let d = x.wrapping_sub(t);
                    *y = d.min(d.wrapping_add(p));
                    let s = *x + t;
                    *x = s.min(s.wrapping_sub(p));
                }
                start += 2 * len;
            }
            len /= 2;
        }
        a.map(u64::from)
    }

    /// Inverse NTT (FIPS 204 Algorithm 42); input and output coefficients in [0, p).
    pub fn intt(&self, w_hat: &[u64; N]) -> [u64; N] {
        let p = self.p as u32;
        let mut a: [u32; N] = std::array::from_fn(|i| w_hat[i] as u32);
        let mut m = 256usize;
        let mut len = 1usize;
        while len < N {
            let mut start = 0usize;
            while start < N {
                m -= 1;
                let z = self.inverse[m];
                let (low, high) = a[start..start + 2 * len].split_at_mut(len);
                for (x, y) in low.iter_mut().zip(high.iter_mut()) {
                    let t = *x;
                    let s = t + *y;
                    *x = s.min(s.wrapping_sub(p));
                    let d = t.wrapping_sub(*y);
                    *y = self.shoup_mul(d.min(d.wrapping_add(p)), z);
                }
                start += 2 * len;
            }
            len *= 2;
        }
        a.map(|x| u64::from(self.shoup_mul(x, self.n_inv_shoup)))
    }

    /// Pointwise product in the NTT domain (FIPS 204 Algorithm 45).
    pub fn pointwise(&self, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        let mut c = [0u64; N];
        for i in 0..N {
            c[i] = self.mul_mod(a[i], b[i]);
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
        let mut rng = SplitMix64::new(7);
        for (t, _) in [NttPrime::fips(), crate::crt::prime2()].into_iter().flat_map(|t| [(t, 0); 4]) {
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
    fn barrett_reduction_matches_the_remainder() {
        let mut rng = SplitMix64::new(13);
        for t in [NttPrime::fips(), crate::crt::prime2()] {
            for x in [0_u64, 1, t.p - 1, t.p, t.p + 1, u64::MAX, u64::MAX - 1, 1 << 63] {
                assert_eq!(t.reduce_u64(x), x % t.p);
            }
            for _ in 0..10_000 {
                let x = rng.next_u64();
                assert_eq!(t.reduce_u64(x), x % t.p);
                let y = rng.next_u64() as i64;
                assert_eq!(t.reduce(y), y.rem_euclid(t.p as i64) as u64);
                let (a, b) = (x % t.p, rng.next_u64() % t.p);
                assert_eq!(t.mul_mod(a, b), a * b % t.p);
                assert_eq!(t.add_mod(a, b), (a + b) % t.p);
                assert_eq!(t.sub_mod(a, b), (a + t.p - b) % t.p);
            }
            assert_eq!(t.reduce(i64::MIN), i64::MIN.rem_euclid(t.p as i64) as u64);
        }
    }

    /// `RUSTUP_TOOLCHAIN=stable taskset -c 0,1,3-15 cargo test -p agg-witness --release --lib
    /// ntt::tests::ntt_timing -- --ignored --nocapture`
    #[test]
    #[ignore = "timing"]
    fn ntt_timing() {
        let t = NttPrime::fips();
        let mut rng = SplitMix64::new(5);
        let mut a = [0u64; N];
        for x in a.iter_mut() {
            *x = rng.next_u64() % t.p;
        }
        let start = std::time::Instant::now();
        for _ in 0..20_000 {
            a = t.intt(&t.ntt(&a));
        }
        println!("ntt + intt: {:.2} us", start.elapsed().as_secs_f64() * 1e6 / 20_000.0);
        // The `%` butterflies of the reference code, for comparison.
        let p = t.p;
        let start = std::time::Instant::now();
        for _ in 0..20_000 {
            let (mut m, mut len) = (0, 128);
            while len >= 1 {
                let mut start = 0;
                while start < N {
                    m += 1;
                    let z = t.zetas[m];
                    for j in start..start + len {
                        let u = z * a[j + len] % p;
                        a[j + len] = (a[j] + p - u) % p;
                        a[j] = (a[j] + u) % p;
                    }
                    start += 2 * len;
                }
                len /= 2;
            }
        }
        println!("ntt with %: {:.2} us", start.elapsed().as_secs_f64() * 1e6 / 20_000.0);
        assert!(a.iter().all(|&x| x < t.p));
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
