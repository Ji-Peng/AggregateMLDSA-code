//! CRT product and representative choices for the ring identity (spec Section 3, Appendix B.9).
//!
//! The ring identity (3.1) is over the integers, so the prover needs (A z) in Z[X]/(X^256 + 1)
//! with A centered (Appendix B.9). With |A| <= (q-1)/2 and |z| <= 2^20 every coefficient of A z
//! is below l 256 (q-1)/2 2^20 < 2^52.1 < M/2 for M = q P2 (about 2^53.9), so the product is
//! recovered exactly from its residues modulo q (FIPS NTT with the key's A_hat) and modulo
//! P2 = 15 2^27 + 1 (NTT with a 512th root of unity), by CRT with the centered representative.

use crate::expand::ExpandedKey;
use crate::native::{Poly, VecK, VecL};
use crate::ntt::{pow_mod, NttPrime};
use crate::params::{K, L, N, Q};
use std::sync::OnceLock;

/// Second CRT prime P2 = 15 2^27 + 1 (31 bits, 2-adicity 27).
pub const P2: u64 = 2_013_265_921;
/// Generator of the multiplicative group modulo P2.
pub const P2_GENERATOR: u64 = 31;
/// Largest |z| for which the CRT product is exact.
pub const Z_CRT_MAX: i64 = 1 << 20;

/// The NTT tables for P2 with zeta = 31^((P2 - 1)/512).
pub fn prime2() -> &'static NttPrime {
    static P: OnceLock<NttPrime> = OnceLock::new();
    P.get_or_init(|| NttPrime::new(P2, pow_mod(P2_GENERATOR, (P2 - 1) / 512, P2)))
}

/// CRT: the integer x in (-M/2, M/2] with x = x1 mod q and x = x2 mod P2, M = q P2.
#[inline]
pub fn crt_centered(x1: u64, x2: u64) -> i64 {
    let q = Q as u64;
    static Q_INV: OnceLock<u64> = OnceLock::new();
    let q_inv = *Q_INV.get_or_init(|| pow_mod(q % P2, P2 - 2, P2));
    let diff = (x2 + P2 - x1 % P2) % P2;
    let k = (diff as u128 * q_inv as u128 % P2 as u128) as u64;
    let m = q as i128 * P2 as i128;
    let mut x = x1 as i128 + q as i128 * k as i128;
    if x > m / 2 {
        x -= m;
    }
    x as i64
}

/// (A z) over the integers by CRT (spec (3.1)); requires |z|_inf <= 2^20.
pub fn az_crt(key: &ExpandedKey, z: &VecL) -> VecK {
    assert!(
        z.iter()
            .flat_map(|p| p.iter())
            .all(|x| x.abs() <= Z_CRT_MAX),
        "CRT product requires |z| <= 2^20"
    );
    let fips = NttPrime::fips();
    let p2 = prime2();
    let z1: Vec<[u64; N]> = z.iter().map(|zi| fips.ntt_signed(zi)).collect();
    let z2: Vec<[u64; N]> = z.iter().map(|zi| p2.ntt_signed(zi)).collect();
    let mut out = [[0i64; N]; K];
    for r in 0..K {
        let mut acc1 = [0u64; N];
        let mut acc2 = [0u64; N];
        for s in 0..L {
            let a1 = fips.pointwise(&key.a_hat[r][s], &z1[s]);
            let a2 = p2.pointwise(&key.a_hat_p2[r][s], &z2[s]);
            for j in 0..N {
                acc1[j] = (acc1[j] + a1[j]) % fips.p;
                acc2[j] = (acc2[j] + a2[j]) % p2.p;
            }
        }
        let x1 = fips.intt(&acc1);
        let x2 = p2.intt(&acc2);
        for j in 0..N {
            out[r][j] = crt_centered(x1[j], x2[j]);
        }
    }
    out
}

/// Negacyclic product a * b over the integers by schoolbook (i128 accumulation).
pub fn negacyclic_schoolbook(a: &[i64; N], b: &[i64; N]) -> [i128; N] {
    let mut c = [0i128; N];
    for i in 0..N {
        if a[i] == 0 {
            continue;
        }
        for j in 0..N {
            let prod = a[i] as i128 * b[j] as i128;
            let k = i + j;
            if k < N {
                c[k] += prod;
            } else {
                c[k - N] -= prod;
            }
        }
    }
    c
}

/// (A z) over the integers by schoolbook, for any z (reference and fallback); i128 entries.
pub fn az_schoolbook(a: &crate::expand::MatrixCoeff, z: &VecL) -> [[i128; N]; K] {
    let mut out = [[0i128; N]; K];
    for r in 0..K {
        for s in 0..L {
            let mut ar: Poly = [0; N];
            for j in 0..N {
                ar[j] = i64::from(a[r][s][j]);
            }
            let prod = negacyclic_schoolbook(&ar, &z[s]);
            for j in 0..N {
                out[r][j] += prod[j];
            }
        }
    }
    out
}

/// (A z) over the integers: CRT when |z| <= 2^20, schoolbook otherwise.
pub fn az_integer(key: &ExpandedKey, z: &VecL) -> [[i128; N]; K] {
    if z.iter()
        .flat_map(|p| p.iter())
        .all(|x| x.abs() <= Z_CRT_MAX)
    {
        let v = az_crt(key, z);
        let mut out = [[0i128; N]; K];
        for r in 0..K {
            for j in 0..N {
                out[r][j] = i128::from(v[r][j]);
            }
        }
        out
    } else {
        az_schoolbook(&key.a, z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gen::{self, SplitMix64};

    #[test]
    fn prime2_tables() {
        let p = prime2();
        assert_eq!(P2, 15 * (1 << 27) + 1);
        assert_eq!(
            pow_mod(P2_GENERATOR, (P2 - 1) / 2, P2),
            P2 - 1,
            "31 is a non-residue"
        );
        assert_eq!(
            pow_mod(p.zetas[1], 2, P2),
            P2 - 1,
            "zetas[1] = zeta^128 is a 4th root of unity"
        );
        // Exactness margin: l 256 (q-1)/2 2^20 < M/2.
        let bound = (L * N) as i128 * ((Q - 1) / 2) as i128 * Z_CRT_MAX as i128;
        assert!(bound < (Q as i128 * P2 as i128) / 2);
    }

    #[test]
    fn crt_centered_recovers_signed_values() {
        let m = Q as i128 * P2 as i128;
        for x in [0i128, 1, -1, 123_456_789_012, -(m / 2) + 1, m / 2] {
            let x1 = x.rem_euclid(Q as i128) as u64;
            let x2 = x.rem_euclid(P2 as i128) as u64;
            assert_eq!(crt_centered(x1, x2) as i128, x);
        }
    }

    #[test]
    fn az_crt_matches_schoolbook() {
        let r = gen::record(4, 0);
        let key = ExpandedKey::from_bytes(&r.pk).unwrap();
        let mut rng = SplitMix64::new(17);
        for bound in [1i64 << 17, Z_CRT_MAX] {
            let mut z = [[0i64; N]; L];
            for zi in z.iter_mut() {
                for x in zi.iter_mut() {
                    *x = rng.below(2 * bound as u64 + 1) as i64 - bound;
                }
            }
            // Extremes in one coefficient.
            z[0][0] = bound;
            z[3][255] = -bound;
            let crt = az_crt(&key, &z);
            let sb = az_schoolbook(&key.a, &z);
            for row in 0..K {
                for j in 0..N {
                    assert_eq!(i128::from(crt[row][j]), sb[row][j]);
                }
            }
        }
    }
}
