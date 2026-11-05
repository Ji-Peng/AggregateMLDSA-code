//! CRT product and representative choices for the ring identity.
//!
//! The ring identity is over the integers, so the prover needs (A z) in Z[X]/(X^256 + 1)
//! with A centered. With |A| <= (q-1)/2 and |z| <= 2^20 every coefficient of A z
//! is below l 256 (q-1)/2 2^20, which is 2^52.0 for l = 4, 2^52.32 for l = 5 and 2^52.81 for
//! l = 7, all below M/2 = 2^52.9 for M = q CRT_P31 (about 2^53.9). The product is therefore
//! recovered exactly from its residues modulo q (FIPS NTT with the key's A_hat) and modulo
//! CRT_P31 = 15 2^27 + 1 (NTT with a 512th root of unity), by CRT with the centered
//! representative. CRT_P31 is not the 28-bit transform prime `P2` of `agg_mldsa_simd`.

use crate::expand::ExpandedKey;
use crate::native::{Poly, VecK, VecL};
use crate::ntt::{pow_mod, NttPrime};
use crate::params::{ParamSet, Set44, K, L, N, Q};
use std::sync::OnceLock;

/// Second CRT prime CRT_P31 = 15 2^27 + 1 (31 bits, 2-adicity 27).
pub const CRT_P31: u64 = 2_013_265_921;
/// Generator of the multiplicative group modulo CRT_P31.
pub const CRT_P31_GENERATOR: u64 = 31;
/// Largest |z| for which the CRT product is exact.
pub const Z_CRT_MAX: i64 = 1 << 20;

/// The NTT tables for CRT_P31 with zeta = 31^((CRT_P31 - 1)/512).
pub fn prime2() -> &'static NttPrime {
    static P: OnceLock<NttPrime> = OnceLock::new();
    P.get_or_init(|| {
        NttPrime::new(
            CRT_P31,
            pow_mod(CRT_P31_GENERATOR, (CRT_P31 - 1) / 512, CRT_P31),
        )
    })
}

/// CRT: the integer x in (-M/2, M/2] with x = x1 mod q and x = x2 mod CRT_P31, M = q CRT_P31.
#[inline]
pub fn crt_centered(x1: u64, x2: u64) -> i64 {
    let q = Q as u64;
    static Q_INV: OnceLock<u64> = OnceLock::new();
    let q_inv = *Q_INV.get_or_init(|| pow_mod(q % CRT_P31, CRT_P31 - 2, CRT_P31));
    let diff = (x2 + CRT_P31 - x1 % CRT_P31) % CRT_P31;
    let k = (diff as u128 * q_inv as u128 % CRT_P31 as u128) as u64;
    let m = q as i128 * CRT_P31 as i128;
    let mut x = x1 as i128 + q as i128 * k as i128;
    if x > m / 2 {
        x -= m;
    }
    x as i64
}

/// (A z) over the integers by CRT for parameter set P; requires |z|_inf <= 2^20.
pub fn az_crt_for<P: ParamSet>(key: &ExpandedKey<P>, z: &P::VecL) -> P::VecK {
    let fips = NttPrime::fips();
    let z1: Vec<[u64; N]> = (0..P::L).map(|s| fips.ntt_signed(&z[s])).collect();
    let mut az_hat = P::zero_vec_k_u64();
    let a_hat = key.a_hat();
    for r in 0..P::K {
        let mut acc = [0u64; N];
        for s in 0..P::L {
            let a_poly = P::get_ntt(&a_hat, r, s);
            let a1 = fips.pointwise(a_poly, &z1[s]);
            for j in 0..N {
                acc[j] = fips.add_mod(acc[j], a1[j]);
            }
        }
        az_hat[r] = acc;
    }
    az_crt_with_for::<P>(key, z, &az_hat)
}

/// (A z) over the integers by CRT; requires |z|_inf <= 2^20.
pub fn az_crt(key: &ExpandedKey, z: &VecL) -> VecK {
    az_crt_for::<Set44>(key, z)
}

/// [`az_crt_for`] with the residue modulo q already in the NTT domain. Requires |z|_inf <= 2^20.
pub fn az_crt_with_for<P: ParamSet>(key: &ExpandedKey<P>, z: &P::VecL, az_hat: &P::VecKU64) -> P::VecK {
    assert!(
        (0..P::L).all(|s| z[s].iter().all(|x| x.abs() <= Z_CRT_MAX)),
        "CRT product requires |z| <= 2^20"
    );
    let fips = NttPrime::fips();
    let p2 = prime2();
    let z2: Vec<[u64; N]> = (0..P::L).map(|s| p2.ntt_signed(&z[s])).collect();
    let mut out = P::zero_vec_k();
    for r in 0..P::K {
        let mut acc2 = [0u64; N];
        for s in 0..P::L {
            let a_coeff = P::get_coeff(&key.a, r, s);
            let a2_hat = p2.ntt_signed(&a_coeff.map(i64::from));
            let a2 = p2.pointwise(&a2_hat, &z2[s]);
            for j in 0..N {
                acc2[j] = p2.add_mod(acc2[j], a2[j]);
            }
        }
        let x1 = fips.intt(&az_hat[r]);
        let x2 = p2.intt(&acc2);
        for j in 0..N {
            out[r][j] = crt_centered(x1[j], x2[j]);
        }
    }
    out
}

/// [`az_crt`] with the residue modulo q already in the NTT domain: `az_hat = A_hat o
/// NTT(z)` in [0, q) (as [`crate::native::verify_internal_trace`] computes it), so only the
/// product modulo CRT_P31 is computed here. Requires |z|_inf <= 2^20.
pub fn az_crt_with(key: &ExpandedKey, z: &VecL, az_hat: &[[u64; N]; K]) -> VecK {
    az_crt_with_for::<Set44>(key, z, az_hat)
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

/// (A z) over the integers by schoolbook for parameter set P; i128 entries.
pub fn az_schoolbook_for<P: ParamSet>(a: &P::MatrixCoeff, z: &P::VecL) -> Vec<[i128; N]> {
    let mut out = vec![[0i128; N]; P::K];
    for r in 0..P::K {
        for s in 0..P::L {
            let a_coeff = P::get_coeff(a, r, s);
            let mut ar: Poly = [0; N];
            for j in 0..N {
                ar[j] = i64::from(a_coeff[j]);
            }
            let prod = negacyclic_schoolbook(&ar, &z[s]);
            for j in 0..N {
                out[r][j] += prod[j];
            }
        }
    }
    out
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

/// (A z) over the integers for parameter set P: CRT when |z| <= 2^20, schoolbook otherwise.
pub fn az_integer_for<P: ParamSet>(key: &ExpandedKey<P>, z: &P::VecL) -> Vec<[i128; N]> {
    az_integer_with_for::<P>(key, z, None)
}

/// [`az_integer_for`], reusing `A_hat o NTT(z)` modulo q when the caller has it.
pub fn az_integer_with_for<P: ParamSet>(
    key: &ExpandedKey<P>,
    z: &P::VecL,
    az_hat: Option<&P::VecKU64>,
) -> Vec<[i128; N]> {
    if (0..P::L).all(|s| z[s].iter().all(|x| x.abs() <= Z_CRT_MAX)) {
        let v = match az_hat {
            Some(az_hat) => az_crt_with_for::<P>(key, z, az_hat),
            None => az_crt_for::<P>(key, z),
        };
        let mut out = vec![[0i128; N]; P::K];
        for r in 0..P::K {
            for j in 0..N {
                out[r][j] = i128::from(v[r][j]);
            }
        }
        out
    } else {
        az_schoolbook_for::<P>(&key.a, z)
    }
}

/// (A z) over the integers: CRT when |z| <= 2^20, schoolbook otherwise.
pub fn az_integer(key: &ExpandedKey, z: &VecL) -> [[i128; N]; K] {
    az_integer_with(key, z, None)
}

/// [`az_integer`], reusing `A_hat o NTT(z)` modulo q when the caller has it (see
/// [`az_crt_with`]).
pub fn az_integer_with(key: &ExpandedKey, z: &VecL, az_hat: Option<&[[u64; N]; K]>) -> [[i128; N]; K] {
    if z.iter()
        .flat_map(|p| p.iter())
        .all(|x| x.abs() <= Z_CRT_MAX)
    {
        let v = match az_hat {
            Some(az_hat) => az_crt_with(key, z, az_hat),
            None => az_crt(key, z),
        };
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
    use crate::params::{Set65, Set87};

    #[test]
    fn prime2_tables() {
        let p = prime2();
        assert_eq!(CRT_P31, 15 * (1 << 27) + 1);
        assert_eq!(
            pow_mod(CRT_P31_GENERATOR, (CRT_P31 - 1) / 2, CRT_P31),
            CRT_P31 - 1,
            "31 is a non-residue"
        );
        assert_eq!(
            pow_mod(p.zetas[1], 2, CRT_P31),
            CRT_P31 - 1,
            "zetas[1] = zeta^128 is a 4th root of unity"
        );
        // Exactness margin: l 256 (q-1)/2 2^20 < M/2 for all parameter sets.
        for l in [Set44::L, Set65::L, Set87::L] {
            let bound = (l * N) as i128 * ((Q - 1) / 2) as i128 * Z_CRT_MAX as i128;
            assert!(bound < (Q as i128 * CRT_P31 as i128) / 2);
        }
    }

    #[test]
    fn crt_centered_recovers_signed_values() {
        let m = Q as i128 * CRT_P31 as i128;
        for x in [0i128, 1, -1, 123_456_789_012, -(m / 2) + 1, m / 2] {
            let x1 = x.rem_euclid(Q as i128) as u64;
            let x2 = x.rem_euclid(CRT_P31 as i128) as u64;
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

    #[test]
    fn az_crt_matches_schoolbook_65() {
        let (sk, pk) = gen::keypair_for::<Set65>(&gen::key_seed(65, 0));
        let _ = sk;
        let key = ExpandedKey::<Set65>::from_bytes_for(&pk).unwrap();
        let mut rng = SplitMix64::new(6517);
        for bound in [1i64 << 19, Z_CRT_MAX] {
            let mut z = Set65::zero_vec_l();
            for s in 0..Set65::L {
                for x in z[s].iter_mut() {
                    *x = rng.below(2 * bound as u64 + 1) as i64 - bound;
                }
            }
            z[0][0] = bound;
            z[Set65::L - 1][255] = -bound;
            let crt = az_crt_for::<Set65>(&key, &z);
            let sb = az_schoolbook_for::<Set65>(&key.a, &z);
            for row in 0..Set65::K {
                for j in 0..N {
                    assert_eq!(i128::from(crt[row][j]), sb[row][j]);
                }
            }
        }
    }

    #[test]
    fn az_crt_matches_schoolbook_87() {
        let (sk, pk) = gen::keypair_for::<Set87>(&gen::key_seed(87, 0));
        let _ = sk;
        let key = ExpandedKey::<Set87>::from_bytes_for(&pk).unwrap();
        let mut rng = SplitMix64::new(8717);
        for bound in [1i64 << 19, Z_CRT_MAX] {
            let mut z = Set87::zero_vec_l();
            for s in 0..Set87::L {
                for x in z[s].iter_mut() {
                    *x = rng.below(2 * bound as u64 + 1) as i64 - bound;
                }
            }
            z[0][0] = bound;
            z[Set87::L - 1][255] = -bound;
            let crt = az_crt_for::<Set87>(&key, &z);
            let sb = az_schoolbook_for::<Set87>(&key.a, &z);
            for row in 0..Set87::K {
                for j in 0..N {
                    assert_eq!(i128::from(crt[row][j]), sb[row][j]);
                }
            }
        }
    }

    #[test]
    fn az_integer_falls_back_to_schoolbook_for_large_z() {
        let r = gen::record(4, 0);
        let key = ExpandedKey::from_bytes(&r.pk).unwrap();
        let mut z = [[0i64; N]; L];
        // Value exceeding Z_CRT_MAX triggers fallback to 128-bit schoolbook
        z[0][0] = Z_CRT_MAX + 1000;
        let az = az_integer(&key, &z);
        let sb = az_schoolbook(&key.a, &z);
        assert_eq!(az, sb);
    }
}
