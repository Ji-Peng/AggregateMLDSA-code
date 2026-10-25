//! Public-key expansion: ExpandA matrix and t1 * 2^d in NTT-free coefficient form (spec Section 2).
//!
//! `ExpandedKey` holds what the verifier precomputes per key (spec Section 2 items 1 to 3):
//! A_hat = ExpandA(rho) (FIPS 204 Algorithms 14, 30, 32), the centered coefficient-form matrix
//! A = ctr_q(NTT^{-1}(A_hat)), tr = H(pk, 64), plus NTT(t1 2^d) for the native verifier and the
//! NTT of A modulo the second CRT prime for the prover's integer product (`crate::crt`).
//! `public_t` computes T = 2^d (c * t1) over the integers from the sparse c (item 5).

use crate::crt;
use crate::native::{ctr_q, pk_decode, tr, Challenge, VecK};
use crate::ntt::NttPrime;
use crate::params::*;
use sha3::digest::{ExtendableOutput, Update, XofReader};

/// A k x l matrix of NTT-domain polynomials.
pub type MatrixNtt = [[[u64; N]; L]; K];
/// A k x l matrix of centered coefficient-form polynomials.
pub type MatrixCoeff = [[[i32; N]; L]; K];

/// SHAKE128 rate in bytes (FIPS 202).
pub const SHAKE128_RATE: usize = 168;
/// SHAKE256 rate in bytes (FIPS 202).
pub const SHAKE256_RATE: usize = 136;

/// CoeffFromThreeBytes (FIPS 204 Algorithm 14); `None` is the symbol "bottom".
pub fn coeff_from_three_bytes(b0: u8, b1: u8, b2: u8) -> Option<u32> {
    let b2p = u32::from(b2 & 0x7F);
    let z = (b2p << 16) | (u32::from(b1) << 8) | u32::from(b0);
    if i64::from(z) < Q {
        Some(z)
    } else {
        None
    }
}

/// RejNTTPoly(rho') (FIPS 204 Algorithm 30) with G = SHAKE128. Returns the polynomial and the
/// number of SHAKE128 output blocks touched (5 except with probability about 2^-132; spec Section 2).
pub fn rej_ntt_poly(seed: &[u8; 34]) -> ([u64; N], usize) {
    let mut g = sha3::Shake128::default();
    g.update(seed);
    let mut reader = g.finalize_xof();
    let mut a = [0u64; N];
    let mut j = 0usize;
    let mut squeezed = 0usize;
    let mut s = [0u8; 3];
    while j < N {
        reader.read(&mut s);
        squeezed += 3;
        if let Some(c) = coeff_from_three_bytes(s[0], s[1], s[2]) {
            a[j] = u64::from(c);
            j += 1;
        }
    }
    (a, squeezed.div_ceil(SHAKE128_RATE))
}

/// ExpandA(rho) (FIPS 204 Algorithm 32): A_hat[r][s] = RejNTTPoly(rho || s || r), and the
/// SHAKE128 block count of each entry.
pub fn expand_a(rho: &[u8; 32]) -> (Box<MatrixNtt>, [[usize; L]; K]) {
    let mut a_hat = Box::new([[[0u64; N]; L]; K]);
    let mut blocks = [[0usize; L]; K];
    for r in 0..K {
        for s in 0..L {
            let mut seed = [0u8; 34];
            seed[..32].copy_from_slice(rho);
            seed[32] = s as u8;
            seed[33] = r as u8;
            let (poly, nb) = rej_ntt_poly(&seed);
            a_hat[r][s] = poly;
            blocks[r][s] = nb;
        }
    }
    (a_hat, blocks)
}

/// Per-key precomputation of the verifier and the prover.
#[derive(Clone, Debug)]
pub struct ExpandedKey {
    /// Encoded public key (1,312 bytes).
    pub pk: Vec<u8>,
    /// Seed rho.
    pub rho: [u8; 32],
    /// t1 in [0, 1023].
    pub t1: [[u16; N]; K],
    /// tr = H(pk, 64).
    pub tr: [u8; TR_BYTES],
    /// A_hat = ExpandA(rho), NTT domain mod q.
    pub a_hat: Box<MatrixNtt>,
    /// SHAKE128 blocks used by each RejNTTPoly call.
    pub expand_blocks: [[usize; L]; K],
    /// A[r][s] = ctr_q(NTT^{-1}(A_hat[r][s])) in [-(q-1)/2, (q-1)/2] (spec Appendix B.9).
    pub a: Box<MatrixCoeff>,
    /// NTT(t1 2^d) mod q (native verifier, Algorithm 8 line 9).
    pub t1_2d_hat: [[u64; N]; K],
    /// NTT of A modulo the second CRT prime (prover's integer product).
    pub a_hat_p2: Box<MatrixNtt>,
}

impl ExpandedKey {
    /// pkDecode plus the expansion; `None` on a wrong length.
    pub fn from_bytes(pk: &[u8]) -> Option<Self> {
        let decoded = pk_decode(pk)?;
        let (a_hat, expand_blocks) = expand_a(&decoded.rho);
        let fips = NttPrime::fips();
        let p2 = crt::prime2();
        let mut a = Box::new([[[0i32; N]; L]; K]);
        let mut a_hat_p2 = Box::new([[[0u64; N]; L]; K]);
        for r in 0..K {
            for s in 0..L {
                let coeff = fips.intt(&a_hat[r][s]);
                let mut centered = [0i64; N];
                for j in 0..N {
                    let c = ctr_q(coeff[j] as i64);
                    a[r][s][j] = c as i32;
                    centered[j] = c;
                }
                a_hat_p2[r][s] = p2.ntt_signed(&centered);
            }
        }
        let mut t1_2d_hat = [[0u64; N]; K];
        for r in 0..K {
            let mut t = [0i64; N];
            for j in 0..N {
                t[j] = i64::from(decoded.t1[r][j]) << D;
            }
            t1_2d_hat[r] = fips.ntt_signed(&t);
        }
        Some(Self {
            pk: pk.to_vec(),
            rho: decoded.rho,
            t1: decoded.t1,
            tr: tr(pk),
            a_hat,
            expand_blocks,
            a,
            t1_2d_hat,
            a_hat_p2,
        })
    }
}

/// T[rho] = 2^d (c * t1[rho]) over Z[X]/(X^256 + 1), from the tau nonzero entries of c
/// (k tau 256 signed additions, no reduction; spec Section 2 item 5). |T| <= B_T < 2^29.
pub fn public_t(c: &Challenge, t1: &[[u16; N]; K]) -> VecK {
    let mut out = [[0i64; N]; K];
    for r in 0..K {
        for &pos in &c.positions {
            let p = usize::from(pos);
            let sign = i64::from(c.coeffs[p]);
            // x^p t1: coefficients m < N - p land at p + m, the others wrap negated to
            // m + p - N (two straight loops, which vectorize, instead of a branch per m).
            let (low, high) = t1[r].split_at(N - p);
            for (slot, &x) in out[r][p..].iter_mut().zip(low) {
                *slot += sign * i64::from(x);
            }
            for (slot, &x) in out[r][..p].iter_mut().zip(high) {
                *slot -= sign * i64::from(x);
            }
        }
        for x in out[r].iter_mut() {
            *x <<= D;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gen;
    use crate::native::{mod_q, sample_in_ball};

    #[test]
    fn coeff_from_three_bytes_cases() {
        assert_eq!(coeff_from_three_bytes(0, 0, 0), Some(0));
        // 2^23 - 1 with the top bit of b2 cleared: z = 0x7FFFFF >= q -> bottom.
        assert_eq!(coeff_from_three_bytes(0xFF, 0xFF, 0xFF), None);
        // q - 1 = 0x7FE000.
        assert_eq!(
            coeff_from_three_bytes(0x00, 0xE0, 0x7F),
            Some((Q - 1) as u32)
        );
        assert_eq!(coeff_from_three_bytes(0x01, 0xE0, 0x7F), None);
        // The top bit of b2 is ignored.
        assert_eq!(
            coeff_from_three_bytes(0x00, 0xE0, 0xFF),
            Some((Q - 1) as u32)
        );
    }

    #[test]
    fn expanded_key_is_consistent() {
        let r = gen::record(3, 1);
        let key = ExpandedKey::from_bytes(&r.pk).unwrap();
        let fips = NttPrime::fips();
        for row in 0..K {
            for s in 0..L {
                // Five SHAKE128 blocks per entry except with probability ~2^-132.
                assert_eq!(key.expand_blocks[row][s], 5);
                let mut back = [0u64; N];
                for j in 0..N {
                    let c = i64::from(key.a[row][s][j]);
                    assert!(c.abs() <= (Q - 1) / 2);
                    back[j] = mod_q(c) as u64;
                }
                assert_eq!(fips.ntt(&back), key.a_hat[row][s]);
            }
        }
    }

    #[test]
    fn public_t_matches_ntt_product_mod_q_and_bound() {
        let r = gen::record(3, 2);
        let key = ExpandedKey::from_bytes(&r.pk).unwrap();
        let c = sample_in_ball(&r.sig[..32]);
        let t = public_t(&c, &key.t1);
        let fips = NttPrime::fips();
        let mut cp = [0i64; N];
        for j in 0..N {
            cp[j] = i64::from(c.coeffs[j]);
        }
        let c_hat = fips.ntt_signed(&cp);
        for row in 0..K {
            let prod = fips.intt(&fips.pointwise(&c_hat, &key.t1_2d_hat[row]));
            for j in 0..N {
                assert!(t[row][j].abs() <= B_T);
                assert_eq!(mod_q(t[row][j]) as u64, prod[j]);
            }
        }
    }
}
