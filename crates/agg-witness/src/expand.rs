//! Public-key expansion: ExpandA matrix and t1 * 2^d in NTT-free coefficient form.
//!
//! `ExpandedKey` holds what the verifier precomputes per key:
//! A_hat = ExpandA(rho) (FIPS 204 Algorithms 14, 30, 32), the centered coefficient-form matrix
//! A = ctr_q(NTT^{-1}(A_hat)), tr = H(pk, 64), and the prover's NTT-domain matrices
//! (`PreparedMatrices`: A_hat mod q and NTT(A) mod the 28-bit transform prime of
//! `agg_mldsa_simd`, in the coefficient order of the SIMD backend). The FIPS-order u64 forms
//! of A_hat, NTT(t1 2^d) and NTT(A) modulo the reference CRT prime (`crate::crt`) are reference
//! methods, recomputed on demand for the native verifier and the reference derivation.
//! `ExpandedKey::many` expands keys in tiles of 8 (batched ExpandA and tr hashing).
//! `public_t` computes T = 2^d (c * t1) over the integers from the sparse c.

use crate::crt;
use crate::native::{pk_decode, Challenge, VecK};
use crate::ntt::NttPrime;
use crate::params::*;
use agg_mldsa_simd::{Backend, Engine, ExpandedA, Poly32};
use core::marker::PhantomData;
use rayon::prelude::*;
use std::sync::Arc;
use sha3::digest::{ExtendableOutput, Update, XofReader};

/// A k x l matrix of NTT-domain polynomials.
pub type MatrixNtt = [[[u64; N]; L]; K];
/// A k x l matrix of centered coefficient-form polynomials.
pub type MatrixCoeff = [[[i32; N]; L]; K];

/// SHAKE128 rate in bytes (FIPS 202).
pub const SHAKE128_RATE: usize = 168;

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
/// number of SHAKE128 output blocks touched (5 except with probability about 2^-132).
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

/// ExpandA for parameter set P: A_hat[r][s] = RejNTTPoly(rho || s || r), and the
/// SHAKE128 block count of each entry.
pub fn expand_a_for<P: ParamSet>(rho: &[u8; 32]) -> (Box<P::MatrixNtt>, P::ExpandBlocks) {
    let mut a_hat = P::zero_matrix_ntt();
    let mut blocks = P::zero_expand_blocks();
    for r in 0..P::K {
        for s in 0..P::L {
            let mut seed = [0u8; 34];
            seed[..32].copy_from_slice(rho);
            seed[32] = s as u8;
            seed[33] = r as u8;
            let (poly, nb) = rej_ntt_poly(&seed);
            *P::get_ntt_mut(&mut a_hat, r, s) = poly;
            P::set_block(&mut blocks, r, s, nb);
        }
    }
    (a_hat, blocks)
}

/// ExpandA(rho) (FIPS 204 Algorithm 32): A_hat[r][s] = RejNTTPoly(rho || s || r), and the
/// SHAKE128 block count of each entry.
pub fn expand_a(rho: &[u8; 32]) -> (Box<MatrixNtt>, [[usize; L]; K]) {
    expand_a_for::<Set44>(rho)
}

/// Keys per expansion tile: one batch of ExpandA streams and tr digests at a time.
pub const KEY_TILE: usize = 8;

/// A k x l matrix of NTT-domain polynomials in the coefficient order of one SIMD backend.
pub type MatrixPoly32 = [[Poly32; L]; K];

/// The prover's NTT-domain copies of A, in the coefficient order of the backend that built
/// them (never serialized; only an engine of the same backend may read them).
#[derive(Clone, Debug)]
pub struct PreparedMatrices {
    /// A_hat = ExpandA(rho) modulo q, canonical in [0, q).
    pub a_hat_q: Box<MatrixPoly32>,
    /// NTT(A) modulo the 28-bit transform prime `agg_mldsa_simd::params::P2`, centered.
    pub a_hat_p2: Box<MatrixPoly32>,
    /// The backend whose coefficient order the two matrices use.
    pub backend: Backend,
}

/// Per-key precomputation of the verifier and the prover.
#[derive(Clone, Debug)]
pub struct ExpandedKey<P: ParamSet = Set44> {
    /// Encoded public key.
    pub pk: Vec<u8>,
    /// Seed rho.
    pub rho: [u8; 32],
    /// t1 in [0, 1023].
    pub t1: P::T1,
    /// tr = H(pk, 64).
    pub tr: [u8; TR_BYTES],
    /// SHAKE128 blocks used by each RejNTTPoly call.
    pub expand_blocks: P::ExpandBlocks,
    /// A[r][s] = ctr_q(NTT^{-1}(A_hat[r][s])) in [-(q-1)/2, (q-1)/2], shared with the
    /// prover's public data instead of copied.
    pub a: Arc<P::MatrixCoeff>,
    /// A_hat modulo q and NTT(A) modulo the transform prime, in backend order.
    pub prepared: P::Prepared,
    _marker: PhantomData<P>,
}

/// A zeroed boxed matrix built on the heap (no matrix-sized stack temporary).
fn boxed_matrix<T: Clone>(zero: T) -> Box<[[T; L]; K]> {
    let rows: Vec<[T; L]> = vec![std::array::from_fn(|_| zero.clone()); K];
    match rows.into_boxed_slice().try_into() {
        Ok(boxed) => boxed,
        Err(_) => unreachable!("k rows"),
    }
}

/// Expands one tile of at most [`KEY_TILE`] encoded keys; `None` for a key of the wrong
/// length.
fn expand_tile(engine: &Engine, pks: &[&[u8]]) -> Vec<Option<ExpandedKey<Set44>>> {
    let decoded: Vec<Option<_>> = pks.iter().map(|pk| pk_decode(pk)).collect();
    let live: Vec<usize> = (0..pks.len()).filter(|&i| decoded[i].is_some()).collect();
    let rhos: Vec<[u8; 32]> = live.iter().map(|&i| decoded[i].as_ref().expect("live").rho).collect();
    let mut expanded = vec![ExpandedA::zero(); live.len()];
    engine.expand_a_many(&rhos, &mut expanded);
    let tr_inputs: Vec<[&[u8]; 4]> = live.iter().map(|&i| [pks[i], &[][..], &[], &[]]).collect();
    let mut trs = vec![0u8; live.len() * TR_BYTES];
    engine.shake256_many(&tr_inputs, TR_BYTES, &mut trs);
    let mut out: Vec<Option<ExpandedKey<Set44>>> = (0..pks.len()).map(|_| None).collect();
    for ((&i, ex), tr) in live.iter().zip(&expanded).zip(trs.as_chunks::<TR_BYTES>().0) {
        let key = decoded[i].as_ref().expect("live");
        let mut a = Set44::zero_matrix_coeff();
        let mut a_hat_q = boxed_matrix(Poly32::zero());
        let mut a_hat_p2 = boxed_matrix(Poly32::zero());
        *a_hat_q = ex.a_hat;
        engine.prepare_matrices(&ex.a_hat, Arc::get_mut(&mut a).expect("a fresh matrix"), &mut a_hat_p2);
        out[i] = Some(ExpandedKey {
            pk: pks[i].to_vec(),
            rho: key.rho,
            t1: key.t1,
            tr: *tr,
            expand_blocks: ex.blocks.map(|row| row.map(usize::from)),
            a,
            prepared: PreparedMatrices { a_hat_q, a_hat_p2, backend: engine.backend() },
            _marker: PhantomData,
        });
    }
    out
}

/// Prepares matrices for Set44 using the process-wide SIMD engine.
pub fn make_prepared_44(rho: &[u8; 32]) -> PreparedMatrices {
    let engine = agg_mldsa_simd::engine();
    let mut expanded = vec![ExpandedA::zero(); 1];
    engine.expand_a_many(&[*rho], &mut expanded);
    let mut a = boxed_matrix([0i32; N]);
    let mut a_hat_q = boxed_matrix(Poly32::zero());
    let mut a_hat_p2 = boxed_matrix(Poly32::zero());
    *a_hat_q = expanded[0].a_hat;
    engine.prepare_matrices(&expanded[0].a_hat, &mut a, &mut a_hat_p2);
    PreparedMatrices {
        a_hat_q,
        a_hat_p2,
        backend: engine.backend(),
    }
}

/// [`ExpandedKey::from_bytes_reference`] on the process-wide SIMD engine, for the sets with
/// polynomial lanes (`K x L` their matrix shape): `ExpandA` runs as one batch of SHAKE128
/// streams, and each entry's centered coefficients come from the engine's inverse transform
/// (the values of the reference path: canonical `A_hat` in, `ctr_q` coefficients out).
pub fn expand_key_engine<P, const K2: usize, const L2: usize>(pk: &[u8]) -> Option<ExpandedKey<P>>
where
    P: ParamSet + agg_mldsa_simd::ParamSet,
{
    let decoded = crate::native::pk_decode_for::<P>(pk)?;
    let engine = agg_mldsa_simd::engine_for::<P>();
    let mut expanded = vec![ExpandedA::<K2, L2>::zero()];
    engine.expand_a_many_kl::<K2, L2>(&[decoded.rho], &mut expanded);
    let ex = &expanded[0];
    let mut a = P::zero_matrix_coeff();
    let fill = Arc::get_mut(&mut a).expect("a fresh matrix");
    let mut expand_blocks = P::zero_expand_blocks();
    for r in 0..K2 {
        for s in 0..L2 {
            let mut poly = ex.a_hat[r][s];
            engine.intt(agg_mldsa_simd::Prime::Q, &mut poly, agg_mldsa_simd::Scale::Plain, agg_mldsa_simd::Form::Centered);
            *P::get_coeff_mut(fill, r, s) = poly.0;
            P::set_block(&mut expand_blocks, r, s, usize::from(ex.blocks[r][s]));
        }
    }
    let mut tr = [0u8; TR_BYTES];
    crate::native::shake256(&[pk], &mut tr);
    let prepared = <P as ParamSet>::make_prepared(&decoded.rho);
    Some(ExpandedKey {
        pk: pk.to_vec(),
        rho: decoded.rho,
        t1: decoded.t1,
        tr,
        expand_blocks,
        a,
        prepared,
        _marker: PhantomData,
    })
}

/// [`expand_key_engine`] of a batch of keys: `tr = H(pk)` of every key as one batch of
/// SHAKE256 streams and `ExpandA` of every decodable key as one batch of SHAKE128 streams;
/// each result equals [`expand_key_engine`]'s.
pub fn expand_keys_engine<P, const K2: usize, const L2: usize>(pks: &[&[u8]]) -> Vec<Option<ExpandedKey<P>>>
where
    P: ParamSet + agg_mldsa_simd::ParamSet,
{
    let engine = agg_mldsa_simd::engine_for::<P>();
    let mut decoded: Vec<_> = pks.iter().map(|pk| crate::native::pk_decode_for::<P>(pk)).collect();
    let live: Vec<usize> = (0..pks.len()).filter(|&i| decoded[i].is_some()).collect();
    let mut trs = vec![[0u8; TR_BYTES]; live.len()];
    let inputs: Vec<[&[u8]; 4]> = live.iter().map(|&i| [pks[i], &[][..], &[][..], &[][..]]).collect();
    engine.shake256_many(&inputs, TR_BYTES, trs.as_flattened_mut());
    let rhos: Vec<[u8; 32]> = live.iter().map(|&i| decoded[i].as_ref().expect("live").rho).collect();
    let mut expanded = vec![ExpandedA::<K2, L2>::zero(); live.len()];
    engine.expand_a_many_kl::<K2, L2>(&rhos, &mut expanded);
    let mut out: Vec<Option<ExpandedKey<P>>> = (0..pks.len()).map(|_| None).collect();
    for ((&i, ex), tr) in live.iter().zip(&expanded).zip(trs) {
        let decoded = decoded[i].take().expect("live");
        let mut a = P::zero_matrix_coeff();
        let fill = Arc::get_mut(&mut a).expect("a fresh matrix");
        let mut expand_blocks = P::zero_expand_blocks();
        for r in 0..K2 {
            for s in 0..L2 {
                let mut poly = ex.a_hat[r][s];
                engine.intt(agg_mldsa_simd::Prime::Q, &mut poly, agg_mldsa_simd::Scale::Plain, agg_mldsa_simd::Form::Centered);
                *P::get_coeff_mut(fill, r, s) = poly.0;
                P::set_block(&mut expand_blocks, r, s, usize::from(ex.blocks[r][s]));
            }
        }
        out[i] = Some(ExpandedKey {
            pk: pks[i].to_vec(),
            rho: decoded.rho,
            t1: decoded.t1,
            tr,
            expand_blocks,
            a,
            prepared: <P as ParamSet>::make_prepared(&decoded.rho),
            _marker: PhantomData,
        });
    }
    out
}

impl<P: ParamSet> ExpandedKey<P> {
    /// Decodes and expands a public key for parameter set P.
    pub fn from_bytes_for(pk: &[u8]) -> Option<Self> {
        P::expand_key(pk)
    }

    /// Reference expansion from public key bytes.
    pub fn from_bytes_reference(pk: &[u8]) -> Option<Self> {
        let decoded = crate::native::pk_decode_for::<P>(pk)?;
        let (a_hat, expand_blocks) = expand_a_for::<P>(&decoded.rho);
        let fips = NttPrime::fips();
        let mut a = P::zero_matrix_coeff();
        let fill = Arc::get_mut(&mut a).expect("a fresh matrix");
        for r in 0..P::K {
            for s in 0..P::L {
                let ntt_poly = P::get_ntt(&a_hat, r, s);
                let coeffs = fips.intt(ntt_poly);
                let row = P::get_coeff_mut(fill, r, s);
                for j in 0..N {
                    row[j] = crate::native::ctr_q(coeffs[j] as i64) as i32;
                }
            }
        }
        let mut tr = [0u8; TR_BYTES];
        crate::native::shake256(&[pk], &mut tr);
        let prepared = P::make_prepared(&decoded.rho);
        Some(ExpandedKey {
            pk: pk.to_vec(),
            rho: decoded.rho,
            t1: decoded.t1,
            tr,
            expand_blocks,
            a,
            prepared,
            _marker: PhantomData,
        })
    }

    /// Recomputes A_hat in FIPS NTT order.
    pub fn a_hat(&self) -> Box<P::MatrixNtt> {
        let (a_hat, _) = expand_a_for::<P>(&self.rho);
        a_hat
    }

    /// NTT(t1 2^d) mod q.
    pub fn t1_2d_hat(&self) -> P::VecKU64 {
        let fips = NttPrime::fips();
        let mut out = P::zero_vec_k_u64();
        for r in 0..P::K {
            out[r] = fips.ntt_signed(&self.t1[r].map(|x| (i64::from(x)) << D));
        }
        out
    }
}

impl ExpandedKey<Set44> {
    /// pkDecode plus the expansion; `None` on a wrong length.
    pub fn from_bytes(pk: &[u8]) -> Option<Self> {
        Self::many(&[pk]).pop().flatten()
    }

    /// [`ExpandedKey::from_bytes`] for many keys, in tiles of [`KEY_TILE`] keys on the Rayon
    /// pool, with the process-wide SIMD engine.
    pub fn many(pks: &[&[u8]]) -> Vec<Option<Self>> {
        Self::many_with(agg_mldsa_simd::engine(), pks)
    }

    /// [`ExpandedKey::many`] with an explicit engine (tests compare every backend).
    #[doc(hidden)]
    pub fn many_with(engine: &Engine, pks: &[&[u8]]) -> Vec<Option<Self>> {
        let tiles: Vec<Vec<Option<Self>>> = pks.par_chunks(KEY_TILE).map(|tile| expand_tile(engine, tile)).collect();
        tiles.into_iter().flatten().collect()
    }

    /// NTT(A) modulo the reference CRT prime [`crt::CRT_P31`] (reference form for [`crate::crt`],
    /// recomputed from A).
    #[must_use]
    pub fn a_hat_p2_reference(&self) -> Box<MatrixNtt> {
        let p2 = crt::prime2();
        let mut out = boxed_matrix([0u64; N]);
        for (out_row, row) in out.iter_mut().zip(self.a.iter()) {
            for (dest, poly) in out_row.iter_mut().zip(row) {
                *dest = p2.ntt_signed(&poly.map(i64::from));
            }
        }
        out
    }
}

/// T[rho] = 2^d (c * t1[rho]) over Z[X]/(X^256 + 1) for parameter set P.
pub fn public_t_for<P: ParamSet>(c: &Challenge, t1: &P::T1) -> P::VecK {
    let mut out = P::zero_vec_k();
    let mut out_i32 = vec![[0i32; N]; P::K];
    public_t_into_for::<P>(c, t1, &mut out_i32);
    for r in 0..P::K {
        for j in 0..N {
            out[r][j] = i64::from(out_i32[r][j]);
        }
    }
    out
}

/// [`public_t_for`] in its stored width, written into `out` (overwritten).
pub fn public_t_into_for<P: ParamSet>(c: &Challenge, t1: &P::T1, out: &mut [[i32; N]]) {
    for r in 0..P::K {
        let row = &mut out[r];
        row.fill(0);
        for &pos in &c.positions {
            let p = usize::from(pos);
            let sign = i32::from(c.coeffs[p]);
            let (low, high) = t1[r].split_at(N - p);
            for (slot, &x) in row[p..].iter_mut().zip(low) {
                *slot += sign * i32::from(x);
            }
            for (slot, &x) in row[..p].iter_mut().zip(high) {
                *slot -= sign * i32::from(x);
            }
        }
        for x in row.iter_mut() {
            *x <<= D;
        }
    }
}

/// T[rho] = 2^d (c * t1[rho]) over Z[X]/(X^256 + 1), from the tau nonzero entries of c
/// (k tau 256 signed additions, no reduction). |T| <= B_T < 2^29.
pub fn public_t(c: &Challenge, t1: &[[u16; N]; K]) -> VecK {
    public_t_for::<Set44>(c, t1)
}

/// [`public_t`] in its stored width, written into `out` (overwritten). Every partial sum is
/// a signed sum of at most tau values below 2^10, so |c * t1| <= tau (2^10 - 1) < 2^16
/// before the shift and |T| < 2^29 after it: all of it fits `i32`.
pub fn public_t_into(c: &Challenge, t1: &[[u16; N]; K], out: &mut [[i32; N]; K]) {
    public_t_into_for::<Set44>(c, t1, out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gen;
    use crate::native::{mod_q, sample_in_ball};

    /// The engine expansion of the sets with lanes equals the reference one.
    #[test]
    fn engine_key_expansion_matches_the_reference() {
        fn check<P: ParamSet>(seed: u64)
        where
            P::T1: PartialEq,
            P::MatrixCoeff: PartialEq,
            P::ExpandBlocks: PartialEq,
        {
            for record in gen::records_for::<P>(seed, 3) {
                let fast = ExpandedKey::<P>::from_bytes_for(&record.pk).unwrap();
                let reference = ExpandedKey::<P>::from_bytes_reference(&record.pk).unwrap();
                assert_eq!(fast.pk, reference.pk);
                assert_eq!(fast.rho, reference.rho);
                assert!(fast.t1 == reference.t1);
                assert_eq!(fast.tr, reference.tr);
                assert!(fast.expand_blocks == reference.expand_blocks, "ExpandA blocks");
                assert!(*fast.a == *reference.a, "centered A");
            }
        }
        check::<Set65>(651);
        check::<Set87>(871);
    }

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
                assert_eq!(fips.ntt(&back), key.a_hat()[row][s]);
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
            let prod = fips.intt(&fips.pointwise(&c_hat, &key.t1_2d_hat()[row]));
            for j in 0..N {
                assert!(t[row][j].abs() <= B_T);
                assert_eq!(mod_q(t[row][j]) as u64, prod[j]);
            }
        }
    }

    #[test]
    fn expanded_key_is_consistent_65() {
        let r = gen::record_for::<Set65>(653, 1);
        let key = ExpandedKey::<Set65>::from_bytes_for(&r.pk).unwrap();
        let fips = NttPrime::fips();
        for row in 0..Set65::K {
            for s in 0..Set65::L {
                assert_eq!(key.expand_blocks[row][s], 5);
                let mut back = [0u64; N];
                for j in 0..N {
                    let c = i64::from(key.a[row][s][j]);
                    assert!(c.abs() <= (Q - 1) / 2);
                    back[j] = mod_q(c) as u64;
                }
                assert_eq!(fips.ntt(&back), key.a_hat()[row][s]);
            }
        }
    }

    /// The batched expansion gives each key's own expansion, and `None` for a key that
    /// does not decode.
    fn batched_keys_match<P: ParamSet>(seed: u64) {
        let records = crate::gen::records_for::<P>(seed, 11);
        let mut pks: Vec<&[u8]> = records.iter().map(|r| r.pk.as_slice()).collect();
        let short = &records[0].pk[..P::PK_BYTES - 1];
        pks.insert(4, short);
        let batch = P::expand_keys(&pks);
        assert_eq!(batch.len(), pks.len());
        for (i, (pk, key)) in pks.iter().zip(&batch).enumerate() {
            match (key, P::expand_key(pk)) {
                (None, None) => {}
                (Some(got), Some(want)) => {
                    assert!(got.pk == want.pk && got.rho == want.rho && got.t1 == want.t1 && got.tr == want.tr, "key {i}");
                    for r in 0..P::K {
                        for s in 0..P::L {
                            assert_eq!(P::get_coeff(&got.a, r, s), P::get_coeff(&want.a, r, s), "key {i}");
                            assert_eq!(P::get_block(&got.expand_blocks, r, s), P::get_block(&want.expand_blocks, r, s));
                        }
                    }
                }
                _ => panic!("key {i}: batched and single expansions disagree on decoding"),
            }
        }
        assert!(batch[4].is_none());
    }

    #[test]
    fn batched_keys_match_single_keys() {
        batched_keys_match::<Set44>(19);
        batched_keys_match::<Set65>(21);
        batched_keys_match::<Set87>(23);
    }

    #[test]
    fn expanded_key_is_consistent_87() {
        let r = gen::record_for::<Set87>(873, 1);
        let key = ExpandedKey::<Set87>::from_bytes_for(&r.pk).unwrap();
        let fips = NttPrime::fips();
        for row in 0..Set87::K {
            for s in 0..Set87::L {
                assert_eq!(key.expand_blocks[row][s], 5);
                let mut back = [0u64; N];
                for j in 0..N {
                    let c = i64::from(key.a[row][s][j]);
                    assert!(c.abs() <= (Q - 1) / 2);
                    back[j] = mod_q(c) as u64;
                }
                assert_eq!(fips.ntt(&back), key.a_hat()[row][s]);
            }
        }
    }

    #[test]
    fn public_t_matches_ntt_product_65_and_87() {
        let r65 = gen::record_for::<Set65>(653, 2);
        let key65 = ExpandedKey::<Set65>::from_bytes_for(&r65.pk).unwrap();
        let c65 = crate::native::sample_in_ball_for::<Set65>(&r65.sig[..Set65::C_TILDE_BYTES]);
        let t65 = public_t_for::<Set65>(&c65, &key65.t1);
        let fips = NttPrime::fips();
        let mut cp65 = [0i64; N];
        for j in 0..N {
            cp65[j] = i64::from(c65.coeffs[j]);
        }
        let c_hat65 = fips.ntt_signed(&cp65);
        for row in 0..Set65::K {
            let prod = fips.intt(&fips.pointwise(&c_hat65, &key65.t1_2d_hat()[row]));
            for j in 0..N {
                assert!(t65[row][j].abs() <= Set65::B_T);
                assert_eq!(mod_q(t65[row][j]) as u64, prod[j]);
            }
        }

        let r87 = gen::record_for::<Set87>(873, 2);
        let key87 = ExpandedKey::<Set87>::from_bytes_for(&r87.pk).unwrap();
        let c87 = crate::native::sample_in_ball_for::<Set87>(&r87.sig[..Set87::C_TILDE_BYTES]);
        let t87 = public_t_for::<Set87>(&c87, &key87.t1);
        let mut cp87 = [0i64; N];
        for j in 0..N {
            cp87[j] = i64::from(c87.coeffs[j]);
        }
        let c_hat87 = fips.ntt_signed(&cp87);
        for row in 0..Set87::K {
            let prod = fips.intt(&fips.pointwise(&c_hat87, &key87.t1_2d_hat()[row]));
            for j in 0..N {
                assert!(t87[row][j].abs() <= Set87::B_T);
                assert_eq!(mod_q(t87[row][j]) as u64, prod[j]);
            }
        }
    }
}
