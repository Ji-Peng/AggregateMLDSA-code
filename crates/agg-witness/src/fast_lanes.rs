//! Batched strict witness derivation on the SIMD engine for the parameter sets whose vector
//! index is a polynomial lane (ML-DSA-65, ML-DSA-87): the counterpart of [`crate::fast`].
//!
//! For a record and its signature, in tiles of [`TILE`] records: decode z (with its norm)
//! and the hint, reuse the instance's `T = 2^d (c t1)` when the signature's c~ is the
//! record's, and compute A z over Z from its residues modulo the two transform primes `P2`,
//! `P3` of `agg_mldsa_simd`: `|A z| <= l 256 (q - 1)/2 gamma1 < 2^52` is below
//! `P2 P3 / 2 ~ 2^54.8`, so the CRT value is A z itself and its residue modulo q needs no
//! transform of its own. The matrix is transformed per record from its centered
//! coefficients (`ExpandedKey::a`), two transforms per entry. Each row finishes without
//! division (`finish_row_2p`: centered CRT, UseHint on `A z - T` modulo q, the residual u
//! and the exact quotient v, with their bounds), w1' is packed, and one batch of SHAKE256
//! digests per tile recomputes c~' = H(mu || w1Encode(w1')).
//!
//! A record that fails any step is derived again by [`crate::derive::derive_with_t_for`], so
//! every error is exactly the reference one, and so is the witness of a record the reference path
//! accepts.

use crate::derive::{derive_with_t_for, DeriveError};
use crate::instance::PublicRecord;
use crate::native::hint_bit_unpack_for;
use crate::params::{ParamSet, N};
use agg_mldsa_simd::{Engine, FinishOut, FinishRow2P, Form, Poly32, Prime, Scale};

/// Records per derivation tile (one batch of commitment digests).
pub const TILE: usize = 8;

/// The derived values of one record: z and per row of `A z - T` the finishing output
/// `(w1', u, v)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaneWitness {
    /// Response z, one polynomial per column, `|z| < gamma1 - beta`.
    pub z: Vec<Poly32>,
    /// Row `rho`: w1' in `[0, m_w)`, u with `|u| <= B_u`, v with `|v| <= B_v`.
    pub rows: Vec<FinishOut>,
}

impl LaneWitness {
    /// An all-zero value of parameter set `P` (scratch before a derivation).
    #[must_use]
    pub fn zero<P: ParamSet>() -> Self {
        Self {
            z: vec![Poly32::zero(); P::L],
            rows: vec![FinishOut::zero(); P::K],
        }
    }

    /// The values of a reference witness (every one fits `i32` after the strict checks).
    fn fill_from<P: ParamSet>(&mut self, w: &crate::derive::SigWitness<P>) {
        let narrow = |x: i64| i32::try_from(x).expect("strict witness values fit i32");
        for (dest, src) in self.z.iter_mut().zip(w.z.as_ref()) {
            *dest = Poly32(src.map(narrow));
        }
        for (r, row) in self.rows.iter_mut().enumerate() {
            row.w1 = w.w1[r];
            row.u = Poly32(w.u[r].map(narrow));
            row.v = Poly32(w.v[r].map(narrow));
        }
    }
}

/// `p` transformed modulo `prime` and reduced to `form`.
fn transformed<P: agg_mldsa_simd::ParamSet>(engine: &Engine<P>, prime: Prime, p: &Poly32, form: Form) -> Poly32 {
    let mut x = *p;
    engine.ntt(prime, &mut x);
    engine.reduce(prime, &mut x, form);
    x
}

/// Everything up to the commitment hash for one record, with `L` the columns of `P`;
/// `false` sends the record to the reference derivation.
fn derive_one<P, const L: usize>(
    engine: &Engine<P>,
    record: &PublicRecord<P>,
    sig: &[u8],
    out: &mut LaneWitness,
    w1_bytes: &mut [u8],
) -> bool
where
    P: ParamSet + agg_mldsa_simd::ParamSet,
{
    debug_assert_eq!(<P as ParamSet>::L, L);
    let k = <P as ParamSet>::K;
    if sig.len() != <P as ParamSet>::SIG_BYTES {
        return false;
    }
    let (c_tilde, rest) = sig.split_at(<P as ParamSet>::C_TILDE_BYTES);
    // The instance's T is the product for the record's c~ only.
    if c_tilde != record.c_tilde.as_ref() {
        return false;
    }
    let (z_bytes, hint) = rest.split_at(L * 32 * <P as ParamSet>::Z_BITS);
    let z_max = engine.unpack_z(z_bytes, &mut out.z);
    if i64::from(z_max) >= <P as ParamSet>::GAMMA1 - <P as ParamSet>::BETA {
        return false;
    }
    let Some(h) = hint_bit_unpack_for::<P>(hint) else {
        return false;
    };
    // |z| <= gamma1 = 2^19 and |A| <= (q - 1)/2 lie inside the forward transforms' input
    // range |a| <= 2^22; z stays unreduced and the matrix is centered modulo each transform
    // prime, as the four-column path reads them.
    let z: [Poly32; L] = std::array::from_fn(|s| out.z[s]);
    let z_hat = |prime: Prime| -> [Poly32; L] {
        std::array::from_fn(|s| {
            let mut x = z[s];
            engine.ntt(prime, &mut x);
            x
        })
    };
    let (z_p2, z_p3) = (z_hat(Prime::P2), z_hat(Prime::P3));
    let key = &record.key;
    let t = &*record.t;
    let mut w1 = [[0u8; N]; 8];
    for r in 0..k {
        let a: [Poly32; L] = std::array::from_fn(|s| Poly32(*<P as ParamSet>::get_coeff(&key.a, r, s)));
        // The Mont inverse scale cancels the dot's 1/R, so the canonical outputs are
        // (A z)_r modulo each prime.
        let row = |prime: Prime, z_hat: &[Poly32; L]| {
            let a_hat: [Poly32; L] = std::array::from_fn(|s| transformed(engine, prime, &a[s], Form::Centered));
            let mut x = Poly32::zero();
            engine.dot_l::<L>(prime, &a_hat, z_hat, &mut x);
            engine.intt(prime, &mut x, Scale::Mont, Form::Canonical);
            x
        };
        let (x_p2, x_p3) = (row(Prime::P2, &z_p2), row(Prime::P3, &z_p3));
        let finish = FinishRow2P {
            xp2: &x_p2,
            xp3: &x_p3,
            t: &t[r],
            h: &h[r],
        };
        if engine.finish_row_2p(finish, &mut out.rows[r]).is_err() {
            return false;
        }
        w1[r] = out.rows[r].w1;
    }
    engine.pack_w1(&w1[..k], w1_bytes);
    true
}

/// Strict derivation of `records[i]` with signature `sigs[i]` into `out[i]` (each
/// [`LaneWitness::zero`] of `P`), in tiles of [`TILE`]. On failure: the lowest index whose
/// signature the reference derivation rejects, with that derivation's error.
///
/// # Panics
/// If the three slices differ in length, or `P` has another column count than 5 or 7.
pub fn derive_tile<P>(
    engine: &Engine<P>,
    records: &[&PublicRecord<P>],
    sigs: &[&[u8]],
    out: &mut [LaneWitness],
) -> Result<(), (usize, DeriveError)>
where
    P: ParamSet + agg_mldsa_simd::ParamSet,
{
    assert_eq!(records.len(), sigs.len(), "one signature per record");
    assert_eq!(records.len(), out.len(), "one output per record");
    let one = match <P as ParamSet>::L {
        5 => derive_one::<P, 5>,
        7 => derive_one::<P, 7>,
        l => panic!("no lane derivation for {l} columns"),
    };
    let digest_bytes = <P as ParamSet>::C_TILDE_BYTES;
    let w1_encode_bytes = <P as ParamSet>::W1_ENCODE_BYTES;
    assert!(w1_encode_bytes <= 1024 && digest_bytes <= 64);
    for (tile, ((records, sigs), out)) in records
        .chunks(TILE)
        .zip(sigs.chunks(TILE))
        .zip(out.chunks_mut(TILE))
        .enumerate()
    {
        let mut ok = [false; TILE];
        let mut w1_bytes = [[0u8; 1024]; TILE];
        for i in 0..records.len() {
            ok[i] = one(engine, records[i], sigs[i], &mut out[i], &mut w1_bytes[i][..w1_encode_bytes]);
        }
        let live: Vec<usize> = (0..records.len()).filter(|&i| ok[i]).collect();
        let inputs: Vec<[&[u8]; 4]> = live
            .iter()
            .map(|&i| [&records[i].mu[..], &w1_bytes[i][..w1_encode_bytes], &[][..], &[][..]])
            .collect();
        let mut digests = [0u8; TILE * 64];
        let digests = &mut digests[..live.len() * digest_bytes];
        engine.shake256_many(&inputs, digest_bytes, digests);
        for (&i, digest) in live.iter().zip(digests.chunks_exact(digest_bytes)) {
            ok[i] = digest == records[i].c_tilde.as_ref();
        }
        for i in (0..records.len()).filter(|&i| !ok[i]) {
            let record = records[i];
            let t: Vec<[i32; N]> = (0..<P as ParamSet>::K).map(|row| record.t[row]).collect();
            match derive_with_t_for::<P>(&record.key, &record.mu, sigs[i], &record.c_tilde, &t) {
                Ok(w) => out[i].fill_from(&w),
                Err(e) => return Err((tile * TILE + i, e)),
            }
        }
    }
    Ok(())
}

/// [`derive_tile`] on the reference derivation alone (any parameter set).
///
/// # Errors
/// As [`derive_tile`].
pub fn derive_tile_reference<P: ParamSet>(
    records: &[&PublicRecord<P>],
    sigs: &[&[u8]],
    out: &mut [LaneWitness],
) -> Result<(), (usize, DeriveError)> {
    for (i, (record, w)) in records.iter().zip(out.iter_mut()).enumerate() {
        let t: Vec<[i32; N]> = (0..P::K).map(|row| record.t[row]).collect();
        let reference = derive_with_t_for::<P>(&record.key, &record.mu, sigs[i], &record.c_tilde, &t).map_err(|e| (i, e))?;
        w.fill_from(&reference);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gen;
    use crate::instance::PublicInstance;
    use crate::params::{Set65, Set87};

    /// The engine path gives the reference witness for honest records, and the reference error for
    /// a corrupted signature (every backend the host supports), on the records of `seed`.
    fn matches_the_reference_derivation<P>(seed: u64)
    where
        P: ParamSet + agg_mldsa_simd::ParamSet,
    {
        let mut records = gen::records_for::<P>(seed, 12);
        records.extend(gen::records_shared_keys_for::<P>(seed.wrapping_add(1), 4, 2));
        let instance = PublicInstance::<P>::from_records_for(&records).unwrap();
        let refs: Vec<&PublicRecord<P>> = instance.records.iter().collect();
        let sigs: Vec<&[u8]> = records.iter().map(|r| r.sig.as_slice()).collect();
        for backend in [agg_mldsa_simd::Backend::Scalar, agg_mldsa_simd::Backend::Avx2, agg_mldsa_simd::Backend::Avx512] {
            let Some(engine) = Engine::<P>::new_for(backend) else { continue };
            let mut out = vec![LaneWitness::zero::<P>(); refs.len()];
            // Honest records never take the fallback.
            for (i, record) in refs.iter().enumerate() {
                let mut w1_bytes = vec![0u8; <P as ParamSet>::W1_ENCODE_BYTES];
                let accepted = match <P as ParamSet>::L {
                    5 => derive_one::<P, 5>(&engine, record, sigs[i], &mut out[i], &mut w1_bytes),
                    _ => derive_one::<P, 7>(&engine, record, sigs[i], &mut out[i], &mut w1_bytes),
                };
                assert!(accepted, "{backend:?} record {i} left the engine path");
                let digest = crate::native::commitment_hash_for::<P>(&record.mu, &w1_bytes);
                assert_eq!(digest.as_ref(), record.c_tilde.as_ref(), "{backend:?} record {i}: w1Encode");
            }
            derive_tile(&engine, &refs, &sigs, &mut out).unwrap();
            for (i, (record, w)) in refs.iter().zip(&out).enumerate() {
                let t: Vec<[i32; N]> = (0..<P as ParamSet>::K).map(|row| record.t[row]).collect();
                let reference = derive_with_t_for::<P>(&record.key, &record.mu, sigs[i], &record.c_tilde, &t).unwrap();
                let mut want = LaneWitness::zero::<P>();
                want.fill_from(&reference);
                assert_eq!(*w, want, "{backend:?} record {i}");
            }
            // A flipped z byte: the reference derivation's verdict.
            let mut bad = records[3].sig.clone();
            bad[<P as ParamSet>::C_TILDE_BYTES + 7] ^= 0x10;
            let mut sigs_bad = sigs.clone();
            sigs_bad[3] = &bad;
            let t: Vec<[i32; N]> = (0..<P as ParamSet>::K).map(|row| refs[3].t[row]).collect();
            let reference = derive_with_t_for::<P>(&refs[3].key, &refs[3].mu, &bad, &refs[3].c_tilde, &t);
            let got = derive_tile(&engine, &refs, &sigs_bad, &mut out);
            match reference {
                Ok(_) => assert!(got.is_ok()),
                Err(e) => assert_eq!(got, Err((3, e))),
            }
        }
    }

    /// A seed that changes on every run, printed so that a failure can be replayed by
    /// passing it to [`matches_the_reference_derivation`].
    fn fresh_seed() -> u64 {
        let ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let seed = (ns as u64) ^ ((ns >> 64) as u64);
        eprintln!("reference-match seed: {seed:#x}");
        seed
    }

    #[test]
    fn ml_dsa_65_engine_path_matches_the_reference_derivation() {
        matches_the_reference_derivation::<Set65>(61);
    }

    #[test]
    fn ml_dsa_87_engine_path_matches_the_reference_derivation() {
        matches_the_reference_derivation::<Set87>(61);
    }

    #[test]
    fn ml_dsa_65_engine_path_matches_the_reference_derivation_on_fresh_seeds() {
        for _ in 0..2 {
            matches_the_reference_derivation::<Set65>(fresh_seed());
        }
    }

    #[test]
    fn ml_dsa_87_engine_path_matches_the_reference_derivation_on_fresh_seeds() {
        for _ in 0..2 {
            matches_the_reference_derivation::<Set87>(fresh_seed());
        }
    }
}
