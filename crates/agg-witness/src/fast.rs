//! Batched strict witness derivation on the SIMD engine (the prover's path).
//!
//! For a record of a [`PublicInstance`] and its signature, in tiles of [`TILE`] records:
//! decode z (with its norm) and the hint, reuse the instance's `T = 2^d (c t1)` when the
//! signature's c~ is the record's, and compute A z over Z from its residues modulo q and
//! modulo the 28-bit transform prime `P2` of `agg_mldsa_simd` (4 + 4 forward transforms of z,
//! 4 + 4 dot products against the key's prepared matrices, 4 + 4 inverse transforms). The
//! residues are exact: |A z| <= l 256 (q-1)/2 gamma1 < q P2 / 2 for every decodable z. Each
//! row then finishes without division (centered CRT, UseHint on `A z - T` modulo q, the
//! residual u and the exact quotient v, with their bounds), w1' is packed, and one batch of
//! SHAKE256 digests per tile recomputes c~' = H(mu || w1Encode(w1')). Strict acceptance
//! matches FIPS 204 Algorithm 8, since T is the honest product for that c~.
//!
//! A record that fails any step (length, hint, norm, another c~ than the instance's, a
//! finishing bound, a commitment mismatch) is derived again by
//! [`crate::derive::derive_with_t`], so every error is exactly the legacy one, and so is the
//! witness of a record the legacy path accepts.
//!
//! [`PublicInstance`]: crate::instance::PublicInstance

use crate::derive::{derive_with_t, DeriveError, SigWitness};
use crate::instance::PublicRecord;
use crate::native::{hint_bit_unpack, SmallK, VecK, VecL};
use crate::params::{BETA, C_TILDE_BYTES, GAMMA1, K, L, N, SIG_BYTES, W1_ENCODE_BYTES, Z_BITS};
use agg_mldsa_simd::{Engine, FinishOut, FinishRow, Form, Poly32, Prime, Scale};

/// Records per derivation tile (one batch of commitment digests).
pub const TILE: usize = 8;

/// Packed z bytes of a signature.
const Z_BYTES: usize = L * 32 * Z_BITS;

/// The derived values of one record: z (natural order) and per row of A z - T the finishing
/// output `(w1', u, v)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FastWitness {
    /// Response z, `|z| < gamma1 - beta`.
    pub z: [Poly32; L],
    /// Row `rho`: w1' in [0, 44), u with |u| <= B_u, v with |v| <= B_v.
    pub rows: [FinishOut; K],
}

impl FastWitness {
    /// An all-zero value (scratch before a derivation).
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            z: [Poly32::zero(); L],
            rows: [FinishOut::zero(); K],
        }
    }

    /// z widened to the legacy vector type.
    #[must_use]
    pub fn z_wide(&self) -> VecL {
        self.z.map(|p| p.0.map(i64::from))
    }

    /// u widened to the legacy vector type.
    #[must_use]
    pub fn u_wide(&self) -> VecK {
        std::array::from_fn(|r| self.rows[r].u.0.map(i64::from))
    }

    /// v widened to the legacy vector type.
    #[must_use]
    pub fn v_wide(&self) -> VecK {
        std::array::from_fn(|r| self.rows[r].v.0.map(i64::from))
    }

    /// w1' per row.
    #[must_use]
    pub fn w1(&self) -> SmallK {
        std::array::from_fn(|r| self.rows[r].w1)
    }

    /// The values of a legacy witness (every one fits `i32` after the strict checks).
    fn fill_from(&mut self, w: &SigWitness) {
        let narrow = |x: i64| i32::try_from(x).expect("strict witness values fit i32");
        for (dest, src) in self.z.iter_mut().zip(&w.z) {
            *dest = Poly32(src.map(narrow));
        }
        for (r, row) in self.rows.iter_mut().enumerate() {
            row.w1 = w.w1[r];
            row.u = Poly32(w.u[r].map(narrow));
            row.v = Poly32(w.v[r].map(narrow));
        }
    }
}

impl Default for FastWitness {
    fn default() -> Self {
        Self::zero()
    }
}

/// Everything up to the commitment hash for one record; `false` sends the record to the
/// legacy derivation.
fn derive_one(
    engine: &Engine,
    record: &PublicRecord,
    sig: &[u8],
    out: &mut FastWitness,
    w1_bytes: &mut [u8; W1_ENCODE_BYTES],
) -> bool {
    if sig.len() != SIG_BYTES {
        return false;
    }
    let (c_tilde, rest) = sig.split_at(C_TILDE_BYTES);
    // The instance's T is the product for the record's c~ only.
    if c_tilde != record.c_tilde {
        return false;
    }
    let (z_bytes, hint) = rest.split_at(Z_BYTES);
    let z_bytes: &[u8; Z_BYTES] = z_bytes.try_into().expect("z bytes");
    let z_max = engine.unpack_z(z_bytes, &mut out.z);
    if i64::from(z_max) >= GAMMA1 - BETA {
        return false;
    }
    let Some(h) = hint_bit_unpack(hint) else {
        return false;
    };
    let prepared = &record.key.prepared;
    assert_eq!(
        prepared.backend,
        engine.backend(),
        "the key was prepared by another SIMD backend"
    );
    // |z| <= gamma1 = 2^17 is inside the forward transforms' input range |a| <= 2^22.
    let mut z_q = out.z;
    let mut z_p = out.z;
    for (zq, zp) in z_q.iter_mut().zip(z_p.iter_mut()) {
        engine.ntt(Prime::Q, zq);
        engine.ntt(Prime::P2, zp);
    }
    let t = &*record.t;
    let mut x_q = Poly32::zero();
    let mut x_p = Poly32::zero();
    let mut w1 = [[0u8; N]; K];
    for r in 0..K {
        // Canonical A_hat mod q and centered NTT(A) mod P2 against the unreduced NTT(z);
        // the Mont inverse scale cancels the dot's 1/R, so the canonical outputs are
        // (A z)_r mod q and mod P2.
        engine.dot4(Prime::Q, &prepared.a_hat_q[r], &z_q, &mut x_q);
        engine.intt(Prime::Q, &mut x_q, Scale::Mont, Form::Canonical);
        engine.dot4(Prime::P2, &prepared.a_hat_p2[r], &z_p, &mut x_p);
        engine.intt(Prime::P2, &mut x_p, Scale::Mont, Form::Canonical);
        let row = FinishRow {
            xq: &x_q,
            xp: &x_p,
            t: &t[r],
            h: &h[r],
        };
        if engine.finish_row(row, &mut out.rows[r]).is_err() {
            return false;
        }
        w1[r] = out.rows[r].w1;
    }
    engine.pack_w1(&w1, w1_bytes);
    true
}

/// Strict derivation of `records[i]` with signature `sigs[i]` into `out[i]`, in tiles of
/// [`TILE`]. On failure: the lowest index whose signature the legacy derivation rejects,
/// with that derivation's error (the outputs of the other records are then unspecified).
///
/// # Panics
/// If the three slices differ in length, or a key was prepared by another backend.
#[doc(hidden)]
pub fn derive_tile_with(
    engine: &Engine,
    records: &[&PublicRecord],
    sigs: &[&[u8]],
    out: &mut [FastWitness],
) -> Result<(), (usize, DeriveError)> {
    assert_eq!(records.len(), sigs.len(), "one signature per record");
    assert_eq!(records.len(), out.len(), "one output per record");
    for (tile, ((records, sigs), out)) in records
        .chunks(TILE)
        .zip(sigs.chunks(TILE))
        .zip(out.chunks_mut(TILE))
        .enumerate()
    {
        let mut ok = [false; TILE];
        let mut w1_bytes = [[0u8; W1_ENCODE_BYTES]; TILE];
        for i in 0..records.len() {
            ok[i] = derive_one(engine, records[i], sigs[i], &mut out[i], &mut w1_bytes[i]);
        }
        let live: Vec<usize> = (0..records.len()).filter(|&i| ok[i]).collect();
        let inputs: Vec<[&[u8]; 4]> = live
            .iter()
            .map(|&i| [&records[i].mu[..], &w1_bytes[i][..], &[][..], &[][..]])
            .collect();
        let mut digests = [[0u8; C_TILDE_BYTES]; TILE];
        engine.shake256_many(
            &inputs,
            C_TILDE_BYTES,
            digests[..live.len()].as_flattened_mut(),
        );
        for (&i, digest) in live.iter().zip(&digests) {
            ok[i] = *digest == records[i].c_tilde;
        }
        for i in (0..records.len()).filter(|&i| !ok[i]) {
            let record = records[i];
            match derive_with_t(&record.key, &record.mu, sigs[i], &record.c_tilde, &record.t) {
                Ok(w) => out[i].fill_from(&w),
                Err(e) => return Err((tile * TILE + i, e)),
            }
        }
    }
    Ok(())
}

/// Whether the engine path alone accepts the record (no legacy fallback), with its values.
/// Tests use it to show that honest records never take the fallback.
#[doc(hidden)]
pub fn fast_path_only_with(
    engine: &Engine,
    record: &PublicRecord,
    sig: &[u8],
) -> Option<FastWitness> {
    let mut out = FastWitness::zero();
    let mut w1_bytes = [0u8; W1_ENCODE_BYTES];
    if !derive_one(engine, record, sig, &mut out, &mut w1_bytes) {
        return None;
    }
    let mut digest = [0u8; C_TILDE_BYTES];
    engine.shake256_many(
        &[[&record.mu[..], &w1_bytes[..], &[], &[]]],
        C_TILDE_BYTES,
        &mut digest,
    );
    (digest == record.c_tilde).then_some(out)
}
