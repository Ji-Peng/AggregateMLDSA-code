//! Public instance: the verifier's per-record precomputation.
//!
//! For live record i: the expanded key (A_hat_i, A_i, tr_i, shared between records with the same
//! key), mu_i = H(tr_i || 0x00 || 0x00 || M_i, 64), c_i = SampleInBall(c~_i) and
//! T_i = 2^d (c_i * t1_i) over Z. The public main-domain table T(j, rho, i) is zero on padded rows.
//! Keys expand in tiles of 8 (`ExpandedKey::many`); records hash in tiles of 64 (batched mu
//! digests and SampleInBall streams), and T is the sparse product written straight into one
//! flat buffer.

use crate::expand::ExpandedKey;
use crate::native::Challenge;
use crate::padding::{main_index, Geometry, GeometryError};
use crate::params::{C_TILDE_BYTES, D, K, N, TR_BYTES};
use agg_mldsa_simd::{Challenge39, Engine};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/// One record's `T = 2^d (c * t1)` (|T| < 2^29, stored as `i32`) inside the batch's single
/// flat allocation (`k 256` values per record, record order). The records share the
/// allocation; a mutable borrow copies it first (`Arc::make_mut`), so a tampered clone of an
/// instance never changes the original.
#[derive(Clone, Debug)]
pub struct TView {
    data: Arc<Vec<i32>>,
    index: usize,
}

impl TView {
    /// Record `index` of a flat `T` buffer of whole records.
    ///
    /// # Panics
    /// If the buffer holds no record `index`.
    #[must_use]
    pub fn new(data: Arc<Vec<i32>>, index: usize) -> Self {
        assert!(index < data.len() / (K * N), "record inside the T buffer");
        Self { data, index }
    }
}

impl std::ops::Deref for TView {
    type Target = [[i32; N]; K];
    fn deref(&self) -> &Self::Target {
        let record = &self.data[self.index * K * N..(self.index + 1) * K * N];
        let (rows, _) = record.as_chunks::<N>();
        rows.try_into().expect("k rows of 256")
    }
}

impl std::ops::DerefMut for TView {
    fn deref_mut(&mut self) -> &mut Self::Target {
        let index = self.index;
        let data = Arc::make_mut(&mut self.data);
        let record = &mut data[index * K * N..(index + 1) * K * N];
        let (rows, _) = record.as_chunks_mut::<N>();
        rows.try_into().expect("k rows of 256")
    }
}

impl PartialEq for TView {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl Eq for TView {}

/// Public data of one live record.
#[derive(Clone, Debug)]
pub struct PublicRecord {
    /// Expanded public key.
    pub key: Arc<ExpandedKey>,
    /// Message M.
    pub msg: Vec<u8>,
    /// c~ from the aggregate.
    pub c_tilde: [u8; C_TILDE_BYTES],
    /// mu.
    pub mu: [u8; TR_BYTES],
    /// c = SampleInBall(c~).
    pub c: Challenge,
    /// T = 2^d (c * t1) over Z.
    pub t: TView,
}

/// Errors building the instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InstanceError {
    /// Bad batch size.
    Geometry(GeometryError),
    /// Public key of record i has the wrong length.
    PublicKey(usize),
}

/// The public instance of a batch.
#[derive(Clone, Debug)]
pub struct PublicInstance {
    /// Batch geometry.
    pub geom: Geometry,
    /// Live records, in order.
    pub records: Vec<PublicRecord>,
}

/// Records per hashing tile of [`PublicInstance::new`] (mu and SampleInBall jobs).
pub const RECORD_TILE: usize = 64;

/// The challenge of [`crate::native::sample_in_ball`] from its sparse form.
#[must_use]
pub fn challenge_from_sparse(c: &Challenge39) -> Challenge {
    let mut coeffs = [0i8; N];
    for (k, &pos) in c.pos.iter().enumerate() {
        coeffs[usize::from(pos)] = if (c.neg >> k) & 1 == 1 { -1 } else { 1 };
    }
    Challenge {
        coeffs,
        positions: c.pos.to_vec(),
    }
}

impl PublicInstance {
    /// Builds the instance from (pk, M, c~) triples; keys are expanded once per distinct pk.
    pub fn new(inputs: &[(&[u8], &[u8], [u8; C_TILDE_BYTES])]) -> Result<Self, InstanceError> {
        Self::new_with(agg_mldsa_simd::engine(), inputs)
    }

    /// [`PublicInstance::new`] with an explicit engine (tests compare every backend).
    #[doc(hidden)]
    pub fn new_with(
        engine: &Engine,
        inputs: &[(&[u8], &[u8], [u8; C_TILDE_BYTES])],
    ) -> Result<Self, InstanceError> {
        let geom = Geometry::new(inputs.len()).map_err(InstanceError::Geometry)?;
        // Full-key dedup; `slot[i]` is the distinct key of record i (each key hashed once).
        let mut distinct: Vec<&[u8]> = Vec::new();
        let mut index: HashMap<&[u8], usize> = HashMap::new();
        let slot: Vec<usize> = inputs
            .iter()
            .map(|(pk, _, _)| {
                *index.entry(pk).or_insert_with(|| {
                    distinct.push(pk);
                    distinct.len() - 1
                })
            })
            .collect();
        let keys: Vec<Option<Arc<ExpandedKey>>> = ExpandedKey::many_with(engine, &distinct)
            .into_iter()
            .map(|key| key.map(Arc::new))
            .collect();
        // The first record whose key does not decode, as the serial order finds it.
        if let Some(i) = slot.iter().position(|&k| keys[k].is_none()) {
            return Err(InstanceError::PublicKey(i));
        }
        let key_of = |i: usize| Arc::clone(keys[slot[i]].as_ref().expect("checked above"));
        // Per tile of records: one batch of mu digests, one of SampleInBall streams, then T
        // straight into the flat buffer (fresh zeroed pages, first touched by the writers).
        let mut mus = vec![[0u8; TR_BYTES]; inputs.len()];
        let mut challenges = vec![Challenge39::zero(); inputs.len()];
        let mut t = vec![0_i32; inputs.len() * K * N];
        mus.par_chunks_mut(RECORD_TILE)
            .zip(challenges.par_chunks_mut(RECORD_TILE))
            .zip(t.par_chunks_mut(RECORD_TILE * K * N))
            .zip(inputs.par_chunks(RECORD_TILE))
            .enumerate()
            .for_each(|(n, (((mus, challenges), t), tile))| {
                let first = n * RECORD_TILE;
                let tile_keys: Vec<&ExpandedKey> =
                    (first..first + tile.len()).map(|i| keys[slot[i]].as_deref().expect("checked above")).collect();
                let mu_inputs: Vec<[&[u8]; 4]> = tile_keys
                    .iter()
                    .zip(tile)
                    .map(|(key, (_, msg, _))| [&key.tr[..], &[0u8, 0u8][..], &[][..], *msg])
                    .collect();
                engine.shake256_many(&mu_inputs, TR_BYTES, mus.as_flattened_mut());
                let seeds: Vec<[u8; C_TILDE_BYTES]> = tile.iter().map(|(_, _, c)| *c).collect();
                engine.sample_in_ball_many(&seeds, challenges);
                let (records_t, _) = t.as_chunks_mut::<{ K * N }>();
                for ((out, c), key) in records_t.iter_mut().zip(challenges.iter()).zip(&tile_keys) {
                    let (rows, _) = out.as_chunks_mut::<N>();
                    engine.sparse_ct(c, &key.t1, D, rows.try_into().expect("k rows of 256"));
                }
            });
        let t = Arc::new(t);
        let records = mus
            .into_par_iter()
            .zip(challenges)
            .zip(inputs)
            .enumerate()
            .map(|(i, ((mu, c), (_, msg, c_tilde)))| PublicRecord {
                key: key_of(i),
                msg: msg.to_vec(),
                c_tilde: *c_tilde,
                mu,
                c: challenge_from_sparse(&c),
                t: TView::new(Arc::clone(&t), i),
            })
            .collect();
        Ok(Self { geom, records })
    }

    /// Builds the instance from records with signatures (c~ = the first 32 signature bytes).
    pub fn from_records(records: &[crate::gen::Record]) -> Result<Self, InstanceError> {
        let inputs: Vec<(&[u8], &[u8], [u8; C_TILDE_BYTES])> = records
            .iter()
            .map(|r| {
                let mut ct = [0u8; C_TILDE_BYTES];
                let n = r.sig.len().min(C_TILDE_BYTES);
                ct[..n].copy_from_slice(&r.sig[..n]);
                (r.pk.as_slice(), r.msg.as_slice(), ct)
            })
            .collect();
        Self::new(&inputs)
    }

    /// The public table T(j, rho, i) on the main domain, zero on padded rows.
    pub fn t_table(&self) -> Vec<i64> {
        let mut out = vec![0i64; self.geom.main_len()];
        for (i, r) in self.records.iter().enumerate() {
            for rho in 0..K {
                for j in 0..N {
                    out[main_index(j, rho, i)] = i64::from(r.t[rho][j]);
                }
            }
        }
        out
    }
}
