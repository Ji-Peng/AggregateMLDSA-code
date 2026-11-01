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
use crate::padding::{Geometry, GeometryError};
use crate::params::{ParamSet, Set44, C_TILDE_BYTES, D, K, N, TR_BYTES};
use agg_mldsa_simd::{Challenge39, Engine};
use core::marker::PhantomData;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/// One record's `T = 2^d (c * t1)` (|T| < 2^29, stored as `i32`) inside the batch's single
/// flat allocation (`k 256` values per record, record order).
#[derive(Clone, Debug)]
pub struct TView<P: ParamSet = Set44> {
    data: Arc<Vec<i32>>,
    index: usize,
    _marker: PhantomData<P>,
}

impl<P: ParamSet> TView<P> {
    /// Record `index` of a flat `T` buffer of whole records.
    ///
    /// # Panics
    /// If the buffer holds no record `index`.
    #[must_use]
    pub fn new(data: Arc<Vec<i32>>, index: usize) -> Self {
        assert!(index < data.len() / (P::K * N), "record inside the T buffer");
        Self {
            data,
            index,
            _marker: PhantomData,
        }
    }
}

impl<P: ParamSet> std::ops::Deref for TView<P> {
    type Target = P::TTarget;
    fn deref(&self) -> &Self::Target {
        let record = &self.data[self.index * P::K * N..(self.index + 1) * P::K * N];
        let (rows, _) = record.as_chunks::<N>();
        P::as_t_target(rows)
    }
}

impl<P: ParamSet> std::ops::DerefMut for TView<P> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        let index = self.index;
        let data = Arc::make_mut(&mut self.data);
        let record = &mut data[index * P::K * N..(index + 1) * P::K * N];
        let (rows, _) = record.as_chunks_mut::<N>();
        P::as_t_target_mut(rows)
    }
}

impl<P: ParamSet> PartialEq for TView<P> {
    fn eq(&self, other: &Self) -> bool {
        let s = &self.data[self.index * P::K * N..(self.index + 1) * P::K * N];
        let o = &other.data[other.index * P::K * N..(other.index + 1) * P::K * N];
        s == o
    }
}

impl<P: ParamSet> Eq for TView<P> {}

/// Public data of one live record.
#[derive(Clone, Debug)]
pub struct PublicRecord<P: ParamSet = Set44> {
    /// Expanded public key.
    pub key: Arc<ExpandedKey<P>>,
    /// Message M.
    pub msg: Vec<u8>,
    /// c~ from the aggregate.
    pub c_tilde: P::CTilde,
    /// mu.
    pub mu: [u8; TR_BYTES],
    /// c = SampleInBall(c~).
    pub c: Challenge,
    /// T = 2^d (c * t1) over Z.
    pub t: TView<P>,
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
pub struct PublicInstance<P: ParamSet = Set44> {
    /// Batch geometry.
    pub geom: Geometry<P>,
    /// Live records, in order.
    pub records: Vec<PublicRecord<P>>,
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

impl PublicInstance<Set44> {
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
        let geom = Geometry::<Set44>::new_for(inputs.len()).map_err(InstanceError::Geometry)?;
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
        let keys: Vec<Option<Arc<ExpandedKey<Set44>>>> = ExpandedKey::many_with(engine, &distinct)
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
                let tile_keys: Vec<&ExpandedKey<Set44>> =
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
}

impl<P: ParamSet> PublicInstance<P> {
    /// Builds the instance for parameter set P from (pk, M, c~) triples.
    pub fn new_for(inputs: &[(&[u8], &[u8], P::CTilde)]) -> Result<Self, InstanceError> {
        let geom = Geometry::<P>::new_for(inputs.len()).map_err(InstanceError::Geometry)?;
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
        let keys: Vec<Option<Arc<ExpandedKey<P>>>> = distinct
            .par_iter()
            .map(|pk| ExpandedKey::<P>::from_bytes_for(pk).map(Arc::new))
            .collect();
        if let Some(i) = slot.iter().position(|&k| keys[k].is_none()) {
            return Err(InstanceError::PublicKey(i));
        }
        let key_of = |i: usize| Arc::clone(keys[slot[i]].as_ref().expect("checked above"));

        let mut mus = vec![[0u8; TR_BYTES]; inputs.len()];
        let mut challenges = Vec::with_capacity(inputs.len());
        let mut t = vec![0_i32; inputs.len() * P::K * N];

        for (i, (_, msg, ct)) in inputs.iter().enumerate() {
            let key = key_of(i);
            mus[i] = crate::native::mu(&key.tr, msg, &[]);
            let c = crate::native::sample_in_ball_for::<P>(ct.as_ref());
            challenges.push(c);
        }

        let (records_t, _) = t.as_chunks_mut::<N>();
        for (i, c) in challenges.iter().enumerate() {
            let key = key_of(i);
            let chunk = &mut records_t[i * P::K..(i + 1) * P::K];
            crate::expand::public_t_into_for::<P>(c, &key.t1, chunk);
        }

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
                c,
                t: TView::new(Arc::clone(&t), i),
            })
            .collect();
        Ok(Self { geom, records })
    }

    /// Builds the instance from records with signatures for parameter set P.
    pub fn from_records_for(records: &[crate::gen::Record]) -> Result<Self, InstanceError> {
        let inputs: Vec<(&[u8], &[u8], P::CTilde)> = records
            .iter()
            .map(|r| {
                let n = r.sig.len().min(P::C_TILDE_BYTES);
                let ct = P::c_tilde_from_slice(&r.sig[..n]);
                (r.pk.as_slice(), r.msg.as_slice(), ct)
            })
            .collect();
        Self::new_for(&inputs)
    }

    /// The public table T(j, rho, i) on the main domain, zero on padded rows.
    pub fn t_table(&self) -> Vec<i64> {
        let mut out = vec![0i64; self.geom.main_len()];
        for (i, r) in self.records.iter().enumerate() {
            for rho in 0..P::K {
                for j in 0..N {
                    let idx = j | (rho << 8) | (i << (8 + P::D_VEC));
                    out[idx] = i64::from(r.t[rho][j]);
                }
            }
        }
        out
    }
}
