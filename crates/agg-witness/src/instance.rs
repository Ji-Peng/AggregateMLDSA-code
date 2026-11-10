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
    challenge_from_sparse_tau(c)
}

/// [`challenge_from_sparse`] for any `tau`.
pub fn challenge_from_sparse_tau<const TAU: usize>(c: &agg_mldsa_simd::Challenge<TAU>) -> Challenge {
    let mut coeffs = [0i8; N];
    for (k, &pos) in c.pos.iter().enumerate() {
        coeffs[usize::from(pos)] = if (c.neg >> k) & 1 == 1 { -1 } else { 1 };
    }
    Challenge {
        coeffs,
        positions: c.pos.to_vec(),
    }
}

/// SampleInBall of a tile of seeds (`seeds.len() <= RECORD_TILE`) and `T = 2^d (c t1)` of
/// each record (`k 256` cells per record in `t`) on `engine`, with `tau = TAU` (`P::TAU`).
fn public_tile<P: ParamSet, const TAU: usize>(
    engine: &Engine,
    seeds: &[&[u8]],
    keys: &[&ExpandedKey<P>],
    t: &mut [i32],
) -> Vec<Challenge> {
    assert_eq!(P::TAU, TAU, "the challenge weight of the parameter set");
    let mut sparse = [agg_mldsa_simd::Challenge::<TAU>::zero(); RECORD_TILE];
    let sparse = &mut sparse[..seeds.len()];
    engine.sample_in_ball_many(seeds, sparse);
    let mut challenges = Vec::with_capacity(seeds.len());
    for ((sparse, key), out) in sparse.iter().zip(keys).zip(t.chunks_exact_mut(P::K * N)) {
        challenges.push(challenge_from_sparse_tau(sparse));
        let (rows, _) = out.as_chunks_mut::<N>();
        engine.sparse_ct_slice(sparse, key.t1.as_ref(), D as u32, rows);
    }
    challenges
}

/// The distinct keys of `pks` in first-occurrence order, and per record the index of its key
/// among them. The keys are hashed in parallel (one 64-bit hash per key, under the process's
/// random hash keys); the serial pass then works on the hashes and compares two keys in full
/// only when their hashes are equal.
fn distinct_keys<'k>(pks: &[&'k [u8]]) -> (Vec<&'k [u8]>, Vec<usize>) {
    use std::hash::BuildHasher;
    let state = std::collections::hash_map::RandomState::new();
    distinct_keys_by(pks, |pk| state.hash_one(pk))
}

/// [`distinct_keys`] under the hash `hash` (tests force collisions).
fn distinct_keys_by<'k>(pks: &[&'k [u8]], hash: impl Fn(&[u8]) -> u64 + Sync) -> (Vec<&'k [u8]>, Vec<usize>) {
    use std::collections::hash_map::Entry;
    let hashes: Vec<u64> = pks.par_iter().map(|pk| hash(pk)).collect();
    let mut distinct: Vec<&[u8]> = Vec::new();
    // The first distinct key of every hash; a different key with an equal hash is placed
    // through `collided`, keyed by its bytes.
    let mut first: HashMap<u64, usize> = HashMap::with_capacity(pks.len());
    let mut collided: HashMap<&[u8], usize> = HashMap::new();
    let mut slot = Vec::with_capacity(pks.len());
    for (&pk, &hash) in pks.iter().zip(&hashes) {
        let index = match first.entry(hash) {
            Entry::Vacant(entry) => {
                distinct.push(pk);
                *entry.insert(distinct.len() - 1)
            }
            Entry::Occupied(entry) if distinct[*entry.get()] == pk => *entry.get(),
            Entry::Occupied(_) => *collided.entry(pk).or_insert_with(|| {
                distinct.push(pk);
                distinct.len() - 1
            }),
        };
        slot.push(index);
    }
    (distinct, slot)
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
        // Full-key dedup; `slot[i]` is the distinct key of record i.
        let pks: Vec<&[u8]> = inputs.iter().map(|&(pk, _, _)| pk).collect();
        let (distinct, slot) = distinct_keys(&pks);
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
        let pks: Vec<&[u8]> = inputs.iter().map(|&(pk, _, _)| pk).collect();
        let (distinct, slot) = distinct_keys(&pks);
        // Keys in batches of eight: one batch of SHAKE256 streams for their `tr` and one of
        // SHAKE128 streams for their matrices.
        let keys: Vec<Option<Arc<ExpandedKey<P>>>> = distinct
            .par_chunks(crate::expand::KEY_TILE)
            .flat_map_iter(|pks| P::expand_keys(pks).into_iter().map(|key| key.map(Arc::new)))
            .collect();
        if let Some(i) = slot.iter().position(|&k| keys[k].is_none()) {
            return Err(InstanceError::PublicKey(i));
        }
        let key_of = |i: usize| Arc::clone(keys[slot[i]].as_ref().expect("checked above"));

        // `mu`, `c = SampleInBall(c~)` and `T = 2^d (c t1)` per tile of records on the SIMD
        // engine (as for ML-DSA-44); `T` goes straight into the flat buffer (fresh zeroed
        // pages, first touched by the writers) and each tile returns its own `(mus, c)` chunk
        // so the calling thread does not allocate and fault a batch-wide `Challenge` vector.
        let engine = agg_mldsa_simd::engine();
        let mut t = vec![0_i32; inputs.len() * P::K * N];
        let tiles: Vec<([[u8; TR_BYTES]; RECORD_TILE], Vec<Challenge>)> = t
            .par_chunks_mut(RECORD_TILE * P::K * N)
            .zip(inputs.par_chunks(RECORD_TILE))
            .enumerate()
            .map(|(n, (t, tile))| {
                let first = n * RECORD_TILE;
                let tile_keys: Vec<&ExpandedKey<P>> =
                    (first..first + tile.len()).map(|i| keys[slot[i]].as_deref().expect("checked above")).collect();
                let mu_inputs: Vec<[&[u8]; 4]> = tile_keys
                    .iter()
                    .zip(tile)
                    .map(|(key, (_, msg, _))| [&key.tr[..], &[0u8, 0u8][..], &[][..], *msg])
                    .collect();
                let mut mus = [[0u8; TR_BYTES]; RECORD_TILE];
                engine.shake256_many(&mu_inputs, TR_BYTES, mus[..tile.len()].as_flattened_mut());
                let seeds: Vec<&[u8]> = tile.iter().map(|(_, _, c)| c.as_ref()).collect();
                let challenges = match P::TAU {
                    39 => public_tile::<P, 39>(engine, &seeds, &tile_keys, t),
                    49 => public_tile::<P, 49>(engine, &seeds, &tile_keys, t),
                    60 => public_tile::<P, 60>(engine, &seeds, &tile_keys, t),
                    _ => seeds
                        .iter()
                        .zip(&tile_keys)
                        .zip(t.chunks_exact_mut(P::K * N))
                        .map(|((seed, key), out)| {
                            let c = crate::native::sample_in_ball_for::<P>(seed);
                            let (rows, _) = out.as_chunks_mut::<N>();
                            crate::expand::public_t_into_for::<P>(&c, &key.t1, rows);
                            c
                        })
                        .collect(),
                };
                (mus, challenges)
            })
            .collect();

        let t = Arc::new(t);
        let records = tiles
            .into_par_iter()
            .zip(inputs.par_chunks(RECORD_TILE))
            .enumerate()
            .flat_map_iter(|(n, ((mus, challenges), tile))| {
                let first = n * RECORD_TILE;
                let t = &t;
                challenges
                    .into_iter()
                    .zip(mus)
                    .zip(tile)
                    .enumerate()
                    .map(move |(k, ((c, mu), (_, msg, c_tilde)))| {
                        let i = first + k;
                        PublicRecord {
                            key: key_of(i),
                            msg: msg.to_vec(),
                            c_tilde: *c_tilde,
                            mu,
                            c,
                            t: TView::new(Arc::clone(t), i),
                        }
                    })
            })
            .collect();
        Ok(Self { geom, records })
    }

    /// Builds the instance from records with signatures for parameter set P. A signature
    /// shorter than c~ gives a zero-padded c~ rather than a panic.
    pub fn from_records_for(records: &[crate::gen::Record]) -> Result<Self, InstanceError> {
        let inputs: Vec<(&[u8], &[u8], P::CTilde)> = records
            .iter()
            .map(|r| (r.pk.as_slice(), r.msg.as_slice(), P::c_tilde_from_slice(&r.sig)))
            .collect();
        Self::new_for(&inputs)
    }

    /// `T` of every live record as one flat buffer in record order (`k 256` values per
    /// record), shared with the records' views, so that the public data can hold it without
    /// a copy. `None` when some record no longer views slot `i` of one common buffer of
    /// exactly the live records (a record whose `T` was changed after construction).
    #[must_use]
    pub fn t_buffer(&self) -> Option<&Arc<Vec<i32>>> {
        let data = &self.records.first()?.t.data;
        let whole = data.len() == self.records.len() * P::K * N;
        let own_slots = self
            .records
            .iter()
            .enumerate()
            .all(|(i, record)| record.t.index == i && Arc::ptr_eq(&record.t.data, data));
        (whole && own_slots).then_some(data)
    }

    /// The public table T(j, rho, i) on the main domain, zero on padded rows: one table per
    /// polynomial lane, row `rho` in lane `rho >> D_VEC` at cell
    /// `j + 256 (rho mod 2^D_VEC) + 2^(8 + D_VEC) i` (one table for ML-DSA-44, one per row
    /// for the sets with `D_VEC = 0`).
    pub fn t_lane_tables(&self) -> Vec<Vec<i64>> {
        let lane_cells = N << P::D_VEC;
        let mut out = vec![vec![0i64; self.geom.n_star * lane_cells]; P::LANES_K];
        for (i, r) in self.records.iter().enumerate() {
            for rho in 0..P::K {
                let (lane, offset) = (rho >> P::D_VEC, (rho & ((1 << P::D_VEC) - 1)) * N);
                for j in 0..N {
                    out[lane][i * lane_cells + offset + j] = i64::from(r.t[rho][j]);
                }
            }
        }
        out
    }

    /// The public table T(j, rho, i) on the main domain of ML-DSA-44 (one polynomial lane),
    /// zero on padded rows.
    ///
    /// # Panics
    /// For a parameter set with several polynomial lanes ([`Self::t_lane_tables`]).
    pub fn t_table(&self) -> Vec<i64> {
        assert_eq!(P::LANES_K, 1, "one main-domain table per polynomial lane");
        self.t_lane_tables().pop().expect("one lane")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{Set65, Set87};

    /// The batched `mu`, SampleInBall and `T` of every record equal the per-record scalar
    /// functions (FIPS 204 Algorithms 8 and 29), on fresh and shared keys.
    fn matches_the_scalar_values<P: ParamSet>(seed: u64) {
        let mut records = crate::gen::records_for::<P>(seed, 70);
        records.extend(crate::gen::records_shared_keys_for::<P>(seed + 1, 9, 3));
        let instance = PublicInstance::<P>::from_records_for(&records).unwrap();
        for (r, rec) in instance.records.iter().zip(&records) {
            assert_eq!(r.mu, crate::native::mu(&r.key.tr, &rec.msg, &[]));
            let c = crate::native::sample_in_ball_for::<P>(&rec.sig[..P::C_TILDE_BYTES]);
            assert_eq!(r.c, c);
            let mut t = vec![[0_i32; N]; P::K];
            crate::expand::public_t_into_for::<P>(&c, &r.key.t1, &mut t);
            for (row, want) in t.iter().enumerate() {
                assert_eq!(&r.t[row], want);
            }
        }
    }

    /// The hashed dedup numbers the keys as a full-key map in record order does, also when
    /// distinct keys share a hash.
    #[test]
    fn distinct_keys_match_a_full_key_map() {
        let keys: Vec<Vec<u8>> = (0..40_u8).map(|k| vec![k % 7, k % 3, 1]).collect();
        let pks: Vec<&[u8]> = keys.iter().map(Vec::as_slice).collect();
        let mut want_distinct: Vec<&[u8]> = Vec::new();
        let mut index: HashMap<&[u8], usize> = HashMap::new();
        let want_slot: Vec<usize> = pks
            .iter()
            .map(|&pk| {
                *index.entry(pk).or_insert_with(|| {
                    want_distinct.push(pk);
                    want_distinct.len() - 1
                })
            })
            .collect();
        assert_eq!(distinct_keys(&pks), (want_distinct.clone(), want_slot.clone()));
        let colliding: [fn(&[u8]) -> u64; 2] = [|_| 0, |pk| u64::from(pk[0] % 2)];
        for hash in colliding {
            assert_eq!(distinct_keys_by(&pks, hash), (want_distinct.clone(), want_slot.clone()));
        }
    }

    #[test]
    fn batched_public_values_match_the_scalar_ones() {
        matches_the_scalar_values::<Set44>(3);
        matches_the_scalar_values::<Set65>(5);
        matches_the_scalar_values::<Set87>(7);
    }
}
