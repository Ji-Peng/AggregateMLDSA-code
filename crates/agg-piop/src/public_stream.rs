//! The verifier's streamed public data.
//!
//! The light precomputation keeps per live record only what the transcript and the Keccak
//! side need before the sum-checks: `pkDecode` (`rho`, `t1`), `tr = H(pk)` once per distinct
//! key, `mu` and `c = SampleInBall(c~)`. Nothing of size `k l 256` is stored: `A_i` and
//! `T_i` are formed inside the one public pass, after the stage-3 rounds have fixed `r^(3)`
//! ([`StreamedPublic::pass`]):
//!
//! - `T(r^(2)) = sum_i eq(r2_sig, i) sum_{rho, j} eq(r2_pos, (j, rho)) T_i[rho][j]`, with
//!   `T_i = 2^d (c_i * t1)` recomputed from the `tau` sparse entries of `c_i`
//!   ([`sparse_t`], `k tau 256` integer additions) and dropped after use;
//! - `w_A(r^(3))`: per distinct key the expansion `A = ctr_q(NTT^{-1}(ExpandA(rho)))`
//!   (or the cached one), its `k l` inner products with the correlation vector `Lambda`
//!   ([`eq_correlation`]), weighted by `sum_{i of the key} eq(r2_sig, i) eq(r3_sig, i)`.
//!
//! Memory is `O(N)` small records plus one `A` per worker thread, instead of 11.8 KB per key
//! and 4 KB per record. New expansions are added to the [`KeyCache`] when one is given.

use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;
use std::sync::Arc;
use std::time::{Duration, Instant};

use aerie_falcon::protocol::algebra::eq_table;
use agg_mldsa_simd::{Challenge39, ExpandedA, Form, Prime, Scale};
use agg_witness::expand::MatrixCoeff;
use agg_witness::instance::InstanceError;
use agg_witness::native::{pk_decode, Challenge};
use agg_witness::padding::Geometry;
use agg_witness::params::{C_TILDE_BYTES, D, D_COEF, D_POS, K, L, N, PK_BYTES, TR_BYTES};
use jolt_field::{Ring, Zero};
use rayon::prelude::*;

use crate::public_data::{
    eq_correlation, sparse_t, KeyCache, SplitWeights, VerifierKey,
};
use crate::OpeningField as E;

/// One distinct key of the batch.
///
/// A cached key holds only the cache entry, whose `t1` and `tr` were derived from the same
/// encoded key (the cache is keyed by the full `pk`; every insertion path decodes `t1` from
/// `pk`, and the cache file recomputes it on load), so the verifier neither decodes nor
/// copies the key again: at 32K keys the copies would be about 110 MB of fresh memory.
#[derive(Clone, Debug)]
pub struct StreamKey {
    source: KeySource,
    /// Live records signed under this key, increasing.
    pub records: Vec<usize>,
}

#[derive(Clone, Debug)]
enum KeySource {
    Cached(Arc<VerifierKey>),
    Fresh(Box<FreshKey>),
}

#[derive(Clone, Debug)]
struct FreshKey {
    pk: Vec<u8>,
    rho: [u8; 32],
    t1: [[u16; N]; K],
    tr: [u8; TR_BYTES],
}

impl StreamKey {
    /// True when the expansion came from the cache.
    #[must_use]
    pub fn is_cached(&self) -> bool {
        matches!(self.source, KeySource::Cached(_))
    }

    /// Encoded public key (the cache key).
    #[must_use]
    pub fn pk(&self) -> &[u8] {
        match &self.source {
            KeySource::Cached(key) => &key.pk,
            KeySource::Fresh(key) => &key.pk,
        }
    }

    /// `t1` of the key.
    #[must_use]
    pub fn t1(&self) -> &[[u16; N]; K] {
        match &self.source {
            KeySource::Cached(key) => &key.t1,
            KeySource::Fresh(key) => &key.t1,
        }
    }

    /// `tr = H(pk, 64)`.
    #[must_use]
    pub fn tr(&self) -> &[u8; TR_BYTES] {
        match &self.source {
            KeySource::Cached(key) => &key.tr,
            KeySource::Fresh(key) => &key.tr,
        }
    }
}

/// One live record.
#[derive(Clone, Debug)]
pub struct StreamRecord {
    /// Index of the record's key in [`StreamedPublic::keys`].
    pub key: usize,
    /// `mu = H(tr || 0 || 0 || M, 64)`.
    pub mu: [u8; TR_BYTES],
    /// `c~` from the aggregate.
    pub c_tilde: [u8; C_TILDE_BYTES],
    /// `c = SampleInBall(c~)` in the packed form of the engine kernels (no heap part, so the
    /// records are one allocation to build and to free).
    packed: Challenge39,
}

impl StreamRecord {
    /// `c = SampleInBall(c~)`.
    #[must_use]
    pub fn challenge(&self) -> Challenge {
        challenge_from_c39(&self.packed)
    }
}

/// What the public pass returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PublicPass {
    /// `T(r^(2))`.
    pub t_at_r2: E,
    /// `w_A(r^(3))` for the ring weight of `r^(2)`.
    pub ring_weight: E,
    /// Keys expanded in this pass (not cached).
    pub expanded: usize,
    /// Wall time of the `T` part.
    pub t_time: Duration,
    /// Wall time of the `A` part (expansion, inverse NTTs, `Lambda` products).
    pub a_time: Duration,
}

/// The verifier's light public data.
#[derive(Clone, Debug)]
pub struct StreamedPublic {
    geom: Geometry,
    keys: Vec<StreamKey>,
    records: Vec<StreamRecord>,
}

/// The distinct keys with their records, and the key index of every record.
pub(crate) type DistinctKeys<'a> = (Vec<(&'a [u8], Vec<usize>)>, Vec<usize>);

/// Groups the records by their full encoded public key: the distinct keys in order of first
/// appearance, each with its records in increasing order, and the key index of every record.
///
/// Keys are equal only when all their bytes are equal. Never group by a prefix: the first 32
/// bytes are `rho`, which the signer chooses, so a key `(rho_victim, t1')` must stay distinct
/// from the victim's key (its records are checked against its own `t1'`). The work is
/// parallel: a keyed SipHash of every key (`RandomState`, so an adversary cannot aim for
/// collisions; a collision only costs a byte comparison), a parallel sort of the record
/// indices by (hash, key bytes, index), one linear scan for the runs of equal keys and a sort
/// of the runs by their first record.
pub(crate) fn distinct_keys<'a, C: Sync>(inputs: &[(&'a [u8], &[u8], C)]) -> DistinctKeys<'a> {
    let state = RandomState::new();
    let hashes: Vec<u64> = inputs
        .par_iter()
        .map(|&(pk, _, _)| state.hash_one(pk))
        .collect();
    let mut order: Vec<usize> = (0..inputs.len()).collect();
    order.par_sort_unstable_by(|&a, &b| {
        hashes[a]
            .cmp(&hashes[b])
            .then_with(|| inputs[a].0.cmp(inputs[b].0))
            .then(a.cmp(&b))
    });
    // Runs of equal keys in `order` (each run increasing in the record index).
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut start = 0;
    for end in 1..=order.len() {
        if end == order.len()
            || hashes[order[end]] != hashes[order[start]]
            || inputs[order[end]].0 != inputs[order[start]].0
        {
            runs.push((start, end));
            start = end;
        }
    }
    runs.par_sort_unstable_by_key(|&(start, _)| order[start]);
    let mut record_keys = vec![0_usize; inputs.len()];
    for (key, &(start, end)) in runs.iter().enumerate() {
        for &i in &order[start..end] {
            record_keys[i] = key;
        }
    }
    let distinct = runs
        .into_iter()
        .map(|(start, end)| (inputs[order[start]].0, order[start..end].to_vec()))
        .collect();
    (distinct, record_keys)
}

/// Converts a sparse `Challenge` to the packed SIMD representation `Challenge39`.
#[cfg(test)]
#[inline]
fn challenge_to_c39(c: &Challenge) -> Challenge39 {
    let mut pos = [0_u8; 39];
    let mut neg = 0_u64;
    for (k, &p) in c.positions.iter().enumerate() {
        pos[k] = p;
        if c.coeffs[p as usize] < 0 {
            neg |= 1_u64 << k;
        }
    }
    Challenge39 { pos, neg }
}

/// Converts a packed SIMD `Challenge39` to the standard `Challenge`.
#[inline]
fn challenge_from_c39(c39: &Challenge39) -> Challenge {
    let mut coeffs = [0_i8; N];
    let mut positions = Vec::with_capacity(39);
    for (k, &pos) in c39.pos.iter().enumerate() {
        let bit = (c39.neg >> k) & 1;
        let sign = if bit != 0 { -1_i8 } else { 1_i8 };
        coeffs[pos as usize] = sign;
        positions.push(pos);
    }
    Challenge { coeffs, positions }
}

impl StreamedPublic {
    /// The light precomputation from `(pk, M, c~)` triples (pure mode, empty context): the
    /// distinct keys by full `pk` bytes ([`distinct_keys`]), `pkDecode` and `tr` per distinct
    /// uncached key (a cached key reuses its entry), `mu` and `SampleInBall` per record.
    pub fn prepare(
        inputs: &[(&[u8], &[u8], [u8; C_TILDE_BYTES])],
        cache: Option<&KeyCache>,
    ) -> Result<Self, InstanceError> {
        Self::prepare_with(agg_mldsa_simd::engine(), inputs, cache)
    }

    /// Prepares public data using the specified SIMD engine.
    #[doc(hidden)]
    pub fn prepare_with(
        engine: &agg_mldsa_simd::Engine,
        inputs: &[(&[u8], &[u8], [u8; C_TILDE_BYTES])],
        cache: Option<&KeyCache>,
    ) -> Result<Self, InstanceError> {
        let geom = Geometry::new(inputs.len()).map_err(InstanceError::Geometry)?;
        let (mut distinct, record_keys) = distinct_keys(inputs);
        let keys: Vec<StreamKey> = distinct
            .par_chunks_mut(8)
            .map(|tile| -> Result<Vec<StreamKey>, InstanceError> {
                let mut tile_keys = Vec::with_capacity(tile.len());
                // Positions in `tile_keys` of the keys that still need `tr`.
                let mut fresh_at = [0_usize; 8];
                let mut fresh_count = 0;
                for (pk, records) in tile.iter_mut() {
                    let source = match cache.and_then(|c| c.get(pk)) {
                        // The length check keeps the error of pkDecode for any cache content.
                        Some(cached) if pk.len() == PK_BYTES => KeySource::Cached(cached),
                        _ => {
                            let decoded =
                                pk_decode(pk).ok_or(InstanceError::PublicKey(records[0]))?;
                            fresh_at[fresh_count] = tile_keys.len();
                            fresh_count += 1;
                            KeySource::Fresh(Box::new(FreshKey {
                                pk: pk.to_vec(),
                                rho: decoded.rho,
                                t1: decoded.t1,
                                tr: [0; TR_BYTES],
                            }))
                        }
                    };
                    tile_keys.push(StreamKey {
                        source,
                        records: std::mem::take(records),
                    });
                }
                if fresh_count > 0 {
                    let mut tr_inputs = [[&[][..]; 4]; 8];
                    for (slot, &at) in tr_inputs.iter_mut().zip(&fresh_at[..fresh_count]) {
                        if let KeySource::Fresh(fresh) = &tile_keys[at].source {
                            *slot = [&fresh.pk, &[], &[], &[]];
                        }
                    }
                    let mut tr_out = [0_u8; 8 * TR_BYTES];
                    engine.shake256_many(
                        &tr_inputs[..fresh_count],
                        TR_BYTES,
                        &mut tr_out[..fresh_count * TR_BYTES],
                    );
                    for (digest, &at) in tr_out.chunks_exact(TR_BYTES).zip(&fresh_at[..fresh_count]) {
                        if let KeySource::Fresh(fresh) = &mut tile_keys[at].source {
                            fresh.tr.copy_from_slice(digest);
                        }
                    }
                }
                Ok(tile_keys)
            })
            .collect::<Result<Vec<Vec<StreamKey>>, InstanceError>>()?
            .into_iter()
            .flatten()
            .collect();

        let records: Vec<StreamRecord> = inputs
            .par_chunks(64)
            .zip(record_keys.par_chunks(64))
            .flat_map_iter(|(input_tile, key_tile)| {
                let count = input_tile.len();
                let mut mu_inputs = [[&[][..]; 4]; 64];
                for j in 0..count {
                    let key_idx = key_tile[j];
                    let msg = input_tile[j].1;
                    mu_inputs[j] = [keys[key_idx].tr(), &[0u8, 0u8], &[], msg];
                }
                let mut mu_out = [0u8; 64 * TR_BYTES];
                engine.shake256_many(
                    &mu_inputs[..count],
                    TR_BYTES,
                    &mut mu_out[..count * TR_BYTES],
                );

                let mut c_tildes = [[0u8; C_TILDE_BYTES]; 64];
                for j in 0..count {
                    c_tildes[j] = input_tile[j].2;
                }
                let mut c39_out = [Challenge39::zero(); 64];
                engine.sample_in_ball_many(&c_tildes[..count], &mut c39_out[..count]);

                (0..count).map(move |j| {
                    let mut mu = [0u8; TR_BYTES];
                    mu.copy_from_slice(&mu_out[j * TR_BYTES..(j + 1) * TR_BYTES]);
                    StreamRecord {
                        key: key_tile[j],
                        mu,
                        c_tilde: input_tile[j].2,
                        packed: c39_out[j],
                    }
                })
            })
            .collect();

        Ok(Self {
            geom,
            keys,
            records,
        })
    }

    /// Batch geometry.
    #[must_use]
    pub const fn geometry(&self) -> Geometry {
        self.geom
    }

    /// Distinct keys.
    #[must_use]
    pub fn keys(&self) -> &[StreamKey] {
        &self.keys
    }

    /// Live records.
    #[must_use]
    pub fn records(&self) -> &[StreamRecord] {
        &self.records
    }

    /// `T_i` of live record `i`, recomputed (`k tau 256` integer additions).
    #[must_use]
    pub fn t(&self, i: usize) -> Box<crate::public_data::PublicT> {
        let record = &self.records[i];
        Box::new(sparse_t(&record.challenge(), self.keys[record.key].t1()))
    }

    /// `T(r^(2))` and `w_A(r^(3))` in one streamed pass (module docs). New
    /// expansions are added to `cache` when one is given.
    pub fn pass(&self, r2: &[E], r3: &[E], cache: Option<&mut KeyCache>) -> PublicPass {
        self.pass_with(agg_mldsa_simd::engine(), r2, r3, cache)
    }

    /// Executes the streamed public pass using the specified SIMD engine.
    #[doc(hidden)]
    pub fn pass_with(
        &self,
        engine: &agg_mldsa_simd::Engine,
        r2: &[E],
        r3: &[E],
        cache: Option<&mut KeyCache>,
    ) -> PublicPass {
        let d_main = self.geom.d_main() as usize;
        assert_eq!(r2.len(), d_main, "a main-domain point");
        assert_eq!(r3.len(), d_main, "a main-domain point");
        let (coef, pos) = (D_COEF as usize, D_POS as usize);
        let eq_pos2 = eq_table(&r2[..pos]);
        let eq_sig2 = eq_table(&r2[pos..]);
        let eq_sig3 = eq_table(&r3[pos..]);

        let start = Instant::now();
        // `<T_i, eq> = 2^d <c_i t1, eq>` with `|c_i t1| < 2^16` (SplitWeights::dot, 16 bits).
        let eq_pos2 = SplitWeights::new(&eq_pos2);
        let t_at_r2 = self
            .records
            .par_iter()
            .enumerate()
            .map(|(i, record)| {
                let mut ct = [[0_i32; N]; K];
                engine.sparse_ct(&record.packed, self.keys[record.key].t1(), 0, &mut ct);
                eq_sig2[i] * eq_pos2.dot::<16>(ct.as_flattened())
            })
            .reduce(E::zero, |a, b| a + b)
            * E::from_u64(1 << D);
        let t_time = start.elapsed();

        let start = Instant::now();
        let lambda = SplitWeights::new(&eq_correlation(&r2[..coef], &r3[..coef]));
        let eq_row = eq_table(&r2[coef..pos]);
        let eq_col = eq_table(&r3[coef..pos]);
        let entry_weights: [[E; L]; K] = std::array::from_fn(|rho| {
            std::array::from_fn(|s| eq_row[rho] * eq_col[s])
        });
        let keep = cache.is_some();

        let per_tile: Vec<(E, Vec<Arc<VerifierKey>>)> = self
            .keys
            .par_chunks(8)
            .map(|tile| {
                let mut fresh_rhos = [[0u8; 32]; 8];
                let mut k_fresh = 0;
                for key in tile {
                    if let KeySource::Fresh(fresh) = &key.source {
                        fresh_rhos[k_fresh] = fresh.rho;
                        k_fresh += 1;
                    }
                }

                let mut expanded = vec![ExpandedA::zero(); k_fresh];
                if k_fresh > 0 {
                    engine.expand_a_many(&fresh_rhos[..k_fresh], &mut expanded);
                }

                let mut fresh_cur = 0;
                let mut tile_fresh_keys = Vec::new();
                let mut tile_weight = E::zero();

                for key in tile {
                    let signature_weight = key
                        .records
                        .iter()
                        .fold(E::zero(), |sum, &i| sum + eq_sig2[i] * eq_sig3[i]);
                    let (weight, fresh_key) = match &key.source {
                        KeySource::Cached(cached) => {
                            (key_weight(&cached.a, &lambda, &entry_weights), None)
                        }
                        KeySource::Fresh(fresh) => {
                            let exp = &mut expanded[fresh_cur];
                            fresh_cur += 1;
                            for row in exp.a_hat.iter_mut() {
                                for poly in row.iter_mut() {
                                    engine.intt(Prime::Q, poly, Scale::Plain, Form::Centered);
                                }
                            }
                            let weight = key_weight_by(
                                |rho, s| &exp.a_hat[rho][s].0,
                                &lambda,
                                &entry_weights,
                            );
                            let fresh_key = keep.then(|| {
                                Arc::new(VerifierKey {
                                    pk: fresh.pk.clone(),
                                    a: Box::new(std::array::from_fn(|rho| {
                                        std::array::from_fn(|s| exp.a_hat[rho][s].0)
                                    })),
                                    t1: fresh.t1,
                                    tr: fresh.tr,
                                })
                            });
                            (weight, fresh_key)
                        }
                    };
                    tile_weight += signature_weight * weight;
                    if let Some(key) = fresh_key {
                        tile_fresh_keys.push(key);
                    }
                }
                (tile_weight, tile_fresh_keys)
            })
            .collect();

        let mut ring_weight = E::zero();
        let expanded = self.keys.iter().filter(|key| !key.is_cached()).count();
        if let Some(cache) = cache {
            for (w, keys) in per_tile {
                ring_weight += w;
                for key in keys {
                    cache.insert(key);
                }
            }
        } else {
            for (w, _) in per_tile {
                ring_weight += w;
            }
        }

        PublicPass {
            t_at_r2,
            ring_weight,
            expanded,
            t_time,
            a_time: start.elapsed(),
        }
    }
}

/// `sum_{rho, s} eq(r2_row, rho) eq(r3_col, s) <A[rho][s], Lambda>` (the per-key factor of
/// `w_A(r^(3))`), with `entry_weights[rho][s] = eq(r2_row, rho) eq(r3_col, s)`.
///
/// `A` entries are centered mod `q`, `|A| <= (q - 1) / 2 < 2^22`, so each of the `k l` inner
/// products of 256 terms is accumulated exactly per base coordinate of `Lambda` and reduced
/// once ([`SplitWeights::dot`] with 22 bits; in `i128` terms the bound is
/// `256 x 2^64 x 2^22 = 2^94 < 2^95`). Out-of-range entries take the exact `i128` path.
fn key_weight(a: &MatrixCoeff, lambda: &SplitWeights, entry_weights: &[[E; L]; K]) -> E {
    key_weight_by(|rho, s| &a[rho][s], lambda, entry_weights)
}

/// [`key_weight`] over entries given by an accessor, so that a transformed matrix can be
/// read in place without a copy.
fn key_weight_by<'a>(
    entry: impl Fn(usize, usize) -> &'a [i32; N],
    lambda: &SplitWeights,
    entry_weights: &[[E; L]; K],
) -> E {
    debug_assert_eq!(lambda.len(), N);
    let mut sum = E::zero();
    for (rho, weights) in entry_weights.iter().enumerate() {
        for (s, &weight) in weights.iter().enumerate() {
            sum += weight * lambda.dot::<22>(entry(rho, s));
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::public_data::{sparse_ct, PublicData};
    use crate::test_support::point;
    use crate::weights::ring_weight_at;
    use agg_witness::gen::{self, SplitMix64};
    use agg_witness::instance::PublicInstance;
    use agg_witness::native::{mu, sample_in_ball, Challenge};
    use agg_witness::params::{P_FIELD, Q, TAU};
    use jolt_field::Ring;
    use std::collections::HashMap;
    use std::hint::black_box;
    use std::time::Instant;

    use crate::BaseField;

    // Plain implementations of the kernels, kept as references.

    fn sparse_t_reference(c: &Challenge, t1: &[[u16; N]; K]) -> crate::public_data::PublicT {
        let mut out = [[0_i32; N]; K];
        for (row, t1_row) in out.iter_mut().zip(t1) {
            for &position in &c.positions {
                let p = usize::from(position);
                let sign = i32::from(c.coeffs[p]);
                for (m, &t) in t1_row.iter().enumerate() {
                    let value = sign * i32::from(t);
                    if p + m < N {
                        row[p + m] += value;
                    } else {
                        row[p + m - N] -= value;
                    }
                }
            }
            for coeff in row.iter_mut() {
                *coeff <<= D;
            }
        }
        out
    }

    fn dot_reference(values: &[i32], weights: &[E]) -> E {
        values
            .iter()
            .zip(weights)
            .fold(E::zero(), |sum, (&value, &weight)| {
                sum + weight.mul_i64(i64::from(value))
            })
    }

    fn key_weight_reference(a: &MatrixCoeff, lambda: &[E], eq_row: &[E], eq_col: &[E]) -> E {
        let mut sum = E::zero();
        for rho in 0..K {
            for s in 0..L {
                sum += eq_row[rho] * eq_col[s] * dot_reference(&a[rho][s], lambda);
            }
        }
        sum
    }

    fn distinct_keys_reference<'a>(
        inputs: &[(&'a [u8], &[u8], [u8; C_TILDE_BYTES])],
    ) -> DistinctKeys<'a> {
        let mut index: HashMap<&[u8], usize> = HashMap::new();
        let mut distinct: Vec<(&[u8], Vec<usize>)> = Vec::new();
        let record_keys = inputs
            .iter()
            .enumerate()
            .map(|(i, &(pk, _, _))| {
                let key = *index.entry(pk).or_insert_with(|| {
                    distinct.push((pk, Vec::new()));
                    distinct.len() - 1
                });
                distinct[key].1.push(i);
                key
            })
            .collect();
        (distinct, record_keys)
    }

    fn random_challenge(rng: &mut SplitMix64) -> Challenge {
        let mut c_tilde = [0_u8; C_TILDE_BYTES];
        rng.fill(&mut c_tilde);
        sample_in_ball(&c_tilde)
    }

    /// `tau` entries of one sign at the given positions.
    fn fixed_challenge(positions: impl Iterator<Item = usize>, sign: i8) -> Challenge {
        let mut coeffs = [0_i8; N];
        let positions: Vec<u8> = positions.map(|p| p as u8).collect();
        for &p in &positions {
            coeffs[usize::from(p)] = sign;
        }
        Challenge { coeffs, positions }
    }

    fn random_t1(rng: &mut SplitMix64) -> [[u16; N]; K] {
        std::array::from_fn(|_| std::array::from_fn(|_| rng.below(1024) as u16))
    }

    fn random_a(rng: &mut SplitMix64) -> Box<MatrixCoeff> {
        let half = (Q - 1) / 2;
        let mut a = Box::new([[[0_i32; N]; L]; K]);
        for entry in a.iter_mut().flatten().flatten() {
            *entry = (rng.below(Q as u64) as i64 - half) as i32;
        }
        a
    }

    fn max_field() -> E {
        E::new(BaseField::from_u64(P_FIELD - 1), BaseField::from_u64(P_FIELD - 1))
    }

    #[test]
    fn sparse_t_matches_the_reference_including_extremes() {
        let mut rng = SplitMix64::new(5);
        for _ in 0..200 {
            let (c, t1) = (random_challenge(&mut rng), random_t1(&mut rng));
            assert_eq!(sparse_t(&c, &t1), sparse_t_reference(&c, &t1));
        }
        let ones = [[1023_u16; N]; K];
        for sign in [1, -1] {
            for c in [
                fixed_challenge(0..TAU, sign),
                fixed_challenge(N - TAU..N, sign),
                fixed_challenge((0..TAU).map(|i| i * 6 + 1), sign),
            ] {
                let t = sparse_t(&c, &ones);
                assert_eq!(t, sparse_t_reference(&c, &ones));
                assert!(t.iter().flatten().all(|v| v.unsigned_abs() < 1 << 29));
            }
        }
    }

    #[test]
    fn split_dot_matches_the_extension_products_including_extremes() {
        let mut rng = SplitMix64::new(6);
        let top = SplitWeights::new(&vec![max_field(); 1024]);
        for weights in [point(&mut rng, 1024), vec![max_field(); 1024]] {
            let split = SplitWeights::new(&weights);
            let check = |values: &[i32]| {
                let expected = dot_reference(values, &weights);
                assert_eq!(split.dot_wide(values), expected);
                assert_eq!(split.dot::<16>(values), expected);
                assert_eq!(split.dot::<22>(values), expected); // 1,024 > 2^9 terms: wide path
                let quarter = SplitWeights::new(&weights[..256]);
                let expected = dot_reference(&values[..256], &weights);
                assert_eq!(quarter.dot::<22>(&values[..256]), expected);
                assert_eq!(quarter.dot::<16>(&values[..256]), expected);
            };
            // In range for 16 bits, then for 22 bits, then anything (the fallback).
            check(&(0..1024).map(|_| rng.below(1 << 17) as i32 - (1 << 16)).collect::<Vec<_>>());
            check(&(0..1024).map(|_| rng.below(1 << 23) as i32 - (1 << 22)).collect::<Vec<_>>());
            check(&(0..1024).map(|_| rng.next_u64() as i32).collect::<Vec<_>>());
            for v in [
                i32::MIN,
                i32::MAX,
                -(1 << 22),
                (1 << 22) - 1,
                1 << 22,
                -(1 << 16),
                (1 << 16) - 1,
                1 << 16,
                -(1 << 16) - 1,
            ] {
                check(&vec![v; 1024]);
            }
            // One out-of-range value among in-range ones.
            let mut values = vec![(1 << 16) - 1; 1024];
            values[777] = 1 << 16;
            check(&values);
            // The fast path is taken exactly on in-range values within the length bound.
            let quarter = SplitWeights::new(&weights[..256]);
            for (v, in16, in22) in [
                (-(1 << 16), true, true),
                ((1 << 16) - 1, true, true),
                (1 << 16, false, true),
                (-(1 << 16) - 1, false, true),
                (-(1 << 22), false, true),
                ((1 << 22) - 1, false, true),
                (1 << 22, false, false),
                (i32::MIN, false, false),
            ] {
                assert_eq!(split.dot_lanes::<16>(&vec![v; 1024]).is_some(), in16, "{v}");
                assert_eq!(quarter.dot_lanes::<22>(&vec![v; 256]).is_some(), in22, "{v}");
            }
            assert!(split.dot_lanes::<22>(&vec![0; 1024]).is_none(), "2^10 > 2^9 terms");
            assert!(split.dot_lanes::<16>(&values).is_none());
        }
        assert_eq!(top.len(), 1024);
    }

    #[test]
    fn key_weight_matches_the_reference_including_extremes() {
        let mut rng = SplitMix64::new(7);
        let lambda: Vec<E> = point(&mut rng, N);
        let (eq_row, eq_col) = (point(&mut rng, K), point(&mut rng, L));
        let weights: [[E; L]; K] =
            std::array::from_fn(|rho| std::array::from_fn(|s| eq_row[rho] * eq_col[s]));
        let split = SplitWeights::new(&lambda);
        for _ in 0..8 {
            let a = random_a(&mut rng);
            assert_eq!(
                key_weight(&a, &split, &weights),
                key_weight_reference(&a, &lambda, &eq_row, &eq_col)
            );
        }
        let top = vec![max_field(); N];
        let top_split = SplitWeights::new(&top);
        let half = ((Q - 1) / 2) as i32;
        for v in [half, -half] {
            let a = Box::new([[[v; N]; L]; K]);
            assert_eq!(
                key_weight(&a, &top_split, &weights),
                key_weight_reference(&a, &top, &eq_row, &eq_col)
            );
        }
    }

    #[test]
    fn parallel_dedup_matches_the_serial_map_on_full_keys() {
        let mut rng = SplitMix64::new(8);
        // Twelve keys; keys 1, 2, 3 share key 0's 32-byte prefix (rho), key 4 differs from
        // key 0 in its last byte only.
        let mut keys: Vec<Vec<u8>> = (0..12)
            .map(|_| {
                let mut k = vec![0_u8; 1312];
                rng.fill(&mut k);
                k
            })
            .collect();
        for i in 1..4 {
            let prefix = keys[0][..32].to_vec();
            keys[i][..32].copy_from_slice(&prefix);
        }
        keys[4] = keys[0].clone();
        keys[4][1311] ^= 1;
        let mut short = keys[5].clone();
        short.truncate(100);
        keys.push(short);
        let picks: Vec<usize> = (0..500).map(|_| rng.below(keys.len() as u64) as usize).collect();
        let inputs: Vec<(&[u8], &[u8], [u8; C_TILDE_BYTES])> = picks
            .iter()
            .map(|&k| (keys[k].as_slice(), &[][..], [0_u8; C_TILDE_BYTES]))
            .collect();
        let (distinct, record_keys) = distinct_keys(&inputs);
        assert_eq!((distinct.clone(), record_keys.clone()), distinct_keys_reference(&inputs));
        let mut seen: Vec<usize> = picks.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(distinct.len(), seen.len());
        for (i, &k) in record_keys.iter().enumerate() {
            assert_eq!(distinct[k].0, keys[picks[i]].as_slice());
        }
    }

    /// Two genuine keys sharing `rho` (the second built under the first one's matrix seed):
    /// they are two keys, every record's `T` uses its own `t1`, and swapping the keys of
    /// two records changes both `T` tables (the end-to-end accept/reject is in
    /// `agg-cli/tests/mode_a.rs`).
    #[test]
    fn keys_sharing_rho_stay_distinct() {
        let (victim_sk, victim) = gen::keypair(&gen::key_seed(40, 0));
        let rho: [u8; 32] = victim[..32].try_into().unwrap();
        let (other_sk, other) = gen::keypair_with_rho(&gen::key_seed(40, 1), &rho);
        assert_eq!(victim[..32], other[..32]);
        let mut records = gen::records(41, 3);
        for i in 0..6_u64 {
            let (sk, pk) = if i % 2 == 0 { (&victim_sk, &victim) } else { (&other_sk, &other) };
            let msg = gen::message(42, i);
            records.push(gen::Record {
                pk: pk.clone(),
                sig: gen::sign(sk, &msg, None),
                msg,
            });
        }
        let instance = PublicInstance::from_records(&records).unwrap();
        let inputs: Vec<(&[u8], &[u8], [u8; C_TILDE_BYTES])> = instance
            .records
            .iter()
            .zip(&records)
            .map(|(record, source)| (source.pk.as_slice(), source.msg.as_slice(), record.c_tilde))
            .collect();
        let stored = PublicData::from_instance(&instance);
        let mut cache = KeyCache::default();
        cache.insert(Arc::new(VerifierKey::from_bytes(&victim).unwrap()));
        for cache in [None, Some(&cache)] {
            let public = StreamedPublic::prepare(&inputs, cache).unwrap();
            assert_eq!(public.keys().len(), 5);
            let victim_t1 = pk_decode(&victim).unwrap().t1;
            let other_t1 = pk_decode(&other).unwrap().t1;
            for i in 3..9 {
                let record = &public.records()[i];
                let key = &public.keys()[record.key];
                let (own, foreign) = if i % 2 == 1 {
                    (&victim, &other_t1)
                } else {
                    (&other, &victim_t1)
                };
                assert_eq!(key.pk(), own.as_slice());
                assert_eq!(key.is_cached(), cache.is_some() && *own == victim);
                assert_eq!(*public.t(i), *stored.t(i));
                assert_ne!(*public.t(i), sparse_t(&record.challenge(), foreign));
                assert_eq!(record.mu, instance.records[i].mu);
            }
        }
    }

    /// Timings of the verifier's light precomputation on 8,192 synthetic records with
    /// distinct keys, all threads (direction only; `cargo test --release -p agg-piop --lib
    /// prepare_profile -- --ignored --nocapture` on CPUs 0,1,3-15).
    #[test]
    #[ignore = "microbenchmark"]
    fn prepare_profile() {
        use crate::binding::{public_records_binding, record_stream};
        let n = 8192;
        let mut rng = SplitMix64::new(10);
        let keys: Vec<Vec<u8>> = (0..n)
            .map(|_| {
                let mut k = vec![0_u8; agg_witness::params::PK_BYTES];
                rng.fill(&mut k);
                k
            })
            .collect();
        let msgs: Vec<Vec<u8>> = (0..n).map(|i| vec![i as u8; 80]).collect();
        let inputs: Vec<(&[u8], &[u8], [u8; C_TILDE_BYTES])> = (0..n)
            .map(|i| {
                let mut c = [0_u8; C_TILDE_BYTES];
                rng.fill(&mut c);
                (keys[i].as_slice(), msgs[i].as_slice(), c)
            })
            .collect();
        let mut cache = KeyCache::default();
        let expanded: Vec<_> = keys
            .par_iter()
            .map(|k| Arc::new(VerifierKey::from_bytes(k).unwrap()))
            .collect();
        for key in expanded {
            cache.insert(key);
        }
        fn time<T>(label: &str, mut f: impl FnMut() -> T) {
            let start = Instant::now();
            black_box(f());
            eprintln!("{label}: first run {:.2} ms", start.elapsed().as_secs_f64() * 1e3);
            let mut runs: Vec<f64> = (0..5)
                .map(|_| {
                    let start = Instant::now();
                    black_box(f());
                    start.elapsed().as_secs_f64() * 1e3
                })
                .collect();
            runs.sort_by(f64::total_cmp);
            eprintln!("{label}: median {:.2} ms (min {:.2})", runs[2], runs[0]);
        }
        time("prepare cached (cold)", || StreamedPublic::prepare(&inputs, Some(&cache)).unwrap());
        time("distinct_keys (parallel)", || distinct_keys(&inputs));
        time("distinct_keys_reference (serial map)", || distinct_keys_reference(&inputs));
        time("pk_decode x N (parallel)", || {
            inputs
                .par_iter()
                .map(|(pk, _, _)| pk_decode(pk).unwrap().t1[0][0])
                .sum::<u16>()
        });
        time("cache.get x N (parallel)", || {
            inputs
                .par_iter()
                .filter(|(pk, _, _)| cache.get(pk).is_some())
                .count()
        });
        time("mu x N (parallel)", || {
            inputs
                .par_iter()
                .map(|(_, m, _)| mu(&[0; 64], m, &[])[0])
                .collect::<Vec<_>>()
        });
        time("sample_in_ball x N (parallel)", || {
            inputs
                .par_iter()
                .map(|(_, _, c)| sample_in_ball(c).positions.len())
                .sum::<usize>()
        });
        time("prepare uncached", || StreamedPublic::prepare(&inputs, None).unwrap());
        time("prepare cached", || StreamedPublic::prepare(&inputs, Some(&cache)).unwrap());
        time("record_stream (parallel)", || record_stream(&inputs));
        time("public_records_binding", || public_records_binding(&inputs));
    }

    /// The pre-engine `prepare` and `pass` (reference for the end-to-end A/B).
    fn baseline_prepare(inputs: &[(&[u8], &[u8], [u8; C_TILDE_BYTES])]) -> Vec<(Vec<u8>, [u8; TR_BYTES], [[u16; N]; K], [u8; 32])> {
        use agg_witness::native::tr as base_tr;
        let (distinct, record_keys) = distinct_keys(inputs);
        let keys: Vec<_> = distinct
            .into_par_iter()
            .map(|(pk, _)| {
                let d = pk_decode(pk).unwrap();
                (pk.to_vec(), base_tr(pk), d.t1, d.rho)
            })
            .collect();
        let sink: Vec<u8> = inputs
            .par_iter()
            .zip(&record_keys)
            .map(|(&(_, msg, c), &key)| {
                mu(&keys[key].1, msg, &[])[0] ^ sample_in_ball(&c).positions[0]
            })
            .collect();
        black_box(sink);
        keys
    }

    fn baseline_pass(
        s: &StreamedPublic,
        r2: &[E],
        r3: &[E],
    ) -> (E, E) {
        use crate::public_data::{inverse_ntt_matrix, sparse_ct as base_sparse_ct};
        use agg_witness::expand::expand_a;
        let (coef, pos) = (D_COEF as usize, D_POS as usize);
        let eq_pos2 = SplitWeights::new(&eq_table(&r2[..pos]));
        let eq_sig2 = eq_table(&r2[pos..]);
        let eq_sig3 = eq_table(&r3[pos..]);
        let t = s
            .records
            .par_iter()
            .enumerate()
            .map(|(i, r)| {
                let ct = base_sparse_ct(&r.challenge(), s.keys[r.key].t1());
                eq_sig2[i] * eq_pos2.dot::<16>(ct.as_flattened())
            })
            .reduce(E::zero, |a, b| a + b);
        let lambda = SplitWeights::new(&eq_correlation(&r2[..coef], &r3[..coef]));
        let eq_row = eq_table(&r2[coef..pos]);
        let eq_col = eq_table(&r3[coef..pos]);
        let ew: [[E; L]; K] =
            std::array::from_fn(|r| std::array::from_fn(|c| eq_row[r] * eq_col[c]));
        let w = s
            .keys
            .par_iter()
            .map(|key| {
                let sw = key.records.iter().fold(E::zero(), |x, &i| x + eq_sig2[i] * eq_sig3[i]);
                let KeySource::Fresh(f) = &key.source else { unreachable!() };
                let a = inverse_ntt_matrix(&expand_a(&f.rho).0);
                sw * key_weight(&a, &lambda, &ew)
            })
            .reduce(E::zero, |a, b| a + b);
        (t, w)
    }

    /// The `T(r2)` part of the pass alone, through the pre-engine kernel (`None`) or an engine.
    fn t_part(s: &StreamedPublic, eq: &SplitWeights, eq_sig2: &[E], e: Option<&agg_mldsa_simd::Engine>) -> E {
        s.records
            .par_iter()
            .enumerate()
            .map(|(i, r)| {
                let ct = match e {
                    None => crate::public_data::sparse_ct(&r.challenge(), s.keys[r.key].t1()),
                    Some(e) => {
                        let mut ct = [[0_i32; N]; K];
                        e.sparse_ct(&r.packed, s.keys[r.key].t1(), 0, &mut ct);
                        ct
                    }
                };
                eq_sig2[i] * eq.dot::<16>(ct.as_flattened())
            })
            .reduce(E::zero, |a, b| a + b)
    }

    /// End-to-end single-thread A/B of `prepare` and `pass` against the pre-engine code
    /// (direction only; `RAYON_NUM_THREADS=1 cargo test --release -p agg-piop --lib
    /// engine_e2e_microbench -- --ignored --nocapture` on one pinned CPU).
    #[test]
    #[ignore = "microbenchmark"]
    fn engine_e2e_microbench() {
        fn best(mut f: impl FnMut()) -> f64 {
            f();
            (0..5)
                .map(|_| {
                    let start = Instant::now();
                    f();
                    start.elapsed().as_secs_f64() * 1e3
                })
                .fold(f64::MAX, f64::min)
        }
        let n = 4096;
        let mut rng = SplitMix64::new(10);
        let keys: Vec<Vec<u8>> = (0..n)
            .map(|_| {
                let mut k = vec![0_u8; PK_BYTES];
                rng.fill(&mut k);
                k
            })
            .collect();
        let msgs: Vec<Vec<u8>> = (0..n).map(|i| vec![i as u8; 80]).collect();
        let inputs: Vec<(&[u8], &[u8], [u8; C_TILDE_BYTES])> = (0..n)
            .map(|i| {
                let mut c = [0_u8; C_TILDE_BYTES];
                rng.fill(&mut c);
                (keys[i].as_slice(), msgs[i].as_slice(), c)
            })
            .collect();
        let d_main = Geometry::new(n).unwrap().d_main() as usize;
        let (r2, r3) = (point(&mut rng, d_main), point(&mut rng, d_main));
        let old_prep = best(|| {
            black_box(baseline_prepare(&inputs));
        });
        let s = StreamedPublic::prepare(&inputs, None).unwrap();
        let old_pass = best(|| {
            black_box(baseline_pass(&s, &r2, &r3));
        });
        let eq = SplitWeights::new(&eq_table(&r2[..D_POS as usize]));
        let eq_sig2 = eq_table(&r2[D_POS as usize..]);
        let old_t = best(|| {
            black_box(t_part(&s, &eq, &eq_sig2, None));
        });
        eprintln!("n = {n}: baseline prepare {old_prep:.2} ms, pass {old_pass:.2} ms, T part {old_t:.2} ms");
        for backend in [
            agg_mldsa_simd::Backend::Scalar,
            agg_mldsa_simd::Backend::Avx2,
            agg_mldsa_simd::Backend::Avx512,
        ] {
            let Some(e) = agg_mldsa_simd::Engine::new(backend) else {
                continue;
            };
            let prep = best(|| {
                black_box(StreamedPublic::prepare_with(&e, &inputs, None).unwrap());
            });
            let pass = best(|| {
                black_box(s.pass_with(&e, &r2, &r3, None));
            });
            let t = best(|| {
                black_box(t_part(&s, &eq, &eq_sig2, Some(&e)));
            });
            eprintln!("{backend:?}: prepare {prep:.2} ms, pass {pass:.2} ms, T part {t:.2} ms");
        }
    }

    /// Single-thread A/B of the engine paths against the pre-engine computations, per
    /// backend (direction only; `cargo test --release -p agg-piop --lib engine_ab_microbench
    /// -- --ignored --nocapture` on one pinned CPU).
    #[test]
    #[ignore = "microbenchmark"]
    fn engine_ab_microbench() {
        use crate::public_data::{inverse_ntt_matrix, sparse_ct as base_sparse_ct};
        use agg_witness::expand::expand_a;
        use agg_witness::native::tr as base_tr;
        fn best(mut f: impl FnMut()) -> f64 {
            f();
            (0..7)
                .map(|_| {
                    let start = Instant::now();
                    f();
                    start.elapsed().as_secs_f64() * 1e3
                })
                .fold(f64::MAX, f64::min)
        }
        let n = 256;
        let mut rng = SplitMix64::new(77);
        let pks: Vec<Vec<u8>> = (0..n)
            .map(|_| {
                let mut k = vec![0_u8; PK_BYTES];
                rng.fill(&mut k);
                k
            })
            .collect();
        let msgs: Vec<Vec<u8>> = (0..n).map(|i| vec![i as u8; 80]).collect();
        let cts: Vec<[u8; C_TILDE_BYTES]> = (0..n)
            .map(|_| {
                let mut c = [0_u8; C_TILDE_BYTES];
                rng.fill(&mut c);
                c
            })
            .collect();
        let challenges: Vec<Challenge> = cts.iter().map(|c| sample_in_ball(c)).collect();
        let t1s: Vec<[[u16; N]; K]> = pks.iter().map(|k| pk_decode(k).unwrap().t1).collect();
        let rhos: Vec<[u8; 32]> = pks.iter().map(|k| k[..32].try_into().unwrap()).collect();
        let trs: Vec<[u8; TR_BYTES]> = pks.iter().map(|k| base_tr(k)).collect();

        let tr_old = best(|| {
            for pk in &pks {
                black_box(base_tr(pk));
            }
        });
        let mu_old = best(|| {
            for (t, m) in trs.iter().zip(&msgs) {
                black_box(mu(t, m, &[]));
            }
        });
        let ball_old = best(|| {
            for c in &cts {
                black_box(sample_in_ball(c));
            }
        });
        let ct_old = best(|| {
            for (c, t) in challenges.iter().zip(&t1s) {
                black_box(base_sparse_ct(c, t));
            }
        });
        let a_old = best(|| {
            for rho in &rhos {
                black_box(inverse_ntt_matrix(&expand_a(rho).0));
            }
        });
        eprintln!(
            "per {n} items, ms: base tr {tr_old:.2}, mu {mu_old:.2}, ball {ball_old:.2}, \
             sparse_ct {ct_old:.2}, expand+intt {a_old:.2}"
        );
        for backend in [
            agg_mldsa_simd::Backend::Scalar,
            agg_mldsa_simd::Backend::Avx2,
            agg_mldsa_simd::Backend::Avx512,
        ] {
            let Some(e) = agg_mldsa_simd::Engine::new(backend) else {
                continue;
            };
            let tr_new = best(|| {
                let mut out = [0_u8; 8 * TR_BYTES];
                for tile in pks.chunks(8) {
                    let ins: Vec<[&[u8]; 4]> = tile.iter().map(|p| [&p[..], &[][..], &[][..], &[][..]]).collect();
                    e.shake256_many(&ins, TR_BYTES, &mut out[..tile.len() * TR_BYTES]);
                    black_box(&out);
                }
            });
            let mu_new = best(|| {
                let mut out = [0_u8; 64 * TR_BYTES];
                for (tt, mt) in trs.chunks(64).zip(msgs.chunks(64)) {
                    let ins: Vec<[&[u8]; 4]> = tt
                        .iter()
                        .zip(mt)
                        .map(|(t, m)| [&t[..], &[0u8, 0u8][..], &[][..], &m[..]])
                        .collect();
                    e.shake256_many(&ins, TR_BYTES, &mut out[..tt.len() * TR_BYTES]);
                    black_box(&out);
                }
            });
            let ball_new = best(|| {
                let mut out = [Challenge39::zero(); 64];
                for ct in cts.chunks(64) {
                    e.sample_in_ball_many(ct, &mut out[..ct.len()]);
                    for o in &out[..ct.len()] {
                        black_box(challenge_from_c39(o));
                    }
                }
            });
            let ct_new = best(|| {
                for (c, t) in challenges.iter().zip(&t1s) {
                    let mut ct = [[0_i32; N]; K];
                    e.sparse_ct(&challenge_to_c39(c), t, 0, &mut ct);
                    black_box(&ct);
                }
            });
            let a_new = best(|| {
                for tile in rhos.chunks(8) {
                    let mut ex = vec![ExpandedA::zero(); tile.len()];
                    e.expand_a_many(tile, &mut ex);
                    for x in &mut ex {
                        for r in 0..K {
                            for s in 0..L {
                                e.intt(Prime::Q, &mut x.a_hat[r][s], Scale::Plain, Form::Centered);
                            }
                        }
                    }
                    black_box(&ex);
                }
            });
            eprintln!(
                "{backend:?}: tr {tr_new:.2}, mu {mu_new:.2}, ball {ball_new:.2}, \
                 sparse_ct {ct_new:.2}, expand+intt {a_new:.2}"
            );
        }
    }

    /// Single-thread timings of the three kernels against the references (direction only;
    /// `cargo test --release -p agg-piop --lib kernel_microbench -- --ignored --nocapture`
    /// on CPUs 0,1,3-15).
    #[test]
    #[ignore = "microbenchmark"]
    fn kernel_microbench() {
        fn time<T>(label: &str, reps: usize, mut f: impl FnMut() -> T) -> f64 {
            for _ in 0..reps / 10 + 1 {
                black_box(f());
            }
            let start = Instant::now();
            for _ in 0..reps {
                black_box(f());
            }
            let ns = start.elapsed().as_nanos() as f64 / reps as f64;
            eprintln!("{label}: {ns:.0} ns per call");
            ns
        }
        let mut rng = SplitMix64::new(9);
        let cases: Vec<(Challenge, [[u16; N]; K])> =
            (0..64).map(|_| (random_challenge(&mut rng), random_t1(&mut rng))).collect();
        let mut i = 0;
        let old = time("sparse_t reference", 20_000, || {
            i = (i + 1) % cases.len();
            sparse_t_reference(&cases[i].0, &cases[i].1)
        });
        let new = time("sparse_t", 20_000, || {
            i = (i + 1) % cases.len();
            sparse_t(&cases[i].0, &cases[i].1)
        });
        eprintln!("sparse_t speedup {:.2}x", old / new);

        let eq = point(&mut rng, 1024);
        let split = SplitWeights::new(&eq);
        let tables: Vec<_> = cases.iter().map(|(c, t1)| sparse_t(c, t1)).collect();
        let products: Vec<_> = cases.iter().map(|(c, t1)| sparse_ct(c, t1)).collect();
        let old = time("T inner product reference (1,024 terms)", 20_000, || {
            i = (i + 1) % tables.len();
            dot_reference(tables[i].as_flattened(), &eq)
        });
        let wide = time("T inner product split i128", 20_000, || {
            i = (i + 1) % tables.len();
            split.dot_wide(tables[i].as_flattened())
        });
        eprintln!("T inner product i128 speedup {:.2}x", old / wide);
        let new = time("T inner product u32 lanes (c t1, 16 bits)", 20_000, || {
            i = (i + 1) % products.len();
            split.dot::<16>(products[i].as_flattened()) * E::from_u64(1 << D)
        });
        eprintln!("T inner product speedup {:.2}x", old / new);
        let old = time("sparse_t + inner, reference", 20_000, || {
            i = (i + 1) % cases.len();
            dot_reference(sparse_t_reference(&cases[i].0, &cases[i].1).as_flattened(), &eq)
        });
        let new = time("sparse_t + inner, new", 20_000, || {
            i = (i + 1) % cases.len();
            let ct = sparse_ct(&cases[i].0, &cases[i].1);
            split.dot::<16>(ct.as_flattened()) * E::from_u64(1 << D)
        });
        eprintln!("per-record T work speedup {:.2}x", old / new);

        // 512 matrices (8 MiB of A, beyond the L2) as in a cached batch.
        let matrices: Vec<Box<MatrixCoeff>> = (0..512).map(|_| random_a(&mut rng)).collect();
        let lambda = point(&mut rng, N);
        let (eq_row, eq_col) = (point(&mut rng, K), point(&mut rng, L));
        let weights: [[E; L]; K] =
            std::array::from_fn(|rho| std::array::from_fn(|s| eq_row[rho] * eq_col[s]));
        let split = SplitWeights::new(&lambda);
        let old = time("key_weight reference", 5_000, || {
            i = (i + 1) % matrices.len();
            key_weight_reference(&matrices[i], &lambda, &eq_row, &eq_col)
        });
        let wide = time("key_weight split i128", 5_000, || {
            i = (i + 1) % matrices.len();
            let mut sum = E::zero();
            for (row, w) in matrices[i].iter().zip(&weights) {
                for (poly, &w) in row.iter().zip(w) {
                    sum += w * split.dot_wide(poly);
                }
            }
            sum
        });
        eprintln!("key_weight i128 speedup {:.2}x", old / wide);
        let new = time("key_weight u32 lanes (22 bits)", 5_000, || {
            i = (i + 1) % matrices.len();
            key_weight(&matrices[i], &split, &weights)
        });
        eprintln!("key_weight speedup {:.2}x", old / new);

    }

    #[test]
    fn streamed_pass_matches_the_stored_evaluations_with_and_without_cache() {
        let mut records = gen::records(31, 5);
        records.extend(gen::records_shared_keys(32, 6, 2));
        let instance = PublicInstance::from_records(&records).unwrap();
        let inputs: Vec<(&[u8], &[u8], [u8; C_TILDE_BYTES])> = instance
            .records
            .iter()
            .zip(&records)
            .map(|(record, source)| (source.pk.as_slice(), source.msg.as_slice(), record.c_tilde))
            .collect();
        let stored = PublicData::from_instance(&instance);
        let d_main = instance.geom.d_main() as usize;
        let mut rng = SplitMix64::new(77);
        let (r2, r3) = (point(&mut rng, d_main), point(&mut rng, d_main));
        let expected_t = stored.t_evaluation(&r2);
        let expected_w = ring_weight_at(&stored, &r2, &r3);

        let uncached = StreamedPublic::prepare(&inputs, None).unwrap();
        assert_eq!(uncached.keys().len(), 7);
        for (i, record) in uncached.records().iter().enumerate() {
            assert_eq!(record.mu, instance.records[i].mu);
            assert_eq!(*uncached.t(i), *stored.t(i));
        }
        let pass = uncached.pass(&r2, &r3, None);
        assert_eq!(
            (pass.t_at_r2, pass.ring_weight, pass.expanded),
            (expected_t, expected_w, 7)
        );

        let mut cache = KeyCache::default();
        let first = StreamedPublic::prepare(&inputs, Some(&cache)).unwrap();
        assert!(first.keys().iter().all(|key| !key.is_cached()));
        let pass = first.pass(&r2, &r3, Some(&mut cache));
        assert_eq!((pass.t_at_r2, pass.ring_weight), (expected_t, expected_w));
        assert_eq!(cache.len(), 7);
        let second = StreamedPublic::prepare(&inputs, Some(&cache)).unwrap();
        assert!(second.keys().iter().all(StreamKey::is_cached));
        let pass = second.pass(&r2, &r3, Some(&mut cache));
        assert_eq!(
            (pass.t_at_r2, pass.ring_weight, pass.expanded),
            (expected_t, expected_w, 0)
        );
        for (i, record) in second.records().iter().enumerate() {
            assert_eq!(record.mu, instance.records[i].mu);
        }

        let mut bad = inputs.clone();
        bad[4].0 = &records[4].pk[..200];
        assert_eq!(
            StreamedPublic::prepare(&bad, None).unwrap_err(),
            InstanceError::PublicKey(4)
        );
    }

    #[test]
    fn streamed_pass_matches_reference_across_all_available_backends() {
        let backends = [
            agg_mldsa_simd::Backend::Scalar,
            agg_mldsa_simd::Backend::Avx2,
            agg_mldsa_simd::Backend::Avx512,
        ];

        for backend in backends {
            let Some(engine) = agg_mldsa_simd::Engine::new(backend) else {
                continue;
            };

            // Test all-distinct keys across various record counts (1, 7, 8, 9, 33).
            for &n in &[1, 7, 8, 9, 33] {
                let records = gen::records(100 + n as u64, n);
                check_backend_streamed_pass(&engine, &records);
            }

            // Test keys sharing rho.
            {
                let (victim_sk, victim) = gen::keypair(&gen::key_seed(200, 0));
                let rho: [u8; 32] = victim[..32].try_into().unwrap();
                let (other_sk, other) = gen::keypair_with_rho(&gen::key_seed(200, 1), &rho);
                assert_eq!(victim[..32], other[..32]);

                let mut records = gen::records(201, 3);
                for i in 0..6_u64 {
                    let (sk, pk) = if i % 2 == 0 { (&victim_sk, &victim) } else { (&other_sk, &other) };
                    let msg = gen::message(202, i);
                    records.push(gen::Record {
                        pk: pk.clone(),
                        sig: gen::sign(sk, &msg, None),
                        msg,
                    });
                }
                check_backend_streamed_pass(&engine, &records);
            }
        }
    }

    fn check_backend_streamed_pass(engine: &agg_mldsa_simd::Engine, records: &[gen::Record]) {
        let instance = PublicInstance::from_records(records).unwrap();
        let inputs: Vec<(&[u8], &[u8], [u8; C_TILDE_BYTES])> = instance
            .records
            .iter()
            .zip(records)
            .map(|(record, source)| (source.pk.as_slice(), source.msg.as_slice(), record.c_tilde))
            .collect();
        let stored = PublicData::from_instance(&instance);
        let d_main = instance.geom.d_main() as usize;
        let mut rng = SplitMix64::new(300 + records.len() as u64);
        let (r2, r3) = (point(&mut rng, d_main), point(&mut rng, d_main));
        let expected_t = stored.t_evaluation(&r2);
        let expected_w = ring_weight_at(&stored, &r2, &r3);

        // Without cache:
        let uncached = StreamedPublic::prepare_with(engine, &inputs, None).unwrap();
        for (i, record) in uncached.records().iter().enumerate() {
            assert_eq!(record.mu, instance.records[i].mu);
            assert_eq!(*uncached.t(i), *stored.t(i));
        }
        let pass = uncached.pass_with(engine, &r2, &r3, None);
        assert_eq!(
            pass.t_at_r2, expected_t,
            "t_at_r2 mismatch without cache on backend {:?}",
            engine.backend()
        );
        assert_eq!(
            pass.ring_weight, expected_w,
            "ring_weight mismatch without cache on backend {:?}",
            engine.backend()
        );
        assert_eq!(pass.expanded, uncached.keys().len());

        // With cache (first call populates, second call reuses):
        let mut cache = KeyCache::default();
        let first = StreamedPublic::prepare_with(engine, &inputs, Some(&cache)).unwrap();
        assert!(first.keys().iter().all(|key| !key.is_cached()));
        let pass_pop = first.pass_with(engine, &r2, &r3, Some(&mut cache));
        assert_eq!(pass_pop.t_at_r2, expected_t);
        assert_eq!(pass_pop.ring_weight, expected_w);
        assert_eq!(cache.len(), uncached.keys().len());

        let second = StreamedPublic::prepare_with(engine, &inputs, Some(&cache)).unwrap();
        assert!(second.keys().iter().all(StreamKey::is_cached));
        let pass_cached = second.pass_with(engine, &r2, &r3, Some(&mut cache));
        assert_eq!(
            pass_cached.t_at_r2, expected_t,
            "t_at_r2 mismatch with cache on backend {:?}",
            engine.backend()
        );
        assert_eq!(
            pass_cached.ring_weight, expected_w,
            "ring_weight mismatch with cache on backend {:?}",
            engine.backend()
        );
        assert_eq!(pass_cached.expanded, 0);
        for (i, record) in second.records().iter().enumerate() {
            assert_eq!(record.mu, instance.records[i].mu);
        }
    }
}
