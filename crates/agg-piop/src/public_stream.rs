//! The verifier's streamed public data (spec Section 11, steps 2 and 6).
//!
//! Step 2 keeps per live record only what the transcript and the Keccak side need before
//! the sum-checks: `pkDecode` (`rho`, `t1`), `tr = H(pk)` once per distinct key, `mu` and
//! `c = SampleInBall(c~)`. Nothing of size `k l 256` is stored: `A_i` and `T_i` are formed
//! inside the one pass of step 6, after the stage-3 rounds have fixed `r^(3)`
//! ([`StreamedPublic::pass`]):
//!
//! - `T(r^(2)) = sum_i eq(r2_sig, i) sum_{rho, j} eq(r2_pos, (j, rho)) T_i[rho][j]`, with
//!   `T_i = 2^d (c_i * t1)` recomputed from the `tau` sparse entries of `c_i`
//!   ([`sparse_t`], `k tau 256` integer additions) and dropped after use;
//! - `w_A(r^(3))` of (7.10): per distinct key the expansion `A = ctr_q(NTT^{-1}(ExpandA(rho)))`
//!   (or the cached one), its `k l` inner products with the correlation vector `Lambda`
//!   ([`eq_correlation`]), weighted by `sum_{i of the key} eq(r2_sig, i) eq(r3_sig, i)`.
//!
//! Memory is `O(N)` small records plus one `A` per worker thread, instead of 11.8 KB per key
//! and 4 KB per record. New expansions are added to the [`KeyCache`] when one is given.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use aerie_falcon::protocol::algebra::eq_table;
use agg_witness::expand::{expand_a, MatrixCoeff};
use agg_witness::instance::InstanceError;
use agg_witness::native::{mu, pk_decode, sample_in_ball, tr, Challenge};
use agg_witness::padding::Geometry;
use agg_witness::params::{C_TILDE_BYTES, D_COEF, D_POS, K, L, N, TR_BYTES};
use jolt_field::{Ring, Zero};
use rayon::prelude::*;

use crate::public_data::{eq_correlation, inverse_ntt_matrix, sparse_t, KeyCache, VerifierKey};
use crate::OpeningField as E;

/// One distinct key of the batch.
#[derive(Clone, Debug)]
pub struct StreamKey {
    /// Encoded public key (the cache key).
    pub pk: Vec<u8>,
    rho: [u8; 32],
    t1: [[u16; N]; K],
    tr: [u8; TR_BYTES],
    cached: Option<Arc<VerifierKey>>,
    /// Live records signed under this key, increasing.
    pub records: Vec<usize>,
}

impl StreamKey {
    /// True when the expansion came from the cache.
    #[must_use]
    pub fn is_cached(&self) -> bool {
        self.cached.is_some()
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
    challenge: Challenge,
}

/// What the pass of step 6 returns.
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

impl StreamedPublic {
    /// Step 2 from `(pk, M, c~)` triples (pure mode, empty context): `pkDecode`, `tr` per
    /// distinct key (taken from the cache when present), `mu` and `SampleInBall` per record.
    pub fn prepare(
        inputs: &[(&[u8], &[u8], [u8; C_TILDE_BYTES])],
        cache: Option<&KeyCache>,
    ) -> Result<Self, InstanceError> {
        let geom = Geometry::new(inputs.len()).map_err(InstanceError::Geometry)?;
        let mut index: HashMap<&[u8], usize> = HashMap::new();
        let mut distinct: Vec<(&[u8], Vec<usize>)> = Vec::new();
        let record_keys: Vec<usize> = inputs
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
        let keys = distinct
            .into_par_iter()
            .map(|(pk, records)| {
                let decoded = pk_decode(pk).ok_or(InstanceError::PublicKey(records[0]))?;
                let cached = cache.and_then(|c| c.get(pk));
                let tr = cached.as_ref().map_or_else(|| tr(pk), |key| key.tr);
                Ok(StreamKey {
                    pk: pk.to_vec(),
                    rho: decoded.rho,
                    t1: decoded.t1,
                    tr,
                    cached,
                    records,
                })
            })
            .collect::<Result<Vec<_>, InstanceError>>()?;
        let records = inputs
            .par_iter()
            .zip(&record_keys)
            .map(|(&(_, msg, c_tilde), &key)| StreamRecord {
                key,
                mu: mu(&keys[key].tr, msg, &[]),
                c_tilde,
                challenge: sample_in_ball(&c_tilde),
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
        Box::new(sparse_t(&record.challenge, &self.keys[record.key].t1))
    }

    /// Step 6: `T(r^(2))` and `w_A(r^(3))` in one streamed pass (module docs). New
    /// expansions are added to `cache` when one is given.
    pub fn pass(&self, r2: &[E], r3: &[E], cache: Option<&mut KeyCache>) -> PublicPass {
        let d_main = self.geom.d_main() as usize;
        assert_eq!(r2.len(), d_main, "a main-domain point");
        assert_eq!(r3.len(), d_main, "a main-domain point");
        let (coef, pos) = (D_COEF as usize, D_POS as usize);
        let eq_pos2 = eq_table(&r2[..pos]);
        let eq_sig2 = eq_table(&r2[pos..]);
        let eq_sig3 = eq_table(&r3[pos..]);

        let start = Instant::now();
        let t_at_r2 = self
            .records
            .par_iter()
            .enumerate()
            .map(|(i, record)| {
                let t = sparse_t(&record.challenge, &self.keys[record.key].t1);
                let inner = t
                    .iter()
                    .flatten()
                    .zip(&eq_pos2)
                    .fold(E::zero(), |sum, (&value, &weight)| {
                        sum + weight.mul_i64(i64::from(value))
                    });
                eq_sig2[i] * inner
            })
            .reduce(E::zero, |a, b| a + b);
        let t_time = start.elapsed();

        let start = Instant::now();
        let lambda = eq_correlation(&r2[..coef], &r3[..coef]);
        let eq_row = eq_table(&r2[coef..pos]);
        let eq_col = eq_table(&r3[coef..pos]);
        let keep = cache.is_some();
        let per_key: Vec<(E, Option<Arc<VerifierKey>>)> = self
            .keys
            .par_iter()
            .map(|key| {
                let signature_weight = key
                    .records
                    .iter()
                    .fold(E::zero(), |sum, &i| sum + eq_sig2[i] * eq_sig3[i]);
                let (weight, fresh) = match &key.cached {
                    Some(cached) => (key_weight(&cached.a, &lambda, &eq_row, &eq_col), None),
                    None => {
                        let a = inverse_ntt_matrix(&expand_a(&key.rho).0);
                        let weight = key_weight(&a, &lambda, &eq_row, &eq_col);
                        let fresh = keep.then(|| {
                            Arc::new(VerifierKey {
                                pk: key.pk.clone(),
                                a,
                                t1: key.t1,
                                tr: key.tr,
                            })
                        });
                        (weight, fresh)
                    }
                };
                (signature_weight * weight, fresh)
            })
            .collect();
        let ring_weight = per_key.iter().fold(E::zero(), |sum, (w, _)| sum + *w);
        let expanded = self.keys.iter().filter(|key| key.cached.is_none()).count();
        if let Some(cache) = cache {
            for (_, fresh) in per_key {
                if let Some(key) = fresh {
                    cache.insert(key);
                }
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
/// (7.10)); `A` entries fit 23 bits.
fn key_weight(a: &MatrixCoeff, lambda: &[E], eq_row: &[E], eq_col: &[E]) -> E {
    let mut sum = E::zero();
    for rho in 0..K {
        for s in 0..L {
            let dot = a[rho][s]
                .iter()
                .zip(lambda)
                .fold(E::zero(), |acc, (&entry, &l)| {
                    acc + l.mul_i64(i64::from(entry))
                });
            sum += eq_row[rho] * eq_col[s] * dot;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::public_data::PublicData;
    use crate::test_support::point;
    use crate::weights::ring_weight_at;
    use agg_witness::gen::{self, SplitMix64};
    use agg_witness::instance::PublicInstance;

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
}
