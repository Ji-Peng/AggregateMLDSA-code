//! Public data of the parameter sets whose vector index is a polynomial lane (ML-DSA-65 and
//! ML-DSA-87): the centered matrices `A_i`, the tables `T_i = 2^d (c_i * t1_i)` and the
//! three public quantities of the lane ring identity.
//!
//! With lanes there is no row coordinate on the main domain `(j, i)`, so stage 3 runs one
//! ring instance per row `rho` with its own batching coefficient `gamma_rho`:
//!
//! ```text
//! sum_rho gamma_rho T~_rho(r^(2)) = sum_s sum_x w_A^(s)(x) z^(s)(x)
//!                                 - sum_rho gamma_rho sum_x eq(r^(2), x) (2 gamma_2 w_1' + u + q v)^(rho)(x),
//! w_A^(s)(j', i) = eq(r^(2)_sig, i) K_{A~_{i,s}, r^(2)_coef}[j'],   A~_{i,s} = sum_rho gamma_rho A_i[rho, s].
//! ```
//!
//! The kernel `K_{a, r}` is linear in `a`, so each distinct key is transformed once per
//! column `s` and every record of that key scales the kernel by `eq(r^(2)_sig, i)`; the
//! verifier's `w_A^(s)(r^(3))` is `sum_i eq(r^(2)_sig, i) eq(r^(3)_sig, i) <A~_{i,s}, Lambda>`
//! with the correlation `Lambda` of `(r^(2)_coef, r^(3)_coef)`, one dot product per key,
//! row and column. Padded records carry zero weight and `T = 0`.
//!
//! The prover builds the data from its instance ([`LanePublicData::from_instance`], `T`
//! stored). The verifier builds it from the `(pk, M, c~)` triples
//! ([`LanePublicData::prepare`]): keys are expanded once per distinct `pk` (or taken from a
//! [`LaneKeyCache`]), and `T` is not stored: [`LanePublicData::t_evaluations`] recomputes
//! `T_i = 2^d (c_i * t1_i)` per record from the sparse challenge in its one pass.

use std::collections::HashMap;
use std::sync::Arc;

use aerie_falcon::protocol::algebra::eq_table;
use agg_witness::expand::ExpandedKey;
use agg_witness::instance::{InstanceError, PublicInstance};
use agg_witness::native::{mu, sample_in_ball_for, Challenge};
use agg_witness::padding::Geometry;
use agg_witness::params::{ParamSet, Set, N, TR_BYTES};
use jolt_field::{Ring, Zero};
use rayon::prelude::*;

use crate::public_data::eq_correlation;
use crate::OpeningField as E;

/// Variables of the coefficient index `j`.
const COEF_VARS: usize = 8;

/// The public part of one key that the lane identities read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaneKey {
    /// The parameter set.
    pub set: Set,
    /// The encoded public key.
    pub pk: Vec<u8>,
    /// `tr = H(pk, 64)`.
    pub tr: [u8; TR_BYTES],
    /// `t1`, one polynomial per row.
    pub t1: Vec<[u16; N]>,
    /// `A[rho][s]`, centered, flat `(rho, s, j)`.
    pub a: Vec<i32>,
}

impl LaneKey {
    /// The lane view of an expanded key.
    #[must_use]
    pub fn from_expanded<P: ParamSet>(key: &ExpandedKey<P>) -> Self {
        let mut a = Vec::with_capacity(P::K * P::L * N);
        for rho in 0..P::K {
            for s in 0..P::L {
                a.extend_from_slice(P::get_coeff(&key.a, rho, s));
            }
        }
        Self {
            set: P::SET,
            pk: key.pk.clone(),
            tr: key.tr,
            t1: (0..P::K).map(|rho| key.t1[rho]).collect(),
            a,
        }
    }
}

/// Expanded keys of one parameter set kept across verifications, keyed by the encoded key.
#[derive(Clone, Debug)]
pub struct LaneKeyCache {
    set: Set,
    keys: HashMap<Vec<u8>, Arc<LaneKey>>,
}

impl LaneKeyCache {
    /// An empty cache for `set`.
    #[must_use]
    pub fn new(set: Set) -> Self {
        Self {
            set,
            keys: HashMap::new(),
        }
    }

    /// The parameter set.
    #[must_use]
    pub const fn set(&self) -> Set {
        self.set
    }

    /// Cached keys.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// True when no key is cached.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// The expansion of `pk`, if cached.
    #[must_use]
    pub fn get(&self, pk: &[u8]) -> Option<Arc<LaneKey>> {
        self.keys.get(pk).cloned()
    }

    /// Adds an expansion of the cache's set (an existing entry for the same key is kept).
    ///
    /// # Panics
    /// For a key of another parameter set.
    pub fn insert(&mut self, key: Arc<LaneKey>) {
        assert_eq!(key.set, self.set, "a key of the cache's parameter set");
        self.keys.entry(key.pk.clone()).or_insert(key);
    }

    /// The cached expansions, in no particular order (to persist the cache).
    pub fn entries(&self) -> impl Iterator<Item = &Arc<LaneKey>> {
        self.keys.values()
    }
}

/// `T` of the live records: stored (the prover's instance), or recomputed from the sparse
/// challenges in each pass (the verifier).
#[derive(Clone, Debug)]
enum LaneT {
    /// `T_i[rho][j]` of every live record, flat `(i, rho, j)`.
    Stored(Vec<i32>),
    /// `c_i = SampleInBall(c~_i)` of every live record.
    Streamed(Vec<Challenge>),
}

/// The public data of a batch of a parameter set with polynomial lanes.
#[derive(Clone, Debug)]
pub struct LanePublicData {
    set: Set,
    k: usize,
    l: usize,
    padded: usize,
    /// Every distinct key.
    keys: Vec<Arc<LaneKey>>,
    /// The key of every live record.
    key_of: Vec<usize>,
    /// `mu_i` of every live record.
    mu: Vec<[u8; TR_BYTES]>,
    /// `T` of every live record.
    t: LaneT,
    /// Keys expanded while building (the rest came from the cache).
    expanded: usize,
}

/// `out[rho] = 2^d (c * t1[rho])` over `Z[X]/(X^256 + 1)` (`agg_witness::expand::public_t_into_for`).
fn t_rows_into(c: &Challenge, t1: &[[u16; N]], d: u32, out: &mut [[i32; N]]) {
    for (row, t1_row) in out.iter_mut().zip(t1) {
        row.fill(0);
        for &pos in &c.positions {
            let p = usize::from(pos);
            let sign = i32::from(c.coeffs[p]);
            let (low, high) = t1_row.split_at(N - p);
            for (slot, &x) in row[p..].iter_mut().zip(low) {
                *slot += sign * i32::from(x);
            }
            for (slot, &x) in row[..p].iter_mut().zip(high) {
                *slot -= sign * i32::from(x);
            }
        }
        for x in row.iter_mut() {
            *x <<= d;
        }
    }
}

impl LanePublicData {
    /// The verifier's data from `(pk, M, c~)` triples (pure mode, empty context): keys
    /// expanded once per distinct `pk`, or taken from `cache` (new expansions are added to
    /// it); `mu` and `SampleInBall(c~)` per record; `T` recomputed in each pass.
    ///
    /// # Errors
    /// On a batch size outside `1..=2^15` or a public key of the wrong length (the first
    /// such record).
    ///
    /// # Panics
    /// If `cache` holds another parameter set.
    pub fn prepare<P: ParamSet>(
        inputs: &[(&[u8], &[u8], P::CTilde)],
        mut cache: Option<&mut LaneKeyCache>,
    ) -> Result<Self, InstanceError>
    where
        P::CTilde: Sync,
    {
        let geom = Geometry::<P>::new_for(inputs.len()).map_err(InstanceError::Geometry)?;
        if let Some(cache) = cache.as_ref() {
            assert_eq!(cache.set, P::SET, "a cache of the batch's parameter set");
        }
        let mut distinct: Vec<&[u8]> = Vec::new();
        let mut first_record = Vec::new();
        let mut index: HashMap<&[u8], usize> = HashMap::new();
        let key_of: Vec<usize> = inputs
            .iter()
            .enumerate()
            .map(|(i, &(pk, _, _))| {
                *index.entry(pk).or_insert_with(|| {
                    distinct.push(pk);
                    first_record.push(i);
                    distinct.len() - 1
                })
            })
            .collect();
        let cached: Vec<Option<Arc<LaneKey>>> =
            distinct.iter().map(|&pk| cache.as_ref().and_then(|c| c.get(pk))).collect();
        let expanded = cached.iter().filter(|hit| hit.is_none()).count();
        let keys = distinct
            .par_iter()
            .zip(cached)
            .zip(&first_record)
            .map(|((pk, hit), &record)| match hit {
                Some(key) => Ok(key),
                None => ExpandedKey::<P>::from_bytes_for(pk)
                    .map(|key| Arc::new(LaneKey::from_expanded(&key)))
                    .ok_or(InstanceError::PublicKey(record)),
            })
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(cache) = cache.as_mut() {
            for key in &keys {
                cache.insert(Arc::clone(key));
            }
        }
        let (mu, challenges): (Vec<[u8; TR_BYTES]>, Vec<Challenge>) = inputs
            .par_iter()
            .zip(key_of.par_iter())
            .map(|(&(_, msg, ref c_tilde), &key)| {
                (mu(&keys[key].tr, msg, &[]), sample_in_ball_for::<P>(c_tilde.as_ref()))
            })
            .unzip();
        Ok(Self {
            set: P::SET,
            k: P::K,
            l: P::L,
            padded: geom.n_star,
            keys,
            key_of,
            mu,
            t: LaneT::Streamed(challenges),
            expanded,
        })
    }

    /// The data of the prover's instance (keys shared by `Arc` identity).
    #[must_use]
    pub fn from_instance<P: ParamSet>(instance: &PublicInstance<P>) -> Self {
        let (k, l) = (P::K, P::L);
        let mut seen: HashMap<*const ExpandedKey<P>, usize> = HashMap::new();
        let mut firsts = Vec::new();
        let key_of: Vec<usize> = instance
            .records
            .iter()
            .enumerate()
            .map(|(i, record)| {
                *seen.entry(Arc::as_ptr(&record.key)).or_insert_with(|| {
                    firsts.push(i);
                    firsts.len() - 1
                })
            })
            .collect();
        let keys = firsts
            .par_iter()
            .map(|&i| Arc::new(LaneKey::from_expanded(&instance.records[i].key)))
            .collect();
        let mut t = Vec::with_capacity(instance.records.len() * k * N);
        for record in &instance.records {
            for rho in 0..k {
                t.extend_from_slice(&record.t[rho]);
            }
        }
        Self {
            set: P::SET,
            k,
            l,
            padded: instance.geom.n_star,
            keys,
            key_of,
            mu: instance.records.iter().map(|r| r.mu).collect(),
            t: LaneT::Stored(t),
            expanded: 0,
        }
    }

    /// The parameter set.
    #[must_use]
    pub const fn set(&self) -> Set {
        self.set
    }

    /// `N*`.
    #[must_use]
    pub const fn padded_records(&self) -> usize {
        self.padded
    }

    /// `mu_i` of every live record.
    #[must_use]
    pub fn mu(&self) -> &[[u8; TR_BYTES]] {
        &self.mu
    }

    /// Keys expanded while building (the rest came from the cache; 0 for the prover's
    /// instance, whose keys were expanded with it).
    #[must_use]
    pub const fn expanded_keys(&self) -> usize {
        self.expanded
    }

    /// Distinct keys.
    #[must_use]
    pub fn keys(&self) -> &[Arc<LaneKey>] {
        &self.keys
    }

    /// Rows `k` of `A`.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.k
    }

    /// Columns `l` of `A`.
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.l
    }

    /// Live records `N`.
    #[must_use]
    pub fn records(&self) -> usize {
        self.key_of.len()
    }

    /// `A_i[rho][s]` of live record `i`.
    #[must_use]
    pub fn a(&self, i: usize, rho: usize, s: usize) -> &[i32] {
        let key = &self.keys[self.key_of[i]];
        &key.a[(rho * self.l + s) * N..(rho * self.l + s + 1) * N]
    }

    /// `f(T_i)` for live record `i`, `T_i` given as one row per `rho` (stored, or
    /// recomputed into a scratch buffer).
    fn with_t_rows<R>(&self, i: usize, f: impl FnOnce(&[[i32; N]]) -> R) -> R {
        match &self.t {
            LaneT::Stored(t) => f(t[i * self.k * N..(i + 1) * self.k * N].as_chunks::<N>().0),
            LaneT::Streamed(challenges) => {
                let mut rows = vec![[0_i32; N]; self.k];
                let d = agg_witness::dispatch_set!(self.set, |P| P::D);
                t_rows_into(&challenges[i], &self.keys[self.key_of[i]].t1, d, &mut rows);
                f(&rows)
            }
        }
    }

    /// `T_i` of live record `i`, one row per `rho` (stored or recomputed).
    #[must_use]
    pub fn t_rows(&self, i: usize) -> Vec<[i32; N]> {
        self.with_t_rows(i, <[[i32; N]]>::to_vec)
    }

    /// The point split `(r_coef, r_sig)` of a main-domain point `(j, i)`.
    fn split<'p>(&self, point: &'p [E]) -> (&'p [E], &'p [E]) {
        assert_eq!(
            point.len(),
            COEF_VARS + self.padded.trailing_zeros() as usize,
            "a main-domain point (j, i)"
        );
        point.split_at(COEF_VARS)
    }

    /// `T~_rho(r) = sum_{(j, i)} eq(r, (j, i)) T_i[rho][j]` for every row `rho`.
    #[must_use]
    pub fn t_evaluations(&self, point: &[E]) -> Vec<E> {
        let (coef, sig) = self.split(point);
        let (eq_coef, eq_sig) = (eq_table(coef), eq_table(sig));
        let k = self.k;
        (0..self.records())
            .into_par_iter()
            .map(|i| {
                self.with_t_rows(i, |rows| {
                    rows.iter()
                        .map(|row| {
                            let inner = row
                                .iter()
                                .zip(&eq_coef)
                                .fold(E::zero(), |acc, (&value, &w)| acc + w.mul_i64(i64::from(value)));
                            eq_sig[i] * inner
                        })
                        .collect::<Vec<E>>()
                })
            })
            .reduce(
                || vec![E::zero(); k],
                |mut acc, part| {
                    for (a, p) in acc.iter_mut().zip(part) {
                        *a += p;
                    }
                    acc
                },
            )
    }

    /// The matrix of live record `i`, flat in `(rho, s, j)` order.
    fn matrix(&self, i: usize) -> &[i32] {
        &self.keys[self.key_of[i]].a
    }

    /// The ring weights `w_A^(s)` on the main domain `(j', i)` (`N* 256` cells each, zero on
    /// padded records) for the point `r2 = r^(2)` and the row coefficients `gammas`: each
    /// record combines its rows with `eq(r^(2)_sig, i) gamma_rho`, transforms and writes each
    /// column once (`agg_simd::negacyclic::lane_ring_weight_rows`).
    ///
    /// # Panics
    /// If `gammas` does not have `k` entries or `r2` is not a main-domain point.
    #[must_use]
    pub fn ring_weight_tables(&self, r2: &[E], gammas: &[E]) -> Vec<Vec<E>> {
        assert_eq!(gammas.len(), self.k, "one coefficient per row");
        let (coef, sig) = self.split(r2);
        let eq_sig = eq_table(sig);
        let matrix = |i: usize| self.matrix(i);
        agg_simd::negacyclic::lane_ring_weight_rows(self.records(), self.l, &matrix, &eq_sig, gammas, coef, self.padded * N)
    }

    /// [`Self::ring_weight_tables`] as lane-major canonical planes for the plane-layout
    /// kernels: column `s` holds words `2 s N* 256 .. 2 (s + 1) N* 256`, its first coordinate
    /// plane then its second (`agg_simd::negacyclic::lane_ring_weight_planes`).
    ///
    /// # Panics
    /// As [`Self::ring_weight_tables`].
    #[must_use]
    pub fn ring_weight_planes(&self, r2: &[E], gammas: &[E]) -> agg_simd::buf::AlignedVec<u64> {
        assert_eq!(gammas.len(), self.k, "one coefficient per row");
        let (coef, sig) = self.split(r2);
        let eq_sig = eq_table(sig);
        let matrix = |i: usize| self.matrix(i);
        agg_simd::negacyclic::lane_ring_weight_planes(self.records(), self.l, &matrix, &eq_sig, gammas, coef, self.padded * N)
    }

    /// `w_A^(s)(r3)` for every column `s` (the verifier's terminal values): one dot product
    /// with `Lambda` per key, row and column, then one product per record and column.
    ///
    /// # Panics
    /// As [`Self::ring_weight_tables`].
    #[must_use]
    pub fn ring_weights_at(&self, r2: &[E], r3: &[E], gammas: &[E]) -> Vec<E> {
        assert_eq!(gammas.len(), self.k, "one coefficient per row");
        let (coef2, sig2) = self.split(r2);
        let (coef3, sig3) = self.split(r3);
        let lambda = eq_correlation(coef2, coef3);
        let (eq2, eq3) = (eq_table(sig2), eq_table(sig3));
        let (k, l) = (self.k, self.l);
        // The integer dots run on the SIMD kernel against `Lambda` held as planes.
        let lambda_planes = agg_simd::buf::KPlanes::from_fn(N, |j| lambda[j]);
        let per_key: Vec<Vec<E>> = self
            .keys
            .par_iter()
            .map(|key| {
                let mut ints = [0_i64; N];
                (0..l)
                    .map(|s| {
                        (0..k).fold(E::zero(), |acc, rho| {
                            let row = &key.a[(rho * l + s) * N..(rho * l + s + 1) * N];
                            for (x, &entry) in ints.iter_mut().zip(row) {
                                *x = i64::from(entry);
                            }
                            acc + gammas[rho] * agg_simd::dot::k_int_dot(lambda_planes.planes(), &ints)
                        })
                    })
                    .collect()
            })
            .collect();
        let mut out = vec![E::zero(); l];
        for (i, &key) in self.key_of.iter().enumerate() {
            let weight = eq2[i] * eq3[i];
            for (slot, &value) in out.iter_mut().zip(&per_key[key]) {
                *slot += weight * value;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::point;
    use aerie_falcon::protocol::algebra::evaluate_mle;
    use agg_witness::gen::{self, SplitMix64};
    use agg_witness::params::{Set65, Set87};

    /// The ring identity per row: `sum_s <w_A^(s), z^(s)> = sum_rho gamma_rho sum_x
    /// eq(r2, x) (A z)_rho(x)` for random `z`, and the verifier's values are the extensions of
    /// the tables; `T~_rho` is the extension of `T_rho`.
    fn ring_weights_pull_eq_back_per_row<P: ParamSet>() {
        let mut records = gen::records_for::<P>(81, 3);
        records.extend(gen::records_shared_keys_for::<P>(82, 2, 1));
        let instance = PublicInstance::<P>::from_records_for(&records).unwrap();
        let public = LanePublicData::from_instance(&instance);
        let (k, l, n_star) = (P::K, P::L, instance.geom.n_star);
        let vars = COEF_VARS + n_star.trailing_zeros() as usize;
        let mut rng = SplitMix64::new(83);
        let (r2, r3) = (point(&mut rng, vars), point(&mut rng, vars));
        let gammas = point(&mut rng, k);
        let tables = public.ring_weight_tables(&r2, &gammas);
        assert_eq!(tables.len(), l);
        // The planes hold the same values, column after column.
        let planes = public.ring_weight_planes(&r2, &gammas);
        let rows = n_star * N;
        assert_eq!(planes.len(), 2 * l * rows);
        for (s, table) in tables.iter().enumerate() {
            for (x, value) in table.iter().enumerate() {
                let (c0, c1) = (planes[2 * s * rows + x], planes[(2 * s + 1) * rows + x]);
                assert_eq!((value.c0().to_canonical_u64(), value.c1().to_canonical_u64()), (c0, c1), "w_A^({s})[{x}]");
            }
        }
        let z: Vec<Vec<i64>> = (0..l)
            .map(|_| (0..n_star * N).map(|_| rng.below(1 << 20) as i64 - (1 << 19)).collect())
            .collect();
        let lhs = tables.iter().zip(&z).fold(E::zero(), |acc, (w, z)| {
            acc + w.iter().zip(z).fold(E::zero(), |a, (&w, &z)| a + w.mul_i64(z))
        });
        let eq_r2 = eq_table(&r2);
        let mut rhs = E::zero();
        for i in 0..public.records() {
            for rho in 0..k {
                for j in 0..N {
                    // (A z)_rho[j] over the integers in Z[X]/(X^256 + 1).
                    let mut az = 0_i128;
                    for s in 0..l {
                        let a = public.a(i, rho, s);
                        for m in 0..N {
                            let (index, sign) = if m <= j { (j - m, 1) } else { (N + j - m, -1) };
                            az += sign * i128::from(a[m]) * i128::from(z[s][i * N + index]);
                        }
                    }
                    rhs += gammas[rho] * eq_r2[j + N * i] * E::from_i128(az);
                }
            }
        }
        assert_eq!(lhs, rhs);
        let at = public.ring_weights_at(&r2, &r3, &gammas);
        for s in 0..l {
            assert_eq!(at[s], evaluate_mle(&tables[s], &r3), "w_A^({s})(r3)");
        }
        // T~_rho is the extension of the instance's lane table of row rho (zero on padding).
        let t = public.t_evaluations(&r2);
        let lanes = instance.t_lane_tables();
        assert_eq!(lanes.len(), k);
        for (rho, table) in lanes.iter().enumerate() {
            assert_eq!(table.len(), n_star * N);
            assert!(table[public.records() * N..].iter().all(|&v| v == 0));
            assert_eq!(t[rho], evaluate_mle(&aerie_falcon::protocol::algebra::lift_table(table), &r2));
        }
    }

    /// The verifier's data from `(pk, M, c~)` equals the prover's: the same keys, `mu`, `T`
    /// rows and public evaluations, with and without a warm cache.
    fn prepared_data_equals_the_instance_data<P: ParamSet>()
    where
        P::CTilde: Sync,
    {
        let mut records = gen::records_for::<P>(91, 4);
        records.extend(gen::records_shared_keys_for::<P>(92, 5, 2));
        let instance = PublicInstance::<P>::from_records_for(&records).unwrap();
        let prover = LanePublicData::from_instance(&instance);
        let inputs: Vec<(&[u8], &[u8], P::CTilde)> = records
            .iter()
            .map(|r| (r.pk.as_slice(), r.msg.as_slice(), P::c_tilde_from_slice(&r.sig)))
            .collect();
        let mut cache = LaneKeyCache::new(P::SET);
        let cold = LanePublicData::prepare::<P>(&inputs, Some(&mut cache)).unwrap();
        assert_eq!((cold.expanded_keys(), cache.len()), (6, 6));
        let warm = LanePublicData::prepare::<P>(&inputs, Some(&mut cache)).unwrap();
        assert_eq!(warm.expanded_keys(), 0);
        let mut rng = SplitMix64::new(93);
        let vars = COEF_VARS + prover.padded_records().trailing_zeros() as usize;
        let (r2, r3, gammas) = (point(&mut rng, vars), point(&mut rng, vars), point(&mut rng, P::K));
        for verifier in [&cold, &warm] {
            assert_eq!(verifier.set(), P::SET);
            assert_eq!(verifier.mu(), prover.mu());
            assert_eq!(verifier.keys().len(), prover.keys().len());
            for (a, b) in verifier.keys().iter().zip(prover.keys()) {
                assert_eq!(**a, **b);
            }
            for i in 0..records.len() {
                assert_eq!(verifier.t_rows(i), prover.t_rows(i), "record {i}");
            }
            assert_eq!(verifier.t_evaluations(&r2), prover.t_evaluations(&r2));
            assert_eq!(verifier.ring_weights_at(&r2, &r3, &gammas), prover.ring_weights_at(&r2, &r3, &gammas));
        }
        // A key of the wrong length is refused at its first record.
        let mut bad = inputs.clone();
        let short = &records[3].pk[..10];
        bad[3].0 = short;
        assert_eq!(LanePublicData::prepare::<P>(&bad, None).err(), Some(InstanceError::PublicKey(3)));
    }

    #[test]
    fn ml_dsa_65_prepared_data_equals_the_instance_data() {
        prepared_data_equals_the_instance_data::<Set65>();
    }

    #[test]
    fn ml_dsa_87_prepared_data_equals_the_instance_data() {
        prepared_data_equals_the_instance_data::<Set87>();
    }

    #[test]
    fn ml_dsa_65_ring_weights_pull_eq_back_per_row() {
        ring_weights_pull_eq_back_per_row::<Set65>();
    }

    #[test]
    fn ml_dsa_87_ring_weights_pull_eq_back_per_row() {
        ring_weights_pull_eq_back_per_row::<Set87>();
    }
}
