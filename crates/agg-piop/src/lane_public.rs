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
//! The kernel `K_{a, r}` is linear in `a`, so the prover combines the rows of each record
//! with `eq(r^(2)_sig, i) gamma_rho` and transforms that combination once per record and
//! column `s`. The verifier's `w_A^(s)(r^(3))` is
//! `sum_i eq(r^(2)_sig, i) eq(r^(3)_sig, i) <A~_{i,s}, Lambda>` with the correlation `Lambda`
//! of `(r^(2)_coef, r^(3)_coef)`, one dot product per key, row and column. Padded records
//! carry zero weight and `T = 0`.
//!
//! The prover builds the data from its instance ([`LanePublicData::from_instance`], `T`
//! stored). The verifier builds [`LaneStreamedPublic`] from the `(pk, M', c~)` triples: keys
//! are decoded once per distinct `pk` (or taken from a [`LaneKeyCache`]), `T` is not stored
//! (each record's `c_i * t1_i` is recomputed from the sparse challenge in the one pass of
//! [`LaneStreamedPublic::t_evaluations`]), and `A` of an uncached key is expanded inside the
//! pass of [`LaneStreamedPublic::ring_weights_at`], eight keys at a time, so that it never
//! leaves the core's caches (it is kept only to fill a cache).

use std::any::Any;
use std::collections::HashMap;
use std::ops::Deref;
use std::sync::Arc;

use aerie_falcon::protocol::algebra::eq_table;
use agg_mldsa_simd::{ExpandedA, Form, Scale};
use agg_witness::expand::ExpandedKey;
use agg_witness::instance::{InstanceError, PublicInstance};
use agg_witness::native::pk_decode_for;
use agg_witness::padding::Geometry;
use agg_witness::params::{ParamSet, Set, Set44, Set65, Set87, N, TR_BYTES};
use jolt_field::{Ring, Zero};
use rayon::prelude::*;

use crate::public_data::{eq_correlation, SplitWeights};
use crate::public_stream::distinct_keys;
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
    pub a: LaneMatrix,
}

impl LaneKey {
    /// The lane view of an expanded key (its matrix shared, not copied).
    #[must_use]
    pub fn from_expanded<P: ParamSet>(key: &ExpandedKey<P>) -> Self {
        Self {
            set: P::SET,
            pk: key.pk.clone(),
            tr: key.tr,
            t1: (0..P::K).map(|rho| key.t1[rho]).collect(),
            a: LaneMatrix::shared::<P>(&key.a),
        }
    }
}

/// The centered matrix `A` of one key, read as one flat `(rho, s, j)` slice: owned by the
/// verifier (decoded or loaded from a cache file), or shared with the prover's expanded key,
/// whose `k x l x 256` array has the same layout, so that the prover holds one copy per key.
#[derive(Clone)]
pub struct LaneMatrix(MatrixStore);

#[derive(Clone)]
enum MatrixStore {
    Owned(Vec<i32>),
    Set65(Arc<<Set65 as ParamSet>::MatrixCoeff>),
    Set87(Arc<<Set87 as ParamSet>::MatrixCoeff>),
}

impl LaneMatrix {
    /// The expanded key's matrix of a set with polynomial lanes, shared; the matrix of any
    /// other set is copied.
    fn shared<P: ParamSet>(a: &Arc<P::MatrixCoeff>) -> Self {
        let any: Arc<dyn Any + Send + Sync> = a.clone();
        let store = match any.downcast() {
            Ok(m) => MatrixStore::Set65(m),
            Err(any) => match any.downcast() {
                Ok(m) => MatrixStore::Set87(m),
                Err(_) => MatrixStore::Owned(
                    (0..P::K)
                        .flat_map(|rho| (0..P::L).flat_map(move |s| *P::get_coeff(a, rho, s)))
                        .collect(),
                ),
            },
        };
        Self(store)
    }
}

impl From<Vec<i32>> for LaneMatrix {
    fn from(a: Vec<i32>) -> Self {
        Self(MatrixStore::Owned(a))
    }
}

impl Deref for LaneMatrix {
    type Target = [i32];
    fn deref(&self) -> &[i32] {
        match &self.0 {
            MatrixStore::Owned(a) => a,
            MatrixStore::Set65(a) => a.as_flattened().as_flattened(),
            MatrixStore::Set87(a) => a.as_flattened().as_flattened(),
        }
    }
}

impl PartialEq for LaneMatrix {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl Eq for LaneMatrix {}

impl std::fmt::Debug for LaneMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (**self).fmt(f)
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

/// The prover's public data of a batch of a parameter set with polynomial lanes.
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
    /// `T_i[rho][j]` of every live record, flat `(i, rho, j)`; the prover shares its
    /// instance's buffer.
    t: Arc<Vec<i32>>,
}

/// `T~_rho = sum_i eq_sig[i] <T_i[rho], eq_coef>` for every row `rho`. `rows(i, out)` writes
/// the rows of `T_i` divided by `2^d` and returns true (16-bit values, `|c_i t1_i| <= tau
/// (2^10 - 1) < 2^16`), or writes `T_i` itself and returns false (a row that is not a
/// multiple of `2^d`, which [`SplitWeights::dot`] sums on its exact path).
fn t_evaluations_by(
    records: usize,
    k: usize,
    d: u32,
    point: (&[E], &[E]),
    rows: impl Fn(usize, &mut [[i32; N]]) -> bool + Sync,
) -> Vec<E> {
    let (coef, sig) = point;
    let eq_coef = SplitWeights::new(&eq_table(coef));
    let eq_sig = eq_table(sig);
    let scale = E::from_u64(1 << d);
    // Per chunk of 64 records: rows given divided by `2^d` go to `scaled`, exact rows to
    // `plain`; the chunk's scratch rows live in its own stack frame.
    let sums = (0..records.div_ceil(64))
        .into_par_iter()
        .map(|chunk| {
            let mut scratch = [[0_i32; N]; MAX_ROWS];
            let scratch_rows = &mut scratch[..k];
            let (mut scaled, mut plain) = (vec![E::zero(); k], vec![E::zero(); k]);
            for i in chunk * 64..(chunk * 64 + 64).min(records) {
                let acc = if rows(i, scratch_rows) { &mut scaled } else { &mut plain };
                for (slot, row) in acc.iter_mut().zip(scratch_rows.iter()) {
                    *slot += eq_sig[i] * eq_coef.dot::<16>(row);
                }
            }
            (scaled, plain)
        })
        .reduce(
            || (vec![E::zero(); k], vec![E::zero(); k]),
            |(mut scaled, mut plain), (s, p)| {
                scaled.iter_mut().zip(s).for_each(|(a, b)| *a += b);
                plain.iter_mut().zip(p).for_each(|(a, b)| *a += b);
                (scaled, plain)
            },
        );
    sums.0.into_iter().zip(sums.1).map(|(scaled, plain)| scale * scaled + plain).collect()
}

/// `sum_i eq2[i] eq3[i] per_key(key_of[i])`: one weight per record against its key's `l`
/// values, reduced over chunks of records in parallel.
fn sum_over_records<'k>(key_of: &[usize], eq2: &[E], eq3: &[E], l: usize, per_key: impl Fn(usize) -> &'k [E] + Sync) -> Vec<E> {
    key_of
        .par_chunks(1024)
        .enumerate()
        .map(|(chunk, keys)| {
            let mut out = vec![E::zero(); l];
            for (i, &key) in (chunk * 1024..).zip(keys) {
                let weight = eq2[i] * eq3[i];
                for (slot, &value) in out.iter_mut().zip(per_key(key)) {
                    *slot += weight * value;
                }
            }
            out
        })
        .reduce(
            || vec![E::zero(); l],
            |mut acc, part| {
                acc.iter_mut().zip(part).for_each(|(a, b)| *a += b);
                acc
            },
        )
}

/// The most rows `k` of a parameter set (ML-DSA-87).
const MAX_ROWS: usize = 8;

impl LanePublicData {
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
        // The instance's flat buffer when every record still views its own slot of it, else
        // a copy (a record whose `T` was changed after the instance was built).
        let t = instance.t_buffer().map_or_else(
            || {
                let mut t = Vec::with_capacity(instance.records.len() * k * N);
                for record in &instance.records {
                    for rho in 0..k {
                        t.extend_from_slice(&record.t[rho]);
                    }
                }
                Arc::new(t)
            },
            Arc::clone,
        );
        Self {
            set: P::SET,
            k,
            l,
            padded: instance.geom.n_star,
            keys,
            key_of,
            mu: instance.records.iter().map(|r| r.mu).collect(),
            t,
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

    /// `T_i` of live record `i`, one row per `rho`.
    #[must_use]
    pub fn t_rows(&self, i: usize) -> &[[i32; N]] {
        self.t[i * self.k * N..(i + 1) * self.k * N].as_chunks::<N>().0
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

    /// `T~_rho(r) = sum_{(j, i)} eq(r, (j, i)) T_i[rho][j]` for every row `rho`. An
    /// instance's `T_i = 2^d (c_i t1_i)` (`agg_witness::expand::public_t_into_for`) is a
    /// multiple of `2^d`, so each row's dot runs on `T_i / 2^d` (16 bits) and is scaled once
    /// per record; a record whose `T_i` is not (a synthetic statement) is summed exactly.
    #[must_use]
    pub fn t_evaluations(&self, point: &[E]) -> Vec<E> {
        let d = agg_witness::dispatch_set!(self.set, |P| P::D);
        let low = (1_i32 << d) - 1;
        t_evaluations_by(self.records(), self.k, d, self.split(point), |i, out| {
            let t = self.t_rows(i);
            let mut bits = 0;
            for (dst, src) in out.iter_mut().zip(t) {
                for (x, &v) in dst.iter_mut().zip(src) {
                    *x = v >> d;
                    bits |= v;
                }
            }
            if bits & low != 0 {
                out.copy_from_slice(t);
            }
            bits & low == 0
        })
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
                let a: &[i32] = &key.a;
                (0..l)
                    .map(|s| {
                        (0..k).fold(E::zero(), |acc, rho| {
                            let row = &a[(rho * l + s) * N..(rho * l + s + 1) * N];
                            for (x, &entry) in ints.iter_mut().zip(row) {
                                *x = i64::from(entry);
                            }
                            acc + gammas[rho] * agg_simd::dot::k_int_dot(lambda_planes.planes(), &ints)
                        })
                    })
                    .collect()
            })
            .collect();
        sum_over_records(&self.key_of, &eq2, &eq3, l, |key| &per_key[key])
    }
}

/// A distinct key of a verified batch: from the cache (expanded), or decoded from its `pk`
/// here and expanded in the pass.
#[derive(Clone, Debug)]
enum StreamKey<'a> {
    Cached(Arc<LaneKey>),
    Fresh(FreshKey<'a>),
}

/// A key decoded by the verifier (`rho` is the first 32 bytes of `pk`).
#[derive(Clone, Debug)]
struct FreshKey<'a> {
    pk: &'a [u8],
    tr: [u8; TR_BYTES],
    t1: Box<[[u16; N]]>,
}

impl StreamKey<'_> {
    fn tr(&self) -> &[u8; TR_BYTES] {
        match self {
            Self::Cached(key) => &key.tr,
            Self::Fresh(key) => &key.tr,
        }
    }

    fn t1(&self) -> &[[u16; N]] {
        match self {
            Self::Cached(key) => &key.t1,
            Self::Fresh(key) => &key.t1,
        }
    }
}

/// `c = SampleInBall(c~)` of one record in sparse form: the first `tau` entries of `pos` are
/// the increasing positions of the nonzero coefficients, bit `m` of `neg` is set when
/// `c[pos[m]] = -1` (the form of `agg_mldsa_simd::Challenge`, for any `tau <= 64`).
fn narrow<const TAU: usize>(pos: &[u8; 64], neg: u64) -> agg_mldsa_simd::Challenge<TAU> {
    agg_mldsa_simd::Challenge {
        pos: pos[..TAU].try_into().expect("tau positions"),
        neg,
    }
}

const _: () = assert!(Set44::TAU == 39 && Set65::TAU == 49 && Set87::TAU == 60);

/// `SampleInBall` of up to 64 seeds of `set` (`agg_mldsa_simd::Engine::sample_in_ball_many`),
/// as positions and sign masks ([`narrow`]).
fn sample_in_ball_tile(
    engine: &agg_mldsa_simd::Engine,
    set: Set,
    seeds: &[&[u8]],
    out: (&mut [[u8; 64]], &mut [u64]),
) {
    fn run<const TAU: usize>(engine: &agg_mldsa_simd::Engine, seeds: &[&[u8]], out: (&mut [[u8; 64]], &mut [u64])) {
        let mut tile = [agg_mldsa_simd::Challenge::<TAU>::zero(); 64];
        let tile = &mut tile[..seeds.len()];
        engine.sample_in_ball_many(seeds, tile);
        for ((pos, neg), c) in out.0.iter_mut().zip(out.1.iter_mut()).zip(tile.iter()) {
            pos[..TAU].copy_from_slice(&c.pos);
            *neg = c.neg;
        }
    }
    match set {
        Set::S44 => run::<39>(engine, seeds, out),
        Set::S65 => run::<49>(engine, seeds, out),
        Set::S87 => run::<60>(engine, seeds, out),
    }
}

/// The verifier's public data of a batch of a parameter set with polynomial lanes, from the
/// `(pk, M', c~)` triples (module docs): the distinct keys (cached, or decoded with `tr`),
/// `mu` and the sparse `c` of every record. Fresh keys borrow their `pk` from the triples.
#[derive(Clone, Debug)]
pub struct LaneStreamedPublic<'a> {
    set: Set,
    k: usize,
    l: usize,
    padded: usize,
    /// Every distinct key.
    keys: Vec<StreamKey<'a>>,
    /// The key of every live record.
    key_of: Vec<usize>,
    /// `mu_i` of every live record.
    mu: Vec<[u8; TR_BYTES]>,
    /// `c_i` of every live record: its positions and its sign mask ([`narrow`]), kept apart
    /// so that both vectors start as zero pages and are first touched by the parallel writer.
    positions: Vec<[u8; 64]>,
    negs: Vec<u64>,
}

impl<'a> LaneStreamedPublic<'a> {
    /// The verifier's data from `(pk, M', c~)` triples: per distinct
    /// `pk` its `cache` entry, or `pkDecode` and `tr = H(pk)` (eight keys per multi-lane
    /// call); per record `mu = H(tr || 0 || 0 || M)` and `SampleInBall(c~)` (64 records per
    /// call).
    ///
    /// # Errors
    /// On a batch size outside `1..=2^15` or a public key of the wrong length (a record
    /// signed under it).
    ///
    /// # Panics
    /// If `cache` holds another parameter set.
    pub fn prepare<P: ParamSet>(
        inputs: &[(&'a [u8], &[u8], P::CTilde)],
        cache: Option<&LaneKeyCache>,
    ) -> Result<Self, InstanceError>
    where
        P::CTilde: Sync,
    {
        let geom = Geometry::<P>::new_for(inputs.len()).map_err(InstanceError::Geometry)?;
        if let Some(cache) = cache {
            assert_eq!(cache.set, P::SET, "a cache of the batch's parameter set");
        }
        let engine = agg_mldsa_simd::engine();
        let (distinct, key_of) = distinct_keys(inputs);
        let keys: Vec<StreamKey<'a>> = distinct
            .par_chunks(8)
            .map(|tile| -> Result<Vec<StreamKey<'a>>, InstanceError> {
                let mut keys = Vec::with_capacity(tile.len());
                let mut fresh = Vec::with_capacity(tile.len());
                for (pk, records) in tile {
                    let key = match cache.and_then(|c| c.get(pk)) {
                        // The length check keeps the error of pkDecode for any cache content.
                        Some(cached) if pk.len() == P::PK_BYTES => StreamKey::Cached(cached),
                        _ => {
                            let decoded = pk_decode_for::<P>(pk).ok_or(InstanceError::PublicKey(records[0]))?;
                            fresh.push(keys.len());
                            StreamKey::Fresh(FreshKey {
                                pk,
                                tr: [0; TR_BYTES],
                                t1: (0..P::K).map(|rho| decoded.t1[rho]).collect(),
                            })
                        }
                    };
                    keys.push(key);
                }
                if !fresh.is_empty() {
                    let inputs: Vec<[&[u8]; 4]> = fresh
                        .iter()
                        .map(|&at| match &keys[at] {
                            StreamKey::Fresh(key) => [key.pk, &[], &[], &[]],
                            StreamKey::Cached(_) => unreachable!("a fresh key"),
                        })
                        .collect();
                    let mut digests = [0_u8; 8 * TR_BYTES];
                    let digests = &mut digests[..fresh.len() * TR_BYTES];
                    engine.shake256_many(&inputs, TR_BYTES, digests);
                    for (&at, digest) in fresh.iter().zip(digests.chunks_exact(TR_BYTES)) {
                        if let StreamKey::Fresh(key) = &mut keys[at] {
                            key.tr.copy_from_slice(digest);
                        }
                    }
                }
                Ok(keys)
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect();
        let mut mu = vec![[0_u8; TR_BYTES]; inputs.len()];
        let mut positions = vec![[0_u8; 64]; inputs.len()];
        let mut negs = vec![0_u64; inputs.len()];
        positions
            .par_chunks_mut(64)
            .zip(negs.par_chunks_mut(64))
            .zip(mu.par_chunks_mut(64))
            .zip(inputs.par_chunks(64).zip(key_of.par_chunks(64)))
            .for_each(|(((positions, negs), mu), (tile, key_tile))| {
                let mu_inputs: Vec<[&[u8]; 4]> = tile
                    .iter()
                    .zip(key_tile)
                    .map(|(&(_, msg, _), &key)| [&keys[key].tr()[..], msg, &[][..], &[][..]])
                    .collect();
                engine.shake256_many(&mu_inputs, TR_BYTES, mu.as_flattened_mut());
                let seeds: Vec<&[u8]> = tile.iter().map(|(_, _, c)| c.as_ref()).collect();
                sample_in_ball_tile(engine, P::SET, &seeds, (positions, negs));
            });
        Ok(Self {
            set: P::SET,
            k: P::K,
            l: P::L,
            padded: geom.n_star,
            keys,
            key_of,
            mu,
            positions,
            negs,
        })
    }

    /// The parameter set.
    #[must_use]
    pub const fn set(&self) -> Set {
        self.set
    }

    /// `mu_i` of every live record.
    #[must_use]
    pub fn mu(&self) -> &[[u8; TR_BYTES]] {
        &self.mu
    }

    /// Live records `N`.
    #[must_use]
    pub fn records(&self) -> usize {
        self.key_of.len()
    }

    /// Distinct keys not found in the cache (expanded by [`Self::ring_weights_at`]).
    #[must_use]
    pub fn expanded_keys(&self) -> usize {
        self.keys.iter().filter(|key| matches!(key, StreamKey::Fresh(_))).count()
    }

    /// `c_i t1_i` of live record `i`, one row per `rho`, into `out`.
    fn ct_rows(&self, engine: &agg_mldsa_simd::Engine, i: usize, out: &mut [[i32; N]]) {
        let (pos, neg, t1) = (&self.positions[i], self.negs[i], self.keys[self.key_of[i]].t1());
        match self.set {
            Set::S44 => engine.sparse_ct_slice(&narrow::<39>(pos, neg), t1, 0, out),
            Set::S65 => engine.sparse_ct_slice(&narrow::<49>(pos, neg), t1, 0, out),
            Set::S87 => engine.sparse_ct_slice(&narrow::<60>(pos, neg), t1, 0, out),
        }
    }

    /// `T_i` of live record `i`, one row per `rho` (recomputed).
    #[must_use]
    pub fn t_rows(&self, i: usize) -> Vec<[i32; N]> {
        let mut rows = vec![[0_i32; N]; self.k];
        self.ct_rows(agg_mldsa_simd::engine(), i, &mut rows);
        let d = agg_witness::dispatch_set!(self.set, |P| P::D);
        rows.iter_mut().flatten().for_each(|x| *x <<= d);
        rows
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

    /// `T~_rho(r)` for every row `rho` ([`LanePublicData::t_evaluations`]), with `c_i t1_i`
    /// recomputed per record on the SIMD engine.
    #[must_use]
    pub fn t_evaluations(&self, point: &[E]) -> Vec<E> {
        let engine = agg_mldsa_simd::engine();
        let d = agg_witness::dispatch_set!(self.set, |P| P::D);
        t_evaluations_by(self.records(), self.k, d, self.split(point), |i, out| {
            self.ct_rows(engine, i, out);
            true
        })
    }

    /// `w_A^(s)(r3)` for every column `s` ([`LanePublicData::ring_weights_at`]): one dot
    /// product with `Lambda` per key, row and column, then one product per record and
    /// column. Uncached keys are expanded here in tiles of eight (`ExpandA`, inverse NTTs);
    /// with `cache` their expansions are added to it.
    ///
    /// # Panics
    /// If `gammas` does not have `k` entries, a point is not a main-domain point, or `cache`
    /// holds another parameter set.
    #[must_use]
    pub fn ring_weights_at(&self, r2: &[E], r3: &[E], gammas: &[E], mut cache: Option<&mut LaneKeyCache>) -> Vec<E> {
        assert_eq!(gammas.len(), self.k, "one coefficient per row");
        if let Some(cache) = cache.as_ref() {
            assert_eq!(cache.set, self.set, "a cache of the batch's parameter set");
        }
        let (coef2, sig2) = self.split(r2);
        let (coef3, sig3) = self.split(r3);
        let lambda = eq_correlation(coef2, coef3);
        let (eq2, eq3) = (eq_table(sig2), eq_table(sig3));
        let lambda_planes = agg_simd::buf::KPlanes::from_fn(N, |j| lambda[j]);
        let keep = cache.is_some();
        let tiles: Vec<(Vec<E>, Vec<Arc<LaneKey>>)> = match self.set {
            Set::S65 => self.key_weights::<6, 5>(&lambda_planes, gammas, keep),
            Set::S87 => self.key_weights::<8, 7>(&lambda_planes, gammas, keep),
            Set::S44 => self.key_weights::<4, 4>(&lambda_planes, gammas, keep),
        };
        let mut per_key = Vec::with_capacity(self.keys.len() * self.l);
        for (weights, fresh) in tiles {
            per_key.extend(weights);
            if let Some(cache) = cache.as_deref_mut() {
                for key in fresh {
                    cache.insert(key);
                }
            }
        }
        let l = self.l;
        sum_over_records(&self.key_of, &eq2, &eq3, l, |key| &per_key[key * l..(key + 1) * l])
    }

    /// Per tile of eight keys: `sum_rho gamma_rho <A[rho][s], Lambda>` for every key and
    /// column `s` (key-major), and with `keep` the expansions of the tile's uncached keys.
    fn key_weights<const K: usize, const L: usize>(
        &self,
        lambda: &agg_simd::buf::KPlanes,
        gammas: &[E],
        keep: bool,
    ) -> Vec<(Vec<E>, Vec<Arc<LaneKey>>)> {
        assert_eq!((K, L), (self.k, self.l), "the set's matrix dimensions");
        let engine = agg_mldsa_simd::engine();
        // `A` entries are centered mod q (`|A| < 2^22`), within the kernel's `2^51` bound.
        let dot = |row: &[i32], ints: &mut [i64; N]| -> E {
            for (x, &entry) in ints.iter_mut().zip(row) {
                *x = i64::from(entry);
            }
            agg_simd::dot::k_int_dot(lambda.planes(), ints)
        };
        self.keys
            .par_chunks(8)
            // `map_init` runs once per job: eight tiles per job bound the scratch fills.
            .with_min_len(8)
            .map_init(
                || vec![ExpandedA::<K, L>::zero(); 8],
                |expanded, tile| {
                    let rhos: Vec<[u8; 32]> = tile
                        .iter()
                        .filter_map(|key| match key {
                            StreamKey::Fresh(key) => Some(key.pk[..32].try_into().expect("rho")),
                            StreamKey::Cached(_) => None,
                        })
                        .collect();
                    let expanded = &mut expanded[..rhos.len()];
                    if !rhos.is_empty() {
                        engine.expand_a_many_kl::<K, L>(&rhos, expanded);
                    }
                    let mut expanded = expanded.iter_mut();
                    let mut ints = [0_i64; N];
                    let mut weights = Vec::with_capacity(tile.len() * L);
                    let mut fresh_keys = Vec::new();
                    for key in tile {
                        match key {
                            StreamKey::Cached(key) => {
                                let a: &[i32] = &key.a;
                                for s in 0..L {
                                    weights.push((0..K).fold(E::zero(), |acc, rho| {
                                        acc + gammas[rho] * dot(&a[(rho * L + s) * N..(rho * L + s + 1) * N], &mut ints)
                                    }));
                                }
                            }
                            StreamKey::Fresh(fresh) => {
                                let a = &mut expanded.next().expect("one expansion per fresh key").a_hat;
                                for poly in a.iter_mut().flatten() {
                                    engine.intt(agg_mldsa_simd::Prime::Q, poly, Scale::Plain, Form::Centered);
                                }
                                for s in 0..L {
                                    weights.push((0..K).fold(E::zero(), |acc, rho| {
                                        acc + gammas[rho] * dot(&a[rho][s].0, &mut ints)
                                    }));
                                }
                                if keep {
                                    let mut flat = Vec::with_capacity(K * L * N);
                                    for poly in a.iter().flatten() {
                                        flat.extend_from_slice(&poly.0);
                                    }
                                    fresh_keys.push(Arc::new(LaneKey {
                                        set: self.set,
                                        pk: fresh.pk.to_vec(),
                                        tr: fresh.tr,
                                        t1: fresh.t1.to_vec(),
                                        a: flat.into(),
                                    }));
                                }
                            }
                        }
                    }
                    (weights, fresh_keys)
                },
            )
            .collect()
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

    /// The verifier's data from `(pk, M', c~)` equals the prover's: the same `mu`, `T` rows,
    /// public evaluations and (in a cache) keys, uncached and with a cold and a warm cache.
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
            .map(|r| (r.pk.as_slice(), r.m_prime.as_slice(), P::c_tilde_from_slice(&r.sig)))
            .collect();
        let mut rng = SplitMix64::new(93);
        let vars = COEF_VARS + prover.padded_records().trailing_zeros() as usize;
        let (r2, r3, gammas) = (point(&mut rng, vars), point(&mut rng, vars), point(&mut rng, P::K));
        let expected = (prover.t_evaluations(&r2), prover.ring_weights_at(&r2, &r3, &gammas));
        // Uncached, then a cold cache (filled by the pass), then the warm cache.
        let mut cache = LaneKeyCache::new(P::SET);
        for (round, use_cache) in [(0, false), (1, true), (2, true)] {
            let verifier = LaneStreamedPublic::prepare::<P>(&inputs, use_cache.then_some(&cache)).unwrap();
            assert_eq!(verifier.set(), P::SET);
            assert_eq!(verifier.expanded_keys(), if round == 2 { 0 } else { 6 }, "round {round}");
            assert_eq!(verifier.mu(), prover.mu());
            for i in 0..records.len() {
                assert_eq!(verifier.t_rows(i), prover.t_rows(i), "record {i}");
            }
            let t = verifier.t_evaluations(&r2);
            let w = verifier.ring_weights_at(&r2, &r3, &gammas, use_cache.then_some(&mut cache));
            assert_eq!((t, w), expected, "round {round}");
            assert_eq!(cache.len(), if round == 0 { 0 } else { 6 }, "round {round}");
        }
        // The cached expansions are the prover's keys.
        for key in prover.keys() {
            assert_eq!(*cache.get(&key.pk).expect("cached"), **key);
        }
        // A key of the wrong length is refused at its first record.
        let mut bad = inputs.clone();
        let short = &records[3].pk[..10];
        bad[3].0 = short;
        assert_eq!(LaneStreamedPublic::prepare::<P>(&bad, None).err(), Some(InstanceError::PublicKey(3)));
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
