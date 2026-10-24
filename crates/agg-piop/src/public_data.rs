//! Public data: the verifier's per-record precomputation (spec Section 2) and the
//! negacyclic `eq` helpers of the public weights (spec Section 7.6, Aerie Section 8).
//!
//! Per distinct key the verifier expands `A_hat = ExpandA(rho)` and keeps only the centered
//! coefficient form `A = ctr_q(NTT^{-1}(A_hat))` (16 inverse NTTs, 11.8 KB per key, MAY be
//! cached across batches through [`KeyCache`]); per live record it forms `mu`,
//! `c = SampleInBall(c~)` and `T = 2^d (c * t1)` over the integers from the `tau` sparse
//! entries of `c` (no reduction, no NTT; `|T| < 2^29`, so `i32`). The public table
//! `T(j, rho, i)` is zero on padded rows; [`PublicData::t_evaluation`] streams it at a point.
//!
//! The ring weight pulls `eq(r_coef, .)` back through a polynomial `a`:
//! [`negacyclic_kernel`] computes `K_{a, r_coef}` of (7.8) by Aerie's factorized transform
//! (8.5) to (8.7) in `O(n log n)`, and [`eq_correlation`] the verifier's 256-entry vector
//! `Lambda` of Aerie (8.8) to (8.10) (the correlation of two `eq` vectors) in `O(n)`.

use std::collections::HashMap;
use std::sync::Arc;

use aerie_falcon::protocol::algebra::eq_table;
use agg_witness::expand::{expand_a, ExpandedKey, MatrixCoeff, MatrixNtt};
use agg_witness::instance::{InstanceError, PublicInstance};
use agg_witness::native::{ctr_q, mu, pk_decode, sample_in_ball, tr, Challenge};
use agg_witness::ntt::NttPrime;
use agg_witness::padding::{main_index, Geometry};
use agg_witness::params::{C_TILDE_BYTES, D, D_POS, K, L, N, TR_BYTES};
use jolt_field::{One, Ring, Zero};
use rayon::prelude::*;

use crate::OpeningField as E;

/// `T_i` of one record: `k` polynomials of 256 integer coefficients, `|T| < 2^29`.
pub type PublicT = [[i32; N]; K];

/// `A = ctr_q(NTT^{-1}(A_hat))` entrywise: 16 inverse NTTs (spec Section 2 item 2).
#[must_use]
pub fn inverse_ntt_matrix(a_hat: &MatrixNtt) -> Box<MatrixCoeff> {
    let fips = NttPrime::fips();
    let mut a = Box::new([[[0_i32; N]; L]; K]);
    for (row, row_hat) in a.iter_mut().zip(a_hat.iter()) {
        for (poly, poly_hat) in row.iter_mut().zip(row_hat.iter()) {
            for (out, coeff) in poly.iter_mut().zip(fips.intt(poly_hat)) {
                *out = ctr_q(coeff as i64) as i32;
            }
        }
    }
    a
}

/// `T[rho] = 2^d (c * t1[rho])` over `Z[X]/(X^256 + 1)` from the `tau` nonzero entries of
/// `c` (`k tau 256` signed additions; spec Section 2 item 5).
#[must_use]
pub fn sparse_t(c: &Challenge, t1: &[[u16; N]; K]) -> PublicT {
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

/// What the verifier keeps per key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifierKey {
    /// Encoded public key.
    pub pk: Vec<u8>,
    /// `A[rho][s]`, centered coefficient form.
    pub a: Box<MatrixCoeff>,
    /// `t1`, coefficients in `[0, 1023]`.
    pub t1: [[u16; N]; K],
    /// `tr = H(pk, 64)`.
    pub tr: [u8; TR_BYTES],
}

impl VerifierKey {
    /// pkDecode, ExpandA and the inverse NTTs; `None` on a wrong length.
    #[must_use]
    pub fn from_bytes(pk: &[u8]) -> Option<Self> {
        let decoded = pk_decode(pk)?;
        let (a_hat, _) = expand_a(&decoded.rho);
        Some(Self {
            pk: pk.to_vec(),
            a: inverse_ntt_matrix(&a_hat),
            t1: decoded.t1,
            tr: tr(pk),
        })
    }

    /// The verifier's part of the prover's expanded key.
    #[must_use]
    pub fn from_expanded(key: &ExpandedKey) -> Self {
        Self {
            pk: key.pk.clone(),
            a: key.a.clone(),
            t1: key.t1,
            tr: key.tr,
        }
    }
}

/// Expanded keys kept across batches (the "cached" verifier of spec Section 11).
#[derive(Clone, Debug, Default)]
pub struct KeyCache {
    keys: HashMap<Vec<u8>, Arc<VerifierKey>>,
}

impl KeyCache {
    /// Number of cached keys.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// True when empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// The cached expansion of `pk`, if any.
    #[must_use]
    pub fn get(&self, pk: &[u8]) -> Option<Arc<VerifierKey>> {
        self.keys.get(pk).cloned()
    }

    /// Adds an expansion (an existing entry for the same key is kept).
    pub fn insert(&mut self, key: Arc<VerifierKey>) {
        self.keys.entry(key.pk.clone()).or_insert(key);
    }

    /// The cached expansions, in no particular order (to persist the cache).
    pub fn entries(&self) -> impl Iterator<Item = &Arc<VerifierKey>> {
        self.keys.values()
    }
}

/// Public data of one live record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicRecordData {
    /// Index of the record's key in [`PublicData::keys`].
    pub key: usize,
    /// `mu = H(tr || 0 || 0 || M, 64)`.
    pub mu: [u8; TR_BYTES],
    /// `c~` from the aggregate.
    pub c_tilde: [u8; C_TILDE_BYTES],
    /// `T = 2^d (c * t1)`.
    pub t: Box<PublicT>,
}

/// The public instance as the verifier precomputes it.
#[derive(Clone, Debug)]
pub struct PublicData {
    geom: Geometry,
    keys: Vec<Arc<VerifierKey>>,
    records: Vec<PublicRecordData>,
}

impl PublicData {
    /// Precomputes the batch from `(pk, M, c~)` triples (pure mode, empty context). Keys
    /// are expanded once per distinct `pk`; with a cache, cached keys are not expanded and
    /// new ones are added.
    pub fn prepare(
        inputs: &[(&[u8], &[u8], [u8; C_TILDE_BYTES])],
        mut cache: Option<&mut KeyCache>,
    ) -> Result<Self, InstanceError> {
        let geom = Geometry::new(inputs.len()).map_err(InstanceError::Geometry)?;
        let mut distinct: Vec<&[u8]> = Vec::new();
        let mut index: HashMap<&[u8], usize> = HashMap::new();
        let mut first_record = Vec::new();
        for (i, &(pk, _, _)) in inputs.iter().enumerate() {
            index.entry(pk).or_insert_with(|| {
                distinct.push(pk);
                first_record.push(i);
                distinct.len() - 1
            });
        }
        let cached: Vec<Option<Arc<VerifierKey>>> = distinct
            .iter()
            .map(|&pk| cache.as_ref().and_then(|c| c.keys.get(pk).cloned()))
            .collect();
        let keys = distinct
            .par_iter()
            .zip(cached)
            .zip(&first_record)
            .map(|((pk, hit), &record)| match hit {
                Some(key) => Ok(key),
                None => VerifierKey::from_bytes(pk)
                    .map(Arc::new)
                    .ok_or(InstanceError::PublicKey(record)),
            })
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(cache) = cache.as_mut() {
            for key in &keys {
                cache
                    .keys
                    .entry(key.pk.clone())
                    .or_insert_with(|| Arc::clone(key));
            }
        }
        let records = inputs
            .par_iter()
            .map(|&(pk, msg, c_tilde)| {
                let key = index[pk];
                let c = sample_in_ball(&c_tilde);
                PublicRecordData {
                    key,
                    mu: mu(&keys[key].tr, msg, &[]),
                    c_tilde,
                    t: Box::new(sparse_t(&c, &keys[key].t1)),
                }
            })
            .collect();
        Ok(Self {
            geom,
            keys,
            records,
        })
    }

    /// The same data taken from the prover's instance (keys shared by `Arc` identity).
    #[must_use]
    pub fn from_instance(instance: &PublicInstance) -> Self {
        let mut keys: Vec<Arc<VerifierKey>> = Vec::new();
        let mut seen: HashMap<*const ExpandedKey, usize> = HashMap::new();
        let records = instance
            .records
            .iter()
            .map(|record| {
                let key = *seen.entry(Arc::as_ptr(&record.key)).or_insert_with(|| {
                    keys.push(Arc::new(VerifierKey::from_expanded(&record.key)));
                    keys.len() - 1
                });
                let mut t = Box::new([[0_i32; N]; K]);
                for (row, source) in t.iter_mut().zip(&record.t) {
                    for (out, &value) in row.iter_mut().zip(source) {
                        *out = i32::try_from(value).expect("|T| < 2^29");
                    }
                }
                PublicRecordData {
                    key,
                    mu: record.mu,
                    c_tilde: record.c_tilde,
                    t,
                }
            })
            .collect();
        Self {
            geom: instance.geom,
            keys,
            records,
        }
    }

    /// Batch geometry.
    #[must_use]
    pub const fn geometry(&self) -> Geometry {
        self.geom
    }

    /// Distinct keys.
    #[must_use]
    pub fn keys(&self) -> &[Arc<VerifierKey>] {
        &self.keys
    }

    /// Live records.
    #[must_use]
    pub fn records(&self) -> &[PublicRecordData] {
        &self.records
    }

    /// `A_i` of live record `i`.
    #[must_use]
    pub fn a(&self, i: usize) -> &MatrixCoeff {
        &self.keys[self.records[i].key].a
    }

    /// `T_i` of live record `i`.
    #[must_use]
    pub fn t(&self, i: usize) -> &PublicT {
        &self.records[i].t
    }

    /// The public table `T(j, rho, i)` on the main domain, zero on padded rows.
    #[must_use]
    pub fn t_table(&self) -> Vec<i64> {
        let mut out = vec![0_i64; self.geom.main_len()];
        for (i, record) in self.records.iter().enumerate() {
            for (rho, row) in record.t.iter().enumerate() {
                for (j, &value) in row.iter().enumerate() {
                    out[main_index(j, rho, i)] = i64::from(value);
                }
            }
        }
        out
    }

    /// `T(r) = sum_x eq(r, x) T(x)` at a main-domain point `r = (r_coef, r_row, r_sig)`,
    /// streamed over the live records (`k 256` multiply-adds per record).
    #[must_use]
    pub fn t_evaluation(&self, point: &[E]) -> E {
        assert_eq!(
            point.len(),
            self.geom.d_main() as usize,
            "main-domain point"
        );
        let (position, signature) = point.split_at(D_POS as usize);
        let eq_position = eq_table(position);
        let eq_signature = eq_table(signature);
        self.records
            .par_iter()
            .enumerate()
            .map(|(i, record)| {
                let inner = record
                    .t
                    .iter()
                    .flatten()
                    .zip(&eq_position)
                    .fold(E::zero(), |sum, (&value, &weight)| {
                        sum + weight.mul_i64(i64::from(value))
                    });
                eq_signature[i] * inner
            })
            .reduce(E::zero, |left, right| left + right)
    }

    /// `A~_{i,s} = sum_rho eq(r_row, rho) A_i[rho, s]` for every column `s` (spec (7.9)).
    #[must_use]
    pub fn a_row_combination(&self, i: usize, r_row: &[E]) -> [[E; N]; L] {
        let weights = eq_table(r_row);
        assert_eq!(weights.len(), K, "two row coordinates");
        let a = self.a(i);
        std::array::from_fn(|s| {
            std::array::from_fn(|j| {
                (0..K).fold(E::zero(), |sum, rho| {
                    sum + weights[rho].mul_i64(i64::from(a[rho][s][j]))
                })
            })
        })
    }
}

/// `w^(t) (X) = (f_t(0) + f_t(1) X) w^(t+1)(X^2)`: returns `a * eq(r, .)` in
/// `K[X]/(X^D + 1)` for `D = a.len() = 2^|r|` by the recurrence (8.7).
fn eq_negacyclic_product(a: &[E], r: &[E]) -> Vec<E> {
    let size = a.len();
    if size == 1 {
        return vec![a[0]];
    }
    let (f0, f1) = (E::one() - r[0], r[0]);
    let half = size / 2;
    let mut even = Vec::with_capacity(half);
    let mut odd = Vec::with_capacity(half);
    for k in 0..half {
        let (a0, a1) = (a[2 * k], a[2 * k + 1]);
        // rot(A_1) = Y A_1 mod (Y^{D/2} + 1).
        let rotated = if k == 0 { -a[size - 1] } else { a[2 * k - 1] };
        even.push(f0 * a0 + f1 * rotated);
        odd.push(f1 * a0 + f0 * a1);
    }
    let even = eq_negacyclic_product(&even, &r[1..]);
    let odd = eq_negacyclic_product(&odd, &r[1..]);
    even.into_iter()
        .zip(odd)
        .flat_map(|(e, o)| [e, o])
        .collect()
}

/// `K_{a, r}[j'] = sum_{a'} a[a'] (-1)^{floor((a' + j') / n)} eq(r, (a' + j') mod n)`
/// (spec (7.8), Aerie (8.2)), so that `sum_j eq(r, j) (a * z)_j = sum_{j'} K[j'] z[j']` for
/// the negacyclic product. Computed as `a^dagger * w^(0)` with `a^dagger[0] = a[0]`,
/// `a^dagger[j] = -a[n - j]` by the factorized transform (8.5) to (8.7), `O(n log n)`.
#[must_use]
pub fn negacyclic_kernel(a: &[E], r_coef: &[E]) -> Vec<E> {
    let n = a.len();
    assert_eq!(n, 1 << r_coef.len(), "a has 2^|r| coefficients");
    let mut adjoint = Vec::with_capacity(n);
    adjoint.push(a[0]);
    adjoint.extend((1..n).map(|j| -a[n - j]));
    eq_negacyclic_product(&adjoint, r_coef)
}

/// `Lambda_a = sum_j eq(right, j) (-1)^{floor((a + j) / n)} eq(left, (a + j) mod n)`
/// (Aerie (8.8)) for every `a < n = 2^|left|`, by the carry trie (8.9), (8.10) in `O(n)`.
/// With `left = r^(2)_coef` and `right = r^(3)_coef`,
/// `sum_{j'} eq(right, j') K_{a, left}[j'] = sum_{a'} a[a'] Lambda_{a'}`.
#[must_use]
pub fn eq_correlation(left: &[E], right: &[E]) -> Vec<E> {
    assert_eq!(left.len(), right.len(), "two coefficient points");
    // states[prefix] = (carry 0, carry 1) over the low bits processed so far.
    let mut states = vec![(E::one(), E::zero())];
    for (&l, &n) in left.iter().zip(right) {
        let p = n * l;
        let g = E::one() - n - l + p + p;
        let (l_minus, n_minus) = (l - p, n - p);
        let width = states.len();
        let mut next = vec![(E::zero(), E::zero()); 2 * width];
        for (prefix, &(d0, d1)) in states.iter().enumerate() {
            next[prefix | width] = (l_minus * d0, n_minus * d0 + g * d1);
            next[prefix] = (g * d0 + l_minus * d1, n_minus * d1);
        }
        states = next;
    }
    states.into_iter().map(|(d0, d1)| d0 - d1).collect()
}

/// Definition of [`negacyclic_kernel`] (quadratic; reference for tests).
#[must_use]
pub fn negacyclic_kernel_direct(a: &[E], r_coef: &[E]) -> Vec<E> {
    let n = a.len();
    let weights = eq_table(r_coef);
    (0..n)
        .map(|j| {
            (0..n).fold(E::zero(), |sum, k| {
                let term = a[k] * weights[(k + j) % n];
                if k + j >= n {
                    sum - term
                } else {
                    sum + term
                }
            })
        })
        .collect()
}

/// Definition of [`eq_correlation`] (quadratic; reference for tests).
#[must_use]
pub fn eq_correlation_direct(left: &[E], right: &[E]) -> Vec<E> {
    let n = 1 << left.len();
    let (wl, wr) = (eq_table(left), eq_table(right));
    (0..n)
        .map(|a| {
            (0..n).fold(E::zero(), |sum, j| {
                let term = wr[j] * wl[(a + j) % n];
                if a + j >= n {
                    sum - term
                } else {
                    sum + term
                }
            })
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::test_support::{boolean_point, point};
    use aerie_falcon::protocol::algebra::{evaluate_mle, lift_table};
    use agg_witness::crt::negacyclic_schoolbook;
    use agg_witness::expand::public_t;
    use agg_witness::gen::{self, SplitMix64};

    fn small_instance(seed: u64) -> (PublicInstance, Vec<gen::Record>) {
        let mut records = gen::records(seed, 5);
        records.extend(gen::records_shared_keys(seed + 1, 6, 2));
        (PublicInstance::from_records(&records).unwrap(), records)
    }

    #[test]
    fn negacyclic_kernel_matches_its_definition() {
        let mut rng = SplitMix64::new(11);
        for vars in 0..=8 {
            let a = point(&mut rng, 1 << vars);
            let r = point(&mut rng, vars);
            assert_eq!(
                negacyclic_kernel(&a, &r),
                negacyclic_kernel_direct(&a, &r),
                "2^{vars}"
            );
        }
    }

    #[test]
    fn negacyclic_kernel_pulls_eq_back_through_the_ring_product() {
        // sum_j eq(r, j) (a * z)_j = sum_j' K_{a, r}[j'] z[j'] for integer a (an A_i entry) and z.
        let (instance, _) = small_instance(3);
        let a_poly = instance.records[0].key.a[1][2];
        let mut rng = SplitMix64::new(5);
        let mut a = [0_i64; N];
        let mut z = [0_i64; N];
        for j in 0..N {
            a[j] = i64::from(a_poly[j]);
            z[j] = rng.below(1 << 18) as i64 - (1 << 17);
        }
        let product = negacyclic_schoolbook(&a, &z);
        let r = point(&mut rng, 8);
        let weights = eq_table(&r);
        let lhs = (0..N).fold(E::zero(), |sum, j| {
            sum + weights[j] * E::from_i128(product[j])
        });
        let kernel = negacyclic_kernel(&lift_table(&a[..]), &r);
        let rhs = (0..N).fold(E::zero(), |sum, j| sum + kernel[j].mul_i64(z[j]));
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn eq_correlation_matches_its_definition_and_the_kernel() {
        let mut rng = SplitMix64::new(17);
        for vars in 0..=8 {
            let (left, right) = (point(&mut rng, vars), point(&mut rng, vars));
            let lambda = eq_correlation(&left, &right);
            assert_eq!(lambda, eq_correlation_direct(&left, &right), "2^{vars}");
            // sum_j' eq(right, j') K_{a, left}[j'] = sum_a' a[a'] Lambda_a'.
            let a = point(&mut rng, 1 << vars);
            let kernel = negacyclic_kernel(&a, &left);
            let weights = eq_table(&right);
            let lhs = kernel
                .iter()
                .zip(&weights)
                .fold(E::zero(), |sum, (&k, &w)| sum + k * w);
            let rhs = a
                .iter()
                .zip(&lambda)
                .fold(E::zero(), |sum, (&x, &l)| sum + x * l);
            assert_eq!(lhs, rhs, "2^{vars}");
        }
        // At Boolean points Lambda is a signed indicator: (-1)^{[a + j >= n]} [a + j = l mod n].
        let n = 16;
        for (l, j) in [(3, 5), (2, 9), (15, 15), (0, 0)] {
            let lambda = eq_correlation(&boolean_point(l, 4), &boolean_point(j, 4));
            for (a, value) in lambda.into_iter().enumerate() {
                let expected = if (a + j) % n != l {
                    0
                } else if a + j >= n {
                    -1
                } else {
                    1
                };
                assert_eq!(value, E::from_i64(expected), "l {l} j {j} a {a}");
            }
        }
    }

    #[test]
    fn sparse_t_and_inverse_ntt_match_the_witness_crate() {
        let (instance, records) = small_instance(7);
        for (record, source) in instance.records.iter().zip(&records) {
            let t = sparse_t(&record.c, &record.key.t1);
            let reference = public_t(&record.c, &record.key.t1);
            for rho in 0..K {
                let c: [i64; N] = std::array::from_fn(|j| i64::from(record.c.coeffs[j]));
                let t1: [i64; N] = std::array::from_fn(|j| i64::from(record.key.t1[rho][j]));
                let product = negacyclic_schoolbook(&c, &t1);
                for j in 0..N {
                    assert_eq!(i64::from(t[rho][j]), reference[rho][j]);
                    assert_eq!(i128::from(t[rho][j]), product[j] << D);
                    assert!(i64::from(t[rho][j]).abs() <= agg_witness::params::B_T);
                }
            }
            assert_eq!(*inverse_ntt_matrix(&record.key.a_hat), *record.key.a);
            let key = VerifierKey::from_bytes(&source.pk).unwrap();
            assert_eq!(key, VerifierKey::from_expanded(&record.key));
        }
        assert!(VerifierKey::from_bytes(&records[0].pk[1..]).is_none());
    }

    #[test]
    fn prepared_data_matches_the_prover_instance_with_and_without_cache() {
        let (instance, records) = small_instance(9);
        let inputs: Vec<(&[u8], &[u8], [u8; C_TILDE_BYTES])> = instance
            .records
            .iter()
            .zip(&records)
            .map(|(record, source)| (source.pk.as_slice(), source.msg.as_slice(), record.c_tilde))
            .collect();
        let from_instance = PublicData::from_instance(&instance);
        let mut cache = KeyCache::default();
        let uncached = PublicData::prepare(&inputs, None).unwrap();
        let first = PublicData::prepare(&inputs, Some(&mut cache)).unwrap();
        assert_eq!(cache.len(), 7, "5 own keys and 2 shared keys");
        let second = PublicData::prepare(&inputs, Some(&mut cache)).unwrap();
        assert_eq!(cache.len(), 7);
        for data in [&uncached, &first, &second] {
            assert_eq!(data.keys().len(), 7);
            assert_eq!(data.records(), from_instance.records());
            assert_eq!(data.geometry(), instance.geom);
            for i in 0..inputs.len() {
                assert_eq!(data.a(i), from_instance.a(i));
                assert_eq!(data.records()[i].mu, instance.records[i].mu);
            }
        }
        // The second batch reused the cached expansions.
        assert!(Arc::ptr_eq(&first.keys()[0], &second.keys()[0]));
        let mut bad = inputs.clone();
        bad[3].0 = &records[3].pk[..100];
        assert_eq!(
            PublicData::prepare(&bad, None).unwrap_err(),
            InstanceError::PublicKey(3)
        );
    }

    #[test]
    fn t_evaluation_is_the_multilinear_extension_of_the_table() {
        let (instance, _) = small_instance(13);
        let data = PublicData::from_instance(&instance);
        let table = data.t_table();
        assert_eq!(table, instance.t_table());
        let d_main = instance.geom.d_main() as usize;
        let mut rng = SplitMix64::new(21);
        let r = point(&mut rng, d_main);
        assert_eq!(data.t_evaluation(&r), evaluate_mle(&lift_table(&table), &r));
        for x in [0, 1023, 5 * 1024 + 300, 10 * 1024 + 7, 15 * 1024 + 900] {
            let expected = E::from_i64(table[x]);
            assert_eq!(
                data.t_evaluation(&boolean_point(x, d_main)),
                expected,
                "x {x}"
            );
        }
        // A~ at a Boolean row point is the row of A; at a random point it is affine in eq.
        let row = data.a_row_combination(4, &boolean_point(2, 2));
        for s in 0..L {
            for j in 0..N {
                assert_eq!(row[s][j], E::from_i64(i64::from(data.a(4)[2][s][j])));
            }
        }
        let r_row = point(&mut rng, 2);
        let weights = eq_table(&r_row);
        let mixed = data.a_row_combination(4, &r_row);
        let expected = (0..K).fold(E::zero(), |sum, rho| {
            sum + weights[rho].mul_i64(i64::from(data.a(4)[rho][3][77]))
        });
        assert_eq!(mixed[3][77], expected);
    }
}
