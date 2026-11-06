//! Public data: the verifier's per-record precomputation and the negacyclic `eq` helpers of
//! the public weights (Aerie).
//!
//! Per distinct key the verifier expands `A_hat = ExpandA(rho)` and keeps only the centered
//! coefficient form `A = ctr_q(NTT^{-1}(A_hat))` (16 inverse NTTs, 11.8 KB per key, MAY be
//! cached across batches through [`KeyCache`]); per live record it forms `mu`,
//! `c = SampleInBall(c~)` and `T = 2^d (c * t1)` over the integers from the `tau` sparse
//! entries of `c` (no reduction, no NTT; `|T| < 2^29`, so `i32`). The public table
//! `T(j, rho, i)` is zero on padded rows; [`PublicData::t_evaluation`] streams it at a point.
//!
//! The ring weight pulls `eq(r_coef, .)` back through a polynomial `a`:
//! [`negacyclic_kernel`] computes `K_{a, r_coef}` by Aerie's factorized transform in
//! `O(n log n)`, and [`eq_correlation`] the verifier's 256-entry vector `Lambda` of Aerie
//! (the correlation of two `eq` vectors) in `O(n)`.

use std::collections::HashMap;
use std::sync::Arc;

use aerie_falcon::protocol::algebra::eq_table;
use agg_witness::expand::{expand_a, ExpandedKey, MatrixCoeff, MatrixNtt};
use agg_witness::instance::{InstanceError, PublicInstance};
use agg_witness::native::{ctr_q, mu, pk_decode, sample_in_ball, tr, Challenge};
use agg_witness::ntt::NttPrime;
use agg_witness::padding::{main_index, Geometry};
use agg_witness::params::{C_TILDE_BYTES, D, D_POS, K, L, N, TR_BYTES};
use jolt_field::{CanonicalEncoding, Fold, One, Ring, Zero};
use rayon::prelude::*;

use crate::BaseField;
use crate::OpeningField as E;

/// `T_i` of one record: `k` polynomials of 256 integer coefficients, `|T| < 2^29`.
pub type PublicT = [[i32; N]; K];

/// `A = ctr_q(NTT^{-1}(A_hat))` entrywise: 16 inverse NTTs.
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
/// `c` (`k tau 256` signed additions).
///
/// Each nonzero `c_p = +-1` adds `+-X^p t1`: coefficient `m < N - p` of `t1` lands at
/// `p + m`, the others wrap to `p + m - N` with the opposite sign, so the shift-add is two
/// contiguous slice updates split at `N - p` (branch-free inner loops that the compiler
/// vectorizes with the baseline SSE2 lanes). Before the shift `|c * t1| <= tau (2^10 - 1)
/// < 2^16`, so no `i32` overflows; after it `|T| < 2^29`.
#[must_use]
pub fn sparse_t(c: &Challenge, t1: &[[u16; N]; K]) -> PublicT {
    let mut out = sparse_ct(c, t1);
    for coeff in out.as_flattened_mut() {
        *coeff <<= D;
    }
    out
}

/// `c * t1` over `Z[X]/(X^256 + 1)` (no factor `2^d`), `|c * t1| <= tau (2^10 - 1) < 2^16`;
/// see [`sparse_t`].
#[must_use]
pub fn sparse_ct(c: &Challenge, t1: &[[u16; N]; K]) -> PublicT {
    let mut out = [[0_i32; N]; K];
    for (row, t1_row) in out.iter_mut().zip(t1) {
        let t: [i32; N] = std::array::from_fn(|m| i32::from(t1_row[m]));
        for &position in &c.positions {
            let p = usize::from(position);
            let (straight_src, wrapped_src) = t.split_at(N - p);
            let (wrapped_dst, straight_dst) = row.split_at_mut(p);
            debug_assert!(c.coeffs[p] == 1 || c.coeffs[p] == -1);
            if c.coeffs[p] > 0 {
                add_assign(straight_dst, straight_src);
                sub_assign(wrapped_dst, wrapped_src);
            } else {
                sub_assign(straight_dst, straight_src);
                add_assign(wrapped_dst, wrapped_src);
            }
        }
    }
    out
}

#[inline(always)]
fn add_assign(dst: &mut [i32], src: &[i32]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d += s;
    }
}

#[inline(always)]
fn sub_assign(dst: &mut [i32], src: &[i32]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d -= s;
    }
}

/// A vector over `E = F_{p^2}` split into the canonical coordinates of its two base
/// coefficients, for exact inner products with small signed integer vectors
/// ([`SplitWeights::dot`]). Multiplying `e = (e_0, e_1)` by an integer `v` is coordinatewise
/// (`v e = (v e_0, v e_1)`, whatever the `Ext2` non-residue), so
/// `sum_j v_j e_j = (sum_j v_j e_{j,0} mod p, sum_j v_j e_{j,1} mod p)`: each coordinate sum is
/// accumulated exactly over the integers and reduced once, instead of one `Ext2` product and
/// its reductions per term.
#[derive(Clone, Debug)]
pub struct SplitWeights {
    /// `[lo32(e_0), hi32(e_0), lo32(e_1), hi32(e_1)]` per weight.
    halves: Vec<[u32; 4]>,
    /// `[e_0, e_1]` per weight.
    coords: Vec<[u64; 2]>,
    /// `sum_j e_{j,0}` and `sum_j e_{j,1}` over the integers.
    sums: [u128; 2],
}

impl SplitWeights {
    /// Splits `weights` into canonical coordinates.
    #[must_use]
    pub fn new(weights: &[E]) -> Self {
        let coords: Vec<[u64; 2]> = weights
            .iter()
            .map(|w| {
                w.coeffs
                    .map(|c| c.to_u64_checked().expect("a canonical fp64 coordinate"))
            })
            .collect();
        let halves = coords
            .iter()
            .map(|&[a, b]| [a as u32, (a >> 32) as u32, b as u32, (b >> 32) as u32])
            .collect();
        let sums = [0, 1].map(|k| coords.iter().map(|c| u128::from(c[k])).sum());
        Self {
            halves,
            coords,
            sums,
        }
    }

    /// Number of weights.
    #[must_use]
    pub fn len(&self) -> usize {
        self.coords.len()
    }

    /// True when empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }

    /// `sum_j values[j] weights[j]` in `E`, exact for any `i32` values; fast when every value
    /// lies in `[-2^BITS, 2^BITS)`. Panics unless `values` has one entry per weight.
    ///
    /// Fast path (`len <= 2^(31 - BITS)`): with the biased values `b_j = v_j + 2^BITS` in
    /// `[0, 2^(BITS+1))` and the 32-bit halves `h` of the coordinates, every product `h b_j`
    /// is below `2^(BITS+33)` and `len` of them below `2^64`, so four `u64` lanes
    /// (`u32 x u32 -> u64` products, which the baseline SSE2 `pmuludq` vectorizes) hold
    /// `sum_j h b_j` exactly; then `sum_j e_j v_j = lo + 2^32 hi - 2^BITS sum_j e_j` in `i128`
    /// (`< 2^97`). The callers: `A` (`|A| <= (q - 1) / 2 < 2^22`, `BITS = 22`, 256 terms
    /// `<= 2^9`) and `c t1` (`|c t1| <= tau (2^10 - 1) < 2^16`, `BITS = 16`, 1,024 terms
    /// `<= 2^15`). If any value is out of range (a corrupted cache entry, say) the exact
    /// `i128` path is taken: terms below `2^95`, at most `2^31` of them below `2^126`.
    #[must_use]
    #[inline]
    pub fn dot<const BITS: u32>(&self, values: &[i32]) -> E {
        self.dot_lanes::<BITS>(values)
            .unwrap_or_else(|| self.dot_wide(values))
    }

    /// The fast path of [`SplitWeights::dot`]: `None` when a value is outside
    /// `[-2^BITS, 2^BITS)` or there are more than `2^(31 - BITS)` values.
    #[must_use]
    #[inline]
    pub fn dot_lanes<const BITS: u32>(&self, values: &[i32]) -> Option<E> {
        assert_eq!(values.len(), self.len(), "one value per weight");
        if BITS > 30 || values.len() > 1 << (31 - BITS) {
            return None;
        }
        let bias = 1_u32 << BITS;
        let (mut acc, mut seen) = ([0_u64; 4], 0_u32);
        for (h, &v) in self.halves.iter().zip(values) {
            let b = (v as u32).wrapping_add(bias);
            seen |= b;
            let b = u64::from(b);
            for k in 0..4 {
                acc[k] = acc[k].wrapping_add(u64::from(h[k]) * b);
            }
        }
        if seen >> (BITS + 1) != 0 {
            return None;
        }
        let [c0, c1] = [0, 1].map(|k| {
            i128::from(acc[2 * k]) + (i128::from(acc[2 * k + 1]) << 32)
                - ((self.sums[k] as i128) << BITS)
        });
        Some(E::new(BaseField::from_i128(c0), BaseField::from_i128(c1)))
    }

    /// `sum_j values[j] weights[j]` accumulated per coordinate in `i128`, exact for any `i32`
    /// values (terms below `2^95`; at most `2^31` of them below `2^126`).
    #[must_use]
    pub fn dot_wide(&self, values: &[i32]) -> E {
        assert_eq!(values.len(), self.len(), "one value per weight");
        let (mut acc0, mut acc1) = (0_i128, 0_i128);
        for (&[w0, w1], &v) in self.coords.iter().zip(values) {
            let v = i128::from(v);
            acc0 += i128::from(w0) * v;
            acc1 += i128::from(w1) * v;
        }
        E::new(BaseField::from_i128(acc0), BaseField::from_i128(acc1))
    }
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

/// Expanded keys kept across batches (the "cached" verifier).
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
}

/// The public instance as the verifier precomputes it.
#[derive(Clone, Debug)]
pub struct PublicData {
    geom: Geometry,
    keys: Vec<Arc<VerifierKey>>,
    records: Vec<PublicRecordData>,
    /// `T_i = 2^d (c_i * t1_i)` of every live record, in one buffer (record order).
    t: Vec<PublicT>,
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
        let (records, t): (Vec<PublicRecordData>, Vec<PublicT>) = inputs
            .par_iter()
            .map(|&(pk, msg, c_tilde)| {
                let key = index[pk];
                let c = sample_in_ball(&c_tilde);
                let record = PublicRecordData {
                    key,
                    mu: mu(&keys[key].tr, msg, &[]),
                    c_tilde,
                };
                (record, sparse_t(&c, &keys[key].t1))
            })
            .unzip();
        Ok(Self {
            geom,
            keys,
            records,
            t,
        })
    }

    /// The same data taken from the prover's instance (keys shared by `Arc` identity).
    /// The keys (one [`VerifierKey`] per distinct expanded key, in first-appearance order)
    /// and the records are built in parallel.
    #[must_use]
    pub fn from_instance(instance: &PublicInstance) -> Self {
        let mut seen: HashMap<*const ExpandedKey, usize> = HashMap::new();
        let mut firsts: Vec<usize> = Vec::new();
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
        let keys: Vec<Arc<VerifierKey>> = firsts
            .par_iter()
            .map(|&i| Arc::new(VerifierKey::from_expanded(&instance.records[i].key)))
            .collect();
        let records = instance
            .records
            .par_iter()
            .zip(key_of.par_iter())
            .map(|(record, &key)| PublicRecordData {
                key,
                mu: record.mu,
                c_tilde: record.c_tilde,
            })
            .collect();
        let t = instance.records.par_iter().map(Self::t_of).collect();
        Self {
            geom: instance.geom,
            keys,
            records,
            t,
        }
    }

    /// A record's `T = c t_1 2^d` (|T| < 2^29, already `i32` in the instance).
    fn t_of(record: &agg_witness::instance::PublicRecord) -> PublicT {
        *record.t
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
        &self.t[i]
    }

    /// The public table `T(j, rho, i)` on the main domain, zero on padded rows.
    #[must_use]
    pub fn t_table(&self) -> Vec<i64> {
        let mut out = vec![0_i64; self.geom.main_len()];
        for (i, t) in self.t.iter().enumerate() {
            for (rho, row) in t.iter().enumerate() {
                for (j, &value) in row.iter().enumerate() {
                    out[main_index(j, rho, i)] = i64::from(value);
                }
            }
        }
        out
    }

    /// `T(r) = sum_x eq(r, x) T(x)` at a main-domain point `r = (r_coef, r_row, r_sig)`,
    /// streamed over the live records. An instance's `T_i = 2^d (c_i t1_i)` is a multiple of
    /// `2^d`, so each record's `k 256`-term dot runs on `T_i / 2^d` (`|c_i t1_i| < 2^16`,
    /// [`SplitWeights::dot`]) and is scaled once; a record whose `T_i` is not (a synthetic
    /// statement) is summed exactly.
    #[must_use]
    pub fn t_evaluation(&self, point: &[E]) -> E {
        assert_eq!(
            point.len(),
            self.geom.d_main() as usize,
            "main-domain point"
        );
        let (position, signature) = point.split_at(D_POS as usize);
        let eq_position = SplitWeights::new(&eq_table(position));
        let eq_signature = eq_table(signature);
        let (scale, low) = (E::from_u64(1 << D), (1_i32 << D) - 1);
        self.t
            .par_iter()
            .enumerate()
            .map(|(i, t)| {
                let mut ct = [0_i32; K * N];
                let mut bits = 0;
                for (x, &v) in ct.iter_mut().zip(t.iter().flatten()) {
                    *x = v >> D;
                    bits |= v;
                }
                if bits & low == 0 {
                    eq_signature[i] * scale * eq_position.dot::<16>(&ct)
                } else {
                    eq_signature[i] * eq_position.dot::<16>(t.as_flattened())
                }
            })
            .reduce(E::zero, |left, right| left + right)
    }

    /// `A~_{i,s} = sum_rho eq(r_row, rho) A_i[rho, s]` for every column `s`.
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

/// `w^(t) (X) = (f_t(0) + f_t(1) X) w^(t+1)(X^2)`: replaces `a` by `a * eq(r, .)` in
/// `K[X]/(X^D + 1)` for `D = a.len() = 2^|r|` by Aerie's recurrence, in place, with a
/// scratch buffer of the same length (no allocation). With `f_t(0) = 1 - r_t` each output
/// of a butterfly is one product: `f0 a0 + f1 rot = a0 + r (rot - a0)`,
/// `f1 a0 + f0 a1 = a1 + r (a0 - a1)`.
#[cfg(test)]
fn eq_negacyclic_product(a: &mut [E], r: &[E], scratch: &mut [E]) {
    let size = a.len();
    if size == 1 {
        return;
    }
    // `x + r (y - x)` is a fold by `r` (jolt-field's precomputed fold context).
    let context = E::precompute(r[0]);
    let half = size / 2;
    let (even, odd) = scratch[..size].split_at_mut(half);
    for k in 0..half {
        let (a0, a1) = (a[2 * k], a[2 * k + 1]);
        // rot(A_1) = Y A_1 mod (Y^{D/2} + 1).
        let rotated = if k == 0 { -a[size - 1] } else { a[2 * k - 1] };
        even[k] = E::fold_one(&context, a0, rotated);
        odd[k] = E::fold_one(&context, a1, a0);
    }
    let (low, high) = a.split_at_mut(half);
    eq_negacyclic_product(even, &r[1..], low);
    eq_negacyclic_product(odd, &r[1..], high);
    for k in 0..half {
        a[2 * k] = scratch[k];
        a[2 * k + 1] = scratch[half + k];
    }
}

/// [`negacyclic_kernel`] into `out` (`out.len() = a.len()`), with a scratch buffer of at
/// least that length (no allocation).
pub fn negacyclic_kernel_into(a: &[E], r_coef: &[E], out: &mut [E], scratch: &mut [E]) {
    NegacyclicKernel::new(r_coef).apply_into(a, out, scratch);

}


#[cfg(test)]
pub(crate) fn negacyclic_kernel_reference_into(a: &[E], r: &[E], out: &mut [E], scratch: &mut [E]) {
    out[0] = a[0];
    for j in 1..a.len() { out[j] = -a[a.len() - j]; }
    eq_negacyclic_product(out, r, scratch);
}

/// A fixed challenge's fold contexts, shared by all
/// record/column transforms of one weight table.
pub(crate) struct NegacyclicKernel {
    contexts: Vec<<E as Fold>::Ctx>,
}

impl NegacyclicKernel {
    pub(crate) fn new(point: &[E]) -> Self {
        Self { contexts: point.iter().map(|&r| E::precompute(r)).collect() }
    }

    pub(crate) fn apply_into(&self, a: &[E], out: &mut [E], scratch: &mut [E]) {
        let n = 1_usize << self.contexts.len();
        assert_eq!(a.len(), n, "a has 2^|r| coefficients");
        assert!(out.len() == n && scratch.len() >= n, "buffers of the polynomial's length");
        let scratch = &mut scratch[..n];
        // Choose the initial buffer so the final level always writes into out.
        let mut source_out = self.contexts.len() % 2 == 0;
        let initial = if source_out { &mut *out } else { &mut *scratch };
        initial[0] = a[0];
        for j in 1..n { initial[j] = -a[n - j]; }
        for (level, context) in self.contexts.iter().enumerate() {
            let (src, dst) = if source_out {
                (&*out, &mut *scratch)
            } else {
                (&*scratch, &mut *out)
            };
            // Keep the already-decimated path in the low index bits. Each
            // butterfly streams contiguous paths, so the last level is already
            // in natural output order and needs no permutation or copy-back.
            let stride = 1_usize << level;
            for base in (0..n).step_by(2 * stride) {
                let (low, high) = dst[base..base + 2 * stride].split_at_mut(stride);
                for path in 0..stride {
                    let a0 = src[base + path];
                    let a1 = src[base + stride + path];
                    let rotated = if base == 0 { -src[n - stride + path] } else { src[base - stride + path] };
                    low[path] = E::fold_one(context, a0, rotated);
                    high[path] = E::fold_one(context, a1, a0);
                }
            }
            source_out = !source_out;
        }
        debug_assert!(source_out);
    }
}

/// `K_{a, r}[j'] = sum_{a'} a[a'] (-1)^{floor((a' + j') / n)} eq(r, (a' + j') mod n)`
/// (Aerie), so that `sum_j eq(r, j) (a * z)_j = sum_{j'} K[j'] z[j']` for
/// the negacyclic product. Computed as `a^dagger * w^(0)` with `a^dagger[0] = a[0]`,
/// `a^dagger[j] = -a[n - j]` by the factorized transform, `O(n log n)`.
#[must_use]
pub fn negacyclic_kernel(a: &[E], r_coef: &[E]) -> Vec<E> {
    let mut out = vec![E::zero(); a.len()];
    let mut scratch = vec![E::zero(); a.len()];
    negacyclic_kernel_into(a, r_coef, &mut out, &mut scratch);
    out
}

/// `Lambda_a = sum_j eq(right, j) (-1)^{floor((a + j) / n)} eq(left, (a + j) mod n)`
/// (Aerie) for every `a < n = 2^|left|`, by the carry trie in `O(n)`.
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
            assert_eq!(*inverse_ntt_matrix(&record.key.a_hat()), *record.key.a);
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
                assert_eq!(data.t(i), from_instance.t(i));
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
    #[test]
    fn iterative_negacyclic_matches_recursive_on_64_records() {
        let mut rng = SplitMix64::new(571);
        for record in 0..64 {
            // Exercise both buffer parities and all production-size levels.
            let bits = record % 9;
            let a: Vec<E> = (0..1 << bits).map(|_| crate::test_support::field(&mut rng)).collect();
            let r = point(&mut rng, bits);
            let mut expected = vec![E::zero(); a.len()];
            expected[0] = a[0];
            for j in 1..a.len() { expected[j] = -a[a.len() - j]; }
            let mut scratch = vec![E::zero(); a.len()];
            eq_negacyclic_product(&mut expected, &r, &mut scratch);
            let mut actual = vec![E::zero(); a.len()];
            NegacyclicKernel::new(&r).apply_into(&a, &mut actual, &mut scratch);
            assert_eq!(actual, expected, "record {record}");
        }
    }

}
