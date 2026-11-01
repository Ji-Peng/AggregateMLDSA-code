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

use std::collections::HashMap;
use std::sync::Arc;

use aerie_falcon::protocol::algebra::eq_table;
use agg_witness::instance::PublicInstance;
use agg_witness::params::{ParamSet, N};
use jolt_field::{Ring, Zero};
use rayon::prelude::*;

use crate::public_data::{eq_correlation, negacyclic_kernel_into};
use crate::OpeningField as E;

/// Variables of the coefficient index `j`.
const COEF_VARS: usize = 8;

/// The public data of a batch of a parameter set with polynomial lanes.
#[derive(Clone, Debug)]
pub struct LanePublicData {
    k: usize,
    l: usize,
    padded: usize,
    /// `A[rho][s]` of every distinct key, centered, flat `(rho, s, j)`.
    keys: Vec<Arc<Vec<i32>>>,
    /// The key of every live record.
    key_of: Vec<usize>,
    /// `T_i[rho][j]` of every live record, flat `(i, rho, j)`.
    t: Vec<i32>,
}

impl LanePublicData {
    /// The data of the prover's instance (keys shared by `Arc` identity).
    #[must_use]
    pub fn from_instance<P: ParamSet>(instance: &PublicInstance<P>) -> Self {
        let (k, l) = (P::K, P::L);
        let mut seen: HashMap<*const agg_witness::expand::ExpandedKey<P>, usize> = HashMap::new();
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
            .map(|&i| {
                let a = &instance.records[i].key.a;
                let mut flat = Vec::with_capacity(k * l * N);
                for rho in 0..k {
                    for s in 0..l {
                        flat.extend_from_slice(P::get_coeff(a, rho, s));
                    }
                }
                Arc::new(flat)
            })
            .collect();
        let mut t = Vec::with_capacity(instance.records.len() * k * N);
        for record in &instance.records {
            for rho in 0..k {
                t.extend_from_slice(&record.t[rho]);
            }
        }
        Self {
            k,
            l,
            padded: instance.geom.n_star,
            keys,
            key_of,
            t,
        }
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
        &key[(rho * self.l + s) * N..(rho * self.l + s + 1) * N]
    }

    /// `T_i[rho]` of live record `i`.
    #[must_use]
    pub fn t(&self, i: usize, rho: usize) -> &[i32] {
        let start = (i * self.k + rho) * N;
        &self.t[start..start + N]
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
                (0..k)
                    .map(|rho| {
                        let inner = self
                            .t(i, rho)
                            .iter()
                            .zip(&eq_coef)
                            .fold(E::zero(), |acc, (&value, &w)| acc + w.mul_i64(i64::from(value)));
                        eq_sig[i] * inner
                    })
                    .collect::<Vec<E>>()
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

    /// `A~_{s} = sum_rho gammas[rho] A[rho, s]` of key `key`, one polynomial per column.
    fn combined(&self, key: usize, gammas: &[E]) -> Vec<Vec<E>> {
        let a = &self.keys[key];
        (0..self.l)
            .map(|s| {
                (0..N)
                    .map(|j| {
                        (0..self.k).fold(E::zero(), |acc, rho| {
                            acc + gammas[rho].mul_i64(i64::from(a[(rho * self.l + s) * N + j]))
                        })
                    })
                    .collect()
            })
            .collect()
    }

    /// The ring weights `w_A^(s)` on the main domain `(j', i)` (`N* 256` cells each, zero on
    /// padded records) for the point `r2 = r^(2)` and the row coefficients `gammas`.
    ///
    /// # Panics
    /// If `gammas` does not have `k` entries or `r2` is not a main-domain point.
    #[must_use]
    pub fn ring_weight_tables(&self, r2: &[E], gammas: &[E]) -> Vec<Vec<E>> {
        assert_eq!(gammas.len(), self.k, "one coefficient per row");
        let (coef, sig) = self.split(r2);
        let eq_sig = eq_table(sig);
        // One transform per distinct key and column; records scale it by eq(r_sig, i).
        let kernels: Vec<Vec<Vec<E>>> = (0..self.keys.len())
            .into_par_iter()
            .map(|key| {
                let mut scratch = vec![E::zero(); N];
                self.combined(key, gammas)
                    .into_iter()
                    .map(|a| {
                        let mut out = vec![E::zero(); N];
                        negacyclic_kernel_into(&a, coef, &mut out, &mut scratch);
                        out
                    })
                    .collect()
            })
            .collect();
        let len = self.padded * N;
        (0..self.l)
            .map(|s| {
                let mut table = vec![E::zero(); len];
                table
                    .par_chunks_mut(N)
                    .take(self.records())
                    .enumerate()
                    .for_each(|(i, out)| {
                        let kernel = &kernels[self.key_of[i]][s];
                        for (slot, &value) in out.iter_mut().zip(kernel) {
                            *slot = eq_sig[i] * value;
                        }
                    });
                table
            })
            .collect()
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
        let per_key: Vec<Vec<E>> = self
            .keys
            .par_iter()
            .map(|a| {
                (0..l)
                    .map(|s| {
                        (0..k).fold(E::zero(), |acc, rho| {
                            let row = &a[(rho * l + s) * N..(rho * l + s + 1) * N];
                            let dot = row
                                .iter()
                                .zip(&lambda)
                                .fold(E::zero(), |d, (&entry, &w)| d + w.mul_i64(i64::from(entry)));
                            acc + gammas[rho] * dot
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

    #[test]
    fn ml_dsa_65_ring_weights_pull_eq_back_per_row() {
        ring_weights_pull_eq_back_per_row::<Set65>();
    }

    #[test]
    fn ml_dsa_87_ring_weights_pull_eq_back_per_row() {
        ring_weights_pull_eq_back_per_row::<Set87>();
    }
}
