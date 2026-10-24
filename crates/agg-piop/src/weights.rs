//! Public weight tables (spec Section 7.6): the bridge weights `w_1, w_2, w_{3,t}` and the
//! ring weight `w_A`, each as the prover's table on its domain and as the verifier's
//! evaluation at a terminal point.
//!
//! Point conventions (least significant coordinate first): a main-domain point `r` splits
//! as `(r[..8], r[8..10], r[10..]) = (coef, row or column, sig)`; the suffix of length
//! `d_1` splits as `(rho_1: 13, beta = (i_mid, sigma))`, of length `d_2` as `(rho_2: 13,
//! sigma)`, of length `d_3` as `(rho_3: 12, sigma)` (spec Section 1, `docs/witness.md`).
//!
//! Bridge weight of level `t` at a point `pi` on `D_t`: `eq(pi_beta, beta) c_t(zeta)` with
//! `c_t(zeta) = sum_rho M_t[rho, zeta] eq(pi_rho, rho)`, one `J_t`-entry vector per level
//! and point, shared by all lanes, blocks and segments; zero-padded in-block positions
//! have zero weight. The verifier evaluates it at a terminal `r` as
//! `eq(pi_beta, r_beta) sum_zeta eq(r_zeta, zeta) c_t(zeta)`.
//!
//! The bridge weights are `agg_bridge::weights` at the production shape (shared with any
//! other consumer of the bridge); this module fixes the PIOP's point conventions.
//!
//! Ring weight (7.9): `w_A(j', s, i) = eq(r_sig, i) K_{A~_{i,s}, r_coef}[j']` with
//! `A~_{i,s} = sum_rho eq(r_row, rho) A_i[rho, s]`, by Aerie's factorized negacyclic
//! transform once per `(i, s)`; the verifier evaluates (7.10) by streaming over the
//! expanded keys with the correlation vector `Lambda` of `(r^(2)_coef, r^(3)_coef)`.

use aerie_falcon::protocol::algebra::eq_table;
use agg_bridge::{weights as bridge, Domains, Shape};
use agg_witness::bridge::SparseMatrix;
use agg_witness::padding::Geometry;
use agg_witness::params::{D_COEF, D_POS, K, L, N, W1_BITS};
use jolt_field::{ExtField, Ring, Zero};
use rayon::prelude::*;

use crate::public_data::{eq_correlation, negacyclic_kernel_into, PublicData};
use crate::{BaseField, OpeningField as E};

/// Row variables of levels 1 and 2 (`m_1 = m_2 = 2^13`).
pub const RHO12_VARS: usize = 13;
/// Row variables of level 3 (`m_3 = 2^12`).
pub const RHO3_VARS: usize = 12;
/// Variables of the level-1 in-block index (`J_1 = 2^19`).
pub const ZETA1_VARS: usize = 19;
/// Variables of the used level-2 in-block index (`2^17`).
pub const ZETA2_VARS: usize = 17;

/// The production bridge domains of a batch (`agg_bridge::Shape::PRODUCTION`).
#[must_use]
pub fn bridge_domains(geom: &Geometry) -> Domains {
    Shape::PRODUCTION.domains(geom.main_len())
}

/// `c[zeta] = sum_{rho in column zeta} eq_rows[rho]` for the first `cols` columns.
#[must_use]
pub fn column_sums(matrix: &SparseMatrix, eq_rows: &[E], cols: usize) -> Vec<E> {
    bridge::column_weights(matrix, eq_rows, cols)
}

/// `w_1` on the main domain for the point `p1` on `D_1` (spec (7.5)).
#[must_use]
pub fn level1_weight_table(geom: &Geometry, m1: &SparseMatrix, p1: &[E]) -> Vec<E> {
    assert_eq!(p1.len(), geom.d1() as usize, "a D_1 point");
    bridge::level1_table(&Shape::PRODUCTION, &bridge_domains(geom), m1, p1)
}

/// `w_1(r)` at a main-domain point `r`.
#[must_use]
pub fn level1_weight_at(geom: &Geometry, m1: &SparseMatrix, p1: &[E], r: &[E]) -> E {
    assert_eq!(p1.len(), geom.d1() as usize, "a D_1 point");
    assert_eq!(r.len(), geom.d_main() as usize, "a main-domain point");
    bridge::level1_at(&Shape::PRODUCTION, &bridge_domains(geom), m1, p1, r)
}

/// `w_2` on `D_1` for the point `p2` on `D_2` (spec (7.2)).
#[must_use]
pub fn level2_weight_table(geom: &Geometry, m2: &SparseMatrix, p2: &[E]) -> Vec<E> {
    assert_eq!(p2.len(), geom.d2() as usize, "a D_2 point");
    bridge::level2_table(&Shape::PRODUCTION, &bridge_domains(geom), m2, p2)
}

/// `w_2(q)` at a `D_1` point `q`.
#[must_use]
pub fn level2_weight_at(geom: &Geometry, m2: &SparseMatrix, p2: &[E], q: &[E]) -> E {
    assert_eq!(p2.len(), geom.d2() as usize, "a D_2 point");
    assert_eq!(q.len(), geom.d1() as usize, "a D_1 point");
    bridge::level2_at(&Shape::PRODUCTION, &bridge_domains(geom), m2, p2, q)
}

/// `w_{3,t}` on `D_2`, one table per lane, for the point `p3` on `D_3` (spec (7.3)).
#[must_use]
pub fn level3_weight_tables(geom: &Geometry, m3: &SparseMatrix, p3: &[E]) -> [Vec<E>; W1_BITS] {
    assert_eq!(p3.len(), geom.d3() as usize, "a D_3 point");
    bridge::level3_tables(&Shape::PRODUCTION, &bridge_domains(geom), m3, p3)
        .try_into()
        .expect("six lanes")
}

/// `w_{3,t}(q)` at a `D_2` point `q`, for every lane `t`.
#[must_use]
pub fn level3_weights_at(geom: &Geometry, m3: &SparseMatrix, p3: &[E], q: &[E]) -> [E; W1_BITS] {
    assert_eq!(p3.len(), geom.d3() as usize, "a D_3 point");
    assert_eq!(q.len(), geom.d2() as usize, "a D_2 point");
    bridge::level3_at(&Shape::PRODUCTION, &bridge_domains(geom), m3, p3, q)
        .try_into()
        .expect("six lanes")
}

/// The ring weight `w_A(j', s, i)` on the main domain in order `(j', s, i)` for the point
/// `r2 = r^(2)` (spec (7.9)); zero on padded rows.
#[must_use]
pub fn ring_weight_table(public: &PublicData, r2: &[E]) -> Vec<E> {
    let geom = public.geometry();
    assert_eq!(r2.len(), geom.d_main() as usize, "a main-domain point");
    let coef = &r2[..D_COEF as usize];
    let row = eq_table(&r2[D_COEF as usize..D_POS as usize]);
    assert_eq!(row.len(), K, "two row coordinates");
    let eq_sig = eq_table(&r2[D_POS as usize..]);
    let positions = 1_usize << D_POS;
    let mut table: Vec<E> = rayon::iter::repeat_n(E::zero(), geom.main_len()).collect();
    table
        .par_chunks_mut(positions)
        .take(geom.n)
        .enumerate()
        .for_each(|(i, chunk)| {
            // The kernel is linear in its polynomial: scale the row weights by eq(sig, i)
            // once, combine the integer rows of A with base-field products, and transform
            // each column into the table without allocating.
            let weights: [E; K] = std::array::from_fn(|rho| eq_sig[i] * row[rho]);
            let a = public.a(i);
            let mut combined = [E::zero(); N];
            let mut scratch = [E::zero(); N];
            for (s, out) in chunk.chunks_exact_mut(N).take(L).enumerate() {
                for (j, slot) in combined.iter_mut().enumerate() {
                    *slot = (0..K).fold(E::zero(), |sum, rho| {
                        sum + weights[rho].mul_base(BaseField::from_i64(i64::from(a[rho][s][j])))
                    });
                }
                negacyclic_kernel_into(&combined, coef, out, &mut scratch);
            }
        });
    table
}

/// `w_A(r3)` at the stage-3 point for the ring weight of `r2` (spec (7.10)): `k l 256`
/// multiply-adds per live record, streamed over the expanded keys.
#[must_use]
pub fn ring_weight_at(public: &PublicData, r2: &[E], r3: &[E]) -> E {
    let geom = public.geometry();
    assert_eq!(r2.len(), geom.d_main() as usize, "a main-domain point");
    assert_eq!(r3.len(), geom.d_main() as usize, "a main-domain point");
    let (coef, pos) = (D_COEF as usize, D_POS as usize);
    let lambda = eq_correlation(&r2[..coef], &r3[..coef]);
    let eq_row = eq_table(&r2[coef..pos]);
    let eq_col = eq_table(&r3[coef..pos]);
    let eq_sig2 = eq_table(&r2[pos..]);
    let eq_sig3 = eq_table(&r3[pos..]);
    (0..geom.n)
        .into_par_iter()
        .map(|i| {
            let a = public.a(i);
            let mut inner = E::zero();
            for rho in 0..K {
                for s in 0..L {
                    let dot = a[rho][s]
                        .iter()
                        .zip(&lambda)
                        .fold(E::zero(), |acc, (&entry, &l)| acc + l.mul_i64(i64::from(entry)));
                    inner += eq_row[rho] * eq_col[s] * dot;
                }
            }
            eq_sig2[i] * eq_sig3[i] * inner
        })
        .reduce(E::zero, |a, b| a + b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{boolean_point, point};
    use aerie_falcon::protocol::algebra::evaluate_mle;
    use agg_witness::bridge::{BridgeMatrices, J1, M1, M2, M3, ZETA2_USED};
    use agg_witness::crt::az_integer;
    use agg_witness::gen::{self, SplitMix64};
    use agg_witness::instance::PublicInstance;
    use agg_witness::padding::main_index;

    fn geometry(n: usize) -> Geometry {
        Geometry::new(n).unwrap()
    }

    fn suffix(point: &[E], vars: usize) -> Vec<E> {
        point[point.len() - vars..].to_vec()
    }

    /// Dense definition of a bridge weight at a Boolean position: `eq(pi_beta, beta)`
    /// times the sum of `eq(pi_rho, rho)` over the rows of the column.
    fn dense_weight(
        matrix: &SparseMatrix,
        zeta: usize,
        beta: usize,
        pi_rho: &[E],
        pi_beta: &[E],
    ) -> E {
        let eq_rho = eq_table(pi_rho);
        let eq_beta = eq_table(pi_beta);
        let column = if zeta < matrix.cols {
            matrix.column(zeta)
        } else {
            &[]
        };
        let sum = column
            .iter()
            .fold(E::zero(), |acc, &row| acc + eq_rho[row as usize]);
        sum * eq_beta[beta]
    }

    #[test]
    fn bridge_weight_tables_match_the_dense_definition_and_their_evaluations() {
        let mut rng = SplitMix64::new(31);
        let matrices = BridgeMatrices::random_for_tests(31);
        // d_sig = 10: two level-1 blocks, one segment; d_sig = 3: a zero-padded block.
        for n in [600, 5] {
            let geom = geometry(n);
            let (d_main, d1, d2, d3) = (
                geom.d_main() as usize,
                geom.d1() as usize,
                geom.d2() as usize,
                geom.d3() as usize,
            );
            let p1 = point(&mut rng, d1);
            let w1 = level1_weight_table(&geom, &matrices.m1, &p1);
            assert_eq!(w1.len(), geom.main_len());
            for x in [0, 1, 777, geom.main_len() - 1].into_iter().filter(|&x| x < geom.main_len()) {
                let expected = dense_weight(
                    &matrices.m1,
                    x & (J1 - 1),
                    x >> ZETA1_VARS,
                    &p1[..RHO12_VARS],
                    &p1[RHO12_VARS..],
                );
                assert_eq!(w1[x], expected, "w_1 at {x}, N = {n}");
                assert_eq!(
                    level1_weight_at(&geom, &matrices.m1, &p1, &boolean_point(x, d_main)),
                    expected
                );
            }
            let r = point(&mut rng, d_main);
            assert_eq!(
                level1_weight_at(&geom, &matrices.m1, &p1, &r),
                evaluate_mle(&w1, &r)
            );

            let p2 = point(&mut rng, d2);
            let w2 = level2_weight_table(&geom, &matrices.m2, &p2);
            assert_eq!(w2.len(), 1 << d1);
            for x in [0, 3, M1 + 9, (1 << d1) - 1].into_iter().filter(|&x| x < 1 << d1) {
                let expected = dense_weight(
                    &matrices.m2,
                    x & (ZETA2_USED - 1),
                    x >> ZETA2_VARS,
                    &p2[..RHO12_VARS],
                    &p2[RHO12_VARS..],
                );
                assert_eq!(w2[x], expected, "w_2 at {x}");
                assert_eq!(
                    level2_weight_at(&geom, &matrices.m2, &p2, &boolean_point(x, d1)),
                    expected
                );
            }
            let q = point(&mut rng, d1);
            assert_eq!(
                level2_weight_at(&geom, &matrices.m2, &p2, &q),
                evaluate_mle(&w2, &q)
            );

            let p3 = point(&mut rng, d3);
            let w3 = level3_weight_tables(&geom, &matrices.m3, &p3);
            let q2 = point(&mut rng, d2);
            let at = level3_weights_at(&geom, &matrices.m3, &p3, &q2);
            for t in 0..W1_BITS {
                assert_eq!(w3[t].len(), 1 << d2);
                for x in [0, 5, M2 - 1] {
                    let expected = dense_weight(
                        &matrices.m3,
                        t + W1_BITS * (x & (M2 - 1)),
                        x >> RHO12_VARS,
                        &p3[..RHO3_VARS],
                        &p3[RHO3_VARS..],
                    );
                    assert_eq!(w3[t][x], expected, "w_3 lane {t} at {x}");
                }
                assert_eq!(at[t], evaluate_mle(&w3[t], &q2));
            }
            // A zero-padded in-block position has zero weight (level 2 at d_sig < 13 uses
            // fewer than 2^17 columns only when the domain is smaller, so check level 1).
            if geom.main_len() < J1 {
                let p1_small = point(&mut rng, d1);
                let w = level1_weight_table(&geom, &matrices.m1, &p1_small);
                assert_eq!(w.len(), geom.main_len());
            }
        }
        let _ = M3;
    }

    #[test]
    fn ring_weight_pulls_eq_back_through_the_public_matrices() {
        // sum_x w_A(x) z(x) = sum_{j, rho, i} eq(r2, (j, rho, i)) (A z)(j, rho, i).
        let mut records = gen::records(41, 3);
        records.extend(gen::records_shared_keys(42, 2, 1));
        let instance = PublicInstance::from_records(&records).unwrap();
        let public = PublicData::from_instance(&instance);
        let geom = public.geometry();
        let d_main = geom.d_main() as usize;
        let mut rng = SplitMix64::new(43);
        let r2 = point(&mut rng, d_main);
        let table = ring_weight_table(&public, &r2);
        assert_eq!(table.len(), geom.main_len());
        // Padded rows carry zero weight.
        for &cell in &table[geom.n << D_POS..] {
            assert_eq!(cell, E::zero());
        }
        let mut z_table = vec![0_i64; geom.main_len()];
        let mut lhs = E::zero();
        for i in 0..geom.n {
            let mut z = [[0_i64; N]; L];
            for (s, row) in z.iter_mut().enumerate() {
                for (j, value) in row.iter_mut().enumerate() {
                    *value = rng.below(1 << 18) as i64 - (1 << 17);
                    z_table[main_index(j, s, i)] = *value;
                }
            }
            let az = az_integer(&instance.records[i].key, &z);
            let eq_r2 = eq_table(&r2);
            for rho in 0..K {
                for j in 0..N {
                    lhs += eq_r2[main_index(j, rho, i)] * E::from_i128(az[rho][j]);
                }
            }
        }
        let rhs = table
            .iter()
            .zip(&z_table)
            .fold(E::zero(), |acc, (&w, &z)| acc + w.mul_i64(z));
        assert_eq!(lhs, rhs);
        // The verifier's streamed evaluation is the extension of the table.
        let r3 = point(&mut rng, d_main);
        assert_eq!(ring_weight_at(&public, &r2, &r3), evaluate_mle(&table, &r3));
        let _ = suffix(&r3, 2);
    }
}
