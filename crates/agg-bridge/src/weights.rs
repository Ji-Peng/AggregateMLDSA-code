//! Public bridge weights (`docs/bridge.md`), generic over the shape and
//! the field (any `jolt_field::Ring`; the PIOP instantiates them with its opening field).
//!
//! For level `t` at a point `pi = (pi_rho, pi_beta)` of `D_t` the summation weight of the
//! level-`t` identity is `w_t(beta, zeta) = eq(pi_beta, beta) c_t(zeta)` with
//! `c_t(zeta) = sum_{rho in S_zeta} eq(pi_rho, rho)`: one `J_t^used`-entry vector per level and
//! point, shared by every lane, block and segment. The prover spreads it over the summed
//! domain ([`level1_table`], [`level2_table`], [`level3_tables`]); the verifier evaluates its
//! multilinear extension at a terminal point `r = (r_zeta, r_beta)` as
//! `eq(pi_beta, r_beta) sum_zeta eq(r_zeta, zeta) c_t(zeta)` ([`level1_at`], [`level2_at`],
//! [`level3_at`]): one equality product over the block variables times a sparse contraction
//! inside one block.
//!
//! Point conventions: least significant coordinate first, `eq_table(p)[i] = eq(p, bits(i))`
//! (Aerie's `eq_table`). Level 1 sums over the input (main) domain `(zeta_1, beta_1)`, level 2
//! over `D_1 = (zeta_2, sigma)`, level 3 over `D_2 = (rho_2, sigma)` with one table per lane
//! (`zeta_3 = t + lanes rho_2` is not a bit split). Zero-padded in-block positions (beyond
//! `J_t^used`, or beyond a domain smaller than one block) have zero weight.

use agg_witness::bridge::SparseMatrix;
use jolt_field::Ring;
use rayon::prelude::*;

use crate::shape::{Domains, Shape};

/// Below this many cells the serial paths are used.
const MIN_PARALLEL: usize = 1 << 13;

/// `eq(point, bits(i))` for every `i < 2^|point|`, coordinate `j` on index bit `j`.
#[must_use]
pub fn eq_table<F: Ring>(point: &[F]) -> Vec<F> {
    let mut table = vec![F::one()];
    for &coordinate in point {
        let length = table.len();
        let mut next = vec![F::zero(); 2 * length];
        let (low, high) = next.split_at_mut(length);
        let fill = |((low, high), &weight): ((&mut F, &mut F), &F)| {
            let scaled = weight * coordinate;
            *low = weight - scaled;
            *high = scaled;
        };
        if 2 * length >= MIN_PARALLEL {
            low.par_iter_mut()
                .zip(high.par_iter_mut())
                .zip(&table)
                .for_each(fill);
        } else {
            low.iter_mut()
                .zip(high.iter_mut())
                .zip(&table)
                .for_each(fill);
        }
        table = next;
    }
    table
}

/// `eq(a, b) = prod_j (a_j b_j + (1 - a_j)(1 - b_j))`.
#[must_use]
pub fn eq_evaluation<F: Ring>(a: &[F], b: &[F]) -> F {
    assert_eq!(a.len(), b.len(), "points of equal length");
    a.iter().zip(b).fold(F::one(), |acc, (&x, &y)| {
        acc * (x * y + (F::one() - x) * (F::one() - y))
    })
}

/// `c[zeta] = sum_{rho in column zeta} eq_rows[rho]` for the first `cols` columns
/// (`k_t` additions per column).
#[must_use]
pub fn column_weights<F: Ring>(matrix: &SparseMatrix, eq_rows: &[F], cols: usize) -> Vec<F> {
    assert_eq!(eq_rows.len(), matrix.rows, "one weight per row");
    assert!(cols <= matrix.cols, "columns within the matrix");
    let column = |zeta: usize| {
        matrix
            .column(zeta)
            .iter()
            .fold(F::zero(), |acc, &row| acc + eq_rows[row as usize])
    };
    if cols >= MIN_PARALLEL {
        (0..cols).into_par_iter().map(column).collect()
    } else {
        (0..cols).map(column).collect()
    }
}

/// `sum_zeta eq(point, zeta) c[zeta]` over the first `c.len()` points of `{0,1}^|point|`.
#[must_use]
pub fn contract<F: Ring>(point: &[F], c: &[F]) -> F {
    let weights = eq_table(point);
    assert!(c.len() <= weights.len(), "the vector fits the cube");
    if c.len() >= MIN_PARALLEL {
        weights
            .par_iter()
            .zip(c)
            .map(|(&w, &v)| w * v)
            .reduce(F::zero, |a, b| a + b)
    } else {
        weights
            .iter()
            .zip(c)
            .fold(F::zero(), |acc, (&w, &v)| acc + w * v)
    }
}

/// The table `x -> c[x mod 2^zeta_vars] eq_blocks[x >> zeta_vars]` of `len` cells (zero where
/// the in-block index is beyond `c`).
#[must_use]
pub fn spread<F: Ring>(c: &[F], zeta_vars: usize, eq_blocks: &[F], len: usize) -> Vec<F> {
    let mask = (1_usize << zeta_vars) - 1;
    let cell = |x: usize| {
        let zeta = x & mask;
        if zeta < c.len() {
            c[zeta] * eq_blocks[x >> zeta_vars]
        } else {
            F::zero()
        }
    };
    if len >= MIN_PARALLEL {
        (0..len).into_par_iter().map(cell).collect()
    } else {
        (0..len).map(cell).collect()
    }
}

/// In-block variables of level 1 (`min(input vars, log2 J_1)`).
#[must_use]
pub fn zeta1_vars(shape: &Shape, domains: &Domains) -> usize {
    domains
        .input_vars()
        .min(shape.level(1).cols.trailing_zeros() as usize)
}

/// In-block variables of level 2 (`min(D_1 vars, log2 J_2^used)`).
#[must_use]
pub fn zeta2_vars(shape: &Shape, domains: &Domains) -> usize {
    domains
        .d1_vars()
        .min(shape.level(2).used.trailing_zeros() as usize)
}

fn rho_vars(shape: &Shape, t: usize) -> usize {
    shape.level(t).rows_log2() as usize
}

/// `c_1` at the row part of a `D_1` point, over the used in-block positions.
#[must_use]
pub fn level1_columns<F: Ring>(
    shape: &Shape,
    domains: &Domains,
    m1: &SparseMatrix,
    p1: &[F],
) -> Vec<F> {
    assert_eq!(p1.len(), domains.d1_vars(), "a D_1 point");
    let zeta_vars = zeta1_vars(shape, domains);
    column_weights(m1, &eq_table(&p1[..rho_vars(shape, 1)]), 1 << zeta_vars)
}

/// `w_1` on the input (main) domain for the point `p1` of `D_1`.
#[must_use]
pub fn level1_table<F: Ring>(
    shape: &Shape,
    domains: &Domains,
    m1: &SparseMatrix,
    p1: &[F],
) -> Vec<F> {
    let (c1, zeta_vars, blocks) = level1_split(shape, domains, m1, p1);
    spread(&c1, zeta_vars, &blocks, domains.input_len)
}

/// The factors of [`level1_table`]: the in-block vector `c_1`, the in-block variable count
/// and the block weights, with `w_1(x) = c_1[x mod 2^zeta_vars] blocks[x >> zeta_vars]`
/// (zero where the in-block index is beyond `c_1`).
#[must_use]
pub fn level1_split<F: Ring>(
    shape: &Shape,
    domains: &Domains,
    m1: &SparseMatrix,
    p1: &[F],
) -> (Vec<F>, usize, Vec<F>) {
    let c1 = level1_columns(shape, domains, m1, p1);
    let blocks = eq_table(&p1[rho_vars(shape, 1)..]);
    (c1, zeta1_vars(shape, domains), blocks)
}

/// `w_1(r)` at an input-domain point `r` (the verifier's terminal formula).
#[must_use]
pub fn level1_at<F: Ring>(
    shape: &Shape,
    domains: &Domains,
    m1: &SparseMatrix,
    p1: &[F],
    r: &[F],
) -> F {
    assert_eq!(r.len(), domains.input_vars(), "an input-domain point");
    let zeta_vars = zeta1_vars(shape, domains);
    let c1 = level1_columns(shape, domains, m1, p1);
    eq_evaluation(&p1[rho_vars(shape, 1)..], &r[zeta_vars..]) * contract(&r[..zeta_vars], &c1)
}

/// `c_2` at the row part of a `D_2` point, over the used in-block positions of `D_1`.
#[must_use]
pub fn level2_columns<F: Ring>(
    shape: &Shape,
    domains: &Domains,
    m2: &SparseMatrix,
    p2: &[F],
) -> Vec<F> {
    assert_eq!(p2.len(), domains.d2_vars(), "a D_2 point");
    let cols = domains.d1_len.min(shape.level(2).used);
    column_weights(m2, &eq_table(&p2[..rho_vars(shape, 2)]), cols)
}

/// `w_2` on `D_1` for the point `p2` of `D_2`.
#[must_use]
pub fn level2_table<F: Ring>(
    shape: &Shape,
    domains: &Domains,
    m2: &SparseMatrix,
    p2: &[F],
) -> Vec<F> {
    let c2 = level2_columns(shape, domains, m2, p2);
    let blocks = eq_table(&p2[rho_vars(shape, 2)..]);
    spread(&c2, zeta2_vars(shape, domains), &blocks, domains.d1_len)
}

/// `w_2(q)` at a `D_1` point `q`.
#[must_use]
pub fn level2_at<F: Ring>(
    shape: &Shape,
    domains: &Domains,
    m2: &SparseMatrix,
    p2: &[F],
    q: &[F],
) -> F {
    assert_eq!(q.len(), domains.d1_vars(), "a D_1 point");
    let zeta_vars = zeta2_vars(shape, domains);
    let c2 = level2_columns(shape, domains, m2, p2);
    eq_evaluation(&p2[rho_vars(shape, 2)..], &q[zeta_vars..]) * contract(&q[..zeta_vars], &c2)
}

/// `c_3` at the row part of a `D_3` point (all `J_3` columns).
#[must_use]
pub fn level3_columns<F: Ring>(
    shape: &Shape,
    domains: &Domains,
    m3: &SparseMatrix,
    p3: &[F],
) -> Vec<F> {
    assert_eq!(p3.len(), domains.d3_vars(), "a D_3 point");
    column_weights(
        m3,
        &eq_table(&p3[..rho_vars(shape, 3)]),
        shape.level(3).used,
    )
}

/// `w_{3,t}` on `D_2`, one table per lane, for the point `p3` of `D_3`.
#[must_use]
pub fn level3_tables<F: Ring>(
    shape: &Shape,
    domains: &Domains,
    m3: &SparseMatrix,
    p3: &[F],
) -> Vec<Vec<F>> {
    let c3 = level3_columns(shape, domains, m3, p3);
    let sigma = eq_table(&p3[rho_vars(shape, 3)..]);
    let rows2 = shape.level(2).rows;
    let rho2_vars = rho_vars(shape, 2);
    (0..shape.lanes)
        .map(|t| {
            let cell = |x: usize| c3[t + shape.lanes * (x & (rows2 - 1))] * sigma[x >> rho2_vars];
            if domains.d2_len >= MIN_PARALLEL {
                (0..domains.d2_len).into_par_iter().map(cell).collect()
            } else {
                (0..domains.d2_len).map(cell).collect()
            }
        })
        .collect()
}

/// `w_{3,t}(q)` at a `D_2` point `q`, for every lane `t`.
#[must_use]
pub fn level3_at<F: Ring>(
    shape: &Shape,
    domains: &Domains,
    m3: &SparseMatrix,
    p3: &[F],
    q: &[F],
) -> Vec<F> {
    assert_eq!(q.len(), domains.d2_vars(), "a D_2 point");
    let c3 = level3_columns(shape, domains, m3, p3);
    let rho2_vars = rho_vars(shape, 2);
    let sigma = eq_evaluation(&p3[rho_vars(shape, 3)..], &q[rho2_vars..]);
    let eq_rho2 = eq_table(&q[..rho2_vars]);
    (0..shape.lanes)
        .map(|t| {
            let inner = eq_rho2
                .iter()
                .enumerate()
                .fold(F::zero(), |acc, (rho2, &w)| {
                    acc + w * c3[t + shape.lanes * rho2]
                });
            sigma * inner
        })
        .collect()
}
