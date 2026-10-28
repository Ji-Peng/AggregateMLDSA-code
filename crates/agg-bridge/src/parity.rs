//! Fp parity tables (`docs/bridge.md`), generic over [`Shape`].
//!
//! Per lane and level the prover proves over the integers
//! `sum_zeta M_t[rho, zeta] v^(t-1)(zeta) = y^(t)(rho) + 2 kappa^(t)(rho)` with
//! `v^(0) = b_t`, `v^(1) = y^(1)_t`, `v^(2) = (y^(2)_t)_t` and `y^(3) = Y`. The integer sum
//! `s` of a row determines both outputs: the parity `y = s mod 2` and the quotient column
//! `kappa = floor(s / 2)` (the carry). Both sides of the identity stay below
//! `1 + 2 floor(W_t / 2) <= W_t < p`, so the field identity is the integer one and reducing it
//! modulo 2 gives `y = XOR_zeta M_t[rho, zeta] v(zeta)` (lifting).
//!
//! Index conventions are those of [`crate::shape`]: level-1 block `beta = x / J_1`, `D_1` cell
//! `rho_1 + m_1 beta`; segment `sigma = D_1 cell / J_2^used`, `D_2` cell `rho_2 + m_2 sigma`;
//! `zeta_3 = t + lanes rho_2`, `D_3` cell `rho_3 + m_3 sigma`. The sums run in parallel over
//! lanes and blocks.

use agg_witness::bridge::{BridgeError, BridgeMatrices, BridgeWitness, SparseMatrix};
use agg_witness::params::W1_BITS;
use rayon::prelude::*;

use crate::shape::Shape;

/// `sum_zeta M_1[rho_1, zeta] v_t(zeta, beta)` on `D_1`, per lane, for any integer input
/// (`lanes[t]` has the same power-of-two length for every lane; positions beyond it are
/// zero padding).
#[must_use]
pub fn level1_sums(shape: &Shape, lanes: &[&[i8]], m1: &SparseMatrix) -> Vec<Vec<i64>> {
    let level = shape.level(1);
    let len = input_len(shape, lanes);
    let domains = shape.domains(len);
    lanes
        .par_iter()
        .map(|lane| {
            let mut out = vec![0_i64; domains.d1_len];
            out.par_chunks_mut(level.rows)
                .enumerate()
                .for_each(|(beta, chunk)| {
                    let start = beta * level.cols;
                    let end = (start + level.cols).min(lane.len());
                    for (zeta, &bit) in lane[start..end].iter().enumerate() {
                        if bit != 0 {
                            let value = i64::from(bit);
                            for &row in m1.column(zeta) {
                                chunk[row as usize] += value;
                            }
                        }
                    }
                });
            out
        })
        .collect()
}

/// `sum_zeta M_2[rho_2, zeta] v_t(zeta, sigma)` on `D_2` from `D_1` tables, per lane; only
/// the first `J_2^used` columns are read.
#[must_use]
pub fn level2_sums(shape: &Shape, y1: &[Vec<i64>], m2: &SparseMatrix) -> Vec<Vec<i64>> {
    let level = shape.level(2);
    assert_eq!(y1.len(), shape.lanes, "one D_1 table per lane");
    let d1_len = y1[0].len();
    let segments = d1_len.div_ceil(level.used);
    y1.par_iter()
        .map(|lane| {
            assert_eq!(lane.len(), d1_len, "lanes of equal length");
            let mut out = vec![0_i64; segments * level.rows];
            out.par_chunks_mut(level.rows)
                .enumerate()
                .for_each(|(sigma, chunk)| {
                    let start = sigma * level.used;
                    let end = (start + level.used).min(lane.len());
                    for (zeta, &value) in lane[start..end].iter().enumerate() {
                        if value != 0 {
                            for &row in m2.column(zeta) {
                                chunk[row as usize] += value;
                            }
                        }
                    }
                });
            out
        })
        .collect()
}

/// `sum_{t, rho_2} M_3[rho_3, t + lanes rho_2] v_t(rho_2, sigma)` on `D_3` from `D_2` tables.
#[must_use]
pub fn level3_sums(shape: &Shape, y2: &[Vec<i64>], m3: &SparseMatrix) -> Vec<i64> {
    let (rows2, rows3) = (shape.level(2).rows, shape.level(3).rows);
    assert_eq!(y2.len(), shape.lanes, "one D_2 table per lane");
    let segments = y2[0].len() / rows2;
    assert!(
        y2.iter().all(|lane| lane.len() == segments * rows2),
        "whole segments"
    );
    let mut out = vec![0_i64; segments * rows3];
    out.par_chunks_mut(rows3)
        .enumerate()
        .for_each(|(sigma, chunk)| {
            for (t, lane) in y2.iter().enumerate() {
                for (rho2, &value) in lane[sigma * rows2..(sigma + 1) * rows2].iter().enumerate() {
                    if value != 0 {
                        for &row in m3.column(t + shape.lanes * rho2) {
                            chunk[row as usize] += value;
                        }
                    }
                }
            }
        });
    out
}

/// The parity `s mod 2` and the quotient column `floor(s / 2)` of every sum.
#[must_use]
pub fn split(sums: &[i64]) -> (Vec<i64>, Vec<i64>) {
    (
        sums.iter().map(|s| s.rem_euclid(2)).collect(),
        sums.iter().map(|s| s.div_euclid(2)).collect(),
    )
}

fn split_lanes(sums: Vec<Vec<i64>>) -> (Vec<Vec<i64>>, Vec<Vec<i64>>) {
    sums.iter().map(|s| split(s)).unzip()
}

fn input_len(shape: &Shape, lanes: &[&[i8]]) -> usize {
    assert_eq!(lanes.len(), shape.lanes, "one input column per lane");
    let len = lanes[0].len();
    assert!(
        lanes.iter().all(|lane| lane.len() == len),
        "lanes of equal length"
    );
    len
}

/// The honest bridge witness of any shape: parities and quotient columns of the three levels
/// and the public `Y`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParityWitness {
    /// `y^(1)_t` on `D_1` (virtual in the PIOP).
    pub y1: Vec<Vec<i64>>,
    /// `kappa^(1)_t` on `D_1`.
    pub kappa1: Vec<Vec<i64>>,
    /// `y^(2)_t` on `D_2` (committed).
    pub y2: Vec<Vec<i64>>,
    /// `kappa^(2)_t` on `D_2`.
    pub kappa2: Vec<Vec<i64>>,
    /// `Y` on `D_3`: `Y_sigma[rho_3]` at `rho_3 + m_3 sigma` (public).
    pub y: Vec<i64>,
    /// `kappa^(3)` on `D_3`.
    pub kappa3: Vec<i64>,
}

impl ParityWitness {
    /// The `agg-witness` form of a six-lane (production) witness.
    pub fn into_witness(self) -> Result<BridgeWitness, BridgeError> {
        let lanes = |v: Vec<Vec<i64>>| -> Result<[Vec<i64>; W1_BITS], BridgeError> {
            v.try_into().map_err(|_| BridgeError::Shape)
        };
        Ok(BridgeWitness {
            y1: lanes(self.y1)?,
            kappa1: lanes(self.kappa1)?,
            y2: lanes(self.y2)?,
            kappa2: lanes(self.kappa2)?,
            y: self.y,
            kappa3: self.kappa3,
        })
    }

    /// `Y` as bits.
    #[must_use]
    pub fn y_bits(&self) -> Vec<u8> {
        self.y.iter().map(|&v| v as u8).collect()
    }
}

/// The honest parities and quotient columns of the inputs `lanes` (main-domain bit columns
/// in production) under `matrices`. Refuses matrices of another shape and, as the honest
/// prover MUST, a row weight above its cap.
pub fn compute(
    shape: &Shape,
    lanes: &[&[i8]],
    matrices: &BridgeMatrices,
) -> Result<ParityWitness, BridgeError> {
    if !shape.matches(matrices) {
        return Err(BridgeError::Shape);
    }
    if !shape.row_weights_ok(matrices) {
        return Err(BridgeError::RowWeight);
    }
    let (y1, kappa1) = split_lanes(level1_sums(shape, lanes, &matrices.m1));
    let (y2, kappa2) = split_lanes(level2_sums(shape, &y1, &matrices.m2));
    let (y, kappa3) = split(&level3_sums(shape, &y2, &matrices.m3));
    Ok(ParityWitness {
        y1,
        kappa1,
        y2,
        kappa2,
        y,
        kappa3,
    })
}

/// The first violated bridge condition found by [`check`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdentityError {
    /// A table has the wrong number of lanes or cells.
    Shape,
    /// Level `level`'s input or output at `(lane, cell)` is not a bit (level 0: the inputs).
    NotBoolean {
        /// 0 (the inputs), 1, 2 or 3.
        level: usize,
        /// The lane (0 for `Y`).
        lane: usize,
        /// The cell.
        cell: usize,
    },
    /// A carry at `(lane, cell)` of `level` is outside `[0, floor(W_t / 2)]`.
    CarryRange {
        /// 1, 2 or 3.
        level: usize,
        /// The lane (0 for level 3).
        lane: usize,
        /// The cell.
        cell: usize,
    },
    /// The integer level identity of `level` fails at `(lane, cell)`.
    Identity {
        /// 1, 2 or 3.
        level: usize,
        /// The lane (0 for level 3).
        lane: usize,
        /// The cell.
        cell: usize,
    },
}

/// Check the three level identities at every Boolean point, with Booleanity of the inputs and
/// of every parity and the carry ranges, by recomputing the integer sums from the witness's
/// own inputs (an independent check of any witness, honest or not).
pub fn check(
    shape: &Shape,
    lanes: &[&[i8]],
    matrices: &BridgeMatrices,
    w: &ParityWitness,
) -> Result<(), IdentityError> {
    if !shape.matches(matrices) || lanes.len() != shape.lanes {
        return Err(IdentityError::Shape);
    }
    let domains = shape.domains(input_len(shape, lanes));
    let shaped = |tables: &[Vec<i64>], len: usize| {
        tables.len() == shape.lanes && tables.iter().all(|t| t.len() == len)
    };
    if !shaped(&w.y1, domains.d1_len)
        || !shaped(&w.kappa1, domains.d1_len)
        || !shaped(&w.y2, domains.d2_len)
        || !shaped(&w.kappa2, domains.d2_len)
        || w.y.len() != domains.d3_len
        || w.kappa3.len() != domains.d3_len
    {
        return Err(IdentityError::Shape);
    }
    for (lane, column) in lanes.iter().enumerate() {
        if let Some(cell) = column.iter().position(|&b| !(0..=1).contains(&b)) {
            return Err(IdentityError::NotBoolean {
                level: 0,
                lane,
                cell,
            });
        }
    }
    let level3_sums = level3_sums(shape, &w.y2, &matrices.m3);
    // Per level: the recomputed sums, the claimed parities and the claimed carries.
    type Level<'a> = (Vec<Vec<i64>>, &'a [Vec<i64>], &'a [Vec<i64>]);
    let levels: [Level<'_>; 3] = [
        (level1_sums(shape, lanes, &matrices.m1), &w.y1, &w.kappa1),
        (level2_sums(shape, &w.y1, &matrices.m2), &w.y2, &w.kappa2),
        (
            vec![level3_sums],
            std::slice::from_ref(&w.y),
            std::slice::from_ref(&w.kappa3),
        ),
    ];
    for (index, (sums, parities, carries)) in levels.iter().enumerate() {
        let level = index + 1;
        let cap = shape.level(level).carry_max();
        for (lane, ((s, y), k)) in sums.iter().zip(*parities).zip(*carries).enumerate() {
            for (cell, ((&s, &y), &k)) in s.iter().zip(y).zip(k).enumerate() {
                if !(0..=1).contains(&y) {
                    return Err(IdentityError::NotBoolean { level, lane, cell });
                }
                if !(0..=cap).contains(&k) {
                    return Err(IdentityError::CarryRange { level, lane, cell });
                }
                if s != y + 2 * k {
                    return Err(IdentityError::Identity { level, lane, cell });
                }
            }
        }
    }
    Ok(())
}
