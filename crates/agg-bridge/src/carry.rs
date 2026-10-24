//! Carry-digit tables with their digit ranges (spec Sections 4.3, 9.2; docs/bridge.md
//! Section 4).
//!
//! A carry `kappa` of level `t` is committed as the digits of its level's layout,
//! `kappa = sum_k W_k Z_k + C` (4.1): production levels 1 and 2 use bases 16, 16, 4
//! (`C = 648`, range `[0, 1023]`), level 3 bases 16, 16 (`C = 136`, `[0, 255]`); the toy
//! shape bases (8, 8), (8, 4), (4, 4). Every digit lies in its balanced alphabet exactly when
//! the carry lies in `[0, floor(W_t / 2)]`, which is what the PIOP's leaf predicates prove.
//!
//! [`group_tables`] lays the committed round-2 groups out in manifest order:
//! `G_1 = kappa1_{t,k}` (lane-major), `G_2 = y2_t` then `kappa2_{t,k}`, `G_3 = kappa3_k`,
//! and the public `Y`.

use agg_witness::bridge::BridgeTables;

use crate::parity::ParityWitness;
use crate::shape::{LevelShape, Shape};

/// A carry outside its digit range (the honest prover never produces one).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CarryRangeError {
    /// The level (1, 2, 3).
    pub level: u64,
    /// The offending carry.
    pub value: i64,
}

impl std::fmt::Display for CarryRangeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "level-{} carry {} outside its digit range",
            self.level, self.value
        )
    }
}

impl std::error::Error for CarryRangeError {}

/// The digits of a carry, least significant first, if it is in range.
pub fn digits(level: &LevelShape, kappa: i64) -> Result<Vec<i64>, CarryRangeError> {
    if !(0..=level.carry_max()).contains(&kappa) {
        return Err(CarryRangeError {
            level: level.t,
            value: kappa,
        });
    }
    Ok(level.carry.decompose(kappa))
}

/// One `i8` column per digit of the carries `values` (each checked to be in range).
pub fn digit_columns(level: &LevelShape, values: &[i64]) -> Result<Vec<Vec<i8>>, CarryRangeError> {
    let mut columns = vec![Vec::with_capacity(values.len()); level.carry.len()];
    for &value in values {
        for (column, digit) in columns.iter_mut().zip(digits(level, value)?) {
            // In range, every digit is in its alphabet, at most 8 in magnitude.
            column.push(digit as i8);
        }
    }
    Ok(columns)
}

/// The carries recomposed by (4.1) from digit columns (any digit values).
#[must_use]
pub fn compose_columns(level: &LevelShape, columns: &[Vec<i8>]) -> Vec<i64> {
    assert_eq!(columns.len(), level.carry.len(), "one column per digit");
    let len = columns.first().map_or(0, Vec::len);
    let weights = level.carry.weights();
    let constant = level.carry.constant();
    (0..len)
        .map(|x| {
            columns
                .iter()
                .zip(&weights)
                .map(|(column, w)| w * i64::from(column[x]))
                .sum::<i64>()
                + constant
        })
        .collect()
}

/// The committed round-2 groups `G_1, G_2, G_3` and the public `Y` of a witness, in
/// manifest order (for the production shape exactly `BridgeTables::from_witness`).
pub fn group_tables(shape: &Shape, w: &ParityWitness) -> Result<BridgeTables, CarryRangeError> {
    let [l1, l2, l3] = &shape.levels;
    let mut g1 = Vec::with_capacity(shape.lanes * l1.carry.len());
    for kappa in &w.kappa1 {
        g1.extend(digit_columns(l1, kappa)?);
    }
    let mut g2: Vec<Vec<i8>> =
        w.y2.iter()
            .map(|lane| lane.iter().map(|&y| y as i8).collect())
            .collect();
    for kappa in &w.kappa2 {
        g2.extend(digit_columns(l2, kappa)?);
    }
    let g3 = digit_columns(l3, &w.kappa3)?;
    Ok(BridgeTables {
        g1,
        g2,
        g3,
        y: w.y_bits(),
    })
}
