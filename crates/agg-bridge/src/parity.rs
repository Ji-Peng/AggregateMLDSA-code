//! Fp parity tables, generic over [`Shape`].
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

/// Direct round-2 bridge tables in their final narrow (`i8` parity) and stacked balanced
/// (`i32` carry `K = kappa - lookup_shift`) storage layouts, with no intermediate `i64`
/// vectors or `split_lanes` passes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectParityTables {
    /// `y^(1)_t` on `D_1` (`i8` bits).
    pub y1: [Vec<i8>; W1_BITS],
    /// Stacked balanced level-1 carries (`8 * d1_len`, `K = kappa - lookup_shift`, padding
    /// `-lookup_shift`).
    pub level1_carries: Vec<i32>,
    /// `y^(2)_t` on `D_2` (`i8` bits).
    pub y2: Vec<Vec<i8>>,
    /// Stacked balanced level-2 carries (`8 * d2_len`).
    pub level2_carries: Vec<i32>,
    /// Public `Y` on `D_3` (`u8` bits).
    pub y: Vec<u8>,
    /// Stacked balanced level-3 carries (`d3_len`).
    pub level3_carries: Vec<i32>,
}

/// Sum one level's sparse matrix into its parity lanes and stacked carries. The six bit
/// planes are packed one lane per 12-bit field, so a row sum fits `u64` without a carry
/// between fields (`acc0`) and the sixth lane in `acc5`.
fn accumulate_level_packed(
    six_bits: &[u8],
    m: &SparseMatrix,
    blocks: usize,
    cols_per_block: usize,
    rows: usize,
    lookup_shift: i32,
    slots: usize,
) -> ([Vec<i8>; W1_BITS], Vec<i32>) {
    let out_len = blocks * rows;
    // Split each block into strips when blocks < 16 so all worker threads participate.
    let strips_per_block = (16usize / blocks.max(1)).clamp(1, 8);
    let strip_cols = cols_per_block.div_ceil(strips_per_block);
    let strip_sums: Vec<(Vec<u64>, Vec<u16>)> = (0..blocks * strips_per_block)
        .into_par_iter()
        .map(|task| {
            let beta = task / strips_per_block;
            let s = task % strips_per_block;
            let block_start = beta * cols_per_block;
            let block_end = (block_start + cols_per_block).min(six_bits.len());
            let local_len = block_end.saturating_sub(block_start);
            let col_start = (s * strip_cols).min(local_len);
            let col_end = ((s + 1) * strip_cols).min(local_len);
            let mut acc0 = vec![0u64; rows];
            let mut acc5 = vec![0u16; rows];
            if col_start < col_end {
                agg_simd::digits::accumulate_six_lanes_strip(
                    &six_bits[block_start..block_end],
                    m.entries.as_slice(),
                    m.k,
                    col_start,
                    col_end,
                    &mut acc0,
                    &mut acc5,
                );
            }
            (acc0, acc5)
        })
        .collect();

    let mut parities: [Vec<i8>; W1_BITS] = std::array::from_fn(|_| vec![0i8; out_len]);
    let mut carries = vec![-lookup_shift; slots * out_len];

    for beta in 0..blocks {
        let mut acc0 = vec![0u64; rows];
        let mut acc5 = vec![0u16; rows];
        for s in 0..strips_per_block {
            let (ref s0, ref s5) = strip_sums[beta * strips_per_block + s];
            for r in 0..rows {
                // Disjoint column strips sum to at most the checked row cap (2047 in
                // production), so no 12-bit field carries into the next one.
                acc0[r] += s0[r];
                acc5[r] += s5[r];
            }
        }
        let base = beta * rows;
        for r in 0..rows {
            let v0 = acc0[r];
            let sums = [
                (v0 & 0xFFF) as i32,
                ((v0 >> 12) & 0xFFF) as i32,
                ((v0 >> 24) & 0xFFF) as i32,
                ((v0 >> 36) & 0xFFF) as i32,
                ((v0 >> 48) & 0xFFF) as i32,
                i32::from(acc5[r]),
            ];
            for t in 0..W1_BITS {
                let s = sums[t];
                parities[t][base + r] = (s & 1) as i8;
                carries[t * out_len + base + r] = (s >> 1) - lookup_shift;
            }
        }
    }
    (parities, carries)
}

/// Compute the production bridge parities and stacked balanced carries directly from the
/// six bit columns without `i64` intermediates; returns `Ok(None)` if any input byte is
/// outside `{0, 1}` (tampered witness fallback).
pub fn compute_direct(
    shape: &Shape,
    lanes: &[&[i8]; W1_BITS],
    matrices: &BridgeMatrices,
    lookup_shift: i32,
    hint_slack_d2: Option<&[i64]>,
) -> Result<Option<DirectParityTables>, BridgeError> {
    if !shape.matches(matrices) {
        return Err(BridgeError::Shape);
    }
    if !shape.row_weights_ok(matrices) {
        return Err(BridgeError::RowWeight);
    }
    // Packed fields hold at most 4095; other shapes use the general reference path.
    if shape.lanes != W1_BITS || shape.levels[..2].iter().any(|l| l.weight_cap >= 4096) {
        return Ok(None);
    }
    let Some(six_bits) = agg_simd::digits::pack_six_bits(lanes) else {
        return Ok(None);
    };
    let domains = shape.domains(lanes[0].len());
    let l1 = shape.level(1);
    let l2 = shape.level(2);
    let l3 = shape.level(3);

    let (y1, level1_carries) = accumulate_level_packed(
        &six_bits,
        &matrices.m1,
        domains.blocks1,
        l1.cols,
        l1.rows,
        lookup_shift,
        8,
    );

    let y1_refs: [&[i8]; W1_BITS] = std::array::from_fn(|t| y1[t].as_slice());
    let six_bits_d1 = agg_simd::digits::pack_six_bits(&y1_refs).expect("y1 parities are bits");
    let (y2_arr, mut level2_carries) = accumulate_level_packed(
        &six_bits_d1,
        &matrices.m2,
        domains.segments,
        l2.used,
        l2.rows,
        lookup_shift,
        8,
    );
    if let Some(slack) = hint_slack_d2 {
        assert_eq!(slack.len(), domains.d2_len);
        let slot6 = &mut level2_carries[6 * domains.d2_len..7 * domains.d2_len];
        for (dst, &v) in slot6.iter_mut().zip(slack) {
            *dst = i32::try_from(v).expect("hint slack fits i32") - lookup_shift;
        }
    }

    let (rows2, rows3) = (l2.rows, l3.rows);
    let mut y = vec![0u8; domains.d3_len];
    let mut level3_carries = vec![-lookup_shift; domains.d3_len];
    for sigma in 0..domains.segments {
        // Inputs are bits, and the checked production row cap bounds each exact sum to
        // [0, 511]. Its carry is in [0, 255] before balancing by lookup_shift.
        let mut sums = vec![0i32; rows3];
        for (t, lane) in y2_arr.iter().enumerate() {
            for (rho2, &val) in lane[sigma * rows2..(sigma + 1) * rows2].iter().enumerate() {
                if val != 0 {
                    for &row in matrices.m3.column(t + shape.lanes * rho2) {
                        sums[row as usize] += i32::from(val);
                    }
                }
            }
        }
        let base = sigma * rows3;
        for (r, &s) in sums.iter().enumerate() {
            y[base + r] = (s & 1) as u8;
            level3_carries[base + r] = (s >> 1) - lookup_shift;
        }
    }

    Ok(Some(DirectParityTables {
        y1,
        level1_carries,
        y2: y2_arr.into(),
        level2_carries,
        y,
        level3_carries,
    }))
}

/// Direct round-2 bridge tables of a four-lane shape (ML-DSA-65 and ML-DSA-87): the lanes of
/// [`DirectParityTables`] as vectors and the carries stacked in the caller's slot counts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectLaneTables {
    /// `y^(1)_t` on `D_1` (`i8` bits), one per lane.
    pub y1: Vec<Vec<i8>>,
    /// Stacked balanced level-1 carries (`slots_1 * d1_len`, padding `-lookup_shift`).
    pub level1_carries: Vec<i32>,
    /// `y^(2)_t` on `D_2` (`i8` bits), one per lane.
    pub y2: Vec<Vec<i8>>,
    /// Stacked balanced level-2 carries (`slots_2 * d2_len`).
    pub level2_carries: Vec<i32>,
    /// Public `Y` on `D_3` (`u8` bits).
    pub y: Vec<u8>,
    /// Stacked balanced level-3 carries (`d3_len`).
    pub level3_carries: Vec<i32>,
}

/// Sum one level's sparse matrix into four parity lanes and stacked carries: the four bit
/// planes are packed one nibble per cell and summed in 16-bit fields of one `u64` per row.
fn accumulate_level_four(
    four_bits: &[u8],
    m: &SparseMatrix,
    blocks: usize,
    cols_per_block: usize,
    rows: usize,
    lookup_shift: i32,
    slots: usize,
) -> (Vec<Vec<i8>>, Vec<i32>) {
    let out_len = blocks * rows;
    // Split each block into strips when blocks < 16 so all worker threads participate.
    let strips_per_block = (16usize / blocks.max(1)).clamp(1, 8);
    let strip_cols = cols_per_block.div_ceil(strips_per_block);
    let strip_sums: Vec<Vec<u64>> = (0..blocks * strips_per_block)
        .into_par_iter()
        .map(|task| {
            let (beta, s) = (task / strips_per_block, task % strips_per_block);
            let block_start = beta * cols_per_block;
            let block_end = (block_start + cols_per_block).min(four_bits.len());
            let local_len = block_end.saturating_sub(block_start);
            let col_start = (s * strip_cols).min(local_len);
            let col_end = ((s + 1) * strip_cols).min(local_len);
            let mut acc = vec![0u64; rows];
            if col_start < col_end {
                agg_simd::digits::accumulate_four_lanes_strip(
                    &four_bits[block_start..block_end],
                    m.entries.as_slice(),
                    m.k,
                    col_start,
                    col_end,
                    &mut acc,
                );
            }
            acc
        })
        .collect();
    let mut parities = vec![vec![0i8; out_len]; 4];
    let mut carries = vec![-lookup_shift; slots * out_len];
    for beta in 0..blocks {
        let base = beta * rows;
        for r in 0..rows {
            // Disjoint column strips sum to at most the checked row cap, below 2^16 per field.
            let v: u64 = (0..strips_per_block).map(|s| strip_sums[beta * strips_per_block + s][r]).sum();
            for (t, lane) in parities.iter_mut().enumerate() {
                let sum = ((v >> (16 * t)) & 0xFFFF) as i32;
                lane[base + r] = (sum & 1) as i8;
                carries[t * out_len + base + r] = (sum >> 1) - lookup_shift;
            }
        }
    }
    (parities, carries)
}

/// [`compute_direct`] for a four-lane shape: the parities and the carries stacked in
/// `slots_1`, `slots_2` slots of `D_1`, `D_2` (the lookup stacks' slot counts) from the four
/// bridge inputs; `Ok(None)` if an input is not a bit or the shape is not a four-lane shape
/// whose level-1 and level-2 caps fit the packed fields.
pub fn compute_direct_four(
    shape: &Shape,
    lanes: &[&[i8]; 4],
    matrices: &BridgeMatrices,
    lookup_shift: i32,
    slots: [usize; 2],
) -> Result<Option<DirectLaneTables>, BridgeError> {
    if !shape.matches(matrices) {
        return Err(BridgeError::Shape);
    }
    if !shape.row_weights_ok(matrices) {
        return Err(BridgeError::RowWeight);
    }
    if shape.lanes != 4 || shape.levels[..2].iter().any(|l| l.weight_cap >= 1 << 16) || slots.iter().any(|&s| s < 4) {
        return Ok(None);
    }
    let Some(four_bits) = agg_simd::digits::pack_four_bits(lanes) else {
        return Ok(None);
    };
    compute_direct_four_packed(shape, &four_bits, matrices, lookup_shift, slots)
}

/// [`compute_direct_four`] on the bridge inputs already packed as nibbles (`four_bits[i] =
/// sum_t b_t[i] << t`, every lane a bit); `Ok(None)` if the shape is not a four-lane shape
/// whose level-1 and level-2 caps fit the packed fields.
pub fn compute_direct_four_packed(
    shape: &Shape,
    four_bits: &[u8],
    matrices: &BridgeMatrices,
    lookup_shift: i32,
    slots: [usize; 2],
) -> Result<Option<DirectLaneTables>, BridgeError> {
    if !shape.matches(matrices) {
        return Err(BridgeError::Shape);
    }
    if !shape.row_weights_ok(matrices) {
        return Err(BridgeError::RowWeight);
    }
    if shape.lanes != 4 || shape.levels[..2].iter().any(|l| l.weight_cap >= 1 << 16) || slots.iter().any(|&s| s < 4) {
        return Ok(None);
    }
    debug_assert!(four_bits.iter().all(|&b| b < 16), "nibbles");
    let domains = shape.domains(four_bits.len());
    let (l1, l2, l3) = (shape.level(1), shape.level(2), shape.level(3));
    let (y1, level1_carries) =
        accumulate_level_four(four_bits, &matrices.m1, domains.blocks1, l1.cols, l1.rows, lookup_shift, slots[0]);
    let y1_refs: [&[i8]; 4] = std::array::from_fn(|t| y1[t].as_slice());
    let four_d1 = agg_simd::digits::pack_four_bits(&y1_refs).expect("y1 parities are bits");
    let (y2, level2_carries) =
        accumulate_level_four(&four_d1, &matrices.m2, domains.segments, l2.used, l2.rows, lookup_shift, slots[1]);
    let (rows2, rows3) = (l2.rows, l3.rows);
    let mut y = vec![0u8; domains.d3_len];
    let mut level3_carries = vec![-lookup_shift; domains.d3_len];
    for sigma in 0..domains.segments {
        // Bit inputs and the checked level-3 row cap bound every sum to [0, W_3].
        let mut sums = vec![0i32; rows3];
        for (t, lane) in y2.iter().enumerate() {
            for (rho2, &val) in lane[sigma * rows2..(sigma + 1) * rows2].iter().enumerate() {
                if val != 0 {
                    for &row in matrices.m3.column(t + shape.lanes * rho2) {
                        sums[row as usize] += 1;
                    }
                }
            }
        }
        let base = sigma * rows3;
        for (r, &s) in sums.iter().enumerate() {
            y[base + r] = (s & 1) as u8;
            level3_carries[base + r] = (s >> 1) - lookup_shift;
        }
    }
    Ok(Some(DirectLaneTables { y1, level1_carries, y2, level2_carries, y, level3_carries }))
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
