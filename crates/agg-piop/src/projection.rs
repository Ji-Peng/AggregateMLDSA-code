//! The `{0, 1}` projection of the ring quotient `v`.
//!
//! After the bridge seed the verifier draws `Psi in {0, 1}^(128 x 1024)` directly as the
//! answer of the site `proj/psi` (16 KiB of the PIOP duplex; row `o`, position `pos` is bit
//! `pos mod 8` of byte `128 o + pos / 8`). The prover commits, in round 2, the outputs
//! `k_{o,i} = sum_pos Psi_o(pos) v(pos, i)` on the projection domain `D_k` (cell
//! `o + 128 i`, `7 + d_sig` variables) through their digits
//! `k = K_0 + 2^16 K_1 + 2^32 K_2 + 2^35 K_3 + 2^34 + 2^31 + 2^15`
//! (`manifest::K16_DIGITS`): `K_0, K_1` are looked up in `T_16` (stack 4), `K_2, K_3 in
//! [-4, 3]` have a leaf and a link each. Stage 3 proves
//! `sum_x W(x) v(x) = c = sum_y eq(r', y) k(y)` with `r'` the `D_k` suffix of `r^(2)` and the
//! rank-one weight `W(pos, i) = P(pos) eq(r'_sig, i)`, `P(pos) = sum_o eq(r'_o, o) Psi_o(pos)`.
//!
//! Soundness (the projection bound): `k in [-2^37, 2^37)` for every row certifies
//! `|v|_c <= 2^38` except with probability `2^-128` over `Psi`, enough for the lifting
//! bound.
//!
//! ML-DSA-65 and ML-DSA-87 have `P = 256 k` positions (1,536 and 2,048), `Psi in {0, 1}^(128
//! x P)` (24 and 32 KiB, row `o` at bytes `P/8 o ..`), five digits
//! (`manifest::K39_DIGITS`, `k in [-2^39, 2^39)`) and `g` row blocks (1 and 2): block `b`
//! holds the positions of rows `rho in [b k/g, (b+1) k/g)`, and the outputs
//! `k_{o,b,i} = sum_{pos in block b} Psi_o(pos) v(pos, i)` sit at cell `o + 128 (b + g i)` of
//! `D_k = (o, b, i)`. The value `v(pos, i)` at `pos = j + 256 rho` is cell `j + 256 i` of
//! lane `rho` of `V_lo`, `V_hi`.

use aerie_falcon::protocol::algebra::{eq_evaluation, eq_table};
use agg_witness::params::{Set, POSITIONS};
use jolt_field::{Ring, Zero};
use rayon::prelude::*;

use crate::manifest::{
    CommitmentManifest, GroupRole, SetShape, K16_DIGITS, K39_DIGITS, PROJ_BYTES, PROJ_ROWS, V16_DIGITS,
};
use crate::source_tables::{Cells, SourceTableError};
use crate::OpeningField as E;

/// Bytes of one row of `Psi` (1,024 bits).
pub const ROW_BYTES: usize = POSITIONS / 8;

/// A pseudo-random `Psi` from a seed (tests and tools that build round-2 tables without a
/// transcript).
#[doc(hidden)]
#[must_use]
pub fn test_psi(seed: u64) -> Vec<u8> {
    let mut rng = agg_witness::gen::SplitMix64::new(seed);
    (0..PROJ_BYTES).map(|_| rng.next_u64() as u8).collect()
}

/// A pseudo-random `Psi` of parameter set `set` (`128 P / 8` bytes) from a seed; for
/// ML-DSA-44 it is [`test_psi`].
#[doc(hidden)]
#[must_use]
pub fn test_psi_for(set: Set, seed: u64) -> Vec<u8> {
    let mut rng = agg_witness::gen::SplitMix64::new(seed);
    (0..PROJ_ROWS * SetShape::of(set).positions / 8).map(|_| rng.next_u64() as u8).collect()
}

/// Bit `pos` of row `o` of a `Psi` with `row_bytes` bytes per row.
#[must_use]
pub fn psi_bit_in(psi: &[u8], row_bytes: usize, o: usize, pos: usize) -> bool {
    (psi[o * row_bytes + pos / 8] >> (pos % 8)) & 1 == 1
}

/// Bit `pos` of row `o` of `Psi`.
///
/// # Panics
/// If `psi` does not have [`PROJ_BYTES`] bytes.
#[must_use]
pub fn psi_bit(psi: &[u8], o: usize, pos: usize) -> bool {
    (psi[o * ROW_BYTES + pos / 8] >> (pos % 8)) & 1 == 1
}

/// The outputs `k_{o,i} = <Psi_o, v_i>` over the integers for the records whose values
/// `v_i` (1,024 integers each, position order `j + 256 rho`) `v` returns; record-major,
/// `PROJ_ROWS` outputs per record. Per record and byte column of `Psi`, the 256 subset sums
/// of its eight positions are tabulated once and read by every row.
///
/// # Panics
/// If `psi` does not have [`PROJ_BYTES`] bytes.
pub fn outputs(psi: &[u8], records: usize, v: impl Fn(usize, usize) -> i64 + Sync) -> Vec<i64> {
    assert_eq!(psi.len(), PROJ_BYTES, "Psi has 128 rows of 1,024 bits");
    agg_simd::mask::masked_row_sums(psi, records, PROJ_ROWS, ROW_BYTES, v)
}

/// The outputs `k_{o,b,i} = sum_{pos in block b} Psi_o(pos) v(i, pos)` of `Psi in {0, 1}^(128
/// x positions)` in `blocks` row blocks of `positions / blocks` consecutive positions, at
/// index `o + 128 (b + blocks i)`; one block gives [`outputs`]'s layout.
///
/// # Panics
/// If `psi` does not have `128 positions / 8` bytes or a block is not whole bytes.
pub fn block_outputs(
    psi: &[u8],
    positions: usize,
    blocks: usize,
    records: usize,
    v: impl Fn(usize, usize) -> i64 + Sync,
) -> Vec<i64> {
    let row_bytes = positions / 8;
    assert_eq!(psi.len(), PROJ_ROWS * row_bytes, "Psi has 128 rows of {positions} bits");
    if blocks == 1 {
        return agg_simd::mask::masked_row_sums(psi, records, PROJ_ROWS, row_bytes, v);
    }
    assert_eq!(row_bytes % blocks, 0, "a row block is whole bytes");
    let block_bytes = row_bytes / blocks;
    let per_block: Vec<Vec<i64>> = (0..blocks)
        .map(|b| {
            let sub: Vec<u8> = psi
                .chunks_exact(row_bytes)
                .flat_map(|row| row[b * block_bytes..(b + 1) * block_bytes].iter().copied())
                .collect();
            let start = 8 * b * block_bytes;
            agg_simd::mask::masked_row_sums(&sub, records, PROJ_ROWS, block_bytes, |i, pos| v(i, start + pos))
        })
        .collect();
    let mut out = vec![0_i64; blocks * records * PROJ_ROWS];
    out.par_chunks_mut(PROJ_ROWS).enumerate().for_each(|(cell, rows)| {
        let (i, b) = (cell / blocks, cell % blocks);
        rows.copy_from_slice(&per_block[b][i * PROJ_ROWS..(i + 1) * PROJ_ROWS]);
    });
    out
}

/// The committed digits of the projection outputs: `K_0 | K_1` stacked (the last lookup
/// stack, `2 * 2^d_k` cells) and `K_2 | K_3` (`2 * 2^d_k` cells; ML-DSA-65 and ML-DSA-87 add
/// `K_4`), at cell `o + 128 i` (`o + 128 (b + g i)` with row blocks). Under `HashField::Fp`
/// relation H-FIPS stacks the hint slack `u_h` as a third column and a padding slot
/// (`K_0 | K_1 | u_h | pad`, [`Self::attach_hint_slack`]), so the lane stays the lookup stack.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectionTables {
    /// `K_0` then `K_1` (stored balanced, looked up as `K + 2^15`), then `u_h` and a padding
    /// slot when [`Self::hint_slot`] is set.
    pub stacked: Vec<i32>,
    /// `K_2` then `K_3` (`[-4, 3]` each, leaf and link), then `K_4` (`[-2, 1]`) for five digits.
    pub top: Vec<i8>,
    /// True when `stacked` also holds the hint slack and a padding slot.
    pub hint_slot: bool,
}

impl ProjectionTables {
    /// The honest digits of the outputs `k` (record-major, `PROJ_ROWS` per record); the
    /// prover aborts when some `k` leaves `[-2^37, 2^37)` (a completeness
    /// condition, not a soundness one) and names the first such record.
    ///
    /// # Errors
    /// [`SourceTableError::ProjectionRange`] for an output outside the range.
    pub fn encode(manifest: &CommitmentManifest, k: &[i64]) -> Result<Self, SourceTableError> {
        let cells = manifest.group(GroupRole::Proj).logical_cells();
        assert_eq!(k.len(), cells, "one output per cell of D_k");
        let form = manifest.proj_digits();
        let (min, max) = form.range();
        // Outputs per record: 128 rows in each row block.
        let per_record = cells / manifest.padded_records();
        if let Some(y) = k.iter().position(|&value| !(min..=max).contains(&value)) {
            return Err(SourceTableError::ProjectionRange {
                record: y / per_record,
                row: y % per_record,
                value: k[y],
            });
        }
        if form.len() == K39_DIGITS.len() {
            return Ok(Self::encode_five(cells, k));
        }
        let mut stacked = vec![0_i32; 2 * cells];
        let mut top = vec![0_i8; 2 * cells];
        let (low, high) = stacked.split_at_mut(cells);
        let (k2, k3) = top.split_at_mut(cells);
        low.par_iter_mut()
            .zip(high.par_iter_mut())
            .zip(k2.par_iter_mut())
            .zip(k3.par_iter_mut())
            .zip(k.par_iter())
            .for_each(|((((k0, k1), k2), k3), &value)| {
                let mut digits = [0_i64; 4];
                K16_DIGITS.decompose_into(value, &mut digits);
                debug_assert!(K16_DIGITS.in_alphabets(&digits), "k in range has in-alphabet digits");
                *k0 = digits[0] as i32;
                *k1 = digits[1] as i32;
                *k2 = digits[2] as i8;
                *k3 = digits[3] as i8;
            });
        Ok(Self { stacked, top, hint_slot: false })
    }

    /// The five digits of in-range outputs ([`K39_DIGITS`]).
    fn encode_five(cells: usize, k: &[i64]) -> Self {
        let mut stacked = vec![0_i32; 2 * cells];
        let mut top = vec![0_i8; 3 * cells];
        let (low, high) = stacked.split_at_mut(cells);
        let (k2, rest) = top.split_at_mut(cells);
        let (k3, k4) = rest.split_at_mut(cells);
        low.par_iter_mut()
            .zip(high.par_iter_mut())
            .zip(k2.par_iter_mut())
            .zip(k3.par_iter_mut())
            .zip(k4.par_iter_mut())
            .zip(k.par_iter())
            .for_each(|(((((k0, k1), k2), k3), k4), &value)| {
                let mut digits = [0_i64; 5];
                K39_DIGITS.decompose_into(value, &mut digits);
                debug_assert!(K39_DIGITS.in_alphabets(&digits), "k in range has in-alphabet digits");
                *k0 = digits[0] as i32;
                *k1 = digits[1] as i32;
                *k2 = digits[2] as i8;
                *k3 = digits[3] as i8;
                *k4 = digits[4] as i8;
            });
        Self { stacked, top, hint_slot: false }
    }

    /// The honest tables of the committed `v` digits: `v = V_lo + 2^16 V_hi + 2^15` per
    /// cell, projected by `psi`; `v_lo`, `v_hi` hold the lanes of `V_lo`, `V_hi` (one for
    /// ML-DSA-44).
    ///
    /// # Errors
    /// As [`Self::encode`].
    ///
    /// # Panics
    /// If the lane counts or the length of `psi` are not the manifest's.
    pub fn build(
        manifest: &CommitmentManifest,
        v_lo: &[&[i32]],
        v_hi: &[&[i32]],
        psi: &[u8],
    ) -> Result<Self, SourceTableError> {
        let records = manifest.padded_records();
        let constant = V16_DIGITS.constant();
        let shape = manifest.shape();
        assert_eq!((v_lo.len(), v_hi.len()), (shape.lanes_k, shape.lanes_k), "one slice per lane");
        let k = if shape.lanes_k == 1 && shape.proj_blocks == 1 {
            let (v_lo, v_hi) = (v_lo[0], v_hi[0]);
            outputs(psi, records, |i, pos| {
                let x = i * POSITIONS + pos;
                i64::from(v_lo[x]) + (i64::from(v_hi[x]) << 16) + constant
            })
        } else {
            let lane_cells = manifest.lane_cells();
            block_outputs(psi, shape.positions, shape.proj_blocks, records, |i, pos| {
                let x = i * lane_cells + pos % lane_cells;
                let lane = pos / lane_cells;
                i64::from(v_lo[lane][x]) + (i64::from(v_hi[lane][x]) << 16) + constant
            })
        };
        Self::encode(manifest, &k)
    }

    /// Append H-FIPS's hint slack as the third slot of the stacked lane (stored `u_h - 2^15`,
    /// one `D_k` table, [`crate::source_tables::hint_slack_table`]) and a padding slot holding
    /// `-2^15` (looked-up value 0).
    ///
    /// # Panics
    /// If a slot is already attached or the table is not one `D_k` column.
    pub fn attach_hint_slack(&mut self, stored: &[i32]) {
        assert!(!self.hint_slot, "one hint slot");
        let cells = self.cells();
        assert_eq!(stored.len(), cells, "the hint slack is one D_k column");
        self.stacked.reserve_exact(2 * cells);
        self.stacked.extend_from_slice(stored);
        self.stacked.resize(4 * cells, -crate::manifest::LOOKUP_SHIFT);
        self.hint_slot = true;
    }

    /// Cells of each digit polynomial.
    fn cells(&self) -> usize {
        self.stacked.len() / if self.hint_slot { 4 } else { 2 }
    }

    /// The digit layout of the tables (four or five digits).
    #[must_use]
    pub fn digits(&self) -> agg_witness::digits::DigitLayout {
        if self.top.len() == 2 * self.cells() { K16_DIGITS } else { K39_DIGITS }
    }

    /// The four (five) `G_k` polynomials in manifest order, then the hint slack when its slot
    /// is attached.
    #[must_use]
    pub fn columns(&self) -> Vec<Cells<'_>> {
        let cells = self.cells();
        vec![Cells::Wide(&self.stacked[..cells]), Cells::Wide(&self.stacked[cells..2 * cells])]
            .into_iter()
            .chain(self.top.chunks_exact(cells).map(Cells::Narrow))
            .chain(self.hint_slot.then(|| Cells::Wide(&self.stacked[2 * cells..3 * cells])))
            .collect()
    }

    /// The outputs recomposed from the digits (any cell values), as field elements.
    #[must_use]
    pub fn value(&self, y: usize) -> E {
        let cells = self.cells();
        let low = E::from_i64(i64::from(self.stacked[y])) + E::from_i64(i64::from(self.stacked[cells + y]) << 16);
        let form = self.digits();
        let top = self
            .top
            .chunks_exact(cells)
            .zip(&form.weights()[2..])
            .fold(E::zero(), |acc, (column, &weight)| acc + E::from_i64(i64::from(column[y]) * weight));
        low + top + E::from_i64(form.constant())
    }
}

/// `P(pos) = sum_o eq(r_o, o) Psi_o(pos)` for the 1,024 positions, `r_o` the seven row
/// coordinates.
#[must_use]
pub fn row_weights(psi: &[u8], r_o: &[E]) -> Vec<E> {
    assert_eq!(psi.len(), PROJ_BYTES, "Psi has 128 rows of 1,024 bits");
    assert_eq!(r_o.len(), PROJ_ROWS.trailing_zeros() as usize);
    let eq = eq_table(r_o);
    let mut p = vec![E::zero(); POSITIONS];
    for (o, &weight) in eq.iter().enumerate() {
        for (pos, slot) in p.iter_mut().enumerate() {
            if psi_bit(psi, o, pos) {
                *slot += weight;
            }
        }
    }
    p
}

/// The point `r'` of the projection identity: the `D_k` suffix of `r^(2)`, split as
/// `(r'_o, r'_sig)` (7 row coordinates, `d_sig` signature coordinates).
#[must_use]
pub fn split_point(manifest: &CommitmentManifest, r2: &[E]) -> (Vec<E>, Vec<E>) {
    let vars = manifest.group(GroupRole::Proj).logical_vars;
    let point = &r2[r2.len() - vars..];
    let rows = vars - manifest.d_sig();
    (point[..rows].to_vec(), point[rows..].to_vec())
}

/// The weight `W(pos, i) = P(pos) eq(r'_sig, i)` on the main domain (prover table).
#[must_use]
pub fn weight_table(manifest: &CommitmentManifest, psi: &[u8], r2: &[E]) -> Vec<E> {
    let (r_o, r_sig) = split_point(manifest, r2);
    let p = row_weights(psi, &r_o);
    let blocks = eq_table(&r_sig);
    agg_bridge::weights::spread(&p, agg_witness::params::D_POS as usize, &blocks, 1 << manifest.d_main())
}

/// [`weight_table`] as coordinate planes, for the plane-layout sum-check groups (the same
/// products, written in parallel without the row-major table).
#[must_use]
pub fn weight_planes(manifest: &CommitmentManifest, psi: &[u8], r2: &[E]) -> agg_simd::buf::KPlanes {
    let (r_o, r_sig) = split_point(manifest, r2);
    let p = row_weights(psi, &r_o);
    let blocks = eq_table(&r_sig);
    let zeta_vars = agg_witness::params::D_POS as usize;
    let mask = (1_usize << zeta_vars) - 1;
    let rows = 1_usize << manifest.d_main();
    let buf = agg_simd::buf::fill_lane_planes(1, rows, |_, x| {
        let w = match p.get(x & mask) {
            Some(&c) => c * blocks[x >> zeta_vars],
            None => E::zero(),
        };
        (w.c0().to_canonical_u64(), w.c1().to_canonical_u64())
    });
    agg_simd::buf::KPlanes { buf, rows }
}

/// `W(r^(3))`: `P~(r^(3)_pos) eq(r'_sig, r^(3)_sig)` (verifier).
#[must_use]
pub fn weight_at(manifest: &CommitmentManifest, psi: &[u8], r2: &[E], r3: &[E]) -> E {
    let (r_o, r_sig) = split_point(manifest, r2);
    let p = row_weights(psi, &r_o);
    let (pos, sig) = r3.split_at(agg_witness::params::D_POS as usize);
    let eq_pos = eq_table(pos);
    let p_at = p.iter().zip(&eq_pos).fold(E::zero(), |acc, (&a, &b)| acc + a * b);
    p_at * eq_evaluation(&r_sig, sig)
}

/// `P(pos) = sum_o eq(r_o, o) Psi_o(pos)` for the `positions` positions of a `Psi` of
/// `positions / 8` bytes per row ([`row_weights`] for ML-DSA-44).
#[must_use]
pub fn row_weights_in(psi: &[u8], positions: usize, r_o: &[E]) -> Vec<E> {
    let row_bytes = positions / 8;
    assert_eq!(psi.len(), PROJ_ROWS * row_bytes, "Psi has 128 rows of {positions} bits");
    assert_eq!(r_o.len(), PROJ_ROWS.trailing_zeros() as usize);
    let eq = eq_table(r_o);
    (0..positions)
        .into_par_iter()
        .map(|pos| {
            eq.iter()
                .enumerate()
                .filter(|&(o, _)| psi_bit_in(psi, row_bytes, o, pos))
                .fold(E::zero(), |acc, (_, &w)| acc + w)
        })
        .collect()
}

/// The point `r'` of the projection identity split as `(r'_o, r'_b, r'_sig)`: 7 row
/// coordinates, `log2 g` row-block coordinates and `d_sig` signature coordinates.
#[must_use]
pub fn split_point_blocks(manifest: &CommitmentManifest, r2: &[E]) -> (Vec<E>, Vec<E>, Vec<E>) {
    let (rows, sig) = split_point(manifest, r2);
    let o_vars = PROJ_ROWS.trailing_zeros() as usize;
    (rows[..o_vars].to_vec(), rows[o_vars..].to_vec(), sig)
}

/// The row-block weight `eq(r'_b, b)` of every polynomial lane of `v`: lane `rho` holds the
/// rows of block `rho / (k / g)` (one block for ML-DSA-44 and ML-DSA-65, so the weight is one).
fn lane_block_weights(manifest: &CommitmentManifest, r_b: &[E]) -> Vec<E> {
    let shape = manifest.shape();
    let eq_b = eq_table(r_b);
    assert_eq!(eq_b.len(), shape.proj_blocks, "one coordinate per row-block bit");
    let rows_per_block = shape.k / shape.proj_blocks;
    let rows_per_lane = 1_usize << shape.d_vec;
    assert!(
        shape.proj_blocks == 1 || rows_per_lane <= rows_per_block,
        "a lane lies inside one row block"
    );
    (0..shape.lanes_k).map(|lane| eq_b[lane * rows_per_lane / rows_per_block]).collect()
}

/// The projection weights `W^(rho)(j, i) = P_rho(j) eq(r'_sig, i) eq(r'_b, blk(rho))` on the
/// main domain, one per polynomial lane `rho` of `v`, with `P_rho(j) = P(j + 256 rho)`, so
/// that `sum_rho sum_x W^(rho)(x) v^(rho)(x) = sum_y eq(r', y) k(y)` over `D_k = (o, b, i)`.
/// For one lane this is [`weight_table`].
#[must_use]
pub fn lane_weight_tables(manifest: &CommitmentManifest, psi: &[u8], r2: &[E]) -> Vec<Vec<E>> {
    let (r_o, r_b, r_sig) = split_point_blocks(manifest, r2);
    let shape = manifest.shape();
    let p = row_weights_in(psi, shape.positions, &r_o);
    let blocks = eq_table(&r_sig);
    let lane_cells = manifest.lane_cells();
    let lane_vars = lane_cells.trailing_zeros() as usize;
    lane_block_weights(manifest, &r_b)
        .into_iter()
        .enumerate()
        .map(|(lane, eq_b)| {
            let p_lane: Vec<E> = p[lane * lane_cells..(lane + 1) * lane_cells].iter().map(|&w| w * eq_b).collect();
            agg_bridge::weights::spread(&p_lane, lane_vars, &blocks, 1 << manifest.d_main())
        })
        .collect()
}

/// [`lane_weight_tables`] as their factors: `W^(rho)(j, i) = low_rho[j] blocks[i]` with
/// `low_rho = P_rho eq(r'_b, blk(rho))` over the lane cells and `blocks = eq(r'_sig, .)`
/// shared by every lane (prover; the stage-3 groups keep the weights factored).
#[must_use]
pub fn lane_weight_factors(manifest: &CommitmentManifest, psi: &[u8], r2: &[E]) -> (Vec<Vec<E>>, Vec<E>) {
    let (r_o, r_b, r_sig) = split_point_blocks(manifest, r2);
    let shape = manifest.shape();
    let p = row_weights_in(psi, shape.positions, &r_o);
    let lane_cells = manifest.lane_cells();
    let lows = lane_block_weights(manifest, &r_b)
        .into_iter()
        .enumerate()
        .map(|(lane, eq_b)| p[lane * lane_cells..(lane + 1) * lane_cells].iter().map(|&w| w * eq_b).collect())
        .collect();
    (lows, eq_table(&r_sig))
}

/// `W^(rho)(r^(3))` for every polynomial lane `rho` of `v` (verifier; [`lane_weight_tables`]).
#[must_use]
pub fn lane_weights_at(manifest: &CommitmentManifest, psi: &[u8], r2: &[E], r3: &[E]) -> Vec<E> {
    let (r_o, r_b, r_sig) = split_point_blocks(manifest, r2);
    let shape = manifest.shape();
    let p = row_weights_in(psi, shape.positions, &r_o);
    let lane_cells = manifest.lane_cells();
    let (pos, sig) = r3.split_at(lane_cells.trailing_zeros() as usize);
    let eq_pos = eq_table(pos);
    let eq_sig = eq_evaluation(&r_sig, sig);
    lane_block_weights(manifest, &r_b)
        .into_iter()
        .enumerate()
        .map(|(lane, eq_b)| {
            let p_at = p[lane * lane_cells..(lane + 1) * lane_cells]
                .iter()
                .zip(&eq_pos)
                .fold(E::zero(), |acc, (&a, &b)| acc + a * b);
            p_at * eq_b * eq_sig
        })
        .collect()
}

/// The honest shared claim `c = sum_y eq(r', y) k(y) = K_0(r') + 2^16 K_1(r') +
/// 2^32 K_2(r') + 2^35 K_3(r') + C_k` from the committed digits of `form`, the first
/// `form.len()` polynomials of `columns` (`G_k` may hold H-FIPS's hint slack after them).
#[must_use]
pub fn claim(columns: &[Cells<'_>], r2: &[E], vars: usize, form: agg_witness::digits::DigitLayout) -> E {
    let point = &r2[r2.len() - vars..];
    let weights = form.weights();
    columns[..form.len()]
        .par_iter()
        .zip(weights.par_iter())
        .map(|(cells, &w)| cells.evaluate(point).mul_i64(w))
        .reduce(E::zero, |a, b| a + b)
        + E::from_i64(form.constant())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::point;
    use agg_witness::gen::SplitMix64;

    fn random_psi(rng: &mut SplitMix64) -> Vec<u8> {
        (0..PROJ_BYTES).map(|_| rng.next_u64() as u8).collect()
    }

    /// The lane weights pull the outputs back: `sum_rho <W^(rho), v^(rho)>` is the extension
    /// of the outputs `k` at the `D_k` suffix of `r^(2)`, row blocks included, and the
    /// verifier's values are the extensions of the tables.
    #[test]
    fn lane_weights_pull_the_outputs_back_per_row_block() {
        use aerie_falcon::protocol::algebra::{evaluate_mle, lift_table};
        for set in [Set::S65, Set::S87] {
            let manifest = CommitmentManifest::for_set(set, 5, crate::manifest::Relation::HExact).unwrap();
            let shape = manifest.shape();
            let psi = test_psi_for(set, 3);
            let mut rng = SplitMix64::new(4);
            let len = 1_usize << manifest.d_main();
            let v: Vec<Vec<i64>> = (0..shape.lanes_k)
                .map(|_| (0..len).map(|_| (rng.next_u64() % (1 << 29)) as i64 - (1 << 28)).collect())
                .collect();
            let records = manifest.padded_records();
            let k = block_outputs(&psi, shape.positions, shape.proj_blocks, records, |i, pos| {
                v[pos / 256][256 * i + pos % 256]
            });
            assert_eq!(k.len(), manifest.group(GroupRole::Proj).logical_cells());
            let r2 = point(&mut rng, manifest.d_main());
            let tables = lane_weight_tables(&manifest, &psi, &r2);
            assert_eq!(tables.len(), shape.lanes_k);
            let lhs = tables.iter().zip(&v).fold(E::zero(), |acc, (w, v)| {
                acc + w.iter().zip(v).fold(E::zero(), |a, (&w, &v)| a + w.mul_i64(v))
            });
            let vars = manifest.group(GroupRole::Proj).logical_vars;
            assert_eq!(lhs, evaluate_mle(&lift_table(&k), &r2[r2.len() - vars..]), "{set:?}");
            let r3 = point(&mut rng, manifest.d_main());
            let at = lane_weights_at(&manifest, &psi, &r2, &r3);
            for (lane, table) in tables.iter().enumerate() {
                assert_eq!(at[lane], evaluate_mle(table, &r3), "{set:?} lane {lane}");
            }
        }
    }

    /// ML-DSA-65 (one row block of 1,536 positions) and ML-DSA-87 (two blocks of 1,024): the
    /// outputs of the lane tables are the direct inner products per block, at cell
    /// `o + 128 (b + g i)`, and the five digits cover exactly `[-2^39, 2^39)`.
    #[test]
    fn lane_sets_project_per_row_block_with_five_digits() {
        use crate::manifest::{Relation, PROJ_BOUND_39};
        for set in [Set::S65, Set::S87] {
            let shape = SetShape::of(set);
            let manifest = CommitmentManifest::for_set(set, 5, Relation::HExact).unwrap();
            let records = manifest.padded_records();
            let (positions, blocks, lanes) = (shape.positions, shape.proj_blocks, shape.lanes_k);
            let row_bytes = positions / 8;
            let psi = test_psi_for(set, 21);
            assert_eq!(psi.len(), manifest.proj_bytes());
            let mut rng = SplitMix64::new(13);
            // V_lo, V_hi lanes of the honest range |v| < 2^29 (cell j + 256 i of lane rho).
            let cells = records * 256;
            let v: Vec<Vec<i64>> = (0..lanes)
                .map(|_| (0..cells).map(|_| (rng.next_u64() % (1 << 30)) as i64 - (1 << 29)).collect())
                .collect();
            let lo: Vec<Vec<i32>> = v.iter().map(|lane| lane.iter().map(|&x| V16_DIGITS.decompose(x)[0] as i32).collect()).collect();
            let hi: Vec<Vec<i32>> = v.iter().map(|lane| lane.iter().map(|&x| V16_DIGITS.decompose(x)[1] as i32).collect()).collect();
            let lo_refs: Vec<&[i32]> = lo.iter().map(Vec::as_slice).collect();
            let hi_refs: Vec<&[i32]> = hi.iter().map(Vec::as_slice).collect();
            let tables = ProjectionTables::build(&manifest, &lo_refs, &hi_refs, &psi).unwrap();
            let d_k = manifest.group(GroupRole::Proj).logical_cells();
            assert_eq!(d_k, PROJ_ROWS * blocks * records);
            assert_eq!(tables.columns().len(), 5);
            assert_eq!(tables.digits(), K39_DIGITS);
            let per_block = positions / blocks;
            for i in 0..records {
                for b in 0..blocks {
                    for o in [0, 1, 77, PROJ_ROWS - 1] {
                        let direct: i64 = (b * per_block..(b + 1) * per_block)
                            .filter(|&pos| psi_bit_in(&psi, row_bytes, o, pos))
                            .map(|pos| v[pos / 256][i * 256 + pos % 256])
                            .sum();
                        let y = o + PROJ_ROWS * (b + blocks * i);
                        assert_eq!(tables.value(y), E::from_i64(direct), "{set} i {i} b {b} o {o}");
                    }
                }
            }
            // The digits cover exactly [-2^39, 2^39); the error names record and row.
            let bound = PROJ_BOUND_39;
            let probes = [-bound, -bound + 1, -(1 << 38) - 3, -1, 0, 1, (1 << 35) + 7, bound - 1];
            let mut k = vec![0_i64; d_k];
            k[..probes.len()].copy_from_slice(&probes);
            let encoded = ProjectionTables::encode(&manifest, &k).unwrap();
            for (y, &value) in probes.iter().enumerate() {
                assert_eq!(encoded.value(y), E::from_i64(value));
                let digits: Vec<i64> = encoded.columns().iter().map(|c| c.value(y)).collect();
                assert_eq!(K39_DIGITS.compose(&digits), value);
                assert!(K39_DIGITS.in_alphabets(&digits));
            }
            let zero = probes.len();
            let digits: Vec<i64> = encoded.columns().iter().map(|c| c.value(zero)).collect();
            assert_eq!(digits, [-(1 << 15), -(1 << 15), -4, -4, 0]);
            let per_record = PROJ_ROWS * blocks;
            for outside in [bound, -bound - 1] {
                k[per_record + 5] = outside;
                assert_eq!(
                    ProjectionTables::encode(&manifest, &k),
                    Err(SourceTableError::ProjectionRange { record: 1, row: 5, value: outside })
                );
            }
            // The claim of the five digits at a point is the outputs' extension.
            let point = point(&mut rng, manifest.group(GroupRole::Proj).logical_vars);
            let outputs: Vec<E> = (0..d_k).map(|y| tables.value(y)).collect();
            assert_eq!(
                claim(&tables.columns(), &point, point.len(), tables.digits()),
                aerie_falcon::protocol::algebra::evaluate_mle(&outputs, &point)
            );
        }
        // One block reproduces `outputs`.
        let psi = test_psi(3);
        let v = |i: usize, pos: usize| (i * 7919 + pos * 31) as i64 % 1000 - 500;
        assert_eq!(block_outputs(&psi, POSITIONS, 1, 3, v), outputs(&psi, 3, v));
    }

    #[test]
    fn outputs_match_the_direct_inner_products() {
        let mut rng = SplitMix64::new(11);
        let psi = random_psi(&mut rng);
        let v: Vec<i64> = (0..3 * POSITIONS).map(|_| (rng.next_u64() % (1 << 27)) as i64 - (1 << 26)).collect();
        let k = outputs(&psi, 3, |i, pos| v[i * POSITIONS + pos]);
        for i in 0..3 {
            for o in 0..PROJ_ROWS {
                let direct: i64 = (0..POSITIONS)
                    .filter(|&pos| psi_bit(&psi, o, pos))
                    .map(|pos| v[i * POSITIONS + pos])
                    .sum();
                assert_eq!(k[i * PROJ_ROWS + o], direct);
            }
        }
    }

    #[test]
    fn digits_cover_exactly_the_range_and_compose_to_k() {
        let manifest = CommitmentManifest::for_batch(8).unwrap();
        let cells = manifest.group(GroupRole::Proj).logical_cells();
        let bound = crate::manifest::PROJ_BOUND;
        let mut k = vec![0_i64; cells];
        let probes = [-bound, -bound + 1, -1, 0, 1, (1 << 32) + 5, bound - 1];
        k[..probes.len()].copy_from_slice(&probes);
        let tables = ProjectionTables::encode(&manifest, &k).unwrap();
        for (y, &value) in k.iter().enumerate() {
            assert_eq!(tables.value(y), E::from_i64(value));
        }
        // The digits of zero: (-2^15, -2^15, -4, 0).
        let zero = probes.len();
        assert_eq!(
            (
                tables.stacked[zero],
                tables.stacked[cells + zero],
                tables.top[zero],
                tables.top[cells + zero]
            ),
            (-(1 << 15), -(1 << 15), -4, 0)
        );
        assert_eq!(K16_DIGITS.constant(), (1 << 34) + (1 << 31) + (1 << 15));
        assert_eq!(K16_DIGITS.range(), (-bound, bound - 1));
        for outside in [bound, -bound - 1, 1 << 40] {
            k[PROJ_ROWS + 5] = outside;
            assert_eq!(
                ProjectionTables::encode(&manifest, &k),
                Err(SourceTableError::ProjectionRange { record: 1, row: 5, value: outside })
            );
        }
    }

    #[test]
    fn the_projection_digits_round_trip_the_full_range() {
        let manifest = CommitmentManifest::for_batch(8).unwrap();
        let cells = manifest.group(GroupRole::Proj).logical_cells();
        let bound = crate::manifest::PROJ_BOUND;
        let probes = [-bound, -bound + 1, -(1 << 32), -1, 0, 1, bound - 2, bound - 1];
        let mut k = vec![0_i64; cells];
        k[..probes.len()].copy_from_slice(&probes);
        let tables = ProjectionTables::encode(&manifest, &k).unwrap();
        for (y, &value) in probes.iter().enumerate() {
            assert_eq!(tables.value(y), E::from_i64(value));
            let digits = [
                i64::from(tables.stacked[y]),
                i64::from(tables.stacked[cells + y]),
                i64::from(tables.top[y]),
                i64::from(tables.top[cells + y]),
            ];
            assert_eq!(K16_DIGITS.compose(&digits), value);
            assert!(K16_DIGITS.in_alphabets(&digits));
        }
        for outside in [bound, bound + 1, -bound - 1] {
            k[PROJ_ROWS + 3] = outside;
            assert!(matches!(
                ProjectionTables::encode(&manifest, &k),
                Err(SourceTableError::ProjectionRange { record: 1, row: 3, value }) if value == outside
            ));
        }
    }

    #[test]
    fn a_coordinate_above_the_certified_bound_leaves_the_range_with_probability_one_half() {
        // `v` differs from zero at one coordinate whose value exceeds 2B = 2^38, so a row
        // passes only if that coordinate's bit is 0; at most half of the 128 rows pass and
        // the batch fails except with probability 2^-128.
        let psi = test_psi(5);
        let x0 = 37;
        let v = 2_i64.pow(38) + 1;
        let bound = crate::manifest::PROJ_BOUND;
        let manifest = CommitmentManifest::for_batch(1).unwrap();
        let records = manifest.padded_records();
        let k = outputs(&psi, records, |i, pos| if i == 0 && pos == x0 { v } else { 0 });
        let passing = k.iter().filter(|&&value| value.abs() <= bound).count();
        let ones = (0..PROJ_ROWS).filter(|&o| psi_bit(&psi, o, x0)).count();
        assert_eq!(passing, (records - 1) * PROJ_ROWS + (PROJ_ROWS - ones));
        assert!(k.iter().any(|&value| value.abs() > bound));
        assert!(matches!(
            ProjectionTables::encode(&manifest, &k),
            Err(SourceTableError::ProjectionRange { record: 0, .. })
        ));
    }

    #[test]
    fn the_weight_table_and_its_evaluation_agree() {
        let manifest = CommitmentManifest::for_batch(8).unwrap();
        let mut rng = SplitMix64::new(12);
        let psi = random_psi(&mut rng);
        let r2 = point(&mut rng, manifest.d_main());
        let r3 = point(&mut rng, manifest.d_main());
        let table = weight_table(&manifest, &psi, &r2);
        let direct = aerie_falcon::protocol::algebra::evaluate_mle(&table, &r3);
        assert_eq!(weight_at(&manifest, &psi, &r2, &r3), direct);
        // On Boolean points the weight is P(pos) eq(r'_sig, i).
        let (r_o, r_sig) = split_point(&manifest, &r2);
        let p = row_weights(&psi, &r_o);
        let sig = eq_table(&r_sig);
        assert_eq!(table[5 + POSITIONS * 3], p[5] * sig[3]);
    }
}
