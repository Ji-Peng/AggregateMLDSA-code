//! The `{0, 1}` projection of the ring quotient `v` (`docs/spec.md`, the projection definition).
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
//! Soundness (`docs/soundness.md`, the projection bound): `k in [-2^37, 2^37)` for every row certifies
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
/// `K_4`), at cell `o + 128 i` (`o + 128 (b + g i)` with row blocks).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectionTables {
    /// `K_0` then `K_1` (stored balanced, looked up as `K + 2^15`).
    pub stacked: Vec<i32>,
    /// `K_2` then `K_3` (`[-4, 3]` each, leaf and link), then `K_4` (`[-2, 1]`) for five digits.
    pub top: Vec<i8>,
}

impl ProjectionTables {
    /// The honest digits of the outputs `k` (record-major, `PROJ_ROWS` per record); the
    /// prover aborts when some `k` leaves `[-2^37, 2^37)` (completeness only:
    /// `docs/spec.md`, the projection definition) and names the first such record.
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
        Ok(Self { stacked, top })
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
        Self { stacked, top }
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

    /// Cells of each digit polynomial.
    fn cells(&self) -> usize {
        self.stacked.len() / 2
    }

    /// The digit layout of the tables (four or five digits).
    #[must_use]
    pub fn digits(&self) -> agg_witness::digits::DigitLayout {
        if self.top.len() == 2 * self.cells() { K16_DIGITS } else { K39_DIGITS }
    }

    /// The four (five) `G_k` polynomials in manifest order.
    #[must_use]
    pub fn columns(&self) -> Vec<Cells<'_>> {
        let cells = self.cells();
        vec![Cells::Wide(&self.stacked[..cells]), Cells::Wide(&self.stacked[cells..])]
            .into_iter()
            .chain(self.top.chunks_exact(cells).map(Cells::Narrow))
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

/// The honest shared claim `c = sum_y eq(r', y) k(y) = K_0(r') + 2^16 K_1(r') +
/// 2^32 K_2(r') + 2^35 K_3(r') + C_k` from the committed digits.
#[must_use]
pub fn claim(columns: &[Cells<'_>], r2: &[E], vars: usize) -> E {
    let point = &r2[r2.len() - vars..];
    let form = if columns.len() == K39_DIGITS.len() { K39_DIGITS } else { K16_DIGITS };
    let weights = form.weights();
    columns
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

    /// ML-DSA-65 (one row block of 1,536 positions) and ML-DSA-87 (two blocks of 1,024): the
    /// outputs of the lane tables are the direct inner products per block, at cell
    /// `o + 128 (b + g i)`, and the five digits cover exactly `[-2^39, 2^39)`.
    #[test]
    fn lane_sets_project_per_row_block_with_five_digits() {
        use crate::manifest::{Relation, PROJ_BOUND_39};
        for set in [Set::S65, Set::S87] {
            let shape = SetShape::of(set);
            let manifest = CommitmentManifest::for_set(set, 5, Relation::Hu).unwrap();
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
                claim(&tables.columns(), &point, point.len()),
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
