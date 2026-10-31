//! Dense polynomial opening and fold operations.
//!
//! Storage is D-free; every ring-shaped operation takes the ring dimension as
//! a method const generic and views the flat coefficients at kernel entry.

use super::poly::DensePoly;
use crate::opaque::DecomposeFoldWitness;
use crate::sources::poly_helpers::{
    balanced_ring_decompose_fold_chunked, balanced_ring_decompose_fold_partitioned,
    cached_digit_decompose_fold_partitioned, cached_i16_decompose_fold_partitioned, sparse_mul_acc,
    SignedDigitBasis, ValidatedSparseChallenges,
};
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use akita_error::AkitaError;
use akita_types::SubfieldMultiplierOpeningPoint;
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, Field};

/// Reject a digit shape the dense fold kernels cannot run, before any cache
/// build or decomposition setup can panic on it.
fn validate_digit_shape(num_digits: usize, log_basis: u32) -> Result<(), AkitaError> {
    SignedDigitBasis::new(log_basis)?;
    if num_digits == 0 {
        return Err(AkitaError::InvalidInput(
            "dense fold digit count must be nonzero".into(),
        ));
    }
    Ok(())
}

/// Fold ring blocks from signed cells with base-field weights.
///
/// The accumulation order matches [`CyclotomicRing::scale_accumulate_into`]:
/// within a block, positions run in order and each coefficient receives
/// `cell * weight + accumulator`.
fn fold_blocks_from_cells<F, T, const D: usize>(
    planes: &[[T; D]],
    scalars: &[F],
    num_positions_per_block: usize,
    to_field: impl Fn(T) -> F + Sync,
) -> Vec<CyclotomicRing<F, D>>
where
    F: Field,
    T: Copy + Send + Sync,
{
    let num_live_blocks = planes.len().div_ceil(num_positions_per_block);
    cfg_into_iter!(0..num_live_blocks)
        .map(|block_index| {
            let start = block_index * num_positions_per_block;
            let end = (start + num_positions_per_block).min(planes.len());
            let mut accumulator = CyclotomicRing::<F, D>::zero();
            for (cells, &weight) in planes[start..end].iter().zip(scalars.iter()) {
                for (destination, &cell) in
                    accumulator.coefficients_mut().iter_mut().zip(cells.iter())
                {
                    *destination = to_field(cell).mul_add(weight, *destination);
                }
            }
            accumulator
        })
        .collect()
}

/// Fold ring blocks from signed cells with sparse ring multipliers.
///
/// The per-position ring carries the same canonical coefficients the field
/// plane would supply, so `mul_accumulate_sparse_rhs_into` sees identical
/// inputs and produces identical accumulations.
fn fold_blocks_ring_from_cells<F, T, const D: usize>(
    planes: &[[T; D]],
    scalars: &[CyclotomicRing<F, D>],
    num_positions_per_block: usize,
    to_field: impl Fn(T) -> F + Sync,
) -> Vec<CyclotomicRing<F, D>>
where
    F: Field + CanonicalEncoding,
    T: Copy + Send + Sync,
{
    let num_live_blocks = planes.len().div_ceil(num_positions_per_block);
    cfg_into_iter!(0..num_live_blocks)
        .map(|block_index| {
            let start = block_index * num_positions_per_block;
            let end = (start + num_positions_per_block).min(planes.len());
            let mut accumulator = CyclotomicRing::<F, D>::zero();
            for (cells, multiplier) in planes[start..end].iter().zip(scalars.iter()) {
                let coefficients: [F; D] = std::array::from_fn(|index| to_field(cells[index]));
                CyclotomicRing::from_coefficients(coefficients)
                    .mul_accumulate_sparse_rhs_into(multiplier, &mut accumulator);
            }
            accumulator
        })
        .collect()
}

impl<F> DensePoly<F>
where
    F: Field + CanonicalEncoding,
{
    pub(crate) fn decompose_fold_chunked<const D: usize>(
        &self,
        challenges: &[SparseChallenge],
        chunk_ranges: &[std::ops::Range<usize>],
        num_positions_per_block: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> Result<Vec<DecomposeFoldWitness>, AkitaError> {
        validate_digit_shape(num_digits, log_basis)?;
        let num_rings = self.live_ring_count::<D>()?;
        let Some((planes, basis)) = self.digit_planes_for::<D>(num_digits, log_basis) else {
            if num_digits == 1 && log_basis == akita_params::MAX_I16_LOG_BASIS {
                if let Some(planes) = self.small_i16_ring_coeffs::<D>() {
                    return chunk_ranges
                        .iter()
                        .map(|range| {
                            let ring_start = range.start * num_positions_per_block;
                            let ring_end = (range.end * num_positions_per_block).min(planes.len());
                            let coefficients = cached_i16_decompose_fold_partitioned::<D>(
                                &planes[ring_start.min(planes.len())..ring_end],
                                &challenges[range.clone()],
                                num_positions_per_block,
                                log_basis,
                            )?;
                            Ok(DecomposeFoldWitness::from_centered_rows(coefficients))
                        })
                        .collect::<Result<Vec<_>, AkitaError>>();
                }
            }
            // A centered-i8 mirror is one exact balanced digit at every basis
            // of at least eight bits, including the signed-16 bases.
            if num_digits == 1 && log_basis > akita_params::MAX_I8_LOG_BASIS {
                if let Some(planes) = self.small_i8_ring_coeffs::<D>() {
                    let basis = SignedDigitBasis::new(log_basis)?;
                    return chunk_ranges
                        .iter()
                        .map(|range| {
                            let ring_start = range.start * num_positions_per_block;
                            let ring_end = (range.end * num_positions_per_block).min(planes.len());
                            let coefficients = cached_digit_decompose_fold_partitioned::<D>(
                                &planes[ring_start.min(planes.len())..ring_end],
                                &challenges[range.clone()],
                                num_positions_per_block,
                                1,
                                basis,
                            )?;
                            Ok(DecomposeFoldWitness::from_centered_rows(coefficients))
                        })
                        .collect::<Result<Vec<_>, AkitaError>>();
                }
            }
            let coeffs = self.ring_coeffs::<D>()?;
            return balanced_ring_decompose_fold_chunked(
                coeffs,
                challenges,
                chunk_ranges,
                num_positions_per_block,
                &BalancedDecomposePow2Params::new(num_digits, log_basis),
            );
        };
        chunk_ranges
            .iter()
            .map(|range| {
                let ring_start = range.start * num_positions_per_block;
                let ring_end = (range.end * num_positions_per_block).min(num_rings);
                let coefficients = cached_digit_decompose_fold_partitioned::<D>(
                    &planes[ring_start.min(num_rings) * num_digits..ring_end * num_digits],
                    &challenges[range.clone()],
                    num_positions_per_block,
                    num_digits,
                    basis,
                )?;
                Ok(DecomposeFoldWitness::from_centered_rows(coefficients))
            })
            .collect::<Result<Vec<_>, AkitaError>>()
    }

    pub(crate) fn fold_blocks<const D: usize>(
        &self,
        scalars: &[F],
        num_positions_per_block: usize,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        let num_rings = self.live_ring_count::<D>()?;
        if let Some(planes) = self.small_i8_ring_coeffs::<D>() {
            debug_assert_eq!(planes.len(), num_rings);
            return Ok(fold_blocks_from_cells(
                planes,
                scalars,
                num_positions_per_block,
                F::from_i8,
            ));
        }
        if let Some(planes) = self.small_i16_ring_coeffs::<D>() {
            debug_assert_eq!(planes.len(), num_rings);
            return Ok(fold_blocks_from_cells(
                planes,
                scalars,
                num_positions_per_block,
                F::from_i16,
            ));
        }
        let coeffs = self.ring_coeffs::<D>()?;
        debug_assert_eq!(coeffs.len(), num_rings);
        let num_live_blocks = num_rings.div_ceil(num_positions_per_block);
        Ok(cfg_into_iter!(0..num_live_blocks)
            .map(|i| {
                let start = i * num_positions_per_block;
                let end = (start + num_positions_per_block).min(num_rings);
                let block = &coeffs[start..end];
                let mut acc = CyclotomicRing::<F, D>::zero();
                for (b_j, &a_j) in block.iter().zip(scalars.iter()) {
                    b_j.scale_accumulate_into(&mut acc, a_j);
                }
                acc
            })
            .collect())
    }

    pub(crate) fn fold_blocks_ring<const D: usize>(
        &self,
        scalars: &[CyclotomicRing<F, D>],
        num_positions_per_block: usize,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        let num_rings = self.live_ring_count::<D>()?;
        if let Some(planes) = self.small_i8_ring_coeffs::<D>() {
            debug_assert_eq!(planes.len(), num_rings);
            return Ok(fold_blocks_ring_from_cells(
                planes,
                scalars,
                num_positions_per_block,
                F::from_i8,
            ));
        }
        if let Some(planes) = self.small_i16_ring_coeffs::<D>() {
            debug_assert_eq!(planes.len(), num_rings);
            return Ok(fold_blocks_ring_from_cells(
                planes,
                scalars,
                num_positions_per_block,
                F::from_i16,
            ));
        }
        let coeffs = self.ring_coeffs::<D>()?;
        debug_assert_eq!(coeffs.len(), num_rings);
        let num_live_blocks = num_rings.div_ceil(num_positions_per_block);
        Ok(cfg_into_iter!(0..num_live_blocks)
            .map(|i| {
                let start = i * num_positions_per_block;
                let end = (start + num_positions_per_block).min(num_rings);
                let block = &coeffs[start..end];
                let mut acc = CyclotomicRing::<F, D>::zero();
                for (b_j, &a_j) in block.iter().zip(scalars.iter()) {
                    b_j.mul_accumulate_sparse_rhs_into(&a_j, &mut acc);
                }
                acc
            })
            .collect())
    }

    pub(crate) fn evaluate_and_fold<const D: usize>(
        &self,
        live_block_weights: &[F],
        position_weights: &[F],
        num_positions_per_block: usize,
    ) -> Result<(CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>), AkitaError> {
        Ok(crate::sources::poly_helpers::fused_evaluate_and_fold_base(
            self.fold_blocks::<D>(position_weights, num_positions_per_block)?,
            live_block_weights,
        ))
    }

    pub(crate) fn evaluate_and_fold_subfield<const D: usize>(
        &self,
        multipliers: &SubfieldMultiplierOpeningPoint<F>,
        num_positions_per_block: usize,
    ) -> Result<(CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>), AkitaError> {
        let position_weights = multipliers.materialize_position_rings::<D>()?;
        let live_block_weights = multipliers.materialize_fold_rings::<D>()?;
        Ok(
            crate::sources::poly_helpers::fused_evaluate_and_fold_materialized(
                self.fold_blocks_ring(&position_weights, num_positions_per_block)?,
                &live_block_weights,
            ),
        )
    }

    #[tracing::instrument(skip_all, name = "DensePoly::decompose_fold")]
    pub(crate) fn decompose_fold<const D: usize>(
        &self,
        challenges: &[SparseChallenge],
        num_positions_per_block: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> Result<DecomposeFoldWitness, AkitaError> {
        validate_digit_shape(num_digits, log_basis)?;
        let num_rings = self.live_ring_count::<D>()?;

        if let Some((digit_planes, basis)) = self.digit_planes_for::<D>(num_digits, log_basis) {
            let coeff_accum = {
                let _span = tracing::info_span!("dense_cached_digit_accumulate").entered();
                cached_digit_decompose_fold_partitioned::<D>(
                    digit_planes,
                    challenges,
                    num_positions_per_block,
                    num_digits,
                    basis,
                )?
            };
            return Ok(DecomposeFoldWitness::from_centered_rows(coeff_accum));
        }

        let params = BalancedDecomposePow2Params::new(num_digits, log_basis);

        // The single-digit scratch is i8; wider bases need the checked i16 kernel below.
        if num_digits == 1 && log_basis <= akita_params::MAX_I8_LOG_BASIS {
            let challenges = ValidatedSparseChallenges::<D>::new(
                challenges,
                num_rings,
                num_positions_per_block,
            )?;
            if let Some(small_coeffs) = self.small_i8_ring_coeffs::<D>() {
                let coeff_accum: Vec<[i32; D]> = {
                    let _span =
                        tracing::info_span!("dense_single_digit_cached_accumulate").entered();
                    cfg_into_iter!(0..num_positions_per_block)
                        .map(|elem_idx| {
                            let mut z_local = [0i32; D];

                            for (block_idx, c_i) in challenges.iter().enumerate() {
                                let global_idx = block_idx * num_positions_per_block + elem_idx;
                                if global_idx >= small_coeffs.len() {
                                    continue;
                                }
                                sparse_mul_acc::<D>(&small_coeffs[global_idx], c_i, &mut z_local);
                            }

                            z_local
                        })
                        .collect()
                };

                return Ok(DecomposeFoldWitness::from_centered_rows(coeff_accum));
            }

            let coeffs = self.ring_coeffs::<D>()?;
            let coeff_accum: Vec<[i32; D]> = {
                let _span = tracing::info_span!("dense_single_digit_accumulate").entered();
                cfg_into_iter!(0..num_positions_per_block)
                    .map(|elem_idx| {
                        let mut z_local = [0i32; D];
                        let mut digit_plane = [[0i8; D]];

                        for (block_idx, c_i) in challenges.iter().enumerate() {
                            let global_idx = block_idx * num_positions_per_block + elem_idx;
                            if global_idx >= num_rings {
                                continue;
                            }
                            let ring = &coeffs[global_idx];
                            ring.balanced_decompose_pow2_i8_into_with_params(
                                &mut digit_plane,
                                &params,
                            );
                            sparse_mul_acc::<D>(&digit_plane[0], c_i, &mut z_local);
                        }

                        z_local
                    })
                    .collect()
            };

            return Ok(DecomposeFoldWitness::from_centered_rows(coeff_accum));
        }

        // A signed-16 mirror holds the single balanced digit of every live
        // coefficient, so the probe reads 2 B/cell rather than the 8 B/cell
        // field plane. The accumulation plan and flushing are the same as for
        // the live-ring decomposition.
        if num_digits == 1 && log_basis == akita_params::MAX_I16_LOG_BASIS {
            if let Some(planes) = self.small_i16_ring_coeffs::<D>() {
                let coeff_accum = cached_i16_decompose_fold_partitioned::<D>(
                    planes,
                    challenges,
                    num_positions_per_block,
                    log_basis,
                )?;
                return Ok(DecomposeFoldWitness::from_centered_rows(coeff_accum));
            }
        }

        // The same single-digit argument holds for a centered-i8 mirror at
        // any signed-16 basis: every cell is already one balanced digit.
        if num_digits == 1 && log_basis > akita_params::MAX_I8_LOG_BASIS {
            if let Some(planes) = self.small_i8_ring_coeffs::<D>() {
                let basis = SignedDigitBasis::new(log_basis)?;
                let coeff_accum = cached_digit_decompose_fold_partitioned::<D>(
                    planes,
                    challenges,
                    num_positions_per_block,
                    1,
                    basis,
                )?;
                return Ok(DecomposeFoldWitness::from_centered_rows(coeff_accum));
            }
        }

        let coeffs = self.ring_coeffs::<D>()?;
        let centered_coeffs = {
            let _span = tracing::info_span!("dense_multi_digit_accumulate").entered();
            balanced_ring_decompose_fold_partitioned::<F, D>(
                coeffs,
                challenges,
                num_positions_per_block,
                &params,
            )?
        };

        Ok(DecomposeFoldWitness::from_centered_rows(centered_coeffs))
    }
}
