//! CpuBackend kernels over dense polynomial views.

use super::poly::DensePoly;
use super::views::{DenseBatchView, DenseView};
use crate::arithmetic::coefficient_packing::{
    coefficient_packing_partials_from_position_source, FusedPackingWeights,
};
use crate::opaque::DecomposeFoldWitness;
use crate::opaque::{aggregate_decompose_fold_witnesses, CpuFoldResponses};
use crate::opaque::{
    CpuBackend, DecomposeFoldBatchPlan, DecomposeFoldPlan, OpeningFoldPlan, RootOpeningSource,
    RootPolyMeta, RootPolyShape, SubringCoefficientPackingBatchKernel,
    SubringCoefficientPackingPartials, SubringCoefficientPackingPlan,
};
use crate::opaque::{OpeningBatchKernel, OpeningFoldKernel, OpeningFoldOutput};
use akita_error::AkitaError;
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced};

impl<F, E, const D: usize> OpeningFoldKernel<DenseView<'_, F, D>, F, D> for CpuBackend<F, E>
where
    F: Field + CanonicalEncoding,
{
    fn evaluate_and_fold(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: DenseView<'_, F, D>,
        plan: OpeningFoldPlan<'_, F>,
    ) -> Result<OpeningFoldOutput<F, D>, AkitaError> {
        let num_positions_per_block = plan.num_positions_per_block();
        if num_positions_per_block == 0 {
            return Err(AkitaError::InvalidInput(
                "num_positions_per_block must be positive".to_string(),
            ));
        }
        let num_live_blocks = source
            .poly
            .live_ring_count::<D>()?
            .div_ceil(num_positions_per_block);
        plan.validate::<D>(num_live_blocks)?;
        let (eval, folded) = match plan {
            OpeningFoldPlan::Base {
                live_block_weights,
                position_weights,
                num_positions_per_block,
            } => source.poly.evaluate_and_fold::<D>(
                live_block_weights,
                position_weights,
                num_positions_per_block,
            )?,
            OpeningFoldPlan::Subfield {
                multipliers,
                num_positions_per_block,
            } => source
                .poly
                .evaluate_and_fold_subfield(multipliers, num_positions_per_block)?,
        };
        Ok(OpeningFoldOutput { eval, folded })
    }

    fn decompose_fold(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: DenseView<'_, F, D>,
        plan: DecomposeFoldPlan<'_>,
    ) -> Result<DecomposeFoldWitness, AkitaError> {
        if plan.num_positions_per_block == 0 {
            return Err(AkitaError::InvalidInput(
                "num_positions_per_block must be positive".to_string(),
            ));
        }
        let num_live_blocks = source
            .poly
            .live_ring_count::<D>()?
            .div_ceil(plan.num_positions_per_block);
        if plan.challenges.len() != num_live_blocks {
            return Err(AkitaError::InvalidSize {
                expected: num_live_blocks,
                actual: plan.challenges.len(),
            });
        }
        source.poly.decompose_fold::<D>(
            plan.challenges,
            plan.num_positions_per_block,
            plan.num_digits,
            plan.log_basis,
        )
    }
}

impl<F, E, const D: usize> OpeningBatchKernel<DenseBatchView<'_, F, D>, F, D> for CpuBackend<F, E>
where
    F: Field + CanonicalEncoding,
{
    fn evaluate_and_fold_batch(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: DenseBatchView<'_, F, D>,
        plan: OpeningFoldPlan<'_, F>,
    ) -> Result<Vec<OpeningFoldOutput<F, D>>, AkitaError> {
        source
            .polys
            .iter()
            .map(|poly| {
                self.evaluate_and_fold(
                    prepared,
                    <DensePoly<F> as RootOpeningSource<F, D>>::opening_view(*poly)?,
                    plan,
                )
            })
            .collect()
    }

    fn decompose_fold_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: DenseBatchView<'_, F, D>,
        plan: DecomposeFoldBatchPlan<'_>,
    ) -> Result<CpuFoldResponses, AkitaError> {
        let (num_positions_per_block, num_digits, log_basis) = plan.scalar_params();
        let challenges_per_poly = plan.validate_uniform_batch(source.polys.iter().map(|poly| {
            RootPolyShape::<F, D>::num_live_ring_elems(*poly).div_ceil(num_positions_per_block)
        }))?;
        match plan {
            DecomposeFoldBatchPlan::Sparse { challenges, .. } => Ok(CpuFoldResponses::sparse(
                aggregate_decompose_fold_witnesses::<D>(
                    // The caches are built serially first, so no task can enter a
                    // lazy slot build while the others are already running.
                    {
                        for poly in source.polys {
                            poly.prime_digit_cache(D, num_digits, log_basis)?;
                        }
                        cfg_into_iter!(0..source.polys.len())
                            .map(|index| {
                                let start = index * challenges_per_poly;
                                source.polys[index].decompose_fold::<D>(
                                    &challenges[start..start + challenges_per_poly],
                                    num_positions_per_block,
                                    num_digits,
                                    log_basis,
                                )
                            })
                            .collect::<Vec<_>>()
                    },
                )?,
            )),
            DecomposeFoldBatchPlan::SparseChunked {
                challenges,
                chunk_ranges,
                ..
            } => {
                let mut by_chunk = (0..chunk_ranges.len())
                    .map(|_| Vec::with_capacity(source.polys.len()))
                    .collect::<Vec<_>>();
                for (poly, poly_challenges) in source
                    .polys
                    .iter()
                    .zip(challenges.as_slice().chunks_exact(challenges_per_poly))
                {
                    for (chunk, witness) in
                        by_chunk.iter_mut().zip(poly.decompose_fold_chunked::<D>(
                            poly_challenges,
                            chunk_ranges,
                            num_positions_per_block,
                            num_digits,
                            log_basis,
                        )?)
                    {
                        chunk.push(Ok(witness));
                    }
                }
                CpuFoldResponses::chunked::<D>(
                    by_chunk
                        .into_iter()
                        .map(aggregate_decompose_fold_witnesses::<D>)
                        .collect::<Result<Vec<_>, _>>()?,
                )
            }
        }
    }
}

/// Run the shared packing loop over one ring-position source.
///
/// `ring_at` returns the `D` canonical coefficients of a live position and
/// `to_field` converts one stored cell to its field element. Both the field
/// plane and the signed mirrors use the same loop and the same position order.
fn packing_partials_from_cells<'a, F, E, T, const D: usize>(
    poly: &DensePoly<F>,
    fused_weights: &FusedPackingWeights<'_, E>,
    ring_at: impl Fn(usize) -> Result<&'a [T; D], AkitaError> + Sync,
    to_field: impl Fn(T) -> F + Sync,
) -> Result<SubringCoefficientPackingPartials<F>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + akita_types::FpExtEncoding<F> + MulBaseUnreduced<F>,
    T: Copy + Send + Sync + 'a,
{
    let point = fused_weights.point();
    let coordinates = coefficient_packing_partials_from_position_source::<F, E, _, D>(
        fused_weights,
        RootPolyMeta::<F>::num_vars(poly),
        ring_at,
        |_, coefficient, ring| to_field(ring[coefficient]),
    )?;
    SubringCoefficientPackingPartials::new(point.geometry(), point.num_live_blocks(), coordinates)
}

fn missing_packing_position() -> AkitaError {
    AkitaError::Internal("dense coefficient-packing position is outside its admitted source".into())
}

pub(crate) fn dense_coefficient_packing_partials<F, E, const D: usize>(
    source: DenseBatchView<'_, F, D>,
    fused_weights: &FusedPackingWeights<'_, E>,
) -> Result<Vec<SubringCoefficientPackingPartials<F>>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + akita_types::FpExtEncoding<F> + MulBaseUnreduced<F>,
{
    let point = fused_weights.point();
    cfg_iter!(source.polys)
        .map(|poly| {
            // Dense roots authenticate the complete Boolean hypercube, so
            // every stored ring is live. Exact-prefix storage is reserved
            // for recursive witness views.
            let live_positions = poly.live_ring_count::<D>()?;
            if live_positions != point.num_live_positions() {
                return Err(AkitaError::InvalidSize {
                    expected: point.num_live_positions(),
                    actual: live_positions,
                });
            }
            if let Some(rings) = poly.small_i8_ring_coeffs::<D>() {
                return packing_partials_from_cells::<F, E, i8, D>(
                    poly,
                    fused_weights,
                    |position| rings.get(position).ok_or_else(missing_packing_position),
                    F::from_i8,
                );
            }
            if let Some(rings) = poly.small_i16_ring_coeffs::<D>() {
                return packing_partials_from_cells::<F, E, i16, D>(
                    poly,
                    fused_weights,
                    |position| rings.get(position).ok_or_else(missing_packing_position),
                    F::from_i16,
                );
            }
            let rings = poly.ring_coeffs::<D>()?;
            packing_partials_from_cells::<F, E, F, D>(
                poly,
                fused_weights,
                |position| {
                    rings
                        .get(position)
                        .map(|ring| ring.coefficients())
                        .ok_or_else(missing_packing_position)
                },
                |value| value,
            )
        })
        .collect()
}

impl<F, E, const D: usize> SubringCoefficientPackingBatchKernel<DenseBatchView<'_, F, D>, F, E, D>
    for CpuBackend<F, E>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + akita_types::FpExtEncoding<F> + MulBaseUnreduced<F>,
{
    fn coefficient_packing_partials_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: DenseBatchView<'_, F, D>,
        plan: SubringCoefficientPackingPlan<'_, E>,
    ) -> Result<Vec<SubringCoefficientPackingPartials<F>>, AkitaError> {
        let fused_weights = FusedPackingWeights::new(plan.point)?;
        dense_coefficient_packing_partials(source, &fused_weights)
    }
}
