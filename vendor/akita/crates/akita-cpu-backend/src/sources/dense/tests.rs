use super::poly::DensePoly;
use crate::commitment::CommitmentSource;
use crate::opaque::RootOpeningSource;
use crate::opaque::RootPolyMeta;
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use jolt_field::Prime128OffsetA7F7 as F;
use jolt_field::{CanonicalEncoding, Ring, Zero};

/// Deterministic centered `i8` cells.
fn random_i8_cells(len: usize, seed: u64) -> Vec<i8> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (((state >> 56) as i16) - 128) as i8
        })
        .collect()
}

/// Deterministic centered `i16` cells spanning the full signed range.
fn random_i16_cells(len: usize, seed: u64) -> Vec<i32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 48) as u32 & 0xFFFF) as i32 - 32_768
        })
        .collect()
}

/// Copy a polynomial and force its canonical field plane, then drop the
/// mirrors so every kernel takes the field-plane path.
fn field_plane_copy<F: jolt_field::Field>(poly: &DensePoly<F>) -> DensePoly<F> {
    let mut copy = poly.clone();
    let _ = copy.field_coeffs();
    copy.small_i8_coeffs = None;
    copy.small_i16_coeffs = None;
    copy
}

#[test]
fn small_mirror_fold_matches_the_field_plane() {
    use jolt_field::{Prime64Offset59 as SmallF, Ring};

    const D: usize = 64;
    const NUM_VARS: usize = 11;
    // A partial last block: 32 rings split into blocks of three.
    const POSITIONS_PER_BLOCK: usize = 3;
    let num_rings = (1usize << NUM_VARS) / D;

    let scalars = (0..num_rings)
        .map(|index| SmallF::from_u64(index as u64 * 7 + 3))
        .collect::<Vec<_>>();
    let rings = (0..num_rings)
        .map(|index| {
            CyclotomicRing::<SmallF, D>::from_coefficients(std::array::from_fn(|slot| {
                SmallF::from_u64((index * 131 + slot * 17 + 5) as u64)
            }))
        })
        .collect::<Vec<_>>();

    let i8_poly = DensePoly::<SmallF>::from_small_i8_cells(
        NUM_VARS,
        &random_i8_cells(1 << NUM_VARS, 0x5eed_1234),
    )
    .unwrap();
    let i16_poly = DensePoly::<SmallF>::from_small_i16_cells(
        NUM_VARS,
        &random_i16_cells(1 << NUM_VARS, 0xbeef_5678),
    )
    .unwrap();
    assert!(!i8_poly.field_plane_is_materialized());
    assert!(!i16_poly.field_plane_is_materialized());

    for poly in [&i8_poly, &i16_poly] {
        let reference = field_plane_copy(poly);
        assert_eq!(
            poly.fold_blocks::<D>(&scalars, POSITIONS_PER_BLOCK)
                .unwrap(),
            reference
                .fold_blocks::<D>(&scalars, POSITIONS_PER_BLOCK)
                .unwrap(),
        );
        assert_eq!(
            poly.fold_blocks_ring::<D>(&rings, POSITIONS_PER_BLOCK)
                .unwrap(),
            reference
                .fold_blocks_ring::<D>(&rings, POSITIONS_PER_BLOCK)
                .unwrap(),
        );
    }
    assert!(!i8_poly.field_plane_is_materialized());
    assert!(!i16_poly.field_plane_is_materialized());
}

#[test]
fn small_mirror_packing_matches_the_field_plane() {
    use crate::arithmetic::coefficient_packing::FusedPackingWeights;
    use akita_params::{BasisMode, SubringCoefficientPackingGeometry};
    use akita_types::PreparedSubringCoefficientPackingPoint;
    use jolt_field::{Prime64Offset59 as SmallF, Ring};

    const D: usize = 64;
    const NUM_VARS: usize = 10;
    // One partial block: 16 rings filled into a 32-position domain.
    const POSITIONS_PER_BLOCK: usize = 32;
    let num_rings = (1usize << NUM_VARS) / D;

    let geometry = SubringCoefficientPackingGeometry::try_new(1, D, 64).unwrap();
    let point_values = (0..NUM_VARS)
        .map(|index| SmallF::from_u64(index as u64 * 3 + 5))
        .collect::<Vec<_>>();
    let point = PreparedSubringCoefficientPackingPoint::<SmallF>::new(
        geometry,
        BasisMode::Lagrange,
        num_rings,
        POSITIONS_PER_BLOCK,
        NUM_VARS,
        &point_values,
    )
    .unwrap();
    let weights = FusedPackingWeights::<SmallF>::new(&point).unwrap();

    let pack = |polys: &[&DensePoly<SmallF>]| {
        let batch =
            <DensePoly<SmallF> as RootOpeningSource<SmallF, D>>::opening_batch(polys).unwrap();
        super::dense_coefficient_packing_partials::<SmallF, SmallF, D>(batch, &weights)
            .unwrap()
            .into_iter()
            .map(|partials| partials.coordinates().to_vec())
            .collect::<Vec<_>>()
    };

    for (poly, cells) in [
        (
            DensePoly::<SmallF>::from_small_i8_cells(
                NUM_VARS,
                &random_i8_cells(1 << NUM_VARS, 0x1357_9bdf),
            )
            .unwrap(),
            "i8",
        ),
        (
            DensePoly::<SmallF>::from_small_i16_cells(
                NUM_VARS,
                &random_i16_cells(1 << NUM_VARS, 0x2468_ace0),
            )
            .unwrap(),
            "i16",
        ),
    ] {
        assert!(!poly.field_plane_is_materialized(), "{cells}");
        let reference = field_plane_copy(&poly);
        assert_eq!(pack(&[&poly]), pack(&[&reference]), "{cells}");
        assert!(
            !poly.field_plane_is_materialized(),
            "{cells} packing materialized the field plane"
        );
    }
}

#[test]
fn small_mirror_fold_pads_the_last_ring() {
    use jolt_field::{Prime64Offset59 as SmallF, Ring};

    // 64 live cells fill a quarter of the single 256-coefficient ring, so the
    // mirror and the field plane both rely on the physical zero padding.
    const D: usize = 256;
    const NUM_VARS: usize = 6;
    const POSITIONS_PER_BLOCK: usize = 2;
    let scalars = vec![SmallF::from_u64(11), SmallF::from_u64(29)];
    let multipliers = vec![CyclotomicRing::<SmallF, D>::from_coefficients(
        std::array::from_fn(|slot| SmallF::from_u64((slot * 7 + 1) as u64)),
    )];

    for (poly, kind) in [
        (
            DensePoly::<SmallF>::from_small_i8_cells(
                NUM_VARS,
                &random_i8_cells(1 << NUM_VARS, 0x0ddc_0ffe),
            )
            .unwrap(),
            "i8",
        ),
        (
            DensePoly::<SmallF>::from_small_i16_cells(
                NUM_VARS,
                &random_i16_cells(1 << NUM_VARS, 0x0bad_f00d),
            )
            .unwrap(),
            "i16",
        ),
    ] {
        let reference = field_plane_copy(&poly);
        assert_eq!(
            poly.fold_blocks::<D>(&scalars, POSITIONS_PER_BLOCK)
                .unwrap(),
            reference
                .fold_blocks::<D>(&scalars, POSITIONS_PER_BLOCK)
                .unwrap(),
            "{kind}",
        );
        assert_eq!(
            poly.fold_blocks_ring::<D>(&multipliers, POSITIONS_PER_BLOCK)
                .unwrap(),
            reference
                .fold_blocks_ring::<D>(&multipliers, POSITIONS_PER_BLOCK)
                .unwrap(),
            "{kind}",
        );
    }
}

#[test]
fn narrow_commit_representation_does_not_materialize_the_field_plane() {
    use crate::commitment::{DenseType, PolynomialType};
    use crate::opaque::CommitInnerPlan;
    use jolt_field::Prime64Offset59 as SmallF;

    const D: usize = 64;
    const NUM_VARS: usize = 10;
    const NUM_RINGS: usize = (1 << NUM_VARS) / D;
    let poly = DensePoly::<SmallF>::from_small_i8_cells(
        NUM_VARS,
        &random_i8_cells(1 << NUM_VARS, 0x2468_1357),
    )
    .unwrap();
    assert!(poly.prime_digit_cache(D, 1, 8).unwrap());
    assert!(!poly.field_plane_is_materialized());

    let plan = CommitInnerPlan {
        ring_dimension: D,
        num_live_blocks: NUM_RINGS,
        n_a: 1,
        num_positions_per_block: 1,
        num_digits_inner: 1,
        log_basis_inner: 8,
    };
    let available =
        <DensePoly<SmallF> as CommitmentSource<SmallF>>::available_polynomial_types(&poly, &plan)
            .unwrap();
    assert_eq!(
        available.as_slice().first().copied(),
        Some(PolynomialType::Dense(DenseType::PredecomposedDigits)),
    );
    assert!(!poly.field_plane_is_materialized());
}

#[test]
fn chunked_fold_matches_windowed_reference_and_global() {
    use akita_challenges::SparseChallenge;

    const D: usize = 64;
    const POSITIONS: usize = 2;
    let poly =
        DensePoly::<F>::from_ring_coeffs((0..8).map(|index| ring::<D>(index * 10)).collect())
            .unwrap();
    let challenges = (0..4)
        .map(|block| SparseChallenge {
            positions: vec![(block * 2) as u32, (block * 2 + 1) as u32].into(),
            coeffs: vec![1, -1].into(),
        })
        .collect::<Vec<_>>();
    let global = poly
        .decompose_fold::<D>(&challenges, POSITIONS, 2, 4)
        .unwrap();

    for chunk_count in [2, 4, 8] {
        let ranges = akita_params::dyadic_block_ranges(challenges.len(), chunk_count).unwrap();
        let chunks = poly
            .decompose_fold_chunked::<D>(&challenges, &ranges, POSITIONS, 2, 4)
            .unwrap();
        assert_eq!(chunks.len(), chunk_count);
        for (range, chunk) in ranges.iter().zip(&chunks) {
            let window = challenges
                .iter()
                .enumerate()
                .map(|(block, challenge)| {
                    if range.contains(&block) {
                        challenge.clone()
                    } else {
                        SparseChallenge {
                            positions: Vec::new().into(),
                            coeffs: Vec::new().into(),
                        }
                    }
                })
                .collect::<Vec<_>>();
            let expected = poly.decompose_fold::<D>(&window, POSITIONS, 2, 4).unwrap();
            assert_eq!(
                chunk.centered_coeffs_flat(),
                expected.centered_coeffs_flat()
            );
        }
        let combined = crate::opaque::aggregate_decompose_fold_witnesses::<D>(
            chunks
                .iter()
                .map(|chunk| Ok::<_, AkitaError>(chunk.clone())),
        )
        .unwrap();
        assert_eq!(
            combined.centered_coeffs_flat(),
            global.centered_coeffs_flat()
        );
    }
}

fn ring<const D: usize>(offset: u64) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(std::array::from_fn(|idx| {
        F::from_u64(offset + idx as u64 + 1)
    }))
}

#[test]
fn ring_fold_matches_dense_multiplication_reference() {
    const D: usize = 8;
    let coeffs = (0..2).map(|idx| ring::<D>(10 * idx)).collect::<Vec<_>>();
    let poly = DensePoly::<F>::from_ring_coeffs(coeffs.clone()).unwrap();
    let scalars = vec![
        ring::<D>(100),
        ring::<D>(200),
        ring::<D>(300),
        ring::<D>(400),
    ];
    let got = poly.fold_blocks_ring(&scalars, 4).unwrap();
    let expected = coeffs
        .chunks(4)
        .map(|block| {
            block
                .iter()
                .zip(scalars.iter())
                .fold(CyclotomicRing::<F, D>::zero(), |acc, (coeff, scalar)| {
                    acc + (*coeff * *scalar)
                })
        })
        .collect::<Vec<_>>();

    assert_eq!(got, expected);
}

#[test]
fn dense_constructor_reuses_owned_evaluation_buffer() {
    let evals = (0..2048).map(F::from_u64).collect::<Vec<_>>();
    let allocation = evals.as_ptr();
    let poly = DensePoly::<F>::from_field_evals(11, evals).unwrap();
    assert_eq!(poly.field_coeffs().as_ptr(), allocation);
}

#[test]
fn dense_source_has_exact_views_across_supported_ring_dimensions() {
    let evals = (1..=32).map(F::from_u64).collect::<Vec<_>>();
    let poly = DensePoly::<F>::from_field_evals(5, evals.clone()).unwrap();

    fn assert_view<const D: usize>(poly: &DensePoly<F>, evals: &[F]) {
        let rings = poly.ring_coeffs::<D>().expect("supported dense view");
        let flat = rings
            .iter()
            .flat_map(|ring| ring.coefficients().iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(&flat[..evals.len()], evals);
        assert!(flat[evals.len()..].iter().all(|value| *value == F::zero()));
    }

    assert_view::<64>(&poly, &evals);
    assert_view::<128>(&poly, &evals);
    assert_view::<256>(&poly, &evals);
    assert_view::<512>(&poly, &evals);
    assert_view::<1024>(&poly, &evals);
}

#[test]
fn dense_ring_constructor_rejects_empty_and_irregular_sources() {
    const D: usize = 512;
    // 96 rings previously exposed only the first 32 of the supplied rings.
    for num_rings in [0, 3, 96] {
        let result =
            DensePoly::<F>::from_ring_coeffs(vec![CyclotomicRing::<F, D>::zero(); num_rings]);
        assert!(matches!(result, Err(AkitaError::InvalidInput(_))));
    }
    let zero_degree = DensePoly::<F>::from_ring_coeffs(vec![CyclotomicRing::<F, 0>::zero()]);
    assert!(matches!(zero_degree, Err(AkitaError::InvalidInput(_))));
    let odd_degree = DensePoly::<F>::from_ring_coeffs(vec![CyclotomicRing::<F, 3>::zero(); 2]);
    assert!(matches!(odd_degree, Err(AkitaError::InvalidInput(_))));
}

#[test]
fn dense_ring_constructor_preserves_the_entire_commitment_source_and_bound_scan() {
    const D: usize = 512;
    let modulus = (-F::from_u64(1)).to_u128_checked().unwrap() + 1;
    // Exercise both small-i8 and full-field storage, including physical padding.
    for num_rings in [1, 32, 64] {
        for reach in [127u64, 128] {
            let mut evals = (0..num_rings * D)
                .map(|idx| F::from_u64((idx % 32) as u64))
                .collect::<Vec<_>>();
            evals[num_rings * D - 2] = -F::from_u64(reach);
            evals[num_rings * D - 1] = F::from_u64(reach);
            let rings = evals
                .chunks_exact(D)
                .map(|chunk| CyclotomicRing::<F, D>::from_coefficients(chunk.try_into().unwrap()))
                .collect::<Vec<_>>();
            let poly = DensePoly::<F>::from_ring_coeffs(rings.clone()).unwrap();

            assert_eq!(1usize << RootPolyMeta::num_vars(&poly), evals.len());
            assert_eq!(&poly.field_coeffs()[..evals.len()], evals);
            assert_eq!(poly.ring_coeffs::<D>().unwrap(), rings);
            let descriptor = <DensePoly<F> as CommitmentSource<F>>::descriptor(&poly).unwrap();
            assert_eq!(descriptor.num_vars(), RootPolyMeta::num_vars(&poly));
            assert_eq!(descriptor.live_coefficient_len(), evals.len());
            assert_eq!(
                <DensePoly<F> as CommitmentSource<F>>::committed_centered_reach(
                    &poly,
                    modulus,
                    modulus / 2,
                )
                .unwrap(),
                (u128::from(reach), u128::from(reach)),
            );
            assert_eq!(
                poly,
                DensePoly::from_field_evals(RootPolyMeta::num_vars(&poly), evals).unwrap(),
            );
        }
    }
}

#[test]
fn dense_field_constructor_rejects_unrepresentable_arities() {
    for num_vars in [usize::BITS as usize, usize::MAX] {
        let result = DensePoly::<F>::from_field_evals(num_vars, vec![F::zero()]);
        assert!(matches!(result, Err(AkitaError::InvalidInput(_))));
    }
}

#[cfg(target_pointer_width = "64")]
#[test]
fn dense_field_constructor_rejects_arity_that_truncates_to_a_valid_shift() {
    let result = DensePoly::<F>::from_field_evals((1usize << 32) + 14, vec![F::zero(); 1 << 14]);
    assert!(matches!(result, Err(AkitaError::InvalidInput(_))));
}

#[test]
fn batch_fold_rejects_mixed_extents_and_count_mismatch() {
    use crate::opaque::{
        CpuBackend, DecomposeFoldBatchPlan, OpeningBatchKernel, RootOpeningSource,
    };
    use akita_challenges::SparseChallenge;

    const D: usize = 64;
    let polys = [
        DensePoly::from_field_evals(6, vec![F::from_u64(1); 1 << 6]).unwrap(),
        DensePoly::from_field_evals(7, vec![F::from_u64(1); 1 << 7]).unwrap(),
    ];
    let challenges = vec![
        SparseChallenge {
            positions: vec![0].into(),
            coeffs: vec![1].into(),
        };
        2
    ];
    let backend = CpuBackend::<F, F>::for_arithmetic_tests();
    let run = |refs: &[&DensePoly<F>]| {
        OpeningBatchKernel::decompose_fold_batch(
            &backend,
            None,
            <DensePoly<F> as RootOpeningSource<F, D>>::opening_batch(refs).unwrap(),
            DecomposeFoldBatchPlan::Sparse {
                challenges: &challenges,
                num_positions_per_block: 1,
                num_digits: 1,
                log_basis: 1,
            },
        )
    };

    assert!(matches!(
        run(&[&polys[0], &polys[1]]),
        Err(AkitaError::InvalidInput(_))
    ));
    assert!(matches!(
        run(&[&polys[0], &polys[0], &polys[0]]),
        Err(AkitaError::InvalidInput(_))
    ));
}

#[test]
fn scalar_fold_rejects_short_and_excess_challenges() {
    use crate::opaque::{CpuBackend, DecomposeFoldPlan, OpeningFoldKernel, RootOpeningSource};
    use akita_challenges::SparseChallenge;

    const D: usize = 8;
    // Two rings at one position per block give two live blocks.
    let poly = DensePoly::from_ring_coeffs(vec![ring::<D>(0), ring::<D>(10)]).unwrap();
    let challenge = SparseChallenge {
        positions: vec![0].into(),
        coeffs: vec![1].into(),
    };
    let backend = CpuBackend::<F, F>::for_arithmetic_tests();
    let run = |count: usize| {
        let challenges = vec![challenge.clone(); count];
        OpeningFoldKernel::decompose_fold(
            &backend,
            None,
            <DensePoly<F> as RootOpeningSource<F, D>>::opening_view(&poly).unwrap(),
            DecomposeFoldPlan {
                challenges: &challenges,
                num_positions_per_block: 1,
                num_digits: 1,
                log_basis: 6,
            },
        )
    };

    assert!(run(2).is_ok());
    for count in [0, 1, 3] {
        assert!(
            matches!(
                run(count),
                Err(AkitaError::InvalidSize { expected: 2, actual }) if actual == count
            ),
            "{count} challenges for two live blocks must be rejected"
        );
    }
}

// An unsupported basis or a zero digit count must be rejected before the
// digit cache build or the decomposition setup, with a cold or a populated cache.
#[test]
fn fold_rejects_unsupported_digit_shapes_with_cold_and_populated_caches() {
    use crate::opaque::{CpuBackend, DecomposeFoldPlan, OpeningFoldKernel, RootOpeningSource};
    use akita_challenges::SparseChallenge;

    const D: usize = 8;
    let challenges = vec![
        SparseChallenge {
            positions: vec![0].into(),
            coeffs: vec![1].into(),
        };
        2
    ];
    let backend = CpuBackend::<F, F>::for_arithmetic_tests();
    let run = |poly: &DensePoly<F>, num_digits: usize, log_basis: u32| {
        OpeningFoldKernel::decompose_fold(
            &backend,
            None,
            <DensePoly<F> as RootOpeningSource<F, D>>::opening_view(poly).unwrap(),
            DecomposeFoldPlan {
                challenges: &challenges,
                num_positions_per_block: 1,
                num_digits,
                log_basis,
            },
        )
    };

    for populate in [false, true] {
        for (num_digits, log_basis) in [(1, 0), (0, 4), (1, 17)] {
            let poly = DensePoly::from_ring_coeffs(vec![ring::<D>(0), ring::<D>(10)]).unwrap();
            if populate {
                run(&poly, 2, 4).unwrap();
            }
            assert!(
                matches!(
                    run(&poly, num_digits, log_basis),
                    Err(AkitaError::InvalidInput(_))
                ),
                "({num_digits}, {log_basis}) with populated cache = {populate}"
            );
            assert!(
                matches!(
                    poly.decompose_fold_chunked::<D>(
                        &challenges,
                        &[0..1, 1..2],
                        1,
                        num_digits,
                        log_basis
                    ),
                    Err(AkitaError::InvalidInput(_))
                ),
                "chunked ({num_digits}, {log_basis}) with populated cache = {populate}"
            );
        }
    }
}

// A +1 monomial challenge must leave every single balanced digit unchanged.
// The expectation is the input integer itself, independent of decomposition.
#[test]
fn single_digit_fold_preserves_signed_i8_i16_boundaries() {
    use akita_challenges::SparseChallenge;
    use jolt_field::Prime64Offset59;
    const D: usize = 128;
    let challenge = SparseChallenge {
        positions: vec![0].into(),
        coeffs: vec![1].into(),
    };
    let mut mismatches = Vec::new();
    for log_basis in [8_u32, 9] {
        let half = 1_i64 << (log_basis - 1);
        for value in [127_i64, 128, 255, -128, -129, -256] {
            if !(-half..half).contains(&value) {
                continue; // Only admissible balanced digits belong in this oracle.
            }
            let mut coefficients = vec![Prime64Offset59::zero(); D];
            coefficients[0] = Prime64Offset59::from_i64(value);
            let poly = DensePoly::from_field_evals(7, coefficients).unwrap();
            let actual = poly
                .decompose_fold::<D>(std::slice::from_ref(&challenge), 1, 1, log_basis)
                .unwrap();
            let mut expected = vec![0_i32; D];
            expected[0] = value as i32;
            if actual.centered_coeffs_flat() != expected {
                mismatches.push((log_basis, value, actual.centered_coeffs_flat()[0]));
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "(basis, expected, actual): {mismatches:?}"
    );
}

#[test]
fn fold_operations_reject_invalid_ring_views() {
    let poly = DensePoly::<F>::from_field_evals(0, vec![F::zero()]).unwrap();
    assert!(matches!(
        poly.fold_blocks::<0>(&[], 1),
        Err(AkitaError::InvalidInput(_))
    ));
    assert!(matches!(
        poly.fold_blocks_ring::<0>(&[], 1),
        Err(AkitaError::InvalidInput(_))
    ));
    assert!(matches!(
        poly.decompose_fold::<0>(&[], 1, 1, 1),
        Err(AkitaError::InvalidInput(_))
    ));
    assert!(matches!(
        poly.decompose_fold_chunked::<0>(&[], &[], 1, 1, 1),
        Err(AkitaError::InvalidInput(_))
    ));
}

#[test]
fn single_digit_fold_validates_challenges_before_both_dense_row_loops() {
    use akita_challenges::SparseChallenge;

    for value in [1, 256] {
        let poly = DensePoly::<F>::from_field_evals(6, vec![F::from_u64(value); 64]).unwrap();
        // Fix the digit cache at another dimension, exercising the two
        // single-digit fallback loops (small-i8 mirror and live field rows).
        assert!(poly.digit_planes_for::<64>(1, 4).is_some());
        assert_eq!(poly.small_i8_coeffs.is_some(), value == 1);
        let challenges = [SparseChallenge {
            positions: vec![128].into(),
            coeffs: vec![1].into(),
        }];
        assert_eq!(
            poly.decompose_fold::<128>(&challenges, 1, 1, 4)
                .unwrap_err(),
            AkitaError::InvalidInput(
                "sparse fold challenge position exceeds the ring dimension".into()
            ),
        );
    }
}

#[test]
fn dense_single_digit_fold_preserves_unused_challenges_and_empty_positions() {
    use akita_challenges::SparseChallenge;

    let poly = DensePoly::<F>::from_field_evals(6, vec![F::from_u64(1); 64]).unwrap();
    assert!(poly.digit_planes_for::<64>(1, 4).is_some());
    let valid = SparseChallenge {
        positions: vec![0].into(),
        coeffs: vec![1].into(),
    };
    let invalid = SparseChallenge {
        positions: vec![128].into(),
        coeffs: vec![1].into(),
    };
    let expected = poly
        .decompose_fold::<128>(std::slice::from_ref(&valid), 1, 1, 4)
        .unwrap();
    let actual = poly
        .decompose_fold::<128>(&[valid, invalid.clone()], 1, 1, 4)
        .unwrap();
    assert_eq!(
        actual.centered_coeffs_flat(),
        expected.centered_coeffs_flat()
    );
    assert!(poly
        .decompose_fold::<128>(&[invalid], 0, 1, 4)
        .unwrap()
        .centered_coeffs_flat()
        .is_empty());
}

mod batch_only {
    use super::*;
    use crate::commitment::{
        AvailablePolynomialTypes, CommitSourceDescriptor, PolynomialRepresentation,
        PolynomialTypeSelection,
    };
    use crate::opaque::*;
    use crate::sources::poly::SourceCoefficients;

    struct BatchOnlySource(DensePoly<F>);

    impl RootPolyMeta<F> for BatchOnlySource {
        fn num_vars(&self) -> usize {
            RootPolyMeta::num_vars(&self.0)
        }
    }
    impl<const D: usize> RootPolyShape<F, D> for BatchOnlySource {
        fn num_ring_elems(&self) -> usize {
            RootPolyShape::<F, D>::num_ring_elems(&self.0)
        }
    }
    impl<const D: usize> RootOpeningSource<F, D> for BatchOnlySource {
        type OpeningView<'a> = ();
        type OpeningBatchView<'a> = Vec<&'a DensePoly<F>>;
        fn opening_view(&self) -> Result<(), AkitaError> {
            Err(AkitaError::InvalidInput("batch-only source".into()))
        }
        fn opening_batch<'a>(polys: &'a [&'a Self]) -> Result<Vec<&'a DensePoly<F>>, AkitaError> {
            Ok(polys.iter().map(|poly| &poly.0).collect())
        }
    }
    impl SourceCoefficients<F> for BatchOnlySource {
        fn source_coefficients(&self) -> Result<std::borrow::Cow<'_, [F]>, AkitaError> {
            self.0.source_coefficients()
        }
    }
    impl CommitmentSource<F> for BatchOnlySource {
        fn descriptor(&self) -> Result<CommitSourceDescriptor, AkitaError> {
            self.0.descriptor()
        }
        fn committed_centered_reach(
            &self,
            modulus: u128,
            threshold: u128,
        ) -> Result<(u128, u128), AkitaError> {
            self.0.committed_centered_reach(modulus, threshold)
        }
        fn available_polynomial_types(
            &self,
            plan: &CommitInnerPlan,
        ) -> Result<AvailablePolynomialTypes, AkitaError> {
            self.0.available_polynomial_types(plan)
        }
        fn represent_as(
            &self,
            selected: PolynomialTypeSelection,
            plan: &CommitInnerPlan,
        ) -> Result<PolynomialRepresentation<'_, F>, AkitaError> {
            self.0.represent_as(selected, plan)
        }
    }
    impl<const D: usize> OpeningBatchKernel<Vec<&DensePoly<F>>, F, D> for CpuBackend<F, F> {
        fn evaluate_and_fold_batch(
            &self,
            prepared: Option<&Self::PreparedSetup>,
            source: Vec<&DensePoly<F>>,
            plan: OpeningFoldPlan<'_, F>,
        ) -> Result<Vec<OpeningFoldOutput<F, D>>, AkitaError> {
            self.evaluate_and_fold_batch(
                prepared,
                <DensePoly<F> as RootOpeningSource<F, D>>::opening_batch(&source)?,
                plan,
            )
        }
        fn decompose_fold_batch(
            &self,
            prepared: Option<&Self::PreparedSetup>,
            source: Vec<&DensePoly<F>>,
            plan: DecomposeFoldBatchPlan<'_>,
        ) -> Result<CpuFoldResponses, AkitaError> {
            self.decompose_fold_batch(
                prepared,
                <DensePoly<F> as RootOpeningSource<F, D>>::opening_batch(&source)?,
                plan,
            )
        }
    }
    impl<const D: usize> SubringCoefficientPackingBatchKernel<Vec<&DensePoly<F>>, F, F, D>
        for CpuBackend<F, F>
    {
        fn coefficient_packing_partials_batch(
            &self,
            prepared: Option<&Self::PreparedSetup>,
            source: Vec<&DensePoly<F>>,
            plan: SubringCoefficientPackingPlan<'_, F>,
        ) -> Result<Vec<SubringCoefficientPackingPartials<F>>, AkitaError> {
            self.coefficient_packing_partials_batch(
                prepared,
                <DensePoly<F> as RootOpeningSource<F, D>>::opening_batch(&source)?,
                plan,
            )
        }
    }

    #[test]
    fn import_source_admits_batch_only_opening_kernels() {
        let setup = crate::AkitaProverSetup::<F>::generate_with_capacity(
            9,
            1,
            akita_params::SetupMatrixCapacity {
                num_field_elements: 4096,
            },
        )
        .unwrap();
        let backend = CpuBackend::<F, F>::new(setup.expanded.clone()).unwrap();
        let poly = DensePoly::from_field_evals(9, vec![F::from_u64(1); 512]).unwrap();
        // No OpeningFoldKernel<()> implementation exists for this source's singleton view.
        assert!(backend.import_source(vec![BatchOnlySource(poly)]).is_ok());
    }
}

#[test]
fn small_i8_bounds_span_parallel_chunks() {
    use jolt_field::Prime64Offset59;

    const N: usize = 1 << 16;
    let mut cells = vec![0i8; N];
    cells[3] = -100;
    cells[N / 2] = 64;
    cells[N - 5] = 127;
    let poly = DensePoly::<Prime64Offset59>::from_small_i8(16, cells).unwrap();
    assert_eq!(poly.small_mirror_bounds(), Some((-100, 127)));
}

#[test]
fn small_i16_mirror_spans_parallel_chunks() {
    use jolt_field::{Prime64Offset59, Ring};

    const N: usize = 1 << 16;
    let mut cells = vec![0i32; N];
    cells[1] = -32_768;
    cells[N / 2] = 17;
    cells[N - 2] = 32_767;
    let evals: Vec<Prime64Offset59> = cells
        .iter()
        .map(|&cell| Prime64Offset59::from_i64(i64::from(cell)))
        .collect();
    let poly = DensePoly::<Prime64Offset59>::from_field_evals(16, evals)
        .unwrap()
        .attach_small_i16(&cells)
        .unwrap();
    let mirror = poly.small_i16_mirror().unwrap();
    assert_eq!(mirror.len(), N);
    assert_eq!(mirror[1], -32_768);
    assert_eq!(mirror[N / 2], 17);
    assert_eq!(mirror[N - 2], 32_767);
    assert!(mirror
        .iter()
        .enumerate()
        .all(|(index, &value)| { matches!(index, 1 | 32_768 | 65_534) || value == 0 }));
    assert_eq!(poly.small_mirror_bounds(), Some((-32_768, 32_767)));
}

#[test]
fn small_i16_mirror_rejects_the_first_out_of_range_cell() {
    use jolt_field::{Prime64Offset59, Ring};

    const N: usize = 1 << 16;
    let mut cells = vec![0i32; N];
    cells[7] = 32_768;
    // A second bad cell in the next mirror chunk: the parallel chunk fold must report the
    // earlier one, not whichever chunk finishes first.
    cells[(1 << 15) + 3] = -32_769;
    let evals = vec![Prime64Offset59::from_i64(0); N];
    let poly = DensePoly::<Prime64Offset59>::from_field_evals(16, evals).unwrap();
    let error = poly.attach_small_i16(&cells).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("32768"), "{message}");
    assert!(
        !message.contains("32769"),
        "the first bad cell must win: {message}"
    );
}
