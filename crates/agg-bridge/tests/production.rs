//! The production shape against the independent `agg-witness` reference, the seed-derived
//! matrices, the weight-2 survival rate and the final-vector encoding.

use agg_bridge::parity::{self, ParityWitness};
use agg_bridge::seed::{derive_matrices, derive_matrix, Level};
use agg_bridge::{carry, final_vector, weights, Shape};
use agg_witness::bridge::{self as reference, BridgeMatrices, BridgeTables};
use agg_witness::gen::SplitMix64;
use agg_witness::padding::Geometry;
use agg_witness::params::W1_BITS;
use jolt_field::{Prime64Offset59 as F, Ring, Zero};

/// Random bits on the live records of `geom`, zero on padded records.
fn bit_lanes(geom: &Geometry, seed: u64) -> Vec<Vec<i8>> {
    let mut rng = SplitMix64::new(seed);
    (0..W1_BITS)
        .map(|_| {
            (0..geom.main_len())
                .map(|x| {
                    if x >> 10 < geom.n {
                        (rng.next_u64() & 1) as i8
                    } else {
                        0
                    }
                })
                .collect()
        })
        .collect()
}

fn check_against_reference(n: usize, matrices: &BridgeMatrices, seed: u64) -> ParityWitness {
    let shape = Shape::PRODUCTION;
    let geom = Geometry::new(n).unwrap();
    let lanes = bit_lanes(&geom, seed);
    let views: Vec<&[i8]> = lanes.iter().map(Vec::as_slice).collect();
    let w = parity::compute(&shape, &views, matrices).unwrap();
    let array: [&[i8]; W1_BITS] = std::array::from_fn(|t| views[t]);
    let expected = reference::compute(&geom, &array, matrices).unwrap();
    assert_eq!(w.clone().into_witness().unwrap(), expected, "N = {n}");
    assert_eq!(
        carry::group_tables(&shape, &w).unwrap(),
        BridgeTables::from_witness(&expected).unwrap()
    );
    assert_eq!(
        w.y_bits(),
        reference::gf2_composite(&geom, &array, matrices)
    );
    parity::check(&shape, &views, matrices, &w).unwrap();
    let domains = shape.domains(geom.main_len());
    assert_eq!(w.y.len(), domains.d3_len);
    assert_eq!(w.kappa1[0].len(), 1 << geom.d1());
    w
}

#[test]
fn production_tables_match_the_agg_witness_reference() {
    let matrices = BridgeMatrices::random_for_tests(7);
    // One block (d_sig = 3, zero-padded in-block positions), two level-1 blocks, and two
    // segments (d_sig = 14).
    for (n, seed) in [(5, 1), (600, 2), ((1 << 13) + 700, 3)] {
        check_against_reference(n, &matrices, seed);
    }
}

#[test]
fn seed_derived_production_matrices_pass_the_honest_abort_and_the_identities() {
    let matrices = derive_matrices(&[0x42; 32]);
    let shape = Shape::PRODUCTION;
    assert!(shape.matches(&matrices) && shape.row_weights_ok(&matrices));
    // Used-column row weights: Bin(J_t^used, k_t / m_t) with means 896, 224, 216.
    let means: Vec<f64> = shape
        .levels
        .iter()
        .zip([&matrices.m1, &matrices.m2, &matrices.m3])
        .map(|(level, m)| {
            let w = level.used_row_weights(m);
            w.iter().map(|&x| f64::from(x)).sum::<f64>() / w.len() as f64
        })
        .collect();
    assert_eq!(means, vec![896.0, 224.0, 216.0]);
    check_against_reference(300, &matrices, 4);
}

#[test]
fn production_weights_agree_with_their_extensions() {
    // The table at Boolean points and the verifier's formula at a random point, d_sig = 10.
    let shape = Shape::PRODUCTION;
    let matrices = BridgeMatrices::random_for_tests(11);
    let geom = Geometry::new(600).unwrap();
    let domains = shape.domains(geom.main_len());
    let mut rng = SplitMix64::new(12);
    let mut point =
        |len: usize| -> Vec<F> { (0..len).map(|_| F::from_u64(rng.next_u64())).collect() };
    let mle = |table: &[F], p: &[F]| {
        weights::eq_table(p)
            .iter()
            .zip(table)
            .fold(F::zero(), |a, (&w, &v)| a + w * v)
    };
    let (p1, r) = (point(domains.d1_vars()), point(domains.input_vars()));
    let table = weights::level1_table(&shape, &domains, &matrices.m1, &p1);
    assert_eq!(
        weights::level1_at(&shape, &domains, &matrices.m1, &p1, &r),
        mle(&table, &r)
    );
    let (p2, q) = (point(domains.d2_vars()), point(domains.d1_vars()));
    let table = weights::level2_table(&shape, &domains, &matrices.m2, &p2);
    assert_eq!(
        weights::level2_at(&shape, &domains, &matrices.m2, &p2, &q),
        mle(&table, &q)
    );
    let (p3, q2) = (point(domains.d3_vars()), point(domains.d2_vars()));
    let tables = weights::level3_tables(&shape, &domains, &matrices.m3, &p3);
    let at = weights::level3_at(&shape, &domains, &matrices.m3, &p3, &q2);
    for t in 0..W1_BITS {
        assert_eq!(at[t], mle(&tables[t], &q2));
    }
}

/// At `(m, k) = (64, 3)`, `2 x 10^6` column pairs drawn by the seed
/// derivation; the pairs with equal row sets (expected `2 x 10^6 / binom(64, 3) = 48.0`)
/// lie within three standard deviations, `[27, 69]`.
#[test]
fn weight_two_survival_rate_is_one_over_binom_64_3() {
    const PAIRS: usize = 2_000_000;
    let level = Level {
        t: 13,
        rows_log2: 6,
        cols: 2 * PAIRS,
        k: 3,
    };
    let matrix = derive_matrix(&[0x13; 32], &level);
    let sorted = |c: usize| {
        let mut rows = [0_u32; 3];
        rows.copy_from_slice(matrix.column(c));
        rows.sort_unstable();
        rows
    };
    let equal = (0..PAIRS)
        .filter(|&i| sorted(2 * i) == sorted(2 * i + 1))
        .count();
    eprintln!("equal row sets: {equal} of {PAIRS} pairs (expected 48.0)");
    assert!((27..=69).contains(&equal), "{equal}");
}

#[test]
fn final_vectors_pack_parse_and_evaluate() {
    let shape = Shape::PRODUCTION;
    assert_eq!(final_vector::segment_bytes(&shape), 512);
    assert_eq!(final_vector::segment_bytes(&Shape::TOY), 2);
    let mut rng = SplitMix64::new(21);
    let bits: Vec<u8> = (0..2 * 4096).map(|_| (rng.next_u64() & 1) as u8).collect();
    let packed = final_vector::pack(&bits);
    assert_eq!(packed.len(), 1024);
    assert_eq!(packed[0] & 1, bits[0]);
    assert_eq!((packed[1] >> 2) & 1, bits[10]);
    assert_eq!(final_vector::parse(&shape, &packed, 2).unwrap(), bits);
    let short = final_vector::parse(&shape, &packed[..1023], 2).unwrap_err();
    assert_eq!((short.expected, short.actual), (1024, 1023));
    assert!(final_vector::parse(&shape, &packed, 1).is_err());
    let point: Vec<F> = (0..13).map(|_| F::from_u64(rng.next_u64())).collect();
    let table: Vec<F> = bits.iter().map(|&b| F::from_u64(u64::from(b))).collect();
    let mle = weights::eq_table(&point)
        .iter()
        .zip(&table)
        .fold(F::zero(), |a, (&w, &v)| a + w * v);
    assert_eq!(final_vector::evaluate(&bits, &point), mle);
    let (y, k) = (F::from_u64(5), F::from_u64(7));
    assert_eq!(final_vector::level3_claimed_sum(y, k), F::from_u64(19));
}
