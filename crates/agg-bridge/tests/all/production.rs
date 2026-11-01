//! The production shape against the independent `agg-witness` reference, the seed-derived
//! matrices, the weight-2 survival rate and the final-vector encoding.

use agg_bridge::parity::{self, ParityWitness};
use agg_bridge::seed::{derive_matrices, derive_matrix, Level};
use agg_bridge::{final_vector, weights, Shape};
use agg_witness::bridge::{self as reference, BridgeMatrices};
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
        w.y_bits(),
        reference::gf2_composite(&geom, &array, matrices)
    );
    parity::check(&shape, &views, matrices, &w).unwrap();
    check_direct(&shape, &array, matrices, &w);
    let domains = shape.domains(geom.main_len());
    assert_eq!(w.y.len(), domains.d3_len);
    assert_eq!(w.kappa1[0].len(), 1 << geom.d1());
    w
}

/// The packed tables equal the general witness in narrow and stacked balanced form.
fn check_direct(shape: &Shape, lanes: &[&[i8]; W1_BITS], matrices: &BridgeMatrices, w: &ParityWitness) {
    const SHIFT: i32 = 1 << 15;
    let direct = parity::compute_direct(shape, lanes, matrices, SHIFT, None).unwrap().expect("bit inputs");
    let narrow = |t: &[i64]| t.iter().map(|&b| b as i8).collect::<Vec<_>>();
    let stacked = |columns: &[Vec<i64>], slots: usize| {
        let cells = columns[0].len();
        let mut out = vec![-SHIFT; slots * cells];
        for (slot, column) in out.chunks_mut(cells).zip(columns) {
            for (o, &k) in slot.iter_mut().zip(column) {
                *o = k as i32 - SHIFT;
            }
        }
        out
    };
    for t in 0..W1_BITS {
        assert_eq!(direct.y1[t], narrow(&w.y1[t]), "y1 lane {t}");
        assert_eq!(direct.y2[t], narrow(&w.y2[t]), "y2 lane {t}");
    }
    assert_eq!(direct.level1_carries, stacked(&w.kappa1, 8), "level-1 carries");
    assert_eq!(direct.level2_carries, stacked(&w.kappa2, 8), "level-2 carries");
    assert_eq!(direct.level3_carries, stacked(std::slice::from_ref(&w.kappa3), 1), "level-3 carries");
    assert_eq!(direct.y, w.y_bits(), "Y");
    // A non-bit input takes the general path.
    let mut tampered = lanes[2].to_vec();
    tampered[0] = 2;
    let mut bad = *lanes;
    bad[2] = &tampered;
    assert!(parity::compute_direct(shape, &bad, matrices, SHIFT, None).unwrap().is_none());
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

/// The four-lane shape of ML-DSA-65 and ML-DSA-87: the packed path equals the general
/// witness in narrow and stacked form, for one zero-padded level-1 block, two blocks and two
/// segments, with the bits of rows 6 and 7 of every record zero (ML-DSA-65); a non-bit input
/// takes the general path.
#[test]
fn four_lane_packed_tables_match_the_general_witness() {
    const SHIFT: i32 = 1 << 15;
    let shape = Shape::PRODUCTION_W4;
    let matrices = agg_bridge::seed::derive_matrices_for(&shape, &[0x65; 32]);
    let mut rng = SplitMix64::new(65);
    for records in [5usize, 300, (1 << 13) + 40] {
        let len = (records.next_power_of_two() * 2048).max(1 << 19);
        let lanes: Vec<Vec<i8>> = (0..4)
            .map(|_| {
                (0..len)
                    .map(|x| i8::from(x / 2048 < records && x % 2048 < 1536 && rng.next_u64() & 1 == 1))
                    .collect()
            })
            .collect();
        let views: [&[i8]; 4] = std::array::from_fn(|t| lanes[t].as_slice());
        let w = parity::compute(&shape, &views, &matrices).unwrap();
        let direct = parity::compute_direct_four(&shape, &views, &matrices, SHIFT, [4, 8]).unwrap().expect("bit inputs");
        let narrow = |t: &[i64]| t.iter().map(|&b| b as i8).collect::<Vec<_>>();
        let stacked = |columns: &[Vec<i64>], slots: usize| {
            let cells = columns[0].len();
            let mut out = vec![-SHIFT; slots * cells];
            for (slot, column) in out.chunks_mut(cells).zip(columns) {
                for (o, &k) in slot.iter_mut().zip(column) {
                    *o = k as i32 - SHIFT;
                }
            }
            out
        };
        for t in 0..4 {
            assert_eq!(direct.y1[t], narrow(&w.y1[t]), "y1 lane {t}");
            assert_eq!(direct.y2[t], narrow(&w.y2[t]), "y2 lane {t}");
        }
        assert_eq!(direct.level1_carries, stacked(&w.kappa1, 4));
        assert_eq!(direct.level2_carries, stacked(&w.kappa2, 8));
        assert_eq!(direct.level3_carries, stacked(std::slice::from_ref(&w.kappa3), 1));
        assert_eq!(direct.y, w.y_bits(), "Y at {records} records");
        let mut tampered = lanes[1].clone();
        tampered[3] = -1;
        let mut bad = views;
        bad[1] = &tampered;
        assert!(parity::compute_direct_four(&shape, &bad, &matrices, SHIFT, [4, 8]).unwrap().is_none());
    }
}
