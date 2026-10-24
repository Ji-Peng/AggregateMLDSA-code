//! The toy instance of docs/bridge.md Section 9 items 2 and 3: `Shape::TOY`, `L = 2^12`
//! input bits from a seeded PRNG, matrices from the seed derivation; every Fp-side table
//! against a direct recomputation with dense `{0, 1}` matrices, the weights against their
//! dense definitions, the GF(2) side against a dense composite, and single-bit tampering.

use agg_bridge::parity::{self, IdentityError, ParityWitness};
use agg_bridge::seed::derive_matrices_for;
use agg_bridge::{carry, final_vector, gf2_walker, weights, Shape};
use agg_witness::bridge::{BridgeMatrices, SparseMatrix};
use agg_witness::gen::SplitMix64;
use jolt_field::{Prime64Offset59 as F, Ring, Zero};

const SEED: [u8; 32] = [0x5a; 32];
const L: usize = 1 << 12;

fn dense(m: &SparseMatrix) -> Vec<Vec<i64>> {
    let mut d = vec![vec![0_i64; m.cols]; m.rows];
    for (zeta, column) in m.entries.chunks(m.k).enumerate() {
        for &row in column {
            d[row as usize][zeta] = 1;
        }
    }
    d
}

/// `Y` from the GF(2) side (`gf2_walker::composite` over the lane-major packed bits).
fn gf2(m: &BridgeMatrices, bits: &[i8]) -> Vec<u8> {
    let layout = gf2_walker::SegmentLayout::new(Shape::TOY, L);
    let mut packed = vec![0_u64; L / 64];
    for (c, &b) in bits.iter().enumerate() {
        packed[c / 64] |= u64::from(b as u8 & 1) << (c % 64);
    }
    gf2_walker::composite(&layout, m, &packed)
}

/// The level-1 and level-2 outputs over GF(2) by a forward XOR walk on dense matrices.
fn gf2_levels(m: &BridgeMatrices, bits: &[i8]) -> (Vec<i64>, Vec<i64>) {
    let shape = Shape::TOY;
    let [l1, l2, _] = &shape.levels;
    let (d1, d2) = (dense(&m.m1), dense(&m.m2));
    let y1: Vec<i64> = (0..shape.blocks1 * l1.rows)
        .map(|cell| {
            let (rho, beta) = (cell % l1.rows, cell / l1.rows);
            (0..l1.cols).fold(0, |acc, zeta| {
                acc ^ (d1[rho][zeta] & i64::from(bits[zeta + l1.cols * beta]))
            })
        })
        .collect();
    let y2 = (0..l2.rows)
        .map(|rho| (0..l2.used).fold(0, |acc, zeta| acc ^ (d2[rho][zeta] & y1[zeta])))
        .collect();
    (y1, y2)
}

fn random_bits(seed: u64) -> Vec<i8> {
    let mut rng = SplitMix64::new(seed);
    (0..L).map(|_| (rng.next_u64() & 1) as i8).collect()
}

fn field(rng: &mut SplitMix64) -> F {
    F::from_u64(rng.next_u64())
}

fn point(rng: &mut SplitMix64, len: usize) -> Vec<F> {
    (0..len).map(|_| field(rng)).collect()
}

fn mle(table: &[F], point: &[F]) -> F {
    weights::eq_table(point)
        .iter()
        .zip(table)
        .fold(F::zero(), |acc, (&w, &v)| acc + w * v)
}

fn ints(table: &[i64]) -> Vec<F> {
    table.iter().map(|&v| F::from_i64(v)).collect()
}

/// Dense integer sums of the three toy levels (one lane, one segment), `i64` arithmetic.
fn dense_sums(m: &BridgeMatrices, bits: &[i8], w: &ParityWitness) -> [Vec<i64>; 3] {
    let shape = Shape::TOY;
    let [l1, l2, l3] = &shape.levels;
    let (d1, d2, d3) = (dense(&m.m1), dense(&m.m2), dense(&m.m3));
    let mut s1 = vec![0_i64; shape.blocks1 * l1.rows];
    for beta in 0..shape.blocks1 {
        for rho in 0..l1.rows {
            s1[rho + l1.rows * beta] = (0..l1.cols)
                .map(|zeta| d1[rho][zeta] * i64::from(bits[zeta + l1.cols * beta]))
                .sum();
        }
    }
    let s2: Vec<i64> = (0..l2.rows)
        .map(|rho| (0..l2.used).map(|zeta| d2[rho][zeta] * w.y1[0][zeta]).sum())
        .collect();
    let s3: Vec<i64> = (0..l3.rows)
        .map(|rho| (0..l2.rows).map(|rho2| d3[rho][rho2] * w.y2[0][rho2]).sum())
        .collect();
    [s1, s2, s3]
}

/// The dense GF(2) composite `M_3 M_2 M_1` (16 rows by 4,096 columns) applied to `bits`.
fn dense_composite(m: &BridgeMatrices, bits: &[i8]) -> Vec<u8> {
    let shape = Shape::TOY;
    let [l1, l2, l3] = &shape.levels;
    let (d1, d2, d3) = (dense(&m.m1), dense(&m.m2), dense(&m.m3));
    // M_2 M_1 restricted to one segment: 32 x 4,096, then M_3: 16 x 4,096.
    let mut m21 = vec![vec![0_i64; L]; l2.rows];
    for (rho2, row) in m21.iter_mut().enumerate() {
        for (x, cell) in row.iter_mut().enumerate() {
            let (zeta1, beta) = (x % l1.cols, x / l1.cols);
            *cell = (0..l1.rows)
                .map(|rho1| d2[rho2][rho1 + l1.rows * beta] * d1[rho1][zeta1])
                .sum::<i64>()
                & 1;
        }
    }
    (0..l3.rows)
        .map(|rho3| {
            let row: Vec<i64> = (0..L)
                .map(|x| {
                    (0..l2.rows)
                        .map(|rho2| d3[rho3][rho2] * m21[rho2][x])
                        .sum::<i64>()
                        & 1
                })
                .collect();
            (row.iter()
                .zip(bits)
                .map(|(c, &b)| c * i64::from(b))
                .sum::<i64>()
                & 1) as u8
        })
        .collect()
}

#[test]
fn toy_instance_matches_a_dense_recomputation() {
    let shape = Shape::TOY;
    let m = derive_matrices_for(&shape, &SEED);
    assert!(shape.row_weights_ok(&m));
    let bits = random_bits(1);
    let w = parity::compute(&shape, &[&bits], &m).unwrap();

    // (a) Sums, parities and carries against dense i64 products; carries in range.
    let sums = dense_sums(&m, &bits, &w);
    let outputs: [(&[i64], &[i64]); 3] = [
        (&w.y1[0], &w.kappa1[0]),
        (&w.y2[0], &w.kappa2[0]),
        (&w.y, &w.kappa3),
    ];
    for (t, (s, (y, k))) in sums.iter().zip(outputs).enumerate() {
        let level = shape.level(t + 1);
        assert_eq!(s.len(), y.len(), "level {}", t + 1);
        for ((&s, &y), &k) in s.iter().zip(y).zip(k) {
            assert_eq!((y, k), (s & 1, s >> 1), "level {}", t + 1);
            assert!((0..=level.carry_max()).contains(&k));
        }
    }
    parity::check(&shape, &[&bits], &m, &w).unwrap();
    assert!(w.y.contains(&1) && w.y.contains(&0));

    // Carry digits: in their alphabets and recomposed exactly.
    let tables = carry::group_tables(&shape, &w).unwrap();
    assert_eq!(
        (tables.g1.len(), tables.g2.len(), tables.g3.len()),
        (2, 1 + 2, 2)
    );
    assert_eq!(
        carry::compose_columns(shape.level(1), &tables.g1),
        w.kappa1[0]
    );
    assert_eq!(
        carry::compose_columns(shape.level(2), &tables.g2[1..]),
        w.kappa2[0]
    );
    assert_eq!(carry::compose_columns(shape.level(3), &tables.g3), w.kappa3);
    for (level, group) in [
        (1, &tables.g1[..]),
        (2, &tables.g2[1..]),
        (3, &tables.g3[..]),
    ] {
        let layout = shape.level(level).carry;
        for (k, column) in group.iter().enumerate() {
            assert!(column
                .iter()
                .all(|&d| layout.alphabet(k).contains(i64::from(d))));
        }
    }
    assert_eq!(tables.y, w.y_bits());

    // (b), (c) The GF(2) side: the forward walk equals the dense composite and the Fp
    // parities reduced modulo 2 at every level.
    let y = gf2(&m, &bits);
    assert_eq!(y, dense_composite(&m, &bits));
    assert_eq!(y, w.y_bits());
    assert_eq!(gf2_levels(&m, &bits), (w.y1[0].clone(), w.y2[0].clone()));
}

#[test]
fn toy_weights_match_their_dense_definitions_and_the_identities() {
    let shape = Shape::TOY;
    let m = derive_matrices_for(&shape, &SEED);
    let bits = random_bits(2);
    let w = parity::compute(&shape, &[&bits], &m).unwrap();
    let domains = shape.domains(L);
    let (dense1, dense2, dense3) = (dense(&m.m1), dense(&m.m2), dense(&m.m3));
    let mut rng = SplitMix64::new(3);
    let bits_f: Vec<F> = bits.iter().map(|&b| F::from_i64(i64::from(b))).collect();
    for _ in 0..10 {
        // Level 1 at a D_1 point: c_1 = M_1^T eq(pi_rho, .), the table, the terminal formula,
        // and sum_x w_1(x) b(x) = (y1 + 2 kappa1)(pi).
        let p1 = point(&mut rng, domains.d1_vars());
        let eq_rho = weights::eq_table(&p1[..6]);
        let eq_beta = weights::eq_table(&p1[6..]);
        let c1: Vec<F> = (0..1 << 10)
            .map(|zeta| {
                (0..64).fold(F::zero(), |a, rho| {
                    a + eq_rho[rho].mul_i64(dense1[rho][zeta])
                })
            })
            .collect();
        assert_eq!(weights::level1_columns(&shape, &domains, &m.m1, &p1), c1);
        let table = weights::level1_table(&shape, &domains, &m.m1, &p1);
        for x in 0..L {
            assert_eq!(table[x], c1[x % 1024] * eq_beta[x / 1024]);
        }
        let r = point(&mut rng, domains.input_vars());
        assert_eq!(
            weights::level1_at(&shape, &domains, &m.m1, &p1, &r),
            mle(&table, &r)
        );
        let lhs = table
            .iter()
            .zip(&bits_f)
            .fold(F::zero(), |a, (&x, &y)| a + x * y);
        let rhs: Vec<i64> = w.y1[0]
            .iter()
            .zip(&w.kappa1[0])
            .map(|(y, k)| y + 2 * k)
            .collect();
        assert_eq!(lhs, mle(&ints(&rhs), &p1));

        // Level 2 on D_1 at a D_2 point (one segment: no block variables).
        let p2 = point(&mut rng, domains.d2_vars());
        let eq2 = weights::eq_table(&p2);
        let table = weights::level2_table(&shape, &domains, &m.m2, &p2);
        for zeta in 0..256 {
            let c = (0..32).fold(F::zero(), |a, rho| a + eq2[rho].mul_i64(dense2[rho][zeta]));
            assert_eq!(table[zeta], c);
        }
        let q = point(&mut rng, domains.d1_vars());
        assert_eq!(
            weights::level2_at(&shape, &domains, &m.m2, &p2, &q),
            mle(&table, &q)
        );
        let lhs = table
            .iter()
            .zip(&w.y1[0])
            .fold(F::zero(), |a, (&x, &y)| a + x.mul_i64(y));
        let rhs: Vec<i64> = w.y2[0]
            .iter()
            .zip(&w.kappa2[0])
            .map(|(y, k)| y + 2 * k)
            .collect();
        assert_eq!(lhs, mle(&ints(&rhs), &p2));

        // Level 3 on D_2 (one lane) at a D_3 point: the right-hand side is the public Y.
        let p3 = point(&mut rng, domains.d3_vars());
        let eq3 = weights::eq_table(&p3);
        let tables = weights::level3_tables(&shape, &domains, &m.m3, &p3);
        for rho2 in 0..32 {
            let c = (0..16).fold(F::zero(), |a, rho| a + eq3[rho].mul_i64(dense3[rho][rho2]));
            assert_eq!(tables[0][rho2], c);
        }
        let q2 = point(&mut rng, domains.d2_vars());
        assert_eq!(
            weights::level3_at(&shape, &domains, &m.m3, &p3, &q2),
            vec![mle(&tables[0], &q2)]
        );
        let lhs = tables[0]
            .iter()
            .zip(&w.y2[0])
            .fold(F::zero(), |a, (&x, &y)| a + x.mul_i64(y));
        let y_at = final_vector::evaluate(&w.y_bits(), &p3);
        assert_eq!(y_at, mle(&ints(&w.y), &p3));
        assert_eq!(
            lhs,
            final_vector::level3_claimed_sum(y_at, mle(&ints(&w.kappa3), &p3))
        );
    }
    // Boolean points: the weights are the dense entries.
    let index = 0b1011_0101_usize;
    let p1: Vec<F> = (0..domains.d1_vars())
        .map(|i| F::from_u64(((index >> i) & 1) as u64))
        .collect();
    let table = weights::level1_table(&shape, &domains, &m.m1, &p1);
    let (rho, beta) = (index & 63, index >> 6);
    for x in 0..L {
        let expected = i64::from(x / 1024 == beta) * dense1[rho][x % 1024];
        assert_eq!(table[x], F::from_i64(expected));
    }
}

fn flipped(bits: &[i8], x: usize) -> Vec<i8> {
    let mut out = bits.to_vec();
    out[x] ^= 1;
    out
}

#[test]
fn single_bit_tampering_on_either_side_changes_y() {
    let shape = Shape::TOY;
    let m = derive_matrices_for(&shape, &SEED);
    let bits = random_bits(4);
    let honest = parity::compute(&shape, &[&bits], &m).unwrap();
    let gf2_y = gf2(&m, &bits);
    assert_eq!(honest.y_bits(), gf2_y);
    let mut rng = SplitMix64::new(5);
    for _ in 0..64 {
        let x = rng.below(L as u64) as usize;
        let tampered = flipped(&bits, x);
        // Fp copy only: honest parities and carries of the tampered bits. The level-1 image
        // of a single-bit difference is the column S_x (Lemma 1, deterministic); with this
        // seed it also survives levels 2 and 3 at every sampled position.
        let fp = parity::compute(&shape, &[&tampered], &m).unwrap();
        parity::check(&shape, &[&tampered], &m, &fp).unwrap();
        assert_ne!(fp.y1, honest.y1, "position {x}");
        assert_ne!(fp.y_bits(), gf2_y, "Fp copy flipped at {x}");
        // GF(2) copy only.
        let other = gf2(&m, &tampered);
        assert_ne!(other, honest.y_bits(), "GF(2) copy flipped at {x}");
        // Both copies: the bridge compares, it does not verify the bits.
        assert_eq!(fp.y_bits(), other);
        // A flipped bit under the stale honest witness breaks the level-1 identity.
        assert!(matches!(
            parity::check(&shape, &[&tampered], &m, &honest),
            Err(IdentityError::Identity { level: 1, .. })
        ));
    }
}

#[test]
fn tampered_parities_and_carries_fail_the_identities() {
    let shape = Shape::TOY;
    let m = derive_matrices_for(&shape, &SEED);
    let bits = random_bits(6);
    let honest = parity::compute(&shape, &[&bits], &m).unwrap();
    let fails = |w: &ParityWitness| parity::check(&shape, &[&bits], &m, w).unwrap_err();
    let mut w = honest.clone();
    w.kappa1[0][9] += 1;
    assert_eq!(
        fails(&w),
        IdentityError::Identity {
            level: 1,
            lane: 0,
            cell: 9
        }
    );
    // A parity of 2 with the carry lowered by one keeps the level-1 sum but is not a bit.
    let mut w = honest.clone();
    let cell = (0..w.y1[0].len()).find(|&c| w.kappa1[0][c] > 0).unwrap();
    w.y1[0][cell] += 2;
    w.kappa1[0][cell] -= 1;
    assert_eq!(
        fails(&w),
        IdentityError::NotBoolean {
            level: 1,
            lane: 0,
            cell
        }
    );
    let mut w = honest.clone();
    w.y2[0][3] ^= 1;
    assert_eq!(
        fails(&w),
        IdentityError::Identity {
            level: 2,
            lane: 0,
            cell: 3
        }
    );
    let mut w = honest.clone();
    w.y[5] ^= 1;
    assert_eq!(
        fails(&w),
        IdentityError::Identity {
            level: 3,
            lane: 0,
            cell: 5
        }
    );
    // A carry beyond floor(W_3 / 2) with a negative parity compensating it.
    let mut w = honest.clone();
    w.kappa3[0] += 16;
    w.y[0] -= 32;
    assert!(matches!(
        fails(&w),
        IdentityError::NotBoolean { level: 3, .. } | IdentityError::CarryRange { level: 3, .. }
    ));
    let mut w = honest.clone();
    w.kappa3.push(0);
    assert_eq!(fails(&w), IdentityError::Shape);
    assert!(carry::group_tables(&shape, &{
        let mut w = honest.clone();
        w.kappa2[0][0] = 32;
        w
    })
    .is_err());
    // Non-Boolean inputs.
    let mut two = bits.clone();
    two[17] = 2;
    assert!(matches!(
        parity::check(&shape, &[&two], &m, &honest),
        Err(IdentityError::NotBoolean {
            level: 0,
            lane: 0,
            cell: 17
        })
    ));
}

/// A matrix of `level`'s shape whose row 0 has weight exactly `weight` over the used columns.
fn boundary_matrix(level: &agg_bridge::LevelShape, weight: usize) -> SparseMatrix {
    let (m, k) = (level.rows, level.k);
    let mut entries = Vec::with_capacity(level.cols * k);
    for c in 0..level.cols {
        if c < weight {
            entries.push(0);
            entries.extend((0..k - 1).map(|i| 1 + ((c * (k - 1) + i) % (m - 1)) as u32));
        } else {
            entries.extend((0..k).map(|i| 1 + ((c * k + i) % (m - 1)) as u32));
        }
    }
    SparseMatrix::new(m, level.cols, k, entries)
}

#[test]
fn carries_at_the_range_boundary_and_the_row_weight_abort() {
    for shape in [Shape::TOY, Shape::PRODUCTION] {
        let random = if shape == Shape::PRODUCTION {
            BridgeMatrices::random_for_tests(9)
        } else {
            derive_matrices_for(&shape, &SEED)
        };
        assert!(shape.row_weights_ok(&random));
        for t in 1..=3 {
            let level = shape.level(t);
            let cap = level.weight_cap as usize;
            let with = |weight: usize| {
                let mut m = random.clone();
                *[&mut m.m1, &mut m.m2, &mut m.m3][t - 1] = boundary_matrix(level, weight);
                m
            };
            assert!(shape.row_weights_ok(&with(cap)), "{} level {t}", shape.name);
            let over = with(cap + 1);
            assert!(!shape.row_weights_ok(&over));
            let len = shape.level(1).cols;
            let ones = vec![1_i8; len];
            let lanes: Vec<&[i8]> = vec![&ones; shape.lanes];
            assert_eq!(
                parity::compute(&shape, &lanes, &over),
                Err(agg_witness::bridge::BridgeError::RowWeight)
            );
            // The largest carry has every digit at the top of its alphabet.
            let top: Vec<i64> = (0..level.carry.len())
                .map(|k| level.carry.alphabet(k).max())
                .collect();
            assert_eq!(carry::digits(level, level.carry_max()).unwrap(), top);
            assert!(carry::digits(level, level.carry_max() + 1).is_err());
            assert!(carry::digits(level, -1).is_err());
            if t == 1 {
                // All-one inputs on a row of weight exactly W_1: sum W_1, parity 1, top carry.
                let w = parity::compute(&shape, &lanes, &with(cap)).unwrap();
                for lane in 0..shape.lanes {
                    assert_eq!((w.y1[lane][0], w.kappa1[lane][0]), (1, level.carry_max()));
                }
                parity::check(&shape, &lanes, &with(cap), &w).unwrap();
            }
        }
    }
    let production = Shape::PRODUCTION;
    assert_eq!(
        carry::digits(production.level(1), 1023).unwrap(),
        vec![7, 7, 1]
    );
    assert_eq!(carry::digits(production.level(3), 255).unwrap(), vec![7, 7]);
}
