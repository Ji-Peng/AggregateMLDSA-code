//! Kernel tests: every evaluation path against an independent reference (the round
//! polynomial from its definition, evaluated at `d + 1` points over full equality tables and
//! interpolated), bit for bit, on random tables with every `Term` kind.

use super::*;
use super::plan::PolyKind;
use crate::range::{hi_map, lo_map, square_map};
use crate::test_support::{field, point};
use aerie_falcon::protocol::algebra::{eq_evaluation, eq_table, evaluate_mle};
use agg_witness::gen::SplitMix64;

fn digits(rng: &mut SplitMix64, len: usize) -> Vec<i8> {
    (0..len).map(|_| rng.below(16) as i8 - 8).collect()
}

fn bits(rng: &mut SplitMix64, len: usize) -> Vec<i8> {
    (0..len).map(|_| rng.below(2) as i8).collect()
}

fn summand(instances: &[GroupInstance], values: &[E]) -> E {
    instances.iter().fold(E::zero(), |s, instance| {
        s + instance.gamma
            * instance
                .terms
                .iter()
                .fold(E::zero(), |t, term| t + term.evaluate(values))
    })
}

/// The direct sum of a group over the Boolean domain.
fn direct_sum(lanes: &[Lane<'_>], instances: &[GroupInstance], eq_point: Option<&[E]>) -> E {
    let len = lanes[0].len();
    let weights = eq_point.map(eq_table);
    (0..len).fold(E::zero(), |acc, x| {
        let values: Vec<E> = lanes.iter().map(|lane| lane.value(x)).collect();
        acc + summand(instances, &values) * weights.as_ref().map_or(E::one(), |w| w[x])
    })
}

/// The reference prover: full `K` tables, the round polynomial `g(X)` at `X = 0..=d` from
/// the definition (`eq` of the bound, current and remaining variables in full), interpolated
/// and compressed. Returns the messages and the terminal lane values.
fn reference(
    lanes: &[Lane<'_>],
    instances: &[GroupInstance],
    eq_point: Option<&[E]>,
    rounds: usize,
    degree: usize,
    challenges: &[E],
) -> (Vec<Vec<E>>, Vec<E>) {
    let vars = lanes[0].len().trailing_zeros() as usize;
    let inactive = rounds - vars;
    let mut tables: Vec<Vec<E>> = lanes
        .iter()
        .map(|lane| (0..lane.len()).map(|i| lane.value(i)).collect())
        .collect();
    let mut claim = direct_sum(lanes, instances, eq_point);
    let mut messages = Vec::new();
    let mut scalar = E::one();
    for round in 0..rounds {
        let message = if round < inactive {
            inactive_message(claim, degree)
        } else {
            let j = round - inactive;
            let half = tables[0].len() / 2;
            let rest = eq_point.map(|tau| eq_table(&tau[j + 1..]));
            let lift = inverse_power_of_two(inactive);
            let evals: Vec<E> = (0..=degree as u64)
                .map(|x| {
                    let xf = E::from_u64(x);
                    let mut sum = E::zero();
                    for k in 0..half {
                        let values: Vec<E> = tables
                            .iter()
                            .map(|t| t[2 * k] + xf * (t[2 * k + 1] - t[2 * k]))
                            .collect();
                        sum += summand(instances, &values) * rest.as_ref().map_or(E::one(), |w| w[k]);
                    }
                    let current = eq_point.map_or(E::one(), |tau| {
                        (E::one() - tau[j]) * (E::one() - xf) + tau[j] * xf
                    });
                    sum * current * scalar * lift
                })
                .collect();
            compress(UnivariatePoly::from_evals(&evals).coefficients(), degree)
        };
        let r = challenges[round];
        claim = evaluate_compressed(&message, claim, r);
        if round >= inactive {
            let j = round - inactive;
            if let Some(tau) = eq_point {
                scalar *= (E::one() - tau[j]) * (E::one() - r) + tau[j] * r;
            }
            for t in &mut tables {
                *t = (0..t.len() / 2).map(|k| t[2 * k] + r * (t[2 * k + 1] - t[2 * k])).collect();
            }
        }
        messages.push(message);
    }
    (messages, tables.iter().map(|t| t[0]).collect())
}

/// Run a group through its rounds with the given challenges, checking the running claim
/// and the terminal equation. Returns the messages and the terminal values.
fn run(
    mut group: Group<'_>,
    instances: &[GroupInstance],
    rounds: usize,
    degree: usize,
    eq_point: Option<&[E]>,
    mut claim: E,
    challenges: &[E],
) -> (Vec<Vec<E>>, Vec<E>) {
    if group.inactive_rounds() > 0 {
        assert_eq!(group.claim(), claim, "direct claim of a lifted group");
    }
    let mut messages = Vec::new();
    for (round, &r) in challenges.iter().enumerate().take(rounds) {
        let message = group.round_message(round);
        assert_eq!(message.len(), degree);
        assert_eq!(group.claim(), claim, "round {round}");
        claim = evaluate_compressed(&message, claim, r);
        group.bind(round, r);
        assert_eq!(group.claim(), claim);
        messages.push(message);
    }
    let values = group.terminal_values();
    let local = &challenges[group.inactive_rounds()..rounds];
    let eq_factor = eq_point.map_or(E::one(), |p| eq_evaluation(p, local));
    let expected = expected_terminal(instances, &values, eq_factor, group.inactive_rounds());
    assert_eq!(claim, expected, "terminal equation");
    (messages, values)
}

/// Every evaluation strategy of the kernel.
fn policies() -> Vec<(&'static str, Policy)> {
    vec![
        ("auto", AUTO),
        ("generic (no coding)", Policy { coding: false, ..AUTO }),
        ("coded, per item", Policy { buckets: false, ..AUTO }),
        ("bivariate forced", Policy { bivariate: Choice::Force, ..AUTO }),
        ("pair masses only", Policy { bivariate: Choice::Forbid, ..AUTO }),
        ("quartet tables forced", Policy { quartet_lut: Choice::Force, bivariate: Choice::Force, ..AUTO }),
        ("quartet tables, no Taylor", Policy { quartet_lut: Choice::Force, taylor: false, ..AUTO }),
        ("pair fold at level 2", Policy { quartet_lut: Choice::Forbid, ..AUTO }),
        ("without the claim recovery", Policy { recover: false, ..AUTO }),
        ("quartet tables without the claim recovery", Policy { recover: false, quartet_lut: Choice::Force, ..AUTO }),
    ]
}

/// Check every strategy against the reference: identical messages and terminal values,
/// terminal values equal to the multilinear extensions at the local point.
fn check_all_paths(
    lanes: &[Lane<'_>],
    instances: &[GroupInstance],
    eq_point: Option<&[E]>,
    rounds: usize,
    degree: usize,
    seed: u64,
) {
    let mut rng = SplitMix64::new(seed);
    let challenges = point(&mut rng, rounds);
    let (expected_messages, expected_values) = reference(lanes, instances, eq_point, rounds, degree, &challenges);
    let claim = direct_sum(lanes, instances, eq_point);
    let vars = lanes[0].len().trailing_zeros() as usize;
    for (name, policy) in policies() {
        let group = Group::with_policy(lanes.to_vec(), instances.to_vec(), eq_point, rounds, degree, policy).unwrap();
        let (messages, values) = run(group, instances, rounds, degree, eq_point, claim, &challenges);
        assert_eq!(messages, expected_messages, "messages, {name}");
        assert_eq!(values, expected_values, "terminal values, {name}");
    }
    let local = &challenges[rounds - vars..];
    for (lane, value) in lanes.iter().zip(&expected_values) {
        let table: Vec<E> = (0..lane.len()).map(|i| lane.value(i)).collect();
        assert_eq!(*value, evaluate_mle(&table, local));
    }
}

fn instance(rng: &mut SplitMix64, terms: Vec<Term>) -> GroupInstance {
    GroupInstance {
        gamma: field(rng),
        terms,
    }
}

/// An equality group with every bucketable term kind: Booleans and the bit-range identity
/// on four bit columns, both leaves on one digit's square, the `A^lo A^hi` product, links,
/// reductions, a base-4 leaf on a five-class column and a weightless affine form.
fn eq_fixture(rng: &mut SplitMix64, vars: usize) -> Vec<Vec<i8>> {
    let mut cells: Vec<Vec<i8>> = (0..4).map(|_| bits(rng, 1 << vars)).collect();
    cells.push(digits(rng, 1 << vars));
    cells.push(digits(rng, 1 << vars));
    // Values in [-2, 2]: five classes, pattern spaces that are not powers of two.
    cells.push((0..1 << vars).map(|_| rng.below(5) as i8 - 2).collect());
    cells
}

fn eq_lanes(cells: &[Vec<i8>]) -> Vec<Lane<'_>> {
    vec![
        Lane::column(&cells[0]),              // 0: b_2
        Lane::column(&cells[1]),              // 1: b_3
        Lane::column(&cells[2]),              // 2: b_4
        Lane::column(&cells[3]),              // 3: b_5
        Lane::mapped(&cells[4], square_map()), // 4: P of digit A
        Lane::mapped(&cells[4], lo_map()),    // 5: A^lo of digit A
        Lane::mapped(&cells[4], hi_map()),    // 6: A^hi of digit A
        Lane::mapped(&cells[5], square_map()), // 7: P of digit B
        Lane::column(&cells[5]),              // 8: digit B
        Lane::mapped(&cells[6], square_map()), // 9: P of a base-4 digit
    ]
}

fn eq_instances(rng: &mut SplitMix64) -> Vec<GroupInstance> {
    vec![
        instance(rng, vec![Term::Bool(0)]),
        instance(rng, vec![Term::Bool(1)]),
        instance(rng, vec![Term::Bool(2), Term::Bool(3)]),
        instance(rng, vec![Term::RangeW1([3, 2, 1, 0])]),
        instance(rng, vec![Term::Leaf(4, LeafPoly::Q8)]),
        instance(rng, vec![Term::Leaf(4, LeafPoly::Q16Hi)]),
        instance(rng, vec![Term::Product(vec![5, 6])]),
        instance(rng, vec![Term::Leaf(7, LeafPoly::Q8)]),
        instance(rng, vec![Term::Link(8)]),
        instance(rng, vec![Term::Product(vec![8])]),
        instance(rng, vec![Term::Leaf(9, LeafPoly::Q4)]),
        instance(
            rng,
            vec![Term::Affine {
                weight: None,
                lanes: vec![(0, -190_464), (8, 16), (1, 4096)],
                constant: 34_952,
            }],
        ),
        instance(rng, vec![Term::Product(vec![1])]),
        // H-FIPS's F_e shape: powers of one lane, a two-lane and a three-lane product, and a
        // constant, expanded by the plan into polynomial, bilinear and generic pieces.
        instance(
            rng,
            vec![Term::Monomials(vec![
                (1, vec![8, 8, 8]),
                (-1, vec![8, 8]),
                (-1, vec![8, 8, 1]),
                (1, vec![8, 1]),
                (3, vec![]),
            ])],
        ),
    ]
}

#[test]
fn every_path_matches_the_reference_on_an_equality_group_with_every_term_kind() {
    let mut rng = SplitMix64::new(11);
    for vars in [9, 10] {
        let cells = eq_fixture(&mut rng, vars);
        let lanes = eq_lanes(&cells);
        let instances = eq_instances(&mut rng);
        for (rounds, degree) in [(vars, 5), (vars + 2, 5), (vars, 6)] {
            let tau = point(&mut rng, vars);
            check_all_paths(&lanes, &instances, Some(&tau), rounds, degree, 100 + rounds as u64);
            let group = Group::new(lanes.clone(), instances.clone(), Some(&tau), rounds, degree).unwrap();
            assert!(group.is_bucketed());
            // Every lane is coded: bits (2 classes), the digit square / leaf maps (8), the
            // digit B under its square and the identity (16), the base-4 square (3).
            let classes: Vec<usize> = group.columns.iter().map(|c| c.classes).collect();
            assert_eq!(classes, vec![2, 2, 2, 2, 8, 16, 3]);
            assert!(group.direct_source.polys.is_empty() && group.direct_source.generic.is_empty());
        }
    }
}

#[test]
fn the_two_round_prefix_applies_when_the_quartets_dwarf_the_pattern_space() {
    let mut rng = SplitMix64::new(12);
    let vars = 16;
    let cells: Vec<Vec<i8>> = vec![digits(&mut rng, 1 << vars), bits(&mut rng, 1 << vars)];
    let lanes = vec![Lane::mapped(&cells[0], square_map()), Lane::column(&cells[1])];
    let instances = vec![
        instance(&mut rng, vec![Term::Leaf(0, LeafPoly::Q8), Term::Leaf(0, LeafPoly::Q16Hi)]),
        instance(&mut rng, vec![Term::Bool(1)]),
    ];
    let tau = point(&mut rng, vars);
    let group = Group::new(lanes.clone(), instances.clone(), Some(&tau), vars, 5).unwrap();
    // 2^14 quartets >= 4 * 8^4.
    assert!(group.bivariate());
    check_all_paths(&lanes, &instances, Some(&tau), vars, 5, 7);
    // An identity-mapped digit has 16^4 quartet patterns: pair masses only at this size.
    let lanes = vec![Lane::column(&cells[0])];
    let instances = vec![instance(&mut rng, vec![Term::Link(0)])];
    let group = Group::new(lanes.clone(), instances.clone(), Some(&tau), vars, 3).unwrap();
    assert!(!group.bivariate());
    check_all_paths(&lanes, &instances, Some(&tau), vars, 3, 8);
}

#[test]
fn weight_groups_match_the_reference() {
    let mut rng = SplitMix64::new(13);
    let vars = 9;
    let weight_a = point(&mut rng, 1 << vars);
    let weight_b = point(&mut rng, 1 << vars);
    let bit_cells: Vec<Vec<i8>> = (0..2).map(|_| bits(&mut rng, 1 << vars)).collect();
    let digit_cells: Vec<Vec<i8>> = (0..3).map(|_| digits(&mut rng, 1 << vars)).collect();
    let ints: Vec<i32> = (0..1 << vars).map(|_| rng.below(2000) as i32 - 1000).collect();
    let folded: Vec<E> = point(&mut rng, 1 << vars);
    let lanes = vec![
        Lane::Source(Source::Field(&weight_a)),
        Lane::Source(Source::Field(&weight_b)),
        Lane::column(&bit_cells[0]),
        Lane::column(&bit_cells[1]),
        Lane::column(&digit_cells[0]),
        Lane::column(&digit_cells[1]),
        Lane::mapped(&digit_cells[2], square_map()),
        Lane::Source(Source::Ints(&ints)),
        Lane::Folded(folded),
    ];
    let instances = vec![
        instance(&mut rng, vec![Term::Product(vec![0, 2])]),
        instance(&mut rng, vec![Term::Product(vec![0, 3])]),
        instance(&mut rng, vec![Term::Product(vec![3, 1])]),
        instance(
            &mut rng,
            vec![Term::Affine {
                weight: Some(0),
                lanes: vec![(4, 1), (5, 16), (6, 256), (7, -3)],
                constant: -2056,
            }],
        ),
        instance(&mut rng, vec![Term::Product(vec![1, 8])]),
        instance(&mut rng, vec![Term::Bool(2)]),
        instance(&mut rng, vec![Term::Product(vec![8])]),
    ];
    for rounds in [vars, vars + 1] {
        check_all_paths(&lanes, &instances, None, rounds, 2, 30 + rounds as u64);
    }
    let group = Group::new(lanes, instances, None, vars, 2).unwrap();
    // The weighted sums absorb the coefficients of the lanes they alone read.
    assert_eq!(group.plan.folded.weighted.len(), 2);
    for &(l, c) in &group.plan.folded.weighted[0].lanes {
        assert_eq!(c == E::one(), (4..=7).contains(&l), "lane {l}");
    }
}

#[test]
fn wide_cells_ints_and_cross_column_products_take_the_per_item_path() {
    let mut rng = SplitMix64::new(14);
    let vars = 8;
    let mut wide = digits(&mut rng, 1 << vars);
    wide[5] = 9; // a tampered digit outside [-8, 7]: 17 classes under the identity
    let narrow = digits(&mut rng, 1 << vars);
    let other = digits(&mut rng, 1 << vars);
    let ints: Vec<i32> = (0..1 << vars).map(|_| rng.below(2000) as i32).collect();
    let tau = point(&mut rng, vars);
    let lanes = vec![
        Lane::column(&wide),
        Lane::mapped(&wide, square_map()),
        Lane::column(&narrow),
        Lane::Source(Source::Ints(&ints)),
        Lane::column(&other),
    ];
    let instances = vec![
        instance(&mut rng, vec![Term::Link(0)]),
        instance(&mut rng, vec![Term::Leaf(1, LeafPoly::Q8)]),
        instance(&mut rng, vec![Term::Link(2)]),
        instance(&mut rng, vec![Term::Bool(3)]),
        instance(&mut rng, vec![Term::Product(vec![2, 4])]),
        instance(&mut rng, vec![Term::Product(vec![0, 2, 4])]),
    ];
    let group = Group::new(lanes.clone(), instances.clone(), Some(&tau), vars, 5).unwrap();
    assert_eq!(group.lane_column, vec![None, None, Some(0), None, Some(1)]);
    // Only the link of the narrow column is bucketed.
    assert_eq!(group.segments.len(), 1);
    check_all_paths(&lanes, &instances, Some(&tau), vars, 5, 40);
    check_all_paths(&lanes, &instances, Some(&tau), vars + 3, 5, 41);
}

#[test]
fn domains_of_one_to_four_variables_and_lifted_groups() {
    let mut rng = SplitMix64::new(15);
    for vars in 0..=4 {
        let cells = eq_fixture(&mut rng, vars);
        let lanes = eq_lanes(&cells);
        let instances = eq_instances(&mut rng);
        let tau = point(&mut rng, vars);
        for extra in [0, 2] {
            if vars + extra == 0 {
                continue;
            }
            if vars > 0 {
                check_all_paths(&lanes, &instances, Some(&tau), vars + extra, 5, 50 + vars as u64);
            }
            check_all_paths(&lanes, &instances, None, vars + extra, 4, 60 + vars as u64);
        }
    }
}

#[test]
fn quartet_and_taylor_tables_reproduce_the_folded_lanes() {
    let mut rng = SplitMix64::new(16);
    let vars = 12;
    let cells = eq_fixture(&mut rng, vars);
    let lanes = eq_lanes(&cells);
    let instances = eq_instances(&mut rng);
    let tau = point(&mut rng, vars);
    let policy = Policy { quartet_lut: Choice::Force, ..AUTO };
    let mut group = Group::with_policy(lanes.clone(), instances, Some(&tau), vars, 5, policy).unwrap();
    let (r0, r1) = (field(&mut rng), field(&mut rng));
    group.round_message(0);
    group.bind(0, r0);
    group.round_message(1);
    group.bind(1, r1);
    let inverse = group.plan.inverse_scales();
    for (l, lane) in lanes.iter().enumerate() {
        let once: Vec<E> = (0..lane.len() / 2).map(|k| lane.value(2 * k) + r0 * (lane.value(2 * k + 1) - lane.value(2 * k))).collect();
        let twice: Vec<E> = (0..once.len() / 2).map(|k| once[2 * k] + r1 * (once[2 * k + 1] - once[2 * k])).collect();
        assert!(matches!(group.states[l], State::Quartet(_)), "lane {l} is a quartet table");
        for (k, &v) in twice.iter().enumerate() {
            assert_eq!(group.value_at(l, k) * inverse[l], v, "lane {l} cell {k}");
        }
    }
    // Taylor tables: every degree >= 2 polynomial of a quartet lane, at every entry.
    assert!(!group.taylor.is_empty());
    for taylor in &group.taylor {
        let State::Quartet(lut) = &group.states[taylor.lane] else { unreachable!() };
        let kind = &group.plan.folded.polys.iter().find(|(l, _)| *l == taylor.lane).unwrap().1;
        let c = kind.coefficients();
        for (e, &v) in lut.iter().enumerate().step_by(97) {
            let (d, x) = (field(&mut rng), field(&mut rng));
            let t = &taylor.lines[e].0[..taylor.degree];
            let direct = c.iter().rev().fold(E::zero(), |acc, &k| acc * (v + x * d) + k);
            let shifted = t.iter().rev().fold(E::zero(), |acc, &k| acc * (x * d) + k) * (x * d) + taylor.kind.value(v);
            assert_eq!(direct, shifted);
        }
    }
}

#[test]
fn compressed_messages_round_trip_and_rejections() {
    let mut rng = SplitMix64::new(3);
    let full = point(&mut rng, 5);
    let claim = full.iter().fold(E::zero(), |s, &c| s + c) + full[0];
    let message = compress(&full, 4);
    assert_eq!(message.len(), 4);
    let x = field(&mut rng);
    let direct = full.iter().rev().fold(E::zero(), |acc, &c| acc * x + c);
    assert_eq!(evaluate_compressed(&message, claim, x), direct);
    assert_eq!(
        inactive_message(claim, 3),
        vec![claim * inverse_power_of_two(1), E::zero(), E::zero()]
    );
    assert_eq!(inverse_power_of_two(3) * E::from_u64(8), E::one());
    assert!(Group::new(vec![], vec![], None, 3, 2).is_err());
    let cells = vec![0_i8; 8];
    let lanes = vec![Lane::column(&cells)];
    let too_high = vec![GroupInstance {
        gamma: E::one(),
        terms: vec![Term::Bool(0)],
    }];
    assert!(Group::new(lanes.clone(), too_high.clone(), None, 3, 1).is_err());
    assert!(Group::new(lanes.clone(), too_high, None, 2, 3).is_err());
    let missing = vec![GroupInstance {
        gamma: E::one(),
        terms: vec![Term::Product(vec![0, 1])],
    }];
    assert!(Group::new(lanes, missing, None, 3, 3).is_err());
}

#[test]
fn leaf_polynomials_vanish_exactly_on_the_alphabets() {
    for z in -128_i64..=127 {
        let p = E::from_i64(z * (z + 1));
        let in4 = (-2..=1).contains(&z);
        let in6 = (-3..=2).contains(&z);
        let in8 = (-4..=3).contains(&z);
        let in16 = (-8..=7).contains(&z);
        assert_eq!(LeafPoly::Q4.evaluate(p) == E::zero(), in4);
        assert_eq!(LeafPoly::Q6.evaluate(p) == E::zero(), in6);
        assert_eq!(LeafPoly::Q8.evaluate(p) == E::zero(), in8);
        let q16 = LeafPoly::Q8.evaluate(p) * LeafPoly::Q16Hi.evaluate(p);
        assert_eq!(q16 == E::zero(), in16);
    }
    assert_eq!(LeafPoly::of_base(16), [LeafPoly::Q8, LeafPoly::Q16Hi]);
    assert_eq!(Term::Leaf(0, LeafPoly::Q16Hi).degree(), 4);
    assert_eq!(IDENTITY_MAP[0xFF], -1);
    assert_eq!(IDENTITY_MAP[7], 7);
}

#[test]
fn leaf_coefficients_expand_the_root_products() {
    assert_eq!(LeafPoly::Q4.coefficients(), vec![0, -2, 1]);
    assert_eq!(LeafPoly::Q6.coefficients(), vec![0, 12, -8, 1]);
    assert_eq!(LeafPoly::Q8.coefficients(), vec![0, -144, 108, -20, 1]);
    assert_eq!(LeafPoly::of_base(6), [LeafPoly::Q6]);
    for poly in [LeafPoly::Q4, LeafPoly::Q6, LeafPoly::Q8, LeafPoly::Q16Hi] {
        let coefficients = poly.coefficients();
        assert_eq!(coefficients.len(), poly.degree() + 1);
        for p in [-3_i64, 0, 2, 7, 56, 1000] {
            let pf = E::from_i64(p);
            let horner = coefficients
                .iter()
                .rev()
                .fold(E::zero(), |acc, &c| acc * pf + E::from_i64(c));
            assert_eq!(horner, poly.evaluate(pf));
        }
    }
}

#[test]
fn compiled_pieces_evaluate_every_term_in_coefficient_form() {
    let mut rng = SplitMix64::new(17);
    let lanes = 8;
    let instances = vec![
        instance(&mut rng, vec![Term::Bool(0), Term::Link(1)]),
        instance(&mut rng, vec![Term::Leaf(2, LeafPoly::Q8), Term::Leaf(2, LeafPoly::Q16Hi)]),
        instance(&mut rng, vec![Term::Leaf(3, LeafPoly::Q4)]),
        instance(&mut rng, vec![Term::Leaf(4, LeafPoly::Q6)]),
        instance(&mut rng, vec![Term::Product(vec![0, 4]), Term::Product(vec![1, 2, 3])]),
        instance(&mut rng, vec![Term::RangeW1([0, 1, 4, 5])]),
        instance(&mut rng, vec![Term::Affine { weight: Some(5), lanes: vec![(0, 3), (4, -7)], constant: 9 }]),
        instance(&mut rng, vec![Term::Affine { weight: None, lanes: vec![(1, 5)], constant: -4 }]),
        instance(&mut rng, vec![Term::Product(vec![6]), Term::Product(vec![7, 0])]),
    ];
    let is_field = vec![false, false, false, false, false, true, false, false];
    let plan = Plan::compile(lanes, &instances, &is_field);
    let a = point(&mut rng, lanes);
    let d = point(&mut rng, lanes);
    let out = plan.source.evaluate(&a, &d, 6, false);
    for x in [0_u64, 1, 2, 5, 77] {
        let xf = E::from_u64(x);
        let values: Vec<E> = a.iter().zip(&d).map(|(&a, &d)| a + xf * d).collect();
        let at_x = out.iter().rev().fold(E::zero(), |acc, &c| acc * xf + c) + plan.constant;
        assert_eq!(at_x, summand(&instances, &values), "X = {x}");
    }
    // The folded phase reads scaled lanes.
    let scaled_a: Vec<E> = a.iter().zip(&plan.scales).map(|(&v, &s)| v * s).collect();
    let scaled_d: Vec<E> = d.iter().zip(&plan.scales).map(|(&v, &s)| v * s).collect();
    let folded = plan.folded.evaluate(&scaled_a, &scaled_d, 6, false);
    assert_eq!(folded, out);
    assert_ne!(plan.scales[6], E::one());
    assert_ne!(plan.scales[7], E::one());
    // The dense pieces read translated lanes; the upper form drops the constant only.
    let quartic = vec![
        instance(&mut rng, vec![Term::Leaf(0, LeafPoly::Q8), Term::Leaf(0, LeafPoly::Q16Hi)]),
        instance(&mut rng, vec![Term::Leaf(1, LeafPoly::Q8)]),
        instance(&mut rng, vec![Term::Link(2), Term::Bool(3)]),
    ];
    let plan = Plan::compile(4, &quartic, &[false; 4]);
    assert!(plan.dense.polys.iter().filter(|(_, k)| matches!(k, PolyKind::Depressed { .. })).count() == 2);
    assert!(!plan.shifts[0].is_zero() && plan.shifts[2].is_zero());
    let shifted: Vec<E> = a.iter().zip(&plan.shifts).map(|(&v, &s)| v - s).collect();
    let expected = plan.folded.evaluate(&a[..4], &d[..4], 5, false);
    let dense = plan.dense.evaluate(&shifted, &d[..4], 5, false);
    assert_eq!(dense, expected);
    let upper = plan.dense.evaluate(&shifted, &d[..4], 5, true);
    assert!(upper[0].is_zero());
    assert_eq!(upper[1..], expected[1..]);
    let all = Plan::compile(lanes, &instances, &is_field);
    let upper = all.source.evaluate(&a, &d, 6, true);
    assert!(upper[0].is_zero());
    assert_eq!(upper[1..], out[1..]);
    // Bilinear products with and without a coefficient, both forms.
    let mut bilinear = Pieces::default();
    bilinear.bilinear.push(plan::Bilinear { c: E::one(), a: 0, b: 1 });
    bilinear.bilinear.push(plan::Bilinear { c: field(&mut rng), a: 2, b: 3 });
    let full = bilinear.evaluate(&a, &d, 3, false);
    let upper = bilinear.evaluate(&a, &d, 3, true);
    assert_eq!(upper[1..], full[1..]);
    let x = field(&mut rng);
    let at = |l: usize| a[l] + x * d[l];
    assert_eq!(
        full[0] + x * full[1] + x * x * full[2],
        at(0) * at(1) + bilinear.bilinear[1].c * at(2) * at(3)
    );
}

/// The integer prefix of `IntPairGroup` (rounds 0 and 1 from the leaves, the first table
/// at level 2) and the table path (leaves beyond the prefix bounds) both match the generic
/// group byte for byte, with and without `M` lanes, at `2^8..2^14` rows.
#[test]
fn integer_prefix_matches_the_generic_group() {
    let mut rng = SplitMix64::new(2718);
    for vars in [8_usize, 9, 11, 14] {
        let rows = 1 << vars;
        let eq_point = point(&mut rng, vars);
        let challenges: Vec<E> = (0..vars).map(|_| field(&mut rng)).collect();
        let ints = |rng: &mut SplitMix64, bound: u64| -> Vec<i32> {
            (0..rows).map(|_| rng.below(2 * bound + 1) as i32 - bound as i32).collect()
        };
        // (S bound, M bound, prefix expected).
        for (with_m, s_bound, m_bound, prefix) in [
            (false, 1 << 15, 0, true),
            (false, 1 << 20, 0, true),
            (false, 1 << 21, 0, false),
            (true, 1 << 16, 1 << 30, true),
            (true, 1 << 17, 1 << 30, false),
        ] {
            let s = [ints(&mut rng, s_bound), ints(&mut rng, s_bound)];
            let m = [ints(&mut rng, m_bound), ints(&mut rng, m_bound)];
            let c: [E; 5] = std::array::from_fn(|_| field(&mut rng));
            let mut lanes = vec![Lane::Source(Source::Ints(&s[0])), Lane::Source(Source::Ints(&s[1]))];
            let mut instances = vec![
                GroupInstance { gamma: c[0], terms: vec![Term::Product(vec![])] },
                GroupInstance { gamma: c[1], terms: vec![Term::Product(vec![0]), Term::Product(vec![1])] },
                GroupInstance { gamma: c[2], terms: vec![Term::Product(vec![0, 1])] },
            ];
            if with_m {
                lanes.push(Lane::Source(Source::Ints(&m[0])));
                lanes.push(Lane::Source(Source::Ints(&m[1])));
                instances[2].terms.extend([Term::Product(vec![2]), Term::Product(vec![3])]);
                instances.push(GroupInstance { gamma: c[3], terms: vec![Term::Product(vec![0, 3]), Term::Product(vec![1, 2])] });
                instances.push(GroupInstance { gamma: c[4], terms: vec![Term::Product(vec![2, 3])] });
            }
            let (expected, _) = reference(&lanes, &instances, Some(&eq_point), vars, 3, &challenges);
            let mut generic = Group::new(lanes, instances, Some(&eq_point), vars, 3).expect("group");
            let m_lanes = with_m.then(|| [m[0].as_slice(), m[1].as_slice()]);
            let mut pair = IntPairGroup::new([&s[0], &s[1]], m_lanes, c, &eq_point).expect("pair");
            assert_eq!(pair.uses_prefix(), prefix, "M {with_m}, S {s_bound}, M {m_bound}");
            for (round, &r) in challenges.iter().enumerate() {
                let got = pair.round_message(round);
                assert_eq!(expected[round], got, "reference, round {round}, M {with_m}, S {s_bound}, {vars} vars");
                assert_eq!(generic.round_message(round), got, "generic, round {round}, M {with_m}, S {s_bound}, {vars} vars");
                generic.bind(round, r);
                pair.bind(round, r);
                assert_eq!(generic.claim(), pair.claim());
            }
            assert_eq!(generic.terminal_values(), pair.terminal_values());
            assert!(pair.terminal_consistent());
        }
    }
}

/// The dedicated groups of `kernel::layer` against the generic group (the same round
/// polynomials, bit for bit) on random tables, with challenges drawn per round.
#[test]
fn dedicated_layer_groups_match_the_generic_group() {
    let mut rng = SplitMix64::new(4711);
    for vars in [1_usize, 2, 5, 13] {
        let rows = 1 << vars;
        let eq_point = point(&mut rng, vars);
        let ints = |rng: &mut SplitMix64, bound: u64| -> Vec<i32> {
            (0..rows).map(|_| rng.below(2 * bound + 1) as i32 - bound as i32).collect()
        };
        let challenges: Vec<E> = (0..vars).map(|_| field(&mut rng)).collect();
        // Runs a group through every round and returns the messages.
        macro_rules! drive {
            ($group:expr) => {{
                let mut messages = Vec::new();
                for (round, &r) in challenges.iter().enumerate() {
                    messages.push($group.round_message(round));
                    $group.bind(round, r);
                }
                messages
            }};
        }

        // A linear combination of integer lanes.
        let lanes: Vec<Vec<i32>> = (0..4).map(|_| ints(&mut rng, 1 << 15)).collect();
        let coefficients: Vec<E> = (0..4).map(|_| field(&mut rng)).collect();
        let generic_lanes = || lanes.iter().map(|l| Lane::Source(Source::Ints(l))).collect::<Vec<_>>();
        let instances: Vec<GroupInstance> = coefficients
            .iter()
            .enumerate()
            .map(|(l, &gamma)| GroupInstance {
                gamma,
                terms: vec![Term::Product(vec![l])],
            })
            .collect();
        let mut generic = Group::new(generic_lanes(), instances, Some(&eq_point), vars, 3).expect("group");
        let mut linear =
            LinearIntGroup::new(lanes.iter().map(|l| Source::Ints(l)).collect(), coefficients.clone(), &eq_point)
                .expect("linear");
        let expected = drive!(generic);
        let got = drive!(linear);
        assert_eq!(expected, got, "linear, {vars} vars");
        assert_eq!(generic.claim(), linear.claim());
        assert!(linear.terminal_consistent());

        // A field weight times an integer combination, without an equality factor.
        {
            let weight: Vec<E> = (0..rows).map(|_| field(&mut rng)).collect();
            let lo = ints(&mut rng, 1 << 15);
            let hi: Vec<i8> = (0..rows).map(|_| rng.below(8) as i8 - 4).collect();
            let gamma = field(&mut rng);
            let lanes = vec![
                Lane::Source(Source::Field(&weight)),
                Lane::Source(Source::Ints(&lo)),
                Lane::column(&hi),
            ];
            let instances = vec![GroupInstance {
                gamma,
                terms: vec![Term::Affine {
                    weight: Some(0),
                    lanes: vec![(1, 1), (2, 1 << 16)],
                    constant: 1 << 15,
                }],
            }];
            let sources = [Source::Field(&weight), Source::Ints(&lo), Source::Bytes { cells: &hi, map: &IDENTITY_MAP }];
            let mut generic = Group::new(lanes, instances, None, vars, 3).expect("group");
            let mut weighted =
                WeightIntGroup::new(&sources, 0, &[0, 1, 1 << 16], 1 << 15, gamma, vars).expect("weighted");
            let expected = drive!(generic);
            let got = drive!(weighted);
            assert_eq!(expected, got, "weighted, {vars} vars");
            assert_eq!(generic.terminal_values(), weighted.terminal_values());
            assert!(weighted.terminal_consistent());
            // A byte map whose entries break the exactness of `g` is rejected.
            let mut wide = IDENTITY_MAP;
            wide[0] = 1 << 43;
            wide[1] = -(1 << 43);
            let sources = [Source::Field(&weight), Source::Ints(&lo), Source::Bytes { cells: &hi, map: &wide }];
            assert!(WeightIntGroup::new(&sources, 0, &[0, 1, 1 << 16], 1 << 15, gamma, vars).is_none());
        }

        // Integer pairs without and with M lanes.
        for with_m in [false, true] {
            let s = [ints(&mut rng, 1 << 16), ints(&mut rng, 1 << 16)];
            let m = [ints(&mut rng, 1 << 30), ints(&mut rng, 1 << 30)];
            let c: [E; 5] = std::array::from_fn(|_| field(&mut rng));
            let mut lanes = vec![Lane::Source(Source::Ints(&s[0])), Lane::Source(Source::Ints(&s[1]))];
            let mut instances = vec![
                GroupInstance { gamma: c[0], terms: vec![Term::Product(vec![])] },
                GroupInstance { gamma: c[1], terms: vec![Term::Product(vec![0]), Term::Product(vec![1])] },
                GroupInstance { gamma: c[2], terms: vec![Term::Product(vec![0, 1])] },
            ];
            if with_m {
                lanes.push(Lane::Source(Source::Ints(&m[0])));
                lanes.push(Lane::Source(Source::Ints(&m[1])));
                instances[2].terms.extend([Term::Product(vec![2]), Term::Product(vec![3])]);
                instances.push(GroupInstance { gamma: c[3], terms: vec![Term::Product(vec![0, 3]), Term::Product(vec![1, 2])] });
                instances.push(GroupInstance { gamma: c[4], terms: vec![Term::Product(vec![2, 3])] });
            }
            let mut generic = Group::new(lanes, instances, Some(&eq_point), vars, 3).expect("group");
            let m_lanes = with_m.then(|| [m[0].as_slice(), m[1].as_slice()]);
            let mut pair = IntPairGroup::new([&s[0], &s[1]], m_lanes, c, &eq_point).expect("pair");
            let expected = drive!(generic);
            let got = drive!(pair);
            assert_eq!(expected, got, "pair, M {with_m}, {vars} vars");
            assert_eq!(generic.terminal_values(), pair.terminal_values());
            assert!(pair.terminal_consistent());
        }

        // The layer group of three trees.
        let layers: Vec<(Vec<E>, Vec<E>)> = (0..3)
            .map(|_| ((0..2 * rows).map(|_| field(&mut rng)).collect(), (0..2 * rows).map(|_| field(&mut rng)).collect()))
            .collect();
        let gammas: Vec<(E, E)> = (0..3).map(|_| (field(&mut rng), field(&mut rng))).collect();
        let mut lanes = Vec::new();
        let mut instances = Vec::new();
        for (t, ((p, q), &(gp, gq))) in layers.iter().zip(&gammas).enumerate() {
            let (p0, p1) = p.split_at(rows);
            let (q0, q1) = q.split_at(rows);
            lanes.extend([p0, p1, q0, q1].map(|c| Lane::Source(Source::Field(c))));
            let (a, b, c, d) = (4 * t, 4 * t + 1, 4 * t + 2, 4 * t + 3);
            instances.push(GroupInstance { gamma: gp, terms: vec![Term::Product(vec![a, d]), Term::Product(vec![b, c])] });
            instances.push(GroupInstance { gamma: gq, terms: vec![Term::Product(vec![c, d])] });
        }
        let mut generic = Group::new(lanes, instances, Some(&eq_point), vars, 3).expect("group");
        let trees = layers
            .iter()
            .zip(&gammas)
            .map(|((p, q), &(gamma_p, gamma_q))| LayerTree { p, q, gamma_p, gamma_q })
            .collect();
        let mut layer = LayerGroup::new(trees, &eq_point).expect("layer");
        let expected = drive!(generic);
        let got = drive!(layer);
        assert_eq!(expected, got, "layer, {vars} vars");
        assert_eq!(generic.terminal_values(), layer.terminal_values());
        assert!(layer.terminal_consistent());
    }
}

/// The lookup sessions' groups compute two rounds per pass; with equality points that
/// have zero and one coordinates (a round whose linear term cannot be recovered from the
/// claim needs `q(1)`, and its pass falls back to one round) and at every domain size from
/// one to ten variables they still match the generic group byte for byte.
#[test]
fn two_round_passes_match_the_generic_group() {
    let mut rng = SplitMix64::new(1618);
    for vars in 1_usize..=10 {
        let rows = 1 << vars;
        for degenerate in [false, true] {
            let mut eq_point = point(&mut rng, vars);
            if degenerate {
                for (i, slot) in eq_point.iter_mut().enumerate() {
                    if i % 3 == 1 {
                        *slot = E::zero();
                    } else if i % 3 == 2 {
                        *slot = E::one();
                    }
                }
            }
            let challenges: Vec<E> = (0..vars).map(|_| field(&mut rng)).collect();
            let ints = |rng: &mut SplitMix64, bound: u64| -> Vec<i32> {
                (0..rows).map(|_| rng.below(2 * bound + 1) as i32 - bound as i32).collect()
            };
            macro_rules! check {
                ($generic:expr, $group:expr, $label:expr) => {{
                    for (round, &r) in challenges.iter().enumerate() {
                        assert_eq!($generic.round_message(round), $group.round_message(round), "{}, round {round}, {vars} vars, degenerate {degenerate}", $label);
                        $generic.bind(round, r);
                        $group.bind(round, r);
                        assert_eq!($generic.claim(), $group.claim());
                    }
                    assert_eq!($generic.terminal_values(), $group.terminal_values());
                    assert!($group.terminal_consistent());
                }};
            }
            // Integer pairs: prefix and table paths, without and with M lanes.
            for (with_m, s_bound) in [(false, 1_u64 << 15), (false, 1 << 21), (true, 1 << 16), (true, 1 << 17)] {
                let s = [ints(&mut rng, s_bound), ints(&mut rng, s_bound)];
                let m = [ints(&mut rng, 1 << 30), ints(&mut rng, 1 << 30)];
                let c: [E; 5] = std::array::from_fn(|_| field(&mut rng));
                let mut lanes = vec![Lane::Source(Source::Ints(&s[0])), Lane::Source(Source::Ints(&s[1]))];
                let mut instances = vec![
                    GroupInstance { gamma: c[0], terms: vec![Term::Product(vec![])] },
                    GroupInstance { gamma: c[1], terms: vec![Term::Product(vec![0]), Term::Product(vec![1])] },
                    GroupInstance { gamma: c[2], terms: vec![Term::Product(vec![0, 1])] },
                ];
                if with_m {
                    lanes.push(Lane::Source(Source::Ints(&m[0])));
                    lanes.push(Lane::Source(Source::Ints(&m[1])));
                    instances[2].terms.extend([Term::Product(vec![2]), Term::Product(vec![3])]);
                    instances.push(GroupInstance { gamma: c[3], terms: vec![Term::Product(vec![0, 3]), Term::Product(vec![1, 2])] });
                    instances.push(GroupInstance { gamma: c[4], terms: vec![Term::Product(vec![2, 3])] });
                }
                let mut generic = Group::new(lanes, instances, Some(&eq_point), vars, 3).expect("group");
                let m_lanes = with_m.then(|| [m[0].as_slice(), m[1].as_slice()]);
                let mut pair = IntPairGroup::new([&s[0], &s[1]], m_lanes, c, &eq_point).expect("pair");
                check!(generic, pair, format!("pair, M {with_m}, S {s_bound}"));
            }
            // The layer group of one and of two trees.
            for count in [1, 2] {
                let layers: Vec<(Vec<E>, Vec<E>)> = (0..count)
                    .map(|_| ((0..2 * rows).map(|_| field(&mut rng)).collect(), (0..2 * rows).map(|_| field(&mut rng)).collect()))
                    .collect();
                let gammas: Vec<(E, E)> = (0..count).map(|_| (field(&mut rng), field(&mut rng))).collect();
                let mut lanes = Vec::new();
                let mut instances = Vec::new();
                for (t, ((p, q), &(gp, gq))) in layers.iter().zip(&gammas).enumerate() {
                    let (p0, p1) = p.split_at(rows);
                    let (q0, q1) = q.split_at(rows);
                    lanes.extend([p0, p1, q0, q1].map(|c| Lane::Source(Source::Field(c))));
                    let (a, b, c, d) = (4 * t, 4 * t + 1, 4 * t + 2, 4 * t + 3);
                    instances.push(GroupInstance { gamma: gp, terms: vec![Term::Product(vec![a, d]), Term::Product(vec![b, c])] });
                    instances.push(GroupInstance { gamma: gq, terms: vec![Term::Product(vec![c, d])] });
                }
                let mut generic = Group::new(lanes, instances, Some(&eq_point), vars, 3).expect("group");
                let trees = layers
                    .iter()
                    .zip(&gammas)
                    .map(|((p, q), &(gamma_p, gamma_q))| LayerTree { p, q, gamma_p, gamma_q })
                    .collect();
                let mut layer = LayerGroup::new(trees, &eq_point).expect("layer");
                check!(generic, layer, format!("layer, {count} trees"));
            }
        }
    }
}

/// Integer pair groups reject lanes whose pair differences would overflow the exact `i64`
/// products of the table path (instead of wrapping in release builds).
#[test]
fn integer_pair_group_rejects_out_of_range_lanes() {
    let mut rng = SplitMix64::new(99);
    let eq_point = point(&mut rng, 3);
    let c: [E; 5] = std::array::from_fn(|_| field(&mut rng));
    let small = vec![0_i32; 8];
    let mut big_m = vec![0_i32; 8];
    big_m[0] = i32::MAX;
    big_m[1] = i32::MIN;
    let mut big_s = vec![0_i32; 8];
    big_s[2] = (1 << 17) + 1;
    assert!(IntPairGroup::new([&small, &small], Some([&big_m, &big_m]), c, &eq_point).is_none());
    assert!(IntPairGroup::new([&big_s, &small], Some([&small, &small]), c, &eq_point).is_none());
    assert!(IntPairGroup::new([&big_m, &small], None, c, &eq_point).is_none());
    assert!(IntPairGroup::new([&small, &small], Some([&small, &small]), c, &eq_point).is_some());
}
