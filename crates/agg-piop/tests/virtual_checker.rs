//! The virtual-table evaluators agree with the integer checker of agg-witness on the Boolean
//! domain, for honest tables and for single-cell mutations (spec Sections 4.3, 5, 7).

mod common;

use agg_piop::manifest::{Column, DigitValue, GroupRole};
use agg_piop::source_tables::{BridgeGroupTables, MainTables, SourceTables};
use agg_piop::virtual_polys::{
    compose_table, digit_tables, leaf, parity1_tables, residual_table, w1_table, MainValues,
};
use agg_piop::OpeningField;
use agg_witness::bridge;
use agg_witness::check::{check, digit_virtual, CheckOptions, CheckReport, Identity};
use agg_witness::padding::main_index;
use agg_witness::params::{GAMMA2, K, L, N, Q};
use agg_witness::tables::{BatchTables, Layout};
use common::{boolean_point, Batch};
use jolt_field::{Ring, Zero};

fn lift(value: i64) -> OpeningField {
    OpeningField::from_i64(value)
}

fn run_checker(batch: &Batch, main: &MainTables, bridge: &BridgeGroupTables) -> CheckReport {
    let tables = BatchTables {
        layout: Layout::H,
        geom: main.geometry(),
        columns: main.columns().to_vec(),
    };
    let bridge_tables = bridge.to_witness_tables();
    check(
        &batch.instance,
        &tables,
        Some((&batch.matrices, &bridge_tables)),
        CheckOptions {
            completeness: false,
        },
    )
}

/// Indices of the listed failures of one identity (and column name, if given).
fn failures(report: &CheckReport, identity: Identity, what: Option<&str>) -> Vec<usize> {
    let mut indices: Vec<usize> = report
        .failures
        .iter()
        .filter(|f| f.identity == identity && what.is_none_or(|w| f.what == w))
        .map(|f| f.index)
        .collect();
    indices.sort_unstable();
    indices
}

fn nonzero(table: &[OpeningField]) -> Vec<usize> {
    (0..table.len())
        .filter(|&x| table[x] != OpeningField::zero())
        .collect()
}

fn small_batch() -> Batch {
    // 50 live records pad to N* = 64.
    Batch::build(common::records(0x51, 40, 10), 0x51)
}

#[test]
fn honest_virtual_tables_agree_with_the_checker() {
    let batch = small_batch();
    let manifest = &batch.manifest;
    let report = run_checker(&batch, &batch.main, &batch.bridge);
    assert!(report.is_ok(), "{:?}", report.failures);
    let tables = batch.tables();

    // w1', z, u, v against the checker's recomposed record values, on every row.
    let witness = BatchTables {
        layout: Layout::H,
        geom: batch.main.geometry(),
        columns: batch.main.columns().to_vec(),
    };
    let composed = |value: DigitValue| compose_table(&value.layout(), &tables.digits(value));
    let (z, u, v) = (
        composed(DigitValue::Z),
        composed(DigitValue::U),
        composed(DigitValue::V),
    );
    let w1 = w1_table(&batch.main.bit_lanes(manifest));
    let d_main = manifest.d_main();
    for i in 0..witness.geom.n_star {
        let values = witness.record_values(i);
        for rho in 0..K {
            for j in 0..N {
                let x = main_index(j, rho, i);
                assert_eq!(w1[x], values.w1[rho][j]);
                assert_eq!(u[x], values.u[rho][j]);
                assert_eq!(v[x], values.v[rho][j]);
            }
        }
        for s in 0..L {
            for j in 0..N {
                assert_eq!(z[main_index(j, s, i)], values.z[s][j]);
            }
        }
        // The verifier's affine evaluators at a Boolean point read the same values.
        let x = main_index(17 * i % N, i % K, i);
        let point = boolean_point(x, d_main);
        let opened: Vec<OpeningField> = batch
            .main
            .columns()
            .iter()
            .map(|column| common::mle_i8(column, &point))
            .collect();
        let at = MainValues::from_evaluations(manifest, &opened);
        assert_eq!(
            (at.z, at.u, at.v, at.w1),
            (lift(z[x]), lift(u[x]), lift(v[x]), lift(w1[x]))
        );
    }

    // The ring residual vanishes on the whole Boolean domain.
    let delta = residual_table(manifest, &batch.instance, &batch.main);
    assert!(nonzero(&delta).is_empty());

    // y^(1) equals the honest integer parities of agg-witness and is Boolean.
    let parities = parity1_tables(tables, &batch.matrices.m1);
    let honest = bridge::compute(
        &batch.main.geometry(),
        &batch.main.bit_lanes(manifest),
        &batch.matrices,
    )
    .unwrap();
    assert_eq!(parities, honest.y1);
    assert!(parities.iter().flatten().all(|&y| y == 0 || y == 1));

    // Digit tables of all 55 digits equal the checker's materialization; leaves vanish.
    let digits = manifest.digit_columns();
    assert_eq!(digits.len(), 55);
    for (_, _, column) in digits {
        let base = column.base().unwrap();
        let cells = tables.column(column);
        let ours = digit_tables(cells, base);
        let reference = digit_virtual(cells, base);
        for x in 0..cells.len() {
            assert_eq!(ours.p[x], OpeningField::from_i128(reference.p[x]));
            assert_eq!(
                leaf(base, ours.p[x]),
                OpeningField::zero(),
                "{}",
                column.name()
            );
            if base == 16 {
                assert_eq!(
                    ours.a_lo[x],
                    OpeningField::from_i128(reference.a_lo[x].unwrap())
                );
                assert_eq!(
                    ours.a_hi[x],
                    OpeningField::from_i128(reference.a_hi[x].unwrap())
                );
            }
        }
    }
}

#[test]
fn residual_agrees_with_the_checker_under_mutations() {
    let batch = small_batch();
    let manifest = &batch.manifest;
    let position = |column: Column| manifest.locate(column).1;

    // A quotient digit: Delta = -q W_0 delta at exactly that position.
    let mut main = batch.main.clone();
    let v0 = position(Column::Digit {
        value: DigitValue::V,
        digit: 0,
    });
    let x = main_index(200, 3, 7);
    let step: i8 = if main.columns()[v0][x] < 7 { 1 } else { -1 };
    main.columns_mut()[v0][x] += step;
    let delta = residual_table(manifest, &batch.instance, &main);
    let report = run_checker(&batch, &main, &batch.bridge);
    assert_eq!(nonzero(&delta), vec![x]);
    assert_eq!(failures(&report, Identity::Ring, None), vec![x]);
    assert_eq!(delta[x], lift(-Q * i64::from(step)));

    // A response digit: (A z) changes along a whole negacyclic row of A_i.
    let mut main = batch.main.clone();
    let z0 = position(Column::Digit {
        value: DigitValue::Z,
        digit: 0,
    });
    let (j0, s0, i0) = (5, 2, 3);
    let cell = main_index(j0, s0, i0);
    let step: i8 = if main.columns()[z0][cell] < 7 { 1 } else { -1 };
    main.columns_mut()[z0][cell] += step;
    let delta = residual_table(manifest, &batch.instance, &main);
    let report = run_checker(&batch, &main, &batch.bridge);
    let changed = nonzero(&delta);
    assert_eq!(changed.len(), report.counts[&Identity::Ring]);
    let listed = failures(&report, Identity::Ring, None);
    assert!(!listed.is_empty() && listed.iter().all(|x| changed.contains(x)));
    let a = &batch.instance.records[i0].key.a;
    for (j, rho) in [(0, 0), (4, 1), (5, 2), (255, 3)] {
        let coefficient = if j >= j0 {
            i64::from(a[rho][s0][j - j0])
        } else {
            -i64::from(a[rho][s0][j + N - j0])
        };
        assert_eq!(
            delta[main_index(j, rho, i0)],
            lift(coefficient * i64::from(step))
        );
    }

    // A w1Encode bit: Delta = -2 gamma_2 2^t at that position.
    let mut main = batch.main.clone();
    let b1 = position(Column::W1Bit(1));
    let x = main_index(9, 1, 11);
    let old = main.columns()[b1][x];
    main.columns_mut()[b1][x] = 1 - old;
    let delta = residual_table(manifest, &batch.instance, &main);
    let report = run_checker(&batch, &main, &batch.bridge);
    assert_eq!(nonzero(&delta), vec![x]);
    assert_eq!(failures(&report, Identity::Ring, None), vec![x]);
    assert_eq!(delta[x], lift(-2 * GAMMA2 * 2 * (1 - 2 * i64::from(old))));
}

#[test]
fn parities_and_digit_leaves_agree_with_the_checker_under_mutations() {
    let batch = small_batch();
    let manifest = &batch.manifest;

    // A level-1 carry digit: y^(1)_t = sum - 2 kappa leaves {0, 1} exactly where the checker's
    // bool_y1 fails.
    let lane = 4_u8;
    let mut bridge = batch.bridge.clone();
    let (role, k0) = manifest.locate(Column::Digit {
        value: DigitValue::Kappa1(lane),
        digit: 0,
    });
    assert_eq!(role, GroupRole::Level1);
    let mutated = [3_usize, 4000, 8191];
    for &index in &mutated {
        let cells = &mut bridge.group_mut(GroupRole::Level1)[k0];
        cells[index] += if cells[index] < 7 { 1 } else { -1 };
    }
    let tables = SourceTables {
        manifest,
        main: &batch.main,
        bridge: &bridge,
    };
    let parities = parity1_tables(tables, &batch.matrices.m1);
    let report = run_checker(&batch, &batch.main, &bridge);
    let non_boolean: Vec<usize> = (0..parities[usize::from(lane)].len())
        .filter(|&x| !matches!(parities[usize::from(lane)][x], 0 | 1))
        .collect();
    assert_eq!(non_boolean, mutated);
    assert_eq!(
        failures(&report, Identity::BoolY1, Some(&format!("y1_{lane}"))),
        mutated
    );
    for (t, lane_table) in parities.iter().enumerate() {
        if t != usize::from(lane) {
            assert!(lane_table.iter().all(|&y| y == 0 || y == 1));
        }
    }

    // Digits outside their alphabets: the leaf (or the base-16 product) is nonzero exactly
    // where the checker reports b16_prod or leaf4.
    let mut main = batch.main.clone();
    let z1 = manifest
        .locate(Column::Digit {
            value: DigitValue::Z,
            digit: 1,
        })
        .1;
    let bad = [
        main_index(0, 0, 0),
        main_index(77, 3, 49),
        main_index(255, 1, 60),
    ];
    for &x in &bad {
        main.columns_mut()[z1][x] = 8;
    }
    let mut bridge = batch.bridge.clone();
    let top = Column::Digit {
        value: DigitValue::Kappa2(2),
        digit: 2,
    };
    let (_, k2) = manifest.locate(top);
    bridge.group_mut(GroupRole::Level2)[k2][100] = -3;
    let report = run_checker(&batch, &main, &batch.bridge);
    let ours = digit_tables(&main.columns()[z1], 16);
    let product_nonzero: Vec<usize> = (0..ours.p.len())
        .filter(|&x| ours.a_lo[x] * ours.a_hi[x] != OpeningField::zero())
        .collect();
    assert_eq!(product_nonzero, bad);
    assert_eq!(failures(&report, Identity::B16Prod, Some("z_1")), bad);
    let leaves = digit_tables(&bridge.group(GroupRole::Level2)[k2], 4);
    let leaf_nonzero: Vec<usize> = (0..leaves.p.len())
        .filter(|&x| leaf(4, leaves.p[x]) != OpeningField::zero())
        .collect();
    assert_eq!(leaf_nonzero, vec![100]);
    let report = run_checker(&batch, &batch.main, &bridge);
    assert_eq!(
        failures(&report, Identity::Leaf4, Some(&top.name())),
        vec![100]
    );
}
