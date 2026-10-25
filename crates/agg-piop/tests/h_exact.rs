//! Relation H-exact (`docs/h-exact.md` step B) at the PIOP level (the lookup and the three
//! stages, no commitments): an honest round trip, the pointwise identities on the Boolean
//! domain, the tamperings of Section 6.4 rejected at the stated place, and synthetic
//! statements at the value boundaries `z = +-130,994` and `u = +-190,466` whose public `T`
//! is moved so that the ring identity still holds over the integers: only the range
//! mechanism (the lookup, the leaf of `D_u`, the Booleanity zero-checks) can reject them.
#![allow(clippy::needless_range_loop)]

mod common;

use agg_piop::manifest::{Column, CommitmentManifest, DigitValue, GroupRole, Relation, LOOKUP_SHIFT, U16_DIGITS};
use agg_piop::source_tables::{BridgeGroupTables, MainTables, SourceTables};
use agg_piop::stage::{suffix, StageError};
use agg_piop::verifier::VerifyError;
use agg_piop::virtual_polys::{residual_table, w1_table};
use agg_piop::OpeningField as E;
use agg_witness::crt::az_integer;
use agg_witness::digits::{leaf_vanishes, p_of};
use agg_witness::padding::main_index;
use agg_witness::params::{B_U, B_Z, N};
use common::{lenient_multiplicities, prove_tables, verify_tables, Batch};
use jolt_field::{Ring, Zero};

fn on_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("test thread");
}

/// 50 live records (N* = 64) under `relation`.
fn small_batch(relation: Relation) -> Batch {
    Batch::build_as(relation, common::records(0x51, 40, 10), 0x51)
}

/// Where the verifier rejected: `0` for the lookup, `s` for stage `s`.
fn rejection(error: &VerifyError) -> Option<usize> {
    match error {
        VerifyError::Lookup(_) => Some(0),
        VerifyError::Stage(StageError::Terminal(stage)) => Some(*stage),
        _ => None,
    }
}

/// Prove (self-checks skipped) and verify the PIOP part of `main` with honest round-2
/// tables whose multiplicities count the in-table cells only; `None` when accepted.
fn rejecting_place(batch: &Batch, main: &MainTables) -> Option<usize> {
    let mut bridge = BridgeGroupTables::build(&batch.manifest, main, &batch.matrices)
        .unwrap_or_else(|_| batch.bridge.clone());
    let m = lenient_multiplicities(SourceTables {
        manifest: &batch.manifest,
        main,
        bridge: &bridge,
    });
    bridge.set_multiplicity(m);
    let tables = SourceTables {
        manifest: &batch.manifest,
        main,
        bridge: &bridge,
    };
    let (proof, _) = prove_tables(batch, tables, true).expect("the unchecked prover runs");
    match verify_tables(batch, bridge.public_y(), &proof) {
        Ok(_) => None,
        Err(error) => {
            let place = rejection(&error);
            assert!(place.is_some(), "unexpected rejection {error:?}");
            place
        }
    }
}

/// The digit cells of one main-domain position of `value`, written as given.
fn write_digits(manifest: &CommitmentManifest, main: &mut MainTables, value: DigitValue, x: usize, digits: &[i64]) {
    for (column, &digit) in agg_piop::manifest::value_columns(value).iter().zip(digits) {
        match manifest.locate(*column) {
            (GroupRole::Main16, _) => *main.wide_cell_mut(manifest, *column, x) = digit as i32,
            (GroupRole::Main, position) => main.columns_mut()[position][x] = digit as i8,
            other => panic!("{other:?}"),
        }
    }
}

/// A synthetic statement: record `i`'s `z[s][j]` (or `u[rho][j]`) set to `value` with the
/// committed digits `digits`, and `T_i` moved so that `A z - T - 2 gamma_2 w_1' - u - q v`
/// stays zero over the integers (`T' = T + A (z' - z)`, resp. `T' = T - (u' - u)`).
enum Target {
    Z,
    U,
}

fn synthetic(batch: &Batch, target: Target, i: usize, row: usize, j: usize, value: i64, digits: &[i64]) -> (Batch, MainTables) {
    let manifest = &batch.manifest;
    let (z, u, _) = batch.main.record_values(manifest, i);
    let mut instance = batch.instance.clone();
    let x = main_index(j, row, i);
    let mut main = batch.main.clone();
    match target {
        Target::Z => {
            let mut moved = z;
            moved[row][j] = value;
            let key = &instance.records[i].key;
            let (before, after) = (az_integer(key, &z), az_integer(key, &moved));
            for rho in 0..4 {
                for k in 0..N {
                    let t = &mut instance.records[i].t[rho][k];
                    *t += i64::try_from(after[rho][k] - before[rho][k]).unwrap();
                    assert!(t.unsigned_abs() < 1 << 31, "T stays an i32");
                }
            }
            write_digits(manifest, &mut main, manifest.z_value(), x, digits);
        }
        Target::U => {
            instance.records[i].t[row][j] -= value - u[row][j];
            write_digits(manifest, &mut main, DigitValue::U, x, digits);
        }
    }
    let synthetic = Batch {
        records: batch.records.clone(),
        instance,
        manifest: manifest.clone(),
        main: batch.main.clone(),
        matrices: batch.matrices.clone(),
        bridge: batch.bridge.clone(),
    };
    // The integer residual still vanishes: the tamper is invisible to the ring identity.
    let residual = residual_table(manifest, &synthetic.instance, &main);
    assert!(residual.iter().all(|r| *r == E::zero()), "ring identity kept");
    (synthetic, main)
}

/// A live position `(i, row, j)` whose honest `z` is within 200 of `target` (so `T` moves
/// by at most `200 (q - 1) / 2`).
fn z_near(batch: &Batch, target: i64) -> (usize, usize, usize) {
    for i in 0..batch.manifest.records() {
        let (z, _, _) = batch.main.record_values(&batch.manifest, i);
        for row in 0..4 {
            for j in 0..N {
                if (z[row][j] - target).abs() <= 200 {
                    return (i, row, j);
                }
            }
        }
    }
    panic!("no z near {target}");
}

#[test]
fn h_exact_piop_round_trip_matches_the_tables_and_the_pointwise_identities() {
    on_large_stack(|| {
        let batch = small_batch(Relation::HExact);
        let manifest = &batch.manifest;
        let tables = batch.tables();
        // Pointwise on the Boolean domain (docs/h-exact.md 6.3): the ring residual is zero,
        // bz_hi and a_z are bits, Q_6(D_u (D_u + 1)) = 0, w_1' < 44.
        assert!(residual_table(manifest, &batch.instance, &batch.main).iter().all(|r| *r == E::zero()));
        for column in manifest.bool_digits() {
            assert!(tables.column(column).to_i64().iter().all(|&b| b == 0 || b == 1), "{}", column.name());
        }
        assert!(tables.column(Column::U_HI).to_i64().iter().all(|&d| leaf_vanishes(6, p_of(d))));
        assert!(w1_table(&batch.main.bit_lanes(manifest)).iter().all(|&w| (0..44).contains(&w)));
        // The honest PIOP verifies and every claim is the extension of its table.
        let (proof, _) = prove_tables(&batch, tables, false).expect("honest prover");
        assert_eq!(proof.stage2.claims.squares.len(), 1);
        assert_eq!(proof.stage3.claims.0.len(), 32);
        assert_eq!(proof.stage2.rounds[0].len(), 4);
        let (points, stage3, _) = verify_tables(&batch, batch.bridge.public_y(), &proof).expect("verifies");
        for group in manifest.groups().iter().filter(|g| g.role != GroupRole::Mult) {
            let point = suffix(&stage3.point, group.logical_vars);
            for &column in &group.columns {
                assert_eq!(stage3.claims.column(manifest, column), tables.column(column).evaluate(&point), "{}", column.name());
            }
        }
        let square: Vec<i64> = tables.column(Column::U_HI).to_i64().iter().map(|&d| d * (d + 1)).collect();
        assert_eq!(proof.stage2.claims.squares[0], tables_mle(&square, &points.r2));
        // A changed a_z claim or P_D claim is rejected at its stage.
        let mut bad = proof.clone();
        let a_z = agg_piop::claims::Stage3Claims::group_offset(manifest, GroupRole::Main) + manifest.locate(Column::A_Z).1;
        bad.stage3.claims.0[a_z] += E::from_u64(1);
        assert_eq!(rejection(&verify_tables(&batch, batch.bridge.public_y(), &bad).unwrap_err()), Some(3));
        let mut bad = proof;
        bad.stage2.claims.squares[0] += E::from_u64(1);
        assert_eq!(rejection(&verify_tables(&batch, batch.bridge.public_y(), &bad).unwrap_err()), Some(2));
    });
}

fn tables_mle(cells: &[i64], point: &[E]) -> E {
    common::mle_i64(cells, point)
}

#[test]
fn h_exact_tamperings_fail_at_the_stated_place() {
    on_large_stack(|| {
        // docs/h-exact.md 6.4, instance level.
        let batch = small_batch(Relation::HExact);
        let manifest = &batch.manifest;
        let x = main_index(17, 2, 5);
        let narrow = |column: Column| manifest.locate(column).1;
        let set = |column: Column, value: i8| {
            let mut main = batch.main.clone();
            main.columns_mut()[narrow(column)][x] = value;
            main
        };
        assert_eq!(rejecting_place(&batch, &batch.main), None, "honest");
        // a_z = 2, bz_hi = 2, bz_hi = -1: stage 3 (bool zero-checks; the ring moves too).
        for (column, value) in [(Column::A_Z, 2), (Column::ZX_HI, 2), (Column::ZX_HI, -1)] {
            assert_eq!(rejecting_place(&batch, &set(column, value)), Some(3), "{} = {value}", column.name());
        }
        // D_u = 3, D_u = -4 (P_D = 12): leaf6 in stage 2.
        for value in [3, -4] {
            assert_eq!(rejecting_place(&batch, &set(Column::U_HI, value)), Some(2), "u_hi = {value}");
        }
        // S_zlo = 2^15, S_ulo = -2^15 - 1: the lookup.
        for (column, stored) in [(Column::ZX_LO, LOOKUP_SHIFT), (Column::U_LO, -LOOKUP_SHIFT - 1)] {
            let mut main = batch.main.clone();
            *main.wide_cell_mut(manifest, column, x) = stored;
            assert_eq!(rejecting_place(&batch, &main), Some(0), "{} = {stored}", column.name());
        }
        // Non-Boolean digits that keep z and the looked-up cell in range, so that the ring
        // identity and the lookup hold: (a_z, bz_hi) from (0, 0) to (-1, 2) with
        // S_zlo - 157 (-130,915 + 131,072 = 157). Only bool[bz_hi], bool[a_z] see it.
        let form = DigitValue::Zx.layout();
        let cells = |m: &MainTables, x: usize| -> Vec<i64> {
            agg_piop::manifest::value_columns(DigitValue::Zx).iter().map(|&c| m.column(manifest, c).value(x)).collect()
        };
        let y = (0..1024 * 50)
            .map(|p| main_index(p % 256, (p / 256) % 4, p / 1024))
            .find(|&y| {
                let d = cells(&batch.main, y);
                d[1] == 0 && d[2] == 0 && d[0] >= -i64::from(LOOKUP_SHIFT) + 157
            })
            .expect("a cell with bz_hi = a_z = 0");
        let honest = cells(&batch.main, y);
        let moved = [honest[0] - 157, 2, -1];
        assert_eq!(form.compose(&moved), form.compose(&honest), "z unchanged");
        let mut main = batch.main.clone();
        write_digits(manifest, &mut main, DigitValue::Zx, y, &moved);
        assert!(residual_table(manifest, &batch.instance, &main).iter().all(|r| *r == E::zero()));
        assert_eq!(rejecting_place(&batch, &main), Some(3), "bool zero-checks alone");
    });
}

#[test]
fn h_exact_value_boundaries_with_synthetic_statements() {
    on_large_stack(|| {
        // docs/h-exact.md 6.4, value boundaries: the tamper keeps the ring identity.
        let exact = small_batch(Relation::HExact);
        let hu = small_batch(Relation::Hu);
        let zx = DigitValue::Zx.layout();
        let z_case = |batch: &Batch, z: i64, digits: &[i64]| {
            let (i, row, j) = z_near(batch, z);
            let (synthetic, main) = synthetic(batch, Target::Z, i, row, j, z, digits);
            assert_eq!(synthetic.manifest.z_value().layout().compose(digits), z);
            rejecting_place(&synthetic, &main)
        };
        // z = +-130,993 (canonical digits): accepted.
        for z in [B_Z, -B_Z] {
            assert_eq!(z_case(&exact, z, &zx.decompose(z)), None, "z = {z}");
        }
        // z = 130,994 as (S_zlo, bz_hi, a_z) = (32,768, 1, 1): the lookup; as (-32,768, 2, 1)
        // or (-32,611, 0, 2) (S_zlo in range): the Booleanity zero-checks.
        assert_eq!(zx.decompose(B_Z + 1), vec![32_768, 1, 1]);
        assert_eq!(z_case(&exact, B_Z + 1, &[32_768, 1, 1]), Some(0));
        assert_eq!(z_case(&exact, B_Z + 1, &[-32_768, 2, 1]), Some(3));
        assert_eq!(z_case(&exact, B_Z + 1, &[-32_611, 0, 2]), Some(3));
        // z = -130,994 with S_zlo = -32,769: the lookup. z = 131,000 (FIPS 204 rejects): the
        // lookup.
        assert_eq!(z_case(&exact, -B_Z - 1, &[-32_769, 0, 0]), Some(0));
        assert_eq!(z_case(&exact, 131_000, &zx.decompose(131_000)), Some(0));
        // H-u proves only |z| < 2^17: the same z = 130,994 and 131,000 are accepted there.
        let z16 = DigitValue::Z.layout();
        for z in [B_Z + 1, 131_000] {
            assert_eq!(z_case(&hu, z, &z16.decompose(z)), None, "H-u z = {z}");
        }
        // u (the same step form in both relations): +-190,465 accepted; 190,466 as
        // (D_u, S_ulo) = (2, 32,768) the lookup, as (3, -30,311) leaf6; -190,466 the lookup.
        let (i, row, j) = (7, 1, 33);
        for batch in [&exact, &hu] {
            let u_case = |u: i64, digits: &[i64]| {
                let (synthetic, main) = synthetic(batch, Target::U, i, row, j, u, digits);
                assert_eq!(U16_DIGITS.compose(digits), u);
                rejecting_place(&synthetic, &main)
            };
            for u in [B_U, -B_U] {
                assert_eq!(u_case(u, &U16_DIGITS.decompose(u)), None, "u = {u}");
            }
            assert_eq!(u_case(B_U + 1, &[32_768, 2]), Some(0));
            assert_eq!(u_case(B_U + 1, &[-30_311, 3]), Some(2));
            assert_eq!(u_case(-B_U - 1, &U16_DIGITS.decompose(-B_U - 1)), Some(0));
        }
    });
}
