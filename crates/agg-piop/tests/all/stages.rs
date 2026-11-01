//! The PIOP part (the lookup into `T_16` and the three sum-check stages)
//! on real batches of ML-DSA-44 signatures: honest round trips whose claims equal the
//! extensions of the committed and virtual tables, and tampered witnesses and proofs
//! rejected at the stated place (`docs/lookup.md`).
#![allow(clippy::needless_range_loop)]

use crate::common;

use aerie_falcon::protocol::algebra::{evaluate_mle, lift_table};
use agg_piop::lookup::LookupError;
use agg_piop::manifest::{Column, GroupRole, LOOKUP_SHIFT};
use agg_piop::prover::PiopProof;
use agg_piop::source_tables::{BridgeGroupTables, MainTables, SourceTables};
use agg_piop::stage::{suffix, StageError};
use agg_piop::verifier::VerifyError;
use agg_piop::virtual_polys::parity1_tables;
use agg_piop::OpeningField as E;
use agg_witness::padding::main_index;
use agg_witness::params::W1_BITS;
use crate::common::{lenient_multiplicities, prove_tables, verify_tables, Batch};
use jolt_field::Ring;

fn on_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("test thread");
}

/// Where the verifier rejected: `0` for the lookup, `s` for stage `s`.
fn rejection(error: &VerifyError) -> Option<usize> {
    match error {
        VerifyError::Lookup(_) => Some(0),
        VerifyError::Stage(StageError::Terminal(stage)) => Some(*stage),
        _ => None,
    }
}

/// Prove (self-checks skipped) and verify the PIOP part of the given tables.
fn rejecting_place(batch: &Batch, main: &MainTables, bridge: &BridgeGroupTables) -> Option<usize> {
    let tables = SourceTables {
        manifest: &batch.manifest,
        main,
        bridge,
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

/// Honest round-2 tables for (possibly tampered) round-1 tables whose looked-up cells are
/// in `T_16`.
fn rebuilt(batch: &Batch, main: &MainTables) -> BridgeGroupTables {
    BridgeGroupTables::build(&batch.manifest, main, &batch.matrices, batch.bridge.psi()).expect("honest round 2")
}

/// The honest round-2 tables with the multiplicities of the in-table cells only.
fn lenient(batch: &Batch, main: &MainTables, mut bridge: BridgeGroupTables) -> BridgeGroupTables {
    let m = lenient_multiplicities(SourceTables {
        manifest: &batch.manifest,
        main,
        bridge: &bridge,
    });
    bridge.set_multiplicity(m);
    bridge
}

fn mle_of(cells: &[i64], point: &[E]) -> E {
    evaluate_mle(&lift_table(cells), point)
}

/// Every claim the honest PIOP leaves equals the extension of its table: the 35 stage-3
/// claims, the multiplicity digits at `pi_m`, the lookup's column claims at the stack
/// points, and the stage-1 and stage-2 terminals of the virtual tables.
fn check_honest(batch: &Batch, proof: &PiopProof) {
    let manifest = &batch.manifest;
    let tables = batch.tables();
    let (points, stage3, pi_m) = verify_tables(batch, batch.bridge.public_y(), proof).expect("honest PIOP verifies");
    // Stage-3 claims: the committed columns at the suffixes of r^(3).
    for group in manifest.groups().iter().filter(|g| g.role != GroupRole::Mult) {
        let point = suffix(&stage3.point, group.logical_vars);
        for &column in group.columns.iter().filter(|c| !c.is_zero()) {
            assert_eq!(
                stage3.claims.column(manifest, column),
                tables.column(column).evaluate(&point),
                "{}",
                column.name()
            );
        }
    }
    // The multiplicity digits at pi_m, and m itself.
    let digits = tables.group(GroupRole::Mult);
    for (k, cells) in digits.iter().enumerate() {
        assert_eq!(proof.multiplicities.0[k], cells.evaluate(&pi_m));
    }
    let m: Vec<i64> = batch.bridge.multiplicity().iter().map(|&c| i64::from(c)).collect();
    assert_eq!(proof.multiplicities.compose(), mle_of(&m, &pi_m));
    // Brute-force multiplicities: every looked-up cell (padding slots included) counted.
    let shape = manifest.lookup_shape();
    let mut count = vec![0_i64; 1 << 16];
    for stack in tables.lookup_stacks().unwrap() {
        for &stored in stack.stacked() {
            count[(stored + LOOKUP_SHIFT) as usize] += 1;
        }
    }
    assert_eq!(count, m);
    assert_eq!(m.iter().sum::<i64>(), shape.lookups(0) as i64);
    // The stack reductions: sum_c w_c f_c(pi_low) at the low points of the stacks.
    for (j, low) in points.lookup.iter().enumerate() {
        assert_eq!(low.len(), shape.stacks[j].vars);
        let columns = manifest.stack_columns(j);
        assert!(columns.iter().all(|c| tables.column(*c).len() == 1 << low.len()));
    }
    // Stage 1 terminal: y^(1) (virtual), y^(2), b at r^(1); stage 2: b and P_Z at r^(2).
    let y1 = parity1_tables(tables, &batch.matrices.m1);
    let r1_1 = suffix(&points.r1, manifest.domain_vars(GroupRole::Level1));
    let r1_2 = suffix(&points.r1, manifest.domain_vars(GroupRole::Level2));
    for t in 0..W1_BITS {
        assert_eq!(proof.stage1.claims.y1[t], mle_of(&y1[t], &r1_1));
        assert_eq!(
            proof.stage1.claims.y2[t],
            tables.column(Column::Parity2(t as u8)).evaluate(&r1_2)
        );
        let bit = tables.column(Column::W1Bit(t as u8, 0));
        assert_eq!(proof.stage1.claims.b[t], bit.evaluate(&points.r1));
        assert_eq!(proof.stage2.claims.b[t], bit.evaluate(&points.r2));
    }
    for (k, column) in manifest.leaf_digits().into_iter().enumerate() {
        let square: Vec<i64> = tables.column(column).to_i64().iter().map(|&z| z * (z + 1)).collect();
        let r2 = suffix(&points.r2, manifest.domain_vars(column.domain()));
        assert_eq!(proof.stage2.claims.squares[k], mle_of(&square, &r2));
    }
    // The kappa^(1) claims before stage 2.
    for t in 0..W1_BITS {
        assert_eq!(
            proof.stage2.kappa1.0[t],
            tables.column(Column::carry(1, t as u8)).evaluate(&r1_1)
        );
    }
}

#[test]
fn piop_round_trip_on_256_signatures_matches_the_tables() {
    on_large_stack(|| {
        let batch = Batch::build(common::records(0x256, 200, 56), 0x256);
        let (proof, run) = prove_tables(&batch, batch.tables(), false).expect("honest prover");
        assert_eq!(run.points.lookup.len(), 5);
        assert_eq!(run.pi_m.len(), 16);
        check_honest(&batch, &proof);
        // A changed round message or terminal claim is rejected at its stage.
        let mut bad = proof.clone();
        bad.stage1.rounds[3][1] += E::from_u64(1);
        assert_eq!(rejection(&verify_tables(&batch, batch.bridge.public_y(), &bad).unwrap_err()), Some(1));
        let mut bad = proof.clone();
        bad.stage2.claims.squares[1] += E::from_u64(1);
        assert_eq!(rejection(&verify_tables(&batch, batch.bridge.public_y(), &bad).unwrap_err()), Some(2));
        let mut bad = proof.clone();
        bad.prestage.kappa2[2] += E::from_u64(1);
        assert_eq!(rejection(&verify_tables(&batch, batch.bridge.public_y(), &bad).unwrap_err()), Some(1));
        let mut bad = proof;
        bad.stage3.claims.0[7] += E::from_u64(1);
        assert_eq!(rejection(&verify_tables(&batch, batch.bridge.public_y(), &bad).unwrap_err()), Some(3));
    });
}

fn small_batch() -> Batch {
    // 50 live records pad to N* = 64.
    Batch::build(common::records(0x51, 40, 10), 0x51)
}

/// A main-domain position of a live record whose `Z_hi` is `hi`, and its cell index.
fn cell_with(batch: &Batch, column: Column, accept: impl Fn(i64) -> bool) -> usize {
    let cells = batch.main.column(&batch.manifest, column);
    (0..1024 * 50)
        .map(|p| main_index(p % 256, (p / 256) % 4, p / 1024))
        .find(|&x| accept(cells.value(x)))
        .expect("a matching cell")
}

#[test]
fn lookup_rejects_main_cells_outside_the_table() {
    on_large_stack(|| {
        let batch = small_batch();
        let manifest = &batch.manifest;
        // z unchanged: Z_lo + 2^16 with Z_hi - 1 (looked-up value in [2^16, 2^17)), and
        // Z_lo - 2^16 with Z_hi + 1 (a negative looked-up value). Only the lookup can tell.
        for (delta, accept) in [(1_i32, -1_i64..=1), (-1, -2..=0)] {
            let x = cell_with(&batch, Column::Z_HI, |v| accept.contains(&v));
            let mut main = batch.main.clone();
            *main.wide_cell_mut(manifest, Column::Z_LO, x) += delta << 16;
            let (_, z_hi) = manifest.locate(Column::Z_HI);
            main.columns_mut()[z_hi][x] -= delta as i8;
            let f = main.column(manifest, Column::Z_LO).value(x) + i64::from(LOOKUP_SHIFT);
            assert!(!(0..1 << 16).contains(&f), "looked-up value {f}");
            let bridge = lenient(&batch, &main, batch.bridge.clone());
            assert_eq!(rejecting_place(&batch, &main, &bridge), Some(0), "delta {delta}");
        }
        // v_hi = 2^16 (stored 2^15) is no longer looked up: v moves by about 2^31, and with
        // the outputs k recomputed the ring identity rejects (stage 3).
        let mut main = batch.main.clone();
        *main.wide_cell_mut(manifest, Column::V_HI, 77) = LOOKUP_SHIFT;
        let bridge = rebuilt(&batch, &main);
        assert_eq!(rejecting_place(&batch, &main, &bridge), Some(3));
    });
}

#[test]
fn lookup_rejects_carries_outside_the_table() {
    on_large_stack(|| {
        let batch = small_batch();
        for (level, stored) in [(1, LOOKUP_SHIFT), (1, -LOOKUP_SHIFT - 1), (2, LOOKUP_SHIFT), (3, -LOOKUP_SHIFT - 7)] {
            let mut bridge = batch.bridge.clone();
            bridge.carries_mut(level)[5] = stored;
            let bridge = lenient(&batch, &batch.main, bridge);
            assert_eq!(
                rejecting_place(&batch, &batch.main, &bridge),
                Some(0),
                "level {level}, stored {stored}"
            );
        }
        // A padding slot of the level-1 stack (lanes 6, 7 are public: stored -2^15, looked-up
        // value 0). Stored 0 looks up 2^15, which is in the table, so the lookup accepts;
        // the verifier's stack reduction assumes the public padding, so stage 3 rejects.
        let mut bridge = batch.bridge.clone();
        let slot = 6 << batch.manifest.domain_vars(GroupRole::Level1);
        bridge.carries_mut(1)[slot + 3] = 0;
        bridge.recount(&batch.manifest, &batch.main).unwrap();
        assert_eq!(rejecting_place(&batch, &batch.main, &bridge), Some(3));
    });
}

#[test]
fn lookup_rejects_a_wrong_multiplicity() {
    on_large_stack(|| {
        let batch = small_batch();
        let honest = batch.bridge.multiplicity().to_vec();
        // One count moved to another entry (same total), and one count added.
        let mut moved = honest.clone();
        let t = honest.iter().position(|&m| m > 0).unwrap();
        moved[t] -= 1;
        moved[(t + 1) % (1 << 16)] += 1;
        let mut added = honest;
        added[12_345] += 1;
        for m in [moved, added] {
            let mut bridge = batch.bridge.clone();
            bridge.set_multiplicity(m);
            assert_eq!(rejecting_place(&batch, &batch.main, &bridge), Some(0));
        }
    });
}

#[test]
fn tampered_lookup_messages_and_claims_are_rejected() {
    on_large_stack(|| {
        let batch = small_batch();
        let (proof, _) = prove_tables(&batch, batch.tables(), false).expect("honest prover");
        let y = batch.bridge.public_y();
        assert!(verify_tables(&batch, y, &proof).is_ok());
        let lookup_error = |bad: &PiopProof| match verify_tables(&batch, y, bad) {
            Err(VerifyError::Lookup(e)) => e,
            other => panic!("expected a lookup rejection, got {other:?}"),
        };
        // A layer-1 value: the root identity.
        let mut bad = proof.clone();
        bad.lookup.layer1[2][0] += E::from_u64(1);
        assert!(matches!(lookup_error(&bad), LookupError::Identity { .. }));
        // A GKR round message of a middle session, a layer terminal, a table leaf.
        let mut bad = proof.clone();
        bad.lookup.sessions[5].rounds[2][1] += E::from_u64(1);
        assert!(matches!(lookup_error(&bad), LookupError::Terminal(_)));
        let mut bad = proof.clone();
        bad.lookup.sessions[3].terminals[1][2] += E::from_u64(1);
        assert!(matches!(lookup_error(&bad), LookupError::Terminal(_)));
        let table_session = proof
            .lookup
            .sessions
            .iter()
            .position(|s| s.rounds.len() == 15)
            .expect("the session of depth 16");
        let mut bad = proof.clone();
        let leaves = &mut bad.lookup.sessions[table_session].leaves;
        let last = leaves.len() - 1;
        leaves[last].swap(0, 1);
        assert!(matches!(lookup_error(&bad), LookupError::Terminal(_)));
        // Swapped multiplicity-digit claims: they no longer compose to m(pi_m).
        let mut bad = proof.clone();
        bad.multiplicities.0.swap(0, 1);
        assert!(matches!(lookup_error(&bad), LookupError::Terminal(0)));
        // A swapped stack leaf pair passes its (symmetric) session but moves the stacked
        // claim; with the honest stage proofs (computed for the honest transcript) the
        // verifier rejects at stage 1. The claim-level rejection (a prover that continues
        // from the swapped leaves) is the unit test
        // `lookup::tests::tampered_messages_and_claims_are_rejected`.
        let deepest = proof.lookup.sessions.len() - 1;
        let mut bad = proof.clone();
        bad.lookup.sessions[deepest].leaves[0].swap(0, 1);
        assert!(verify_tables(&batch, y, &bad).is_err());
        // A truncated session.
        let mut bad = proof;
        bad.lookup.sessions.pop();
        assert!(matches!(lookup_error(&bad), LookupError::Shape(_)));
    });
}

#[test]
fn swapped_column_claims_are_rejected() {
    on_large_stack(|| {
        let batch = small_batch();
        let manifest = &batch.manifest;
        let (proof, _) = prove_tables(&batch, batch.tables(), false).expect("honest prover");
        let position = |column: Column| {
            let (role, at) = manifest.locate(column);
            agg_piop::claims::Stage3Claims::group_offset(manifest, role) + at
        };
        // Two lookup columns of the main stack, two carries of one stack, a carry and a parity.
        for (a, b) in [
            (Column::Z_LO, Column::U_LO),
            (Column::V_LO, Column::V_HI),
            (Column::carry(1, 0), Column::carry(1, 1)),
            (Column::Parity2(0), Column::carry(2, 0)),
        ] {
            let mut bad = proof.clone();
            bad.stage3.claims.0.swap(position(a), position(b));
            assert_eq!(
                rejection(&verify_tables(&batch, batch.bridge.public_y(), &bad).unwrap_err()),
                Some(3),
                "{} <-> {}",
                a.name(),
                b.name()
            );
        }
    });
}

#[test]
fn values_valid_only_for_another_range_are_rejected() {
    on_large_stack(|| {
        let batch = small_batch();
        let manifest = &batch.manifest;
        // Z_hi = 2 is in U_hi's base-6 alphabet but not in Z_hi's base-4 one: the leaf.
        let mut main = batch.main.clone();
        let (_, z_hi) = manifest.locate(Column::Z_HI);
        main.columns_mut()[z_hi][main_index(3, 1, 4)] = 2;
        let bridge = rebuilt(&batch, &main);
        assert_eq!(rejecting_place(&batch, &main, &bridge), Some(2));
        // U_hi = 3 and U_hi = -4 are in a base-8 alphabet but outside H-u's
        // base-6 one [-3, 2]: leaf6 (stage 2).
        for value in [3_i8, -4] {
            let mut main = batch.main.clone();
            let (_, u_hi) = manifest.locate(Column::U_HI);
            main.columns_mut()[u_hi][main_index(9, 2, 7)] = value;
            let bridge = rebuilt(&batch, &main);
            assert_eq!(rejecting_place(&batch, &main, &bridge), Some(2), "u_hi = {value}");
        }
        // A carry stored unshifted (K = kappa instead of kappa - 2^15) is in the table, but
        // as a carry it means kappa + 2^15: the level-1 identity of stage 1 fails.
        let mut bridge = batch.bridge.clone();
        let cell = bridge.carries(1).iter().position(|&k| k > -LOOKUP_SHIFT).expect("a nonzero carry");
        bridge.carries_mut(1)[cell] += LOOKUP_SHIFT;
        bridge.recount(manifest, &batch.main).expect("in the table");
        assert_eq!(rejecting_place(&batch, &batch.main, &bridge), Some(1));
        // A low part valid only for a wider range: U_lo + 63,079 with U_hi - 1 keeps u
        // (relation H-u's step form), U_hi in its base-6 alphabet, so the leaves and the
        // ring identity hold; when the looked-up value lands in [2^16, 2^17) only the lookup
        // sees it. (A value booked under another table is the lookup.rs unit test
        // `two_tables_verify_and_a_cross_table_value_is_rejected`; there is one table here.)
        // V_lo + 2^16 with V_hi - 1 keeps v and is accepted: the digits of v are not
        // range-checked (tests/all/projection.rs).
        let u_cells = batch.main.column(manifest, Column::U_LO);
        let u_hi_cells = batch.main.column(manifest, Column::U_HI);
        let u_cell = (0..1024 * 50)
            .map(|p| main_index(p % 256, (p / 256) % 4, p / 1024))
            .find(|&x| u_hi_cells.value(x) > -3 && u_cells.value(x) + i64::from(LOOKUP_SHIFT) + 63_079 >= 1 << 16)
            .expect("a matching cell");
        let mut main = batch.main.clone();
        *main.wide_cell_mut(manifest, Column::U_LO, u_cell) += 63_079;
        main.columns_mut()[manifest.locate(Column::U_HI).1][u_cell] -= 1;
        let f = main.column(manifest, Column::U_LO).value(u_cell) + i64::from(LOOKUP_SHIFT);
        assert!((1 << 16..1 << 17).contains(&f), "looked-up value {f}");
        let bridge = lenient(&batch, &main, batch.bridge.clone());
        assert_eq!(rejecting_place(&batch, &main, &bridge), Some(0));
    });
}

#[test]
fn tampered_witnesses_fail_at_the_stated_stage() {
    on_large_stack(|| {
        let batch = small_batch();
        let manifest = &batch.manifest;
        let x = main_index(17, 2, 5);
        let narrow = |column: Column| manifest.locate(column).1;
        let case = |label: &str, main: MainTables, expected: Option<usize>| {
            let bridge = rebuilt(&batch, &main);
            assert_eq!(rejecting_place(&batch, &main, &bridge), expected, "{label}");
        };
        // A 16-bit column changed inside the table: the ring identity (stage 3).
        for column in [Column::Z_LO, Column::U_LO, Column::V_LO, Column::V_HI] {
            let mut main = batch.main.clone();
            let cell = main.wide_cell_mut(manifest, column, x);
            *cell = if *cell < LOOKUP_SHIFT - 1 { *cell + 1 } else { *cell - 1 };
            case(&column.name(), main, Some(3));
        }
        // Z_hi changed inside its alphabet: the ring identity.
        let mut main = batch.main.clone();
        let cell = &mut main.columns_mut()[narrow(Column::Z_HI)][x];
        *cell = if *cell < 1 { *cell + 1 } else { *cell - 1 };
        case("z_hi", main, Some(3));
        // A flipped bit (round-2 tables rebuilt): w_1' changes, the ring identity fails.
        let mut main = batch.main.clone();
        main.columns_mut()[narrow(Column::W1Bit(0, 0))][x] ^= 1;
        case("flipped bit", main, Some(3));
        // A non-Boolean bit and w_1' = 44: stage 1.
        let mut main = batch.main.clone();
        let column = narrow(Column::W1Bit(1, 0));
        let y = (0..1024)
            .map(|p| main_index(p % 256, p / 256, 7))
            .find(|&y| main.columns()[column][y] == 0)
            .unwrap();
        main.columns_mut()[column][y] = 2;
        case("bit = 2", main, Some(1));
        let mut main = batch.main.clone();
        for (t, bit) in [0, 0, 1, 1, 0, 1].into_iter().enumerate() {
            main.columns_mut()[narrow(Column::W1Bit(t as u8, 0))][x] = bit;
        }
        case("w1' = 44", main, Some(1));
        // The honest tables pass.
        case("honest", batch.main.clone(), None);
    });
}

#[test]
fn tampered_bridge_tables_fail_in_stage_one() {
    on_large_stack(|| {
        let batch = small_batch();
        let manifest = &batch.manifest;
        let check = |label: &str, bridge: BridgeGroupTables| {
            assert_eq!(rejecting_place(&batch, &batch.main, &bridge), Some(1), "{label}");
        };
        for level in 1..=3 {
            let mut bridge = batch.bridge.clone();
            bridge.carries_mut(level)[3] += 1;
            bridge.recount(manifest, &batch.main).unwrap();
            check(&format!("level-{level} carry"), bridge);
        }
        let mut bridge = batch.bridge.clone();
        bridge.parities_mut()[2][5] ^= 1;
        check("level-2 parity", bridge);
        let mut bridge = batch.bridge.clone();
        bridge.public_y_mut()[11] ^= 1;
        check("Y bit", bridge);
    });
}

#[test]
fn padded_batch_of_ten_records_round_trips() {
    on_large_stack(|| {
        let batch = Batch::build(common::records(0x10, 7, 3), 0x10);
        assert_eq!(batch.manifest.padded_records(), 16);
        let (proof, _) = prove_tables(&batch, batch.tables(), false).expect("honest prover");
        check_honest(&batch, &proof);
    });
}
