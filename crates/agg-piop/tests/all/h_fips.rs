//! Relation H-FIPS at the PIOP level (the lookup and the
//! three stages, no commitments): honest round trips (random records, and the synthetic
//! weight-80 records with the coefficient `w' = q - gamma2, h = 0` of
//! `agg_witness::gen::hint_corner_record`), the pointwise identities on the Boolean domain,
//! and tamperings of every H-FIPS check rejected at the stated place (`0` the lookup, `2` and
//! `3` the stages), including ones that keep `u` (hence the ring identity) or `L` unchanged so
//! that only the targeted check can catch them.
#![allow(clippy::needless_range_loop)]

use crate::common;

use agg_piop::manifest::{Column, CommitmentManifest, GroupRole, Relation, LOOKUP_SHIFT};
use agg_piop::source_tables::{hint_slack_table, BridgeGroupTables, MainTables, SourceTables};
use agg_piop::stage::{suffix, StageError};
use agg_piop::verifier::VerifyError;
use agg_piop::virtual_polys::{residual_table, u_table, w1_table};
use agg_piop::OpeningField as E;
use agg_witness::derive::derive;
use agg_witness::gen::{self, HINT_CORNER, HINT_ZERO};
use agg_witness::hfips::{honest_coeff, HFipsRecord, L_STEP};
use agg_witness::instance::PublicInstance;
use agg_witness::padding::main_index;
use agg_witness::params::{GAMMA2, N, OMEGA, POSITIONS};
use crate::common::{lenient_multiplicities, prove_tables, verify_tables, Batch};
use jolt_field::{Ring, Zero};

fn on_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("test thread");
}

/// 50 live honest records (N* = 64).
fn small_batch() -> Batch {
    Batch::build_as(Relation::HFips, common::records(0xF1F5, 40, 10), 0xF1F5)
}

/// 6 honest records and the synthetic weight-80 records of seeds 1 and 2 (N* = 8).
fn corner_batch() -> Batch {
    let mut records = common::records(0xF1F6, 6, 0);
    records.push(gen::hint_corner_record(1));
    records.push(gen::hint_corner_record(2));
    Batch::build_as(Relation::HFips, records, 0xF1F6)
}

/// Where the verifier rejected: `0` for the lookup, `s` for stage `s`.
fn rejection(error: &VerifyError) -> Option<usize> {
    match error {
        VerifyError::Lookup(_) => Some(0),
        VerifyError::Stage(StageError::Terminal(stage)) => Some(*stage),
        _ => None,
    }
}

/// Prove (self-checks skipped) and verify the PIOP part of `main` with round-2 tables built
/// from it (the hint slack from `main`), the level-2 stack cell of `u_h` overridden by
/// `uh_cell` if given, and multiplicities counting the in-table cells only; `None` when
/// accepted.
fn rejecting_place_with(batch: &Batch, main: &MainTables, uh_cell: Option<(usize, i32)>) -> Option<usize> {
    // An out-of-table main cell makes the honest recount abort: keep the honest round 2.
    let mut bridge = BridgeGroupTables::build(&batch.manifest, main, &batch.matrices, batch.bridge.psi())
        .unwrap_or_else(|_| batch.bridge.clone());
    if let Some((index, stored)) = uh_cell {
        let cells = batch.manifest.group(GroupRole::Level2).logical_cells();
        bridge.carries_mut(2)[6 * cells + index] = stored;
    }
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

fn rejecting_place(batch: &Batch, main: &MainTables) -> Option<usize> {
    rejecting_place_with(batch, main, None)
}

/// The H-FIPS cells `(s, delta, e, l_lo stored, l_hi)` of main-domain position `x`.
fn cells(manifest: &CommitmentManifest, main: &MainTables, x: usize) -> [i64; 5] {
    [Column::SIGN, Column::DELTA, Column::WRAP, Column::L_LO, Column::L_HI].map(|c| main.column(manifest, c).value(x))
}

/// Write the H-FIPS cells of position `x`.
fn write(manifest: &CommitmentManifest, main: &mut MainTables, x: usize, values: [i64; 5]) {
    for (column, value) in [Column::SIGN, Column::DELTA, Column::WRAP, Column::L_LO, Column::L_HI].into_iter().zip(values) {
        match manifest.locate(column) {
            (GroupRole::Main16, _) => *main.wide_cell_mut(manifest, column, x) = value as i32,
            (GroupRole::Main, position) => main.columns_mut()[position][x] = value as i8,
            other => panic!("{other:?}"),
        }
    }
}

/// `u` of the H-FIPS form from the cells.
fn u_of(c: [i64; 5]) -> i64 {
    let l = c[3] + i64::from(LOOKUP_SHIFT) + L_STEP * c[4];
    GAMMA2 * c[0] + l + 1 - GAMMA2 - 2 * GAMMA2 * c[1] - c[2]
}

/// The first live position whose cells satisfy `pred` (cells, `w_1'`).
fn find(batch: &Batch, pred: impl Fn([i64; 5], i64) -> bool) -> usize {
    let w1 = w1_table(&batch.main.bit_lanes(&batch.manifest));
    (0..batch.manifest.records() * POSITIONS)
        .find(|&x| pred(cells(&batch.manifest, &batch.main, x), w1[x]))
        .expect("a matching position")
}

/// The pointwise H-FIPS identities on the Boolean domain and the
/// agreement of the committed values with the witness crate.
fn check_pointwise(batch: &Batch) {
    let manifest = &batch.manifest;
    let main = &batch.main;
    let w1 = w1_table(&main.bit_lanes(manifest));
    for x in 0..manifest.padded_records() * POSITIONS {
        let [s, delta, e, l_lo, l_hi] = cells(manifest, main, x);
        assert_eq!(s * (s - 1), 0, "F1 at {x}");
        assert_eq!(delta * (delta - (2 * s - 1)), 0, "F2 at {x}");
        assert_eq!(e * (e - 1) * (e - delta), 0, "F3 at {x}");
        assert_eq!(e * (e + 1) * w1[x], 0, "F4 at {x}");
        assert_eq!(e * (e - 1) * (w1[x] - 43), 0, "F5 at {x}");
        assert_eq!(2 * e * w1[x] + 43 * e * e - 43 * e, 0, "F45 at {x}");
        assert_eq!(l_hi * (l_hi - 1), 0, "F6 at {x}");
        assert!((0..1 << 16).contains(&(l_lo + i64::from(LOOKUP_SHIFT))), "L_lo at {x}");
    }
    // F7 per signature, padded ones included (u_h = 80 there).
    let slack = main.hint_slack();
    assert_eq!(slack.len(), manifest.padded_records());
    for i in 0..manifest.padded_records() {
        let weight: i64 = (0..POSITIONS)
            .map(|p| main.column(manifest, Column::DELTA).value(i * POSITIONS + p).pow(2))
            .sum();
        assert_eq!(weight + slack[i], OMEGA as i64, "F7 of signature {i}");
        if i >= manifest.records() {
            assert_eq!(slack[i], OMEGA as i64);
        }
    }
    // The u of the form is the derived u, so the ring residual vanishes.
    assert!(residual_table(manifest, &batch.instance, main).iter().all(|r| *r == E::zero()));
    let u = u_table(manifest, main);
    for (i, (p, r)) in batch.instance.records.iter().zip(&batch.records).enumerate() {
        let w = derive(&p.key, &p.mu, &r.sig).expect("derive");
        let record = HFipsRecord::from_witness(&w).expect("H-FIPS witness");
        assert_eq!(record.u_h, slack[i], "u_h of record {i}");
        for rho in 0..4 {
            for j in 0..N {
                let x = main_index(j, rho, i);
                assert_eq!(u[x], w.u[rho][j]);
                let c = record.coeffs[rho][j];
                let got = cells(manifest, main, x);
                assert_eq!(
                    got,
                    [(1 + c.sigma) / 2, c.delta, c.e, c.l_lo - i64::from(LOOKUP_SHIFT), c.l_hi],
                    "record {i} position ({rho}, {j})"
                );
            }
        }
    }
    // u_h on D_2: u_h(i) at i 2^(d_2 - d_sig), zero elsewhere.
    let table = hint_slack_table(manifest, slack);
    let column = batch.bridge.group(manifest, GroupRole::Level2)[12].to_i64();
    assert_eq!(column.len(), table.len());
    for (stored, value) in column.iter().zip(&table) {
        assert_eq!(stored + i64::from(LOOKUP_SHIFT), *value);
    }
}

#[test]
fn h_fips_piop_round_trip_matches_the_tables_and_the_pointwise_identities() {
    on_large_stack(|| {
        let batch = small_batch();
        let manifest = &batch.manifest;
        check_pointwise(&batch);
        let tables = batch.tables();
        let (proof, _) = prove_tables(&batch, tables, false).expect("honest prover");
        // The leaf digits are the projection digits K_2 and K_3.
        assert_eq!(proof.stage2.claims.squares.len(), 2);
        assert_eq!(proof.stage2.claims.fips.len(), 2);
        assert_eq!(proof.stage3.claims.0.len(), 40);
        assert_eq!(proof.stage2.rounds[0].len(), 5);
        let (points, stage3, _) = verify_tables(&batch, batch.bridge.public_y(), &proof).expect("verifies");
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
        // Stage 2's delta and e claims are their extensions at r^(2).
        for (k, column) in [Column::DELTA, Column::WRAP].into_iter().enumerate() {
            assert_eq!(proof.stage2.claims.fips[k], tables.column(column).evaluate(&points.r2));
        }
        // A changed delta claim is rejected (stage 3's reduction).
        let mut bad = proof.clone();
        bad.stage2.claims.fips[0] += E::from_u64(1);
        assert!(verify_tables(&batch, batch.bridge.public_y(), &bad).is_err());
        let mut bad = proof;
        let uh = agg_piop::claims::Stage3Claims::group_offset(manifest, GroupRole::Level2) + 12;
        bad.stage3.claims.0[uh] += E::from_u64(1);
        assert_eq!(rejection(&verify_tables(&batch, batch.bridge.public_y(), &bad).unwrap_err()), Some(3));
    });
}

#[test]
fn h_fips_weight_80_records_with_the_corner_prove_and_verify() {
    on_large_stack(|| {
        let batch = corner_batch();
        let manifest = &batch.manifest;
        check_pointwise(&batch);
        let slack = batch.main.hint_slack();
        for i in [6, 7] {
            assert_eq!(slack[i], 0, "record {i} has hint weight 80");
            let corner = main_index(HINT_CORNER.1, HINT_CORNER.0, i);
            assert_eq!(
                cells(manifest, &batch.main, corner),
                [0, 0, 1, -i64::from(LOOKUP_SHIFT), 0],
                "the corner witness (sigma, L, delta, e) = (-1, 0, 0, 1)"
            );
            let zero = main_index(HINT_ZERO.1, HINT_ZERO.0, i);
            assert_eq!(cells(manifest, &batch.main, zero)[..3], [0, -1, -1], "the e = -1 wrap");
        }
        assert_eq!(rejecting_place(&batch, &batch.main), None, "honest");
        let (proof, _) = prove_tables(&batch, batch.tables(), false).expect("honest prover");
        verify_tables(&batch, batch.bridge.public_y(), &proof).expect("verifies");
        // The witness of the corner (sigma, L, delta, e) = (1, gamma2 - 1, 1, 0) keeps
        // u (and the ring identity) but makes the weight 81: with u_h = 0 the weight identity
        // fails in stage 3; with u_h = -1 the lookup rejects.
        let corner = main_index(HINT_CORNER.1, HINT_CORNER.0, 6);
        let honest = cells(manifest, &batch.main, corner);
        let ideator = [1, 1, 0, GAMMA2 - 1 - L_STEP - i64::from(LOOKUP_SHIFT), 1];
        assert_eq!(u_of(ideator), u_of(honest));
        let mut main = batch.main.clone();
        write(manifest, &mut main, corner, ideator);
        assert!(residual_table(manifest, &batch.instance, &main).iter().all(|r| *r == E::zero()));
        assert_eq!(rejecting_place(&batch, &main), Some(3), "weight 81 with u_h = 0");
        let uh_index = 6 << (manifest.group(GroupRole::Level2).logical_vars - manifest.d_sig());
        assert_eq!(
            rejecting_place_with(&batch, &main, Some((uh_index, -LOOKUP_SHIFT - 1))),
            Some(0),
            "u_h = -1"
        );
    });
}

#[test]
fn h_fips_tamperings_fail_at_the_stated_place() {
    on_large_stack(|| {
        let batch = small_batch();
        let manifest = &batch.manifest;
        assert_eq!(rejecting_place(&batch, &batch.main), None, "honest");
        let tampered = |x: usize, f: &dyn Fn([i64; 5]) -> [i64; 5]| -> Option<usize> {
            let mut main = batch.main.clone();
            let values = f(cells(manifest, &main, x));
            write(manifest, &mut main, x, values);
            rejecting_place(&batch, &main)
        };
        let keeps_u = |x: usize, f: &dyn Fn([i64; 5]) -> [i64; 5]| {
            let before = cells(manifest, &batch.main, x);
            assert_eq!(u_of(f(before)), u_of(before), "the tamper keeps u at {x}");
        };
        let plain = find(&batch, |c, w| c[1] == 0 && c[2] == 0 && w != 0 && w != 43 && c[0] == 1);
        // F1: s = 2 (stage 3; the ring moves too).
        assert_eq!(tampered(plain, &|c| [2, c[1], c[2], c[3], c[4]]), Some(3), "s = 2");
        // F2: delta = -sigma = -1 at s = 1 (stage 3; the ring and the weight move too).
        assert_eq!(tampered(plain, &|c| [c[0], -1, c[2], c[3], c[4]]), Some(3), "delta = -sigma");
        // F3: e = -1 at delta = 0 with L - 1 (u = gamma_2 s + L + 1 - gamma_2 - e) keeping u:
        // only F3 (stage 2) and F45 (stage 3) see it; stage 2 comes first.
        let f3 = |c: [i64; 5]| [c[0], c[1], -1, c[3] - 1, c[4]];
        let f3_at = find(&batch, |c, w| {
            c[0] == 1 && c[1] == 0 && c[2] == 0 && c[4] == 0 && c[3] >= 1 - i64::from(LOOKUP_SHIFT) && w != 0 && w != 43
        });
        keeps_u(f3_at, &f3);
        assert_eq!(tampered(f3_at, &f3), Some(2), "e = -1 at delta = 0");
        // F4: e = 1 at w1' != 0 with L + 1 keeping u: F3 holds (e = 1), only F45 sees it.
        let f4 = |c: [i64; 5]| [c[0], c[1], 1, c[3] + 1, c[4]];
        let f4_at = find(&batch, |c, w| {
            c[0] == 1 && c[1] == 0 && c[2] == 0 && c[4] == 0 && c[3] + 1 < i64::from(LOOKUP_SHIFT) && w != 0
        });
        keeps_u(f4_at, &f4);
        assert_eq!(tampered(f4_at, &f4), Some(3), "e = 1 at w1' != 0 (F45 alone)");
        // F5: e = -1 at delta = -1 and w1' != 43 with L - 1 keeping u: F3 holds (e = delta),
        // only F45 sees it.
        let f5 = |c: [i64; 5]| [c[0], c[1], -1, c[3] - 1, c[4]];
        let f5_at = find(&batch, |c, w| {
            c[0] == 0 && c[1] == -1 && c[2] == 0 && c[3] >= 1 - i64::from(LOOKUP_SHIFT) && w != 43
        });
        keeps_u(f5_at, &f5);
        assert_eq!(tampered(f5_at, &f5), Some(3), "e = -1 at w1' != 43 (F45 alone)");
        // F6: l_hi = -1 with l_lo + 29,696 keeps L and u: only bool[l_hi].
        let f6 = |c: [i64; 5]| [c[0], c[1], c[2], c[3] + L_STEP, -1];
        let f6_at = find(&batch, |c, _| c[4] == 0 && c[3] + L_STEP < i64::from(LOOKUP_SHIFT));
        keeps_u(f6_at, &f6);
        assert_eq!(tampered(f6_at, &f6), Some(3), "l_hi = -1");
        // The other split of an L in the overlap is a valid witness.
        let split = |c: [i64; 5]| [c[0], c[1], c[2], c[3] - L_STEP, 1];
        let split_at = find(&batch, |c, _| c[4] == 0 && c[3] - L_STEP >= -i64::from(LOOKUP_SHIFT));
        keeps_u(split_at, &split);
        assert_eq!(tampered(split_at, &split), None, "the other L split");
        // l_lo = 2^16 (stored 2^15): the lookup.
        assert_eq!(
            tampered(plain, &|c| [c[0], c[1], c[2], i64::from(LOOKUP_SHIFT), c[4]]),
            Some(0),
            "l_lo = 2^16"
        );
        // F7: u_h + 1 (stage 3), u_h - 1 (stage 3), and a dropped hint (delta -> 0, u_h + 1:
        // the ring moves, stage 3).
        for delta in [1_i64, -1] {
            let mut main = batch.main.clone();
            main.hint_slack_mut()[3] += delta;
            assert_eq!(rejecting_place(&batch, &main), Some(3), "u_h {delta:+}");
        }
        let hinted = find(&batch, |c, _| c[1] != 0 && c[2] == 0);
        let mut main = batch.main.clone();
        let mut values = cells(manifest, &main, hinted);
        values[1] = 0;
        write(manifest, &mut main, hinted, values);
        main.hint_slack_mut()[hinted / POSITIONS] += 1;
        assert_eq!(rejecting_place(&batch, &main), Some(3), "a dropped hint");
        // A u_h cell outside T_16 at a padded D_2 position (value 0 honestly): the lookup.
        let cells2 = manifest.group(GroupRole::Level2).logical_cells();
        assert_eq!(
            rejecting_place_with(&batch, &batch.main, Some((cells2 - 1, LOOKUP_SHIFT))),
            Some(0),
            "u_h cell = 2^16"
        );
    });
}

#[test]
fn h_fips_records_decode_and_encode_through_the_main_tables() {
    // record_values decodes u by the H-FIPS form; the hint slack is 80 minus the hint weight.
    let records = common::records(0xF1F7, 5, 0);
    let instance = PublicInstance::from_records(&records).unwrap();
    let manifest = CommitmentManifest::for_relation(records.len(), Relation::HFips).unwrap();
    let signatures: Vec<&[u8]> = records.iter().map(|r| r.sig.as_slice()).collect();
    let main = MainTables::build(&manifest, &instance, &signatures).unwrap();
    for (i, (p, r)) in instance.records.iter().zip(&records).enumerate() {
        let w = derive(&p.key, &p.mu, &r.sig).unwrap();
        let (z, u, v) = main.record_values(&manifest, i);
        assert_eq!((z, u, v), (w.z, w.u, w.v));
        let weight: i64 = w.h.iter().flatten().map(|&b| i64::from(b)).sum();
        assert_eq!(main.hint_slack()[i], OMEGA as i64 - weight);
        let c = honest_coeff(w.w[0][0], w.h[0][0] != 0);
        assert_eq!(main.column(&manifest, Column::DELTA).value(main_index(0, 0, i)), c.delta);
    }
    assert_eq!(main.hint_slack()[5..], [OMEGA as i64; 3]);
}

/// Stage timings of H-u and H-FIPS on the same records (diagnostic, run with
/// `AGG_STAGE_TRACE=1 ... -- --ignored --nocapture h_fips_stage_trace`; `AGG_TRACE_N` sets
/// the batch size, default 8,192).
#[test]
#[ignore = "diagnostic: stage timings at 8,192 records"]
fn h_fips_stage_trace() {
    on_large_stack(|| {
        let n: usize = std::env::var("AGG_TRACE_N").ok().and_then(|v| v.parse().ok()).unwrap_or(8192);
        let records = common::records(0xF1F8, n - n / 32, n / 32);
        for relation in [Relation::Hu, Relation::HFips] {
            let batch = Batch::build_as(relation, records.clone(), 0xF1F8);
            let start = std::time::Instant::now();
            let (proof, _) = prove_tables(&batch, batch.tables(), false).expect("prover");
            eprintln!("{relation}: PIOP part (lookup and stages) {:.1} ms", start.elapsed().as_secs_f64() * 1e3);
            verify_tables(&batch, batch.bridge.public_y(), &proof).expect("verifies");
        }
    });
}
