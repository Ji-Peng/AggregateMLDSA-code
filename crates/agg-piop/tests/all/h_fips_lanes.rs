//! Relation H-FIPS with polynomial lanes (ML-DSA-65, ML-DSA-87) on the synthetic records of
//! hint weight exactly omega (`agg_witness::gen::hint_corner_record_for`): the coefficient
//! `w' = q - gamma2, h = 0` and the `e = -1` wrap to `w_1' = m - 1 = 15` are committed as the
//! H-FIPS checks require, the committed lane cells equal the generic H-FIPS witness that the
//! native checker accepts, the smallest lane batch with these records proves and verifies, and
//! a hint slack moved off zero is rejected by the weight identity.

use crate::common::{LaneBatch, PiopBatch};

use agg_piop::manifest::{Column, FipsColumn, Relation};
use agg_piop::source_tables::{BridgeGroupTables, MainTables, SourceTables};
use agg_piop::stage::StageError;
use agg_piop::verifier::VerifyError;
use agg_witness::derive::derive_for;
use agg_witness::gen::{self, HINT_CORNER, HINT_ZERO};
use agg_witness::hfips::{check_batch_for, HFipsRecordOf};
use agg_witness::instance::PublicInstance;
use agg_witness::params::{ParamSet, Set65, Set87};

fn on_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("test thread");
}

/// Prove (self-checks skipped) and verify with honest round-2 tables of `main`: `0` for the
/// lookup, `s` for stage `s`, `None` when accepted.
fn rejecting_place(batch: &LaneBatch, main: &MainTables) -> Option<usize> {
    let bridge = BridgeGroupTables::build(&batch.manifest, main, &batch.matrices, batch.bridge.psi()).expect("round 2");
    let tables = SourceTables {
        manifest: &batch.manifest,
        main,
        bridge: &bridge,
    };
    let (proof, _) = batch.prove(tables, true).expect("the unchecked prover runs");
    match batch.verify(bridge.public_y(), &proof) {
        Ok(_) => None,
        Err(VerifyError::Lookup(_)) => Some(0),
        Err(VerifyError::Stage(StageError::Terminal(stage))) => Some(stage),
        Err(other) => panic!("unexpected rejection {other:?}"),
    }
}

/// 15 honest records and the weight-omega records of seeds 1 and 2: 17 records, the smallest
/// lane batch (`N* = 32`).
fn corner_round_trip<P: ParamSet>(seed: u64) {
    let mut records = gen::records_for::<P>(seed, 15);
    records.push(gen::hint_corner_record_for::<P>(1));
    records.push(gen::hint_corner_record_for::<P>(2));
    let batch = LaneBatch::from_records::<P>(Relation::HFips, records, seed);
    let manifest = &batch.manifest;
    let lane_cells = manifest.lane_cells();
    let cell = |column: Column, x: usize| batch.main.column(manifest, column).value(x);
    let slack = batch.main.hint_slack();
    for i in [15, 16] {
        assert_eq!(slack[i], 0, "record {i} has hint weight omega = {}", P::OMEGA);
        // The corner w' = q - gamma2, h = 0: sigma = -1 (s = 0), delta = 0, e = 1.
        let (rho, j) = HINT_CORNER;
        let corner = FipsColumn::ALL.map(|c| cell(Column::Fips(c, rho as u8), j + lane_cells * i));
        assert_eq!(corner, [0, 0, 1], "record {i}: (s, delta, e) at the corner");
        // w' = 0, h = 1: delta = -1 and the wrap e = -1 to w_1' = m - 1.
        let (rho, j) = HINT_ZERO;
        let x = j + lane_cells * i;
        let zero = FipsColumn::ALL.map(|c| cell(Column::Fips(c, rho as u8), x));
        assert_eq!(zero, [0, -1, -1], "record {i}: (s, delta, e) of the wrap");
        let w1: i64 = (0..4).map(|t| cell(Column::W1Bit(t, rho as u8), x) << t).sum();
        assert_eq!(w1, P::M_W - 1, "record {i}: the wrap lands on m - 1");
    }
    // The generic H-FIPS witness of every record passes the native checker (padded rows up
    // to N* included) and is the one the lanes commit: (s, delta, e), the w1 bits and u_h.
    let instance = PublicInstance::<P>::from_records_for(&batch.records).expect("instance");
    let witnesses: Vec<HFipsRecordOf<P>> = instance
        .records
        .iter()
        .zip(&batch.records)
        .map(|(p, r)| {
            let w = derive_for::<P>(&p.key, &p.mu, &r.sig).expect("the signature is accepted");
            HFipsRecordOf::<P>::from_witness(&w).expect("H-FIPS witness")
        })
        .collect();
    let report = check_batch_for::<P>(&instance, &witnesses);
    assert!(report.is_ok(), "{:?}", report.failures);
    assert_eq!(witnesses[16].hint_weight(), P::OMEGA as i64);
    let columns: Vec<_> = (0..P::K)
        .map(|rho| {
            let fips = FipsColumn::ALL.map(|c| batch.main.column(manifest, Column::Fips(c, rho as u8)));
            let bits: Vec<_> = (0..4).map(|t| batch.main.column(manifest, Column::W1Bit(t, rho as u8))).collect();
            (fips, bits)
        })
        .collect();
    for (i, rec) in witnesses.iter().enumerate() {
        assert_eq!(slack[i], rec.u_h, "record {i}: hint slack");
        let w1 = rec.w1();
        for (rho, (fips, bits)) in columns.iter().enumerate() {
            for j in 0..lane_cells {
                let x = j + lane_cells * i;
                let c = rec.coeffs[rho][j];
                let cells = [fips[0].value(x), fips[1].value(x), fips[2].value(x)];
                assert_eq!(cells, [c.s(), c.delta, c.e], "record {i}: (s, delta, e) at ({rho}, {j})");
                let bits: i64 = bits.iter().enumerate().map(|(t, b)| b.value(x) << t).sum();
                assert_eq!(bits, w1[rho][j], "record {i}: w1 at ({rho}, {j})");
            }
        }
    }
    let (proof, _) = batch.prove(batch.tables(), false).expect("honest prover");
    batch.verify(batch.bridge.public_y(), &proof).expect("verifies");
    // The hint slack of a weight-omega record cannot move off zero: u_h = 1 claims weight
    // omega - 1, which the weight identity of stage 3 rejects.
    let mut main = batch.main.clone();
    main.hint_slack_mut()[16] += 1;
    assert_eq!(rejecting_place(&batch, &main), Some(3), "u_h = 1");
}

#[test]
fn ml_dsa_65_h_fips_weight_omega_records_prove_and_verify() {
    on_large_stack(|| corner_round_trip::<Set65>(0x65c));
}

#[test]
fn ml_dsa_87_h_fips_weight_omega_records_prove_and_verify() {
    on_large_stack(|| corner_round_trip::<Set87>(0x87c));
}
