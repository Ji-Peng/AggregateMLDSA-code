//! Round-1 and round-2 tables of real ML-DSA-44 batches commit through agg-pcs-config and
//! open in one batched Akita proof (spec Sections 4, 7.5 and 10; Section 13 PCS item).

mod common;

use agg_piop::manifest::{DigitValue, GroupRole};
use agg_piop::pcs::{suffix_points, verifier_for, TableCommitter};
use agg_piop::virtual_polys::{compose_table, w1_table, CarryValues, MainValues};
use agg_piop::{BaseField, OpeningField};
use agg_witness::check::{check, CheckOptions};
use agg_witness::gen::SplitMix64;
use agg_witness::tables::{BatchTables, Layout};
use common::{mle_i64, point, Batch};
use jolt_field::{ExtField, Ring};

fn on_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("test thread");
}

/// The values the verifier forms from the opened claims equal the multilinear extensions
/// of the composed tables at the opening points.
fn check_virtual_claims(
    batch: &Batch,
    points: &[Vec<OpeningField>; 4],
    values: &[Vec<OpeningField>],
) {
    let manifest = &batch.manifest;
    let tables = batch.tables();
    let main = MainValues::from_evaluations(manifest, &values[0]);
    let composed = |value: DigitValue| compose_table(&value.layout(), &tables.digits(value));
    assert_eq!(main.z, mle_i64(&composed(DigitValue::Z), &points[0]));
    assert_eq!(main.u, mle_i64(&composed(DigitValue::U), &points[0]));
    assert_eq!(main.v, mle_i64(&composed(DigitValue::V), &points[0]));
    let w1 = w1_table(&batch.main.bit_lanes(manifest));
    assert_eq!(main.w1, mle_i64(&w1, &points[0]));
    for (role, kappa) in [
        (
            GroupRole::Level1,
            (0..6).map(DigitValue::Kappa1).collect::<Vec<_>>(),
        ),
        (GroupRole::Level2, (0..6).map(DigitValue::Kappa2).collect()),
        (GroupRole::Level3, vec![DigitValue::Kappa3]),
    ] {
        let carries = CarryValues::from_evaluations(manifest, role, &values[role.index()]);
        for (claimed, value) in carries.kappa.iter().zip(kappa) {
            assert_eq!(*claimed, mle_i64(&composed(value), &points[role.index()]));
        }
        assert_eq!(
            carries.parity.len(),
            if role == GroupRole::Level2 { 6 } else { 0 }
        );
    }
}

fn commit_and_open(n_own: usize, n_shared: usize, seed: u64, suffix: bool) {
    let batch = Batch::build(common::records(seed, n_own, n_shared), seed);
    let manifest = batch.manifest.clone();

    // The committed tables are exactly the ones the integer checker accepts.
    let witness_tables = BatchTables {
        layout: Layout::H,
        geom: batch.main.geometry(),
        columns: batch.main.columns().to_vec(),
    };
    let bridge_tables = batch.bridge.to_witness_tables();
    let report = check(
        &batch.instance,
        &witness_tables,
        Some((&batch.matrices, &bridge_tables)),
        CheckOptions::default(),
    );
    assert!(report.is_ok(), "{:?}", report.failures);

    let committer = TableCommitter::new(manifest.clone()).expect("setup");
    // Round 1 (event 1) before the bridge seed, round 2 (event 4) after it.
    let main = committer.commit_round1(&batch.main).expect("round 1");
    let [level1, level2, level3] = committer.commit_round2(&batch.bridge).expect("round 2");
    let again = committer.commit_round1(&batch.main).expect("round 1 again");
    assert_eq!(
        main.serialized().unwrap(),
        again.serialized().unwrap(),
        "the event-1 payload is deterministic"
    );
    let commitments = [&main, &level1, &level2, &level3].map(|c| c.commitment().clone());
    for group in [&main, &level1, &level2, &level3] {
        assert!(!group.serialized().unwrap().is_empty());
    }

    let mut rng = SplitMix64::new(seed ^ 0xC0FFEE);
    let points = if suffix {
        let r3 = point(&mut rng, manifest.d_main());
        suffix_points(&manifest, &r3).unwrap()
    } else {
        manifest
            .groups()
            .clone()
            .map(|group| point(&mut rng, group.logical_vars))
    };
    let session = [0x3c; 32];
    let (proof, claims) = committer
        .open(
            [main, level1, level2, level3],
            batch.tables(),
            &points,
            &session,
        )
        .expect("batched opening");
    eprintln!(
        "N = {}: {} claims, opening proof {} bytes",
        manifest.records(),
        claims.iter().map(|c| c.values.len()).sum::<usize>(),
        proof.len()
    );
    assert_eq!(claims.iter().map(|c| c.values.len()).sum::<usize>(), 67);
    let values: Vec<Vec<OpeningField>> = claims.iter().map(|c| c.values.clone()).collect();
    check_virtual_claims(&batch, &points, &values);

    // The seed-only verifier and the prover-derived verifier accept.
    let verifier = verifier_for(&manifest).expect("verifier setup");
    verifier
        .verify(&commitments, &claims, &proof, &session)
        .expect("opening verifies");
    committer
        .verifier()
        .unwrap()
        .verify(&commitments, &claims, &proof, &session)
        .expect("prover-derived verifier accepts");

    // A changed claim, a changed session and swapped commitments are rejected.
    for group in 0..4 {
        let mut wrong = claims.clone();
        wrong[group].values[0] += OpeningField::lift_base(BaseField::from_u64(1));
        assert!(verifier
            .verify(&commitments, &wrong, &proof, &session)
            .is_err());
    }
    assert!(verifier
        .verify(&commitments, &claims, &proof, &[0xc3; 32])
        .is_err());
    let mut swapped = commitments.clone();
    swapped.swap(1, 2);
    assert!(verifier
        .verify(&swapped, &claims, &proof, &session)
        .is_err());
}

#[test]
fn tables_of_256_signatures_commit_and_open_in_one_batch() {
    on_large_stack(|| commit_and_open(200, 56, 0x256, true));
}

#[test]
fn padded_batch_commits_and_opens_at_independent_points() {
    // 100 live records pad to N* = 128; padded rows carry the digits of zero.
    on_large_stack(|| commit_and_open(70, 30, 0x100, false));
}
