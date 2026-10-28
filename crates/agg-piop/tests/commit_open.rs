//! Round-1 and round-2 tables of real ML-DSA-44 batches (two main groups of widths 16 and
//! 4, three width-16 bridge groups and the width-4 multiplicity group) commit through
//! agg-pcs-config and open in one batched Akita proof.

mod common;

use agg_piop::manifest::{Column, DigitValue, GroupRole, CARRY16_DIGITS};
use agg_piop::pcs::{group_points, verifier_for, TableCommitter};
use agg_piop::source_tables::Cells;
use agg_piop::virtual_polys::{compose_table, w1_table, CarryValues, MainValues};
use agg_piop::{BaseField, OpeningField};
use agg_witness::gen::SplitMix64;
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
    points: &[Vec<OpeningField>; 6],
    values: &[Vec<OpeningField>],
) {
    let manifest = &batch.manifest;
    let tables = batch.tables();
    assert_eq!(points[0], points[1], "both main groups at one point");
    let main = MainValues::from_claims(manifest, |column: Column| {
        let (role, position) = manifest.locate(column);
        values[role.index()][position]
    });
    let composed = |value: DigitValue| compose_table(&value.layout(), &tables.digits(value));
    assert_eq!(main.z, mle_i64(&composed(manifest.z_value()), &points[0]));
    assert_eq!(main.u, mle_i64(&composed(DigitValue::U), &points[0]));
    assert_eq!(main.v, mle_i64(&composed(DigitValue::V), &points[0]));
    let w1 = w1_table(&batch.main.bit_lanes(manifest));
    assert_eq!(main.w1, mle_i64(&w1, &points[0]));
    for (role, level, lanes) in [(GroupRole::Level1, 1, 6), (GroupRole::Level2, 2, 6), (GroupRole::Level3, 3, 1)] {
        let carries = CarryValues::from_evaluations(manifest, role, &values[role.index()]);
        for (t, claimed) in carries.kappa.iter().enumerate().take(lanes) {
            let kappa = compose_table(&agg_witness::digits::ValueForm::Radix(CARRY16_DIGITS), &[tables.column(Column::carry(level, t as u8))]);
            assert_eq!(*claimed, mle_i64(&kappa, &points[role.index()]));
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

    let committer = TableCommitter::new(manifest.clone()).expect("setup");
    // Round 1 (event 1) before the bridge seed, round 2 (event 4) after it.
    let [main16, main] = committer.commit_round1(&batch.main).expect("round 1");
    let [level1, level2, level3, mult] = committer.commit_round2(&batch.bridge).expect("round 2");
    let [_, again] = committer.commit_round1(&batch.main).expect("round 1 again");
    assert_eq!(
        main.serialized().unwrap(),
        again.serialized().unwrap(),
        "the event-1 payload is deterministic"
    );
    let groups = [&main16, &main, &level1, &level2, &level3, &mult];
    let commitments = groups.map(|c| c.commitment().clone());
    for group in groups {
        assert!(!group.serialized().unwrap().is_empty());
    }
    // The multiplicity digits are width-4 cells.
    assert!(matches!(batch.tables().group(GroupRole::Mult)[0], Cells::Narrow(_)));

    let mut rng = SplitMix64::new(seed ^ 0xC0FFEE);
    let points = if suffix {
        let r3 = point(&mut rng, manifest.d_main());
        let pi_m = point(&mut rng, 16);
        group_points(&manifest, &r3, &pi_m).unwrap()
    } else {
        let mut points = manifest
            .groups()
            .clone()
            .map(|group| point(&mut rng, group.logical_vars));
        points[1] = points[0].clone();
        points
    };
    let session = [0x3c; 32];
    let (proof, claims) = committer
        .open(
            [main16, main, level1, level2, level3, mult],
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
    assert_eq!(claims.iter().map(|c| c.values.len()).sum::<usize>(), 38);
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
    for group in 0..6 {
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
