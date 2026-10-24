//! Two-round commit of a dummy batch of every group and one batched Akita
//! opening of all four groups (spec Section 13, PCS config item), at
//! `N = 2^8` and `N = 2^10` signatures.
#![cfg(feature = "schedules")]

use agg_pcs_config::catalogs::Group;
use agg_pcs_config::commit::{evaluate_signed_table, PcsProver, PcsVerifier};
use agg_pcs_config::preset::{BaseField, OpeningField};
use agg_pcs_config::work_plan::WorkPlan;
use jolt_field::{ExtField, Ring};

#[path = "support/dummy.rs"]
mod dummy;
use dummy::{claims, dummy_tables, Rng};

fn commit_and_open(records: usize) {
    let plan = WorkPlan::for_records(records).expect("work plan");
    let mut rng = Rng(records as u64);
    let tables = Group::ALL.map(|group| dummy_tables(&plan, group, &mut rng));
    let prover = PcsProver::new(plan.clone()).expect("prover setup");

    // Round 1 before the bridge seed, round 2 after it.
    let main = prover.commit_round1(&tables[0]).expect("round 1");
    let [level1, level2, level3] = prover
        .commit_round2(&tables[1], &tables[2], &tables[3])
        .expect("round 2");
    let commitments = [
        main.commitment().clone(),
        level1.commitment().clone(),
        level2.commitment().clone(),
        level3.commitment().clone(),
    ];
    for group in [&main, &level1, &level2, &level3] {
        assert!(!group.serialized().expect("encoding").is_empty());
    }

    let claims = claims(&plan, &tables, &mut rng);
    let session = [0x5a; 32];
    let proof = prover
        .open([main, level1, level2, level3], &claims, &session)
        .expect("batched opening");
    eprintln!("N = {records}: opening proof {} bytes", proof.len());

    // The seed-only verifier needs no prover state.
    let verifier = PcsVerifier::new(plan).expect("verifier setup");
    verifier
        .verify(&commitments, &claims, &proof, &session)
        .expect("opening verifies");
    prover
        .verifier()
        .expect("prover-derived verifier")
        .verify(&commitments, &claims, &proof, &session)
        .expect("prover-derived verifier accepts");

    // A changed claim, a changed session and a swapped commitment are rejected.
    let mut wrong = claims.clone();
    wrong[2].values[5] += OpeningField::lift_base(BaseField::from_u64(1));
    assert!(verifier
        .verify(&commitments, &wrong, &proof, &session)
        .is_err());
    assert!(verifier
        .verify(&commitments, &claims, &proof, &[0xa5; 32])
        .is_err());
    let mut swapped = commitments.clone();
    swapped.swap(1, 2);
    assert!(verifier
        .verify(&swapped, &claims, &proof, &session)
        .is_err());
}

fn on_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("test thread");
}

#[test]
fn two_round_commit_and_batched_opening_at_256() {
    on_large_stack(|| commit_and_open(1 << 8));
}

#[test]
fn two_round_commit_and_batched_opening_at_1024_with_padding() {
    // 1000 live records pad to N* = 1024; padded rows carry the digits of zero.
    on_large_stack(|| commit_and_open(1000));
}

#[test]
fn malformed_tables_are_rejected_before_akita() {
    let plan = WorkPlan::for_records(1 << 8).expect("work plan");
    let prover = PcsProver::new(plan.clone()).expect("prover setup");
    let cells = plan.logical_cells(Group::Level3);
    assert!(prover
        .commit_group(Group::Level3, &[vec![0; cells]])
        .is_err());
    assert!(prover
        .commit_group(Group::Level3, &[vec![0; cells], vec![0; cells - 1]])
        .is_err());
    let mut out_of_range = vec![vec![0_i8; cells], vec![0_i8; cells]];
    out_of_range[1][3] = 8;
    assert!(prover.commit_group(Group::Level3, &out_of_range).is_err());
}

#[test]
fn evaluation_matches_table_at_boolean_points() {
    let table: Vec<i8> = (0..16).map(|i| (i % 16) as i8 - 8).collect();
    for index in 0..16_usize {
        let point: Vec<OpeningField> = (0..4)
            .map(|bit| OpeningField::lift_base(BaseField::from_u64(((index >> bit) & 1) as u64)))
            .collect();
        assert_eq!(
            evaluate_signed_table(&table, &point),
            OpeningField::lift_base(BaseField::from_i64(i64::from(table[index])))
        );
    }
}
