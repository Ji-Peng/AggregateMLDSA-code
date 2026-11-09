//! Two-round commit of a dummy batch of every group (two families, width 4 and
//! width 16) and one batched Akita opening of all seven groups, at `N = 2^8` and
//! `N = 2^10` signatures.
#![cfg(feature = "schedules")]

use agg_pcs_config::catalogs::{Group, Relation};
use agg_pcs_config::commit::{evaluate_signed_table, PcsProver, PcsVerifier};
use agg_pcs_config::preset::{BaseField, OpeningField};
use agg_pcs_config::work_plan::WorkPlan;
use jolt_field::{ExtField, Ring};

use crate::dummy::{claims, dummy_tables, Rng};

fn commit_and_open(records: usize, relation: Relation) {
    let plan = WorkPlan::for_records(records, relation).expect("work plan");
    let mut rng = Rng(records as u64);
    let tables = Group::ALL.map(|group| dummy_tables(&plan, group, &mut rng));
    let prover = PcsProver::new(plan.clone()).expect("prover setup");

    // Round 1 before the bridge seed, round 2 after it, each group under its family.
    let groups = Group::ALL.map(|group| {
        prover
            .commit_group(group, &tables[group.index()])
            .unwrap_or_else(|e| panic!("{group:?}: {e}"))
    });
    let commitments = std::array::from_fn(|i| groups[i].commitment().clone());
    for group in &groups {
        assert!(!group.serialized().expect("encoding").is_empty());
    }

    let claims = claims(&plan, &tables, &mut rng);
    let session = [0x5a; 32];
    let proof = prover
        .open(groups, &claims, &session)
        .expect("batched opening");
    eprintln!("N = {records}, {relation}: opening proof {} bytes", proof.len());

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
fn h_exact_commit_and_batched_opening_at_256_and_1024() {
    // G_main4 at 9 polynomials (H-exact), the p16x grouped rows.
    on_large_stack(|| {
        commit_and_open(1 << 8, Relation::HExact);
        commit_and_open(1000, Relation::HExact);
    });
}

#[test]
fn h_fips_commit_and_batched_opening_at_256_and_1024() {
    // G_main4 at 12 polynomials and G_2 at 13 (H-FIPS), the p16f grouped rows; 1000 live
    // records pad to N* = 1024, and the padded rows carry the digits of zero.
    on_large_stack(|| {
        commit_and_open(1 << 8, Relation::HFips);
        commit_and_open(1000, Relation::HFips);
    });
}

#[test]
fn malformed_tables_are_rejected_before_akita() {
    let plan = WorkPlan::for_records(1 << 8, Relation::HFips).expect("work plan");
    let prover = PcsProver::new(plan.clone()).expect("prover setup");
    let cells = plan.logical_cells(Group::Level3);
    let count = plan.num_polynomials(Group::Level3);
    assert!(prover
        .commit_group(Group::Level3, &vec![vec![0_i32; cells]; count + 1])
        .is_err());
    assert!(prover
        .commit_group(Group::Level3, &vec![vec![0_i32; cells - 1]; count])
        .is_err());
    // Width 16 admits [-2^15, 2^15 - 1]; width 4 admits [-8, 7].
    let mut out_of_range = vec![vec![0_i32; cells]; count];
    out_of_range[0][3] = 1 << 15;
    assert!(prover.commit_group(Group::Level3, &out_of_range).is_err());
    out_of_range[0][3] = (1 << 15) - 1;
    assert!(prover.commit_group(Group::Level3, &out_of_range).is_ok());
    let main_cells = plan.logical_cells(Group::Main);
    let mut narrow = vec![vec![0_i8; main_cells]; 12];
    narrow[2][5] = 8;
    assert!(prover.commit_group(Group::Main, &narrow).is_err());
    // An H-FIPS plan refuses a 9-polynomial G_main4 (H-exact's count).
    assert!(prover
        .commit_group(Group::Main, &vec![vec![0_i8; main_cells]; 9])
        .is_err());
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
