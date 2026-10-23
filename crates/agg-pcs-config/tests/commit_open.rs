//! Two-round commit of a dummy batch of every group and one batched Akita
//! opening of all four groups (spec Section 13, PCS config item), at
//! `N = 2^8` and `N = 2^10` signatures.
#![cfg(feature = "schedules")]

use agg_pcs_config::catalogs::Group;
use agg_pcs_config::commit::{evaluate_signed_table, GroupClaim, PcsProver, PcsVerifier};
use agg_pcs_config::preset::{BaseField, OpeningField};
use agg_pcs_config::work_plan::WorkPlan;
use jolt_field::{ExtField, Ring};

/// SplitMix64, a deterministic test source.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A value of the group's honest alphabet: Booleans for the bit and
    /// parity columns, balanced digits otherwise (every one inside `[-8, 7]`).
    fn digit(&mut self, base: i64) -> i8 {
        let value = (self.next() % base as u64) as i64 - base / 2;
        i8::try_from(value).expect("small digit")
    }

    fn field(&mut self) -> OpeningField {
        OpeningField::from_base_slice(&[
            BaseField::from_u64(self.next()),
            BaseField::from_u64(self.next()),
        ])
    }
}

/// Digit bases of each group's polynomials, spec Section 4.1 (`0` marks a
/// Boolean column).
fn bases(group: Group) -> Vec<i64> {
    match group {
        Group::Main => [
            16, 16, 16, 16, 4, 16, 16, 16, 16, 8, 16, 16, 16, 16, 16, 16, 8, 0, 0, 0, 0, 0, 0,
        ]
        .to_vec(),
        Group::Level1 => [16, 16, 4].repeat(6),
        Group::Level2 => {
            let mut bases = vec![0; 6];
            bases.extend([16, 16, 4].repeat(6));
            bases
        }
        Group::Level3 => vec![16, 16],
    }
}

/// Dummy tables of one group; the rows of padded signatures (`i >= live`) are
/// the canonical digits of zero, `-b/2`, and zero bits (spec Section 4.2).
fn dummy_tables(plan: &WorkPlan, group: Group, rng: &mut Rng) -> Vec<Vec<i8>> {
    let cells = plan.logical_cells(group);
    let bases = bases(group);
    assert_eq!(bases.len(), group.num_polynomials());
    bases
        .iter()
        .map(|&base| {
            (0..cells)
                .map(|cell| {
                    // Main-domain index order is (j, rho or s, i): the
                    // signature index is the top d_sig bits.
                    let padded = group == Group::Main && (cell >> 10) >= plan.records;
                    match (base, padded) {
                        (0, true) => 0,
                        (0, false) => (rng.next() & 1) as i8,
                        (_, true) => i8::try_from(-base / 2).unwrap(),
                        (_, false) => rng.digit(base),
                    }
                })
                .collect()
        })
        .collect()
}

fn claims(plan: &WorkPlan, tables: &[Vec<Vec<i8>>; 4], rng: &mut Rng) -> [GroupClaim; 4] {
    Group::ALL.map(|group| {
        let point: Vec<OpeningField> = (0..plan.logical_vars(group)).map(|_| rng.field()).collect();
        let values = tables[group.index()]
            .iter()
            .map(|table| evaluate_signed_table(table, &point))
            .collect();
        GroupClaim { point, values }
    })
}

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
