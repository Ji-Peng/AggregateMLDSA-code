//! Akita commit + open alone: setup, the two-round commit of dummy
//! tables of all six groups, one batched opening, and the verifier, at `N` records
//! (default 1,024). Medians over `AGG_AKITA_REPS` runs (default 5); the opening proof
//! size is printed. Run pinned, never on CPU 2:
//! `taskset -c 0,1,3-15 cargo bench -p agg-pcs-config --bench commit_open -- [N]`.
#![cfg_attr(not(feature = "schedules"), allow(unused))]

#[cfg(feature = "schedules")]
#[path = "../tests/support/dummy.rs"]
mod dummy;

#[cfg(feature = "schedules")]
fn run(records: usize, reps: usize) {
    use agg_pcs_config::catalogs::Group;
    use agg_pcs_config::commit::{PcsProver, PcsVerifier};
    use agg_pcs_config::work_plan::WorkPlan;
    use dummy::{claims, dummy_tables, Rng};
    use std::time::{Duration, Instant};

    fn median(mut v: Vec<Duration>) -> f64 {
        v.sort();
        v[v.len() / 2].as_secs_f64() * 1e3
    }

    let plan = WorkPlan::for_records(records, agg_pcs_config::catalogs::Relation::Hu)
        .expect("work plan");
    let mut rng = Rng(records as u64);
    let tables = Group::ALL.map(|group| dummy_tables(&plan, group, &mut rng));
    let claims = claims(&plan, &tables, &mut rng);
    let session = [0x5a; 32];
    let (mut setup, mut r1, mut r2, mut open, mut vsetup, mut verify) =
        (vec![], vec![], vec![], vec![], vec![], vec![]);
    let mut proof_len = 0;
    let mut commit_bytes = 0;
    for _ in 0..reps {
        let t = Instant::now();
        let prover = PcsProver::new(plan.clone()).expect("prover setup");
        setup.push(t.elapsed());
        let commit = |group: Group| prover.commit_group(group, &tables[group.index()]).expect("commit");
        let t = Instant::now();
        let (main16, main) = (commit(Group::Main16), commit(Group::Main));
        r1.push(t.elapsed());
        let t = Instant::now();
        let [l1, l2, l3, mult] = [Group::Level1, Group::Level2, Group::Level3, Group::Mult].map(commit);
        r2.push(t.elapsed());
        let groups = [main16, main, l1, l2, l3, mult];
        commit_bytes = groups
            .iter()
            .map(|g| g.serialized().expect("encoding").len())
            .sum();
        let commitments: [_; 6] = std::array::from_fn(|i| groups[i].commitment().clone());
        let t = Instant::now();
        let proof = prover.open(groups, &claims, &session).expect("opening");
        open.push(t.elapsed());
        proof_len = proof.len();
        let t = Instant::now();
        let verifier = PcsVerifier::new(plan.clone()).expect("verifier setup");
        vsetup.push(t.elapsed());
        let t = Instant::now();
        verifier
            .verify(&commitments, &claims, &proof, &session)
            .expect("opening verifies");
        verify.push(t.elapsed());
    }
    println!(
        "akita commit_open: N = {records}, reps {reps}, median ms: prover setup {:.1}, round-1 commit {:.1}, \
         round-2 commit {:.1}, open {:.1}, commit+open {:.1}; verifier setup {:.1}, verify {:.1}; \
         commitments {commit_bytes} B, opening proof {proof_len} B",
        median(setup.clone()),
        median(r1.clone()),
        median(r2.clone()),
        median(open.clone()),
        median(r1) + median(r2) + median(open),
        median(vsetup),
        median(verify),
    );
}

fn main() {
    #[cfg(feature = "schedules")]
    {
        let records = std::env::args()
            .skip(1)
            .find_map(|a| a.parse::<usize>().ok())
            .unwrap_or(1024);
        let reps = std::env::var("AGG_AKITA_REPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5);
        std::thread::Builder::new()
            .stack_size(256 << 20)
            .spawn(move || run(records, reps))
            .expect("spawn")
            .join()
            .expect("bench thread");
    }
}
