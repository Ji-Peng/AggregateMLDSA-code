//! Akita's transcript proof-of-work plan for every grouped catalog row
//! (`docs/soundness.md` Sections 3.7 and 4.1): the pinned Akita revision derives the only
//! accepted plan from the row's schedule and the opening layout and binds its digest into
//! the instance descriptor, on both the prover and the verifier side. This test pins the
//! plan of each row (sites, difficulties, digest) as a regression guard against a catalog or
//! Akita change.
#![cfg(feature = "schedules")]

use agg_pcs_config::catalogs::{Group, CATALOG_D_SIG};
use agg_pcs_config::preset::AggFp64SignedWidth4;
use agg_pcs_config::work_plan::WorkPlan;
use akita_config::derive_transcript_grinding_plan;
use akita_params::{GrindingPlan, GrindingQueryKind, OpeningClaimsLayout};

/// The plan of the grouped row of `N* = 2^d_sig` for the opening of its six groups.
fn row_plan(d_sig: usize) -> GrindingPlan {
    let plan = WorkPlan::for_records(1 << d_sig).expect("work plan");
    let row = AggFp64SignedWidth4::catalog()
        .expect("catalog")
        .resolve_key(&plan.schedule_key())
        .expect("grouped row");
    let precommitted: Vec<_> = Group::ALL
        .iter()
        .filter(|g| !g.is_final())
        .map(|&g| plan.profile(g).group)
        .collect();
    let last = Group::ALL
        .iter()
        .find(|g| g.is_final())
        .expect("final group");
    let layout = OpeningClaimsLayout::from_root_groups(&precommitted, plan.profile(*last).group)
        .expect("opening layout");
    derive_transcript_grinding_plan::<AggFp64SignedWidth4>(row.schedule(), &layout)
        .expect("grinding plan")
}

/// (protected sites, sum g, max g, sum 2^g) of the proof-of-work runs with work, and the
/// nonce bits of every run (the fold-response nonces included).
fn summary(plan: &GrindingPlan) -> (u64, u64, u8, u64, u64) {
    let pow = plan
        .runs()
        .iter()
        .filter(|r| matches!(r.kind(), GrindingQueryKind::ProofOfWork) && r.grind_bits() > 0);
    let (n, sum, max, work) = pow.fold((0, 0, 0, 0), |(n, sum, max, work), r| {
        let m = r.multiplicity();
        (
            n + m,
            sum + m * u64::from(r.grind_bits()),
            max.max(r.grind_bits()),
            work + m * (1 << r.grind_bits()),
        )
    });
    let bits = plan
        .runs()
        .iter()
        .map(|r| r.multiplicity() * u64::from(r.nonce_bits()))
        .sum();
    (n, sum, max, work, bits)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn akita_plans_of_the_grouped_rows_are_pinned() {
    // docs/soundness.md Section 3.7: (d_sig, PoW sites, sum g, max g, sum 2^g, nonce bits).
    let table: [(usize, u64, u64, u8, u64, u64); 13] = [
        (3, 401, 984, 9, 3_888, 3_863),
        (4, 401, 984, 9, 3_888, 3_863),
        (5, 401, 984, 9, 3_888, 3_863),
        (6, 401, 985, 10, 4_400, 3_864),
        (7, 401, 985, 10, 4_400, 3_864),
        (8, 401, 985, 10, 4_400, 3_864),
        (9, 406, 980, 10, 5_120, 3_894),
        (10, 388, 967, 12, 7_692, 3_755),
        (11, 430, 1_053, 12, 7_868, 4_135),
        (12, 452, 1_121, 12, 8_304, 4_369),
        (13, 455, 1_129, 12, 8_832, 4_398),
        (14, 457, 1_132, 12, 8_076, 4_415),
        (15, 459, 1_137, 12, 9_872, 4_434),
    ];
    assert_eq!(CATALOG_D_SIG, (3..=15).collect::<Vec<_>>().as_slice());
    let mut digests = Vec::new();
    for (d_sig, sites, sum, max, work, bits) in table {
        let plan = row_plan(d_sig);
        let digest = plan.digest().expect("digest");
        eprintln!(
            "d_sig {d_sig:>2}: {:?} digest {}",
            summary(&plan),
            hex(&digest)
        );
        let got = summary(&plan);
        assert_eq!(got, (sites, sum, max, work, bits), "d_sig {d_sig}");
        // Every target within Akita's transition cap and 7 bits of slack.
        assert!(plan.runs().iter().all(|r| r.grind_bits() <= 25
            && (r.grind_bits() == 0 || r.nonce_bits() == r.grind_bits() + 7)));
        digests.push(hex(&digest));
    }
    // The first plan-digest bytes reported in docs/soundness.md Section 4.1.
    assert!(digests[0].starts_with("225708cb"), "{}", digests[0]);
    assert!(digests[12].starts_with("4f5a6026"), "{}", digests[12]);
    assert_eq!(
        PINNED,
        digests
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .as_slice()
    );
}

/// The plan digests of the 13 grouped rows (`d_sig` 3 to 15) at Akita `e023b6a`.
const PINNED: &[&str] = &[
    "225708cbc2955271cbe9c9357de749875aeda1900a94bc4c7593a9f961bfcb62",
    "7c4431eaa095aa5315c240f30f983907d524361347e09fd39f3b7b4858171981",
    "132e4ad7819008d25e5b30c86847b5eef066005e72afe5a8297916dcb6a2e638",
    "e85c8ab87d42fe87951ce90da10ae83aa7dea5c4508608c22894cb4277b663c7",
    "f2a0bdaeed29a12b04d39a0455c994e1e859476d21c364ccd2462e59ee621d9e",
    "3ac4bfba35af9f76e29b5effdf85416aa46706cb437cfc4e9a024935cb6f015f",
    "d3977f7c94b23825ddd5ea088c1ec81cf13089128514d5e213808144b53b737e",
    "8611ed83429b3773e36f0abc4d471d75d06ec20d2a1eed59c02a98b2cbfbd379",
    "b0eff2da80859051953f3de5e2d49f75962c5c2504de008ccdd9031abbc0cfdc",
    "3384fcb6875813fa60058ed4625d2b4cef2529d1aeda0befb114495554405ad1",
    "79772eef3ee392501ab8f6055d00e627008f3b0182f5c138d6a1b7d78f2d23bd",
    "5bca18a0a4efeee42d646b6b72c94baf1542843b75f6b805cf1e7775e15f6a31",
    "4f5a6026543ff6e0b2a80c74e633110d1f0bfd6fd8606413c6a4300284cd6eb2",
];
