//! Akita's transcript proof-of-work plan for every grouped catalog row
//! (`docs/soundness.md`): the pinned Akita revision derives the only
//! accepted plan from the row's schedule and the opening layout and binds its digest into
//! the instance descriptor, on both the prover and the verifier side. This test pins the
//! plan of each row (sites, difficulties, digest) as a regression guard against a catalog or
//! Akita change.
#![cfg(feature = "schedules")]

use agg_pcs_config::catalogs::{Group, Relation, Set, CATALOG_D_SIG};
use agg_pcs_config::preset::AggFp64SignedWidth4;
use agg_pcs_config::work_plan::WorkPlan;
use akita_config::derive_transcript_grinding_plan;
use akita_params::{GrindingPlan, GrindingQueryKind, OpeningClaimsLayout};

/// The plan of the grouped row of `set` at `N* = 2^d_sig` for the opening of its seven groups.
fn row_plan_for_set(set: Set, d_sig: usize, relation: Relation) -> GrindingPlan {
    let plan = WorkPlan::for_set(set, 1 << d_sig, relation).expect("work plan");
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

/// The plan of the grouped row of `N* = 2^d_sig` for the opening of its seven groups.
fn row_plan(d_sig: usize, relation: Relation) -> GrindingPlan {
    row_plan_for_set(Set::S44, d_sig, relation)
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
    // (d_sig, PoW sites, sum g, max g, sum 2^g, nonce bits).
    let table: [(usize, u64, u64, u8, u64, u64); 13] = [
        (3, 401, 984, 9, 3_888, 3_863),
        (4, 401, 984, 9, 3_888, 3_863),
        (5, 401, 984, 9, 3_888, 3_863),
        (6, 401, 985, 10, 4_400, 3_864),
        (7, 406, 980, 10, 5_120, 3_894),
        (8, 406, 980, 10, 5_120, 3_894),
        (9, 388, 966, 10, 5_132, 3_754),
        (10, 388, 966, 12, 7_436, 3_754),
        (11, 452, 1_121, 12, 8_304, 4_369),
        (12, 452, 1_120, 12, 8_048, 4_368),
        (13, 457, 1_133, 12, 8_332, 4_416),
        (14, 459, 1_137, 12, 9_872, 4_434),
        (15, 461, 1_142, 12, 9_884, 4_453),
    ];
    assert_eq!(CATALOG_D_SIG, (3..=15).collect::<Vec<_>>().as_slice());
    let mut digests = Vec::new();
    let (mut got, mut want) = (Vec::new(), Vec::new());
    for (d_sig, sites, sum, max, work, bits) in table {
        let plan = row_plan(d_sig, Relation::Hu);
        let digest = plan.digest().expect("digest");
        eprintln!(
            "d_sig {d_sig:>2}: {:?} digest {}",
            summary(&plan),
            hex(&digest)
        );
        got.push((d_sig, summary(&plan)));
        want.push((d_sig, (sites, sum, max, work, bits)));
        // Every target within Akita's transition cap and 7 bits of slack.
        assert!(plan.runs().iter().all(|r| r.grind_bits() <= 25
            && (r.grind_bits() == 0 || r.nonce_bits() == r.grind_bits() + 7)));
        digests.push(hex(&digest));
    }
    assert_eq!(got, want);
    // The first plan-digest bytes reported in `docs/soundness.md`.
    assert!(digests[0].starts_with("88f7a3a6"), "{}", digests[0]);
    assert!(digests[12].starts_with("a0119a5e"), "{}", digests[12]);
    assert_eq!(
        PINNED,
        digests
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .as_slice()
    );
}

/// The plan digests of the 13 grouped rows (`d_sig` 3 to 15) at the pinned Akita revision.
const PINNED: &[&str] = &[
    "88f7a3a6649e8ab911e3c4bfb8e975fce6bc6e60697c6065ac76d792e22e18ab",
    "88f7a3a6649e8ab911e3c4bfb8e975fce6bc6e60697c6065ac76d792e22e18ab",
    "6e66760deece12609bc8b110bbb6c334c3a3611d7b224498c7f5d048a135bb9b",
    "d08698a4950306f67caa9ae4ba72bce5591756ba462b1a609de810e08850914b",
    "5160b5d3a19bcfda3dc8c7550f43cef4811afa8b0707ef74712cd402f424bf0d",
    "88a686c77c3d03f1074a09c9e5558ed1ee61ae4d42e2c003bae9a62da0987191",
    "8a5fc03e2bca9f1a395ef4cf763238eda7b0892c97dbef1d942d95f656f12e13",
    "e3fccec9d89950349f4eb12bc8d00b40735ffca161771df4bd6d26f0ce49dcb3",
    "47a0be13018792e2f8bbe49857bcb815bc9261fc4940f1c8e20c8dd4135751a8",
    "e59b279cb97130389bcd2a221d9ee91f1141af0ce1788ab14fd7c7cbf6d0f11b",
    "824b83d11ecf4d6060ca6ef9e317ffdccef2139941f9b52ccd34ea263f21ea2f",
    "fb524f367c1c30b8b5b7474313a784961fb03a830aa20e5714cb4fb5faf24e95",
    "a0119a5ec75e86453396e205c70d850beaa56f5ac19a8e4529b927085eabd994",
];

/// The Akita plans of the 13 H-exact grouped rows (`mldsa-44-p16x-batch-<N*>-mult-final`,
/// `G_main4` at 9 polynomials): the same checks as for H-u, the
/// pinned digests below, and every digest distinct from the H-u row's.
#[test]
fn akita_plans_of_the_h_exact_rows_are_pinned() {
    assert_eq!(
        agg_pcs_config::catalogs::CATALOG_D_SIG_H_EXACT,
        (3..=15).collect::<Vec<_>>().as_slice()
    );
    let mut digests = Vec::new();
    let mut summaries = Vec::new();
    for d_sig in 3..=15 {
        let plan = row_plan(d_sig, Relation::HExact);
        let digest = hex(&plan.digest().expect("digest"));
        let got = summary(&plan);
        eprintln!("H-exact d_sig {d_sig:>2}: {got:?} digest {digest}");
        assert!(plan.runs().iter().all(|r| r.grind_bits() <= 25
            && (r.grind_bits() == 0 || r.nonce_bits() == r.grind_bits() + 7)));
        assert_ne!(digest, PINNED[d_sig - 3], "H-exact and H-u rows differ");
        summaries.push(got);
        digests.push(digest);
    }
    assert_eq!(summaries, H_EXACT_TABLE);
    assert_eq!(
        PINNED_H_EXACT,
        digests.iter().map(String::as_str).collect::<Vec<_>>().as_slice()
    );
}

/// The Akita plans of the 13 H-FIPS grouped rows (`mldsa-44-p16f-batch-<N*>-mult-final`,
/// `G_main4` at 12 polynomials and `G_2` at 13), printed for the
/// record: every row stays within Akita's transition cap with 7 bits of nonce slack, and
/// every digest differs from the H-u and H-exact rows' of the same size.
#[test]
fn akita_plans_of_the_h_fips_rows_are_well_formed() {
    assert_eq!(
        agg_pcs_config::catalogs::CATALOG_D_SIG_H_FIPS,
        (3..=15).collect::<Vec<_>>().as_slice()
    );
    let mut digests = Vec::new();
    for d_sig in 3..=15 {
        let plan = row_plan(d_sig, Relation::HFips);
        let digest = hex(&plan.digest().expect("digest"));
        eprintln!("H-FIPS d_sig {d_sig:>2}: {:?} digest {digest}", summary(&plan));
        assert!(plan.runs().iter().all(|r| r.grind_bits() <= 25
            && (r.grind_bits() == 0 || r.nonce_bits() == r.grind_bits() + 7)));
        assert_ne!(digest, PINNED[d_sig - 3], "H-FIPS and H-u rows differ");
        assert_ne!(digest, PINNED_H_EXACT[d_sig - 3], "H-FIPS and H-exact rows differ");
        digests.push(digest);
    }
    assert_eq!(PINNED_H_FIPS, digests.iter().map(String::as_str).collect::<Vec<_>>().as_slice());
}

/// The plan digests of the 13 H-FIPS grouped rows at the pinned Akita revision.
const PINNED_H_FIPS: &[&str] = &[
    "1646240bee27d2e09219fb5187fa7207e8da8b5e27c80040ce9fb6e2be75ac99",
    "c3993eaa18ab421f040934ef45d20729ae79c2f8f02fb7b84c863b5e53035347",
    "c3993eaa18ab421f040934ef45d20729ae79c2f8f02fb7b84c863b5e53035347",
    "9e48036b863f41dfcc9898d9f9d4c442a631996e1ee69fb0339d10d6d4977f70",
    "fea2883525136da9109d02f8fb2e980eaf76fc6175b3ebb3122707454f1fa902",
    "81aad52b68d933bd08874a0eb5e3d133f05cb157a16d7af31ed0abf9bc1c63c0",
    "fc3a03b728c74ff2fa8a87ed55c95522d2bbbf1723cde7539b630a61560bfc3e",
    "2a757b0b89b20d23f6b7d2460583c66ec89e5d11450e1541a0a599c727a483a1",
    "ddd3a8004f74650e916af4002f88bb7d6b31a8dd5c86e83ac779206eed13dbed",
    "985808fbe87ebbb5947bd3ccf45e2617e650bd62cd3f63473dbbbfbd3e8875a4",
    "973e390dfc0f54b4d99ff69fca16a611dfae865e7ff924debceca3171e4bdd54",
    "834f81c85835352060e585594827dcf588a26d327870e4a11b930895bcedffce",
    "81ac9d6b8f79ac01674517182d89bd4a44a08670e9439181a0570f098514434e",
];

/// (PoW sites, sum g, max g, sum 2^g, nonce bits) of the H-exact rows, `d_sig` 3 to 15: the
/// H-u values at every size in the seven-group layout.
const H_EXACT_TABLE: [(u64, u64, u8, u64, u64); 13] = [
    (401, 984, 9, 3_888, 3_863),
    (401, 984, 9, 3_888, 3_863),
    (401, 984, 9, 3_888, 3_863),
    (401, 985, 10, 4_400, 3_864),
    (406, 980, 10, 5_120, 3_894),
    (406, 980, 10, 5_120, 3_894),
    (388, 965, 10, 4_620, 3_753),
    (430, 1_053, 12, 7_868, 4_135),
    (452, 1_121, 12, 8_304, 4_369),
    (452, 1_120, 12, 8_048, 4_368),
    (457, 1_133, 12, 8_332, 4_416),
    (459, 1_137, 12, 9_872, 4_434),
    (461, 1_142, 12, 9_884, 4_453),
];

/// The plan digests of the 13 H-exact grouped rows at the pinned Akita revision. The rows of
/// `d_sig` 3 and 4 have equal plan digests, which is benign: Akita's
/// `GrindingPlan::canonical_bytes` covers only the policy, the challenge order and the runs
/// (site, target), not the layout, and those two rows draw identical site lists (same fold
/// families and targets). The layout (13 against 14 variables) is bound separately through
/// the schedule key, the row name and Akita's instance descriptor, and all 52 width-4
/// catalog schedules hash differently.
const PINNED_H_EXACT: &[&str] = &[
    "53813c7ddb89d832e05f1f40240acdd31f147c805b06b6b70af4b75e46708bf4",
    "53813c7ddb89d832e05f1f40240acdd31f147c805b06b6b70af4b75e46708bf4",
    "7a1851db570437fe4a273c6ddc6dc794cbbdbb663d8cc28445f406707566d109",
    "1e23df99df8a7521f61ac77208ce5b5e996e575dabbb2eec737b772c3ee6c013",
    "c476ae0f93663e5166b942b2f0e436226fbc3c325167a2c37d825365eac5b112",
    "5ea62860a304f306514014290250ae47845e43d1a284ce00004449a58d8126a2",
    "b0f906130de7c92388ffaeb5425ef106731d8bcaf0683f769e877e17c31b4afa",
    "f38a1e6c2804d83d4366eeb38ce36a49affe3fac0b87c069be54445550c9b2b2",
    "a7e9e86f252fdaa61e7e3ecad5fb439f2bc3d77a731f2fedcefe521ea20bfc11",
    "b2a92ba31da4538572346ca7aa2811b94990eab7e3f7745d4d911061112ec5cf",
    "a812cae0ca65d3d8054184068290bc6495e09db717b15352065bddb6a7284255",
    "d1720056835235f087f1a20fa4921cba15aecd01c9232b075299ea98e79e660b",
    "a9a5c092d719dc1df17d47e76910fbe8d46db6a6f8ffc0db1c994953e2affe4c",
];

/// The Akita plans of the 13 ML-DSA-65 grouped rows (`mldsa-65-p16-batch-<N*>-mult-final`,
/// relation H-u): verified well-formed with 7 bits of nonce slack and pinned digests.
#[test]
fn akita_plans_of_the_mldsa_65_rows_are_pinned() {
    let mut digests = Vec::new();
    let mut summaries = Vec::new();
    for (d_sig, _sites, _sum, _max, _work, _bits) in S65_TABLE {
        let plan = row_plan_for_set(Set::S65, d_sig, Relation::Hu);
        let digest = hex(&plan.digest().expect("digest"));
        let got = summary(&plan);
        eprintln!("ML-DSA-65 d_sig {d_sig:>2}: {got:?} digest {digest}");
        assert!(plan.runs().iter().all(|r| r.grind_bits() <= 25
            && (r.grind_bits() == 0 || r.nonce_bits() == r.grind_bits() + 7)));
        assert_ne!(digest, PINNED[d_sig - 3], "ML-DSA-65 and ML-DSA-44 rows differ");
        summaries.push(got);
        digests.push(digest);
    }
    let expected_summaries: Vec<_> = S65_TABLE.iter().map(|&(_, a, b, c, d, e)| (a, b, c, d, e)).collect();
    assert_eq!(summaries, expected_summaries);
    assert_eq!(
        PINNED_S65,
        digests.iter().map(String::as_str).collect::<Vec<_>>().as_slice()
    );
}

/// The Akita plans of the 13 ML-DSA-87 grouped rows (`mldsa-87-p16-batch-<N*>-mult-final`,
/// relation H-u): verified well-formed with 7 bits of nonce slack and pinned digests.
#[test]
fn akita_plans_of_the_mldsa_87_rows_are_pinned() {
    let mut digests = Vec::new();
    let mut summaries = Vec::new();
    for (d_sig, _sites, _sum, _max, _work, _bits) in S87_TABLE {
        let plan = row_plan_for_set(Set::S87, d_sig, Relation::Hu);
        let digest = hex(&plan.digest().expect("digest"));
        let got = summary(&plan);
        eprintln!("ML-DSA-87 d_sig {d_sig:>2}: {got:?} digest {digest}");
        assert!(plan.runs().iter().all(|r| r.grind_bits() <= 25
            && (r.grind_bits() == 0 || r.nonce_bits() == r.grind_bits() + 7)));
        assert_ne!(digest, PINNED[d_sig - 3], "ML-DSA-87 and ML-DSA-44 rows differ");
        assert_ne!(digest, PINNED_S65[d_sig - 3], "ML-DSA-87 and ML-DSA-65 rows differ");
        summaries.push(got);
        digests.push(digest);
    }
    let expected_summaries: Vec<_> = S87_TABLE.iter().map(|&(_, a, b, c, d, e)| (a, b, c, d, e)).collect();
    assert_eq!(summaries, expected_summaries);
    assert_eq!(
        PINNED_S87,
        digests.iter().map(String::as_str).collect::<Vec<_>>().as_slice()
    );
}

/// (d_sig, PoW sites, sum g, max g, sum 2^g, nonce bits) of the ML-DSA-65 rows under relation H-u.
const S65_TABLE: [(usize, u64, u64, u8, u64, u64); 13] = [
    (3, 406, 979, 10, 4_416, 3_893),
    (4, 406, 979, 10, 4_416, 3_893),
    (5, 406, 979, 10, 4_416, 3_893),
    (6, 406, 979, 10, 4_416, 3_893),
    (7, 388, 967, 10, 5_196, 3_755),
    (8, 388, 967, 10, 5_196, 3_755),
    (9, 388, 966, 10, 4_684, 3_754),
    (10, 388, 965, 10, 4_428, 3_753),
    (11, 452, 1_120, 10, 5_296, 4_368),
    (12, 455, 1_131, 12, 9_920, 4_400),
    (13, 457, 1_133, 12, 8_140, 4_416),
    (14, 459, 1_138, 12, 9_936, 4_435),
    (15, 461, 1_140, 12, 8_156, 4_451),
];

/// The plan digests of the 13 ML-DSA-65 grouped rows at the pinned Akita revision.
const PINNED_S65: &[&str] = &[
    "9938bb7375a3d8effb6c763cb5464d561e626d390c634ae9b1ad7d56500401a8",
    "9938bb7375a3d8effb6c763cb5464d561e626d390c634ae9b1ad7d56500401a8",
    "9938bb7375a3d8effb6c763cb5464d561e626d390c634ae9b1ad7d56500401a8",
    "9938bb7375a3d8effb6c763cb5464d561e626d390c634ae9b1ad7d56500401a8",
    "f044b32e8f8de264a6d7362521b3570b844321b3420f883bbae0da19d33f738e",
    "3a5cf417569d3b6878c456b2db2bde8ea560b93f968af531c6b256ef3f41737f",
    "49838b5c1a2d80fc664dcb13db6784e903b7b0bb389f90f96003f5778e81fcc2",
    "2472f8a767652a253bca23948c56e4d12b91f1243bec5a5625393ece771af788",
    "96aee222e091a016c145baea167d53e29dd5cfb984cbcd81b492923ead1feb6d",
    "3045c1782e00606e866e2b56bffba4b1574523923010117fe38cb23d7aaa6b29",
    "e79f39ad2499e006767187a600c2716c3449cf4d348e74956146ce90c7bd6d70",
    "3fcbaf4b03cb19242acfd8f8e6474e0661a143a9fd5f024f2cfbff253730705e",
    "578c40b1d47e6b6c0a322339584bb2179bd6acbe4fe34e9309ddd80848ed735a",
];

/// (d_sig, PoW sites, sum g, max g, sum 2^g, nonce bits) of the ML-DSA-87 rows under relation H-u.
const S87_TABLE: [(usize, u64, u64, u8, u64, u64); 13] = [
    (3, 388, 967, 10, 5_196, 3_755),
    (4, 388, 967, 10, 5_196, 3_755),
    (5, 388, 967, 10, 5_196, 3_755),
    (6, 388, 966, 10, 4_684, 3_754),
    (7, 388, 966, 10, 4_684, 3_754),
    (8, 388, 966, 10, 4_684, 3_754),
    (9, 430, 1052, 10, 4_860, 4_134),
    (10, 430, 1051, 10, 4_604, 4_133),
    (11, 452, 1_121, 12, 8_112, 4_369),
    (12, 457, 1_134, 12, 8_396, 4_417),
    (13, 459, 1_138, 12, 9_936, 4_435),
    (14, 461, 1_143, 12, 9_948, 4_454),
    (15, 461, 1_140, 12, 8_156, 4_451),
];

/// The plan digests of the 13 ML-DSA-87 grouped rows at the pinned Akita revision.
const PINNED_S87: &[&str] = &[
    "ea906f436336d6953ff82bbb1497b326bba95853069dfc67f385e5a174da17a3",
    "ea906f436336d6953ff82bbb1497b326bba95853069dfc67f385e5a174da17a3",
    "ea906f436336d6953ff82bbb1497b326bba95853069dfc67f385e5a174da17a3",
    "59ad7b990b48deaa9fef9be68b6a8d0cebc149ef483024fa3d1019f08b5e3a18",
    "3692bc922c18a2e79ccf4eaf7db171042e3dafa305def556e61c0194d830bdee",
    "b9ae43f5a428889acb7c517d20a20c764acf38951218a6e518cce2c638dc42b6",
    "8b7c172cb6e23bcc614367687468e1dbf3ac185919e49c2af110a705056a4b17",
    "7acd5f8e63812c1e76612b570ab08ec47c400ad32d07196ee695a7a74162cfa9",
    "f33d456a897853c35c3cb93b5b248234b130c590251f4b16967fe7900abc6d52",
    "de0f26f294616cac7e410da782c53c65c3509e192a9b66be9877b49ccbc4377c",
    "7ed11d214544a1894cd8c8fe76c5c5f27a95835e527d54fa717258a10e8a367a",
    "652a11077cb98b4c3922078ba0e0245f2d29a56521811f45a30f38d9e0ea0f6d",
    "437c6eba5e56883bb03c08f7145a5649aaf8cc2eae24322803529c8a8a7ef104",
];
