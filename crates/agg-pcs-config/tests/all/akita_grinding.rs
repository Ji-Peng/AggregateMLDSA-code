//! Akita's transcript proof-of-work plan for every grouped catalog row
//! (`docs/soundness.md`): the pinned Akita revision derives the only
//! accepted plan from the row's schedule and the opening layout and binds its digest into
//! the instance descriptor, on both the prover and the verifier side. This test pins the
//! plan of each row (sites, difficulties, digest) as a regression guard against a catalog or
//! Akita change.
#![cfg(feature = "schedules")]

use agg_pcs_config::catalogs::{Group, Relation, CATALOG_D_SIG};
use agg_pcs_config::preset::AggFp64SignedWidth4;
use agg_pcs_config::work_plan::WorkPlan;
use akita_config::derive_transcript_grinding_plan;
use akita_params::{GrindingPlan, GrindingQueryKind, OpeningClaimsLayout};

/// The plan of the grouped row of `N* = 2^d_sig` for the opening of its seven groups.
fn row_plan(d_sig: usize, relation: Relation) -> GrindingPlan {
    let plan = WorkPlan::for_records(1 << d_sig, relation).expect("work plan");
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
    // (d_sig, PoW sites, sum g, max g, sum 2^g, nonce bits).
    let table: [(usize, u64, u64, u8, u64, u64); 13] = [
        (3, 401, 984, 9, 3_888, 3_863),
        (4, 401, 984, 9, 3_888, 3_863),
        (5, 401, 984, 9, 3_888, 3_863),
        (6, 401, 985, 10, 4_400, 3_864),
        (7, 406, 980, 10, 5_120, 3_894),
        (8, 406, 980, 10, 5_120, 3_894),
        (9, 388, 966, 10, 5_132, 3_754),
        (10, 388, 967, 12, 7_692, 3_755),
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
    assert!(digests[0].starts_with("28f85737"), "{}", digests[0]);
    assert!(digests[12].starts_with("42566443"), "{}", digests[12]);
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
    "28f85737d91c69f8e2bf28311672f537cc4fe8318a7889666afec42f3006ebb6",
    "28f85737d91c69f8e2bf28311672f537cc4fe8318a7889666afec42f3006ebb6",
    "6190c528bbb8013bf984831bd67d2776ed07bf4962a6988fc766d4e9ff4de3c6",
    "57cf539168b29bf5370ce1182e76598735e1b140356dbcf84884ea5a7e57fc23",
    "e84823700fef62202066116dc63dcf743b24faecd59297d26c9000d7b57aef30",
    "64a048ff368e4959c802ba143b877bd02aefdc8c750c4bf19a8d8c3f46129b8a",
    "472ee9543d4812f90960d053b4d924b303eebd4a0e998bbad6344fda246f3290",
    "d17c54ac71bf131f68a119f0d20dabba4dcad5529f3385a7234cf6a4ccbb7d0a",
    "cd03815e65126ec59718522c11ca248682840d61c2c2f66ca094ef88a16ce26e",
    "a5f3bc210d8473d7ab61d4da25293efd637b84c1b5700585d4ecbb4215c86e8d",
    "7376c898144090fde30641d0c42fd58b148453402c38874949a1b4547bb3397b",
    "54cfa6ce2bce0bee8f4b6dbac9095510471dc965a144b10a1336f0f33090cd1b",
    "4256644335744082d053ec61cdccb3b36b0b094ba8824b0a3a23cff98e0dc433",
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
    "c2cab4d90a7334f98832722b02df3d105edec29d6bc4fb21e6a55b543e120dd4",
    "c2cab4d90a7334f98832722b02df3d105edec29d6bc4fb21e6a55b543e120dd4",
    "8621dc3d473dd9e92ba85ead4bbf99fb12a87c4269990e037b9c1192b0f4e69e",
    "79aef977898df6cc5187605af45f3a168db3ebe2c3de276c5bdeb6ed08213bf3",
    "b9b799639628f6f42f99f7d4cb61666b5fed514fe2cd2bc29d743da094330e49",
    "4a70b936032eb69db5bf8d68422cc8ba6b36ff175d661a6f5402537d0e876e1c",
    "03a70014fe8da1e0cb5005fdfe3789c65039a8575bc2c9926b04b7ebbf6c4f52",
    "a3aedf74f90e5cfe076f4b5cf7daf67695dc321875f74793a6a11563f679f755",
    "e0f050e56f2cebce23e60d6225e25887f8286f727ab133de025b5708511a307c",
    "2c9816f25bd6e427892b42e33db34cff39c29dbe94a955f2bb92b1e1c6650bf8",
    "a64100f497537c1d6155269a68112ba52615deae827f0ee48e609381fa7a6038",
    "724665515e0941a6cc8d7fd032985f7402bb85311c8d020fe489074620e44271",
    "f8b3865877528f9372196be74f8a1127687e17ac27de080e595b96ffaab55f8c",
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
    (388, 966, 10, 5_132, 3_754),
    (388, 967, 12, 7_692, 3_755),
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
    "dc0079ce937139db5790f740d90fde865c9f906550c2420a832927bea745be23",
    "dc0079ce937139db5790f740d90fde865c9f906550c2420a832927bea745be23",
    "8e417b227644ffe94429bd6dfd42c4c68ead0dea2dd7f21c3af039d59107a84a",
    "60075e649a96878bfdc0702dd450165a3df07e9c84f2f7dc4ed35ced7b016891",
    "0610d751990aa40f65f76781c6cb07cdec7e0f7d2ffc84a4ce57cc04d083ff1d",
    "a2e3ef144d1d5e97581e9bc12d08a6faadab3b8eeb482ff293d0e3b9c51cac5c",
    "ee90f304f41fb187b9e50478748ce5ec5866ddb0c0d6a31a549836919f1d0572",
    "b0311507b12e4cd2083c9fb830a73df402243a0692b34ce60d10eebbc8f21eab",
    "89a821f5945b5129dbc66083ebfeceea7b283ce60ae7136aab450e301acc2c28",
    "df8c01781862024ee953d7e4d02c32a7b7ecbc3bd2406b42c848e531e41d5879",
    "c1bfe3fe7306ac5f58fcfe278cee58ab103a952e13748421b431699074c97823",
    "a13565f0ba77f732f6d857b21a868858ab5181b7df5b42a22cad1b5bd3dd61ee",
    "01b4c2c0397ce3dcb606ff686d4a04fcdef9df265fc3ff19db8ce55bec2ed74b",
];
