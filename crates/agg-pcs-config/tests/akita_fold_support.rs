//! Hypothesis H1 of `docs/soundness.md` Section 7, discharged (Section 4.6 there): every
//! sparse fold challenge of every row of both generated catalogs (the 26 grouped width-4
//! rows of relations H-u and H-exact, the width-4 scalar rows of `G_main4` and the width-16
//! scalar rows of `G_main16`, `G_1`, `G_2`, `G_3`) is drawn from a family whose accepted support, after the sampler's
//! rejection, is at least `2^128`.
//!
//! For every level (root, recursive folds, terminal) and every group of a level, the test
//! runs the pinned Akita revision's own dispatch (`akita_types::draw_group_fold_challenges`
//! for the nonterminal levels, `draw_folding_challenges_with_rejection` with the terminal
//! verifier's arguments for the terminal) against a recording `FoldDraw`. The recorder reads
//! the draw's domain-separated absorb payload, which encodes the draw ring dimension, the
//! family `(count_pm1, count_pm2)`, the coefficient-packing domain and the operator-norm
//! rejection policy, and returns a fixed seed, so the real sampler (with its rejection loop)
//! then produces the challenges, which are checked against the family. The test recomputes
//! the dispatch from the schedule and requires both to agree.
//!
//! Support of a draw (Akita Lemma 10.1: conditioned on success the output is uniform on
//! the accepted set):
//! * no rejection: the exact family size `C(D, w) C(w, count_pm1) 2^w`, `w = count_pm1 +
//!   count_pm2`, compared with `2^128` in integers;
//! * operator-norm rejection: the certified accepted count of the policy's certificate in the
//!   pinned Akita tree (`scripts/operator_norm/d128/cert_d128_w31_gamma13.json`,
//!   `.../d64/cert_d64_a31_b11_gamma18.json`): `ceil(N_raw p0)`, with `p0` the exact rational
//!   lower bound on the accepted fraction of the strict runtime predicate; recomputed from the
//!   certificate files by `docs/scripts/fold_support.py`.
#![cfg(feature = "schedules")]

use std::collections::BTreeMap;

use agg_pcs_config::preset::{AggFp64SignedWidth16, AggFp64SignedWidth4, BaseField, OpeningField};
use akita_challenges::{
    fold_challenge_sample_label, Challenges, FoldChallengeDrawDomain, FoldDraw,
    OperatorNormRejection, SparseChallengeConfig, MIN_FOLD_CHALLENGE_ENTROPY_BITS,
};
use akita_error::AkitaError;
use akita_params::{FoldSchedule, GroupOpenPhaseParams, InnerCommitSecurityRoute, OpeningMethod};
use akita_types::{draw_group_fold_challenges, OpeningFamily};

/// `2^128` in decimal (39 digits).
const TWO_POW_128: &str = "340282366920938463463374607431768211456";

/// Domain bytes of a coefficient-packing draw (Akita `fold_draw.rs`).
const PACKING_DOMAIN: &[u8] = b"akita/subring-coefficient-packing-fold-challenge/v1";

/// A certified operator-norm rejection policy and the accepted count of its certificate.
struct Certified {
    policy: OperatorNormRejection,
    ring_d: usize,
    family: SparseChallengeConfig,
    /// `ceil(N_raw p0)` of the certificate, decimal (`docs/scripts/fold_support.py`).
    accepted: &'static str,
    /// `log2` of `accepted`, for the report (`docs/scripts/fold_support.py`).
    log2_accepted: f64,
}

/// The two certified policies of Akita `e023b6a` (`OperatorNormRejection::D128_SELECTIVE_L2`,
/// `D64_SELECTIVE_L2`). Their predicate parameters are those of the certificates'
/// `fixed_point_containment` (threshold, fractional bits, root error, margin).
fn certified() -> [Certified; 2] {
    [
        Certified {
            policy: OperatorNormRejection::D128_SELECTIVE_L2,
            ring_d: 128,
            family: SparseChallengeConfig::pm1_only(31),
            accepted: "502822610611069237129947393810277364817",
            log2_accepted: 128.563_317,
        },
        Certified {
            policy: OperatorNormRejection::D64_SELECTIVE_L2,
            ring_d: 64,
            family: SparseChallengeConfig {
                count_pm1: 31,
                count_pm2: 11,
            },
            accepted: "355332842887957753779026572075354372186",
            log2_accepted: 128.062_438,
        },
    ]
}

/// `a >= b` for decimal strings without leading zeros.
fn dec_ge(a: &str, b: &str) -> bool {
    (a.len(), a) >= (b.len(), b)
}

fn binom(n: u128, k: u128) -> Option<u128> {
    let k = k.min(n - k);
    let mut acc: u128 = 1;
    for i in 0..k {
        // acc * (n - i) / (i + 1) is exact at every step.
        acc = acc.checked_mul(n - i)? / (i + 1);
    }
    Some(acc)
}

/// Exact test `C(D, w) C(w, count_pm1) 2^w >= 2^128` and the family's `log2` size.
fn raw_support(ring_d: usize, family: SparseChallengeConfig) -> (bool, f64) {
    let w = family.weight() as u128;
    let d = ring_d as u128;
    assert!(w <= d);
    let log2 = |n: u128, k: u128| -> f64 {
        (0..k)
            .map(|i| ((n - i) as f64 / (i + 1) as f64).log2())
            .sum()
    };
    let bits = log2(d, w) + log2(w, family.count_pm1 as u128) + w as f64;
    let ge = match (binom(d, w), binom(w, family.count_pm1 as u128)) {
        (Some(a), Some(b)) => match a.checked_mul(b) {
            // a b 2^w >= 2^128  <=>  a b >= 2^(128 - w) (w < 128 in every family here).
            Some(ab) => w >= 128 || ab >= 1_u128 << (128 - w),
            None => true,
        },
        _ => true,
    };
    (ge, bits)
}

/// What one draw absorbs before its seed is squeezed.
#[derive(Clone, Debug, PartialEq, Eq)]
struct DrawRecord {
    ring_d: usize,
    family: SparseChallengeConfig,
    packing: bool,
    rejection: Option<[u8; 17]>,
}

/// A `FoldDraw` that records the payload and answers a fixed seed.
struct Recorder {
    expected_label: Vec<u8>,
    coordinates: usize,
    record: Option<DrawRecord>,
}

impl FoldDraw for Recorder {
    fn absorb_and_squeeze(&mut self, payload: &[u8]) -> Result<[u8; 32], AkitaError> {
        let rest = payload
            .strip_prefix(self.expected_label.as_slice())
            .expect("the draw label of Akita fold_challenge_sample_label");
        let u64_at = |b: &[u8], i: usize| u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        assert_eq!(
            u64_at(rest, 0) as usize,
            self.coordinates,
            "coordinate count"
        );
        let ring_d = u64_at(rest, 8) as usize;
        assert_eq!(rest[16], 0, "sparse family domain tag");
        let family = SparseChallengeConfig {
            count_pm1: u64_at(rest, 17) as usize,
            count_pm2: u64_at(rest, 25) as usize,
        };
        let mut tail = &rest[33..];
        let packing = tail.starts_with(PACKING_DOMAIN);
        if packing {
            tail = &tail[PACKING_DOMAIN.len()..];
        }
        let rejection = match tail.len() {
            0 => None,
            17 => Some(tail.try_into().unwrap()),
            n => panic!("unexpected {n} trailing payload bytes"),
        };
        self.record = Some(DrawRecord {
            ring_d,
            family,
            packing,
            rejection,
        });
        Ok([0x5a; 32])
    }
}

/// Every sampled challenge is a member of `family` in dimension `ring_d`.
fn check_members(challenges: &Challenges, ring_d: usize, family: SparseChallengeConfig) {
    for c in challenges.as_slice() {
        let mut seen = vec![false; ring_d];
        assert_eq!(c.positions.len(), family.weight());
        let (mut ones, mut twos) = (0, 0);
        for (&p, &v) in c.positions.iter().zip(c.coeffs.iter()) {
            let p = p as usize;
            assert!(
                p < ring_d && !seen[p],
                "distinct positions below the dimension"
            );
            seen[p] = true;
            match v.unsigned_abs() {
                1 => ones += 1,
                2 => twos += 1,
                m => panic!("coefficient magnitude {m}"),
            }
        }
        assert_eq!((ones, twos), (family.count_pm1, family.count_pm2));
    }
}

/// One draw site of a schedule and its support floor.
#[derive(Debug)]
struct Site {
    level: usize,
    group: usize,
    method: &'static str,
    ring_d: usize,
    family: SparseChallengeConfig,
    rejection: Option<usize>,
    bits: f64,
    at_least_128: bool,
}

/// The rejection policy `draw_group_fold_challenges` selects (its code, restated).
fn expected_rejection(params: &GroupOpenPhaseParams) -> Option<OperatorNormRejection> {
    let m = params.inner_commit_matrix_params();
    match params.opening_method() {
        OpeningMethod::EvaluationTrace => {
            matches!(m.security_route(), InnerCommitSecurityRoute::L2 { .. })
                .then(|| {
                    akita_challenges::selective_l2_operator_norm_rejection(
                        m.ring_dimension(),
                        &params.fold_challenge_config(),
                    )
                })
                .flatten()
        }
        OpeningMethod::SubringCoefficientPacking { .. } => None,
    }
}

fn floor_of(record: &DrawRecord) -> (Option<usize>, f64, bool) {
    match record.rejection {
        None => {
            let (ge, bits) = raw_support(record.ring_d, record.family);
            (None, bits, ge)
        }
        Some(bytes) => {
            let certs = certified();
            let (i, cert) = certs
                .iter()
                .enumerate()
                .find(|(_, c)| c.policy.domain_separator_bytes() == bytes)
                .expect("a certified operator-norm policy");
            assert_eq!(
                (record.ring_d, record.family),
                (cert.ring_d, cert.family),
                "the certificate covers exactly this family"
            );
            (
                Some(i),
                cert.log2_accepted,
                dec_ge(cert.accepted, TWO_POW_128),
            )
        }
    }
}

/// Every draw site of one schedule, through the pinned dispatch and sampler.
fn sites(schedule: &FoldSchedule, root_claims: &[usize]) -> Vec<Site> {
    let mut out = Vec::new();
    let levels = std::iter::once(&schedule.root).chain(&schedule.recursive_folds);
    for (level, fold) in levels.enumerate() {
        for (group, params) in fold.params.groups().iter().enumerate() {
            let claims = if level == 0 { root_claims[group] } else { 1 };
            let coordinates = params.num_live_blocks() * claims;
            let mut rec = Recorder {
                expected_label: fold_challenge_sample_label(
                    group,
                    params.num_live_blocks(),
                    claims,
                )
                .unwrap(),
                coordinates,
                record: None,
            };
            let drawn = draw_group_fold_challenges::<BaseField, OpeningField, _>(
                &mut rec, params, group, claims,
            )
            .expect("the pinned dispatch draws this group");
            let record = rec.record.expect("one draw");
            // The recorded draw is the one the schedule prescribes.
            let (method, ring_d, packing) = match params.opening_method() {
                OpeningMethod::EvaluationTrace => (
                    "trace",
                    params.inner_commit_matrix_params().ring_dimension(),
                    false,
                ),
                OpeningMethod::SubringCoefficientPacking {
                    challenge_subring_dimension,
                } => ("packing", challenge_subring_dimension, true),
            };
            assert_eq!(
                record,
                DrawRecord {
                    ring_d,
                    family: params.fold_challenge_config(),
                    packing,
                    rejection: expected_rejection(params).map(|p| p.domain_separator_bytes()),
                },
                "level {level} group {group}"
            );
            let canonical = match &drawn {
                OpeningFamily::EvaluationTrace(c) => c,
                OpeningFamily::SubringCoefficientPacking(c) => c.canonical(),
            };
            assert_eq!(canonical.len(), coordinates);
            check_members(canonical, ring_d, record.family);
            let (rejection, bits, at_least_128) = floor_of(&record);
            out.push(Site {
                level,
                group,
                method,
                ring_d,
                family: record.family,
                rejection,
                bits,
                at_least_128,
            });
        }
    }
    // Terminal: the arguments of `akita_verifier::fold::terminal` (one group, one claim).
    let t = &schedule.terminal;
    let level = schedule.recursive_folds.len() + 1;
    let rejection = t.response_l2_sq_cap().map(|_| {
        akita_challenges::selective_l2_operator_norm_rejection(t.d_a(), &t.fold_challenge_config)
            .expect("a terminal L2 route has a certified family")
    });
    let mut rec = Recorder {
        expected_label: fold_challenge_sample_label(0, t.blocks.live_blocks, 1).unwrap(),
        coordinates: t.blocks.live_blocks,
        record: None,
    };
    let drawn = rec
        .draw_folding_challenges_with_rejection(
            FoldChallengeDrawDomain::EvaluationTrace,
            t.d_a(),
            0,
            t.blocks.live_blocks,
            1,
            &t.fold_challenge_config,
            rejection,
        )
        .expect("the terminal draw");
    let record = rec.record.take().expect("one draw");
    assert_eq!(
        record,
        DrawRecord {
            ring_d: t.d_a(),
            family: t.fold_challenge_config,
            packing: false,
            rejection: rejection.map(|p| p.domain_separator_bytes()),
        }
    );
    check_members(&drawn, t.d_a(), record.family);
    let (rejection, bits, at_least_128) = floor_of(&record);
    out.push(Site {
        level,
        group: 0,
        method: "terminal",
        ring_d: t.d_a(),
        family: record.family,
        rejection,
        bits,
        at_least_128,
    });
    out
}

/// Distinct (method, ring dimension, family, rejection) of a catalog: count and floor.
type Census = BTreeMap<(&'static str, usize, usize, usize, Option<usize>), (usize, f64)>;

fn audit_catalog(
    name: &str,
    rows: Vec<(Vec<usize>, FoldSchedule)>,
    expected_rows: usize,
) -> (Census, f64, usize) {
    assert_eq!(rows.len(), expected_rows, "{name}: rows");
    let mut census = Census::new();
    let mut min_bits = f64::INFINITY;
    let mut grouped = 0;
    for (row, (claims, schedule)) in rows.iter().enumerate() {
        let row_sites = sites(schedule, claims);
        let row_min = row_sites
            .iter()
            .map(|s| s.bits)
            .fold(f64::INFINITY, f64::min);
        let root = &schedule.root.params;
        grouped += usize::from(claims.len() == 6);
        eprintln!(
            "{name} row {row:>2}: {} root group(s), own group {} vars x {} polys, {} levels, \
             {} draws, min {row_min:.4} bits",
            claims.len(),
            root.final_group().profile.group.num_vars(),
            root.final_group().profile.group.num_polynomials(),
            schedule.num_fold_levels(),
            row_sites.len(),
        );
        for s in row_sites {
            assert!(
                s.at_least_128,
                "{name} row {row}: level {} group {} ({} D = {}, {:?}, rejection {:?}) has \
                 {:.4} bits of accepted support, below {MIN_FOLD_CHALLENGE_ENTROPY_BITS}",
                s.level, s.group, s.method, s.ring_d, s.family, s.rejection, s.bits
            );
            min_bits = min_bits.min(s.bits);
            let e = census
                .entry((
                    s.method,
                    s.ring_d,
                    s.family.count_pm1,
                    s.family.count_pm2,
                    s.rejection,
                ))
                .or_insert((0, s.bits));
            e.0 += 1;
        }
    }
    for ((method, d, a, b, rej), (n, bits)) in &census {
        eprintln!(
            "{name}: {method:>8} D {d:>4} ({a:>2}, {b:>2}) rejection {:<12} {n:>4} draws  \
             {bits:.4} bits",
            match rej {
                None => "none".to_string(),
                Some(0) => "op-norm 13".to_string(),
                Some(_) => "op-norm 18".to_string(),
            }
        );
    }
    eprintln!(
        "{name}: {} rows ({grouped} grouped), minimum accepted support {min_bits:.4} bits",
        rows.len()
    );
    (census, min_bits, grouped)
}

/// The rows of a catalog with their root claim counts (polynomials per root group).
fn rows_of(catalog: &akita_schedules::ValidatedScheduleCatalog) -> Vec<(Vec<usize>, FoldSchedule)> {
    catalog
        .rows()
        .map(|row| {
            let layout = row.profiles().opening_layout().expect("opening layout");
            let claims = (0..layout.num_groups())
                .map(|g| layout.group_layout(g).expect("group").num_polynomials())
                .collect();
            (claims, row.schedule().clone())
        })
        .collect()
}

#[test]
fn certified_floors_and_policies() {
    for c in certified() {
        assert!(dec_ge(c.accepted, TWO_POW_128));
        assert!(c.log2_accepted > 128.0);
        // Exactly the certified family and predicate (the certificates' parameters).
        c.policy
            .validate(c.ring_d, &c.family)
            .expect("policy covers the family");
        let expected = match c.ring_d {
            128 => (13, 48, 4, 351),
            64 => (18, 48, 4, 600),
            _ => unreachable!(),
        };
        let p = c.policy;
        assert_eq!(
            (
                p.threshold,
                p.fractional_bits,
                p.root_coordinate_error_units,
                p.rounding_margin_units
            ),
            expected
        );
        // The raw family is larger than the accepted floor.
        let (ge, raw) = raw_support(c.ring_d, c.family);
        assert!(ge && raw > c.log2_accepted);
    }
    // The exact family sizes at the two edges of Akita's production ladder.
    assert!(
        raw_support(
            64,
            SparseChallengeConfig {
                count_pm1: 31,
                count_pm2: 10
            }
        )
        .0
    );
    assert!(
        !raw_support(
            64,
            SparseChallengeConfig {
                count_pm1: 30,
                count_pm2: 10
            }
        )
        .0
    );
    assert!(raw_support(128, SparseChallengeConfig::pm1_only(31)).0);
    assert!(!raw_support(128, SparseChallengeConfig::pm1_only(30)).0);
}

#[test]
fn every_fold_challenge_family_has_128_bits_of_accepted_support() {
    let w4 = AggFp64SignedWidth4::catalog().expect("width-4 catalog");
    let w16 = AggFp64SignedWidth16::catalog().expect("width-16 catalog");
    // Width 4, per relation (H-u, H-exact): 13 grouped rows and 13 scalar rows of G_main4
    // (8 or 9 polynomials), 52 rows; width 16: 13 scalar rows each of G_main16, G_1, G_2,
    // G_3, with equal layouts sharing one row (the same for both relations).
    let (c4, m4, g4) = audit_catalog("width4", rows_of(w4.catalog()), w4.catalog().len());
    let (c16, m16, g16) = audit_catalog("width16", rows_of(w16.catalog()), w16.catalog().len());
    assert!(m4 >= 128.0 && m16 >= 128.0);
    // The 39 grouped rows (six root groups, one per N* = 2^3 .. 2^15 and relation: H-u,
    // H-exact, H-FIPS) are in the width-4 catalog (the family of the final group G_m).
    assert_eq!((g4, g16), (39, 0));
    // The families that occur (docs/soundness.md Section 4.6).
    let families = |c: &Census| -> Vec<(usize, usize, usize, Option<usize>)> {
        let mut v: Vec<_> = c.keys().map(|&(_, d, a, b, r)| (d, a, b, r)).collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    eprintln!("width4 families {:?}", families(&c4));
    eprintln!("width16 families {:?}", families(&c16));
}

#[test]
fn grouped_rows_by_batch_size() {
    use agg_pcs_config::catalogs::Relation;
    use agg_pcs_config::work_plan::WorkPlan;
    // docs/soundness.md 4.6: (d_sig, draws, minimum accepted support in bits, rounded down
    // to 4 decimals) of the grouped row of N* = 2^d_sig, per relation. H-u's row at 2^15
    // avoids the D = 64 packing family; every H-exact row draws it (its level 1 at 2^15).
    let expected_h_u: [(usize, usize, f64); 13] = [
        (3, 11, 128.0881),
        (4, 11, 128.0881),
        (5, 11, 128.0881),
        (6, 11, 128.0881),
        (7, 11, 128.0881),
        (8, 11, 128.0881),
        (9, 11, 128.0881),
        (10, 11, 128.0881),
        (11, 11, 128.0881),
        (12, 12, 128.0881),
        (13, 12, 128.0881),
        (14, 12, 128.0881),
        (15, 12, 128.5633),
    ];
    let expected_h_exact: [(usize, usize, f64); 13] = [
        (3, 11, 128.0881),
        (4, 11, 128.0881),
        (5, 11, 128.0881),
        (6, 11, 128.0881),
        (7, 11, 128.0881),
        (8, 11, 128.0881),
        (9, 11, 128.0881),
        (10, 11, 128.0881),
        (11, 11, 128.0881),
        (12, 12, 128.0881),
        (13, 12, 128.0881),
        (14, 12, 128.0881),
        (15, 12, 128.0881),
    ];
    let catalog = AggFp64SignedWidth4::catalog().expect("catalog");
    let (mut got, mut want) = (Vec::new(), Vec::new());
    for relation in Relation::ALL {
        let expected: &[(usize, usize, f64)] = match relation {
            Relation::Hu => &expected_h_u,
            Relation::HExact => &expected_h_exact,
            Relation::HFips => &EXPECTED_H_FIPS,
        };
        assert_eq!(relation.catalog_d_sigs().len(), expected.len());
        for &(d_sig, draws, min_bits) in expected {
            let plan = WorkPlan::for_records(1 << d_sig, relation).expect("work plan");
            let row = catalog
                .resolve_key(&plan.schedule_key())
                .expect("grouped row");
            let (claims, schedule) = rows_of_one(row.profiles(), row.schedule());
            let row_sites = sites(&schedule, &claims);
            let min = row_sites
                .iter()
                .map(|s| s.bits)
                .fold(f64::INFINITY, f64::min);
            let line: Vec<String> = row_sites
                .iter()
                .map(|s| {
                    format!(
                        "L{}g{}:{}{}({},{}){}",
                        s.level,
                        s.group,
                        &s.method[..1],
                        s.ring_d,
                        s.family.count_pm1,
                        s.family.count_pm2,
                        if s.rejection.is_some() { "r" } else { "" }
                    )
                })
                .collect();
            eprintln!(
                "{} d_sig {d_sig:>2}: {} draws, min {min:.4} bits: {}",
                relation.name(),
                row_sites.len(),
                line.join(" ")
            );
            assert!(row_sites.iter().all(|s| s.at_least_128));
            got.push((relation, d_sig, row_sites.len(), (min * 1e4).floor() / 1e4));
            want.push((relation, d_sig, draws, min_bits));
        }
    }
    assert_eq!(got, want);
}

/// (d_sig, draws, minimum accepted support in bits) of the H-FIPS grouped rows
/// (`mldsa-44-l16f-batch-<N*>-mult-final`), one entry per checked-in batch size.
const EXPECTED_H_FIPS: [(usize, usize, f64); 13] = [
    (3, 11, 128.0881),
    (4, 11, 128.0881),
    (5, 11, 128.0881),
    (6, 11, 128.0881),
    (7, 11, 128.0881),
    (8, 11, 128.0881),
    (9, 11, 128.0881),
    (10, 11, 128.0881),
    (11, 11, 128.0881),
    (12, 12, 128.0881),
    (13, 12, 128.5633),
    (14, 12, 128.0881),
    (15, 12, 128.5633),
];

/// Claims per root group and the schedule of one resolved row.
fn rows_of_one(
    profiles: &akita_params::CommittedGroupBatchProfile,
    schedule: &FoldSchedule,
) -> (Vec<usize>, FoldSchedule) {
    let layout = profiles.opening_layout().expect("opening layout");
    let claims = (0..layout.num_groups())
        .map(|g| layout.group_layout(g).expect("group").num_polynomials())
        .collect();
    (claims, schedule.clone())
}
