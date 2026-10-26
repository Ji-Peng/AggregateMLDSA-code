//! Mode A end to end (spec Sections 10, 11, 13): the aggregator on real ML-DSA-44 records
//! and the verifier on `(pk_i, M_i)` and the aggregate.
//!
//! - Round trips at 64, 256 (two Flock parts), 512 and (`Slim`) 1,024 signatures under each mode-A Flock
//!   profile (`Fast`, `Slim`, `Slim8`): the aggregate size per part, the prover's and the verifier's
//!   step costs (uncached and cached keys) are printed (`--nocapture`), the cached and
//!   uncached paths agree, the persistent key cache round-trips, the verifier reads the
//!   profile from the aggregate and a verifier pinned to the other profile rejects it.
//! - Corrupted aggregates (`corrupted_aggregates_are_rejected`): every byte region of the
//!   wire format, the mode-A segment items (part size, part count, shares, part proofs,
//!   commitments), the tie nonce and the tied-part proof version, the PIOP work nonces, a
//!   plan-digest mismatch, the records (order, message, key, count) and the configuration.
//! - Tampered witnesses (`tampered_witnesses_are_rejected_end_to_end`): every witness
//!   tampering of the earlier stages (`agg-piop/tests/mode_b.rs`, `tests/joint_bridge.rs`,
//!   `agg-keccak/tests/{proofs,bridge}.rs`) pushed through the mode-A aggregator (stages
//!   unchecked) and rejected by the verifier at the stated step.
//! - Invalid signatures are refused by the aggregator (`agg-witness/tests/negative.rs`).
//!
//! Run with `--release`.

use std::time::Instant;

use agg_cli::cache;
use agg_cli::mode_a::{
    AggregateError, AggregateHooks, AggregateVerifier, Aggregator, AggregatorConfig,
    SegmentRejection, VerifyCostReport, DEFAULT_PART_LOG2, FLOCK_PROFILE,
};
use agg_cli::wire::{ItemError, SegmentCommitments, SegmentProofs, SegmentShares};
use agg_keccak::sponge::{sponge_gate, SpongeRecord, W1_INPUT_BASE};
use agg_keccak::tied;
use agg_keccak::witness::pack_slot;
use agg_piop::binding::FlockProfile;
use agg_piop::manifest::{Column, CommitmentManifest, GroupRole, Relation};
use agg_piop::mode_b::PublicBitsVerifier;
use agg_piop::prover::{ProveError, SeededMatrices};
use agg_piop::public_data::KeyCache;
use agg_piop::source_tables::{BridgeGroupTables, MainTables};
use agg_piop::stage::StageError;
use agg_piop::transcript::TranscriptError;
use agg_piop::verifier::{verify_timed, ConfigurationError, VerifyError};
use agg_piop::wire::AggregateProof;
use agg_witness::gen::{self, Record};
use agg_witness::padding::main_index;
use agg_witness::params::{C_TILDE_BYTES, K, OMEGA, SIG_BYTES, Z_BITS};

fn on_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("test thread");
}

/// Real records: `own` with their own key, `shared` over three shared keys.
fn records(seed: u64, own: usize, shared: usize) -> Vec<Record> {
    let mut records = gen::records(seed, own);
    records.extend(gen::records_shared_keys(seed + 1, shared, 3));
    records
}

fn signed(records: &[Record]) -> Vec<(&[u8], &[u8], &[u8])> {
    records
        .iter()
        .map(|r| (r.pk.as_slice(), r.msg.as_slice(), r.sig.as_slice()))
        .collect()
}

fn statements(records: &[Record]) -> Vec<(&[u8], &[u8])> {
    records
        .iter()
        .map(|r| (r.pk.as_slice(), r.msg.as_slice()))
        .collect()
}

fn rejected(report: &VerifyCostReport) -> &VerifyError {
    report
        .report
        .outcome
        .as_ref()
        .expect_err("the verifier must reject")
}

fn peak_mib() -> f64 {
    agg_cli::rss::peak_kib().map_or(0.0, |k| k as f64 / 1024.0)
}

/// Another mode-A Flock profile.
fn other(profile: FlockProfile) -> FlockProfile {
    if profile == FlockProfile::Fast {
        FlockProfile::Slim
    } else {
        FlockProfile::Fast
    }
}

/// An honest round trip under `profile`: aggregate, check the `c~_i` list and the size
/// report, verify uncached, then with an in-memory and a persisted key cache; a verifier
/// pinned to the other profile rejects the aggregate at the item's profile byte.
fn round_trip(
    n_own: usize,
    n_shared: usize,
    seed: u64,
    part_log2: u32,
    parts: usize,
    profile: FlockProfile,
) {
    round_trip_as(Relation::Hu, n_own, n_shared, seed, part_log2, parts, profile);
}

/// [`round_trip`] under `relation`; a verifier of the other relation rejects the aggregate
/// before event 0 (`docs/h-exact.md` 1.3).
fn round_trip_as(
    relation: Relation,
    n_own: usize,
    n_shared: usize,
    seed: u64,
    part_log2: u32,
    parts: usize,
    profile: FlockProfile,
) {
    let records = records(seed, n_own, n_shared);
    let n = records.len();
    let aggregator = Aggregator::new(AggregatorConfig {
        part_log2,
        profile,
        relation,
        ..AggregatorConfig::default()
    });
    let start = Instant::now();
    let aggregate = aggregator.aggregate(&signed(&records)).expect("aggregate");
    eprintln!(
        "N = {n}: aggregated in {:.2} s",
        start.elapsed().as_secs_f64()
    );
    eprintln!("{}", aggregate.report);
    eprintln!(
        "N = {n}: aggregate {} B, {:.1} B per signature, process peak RSS so far {:.0} MiB",
        aggregate.bytes.len(),
        aggregate.report.bytes_per_signature(),
        peak_mib()
    );
    assert_eq!(aggregate.report.size.total(), aggregate.bytes.len());
    assert_eq!(aggregate.report.parts.len(), parts, "Flock parts");
    assert_eq!(aggregate.c_tilde().len(), n);
    for (c, r) in aggregate.c_tilde().iter().zip(&records) {
        assert_eq!(
            c[..],
            r.sig[..C_TILDE_BYTES],
            "c~ list is the signatures' c~"
        );
    }
    assert_eq!(
        AggregateProof::from_bytes(&aggregate.bytes).expect("parse"),
        aggregate.proof
    );
    let items = SegmentCommitments::from_bytes(&aggregate.proof.flock_commitments[0]).unwrap();
    assert_eq!(items.commitments.len(), parts);
    assert_eq!(items.profile, profile as u8);
    assert_eq!(aggregate.report.profile, profile);
    assert_eq!((aggregate.report.relation, aggregate.proof.relation), (relation, relation));
    assert_eq!(
        items.d_part,
        part_log2.min(n.next_power_of_two().max(8).trailing_zeros())
    );

    let statements = statements(&records);
    if profile == FlockProfile::Slim8 {
        // A default verifier only accepts Fast and Slim; Slim8 requires opt-in via require().
        let default_verifier = AggregateVerifier::new().relation(relation);
        let rejected = default_verifier.verify(&statements, &aggregate.bytes, None);
        assert!(!rejected.accepted(), "default verifier must reject Slim8 without opt-in");
        assert!(matches!(
            rejected.rejection,
            Some((0, SegmentRejection::Item(ItemError::Profile(p)))) if p == FlockProfile::Slim8 as u8
        ));
        assert!(rejected.parts.is_empty(), "no Flock part run on rejected Slim8");
    }

    let verifier = if profile == FlockProfile::Slim8 {
        AggregateVerifier::new().require(FlockProfile::Slim8)
    } else {
        AggregateVerifier::new()
    }
    .relation(relation);
    let uncached = verifier.verify(&statements, &aggregate.bytes, None);
    eprintln!("{uncached}");
    assert_eq!(uncached.profile, profile, "the profile read from the aggregate");
    assert_eq!(uncached.relation, relation);
    let verified = uncached.report.outcome.as_ref().expect("honest aggregate");
    let distinct = {
        let mut keys: Vec<&[u8]> = records.iter().map(|r| r.pk.as_slice()).collect();
        keys.sort();
        keys.dedup();
        keys.len()
    };
    assert_eq!(verified.expanded_keys, distinct);
    assert_eq!(verified.records, n);
    assert_eq!(verified.bridge_seed, aggregate.report.bridge_seed);
    assert_eq!(uncached.parts.len(), parts);
    assert!(uncached.parts.iter().all(|p| p.accepted));
    let mut key_cache = KeyCache::default();
    let warm = verifier.verify(&statements, &aggregate.bytes, Some(&mut key_cache));
    assert!(warm.accepted());
    assert_eq!(key_cache.len(), distinct);
    let cached = verifier.verify(&statements, &aggregate.bytes, Some(&mut key_cache));
    eprintln!("{cached}");
    assert_eq!(
        cached
            .report
            .outcome
            .as_ref()
            .expect("cached")
            .expanded_keys,
        0
    );
    // The persisted cache (keyed by SHA3-256(pk)) gives the same cached path.
    let mut reloaded = cache::from_bytes(&cache::to_bytes(&key_cache)).expect("cache file");
    let from_file = verifier.verify(&statements, &aggregate.bytes, Some(&mut reloaded));
    assert_eq!(
        from_file
            .report
            .outcome
            .as_ref()
            .expect("from file")
            .expanded_keys,
        0
    );
    eprintln!(
        "N = {n}: verify uncached {:.1} ms, cached {:.1} ms; process peak RSS {:.0} MiB",
        uncached.total().as_secs_f64() * 1e3,
        cached.total().as_secs_f64() * 1e3,
        peak_mib()
    );
    // Pinned verifiers: the aggregate's own profile accepts, the other one rejects at the
    // profile byte of the event-2 item (before any Flock work).
    let pinned = AggregateVerifier::new().require(profile).relation(relation);
    assert!(pinned.verify(&statements, &aggregate.bytes, None).accepted());
    let other_relation = match relation {
        Relation::Hu => Relation::HExact,
        Relation::HExact => Relation::Hu,
    };
    let crossed = pinned.relation(other_relation).verify(&statements, &aggregate.bytes, None);
    assert!(
        matches!(rejected(&crossed), VerifyError::Relation { expected, found } if *expected == other_relation && *found == relation),
        "{crossed}"
    );
    let wrong = AggregateVerifier::new()
        .require(other(profile))
        .relation(relation)
        .verify(&statements, &aggregate.bytes, None);
    expect_rejected("verifier pinned to the other profile", &wrong, keccak);
    assert!(matches!(
        wrong.rejection,
        Some((0, SegmentRejection::Item(ItemError::Profile(p)))) if p == profile as u8
    ));
    assert!(wrong.parts.is_empty(), "no Flock part was run");
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Flock and Akita proofs: run with --release"
)]
fn round_trip_at_64_signatures() {
    on_large_stack(|| round_trip(48, 16, 0xa064, DEFAULT_PART_LOG2, 1, FlockProfile::Fast));
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Flock and Akita proofs: run with --release"
)]
fn round_trip_at_64_signatures_slim() {
    on_large_stack(|| round_trip(48, 16, 0xb064, DEFAULT_PART_LOG2, 1, FlockProfile::Slim));
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Flock and Akita proofs: run with --release"
)]
fn round_trip_at_256_signatures_in_two_flock_parts() {
    on_large_stack(|| round_trip(200, 56, 0xa256, 7, 2, FlockProfile::Fast));
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Flock and Akita proofs: run with --release"
)]
fn round_trip_at_256_signatures_in_two_flock_parts_slim() {
    on_large_stack(|| round_trip(200, 56, 0xb256, 7, 2, FlockProfile::Slim));
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Flock and Akita proofs: run with --release"
)]
fn h_exact_round_trips_at_64_and_256_signatures_slim() {
    on_large_stack(|| {
        round_trip_as(Relation::HExact, 48, 16, 0xe064, DEFAULT_PART_LOG2, 1, FlockProfile::Slim);
        round_trip_as(Relation::HExact, 200, 56, 0xe256, 7, 2, FlockProfile::Slim);
    });
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Flock and Akita proofs: run with --release"
)]
fn round_trip_at_1024_signatures_slim() {
    on_large_stack(|| round_trip(800, 224, 0xb400, DEFAULT_PART_LOG2, 1, FlockProfile::Slim));
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Flock and Akita proofs: run with --release"
)]
fn round_trip_at_512_signatures() {
    on_large_stack(|| round_trip(400, 112, 0xa512, DEFAULT_PART_LOG2, 1, FlockProfile::Fast));
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Flock and Akita proofs: run with --release"
)]
fn round_trip_at_512_signatures_slim() {
    on_large_stack(|| round_trip(400, 112, 0xb512, DEFAULT_PART_LOG2, 1, FlockProfile::Slim));
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Flock and Akita proofs: run with --release"
)]
fn round_trips_at_64_256_and_512_signatures_slim8() {
    on_large_stack(|| {
        round_trip(48, 16, 0xc064, DEFAULT_PART_LOG2, 1, FlockProfile::Slim8);
        round_trip(200, 56, 0xc256, 7, 2, FlockProfile::Slim8);
        round_trip(400, 112, 0xc512, DEFAULT_PART_LOG2, 1, FlockProfile::Slim8);
    });
}

/// Expect a rejection matching `check`, print it.
fn expect_rejected(label: &str, report: &VerifyCostReport, check: impl Fn(&VerifyError) -> bool) {
    let error = rejected(report);
    eprintln!(
        "{label}: rejected with {error}{}",
        report
            .rejection
            .as_ref()
            .map(|(s, why)| format!(" (segment {s}: {why:?})"))
            .unwrap_or_default()
    );
    assert!(check(error), "{label}: {error:?}");
}

fn at_stage(s: usize) -> impl Fn(&VerifyError) -> bool {
    move |e| matches!(e, VerifyError::Stage(StageError::Terminal(t)) if *t == s)
}

/// A corrupted absorbed message: rejected at the terminal check of stage `s`, or earlier at
/// the next challenge's work nonce, whose predicate depends on the message.
fn at_stage_or_work(s: usize) -> impl Fn(&VerifyError) -> bool {
    move |e| {
        at_stage(s)(e)
            || matches!(
                e,
                VerifyError::Stage(StageError::Transcript(TranscriptError::Work { .. }))
                    | VerifyError::Transcript(TranscriptError::Work { .. })
            )
    }
}

fn keccak(e: &VerifyError) -> bool {
    matches!(e, VerifyError::Keccak(0))
}

/// Re-encode segment 0's items of a proof (event 2, the event-6 shares, the join's proofs)
/// after `f` changed them.
fn with_all_items(
    proof: &AggregateProof,
    f: impl Fn(&mut SegmentCommitments, &mut SegmentShares, &mut SegmentProofs),
) -> AggregateProof {
    let mut c = SegmentCommitments::from_bytes(&proof.flock_commitments[0]).unwrap();
    let parts = c.commitments.len();
    let mut s = SegmentShares::from_bytes(&proof.flock_shares[0], parts).unwrap();
    let mut p = SegmentProofs::from_bytes(&proof.flock_proofs[0], parts).unwrap();
    f(&mut c, &mut s, &mut p);
    let mut out = proof.clone();
    out.flock_commitments[0] = c.to_bytes();
    out.flock_shares[0] = s.to_bytes();
    out.flock_proofs[0] = p.to_bytes();
    out
}

/// [`with_all_items`] on the event-2 item and the join's proof item.
fn with_items(
    proof: &AggregateProof,
    f: impl Fn(&mut SegmentCommitments, &mut SegmentProofs),
) -> AggregateProof {
    with_all_items(proof, |c, _, p| f(c, p))
}

/// A Flock verifier that records the Flock verdict and accepts anyway (to observe the F_p
/// side on its own, as in `agg-piop/tests/joint_bridge.rs`).
struct Lenient<'g>(agg_cli::mode_a::FlockVerifier<'g>);

impl agg_piop::verifier::KeccakVerifier for Lenient<'_> {
    fn profile(&self) -> FlockProfile {
        agg_piop::verifier::KeccakVerifier::profile(&self.0)
    }

    fn plan(
        &self,
        geometry: &agg_witness::padding::Geometry,
        commitments: &[Vec<u8>],
    ) -> Option<agg_piop::pow::KeccakPlan> {
        self.0.plan(geometry, commitments)
    }

    fn verify(&self, segment: &agg_piop::verifier::SegmentContext<'_>) -> bool {
        let _ = self.0.verify(segment);
        true
    }
}

/// The mode-A verifier with every tie difficulty of its plan raised by one (a plan-digest
/// mismatch in event 0).
struct OtherPlan<'g>(agg_cli::mode_a::FlockVerifier<'g>);

impl agg_piop::verifier::KeccakVerifier for OtherPlan<'_> {
    fn profile(&self) -> FlockProfile {
        agg_piop::verifier::KeccakVerifier::profile(&self.0)
    }

    fn plan(
        &self,
        geometry: &agg_witness::padding::Geometry,
        commitments: &[Vec<u8>],
    ) -> Option<agg_piop::pow::KeccakPlan> {
        let mut plan = self.0.plan(geometry, commitments)?;
        plan.parts.iter_mut().for_each(|w| w.tie_bits += 1);
        Some(plan)
    }

    fn verify(&self, segment: &agg_piop::verifier::SegmentContext<'_>) -> bool {
        self.0.verify(segment)
    }
}

/// Two genuine keys sharing `rho` (memo 3.4 (c): `rho` is the signer's choice, so the
/// verifier must key its expansions and records by the full `pk`): an aggregate over records
/// of both keys verifies (uncached, with only the first key cached, fully cached), and the
/// records verify only against their own key: naming the other key for a record is
/// rejected, cached or not.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Flock and Akita proofs: run with --release"
)]
fn keys_sharing_rho_verify_only_against_their_own_t1() {
    on_large_stack(|| {
        let (victim_sk, victim) = gen::keypair(&gen::key_seed(0x5a, 0));
        let rho: [u8; 32] = victim[..32].try_into().unwrap();
        let (other_sk, other) = gen::keypair_with_rho(&gen::key_seed(0x5a, 1), &rho);
        let mut records = gen::records(0x5b, 4);
        for i in 0..8_u64 {
            let (sk, pk) = if i % 2 == 0 { (&victim_sk, &victim) } else { (&other_sk, &other) };
            let msg = gen::message(0x5c, i);
            let sig = gen::sign(sk, &msg, None);
            records.push(Record { pk: pk.clone(), msg, sig });
        }
        let aggregate = Aggregator::new(AggregatorConfig::default())
            .aggregate(&signed(&records))
            .expect("aggregate");
        let public = statements(&records);
        let verifier = AggregateVerifier::new();
        let uncached = verifier.verify(&public, &aggregate.bytes, None);
        assert!(uncached.accepted(), "{uncached}");
        assert_eq!(uncached.report.outcome.as_ref().unwrap().expanded_keys, 6);
        let mut victim_only = KeyCache::default();
        victim_only.insert(std::sync::Arc::new(
            agg_piop::public_data::VerifierKey::from_bytes(&victim).unwrap(),
        ));
        let partly = verifier.verify(&public, &aggregate.bytes, Some(&mut victim_only));
        assert!(partly.accepted(), "{partly}");
        assert_eq!(partly.report.outcome.as_ref().unwrap().expanded_keys, 5);
        let mut full = victim_only.clone();
        let warm = verifier.verify(&public, &aggregate.bytes, Some(&mut full));
        assert!(warm.accepted());
        assert_eq!(full.len(), 6);
        // Record 5 is signed under `other`, record 4 under `victim`.
        for (record, wrong) in [(5, &victim), (4, &other)] {
            let mut swapped = public.clone();
            swapped[record].0 = wrong.as_slice();
            for cache in [None, Some(full.clone())] {
                let mut cache = cache;
                let report = verifier.verify(&swapped, &aggregate.bytes, cache.as_mut());
                assert!(!report.accepted(), "record {record} under the other key: {report}");
            }
        }
    });
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Flock and Akita proofs: run with --release"
)]
#[allow(clippy::too_many_lines)]
fn corrupted_aggregates_are_rejected() {
    on_large_stack(|| {
        // 20 records (d_seg = 5) in Flock parts of 16 (16 and 4 live records): the most parts
        // a segment may hold (two, `d_part >= d_seg - 1`).
        let records = records(0xc020, 14, 6);
        let aggregate = Aggregator::new(AggregatorConfig::with_part_log2(3))
            .aggregate(&signed(&records))
            .expect("aggregate");
        let honest = aggregate.proof.clone();
        let bytes = aggregate.bytes.clone();
        let public = statements(&records);
        let verifier = AggregateVerifier::new();
        let honest_report = verifier.verify(&public, &bytes, None);
        assert!(honest_report.accepted(), "{honest_report}");
        assert_eq!(honest_report.parts.len(), 2);
        let verify = |proof: &AggregateProof| verifier.verify(&public, &proof.to_bytes(), None);
        let part = |k: usize| move |r: &VerifyCostReport| matches!(r.rejection, Some((0, SegmentRejection::Part(p, _))) if p == k);

        // The c~ list: bound by the record digest and the Flock statement.
        let mut p = honest.clone();
        p.c_tilde[3][0] ^= 1;
        expect_rejected("c~ of record 3", &verify(&p), keccak);

        // The final vector and the shares.
        let mut p = honest.clone();
        p.final_vectors[0][100] ^= 4;
        let r = verify(&p);
        expect_rejected("Y_sigma bit", &r, keccak);
        assert!(part(1)(&r), "the implied last share is wrong");
        let p = with_all_items(&honest, |_, s, _| s.shares[0][3] ^= 1);
        let r = verify(&p);
        expect_rejected("share of part 0", &r, keccak);
        assert!(part(0)(&r));

        // The part proofs and commitments.
        let p = with_items(&honest, |_, p| p.proofs.swap(0, 1));
        expect_rejected("proofs of parts 0 and 1 swapped", &verify(&p), keccak);
        let p = with_items(&honest, |_, p| {
            let mid = p.proofs[1].len() / 2;
            p.proofs[1][mid] ^= 0x10;
        });
        let r = verify(&p);
        expect_rejected("byte of part 1's proof", &r, keccak);
        assert!(part(1)(&r));
        let p = with_items(&honest, |c, _| c.commitments[1][7] ^= 0x20);
        expect_rejected("byte of part 1's commitment", &verify(&p), keccak);

        // The segment items' structure. The profile byte: another mode-A profile makes the
        // verifier run under it, against the profile event 0 bound; an unknown id is refused.
        let p = with_items(&honest, |c, _| c.profile = other(FLOCK_PROFILE) as u8);
        expect_rejected("profile byte of the item", &verify(&p), |_| true);
        let p = with_items(&honest, |c, _| c.profile = FlockProfile::PublicBits as u8);
        let r = verify(&p);
        expect_rejected("profile byte 3 (mode B)", &r, keccak);
        assert!(matches!(
            r.rejection,
            Some((0, SegmentRejection::Item(ItemError::Profile(3))))
        ));
        let p = with_items(&honest, |c, _| c.d_part = 5);
        let r = verify(&p);
        expect_rejected("part size 2^5", &r, keccak);
        assert!(matches!(
            r.rejection,
            Some((
                0,
                SegmentRejection::Item(ItemError::Parts {
                    expected: 1,
                    actual: 2
                })
            ))
        ));
        // An inadmissible part size (below 2^2, or below d_seg - 1 = 4: more than two parts)
        // is refused while the plan of event 0 is derived.
        let p = with_items(&honest, |c, _| c.d_part = 2);
        expect_rejected("part size 2^2", &verify(&p), |e| matches!(e, VerifyError::Plan));
        let p = with_items(&honest, |c, _| c.d_part = 3);
        expect_rejected("part size 2^3", &verify(&p), |e| matches!(e, VerifyError::Plan));
        let p = with_all_items(&honest, |c, s, p| {
            c.commitments.pop();
            p.proofs.pop();
            s.shares.pop();
        });
        let r = verify(&p);
        expect_rejected("last part dropped", &r, keccak);
        assert!(matches!(
            r.rejection,
            Some((
                0,
                SegmentRejection::Item(ItemError::Parts {
                    expected: 2,
                    actual: 1
                })
            ))
        ));
        let mut p = honest.clone();
        p.flock_proofs[0].truncate(100);
        let r = verify(&p);
        expect_rejected("truncated proof item", &r, keccak);
        assert!(matches!(
            r.rejection,
            Some((0, SegmentRejection::Item(ItemError::Truncated)))
        ));

        // The tie work and binding (tied-part proof version 5, docs/soundness.md 3.6, 4.5,
        // 6.3): part 1's tie nonce one below the honest (smallest passing) one, or of 17 bits
        // (Slim at d = 4 with 4 records: nu_s = 3, tie numerator 50 x 6 = 300, 9 bits of work,
        // a nonce of at most 16 bits), fails the predicate; a version-4 proof is refused.
        let nonce_of = |bytes: &[u8]| {
            u32::from_le_bytes(bytes[tied::NONCE_OFFSET..tied::NONCE_OFFSET + 4].try_into().unwrap())
        };
        let set_nonce = |v: u32| {
            with_items(&honest, move |_, p| {
                p.proofs[1][tied::NONCE_OFFSET..tied::NONCE_OFFSET + 4].copy_from_slice(&v.to_le_bytes());
            })
        };
        let honest_nonce = {
            let c = SegmentCommitments::from_bytes(&honest.flock_commitments[0]).unwrap();
            let p = SegmentProofs::from_bytes(&honest.flock_proofs[0], c.commitments.len()).unwrap();
            assert_eq!(p.proofs[1][6], tied::VERSION);
            nonce_of(&p.proofs[1])
        };
        assert!(honest_nonce < 1 << 16);
        let tie_rejected = |r: &VerifyCostReport| {
            matches!(&r.rejection, Some((0, SegmentRejection::Part(1, why))) if why.contains("tie nonce"))
        };
        if honest_nonce > 0 {
            let r = verify(&set_nonce(honest_nonce - 1));
            expect_rejected("tie nonce below the honest one", &r, keccak);
            assert!(tie_rejected(&r));
        }
        let r = verify(&set_nonce(honest_nonce + (1 << 16)));
        expect_rejected("tie nonce of 17 bits", &r, keccak);
        assert!(tie_rejected(&r));
        let r = verify(&with_items(&honest, |_, p| p.proofs[1][6] = 4));
        expect_rejected("tied-part proof version 4", &r, keccak);
        assert!(matches!(&r.rejection, Some((0, SegmentRejection::Part(1, why))) if why.contains("version")));
        // The PIOP work nonces: a changed byte of the stream is rejected.
        let mut p = honest.clone();
        p.work[0] ^= 0x01;
        expect_rejected("PIOP work nonce", &verify(&p), |_| true);
        // Event 0 binds the plan: a verifier whose plan names other tie difficulties derives
        // another transcript (here sd, so the parity chain's matrices differ).
        let gate = sponge_gate();
        let other_plan = verify_timed(
            &public,
            &bytes,
            &SeededMatrices,
            &OtherPlan(agg_cli::mode_a::FlockVerifier::new(&gate)),
            FLOCK_PROFILE,
            Relation::Hu,
            None,
        );
        assert!(matches!(other_plan.outcome, Err(VerifyError::Keccak(0))));

        // PIOP and opening.
        let mut p = honest.clone();
        p.prestage.kappa3 = p.prestage.kappa2[0];
        expect_rejected("pre-stage claim", &verify(&p), at_stage_or_work(1));
        let mut p = honest.clone();
        p.lookup.sessions[4].rounds[2][0] = p.lookup.sessions[4].rounds[2][1];
        expect_rejected("lookup round", &verify(&p), |e| matches!(e, VerifyError::Lookup(_)));
        let mut p = honest.clone();
        p.stage1.rounds[2][0] = p.stage1.rounds[2][1];
        expect_rejected("stage-1 round", &verify(&p), at_stage_or_work(1));
        let mut p = honest.clone();
        p.stage2.rounds[3][1] = p.stage2.rounds[3][0];
        expect_rejected("stage-2 round", &verify(&p), at_stage_or_work(2));
        let mut p = honest.clone();
        p.stage3.rounds[0][0] = p.stage3.rounds[0][1];
        expect_rejected("stage-3 round", &verify(&p), at_stage_or_work(3));
        let mut p = honest.clone();
        p.stage3.claims.0[20] = p.stage3.claims.0[21];
        expect_rejected("stage-3 claim", &verify(&p), at_stage(3));
        let mut p = honest.clone();
        let last = p.opening.len() - 1;
        p.opening[last] ^= 0x20;
        expect_rejected("opening byte", &verify(&p), |e| {
            matches!(e, VerifyError::Opening(_))
        });
        // Commitments: a changed G_main byte, the round-2 commitments in another order.
        let mut p = honest.clone();
        p.main_commitment[40] ^= 1;
        expect_rejected("G_main commitment byte", &verify(&p), |_| true);
        let mut p = honest.clone();
        p.bridge_commitments.swap(0, 1);
        expect_rejected("round-2 commitments reordered", &verify(&p), |_| true);

        // The records.
        let mut swapped = public.clone();
        swapped.swap(4, 5);
        expect_rejected(
            "records 4 and 5 swapped",
            &verifier.verify(&swapped, &bytes, None),
            |_| true,
        );
        let other_msg = [records[7].msg.as_slice(), b"!"].concat();
        let mut wrong_msg = public.clone();
        wrong_msg[7].1 = &other_msg;
        expect_rejected(
            "message of record 7",
            &verifier.verify(&wrong_msg, &bytes, None),
            keccak,
        );
        let mut wrong_key = public.clone();
        wrong_key[0].0 = public[1].0;
        assert_ne!(public[0].0, public[1].0);
        expect_rejected(
            "key of record 0",
            &verifier.verify(&wrong_key, &bytes, None),
            |_| true,
        );
        expect_rejected(
            "one record missing",
            &verifier.verify(&public[1..], &bytes, None),
            |e| matches!(e, VerifyError::RecordCount),
        );

        // Framing.
        expect_rejected(
            "truncated aggregate",
            &verifier.verify(&public, &bytes[..bytes.len() - 1], None),
            |e| matches!(e, VerifyError::Wire(_)),
        );
        let mut trailing = bytes.clone();
        trailing.push(0);
        expect_rejected(
            "trailing byte",
            &verifier.verify(&public, &trailing, None),
            |e| matches!(e, VerifyError::Wire(_)),
        );
        let mut version = bytes.clone();
        version[0] ^= 1;
        expect_rejected(
            "wire version",
            &verifier.verify(&public, &version, None),
            |e| matches!(e, VerifyError::Wire(_)),
        );

        // Configuration: the Flock side under another profile is refused before parsing;
        // the mode-B checks under profile 3 reject the Flock items.
        let gate = sponge_gate();
        let flock = agg_cli::mode_a::FlockVerifier::new(&gate);
        let mismatch = verify_timed(
            &public,
            &bytes,
            &SeededMatrices,
            &flock,
            FlockProfile::None,
            Relation::Hu,
            None,
        );
        assert!(matches!(
            mismatch.outcome,
            Err(VerifyError::Configuration(
                ConfigurationError::ProfileMismatch { .. }
            ))
        ));
        let mode_b = verify_timed(
            &public,
            &bytes,
            &SeededMatrices,
            &PublicBitsVerifier,
            FlockProfile::PublicBits,
            Relation::Hu,
            None,
        );
        assert!(matches!(mode_b.outcome, Err(VerifyError::Keccak(0))));
        // The honest aggregate still verifies after all this.
        assert!(verifier.verify(&public, &bytes, None).accepted());
    });
}

/// More than two tied parts in a segment (docs/soundness.md 3.5): an honest prover that
/// splits 20 records (d_seg = 5) into parts of 8 (three parts) builds an aggregate that every
/// verifier refuses, because `sd`'s bad set would be a union over three Johnson lists.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Flock and Akita proofs: run with --release"
)]
fn three_parts_in_a_segment_are_refused() {
    on_large_stack(|| {
        let records = records(0xc320, 14, 6);
        let hooks = AggregateHooks {
            part_floor: Some(3),
            ..AggregateHooks::default()
        };
        let aggregate = Aggregator::new(AggregatorConfig::with_part_log2(3))
            .aggregate_with(&signed(&records), &hooks)
            .expect("aggregate");
        let items = SegmentCommitments::from_bytes(&aggregate.proof.flock_commitments[0]).unwrap();
        assert_eq!((items.d_part, items.commitments.len()), (3, 3));
        let report = AggregateVerifier::new().verify(&statements(&records), &aggregate.bytes, None);
        expect_rejected("three parts in segment 0", &report, |e| matches!(e, VerifyError::Plan));
        assert!(report.parts.is_empty(), "no Flock part was run");
    });
}

/// Mutable access to one `G_main4` cell.
fn cell<'a>(
    manifest: &CommitmentManifest,
    main: &'a mut MainTables,
    column: Column,
    x: usize,
) -> &'a mut i8 {
    let (role, position) = manifest.locate(column);
    assert_eq!(role, GroupRole::Main);
    &mut main.columns_mut()[position][x]
}

/// Bit `1536 rho + 6 j + t` of a `w1Encode` string (FIPS 204 Algorithms 16, 28).
const fn stream_bit(t: usize, j: usize, rho: usize) -> usize {
    1536 * rho + 6 * j + t
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Flock and Akita proofs: run with --release"
)]
#[allow(clippy::too_many_lines)]
fn tampered_witnesses_are_rejected_end_to_end() {
    on_large_stack(|| {
        // 12 live records (N* = 16, one Flock part) keep the 17 prover runs short.
        let records = records(0x7a12, 9, 3);
        let signed = signed(&records);
        let public = statements(&records);
        let aggregator = Aggregator::new(AggregatorConfig::default());
        let verifier = AggregateVerifier::new();
        let gate = sponge_gate();
        let run =
            |label: &str, hooks: &AggregateHooks<'_>, check: &dyn Fn(&VerifyError) -> bool| {
                match aggregator.aggregate_with(&signed, hooks) {
                    Ok(aggregate) => {
                        let report = verifier.verify(&public, &aggregate.bytes, None);
                        expect_rejected(label, &report, check);
                        aggregate.bytes
                    }
                    Err(e) => panic!("{label}: the tampered prover failed: {e}"),
                }
            };
        let honest = aggregator.aggregate(&signed).expect("honest");
        assert!(verifier.verify(&public, &honest.bytes, None).accepted());

        // The tampered position: b_t(j = 17, rho = 2) of record 5 (main_index(17, 2, 5)).
        let (i, j, rho) = (5, 17, 2);
        let main =
            |f: &'static (dyn Fn(&CommitmentManifest, &mut MainTables) + Sync)| AggregateHooks {
                tamper_main: Some(f),
                ..AggregateHooks::default()
            };

        // A 16-bit z or v column changed inside T_16 (the round-2 multiplicities are
        // recounted): the ring identity (stage 3).
        run(
            "z_lo",
            &main(&|m, t| {
                let c = t.wide_cell_mut(m, Column::Z_LO, main_index(17, 2, 5));
                *c = if *c < (1 << 15) - 1 { *c + 1 } else { *c - 1 };
            }),
            &at_stage(3),
        );
        run(
            "v_lo",
            &main(&|m, t| {
                let c = t.wide_cell_mut(m, Column::V_LO, main_index(17, 2, 5));
                *c = if *c < (1 << 15) - 1 { *c + 1 } else { *c - 1 };
            }),
            &at_stage(3),
        );
        // A committed bit flipped, the Flock side honest: the F_p side's Y_sigma (of the
        // flipped bits) differs from the Flock side's M b' (Lemma 1), so the Flock part fails.
        run(
            "committed bit flipped",
            &main(&|m, t| *cell(m, t, Column::W1Bit(0), main_index(17, 2, 5)) ^= 1),
            &keccak,
        );
        // The same flip with the round-2 tables (and Y_sigma) of the honest bits: the Flock
        // part accepts and the F_p side fails at the stage-1 terminal (joint bridge test).
        let flipped = |m: &CommitmentManifest, t: &mut MainTables| {
            *cell(m, t, Column::W1Bit(0), main_index(17, 2, 5)) ^= 1;
        };
        let bytes = run(
            "committed bit flipped, stale round 2",
            &AggregateHooks {
                tamper_main: Some(&flipped),
                stale_bridge: true,
                ..AggregateHooks::default()
            },
            &at_stage(1),
        );
        let report = verifier.verify(&public, &bytes, None);
        assert!(
            report.parts.iter().all(|p| p.accepted),
            "the Flock part accepts"
        );
        // A non-Boolean bit (0 -> 2 keeps every parity): Booleanity in stage 1.
        run(
            "bit = 2",
            &main(&|m, t| {
                let y = (0..1024)
                    .map(|p| main_index(p % 256, p / 256, 7))
                    .find(|&y| *cell(m, t, Column::W1Bit(1), y) == 0)
                    .expect("a zero bit");
                *cell(m, t, Column::W1Bit(1), y) = 2;
            }),
            &at_stage(1),
        );
        // w1' = 44 in the committed bits only: the bridge (Flock part) objects.
        let w44 = |m: &CommitmentManifest, t: &mut MainTables| {
            for (b, bit) in [0, 0, 1, 1, 0, 1].into_iter().enumerate() {
                *cell(m, t, Column::W1Bit(b as u8), main_index(17, 2, 5)) = bit;
            }
        };
        run(
            "w1' = 44, committed bits only",
            &AggregateHooks {
                tamper_main: Some(&w44),
                ..AggregateHooks::default()
            },
            &keccak,
        );
        // w1' = 44 in both copies: the sponge hashes to another c~ (Flock part); with the
        // Flock verdict ignored the F_p side's range identity fails in stage 1.
        let bytes = run(
            "w1' = 44, both copies",
            &AggregateHooks {
                tamper_main: Some(&w44),
                flock_from_tampered: true,
                ..AggregateHooks::default()
            },
            &keccak,
        );
        let lenient = Lenient(agg_cli::mode_a::FlockVerifier::new(&gate));
        let outcome = verify_timed(
            &public,
            &bytes,
            &SeededMatrices,
            &lenient,
            FLOCK_PROFILE,
            Relation::Hu,
            None,
        );
        let error = outcome.outcome.expect_err("the range identity");
        eprintln!("w1' = 44, both copies, Flock verdict ignored: rejected with {error}");
        assert!(at_stage(1)(&error), "{error:?}");
        // A 16-bit value outside T_16 (stored 2^15): Akita refuses to commit it (the
        // width-16 coefficient bound is the balanced 16-bit interval), so no aggregate
        // exists to verify; the verifier's rejection of a looked-up value outside T_16 is
        // tested at the PIOP level (agg-piop tests/stages.rs, without commitments).
        let Err(error) = aggregator.aggregate_with(
            &signed,
            &main(&|m, t| *t.wide_cell_mut(m, Column::U_LO, main_index(17, 2, 5)) = 1 << 15),
        ) else {
            panic!("Akita refuses a stored value outside the width-16 interval");
        };
        eprintln!("u_lo = 2^15: the prover fails ({error})");
        assert!(
            matches!(error, AggregateError::Prove(ProveError::Pcs(_)))
                && error.to_string().contains("exceeds the scheduled coefficient bound"),
            "{error}"
        );
        // Base-4 and base-6 digits outside their alphabets: the leaves of stage 2 (U_hi = -4
        // is in relaxed H's base-8 alphabet, not in relation H-u's base-6 one).
        run(
            "base-4 digit",
            &main(&|m, t| *cell(m, t, Column::Z_HI, main_index(17, 2, 5)) = 3),
            &at_stage(2),
        );
        run(
            "base-6 digit",
            &main(&|m, t| *cell(m, t, Column::U_HI, main_index(17, 2, 5)) = -4),
            &at_stage(2),
        );

        // Round-2 tampering: a level-1 carry, a level-2 parity, a final-vector bit.
        let bridge = |f: &'static (dyn Fn(&CommitmentManifest, &mut BridgeGroupTables) + Sync)| {
            AggregateHooks {
                tamper_bridge: Some(f),
                ..AggregateHooks::default()
            }
        };
        run(
            "level-1 carry",
            &bridge(&|_, b| {
                let carries = b.carries_mut(1);
                let old = carries[3];
                let new = if old < (1 << 15) - 1 { old + 1 } else { old - 1 };
                carries[3] = new;
                // The carry stays in T_16 and the multiplicities follow it, so only the
                // bridge identities object.
                let mut m = b.multiplicity().to_vec();
                m[(old + (1 << 15)) as usize] -= 1;
                m[(new + (1 << 15)) as usize] += 1;
                b.set_multiplicity(m);
            }),
            &at_stage(1),
        );
        run(
            "level-2 parity",
            &bridge(&|_, b| b.parities_mut()[2][5] ^= 1),
            &at_stage(1),
        );
        run(
            "final-vector bit",
            &bridge(&|_, b| b.public_y_mut()[11] ^= 1),
            &keccak,
        );

        // Flock side (agg-keccak tests): the record's bit flipped in both Flock copies with
        // the published c~; in the bridge slot copy only; in the sponge input only with the
        // stale chi witness.
        let q = stream_bit(0, j, rho);
        let flip_record =
            move |recs: &mut Vec<SpongeRecord>| recs[i].w1_encode[q / 8] ^= 1 << (q % 8);
        run(
            "Flock records: bit flipped in both copies",
            &AggregateHooks {
                tamper_flock_records: Some(&flip_record),
                ..AggregateHooks::default()
            },
            &keccak,
        );
        let slot_only = move |seg: &agg_keccak::tied::TiedPart<'_>,
                              _: usize,
                              _: usize,
                              recs: &[SpongeRecord]| {
            let (sponge, mut z) = seg.honest_parts(recs);
            let c = seg.segment().column(0, j + 256 * rho + 1024 * i);
            z[c / 64] ^= 1 << (c % 64);
            (sponge, z)
        };
        run(
            "Flock: bit flipped in the parity chain's copy only",
            &AggregateHooks {
                flock_witness: Some(&slot_only),
                ..AggregateHooks::default()
            },
            &keccak,
        );
        let gate_ref = &gate;
        let sponge_only = move |seg: &agg_keccak::tied::TiedPart<'_>,
                                _: usize,
                                _: usize,
                                recs: &[SpongeRecord]| {
            let (_, z) = seg.honest_parts(recs);
            let inputs: Vec<Vec<u64>> = recs.iter().map(SpongeRecord::input_bits).collect();
            let col = gate_ref.layout().input_col(W1_INPUT_BASE + 4000);
            let sponge = pack_slot(gate_ref, seg.nu(), recs.len(), |r, zz, a, b| {
                gate_ref.honest_z(&inputs[r], zz);
                if r == i {
                    zz[col / 64] ^= 1 << (col % 64);
                }
                gate_ref.apply_ab(zz, a, b);
            });
            (sponge, z)
        };
        run(
            "Flock: sponge input bit flipped, stale chi witness",
            &AggregateHooks {
                flock_witness: Some(&sponge_only),
                ..AggregateHooks::default()
            },
            &keccak,
        );
    });
}

/// The aggregator verifies every signature strictly while deriving its witness
/// (`agg-witness/tests/negative.rs`): flipped `c~`, `z` and hint bits, a wrong message, and
/// malformed lengths are refused with the record's index; nothing is proven.
#[test]
fn invalid_signatures_are_refused_by_the_aggregator() {
    on_large_stack(|| {
        let records = records(0x5160, 6, 2);
        let aggregator = Aggregator::new(AggregatorConfig::default());
        let refuse = |label: &str, k: usize, f: &dyn Fn(&mut Record)| {
            let mut bad = records.clone();
            f(&mut bad[k]);
            match aggregator.aggregate(&signed(&bad)) {
                Err(e @ (AggregateError::Signature(i, _) | AggregateError::Malformed(i, _))) => {
                    assert_eq!(i, k, "{label}: {e}");
                    eprintln!("{label}: refused ({e})");
                }
                Err(e) => panic!("{label}: unexpected error {e}"),
                Ok(_) => panic!("{label}: an invalid signature was aggregated"),
            }
        };
        refuse("c~ bit", 3, &|r| r.sig[17] ^= 8);
        let z_bit = 32 * 8 + 100 * Z_BITS + 16;
        refuse("z bit", 5, &|r| r.sig[z_bit / 8] ^= 1 << (z_bit % 8));
        refuse("hint byte", 1, &|r| {
            r.sig[SIG_BYTES - OMEGA - K + 3] ^= 0x40
        });
        refuse("message", 6, &|r| r.msg.push(0));
        refuse("signature length", 2, &|r| {
            r.sig.pop();
        });
        refuse("public key length", 0, &|r| {
            r.pk.push(0);
        });
        assert!(matches!(
            aggregator.aggregate(&[]),
            Err(AggregateError::Count(0))
        ));
    });
}
