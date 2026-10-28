//! Mode B end to end: `prove_mode_b` / `verify_mode_b` round trips at 64, 256 and 1,024
//! signatures with the seed-derived matrices (`SeededMatrices`), the proof size and the
//! verifier's step timings printed, corrupted aggregates rejected, and every witness
//! tampering of `tests/stages.rs` pushed through the full prover (unchecked stages) and
//! rejected by the verifier.
#![allow(clippy::needless_range_loop)]

mod common;

use agg_piop::binding::FlockProfile;
use agg_piop::manifest::{Column, CommitmentManifest, GroupRole, Relation, LOOKUP_SHIFT, U16_DIGITS};
use agg_piop::mode_b::{prove_mode_b, public_bits, verify_mode_b, PublicBits, PublicBitsVerifier};
use agg_piop::prover::{
    main_tables, prove_with_tables, BridgeDerivation, BridgeTamper, FixedMatrices, ProveHooks,
    ProverInputs, ProverOutput, SeededMatrices,
};
use agg_piop::pow::{KeccakPlan, PowPlan, SiteClass, WorkStreamError};
use agg_piop::public_data::KeyCache;
use agg_piop::source_tables::{BridgeGroupTables, MainTables};
use agg_piop::transcript::TranscriptError;
use agg_piop::stage::StageError;
use agg_piop::verifier::{
    verify_timed, BitsContext, ConfigurationError, KeccakVerifier, SegmentContext, VerifyError,
    VerifyReport,
};
use agg_bridge::seed::derive_matrices;
use agg_witness::bridge::{BridgeMatrices, J1, K1};
use agg_witness::gen::Record;
use agg_witness::instance::PublicInstance;
use agg_witness::native::commitment_hash;
use agg_witness::padding::{main_index, Geometry};
use agg_witness::params::{C_TILDE_BYTES, GAMMA2};

fn on_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("test thread");
}

/// The honest inputs of a batch and the relation to prove.
struct Fixture {
    records: Vec<Record>,
    instance: PublicInstance,
    relation: Relation,
}

impl Fixture {
    fn new(seed: u64, own: usize, shared: usize) -> Self {
        Self::with_relation(Relation::Hu, seed, own, shared)
    }

    fn with_relation(relation: Relation, seed: u64, own: usize, shared: usize) -> Self {
        let records = common::records(seed, own, shared);
        let instance = PublicInstance::from_records(&records).expect("instance");
        Self {
            records,
            instance,
            relation,
        }
    }

    fn inputs(&self) -> Vec<(&[u8], &[u8], [u8; C_TILDE_BYTES])> {
        self.records
            .iter()
            .map(|r| {
                let mut ct = [0_u8; C_TILDE_BYTES];
                ct.copy_from_slice(&r.sig[..C_TILDE_BYTES]);
                (r.pk.as_slice(), r.msg.as_slice(), ct)
            })
            .collect()
    }

    fn signatures(&self) -> Vec<&[u8]> {
        self.records.iter().map(|r| r.sig.as_slice()).collect()
    }

    fn public(&self) -> Vec<(&[u8], &[u8])> {
        self.records
            .iter()
            .map(|r| (r.pk.as_slice(), r.msg.as_slice()))
            .collect()
    }

    fn prove(&self, bridge: &impl BridgeDerivation) -> ProverOutput {
        let inputs = self.inputs();
        let signatures = self.signatures();
        prove_mode_b(
            &ProverInputs {
                instance: &self.instance,
                records: &inputs,
                signatures: &signatures,
                relation: self.relation,
            },
            bridge,
        )
        .expect("prove")
    }
}

fn report_line(label: &str, report: &VerifyReport) {
    eprintln!(
        "{label}: accepted {}, {}",
        report.accepted(),
        report.timings
    );
}

/// An honest round trip with the matrices derived from `sd`: proof size,
/// verifier timings uncached and cached, and the public bits satisfying the hash relation.
fn round_trip(
    n_own: usize,
    n_shared: usize,
    seed: u64,
) -> (Fixture, SeededMatrices, ProverOutput, Vec<u8>) {
    round_trip_as(Relation::Hu, n_own, n_shared, seed)
}

/// [`round_trip`] under `relation`.
fn round_trip_as(
    relation: Relation,
    n_own: usize,
    n_shared: usize,
    seed: u64,
) -> (Fixture, SeededMatrices, ProverOutput, Vec<u8>) {
    let fixture = Fixture::with_relation(relation, seed, n_own, n_shared);
    let n = n_own + n_shared;
    let bridge = SeededMatrices;
    let output = fixture.prove(&bridge);
    let bytes = output.proof.to_bytes();
    let size = output.proof.size();
    assert_eq!(size.total(), bytes.len());
    eprintln!(
        "N = {n}, {relation}: aggregate {} bytes = header {} + c~ {} + public bits {} + Akita \
         commitments {} + final vectors {} + Keccak proofs {} + lookup {} + PIOP {} + opening {}",
        bytes.len(),
        size.header,
        size.c_tilde,
        size.keccak_commitments,
        size.akita_commitments,
        size.final_vectors,
        size.keccak_proofs,
        size.lookup,
        size.piop,
        size.opening
    );
    eprintln!("N = {n}: prover {}", output.timings);
    for i in [0, n / 2, n - 1] {
        let bits = public_bits(&output.proof, i).expect("record bits");
        let record = &fixture.instance.records[i];
        assert_eq!(
            commitment_hash(&record.mu, bits.try_into().unwrap()),
            record.c_tilde
        );
    }
    assert!(public_bits(&output.proof, n).is_none());

    let public = fixture.public();
    assert_eq!(output.proof.relation, relation);
    assert_eq!(bytes[2], relation.id());
    let uncached = verify_mode_b(&public, &bytes, &bridge, relation, None);
    report_line(&format!("N = {n}: verifier uncached"), &uncached);
    let verified = uncached.outcome.expect("verify");
    assert_eq!(verified.records, n);
    assert_eq!(verified.bridge_seed, output.bridge_seed);
    assert_eq!(verified.session, output.session);
    let mut cache = KeyCache::default();
    let first = verify_mode_b(&public, &bytes, &bridge, relation, Some(&mut cache));
    assert_eq!(
        first
            .outcome
            .expect("verify, filling the cache")
            .expanded_keys,
        verified.expanded_keys
    );
    let cached = verify_mode_b(&public, &bytes, &bridge, relation, Some(&mut cache));
    report_line(&format!("N = {n}: verifier cached keys"), &cached);
    assert_eq!(
        cached
            .outcome
            .expect("verify with cached keys")
            .expanded_keys,
        0
    );
    drop(public);
    (fixture, bridge, output, bytes)
}

fn rejected(report: VerifyReport) -> VerifyError {
    report.outcome.expect_err("the verifier must reject")
}

#[test]
fn mode_b_round_trip_at_256_signatures_and_corrupted_aggregates_are_rejected() {
    on_large_stack(|| {
        let (fixture, bridge, output, bytes) = round_trip(200, 56, 0xb256);
        let public = fixture.public();
        let verify = |proof: &agg_piop::wire::AggregateProof| {
            verify_mode_b(&public, &proof.to_bytes(), &bridge, Relation::Hu, None)
        };

        // The public bits: a flipped bit breaks the hash, a value of 44 or more the range.
        let mut flipped = output.proof.clone();
        flipped.flock_commitments[0][5 * 768 + 300] ^= 0x10;
        assert!(matches!(rejected(verify(&flipped)), VerifyError::Keccak(0)));
        let mut wide = output.proof.clone();
        wide.flock_commitments[0][9 * 768] |= 0x3f;
        assert!(matches!(rejected(verify(&wide)), VerifyError::Keccak(0)));
        let mut short = output.proof.clone();
        short.flock_commitments[0].truncate(768 * 255);
        assert!(matches!(rejected(verify(&short)), VerifyError::Keccak(0)));
        // A changed c~ fails the hash; a changed final vector fails the projection.
        let mut c_tilde = output.proof.clone();
        c_tilde.c_tilde[17][3] ^= 1;
        assert!(matches!(rejected(verify(&c_tilde)), VerifyError::Keccak(0)));
        let mut final_vector = output.proof.clone();
        final_vector.final_vectors[0][100] ^= 4;
        assert!(matches!(
            rejected(verify(&final_vector)),
            VerifyError::Keccak(0)
        ));
        let mut keccak_proof = output.proof.clone();
        keccak_proof.flock_proofs[0] = vec![0];
        assert!(matches!(
            rejected(verify(&keccak_proof)),
            VerifyError::Keccak(0)
        ));

        // PIOP and opening corruptions.
        let mut lookup = output.proof.clone();
        lookup.lookup.sessions[6].rounds[3][2] = lookup.lookup.sessions[6].rounds[3][0];
        assert!(matches!(rejected(verify(&lookup)), VerifyError::Lookup(_)));
        // A changed absorbed message changes the next challenge's work predicate, so the
        // verifier may reject at the next nonce instead of the terminal check.
        let mut prestage = output.proof.clone();
        prestage.prestage.kappa3 = prestage.prestage.kappa2[0];
        assert!(matches!(
            rejected(verify(&prestage)),
            VerifyError::Stage(StageError::Terminal(1)) | VerifyError::Stage(StageError::Transcript(TranscriptError::Work { .. }))
        ));
        let mut round = output.proof.clone();
        round.stage2.rounds[3][1] = round.stage2.rounds[3][0];
        assert!(matches!(
            rejected(verify(&round)),
            VerifyError::Stage(StageError::Terminal(2)) | VerifyError::Stage(StageError::Transcript(TranscriptError::Work { .. }))
        ));
        let mut stage3 = output.proof.clone();
        stage3.stage3.rounds[0][0] = stage3.stage3.rounds[0][1];
        assert!(matches!(
            rejected(verify(&stage3)),
            VerifyError::Stage(StageError::Terminal(3)) | VerifyError::Stage(StageError::Transcript(TranscriptError::Work { .. }))
        ));
        let mut claim = output.proof.clone();
        claim.stage3.claims.0[20] = claim.stage3.claims.0[21];
        assert!(matches!(
            rejected(verify(&claim)),
            VerifyError::Stage(StageError::Terminal(3)) | VerifyError::Stage(StageError::Transcript(TranscriptError::Work { .. }))
        ));
        let mut opening = output.proof.clone();
        let last = opening.opening.len() - 1;
        opening.opening[last] ^= 0x20;
        assert!(matches!(
            rejected(verify(&opening)),
            VerifyError::Opening(_)
        ));

        // Work nonces (`docs/soundness.md`): a changed nonce at one site of
        // every class is rejected (the prover's nonce is the smallest passing one, so a
        // smaller one fails the predicate; from 0 the next one changes the challenge).
        let manifest = CommitmentManifest::for_batch(256).unwrap();
        let plan = PowPlan::new(&manifest, FlockProfile::PublicBits, &KeccakPlan::default());
        assert_eq!(output.proof.work.len(), plan.work_bits().div_ceil(8));
        assert_eq!(output.proof.size().work, 333, "2,660 bits at N* = 2^8");
        let mut offset = 0;
        let mut seen = std::collections::HashSet::new();
        for site in plan.sites() {
            let width = usize::from(site.nonce_bits());
            let first_of_kind = match site.class {
                SiteClass::StageRound | SiteClass::StageGamma => seen.insert((site.class, site.i)),
                class => seen.insert((class, 0)),
            };
            if width > 0 && first_of_kind {
                let read = |bytes: &[u8]| (0..width).fold(0_u32, |v, k| {
                    let at = offset + k;
                    v | (u32::from(bytes[at / 8] >> (at % 8) & 1) << k)
                });
                let honest = read(&output.proof.work);
                let wrong = if honest > 0 { honest - 1 } else { 1 };
                let mut tampered = output.proof.clone();
                for k in 0..width {
                    let at = offset + k;
                    tampered.work[at / 8] &= !(1 << (at % 8));
                    tampered.work[at / 8] |= (((wrong >> k) & 1) as u8) << (at % 8);
                }
                assert_eq!(read(&tampered.work), wrong);
                let err = rejected(verify(&tampered));
                if honest > 0 {
                    // The predicate fails at the site itself.
                    assert!(format!("{err}").contains("work nonce rejected"), "{site:?}: {err}");
                }
            }
            offset += width;
        }
        assert_eq!(offset, plan.work_bits());
        assert_eq!(seen.len(), 4 + 3 + 3 + 1, "every class (X has no nonce)");
        // The stream's padding bits are zero and its length exact.
        let mut padded = output.proof.clone();
        *padded.work.last_mut().unwrap() |= 0x80;
        assert!(matches!(
            rejected(verify(&padded)),
            VerifyError::Transcript(TranscriptError::WorkStream(WorkStreamError::Padding))
        ));
        let mut short = output.proof.clone();
        short.work.pop();
        assert!(matches!(rejected(verify(&short)), VerifyError::Wire(_)));
        let mut long = output.proof.clone();
        long.work.push(0);
        assert!(matches!(rejected(verify(&long)), VerifyError::Wire(_)));
        // Event 0 binds the plan: a verifier whose plan has a tie site derives another sd.
        let other_plan = verify_timed(&public, &bytes, &bridge, &WithTie, FlockProfile::PublicBits, Relation::Hu, None);
        assert!(matches!(rejected(other_plan), VerifyError::Keccak(0)));

        // Records, profile and framing.
        let mut swapped = public.clone();
        swapped.swap(10, 11);
        assert!(!verify_mode_b(&swapped, &bytes, &bridge, Relation::Hu, None).accepted());
        assert!(matches!(
            rejected(verify_mode_b(&public[1..], &bytes, &bridge, Relation::Hu, None)),
            VerifyError::RecordCount
        ));
        // Event 0 binds the profile: under profile 0 the transcript, hence sd and the
        // seed-derived matrices, change, and the projection of the public bits no longer
        // matches Y_sigma.
        let other_profile = verify_timed(
            &public,
            &bytes,
            &bridge,
            &AsProfile(FlockProfile::None),
            FlockProfile::None,
            Relation::Hu,
            None,
        );
        assert!(matches!(rejected(other_profile), VerifyError::Keccak(0)));
        // A Keccak verifier for another profile is refused before the proof is read.
        let mismatch = verify_timed(
            &public,
            &bytes,
            &bridge,
            &PublicBitsVerifier,
            FlockProfile::None,
            Relation::Hu,
            None,
        );
        assert!(matches!(
            rejected(mismatch),
            VerifyError::Configuration(ConfigurationError::ProfileMismatch { .. })
        ));

        // The matrices are those of the transcript's sd: the same matrices as a fixed
        // stand-in verify, the matrices of any other seed (or test matrices) do not.
        let verified_seed = verify_mode_b(&public, &bytes, &bridge, Relation::Hu, None)
            .outcome
            .expect("verify")
            .bridge_seed;
        assert_eq!(verified_seed, output.bridge_seed);
        let same = FixedMatrices(derive_matrices(&output.bridge_seed));
        assert!(verify_mode_b(&public, &bytes, &same, Relation::Hu, None).accepted());
        let mut other_seed = output.bridge_seed;
        other_seed[0] ^= 1;
        let other_seed = FixedMatrices(derive_matrices(&other_seed));
        assert!(matches!(
            rejected(verify_mode_b(&public, &bytes, &other_seed, Relation::Hu, None)),
            VerifyError::Keccak(0)
        ));
        let other = FixedMatrices(BridgeMatrices::random_for_tests(0x99));
        assert!(matches!(
            rejected(verify_mode_b(&public, &bytes, &other, Relation::Hu, None)),
            VerifyError::Keccak(0)
        ));
        assert!(matches!(
            rejected(verify_mode_b(
                &public,
                &bytes[..bytes.len() - 1],
                &bridge,
                Relation::Hu,
                None
            )),
            VerifyError::Wire(_)
        ));
    });
}

#[test]
fn mode_b_round_trip_at_64_signatures() {
    on_large_stack(|| {
        let _ = round_trip(60, 4, 0xb064);
    });
}

/// Relation H-exact (`docs/h-exact.md`, protocol level): honest round trips at
/// 1, 64 and 256 records; the verifier takes the relation from its caller, so an H-exact
/// proof verified as H-u and an H-u proof verified as H-exact are rejected before event 0,
/// a rewritten relation byte fails, and an unknown relation byte is malformed.
#[test]
fn mode_b_h_exact_round_trips_and_the_relation_is_the_callers() {
    on_large_stack(|| {
        let _ = round_trip_as(Relation::HExact, 1, 0, 0xe001);
        let _ = round_trip_as(Relation::HExact, 60, 4, 0xe064);
        let (fixture, bridge, output, bytes) = round_trip_as(Relation::HExact, 200, 56, 0xe256);
        let public = fixture.public();
        assert_eq!(output.proof.stage2.claims.squares.len(), 1);
        assert_eq!(output.proof.stage3.claims.0.len(), 32);
        // 256 records: d_main = 18 rounds of 4 + 4 + 3 elements, 76 claims.
        assert_eq!(output.proof.size().piop, 16 * (76 + 11 * 18));
        let as_hu = verify_mode_b(&public, &bytes, &bridge, Relation::Hu, None);
        assert!(matches!(
            rejected(as_hu),
            VerifyError::Relation {
                expected: Relation::Hu,
                found: Relation::HExact
            }
        ));
        // The H-exact header rewritten to H-u: verified as H-u the transcript (event 0,
        // the catalog row, the plan) differs; verified as H-exact the flag mismatches.
        let mut rewritten = bytes.clone();
        rewritten[2] = Relation::Hu.id();
        assert!(!verify_mode_b(&public, &rewritten, &bridge, Relation::Hu, None).accepted());
        assert!(matches!(
            rejected(verify_mode_b(&public, &rewritten, &bridge, Relation::HExact, None)),
            VerifyError::Relation { .. }
        ));
        // Relation byte 2 names H-FIPS: an H-exact proof rewritten to it is malformed or
        // mismatched, never accepted; byte 3 is unknown.
        let mut fips = bytes.clone();
        fips[2] = Relation::HFips.id();
        assert!(!verify_mode_b(&public, &fips, &bridge, Relation::HExact, None).accepted());
        assert!(!verify_mode_b(&public, &fips, &bridge, Relation::HFips, None).accepted());
        let mut unknown = bytes.clone();
        unknown[2] = 3;
        assert!(matches!(
            rejected(verify_mode_b(&public, &unknown, &bridge, Relation::HExact, None)),
            VerifyError::Wire(_)
        ));
        // An H-u proof of the same records verified as H-exact, and rewritten to H-exact.
        let hu = Fixture::new(0xe256, 200, 56).prove(&bridge);
        let hu_bytes = hu.proof.to_bytes();
        assert!(verify_mode_b(&public, &hu_bytes, &bridge, Relation::Hu, None).accepted());
        assert!(matches!(
            rejected(verify_mode_b(&public, &hu_bytes, &bridge, Relation::HExact, None)),
            VerifyError::Relation {
                expected: Relation::HExact,
                found: Relation::Hu
            }
        ));
        let mut promoted = hu_bytes.clone();
        promoted[2] = Relation::HExact.id();
        assert!(!verify_mode_b(&public, &promoted, &bridge, Relation::HExact, None).accepted());
        // A wrong nonce at the stage-2 gamma site (10 bits under H-exact) is rejected.
        let manifest = CommitmentManifest::for_relation(256, Relation::HExact).unwrap();
        let plan = PowPlan::new(&manifest, FlockProfile::PublicBits, &KeccakPlan::default());
        let mut offset = 0;
        for site in plan.sites() {
            if site.class == SiteClass::StageGamma && site.i == 2 {
                assert_eq!((site.bits, site.nonce_bits()), (3, 10));
                let mut wrong = output.proof.clone();
                wrong.work[offset / 8] ^= 1 << (offset % 8);
                assert!(!verify_mode_b(&public, &wrong.to_bytes(), &bridge, Relation::HExact, None)
                    .accepted());
                break;
            }
            offset += usize::from(site.nonce_bits());
        }
        assert!(offset > 0, "the stage-2 gamma site is in the plan");
    });
}

/// Relation H-FIPS (`docs/h-exact.md`, protocol level): honest round trips at
/// 1, 64 and 256 records (the 256-record batch includes the synthetic weight-80 records with
/// the coefficient `w' = q - gamma2, h = 0`), the claim counts, and cross-relation
/// verification: an H-FIPS proof verified as H-u or H-exact and an H-exact proof verified as
/// H-FIPS are rejected before event 0, a rewritten relation byte fails.
#[test]
fn mode_b_h_fips_round_trips_and_cross_relation_rejections() {
    on_large_stack(|| {
        let _ = round_trip_as(Relation::HFips, 1, 0, 0xf001);
        let _ = round_trip_as(Relation::HFips, 60, 4, 0xf064);
        // 254 honest records and two synthetic weight-80 corner records.
        let mut records = common::records(0xf256, 200, 54);
        records.push(agg_witness::gen::hint_corner_record(1));
        records.push(agg_witness::gen::hint_corner_record(3));
        let instance = PublicInstance::from_records(&records).expect("instance");
        let fixture = Fixture {
            records,
            instance,
            relation: Relation::HFips,
        };
        let bridge = SeededMatrices;
        let output = fixture.prove(&bridge);
        let bytes = output.proof.to_bytes();
        let public = fixture.public();
        assert_eq!(bytes[2], 2);
        assert!(output.proof.stage2.claims.squares.is_empty());
        assert_eq!(output.proof.stage2.claims.fips.len(), 2);
        assert_eq!(output.proof.stage3.claims.0.len(), 36);
        // 256 records: d_main = 18 rounds of 4 + 4 + 3 elements, 13 + 18 + 6 + 8 + 36 claims.
        assert_eq!(output.proof.size().piop, 16 * (81 + 11 * 18));
        eprintln!("N = 256, H-FIPS (with two weight-80 corner records): aggregate {} bytes, prover {}", bytes.len(), output.timings);
        let report = verify_mode_b(&public, &bytes, &bridge, Relation::HFips, None);
        report_line("N = 256 H-FIPS: verifier uncached", &report);
        assert!(report.accepted());
        for expected in [Relation::Hu, Relation::HExact] {
            assert!(matches!(
                rejected(verify_mode_b(&public, &bytes, &bridge, expected, None)),
                VerifyError::Relation { found: Relation::HFips, .. }
            ));
        }
        for flag in [0_u8, 1] {
            let mut rewritten = bytes.clone();
            rewritten[2] = flag;
            for expected in Relation::ALL {
                assert!(!verify_mode_b(&public, &rewritten, &bridge, expected, None).accepted());
            }
        }
        // The same records proved under H-exact, verified as H-FIPS (rejected before event 0)
        // and promoted to H-FIPS (rejected).
        let exact = Fixture {
            records: fixture.records.clone(),
            instance: fixture.instance.clone(),
            relation: Relation::HExact,
        };
        let exact_bytes = exact.prove(&bridge).proof.to_bytes();
        assert!(verify_mode_b(&public, &exact_bytes, &bridge, Relation::HExact, None).accepted());
        assert!(matches!(
            rejected(verify_mode_b(&public, &exact_bytes, &bridge, Relation::HFips, None)),
            VerifyError::Relation {
                expected: Relation::HFips,
                found: Relation::HExact
            }
        ));
        let mut promoted = exact_bytes;
        promoted[2] = Relation::HFips.id();
        assert!(!verify_mode_b(&public, &promoted, &bridge, Relation::HFips, None).accepted());
        // A wrong nonce at the stage-2 gamma site (g = 3, 10-bit nonce) is rejected.
        let manifest = CommitmentManifest::for_relation(256, Relation::HFips).unwrap();
        let plan = PowPlan::new(&manifest, FlockProfile::PublicBits, &KeccakPlan::default());
        let mut offset = 0;
        for site in plan.sites() {
            if site.class == SiteClass::StageGamma && site.i == 2 {
                assert_eq!((site.bits, site.nonce_bits()), (3, 10));
                let mut wrong = output.proof.clone();
                wrong.work[offset / 8] ^= 1 << (offset % 8);
                assert!(!verify_mode_b(&public, &wrong.to_bytes(), &bridge, Relation::HFips, None).accepted());
                break;
            }
            offset += usize::from(site.nonce_bits());
        }
        assert!(offset > 0);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore = "1,024 signatures: run with --release")]
fn mode_b_round_trip_at_1024_signatures() {
    on_large_stack(|| {
        let _ = round_trip(1000, 24, 0xb1024);
    });
}

/// The prover run on tampered tables (unchecked stages) with the honest public bits.
fn prove_tampered(
    fixture: &Fixture,
    bridge: &FixedMatrices,
    main: &MainTables,
    tamper_bridge: Option<BridgeTamper<'_>>,
) -> Result<Vec<u8>, String> {
    let inputs = fixture.inputs();
    let signatures = fixture.signatures();
    let prover_inputs = ProverInputs {
        instance: &fixture.instance,
        records: &inputs,
        signatures: &signatures,
        relation: fixture.relation,
    };
    let (manifest, honest) = main_tables(&prover_inputs).map_err(|e| e.to_string())?;
    let side = PublicBits::from_tables(&manifest, &honest);
    let hooks = ProveHooks {
        unchecked: true,
        tamper_bridge,
    };
    prove_with_tables(
        &prover_inputs,
        &manifest,
        main,
        bridge,
        &side,
        FlockProfile::PublicBits,
        &hooks,
    )
    .map(|output| output.proof.to_bytes())
    .map_err(|e| e.to_string())
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

#[test]
fn every_stage_tampering_is_rejected_end_to_end() {
    on_large_stack(|| {
        // 50 live records (N* = 64, d_main = 16) keep the 12 prover runs short.
        let fixture = Fixture::new(0x7a, 40, 10);
        let bridge = FixedMatrices(BridgeMatrices::random_for_tests(0x7a));
        let inputs = fixture.inputs();
        let signatures = fixture.signatures();
        let (manifest, honest) = main_tables(&ProverInputs {
            instance: &fixture.instance,
            records: &inputs,
            signatures: &signatures,
            relation: Relation::Hu,
        })
        .unwrap();
        let public = fixture.public();
        let verify = |bytes: &[u8]| verify_mode_b(&public, bytes, &bridge, Relation::Hu, None);
        let honest_bytes = prove_tampered(&fixture, &bridge, &honest, None).unwrap();
        assert!(
            verify(&honest_bytes).accepted(),
            "the unchecked prover is honest on honest tables"
        );

        let x = main_index(17, 2, 5);
        let tampered = |f: &dyn Fn(&mut MainTables)| {
            let mut main = honest.clone();
            f(&mut main);
            main
        };
        let step = |d: i32| if d < LOOKUP_SHIFT - 1 { d + 1 } else { d - 1 };
        let expect = |label: &str, main: &MainTables, check: &dyn Fn(&VerifyError) -> bool| {
            let bytes = prove_tampered(&fixture, &bridge, main, None).expect(label);
            let error = rejected(verify(&bytes));
            eprintln!("{label}: rejected with {error}");
            assert!(check(&error), "{label}: {error:?}");
        };
        let at_stage = |s: usize| move |e: &VerifyError| matches!(e, VerifyError::Stage(StageError::Terminal(t)) if *t == s);

        // A 16-bit column changed inside the table: the ring identity (stage 3).
        for column in [Column::Z_LO, Column::V_LO] {
            expect(
                &column.name(),
                &tampered(&|m| {
                    let cell = m.wide_cell_mut(&manifest, column, x);
                    *cell = step(*cell);
                }),
                &at_stage(3),
            );
        }
        // A committed bit flipped: committed and public bits differ, so the bridge final
        // vector no longer matches the projection of the public bits.
        expect(
            "flipped bit",
            &tampered(&|m| *cell(&manifest, m, Column::W1Bit(0), x) ^= 1),
            &|e| matches!(e, VerifyError::Keccak(0)),
        );
        // A non-Boolean bit (0 -> 2 keeps every parity): Booleanity in stage 1.
        let even = (0..1024)
            .map(|p| main_index(p % 256, p / 256, 7))
            .find(|&y| *cell(&manifest, &mut honest.clone(), Column::W1Bit(1), y) == 0)
            .unwrap();
        expect(
            "bit = 2",
            &tampered(&|m| *cell(&manifest, m, Column::W1Bit(1), even) = 2),
            &at_stage(1),
        );
        // w1' = 44 in the committed bits only: the projection.
        let w44 = tampered(&|m| {
            for (t, bit) in [0, 0, 1, 1, 0, 1].into_iter().enumerate() {
                *cell(&manifest, m, Column::W1Bit(t as u8), x) = bit;
            }
        });
        expect("w1' = 44", &w44, &|e| matches!(e, VerifyError::Keccak(0)));
        // A 16-bit value outside [-2^15, 2^15) (looked-up value 2^16): Akita refuses to
        // commit it, because the width-16 coefficient bound is the balanced 16-bit
        // interval, so no aggregate exists to verify. The verifier's rejection of a
        // looked-up value outside T_16 is tested at the PIOP level (tests/stages.rs,
        // without commitments): a cheating prover's commitment to such a value is bound
        // only by the lookup, not by this admission check.
        let wide = tampered(&|m| *m.wide_cell_mut(&manifest, Column::U_LO, x) = LOOKUP_SHIFT);
        let error = prove_tampered(&fixture, &bridge, &wide, None)
            .expect_err("Akita refuses a stored value outside the width-16 interval");
        eprintln!("u_lo = 2^15: the prover fails ({error})");
        assert!(
            error.starts_with("pcs: commit: akita:")
                && error.contains("exceeds the scheduled coefficient bound"),
            "{error}"
        );
        // Base-4 and base-6 digits outside their alphabets: the leaves of stage 2 (U_hi = 3
        // is in a base-8 alphabet, not in H-u's base-6 one).
        expect(
            "base-4 digit",
            &tampered(&|m| *cell(&manifest, m, Column::Z_HI, x) = 3),
            &at_stage(2),
        );
        expect(
            "base-6 digit",
            &tampered(&|m| *cell(&manifest, m, Column::U_HI, x) = 3),
            &at_stage(2),
        );

        // Round-2 tampering: a level-1 carry (m recounted), a level-2 parity, a
        // final-vector bit.
        let run_bridge = |label: &str,
                          tamper: &dyn Fn(&CommitmentManifest, &mut BridgeGroupTables),
                          check: &dyn Fn(&VerifyError) -> bool| {
            let bytes = prove_tampered(&fixture, &bridge, &honest, Some(tamper)).expect(label);
            let error = rejected(verify(&bytes));
            eprintln!("{label}: rejected with {error}");
            assert!(check(&error), "{label}: {error:?}");
        };
        run_bridge(
            "level-1 carry",
            &|m, b| {
                let carries = b.carries_mut(1);
                carries[3] = step(carries[3]);
                b.recount(m, &honest).unwrap();
            },
            &at_stage(1),
        );
        run_bridge(
            "level-2 parity",
            &|_, b| b.parities_mut()[2][5] ^= 1,
            &at_stage(1),
        );
        run_bridge(
            "final-vector bit",
            &|_, b| b.public_y_mut()[11] ^= 1,
            &|e| matches!(e, VerifyError::Keccak(0)),
        );
        // A wrong multiplicity: the lookup.
        run_bridge(
            "multiplicity",
            &|_, b| {
                let mut m = b.multiplicity().to_vec();
                m[7] += 1;
                b.set_multiplicity(m);
            },
            &|e| matches!(e, VerifyError::Lookup(_)),
        );
    });
}

/// The mode-B checks under another declared profile.
struct AsProfile(FlockProfile);

impl KeccakVerifier for AsProfile {
    fn profile(&self) -> FlockProfile {
        self.0
    }

    fn verify(&self, segment: &SegmentContext<'_>) -> bool {
        PublicBitsVerifier.verify(segment)
    }

    fn check_bits(&self, bits: &BitsContext<'_>) -> bool {
        PublicBitsVerifier.check_bits(bits)
    }
}

/// The projection predicate alone, without the direct bit check (`check_bits`).
struct ProjectionOnly;

impl KeccakVerifier for ProjectionOnly {
    fn profile(&self) -> FlockProfile {
        FlockProfile::PublicBits
    }

    fn verify(&self, segment: &SegmentContext<'_>) -> bool {
        PublicBitsVerifier.verify(segment)
    }
}

/// Mode B's verifier with a plan that names one tie site (a plan-digest mismatch).
struct WithTie;

impl KeccakVerifier for WithTie {
    fn profile(&self) -> FlockProfile {
        FlockProfile::PublicBits
    }

    fn plan(&self, _geometry: &Geometry, _commitments: &[Vec<u8>]) -> Option<KeccakPlan> {
        Some(KeccakPlan {
            span: 0,
            d_part: 0,
            parts: vec![agg_bridge::pow::PartWork {
                tie_bits: 10,
                eq_bits: 10,
                round_bits: 7,
                rounds: 23,
            }],
        })
    }

    fn verify(&self, segment: &SegmentContext<'_>) -> bool {
        PublicBitsVerifier.verify(segment)
    }

    fn check_bits(&self, bits: &BitsContext<'_>) -> bool {
        PublicBitsVerifier.check_bits(bits)
    }
}

#[test]
fn committed_bits_other_than_the_public_ones_are_caught_by_the_bit_check() {
    on_large_stack(|| {
        // Fixed matrices with two equal M_1 columns let a prover flip b_0 at both positions
        // without changing any parity, and u absorbs the change of w1' in the ring identity:
        // every PIOP check passes, the projection of the public bits matches, and only the
        // direct check of b_t(r^(3)) against the public bits rejects.
        let fixture = Fixture::new(0x5b, 20, 4);
        let inputs = fixture.inputs();
        let signatures = fixture.signatures();
        let (manifest, honest) = main_tables(&ProverInputs {
            instance: &fixture.instance,
            records: &inputs,
            signatures: &signatures,
            relation: Relation::Hu,
        })
        .unwrap();
        let read = |m: &MainTables, column: Column, x: usize| m.column(&manifest, column).value(x);
        let u_at = |m: &MainTables, x: usize| {
            U16_DIGITS.compose(&[read(m, Column::U_LO, x), read(m, Column::U_HI, x)])
        };
        // Two positions of record 3 whose u can absorb -2 gamma_2 (w1' bit 0 flipped).
        let shifted = |x: usize| {
            let delta = if read(&honest, Column::W1Bit(0), x) == 0 {
                1
            } else {
                -1
            };
            let u = u_at(&honest, x) - 2 * GAMMA2 * delta;
            (-190_465..=190_465).contains(&u).then_some(u)
        };
        let positions: Vec<(usize, i64)> = (0..1024)
            .map(|p| main_index(p % 256, p / 256, 3))
            .filter_map(|x| shifted(x).map(|u| (x, u)))
            .take(2)
            .collect();
        assert_eq!(positions.len(), 2);
        let mut tampered = honest.clone();
        for &(x, u) in &positions {
            *cell(&manifest, &mut tampered, Column::W1Bit(0), x) ^= 1;
            let digits = U16_DIGITS.decompose(u);
            *tampered.wide_cell_mut(&manifest, Column::U_LO, x) = digits[0] as i32;
            *cell(&manifest, &mut tampered, Column::U_HI, x) = digits[1] as i8;
        }
        let mut matrices = BridgeMatrices::random_for_tests(0x5c);
        let (z1, z2) = (positions[0].0 & (J1 - 1), positions[1].0 & (J1 - 1));
        let column: Vec<u32> = matrices.m1.column(z1).to_vec();
        matrices.m1.entries[z2 * K1..(z2 + 1) * K1].copy_from_slice(&column);
        let bridge = FixedMatrices(matrices);

        let bytes = prove_tampered(&fixture, &bridge, &tampered, None).expect("prove");
        let public = fixture.public();
        let report = verify_mode_b(&public, &bytes, &bridge, Relation::Hu, None);
        assert!(
            matches!(report.outcome, Err(VerifyError::Bits)),
            "{:?}",
            report.outcome
        );
        let without = verify_timed(
            &public,
            &bytes,
            &bridge,
            &ProjectionOnly,
            FlockProfile::PublicBits,
            Relation::Hu,
            None,
        );
        assert!(
            without.accepted(),
            "the projection alone does not bind the bits under fixed matrices"
        );
    });
}
