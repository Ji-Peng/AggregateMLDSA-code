//! End to end: a full prover run at 256 signatures produces a proof blob that the
//! verifier accepts from the public records alone, and corrupted blobs are rejected.

use crate::common;

use agg_piop::manifest::Relation;
use agg_piop::transcript::TranscriptError;
use agg_piop::binding::FlockProfile;
use agg_piop::opening::opening_claims;
use agg_piop::pcs::opening_claims as evaluate_claims;
use agg_piop::prover::{main_tables, prove, FixedMatrices, NoKeccak, ProveError, ProverInputs};
use agg_piop::source_tables::{BridgeGroupTables, SourceTables};
use agg_piop::public_data::KeyCache;
use agg_piop::verifier::{verify, ConfigurationError, NoKeccakVerifier, VerifyError};
use agg_piop::wire::AggregateProof;
use agg_witness::bridge::BridgeMatrices;
use agg_witness::instance::PublicInstance;
use agg_witness::params::C_TILDE_BYTES;
use jolt_field::Ring;

fn on_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("test thread");
}

#[test]
fn full_prover_run_at_256_signatures_verifies_and_corruptions_are_rejected() {
    on_large_stack(|| {
        let records = common::records(0xe2e, 200, 56);
        let instance = PublicInstance::from_records(&records).unwrap();
        let inputs: Vec<(&[u8], &[u8], [u8; C_TILDE_BYTES])> = records
            .iter()
            .map(|r| {
                let mut ct = [0_u8; C_TILDE_BYTES];
                ct.copy_from_slice(&r.sig[..C_TILDE_BYTES]);
                (r.pk.as_slice(), r.msg.as_slice(), ct)
            })
            .collect();
        let signatures: Vec<&[u8]> = records.iter().map(|r| r.sig.as_slice()).collect();
        let bridge = FixedMatrices(BridgeMatrices::random_for_tests(0xe2e));
        let prover_inputs = ProverInputs {
            instance: &instance,
            records: &inputs,
            signatures: &signatures,
            relation: Relation::Hu,
        };
        // Mode A over matrices that ignore the seed is refused before any work.
        for profile in [FlockProfile::Fast, FlockProfile::Slim, FlockProfile::Slim8] {
            assert!(matches!(
                prove(&prover_inputs, &bridge, &NoKeccak, profile),
                Err(ProveError::Internal(_))
            ));
        }
        let start = std::time::Instant::now();
        let output = prove(&prover_inputs, &bridge, &NoKeccak, FlockProfile::None).expect("prove");
        let prove_time = start.elapsed();
        let bytes = output.proof.to_bytes();
        eprintln!(
            "N = 256: proof {} bytes (lookup {} bytes, PIOP {} bytes, opening {} bytes), prover {:.2?}",
            bytes.len(),
            output.proof.lookup_bytes(),
            output.proof.piop_bytes(),
            output.proof.opening.len(),
            prove_time
        );
        assert_eq!(AggregateProof::from_bytes(&bytes).unwrap(), output.proof);
        assert_eq!(output.flock_seeds.len(), 1);
        assert_eq!(output.proof.stage1.rounds.len(), 18);
        // The opening takes its claims from the stage-3 terminal values; they are the
        // evaluations of the committed tables at the group points.
        {
            let (manifest, main) = main_tables(&prover_inputs).unwrap();
            let matrices = BridgeMatrices::random_for_tests(0xe2e);
            let bridge_tables = BridgeGroupTables::build(&manifest, &main, &matrices, &output.psi).unwrap();
            let tables = SourceTables {
                manifest: &manifest,
                main: &main,
                bridge: &bridge_tables,
            };
            let claims = opening_claims(
                &manifest,
                &output.points.1,
                &output.proof.stage3.claims,
                &output.pi_m,
                &output.proof.multiplicities,
            )
            .unwrap();
            let points = claims.clone().map(|c| c.point);
            assert_eq!(evaluate_claims(tables, &points).unwrap(), claims);
        }

        let public: Vec<(&[u8], &[u8])> = records
            .iter()
            .map(|r| (r.pk.as_slice(), r.msg.as_slice()))
            .collect();
        let mut cache = KeyCache::default();
        let start = std::time::Instant::now();
        let verified = verify(
            &public,
            &bytes,
            &bridge,
            &NoKeccakVerifier,
            FlockProfile::None,
            Relation::Hu,
            Some(&mut cache),
        )
        .expect("verify");
        eprintln!("verifier (uncached) {:.2?}", start.elapsed());
        assert_eq!(verified.records, 256);
        assert_eq!(verified.bridge_seed, output.bridge_seed);
        assert_eq!(verified.session, output.session);
        let start = std::time::Instant::now();
        verify(
            &public,
            &bytes,
            &bridge,
            &NoKeccakVerifier,
            FlockProfile::None,
            Relation::Hu,
            Some(&mut cache),
        )
        .expect("verify with cached keys");
        eprintln!("verifier (cached keys) {:.2?}", start.elapsed());

        // The work nonce of the projection matrix Psi is the first 8 bits of the packed
        // stream (one bit of grinding). The prover searches from zero, so the nonce below
        // the honest one fails the predicate at the projection challenge, which follows the
        // instance, the commitments, the Flock commitments, the bridge seed and the label.
        const PSI_EVENT: usize = 6;
        let honest_psi = u32::from(output.proof.work[0]);
        assert!(honest_psi > 0, "the fixture's projection nonce is above zero");
        let mut psi = output.proof.clone();
        psi.work[0] = (honest_psi - 1) as u8;
        assert!(matches!(
            verify(
                &public,
                &psi.to_bytes(),
                &bridge,
                &NoKeccakVerifier,
                FlockProfile::None,
                Relation::Hu,
                None,
            ),
            Err(VerifyError::Transcript(TranscriptError::Work { index: PSI_EVENT }))
        ));

        // A wrong record list, a wrong c~, other matrices, a different Flock profile.
        let mut fewer = public.clone();
        fewer.pop();
        assert!(matches!(
            verify(&fewer, &bytes, &bridge, &NoKeccakVerifier, FlockProfile::None, Relation::Hu, None),
            Err(VerifyError::RecordCount)
        ));
        let mut swapped = public.clone();
        swapped.swap(3, 4);
        assert!(verify(&swapped, &bytes, &bridge, &NoKeccakVerifier, FlockProfile::None, Relation::Hu, None).is_err());
        let mut c_tilde = output.proof.clone();
        c_tilde.c_tilde[7][0] ^= 1;
        assert!(verify(&public, &c_tilde.to_bytes(), &bridge, &NoKeccakVerifier, FlockProfile::None, Relation::Hu, None).is_err());
        let other = FixedMatrices(BridgeMatrices::random_for_tests(0x1234));
        assert!(matches!(
            verify(&public, &bytes, &other, &NoKeccakVerifier, FlockProfile::None, Relation::Hu, None),
            Err(VerifyError::Stage(_))
        ));
        // A changed lookup message (a GKR round) or layer-1 value: the lookup.
        let mut lookup = output.proof.clone();
        lookup.lookup.sessions[4].rounds[1][0] += agg_piop::OpeningField::from_u64(1);
        assert!(matches!(
            verify(&public, &lookup.to_bytes(), &bridge, &NoKeccakVerifier, FlockProfile::None, Relation::Hu, None),
            Err(VerifyError::Lookup(_))
        ));
        let mut layer = output.proof.clone();
        layer.lookup.layer1[0][3] += agg_piop::OpeningField::from_u64(1);
        assert!(matches!(
            verify(&public, &layer.to_bytes(), &bridge, &NoKeccakVerifier, FlockProfile::None, Relation::Hu, None),
            Err(VerifyError::Lookup(_))
        ));
        // A changed multiplicity-digit claim, compensated so that the digits still compose
        // to m(pi_m): the lookup's own check passes; the claims are absorbed, so the honest
        // stage proofs no longer match the transcript: the next challenge's work nonce
        // (tau) fails (and the Akita opening of G_m would reject the claims in any case).
        let mut digits = output.proof.clone();
        digits.multiplicities.0[0] += agg_piop::OpeningField::from_u64(16);
        digits.multiplicities.0[1] -= agg_piop::OpeningField::from_u64(1);
        assert!(matches!(
            verify(&public, &digits.to_bytes(), &bridge, &NoKeccakVerifier, FlockProfile::None, Relation::Hu, None),
            Err(VerifyError::Transcript(TranscriptError::Work { .. }) | VerifyError::Stage(_) | VerifyError::Opening(_))
        ));
        // The no-Keccak stand-in cannot verify a mode-A profile (refused before parsing).
        assert!(matches!(
            verify(&public, &bytes, &bridge, &NoKeccakVerifier, FlockProfile::Fast, Relation::Hu, None),
            Err(VerifyError::Configuration(ConfigurationError::ProfileMismatch { .. }))
        ));

        // A changed round message (rejected at the next round's work nonce or the terminal),
        // terminal claim, pre-stage claim or opening byte.
        let mut round = output.proof.clone();
        round.stage2.rounds[5][2] = round.stage2.rounds[5][0];
        assert!(matches!(
            verify(&public, &round.to_bytes(), &bridge, &NoKeccakVerifier, FlockProfile::None, Relation::Hu, None),
            Err(VerifyError::Stage(
                agg_piop::stage::StageError::Terminal(2)
                    | agg_piop::stage::StageError::Transcript(TranscriptError::Work { .. })
            ))
        ));
        let mut claim = output.proof.clone();
        claim.stage3.claims.0[20] = claim.stage3.claims.0[21];
        assert!(matches!(
            verify(&public, &claim.to_bytes(), &bridge, &NoKeccakVerifier, FlockProfile::None, Relation::Hu, None),
            Err(VerifyError::Stage(agg_piop::stage::StageError::Terminal(3)))
        ));
        let mut prestage = output.proof.clone();
        prestage.prestage.y2[0] = prestage.prestage.y2[1] + prestage.prestage.kappa3;
        assert!(verify(&public, &prestage.to_bytes(), &bridge, &NoKeccakVerifier, FlockProfile::None, Relation::Hu, None).is_err());
        let mut opening = output.proof.clone();
        let last = opening.opening.len() - 1;
        opening.opening[last] ^= 0x40;
        assert!(matches!(
            verify(&public, &opening.to_bytes(), &bridge, &NoKeccakVerifier, FlockProfile::None, Relation::Hu, None),
            Err(VerifyError::Opening(_))
        ));
        let mut commitment = output.proof.clone();
        commitment.main_commitment.pop();
        assert!(matches!(
            verify(&public, &commitment.to_bytes(), &bridge, &NoKeccakVerifier, FlockProfile::None, Relation::Hu, None),
            Err(VerifyError::Commitment(agg_piop::manifest::GroupRole::Main))
        ));
        // Mode B rejects a non-empty Keccak payload.
        let mut keccak = output.proof.clone();
        keccak.flock_proofs[0] = vec![1];
        assert!(matches!(
            verify(&public, &keccak.to_bytes(), &bridge, &NoKeccakVerifier, FlockProfile::None, Relation::Hu, None),
            Err(VerifyError::Keccak(0))
        ));
        assert!(matches!(
            verify(&public, &bytes[..100], &bridge, &NoKeccakVerifier, FlockProfile::None, Relation::Hu, None),
            Err(VerifyError::Wire(_))
        ));
    });
}

#[test]
fn padded_batch_of_ten_signatures_round_trips() {
    on_large_stack(|| {
        let records = common::records(0x0a, 7, 3);
        let instance = PublicInstance::from_records(&records).unwrap();
        let inputs: Vec<(&[u8], &[u8], [u8; C_TILDE_BYTES])> = records
            .iter()
            .map(|r| {
                let mut ct = [0_u8; C_TILDE_BYTES];
                ct.copy_from_slice(&r.sig[..C_TILDE_BYTES]);
                (r.pk.as_slice(), r.msg.as_slice(), ct)
            })
            .collect();
        let signatures: Vec<&[u8]> = records.iter().map(|r| r.sig.as_slice()).collect();
        let bridge = FixedMatrices(BridgeMatrices::random_for_tests(0x0a));
        let output = prove(
            &ProverInputs {
                instance: &instance,
                records: &inputs,
                signatures: &signatures,
                relation: Relation::Hu,
            },
            &bridge,
            &NoKeccak,
            FlockProfile::None,
        )
        .expect("prove");
        let bytes = output.proof.to_bytes();
        eprintln!("N = 10 (N* = 16): proof {} bytes", bytes.len());
        let public: Vec<(&[u8], &[u8])> = records
            .iter()
            .map(|r| (r.pk.as_slice(), r.msg.as_slice()))
            .collect();
        verify(&public, &bytes, &bridge, &NoKeccakVerifier, FlockProfile::None, Relation::Hu, None).expect("verify");
    });
}
