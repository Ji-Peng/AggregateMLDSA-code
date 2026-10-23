//! End to end: a full prover run at 256 signatures produces a proof blob that the
//! verifier accepts from the public records alone, and corrupted blobs are rejected
//! (spec Sections 10, 11, 13).

mod common;

use agg_piop::binding::FlockProfile;
use agg_piop::prover::{prove, FixedMatrices, NoKeccak, ProveError, ProverInputs};
use agg_piop::public_data::KeyCache;
use agg_piop::verifier::{verify, ConfigurationError, NoKeccakVerifier, VerifyError};
use agg_piop::wire::AggregateProof;
use agg_witness::bridge::BridgeMatrices;
use agg_witness::instance::PublicInstance;
use agg_witness::params::C_TILDE_BYTES;

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
        };
        // Mode A over matrices that ignore the seed is refused before any work.
        for profile in [FlockProfile::Fast, FlockProfile::LaBinius] {
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
            "N = 256: proof {} bytes (PIOP {} bytes, opening {} bytes), prover {:.2?}",
            bytes.len(),
            output.proof.piop_bytes(),
            output.proof.opening.len(),
            prove_time
        );
        assert_eq!(AggregateProof::from_bytes(&bytes).unwrap(), output.proof);
        assert_eq!(output.flock_seeds.len(), 1);
        assert_eq!(output.proof.stage1.rounds.len(), 18);

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
            Some(&mut cache),
        )
        .expect("verify with cached keys");
        eprintln!("verifier (cached keys) {:.2?}", start.elapsed());

        // A wrong record list, a wrong c~, other matrices, a different Flock profile.
        let mut fewer = public.clone();
        fewer.pop();
        assert!(matches!(
            verify(&fewer, &bytes, &bridge, &NoKeccakVerifier, FlockProfile::None, None),
            Err(VerifyError::RecordCount)
        ));
        let mut swapped = public.clone();
        swapped.swap(3, 4);
        assert!(verify(&swapped, &bytes, &bridge, &NoKeccakVerifier, FlockProfile::None, None).is_err());
        let mut c_tilde = output.proof.clone();
        c_tilde.c_tilde[7][0] ^= 1;
        assert!(verify(&public, &c_tilde.to_bytes(), &bridge, &NoKeccakVerifier, FlockProfile::None, None).is_err());
        let other = FixedMatrices(BridgeMatrices::random_for_tests(0x1234));
        assert!(matches!(
            verify(&public, &bytes, &other, &NoKeccakVerifier, FlockProfile::None, None),
            Err(VerifyError::Stage(_))
        ));
        // The no-Keccak stand-in cannot verify a mode-A profile (refused before parsing).
        assert!(matches!(
            verify(&public, &bytes, &bridge, &NoKeccakVerifier, FlockProfile::Fast, None),
            Err(VerifyError::Configuration(ConfigurationError::ProfileMismatch { .. }))
        ));

        // A changed round message, terminal claim, pre-stage claim or opening byte.
        let mut round = output.proof.clone();
        round.stage2.rounds[5][2] = round.stage2.rounds[5][0];
        assert!(matches!(
            verify(&public, &round.to_bytes(), &bridge, &NoKeccakVerifier, FlockProfile::None, None),
            Err(VerifyError::Stage(agg_piop::stage::StageError::Terminal(2)))
        ));
        let mut claim = output.proof.clone();
        claim.stage3.claims.0[40] = claim.stage3.claims.0[41];
        assert!(matches!(
            verify(&public, &claim.to_bytes(), &bridge, &NoKeccakVerifier, FlockProfile::None, None),
            Err(VerifyError::Stage(agg_piop::stage::StageError::Terminal(3)))
        ));
        let mut prestage = output.proof.clone();
        prestage.prestage.y2[0] = prestage.prestage.y2[1] + prestage.prestage.kappa3[0];
        assert!(verify(&public, &prestage.to_bytes(), &bridge, &NoKeccakVerifier, FlockProfile::None, None).is_err());
        let mut opening = output.proof.clone();
        let last = opening.opening.len() - 1;
        opening.opening[last] ^= 0x40;
        assert!(matches!(
            verify(&public, &opening.to_bytes(), &bridge, &NoKeccakVerifier, FlockProfile::None, None),
            Err(VerifyError::Opening(_))
        ));
        let mut commitment = output.proof.clone();
        commitment.main_commitment.pop();
        assert!(matches!(
            verify(&public, &commitment.to_bytes(), &bridge, &NoKeccakVerifier, FlockProfile::None, None),
            Err(VerifyError::Commitment(agg_piop::manifest::GroupRole::Main))
        ));
        // Mode B rejects a non-empty Keccak payload.
        let mut keccak = output.proof.clone();
        keccak.flock_proofs[0] = vec![1];
        assert!(matches!(
            verify(&public, &keccak.to_bytes(), &bridge, &NoKeccakVerifier, FlockProfile::None, None),
            Err(VerifyError::Keccak(0))
        ));
        assert!(matches!(
            verify(&public, &bytes[..100], &bridge, &NoKeccakVerifier, FlockProfile::None, None),
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
        verify(&public, &bytes, &bridge, &NoKeccakVerifier, FlockProfile::None, None).expect("verify");
    });
}
