//! The fork-join transcript (deviation K22, `docs/soundness.md` 12.2): the Keccak side's proof
//! items are absorbed at the join, after the stage-3 claims and before the PCS session
//! digest. So no PIOP challenge depends on them, while the session (and with it the Akita
//! opening) binds every byte; the side's challengers receive the digest of the event-6
//! record. The prover's two schedules (the Flock proofs on their own pool beside the PIOP, or
//! one after the other) produce the same proof.
//!
//! This binary sets `AGG_FLOCK_THREADS`, so it holds one test function only.

mod common;

use std::sync::Mutex;

use agg_piop::binding::FlockProfile;
use agg_piop::manifest::Relation;
use agg_piop::prover::{prove, FixedMatrices, KeccakSide, ProveError, ProverInputs, ProverOutput};
use agg_piop::transcript::flock_record_digest;
use agg_piop::verifier::{verify, KeccakVerifier, SegmentContext, VerifyError};
use agg_witness::bridge::BridgeMatrices;
use agg_witness::instance::PublicInstance;
use agg_witness::params::C_TILDE_BYTES;

/// A Keccak side without commitments whose proof items are fixed bytes; it records the
/// event-6 record digest it was given.
struct FixedProofs {
    proof: Vec<u8>,
    record: Mutex<Option<[u8; 32]>>,
}

impl FixedProofs {
    fn new(proof: &[u8]) -> Self {
        Self {
            proof: proof.to_vec(),
            record: Mutex::new(None),
        }
    }
}

impl KeccakSide for FixedProofs {
    fn commitments(&self, segments: usize) -> Vec<Vec<u8>> {
        vec![Vec::new(); segments]
    }

    fn prove(
        &self,
        _bridge_seed: &[u8; 32],
        _matrices: &BridgeMatrices,
        seeds: &[[u8; 32]],
        _final_vectors: &[Vec<u8>],
        record: &[u8; 32],
    ) -> Result<Vec<Vec<u8>>, ProveError> {
        *self.record.lock().unwrap() = Some(*record);
        Ok(vec![self.proof.clone(); seeds.len()])
    }
}

/// Accepts any proof item with empty commitment and share items (profile 0, test hooks):
/// the verifier's own checks of the transcript are what is under test.
struct AnyProof;

impl KeccakVerifier for AnyProof {
    fn profile(&self) -> FlockProfile {
        FlockProfile::None
    }

    fn verify(&self, segment: &SegmentContext<'_>) -> bool {
        segment.commitment.is_empty() && segment.shares.is_empty()
    }
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
fn the_join_binds_the_flock_proofs_and_no_piop_challenge_depends_on_them() {
    on_large_stack(|| {
        let records = common::records(0xf01, 24, 8);
        let instance = PublicInstance::from_records(&records).unwrap();
        let inputs: Vec<(&[u8], &[u8], [u8; C_TILDE_BYTES])> = records
            .iter()
            .map(|r| (r.pk.as_slice(), r.msg.as_slice(), r.sig[..C_TILDE_BYTES].try_into().unwrap()))
            .collect();
        let signatures: Vec<&[u8]> = records.iter().map(|r| r.sig.as_slice()).collect();
        let bridge = FixedMatrices(BridgeMatrices::random_for_tests(0xf01));
        let prover_inputs = ProverInputs {
            instance: &instance,
            records: &inputs,
            signatures: &signatures,
            relation: Relation::Hu,
        };
        let run = |threads: &str, side: &FixedProofs| -> ProverOutput {
            std::env::set_var("AGG_FLOCK_THREADS", threads);
            let output = prove(&prover_inputs, &bridge, side, FlockProfile::None).expect("prove");
            std::env::remove_var("AGG_FLOCK_THREADS");
            output
        };

        // The fork (2 threads) and the serial schedule give the same proof.
        let side = FixedProofs::new(&[1, 2, 3]);
        let forked = run("2", &side);
        let record = side.record.lock().unwrap().expect("the side proved");
        let serial = run("0", &FixedProofs::new(&[1, 2, 3]));
        if rayon::current_num_threads() > 1 {
            assert_eq!(forked.timings.flock_threads, 2);
            assert!(forked.timings.fork_join > std::time::Duration::ZERO);
        }
        assert_eq!(serial.timings.flock_threads, 0);
        assert_eq!(forked.proof.to_bytes(), serial.proof.to_bytes(), "same proof either way");
        assert_eq!(forked.proof.flock_proofs, vec![vec![1, 2, 3]]);
        // The side was given the digest of the event-6 record (final vectors, share items).
        assert_eq!(
            record,
            flock_record_digest(&forked.proof.final_vectors, &forked.proof.flock_shares)
        );

        // Other proof bytes: every PIOP challenge, message and nonce is the same, the PCS
        // session digest and the opening differ.
        let other = run("2", &FixedProofs::new(&[9]));
        let (a, b) = (&forked, &other);
        assert_eq!(a.points.0.tau, b.points.0.tau);
        assert_eq!(a.points.0.r1, b.points.0.r1);
        assert_eq!(a.points.0.r2, b.points.0.r2);
        assert_eq!(a.points.0.lookup, b.points.0.lookup);
        assert_eq!(a.points.1, b.points.1, "r^(3)");
        assert_eq!(a.pi_m, b.pi_m);
        assert_eq!(a.proof.lookup, b.proof.lookup);
        assert_eq!(a.proof.stage1, b.proof.stage1);
        assert_eq!(a.proof.stage2, b.proof.stage2);
        assert_eq!(a.proof.stage3, b.proof.stage3);
        assert_eq!(a.proof.work, b.proof.work, "the PIOP nonces precede the join");
        assert_ne!(a.session, b.session, "the PCS session digest binds the proof items");
        assert_ne!(a.proof.opening, b.proof.opening);

        // The verifier accepts each proof as made, and rejects a proof whose items were
        // replaced by the other proof's at the Akita opening: the PIOP stages pass (their
        // challenges do not depend on the items), the session digest does not match.
        let public: Vec<(&[u8], &[u8])> = records
            .iter()
            .map(|r| (r.pk.as_slice(), r.msg.as_slice()))
            .collect();
        let check = |proof: &agg_piop::wire::AggregateProof| {
            verify(&public, &proof.to_bytes(), &bridge, &AnyProof, FlockProfile::None, Relation::Hu, None)
        };
        check(&a.proof).expect("verify the forked proof");
        check(&b.proof).expect("verify the other proof");
        let mut moved = a.proof.clone();
        moved.flock_proofs = b.proof.flock_proofs.clone();
        assert!(matches!(check(&moved), Err(VerifyError::Opening(_))), "items moved");
        let mut tampered = a.proof.clone();
        tampered.flock_proofs[0][1] ^= 1;
        assert!(matches!(check(&tampered), Err(VerifyError::Opening(_))), "one byte changed");
        // A share item changes event 6: the PIOP transcript diverges before the join.
        let mut share = a.proof.clone();
        share.flock_shares[0] = vec![0];
        assert!(matches!(check(&share), Err(VerifyError::Keccak(0))), "AnyProof refuses shares");
    });
}
