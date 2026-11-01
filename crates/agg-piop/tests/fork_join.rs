//! The fork-join transcript (`docs/soundness.md`): the transcript never absorbs the Keccak
//! side's proof items, so no PIOP challenge, no PCS session digest and no Akita challenge
//! depends on them; the side's challengers receive the digest of the final-vector record,
//! and the side's verifier is the only check on the items. The share items are absorbed at
//! the final-vector step, before the lookup challenge `X`, so every PIOP challenge depends on
//! them. The prover's three schedules (the Flock proofs on their own pool beside the PIOP and
//! the opening, beside the PIOP with the opening after the join, or one after the other)
//! produce the same proof.
//!
//! This binary sets `AGG_FLOCK_THREADS` and `AGG_LATE_JOIN`, so it holds one test function
//! only.

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

/// A Keccak side without commitments whose share and proof items are fixed bytes; it
/// records the final-vector record digest it was given.
struct FixedProofs {
    share: Vec<u8>,
    proof: Vec<u8>,
    record: Mutex<Option<[u8; 32]>>,
}

impl FixedProofs {
    fn new(proof: &[u8]) -> Self {
        Self::with_share(&[], proof)
    }

    fn with_share(share: &[u8], proof: &[u8]) -> Self {
        Self {
            share: share.to_vec(),
            proof: proof.to_vec(),
            record: Mutex::new(None),
        }
    }
}

impl KeccakSide for FixedProofs {
    fn commitments(&self, segments: usize) -> Vec<Vec<u8>> {
        vec![Vec::new(); segments]
    }

    fn shares(
        &self,
        _matrices: &BridgeMatrices,
        final_vectors: &[Vec<u8>],
    ) -> Result<Vec<Vec<u8>>, ProveError> {
        Ok(vec![self.share.clone(); final_vectors.len()])
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

/// Accepts exactly one proof item, with empty commitment and share items (profile 0, test
/// hooks): a Keccak side that checks its items.
struct ExactProof(Vec<u8>);

impl KeccakVerifier for ExactProof {
    fn profile(&self) -> FlockProfile {
        FlockProfile::None
    }

    fn verify(&self, segment: &SegmentContext<'_>) -> bool {
        segment.commitment.is_empty() && segment.shares.is_empty() && segment.proof == self.0
    }
}

/// Accepts any share and proof item with an empty commitment (profile 0, test hooks), so
/// that a changed share item reaches the PIOP checks.
struct AnyShares;

impl KeccakVerifier for AnyShares {
    fn profile(&self) -> FlockProfile {
        FlockProfile::None
    }

    fn verify(&self, segment: &SegmentContext<'_>) -> bool {
        segment.commitment.is_empty()
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
fn no_piop_or_akita_challenge_depends_on_the_flock_proofs() {
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
        let run_with = |threads: &str, late_join: &str, side: &FixedProofs| -> ProverOutput {
            std::env::set_var("AGG_FLOCK_THREADS", threads);
            std::env::set_var("AGG_LATE_JOIN", late_join);
            let output = prove(&prover_inputs, &bridge, side, FlockProfile::None).expect("prove");
            std::env::remove_var("AGG_FLOCK_THREADS");
            std::env::remove_var("AGG_LATE_JOIN");
            output
        };
        let run = |threads: &str, side: &FixedProofs| run_with(threads, "1", side);

        // The fork (2 threads) with the opening beside the Flock proofs, the fork with the
        // opening after the join, and the serial schedule give the same proof.
        let side = FixedProofs::new(&[1, 2, 3]);
        let forked = run("2", &side);
        let record = side.record.lock().unwrap().expect("the side proved");
        let early = run_with("2", "0", &FixedProofs::new(&[1, 2, 3]));
        let serial = run("0", &FixedProofs::new(&[1, 2, 3]));
        if rayon::current_num_threads() > 1 {
            assert_eq!(forked.timings.flock_threads, 2);
            assert!(forked.timings.fork_join > std::time::Duration::ZERO);
            // Beside the Flock proofs the opening runs inside the fork-join wall time.
            assert!(forked.timings.fork_join >= forked.timings.opening);
            assert!(forked.timings.opening_threads >= 1);
        }
        assert_eq!(serial.timings.flock_threads, 0);
        assert_eq!(forked.proof.to_bytes(), serial.proof.to_bytes(), "same proof either way");
        assert_eq!(forked.proof.to_bytes(), early.proof.to_bytes(), "opening before or after the join");
        assert_eq!(forked.session, early.session);
        assert_eq!(forked.proof.flock_proofs, vec![vec![1, 2, 3]]);
        // The side was given the digest of the final-vector record (final vectors, share items).
        assert_eq!(
            record,
            flock_record_digest(&forked.proof.final_vectors, &forked.proof.flock_shares)
        );

        // Other proof bytes: every PIOP challenge, message and nonce, the PCS session digest
        // and the opening are the same.
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
        assert_eq!(a.proof.work, b.proof.work, "no nonce depends on the proof items");
        assert_eq!(a.session, b.session, "the PCS session digest omits the proof items");
        assert_eq!(a.proof.opening, b.proof.opening, "so does the Akita opening");

        // A verifier that accepts any proof item accepts each proof as made, and also a proof
        // whose items were replaced or changed: the transcript does not bind them. A Keccak
        // side that checks its items rejects them there, and only there.
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
        check(&moved).expect("items moved, not bound by the transcript");
        let mut tampered = a.proof.clone();
        tampered.flock_proofs[0][1] ^= 1;
        check(&tampered).expect("one byte changed, not bound by the transcript");
        let strict = |proof: &agg_piop::wire::AggregateProof| {
            verify(&public, &proof.to_bytes(), &bridge, &ExactProof(vec![1, 2, 3]), FlockProfile::None, Relation::Hu, None)
        };
        strict(&a.proof).expect("verify the forked proof against its own item");
        assert!(matches!(strict(&moved), Err(VerifyError::Keccak(0))), "items moved");
        assert!(matches!(strict(&tampered), Err(VerifyError::Keccak(0))), "one byte changed");
        // AnyProof refuses a non-empty share item at the Keccak side's check.
        let mut share = a.proof.clone();
        share.flock_shares[0] = vec![0];
        assert!(matches!(check(&share), Err(VerifyError::Keccak(0))), "AnyProof refuses shares");

        // The share items are bound at the final-vector step, before `X`: a side that sends a share item
        // gets another record digest and other PIOP challenges, from the lookup on.
        let shared_side = FixedProofs::with_share(&[7, 7], &[1, 2, 3]);
        let shared = run("2", &shared_side);
        let shared_record = shared_side.record.lock().unwrap().expect("the side proved");
        assert_eq!(shared.proof.flock_shares, vec![vec![7, 7]]);
        assert_eq!(shared.proof.final_vectors, a.proof.final_vectors, "Y precedes the shares");
        assert_eq!(
            shared_record,
            flock_record_digest(&shared.proof.final_vectors, &shared.proof.flock_shares)
        );
        assert_ne!(shared_record, record, "the record digest binds the share items");
        assert_ne!(shared.points.0.lookup, a.points.0.lookup, "the lookup challenges follow event 6");
        assert_ne!(shared.points.0.tau, a.points.0.tau);
        assert_ne!(shared.points.1, a.points.1, "r^(3)");
        assert_ne!(shared.session, a.session);
        // A verifier that accepts any share item accepts both proofs as made, and rejects a
        // proof whose share item was replaced before the join: the replayed challenges from
        // `X` on differ from the prover's, so the lookup, a stage or a work nonce fails.
        let check_shares = |proof: &agg_piop::wire::AggregateProof| {
            verify(&public, &proof.to_bytes(), &bridge, &AnyShares, FlockProfile::None, Relation::Hu, None)
        };
        check_shares(&a.proof).expect("verify the proof without shares");
        check_shares(&shared.proof).expect("verify the proof with a share item");
        for (label, proof, item) in [("added", &a.proof, vec![7, 7]), ("changed", &shared.proof, vec![7, 6])] {
            let mut moved = proof.clone();
            moved.flock_shares[0] = item;
            match check_shares(&moved) {
                Err(VerifyError::Lookup(_) | VerifyError::Stage(_) | VerifyError::Transcript(_)) => {}
                other => panic!("share item {label}: expected a rejection before the join, got {other:?}"),
            }
        }
    });
}
