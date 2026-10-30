//! The forked PIOP transcript (`docs/spec.md` Section 10): after event 6 branch L (the
//! lookup) and branch S (`tau` through stage 2) run on two clones of the sponge, and the
//! join absorbs their digests, L's then S's, before stage 3. The prover's schedules (the
//! branches side by side off a pool, inside a pool by `rayon::join`, on a one-thread pool,
//! with branch S on its own pool, or one after the other with `AGG_PIOP_FORK=0`) give the
//! same proof. A changed lookup
//! message, a changed stage-2 message, the digests joined in the wrong order and the two
//! branches' nonces exchanged in the stream are each rejected.
//!
//! This binary sets `AGG_PIOP_FORK`, so it holds one test function only.

mod common;

use agg_piop::binding::FlockProfile;
use agg_piop::manifest::{CommitmentManifest, Relation};
use agg_piop::lookup::LookupError;
use agg_piop::pow::{SiteClass, WorkReader, WorkWriter};
use agg_piop::prover::{main_tables, prove, prove_with_tables, FixedMatrices, NoKeccak, ProveHooks, ProverInputs};
use agg_piop::stage::StageError;
use agg_piop::transcript::{Registry, TranscriptError};
use agg_piop::verifier::{verify, NoKeccakVerifier, VerifyError};
use agg_piop::wire::AggregateProof;
use agg_piop::OpeningField;
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

/// The nonce stream with branch S's nonces moved in front of branch L's.
fn branches_exchanged(manifest: &CommitmentManifest, work: &[u8]) -> Vec<u8> {
    let registry = Registry::new(manifest);
    let width = |pred: &dyn Fn(SiteClass, u16) -> bool| -> usize {
        registry
            .work_sites()
            .filter(|s| pred(s.class, s.i))
            .map(|s| usize::from(s.nonce_bits()))
            .sum()
    };
    let lookup = width(&|c, _| {
        matches!(c, SiteClass::LookupX | SiteClass::LookupMu | SiteClass::LookupGamma | SiteClass::LookupRound)
    });
    let stages = width(&|c, i| c == SiteClass::Tau || (matches!(c, SiteClass::StageGamma | SiteClass::StageRound) && i < 3));
    let total = registry.work_bits();
    let mut reader = WorkReader::new(work, total).expect("the honest stream");
    let mut bits = |n: usize| -> Vec<u32> { (0..n).map(|_| reader.read(1).unwrap()).collect() };
    let (l, s, rest) = (bits(lookup), bits(stages), bits(total - lookup - stages));
    let mut out = WorkWriter::default();
    for bit in s.iter().chain(&l).chain(&rest) {
        out.push(*bit, 1);
    }
    out.finish()
}

#[test]
fn forked_transcript_schedules_agree_and_tampering_is_rejected() {
    on_large_stack(|| {
        let records = common::records(0xf0c, 48, 16);
        let instance = PublicInstance::from_records(&records).unwrap();
        let inputs: Vec<(&[u8], &[u8], [u8; C_TILDE_BYTES])> = records
            .iter()
            .map(|r| (r.pk.as_slice(), r.msg.as_slice(), r.sig[..C_TILDE_BYTES].try_into().unwrap()))
            .collect();
        let signatures: Vec<&[u8]> = records.iter().map(|r| r.sig.as_slice()).collect();
        let bridge = FixedMatrices(BridgeMatrices::random_for_tests(0xf0c));
        let prover_inputs = ProverInputs {
            instance: &instance,
            records: &inputs,
            signatures: &signatures,
            relation: Relation::Hu,
        };
        let public: Vec<(&[u8], &[u8])> = records
            .iter()
            .map(|r| (r.pk.as_slice(), r.msg.as_slice()))
            .collect();
        let check = |proof: &AggregateProof| {
            verify(&public, &proof.to_bytes(), &bridge, &NoKeccakVerifier, FlockProfile::None, Relation::Hu, None)
        };
        let prove_now = || prove(&prover_inputs, &bridge, &NoKeccak, FlockProfile::None).expect("prove");

        // Off any pool (S on its own thread), inside a pool (rayon::join), on a one-thread
        // pool, and sequentially: one proof.
        let forked = prove_now();
        let bytes = forked.proof.to_bytes();
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .stack_size(64 << 20)
                .build()
                .unwrap();
            assert_eq!(pool.install(prove_now).proof.to_bytes(), bytes, "{threads}-thread pool");
        }
        std::env::set_var("AGG_PIOP_FORK", "0");
        let serial = prove_now();
        std::env::remove_var("AGG_PIOP_FORK");
        assert_eq!(serial.proof.to_bytes(), bytes, "sequential branches");
        // Branch S on its own pool (`AGG_PIOP_S_THREADS`): L keeps the main pool.
        std::env::set_var("AGG_PIOP_S_THREADS", "2");
        let own_pool = prove_now();
        std::env::remove_var("AGG_PIOP_S_THREADS");
        assert_eq!(own_pool.proof.to_bytes(), bytes, "stage-branch pool");
        check(&forked.proof).expect("verify the forked proof");

        // A changed lookup message (a GKR round of branch L).
        let mut lookup = forked.proof.clone();
        lookup.lookup.sessions[2].rounds[0][1] += OpeningField::from_u64(1);
        assert!(
            matches!(check(&lookup), Err(VerifyError::Lookup(_))),
            "lookup message: {:?}",
            check(&lookup)
        );
        // A changed stage-2 round message (branch S): the next round's nonce or the terminal.
        let mut stage2 = forked.proof.clone();
        stage2.stage2.rounds[3][1] += OpeningField::from_u64(1);
        assert!(
            matches!(
                check(&stage2),
                Err(VerifyError::Stage(StageError::Terminal(2) | StageError::Transcript(TranscriptError::Work { .. })))
            ),
            "stage-2 message: {:?}",
            check(&stage2)
        );

        // The digests joined in the wrong order: both branches are as before, stage 3 is
        // drawn from another trunk state and fails at its first work nonce or terminal.
        let (manifest, main) = main_tables(&prover_inputs).unwrap();
        let swapped = prove_with_tables(
            &prover_inputs,
            &manifest,
            &main,
            &bridge,
            &NoKeccak,
            FlockProfile::None,
            &ProveHooks {
                swap_join: true,
                ..ProveHooks::default()
            },
        )
        .expect("prove with the digests swapped")
        .proof;
        let honest = prove_with_tables(
            &prover_inputs,
            &manifest,
            &main,
            &bridge,
            &NoKeccak,
            FlockProfile::None,
            &ProveHooks::default(),
        )
        .expect("prove")
        .proof;
        assert_eq!(honest.to_bytes(), bytes);
        assert_eq!(swapped.lookup, honest.lookup);
        assert_eq!(swapped.stage1, honest.stage1);
        assert_eq!(swapped.stage2, honest.stage2);
        assert_ne!(swapped.stage3, honest.stage3);
        assert!(
            matches!(check(&swapped), Err(VerifyError::Stage(_) | VerifyError::Transcript(_))),
            "swapped digests: {:?}",
            check(&swapped)
        );

        // The two branches' nonces exchanged in the packed stream (S's in L's slot).
        let mut exchanged = forked.proof.clone();
        exchanged.work = branches_exchanged(&manifest, &forked.proof.work);
        assert_ne!(exchanged.work, forked.proof.work);
        assert!(
            matches!(
                check(&exchanged),
                Err(VerifyError::Lookup(LookupError::Transcript(ref why)))
                    if why.contains("work nonce rejected")
            ),
            "exchanged nonces: {:?}",
            check(&exchanged)
        );
    });
}
