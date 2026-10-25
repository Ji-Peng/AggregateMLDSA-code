//! The prover with a one-thread rayon pool (deviation P12): `taskset -c 2` benchmarks and
//! `RAYON_NUM_THREADS=1` give rayon exactly one worker. The round-1 commits once blocked
//! that worker on OS threads whose Akita loops needed it, and the prover hung in round 1.
//! Both tests run a whole prover (rounds 1 and 2, the lookup, the stages and the opening)
//! and the verifier, under a timeout, once from a plain thread and once from inside an
//! installed one-thread pool (the caller is then a rayon worker). This binary sets the
//! global pool to one thread, so it must hold only tests that want that.

mod common;

use std::sync::mpsc;
use std::sync::Once;
use std::time::Duration;

use agg_piop::manifest::Relation;
use agg_piop::binding::FlockProfile;
use agg_piop::prover::{prove, FixedMatrices, NoKeccak, ProverInputs};
use agg_piop::verifier::{verify, NoKeccakVerifier};
use agg_witness::bridge::BridgeMatrices;
use agg_witness::instance::PublicInstance;
use agg_witness::params::C_TILDE_BYTES;

/// Far above the release-build run time (about a second); a deadlock fails instead of
/// hanging the test run.
const TIMEOUT: Duration = Duration::from_secs(600);

/// The global pool of this test binary has one thread, whichever test runs first.
fn one_thread_global_pool() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build_global()
            .expect("the first rayon use of this binary");
    });
    assert_eq!(rayon::current_num_threads(), 1);
}

/// Prove and verify a padded batch of ten records (`N* = 16`).
fn prove_and_verify_ten(seed: u64) {
    let records = common::records(seed, 7, 3);
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
    let bridge = FixedMatrices(BridgeMatrices::random_for_tests(seed));
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
    let public: Vec<(&[u8], &[u8])> = records
        .iter()
        .map(|r| (r.pk.as_slice(), r.msg.as_slice()))
        .collect();
    verify(
        &public,
        &output.proof.to_bytes(),
        &bridge,
        &NoKeccakVerifier,
        FlockProfile::None,
        Relation::Hu,
        None,
    )
    .expect("verify");
}

/// Run `f` on a large-stack thread and fail if it does not finish within [`TIMEOUT`].
fn within_timeout(f: impl FnOnce() + Send + 'static) {
    let (done, finished) = mpsc::channel();
    let worker = std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(move || {
            f();
            done.send(()).expect("receiver alive");
        })
        .expect("spawn");
    match finished.recv_timeout(TIMEOUT) {
        Ok(()) => worker.join().expect("test thread"),
        // The sender was dropped without a message: the thread panicked.
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            std::panic::resume_unwind(worker.join().expect_err("the thread panicked"));
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            panic!("the prover did not finish within {TIMEOUT:?} with one rayon thread (deadlock)")
        }
    }
}

#[test]
fn prover_called_from_a_plain_thread_runs_with_one_global_rayon_thread() {
    one_thread_global_pool();
    within_timeout(|| prove_and_verify_ten(0x51));
}

#[test]
fn prover_called_from_a_one_thread_pool_worker_runs() {
    one_thread_global_pool();
    within_timeout(|| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .stack_size(256 << 20)
            .build()
            .expect("pool");
        pool.install(|| {
            assert!(rayon::current_thread_index().is_some());
            prove_and_verify_ten(0x52);
        });
    });
}
