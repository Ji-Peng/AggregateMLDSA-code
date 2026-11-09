//! Benchmark of `PublicInstance::new`, `MainTables::build` for H-FIPS
//! and the streamed verifier (StreamedPublic::prepare + pass).

use std::time::Duration;

use agg_bench::harness::{median_duration, timed};
use agg_piop::manifest::{CommitmentManifest, Relation};
use agg_piop::public_stream::StreamedPublic;
use agg_piop::source_tables::MainTables;
use agg_piop::{BaseField, OpeningField};
use agg_witness::gen::{records_shared_keys, SplitMix64};
use agg_witness::instance::PublicInstance;
use agg_witness::params::C_TILDE_BYTES;

fn point(rng: &mut SplitMix64, len: usize) -> Vec<OpeningField> {
    (0..len)
        .map(|_| {
            let c0 = BaseField::solinas_reduce(rng.next_u64() as u128);
            let c1 = BaseField::solinas_reduce(rng.next_u64() as u128);
            OpeningField::new(c0, c1)
        })
        .collect()
}

fn parse_env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let n = parse_env_usize("STEP12_N", 1024);
    let n_keys = parse_env_usize("STEP12_KEYS", 1024);
    let reps = parse_env_usize("STEP12_REPS", 2);

    let backend = agg_mldsa_simd::engine().backend();

    println!("================================================================================");
    println!("ML-DSA SIMD Step 1 and 2 Benchmark (step12)");
    println!("N = {n}, Keys = {n_keys}, Reps = {reps} (1 warmup + {reps} timed), Backend = {backend:?}");
    println!("================================================================================");

    let seed = 0x1234_5678u64;
    println!("Generating {n} records over {n_keys} shared keys...");
    let records = records_shared_keys(seed, n, n_keys);
    let inputs: Vec<(&[u8], &[u8], [u8; C_TILDE_BYTES])> = records
        .iter()
        .map(|r| {
            let c_tilde: [u8; C_TILDE_BYTES] = r.sig[..C_TILDE_BYTES].try_into().unwrap();
            (r.pk.as_slice(), r.msg.as_slice(), c_tilde)
        })
        .collect();
    let signatures: Vec<&[u8]> = records.iter().map(|r| r.sig.as_slice()).collect();
    let manifest = CommitmentManifest::for_relation(n, Relation::HFips).unwrap();

    let mut step1_times = Vec::with_capacity(reps);
    let mut step2_times = Vec::with_capacity(reps);
    let mut verifier_times = Vec::with_capacity(reps);

    let total_runs = 1 + reps;
    for run in 0..total_runs {
        let is_warmup = run == 0;

        // Step 1: PublicInstance::new
        let (instance, dur_step1) = timed(|| PublicInstance::new(&inputs).unwrap());

        // Step 2: MainTables::build for H-FIPS
        let (_tables, dur_step2) =
            timed(|| MainTables::build(&manifest, &instance, &signatures).unwrap());

        // Verifier: StreamedPublic::prepare + pass without cache
        let (streamed, dur_prepare) = timed(|| StreamedPublic::prepare(&inputs, None).unwrap());
        let d_main = instance.geom.d_main() as usize;
        let mut rng = SplitMix64::new(77 + run as u64);
        let (r2, r3) = (point(&mut rng, d_main), point(&mut rng, d_main));
        let (_pass, dur_pass) = timed(|| streamed.pass(&r2, &r3, None));
        let dur_verifier = dur_prepare + dur_pass;

        if !is_warmup {
            step1_times.push(dur_step1);
            step2_times.push(dur_step2);
            verifier_times.push(dur_verifier);
            println!(
                "  Run {}: Step 1: {:.2?} | Step 2: {:.2?} | Verifier: {:.2?} (prepare {:.2?}, pass {:.2?})",
                run, dur_step1, dur_step2, dur_verifier, dur_prepare, dur_pass
            );
        } else {
            println!(
                "  Warmup: Step 1: {:.2?} | Step 2: {:.2?} | Verifier: {:.2?} (prepare {:.2?}, pass {:.2?})",
                dur_step1, dur_step2, dur_verifier, dur_prepare, dur_pass
            );
        }
    }

    let med_step1 = median_duration(&step1_times).unwrap_or(Duration::ZERO);
    let med_step2 = median_duration(&step2_times).unwrap_or(Duration::ZERO);
    let med_verifier = median_duration(&verifier_times).unwrap_or(Duration::ZERO);

    println!("--------------------------------------------------------------------------------");
    println!("Medians (Backend = {backend:?}):");
    println!("  Step 1 (PublicInstance::new):                         {:.2?}", med_step1);
    println!("  Step 2 (MainTables::build for H-FIPS):                 {:.2?}", med_step2);
    println!("  StreamedPublic (prepare + pass, uncached):             {:.2?}", med_verifier);
    println!("================================================================================");
}
