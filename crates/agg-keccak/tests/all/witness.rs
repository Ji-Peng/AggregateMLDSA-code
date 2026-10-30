//! The event-2 witness build of a tied part: equality of the batched builder with the
//! per-instance reference (`pack_slot` over `ChainGate::instance_rows`) and an ignored
//! timing run. Run with `--release`.

use std::time::Instant;

use agg_keccak::{
    prover::Profile,
    sponge::{self, SpongeRecord, INPUT_BITS},
    tied::TiedPart,
    witness::pack_slot,
};
use agg_mldsa_simd::{Backend, Engine};
use flock_core::test_rng::Rng;

fn random_record(rng: &mut Rng) -> SpongeRecord {
    let mut mu = [0u8; 64];
    let mut w1 = [0u8; 768];
    mu.iter_mut().for_each(|b| *b = rng.next_u32() as u8);
    w1.iter_mut().for_each(|b| *b = rng.next_u32() as u8);
    SpongeRecord { mu, w1_encode: w1 }
}

fn records(n: usize, seed: u64) -> Vec<SpongeRecord> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| random_record(&mut rng)).collect()
}

/// `honest_parts` equals the reference packing of `instance_rows` on every Keccak
/// backend, for record counts that are not multiples of the lane counts (4, 8) and
/// that cross a scatter batch (on a one-thread pool); the part's input bits do not depend on the backend.
#[test]
fn honest_parts_match_reference_packing() {
    let gate = sponge::sponge_gate();
    let engines: Vec<Engine> = [Backend::Scalar, Backend::Avx2, Backend::Avx512]
        .into_iter()
        .filter_map(Engine::new)
        .collect();
    // On one thread the witness builder scratches 128 instances at a time, so 300
    // records span three batches; on the global pool the batch grows with the threads.
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    for (n, seed) in [(1usize, 1u64), (5, 2), (13, 3), (64, 4), (300, 5)] {
        let recs = records(n, seed);
        let d = TiedPart::d_seg_for(n);
        let part = TiedPart::new(&gate, n, d, Profile::Fast);
        let inputs: Vec<Vec<u64>> = recs.iter().map(SpongeRecord::input_bits).collect();
        let want = pack_slot(&gate, part.nu(), n, |j, z, a, b| {
            gate.instance_rows(&inputs[j], z, a, b);
        });
        let (_, want_bits) = part.honest_parts_with(&engines[0], &recs);
        for e in &engines {
            for pool in [None, Some(&one)] {
                let run = || part.honest_parts_with(e, &recs);
                let (got, bits) = pool.map_or_else(run, |p| p.install(run));
                let tag = format!("{:?}, n = {n}, one thread {}", e.backend(), pool.is_some());
                assert!(got.z == want.z, "z, {tag}");
                assert!(got.a == want.a, "a, {tag}");
                assert!(got.b == want.b, "b, {tag}");
                assert!(bits == want_bits, "bits, {tag}");
            }
        }
    }
}

/// Wall time of the event-2 witness build (`TiedPart::honest_parts`) and of the
/// per-instance reference. `AGG_WITNESS_N` records (default 4,096), one warmup and
/// `AGG_WITNESS_REPS` runs (default 2).
#[test]
#[ignore = "timing run"]
fn witness_build_timing() {
    let n: usize = std::env::var("AGG_WITNESS_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);
    let reps: usize = std::env::var("AGG_WITNESS_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let gate = sponge::sponge_gate();
    let recs = records(n, 99);
    let d = TiedPart::d_seg_for(n.min(1 << 13));
    let part = TiedPart::new(&gate, n, d, Profile::Fast);
    let reference = || {
        let words = INPUT_BITS / 64;
        let mut inputs = vec![0u64; n * words];
        for (out, r) in inputs.chunks_exact_mut(words).zip(&recs) {
            r.input_bits_into(out);
        }
        pack_slot(&gate, part.nu(), n, |j, z, a, b| {
            gate.instance_rows(&inputs[j * words..(j + 1) * words], z, a, b);
        })
    };
    for label in ["honest_parts", "reference"] {
        let mut ms = Vec::new();
        for rep in 0..=reps {
            let t = Instant::now();
            if label == "honest_parts" {
                std::hint::black_box(part.honest_parts(&recs));
            } else {
                std::hint::black_box(reference());
            }
            if rep > 0 {
                ms.push(t.elapsed().as_secs_f64() * 1e3);
            }
        }
        eprintln!("witness build n = {n} {label}: {ms:.1?} ms");
    }
}
