//! Placeholder bench target (harness = false): prints the build info and the
//! median time of a trivial SHAKE256 batch, so scripts/bench.sh has a target.

use std::time::Instant;

fn main() {
    let msg = [0u8; 64];
    let inputs: Vec<&[u8]> = (0..256).map(|_| &msg[..]).collect();
    let mut samples = Vec::new();
    for _ in 0..11 {
        let t = Instant::now();
        let out = agg_keccak::aerie_keccak::shake256_64_batch(&inputs);
        std::hint::black_box(out);
        samples.push(t.elapsed().as_secs_f64() * 1e6);
    }
    let med = agg_bench::median(&mut samples).unwrap_or(f64::NAN);
    println!("{}", agg_bench::build_info());
    println!("shake256_64_batch(256 x 64 B): median {med:.1} us over 11 runs");
}
