//! Mode-B benchmark (criterion-free): prove time, verify time
//! (uncached and with cached expanded keys), peak RSS of the prover and of the verifier
//! (`VmHWM`), and the aggregate size, at 256, 512 and 1,024 signatures by default.
//!
//! Usage (never on CPU 2):
//!
//! ```text
//! taskset -c 0,1,3-15 cargo bench -p agg-bench --bench mode_b -- [N ...]
//!     [--warmup W] [--reps R] [--shared S]
//! ```
//!
//! Protocol: `W` warmup runs (default 1, discarded), then `R` timed runs (default 5),
//! medians reported, for the prover and for each verifier mode. Peak RSS is the maximum over
//! all runs, warmup included.
//!
//! Every size runs in its own prover process, which hands the proof to its own verifier
//! process over a pipe, so each peak RSS belongs to one role. Records: `N - S` under their
//! own keys and `S` (default `N / 32`) under three shared keys; the bridge matrices are
//! derived from the bridge seed (`SeededMatrices`), so prover and verifier
//! times include the derivation (verifier step `matrices`). Times are medians over the timed
//! runs. With `AGG_PROOF_OUT=<dir>` the prover child writes the aggregate of its last run
//! to `<dir>/mode_b-<N>.bin` (byte-identity checks across builds). `AGG_RELATION=h-exact`
//! proves and verifies relation H-exact (default `h-u`); the relation is
//! printed in the header and in every row.

use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::time::Duration;

use agg_bench::harness::{git_revision, median_duration, timed, Batch};
use agg_bench::memory::{mib, peak_rss_kib, reset_peak, rss_kib};
use agg_piop::manifest::Relation;
use agg_piop::mode_b::{prove_mode_b, verify_mode_b};
use agg_piop::prover::{ProverInputs, SeededMatrices};
use agg_piop::public_data::KeyCache;
use agg_piop::verifier::VerifyTimings;
use agg_witness::instance::PublicInstance;

const SEED: u64 = 0x00b0_0b0b;

struct Options {
    sizes: Vec<usize>,
    warmup: usize,
    reps: usize,
    shared: Option<usize>,
    role: Role,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    Driver,
    Prover(usize),
    Verifier(usize),
}

fn parse_options() -> Options {
    let mut options = Options {
        sizes: Vec::new(),
        warmup: 1,
        reps: 5,
        shared: None,
        role: Role::Driver,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = |name: &str| -> usize {
            args.next()
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| panic!("{name} takes a number"))
        };
        match arg.as_str() {
            "--warmup" => options.warmup = value("--warmup"),
            "--reps" => options.reps = value("--reps").max(1),
            "--shared" => options.shared = Some(value("--shared")),
            "--prover-child" => options.role = Role::Prover(value("--prover-child")),
            "--verifier-child" => options.role = Role::Verifier(value("--verifier-child")),
            // `cargo bench` appends `--bench`; ignore other flags too.
            other if other.starts_with("--") => {}
            other => options.sizes.push(other.parse().expect("a batch size")),
        }
    }
    if options.sizes.is_empty() {
        options.sizes = vec![256, 512, 1024];
    }
    options
}

impl Options {
    fn shared_for(&self, n: usize) -> usize {
        self.shared.unwrap_or(n / 32).min(n)
    }

    fn child_args(&self, n: usize) -> Vec<String> {
        vec![
            "--warmup".into(),
            self.warmup.to_string(),
            "--reps".into(),
            self.reps.to_string(),
            "--shared".into(),
            self.shared_for(n).to_string(),
        ]
    }
}

fn batch(options: &Options, n: usize) -> Batch {
    let shared = options.shared_for(n);
    Batch::generate(SEED ^ n as u64, n - shared, shared)
}

fn bridge() -> SeededMatrices {
    SeededMatrices
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// Field-wise medians of step timings.
fn median_steps(samples: &[VerifyTimings]) -> Vec<(&'static str, Duration)> {
    let names = samples[0].steps();
    let mut out: Vec<(&'static str, Duration)> = names
        .iter()
        .enumerate()
        .map(|(k, (name, _))| {
            let column: Vec<Duration> = samples.iter().map(|s| s.steps()[k].1).collect();
            (*name, median_duration(&column).unwrap_or_default())
        })
        .collect();
    let totals: Vec<Duration> = samples.iter().map(|s| s.total).collect();
    out.push(("total", median_duration(&totals).unwrap_or_default()));
    out
}

/// The relation of this run (`AGG_RELATION`, default H-u); prover and verifier children
/// inherit the environment.
fn relation() -> Relation {
    std::env::var("AGG_RELATION").map_or(Relation::Hu, |v| {
        Relation::parse(&v).unwrap_or_else(|| panic!("AGG_RELATION: h-u, h-exact or h-fips, not {v:?}"))
    })
}

fn steps_line(steps: &[(&'static str, Duration)]) -> String {
    steps
        .iter()
        .map(|(name, d)| format!("{name} {:.2}", ms(*d)))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The verifier process: reads the proof from stdin, verifies uncached and cached.
fn run_verifier(options: &Options, n: usize) {
    let batch = batch(options, n);
    let public = batch.public();
    let mut bytes = Vec::new();
    std::io::stdin()
        .read_to_end(&mut bytes)
        .expect("proof on stdin");
    let bridge = bridge();
    let reset = reset_peak();
    let base = rss_kib().unwrap_or(0);
    let mut uncached = Vec::new();
    for run in 0..options.warmup + options.reps {
        let report = verify_mode_b(&public, &bytes, &bridge, relation(), None);
        assert!(report.accepted(), "verify: {:?}", report.outcome);
        if run >= options.warmup {
            uncached.push(report.timings);
        }
    }
    let peak_uncached = peak_rss_kib().unwrap_or(0);
    let mut cache = KeyCache::default();
    assert!(verify_mode_b(&public, &bytes, &bridge, relation(), Some(&mut cache)).accepted());
    let mut cached = Vec::new();
    for run in 0..options.warmup + options.reps {
        let report = verify_mode_b(&public, &bytes, &bridge, relation(), Some(&mut cache));
        let verified = report.outcome.expect("verify with cached keys");
        assert_eq!(verified.expanded_keys, 0);
        if run >= options.warmup {
            cached.push(report.timings);
        }
    }
    let peak_cached = peak_rss_kib().unwrap_or(0);
    let uncached = median_steps(&uncached);
    let cached = median_steps(&cached);
    println!(
        "  verifier uncached (ms, medians of {}, {} warmup): {}",
        options.reps,
        options.warmup,
        steps_line(&uncached)
    );
    println!(
        "  verifier cached keys (ms, medians of {}, {} warmup): {}",
        options.reps,
        options.warmup,
        steps_line(&cached)
    );
    println!(
        "VERIFY verify_uncached_ms={:.2} verify_cached_ms={:.2} verifier_peak_mib={:.1} \
         verifier_cached_peak_mib={:.1} verifier_base_mib={:.1} peak_reset={}",
        ms(uncached.last().unwrap().1),
        ms(cached.last().unwrap().1),
        mib(peak_uncached),
        mib(peak_cached),
        mib(base),
        reset
    );
}

/// The prover process: generates the batch, proves, hands the proof to a verifier process.
fn run_prover(options: &Options, n: usize) {
    let (batch, generation) = timed(|| batch(options, n));
    // Mode A times the instance inside its prove; it is printed here so the two compare.
    let (instance, instance_time) = timed(|| PublicInstance::from_records(&batch.records).expect("instance"));
    let inputs = batch.inputs();
    let signatures = batch.signatures();
    let bridge = bridge();
    let reset = reset_peak();
    let base = rss_kib().unwrap_or(0);
    let mut times = Vec::new();
    let mut steps = Vec::new();
    let mut last = None;
    for run in 0..options.warmup + options.reps {
        let output = prove_mode_b(
            &ProverInputs {
                instance: &instance,
                records: &inputs,
                signatures: &signatures,
                relation: relation(),
            },
            &bridge,
        )
        .expect("prove");
        if run >= options.warmup {
            times.push(output.timings.total);
            steps.push(output.timings.to_string());
        }
        last = Some(output);
    }
    let prover_peak = peak_rss_kib().unwrap_or(0);
    let output = last.expect("one run");
    let bytes = output.proof.to_bytes();
    let size = output.proof.size();
    // `AGG_PROOF_OUT=<dir>`: keep the aggregate for a byte-identity check across builds.
    if let Some(dir) = std::env::var_os("AGG_PROOF_OUT") {
        let path = std::path::Path::new(&dir).join(format!("mode_b-{n}.bin"));
        std::fs::write(&path, &bytes).expect("write the aggregate");
        println!("  aggregate written to {}", path.display());
    }
    println!(
        "  N = {n}: {} records under own keys, {} under 3 shared keys; generation {:.2} s",
        n - options.shared_for(n),
        options.shared_for(n),
        generation.as_secs_f64()
    );
    let median_run = {
        let mut order: Vec<usize> = (0..times.len()).collect();
        order.sort_by_key(|&i| times[i]);
        order[(times.len() - 1) / 2]
    };
    println!(
        "  prover steps (ms, median run of {}, {} warmup): {}",
        options.reps, options.warmup, steps[median_run]
    );
    println!(
        "  prove (s): median {:.3}, min {:.3}; public instance (once, before the runs) {:.3}",
        median_duration(&times).unwrap_or_default().as_secs_f64(),
        times.iter().min().copied().unwrap_or_default().as_secs_f64(),
        instance_time.as_secs_f64()
    );
    println!(
        "  aggregate {} B = header {} + c~ {} + public bits {} + Akita commitments {} + final vectors {} \
         + Keccak proofs {} + lookup {} + PIOP {} + opening {}",
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
    let manifest = agg_piop::manifest::CommitmentManifest::for_relation(n, relation()).expect("manifest");
    #[allow(clippy::cast_precision_loss)]
    let per_signature = manifest.total_logical_cells() as f64 / n as f64;
    println!(
        "  committed cells: {} in {} polynomials ({:.1} per signature)",
        manifest.total_logical_cells(),
        manifest.total_polynomials(),
        per_signature
    );
    drop(output);
    let mut child = Command::new(std::env::current_exe().expect("own path"))
        .arg("--verifier-child")
        .arg(n.to_string())
        .args(options.child_args(n))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("verifier process");
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(&bytes)
        .expect("send proof");
    let out = child.wait_with_output().expect("verifier output");
    assert!(out.status.success(), "verifier process failed");
    let text = String::from_utf8_lossy(&out.stdout);
    let mut verify_fields = String::new();
    for line in text.lines() {
        match line.strip_prefix("VERIFY ") {
            Some(fields) => verify_fields = fields.to_string(),
            None => println!("{line}"),
        }
    }
    let prove = median_duration(&times).unwrap_or_default();
    println!(
        "RESULT n={n} prove_s={:.3} prover_peak_mib={:.1} prover_base_mib={:.1} proof_bytes={} \
         public_bits_bytes={} without_bits_bytes={} {verify_fields} prover_peak_reset={reset}",
        prove.as_secs_f64(),
        mib(prover_peak),
        mib(base),
        bytes.len(),
        size.keccak_commitments,
        bytes.len() - size.keccak_commitments
    );
}

fn field<'a>(line: &'a str, key: &str) -> &'a str {
    line.split_whitespace()
        .find_map(|kv| kv.strip_prefix(key)?.strip_prefix('='))
        .unwrap_or("?")
}

fn main() {
    let options = parse_options();
    match options.role {
        Role::Prover(n) => return run_prover(&options, n),
        Role::Verifier(n) => return run_verifier(&options, n),
        Role::Driver => {}
    }
    println!("mode-B benchmark, git {}, relation {}", git_revision(), relation());
    println!(
        "{}, threads {}, warmup {}, reps {} (medians over the timed runs)",
        agg_bench::build_info(),
        std::thread::available_parallelism().map_or(0, std::num::NonZero::get),
        options.warmup,
        options.reps
    );
    let mut rows = Vec::new();
    for &n in &options.sizes {
        let out = Command::new(std::env::current_exe().expect("own path"))
            .arg("--prover-child")
            .arg(n.to_string())
            .args(options.child_args(n))
            .stdout(Stdio::piped())
            .output()
            .expect("prover process");
        assert!(out.status.success(), "prover process for N = {n} failed");
        let text = String::from_utf8_lossy(&out.stdout).to_string();
        for line in text.lines() {
            if let Some(result) = line.strip_prefix("RESULT ") {
                rows.push(result.to_string());
            } else {
                println!("{line}");
            }
        }
    }
    println!();
    println!(
        "| relation | N | prove (s) | verify uncached (ms) | verify cached (ms) | prover peak RSS (MiB) \
         | prover RSS before proving (MiB) | verifier peak RSS (MiB) | verifier RSS before verifying (MiB) \
         | aggregate (B) | of which public bits (B) | without bits (B) |"
    );
    println!("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
    for row in &rows {
        println!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            relation(),
            field(row, "n"),
            field(row, "prove_s"),
            field(row, "verify_uncached_ms"),
            field(row, "verify_cached_ms"),
            field(row, "prover_peak_mib"),
            field(row, "prover_base_mib"),
            field(row, "verifier_peak_mib"),
            field(row, "verifier_base_mib"),
            field(row, "proof_bytes"),
            field(row, "public_bits_bytes"),
            field(row, "without_bits_bytes"),
        );
    }
}
