//! Subcommands of the `agg` tool: `keygen-batch`, `sign-batch`, `aggregate`, `verify` and
//! `bench`. Batch files are those of [`crate::record`]; the aggregate is the mode-A wire format
//! ([`crate::mode_a`]).

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use agg_piop::binding::FlockProfile;
use agg_piop::manifest::Relation;
use agg_piop::public_data::KeyCache;
use agg_witness::gen;
use rayon::prelude::*;

use crate::mode_a::{
    size_lines, AggregateVerifier, Aggregator, AggregatorConfig, ProveReport, SignedRecord,
    DEFAULT_PART_LOG2, FLOCK_PROFILE,
};
use crate::record::{self, KeyEntry, RecordEntry};
use crate::{cache, rss};

/// Usage text.
pub const USAGE: &str = "\
usage: agg <command> [options]

commands:
  keygen-batch --count K [--seed S] --out KEYS
      K ML-DSA-44 key pairs from seeds derived from S (test keys: the file holds the seeds).
  sign-batch --keys KEYS --count N [--seed S] [--msg-len L] --out RECORDS
      N signed records (pure mode, empty context); record i is signed by key i mod K,
      messages are derived from S (length L, or pseudo-random in [0, 160)).
  aggregate --records RECORDS --out AGGREGATE [--part-log2 D] [--flock-profile P]
            [--flock-span on|off] [--relation R]
      Mode-A aggregate of the records (strict FIPS 204 verification of every signature);
      Flock parts of 2^D record positions (default 13, one tied part per segment); Flock
      profile P = fast, slim or slim8 (Ligerito rate 1/2, 1/4 or 1/8, all 128-bit in
      Flock's accounting; default fast); with --flock-span on (default) a
      batch of more than one segment in whole-segment parts is one tied part over all
      segments. Relation R = h-u (default: z in [-2^17, 2^17), |u| <= 190,465), h-exact
      (the FIPS 204 bounds |z| <= 130,993, |u| <= 190,465) or h-fips (also the hint:
      the extracted (c~, z, h) is a FIPS 204 signature).
  verify --records RECORDS --aggregate AGGREGATE [--cache CACHE] [--flock-profile P]
         [--relation R]
      Verify the aggregate for the records' (pk, M) (signatures are ignored), under the
      Flock profile its items name (or only under P) and under relation R (default h-u; an
      aggregate of another relation is rejected). With --cache, expanded keys are read
      from and added to the cache file. Exit status 0 on accept, 1 on reject.
  bench --count N [--keys K] [--seed S] [--warmup W] [--reps R] [--part-log2 D]
        [--flock-profile P] [--flock-span on|off] [--relation R]
      Generate N records over K keys (default N); aggregate W + R times, verify W + R times
      uncached and W + R times with cached keys (W warmup runs discarded, default W = 1,
      R = 5); print the median (and min) prove time, the verify medians, peak RSS of the
      prover and of the verifier (max over all runs), and the aggregate size per
      signature. Set AGG_GIT_REV to print the revision.

environment:
  AGG_FLOCK_THREADS=K   threads of the pool that proves the Flock parts beside the PIOP
                        (aggregate, bench; default 3/8 of the rayon pool rounded up, 12 on a
                        32-thread host; 0: one after the other; at most the logical CPUs or
                        the rayon pool; a value that is not a number warns and falls back to
                        the default).
  AGG_PIOP_S_THREADS=K  threads of the stage branch's own pool beside the Flock fork
                        (aggregate, bench; default half at 32 threads or more, 3/8 rounded
                        to nearest and at least 2 below; only with a running fork and at
                        least 4 PIOP threads; 0: the shared rayon pool). At 16 and 32
                        threads the defaults are 6 and 16, and the three pools add up to 28
                        and 60 threads.
  AGG_PIOP_THREADS=K    threads of the PIOP pool beside the Flock fork (aggregate, bench;
                        default the rayon pool; PIOP + Flock capped at the logical CPUs).
  AGG_PIOP_FORK=0       run the transcript branches one after the other (the proof is the
                        same).
  AGG_OPENING_THREADS=K threads of the pool that runs the Akita opening (aggregate, bench;
                        default one per physical core, at most the rayon pool; 0 or at least
                        the rayon pool: the rayon pool itself).
  AGG_HUGEPAGE=0        turn the huge-page advice off (memory behaviour only; the proof is
                        the same).
  AGG_ALLOC_CACHE=GiB   cap the cache of freed large blocks (default 6; 0 disables it).
  AGG_ALLOC_POISON=1    fill reused blocks with 0xA5 (a test mode for uninitialised reads).
  AGG_W1_SPLIT=0        build the level-1 weight table instead of its two factors (memory and
                        time only; the proof is the same).
  AGG_PHASE_STAMPS      (any value) print a line phase-stamp <step> <unix ns> on stderr at the
                        end of every prover step (diagnostics; the aggregate is the same).
  AGG_PARITY_EARLY      (any value) build the stages' parity tables beside the round-2
                        commitment instead of in the fork (opt-in; the aggregate is the same).";

/// A subcommand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Command {
    /// `keygen-batch`.
    KeygenBatch,
    /// `sign-batch`.
    SignBatch,
    /// `aggregate`.
    Aggregate,
    /// `verify`.
    Verify,
    /// `bench`.
    Bench,
}

/// Parses the subcommand name. Returns `None` for anything unknown.
#[must_use]
pub fn parse_command(arg: Option<&str>) -> Option<Command> {
    match arg? {
        "keygen-batch" => Some(Command::KeygenBatch),
        "sign-batch" => Some(Command::SignBatch),
        "aggregate" => Some(Command::Aggregate),
        "verify" => Some(Command::Verify),
        "bench" => Some(Command::Bench),
        _ => None,
    }
}

/// `--name value` options.
#[derive(Debug, Default)]
pub struct Options(HashMap<String, String>);

impl Options {
    /// Parse `--name value` pairs.
    ///
    /// # Errors
    /// On a stray argument, a missing value or a repeated option.
    pub fn parse(args: &[String]) -> Result<Self, String> {
        let mut map = HashMap::new();
        let mut it = args.iter();
        while let Some(arg) = it.next() {
            let name = arg
                .strip_prefix("--")
                .ok_or_else(|| format!("unexpected argument {arg:?}"))?;
            let value = it.next().ok_or_else(|| format!("--{name} needs a value"))?;
            if map.insert(name.to_string(), value.clone()).is_some() {
                return Err(format!("--{name} given twice"));
            }
        }
        Ok(Self(map))
    }

    fn take(&mut self, name: &str) -> Option<String> {
        self.0.remove(name)
    }

    fn path(&mut self, name: &str) -> Result<PathBuf, String> {
        self.take(name)
            .map(PathBuf::from)
            .ok_or_else(|| format!("--{name} is required"))
    }

    fn number<T: std::str::FromStr>(
        &mut self,
        name: &str,
        default: Option<T>,
    ) -> Result<T, String> {
        match self.take(name) {
            Some(v) => v
                .parse()
                .map_err(|_| format!("--{name}: not a number: {v:?}")),
            None => default.ok_or_else(|| format!("--{name} is required")),
        }
    }

    /// `--name fast|slim|slim8`: a mode-A Flock profile.
    fn flock_profile(&mut self, name: &str) -> Result<Option<FlockProfile>, String> {
        self.take(name)
            .map(|v| {
                parse_flock_profile(&v)
                    .ok_or_else(|| format!("--{name}: fast, slim or slim8, not {v:?}"))
            })
            .transpose()
    }

    /// `--name h-u|h-exact|h-fips`: the relation, H-u when absent.
    fn relation(&mut self, name: &str) -> Result<Relation, String> {
        match self.take(name) {
            None => Ok(Relation::Hu),
            Some(v) => Relation::parse(&v).ok_or_else(|| format!("--{name}: h-u, h-exact or h-fips, not {v:?}")),
        }
    }

    /// `--name on|off`, `default` when absent.
    fn switch(&mut self, name: &str, default: bool) -> Result<bool, String> {
        match self.take(name).as_deref() {
            None => Ok(default),
            Some("on") => Ok(true),
            Some("off") => Ok(false),
            Some(v) => Err(format!("--{name}: on or off, not {v:?}")),
        }
    }

    fn finish(self) -> Result<(), String> {
        match self.0.keys().next() {
            Some(name) => Err(format!("unknown option --{name}")),
            None => Ok(()),
        }
    }
}

/// A mode-A Flock profile by name: `fast`, `slim` or `slim8`.
#[must_use]
pub fn parse_flock_profile(name: &str) -> Option<FlockProfile> {
    match name {
        "fast" => Some(FlockProfile::Fast),
        "slim" => Some(FlockProfile::Slim),
        "slim8" => Some(FlockProfile::Slim8),
        _ => None,
    }
}

/// Exit status of a command: 0 success (or accept), 1 reject.
pub type Status = u8;

/// Run `command` with its options.
///
/// # Errors
/// On bad options, unreadable or malformed files, or an aggregator failure.
pub fn run(command: Command, args: &[String]) -> Result<Status, Box<dyn std::error::Error>> {
    let opts = Options::parse(args)?;
    match command {
        Command::KeygenBatch => keygen_batch(opts),
        Command::SignBatch => sign_batch(opts),
        Command::Aggregate => aggregate(opts),
        Command::Verify => verify(opts),
        Command::Bench => bench(opts),
    }
}

/// `K` key pairs from the seeds `gen::key_seed(seed, i)`.
#[must_use]
pub fn generate_keys(count: usize, seed: u64) -> Vec<KeyEntry> {
    (0..count as u64)
        .into_par_iter()
        .map(|i| {
            let xi = gen::key_seed(seed, i);
            let (_, pk) = gen::keypair(&xi);
            KeyEntry { seed: xi, pk }
        })
        .collect()
}

/// The message of record `i`: `msg_len` bytes, or a pseudo-random length in `[0, 160)`.
#[must_use]
pub fn message(seed: u64, i: u64, msg_len: Option<usize>) -> Vec<u8> {
    match msg_len {
        None => gen::message(seed, i),
        Some(len) => {
            let mut m = vec![0_u8; len];
            gen::derive_bytes(b"cli-msg", &[seed, i], &mut m);
            m
        }
    }
}

/// `N` records signed (hedged, randomness derived from the seed) round-robin by `keys`.
#[must_use]
pub fn sign_records(
    keys: &[KeyEntry],
    count: usize,
    seed: u64,
    msg_len: Option<usize>,
) -> Vec<RecordEntry> {
    assert!(!keys.is_empty(), "at least one key");
    let signers: Vec<_> = keys.par_iter().map(|k| gen::keypair(&k.seed)).collect();
    (0..count)
        .into_par_iter()
        .map(|i| {
            let (sk, pk) = &signers[i % signers.len()];
            let msg = message(seed, i as u64, msg_len);
            let mut rnd = [0_u8; 32];
            gen::derive_bytes(b"cli-rnd", &[seed, i as u64], &mut rnd);
            RecordEntry {
                pk: pk.clone(),
                sig: gen::sign(sk, &msg, Some(&rnd)),
                msg,
            }
        })
        .collect()
}

fn keygen_batch(mut opts: Options) -> Result<Status, Box<dyn std::error::Error>> {
    let count: usize = opts.number("count", None)?;
    let seed: u64 = opts.number("seed", Some(1))?;
    let out = opts.path("out")?;
    opts.finish()?;
    let start = Instant::now();
    let keys = generate_keys(count, seed);
    std::fs::write(&out, record::keys_to_bytes(&keys))?;
    println!(
        "wrote {count} key pairs to {} in {:.1} ms",
        out.display(),
        ms(start.elapsed())
    );
    Ok(0)
}

fn sign_batch(mut opts: Options) -> Result<Status, Box<dyn std::error::Error>> {
    let keys = record::read_keys(&opts.path("keys")?)?;
    let count: usize = opts.number("count", None)?;
    let seed: u64 = opts.number("seed", Some(1))?;
    let msg_len = opts
        .take("msg-len")
        .map(|v| v.parse::<usize>())
        .transpose()?;
    let out = opts.path("out")?;
    opts.finish()?;
    if keys.is_empty() {
        return Err("the key file is empty".into());
    }
    let start = Instant::now();
    let records = sign_records(&keys, count, seed, msg_len);
    std::fs::write(&out, record::records_to_bytes(&records))?;
    println!(
        "wrote {count} signed records ({} keys) to {} in {:.1} ms",
        keys.len(),
        out.display(),
        ms(start.elapsed())
    );
    Ok(0)
}

fn signed(records: &[RecordEntry]) -> Vec<SignedRecord<'_>> {
    records
        .iter()
        .map(|r| (r.pk.as_slice(), r.msg.as_slice(), r.sig.as_slice()))
        .collect()
}

fn statements(records: &[RecordEntry]) -> Vec<(&[u8], &[u8])> {
    records
        .iter()
        .map(|r| (r.pk.as_slice(), r.msg.as_slice()))
        .collect()
}

fn aggregate(mut opts: Options) -> Result<Status, Box<dyn std::error::Error>> {
    let records = record::read_records(&opts.path("records")?)?;
    let out = opts.path("out")?;
    let part_log2: u32 = opts.number("part-log2", Some(DEFAULT_PART_LOG2))?;
    let profile = opts.flock_profile("flock-profile")?.unwrap_or(FLOCK_PROFILE);
    let span = opts.switch("flock-span", true)?;
    let relation = opts.relation("relation")?;
    opts.finish()?;
    let aggregator = Aggregator::new(AggregatorConfig {
        part_log2,
        profile,
        span,
        relation,
    });
    let aggregate = run_large(move || aggregator.aggregate(&signed(&records)))?;
    std::fs::write(&out, &aggregate.bytes)?;
    println!("{}", aggregate.report);
    println!("wrote {} B to {}", aggregate.bytes.len(), out.display());
    Ok(0)
}

fn verify(mut opts: Options) -> Result<Status, Box<dyn std::error::Error>> {
    let records = record::read_records(&opts.path("records")?)?;
    let bytes = std::fs::read(opts.path("aggregate")?)?;
    let cache_path = opts.take("cache").map(PathBuf::from);
    let require = opts.flock_profile("flock-profile")?;
    let relation = opts.relation("relation")?;
    opts.finish()?;
    let mut key_cache = match &cache_path {
        Some(p) => Some(cache::load(p)?),
        None => None,
    };
    let before = key_cache.as_ref().map_or(0, KeyCache::len);
    let (report, key_cache) = run_large(move || {
        let verifier = match require {
            Some(p) => AggregateVerifier::new().require(p),
            None => AggregateVerifier::new(),
        }
        .relation(relation);
        let report = verifier.verify(&statements(&records), &bytes, key_cache.as_mut());
        (report, key_cache)
    });
    println!("{report}");
    if let (Some(p), Some(c)) = (&cache_path, &key_cache) {
        if c.len() != before {
            cache::save(p, c)?;
            println!(
                "key cache {}: {} keys ({} new)",
                p.display(),
                c.len(),
                c.len() - before
            );
        }
    }
    Ok(if report.accepted() { 0 } else { 1 })
}

/// Default warmup runs of `bench` (discarded).
pub const DEFAULT_WARMUP: usize = 1;
/// Default timed runs of `bench`.
pub const DEFAULT_REPS: usize = 5;

/// Results of the `bench` subcommand.
#[derive(Clone, Debug)]
pub struct BenchResult {
    /// Warmup runs per measurement (discarded).
    pub warmup: usize,
    /// The prover's report of the median timed run.
    pub prove: ProveReport,
    /// Prover wall times of the timed runs.
    pub prove_times: Vec<Duration>,
    /// Prover peak RSS (KiB) over all prover runs (warmup included), if measurable.
    pub prove_peak_kib: Option<u64>,
    /// Verifier wall times without a key cache (timed runs).
    pub verify_uncached: Vec<Duration>,
    /// Verifier wall times with every key cached (timed runs).
    pub verify_cached: Vec<Duration>,
    /// Verifier peak RSS (KiB) over all verifier runs, if measurable.
    pub verify_peak_kib: Option<u64>,
    /// The step report of the last uncached run.
    pub uncached_report: String,
    /// The step report of the last cached run.
    pub cached_report: String,
}

/// Median of a non-empty sample.
#[must_use]
pub fn median(samples: &[Duration]) -> Duration {
    let mut s = samples.to_vec();
    s.sort();
    let n = s.len();
    assert!(n > 0, "a non-empty sample");
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2
    }
}

/// Index of the median sample (the lower middle one for an even count).
#[must_use]
pub fn median_index(samples: &[Duration]) -> usize {
    assert!(!samples.is_empty(), "a non-empty sample");
    let mut order: Vec<usize> = (0..samples.len()).collect();
    order.sort_by_key(|&i| samples[i]);
    order[(samples.len() - 1) / 2]
}

/// Run `run` `warmup + reps` times and return the results of the last `reps` runs (the
/// warmup runs are discarded). Stops at the first error.
///
/// # Errors
/// The first error of `run`.
pub fn timed_runs<T, E>(
    warmup: usize,
    reps: usize,
    mut run: impl FnMut() -> Result<T, E>,
) -> Result<Vec<T>, E> {
    for _ in 0..warmup {
        run()?;
    }
    (0..reps).map(|_| run()).collect()
}

/// Aggregate `records` `warmup + reps` times, then verify the aggregate of a timed run
/// `warmup + reps` times uncached and `warmup + reps` times with cached keys (after one
/// extra run that fills the cache); warmup runs are discarded. `reps` is at least 1.
///
/// # Errors
/// On an aggregator failure or a rejected aggregate.
pub fn bench_records(
    records: &[RecordEntry],
    config: AggregatorConfig,
    warmup: usize,
    reps: usize,
) -> Result<BenchResult, String> {
    let reps = reps.max(1);
    let aggregator = Aggregator::new(config);
    let signed = signed(records);
    agg_piop::alloc::release_cache();
    let reset = rss::reset_peak();
    let mut bytes = Vec::new();
    let proves = timed_runs(warmup, reps, || {
        // Start each prove without blocks cached by the previous one, as a new
        // process would.
        agg_piop::alloc::release_cache();
        let aggregate = aggregator.aggregate(&signed).map_err(|e| e.to_string())?;
        // Every run's aggregate has the same size; keep the latest (a timed run's at the end).
        bytes = aggregate.bytes;
        Ok::<_, String>(aggregate.report)
    })?;
    let prove_peak_kib = rss::peak_kib().filter(|_| reset);
    drop(aggregator);
    let prove_times: Vec<Duration> = proves.iter().map(|r| r.total).collect();
    let prove = proves[median_index(&prove_times)].clone();
    drop(proves);
    // The verifier's peak must not count blocks the prover left in the cache.
    agg_piop::alloc::release_cache();
    let reset = rss::reset_peak();
    let verifier = AggregateVerifier::new()
        .require(config.profile)
        .relation(config.relation);
    let statements = statements(records);
    let mut uncached_report = String::new();
    let verify_uncached = timed_runs(warmup, reps, || {
        let report = verifier.verify(&statements, &bytes, None);
        if !report.accepted() {
            return Err(format!("the honest aggregate was rejected:\n{report}"));
        }
        uncached_report = report.to_string();
        Ok(report.total())
    })?;
    let mut key_cache = KeyCache::default();
    let fill = verifier.verify(&statements, &bytes, Some(&mut key_cache));
    if !fill.accepted() {
        return Err(format!("the honest aggregate was rejected:\n{fill}"));
    }
    let mut cached_report = String::new();
    let verify_cached = timed_runs(warmup, reps, || {
        let report = verifier.verify(&statements, &bytes, Some(&mut key_cache));
        if !report.accepted() {
            return Err(format!("the honest aggregate was rejected:\n{report}"));
        }
        cached_report = report.to_string();
        Ok(report.total())
    })?;
    let verify_peak_kib = rss::peak_kib().filter(|_| reset);
    Ok(BenchResult {
        warmup,
        prove,
        prove_times,
        prove_peak_kib,
        verify_uncached,
        verify_cached,
        verify_peak_kib,
        uncached_report,
        cached_report,
    })
}

fn bench(mut opts: Options) -> Result<Status, Box<dyn std::error::Error>> {
    let count: usize = opts.number("count", None)?;
    let n_keys: usize = opts.number("keys", Some(count))?;
    let seed: u64 = opts.number("seed", Some(1))?;
    let warmup: usize = opts.number("warmup", Some(DEFAULT_WARMUP))?;
    let reps: usize = opts.number("reps", Some(DEFAULT_REPS))?;
    let part_log2: u32 = opts.number("part-log2", Some(DEFAULT_PART_LOG2))?;
    let profile = opts.flock_profile("flock-profile")?.unwrap_or(FLOCK_PROFILE);
    let span = opts.switch("flock-span", true)?;
    let relation = opts.relation("relation")?;
    opts.finish()?;
    if n_keys == 0 || count == 0 {
        return Err("--count and --keys must be positive".into());
    }
    if reps == 0 {
        return Err("--reps must be positive".into());
    }
    let start = Instant::now();
    let keys = generate_keys(n_keys.min(count), seed);
    let records = sign_records(&keys, count, seed, None);
    let generated = start.elapsed();
    let result = run_large(move || {
        bench_records(
            &records,
            AggregatorConfig {
                part_log2,
                profile,
                span,
                relation,
            },
            warmup,
            reps,
        )
    })?;
    let mib = |k: Option<u64>| {
        k.map_or_else(
            || "n/a".to_string(),
            |k| format!("{:.0} MiB", k as f64 / 1024.0),
        )
    };
    let protocol = format!("median of {reps}, {warmup} warmup");
    let mldsa_engine = agg_mldsa_simd::engine();
    let mldsa_backend = match mldsa_engine.backend() {
        agg_mldsa_simd::dispatch::Backend::Scalar => "scalar",
        agg_mldsa_simd::dispatch::Backend::Avx2 => "avx2",
        agg_mldsa_simd::dispatch::Backend::Avx512 => {
            if mldsa_engine.has_vbmi() {
                "avx512+vbmi"
            } else {
                "avx512"
            }
        }
    };
    println!(
        "agg bench: N = {count}, {} keys, part_log2 {part_log2}, Flock profile {profile:?}, span {span}, relation {relation}, warmup {warmup}, reps {reps}, revision {}, avx512 feature {}, aerie backend {}, mldsa backend {}",
        n_keys.min(count),
        std::env::var("AGG_GIT_REV").unwrap_or_else(|_| "unknown (set AGG_GIT_REV)".into()),
        agg_piop::avx512_enabled(),
        agg_piop::aerie_simd_backend(),
        mldsa_backend
    );
    println!(
        "record generation {:.1} ms (not part of the measurements)",
        ms(generated)
    );
    println!("median prover run ({protocol}):");
    println!("{}", result.prove);
    println!("last uncached verifier run:");
    println!("{}", result.uncached_report);
    println!("last cached verifier run:");
    println!("{}", result.cached_report);
    let flock = result.prove.timings.keccak;
    println!("summary:");
    println!(
        "  prove                    {:>10.1} ms ({protocol}; min {:.1} ms)",
        ms(median(&result.prove_times)),
        ms(*result.prove_times.iter().min().expect("reps >= 1"))
    );
    let t = &result.prove.timings;
    println!("    opening pool: {} threads", t.opening_threads);
    if t.flock_threads > 0 {
        // Fork-join: the Flock proofs overlap the PIOP steps.
        println!(
            "    median run: Flock parts {:.1} ms (proofs {:.1} ms on {} threads beside the PIOP), \
             fork-join wall {:.1} ms, outside the fork {:.1} ms",
            ms(flock),
            ms(t.flock_prove),
            t.flock_threads,
            ms(t.fork_join),
            ms(result.prove.total.saturating_sub(t.fork_join))
        );
    } else {
        println!(
            "    median run: Flock parts {:.1} ms, everything else {:.1} ms",
            ms(flock),
            ms(result.prove.total.saturating_sub(flock))
        );
    }
    println!(
        "  verify, uncached keys    {:>10.1} ms ({protocol})",
        ms(median(&result.verify_uncached))
    );
    println!(
        "  verify, cached keys      {:>10.1} ms ({protocol})",
        ms(median(&result.verify_cached))
    );
    println!(
        "  prover peak RSS          {:>13} (max over all runs)",
        mib(result.prove_peak_kib)
    );
    println!(
        "  verifier peak RSS        {:>13} (max over all runs)",
        mib(result.verify_peak_kib)
    );
    println!(
        "  aggregate                {:>10} B, {:.1} B per signature",
        result.prove.bytes(),
        result.prove.bytes_per_signature()
    );
    println!("{}", size_lines(&result.prove.size, count));
    Ok(0)
}

/// Run `f` on a thread with a large stack (the PIOP prover recurses deeply at large `N`).
pub fn run_large<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("worker thread")
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn parses_commands_and_options() {
        assert_eq!(
            parse_command(Some("keygen-batch")),
            Some(Command::KeygenBatch)
        );
        assert_eq!(parse_command(Some("sign-batch")), Some(Command::SignBatch));
        assert_eq!(parse_command(Some("aggregate")), Some(Command::Aggregate));
        assert_eq!(parse_command(Some("verify")), Some(Command::Verify));
        assert_eq!(parse_command(Some("bench")), Some(Command::Bench));
        assert_eq!(parse_command(Some("prove")), None);
        assert_eq!(parse_command(None), None);
        let mut o = Options::parse(&strings(&["--count", "5", "--out", "x"])).unwrap();
        assert_eq!(o.number::<usize>("count", None), Ok(5));
        assert_eq!(o.number::<u64>("seed", Some(9)), Ok(9));
        assert!(o.number::<usize>("reps", None).is_err());
        assert_eq!(o.path("out").unwrap(), PathBuf::from("x"));
        assert!(o.finish().is_ok());
        assert!(Options::parse(&strings(&["count"])).is_err());
        assert!(Options::parse(&strings(&["--count"])).is_err());
        assert!(Options::parse(&strings(&["--a", "1", "--a", "2"])).is_err());
        let o = Options::parse(&strings(&["--bogus", "1"])).unwrap();
        assert!(o.finish().is_err());
        let mut o = Options::parse(&strings(&["--flock-profile", "slim"])).unwrap();
        assert_eq!(o.flock_profile("flock-profile"), Ok(Some(FlockProfile::Slim)));
        assert_eq!(o.flock_profile("flock-profile"), Ok(None));
        let mut o = Options::parse(&strings(&["--flock-profile", "secure"])).unwrap();
        assert!(o.flock_profile("flock-profile").is_err());
        assert_eq!(parse_flock_profile("fast"), Some(FlockProfile::Fast));
        assert_eq!(parse_flock_profile("slim8"), Some(FlockProfile::Slim8));
        let mut o = Options::parse(&strings(&["--relation", "h-exact"])).unwrap();
        assert_eq!(o.relation("relation"), Ok(Relation::HExact));
        assert_eq!(o.relation("relation"), Ok(Relation::Hu));
        let mut o = Options::parse(&strings(&["--relation", "h-fips"])).unwrap();
        assert_eq!(o.relation("relation"), Ok(Relation::HFips));
        let mut o = Options::parse(&strings(&["--relation", "fips"])).unwrap();
        assert!(o.relation("relation").is_err());
        let mut o = Options::parse(&strings(&["--flock-span", "off", "--x", "maybe"])).unwrap();
        assert_eq!(o.switch("flock-span", true), Ok(false));
        assert_eq!(o.switch("flock-span", true), Ok(true));
        assert!(o.switch("x", true).is_err());
        assert_eq!(
            median(&[
                Duration::from_millis(3),
                Duration::from_millis(1),
                Duration::from_millis(2)
            ]),
            Duration::from_millis(2)
        );
        assert_eq!(
            median(&[Duration::from_millis(4), Duration::from_millis(2)]),
            Duration::from_millis(3)
        );
    }

    #[test]
    fn bench_options_default_to_one_warmup_and_five_reps() {
        let mut o = Options::parse(&strings(&["--count", "4"])).unwrap();
        assert_eq!(o.number("warmup", Some(DEFAULT_WARMUP)), Ok(1usize));
        assert_eq!(o.number("reps", Some(DEFAULT_REPS)), Ok(5usize));
        let mut o = Options::parse(&strings(&["--warmup", "0", "--reps", "2"])).unwrap();
        assert_eq!(o.number("warmup", Some(DEFAULT_WARMUP)), Ok(0usize));
        assert_eq!(o.number("reps", Some(DEFAULT_REPS)), Ok(2usize));
        assert!(o.finish().is_ok());
    }

    #[test]
    fn warmup_runs_are_excluded() {
        let ms = Duration::from_millis;
        // Runs 0 and 1 are warmup; only runs 2..=6 are kept.
        let mut calls = 0u64;
        let kept = timed_runs(2, 5, || {
            calls += 1;
            Ok::<_, ()>(calls)
        })
        .unwrap();
        assert_eq!(calls, 7);
        assert_eq!(kept, vec![3, 4, 5, 6, 7]);
        // A slow warmup run does not reach the median.
        let mut samples = vec![ms(1000), ms(5), ms(3), ms(4)].into_iter();
        let kept = timed_runs(1, 3, || Ok::<_, ()>(samples.next().unwrap())).unwrap();
        assert_eq!(kept, vec![ms(5), ms(3), ms(4)]);
        assert_eq!(median(&kept), ms(4));
        assert_eq!(kept[median_index(&kept)], ms(4));
        assert_eq!(median_index(&[ms(4), ms(2), ms(9), ms(1)]), 1);
        assert_eq!(timed_runs(0, 0, || Err::<(), _>("never")), Ok(vec![]));
        assert_eq!(timed_runs(1, 3, || Err::<(), _>("fail")), Err("fail"));
    }

    #[test]
    fn generated_keys_and_records_are_deterministic_and_valid() {
        let keys = generate_keys(2, 7);
        assert_eq!(keys, generate_keys(2, 7));
        assert_ne!(keys[0].pk, keys[1].pk);
        let records = sign_records(&keys, 3, 7, Some(10));
        assert_eq!(records[0].pk, records[2].pk);
        assert_eq!(records[1].msg.len(), 10);
        for r in &records {
            assert!(gen::crate_verify(&r.pk, &r.msg, &r.sig));
        }
        let free = sign_records(&keys, 2, 8, None);
        assert!(free.iter().all(|r| r.msg.len() < 160));
    }
}
