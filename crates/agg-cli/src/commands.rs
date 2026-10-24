//! Subcommands of the `agg` tool: `keygen-batch`, `sign-batch`, `aggregate`, `verify` and
//! `bench` (docs/spec.md Section 14, milestone M3). Batch files are those of
//! [`crate::record`]; the aggregate is the mode-A wire format ([`crate::mode_a`]).

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use agg_piop::public_data::KeyCache;
use agg_witness::gen;
use rayon::prelude::*;

use crate::mode_a::{
    size_lines, AggregateVerifier, Aggregator, AggregatorConfig, ProveReport, SignedRecord,
    DEFAULT_PART_LOG2,
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
  aggregate --records RECORDS --out AGGREGATE [--part-log2 D]
      Mode-A aggregate of the records (strict FIPS 204 verification of every signature);
      Flock parts of 2^D record positions (default 9).
  verify --records RECORDS --aggregate AGGREGATE [--cache CACHE]
      Verify the aggregate for the records' (pk, M) (signatures are ignored). With
      --cache, expanded keys are read from and added to the cache file. Exit status 0 on
      accept, 1 on reject.
  bench --count N [--keys K] [--seed S] [--reps R] [--part-log2 D]
      Generate N records over K keys (default N), aggregate once, verify R times uncached
      and R times with cached keys (default 3); print medians, peak RSS of the prover and
      of the verifier, and the aggregate size per signature. Set AGG_GIT_REV to print the
      revision.";

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

    fn finish(self) -> Result<(), String> {
        match self.0.keys().next() {
            Some(name) => Err(format!("unknown option --{name}")),
            None => Ok(()),
        }
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
    opts.finish()?;
    let aggregator = Aggregator::new(AggregatorConfig { part_log2 });
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
    opts.finish()?;
    let mut key_cache = match &cache_path {
        Some(p) => Some(cache::load(p)?),
        None => None,
    };
    let before = key_cache.as_ref().map_or(0, KeyCache::len);
    let (report, key_cache) = run_large(move || {
        let verifier = AggregateVerifier::new();
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

/// Results of the `bench` subcommand.
#[derive(Clone, Debug)]
pub struct BenchResult {
    /// The prover's report.
    pub prove: ProveReport,
    /// Prover peak RSS (KiB), if measurable.
    pub prove_peak_kib: Option<u64>,
    /// Verifier wall times without a key cache.
    pub verify_uncached: Vec<Duration>,
    /// Verifier wall times with every key cached.
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

/// Aggregate `records` once and verify the aggregate `reps` times uncached and `reps` times
/// with cached keys.
///
/// # Errors
/// On an aggregator failure or a rejected aggregate.
pub fn bench_records(
    records: &[RecordEntry],
    config: AggregatorConfig,
    reps: usize,
) -> Result<BenchResult, String> {
    let reps = reps.max(1);
    let aggregator = Aggregator::new(config);
    let reset = rss::reset_peak();
    let aggregate = aggregator
        .aggregate(&signed(records))
        .map_err(|e| e.to_string())?;
    let prove_peak_kib = rss::peak_kib().filter(|_| reset);
    drop(aggregator);
    let bytes = aggregate.bytes.clone();
    let prove = aggregate.report.clone();
    drop(aggregate);
    let reset = rss::reset_peak();
    let verifier = AggregateVerifier::new();
    let statements = statements(records);
    let mut verify_uncached = Vec::with_capacity(reps);
    let mut uncached_report = String::new();
    for _ in 0..reps {
        let report = verifier.verify(&statements, &bytes, None);
        if !report.accepted() {
            return Err(format!("the honest aggregate was rejected:\n{report}"));
        }
        verify_uncached.push(report.total());
        uncached_report = report.to_string();
    }
    let mut key_cache = KeyCache::default();
    let warm = verifier.verify(&statements, &bytes, Some(&mut key_cache));
    if !warm.accepted() {
        return Err(format!("the honest aggregate was rejected:\n{warm}"));
    }
    let mut verify_cached = Vec::with_capacity(reps);
    let mut cached_report = String::new();
    for _ in 0..reps {
        let report = verifier.verify(&statements, &bytes, Some(&mut key_cache));
        if !report.accepted() {
            return Err(format!("the honest aggregate was rejected:\n{report}"));
        }
        verify_cached.push(report.total());
        cached_report = report.to_string();
    }
    let verify_peak_kib = rss::peak_kib().filter(|_| reset);
    Ok(BenchResult {
        prove,
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
    let reps: usize = opts.number("reps", Some(3))?;
    let part_log2: u32 = opts.number("part-log2", Some(DEFAULT_PART_LOG2))?;
    opts.finish()?;
    if n_keys == 0 || count == 0 {
        return Err("--count and --keys must be positive".into());
    }
    let start = Instant::now();
    let keys = generate_keys(n_keys.min(count), seed);
    let records = sign_records(&keys, count, seed, None);
    let generated = start.elapsed();
    let result = run_large(move || bench_records(&records, AggregatorConfig { part_log2 }, reps))?;
    let mib = |k: Option<u64>| {
        k.map_or_else(
            || "n/a".to_string(),
            |k| format!("{:.0} MiB", k as f64 / 1024.0),
        )
    };
    println!(
        "agg bench: N = {count}, {} keys, part_log2 {part_log2}, reps {reps}, revision {}, avx512 feature {}, aerie backend {}",
        n_keys.min(count),
        std::env::var("AGG_GIT_REV").unwrap_or_else(|_| "unknown (set AGG_GIT_REV)".into()),
        agg_piop::avx512_enabled(),
        agg_piop::aerie_simd_backend()
    );
    println!(
        "record generation {:.1} ms (not part of the measurements)",
        ms(generated)
    );
    println!("{}", result.prove);
    println!("{}", result.uncached_report);
    println!("{}", result.cached_report);
    println!("summary:");
    println!(
        "  prove                    {:>10.1} ms (one run)",
        ms(result.prove.total)
    );
    println!(
        "  verify, uncached keys    {:>10.1} ms (median of {})",
        ms(median(&result.verify_uncached)),
        result.verify_uncached.len()
    );
    println!(
        "  verify, cached keys      {:>10.1} ms (median of {})",
        ms(median(&result.verify_cached)),
        result.verify_cached.len()
    );
    println!(
        "  prover peak RSS          {:>13}",
        mib(result.prove_peak_kib)
    );
    println!(
        "  verifier peak RSS        {:>13}",
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
