//! Subcommands of the `agg` tool: `keygen-batch`, `sign-batch`, `aggregate`, `verify` and
//! `bench`. Batch files are those of [`crate::record`]; the aggregate is the mode-A wire format
//! ([`crate::mode_a`]).

use std::collections::HashMap;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::Path;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use agg_piop::binding::FlockProfile;
use agg_piop::manifest::Relation;
use agg_witness::gen;
use agg_witness::params::{dispatch_set, ParamSet, Set};
use rayon::prelude::*;

use crate::mode_a::{
    size_lines, AggregateVerifier, Aggregator, AggregatorConfig, KeyCacheMut, ProveReport,
    SignedRecord, DEFAULT_PART_LOG2, FLOCK_PROFILE,
};
use crate::record::{self, KeyEntry, RecordEntry};
use crate::{cache, rss};

/// Usage text.
pub const USAGE: &str = "\
usage: agg <command> [options]

Every command takes --params 44|65|87, the ML-DSA parameter set (default 44). Key and
record files of a set are refused by the commands of another set; an aggregate names its
set, and verify rejects one of another set than --params.

commands:
  keygen-batch --count K [--seed S] [--params 44|65|87] --out KEYS
      K ML-DSA key pairs from seeds derived from S (test keys: the file holds the seeds).
  sign-batch --keys KEYS --count N [--seed S] [--msg-len L] [--params 44|65|87]
             --out RECORDS
      N signed records (pure mode, empty context); record i is signed by key i mod K,
      messages are derived from S (length L, or pseudo-random in [0, 160)).
  aggregate --records RECORDS --out AGGREGATE [--params 44|65|87] [--part-log2 D]
            [--flock-profile P] [--flock-span on|off] [--relation R]
      Mode-A aggregate of the records (strict FIPS 204 verification of every signature);
      Flock parts of 2^D record positions (default 13, one tied part per segment); Flock
      profile P = fast, slim or slim8 (Ligerito rate 1/2, 1/4 or 1/8, all 128-bit in
      Flock's accounting; default fast); with --flock-span on (default) a
      batch of more than one segment in whole-segment parts is one tied part over all
      segments. Relation R = h-u (default: z in [-2^17, 2^17), |u| <= 190,465), h-exact
      (the FIPS 204 bounds |z| <= 130,993, |u| <= 190,465) or h-fips (also the hint:
      the extracted (c~, z, h) is a FIPS 204 signature). ML-DSA-65 and ML-DSA-87 have
      relation h-u only.
  verify --records RECORDS --aggregate AGGREGATE [--params 44|65|87] [--cache CACHE]
         [--flock-profile P] [--relation R]
      Verify the aggregate for the records' (pk, M) (signatures are ignored), under the
      Flock profile its items name (or only under P), under relation R (default h-u; an
      aggregate of another relation is rejected) and for the set of --params. With
      --cache, expanded keys are read from and added to the cache file (one file per
      set). Exit status 0 on accept, 1 on reject.
  bench --count N [--keys K] [--seed S] [--warmup W] [--reps R] [--params 44|65|87]
        [--part-log2 D] [--flock-profile P] [--flock-span on|off] [--relation R]
        [--json PATH]
      Generate N records over K keys (default N); aggregate W + R times, verify W + R times
      uncached and W + R times with cached keys (W warmup runs discarded, default W = 1,
      R = 5); print the median (and min) prove time, the verify medians, peak RSS of the
      prover and of the verifier (max over all runs), and the aggregate size per
      signature. With --json, also write every timed run, the aggregate size and
      SHA-256, and the selected kernels as JSON. Set AGG_GIT_REV to print the revision.

environment:
  AGG_FLOCK_THREADS=K   threads of the pool that proves the Flock parts beside the PIOP
                        (aggregate, bench; default 3/8 of the rayon pool rounded up, or 7/16
                        when Flock's GF(2^128) kernels run below AVX-512 — 12 or 14 on a
                        32-thread host; 0: one after the other; at most the logical CPUs or
                        the rayon pool; a value that is not a number warns and falls back to
                        the default).
  AGG_PIOP_S_THREADS=K  threads of the stage branch's own pool beside the Flock fork
                        (aggregate, bench; default half at 32 threads or more, 3/8 rounded
                        to nearest and at least 2 below; only with a running fork and at
                        least 4 PIOP threads; 0: the shared rayon pool). At 16 and 32
                        threads the stage defaults are 6 and 16; all three pools then add
                        up to 28 and 60 threads with an AVX-512 Flock pool, and 30 and 64
                        with the wider AVX2 share.
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
  AGG_ISA=LEVEL         cap every run-time SIMD detector: avx512ifma, avx512, avx2 (x86-64-v3
                        plus PCLMULQDQ) or scalar; unset detects. A cap only lowers a level,
                        and the proof is the same at every level.
  AGG_PIOP_LAYOUT=KIND  sum-check table layout: plane or row. Unset selects plane exactly
                        when the IFMA backend is selected, row otherwise; the proof is the
                        same either way.
  AGG_SIMD_REPORT=1     print, before the command, the level each crate selected (diagnostics).";

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

    /// `--name 44|65|87`: the ML-DSA parameter set, ML-DSA-44 when absent.
    fn params(&mut self, name: &str) -> Result<Set, String> {
        match self.take(name) {
            None => Ok(Set::S44),
            Some(v) => parse_params(&v).ok_or_else(|| format!("--{name}: 44, 65 or 87, not {v:?}")),
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

/// An ML-DSA parameter set by name: `44`, `65` or `87` (or `ML-DSA-44` and so on).
#[must_use]
pub fn parse_params(name: &str) -> Option<Set> {
    let short = name
        .strip_prefix("ML-DSA-")
        .or_else(|| name.strip_prefix("ml-dsa-"))
        .unwrap_or(name);
    Set::ALL.into_iter().find(|set| set.name() == short)
}

/// The encoded public-key length of `set` (FIPS 204 Table 2).
#[must_use]
pub const fn pk_bytes(set: Set) -> usize {
    dispatch_set!(set, |P| P::PK_BYTES)
}

/// Refuses keys of another length than `set`'s, naming the set they look like.
fn check_keys<'a>(set: Set, pks: impl IntoIterator<Item = &'a [u8]>, what: &str) -> Result<(), String> {
    for (i, pk) in pks.into_iter().enumerate() {
        if pk.len() != pk_bytes(set) {
            let hint = Set::ALL
                .into_iter()
                .find(|s| pk_bytes(*s) == pk.len())
                .map_or_else(String::new, |s| format!(" (an ML-DSA-{} key: --params {})", s.name(), s.name()));
            return Err(format!(
                "{what} {i}: a {}-byte public key, not ML-DSA-{}'s {}{hint}",
                pk.len(),
                set.name(),
                pk_bytes(set)
            ));
        }
    }
    Ok(())
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

/// `K` ML-DSA-44 key pairs from the seeds `gen::key_seed(seed, i)`.
#[must_use]
pub fn generate_keys(count: usize, seed: u64) -> Vec<KeyEntry> {
    generate_keys_for(Set::S44, count, seed)
}

/// `K` key pairs of `set` from the seeds `gen::key_seed(seed, i)`.
#[must_use]
pub fn generate_keys_for(set: Set, count: usize, seed: u64) -> Vec<KeyEntry> {
    dispatch_set!(set, |P| {
        (0..count as u64)
            .into_par_iter()
            .map(|i| {
                let xi = gen::key_seed(seed, i);
                let (_, pk) = gen::keypair_for::<P>(&xi);
                KeyEntry { seed: xi, pk }
            })
            .collect()
    })
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

/// `N` ML-DSA-44 records signed (hedged, randomness derived from the seed) round-robin by
/// `keys`.
#[must_use]
pub fn sign_records(
    keys: &[KeyEntry],
    count: usize,
    seed: u64,
    msg_len: Option<usize>,
) -> Vec<RecordEntry> {
    sign_records_for(Set::S44, keys, count, seed, msg_len)
}

/// [`sign_records`] for the key pairs of `set` (re-derived from the key file's seeds).
#[must_use]
pub fn sign_records_for(
    set: Set,
    keys: &[KeyEntry],
    count: usize,
    seed: u64,
    msg_len: Option<usize>,
) -> Vec<RecordEntry> {
    assert!(!keys.is_empty(), "at least one key");
    dispatch_set!(set, |P| {
        let signers: Vec<_> = keys.par_iter().map(|k| gen::keypair_for::<P>(&k.seed)).collect();
        (0..count)
            .into_par_iter()
            .map(|i| {
                let (sk, pk) = &signers[i % signers.len()];
                let msg = message(seed, i as u64, msg_len);
                let mut rnd = [0_u8; 32];
                gen::derive_bytes(b"cli-rnd", &[seed, i as u64], &mut rnd);
                RecordEntry {
                    pk: pk.clone(),
                    sig: gen::sign_for::<P>(sk, &msg, Some(&rnd)),
                    msg,
                }
            })
            .collect()
    })
}

fn keygen_batch(mut opts: Options) -> Result<Status, Box<dyn std::error::Error>> {
    let count: usize = opts.number("count", None)?;
    let seed: u64 = opts.number("seed", Some(1))?;
    let set = opts.params("params")?;
    let out = opts.path("out")?;
    opts.finish()?;
    let start = Instant::now();
    let keys = generate_keys_for(set, count, seed);
    std::fs::write(&out, record::keys_to_bytes(&keys))?;
    println!(
        "wrote {count} ML-DSA-{} key pairs to {} in {:.1} ms",
        set.name(),
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
    let set = opts.params("params")?;
    let out = opts.path("out")?;
    opts.finish()?;
    if keys.is_empty() {
        return Err("the key file is empty".into());
    }
    check_keys(set, keys.iter().map(|k| k.pk.as_slice()), "key")?;
    let start = Instant::now();
    let records = sign_records_for(set, &keys, count, seed, msg_len);
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
    let set = opts.params("params")?;
    opts.finish()?;
    check_keys(set, records.iter().map(|r| r.pk.as_slice()), "record")?;
    let aggregator = Aggregator::new(AggregatorConfig {
        part_log2,
        profile,
        span,
        relation,
        set,
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
    let set = opts.params("params")?;
    opts.finish()?;
    check_keys(set, records.iter().map(|r| r.pk.as_slice()), "record")?;
    let mut key_cache = match &cache_path {
        Some(p) => Some(cache::load(p, set)?),
        None => None,
    };
    let before = key_cache.as_ref().map_or(0, cache::KeyStore::len);
    let (report, key_cache) = run_large(move || {
        let verifier = match require {
            Some(p) => AggregateVerifier::for_set(set).require(p),
            None => AggregateVerifier::for_set(set),
        }
        .relation(relation);
        let cache = key_cache.as_mut().map_or(KeyCacheMut::None, cache::KeyStore::as_mut);
        let report = verifier.verify_with(&statements(&records), &bytes, cache);
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
    /// Size of the aggregate in bytes.
    pub aggregate_bytes: usize,
    /// SHA-256 of the aggregate, lower-case hexadecimal.
    pub aggregate_sha256: String,
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

/// Release every reusable buffer a previous run left behind: wait for detached drops, empty
/// the allocator cache and clear the Flock scratch pools. A timed run then starts like a new
/// process.
fn cold_start() {
    agg_pcs_config::sched::wait_background_drops();
    agg_piop::alloc::release_cache();
    agg_keccak::flock_core::scratch::clear();
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
    cold_start();
    let reset = rss::reset_peak();
    let mut bytes = Vec::new();
    let proves = timed_runs(warmup, reps, || {
        // Start each prove without blocks cached by the previous one, as a new process would.
        cold_start();
        let aggregate = aggregator.aggregate(&signed).map_err(|e| e.to_string())?;
        // Every run's aggregate has the same size; keep the latest (a timed run's at the end).
        bytes = aggregate.bytes;
        Ok::<_, String>(aggregate.report)
    })?;
    agg_pcs_config::sched::wait_background_drops();
    let prove_peak_kib = rss::peak_kib().filter(|_| reset);
    drop(aggregator);
    let prove_times: Vec<Duration> = proves.iter().map(|r| r.total).collect();
    let aggregate_bytes = bytes.len();
    let aggregate_sha256 = sha256_hex(&bytes);
    let prove = proves[median_index(&prove_times)].clone();
    drop(proves);
    // The verifier's peak must not count blocks the prover left in the caches.
    cold_start();
    let reset = rss::reset_peak();
    let verifier = AggregateVerifier::for_set(config.set)
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
    let mut key_cache = cache::KeyStore::new(config.set);
    let fill = verifier.verify_with(&statements, &bytes, key_cache.as_mut());
    if !fill.accepted() {
        return Err(format!("the honest aggregate was rejected:\n{fill}"));
    }
    let mut cached_report = String::new();
    let verify_cached = timed_runs(warmup, reps, || {
        let report = verifier.verify_with(&statements, &bytes, key_cache.as_mut());
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
        aggregate_bytes,
        aggregate_sha256,
        uncached_report,
        cached_report,
    })
}

/// Lower-case SHA-256 of `bytes`.
fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest as _;
    let mut out = String::with_capacity(64);
    for byte in sha2::Sha256::digest(bytes) {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// The ML-DSA engine selected for this process.
fn mldsa_backend_name() -> &'static str {
    let engine = agg_mldsa_simd::engine();
    match engine.backend() {
        agg_mldsa_simd::dispatch::Backend::Scalar => "scalar",
        agg_mldsa_simd::dispatch::Backend::Avx2 => "avx2",
        agg_mldsa_simd::dispatch::Backend::Avx512 => {
            if engine.has_vbmi() {
                "avx512+vbmi"
            } else {
                "avx512"
            }
        }
    }
}

/// An environment variable as a JSON value: a string when set, `null` otherwise.
fn json_env(name: &str) -> String {
    match std::env::var(name) {
        Ok(value) if !value.is_empty() => json_string(&value),
        _ => "null".to_string(),
    }
}

/// The source revision: from `AGG_GIT_REV` in the environment if set, or embedded
/// from `AGG_GIT_REV` at build time if available.
fn git_revision() -> Option<String> {
    std::env::var("AGG_GIT_REV")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| {
            option_env!("AGG_GIT_REV")
                .map(|s| s.to_string())
                .filter(|s| !s.is_empty())
        })
}

/// A JSON string. The benchmark values are ASCII, but escape the characters
/// that would make the output invalid if an environment variable held them.
fn json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// A JSON array of millisecond values.
fn json_ms(samples: &[Duration]) -> String {
    let values: Vec<String> = samples.iter().map(|d| format!("{:.3}", ms(*d))).collect();
    format!("[{}]", values.join(", "))
}

/// Write the machine-readable form of a benchmark result.
///
/// The document holds every timed run (not only the medians), the aggregate size
/// and hash, and the kernels each dispatcher selected. The caller passes the
/// settings so that the file describes the whole invocation.
#[allow(clippy::too_many_arguments)]
fn write_bench_json(
    path: &Path,
    set: Set,
    count: usize,
    n_keys: usize,
    seed: u64,
    part_log2: u32,
    profile: FlockProfile,
    span: bool,
    relation: Relation,
    result: &BenchResult,
) -> std::io::Result<()> {
    let mut out = String::new();
    let _ = writeln!(out, "{{");
    let _ = writeln!(out, "  \"tool\": \"agg bench\",");
    let _ = writeln!(out, "  \"json_version\": 1,");
    let _ = writeln!(out, "  \"params\": {},", json_string(set.name()));
    let _ = writeln!(out, "  \"count\": {count},");
    let _ = writeln!(out, "  \"keys\": {n_keys},");
    let _ = writeln!(out, "  \"seed\": {seed},");
    let _ = writeln!(out, "  \"part_log2\": {part_log2},");
    let _ = writeln!(
        out,
        "  \"flock_profile\": {},",
        json_string(&format!("{profile:?}"))
    );
    let _ = writeln!(out, "  \"flock_span\": {span},");
    let _ = writeln!(
        out,
        "  \"relation\": {},",
        json_string(&relation.to_string().to_lowercase())
    );
    let _ = writeln!(out, "  \"warmup\": {},", result.warmup);
    let _ = writeln!(out, "  \"reps\": {},", result.prove_times.len());
    let revision_str = git_revision().map_or_else(|| "null".to_string(), |r| json_string(&r));
    let _ = writeln!(out, "  \"revision\": {revision_str},");
    let _ = writeln!(out, "  \"kernels\": {{");
    let _ = writeln!(
        out,
        "    \"avx512_feature\": {},",
        agg_piop::avx512_enabled()
    );
    let _ = writeln!(
        out,
        "    \"keccak_avx512_feature\": {},",
        agg_keccak::avx512_enabled()
    );
    let _ = writeln!(
        out,
        "    \"aerie_simd_backend\": {},",
        json_string(agg_piop::aerie_simd_backend())
    );
    let _ = writeln!(
        out,
        "    \"mldsa_backend\": {},",
        json_string(mldsa_backend_name())
    );
    let _ = writeln!(out, "    \"AGG_ISA\": {},", json_env("AGG_ISA"));
    let _ = writeln!(
        out,
        "    \"AGG_PIOP_LAYOUT\": {},",
        json_env("AGG_PIOP_LAYOUT")
    );
    let _ = writeln!(out, "    \"AGG_SIMD\": {},", json_env("AGG_SIMD"));
    let _ = writeln!(
        out,
        "    \"AGG_MLDSA_SIMD\": {},",
        json_env("AGG_MLDSA_SIMD")
    );
    let _ = writeln!(out, "    \"FLOCK_SIMD\": {},", json_env("FLOCK_SIMD"));
    let _ = writeln!(
        out,
        "    \"AKITA_SCALAR_NTT\": {}",
        json_env("AKITA_SCALAR_NTT")
    );
    let _ = writeln!(out, "  }},");
    let _ = writeln!(out, "  \"prove_ms\": {},", json_ms(&result.prove_times));
    let _ = writeln!(
        out,
        "  \"verify_uncached_ms\": {},",
        json_ms(&result.verify_uncached)
    );
    let _ = writeln!(
        out,
        "  \"verify_cached_ms\": {},",
        json_ms(&result.verify_cached)
    );
    let optional = |v: Option<u64>| v.map_or_else(|| "null".to_string(), |v| v.to_string());
    let _ = writeln!(
        out,
        "  \"prove_peak_kib\": {},",
        optional(result.prove_peak_kib)
    );
    let _ = writeln!(
        out,
        "  \"verify_peak_kib\": {},",
        optional(result.verify_peak_kib)
    );
    let _ = writeln!(out, "  \"aggregate_bytes\": {},", result.aggregate_bytes);
    let _ = writeln!(
        out,
        "  \"bytes_per_signature\": {:.3},",
        result.prove.bytes_per_signature()
    );
    let _ = writeln!(
        out,
        "  \"aggregate_sha256\": {},",
        json_string(&result.aggregate_sha256)
    );
    let _ = writeln!(out, "  \"accepted\": true");
    let _ = writeln!(out, "}}");
    let mut file = std::fs::File::create(path)?;
    file.write_all(out.as_bytes())
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
    let set = opts.params("params")?;
    let json_path = opts.take("json").map(PathBuf::from);
    opts.finish()?;
    if n_keys == 0 || count == 0 {
        return Err("--count and --keys must be positive".into());
    }
    if reps == 0 {
        return Err("--reps must be positive".into());
    }
    let start = Instant::now();
    let keys = generate_keys_for(set, n_keys.min(count), seed);
    let records = sign_records_for(set, &keys, count, seed, None);
    let generated = start.elapsed();
    let result = run_large(move || {
        bench_records(
            &records,
            AggregatorConfig {
                part_log2,
                profile,
                span,
                relation,
                set,
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
    let mldsa_backend = mldsa_backend_name();
    println!(
        "agg bench: ML-DSA-{}, N = {count}, {} keys, part_log2 {part_log2}, Flock profile {profile:?}, span {span}, relation {relation}, warmup {warmup}, reps {reps}, revision {}, avx512 feature {}, aerie backend {}, mldsa backend {}",
        set.name(),
        n_keys.min(count),
        git_revision().unwrap_or_else(|| "unknown (set AGG_GIT_REV)".into()),
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
    if let Some(path) = json_path {
        write_bench_json(
            &path,
            set,
            count,
            n_keys.min(count),
            seed,
            part_log2,
            profile,
            span,
            relation,
            &result,
        )?;
    }
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
    fn sha256_is_the_standard_digest() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
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
