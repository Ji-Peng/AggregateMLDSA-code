//! The aggregate prover: commitment rounds, bridge seed, Keccak side, the lookup into
//! `T_16`, the three stages and the Akita opening in transcript order (`docs/lookup.md`).
//! [`prove_piop`] is the PIOP part after the commitments (lookup, pre-stage claims, stages
//! 1 to 3), shared with the stage tests.
//!
//! Fork-join (`docs/perf.md`): the final-vector step absorbs the final vectors and the Keccak side's
//! share items only; the transcript never absorbs the Flock proofs. No PIOP or Akita
//! challenge depends on them, so after the final-vector step the prover runs
//! [`KeccakSide::prove`] on its own rayon pool ([`flock_pool_threads`]) while the PIOP
//! (lookup, pre-stage claims, stages) and the Akita opening run on the rest of the threads;
//! the join only collects the proofs (`AGG_LATE_JOIN=0` moves the opening after it).
//!
//! Two hooks connect the parts owned by other crates: [`BridgeDerivation`] maps the bridge
//! seed `sd` to the matrices ([`SeededMatrices`] is the seed derivation,
//! `agg_bridge::seed`) and [`KeccakSide`] produces the Flock commitments and sub-proofs
//! (`agg-keccak`; `mode_b::PublicBits` in mode B). The stand-ins `FixedMatrices` (matrices
//! that ignore the seed) and `NoKeccak` (Flock profile 0) exist only in unit tests and with
//! the `test-hooks` feature (`docs/deviations.md`).

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use agg_pcs_config::commit::{CommitError, GroupClaim, GroupCommitment, PreparedOpening};
use agg_witness::bridge::BridgeMatrices;
use agg_witness::instance::PublicInstance;
use agg_witness::params::{ParamSet, Set, Set44};

use crate::binding::{instance_record, public_records_binding, FlockProfile};
use crate::claims::{MultiplicityClaims, PreStageClaims};
use crate::lookup::{prove_in, LookupError, LookupProof, LookupShape, Table};
use crate::manifest::{CommitmentManifest, GroupRole, Relation, GROUP_COUNT, LOOKUP_CHALLENGES};
use crate::opening::opening_claims;
use crate::pcs::{PcsError, TableCommitter};
use crate::pow::{KeccakPlan, PowPlan};
use crate::lane_public::LanePublicData;
use crate::public_data::PublicData;
use crate::source_tables::{BridgeGroupTables, MainTables, SourceTableError, SourceTables};
use crate::stage::{suffix, LaneId, Points, StageError, WeightId};
use crate::stage1::Stage1Proof;
use crate::stage2::Stage2Proof;
use crate::stage3::{LookupInputs, Stage3Proof};
use crate::tables::ProverTables;
use crate::transcript::{
    commitment_site, flock_record_digest, Transcript, TranscriptError, BRIDGE_SEED_SITE,
    FINAL_VECTORS_SITE, FLOCK_COMMIT_SITE, FLOCK_SEEDS_SITE,
    FLOCK_SHARES_SITE, MULTIPLICITY_SITE, PRESTAGE_SITE, TAU_SITE,
};
use crate::wire::{pack_final_vector, AggregateProof};
use crate::{stage1, stage2, stage3, OpeningField as E};

/// Maps the bridge seed to the three matrices.
pub trait BridgeDerivation {
    /// The matrices of `seed` (borrowed when they do not depend on it).
    fn matrices(&self, seed: &[u8; 32]) -> Cow<'_, BridgeMatrices>;

    /// True only when [`Self::matrices`] is the seed derivation ([`SeededMatrices`]), so that
    /// the matrices are uniform and independent of everything absorbed before `sd` and the
    /// bridge's soundness bound applies. Mode A (Flock profiles `Fast`, `Slim` and `Slim8`) is
    /// refused by [`prove`] and by `verifier::verify` otherwise.
    fn seed_derived(&self) -> bool {
        false
    }

    /// The matrices of `seed` for a batch of parameter set `set` (the bridge shape depends
    /// on the set: `agg_bridge::Shape::production`). ML-DSA-44 reads [`Self::matrices`]; a
    /// derivation without a per-set shape refuses the other sets.
    ///
    /// # Panics
    /// For a set other than ML-DSA-44 unless overridden.
    fn matrices_for(&self, set: Set, seed: &[u8; 32]) -> Cow<'_, BridgeMatrices> {
        assert_eq!(set, Set::S44, "this bridge derivation has ML-DSA-44's shape only");
        self.matrices(seed)
    }
}

/// The seed-derived matrices: `M_1, M_2, M_3` derived from `sd` with cSHAKE256
/// (`agg_bridge::seed::derive_matrices`). The derivation for prover and verifier builds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SeededMatrices;

impl BridgeDerivation for SeededMatrices {
    fn matrices(&self, seed: &[u8; 32]) -> Cow<'_, BridgeMatrices> {
        Cow::Owned(agg_bridge::seed::derive_matrices(seed))
    }

    fn seed_derived(&self) -> bool {
        true
    }

    fn matrices_for(&self, set: Set, seed: &[u8; 32]) -> Cow<'_, BridgeMatrices> {
        match set {
            Set::S44 => self.matrices(seed),
            _ => Cow::Owned(agg_bridge::seed::derive_matrices_for(
                &agg_bridge::Shape::production(set),
                seed,
            )),
        }
    }
}

/// Fixed matrices whatever the seed (test stand-in: unit tests and the `test-hooks`
/// feature only). The prover knows them before it commits `G_main`, so the bridge's
/// soundness bound does not hold; mode A refuses them, mode B stays sound through its direct
/// bit check (`docs/deviations.md`).
#[cfg(any(test, feature = "test-hooks"))]
pub struct FixedMatrices(pub BridgeMatrices);

#[cfg(any(test, feature = "test-hooks"))]
impl BridgeDerivation for FixedMatrices {
    fn matrices(&self, _seed: &[u8; 32]) -> Cow<'_, BridgeMatrices> {
        Cow::Borrowed(&self.0)
    }
}

/// The Keccak argument's contributions (the commitment phase, the final-vector step and the proof items).
pub trait KeccakSide {
    /// The Keccak side's part of the proof-of-work plan (its tied parts' tie sites,
    /// `docs/soundness.md`), a function of the public batch geometry, needed
    /// before the instance record. The default (no tied parts) fits a side without Flock sub-proofs.
    fn plan(&self, _segments: usize) -> KeccakPlan {
        KeccakPlan::default()
    }

    /// The `S` Flock commitments, before the bridge seed.
    fn commitments(&self, segments: usize) -> Vec<Vec<u8>>;

    /// The `S` share items of the final-vector step, absorbed after the final vectors `Y_sigma` (one bit
    /// per byte in `final_vectors`) and before the lookup: data the Flock proofs are checked
    /// against that is not a function of `Y_sigma` (mode A: the shares of a segment's parts
    /// but the last, under the matrices of `sd`). The default is `S` empty items.
    ///
    /// # Errors
    /// [`ProveError::Keccak`] if the side cannot produce them.
    fn shares(
        &self,
        _matrices: &BridgeMatrices,
        final_vectors: &[Vec<u8>],
    ) -> Result<Vec<Vec<u8>>, ProveError> {
        Ok(vec![Vec::new(); final_vectors.len()])
    }

    /// Whether [`Self::prove`] does work worth a thread pool of its own (default `true`). A
    /// side whose proof items are empty (mode B, profile 0) returns `false`, and the prover
    /// then skips the fork: the transcript and the proof are the same
    /// either way.
    fn proves_concurrently(&self) -> bool {
        true
    }

    /// The `S` Flock proofs, which the transcript never absorbs. They may be computed
    /// concurrently with the PIOP and the Akita opening, so they depend on nothing after the final-vector step: `bridge_seed` is `sd`
    /// and `matrices` its `M_1, M_2, M_3`, `seeds` are the challenger seeds
    /// `fs_sigma` (the Flock seed step), `final_vectors` the `Y_sigma` and `record` the digest of the
    /// final-vector record ([`crate::transcript::flock_record_digest`]: the final vectors and the
    /// share items), which the side MUST bind into every challenger of its sub-proofs, so
    /// that their challenge addresses encode the input of the lookup challenge `X`
    /// (`docs/soundness.md`).
    ///
    /// # Errors
    /// [`ProveError::Keccak`] if a sub-proof cannot be made (an exhausted proof-of-work
    /// search).
    fn prove(
        &self,
        bridge_seed: &[u8; 32],
        matrices: &BridgeMatrices,
        seeds: &[[u8; 32]],
        final_vectors: &[Vec<u8>],
        record: &[u8; 32],
    ) -> Result<Vec<Vec<u8>>, ProveError>;
}

/// No Keccak argument (Flock profile 0): empty commitments and proofs, so `c~_i` is not
/// tied to the committed bits (test stand-in: unit tests and the `test-hooks` feature
/// only; mode B is `mode_b::PublicBits`).
#[cfg(any(test, feature = "test-hooks"))]
pub struct NoKeccak;

#[cfg(any(test, feature = "test-hooks"))]
impl KeccakSide for NoKeccak {
    fn commitments(&self, segments: usize) -> Vec<Vec<u8>> {
        vec![Vec::new(); segments]
    }

    fn proves_concurrently(&self) -> bool {
        false
    }

    fn prove(
        &self,
        _bridge_seed: &[u8; 32],
        _matrices: &BridgeMatrices,
        seeds: &[[u8; 32]],
        _final_vectors: &[Vec<u8>],
        _record: &[u8; 32],
    ) -> Result<Vec<Vec<u8>>, ProveError> {
        Ok(vec![Vec::new(); seeds.len()])
    }
}

/// Threads of the Flock fork pool. `AGG_FLOCK_THREADS` overrides the default
/// (0 disables the fork). Without an override the default is half of the current pool
/// rounded up when Flock's GF(2^128) kernels run below the AVX-512 width, and 3/8 of the
/// current pool rounded up otherwise, capped at [`max_flock_threads`]. The narrower
/// vectors do half as many field elements per instruction, so the proof kernels need the
/// larger share of the pool to keep the fork-join wall short. A Rayon worker or a
/// one-thread pool disables the fork to avoid blocking the pool.
#[must_use]
pub fn flock_pool_threads() -> usize {
    if crate::pcs::on_calling_thread() {
        return 0;
    }
    let default = || default_flock_threads(rayon::current_num_threads());
    match std::env::var("AGG_FLOCK_THREADS") {
        Err(std::env::VarError::NotPresent) => default(),
        Ok(v) => match v.trim().parse::<usize>() {
            Ok(threads) => threads.min(max_flock_threads()),
            Err(_) => {
                warn_once(&format!("AGG_FLOCK_THREADS={v:?} is not a thread count; using {}", default()));
                default()
            }
        },
        Err(std::env::VarError::NotUnicode(_)) => {
            warn_once("AGG_FLOCK_THREADS is not Unicode; using the default pool size");
            default()
        }
    }
}

/// [`flock_pool_threads`] on a pool of `threads` workers: the share that
/// Flock's proof kernels need. The AVX-512 kernels process four GF(2^128)
/// elements per instruction and take 3/8 of the pool. The narrower AVX2 and
/// portable kernels need more instructions per element, but a pool of half the
/// workers starves the PIOP that runs beside them; 7/16 keeps the fork-join
/// wall short without that starvation.
fn default_flock_threads(threads: usize) -> usize {
    flock_default_for(threads, flock_core::f128_avx512_kernels())
}

/// [`default_flock_threads`] with the kernel width given, for tests.
fn flock_default_for(threads: usize, avx512: bool) -> usize {
    if avx512 {
        (3 * threads).div_ceil(8)
    } else {
        (7 * threads).div_ceil(16)
    }
}

/// Logical CPUs of the machine (`available_parallelism`, at least one).
fn logical_cpus() -> usize {
    std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
}

/// Threads of the pool that runs the Akita opening: `AGG_OPENING_THREADS` if set
/// (`0` or a value of at least the global pool: the global pool itself), else one per
/// physical core, capped by the global pool. The opening is a chain of small parallel
/// loops; on SMT hosts a second thread per core adds idle spinning and futex wake-ups
/// beside the critical thread instead of throughput (4K opening on an 8-core, 16-thread host:
/// about 0.9 s on 16 threads, 0.6-0.75 s on 8, direction only).
#[must_use]
pub fn opening_pool_threads() -> usize {
    let global = rayon::current_num_threads();
    let default = || physical_cores().map_or(global, |cores| cores.min(global));
    match std::env::var("AGG_OPENING_THREADS") {
        Err(_) => default(),
        Ok(v) => v.trim().parse::<usize>().unwrap_or_else(|_| {
            warn_once(&format!("AGG_OPENING_THREADS={v:?} is not a thread count; using {}", default()));
            default()
        }),
    }
}

/// Whether the Akita opening runs inside the PIOP branch of the fork, beside the tail of
/// the Flock proofs (the default), or after both branches have joined (`AGG_LATE_JOIN=0`).
/// The transcript never absorbs the Flock proofs, so the PCS session digest, the nonces and
/// the opening do not depend on where the opening runs: both schedules give the same proof
/// bytes, and the knob is an A/B switch of the schedule only.
fn opening_beside_flock() -> bool {
    match std::env::var("AGG_LATE_JOIN") {
        Err(_) => true,
        Ok(v) => match v.trim() {
            "0" => false,
            "1" | "" => true,
            other => {
                warn_once(&format!("AGG_LATE_JOIN={other:?} is neither 0 nor 1; using 1"));
                true
            }
        },
    }
}

/// The opening's finish pool when it runs beside the Flock proofs: the PIOP's share of the
/// threads (the PIOP pool of `AGG_PIOP_THREADS`, else the global pool less the Flock pool,
/// at least one), and no more than [`opening_pool_threads`]'s default. An explicit
/// `AGG_OPENING_THREADS` is taken as given.
fn opening_threads_beside(flock_threads: usize) -> usize {
    if std::env::var_os("AGG_OPENING_THREADS").is_some() {
        return opening_pool_threads();
    }
    let share = piop_pool_size(flock_threads)
        .unwrap_or_else(|| rayon::current_num_threads().saturating_sub(flock_threads).max(1));
    opening_pool_threads().min(share)
}

/// The CPUs this process may run on, from the `Cpus_allowed_list` line of
/// `/proc/self/status`, or `None` where the file cannot be read or parsed.
fn allowed_cpus() -> Option<Vec<u32>> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let list = status.lines().find_map(|line| line.strip_prefix("Cpus_allowed_list:"))?.trim();
    parse_cpu_list(list)
}

/// A `Cpus_allowed_list` value (`0-3,5,7-9`) as a CPU list, or `None` when malformed.
fn parse_cpu_list(list: &str) -> Option<Vec<u32>> {
    let mut cpus = Vec::new();
    for part in list.split(',') {
        let part = part.trim();
        if part.is_empty() {
            return None;
        }
        match part.split_once('-') {
            Some((lo, hi)) => {
                let lo: u32 = lo.parse().ok()?;
                let hi: u32 = hi.parse().ok()?;
                if hi < lo || hi - lo > 1 << 16 {
                    return None;
                }
                cpus.extend(lo..=hi);
            }
            None => cpus.push(part.parse().ok()?),
        }
    }
    (!cpus.is_empty()).then_some(cpus)
}

/// Physical cores among the CPUs the process may use (distinct sibling lists in the Linux
/// CPU topology), or `None` where it cannot be read. The affinity set matters under
/// `taskset`, Slurm and cgroups: counting `cpu0..N-1` includes cores outside the allocation
/// and can miss allowed ones.
fn physical_cores() -> Option<usize> {
    let allowed = allowed_cpus()?;
    let mut cores = std::collections::BTreeSet::new();
    for cpu in allowed {
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list");
        cores.insert(std::fs::read_to_string(path).ok()?.trim().to_owned());
    }
    (!cores.is_empty()).then_some(cores.len())
}

/// Run `f` on a pool of `threads` threads, or on the current pool when `threads` is 0 or not
/// below the current pool's size (or the pool cannot be built).
fn on_pool<T: Send>(threads: usize, name: &str, f: impl FnOnce() -> T + Send) -> T {
    if threads == 0 || threads >= rayon::current_num_threads() {
        return f();
    }
    let name = name.to_owned();
    match rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .stack_size(FLOCK_POOL_STACK)
        .thread_name(move |i| format!("{name}-{i}"))
        .build()
    {
        Ok(pool) => pool.install(f),
        Err(_) => f(),
    }
}

/// The largest Flock pool: the global pool or the machine's logical CPUs, whichever is
/// larger (more threads than that only oversubscribe the cores).
#[must_use]
pub fn max_flock_threads() -> usize {
    let cpus = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    cpus.max(rayon::current_num_threads())
}

/// With `AGG_PHASE_STAMPS` set, print `phase-stamp <name> <unix time in ns>` on stderr when
/// a prover step ends, so that an external CPU sampler can be aligned with the steps
/// (`docs/perf.md`). Prover diagnostics only; off by default, no effect on the proof.
pub fn phase_stamp(name: &str) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ON.get_or_init(|| std::env::var_os("AGG_PHASE_STAMPS").is_some()) {
        let ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        eprintln!("phase-stamp {name} {ns}");
    }
}

fn warn_once(message: &str) {
    static WARNED: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let warned = WARNED.get_or_init(|| Mutex::new(HashSet::new()));
    let mut warned = warned.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if warned.insert(message.to_owned()) {
        eprintln!("agg-piop: {message}");
    }
}

/// Stack size of the Flock pool's workers (the Flock prover nests large frames; a virtual
/// reservation, as Flock's own 8 MiB pool and the commit threads).
const FLOCK_POOL_STACK: usize = 64 << 20;

/// Branch S's own pools, keyed by `AGG_PIOP_S_THREADS`, built on the first use of each
/// count and kept for the process lifetime (range: one pool per distinct `k`, at most one
/// pool per prove). The entries are leaked because a branch holds `&'static` without
/// keeping the map locked.
static STAGES_POOLS: OnceLock<Mutex<HashMap<usize, &'static rayon::ThreadPool>>> =
    OnceLock::new();

/// The pool of `threads` workers that runs branch S, built on first use.
fn stages_pool(threads: usize) -> Option<&'static rayon::ThreadPool> {
    let pools = STAGES_POOLS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut pools = pools.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(&pool) = pools.get(&threads) {
        return Some(pool);
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .stack_size(FLOCK_POOL_STACK)
        .thread_name(|i| format!("agg-piop-s-{i}"))
        .build()
        .ok()?;
    let pool: &'static rayon::ThreadPool = Box::leak(Box::new(pool));
    pools.insert(threads, pool);
    Some(pool)
}

/// The PIOP pool size for the fork, or `None` for the caller's pool.
/// `AGG_PIOP_THREADS` sets it; an invalid value is reported and ignored. The sum
/// with the Flock pool is capped at the logical CPU count, because Rayon workers
/// that spin at a round barrier cost their sibling issue slots. Without the knob
/// the caller's pool is kept.
fn piop_pool_size(flock_threads: usize) -> Option<usize> {
    let requested = match std::env::var("AGG_PIOP_THREADS") {
        Err(_) => None,
        Ok(v) => match v.trim().parse::<usize>() {
            Ok(0) => {
                warn_once("AGG_PIOP_THREADS=0 is not a thread count; using the default");
                None
            }
            Ok(n) => Some(n.min(max_flock_threads())),
            Err(_) => {
                warn_once(&format!(
                    "AGG_PIOP_THREADS={v:?} is not a thread count; using the default"
                ));
                None
            }
        },
    };
    let mut threads = requested?;
    let cpus = logical_cpus();
    if threads.saturating_add(flock_threads) > cpus {
        // At least one PIOP thread: 0 would mean the full global pool.
        let capped = cpus.saturating_sub(flock_threads).max(1);
        warn_once(&format!(
            "PIOP {threads} + Flock {flock_threads} threads exceed {cpus} logical CPUs; \
             capping the PIOP pool at {capped}"
        ));
        threads = capped;
    }
    (threads > 0 && threads < rayon::current_num_threads()).then_some(threads)
}

/// Run `flock` on a fresh rayon pool of `threads` threads (on its own OS thread) while
/// `piop` runs on the caller (its parallel loops on the global pool); return both results
/// and the wall time of `flock`. The caller is not a rayon worker ([`flock_pool_threads`]
/// is 0 there), so no worker blocks on the OS thread. The PIOP pool follows
/// [`piop_pool_size`].
fn fork_join<A: Send, B: Send>(
    threads: usize,
    flock: impl FnOnce() -> A + Send,
    piop: impl FnOnce() -> B + Send,
) -> ((A, Duration), B) {
    let trace = std::env::var("AGG_SCHED_TRACE").as_deref() == Ok("1");
    std::thread::scope(|scope| {
        let handle = std::thread::Builder::new()
            .name("agg-flock".into())
            .stack_size(crate::pcs::COMMIT_THREAD_STACK)
            .spawn_scoped(scope, move || {
                let start = Instant::now();
                let out = match rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .stack_size(FLOCK_POOL_STACK)
                    .thread_name(|i| format!("agg-flock-{i}"))
                    .build()
                {
                    Ok(pool) => pool.install(flock),
                    // No pool: the Flock side's loops run on the global pool instead.
                    Err(_) => flock(),
                };
                (out, start.elapsed(), Instant::now())
            })
            .expect("spawn the Flock thread");
        let piop_start = Instant::now();
        let piop_threads = piop_pool_size(threads);
        let b = if let Some(n) = piop_threads {
            match rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .stack_size(FLOCK_POOL_STACK)
                .thread_name(|i| format!("agg-piop-{i}"))
                .build()
            {
                Ok(pool) => pool.install(piop),
                Err(_) => piop(),
            }
        } else {
            piop()
        };
        let piop_end = Instant::now();
        let (a, flock_busy, flock_end) = crate::pcs::join_os_thread(handle);
        let join = Instant::now();
        if trace {
            eprintln!("[sched] item=fork piop_busy_ms={:.3} piop_idle_ms={:.3} flock_busy_ms={:.3} flock_idle_ms={:.3}", (piop_end - piop_start).as_secs_f64() * 1e3, (join - piop_end).as_secs_f64() * 1e3, flock_busy.as_secs_f64() * 1e3, (join - flock_end).as_secs_f64() * 1e3);
        }
        ((a, flock_busy), b)
    })
}

/// A prover failure.
#[derive(Debug)]
pub enum ProveError {
    /// Table construction.
    Tables(SourceTableError),
    /// Commitment or opening.
    Pcs(PcsError),
    /// Transcript order.
    Transcript(TranscriptError),
    /// A stage.
    Stage(StageError),
    /// The lookup (an honest prover's self-check failed or a shape is wrong).
    Lookup(LookupError),
    /// The Keccak side could not make its sub-proofs.
    Keccak(String),
    /// Inconsistent inputs or an internal check.
    Internal(String),
}

impl std::fmt::Display for ProveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tables(e) => write!(f, "tables: {e}"),
            Self::Pcs(e) => write!(f, "pcs: {e}"),
            Self::Transcript(e) => write!(f, "transcript: {e}"),
            Self::Stage(e) => write!(f, "stage: {e}"),
            Self::Lookup(e) => write!(f, "lookup: {e}"),
            Self::Keccak(e) => write!(f, "keccak: {e}"),
            Self::Internal(e) => write!(f, "prover: {e}"),
        }
    }
}

impl std::error::Error for ProveError {}

impl From<SourceTableError> for ProveError {
    fn from(e: SourceTableError) -> Self {
        Self::Tables(e)
    }
}
impl From<PcsError> for ProveError {
    fn from(e: PcsError) -> Self {
        Self::Pcs(e)
    }
}
impl From<CommitError> for ProveError {
    fn from(e: CommitError) -> Self {
        Self::Pcs(PcsError::Commit(e))
    }
}
impl From<TranscriptError> for ProveError {
    fn from(e: TranscriptError) -> Self {
        Self::Transcript(e)
    }
}
impl From<StageError> for ProveError {
    fn from(e: StageError) -> Self {
        Self::Stage(e)
    }
}
impl From<LookupError> for ProveError {
    fn from(e: LookupError) -> Self {
        Self::Lookup(e)
    }
}

/// The prover's inputs: the public instance of the records and their signatures, for the
/// parameter set `P` (ML-DSA-44 by default).
pub struct ProverInputs<'a, P: ParamSet = Set44> {
    /// The public instance (keys expanded, `mu`, `c`, `T`).
    pub instance: &'a PublicInstance<P>,
    /// `(pk, M, c~)` per live record, in order (bound by the record digest).
    pub records: &'a [(&'a [u8], &'a [u8], P::CTilde)],
    /// The signatures, in order.
    pub signatures: &'a [&'a [u8]],
    /// The relation to prove (`docs/h-exact.md`): H-u, H-exact or H-FIPS.
    pub relation: Relation,
}

/// Wall time of each prover step (in event order).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProveTimings {
    /// Witness derivation and the round-1 tables (`G_main16`, `G_main4`).
    pub tables: Duration,
    /// Akita setup from the work plan.
    pub setup: Duration,
    /// The round-1 commitments (concurrent with the Flock commitments).
    pub round1: Duration,
    /// The Keccak side: commitments, shares and proofs (the proofs overlap the PIOP steps
    /// and, by default, the opening under the fork-join).
    pub keccak: Duration,
    /// The Flock proofs alone (on their own pool under the fork-join).
    pub flock_prove: Duration,
    /// Wall time from the fork to the join (Flock proofs beside the parity tables, the
    /// lookup, the pre-stage claims, the stages and, unless `AGG_LATE_JOIN=0`, the
    /// opening); zero without the fork.
    pub fork_join: Duration,
    /// Threads of the Flock pool (0: no fork, [`flock_pool_threads`]).
    pub flock_threads: usize,
    /// Matrices, the round-2 tables (parities, carries, multiplicities) and their
    /// commitments.
    pub round2: Duration,
    /// The lookup (fraction trees, sessions, multiplicity claims).
    pub lookup: Duration,
    /// `tau` and the pre-stage claims.
    pub prestage: Duration,
    /// Stage 1, including the `w_2, w_3` weight tables.
    pub stage1: Duration,
    /// Stage 2, including the `w_1` weight table.
    pub stage2: Duration,
    /// Stage 3, including the `w_A` weight table.
    pub stage3: Duration,
    /// The finish part of the batched Akita opening. The prepare part runs earlier, in the
    /// PIOP branch; under the fork both are inside [`Self::fork_join`] unless
    /// `AGG_LATE_JOIN=0` puts the finish after the join.
    pub opening: Duration,
    /// Threads of the pool of the opening's finish part ([`opening_pool_threads`], capped
    /// at the PIOP's share beside the Flock proofs; the current pool's size when the finish
    /// runs on it).
    pub opening_threads: usize,
    /// End to end.
    pub total: Duration,
}

impl std::fmt::Display for ProveTimings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        write!(
            f,
            "tables {:.1} ms, setup {:.1} ms, round 1 {:.1} ms, keccak {:.1} ms (flock prove \
             {:.1} ms on {} threads, fork-join {:.1} ms), round 2 {:.1} ms, \
             lookup {:.1} ms, prestage {:.1} ms, stage 1 {:.1} ms, stage 2 {:.1} ms, \
             stage 3 {:.1} ms, opening {:.1} ms on {} threads, total {:.1} ms",
            ms(self.tables),
            ms(self.setup),
            ms(self.round1),
            ms(self.keccak),
            ms(self.flock_prove),
            self.flock_threads,
            ms(self.fork_join),
            ms(self.round2),
            ms(self.lookup),
            ms(self.prestage),
            ms(self.stage1),
            ms(self.stage2),
            ms(self.stage3),
            ms(self.opening),
            self.opening_threads,
            ms(self.total)
        )
    }
}

/// What the prover produced.
pub struct ProverOutput {
    /// The proof.
    pub proof: AggregateProof,
    /// The bridge seed `sd`.
    pub bridge_seed: [u8; 32],
    /// The projection matrix `Psi` (the answer of the site `proj/psi`, after `sd`).
    pub psi: Vec<u8>,
    /// The Flock challenger seeds `fs_sigma` (the Flock seed step).
    pub flock_seeds: Vec<[u8; 32]>,
    /// The PCS session digest (drawn after stage 3).
    pub session: [u8; 32],
    /// The stage points `(tau, r^(1), r^(2), lookup points; r^(3))`.
    pub points: (Points, Vec<crate::OpeningField>),
    /// The lookup's table point `pi_m` (the opening point of `G_m`).
    pub pi_m: Vec<crate::OpeningField>,
    /// Wall time per step.
    pub timings: ProveTimings,
}

fn seeds_of(bytes: &[u8]) -> Vec<[u8; 32]> {
    bytes.as_chunks::<32>().0.to_vec()
}

/// A change applied to the honest round-2 tables before they are committed.
pub type BridgeTamper<'a> = &'a dyn Fn(&CommitmentManifest, &mut BridgeGroupTables);

/// Adversarial-test hooks of [`prove_with_tables`]: skip the honest prover's self-checks so
/// that a tampered witness reaches the verifier.
#[doc(hidden)]
#[derive(Default)]
pub struct ProveHooks<'a> {
    /// Use the stages' unchecked provers (no claimed-sum self-check).
    pub unchecked: bool,
    /// Change the honest round-2 tables before they are committed.
    pub tamper_bridge: Option<BridgeTamper<'a>>,
    /// Absorb the two branch digests at the join in the wrong order (S, then L).
    pub swap_join: bool,
}

/// A running clock for step timings.
struct Lap(Instant);

impl Lap {
    fn new() -> Self {
        Self(Instant::now())
    }

    fn lap(&mut self) -> Duration {
        let now = Instant::now();
        let elapsed = now - self.0;
        self.0 = now;
        elapsed
    }
}

/// Mode A (Flock profiles `Fast`, `Slim`, `Slim8`) only with the seed-derived matrices: a proof
/// over other matrices would be refused by the verifier.
fn check_bridge(bridge: &impl BridgeDerivation, flock: FlockProfile) -> Result<(), ProveError> {
    if flock.is_mode_a() && !bridge.seed_derived() {
        return Err(ProveError::Internal(
            "mode A (Flock profiles Fast, Slim, Slim8) needs the seed-derived matrices".into(),
        ));
    }
    Ok(())
}

/// Prove the aggregate: derive every witness (strict FIPS 204 verification), build the
/// round-1 tables, and run [`prove_with_tables`].
pub fn prove(
    inputs: &ProverInputs<'_>,
    bridge: &impl BridgeDerivation,
    keccak: &(impl KeccakSide + Sync),
    flock: FlockProfile,
) -> Result<ProverOutput, ProveError> {
    check_bridge(bridge, flock)?;
    let start = Instant::now();
    let (manifest, main) = main_tables(inputs)?;
    let tables = start.elapsed();
    let mut output = prove_with_tables(
        inputs,
        &manifest,
        &main,
        bridge,
        keccak,
        flock,
        &ProveHooks::default(),
    )?;
    output.timings.tables = tables;
    output.timings.total += tables;
    Ok(output)
}

/// The manifest and the round-1 tables of `inputs` (strict derivation).
pub fn main_tables(inputs: &ProverInputs<'_>) -> Result<(CommitmentManifest, MainTables), ProveError> {
    let n = inputs.records.len();
    if inputs.instance.records.len() != n || inputs.signatures.len() != n {
        return Err(ProveError::Internal("record counts differ".into()));
    }
    let manifest = CommitmentManifest::for_relation(n, inputs.relation).map_err(PcsError::Plan)?;
    let main = MainTables::build(&manifest, inputs.instance, inputs.signatures)?;
    Ok((manifest, main))
}

/// The smallest batch of a parameter set with polynomial lanes (ML-DSA-65, ML-DSA-87). Its
/// main domain has `8 + d_sig` variables, and the stages read the 13-variable level-1 and
/// level-2 bridge domains on suffixes of the main-domain points, so `d_sig >= 5` (`N* >= 32`,
/// at least 17 records).
pub const LANE_MIN_RECORDS: usize = 17;

/// [`main_tables`] for any parameter set: the manifest of `P`'s batch and the round-1
/// tables of the generic derivation (`MainTables::build_for`). ML-DSA-44 callers use
/// [`main_tables`], which runs the SIMD engine.
pub fn main_tables_for<P: ParamSet>(
    inputs: &ProverInputs<'_, P>,
) -> Result<(CommitmentManifest, MainTables), ProveError> {
    let n = inputs.records.len();
    if inputs.instance.records.len() != n || inputs.signatures.len() != n {
        return Err(ProveError::Internal("record counts differ".into()));
    }
    if P::SET != Set::S44 && n < LANE_MIN_RECORDS {
        return Err(ProveError::Internal(format!(
            "{} batches hold at least {LANE_MIN_RECORDS} records",
            P::NAME
        )));
    }
    let manifest = CommitmentManifest::for_set(P::SET, n, inputs.relation).map_err(PcsError::Plan)?;
    let main = MainTables::build_for(&manifest, inputs.instance, inputs.signatures)?;
    Ok((manifest, main))
}

/// The PIOP part of a proof (the lookup and the sum-check stages): the lookup, the multiplicity-digit claims,
/// the pre-stage claims and the three stages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiopProof {
    /// The lookup.
    pub lookup: LookupProof,
    /// `m_k(pi_m)`.
    pub multiplicities: MultiplicityClaims,
    /// The pre-stage claims.
    pub prestage: PreStageClaims,
    /// Stage 1.
    pub stage1: Stage1Proof,
    /// Stage 2.
    pub stage2: Stage2Proof,
    /// Stage 3.
    pub stage3: Stage3Proof,
}

/// The prover's points and step times of the PIOP part.
#[derive(Clone, Debug, Default)]
pub struct PiopRun {
    /// `tau, r^(1), r^(2)` and the lookup stack points.
    pub points: Points,
    /// `r^(3)`.
    pub r3: Vec<E>,
    /// The table point `pi_m`.
    pub pi_m: Vec<E>,
    /// The lookup.
    pub lookup: Duration,
    /// `tau` and the pre-stage claims.
    pub prestage: Duration,
    /// Stage 1 with the `w_2, w_3` tables.
    pub stage1: Duration,
    /// Stage 2 with `w_1`.
    pub stage2: Duration,
    /// Stage 3 with `w_A`.
    pub stage3: Duration,
}

/// The lookup shape of the batch must be the manifest's (a self-check of the stacks).
fn check_lookup_shape(manifest: &CommitmentManifest, shape: &LookupShape) -> Result<(), ProveError> {
    if *shape != manifest.lookup_shape() {
        return Err(ProveError::Internal("lookup stacks differ from the manifest".into()));
    }
    Ok(())
}

/// Prove the PIOP part (the lookup and the sum-check stages) on a transcript that has absorbed the instance record through the final-vector step (the
/// commitments, the seeds, the final vectors and the Keccak side's share items). `Psi` is
/// read from the transcript, so the projection matrix cannot disagree with the bridge-seed step.
/// `unchecked` skips the honest prover's self-checks (lookup root identity and session
/// claims, stage claimed sums) so that tests can hand a tampered witness to the verifier.
#[allow(clippy::too_many_arguments)]
pub fn prove_piop(
    manifest: &CommitmentManifest,
    tables: SourceTables<'_>,
    matrices: &BridgeMatrices,
    public: &PublicData,
    transcript: &mut Transcript,
    unchecked: bool,
) -> Result<(PiopProof, PiopRun), ProveError> {
    prove_piop_with(manifest, tables, None, matrices, PiopPublic::Rows(public), transcript, unchecked, false, false)
}

/// The public data the PIOP prover reads in stage 3.
#[derive(Clone, Copy, Debug)]
pub enum PiopPublic<'p> {
    /// ML-DSA-44: the rows of `A` and `T` on the main domain `(j, rho, i)`.
    Rows(&'p PublicData),
    /// The sets with polynomial lanes.
    Lanes(&'p LanePublicData),
}

/// [`prove_piop`] for a parameter set with polynomial lanes (ML-DSA-65, ML-DSA-87).
#[allow(clippy::too_many_arguments)]
pub fn prove_piop_lanes(
    manifest: &CommitmentManifest,
    tables: SourceTables<'_>,
    matrices: &BridgeMatrices,
    public: &LanePublicData,
    transcript: &mut Transcript,
    unchecked: bool,
) -> Result<(PiopProof, PiopRun), ProveError> {
    prove_piop_with(manifest, tables, None, matrices, PiopPublic::Lanes(public), transcript, unchecked, false, false)
}

/// Whether the two transcript branches run concurrently (`AGG_PIOP_FORK`, default on; `0`
/// or `off` runs L, then S, on the calling thread). The transcript and the proof bytes are
/// the same either way.
fn piop_fork_on() -> bool {
    !matches!(std::env::var("AGG_PIOP_FORK").as_deref(), Ok("0" | "off"))
}

/// Threads of branch S's own pool (`AGG_PIOP_S_THREADS`), or `None` for the shared pool.
/// An explicit count wins; `0` selects the shared pool. Unset, the default of
/// [`default_stage_threads`] applies, and so does a value that is not a count after one
/// report. An explicit count is capped at [`max_flock_threads`], the machine's logical CPUs
/// or the global pool size, whichever is larger; a larger value only oversubscribes the
/// cores, and each distinct count keeps a leaked pool for the process lifetime.
fn piop_s_threads(flock_forks: bool) -> Option<usize> {
    match std::env::var("AGG_PIOP_S_THREADS") {
        Err(_) => default_stage_threads(flock_forks, rayon::current_num_threads()),
        Ok(v) => match v.trim().parse::<usize>() {
            Ok(0) => None,
            Ok(n) => {
                let capped = n.min(max_flock_threads());
                if capped != n {
                    warn_once(&format!(
                        "AGG_PIOP_S_THREADS={n} exceeds the logical CPU count; \
                         capping the stage pool at {capped}"
                    ));
                }
                Some(capped)
            }
            Err(_) => {
                warn_once(&format!(
                    "AGG_PIOP_S_THREADS={v:?} is not a thread count; using the default"
                ));
                default_stage_threads(flock_forks, rayon::current_num_threads())
            }
        },
    }
}

/// The default size of branch S's own pool on a PIOP pool of `threads` workers and a Flock
/// pool of `flock` threads, or `None` for the shared pool. Branch S gets its own pool only
/// while the Flock fork runs and the PIOP pool has at least four threads. At the two
/// measured PIOP sizes, 16 and 32 workers, the default is 3/8 of the pool rounded to
/// nearest (at least 2) below 32 and half at 32 or more: 6 and 16 workers on a 16C32T host.
/// Elsewhere the default is capped so that PIOP + stage + Flock stays within the `cpus`
/// logical CPUs, and the pool is shared when that leaves fewer than two workers. A separate
/// pool leaves every worker of the main PIOP pool to branch L, so the lookup's round
/// barriers do not wait behind stage jobs in the same injector.
fn stage_threads(flock_forks: bool, threads: usize, flock: usize, cpus: usize) -> Option<usize> {
    if !flock_forks || threads < 4 {
        return None;
    }
    let stage = if threads >= 32 {
        threads / 2
    } else {
        // round(threads * 3 / 8) in integers; 16 gives 6 and 4 gives 2.
        ((3 * threads + 4) / 8).max(2)
    };
    if matches!(threads, 16 | 32) {
        return Some(stage);
    }
    let budget = cpus.saturating_sub(threads.saturating_add(flock));
    (budget >= 2).then(|| stage.min(budget))
}

/// [`stage_threads`] with the running Flock pool and the logical CPU count.
fn default_stage_threads(flock_forks: bool, threads: usize) -> Option<usize> {
    stage_threads(flock_forks, threads, flock_pool_threads(), logical_cpus())
}

/// Run branch L (`lookup`) and branch S (`stages`) of the forked transcript, concurrently
/// unless [`piop_fork_on`] says otherwise. With `AGG_PIOP_S_THREADS=k` (or the default of
/// [`default_stage_threads`] while `flock_forks`), S runs on its own pool of `k` workers
/// and L keeps every worker of the main PIOP pool; otherwise both feed the pool the PIOP
/// runs on: inside a rayon pool (the PIOP pool, when `AGG_PIOP_THREADS` sets one)
/// by `rayon::join`; off the pool (the caller's thread, which carries a large stack) with S
/// on a scoped thread whose stack is that of the PIOP pool's workers, so that neither
/// branch's sequential driver lands on a global-pool worker with rayon's default stack.
/// Off the pool with a one-thread global pool, both branches run on the caller: an OS
/// thread for S would add a second core's worth of work on one core and no parallelism.
/// `flock_forks` reports whether the Flock proofs run beside the PIOP.
fn run_branches<A: Send, B: Send>(
    lookup: impl FnOnce() -> A + Send,
    stages: impl FnOnce() -> B + Send,
    flock_forks: bool,
) -> (A, B) {
    if !piop_fork_on() {
        let a = lookup();
        return (a, stages());
    }
    let on_worker = rayon::current_thread_index().is_some();
    if !on_worker && rayon::current_num_threads() <= 1 {
        let a = lookup();
        return (a, stages());
    }
    if let Some(s_pool) = piop_s_threads(flock_forks).and_then(stages_pool) {
        return std::thread::scope(|scope| {
            let handle = std::thread::Builder::new()
                .name("agg-piop-stages-driver".into())
                .stack_size(FLOCK_POOL_STACK)
                .spawn_scoped(scope, move || s_pool.install(stages))
                .expect("spawn the stage branch");
            let a = lookup();
            (a, crate::pcs::join_os_thread(handle))
        });
    }
    if on_worker {
        return rayon::join(lookup, stages);
    }
    std::thread::scope(|scope| {
        let handle = std::thread::Builder::new()
            .name("agg-piop-stages".into())
            .stack_size(FLOCK_POOL_STACK)
            .spawn_scoped(scope, stages)
            .expect("spawn the stage branch");
        let a = lookup();
        (a, crate::pcs::join_os_thread(handle))
    })
}

/// What branch L hands to the join.
struct LookupBranch {
    proof: LookupProof,
    multiplicities: MultiplicityClaims,
    pi_m: Vec<E>,
    inputs: LookupInputs,
    stack_points: Vec<Vec<E>>,
    time: Duration,
}

/// What branch S hands to the join.
struct StagesBranch<'a> {
    points: Points,
    prestage: PreStageClaims,
    tables: ProverTables<'a>,
    stage1: (Stage1Proof, crate::stage1::Stage1Output),
    stage2: (Stage2Proof, crate::stage2::Stage2Output),
    /// `T(r^(2))` (`T~_rho(r^(2))` per row with lanes) for stage 3's claimed sums: a function
    /// of public data and `r^(2)` only.
    t_at_r2: Vec<E>,
    times: [Duration; 3],
    /// Wall time of the stage-3 precomputation (`w_A` and `T(r^(2))`).
    tail: Duration,
}

/// Branch L: the lookup into `T_16` and the digit claims of `m`.
fn prove_lookup_branch(
    manifest: &CommitmentManifest,
    tables: SourceTables<'_>,
    transcript: &mut Transcript,
    unchecked: bool,
    trace: bool,
) -> Result<LookupBranch, ProveError> {
    let clock = Instant::now();
    let stacks = tables.lookup_stacks()?;
    let multiplicity = vec![tables.bridge.multiplicity().to_vec()];
    let output = prove_in(
        &stacks,
        &[Table::T16],
        &multiplicity,
        LOOKUP_CHALLENGES,
        transcript,
        !unchecked,
    )?;
    check_lookup_shape(manifest, &LookupShape::of(&stacks, &[Table::T16], LOOKUP_CHALLENGES))?;
    drop(stacks);
    let (pi_m, m_at_pi) = output.claims.multiplicities[0].clone();
    let digits = tables.group(GroupRole::Mult);
    let multiplicities = MultiplicityClaims(std::array::from_fn(|k| digits[k].evaluate(&pi_m)));
    if !unchecked && multiplicities.compose() != m_at_pi {
        return Err(ProveError::Internal("multiplicity digits disagree with m".into()));
    }
    transcript.absorb_extensions(MULTIPLICITY_SITE, &multiplicities.0)?;
    let (inputs, stack_points) = LookupInputs::new(manifest, &output.claims);
    if trace {
        let sessions: f64 = output.stats.sessions.iter().sum();
        eprintln!(
            "lookup: trees {:.1} ms, sessions {:.1} ms, stored layers {:.1} MiB",
            output.stats.trees * 1e3,
            sessions * 1e3,
            output.stats.layer_bytes as f64 / f64::from(1 << 20)
        );
    }
    phase_stamp("lookup");
    Ok(LookupBranch {
        proof: output.proof,
        multiplicities,
        pi_m,
        inputs,
        stack_points,
        time: clock.elapsed(),
    })
}

/// Branch S: `tau`, the pre-stage claims, stages 1 and 2.
#[allow(clippy::too_many_arguments)]
fn prove_stages_branch<'a>(
    manifest: &CommitmentManifest,
    tables: SourceTables<'a>,
    prebuilt: Option<ProverTables<'a>>,
    matrices: &BridgeMatrices,
    psi: &[u8],
    public: PiopPublic<'_>,
    transcript: &mut Transcript,
    unchecked: bool,
    trace: bool,
) -> Result<StagesBranch<'a>, ProveError> {
    let mut clock = Lap::new();
    // Tau and the pre-stage claims.
    let mut points = Points {
        tau: transcript.challenge_extensions(TAU_SITE, manifest.d_main())?,
        ..Points::default()
    };
    let prestage = PreStageClaims::evaluate(manifest, tables, &points.tau);
    transcript.absorb_extensions(PRESTAGE_SITE, &prestage.to_vec())?;
    let y_at_tau3 = agg_bridge::final_vector::evaluate(
        tables.bridge.public_y(),
        &suffix(&points.tau, manifest.group(GroupRole::Level3).logical_vars),
    );
    let prestage_time = clock.lap();
    phase_stamp("prestage");

    // Stages 1 and 2.
    let weights_clock = Instant::now();
    let mut prover_tables = prebuilt.unwrap_or_else(|| ProverTables::new(tables, matrices));
    let parity_time = weights_clock.elapsed();
    prover_tables.set_tau(matrices, &points.tau);
    if trace {
        eprintln!(
            "weights: parity1 tables {:.1} ms, w_2/w_3 {:.1} ms",
            parity_time.as_secs_f64() * 1e3,
            (weights_clock.elapsed() - parity_time).as_secs_f64() * 1e3
        );
    }
    let (stage1_proof, stage1) = if unchecked {
        stage1::prove_unchecked(manifest, &prover_tables, &points, &prestage, y_at_tau3, transcript)?
    } else {
        stage1::prove(manifest, &prover_tables, &points, &prestage, y_at_tau3, transcript)?
    };
    points.r1 = stage1.point.clone();
    let stage1_time = clock.lap();
    phase_stamp("stage1");
    let weights_clock = Instant::now();
    prover_tables.set_r1(matrices, &points.r1);
    if trace {
        eprintln!("weights: w_1 {:.1} ms", weights_clock.elapsed().as_secs_f64() * 1e3);
    }
    // The split replaces `w_1` with an empty placeholder lane in `ProverTables::lane`; only
    // stage 2's first group may read it, because the split driver substitutes its factors
    // there. Any other position would silently use the empty table.
    let stage2_spec = stage2::spec(manifest);
    assert_eq!(
        stage2_spec.groups[0].lanes.first(),
        Some(&LaneId::Weight(WeightId::Level1(0))),
        "stage 2's first group must read w_1 in lane 0"
    );
    assert!(
        manifest.has_lanes()
            || stage2_spec.groups.iter().all(|group| group
                .lanes
                .iter()
                .position(|&lane| lane == LaneId::Weight(WeightId::Level1(0)))
                .is_none_or(|position| position == 0)),
        "w_1 may only appear as the first lane of a stage-2 group"
    );
    let (stage2_proof, stage2) = if unchecked {
        stage2::prove_unchecked(manifest, &prover_tables, &points, &stage1, transcript)?
    } else {
        stage2::prove(manifest, &prover_tables, &points, &stage1, transcript)?
    };
    points.r2 = stage2.point.clone();
    let stage2_time = clock.lap();
    phase_stamp("stage2");
    // Stage 3's inputs that need no stage-3 challenge: `w_A`, the ring weight built from
    // `r^(2)` and the public matrices, and `T(r^(2))`, the public T table's multilinear
    // extension at `r^(2)`. Both depend only on data fixed when stage 2 ends, so they run
    // at the tail of branch S instead of after the join. The transcript does not see them,
    // and the bytes are the same: this is a scheduling move, not a change of the messages.
    let tail_clock = Instant::now();
    let t_at_r2 = match public {
        PiopPublic::Rows(public) => {
            prover_tables.set_r2(public, &points.r2);
            vec![public.t_evaluation(&points.r2)]
        }
        PiopPublic::Lanes(public) => {
            prover_tables.drop_level1();
            public.t_evaluations(&points.r2)
        }
    };
    prover_tables.set_projection(psi, &points.r2);
    let tail = tail_clock.elapsed();
    if trace {
        eprintln!("weights: w_A, W and T(r^(2)) {:.1} ms", tail.as_secs_f64() * 1e3);
    }
    Ok(StagesBranch {
        points,
        prestage,
        tables: prover_tables,
        stage1: (stage1_proof, stage1),
        stage2: (stage2_proof, stage2),
        t_at_r2,
        times: [prestage_time, stage1_time, stage2_time],
        tail,
    })
}

/// [`prove_piop`] with the challenge-independent prover tables (the parities `y^(1)`)
/// built beforehand, e.g. while the Flock span is proven (prover only). The transcript
/// forks after the final-vector step: branch L (the lookup) and branch S (`tau` through stage 2) run
/// side by side ([`run_branches`]), the join absorbs their digests, stage 3 follows on the
/// trunk. `swap_join` is a test hook that absorbs the digests in the wrong order, and
/// `flock_forks` reports whether the Flock proofs run beside the PIOP.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn prove_piop_with<'a>(
    manifest: &CommitmentManifest,
    tables: SourceTables<'a>,
    prebuilt: Option<ProverTables<'a>>,
    matrices: &BridgeMatrices,
    public: PiopPublic<'_>,
    transcript: &mut Transcript,
    unchecked: bool,
    swap_join: bool,
    flock_forks: bool,
) -> Result<(PiopProof, PiopRun), ProveError> {
    let fork_clock = Instant::now();
    let mut run = PiopRun::default();
    let trace = std::env::var_os("AGG_STAGE_TRACE").is_some();
    let psi = transcript.psi()?.to_vec();

    // The fork: branch L (the lookup) and branch S on two clones of the sponge.
    let (mut lookup_transcript, mut stages_transcript) = transcript.fork()?;
    let (lookup, stages) = run_branches(
        || prove_lookup_branch(manifest, tables, &mut lookup_transcript, unchecked, trace),
        || {
            prove_stages_branch(
                manifest,
                tables,
                prebuilt,
                matrices,
                &psi,
                public,
                &mut stages_transcript,
                unchecked,
                trace,
            )
        },
        flock_forks,
    );
    let (lookup, stages) = (lookup?, stages?);
    if swap_join {
        transcript.join_swapped(lookup_transcript, stages_transcript)?;
    } else {
        transcript.join(lookup_transcript, stages_transcript)?;
    }
    let fork_wall = fork_clock.elapsed();
    if trace {
        let stages_time: Duration = stages.times.iter().sum::<Duration>() + stages.tail;
        eprintln!(
            "fork: lookup {:.1} ms, tau to T(r^(2)) {:.1} ms, wall {:.1} ms (concurrent: {})",
            lookup.time.as_secs_f64() * 1e3,
            stages_time.as_secs_f64() * 1e3,
            fork_wall.as_secs_f64() * 1e3,
            piop_fork_on()
        );
    }
    run.lookup = lookup.time;
    [run.prestage, run.stage1, run.stage2] = stages.times;
    let StagesBranch {
        mut points,
        prestage,
        tables: prover_tables,
        stage1: (stage1_proof, stage1),
        stage2: (stage2_proof, stage2),
        t_at_r2,
        ..
    } = stages;
    points.lookup = lookup.stack_points;

    // Stage 3 on the trunk.
    let clock = Instant::now();
    let prove3 = if unchecked {
        stage3::prove_unchecked
    } else {
        stage3::prove
    };
    let ring = match public {
        PiopPublic::Rows(_) => stage3::RingInputs::Rows(t_at_r2[0]),
        PiopPublic::Lanes(public) => stage3::RingInputs::Lanes { t: &t_at_r2, public },
    };
    let (stage3_proof, stage3) = prove3(
        manifest,
        &prover_tables,
        ring,
        &points,
        &prestage,
        &stage1,
        &stage2,
        &lookup.inputs,
        transcript,
    )?;
    // The owned buffers (parities and weights, several GiB at 32K) are released on a
    // detached thread, off the PIOP branch's critical path. Large blocks go back to the
    // allocator cache (up to its cap) rather than being unmapped, so a later prove can
    // reuse them; `prover_tables` itself only borrows the committed tables.
    let ProverTables {
        parity1,
        w1,
        w1_lanes,
        w2,
        w3,
        wa,
        wp,
        wp_lanes,
        ..
    } = prover_tables;
    agg_pcs_config::sched::drop_in_background("stage tables", (parity1, w1, w2, w3, wa, wp, (w1_lanes, wp_lanes)));
    run.stage3 = clock.elapsed();
    phase_stamp("stage3");
    run.points = points;
    run.r3 = stage3.point;
    run.pi_m = lookup.pi_m;
    Ok((
        PiopProof {
            lookup: lookup.proof,
            multiplicities: lookup.multiplicities,
            prestage,
            stage1: stage1_proof,
            stage2: stage2_proof,
            stage3: stage3_proof,
        },
        run,
    ))
}

/// Prove from given round-1 tables (the whole transcript with the join). `hooks` are for
/// adversarial tests; an honest caller passes `ProveHooks::default()`.
pub fn prove_with_tables(
    inputs: &ProverInputs<'_>,
    manifest: &CommitmentManifest,
    main: &MainTables,
    bridge: &impl BridgeDerivation,
    keccak: &(impl KeccakSide + Sync),
    flock: FlockProfile,
    hooks: &ProveHooks<'_>,
) -> Result<ProverOutput, ProveError> {
    prove_with_tables_using(None, None, inputs, manifest, main, bridge, keccak, flock, hooks)
}

/// Compute the part of the opening that depends only on the commitments and
/// the stage-3 points. It can run beside the Flock proofs because it reads no
/// transcript challenge beyond the points already fixed by stage 3.
///
/// This runs inside the PIOP branch, on the PIOP pool: `AGG_OPENING_THREADS` sizes only the
/// finish part, and the time spent here is not part of `timings.opening` (under the fork it
/// is inside `timings.fork_join`).
fn prepare_opening(
    committer: &TableCommitter,
    manifest: &CommitmentManifest,
    commitments: [GroupCommitment; GROUP_COUNT],
    proof: &PiopProof,
    run: &PiopRun,
) -> Result<([GroupClaim; GROUP_COUNT], PreparedOpening), ProveError> {
    let claims = opening_claims(
        manifest,
        &run.r3,
        &proof.stage3.claims,
        &run.pi_m,
        &proof.multiplicities,
    )?;
    let prepared = committer.prepare_open_claims(commitments, &claims)?;
    Ok((claims, prepared))
}

/// The opening, before or after its transcript-bound finish.
enum OpeningState {
    /// Prepared; the finish runs after the join.
    Prepared(PreparedOpening),
    /// Finished inside the PIOP branch, beside the Flock proofs.
    Finished(FinishedOpening),
}

/// The finished opening with what the prover reports about it.
struct FinishedOpening {
    session: [u8; 32],
    work: Vec<u8>,
    opening: Vec<u8>,
    threads: usize,
    time: Duration,
}

/// Export the PCS session digest (the transcript's last event) and run the opening's finish
/// on a pool of `threads` workers (0: the current pool). The transcript must have absorbed
/// the stage-3 claims; the Flock proofs are not part of it.
fn finish_opening(
    committer: &TableCommitter,
    tables: SourceTables<'_>,
    transcript: &mut Transcript,
    claims: &[GroupClaim; GROUP_COUNT],
    prepared: PreparedOpening,
    threads: usize,
    unchecked: bool,
) -> Result<FinishedOpening, ProveError> {
    let session = transcript.pcs_session()?;
    transcript.ensure_complete()?;
    let work = transcript.work_stream();
    // The stage-3 terminal values are the multilinear extensions of the committed tables at
    // the group points (the kernel's terminal values; `tests/all/prove_verify.rs` checks them
    // against the table evaluations), so the opening takes them directly instead of
    // evaluating every table again.
    debug_assert!(
        unchecked
            || crate::pcs::opening_claims(tables, &claims.clone().map(|c| c.point))
                .is_ok_and(|recomputed| recomputed == *claims),
        "stage-3 terminal values differ from the table evaluations"
    );
    let pool = rayon::current_num_threads();
    let used = if threads == 0 { pool } else { threads.min(pool) };
    let start = Instant::now();
    let opening = on_pool(threads, "agg-open", || {
        committer.finish_open_claims(prepared, &session)
    })?;
    let time = start.elapsed();
    phase_stamp("opening");
    Ok(FinishedOpening {
        session,
        work,
        opening,
        threads: used,
        time,
    })
}

/// [`prove_with_tables`] with the Akita prover setup and the public data built
/// beforehand. Both depend only on the records and the instance, so a caller can build
/// them on its own thread while the tables are computed. `None`, or a setup of another
/// manifest, builds it here.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
pub fn prove_with_tables_using(
    committer: Option<TableCommitter>,
    public: Option<PublicData>,
    inputs: &ProverInputs<'_>,
    manifest: &CommitmentManifest,
    main: &MainTables,
    bridge: &impl BridgeDerivation,
    keccak: &(impl KeccakSide + Sync),
    flock: FlockProfile,
    hooks: &ProveHooks<'_>,
) -> Result<ProverOutput, ProveError> {
    let total = Instant::now();
    let n = inputs.records.len();
    if manifest.records() != n || main.geometry().n != n || manifest.set() != Set::S44 {
        return Err(ProveError::Internal("tables and records differ".into()));
    }
    check_bridge(bridge, flock)?;
    let public = public.unwrap_or_else(|| PublicData::from_instance(inputs.instance));
    prove_core(
        total,
        committer,
        PiopPublic::Rows(&public),
        inputs.records,
        manifest,
        main,
        bridge,
        keccak,
        flock,
        hooks,
    )
}

/// [`prove_with_tables_using`] for a parameter set with polynomial lanes (ML-DSA-65,
/// ML-DSA-87): the public data is the instance's [`LanePublicData`] (built here unless
/// given), the bridge matrices are the set's ([`BridgeDerivation::matrices_for`]).
#[allow(clippy::too_many_arguments)]
pub fn prove_lanes_with_tables_using<P: ParamSet>(
    committer: Option<TableCommitter>,
    public: Option<LanePublicData>,
    inputs: &ProverInputs<'_, P>,
    manifest: &CommitmentManifest,
    main: &MainTables,
    bridge: &impl BridgeDerivation,
    keccak: &(impl KeccakSide + Sync),
    flock: FlockProfile,
    hooks: &ProveHooks<'_>,
) -> Result<ProverOutput, ProveError>
where
    P::CTilde: Sync,
{
    let total = Instant::now();
    let n = inputs.records.len();
    if P::SET == Set::S44 {
        return Err(ProveError::Internal("ML-DSA-44 has no polynomial lanes".into()));
    }
    if manifest.records() != n || main.geometry().n != n || manifest.set() != P::SET {
        return Err(ProveError::Internal("tables and records differ".into()));
    }
    if n < LANE_MIN_RECORDS {
        return Err(ProveError::Internal(format!(
            "{} batches hold at least {LANE_MIN_RECORDS} records",
            P::NAME
        )));
    }
    check_bridge(bridge, flock)?;
    let public = match public {
        Some(public) if public.set() == P::SET && public.records() == n => public,
        _ => LanePublicData::from_instance(inputs.instance),
    };
    prove_core(
        total,
        committer,
        PiopPublic::Lanes(&public),
        inputs.records,
        manifest,
        main,
        bridge,
        keccak,
        flock,
        hooks,
    )
}

/// The prover after the public data: the Akita setup, the transcript from the instance
/// record to the opening, the Flock fork and the proof. `total` is the caller's start.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn prove_core<C: AsRef<[u8]> + Sync>(
    total: Instant,
    committer: Option<TableCommitter>,
    public: PiopPublic<'_>,
    records: &[(&[u8], &[u8], C)],
    manifest: &CommitmentManifest,
    main: &MainTables,
    bridge: &impl BridgeDerivation,
    keccak: &(impl KeccakSide + Sync),
    flock: FlockProfile,
    hooks: &ProveHooks<'_>,
) -> Result<ProverOutput, ProveError> {
    let mut clock = Lap(total);
    let mut timings = ProveTimings::default();
    let n = records.len();
    let segments = manifest.segments();
    let committer = match committer {
        Some(prebuilt) if prebuilt.manifest() == manifest => prebuilt,
        _ => TableCommitter::new(manifest.clone())?,
    };
    timings.setup = clock.lap();
    phase_stamp("setup");

    // The instance record (with the proof-of-work plan's digest),
    // round-1 commitments, Flock commitments, bridge seed.
    let digest = public_records_binding(records);
    let plan = PowPlan::new(manifest, flock, &keccak.plan(segments));
    let record = instance_record(manifest, &digest, flock, &plan.digest());
    let mut transcript = Transcript::new(manifest, &record)?;
    // The round-1 commitments and the Flock commitments depend only on the witness, so they
    // run concurrently (both are memory bound and leave threads idle on their own); the
    // transcript absorbs them in event order afterwards. Each step's time is its own
    // closure's, so in mode A the two overlap in wall time. `overlap`, not `rayon::join`:
    // the commit side must not run on a rayon worker that then blocks on its commit threads
    // (a one-thread pool would deadlock).
    let sched_trace = std::env::var("AGG_SCHED_TRACE").as_deref() == Ok("1");
    let (
        (round1, round1_time, akita_start, akita_end),
        (flock_commitments, keccak_time, flock_start, flock_end),
    ) = crate::pcs::overlap(
        || {
            let start = Instant::now();
            let out = committer.commit_round1(main);
            phase_stamp("round1-commit");
            let end = Instant::now();
            (out, end - start, start, end)
        },
        || {
            let start = Instant::now();
            let out = keccak.commitments(segments);
            phase_stamp("flock-commit");
            let end = Instant::now();
            (out, end - start, start, end)
        },
    );
    if sched_trace {
        let overlap = akita_end
            .min(flock_end)
            .saturating_duration_since(akita_start.max(flock_start));
        let wall = akita_end.max(flock_end) - akita_start.min(flock_start);
        eprintln!(
            "[sched] item=commits akita_ms={:.3} flock_ms={:.3} overlap_ms={:.3} wall_ms={:.3}",
            round1_time.as_secs_f64() * 1e3,
            keccak_time.as_secs_f64() * 1e3,
            overlap.as_secs_f64() * 1e3,
            wall.as_secs_f64() * 1e3
        );
    }
    let [main16_round1, main_round1] = round1?;
    let main16_commitment = main16_round1.serialized()?;
    let main_commitment = main_round1.serialized()?;
    transcript.absorb(commitment_site(GroupRole::Main16), 1, &main16_commitment)?;
    transcript.absorb(commitment_site(GroupRole::Main), 1, &main_commitment)?;
    if flock_commitments.len() != segments {
        return Err(ProveError::Internal(
            "one Flock commitment per segment".into(),
        ));
    }
    transcript.absorb_items(FLOCK_COMMIT_SITE, &flock_commitments)?;
    let bridge_seed: [u8; 32] = transcript
        .challenge_bytes(BRIDGE_SEED_SITE, 1)?
        .try_into()
        .expect("32 bytes");
    // The projection matrix of v, drawn directly as the site's answer (`projection`).
    let psi = transcript.challenge_psi()?;
    clock.lap();
    phase_stamp("events1-3");
    timings.round1 = round1_time;
    timings.keccak = keccak_time;

    // Round-2 commitments, Flock seeds, final vectors and share items.
    let matrices = bridge.matrices_for(manifest.set(), &bridge_seed);
    let mut bridge_tables = BridgeGroupTables::build(manifest, main, &matrices, &psi)?;
    if let Some(tamper) = hooks.tamper_bridge {
        tamper(manifest, &mut bridge_tables);
        // A tampered (or replaced) table set no longer matches `main`: the parities are
        // recomputed from the committed bits.
        bridge_tables.drop_parity1_cache();
    }
    // The honest level-1 parities move to the stage tables; only `ProverTables` reads them.
    let parity1_cache = bridge_tables.take_parity1();
    let tables = SourceTables {
        manifest,
        main,
        bridge: &bridge_tables,
    };
    let round2 = committer.commit_round2(&bridge_tables)?;
    let mut bridge_commitments: [Vec<u8>; 5] = Default::default();
    for (slot, group) in bridge_commitments.iter_mut().zip(&round2) {
        *slot = group.serialized()?;
        transcript.absorb(commitment_site(group.group), 1, slot)?;
    }
    let flock_seeds = seeds_of(&transcript.challenge_bytes(FLOCK_SEEDS_SITE, segments)?);
    let final_bits: Vec<Vec<u8>> = (0..segments)
        .map(|sigma| bridge_tables.segment_y(sigma).to_vec())
        .collect();
    let final_vectors: Vec<[u8; 512]> = final_bits.iter().map(|bits| pack_final_vector(bits)).collect();
    transcript.absorb(FINAL_VECTORS_SITE, segments, &final_vectors.concat())?;
    timings.round2 = clock.lap();
    phase_stamp("round2");
    let flock_shares = keccak.shares(&matrices, &final_bits)?;
    if flock_shares.len() != segments {
        return Err(ProveError::Internal("one share item per segment".into()));
    }
    transcript.absorb_items(FLOCK_SHARES_SITE, &flock_shares)?;
    let flock_record = flock_record_digest(&final_vectors, &flock_shares);
    timings.keccak += clock.lap();
    phase_stamp("shares");

    let commitments: [GroupCommitment; GROUP_COUNT] = {
        let [l1, l2, l3, k, m] = round2;
        [main16_round1, main_round1, l1, l2, l3, k, m]
    };
    // The fork: the Flock proofs depend on nothing after the final-vector step, and no PIOP
    // or Akita challenge depends on them (the transcript never absorbs them; their own
    // challengers bind them to the statement), so they are proven on their own pool while
    // the parity tables, the lookup, the pre-stage claims, the stages (stages 1 to 3) and,
    // by default, the Akita opening run on the PIOP's share of the threads.
    let flock_threads = if keccak.proves_concurrently() { flock_pool_threads() } else { 0 };
    let prove_flock = || {
        let out = keccak.prove(&bridge_seed, &matrices, &flock_seeds, &final_bits, &flock_record);
        phase_stamp("flock-prove");
        out
    };
    let (flock_proofs, piop_side) = if flock_threads > 0 {
        let fork = Instant::now();
        let (unchecked, swap_join) = (hooks.unchecked, hooks.swap_join);
        let beside = opening_beside_flock().then(|| opening_threads_beside(flock_threads));
        let ((flock_proofs, flock_time), piop_side) =
            fork_join(flock_threads, prove_flock, || {
                let prover_tables =
                    ProverTables::from_parity1_cache(tables, parity1_cache, &matrices);
                phase_stamp("parity-tables");
                let (proof, run) = prove_piop_with(
                    manifest,
                    tables,
                    Some(prover_tables),
                    &matrices,
                    public,
                    &mut transcript,
                    unchecked,
                    swap_join,
                    // The stage pool default applies only beside a running Flock fork.
                    true,
                )?;
                // Prepare runs on this pool, so it is part of the fork-join wall time.
                let (claims, prepared) =
                    prepare_opening(&committer, manifest, commitments, &proof, &run)?;
                phase_stamp("opening-prepare");
                let state = match beside {
                    Some(threads) => OpeningState::Finished(finish_opening(
                        &committer,
                        tables,
                        &mut transcript,
                        &claims,
                        prepared,
                        threads,
                        unchecked,
                    )?),
                    None => OpeningState::Prepared(prepared),
                };
                Ok::<_, ProveError>((proof, run, claims, state))
            });
        timings.fork_join = fork.elapsed();
        phase_stamp("join");
        timings.flock_prove = flock_time;
        timings.keccak += flock_time;
        clock.lap();
        (flock_proofs, piop_side)
    } else {
        // No fork: the Flock proofs first, with the parity tables of the stages (which need
        // no challenge) built beside them, then the PIOP.
        let (flock_proofs, prover_tables) = crate::pcs::overlap(
            || {
                let start = Instant::now();
                (prove_flock(), start.elapsed())
            },
            || {
                let out = ProverTables::from_parity1_cache(tables, parity1_cache, &matrices);
                phase_stamp("parity-tables");
                out
            },
        );
        let (flock_proofs, flock_time) = flock_proofs;
        timings.flock_prove = flock_time;
        timings.keccak += clock.lap();
        let flock_proofs = flock_proofs?;
        let piop_side = prove_piop_with(
            manifest,
            tables,
            Some(prover_tables),
            &matrices,
            public,
            &mut transcript,
            hooks.unchecked,
            hooks.swap_join,
            false,
        )
        .and_then(|(proof, run)| {
            let (claims, prepared) =
                prepare_opening(&committer, manifest, commitments, &proof, &run)?;
            Ok((proof, run, claims, OpeningState::Prepared(prepared)))
        });
        phase_stamp("opening-prepare");
        clock.lap();
        (Ok(flock_proofs), piop_side)
    };
    timings.flock_threads = flock_threads;
    let (piop, run, claims, state) = piop_side?;
    let flock_proofs = flock_proofs?;
    timings.lookup = run.lookup;
    timings.prestage = run.prestage;
    timings.stage1 = run.stage1;
    timings.stage2 = run.stage2;
    timings.stage3 = run.stage3;
    if flock_proofs.len() != segments {
        return Err(ProveError::Internal("one Flock proof per segment".into()));
    }

    // The opening, unless it ran beside the Flock proofs.
    let FinishedOpening {
        session,
        work,
        opening,
        threads: opening_threads,
        time: opening_time,
    } = match state {
        OpeningState::Finished(finished) => finished,
        OpeningState::Prepared(prepared) => finish_opening(
            &committer,
            tables,
            &mut transcript,
            &claims,
            prepared,
            opening_pool_threads(),
            hooks.unchecked,
        )?,
    };
    timings.opening_threads = opening_threads;
    timings.opening = opening_time;
    let proof = AggregateProof {
        set: manifest.set(),
        relation: manifest.relation(),
        records: n,
        c_tilde: records.iter().map(|r| r.2.as_ref().to_vec()).collect(),
        main16_commitment,
        main_commitment,
        flock_commitments,
        bridge_commitments,
        final_vectors,
        flock_shares,
        flock_proofs,
        lookup: piop.lookup,
        multiplicities: piop.multiplicities,
        prestage: piop.prestage,
        stage1: piop.stage1,
        stage2: piop.stage2,
        stage3: piop.stage3,
        work,
        opening,
    };
    timings.total = total.elapsed();
    Ok(ProverOutput {
        proof,
        bridge_seed,
        psi,
        flock_seeds,
        session,
        points: (run.points, run.r3),
        pi_m: run.pi_m,
        timings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_matrices_are_the_spec_derivation_of_the_seed() {
        let (a, b) = ([0x11_u8; 32], [0x22_u8; 32]);
        let derived = SeededMatrices.matrices(&a);
        assert!(matches!(derived, Cow::Owned(_)));
        assert_eq!(*derived, agg_bridge::seed::derive_matrices(&a));
        assert!(derived.has_spec_shapes() && derived.row_weights_ok());
        let other = SeededMatrices.matrices(&b);
        assert_ne!(derived.m1, other.m1);
        assert_ne!(derived.m2, other.m2);
        assert_ne!(derived.m3, other.m3);
        assert!(SeededMatrices.seed_derived());
    }

    #[test]
    fn fixed_matrices_ignore_the_seed_and_are_not_seed_derived() {
        let fixed = FixedMatrices(BridgeMatrices::random_for_tests(3));
        assert_eq!(*fixed.matrices(&[0; 32]), *fixed.matrices(&[1; 32]));
        assert!(!fixed.seed_derived());
    }

    #[test]
    fn cpu_lists_parse_ranges_and_singletons() {
        assert_eq!(parse_cpu_list("0-3,5,7-9"), Some(vec![0, 1, 2, 3, 5, 7, 8, 9]));
        assert_eq!(parse_cpu_list("8"), Some(vec![8]));
        assert_eq!(parse_cpu_list("4-6"), Some(vec![4, 5, 6]));
        assert_eq!(parse_cpu_list(""), None);
        assert_eq!(parse_cpu_list("3-1"), None);
        assert_eq!(parse_cpu_list("0-x"), None);
        assert!(allowed_cpus().is_some_and(|cpus| !cpus.is_empty()));
    }

    #[test]
    fn the_default_stage_pool_follows_the_piop_pool() {
        // No fork, or fewer than four PIOP threads: the shared pool.
        for threads in [1, 2, 3, 4, 8, 16, 32, 64] {
            assert_eq!(stage_threads(false, threads, 8, 64), None);
        }
        assert_eq!(stage_threads(true, 1, 1, 64), None);
        assert_eq!(stage_threads(true, 2, 1, 64), None);
        assert_eq!(stage_threads(true, 3, 2, 64), None);
        assert_eq!(stage_threads(true, 4, 2, 64), Some(2));
        assert_eq!(stage_threads(true, 8, 3, 64), Some(3));
        assert_eq!(stage_threads(true, 24, 9, 64), Some(9));
        // The measured PIOP sizes keep their pools even on a 16-CPU host.
        assert_eq!(stage_threads(true, 16, 6, 16), Some(6));
        assert_eq!(stage_threads(true, 32, 12, 32), Some(16));
        // Elsewhere the pool is capped to fit PIOP + stage + Flock in the CPU count.
        assert_eq!(stage_threads(true, 36, 14, 60), Some(10));
        assert_eq!(stage_threads(true, 36, 14, 48), None);
        assert_eq!(stage_threads(true, 64, 24, 64), None);
        assert_eq!(stage_threads(true, 64, 24, 128), Some(32));
    }

    #[test]
    fn the_default_flock_pool_follows_the_kernel_width() {
        for threads in [1, 4, 8, 16, 32, 64, 96] {
            assert_eq!(flock_default_for(threads, true), (3 * threads).div_ceil(8));
            assert_eq!(flock_default_for(threads, false), (7 * threads).div_ceil(16));
        }
        // The measured sizes: 12 (AVX-512) or 14 (AVX2) at 32 threads,
        // and 6 or 7 at 16 threads.
        assert_eq!(flock_default_for(32, true), 12);
        assert_eq!(flock_default_for(32, false), 14);
        assert_eq!(flock_default_for(16, true), 6);
        assert_eq!(flock_default_for(16, false), 7);
    }
}
