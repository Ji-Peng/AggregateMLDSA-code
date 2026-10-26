//! The aggregate prover: commitment rounds, bridge seed, Keccak side, the lookup into
//! `T_16`, the three stages and the Akita opening in transcript order (spec Sections 7, 10;
//! lookup layout L16, `docs/lookup.md`). [`prove_piop`] is the PIOP part after the
//! commitments (lookup, pre-stage claims, stages 1 to 3), shared with the stage tests.
//!
//! Fork-join (deviation K22, lever P1 of `docs/perf.md`): event 6 absorbs the final vectors
//! and the Keccak side's share items only; the Flock proofs are absorbed at the join event
//! after the stage-3 claims. No PIOP challenge depends on them, so after event 6 the prover
//! runs [`KeccakSide::prove`] on its own rayon pool ([`flock_pool_threads`]) while the PIOP
//! (lookup, pre-stage claims, stages) runs on the global pool, then absorbs the proofs.
//!
//! Two hooks connect the parts owned by other crates: [`BridgeDerivation`] maps the bridge
//! seed `sd` to the matrices ([`SeededMatrices`] is the derivation of spec Section 9.1,
//! `agg_bridge::seed`) and [`KeccakSide`] produces the Flock commitments and sub-proofs
//! (`agg-keccak`, spec Section 8; `mode_b::PublicBits` in mode B). The stand-ins
//! `FixedMatrices` (matrices that ignore the seed) and `NoKeccak` (Flock profile 0) exist
//! only in unit tests and with the `test-hooks` feature (`docs/deviations.md` P4).

use std::borrow::Cow;
use std::time::{Duration, Instant};

use agg_pcs_config::commit::{CommitError, GroupCommitment};
use agg_witness::bridge::BridgeMatrices;
use agg_witness::instance::PublicInstance;
use agg_witness::params::C_TILDE_BYTES;

use crate::binding::{instance_record, public_records_binding, FlockProfile};
use crate::claims::{MultiplicityClaims, PreStageClaims};
use crate::lookup::{prove_in, LookupError, LookupProof, LookupShape, Table};
use crate::manifest::{CommitmentManifest, GroupRole, Relation, LOOKUP_CHALLENGES};
use crate::opening::opening_claims;
use crate::pcs::{PcsError, TableCommitter};
use crate::pow::{KeccakPlan, PowPlan};
use crate::public_data::PublicData;
use crate::source_tables::{BridgeGroupTables, MainTables, SourceTableError, SourceTables};
use crate::stage::{suffix, Points, StageError};
use crate::stage1::Stage1Proof;
use crate::stage2::Stage2Proof;
use crate::stage3::{LookupInputs, Stage3Proof};
use crate::tables::ProverTables;
use crate::transcript::{
    commitment_site, flock_record_digest, Transcript, TranscriptError, BRIDGE_SEED_SITE,
    FINAL_VECTORS_SITE, FLOCK_COMMIT_SITE, FLOCK_PROOFS_SITE, FLOCK_SEEDS_SITE,
    FLOCK_SHARES_SITE, MULTIPLICITY_SITE, PRESTAGE_SITE, TAU_SITE,
};
use crate::wire::{pack_final_vector, AggregateProof};
use crate::{stage1, stage2, stage3, OpeningField as E};

/// Maps the bridge seed to the three matrices (spec Section 9.1).
pub trait BridgeDerivation {
    /// The matrices of `seed` (borrowed when they do not depend on it).
    fn matrices(&self, seed: &[u8; 32]) -> Cow<'_, BridgeMatrices>;

    /// True only when [`Self::matrices`] is the derivation of spec Section 9.1, so that the
    /// matrices are uniform and independent of everything absorbed before `sd` and the
    /// bound of spec 9.4 applies. Mode A (Flock profiles `Fast` and `LaBinius`) is refused
    /// by [`prove`] and by `verifier::verify` otherwise.
    fn seed_derived(&self) -> bool {
        false
    }
}

/// The matrices of spec Section 9.1: `M_1, M_2, M_3` derived from `sd` with cSHAKE256
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
}

/// Fixed matrices whatever the seed (test stand-in: unit tests and the `test-hooks`
/// feature only). The prover knows them before it commits `G_main`, so the bound of spec
/// 9.4 does not hold; mode A refuses them, mode B stays sound through its direct bit check
/// (`docs/deviations.md` P9).
#[cfg(any(test, feature = "test-hooks"))]
pub struct FixedMatrices(pub BridgeMatrices);

#[cfg(any(test, feature = "test-hooks"))]
impl BridgeDerivation for FixedMatrices {
    fn matrices(&self, _seed: &[u8; 32]) -> Cow<'_, BridgeMatrices> {
        Cow::Borrowed(&self.0)
    }
}

/// The Keccak argument's transcript contributions (spec Section 10: events 2 and 6 and the
/// join event).
pub trait KeccakSide {
    /// The Keccak side's part of the proof-of-work plan (its tied parts' tie sites,
    /// `docs/soundness.md` Section 6.4), a function of the public batch geometry, needed
    /// before event 0. The default (no tied parts) fits a side without Flock sub-proofs.
    fn plan(&self, _segments: usize) -> KeccakPlan {
        KeccakPlan::default()
    }

    /// The `S` Flock commitments, before the bridge seed.
    fn commitments(&self, segments: usize) -> Vec<Vec<u8>>;

    /// The `S` share items of event 6, absorbed after the final vectors `Y_sigma` (one bit
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
    /// then skips the fork of deviation K22: the transcript and the proof are the same
    /// either way.
    fn proves_concurrently(&self) -> bool {
        true
    }

    /// The `S` Flock proofs of the join event. They may be computed concurrently with the
    /// PIOP (deviation K22), so they depend on nothing after event 6: `bridge_seed` is `sd`
    /// (event 3) and `matrices` its `M_1, M_2, M_3`, `seeds` are the challenger seeds
    /// `fs_sigma` (event 5), `final_vectors` the `Y_sigma` and `record` the digest of the
    /// event-6 record ([`crate::transcript::flock_record_digest`]: the final vectors and the
    /// share items), which the side MUST bind into every challenger of its sub-proofs, so
    /// that their challenge addresses encode the input of the lookup challenge `X`
    /// (`docs/soundness.md` 12.2, Lemma 12.3 (a)).
    ///
    /// # Errors
    /// [`ProveError::Keccak`] if a sub-proof cannot be made (an exhausted proof-of-work
    /// search, `docs/soundness.md` 6.5).
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

/// Threads of the Flock pool of the fork-join (deviation K22): `AGG_FLOCK_THREADS` if set
/// (`0`: no fork, the Flock proofs and then the PIOP on the caller's pool), else half the
/// global pool rounded up (8 of 16 threads on the cloud machine, the better split of
/// `ref/notes/bench-cloud.md` Phase 10 (6); a global pool of 8 threads alone was slower,
/// Phase 10 (1)). A value that is not a number is reported once on stderr and replaced by
/// the default (a measurement must not silently run another pool size; the bench report
/// prints the size used); a larger value than [`max_flock_threads`] is clamped to it. `0`
/// on a rayon worker or with a one-thread global pool, where an extra pool gains nothing or a
/// blocked worker could deadlock (deviation P12).
#[must_use]
pub fn flock_pool_threads() -> usize {
    if crate::pcs::on_calling_thread() {
        return 0;
    }
    let default = || rayon::current_num_threads().div_ceil(2);
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

/// The largest Flock pool: the global pool or the machine's logical CPUs, whichever is
/// larger (more threads than that only oversubscribe the cores).
#[must_use]
pub fn max_flock_threads() -> usize {
    let cpus = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    cpus.max(rayon::current_num_threads())
}

fn warn_once(message: &str) {
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| eprintln!("agg-piop: {message}"));
}

/// Stack size of the Flock pool's workers (the Flock prover nests large frames; a virtual
/// reservation, as Flock's own 8 MiB pool and the commit threads).
const FLOCK_POOL_STACK: usize = 64 << 20;

/// Run `flock` on a fresh rayon pool of `threads` threads (on its own OS thread) while
/// `piop` runs on the caller (its parallel loops on the global pool); return both results
/// and the wall time of `flock`. The caller is not a rayon worker ([`flock_pool_threads`]
/// is 0 there), so no worker blocks on the OS thread.
fn fork_join<A: Send, B>(
    threads: usize,
    flock: impl FnOnce() -> A + Send,
    piop: impl FnOnce() -> B,
) -> ((A, Duration), B) {
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
                (out, start.elapsed())
            })
            .expect("spawn the Flock thread");
        let b = piop();
        (crate::pcs::join_os_thread(handle), b)
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

/// The prover's inputs: the public instance of the records and their signatures.
pub struct ProverInputs<'a> {
    /// The public instance (keys expanded, `mu`, `c`, `T`).
    pub instance: &'a PublicInstance,
    /// `(pk, M, c~)` per live record, in order (bound by the record digest).
    pub records: &'a [(&'a [u8], &'a [u8], [u8; C_TILDE_BYTES])],
    /// The signatures, in order.
    pub signatures: &'a [&'a [u8]],
    /// The relation to prove (`docs/h-exact.md` Section 1): H-u or H-exact.
    pub relation: Relation,
}

/// Wall time of each prover step (spec Section 10 event order).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProveTimings {
    /// Witness derivation and the round-1 tables (`G_main16`, `G_main4`).
    pub tables: Duration,
    /// Akita setup from the work plan.
    pub setup: Duration,
    /// Event 1: the round-1 commitments (concurrent with the event-2 commitments).
    pub round1: Duration,
    /// Events 2, 6 and the join: the Keccak side (commitments, shares and proofs; the proofs
    /// overlap the PIOP steps under the fork-join).
    pub keccak: Duration,
    /// The Flock proofs alone (on their own pool under the fork-join).
    pub flock_prove: Duration,
    /// Wall time from the fork to the join (Flock proofs beside the parity tables, the
    /// lookup, the pre-stage claims and the stages); zero without the fork.
    pub fork_join: Duration,
    /// Threads of the Flock pool (0: no fork, [`flock_pool_threads`]).
    pub flock_threads: usize,
    /// Matrices, the round-2 tables (parities, carries, multiplicities) and their
    /// commitments (event 4).
    pub round2: Duration,
    /// Event 7: the lookup (fraction trees, sessions, multiplicity claims).
    pub lookup: Duration,
    /// Events 8 and 9: `tau` and the pre-stage claims.
    pub prestage: Duration,
    /// Stage 1, including the `w_2, w_3` weight tables.
    pub stage1: Duration,
    /// Stage 2, including the `w_1` weight table.
    pub stage2: Duration,
    /// Stage 3, including the `w_A` weight table.
    pub stage3: Duration,
    /// Event 13: the batched Akita opening.
    pub opening: Duration,
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
             stage 3 {:.1} ms, opening {:.1} ms, total {:.1} ms",
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
            ms(self.total)
        )
    }
}

/// What the prover produced.
pub struct ProverOutput {
    /// The proof.
    pub proof: AggregateProof,
    /// The bridge seed `sd` (event 3).
    pub bridge_seed: [u8; 32],
    /// The Flock challenger seeds `fs_sigma` (event 5).
    pub flock_seeds: Vec<[u8; 32]>,
    /// The PCS session digest (event 12).
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

/// Mode A (Flock profiles `Fast`, `LaBinius`) only with the matrices of spec 9.1: a proof
/// over other matrices would be refused by the verifier.
fn check_bridge(bridge: &impl BridgeDerivation, flock: FlockProfile) -> Result<(), ProveError> {
    if flock.is_mode_a() && !bridge.seed_derived() {
        return Err(ProveError::Internal(
            "mode A (Flock profiles Fast, LaBinius) needs the seed-derived matrices of spec 9.1"
                .into(),
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

/// The PIOP part of a proof (events 7 to 12): the lookup, the multiplicity-digit claims,
/// the pre-stage claims and the three stages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiopProof {
    /// Event 7: the lookup.
    pub lookup: LookupProof,
    /// Event 7: `m_k(pi_m)`.
    pub multiplicities: MultiplicityClaims,
    /// Event 9.
    pub prestage: PreStageClaims,
    /// Event 10.
    pub stage1: Stage1Proof,
    /// Event 11.
    pub stage2: Stage2Proof,
    /// Event 12.
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
    /// The lookup (event 7).
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

/// Prove the PIOP part (events 7 to 12) on a transcript that has absorbed events 0 to 6 (the
/// commitments, the seeds, the final vectors and the Keccak side's share items). `unchecked` skips the honest prover's self-checks
/// (lookup root identity and session claims, stage claimed sums) so that tests can hand a
/// tampered witness to the verifier.
pub fn prove_piop(
    manifest: &CommitmentManifest,
    tables: SourceTables<'_>,
    matrices: &BridgeMatrices,
    public: &PublicData,
    transcript: &mut Transcript,
    unchecked: bool,
) -> Result<(PiopProof, PiopRun), ProveError> {
    prove_piop_with(manifest, tables, None, matrices, public, transcript, unchecked)
}

/// [`prove_piop`] with the challenge-independent prover tables (the parities `y^(1)`)
/// built beforehand, e.g. while the Flock span is proven (prover only).
#[allow(clippy::too_many_lines)]
fn prove_piop_with<'a>(
    manifest: &CommitmentManifest,
    tables: SourceTables<'a>,
    prebuilt: Option<ProverTables<'a>>,
    matrices: &BridgeMatrices,
    public: &PublicData,
    transcript: &mut Transcript,
    unchecked: bool,
) -> Result<(PiopProof, PiopRun), ProveError> {
    let mut clock = Lap::new();
    let mut run = PiopRun::default();
    let trace = std::env::var_os("AGG_STAGE_TRACE").is_some();

    // Event 7: the lookup into T_16 (docs/lookup.md Section 1) and the digit claims of m.
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
    let (lookup_inputs, stack_points) = LookupInputs::new(manifest, &output.claims);
    if trace {
        let sessions: f64 = output.stats.sessions.iter().sum();
        eprintln!(
            "lookup: trees {:.1} ms, sessions {:.1} ms, stored layers {:.1} MiB",
            output.stats.trees * 1e3,
            sessions * 1e3,
            output.stats.layer_bytes as f64 / f64::from(1 << 20)
        );
    }
    run.lookup = clock.lap();

    // Events 8 and 9: tau and the pre-stage claims.
    let mut points = Points {
        tau: transcript.challenge_extensions(TAU_SITE, manifest.d_main())?,
        lookup: stack_points,
        ..Points::default()
    };
    let prestage = PreStageClaims::evaluate(manifest, tables, &points.tau);
    transcript.absorb_extensions(PRESTAGE_SITE, &prestage.to_vec())?;
    let y_at_tau3 = agg_bridge::final_vector::evaluate(
        tables.bridge.public_y(),
        &suffix(&points.tau, manifest.group(GroupRole::Level3).logical_vars),
    );
    run.prestage = clock.lap();

    // Stages 1 to 3.
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
    run.stage1 = clock.lap();
    let weights_clock = Instant::now();
    prover_tables.set_r1(matrices, &points.r1);
    if trace {
        eprintln!("weights: w_1 {:.1} ms", weights_clock.elapsed().as_secs_f64() * 1e3);
    }
    let (stage2_proof, stage2) = if unchecked {
        stage2::prove_unchecked(manifest, &prover_tables, &points, &stage1, transcript)?
    } else {
        stage2::prove(manifest, &prover_tables, &points, &stage1, transcript)?
    };
    points.r2 = stage2.point.clone();
    run.stage2 = clock.lap();
    let weights_clock = Instant::now();
    prover_tables.set_r2(public, &points.r2);
    if trace {
        eprintln!("weights: w_A {:.1} ms", weights_clock.elapsed().as_secs_f64() * 1e3);
    }
    let prove3 = if unchecked {
        stage3::prove_unchecked
    } else {
        stage3::prove
    };
    let (stage3_proof, stage3) = prove3(
        manifest,
        &prover_tables,
        public,
        &points,
        &prestage,
        &stage1,
        &stage2,
        &lookup_inputs,
        transcript,
    )?;
    drop(prover_tables);
    run.stage3 = clock.lap();
    run.points = points;
    run.r3 = stage3.point;
    run.pi_m = pi_m;
    Ok((
        PiopProof {
            lookup: output.proof,
            multiplicities,
            prestage,
            stage1: stage1_proof,
            stage2: stage2_proof,
            stage3: stage3_proof,
        },
        run,
    ))
}

/// Prove from given round-1 tables (events 0 to 13 with the join). `hooks` are for
/// adversarial tests; an honest caller passes `ProveHooks::default()`.
#[allow(clippy::too_many_lines)]
pub fn prove_with_tables(
    inputs: &ProverInputs<'_>,
    manifest: &CommitmentManifest,
    main: &MainTables,
    bridge: &impl BridgeDerivation,
    keccak: &(impl KeccakSide + Sync),
    flock: FlockProfile,
    hooks: &ProveHooks<'_>,
) -> Result<ProverOutput, ProveError> {
    let total = Instant::now();
    let mut clock = Lap::new();
    let mut timings = ProveTimings::default();
    let n = inputs.records.len();
    if manifest.records() != n || main.geometry().n != n {
        return Err(ProveError::Internal("tables and records differ".into()));
    }
    check_bridge(bridge, flock)?;
    let segments = manifest.segments();
    let public = PublicData::from_instance(inputs.instance);
    let committer = TableCommitter::new(manifest.clone())?;
    timings.setup = clock.lap();

    // Events 0 to 3: instance (with the proof-of-work plan's digest, docs/soundness.md 6.4),
    // round-1 commitments, Flock commitments, bridge seed.
    let digest = public_records_binding(inputs.records);
    let plan = PowPlan::new(manifest, flock, &keccak.plan(segments));
    let record = instance_record(manifest, &digest, flock, &plan.digest());
    let mut transcript = Transcript::new(manifest, &record)?;
    // The round-1 commitments and the Flock commitments depend only on the witness, so they
    // run concurrently (both are memory bound and leave threads idle on their own); the
    // transcript absorbs them in event order afterwards. Each step's time is its own
    // closure's, so in mode A the two overlap in wall time. `overlap`, not `rayon::join`:
    // the commit side must not run on a rayon worker that then blocks on its commit threads
    // (one-thread pools deadlocked; deviation P12).
    let ((round1, round1_time), (flock_commitments, keccak_time)) = crate::pcs::overlap(
        || {
            let start = Instant::now();
            (committer.commit_round1(main), start.elapsed())
        },
        || {
            let start = Instant::now();
            (keccak.commitments(segments), start.elapsed())
        },
    );
    let [main16_round1, main_round1] = round1?;
    let main16_commitment = main16_round1.serialized()?;
    let main_commitment = main_round1.serialized()?;
    transcript.absorb(commitment_site(GroupRole::Main16), 1, &main16_commitment)?;
    transcript.absorb(commitment_site(GroupRole::Main), 1, &main_commitment)?;
    if flock_commitments.len() != segments {
        return Err(ProveError::Internal("one Flock commitment per segment".into()));
    }
    transcript.absorb_items(FLOCK_COMMIT_SITE, &flock_commitments)?;
    let bridge_seed: [u8; 32] = transcript
        .challenge_bytes(BRIDGE_SEED_SITE, 1)?
        .try_into()
        .expect("32 bytes");
    clock.lap();
    timings.round1 = round1_time;
    timings.keccak = keccak_time;

    // Events 4 to 6: round-2 commitments, Flock seeds, final vectors and share items.
    let matrices = bridge.matrices(&bridge_seed);
    let mut bridge_tables = BridgeGroupTables::build(manifest, main, &matrices)?;
    if let Some(tamper) = hooks.tamper_bridge {
        tamper(manifest, &mut bridge_tables);
    }
    let round2 = committer.commit_round2(&bridge_tables)?;
    let mut bridge_commitments: [Vec<u8>; 4] = Default::default();
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
    let flock_shares = keccak.shares(&matrices, &final_bits)?;
    if flock_shares.len() != segments {
        return Err(ProveError::Internal("one share item per segment".into()));
    }
    transcript.absorb_items(FLOCK_SHARES_SITE, &flock_shares)?;
    let flock_record = flock_record_digest(&final_vectors, &flock_shares);
    timings.keccak += clock.lap();
    let tables = SourceTables {
        manifest,
        main,
        bridge: &bridge_tables,
    };

    // The fork (deviation K22): the Flock proofs depend on nothing after event 6, and no
    // PIOP challenge depends on them, so they are proven on their own pool while the parity
    // tables, the lookup (event 7), the pre-stage claims (events 8, 9) and the stages
    // (events 10 to 12) run on the global pool; the join absorbs them after stage 3.
    let flock_threads = if keccak.proves_concurrently() { flock_pool_threads() } else { 0 };
    let prove_flock = || keccak.prove(&bridge_seed, &matrices, &flock_seeds, &final_bits, &flock_record);
    let (flock_proofs, piop) = if flock_threads > 0 {
        let fork = Instant::now();
        let ((flock_proofs, flock_time), piop) = fork_join(flock_threads, prove_flock, || {
            let prover_tables = ProverTables::new(tables, &matrices);
            prove_piop_with(
                manifest,
                tables,
                Some(prover_tables),
                &matrices,
                &public,
                &mut transcript,
                hooks.unchecked,
            )
        });
        timings.fork_join = fork.elapsed();
        timings.flock_prove = flock_time;
        timings.keccak += flock_time;
        clock.lap();
        (flock_proofs, piop)
    } else {
        // No fork: the Flock proofs first, with the parity tables of the stages (which need
        // no challenge) built beside them, then the PIOP.
        let (flock_proofs, prover_tables) = crate::pcs::overlap(
            || {
                let start = Instant::now();
                (prove_flock(), start.elapsed())
            },
            || ProverTables::new(tables, &matrices),
        );
        let (flock_proofs, flock_time) = flock_proofs;
        timings.flock_prove = flock_time;
        timings.keccak += clock.lap();
        let flock_proofs = flock_proofs?;
        let piop = prove_piop_with(
            manifest,
            tables,
            Some(prover_tables),
            &matrices,
            &public,
            &mut transcript,
            hooks.unchecked,
        );
        clock.lap();
        (Ok(flock_proofs), piop)
    };
    timings.flock_threads = flock_threads;
    let (piop, run) = piop?;
    let flock_proofs = flock_proofs?;
    timings.lookup = run.lookup;
    timings.prestage = run.prestage;
    timings.stage1 = run.stage1;
    timings.stage2 = run.stage2;
    timings.stage3 = run.stage3;

    // The join: the Flock proofs, after the stage-3 claims.
    if flock_proofs.len() != segments {
        return Err(ProveError::Internal("one Flock proof per segment".into()));
    }
    transcript.absorb_items(FLOCK_PROOFS_SITE, &flock_proofs)?;

    // Event 13: the opening.
    let session = transcript.pcs_session()?;
    transcript.ensure_complete()?;
    let work = transcript.work_stream();
    let claims = opening_claims(manifest, &run.r3, &piop.stage3.claims, &run.pi_m, &piop.multiplicities)?;
    // The stage-3 terminal values are the multilinear extensions of the committed tables at
    // the group points (the kernel's terminal values; `tests/prove_verify.rs` checks them
    // against the table evaluations), so the opening takes them directly instead of
    // evaluating every table again.
    debug_assert!(
        hooks.unchecked
            || crate::pcs::opening_claims(tables, &claims.clone().map(|c| c.point))
                .is_ok_and(|recomputed| recomputed == claims),
        "stage-3 terminal values differ from the table evaluations"
    );
    let commitments: [GroupCommitment; 6] = {
        let [l1, l2, l3, m] = round2;
        [main16_round1, main_round1, l1, l2, l3, m]
    };
    let opening = committer.open_claims(commitments, &claims, &session)?;
    timings.opening = clock.lap();
    let proof = AggregateProof {
        relation: manifest.relation(),
        records: n,
        c_tilde: inputs.records.iter().map(|r| r.2).collect(),
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
}
