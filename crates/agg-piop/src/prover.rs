//! The aggregate prover: commitment rounds, bridge seed, Keccak side, the three stages
//! and the Akita opening in transcript order (spec Sections 7, 10).
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
use crate::claims::PreStageClaims;
use crate::manifest::{CommitmentManifest, GroupRole};
use crate::opening::opening_claims;
use crate::pcs::{PcsError, TableCommitter};
use crate::public_data::PublicData;
use crate::source_tables::{BridgeGroupTables, MainTables, SourceTableError, SourceTables};
use crate::stage::{suffix, Points, StageError};
use crate::tables::ProverTables;
use crate::transcript::{
    commitment_site, Transcript, TranscriptError, BRIDGE_SEED_SITE, FINAL_VECTORS_SITE,
    FLOCK_COMMIT_SITE, FLOCK_PROOFS_SITE, FLOCK_SEEDS_SITE, PRESTAGE_SITE, TAU_SITE,
};
use crate::wire::{pack_final_vector, AggregateProof};
use crate::{stage1, stage2, stage3};

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

/// The Keccak argument's two transcript contributions (spec Section 10, events 2 and 6).
pub trait KeccakSide {
    /// The `S` Flock commitments, before the bridge seed.
    fn commitments(&self, segments: usize) -> Vec<Vec<u8>>;
    /// The `S` Flock proofs, after the challenger seeds `fs_sigma` and with the final
    /// vectors `Y_sigma` known. `bridge_seed` is `sd` (event 3) and `matrices` its
    /// `M_1, M_2, M_3`: the bridge slot's rows are generated from them and its circuit
    /// digest binds `sd` (docs/bridge.md Section 5, `docs/deviations.md` P11).
    fn prove(
        &self,
        bridge_seed: &[u8; 32],
        matrices: &BridgeMatrices,
        seeds: &[[u8; 32]],
        final_vectors: &[Vec<u8>],
    ) -> Vec<Vec<u8>>;
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

    fn prove(
        &self,
        _bridge_seed: &[u8; 32],
        _matrices: &BridgeMatrices,
        seeds: &[[u8; 32]],
        _final_vectors: &[Vec<u8>],
    ) -> Vec<Vec<u8>> {
        vec![Vec::new(); seeds.len()]
    }
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

/// The prover's inputs: the public instance of the records and their signatures.
pub struct ProverInputs<'a> {
    /// The public instance (keys expanded, `mu`, `c`, `T`).
    pub instance: &'a PublicInstance,
    /// `(pk, M, c~)` per live record, in order (bound by the record digest).
    pub records: &'a [(&'a [u8], &'a [u8], [u8; C_TILDE_BYTES])],
    /// The signatures, in order.
    pub signatures: &'a [&'a [u8]],
}

/// Wall time of each prover step (spec Section 10 event order).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProveTimings {
    /// Witness derivation and the round-1 tables (`G_main`).
    pub tables: Duration,
    /// Akita setup from the work plan.
    pub setup: Duration,
    /// Event 1: the round-1 commitment.
    pub round1: Duration,
    /// Events 2 and 6: the Keccak side (commitments and proofs).
    pub keccak: Duration,
    /// Matrices, the round-2 tables (parities and carries) and their commitments (event 4).
    pub round2: Duration,
    /// Events 7 and 8: `tau` and the pre-stage claims.
    pub prestage: Duration,
    /// Stage 1, including the `w_2, w_3` weight tables.
    pub stage1: Duration,
    /// Stage 2, including the `w_1` weight table.
    pub stage2: Duration,
    /// Stage 3, including the `w_A` weight table.
    pub stage3: Duration,
    /// Event 12: the batched Akita opening.
    pub opening: Duration,
    /// End to end.
    pub total: Duration,
}

impl std::fmt::Display for ProveTimings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        write!(
            f,
            "tables {:.1} ms, setup {:.1} ms, round 1 {:.1} ms, keccak {:.1} ms, round 2 {:.1} ms, \
             prestage {:.1} ms, stage 1 {:.1} ms, stage 2 {:.1} ms, stage 3 {:.1} ms, \
             opening {:.1} ms, total {:.1} ms",
            ms(self.tables),
            ms(self.setup),
            ms(self.round1),
            ms(self.keccak),
            ms(self.round2),
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
    /// The stage points `(tau, r^(1), r^(2), r^(3))`.
    pub points: (Points, Vec<crate::OpeningField>),
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
    keccak: &impl KeccakSide,
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
    let manifest = CommitmentManifest::for_batch(n).map_err(PcsError::Plan)?;
    let main = MainTables::build(&manifest, inputs.instance, inputs.signatures)?;
    Ok((manifest, main))
}

/// Prove from given round-1 tables (events 0 to 12). `hooks` are for adversarial tests;
/// an honest caller passes `ProveHooks::default()`.
#[allow(clippy::too_many_lines)]
pub fn prove_with_tables(
    inputs: &ProverInputs<'_>,
    manifest: &CommitmentManifest,
    main: &MainTables,
    bridge: &impl BridgeDerivation,
    keccak: &impl KeccakSide,
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

    // Events 0 to 3: instance, round-1 commitment, Flock commitments, bridge seed.
    let digest = public_records_binding(inputs.records);
    let record = instance_record(manifest, &digest, flock);
    let mut transcript = Transcript::new(manifest, &record)?;
    let round1 = committer.commit_round1(main)?;
    let main_commitment = round1.serialized()?;
    transcript.absorb(commitment_site(GroupRole::Main), 1, &main_commitment)?;
    timings.round1 = clock.lap();
    let flock_commitments = keccak.commitments(segments);
    if flock_commitments.len() != segments {
        return Err(ProveError::Internal("one Flock commitment per segment".into()));
    }
    transcript.absorb_items(FLOCK_COMMIT_SITE, &flock_commitments)?;
    let bridge_seed: [u8; 32] = transcript
        .challenge_bytes(BRIDGE_SEED_SITE, 1)?
        .try_into()
        .expect("32 bytes");
    timings.keccak = clock.lap();

    // Events 4 to 6: round-2 commitments, Flock seeds, final vectors and Flock proofs.
    let matrices = bridge.matrices(&bridge_seed);
    let mut bridge_tables = BridgeGroupTables::build(manifest, main, &matrices)?;
    if let Some(tamper) = hooks.tamper_bridge {
        tamper(manifest, &mut bridge_tables);
    }
    let round2 = committer.commit_round2(&bridge_tables)?;
    let mut bridge_commitments: [Vec<u8>; 3] = Default::default();
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
    let flock_proofs = keccak.prove(&bridge_seed, &matrices, &flock_seeds, &final_bits);
    if flock_proofs.len() != segments {
        return Err(ProveError::Internal("one Flock proof per segment".into()));
    }
    transcript.absorb_items(FLOCK_PROOFS_SITE, &flock_proofs)?;
    timings.keccak += clock.lap();

    // Events 7 and 8: tau and the pre-stage claims.
    let mut points = Points {
        tau: transcript.challenge_extensions(TAU_SITE, manifest.d_main())?,
        ..Points::default()
    };
    let tables = SourceTables {
        manifest,
        main,
        bridge: &bridge_tables,
    };
    let prestage = PreStageClaims::evaluate(manifest, tables, &points.tau);
    transcript.absorb_extensions(PRESTAGE_SITE, &prestage.to_vec())?;
    let y_at_tau3 = agg_bridge::final_vector::evaluate(
        bridge_tables.public_y(),
        &suffix(&points.tau, manifest.group(GroupRole::Level3).logical_vars),
    );
    timings.prestage = clock.lap();

    // Stages 1 to 3.
    let trace = std::env::var_os("AGG_STAGE_TRACE").is_some();
    let weights_clock = Instant::now();
    let mut prover_tables = ProverTables::new(tables, &matrices);
    let parity_time = weights_clock.elapsed();
    prover_tables.set_tau(&matrices, &points.tau);
    if trace {
        eprintln!(
            "weights: parity1 tables {:.1} ms, w_2/w_3 {:.1} ms",
            parity_time.as_secs_f64() * 1e3,
            (weights_clock.elapsed() - parity_time).as_secs_f64() * 1e3
        );
    }
    let (stage1_proof, stage1) = if hooks.unchecked {
        stage1::prove_unchecked(manifest, &prover_tables, &points, &prestage, y_at_tau3, &mut transcript)?
    } else {
        stage1::prove(manifest, &prover_tables, &points, &prestage, y_at_tau3, &mut transcript)?
    };
    points.r1 = stage1.point.clone();
    timings.stage1 = clock.lap();
    let weights_clock = Instant::now();
    prover_tables.set_r1(&matrices, &points.r1);
    if trace {
        eprintln!("weights: w_1 {:.1} ms", weights_clock.elapsed().as_secs_f64() * 1e3);
    }
    let (stage2_proof, stage2) = if hooks.unchecked {
        stage2::prove_unchecked(manifest, &prover_tables, &points, &stage1, &mut transcript)?
    } else {
        stage2::prove(manifest, &prover_tables, &points, &stage1, &mut transcript)?
    };
    points.r2 = stage2.point.clone();
    timings.stage2 = clock.lap();
    let weights_clock = Instant::now();
    prover_tables.set_r2(&public, &points.r2);
    if trace {
        eprintln!("weights: w_A {:.1} ms", weights_clock.elapsed().as_secs_f64() * 1e3);
    }
    let (stage3_proof, stage3) = if hooks.unchecked {
        stage3::prove_unchecked(
            manifest,
            &prover_tables,
            &public,
            &points,
            &prestage,
            &stage1,
            &stage2,
            &mut transcript,
        )?
    } else {
        stage3::prove(
            manifest,
            &prover_tables,
            &public,
            &points,
            &prestage,
            &stage1,
            &stage2,
            &mut transcript,
        )?
    };
    drop(prover_tables);
    timings.stage3 = clock.lap();

    // Event 12: the opening.
    let session = transcript.pcs_session()?;
    transcript.ensure_complete()?;
    let claims = opening_claims(manifest, &stage3.point, &stage3.claims)?;
    let group_points = claims.clone().map(|c| c.point);
    let commitments: [GroupCommitment; 4] = {
        let [l1, l2, l3] = round2;
        [round1, l1, l2, l3]
    };
    let (opening, recomputed) = committer.open(commitments, tables, &group_points, &session)?;
    if recomputed.iter().zip(&claims).any(|(a, b)| a.values != b.values) {
        return Err(ProveError::Internal(
            "stage-3 terminal values differ from the table evaluations".into(),
        ));
    }
    timings.opening = clock.lap();
    let proof = AggregateProof {
        records: n,
        c_tilde: inputs.records.iter().map(|r| r.2).collect(),
        main_commitment,
        flock_commitments,
        bridge_commitments,
        final_vectors,
        flock_proofs,
        prestage,
        stage1: stage1_proof,
        stage2: stage2_proof,
        stage3: stage3_proof,
        opening,
    };
    timings.total = total.elapsed();
    Ok(ProverOutput {
        proof,
        bridge_seed,
        flock_seeds,
        session,
        points: (points, stage3.point),
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
