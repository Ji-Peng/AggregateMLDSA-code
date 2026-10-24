//! The aggregate verifier (spec Section 11): parse, precompute the light public data,
//! replay the transcript, run the Keccak side's checks, verify the three stages with the
//! public weights, stream the public matrices once, and verify the Akita opening. Every
//! step is timed ([`VerifyTimings`]); [`verify_timed`] returns the outcome with the timings.
//!
//! Steps and their spec items:
//!
//! | Step | Spec 11 | Work |
//! | --- | --- | --- |
//! | `parse` | 1 | wire parse (shape from `N`), manifest, PCS verifier from the plan's seed |
//! | `public_data` | 2 | `pkDecode`, `tr` per key, `mu`, `SampleInBall` per record ([`StreamedPublic`]); record binding |
//! | `transcript` | 3 | events 0 to 6: commitments parsed and absorbed, `sd`, `fs_sigma` |
//! | `matrices` | 3 | `M_1, M_2, M_3` from `sd` ([`BridgeDerivation`]; [`SeededMatrices`](crate::prover::SeededMatrices) is spec 9.1) |
//! | `keccak` | 3 | per segment [`KeccakVerifier::verify`] (Flock sub-proof, or mode B's public-bit checks) |
//! | `prestage` | 4 | `tau`, `Y(tau_3)`, pre-stage claims |
//! | `stage1`, `stage2` | 5, 6 | rounds and terminal checks with `w_2, w_3, w_1` and the leaf predicates |
//! | `stage3_rounds` | 6 | stage-3 rounds and terminal claims, ring claim deferred |
//! | `public_pass` | 6 | one pass over the records and keys: `T(r^(2))`, `w_A(r^(3))` ([`StreamedPublic::pass`]) |
//! | `stage3_terminal` | 7 | the batched stage-3 terminal check |
//! | `bits` | 7 | [`KeccakVerifier::check_bits`] (mode B: committed bits against the public bits) |
//! | `opening` | 7 | the batched Akita opening of the 67 claims |
//!
//! Before step 1 the verifier checks its own configuration ([`check_configuration`]) and
//! refuses one under which an accepted proof would not establish the relation: a
//! [`KeccakVerifier`] for another Flock profile than the requested one; mode A (profiles
//! `Fast`, `LaBinius`) without the seed-derived matrices of spec 9.1 (the bound of spec 9.4
//! needs `M_t` sampled after `G_main` and the Flock commitments are fixed); and profile 0
//! (no Keccak argument, `c~_i` not tied to the bits) outside unit tests and the
//! `test-hooks` feature (`docs/deviations.md` P4).

use std::time::{Duration, Instant};

use agg_pcs_config::commit::GroupClaim;
use agg_witness::bridge::BridgeMatrices;
use agg_witness::instance::InstanceError;
use agg_witness::padding::Geometry;
use agg_witness::params::{C_TILDE_BYTES, TR_BYTES, W1_BITS};
use akita_types::Commitment;

use crate::binding::{instance_record, public_records_binding, FlockProfile};
use crate::manifest::{Column, CommitmentManifest, GroupRole, GROUP_COUNT};
use crate::opening::{opening_claims, parse_commitment};
use crate::pcs::{verifier_for, PcsError};
use crate::prover::BridgeDerivation;
use crate::public_data::KeyCache;
use crate::public_stream::StreamedPublic;
use crate::stage::{suffix, Points, StageError};
use crate::transcript::{
    commitment_site, Transcript, TranscriptError, BRIDGE_SEED_SITE, FINAL_VECTORS_SITE,
    FLOCK_COMMIT_SITE, FLOCK_PROOFS_SITE, FLOCK_SEEDS_SITE, PRESTAGE_SITE, TAU_SITE,
};
use crate::wire::{unpack_final_vector, AggregateProof, WireError};
use crate::{stage1, stage2, stage3, BaseField, OpeningField as E};

/// What the Keccak side sees of one segment (spec Section 11 step 3).
pub struct SegmentContext<'a> {
    /// Segment index.
    pub sigma: usize,
    /// Batch geometry.
    pub geometry: Geometry,
    /// The segment's event-2 item (a Flock commitment; mode B: the public bits).
    pub commitment: &'a [u8],
    /// The challenger seed `fs_sigma` (event 5).
    pub seed: &'a [u8; 32],
    /// The bridge seed `sd` (event 3), which the bridge slot's circuit digest binds
    /// (docs/bridge.md Section 5, `docs/deviations.md` P11).
    pub bridge_seed: &'a [u8; 32],
    /// `mu_i` of the segment's live records.
    pub mu: &'a [[u8; TR_BYTES]],
    /// `c~_i` of the segment's live records.
    pub c_tilde: &'a [[u8; C_TILDE_BYTES]],
    /// `Y_sigma`, 4,096 bits, one per byte.
    pub final_vector: &'a [u8],
    /// The segment's event-6 proof item.
    pub proof: &'a [u8],
    /// The bridge matrices of `sd`.
    pub matrices: &'a BridgeMatrices,
}

/// What the Keccak side sees after stage 3 (mode B's direct check of the bits).
pub struct BitsContext<'a> {
    /// Batch geometry.
    pub geometry: Geometry,
    /// The `S` event-2 items.
    pub commitments: &'a [Vec<u8>],
    /// The stage-3 terminal point `r^(3)` (the opening point of `G_main`).
    pub point: &'a [E],
    /// The claimed `b_t(r^(3))`, `t < 6`, authenticated by the opening.
    pub claims: [E; W1_BITS],
}

/// The Keccak argument's verifier side (spec Section 8).
pub trait KeccakVerifier {
    /// The Flock profile this side verifies (bound in event 0). [`verify`] refuses to run
    /// with a different requested profile.
    fn profile(&self) -> FlockProfile;

    /// Verify segment `sigma`: the sub-proof against its commitment, challenger seed,
    /// public inputs and final vector.
    fn verify(&self, segment: &SegmentContext<'_>) -> bool;

    /// An optional check of the committed bits at `r^(3)`; the Flock side has none.
    fn check_bits(&self, _bits: &BitsContext<'_>) -> bool {
        true
    }
}

/// No Keccak argument (Flock profile 0): accepts empty commitments and proofs only. The
/// bits are not tied to the commitment hashes, so an accepted proof says nothing about
/// `c~_i`; a test stand-in for the PIOP alone (unit tests and the `test-hooks` feature).
#[cfg(any(test, feature = "test-hooks"))]
pub struct NoKeccakVerifier;

#[cfg(any(test, feature = "test-hooks"))]
impl KeccakVerifier for NoKeccakVerifier {
    fn profile(&self) -> FlockProfile {
        FlockProfile::None
    }

    fn verify(&self, segment: &SegmentContext<'_>) -> bool {
        segment.commitment.is_empty() && segment.proof.is_empty()
    }
}

/// A verifier configuration under which acceptance would not establish the relation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfigurationError {
    /// The [`KeccakVerifier`] verifies another Flock profile than the requested one.
    ProfileMismatch {
        /// The profile passed to [`verify`] (bound in event 0).
        requested: FlockProfile,
        /// [`KeccakVerifier::profile`].
        keccak: FlockProfile,
    },
    /// Mode A without the seed-derived matrices of spec 9.1
    /// ([`BridgeDerivation::seed_derived`] is false).
    UnseededMatrices(FlockProfile),
    /// Flock profile 0 (no Keccak argument) outside unit tests and the `test-hooks` feature.
    NoKeccakArgument,
}

impl std::fmt::Display for ConfigurationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ProfileMismatch { requested, keccak } => write!(
                f,
                "Flock profile {requested:?} requested, the Keccak verifier implements {keccak:?}"
            ),
            Self::UnseededMatrices(profile) => write!(
                f,
                "Flock profile {profile:?} (mode A) needs the seed-derived matrices of spec 9.1"
            ),
            Self::NoKeccakArgument => write!(
                f,
                "Flock profile 0 ties no c~ to the committed bits (test-hooks builds only)"
            ),
        }
    }
}

/// The configuration policy (module docs): `seed_derived` is
/// [`BridgeDerivation::seed_derived`], `hooks` is [`crate::test_hooks_enabled`].
pub fn configuration_policy(
    requested: FlockProfile,
    keccak: FlockProfile,
    seed_derived: bool,
    hooks: bool,
) -> Result<(), ConfigurationError> {
    if requested != keccak {
        return Err(ConfigurationError::ProfileMismatch { requested, keccak });
    }
    match requested {
        FlockProfile::Fast | FlockProfile::LaBinius if !seed_derived => {
            Err(ConfigurationError::UnseededMatrices(requested))
        }
        FlockProfile::None if !hooks => Err(ConfigurationError::NoKeccakArgument),
        _ => Ok(()),
    }
}

/// Check that `bridge` and `keccak` can verify proofs under `flock` (module docs).
pub fn check_configuration(
    bridge: &impl BridgeDerivation,
    keccak: &impl KeccakVerifier,
    flock: FlockProfile,
) -> Result<(), ConfigurationError> {
    configuration_policy(
        flock,
        keccak.profile(),
        bridge.seed_derived(),
        crate::test_hooks_enabled(),
    )
}

/// Where a proof was rejected.
#[derive(Debug)]
pub enum VerifyError {
    /// The verifier's own configuration is refused (nothing was checked).
    Configuration(ConfigurationError),
    /// Malformed proof bytes.
    Wire(WireError),
    /// The record list does not match the proof's `N`.
    RecordCount,
    /// A public key failed to decode.
    PublicKey(usize),
    /// A commitment failed to parse.
    Commitment(GroupRole),
    /// Transcript replay fault.
    Transcript(TranscriptError),
    /// The Keccak side of a segment was rejected.
    Keccak(usize),
    /// A stage was rejected.
    Stage(StageError),
    /// The committed bits differ from the Keccak side's bits at `r^(3)`.
    Bits,
    /// The Akita opening was rejected.
    Opening(PcsError),
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Configuration(e) => write!(f, "configuration: {e}"),
            Self::Wire(e) => write!(f, "wire: {e}"),
            Self::RecordCount => write!(f, "record count differs from the proof"),
            Self::PublicKey(i) => write!(f, "public key {i} is malformed"),
            Self::Commitment(g) => write!(f, "{g:?}: commitment bytes are malformed"),
            Self::Transcript(e) => write!(f, "transcript: {e}"),
            Self::Keccak(s) => write!(f, "keccak side of segment {s} rejected"),
            Self::Stage(e) => write!(f, "{e}"),
            Self::Bits => write!(f, "committed bits differ from the public bits"),
            Self::Opening(e) => write!(f, "opening: {e}"),
        }
    }
}

impl std::error::Error for VerifyError {}

impl From<WireError> for VerifyError {
    fn from(e: WireError) -> Self {
        Self::Wire(e)
    }
}
impl From<TranscriptError> for VerifyError {
    fn from(e: TranscriptError) -> Self {
        Self::Transcript(e)
    }
}
impl From<StageError> for VerifyError {
    fn from(e: StageError) -> Self {
        Self::Stage(e)
    }
}

/// What an accepted proof establishes.
#[derive(Clone, Debug)]
pub struct VerifiedAggregate {
    /// Live records.
    pub records: usize,
    /// The bridge seed.
    pub bridge_seed: [u8; 32],
    /// The PCS session.
    pub session: [u8; 32],
    /// Keys expanded in this run (the rest came from the cache).
    pub expanded_keys: usize,
}

/// Wall time of each verifier step (module table). A rejected run stops at the failing
/// step; later steps stay zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VerifyTimings {
    /// Wire parse, manifest, PCS verifier.
    pub parse: Duration,
    /// `pkDecode`, `tr`, `mu`, `SampleInBall`; record binding.
    pub public_data: Duration,
    /// Events 0 to 6 without the matrices and the Keccak side.
    pub transcript: Duration,
    /// The bridge matrices from `sd`.
    pub matrices: Duration,
    /// The Keccak side per segment.
    pub keccak: Duration,
    /// `tau`, `Y(tau_3)`, pre-stage claims.
    pub prestage: Duration,
    /// Stage 1.
    pub stage1: Duration,
    /// Stage 2.
    pub stage2: Duration,
    /// Stage-3 rounds and terminal claims.
    pub stage3_rounds: Duration,
    /// The public pass: `T(r^(2))` part.
    pub public_t: Duration,
    /// The public pass: `w_A(r^(3))` part (ExpandA and inverse NTTs when uncached).
    pub public_a: Duration,
    /// The stage-3 terminal check.
    pub stage3_terminal: Duration,
    /// The Keccak side's check of the bits at `r^(3)`.
    pub bits: Duration,
    /// The Akita opening.
    pub opening: Duration,
    /// End to end.
    pub total: Duration,
}

impl VerifyTimings {
    /// `(name, duration)` per step, in order.
    #[must_use]
    pub fn steps(&self) -> [(&'static str, Duration); 14] {
        [
            ("parse", self.parse),
            ("public_data", self.public_data),
            ("transcript", self.transcript),
            ("matrices", self.matrices),
            ("keccak", self.keccak),
            ("prestage", self.prestage),
            ("stage1", self.stage1),
            ("stage2", self.stage2),
            ("stage3_rounds", self.stage3_rounds),
            ("public_t", self.public_t),
            ("public_a", self.public_a),
            ("stage3_terminal", self.stage3_terminal),
            ("bits", self.bits),
            ("opening", self.opening),
        ]
    }
}

impl std::fmt::Display for VerifyTimings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (name, duration) in self.steps() {
            write!(f, "{name} {:.2} ms, ", duration.as_secs_f64() * 1e3)?;
        }
        write!(f, "total {:.2} ms", self.total.as_secs_f64() * 1e3)
    }
}

/// The verifier's verdict with the step timings.
#[derive(Debug)]
pub struct VerifyReport {
    /// Accept (`Ok`) or the first failing check.
    pub outcome: Result<VerifiedAggregate, VerifyError>,
    /// Wall time per step.
    pub timings: VerifyTimings,
}

impl VerifyReport {
    /// True when the proof was accepted.
    #[must_use]
    pub fn accepted(&self) -> bool {
        self.outcome.is_ok()
    }
}

/// Verify `proof_bytes` for the records `(pk_i, M_i)`; the `c~_i` come from the proof. An
/// unsound configuration ([`check_configuration`]) is refused before the proof is read.
pub fn verify(
    records: &[(&[u8], &[u8])],
    proof_bytes: &[u8],
    bridge: &impl BridgeDerivation,
    keccak: &impl KeccakVerifier,
    flock: FlockProfile,
    cache: Option<&mut KeyCache>,
) -> Result<VerifiedAggregate, VerifyError> {
    verify_timed(records, proof_bytes, bridge, keccak, flock, cache).outcome
}

/// [`verify`] with the wall time of every step.
pub fn verify_timed(
    records: &[(&[u8], &[u8])],
    proof_bytes: &[u8],
    bridge: &impl BridgeDerivation,
    keccak: &impl KeccakVerifier,
    flock: FlockProfile,
    cache: Option<&mut KeyCache>,
) -> VerifyReport {
    let start = Instant::now();
    let mut timings = VerifyTimings::default();
    let outcome = run(
        records,
        proof_bytes,
        bridge,
        keccak,
        flock,
        cache,
        &mut timings,
    );
    timings.total = start.elapsed();
    VerifyReport { outcome, timings }
}

/// A running clock for step timings.
struct Lap(Instant);

impl Lap {
    fn lap(&mut self) -> Duration {
        let now = Instant::now();
        let elapsed = now - self.0;
        self.0 = now;
        elapsed
    }
}

#[allow(clippy::too_many_lines)]
fn run(
    records: &[(&[u8], &[u8])],
    proof_bytes: &[u8],
    bridge: &impl BridgeDerivation,
    keccak: &impl KeccakVerifier,
    flock: FlockProfile,
    cache: Option<&mut KeyCache>,
    timings: &mut VerifyTimings,
) -> Result<VerifiedAggregate, VerifyError> {
    check_configuration(bridge, keccak, flock).map_err(VerifyError::Configuration)?;
    let mut clock = Lap(Instant::now());

    // Step 1: parse.
    let proof = AggregateProof::from_bytes(proof_bytes)?;
    let n = proof.records;
    if records.len() != n || proof.c_tilde.len() != n {
        return Err(VerifyError::RecordCount);
    }
    let manifest =
        CommitmentManifest::for_batch(n).map_err(|e| VerifyError::Opening(PcsError::Plan(e)))?;
    let segments = manifest.segments();
    let pcs = verifier_for(&manifest).map_err(VerifyError::Opening)?;
    let plan = pcs.plan();
    timings.parse = clock.lap();

    // Step 2: light public data and the record binding.
    let inputs: Vec<(&[u8], &[u8], [u8; C_TILDE_BYTES])> = records
        .iter()
        .zip(&proof.c_tilde)
        .map(|(&(pk, msg), &c)| (pk, msg, c))
        .collect();
    let public = StreamedPublic::prepare(&inputs, cache.as_deref()).map_err(|e| match e {
        InstanceError::PublicKey(i) => VerifyError::PublicKey(i),
        InstanceError::Geometry(_) => VerifyError::RecordCount,
    })?;
    let geometry = public.geometry();
    let digest = public_records_binding(&inputs);
    timings.public_data = clock.lap();

    // Step 3: events 0 to 6.
    let record = instance_record(&manifest, &digest, flock);
    let mut transcript = Transcript::new(&manifest, &record)?;
    let mut commitments: Vec<Commitment<BaseField>> = Vec::with_capacity(GROUP_COUNT);
    let main_commitment = parse_commitment(plan, GroupRole::Main, &proof.main_commitment)
        .map_err(|_| VerifyError::Commitment(GroupRole::Main))?;
    transcript.absorb(commitment_site(GroupRole::Main), 1, &proof.main_commitment)?;
    commitments.push(main_commitment);
    transcript.absorb_items(FLOCK_COMMIT_SITE, &proof.flock_commitments)?;
    let bridge_seed: [u8; 32] = transcript
        .challenge_bytes(BRIDGE_SEED_SITE, 1)?
        .try_into()
        .expect("32 bytes");
    timings.transcript = clock.lap();
    let matrices = bridge.matrices(&bridge_seed);
    timings.matrices = clock.lap();
    for (role, bytes) in [GroupRole::Level1, GroupRole::Level2, GroupRole::Level3]
        .into_iter()
        .zip(&proof.bridge_commitments)
    {
        let commitment =
            parse_commitment(plan, role, bytes).map_err(|_| VerifyError::Commitment(role))?;
        transcript.absorb(commitment_site(role), 1, bytes)?;
        commitments.push(commitment);
    }
    let flock_seeds: Vec<[u8; 32]> = transcript
        .challenge_bytes(FLOCK_SEEDS_SITE, segments)?
        .as_chunks::<32>()
        .0
        .to_vec();
    transcript.absorb(FINAL_VECTORS_SITE, segments, &proof.final_vectors.concat())?;
    transcript.absorb_items(FLOCK_PROOFS_SITE, &proof.flock_proofs)?;
    timings.transcript += clock.lap();
    let mu: Vec<[u8; TR_BYTES]> = public.records().iter().map(|r| r.mu).collect();
    // The public table Y on D_3 (Y_sigma[rho_3] at rho_3 + 2^12 sigma), docs/bridge.md 6.
    let mut y_table: Vec<u8> = Vec::with_capacity(4096 * segments);
    #[allow(clippy::needless_range_loop)]
    for sigma in 0..segments {
        let bits = unpack_final_vector(&proof.final_vectors[sigma]);
        // A trailing segment may hold padded records only (`2^13 sigma >= N`).
        let first = (sigma << 13).min(n);
        let live = first..first + geometry.live_in_segment(sigma);
        let segment = SegmentContext {
            sigma,
            geometry,
            commitment: &proof.flock_commitments[sigma],
            seed: &flock_seeds[sigma],
            bridge_seed: &bridge_seed,
            mu: &mu[live.clone()],
            c_tilde: &proof.c_tilde[live],
            final_vector: &bits,
            proof: &proof.flock_proofs[sigma],
            matrices: &matrices,
        };
        if !keccak.verify(&segment) {
            return Err(VerifyError::Keccak(sigma));
        }
        y_table.extend_from_slice(&bits);
    }
    timings.keccak = clock.lap();

    // Step 4: tau, the pre-stage claims, Y(tau_3).
    let mut points = Points {
        tau: transcript.challenge_extensions(TAU_SITE, manifest.d_main())?,
        ..Points::default()
    };
    transcript.absorb_extensions(PRESTAGE_SITE, &proof.prestage.to_vec())?;
    let y_at_tau3 = agg_bridge::final_vector::evaluate(
        &y_table,
        &suffix(&points.tau, manifest.group(GroupRole::Level3).logical_vars),
    );
    timings.prestage = clock.lap();

    // Steps 5 to 7: the stages; the public pass runs inside stage 3 once r^(3) is known.
    let stage1 = stage1::verify(
        &manifest,
        &matrices,
        &points,
        &proof.prestage,
        y_at_tau3,
        &proof.stage1,
        &mut transcript,
    )?;
    points.r1 = stage1.point.clone();
    timings.stage1 = clock.lap();
    let stage2 = stage2::verify(
        &manifest,
        &matrices,
        &points,
        &stage1,
        &proof.stage2,
        &mut transcript,
    )?;
    points.r2 = stage2.point.clone();
    timings.stage2 = clock.lap();
    let stage3_start = Instant::now();
    let mut pass_window = None;
    let mut pass_result = None;
    let stage3 = stage3::verify_streamed(
        &manifest,
        &points,
        &proof.prestage,
        &stage1,
        &stage2,
        &proof.stage3,
        &mut transcript,
        |r2, r3| {
            let started = Instant::now();
            let pass = public.pass(r2, r3, cache);
            pass_window = Some((started, Instant::now()));
            pass_result = Some(pass);
            (pass.t_at_r2, pass.ring_weight)
        },
    );
    let stage3_end = Instant::now();
    clock.0 = stage3_end;
    match (pass_window, pass_result) {
        (Some((started, finished)), Some(pass)) => {
            timings.stage3_rounds = started - stage3_start;
            timings.public_t = pass.t_time;
            timings.public_a = pass.a_time;
            timings.stage3_terminal = stage3_end - finished;
        }
        _ => timings.stage3_rounds = stage3_end - stage3_start,
    }
    let stage3 = stage3?;
    let expanded_keys = pass_result.map_or(0, |pass| pass.expanded);

    // The Keccak side's check of the committed bits at r^(3).
    let bits = BitsContext {
        geometry,
        commitments: &proof.flock_commitments,
        point: &stage3.point,
        claims: std::array::from_fn(|t| stage3.claims.column(&manifest, Column::W1Bit(t as u8))),
    };
    if !keccak.check_bits(&bits) {
        return Err(VerifyError::Bits);
    }
    timings.bits = clock.lap();

    // Event 12: the opening.
    let session = transcript.pcs_session()?;
    transcript.ensure_complete()?;
    let claims: [GroupClaim; GROUP_COUNT] =
        opening_claims(&manifest, &stage3.point, &stage3.claims).map_err(VerifyError::Opening)?;
    let commitments: [Commitment<BaseField>; GROUP_COUNT] =
        commitments.try_into().expect("four commitments");
    pcs.verify(&commitments, &claims, &proof.opening, &session)
        .map_err(|e| VerifyError::Opening(PcsError::Commit(e)))?;
    timings.opening = clock.lap();
    Ok(VerifiedAggregate {
        records: n,
        bridge_seed,
        session,
        expanded_keys,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prover::{FixedMatrices, SeededMatrices};
    use agg_witness::bridge::BridgeMatrices;

    const PROFILES: [FlockProfile; 4] = [
        FlockProfile::None,
        FlockProfile::Fast,
        FlockProfile::LaBinius,
        FlockProfile::PublicBits,
    ];

    #[test]
    fn configuration_policy_refuses_exactly_the_unsound_configurations() {
        for requested in PROFILES {
            for keccak in PROFILES {
                for seed_derived in [false, true] {
                    for hooks in [false, true] {
                        let outcome = configuration_policy(requested, keccak, seed_derived, hooks);
                        let expected = if requested != keccak {
                            Err(ConfigurationError::ProfileMismatch { requested, keccak })
                        } else if requested.is_mode_a() && !seed_derived {
                            Err(ConfigurationError::UnseededMatrices(requested))
                        } else if requested == FlockProfile::None && !hooks {
                            Err(ConfigurationError::NoKeccakArgument)
                        } else {
                            Ok(())
                        };
                        assert_eq!(
                            outcome, expected,
                            "{requested:?} {keccak:?} {seed_derived} {hooks}"
                        );
                    }
                }
            }
        }
        // Mode A runs only with seeded matrices, whatever the build; mode B with either.
        let policy = |profile, seed_derived, hooks| {
            configuration_policy(profile, profile, seed_derived, hooks)
        };
        assert!(policy(FlockProfile::Fast, false, true).is_err());
        assert!(policy(FlockProfile::LaBinius, true, false).is_ok());
        assert!(policy(FlockProfile::PublicBits, false, false).is_ok());
    }

    /// A Keccak side that accepts everything under a chosen profile (the shape of an
    /// unsound mode-A configuration).
    struct AcceptAll(FlockProfile);

    impl KeccakVerifier for AcceptAll {
        fn profile(&self) -> FlockProfile {
            self.0
        }

        fn verify(&self, _segment: &SegmentContext<'_>) -> bool {
            true
        }
    }

    fn refused(
        bridge: &impl BridgeDerivation,
        keccak: &impl KeccakVerifier,
        flock: FlockProfile,
    ) -> Option<ConfigurationError> {
        // Garbage proof bytes: a configuration that passes fails at the wire parse.
        match verify(&[], &[0xff; 16], bridge, keccak, flock, None) {
            Err(VerifyError::Configuration(e)) => Some(e),
            Err(VerifyError::Wire(_)) => None,
            other => panic!("unexpected outcome {other:?}"),
        }
    }

    #[test]
    fn verify_refuses_fixed_matrices_in_mode_a_and_mismatched_profiles_before_parsing() {
        let fixed = FixedMatrices(BridgeMatrices::random_for_tests(9));
        for profile in [FlockProfile::Fast, FlockProfile::LaBinius] {
            assert_eq!(
                refused(&fixed, &AcceptAll(profile), profile),
                Some(ConfigurationError::UnseededMatrices(profile))
            );
            assert_eq!(refused(&SeededMatrices, &AcceptAll(profile), profile), None);
            // The no-Keccak stand-in cannot serve a mode-A profile.
            assert_eq!(
                refused(&SeededMatrices, &NoKeccakVerifier, profile),
                Some(ConfigurationError::ProfileMismatch {
                    requested: profile,
                    keccak: FlockProfile::None
                })
            );
        }
        // Profile 0 is accepted in unit tests (test hooks on) and mode B with either bridge.
        assert_eq!(refused(&fixed, &NoKeccakVerifier, FlockProfile::None), None);
        let public_bits = AcceptAll(FlockProfile::PublicBits);
        assert_eq!(refused(&fixed, &public_bits, FlockProfile::PublicBits), None);
        assert_eq!(refused(&SeededMatrices, &public_bits, FlockProfile::PublicBits), None);
        let fast = AcceptAll(FlockProfile::Fast);
        let report = verify_timed(&[], &[], &fixed, &fast, FlockProfile::Fast, None);
        assert!(matches!(report.outcome, Err(VerifyError::Configuration(_))));
        assert_eq!(report.timings.parse, Duration::ZERO, "refused before step 1");
    }
}
