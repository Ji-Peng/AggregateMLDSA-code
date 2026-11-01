//! The aggregate verifier: parse, precompute the light public data, replay the transcript,
//! run the Keccak side's checks, verify the three stages with the public weights, stream the
//! public matrices once, and verify the Akita opening. Every step is timed
//! ([`VerifyTimings`]); [`verify_timed`] returns the outcome with the timings.
//!
//! Steps:
//!
//! | Step | Work |
//! | --- | --- |
//! | `parse` | wire parse (shape from `N`), manifest, PCS verifier from the plan's seed |
//! | `public_data` | `pkDecode`, `tr` per key, `mu`, `SampleInBall` per record ([`StreamedPublic`]); record binding |
//! | `transcript` | the six commitments parsed and absorbed, `sd`, `fs_sigma`, `Y_sigma` and the share items |
//! | `matrices` | `M_1, M_2, M_3` from `sd` ([`BridgeDerivation`]; [`SeededMatrices`](crate::prover::SeededMatrices) in mode A) |
//! | `keccak` | per segment [`KeccakVerifier::verify`] (Flock sub-proof, or mode B's public-bit checks); the transcript never absorbs the proof items |
//! | `lookup` | the lookup into `T_16` (two challenges, the fraction-tree sessions) and the multiplicity-digit claims |
//! | `prestage` | `tau`, `Y(tau_3)`, pre-stage claims |
//! | `stage1`, `stage2` | rounds and terminal checks with `w_2, w_3, w_1` and the leaf predicates |
//! | `stage3_rounds` | stage-3 rounds and terminal claims, ring claim deferred |
//! | `public_pass` | one pass over the records and keys: `T(r^(2))`, `w_A(r^(3))` ([`StreamedPublic::pass`]) |
//! | `stage3_terminal` | the batched stage-3 terminal check |
//! | `bits` | [`KeccakVerifier::check_bits`] (mode B: committed bits against the public bits) |
//! | `opening` | the batched Akita opening of the 53 claims (H-u) |
//!
//! Before parsing the verifier checks its own configuration ([`check_configuration`]) and
//! refuses one under which an accepted proof would not establish the relation: a
//! [`KeccakVerifier`] for another Flock profile than the requested one; mode A (profiles
//! `Fast`, `Slim`, `Slim8`) without the seed-derived matrices (the bridge's soundness bound
//! needs `M_t` sampled after `G_main` and the Flock commitments are fixed); and profile 0
//! (no Keccak argument, `c~_i` not tied to the bits) outside unit tests and the
//! `test-hooks` feature (`docs/deviations.md`).

use std::time::{Duration, Instant};

use agg_pcs_config::commit::GroupClaim;
use agg_witness::bridge::BridgeMatrices;
use agg_witness::instance::InstanceError;
use agg_witness::padding::Geometry;
use agg_witness::params::{ParamSet, Set, Set65, Set87, C_TILDE_BYTES, TR_BYTES, W1_BITS};
use akita_types::Commitment;

use crate::binding::{instance_record, public_records_binding, FlockProfile};
use crate::lookup::{verify_in, LookupError};
use crate::manifest::{Column, CommitmentManifest, GroupRole, Relation, GROUP_COUNT};
use crate::opening::{opening_claims, parse_commitment};
use crate::pcs::{verifier_for, PcsError};
use crate::pow::{KeccakPlan, PowPlan};
use crate::prover::{BridgeDerivation, PiopProof};
use crate::lane_public::{LaneKeyCache, LanePublicData};
use crate::public_data::KeyCache;
use crate::public_stream::StreamedPublic;
use crate::stage::{suffix, Points, StageError};
use crate::stage3::{LookupInputs, Stage3Output};
use crate::transcript::{
    commitment_site, flock_record_digest, Transcript, TranscriptError, BRIDGE_SEED_SITE,
    FINAL_VECTORS_SITE, FLOCK_COMMIT_SITE, FLOCK_SEEDS_SITE,
    FLOCK_SHARES_SITE, MULTIPLICITY_SITE, PRESTAGE_SITE, TAU_SITE,
};
use crate::wire::{unpack_final_vector, AggregateProof, WireError};
use crate::{stage1, stage2, stage3, BaseField, OpeningField as E};

/// What the Keccak side sees of one segment.
pub struct SegmentContext<'a> {
    /// Segment index.
    pub sigma: usize,
    /// Batch geometry.
    pub geometry: Geometry,
    /// The segment's commitment-phase item (a Flock commitment; mode B: the public bits).
    pub commitment: &'a [u8],
    /// The challenger seed `fs_sigma` (the Flock seed step).
    pub seed: &'a [u8; 32],
    /// The bridge seed `sd`, whose matrices define the parity chain's weight
    /// (`docs/bridge.md`, `docs/parity-chain.md`).
    pub bridge_seed: &'a [u8; 32],
    /// `mu_i` of the segment's live records.
    pub mu: &'a [[u8; TR_BYTES]],
    /// `c~_i` of the segment's live records, concatenated (`lambda / 4` bytes each: 32,
    /// 48 or 64 for ML-DSA-44, 65 and 87).
    pub c_tilde: &'a [u8],
    /// `Y_sigma`, 4,096 bits, one per byte.
    pub final_vector: &'a [u8],
    /// The segment's share item.
    pub shares: &'a [u8],
    /// The digest of the final-vector record (`crate::transcript::flock_record_digest`), which
    /// the sub-proofs' challengers bind.
    pub record: &'a [u8; 32],
    /// The segment's proof item (never absorbed by the transcript).
    pub proof: &'a [u8],
    /// The bridge matrices of `sd`.
    pub matrices: &'a BridgeMatrices,
}

/// What the Keccak side sees after stage 3 (mode B's direct check of the bits).
pub struct BitsContext<'a> {
    /// Batch geometry.
    pub geometry: Geometry,
    /// The `S` commitment-phase items.
    pub commitments: &'a [Vec<u8>],
    /// The stage-3 terminal point `r^(3)` (the opening point of `G_main`).
    pub point: &'a [E],
    /// The claimed `b_t(r^(3))`, `t < 6`, authenticated by the opening.
    pub claims: [E; W1_BITS],
}

/// The Keccak argument's verifier side.
pub trait KeccakVerifier {
    /// The Flock profile this side verifies (bound in the instance record). [`verify`] refuses to run
    /// with a different requested profile.
    fn profile(&self) -> FlockProfile;

    /// The Keccak side's part of the proof-of-work plan (`docs/soundness.md`)
    /// from the batch geometry and the `S` commitment-phase items (the tied parts' geometry), before
    /// the instance record; `None` refuses the items. The default (no tied parts) fits a side without
    /// Flock sub-proofs.
    fn plan(&self, _geometry: &Geometry, _commitments: &[Vec<u8>]) -> Option<KeccakPlan> {
        Some(KeccakPlan::default())
    }

    /// Verify segment `sigma`: the sub-proof (which the transcript never absorbs) against its
    /// commitment, challenger seed, public inputs, final vector, share item and the final-vector
    /// record digest. The verifier calls it right after replaying the final-vector step: the outcome does
    /// not depend on the PIOP, and an early check gives a precise rejection.
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
        segment.commitment.is_empty() && segment.shares.is_empty() && segment.proof.is_empty()
    }
}

/// A verifier configuration under which acceptance would not establish the relation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfigurationError {
    /// The [`KeccakVerifier`] verifies another Flock profile than the requested one.
    ProfileMismatch {
        /// The profile passed to [`verify`] (bound in the instance record).
        requested: FlockProfile,
        /// [`KeccakVerifier::profile`].
        keccak: FlockProfile,
    },
    /// Mode A without the seed-derived matrices
    /// ([`BridgeDerivation::seed_derived`] is false).
    UnseededMatrices(FlockProfile),
    /// Flock profile 0 (no Keccak argument) outside unit tests and the `test-hooks` feature.
    NoKeccakArgument,
    /// Mode B (public bits) for a parameter set other than ML-DSA-44: its direct bit check
    /// reads ML-DSA-44's six `w_1` bits per coefficient.
    PublicBitsSet(Set),
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
            Self::PublicBitsSet(set) => write!(
                f,
                "mode B (public bits) is defined for ML-DSA-44 only, not ML-DSA-{}",
                set.name()
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
        _ if requested.is_mode_a() && !seed_derived => {
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
    /// The aggregate names a mode-A Flock profile that a default verifier does not accept.
    ProfileNotAccepted(FlockProfile),
    /// Malformed proof bytes.
    Wire(WireError),
    /// The proof's relation byte differs from the relation the caller expects (checked
    /// before the instance record: a proof of a weaker relation is no proof of
    /// the stronger one, and the relation is never trusted from the proof).
    Relation {
        /// The caller's relation.
        expected: Relation,
        /// The proof header's relation.
        found: Relation,
    },
    /// The proof's parameter set differs from the one the verifier expects (checked before
    /// the instance record, like the relation).
    ParamSet {
        /// The caller's parameter set.
        expected: Set,
        /// The proof header's parameter set.
        found: Set,
    },
    /// The record list does not match the proof's `N`.
    RecordCount,
    /// A public key failed to decode.
    PublicKey(usize),
    /// A commitment failed to parse.
    Commitment(GroupRole),
    /// Transcript replay fault.
    Transcript(TranscriptError),
    /// The Keccak side refused the commitment-phase items while deriving the proof-of-work plan.
    Plan,
    /// The Keccak side of a segment was rejected.
    Keccak(usize),
    /// The lookup into `T_16` was rejected (root identity, a session, a shape, or the
    /// multiplicity digits disagree with the table claim).
    Lookup(LookupError),
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
            Self::ProfileNotAccepted(profile) => write!(
                f,
                "Flock profile {profile:?} is not accepted by default; pass --flock-profile {}",
                match profile {
                    FlockProfile::Slim8 => "slim8",
                    FlockProfile::Slim => "slim",
                    _ => "fast",
                }
            ),
            Self::Wire(e) => write!(f, "wire: {e}"),
            Self::Relation { expected, found } => {
                write!(f, "the proof claims relation {found}, the verifier expects {expected}")
            }
            Self::ParamSet { expected, found } => {
                write!(f, "the proof is for ML-DSA-{found}, the verifier expects ML-DSA-{expected}")
            }
            Self::RecordCount => write!(f, "record count differs from the proof"),
            Self::PublicKey(i) => write!(f, "public key {i} is malformed"),
            Self::Commitment(g) => write!(f, "{g:?}: commitment bytes are malformed"),
            Self::Transcript(e) => write!(f, "transcript: {e}"),
            Self::Plan => write!(f, "keccak items refused while deriving the proof-of-work plan"),
            Self::Keccak(s) => write!(f, "keccak side of segment {s} rejected"),
            Self::Lookup(e) => write!(f, "lookup: {e}"),
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
impl From<LookupError> for VerifyError {
    fn from(e: LookupError) -> Self {
        Self::Lookup(e)
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
    /// The transcript through the final-vector step without the matrices.
    pub transcript: Duration,
    /// The bridge matrices from `sd`.
    pub matrices: Duration,
    /// The Keccak side per segment (checked after the final-vector step; its proof items are
    /// never absorbed).
    pub keccak: Duration,
    /// The lookup and the multiplicity-digit claims.
    pub lookup: Duration,
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
    pub fn steps(&self) -> [(&'static str, Duration); 15] {
        [
            ("parse", self.parse),
            ("public_data", self.public_data),
            ("transcript", self.transcript),
            ("matrices", self.matrices),
            ("keccak", self.keccak),
            ("lookup", self.lookup),
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

/// Verify `proof_bytes` for the records `(pk_i, M_i)` under `relation`, the relation the
/// caller expects (never read from the proof: a proof whose header names another relation
/// is rejected before the instance record); the `c~_i` come from the proof. An unsound configuration
/// ([`check_configuration`]) is refused before the proof is read.
pub fn verify(
    records: &[(&[u8], &[u8])],
    proof_bytes: &[u8],
    bridge: &impl BridgeDerivation,
    keccak: &impl KeccakVerifier,
    flock: FlockProfile,
    relation: Relation,
    cache: Option<&mut KeyCache>,
) -> Result<VerifiedAggregate, VerifyError> {
    verify_timed(records, proof_bytes, bridge, keccak, flock, relation, cache).outcome
}

/// [`verify`] with the wall time of every step.
pub fn verify_timed(
    records: &[(&[u8], &[u8])],
    proof_bytes: &[u8],
    bridge: &impl BridgeDerivation,
    keccak: &impl KeccakVerifier,
    flock: FlockProfile,
    relation: Relation,
    cache: Option<&mut KeyCache>,
) -> VerifyReport {
    let start = Instant::now();
    let mut timings = VerifyTimings::default();
    let outcome = run(
        Set::S44,
        records,
        proof_bytes,
        bridge,
        keccak,
        flock,
        relation,
        Keys::Rows(cache),
        &mut timings,
    );
    timings.total = start.elapsed();
    VerifyReport { outcome, timings }
}

/// [`verify_timed`] for an aggregate of a parameter set with polynomial lanes (ML-DSA-65 or
/// ML-DSA-87, relation H-u only): `set` is the caller's, never read from the proof (a proof
/// whose header names another set is rejected before the instance record). With `cache`
/// (of `set`), keys found there are not expanded again and new expansions are added.
///
/// # Panics
/// If `set` is ML-DSA-44 (use [`verify_timed`]) or `cache` holds another set.
#[allow(clippy::too_many_arguments)]
pub fn verify_lanes_timed(
    set: Set,
    records: &[(&[u8], &[u8])],
    proof_bytes: &[u8],
    bridge: &impl BridgeDerivation,
    keccak: &impl KeccakVerifier,
    flock: FlockProfile,
    relation: Relation,
    cache: Option<&mut LaneKeyCache>,
) -> VerifyReport {
    assert_ne!(set, Set::S44, "ML-DSA-44 has no polynomial lanes");
    if let Some(cache) = cache.as_ref() {
        assert_eq!(cache.set(), set, "a key cache of the verified parameter set");
    }
    let start = Instant::now();
    let mut timings = VerifyTimings::default();
    let outcome = run(
        set,
        records,
        proof_bytes,
        bridge,
        keccak,
        flock,
        relation,
        Keys::Lanes(cache),
        &mut timings,
    );
    timings.total = start.elapsed();
    VerifyReport { outcome, timings }
}

/// The verifier's key cache for the expected parameter set.
enum Keys<'c> {
    /// ML-DSA-44.
    Rows(Option<&'c mut KeyCache>),
    /// A set with polynomial lanes.
    Lanes(Option<&'c mut LaneKeyCache>),
}

/// The public data of the verified batch.
enum Public<'c> {
    /// ML-DSA-44: the streamed rows and the key cache the public pass fills.
    Rows(StreamedPublic, Option<&'c mut KeyCache>),
    /// A set with polynomial lanes (keys already expanded or taken from the cache).
    Lanes(LanePublicData),
}

/// The lane public data of `(pk_i, M_i, c~_i)` and the record digest.
fn prepare_lanes<P: ParamSet>(
    records: &[(&[u8], &[u8])],
    c_tilde: &[Vec<u8>],
    cache: Option<&mut LaneKeyCache>,
) -> Result<(LanePublicData, [u8; 32]), VerifyError>
where
    P::CTilde: Sync,
{
    let inputs: Vec<(&[u8], &[u8], P::CTilde)> = records
        .iter()
        .zip(c_tilde)
        .map(|(&(pk, msg), c)| (pk, msg, P::c_tilde_from_slice(c)))
        .collect();
    let public = LanePublicData::prepare::<P>(&inputs, cache).map_err(|e| match e {
        InstanceError::PublicKey(i) => VerifyError::PublicKey(i),
        InstanceError::Geometry(_) => VerifyError::RecordCount,
    })?;
    Ok((public, public_records_binding(&inputs)))
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

#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn run(
    set: Set,
    records: &[(&[u8], &[u8])],
    proof_bytes: &[u8],
    bridge: &impl BridgeDerivation,
    keccak: &impl KeccakVerifier,
    flock: FlockProfile,
    relation: Relation,
    keys: Keys<'_>,
    timings: &mut VerifyTimings,
) -> Result<VerifiedAggregate, VerifyError> {
    check_configuration(bridge, keccak, flock).map_err(VerifyError::Configuration)?;
    if set != Set::S44 && flock == FlockProfile::PublicBits {
        return Err(VerifyError::Configuration(ConfigurationError::PublicBitsSet(set)));
    }
    let mut clock = Lap(Instant::now());

    // Parse: the header's relation must be the caller's, and the manifest (the instance record, the
    // catalog row, the plan) comes from the caller's relation.
    let proof = AggregateProof::from_bytes(proof_bytes)?;
    // The parameter set is the caller's, like the relation.
    if proof.set != set {
        return Err(VerifyError::ParamSet {
            expected: set,
            found: proof.set,
        });
    }
    if proof.relation != relation {
        return Err(VerifyError::Relation {
            expected: relation,
            found: proof.relation,
        });
    }
    let n = proof.records;
    if records.len() != n || proof.c_tilde.len() != n {
        return Err(VerifyError::RecordCount);
    }
    if set != Set::S44 && n < crate::prover::LANE_MIN_RECORDS {
        return Err(VerifyError::RecordCount);
    }
    let manifest = CommitmentManifest::for_set(set, n, relation)
        .map_err(|e| VerifyError::Opening(PcsError::Plan(e)))?;
    let segments = manifest.segments();
    let pcs = verifier_for(&manifest).map_err(VerifyError::Opening)?;
    let plan = pcs.plan();
    timings.parse = clock.lap();

    // Light public data and the record binding.
    let c_tilde: Vec<u8> = proof.c_tilde.concat();
    let (public, geometry, digest, mu) = match keys {
        Keys::Rows(cache) => {
            let c_tilde: Vec<[u8; C_TILDE_BYTES]> = proof
                .c_tilde
                .iter()
                .map(|c| c.as_slice().try_into().expect("the parser reads 32 bytes per ML-DSA-44 record"))
                .collect();
            let inputs: Vec<(&[u8], &[u8], [u8; C_TILDE_BYTES])> = records
                .iter()
                .zip(&c_tilde)
                .map(|(&(pk, msg), &c)| (pk, msg, c))
                .collect();
            let public = StreamedPublic::prepare(&inputs, cache.as_deref()).map_err(|e| match e {
                InstanceError::PublicKey(i) => VerifyError::PublicKey(i),
                InstanceError::Geometry(_) => VerifyError::RecordCount,
            })?;
            let geometry = public.geometry();
            let digest = public_records_binding(&inputs);
            let mu: Vec<[u8; TR_BYTES]> = public.records().iter().map(|r| r.mu).collect();
            (Public::Rows(public, cache), geometry, digest, mu)
        }
        Keys::Lanes(cache) => {
            let (public, digest) = match set {
                Set::S65 => prepare_lanes::<Set65>(records, &proof.c_tilde, cache)?,
                Set::S87 => prepare_lanes::<Set87>(records, &proof.c_tilde, cache)?,
                Set::S44 => unreachable!("ML-DSA-44 is verified on its rows"),
            };
            // The Keccak side's geometry depends on the record count only.
            let geometry = Geometry::new(n).map_err(|_| VerifyError::RecordCount)?;
            let mu = public.mu().to_vec();
            (Public::Lanes(public), geometry, digest, mu)
        }
    };
    let width = crate::wire::c_tilde_bytes(set);
    timings.public_data = clock.lap();

    // The transcript through the final-vector step; the instance record binds the proof-of-work plan, derived from the public
    // geometry and the tied parts named by the commitment-phase items.
    let keccak_plan = keccak
        .plan(&geometry, &proof.flock_commitments)
        .ok_or(VerifyError::Plan)?;
    let pow_plan = PowPlan::new(&manifest, flock, &keccak_plan);
    let record = instance_record(&manifest, &digest, flock, &pow_plan.digest());
    let mut transcript = Transcript::replay(&manifest, &record, &proof.work)?;
    let mut commitments: Vec<Commitment<BaseField>> = Vec::with_capacity(GROUP_COUNT);
    for (role, bytes) in [
        (GroupRole::Main16, &proof.main16_commitment),
        (GroupRole::Main, &proof.main_commitment),
    ] {
        let commitment =
            parse_commitment(plan, role, bytes).map_err(|_| VerifyError::Commitment(role))?;
        transcript.absorb(commitment_site(role), 1, bytes)?;
        commitments.push(commitment);
    }
    transcript.absorb_items(FLOCK_COMMIT_SITE, &proof.flock_commitments)?;
    let bridge_seed: [u8; 32] = transcript
        .challenge_bytes(BRIDGE_SEED_SITE, 1)?
        .try_into()
        .expect("32 bytes");
    // Draw `Psi` at its own site to advance the transcript; the PIOP entry point reads the
    // same bytes back from the transcript, so no caller can substitute another matrix.
    transcript.challenge_psi()?;
    timings.transcript = clock.lap();
    let matrices = bridge.matrices_for(set, &bridge_seed);
    timings.matrices = clock.lap();
    for (role, bytes) in [GroupRole::Level1, GroupRole::Level2, GroupRole::Level3, GroupRole::Proj, GroupRole::Mult]
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
    transcript.absorb_items(FLOCK_SHARES_SITE, &proof.flock_shares)?;
    let flock_record = flock_record_digest(&proof.final_vectors, &proof.flock_shares);
    // The public table Y on D_3 (Y_sigma[rho_3] at rho_3 + 2^12 sigma).
    let final_bits: Vec<Vec<u8>> = proof.final_vectors.iter().map(unpack_final_vector).collect();
    let y_table: Vec<u8> = final_bits.concat();
    timings.transcript += clock.lap();
    // The Keccak side per segment, on the final-vector step and the proof items (its outcome
    // does not depend on the PIOP's challenges, so it is checked first).
    for (sigma, bits) in final_bits.iter().enumerate() {
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
            c_tilde: &c_tilde[live.start * width..live.end * width],
            final_vector: bits,
            shares: &proof.flock_shares[sigma],
            record: &flock_record,
            proof: &proof.flock_proofs[sigma],
            matrices: &matrices,
        };
        if !keccak.verify(&segment) {
            return Err(VerifyError::Keccak(sigma));
        }
    }
    timings.keccak = clock.lap();

    // The lookup, tau, the pre-stage claims and the stages; the public pass
    // runs inside stage 3 once r^(3) is known.
    let piop = PiopProof {
        lookup: proof.lookup.clone(),
        multiplicities: proof.multiplicities.clone(),
        prestage: proof.prestage.clone(),
        stage1: proof.stage1.clone(),
        stage2: proof.stage2.clone(),
        stage3: proof.stage3.clone(),
    };
    // The public pass: its window and (T time, A time, keys expanded in this run).
    let mut pass_window = None;
    let mut pass_result: Option<(Duration, Duration, usize)> = None;
    let verified = match public {
        Public::Rows(public, cache) => verify_piop(
            &manifest,
            &matrices,
            &y_table,
            &piop,
            &mut transcript,
            timings,
            |r2, r3| {
                let started = Instant::now();
                let pass = public.pass(r2, r3, cache);
                pass_window = Some((started, Instant::now()));
                pass_result = Some((pass.t_time, pass.a_time, pass.expanded));
                (pass.t_at_r2, pass.ring_weight)
            },
        ),
        Public::Lanes(public) => verify_piop_lanes(
            &manifest,
            &matrices,
            &y_table,
            &piop,
            &mut transcript,
            timings,
            |r2, r3, gammas| {
                let started = Instant::now();
                let t = public.t_evaluations(r2);
                let middle = Instant::now();
                let w = public.ring_weights_at(r2, r3, gammas);
                let finished = Instant::now();
                pass_window = Some((started, finished));
                pass_result = Some((middle - started, finished - middle, public.expanded_keys()));
                (t, w)
            },
        ),
    };
    clock.0 = verified.stage3_end;
    match (pass_window, pass_result) {
        (Some((started, finished)), Some((t_time, a_time, _))) => {
            timings.stage3_rounds = started - verified.stage3_start;
            timings.public_t = t_time;
            timings.public_a = a_time;
            timings.stage3_terminal = verified.stage3_end - finished;
        }
        _ => timings.stage3_rounds = verified.stage3_end - verified.stage3_start,
    }
    let (_, stage3, pi_m) = verified.outcome?;
    let expanded_keys = pass_result.map_or(0, |pass| pass.2);

    // The Flock proofs, checked above by the Keccak side's verifier, are not absorbed: no
    // PIOP or Akita challenge depends on them, and their own challengers bind them.

    // The Keccak side's check of the committed bits at r^(3) (mode B, ML-DSA-44 only).
    if set == Set::S44 {
        let bits = BitsContext {
            geometry,
            commitments: &proof.flock_commitments,
            point: &stage3.point,
            claims: std::array::from_fn(|t| stage3.claims.column(&manifest, Column::W1Bit(t as u8, 0))),
        };
        if !keccak.check_bits(&bits) {
            return Err(VerifyError::Bits);
        }
    }
    timings.bits = clock.lap();

    // The opening.
    let session = transcript.pcs_session()?;
    transcript.ensure_complete()?;
    let claims: [GroupClaim; GROUP_COUNT] = opening_claims(
        &manifest,
        &stage3.point,
        &stage3.claims,
        &pi_m,
        &proof.multiplicities,
    )
    .map_err(VerifyError::Opening)?;
    let commitments: [Commitment<BaseField>; GROUP_COUNT] =
        commitments.try_into().expect("seven commitments");
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

/// What [`verify_piop`] returns: the outcome (`(tau, r^(1), r^(2), lookup points)`, the
/// stage-3 output and the table point `pi_m`) and the stage-3 window for the timings.
pub struct PiopVerified {
    /// Accept with the points and claims, or the first failing check.
    pub outcome: Result<(Points, Stage3Output, Vec<E>), VerifyError>,
    /// When stage 3 started.
    pub stage3_start: Instant,
    /// When stage 3 ended.
    pub stage3_end: Instant,
}

/// Verify the PIOP part (the lookup and the sum-check stages) on a transcript that has absorbed the instance record through the final-vector step (the
/// final vectors and the share items; the Flock proofs are never absorbed):
/// the lookup (`lookup::verify_in`, the claims on the stacked lanes and on `m`), the
/// multiplicity-digit claims (`sum_k 16^k m_k(pi_m) = m(pi_m)`), `tau`, the pre-stage claims
/// with `Y(tau_3)` from `y_table`, and stages 1 to 3 with the public pass deferred
/// (`public_pass(r^(2), r^(3)) = (T(r^(2)), w_A(r^(3)))`). The stage-3 claims and the
/// digit claims are then the opening claims; the caller verifies the opening. `Psi` is read
/// from the transcript's bridge-seed answer, so it cannot be supplied or reused by a caller.
#[allow(clippy::too_many_arguments)]
pub fn verify_piop(
    manifest: &CommitmentManifest,
    matrices: &BridgeMatrices,
    y_table: &[u8],
    proof: &PiopProof,
    transcript: &mut Transcript,
    timings: &mut VerifyTimings,
    public_pass: impl FnOnce(&[E], &[E]) -> (E, E),
) -> PiopVerified {
    verify_piop_with(manifest, matrices, y_table, proof, transcript, timings, |r2, r3, _| {
        let (t, w) = public_pass(r2, r3);
        (vec![t], vec![w])
    })
}

/// [`verify_piop`] for a parameter set with polynomial lanes (ML-DSA-65, ML-DSA-87):
/// `public_pass(r^(2), r^(3), gamma_rows)` returns `T~_rho(r^(2))` per row and
/// `w_A^(s)(r^(3))` per lane of `z` for the ring coefficients `gamma_rows`
/// (`lane_public::LanePublicData::{t_evaluations, ring_weights_at}` on the data that
/// `LanePublicData::prepare` builds from the `(pk, M, c~)` triples).
#[allow(clippy::too_many_arguments)]
pub fn verify_piop_lanes(
    manifest: &CommitmentManifest,
    matrices: &BridgeMatrices,
    y_table: &[u8],
    proof: &PiopProof,
    transcript: &mut Transcript,
    timings: &mut VerifyTimings,
    public_pass: impl FnOnce(&[E], &[E], &[E]) -> (Vec<E>, Vec<E>),
) -> PiopVerified {
    verify_piop_with(manifest, matrices, y_table, proof, transcript, timings, public_pass)
}

#[allow(clippy::too_many_arguments)]
fn verify_piop_with(
    manifest: &CommitmentManifest,
    matrices: &BridgeMatrices,
    y_table: &[u8],
    proof: &PiopProof,
    transcript: &mut Transcript,
    timings: &mut VerifyTimings,
    public_pass: impl FnOnce(&[E], &[E], &[E]) -> (Vec<E>, Vec<E>),
) -> PiopVerified {
    let mut clock = Lap(Instant::now());
    let mut stage3_start = clock.0;
    let outcome = (|| {
        // `Psi` is transcript-owned: the bridge-seed step drew it before the round-2 commitments, and
        // the verifier reads it back instead of accepting one from the caller.
        let psi = transcript.psi().map_err(VerifyError::Transcript)?.to_vec();
        // The fork after the final-vector step: the verifier replays branch L, then branch S, each on its
        // own clone of the sponge, then joins them (L's digest, then S's) as the prover did.
        let (mut lookup_transcript, mut stages_transcript) = transcript.fork()?;

        // Branch L: the lookup and the digits of m.
        let lookup = verify_in(&manifest.lookup_shape(), &proof.lookup, &mut lookup_transcript)?;
        lookup_transcript.absorb_extensions(MULTIPLICITY_SITE, &proof.multiplicities.0)?;
        let (pi_m, m_at_pi) = lookup.multiplicities[0].clone();
        if proof.multiplicities.compose() != m_at_pi {
            return Err(VerifyError::Lookup(LookupError::Terminal(0)));
        }
        let (lookup_inputs, stack_points) = LookupInputs::new(manifest, &lookup);
        timings.lookup = clock.lap();

        // Branch S: tau, the pre-stage claims, Y(tau_3), stages 1 and 2.
        let transcript_s = &mut stages_transcript;
        let mut points = Points {
            tau: transcript_s.challenge_extensions(TAU_SITE, manifest.d_main())?,
            ..Points::default()
        };
        if proof.prestage.y2.len() != manifest.shape().w1_bits || proof.prestage.kappa2.len() != manifest.shape().w1_bits {
            return Err(VerifyError::Stage(StageError::Shape("pre-stage claims".into())));
        }
        transcript_s.absorb_extensions(PRESTAGE_SITE, &proof.prestage.to_vec())?;
        let y_at_tau3 = agg_bridge::final_vector::evaluate(
            y_table,
            &suffix(&points.tau, manifest.group(GroupRole::Level3).logical_vars),
        );
        timings.prestage = clock.lap();

        let stage1 = stage1::verify(
            manifest,
            matrices,
            &points,
            &proof.prestage,
            y_at_tau3,
            &proof.stage1,
            transcript_s,
        )?;
        points.r1 = stage1.point.clone();
        timings.stage1 = clock.lap();
        let stage2 = stage2::verify(manifest, matrices, &points, &stage1, &proof.stage2, transcript_s)?;
        points.r2 = stage2.point.clone();
        timings.stage2 = clock.lap();

        // The join, then stage 3 on the trunk.
        transcript.join(lookup_transcript, stages_transcript)?;
        points.lookup = stack_points;
        stage3_start = clock.0;
        let stage3 = stage3::verify_streamed(
            manifest,
            &points,
            &proof.prestage,
            &stage1,
            &stage2,
            &lookup_inputs,
            &psi,
            &proof.stage3,
            transcript,
            public_pass,
        )?;
        Ok((points, stage3, pi_m))
    })();
    PiopVerified {
        outcome,
        stage3_start,
        stage3_end: Instant::now(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prover::{FixedMatrices, SeededMatrices};
    use agg_witness::bridge::BridgeMatrices;

    const PROFILES: [FlockProfile; 5] = [
        FlockProfile::None,
        FlockProfile::Fast,
        FlockProfile::PublicBits,
        FlockProfile::Slim,
        FlockProfile::Slim8,
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
        assert!(policy(FlockProfile::Slim, true, false).is_ok());
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
        match verify(&[], &[0xff; 16], bridge, keccak, flock, Relation::Hu, None) {
            Err(VerifyError::Configuration(e)) => Some(e),
            Err(VerifyError::Wire(_)) => None,
            other => panic!("unexpected outcome {other:?}"),
        }
    }

    #[test]
    fn verify_refuses_fixed_matrices_in_mode_a_and_mismatched_profiles_before_parsing() {
        let fixed = FixedMatrices(BridgeMatrices::random_for_tests(9));
        for profile in [
            FlockProfile::Fast,
            FlockProfile::Slim,
            FlockProfile::Slim8,
        ] {
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
        let report = verify_timed(&[], &[], &fixed, &fast, FlockProfile::Fast, Relation::Hu, None);
        assert!(matches!(report.outcome, Err(VerifyError::Configuration(_))));
        assert_eq!(report.timings.parse, Duration::ZERO, "refused before the parse");
    }
}
