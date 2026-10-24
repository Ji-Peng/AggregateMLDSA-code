//! Shared stage machinery: the declarative instance table of a stage, the prover driver
//! (batching with `gamma`, ordinary compressed rounds, lifted local domains) and the
//! verifier replay with the terminal equation (spec Section 7.1).
//!
//! A [`StageSpec`] lists the groups of a stage: each group names its local domain, its
//! shared equality point (one of `tau`, `r^(1)`, `r^(2)` restricted to the domain, or
//! none), the lanes it reads ([`LaneId`]) and the instances batched on it with their
//! position in the stage's `gamma` vector. The prover resolves lane ids to tables, the
//! verifier to claimed evaluations, so both sides evaluate the same [`Term`]s.

use aerie_falcon::protocol::algebra::eq_evaluation;
use jolt_field::{One, Zero};

use crate::kernel::{evaluate_compressed, expected_terminal, Group, GroupInstance, Lane, Term};
use crate::manifest::{Column, CommitmentManifest, GroupRole};
use crate::transcript::{gamma_site, round_sites, Transcript, TranscriptError};
use crate::OpeningField as E;

/// A public weight table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WeightId {
    /// `w_1` on the main domain (spec (7.5)).
    Level1,
    /// `w_2` on `D_1` (spec (7.2)).
    Level2,
    /// `w_{3,t}` on `D_2` (spec (7.3)).
    Level3(u8),
    /// `w_A` on the main domain in order `(j', s, i)` (spec (7.9)).
    Ring,
}

/// A table read by a stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LaneId {
    /// A committed polynomial.
    Column(Column),
    /// `A_Z^lo = Q_16^lo(P_Z)` of a base-16 digit (virtual).
    DigitLo(Column),
    /// `A_Z^hi = Q_16^hi(P_Z)` of a base-16 digit (virtual).
    DigitHi(Column),
    /// `P_Z = Z (Z + 1)` of a digit (virtual).
    DigitSquare(Column),
    /// `y^(1)_t` on `D_1` (virtual, spec Section 5).
    Parity1(u8),
    /// A public weight.
    Weight(WeightId),
}

impl LaneId {
    /// The domain of the table.
    #[must_use]
    pub fn domain(self) -> GroupRole {
        match self {
            Self::Column(c) | Self::DigitLo(c) | Self::DigitHi(c) | Self::DigitSquare(c) => {
                c.group()
            }
            Self::Parity1(_) => GroupRole::Level1,
            Self::Weight(WeightId::Level1 | WeightId::Ring) => GroupRole::Main,
            Self::Weight(WeightId::Level2) => GroupRole::Level1,
            Self::Weight(WeightId::Level3(_)) => GroupRole::Level2,
        }
    }
}

/// The shared equality point of a group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EqPoint {
    /// No equality factor (the weight is a lane).
    None,
    /// `tau` (spec Section 7.1).
    Tau,
    /// `r^(1)`.
    R1,
    /// `r^(2)`.
    R2,
}

/// One instance: its position in the stage's `gamma` vector and its summands.
#[derive(Clone, Debug)]
pub struct InstanceSpec {
    /// Index into `gamma`.
    pub gamma: usize,
    /// The summands.
    pub terms: Vec<Term>,
}

/// The instances batched on one local domain with one shared equality point.
#[derive(Clone, Debug)]
pub struct GroupSpec {
    /// Local domain.
    pub domain: GroupRole,
    /// Shared equality point.
    pub eq: EqPoint,
    /// The lanes, indexed by the terms.
    pub lanes: Vec<LaneId>,
    /// The instances.
    pub instances: Vec<InstanceSpec>,
}

impl GroupSpec {
    /// Position of a lane in the group.
    #[must_use]
    pub fn lane(&self, id: LaneId) -> usize {
        self.lanes
            .iter()
            .position(|&l| l == id)
            .unwrap_or_else(|| panic!("lane {id:?} is not in the group"))
    }
}

/// The instance table of one stage.
#[derive(Clone, Debug)]
pub struct StageSpec {
    /// Stage number (1 to 3).
    pub stage: usize,
    /// Message degree.
    pub degree: usize,
    /// Number of batching coefficients.
    pub gamma_count: usize,
    /// The groups.
    pub groups: Vec<GroupSpec>,
}

impl StageSpec {
    /// Check that every gamma index appears (an instance may spread its summands over
    /// several groups) and every lane sits on its group's domain.
    pub fn validate(&self) -> Result<(), String> {
        let mut seen = vec![false; self.gamma_count];
        for group in &self.groups {
            for lane in &group.lanes {
                if lane.domain() != group.domain {
                    return Err(format!("lane {lane:?} is not on domain {:?}", group.domain));
                }
            }
            for instance in &group.instances {
                let slot = seen
                    .get_mut(instance.gamma)
                    .ok_or_else(|| format!("gamma index {} out of range", instance.gamma))?;
                *slot = true;
            }
        }
        if let Some(missing) = seen.iter().position(|&s| !s) {
            return Err(format!("gamma index {missing} unused"));
        }
        Ok(())
    }

    /// Every lane id of the stage, in group order, without repetition.
    #[must_use]
    pub fn lane_ids(&self) -> Vec<LaneId> {
        let mut out: Vec<LaneId> = Vec::new();
        for group in &self.groups {
            for &lane in &group.lanes {
                if !out.contains(&lane) {
                    out.push(lane);
                }
            }
        }
        out
    }
}

/// The verifier-side challenge points known when a stage runs.
#[derive(Clone, Debug, Default)]
pub struct Points {
    /// `tau`.
    pub tau: Vec<E>,
    /// `r^(1)`, after stage 1.
    pub r1: Vec<E>,
    /// `r^(2)`, after stage 2.
    pub r2: Vec<E>,
}

/// The last `vars` coordinates of a main-domain point.
#[must_use]
pub fn suffix(point: &[E], vars: usize) -> Vec<E> {
    point[point.len() - vars..].to_vec()
}

impl Points {
    /// The equality point of a group: the named point restricted to the domain.
    #[must_use]
    pub fn eq_point(&self, eq: EqPoint, manifest: &CommitmentManifest, domain: GroupRole) -> Option<Vec<E>> {
        let vars = manifest.group(domain).logical_vars;
        let full = match eq {
            EqPoint::None => return None,
            EqPoint::Tau => &self.tau,
            EqPoint::R1 => &self.r1,
            EqPoint::R2 => &self.r2,
        };
        assert_eq!(full.len(), manifest.d_main(), "a main-domain point");
        Some(suffix(full, vars))
    }
}

/// A stage failure.
#[derive(Debug)]
pub enum StageError {
    /// The transcript rejected an event.
    Transcript(TranscriptError),
    /// A proof message has the wrong shape.
    Shape(String),
    /// The terminal equation failed (verifier) or an internal inconsistency (prover).
    Terminal(usize),
    /// A kernel or specification error.
    Internal(String),
}

impl std::fmt::Display for StageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transcript(e) => write!(f, "transcript: {e}"),
            Self::Shape(what) => write!(f, "malformed stage proof: {what}"),
            Self::Terminal(stage) => write!(f, "stage {stage} terminal check failed"),
            Self::Internal(what) => write!(f, "stage internal error: {what}"),
        }
    }
}

impl std::error::Error for StageError {}

impl From<TranscriptError> for StageError {
    fn from(e: TranscriptError) -> Self {
        Self::Transcript(e)
    }
}

/// What the prover keeps after a stage: the messages, the point and the terminal lane
/// values (per group, in lane order).
#[derive(Clone, Debug)]
pub struct StageRun {
    /// The batching coefficients.
    pub gammas: Vec<E>,
    /// Compressed round messages.
    pub messages: Vec<Vec<E>>,
    /// The terminal point `r^(s)`.
    pub point: Vec<E>,
    /// Terminal lane values per group.
    pub lane_values: Vec<Vec<E>>,
}

impl StageRun {
    /// The terminal value of a lane (looked up in the first group reading it).
    #[must_use]
    pub fn value(&self, spec: &StageSpec, id: LaneId) -> E {
        for (group, values) in spec.groups.iter().zip(&self.lane_values) {
            if let Some(position) = group.lanes.iter().position(|&l| l == id) {
                return values[position];
            }
        }
        panic!("lane {id:?} is not read by the stage")
    }
}

fn gamma_weighted_claim(gammas: &[E], claims: &[E]) -> E {
    gammas
        .iter()
        .zip(claims)
        .fold(E::zero(), |acc, (&g, &c)| acc + g * c)
}

/// Run a stage as the prover: sample `gamma`, batch the groups, send the rounds.
/// `claims[i]` is the claimed sum of the instance with gamma index `i`; `resolve` maps a
/// lane id to its table. The tables' actual sums are checked against the claimed sums
/// (an honest prover's self-check); [`prove_stage_unchecked`] skips that check so tests
/// can hand a tampered witness to the verifier.
pub fn prove_stage<'a>(
    spec: &StageSpec,
    manifest: &CommitmentManifest,
    points: &Points,
    claims: &[E],
    resolve: impl Fn(LaneId) -> Lane<'a>,
    transcript: &mut Transcript,
) -> Result<StageRun, StageError> {
    prove_stage_with(spec, manifest, points, claims, resolve, transcript, true)
}

/// [`prove_stage`] without the prover's claim self-check.
#[doc(hidden)]
pub fn prove_stage_unchecked<'a>(
    spec: &StageSpec,
    manifest: &CommitmentManifest,
    points: &Points,
    claims: &[E],
    resolve: impl Fn(LaneId) -> Lane<'a>,
    transcript: &mut Transcript,
) -> Result<StageRun, StageError> {
    prove_stage_with(spec, manifest, points, claims, resolve, transcript, false)
}

fn prove_stage_with<'a>(
    spec: &StageSpec,
    manifest: &CommitmentManifest,
    points: &Points,
    claims: &[E],
    resolve: impl Fn(LaneId) -> Lane<'a>,
    transcript: &mut Transcript,
    check: bool,
) -> Result<StageRun, StageError> {
    spec.validate().map_err(StageError::Internal)?;
    if claims.len() != spec.gamma_count {
        return Err(StageError::Internal("one claim per gamma".into()));
    }
    let gammas = transcript.challenge_extensions(&gamma_site(spec.stage), spec.gamma_count)?;
    let rounds = manifest.d_main();
    let mut groups = Vec::with_capacity(spec.groups.len());
    for group in &spec.groups {
        let lanes: Vec<Lane<'a>> = group.lanes.iter().map(|&id| resolve(id)).collect();
        let instances: Vec<GroupInstance> = group
            .instances
            .iter()
            .map(|instance| GroupInstance {
                gamma: gammas[instance.gamma],
                terms: instance.terms.clone(),
            })
            .collect();
        let eq_point = points.eq_point(group.eq, manifest, group.domain);
        groups.push(
            Group::new(lanes, instances, eq_point.as_deref(), rounds, spec.degree)
                .map_err(StageError::Internal)?,
        );
    }
    let mut claim = gamma_weighted_claim(&gammas, claims);
    let mut messages = Vec::with_capacity(rounds);
    let mut point = Vec::with_capacity(rounds);
    let trace = std::env::var_os("AGG_STAGE_TRACE").is_some();
    // Per group: (message time, bind time) summed over rounds; per round: total.
    let mut group_times = vec![(0_f64, 0_f64); groups.len()];
    let mut round_times = Vec::with_capacity(rounds);
    for round in 0..rounds {
        let round_start = std::time::Instant::now();
        let mut message = vec![E::zero(); spec.degree];
        for (g, group) in groups.iter_mut().enumerate() {
            let start = std::time::Instant::now();
            for (slot, value) in message.iter_mut().zip(group.round_message(round)) {
                *slot += value;
            }
            group_times[g].0 += start.elapsed().as_secs_f64();
        }
        let (message_site, challenge_site) = round_sites(spec.stage, round);
        transcript.absorb_extensions(&message_site, &message)?;
        let challenge = transcript.challenge_extensions(&challenge_site, 1)?[0];
        claim = evaluate_compressed(&message, claim, challenge);
        for (g, group) in groups.iter_mut().enumerate() {
            let start = std::time::Instant::now();
            group.bind(round, challenge);
            group_times[g].1 += start.elapsed().as_secs_f64();
        }
        round_times.push(round_start.elapsed().as_secs_f64());
        messages.push(message);
        point.push(challenge);
    }
    if trace {
        eprintln!("stage {} trace ({} rounds, degree {}):", spec.stage, rounds, spec.degree);
        for (g, group) in spec.groups.iter().enumerate() {
            eprintln!(
                "  group {g:2} {:?} eq {:?}: {:3} lanes, {:3} instances, message {:8.1} ms, bind {:8.1} ms",
                group.domain,
                group.eq,
                group.lanes.len(),
                group.instances.len(),
                group_times[g].0 * 1e3,
                group_times[g].1 * 1e3
            );
        }
        let per_round: Vec<String> = round_times.iter().map(|t| format!("{:.0}", t * 1e3)).collect();
        eprintln!("  rounds (ms): {}", per_round.join(" "));
    }
    let group_claims = groups.iter().fold(E::zero(), |acc, g| acc + g.claim());
    if check && group_claims != claim {
        return Err(StageError::Terminal(spec.stage));
    }
    let lane_values = groups.iter().map(Group::terminal_values).collect();
    Ok(StageRun {
        gammas,
        messages,
        point,
        lane_values,
    })
}

/// What the verifier keeps after a stage.
#[derive(Clone, Debug)]
pub struct StageReplay {
    /// The batching coefficients.
    pub gammas: Vec<E>,
    /// The terminal point.
    pub point: Vec<E>,
    /// The running claim after the last round.
    pub final_claim: E,
}

/// Replay the rounds of a stage as the verifier: sample `gamma`, absorb the messages and
/// sample the challenges. The terminal check follows in [`check_terminal`] once the
/// terminal claims have been absorbed.
pub fn replay_stage(
    spec: &StageSpec,
    manifest: &CommitmentManifest,
    claims: &[E],
    messages: &[Vec<E>],
    transcript: &mut Transcript,
) -> Result<StageReplay, StageError> {
    spec.validate().map_err(StageError::Internal)?;
    if claims.len() != spec.gamma_count {
        return Err(StageError::Internal("one claim per gamma".into()));
    }
    let rounds = manifest.d_main();
    if messages.len() != rounds || messages.iter().any(|m| m.len() != spec.degree) {
        return Err(StageError::Shape(format!("stage {} rounds", spec.stage)));
    }
    let gammas = transcript.challenge_extensions(&gamma_site(spec.stage), spec.gamma_count)?;
    let mut claim = gamma_weighted_claim(&gammas, claims);
    let mut point = Vec::with_capacity(rounds);
    for (round, message) in messages.iter().enumerate() {
        let (message_site, challenge_site) = round_sites(spec.stage, round);
        transcript.absorb_extensions(&message_site, message)?;
        let challenge = transcript.challenge_extensions(&challenge_site, 1)?[0];
        claim = evaluate_compressed(message, claim, challenge);
        point.push(challenge);
    }
    Ok(StageReplay {
        gammas,
        point,
        final_claim: claim,
    })
}

/// The verifier's terminal equation: the final claim equals the sum over groups of
/// `2^-inactive eq(point, r) sum_i gamma_i term_i(values)`, with `values` the claimed
/// terminal evaluations (`resolve`) of the group's lanes.
pub fn check_terminal(
    spec: &StageSpec,
    manifest: &CommitmentManifest,
    points: &Points,
    replay: &StageReplay,
    resolve: impl Fn(LaneId) -> E,
) -> Result<(), StageError> {
    let rounds = manifest.d_main();
    let mut expected = E::zero();
    for group in &spec.groups {
        let vars = manifest.group(group.domain).logical_vars;
        let local = suffix(&replay.point, vars);
        let eq_factor = points
            .eq_point(group.eq, manifest, group.domain)
            .map_or_else(E::one, |eq| eq_evaluation(&eq, &local));
        let values: Vec<E> = group.lanes.iter().map(|&id| resolve(id)).collect();
        let instances: Vec<GroupInstance> = group
            .instances
            .iter()
            .map(|instance| GroupInstance {
                gamma: replay.gammas[instance.gamma],
                terms: instance.terms.clone(),
            })
            .collect();
        expected += expected_terminal(&instances, &values, eq_factor, rounds - vars);
    }
    if expected != replay.final_claim {
        return Err(StageError::Terminal(spec.stage));
    }
    Ok(())
}
