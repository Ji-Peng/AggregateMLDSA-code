//! Shared stage machinery: the declarative instance table of a stage, the prover driver
//! (batching with `gamma`, ordinary compressed rounds, lifted local domains) and the
//! verifier replay with the terminal equation.
//!
//! A [`StageSpec`] lists the groups of a stage: each group names its local domain, its
//! shared equality point (one of `tau`, `r^(1)`, `r^(2)` restricted to the domain, or
//! none), the lanes it reads ([`LaneId`]) and the instances batched on it with their
//! position in the stage's `gamma` vector. The prover resolves lane ids to tables, the
//! verifier to claimed evaluations, so both sides evaluate the same [`Term`]s.

use aerie_falcon::protocol::algebra::eq_evaluation;
use jolt_field::{Field, One, Ring, Zero};
use rayon::prelude::*;

use crate::kernel::{
    evaluate_compressed, expected_terminal, layout, Group, GroupInstance, Lane, Layout, LinearIntGroup,
    PlaneLinearIntGroup, PlaneWeightIntGroup, Source, Term, WeightIntGroup,
};
use crate::manifest::{Column, CommitmentManifest, GroupRole};
use crate::split_weight::{SplitGroup, SplitWeight};
use crate::transcript::{gamma_site, round_sites, Transcript, TranscriptError};
use crate::OpeningField as E;

/// A public weight table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WeightId {
    /// `w_1` on the main domain.
    Level1,
    /// `w_2` on `D_1`.
    Level2,
    /// `w_{3,t}` on `D_2`.
    Level3(u8),
    /// `w_A` on the main domain in order `(j', s, i)`.
    Ring,
    /// The projection weight `W(pos, i) = P(pos) eq(r'_sig, i)` on the main domain
    /// (`projection`).
    Proj,
}

/// A table read by a stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LaneId {
    /// A committed polynomial.
    Column(Column),
    /// `P_Z = Z (Z + 1)` of a leaf-checked digit (virtual; `Z_hi`, `U_hi`).
    DigitSquare(Column),
    /// `y^(1)_t` on `D_1` (virtual).
    Parity1(u8),
    /// A public weight.
    Weight(WeightId),
}

impl LaneId {
    /// The domain of the table.
    #[must_use]
    pub fn domain(self) -> GroupRole {
        match self {
            Self::Column(c) | Self::DigitSquare(c) => c.domain(),
            Self::Parity1(_) => GroupRole::Level1,
            Self::Weight(WeightId::Level1 | WeightId::Ring | WeightId::Proj) => GroupRole::Main,
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
    /// `tau`.
    Tau,
    /// `r^(1)`.
    R1,
    /// `r^(2)`.
    R2,
    /// The low coordinates of the final point of lookup stack `j` (`docs/lookup.md`),
    /// on the stack's domain.
    Lookup(usize),
    /// `(1/2, ..., 1/2, r^(2)_sig)` on the domain: the signature coordinates of `r^(2)` (its
    /// last `d_sig`) and `1/2` on the others, so that `eq` is `2^-(vars - d_sig)
    /// eq(r^(2)_sig, i)` at every cell of signature `i` (H-FIPS's hint-weight identity,
    /// `docs/h-exact.md`).
    HalfR2,
}

/// One instance (or one part of an instance): its position in the stage's `gamma` vector,
/// a public scale in `K` and its summands. The batched coefficient is `gamma * scale`; an
/// instance with `K` coefficients (a lookup stack reduction with the column weights
/// `eq(pi_top, c)`) is one part per coefficient on the same gamma index.
#[derive(Clone, Debug)]
pub struct InstanceSpec {
    /// Index into `gamma`.
    pub gamma: usize,
    /// The public scale of this part (one for an ordinary instance).
    pub scale: E,
    /// The summands.
    pub terms: Vec<Term>,
}

impl InstanceSpec {
    /// An instance with scale one.
    #[must_use]
    pub fn new(gamma: usize, terms: Vec<Term>) -> Self {
        Self {
            gamma,
            scale: E::one(),
            terms,
        }
    }
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
    /// The low coordinates of every lookup stack's final point (after the lookup), each
    /// on its stack's domain.
    pub lookup: Vec<Vec<E>>,
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
            EqPoint::Lookup(stack) => {
                let point = &self.lookup[stack];
                assert_eq!(point.len(), vars, "a stack point on its domain");
                return Some(point.clone());
            }
            EqPoint::HalfR2 => {
                let d_sig = manifest.d_sig();
                assert!(vars >= d_sig, "the domain holds the signature variables");
                let half = E::from_u64(2).inverse().expect("two is invertible");
                let mut point = vec![half; vars - d_sig];
                point.extend_from_slice(&suffix(&self.r2, d_sig));
                return Some(point);
            }
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
    prove_stage_with(spec, manifest, points, claims, resolve, None, transcript, true)
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
    prove_stage_with(spec, manifest, points, claims, resolve, None, transcript, false)
}

/// The groups of a round run concurrently while the main domain has at least this many
/// unbound variables (`2^12` rows); below it the hand-off costs more than the overlap saves.
const CONCURRENT_ROUNDS_MIN_VARS: usize = 12;

/// [`prove_stage`] (or its unchecked variant) with `w_1` given as its factors: a group
/// whose first lane is the level-1 weight then runs the split rounds of
/// [`crate::split_weight`] when its shape allows, and otherwise on the materialized table.
#[doc(hidden)]
pub fn prove_stage_split<'a>(
    spec: &StageSpec,
    manifest: &CommitmentManifest,
    points: &Points,
    claims: &[E],
    resolve: impl Fn(LaneId) -> Lane<'a>,
    split: Option<&SplitWeight>,
    transcript: &mut Transcript,
    check: bool,
) -> Result<StageRun, StageError> {
    prove_stage_with(spec, manifest, points, claims, resolve, split, transcript, check)
}

#[allow(clippy::too_many_arguments)]
fn prove_stage_with<'a>(
    spec: &StageSpec,
    manifest: &CommitmentManifest,
    points: &Points,
    claims: &[E],
    resolve: impl Fn(LaneId) -> Lane<'a>,
    split: Option<&SplitWeight>,
    transcript: &mut Transcript,
    check: bool,
) -> Result<StageRun, StageError> {
    spec.validate().map_err(StageError::Internal)?;
    if claims.len() != spec.gamma_count {
        return Err(StageError::Internal("one claim per gamma".into()));
    }
    let gammas = transcript.challenge_extensions(&gamma_site(spec.stage), spec.gamma_count)?;
    let rounds = manifest.d_main();
    // Buffers of the plane groups of this stage.
    let mut pool = agg_simd::buf::Pool::default();
    let mut groups = Vec::with_capacity(spec.groups.len());
    // The spec group of every prover group, and the construction time per spec group.
    let mut owner = Vec::with_capacity(spec.groups.len());
    let mut build_times = vec![0_f64; spec.groups.len()];
    for (g, group) in spec.groups.iter().enumerate() {
        let build = std::time::Instant::now();
        let mut lanes: Vec<Lane<'a>> = group.lanes.iter().map(|&id| resolve(id)).collect();
        let instances: Vec<GroupInstance> = group
            .instances
            .iter()
            .map(|instance| GroupInstance {
                gamma: gammas[instance.gamma] * instance.scale,
                terms: instance.terms.clone(),
            })
            .collect();
        let eq_point = points.eq_point(group.eq, manifest, group.domain);
        if let (Some(split), Some(&LaneId::Weight(WeightId::Level1))) = (split, group.lanes.first()) {
            if group.domain == GroupRole::Main && eq_point.is_none() {
                if let Some(grouped) = SplitGroup::new(split, &lanes[1..], &instances, spec.degree) {
                    groups.push(StageGroup::Split(grouped));
                    owner.push(g);
                    build_times[g] = build.elapsed().as_secs_f64();
                    continue;
                }
            }
            // The generic kernel needs the table the placeholder lane stands for.
            lanes[0] = Lane::Folded(split.materialize());
        }
        if let Some(linear) =
            linear_group(spec, g, &lanes, &instances, eq_point.as_deref(), rounds, &mut pool)
        {
            groups.push(StageGroup::Linear(linear));
            owner.push(g);
            build_times[g] = build.elapsed().as_secs_f64();
            continue;
        }
        if eq_point.is_none() {
            if let Some(weighted) = weight_group(&lanes, &instances, rounds, &mut pool) {
                groups.push(StageGroup::Weight(weighted));
                owner.push(g);
                build_times[g] = build.elapsed().as_secs_f64();
                continue;
            }
        }
        let (instances, linear) =
            split_linear(&lanes, instances, eq_point.as_deref(), rounds, &mut pool);
        groups.push(StageGroup::Generic(
            Group::new(lanes, instances, eq_point.as_deref(), rounds, spec.degree).map_err(StageError::Internal)?,
        ));
        owner.push(g);
        if let Some(linear) = linear {
            groups.push(StageGroup::Extra(linear));
            owner.push(g);
        }
        build_times[g] = build.elapsed().as_secs_f64();
    }
    let arena = crate::kernel::TableArena::shared();
    for group in &mut groups { group.set_arena(&arena); }
    let mut claim = gamma_weighted_claim(&gammas, claims);
    let mut messages = Vec::with_capacity(rounds);
    let mut point = Vec::with_capacity(rounds);
    let trace = std::env::var_os("AGG_STAGE_TRACE").is_some();
    if trace {
        crate::alloc::take_stats();
    }
    // Per group: (message time, bind time) summed over rounds; per round: total.
    let mut group_times = vec![(0_f64, 0_f64); spec.groups.len()];
    let mut round_times = Vec::with_capacity(rounds);
    for round in 0..rounds {
        let round_start = std::time::Instant::now();
        // While the main-domain tables are large, the groups' messages are computed
        // concurrently (each group is parallel inside, and the small-domain groups fill the
        // pool beside the main-domain ones); the late rounds are too small to pay for the
        // task hand-off and run in order. Either way the parts are summed in the fixed group
        // order, so the message does not depend on the scheduling. With concurrency the
        // per-group trace times overlap.
        let concurrent = rounds - round >= CONCURRENT_ROUNDS_MIN_VARS;
        let message_part = |group: &mut StageGroup<'a>| {
            let start = std::time::Instant::now();
            let part = group.round_message(round);
            (part, start.elapsed().as_secs_f64())
        };
        let parts: Vec<(Vec<E>, f64)> = if concurrent {
            groups.par_iter_mut().map(message_part).collect()
        } else {
            groups.iter_mut().map(message_part).collect()
        };
        let mut message = vec![E::zero(); spec.degree];
        for (g, (part, time)) in parts.into_iter().enumerate() {
            for (slot, value) in message.iter_mut().zip(part) {
                *slot += value;
            }
            group_times[owner[g]].0 += time;
        }
        let (message_site, challenge_site) = round_sites(spec.stage, round);
        transcript.absorb_extensions(&message_site, &message)?;
        let challenge = transcript.challenge_extensions(&challenge_site, 1)?[0];
        claim = evaluate_compressed(&message, claim, challenge);
        let bind_group = |group: &mut StageGroup<'a>| {
            let start = std::time::Instant::now();
            group.bind(round, challenge);
            start.elapsed().as_secs_f64()
        };
        let bind_times: Vec<f64> = if concurrent {
            groups.par_iter_mut().map(bind_group).collect()
        } else {
            groups.iter_mut().map(bind_group).collect()
        };
        for (g, time) in bind_times.into_iter().enumerate() {
            group_times[owner[g]].1 += time;
        }
        round_times.push(round_start.elapsed().as_secs_f64());
        messages.push(message);
        point.push(challenge);
    }
    if trace {
        // Large blocks allocated during the stage, process-wide (stages 1 and 2 overlap
        // the lookup branch, stage 3 overlaps the Flock branch).
        let (fresh, reused) = crate::alloc::take_stats();
        eprintln!(
            "stage {} trace ({} rounds, degree {}): large blocks fresh {} MiB, reused {} MiB",
            spec.stage,
            rounds,
            spec.degree,
            fresh >> 20,
            reused >> 20
        );
        for (g, group) in spec.groups.iter().enumerate() {
            eprintln!(
                "  group {g:2} {:?} eq {:?}: {:3} lanes, {:3} instances, build {:6.1} ms, message {:8.1} ms, bind {:8.1} ms",
                group.domain,
                group.eq,
                group.lanes.len(),
                group.instances.len(),
                build_times[g] * 1e3,
                group_times[g].0 * 1e3,
                group_times[g].1 * 1e3
            );
        }
        let per_round: Vec<String> = round_times.iter().map(|t| format!("{:.0}", t * 1e3)).collect();
        eprintln!("  rounds (ms): {}", per_round.join(" "));
    }
    let group_claims = groups.iter().fold(E::zero(), |acc, g| acc + g.claim());
    if check && (group_claims != claim || groups.iter().any(|g| !g.consistent())) {
        return Err(StageError::Terminal(spec.stage));
    }
    // The extra linear groups split off a generic group carry no lanes of their own.
    groups.retain(|g| !matches!(g, StageGroup::Extra(_)));
    let mut lane_values: Vec<Option<Vec<E>>> = groups.iter().map(StageGroup::terminal_values).collect();
    // A linear group's lanes are read by another group at the same point (`linear_group`).
    for g in 0..groups.len() {
        if lane_values[g].is_some() {
            continue;
        }
        let values = spec.groups[g]
            .lanes
            .iter()
            .map(|&id| {
                spec.groups
                    .iter()
                    .zip(&lane_values)
                    .find_map(|(other, values)| {
                        let position = other.lanes.iter().position(|&l| l == id)?;
                        values.as_ref().map(|v| v[position])
                    })
                    .expect("a linear group's lane is read by a generic group")
            })
            .collect();
        lane_values[g] = Some(values);
    }
    let lane_values = lane_values.into_iter().map(|v| v.expect("filled")).collect();
    for group in groups {
        group.recycle(&mut pool);
    }
    Ok(StageRun {
        gammas,
        messages,
        point,
        lane_values,
    })
}

/// A prover group of a stage: the generic kernel or the linear path of `kernel::layer`.
#[allow(clippy::large_enum_variant)]
enum StageGroup<'a> {
    Generic(Group<'a>),
    Linear(LinearGroup<'a>),
    /// The linear integer part split off the preceding generic group ([`split_linear`]).
    Extra(LinearGroup<'a>),
    /// A weight times an integer combination ([`weight_group`]).
    Weight(WeightGroup<'a>),
    /// The level-1 weight as its factors ([`SplitGroup`]).
    Split(SplitGroup<'a>),
}

impl StageGroup<'_> {
    fn set_arena(&mut self, arena: &crate::kernel::SharedArena) {
        match self {
            Self::Generic(g) => g.set_arena(arena),
            Self::Linear(LinearGroup::Row(g)) | Self::Extra(LinearGroup::Row(g)) => g.set_arena(arena),
            Self::Linear(LinearGroup::Plane(_)) | Self::Extra(LinearGroup::Plane(_)) => {}
            Self::Weight(WeightGroup::Row(g)) => g.set_arena(arena),
            Self::Weight(WeightGroup::Plane(_)) => {}
            Self::Split(g) => g.set_arena(arena),
        }
    }

    fn round_message(&mut self, round: usize) -> Vec<E> {
        match self {
            Self::Generic(g) => g.round_message(round),
            Self::Linear(g) | Self::Extra(g) => g.round_message(round),
            Self::Weight(g) => g.round_message(round),
            Self::Split(g) => g.round_message(round),
        }
    }

    fn bind(&mut self, round: usize, challenge: E) {
        match self {
            Self::Generic(g) => g.bind(round, challenge),
            Self::Linear(g) | Self::Extra(g) => g.bind(round, challenge),
            Self::Weight(g) => g.bind(round, challenge),
            Self::Split(g) => g.bind(round, challenge),
        }
    }

    fn claim(&self) -> E {
        match self {
            Self::Generic(g) => g.claim(),
            Self::Linear(g) | Self::Extra(g) => g.claim(),
            Self::Weight(g) => g.claim(),
            Self::Split(g) => g.claim(),
        }
    }

    fn consistent(&self) -> bool {
        match self {
            Self::Generic(_) => true,
            // The split group keeps its factors rather than the materialized `w_1` lane, so
            // the prover cannot recompute the terminal values without undoing the split.
            // The verifier checks them independently through `level1_weight_at`, and the
            // equivalence test in `split_weight` pins the messages, claim and terminals
            // against the materialized generic group.
            Self::Split(_) => true,
            Self::Linear(g) | Self::Extra(g) => g.terminal_consistent(),
            Self::Weight(g) => g.terminal_consistent(),
        }
    }

    fn terminal_values(&self) -> Option<Vec<E>> {
        match self {
            Self::Generic(g) => Some(g.terminal_values()),
            Self::Linear(_) | Self::Extra(_) => None,
            Self::Weight(g) => Some(g.terminal_values()),
            Self::Split(g) => Some(g.terminal_values()),
        }
    }

    /// Return the plane buffers of the group to `pool`.
    fn recycle(self, pool: &mut agg_simd::buf::Pool) {
        match self {
            Self::Weight(group) => group.recycle(pool),
            Self::Linear(LinearGroup::Plane(group)) | Self::Extra(LinearGroup::Plane(group)) => {
                group.recycle(pool)
            }
            Self::Generic(_)
            | Self::Linear(LinearGroup::Row(_))
            | Self::Extra(LinearGroup::Row(_))
            | Self::Split(_) => {}
        }
    }
}

/// The linear integer group of either kernel layout.
enum LinearGroup<'a> {
    Row(LinearIntGroup<'a>),
    Plane(PlaneLinearIntGroup<'a>),
}

impl LinearGroup<'_> {
    fn claim(&self) -> E {
        match self {
            Self::Row(g) => g.claim(),
            Self::Plane(g) => g.claim(),
        }
    }

    fn round_message(&mut self, round: usize) -> Vec<E> {
        match self {
            Self::Row(g) => g.round_message(round),
            Self::Plane(g) => g.round_message(round),
        }
    }

    fn bind(&mut self, round: usize, challenge: E) {
        match self {
            Self::Row(g) => g.bind(round, challenge),
            Self::Plane(g) => g.bind(round, challenge),
        }
    }

    fn terminal_consistent(&self) -> bool {
        match self {
            Self::Row(g) => g.terminal_consistent(),
            Self::Plane(g) => g.terminal_consistent(),
        }
    }
}

/// The weight-times-integer group of either kernel layout.
enum WeightGroup<'a> {
    Row(WeightIntGroup<'a>),
    Plane(PlaneWeightIntGroup<'a>),
}

impl WeightGroup<'_> {
    fn claim(&self) -> E {
        match self {
            Self::Row(g) => g.claim(),
            Self::Plane(g) => g.claim(),
        }
    }

    fn round_message(&mut self, round: usize) -> Vec<E> {
        match self {
            Self::Row(g) => g.round_message(round),
            Self::Plane(g) => g.round_message(round),
        }
    }

    fn bind(&mut self, round: usize, challenge: E) {
        match self {
            Self::Row(g) => g.bind(round, challenge),
            Self::Plane(g) => g.bind(round, challenge),
        }
    }

    fn terminal_values(&self) -> Vec<E> {
        match self {
            Self::Row(g) => g.terminal_values(),
            Self::Plane(g) => g.terminal_values(),
        }
    }

    fn terminal_consistent(&self) -> bool {
        match self {
            Self::Row(g) => g.terminal_consistent(),
            Self::Plane(g) => g.terminal_consistent(),
        }
    }

    fn recycle(self, pool: &mut agg_simd::buf::Pool) {
        match self {
            Self::Row(_) => {}
            Self::Plane(g) => g.recycle(pool),
        }
    }
}

/// The linear path for group `g` (prover only; the round messages are the generic
/// kernel's): an equality group on the full domain whose lanes are all integer tables read
/// only by single-lane products (the stage-3 reductions of the main lookup stack), each lane
/// also read by a generic group of the stage, which supplies its terminal value.
fn linear_group<'a>(
    spec: &StageSpec,
    g: usize,
    lanes: &[Lane<'a>],
    instances: &[GroupInstance],
    eq_point: Option<&[E]>,
    rounds: usize,
    pool: &mut agg_simd::buf::Pool,
) -> Option<LinearGroup<'a>> {
    let point = eq_point.filter(|p| p.len() == rounds && rounds > 0)?;
    let cells: Vec<Source<'a>> = lanes
        .iter()
        .map(|lane| match lane {
            // Byte lanes stay on the generic kernel's coded path, which is faster for them.
            Lane::Source(source @ Source::Ints(_)) => Some(*source),
            _ => None,
        })
        .collect::<Option<_>>()?;
    let mut coefficients = vec![E::zero(); lanes.len()];
    for instance in instances {
        for term in &instance.terms {
            match term {
                Term::Product(read) if read.len() == 1 => coefficients[read[0]] += instance.gamma,
                _ => return None,
            }
        }
    }
    let ids = &spec.groups[g].lanes;
    let elsewhere = |id: &LaneId| {
        spec.groups
            .iter()
            .enumerate()
            .any(|(other, group)| other != g && group.lanes.contains(id) && !linear_shape(group))
    };
    if !ids.iter().all(elsewhere) {
        return None;
    }
    if layout() == Layout::Plane {
        if let Some(group) = PlaneLinearIntGroup::new(cells.clone(), coefficients.clone(), point, pool) {
            return Some(LinearGroup::Plane(group));
        }
    }
    LinearIntGroup::new(cells, coefficients, point).map(LinearGroup::Row)
}

/// The weighted path (prover only; the generic kernel's messages): a full-domain group
/// without an equality factor whose single instance is one weighted affine term
/// `gamma w (sum_l c_l f_l + c)` with a field weight `w` and integer or byte lanes `f_l`
/// (stage 3's `w_A z`).
fn weight_group<'a>(
    lanes: &[Lane<'a>],
    instances: &[GroupInstance],
    rounds: usize,
    pool: &mut agg_simd::buf::Pool,
) -> Option<WeightGroup<'a>> {
    let [instance] = instances else { return None };
    let [Term::Affine {
        weight: Some(weight),
        lanes: affine,
        constant,
    }] = instance.terms.as_slice()
    else {
        return None;
    };
    let sources: Vec<Source<'a>> = lanes
        .iter()
        .map(|lane| match lane {
            Lane::Source(source) => Some(*source),
            Lane::Folded(_) => None,
        })
        .collect::<Option<_>>()?;
    let mut coefficients = vec![0_i64; lanes.len()];
    for &(l, c) in affine {
        if l == *weight || l >= lanes.len() {
            return None;
        }
        coefficients[l] += c;
    }
    if layout() == Layout::Plane {
        if let Some(group) =
            PlaneWeightIntGroup::new(&sources, *weight, &coefficients, *constant, instance.gamma, rounds, pool)
        {
            return Some(WeightGroup::Plane(group));
        }
    }
    WeightIntGroup::new(&sources, *weight, &coefficients, *constant, instance.gamma, rounds)
        .map(WeightGroup::Row)
}

/// Split the linear terms on integer lanes off an equality group on the full domain: a
/// lane that is an `Ints` table and read only by single-lane products or unweighted affine
/// sums moves its terms (with their coefficients) to one [`LinearIntGroup`]; the lane stays
/// in the generic group, unread, so that the group still folds it and yields its terminal
/// value (prover only; the sum of the two groups' messages is the original group's).
fn split_linear<'a>(
    lanes: &[Lane<'a>],
    instances: Vec<GroupInstance>,
    eq_point: Option<&[E]>,
    rounds: usize,
    pool: &mut agg_simd::buf::Pool,
) -> (Vec<GroupInstance>, Option<LinearGroup<'a>>) {
    let Some(point) = eq_point.filter(|p| p.len() == rounds && rounds > 0) else {
        return (instances, None);
    };
    // Lanes read anywhere outside a linear position are not candidates.
    let mut candidate: Vec<bool> = lanes
        .iter()
        .map(|lane| matches!(lane, Lane::Source(Source::Ints(_))))
        .collect();
    for instance in &instances {
        for term in &instance.terms {
            match term {
                Term::Product(read) if read.len() == 1 => {}
                Term::Affine { weight: None, .. } => {}
                other => {
                    for l in other.lanes() {
                        candidate[l] = false;
                    }
                }
            }
        }
    }
    if !candidate.iter().any(|&c| c) {
        return (instances, None);
    }
    let mut coefficients = vec![E::zero(); lanes.len()];
    let instances: Vec<GroupInstance> = instances
        .into_iter()
        .map(|instance| {
            let gamma = instance.gamma;
            let terms = instance
                .terms
                .into_iter()
                .filter_map(|term| match term {
                    Term::Product(read) if read.len() == 1 && candidate[read[0]] => {
                        coefficients[read[0]] += gamma;
                        None
                    }
                    Term::Affine {
                        weight: None,
                        lanes: affine,
                        constant,
                    } => {
                        let kept: Vec<(usize, i64)> = affine
                            .into_iter()
                            .filter(|&(l, c)| {
                                if candidate[l] {
                                    coefficients[l] += gamma.mul_i64(c);
                                }
                                !candidate[l]
                            })
                            .collect();
                        (!kept.is_empty() || constant != 0).then_some(Term::Affine {
                            weight: None,
                            lanes: kept,
                            constant,
                        })
                    }
                    other => Some(other),
                })
                .collect();
            GroupInstance { gamma, terms }
        })
        .collect();
    let (cells, coefficients): (Vec<Source<'a>>, Vec<E>) = lanes
        .iter()
        .zip(coefficients)
        .zip(&candidate)
        .filter(|(_, &c)| c)
        .map(|((lane, coefficient), _)| match lane {
            Lane::Source(source @ Source::Ints(_)) => (*source, coefficient),
            _ => unreachable!("candidates are integer lanes"),
        })
        .unzip();
    let linear = if layout() == Layout::Plane {
        PlaneLinearIntGroup::new(cells.clone(), coefficients.clone(), point, pool)
            .map(LinearGroup::Plane)
    } else {
        None
    };
    let linear = linear.or_else(|| LinearIntGroup::new(cells, coefficients, point).map(LinearGroup::Row));
    // `new` refuses only lanes of the wrong length, which `Group::new` then reports.
    (instances, linear)
}

/// True when a group spec has the shape [`linear_group`] may take (single-lane products
/// only); such a group never supplies terminal values to another.
fn linear_shape(group: &GroupSpec) -> bool {
    group.instances.iter().all(|instance| {
        instance
            .terms
            .iter()
            .all(|term| matches!(term, Term::Product(read) if read.len() == 1))
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
                gamma: replay.gammas[instance.gamma] * instance.scale,
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
