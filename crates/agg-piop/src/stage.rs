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
use crate::split_weight::{LaneSplitGroup, Level1Split, RankOneGroup, RankOneWeight, SplitGroup};
use crate::transcript::{gamma_site, round_sites, Transcript, TranscriptError};
use crate::OpeningField as E;

/// A public weight table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WeightId {
    /// `w_1` on the main domain, in polynomial lane `rho` (`w_1^(rho)`; lane 0 for ML-DSA-44).
    Level1(u8),
    /// `w_2` on `D_1`.
    Level2,
    /// `w_{3,t}` on `D_2`.
    Level3(u8),
    /// `w_A` on the main domain in order `(j', s, i)`; with polynomial lanes, the weight of
    /// lane `s` of `z` (`w_A^(s)`, the rows combined by the stage-3 ring coefficients).
    Ring(u8),
    /// The projection weight `W(pos, i) = P(pos) eq(r'_sig, i)` on the main domain
    /// (`projection`), in polynomial lane `rho` of `v`.
    Proj(u8),
}

/// A table read by a stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LaneId {
    /// A committed polynomial.
    Column(Column),
    /// `P_Z = Z (Z + 1)` of a leaf-checked digit (virtual; `U_hi`, `K_2`, `K_3`).
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
            Self::Weight(WeightId::Level1(_) | WeightId::Ring(_) | WeightId::Proj(_)) => GroupRole::Main,
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
    /// The low coordinates of the final point of lookup stack `j`,
    /// on the stack's domain.
    Lookup(usize),
    /// `(1/2, ..., 1/2, r^(2)_sig)` on the domain: the signature coordinates of `r^(2)` (its
    /// last `d_sig`) and `1/2` on the others, so that `eq` is `2^-(vars - d_sig)
    /// eq(r^(2)_sig, i)` at every cell of signature `i` (H-FIPS's hint-weight identity).
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

/// The `gamma` index of an instance part outside the batching vector: its coefficient is its
/// scale alone, because the batching coefficients it belongs to are already folded into one
/// of its public weight lanes (the ring weights `w_A^(s)` of the polynomial lanes, which
/// combine the rows `rho` by the stage-3 coefficients of the instances `ring[rho]`).
pub const UNBATCHED: usize = usize::MAX;

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

    /// A part with coefficient one outside the batching vector ([`UNBATCHED`]).
    #[must_use]
    pub fn unbatched(terms: Vec<Term>) -> Self {
        Self::new(UNBATCHED, terms)
    }

    /// The batched coefficient `gamma * scale` (the scale alone for an [`UNBATCHED`] part).
    #[must_use]
    pub fn coefficient(&self, gammas: &[E]) -> E {
        if self.gamma == UNBATCHED {
            self.scale
        } else {
            gammas[self.gamma] * self.scale
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
            for instance in group.instances.iter().filter(|i| i.gamma != UNBATCHED) {
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
    split: Option<Level1Split<'_>>,
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
    split: Option<Level1Split<'_>>,
    transcript: &mut Transcript,
    check: bool,
) -> Result<StageRun, StageError> {
    let gammas = draw_gammas(spec, claims, transcript)?;
    prove_stage_drawn(spec, manifest, points, claims, gammas, resolve, split, None, transcript, check)
}

/// Validate a stage's table and claims and draw its batching coefficients `gamma` (the
/// first event of every stage).
pub fn draw_gammas(spec: &StageSpec, claims: &[E], transcript: &mut Transcript) -> Result<Vec<E>, StageError> {
    spec.validate().map_err(StageError::Internal)?;
    if claims.len() != spec.gamma_count {
        return Err(StageError::Internal("one claim per gamma".into()));
    }
    Ok(transcript.challenge_extensions(&gamma_site(spec.stage), spec.gamma_count)?)
}

/// The rounds of a stage whose coefficients [`draw_gammas`] drew, so that the caller can
/// build lanes that depend on them (the lane ring weights of stage 3) before `resolve` runs.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn prove_stage_drawn<'a>(
    spec: &StageSpec,
    manifest: &CommitmentManifest,
    points: &Points,
    claims: &[E],
    gammas: Vec<E>,
    resolve: impl Fn(LaneId) -> Lane<'a>,
    split: Option<Level1Split<'_>>,
    rank_one: Option<&dyn Fn(LaneId) -> Option<RankOneWeight<'a>>>,
    transcript: &mut Transcript,
    check: bool,
) -> Result<StageRun, StageError> {
    let rounds = manifest.d_main();
    // The lanes, coefficients and equality point of every spec group, gathered in order (all
    // three are views or small vectors). Construction reads no transcript state, so the groups
    // are then built side by side (each parallel inside) and placed back in spec order: the
    // serial builds of the large groups would otherwise each leave most of the pool idle.
    let build_clock = std::time::Instant::now();
    let prepared: Vec<Prepared<'a>> = spec
        .groups
        .iter()
        .enumerate()
        .map(|(g, group)| {
            let lanes: Vec<Lane<'a>> = group.lanes.iter().map(|&id| resolve(id)).collect();
            let instances: Vec<GroupInstance> = group
                .instances
                .iter()
                .map(|instance| GroupInstance {
                    gamma: instance.coefficient(&gammas),
                    terms: instance.terms.clone(),
                })
                .collect();
            let eq_point = points.eq_point(group.eq, manifest, group.domain);
            let factored = rank_one
                .and_then(|rank_one| group.lanes.iter().enumerate().find_map(|(l, &id)| rank_one(id).map(|w| (l, w))));
            Prepared { g, lanes, instances, eq_point, factored }
        })
        .collect();
    let has_lanes = manifest.has_lanes();
    let built: Vec<(usize, Result<Vec<StageGroup<'a>>, StageError>, f64)> = prepared
        .into_par_iter()
        .map(|prepared| {
            let start = std::time::Instant::now();
            let g = prepared.g;
            let built = build_group(spec, prepared, split, has_lanes, rounds);
            (g, built, start.elapsed().as_secs_f64())
        })
        .collect();
    let mut groups = Vec::with_capacity(spec.groups.len());
    // The spec group of every prover group, and the construction time per spec group (the
    // builds overlap, so the times sum to more than the build phase).
    let mut owner = Vec::with_capacity(spec.groups.len());
    let mut build_times = vec![0_f64; spec.groups.len()];
    for (g, built, elapsed) in built {
        for group in built? {
            groups.push(group);
            owner.push(g);
        }
        build_times[g] = elapsed;
    }
    let build_wall = build_clock.elapsed().as_secs_f64();
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
        // The dense-round path of each spec group's generic kernel (kernel lanes, tail slots).
        let mut kernels = vec![String::new(); spec.groups.len()];
        for (built, &g) in groups.iter().zip(&owner) {
            if let Some((lanes, slots)) = built.kernel_shape() {
                kernels[g] = match slots {
                    Some(slots) => format!(", tail on ({lanes} lanes, {slots} slots)"),
                    None => format!(", tail off ({lanes} lanes)"),
                };
            }
        }
        for (g, group) in spec.groups.iter().enumerate() {
            eprintln!(
                "  group {g:2} {:?} eq {:?}: {:3} lanes, {:3} instances, build {:6.1} ms, message {:8.1} ms, bind {:8.1} ms{}",
                group.domain,
                group.eq,
                group.lanes.len(),
                group.instances.len(),
                build_times[g] * 1e3,
                group_times[g].0 * 1e3,
                group_times[g].1 * 1e3,
                kernels[g]
            );
        }
        let per_round: Vec<String> = round_times.iter().map(|t| format!("{:.0}", t * 1e3)).collect();
        eprintln!("  rounds (ms): {}", per_round.join(" "));
        eprintln!("  group builds {:.1} ms (side by side)", build_wall * 1e3);
    }
    let group_claims = groups.iter().fold(E::zero(), |acc, g| acc + g.claim());
    if check && (group_claims != claim || groups.iter().any(|g| !g.consistent())) {
        return Err(StageError::Terminal(spec.stage));
    }
    // The extra linear groups split off a generic group carry no lanes of their own.
    groups.retain(|g| !matches!(g, StageGroup::Extra(_)));
    let terminal_clock = std::time::Instant::now();
    // Every lane is evaluated once, by the first group that reads it and yields terminal
    // values (a linear group yields none); the other groups reading it copy that value, the
    // same multilinear evaluation at the same point.
    let mut first_reader: std::collections::HashMap<LaneId, (usize, usize)> = std::collections::HashMap::new();
    let needed: Vec<Vec<bool>> = spec
        .groups
        .iter()
        .zip(&groups)
        .enumerate()
        .map(|(g, (group, built))| {
            let yields = built.yields_values();
            group
                .lanes
                .iter()
                .enumerate()
                .map(|(position, &id)| {
                    yields && {
                        let first = !first_reader.contains_key(&id);
                        if first {
                            first_reader.insert(id, (g, position));
                        }
                        first
                    }
                })
                .collect()
        })
        .collect();
    let own: Vec<Option<Vec<E>>> = groups
        .par_iter()
        .zip(&needed)
        .map(|(group, needed)| group.terminal_values(needed))
        .collect();
    let lane_values: Vec<Vec<E>> = spec
        .groups
        .iter()
        .map(|group| {
            group
                .lanes
                .iter()
                .map(|id| {
                    let &(g, position) = first_reader.get(id).expect("every lane is read by a group that yields values");
                    own[g].as_ref().expect("the first reader yields values")[position]
                })
                .collect()
        })
        .collect();
    if trace {
        eprintln!("  terminal values {:.1} ms", terminal_clock.elapsed().as_secs_f64() * 1e3);
    }
    let mut pool = agg_simd::buf::Pool::default();
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
    /// The level-1 weights of polynomial lanes as their factors ([`LaneSplitGroup`]).
    LaneSplit(LaneSplitGroup<'a>),
    /// A weight kept as two factors ([`RankOneGroup`]).
    RankOne(RankOneGroup<'a>),
    /// An equality group with its linear-only lanes folded into one lane, or moved to a
    /// linear group ([`split_linear`]) ([`FoldedGroup`]).
    Folded(FoldedGroup<'a>),
}

impl StageGroup<'_> {
    fn set_arena(&mut self, arena: &crate::kernel::SharedArena) {
        match self {
            Self::Generic(g) => g.set_arena(arena),
            Self::Linear(LinearGroup::Row(g)) | Self::Extra(LinearGroup::Row(g)) => g.set_arena(arena),
            Self::Linear(LinearGroup::Plane(_) | LinearGroup::Planes(_))
            | Self::Extra(LinearGroup::Plane(_) | LinearGroup::Planes(_)) => {}
            Self::Weight(WeightGroup::Row(g)) => g.set_arena(arena),
            Self::Weight(WeightGroup::Plane(_)) => {}
            Self::Split(g) => g.set_arena(arena),
            Self::LaneSplit(g) => g.set_arena(arena),
            Self::RankOne(g) => g.set_arena(arena),
            Self::Folded(g) => g.group.set_arena(arena),
        }
    }

    fn round_message(&mut self, round: usize) -> Vec<E> {
        match self {
            Self::Generic(g) => g.round_message(round),
            Self::Linear(g) | Self::Extra(g) => g.round_message(round),
            Self::Weight(g) => g.round_message(round),
            Self::Split(g) => g.round_message(round),
            Self::LaneSplit(g) => g.round_message(round),
            Self::RankOne(g) => g.round_message(round),
            Self::Folded(g) => g.group.round_message(round),
        }
    }

    fn bind(&mut self, round: usize, challenge: E) {
        match self {
            Self::Generic(g) => g.bind(round, challenge),
            Self::Linear(g) | Self::Extra(g) => g.bind(round, challenge),
            Self::Weight(g) => g.bind(round, challenge),
            Self::Split(g) => g.bind(round, challenge),
            Self::LaneSplit(g) => g.bind(round, challenge),
            Self::RankOne(g) => g.bind(round, challenge),
            Self::Folded(g) => g.bind(round, challenge),
        }
    }

    fn claim(&self) -> E {
        match self {
            Self::Generic(g) => g.claim(),
            Self::Linear(g) | Self::Extra(g) => g.claim(),
            Self::Weight(g) => g.claim(),
            Self::Split(g) => g.claim(),
            Self::LaneSplit(g) => g.claim(),
            Self::RankOne(g) => g.claim(),
            Self::Folded(g) => g.group.claim(),
        }
    }

    fn consistent(&self) -> bool {
        match self {
            Self::Generic(_) => true,
            // These groups keep factored weights or folded lanes rather than the materialized
            // lanes, so the prover cannot recompute the terminal values without undoing them.
            // The verifier evaluates every weight itself (`level1_weight_at`,
            // `level1_lane_weights_at`, `lane_weights_at`, `ring_weights_at`), and the
            // equivalence tests pin the messages, claims and terminals against the
            // materialized generic group.
            Self::Split(_) | Self::LaneSplit(_) | Self::RankOne(_) | Self::Folded(_) => true,
            Self::Linear(g) | Self::Extra(g) => g.terminal_consistent(),
            Self::Weight(g) => g.terminal_consistent(),
        }
    }

    /// For a group on the generic kernel: its lane count and, when its dense rounds run on
    /// the plane tail, the tail's slot count.
    fn kernel_shape(&self) -> Option<(usize, Option<usize>)> {
        match self {
            Self::Generic(g) => Some(g.tail_shape()),
            Self::Folded(g) => Some(g.group.tail_shape()),
            _ => None,
        }
    }

    /// True when the group yields terminal values (a linear group reads lanes whose values
    /// another group supplies).
    fn yields_values(&self) -> bool {
        !matches!(self, Self::Linear(_) | Self::Extra(_))
    }

    /// The terminal values of the group's lanes; the lanes with `needed[l]` false may hold any
    /// value (their value comes from another group), so a group skips their evaluation where
    /// it costs a pass over a column.
    fn terminal_values(&self, needed: &[bool]) -> Option<Vec<E>> {
        match self {
            Self::Generic(g) => Some(g.terminal_values()),
            Self::Linear(_) | Self::Extra(_) => None,
            Self::Weight(g) => Some(g.terminal_values(needed)),
            Self::Split(g) => Some(g.terminal_values()),
            Self::LaneSplit(g) => Some(g.terminal_values()),
            Self::RankOne(g) => Some(g.terminal_values(needed)),
            Self::Folded(g) => Some(g.terminal_values(needed)),
        }
    }

    /// Return the plane buffers of the group to `pool`.
    fn recycle(self, pool: &mut agg_simd::buf::Pool) {
        match self {
            Self::Weight(group) => group.recycle(pool),
            Self::Linear(LinearGroup::Plane(group)) | Self::Extra(LinearGroup::Plane(group)) => {
                group.recycle(pool)
            }
            Self::Linear(LinearGroup::Planes(groups)) | Self::Extra(LinearGroup::Planes(groups)) => {
                for group in groups {
                    group.recycle(pool);
                }
            }
            Self::Generic(_)
            | Self::Linear(LinearGroup::Row(_))
            | Self::Extra(LinearGroup::Row(_))
            | Self::Split(_)
            | Self::LaneSplit(_)
            | Self::RankOne(_)
            | Self::Folded(_) => {}
        }
    }
}

/// The linear integer group of either kernel layout.
enum LinearGroup<'a> {
    Row(LinearIntGroup<'a>),
    Plane(PlaneLinearIntGroup<'a>),
    /// More columns than one plane group holds: one plane group per run of
    /// [`PLANE_LINEAR_COLUMNS`] columns, whose messages and claims add up.
    Planes(Vec<PlaneLinearIntGroup<'a>>),
}

/// Columns of one plane-layout linear group (`kernel::plane::LinearIntGroup`).
const PLANE_LINEAR_COLUMNS: usize = 4;

/// The plane-layout linear group of `cells` with `coefficients`, split into runs of
/// [`PLANE_LINEAR_COLUMNS`] columns when there are more (the eight-column main lookup
/// stacks of ML-DSA-65 and ML-DSA-87); `None` when a run does not fit the plane group.
fn plane_linear<'a>(
    cells: &[Source<'a>],
    coefficients: &[E],
    point: &[E],
    pool: &mut agg_simd::buf::Pool,
) -> Option<LinearGroup<'a>> {
    if cells.len() <= PLANE_LINEAR_COLUMNS {
        return PlaneLinearIntGroup::new(cells.to_vec(), coefficients.to_vec(), point, pool).map(LinearGroup::Plane);
    }
    cells
        .chunks(PLANE_LINEAR_COLUMNS)
        .zip(coefficients.chunks(PLANE_LINEAR_COLUMNS))
        .map(|(cells, coefficients)| PlaneLinearIntGroup::new(cells.to_vec(), coefficients.to_vec(), point, pool))
        .collect::<Option<Vec<_>>>()
        .map(LinearGroup::Planes)
}

impl LinearGroup<'_> {
    fn claim(&self) -> E {
        match self {
            Self::Row(g) => g.claim(),
            Self::Plane(g) => g.claim(),
            Self::Planes(groups) => groups.iter().fold(E::zero(), |acc, g| acc + g.claim()),
        }
    }

    fn round_message(&mut self, round: usize) -> Vec<E> {
        match self {
            Self::Row(g) => g.round_message(round),
            Self::Plane(g) => g.round_message(round),
            Self::Planes(groups) => {
                let mut parts = groups.iter_mut().map(|g| g.round_message(round));
                let mut message = parts.next().expect("at least one part");
                for part in parts {
                    for (slot, value) in message.iter_mut().zip(part) {
                        *slot += value;
                    }
                }
                message
            }
        }
    }

    fn bind(&mut self, round: usize, challenge: E) {
        match self {
            Self::Row(g) => g.bind(round, challenge),
            Self::Plane(g) => g.bind(round, challenge),
            Self::Planes(groups) => {
                for g in groups {
                    g.bind(round, challenge);
                }
            }
        }
    }

    fn terminal_consistent(&self) -> bool {
        match self {
            Self::Row(g) => g.terminal_consistent(),
            Self::Plane(g) => g.terminal_consistent(),
            Self::Planes(groups) => groups.iter().all(PlaneLinearIntGroup::terminal_consistent),
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

    fn terminal_values(&self, needed: &[bool]) -> Vec<E> {
        match self {
            Self::Row(g) => g.terminal_values(),
            Self::Plane(g) => g.terminal_values_of(needed),
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
        if let Some(group) = plane_linear(&cells, &coefficients, point, pool) {
            return Some(group);
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

/// A spec group's inputs to [`build_group`].
struct Prepared<'a> {
    /// The spec group.
    g: usize,
    lanes: Vec<Lane<'a>>,
    instances: Vec<GroupInstance>,
    eq_point: Option<Vec<E>>,
    /// The first lane that is a weight kept as two factors, with its factors.
    factored: Option<(usize, RankOneWeight<'a>)>,
}

/// The prover groups of one spec group, in order: one group, or a generic group followed by
/// the linear group split off it ([`split_linear`]). Each path is parallel inside; the groups
/// of a stage are built concurrently, each with its own buffer pool (the stage's pool holds
/// no spare buffers before its rounds, so every buffer is a fresh one either way).
fn build_group<'a>(
    spec: &StageSpec,
    prepared: Prepared<'a>,
    split: Option<Level1Split<'_>>,
    has_lanes: bool,
    rounds: usize,
) -> Result<Vec<StageGroup<'a>>, StageError> {
    let Prepared { g, mut lanes, instances, eq_point, factored } = prepared;
    let group = &spec.groups[g];
    let mut pool = agg_simd::buf::Pool::default();
    if let (Some(split), Some(&LaneId::Weight(WeightId::Level1(0)))) = (split, group.lanes.first()) {
        if group.domain == GroupRole::Main && eq_point.is_none() {
            let grouped = match split {
                Level1Split::Table(split) => {
                    SplitGroup::new(split, &lanes[1..], &instances, spec.degree).map(StageGroup::Split)
                }
                Level1Split::Lanes(split) => {
                    LaneSplitGroup::new(split, &lanes[split.c.len()..], &instances, spec.degree)
                        .map(StageGroup::LaneSplit)
                }
            };
            if let Some(grouped) = grouped {
                return Ok(vec![grouped]);
            }
        }
        // The generic kernel needs the tables the placeholder lanes stand for.
        match split {
            Level1Split::Table(split) => lanes[0] = Lane::Folded(split.materialize()),
            Level1Split::Lanes(split) => {
                for (rho, lane) in lanes.iter_mut().take(split.c.len()).enumerate() {
                    *lane = Lane::Folded(split.materialize(rho));
                }
            }
        }
    }
    // A weight kept as two factors: the rank-one group, else its materialized table.
    if let Some((l, weight)) = factored {
        if eq_point.is_none() {
            if let Some(grouped) = RankOneGroup::new(weight, l, &lanes, &instances, rounds, spec.degree) {
                return Ok(vec![StageGroup::RankOne(grouped)]);
            }
        }
        lanes[l] = Lane::Folded(weight.materialize());
    }
    if let Some(linear) = linear_group(spec, g, &lanes, &instances, eq_point.as_deref(), rounds, &mut pool) {
        return Ok(vec![StageGroup::Linear(linear)]);
    }
    if eq_point.is_none() {
        if let Some(weighted) = weight_group(&lanes, &instances, rounds, &mut pool) {
            return Ok(vec![StageGroup::Weight(weighted)]);
        }
    }
    let generic = |lanes, instances| {
        Group::new(lanes, instances, eq_point.as_deref(), rounds, spec.degree).map_err(StageError::Internal)
    };
    // The sets with polynomial lanes: a full-domain equality group with linear-only lanes as
    // a [`FoldedGroup`], every other as a generic [`Group`].
    if has_lanes {
        let group = match eq_point.as_deref().filter(|p| p.len() == rounds && rounds > 0) {
            Some(point) => match FoldedGroup::new(lanes, instances, point, rounds, spec.degree) {
                Ok(folded) => StageGroup::Folded(folded?),
                Err((lanes, instances)) => StageGroup::Generic(generic(lanes, instances)?),
            },
            None => StageGroup::Generic(generic(lanes, instances)?),
        };
        return Ok(vec![group]);
    }
    // A full-domain equality group whose linear-only integer lanes moved to a linear group
    // keeps the other lanes; the moved lanes' terminal values are evaluated at the end.
    Ok(match split_linear(lanes, instances, eq_point.as_deref(), rounds, &mut pool) {
        Split::Kept(lanes, instances) => vec![StageGroup::Generic(generic(lanes, instances)?)],
        Split::Dropped { lanes, instances, map, linear } => {
            let group = generic(lanes, instances)?;
            vec![
                StageGroup::Folded(FoldedGroup { group, lanes: map, challenges: Vec::new() }),
                StageGroup::Extra(linear),
            ]
        }
    })
}

/// A lane of a [`FoldedGroup`]'s spec group.
enum FoldedLane<'a> {
    /// Lane `index` of the generic group.
    Kept(usize),
    /// A linear-only lane, evaluated at the bound point at the end.
    Dropped(Source<'a>),
}

/// An equality group on the full main domain (the sets with polynomial lanes) whose lanes
/// read only by single-lane products or unweighted affine sums are folded into one field lane
/// `g = sum_l c_l f_l` with their batched coefficients: the generic group carries the other
/// lanes and `g` (one product term), and the folded lanes' values come from their multilinear
/// evaluation at the bound point (prover only). The round polynomial is the same polynomial,
/// so the messages equal the unfolded group's. [`split_linear`] builds the same structure
/// without `g`: the dropped lanes' terms then sit in a separate linear group.
struct FoldedGroup<'a> {
    group: Group<'a>,
    lanes: Vec<FoldedLane<'a>>,
    challenges: Vec<E>,
}

/// Cells per task of the pass that materializes `g`. Each folded lane reads and writes the
/// task's accumulator (32 B per cell), so 1,024 cells keep it at 32 KiB, inside the L1 data
/// cache.
const FOLD_CHUNK: usize = 1 << 10;

/// The term with every lane index `l` replaced by `index[l]`.
fn remap_term(term: Term, index: &[usize]) -> Term {
    match term {
        Term::Product(read) => Term::Product(read.into_iter().map(|l| index[l]).collect()),
        Term::Bool(l) => Term::Bool(index[l]),
        Term::Link(l) => Term::Link(index[l]),
        Term::Leaf(l, poly) => Term::Leaf(index[l], poly),
        Term::RangeW1(lanes) => Term::RangeW1(lanes.map(|l| index[l])),
        Term::Affine { weight, lanes, constant } => Term::Affine {
            weight: weight.map(|l| index[l]),
            lanes: lanes.into_iter().map(|(l, c)| (index[l], c)).collect(),
            constant,
        },
        Term::Monomials(monomials) => Term::Monomials(
            monomials.into_iter().map(|(c, m)| (c, m.into_iter().map(|l| index[l]).collect())).collect(),
        ),
    }
}

/// A lane of the combination `g` of [`fold_combination`].
enum Folding<'a> {
    /// An `i32` column with the canonical coordinates of its coefficient.
    Ints(&'a [i32], [i128; 2]),
    /// A byte column with the canonical coordinates of `c map[byte]` for every byte.
    Bytes(&'a [i8], Box<[[u64; 2]; 256]>),
}

/// `g = sum_l c_l f_l` over `rows` cells for integer or byte lanes `f_l` with coefficients
/// `c_l` given by their canonical coordinates. Per coordinate every lane adds below 2^95 in
/// magnitude (a canonical coefficient below 2^64 times an `i32` cell, or a canonical table
/// entry below 2^64), so fewer than 2^30 lanes sum exactly inside an `i128`, reduced once per
/// cell. A byte lane adds its table entry at the cell's byte instead of multiplying. Every
/// cell is written by the parallel pass (first touch there, no zero fill).
fn fold_combination(folded: &[(Source<'_>, [u64; 2])], rows: usize) -> Vec<E> {
    let lanes: Vec<Folding<'_>> = folded
        .iter()
        .map(|(source, [c0, c1])| match source {
            Source::Ints(column) => Folding::Ints(column, [i128::from(*c0), i128::from(*c1)]),
            Source::Bytes { cells, map } => {
                let c = E::new(agg_simd::F::from_u64(*c0), agg_simd::F::from_u64(*c1));
                Folding::Bytes(
                    cells,
                    Box::new(std::array::from_fn(|byte| {
                        let x = c.mul_i64(map[byte]);
                        [x.c0().to_canonical_u64(), x.c1().to_canonical_u64()]
                    })),
                )
            }
            _ => unreachable!("integer lanes only"),
        })
        .collect();
    let mut g = crate::kernel::buf::uninit::<E>(rows);
    g.par_chunks_mut(FOLD_CHUNK).enumerate().for_each_init(Vec::new, |acc, (chunk, out)| {
        let first = chunk * FOLD_CHUNK;
        let n = out.len();
        acc.clear();
        acc.resize(n, [0_i128; 2]);
        for lane in &lanes {
            match lane {
                Folding::Ints(column, [c0, c1]) => {
                    for (a, &v) in acc.iter_mut().zip(&column[first..first + n]) {
                        let x = i128::from(v);
                        a[0] += c0 * x;
                        a[1] += c1 * x;
                    }
                }
                Folding::Bytes(column, table) => {
                    for (a, &v) in acc.iter_mut().zip(&column[first..first + n]) {
                        let [t0, t1] = table[v as u8 as usize];
                        a[0] += i128::from(t0);
                        a[1] += i128::from(t1);
                    }
                }
            }
        }
        for (slot, &[a0, a1]) in out.iter_mut().zip(acc.iter()) {
            slot.write(E::new(agg_simd::F::from_i128(a0), agg_simd::F::from_i128(a1)));
        }
    });
    // SAFETY: the chunks cover all `rows` slots and the pass wrote each one.
    #[allow(unsafe_code)]
    unsafe { g.assume_init() }.into_vec()
}

/// Per lane of a [`FoldedGroup`]: the batched linear coefficient of a kept integer lane whose
/// every term reads it alone (zero for the other lanes), the part [`FoldedGroup::new`] moves
/// into `g`.
fn linear_parts(lanes: &[Lane<'_>], instances: &[GroupInstance], candidate: &[bool], rows: usize) -> Vec<E> {
    let mut eligible: Vec<bool> = lanes
        .iter()
        .zip(candidate)
        .map(|(lane, &c)| {
            !c && match lane {
                Lane::Source(source @ Source::Ints(_)) => source.len() == rows,
                Lane::Source(source @ Source::Bytes { map, .. }) => {
                    source.len() == rows && map.iter().all(|v| i32::try_from(*v).is_ok())
                }
                _ => false,
            }
        })
        .collect();
    let mut parts = vec![E::zero(); lanes.len()];
    for instance in instances {
        let gamma = instance.gamma;
        for term in &instance.terms {
            match term {
                Term::Product(read) if read.len() == 1 => parts[read[0]] += gamma,
                Term::Product(read) if read.len() == 2 && read[0] == read[1] => {}
                Term::Bool(l) => parts[*l] -= gamma,
                Term::Link(l) => parts[*l] += gamma,
                Term::Leaf(l, poly) => parts[*l] += gamma.mul_i64(poly.coefficients()[1]),
                Term::Affine { weight: None, lanes: affine, .. } => {
                    for &(l, c) in affine {
                        parts[l] += gamma.mul_i64(c);
                    }
                }
                other => {
                    for l in other.lanes() {
                        eligible[l] = false;
                    }
                }
            }
        }
    }
    for (part, eligible) in parts.iter_mut().zip(eligible) {
        if !eligible {
            *part = E::zero();
        }
    }
    parts
}

impl<'a> FoldedGroup<'a> {
    /// The folded group, or the inputs back when fewer than two lanes are linear-only.
    #[allow(clippy::type_complexity)]
    fn new(
        lanes: Vec<Lane<'a>>,
        instances: Vec<GroupInstance>,
        point: &[E],
        rounds: usize,
        degree: usize,
    ) -> Result<Result<Self, StageError>, (Vec<Lane<'a>>, Vec<GroupInstance>)> {
        let rows = 1_usize << rounds;
        let mut candidate: Vec<bool> = lanes
            .iter()
            .map(|lane| {
                let narrow = |v: &i64| i32::try_from(*v).is_ok();
                match lane {
                    Lane::Source(source @ Source::Ints(_)) => source.len() == rows,
                    Lane::Source(source @ Source::Bytes { map, .. }) => source.len() == rows && map.iter().all(narrow),
                    _ => false,
                }
            })
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
        if candidate.iter().filter(|&&c| c).count() < 2 {
            return Err((lanes, instances));
        }
        // A kept integer lane read only by terms in itself alone (squares, Booleanity and
        // link checks, leaf polynomials, linear parts) also moves the linear part of its
        // polynomial into `g`. Its original linear term stays, and a cancelling single-lane
        // product `-c1 f_l` is added; the plan merges the two per-lane coefficients to an
        // exact zero, so the dense rounds need one coefficient slot per such lane instead of
        // two. A plan that stopped merging per-lane coefficients would bring back both slots.
        let linear_parts = linear_parts(&lanes, &instances, &candidate, rows);
        // The batched coefficient of every folded lane; the folded parts leave the terms.
        let mut coefficients = vec![E::zero(); lanes.len()];
        let mut instances: Vec<GroupInstance> = instances
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
                        Term::Affine { weight: None, lanes: affine, constant } => {
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
        for (l, &c1) in linear_parts.iter().enumerate() {
            if !c1.is_zero() {
                coefficients[l] += c1;
                instances.push(GroupInstance { gamma: -c1, terms: vec![Term::Product(vec![l])] });
            }
        }
        let folded: Vec<(Source<'a>, [u64; 2])> = lanes
            .iter()
            .zip(&candidate)
            .zip(&coefficients)
            .enumerate()
            .filter(|(l, ((_, &c), _))| c || !linear_parts[*l].is_zero())
            .map(|(_, x)| x)
            .map(|((lane, _), coefficient)| match lane {
                Lane::Source(source) => {
                    (*source, [coefficient.c0().to_canonical_u64(), coefficient.c1().to_canonical_u64()])
                }
                Lane::Folded(_) => unreachable!("candidates are sources"),
            })
            .collect();
        let g = fold_combination(&folded, rows);
        let mut index = vec![usize::MAX; lanes.len()];
        let mut kept_lanes = Vec::new();
        let mut map = Vec::with_capacity(lanes.len());
        for (l, lane) in lanes.into_iter().enumerate() {
            if candidate[l] {
                let Lane::Source(source) = lane else { unreachable!("candidates are sources") };
                map.push(FoldedLane::Dropped(source));
            } else {
                index[l] = kept_lanes.len();
                map.push(FoldedLane::Kept(kept_lanes.len()));
                kept_lanes.push(lane);
            }
        }
        for instance in &mut instances {
            instance.terms = std::mem::take(&mut instance.terms).into_iter().map(|t| remap_term(t, &index)).collect();
        }
        let g_lane = kept_lanes.len();
        kept_lanes.push(Lane::Folded(g));
        instances.push(GroupInstance { gamma: E::one(), terms: vec![Term::Product(vec![g_lane])] });
        Ok(Group::new(kept_lanes, instances, Some(point), rounds, degree)
            .map_err(StageError::Internal)
            .map(|group| Self { group, lanes: map, challenges: Vec::new() }))
    }

    fn bind(&mut self, round: usize, challenge: E) {
        self.challenges.push(challenge);
        self.group.bind(round, challenge);
    }

    /// The values of the spec group's lanes at the bound point (zero for a dropped lane that
    /// is not `needed`).
    fn terminal_values(&self, needed: &[bool]) -> Vec<E> {
        let kept = self.group.terminal_values();
        let (first, second) = agg_simd::eval::split_eq_tables(&self.challenges);
        self.lanes
            .par_iter()
            .zip(needed)
            .map(|(lane, &needed)| match lane {
                FoldedLane::Kept(i) => kept[*i],
                FoldedLane::Dropped(_) if !needed => E::zero(),
                FoldedLane::Dropped(Source::Ints(cells)) => agg_simd::eval::column_at(cells, &first, &second),
                FoldedLane::Dropped(Source::Bytes { cells, map }) => {
                    agg_simd::eval::column_at(&agg_simd::eval::Mapped { cells, map }, &first, &second)
                }
                FoldedLane::Dropped(_) => unreachable!("integer lanes only"),
            })
            .collect()
    }
}

/// The outcome of [`split_linear`].
enum Split<'a> {
    /// No lane moved: the group's lanes and instances unchanged.
    Kept(Vec<Lane<'a>>, Vec<GroupInstance>),
    /// The linear-only integer lanes moved to `linear`; `lanes` and `instances` are the
    /// generic group without them, `map` places the spec group's lanes.
    Dropped {
        lanes: Vec<Lane<'a>>,
        instances: Vec<GroupInstance>,
        map: Vec<FoldedLane<'a>>,
        linear: LinearGroup<'a>,
    },
}

/// Split the linear terms on integer lanes off an equality group on the full domain: a
/// lane that is an `Ints` table and read only by single-lane products or unweighted affine
/// sums moves its terms (with their coefficients) to one [`LinearIntGroup`] and leaves the
/// generic group, so the generic group neither folds nor carries it; its terminal value is
/// its multilinear evaluation at the bound point (prover only; the sum of the two groups'
/// messages is the original group's).
fn split_linear<'a>(
    lanes: Vec<Lane<'a>>,
    instances: Vec<GroupInstance>,
    eq_point: Option<&[E]>,
    rounds: usize,
    pool: &mut agg_simd::buf::Pool,
) -> Split<'a> {
    let Some(point) = eq_point.filter(|p| p.len() == rounds && rounds > 0) else {
        return Split::Kept(lanes, instances);
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
    // The generic group keeps at least one lane.
    if !candidate.iter().any(|&c| c) || candidate.iter().all(|&c| c) {
        return Split::Kept(lanes, instances);
    }
    let cells: Vec<Source<'a>> = lanes
        .iter()
        .zip(&candidate)
        .filter(|(_, &c)| c)
        .map(|(lane, _)| match lane {
            Lane::Source(source @ Source::Ints(_)) => *source,
            _ => unreachable!("candidates are integer lanes"),
        })
        .collect();
    if cells.iter().any(|source| source.len() != 1 << rounds) {
        // `Group::new` reports the wrong length.
        return Split::Kept(lanes, instances);
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
    let coefficients: Vec<E> = coefficients.into_iter().zip(&candidate).filter(|(_, &c)| c).map(|(e, _)| e).collect();
    let linear = if layout() == Layout::Plane {
        plane_linear(&cells, &coefficients, point, pool)
    } else {
        None
    };
    let Some(linear) = linear.or_else(|| LinearIntGroup::new(cells, coefficients, point).map(LinearGroup::Row)) else {
        unreachable!("integer lanes of the full domain");
    };
    let mut index = vec![usize::MAX; lanes.len()];
    let mut kept_lanes = Vec::new();
    let mut map = Vec::with_capacity(lanes.len());
    for (l, lane) in lanes.into_iter().enumerate() {
        if candidate[l] {
            let Lane::Source(source) = lane else { unreachable!("candidates are sources") };
            map.push(FoldedLane::Dropped(source));
        } else {
            index[l] = kept_lanes.len();
            map.push(FoldedLane::Kept(kept_lanes.len()));
            kept_lanes.push(lane);
        }
    }
    let instances = instances
        .into_iter()
        .map(|instance| GroupInstance {
            gamma: instance.gamma,
            terms: instance.terms.into_iter().map(|t| remap_term(t, &index)).collect(),
        })
        .collect();
    Split::Dropped { lanes: kept_lanes, instances, map, linear }
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
                gamma: instance.coefficient(&replay.gammas),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{field, point};
    use agg_witness::gen::SplitMix64;

    /// A folded group (linear-only lanes in `g`, the linear parts of the squared, Boolean and
    /// link lanes moved into `g`) and a split group (linear-only integer lanes in a linear
    /// group) give the generic group's messages, claim and terminal values.
    #[test]
    fn folded_and_split_groups_match_the_generic_group() {
        let mut rng = SplitMix64::new(29);
        let vars = 9;
        let len = 1_usize << vars;
        let ints: Vec<Vec<i32>> =
            (0..3).map(|_| (0..len).map(|_| (rng.next_u64() % (1 << 16)) as i32 - (1 << 15)).collect()).collect();
        let bits: Vec<Vec<i8>> = (0..3).map(|_| (0..len).map(|_| (rng.next_u64() & 1) as i8).collect()).collect();
        let digits: Vec<i8> = (0..len).map(|_| (rng.next_u64() % 9) as i8 - 4).collect();
        let mut lanes: Vec<Lane<'_>> = ints.iter().map(|c| Lane::Source(Source::Ints(c))).collect();
        lanes.extend(bits.iter().map(|c| Lane::column(c)));
        lanes.push(Lane::column(&digits));
        // Lanes 0, 1 (integers) and 3 (a bit) are linear only; lane 2 is squared, lane 4 is
        // Boolean, lane 5 a bit read linearly and by its Booleanity check, lane 6 a link.
        let instances: Vec<GroupInstance> = vec![
            GroupInstance { gamma: field(&mut rng), terms: vec![Term::Affine { weight: None, lanes: vec![(0, 3), (1, -7), (3, 2), (6, 5)], constant: 11 }] },
            GroupInstance { gamma: field(&mut rng), terms: vec![Term::Product(vec![2, 2]), Term::Product(vec![2])] },
            GroupInstance { gamma: field(&mut rng), terms: vec![Term::Bool(4)] },
            GroupInstance { gamma: field(&mut rng), terms: vec![Term::Bool(5)] },
            GroupInstance { gamma: field(&mut rng), terms: vec![Term::Product(vec![5])] },
            GroupInstance { gamma: field(&mut rng), terms: vec![Term::Link(6)] },
            GroupInstance { gamma: field(&mut rng), terms: vec![Term::Product(vec![1])] },
        ];
        let eq_point = point(&mut rng, vars);
        let degree = 3;
        let mut want = Group::new(lanes.clone(), instances.clone(), Some(&eq_point), vars, degree).unwrap();
        let mut folded = match FoldedGroup::new(lanes.clone(), instances.clone(), &eq_point, vars, degree) {
            Ok(group) => group.unwrap(),
            Err(_) => panic!("the group folds"),
        };
        let mut pool = agg_simd::buf::Pool::default();
        let Split::Dropped { lanes: kept, instances: rest, map, mut linear } =
            split_linear(lanes, instances, Some(&eq_point), vars, &mut pool)
        else {
            panic!("the integer lanes split off")
        };
        let mut split = FoldedGroup {
            group: Group::new(kept, rest, Some(&eq_point), vars, degree).unwrap(),
            lanes: map,
            challenges: Vec::new(),
        };
        assert_eq!(folded.group.claim(), want.claim());
        assert_eq!(split.group.claim() + linear.claim(), want.claim());
        for round in 0..vars {
            let message = want.round_message(round);
            assert_eq!(folded.group.round_message(round), message, "folded, round {round}");
            let parts: Vec<E> = split
                .group
                .round_message(round)
                .into_iter()
                .zip(linear.round_message(round))
                .map(|(a, b)| a + b)
                .collect();
            assert_eq!(parts, message, "split, round {round}");
            let r = field(&mut rng);
            want.bind(round, r);
            folded.bind(round, r);
            split.bind(round, r);
            linear.bind(round, r);
        }
        let want = want.terminal_values();
        assert_eq!(folded.terminal_values(&[true; 7]), want);
        assert_eq!(split.terminal_values(&[true; 7]), want);
        // A dropped lane another group evaluates reads zero; kept lanes always come back.
        let mut partial = want.clone();
        partial[0] = E::zero();
        assert_eq!(split.terminal_values(&[false, true, true, true, true, true, true]), partial);
    }

    /// Eight integer columns (the main lookup stack of ML-DSA-65 and ML-DSA-87) in two plane
    /// groups of four give the messages, claim and binding of the row group over all eight.
    #[test]
    fn eight_linear_columns_run_as_two_plane_groups() {
        let mut rng = SplitMix64::new(17);
        let vars = 12;
        let columns: Vec<Vec<i32>> = (0..8)
            .map(|_| (0..1 << vars).map(|_| (rng.next_u64() % (1 << 16)) as i32 - (1 << 15)).collect())
            .collect();
        let cells: Vec<Source<'_>> = columns.iter().map(|c| Source::Ints(c)).collect();
        let coefficients = point(&mut rng, 8);
        let eq_point = point(&mut rng, vars);
        let mut pool = agg_simd::buf::Pool::default();
        let mut planes = plane_linear(&cells, &coefficients, &eq_point, &mut pool).expect("plane groups");
        assert!(matches!(&planes, LinearGroup::Planes(groups) if groups.len() == 2));
        let mut row = LinearGroup::Row(LinearIntGroup::new(cells, coefficients, &eq_point).expect("row group"));
        assert_eq!(planes.claim(), row.claim());
        for round in 0..vars {
            assert_eq!(planes.round_message(round), row.round_message(round), "round {round}");
            let r = field(&mut rng);
            planes.bind(round, r);
            row.bind(round, r);
        }
        assert!(planes.terminal_consistent() && row.terminal_consistent());
    }
}
