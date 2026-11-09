//! Stage 1: bridge Booleanity and levels 2 and 3. Every instance lives on a bridge domain
//! (`D_1`, `D_2`); the main domain has no stage-1 instance. The Booleanity of the w1Encode
//! bits is a stage-3 zero-check at `r^(2)` and ML-DSA-44's bit-range identity a stage-2
//! zero-check at `r^(1)`: both read lanes those stages read anyway, and their terminals land
//! at points where the bits are claimed already. The lookup into `T_16` proves the other
//! ranges.
//!
//! `gamma^(1)` order with `w` bits of `w_1'` (6 for ML-DSA-44, 4 otherwise): `bool_y1[t]`
//! (`0..w`), `level2[t]` (`w..2w`), `level3` (`2w`), `bool_y2[t]` (`2w+1..3w+1`): 19
//! instances for ML-DSA-44 and 13 otherwise, all of degree 3.
//!
//! Under `HashField::Fp` there is no bridge: stage 1 holds only the Keccak zero-checks
//! (`keccak_fp::specs::stage1_groups`, `65 Q + L` instances, degree 4 with the cubic
//! `chi` and parity terms), and its terminal claims are the Keccak lane claims.

use std::collections::HashMap;

use agg_fpkeccak::KeccakLayout;
use agg_witness::bridge::BridgeMatrices;
use agg_witness::padding::Geometry;
use agg_witness::params::{TR_BYTES, W1_BITS};
use jolt_field::Zero;

use crate::claims::{PreStageClaims, Stage1Claims};
use crate::keccak_fp::specs::stage1_groups;
use crate::keccak_fp::virtuals::{c_tilde_mle_at_lanes, mu_mle_at_lanes};
use crate::kernel::Term;
use crate::manifest::{Column, CommitmentManifest, GroupRole, HashField};
use crate::stage::{
    check_terminal, prove_stage, prove_stage_unchecked, replay_stage, suffix, EqPoint, GroupSpec, InstanceSpec, LaneId,
    Points, StageError, StageSpec, WeightId,
};
use crate::tables::ProverTables;
use crate::transcript::{terminal_site, Transcript};
use crate::weights::{level2_weight_at, level2_weight_at_for, level3_weights_at, level3_weights_at_for};
use crate::OpeningField as E;

/// Stage number.
pub const STAGE: usize = 1;
/// Message degree (Booleanity with its equality factor) under `HashField::Gf2`, the same
/// for every set.
pub const DEGREE: usize = 3;
/// Message degree under `HashField::Fp` (the cubic Keccak zero-checks with their equality
/// factor).
pub const FP_DEGREE: usize = 4;
/// Batching coefficients of ML-DSA-44 under `HashField::Gf2` ([`gamma_count`]).
pub const GAMMA_COUNT: usize = 19;

/// The message degree of `manifest`'s parameter set and hash field ([`DEGREE`] under
/// `Gf2`, [`FP_DEGREE`] under `Fp`).
#[must_use]
pub fn degree(manifest: &CommitmentManifest) -> usize {
    match manifest.hash_field() {
        HashField::Gf2 => DEGREE,
        HashField::Fp => FP_DEGREE,
    }
}

/// The `gamma^(1)` layout of a parameter set: the bridge checks over `w` bits of `w_1'`
/// under `HashField::Gf2`, the Keccak zero-check instances under `HashField::Fp`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gammas {
    /// Bits of `w_1'`.
    pub bits: usize,
    /// Commitment-hash field (`Gf2` or `Fp`).
    pub hash_field: HashField,
    /// Keccak zero-check instances (`0` under `Gf2`, `65 Q + L` under `Fp`).
    pub keccak: usize,
}

impl Gammas {
    /// The layout of `manifest`'s parameter set and hash field.
    #[must_use]
    pub fn new(manifest: &CommitmentManifest) -> Self {
        let keccak = match manifest.hash_field() {
            HashField::Gf2 => 0,
            HashField::Fp => {
                let layout = KeccakLayout::for_set(manifest.set());
                65 * layout.slot_groups + layout.output_lanes
            }
        };
        Self {
            bits: manifest.shape().w1_bits,
            hash_field: manifest.hash_field(),
            keccak,
        }
    }

    /// Batching coefficients of the bridge checks (`HashField::Gf2` only).
    const fn bridge_count(self) -> usize {
        match self.hash_field {
            HashField::Gf2 => 3 * self.bits + 1,
            HashField::Fp => 0,
        }
    }

    /// Gamma index of `bool_y1[t]` (`HashField::Gf2`).
    #[must_use]
    pub const fn bool_y1(self, t: usize) -> usize {
        t
    }

    /// Gamma index of `level2[t]` (`HashField::Gf2`).
    #[must_use]
    pub const fn level2(self, t: usize) -> usize {
        self.bits + t
    }

    /// Gamma index of `level3` (`HashField::Gf2`).
    #[must_use]
    pub const fn level3(self) -> usize {
        2 * self.bits
    }

    /// Gamma index of `bool_y2[t]` (`HashField::Gf2`).
    #[must_use]
    pub const fn bool_y2(self, t: usize) -> usize {
        2 * self.bits + 1 + t
    }

    /// First gamma index of the Keccak zero-checks (`HashField::Fp`).
    #[must_use]
    pub const fn keccak_offset(self) -> usize {
        self.bridge_count()
    }

    /// Batching coefficients.
    #[must_use]
    pub const fn count(self) -> usize {
        self.bridge_count() + self.keccak
    }
}

/// Batching coefficients of `manifest`'s parameter set and hash field (under `Gf2`: 19 or
/// 13; under `Fp`: 719 for ML-DSA-44).
#[must_use]
pub fn gamma_count(manifest: &CommitmentManifest) -> usize {
    Gammas::new(manifest).count()
}

/// Gamma index of `bool_y1[t]`.
#[must_use]
pub const fn gamma_bool_y1(t: usize) -> usize {
    t
}
/// Gamma index of `level2[t]` (ML-DSA-44).
#[must_use]
pub const fn gamma_level2(t: usize) -> usize {
    W1_BITS + t
}
/// Gamma index of `level3` (ML-DSA-44).
pub const GAMMA_LEVEL3: usize = 2 * W1_BITS;
/// Gamma index of `bool_y2[t]` (ML-DSA-44).
#[must_use]
pub const fn gamma_bool_y2(t: usize) -> usize {
    2 * W1_BITS + 1 + t
}

/// The stage-1 proof: the round messages and the 12 terminal claims.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1Proof {
    /// Compressed degree-3 messages (3 elements each).
    pub rounds: Vec<Vec<E>>,
    /// Terminal claims.
    pub claims: Stage1Claims,
}

/// What later stages need from stage 1.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1Output {
    /// `r^(1)`.
    pub point: Vec<E>,
    /// Terminal claims.
    pub claims: Stage1Claims,
}

/// The instance table.
#[must_use]
pub fn spec(manifest: &CommitmentManifest) -> StageSpec {
    let gammas = Gammas::new(manifest);
    let lanes = 0..gammas.bits;
    let mut groups = Vec::new();
    match manifest.hash_field() {
        HashField::Gf2 => {
        // One equality group per bridge domain, holding the Booleanity checks.
        groups.push(GroupSpec {
            domain: GroupRole::Level1,
            eq: EqPoint::Tau,
            lanes: lanes.clone().map(|t| LaneId::Parity1(t as u8)).collect(),
            instances: lanes.clone().map(|t| InstanceSpec::new(gammas.bool_y1(t), vec![Term::Bool(t)])).collect(),
        });
        groups.push(GroupSpec {
            domain: GroupRole::Level2,
            eq: EqPoint::Tau,
            lanes: lanes.clone().map(|t| LaneId::Column(Column::Parity2(t as u8))).collect(),
            instances: lanes.clone().map(|t| InstanceSpec::new(gammas.bool_y2(t), vec![Term::Bool(t)])).collect(),
        });
        // level2[t] on D_1 with the weight w_2.
        groups.push(GroupSpec {
            domain: GroupRole::Level1,
            eq: EqPoint::None,
            lanes: std::iter::once(LaneId::Weight(WeightId::Level2))
                .chain(lanes.clone().map(|t| LaneId::Parity1(t as u8)))
                .collect(),
            instances: lanes
                .clone()
                .map(|t| InstanceSpec::new(gammas.level2(t), vec![Term::Product(vec![0, 1 + t])]))
                .collect(),
        });
        // level3 on D_2 summed over the bits with the weights w_{3,t}.
        groups.push(GroupSpec {
            domain: GroupRole::Level2,
            eq: EqPoint::None,
            lanes: lanes
                .clone()
                .map(|t| LaneId::Weight(WeightId::Level3(t as u8)))
                .chain(lanes.clone().map(|t| LaneId::Column(Column::Parity2(t as u8))))
                .collect(),
            instances: vec![InstanceSpec::new(
                gammas.level3(),
                lanes
                    .clone()
                    .map(|t| Term::Product(vec![t, gammas.bits + t]))
                    .collect(),
            )],
        });
        }
        HashField::Fp => {
            let (keccak, used) = stage1_groups(manifest, gammas.keccak_offset());
            debug_assert_eq!(used, gammas.keccak);
            groups.extend(keccak);
        }
    }
    StageSpec {
        stage: STAGE,
        degree: degree(manifest),
        gamma_count: gammas.count(),
        groups,
        folded: 0..0,
    }
}

/// The claimed sums in gamma order: zero for the zero-checks, the level-2 and level-3 sums
/// of the pre-stage claims for the bridge level instances under `HashField::Gf2`.
#[must_use]
pub fn claimed_sums(manifest: &CommitmentManifest, prestage: &PreStageClaims, y_at_tau3: E) -> Vec<E> {
    let gammas = Gammas::new(manifest);
    let mut claims = vec![E::zero(); gammas.count()];
    if manifest.hash_field() == HashField::Gf2 {
        for t in 0..gammas.bits {
            let kappa2 = prestage.kappa2_value(t);
            claims[gammas.level2(t)] = prestage.y2[t] + kappa2 + kappa2;
        }
        claims[gammas.level3()] =
            agg_bridge::final_vector::level3_claimed_sum(y_at_tau3, prestage.kappa3_value());
    }
    claims
}

/// Public per-record hash inputs (`mu_i` and `c~_i`) for the verifier's evaluation of the
/// `KeccakMu` and `KeccakCTilde` tables under `HashField::Fp`.
#[derive(Clone, Copy, Debug)]
pub struct KeccakPublicInputs<'a> {
    /// `mu_i` per live record (padded records `i >= N` have `mu = 0`).
    pub mu: &'a [[u8; TR_BYTES]],
    /// `c~_i` per live record (padded records `i >= N` use `zero_message_c_tilde`).
    pub c_tilde: &'a [Vec<u8>],
}

/// True for a committed or virtual Keccak lane whose terminal evaluation travels in the
/// stage claims (`Column` on a Keccak group, `KeccakRot`, `KeccakPrev`, `KeccakPrevIota`,
/// `KeccakBlk`).
#[must_use]
pub const fn is_keccak_claim_lane(id: LaneId) -> bool {
    matches!(
        id,
        LaneId::Column(
            Column::KeccakA { .. } | Column::KeccakC { .. } | Column::KeccakCPrime { .. }
        ) | LaneId::KeccakRot(_)
            | LaneId::KeccakPrev(_)
            | LaneId::KeccakPrevIota(_)
            | LaneId::KeccakBlk { .. }
    )
}

/// True for any Keccak-domain lane (claimed or public).
#[must_use]
pub const fn is_keccak_lane(id: LaneId) -> bool {
    is_keccak_claim_lane(id) || matches!(id, LaneId::KeccakMu(_) | LaneId::KeccakCTilde(_))
}

/// Every non-public Keccak lane of `spec` in `spec.lane_ids()` order.
#[must_use]
pub fn keccak_claim_lanes(spec: &StageSpec) -> Vec<LaneId> {
    spec.lane_ids()
        .into_iter()
        .filter(|&id| is_keccak_claim_lane(id))
        .collect()
}

thread_local! {
    /// Verifier time spent on this thread on the Keccak terminal work of the stages: the
    /// public `mu` and `c~` table evaluations and the stage-3 Keccak reduction weights.
    static KECCAK_TERMINAL: std::cell::Cell<std::time::Duration> =
        const { std::cell::Cell::new(std::time::Duration::ZERO) };
}

/// Run `f` and add its wall time to this thread's Keccak terminal clock.
pub(crate) fn timed_keccak_terminal<T>(f: impl FnOnce() -> T) -> T {
    let started = std::time::Instant::now();
    let out = f();
    KECCAK_TERMINAL.with(|clock| clock.set(clock.get() + started.elapsed()));
    out
}

/// This thread's Keccak terminal clock since the last call (the clock is reset).
pub(crate) fn take_keccak_terminal() -> std::time::Duration {
    KECCAK_TERMINAL.with(|clock| clock.replace(std::time::Duration::ZERO))
}

/// Precomputed verifier table of the terminal evaluations of every Keccak lane read by a
/// stage: non-public Keccak lanes are indexed from `keccak_claims` (in [`keccak_claim_lanes`]
/// order) and every public `KeccakMu` and `KeccakCTilde` lane read by `spec` is evaluated
/// once at the Keccak suffix of `point`.
#[derive(Clone, Debug)]
pub struct KeccakResolver {
    stage: usize,
    values: HashMap<LaneId, E>,
}

impl KeccakResolver {
    /// Build the Keccak lane table for `spec` at the stage terminal `point`.
    #[must_use]
    pub fn new(
        manifest: &CommitmentManifest,
        spec: &StageSpec,
        point: &[E],
        keccak_claims: &[E],
        public: Option<KeccakPublicInputs<'_>>,
    ) -> Self {
        let claim_lanes = keccak_claim_lanes(spec);
        assert_eq!(
            keccak_claims.len(),
            claim_lanes.len(),
            "stage {} expected {} Keccak lane claims, got {}",
            spec.stage,
            claim_lanes.len(),
            keccak_claims.len()
        );
        let mut values = HashMap::with_capacity(claim_lanes.len() + 16);
        for (id, &val) in claim_lanes.into_iter().zip(keccak_claims) {
            values.insert(id, val);
        }
        let mut mu_lanes = Vec::new();
        let mut c_tilde_lanes = Vec::new();
        for id in spec.lane_ids() {
            match id {
                LaneId::KeccakMu(lane) => mu_lanes.push(lane),
                LaneId::KeccakCTilde(lane) => c_tilde_lanes.push(lane),
                _ => {}
            }
        }
        if !mu_lanes.is_empty() || !c_tilde_lanes.is_empty() {
            let public = public.unwrap_or_else(|| {
                panic!("KeccakPublicInputs required for the public Keccak lanes of stage {}", spec.stage)
            });
            let r_k = suffix(point, manifest.d_keccak());
            let as_index = |lanes: &[u8]| lanes.iter().map(|&l| usize::from(l)).collect::<Vec<_>>();
            if !mu_lanes.is_empty() {
                let vals = mu_mle_at_lanes(public.mu, &r_k, &as_index(&mu_lanes));
                values.extend(mu_lanes.iter().map(|&l| LaneId::KeccakMu(l)).zip(vals));
            }
            if !c_tilde_lanes.is_empty() {
                let layout = KeccakLayout::for_set(manifest.set());
                let vals = c_tilde_mle_at_lanes(&layout, public.c_tilde, &r_k, &as_index(&c_tilde_lanes));
                values.extend(c_tilde_lanes.iter().map(|&l| LaneId::KeccakCTilde(l)).zip(vals));
            }
        }
        Self {
            stage: spec.stage,
            values,
        }
    }

    /// Look up the terminal evaluation of a Keccak lane `id`.
    #[must_use]
    pub fn resolve(&self, id: LaneId) -> E {
        assert!(is_keccak_lane(id), "{id:?} is not a Keccak lane");
        *self
            .values
            .get(&id)
            .unwrap_or_else(|| panic!("Keccak lane {id:?} is not read by stage {}", self.stage))
    }
}

/// Resolve a Keccak lane at the stage terminal from a precomputed [`KeccakResolver`].
#[must_use]
pub fn resolve_keccak(resolver: &KeccakResolver, id: LaneId) -> E {
    resolver.resolve(id)
}

/// Prove stage 1. `points.tau` is set; `tables` holds `w_2, w_{3,t}` (under `Gf2`) or the
/// Keccak prover tables (under `Fp`).
pub fn prove(
    manifest: &CommitmentManifest,
    tables: &ProverTables<'_>,
    points: &Points,
    prestage: &PreStageClaims,
    y_at_tau3: E,
    transcript: &mut Transcript,
) -> Result<(Stage1Proof, Stage1Output), StageError> {
    prove_with(manifest, tables, points, prestage, y_at_tau3, transcript, true)
}

/// [`prove`] without the prover's claim self-check (tests).
#[doc(hidden)]
pub fn prove_unchecked(
    manifest: &CommitmentManifest,
    tables: &ProverTables<'_>,
    points: &Points,
    prestage: &PreStageClaims,
    y_at_tau3: E,
    transcript: &mut Transcript,
) -> Result<(Stage1Proof, Stage1Output), StageError> {
    prove_with(manifest, tables, points, prestage, y_at_tau3, transcript, false)
}

fn prove_with(
    manifest: &CommitmentManifest,
    tables: &ProverTables<'_>,
    points: &Points,
    prestage: &PreStageClaims,
    y_at_tau3: E,
    transcript: &mut Transcript,
    check: bool,
) -> Result<(Stage1Proof, Stage1Output), StageError> {
    let spec = spec(manifest);
    let claims = claimed_sums(manifest, prestage, y_at_tau3);
    let run = if check {
        prove_stage(&spec, manifest, points, &claims, |id| tables.lane(id), transcript)?
    } else {
        prove_stage_unchecked(&spec, manifest, points, &claims, |id| tables.lane(id), transcript)?
    };
    let value = |id| run.value(&spec, id);
    let bits = 0..manifest.shape().w1_bits as u8;
    let (y1, y2) = match manifest.hash_field() {
        HashField::Gf2 => (
            bits.clone().map(|t| value(LaneId::Parity1(t))).collect(),
            bits.map(|t| value(LaneId::Column(Column::Parity2(t)))).collect(),
        ),
        HashField::Fp => (Vec::new(), Vec::new()),
    };
    let claims = Stage1Claims {
        y1,
        y2,
        keccak: keccak_claim_lanes(&spec).into_iter().map(value).collect(),
    };
    transcript.absorb_extensions(&terminal_site(STAGE), &claims.to_vec())?;
    Ok((
        Stage1Proof {
            rounds: run.messages,
            claims: claims.clone(),
        },
        Stage1Output {
            point: run.point,
            claims,
        },
    ))
}

/// The verifier's bridge weights at the stage-1 terminal `r^(1)`: `w_2(r^(1)_1)` for the
/// point `tau_2` and `w_{3,t}(r^(1)_2)` for `tau_3`, each one
/// `c_t` build and one sparse contraction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BridgeWeightsAt {
    /// `w_2(r^(1)_1)`.
    pub level2: E,
    /// `w_{3,t}(r^(1)_2)`, per bit.
    pub level3: Vec<E>,
}

impl BridgeWeightsAt {
    /// Evaluate both weights at the stage-1 terminal point `r1` (a main-domain point).
    #[must_use]
    pub fn evaluate(manifest: &CommitmentManifest, geom: &Geometry, matrices: &BridgeMatrices, tau: &[E], r1: &[E]) -> Self {
        let vars = |role| manifest.group(role).logical_vars;
        let (tau2, tau3) = (suffix(tau, vars(GroupRole::Level2)), suffix(tau, vars(GroupRole::Level3)));
        let (r1_1, r1_2) = (suffix(r1, vars(GroupRole::Level1)), suffix(r1, vars(GroupRole::Level2)));
        if manifest.has_lanes() {
            return Self {
                level2: level2_weight_at_for(manifest, &matrices.m2, &tau2, &r1_1),
                level3: level3_weights_at_for(manifest, &matrices.m3, &tau3, &r1_2),
            };
        }
        Self {
            level2: level2_weight_at(geom, &matrices.m2, &tau2, &r1_1),
            level3: level3_weights_at(geom, &matrices.m3, &tau3, &r1_2).into(),
        }
    }
}

/// The verifier's value of a stage-1 lane at the terminal.
fn resolve(weights: Option<&BridgeWeightsAt>, keccak: &KeccakResolver, claims: &Stage1Claims, id: LaneId) -> E {
    match id {
        LaneId::Parity1(t) => claims.y1[usize::from(t)],
        LaneId::Column(Column::Parity2(t)) => claims.y2[usize::from(t)],
        LaneId::Weight(WeightId::Level2) => weights.expect("Gf2 bridge weights").level2,
        LaneId::Weight(WeightId::Level3(t)) => weights.expect("Gf2 bridge weights").level3[usize::from(t)],
        other if is_keccak_lane(other) => resolve_keccak(keccak, other),
        other => panic!("{other:?} is not a stage-1 lane"),
    }
}

/// Verify stage 1.
#[allow(clippy::too_many_arguments)]
pub fn verify(
    manifest: &CommitmentManifest,
    matrices: &BridgeMatrices,
    points: &Points,
    prestage: &PreStageClaims,
    y_at_tau3: E,
    keccak_public: Option<KeccakPublicInputs<'_>>,
    proof: &Stage1Proof,
    transcript: &mut Transcript,
) -> Result<Stage1Output, StageError> {
    let spec = spec(manifest);
    let claims = claimed_sums(manifest, prestage, y_at_tau3);
    let y_len = Stage1Claims::y_len(manifest);
    if proof.claims.y1.len() != y_len
        || proof.claims.y2.len() != y_len
        || proof.claims.keccak.len() != Stage1Claims::keccak_len(manifest)
    {
        return Err(StageError::Shape("stage 1 claims".into()));
    }
    let replay = replay_stage(&spec, manifest, &claims, &proof.rounds, transcript)?;
    transcript.absorb_extensions(&terminal_site(STAGE), &proof.claims.to_vec())?;
    let weights = match manifest.hash_field() {
        HashField::Gf2 => {
            let geom = Geometry::new(manifest.records()).map_err(|e| StageError::Internal(format!("{e:?}")))?;
            Some(BridgeWeightsAt::evaluate(manifest, &geom, matrices, &points.tau, &replay.point))
        }
        HashField::Fp => None,
    };
    let keccak = timed_keccak_terminal(|| {
        KeccakResolver::new(manifest, &spec, &replay.point, &proof.claims.keccak, keccak_public)
    });
    check_terminal(&spec, manifest, points, &replay, |id| {
        resolve(weights.as_ref(), &keccak, &proof.claims, id)
    })?;
    Ok(Stage1Output {
        point: replay.point,
        claims: proof.claims.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_one_has_nineteen_bridge_instances_in_spec_order() {
        let manifest = CommitmentManifest::for_batch(8).unwrap();
        let spec = spec(&manifest);
        spec.validate().unwrap();
        assert_eq!(spec.gamma_count, 19);
        assert_eq!(spec.groups.len(), 4);
        let instances: usize = spec.groups.iter().map(|g| g.instances.len()).sum();
        assert_eq!(instances, 19);
        // No instance on the main domain; every term fits the degree with its equality factor.
        for group in &spec.groups {
            assert_ne!(group.domain, GroupRole::Main);
            let inner = usize::from(group.eq != EqPoint::None);
            for instance in &group.instances {
                for term in &instance.terms {
                    assert!(term.degree() + inner <= DEGREE);
                }
            }
        }
        assert_eq!(spec.lane_ids().len(), 6 + 6 + 1 + 6);
        // The general layout reduces to the pinned ML-DSA-44 indices.
        let gammas = Gammas::new(&manifest);
        for t in 0..W1_BITS {
            assert_eq!(gammas.bool_y1(t), gamma_bool_y1(t));
            assert_eq!(gammas.level2(t), gamma_level2(t));
            assert_eq!(gammas.bool_y2(t), gamma_bool_y2(t));
        }
        assert_eq!((gammas.level3(), gammas.count()), (GAMMA_LEVEL3, GAMMA_COUNT));
        assert_eq!(degree(&manifest), DEGREE);
    }

    /// ML-DSA-65 and ML-DSA-87: the same bridge instances over four bits, 13 coefficients.
    #[test]
    fn lane_sets_have_only_bridge_instances() {
        use agg_witness::params::Set;
        for set in [Set::S65, Set::S87] {
            let manifest = CommitmentManifest::for_set(set, 256, crate::manifest::Relation::HExact).unwrap();
            let spec = spec(&manifest);
            spec.validate().unwrap();
            assert_eq!((spec.gamma_count, spec.degree, degree(&manifest)), (13, 3, 3));
            for group in &spec.groups {
                assert_ne!(group.domain, GroupRole::Main);
                let inner = usize::from(group.eq != EqPoint::None);
                for instance in &group.instances {
                    for term in &instance.terms {
                        assert!(term.degree() + inner <= 3);
                    }
                }
            }
        }
    }

    /// `HashField::Fp`: no bridge and no lattice instance; the Keccak zero-checks alone, of
    /// degree 4.
    #[test]
    fn fp_stage_one_holds_only_the_keccak_groups() {
        let gf2 = CommitmentManifest::for_batch(8).unwrap();
        assert_eq!(gamma_count(&gf2), GAMMA_COUNT);

        let manifest = CommitmentManifest::for_relation_fp(8, crate::manifest::Relation::HExact).unwrap();
        let gammas = Gammas::new(&manifest);
        assert_eq!((gammas.keccak_offset(), gammas.keccak), (0, 719));
        assert_eq!(gamma_count(&manifest), 719);
        assert_eq!(degree(&manifest), FP_DEGREE);

        let spec = spec(&manifest);
        spec.validate().unwrap();
        assert_eq!((spec.gamma_count, spec.degree), (719, FP_DEGREE));
        for group in &spec.groups {
            assert_eq!(group.domain, GroupRole::KeccakA(0));
            assert!(matches!(group.eq, EqPoint::Tau | EqPoint::TauMasked(_)));
            let inner = usize::from(group.eq != EqPoint::None);
            for instance in &group.instances {
                for term in &instance.terms {
                    assert!(term.degree() + inner <= FP_DEGREE);
                }
            }
        }

        let dummy_pre = PreStageClaims {
            y2: Vec::new(),
            kappa2: Vec::new(),
            kappa3: E::zero(),
        };
        let sums = claimed_sums(&manifest, &dummy_pre, E::zero());
        assert_eq!(sums.len(), 719);
        assert!(sums.iter().all(|&c| c == E::zero()));
        assert_eq!(keccak_claim_lanes(&spec).len(), Stage1Claims::keccak_len(&manifest));
        assert_eq!(Stage1Claims::len(&manifest), Stage1Claims::keccak_len(&manifest));
    }
}
