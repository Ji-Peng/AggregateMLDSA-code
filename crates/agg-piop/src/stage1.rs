//! Stage 1: bridge Booleanity and levels 2 and 3, w1Encode Booleanity and, for ML-DSA-44,
//! the bit-range identity. The lookup into `T_16` proves the other ranges.
//!
//! `gamma^(1)` order with `w` bits of `w_1'` (6 for ML-DSA-44, 4 otherwise) and `k_L`
//! polynomial lanes per bit (1 for ML-DSA-44, `k` otherwise): `bool_y1[t]` (`0..w`),
//! `level2[t]` (`w..2w`), `level3` (`2w`), `bool_y2[t]` (`2w+1..3w+1`), `bool_b[t, rho]`
//! (`3w+1 + t k_L + rho`), then `range_w1` for ML-DSA-44 only (`w_1' < 44` needs the
//! range identity there; four bits give `w_1' < 16 = m_w` by themselves). ML-DSA-44 has 26
//! instances and degree 4 (the range identity is cubic), ML-DSA-65 37 and ML-DSA-87 45, both
//! of degree 3.

use agg_witness::bridge::BridgeMatrices;
use agg_witness::padding::Geometry;
use agg_witness::params::W1_BITS;
use jolt_field::Zero;

use crate::claims::{bit_columns, PreStageClaims, Stage1Claims};
use crate::kernel::Term;
use crate::manifest::{Column, CommitmentManifest, GroupRole};
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
/// Message degree of ML-DSA-44 ([`degree`]).
pub const DEGREE: usize = 4;
/// Batching coefficients of ML-DSA-44 ([`gamma_count`]).
pub const GAMMA_COUNT: usize = 26;

/// True when the set checks `w_1' < m_w` by the cubic range identity (six bits).
fn has_range_w1(manifest: &CommitmentManifest) -> bool {
    manifest.shape().w1_bits == W1_BITS
}

/// The message degree: 4 with the range identity (ML-DSA-44), 3 otherwise.
#[must_use]
pub fn degree(manifest: &CommitmentManifest) -> usize {
    if has_range_w1(manifest) {
        DEGREE
    } else {
        3
    }
}

/// The `gamma^(1)` layout of a parameter set: `w` bits of `w_1'` in `lanes` polynomial
/// lanes each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gammas {
    /// Bits of `w_1'`.
    pub bits: usize,
    /// Polynomial lanes per bit.
    pub lanes: usize,
    /// The range identity of ML-DSA-44.
    pub range_w1: bool,
}

impl Gammas {
    /// The layout of `manifest`'s parameter set.
    #[must_use]
    pub fn new(manifest: &CommitmentManifest) -> Self {
        Self {
            bits: manifest.shape().w1_bits,
            lanes: manifest.shape().lanes_k,
            range_w1: has_range_w1(manifest),
        }
    }

    /// Gamma index of `bool_y1[t]`.
    #[must_use]
    pub const fn bool_y1(self, t: usize) -> usize {
        t
    }

    /// Gamma index of `level2[t]`.
    #[must_use]
    pub const fn level2(self, t: usize) -> usize {
        self.bits + t
    }

    /// Gamma index of `level3`.
    #[must_use]
    pub const fn level3(self) -> usize {
        2 * self.bits
    }

    /// Gamma index of `bool_y2[t]`.
    #[must_use]
    pub const fn bool_y2(self, t: usize) -> usize {
        2 * self.bits + 1 + t
    }

    /// Gamma index of `bool_b[t, rho]` (bit-major).
    #[must_use]
    pub const fn bool_b(self, t: usize, lane: usize) -> usize {
        3 * self.bits + 1 + t * self.lanes + lane
    }

    /// Gamma index of `range_w1` (ML-DSA-44 only).
    #[must_use]
    pub const fn range_w1(self) -> usize {
        3 * self.bits + 1 + self.bits * self.lanes
    }

    /// Batching coefficients.
    #[must_use]
    pub const fn count(self) -> usize {
        self.range_w1() + if self.range_w1 { 1 } else { 0 }
    }
}

/// Batching coefficients of `manifest`'s parameter set (26, 37 or 45).
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
/// Gamma index of `bool_b[t]` (ML-DSA-44).
#[must_use]
pub const fn gamma_bool_b(t: usize) -> usize {
    3 * W1_BITS + 1 + t
}
/// Gamma index of `range_w1` (ML-DSA-44).
pub const GAMMA_RANGE_W1: usize = 4 * W1_BITS + 1;

/// The stage-1 proof: the round messages and the 18 terminal claims.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1Proof {
    /// Compressed degree-4 messages (4 elements each).
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
    let bits = bit_columns(manifest);
    let mut groups = Vec::new();
    // One equality group per domain, holding the Booleanity checks.
    for domain in [GroupRole::Main, GroupRole::Level1, GroupRole::Level2] {
        let mut lane_ids = Vec::new();
        let mut instances = Vec::new();
        match domain {
            GroupRole::Main => {
                lane_ids.extend(bits.iter().map(|&c| LaneId::Column(c)));
                instances.extend(bits.iter().enumerate().map(|(lane, &column)| {
                    let Column::W1Bit(t, rho) = column else { unreachable!("bit columns") };
                    InstanceSpec::new(gammas.bool_b(usize::from(t), usize::from(rho)), vec![Term::Bool(lane)])
                }));
                if gammas.range_w1 {
                    instances.push(InstanceSpec::new(gammas.range_w1(), vec![Term::RangeW1([5, 4, 3, 2])]));
                }
            }
            GroupRole::Level1 => {
                lane_ids.extend(lanes.clone().map(|t| LaneId::Parity1(t as u8)));
                instances.extend(
                    lanes
                        .clone()
                        .map(|t| InstanceSpec::new(gammas.bool_y1(t), vec![Term::Bool(t)])),
                );
            }
            _ => {
                lane_ids.extend(lanes.clone().map(|t| LaneId::Column(Column::Parity2(t as u8))));
                instances.extend(
                    lanes
                        .clone()
                        .map(|t| InstanceSpec::new(gammas.bool_y2(t), vec![Term::Bool(t)])),
                );
            }
        }
        groups.push(GroupSpec {
            domain,
            eq: EqPoint::Tau,
            lanes: lane_ids,
            instances,
        });
    }
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
    StageSpec {
        stage: STAGE,
        degree: degree(manifest),
        gamma_count: gammas.count(),
        groups,
    }
}

/// The claimed sums in gamma order: zero for the zero-checks, the level-2 and level-3 sums
/// of the pre-stage claims for the level instances.
#[must_use]
pub fn claimed_sums(manifest: &CommitmentManifest, prestage: &PreStageClaims, y_at_tau3: E) -> Vec<E> {
    let gammas = Gammas::new(manifest);
    let mut claims = vec![E::zero(); gammas.count()];
    for t in 0..gammas.bits {
        let kappa2 = prestage.kappa2_value(t);
        claims[gammas.level2(t)] = prestage.y2[t] + kappa2 + kappa2;
    }
    claims[gammas.level3()] =
        agg_bridge::final_vector::level3_claimed_sum(y_at_tau3, prestage.kappa3_value());
    claims
}

/// Prove stage 1. `points.tau` is set; `tables` holds `w_2, w_{3,t}`.
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
    let claims = Stage1Claims {
        y1: bits.clone().map(|t| value(LaneId::Parity1(t))).collect(),
        y2: bits.map(|t| value(LaneId::Column(Column::Parity2(t)))).collect(),
        b: bit_columns(manifest).into_iter().map(|c| value(LaneId::Column(c))).collect(),
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
fn resolve(manifest: &CommitmentManifest, weights: &BridgeWeightsAt, claims: &Stage1Claims, id: LaneId) -> E {
    match id {
        LaneId::Parity1(t) => claims.y1[usize::from(t)],
        LaneId::Column(Column::Parity2(t)) => claims.y2[usize::from(t)],
        LaneId::Column(Column::W1Bit(t, lane)) => {
            claims.b[usize::from(t) * manifest.shape().lanes_k + usize::from(lane)]
        }
        LaneId::Weight(WeightId::Level2) => weights.level2,
        LaneId::Weight(WeightId::Level3(t)) => weights.level3[usize::from(t)],
        other => panic!("{other:?} is not a stage-1 lane"),
    }
}

/// Verify stage 1.
pub fn verify(
    manifest: &CommitmentManifest,
    matrices: &BridgeMatrices,
    points: &Points,
    prestage: &PreStageClaims,
    y_at_tau3: E,
    proof: &Stage1Proof,
    transcript: &mut Transcript,
) -> Result<Stage1Output, StageError> {
    let spec = spec(manifest);
    let claims = claimed_sums(manifest, prestage, y_at_tau3);
    if proof.claims.y1.len() != manifest.shape().w1_bits
        || proof.claims.y2.len() != manifest.shape().w1_bits
        || proof.claims.b.len() != bit_columns(manifest).len()
    {
        return Err(StageError::Shape("stage 1 claims".into()));
    }
    let replay = replay_stage(&spec, manifest, &claims, &proof.rounds, transcript)?;
    transcript.absorb_extensions(&terminal_site(STAGE), &proof.claims.to_vec())?;
    let geom = Geometry::new(manifest.records()).map_err(|e| StageError::Internal(format!("{e:?}")))?;
    let weights = BridgeWeightsAt::evaluate(manifest, &geom, matrices, &points.tau, &replay.point);
    check_terminal(&spec, manifest, points, &replay, |id| {
        resolve(manifest, &weights, &proof.claims, id)
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
    fn stage_one_has_twenty_six_instances_in_spec_order() {
        let manifest = CommitmentManifest::for_batch(8).unwrap();
        let spec = spec(&manifest);
        spec.validate().unwrap();
        assert_eq!(spec.gamma_count, 26);
        assert_eq!(spec.groups.len(), 5);
        let instances: usize = spec.groups.iter().map(|g| g.instances.len()).sum();
        assert_eq!(instances, 26);
        assert_eq!(GAMMA_RANGE_W1, 25);
        // Every term fits the degree with its equality factor.
        for group in &spec.groups {
            let inner = usize::from(group.eq != EqPoint::None);
            for instance in &group.instances {
                for term in &instance.terms {
                    assert!(term.degree() + inner <= DEGREE);
                }
            }
        }
        assert_eq!(spec.lane_ids().len(), 6 + 6 + 6 + 1 + 6);
        // The general layout reduces to the pinned ML-DSA-44 indices.
        let gammas = Gammas::new(&manifest);
        for t in 0..W1_BITS {
            assert_eq!(gammas.bool_y1(t), gamma_bool_y1(t));
            assert_eq!(gammas.level2(t), gamma_level2(t));
            assert_eq!(gammas.bool_y2(t), gamma_bool_y2(t));
            assert_eq!(gammas.bool_b(t, 0), gamma_bool_b(t));
        }
        assert_eq!((gammas.level3(), gammas.range_w1(), gammas.count()), (GAMMA_LEVEL3, GAMMA_RANGE_W1, GAMMA_COUNT));
        assert_eq!(degree(&manifest), DEGREE);
    }

    /// ML-DSA-65 and ML-DSA-87: one Booleanity check per bit and lane, no range identity,
    /// degree 3; 37 and 45 instances.
    #[test]
    fn lane_sets_check_every_bit_lane_and_drop_the_range_identity() {
        use agg_witness::params::Set;
        for (set, k, count) in [(Set::S65, 6, 37), (Set::S87, 8, 45)] {
            let manifest = CommitmentManifest::for_set(set, 256, crate::manifest::Relation::HExact).unwrap();
            let spec = spec(&manifest);
            spec.validate().unwrap();
            assert_eq!((spec.gamma_count, spec.degree, degree(&manifest)), (count, 3, 3));
            let main = &spec.groups[0];
            assert_eq!(main.lanes.len(), 4 * k);
            assert_eq!(main.instances.len(), 4 * k);
            let gammas = Gammas::new(&manifest);
            assert_eq!(gammas.bool_b(3, k - 1), count - 1);
            assert_eq!(main.lane(LaneId::Column(Column::W1Bit(2, 1))), 2 * k + 1);
            for group in &spec.groups {
                let inner = usize::from(group.eq != EqPoint::None);
                for instance in &group.instances {
                    for term in &instance.terms {
                        assert!(term.degree() + inner <= 3);
                        assert!(!matches!(term, Term::RangeW1(_)));
                    }
                }
            }
        }
    }
}
