//! Stage 2 (spec Section 7.4, degree 4 in relation H-u): the level-1 definition (7.5) and the
//! two range leaves of the small digits on the virtual tables `P_Z` (lookup layout L16: every
//! other range is the lookup into `T_16`).
//!
//! `gamma^(2)` order: `def_y1[t]` (0..6), `leaf4[Z_hi]` (6), `leaf6[U_hi]` (7). The leaf of the
//! six-valued `U_hi` (`Q_6`, degree 3 in `P_Z`) sets the stage degree `1 + 3 = 4`; relaxed H's
//! `leaf8` had degree 5.

use agg_witness::bridge::BridgeMatrices;
use agg_witness::padding::Geometry;
use agg_witness::params::W1_BITS;
use jolt_field::Zero;

use crate::claims::{Kappa1Claims, Stage1Claims, Stage2Claims};
use crate::kernel::{LeafPoly, Term};
use crate::manifest::{Column, CommitmentManifest, GroupRole};
use crate::range::{digits, Digit};
use crate::stage::{
    check_terminal, prove_stage, prove_stage_unchecked, replay_stage, suffix, EqPoint, GroupSpec, InstanceSpec, LaneId,
    Points, StageError, StageSpec, WeightId,
};
use crate::stage1::Stage1Output;
use crate::tables::ProverTables;
use crate::transcript::{terminal_site, Transcript, KAPPA1_SITE};
use crate::weights::level1_weight_at;
use crate::OpeningField as E;

/// Stage number.
pub const STAGE: usize = 2;
/// Message degree (`leaf6[U_hi]`: the `eq` factor times `Q_6(P_Z)`).
pub const DEGREE: usize = 4;
/// Batching coefficients.
pub const GAMMA_COUNT: usize = W1_BITS + 2;

/// Gamma index of `def_y1[t]`.
#[must_use]
pub const fn gamma_def_y1(t: usize) -> usize {
    t
}
/// Gamma index of the leaf of the `k`-th leaf-checked digit (`Z_hi`, then `U_hi`).
#[must_use]
pub const fn gamma_leaf(k: usize) -> usize {
    W1_BITS + k
}

/// The stage-2 proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2Proof {
    /// The 6 claims `K^(1)_t(r^(1)_1)` sent before `gamma^(2)`.
    pub kappa1: Kappa1Claims,
    /// Compressed degree-4 messages (4 elements each).
    pub rounds: Vec<Vec<E>>,
    /// Terminal claims.
    pub claims: Stage2Claims,
}

/// What stage 3 needs from stage 2.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2Output {
    /// `r^(2)`.
    pub point: Vec<E>,
    /// The level-1 carry claims.
    pub kappa1: Kappa1Claims,
    /// Terminal claims.
    pub claims: Stage2Claims,
}

/// The leaf polynomial of a leaf-checked digit.
fn leaf_poly(digit: &Digit) -> LeafPoly {
    match digit.base {
        6 => LeafPoly::Q6,
        4 => LeafPoly::Q4,
        other => panic!("unsupported leaf base {other}"),
    }
}

/// The instance table.
#[must_use]
pub fn spec(manifest: &CommitmentManifest) -> StageSpec {
    let lanes = 0..W1_BITS;
    let all = digits(manifest);
    let mut groups = Vec::new();
    // def_y1[t] on the main domain with the weight w_1.
    groups.push(GroupSpec {
        domain: GroupRole::Main,
        eq: EqPoint::None,
        lanes: std::iter::once(LaneId::Weight(WeightId::Level1))
            .chain(lanes.clone().map(|t| LaneId::Column(Column::W1Bit(t as u8))))
            .collect(),
        instances: lanes
            .map(|t| InstanceSpec::new(gamma_def_y1(t), vec![Term::Product(vec![0, 1 + t])]))
            .collect(),
    });
    // The two leaves on the main domain.
    groups.push(GroupSpec {
        domain: GroupRole::Main,
        eq: EqPoint::R1,
        lanes: all.iter().map(|d| LaneId::DigitSquare(d.column)).collect(),
        instances: all
            .iter()
            .enumerate()
            .map(|(k, digit)| InstanceSpec::new(gamma_leaf(k), vec![Term::Leaf(k, leaf_poly(digit))]))
            .collect(),
    });
    StageSpec {
        stage: STAGE,
        degree: DEGREE,
        gamma_count: GAMMA_COUNT,
        groups,
    }
}

/// The claimed sums in gamma order: (7.5) for `def_y1`, zero for the two leaves.
#[must_use]
pub fn claimed_sums(stage1: &Stage1Claims, kappa1: &Kappa1Claims) -> Vec<E> {
    let mut claims = vec![E::zero(); GAMMA_COUNT];
    for t in 0..W1_BITS {
        let kappa = kappa1.value(t);
        claims[gamma_def_y1(t)] = stage1.y1[t] + kappa + kappa;
    }
    claims
}

/// Prove stage 2. `points.tau, points.r1` are set; `tables` holds `w_1`.
pub fn prove(
    manifest: &CommitmentManifest,
    tables: &ProverTables<'_>,
    points: &Points,
    stage1: &Stage1Output,
    transcript: &mut Transcript,
) -> Result<(Stage2Proof, Stage2Output), StageError> {
    prove_with(manifest, tables, points, stage1, transcript, true)
}

/// [`prove`] without the prover's claim self-check (tests).
#[doc(hidden)]
pub fn prove_unchecked(
    manifest: &CommitmentManifest,
    tables: &ProverTables<'_>,
    points: &Points,
    stage1: &Stage1Output,
    transcript: &mut Transcript,
) -> Result<(Stage2Proof, Stage2Output), StageError> {
    prove_with(manifest, tables, points, stage1, transcript, false)
}

fn prove_with(
    manifest: &CommitmentManifest,
    tables: &ProverTables<'_>,
    points: &Points,
    stage1: &Stage1Output,
    transcript: &mut Transcript,
    check: bool,
) -> Result<(Stage2Proof, Stage2Output), StageError> {
    let kappa1 = Kappa1Claims::evaluate(manifest, tables.tables, &stage1.point);
    transcript.absorb_extensions(KAPPA1_SITE, &kappa1.to_vec())?;
    let spec = spec(manifest);
    let claims = claimed_sums(&stage1.claims, &kappa1);
    let run = if check {
        prove_stage(&spec, manifest, points, &claims, |id| tables.lane(id), transcript)?
    } else {
        prove_stage_unchecked(&spec, manifest, points, &claims, |id| tables.lane(id), transcript)?
    };
    let value = |id| run.value(&spec, id);
    let claims = Stage2Claims {
        b: std::array::from_fn(|t| value(LaneId::Column(Column::W1Bit(t as u8)))),
        squares: {
            let all = digits(manifest);
            [0, 1].map(|k| value(LaneId::DigitSquare(all[k].column)))
        },
    };
    transcript.absorb_extensions(&terminal_site(STAGE), &claims.to_vec())?;
    Ok((
        Stage2Proof {
            kappa1: kappa1.clone(),
            rounds: run.messages,
            claims: claims.clone(),
        },
        Stage2Output {
            point: run.point,
            kappa1,
            claims,
        },
    ))
}

/// Verify stage 2.
pub fn verify(
    manifest: &CommitmentManifest,
    matrices: &BridgeMatrices,
    points: &Points,
    stage1: &Stage1Output,
    proof: &Stage2Proof,
    transcript: &mut Transcript,
) -> Result<Stage2Output, StageError> {
    transcript.absorb_extensions(KAPPA1_SITE, &proof.kappa1.to_vec())?;
    let spec = spec(manifest);
    let claims = claimed_sums(&stage1.claims, &proof.kappa1);
    let replay = replay_stage(&spec, manifest, &claims, &proof.rounds, transcript)?;
    let all = digits(manifest);
    transcript.absorb_extensions(&terminal_site(STAGE), &proof.claims.to_vec())?;
    let geom = Geometry::new(manifest.records()).map_err(|e| StageError::Internal(format!("{e:?}")))?;
    let r1_1 = suffix(&stage1.point, manifest.group(GroupRole::Level1).logical_vars);
    check_terminal(&spec, manifest, points, &replay, |id| match id {
        LaneId::Column(Column::W1Bit(t)) => proof.claims.b[usize::from(t)],
        LaneId::DigitSquare(c) => {
            let k = all.iter().position(|d| d.column == c).expect("a digit");
            proof.claims.squares[k]
        }
        LaneId::Weight(WeightId::Level1) => {
            level1_weight_at(&geom, &matrices.m1, &r1_1, &replay.point)
        }
        other => panic!("{other:?} is not a stage-2 lane"),
    })?;
    Ok(Stage2Output {
        point: replay.point,
        kappa1: proof.kappa1.clone(),
        claims: proof.claims.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_two_has_eight_instances_in_spec_order() {
        let manifest = CommitmentManifest::for_batch(8).unwrap();
        let spec = spec(&manifest);
        spec.validate().unwrap();
        assert_eq!(spec.groups.len(), 2);
        let instances: usize = spec.groups.iter().map(|g| g.instances.len()).sum();
        assert_eq!(instances, 8);
        let leaves = &spec.groups[1];
        let z_hi = leaves.lane(LaneId::DigitSquare(Column::Z_HI));
        let u_hi = leaves.lane(LaneId::DigitSquare(Column::U_HI));
        assert_eq!(leaves.instances[0].gamma, gamma_leaf(0));
        assert_eq!(leaves.instances[0].terms, vec![Term::Leaf(z_hi, LeafPoly::Q4)]);
        assert_eq!(leaves.instances[1].terms, vec![Term::Leaf(u_hi, LeafPoly::Q6)]);
        let mut max = 0;
        for group in &spec.groups {
            let inner = usize::from(group.eq != EqPoint::None);
            for instance in &group.instances {
                for term in &instance.terms {
                    assert!(term.degree() + inner <= DEGREE);
                    max = max.max(term.degree() + inner);
                }
            }
        }
        // The degree is tight: leaf6[U_hi] reaches it.
        assert_eq!(max, DEGREE);
        assert_eq!(DEGREE, crate::transcript::STAGE_DEGREES[STAGE - 1]);
    }
}
