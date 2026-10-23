//! Stage 2 (spec Section 7.4, degree 5): the level-1 definition (7.5) and the 95 range
//! leaves on the virtual tables `P_Z`.
//!
//! `gamma^(2)` order: `def_y1[t]` (0..6), `leaf16[Z, lo], leaf16[Z, hi]` per base-16
//! digit (6..86, lo before hi, digit order of spec Section 4.1), `leaf8[Z]` for `u_4, v_6`
//! (86, 87), `leaf4[Z]` for the 13 base-4 digits (88..101) in digit order.

use agg_witness::bridge::BridgeMatrices;
use agg_witness::padding::Geometry;
use agg_witness::params::W1_BITS;
use jolt_field::Zero;

use crate::claims::{Kappa1Claims, Stage1Claims, Stage2Claims};
use crate::kernel::{LeafPoly, Term};
use crate::manifest::{Column, CommitmentManifest, GroupRole};
use crate::range::{base16_digits, digits, digits_of_base, Digit};
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
/// Message degree.
pub const DEGREE: usize = 5;
/// Batching coefficients.
pub const GAMMA_COUNT: usize = 101;

/// Gamma index of `def_y1[t]`.
#[must_use]
pub const fn gamma_def_y1(t: usize) -> usize {
    t
}
/// Gamma indices `(lo, hi)` of the leaves of the `k`-th base-16 digit.
#[must_use]
pub const fn gamma_leaf16(k: usize) -> (usize, usize) {
    (W1_BITS + 2 * k, W1_BITS + 2 * k + 1)
}
/// Gamma index of the `k`-th base-8 leaf.
#[must_use]
pub const fn gamma_leaf8(k: usize) -> usize {
    W1_BITS + 80 + k
}
/// Gamma index of the `k`-th base-4 leaf.
#[must_use]
pub const fn gamma_leaf4(k: usize) -> usize {
    W1_BITS + 82 + k
}

/// The stage-2 proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2Proof {
    /// The 18 claims `kappa^(1)_{t,k}(r^(1)_1)` sent before `gamma^(2)`.
    pub kappa1: Kappa1Claims,
    /// Compressed degree-5 messages (5 elements each).
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

/// The leaf instances of a digit, as `(gamma, polynomial)`.
fn leaves(digit: &Digit, base16: &[Digit], base8: &[Digit], base4: &[Digit]) -> Vec<(usize, LeafPoly)> {
    let index = |list: &[Digit]| list.iter().position(|d| d == digit).expect("digit listed");
    match digit.base {
        16 => {
            let (lo, hi) = gamma_leaf16(index(base16));
            vec![(lo, LeafPoly::Q8), (hi, LeafPoly::Q16Hi)]
        }
        8 => vec![(gamma_leaf8(index(base8)), LeafPoly::Q8)],
        4 => vec![(gamma_leaf4(index(base4)), LeafPoly::Q4)],
        other => panic!("unsupported base {other}"),
    }
}

/// The instance table.
#[must_use]
pub fn spec(manifest: &CommitmentManifest) -> StageSpec {
    let lanes = 0..W1_BITS;
    let all = digits(manifest);
    let base16 = base16_digits(manifest);
    let base8 = digits_of_base(manifest, 8);
    let base4 = digits_of_base(manifest, 4);
    let mut groups = Vec::new();
    // def_y1[t] on the main domain with the weight w_1.
    groups.push(GroupSpec {
        domain: GroupRole::Main,
        eq: EqPoint::None,
        lanes: std::iter::once(LaneId::Weight(WeightId::Level1))
            .chain(lanes.clone().map(|t| LaneId::Column(Column::W1Bit(t as u8))))
            .collect(),
        instances: lanes
            .map(|t| InstanceSpec {
                gamma: gamma_def_y1(t),
                terms: vec![Term::Product(vec![0, 1 + t])],
            })
            .collect(),
    });
    // The leaves, one equality group per domain.
    for domain in GroupRole::ALL {
        let mut lane_ids = Vec::new();
        let mut instances = Vec::new();
        for digit in all.iter().filter(|d| d.group == domain) {
            let lane = lane_ids.len();
            lane_ids.push(LaneId::DigitSquare(digit.column));
            for (gamma, poly) in leaves(digit, &base16, &base8, &base4) {
                instances.push(InstanceSpec {
                    gamma,
                    terms: vec![Term::Leaf(lane, poly)],
                });
            }
        }
        groups.push(GroupSpec {
            domain,
            eq: EqPoint::R1,
            lanes: lane_ids,
            instances,
        });
    }
    StageSpec {
        stage: STAGE,
        degree: DEGREE,
        gamma_count: GAMMA_COUNT,
        groups,
    }
}

/// The claimed sums in gamma order: (7.5) for `def_y1`, the stage-1 products for the
/// base-16 leaves, zero for the base-8 and base-4 leaves.
#[must_use]
pub fn claimed_sums(stage1: &Stage1Claims, kappa1: &Kappa1Claims) -> Vec<E> {
    let mut claims = vec![E::zero(); GAMMA_COUNT];
    for t in 0..W1_BITS {
        let kappa = kappa1.value(t);
        claims[gamma_def_y1(t)] = stage1.y1[t] + kappa + kappa;
    }
    for (k, pair) in stage1.products.iter().enumerate() {
        let (lo, hi) = gamma_leaf16(k);
        claims[lo] = pair[0];
        claims[hi] = pair[1];
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
        squares: digits(manifest)
            .iter()
            .map(|d| value(LaneId::DigitSquare(d.column)))
            .collect(),
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
    if proof.claims.squares.len() != all.len() {
        return Err(StageError::Shape("stage 2 squares".into()));
    }
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
    fn stage_two_has_one_hundred_and_one_instances_in_spec_order() {
        let manifest = CommitmentManifest::for_batch(8).unwrap();
        let spec = spec(&manifest);
        spec.validate().unwrap();
        assert_eq!(spec.groups.len(), 5);
        let instances: usize = spec.groups.iter().map(|g| g.instances.len()).sum();
        assert_eq!(instances, 101);
        assert_eq!(gamma_leaf16(39), (84, 85));
        assert_eq!(gamma_leaf8(1), 87);
        assert_eq!(gamma_leaf4(12), 100);
        // u_4 is the first base-8 leaf, z_4 the first base-4 leaf.
        let main = &spec.groups[1];
        let u4 = main.lane(LaneId::DigitSquare(Column::Digit {
            value: crate::manifest::DigitValue::U,
            digit: 4,
        }));
        let leaf8 = main
            .instances
            .iter()
            .find(|i| i.gamma == gamma_leaf8(0))
            .unwrap();
        assert_eq!(leaf8.terms, vec![Term::Leaf(u4, LeafPoly::Q8)]);
        let z4 = main.lane(LaneId::DigitSquare(Column::Digit {
            value: crate::manifest::DigitValue::Z,
            digit: 4,
        }));
        let leaf4 = main
            .instances
            .iter()
            .find(|i| i.gamma == gamma_leaf4(0))
            .unwrap();
        assert_eq!(leaf4.terms, vec![Term::Leaf(z4, LeafPoly::Q4)]);
        for group in &spec.groups {
            let inner = usize::from(group.eq != EqPoint::None);
            for instance in &group.instances {
                for term in &instance.terms {
                    assert!(term.degree() + inner <= DEGREE);
                }
            }
        }
    }
}
