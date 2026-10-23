//! Stage 1 (spec Section 7.3, degree 4): bridge Booleanity and levels 2 and 3, w1Encode
//! Booleanity and the bit-range identity (7.4), and the 40 `b16_prod` products.
//!
//! `gamma^(1)` order: `bool_y1[t]` (0..6), `level2[t]` (6..12), `level3` (12),
//! `bool_y2[t]` (13..19), `bool_b[t]` (19..25), `range_w1` (25), `b16_prod[Z]` (26..66) in
//! the digit order of spec Section 4.1.

use agg_witness::bridge::BridgeMatrices;
use agg_witness::padding::Geometry;
use agg_witness::params::W1_BITS;
use jolt_field::Zero;

use crate::claims::{PreStageClaims, Stage1Claims};
use crate::kernel::Term;
use crate::manifest::{Column, CommitmentManifest, GroupRole};
use crate::range::base16_digits;
use crate::stage::{
    check_terminal, prove_stage, prove_stage_unchecked, replay_stage, suffix, EqPoint, GroupSpec, InstanceSpec, LaneId,
    Points, StageError, StageSpec, WeightId,
};
use crate::tables::ProverTables;
use crate::transcript::{terminal_site, Transcript};
use crate::weights::{level2_weight_at, level3_weights_at};
use crate::OpeningField as E;

/// Stage number.
pub const STAGE: usize = 1;
/// Message degree.
pub const DEGREE: usize = 4;
/// Batching coefficients.
pub const GAMMA_COUNT: usize = 66;

/// Gamma index of `bool_y1[t]`.
#[must_use]
pub const fn gamma_bool_y1(t: usize) -> usize {
    t
}
/// Gamma index of `level2[t]`.
#[must_use]
pub const fn gamma_level2(t: usize) -> usize {
    W1_BITS + t
}
/// Gamma index of `level3`.
pub const GAMMA_LEVEL3: usize = 2 * W1_BITS;
/// Gamma index of `bool_y2[t]`.
#[must_use]
pub const fn gamma_bool_y2(t: usize) -> usize {
    2 * W1_BITS + 1 + t
}
/// Gamma index of `bool_b[t]`.
#[must_use]
pub const fn gamma_bool_b(t: usize) -> usize {
    3 * W1_BITS + 1 + t
}
/// Gamma index of `range_w1`.
pub const GAMMA_RANGE_W1: usize = 4 * W1_BITS + 1;
/// Gamma index of `b16_prod[k]` for the `k`-th base-16 digit.
#[must_use]
pub const fn gamma_b16_prod(k: usize) -> usize {
    4 * W1_BITS + 2 + k
}

/// The stage-1 proof: the round messages and the 98 terminal claims.
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
    let lanes = 0..W1_BITS;
    let base16 = base16_digits(manifest);
    let mut groups = Vec::new();
    // One equality group per domain, holding the Booleanity checks and the products.
    for domain in GroupRole::ALL {
        let mut lane_ids = Vec::new();
        let mut instances = Vec::new();
        match domain {
            GroupRole::Main => {
                lane_ids.extend(lanes.clone().map(|t| LaneId::Column(Column::W1Bit(t as u8))));
                instances.extend(lanes.clone().map(|t| InstanceSpec {
                    gamma: gamma_bool_b(t),
                    terms: vec![Term::Bool(t)],
                }));
                instances.push(InstanceSpec {
                    gamma: GAMMA_RANGE_W1,
                    terms: vec![Term::RangeW1([5, 4, 3, 2])],
                });
            }
            GroupRole::Level1 => {
                lane_ids.extend(lanes.clone().map(|t| LaneId::Parity1(t as u8)));
                instances.extend(lanes.clone().map(|t| InstanceSpec {
                    gamma: gamma_bool_y1(t),
                    terms: vec![Term::Bool(t)],
                }));
            }
            GroupRole::Level2 => {
                lane_ids.extend(lanes.clone().map(|t| LaneId::Column(Column::Parity2(t as u8))));
                instances.extend(lanes.clone().map(|t| InstanceSpec {
                    gamma: gamma_bool_y2(t),
                    terms: vec![Term::Bool(t)],
                }));
            }
            GroupRole::Level3 => {}
        }
        for (k, digit) in base16.iter().enumerate() {
            if digit.group != domain {
                continue;
            }
            let lo = lane_ids.len();
            lane_ids.push(LaneId::DigitLo(digit.column));
            lane_ids.push(LaneId::DigitHi(digit.column));
            instances.push(InstanceSpec {
                gamma: gamma_b16_prod(k),
                terms: vec![Term::Product(vec![lo, lo + 1])],
            });
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
            .map(|t| InstanceSpec {
                gamma: gamma_level2(t),
                terms: vec![Term::Product(vec![0, 1 + t])],
            })
            .collect(),
    });
    // level3 on D_2 summed over the six lanes with the weights w_{3,t}.
    groups.push(GroupSpec {
        domain: GroupRole::Level2,
        eq: EqPoint::None,
        lanes: lanes
            .clone()
            .map(|t| LaneId::Weight(WeightId::Level3(t as u8)))
            .chain(lanes.clone().map(|t| LaneId::Column(Column::Parity2(t as u8))))
            .collect(),
        instances: vec![InstanceSpec {
            gamma: GAMMA_LEVEL3,
            terms: lanes
                .clone()
                .map(|t| Term::Product(vec![t, W1_BITS + t]))
                .collect(),
        }],
    });
    StageSpec {
        stage: STAGE,
        degree: DEGREE,
        gamma_count: GAMMA_COUNT,
        groups,
    }
}

/// The claimed sums in gamma order: zero for the zero-checks, (7.2) and (7.3) for the
/// level instances.
#[must_use]
pub fn claimed_sums(prestage: &PreStageClaims, y_at_tau3: E) -> Vec<E> {
    let mut claims = vec![E::zero(); GAMMA_COUNT];
    for t in 0..W1_BITS {
        let kappa2 = prestage.kappa2_value(t);
        claims[gamma_level2(t)] = prestage.y2[t] + kappa2 + kappa2;
    }
    let kappa3 = prestage.kappa3_value();
    claims[GAMMA_LEVEL3] = y_at_tau3 + kappa3 + kappa3;
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
    let claims = claimed_sums(prestage, y_at_tau3);
    let run = if check {
        prove_stage(&spec, manifest, points, &claims, |id| tables.lane(id), transcript)?
    } else {
        prove_stage_unchecked(&spec, manifest, points, &claims, |id| tables.lane(id), transcript)?
    };
    let value = |id| run.value(&spec, id);
    let claims = Stage1Claims {
        y1: std::array::from_fn(|t| value(LaneId::Parity1(t as u8))),
        y2: std::array::from_fn(|t| value(LaneId::Column(Column::Parity2(t as u8)))),
        b: std::array::from_fn(|t| value(LaneId::Column(Column::W1Bit(t as u8)))),
        products: base16_digits(manifest)
            .iter()
            .map(|d| [value(LaneId::DigitLo(d.column)), value(LaneId::DigitHi(d.column))])
            .collect(),
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

/// The verifier's value of a stage-1 lane at the terminal.
#[allow(clippy::too_many_arguments)]
fn resolve(
    manifest: &CommitmentManifest,
    geom: &Geometry,
    matrices: &BridgeMatrices,
    tau: &[E],
    point: &[E],
    claims: &Stage1Claims,
    products: &[(Column, usize)],
    id: LaneId,
) -> E {
    match id {
        LaneId::Parity1(t) => claims.y1[usize::from(t)],
        LaneId::Column(Column::Parity2(t)) => claims.y2[usize::from(t)],
        LaneId::Column(Column::W1Bit(t)) => claims.b[usize::from(t)],
        LaneId::DigitLo(c) | LaneId::DigitHi(c) => {
            let k = products
                .iter()
                .find(|(column, _)| *column == c)
                .map(|(_, k)| *k)
                .expect("a base-16 digit");
            claims.products[k][usize::from(matches!(id, LaneId::DigitHi(_)))]
        }
        LaneId::Weight(WeightId::Level2) => {
            let tau2 = suffix(tau, manifest.group(GroupRole::Level2).logical_vars);
            let r1_1 = suffix(point, manifest.group(GroupRole::Level1).logical_vars);
            level2_weight_at(geom, &matrices.m2, &tau2, &r1_1)
        }
        LaneId::Weight(WeightId::Level3(t)) => {
            let tau3 = suffix(tau, manifest.group(GroupRole::Level3).logical_vars);
            let r1_2 = suffix(point, manifest.group(GroupRole::Level2).logical_vars);
            level3_weights_at(geom, &matrices.m3, &tau3, &r1_2)[usize::from(t)]
        }
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
    let claims = claimed_sums(prestage, y_at_tau3);
    let replay = replay_stage(&spec, manifest, &claims, &proof.rounds, transcript)?;
    if proof.claims.products.len() != 40 {
        return Err(StageError::Shape("stage 1 products".into()));
    }
    transcript.absorb_extensions(&terminal_site(STAGE), &proof.claims.to_vec())?;
    let geom = Geometry::new(manifest.records()).map_err(|e| StageError::Internal(format!("{e:?}")))?;
    let products: Vec<(Column, usize)> = base16_digits(manifest)
        .iter()
        .enumerate()
        .map(|(k, d)| (d.column, k))
        .collect();
    check_terminal(&spec, manifest, points, &replay, |id| {
        resolve(
            manifest,
            &geom,
            matrices,
            &points.tau,
            &replay.point,
            &proof.claims,
            &products,
            id,
        )
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
    fn stage_one_has_sixty_six_instances_in_spec_order() {
        let manifest = CommitmentManifest::for_batch(8).unwrap();
        let spec = spec(&manifest);
        spec.validate().unwrap();
        assert_eq!(spec.gamma_count, 66);
        assert_eq!(spec.groups.len(), 6);
        let instances: usize = spec.groups.iter().map(|g| g.instances.len()).sum();
        assert_eq!(instances, 66);
        assert_eq!(gamma_b16_prod(39), 65);
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
        assert_eq!(spec.lane_ids().len(), 6 + 6 + 6 + 80 + 1 + 6);
    }
}
