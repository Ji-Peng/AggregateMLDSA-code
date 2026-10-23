//! Stage 3 (spec Section 7.5, degree 3): the ring identity (7.6) with `(A z)` pulled back
//! through the public matrices, the 55 range links, and the 62 evaluation reductions to
//! `r^(3)`.
//!
//! `gamma^(3)` order: `ring` (0), `link[Z]` (1..56) in digit order, then the reductions:
//! `b_t` from `r^(1)` (56..62), `b_t` from `r^(2)` (62..68), `y^(2)_t` from `tau_2`
//! (68..74), `y^(2)_t` from `r^(1)_2` (74..80), `kappa^(2)_{t,k}` from `tau_2` (80..98,
//! `t`-major), `kappa^(3)_k` from `tau_3` (98..100), `kappa^(1)_{t,k}` from `r^(1)_1`
//! (100..118).
//!
//! The ring instance is one instance with two summands on two groups: `w_A z` (weight
//! lane, no equality factor) and `-eq(r^(2), x)(2 gamma_2 w_1' + u + q v)` (equality
//! group of `r^(2)`), both affine in the committed digits and bits.

use agg_witness::digits::{U_DIGITS, V_DIGITS, Z_DIGITS};
use agg_witness::params::{GAMMA2, Q, W1_BITS};
use jolt_field::{One, Zero};

use crate::claims::{Kappa1Claims, PreStageClaims, Stage1Claims, Stage2Claims, Stage3Claims};
use crate::kernel::Term;
use crate::manifest::{Column, CommitmentManifest, DigitValue, GroupRole};
use crate::public_data::PublicData;
use crate::range::digits;
use crate::stage::{
    check_terminal, prove_stage, prove_stage_unchecked, replay_stage, EqPoint, GroupSpec, InstanceSpec, LaneId, Points,
    StageError, StageSpec, WeightId,
};
use crate::stage1::Stage1Output;
use crate::stage2::Stage2Output;
use crate::tables::ProverTables;
use crate::transcript::{terminal_site, Transcript};
use crate::weights::ring_weight_at;
use crate::OpeningField as E;

/// Stage number.
pub const STAGE: usize = 3;
/// Message degree.
pub const DEGREE: usize = 3;
/// Batching coefficients.
pub const GAMMA_COUNT: usize = 118;
/// Digits (links).
pub const LINKS: usize = 55;

/// Gamma index of `ring`.
pub const GAMMA_RING: usize = 0;
/// Gamma index of `link[k]`.
#[must_use]
pub const fn gamma_link(k: usize) -> usize {
    1 + k
}
/// Gamma index of the reduction of `b_t` from `r^(1)`.
#[must_use]
pub const fn gamma_b_r1(t: usize) -> usize {
    1 + LINKS + t
}
/// Gamma index of the reduction of `b_t` from `r^(2)`.
#[must_use]
pub const fn gamma_b_r2(t: usize) -> usize {
    1 + LINKS + W1_BITS + t
}
/// Gamma index of the reduction of `y^(2)_t` from `tau_2`.
#[must_use]
pub const fn gamma_y2_tau(t: usize) -> usize {
    1 + LINKS + 2 * W1_BITS + t
}
/// Gamma index of the reduction of `y^(2)_t` from `r^(1)_2`.
#[must_use]
pub const fn gamma_y2_r1(t: usize) -> usize {
    1 + LINKS + 3 * W1_BITS + t
}
/// Gamma index of the reduction of `kappa^(2)_{t,k}` from `tau_2`.
#[must_use]
pub const fn gamma_kappa2_tau(t: usize, k: usize) -> usize {
    1 + LINKS + 4 * W1_BITS + 3 * t + k
}
/// Gamma index of the reduction of `kappa^(3)_k` from `tau_3`.
#[must_use]
pub const fn gamma_kappa3_tau(k: usize) -> usize {
    1 + LINKS + 7 * W1_BITS + k
}
/// Gamma index of the reduction of `kappa^(1)_{t,k}` from `r^(1)_1`.
#[must_use]
pub const fn gamma_kappa1_r1(t: usize, k: usize) -> usize {
    1 + LINKS + 7 * W1_BITS + 2 + 3 * t + k
}

/// The stage-3 proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3Proof {
    /// Compressed degree-3 messages (3 elements each).
    pub rounds: Vec<Vec<E>>,
    /// The 67 opening claims at `r^(3)` and its suffixes.
    pub claims: Stage3Claims,
}

/// What the opening needs from stage 3.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3Output {
    /// `r^(3)`.
    pub point: Vec<E>,
    /// The opening claims.
    pub claims: Stage3Claims,
}

fn digit_columns(value: DigitValue) -> Vec<Column> {
    (0..value.layout().len() as u8)
        .map(|digit| Column::Digit { value, digit })
        .collect()
}

fn kappa_columns(values: impl Iterator<Item = DigitValue>) -> Vec<LaneId> {
    values
        .flat_map(|value| digit_columns(value).into_iter().map(LaneId::Column))
        .collect()
}

fn reductions(lanes: &[LaneId], gamma: impl Fn(usize) -> usize) -> Vec<InstanceSpec> {
    (0..lanes.len())
        .map(|l| InstanceSpec {
            gamma: gamma(l),
            terms: vec![Term::Product(vec![l])],
        })
        .collect()
}

/// The instance table.
#[must_use]
pub fn spec(manifest: &CommitmentManifest) -> StageSpec {
    let lanes = 0..W1_BITS;
    let all = digits(manifest);
    let mut groups = Vec::new();

    // Main domain, eq(r^(2)): the ring's committed part, the 17 main links and the b_t
    // reductions from r^(2). Lane index = position in G_main.
    let main = manifest.group(GroupRole::Main);
    let position = |column: Column| main.position(column).expect("a G_main column");
    let mut ring_b = Vec::new();
    for t in 0..W1_BITS {
        ring_b.push((position(Column::W1Bit(t as u8)), -((2 * GAMMA2) << t)));
    }
    for (&weight, column) in U_DIGITS.weights().iter().zip(digit_columns(DigitValue::U)) {
        ring_b.push((position(column), -weight));
    }
    for (&weight, column) in V_DIGITS.weights().iter().zip(digit_columns(DigitValue::V)) {
        ring_b.push((position(column), -(Q * weight)));
    }
    let mut instances = vec![InstanceSpec {
        gamma: GAMMA_RING,
        terms: vec![Term::Affine {
            weight: None,
            lanes: ring_b,
            constant: -(U_DIGITS.constant() + Q * V_DIGITS.constant()),
        }],
    }];
    for (k, digit) in all.iter().enumerate().filter(|(_, d)| d.group == GroupRole::Main) {
        instances.push(InstanceSpec {
            gamma: gamma_link(k),
            terms: vec![Term::Link(digit.position)],
        });
    }
    instances.extend(lanes.clone().map(|t| InstanceSpec {
        gamma: gamma_b_r2(t),
        terms: vec![Term::Product(vec![position(Column::W1Bit(t as u8))])],
    }));
    groups.push(GroupSpec {
        domain: GroupRole::Main,
        eq: EqPoint::R2,
        lanes: main.columns.iter().map(|&c| LaneId::Column(c)).collect(),
        instances,
    });

    // Main domain, no equality factor: w_A times the recomposed z.
    let z_lanes: Vec<(usize, i64)> = Z_DIGITS
        .weights()
        .iter()
        .enumerate()
        .map(|(k, &weight)| (1 + k, weight))
        .collect();
    groups.push(GroupSpec {
        domain: GroupRole::Main,
        eq: EqPoint::None,
        lanes: std::iter::once(LaneId::Weight(WeightId::Ring))
            .chain(digit_columns(DigitValue::Z).into_iter().map(LaneId::Column))
            .collect(),
        instances: vec![InstanceSpec {
            gamma: GAMMA_RING,
            terms: vec![Term::Affine {
                weight: Some(0),
                lanes: z_lanes,
                constant: Z_DIGITS.constant(),
            }],
        }],
    });

    // Main domain, eq(r^(1)): b_t reductions from r^(1).
    let bits: Vec<LaneId> = lanes
        .clone()
        .map(|t| LaneId::Column(Column::W1Bit(t as u8)))
        .collect();
    groups.push(GroupSpec {
        domain: GroupRole::Main,
        eq: EqPoint::R1,
        instances: reductions(&bits, gamma_b_r1),
        lanes: bits,
    });

    // D_1: links at r^(2)_1 and reductions from r^(1)_1 of the kappa^(1) digits.
    let kappa1 = kappa_columns((0..W1_BITS as u8).map(DigitValue::Kappa1));
    let links = |domain: GroupRole, lane_ids: &[LaneId]| -> Vec<InstanceSpec> {
        all.iter()
            .enumerate()
            .filter(|(_, d)| d.group == domain)
            .map(|(k, d)| InstanceSpec {
                gamma: gamma_link(k),
                terms: vec![Term::Link(
                    lane_ids
                        .iter()
                        .position(|&l| l == LaneId::Column(d.column))
                        .expect("digit lane"),
                )],
            })
            .collect()
    };
    groups.push(GroupSpec {
        domain: GroupRole::Level1,
        eq: EqPoint::R2,
        instances: links(GroupRole::Level1, &kappa1),
        lanes: kappa1.clone(),
    });
    groups.push(GroupSpec {
        domain: GroupRole::Level1,
        eq: EqPoint::R1,
        instances: reductions(&kappa1, |l| gamma_kappa1_r1(l / 3, l % 3)),
        lanes: kappa1,
    });

    // D_2: links of the kappa^(2) digits at r^(2)_2; reductions of y^(2) and kappa^(2)
    // from tau_2; reductions of y^(2) from r^(1)_2.
    let kappa2 = kappa_columns((0..W1_BITS as u8).map(DigitValue::Kappa2));
    groups.push(GroupSpec {
        domain: GroupRole::Level2,
        eq: EqPoint::R2,
        instances: links(GroupRole::Level2, &kappa2),
        lanes: kappa2.clone(),
    });
    let y2: Vec<LaneId> = lanes
        .clone()
        .map(|t| LaneId::Column(Column::Parity2(t as u8)))
        .collect();
    let mut tau2_lanes = y2.clone();
    tau2_lanes.extend_from_slice(&kappa2);
    let mut tau2_instances = reductions(&y2, gamma_y2_tau);
    tau2_instances.extend((0..kappa2.len()).map(|l| InstanceSpec {
        gamma: gamma_kappa2_tau(l / 3, l % 3),
        terms: vec![Term::Product(vec![W1_BITS + l])],
    }));
    groups.push(GroupSpec {
        domain: GroupRole::Level2,
        eq: EqPoint::Tau,
        lanes: tau2_lanes,
        instances: tau2_instances,
    });
    groups.push(GroupSpec {
        domain: GroupRole::Level2,
        eq: EqPoint::R1,
        instances: reductions(&y2, gamma_y2_r1),
        lanes: y2,
    });

    // D_3: links at r^(2)_3 and reductions from tau_3 of the kappa^(3) digits.
    let kappa3 = kappa_columns(std::iter::once(DigitValue::Kappa3));
    groups.push(GroupSpec {
        domain: GroupRole::Level3,
        eq: EqPoint::R2,
        instances: links(GroupRole::Level3, &kappa3),
        lanes: kappa3.clone(),
    });
    groups.push(GroupSpec {
        domain: GroupRole::Level3,
        eq: EqPoint::Tau,
        instances: reductions(&kappa3, gamma_kappa3_tau),
        lanes: kappa3,
    });
    StageSpec {
        stage: STAGE,
        degree: DEGREE,
        gamma_count: GAMMA_COUNT,
        groups,
    }
}

/// The claimed sums in gamma order.
#[must_use]
pub fn claimed_sums(
    t_at_r2: E,
    prestage: &PreStageClaims,
    stage1: &Stage1Claims,
    kappa1: &Kappa1Claims,
    stage2: &Stage2Claims,
) -> Vec<E> {
    let mut claims = vec![E::zero(); GAMMA_COUNT];
    claims[GAMMA_RING] = t_at_r2;
    for (k, &square) in stage2.squares.iter().enumerate() {
        claims[gamma_link(k)] = square;
    }
    for t in 0..W1_BITS {
        claims[gamma_b_r1(t)] = stage1.b[t];
        claims[gamma_b_r2(t)] = stage2.b[t];
        claims[gamma_y2_tau(t)] = prestage.y2[t];
        claims[gamma_y2_r1(t)] = stage1.y2[t];
        for k in 0..3 {
            claims[gamma_kappa2_tau(t, k)] = prestage.kappa2[t][k];
            claims[gamma_kappa1_r1(t, k)] = kappa1.0[t][k];
        }
    }
    for k in 0..2 {
        claims[gamma_kappa3_tau(k)] = prestage.kappa3[k];
    }
    claims
}

/// Prove stage 3. `points` holds `tau, r^(1), r^(2)`; `tables` holds `w_A`.
#[allow(clippy::too_many_arguments)]
pub fn prove(
    manifest: &CommitmentManifest,
    tables: &ProverTables<'_>,
    public: &PublicData,
    points: &Points,
    prestage: &PreStageClaims,
    stage1: &Stage1Output,
    stage2: &Stage2Output,
    transcript: &mut Transcript,
) -> Result<(Stage3Proof, Stage3Output), StageError> {
    prove_with(manifest, tables, public, points, prestage, stage1, stage2, transcript, true)
}

/// [`prove`] without the prover's claim self-check (tests).
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn prove_unchecked(
    manifest: &CommitmentManifest,
    tables: &ProverTables<'_>,
    public: &PublicData,
    points: &Points,
    prestage: &PreStageClaims,
    stage1: &Stage1Output,
    stage2: &Stage2Output,
    transcript: &mut Transcript,
) -> Result<(Stage3Proof, Stage3Output), StageError> {
    prove_with(manifest, tables, public, points, prestage, stage1, stage2, transcript, false)
}

#[allow(clippy::too_many_arguments)]
fn prove_with(
    manifest: &CommitmentManifest,
    tables: &ProverTables<'_>,
    public: &PublicData,
    points: &Points,
    prestage: &PreStageClaims,
    stage1: &Stage1Output,
    stage2: &Stage2Output,
    transcript: &mut Transcript,
    check: bool,
) -> Result<(Stage3Proof, Stage3Output), StageError> {
    let spec = spec(manifest);
    let t_at_r2 = public.t_evaluation(&stage2.point);
    let claims = claimed_sums(
        t_at_r2,
        prestage,
        &stage1.claims,
        &stage2.kappa1,
        &stage2.claims,
    );
    let run = if check {
        prove_stage(&spec, manifest, points, &claims, |id| tables.lane(id), transcript)?
    } else {
        prove_stage_unchecked(&spec, manifest, points, &claims, |id| tables.lane(id), transcript)?
    };
    let values: Vec<E> = manifest
        .groups()
        .iter()
        .flat_map(|group| group.columns.iter().map(|&c| run.value(&spec, LaneId::Column(c))))
        .collect();
    let claims = Stage3Claims(values);
    transcript.absorb_extensions(&terminal_site(STAGE), &claims.0)?;
    Ok((
        Stage3Proof {
            rounds: run.messages,
            claims: claims.clone(),
        },
        Stage3Output {
            point: run.point,
            claims,
        },
    ))
}

/// Verify stage 3 against stored public data; `T(r^(2))` and `w_A(r^(3))` are the
/// verifier's own evaluations (through [`verify_streamed`]).
#[allow(clippy::too_many_arguments)]
pub fn verify(
    manifest: &CommitmentManifest,
    public: &PublicData,
    points: &Points,
    prestage: &PreStageClaims,
    stage1: &Stage1Output,
    stage2: &Stage2Output,
    proof: &Stage3Proof,
    transcript: &mut Transcript,
) -> Result<Stage3Output, StageError> {
    verify_streamed(
        manifest,
        points,
        prestage,
        stage1,
        stage2,
        proof,
        transcript,
        |r2, r3| (public.t_evaluation(r2), ring_weight_at(public, r2, r3)),
    )
}

/// Verify stage 3 with the public pass deferred (spec Section 11 step 6): the rounds are
/// replayed with the ring claim `T(r^(2))` set to zero, the terminal claims absorbed, and
/// only then `public_pass(r^(2), r^(3))` returns `(T(r^(2)), w_A(r^(3)))`, so both come
/// from one pass over the public matrices. Compressed rounds make the final claim affine
/// in the initial claim with slope `prod_k r^(3)_k` (each round is
/// `g(x) = claim x + terms free of the claim`), so the deferred claim enters as
/// `gamma_ring T(r^(2)) prod_k r^(3)_k`.
#[allow(clippy::too_many_arguments)]
pub fn verify_streamed(
    manifest: &CommitmentManifest,
    points: &Points,
    prestage: &PreStageClaims,
    stage1: &Stage1Output,
    stage2: &Stage2Output,
    proof: &Stage3Proof,
    transcript: &mut Transcript,
    public_pass: impl FnOnce(&[E], &[E]) -> (E, E),
) -> Result<Stage3Output, StageError> {
    let spec = spec(manifest);
    let claims = claimed_sums(
        E::zero(),
        prestage,
        &stage1.claims,
        &stage2.kappa1,
        &stage2.claims,
    );
    let mut replay = replay_stage(&spec, manifest, &claims, &proof.rounds, transcript)?;
    if proof.claims.0.len() != Stage3Claims::LEN {
        return Err(StageError::Shape("stage 3 claims".into()));
    }
    transcript.absorb_extensions(&terminal_site(STAGE), &proof.claims.0)?;
    let (t_at_r2, ring_weight) = public_pass(&stage2.point, &replay.point);
    let slope = replay.point.iter().fold(E::one(), |acc, &r| acc * r);
    replay.final_claim += replay.gammas[GAMMA_RING] * t_at_r2 * slope;
    check_terminal(&spec, manifest, points, &replay, |id| match id {
        LaneId::Column(c) => proof.claims.column(manifest, c),
        LaneId::Weight(WeightId::Ring) => ring_weight,
        other => panic!("{other:?} is not a stage-3 lane"),
    })?;
    Ok(Stage3Output {
        point: replay.point,
        claims: proof.claims.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::evaluate_compressed;
    use crate::test_support::{field, point};
    use agg_witness::gen::SplitMix64;

    #[test]
    fn compressed_rounds_are_affine_in_the_claim_with_the_challenge_product_as_slope() {
        let mut rng = SplitMix64::new(5);
        let messages: Vec<Vec<E>> = (0..7).map(|_| point(&mut rng, DEGREE)).collect();
        let challenges = point(&mut rng, 7);
        let run = |claim: E| {
            messages
                .iter()
                .zip(&challenges)
                .fold(claim, |c, (m, &x)| evaluate_compressed(m, c, x))
        };
        let (deferred, rest) = (field(&mut rng), field(&mut rng));
        let slope = challenges.iter().fold(E::one(), |acc, &x| acc * x);
        assert_eq!(run(rest + deferred), run(rest) + deferred * slope);
    }

    #[test]
    fn stage_three_has_one_hundred_and_eighteen_instances_in_spec_order() {
        let manifest = CommitmentManifest::for_batch(8).unwrap();
        let spec = spec(&manifest);
        spec.validate().unwrap();
        assert_eq!(spec.groups.len(), 10);
        // The ring instance spans two groups; every other gamma index appears once.
        let mut counts = vec![0; GAMMA_COUNT];
        for group in &spec.groups {
            for instance in &group.instances {
                counts[instance.gamma] += 1;
            }
        }
        assert_eq!(counts[GAMMA_RING], 2);
        assert!(counts[1..].iter().all(|&c| c == 1));
        assert_eq!(gamma_kappa1_r1(5, 2), 117);
        assert_eq!(gamma_kappa3_tau(0), 98);
        for group in &spec.groups {
            let inner = usize::from(group.eq != EqPoint::None);
            for instance in &group.instances {
                for term in &instance.terms {
                    assert!(term.degree() + inner <= DEGREE, "{term:?}");
                }
            }
        }
        // Every committed column is read at r^(3).
        for group in manifest.groups() {
            for &column in &group.columns {
                assert!(spec.lane_ids().contains(&LaneId::Column(column)));
            }
        }
    }
}
