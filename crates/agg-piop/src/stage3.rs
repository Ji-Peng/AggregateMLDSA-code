//! Stage 3 (spec Section 7.5, degree 3): the ring identity (7.6) with `(A z)` pulled back
//! through the public matrices, the two range links, the evaluation reductions to
//! `r^(3)`, and the four lookup-stack reductions (lookup layout L16, `docs/lookup.md`
//! Section 6 item 4).
//!
//! `gamma^(3)` order: `ring` (0), `link[Z_hi]` (1), `link[U_hi]` (2), then the reductions:
//! `b_t` from `r^(1)` (3..9), `b_t` from `r^(2)` (9..15), `y^(2)_t` from `tau_2` (15..21),
//! `y^(2)_t` from `r^(1)_2` (21..27), `K^(2)_t` from `tau_2` (27..33), `K^(3)` from
//! `tau_3` (33), `K^(1)_t` from `r^(1)_1` (34..40), and the lookup stacks `j = 0..4`
//! (40..44): `sum_x eq(pi_j, x) sum_c w_{j,c} f_c(x) = g_j(pi_j, pi_top) + 2^15 sum_{c >=
//! cols} w_{j,c}` with `w_{j,c} = eq(pi_top, c)` (`StackShape::column_claim`), one part
//! per column with scale `w_{j,c}`.
//!
//! The ring instance is one instance with two summands on two groups: `w_A z` (weight
//! lane, no equality factor) and `-eq(r^(2), x)(2 gamma_2 w_1' + u + q v)` (equality
//! group of `r^(2)`), both affine in the committed columns: `z = Z_lo + 2^16 Z_hi + 2^15`,
//! `u = U_lo + 63,079 U_hi + 31,540` (relation H-u), `v = V_lo + 2^16 V_hi + 2^15`.

use agg_witness::params::{GAMMA2, Q, W1_BITS};
use jolt_field::{One, Zero};

use crate::claims::{Kappa1Claims, PreStageClaims, Stage1Claims, Stage2Claims, Stage3Claims};
use crate::kernel::Term;
use crate::lookup::LookupClaims;
use crate::manifest::{
    stack_columns, Column, CommitmentManifest, GroupRole, STACK_DOMAINS, U16_DIGITS, V16_DIGITS,
    Z16_DIGITS,
};
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
/// Leaf-checked digits (links).
pub const LINKS: usize = 2;
/// Lookup stacks (reductions).
pub const STACKS: usize = 4;
/// Batching coefficients.
pub const GAMMA_COUNT: usize = 1 + LINKS + 4 * W1_BITS + W1_BITS + 1 + W1_BITS + STACKS;

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
/// Gamma index of the reduction of `K^(2)_t` from `tau_2`.
#[must_use]
pub const fn gamma_kappa2_tau(t: usize) -> usize {
    1 + LINKS + 4 * W1_BITS + t
}
/// Gamma index of the reduction of `K^(3)` from `tau_3`.
pub const GAMMA_KAPPA3_TAU: usize = 1 + LINKS + 5 * W1_BITS;
/// Gamma index of the reduction of `K^(1)_t` from `r^(1)_1`.
#[must_use]
pub const fn gamma_kappa1_r1(t: usize) -> usize {
    2 + LINKS + 5 * W1_BITS + t
}
/// Gamma index of the reduction of lookup stack `j`.
#[must_use]
pub const fn gamma_stack(j: usize) -> usize {
    2 + LINKS + 6 * W1_BITS + j
}

/// The stage-3 proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3Proof {
    /// Compressed degree-3 messages (3 elements each).
    pub rounds: Vec<Vec<E>>,
    /// The 31 opening claims at `r^(3)` and its suffixes.
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

/// The column weights `eq(pi_top, c)` of every lookup stack's real columns (the scales of
/// the stack reductions), from the lookup's stack claims.
#[must_use]
pub fn stack_weights(manifest: &CommitmentManifest, lookup: &LookupClaims) -> Vec<Vec<E>> {
    let shape = manifest.lookup_shape();
    shape
        .stacks
        .iter()
        .zip(&lookup.stacks)
        .map(|(stack, (point, _))| stack.column_weights(point))
        .collect()
}

/// The stack reduction's claimed sums and the low points (the equality points of the stack
/// groups), from the lookup's stack claims.
#[must_use]
pub fn stack_reductions(manifest: &CommitmentManifest, lookup: &LookupClaims) -> (Vec<E>, Vec<Vec<E>>) {
    let shape = manifest.lookup_shape();
    shape
        .stacks
        .iter()
        .zip(&lookup.stacks)
        .map(|(stack, (point, value))| (stack.column_claim(point, *value), point[..stack.vars].to_vec()))
        .unzip()
}

fn reductions(lanes: &[LaneId], gamma: impl Fn(usize) -> usize) -> Vec<InstanceSpec> {
    (0..lanes.len())
        .map(|l| InstanceSpec::new(gamma(l), vec![Term::Product(vec![l])]))
        .collect()
}

fn column_lanes(columns: &[Column]) -> Vec<LaneId> {
    columns.iter().map(|&c| LaneId::Column(c)).collect()
}

/// The instance table; `weights[j]` are the column weights of lookup stack `j`
/// ([`stack_weights`]).
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn spec(manifest: &CommitmentManifest, weights: &[Vec<E>]) -> StageSpec {
    assert_eq!(weights.len(), STACKS, "one weight vector per lookup stack");
    let lanes = 0..W1_BITS;
    let all = digits(manifest);
    let mut groups = Vec::new();

    // Main domain, eq(r^(2)): the ring's committed part (u, v, w_1'), the two links and the
    // b_t reductions from r^(2).
    let mut main_columns = vec![Column::U_LO, Column::V_LO, Column::V_HI, Column::Z_HI, Column::U_HI];
    main_columns.extend(lanes.clone().map(|t| Column::W1Bit(t as u8)));
    let position = |column: Column| main_columns.iter().position(|&c| c == column).expect("lane");
    let [u_weights, v_weights] = [U16_DIGITS.weights(), V16_DIGITS.weights()];
    let mut ring_b: Vec<(usize, i64)> = (0..W1_BITS)
        .map(|t| (position(Column::W1Bit(t as u8)), -((2 * GAMMA2) << t)))
        .collect();
    ring_b.push((position(Column::U_LO), -u_weights[0]));
    ring_b.push((position(Column::U_HI), -u_weights[1]));
    ring_b.push((position(Column::V_LO), -(Q * v_weights[0])));
    ring_b.push((position(Column::V_HI), -(Q * v_weights[1])));
    let mut instances = vec![InstanceSpec::new(
        GAMMA_RING,
        vec![Term::Affine {
            weight: None,
            lanes: ring_b,
            constant: -(U16_DIGITS.constant() + Q * V16_DIGITS.constant()),
        }],
    )];
    for (k, digit) in all.iter().enumerate() {
        instances.push(InstanceSpec::new(gamma_link(k), vec![Term::Link(position(digit.column))]));
    }
    instances.extend(lanes.clone().map(|t| {
        InstanceSpec::new(gamma_b_r2(t), vec![Term::Product(vec![position(Column::W1Bit(t as u8))])])
    }));
    groups.push(GroupSpec {
        domain: GroupRole::Main,
        eq: EqPoint::R2,
        lanes: column_lanes(&main_columns),
        instances,
    });

    // Main domain, no equality factor: w_A times z = Z_lo + 2^16 Z_hi + 2^15.
    let z_weights = Z16_DIGITS.weights();
    groups.push(GroupSpec {
        domain: GroupRole::Main,
        eq: EqPoint::None,
        lanes: vec![
            LaneId::Weight(WeightId::Ring),
            LaneId::Column(Column::Z_LO),
            LaneId::Column(Column::Z_HI),
        ],
        instances: vec![InstanceSpec::new(
            GAMMA_RING,
            vec![Term::Affine {
                weight: Some(0),
                lanes: vec![(1, z_weights[0]), (2, z_weights[1])],
                constant: Z16_DIGITS.constant(),
            }],
        )],
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

    // D_1, eq(r^(1)_1): K^(1)_t reductions.
    let kappa1 = column_lanes(&stack_columns(1));
    groups.push(GroupSpec {
        domain: GroupRole::Level1,
        eq: EqPoint::R1,
        instances: reductions(&kappa1, gamma_kappa1_r1),
        lanes: kappa1,
    });

    // D_2, eq(tau_2): y^(2)_t and K^(2)_t reductions; eq(r^(1)_2): y^(2)_t.
    let kappa2 = column_lanes(&stack_columns(2));
    let y2: Vec<LaneId> = lanes
        .clone()
        .map(|t| LaneId::Column(Column::Parity2(t as u8)))
        .collect();
    let mut tau2_lanes = y2.clone();
    tau2_lanes.extend_from_slice(&kappa2);
    let mut tau2_instances = reductions(&y2, gamma_y2_tau);
    tau2_instances.extend(
        (0..kappa2.len())
            .map(|l| InstanceSpec::new(gamma_kappa2_tau(l), vec![Term::Product(vec![W1_BITS + l])])),
    );
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

    // D_3, eq(tau_3): K^(3).
    let kappa3 = column_lanes(&stack_columns(3));
    groups.push(GroupSpec {
        domain: GroupRole::Level3,
        eq: EqPoint::Tau,
        instances: reductions(&kappa3, |_| GAMMA_KAPPA3_TAU),
        lanes: kappa3,
    });

    // The lookup stacks: one equality group per stack at its low point, one part per column
    // with the column weight as scale.
    for (j, stack_weights) in weights.iter().enumerate() {
        let columns = stack_columns(j);
        assert_eq!(stack_weights.len(), columns.len(), "one weight per stack column");
        groups.push(GroupSpec {
            domain: STACK_DOMAINS[j],
            eq: EqPoint::Lookup(j),
            lanes: column_lanes(&columns),
            instances: stack_weights
                .iter()
                .enumerate()
                .map(|(c, &scale)| InstanceSpec {
                    gamma: gamma_stack(j),
                    scale,
                    terms: vec![Term::Product(vec![c])],
                })
                .collect(),
        });
    }
    StageSpec {
        stage: STAGE,
        degree: DEGREE,
        gamma_count: GAMMA_COUNT,
        groups,
    }
}

/// The claimed sums in gamma order; `stacks[j]` is stack `j`'s column claim
/// ([`stack_reductions`]).
#[must_use]
pub fn claimed_sums(
    t_at_r2: E,
    prestage: &PreStageClaims,
    stage1: &Stage1Claims,
    kappa1: &Kappa1Claims,
    stage2: &Stage2Claims,
    stacks: &[E],
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
        claims[gamma_kappa2_tau(t)] = prestage.kappa2[t];
        claims[gamma_kappa1_r1(t)] = kappa1.0[t];
    }
    claims[GAMMA_KAPPA3_TAU] = prestage.kappa3;
    for (j, &claim) in stacks.iter().enumerate() {
        claims[gamma_stack(j)] = claim;
    }
    claims
}

/// The lookup inputs of stage 3: the stack weights, the stack claims and the low points.
#[derive(Clone, Debug)]
pub struct LookupInputs {
    /// `w_{j,c}` per stack.
    pub weights: Vec<Vec<E>>,
    /// The column claim of each stack.
    pub claims: Vec<E>,
}

impl LookupInputs {
    /// From the lookup's claims; also returns the low points for [`Points::lookup`].
    #[must_use]
    pub fn new(manifest: &CommitmentManifest, lookup: &LookupClaims) -> (Self, Vec<Vec<E>>) {
        let weights = stack_weights(manifest, lookup);
        let (claims, points) = stack_reductions(manifest, lookup);
        (Self { weights, claims }, points)
    }
}

/// Prove stage 3. `points` holds `tau, r^(1), r^(2)` and the lookup stack points;
/// `tables` holds `w_A`.
#[allow(clippy::too_many_arguments)]
pub fn prove(
    manifest: &CommitmentManifest,
    tables: &ProverTables<'_>,
    public: &PublicData,
    points: &Points,
    prestage: &PreStageClaims,
    stage1: &Stage1Output,
    stage2: &Stage2Output,
    lookup: &LookupInputs,
    transcript: &mut Transcript,
) -> Result<(Stage3Proof, Stage3Output), StageError> {
    prove_with(manifest, tables, public, points, prestage, stage1, stage2, lookup, transcript, true)
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
    lookup: &LookupInputs,
    transcript: &mut Transcript,
) -> Result<(Stage3Proof, Stage3Output), StageError> {
    prove_with(manifest, tables, public, points, prestage, stage1, stage2, lookup, transcript, false)
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
    lookup: &LookupInputs,
    transcript: &mut Transcript,
    check: bool,
) -> Result<(Stage3Proof, Stage3Output), StageError> {
    let spec = spec(manifest, &lookup.weights);
    let t_at_r2 = public.t_evaluation(&stage2.point);
    let claims = claimed_sums(
        t_at_r2,
        prestage,
        &stage1.claims,
        &stage2.kappa1,
        &stage2.claims,
        &lookup.claims,
    );
    let run = if check {
        prove_stage(&spec, manifest, points, &claims, |id| tables.lane(id), transcript)?
    } else {
        prove_stage_unchecked(&spec, manifest, points, &claims, |id| tables.lane(id), transcript)?
    };
    let values: Vec<E> = manifest
        .groups()
        .iter()
        .filter(|group| group.role != GroupRole::Mult)
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
    lookup: &LookupInputs,
    proof: &Stage3Proof,
    transcript: &mut Transcript,
) -> Result<Stage3Output, StageError> {
    verify_streamed(
        manifest,
        points,
        prestage,
        stage1,
        stage2,
        lookup,
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
    lookup: &LookupInputs,
    proof: &Stage3Proof,
    transcript: &mut Transcript,
    public_pass: impl FnOnce(&[E], &[E]) -> (E, E),
) -> Result<Stage3Output, StageError> {
    let spec = spec(manifest, &lookup.weights);
    let claims = claimed_sums(
        E::zero(),
        prestage,
        &stage1.claims,
        &stage2.kappa1,
        &stage2.claims,
        &lookup.claims,
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
    fn stage_three_has_forty_four_instances_in_spec_order() {
        let manifest = CommitmentManifest::for_batch(8).unwrap();
        let mut rng = SplitMix64::new(3);
        let weights: Vec<Vec<E>> = (0..STACKS)
            .map(|j| point(&mut rng, stack_columns(j).len()))
            .collect();
        let spec = spec(&manifest, &weights);
        spec.validate().unwrap();
        assert_eq!(GAMMA_COUNT, 44);
        assert_eq!(spec.groups.len(), 7 + STACKS);
        // The ring instance spans two groups and each stack reduction one part per column;
        // every other gamma index appears once.
        let mut counts = vec![0; GAMMA_COUNT];
        for group in &spec.groups {
            for instance in &group.instances {
                counts[instance.gamma] += 1;
            }
        }
        assert_eq!(counts[GAMMA_RING], 2);
        assert_eq!(
            (0..STACKS).map(|j| counts[gamma_stack(j)]).collect::<Vec<_>>(),
            [4, 6, 6, 1]
        );
        assert!(counts[1..gamma_stack(0)].iter().all(|&c| c == 1));
        assert_eq!(gamma_kappa1_r1(5), 39);
        assert_eq!(GAMMA_KAPPA3_TAU, 33);
        assert_eq!(gamma_stack(3), 43);
        for group in &spec.groups {
            let inner = usize::from(group.eq != EqPoint::None);
            for instance in &group.instances {
                for term in &instance.terms {
                    assert!(term.degree() + inner <= DEGREE, "{term:?}");
                }
            }
        }
        // Every committed column but the multiplicity digits is read at r^(3).
        for group in manifest.groups() {
            for &column in &group.columns {
                assert_eq!(
                    spec.lane_ids().contains(&LaneId::Column(column)),
                    group.role != GroupRole::Mult,
                    "{}",
                    column.name()
                );
            }
        }
    }
}
