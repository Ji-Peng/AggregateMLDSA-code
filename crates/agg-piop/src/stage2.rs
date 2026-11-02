//! Stage 2 (degree 5 in relations H-u and H-exact): the level-1 definition and the range
//! leaves of the small digits on the virtual tables `P_Z` (every other range is the lookup
//! into `T_16` or, for the Boolean digits of H-exact, a zero-check in stage 3).
//!
//! `gamma^(2)` order: `def_y1[t]` (0..6), then one leaf per leaf-checked digit: H-u has
//! `leaf4[Z_hi]` (6), `leaf6[U_hi]` (7) and `leaf8[K_2]`, `leaf8[K_3]` (8, 9), H-exact
//! `leaf6[U_hi]` (6) and `leaf8[K_2]`, `leaf8[K_3]` (7, 8). `K_2, K_3`
//! are the top digits of the projection outputs of `v` (`projection`); their leaves are on the
//! projection domain at the suffix of `r^(1)`. After
//! the terminal claims the prover sends the shared claim `c` of the projection identity.
//! The leaf of the eight-valued `K_2, K_3` (`Q_8`, degree 4 in `P_Z`) sets the stage degree
//! `1 + 4 = 5` in both relations.
//!
//! With polynomial lanes (ML-DSA-65, ML-DSA-87) `def_y1[t]` (`0..4`) is one instance with
//! `k` products `w_1^(rho) b_t^(rho)`, one per lane of the bit, and the leaves follow the
//! lanes in manifest order: `leaf8[Z_mid^(s)]` (`l`), `leaf8[U_mid^(rho)]` (`k`), then
//! `leaf8[K_2]`, `leaf8[K_3]`, `leaf4[K_4]` on the projection domain (18 and 22 instances).
//!
//! Relation H-FIPS has `leaf8[K_2]`, `leaf8[K_3]` (gamma 6, 7) as its
//! leaves and adds one instance after them (gamma 8): the degree-3 zero-check
//! `F3 = e (e - 1)(e - delta)` at `r^(1)`
//! (degree 4 with `eq`; stage 3 has degree 3). Its terminal claims add `delta(r^(2))` and
//! `e(r^(2))`, reduced in stage 3. Every other H-FIPS check has degree at most 2 and sits in
//! stage 3.
use agg_witness::bridge::BridgeMatrices;
use agg_witness::padding::Geometry;
use agg_witness::params::W1_BITS;
use jolt_field::Zero;

use crate::claims::{bit_columns, Kappa1Claims, Stage1Claims, Stage2Claims};
use crate::kernel::{LeafPoly, Term};
use crate::manifest::{Column, CommitmentManifest, GroupRole};
use crate::range::{digits, Digit};
use crate::stage::{
    check_terminal, prove_stage_split, replay_stage, suffix, EqPoint, GroupSpec, InstanceSpec, LaneId,
    Points, StageError, StageSpec, WeightId,
};
use crate::stage1::Stage1Output;
use crate::tables::ProverTables;
use crate::transcript::{terminal_site, Transcript, KAPPA1_SITE, PROJ_CLAIM_SITE};
use crate::weights::{level1_lane_weights_at, level1_weight_at};
use crate::OpeningField as E;

/// Stage number.
pub const STAGE: usize = 2;
/// Message degree (`leaf8[K_2]`: the `eq` factor times `Q_8(P_Z)`).
pub const DEGREE: usize = 5;
/// H-FIPS instances of stage 2 (`F3`).
pub const FIPS_INSTANCES: usize = 1;

/// Batching coefficients: 10 under H-u, 9 under H-exact and H-FIPS (the leaves of `K_2`
/// and `K_3` in each); 18 and 22 for ML-DSA-65 and ML-DSA-87.
#[must_use]
pub fn gamma_count(manifest: &CommitmentManifest) -> usize {
    bits(manifest) + manifest.leaf_digits().len() + if manifest.is_fips() { FIPS_INSTANCES } else { 0 }
}

/// Bits of `w_1'` (the `def_y1` instances).
fn bits(manifest: &CommitmentManifest) -> usize {
    manifest.shape().w1_bits
}

/// Gamma index of H-FIPS's `F3`.
#[must_use]
pub fn gamma_fips(manifest: &CommitmentManifest) -> usize {
    bits(manifest) + manifest.leaf_digits().len()
}

/// The H-FIPS terminal lanes of stage 2 beyond the bits, in wire order: `delta`, `e` at
/// `r^(2)`.
pub const FIPS_TERMINALS: [Column; 2] = [Column::DELTA, Column::WRAP];

/// `F3 = e (e - 1)(e - delta) = e^3 - e^2 - e^2 delta + e delta` in the lanes `[delta, e]`.
#[must_use]
pub fn fips_edge_term() -> Term {
    let (delta, e) = (0, 1);
    Term::Monomials(vec![
        (1, vec![e, e, e]),
        (-1, vec![e, e]),
        (-1, vec![e, e, delta]),
        (1, vec![e, delta]),
    ])
}

/// Gamma index of `def_y1[t]`.
#[must_use]
pub const fn gamma_def_y1(t: usize) -> usize {
    t
}
/// Gamma index of the leaf of the `k`-th leaf-checked digit (`manifest.leaf_digits()`) for
/// ML-DSA-44 ([`gamma_leaf_for`]).
#[must_use]
pub const fn gamma_leaf(k: usize) -> usize {
    W1_BITS + k
}

/// Gamma index of the leaf of the `k`-th leaf-checked digit of any parameter set.
#[must_use]
pub fn gamma_leaf_for(manifest: &CommitmentManifest, k: usize) -> usize {
    bits(manifest) + k
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
    /// The shared claim `c` of the projection identity (`projection`), sent after the
    /// terminal claims.
    pub projection: E,
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
    /// The shared claim `c` of the projection identity.
    pub projection: E,
}

/// The leaf polynomial of a leaf-checked digit.
fn leaf_poly(digit: &Digit) -> LeafPoly {
    match digit.base {
        6 => LeafPoly::Q6,
        8 => LeafPoly::Q8,
        4 => LeafPoly::Q4,
        other => panic!("unsupported leaf base {other}"),
    }
}

/// The instance table.
#[must_use]
pub fn spec(manifest: &CommitmentManifest) -> StageSpec {
    let all = digits(manifest);
    let mut groups = Vec::new();
    // def_y1[t] on the main domain: the weights w_1^(rho), then the bits b_t^(rho)
    // bit-major; instance t is sum_rho w_1^(rho) b_t^(rho).
    let weight_lanes = manifest.shape().lanes_k;
    let bit_lanes = bit_columns(manifest);
    groups.push(GroupSpec {
        domain: GroupRole::Main,
        eq: EqPoint::None,
        lanes: (0..weight_lanes as u8)
            .map(|rho| LaneId::Weight(WeightId::Level1(rho)))
            .chain(bit_lanes.iter().map(|&c| LaneId::Column(c)))
            .collect(),
        instances: (0..bits(manifest))
            .map(|t| {
                let terms = (0..weight_lanes)
                    .map(|rho| Term::Product(vec![rho, weight_lanes + t * weight_lanes + rho]))
                    .collect();
                InstanceSpec::new(gamma_def_y1(t), terms)
            })
            .collect(),
    });
    // The leaves, one group per domain (the main domain: Z_hi, U_hi; the projection domain:
    // K_2, K_3), each at the domain's suffix of r^(1).
    for domain in [GroupRole::Main, GroupRole::Proj] {
        let local: Vec<(usize, &Digit)> =
            all.iter().enumerate().filter(|(_, d)| d.column.domain() == domain).collect();
        if local.is_empty() {
            continue;
        }
        groups.push(GroupSpec {
            domain,
            eq: EqPoint::R1,
            lanes: local.iter().map(|(_, d)| LaneId::DigitSquare(d.column)).collect(),
            instances: local
                .iter()
                .enumerate()
                .map(|(lane, &(k, digit))| {
                    InstanceSpec::new(gamma_leaf_for(manifest, k), vec![Term::Leaf(lane, leaf_poly(digit))])
                })
                .collect(),
        });
    }
    if manifest.is_fips() {
        // F3 on the main domain at r^(1).
        groups.push(GroupSpec {
            domain: GroupRole::Main,
            eq: EqPoint::R1,
            lanes: FIPS_TERMINALS.iter().map(|&c| LaneId::Column(c)).collect(),
            instances: vec![InstanceSpec::new(gamma_fips(manifest), vec![fips_edge_term()])],
        });
    }
    StageSpec {
        stage: STAGE,
        degree: DEGREE,
        gamma_count: gamma_count(manifest),
        groups,
    }
}

/// The claimed sums in gamma order: the level-1 sum for `def_y1`, zero for the leaves and
/// H-FIPS's `F3`.
#[must_use]
pub fn claimed_sums(manifest: &CommitmentManifest, stage1: &Stage1Claims, kappa1: &Kappa1Claims) -> Vec<E> {
    let mut claims = vec![E::zero(); gamma_count(manifest)];
    for t in 0..bits(manifest) {
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
    let claims = claimed_sums(manifest, &stage1.claims, &kappa1);
    let run = if check {
        prove_stage_split(&spec, manifest, points, &claims, |id| tables.lane(id), tables.level1_split(), transcript, true)?
    } else {
        prove_stage_split(&spec, manifest, points, &claims, |id| tables.lane(id), tables.level1_split(), transcript, false)?
    };
    let value = |id| run.value(&spec, id);
    let claims = Stage2Claims {
        b: bit_columns(manifest).into_iter().map(|c| value(LaneId::Column(c))).collect(),
        squares: digits(manifest)
            .iter()
            .map(|digit| value(LaneId::DigitSquare(digit.column)))
            .collect(),
        fips: if manifest.is_fips() {
            FIPS_TERMINALS.iter().map(|&c| value(LaneId::Column(c))).collect()
        } else {
            Vec::new()
        },
    };
    transcript.absorb_extensions(&terminal_site(STAGE), &claims.to_vec())?;
    let projection = crate::projection::claim(
        &tables.tables.group(GroupRole::Proj),
        &run.point,
        manifest.group(GroupRole::Proj).logical_vars,
    );
    transcript.absorb_extensions(PROJ_CLAIM_SITE, &[projection])?;
    Ok((
        Stage2Proof {
            kappa1: kappa1.clone(),
            rounds: run.messages,
            claims: claims.clone(),
            projection,
        },
        Stage2Output {
            point: run.point,
            kappa1,
            claims,
            projection,
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
    if proof.kappa1.0.len() != Kappa1Claims::len(manifest) || proof.claims.b.len() != bit_columns(manifest).len() {
        return Err(StageError::Shape("stage 2 bit claims".into()));
    }
    transcript.absorb_extensions(KAPPA1_SITE, &proof.kappa1.to_vec())?;
    let spec = spec(manifest);
    let claims = claimed_sums(manifest, &stage1.claims, &proof.kappa1);
    let replay = replay_stage(&spec, manifest, &claims, &proof.rounds, transcript)?;
    let all = digits(manifest);
    let fips = if manifest.is_fips() { FIPS_TERMINALS.len() } else { 0 };
    if proof.claims.squares.len() != all.len() || proof.claims.fips.len() != fips {
        return Err(StageError::Shape("stage 2 claims".into()));
    }
    transcript.absorb_extensions(&terminal_site(STAGE), &proof.claims.to_vec())?;
    transcript.absorb_extensions(PROJ_CLAIM_SITE, &[proof.projection])?;
    let r1_1 = suffix(&stage1.point, manifest.group(GroupRole::Level1).logical_vars);
    let level1 = if manifest.has_lanes() {
        level1_lane_weights_at(manifest, &matrices.m1, &r1_1, &replay.point)
    } else {
        let geom = Geometry::new(manifest.records()).map_err(|e| StageError::Internal(format!("{e:?}")))?;
        vec![level1_weight_at(&geom, &matrices.m1, &r1_1, &replay.point)]
    };
    let lanes = manifest.shape().lanes_k;
    check_terminal(&spec, manifest, points, &replay, |id| match id {
        LaneId::Column(Column::W1Bit(t, lane)) => proof.claims.b[usize::from(t) * lanes + usize::from(lane)],
        LaneId::Column(c) if FIPS_TERMINALS.contains(&c) && manifest.is_fips() => {
            proof.claims.fips[FIPS_TERMINALS.iter().position(|&f| f == c).expect("listed")]
        }
        LaneId::DigitSquare(c) => {
            let k = all.iter().position(|d| d.column == c).expect("a digit");
            proof.claims.squares[k]
        }
        LaneId::Weight(WeightId::Level1(lane)) => level1[usize::from(lane)],
        other => panic!("{other:?} is not a stage-2 lane"),
    })?;
    Ok(Stage2Output {
        point: replay.point,
        kappa1: proof.kappa1.clone(),
        claims: proof.claims.clone(),
        projection: proof.projection,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Ring;

    #[test]
    fn h_exact_stage_two_has_nine_instances_leaf6_and_the_k_leaves() {
        let manifest = CommitmentManifest::for_relation(8, crate::manifest::Relation::HExact).unwrap();
        let spec = spec(&manifest);
        spec.validate().unwrap();
        assert_eq!(gamma_count(&manifest), 9);
        let instances: usize = spec.groups.iter().map(|g| g.instances.len()).sum();
        assert_eq!(instances, 9);
        let leaves = &spec.groups[1];
        let u_hi = leaves.lane(LaneId::DigitSquare(Column::U_HI));
        assert_eq!(leaves.instances.len(), 1);
        assert_eq!(leaves.instances[0].gamma, 6);
        assert_eq!(leaves.instances[0].terms, vec![Term::Leaf(u_hi, LeafPoly::Q6)]);
        let k_leaf = &spec.groups[2];
        assert_eq!((k_leaf.domain, k_leaf.eq), (GroupRole::Proj, EqPoint::R1));
        assert_eq!(k_leaf.lanes, vec![LaneId::DigitSquare(Column::K_2), LaneId::DigitSquare(Column::K_3)]);
        assert_eq!(k_leaf.instances[0].gamma, 7);
        assert_eq!(k_leaf.instances[0].terms, vec![Term::Leaf(0, LeafPoly::Q8)]);
        assert_eq!(k_leaf.instances[1].gamma, 8);
        assert_eq!(k_leaf.instances[1].terms, vec![Term::Leaf(1, LeafPoly::Q8)]);
        assert_eq!(spec.degree, DEGREE);
    }

    #[test]
    fn h_fips_stage_two_has_def_y1_and_f3() {
        let manifest = CommitmentManifest::for_relation(8, crate::manifest::Relation::HFips).unwrap();
        let spec = spec(&manifest);
        spec.validate().unwrap();
        assert_eq!(gamma_count(&manifest), 9);
        // def_y1, the leaves of K_2, K_3 on the projection domain (H-FIPS has no
        // main-domain leaf digit) and F3 at r^(1).
        assert_eq!(spec.groups.len(), 3);
        assert_eq!(spec.groups[1].lanes, vec![LaneId::DigitSquare(Column::K_2), LaneId::DigitSquare(Column::K_3)]);
        assert_eq!(spec.groups[1].instances[0].gamma, 6);
        let f3 = &spec.groups[2];
        assert_eq!(f3.eq, EqPoint::R1);
        assert_eq!(f3.lanes, vec![LaneId::Column(Column::DELTA), LaneId::Column(Column::WRAP)]);
        assert_eq!(f3.instances[0].gamma, 8);
        assert_eq!(f3.instances[0].terms, vec![fips_edge_term()]);
        assert!(fips_edge_term().degree() + 1 <= DEGREE);
        // F3 vanishes exactly on e in {0, 1, delta} for delta in {-1, 0, 1}.
        for delta in -1_i64..=1 {
            for e in -3_i64..=3 {
                let values = [E::from_i64(delta), E::from_i64(e)];
                let zero = fips_edge_term().evaluate(&values) == E::zero();
                assert_eq!(zero, e == 0 || e == 1 || e == delta, "delta {delta} e {e}");
            }
        }
    }

    #[test]
    fn stage_two_has_ten_instances_in_spec_order() {
        let manifest = CommitmentManifest::for_batch(8).unwrap();
        let spec = spec(&manifest);
        spec.validate().unwrap();
        assert_eq!(spec.groups.len(), 3);
        let instances: usize = spec.groups.iter().map(|g| g.instances.len()).sum();
        assert_eq!(instances, 10);
        // leaf8[K_2], leaf8[K_3] on the projection domain at the suffix of r^(1), gamma 8, 9.
        let k_leaf = &spec.groups[2];
        assert_eq!((k_leaf.domain, k_leaf.eq), (GroupRole::Proj, EqPoint::R1));
        assert_eq!(k_leaf.instances[0].gamma, gamma_leaf(2));
        assert_eq!(k_leaf.instances[0].terms, vec![Term::Leaf(0, LeafPoly::Q8)]);
        assert_eq!(k_leaf.instances[1].gamma, gamma_leaf(3));
        assert_eq!(k_leaf.instances[1].terms, vec![Term::Leaf(1, LeafPoly::Q8)]);
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
        // The degree is tight: leaf8[K_2] reaches it.
        assert_eq!(max, DEGREE);
        assert_eq!(DEGREE, crate::transcript::STAGE_DEGREES[STAGE - 1]);
    }
}
