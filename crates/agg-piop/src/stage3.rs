//! Stage 3 (spec Section 7.5, degree 3): the ring identity (7.6) with `(A z)` pulled back
//! through the public matrices, the range links, the Booleanity zero-checks of H-exact's
//! bit digits, the evaluation reductions to `r^(3)`, and the four lookup-stack reductions
//! (lookup layout L16, `docs/lookup.md` Section 6 item 4).
//!
//! `gamma^(3)` order ([`Gammas`], with `o = 1 + links + bools`): `ring` (0), one `link` per
//! leaf-checked digit (H-u: `link[Z_hi]` 1, `link[U_hi]` 2; H-exact: `link[U_hi]` 1), one
//! `bool` per Boolean value digit (H-exact: `bool[bz_hi]` 2, `bool[a_z]` 3; claim 0,
//! `docs/h-exact.md` 3.4), then the reductions: `b_t` from `r^(1)` (`o..o+6`), `b_t` from
//! `r^(2)`, `y^(2)_t` from `tau_2`, `y^(2)_t` from `r^(1)_2`, `K^(2)_t` from `tau_2`, `K^(3)`
//! from `tau_3`, `K^(1)_t` from `r^(1)_1`, and the lookup stacks `j = 0..4`: `sum_x eq(pi_j,
//! x) sum_c w_{j,c} f_c(x) = g_j(pi_j, pi_top) + 2^15 sum_{c >= cols} w_{j,c}` with `w_{j,c}
//! = eq(pi_top, c)` (`StackShape::column_claim`), one part per column with scale `w_{j,c}`.
//! H-u has 44 coefficients, H-exact 45.
//!
//! The ring instance is one instance with two summands on two groups: `w_A z` (weight
//! lane, no equality factor) and `-eq(r^(2), x)(2 gamma_2 w_1' + u + q v)` (equality
//! group of `r^(2)`), both affine in the committed columns: `z = Z_lo + 2^16 Z_hi + 2^15`
//! (H-u) or `z = S_zlo + 2^16 B_zhi + 130,915 a_z - 98,225` (H-exact), `u = U_lo + 63,079
//! U_hi + 31,540`, `v = V_lo + 2^16 V_hi + 2^15`.
//!
//! Relation H-FIPS (`docs/h-exact.md` Section 10.7): `bool` for `bz_hi`, `l_hi`, `a_z`, then
//! four H-FIPS checks after the bools ([`Gammas::fips_check`]): `F1 = s (s - 1)`,
//! `F2 = delta (delta - 2 s + 1)` (`delta (delta - sigma)` with `sigma = 2 s - 1`),
//! `F45 = 2 e w_1' + 43 e^2 - 43 e` (equivalent to `e (e + 1) w_1' = 0` and
//! `e (e - 1)(w_1' - 43) = 0` once `F1`, `F2` and stage 2's `F3` give `e in {-1, 0, 1}`), and
//! the hint-weight identity `F7`, `sum_i eq(r^(2)_sig, i) (sum_pos delta^2 + sum_hi u_h) =
//! omega`, one instance with two parts at [`EqPoint::HalfR2`] (`delta^2` on the main domain
//! with scale `2^(d_main - d_sig)`, `u_h = U + 2^15` on `D_2` with scale `2^(d_2 - d_sig)`);
//! after the 37 reductions the reductions of stage 2's `delta(r^(2))`, `e(r^(2))`. The ring
//! reads `u = (2 s - 1) L + s - 2 gamma_2 delta - e` with `L = l_lo + 29,696 l_hi + 2^15`:
//! affine terms and the two products `s l_lo`, `s l_hi` (parts with scales -2 and -59,392).
//! H-FIPS has 51 coefficients.

use agg_witness::params::{GAMMA2, OMEGA, Q, W1_BITS};
use jolt_field::{One, Ring, Zero};

use crate::claims::{Kappa1Claims, PreStageClaims, Stage1Claims, Stage2Claims, Stage3Claims};
use crate::kernel::Term;
use crate::lookup::LookupClaims;
use crate::manifest::{
    value_columns, Column, CommitmentManifest, FipsColumn, GroupRole, L16_SLACK, LOOKUP_SHIFT, STACK_DOMAINS,
    U16_DIGITS, V16_DIGITS,
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
/// Lookup stacks (reductions).
pub const STACKS: usize = 4;

/// Gamma index of `ring`.
pub const GAMMA_RING: usize = 0;

/// H-FIPS checks of stage 3 (`F1`, `F2`, `F45`, `F7`).
pub const FIPS_CHECKS: usize = 4;
/// H-FIPS reductions of stage 3 (stage 2's `delta(r^(2))`, `e(r^(2))`).
pub const FIPS_REDUCTIONS: usize = 2;

/// The `gamma^(3)` layout of one relation: `links` leaf-checked digits (2 under H-u, 1 under
/// H-exact, 0 under H-FIPS), `bools` Boolean value digits (0 under H-u, 2 under H-exact, 3
/// under H-FIPS) and the H-FIPS checks and reductions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gammas {
    /// Link instances.
    pub links: usize,
    /// Booleanity zero-checks.
    pub bools: usize,
    /// Relation H-FIPS.
    pub fips: bool,
}

impl Gammas {
    /// The layout of `manifest`'s relation.
    #[must_use]
    pub fn new(manifest: &CommitmentManifest) -> Self {
        Self {
            links: manifest.leaf_digits().len(),
            bools: manifest.bool_digits().len(),
            fips: manifest.is_fips(),
        }
    }

    const fn fips_checks(self) -> usize {
        if self.fips {
            FIPS_CHECKS
        } else {
            0
        }
    }

    const fn fips_reductions(self) -> usize {
        if self.fips {
            FIPS_REDUCTIONS
        } else {
            0
        }
    }

    /// First reduction index `o = 1 + links + bools (+ 4 H-FIPS checks)`.
    const fn first_reduction(self) -> usize {
        1 + self.links + self.bools + self.fips_checks()
    }

    /// Batching coefficients: 44 under H-u, 45 under H-exact, 51 under H-FIPS.
    #[must_use]
    pub const fn count(self) -> usize {
        self.first_reduction() + 6 * W1_BITS + 1 + self.fips_reductions() + STACKS
    }

    /// Gamma index of H-FIPS check `k` (`F1, F2, F45, F7`).
    #[must_use]
    pub const fn fips_check(self, k: usize) -> usize {
        1 + self.links + self.bools + k
    }

    /// Gamma index of H-FIPS reduction `k` (`delta`, `e` from `r^(2)`).
    #[must_use]
    pub const fn fips_reduction(self, k: usize) -> usize {
        self.first_reduction() + 1 + 6 * W1_BITS + k
    }

    /// Gamma index of `link[k]`.
    #[must_use]
    pub const fn link(self, k: usize) -> usize {
        1 + k
    }

    /// Gamma index of `bool[k]`.
    #[must_use]
    pub const fn bool(self, k: usize) -> usize {
        1 + self.links + k
    }

    /// Gamma index of the reduction of `b_t` from `r^(1)`.
    #[must_use]
    pub const fn b_r1(self, t: usize) -> usize {
        self.first_reduction() + t
    }

    /// Gamma index of the reduction of `b_t` from `r^(2)`.
    #[must_use]
    pub const fn b_r2(self, t: usize) -> usize {
        self.first_reduction() + W1_BITS + t
    }

    /// Gamma index of the reduction of `y^(2)_t` from `tau_2`.
    #[must_use]
    pub const fn y2_tau(self, t: usize) -> usize {
        self.first_reduction() + 2 * W1_BITS + t
    }

    /// Gamma index of the reduction of `y^(2)_t` from `r^(1)_2`.
    #[must_use]
    pub const fn y2_r1(self, t: usize) -> usize {
        self.first_reduction() + 3 * W1_BITS + t
    }

    /// Gamma index of the reduction of `K^(2)_t` from `tau_2`.
    #[must_use]
    pub const fn kappa2_tau(self, t: usize) -> usize {
        self.first_reduction() + 4 * W1_BITS + t
    }

    /// Gamma index of the reduction of `K^(3)` from `tau_3`.
    #[must_use]
    pub const fn kappa3_tau(self) -> usize {
        self.first_reduction() + 5 * W1_BITS
    }

    /// Gamma index of the reduction of `K^(1)_t` from `r^(1)_1`.
    #[must_use]
    pub const fn kappa1_r1(self, t: usize) -> usize {
        self.first_reduction() + 1 + 5 * W1_BITS + t
    }

    /// Gamma index of the reduction of lookup stack `j`.
    #[must_use]
    pub const fn stack(self, j: usize) -> usize {
        self.first_reduction() + 1 + 6 * W1_BITS + self.fips_reductions() + j
    }
}

/// Batching coefficients of `manifest`'s relation ([`Gammas::count`]).
#[must_use]
pub fn gamma_count(manifest: &CommitmentManifest) -> usize {
    Gammas::new(manifest).count()
}

/// The stage-3 proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3Proof {
    /// Compressed degree-3 messages (3 elements each).
    pub rounds: Vec<Vec<E>>,
    /// The opening claims at `r^(3)` and its suffixes (31 under H-u, 32 under H-exact).
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
    let gammas = Gammas::new(manifest);
    let z_columns = value_columns(manifest.z_value());
    let mut groups = Vec::new();

    // Main domain, eq(r^(2)): the ring's committed part (u, v, w_1'), the links, the
    // Booleanity zero-checks and the b_t reductions from r^(2). The high digits of z
    // (Z_hi under H-u; bz_hi, a_z under H-exact) ride along for the link or the zero-checks;
    // H-FIPS reads l_lo, l_hi in the slots of U_lo, U_hi and adds s, delta, e.
    let u_columns = value_columns(manifest.u_slot_value());
    let mut main_columns = vec![u_columns[0], Column::V_LO, Column::V_HI, z_columns[1], u_columns[1]];
    main_columns.extend(z_columns[2..].iter().copied());
    main_columns.extend(lanes.clone().map(|t| Column::W1Bit(t as u8)));
    if gammas.fips {
        main_columns.extend(FipsColumn::ALL.map(Column::Fips));
    }
    let position = |column: Column| main_columns.iter().position(|&c| c == column).expect("lane");
    let v_weights = V16_DIGITS.weights();
    let mut ring_b: Vec<(usize, i64)> = (0..W1_BITS)
        .map(|t| (position(Column::W1Bit(t as u8)), -((2 * GAMMA2) << t)))
        .collect();
    ring_b.push((position(Column::V_LO), -(Q * v_weights[0])));
    ring_b.push((position(Column::V_HI), -(Q * v_weights[1])));
    let mut instances = Vec::new();
    if gammas.fips {
        // -u = -(2 s - 1)(l_lo + 29,696 l_hi + 2^15) - s + 2 gamma_2 delta + e: the affine part
        // here, the products -2 s l_lo and -59,392 s l_hi as scaled parts.
        let (sign, delta, wrap) = (position(Column::SIGN), position(Column::DELTA), position(Column::WRAP));
        let (l_lo, l_hi) = (position(Column::L_LO), position(Column::L_HI));
        let step = L16_SLACK.slack_weight;
        ring_b.push((l_lo, 1));
        ring_b.push((l_hi, step));
        ring_b.push((sign, -(2 * i64::from(LOOKUP_SHIFT) + 1)));
        ring_b.push((delta, 2 * GAMMA2));
        ring_b.push((wrap, 1));
        instances.push(InstanceSpec::new(
            GAMMA_RING,
            vec![Term::Affine {
                weight: None,
                lanes: ring_b,
                constant: i64::from(LOOKUP_SHIFT) - Q * V16_DIGITS.constant(),
            }],
        ));
        for (lane, scale) in [(l_lo, -2), (l_hi, -2 * step)] {
            instances.push(InstanceSpec {
                gamma: GAMMA_RING,
                scale: E::from_i64(scale),
                terms: vec![Term::Product(vec![sign, lane])],
            });
        }
    } else {
        let u_weights = U16_DIGITS.weights();
        ring_b.push((position(Column::U_LO), -u_weights[0]));
        ring_b.push((position(Column::U_HI), -u_weights[1]));
        instances.push(InstanceSpec::new(
            GAMMA_RING,
            vec![Term::Affine {
                weight: None,
                lanes: ring_b,
                constant: -(U16_DIGITS.constant() + Q * V16_DIGITS.constant()),
            }],
        ));
    }
    for (k, digit) in all.iter().enumerate() {
        instances.push(InstanceSpec::new(gammas.link(k), vec![Term::Link(position(digit.column))]));
    }
    for (k, &column) in manifest.bool_digits().iter().enumerate() {
        instances.push(InstanceSpec::new(gammas.bool(k), vec![Term::Bool(position(column))]));
    }
    if gammas.fips {
        let (sign, delta, wrap) = (position(Column::SIGN), position(Column::DELTA), position(Column::WRAP));
        // F1 = s (s - 1).
        instances.push(InstanceSpec::new(gammas.fips_check(0), vec![Term::Bool(sign)]));
        // F2 = delta^2 + delta - 2 delta s.
        instances.push(InstanceSpec::new(
            gammas.fips_check(1),
            vec![Term::Product(vec![delta, delta]), Term::Product(vec![delta])],
        ));
        instances.push(InstanceSpec {
            gamma: gammas.fips_check(1),
            scale: E::from_i64(-2),
            terms: vec![Term::Product(vec![delta, sign])],
        });
        // F45 = sum_t 2^(t+1) e b_t + 43 e^2 - 43 e.
        let f45 = gammas.fips_check(2);
        for t in 0..W1_BITS {
            instances.push(InstanceSpec {
                gamma: f45,
                scale: E::from_i64(2 << t),
                terms: vec![Term::Product(vec![wrap, position(Column::W1Bit(t as u8))])],
            });
        }
        let top = 43;
        instances.push(InstanceSpec {
            gamma: f45,
            scale: E::from_i64(top),
            terms: vec![Term::Product(vec![wrap, wrap])],
        });
        instances.push(InstanceSpec {
            gamma: f45,
            scale: E::from_i64(-top),
            terms: vec![Term::Product(vec![wrap])],
        });
        // The reductions of stage 2's delta(r^(2)) and e(r^(2)).
        for (k, lane) in [delta, wrap].into_iter().enumerate() {
            instances.push(InstanceSpec::new(gammas.fips_reduction(k), vec![Term::Product(vec![lane])]));
        }
    }
    instances.extend(lanes.clone().map(|t| {
        InstanceSpec::new(gammas.b_r2(t), vec![Term::Product(vec![position(Column::W1Bit(t as u8))])])
    }));
    groups.push(GroupSpec {
        domain: GroupRole::Main,
        eq: EqPoint::R2,
        lanes: column_lanes(&main_columns),
        instances,
    });
    if gammas.fips {
        // F7: sum_pos delta^2 (main) plus sum_hi u_h (D_2), both at (1/2, ..., r^(2)_sig).
        let gamma = gammas.fips_check(3);
        let lifted = |role: GroupRole| -> E {
            E::from_u64(1_u64 << (manifest.group(role).logical_vars - manifest.d_sig()))
        };
        groups.push(GroupSpec {
            domain: GroupRole::Main,
            eq: EqPoint::HalfR2,
            lanes: vec![LaneId::Column(Column::DELTA)],
            instances: vec![InstanceSpec {
                gamma,
                scale: lifted(GroupRole::Main),
                terms: vec![Term::Product(vec![0, 0])],
            }],
        });
        groups.push(GroupSpec {
            domain: GroupRole::Level2,
            eq: EqPoint::HalfR2,
            lanes: vec![LaneId::Column(Column::HINT_SLACK)],
            instances: vec![InstanceSpec {
                gamma,
                scale: lifted(GroupRole::Level2),
                terms: vec![Term::Affine {
                    weight: None,
                    lanes: vec![(0, 1)],
                    constant: i64::from(LOOKUP_SHIFT),
                }],
            }],
        });
    }

    // Main domain, no equality factor: w_A times z (z = Z_lo + 2^16 Z_hi + 2^15 under H-u,
    // z = S_zlo + 2^16 B_zhi + 130,915 a_z - 98,225 under H-exact).
    let z_form = manifest.z_value().layout();
    groups.push(GroupSpec {
        domain: GroupRole::Main,
        eq: EqPoint::None,
        lanes: std::iter::once(LaneId::Weight(WeightId::Ring))
            .chain(column_lanes(&z_columns))
            .collect(),
        instances: vec![InstanceSpec::new(
            GAMMA_RING,
            vec![Term::Affine {
                weight: Some(0),
                lanes: z_form.weights().into_iter().enumerate().map(|(k, w)| (1 + k, w)).collect(),
                constant: z_form.constant(),
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
        instances: reductions(&bits, |t| gammas.b_r1(t)),
        lanes: bits,
    });

    // D_1, eq(r^(1)_1): K^(1)_t reductions.
    let kappa1 = column_lanes(&manifest.stack_columns(1));
    groups.push(GroupSpec {
        domain: GroupRole::Level1,
        eq: EqPoint::R1,
        instances: reductions(&kappa1, |t| gammas.kappa1_r1(t)),
        lanes: kappa1,
    });

    // D_2, eq(tau_2): y^(2)_t and K^(2)_t reductions; eq(r^(1)_2): y^(2)_t.
    let kappa2: Vec<LaneId> = lanes.clone().map(|t| LaneId::Column(Column::carry(2, t as u8))).collect();
    let y2: Vec<LaneId> = lanes
        .clone()
        .map(|t| LaneId::Column(Column::Parity2(t as u8)))
        .collect();
    let mut tau2_lanes = y2.clone();
    tau2_lanes.extend_from_slice(&kappa2);
    let mut tau2_instances = reductions(&y2, |t| gammas.y2_tau(t));
    tau2_instances.extend(
        (0..kappa2.len())
            .map(|l| InstanceSpec::new(gammas.kappa2_tau(l), vec![Term::Product(vec![W1_BITS + l])])),
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
        instances: reductions(&y2, |t| gammas.y2_r1(t)),
        lanes: y2,
    });

    // D_3, eq(tau_3): K^(3).
    let kappa3 = column_lanes(&manifest.stack_columns(3));
    groups.push(GroupSpec {
        domain: GroupRole::Level3,
        eq: EqPoint::Tau,
        instances: reductions(&kappa3, |_| gammas.kappa3_tau()),
        lanes: kappa3,
    });

    // The lookup stacks: one equality group per stack at its low point, one part per column
    // with the column weight as scale.
    for (j, stack_weights) in weights.iter().enumerate() {
        let columns = manifest.stack_columns(j);
        assert_eq!(stack_weights.len(), columns.len(), "one weight per stack column");
        groups.push(GroupSpec {
            domain: STACK_DOMAINS[j],
            eq: EqPoint::Lookup(j),
            lanes: column_lanes(&columns),
            instances: stack_weights
                .iter()
                .enumerate()
                .map(|(c, &scale)| InstanceSpec {
                    gamma: gammas.stack(j),
                    scale,
                    terms: vec![Term::Product(vec![c])],
                })
                .collect(),
        });
    }
    StageSpec {
        stage: STAGE,
        degree: DEGREE,
        gamma_count: gammas.count(),
        groups,
    }
}

/// The claimed sums in gamma order (the Booleanity zero-checks claim zero); `stacks[j]` is
/// stack `j`'s column claim ([`stack_reductions`]).
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn claimed_sums(
    manifest: &CommitmentManifest,
    t_at_r2: E,
    prestage: &PreStageClaims,
    stage1: &Stage1Claims,
    kappa1: &Kappa1Claims,
    stage2: &Stage2Claims,
    stacks: &[E],
) -> Vec<E> {
    let gammas = Gammas::new(manifest);
    let mut claims = vec![E::zero(); gammas.count()];
    claims[GAMMA_RING] = t_at_r2;
    for (k, &square) in stage2.squares.iter().enumerate() {
        claims[gammas.link(k)] = square;
    }
    if gammas.fips {
        claims[gammas.fips_check(3)] = E::from_u64(OMEGA as u64);
        for (k, &value) in stage2.fips.iter().enumerate() {
            claims[gammas.fips_reduction(k)] = value;
        }
    }
    for t in 0..W1_BITS {
        claims[gammas.b_r1(t)] = stage1.b[t];
        claims[gammas.b_r2(t)] = stage2.b[t];
        claims[gammas.y2_tau(t)] = prestage.y2[t];
        claims[gammas.y2_r1(t)] = stage1.y2[t];
        claims[gammas.kappa2_tau(t)] = prestage.kappa2[t];
        claims[gammas.kappa1_r1(t)] = kappa1.0[t];
    }
    claims[gammas.kappa3_tau()] = prestage.kappa3;
    for (j, &claim) in stacks.iter().enumerate() {
        claims[gammas.stack(j)] = claim;
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
        manifest,
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
        manifest,
        E::zero(),
        prestage,
        &stage1.claims,
        &stage2.kappa1,
        &stage2.claims,
        &lookup.claims,
    );
    let mut replay = replay_stage(&spec, manifest, &claims, &proof.rounds, transcript)?;
    if proof.claims.0.len() != Stage3Claims::len(manifest) {
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
        check_spec(&CommitmentManifest::for_batch(8).unwrap(), 44);
        let gammas = Gammas::new(&CommitmentManifest::for_batch(8).unwrap());
        assert_eq!((gammas.kappa1_r1(5), gammas.kappa3_tau(), gammas.stack(3)), (39, 33, 43));
    }

    #[test]
    fn h_exact_stage_three_has_forty_five_instances_with_two_bool_checks() {
        let manifest =
            CommitmentManifest::for_relation(8, crate::manifest::Relation::HExact).unwrap();
        let spec = check_spec(&manifest, 45);
        let gammas = Gammas::new(&manifest);
        assert_eq!((gammas.links, gammas.bools), (1, 2));
        assert_eq!((gammas.bool(0), gammas.bool(1), gammas.b_r1(0)), (2, 3, 4));
        assert_eq!((gammas.kappa3_tau(), gammas.stack(3)), (34, 44));
        // docs/h-exact.md 3.4: link[u_hi] at 1, bool[bz_hi] at 2, bool[a_z] at 3, all in the
        // eq(r^(2)) group; the w_A group reads bz_lo, bz_hi, a_z with weights (1, 2^16,
        // 130,915) and constant -98,225.
        let main = &spec.groups[0];
        let lane = |c| main.lane(LaneId::Column(c));
        let terms: Vec<(usize, Vec<Term>)> =
            main.instances[1..4].iter().map(|i| (i.gamma, i.terms.clone())).collect();
        assert_eq!(
            terms,
            [
                (1, vec![Term::Link(lane(Column::U_HI))]),
                (2, vec![Term::Bool(lane(Column::ZX_HI))]),
                (3, vec![Term::Bool(lane(Column::A_Z))]),
            ]
        );
        let ring = &spec.groups[1];
        assert_eq!(ring.lanes.len(), 4);
        assert_eq!(
            ring.instances[0].terms,
            vec![Term::Affine {
                weight: Some(0),
                lanes: vec![(1, 1), (2, 1 << 16), (3, 130_915)],
                constant: -98_225,
            }]
        );
    }

    #[test]
    fn h_fips_stage_three_has_fifty_one_instances() {
        let manifest = CommitmentManifest::for_relation(8, crate::manifest::Relation::HFips).unwrap();
        let mut rng = SplitMix64::new(4);
        let weights: Vec<Vec<E>> = (0..STACKS)
            .map(|j| point(&mut rng, manifest.stack_columns(j).len()))
            .collect();
        let spec = spec(&manifest, &weights);
        spec.validate().unwrap();
        let gammas = Gammas::new(&manifest);
        assert_eq!((gammas.links, gammas.bools, gammas.fips), (0, 3, true));
        assert_eq!(gammas.count(), 51);
        assert_eq!(spec.gamma_count, 51);
        // ring 0, bool[bz_hi, l_hi, a_z] 1..3, F1, F2, F45, F7 4..7, reductions from 8, the
        // delta and e reductions 45, 46, the stacks 47..50.
        assert_eq!((gammas.fips_check(0), gammas.fips_check(3), gammas.b_r1(0)), (4, 7, 8));
        assert_eq!((gammas.fips_reduction(0), gammas.fips_reduction(1)), (45, 46));
        assert_eq!((gammas.stack(0), gammas.stack(3)), (47, 50));
        // Parts per gamma index: the ring spans the eq(r^(2)) group (affine part and the two
        // products s l_lo, s l_hi) and the w_A group; F2 two parts, F45 eight, F7 two (main and
        // D_2), the stacks one per column (u_h is stack 2's seventh column).
        let mut counts = vec![0; 51];
        for group in &spec.groups {
            for instance in &group.instances {
                counts[instance.gamma] += 1;
            }
        }
        assert_eq!(counts[GAMMA_RING], 4);
        assert_eq!(counts[gammas.fips_check(1)], 2);
        assert_eq!(counts[gammas.fips_check(2)], 8);
        assert_eq!(counts[gammas.fips_check(3)], 2);
        assert_eq!(
            (0..STACKS).map(|j| counts[gammas.stack(j)]).collect::<Vec<_>>(),
            [4, 6, 7, 1]
        );
        for group in &spec.groups {
            let inner = usize::from(group.eq != EqPoint::None);
            for instance in &group.instances {
                for term in &instance.terms {
                    assert!(term.degree() + inner <= DEGREE, "{term:?}");
                }
            }
        }
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
        // F7's two parts at (1/2, ..., r^(2)_sig) with scales 2^(d_main - d_sig) = 2^10 and
        // 2^(d_2 - d_sig) = 2^10 (d_sig = 3, d_2 = 13).
        let f7: Vec<&GroupSpec> = spec.groups.iter().filter(|g| g.eq == EqPoint::HalfR2).collect();
        assert_eq!(f7.len(), 2);
        assert_eq!((f7[0].domain, f7[1].domain), (GroupRole::Main, GroupRole::Level2));
        for group in f7 {
            assert_eq!(group.instances[0].scale, E::from_u64(1 << 10));
        }
    }

    fn check_spec(manifest: &CommitmentManifest, count: usize) -> StageSpec {
        let mut rng = SplitMix64::new(3);
        let weights: Vec<Vec<E>> = (0..STACKS)
            .map(|j| point(&mut rng, manifest.stack_columns(j).len()))
            .collect();
        let spec = spec(manifest, &weights);
        spec.validate().unwrap();
        let gammas = Gammas::new(manifest);
        assert_eq!(gammas.count(), count);
        assert_eq!(spec.gamma_count, count);
        assert_eq!(spec.groups.len(), 7 + STACKS);
        // The ring instance spans two groups and each stack reduction one part per column;
        // every other gamma index appears once.
        let mut counts = vec![0; count];
        for group in &spec.groups {
            for instance in &group.instances {
                counts[instance.gamma] += 1;
            }
        }
        assert_eq!(counts[GAMMA_RING], 2);
        assert_eq!(
            (0..STACKS).map(|j| counts[gammas.stack(j)]).collect::<Vec<_>>(),
            [4, 6, 6, 1]
        );
        assert!(counts[1..gammas.stack(0)].iter().all(|&c| c == 1));
        assert_eq!(gammas.stack(STACKS - 1), count - 1);
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
        spec
    }
}
