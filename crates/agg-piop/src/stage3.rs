//! Stage 3 (degree 3): the ring identity with `(A z)` pulled back through the public
//! matrices, the range links, the Booleanity zero-checks of H-exact's bit digits, the
//! evaluation reductions to `r^(3)`, the five lookup-stack reductions and
//! the two sides of the projection identity of `v` (`projection`).
//!
//! `gamma^(3)` order ([`Gammas`], with `o = 1 + links + bools`): `ring` (0), one `link` per
//! leaf-checked digit (H-u: `link[Z_hi]` 1, `link[U_hi]` 2, `link[K_2]` 3, `link[K_3]` 4;
//! H-exact: `link[U_hi]` 1, `link[K_2]` 2, `link[K_3]` 3), one `bool` per Boolean value digit (H-exact: `bool[bz_hi]`
//! 4, `bool[a_z]` 5; claim 0), then the reductions: `b_t` from `r^(1)`
//! (`o..o+6`), `b_t` from `r^(2)`, `y^(2)_t` from `tau_2`, `y^(2)_t` from `r^(1)_2`, `K^(2)_t`
//! from `tau_2`, `K^(3)` from `tau_3`, `K^(1)_t` from `r^(1)_1`, the lookup stacks
//! `j = 0..5`: `sum_x eq(pi_j, x) sum_c w_{j,c} f_c(x) = g_j(pi_j, pi_top) + 2^15 sum_{c >= cols}
//! w_{j,c}` with `w_{j,c} = eq(pi_top, c)` (`StackShape::column_claim`), one part per column
//! with scale `w_{j,c}`, and last `proj_v`, `proj_k`: `sum_x W(x) (V_lo + 2^16 V_hi + 2^15)(x)`
//! on the main domain (the weight lane `W`, no equality factor) and `sum_y eq(r', y) (K_0 +
//! 2^16 K_1 + 2^32 K_2 + 2^35 K_3 + 2^34 + 2^31 + 2^15)(y)` on `D_k` at the `D_k` suffix
//! `r'` of `r^(2)`, both
//! with the claim `c` sent after stage 2, so that `k` is bound to `v`. For ML-DSA-44, H-u has
//! 49 coefficients and H-exact 50.
//!
//! The ring instance is one instance with two summands on two groups: `w_A z` (weight
//! lane, no equality factor) and `-eq(r^(2), x)(2 gamma_2 w_1' + u + q v)` (equality
//! group of `r^(2)`), both affine in the committed columns: `z = Z_lo + 2^16 Z_hi + 2^15`
//! (H-u) or `z = S_zlo + 2^16 B_zhi + 130,915 a_z - 98,225` (H-exact), `u = U_lo + 63,079
//! U_hi + 31,540`, `v = V_lo + 2^16 V_hi + 2^15`.
//!
//! Relation H-FIPS: `link[K_2]`, `link[K_3]`, `bool` for `bz_hi`, `l_hi`, `a_z`, then
//! four H-FIPS checks after the bools ([`Gammas::fips_check`]): `F_s = s (s - 1)`,
//! `F_delta = delta (delta - 2 s + 1)` (`delta (delta - sigma)` with `sigma = 2 s - 1`),
//! `F_w1 = 2 e w_1' + 43 e^2 - 43 e` (equivalent to `e (e + 1) w_1' = 0` and
//! `e (e - 1)(w_1' - 43) = 0` once `F_s`, `F_delta` and stage 2's `F_e` give `e in {-1, 0, 1}`), and
//! the hint-weight identity `F_omega`, `sum_i eq(r^(2)_sig, i) (sum_pos delta^2 + sum_hi u_h) =
//! omega`, one instance with two parts at [`EqPoint::HalfR2`] (`delta^2` on the main domain
//! with scale `2^(d_main - d_sig)`, `u_h = U + 2^15` on `D_2` with scale `2^(d_2 - d_sig)`);
//! after the 37 reductions the reductions of stage 2's `delta(r^(2))`, `e(r^(2))`. The ring
//! reads the affine `u = gamma_2 s + L + 1 - gamma_2 - 2 gamma_2 delta - e` with
//! `L = l_lo + 29,696 l_hi + 2^15`. H-FIPS has 56 coefficients (ML-DSA-44).
//!
//! Polynomial lanes (ML-DSA-65, ML-DSA-87; relations H-u and H-exact, which differ only in
//! the constants of the affine `z`): the main domain has no row
//! coordinate, so the ring identity is `k` instances `ring[rho]` (`0..k`), each with the
//! claimed sum `T~_rho(r^(2))` and the summand `-eq(r^(2), x)(2 gamma_2 w_1' + u + q v)^(rho)`;
//! their `A z` parts are one unbatched part per lane `s` of `z`, `w_A^(s) z^(s)`, whose weight
//! combines the rows by the coefficients `gamma_rho` of the instances (`lane_public`).
//! Then `l + k + 3` links (`Z_mid^(s)`, `U_mid^(rho)`, `K_2`, `K_3`, `K_4`), `l + k` Booleanity
//! checks (`a_z^(s)`, `a_u^(rho)`), the reductions of the `4 k` bit polynomials from `r^(1)`
//! and from `r^(2)`, the level reductions over four bits, the stacks of the set (7 or 8), and
//! the projection identity, whose `v` side has one weight `W^(rho)` per lane of `v`
//! (`projection::lane_weight_tables`): 105 coefficients for ML-DSA-65, 132 for ML-DSA-87.
//! Relation H-FIPS with lanes: the links lose `U_mid^(rho)` (the magnitude `L` of row `rho`
//! has the bits `l_mid^(rho)`, `a_l^(rho)`, checked by Booleanity zero-checks), and the H-FIPS
//! checks are `F_s^(rho)`, `F_delta^(rho)`, `F_w1^(rho)` per row lane (with the four bits of
//! the row and `m_w - 1 = 15`), one `F_omega` over every lane of `delta` and `u_h` (claim
//! `omega` of the set), and the reductions of stage 2's `delta^(rho)(r^(2))`, `e^(rho)(r^(2))`:
//! 136 coefficients for ML-DSA-65, 173 for ML-DSA-87.

use agg_witness::params::{N, Q};
use jolt_field::{One, Ring, Zero};

use crate::claims::{bit_columns, Kappa1Claims, PreStageClaims, Stage1Claims, Stage2Claims, Stage3Claims};
use crate::kernel::{Lane, Source, Term};
use crate::lane_public::LanePublicData;
use crate::lookup::LookupClaims;
use crate::manifest::{value_columns, Column, CommitmentManifest, FipsColumn, GroupRole, LOOKUP_SHIFT, V16_DIGITS};
use crate::range::digits;
use crate::stage::{
    check_terminal, draw_gammas, prove_stage, prove_stage_drawn, prove_stage_unchecked, replay_stage, EqPoint,
    GroupSpec, InstanceSpec, LaneId, Points, StageError, StageSpec, WeightId,
};
use crate::stage1::Stage1Output;
use crate::stage2::Stage2Output;
use crate::tables::ProverTables;
use crate::transcript::{terminal_site, Transcript};
use crate::OpeningField as E;

/// Stage number.
pub const STAGE: usize = 3;
/// Message degree.
pub const DEGREE: usize = 3;
/// Lookup stacks (reductions) of ML-DSA-44 ([`Gammas::stacks`] for every set).
pub const STACKS: usize = 5;
/// The two sides of the projection identity (`projection`): the weighted sum of `v` and the
/// reduction of `k` at `r'`.
pub const PROJ_INSTANCES: usize = 2;

/// Gamma index of `ring`.
pub const GAMMA_RING: usize = 0;

/// H-FIPS checks of stage 3 of ML-DSA-44 (`F_s`, `F_delta`, `F_w1`, `F_omega`; `3 LANES_K + 1`
/// in general).
pub const FIPS_CHECKS: usize = 4;
/// H-FIPS reductions of stage 3 of ML-DSA-44 (stage 2's `delta(r^(2))`, `e(r^(2))`; `2 LANES_K`
/// in general).
pub const FIPS_REDUCTIONS: usize = 2;

/// The `gamma^(3)` layout of one parameter set and relation: `rings` ring instances (one,
/// or one per row `rho` with polynomial lanes), `links` leaf-checked digits (4 under H-u, 3
/// under H-exact, 2 under H-FIPS; `l + k + 3` with lanes), `bools` Boolean value digits (0
/// under H-u, 2 under H-exact, 3 under H-FIPS; `l + k` with lanes: `a_z^(s)`, `a_u^(rho)`),
/// the H-FIPS checks and reductions, `bits` bits of `w_1'` in `bit_lanes` polynomials, and
/// `stacks` lookup stacks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gammas {
    /// Ring instances (`LANES_K`).
    pub rings: usize,
    /// Link instances.
    pub links: usize,
    /// Booleanity zero-checks.
    pub bools: usize,
    /// Relation H-FIPS.
    pub fips: bool,
    /// Row lanes (`LANES_K`): the H-FIPS checks and reductions are per row lane.
    pub lanes: usize,
    /// Bits of `w_1'`.
    pub bits: usize,
    /// Bit polynomials `b_t^(rho)` (`bits` times the polynomial lanes).
    pub bit_lanes: usize,
    /// Lookup stacks.
    pub stacks: usize,
}

impl Gammas {
    /// The layout of `manifest`'s parameter set and relation.
    #[must_use]
    pub fn new(manifest: &CommitmentManifest) -> Self {
        let shape = manifest.shape();
        Self {
            rings: shape.lanes_k,
            links: manifest.leaf_digits().len(),
            bools: manifest.bool_digits().len(),
            fips: manifest.is_fips(),
            lanes: shape.lanes_k,
            bits: shape.w1_bits,
            bit_lanes: shape.w1_bits * shape.lanes_k,
            stacks: manifest.lookup_shape().stacks.len(),
        }
    }

    const fn fips_checks(self) -> usize {
        if self.fips {
            3 * self.lanes + 1
        } else {
            0
        }
    }

    const fn fips_reductions(self) -> usize {
        if self.fips {
            2 * self.lanes
        } else {
            0
        }
    }

    /// First reduction index `o = rings + links + bools (+ 4 H-FIPS checks)`.
    const fn first_reduction(self) -> usize {
        self.rings + self.links + self.bools + self.fips_checks()
    }

    /// The reductions of the level parities and carries start after the two bit reductions.
    const fn level_reductions(self) -> usize {
        self.first_reduction() + 2 * self.bit_lanes
    }

    /// Batching coefficients: for ML-DSA-44 49 under H-u, 50 under H-exact, 56 under H-FIPS;
    /// 105 for ML-DSA-65 and 132 for ML-DSA-87 (136 and 173 under H-FIPS).
    #[must_use]
    pub const fn count(self) -> usize {
        self.stack(self.stacks) + PROJ_INSTANCES
    }

    /// Gamma index of `ring[rho]` (`ring` at 0 for ML-DSA-44).
    #[must_use]
    pub const fn ring(self, rho: usize) -> usize {
        GAMMA_RING + rho
    }

    /// Gamma index of the projection identity's `v` side, `sum_x W(x) v(x) = c`.
    #[must_use]
    pub const fn proj_v(self) -> usize {
        self.stack(self.stacks)
    }

    /// Gamma index of the projection identity's `k` side, `sum_y eq(r', y) k(y) = c`.
    #[must_use]
    pub const fn proj_k(self) -> usize {
        self.stack(self.stacks) + 1
    }

    /// Gamma index of H-FIPS check `k` of ML-DSA-44 (`F_s, F_delta, F_w1, F_omega`; with one
    /// row lane, [`Gammas::fips_lane_check`] and [`Gammas::fips_omega`] give the same indices).
    #[must_use]
    pub const fn fips_check(self, k: usize) -> usize {
        self.rings + self.links + self.bools + k
    }

    /// Gamma index of H-FIPS check `k` (`F_s`, `F_delta`, `F_w1` for `k = 0, 1, 2`) of row lane
    /// `rho`: every lane's `F_s`, then every lane's `F_delta`, then every lane's `F_w1`.
    #[must_use]
    pub const fn fips_lane_check(self, k: usize, rho: usize) -> usize {
        self.fips_check(k * self.lanes + rho)
    }

    /// Gamma index of `F_omega`, after the per-lane checks.
    #[must_use]
    pub const fn fips_omega(self) -> usize {
        self.fips_check(3 * self.lanes)
    }

    /// Gamma index of H-FIPS reduction `k` of ML-DSA-44 (`delta`, `e` from `r^(2)`).
    #[must_use]
    pub const fn fips_reduction(self, k: usize) -> usize {
        self.level_reductions() + 4 * self.bits + 1 + k
    }

    /// Gamma index of the H-FIPS reduction of stage 2's terminal claim `k` (the order of
    /// `stage2::fips_terminals`: `delta^(rho)` for every lane, then `e^(rho)`).
    #[must_use]
    pub const fn fips_terminal_reduction(self, k: usize) -> usize {
        self.fips_reduction(k)
    }

    /// Gamma index of `link[k]`.
    #[must_use]
    pub const fn link(self, k: usize) -> usize {
        self.rings + k
    }

    /// Gamma index of `bool[k]`.
    #[must_use]
    pub const fn bool(self, k: usize) -> usize {
        self.rings + self.links + k
    }

    /// Gamma index of the reduction of bit polynomial `i` (`claims::bit_columns` order)
    /// from `r^(1)`.
    #[must_use]
    pub const fn b_r1(self, i: usize) -> usize {
        self.first_reduction() + i
    }

    /// Gamma index of the reduction of bit polynomial `i` from `r^(2)`.
    #[must_use]
    pub const fn b_r2(self, i: usize) -> usize {
        self.first_reduction() + self.bit_lanes + i
    }

    /// Gamma index of the reduction of `y^(2)_t` from `tau_2`.
    #[must_use]
    pub const fn y2_tau(self, t: usize) -> usize {
        self.level_reductions() + t
    }

    /// Gamma index of the reduction of `y^(2)_t` from `r^(1)_2`.
    #[must_use]
    pub const fn y2_r1(self, t: usize) -> usize {
        self.level_reductions() + self.bits + t
    }

    /// Gamma index of the reduction of `K^(2)_t` from `tau_2`.
    #[must_use]
    pub const fn kappa2_tau(self, t: usize) -> usize {
        self.level_reductions() + 2 * self.bits + t
    }

    /// Gamma index of the reduction of `K^(3)` from `tau_3`.
    #[must_use]
    pub const fn kappa3_tau(self) -> usize {
        self.level_reductions() + 3 * self.bits
    }

    /// Gamma index of the reduction of `K^(1)_t` from `r^(1)_1`.
    #[must_use]
    pub const fn kappa1_r1(self, t: usize) -> usize {
        self.level_reductions() + 3 * self.bits + 1 + t
    }

    /// Gamma index of the reduction of lookup stack `j`.
    #[must_use]
    pub const fn stack(self, j: usize) -> usize {
        self.level_reductions() + 4 * self.bits + 1 + self.fips_reductions() + j
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
    /// The opening claims at `r^(3)` and its suffixes (for ML-DSA-44: 35 under H-u, 36 under
    /// H-exact, 40 under H-FIPS; `Stage3Claims::len` in general).
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
    let gammas = Gammas::new(manifest);
    assert_eq!(weights.len(), gammas.stacks, "one weight vector per lookup stack");
    let shape = manifest.shape();
    let (lanes_k, lanes_l) = (shape.lanes_k as u8, shape.lanes_l as u8);
    let all = digits(manifest);
    let (z_value, u_value) = (manifest.z_value(), manifest.u_slot_value());
    let (z_form, u_form) = (z_value.layout(), u_value.layout());
    let bits = bit_columns(manifest);
    let mut groups = Vec::new();

    // Main domain, eq(r^(2)): the ring's committed part (u, v, w_1') per row, the links,
    // the Booleanity zero-checks and the bit reductions from r^(2). The higher digits of z
    // (Z_hi under H-u; bz_hi, a_z under H-exact; Z_mid, a_z per lane) ride along for the
    // links or the zero-checks; H-FIPS reads l_lo, l_hi in the slots of U_lo, U_hi and adds
    // s, delta, e. Lane order: (U_lo, V_lo, V_hi) per row lane, the second digits of z
    // and u, their further digits, the bits.
    let mut main_columns = Vec::new();
    for lane in 0..lanes_k {
        main_columns.extend([
            Column::lane_digit(u_value, 0, lane),
            Column::V_LO.in_lane(lane),
            Column::V_HI.in_lane(lane),
        ]);
    }
    main_columns.extend((0..lanes_l).map(|s| Column::lane_digit(z_value, 1, s)));
    main_columns.extend((0..lanes_k).map(|rho| Column::lane_digit(u_value, 1, rho)));
    for digit in 2..z_form.len() as u8 {
        main_columns.extend((0..lanes_l).map(|s| Column::lane_digit(z_value, digit, s)));
    }
    for digit in 2..u_form.len() as u8 {
        main_columns.extend((0..lanes_k).map(|rho| Column::lane_digit(u_value, digit, rho)));
    }
    main_columns.extend(bits.iter().copied());
    if gammas.fips {
        for column in FipsColumn::ALL {
            main_columns.extend(manifest.fips_lanes(column));
        }
    }
    let position = |column: Column| main_columns.iter().position(|&c| c == column).expect("lane");
    let v_weights = V16_DIGITS.weights();
    let gamma2 = manifest.gamma2();
    let two_gamma2 = 2 * gamma2;
    let mut instances = Vec::new();
    for rho in 0..lanes_k {
        // ring[rho]: -(2 gamma_2 w_1' + u + q v) of row lane rho.
        let mut ring_b: Vec<(usize, i64)> = (0..shape.w1_bits as u8)
            .map(|t| (position(Column::W1Bit(t, rho)), -(two_gamma2 << t)))
            .collect();
        ring_b.push((position(Column::V_LO.in_lane(rho)), -(Q * v_weights[0])));
        ring_b.push((position(Column::V_HI.in_lane(rho)), -(Q * v_weights[1])));
        if gammas.fips {
            // -u = -(gamma_2 s + L + 1 - gamma_2) + 2 gamma_2 delta + e with the affine L of the
            // set (ML-DSA-44: l_lo + 29,696 l_hi + 2^15; lanes: l_lo + 2^16 l_mid + 130,816 a_l
            // + 2^15).
            let fips = |column: FipsColumn| position(Column::Fips(column, rho));
            for (digit, weight) in u_form.weights().into_iter().enumerate() {
                ring_b.push((position(Column::lane_digit(u_value, digit as u8, rho)), -weight));
            }
            ring_b.push((fips(FipsColumn::Sign), -gamma2));
            ring_b.push((fips(FipsColumn::Delta), two_gamma2));
            ring_b.push((fips(FipsColumn::Wrap), 1));
            instances.push(InstanceSpec::new(
                gammas.ring(usize::from(rho)),
                vec![Term::Affine {
                    weight: None,
                    lanes: ring_b,
                    constant: -(u_form.constant() + 1 - gamma2) - Q * V16_DIGITS.constant(),
                }],
            ));
        } else {
            for (digit, weight) in u_form.weights().into_iter().enumerate() {
                ring_b.push((position(Column::lane_digit(u_value, digit as u8, rho)), -weight));
            }
            instances.push(InstanceSpec::new(
                gammas.ring(usize::from(rho)),
                vec![Term::Affine {
                    weight: None,
                    lanes: ring_b,
                    constant: -(u_form.constant() + Q * V16_DIGITS.constant()),
                }],
            ));
        }
    }
    for (k, digit) in all.iter().enumerate() {
        if digit.column.domain() == GroupRole::Main {
            instances.push(InstanceSpec::new(gammas.link(k), vec![Term::Link(position(digit.column))]));
        }
    }
    for (k, &column) in manifest.bool_digits().iter().enumerate() {
        instances.push(InstanceSpec::new(gammas.bool(k), vec![Term::Bool(position(column))]));
    }
    if gammas.fips {
        let fips = |column: FipsColumn, rho: usize| position(Column::Fips(column, rho as u8));
        // F_s = s (s - 1), per row lane.
        for rho in 0..gammas.lanes {
            instances.push(InstanceSpec::new(gammas.fips_lane_check(0, rho), vec![Term::Bool(fips(FipsColumn::Sign, rho))]));
        }
        // F_delta = delta^2 + delta - 2 delta s, per row lane.
        for rho in 0..gammas.lanes {
            let (sign, delta) = (fips(FipsColumn::Sign, rho), fips(FipsColumn::Delta, rho));
            instances.push(InstanceSpec::new(
                gammas.fips_lane_check(1, rho),
                vec![Term::Product(vec![delta, delta]), Term::Product(vec![delta])],
            ));
            instances.push(InstanceSpec {
                gamma: gammas.fips_lane_check(1, rho),
                scale: E::from_i64(-2),
                terms: vec![Term::Product(vec![delta, sign])],
            });
        }
        // F_w1 = sum_t 2^(t+1) e b_t + top e^2 - top e with top = m_w - 1, per row lane.
        let top = manifest.w1_top();
        for rho in 0..gammas.lanes {
            let (f_w1, wrap) = (gammas.fips_lane_check(2, rho), fips(FipsColumn::Wrap, rho));
            for t in 0..gammas.bits {
                instances.push(InstanceSpec {
                    gamma: f_w1,
                    scale: E::from_i64(2 << t),
                    terms: vec![Term::Product(vec![wrap, position(Column::W1Bit(t as u8, rho as u8))])],
                });
            }
            instances.push(InstanceSpec {
                gamma: f_w1,
                scale: E::from_i64(top),
                terms: vec![Term::Product(vec![wrap, wrap])],
            });
            instances.push(InstanceSpec {
                gamma: f_w1,
                scale: E::from_i64(-top),
                terms: vec![Term::Product(vec![wrap])],
            });
        }
        // The reductions of stage 2's delta^(rho)(r^(2)) and e^(rho)(r^(2)).
        for (k, column) in crate::stage2::fips_terminals(manifest).into_iter().enumerate() {
            instances.push(InstanceSpec::new(gammas.fips_terminal_reduction(k), vec![Term::Product(vec![position(column)])]));
        }
    }
    instances.extend(
        bits.iter()
            .enumerate()
            .map(|(i, &column)| InstanceSpec::new(gammas.b_r2(i), vec![Term::Product(vec![position(column)])])),
    );
    groups.push(GroupSpec {
        domain: GroupRole::Main,
        eq: EqPoint::R2,
        lanes: column_lanes(&main_columns),
        instances,
    });
    if gammas.fips {
        // F_omega: sum_pos delta^2 over every row lane (main) plus sum_hi u_h (D_2), both at
        // (1/2, ..., r^(2)_sig).
        let gamma = gammas.fips_omega();
        let lifted = |role: GroupRole| -> E {
            E::from_u64(1_u64 << (manifest.group(role).logical_vars - manifest.d_sig()))
        };
        groups.push(GroupSpec {
            domain: GroupRole::Main,
            eq: EqPoint::HalfR2,
            lanes: manifest.fips_lanes(FipsColumn::Delta).into_iter().map(LaneId::Column).collect(),
            instances: vec![InstanceSpec {
                gamma,
                scale: lifted(GroupRole::Main),
                terms: (0..gammas.lanes).map(|rho| Term::Product(vec![rho, rho])).collect(),
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
    // z = S_zlo + 2^16 B_zhi + 130,915 a_z - 98,225 under H-exact), one group per lane of z.
    // With polynomial lanes the weight w_A^(s) already carries the row coefficients
    // gamma_rho of the instances ring[rho], so the part is unbatched.
    for s in 0..lanes_l {
        let terms = vec![Term::Affine {
            weight: Some(0),
            lanes: z_form.weights().into_iter().enumerate().map(|(k, w)| (1 + k, w)).collect(),
            constant: z_form.constant(),
        }];
        groups.push(GroupSpec {
            domain: GroupRole::Main,
            eq: EqPoint::None,
            lanes: std::iter::once(LaneId::Weight(WeightId::Ring(s)))
                .chain((0..z_form.len() as u8).map(|digit| LaneId::Column(Column::lane_digit(z_value, digit, s))))
                .collect(),
            instances: vec![if manifest.has_lanes() {
                InstanceSpec::unbatched(terms)
            } else {
                InstanceSpec::new(GAMMA_RING, terms)
            }],
        });
    }

    // Main domain, eq(r^(1)): bit reductions from r^(1).
    let bit_ids = column_lanes(&bits);
    groups.push(GroupSpec {
        domain: GroupRole::Main,
        eq: EqPoint::R1,
        instances: reductions(&bit_ids, |i| gammas.b_r1(i)),
        lanes: bit_ids,
    });

    // D_1, eq(r^(1)_1): K^(1)_t reductions.
    let kappa1 = column_lanes(&manifest.stack_columns(manifest.carry_stack(1)));
    groups.push(GroupSpec {
        domain: GroupRole::Level1,
        eq: EqPoint::R1,
        instances: reductions(&kappa1, |t| gammas.kappa1_r1(t)),
        lanes: kappa1,
    });

    // D_2, eq(tau_2): y^(2)_t and K^(2)_t reductions; eq(r^(1)_2): y^(2)_t.
    let levels = 0..gammas.bits as u8;
    let kappa2: Vec<LaneId> = levels.clone().map(|t| LaneId::Column(Column::carry(2, t))).collect();
    let y2: Vec<LaneId> = levels.map(|t| LaneId::Column(Column::Parity2(t))).collect();
    let mut tau2_lanes = y2.clone();
    tau2_lanes.extend_from_slice(&kappa2);
    let mut tau2_instances = reductions(&y2, |t| gammas.y2_tau(t));
    tau2_instances.extend(
        (0..kappa2.len())
            .map(|l| InstanceSpec::new(gammas.kappa2_tau(l), vec![Term::Product(vec![gammas.bits + l])])),
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
    let kappa3 = column_lanes(&manifest.stack_columns(manifest.carry_stack(3)));
    groups.push(GroupSpec {
        domain: GroupRole::Level3,
        eq: EqPoint::Tau,
        instances: reductions(&kappa3, |_| gammas.kappa3_tau()),
        lanes: kappa3,
    });

    // The projection identity (`projection`), both sides with the shared claim c: on the
    // main domain without an equality factor, W^(rho) (V_lo + 2^16 V_hi + 2^15) per lane of v;
    // on the projection domain at the suffix r' of r^(2), the composition of the digits of
    // k (K_0 + 2^16 K_1 + 2^32 K_2 + 2^35 K_3 + 2^34 + 2^31 + 2^15 for ML-DSA-44), beside the
    // links of its leaf-checked digits.
    let v_form = V16_DIGITS;
    for lane in 0..lanes_k {
        groups.push(GroupSpec {
            domain: GroupRole::Main,
            eq: EqPoint::None,
            lanes: vec![
                LaneId::Weight(WeightId::Proj(lane)),
                LaneId::Column(Column::V_LO.in_lane(lane)),
                LaneId::Column(Column::V_HI.in_lane(lane)),
            ],
            instances: vec![InstanceSpec::new(
                gammas.proj_v(),
                vec![Term::Affine {
                    weight: Some(0),
                    lanes: v_form.weights().into_iter().enumerate().map(|(k, w)| (1 + k, w)).collect(),
                    constant: v_form.constant(),
                }],
            )],
        });
    }
    let k_columns = value_columns(manifest.k_value());
    let k_form = manifest.proj_digits();
    let mut proj_instances: Vec<InstanceSpec> = all
        .iter()
        .enumerate()
        .filter(|(_, digit)| digit.column.domain() == GroupRole::Proj)
        .map(|(k, digit)| {
            let lane = k_columns.iter().position(|&c| c == digit.column).expect("a k digit");
            InstanceSpec::new(gammas.link(k), vec![Term::Link(lane)])
        })
        .collect();
    proj_instances.push(InstanceSpec::new(
        gammas.proj_k(),
        vec![Term::Affine {
            weight: None,
            lanes: k_form.weights().into_iter().enumerate().collect(),
            constant: k_form.constant(),
        }],
    ));
    groups.push(GroupSpec {
        domain: GroupRole::Proj,
        eq: EqPoint::R2,
        lanes: column_lanes(&k_columns),
        instances: proj_instances,
    });

    // The lookup stacks: one equality group per stack at its low point, one part per column
    // with the column weight as scale.
    let domains = manifest.stack_domains();
    for (j, stack_weights) in weights.iter().enumerate() {
        let columns = manifest.stack_columns(j);
        assert_eq!(stack_weights.len(), columns.len(), "one weight per stack column");
        groups.push(GroupSpec {
            domain: domains[j],
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

/// The claimed sums in gamma order (the Booleanity zero-checks claim zero); `t_at_r2[rho]`
/// is `T~_rho(r^(2))` (one value, `T(r^(2))`, for ML-DSA-44) and `stacks[j]` stack `j`'s
/// column claim ([`stack_reductions`]).
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn claimed_sums(
    manifest: &CommitmentManifest,
    t_at_r2: &[E],
    prestage: &PreStageClaims,
    stage1: &Stage1Claims,
    kappa1: &Kappa1Claims,
    stage2: &Stage2Claims,
    stacks: &[E],
    stage2_projection: E,
) -> Vec<E> {
    let gammas = Gammas::new(manifest);
    assert_eq!(t_at_r2.len(), gammas.rings, "one T claim per ring instance");
    let mut claims = vec![E::zero(); gammas.count()];
    for (rho, &t) in t_at_r2.iter().enumerate() {
        claims[gammas.ring(rho)] = t;
    }
    for (k, &square) in stage2.squares.iter().enumerate() {
        claims[gammas.link(k)] = square;
    }
    if gammas.fips {
        claims[gammas.fips_omega()] = E::from_u64(manifest.omega() as u64);
        for (k, &value) in stage2.fips.iter().enumerate() {
            claims[gammas.fips_terminal_reduction(k)] = value;
        }
    }
    for i in 0..gammas.bit_lanes {
        claims[gammas.b_r1(i)] = stage1.b[i];
        claims[gammas.b_r2(i)] = stage2.b[i];
    }
    for t in 0..gammas.bits {
        claims[gammas.y2_tau(t)] = prestage.y2[t];
        claims[gammas.y2_r1(t)] = stage1.y2[t];
        claims[gammas.kappa2_tau(t)] = prestage.kappa2[t];
        claims[gammas.kappa1_r1(t)] = kappa1.0[t];
    }
    claims[gammas.kappa3_tau()] = prestage.kappa3;
    for (j, &claim) in stacks.iter().enumerate() {
        claims[gammas.stack(j)] = claim;
    }
    claims[gammas.proj_v()] = stage2_projection;
    claims[gammas.proj_k()] = stage2_projection;
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

/// The ring inputs of stage 3 that depend only on `r^(2)` and public data.
#[derive(Clone, Copy, Debug)]
pub enum RingInputs<'p> {
    /// ML-DSA-44: `T(r^(2))`; `w_A` is in the prover tables.
    Rows(E),
    /// Polynomial lanes: `T~_rho(r^(2))` per row, and the public data from which the ring
    /// weights `w_A^(s)` are built once `gamma^(3)` is drawn (they combine the rows by the
    /// coefficients of the instances `ring[rho]`).
    Lanes {
        /// `T~_rho(r^(2))`.
        t: &'p [E],
        /// The public matrices.
        public: &'p LanePublicData,
    },
}

impl RingInputs<'_> {
    /// The claimed sums of the ring instances.
    #[must_use]
    pub fn t_values(&self) -> Vec<E> {
        match self {
            Self::Rows(t) => vec![*t],
            Self::Lanes { t, .. } => t.to_vec(),
        }
    }
}

/// The lane ring weights `w_A^(s)` of the sets with polynomial lanes, in the layout of the
/// stage kernels.
enum RingWeights {
    /// One `K` table per column `s`.
    Rows(Vec<Vec<E>>),
    /// Lane-major canonical planes of `rows` rows per column.
    Planes(agg_simd::buf::AlignedVec<u64>, usize),
}

impl RingWeights {
    /// The lane of column `s`.
    fn lane(&self, s: usize) -> Lane<'_> {
        match self {
            Self::Rows(tables) => Lane::Source(Source::Field(&tables[s])),
            Self::Planes(buf, rows) => {
                let (c0, c1) = buf[2 * s * rows..2 * (s + 1) * rows].split_at(*rows);
                Lane::Source(Source::PlaneSlice((c0, c1)))
            }
        }
    }
}

/// Prove stage 3. `points` holds `tau, r^(1), r^(2)` and the lookup stack points;
/// `tables` holds `w_A` (ML-DSA-44) and `W`; `ring` the ring inputs, which the caller may
/// compute before the join because they depend only on `r^(2)` and public data.
#[allow(clippy::too_many_arguments)]
pub fn prove(
    manifest: &CommitmentManifest,
    tables: &ProverTables<'_>,
    ring: RingInputs<'_>,
    points: &Points,
    prestage: &PreStageClaims,
    stage1: &Stage1Output,
    stage2: &Stage2Output,
    lookup: &LookupInputs,
    transcript: &mut Transcript,
) -> Result<(Stage3Proof, Stage3Output), StageError> {
    prove_with(manifest, tables, ring, points, prestage, stage1, stage2, lookup, transcript, true)
}

/// [`prove`] without the prover's claim self-check (tests).
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn prove_unchecked(
    manifest: &CommitmentManifest,
    tables: &ProverTables<'_>,
    ring: RingInputs<'_>,
    points: &Points,
    prestage: &PreStageClaims,
    stage1: &Stage1Output,
    stage2: &Stage2Output,
    lookup: &LookupInputs,
    transcript: &mut Transcript,
) -> Result<(Stage3Proof, Stage3Output), StageError> {
    prove_with(manifest, tables, ring, points, prestage, stage1, stage2, lookup, transcript, false)
}

#[allow(clippy::too_many_arguments)]
fn prove_with(
    manifest: &CommitmentManifest,
    tables: &ProverTables<'_>,
    ring: RingInputs<'_>,
    points: &Points,
    prestage: &PreStageClaims,
    stage1: &Stage1Output,
    stage2: &Stage2Output,
    lookup: &LookupInputs,
    transcript: &mut Transcript,
    check: bool,
) -> Result<(Stage3Proof, Stage3Output), StageError> {
    let spec = spec(manifest, &lookup.weights);
    let claims = claimed_sums(
        manifest,
        &ring.t_values(),
        prestage,
        &stage1.claims,
        &stage2.kappa1,
        &stage2.claims,
        &lookup.claims,
        stage2.projection,
    );
    let run = match ring {
        RingInputs::Rows(_) if check => {
            prove_stage(&spec, manifest, points, &claims, |id| tables.lane(id), transcript)?
        }
        RingInputs::Rows(_) => prove_stage_unchecked(&spec, manifest, points, &claims, |id| tables.lane(id), transcript)?,
        RingInputs::Lanes { public, .. } => {
            let gammas = draw_gammas(&spec, &claims, transcript)?;
            let rows = Gammas::new(manifest);
            let row_gammas: Vec<E> = (0..rows.rings).map(|rho| gammas[rows.ring(rho)]).collect();
            let clock = std::time::Instant::now();
            // The plane-layout kernels read the weights as canonical planes, built in place;
            // the row layout reads one `K` table per column.
            let ring = if crate::kernel::layout() == crate::kernel::Layout::Plane {
                RingWeights::Planes(public.ring_weight_planes(&points.r2, &row_gammas), public.padded_records() * N)
            } else {
                RingWeights::Rows(public.ring_weight_tables(&points.r2, &row_gammas))
            };
            if std::env::var_os("AGG_STAGE_TRACE").is_some() {
                eprintln!("weights: lane w_A {:.1} ms", clock.elapsed().as_secs_f64() * 1e3);
            }
            let resolve = |id| match id {
                LaneId::Weight(WeightId::Ring(s)) => ring.lane(usize::from(s)),
                other => tables.lane(other),
            };
            let rank_one = |id| tables.rank_one(id);
            prove_stage_drawn(&spec, manifest, points, &claims, gammas, resolve, None, Some(&rank_one), transcript, check)?
        }
    };
    let values: Vec<E> = manifest
        .groups()
        .iter()
        .filter(|group| group.role != GroupRole::Mult)
        .flat_map(|group| {
            group
                .columns
                .iter()
                .filter(|c| !c.is_zero())
                .map(|&c| run.value(&spec, LaneId::Column(c)))
        })
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

/// The verifier's public pass of stage 3: `public_pass(r^(2), r^(3), gamma_rows)` returns
/// `(T~_rho(r^(2)) per ring instance, w_A^(s)(r^(3)) per lane of z)`, with `gamma_rows` the
/// coefficients of the instances `ring[rho]` (one value each for ML-DSA-44, whose `w_A`
/// combines the rows by `eq(r^(2)_row, rho)` and ignores them).
///
/// The rounds are replayed with the ring claims set to zero, the terminal claims
/// absorbed, and only then the pass runs, so that both public quantities come from one pass
/// over the public matrices. Compressed rounds make the final claim affine in the initial
/// claim with slope `prod_k r^(3)_k` (each round is `g(x) = claim x + terms free of the
/// claim`), so the deferred claims enter as `sum_rho gamma_rho T~_rho(r^(2)) prod_k r^(3)_k`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_streamed(
    manifest: &CommitmentManifest,
    points: &Points,
    prestage: &PreStageClaims,
    stage1: &Stage1Output,
    stage2: &Stage2Output,
    lookup: &LookupInputs,
    psi: &[u8],
    proof: &Stage3Proof,
    transcript: &mut Transcript,
    public_pass: impl FnOnce(&[E], &[E], &[E]) -> (Vec<E>, Vec<E>),
) -> Result<Stage3Output, StageError> {
    assert_eq!(psi.len(), manifest.proj_bytes(), "Psi has 128 rows of P bits");
    let gammas = Gammas::new(manifest);
    let spec = spec(manifest, &lookup.weights);
    let claims = claimed_sums(
        manifest,
        &vec![E::zero(); gammas.rings],
        prestage,
        &stage1.claims,
        &stage2.kappa1,
        &stage2.claims,
        &lookup.claims,
        stage2.projection,
    );
    let mut replay = replay_stage(&spec, manifest, &claims, &proof.rounds, transcript)?;
    if proof.claims.0.len() != Stage3Claims::len(manifest) {
        return Err(StageError::Shape("stage 3 claims".into()));
    }
    transcript.absorb_extensions(&terminal_site(STAGE), &proof.claims.0)?;
    let row_gammas: Vec<E> = (0..gammas.rings).map(|rho| replay.gammas[gammas.ring(rho)]).collect();
    let (t_at_r2, ring_weights) = public_pass(&stage2.point, &replay.point, &row_gammas);
    let lanes_l = manifest.shape().lanes_l;
    if t_at_r2.len() != gammas.rings || ring_weights.len() != lanes_l {
        return Err(StageError::Internal("public pass shape".into()));
    }
    let slope = replay.point.iter().fold(E::one(), |acc, &r| acc * r);
    let deferred = row_gammas.iter().zip(&t_at_r2).fold(E::zero(), |acc, (&g, &t)| acc + g * t);
    replay.final_claim += deferred * slope;
    let proj_weights = if manifest.has_lanes() {
        crate::projection::lane_weights_at(manifest, psi, &stage2.point, &replay.point)
    } else {
        vec![crate::projection::weight_at(manifest, psi, &stage2.point, &replay.point)]
    };
    check_terminal(&spec, manifest, points, &replay, |id| match id {
        LaneId::Column(c) => proof.claims.column(manifest, c),
        LaneId::Weight(WeightId::Ring(s)) => ring_weights[usize::from(s)],
        LaneId::Weight(WeightId::Proj(lane)) => proj_weights[usize::from(lane)],
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
    use crate::manifest::L16_SLACK;
    use agg_witness::params::{GAMMA2, W1_BITS};
    use crate::claims::{Kappa1Claims, PreStageClaims, Stage1Claims, Stage2Claims};
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
    fn stage_three_has_forty_nine_instances_in_spec_order() {
        let spec = check_spec(&CommitmentManifest::for_batch(8).unwrap(), 49);
        let gammas = Gammas::new(&CommitmentManifest::for_batch(8).unwrap());
        assert_eq!((gammas.kappa1_r1(5), gammas.kappa3_tau(), gammas.stack(4)), (41, 35, 46));
        assert_eq!((gammas.link(2), gammas.link(3), gammas.proj_v(), gammas.proj_k()), (3, 4, 47, 48));
        // The projection identity: W (V_lo + 2^16 V_hi + 2^15) on the main domain without an
        // equality factor; K_0 + 2^16 K_1 + 2^32 K_2 + 2^35 K_3 + 2^34 + 2^31 + 2^15 and the
        // links of K_2, K_3 on the projection domain at r^(2).
        let v_side = spec.groups.iter().find(|g| g.lanes.first() == Some(&LaneId::Weight(WeightId::Proj(0)))).unwrap();
        assert_eq!((v_side.domain, v_side.eq), (GroupRole::Main, EqPoint::None));
        assert_eq!(
            v_side.instances[0].terms,
            vec![Term::Affine { weight: Some(0), lanes: vec![(1, 1), (2, 1 << 16)], constant: 1 << 15 }]
        );
        let k_side = spec.groups.iter().find(|g| g.domain == GroupRole::Proj && g.eq == EqPoint::R2).unwrap();
        assert_eq!(
            k_side.instances.iter().map(|i| (i.gamma, i.terms.clone())).collect::<Vec<_>>(),
            vec![
                (3, vec![Term::Link(2)]),
                (4, vec![Term::Link(3)]),
                (
                    48,
                    vec![Term::Affine {
                        weight: None,
                        lanes: vec![(0, 1), (1, 1 << 16), (2, 1 << 32), (3, 1 << 35)],
                        constant: (1 << 34) + (1 << 31) + (1 << 15)
                    }]
                ),
            ]
        );
    }

    #[test]
    fn both_sides_of_the_projection_identity_claim_the_one_sent_value() {
        let manifest = CommitmentManifest::for_batch(8).unwrap();
        let gammas = Gammas::new(&manifest);
        let mut rng = SplitMix64::new(8);
        let c = field(&mut rng);
        let claims = claimed_sums(
            &manifest,
            &[E::zero()],
            &PreStageClaims::from_slice(&point(&mut rng, PreStageClaims::LEN)).unwrap(),
            &Stage1Claims::from_slice(&point(&mut rng, Stage1Claims::LEN)).unwrap(),
            &Kappa1Claims::from_slice(&point(&mut rng, Kappa1Claims::LEN)).unwrap(),
            &Stage2Claims::from_slice(&manifest, &point(&mut rng, Stage2Claims::len(&manifest))).unwrap(),
            &point(&mut rng, STACKS),
            c,
        );
        assert_eq!((claims[gammas.proj_v()], claims[gammas.proj_k()]), (c, c));
    }

    #[test]
    fn h_exact_stage_three_has_fifty_instances_with_two_bool_checks() {
        let manifest =
            CommitmentManifest::for_relation(8, crate::manifest::Relation::HExact).unwrap();
        let spec = check_spec(&manifest, 50);
        let gammas = Gammas::new(&manifest);
        assert_eq!((gammas.links, gammas.bools), (3, 2));
        assert_eq!((gammas.bool(0), gammas.bool(1), gammas.b_r1(0)), (4, 5, 6));
        assert_eq!((gammas.kappa3_tau(), gammas.stack(4)), (36, 47));
        // H-exact: link[u_hi] at 1 (link[k_2], link[k_3] at 2, 3 are on the projection
        // domain), bool[bz_hi] at 4, bool[a_z] at 5, all in the eq(r^(2)) group; the w_A group reads
        // bz_lo, bz_hi, a_z with weights (1, 2^16, 130,915) and constant -98,225.
        let main = &spec.groups[0];
        let lane = |c| main.lane(LaneId::Column(c));
        let terms: Vec<(usize, Vec<Term>)> =
            main.instances[1..4].iter().map(|i| (i.gamma, i.terms.clone())).collect();
        assert_eq!(
            terms,
            [
                (1, vec![Term::Link(lane(Column::U_HI))]),
                (4, vec![Term::Bool(lane(Column::ZX_HI))]),
                (5, vec![Term::Bool(lane(Column::A_Z))]),
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
    fn h_fips_stage_three_has_fifty_six_instances() {
        let manifest = CommitmentManifest::for_relation(8, crate::manifest::Relation::HFips).unwrap();
        let mut rng = SplitMix64::new(4);
        let weights: Vec<Vec<E>> = (0..STACKS)
            .map(|j| point(&mut rng, manifest.stack_columns(j).len()))
            .collect();
        let spec = spec(&manifest, &weights);
        spec.validate().unwrap();
        let gammas = Gammas::new(&manifest);
        assert_eq!((gammas.links, gammas.bools, gammas.fips), (2, 3, true));
        assert_eq!(gammas.count(), 56);
        assert_eq!(spec.gamma_count, 56);
        // ring 0, link[k_2, k_3] 1, 2, bool[bz_hi, l_hi, a_z] 3..5, F_s, F_delta, F_w1, F_omega 6..9,
        // reductions from 10, the delta and e reductions 47, 48, the stacks 49..53, the
        // projection 54, 55.
        assert_eq!((gammas.fips_check(0), gammas.fips_check(3), gammas.b_r1(0)), (6, 9, 10));
        assert_eq!((gammas.fips_reduction(0), gammas.fips_reduction(1)), (47, 48));
        assert_eq!((gammas.stack(0), gammas.stack(4)), (49, 53));
        assert_eq!((gammas.proj_v(), gammas.proj_k()), (54, 55));
        // Parts per gamma index: the ring spans the eq(r^(2)) group (one affine part) and the
        // w_A group; F_delta two parts, F_w1 eight, F_omega two (main and D_2), the stacks one per column
        // (u_h is stack 2's seventh column).
        let mut counts = vec![0; 56];
        for group in &spec.groups {
            for instance in &group.instances {
                counts[instance.gamma] += 1;
            }
        }
        assert_eq!(counts[GAMMA_RING], 2);
        assert_eq!(counts[gammas.fips_check(1)], 2);
        assert_eq!(counts[gammas.fips_check(2)], 8);
        assert_eq!(counts[gammas.fips_check(3)], 2);
        assert_eq!(
            (0..STACKS).map(|j| counts[gammas.stack(j)]).collect::<Vec<_>>(),
            [2, 6, 7, 1, 2]
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
                    group.role != GroupRole::Mult && !column.is_zero(),
                    "{}",
                    column.name()
                );
            }
        }
        // F_omega's two parts at (1/2, ..., r^(2)_sig) with scales 2^(d_main - d_sig) = 2^10 and
        // 2^(d_2 - d_sig) = 2^10 (d_sig = 3, d_2 = 13).
        let f_omega: Vec<&GroupSpec> = spec.groups.iter().filter(|g| g.eq == EqPoint::HalfR2).collect();
        assert_eq!(f_omega.len(), 2);
        assert_eq!((f_omega[0].domain, f_omega[1].domain), (GroupRole::Main, GroupRole::Level2));
        for group in f_omega {
            assert_eq!(group.instances[0].scale, E::from_u64(1 << 10));
        }
    }

    /// The H-FIPS checks of stage 3 evaluate to their polynomials at
    /// random field values, so each zero-check is tested on its own: the tamper tests of
    /// `tests/all/h_fips.rs` for `F_s` (`s = 2`) and `F_delta` (`delta = -sigma`) also move the ring
    /// identity, and an in-range `L` cannot isolate `F_delta`. Also pins the ring's H-FIPS slice to
    /// `-u = -(gamma_2 s + L + 1 - gamma_2) + 2 gamma_2 delta + e` with
    /// `L = l_lo + 2^15 + 29,696 l_hi` (`l_lo` stored shifted by `2^15`).
    #[test]
    fn h_fips_checks_evaluate_to_their_polynomials() {
        let manifest = CommitmentManifest::for_relation(8, crate::manifest::Relation::HFips).unwrap();
        let mut rng = SplitMix64::new(6);
        let weights: Vec<Vec<E>> = (0..STACKS)
            .map(|j| point(&mut rng, manifest.stack_columns(j).len()))
            .collect();
        let spec = spec(&manifest, &weights);
        let gammas = Gammas::new(&manifest);
        let main = spec
            .groups
            .iter()
            .find(|g| g.eq == EqPoint::R2 && g.lanes.contains(&LaneId::Column(Column::SIGN)))
            .expect("the main eq(r^(2)) group");
        let part = |values: &[E], gamma: usize| -> E {
            main.instances
                .iter()
                .filter(|i| i.gamma == gamma)
                .fold(E::zero(), |acc, i| {
                    acc + i.scale * i.terms.iter().fold(E::zero(), |a, t| a + t.evaluate(values))
                })
        };
        let (one, two, c43) = (E::one(), E::from_u64(2), E::from_u64(43));
        let fips_lanes = [Column::L_LO, Column::L_HI, Column::SIGN, Column::DELTA, Column::WRAP];
        for _ in 0..64 {
            let values = point(&mut rng, main.lanes.len());
            let v = |c| values[main.lane(LaneId::Column(c))];
            let (s, delta, e) = (v(Column::SIGN), v(Column::DELTA), v(Column::WRAP));
            assert_eq!(part(&values, gammas.fips_check(0)), s * (s - one), "F_s");
            assert_eq!(part(&values, gammas.fips_check(1)), delta * (delta - two * s + one), "F_delta");
            let w1 = (0..W1_BITS).fold(E::zero(), |acc, t| {
                acc + v(Column::W1Bit(t as u8, 0)) * E::from_u64(1 << t)
            });
            assert_eq!(part(&values, gammas.fips_check(2)), two * e * w1 + c43 * e * e - c43 * e, "F_w1");
            // The ring's affine part moves with the five H-FIPS lanes exactly as -u does.
            let mut cleared = values.clone();
            for &c in &fips_lanes {
                cleared[main.lane(LaneId::Column(c))] = E::zero();
            }
            let gamma2 = E::from_i64(GAMMA2);
            let minus_u_slice = two * gamma2 * delta + e
                - (gamma2 * s + v(Column::L_LO) + E::from_u64(29_696) * v(Column::L_HI));
            assert_eq!(
                part(&values, GAMMA_RING) - part(&cleared, GAMMA_RING),
                minus_u_slice,
                "ring"
            );
        }
        // The constants behind the slice: L's step and the lookup shift of l_lo.
        assert_eq!((L16_SLACK.slack_weight, LOOKUP_SHIFT), (29_696, 1 << 15));
        // On integers, F_s and F_delta vanish exactly on s in {0, 1} and delta in {0, 2 s - 1}.
        for s in -2_i64..=3 {
            for delta in -3_i64..=3 {
                let mut values = vec![E::zero(); main.lanes.len()];
                values[main.lane(LaneId::Column(Column::SIGN))] = E::from_i64(s);
                values[main.lane(LaneId::Column(Column::DELTA))] = E::from_i64(delta);
                let f_s = part(&values, gammas.fips_check(0)) == E::zero();
                let f_delta = part(&values, gammas.fips_check(1)) == E::zero();
                assert_eq!(f_s, s == 0 || s == 1, "s {s}");
                let expected = (s == 0 || s == 1) && (delta == 0 || delta == 2 * s - 1);
                assert_eq!(f_s && f_delta, expected, "s {s} delta {delta}");
            }
        }
    }

    /// ML-DSA-65 and ML-DSA-87: `k` ring instances whose `A z` parts are the `l` unbatched
    /// weight groups, `l + k + 3` links, `l + k` Booleanity checks, `4 k` bit polynomials
    /// reduced from `r^(1)` and `r^(2)`, one `W^(rho)` group per lane of `v`, and every
    /// committed polynomial read at `r^(3)`.
    #[test]
    fn lane_sets_have_one_ring_instance_per_row() {
        use agg_witness::params::Set;
        for (set, k, l, stacks, count) in [(Set::S65, 6, 5, 7, 105), (Set::S87, 8, 7, 8, 132)] {
            let manifest = CommitmentManifest::for_set(set, 256, crate::manifest::Relation::Hu).unwrap();
            let gammas = Gammas::new(&manifest);
            assert_eq!(
                (gammas.rings, gammas.links, gammas.bools, gammas.bits, gammas.bit_lanes, gammas.stacks),
                (k, l + k + 3, l + k, 4, 4 * k, stacks)
            );
            assert_eq!(gammas.count(), count);
            let mut rng = SplitMix64::new(11);
            let weights: Vec<Vec<E>> = (0..stacks)
                .map(|j| point(&mut rng, manifest.stack_columns(j).len()))
                .collect();
            let spec = spec(&manifest, &weights);
            spec.validate().unwrap();
            assert_eq!(spec.gamma_count, count);
            let mut counts = vec![0; count];
            let mut unbatched = 0;
            for group in &spec.groups {
                let inner = usize::from(group.eq != EqPoint::None);
                for instance in &group.instances {
                    if instance.gamma == crate::stage::UNBATCHED {
                        unbatched += 1;
                        assert_eq!(group.lanes[0], LaneId::Weight(WeightId::Ring(unbatched as u8 - 1)));
                    } else {
                        counts[instance.gamma] += 1;
                    }
                    for term in &instance.terms {
                        assert!(term.degree() + inner <= DEGREE, "{term:?}");
                    }
                }
            }
            assert_eq!(unbatched, l);
            // Each ring instance is one affine part in the eq(r^(2)) group; the v side of the
            // projection one part per lane of v; each stack one part per column.
            assert!((0..k).all(|rho| counts[gammas.ring(rho)] == 1));
            assert_eq!(counts[gammas.proj_v()], k);
            let sizes: Vec<usize> = (0..stacks).map(|j| manifest.stack_columns(j).len()).collect();
            assert_eq!((0..stacks).map(|j| counts[gammas.stack(j)]).collect::<Vec<_>>(), sizes);
            assert!(counts.iter().all(|&c| c >= 1));
            for group in manifest.groups() {
                for &column in &group.columns {
                    assert_eq!(
                        spec.lane_ids().contains(&LaneId::Column(column)),
                        group.role != GroupRole::Mult && !column.is_zero(),
                        "{}",
                        column.name()
                    );
                }
            }
            // ring[rho] reads -(2 gamma_2 2^t) b_t^(rho), -q v^(rho) and -u^(rho) of its row only.
            let main = &spec.groups[0];
            let ring = main.instances.iter().find(|i| i.gamma == gammas.ring(2)).unwrap();
            let Term::Affine { lanes, constant, .. } = &ring.terms[0] else { panic!("affine") };
            let read: Vec<Column> = lanes
                .iter()
                .map(|&(lane, _)| match main.lanes[lane] {
                    LaneId::Column(c) => c,
                    other => panic!("{other:?}"),
                })
                .collect();
            assert!(read.iter().all(|c| c.lane() == 2), "{read:?}");
            assert_eq!(lanes[3], (main.lane(LaneId::Column(Column::W1Bit(3, 2))), -(2 * 261_888) << 3));
            // u = U_lo + 2^16 U_mid + 523,267 a_u - 228,865, v = V_lo + 2^16 V_hi + 2^15.
            assert_eq!(*constant, 228_865 - Q * (1 << 15));
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
        assert_eq!(spec.groups.len(), 9 + STACKS);
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
            [2, 6, 6, 1, 2]
        );
        assert!(counts[1..gammas.stack(0)].iter().all(|&c| c == 1));
        assert_eq!((gammas.stack(STACKS - 1), gammas.proj_k()), (count - 3, count - 1));
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
                    group.role != GroupRole::Mult && !column.is_zero(),
                    "{}",
                    column.name()
                );
            }
        }
        spec
    }
}
