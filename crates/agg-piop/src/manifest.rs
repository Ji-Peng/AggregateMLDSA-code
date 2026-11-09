//! Exact commitment manifest of one batch (the instance record).
//!
//! Mirrors `aerie-falcon/src/protocol/manifest.rs`: the manifest fixes the canonical wire
//! order, the commitment round, the preset family, the logical and physical domain of every
//! Akita group, and the role and alphabet of each committed polynomial, and serializes the
//! public profile bound in the instance record before any commitment. The seven groups are `G_main16`
//! (the looked-up low digits of `z` and `u`, `V_lo, V_hi`, width 16) and `G_main4` (the high
//! digits of `z` and `u`, `b_0..b_5`, the slack bit `a_z` and H-FIPS's columns, width 4) in
//! round 1, before the bridge seed, and `G_1` (`kappa^(1)_t`), `G_2` (`y^(2)_t`,
//! `kappa^(2)_t`), `G_3` (`kappa^(3)` and zero polynomials up to `G_2`'s layout), `G_k` (the
//! digits `K_0, K_1, K_2, K_3` of the projection outputs `k = <Psi_o, v_i>`), all width 16, and
//! `G_m` (the seven digits of the multiplicity column of `T_16`, width 4) in round 2; `G_m` is
//! Akita's final group, so PCS execution follows wire order. The alphabets listed here are
//! the honest producer's; the ranges are proven inside the PIOP (the lookup into `T_16`, the
//! base-4 and base-6 leaves), never by Akita. `V_lo, V_hi` have no range check: the
//! projection bounds `v` through `k`.
//!
//! Relation H-exact (relation byte 1, a reference relation): `u` exact,
//! `u = B_ulo + 63,079 H_u - 190,465` with `B_ulo` the looked-up `U_lo + 2^15` and
//! `H_u = U_hi + 3` a six-valued digit, so `|u| <= B_u = 190,465`, and `z` exact,
//! `z = B_zlo + 2^16 B_zhi + 130,915 a_z - 130,993` with `B_zlo = S_zlo + 2^15` looked up (the
//! slot of `Z_lo`) and `B_zhi`, `a_z` Boolean (`B_zhi` in the slot of `Z_hi`, `a_z` appended
//! to `G_main4` after the bits), so `|z| <= B_z = 130,993`. The value `z` of H-exact is
//! [`DigitValue::Zx`] with the columns `bz_lo`, `bz_hi`, `a_z`; the Booleanity of `bz_hi`
//! and `a_z` is proven by two zero-checks at `r^(2)` in stage 3.
//!
//! Relation H-FIPS (relation byte 2, the default): `z` as in H-exact, and in
//! place of `u`'s digits the per-coefficient witness of `agg_witness::hfips`: the magnitude
//! `L = L_lo + 29,696 L_hi` ([`DigitValue::L`]: `l_lo` looked up in the slot of `U_lo`, the
//! bit `l_hi` in the slot of `U_hi`), the sign bit `s = (1 + sigma)/2`, `delta` and `e`
//! ([`FipsColumn`], appended to `G_main4` after `a_z`), so
//! `u = gamma_2 s + L + 1 - gamma_2 - 2 gamma_2 delta - e`; and the per-signature hint slack `u_h`
//! ([`DigitValue::HintSlack`]), a thirteenth `G_2` polynomial on `D_2` (whose last `d_sig`
//! variables are the signature index) in the free slot 6 of lookup stack 2.
//!
//! ML-DSA-65 and ML-DSA-87 (every relation, [`SetShape`]): the vector index is a
//! polynomial lane instead of two domain variables, so the main domain is `(j, i)` with
//! `8 + d_sig` variables and every main-domain column is a family of `k` or `l` polynomials
//! ([`Column::lane`]). `z` and `u` take digit-slack forms with a base-8 middle digit and a
//! Boolean slack ([`DigitValue::Zdx`]: `|z| <= B_z` exactly, [`DigitValue::Ud`]) under
//! H-exact; H-FIPS takes H-exact's `z`, the magnitude
//! [`DigitValue::Ld`] in the slots of `u` and the columns `s, delta, e` per row lane, with the
//! hint slack `u_h` in `G_2` and in a fifth slot of the level-2 lookup stack. `w_1'` has four bits, the
//! projection outputs five digits ([`K39_DIGITS`]), and the looked-up main columns split
//! into stacks of 8, 2, 1 (8, 4, 2, 1) polynomials ([`main_stack_sizes`]).

use agg_pcs_config::catalogs::{
    BatchLayout, Family, Group, Set, MULTIPLICITY_DIGITS, TABLE_VARS,
};
pub use agg_pcs_config::catalogs::{
    HashField, Relation, GROUP_COUNT_FP, KECCAK_GROUP_COUNT, RELATION_H_EXACT, RELATION_H_FIPS,
};
use agg_pcs_config::work_plan::{padded_size, PlanError, WorkPlan};
use agg_witness::digits::{
    z_digit_slack_exact, Alphabet, BitSlackLayout, DigitLayout, SlackLayout, StepLayout, ValueForm, L_SLACK,
    L_BIT_SLACK, U_DIGIT_SLACK, Z_BIT_SLACK,
};
use agg_witness::params::{ParamSet, B_U, STEP_U, U_TOP_BASE, W1_BITS};

use crate::lookup::{LookupShape, StackShape, Table};

/// Role of a committed group; the seven groups in wire order.
pub type GroupRole = Group;

/// PCS preset family of a group (`agg_pcs_config::catalogs::Family`).
pub type PcsFamily = Family;

/// Wire version of this protocol (the Keccak side's tied parts of proof version 5 with the
/// virtual parity chain and its work sites in the plan).
pub const WIRE_VERSION: u16 = 4;
/// Identifier of the digit layout: 2 is the lookup layout with the projected quotient
/// (ranges by the lookup into `T_16`; `v` bounded through the digits of its projection
/// outputs `k`).
pub const DIGIT_LAYOUT_ID: u8 = 2;
/// Number of committed groups.
pub const GROUP_COUNT: usize = agg_pcs_config::catalogs::GROUP_COUNT;
/// Independent lookup challenges `X_1, X_2`.
pub const LOOKUP_CHALLENGES: usize = 2;
/// The balanced storage shift of every looked-up column: the looked-up value is
/// `f = stored + 2^15`.
pub const LOOKUP_SHIFT: i32 = 1 << 15;

/// Relation H-exact: `z = S_zlo + 2^16 B_zhi + 130,915 a_z - 98,225` over exactly `[-B_z, B_z]`:
/// `S_zlo = B_zlo - 2^15` looked up, `B_zhi, a_z` bits.
pub const ZX16_DIGITS: BitSlackLayout = Z_BIT_SLACK;
/// `B_ulo = U_lo + 2^15` in `[0, 2^16)` (base `2^16`, unsigned).
pub const ULO16_DIGITS: DigitLayout = DigitLayout {
    name: "u",
    bases: &[1 << 16],
    signed: false,
};
/// Relation H-exact: `u = B_ulo + 63,079 H_u - 190,465` with `B_ulo = U_lo + 2^15` looked up and
/// `H_u = U_hi + 3 in [0, 6)`, so `u = U_lo + 63,079 U_hi + 31,540` over exactly `[-B_u, B_u]`
/// (`5 * 63,079 + 2^16 = 2 B_u + 1`).
pub const U16_DIGITS: StepLayout = StepLayout {
    low: ULO16_DIGITS,
    step: STEP_U,
    top_base: U_TOP_BASE,
    offset: B_U,
};
/// `v + 2^31 = v_lo + 2^16 v_hi` (bases `2^16, 2^16`, signed). The honest encoding only:
/// `V_lo`, `V_hi` are not range-checked; the projection outputs `k` bound `v` (`K16_DIGITS`).
pub const V16_DIGITS: DigitLayout = DigitLayout {
    name: "v",
    bases: &[1 << 16, 1 << 16],
    signed: true,
};
/// The projection outputs `k_{o,i} = <Psi_o, v_i>`: `k + 2^37 = k_0 + 2^16 k_1 + 2^32 k_2 +
/// 2^35 k_3` (bases `2^16, 2^16, 8, 8`, signed), so
/// `k = K_0 + 2^16 K_1 + 2^32 K_2 + 2^35 K_3 + 2^34 + 2^31 + 2^15` over exactly
/// `[-2^37, 2^37)`; `K_0, K_1` looked up in `T_16`, `K_2, K_3` leaves `leaf8` and links.
pub const K16_DIGITS: DigitLayout = DigitLayout {
    name: "k",
    bases: &[1 << 16, 1 << 16, 8, 8],
    signed: true,
};
/// Rows of the projection matrix `Psi in {0, 1}^(128 x 1024)`.
pub const PROJ_ROWS: usize = 1 << agg_pcs_config::catalogs::PROJ_ROW_VARS;
/// Bytes of `Psi`, squeezed as the answer of the site `proj/psi` (16 KiB).
pub const PROJ_BYTES: usize = PROJ_ROWS * agg_witness::params::POSITIONS / 8;
/// The bound `B` of the projection outputs: `k in [-B, B)`.
pub const PROJ_BOUND: i64 = 1 << 37;
/// The projection outputs of ML-DSA-65 and ML-DSA-87: `k + 2^39 = k_0 + 2^16 k_1 + 2^32 k_2 +
/// 2^35 k_3 + 2^38 k_4` (bases `2^16, 2^16, 8, 8, 4`, signed), so
/// `k = K_0 + 2^16 K_1 + 2^32 K_2 + 2^35 K_3 + 2^38 K_4 + 2^37 + 2^34 + 2^31 + 2^15` over
/// exactly `[-2^39, 2^39)`; `K_0, K_1` looked up in `T_16`, `K_2, K_3` leaves `leaf8`, `K_4` a
/// leaf `leaf4`, each with a link. The honest outputs need `(P / g) B_v <= 2^39`, which holds
/// for both sets with `g = 1` (ML-DSA-65) and `g = 2` row blocks (ML-DSA-87).
pub const K39_DIGITS: DigitLayout = DigitLayout {
    name: "k",
    bases: &[1 << 16, 1 << 16, 8, 8, 4],
    signed: true,
};
/// The bound `B` of the projection outputs of ML-DSA-65 and ML-DSA-87: `k in [-B, B)`.
pub const PROJ_BOUND_39: i64 = 1 << 39;
/// Relation H-FIPS: `L = L_lo + 29,696 L_hi` with `L_lo = S_lo + 2^15` looked up and the bit
/// `L_hi`, so `L = S_lo + 29,696 L_hi + 2^15` over exactly `[0, gamma_2 - 1]`.
pub const L16_SLACK: SlackLayout = L_SLACK;
/// A carry as one balanced 16-bit column (base `2^16`, unsigned): `kappa = K + 2^15`.
pub const CARRY16_DIGITS: DigitLayout = DigitLayout {
    name: "kappa",
    bases: &[1 << 16],
    signed: false,
};
/// The base of a column looked up in `T_16`.
pub const LOOKUP_BASE: u32 = 1 << 16;
/// Alphabet code of a looked-up column in the instance-record bytes (`log2 |T_16|` with the high
/// bit set).
pub const LOOKUP_ALPHABET_CODE: u8 = 0x80 | 16;
/// Alphabet code of a multiplicity digit (base 16, not range-checked).
pub const MULTIPLICITY_ALPHABET_CODE: u8 = 0x40 | 16;
/// Alphabet code of a ternary column (`{-1, 0, 1}`: H-FIPS's `delta` and `e`).
pub const TERNARY_ALPHABET_CODE: u8 = 0x20 | 3;
/// Alphabet code of a 16-bit column that no range check reads (`V_lo`, `V_hi`: the
/// projection bounds `v` instead): the flag `0x10` with `log2 |T_16| = 16`, which is the same
/// bit, so the code is `0x10`.
pub const UNCHECKED_ALPHABET_CODE: u8 = 0x10;
/// Alphabet code of a zero polynomial (`G_3`'s padding to `G_2`'s layout).
pub const ZERO_ALPHABET_CODE: u8 = 0x08;

/// Canonical transcript byte of a group role (its wire position).
#[must_use]
pub const fn role_id(role: GroupRole) -> u8 {
    role.index() as u8
}

/// The dimensions of a parameter set that shape its committed columns.
///
/// The main domain is `(j, rho_hi, i)` with `d_vec` vector-index variables `rho_hi`; a
/// `k`-vector column is a family of `lanes_k = k / 2^d_vec` polynomials (lane `rho_lo`), so
/// row `rho` of a record sits in lane `rho >> d_vec` at cells `256 (rho mod 2^d_vec) + j`.
/// ML-DSA-44 has `d_vec = 2` and one lane per column; ML-DSA-65 and ML-DSA-87 have `d_vec = 0`
/// and one polynomial per row (`k` and `l` are not powers of two, so no dead cells are
/// committed).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SetShape {
    /// Rows `k` of `A`.
    pub k: usize,
    /// Columns `l` of `A`.
    pub l: usize,
    /// Vector-index variables of the main domain.
    pub d_vec: u32,
    /// Polynomials of a `k`-vector column.
    pub lanes_k: usize,
    /// Polynomials of an `l`-vector column.
    pub lanes_l: usize,
    /// Bits of `w_1'` (the bit columns `b_t`, `t < w1_bits`).
    pub w1_bits: usize,
    /// Positions `P = 256 k` of one signature in the bridge and the projection.
    pub positions: usize,
    /// `P` rounded up to a power of two: the bridge's input positions per signature.
    pub positions_pad: usize,
    /// Row blocks `g` of the projection.
    pub proj_blocks: usize,
}

impl SetShape {
    /// The shape of `set`.
    #[must_use]
    pub const fn of(set: Set) -> Self {
        agg_witness::dispatch_set!(set, |P| Self::of_params::<P>())
    }

    const fn of_params<P: ParamSet>() -> Self {
        Self {
            k: P::K,
            l: P::L,
            d_vec: P::D_VEC,
            lanes_k: P::LANES_K,
            lanes_l: P::LANES_L,
            w1_bits: P::W1_BITS,
            positions: P::P_POSITIONS,
            positions_pad: P::P_PAD,
            proj_blocks: P::PROJ_BLOCKS,
        }
    }

    /// Cells of one main-domain polynomial per signature: `256 2^d_vec`.
    #[must_use]
    pub const fn lane_cells(self) -> usize {
        256 << self.d_vec
    }

    /// The lane of row `row` of a vector and the offset of its 256 cells inside the lane's
    /// cells of one signature.
    #[must_use]
    pub const fn row_lane(self, row: usize) -> (usize, usize) {
        (row >> self.d_vec, (row & ((1 << self.d_vec) - 1)) * 256)
    }
}

/// A value committed through its balanced digits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DigitValue {
    /// The response `z` of relation H-exact: `bz_lo` (looked up), `bz_hi`, `a_z` (bits).
    Zx,
    /// The UseHint residual `u` (main domain; relation H-exact).
    U,
    /// Relation H-FIPS's magnitude `L = L_lo + 29,696 L_hi` (main domain): `l_lo` (looked
    /// up), `l_hi` (bit).
    L,
    /// The ring quotient `v` (main domain; its digits are not range-checked).
    V,
    /// The projection outputs `k_{o,i} = <Psi_o, v_i>` on the projection domain `(o, i)`.
    K,
    /// The response `z` of ML-DSA-65 and ML-DSA-87 (relations H-exact and H-FIPS, one
    /// polynomial per vector entry `s`): `z + B_z = z_lo + 2^16 z_mid + (2 B_z + 1 - 2^19) a_z`
    /// with `Z_lo` looked up, `Z_mid` base 8 (leaf and link) and the Boolean `a_z`, so
    /// `|z| <= B_z = gamma_1 - beta - 1` exactly (`agg_witness::digits::z_digit_slack_exact`).
    Zdx(Set),
    /// The UseHint residual `u` of ML-DSA-65 and ML-DSA-87: `u + B_u = B_ulo + 2^16 H_u +
    /// 523,267 a_u` with `U_lo` looked up, `U_mid = H_u - 4` base 8 (leaf and link) and the
    /// Boolean `a_u`, so `|u| <= B_u = 523,777` exactly.
    Ud,
    /// The projection outputs of ML-DSA-65 and ML-DSA-87 on `D_k = (o, b, i)`: five digits
    /// ([`K39_DIGITS`]).
    K5,
    /// Relation H-FIPS's magnitude `L` of ML-DSA-65 and ML-DSA-87 (one polynomial per row
    /// `rho`): `L = L_lo + 2^16 L_mid + 130,816 a_L` with `l_lo` looked up (the slot of `U_lo`)
    /// and the bits `l_mid`, `a_l` (the slots of `U_mid`, `a_u`), so `L in [0, gamma_2 - 1]`
    /// exactly (`agg_witness::digits::L_BIT_SLACK`).
    Ld,
    /// Relation H-FIPS's hint slack `u_h = omega - sum delta^2` on `D_2` (looked up).
    HintSlack,
    /// Level-1 carry `kappa^(1)_t` (domain `D_1`).
    Kappa1(u8),
    /// Level-2 carry `kappa^(2)_t` (domain `D_2`).
    Kappa2(u8),
    /// Level-3 carry `kappa^(3)` (domain `D_3`).
    Kappa3,
}

impl DigitValue {
    /// The value form (digit composition; the step form for `u`, H-exact's bit-slack form
    /// for `z`).
    #[must_use]
    pub const fn layout(self) -> ValueForm {
        match self {
            Self::Zx => ValueForm::BitSlack(ZX16_DIGITS),
            Self::U => ValueForm::Step(U16_DIGITS),
            Self::L => ValueForm::Slack(L16_SLACK),
            Self::V => ValueForm::Radix(V16_DIGITS),
            Self::K => ValueForm::Radix(K16_DIGITS),
            Self::Zdx(set) => ValueForm::DigitSlack(z_digit_slack_exact(agg_witness::dispatch_set!(set, |P| P::B_Z))),
            Self::Ld => ValueForm::BitSlack(L_BIT_SLACK),
            Self::Ud => ValueForm::DigitSlack(U_DIGIT_SLACK),
            Self::K5 => ValueForm::Radix(K39_DIGITS),
            Self::Kappa1(_) | Self::Kappa2(_) | Self::Kappa3 | Self::HintSlack => {
                ValueForm::Radix(CARRY16_DIGITS)
            }
        }
    }

    /// The group holding digit `digit`.
    #[must_use]
    pub const fn group(self, digit: u8) -> GroupRole {
        match self {
            Self::Zx | Self::Zdx(_) | Self::U | Self::Ud | Self::L | Self::Ld if digit == 0 => {
                Group::Main16
            }
            Self::Zx | Self::Zdx(_) | Self::U | Self::Ud | Self::L | Self::Ld => Group::Main,
            Self::V => Group::Main16,
            Self::K | Self::K5 => Group::Proj,
            Self::Kappa1(_) => Group::Level1,
            Self::Kappa2(_) | Self::HintSlack => Group::Level2,
            Self::Kappa3 => Group::Level3,
        }
    }
}

/// The three per-coefficient columns of relation H-FIPS that are not digits of a value:
/// in `G_main4` after `a_z`, in this order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FipsColumn {
    /// The sign bit `s = (1 + sigma)/2` in `{0, 1}` (`sigma = 1` iff `r0 > 0`).
    Sign,
    /// The hint move `delta in {0, sigma}` (`delta^2` is the hint bit).
    Delta,
    /// The wrap `e in {-1, 0, 1}`.
    Wrap,
}

impl FipsColumn {
    /// The three columns in `G_main4` order.
    pub const ALL: [Self; 3] = [Self::Sign, Self::Delta, Self::Wrap];
}

/// What one committed polynomial holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Column {
    /// Digit `digit` (least significant first) of a digit-decomposed value.
    Digit {
        /// The value.
        value: DigitValue,
        /// Digit index.
        digit: u8,
        /// The polynomial lane of a main-domain value ([`SetShape`]; 0 for ML-DSA-44 and
        /// for every value off the main domain).
        lane: u8,
    },
    /// The w1Encode bit `b_t` (main domain) in polynomial lane `lane` (`b_t^(rho)` of
    /// ML-DSA-65 and ML-DSA-87; lane 0 for ML-DSA-44): `W1Bit(t, lane)`.
    W1Bit(u8, u8),
    /// The level-2 parity `y^(2)_t` (domain `D_2`).
    Parity2(u8),
    /// Digit `k` of the multiplicity column `m` of `T_16` (`2^16` cells).
    Multiplicity(u8),
    /// A per-coefficient column of relation H-FIPS (main domain, `G_main4`) in polynomial
    /// lane `lane` (row `rho` of ML-DSA-65 and ML-DSA-87; lane 0 for ML-DSA-44): `Fips(column, lane)`.
    Fips(FipsColumn, u8),
    /// A zero polynomial of `G_3` (`k >= 1`), committed so that `G_3` has `G_2`'s Akita
    /// layout (`agg_pcs_config::catalogs::Group::physical_vars`); no stage reads it and its
    /// opening claim is zero.
    Zero(u8),
    /// Post-theta Keccak state lane `A'[x, y]` (`x, y in 0..5`) in slot group `q` (`0..Q`)
    /// under `HashField::Fp`.
    KeccakA {
        /// Column coordinate `x in 0..5`.
        x: u8,
        /// Row coordinate `y in 0..5`.
        y: u8,
        /// Slot-group index `q in 0..Q`.
        q: u8,
    },
    /// Pre-theta Keccak column parity `C[x]` (`x in 0..5`) in slot group `q` (`0..Q`)
    /// under `HashField::Fp`.
    KeccakC {
        /// Column coordinate `x in 0..5`.
        x: u8,
        /// Slot-group index `q in 0..Q`.
        q: u8,
    },
    /// Post-theta Keccak column parity `C'[x]` (`x in 0..5`) in slot group `q` (`0..Q`)
    /// under `HashField::Fp`.
    KeccakCPrime {
        /// Column coordinate `x in 0..5`.
        x: u8,
        /// Slot-group index `q in 0..Q`.
        q: u8,
    },
}

impl Column {
    /// `bz_lo` (H-exact `S_zlo`, the slot of `Z_lo`).
    pub const ZX_LO: Self = Self::Digit {
        value: DigitValue::Zx,
        digit: 0,
        lane: 0,
    };
    /// `bz_hi` (H-exact `B_zhi`, a bit in the slot of `Z_hi`).
    pub const ZX_HI: Self = Self::Digit {
        value: DigitValue::Zx,
        digit: 1,
        lane: 0,
    };
    /// `a_z` (H-exact slack bit, after the bits of `G_main4`).
    pub const A_Z: Self = Self::Digit {
        value: DigitValue::Zx,
        digit: 2,
        lane: 0,
    };
    /// `U_lo`.
    pub const U_LO: Self = Self::Digit {
        value: DigitValue::U,
        digit: 0,
        lane: 0,
    };
    /// `U_hi`.
    pub const U_HI: Self = Self::Digit {
        value: DigitValue::U,
        digit: 1,
        lane: 0,
    };
    /// `l_lo` (H-FIPS `L_lo - 2^15`, the slot of `U_lo`).
    pub const L_LO: Self = Self::Digit {
        value: DigitValue::L,
        digit: 0,
        lane: 0,
    };
    /// `l_hi` (H-FIPS `L_hi`, a bit in the slot of `U_hi`).
    pub const L_HI: Self = Self::Digit {
        value: DigitValue::L,
        digit: 1,
        lane: 0,
    };
    /// H-FIPS's sign bit `s`.
    pub const SIGN: Self = Self::Fips(FipsColumn::Sign, 0);
    /// H-FIPS's `delta`.
    pub const DELTA: Self = Self::Fips(FipsColumn::Delta, 0);
    /// H-FIPS's `e`.
    pub const WRAP: Self = Self::Fips(FipsColumn::Wrap, 0);
    /// H-FIPS's hint slack `u_h` (stored `u_h - 2^15`, on `D_2`).
    pub const HINT_SLACK: Self = Self::Digit {
        value: DigitValue::HintSlack,
        digit: 0,
        lane: 0,
    };
    /// `V_lo`.
    pub const V_LO: Self = Self::Digit {
        value: DigitValue::V,
        digit: 0,
        lane: 0,
    };
    /// `V_hi`.
    pub const V_HI: Self = Self::Digit {
        value: DigitValue::V,
        digit: 1,
        lane: 0,
    };
    /// `K_0` (projection outputs, looked up).
    pub const K_0: Self = Self::Digit {
        value: DigitValue::K,
        digit: 0,
        lane: 0,
    };
    /// `K_1` (projection outputs, looked up).
    pub const K_1: Self = Self::Digit {
        value: DigitValue::K,
        digit: 1,
        lane: 0,
    };
    /// `K_2` (projection outputs, base 8, leaf and link).
    pub const K_2: Self = Self::Digit {
        value: DigitValue::K,
        digit: 2,
        lane: 0,
    };
    /// `K_3` (projection outputs, base 8, leaf and link).
    pub const K_3: Self = Self::Digit {
        value: DigitValue::K,
        digit: 3,
        lane: 0,
    };

    /// The carry column `kappa^(l)_t` (`l = 1, 2`) or `kappa^(3)` (`l = 3`).
    #[must_use]
    pub const fn carry(level: u8, t: u8) -> Self {
        let value = match level {
            1 => DigitValue::Kappa1(t),
            2 => DigitValue::Kappa2(t),
            _ => DigitValue::Kappa3,
        };
        Self::Digit { value, digit: 0, lane: 0 }
    }

    /// Digit `digit` of `value` in lane `lane`.
    #[must_use]
    pub const fn lane_digit(value: DigitValue, digit: u8, lane: u8) -> Self {
        Self::Digit { value, digit, lane }
    }

    /// The polynomial lane of the column (0 for a column with one lane).
    #[must_use]
    pub const fn lane(self) -> u8 {
        match self {
            Self::Digit { lane, .. } | Self::W1Bit(_, lane) | Self::Fips(_, lane) => lane,
            _ => 0,
        }
    }

    /// The same digit or bit column in lane `lane` (other columns have one lane and are
    /// returned unchanged).
    #[must_use]
    pub const fn in_lane(self, lane: u8) -> Self {
        match self {
            Self::Digit { value, digit, .. } => Self::Digit { value, digit, lane },
            Self::W1Bit(t, _) => Self::W1Bit(t, lane),
            Self::Fips(column, _) => Self::Fips(column, lane),
            other => other,
        }
    }

    /// The honest alphabet (of the stored value).
    #[must_use]
    pub fn alphabet(self) -> Alphabet {
        match self {
            Self::Digit { value, digit, .. } => value.layout().alphabet(usize::from(digit)),
            Self::W1Bit(..)
            | Self::Parity2(_)
            | Self::Fips(FipsColumn::Sign, _)
            | Self::Zero(_)
            | Self::KeccakA { .. }
            | Self::KeccakC { .. }
            | Self::KeccakCPrime { .. } => Alphabet::Bool,
            Self::Fips(FipsColumn::Delta | FipsColumn::Wrap, _) => Alphabet::Ternary,
            Self::Multiplicity(_) => Alphabet::Balanced(16),
        }
    }

    /// Digit base, `None` for a Boolean column (also `bz_hi`, `a_z`) or a multiplicity digit.
    #[must_use]
    pub fn base(self) -> Option<u32> {
        match self {
            Self::Digit { value, digit, .. } => value.layout().base(usize::from(digit)),
            Self::W1Bit(..)
            | Self::Parity2(_)
            | Self::Multiplicity(_)
            | Self::Fips(..)
            | Self::Zero(_)
            | Self::KeccakA { .. }
            | Self::KeccakC { .. }
            | Self::KeccakCPrime { .. } => None,
        }
    }

    /// True for a column looked up in `T_16` (stored balanced, `f = stored + 2^15`). The
    /// digits of `v` have base `2^16` but are not looked up.
    #[must_use]
    pub fn is_lookup(self) -> bool {
        self.base() == Some(LOOKUP_BASE) && !self.is_unchecked()
    }

    /// True for the digits of `v`: committed, read by the ring identity and the projection
    /// identity, and range-checked by neither the lookup nor a leaf.
    #[must_use]
    pub const fn is_unchecked(self) -> bool {
        matches!(self, Self::Digit { value: DigitValue::V, .. })
    }

    /// True for a digit range-checked by a leaf and a link (`U_hi` base 6,
    /// the projection digits `K_2, K_3` base 8; for ML-DSA-65 and ML-DSA-87 `Z_mid`, `U_mid`
    /// base 8 and `K_4` base 4).
    #[must_use]
    pub fn is_leaf_digit(self) -> bool {
        matches!(self.base(), Some(4 | 6 | 8))
    }

    /// True when the column is stored as `i32` cells (every 16-bit column).
    #[must_use]
    pub fn is_wide(self) -> bool {
        self.base() == Some(LOOKUP_BASE)
    }

    /// The group holding the column.
    #[must_use]
    pub const fn group(self) -> GroupRole {
        match self {
            Self::Digit { value, digit, .. } => value.group(digit),
            Self::W1Bit(..) | Self::Fips(..) => Group::Main,
            Self::KeccakA { y, .. } => Group::KeccakA(y),
            Self::KeccakC { .. } => Group::KeccakC,
            Self::KeccakCPrime { .. } => Group::KeccakCPrime,
            Self::Parity2(_) => Group::Level2,
            Self::Multiplicity(_) => Group::Mult,
            Self::Zero(_) => Group::Level3,
        }
    }

    /// True for a zero polynomial of `G_3` (not read by any stage, opening claim zero).
    #[must_use]
    pub const fn is_zero(self) -> bool {
        matches!(self, Self::Zero(_))
    }

    /// The domain the column lives on (the main domain for both main groups; `KeccakA(0)` for
    /// all Keccak groups).
    #[must_use]
    pub const fn domain(self) -> GroupRole {
        match self.group() {
            Group::Main16 | Group::Main => Group::Main,
            Group::KeccakA(_) | Group::KeccakC | Group::KeccakCPrime => Group::KeccakA(0),
            other => other,
        }
    }

    /// True for a Boolean digit of a value (`bz_hi`, `a_z`; H-FIPS also `l_hi`), proven by a
    /// zero-check at `r^(2)` in stage 3.
    #[must_use]
    pub fn is_bool_digit(self) -> bool {
        matches!(self, Self::Digit { .. }) && self.alphabet() == Alphabet::Bool
    }

    /// Column name (`z_lo`, `u_hi`, `bz_hi`, `a_z`, `b_0`, `kappa1_2`, `y2_0`, `m_4`, ...), with
    /// the lane in brackets when it is not 0 (`u_mid[3]`, `b_2[5]`).
    #[must_use]
    pub fn name(self) -> String {
        let name = self.lane_free_name();
        match self.lane() {
            0 => name,
            lane => format!("{name}[{lane}]"),
        }
    }

    fn lane_free_name(self) -> String {
        let half = |digit: u8| if digit == 0 { "lo" } else { "hi" };
        match self {
            Self::Digit { value, digit, .. } => match value {
                DigitValue::Zx => ["bz_lo", "bz_hi", "a_z"][usize::from(digit)].to_owned(),
                DigitValue::U => format!("u_{}", half(digit)),
                DigitValue::L => format!("l_{}", half(digit)),
                DigitValue::V => format!("v_{}", half(digit)),
                DigitValue::K | DigitValue::K5 => format!("k_{digit}"),
                DigitValue::Zdx(_) => ["z_lo", "z_mid", "a_z"][usize::from(digit)].to_owned(),
                DigitValue::Ud => ["u_lo", "u_mid", "a_u"][usize::from(digit)].to_owned(),
                DigitValue::Ld => ["l_lo", "l_mid", "a_l"][usize::from(digit)].to_owned(),
                DigitValue::HintSlack => "u_h".to_owned(),
                DigitValue::Kappa1(t) => format!("kappa1_{t}"),
                DigitValue::Kappa2(t) => format!("kappa2_{t}"),
                DigitValue::Kappa3 => "kappa3".to_owned(),
            },
            Self::W1Bit(t, _) => format!("b_{t}"),
            Self::Parity2(t) => format!("y2_{t}"),
            Self::Multiplicity(k) => format!("m_{k}"),
            Self::Fips(FipsColumn::Sign, _) => "s".to_owned(),
            Self::Fips(FipsColumn::Delta, _) => "delta".to_owned(),
            Self::Fips(FipsColumn::Wrap, _) => "e".to_owned(),
            Self::Zero(k) => format!("zero_{k}"),
            Self::KeccakA { x, y, q } => format!("keccak_a_{x}_{y}_{q}"),
            Self::KeccakC { x, q } => format!("keccak_c_{x}_{q}"),
            Self::KeccakCPrime { x, q } => format!("keccak_cp_{x}_{q}"),
        }
    }

    /// Transcript code of the alphabet: the digit base of a leaf-checked digit (4, 6, 8),
    /// [`LOOKUP_ALPHABET_CODE`] for a column looked up in `T_16`,
    /// [`MULTIPLICITY_ALPHABET_CODE`] for a multiplicity digit, [`TERNARY_ALPHABET_CODE`] for
    /// H-FIPS's `delta` and `e`, 0 for a Boolean.
    #[must_use]
    pub fn alphabet_code(self) -> u8 {
        match self {
            Self::Multiplicity(_) => MULTIPLICITY_ALPHABET_CODE,
            Self::Zero(_) => ZERO_ALPHABET_CODE,
            Self::Fips(FipsColumn::Delta | FipsColumn::Wrap, _) => TERNARY_ALPHABET_CODE,
            column if column.is_unchecked() => UNCHECKED_ALPHABET_CODE,
            column if column.is_lookup() => LOOKUP_ALPHABET_CODE,
            column => column.base().map_or(0, |base| base as u8),
        }
    }
}

/// The value `z` of ML-DSA-44 (`Zx` under every relation).
#[must_use]
pub const fn z_value(_relation: Relation) -> DigitValue {
    DigitValue::Zx
}

/// The digit-decomposed value in the slots of `U_lo, U_hi`: `u` (H-exact) or H-FIPS's
/// magnitude `L`.
#[must_use]
pub const fn u_slot_value(relation: Relation) -> DigitValue {
    match relation {
        Relation::HExact => DigitValue::U,
        Relation::HFips => DigitValue::L,
    }
}

/// The value `z` of `relation` under `set` ([`z_value`] for ML-DSA-44, [`DigitValue::Zdx`]
/// for ML-DSA-65 and ML-DSA-87).
#[must_use]
pub const fn z_value_for(set: Set, relation: Relation) -> DigitValue {
    match set {
        Set::S44 => z_value(relation),
        Set::S65 | Set::S87 => DigitValue::Zdx(set),
    }
}

/// The value in the slots of `U_lo` under `set` ([`u_slot_value`] for ML-DSA-44,
/// [`DigitValue::Ud`] otherwise).
#[must_use]
pub const fn u_slot_value_for(set: Set, relation: Relation) -> DigitValue {
    match (set, relation) {
        (Set::S44, _) => u_slot_value(relation),
        (Set::S65 | Set::S87, Relation::HFips) => DigitValue::Ld,
        (Set::S65 | Set::S87, _) => DigitValue::Ud,
    }
}

/// The projection outputs of `set`: [`DigitValue::K`] (four digits, `B = 2^37`) for
/// ML-DSA-44, [`DigitValue::K5`] (five digits, `B = 2^39`) otherwise.
#[must_use]
pub const fn k_value(set: Set) -> DigitValue {
    match set {
        Set::S44 => DigitValue::K,
        Set::S65 | Set::S87 => DigitValue::K5,
    }
}

/// The committed columns of a value, in digit order (lane 0).
#[must_use]
pub fn value_columns(value: DigitValue) -> Vec<Column> {
    value_lane_columns(value, 0)
}

/// The committed columns of a value in lane `lane`, in digit order.
#[must_use]
pub fn value_lane_columns(value: DigitValue, lane: u8) -> Vec<Column> {
    (0..value.layout().len() as u8)
        .map(|digit| Column::Digit { value, digit, lane })
        .collect()
}

/// The canonical polynomial order of a group under `relation`: H-exact
/// puts `bz_lo, bz_hi` in the slots of `Z_lo, Z_hi` and appends `a_z` to `G_main4`; H-FIPS
/// also puts `l_lo, l_hi` in the slots of `U_lo, U_hi`, appends `s, delta, e` to `G_main4`
/// and `u_h` to `G_2`.
#[must_use]
pub fn group_columns(role: GroupRole, relation: Relation) -> Vec<Column> {
    let lanes = 0..W1_BITS as u8;
    let z = value_columns(z_value(relation));
    let u = value_columns(u_slot_value(relation));
    let fips = relation == Relation::HFips;
    let slots = Group::keccak_slot_groups(Set::S44) as u8;
    match role {
        Group::Main16 => vec![z[0], u[0], Column::V_LO, Column::V_HI],
        Group::Main => [z[1], u[1]]
            .into_iter()
            .chain(lanes.map(|t| Column::W1Bit(t, 0)))
            .chain(z[2..].iter().copied())
            .chain(FipsColumn::ALL.into_iter().map(|c| Column::Fips(c, 0)).filter(|_| fips))
            .collect(),
        Group::KeccakA(y) => (0..5_u8)
            .flat_map(|x| (0..slots).map(move |q| Column::KeccakA { x, y, q }))
            .collect(),
        Group::KeccakC => (0..5_u8)
            .flat_map(|x| (0..slots).map(move |q| Column::KeccakC { x, q }))
            .collect(),
        Group::KeccakCPrime => (0..5_u8)
            .flat_map(|x| (0..slots).map(move |q| Column::KeccakCPrime { x, q }))
            .collect(),
        Group::Level1 => lanes.map(|t| Column::carry(1, t)).collect(),
        Group::Level2 => lanes
            .clone()
            .map(Column::Parity2)
            .chain(lanes.map(|t| Column::carry(2, t)))
            .chain(std::iter::once(Column::HINT_SLACK).filter(|_| fips))
            .collect(),
        Group::Level3 => std::iter::once(Column::carry(3, 0))
            .chain((1..Group::Level3.num_polynomials(Set::S44, relation) as u8).map(Column::Zero))
            .collect(),
        Group::Proj => value_columns(DigitValue::K),
        Group::Mult => (0..MULTIPLICITY_DIGITS as u8)
            .map(Column::Multiplicity)
            .collect(),
    }
}

/// The canonical polynomial order of a group of `set` under `relation`: [`group_columns`] for
/// ML-DSA-44; for ML-DSA-65 and ML-DSA-87 every main-domain column is a family of lanes, in
/// lane order inside the family (H-FIPS puts `L_lo, L_mid, a_L` in the
/// slots of `U_lo, U_mid, a_u`, appends `s^(rho)`, then `delta^(rho)`, then `e^(rho)` to
/// `G_main4` and `u_h` to `G_2`):
///
/// * `G_main16`: `Z_lo^(s)`, `U_lo^(rho)`, `V_lo^(rho)`, `V_hi^(rho)`, so the looked-up
///   columns `Z_lo, U_lo` are the first `l + k` polynomials of the stacked lane;
/// * `G_main4`: `Z_mid^(s)`, `a_z^(s)`, `U_mid^(rho)`, `a_u^(rho)`, then the bits
///   `b_t^(rho)` bit-major (`t` outer, `rho` inner);
/// * `KeccakA(y)`, `KeccakC`, `KeccakCPrime`: `x in 0..5` outer, `q in 0..Q` inner (`Q = 11` for
///   ML-DSA-44 and ML-DSA-65, `14` for ML-DSA-87);
/// * `G_1`: `kappa^(1)_t`; `G_2`: `y^(2)_t`, `kappa^(2)_t`; `G_3`: `kappa^(3)` and zero
///   polynomials up to `G_2`'s count; `G_k`: the five projection digits; `G_m` as for ML-DSA-44.
///
/// # Panics
/// For ML-DSA-65 or ML-DSA-87 under a relation the set does not define
/// ([`Relation::is_available`]).
#[must_use]
pub fn group_columns_for(set: Set, role: GroupRole, relation: Relation) -> Vec<Column> {
    if set == Set::S44 {
        return group_columns(role, relation);
    }
    assert!(relation.is_available(set), "relation {relation} is not available for ML-DSA-{}", set.name());
    let slots = Group::keccak_slot_groups(set) as u8;
    match role {
        Group::KeccakA(y) => {
            return (0..5_u8)
                .flat_map(|x| (0..slots).map(move |q| Column::KeccakA { x, y, q }))
                .collect();
        }
        Group::KeccakC => {
            return (0..5_u8)
                .flat_map(|x| (0..slots).map(move |q| Column::KeccakC { x, q }))
                .collect();
        }
        Group::KeccakCPrime => {
            return (0..5_u8)
                .flat_map(|x| (0..slots).map(move |q| Column::KeccakCPrime { x, q }))
                .collect();
        }
        _ => {}
    }
    let shape = SetShape::of(set);
    let family = |value: DigitValue, digit: u8, lanes: usize| {
        (0..lanes as u8).map(move |lane| Column::Digit { value, digit, lane })
    };
    let (z, u) = (z_value_for(set, relation), u_slot_value_for(set, relation));
    let (lanes_k, lanes_l) = (shape.lanes_k, shape.lanes_l);
    let bits = 0..shape.w1_bits as u8;
    let fips = relation == Relation::HFips;
    let fips_columns = FipsColumn::ALL
        .into_iter()
        .filter(|_| fips)
        .flat_map(move |c| (0..lanes_k as u8).map(move |lane| Column::Fips(c, lane)));
    match role {
        Group::Main16 => family(z, 0, lanes_l)
            .chain(family(u, 0, lanes_k))
            .chain(family(DigitValue::V, 0, lanes_k))
            .chain(family(DigitValue::V, 1, lanes_k))
            .collect(),
        Group::Main => family(z, 1, lanes_l)
            .chain(family(z, 2, lanes_l))
            .chain(family(u, 1, lanes_k))
            .chain(family(u, 2, lanes_k))
            .chain(bits.flat_map(|t| (0..lanes_k as u8).map(move |lane| Column::W1Bit(t, lane))))
            .chain(fips_columns)
            .collect(),
        Group::Level1 => bits.map(|t| Column::carry(1, t)).collect(),
        Group::Level2 => bits
            .clone()
            .map(Column::Parity2)
            .chain(bits.map(|t| Column::carry(2, t)))
            .chain(std::iter::once(Column::HINT_SLACK).filter(|_| fips))
            .collect(),
        Group::Level3 => std::iter::once(Column::carry(3, 0))
            .chain((1..Group::Level3.num_polynomials(set, relation) as u8).map(Column::Zero))
            .collect(),
        Group::Proj => value_columns(DigitValue::K5),
        Group::Mult => (0..MULTIPLICITY_DIGITS as u8).map(Column::Multiplicity).collect(),
        Group::KeccakA(_) | Group::KeccakC | Group::KeccakCPrime => unreachable!(),
    }
}

/// One committed group's exact public shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupDescriptor {
    /// Role (wire position).
    pub role: GroupRole,
    /// Commitment round (1 before the bridge seed, 2 after it).
    pub round: u8,
    /// PCS family.
    pub family: PcsFamily,
    /// Variables of the logical domain.
    pub logical_vars: usize,
    /// Variables of the committed (zero-extended) table.
    pub physical_vars: usize,
    /// Polynomials in canonical order.
    pub columns: Vec<Column>,
}

impl GroupDescriptor {
    fn new(role: GroupRole, set: Set, d_sig: usize, relation: Relation) -> Self {
        Self {
            role,
            round: role.round(),
            family: role.family(),
            logical_vars: role.logical_vars(set, d_sig),
            physical_vars: role.physical_vars(set, d_sig),
            columns: group_columns_for(set, role, relation),
        }
    }

    /// Logical cells per polynomial.
    #[must_use]
    pub const fn logical_cells(&self) -> usize {
        1 << self.logical_vars
    }

    /// Trailing zero pad variables.
    #[must_use]
    pub const fn pad_vars(&self) -> usize {
        self.physical_vars - self.logical_vars
    }

    /// Number of polynomials.
    #[must_use]
    pub fn polynomial_count(&self) -> usize {
        self.columns.len()
    }

    /// The polynomials opened through stage-3 claims (every polynomial but the zero ones),
    /// in group order.
    #[must_use]
    pub fn claimed_count(&self) -> usize {
        self.columns.iter().filter(|c| !c.is_zero()).count()
    }

    /// Position of `column` in the group, if it belongs to it.
    #[must_use]
    pub fn position(&self, column: Column) -> Option<usize> {
        self.columns.iter().position(|&c| c == column)
    }
}

/// The looked-up columns of one lookup stack, in slot order
/// (H-FIPS puts `u_h` in slot 6 of stack 2): `[Z_lo | U_lo]` (the first two `G_main16`
/// columns, so the stack is the first half of the stacked lane), the carries of levels 1 to
/// 3, and `[K_0 | K_1]` on the projection domain.
#[must_use]
pub fn stack_columns(stack: usize, relation: Relation) -> Vec<Column> {
    let lanes = 0..W1_BITS as u8;
    match stack {
        0 => group_columns(Group::Main16, relation)
            .into_iter()
            .filter(|c| c.is_lookup())
            .collect(),
        1 => lanes.map(|t| Column::carry(1, t)).collect(),
        2 => lanes
            .map(|t| Column::carry(2, t))
            .chain(std::iter::once(Column::HINT_SLACK).filter(|_| relation == Relation::HFips))
            .collect(),
        3 => vec![Column::carry(3, 0)],
        4 => vec![Column::K_0, Column::K_1],
        _ => panic!("five lookup stacks"),
    }
}

/// The domain of each lookup stack of ML-DSA-44: the main domain, `D_1`, `D_2`, `D_3` and the
/// projection domain `D_k` ([`stack_domains`] for every set).
pub const STACK_DOMAINS: [GroupRole; 5] =
    [Group::Main, Group::Level1, Group::Level2, Group::Level3, Group::Proj];

/// The column counts of the main-domain lookup stacks: the `l + k` looked-up polynomials of
/// `G_main16` (`Z_lo` and `U_lo` lanes, in group order) split by the binary decomposition of
/// their count, largest stack first, so that no stack pads its slots to a power of two:
/// `[2]` (ML-DSA-44), `[8, 2, 1]` (ML-DSA-65), `[8, 4, 2, 1]` (ML-DSA-87).
#[must_use]
pub fn main_stack_sizes(set: Set) -> Vec<usize> {
    let shape = SetShape::of(set);
    let count = shape.lanes_k + shape.lanes_l;
    (0..usize::BITS).rev().map(|bit| count & (1 << bit)).filter(|&size| size != 0).collect()
}

/// The domain of each lookup stack of `set`: the main-domain stacks
/// ([`main_stack_sizes`]), then `D_1`, `D_2`, `D_3` and `D_k`.
#[must_use]
pub fn stack_domains(set: Set) -> Vec<GroupRole> {
    std::iter::repeat_n(Group::Main, main_stack_sizes(set).len())
        .chain([Group::Level1, Group::Level2, Group::Level3, Group::Proj])
        .collect()
}

/// The looked-up columns of lookup stack `stack` of `set`, in slot order: the main-domain
/// stacks take consecutive runs of the looked-up `G_main16` polynomials, then the carries
/// of levels 1 to 3 and `[K_0 | K_1]` ([`stack_columns`] for ML-DSA-44).
///
/// # Panics
/// For a stack index beyond the set's stacks.
#[must_use]
pub fn stack_columns_for(set: Set, stack: usize, relation: Relation) -> Vec<Column> {
    if set == Set::S44 {
        return stack_columns(stack, relation);
    }
    let sizes = main_stack_sizes(set);
    let bits = 0..SetShape::of(set).w1_bits as u8;
    if stack < sizes.len() {
        let start: usize = sizes[..stack].iter().sum();
        return group_columns_for(set, Group::Main16, relation)
            .into_iter()
            .filter(|c| c.is_lookup())
            .skip(start)
            .take(sizes[stack])
            .collect();
    }
    match stack - sizes.len() {
        0 => bits.map(|t| Column::carry(1, t)).collect(),
        1 => bits
            .map(|t| Column::carry(2, t))
            .chain(std::iter::once(Column::HINT_SLACK).filter(|_| relation == Relation::HFips))
            .collect(),
        2 => vec![Column::carry(3, 0)],
        3 => value_columns(DigitValue::K5)[..2].to_vec(),
        _ => panic!("{} lookup stacks", sizes.len() + 4),
    }
}

/// Complete public commitment manifest of one batch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitmentManifest {
    set: Set,
    records: usize,
    padded: usize,
    d_sig: usize,
    relation: Relation,
    hash_field: HashField,
    groups: Vec<GroupDescriptor>,
    catalog_row: String,
}

impl CommitmentManifest {
    /// The `HashField::Gf2` ML-DSA-44 manifest of a batch of `records` live records
    /// (`1 <= N <= 2^15`) under the default relation H-FIPS.
    pub fn for_batch(records: usize) -> Result<Self, PlanError> {
        Self::for_relation(records, Relation::default())
    }

    /// The `HashField::Fp` manifest of a ML-DSA-44 batch of `records` live records under the
    /// default relation H-FIPS.
    pub fn for_batch_fp(records: usize) -> Result<Self, PlanError> {
        Self::for_relation_fp(records, Relation::default())
    }

    /// The `HashField::Gf2` ML-DSA-44 manifest of a batch of `records` live records under `relation`.
    pub fn for_relation(records: usize, relation: Relation) -> Result<Self, PlanError> {
        Self::for_set(Set::S44, records, relation)
    }

    /// The `HashField::Fp` ML-DSA-44 manifest of a batch of `records` live records under `relation`.
    pub fn for_relation_fp(records: usize, relation: Relation) -> Result<Self, PlanError> {
        Self::for_set_fp(Set::S44, records, relation)
    }

    /// The `HashField::Gf2` manifest of a batch of `records` live records of parameter set
    /// `set` under `relation`.
    pub fn for_set(set: Set, records: usize, relation: Relation) -> Result<Self, PlanError> {
        Self::for_hash_field(set, records, relation, HashField::Gf2)
    }

    /// The `HashField::Fp` manifest of a batch of `records` live records of parameter set
    /// `set` under `relation`.
    pub fn for_set_fp(set: Set, records: usize, relation: Relation) -> Result<Self, PlanError> {
        Self::for_hash_field(set, records, relation, HashField::Fp)
    }

    /// The manifest of a batch of `records` live records of parameter set `set` under
    /// `relation` and `hash_field`.
    pub fn for_hash_field(
        set: Set,
        records: usize,
        relation: Relation,
        hash_field: HashField,
    ) -> Result<Self, PlanError> {
        let (padded, d_sig) = padded_size(records)?;
        let layout = BatchLayout::for_hash_field(set, d_sig, relation, hash_field)
            .ok_or(PlanError::NotInCatalog { d_sig, relation })?;
        Ok(Self {
            set,
            records,
            padded,
            d_sig,
            relation,
            hash_field,
            groups: hash_field
                .groups()
                .iter()
                .map(|&role| GroupDescriptor::new(role, set, d_sig, relation))
                .collect(),
            catalog_row: layout.row_name(),
        })
    }

    /// The parameter set.
    #[must_use]
    pub const fn set(&self) -> Set {
        self.set
    }

    /// The field in which the commitment hash is proven (`Gf2` or `Fp`).
    #[must_use]
    pub const fn hash_field(&self) -> HashField {
        self.hash_field
    }

    /// The dimensions of the parameter set.
    #[must_use]
    pub const fn shape(&self) -> SetShape {
        SetShape::of(self.set)
    }

    /// True when the vector index is a polynomial lane (ML-DSA-65 and ML-DSA-87): every
    /// main-domain column is a family of `k` or `l` polynomials on `(j, i)`.
    #[must_use]
    pub const fn has_lanes(&self) -> bool {
        self.shape().d_vec == 0
    }

    /// `gamma_2` of the parameter set (the weight `2 gamma_2` of `w_1'` in the ring identity).
    #[must_use]
    pub const fn gamma2(&self) -> i64 {
        agg_witness::dispatch_set!(self.set, |P| P::GAMMA2)
    }

    /// Cells of one main-domain polynomial per signature (`2^(d_main - d_sig)`).
    #[must_use]
    pub fn lane_cells(&self) -> usize {
        1 << (self.d_main() - self.d_sig)
    }

    /// The relation the proof certifies.
    #[must_use]
    pub const fn relation(&self) -> Relation {
        self.relation
    }

    /// Live records `N`.
    #[must_use]
    pub const fn records(&self) -> usize {
        self.records
    }

    /// `N*`.
    #[must_use]
    pub const fn padded_records(&self) -> usize {
        self.padded
    }

    /// `log2 N*`.
    #[must_use]
    pub const fn d_sig(&self) -> usize {
        self.d_sig
    }

    /// `d_main = 8 + d_vec + d_sig` (`10 + d_sig` for ML-DSA-44).
    #[must_use]
    pub fn d_main(&self) -> usize {
        self.group(Group::Main).logical_vars
    }

    /// `d_K = 10 + d_sig`, the logical variable count of the Keccak domain.
    #[must_use]
    pub const fn d_keccak(&self) -> usize {
        agg_pcs_config::catalogs::D_POS + self.d_sig
    }

    /// Rounds of each sum-check stage: `d_main` under `Gf2`, `max(d_main, d_K)` under `Fp`.
    #[must_use]
    pub fn d_stage(&self) -> usize {
        match self.hash_field {
            HashField::Gf2 => self.d_main(),
            HashField::Fp => self.d_main().max(self.d_keccak()),
        }
    }

    /// Number of `2^13`-signature segments `S`.
    #[must_use]
    pub const fn segments(&self) -> usize {
        1 << self.d_sig.saturating_sub(13)
    }

    /// The grouped catalog row id bound in the instance record.
    #[must_use]
    pub fn catalog_row(&self) -> &str {
        &self.catalog_row
    }

    /// Every group in wire order.
    #[must_use]
    pub fn groups(&self) -> &[GroupDescriptor] {
        &self.groups
    }

    /// The descriptor of `role`.
    #[must_use]
    pub fn group(&self, role: GroupRole) -> &GroupDescriptor {
        &self.groups[role.index_in(self.hash_field)]
    }

    /// Logical variables of a domain (the main domain for both main groups).
    #[must_use]
    pub fn domain_vars(&self, domain: GroupRole) -> usize {
        self.group(domain).logical_vars
    }

    /// `(group, position)` of a column.
    #[must_use]
    pub fn locate(&self, column: Column) -> (GroupRole, usize) {
        let role = column.group();
        let position = self
            .group(role)
            .position(column)
            .expect("every column belongs to its group");
        (role, position)
    }

    /// The value `z` of the set and relation ([`z_value_for`]).
    #[must_use]
    pub const fn z_value(&self) -> DigitValue {
        z_value_for(self.set, self.relation)
    }

    /// The value in the slots of `U_lo` ([`u_slot_value_for`]).
    #[must_use]
    pub const fn u_slot_value(&self) -> DigitValue {
        u_slot_value_for(self.set, self.relation)
    }

    /// The projection outputs ([`k_value`]).
    #[must_use]
    pub const fn k_value(&self) -> DigitValue {
        k_value(self.set)
    }

    /// The digit layout of the projection outputs ([`K16_DIGITS`] or [`K39_DIGITS`]).
    #[must_use]
    pub const fn proj_digits(&self) -> DigitLayout {
        match self.set {
            Set::S44 => K16_DIGITS,
            Set::S65 | Set::S87 => K39_DIGITS,
        }
    }

    /// The bound `B` of the projection outputs (`k in [-B, B)`).
    #[must_use]
    pub const fn proj_bound(&self) -> i64 {
        match self.set {
            Set::S44 => PROJ_BOUND,
            Set::S65 | Set::S87 => PROJ_BOUND_39,
        }
    }

    /// Bytes of `Psi in {0, 1}^(128 x P)`: 16, 24 or 32 KiB.
    #[must_use]
    pub const fn proj_bytes(&self) -> usize {
        PROJ_ROWS * self.shape().positions / 8
    }

    /// True under relation H-FIPS (the hint is part of the relation).
    #[must_use]
    pub fn is_fips(&self) -> bool {
        self.relation == Relation::HFips
    }

    /// `omega` of the parameter set (the right side of H-FIPS's hint-weight identity).
    #[must_use]
    pub const fn omega(&self) -> usize {
        agg_witness::dispatch_set!(self.set, |P| P::OMEGA)
    }

    /// `m_w - 1` of the parameter set (the top value of `w_1'`, the target of H-FIPS's `e = -1`
    /// wrap): 43 for ML-DSA-44, 15 for ML-DSA-65 and ML-DSA-87.
    #[must_use]
    pub const fn w1_top(&self) -> i64 {
        agg_witness::dispatch_set!(self.set, |P| P::M_W - 1)
    }

    /// The H-FIPS column `column` of every row lane (one for ML-DSA-44).
    #[must_use]
    pub fn fips_lanes(&self, column: FipsColumn) -> Vec<Column> {
        (0..self.shape().lanes_k as u8).map(|lane| Column::Fips(column, lane)).collect()
    }

    /// The looked-up columns of lookup stack `stack` ([`stack_columns_for`]).
    #[must_use]
    pub fn stack_columns(&self, stack: usize) -> Vec<Column> {
        match self.hash_field {
            HashField::Gf2 => stack_columns_for(self.set, stack, self.relation),
            HashField::Fp => {
                let main_stacks = self.main_stacks();
                if stack < main_stacks {
                    stack_columns_for(self.set, stack, self.relation)
                } else if stack == main_stacks {
                    value_columns(self.k_value())[..2].to_vec()
                } else {
                    panic!("{} lookup stacks under Fp", main_stacks + 1)
                }
            }
        }
    }

    /// The domain of every lookup stack ([`stack_domains`]).
    #[must_use]
    pub fn stack_domains(&self) -> Vec<GroupRole> {
        match self.hash_field {
            HashField::Gf2 => stack_domains(self.set),
            HashField::Fp => std::iter::repeat_n(Group::Main, self.main_stacks())
                .chain([Group::Proj])
                .collect(),
        }
    }

    /// The number of main-domain lookup stacks (1 for ML-DSA-44, 3 for ML-DSA-65, 4 for
    /// ML-DSA-87).
    #[must_use]
    pub fn main_stacks(&self) -> usize {
        main_stack_sizes(self.set).len()
    }

    /// The lookup stack of the carries of level `level` (`1..=3`, `HashField::Gf2` only).
    #[must_use]
    pub fn carry_stack(&self, level: usize) -> usize {
        assert_eq!(self.hash_field, HashField::Gf2, "no bridge carry stacks under Fp");
        assert!((1..=3).contains(&level), "levels 1 to 3");
        self.main_stacks() + level - 1
    }

    /// The lookup stack `[K_0 | K_1]` of the projection digits (the last one).
    #[must_use]
    pub fn proj_stack(&self) -> usize {
        match self.hash_field {
            HashField::Gf2 => self.main_stacks() + 3,
            HashField::Fp => self.main_stacks(),
        }
    }

    /// The leaf-checked digits in the order of the leaf and link instances: `U_hi` (base 6)
    /// and `K_2, K_3` (base 8) under H-exact; `K_2, K_3` under H-FIPS.
    #[must_use]
    pub fn leaf_digits(&self) -> Vec<Column> {
        self.groups
            .iter()
            .flat_map(|group| group.columns.iter().copied())
            .filter(|column| column.is_leaf_digit())
            .collect()
    }

    /// The Boolean value digits proven by a zero-check at `r^(2)` (stage 3), in manifest
    /// order: `bz_hi` and `a_z` under H-exact, also `l_hi` under H-FIPS.
    #[must_use]
    pub fn bool_digits(&self) -> Vec<Column> {
        self.groups
            .iter()
            .flat_map(|group| group.columns.iter().copied())
            .filter(|column| column.is_bool_digit())
            .collect()
    }

    /// Committed polynomials opened at `r^(3)` through stage-3 claims (every group but `G_m`,
    /// without `G_3`'s zero polynomials): 36 under H-exact, 40 under H-FIPS.
    #[must_use]
    pub fn stage3_polynomials(&self) -> usize {
        self.groups
            .iter()
            .filter(|g| g.role != Group::Mult)
            .map(GroupDescriptor::claimed_count)
            .sum()
    }

    /// The public shape of the lookup: under `Gf2` the five stacks
    /// `[Z_lo | U_lo]` on the main domain, `kappa^(1)` (6 columns in 8 slots) on `D_1`,
    /// `kappa^(2)` on `D_2`, `kappa^(3)` on `D_3`, `[K_0 | K_1]` on `D_k`; under `Fp` the
    /// main-domain stack(s) `[Z_lo | U_lo]` and `[K_0 | K_1]`; one table `T_16`, two
    /// challenges.
    #[must_use]
    pub fn lookup_shape(&self) -> LookupShape {
        LookupShape {
            stacks: self
                .stack_domains()
                .iter()
                .enumerate()
                .map(|(stack, &domain)| StackShape {
                    columns: self.stack_columns(stack).len(),
                    vars: self.domain_vars(domain),
                    table: 0,
                    shift: LOOKUP_SHIFT,
                })
                .collect(),
            tables: vec![Table::new(TABLE_VARS as u32)],
            challenges: LOOKUP_CHALLENGES,
        }
    }

    /// Total committed polynomials (54 under H-exact, 59 under H-FIPS).
    #[must_use]
    pub fn total_polynomials(&self) -> usize {
        self.groups
            .iter()
            .map(GroupDescriptor::polynomial_count)
            .sum()
    }

    /// Total logical committed cells.
    #[must_use]
    pub fn total_logical_cells(&self) -> usize {
        self.groups
            .iter()
            .map(|group| group.polynomial_count() * group.logical_cells())
            .sum()
    }

    /// True when the work plan commits exactly these shapes (same batch, same physical
    /// layouts and polynomial counts).
    #[must_use]
    pub fn matches_plan(&self, plan: &WorkPlan) -> bool {
        plan.records == self.records
            && plan.padded == self.padded
            && plan.hash_field() == self.hash_field
            && plan.groups().len() == self.groups.len()
            && self.groups.iter().all(|group| {
                let profile = plan.profile(group.role).group;
                plan.logical_vars(group.role) == group.logical_vars
                    && profile.num_vars() == group.physical_vars
                    && profile.num_polynomials() == group.polynomial_count()
            })
            && plan.relation() == self.relation
            && plan.set() == self.set
            && plan.layout.row_name() == self.catalog_row
    }

    /// Canonical byte encoding of the manifest part of the instance record.
    ///
    /// `LE16` wire version, parameter-set byte (`0x44`, `0x65`, `0x87`), relation byte ([`Relation::id`],
    /// `1` = H-exact, `2` = H-FIPS), `LE64` `N`, `LE64` `N*`, digit layout id
    /// ([`DIGIT_LAYOUT_ID`] = 2, the lookup layout with the projected quotient), group count
    /// (`7` under `Gf2`, `11` under `Fp`); per group: role, round, family (0 = width 4, 1 =
    /// width 16), logical and physical variable counts, polynomial count and one alphabet byte
    /// per polynomial ([`Column::alphabet_code`]); the catalog row id (`LE16` length, ASCII);
    /// under `Gf2` only, the bridge parameters `(J_t, m_t, k_t)` of levels 1 to 3 (`LE64`,
    /// `LE64`, `U8`); the lookup layout: challenge count, table count, one width byte per
    /// table, stack count, and per stack the column count, variables, table and storage shift
    /// (`U8, U8, U8, LE32`). The public-record digest and the Keccak-side profile id complete
    /// the instance record elsewhere.
    #[must_use]
    pub fn binding_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(160);
        bytes.extend_from_slice(&WIRE_VERSION.to_le_bytes());
        bytes.push(self.set.id());
        bytes.push(self.relation.id());
        bytes.extend_from_slice(&(self.records as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.padded as u64).to_le_bytes());
        bytes.push(DIGIT_LAYOUT_ID);
        bytes.push(self.groups.len() as u8);
        for group in &self.groups {
            bytes.push(role_id(group.role));
            bytes.push(group.round);
            bytes.push(group.family.transcript_id());
            bytes.push(group.logical_vars as u8);
            bytes.push(group.physical_vars as u8);
            bytes.push(group.polynomial_count() as u8);
            bytes.extend(group.columns.iter().map(|column| column.alphabet_code()));
        }
        bytes.extend_from_slice(&(self.catalog_row.len() as u16).to_le_bytes());
        bytes.extend_from_slice(self.catalog_row.as_bytes());
        if self.hash_field == HashField::Gf2 {
            // The bridge shape (J_t, m_t, k_t); this wire version admits the set's production
            // shape only. Under Fp there is no bridge and this field is omitted.
            for level in &agg_bridge::Shape::production(self.set).levels {
                bytes.extend_from_slice(&(level.cols as u64).to_le_bytes());
                bytes.extend_from_slice(&(level.rows as u64).to_le_bytes());
                bytes.push(level.k as u8);
            }
        }
        let lookup = self.lookup_shape();
        bytes.push(lookup.challenges as u8);
        bytes.push(lookup.tables.len() as u8);
        bytes.extend(lookup.tables.iter().map(|table| table.width as u8));
        bytes.push(lookup.stacks.len() as u8);
        for stack in &lookup.stacks {
            bytes.push(stack.columns as u8);
            bytes.push(stack.vars as u8);
            bytes.push(stack.table as u8);
            bytes.extend_from_slice(&stack.shift.to_le_bytes());
        }
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_shapes_and_digit_counts_match_the_committed_layout() {
        let manifest = CommitmentManifest::for_relation(256, Relation::HExact).unwrap();
        let counts: Vec<usize> = manifest
            .groups()
            .iter()
            .map(GroupDescriptor::polynomial_count)
            .collect();
        assert_eq!(counts, [4, 9, 6, 12, 12, 4, 7]);
        assert_eq!(manifest.total_polynomials(), 54);
        assert_eq!(manifest.stage3_polynomials(), 36);
        // G_3: the carry, then 11 zero polynomials (G_2's layout).
        let level3 = &manifest.group(Group::Level3).columns;
        assert_eq!((level3[0], level3[1], level3[11]), (Column::carry(3, 0), Column::Zero(1), Column::Zero(11)));
        assert_eq!(manifest.group(Group::Level3).claimed_count(), 1);
        assert_eq!(Column::Zero(1).alphabet_code(), ZERO_ALPHABET_CODE);
        // G_k: K_0, K_1 (looked up), K_2, K_3 (base 8, leaf and link); V_lo, V_hi unchecked.
        assert_eq!(
            manifest.group(Group::Proj).columns,
            [Column::K_0, Column::K_1, Column::K_2, Column::K_3]
        );
        assert!(Column::K_0.is_lookup() && Column::K_1.is_lookup());
        assert!(Column::K_2.is_leaf_digit() && Column::K_3.is_leaf_digit());
        assert!(!Column::V_LO.is_lookup() && Column::V_LO.is_wide() && Column::V_HI.is_unchecked());
        assert_eq!(Column::V_HI.alphabet_code(), UNCHECKED_ALPHABET_CODE);
        assert_eq!(K16_DIGITS.range(), (-PROJ_BOUND, PROJ_BOUND - 1));
        assert_eq!(K16_DIGITS.constant(), (1 << 34) + (1 << 31) + (1 << 15));
        assert_eq!(K16_DIGITS.decompose(0), vec![-(1 << 15), -(1 << 15), -4, 0]);
        assert_eq!(manifest.bool_digits(), [Column::ZX_HI, Column::A_Z]);
        assert_eq!(manifest.z_value(), DigitValue::Zx);
        let lookups: Vec<Column> = manifest
            .groups()
            .iter()
            .flat_map(|g| g.columns.clone())
            .filter(|c| c.is_lookup())
            .collect();
        // 2 main-domain columns, 6 + 6 + 1 carries and K_0, K_1.
        assert_eq!(lookups.len(), 17);
        let names: Vec<String> = manifest.leaf_digits().iter().map(|c| c.name()).collect();
        assert_eq!(names, ["u_hi", "k_2", "k_3"]);
        assert_eq!(Column::U_HI.base(), Some(6));
        assert_eq!(Column::U_HI.alphabet(), Alphabet::Balanced(6));
        assert!(Column::U_HI.is_leaf_digit());
        for group in manifest.groups() {
            assert_eq!(group.round, group.role.round());
            assert_eq!(group.family, group.role.family());
            assert_eq!(usize::from(role_id(group.role)), group.role.index());
            for (position, &column) in group.columns.iter().enumerate() {
                assert_eq!(manifest.locate(column), (group.role, position));
                assert_eq!(column.group(), group.role);
            }
        }
        // Every looked-up column is a width-16 column, every width-4 column fits [-8, 7].
        for group in manifest.groups() {
            for column in &group.columns {
                let alphabet = column.alphabet();
                assert!(group.family.min_value() <= alphabet.min());
                assert!(alphabet.max() <= group.family.max_value());
                if column.is_lookup() {
                    assert_eq!(group.family, PcsFamily::Width16);
                    assert_eq!((alphabet.min(), alphabet.max()), (-(1 << 15), (1 << 15) - 1));
                }
            }
        }
    }

    #[test]
    fn h_exact_groups_follow_the_h_exact_layout() {
        let manifest = CommitmentManifest::for_relation(256, Relation::HExact).unwrap();
        let counts: Vec<usize> = manifest
            .groups()
            .iter()
            .map(GroupDescriptor::polynomial_count)
            .collect();
        assert_eq!(counts, [4, 9, 6, 12, 12, 4, 7]);
        assert_eq!((manifest.total_polynomials(), manifest.stage3_polynomials()), (54, 36));
        let names = |role| -> Vec<String> {
            manifest.group(role).columns.iter().map(|c| c.name()).collect()
        };
        assert_eq!(names(Group::Main16), ["bz_lo", "u_lo", "v_lo", "v_hi"]);
        assert_eq!(
            names(Group::Main),
            ["bz_hi", "u_hi", "b_0", "b_1", "b_2", "b_3", "b_4", "b_5", "a_z"]
        );
        let leaf: Vec<String> = manifest.leaf_digits().iter().map(|c| c.name()).collect();
        let bools: Vec<String> = manifest.bool_digits().iter().map(|c| c.name()).collect();
        assert_eq!(
            (leaf, bools),
            (
                vec!["u_hi".to_owned(), "k_2".to_owned(), "k_3".to_owned()],
                vec!["bz_hi".to_owned(), "a_z".to_owned()]
            )
        );
        assert_eq!(manifest.z_value(), DigitValue::Zx);
        assert_eq!(value_columns(DigitValue::Zx), [Column::ZX_LO, Column::ZX_HI, Column::A_Z]);
        assert!(Column::ZX_LO.is_lookup() && !Column::ZX_HI.is_lookup());
        assert_eq!((Column::ZX_HI.base(), Column::A_Z.alphabet()), (None, Alphabet::Bool));
        assert_eq!((Column::ZX_HI.alphabet_code(), Column::A_Z.alphabet_code()), (0, 0));
        assert!(Column::ZX_HI.is_bool_digit() && !Column::W1Bit(0, 0).is_bool_digit());
        // The main lookup stack holds bz_lo and u_lo.
        assert_eq!(manifest.stack_columns(0), [Column::ZX_LO, Column::U_LO]);
        for group in manifest.groups() {
            for (position, &column) in group.columns.iter().enumerate() {
                assert_eq!(manifest.locate(column), (group.role, position));
                let alphabet = column.alphabet();
                assert!(group.family.min_value() <= alphabet.min());
                assert!(alphabet.max() <= group.family.max_value());
            }
        }
        // b_t keep positions 2..7 of G_main4 under both relations.
        let fips = CommitmentManifest::for_batch(256).unwrap();
        assert_eq!(fips.relation(), Relation::HFips);
        for t in 0..W1_BITS as u8 {
            assert_eq!(manifest.locate(Column::W1Bit(t, 0)), fips.locate(Column::W1Bit(t, 0)));
            assert_eq!(manifest.locate(Column::W1Bit(t, 0)), (Group::Main, 2 + usize::from(t)));
        }
        // H-exact z: z = S_zlo + 2^16 B_zhi + 130,915 a_z - 98,225 over exactly [-B_z, B_z].
        let form = DigitValue::Zx.layout();
        assert_eq!((form.weights(), form.constant()), (vec![1, 1 << 16, 130_915], -98_225));
        assert_eq!(form.range(), (-130_993, 130_993));
    }

    #[test]
    fn h_fips_groups_follow_the_h_fips_layout() {
        let manifest = CommitmentManifest::for_relation(256, Relation::HFips).unwrap();
        assert!(manifest.is_fips());
        let counts: Vec<usize> = manifest
            .groups()
            .iter()
            .map(GroupDescriptor::polynomial_count)
            .collect();
        assert_eq!(counts, [4, 12, 6, 13, 13, 4, 7]);
        assert_eq!((manifest.total_polynomials(), manifest.stage3_polynomials()), (59, 40));
        let names = |role| -> Vec<String> {
            manifest.group(role).columns.iter().map(|c| c.name()).collect()
        };
        assert_eq!(names(Group::Main16), ["bz_lo", "l_lo", "v_lo", "v_hi"]);
        assert_eq!(
            names(Group::Main),
            ["bz_hi", "l_hi", "b_0", "b_1", "b_2", "b_3", "b_4", "b_5", "a_z", "s", "delta", "e"]
        );
        assert_eq!(names(Group::Level2).last().map(String::as_str), Some("u_h"));
        assert_eq!(manifest.leaf_digits(), [Column::K_2, Column::K_3]);
        let bools: Vec<String> = manifest.bool_digits().iter().map(|c| c.name()).collect();
        assert_eq!(bools, ["bz_hi", "l_hi", "a_z"]);
        assert_eq!((manifest.z_value(), manifest.u_slot_value()), (DigitValue::Zx, DigitValue::L));
        // Alphabets and codes: s a bit, delta and e ternary, l_lo and u_h looked up.
        assert_eq!(Column::SIGN.alphabet(), Alphabet::Bool);
        assert_eq!(Column::DELTA.alphabet(), Alphabet::Ternary);
        assert_eq!(
            (Column::SIGN.alphabet_code(), Column::DELTA.alphabet_code(), Column::WRAP.alphabet_code()),
            (0, TERNARY_ALPHABET_CODE, TERNARY_ALPHABET_CODE)
        );
        assert!(Column::L_LO.is_lookup() && Column::HINT_SLACK.is_lookup());
        assert!(Column::L_HI.is_bool_digit() && !Column::SIGN.is_bool_digit());
        assert_eq!(Column::HINT_SLACK.domain(), Group::Level2);
        // u_h takes the free slot 6 of stack 2: the lookup keeps its depths and L.
        assert_eq!(manifest.stack_columns(2).len(), 7);
        assert_eq!(manifest.stack_columns(2)[6], Column::HINT_SLACK);
        let exact = CommitmentManifest::for_relation(256, Relation::HExact).unwrap().lookup_shape();
        let fips = manifest.lookup_shape();
        assert_eq!((fips.stacks[2].columns, exact.stacks[2].columns), (7, 6));
        for (a, b) in fips.stacks.iter().zip(&exact.stacks) {
            assert_eq!(a.depth(), b.depth());
        }
        assert_eq!(fips.lookups(0), exact.lookups(0));
        for group in manifest.groups() {
            for (position, &column) in group.columns.iter().enumerate() {
                assert_eq!(manifest.locate(column), (group.role, position));
                let alphabet = column.alphabet();
                assert!(group.family.min_value() <= alphabet.min());
                assert!(alphabet.max() <= group.family.max_value());
            }
        }
        // L = l_lo + 29,696 l_hi + 2^15 over exactly [0, gamma_2 - 1].
        let form = DigitValue::L.layout();
        assert_eq!((form.weights(), form.constant()), (vec![1, 29_696], 1 << 15));
        assert_eq!(form.range(), (0, 95_231));
        // 16 main-domain columns, u_h one element per signature, 4 * 128 projection digits
        // and G_3's 12 zero polynomials (2^14 logical cells each) at 2^15: 17,025.5.
        let full = CommitmentManifest::for_relation(1 << 15, Relation::HFips).unwrap();
        let per_signature = full.total_logical_cells() as f64 / f64::from(1 << 15);
        assert!((per_signature - 17_025.5).abs() < 1e-9, "{per_signature}");
        assert_eq!(full.catalog_row(), "mldsa-44-p16f-batch-32768-mult-final");
        // Relation byte 2, G_main4 at 12 polynomials with the ternary codes.
        let bytes = manifest.binding_bytes();
        assert_eq!(bytes[3], RELATION_H_FIPS);
        let main4 = &bytes[4 + 16 + 2 + 10..];
        assert_eq!(
            &main4[..18],
            &[1, 1, 0, 18, 18, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x23, 0x23]
        );
        assert_ne!(
            bytes,
            CommitmentManifest::for_relation(256, Relation::HExact).unwrap().binding_bytes()
        );
    }

    #[test]
    fn digit_layouts_give_the_l16_ranges() {
        // v = V_lo + 2^16 V_hi + 2^15.
        assert_eq!(V16_DIGITS.constant(), 1 << 15);
        assert_eq!(V16_DIGITS.weights(), vec![1, 1 << 16]);
        // H-exact: u = U_lo + 63,079 U_hi + 31,540 over exactly [-190,465, 190,465].
        assert_eq!(U16_DIGITS.weights(), vec![1, 63_079]);
        assert_eq!(U16_DIGITS.constant(), 31_540);
        assert_eq!(U16_DIGITS.range(), (-190_465, 190_465));
        assert_eq!(U16_DIGITS.decompose(190_465), vec![(1 << 15) - 1, 2]);
        assert_eq!(U16_DIGITS.decompose(-190_465), vec![-(1 << 15), -3]);
        for x in [-190_465, -100_000, -1, 0, 1, 63_079, 190_465] {
            let digits = U16_DIGITS.decompose(x);
            assert!(U16_DIGITS.in_alphabets(&digits), "{x}");
            assert_eq!(U16_DIGITS.compose(&digits), x);
        }
        for x in [190_466, -190_466, 1 << 18] {
            assert!(!U16_DIGITS.in_alphabets(&U16_DIGITS.decompose(x)), "{x}");
        }
        assert_eq!(V16_DIGITS.range(), (-(1 << 31), (1 << 31) - 1));
        assert_eq!(CARRY16_DIGITS.range(), (0, (1 << 16) - 1));
        assert_eq!(CARRY16_DIGITS.constant(), 1 << 15);
        // The digits of zero: V_lo = -2^15, V_hi = 0; U_lo = -31,540 (B_ulo = 1,228),
        // U_hi = 0 (H_u = 3).
        assert_eq!(U16_DIGITS.decompose(0), vec![-31_540, 0]);
        assert_eq!(V16_DIGITS.decompose(0), vec![-(1 << 15), 0]);
        for x in [-67_068_447, 0, 67_068_447, -(1 << 31), (1 << 31) - 1] {
            let digits = V16_DIGITS.decompose(x);
            assert!(V16_DIGITS.in_alphabets(&digits));
            assert_eq!(V16_DIGITS.compose(&digits), x);
        }
    }

    #[test]
    fn committed_elements_per_signature() {
        // H-exact: 13,312 main-domain elements, 96 + 12 + 0.5 bridge elements, 5.5 for G_3's
        // 11 zero polynomials, 4 * 128 projection digits and the 7 * 2^16 multiplicity cells
        // (14 per signature) at N = 2^15: 13,952; 14,560 physical at N = 1,024.
        let full = CommitmentManifest::for_relation(1 << 15, Relation::HExact).unwrap();
        assert_eq!(full.d_main(), 25);
        assert_eq!(full.segments(), 4);
        let per_signature = full.total_logical_cells() as f64 / f64::from(1 << 15);
        assert!((per_signature - 13_952.0).abs() < 1e-9, "{per_signature}");
        assert_eq!(full.catalog_row(), "mldsa-44-p16x-batch-32768-mult-final");
        let small = CommitmentManifest::for_batch(256).unwrap();
        assert_eq!(small.d_main(), 18);
        assert_eq!(small.segments(), 1);
        let logical: Vec<usize> = small.groups().iter().map(|g| g.logical_vars).collect();
        let physical: Vec<usize> = small.groups().iter().map(|g| g.physical_vars).collect();
        assert_eq!(logical, [18, 18, 13, 13, 12, 15, 16]);
        assert_eq!(physical, [18, 18, 13, 13, 13, 15, 16]);
        assert_eq!(small.group(Group::Level3).pad_vars(), 1);
        assert_eq!(small.catalog_row(), "mldsa-44-p16f-batch-256-mult-final");
        // G_3 has G_2's physical layout at every size (one trailing pad variable).
        for d_sig in 3..=15 {
            let m = CommitmentManifest::for_batch(1 << d_sig).unwrap();
            let (l2, l3) = (m.group(Group::Level2), m.group(Group::Level3));
            assert_eq!((l3.physical_vars, l3.polynomial_count()), (l2.physical_vars, l2.polynomial_count()));
            assert_eq!(l3.pad_vars(), 1);
        }
        let physical = |m: &CommitmentManifest| -> usize {
            m.groups()
                .iter()
                .map(|g| g.polynomial_count() << g.physical_vars)
                .sum()
        };
        let k = CommitmentManifest::for_relation(1024, Relation::HExact).unwrap();
        assert_eq!(physical(&k) / 1024, 14_560);
        assert_eq!(physical(&CommitmentManifest::for_batch(1024).unwrap()) / 1024, 17_648);
    }

    #[test]
    fn lookup_shape_follows_the_domains() {
        let manifest = CommitmentManifest::for_batch(1 << 15).unwrap();
        let shape = manifest.lookup_shape();
        shape.validate().unwrap();
        let depths: Vec<usize> = shape.stacks.iter().map(|s| s.depth()).collect();
        assert_eq!(depths, [26, 22, 18, 14, 23]);
        assert_eq!(shape.lookups(0), 79_970_304);
        assert_eq!(shape.max_depth(), 26);
        for (stack, columns) in (0..5).map(|s| (s, stack_columns(s, Relation::HFips))) {
            assert_eq!(shape.stacks[stack].columns, columns.len());
            for column in columns {
                assert!(column.is_lookup());
                assert_eq!(column.domain(), STACK_DOMAINS[stack]);
            }
        }
    }

    #[test]
    fn manifest_matches_the_work_plan() {
        for records in [1, 256, 1000, 5000] {
            let manifest = CommitmentManifest::for_batch(records).unwrap();
            let plan = WorkPlan::for_records(records, Relation::HFips).unwrap();
            assert!(manifest.matches_plan(&plan), "N = {records}");
        }
        for &d_sig in Relation::HExact.catalog_d_sigs() {
            let records = (1 << d_sig) - 3;
            let manifest = CommitmentManifest::for_relation(records, Relation::HExact).unwrap();
            let plan = WorkPlan::for_records(records, Relation::HExact).unwrap();
            assert!(manifest.matches_plan(&plan), "H-exact N = {records}");
            let fips = WorkPlan::for_records(records, Relation::HFips).unwrap();
            assert!(!manifest.matches_plan(&fips), "H-exact against the H-FIPS plan");
        }
        let other = WorkPlan::for_records(512, Relation::HFips).unwrap();
        assert!(!CommitmentManifest::for_batch(256)
            .unwrap()
            .matches_plan(&other));
    }

    #[test]
    fn binding_bytes_are_pinned_and_separate_batches() {
        let manifest = CommitmentManifest::for_batch(200).unwrap();
        let bytes = manifest.binding_bytes();
        let mut expected = vec![4, 0, 0x44, 2];
        expected.extend_from_slice(&200_u64.to_le_bytes());
        expected.extend_from_slice(&256_u64.to_le_bytes());
        expected.extend_from_slice(&[2, 7]);
        assert_eq!(&bytes[..expected.len()], expected.as_slice());
        // G_main16: role 0, round 1, family 1, 18 logical and physical vars, 4 polys: bz_lo and
        // l_lo looked up, v_lo and v_hi unchecked.
        let main16 = &bytes[expected.len()..];
        assert_eq!(&main16[..6], &[0, 1, 1, 18, 18, 4]);
        assert_eq!(&main16[6..10], &[0x90, 0x90, 0x10 | 16, 0x10 | 16]);
        // G_main4: role 1, round 1, family 0, then bz_hi, l_hi, six bits, a_z, s (bits) and
        // delta, e (ternary).
        let main4 = &main16[10..];
        assert_eq!(&main4[..18], &[1, 1, 0, 18, 18, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x23, 0x23]);
        assert_eq!(manifest.relation(), Relation::HFips);
        assert_eq!(Relation::from_id(RELATION_H_EXACT), Some(Relation::HExact));
        assert_eq!(Relation::from_id(RELATION_H_FIPS), Some(Relation::HFips));
        assert_eq!(Relation::from_id(0), None);
        assert_eq!(Relation::from_id(3), None);
        assert_eq!(
            CommitmentManifest::for_relation(200, Relation::HFips).unwrap(),
            manifest
        );
        let row = manifest.catalog_row().len();
        assert_eq!(
            bytes.len(),
            expected.len() + 7 * 6 + 59 + 2 + row + 3 * 17 + 3 + 1 + 5 * 7
        );
        // Live count, padded count and the catalog row all enter the binding.
        let full = CommitmentManifest::for_batch(256).unwrap();
        assert_ne!(full.binding_bytes(), bytes);
        assert_ne!(
            CommitmentManifest::for_batch(512).unwrap().binding_bytes(),
            full.binding_bytes()
        );
        assert_eq!(
            CommitmentManifest::for_batch(200).unwrap().binding_bytes(),
            bytes
        );
        // H-exact: relation byte 1, G_main4 at 9 polynomials
        // with codes [0, 6, 0, 0, 0, 0, 0, 0, 0], G_main16 unchanged, the p16x row.
        let exact = CommitmentManifest::for_relation(200, Relation::HExact).unwrap();
        let xbytes = exact.binding_bytes();
        let mut xexpected = expected.clone();
        xexpected[3] = 1;
        assert_eq!(&xbytes[..xexpected.len()], xexpected.as_slice());
        let xmain16 = &xbytes[xexpected.len()..];
        assert_eq!(&xmain16[..10], &main16[..10]);
        assert_eq!(&xmain16[10..25], &[1, 1, 0, 18, 18, 9, 0, 6, 0, 0, 0, 0, 0, 0, 0]);
        let xrow = exact.catalog_row().len();
        assert_eq!(xrow, row);
        assert_eq!(
            xbytes.len(),
            expected.len() + 7 * 6 + 54 + 2 + xrow + 3 * 17 + 3 + 1 + 5 * 7
        );
        assert_ne!(xbytes, bytes);
    }

    /// ML-DSA-65 and ML-DSA-87: one polynomial per vector entry, the group order of
    /// `group_columns_for`, the five projection digits and the split main lookup stack.
    #[test]
    fn lane_sets_follow_the_lane_layout() {
        // (set, polynomial counts, stage-3 claims, per-signature cells at 2^15, stacks,
        // stack depths at 2^15, lookups L at 2^15).
        let cases = [
            (Set::S65, [23, 46, 4, 8, 8, 5, 7], 87, 18_458, vec![8, 2, 1, 4, 4, 1, 2], vec![26, 24, 23, 22, 17, 14, 23], 105_005_056),
            (Set::S87, [31, 62, 4, 8, 8, 5, 7], 111, 25_242, vec![8, 4, 2, 1, 4, 4, 1, 2], vec![26, 25, 24, 23, 22, 17, 14, 24], 146_948_096),
        ];
        for (set, counts, claims, cells, stacks, depths, lookups) in cases {
            let shape = SetShape::of(set);
            assert_eq!((shape.d_vec, shape.lanes_k, shape.lanes_l, shape.w1_bits), (0, shape.k, shape.l, 4));
            assert_eq!(shape.lane_cells(), 256);
            assert_eq!(shape.row_lane(5), (5, 0));
            let (k, l) = (shape.k, shape.l);
            let manifest = CommitmentManifest::for_set(set, 256, Relation::HExact).unwrap();
            assert_eq!(manifest.set(), set);
            let got: Vec<usize> = manifest.groups().iter().map(GroupDescriptor::polynomial_count).collect();
            assert_eq!(got, counts, "{set}");
            assert_eq!(manifest.total_polynomials(), counts.iter().sum::<usize>());
            assert_eq!(manifest.stage3_polynomials(), claims);
            assert_eq!(manifest.d_main(), 8 + 8);
            assert_eq!(manifest.lane_cells(), 256);
            // Every group agrees with the catalog layout and every column with its group.
            for group in manifest.groups() {
                assert_eq!(group.logical_vars, group.role.logical_vars(set, 8));
                assert_eq!(group.polynomial_count(), group.role.num_polynomials(set, Relation::HExact));
                for (position, &column) in group.columns.iter().enumerate() {
                    assert_eq!(manifest.locate(column), (group.role, position));
                    assert_eq!(column.group(), group.role);
                    let alphabet = column.alphabet();
                    assert!(group.family.min_value() <= alphabet.min());
                    assert!(alphabet.max() <= group.family.max_value());
                    if column.is_lookup() {
                        assert_eq!(group.family, PcsFamily::Width16);
                    }
                }
            }
            // G_main16: Z_lo lanes, U_lo lanes, V_lo lanes, V_hi lanes.
            let main16 = &manifest.group(Group::Main16).columns;
            assert_eq!(main16[0], Column::lane_digit(DigitValue::Zdx(set), 0, 0));
            assert_eq!(main16[l], Column::lane_digit(DigitValue::Ud, 0, 0));
            assert_eq!(main16[l + k - 1].name(), format!("u_lo[{}]", k - 1));
            assert_eq!(main16[l + k], Column::V_LO);
            assert_eq!(main16[l + 2 * k], Column::V_HI);
            assert_eq!(main16[l + 3 * k - 1], Column::V_HI.in_lane(k as u8 - 1));
            // G_main4: Z_mid, a_z, U_mid, a_u lanes, then the bits bit-major.
            let main4: Vec<String> = manifest.group(Group::Main).columns.iter().map(|c| c.name()).collect();
            assert_eq!((main4[0].as_str(), main4[l].as_str(), main4[2 * l].as_str()), ("z_mid", "a_z", "u_mid"));
            assert_eq!((main4[2 * l + k].as_str(), main4[2 * l + 2 * k].as_str()), ("a_u", "b_0"));
            assert_eq!(manifest.locate(Column::W1Bit(3, 2)).1, 2 * l + 2 * k + 3 * k + 2);
            assert_eq!(main4.len(), 2 * l + 2 * k + 4 * k);
            // Alphabets and codes: base-8 middle digits, Boolean slacks, a base-4 K_4.
            let z_mid = Column::lane_digit(DigitValue::Zdx(set), 1, 2);
            let a_u = Column::lane_digit(DigitValue::Ud, 2, 1);
            let k4 = Column::lane_digit(DigitValue::K5, 4, 0);
            assert_eq!((z_mid.alphabet(), z_mid.alphabet_code()), (Alphabet::Balanced(8), 8));
            assert_eq!((a_u.alphabet(), a_u.alphabet_code()), (Alphabet::Bool, 0));
            assert_eq!((k4.alphabet(), k4.alphabet_code()), (Alphabet::Balanced(4), 4));
            assert!(Column::lane_digit(DigitValue::Zdx(set), 0, 4).is_lookup());
            assert!(z_mid.is_leaf_digit() && k4.is_leaf_digit() && a_u.is_bool_digit());
            assert_eq!((z_mid.lane(), z_mid.in_lane(0).lane()), (2, 0));
            let leaves: Vec<String> = manifest.leaf_digits().iter().map(|c| c.name()).collect();
            assert_eq!(leaves.len(), l + k + 3);
            assert_eq!(&leaves[l + k..], ["k_2", "k_3", "k_4"]);
            assert_eq!(manifest.bool_digits().len(), l + k);
            assert_eq!(manifest.group(Group::Proj).columns, value_columns(DigitValue::K5));
            assert_eq!((manifest.z_value(), manifest.u_slot_value(), manifest.k_value()), (DigitValue::Zdx(set), DigitValue::Ud, DigitValue::K5));
            assert_eq!(manifest.proj_bytes(), 128 * 256 * k / 8);
            assert_eq!(manifest.proj_bound(), 1 << 39);
            // G_3: kappa^(3) and seven zero polynomials.
            let level3 = &manifest.group(Group::Level3);
            assert_eq!((level3.columns[0], level3.columns[7], level3.claimed_count()), (Column::carry(3, 0), Column::Zero(7), 1));
            // Committed cells per signature at 2^15: 256 per main polynomial.
            let full = CommitmentManifest::for_set(set, 1 << 15, Relation::HExact).unwrap();
            let main_cells = (counts[0] + counts[1]) * 256;
            assert_eq!(main_cells, if set == Set::S65 { 17_664 } else { 23_808 });
            assert_eq!(full.total_logical_cells(), cells << 15, "{set}");
            assert_eq!(full.catalog_row(), format!("mldsa-{}-p16x-batch-32768-mult-final", set.name()));
            // Lookup stacks: the main stack split 8, 2, 1 (8, 4, 2, 1); four carry lanes in four
            // slots, so no stack pads.
            assert_eq!(main_stack_sizes(set), stacks[..stacks.len() - 4]);
            let shape = full.lookup_shape();
            shape.validate().unwrap();
            let columns: Vec<usize> = shape.stacks.iter().map(|s| s.columns).collect();
            assert_eq!(columns, stacks);
            assert!(shape.stacks.iter().all(|s| s.slots() == s.columns));
            let got: Vec<usize> = shape.stacks.iter().map(|s| s.depth()).collect();
            assert_eq!(got, depths);
            assert_eq!(shape.lookups(0), lookups);
            let mut seen = Vec::new();
            for (stack, &domain) in full.stack_domains().iter().enumerate() {
                for column in full.stack_columns(stack) {
                    assert!(column.is_lookup(), "{}", column.name());
                    assert_eq!(column.domain(), domain);
                    seen.push(column);
                }
            }
            let mut looked_up: Vec<Column> =
                full.groups().iter().flat_map(|g| g.columns.clone()).filter(|c| c.is_lookup()).collect();
            looked_up.sort_by_key(|c| full.locate(*c));
            seen.sort_by_key(|c| full.locate(*c));
            assert_eq!(seen, looked_up, "every looked-up column in exactly one stack");
            assert_eq!(full.carry_stack(1), full.main_stacks());
            assert_eq!(full.stack_columns(full.proj_stack()), value_columns(DigitValue::K5)[..2]);
            // The binding bytes carry the set, the lane layout and the four-lane bridge shape.
            let manifest = CommitmentManifest::for_set(set, 200, Relation::HExact).unwrap();
            let bytes = manifest.binding_bytes();
            assert_eq!(&bytes[..4], &[4, 0, set.id(), RELATION_H_EXACT]);
            let main16 = &bytes[4 + 16 + 2..];
            assert_eq!(&main16[..6], &[0, 1, 1, 16, 16, counts[0] as u8]);
            assert!(main16[6..6 + l + k].iter().all(|&c| c == LOOKUP_ALPHABET_CODE));
            assert!(main16[6 + l + k..6 + counts[0]].iter().all(|&c| c == UNCHECKED_ALPHABET_CODE));
            let row = manifest.catalog_row().len();
            let polys: usize = counts.iter().sum();
            assert_eq!(bytes.len(), 4 + 16 + 2 + 7 * 6 + polys + 2 + row + 3 * 17 + 3 + 1 + stacks.len() * 7);
            let bridge = &bytes[4 + 16 + 2 + 7 * 6 + polys + 2 + row..][..3 * 17];
            assert_eq!(&bridge[34..42], &32_768_u64.to_le_bytes());
            assert_ne!(bytes, CommitmentManifest::for_batch(200).unwrap().binding_bytes());
            assert!(manifest.matches_plan(&WorkPlan::for_set(set, 200, Relation::HExact).unwrap()));
            assert!(!manifest.matches_plan(&WorkPlan::for_records(200, Relation::HExact).unwrap()));
            // H-FIPS: the magnitude L and three columns per row lane, the hint slack in G_2 and
            // in a fifth slot of the level-2 lookup stack.
            let fips = CommitmentManifest::for_set(set, 200, Relation::HFips).unwrap();
            let (k, l) = (fips.shape().lanes_k, fips.shape().lanes_l);
            assert_eq!(fips.u_slot_value(), DigitValue::Ld);
            assert_eq!(fips.group(Group::Main).polynomial_count(), 2 * l + 9 * k);
            assert_eq!(fips.group(Group::Level2).polynomial_count(), 9);
            assert_eq!(fips.stack_columns(fips.carry_stack(2)).len(), 5);
            assert_eq!(fips.bool_digits().len(), l + 2 * k);
            assert_eq!(fips.leaf_digits().len(), l + 3);
            assert_eq!(fips.catalog_row(), format!("mldsa-{}-p16f-batch-256-mult-final", set.name()));
            // H-exact's own catalog row name and plan.
            assert_eq!(manifest.catalog_row(), format!("mldsa-{}-p16x-batch-256-mult-final", set.name()));
            assert!(!manifest.matches_plan(&WorkPlan::for_set(set, 200, Relation::HFips).unwrap()));
            let fips_bytes = fips.binding_bytes();
            assert_eq!(&fips_bytes[..4], &[4, 0, set.id(), RELATION_H_FIPS]);
            assert_ne!(fips_bytes, bytes);
        }
        // ML-DSA-44 keeps one lane per column and its own stacks.
        let shape = SetShape::of(Set::S44);
        assert_eq!((shape.d_vec, shape.lanes_k, shape.lanes_l, shape.w1_bits, shape.lane_cells()), (2, 1, 1, 6, 1024));
        assert_eq!(shape.row_lane(3), (0, 768));
        assert_eq!(main_stack_sizes(Set::S44), [2]);
        assert_eq!(stack_domains(Set::S44), STACK_DOMAINS);
        assert_eq!(CommitmentManifest::for_set(Set::S44, 300, Relation::HExact).unwrap(), CommitmentManifest::for_relation(300, Relation::HExact).unwrap());
    }

    /// The seven balanced base-16 multiplicity digits hold the largest honest count of one
    /// `T_16` entry for every set and relation. The worst batch has `N = 2^14 + 1` records
    /// (`N* = 2^15`, so almost half the records are padding). Counted as if they all hit one
    /// entry: the public padding slots of every stack, every cell of the bridge-domain and
    /// projection stacks (carries and `K_0, K_1` cluster near zero), and the main-domain cells
    /// of the padding records. The live main-domain cells hold the low 16-bit digits of `z`
    /// and `u`, which honest signatures spread over `2^16` values, so they are left out.
    #[test]
    fn multiplicity_digits_hold_the_largest_honest_count() {
        use crate::lookup::MULTIPLICITY_CAPACITY;
        let records = (1 << 14) + 1;
        for set in [Set::S44, Set::S65, Set::S87] {
            for relation in Relation::ALL {
                if !relation.is_available(set) {
                    continue;
                }
                let manifest = CommitmentManifest::for_set(set, records, relation).unwrap();
                let (padded, live) = (manifest.padded_records(), manifest.records());
                let shape = manifest.lookup_shape();
                let mut bound = 0_usize;
                for (stack, domain) in shape.stacks.iter().zip(manifest.stack_domains()) {
                    let cells = 1_usize << stack.vars;
                    bound += (stack.slots() - stack.columns) * cells;
                    bound += if domain == Group::Main {
                        stack.columns * cells / padded * (padded - live)
                    } else {
                        stack.columns * cells
                    };
                }
                assert!(
                    i64::try_from(bound).unwrap() < MULTIPLICITY_CAPACITY,
                    "ML-DSA-{set} {relation}: {bound} cells may share one entry"
                );
            }
        }
    }

    #[test]
    fn lane_set_forms_cover_their_ranges() {
        // z of ML-DSA-65: exactly [-B_z, B_z] with B_z = 524,091, slack weight 523,895.
        let z = DigitValue::Zdx(Set::S65).layout();
        let b_z = 524_091;
        assert_eq!(z.range(), (-b_z, b_z));
        assert_eq!((z.weights(), z.constant()), (vec![1, 1 << 16, 523_895], (1 << 15) + (4 << 16) - b_z));
        // u: exactly [-523,777, 523,777], digits of zero (32,257, 3, 0).
        let u = DigitValue::Ud.layout();
        assert_eq!(u.range(), (-523_777, 523_777));
        assert_eq!(u.decompose(0), vec![32_257, 3, 0]);
        assert_eq!((u.weights(), u.constant()), (vec![1, 1 << 16, 523_267], -228_865));
        for (form, inside, outside) in [
            (z, [-b_z, -1, 0, b_z], [-b_z - 1, b_z + 1]),
            (u, [-523_777, -1, 0, 523_777], [-523_778, 523_778]),
        ] {
            for x in inside {
                let digits = form.decompose(x);
                assert!(form.in_alphabets(&digits), "{x}");
                assert_eq!(form.compose(&digits), x);
            }
            for x in outside {
                assert!(!form.in_alphabets(&form.decompose(x)), "{x}");
            }
        }
        // k: [-2^39, 2^39), digits of zero (-2^15, -2^15, -4, -4, 0).
        assert_eq!(K39_DIGITS.range(), (-PROJ_BOUND_39, PROJ_BOUND_39 - 1));
        assert_eq!(K39_DIGITS.constant(), (1 << 37) + (1 << 34) + (1 << 31) + (1 << 15));
        assert_eq!(K39_DIGITS.decompose(0), vec![-(1 << 15), -(1 << 15), -4, -4, 0]);
        assert_eq!(K39_DIGITS.weights(), vec![1, 1 << 16, 1 << 32, 1 << 35, 1 << 38]);
        for x in [-PROJ_BOUND_39, -1, 0, PROJ_BOUND_39 - 1] {
            assert!(K39_DIGITS.in_alphabets(&K39_DIGITS.decompose(x)));
        }
        for x in [-PROJ_BOUND_39 - 1, PROJ_BOUND_39] {
            assert!(!K39_DIGITS.in_alphabets(&K39_DIGITS.decompose(x)));
        }
    }

    /// The ML-DSA-44 binding bytes of every relation at a spread of batch sizes, pinned by
    /// their SHA3-256 digest (the instance record must not move when other sets are added).
    #[test]
    fn ml_dsa_44_binding_bytes_are_pinned() {
        use sha3::{Digest, Sha3_256};
        let mut hasher = Sha3_256::new();
        for relation in Relation::ALL {
            for records in [1, 8, 9, 200, 256, 1000, 1024, 5000, 8192, 8193, 20_000, 1 << 15] {
                let bytes = CommitmentManifest::for_relation(records, relation).unwrap().binding_bytes();
                hasher.update((bytes.len() as u64).to_le_bytes());
                hasher.update(&bytes);
            }
        }
        let digest: String = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(digest, "f9a87b92bcdab520c415d38983fff34b393bd3d1f72a6a76f10b7a3e0290e0ef");
    }

    #[test]
    fn single_record_uses_the_minimum_shape_and_bad_sizes_reject() {
        let single = CommitmentManifest::for_batch(1).unwrap();
        let minimum = CommitmentManifest::for_batch(8).unwrap();
        assert_eq!(single.padded_records(), 8);
        assert_eq!(single.groups(), minimum.groups());
        assert_ne!(single.binding_bytes(), minimum.binding_bytes());
        assert!(CommitmentManifest::for_batch(0).is_err());
        assert!(CommitmentManifest::for_batch((1 << 15) + 1).is_err());
    }

    #[test]
    fn fp_manifest_groups_columns_and_binding_bytes() {
        let manifest = CommitmentManifest::for_relation_fp(1024, Relation::HExact).unwrap();
        assert_eq!(manifest.hash_field(), HashField::Fp);
        assert_eq!(manifest.groups().len(), GROUP_COUNT_FP);
        assert_eq!(manifest.catalog_row(), "mldsa-44-p16x-fp-batch-1024-mult-final");
        let counts: Vec<usize> = manifest
            .groups()
            .iter()
            .map(GroupDescriptor::polynomial_count)
            .collect();
        assert_eq!(counts, [4, 9, 55, 55, 55, 55, 55, 55, 55, 4, 7]);
        assert_eq!(manifest.total_polynomials(), 409);
        assert_eq!(manifest.stage3_polynomials(), 402);

        // Keccak columns: alphabet Bool, code 0, locate and group round-trip.
        for y in 0..5_u8 {
            let group = manifest.group(Group::KeccakA(y));
            assert_eq!(group.round, 1);
            assert_eq!(group.family, PcsFamily::Width4);
            assert_eq!(group.logical_vars, 20);
            assert_eq!(group.physical_vars, 20);
            assert_eq!(group.columns.len(), 55);
            for x in 0..5_u8 {
                for q in 0..11_u8 {
                    let col = Column::KeccakA { x, y, q };
                    assert_eq!(col.alphabet(), Alphabet::Bool);
                    assert_eq!(col.alphabet_code(), 0);
                    assert_eq!(col.group(), Group::KeccakA(y));
                    assert_eq!(
                        manifest.locate(col),
                        (Group::KeccakA(y), usize::from(x) * 11 + usize::from(q))
                    );
                }
            }
        }
        for x in 0..5_u8 {
            for q in 0..11_u8 {
                let c = Column::KeccakC { x, q };
                let cp = Column::KeccakCPrime { x, q };
                assert_eq!((c.alphabet_code(), cp.alphabet_code()), (0, 0));
                assert_eq!(
                    manifest.locate(c),
                    (Group::KeccakC, usize::from(x) * 11 + usize::from(q))
                );
                assert_eq!(
                    manifest.locate(cp),
                    (Group::KeccakCPrime, usize::from(x) * 11 + usize::from(q))
                );
            }
        }

        // Lookup shape under Fp for ML-DSA-44: two stacks ([Z_lo | U_lo], [K_0 | K_1]).
        let lookup = manifest.lookup_shape();
        assert_eq!(lookup.stacks.len(), 2);
        assert_eq!(manifest.stack_columns(0), [Column::ZX_LO, Column::U_LO]);
        assert_eq!(manifest.stack_columns(1), [Column::K_0, Column::K_1]);
        assert_eq!((lookup.stacks[0].columns, lookup.stacks[0].vars), (2, 20));
        assert_eq!((lookup.stacks[1].columns, lookup.stacks[1].vars), (2, 17));

        // Binding bytes under Fp: group count 11, no bridge parameters, Fp row id, 2 stacks.
        let bytes = manifest.binding_bytes();
        assert_eq!(bytes[21], GROUP_COUNT_FP as u8);
        let row = manifest.catalog_row().len();
        let expected_len = 22 + GROUP_COUNT_FP * 6 + 409 + 2 + row + 3 + 1 + 2 * 7;
        assert_eq!(bytes.len(), expected_len);
        // Lookup trailer at the end: challenges 2, 1 table of width 16, 2 stacks.
        let trailer = &bytes[bytes.len() - (4 + 2 * 7)..];
        assert_eq!(&trailer[..4], &[2, 1, 16, 2]);
        assert_eq!(&trailer[4..11], &[2, 20, 0, 0x00, 0x80, 0x00, 0x00]);
        assert_eq!(&trailer[11..18], &[2, 17, 0, 0x00, 0x80, 0x00, 0x00]);

        // ML-DSA-87 under Fp: 70 polynomials per Keccak group (Q = 14).
        let m87 = CommitmentManifest::for_set_fp(Set::S87, 1024, Relation::HExact).unwrap();
        assert_eq!(m87.group(Group::KeccakA(0)).polynomial_count(), 70);
        assert_eq!(m87.group(Group::KeccakA(0)).logical_vars, 20);
        assert_eq!(m87.catalog_row(), "mldsa-87-p16x-fp-batch-1024-mult-final");
    }
}
