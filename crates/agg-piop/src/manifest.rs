//! Exact commitment manifest of one batch (event 0; `docs/lookup.md`).
//!
//! Mirrors `aerie-falcon/src/protocol/manifest.rs`: the manifest fixes the canonical wire
//! order, the commitment round, the preset family, the logical and physical domain of every
//! Akita group, and the role and alphabet of each committed polynomial, and serializes the
//! public profile bound in event 0 before any commitment. The six groups are `G_main16`
//! (`Z_lo, U_lo, V_lo, V_hi`, width 16) and `G_main4` (`Z_hi, U_hi, b_0..b_5`, width 4) in
//! round 1, before the bridge seed, and `G_1` (`kappa^(1)_t`), `G_2` (`y^(2)_t`,
//! `kappa^(2)_t`), `G_3` (`kappa^(3)`), all width 16, and `G_m` (the seven digits of the
//! multiplicity column of `T_16`, width 4) in round 2; `G_m` is Akita's final group, so PCS
//! execution follows wire order. The alphabets listed here are the honest producer's; the
//! ranges are proven inside the PIOP (the lookup into `T_16`, the base-4 and base-6 leaves),
//! never by Akita.
//!
//! Relation H-u (the default, `docs/h-exact.md`): `z` relaxed to `[-2^17, 2^17)`, `u`
//! exact, `u = B_ulo + 63,079 H_u - 190,465` with `B_ulo` the looked-up `U_lo + 2^15` and
//! `H_u = U_hi + 3` a six-valued digit, so `|u| <= B_u = 190,465`.
//!
//! Relation H-exact (relation byte 1, `docs/h-exact.md`): `u` as in H-u and `z` exact,
//! `z = B_zlo + 2^16 B_zhi + 130,915 a_z - 130,993` with `B_zlo = S_zlo + 2^15` looked up (the
//! slot of `Z_lo`) and `B_zhi`, `a_z` Boolean (`B_zhi` in the slot of `Z_hi`, `a_z` appended
//! to `G_main4` after the bits), so `|z| <= B_z = 130,993`. The value `z` of H-exact is
//! [`DigitValue::Zx`] with the columns `bz_lo`, `bz_hi`, `a_z`; the Booleanity of `bz_hi`
//! and `a_z` is proven by two zero-checks at `r^(2)` in stage 3.
//!
//! Relation H-FIPS (relation byte 2, `docs/h-exact.md`): `z` as in H-exact, and in
//! place of `u`'s digits the per-coefficient witness of `agg_witness::hfips`: the magnitude
//! `L = L_lo + 29,696 L_hi` ([`DigitValue::L`]: `l_lo` looked up in the slot of `U_lo`, the
//! bit `l_hi` in the slot of `U_hi`), the sign bit `s = (1 + sigma)/2`, `delta` and `e`
//! ([`FipsColumn`], appended to `G_main4` after `a_z`), so
//! `u = gamma_2 s + L + 1 - gamma_2 - 2 gamma_2 delta - e`; and the per-signature hint slack `u_h`
//! ([`DigitValue::HintSlack`]), a thirteenth `G_2` polynomial on `D_2` (whose last `d_sig`
//! variables are the signature index) in the free slot 6 of lookup stack 2.

use agg_pcs_config::catalogs::{BatchLayout, Family, Group, MULTIPLICITY_DIGITS, TABLE_VARS};
pub use agg_pcs_config::catalogs::{Relation, RELATION_HU, RELATION_H_EXACT, RELATION_H_FIPS};
use agg_pcs_config::work_plan::{padded_size, PlanError, WorkPlan};
use agg_witness::digits::{Alphabet, BitSlackLayout, DigitLayout, SlackLayout, StepLayout, ValueForm, L_SLACK, Z_BIT_SLACK};
use agg_witness::params::{B_U, STEP_U, U_TOP_BASE, W1_BITS};

use crate::lookup::{LookupShape, StackShape, Table};

/// Role of a committed group; the six groups in wire order.
pub type GroupRole = Group;

/// PCS preset family of a group (`agg_pcs_config::catalogs::Family`).
pub type PcsFamily = Family;

/// Wire version of this protocol (the Keccak side's tied parts of proof version 5 with the
/// virtual parity chain and its work sites in the plan).
pub const WIRE_VERSION: u16 = 4;
/// Parameter-set byte of ML-DSA-44 (event 0).
pub const PARAMETER_SET: u8 = 0x44;
/// Identifier of the digit layout: 1 is the lookup layout (ranges by the lookup into
/// `T_16`).
pub const DIGIT_LAYOUT_ID: u8 = 1;
/// Number of committed groups.
pub const GROUP_COUNT: usize = agg_pcs_config::catalogs::GROUP_COUNT;
/// Independent lookup challenges `X_1, X_2` (`docs/lookup.md`).
pub const LOOKUP_CHALLENGES: usize = 2;
/// The balanced storage shift of every looked-up column: the looked-up value is
/// `f = stored + 2^15` (`docs/lookup.md`).
pub const LOOKUP_SHIFT: i32 = 1 << 15;

/// `z + 2^17 = z_lo + 2^16 z_hi` (bases `2^16, 4`, signed): `z = Z_lo + 2^16 Z_hi + 2^15`.
pub const Z16_DIGITS: DigitLayout = DigitLayout {
    name: "z",
    bases: &[1 << 16, 4],
    signed: true,
};
/// Relation H-exact: `z = S_zlo + 2^16 B_zhi + 130,915 a_z - 98,225` over exactly `[-B_z, B_z]`
/// (`docs/h-exact.md`): `S_zlo = B_zlo - 2^15` looked up, `B_zhi, a_z` bits.
pub const ZX16_DIGITS: BitSlackLayout = Z_BIT_SLACK;
/// `B_ulo = U_lo + 2^15` in `[0, 2^16)` (base `2^16`, unsigned).
pub const ULO16_DIGITS: DigitLayout = DigitLayout {
    name: "u",
    bases: &[1 << 16],
    signed: false,
};
/// Relation H-u: `u = B_ulo + 63,079 H_u - 190,465` with `B_ulo = U_lo + 2^15` looked up and
/// `H_u = U_hi + 3 in [0, 6)`, so `u = U_lo + 63,079 U_hi + 31,540` over exactly `[-B_u, B_u]`
/// (`5 * 63,079 + 2^16 = 2 B_u + 1`).
pub const U16_DIGITS: StepLayout = StepLayout {
    low: ULO16_DIGITS,
    step: STEP_U,
    top_base: U_TOP_BASE,
    offset: B_U,
};
/// `v + 2^31 = v_lo + 2^16 v_hi` (bases `2^16, 2^16`, signed).
pub const V16_DIGITS: DigitLayout = DigitLayout {
    name: "v",
    bases: &[1 << 16, 1 << 16],
    signed: true,
};
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
/// Alphabet code of a looked-up column in the event-0 bytes (`log2 |T_16|` with the high
/// bit set).
pub const LOOKUP_ALPHABET_CODE: u8 = 0x80 | 16;
/// Alphabet code of a multiplicity digit (base 16, not range-checked).
pub const MULTIPLICITY_ALPHABET_CODE: u8 = 0x40 | 16;
/// Alphabet code of a ternary column (`{-1, 0, 1}`: H-FIPS's `delta` and `e`).
pub const TERNARY_ALPHABET_CODE: u8 = 0x20 | 3;

/// Canonical transcript byte of a group role (its wire position).
#[must_use]
pub const fn role_id(role: GroupRole) -> u8 {
    role.index() as u8
}

/// A value committed through its balanced digits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DigitValue {
    /// The response `z` of relation H-u (main domain, indexed by `(j, s, i)`).
    Z,
    /// The response `z` of relation H-exact: `bz_lo` (looked up), `bz_hi`, `a_z` (bits).
    Zx,
    /// The UseHint residual `u` (main domain; relations H-u and H-exact).
    U,
    /// Relation H-FIPS's magnitude `L = L_lo + 29,696 L_hi` (main domain): `l_lo` (looked
    /// up), `l_hi` (bit).
    L,
    /// The ring quotient `v` (main domain).
    V,
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
            Self::Z => ValueForm::Radix(Z16_DIGITS),
            Self::Zx => ValueForm::BitSlack(ZX16_DIGITS),
            Self::U => ValueForm::Step(U16_DIGITS),
            Self::L => ValueForm::Slack(L16_SLACK),
            Self::V => ValueForm::Radix(V16_DIGITS),
            Self::Kappa1(_) | Self::Kappa2(_) | Self::Kappa3 | Self::HintSlack => {
                ValueForm::Radix(CARRY16_DIGITS)
            }
        }
    }

    /// The group holding digit `digit`.
    #[must_use]
    pub const fn group(self, digit: u8) -> GroupRole {
        match self {
            Self::Z | Self::Zx | Self::U | Self::L if digit == 0 => Group::Main16,
            Self::Z | Self::Zx | Self::U | Self::L => Group::Main,
            Self::V => Group::Main16,
            Self::Kappa1(_) => Group::Level1,
            Self::Kappa2(_) | Self::HintSlack => Group::Level2,
            Self::Kappa3 => Group::Level3,
        }
    }
}

/// The three per-coefficient columns of relation H-FIPS that are not digits of a value
/// (`docs/h-exact.md`): in `G_main4` after `a_z`, in this order.
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
    },
    /// The w1Encode bit `b_t` (main domain).
    W1Bit(u8),
    /// The level-2 parity `y^(2)_t` (domain `D_2`).
    Parity2(u8),
    /// Digit `k` of the multiplicity column `m` of `T_16` (`2^16` cells).
    Multiplicity(u8),
    /// A per-coefficient column of relation H-FIPS (main domain, `G_main4`).
    Fips(FipsColumn),
}

impl Column {
    /// `Z_lo`.
    pub const Z_LO: Self = Self::Digit {
        value: DigitValue::Z,
        digit: 0,
    };
    /// `Z_hi`.
    pub const Z_HI: Self = Self::Digit {
        value: DigitValue::Z,
        digit: 1,
    };
    /// `bz_lo` (H-exact `S_zlo`, the slot of `Z_lo`).
    pub const ZX_LO: Self = Self::Digit {
        value: DigitValue::Zx,
        digit: 0,
    };
    /// `bz_hi` (H-exact `B_zhi`, a bit in the slot of `Z_hi`).
    pub const ZX_HI: Self = Self::Digit {
        value: DigitValue::Zx,
        digit: 1,
    };
    /// `a_z` (H-exact slack bit, after the bits of `G_main4`).
    pub const A_Z: Self = Self::Digit {
        value: DigitValue::Zx,
        digit: 2,
    };
    /// `U_lo`.
    pub const U_LO: Self = Self::Digit {
        value: DigitValue::U,
        digit: 0,
    };
    /// `U_hi`.
    pub const U_HI: Self = Self::Digit {
        value: DigitValue::U,
        digit: 1,
    };
    /// `l_lo` (H-FIPS `L_lo - 2^15`, the slot of `U_lo`).
    pub const L_LO: Self = Self::Digit {
        value: DigitValue::L,
        digit: 0,
    };
    /// `l_hi` (H-FIPS `L_hi`, a bit in the slot of `U_hi`).
    pub const L_HI: Self = Self::Digit {
        value: DigitValue::L,
        digit: 1,
    };
    /// H-FIPS's sign bit `s`.
    pub const SIGN: Self = Self::Fips(FipsColumn::Sign);
    /// H-FIPS's `delta`.
    pub const DELTA: Self = Self::Fips(FipsColumn::Delta);
    /// H-FIPS's `e`.
    pub const WRAP: Self = Self::Fips(FipsColumn::Wrap);
    /// H-FIPS's hint slack `u_h` (stored `u_h - 2^15`, on `D_2`).
    pub const HINT_SLACK: Self = Self::Digit {
        value: DigitValue::HintSlack,
        digit: 0,
    };
    /// `V_lo`.
    pub const V_LO: Self = Self::Digit {
        value: DigitValue::V,
        digit: 0,
    };
    /// `V_hi`.
    pub const V_HI: Self = Self::Digit {
        value: DigitValue::V,
        digit: 1,
    };

    /// The carry column `kappa^(l)_t` (`l = 1, 2`) or `kappa^(3)` (`l = 3`).
    #[must_use]
    pub const fn carry(level: u8, t: u8) -> Self {
        let value = match level {
            1 => DigitValue::Kappa1(t),
            2 => DigitValue::Kappa2(t),
            _ => DigitValue::Kappa3,
        };
        Self::Digit { value, digit: 0 }
    }

    /// The honest alphabet (of the stored value).
    #[must_use]
    pub fn alphabet(self) -> Alphabet {
        match self {
            Self::Digit { value, digit } => value.layout().alphabet(usize::from(digit)),
            Self::W1Bit(_) | Self::Parity2(_) | Self::Fips(FipsColumn::Sign) => Alphabet::Bool,
            Self::Fips(FipsColumn::Delta | FipsColumn::Wrap) => Alphabet::Ternary,
            Self::Multiplicity(_) => Alphabet::Balanced(16),
        }
    }

    /// Digit base, `None` for a Boolean column (also `bz_hi`, `a_z`) or a multiplicity digit.
    #[must_use]
    pub fn base(self) -> Option<u32> {
        match self {
            Self::Digit { value, digit } => value.layout().base(usize::from(digit)),
            Self::W1Bit(_) | Self::Parity2(_) | Self::Multiplicity(_) | Self::Fips(_) => None,
        }
    }

    /// True for a column looked up in `T_16` (stored balanced, `f = stored + 2^15`).
    #[must_use]
    pub fn is_lookup(self) -> bool {
        self.base() == Some(LOOKUP_BASE)
    }

    /// True for a digit range-checked by a leaf and a link (`Z_hi` base 4, `U_hi` base 6).
    #[must_use]
    pub fn is_leaf_digit(self) -> bool {
        matches!(self.base(), Some(4 | 6))
    }

    /// True when the column is stored as `i32` cells (every looked-up column).
    #[must_use]
    pub fn is_wide(self) -> bool {
        self.is_lookup()
    }

    /// The group holding the column.
    #[must_use]
    pub const fn group(self) -> GroupRole {
        match self {
            Self::Digit { value, digit } => value.group(digit),
            Self::W1Bit(_) | Self::Fips(_) => Group::Main,
            Self::Parity2(_) => Group::Level2,
            Self::Multiplicity(_) => Group::Mult,
        }
    }

    /// The domain the column lives on (the main domain for both main groups).
    #[must_use]
    pub const fn domain(self) -> GroupRole {
        match self.group() {
            Group::Main16 | Group::Main => Group::Main,
            other => other,
        }
    }

    /// True for a Boolean digit of a value (`bz_hi`, `a_z`; H-FIPS also `l_hi`), proven by a
    /// zero-check at `r^(2)` in stage 3.
    #[must_use]
    pub fn is_bool_digit(self) -> bool {
        matches!(self, Self::Digit { .. }) && self.alphabet() == Alphabet::Bool
    }

    /// Column name (`z_lo`, `u_hi`, `bz_hi`, `a_z`, `b_0`, `kappa1_2`, `y2_0`, `m_4`, ...).
    #[must_use]
    pub fn name(self) -> String {
        let half = |digit: u8| if digit == 0 { "lo" } else { "hi" };
        match self {
            Self::Digit { value, digit } => match value {
                DigitValue::Z => format!("z_{}", half(digit)),
                DigitValue::Zx => ["bz_lo", "bz_hi", "a_z"][usize::from(digit)].to_owned(),
                DigitValue::U => format!("u_{}", half(digit)),
                DigitValue::L => format!("l_{}", half(digit)),
                DigitValue::V => format!("v_{}", half(digit)),
                DigitValue::HintSlack => "u_h".to_owned(),
                DigitValue::Kappa1(t) => format!("kappa1_{t}"),
                DigitValue::Kappa2(t) => format!("kappa2_{t}"),
                DigitValue::Kappa3 => "kappa3".to_owned(),
            },
            Self::W1Bit(t) => format!("b_{t}"),
            Self::Parity2(t) => format!("y2_{t}"),
            Self::Multiplicity(k) => format!("m_{k}"),
            Self::Fips(FipsColumn::Sign) => "s".to_owned(),
            Self::Fips(FipsColumn::Delta) => "delta".to_owned(),
            Self::Fips(FipsColumn::Wrap) => "e".to_owned(),
        }
    }

    /// Transcript code of the alphabet: the digit base of a leaf-checked digit (4, 6),
    /// [`LOOKUP_ALPHABET_CODE`] for a column looked up in `T_16`,
    /// [`MULTIPLICITY_ALPHABET_CODE`] for a multiplicity digit, [`TERNARY_ALPHABET_CODE`] for
    /// H-FIPS's `delta` and `e`, 0 for a Boolean.
    #[must_use]
    pub fn alphabet_code(self) -> u8 {
        match self {
            Self::Multiplicity(_) => MULTIPLICITY_ALPHABET_CODE,
            Self::Fips(FipsColumn::Delta | FipsColumn::Wrap) => TERNARY_ALPHABET_CODE,
            column if column.is_lookup() => LOOKUP_ALPHABET_CODE,
            column => column.base().map_or(0, |base| base as u8),
        }
    }
}

/// The value `z` of `relation` (`Z` for H-u, `Zx` for H-exact and H-FIPS).
#[must_use]
pub const fn z_value(relation: Relation) -> DigitValue {
    match relation {
        Relation::Hu => DigitValue::Z,
        Relation::HExact | Relation::HFips => DigitValue::Zx,
    }
}

/// The digit-decomposed value in the slots of `U_lo, U_hi`: `u` (H-u, H-exact) or H-FIPS's
/// magnitude `L`.
#[must_use]
pub const fn u_slot_value(relation: Relation) -> DigitValue {
    match relation {
        Relation::Hu | Relation::HExact => DigitValue::U,
        Relation::HFips => DigitValue::L,
    }
}

/// The committed columns of a value, in digit order.
#[must_use]
pub fn value_columns(value: DigitValue) -> Vec<Column> {
    (0..value.layout().len() as u8)
        .map(|digit| Column::Digit { value, digit })
        .collect()
}

/// The canonical polynomial order of a group under `relation` (`docs/h-exact.md`): H-exact
/// puts `bz_lo, bz_hi` in the slots of `Z_lo, Z_hi` and appends `a_z` to `G_main4`; H-FIPS
/// also puts `l_lo, l_hi` in the slots of `U_lo, U_hi`, appends `s, delta, e` to `G_main4`
/// and `u_h` to `G_2`.
#[must_use]
pub fn group_columns(role: GroupRole, relation: Relation) -> Vec<Column> {
    let lanes = 0..W1_BITS as u8;
    let z = value_columns(z_value(relation));
    let u = value_columns(u_slot_value(relation));
    let fips = relation == Relation::HFips;
    match role {
        Group::Main16 => vec![z[0], u[0], Column::V_LO, Column::V_HI],
        Group::Main => [z[1], u[1]]
            .into_iter()
            .chain(lanes.map(Column::W1Bit))
            .chain(z[2..].iter().copied())
            .chain(FipsColumn::ALL.into_iter().map(Column::Fips).filter(|_| fips))
            .collect(),
        Group::Level1 => lanes.map(|t| Column::carry(1, t)).collect(),
        Group::Level2 => lanes
            .clone()
            .map(Column::Parity2)
            .chain(lanes.map(|t| Column::carry(2, t)))
            .chain(std::iter::once(Column::HINT_SLACK).filter(|_| fips))
            .collect(),
        Group::Level3 => vec![Column::carry(3, 0)],
        Group::Mult => (0..MULTIPLICITY_DIGITS as u8)
            .map(Column::Multiplicity)
            .collect(),
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
    fn new(role: GroupRole, d_sig: usize, relation: Relation) -> Self {
        Self {
            role,
            round: role.round(),
            family: role.family(),
            logical_vars: role.logical_vars(d_sig),
            physical_vars: role.physical_vars(d_sig),
            columns: group_columns(role, relation),
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

    /// Position of `column` in the group, if it belongs to it.
    #[must_use]
    pub fn position(&self, column: Column) -> Option<usize> {
        self.columns.iter().position(|&c| c == column)
    }
}

/// The looked-up columns of one lookup stack (`docs/lookup.md`), in slot order
/// (H-FIPS puts `u_h` in slot 6 of stack 2).
#[must_use]
pub fn stack_columns(stack: usize, relation: Relation) -> Vec<Column> {
    let lanes = 0..W1_BITS as u8;
    match stack {
        0 => group_columns(Group::Main16, relation),
        1 => lanes.map(|t| Column::carry(1, t)).collect(),
        2 => lanes
            .map(|t| Column::carry(2, t))
            .chain(std::iter::once(Column::HINT_SLACK).filter(|_| relation == Relation::HFips))
            .collect(),
        3 => vec![Column::carry(3, 0)],
        _ => panic!("four lookup stacks"),
    }
}

/// The domain of each lookup stack: the main domain, `D_1`, `D_2`, `D_3`.
pub const STACK_DOMAINS: [GroupRole; 4] = [Group::Main, Group::Level1, Group::Level2, Group::Level3];

/// Complete public commitment manifest of one batch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitmentManifest {
    records: usize,
    padded: usize,
    d_sig: usize,
    relation: Relation,
    groups: [GroupDescriptor; GROUP_COUNT],
    catalog_row: String,
}

impl CommitmentManifest {
    /// The manifest of a batch of `records` live records (`1 <= N <= 2^15`) under the
    /// default relation H-u.
    pub fn for_batch(records: usize) -> Result<Self, PlanError> {
        Self::for_relation(records, Relation::default())
    }

    /// The manifest of a batch of `records` live records under `relation`.
    pub fn for_relation(records: usize, relation: Relation) -> Result<Self, PlanError> {
        let (padded, d_sig) = padded_size(records)?;
        let layout = BatchLayout::new(d_sig, relation).expect("d_sig within the wire range");
        Ok(Self {
            records,
            padded,
            d_sig,
            relation,
            groups: Group::ALL.map(|role| GroupDescriptor::new(role, d_sig, relation)),
            catalog_row: layout.row_name(),
        })
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

    /// `d_main = 10 + d_sig`.
    #[must_use]
    pub const fn d_main(&self) -> usize {
        self.groups[Group::Main.index()].logical_vars
    }

    /// Number of `2^13`-signature segments `S`.
    #[must_use]
    pub const fn segments(&self) -> usize {
        1 << self.d_sig.saturating_sub(13)
    }

    /// The grouped catalog row id bound in event 0.
    #[must_use]
    pub fn catalog_row(&self) -> &str {
        &self.catalog_row
    }

    /// Every group in wire order.
    #[must_use]
    pub const fn groups(&self) -> &[GroupDescriptor; GROUP_COUNT] {
        &self.groups
    }

    /// The descriptor of `role`.
    #[must_use]
    pub const fn group(&self, role: GroupRole) -> &GroupDescriptor {
        &self.groups[role.index()]
    }

    /// Logical variables of a domain (the main domain for both main groups).
    #[must_use]
    pub const fn domain_vars(&self, domain: GroupRole) -> usize {
        self.groups[domain.index()].logical_vars
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

    /// The value `z` of the relation ([`z_value`]).
    #[must_use]
    pub const fn z_value(&self) -> DigitValue {
        z_value(self.relation)
    }

    /// The value in the slots of `U_lo, U_hi` ([`u_slot_value`]).
    #[must_use]
    pub const fn u_slot_value(&self) -> DigitValue {
        u_slot_value(self.relation)
    }

    /// True under relation H-FIPS (the hint is part of the relation).
    #[must_use]
    pub fn is_fips(&self) -> bool {
        self.relation == Relation::HFips
    }

    /// The looked-up columns of lookup stack `stack` ([`stack_columns`]).
    #[must_use]
    pub fn stack_columns(&self, stack: usize) -> Vec<Column> {
        stack_columns(stack, self.relation)
    }

    /// The leaf-checked digits in the order of the leaf and link instances: `Z_hi` (base 4)
    /// and `U_hi` (base 6) under H-u, `U_hi` alone under H-exact.
    #[must_use]
    pub fn leaf_digits(&self) -> Vec<Column> {
        self.groups
            .iter()
            .flat_map(|group| group.columns.iter().copied())
            .filter(|column| column.is_leaf_digit())
            .collect()
    }

    /// The Boolean value digits proven by a zero-check at `r^(2)` (stage 3), in manifest
    /// order: none under H-u, `bz_hi` and `a_z` under H-exact.
    #[must_use]
    pub fn bool_digits(&self) -> Vec<Column> {
        self.groups
            .iter()
            .flat_map(|group| group.columns.iter().copied())
            .filter(|column| column.is_bool_digit())
            .collect()
    }

    /// Committed polynomials opened at `r^(3)` (every group but `G_m`): 31 under H-u, 32
    /// under H-exact.
    #[must_use]
    pub fn stage3_polynomials(&self) -> usize {
        self.total_polynomials() - MULTIPLICITY_DIGITS
    }

    /// The public shape of the lookup (`docs/lookup.md`): the four stacks
    /// `[Z_lo | U_lo | V_lo | V_hi]` on the main domain, `kappa^(1)` (6 columns in 8 slots)
    /// on `D_1`, `kappa^(2)` on `D_2`, `kappa^(3)` on `D_3`, one table `T_16`, two
    /// challenges.
    #[must_use]
    pub fn lookup_shape(&self) -> LookupShape {
        LookupShape {
            stacks: STACK_DOMAINS
                .iter()
                .enumerate()
                .map(|(stack, &domain)| StackShape {
                    columns: stack_columns(stack, self.relation).len(),
                    vars: self.domain_vars(domain),
                    table: 0,
                    shift: LOOKUP_SHIFT,
                })
                .collect(),
            tables: vec![Table::new(TABLE_VARS as u32)],
            challenges: LOOKUP_CHALLENGES,
        }
    }

    /// Total committed polynomials (38 under H-u, 39 under H-exact).
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
            && self.groups.iter().all(|group| {
                let profile = plan.profile(group.role).group;
                plan.logical_vars(group.role) == group.logical_vars
                    && profile.num_vars() == group.physical_vars
                    && profile.num_polynomials() == group.polynomial_count()
            })
            && plan.relation() == self.relation
            && plan.layout.row_name() == self.catalog_row
    }

    /// Canonical byte encoding of the manifest part of event 0.
    ///
    /// `LE16` wire version, parameter-set byte `0x44`, relation byte ([`Relation::id`],
    /// `0` = H-u, `1` = H-exact), `LE64` `N`, `LE64` `N*`, digit layout id (1 = the lookup
    /// layout), group count; per group: role, round, family (0 = width 4, 1 = width 16),
    /// logical and physical variable counts, polynomial count and one alphabet byte per
    /// polynomial ([`Column::alphabet_code`]); the catalog row id (`LE16` length, ASCII);
    /// the bridge parameters `(J_t, m_t, k_t)` of levels 1 to 3 (`LE64`, `LE64`, `U8`); the
    /// lookup layout (challenge count, table width, and per stack the column count,
    /// variables and storage shift, `U8, U8, U8, U8, LE32`). The public-record digest and
    /// the Flock profile id complete event 0 elsewhere.
    #[must_use]
    pub fn binding_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(160);
        bytes.extend_from_slice(&WIRE_VERSION.to_le_bytes());
        bytes.push(PARAMETER_SET);
        bytes.push(self.relation.id());
        bytes.extend_from_slice(&(self.records as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.padded as u64).to_le_bytes());
        bytes.push(DIGIT_LAYOUT_ID);
        bytes.push(GROUP_COUNT as u8);
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
        // The bridge shape (J_t, m_t, k_t); this wire version admits the production shape only.
        for level in &agg_bridge::Shape::PRODUCTION.levels {
            bytes.extend_from_slice(&(level.cols as u64).to_le_bytes());
            bytes.extend_from_slice(&(level.rows as u64).to_le_bytes());
            bytes.push(level.k as u8);
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
    fn group_shapes_and_digit_counts_match_spec_section_4() {
        let manifest = CommitmentManifest::for_batch(256).unwrap();
        let counts: Vec<usize> = manifest
            .groups()
            .iter()
            .map(GroupDescriptor::polynomial_count)
            .collect();
        assert_eq!(counts, [4, 8, 6, 12, 1, 7]);
        assert_eq!(manifest.total_polynomials(), 38);
        assert_eq!(manifest.stage3_polynomials(), 31);
        assert!(manifest.bool_digits().is_empty());
        assert_eq!(manifest.z_value(), DigitValue::Z);
        let lookups: Vec<Column> = manifest
            .groups()
            .iter()
            .flat_map(|g| g.columns.clone())
            .filter(|c| c.is_lookup())
            .collect();
        // 4 main-domain columns and 6 + 6 + 1 carries.
        assert_eq!(lookups.len(), 17);
        let names: Vec<String> = manifest.leaf_digits().iter().map(|c| c.name()).collect();
        assert_eq!(names, ["z_hi", "u_hi"]);
        assert_eq!(Column::Z_HI.base(), Some(4));
        assert_eq!(Column::U_HI.base(), Some(6));
        assert_eq!(Column::U_HI.alphabet(), Alphabet::Balanced(6));
        assert!(Column::Z_HI.is_leaf_digit() && Column::U_HI.is_leaf_digit());
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
    fn h_exact_groups_follow_docs_h_exact_section_2() {
        let manifest = CommitmentManifest::for_relation(256, Relation::HExact).unwrap();
        let counts: Vec<usize> = manifest
            .groups()
            .iter()
            .map(GroupDescriptor::polynomial_count)
            .collect();
        assert_eq!(counts, [4, 9, 6, 12, 1, 7]);
        assert_eq!((manifest.total_polynomials(), manifest.stage3_polynomials()), (39, 32));
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
        assert_eq!((leaf, bools), (vec!["u_hi".to_owned()], vec!["bz_hi".to_owned(), "a_z".to_owned()]));
        assert_eq!(manifest.z_value(), DigitValue::Zx);
        assert_eq!(value_columns(DigitValue::Zx), [Column::ZX_LO, Column::ZX_HI, Column::A_Z]);
        assert!(Column::ZX_LO.is_lookup() && !Column::ZX_HI.is_lookup());
        assert_eq!((Column::ZX_HI.base(), Column::A_Z.alphabet()), (None, Alphabet::Bool));
        assert_eq!((Column::ZX_HI.alphabet_code(), Column::A_Z.alphabet_code()), (0, 0));
        assert!(Column::ZX_HI.is_bool_digit() && !Column::W1Bit(0).is_bool_digit());
        // The lookup stacks keep their shapes: bz_lo takes the slot of z_lo.
        assert_eq!(manifest.stack_columns(0), [Column::ZX_LO, Column::U_LO, Column::V_LO, Column::V_HI]);
        assert_eq!(
            manifest.lookup_shape(),
            CommitmentManifest::for_batch(256).unwrap().lookup_shape()
        );
        for group in manifest.groups() {
            for (position, &column) in group.columns.iter().enumerate() {
                assert_eq!(manifest.locate(column), (group.role, position));
                let alphabet = column.alphabet();
                assert!(group.family.min_value() <= alphabet.min());
                assert!(alphabet.max() <= group.family.max_value());
            }
        }
        // b_t keep positions 2..7 of G_main4 under both relations.
        let hu = CommitmentManifest::for_batch(256).unwrap();
        for t in 0..W1_BITS as u8 {
            assert_eq!(manifest.locate(Column::W1Bit(t)), hu.locate(Column::W1Bit(t)));
        }
        // H-exact z: z = S_zlo + 2^16 B_zhi + 130,915 a_z - 98,225 over exactly [-B_z, B_z].
        let form = DigitValue::Zx.layout();
        assert_eq!((form.weights(), form.constant()), (vec![1, 1 << 16, 130_915], -98_225));
        assert_eq!(form.range(), (-130_993, 130_993));
    }

    #[test]
    fn h_fips_groups_follow_docs_h_exact_section_10() {
        let manifest = CommitmentManifest::for_relation(256, Relation::HFips).unwrap();
        assert!(manifest.is_fips());
        let counts: Vec<usize> = manifest
            .groups()
            .iter()
            .map(GroupDescriptor::polynomial_count)
            .collect();
        assert_eq!(counts, [4, 12, 6, 13, 1, 7]);
        assert_eq!((manifest.total_polynomials(), manifest.stage3_polynomials()), (43, 36));
        let names = |role| -> Vec<String> {
            manifest.group(role).columns.iter().map(|c| c.name()).collect()
        };
        assert_eq!(names(Group::Main16), ["bz_lo", "l_lo", "v_lo", "v_hi"]);
        assert_eq!(
            names(Group::Main),
            ["bz_hi", "l_hi", "b_0", "b_1", "b_2", "b_3", "b_4", "b_5", "a_z", "s", "delta", "e"]
        );
        assert_eq!(names(Group::Level2).last().map(String::as_str), Some("u_h"));
        assert!(manifest.leaf_digits().is_empty());
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
        let (fips, hu) = (manifest.lookup_shape(), CommitmentManifest::for_batch(256).unwrap().lookup_shape());
        assert_eq!(fips.stacks[2].columns, 7);
        for (a, b) in fips.stacks.iter().zip(&hu.stacks) {
            assert_eq!(a.depth(), b.depth());
        }
        assert_eq!(fips.lookups(0), hu.lookups(0));
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
        // 16 main-domain columns, u_h one element per signature at 2^15: 16,507.5.
        let full = CommitmentManifest::for_relation(1 << 15, Relation::HFips).unwrap();
        let per_signature = full.total_logical_cells() as f64 / f64::from(1 << 15);
        assert!((per_signature - 16_507.5).abs() < 1e-9, "{per_signature}");
        assert_eq!(full.catalog_row(), "mldsa-44-l16f-batch-32768-mult-final");
        // Event 0: relation byte 2, G_main4 at 12 polynomials with the ternary codes.
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
        // z = Z_lo + 2^16 Z_hi + 2^15 and likewise for v.
        for layout in [Z16_DIGITS, V16_DIGITS] {
            assert_eq!(layout.constant(), 1 << 15);
            assert_eq!(layout.weights(), vec![1, 1 << 16]);
        }
        // H-u: u = U_lo + 63,079 U_hi + 31,540 over exactly [-190,465, 190,465].
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
        assert_eq!(Z16_DIGITS.range(), (-(1 << 17), (1 << 17) - 1));
        assert_eq!(V16_DIGITS.range(), (-(1 << 31), (1 << 31) - 1));
        assert_eq!(CARRY16_DIGITS.range(), (0, (1 << 16) - 1));
        assert_eq!(CARRY16_DIGITS.constant(), 1 << 15);
        // The digits of zero: Z_lo = V_lo = -2^15, Z_hi = V_hi = 0; U_lo = -31,540 (B_ulo =
        // 1,228), U_hi = 0 (H_u = 3).
        assert_eq!(Z16_DIGITS.decompose(0), vec![-(1 << 15), 0]);
        assert_eq!(U16_DIGITS.decompose(0), vec![-31_540, 0]);
        assert_eq!(V16_DIGITS.decompose(0), vec![-(1 << 15), 0]);
        for x in [-(1 << 17), -130_993, -1, 0, 1, 130_993, (1 << 17) - 1] {
            let digits = Z16_DIGITS.decompose(x);
            assert!(Z16_DIGITS.in_alphabets(&digits));
            assert_eq!(Z16_DIGITS.compose(&digits), x);
        }
        for x in [-67_068_447, 0, 67_068_447, -(1 << 31), (1 << 31) - 1] {
            let digits = V16_DIGITS.decompose(x);
            assert!(V16_DIGITS.in_alphabets(&digits));
            assert_eq!(V16_DIGITS.compose(&digits), x);
        }
    }

    #[test]
    fn committed_elements_per_signature() {
        // 12,288 main-domain elements, 96 + 12 + 0.5 bridge
        // elements and the 7 * 2^16 multiplicity cells (14 per signature) at N = 2^15.
        let full = CommitmentManifest::for_batch(1 << 15).unwrap();
        assert_eq!(full.d_main(), 25);
        assert_eq!(full.segments(), 4);
        let per_signature = full.total_logical_cells() as f64 / f64::from(1 << 15);
        assert!((per_signature - 12_410.5).abs() < 1e-9, "{per_signature}");
        let small = CommitmentManifest::for_batch(256).unwrap();
        assert_eq!(small.d_main(), 18);
        assert_eq!(small.segments(), 1);
        let logical: Vec<usize> = small.groups().iter().map(|g| g.logical_vars).collect();
        let physical: Vec<usize> = small.groups().iter().map(|g| g.physical_vars).collect();
        assert_eq!(logical, [18, 18, 13, 13, 12, 16]);
        assert_eq!(physical, [18, 18, 13, 13, 13, 16]);
        assert_eq!(small.group(Group::Level3).pad_vars(), 1);
        assert_eq!(small.catalog_row(), "mldsa-44-l16-batch-256-mult-final");
        // H-exact: one more main-domain column, 13,434.5 elements per signature at 2^15;
        // 13,960 physical at N = 1,024.
        let exact = CommitmentManifest::for_relation(1 << 15, Relation::HExact).unwrap();
        let per_signature = exact.total_logical_cells() as f64 / f64::from(1 << 15);
        assert!((per_signature - 13_434.5).abs() < 1e-9, "{per_signature}");
        assert_eq!(exact.catalog_row(), "mldsa-44-l16x-batch-32768-mult-final");
        let physical = |m: &CommitmentManifest| -> usize {
            m.groups()
                .iter()
                .map(|g| g.polynomial_count() << g.physical_vars)
                .sum()
        };
        let k = CommitmentManifest::for_relation(1024, Relation::HExact).unwrap();
        assert_eq!(physical(&k) / 1024, 13_960);
        assert_eq!(physical(&CommitmentManifest::for_batch(1024).unwrap()) / 1024, 12_936);
    }

    #[test]
    fn lookup_shape_follows_the_domains() {
        let manifest = CommitmentManifest::for_batch(1 << 15).unwrap();
        let shape = manifest.lookup_shape();
        shape.validate().unwrap();
        let depths: Vec<usize> = shape.stacks.iter().map(|s| s.depth()).collect();
        assert_eq!(depths, [27, 22, 18, 14]);
        assert_eq!(shape.lookups(0), 138_690_560);
        for (stack, columns) in (0..4).map(|s| (s, stack_columns(s, Relation::Hu))) {
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
            let plan = WorkPlan::for_records(records, Relation::Hu).unwrap();
            assert!(manifest.matches_plan(&plan), "N = {records}");
        }
        for &d_sig in Relation::HExact.catalog_d_sigs() {
            let records = (1 << d_sig) - 3;
            let manifest = CommitmentManifest::for_relation(records, Relation::HExact).unwrap();
            let plan = WorkPlan::for_records(records, Relation::HExact).unwrap();
            assert!(manifest.matches_plan(&plan), "H-exact N = {records}");
            let hu = WorkPlan::for_records(records, Relation::Hu).unwrap();
            assert!(!manifest.matches_plan(&hu), "H-exact against the H-u plan");
        }
        let other = WorkPlan::for_records(512, Relation::Hu).unwrap();
        assert!(!CommitmentManifest::for_batch(256)
            .unwrap()
            .matches_plan(&other));
    }

    #[test]
    fn binding_bytes_are_pinned_and_separate_batches() {
        let manifest = CommitmentManifest::for_batch(200).unwrap();
        let bytes = manifest.binding_bytes();
        let mut expected = vec![4, 0, 0x44, 0];
        expected.extend_from_slice(&200_u64.to_le_bytes());
        expected.extend_from_slice(&256_u64.to_le_bytes());
        expected.extend_from_slice(&[1, 6]);
        assert_eq!(&bytes[..expected.len()], expected.as_slice());
        // G_main16: role 0, round 1, family 1, 18 logical and physical vars, 4 polys.
        let main16 = &bytes[expected.len()..];
        assert_eq!(&main16[..6], &[0, 1, 1, 18, 18, 4]);
        assert_eq!(&main16[6..10], &[0x90; 4]);
        // G_main4: role 1, round 1, family 0, then z_hi (4), u_hi (6, H-u) and six bits.
        let main4 = &main16[10..];
        assert_eq!(&main4[..14], &[1, 1, 0, 18, 18, 8, 4, 6, 0, 0, 0, 0, 0, 0]);
        assert_eq!(manifest.relation(), Relation::Hu);
        assert_eq!(Relation::from_id(RELATION_HU), Some(Relation::Hu));
        assert_eq!(Relation::from_id(RELATION_H_EXACT), Some(Relation::HExact));
        assert_eq!(Relation::from_id(RELATION_H_FIPS), Some(Relation::HFips));
        assert_eq!(Relation::from_id(3), None);
        assert_eq!(
            CommitmentManifest::for_relation(200, Relation::Hu).unwrap(),
            manifest
        );
        let row = manifest.catalog_row().len();
        assert_eq!(
            bytes.len(),
            expected.len() + 6 * 6 + 38 + 2 + row + 3 * 17 + 3 + 1 + 4 * 7
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
        // with codes [0, 6, 0, 0, 0, 0, 0, 0, 0], G_main16 unchanged, the l16x row.
        let exact = CommitmentManifest::for_relation(200, Relation::HExact).unwrap();
        let xbytes = exact.binding_bytes();
        let mut xexpected = expected.clone();
        xexpected[3] = 1;
        assert_eq!(&xbytes[..xexpected.len()], xexpected.as_slice());
        let xmain16 = &xbytes[xexpected.len()..];
        assert_eq!(&xmain16[..10], &main16[..10]);
        assert_eq!(&xmain16[10..25], &[1, 1, 0, 18, 18, 9, 0, 6, 0, 0, 0, 0, 0, 0, 0]);
        let xrow = exact.catalog_row().len();
        assert_eq!(xrow, row + 1);
        assert_eq!(
            xbytes.len(),
            expected.len() + 6 * 6 + 39 + 2 + xrow + 3 * 17 + 3 + 1 + 4 * 7
        );
        assert_ne!(xbytes, bytes);
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
}
