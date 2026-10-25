//! Exact commitment manifest of one batch (spec Sections 4.1, 4.3 and 10, event 0), lookup
//! layout L16 (`docs/lookup.md`).
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
//! ranges are proven inside the PIOP (the lookup into `T_16`, the base-4 and base-8 leaves),
//! never by Akita (Remark 10.41).

use agg_pcs_config::catalogs::{BatchLayout, Family, Group, MULTIPLICITY_DIGITS, TABLE_VARS};
use agg_pcs_config::work_plan::{padded_size, PlanError, WorkPlan};
use agg_witness::digits::{Alphabet, DigitLayout};
use agg_witness::params::W1_BITS;

use crate::lookup::{LookupShape, StackShape, Table};

/// Role of a committed group; the six groups of spec Section 4.1 in wire order.
pub type GroupRole = Group;

/// PCS preset family of a group (`agg_pcs_config::catalogs::Family`).
pub type PcsFamily = Family;

/// Wire version of this protocol (spec Status: version 2, the lookup layout L16 with the
/// proof-of-work schedule of `docs/soundness.md`: the plan digest in event 0 and the packed
/// PIOP nonce stream; version 1 had no work).
pub const WIRE_VERSION: u16 = 2;
/// Parameter-set byte of ML-DSA-44 (spec Section 10, event 0).
pub const PARAMETER_SET: u8 = 0x44;
/// Identifier of the digit layout of spec Section 4.3: 1 is the lookup layout L16 (the
/// base-16 layout 0 of wire version 0 is retired).
pub const DIGIT_LAYOUT_ID: u8 = 1;
/// Feature-flag byte of relation H (`agg_witness::tables::Layout::H`).
pub const RELATION_H: u8 = 0;
/// Number of committed groups.
pub const GROUP_COUNT: usize = agg_pcs_config::catalogs::GROUP_COUNT;
/// Number of committed polynomials over all groups.
pub const POLYNOMIAL_COUNT: usize = 38;
/// Number of committed polynomials opened at `r^(3)` (every group but `G_m`).
pub const STAGE3_POLYNOMIALS: usize = POLYNOMIAL_COUNT - MULTIPLICITY_DIGITS;
/// Independent lookup challenges `X_1, X_2` (`docs/lookup.md` Section 2).
pub const LOOKUP_CHALLENGES: usize = 2;
/// The balanced storage shift of every looked-up column: the looked-up value is
/// `f = stored + 2^15` (`docs/lookup.md` Section 1).
pub const LOOKUP_SHIFT: i32 = 1 << 15;

/// `z + 2^17 = z_lo + 2^16 z_hi` (bases `2^16, 4`, signed): `z = Z_lo + 2^16 Z_hi + 2^15`.
pub const Z16_DIGITS: DigitLayout = DigitLayout {
    name: "z",
    bases: &[1 << 16, 4],
    signed: true,
};
/// `u + 2^18 = u_lo + 2^16 u_hi` (bases `2^16, 8`, signed).
pub const U16_DIGITS: DigitLayout = DigitLayout {
    name: "u",
    bases: &[1 << 16, 8],
    signed: true,
};
/// `v + 2^31 = v_lo + 2^16 v_hi` (bases `2^16, 2^16`, signed).
pub const V16_DIGITS: DigitLayout = DigitLayout {
    name: "v",
    bases: &[1 << 16, 1 << 16],
    signed: true,
};
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

/// Canonical transcript byte of a group role (its wire position).
#[must_use]
pub const fn role_id(role: GroupRole) -> u8 {
    role.index() as u8
}

/// A value committed through its balanced digits (spec (4.1) with the L16 bases).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DigitValue {
    /// The response `z` (main domain, indexed by `(j, s, i)`).
    Z,
    /// The UseHint residual `u` (main domain).
    U,
    /// The ring quotient `v` (main domain).
    V,
    /// Level-1 carry `kappa^(1)_t` (domain `D_1`).
    Kappa1(u8),
    /// Level-2 carry `kappa^(2)_t` (domain `D_2`).
    Kappa2(u8),
    /// Level-3 carry `kappa^(3)` (domain `D_3`).
    Kappa3,
}

impl DigitValue {
    /// The digit layout of spec Section 4.3 (L16).
    #[must_use]
    pub const fn layout(self) -> DigitLayout {
        match self {
            Self::Z => Z16_DIGITS,
            Self::U => U16_DIGITS,
            Self::V => V16_DIGITS,
            Self::Kappa1(_) | Self::Kappa2(_) | Self::Kappa3 => CARRY16_DIGITS,
        }
    }

    /// The group holding digit `digit`.
    #[must_use]
    pub const fn group(self, digit: u8) -> GroupRole {
        match self {
            Self::Z | Self::U if digit == 0 => Group::Main16,
            Self::Z | Self::U => Group::Main,
            Self::V => Group::Main16,
            Self::Kappa1(_) => Group::Level1,
            Self::Kappa2(_) => Group::Level2,
            Self::Kappa3 => Group::Level3,
        }
    }
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
            Self::W1Bit(_) | Self::Parity2(_) => Alphabet::Bool,
            Self::Multiplicity(_) => Alphabet::Balanced(16),
        }
    }

    /// Digit base, `None` for a Boolean column or a multiplicity digit.
    #[must_use]
    pub fn base(self) -> Option<u32> {
        match self {
            Self::Digit { value, digit } => Some(value.layout().bases[usize::from(digit)]),
            Self::W1Bit(_) | Self::Parity2(_) | Self::Multiplicity(_) => None,
        }
    }

    /// True for a column looked up in `T_16` (stored balanced, `f = stored + 2^15`).
    #[must_use]
    pub fn is_lookup(self) -> bool {
        self.base() == Some(LOOKUP_BASE)
    }

    /// True for a digit range-checked by a leaf and a link (`Z_hi` base 4, `U_hi` base 8).
    #[must_use]
    pub fn is_leaf_digit(self) -> bool {
        matches!(self.base(), Some(4 | 8))
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
            Self::W1Bit(_) => Group::Main,
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

    /// Column name (`z_lo`, `u_hi`, `b_0`, `kappa1_2`, `y2_0`, `kappa3`, `m_4`, ...).
    #[must_use]
    pub fn name(self) -> String {
        let half = |digit: u8| if digit == 0 { "lo" } else { "hi" };
        match self {
            Self::Digit { value, digit } => match value {
                DigitValue::Z => format!("z_{}", half(digit)),
                DigitValue::U => format!("u_{}", half(digit)),
                DigitValue::V => format!("v_{}", half(digit)),
                DigitValue::Kappa1(t) => format!("kappa1_{t}"),
                DigitValue::Kappa2(t) => format!("kappa2_{t}"),
                DigitValue::Kappa3 => "kappa3".to_owned(),
            },
            Self::W1Bit(t) => format!("b_{t}"),
            Self::Parity2(t) => format!("y2_{t}"),
            Self::Multiplicity(k) => format!("m_{k}"),
        }
    }

    /// Transcript code of the alphabet: the digit base of a leaf-checked digit (4, 8),
    /// [`LOOKUP_ALPHABET_CODE`] for a column looked up in `T_16`,
    /// [`MULTIPLICITY_ALPHABET_CODE`] for a multiplicity digit, 0 for a Boolean.
    #[must_use]
    pub fn alphabet_code(self) -> u8 {
        match self {
            Self::Multiplicity(_) => MULTIPLICITY_ALPHABET_CODE,
            column if column.is_lookup() => LOOKUP_ALPHABET_CODE,
            column => column.base().map_or(0, |base| base as u8),
        }
    }
}

/// The canonical polynomial order of a group (spec Section 4.1, L16).
#[must_use]
pub fn group_columns(role: GroupRole) -> Vec<Column> {
    let lanes = 0..W1_BITS as u8;
    match role {
        Group::Main16 => vec![Column::Z_LO, Column::U_LO, Column::V_LO, Column::V_HI],
        Group::Main => [Column::Z_HI, Column::U_HI]
            .into_iter()
            .chain(lanes.map(Column::W1Bit))
            .collect(),
        Group::Level1 => lanes.map(|t| Column::carry(1, t)).collect(),
        Group::Level2 => lanes
            .clone()
            .map(Column::Parity2)
            .chain(lanes.map(|t| Column::carry(2, t)))
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
    fn new(role: GroupRole, d_sig: usize) -> Self {
        Self {
            role,
            round: role.round(),
            family: role.family(),
            logical_vars: role.logical_vars(d_sig),
            physical_vars: role.physical_vars(d_sig),
            columns: group_columns(role),
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

/// The looked-up columns of one lookup stack (`docs/lookup.md` Section 1), in slot order.
#[must_use]
pub fn stack_columns(stack: usize) -> Vec<Column> {
    let lanes = 0..W1_BITS as u8;
    match stack {
        0 => group_columns(Group::Main16),
        1 => lanes.map(|t| Column::carry(1, t)).collect(),
        2 => lanes.map(|t| Column::carry(2, t)).collect(),
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
    groups: [GroupDescriptor; GROUP_COUNT],
    catalog_row: String,
}

impl CommitmentManifest {
    /// The manifest of a batch of `records` live records (`1 <= N <= 2^15`).
    pub fn for_batch(records: usize) -> Result<Self, PlanError> {
        let (padded, d_sig) = padded_size(records)?;
        let layout = BatchLayout::new(d_sig).expect("d_sig within the wire range");
        Ok(Self {
            records,
            padded,
            d_sig,
            groups: Group::ALL.map(|role| GroupDescriptor::new(role, d_sig)),
            catalog_row: layout.row_name(),
        })
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

    /// The leaf-checked digits `Z_hi` (base 4) and `U_hi` (base 8), in that order (the order
    /// of the leaf and link instances).
    #[must_use]
    pub fn leaf_digits(&self) -> Vec<Column> {
        vec![Column::Z_HI, Column::U_HI]
    }

    /// The public shape of the lookup (`docs/lookup.md` Section 1): the four stacks
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
                    columns: stack_columns(stack).len(),
                    vars: self.domain_vars(domain),
                    table: 0,
                    shift: LOOKUP_SHIFT,
                })
                .collect(),
            tables: vec![Table::new(TABLE_VARS as u32)],
            challenges: LOOKUP_CHALLENGES,
        }
    }

    /// Total committed polynomials (38).
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
            && plan.layout.row_name() == self.catalog_row
    }

    /// Canonical byte encoding of the manifest part of event 0 (spec Section 10).
    ///
    /// `LE16` wire version, parameter-set byte `0x44`, relation flag (`0` = H), `LE64` `N`,
    /// `LE64` `N*`, digit layout id (1 = L16), group count; per group: role, round, family
    /// (0 = width 4, 1 = width 16), logical and physical variable counts, polynomial count
    /// and one alphabet byte per polynomial ([`Column::alphabet_code`]); the catalog row id
    /// (`LE16` length, ASCII); the bridge parameters `(J_t, m_t, k_t)` of levels 1 to 3
    /// (`LE64`, `LE64`, `U8`); the lookup layout (challenge count, table width, and per
    /// stack the column count, variables and storage shift, `U8, U8, U8, U8, LE32`). The
    /// public-record digest and the Flock profile id complete event 0 elsewhere.
    #[must_use]
    pub fn binding_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(160);
        bytes.extend_from_slice(&WIRE_VERSION.to_le_bytes());
        bytes.push(PARAMETER_SET);
        bytes.push(RELATION_H);
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
        assert_eq!(manifest.total_polynomials(), POLYNOMIAL_COUNT);
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
        assert_eq!(Column::U_HI.base(), Some(8));
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
    fn digit_layouts_give_the_l16_ranges() {
        // z = Z_lo + 2^16 Z_hi + 2^15 and likewise for u and v (docs/lookup.md Section 1).
        for layout in [Z16_DIGITS, U16_DIGITS, V16_DIGITS] {
            assert_eq!(layout.constant(), 1 << 15);
            assert_eq!(layout.weights(), vec![1, 1 << 16]);
        }
        assert_eq!(Z16_DIGITS.range(), (-(1 << 17), (1 << 17) - 1));
        assert_eq!(U16_DIGITS.range(), (-(1 << 18), (1 << 18) - 1));
        assert_eq!(V16_DIGITS.range(), (-(1 << 31), (1 << 31) - 1));
        assert_eq!(CARRY16_DIGITS.range(), (0, (1 << 16) - 1));
        assert_eq!(CARRY16_DIGITS.constant(), 1 << 15);
        // The digits of zero: Z_lo = U_lo = V_lo = -2^15, Z_hi = U_hi = V_hi = 0.
        assert_eq!(Z16_DIGITS.decompose(0), vec![-(1 << 15), 0]);
        assert_eq!(U16_DIGITS.decompose(0), vec![-(1 << 15), 0]);
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
        // docs/lookup.md Section 3: 12,288 main-domain elements, 96 + 12 + 0.5 bridge
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
    }

    #[test]
    fn lookup_shape_follows_the_domains() {
        let manifest = CommitmentManifest::for_batch(1 << 15).unwrap();
        let shape = manifest.lookup_shape();
        shape.validate().unwrap();
        let depths: Vec<usize> = shape.stacks.iter().map(|s| s.depth()).collect();
        assert_eq!(depths, [27, 22, 18, 14]);
        assert_eq!(shape.lookups(0), 138_690_560);
        for (stack, columns) in (0..4).map(|s| (s, stack_columns(s))) {
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
            let plan = WorkPlan::for_records(records).unwrap();
            assert!(manifest.matches_plan(&plan), "N = {records}");
        }
        let other = WorkPlan::for_records(512).unwrap();
        assert!(!CommitmentManifest::for_batch(256)
            .unwrap()
            .matches_plan(&other));
    }

    #[test]
    fn binding_bytes_are_pinned_and_separate_batches() {
        let manifest = CommitmentManifest::for_batch(200).unwrap();
        let bytes = manifest.binding_bytes();
        let mut expected = vec![2, 0, 0x44, 0];
        expected.extend_from_slice(&200_u64.to_le_bytes());
        expected.extend_from_slice(&256_u64.to_le_bytes());
        expected.extend_from_slice(&[1, 6]);
        assert_eq!(&bytes[..expected.len()], expected.as_slice());
        // G_main16: role 0, round 1, family 1, 18 logical and physical vars, 4 polys.
        let main16 = &bytes[expected.len()..];
        assert_eq!(&main16[..6], &[0, 1, 1, 18, 18, 4]);
        assert_eq!(&main16[6..10], &[0x90; 4]);
        // G_main4: role 1, round 1, family 0, then z_hi (4), u_hi (8) and six bits.
        let main4 = &main16[10..];
        assert_eq!(&main4[..14], &[1, 1, 0, 18, 18, 8, 4, 8, 0, 0, 0, 0, 0, 0]);
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
