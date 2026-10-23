//! Exact commitment manifest of one batch (spec Sections 4.1, 4.3 and 10, event 0).
//!
//! Mirrors `aerie-falcon/src/protocol/manifest.rs`: the manifest fixes the canonical wire
//! order, the commitment round, the logical and physical domain of every Akita group, and the
//! role and alphabet of each committed polynomial, and serializes the public profile bound in
//! event 0 before any commitment. The four groups are `G_main` (round 1, before the bridge
//! seed) and `G_1, G_2, G_3` (round 2, after it); `G_3` is Akita's final group, so PCS
//! execution follows wire order. Every polynomial uses the width-4 preset (one balanced
//! base-16 inner digit); the alphabets listed here are the honest producer's, proven inside
//! the PIOP by the digit checks of Section 4.3, never by Akita (Remark 10.41).

use agg_pcs_config::catalogs::{BatchLayout, Group};
use agg_pcs_config::work_plan::{padded_size, PlanError, WorkPlan};
use agg_witness::bridge::{J1, J2, J3, K1, K2, K3, M1, M2, M3};
use agg_witness::digits::{
    Alphabet, DigitLayout, KAPPA12_DIGITS, KAPPA3_DIGITS, U_DIGITS, V_DIGITS, Z_DIGITS,
};
use agg_witness::params::W1_BITS;

/// Role of a committed group; the four groups of spec Section 4.1 in wire order.
pub type GroupRole = Group;

/// Wire version of this protocol (spec Status: version 0, unfrozen).
pub const WIRE_VERSION: u16 = 0;
/// Parameter-set byte of ML-DSA-44 (spec Section 10, event 0).
pub const PARAMETER_SET: u8 = 0x44;
/// Identifier of the digit layout of spec Section 4.3 (the only one admitted).
pub const DIGIT_LAYOUT_ID: u8 = 0;
/// Feature-flag byte of relation H (`agg_witness::tables::Layout::H`).
pub const RELATION_H: u8 = 0;
/// Number of committed groups.
pub const GROUP_COUNT: usize = 4;
/// Number of committed polynomials over all groups.
pub const POLYNOMIAL_COUNT: usize = 67;
/// Number of committed digit polynomials (spec Section 4.3: 40 base 16, 2 base 8, 13 base 4).
pub const DIGIT_COUNT: usize = 55;

/// Canonical transcript byte of a group role (its wire position).
#[must_use]
pub const fn role_id(role: GroupRole) -> u8 {
    role.index() as u8
}

/// PCS schedule family of a group. Every group uses the width-4 preset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcsFamily {
    /// `agg_pcs_config::preset::AggFp64SignedWidth4`: values in `[-8, 7]`.
    Width4,
}

impl PcsFamily {
    /// Canonical transcript byte.
    #[must_use]
    pub const fn transcript_id(self) -> u8 {
        match self {
            Self::Width4 => 0,
        }
    }
}

/// A value committed through its balanced digits (spec (4.1)).
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
    /// The digit layout of spec Section 4.3.
    #[must_use]
    pub const fn layout(self) -> DigitLayout {
        match self {
            Self::Z => Z_DIGITS,
            Self::U => U_DIGITS,
            Self::V => V_DIGITS,
            Self::Kappa1(_) | Self::Kappa2(_) => KAPPA12_DIGITS,
            Self::Kappa3 => KAPPA3_DIGITS,
        }
    }

    /// The group holding the digits.
    #[must_use]
    pub const fn group(self) -> GroupRole {
        match self {
            Self::Z | Self::U | Self::V => Group::Main,
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
}

impl Column {
    /// The honest alphabet.
    #[must_use]
    pub fn alphabet(self) -> Alphabet {
        match self {
            Self::Digit { value, digit } => value.layout().alphabet(usize::from(digit)),
            Self::W1Bit(_) | Self::Parity2(_) => Alphabet::Bool,
        }
    }

    /// Digit base, `None` for a Boolean column.
    #[must_use]
    pub fn base(self) -> Option<u32> {
        match self {
            Self::Digit { value, digit } => Some(value.layout().bases[usize::from(digit)]),
            Self::W1Bit(_) | Self::Parity2(_) => None,
        }
    }

    /// The group holding the column.
    #[must_use]
    pub const fn group(self) -> GroupRole {
        match self {
            Self::Digit { value, .. } => value.group(),
            Self::W1Bit(_) => Group::Main,
            Self::Parity2(_) => Group::Level2,
        }
    }

    /// Column name, equal to the names used by `agg-witness` (`z_3`, `b_0`, `kappa1_2_1`, ...).
    #[must_use]
    pub fn name(self) -> String {
        match self {
            Self::Digit { value, digit } => match value {
                DigitValue::Z => format!("z_{digit}"),
                DigitValue::U => format!("u_{digit}"),
                DigitValue::V => format!("v_{digit}"),
                DigitValue::Kappa1(t) => format!("kappa1_{t}_{digit}"),
                DigitValue::Kappa2(t) => format!("kappa2_{t}_{digit}"),
                DigitValue::Kappa3 => format!("kappa3_{digit}"),
            },
            Self::W1Bit(t) => format!("b_{t}"),
            Self::Parity2(t) => format!("y2_{t}"),
        }
    }

    /// Transcript code of the alphabet: the digit base, or 0 for a Boolean.
    #[must_use]
    pub fn alphabet_code(self) -> u8 {
        self.base().map_or(0, |base| base as u8)
    }
}

fn digits_of(value: DigitValue) -> impl Iterator<Item = Column> {
    (0..value.layout().len() as u8).map(move |digit| Column::Digit { value, digit })
}

/// The canonical polynomial order of a group (spec Section 4.1, `docs/witness.md` W3).
#[must_use]
pub fn group_columns(role: GroupRole) -> Vec<Column> {
    let lanes = 0..W1_BITS as u8;
    match role {
        Group::Main => digits_of(DigitValue::Z)
            .chain(digits_of(DigitValue::U))
            .chain(digits_of(DigitValue::V))
            .chain(lanes.map(Column::W1Bit))
            .collect(),
        Group::Level1 => lanes
            .flat_map(|t| digits_of(DigitValue::Kappa1(t)))
            .collect(),
        Group::Level2 => lanes
            .clone()
            .map(Column::Parity2)
            .chain(lanes.flat_map(|t| digits_of(DigitValue::Kappa2(t))))
            .collect(),
        Group::Level3 => digits_of(DigitValue::Kappa3).collect(),
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
            family: PcsFamily::Width4,
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
        self.groups[0].logical_vars
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

    /// Every digit polynomial `(group, position, column)` in the order of the table of spec
    /// Section 4.1 (the order of the `b16_prod`, leaf and link instances).
    #[must_use]
    pub fn digit_columns(&self) -> Vec<(GroupRole, usize, Column)> {
        self.groups
            .iter()
            .flat_map(|group| {
                group
                    .columns
                    .iter()
                    .enumerate()
                    .filter(|(_, column)| column.base().is_some())
                    .map(|(position, &column)| (group.role, position, column))
            })
            .collect()
    }

    /// Total committed polynomials (67).
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
    /// `LE64` `N*`, digit layout id, group count; per group: role, round, family, logical
    /// and physical variable counts, polynomial count and one alphabet byte per polynomial
    /// (the digit base, `0` for a Boolean); the catalog row id (`LE16` length, ASCII); the
    /// bridge parameters `(J_t, m_t, k_t)` of levels 1 to 3 (`LE64`, `LE64`, `U8`). The
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
        for (j, m, k) in [(J1, M1, K1), (J2, M2, K2), (J3, M3, K3)] {
            bytes.extend_from_slice(&(j as u64).to_le_bytes());
            bytes.extend_from_slice(&(m as u64).to_le_bytes());
            bytes.push(k as u8);
        }
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agg_witness::bridge::BridgeTables;
    use agg_witness::tables::Layout;

    #[test]
    fn group_shapes_and_digit_counts_match_spec_section_4() {
        let manifest = CommitmentManifest::for_batch(256).unwrap();
        let counts: Vec<usize> = manifest
            .groups()
            .iter()
            .map(GroupDescriptor::polynomial_count)
            .collect();
        assert_eq!(counts, [23, 18, 24, 2]);
        assert_eq!(manifest.total_polynomials(), POLYNOMIAL_COUNT);
        let digits = manifest.digit_columns();
        assert_eq!(digits.len(), DIGIT_COUNT);
        let count_base = |base| digits.iter().filter(|d| d.2.base() == Some(base)).count();
        assert_eq!((count_base(16), count_base(8), count_base(4)), (40, 2, 13));
        // The base-8 digits are u_4 and v_6.
        let base8: Vec<String> = digits
            .iter()
            .filter(|d| d.2.base() == Some(8))
            .map(|d| d.2.name())
            .collect();
        assert_eq!(base8, ["u_4", "v_6"]);
        for (role, position, column) in digits {
            assert_eq!(manifest.locate(column), (role, position));
            assert_eq!(column.group(), role);
        }
        for group in manifest.groups() {
            assert_eq!(group.round, group.role.round());
            assert_eq!(group.family, PcsFamily::Width4);
            assert_eq!(usize::from(role_id(group.role)), group.role.index());
        }
    }

    #[test]
    fn columns_match_the_witness_tables_by_name_and_alphabet() {
        let main: Vec<(String, Alphabet)> = Layout::H
            .columns()
            .specs
            .into_iter()
            .map(|spec| (spec.name, spec.alphabet))
            .collect();
        let [g1, g2, g3] = BridgeTables::specs();
        for (role, expected) in Group::ALL.into_iter().zip([main, g1, g2, g3]) {
            let actual: Vec<(String, Alphabet)> = group_columns(role)
                .into_iter()
                .map(|column| (column.name(), column.alphabet()))
                .collect();
            assert_eq!(actual, expected, "{role:?}");
        }
    }

    #[test]
    fn committed_elements_per_signature() {
        // Spec Section 4.1: 23,552 main-domain elements plus 288 + 24 + 1 bridge elements at N = 2^15.
        let full = CommitmentManifest::for_batch(1 << 15).unwrap();
        assert_eq!(full.d_main(), 25);
        assert_eq!(full.segments(), 4);
        assert_eq!(full.total_logical_cells(), 23_865 << 15);
        let small = CommitmentManifest::for_batch(256).unwrap();
        assert_eq!(small.d_main(), 18);
        assert_eq!(small.segments(), 1);
        let logical: Vec<usize> = small.groups().iter().map(|g| g.logical_vars).collect();
        let physical: Vec<usize> = small.groups().iter().map(|g| g.physical_vars).collect();
        assert_eq!(logical, [18, 13, 13, 12]);
        assert_eq!(physical, [18, 13, 13, 13]);
        assert_eq!(small.group(Group::Level3).pad_vars(), 1);
        assert_eq!(
            small.catalog_row(),
            "mldsa-44-mode-a-batch-256-level3-final"
        );
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
        let mut expected = vec![0, 0, 0x44, 0];
        expected.extend_from_slice(&200_u64.to_le_bytes());
        expected.extend_from_slice(&256_u64.to_le_bytes());
        expected.extend_from_slice(&[0, 4]);
        assert_eq!(&bytes[..expected.len()], expected.as_slice());
        // Group G_main header: role 0, round 1, family 0, 18 logical and physical vars, 23 polys.
        let main = &bytes[expected.len()..];
        assert_eq!(&main[..6], &[0, 1, 0, 18, 18, 23]);
        assert_eq!(
            &main[6..29],
            &[16, 16, 16, 16, 4, 16, 16, 16, 16, 8, 16, 16, 16, 16, 16, 16, 8, 0, 0, 0, 0, 0, 0]
        );
        let row = manifest.catalog_row().len();
        assert_eq!(bytes.len(), expected.len() + 4 * 6 + 67 + 2 + row + 3 * 17);
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
