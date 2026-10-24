//! Group layouts of spec Section 4.1 (lookup layout L16, `docs/lookup.md`) and the
//! generated fold-schedule catalogs.
//!
//! The six witness groups, in wire and catalog order, are `G_main16` and `G_main4`
//! (round 1) and `G_1`, `G_2`, `G_3`, `G_m` (round 2); `G_m` (the multiplicity digits of
//! the lookup table `T_16`) is Akita's final group and the others are precommitted groups.
//! Two presets share one batch: the width-16 family (`G_main16`, `G_1`, `G_2`, `G_3`) and
//! the width-4 family (`G_main4`, `G_m`). Akita freezes the source bound into each group's
//! own profile, so a precommitted group of either family enters the final family's grouped
//! row by its scalar producer profile (Akita's mixed-bound batches; the fp128 test
//! `bounded_dense_precommit_with_onehot_final_group` exercises the same path).
//!
//! A catalog row is keyed by the complete ordered layout, so every batch size
//! `N* = 2^d_sig` needs one grouped row in the width-4 catalog plus one scalar row per
//! precommitted layout in its own family's catalog, whose root profile is the producer
//! profile of that group.
//!
//! [`CATALOG_D_SIG`] lists the batch sizes whose rows are checked in;
//! `tests/schedule_catalog_regen.rs` regenerates them (docs/pcs-config.md).

use akita_params::PolynomialGroupLayout;

/// Log2 of the PCS floor: every physical table has at least `2^13` cells
/// (Aerie `PCS_MIN_CELLS_LOG2`); smaller logical tables are zero-extended
/// with trailing pad variables.
pub const PCS_MIN_CELLS_LOG2: usize = 13;
/// Position variables `(j, rho or s)` of the main domain.
pub const D_POS: usize = 10;
/// Smallest `d_sig` (`N* = max(next_power_of_two(N), 8)`).
pub const MIN_D_SIG: usize = 3;
/// Largest `d_sig` of this wire version (`N <= 2^15`).
pub const MAX_D_SIG: usize = 15;
/// Signatures per segment, log2 (spec Section 1).
pub const D_SEG_MAX: usize = 13;
/// Variables of the lookup table `T_16` and of its multiplicity column.
pub const TABLE_VARS: usize = 16;
/// Balanced base-16 digits of one multiplicity (`m < 7 (16^7 - 1) / 15`).
pub const MULTIPLICITY_DIGITS: usize = 7;
/// Groups of one batch.
pub const GROUP_COUNT: usize = 6;

/// Batch sizes (`d_sig`) whose catalog rows are generated and checked in:
/// every `N* = 2^3 .. 2^15` of the wire version (spec Section 13).
pub const CATALOG_D_SIG: &[usize] = &[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// The commitment preset (schedule family) of a group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Family {
    /// `preset::AggFp64SignedWidth4`: values in `[-8, 7]`.
    Width4,
    /// `preset::AggFp64SignedWidth16`: values in `[-2^15, 2^15 - 1]`.
    Width16,
}

impl Family {
    /// Both families.
    pub const ALL: [Self; 2] = [Self::Width4, Self::Width16];

    /// Canonical transcript byte (event 0).
    #[must_use]
    pub const fn transcript_id(self) -> u8 {
        match self {
            Self::Width4 => 0,
            Self::Width16 => 1,
        }
    }

    /// Smallest admitted value.
    #[must_use]
    pub const fn min_value(self) -> i64 {
        match self {
            Self::Width4 => -8,
            Self::Width16 => -(1 << 15),
        }
    }

    /// Largest admitted value.
    #[must_use]
    pub const fn max_value(self) -> i64 {
        match self {
            Self::Width4 => 7,
            Self::Width16 => (1 << 15) - 1,
        }
    }
}

/// The six committed groups of spec Section 4.1 (L16), in wire order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Group {
    /// `Z_lo, U_lo, V_lo, V_hi` on the main domain (round 1, width 16): the 16-bit
    /// lookup columns.
    Main16,
    /// `Z_hi, U_hi, b_0..b_5` on the main domain (round 1, width 4).
    Main,
    /// Level-1 carries `kappa^(1)_t` on `D_1` (round 2, width 16).
    Level1,
    /// Level-2 parities `y^(2)_t` and carries `kappa^(2)_t` on `D_2` (round 2, width 16).
    Level2,
    /// The level-3 carry `kappa^(3)` on `D_3` (round 2, width 16).
    Level3,
    /// The seven balanced base-16 digits of the multiplicity column `m` of `T_16`
    /// (round 2, width 4, Akita final group).
    Mult,
}

impl Group {
    /// All groups in wire and catalog order.
    pub const ALL: [Self; GROUP_COUNT] = [
        Self::Main16,
        Self::Main,
        Self::Level1,
        Self::Level2,
        Self::Level3,
        Self::Mult,
    ];

    /// Position in the wire order.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Main16 => 0,
            Self::Main => 1,
            Self::Level1 => 2,
            Self::Level2 => 3,
            Self::Level3 => 4,
            Self::Mult => 5,
        }
    }

    /// Commitment round (spec Section 10): 1 before the bridge seed, 2 after.
    #[must_use]
    pub const fn round(self) -> u8 {
        match self {
            Self::Main16 | Self::Main => 1,
            Self::Level1 | Self::Level2 | Self::Level3 | Self::Mult => 2,
        }
    }

    /// Whether this is Akita's final group (the last group of the batch).
    #[must_use]
    pub const fn is_final(self) -> bool {
        matches!(self, Self::Mult)
    }

    /// The preset of the group.
    #[must_use]
    pub const fn family(self) -> Family {
        match self {
            Self::Main | Self::Mult => Family::Width4,
            Self::Main16 | Self::Level1 | Self::Level2 | Self::Level3 => Family::Width16,
        }
    }

    /// Number of committed polynomials (spec Section 4.1, L16):
    /// `4 + 8` on the main domain, 6 carries, `6 + 6` parities and carries, 1 carry, and
    /// the 7 multiplicity digits.
    #[must_use]
    pub const fn num_polynomials(self) -> usize {
        match self {
            Self::Main16 => 4,
            Self::Main => 2 + 6,
            Self::Level1 => 6,
            Self::Level2 => 6 + 6,
            Self::Level3 => 1,
            Self::Mult => MULTIPLICITY_DIGITS,
        }
    }

    /// Logical variable count of the group's domain at `d_sig`
    /// (spec Section 1): `d_main = 10 + d_sig`, `d_1 = 13 + max(0, d_sig - 9)`,
    /// `d_2 = 13 + max(0, d_sig - 13)`, `d_3 = 12 + max(0, d_sig - 13)`; 16 for `G_m`.
    #[must_use]
    pub const fn logical_vars(self, d_sig: usize) -> usize {
        match self {
            Self::Main16 | Self::Main => D_POS + d_sig,
            Self::Level1 => 13 + d_sig.saturating_sub(9),
            Self::Level2 => 13 + d_sig.saturating_sub(D_SEG_MAX),
            Self::Level3 => 12 + d_sig.saturating_sub(D_SEG_MAX),
            Self::Mult => TABLE_VARS,
        }
    }

    /// Physical (committed) variable count: the logical count raised to the
    /// PCS floor.
    #[must_use]
    pub const fn physical_vars(self, d_sig: usize) -> usize {
        let logical = self.logical_vars(d_sig);
        if logical < PCS_MIN_CELLS_LOG2 {
            PCS_MIN_CELLS_LOG2
        } else {
            logical
        }
    }

    /// Akita group layout at `d_sig`.
    #[must_use]
    pub const fn layout(self, d_sig: usize) -> PolynomialGroupLayout {
        PolynomialGroupLayout::new(self.physical_vars(d_sig), self.num_polynomials())
    }

    /// Short label used in catalog row names and logs.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Main16 => "main16",
            Self::Main => "main",
            Self::Level1 => "level1",
            Self::Level2 => "level2",
            Self::Level3 => "level3",
            Self::Mult => "mult",
        }
    }
}

/// The ordered six-group layout of one batch size `N* = 2^d_sig`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchLayout {
    /// `log2 N*`.
    pub d_sig: usize,
    /// Physical layouts of `G_main16, G_main4, G_1, G_2, G_3, G_m`.
    pub groups: [PolynomialGroupLayout; GROUP_COUNT],
}

impl BatchLayout {
    /// The layout at `d_sig`; `None` outside `[MIN_D_SIG, MAX_D_SIG]`.
    #[must_use]
    pub fn new(d_sig: usize) -> Option<Self> {
        (MIN_D_SIG..=MAX_D_SIG).contains(&d_sig).then(|| Self {
            d_sig,
            groups: Group::ALL.map(|group| group.layout(d_sig)),
        })
    }

    /// The precommitted groups (every group but `G_m`), in order.
    #[must_use]
    pub fn precommitted(&self) -> &[PolynomialGroupLayout] {
        &self.groups[..GROUP_COUNT - 1]
    }

    /// The precommitted groups with their roles, in order.
    #[must_use]
    pub fn precommitted_groups(&self) -> Vec<(Group, PolynomialGroupLayout)> {
        Group::ALL[..GROUP_COUNT - 1]
            .iter()
            .map(|&group| (group, self.groups[group.index()]))
            .collect()
    }

    /// The final group `G_m`.
    #[must_use]
    pub const fn final_group(&self) -> PolynomialGroupLayout {
        self.groups[GROUP_COUNT - 1]
    }

    /// Total committed polynomials (38).
    #[must_use]
    pub fn total_polynomials(&self) -> usize {
        self.groups
            .iter()
            .map(|group| group.num_polynomials())
            .sum()
    }

    /// Largest physical variable count (the main domain).
    #[must_use]
    pub fn max_physical_vars(&self) -> usize {
        self.groups
            .iter()
            .map(|group| group.num_vars())
            .max()
            .expect("six groups")
    }

    /// Catalog row name of the grouped row.
    #[must_use]
    pub fn row_name(&self) -> String {
        format!("mldsa-44-l16-batch-{}-mult-final", 1_usize << self.d_sig)
    }

    /// Whether this batch size has checked-in catalog rows.
    #[must_use]
    pub fn in_catalog(&self) -> bool {
        CATALOG_D_SIG.contains(&self.d_sig)
    }
}

/// Every distinct scalar (precommitted-producer) layout of `family` needed by the batch
/// sizes in `d_sigs`, sorted by `(num_vars, num_polynomials)`.
#[must_use]
pub fn scalar_layouts(d_sigs: &[usize], family: Family) -> Vec<PolynomialGroupLayout> {
    let mut layouts: Vec<PolynomialGroupLayout> = d_sigs
        .iter()
        .filter_map(|&d_sig| BatchLayout::new(d_sig))
        .flat_map(|layout| {
            layout
                .precommitted_groups()
                .into_iter()
                .filter(|(group, _)| group.family() == family)
                .map(|(_, layout)| layout)
                .collect::<Vec<_>>()
        })
        .collect();
    layouts.sort_by_key(|layout| (layout.num_vars(), layout.num_polynomials()));
    layouts.dedup();
    layouts
}

/// The checked-in width-4 catalog artifact (scalar `G_main4` rows and the grouped rows).
#[cfg(feature = "schedules")]
pub(crate) const WIDTH4_ARTIFACT: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/schedules/agg_mldsa_fp64_signed_width4.aks"
));

/// The checked-in width-16 catalog artifact (scalar rows of `G_main16, G_1, G_2, G_3`).
#[cfg(feature = "schedules")]
pub(crate) const WIDTH16_ARTIFACT: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/schedules/agg_mldsa_fp64_signed_width16.aks"
));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_shapes_match_spec_section_4() {
        assert_eq!(Group::ALL.map(Group::num_polynomials), [4, 8, 6, 12, 1, 7]);
        let total: usize = Group::ALL.iter().map(|group| group.num_polynomials()).sum();
        assert_eq!(total, 38);
        // Spec Section 1 table at N* = 2^15.
        assert_eq!(Group::Main16.logical_vars(15), 25);
        assert_eq!(Group::Main.logical_vars(15), 25);
        assert_eq!(Group::Level1.logical_vars(15), 19);
        assert_eq!(Group::Level2.logical_vars(15), 15);
        assert_eq!(Group::Level3.logical_vars(15), 14);
        assert_eq!(Group::Mult.logical_vars(3), 16);
        // Level 3 sits below the floor up to one segment.
        assert_eq!(Group::Level3.logical_vars(13), 12);
        assert_eq!(Group::Level3.physical_vars(13), 13);
        assert_eq!(Group::Main.physical_vars(3), 13);
        for group in Group::ALL {
            assert_eq!(Group::ALL[group.index()], group);
            assert_eq!(group.is_final(), group == Group::Mult);
        }
        assert_eq!(
            Group::ALL.map(Group::family),
            [
                Family::Width16,
                Family::Width4,
                Family::Width16,
                Family::Width16,
                Family::Width16,
                Family::Width4
            ]
        );
    }

    #[test]
    fn batch_layouts_for_the_catalog_sizes() {
        let shape = |d_sig| {
            BatchLayout::new(d_sig)
                .unwrap()
                .groups
                .map(|group| (group.num_vars(), group.num_polynomials()))
        };
        assert_eq!(
            shape(8),
            [(18, 4), (18, 8), (13, 6), (13, 12), (13, 1), (16, 7)]
        );
        assert_eq!(
            shape(15),
            [(25, 4), (25, 8), (19, 6), (15, 12), (14, 1), (16, 7)]
        );
        assert!(BatchLayout::new(2).is_none());
        assert!(BatchLayout::new(16).is_none());
        let layout = BatchLayout::new(10).unwrap();
        assert_eq!(layout.total_polynomials(), 38);
        assert_eq!(layout.max_physical_vars(), 20);
        assert_eq!(layout.row_name(), "mldsa-44-l16-batch-1024-mult-final");
        assert_eq!(layout.precommitted().len(), 5);
        assert_eq!(layout.final_group(), PolynomialGroupLayout::new(16, 7));
    }

    #[test]
    fn scalar_layouts_are_deduplicated_per_family() {
        let shapes = |family| -> Vec<(usize, usize)> {
            scalar_layouts(&[8, 10], family)
                .iter()
                .map(|layout| (layout.num_vars(), layout.num_polynomials()))
                .collect()
        };
        assert_eq!(shapes(Family::Width4), vec![(18, 8), (20, 8)]);
        assert_eq!(
            shapes(Family::Width16),
            vec![(13, 1), (13, 6), (13, 12), (14, 6), (18, 4), (20, 4)]
        );
    }
}
