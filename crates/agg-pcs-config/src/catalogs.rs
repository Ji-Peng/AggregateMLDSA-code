//! Group layouts of spec Section 4.1 and the generated fold-schedule catalog.
//!
//! The four witness groups, in wire and catalog order, are `G_main` (round 1)
//! and `G_1`, `G_2`, `G_3` (round 2); `G_3` is Akita's final group and the
//! others are precommitted groups. A catalog row is keyed by the complete
//! ordered layout, so every batch size `N* = 2^d_sig` needs one grouped row
//! plus one scalar (single-group) row per precommitted layout, whose root
//! profile is the producer profile of that group.
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

/// Batch sizes (`d_sig`) whose catalog rows are generated and checked in:
/// every `N* = 2^3 .. 2^15` of the wire version (spec Section 13).
pub const CATALOG_D_SIG: &[usize] = &[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// The four committed groups of spec Section 4.1, in wire order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Group {
    /// `z_0..z_4, u_0..u_4, v_0..v_6, b_0..b_5` on the main domain (round 1).
    Main,
    /// Level-1 carry digits `kappa^(1)_{t,k}` on `D_1` (round 2).
    Level1,
    /// Level-2 parities `y^(2)_t` and carry digits `kappa^(2)_{t,k}` on `D_2` (round 2).
    Level2,
    /// Level-3 carry digits `kappa^(3)_k` on `D_3` (round 2, Akita final group).
    Level3,
}

impl Group {
    /// All groups in wire and catalog order.
    pub const ALL: [Self; 4] = [Self::Main, Self::Level1, Self::Level2, Self::Level3];

    /// Position in the wire order.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Main => 0,
            Self::Level1 => 1,
            Self::Level2 => 2,
            Self::Level3 => 3,
        }
    }

    /// Commitment round (spec Section 10): 1 before the bridge seed, 2 after.
    #[must_use]
    pub const fn round(self) -> u8 {
        match self {
            Self::Main => 1,
            Self::Level1 | Self::Level2 | Self::Level3 => 2,
        }
    }

    /// Whether this is Akita's final group (the last group of the batch).
    #[must_use]
    pub const fn is_final(self) -> bool {
        matches!(self, Self::Level3)
    }

    /// Number of committed polynomials (spec Section 4.1):
    /// main `5 + 5 + 7 + 6`, level 1 `3 * 6`, level 2 `6 + 3 * 6`, level 3 `2`.
    #[must_use]
    pub const fn num_polynomials(self) -> usize {
        match self {
            Self::Main => 5 + 5 + 7 + 6,
            Self::Level1 => 3 * 6,
            Self::Level2 => 6 + 3 * 6,
            Self::Level3 => 2,
        }
    }

    /// Logical variable count of the group's domain at `d_sig`
    /// (spec Section 1): `d_main = 10 + d_sig`, `d_1 = 13 + max(0, d_sig - 9)`,
    /// `d_2 = 13 + max(0, d_sig - 13)`, `d_3 = 12 + max(0, d_sig - 13)`.
    #[must_use]
    pub const fn logical_vars(self, d_sig: usize) -> usize {
        match self {
            Self::Main => D_POS + d_sig,
            Self::Level1 => 13 + d_sig.saturating_sub(9),
            Self::Level2 => 13 + d_sig.saturating_sub(D_SEG_MAX),
            Self::Level3 => 12 + d_sig.saturating_sub(D_SEG_MAX),
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
            Self::Main => "main",
            Self::Level1 => "level1",
            Self::Level2 => "level2",
            Self::Level3 => "level3",
        }
    }
}

/// The ordered four-group layout of one batch size `N* = 2^d_sig`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchLayout {
    /// `log2 N*`.
    pub d_sig: usize,
    /// Physical layouts of `G_main, G_1, G_2, G_3`.
    pub groups: [PolynomialGroupLayout; 4],
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

    /// The precommitted groups `G_main, G_1, G_2`, in order.
    #[must_use]
    pub fn precommitted(&self) -> &[PolynomialGroupLayout] {
        &self.groups[..3]
    }

    /// The final group `G_3`.
    #[must_use]
    pub const fn final_group(&self) -> PolynomialGroupLayout {
        self.groups[3]
    }

    /// Total committed polynomials (67).
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
            .expect("four groups")
    }

    /// Catalog row name of the grouped row.
    #[must_use]
    pub fn row_name(&self) -> String {
        format!(
            "mldsa-44-mode-a-batch-{}-level3-final",
            1_usize << self.d_sig
        )
    }

    /// Whether this batch size has checked-in catalog rows.
    #[must_use]
    pub fn in_catalog(&self) -> bool {
        CATALOG_D_SIG.contains(&self.d_sig)
    }
}

/// Every distinct scalar (precommitted-producer) layout needed by the
/// batch sizes in `d_sigs`, sorted by `(num_vars, num_polynomials)`.
#[must_use]
pub fn scalar_layouts(d_sigs: &[usize]) -> Vec<PolynomialGroupLayout> {
    let mut layouts: Vec<PolynomialGroupLayout> = d_sigs
        .iter()
        .filter_map(|&d_sig| BatchLayout::new(d_sig))
        .flat_map(|layout| layout.precommitted().to_vec())
        .collect();
    layouts.sort_by_key(|layout| (layout.num_vars(), layout.num_polynomials()));
    layouts.dedup();
    layouts
}

/// The checked-in catalog artifact.
#[cfg(feature = "schedules")]
pub(crate) const WIDTH4_ARTIFACT: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/schedules/agg_mldsa_fp64_signed_width4.aks"
));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_shapes_match_spec_section_4() {
        assert_eq!(Group::ALL.map(Group::num_polynomials), [23, 18, 24, 2]);
        let total: usize = Group::ALL.iter().map(|group| group.num_polynomials()).sum();
        assert_eq!(total, 67);
        // Spec Section 1 table at N* = 2^15.
        assert_eq!(Group::Main.logical_vars(15), 25);
        assert_eq!(Group::Level1.logical_vars(15), 19);
        assert_eq!(Group::Level2.logical_vars(15), 15);
        assert_eq!(Group::Level3.logical_vars(15), 14);
        // Level 3 sits below the floor up to one segment.
        assert_eq!(Group::Level3.logical_vars(13), 12);
        assert_eq!(Group::Level3.physical_vars(13), 13);
        assert_eq!(Group::Main.physical_vars(3), 13);
    }

    #[test]
    fn batch_layouts_for_the_catalog_sizes() {
        let shape = |d_sig| {
            BatchLayout::new(d_sig)
                .unwrap()
                .groups
                .map(|group| (group.num_vars(), group.num_polynomials()))
        };
        assert_eq!(shape(8), [(18, 23), (13, 18), (13, 24), (13, 2)]);
        assert_eq!(shape(10), [(20, 23), (14, 18), (13, 24), (13, 2)]);
        assert_eq!(shape(13), [(23, 23), (17, 18), (13, 24), (13, 2)]);
        assert_eq!(shape(15), [(25, 23), (19, 18), (15, 24), (14, 2)]);
        assert!(BatchLayout::new(2).is_none());
        assert!(BatchLayout::new(16).is_none());
        let layout = BatchLayout::new(10).unwrap();
        assert_eq!(layout.total_polynomials(), 67);
        assert_eq!(layout.max_physical_vars(), 20);
        assert_eq!(layout.row_name(), "mldsa-44-mode-a-batch-1024-level3-final");
    }

    #[test]
    fn scalar_layouts_are_deduplicated() {
        let layouts = scalar_layouts(&[8, 10]);
        let shapes: Vec<_> = layouts
            .iter()
            .map(|layout| (layout.num_vars(), layout.num_polynomials()))
            .collect();
        assert_eq!(
            shapes,
            vec![(13, 18), (13, 24), (14, 18), (18, 23), (20, 23)]
        );
    }
}
