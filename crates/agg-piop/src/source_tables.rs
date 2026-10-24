//! Batch-level committed source tables (spec Sections 4.1, 4.2 and 9.2).
//!
//! Mirrors `aerie-prover/src/protocol/source_tables.rs`: one holder per commitment round,
//! built from the `agg-witness` batch and checked against the [`CommitmentManifest`]:
//!
//! * [`MainTables`], round 1: `G_main` (`z, u, v` digits and the six bits `b_t`) from
//!   `agg_witness::tables::BatchTables` of relation H. Padded rows carry the canonical
//!   encoding of the zero witness (digits `-b/2`, top signed digit 0, zero bits).
//! * [`BridgeGroupTables`], round 2: `G_1, G_2, G_3` (carry digits and level-2 parities) and
//!   the public `Y`, computed by `agg_bridge::parity` (sums, parities, quotient columns) and
//!   `agg_bridge::carry` (carry digits) at the production shape from the committed bits and
//!   the seed-derived matrices. The honest prover aborts on a row weight above the cap.
//!
//! Construction validates the shape and the honest alphabet of every polynomial. This is
//! an encoder self-check (the honest builder cannot fail it), not a soundness check: the
//! PIOP proves every alphabet (Section 4.3).

use agg_bridge::{carry, parity, Shape};
use agg_witness::bridge::{BridgeError, BridgeMatrices, BridgeTables, M3};
use agg_witness::instance::PublicInstance;
use agg_witness::padding::Geometry;
use agg_witness::params::W1_BITS;
use agg_witness::tables::{BatchTables, BuildError, Layout};
use rayon::prelude::*;

use crate::manifest::{Column, CommitmentManifest, DigitValue, GroupDescriptor, GroupRole};

/// A failure while building or checking the committed tables.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SourceTableError {
    /// The batch is not of relation H.
    Layout(Layout),
    /// The batch has another record count than the manifest.
    Records {
        /// Manifest `N`.
        expected: usize,
        /// Batch `N`.
        actual: usize,
    },
    /// A group has the wrong number of polynomials.
    ColumnCount {
        /// The group.
        group: GroupRole,
        /// Manifest count.
        expected: usize,
        /// Received count.
        actual: usize,
    },
    /// A polynomial has the wrong number of logical cells.
    ColumnLength {
        /// The group.
        group: GroupRole,
        /// Polynomial position.
        column: usize,
        /// Manifest cells.
        expected: usize,
        /// Received cells.
        actual: usize,
    },
    /// A cell lies outside its polynomial's honest alphabet.
    Alphabet {
        /// The group.
        group: GroupRole,
        /// Polynomial position.
        column: usize,
        /// Cell index.
        cell: usize,
        /// The value.
        value: i8,
    },
    /// The public `Y` has the wrong length or a non-bit entry.
    PublicY,
    /// Witness derivation or encoding failed.
    Build(BuildError),
    /// The bridge matrices are malformed or a row weight exceeds its cap (honest abort).
    Bridge(BridgeError),
    /// A carry digit does not fit a committed cell.
    Encode(&'static str),
}

impl std::fmt::Display for SourceTableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for SourceTableError {}

/// Shape and alphabet self-check of one group against its descriptor.
pub fn validate_group(
    descriptor: &GroupDescriptor,
    columns: &[Vec<i8>],
) -> Result<(), SourceTableError> {
    let group = descriptor.role;
    if columns.len() != descriptor.polynomial_count() {
        return Err(SourceTableError::ColumnCount {
            group,
            expected: descriptor.polynomial_count(),
            actual: columns.len(),
        });
    }
    for (position, (cells, &column)) in columns.iter().zip(&descriptor.columns).enumerate() {
        if cells.len() != descriptor.logical_cells() {
            return Err(SourceTableError::ColumnLength {
                group,
                column: position,
                expected: descriptor.logical_cells(),
                actual: cells.len(),
            });
        }
        let alphabet = column.alphabet();
        let (min, max) = (alphabet.min(), alphabet.max());
        let bad = cells
            .par_iter()
            .position_first(|&value| !(min..=max).contains(&i64::from(value)));
        if let Some(cell) = bad {
            return Err(SourceTableError::Alphabet {
                group,
                column: position,
                cell,
                value: cells[cell],
            });
        }
    }
    Ok(())
}

/// Round-1 tables: the `G_main` polynomials in manifest order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MainTables {
    geom: Geometry,
    columns: Vec<Vec<i8>>,
}

impl MainTables {
    /// Takes the tables of an `agg-witness` batch (relation H) and checks them against the
    /// manifest.
    pub fn from_batch(
        manifest: &CommitmentManifest,
        batch: BatchTables,
    ) -> Result<Self, SourceTableError> {
        if batch.layout != Layout::H {
            return Err(SourceTableError::Layout(batch.layout));
        }
        if batch.geom.n != manifest.records() || batch.geom.n_star != manifest.padded_records() {
            return Err(SourceTableError::Records {
                expected: manifest.records(),
                actual: batch.geom.n,
            });
        }
        validate_group(manifest.group(GroupRole::Main), &batch.columns)?;
        Ok(Self {
            geom: batch.geom,
            columns: batch.columns,
        })
    }

    /// Derives every live witness (strict FIPS 204 verification) and builds `G_main`.
    pub fn build(
        manifest: &CommitmentManifest,
        instance: &PublicInstance,
        signatures: &[&[u8]],
    ) -> Result<Self, SourceTableError> {
        let batch =
            BatchTables::build(Layout::H, instance, signatures).map_err(SourceTableError::Build)?;
        Self::from_batch(manifest, batch)
    }

    /// Batch geometry.
    #[must_use]
    pub const fn geometry(&self) -> Geometry {
        self.geom
    }

    /// The 23 polynomials in manifest order (`2^d_main` cells each).
    #[must_use]
    pub fn columns(&self) -> &[Vec<i8>] {
        &self.columns
    }

    /// One polynomial of `G_main`.
    #[must_use]
    pub fn column(&self, manifest: &CommitmentManifest, column: Column) -> &[i8] {
        let (role, position) = manifest.locate(column);
        assert_eq!(
            role,
            GroupRole::Main,
            "{} is not a G_main column",
            column.name()
        );
        &self.columns[position]
    }

    /// The digit polynomials of `z`, `u` or `v`, least significant first.
    #[must_use]
    pub fn digits(&self, manifest: &CommitmentManifest, value: DigitValue) -> Vec<&[i8]> {
        (0..value.layout().len() as u8)
            .map(|digit| self.column(manifest, Column::Digit { value, digit }))
            .collect()
    }

    /// The six bit lanes `b_t`.
    #[must_use]
    pub fn bit_lanes(&self, manifest: &CommitmentManifest) -> [&[i8]; W1_BITS] {
        std::array::from_fn(|t| self.column(manifest, Column::W1Bit(t as u8)))
    }

    /// Mutable access for adversarial tests (the alphabet is not rechecked).
    #[doc(hidden)]
    pub fn columns_mut(&mut self) -> &mut [Vec<i8>] {
        &mut self.columns
    }
}

/// Round-2 tables: `G_1, G_2, G_3` in manifest order and the public bridge vectors `Y`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BridgeGroupTables {
    level1: Vec<Vec<i8>>,
    level2: Vec<Vec<i8>>,
    level3: Vec<Vec<i8>>,
    y: Vec<u8>,
}

impl BridgeGroupTables {
    /// The honest round-2 tables for the committed bits of `main` and the matrices derived
    /// from the bridge seed (spec Sections 9.1, 9.2).
    pub fn build(
        manifest: &CommitmentManifest,
        main: &MainTables,
        matrices: &BridgeMatrices,
    ) -> Result<Self, SourceTableError> {
        let shape = Shape::PRODUCTION;
        let witness = parity::compute(&shape, &main.bit_lanes(manifest), matrices)
            .map_err(SourceTableError::Bridge)?;
        let tables = carry::group_tables(&shape, &witness)
            .map_err(|_| SourceTableError::Encode("carry outside its digit range"))?;
        Self::from_tables(manifest, tables)
    }

    /// Takes `agg-witness` bridge tables and checks them against the manifest.
    pub fn from_tables(
        manifest: &CommitmentManifest,
        tables: BridgeTables,
    ) -> Result<Self, SourceTableError> {
        validate_group(manifest.group(GroupRole::Level1), &tables.g1)?;
        validate_group(manifest.group(GroupRole::Level2), &tables.g2)?;
        validate_group(manifest.group(GroupRole::Level3), &tables.g3)?;
        if tables.y.len() != manifest.group(GroupRole::Level3).logical_cells()
            || tables.y.iter().any(|&bit| bit > 1)
        {
            return Err(SourceTableError::PublicY);
        }
        Ok(Self {
            level1: tables.g1,
            level2: tables.g2,
            level3: tables.g3,
            y: tables.y,
        })
    }

    /// The polynomials of a round-2 group.
    #[must_use]
    pub fn group(&self, role: GroupRole) -> &[Vec<i8>] {
        match role {
            GroupRole::Level1 => &self.level1,
            GroupRole::Level2 => &self.level2,
            GroupRole::Level3 => &self.level3,
            GroupRole::Main => panic!("G_main is a round-1 group"),
        }
    }

    /// The public `Y` on `D_3` (`Y_sigma[rho_3]` at `rho_3 + 2^12 sigma`).
    #[must_use]
    pub fn public_y(&self) -> &[u8] {
        &self.y
    }

    /// `Y_sigma`, the 4,096 bits of segment `sigma` (event-6 payload, 512 bytes packed).
    #[must_use]
    pub fn segment_y(&self, sigma: usize) -> &[u8] {
        &self.y[sigma * M3..(sigma + 1) * M3]
    }

    /// Mutable access for adversarial tests (the alphabet is not rechecked).
    #[doc(hidden)]
    pub fn group_mut(&mut self, role: GroupRole) -> &mut [Vec<i8>] {
        match role {
            GroupRole::Level1 => &mut self.level1,
            GroupRole::Level2 => &mut self.level2,
            GroupRole::Level3 => &mut self.level3,
            GroupRole::Main => panic!("G_main is a round-1 group"),
        }
    }

    /// The equivalent `agg-witness` tables (for the integer checker).
    #[must_use]
    pub fn to_witness_tables(&self) -> BridgeTables {
        BridgeTables {
            g1: self.level1.clone(),
            g2: self.level2.clone(),
            g3: self.level3.clone(),
            y: self.y.clone(),
        }
    }
}

/// Both rounds' tables, addressed by group and column.
#[derive(Clone, Copy, Debug)]
pub struct SourceTables<'a> {
    /// The manifest the tables were checked against.
    pub manifest: &'a CommitmentManifest,
    /// Round 1.
    pub main: &'a MainTables,
    /// Round 2.
    pub bridge: &'a BridgeGroupTables,
}

impl<'a> SourceTables<'a> {
    /// The polynomials of `role`, in manifest order.
    #[must_use]
    pub fn group(&self, role: GroupRole) -> &'a [Vec<i8>] {
        match role {
            GroupRole::Main => self.main.columns(),
            other => self.bridge.group(other),
        }
    }

    /// One committed polynomial.
    #[must_use]
    pub fn column(&self, column: Column) -> &'a [i8] {
        let (role, position) = self.manifest.locate(column);
        &self.group(role)[position]
    }

    /// The digit polynomials of `value`, least significant first.
    #[must_use]
    pub fn digits(&self, value: DigitValue) -> Vec<&'a [i8]> {
        (0..value.layout().len() as u8)
            .map(|digit| self.column(Column::Digit { value, digit }))
            .collect()
    }
}
