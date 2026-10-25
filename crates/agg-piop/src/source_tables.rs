//! Batch-level committed source tables (spec Sections 4.1, 4.2 and 9.2; lookup layout L16,
//! `docs/lookup.md` Sections 1 and 6).
//!
//! Mirrors `aerie-prover/src/protocol/source_tables.rs`: one holder per commitment round,
//! built from the derived witnesses and checked against the [`CommitmentManifest`]:
//!
//! * [`MainTables`], round 1: `G_main16` (`Z_lo, U_lo, V_lo, V_hi`, stored balanced as
//!   `i32` cells in one stacked lane, column `c` at cells `c 2^d_main ..`) and `G_main4`
//!   (`Z_hi, U_hi, b_0..b_5`, `i8` cells), from `z + 2^17 = z_lo + 2^16 z_hi`,
//!   `v + 2^31 = v_lo + 2^16 v_hi` with the stored values `Z_lo = z_lo - 2^15`,
//!   `Z_hi = z_hi - 2` and likewise (spec (4.1) with the L16 bases), and relation H-u's
//!   `u + 190,465 = B_ulo + 63,079 H_u` (`H_u = min(floor((u + 190,465) / 63,079), 5)`) stored
//!   as `U_lo = B_ulo - 2^15`, `U_hi = H_u - 3`. Padded rows carry the digits of zero
//!   (`Z_lo = V_lo = -2^15`, `U_lo = -31,540`, the rest 0).
//! * [`BridgeGroupTables`], round 2: the carries `kappa^(1)_t, kappa^(2)_t, kappa^(3)` as
//!   balanced 16-bit columns (`K = kappa - 2^15`, stacked: 6 lanes in 8 slots on `D_1` and
//!   `D_2`, the padding slots holding `-2^15`), the level-2 parities, the public `Y`, and
//!   the multiplicity column `m` of `T_16` over every looked-up cell with its seven balanced
//!   base-16 digits (`G_m`). Parities and carries come from `agg_bridge::parity` at the
//!   production shape; the honest prover aborts on a row weight above the cap.
//!
//! Construction validates the shape and the honest alphabet of every polynomial. This is
//! an encoder self-check (the honest builder cannot fail it), not a soundness check: the
//! PIOP proves every range (the lookup into `T_16`, the leaves of `Z_hi`, `U_hi`).

use agg_bridge::{parity, Shape};
use agg_witness::bridge::{BridgeError, BridgeMatrices, M3};
use agg_witness::derive::{derive_with_t, DeriveError};
use agg_witness::instance::PublicInstance;
use agg_witness::native::{VecK, VecL};
use agg_witness::padding::{main_index, Geometry, GeometryError};
use agg_witness::params::{K, L, N, POSITIONS, W1_BITS};
use rayon::prelude::*;

use crate::kernel::{Lane, Source};
use crate::lookup::{multiplicities, multiplicity_digits, LookupError, LookupStack, StackShape, Table};
use crate::manifest::{
    stack_columns, Column, CommitmentManifest, DigitValue, GroupDescriptor, GroupRole,
    LOOKUP_SHIFT, STACK_DOMAINS,
};
use agg_witness::digits::ValueForm;
use crate::OpeningField as E;

/// A failure while building or checking the committed tables.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SourceTableError {
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
        value: i64,
    },
    /// The public `Y` has the wrong length or a non-bit entry.
    PublicY,
    /// Witness derivation failed (record index and reason).
    Derive(usize, DeriveError),
    /// The batch geometry is invalid.
    Geometry(GeometryError),
    /// A value is outside the L16 range of its decomposition (honest abort).
    Encode(&'static str),
    /// The bridge matrices are malformed or a row weight exceeds its cap (honest abort).
    Bridge(BridgeError),
    /// A looked-up cell is outside `T_16` (honest abort) or the multiplicities overflow.
    Lookup(LookupError),
}

impl std::fmt::Display for SourceTableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for SourceTableError {}

/// The cells of one committed column: `i8` (width-4 columns) or `i32` (the 16-bit lookup
/// columns, stored balanced).
#[derive(Clone, Copy, Debug)]
pub enum Cells<'a> {
    /// `i8` cells.
    Narrow(&'a [i8]),
    /// `i32` cells.
    Wide(&'a [i32]),
}

impl<'a> Cells<'a> {
    /// Number of cells.
    #[must_use]
    pub const fn len(&self) -> usize {
        match self {
            Self::Narrow(cells) => cells.len(),
            Self::Wide(cells) => cells.len(),
        }
    }

    /// True when empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The value of cell `index`.
    #[must_use]
    pub fn value(&self, index: usize) -> i64 {
        match self {
            Self::Narrow(cells) => i64::from(cells[index]),
            Self::Wide(cells) => i64::from(cells[index]),
        }
    }

    /// Every value, widened.
    #[must_use]
    pub fn to_i64(&self) -> Vec<i64> {
        match self {
            Self::Narrow(cells) => cells.iter().map(|&c| i64::from(c)).collect(),
            Self::Wide(cells) => cells.iter().map(|&c| i64::from(c)).collect(),
        }
    }

    /// The kernel lane over the cells (coded byte lane or `Ints` lane).
    #[must_use]
    pub fn lane(&self) -> Lane<'a> {
        match *self {
            Self::Narrow(cells) => Lane::column(cells),
            Self::Wide(cells) => Lane::Source(Source::Ints(cells)),
        }
    }

    /// The multilinear extension at `point` (least significant variable first).
    #[must_use]
    pub fn evaluate(&self, point: &[E]) -> E {
        match self {
            Self::Narrow(cells) => agg_pcs_config::commit::evaluate_signed_table(cells, point),
            Self::Wide(cells) => agg_pcs_config::commit::evaluate_signed_table(cells, point),
        }
    }

    /// The `i8` cells.
    ///
    /// # Panics
    /// On a wide column.
    #[must_use]
    pub fn narrow(&self) -> &'a [i8] {
        match *self {
            Self::Narrow(cells) => cells,
            Self::Wide(_) => panic!("a wide column"),
        }
    }

    /// The `i32` cells.
    ///
    /// # Panics
    /// On a narrow column.
    #[must_use]
    pub fn wide(&self) -> &'a [i32] {
        match *self {
            Self::Wide(cells) => cells,
            Self::Narrow(_) => panic!("a narrow column"),
        }
    }
}

/// Shape and alphabet self-check of one group against its descriptor.
pub fn validate_group(descriptor: &GroupDescriptor, columns: &[Cells<'_>]) -> Result<(), SourceTableError> {
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
        let bad = (0..cells.len())
            .into_par_iter()
            .position_first(|cell| !(min..=max).contains(&cells.value(cell)));
        if let Some(cell) = bad {
            return Err(SourceTableError::Alphabet {
                group,
                column: position,
                cell,
                value: cells.value(cell),
            });
        }
    }
    Ok(())
}

/// The L16 values of one record: `z, u, v` and the six bit lanes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordValues {
    /// `z[s][j]`.
    pub z: VecL,
    /// `u[rho][j]`.
    pub u: VecK,
    /// `v[rho][j]`.
    pub v: VecK,
    /// `w_1'[rho][j]` (bit `t` is `b_t`).
    pub w1: [[u8; N]; K],
}

impl RecordValues {
    /// The canonical all-zero witness of a padded row (spec Section 4.2).
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            z: [[0; N]; L],
            u: [[0; N]; K],
            v: [[0; N]; K],
            w1: [[0; N]; K],
        }
    }
}

/// One record's encoded cells: `4 * 1024` wide cells (`Z_lo, U_lo, V_lo, V_hi`) and
/// `8 * 1024` narrow cells (`Z_hi, U_hi, b_0..b_5`).
struct RecordBlock {
    wide: Vec<i32>,
    narrow: Vec<i8>,
}

fn split_into(value: i64, layout: ValueForm, what: &'static str) -> Result<[i64; 2], SourceTableError> {
    let digits = layout.decompose(value);
    if !layout.in_alphabets(&digits) {
        return Err(SourceTableError::Encode(what));
    }
    Ok([digits[0], digits[1]])
}

fn encode_record(values: &RecordValues) -> Result<RecordBlock, SourceTableError> {
    let mut wide = vec![0_i32; 4 * POSITIONS];
    let mut narrow = vec![0_i8; 8 * POSITIONS];
    for row in 0..K {
        for j in 0..N {
            let pos = j + N * row;
            // z is indexed by (j, s); u, v and the bits by (j, rho): both are j + 256 * row.
            let [z_lo, z_hi] = split_into(values.z[row][j], DigitValue::Z.layout(), "z outside [-2^17, 2^17)")?;
            let [u_lo, u_hi] = split_into(values.u[row][j], DigitValue::U.layout(), "u outside [-190,465, 190,465]")?;
            let [v_lo, v_hi] = split_into(values.v[row][j], DigitValue::V.layout(), "v outside [-2^31, 2^31)")?;
            for (c, value) in [z_lo, u_lo, v_lo, v_hi].into_iter().enumerate() {
                wide[c * POSITIONS + pos] = i32::try_from(value).expect("16-bit digit");
            }
            narrow[pos] = i8::try_from(z_hi).expect("base-4 digit");
            narrow[POSITIONS + pos] = i8::try_from(u_hi).expect("base-6 digit");
            let w1 = values.w1[row][j];
            if w1 >= 64 {
                return Err(SourceTableError::Encode("w1' above six bits"));
            }
            for t in 0..W1_BITS {
                narrow[(2 + t) * POSITIONS + pos] = ((w1 >> t) & 1) as i8;
            }
        }
    }
    Ok(RecordBlock { wide, narrow })
}

/// Round-1 tables: `G_main16` (one stacked `i32` lane) and `G_main4` (`i8` columns).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MainTables {
    geom: Geometry,
    wide: Vec<i32>,
    narrow: Vec<Vec<i8>>,
}

impl MainTables {
    /// Encodes the record values (live records in order; padded rows get the zero
    /// witness) and checks the tables against the manifest.
    pub fn from_values(manifest: &CommitmentManifest, values: &[RecordValues]) -> Result<Self, SourceTableError> {
        let blocks = values
            .par_iter()
            .map(encode_record)
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_blocks(manifest, &blocks)
    }

    fn from_blocks(manifest: &CommitmentManifest, blocks: &[RecordBlock]) -> Result<Self, SourceTableError> {
        let geom = Geometry::new(blocks.len()).map_err(SourceTableError::Geometry)?;
        if geom.n != manifest.records() || geom.n_star != manifest.padded_records() {
            return Err(SourceTableError::Records {
                expected: manifest.records(),
                actual: geom.n,
            });
        }
        let zero = encode_record(&RecordValues::zero()).expect("zero record encodes");
        let cells = geom.main_len();
        let mut wide = vec![0_i32; 4 * cells];
        wide.par_chunks_mut(POSITIONS).enumerate().for_each(|(chunk, out)| {
            let (c, i) = (chunk / geom.n_star, chunk % geom.n_star);
            let block = blocks.get(i).unwrap_or(&zero);
            out.copy_from_slice(&block.wide[c * POSITIONS..(c + 1) * POSITIONS]);
        });
        let narrow: Vec<Vec<i8>> = (0..8)
            .into_par_iter()
            .map(|c| {
                let mut column = vec![0_i8; cells];
                for (i, chunk) in column.chunks_mut(POSITIONS).enumerate() {
                    let block = blocks.get(i).unwrap_or(&zero);
                    chunk.copy_from_slice(&block.narrow[c * POSITIONS..(c + 1) * POSITIONS]);
                }
                column
            })
            .collect();
        let tables = Self { geom, wide, narrow };
        tables.validate(manifest)?;
        Ok(tables)
    }

    /// Derives every live witness (strict FIPS 204 verification) and builds both main
    /// groups.
    pub fn build(
        manifest: &CommitmentManifest,
        instance: &PublicInstance,
        signatures: &[&[u8]],
    ) -> Result<Self, SourceTableError> {
        if signatures.len() != instance.records.len() {
            return Err(SourceTableError::Records {
                expected: instance.records.len(),
                actual: signatures.len(),
            });
        }
        let blocks = instance
            .records
            .par_iter()
            .zip(signatures.par_iter())
            .enumerate()
            .map(|(i, (record, signature))| {
                // The instance computed T for this c~ already.
                let witness = derive_with_t(&record.key, &record.mu, signature, &record.c_tilde, &record.t)
                    .map_err(|e| SourceTableError::Derive(i, e))?;
                encode_record(&RecordValues {
                    z: witness.z,
                    u: witness.u,
                    v: witness.v,
                    w1: witness.w1,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_blocks(manifest, &blocks)
    }

    /// The shape and alphabet self-check of both groups.
    pub fn validate(&self, manifest: &CommitmentManifest) -> Result<(), SourceTableError> {
        validate_group(manifest.group(GroupRole::Main16), &self.wide_columns())?;
        let narrow: Vec<Cells<'_>> = self.narrow.iter().map(|c| Cells::Narrow(c)).collect();
        validate_group(manifest.group(GroupRole::Main), &narrow)
    }

    /// Batch geometry.
    #[must_use]
    pub const fn geometry(&self) -> Geometry {
        self.geom
    }

    /// The stacked `G_main16` lane `Z_lo | U_lo | V_lo | V_hi` (`4 * 2^d_main` cells).
    #[must_use]
    pub fn wide(&self) -> &[i32] {
        &self.wide
    }

    /// The four `G_main16` columns in manifest order.
    #[must_use]
    pub fn wide_columns(&self) -> Vec<Cells<'_>> {
        self.wide
            .chunks_exact(self.geom.main_len())
            .map(Cells::Wide)
            .collect()
    }

    /// The eight `G_main4` polynomials in manifest order (`2^d_main` cells each).
    #[must_use]
    pub fn columns(&self) -> &[Vec<i8>] {
        &self.narrow
    }

    /// One polynomial of either main group.
    #[must_use]
    pub fn column(&self, manifest: &CommitmentManifest, column: Column) -> Cells<'_> {
        let (role, position) = manifest.locate(column);
        match role {
            GroupRole::Main16 => {
                let cells = self.geom.main_len();
                Cells::Wide(&self.wide[position * cells..(position + 1) * cells])
            }
            GroupRole::Main => Cells::Narrow(&self.narrow[position]),
            _ => panic!("{} is not a main-domain column", column.name()),
        }
    }

    /// The six bit lanes `b_t`.
    #[must_use]
    pub fn bit_lanes(&self, manifest: &CommitmentManifest) -> [&[i8]; W1_BITS] {
        std::array::from_fn(|t| self.column(manifest, Column::W1Bit(t as u8)).narrow())
    }

    /// The values `z, u, v` of one record recomposed by (4.1) (any cell values).
    #[must_use]
    pub fn record_values(&self, manifest: &CommitmentManifest, i: usize) -> (VecL, VecK, VecK) {
        let at = |column: Column, pos: usize| self.column(manifest, column).value(i * POSITIONS + pos);
        let compose = |value: DigitValue, lo: Column, hi: Column, pos: usize| {
            value.layout().compose(&[at(lo, pos), at(hi, pos)])
        };
        let mut z = [[0; N]; L];
        let mut u = [[0; N]; K];
        let mut v = [[0; N]; K];
        for row in 0..K {
            for j in 0..N {
                let pos = j + N * row;
                z[row][j] = compose(DigitValue::Z, Column::Z_LO, Column::Z_HI, pos);
                u[row][j] = compose(DigitValue::U, Column::U_LO, Column::U_HI, pos);
                v[row][j] = compose(DigitValue::V, Column::V_LO, Column::V_HI, pos);
            }
        }
        (z, u, v)
    }

    /// Mutable `G_main4` columns for adversarial tests (the alphabet is not rechecked).
    #[doc(hidden)]
    pub fn columns_mut(&mut self) -> &mut [Vec<i8>] {
        &mut self.narrow
    }

    /// Mutable stacked `G_main16` lane for adversarial tests.
    #[doc(hidden)]
    pub fn wide_mut(&mut self) -> &mut [i32] {
        &mut self.wide
    }

    /// Mutable cell of a wide main column (adversarial tests).
    #[doc(hidden)]
    pub fn wide_cell_mut(&mut self, manifest: &CommitmentManifest, column: Column, x: usize) -> &mut i32 {
        let (role, position) = manifest.locate(column);
        assert_eq!(role, GroupRole::Main16, "a G_main16 column");
        &mut self.wide[position * self.geom.main_len() + x]
    }
}

/// `x = j + 256 rho + 1024 i` (re-exported for tests and tools).
#[must_use]
pub fn main_cell(j: usize, rho: usize, i: usize) -> usize {
    main_index(j, rho, i)
}

/// The stacked carries of one level: `6` columns in `8` slots (levels 1, 2) or one column
/// (level 3), each slot of `2^d_t` cells, the padding slots holding `-2^15`.
fn stack_carries(columns: &[Vec<i64>], slots: usize) -> Result<Vec<i32>, SourceTableError> {
    let cells = columns.first().map_or(0, Vec::len);
    let mut stacked = vec![-LOOKUP_SHIFT; slots * cells];
    for (slot, column) in stacked.chunks_mut(cells).zip(columns) {
        for (out, &kappa) in slot.iter_mut().zip(column) {
            if !(0..1 << 16).contains(&kappa) {
                return Err(SourceTableError::Encode("carry outside [0, 2^16)"));
            }
            *out = i32::try_from(kappa).expect("16 bits") - LOOKUP_SHIFT;
        }
    }
    Ok(stacked)
}

/// Round-2 tables: the carries of `G_1, G_2, G_3` (stacked, balanced), the level-2
/// parities, the public bridge vectors `Y`, and the multiplicity column `m` of `T_16`
/// with its digits (`G_m`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BridgeGroupTables {
    level1: Vec<i32>,
    parities: Vec<Vec<i8>>,
    level2: Vec<i32>,
    level3: Vec<i32>,
    y: Vec<u8>,
    multiplicity: Vec<i32>,
    digits: Vec<Vec<i8>>,
}

impl BridgeGroupTables {
    /// The honest round-2 tables for the committed bits of `main` and the matrices derived
    /// from the bridge seed (spec Sections 9.1, 9.2), with the multiplicities of every
    /// looked-up cell of both rounds.
    pub fn build(
        manifest: &CommitmentManifest,
        main: &MainTables,
        matrices: &BridgeMatrices,
    ) -> Result<Self, SourceTableError> {
        let shape = Shape::PRODUCTION;
        let witness = parity::compute(&shape, &main.bit_lanes(manifest), matrices)
            .map_err(SourceTableError::Bridge)?;
        let narrow_bits = |table: &Vec<i64>| -> Result<Vec<i8>, SourceTableError> {
            table
                .iter()
                .map(|&bit| i8::try_from(bit).map_err(|_| SourceTableError::Encode("parity")))
                .collect()
        };
        let mut tables = Self {
            level1: stack_carries(&witness.kappa1, 8)?,
            parities: witness.y2.iter().map(narrow_bits).collect::<Result<_, _>>()?,
            level2: stack_carries(&witness.kappa2, 8)?,
            level3: stack_carries(std::slice::from_ref(&witness.kappa3), 1)?,
            y: witness.y_bits(),
            multiplicity: Vec::new(),
            digits: Vec::new(),
        };
        tables.recount(manifest, main)?;
        tables.validate(manifest)?;
        Ok(tables)
    }

    /// Recompute the multiplicities `m` and their digits from the current cells of both
    /// rounds (`lookup::multiplicities`); an out-of-table cell is an honest abort.
    pub fn recount(&mut self, manifest: &CommitmentManifest, main: &MainTables) -> Result<(), SourceTableError> {
        let m = {
            let stacks = lookup_stacks(manifest, main, self).map_err(SourceTableError::Lookup)?;
            multiplicities(&stacks, &[Table::T16]).map_err(SourceTableError::Lookup)?
        };
        self.set_multiplicity(m.into_iter().next().expect("one table"));
        Ok(())
    }

    /// Set `m` (and its digits) directly (adversarial tests and [`Self::recount`]).
    ///
    /// # Panics
    /// If a count does not fit seven balanced base-16 digits.
    #[doc(hidden)]
    pub fn set_multiplicity(&mut self, m: Vec<i32>) {
        self.digits = multiplicity_digits(&m);
        self.multiplicity = m;
    }

    /// The shape and alphabet self-check of the four round-2 groups.
    pub fn validate(&self, manifest: &CommitmentManifest) -> Result<(), SourceTableError> {
        for role in [GroupRole::Level1, GroupRole::Level2, GroupRole::Level3, GroupRole::Mult] {
            validate_group(manifest.group(role), &self.group(manifest, role))?;
        }
        if self.y.len() != manifest.group(GroupRole::Level3).logical_cells() || self.y.iter().any(|&bit| bit > 1) {
            return Err(SourceTableError::PublicY);
        }
        Ok(())
    }

    /// The polynomials of a round-2 group, in manifest order.
    #[must_use]
    pub fn group(&self, manifest: &CommitmentManifest, role: GroupRole) -> Vec<Cells<'_>> {
        fn carries(stacked: &[i32], cells: usize, count: usize) -> Vec<Cells<'_>> {
            stacked.chunks_exact(cells).take(count).map(Cells::Wide).collect()
        }
        let cells = manifest.group(role).logical_cells();
        match role {
            GroupRole::Level1 => carries(&self.level1, cells, W1_BITS),
            GroupRole::Level2 => self
                .parities
                .iter()
                .map(|p| Cells::Narrow(p))
                .chain(carries(&self.level2, cells, W1_BITS))
                .collect(),
            GroupRole::Level3 => carries(&self.level3, cells, 1),
            GroupRole::Mult => self.digits.iter().map(|d| Cells::Narrow(d)).collect(),
            GroupRole::Main16 | GroupRole::Main => panic!("a round-1 group"),
        }
    }

    /// The stacked carry lane of level `1..=3` (lookup stack `level`).
    #[must_use]
    pub fn carries(&self, level: usize) -> &[i32] {
        match level {
            1 => &self.level1,
            2 => &self.level2,
            3 => &self.level3,
            _ => panic!("levels 1 to 3"),
        }
    }

    /// Mutable stacked carry lane (adversarial tests; `m` is not recounted).
    #[doc(hidden)]
    pub fn carries_mut(&mut self, level: usize) -> &mut [i32] {
        match level {
            1 => &mut self.level1,
            2 => &mut self.level2,
            3 => &mut self.level3,
            _ => panic!("levels 1 to 3"),
        }
    }

    /// The level-2 parities `y^(2)_t`.
    #[must_use]
    pub fn parities(&self) -> &[Vec<i8>] {
        &self.parities
    }

    /// Mutable parities (adversarial tests).
    #[doc(hidden)]
    pub fn parities_mut(&mut self) -> &mut [Vec<i8>] {
        &mut self.parities
    }

    /// The multiplicity column `m` of `T_16` (`2^16` counts).
    #[must_use]
    pub fn multiplicity(&self) -> &[i32] {
        &self.multiplicity
    }

    /// The public `Y` on `D_3` (`Y_sigma[rho_3]` at `rho_3 + 2^12 sigma`).
    #[must_use]
    pub fn public_y(&self) -> &[u8] {
        &self.y
    }

    /// Mutable public `Y` (adversarial tests).
    #[doc(hidden)]
    pub fn public_y_mut(&mut self) -> &mut [u8] {
        &mut self.y
    }

    /// `Y_sigma`, the 4,096 bits of segment `sigma` (event-6 payload, 512 bytes packed).
    #[must_use]
    pub fn segment_y(&self, sigma: usize) -> &[u8] {
        &self.y[sigma * M3..(sigma + 1) * M3]
    }

    /// The carries `kappa^(level)` as integers (`kappa = K + 2^15`), per lane.
    #[must_use]
    pub fn carry_values(&self, manifest: &CommitmentManifest, level: usize) -> Vec<Vec<i64>> {
        let role = STACK_DOMAINS[level];
        let cells = manifest.group(role).logical_cells();
        let count = if level == 3 { 1 } else { W1_BITS };
        self.carries(level)
            .chunks_exact(cells)
            .take(count)
            .map(|slot| slot.iter().map(|&k| i64::from(k + LOOKUP_SHIFT)).collect())
            .collect()
    }
}

/// The four lookup stacks of the batch (`docs/lookup.md` Section 1), borrowed from the
/// tables: `[Z_lo | U_lo | V_lo | V_hi]` and the stacked carries of levels 1 to 3.
pub fn lookup_stacks<'a>(
    manifest: &CommitmentManifest,
    main: &'a MainTables,
    bridge: &'a BridgeGroupTables,
) -> Result<Vec<LookupStack<'a>>, LookupError> {
    let shape = manifest.lookup_shape();
    let lanes: [&[i32]; 4] = [main.wide(), bridge.carries(1), bridge.carries(2), bridge.carries(3)];
    shape
        .stacks
        .iter()
        .zip(lanes)
        .map(|(&stack, lane): (&StackShape, &[i32])| LookupStack::borrowed(stack, lane))
        .collect()
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
    pub fn group(&self, role: GroupRole) -> Vec<Cells<'a>> {
        match role {
            GroupRole::Main16 => self.main.wide_columns(),
            GroupRole::Main => self.main.columns().iter().map(|c| Cells::Narrow(c)).collect(),
            other => self.bridge.group(self.manifest, other),
        }
    }

    /// One committed polynomial.
    #[must_use]
    pub fn column(&self, column: Column) -> Cells<'a> {
        let (role, position) = self.manifest.locate(column);
        self.group(role)[position]
    }

    /// The two digit polynomials of `z`, `u` or `v` (`lo`, `hi`).
    #[must_use]
    pub fn digits(&self, value: DigitValue) -> [Cells<'a>; 2] {
        [0, 1].map(|digit| self.column(Column::Digit { value, digit }))
    }

    /// The four lookup stacks.
    pub fn lookup_stacks(&self) -> Result<Vec<LookupStack<'a>>, LookupError> {
        lookup_stacks(self.manifest, self.main, self.bridge)
    }

    /// The looked-up columns of stack `stack`, in slot order.
    #[must_use]
    pub fn stack_cells(&self, stack: usize) -> Vec<Cells<'a>> {
        stack_columns(stack).into_iter().map(|c| self.column(c)).collect()
    }
}
