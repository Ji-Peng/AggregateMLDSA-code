//! Batch-level committed source tables (`docs/lookup.md`).
//!
//! Mirrors `aerie-prover/src/protocol/source_tables.rs`: one holder per commitment round,
//! built from the derived witnesses and checked against the [`CommitmentManifest`]:
//!
//! * [`MainTables`], round 1: `G_main16` (`Z_lo, U_lo, V_lo, V_hi`, stored balanced as
//!   `i32` cells in one stacked lane, column `c` at cells `c 2^d_main ..`) and `G_main4`
//!   (`Z_hi, U_hi, b_0..b_5`, `i8` cells), from `z + 2^17 = z_lo + 2^16 z_hi`,
//!   `v + 2^31 = v_lo + 2^16 v_hi` with the stored values `Z_lo = z_lo - 2^15`,
//!   `Z_hi = z_hi - 2` and likewise, and relation H-u's
//!   `u + 190,465 = B_ulo + 63,079 H_u` (`H_u = min(floor((u + 190,465) / 63,079), 5)`) stored
//!   as `U_lo = B_ulo - 2^15`, `U_hi = H_u - 3`. Padded rows carry the digits of zero
//!   (`Z_lo = V_lo = -2^15`, `U_lo = -31,540`, the rest 0). Under relation H-exact `z` is
//!   `z + 130,993 = B_zlo + 2^16 B_zhi + 130,915 a_z` (`a_z = [z >= 79]`, `docs/h-exact.md`)
//!   stored as `bz_lo = B_zlo - 2^15` (the slot of `Z_lo`), the bit `bz_hi` (the slot of
//!   `Z_hi`) and the bit `a_z` (the ninth `G_main4` column); the digits of zero are
//!   `(bz_lo, bz_hi, a_z) = (32,689, 1, 0)`. Under relation H-FIPS `z` is H-exact's, and
//!   each coefficient's canonical witness `agg_witness::hfips::honest_coeff(w', h*)`, with
//!   `w' = 2 gamma_2 w_1' + u mod q` and `h* = [HighBits(w') != w_1']` (the signature's own
//!   hint for an honest record), gives `l_lo = L_lo - 2^15`, `l_hi` (the slots of `U_lo`,
//!   `U_hi`), the sign bit `s = (1 + sigma)/2`, `delta` and `e` (after `a_z`); each record's
//!   hint slack `u_h = 80 - sum delta^2` is kept for `G_2`. Padded rows: `(s, L, delta, e) =
//!   (0, gamma_2 - 1, 0, 0)`, `u_h = 80`.
//! * [`BridgeGroupTables`], round 2: the carries `kappa^(1)_t, kappa^(2)_t, kappa^(3)` as
//!   balanced 16-bit columns (`K = kappa - 2^15`, stacked: 6 lanes in 8 slots on `D_1` and
//!   `D_2`, the padding slots holding `-2^15`), the level-2 parities, the public `Y`, and
//!   the multiplicity column `m` of `T_16` over every looked-up cell with its seven balanced
//!   base-16 digits (`G_m`). Parities and carries come from `agg_bridge::parity` at the
//!   production shape; the honest prover aborts on a row weight above the cap. Under H-FIPS
//!   the hint slack `u_h` takes slot 6 of the level-2 stack (and is `G_2`'s thirteenth
//!   polynomial): `u_h(i)` at `D_2` index `i 2^(d_2 - d_sig)`, zero elsewhere.
//!
//! Construction validates the shape and the honest alphabet of every polynomial. This is
//! an encoder self-check (the honest builder cannot fail it), not a soundness check: the
//! PIOP proves every range (the lookup into `T_16`, the leaves of `Z_hi`, `U_hi`, the
//! zero-checks of `bz_hi`, `a_z`).

use agg_bridge::{parity, Shape};
use agg_witness::bridge::{BridgeError, BridgeMatrices, M3};
use agg_mldsa_simd::{Engine, HuCols};
use agg_witness::derive::{derive_with_t, DeriveError};
use agg_witness::fast::{derive_tile_with, FastWitness, TILE};
use agg_witness::hfips::honest_coeff;
use agg_witness::instance::{PublicInstance, PublicRecord};
use agg_witness::native::{high_bits, mod_q, use_hint, VecK, VecL};
use agg_witness::padding::{main_index, Geometry, GeometryError};
use agg_witness::params::{GAMMA2, K, L, N, OMEGA, POSITIONS, W1_BITS};
use rayon::prelude::*;

use crate::kernel::{Lane, Source};
use crate::lookup::{multiplicities, multiplicity_digits, LookupError, LookupStack, StackShape, Table};
use crate::manifest::{
    value_columns, Column, CommitmentManifest, DigitValue, FipsColumn, GroupDescriptor, GroupRole,
    Relation, LOOKUP_SHIFT, STACK_DOMAINS,
};
use std::mem::MaybeUninit;
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
    /// A value is outside the range of its digit decomposition (honest abort).
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

/// Cells per task of the alphabet scan in [`validate_group`].
const VALIDATE_CHUNK: usize = 1 << 16;

/// True when every cell of `cells[range]` lies in `[min, max]` (a branch-free min/max scan
/// that vectorises; the bad cell itself is located afterwards, only on failure).
fn chunk_in_range(cells: &Cells<'_>, range: std::ops::Range<usize>, min: i64, max: i64) -> bool {
    fn scan<T: Copy + Ord + Into<i64>>(cells: &[T], min: i64, max: i64) -> bool {
        let Some(&first) = cells.first() else { return true };
        let (lo, hi) = cells.iter().fold((first, first), |(lo, hi), &c| (lo.min(c), hi.max(c)));
        min <= lo.into() && hi.into() <= max
    }
    match *cells {
        Cells::Narrow(c) => scan(&c[range], min, max),
        Cells::Wide(c) => scan(&c[range], min, max),
    }
}

/// Shape and alphabet self-check of one group against its descriptor. The error is the one a
/// column-by-column, cell-by-cell walk meets first; the scan itself runs over all
/// (column, chunk) pairs at once, so the pool is not drained after every column.
pub fn validate_group(descriptor: &GroupDescriptor, columns: &[Cells<'_>]) -> Result<(), SourceTableError> {
    let group = descriptor.role;
    if columns.len() != descriptor.polynomial_count() {
        return Err(SourceTableError::ColumnCount {
            group,
            expected: descriptor.polynomial_count(),
            actual: columns.len(),
        });
    }
    let expected = descriptor.logical_cells();
    // The walk stops at the first column of the wrong length; only the columns before it
    // are scanned.
    let checked = columns.iter().position(|cells| cells.len() != expected).unwrap_or(columns.len());
    let bounds: Vec<(i64, i64)> = descriptor.columns[..checked]
        .iter()
        .map(|column| (column.alphabet().min(), column.alphabet().max()))
        .collect();
    let chunks = expected.div_ceil(VALIDATE_CHUNK);
    let bad_column = (0..checked * chunks)
        .into_par_iter()
        .filter(|&task| {
            let (position, chunk) = (task / chunks, task % chunks);
            let start = chunk * VALIDATE_CHUNK;
            let (min, max) = bounds[position];
            !chunk_in_range(&columns[position], start..expected.min(start + VALIDATE_CHUNK), min, max)
        })
        .map(|task| task / chunks)
        .min();
    if let Some(position) = bad_column {
        let cells = &columns[position];
        let (min, max) = bounds[position];
        let cell = (0..cells.len())
            .into_par_iter()
            .position_first(|cell| !(min..=max).contains(&cells.value(cell)))
            .expect("the scan found a bad cell in this column");
        return Err(SourceTableError::Alphabet {
            group,
            column: position,
            cell,
            value: cells.value(cell),
        });
    }
    if checked < columns.len() {
        return Err(SourceTableError::ColumnLength {
            group,
            column: checked,
            expected,
            actual: columns[checked].len(),
        });
    }
    Ok(())
}

/// The values of one record: `z, u, v` and the six bit lanes.
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
    /// The canonical all-zero witness of a padded row.
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

/// The H-FIPS witness of one coefficient from `(u, w_1')` (`docs/h-exact.md`):
/// `w' = 2 gamma_2 w_1' + u mod q`, `h* = [HighBits(w') != w_1']`, the canonical
/// `honest_coeff(w', h*)`; `None` when `w_1'` is not reachable from `w'` by any hint bit or
/// the form's `u` differs (an input outside the honest relation).
#[must_use]
pub fn fips_coefficient(u: i64, w1: i64) -> Option<agg_witness::hfips::Coeff> {
    let wq = mod_q(2 * GAMMA2 * w1 + u);
    let hint = high_bits(wq) != w1;
    if use_hint(hint, wq) != w1 {
        return None;
    }
    let c = honest_coeff(wq, hint);
    (c.u() == u).then_some(c)
}

/// The final column holding a digit of a main group.
#[derive(Clone, Copy, Debug)]
enum Slot {
    Wide(usize),
    Narrow(usize),
}

/// Most digits of one committed value (the encoder decomposes into a stack buffer).
const MAX_DIGITS: usize = 8;

/// The relation's encoder: the value forms of `z, u, v` (H-FIPS: `z, L, v`) with the slot of
/// each digit, the slots of the bits and, under H-FIPS, of `s, delta, e` (resolved once per
/// batch from the manifest).
struct Encoder {
    values: [(ValueForm, Vec<Slot>, &'static str); 3],
    bits: [usize; W1_BITS],
    fips: Option<[usize; 3]>,
    narrow: usize,
}

impl Encoder {
    fn new(manifest: &CommitmentManifest) -> Self {
        let slot = |column: Column| match manifest.locate(column) {
            (GroupRole::Main16, position) => Slot::Wide(position),
            (GroupRole::Main, position) => Slot::Narrow(position),
            (role, _) => panic!("{} is in {role:?}, not a main group", column.name()),
        };
        let value = |value: DigitValue, what: &'static str| {
            (value.layout(), value_columns(value).into_iter().map(slot).collect(), what)
        };
        let z_range = match manifest.z_value() {
            DigitValue::Zx => "z outside [-130,993, 130,993]",
            _ => "z outside [-2^17, 2^17)",
        };
        let narrow_slot = |column: Column| match slot(column) {
            Slot::Narrow(position) => position,
            Slot::Wide(_) => unreachable!("{} is narrow", column.name()),
        };
        let u_what = if manifest.is_fips() {
            "L outside [0, gamma_2)"
        } else {
            "u outside [-190,465, 190,465]"
        };
        let values = [
            value(manifest.z_value(), z_range),
            value(manifest.u_slot_value(), u_what),
            value(DigitValue::V, "v outside [-2^31, 2^31)"),
        ];
        assert!(values.iter().all(|(form, _, _)| form.len() <= MAX_DIGITS), "at most {MAX_DIGITS} digits");
        Self {
            values,
            bits: std::array::from_fn(|t| narrow_slot(Column::W1Bit(t as u8))),
            fips: manifest
                .is_fips()
                .then(|| FipsColumn::ALL.map(|c| narrow_slot(Column::Fips(c)))),
            narrow: manifest.group(GroupRole::Main).polynomial_count(),
        }
    }

    fn encode(
        &self,
        values: &RecordValues,
        wide: &mut [&mut [std::mem::MaybeUninit<i32>]],
        narrow: &mut [&mut [std::mem::MaybeUninit<i8>]],
    ) -> Result<i64, SourceTableError> {
        let mut weight = 0_i64;
        for row in 0..K {
            for j in 0..N {
                let pos = j + N * row;
                // H-FIPS: the coefficient's witness; its magnitude L takes the slot of u.
                let middle = match self.fips {
                    None => values.u[row][j],
                    Some(slots) => {
                        let c = fips_coefficient(values.u[row][j], i64::from(values.w1[row][j]))
                            .ok_or(SourceTableError::Encode("w1' not reachable from w' by UseHint"))?;
                        let cells = [(1 + c.sigma) / 2, c.delta, c.e];
                        for (&slot, cell) in slots.iter().zip(cells) {
                            narrow[slot][pos].write(cell as i8);
                        }
                        weight += c.weight();
                        c.l()
                    }
                };
                // z is indexed by (j, s); u, v and the bits by (j, rho): both are j + 256 * row.
                let inputs = [values.z[row][j], middle, values.v[row][j]];
                for ((form, slots, what), x) in self.values.iter().zip(inputs) {
                    let mut buffer = [0_i64; MAX_DIGITS];
                    let digits = &mut buffer[..form.len()];
                    form.decompose_into(x, digits);
                    if !form.in_alphabets(digits) {
                        return Err(SourceTableError::Encode(what));
                    }
                    for (&slot, &digit) in slots.iter().zip(&*digits) {
                        match slot {
                            Slot::Wide(c) => {
                                wide[c][pos].write(i32::try_from(digit).expect("16-bit digit"));
                            }
                            Slot::Narrow(c) => {
                                narrow[c][pos].write(i8::try_from(digit).expect("small digit"));
                            }
                        }
                    }
                }
                let w1 = values.w1[row][j];
                if w1 >= 64 {
                    return Err(SourceTableError::Encode("w1' above six bits"));
                }
                for (t, &c) in self.bits.iter().enumerate() {
                    narrow[c][pos].write(((w1 >> t) & 1) as i8);
                }
            }
        }
        let hint_slack = OMEGA as i64 - weight;
        if hint_slack < 0 {
            return Err(SourceTableError::Encode("hint weight above omega"));
        }
        Ok(hint_slack)
    }
}

/// The final-column views of one record: the `G_main16` columns, then the `G_main4` columns
/// (manifest order, `POSITIONS` cells each).
type RecordViews<'a> = (Vec<&'a mut [MaybeUninit<i32>]>, Vec<&'a mut [MaybeUninit<i8>]>);

/// Relation H-u's manifest slots in the column order of the engine's digit kernel:
/// `Z_lo, U_lo, V_lo, V_hi` (wide) and `Z_hi, U_hi, b_0..b_5` (narrow).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HuSlots {
    wide: [usize; 4],
    narrow: [usize; 2 + W1_BITS],
}

impl HuSlots {
    fn new(manifest: &CommitmentManifest) -> Self {
        assert_eq!(manifest.relation(), Relation::Hu, "the digit kernel encodes relation H-u");
        let at = |column: Column, role: GroupRole| match manifest.locate(column) {
            (r, position) if r == role => position,
            (r, _) => panic!("{} is in {r:?}", column.name()),
        };
        let narrow = [Column::Z_HI, Column::U_HI]
            .into_iter()
            .chain((0..W1_BITS as u8).map(Column::W1Bit));
        Self {
            wide: [Column::Z_LO, Column::U_LO, Column::V_LO, Column::V_HI].map(|c| at(c, GroupRole::Main16)),
            narrow: std::array::from_fn({
                let slots: Vec<usize> = narrow.map(|c| at(c, GroupRole::Main)).collect();
                move |k| slots[k]
            }),
        }
    }

    /// Every cell of one live record (row `rho` at cells `256 rho ..` of each column).
    fn write(
        &self,
        engine: &Engine,
        w: &FastWitness,
        wide: &mut [&mut [MaybeUninit<i32>]],
        narrow: &mut [&mut [MaybeUninit<i8>]],
    ) {
        for (rho, (z, out)) in w.z.iter().zip(&w.rows).enumerate() {
            let cells = rho * N..(rho + 1) * N;
            let wide = wide.get_disjoint_mut(self.wide).expect("distinct wide slots");
            let narrow = narrow.get_disjoint_mut(self.narrow).expect("distinct narrow slots");
            let cols = HuCols {
                wide: wide.map(|c| &mut c[cells.clone()]),
                narrow: narrow.map(|c| &mut c[cells.clone()]),
            };
            engine.hu_columns(z, out, cols);
        }
    }
}

/// Round-1 tables: `G_main16` (one stacked `i32` lane) and `G_main4` (`i8` columns), and
/// under H-FIPS the hint slack of every record (committed in round 2, `G_2`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MainTables {
    geom: Geometry,
    wide: Vec<i32>,
    narrow: Vec<Vec<i8>>,
    hint_slack: Vec<i64>,
}

impl MainTables {
    /// Encodes the record values (live records in order; padded rows get the zero
    /// witness) and checks the tables against the manifest.
    pub fn from_values(manifest: &CommitmentManifest, values: &[RecordValues]) -> Result<Self, SourceTableError> {
        Self::encode_records(manifest, values.len(), |i| Ok(values[i].clone()))
    }

    /// Encode record by record (live records from `record`, padded rows the zero witness).
    fn encode_records(
        manifest: &CommitmentManifest,
        count: usize,
        record: impl Fn(usize) -> Result<RecordValues, SourceTableError> + Sync,
    ) -> Result<Self, SourceTableError> {
        Self::encode_tiles(manifest, count, 1, |encoder, i, views| {
            let (wide, narrow) = &mut views[0];
            let values = if i < count { record(i)? } else { RecordValues::zero() };
            Ok(vec![encoder.encode(&values, wide, narrow)?])
        })
    }

    /// Allocate final columns and give each worker the disjoint views of a tile of `tile`
    /// consecutive records (the last tile may be shorter); `fill(encoder, first, views)`
    /// writes every cell of records `first..first + views.len()` (live and padded) and
    /// returns their hint slacks.
    #[allow(unsafe_code)]
    fn encode_tiles(
        manifest: &CommitmentManifest,
        count: usize,
        tile: usize,
        fill: impl Fn(&Encoder, usize, &mut [RecordViews<'_>]) -> Result<Vec<i64>, SourceTableError> + Sync,
    ) -> Result<Self, SourceTableError> {
        let encoder = Encoder::new(manifest);
        let geom = Geometry::new(count).map_err(SourceTableError::Geometry)?;
        if geom.n != manifest.records() || geom.n_star != manifest.padded_records() {
            return Err(SourceTableError::Records { expected: manifest.records(), actual: geom.n });
        }
        let cells = geom.main_len();
        let mut wide = crate::kernel::buf::uninit::<i32>(4 * cells);
        let mut narrow: Vec<_> = (0..encoder.narrow).map(|_| crate::kernel::buf::uninit::<i8>(cells)).collect();
        let mut views: Vec<RecordViews<'_>> =
            (0..geom.n_star).map(|_| (Vec::with_capacity(4), Vec::with_capacity(encoder.narrow))).collect();
        for column in wide.chunks_exact_mut(cells) {
            for (i, chunk) in column.chunks_exact_mut(POSITIONS).enumerate() {
                views[i].0.push(chunk);
            }
        }
        for column in &mut narrow {
            for (i, chunk) in column.chunks_exact_mut(POSITIONS).enumerate() {
                views[i].1.push(chunk);
            }
        }
        let tiles = views
            .par_chunks_mut(tile)
            .enumerate()
            .map(|(t, views)| fill(&encoder, t * tile, views))
            .collect::<Result<Vec<_>, _>>()?;
        let hint_slack: Vec<i64> = tiles.into_iter().flatten().collect();
        // SAFETY: successful encoding writes every digit, bit and FIPS cell of every
        // record, including padding. On error the MaybeUninit allocations just drop.
        let wide = unsafe { wide.assume_init().into_vec() };
        let narrow = narrow.into_iter().map(|column| unsafe { column.assume_init().into_vec() }).collect();
        let tables = Self {
            geom, wide, narrow,
            hint_slack: if encoder.fips.is_some() { hint_slack } else { Vec::new() },
        };
        tables.validate(manifest)?;
        Ok(tables)
    }

    /// Derive witnesses and encode them directly into their final columns, on the
    /// process-wide SIMD engine (see [`MainTables::build_with`]).
    pub fn build(
        manifest: &CommitmentManifest,
        instance: &PublicInstance,
        signatures: &[&[u8]],
    ) -> Result<Self, SourceTableError> {
        Self::build_with(agg_mldsa_simd::engine(), manifest, instance, signatures)
    }

    /// [`MainTables::build`] on `engine`: records in tiles of [`TILE`] through the batched
    /// derivation of [`derive_tile_with`] (its errors are the legacy derivation's); under
    /// H-u the engine writes the digits of each row straight into the record's final
    /// column views, under H-exact and H-FIPS the generic encoder takes the values.
    /// Identical to [`MainTables::build_reference`].
    #[doc(hidden)]
    pub fn build_with(
        engine: &Engine,
        manifest: &CommitmentManifest,
        instance: &PublicInstance,
        signatures: &[&[u8]],
    ) -> Result<Self, SourceTableError> {
        if signatures.len() != instance.records.len() {
            return Err(SourceTableError::Records {
                expected: instance.records.len(), actual: signatures.len(),
            });
        }
        let count = instance.records.len();
        let hu = (manifest.relation() == Relation::Hu).then(|| HuSlots::new(manifest));
        Self::encode_tiles(manifest, count, TILE, |encoder, first, views| {
            // Live records of the tile; a tile past the last record is all padding.
            let live = first.min(count)..(first + views.len()).min(count);
            let records: Vec<&PublicRecord> = instance.records[live.clone()].iter().collect();
            let mut derived = vec![FastWitness::zero(); live.len()];
            derive_tile_with(engine, &records, &signatures[live], &mut derived)
                .map_err(|(i, e)| SourceTableError::Derive(first + i, e))?;
            let mut slack = Vec::with_capacity(views.len());
            for (k, (wide, narrow)) in views.iter_mut().enumerate() {
                let Some(w) = derived.get(k) else {
                    slack.push(encoder.encode(&RecordValues::zero(), wide, narrow)?);
                    continue;
                };
                match &hu {
                    Some(slots) => {
                        slots.write(engine, w, wide, narrow);
                        slack.push(OMEGA as i64);
                    }
                    None => {
                        let values = RecordValues { z: w.z_wide(), u: w.u_wide(), v: w.v_wide(), w1: w.w1() };
                        slack.push(encoder.encode(&values, wide, narrow)?);
                    }
                }
            }
            Ok(slack)
        })
    }

    /// The record-by-record build on the legacy derivation (`derive_with_t`) and the generic
    /// encoder: the test oracle of [`MainTables::build`].
    #[doc(hidden)]
    pub fn build_reference(
        manifest: &CommitmentManifest,
        instance: &PublicInstance,
        signatures: &[&[u8]],
    ) -> Result<Self, SourceTableError> {
        if signatures.len() != instance.records.len() {
            return Err(SourceTableError::Records {
                expected: instance.records.len(), actual: signatures.len(),
            });
        }
        Self::encode_records(manifest, instance.records.len(), |i| {
            let record = &instance.records[i];
            let witness = derive_with_t(&record.key, &record.mu, signatures[i], &record.c_tilde, &record.t)
                .map_err(|e| SourceTableError::Derive(i, e))?;
            Ok(RecordValues { z: witness.z, u: witness.u, v: witness.v, w1: witness.w1 })
        })
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

    /// The H-FIPS hint slack `u_h = 80 - sum delta^2` per record (`N*` entries, padded rows
    /// 80); empty under the other relations.
    #[must_use]
    pub fn hint_slack(&self) -> &[i64] {
        &self.hint_slack
    }

    /// Mutable hint slack (adversarial tests; the round-2 tables take it at build time).
    #[doc(hidden)]
    pub fn hint_slack_mut(&mut self) -> &mut [i64] {
        &mut self.hint_slack
    }

    /// The stacked `G_main16` lane `Z_lo | U_lo | V_lo | V_hi` (`bz_lo` for `Z_lo` under
    /// H-exact; `4 * 2^d_main` cells).
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

    /// The eight (H-u) or nine (H-exact) `G_main4` polynomials in manifest order (`2^d_main`
    /// cells each).
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

    /// The values `z, u, v` of one record recomposed by the relation's value forms (any cell
    /// values); under H-FIPS `u = gamma_2 s + L + 1 - gamma_2 - 2 gamma_2 delta - e` ([`fips_u`]).
    #[must_use]
    pub fn record_values(&self, manifest: &CommitmentManifest, i: usize) -> (VecL, VecK, VecK) {
        let columns = |value: DigitValue| -> Vec<Cells<'_>> {
            value_columns(value).into_iter().map(|c| self.column(manifest, c)).collect()
        };
        let forms = [manifest.z_value(), manifest.u_slot_value(), DigitValue::V]
            .map(|value| (value.layout(), columns(value)));
        let compose = |k: usize, pos: usize| {
            let (form, cells) = &forms[k];
            let digits: Vec<i64> = cells.iter().map(|c| c.value(i * POSITIONS + pos)).collect();
            form.compose(&digits)
        };
        let fips = manifest
            .is_fips()
            .then(|| FipsColumn::ALL.map(|c| self.column(manifest, Column::Fips(c))));
        let mut z = [[0; N]; L];
        let mut u = [[0; N]; K];
        let mut v = [[0; N]; K];
        for row in 0..K {
            for j in 0..N {
                let pos = j + N * row;
                z[row][j] = compose(0, pos);
                u[row][j] = match &fips {
                    None => compose(1, pos),
                    Some([s, delta, e]) => {
                        let x = i * POSITIONS + pos;
                        fips_u(compose(1, pos), s.value(x), delta.value(x), e.value(x))
                    }
                };
                v[row][j] = compose(2, pos);
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

/// H-FIPS's `u = gamma_2 s + L + 1 - gamma_2 - 2 gamma_2 delta - e` from `L`, the sign bit
/// `s = (1 + sigma)/2`, `delta` and `e` (`docs/h-exact.md`), any integers.
#[must_use]
pub const fn fips_u(l: i64, s: i64, delta: i64, e: i64) -> i64 {
    GAMMA2 * s + l + 1 - GAMMA2 - 2 * GAMMA2 * delta - e
}

/// `x = j + 256 rho + 1024 i` (re-exported for tests and tools).
#[must_use]
pub fn main_cell(j: usize, rho: usize, i: usize) -> usize {
    main_index(j, rho, i)
}

/// H-FIPS's hint slack as a `D_2` table (`docs/h-exact.md`): `u_h(i)` at index
/// `i 2^(d_2 - d_sig)` (the low `d_2 - d_sig` variables of `D_2` are position variables when
/// `d_sig < 13`), zero elsewhere.
///
/// # Panics
/// If `hint_slack` does not hold `N*` values.
#[must_use]
pub fn hint_slack_table(manifest: &CommitmentManifest, hint_slack: &[i64]) -> Vec<i64> {
    assert_eq!(hint_slack.len(), manifest.padded_records(), "one slack per signature");
    let vars = manifest.group(GroupRole::Level2).logical_vars;
    let shift = vars - manifest.d_sig();
    let mut table = vec![0_i64; 1 << vars];
    for (i, &slack) in hint_slack.iter().enumerate() {
        table[i << shift] = slack;
    }
    table
}

/// The stacked carries of one level: `6` columns in `8` slots (levels 1, 2; H-FIPS adds
/// `u_h` in slot 6 of level 2) or one column (level 3), each slot of `2^d_t` cells, the
/// padding slots holding `-2^15`.
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
#[derive(Clone, Debug)]
pub struct BridgeGroupTables {
    /// Honest level-1 parities, invalidated when the committed level-1 carries change and
    /// moved out by the honest driver; not part of the committed content (`PartialEq`).
    parity1: Option<[Vec<i8>; W1_BITS]>,
    level1: Vec<i32>,
    parities: Vec<Vec<i8>>,
    level2: Vec<i32>,
    level3: Vec<i32>,
    y: Vec<u8>,
    multiplicity: Vec<i32>,
    digits: Vec<Vec<i8>>,
}

/// Equality of the committed content: the level-1 parity cache is left out, so a table set
/// that dropped or handed over its cache equals one that keeps it.
impl PartialEq for BridgeGroupTables {
    fn eq(&self, other: &Self) -> bool {
        self.level1 == other.level1
            && self.parities == other.parities
            && self.level2 == other.level2
            && self.level3 == other.level3
            && self.y == other.y
            && self.multiplicity == other.multiplicity
            && self.digits == other.digits
    }
}

impl Eq for BridgeGroupTables {}

impl BridgeGroupTables {
    /// The honest round-2 tables for the committed bits of `main` and the matrices derived
    /// from the bridge seed, with the multiplicities of every
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
        let mut level2_columns = witness.kappa2.clone();
        if manifest.is_fips() {
            level2_columns.push(hint_slack_table(manifest, main.hint_slack()));
        }
        let mut tables = Self {
            parity1: Some(witness.y1.iter().map(narrow_bits).collect::<Result<Vec<_>, _>>()?
                .try_into().expect("six lanes")),
            level1: stack_carries(&witness.kappa1, 8)?,
            parities: witness.y2.iter().map(narrow_bits).collect::<Result<_, _>>()?,
            level2: stack_carries(&level2_columns, 8)?,
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
                .chain(carries(&self.level2, cells, manifest.stack_columns(2).len()))
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
        if level == 1 { self.parity1 = None; }
        match level {
            1 => &mut self.level1,
            2 => &mut self.level2,
            3 => &mut self.level3,
            _ => panic!("levels 1 to 3"),
        }
    }

    /// Cached parities (unit tests; the honest driver moves them out with `take_parity1`).
    #[cfg(test)]
    pub(crate) fn parity1_cached(&self) -> Option<&[Vec<i8>; W1_BITS]> {
        self.parity1.as_ref()
    }

    /// Move the cached parities out (the honest driver's single use), so they are neither
    /// copied on the PIOP branch nor kept twice through the stages.
    pub(crate) fn take_parity1(&mut self) -> Option<[Vec<i8>; W1_BITS]> {
        self.parity1.take()
    }

    /// Forget the cached level-1 parities (the tables were changed after `build`).
    pub(crate) fn drop_parity1_cache(&mut self) {
        self.parity1 = None;
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

/// The four lookup stacks of the batch (`docs/lookup.md`), borrowed from the
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

    /// The digit polynomials of `z`, `u` or `v` in digit order (`lo`, `hi`; H-exact's `z`
    /// adds `a_z`).
    #[must_use]
    pub fn digits(&self, value: DigitValue) -> Vec<Cells<'a>> {
        value_columns(value).into_iter().map(|c| self.column(c)).collect()
    }

    /// The four lookup stacks.
    pub fn lookup_stacks(&self) -> Result<Vec<LookupStack<'a>>, LookupError> {
        lookup_stacks(self.manifest, self.main, self.bridge)
    }

    /// The looked-up columns of stack `stack`, in slot order.
    #[must_use]
    pub fn stack_cells(&self, stack: usize) -> Vec<Cells<'a>> {
        self.manifest.stack_columns(stack).into_iter().map(|c| self.column(c)).collect()
    }
}

#[cfg(test)]
mod hu_slot_tests {
    use super::*;

    #[test]
    fn hu_slots_follow_the_digit_kernel_order() {
        for records in [1, 20, 1 << 10] {
            let manifest = CommitmentManifest::for_relation(records, Relation::Hu).unwrap();
            let slots = HuSlots::new(&manifest);
            let main16 = manifest.group(GroupRole::Main16);
            let main = manifest.group(GroupRole::Main);
            let wide = [Column::Z_LO, Column::U_LO, Column::V_LO, Column::V_HI];
            for (&slot, column) in slots.wide.iter().zip(wide) {
                assert_eq!(main16.columns[slot], column);
            }
            let narrow: Vec<Column> = [Column::Z_HI, Column::U_HI]
                .into_iter()
                .chain((0..W1_BITS as u8).map(Column::W1Bit))
                .collect();
            for (&slot, &column) in slots.narrow.iter().zip(&narrow) {
                assert_eq!(main.columns[slot], column);
            }
            assert_eq!(main.polynomial_count(), slots.narrow.len());
            // Today's manifest order is the kernel's.
            assert_eq!(slots.wide, [0, 1, 2, 3]);
            assert_eq!(slots.narrow, std::array::from_fn(|k| k));
        }
    }
}
