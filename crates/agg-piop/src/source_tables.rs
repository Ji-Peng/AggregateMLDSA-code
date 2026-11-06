//! Batch-level committed source tables.
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
//!   `z + 130,993 = B_zlo + 2^16 B_zhi + 130,915 a_z` (`a_z = [z >= 79]`)
//!   stored as `bz_lo = B_zlo - 2^15` (the slot of `Z_lo`), the bit `bz_hi` (the slot of
//!   `Z_hi`) and the bit `a_z` (the ninth `G_main4` column); the digits of zero are
//!   `(bz_lo, bz_hi, a_z) = (32,689, 1, 0)`. Under relation H-FIPS `z` is H-exact's, and
//!   each coefficient's canonical witness `agg_witness::hfips::honest_coeff(w', h*)`, with
//!   `w' = 2 gamma_2 w_1' + u mod q` and `h* = [HighBits(w') != w_1']` (the signature's own
//!   hint for an honest record), gives `l_lo = L_lo - 2^15`, `l_hi` (the slots of `U_lo`,
//!   `U_hi`), the sign bit `s = (1 + sigma)/2`, `delta` and `e` (after `a_z`); each record's
//!   hint slack `u_h = 80 - sum delta^2` is kept for `G_2`. Padded rows: `(s, L, delta, e) =
//!   (0, gamma_2 - 1, 0, 0)`, `u_h = 80`.
//! * [`BridgeGroupTables`], round 2: the projection digits of `v` under `Psi` (`G_k`,
//!   `projection::ProjectionTables`; the honest prover aborts on an output outside
//!   `[-2^37, 2^37)`), the zero polynomials of `G_3`, the carries `kappa^(1)_t, kappa^(2)_t,
//!   kappa^(3)` as balanced 16-bit columns (`K = kappa - 2^15`, stacked: 6 lanes in 8 slots on `D_1` and
//!   `D_2`, the padding slots holding `-2^15`), the level-2 parities, the public `Y`, and
//!   the multiplicity column `m` of `T_16` over every looked-up cell with its seven balanced
//!   base-16 digits (`G_m`). Parities and carries come from `agg_bridge::parity` at the
//!   production shape; the honest prover aborts on a row weight above the cap. Under H-FIPS
//!   the hint slack `u_h` takes slot 6 of the level-2 stack (and is `G_2`'s thirteenth
//!   polynomial): `u_h(i)` at `D_2` index `i 2^(d_2 - d_sig)`, zero elsewhere.
//!
//! Construction validates the shape and the honest alphabet of every polynomial. This is
//! an encoder self-check (the honest builder cannot fail it), not a soundness check: the
//! PIOP proves every range (the lookup into `T_16`, the leaves of `Z_hi`, `U_hi`, `K_2`, `K_3`, the
//! zero-checks of `bz_hi`, `a_z`); `V_lo`, `V_hi` have none (the projection bounds `v`).

use agg_bridge::{parity, Shape};
use agg_simd::buf::AlignedVec;
use agg_witness::bridge::{BridgeError, BridgeMatrices, M3};
use agg_mldsa_simd::{Engine, HuCols};
use agg_witness::derive::{derive_with_t, derive_with_t_for, DeriveError};
use agg_witness::fast::{derive_tile_with, FastWitness, TILE};
use agg_witness::fast_lanes::LaneWitness;
use agg_witness::hfips::{coefficient_of, honest_coeff};
use agg_witness::instance::{PublicInstance, PublicRecord};
use agg_witness::native::{high_bits, mod_q, use_hint, VecK, VecL};
use agg_witness::padding::{Geometry, GeometryError};
use agg_witness::params::{ParamSet, Set, Set44, GAMMA2, K, L, N, OMEGA, W1_BITS};
use rayon::prelude::*;

use crate::kernel::{Lane, Source};
use crate::projection::ProjectionTables;
use crate::lookup::{multiplicities, multiplicity_digits, LookupError, LookupStack, StackShape, Table};
use crate::manifest::{
    value_columns, value_lane_columns, Column, CommitmentManifest, DigitValue, FipsColumn,
    GroupDescriptor, GroupRole, Relation, SetShape, LOOKUP_SHIFT,
};
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
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
    /// A projection output `k = <Psi_o, v_i>` of record `record` is outside `[-2^37, 2^37)`
    /// (honest abort, completeness only: the aggregator can drop the record and retry).
    ProjectionRange {
        /// The record (`i`).
        record: usize,
        /// The row (`o`).
        row: usize,
        /// The output.
        value: i64,
    },
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

    /// The multilinear extension at `point` (least significant variable first): integer dots
    /// against the split equality tables of `point` (`agg_simd::eval::column_at`, parallel
    /// over blocks for large columns).
    ///
    /// # Panics
    /// If the column does not have `2^|point|` cells.
    #[must_use]
    pub fn evaluate(&self, point: &[E]) -> E {
        assert_eq!(self.len(), 1 << point.len(), "table length must be 2^|point|");
        let (first, second) = agg_simd::eval::split_eq_tables(point);
        match *self {
            Self::Narrow(cells) => agg_simd::eval::column_at(
                &agg_simd::eval::Mapped {
                    cells,
                    map: &crate::kernel::IDENTITY_MAP,
                },
                &first,
                &second,
            ),
            Self::Wide(cells) => agg_simd::eval::column_at(&cells, &first, &second),
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

/// Thread-summed nanoseconds of the witness derivation and of the encoding in the
/// [`MainTables`] builds, printed per build under `AGG_STAGE_TRACE` (builds that run at
/// the same time share them).
static DERIVE_NS: AtomicU64 = AtomicU64::new(0);
static ENCODE_NS: AtomicU64 = AtomicU64::new(0);

/// Add the time since `since` to `counter`; the current instant.
fn add_elapsed(counter: &AtomicU64, since: Instant) -> Instant {
    let now = Instant::now();
    counter.fetch_add(u64::try_from((now - since).as_nanos()).unwrap_or(u64::MAX), Ordering::Relaxed);
    now
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
    check_group(descriptor, columns, true)
}

/// [`validate_group`]; `scan = false` keeps the shape checks and skips the alphabet scan.
fn check_group(descriptor: &GroupDescriptor, columns: &[Cells<'_>], scan: bool) -> Result<(), SourceTableError> {
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
    let scanned = if scan { checked } else { 0 };
    let bounds: Vec<(i64, i64)> = descriptor.columns[..scanned]
        .iter()
        .map(|column| (column.alphabet().min(), column.alphabet().max()))
        .collect();
    let chunks = expected.div_ceil(VALIDATE_CHUNK);
    let bad_column = (0..scanned * chunks)
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

/// The values of one record of parameter set `P`: `z, u, v` and `w_1'` (whose bits are the
/// bit columns).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordValues<P: ParamSet = Set44> {
    /// `z[s][j]`.
    pub z: P::VecL,
    /// `u[rho][j]`.
    pub u: P::VecK,
    /// `v[rho][j]`.
    pub v: P::VecK,
    /// `w_1'[rho][j]` (bit `t` is `b_t`).
    pub w1: P::SmallK,
}

impl RecordValues<Set44> {
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

impl<P: ParamSet> RecordValues<P> {
    /// The canonical all-zero witness of a padded row of any set.
    #[must_use]
    pub fn zeroed() -> Self {
        Self {
            z: P::zero_vec_l(),
            u: P::zero_vec_k(),
            v: P::zero_vec_k(),
            w1: P::zero_small_k(),
        }
    }
}

/// The H-FIPS witness of one coefficient from `(u, w_1')`:
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
/// each digit in each polynomial lane, the slots of the bits and, under H-FIPS, of `s, delta,
/// e` (resolved once per batch from the manifest). Row `rho` of a vector sits in lane
/// `rho >> d_vec` at cells `256 (rho mod 2^d_vec) ..` of the record ([`SetShape::row_lane`]).
struct Encoder {
    /// Per value: the form, `slots[lane][digit]` and the abort message.
    values: [(ValueForm, Vec<Vec<Slot>>, &'static str); 3],
    /// `bits[t][lane]`: the narrow slot of `b_t` in lane `lane`.
    bits: Vec<Vec<usize>>,
    /// Under H-FIPS, the narrow slots of `s, delta, e` per row lane.
    fips: Option<Vec<[usize; 3]>>,
    wide: usize,
    narrow: usize,
    shape: SetShape,
    w1_what: &'static str,
    /// Per value: the shift form and the alphabet bounds of each digit, when every value has
    /// one (the path of [`Encoder::encode_rows`]).
    shift: Option<[FastValue; 3]>,
}

impl Encoder {
    fn new(manifest: &CommitmentManifest) -> Self {
        let shape = manifest.shape();
        let slot = |column: Column| match manifest.locate(column) {
            (GroupRole::Main16, position) => Slot::Wide(position),
            (GroupRole::Main, position) => Slot::Narrow(position),
            (role, _) => panic!("{} is in {role:?}, not a main group", column.name()),
        };
        let value = |value: DigitValue, lanes: usize, what: &'static str| {
            let slots = (0..lanes as u8)
                .map(|lane| value_lane_columns(value, lane).into_iter().map(slot).collect())
                .collect();
            (value.layout(), slots, what)
        };
        let z_range = match manifest.z_value() {
            DigitValue::Zx => "z outside [-130,993, 130,993]",
            DigitValue::Zd => "z outside [-2^19, 2^19)",
            DigitValue::Zdx(_) => "z outside [-B_z, B_z]",
            _ => "z outside [-2^17, 2^17)",
        };
        let narrow_slot = |column: Column| match slot(column) {
            Slot::Narrow(position) => position,
            Slot::Wide(_) => unreachable!("{} is narrow", column.name()),
        };
        let u_what = match manifest.u_slot_value() {
            DigitValue::L => "L outside [0, gamma_2)",
            DigitValue::Ud => "u outside [-523,777, 523,777]",
            _ => "u outside [-190,465, 190,465]",
        };
        let values = [
            value(manifest.z_value(), shape.lanes_l, z_range),
            value(manifest.u_slot_value(), shape.lanes_k, u_what),
            value(DigitValue::V, shape.lanes_k, "v outside [-2^31, 2^31)"),
        ];
        assert!(values.iter().all(|(form, _, _)| form.len() <= MAX_DIGITS), "at most {MAX_DIGITS} digits");
        Self {
            values,
            bits: (0..shape.w1_bits as u8)
                .map(|t| (0..shape.lanes_k as u8).map(|lane| narrow_slot(Column::W1Bit(t, lane))).collect())
                .collect(),
            fips: manifest.is_fips().then(|| {
                (0..shape.lanes_k as u8)
                    .map(|lane| FipsColumn::ALL.map(|c| narrow_slot(Column::Fips(c, lane))))
                    .collect()
            }),
            wide: manifest.group(GroupRole::Main16).polynomial_count(),
            narrow: manifest.group(GroupRole::Main).polynomial_count(),
            shape,
            w1_what: if shape.w1_bits == 6 { "w1' above six bits" } else { "w1' above four bits" },
            shift: None,
        }
        .with_shift_forms()
    }

    /// Whether every slot the encoder writes has, in the manifest, the alphabet the encoder
    /// checks it against: the digit alphabets of the three value forms, `{0, 1}` for the
    /// bits of `w_1'` and the H-FIPS sign, `{-1, 0, 1}` for `delta` and `e`.
    fn alphabets_match(&self, manifest: &CommitmentManifest) -> bool {
        let bounds = |role: GroupRole, position: usize| {
            let alphabet = manifest.group(role).columns[position].alphabet();
            (alphabet.min(), alphabet.max())
        };
        let slot_bounds = |slot: Slot| match slot {
            Slot::Wide(c) => bounds(GroupRole::Main16, c),
            Slot::Narrow(c) => bounds(GroupRole::Main, c),
        };
        let values = self.values.iter().all(|(form, slots, _)| {
            slots.iter().all(|lane| {
                lane.len() == form.len()
                    && lane.iter().enumerate().all(|(k, &slot)| {
                        slot_bounds(slot) == (form.alphabet(k).min(), form.alphabet(k).max())
                    })
            })
        });
        let bits = self.bits.iter().flatten().all(|&c| bounds(GroupRole::Main, c) == (0, 1));
        let fips = self.fips.iter().flatten().all(|&[s, delta, e]| {
            bounds(GroupRole::Main, s) == (0, 1)
                && bounds(GroupRole::Main, delta) == (-1, 1)
                && bounds(GroupRole::Main, e) == (-1, 1)
        });
        values && bits && fips
    }

    /// Resolve the shift forms of the three values (every set and relation has them).
    fn with_shift_forms(mut self) -> Self {
        let forms: Option<Vec<_>> = self
            .values
            .iter()
            .map(|(form, _, _)| {
                ShiftForm::of(form).map(|f| {
                    let bounds = (0..form.len())
                        .map(|k| (form.alphabet(k).min() as i32, form.alphabet(k).max() as i32))
                        .collect();
                    (f, bounds)
                })
            })
            .collect();
        self.shift = forms.and_then(|f| f.try_into().ok());
        self
    }

    /// Decompose `x` by value `value`'s form into its slots of lane `lane` at cell `pos`.
    #[inline]
    fn put(
        &self,
        value: usize,
        lane: usize,
        pos: usize,
        x: i64,
        wide: &mut [&mut [MaybeUninit<i32>]],
        narrow: &mut [&mut [MaybeUninit<i8>]],
    ) -> Result<(), SourceTableError> {
        let (form, slots, what) = &self.values[value];
        let mut buffer = [0_i64; MAX_DIGITS];
        let digits = &mut buffer[..form.len()];
        form.decompose_into(x, digits);
        if !form.in_alphabets(digits) {
            return Err(SourceTableError::Encode(what));
        }
        for (&slot, &digit) in slots[lane].iter().zip(&*digits) {
            match slot {
                Slot::Wide(c) => {
                    wide[c][pos].write(i32::try_from(digit).expect("16-bit digit"));
                }
                Slot::Narrow(c) => {
                    narrow[c][pos].write(i8::try_from(digit).expect("small digit"));
                }
            }
        }
        Ok(())
    }

    fn encode_values<P: ParamSet>(
        &self,
        values: &RecordValues<P>,
        wide: &mut [&mut [MaybeUninit<i32>]],
        narrow: &mut [&mut [MaybeUninit<i8>]],
    ) -> Result<i64, SourceTableError> {
        self.encode::<P>(values.z.as_ref(), values.u.as_ref(), values.v.as_ref(), values.w1.as_ref(), wide, narrow)
    }

    fn encode<P: ParamSet>(
        &self,
        z: &[[i64; N]],
        u: &[[i64; N]],
        v: &[[i64; N]],
        w1: &[[u8; N]],
        wide: &mut [&mut [MaybeUninit<i32>]],
        narrow: &mut [&mut [MaybeUninit<i8>]],
    ) -> Result<i64, SourceTableError> {
        let shape = self.shape;
        let mut weight = 0_i64;
        for row in 0..shape.k.max(shape.l) {
            let (lane, base) = shape.row_lane(row);
            for j in 0..N {
                let pos = base + j;
                // H-FIPS: the coefficient's witness; its magnitude L takes the slot of u.
                let middle = if row < shape.k {
                    Some(match &self.fips {
                        None => u[row][j],
                        Some(slots) => {
                            let unreachable = SourceTableError::Encode("w1' not reachable from w' by UseHint");
                            let (u, w1) = (u[row][j], i64::from(w1[row][j]));
                            let (l, sigma, delta, e) = if P::SET == Set::S44 {
                                let c = fips_coefficient(u, w1).ok_or(unreachable)?;
                                (c.l(), c.sigma, c.delta, c.e)
                            } else {
                                let c = coefficient_of::<P>(u, w1).ok_or(unreachable)?;
                                (c.l, c.sigma, c.delta, c.e)
                            };
                            for (&slot, cell) in slots[lane].iter().zip([(1 + sigma) / 2, delta, e]) {
                                narrow[slot][pos].write(cell as i8);
                            }
                            weight += delta * delta;
                            l
                        }
                    })
                } else {
                    None
                };
                // z is indexed by (j, s); u, v and the bits by (j, rho).
                if row < shape.l {
                    self.put(0, lane, pos, z[row][j], wide, narrow)?;
                }
                let Some(middle) = middle else { continue };
                self.put(1, lane, pos, middle, wide, narrow)?;
                self.put(2, lane, pos, v[row][j], wide, narrow)?;
                let w1 = w1[row][j];
                if usize::from(w1) >> shape.w1_bits != 0 {
                    return Err(SourceTableError::Encode(self.w1_what));
                }
                for (t, slots) in self.bits.iter().enumerate() {
                    narrow[slots[lane]][pos].write(((w1 >> t) & 1) as i8);
                }
            }
        }
        let hint_slack = P::OMEGA as i64 - weight;
        if hint_slack < 0 {
            return Err(SourceTableError::Encode("hint weight above omega"));
        }
        Ok(hint_slack)
    }
}

/// A value form decomposed with 32-bit shifts, masks and compares instead of the generic
/// divisions of `ValueForm::decompose_into`: every form of the main tables with one low digit
/// of base `2^low_bits` (and radix forms with power-of-two bases). On `[-2^29, 2^29]` (and
/// all of `i32` for `Radix`) the digits equal `decompose_into`'s, and outside that range an
/// out-of-alphabet digit still fails `bounds`, the form's alphabets.
#[derive(Clone, Copy, Debug)]
enum ShiftForm {
    /// `ValueForm::DigitSlack`: `y = x + offset`, `a = [y >= threshold]`, `rest = y - w a`,
    /// `mid = floor(rest / 2^j)`, the low digit `rest mod 2^j` stored balanced.
    DigitSlack { low_bits: u32, mid_half: i32, threshold: i32, slack_weight: i32, offset: i32 },
    /// `ValueForm::BitSlack`: `y = x + offset`, `a = [y >= 2^(j+1)]`, `rest = y - w a`,
    /// `hi = clamp(floor(rest / 2^j), 0, 1)`, the low digit `rest - 2^j hi` stored balanced.
    BitSlack { low_bits: u32, slack_weight: i32, offset: i32 },
    /// `ValueForm::Slack`: `y = x + offset`, `a = [y >= 2^j]`, the low digit `y - w a` stored
    /// balanced, then `a`.
    Slack { low_bits: u32, slack_weight: i32, offset: i32 },
    /// `ValueForm::Step`: `y = x + offset`, `h = clamp(floor(y / step), 0, c - 1)` (the count
    /// of the multiples `t step <= y` for `0 < t < c`), the low digit `y - step h` stored
    /// balanced, then `h - c/2`.
    Step { low_bits: u32, step: i32, top_base: i32, offset: i32 },
    /// `ValueForm::Radix` with two or three power-of-two bases: `y = x + shift` peeled from the
    /// bottom, the top digit taking the excess (`shift = B / 2` for signed forms adds `b_top / 2`
    /// to the top quotient, which cancels `-b_top / 2` and leaves `x >> low_bits`).
    Radix { bits: [u32; 3], digits: usize, signed: bool },
}

/// The most digits of a value of the main tables.
const SHIFT_DIGITS: usize = 3;

impl ShiftForm {
    /// The shift form of `form`, if its bases allow one.
    fn of(form: &ValueForm) -> Option<Self> {
        let pow2 = |b: u32| b > 1 && b.is_power_of_two();
        match form {
            ValueForm::DigitSlack(l)
                if l.low.len() == 1 && !l.low.signed && pow2(l.low.bases[0]) && l.mid_base % 2 == 0 =>
            {
                Some(Self::DigitSlack {
                    low_bits: l.low.bases[0].trailing_zeros(),
                    mid_half: (l.mid_base / 2) as i32,
                    threshold: i32::try_from(l.low.product() * i64::from(l.mid_base)).ok()?,
                    slack_weight: i32::try_from(l.slack_weight).ok()?,
                    offset: i32::try_from(l.offset).ok()?,
                })
            }
            ValueForm::BitSlack(l) if l.low.len() == 1 && !l.low.signed && pow2(l.low.bases[0]) => {
                Some(Self::BitSlack {
                    low_bits: l.low.bases[0].trailing_zeros(),
                    slack_weight: i32::try_from(l.slack_weight).ok()?,
                    offset: i32::try_from(l.offset).ok()?,
                })
            }
            ValueForm::Slack(l) if l.digits.len() == 1 && !l.digits.signed && pow2(l.digits.bases[0]) => {
                Some(Self::Slack {
                    low_bits: l.digits.bases[0].trailing_zeros(),
                    slack_weight: i32::try_from(l.slack_weight).ok()?,
                    offset: i32::try_from(l.offset).ok()?,
                })
            }
            ValueForm::Step(l)
                if l.low.len() == 1 && !l.low.signed && pow2(l.low.bases[0]) && l.step > 0 && (2..=16).contains(&l.top_base) =>
            {
                Some(Self::Step {
                    low_bits: l.low.bases[0].trailing_zeros(),
                    step: i32::try_from(l.step).ok()?,
                    top_base: l.top_base as i32,
                    offset: i32::try_from(l.offset).ok()?,
                })
            }
            ValueForm::Radix(l) if (2..=SHIFT_DIGITS).contains(&l.len()) && l.bases.iter().all(|&b| pow2(b)) => {
                let mut bits = [0; SHIFT_DIGITS];
                for (dst, &b) in bits.iter_mut().zip(l.bases) {
                    *dst = b.trailing_zeros();
                }
                Some(Self::Radix { bits, digits: l.len(), signed: l.signed })
            }
            _ => None,
        }
    }

    /// The digits of the 256 values `xs` into `out[k]` (digit `k`); the forms are written as
    /// branch-free `i32` loops over the coefficients so that they vectorise 8-wide on AVX2.
    #[inline(always)]
    fn decompose_poly(self, xs: &[i32; N], out: &mut [[i32; N]; SHIFT_DIGITS]) {
        let [d0, d1, d2] = out;
        match self {
            Self::DigitSlack { low_bits, mid_half, threshold, slack_weight, offset } => {
                let mask = (1_i32 << low_bits) - 1;
                let half = 1_i32 << (low_bits - 1);
                for j in 0..N {
                    let y = xs[j].clamp(-(1 << 29), 1 << 29) + offset;
                    let a = i32::from(y >= threshold);
                    let rest = y - slack_weight * a;
                    d0[j] = (rest & mask) - half;
                    d1[j] = (rest >> low_bits) - mid_half;
                    d2[j] = a;
                }
            }
            Self::BitSlack { low_bits, slack_weight, offset } => {
                let lo = 1_i32 << low_bits;
                let half = lo >> 1;
                for j in 0..N {
                    let y = xs[j].clamp(-(1 << 29), 1 << 29) + offset;
                    let a = i32::from(y >= 2 * lo);
                    let rest = y - slack_weight * a;
                    let hi = (rest >> low_bits).clamp(0, 1);
                    d0[j] = rest - lo * hi - half;
                    d1[j] = hi;
                    d2[j] = a;
                }
            }
            Self::Slack { low_bits, slack_weight, offset } => {
                let lo = 1_i32 << low_bits;
                let half = lo >> 1;
                for j in 0..N {
                    let y = xs[j].clamp(-(1 << 29), 1 << 29) + offset;
                    let a = i32::from(y >= lo);
                    d0[j] = y - slack_weight * a - half;
                    d1[j] = a;
                }
            }
            Self::Step { low_bits, step, top_base, offset } => {
                let half = 1_i32 << (low_bits - 1);
                d1.fill(-(top_base / 2));
                for t in 1..top_base {
                    let threshold = t * step - offset;
                    for j in 0..N {
                        d1[j] += i32::from(xs[j] >= threshold);
                    }
                }
                for j in 0..N {
                    let y = xs[j].clamp(-(1 << 29), 1 << 29) + offset;
                    let h = d1[j] + top_base / 2;
                    d0[j] = y - step * h - half;
                }
            }
            Self::Radix { bits, digits, signed } => {
                let digit = |x: i32, b: u32| (x & ((1 << b) - 1)) - (1 << (b - 1));
                if digits == 2 {
                    let top_bias = if signed { 0 } else { 1_i32 << (bits[1] - 1) };
                    for j in 0..N {
                        let x = xs[j];
                        d0[j] = digit(x, bits[0]);
                        d1[j] = (x >> bits[0]) - top_bias;
                    }
                } else {
                    let top_bias = if signed { 0 } else { 1_i32 << (bits[2] - 1) };
                    for j in 0..N {
                        let x = xs[j];
                        d0[j] = digit(x, bits[0]);
                        d1[j] = digit(x >> bits[0], bits[1]);
                        d2[j] = (x >> (bits[0] + bits[1])) - top_bias;
                    }
                }
            }
        }
    }
}

/// The H-FIPS witness of one row from `(u, w_1')`, [`fips_coefficient`] (ML-DSA-44) or
/// [`coefficient_of`] (the other sets) per coefficient, as branch-free arithmetic over the
/// row that the compiler vectorises: the magnitudes `L` into `l`, `(s, delta, e)` into
/// `cells`, and the row's `sum delta^2`. `None` when a coefficient lies outside
/// `|u| <= B_u`, `w_1' < m_w` (where the two agree; checked exhaustively in the tests) or has
/// no witness.
#[inline(always)]
fn fips_row<P: ParamSet>(u: &[i32; N], w1: &[u8; N], l: &mut [i32; N], cells: &mut [[i8; N]; 3]) -> Option<i64> {
    let mut bad = false;
    let mut weight = 0_i32;
    let [s_cells, delta_cells, e_cells] = cells;
    for j in 0..N {
        let (ok, lj, s, delta, e) = fips_cell::<P>(u[j], w1[j]);
        bad |= !ok;
        l[j] = lj;
        s_cells[j] = s as i8;
        delta_cells[j] = delta as i8;
        e_cells[j] = e as i8;
        weight += delta * delta;
    }
    (!bad).then_some(i64::from(weight))
}

/// One coefficient of [`fips_row`]: whether it is accepted, then `(L, s, delta, e)`.
#[inline(always)]
fn fips_cell<P: ParamSet>(u: i32, w1: u8) -> (bool, i32, i32, i32, i32) {
    let (q, g2, m) = (P::Q as i32, P::GAMMA2 as i32, P::M_W as i32);
    let alpha = 2 * g2;
    let w1 = i32::from(w1);
    let mut ok = u.unsigned_abs() <= P::B_U as u32 && w1 < m;
    // w' = 2 gamma2 w1' + u mod q from (-q, 2q) inside the checked domain.
    let x = alpha.wrapping_mul(w1).wrapping_add(u);
    let x = x + ((x >> 31) & q);
    let wq = x - (((q - 1 - x) >> 31) & q);
    // Decompose(w'): r0 = w' mod+- 2 gamma2 in (-gamma2, gamma2], r1 = (w' - r0) / (2 gamma2),
    // and (0, r0 - 1) when w' - r0 = q - 1.
    let r1 = ((wq + g2 - 1) as u32 / alpha as u32) as i32;
    let r0 = wq.wrapping_sub(alpha.wrapping_mul(r1));
    let corner = alpha.wrapping_mul(r1) == q - 1;
    let (r1, r0) = if corner { (0, r0 - 1) } else { (r1, r0) };
    // h* = [HighBits(w') != w1'], and UseHint(h*, w') must give w1' back.
    let hint = r1 != w1;
    let moved = if r0 > 0 { r1 + 1 } else { r1 - 1 };
    let moved = if moved == m { 0 } else if moved < 0 { m - 1 } else { moved };
    ok &= (if hint { moved } else { r1 }) == w1;
    // The honest witness of (w', h*).
    let hv = i32::from(hint);
    let big = wq >= q - g2;
    let sigma = if big || r0 <= 0 { -1 } else { 1 };
    let l = if big { wq - (q - g2) } else { (r0 - 1) + (((r0 - 1) >> 31) & g2) };
    let delta = if big { -hv } else { hv * sigma };
    let wraps = (sigma == 1 && r1 == m - 1) || (sigma == -1 && r1 == 0);
    let e = if big { 1 - hv } else if hint && wraps { sigma } else { 0 };
    let s = (1 + sigma) >> 1;
    // The witness must give u back: r0 = gamma2 s + L + 1 - gamma2, u = r0 - 2 gamma2 delta - e.
    ok &= (g2 * s + l + 1 - g2).wrapping_sub(alpha.wrapping_mul(delta)) - e == u;
    (ok, l, s, delta, e)
}

/// True when every entry of `digits` lies in `[min, max]` (a min/max fold that vectorises).
#[inline(always)]
fn digits_in(digits: &[i32; N], (min, max): (i32, i32)) -> bool {
    let (lo, hi) = digits.iter().fold((i32::MAX, i32::MIN), |(lo, hi), &d| (lo.min(d), hi.max(d)));
    min <= lo && hi <= max
}

/// The fast form of one committed value: its shift form and the alphabet of each digit.
type FastValue = (ShiftForm, Vec<(i32, i32)>);

impl Encoder {
    /// [`Encoder::encode`] on the engine witness (`z` per column, `(w1', u, v)` per row) with
    /// the shift forms: the same cells and hint slack, or `None` when the encoder has no
    /// shift forms or any check fails. The caller then runs [`Encoder::encode`], which
    /// rewrites every cell of the record and returns its error.
    fn encode_rows<P: ParamSet>(
        &self,
        z: &[agg_mldsa_simd::Poly32],
        rows: &[agg_mldsa_simd::FinishOut],
        wide: &mut [&mut [MaybeUninit<i32>]],
        narrow: &mut [&mut [MaybeUninit<i8>]],
    ) -> Option<i64> {
        let fast = self.shift.as_ref()?;
        let shape = self.shape;
        let mut digits = [[0_i32; N]; SHIFT_DIGITS];
        let mut put = |value: usize, lane: usize, base: usize, xs: &[i32; N], wide: &mut [&mut [MaybeUninit<i32>]], narrow: &mut [&mut [MaybeUninit<i8>]]| -> bool {
            let (form, bounds) = &fast[value];
            form.decompose_poly(xs, &mut digits);
            if !digits.iter().zip(bounds).all(|(d, &b)| digits_in(d, b)) {
                return false;
            }
            for (&slot, d) in self.values[value].1[lane].iter().zip(&digits) {
                match slot {
                    // Alphabet-checked digits: 16-bit wide, small narrow.
                    Slot::Wide(c) => {
                        for (cell, &x) in wide[c][base..base + N].iter_mut().zip(d) {
                            cell.write(x);
                        }
                    }
                    Slot::Narrow(c) => {
                        for (cell, &x) in narrow[c][base..base + N].iter_mut().zip(d) {
                            cell.write(x as i8);
                        }
                    }
                }
            }
            true
        };
        let mut weight = 0_i64;
        let mut middle = [0_i32; N];
        for row in 0..shape.k.max(shape.l) {
            let (lane, base) = shape.row_lane(row);
            if row < shape.l && !put(0, lane, base, &z[row].0, wide, narrow) {
                return None;
            }
            if row >= shape.k {
                continue;
            }
            let out = &rows[row];
            let u = match &self.fips {
                None => &out.u.0,
                Some(slots) => {
                    // H-FIPS: the coefficient's witness; its magnitude L takes the slot of u.
                    let mut cells = [[0_i8; N]; 3];
                    weight += fips_row::<P>(&out.u.0, &out.w1, &mut middle, &mut cells)?;
                    for (&slot, cells) in slots[lane].iter().zip(&cells) {
                        for (cell, &x) in narrow[slot][base..base + N].iter_mut().zip(cells) {
                            cell.write(x);
                        }
                    }
                    &middle
                }
            };
            if !put(1, lane, base, u, wide, narrow) || !put(2, lane, base, &out.v.0, wide, narrow) {
                return None;
            }
            if out.w1.iter().any(|&w| usize::from(w) >> shape.w1_bits != 0) {
                return None;
            }
            for (t, slots) in self.bits.iter().enumerate() {
                for (cell, &w) in narrow[slots[lane]][base..base + N].iter_mut().zip(&out.w1) {
                    cell.write(((w >> t) & 1) as i8);
                }
            }
        }
        let hint_slack = P::OMEGA as i64 - weight;
        (hint_slack >= 0).then_some(hint_slack)
    }
}

/// The final-column views of one record: the `G_main16` columns, then the `G_main4` columns
/// (manifest order, [`CommitmentManifest::lane_cells`] cells each).
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
            .chain((0..W1_BITS as u8).map(|t| Column::W1Bit(t, 0)));
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

/// The bridge inputs of the four bit lanes packed as nibbles: cell `j + 256 rho + P_pad i`
/// holds `sum_t (b_t & 1) << t` of record `i`'s committed bit cells at that position, and
/// the padded positions `256 k .. P_pad` are zero. One pass over the bit columns replaces
/// four `i8` lanes ([`MainTables::bridge_lanes`]) and their packing.
pub struct BridgeNibbles<'a> {
    columns: [Vec<&'a [i8]>; 4],
    shape: SetShape,
    lane_cells: usize,
    records: usize,
}

impl BridgeNibbles<'_> {
    /// Record `i`'s `P_pad` cells into `out`; whether its cells are all bits.
    pub fn record_into(&self, i: usize, out: &mut [u8]) -> bool {
        let shape = self.shape;
        assert_eq!(out.len(), shape.positions_pad, "one record's padded positions");
        let mut ok = true;
        for (row, dst) in out[..shape.positions].chunks_exact_mut(N).enumerate() {
            let (lane, base) = shape.row_lane(row);
            let start = i * self.lane_cells + base;
            let src: [&[i8]; 4] = std::array::from_fn(|t| &self.columns[t][lane][start..start + N]);
            ok &= agg_simd::digits::pack_four_bits_into(src, dst);
        }
        out[shape.positions..].fill(0);
        ok
    }

    /// Every record's cells (`P_pad` per padded record) and whether they are all bits.
    #[must_use]
    pub fn all(&self) -> (Vec<u8>, bool) {
        let pad = self.shape.positions_pad;
        let mut out = vec![0_u8; pad * self.records];
        let ok = out
            .par_chunks_mut(pad)
            .enumerate()
            .map(|(i, record)| self.record_into(i, record))
            .reduce(|| true, |a, b| a && b);
        (out, ok)
    }
}

/// Round-1 tables: `G_main16` (one stacked `i32` lane) and `G_main4` (`i8` columns), and
/// under H-FIPS the hint slack of every record (committed in round 2, `G_2`). Every
/// polynomial holds [`CommitmentManifest::lane_cells`] cells per record: 1,024 for ML-DSA-44
/// (all four rows of a vector), 256 for ML-DSA-65 and ML-DSA-87 (one row per lane).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MainTables {
    geom: Geometry,
    set: Set,
    lane_cells: usize,
    wide: AlignedVec<i32>,
    narrow: Vec<AlignedVec<i8>>,
    hint_slack: Vec<i64>,
}

impl MainTables {
    /// Encodes the record values (live records in order; padded rows get the zero
    /// witness) and checks the tables against the manifest.
    ///
    /// # Panics
    /// If the manifest is of another parameter set than `P`.
    pub fn from_values<P: ParamSet>(
        manifest: &CommitmentManifest,
        values: &[RecordValues<P>],
    ) -> Result<Self, SourceTableError> {
        Self::encode_records(manifest, values.len(), |i| Ok(values[i].clone()))
    }

    /// Encode record by record (live records from `record`, padded rows the zero witness).
    fn encode_records<P: ParamSet>(
        manifest: &CommitmentManifest,
        count: usize,
        record: impl Fn(usize) -> Result<RecordValues<P>, SourceTableError> + Sync,
    ) -> Result<Self, SourceTableError> {
        assert_eq!(manifest.set(), P::SET, "values of the manifest's parameter set");
        Self::encode_tiles(manifest, count, 1, true, |encoder, i, views| {
            let (wide, narrow) = &mut views[0];
            let values = if i < count { record(i)? } else { RecordValues::zeroed() };
            Ok(vec![encoder.encode_values(&values, wide, narrow)?])
        })
    }

    /// Allocate final columns and give each worker the disjoint views of a tile of `tile`
    /// consecutive records (the last tile may be shorter); `fill(encoder, first, views)`
    /// writes every cell of records `first..first + views.len()` (live and padded) and
    /// returns their hint slacks. `checked` says that `fill` checks every cell it writes
    /// against its digit's alphabet (or writes a cell whose alphabet holds it by
    /// construction: a bit of `w_1'`, an H-FIPS sign, `delta` or `e`); the alphabet scan of
    /// the self-check is then skipped once the encoder's alphabets are found to be the
    /// manifest's.
    #[allow(unsafe_code)]
    fn encode_tiles(
        manifest: &CommitmentManifest,
        count: usize,
        tile: usize,
        checked: bool,
        fill: impl Fn(&Encoder, usize, &mut [RecordViews<'_>]) -> Result<Vec<i64>, SourceTableError> + Sync,
    ) -> Result<Self, SourceTableError> {
        let start = Instant::now();
        let encoder = Encoder::new(manifest);
        let geom = Geometry::new(count).map_err(SourceTableError::Geometry)?;
        if geom.n != manifest.records() || geom.n_star != manifest.padded_records() {
            return Err(SourceTableError::Records { expected: manifest.records(), actual: geom.n });
        }
        let lane_cells = manifest.lane_cells();
        let cells = geom.n_star * lane_cells;
        let mut wide = AlignedVec::<i32>::uninit(encoder.wide * cells);
        let mut narrow: Vec<_> = (0..encoder.narrow).map(|_| AlignedVec::<i8>::uninit(cells)).collect();
        // The columns are cut per tile here and per record inside the tile's task: a
        // record-major list of every record's views would cost one push per column and
        // record on this thread.
        let tile_cells = tile * lane_cells;
        let mut tile_views: Vec<RecordViews<'_>> = (0..geom.n_star.div_ceil(tile))
            .map(|_| (Vec::with_capacity(encoder.wide), Vec::with_capacity(encoder.narrow)))
            .collect();
        for column in wide.chunks_exact_mut(cells) {
            for (t, chunk) in column.chunks_mut(tile_cells).enumerate() {
                tile_views[t].0.push(chunk);
            }
        }
        for column in &mut narrow {
            for (t, chunk) in column.chunks_mut(tile_cells).enumerate() {
                tile_views[t].1.push(chunk);
            }
        }
        let trace = std::env::var_os("AGG_STAGE_TRACE").is_some();
        let (derive0, encode0) = (DERIVE_NS.load(Ordering::Relaxed), ENCODE_NS.load(Ordering::Relaxed));
        let filling = Instant::now();
        let tiles = tile_views
            .into_par_iter()
            .enumerate()
            .map(|(t, (wide, narrow))| {
                let records = (geom.n_star - t * tile).min(tile);
                let mut views: Vec<RecordViews<'_>> = (0..records)
                    .map(|_| (Vec::with_capacity(encoder.wide), Vec::with_capacity(encoder.narrow)))
                    .collect();
                for column in wide {
                    for (i, chunk) in column.chunks_exact_mut(lane_cells).enumerate() {
                        views[i].0.push(chunk);
                    }
                }
                for column in narrow {
                    for (i, chunk) in column.chunks_exact_mut(lane_cells).enumerate() {
                        views[i].1.push(chunk);
                    }
                }
                fill(&encoder, t * tile, &mut views)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let filled = Instant::now();
        let hint_slack: Vec<i64> = tiles.into_iter().flatten().collect();
        // SAFETY: successful encoding writes every digit, bit and FIPS cell of every
        // record, including padding. On error the MaybeUninit allocations just drop.
        let wide = unsafe { wide.assume_init() };
        let narrow = narrow.into_iter().map(|column| unsafe { column.assume_init() }).collect();
        let tables = Self {
            geom, set: manifest.set(), lane_cells, wide, narrow,
            hint_slack: if encoder.fips.is_some() { hint_slack } else { Vec::new() },
        };
        let scan = !(checked && encoder.alphabets_match(manifest));
        tables.check(manifest, scan)?;
        if trace {
            let ms = |ns: u64| ns as f64 / 1e6;
            eprintln!(
                "main tables: set-up {:.1} ms, derive and encode {:.1} ms (thread sums: derive {:.1} ms, encode {:.1} ms), self-check {:.1} ms (alphabet scan {scan})",
                (filling - start).as_secs_f64() * 1e3,
                (filled - filling).as_secs_f64() * 1e3,
                ms(DERIVE_NS.load(Ordering::Relaxed).wrapping_sub(derive0)),
                ms(ENCODE_NS.load(Ordering::Relaxed).wrapping_sub(encode0)),
                filled.elapsed().as_secs_f64() * 1e3,
            );
        }
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
    ///
    /// # Panics
    /// If the manifest is not ML-DSA-44's ([`MainTables::build_for`] builds every set).
    #[doc(hidden)]
    pub fn build_with(
        engine: &Engine,
        manifest: &CommitmentManifest,
        instance: &PublicInstance,
        signatures: &[&[u8]],
    ) -> Result<Self, SourceTableError> {
        assert_eq!(manifest.set(), Set::S44, "the engine path encodes ML-DSA-44");
        if signatures.len() != instance.records.len() {
            return Err(SourceTableError::Records {
                expected: instance.records.len(), actual: signatures.len(),
            });
        }
        let count = instance.records.len();
        let hu = (manifest.relation() == Relation::Hu).then(|| HuSlots::new(manifest));
        // The digit kernel of relation H-u writes its cells unchecked.
        Self::encode_tiles(manifest, count, TILE, hu.is_none(), |encoder, first, views| {
            // Live records of the tile; a tile past the last record is all padding.
            let live = first.min(count)..(first + views.len()).min(count);
            let records: Vec<&PublicRecord> = instance.records[live.clone()].iter().collect();
            let clock = Instant::now();
            let mut derived = vec![FastWitness::zero(); live.len()];
            derive_tile_with(engine, &records, &signatures[live], &mut derived)
                .map_err(|(i, e)| SourceTableError::Derive(first + i, e))?;
            let clock = add_elapsed(&DERIVE_NS, clock);
            let mut slack = Vec::with_capacity(views.len());
            for (k, (wide, narrow)) in views.iter_mut().enumerate() {
                let Some(w) = derived.get(k) else {
                    slack.push(encoder.encode_values(&RecordValues::zero(), wide, narrow)?);
                    continue;
                };
                match &hu {
                    Some(slots) => {
                        slots.write(engine, w, wide, narrow);
                        slack.push(OMEGA as i64);
                    }
                    None => match encoder.encode_rows::<Set44>(&w.z, &w.rows, wide, narrow) {
                        Some(encoded) => slack.push(encoded),
                        None => {
                            let values = RecordValues::<Set44> { z: w.z_wide(), u: w.u_wide(), v: w.v_wide(), w1: w.w1() };
                            slack.push(encoder.encode_values(&values, wide, narrow)?);
                        }
                    },
                }
            }
            add_elapsed(&ENCODE_NS, clock);
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
        Self::encode_records::<Set44>(manifest, instance.records.len(), |i| {
            let record = &instance.records[i];
            let witness = derive_with_t(&record.key, &record.mu, signatures[i], &record.c_tilde, &record.t)
                .map_err(|e| SourceTableError::Derive(i, e))?;
            Ok(RecordValues { z: witness.z, u: witness.u, v: witness.v, w1: witness.w1 })
        })
    }

    /// The tables of any parameter set from the legacy derivation (`derive_with_t_for`) and
    /// the generic encoder, records in tiles of [`TILE`] in parallel. For ML-DSA-44 it equals
    /// [`MainTables::build`], which runs the SIMD engine instead.
    ///
    /// # Panics
    /// If the manifest is of another parameter set than `P`.
    pub fn build_for<P: ParamSet>(
        manifest: &CommitmentManifest,
        instance: &PublicInstance<P>,
        signatures: &[&[u8]],
    ) -> Result<Self, SourceTableError> {
        assert_eq!(manifest.set(), P::SET, "an instance of the manifest's parameter set");
        if signatures.len() != instance.records.len() {
            return Err(SourceTableError::Records {
                expected: instance.records.len(), actual: signatures.len(),
            });
        }
        let count = instance.records.len();
        if P::SET == Set::S44 {
            return Self::encode_tiles(manifest, count, TILE, true, |encoder, first, views| {
                let mut slack = Vec::with_capacity(views.len());
                for (k, (wide, narrow)) in views.iter_mut().enumerate() {
                    let i = first + k;
                    let values = match instance.records.get(i) {
                        None => RecordValues::<P>::zeroed(),
                        Some(record) => {
                            let t: Vec<[i32; N]> = (0..P::K).map(|row| record.t[row]).collect();
                            let witness =
                                derive_with_t_for::<P>(&record.key, &record.mu, signatures[i], &record.c_tilde, &t)
                                    .map_err(|e| SourceTableError::Derive(i, e))?;
                            RecordValues::<P> { z: witness.z, u: witness.u, v: witness.v, w1: witness.w1 }
                        }
                    };
                    slack.push(encoder.encode_values(&values, wide, narrow)?);
                }
                Ok(slack)
            });
        }
        // The sets with polynomial lanes derive each tile on the SIMD engine
        // (`agg_witness::fast_lanes`; its errors are the legacy derivation's).
        Self::encode_tiles(manifest, count, TILE, true, |encoder, first, views| {
            let live = first.min(count)..(first + views.len()).min(count);
            let records: Vec<&PublicRecord<P>> = instance.records[live.clone()].iter().collect();
            let clock = Instant::now();
            let mut derived = vec![LaneWitness::zero::<P>(); live.len()];
            P::derive_lane_tile(&records, &signatures[live], &mut derived)
                .map_err(|(i, e)| SourceTableError::Derive(first + i, e))?;
            let clock = add_elapsed(&DERIVE_NS, clock);
            let mut slack = Vec::with_capacity(views.len());
            for (k, (wide, narrow)) in views.iter_mut().enumerate() {
                if let Some(w) = derived.get(k) {
                    if let Some(encoded) = encoder.encode_rows::<P>(&w.z, &w.rows, wide, narrow) {
                        slack.push(encoded);
                        continue;
                    }
                }
                let values = match derived.get(k) {
                    None => RecordValues::<P>::zeroed(),
                    Some(w) => {
                        let mut values = RecordValues::<P>::zeroed();
                        for (dest, src) in values.z.as_mut().iter_mut().zip(&w.z) {
                            *dest = src.0.map(i64::from);
                        }
                        for (r, row) in w.rows.iter().enumerate() {
                            values.u[r] = row.u.0.map(i64::from);
                            values.v[r] = row.v.0.map(i64::from);
                            values.w1[r] = row.w1;
                        }
                        values
                    }
                };
                slack.push(encoder.encode_values(&values, wide, narrow)?);
            }
            add_elapsed(&ENCODE_NS, clock);
            Ok(slack)
        })
    }

    /// The shape and alphabet self-check of both groups.
    pub fn validate(&self, manifest: &CommitmentManifest) -> Result<(), SourceTableError> {
        self.check(manifest, true)
    }

    /// [`MainTables::validate`]; `scan = false` keeps the shape checks only.
    fn check(&self, manifest: &CommitmentManifest, scan: bool) -> Result<(), SourceTableError> {
        check_group(manifest.group(GroupRole::Main16), &self.wide_columns(), scan)?;
        let narrow: Vec<Cells<'_>> = self.narrow.iter().map(|c| Cells::Narrow(c)).collect();
        check_group(manifest.group(GroupRole::Main), &narrow, scan)
    }

    /// Batch geometry: the record counts and `d_sig` of any set; its main-domain helpers
    /// (`d_main`, `main_len`) are ML-DSA-44's, so other sets read
    /// [`CommitmentManifest::d_main`] and [`MainTables::main_len`].
    #[must_use]
    pub const fn geometry(&self) -> Geometry {
        self.geom
    }

    /// The parameter set of the tables.
    #[must_use]
    pub const fn set(&self) -> Set {
        self.set
    }

    /// Cells of one main-domain polynomial: `N* 2^(d_main - d_sig)`.
    #[must_use]
    pub const fn main_len(&self) -> usize {
        self.geom.n_star * self.lane_cells
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
    /// H-exact; `4 * 2^d_main` cells; for ML-DSA-65 and ML-DSA-87 every column is its lanes in
    /// order, `l + 3k` polynomials).
    #[must_use]
    pub fn wide(&self) -> &[i32] {
        &self.wide
    }

    /// The four `G_main16` columns in manifest order.
    #[must_use]
    pub fn wide_columns(&self) -> Vec<Cells<'_>> {
        self.wide
            .chunks_exact(self.main_len())
            .map(Cells::Wide)
            .collect()
    }

    /// The eight (H-u) or nine (H-exact) `G_main4` polynomials in manifest order (`2^d_main`
    /// cells each).
    #[must_use]
    pub fn columns(&self) -> &[AlignedVec<i8>] {
        &self.narrow
    }

    /// One polynomial of either main group.
    #[must_use]
    pub fn column(&self, manifest: &CommitmentManifest, column: Column) -> Cells<'_> {
        let (role, position) = manifest.locate(column);
        match role {
            GroupRole::Main16 => {
                let cells = self.main_len();
                Cells::Wide(&self.wide[position * cells..(position + 1) * cells])
            }
            GroupRole::Main => Cells::Narrow(&self.narrow[position]),
            _ => panic!("{} is not a main-domain column", column.name()),
        }
    }

    /// The six bit lanes `b_t` (ML-DSA-44).
    #[must_use]
    pub fn bit_lanes(&self, manifest: &CommitmentManifest) -> [&[i8]; W1_BITS] {
        std::array::from_fn(|t| self.column(manifest, Column::W1Bit(t as u8, 0)).narrow())
    }

    /// The lanes of bit `b_t`, in lane order (one for ML-DSA-44).
    #[must_use]
    pub fn bit_columns(&self, manifest: &CommitmentManifest, t: usize) -> Vec<&[i8]> {
        (0..manifest.shape().lanes_k as u8)
            .map(|lane| self.column(manifest, Column::W1Bit(t as u8, lane)).narrow())
            .collect()
    }

    /// The bridge inputs: per bit `t`, the bits of every record at position
    /// `j + 256 rho + P_pad i` (`P_pad` the padded positions of a signature, the positions
    /// `256 k ..P_pad` zero). For ML-DSA-44 these are the bit columns themselves.
    #[must_use]
    pub fn bridge_lanes(&self, manifest: &CommitmentManifest) -> Vec<Vec<i8>> {
        let shape = manifest.shape();
        let lane_cells = self.lane_cells;
        (0..shape.w1_bits)
            .map(|t| {
                let columns = self.bit_columns(manifest, t);
                let mut flat = vec![0_i8; shape.positions_pad * self.geom.n_star];
                flat.par_chunks_mut(shape.positions_pad).enumerate().for_each(|(i, record)| {
                    for (row, out) in record[..shape.positions].chunks_exact_mut(N).enumerate() {
                        let (lane, base) = shape.row_lane(row);
                        let start = i * lane_cells + base;
                        out.copy_from_slice(&columns[lane][start..start + N]);
                    }
                });
                flat
            })
            .collect()
    }

    /// The four bit lanes of an ML-DSA-65 or ML-DSA-87 table set as one nibble per bridge
    /// position, read straight from the committed bit columns; `None` for six bit lanes.
    #[must_use]
    pub fn bridge_nibble_source(&self, manifest: &CommitmentManifest) -> Option<BridgeNibbles<'_>> {
        let shape = manifest.shape();
        (shape.w1_bits == 4).then(|| BridgeNibbles {
            columns: std::array::from_fn(|t| self.bit_columns(manifest, t)),
            shape,
            lane_cells: self.lane_cells,
            records: self.geom.n_star,
        })
    }

    /// The values `z, u, v` of one ML-DSA-44 record ([`MainTables::record_values_for`]).
    #[must_use]
    pub fn record_values(&self, manifest: &CommitmentManifest, i: usize) -> (VecL, VecK, VecK) {
        self.record_values_for::<Set44>(manifest, i)
    }

    /// The values `z, u, v` of one record recomposed by the relation's value forms (any cell
    /// values); under H-FIPS `u = gamma_2 s + L + 1 - gamma_2 - 2 gamma_2 delta - e` ([`fips_u`]).
    ///
    /// # Panics
    /// If the manifest is of another parameter set than `P`.
    #[must_use]
    pub fn record_values_for<P: ParamSet>(
        &self,
        manifest: &CommitmentManifest,
        i: usize,
    ) -> (P::VecL, P::VecK, P::VecK) {
        assert_eq!(manifest.set(), P::SET, "values of the manifest's parameter set");
        let shape = manifest.shape();
        let lanes = [shape.lanes_l, shape.lanes_k, shape.lanes_k];
        let values = [manifest.z_value(), manifest.u_slot_value(), DigitValue::V];
        let forms: Vec<(ValueForm, Vec<Vec<Cells<'_>>>)> = values
            .iter()
            .zip(lanes)
            .map(|(&value, lanes)| {
                let cells = (0..lanes as u8)
                    .map(|lane| {
                        value_lane_columns(value, lane).into_iter().map(|c| self.column(manifest, c)).collect()
                    })
                    .collect();
                (value.layout(), cells)
            })
            .collect();
        let compose = |k: usize, lane: usize, x: usize| {
            let (form, cells) = &forms[k];
            let digits: Vec<i64> = cells[lane].iter().map(|c| c.value(x)).collect();
            form.compose(&digits)
        };
        let fips: Option<Vec<[Cells<'_>; 3]>> = manifest.is_fips().then(|| {
            (0..shape.lanes_k as u8)
                .map(|lane| FipsColumn::ALL.map(|c| self.column(manifest, Column::Fips(c, lane))))
                .collect()
        });
        let (mut z, mut u, mut v) = (P::zero_vec_l(), P::zero_vec_k(), P::zero_vec_k());
        for row in 0..shape.k.max(shape.l) {
            let (lane, base) = shape.row_lane(row);
            for j in 0..N {
                let x = i * self.lane_cells + base + j;
                if row < shape.l {
                    z[row][j] = compose(0, lane, x);
                }
                if row < shape.k {
                    u[row][j] = match &fips {
                        None => compose(1, lane, x),
                        Some(lanes) => {
                            let [s, delta, e] = &lanes[lane];
                            fips_u_with(P::GAMMA2, compose(1, lane, x), s.value(x), delta.value(x), e.value(x))
                        }
                    };
                    v[row][j] = compose(2, lane, x);
                }
            }
        }
        (z, u, v)
    }

    /// Mutable `G_main4` columns for adversarial tests (the alphabet is not rechecked).
    #[doc(hidden)]
    pub fn columns_mut(&mut self) -> &mut [AlignedVec<i8>] {
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
        let cells = self.main_len();
        &mut self.wide[position * cells + x]
    }
}

/// H-FIPS's `u = gamma_2 s + L + 1 - gamma_2 - 2 gamma_2 delta - e` from `L`, the sign bit
/// `s = (1 + sigma)/2`, `delta` and `e`, any integers.
#[must_use]
pub const fn fips_u(l: i64, s: i64, delta: i64, e: i64) -> i64 {
    fips_u_with(GAMMA2, l, s, delta, e)
}

/// [`fips_u`] with the `gamma_2` of any parameter set.
#[must_use]
pub const fn fips_u_with(gamma2: i64, l: i64, s: i64, delta: i64, e: i64) -> i64 {
    gamma2 * s + l + 1 - gamma2 - 2 * gamma2 * delta - e
}

/// H-FIPS's hint slack as a `D_2` table: `u_h(i)` at index
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
    parity1: Option<Vec<Vec<i8>>>,
    level1: Vec<i32>,
    parities: Vec<Vec<i8>>,
    level2: Vec<i32>,
    level3: Vec<i32>,
    y: Vec<u8>,
    proj: ProjectionTables,
    /// The projection matrix the digits were built under (public, not committed content).
    psi: Vec<u8>,
    /// The cells of `G_3`'s zero polynomials (shared by all of them).
    zeros: Vec<i8>,
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
            && self.proj == other.proj
            && self.multiplicity == other.multiplicity
            && self.digits == other.digits
    }
}

impl Eq for BridgeGroupTables {}

impl BridgeGroupTables {
    /// The honest round-2 tables for the committed bits of `main` and the matrices derived
    /// from the bridge seed, the projection digits of the committed `v` under `psi`
    /// (`projection`), with the multiplicities of every looked-up cell of both rounds.
    pub fn build(
        manifest: &CommitmentManifest,
        main: &MainTables,
        matrices: &BridgeMatrices,
        psi: &[u8],
    ) -> Result<Self, SourceTableError> {
        let lanes = 0..manifest.shape().lanes_k as u8;
        let v_lo: Vec<&[i32]> = lanes.clone().map(|lane| main.column(manifest, Column::V_LO.in_lane(lane)).wide()).collect();
        let v_hi: Vec<&[i32]> = lanes.map(|lane| main.column(manifest, Column::V_HI.in_lane(lane)).wide()).collect();
        let proj = ProjectionTables::build(manifest, &v_lo, &v_hi, psi)?;
        let shape = Shape::production(manifest.set());
        if manifest.set() != Set::S44 {
            // Four bit lanes over the padded bridge inputs, packed as nibbles straight from
            // the bit columns: the packed path, or the general one for inputs that are not
            // bits.
            let lookup = manifest.lookup_shape();
            let slots = |level: usize| lookup.stacks[manifest.carry_stack(level)].slots();
            // The packed path writes the level-3 carries as one slot and the four level-2
            // carries in the first slots of their stack; H-FIPS's hint slack takes the slot
            // after them, as on the general path.
            assert_eq!(slots(3), 1, "the level-3 carries of a four-lane set fill one slot");
            let hint_slack = manifest.is_fips().then(|| hint_slack_table(manifest, main.hint_slack()));
            let slack_fits = hint_slack.as_ref().is_none_or(|slack| {
                slots(2) > 4 && slack.iter().all(|&u| (0..1 << 16).contains(&u))
            });
            let packed = slack_fits.then(|| main.bridge_nibble_source(manifest).map(|src| src.all())).flatten();
            if let Some((nibbles, true)) = packed {
                let direct = parity::compute_direct_four_packed(
                    &shape,
                    &nibbles,
                    matrices,
                    LOOKUP_SHIFT,
                    [slots(1), slots(2)],
                )
                .map_err(SourceTableError::Bridge)?;
                if let Some(mut direct) = direct {
                    if let Some(slack) = &hint_slack {
                        let cells = direct.level2_carries.len() / slots(2);
                        assert_eq!(slack.len(), cells, "the hint slack fills one level-2 slot");
                        for (dst, &u) in direct.level2_carries[4 * cells..5 * cells].iter_mut().zip(slack) {
                            *dst = u as i32 - LOOKUP_SHIFT;
                        }
                    }
                    let mut tables = Self {
                        parity1: Some(direct.y1),
                        level1: direct.level1_carries,
                        parities: direct.y2,
                        level2: direct.level2_carries,
                        level3: direct.level3_carries,
                        y: direct.y,
                        proj,
                        psi: psi.to_vec(),
                        zeros: vec![0; manifest.group(GroupRole::Level3).logical_cells()],
                        multiplicity: Vec::new(),
                        digits: Vec::new(),
                    };
                    tables.recount(manifest, main)?;
                    tables.validate(manifest)?;
                    return Ok(tables);
                }
            }
            let inputs = main.bridge_lanes(manifest);
            let inputs: Vec<&[i8]> = inputs.iter().map(Vec::as_slice).collect();
            let witness = parity::compute(&shape, &inputs, matrices).map_err(SourceTableError::Bridge)?;
            // H-FIPS: the hint slack in the slot after the four level-2 carries.
            return Self::from_witness(manifest, main, witness, hint_slack.into_iter().collect(), proj, psi);
        }
        let bit_lanes = main.bit_lanes(manifest);
        let hint_slack = manifest
            .is_fips()
            .then(|| hint_slack_table(manifest, main.hint_slack()));
        // The packed path runs on every tier (only its bit packing has per-tier kernels);
        // the general path below takes inputs that are not bits.
        let direct = parity::compute_direct(
            &shape,
            &bit_lanes,
            matrices,
            LOOKUP_SHIFT,
            hint_slack.as_deref(),
        )
        .map_err(SourceTableError::Bridge)?;
        if let Some(direct) = direct {
            let mut tables = Self {
                parity1: Some(direct.y1.into()),
                level1: direct.level1_carries,
                parities: direct.y2,
                level2: direct.level2_carries,
                level3: direct.level3_carries,
                y: direct.y,
                proj,
                psi: psi.to_vec(),
                zeros: vec![0; manifest.group(GroupRole::Level3).logical_cells()],
                multiplicity: Vec::new(),
                digits: Vec::new(),
            };
            tables.recount(manifest, main)?;
            tables.validate(manifest)?;
            return Ok(tables);
        }
        let witness = parity::compute(&shape, &bit_lanes, matrices)
            .map_err(SourceTableError::Bridge)?;
        Self::from_witness(manifest, main, witness, hint_slack.into_iter().collect(), proj, psi)
    }

    /// The tables of a general-path parity witness: the carries stacked in their lookup
    /// stacks' slots (`extra_level2` after the level-2 carries: H-FIPS's hint slack), the
    /// level-1 parities cached.
    fn from_witness(
        manifest: &CommitmentManifest,
        main: &MainTables,
        witness: parity::ParityWitness,
        extra_level2: Vec<Vec<i64>>,
        proj: ProjectionTables,
        psi: &[u8],
    ) -> Result<Self, SourceTableError> {
        let narrow_bits = |table: &Vec<i64>| -> Result<Vec<i8>, SourceTableError> {
            table
                .iter()
                .map(|&bit| i8::try_from(bit).map_err(|_| SourceTableError::Encode("parity")))
                .collect()
        };
        let lookup = manifest.lookup_shape();
        let slots = |level: usize| lookup.stacks[manifest.carry_stack(level)].slots();
        let mut level2_columns = witness.kappa2.clone();
        level2_columns.extend(extra_level2);
        let parity1: Vec<Vec<i8>> = witness.y1.iter().map(narrow_bits).collect::<Result<_, _>>()?;
        let mut tables = Self {
            parity1: Some(parity1),
            level1: stack_carries(&witness.kappa1, slots(1))?,
            parities: witness.y2.iter().map(narrow_bits).collect::<Result<_, _>>()?,
            level2: stack_carries(&level2_columns, slots(2))?,
            level3: stack_carries(std::slice::from_ref(&witness.kappa3), slots(3))?,
            y: witness.y_bits(),
            proj,
            psi: psi.to_vec(),
            zeros: vec![0; manifest.group(GroupRole::Level3).logical_cells()],
            multiplicity: Vec::new(),
            digits: Vec::new(),
        };
        tables.recount(manifest, main)?;
        tables.validate(manifest)?;
        Ok(tables)
    }

    /// Recompute the multiplicities `m` and their digits from the current cells of both
    /// rounds (`lookup::multiplicities`); an out-of-table cell, or a count above the seven
    /// digits' capacity (`lookup::MULTIPLICITY_CAPACITY`), is an honest abort.
    pub fn recount(&mut self, manifest: &CommitmentManifest, main: &MainTables) -> Result<(), SourceTableError> {
        let m = {
            let stacks = lookup_stacks(manifest, main, self).map_err(SourceTableError::Lookup)?;
            multiplicities(&stacks, &[Table::T16]).map_err(SourceTableError::Lookup)?
        };
        let m = m.into_iter().next().expect("one table");
        self.digits = multiplicity_digits(&m).map_err(SourceTableError::Lookup)?;
        self.multiplicity = m;
        Ok(())
    }

    /// Set `m` (and its digits) directly (adversarial tests and [`Self::recount`]).
    ///
    /// # Panics
    /// If a count does not fit seven balanced base-16 digits.
    #[doc(hidden)]
    pub fn set_multiplicity(&mut self, m: Vec<i32>) {
        self.digits = multiplicity_digits(&m).expect("a count within seven balanced base-16 digits");
        self.multiplicity = m;
    }

    /// The shape and alphabet self-check of the five round-2 groups (`G_1`, `G_2`, `G_3`,
    /// `G_k`, `G_m`).
    pub fn validate(&self, manifest: &CommitmentManifest) -> Result<(), SourceTableError> {
        for role in [GroupRole::Level1, GroupRole::Level2, GroupRole::Level3, GroupRole::Proj, GroupRole::Mult] {
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
        let count = |level: usize| manifest.stack_columns(manifest.carry_stack(level)).len();
        match role {
            GroupRole::Level1 => carries(&self.level1, cells, count(1)),
            GroupRole::Level2 => self
                .parities
                .iter()
                .map(|p| Cells::Narrow(p))
                .chain(carries(&self.level2, cells, count(2)))
                .collect(),
            GroupRole::Level3 => {
                let zeros = manifest.group(role).polynomial_count() - 1;
                carries(&self.level3, cells, 1)
                    .into_iter()
                    .chain(std::iter::repeat_n(Cells::Narrow(&self.zeros), zeros))
                    .collect()
            }
            GroupRole::Proj => self.proj.columns(),
            GroupRole::Mult => self.digits.iter().map(|d| Cells::Narrow(d)).collect(),
            GroupRole::Main16 | GroupRole::Main => panic!("a round-1 group"),
        }
    }

    /// The stacked carry lane of level `1..=3` (lookup stack
    /// [`CommitmentManifest::carry_stack`]).
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
    pub(crate) fn parity1_cached(&self) -> Option<&Vec<Vec<i8>>> {
        self.parity1.as_ref()
    }

    /// Move the cached parities out (the honest driver's single use), so they are neither
    /// copied on the PIOP branch nor kept twice through the stages.
    pub(crate) fn take_parity1(&mut self) -> Option<Vec<Vec<i8>>> {
        self.parity1.take()
    }

    /// Forget the cached level-1 parities (the tables were changed after `build`).
    pub(crate) fn drop_parity1_cache(&mut self) {
        self.parity1 = None;
    }

    /// The projection digits (`G_k`).
    #[must_use]
    pub fn projection(&self) -> &ProjectionTables {
        &self.proj
    }

    /// The projection matrix `Psi` the digits were built under.
    #[must_use]
    pub fn psi(&self) -> &[u8] {
        &self.psi
    }

    /// Mutable projection digits (adversarial tests; `m` is not recounted).
    #[doc(hidden)]
    pub fn projection_mut(&mut self) -> &mut ProjectionTables {
        &mut self.proj
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

    /// `Y_sigma`, the 4,096 bits of segment `sigma` (final-vector payload, 512 bytes packed).
    #[must_use]
    pub fn segment_y(&self, sigma: usize) -> &[u8] {
        &self.y[sigma * M3..(sigma + 1) * M3]
    }

    /// The carries `kappa^(level)` as integers (`kappa = K + 2^15`), per lane.
    #[must_use]
    pub fn carry_values(&self, manifest: &CommitmentManifest, level: usize) -> Vec<Vec<i64>> {
        let role = [GroupRole::Level1, GroupRole::Level2, GroupRole::Level3][level - 1];
        let cells = manifest.group(role).logical_cells();
        let count = if level == 3 { 1 } else { manifest.shape().w1_bits };
        self.carries(level)
            .chunks_exact(cells)
            .take(count)
            .map(|slot| slot.iter().map(|&k| i64::from(k + LOOKUP_SHIFT)).collect())
            .collect()
    }
}

/// The lookup stacks of the batch, borrowed from the tables: the
/// main-domain stacks (consecutive runs of the stacked `G_main16` lane: `[Z_lo | U_lo]`, its
/// first half, for ML-DSA-44; runs of 8, 2, 1 or 8, 4, 2, 1 of the `Z_lo, U_lo` lanes for
/// ML-DSA-65 and ML-DSA-87), the stacked carries of levels 1 to 3 and `[K_0 | K_1]` on the
/// projection domain.
pub fn lookup_stacks<'a>(
    manifest: &CommitmentManifest,
    main: &'a MainTables,
    bridge: &'a BridgeGroupTables,
) -> Result<Vec<LookupStack<'a>>, LookupError> {
    let shape = manifest.lookup_shape();
    let mut lanes: Vec<&[i32]> = Vec::with_capacity(shape.stacks.len());
    let mut start = 0;
    for stack in &shape.stacks[..manifest.main_stacks()] {
        debug_assert_eq!(stack.slots(), stack.columns, "main-domain stacks have no padding slots");
        let end = start + stack.leaves();
        lanes.push(&main.wide()[start..end]);
        start = end;
    }
    lanes.extend([bridge.carries(1), bridge.carries(2), bridge.carries(3), &bridge.projection().stacked]);
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

    /// The lookup stacks.
    pub fn lookup_stacks(&self) -> Result<Vec<LookupStack<'a>>, LookupError> {
        lookup_stacks(self.manifest, self.main, self.bridge)
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
                .chain((0..W1_BITS as u8).map(|t| Column::W1Bit(t, 0)))
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

#[cfg(test)]
mod lane_tests {
    use super::*;
    use agg_witness::bridge::SparseMatrix;
    use agg_witness::params::{Set65, Set87};
    use jolt_field::Ring;

    /// The shift forms decompose as the forms do, at the alphabet edges and beyond.
    #[test]
    fn shift_forms_match_the_value_forms() {
        for set in [Set::S44, Set::S65, Set::S87] {
            for relation in [Relation::Hu, Relation::HExact, Relation::HFips] {
                let n = if set == Set::S44 { 16 } else { 64 };
                let manifest = CommitmentManifest::for_set(set, n, relation).unwrap();
                let encoder = Encoder::new(&manifest);
                let shift = encoder.shift.as_ref().expect("shift forms for every set and relation");
                for ((form, _, _), (fast, bounds)) in encoder.values.iter().zip(shift) {
                    let mut probes: Vec<i32> = vec![i32::MIN, i32::MAX, 0, -1, 1];
                    let edges = [1_i32 << 15, 1 << 16, 1 << 17, 1 << 19, 29_696, 63_079, 95_232, 98_225, 130_816,
                        130_915, 130_993, 190_465, 261_888, 523_267, 523_777, 524_091, 1 << 30];
                    for edge in edges {
                        for t in 0..8 {
                            for d in -2..=2 {
                                probes.extend([edge.wrapping_mul(t).wrapping_add(d), edge.wrapping_mul(-t).wrapping_add(d)]);
                            }
                        }
                    }
                    let mut x = 0x9e37_79b9_i64;
                    for _ in 0..4096 {
                        x = (x * 6_364_136_223_846_793 + 1) % (1 << 33);
                        probes.push((x - (1 << 32) / 2) as i32);
                        probes.push(((x % (1 << 21)) - (1 << 20)) as i32);
                    }
                    probes.resize(probes.len().next_multiple_of(N), 0);
                    for chunk in probes.chunks_exact(N) {
                        let xs: &[i32; N] = chunk.try_into().unwrap();
                        let mut got = [[0_i32; N]; SHIFT_DIGITS];
                        fast.decompose_poly(xs, &mut got);
                        for (j, &x) in xs.iter().enumerate() {
                            let mut want = [0_i64; MAX_DIGITS];
                            form.decompose_into(i64::from(x), &mut want[..form.len()]);
                            let got_j: Vec<i64> = got[..form.len()].iter().map(|d| i64::from(d[j])).collect();
                            let inside = got[..form.len()].iter().zip(bounds).all(|(d, &(min, max))| min <= d[j] && d[j] <= max);
                            assert_eq!(inside, form.in_alphabets(&want[..form.len()]), "{set:?} {relation:?} x = {x}");
                            if (-(1 << 29)..=1 << 29).contains(&x) || inside {
                                assert_eq!(got_j[..], want[..form.len()], "{set:?} {relation:?} x = {x}");
                            }
                        }
                    }
                }
            }
        }
    }

    /// The row form of the H-FIPS witness agrees with the per-coefficient one on every
    /// `(u, w_1')` with `|u| <= B_u + 2` and `w_1' < 2^w1_bits`: it accepts exactly the
    /// coefficients with `|u| <= B_u`, `w_1' < m_w` that have a witness, with that witness.
    fn fips_row_matches_the_coefficients<P: ParamSet>() {
        let b_u = P::B_U as i32;
        let mut accepted = Vec::new();
        for w1 in 0..1_u8 << P::W1_BITS {
            for u in -b_u - 2..=b_u + 2 {
                let (uj, wj) = (i64::from(u), i64::from(w1));
                let want = if uj.abs() > P::B_U || wj >= P::M_W {
                    None
                } else if P::SET == Set::S44 {
                    fips_coefficient(uj, wj).map(|c| (c.l(), (1 + c.sigma) / 2, c.delta, c.e))
                } else {
                    coefficient_of::<P>(uj, wj).map(|c| (c.l, (1 + c.sigma) / 2, c.delta, c.e))
                };
                let (ok, l, s, delta, e) = fips_cell::<P>(u, w1);
                let got = ok.then_some((i64::from(l), i64::from(s), i64::from(delta), i64::from(e)));
                assert_eq!(got, want, "{:?} u = {u} w1 = {w1}", P::SET);
                if ok {
                    accepted.push((u, w1, got.unwrap()));
                }
            }
        }
        // The row form on accepted coefficients, and with one rejected coefficient.
        for chunk in accepted.chunks_exact(N).step_by(97) {
            let u: [i32; N] = std::array::from_fn(|j| chunk[j].0);
            let mut w1: [u8; N] = std::array::from_fn(|j| chunk[j].1);
            let (mut l, mut cells) = ([0_i32; N], [[0_i8; N]; 3]);
            let weight = fips_row::<P>(&u, &w1, &mut l, &mut cells).expect("accepted coefficients");
            assert_eq!(weight, chunk.iter().map(|c| c.2 .2 * c.2 .2).sum::<i64>());
            for (j, c) in chunk.iter().enumerate() {
                let cell = (i64::from(l[j]), i64::from(cells[0][j]), i64::from(cells[1][j]), i64::from(cells[2][j]));
                assert_eq!(cell, c.2);
            }
            w1[N / 2] = P::M_W as u8;
            assert_eq!(fips_row::<P>(&u, &w1, &mut l, &mut cells), None);
        }
    }

    #[test]
    fn fips_rows_match_the_coefficients() {
        fips_row_matches_the_coefficients::<Set44>();
        fips_row_matches_the_coefficients::<agg_witness::params::Set65>();
        fips_row_matches_the_coefficients::<agg_witness::params::Set87>();
    }

    /// Five live records of `P` (eight with padding), their instance, signatures and tables.
    fn batch<P: ParamSet>(seed: u64) -> (CommitmentManifest, Vec<agg_witness::gen::Record>, PublicInstance<P>, MainTables) {
        let records = agg_witness::gen::records_for::<P>(seed, 5);
        let instance = PublicInstance::<P>::from_records_for(&records).unwrap();
        let signatures: Vec<&[u8]> = records.iter().map(|r| r.sig.as_slice()).collect();
        let manifest = CommitmentManifest::for_set(P::SET, 5, Relation::Hu).unwrap();
        let main = MainTables::build_for::<P>(&manifest, &instance, &signatures).unwrap();
        (manifest, records, instance, main)
    }

    fn main_tables_follow_the_lane_layout<P: ParamSet>() {
        let (manifest, records, instance, main) = batch::<P>(0x65);
        let shape = manifest.shape();
        let (k, l) = (P::K, P::L);
        let cells = 8 * 256;
        assert_eq!((main.set(), main.main_len()), (P::SET, cells));
        assert_eq!(main.wide().len(), (l + 3 * k) * cells);
        assert_eq!(main.columns().len(), 2 * l + 2 * k + 4 * k);
        // Padded records hold the digits of zero in every lane.
        let padded = 6 * 256 + 17;
        for lane in 0..k as u8 {
            let value = |column: Column| main.column(&manifest, column.in_lane(lane)).value(padded);
            assert_eq!(value(Column::lane_digit(DigitValue::Ud, 0, 0)), 32_257);
            assert_eq!(value(Column::lane_digit(DigitValue::Ud, 1, 0)), 3);
            assert_eq!(value(Column::lane_digit(DigitValue::Ud, 2, 0)), 0);
            assert_eq!((value(Column::V_LO), value(Column::V_HI)), (-32_768, 0));
            for t in 0..4 {
                assert_eq!(value(Column::W1Bit(t, 0)), 0);
            }
        }
        for lane in 0..l as u8 {
            let value = |digit| main.column(&manifest, Column::lane_digit(DigitValue::Zd, digit, lane)).value(padded);
            assert_eq!((value(0), value(1), value(2)), (-32_768, -4, 1));
        }
        // The cells recompose to the derived values, record by record and lane by lane.
        let mut values = Vec::new();
        for (i, (record, signed)) in instance.records.iter().zip(&records).enumerate() {
            let w = agg_witness::derive::derive_for::<P>(&record.key, &record.mu, &signed.sig).unwrap();
            let (z, u, v) = main.record_values_for::<P>(&manifest, i);
            assert_eq!((&z, &u, &v), (&w.z, &w.u, &w.v), "record {i}");
            for row in 0..k {
                for j in [0, 99, 255] {
                    let bits: u8 = (0..4)
                        .map(|t| (main.column(&manifest, Column::W1Bit(t, row as u8)).value(i * 256 + j) as u8) << t)
                        .sum();
                    assert_eq!(bits, w.w1[row][j]);
                }
            }
            values.push(RecordValues::<P> { z: w.z, u: w.u, v: w.v, w1: w.w1 });
        }
        assert_eq!(MainTables::from_values(&manifest, &values).unwrap(), main);
        // u = +-523,777 and z in [-2^19, 2^19) encode; one past either edge is an honest abort.
        for (u, ok) in [(523_777, true), (-523_777, true), (523_778, false), (-523_778, false)] {
            let mut edge = values.clone();
            edge[2].u[k - 1][9] = u;
            let built = MainTables::from_values(&manifest, &edge);
            assert_eq!(built.is_ok(), ok, "u = {u}");
            if let Ok(tables) = built {
                assert_eq!(tables.record_values_for::<P>(&manifest, 2).1[k - 1][9], u);
            }
        }
        for (z, ok) in [(-(1 << 19), true), ((1 << 19) - 1, true), (1 << 19, false), (-(1 << 19) - 1, false)] {
            let mut edge = values.clone();
            edge[1].z[l - 1][3] = z;
            let built = MainTables::from_values(&manifest, &edge);
            assert_eq!(built.is_ok(), ok, "z = {z}");
            if let Ok(tables) = built {
                assert_eq!(tables.record_values_for::<P>(&manifest, 1).0[l - 1][3], z);
            }
        }
        let mut wide_w1 = values.clone();
        wide_w1[0].w1[0][0] = 16;
        assert_eq!(MainTables::from_values(&manifest, &wide_w1), Err(SourceTableError::Encode("w1' above four bits")));
        // A base-8 middle digit outside [-4, 3] fails the self-check at its lane's column.
        let u_mid = manifest.locate(Column::lane_digit(DigitValue::Ud, 1, 2)).1;
        let mut bad = main.clone();
        bad.columns_mut()[u_mid][300] = 4;
        assert_eq!(
            bad.validate(&manifest).unwrap_err(),
            SourceTableError::Alphabet { group: GroupRole::Main, column: u_mid, cell: 300, value: 4 }
        );
        // The bridge inputs put row rho of record i at j + 256 rho + P_pad i.
        let inputs = main.bridge_lanes(&manifest);
        assert_eq!(inputs.len(), 4);
        for (t, input) in inputs.iter().enumerate() {
            assert_eq!(input.len(), 8 * shape.positions_pad);
            let columns = main.bit_columns(&manifest, t);
            for (i, row, j) in [(0, 0, 0), (3, k - 1, 200), (4, 2, 255)] {
                assert_eq!(input[i * shape.positions_pad + 256 * row + j], columns[row][i * 256 + j]);
            }
            assert!(input[shape.positions..shape.positions_pad].iter().all(|&b| b == 0));
        }
        // The nibble source packs the same inputs; a non-bit cell keeps its low bit and clears
        // the all-bits flag.
        let nibbles_of = |tables: &MainTables| {
            let lanes = tables.bridge_lanes(&manifest);
            (0..lanes[0].len())
                .map(|x| (0..4).map(|t| ((lanes[t][x] as u8) & 1) << t).sum::<u8>())
                .collect::<Vec<u8>>()
        };
        let (nibbles, all_bits) = main.bridge_nibble_source(&manifest).expect("four lanes").all();
        assert!(all_bits);
        assert_eq!(nibbles, nibbles_of(&main));
        let bit_col = manifest.locate(Column::W1Bit(2, 1)).1;
        let mut not_bit = main.clone();
        not_bit.columns_mut()[bit_col][300] = 3;
        let (got, all_bits) = not_bit.bridge_nibble_source(&manifest).expect("four lanes").all();
        assert!(!all_bits);
        assert_eq!(got, nibbles_of(&not_bit));
        assert_ne!(got, nibbles);

        // Round 2 under the four-lane shape: four carry lanes in four slots on D_1 and D_2.
        let bridge_shape = Shape::production(P::SET);
        let level = |t: usize| bridge_shape.level(t);
        let matrices = BridgeMatrices {
            m1: SparseMatrix::random_for_tests(1, level(1).rows, level(1).cols, level(1).k),
            m2: SparseMatrix::random_for_tests(2, level(2).rows, level(2).cols, level(2).k),
            m3: SparseMatrix::random_for_tests(3, level(3).rows, level(3).cols, level(3).k),
        };
        let psi = crate::projection::test_psi_for(P::SET, 9);
        let bridge = BridgeGroupTables::build(&manifest, &main, &matrices, &psi).unwrap();
        // The packed four-lane path (taken: the lookup stacks have four slots or more) builds
        // the general path's tables.
        let views: Vec<&[i8]> = inputs.iter().map(Vec::as_slice).collect();
        let lookup = manifest.lookup_shape();
        let slots = |level: usize| lookup.stacks[manifest.carry_stack(level)].slots();
        let four: [&[i8]; 4] = std::array::from_fn(|t| views[t]);
        assert!(parity::compute_direct_four(&bridge_shape, &four, &matrices, LOOKUP_SHIFT, [slots(1), slots(2)])
            .unwrap()
            .is_some());
        let general = BridgeGroupTables::from_witness(
            &manifest,
            &main,
            parity::compute(&bridge_shape, &views, &matrices).unwrap(),
            Vec::new(),
            bridge.proj.clone(),
            &psi,
        )
        .unwrap();
        assert_eq!(bridge, general);
        assert_eq!(bridge.parity1_cached(), general.parity1_cached());
        let counts: Vec<usize> = [GroupRole::Level1, GroupRole::Level2, GroupRole::Level3, GroupRole::Proj, GroupRole::Mult]
            .iter()
            .map(|&role| bridge.group(&manifest, role).len())
            .collect();
        assert_eq!(counts, [4, 8, 8, 5, 7]);
        assert_eq!(bridge.carries(1).len(), 4 * manifest.group(GroupRole::Level1).logical_cells());
        assert_eq!(bridge.carry_values(&manifest, 2).len(), 4);
        assert_eq!(bridge.public_y().len(), 1 << 12);
        let witness = parity::compute(
            &bridge_shape,
            &inputs.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            &matrices,
        )
        .unwrap();
        assert_eq!(bridge.carry_values(&manifest, 1), witness.kappa1);
        assert_eq!(bridge.public_y(), witness.y_bits());
        // The lookup stacks tile the looked-up lanes; m counts every leaf.
        let shape = manifest.lookup_shape();
        let stacks = lookup_stacks(&manifest, &main, &bridge).unwrap();
        assert_eq!(stacks.len(), shape.stacks.len());
        let leaves: usize = shape.stacks.iter().map(|s| s.leaves()).sum();
        let main_leaves: usize = shape.stacks[..manifest.main_stacks()].iter().map(|s| s.leaves()).sum();
        assert_eq!(main_leaves, (l + k) * cells);
        let counted: i64 = bridge.multiplicity().iter().map(|&m| i64::from(m)).sum();
        assert_eq!(counted, leaves as i64);
        // The projection digits are those of the committed v under Psi.
        let tables = SourceTables { manifest: &manifest, main: &main, bridge: &bridge };
        assert_eq!(tables.group(GroupRole::Proj).len(), 5);
        let (_, _, v) = main.record_values_for::<P>(&manifest, 3);
        let row_bytes = shape_positions_bytes(&manifest);
        let blocks = manifest.shape().proj_blocks;
        let per_block = manifest.shape().positions / blocks;
        let direct: i64 = (0..per_block)
            .filter(|&pos| crate::projection::psi_bit_in(&psi, row_bytes, 7, pos))
            .map(|pos| v[pos / 256][pos % 256])
            .sum();
        assert_eq!(bridge.projection().value(7 + 128 * blocks * 3), E::from_i64(direct));
    }

    fn shape_positions_bytes(manifest: &CommitmentManifest) -> usize {
        manifest.shape().positions / 8
    }

    #[test]
    fn ml_dsa_65_tables_follow_the_lane_layout() {
        main_tables_follow_the_lane_layout::<Set65>();
    }

    #[test]
    fn ml_dsa_87_tables_follow_the_lane_layout() {
        main_tables_follow_the_lane_layout::<Set87>();
    }

    /// The generic path builds ML-DSA-44's tables byte for byte, and its bridge inputs are the
    /// bit columns.
    #[test]
    fn the_generic_path_reproduces_ml_dsa_44() {
        let records = agg_witness::gen::records(0x44, 5);
        let instance = PublicInstance::from_records(&records).unwrap();
        let signatures: Vec<&[u8]> = records.iter().map(|r| r.sig.as_slice()).collect();
        for relation in Relation::ALL {
            let manifest = CommitmentManifest::for_relation(5, relation).unwrap();
            let fast = MainTables::build(&manifest, &instance, &signatures).unwrap();
            assert_eq!(MainTables::build_for::<Set44>(&manifest, &instance, &signatures).unwrap(), fast);
            let inputs = fast.bridge_lanes(&manifest);
            let bits = fast.bit_lanes(&manifest);
            assert_eq!(inputs.len(), W1_BITS);
            for (input, bits) in inputs.iter().zip(bits) {
                assert_eq!(input.as_slice(), bits);
            }
        }
    }
}
