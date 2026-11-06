//! Generic sum-check kernels: narrow source tables read through codes and lookup tables
//! for the first active rounds, and groups of batched instances over one local domain.
//!
//! Mirrors the shape of Aerie's kernels (`aerie-prover/src/protocol/kernels/`): every
//! committed table stays an `i8` array (with one byte of pair code per pair of cells)
//! until the third active round, and only the eighth-size folded table is lifted to `K`.
//! A [`Group`] holds the lanes of one local domain, an optional shared equality factor
//! (Akita's `GruenSplitEq`, so the equality table of the remaining variables is never
//! materialized) and the gamma-weighted [`Term`]s of the instances batched on it. The round
//! message of a group is the compressed ordinary message of Aerie: the coefficients of
//! the degree-`d` round polynomial without the linear one, which the verifier recovers from
//! the running claim (`g(0) + g(1) = claim`).
//!
//! Instances on a smaller domain than the stage are inactive in the leading rounds, where
//! the message is the constant `claim / 2`, and are lifted by `2^-inactive` (Aerie).
//!
//! Round schedule (every path computes the same round
//! polynomials, so the proof bytes do not depend on it):
//!
//! - The terms are compiled into a [`plan::Plan`]: one polynomial per lane, bilinear
//!   products, weighted sums `w (sum c_l v_l + c)` and residual generic terms, all
//!   evaluated in coefficient form at `v = a + X d`.
//! - An `i8` column is *coded* when its cells fall into at most 16 classes under the maps of
//!   the lanes reading it (8 for a digit read through `z (z + 1)`, 16 under the identity, 2
//!   for a bit). Pieces on coded lanes are *bucketed* by pair or quartet patterns of their
//!   column (or of a few columns for a joint term); the rest (weighted sums, wide cells, the
//!   `i32` fallback) are evaluated per item.
//! - Level 0 (first active round): when the quartets outnumber the largest quartet space
//!   by `BUCKET_MIN_RATIO`, one pass accumulates quartet masses with the round-1 equality
//!   weights and serves rounds 0 and 1 (Aerie's bivariate prefix); else pair masses.
//! - Bind 0: coded lanes become pair lookup tables (no storage per cell); bind 1: quartet
//!   lookup tables (with Taylor tables of their polynomials) when the quartet space is at
//!   most 4,096, else the pair table with the second challenge; bind 2 materializes every
//!   lane in one row-major `K` table at an eighth of the domain; later rounds are dense.

#[allow(unsafe_code)]
pub(crate) mod buf;
mod arena;
pub(crate) use arena::{lock_arena, SharedArena, TableArena};
mod coded;
mod field;
mod layer;
mod plan;
mod plane;
mod tail;
#[cfg(test)]
mod tests;

use akita_algebra::split_eq::GruenSplitEq;
use jolt_field::{Field, Fold, One, Ring, Unreduced, Zero};
use jolt_poly::UnivariatePoly;
use rayon::prelude::*;

pub(crate) use self::field::{dot3, fm, fm2, fmi, residue, Acc};
#[cfg(test)]
pub(crate) use self::field::mad_int;
pub use self::layer::{IntPairGroup, LayerGroup, LayerTree, LinearIntGroup, WeightIntGroup};
pub use self::plane::{
    LayerGroup as PlaneLayerGroup, LayerTree as PlaneLayerTree, LeafGroup as PlaneLeafGroup,
    LinearIntGroup as PlaneLinearIntGroup, WeightIntGroup as PlaneWeightIntGroup,
};
use self::coded::{code_column, masses, Column, ItemWeights, SegmentShape};
use self::field::{fold, lift, mad, plain_sum, split_sum, MIN_PARALLEL_ITEMS};
use self::plan::{Pieces, Plan};
use self::tail::{Tail, TailTerms};
use crate::OpeningField as E;

/// The layout of the sum-check kernel tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// Row-major tables and the dedicated groups over integer rows.
    Row,
    /// Lane-major `K` planes and the fused passes.
    Plane,
}

/// The layout this process uses.
///
/// `AGG_PIOP_LAYOUT=plane|row` names one; unset selects planes whenever a vector
/// backend of `agg-simd` is selected (IFMA, AVX-512F or AVX2), and rows on the portable
/// backend, where the row-major groups' scalar arithmetic is the faster form.
///
/// # Panics
/// When `AGG_PIOP_LAYOUT` names an unknown layout.
#[must_use]
pub fn layout() -> Layout {
    use std::sync::OnceLock;
    static LAYOUT: OnceLock<Layout> = OnceLock::new();
    *LAYOUT.get_or_init(|| {
        match std::env::var("AGG_PIOP_LAYOUT")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .as_deref()
        {
            Some("plane") => Layout::Plane,
            Some("row") => Layout::Row,
            Some(other) => panic!("invalid AGG_PIOP_LAYOUT '{other}': expected plane or row"),
            None => {
                use agg_simd::detect::{backend, Backend};
                match backend() {
                    Backend::Avx512 | Backend::Avx512F | Backend::Avx2 => Layout::Plane,
                    Backend::Portable => Layout::Row,
                }
            }
        }
    })
}

/// Bucketing pays only when the items outnumber the pattern space by this factor
/// (Aerie's `BUCKET_MIN_RATIO`).
const BUCKET_MIN_RATIO: usize = 4;
/// Largest joint pair space of a bucketed multi-column term.
const MAX_JOINT_PAIRS: usize = 1 << 12;
/// Largest quartet space of a segment in the two-round (bivariate) prefix.
const MAX_QUARTET_SPACE: usize = 1 << 16;
/// `C(i, j)` for `i, j < 6`.
const BINOMIAL: [[i64; 6]; 6] = [
    [1, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0],
    [1, 2, 1, 0, 0, 0],
    [1, 3, 3, 1, 0, 0],
    [1, 4, 6, 4, 1, 0],
    [1, 5, 10, 10, 5, 1],
];
/// Largest quartet space of a quartet lookup table (64 KiB per lane).
const MAX_QUARTET_LUT: usize = 1 << 12;

/// A read-only integer or field table, the state of a lane before its first fold.
#[derive(Clone, Copy, Debug)]
pub enum Source<'a> {
    /// A borrowed interval of canonical extension planes.
    PlaneSlice((&'a [u64], &'a [u64])),
    /// Canonical extension coordinate planes, produced directly by a public weight builder.
    Planes(&'a agg_simd::buf::KPlanes),
    /// Committed `i8` cells seen through a per-cell integer map (`map[cell as u8]`).
    Bytes {
        /// The cells.
        cells: &'a [i8],
        /// The map applied to every cell (identity for a committed column).
        map: &'a [i64; 256],
    },
    /// A small-integer table (the virtual parities `y^(1)` when a value exceeds `i8`).
    Ints(&'a [i32]),
    /// A materialized field table (a public weight).
    Field(&'a [E]),
}

/// The identity map of a committed column.
pub static IDENTITY_MAP: [i64; 256] = identity_map();

const fn identity_map() -> [i64; 256] {
    let mut map = [0_i64; 256];
    let mut byte = 0;
    while byte < 256 {
        map[byte] = (byte as u8 as i8) as i64;
        byte += 1;
    }
    map
}

/// The map `cell -> f(cell)` for a function of the digit value.
#[must_use]
pub fn digit_map(f: impl Fn(i64) -> i64) -> [i64; 256] {
    std::array::from_fn(|byte| f(i64::from(byte as u8 as i8)))
}

impl Source<'_> {
    /// Number of cells.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Bytes { cells, .. } => cells.len(),
            Self::Ints(cells) => cells.len(),
            Self::Field(cells) => cells.len(),
            Self::Planes(cells) => cells.rows,
            Self::PlaneSlice(cells) => cells.0.len(),
        }
    }

    /// True when empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The lifted value of cell `index`.
    #[inline]
    #[must_use]
    pub fn value(&self, index: usize) -> E {
        match self {
            Self::Bytes { cells, map } => E::from_i64(map[cells[index] as u8 as usize]),
            Self::Ints(cells) => E::from_i64(i64::from(cells[index])),
            Self::Field(cells) => cells[index],
            Self::Planes(cells) => cells.get(index),
            Self::PlaneSlice(cells) => agg_simd::scalar::get(*cells, index),
        }
    }

    /// The table folded by `challenge` on its low variable: `v(0, y) + r (v(1, y) - v(0, y))`.
    fn fold(&self, challenge: E) -> Vec<E> {
        let half = self.len() / 2;
        let context = E::precompute(challenge);
        let fold_pair = |k: usize| match self {
            Self::Bytes { cells, map } => {
                let low = map[cells[2 * k] as u8 as usize];
                let high = map[cells[2 * k + 1] as u8 as usize];
                E::from_i64(low) + fmi(challenge, high - low)
            }
            Self::Ints(cells) => {
                let (low, high) = (i64::from(cells[2 * k]), i64::from(cells[2 * k + 1]));
                E::from_i64(low) + fmi(challenge, high - low)
            }
            Self::Field(cells) => fold(&context, cells[2 * k], cells[2 * k + 1]),
            Self::Planes(cells) => fold(&context, cells.get(2 * k), cells.get(2 * k + 1)),
            Self::PlaneSlice(cells) => fold(
                &context,
                agg_simd::scalar::get(*cells, 2 * k),
                agg_simd::scalar::get(*cells, 2 * k + 1),
            ),
        };
        if half >= MIN_PARALLEL_ITEMS {
            (0..half).into_par_iter().map(fold_pair).collect()
        } else {
            (0..half).map(fold_pair).collect()
        }
    }
}

/// One multilinear table being bound variable by variable (the input of a [`Group`]).
#[derive(Clone, Debug)]
pub enum Lane<'a> {
    /// Not yet folded: the narrow source.
    Source(Source<'a>),
    /// A materialized table.
    Folded(Vec<E>),
}

impl<'a> Lane<'a> {
    /// A lane over a committed column.
    #[must_use]
    pub fn column(cells: &'a [i8]) -> Self {
        Self::Source(Source::Bytes {
            cells,
            map: &IDENTITY_MAP,
        })
    }

    /// A lane over a mapped column.
    #[must_use]
    pub fn mapped(cells: &'a [i8], map: &'a [i64; 256]) -> Self {
        Self::Source(Source::Bytes { cells, map })
    }

    /// Cells remaining.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Source(source) => source.len(),
            Self::Folded(cells) => cells.len(),
        }
    }

    /// True when empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The value at `index` of the current (partially bound) table.
    #[inline]
    #[must_use]
    pub fn value(&self, index: usize) -> E {
        match self {
            Self::Source(source) => source.value(index),
            Self::Folded(cells) => cells[index],
        }
    }

    /// Bind the low variable to `challenge`.
    pub fn fold(&mut self, challenge: E) {
        let folded = match self {
            Self::Source(source) => source.fold(challenge),
            Self::Folded(cells) => {
                let context = E::precompute(challenge);
                cells
                    .chunks_exact(2)
                    .map(|pair| fold(&context, pair[0], pair[1]))
                    .collect()
            }
        };
        *self = Self::Folded(folded);
    }

    /// The fully bound value (the multilinear extension at the stage point).
    #[must_use]
    pub fn terminal(&self) -> E {
        assert_eq!(self.len(), 1, "lane fully bound");
        self.value(0)
    }

    /// True for a lane over field values (a weight or a materialized table).
    fn is_field(&self) -> bool {
        matches!(
            self,
            Self::Source(Source::Field(_) | Source::Planes(_) | Source::PlaneSlice(_))
                | Self::Folded(_)
        )
    }
}

/// The leaf predicate of a digit alphabet, as a polynomial in `P_Z`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafPoly {
    /// `Q_4(P) = P (P - 2)`.
    Q4,
    /// `Q_6(P) = P (P - 2)(P - 6)`: the six-valued digit `[-3, 2]` (relation H-u's `U_hi`).
    Q6,
    /// `Q_8(P) = Q_16^lo(P) = P (P - 2)(P - 6)(P - 12)`.
    Q8,
    /// `Q_16^hi(P) = (P - 20)(P - 30)(P - 42)(P - 56)`.
    Q16Hi,
}

impl LeafPoly {
    /// The roots.
    #[must_use]
    pub const fn roots(self) -> &'static [i64] {
        match self {
            Self::Q4 => &[0, 2],
            Self::Q6 => &[0, 2, 6],
            Self::Q8 => &[0, 2, 6, 12],
            Self::Q16Hi => &[20, 30, 42, 56],
        }
    }

    /// Degree in `P`.
    #[must_use]
    pub const fn degree(self) -> usize {
        self.roots().len()
    }

    /// `Q(p)`.
    #[must_use]
    pub fn evaluate(self, p: E) -> E {
        self.roots()
            .iter()
            .fold(E::one(), |acc, &root| acc * (p - E::from_i64(root)))
    }

    /// The monomial coefficients of `Q(P)`, constant term first.
    #[must_use]
    pub fn coefficients(self) -> Vec<i64> {
        let mut coefficients = vec![1_i64];
        for &root in self.roots() {
            // Multiply by (P - root).
            let mut next = vec![0_i64; coefficients.len() + 1];
            for (k, &c) in coefficients.iter().enumerate() {
                next[k + 1] += c;
                next[k] -= c * root;
            }
            coefficients = next;
        }
        coefficients
    }

    /// The leaf of a digit base: `Q_4`, `Q_6`, `Q_8`, or both halves of `Q_16` (lo then hi).
    #[must_use]
    pub fn of_base(base: u32) -> Vec<Self> {
        match base {
            4 => vec![Self::Q4],
            6 => vec![Self::Q6],
            8 => vec![Self::Q8],
            16 => vec![Self::Q8, Self::Q16Hi],
            _ => panic!("unsupported digit base {base}"),
        }
    }
}

/// The summand of one instance (without the shared equality factor), a polynomial in the
/// lane values of its group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Term {
    /// The product of the listed lanes.
    Product(Vec<usize>),
    /// `v (v - 1)` of one lane.
    Bool(usize),
    /// `b_5 (b_4 + b_3 b_2)`, lanes in that order.
    RangeW1([usize; 4]),
    /// `Q(P)` for a leaf predicate on the lane `P_Z`.
    Leaf(usize, LeafPoly),
    /// `Z (Z + 1)` of one lane.
    Link(usize),
    /// `weight * (sum c_k v_k + c)`: an optional weight lane times an affine combination.
    Affine {
        /// The weight lane, if any.
        weight: Option<usize>,
        /// `(lane, coefficient)` pairs.
        lanes: Vec<(usize, i64)>,
        /// The constant.
        constant: i64,
    },
    /// `sum_k c_k prod_{l in m_k} v_l`: a polynomial with integer coefficients, each monomial
    /// a list of lanes (with repetition; an empty list is a constant). Relation H-FIPS's
    /// wrap constraints; evaluated on the generic path.
    Monomials(Vec<(i64, Vec<usize>)>),
}

impl Term {
    /// Degree in the lane values.
    #[must_use]
    pub fn degree(&self) -> usize {
        match self {
            Self::Product(lanes) => lanes.len(),
            Self::Bool(_) | Self::Link(_) => 2,
            Self::RangeW1(_) => 3,
            Self::Leaf(_, poly) => poly.degree(),
            Self::Affine { weight, .. } => 1 + usize::from(weight.is_some()),
            Self::Monomials(monomials) => monomials.iter().map(|(_, m)| m.len()).max().unwrap_or(0),
        }
    }

    /// The summand at the given lane values.
    #[inline]
    #[must_use]
    pub fn evaluate(&self, values: &[E]) -> E {
        match self {
            Self::Product(lanes) => lanes.iter().fold(E::one(), |acc, &l| acc * values[l]),
            Self::Bool(l) => values[*l] * (values[*l] - E::one()),
            Self::RangeW1([b5, b4, b3, b2]) => {
                values[*b5] * (values[*b4] + values[*b3] * values[*b2])
            }
            Self::Leaf(l, poly) => poly.evaluate(values[*l]),
            Self::Link(l) => values[*l] * (values[*l] + E::one()),
            Self::Affine {
                weight,
                lanes,
                constant,
            } => {
                let affine = lanes
                    .iter()
                    .fold(E::from_i64(*constant), |acc, &(l, c)| {
                        acc + values[l].mul_i64(c)
                    });
                match weight {
                    Some(w) => values[*w] * affine,
                    None => affine,
                }
            }
            Self::Monomials(monomials) => monomials.iter().fold(E::zero(), |acc, (c, m)| {
                acc + m
                    .iter()
                    .fold(E::one(), |p, &l| p * values[l])
                    .mul_i64(*c)
            }),
        }
    }

    /// Every lane the term reads.
    #[must_use]
    pub fn lanes(&self) -> Vec<usize> {
        match self {
            Self::Product(lanes) => lanes.clone(),
            Self::Bool(l) | Self::Link(l) | Self::Leaf(l, _) => vec![*l],
            Self::RangeW1(lanes) => lanes.to_vec(),
            Self::Affine { weight, lanes, .. } => weight
                .iter()
                .copied()
                .chain(lanes.iter().map(|&(l, _)| l))
                .collect(),
            Self::Monomials(monomials) => {
                let mut lanes: Vec<usize> = monomials.iter().flat_map(|(_, m)| m.iter().copied()).collect();
                lanes.sort_unstable();
                lanes.dedup();
                lanes
            }
        }
    }
}

/// `2^-k`.
#[must_use]
pub fn inverse_power_of_two(k: usize) -> E {
    let half = E::from_u64(2).inverse().expect("two is invertible");
    (0..k).fold(E::one(), |acc, _| acc * half)
}

/// The compressed message of a degree-`degree` round polynomial: its coefficients without
/// the linear one (`degree` elements).
#[must_use]
pub fn compress(full: &[E], degree: usize) -> Vec<E> {
    let mut out = Vec::with_capacity(degree);
    out.push(full.first().copied().unwrap_or_else(E::zero));
    out.extend((2..=degree).map(|k| full.get(k).copied().unwrap_or_else(E::zero)));
    out
}

/// Recover the linear coefficient from `g(0) + g(1) = claim` and evaluate at `x`.
#[must_use]
pub fn evaluate_compressed(message: &[E], claim: E, x: E) -> E {
    let c0 = message[0];
    let rest = message[1..].iter().fold(E::zero(), |acc, &c| acc + c);
    let c1 = claim - c0 - c0 - rest;
    // Horner over [c0, c1, c2, ..., cd].
    let mut acc = E::zero();
    for &c in message[1..].iter().rev() {
        acc = acc * x + c;
    }
    acc = acc * x + c1;
    acc * x + c0
}

/// The constant message of an inactive round: `claim / 2`.
#[must_use]
pub fn inactive_message(claim: E, degree: usize) -> Vec<E> {
    let mut out = vec![E::zero(); degree];
    out[0] = claim * inverse_power_of_two(1);
    out
}

/// One instance's terms and gamma within a group.
#[derive(Clone, Debug)]
pub struct GroupInstance {
    /// The batching coefficient.
    pub gamma: E,
    /// The summands (their sum is the instance's summand).
    pub terms: Vec<Term>,
}

/// The verifier's expected contribution of a group at the terminal: `2^-inactive` times
/// the equality factor `eq(point, r)` (if any) times the gamma-weighted terms at the
/// claimed lane values.
#[must_use]
pub fn expected_terminal(
    instances: &[GroupInstance],
    values: &[E],
    eq_factor: E,
    inactive: usize,
) -> E {
    let sum = instances.iter().fold(E::zero(), |acc, instance| {
        let terms = instance
            .terms
            .iter()
            .fold(E::zero(), |s, term| s + term.evaluate(values));
        acc + instance.gamma * terms
    });
    sum * eq_factor * inverse_power_of_two(inactive)
}

// ---------------------------------------------------------------------------------------
// The group.
// ---------------------------------------------------------------------------------------

/// A strategy choice (tests force each path; production uses `Auto`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Choice {
    Auto,
    #[cfg_attr(not(test), allow(dead_code))]
    Force,
    #[cfg_attr(not(test), allow(dead_code))]
    Forbid,
}

/// The evaluation strategy of a group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Policy {
    /// Code the byte columns (else every lane is lifted at its first fold).
    coding: bool,
    /// Bucket the pieces on coded lanes in the compact rounds (else per item).
    buckets: bool,
    /// The two-round prefix from quartet masses.
    bivariate: Choice,
    /// Quartet lookup tables at level 2.
    quartet_lut: Choice,
    /// Taylor tables of the lane polynomials on quartet lookup lanes.
    taylor: bool,
    /// Recover the constant coefficient of the dense rounds from the running claim.
    recover: bool,
}

const AUTO: Policy = Policy {
    coding: true,
    buckets: true,
    bivariate: Choice::Auto,
    quartet_lut: Choice::Auto,
    taylor: true,
    recover: true,
};

/// A bucketed segment: the pieces on one coded column, or on the columns of a joint term.
#[derive(Clone, Debug)]
struct Segment {
    shape: SegmentShape,
    source: Pieces,
    folded: Pieces,
    /// The lanes read, with the position of each lane's column in `shape.columns`.
    lanes: Vec<(usize, usize)>,
}

/// A weighted sum of the source phase with per-byte tables of its byte lanes (so the first
/// round adds table entries instead of multiplying by the coefficients).
#[derive(Clone, Debug)]
struct WeightedSource<'a> {
    weight: usize,
    tables: Vec<(&'a [i8], Box<[E; 256]>)>,
    others: Vec<(usize, E)>,
    constant: E,
}

/// The state of one lane at the current level.
#[derive(Clone, Debug)]
enum State {
    /// Level 0: read from the source.
    Source,
    /// A coded lane after one fold: cell `k` is `lut[pairs[k]]` (scaled).
    Pair(Vec<E>),
    /// A coded lane after two folds: cell `k` is `lut[pairs[2k] * n^2 + pairs[2k + 1]]`.
    Quartet(Vec<E>),
    /// A coded lane after two folds without a quartet table: the pair table and the
    /// second challenge.
    PairFold(Vec<E>, E),
    /// An uncoded lane after at least one fold (scaled), until the dense table.
    Table(Vec<E>),
    /// A column of the dense table.
    Dense,
}

/// Direct access to a lane in a compact state (no per-cell indirection through the group).
#[derive(Clone, Copy)]
enum Reader<'r> {
    Pair { pairs: &'r [u8], lut: &'r [E] },
    Quartet { pairs: &'r [u8], lut: &'r [E], space: usize },
    PairFold { pairs: &'r [u8], lut: &'r [E], r: E },
    Table(&'r [E]),
}

impl Reader<'_> {
    /// The value at `cell` of the current level.
    #[inline(always)]
    fn at(&self, cell: usize) -> E {
        match *self {
            Self::Pair { pairs, lut } => lut[pairs[cell] as usize],
            Self::Quartet { pairs, lut, space } => {
                lut[pairs[2 * cell] as usize * space + pairs[2 * cell + 1] as usize]
            }
            Self::PairFold { pairs, lut, r } => {
                let even = lut[pairs[2 * cell] as usize];
                even + fm(r, lut[pairs[2 * cell + 1] as usize] - even)
            }
            Self::Table(table) => table[cell],
        }
    }

    /// The values at cells `2k` and `2k + 1`.
    #[inline(always)]
    fn item(&self, k: usize) -> (E, E) {
        match *self {
            Self::Pair { pairs, lut } => (lut[pairs[2 * k] as usize], lut[pairs[2 * k + 1] as usize]),
            Self::Quartet { pairs, lut, space } => {
                let p = &pairs[4 * k..4 * k + 4];
                (
                    lut[p[0] as usize * space + p[1] as usize],
                    lut[p[2] as usize * space + p[3] as usize],
                )
            }
            Self::PairFold { pairs, lut, r } => {
                let p = &pairs[4 * k..4 * k + 4];
                let (e0, e1) = (lut[p[0] as usize], lut[p[2] as usize]);
                (e0 + fm(r, lut[p[1] as usize] - e0), e1 + fm(r, lut[p[3] as usize] - e1))
            }
            Self::Table(table) => (table[2 * k], table[2 * k + 1]),
        }
    }
}

/// The Taylor coefficients `P^(i)(v) / i!`, `i = 1..=4`, of a lane polynomial at one
/// quartet-table entry, on one cache line.
#[derive(Clone, Copy, Debug)]
#[repr(align(64))]
struct TaylorLine([E; 4]);

/// Per lane: the Taylor coefficients of its polynomial (degree 2 to 4) at every quartet
/// table entry; the constant coefficient comes from the claim or from `kind`.
#[derive(Clone, Debug)]
struct Taylor {
    lane: usize,
    degree: usize,
    kind: plan::PolyKind,
    lines: Vec<TaylorLine>,
}

/// The lanes of one local domain and the instances batched on them.
pub struct Group<'a> {
    eq: Option<GruenSplitEq<E>>,
    /// The local equality point (empty without an equality factor).
    tau: Vec<E>,
    sources: Vec<Lane<'a>>,
    plan: Plan,
    columns: Vec<Column>,
    lane_column: Vec<Option<usize>>,
    segments: Vec<Segment>,
    /// The pieces evaluated per item in the compact rounds (source, folded phase).
    direct_source: Pieces,
    direct_folded: Pieces,
    /// The weighted sums of `direct_source` with byte tables; `direct_source_rest` the
    /// other direct pieces.
    weighted_source: Vec<WeightedSource<'a>>,
    direct_source_rest: Pieces,
    states: Vec<State>,
    taylor: Vec<Taylor>,
    /// The folded pieces without the polynomials of the Taylor lanes (level 2).
    level2_pieces: Pieces,
    /// Lane-major `K` table of every lane from level 3 on: lane `l` occupies
    /// `[l * dense_stride, l * dense_stride + dense_rows)` (the stride is the row count at
    /// level 3; a fold halves `dense_rows` in place).
    dense: Vec<E>,
    dense_stride: usize,
    dense_rows: usize,
    /// The term lists of the plane path, until bind 2 builds the table.
    tail_terms: Option<TailTerms>,
    /// The dense rounds on lane-major planes (plane layout only).
    tail: Option<Tail>,
    arena: Option<SharedArena>,
    challenges: Vec<E>,
    /// Quartet masses of the bucketed segments (the two-round prefix), kept for level 1.
    masses: Option<Vec<Vec<E>>>,
    /// The level-0 inner polynomial computed for the direct claim of a lifted group.
    cached_level0: Option<Vec<E>>,
    /// The inner polynomial of the next dense round, computed by the fold that produced its
    /// table (without the constant term, with coefficient 0 only when not recovered).
    cached_next: Option<Vec<E>>,
    policy: Policy,
    degree: usize,
    inactive: usize,
    local_vars: usize,
    bound: usize,
    scale: E,
    claim: E,
    pending: Vec<E>,
}

/// Per-item scratch: lane values and differences.
struct Scratch {
    a: Vec<E>,
    d: Vec<E>,
}

impl<'a> Group<'a> {
    pub(crate) fn set_arena(&mut self, arena: &SharedArena) {
        self.arena = Some(arena.clone());
    }

    /// A group on a local domain of `2^local_vars` cells inside a stage of `rounds`
    /// rounds and message degree `degree`. `eq_point` is the shared equality point of
    /// the group's instances (length `local_vars`), if any.
    ///
    /// The group's gamma-weighted claim is computed from the tables: directly at
    /// construction when the group has inactive leading rounds (its local domain is
    /// small), else from the first round's polynomial (`g(0) + g(1)`), so a split
    /// instance may spread its summands over several groups and the stage driver checks
    /// the sum of the group claims against the claimed sums at the end.
    pub fn new(
        lanes: Vec<Lane<'a>>,
        instances: Vec<GroupInstance>,
        eq_point: Option<&[E]>,
        rounds: usize,
        degree: usize,
    ) -> Result<Self, String> {
        Self::with_policy(lanes, instances, eq_point, rounds, degree, AUTO)
    }

    fn with_policy(
        lanes: Vec<Lane<'a>>,
        instances: Vec<GroupInstance>,
        eq_point: Option<&[E]>,
        rounds: usize,
        degree: usize,
        policy: Policy,
    ) -> Result<Self, String> {
        let local_vars = lanes
            .first()
            .map(|lane| lane.len().trailing_zeros() as usize)
            .ok_or("a group needs at least one lane")?;
        for lane in &lanes {
            if lane.len() != 1 << local_vars {
                return Err(format!(
                    "lane of {} cells in a group of 2^{local_vars}",
                    lane.len()
                ));
            }
        }
        if local_vars > rounds {
            return Err(format!("local domain 2^{local_vars} exceeds {rounds} rounds"));
        }
        let inner = usize::from(eq_point.is_some());
        for instance in &instances {
            for term in &instance.terms {
                if term.degree() + inner > degree {
                    return Err(format!("term {term:?} exceeds degree {degree}"));
                }
                if term.lanes().iter().any(|&l| l >= lanes.len()) {
                    return Err(format!("term {term:?} reads a missing lane"));
                }
            }
        }
        if degree + 1 - inner > field::MAX_COEFFICIENTS {
            return Err(format!("degree {degree} exceeds the kernel's maximum"));
        }
        let eq = match eq_point {
            Some(point) => {
                if point.len() != local_vars {
                    return Err("equality point of the wrong length".into());
                }
                Some(GruenSplitEq::new(point).map_err(|e| e.to_string())?)
            }
            None => None,
        };
        let is_field: Vec<bool> = lanes.iter().map(Lane::is_field).collect();
        let plan = Plan::compile(lanes.len(), &instances, &is_field);

        // Code the byte columns (by pointer) under the maps of all lanes reading them.
        let mut columns: Vec<Column> = Vec::new();
        let mut lane_column: Vec<Option<usize>> = vec![None; lanes.len()];
        if policy.coding && local_vars >= 1 {
            let mut keys: Vec<(*const i8, usize)> = Vec::new();
            let mut lane_key: Vec<Option<usize>> = vec![None; lanes.len()];
            for (l, lane) in lanes.iter().enumerate() {
                if let Lane::Source(Source::Bytes { cells, .. }) = lane {
                    let key = (cells.as_ptr(), cells.len());
                    let index = keys.iter().position(|k| *k == key).unwrap_or_else(|| {
                        keys.push(key);
                        keys.len() - 1
                    });
                    lane_key[l] = Some(index);
                }
            }
            // The columns are coded side by side (each pass is parallel inside as well); the
            // results are placed in key order, so the column indices do not depend on it.
            let coded: Vec<(Vec<usize>, Option<Column>)> = (0..keys.len())
                .into_par_iter()
                .map(|key| {
                    let members: Vec<usize> = (0..lanes.len()).filter(|&l| lane_key[l] == Some(key)).collect();
                    let Lane::Source(Source::Bytes { cells, .. }) = lanes[members[0]] else {
                        unreachable!("a byte lane")
                    };
                    let maps: Vec<&[i64; 256]> = members
                        .iter()
                        .map(|&l| match lanes[l] {
                            Lane::Source(Source::Bytes { map, .. }) => map,
                            _ => unreachable!("a byte lane"),
                        })
                        .collect();
                    let column = code_column(cells, &maps);
                    (members, column)
                })
                .collect();
            for (members, column) in coded {
                if let Some(column) = column {
                    columns.push(column);
                    for &l in &members {
                        lane_column[l] = Some(columns.len() - 1);
                    }
                }
            }
        }

        // Split the pieces into bucketed segments and per-item (direct) pieces.
        let mut segments: Vec<Segment> = Vec::new();
        let mut direct_source = Pieces::default();
        let mut direct_folded = Pieces::default();
        let segment_of = |segments: &mut Vec<Segment>, cols: Vec<usize>, columns: &[Column]| -> usize {
            if let Some(i) = segments.iter().position(|s| s.shape.columns == cols) {
                return i;
            }
            segments.push(Segment {
                shape: SegmentShape::new(cols, columns),
                source: Pieces::default(),
                folded: Pieces::default(),
                lanes: Vec::new(),
            });
            segments.len() - 1
        };
        let joint_columns = |piece_lanes: &[usize]| -> Option<Vec<usize>> {
            if !policy.buckets {
                return None;
            }
            let mut cols = Vec::new();
            for &l in piece_lanes {
                let c = lane_column[l]?;
                if !cols.contains(&c) {
                    cols.push(c);
                }
            }
            cols.sort_unstable();
            let space: usize = cols.iter().map(|&c| columns[c].pair_space()).product();
            (space <= MAX_JOINT_PAIRS).then_some(cols)
        };
        for (i, (l, kind)) in plan.source.polys.iter().enumerate() {
            let folded = plan.folded.polys[i].clone();
            match joint_columns(&[*l]) {
                Some(cols) => {
                    let s = segment_of(&mut segments, cols, &columns);
                    segments[s].source.polys.push((*l, kind.clone()));
                    segments[s].folded.polys.push(folded);
                }
                None => {
                    direct_source.polys.push((*l, kind.clone()));
                    direct_folded.polys.push(folded);
                }
            }
        }
        for (i, b) in plan.source.bilinear.iter().enumerate() {
            let folded = plan.folded.bilinear[i].clone();
            match joint_columns(&[b.a, b.b]) {
                Some(cols) => {
                    let s = segment_of(&mut segments, cols, &columns);
                    segments[s].source.bilinear.push(b.clone());
                    segments[s].folded.bilinear.push(folded);
                }
                None => {
                    direct_source.bilinear.push(b.clone());
                    direct_folded.bilinear.push(folded);
                }
            }
        }
        for (i, (gamma, term)) in plan.source.generic.iter().enumerate() {
            let folded = plan.folded.generic[i].clone();
            match joint_columns(&term.lanes()) {
                Some(cols) => {
                    let s = segment_of(&mut segments, cols, &columns);
                    segments[s].source.generic.push((*gamma, term.clone()));
                    segments[s].folded.generic.push(folded);
                }
                None => {
                    direct_source.generic.push((*gamma, term.clone()));
                    direct_folded.generic.push(folded);
                }
            }
        }
        direct_source.weighted = plan.source.weighted.clone();
        direct_folded.weighted = plan.folded.weighted.clone();
        for segment in &mut segments {
            segment.lanes = segment
                .source
                .lanes()
                .into_iter()
                .map(|l| {
                    let c = lane_column[l].expect("bucketed lanes are coded");
                    let position = segment.shape.columns.iter().position(|&cc| cc == c).expect("listed");
                    (l, position)
                })
                .collect();
        }
        // Byte tables of the weighted sums for the first round.
        let mut weighted_source = Vec::new();
        for form in &direct_source.weighted {
            let mut tables: Vec<(&'a [i8], Box<[E; 256]>)> = Vec::new();
            let mut others = Vec::new();
            for &(l, c) in &form.lanes {
                match lanes[l] {
                    Lane::Source(Source::Bytes { cells, map }) => {
                        let entry = |byte: usize| fm(c, E::from_i64(map[byte]));
                        match tables.iter_mut().find(|(t, _)| std::ptr::eq(*t, cells)) {
                            Some((_, table)) => {
                                for (byte, slot) in table.iter_mut().enumerate() {
                                    *slot += entry(byte);
                                }
                            }
                            None => tables.push((cells, Box::new(std::array::from_fn(entry)))),
                        }
                    }
                    _ => others.push((l, c)),
                }
            }
            weighted_source.push(WeightedSource {
                weight: form.weight,
                tables,
                others,
                constant: form.constant,
            });
        }
        let direct_source_rest = Pieces {
            weighted: Vec::new(),
            ..direct_source.clone()
        };
        let lane_count = lanes.len();
        // The tail on planes needs an equality factor (so a cell's constant enters once),
        // a degree at least three, and lanes without a shift.
        let tail_terms = (layout() == Layout::Plane
            && eq.is_some()
            && degree >= 3
            && local_vars >= 4
            && plan.shifts.iter().all(Zero::is_zero))
        .then(|| TailTerms::compile(&plan.dense, lane_count))
        .flatten();
        let mut group = Self {
            eq,
            tau: eq_point.map(<[E]>::to_vec).unwrap_or_default(),
            sources: lanes,
            plan,
            columns,
            lane_column,
            segments,
            direct_source,
            direct_folded,
            weighted_source,
            direct_source_rest,
            states: vec![State::Source; lane_count],
            taylor: Vec::new(),
            level2_pieces: Pieces::default(),
            dense: Vec::new(),
            dense_stride: 0,
            dense_rows: 0,
            tail_terms,
            tail: None,
            arena: None,
            challenges: Vec::new(),
            masses: None,
            cached_level0: None,
            cached_next: None,
            policy,
            degree,
            inactive: rounds - local_vars,
            local_vars,
            bound: 0,
            scale: E::one(),
            claim: E::zero(),
            pending: Vec::new(),
        };
        if group.inactive > 0 {
            group.claim = group.direct_claim();
        }
        Ok(group)
    }

    /// The gamma-weighted sum of the instances over the local Boolean domain, from the
    /// level-0 inner polynomial (`l(0) q(0) + l(1) q(1)`), which is kept for the first
    /// active round.
    fn direct_claim(&mut self) -> E {
        if self.local_vars == 0 {
            let values: Vec<E> = self.sources.iter().map(|lane| lane.value(0)).collect();
            return self.single_cell_sum(&values);
        }
        let q = self.level0();
        let at_zero = q[0];
        let at_one = q.iter().fold(E::zero(), |s, &c| s + c);
        let claim = match &self.eq {
            Some(eq) => {
                let tau = eq.current_tau();
                (E::one() - tau) * at_zero + tau * at_one
            }
            None => at_zero + at_one,
        };
        self.cached_level0 = Some(q);
        claim
    }

    /// The summand of a one-cell domain.
    fn single_cell_sum(&self, values: &[E]) -> E {
        let zero = vec![E::zero(); values.len()];
        let out = self.plan.source.evaluate(values, &zero, self.coefficient_count(), false);
        out[0] + self.plan.constant
    }

    /// The current gamma-weighted claim.
    #[must_use]
    pub fn claim(&self) -> E {
        self.claim
    }

    /// True when some pieces are bucketed in the compact rounds.
    #[must_use]
    pub fn is_bucketed(&self) -> bool {
        !self.segments.is_empty()
    }

    /// Leading inactive rounds.
    #[must_use]
    pub fn inactive_rounds(&self) -> usize {
        self.inactive
    }

    /// Coefficients of the inner polynomial `q` (degree `d - 1` with an equality factor).
    fn coefficient_count(&self) -> usize {
        if self.eq.is_some() {
            self.degree
        } else {
            self.degree + 1
        }
    }

    /// The active level of the next round (0 before and during the inactive rounds).
    fn level(&self) -> usize {
        self.bound.saturating_sub(self.inactive)
    }

    /// Pairs at the current level.
    fn pairs(&self) -> usize {
        (1 << (self.local_vars - self.level())) / 2
    }

    /// The compressed message of `round` (`degree` elements), already scaled by the
    /// gammas and the lift.
    pub fn round_message(&mut self, round: usize) -> Vec<E> {
        assert_eq!(round, self.bound, "rounds in order");
        if round < self.inactive {
            let message = inactive_message(self.claim, self.degree);
            self.pending = message.clone();
            return message;
        }
        // After the first active round the claim is this group's exact sum, so the
        // constant coefficient of `q` follows from the others (exact field arithmetic: the
        // same polynomial as a direct evaluation).
        let level = self.level();
        let on_planes = level >= 3 && self.tail.is_some();
        let recovery = if level == 0 || on_planes { None } else { self.recovery() };
        let upper = recovery.is_some();
        let mut q = match level {
            0 => match self.cached_level0.take() {
                Some(q) => q,
                None => self.level0(),
            },
            1 => self.level1(upper),
            2 => self.level2(upper),
            _ if on_planes => self.plane_round(),
            _ => {
                let mut q = match self.cached_next.take() {
                    Some(q) => q,
                    None => self.dense_items(upper),
                };
                if !upper {
                    q[0] += self.constant_term();
                }
                q
            }
        };
        if let Some(recovery) = recovery {
            q[0] = self.recovered_constant(&q, recovery);
        }
        let full: Vec<E> = match &self.eq {
            Some(eq) => eq.gruen_mul(&UnivariatePoly::new(q)).into_coefficients(),
            None => q,
        }
        .into_iter()
        .map(|c| c * self.scale)
        .collect();
        let at_zero_plus_one = full.iter().fold(E::zero(), |s, &c| s + c) + full[0];
        if round == 0 {
            self.claim = at_zero_plus_one;
        }
        debug_assert_eq!(at_zero_plus_one, self.claim, "g(0) + g(1) = claim");
        let message = compress(&full, self.degree);
        self.pending = message.clone();
        message
    }

    /// A round over the dense planes: the coefficients of the inner polynomial `q`, the
    /// linear one from the evaluation at one, or from the running claim when the equality
    /// factor allows it.
    fn plane_round(&mut self) -> Vec<E> {
        let eq = self.eq.as_ref().expect("plane groups have an equality factor");
        let tail = self.tail.as_mut().expect("the plane table");
        let full = !eq.can_recover_linear_q_term_from_claim();
        let [q0, at_one, q2] = tail.message(eq, full, self.plan.constant);
        let q1 = if full {
            at_one - q0 - q2
        } else {
            // claim / lift = l(0) q(0) + l(1) q(1).
            let (l0, l1) = eq.linear_factor_evals();
            let unscaled = self.claim * self.scale.inverse().expect("nonzero lift");
            (unscaled - l0 * q0) * l1.inverse().expect("recoverable round") - q0 - q2
        };
        let mut q = vec![E::zero(); self.coefficient_count()];
        q[..3].copy_from_slice(&[q0, q1, q2]);
        q
    }

    /// The constant of every cell, summed over the pairs with their weights.
    fn constant_term(&self) -> E {
        if self.plan.constant.is_zero() {
            return E::zero();
        }
        match &self.eq {
            // The remaining equality weights sum to one.
            Some(_) => self.plan.constant,
            None => self.plan.constant * E::from_u64(self.pairs() as u64),
        }
    }

    /// `sum_k w_k c(k)` over the pairs of the current level, with the remaining equality
    /// weights (or weight one).
    fn item_sum(&self, item: impl Fn(usize, &mut Scratch, &mut [Acc]) + Sync) -> Vec<E> {
        let width = self.coefficient_count();
        let lanes = self.sources.len();
        let init = || Scratch {
            a: vec![E::zero(); lanes],
            d: vec![E::zero(); lanes],
        };
        match &self.eq {
            Some(eq) => {
                let (first, second) = eq.remaining_eq_tables();
                split_sum(first, second, width, init, item)
            }
            None => plain_sum(self.pairs(), width, init, item),
        }
    }

    /// True when rounds 0 and 1 run from quartet masses.
    fn bivariate(&self) -> bool {
        if self.local_vars < 2 || self.segments.is_empty() {
            return false;
        }
        let largest = self
            .segments
            .iter()
            .map(|s| s.shape.pair_space * s.shape.pair_space)
            .max()
            .unwrap_or(0);
        if largest > MAX_QUARTET_SPACE {
            return false;
        }
        match self.policy.bivariate {
            Choice::Force => true,
            Choice::Forbid => false,
            Choice::Auto => (1_usize << (self.local_vars - 2)) >= BUCKET_MIN_RATIO * largest,
        }
    }

    /// The source values `(a, b)` of a lane at level-0 pattern classes.
    fn class_values(&self, l: usize, column: &Column, k0: usize, k1: usize) -> (E, E) {
        let Lane::Source(Source::Bytes { map, .. }) = self.sources[l] else {
            unreachable!("coded lanes are byte lanes")
        };
        let a = map[column.reps[k0] as u8 as usize];
        let b = map[column.reps[k1] as u8 as usize];
        (E::from_i64(a), E::from_i64(b - a))
    }

    /// `sum_p weight[p] F_s(p)` over the pair patterns of a segment (source phase).
    fn pair_pattern_sum(&self, segment: &Segment, weights: &[E], out: &mut [Acc]) {
        let mut a = vec![E::zero(); self.sources.len()];
        let mut d = vec![E::zero(); self.sources.len()];
        let mut coefficients = vec![Acc::zero(); out.len()];
        for (pattern, &weight) in weights.iter().enumerate() {
            if weight.is_zero() {
                continue;
            }
            for &(l, position) in &segment.lanes {
                let column = &self.columns[segment.shape.columns[position]];
                let (k0, k1) = column.split(segment.shape.component(&self.columns, pattern, position));
                let (va, vd) = self.class_values(l, column, k0, k1);
                a[l] = va;
                d[l] = vd;
            }
            coefficients.iter_mut().for_each(|c| *c = Acc::zero());
            segment.source.add_at(&a, &d, &mut coefficients);
            for (slot, &c) in out.iter_mut().zip(&coefficients) {
                *slot += weight.mul_unreduced(E::reduce_product(c));
            }
        }
    }

    /// The first active round.
    fn level0(&mut self) -> Vec<E> {
        let width = self.coefficient_count();
        let mut q = vec![E::zero(); width];
        q[0] += self.constant_term();
        if !self.segments.is_empty() {
            let shapes: Vec<SegmentShape> = self.segments.iter().map(|s| s.shape.clone()).collect();
            // Per segment, the weight of every pair pattern.
            let pattern_weights: Vec<Vec<E>> = if self.bivariate() {
                let quartets = 1 << (self.local_vars - 2);
                let weights = match &self.eq {
                    Some(eq) => {
                        let (first, second) = eq.remaining_eq_tables_after(1).expect("two local variables");
                        ItemWeights::Split(first, second)
                    }
                    None => ItemWeights::One,
                };
                let quartet_masses = masses(&self.columns, &shapes, quartets, true, &weights);
                let (l0, l1) = match &self.eq {
                    Some(_) => (E::one() - self.tau[1], self.tau[1]),
                    None => (E::one(), E::one()),
                };
                let marginals = quartet_masses
                    .par_iter()
                    .zip(&shapes)
                    .map(|(m, shape)| {
                        let space = shape.pair_space;
                        let mut left = vec![E::zero(); space];
                        let mut right = vec![E::zero(); space];
                        for (pattern, &mass) in m.iter().enumerate() {
                            left[pattern / space] += mass;
                            right[pattern % space] += mass;
                        }
                        left.iter()
                            .zip(&right)
                            .map(|(&x, &y)| l0 * x + l1 * y)
                            .collect()
                    })
                    .collect();
                self.masses = Some(quartet_masses);
                marginals
            } else {
                let weights = match &self.eq {
                    Some(eq) => {
                        let (first, second) = eq.remaining_eq_tables();
                        ItemWeights::Split(first, second)
                    }
                    None => ItemWeights::One,
                };
                masses(&self.columns, &shapes, self.pairs(), false, &weights)
            };
            let sums: Vec<Vec<Acc>> = self
                .segments
                .par_iter()
                .zip(&pattern_weights)
                .map(|(segment, weights)| {
                    let mut out = vec![Acc::zero(); width];
                    self.pair_pattern_sum(segment, weights, &mut out);
                    out
                })
                .collect();
            for sum in sums {
                for (slot, value) in q.iter_mut().zip(sum) {
                    *slot += E::reduce_product(value);
                }
            }
        }
        if !self.direct_source.is_empty() {
            let rest = &self.direct_source_rest;
            let rest_lanes = rest.lanes();
            let weighted = &self.weighted_source;
            let sources = &self.sources;
            let direct = self.item_sum(|k, scratch, out| {
                for &l in &rest_lanes {
                    let (va, vb) = (sources[l].value(2 * k), sources[l].value(2 * k + 1));
                    scratch.a[l] = va;
                    scratch.d[l] = vb - va;
                }
                rest.add_at(&scratch.a, &scratch.d, out);
                for form in weighted {
                    let (mut sa, mut sb) = (form.constant, form.constant);
                    for (cells, table) in &form.tables {
                        sa += table[cells[2 * k] as u8 as usize];
                        sb += table[cells[2 * k + 1] as u8 as usize];
                    }
                    for &(l, c) in &form.others {
                        sa += fm(c, sources[l].value(2 * k));
                        sb += fm(c, sources[l].value(2 * k + 1));
                    }
                    let wa = sources[form.weight].value(2 * k);
                    let wb = sources[form.weight].value(2 * k + 1);
                    plan::add_bilinear(E::one(), wa, wb - wa, sa, sb - sa, out);
                }
            });
            for (slot, value) in q.iter_mut().zip(direct) {
                *slot += value;
            }
        }
        q
    }

    /// The value of lane `l` at cell `cell` of the current level (1 or 2, or dense).
    fn value_at(&self, l: usize, cell: usize) -> E {
        match &self.states[l] {
            State::Source => self.sources[l].value(cell),
            State::Dense => match &self.tail {
                Some(tail) => tail.get(l, cell),
                None => self.dense[l * self.dense_stride + cell] + self.plan.shifts[l],
            },
            _ => self.reader(l).at(cell),
        }
    }

    /// The reader of a lane in a compact state after the first fold.
    fn reader(&self, l: usize) -> Reader<'_> {
        let pairs = || &self.columns[self.lane_column[l].expect("coded")].pairs[..];
        match &self.states[l] {
            State::Pair(lut) => Reader::Pair { pairs: pairs(), lut },
            State::Quartet(lut) => Reader::Quartet {
                pairs: pairs(),
                lut,
                space: self.columns[self.lane_column[l].expect("coded")].pair_space(),
            },
            State::PairFold(lut, r) => Reader::PairFold {
                pairs: pairs(),
                lut,
                r: *r,
            },
            State::Table(table) => Reader::Table(table),
            other => unreachable!("no reader for {other:?}"),
        }
    }

    /// `sum_k w_k pieces(k)` per item at the current compact level (1 or 2); with `upper`,
    /// the coefficients from 1 up only.
    fn per_item(&self, pieces: &Pieces, upper: bool) -> Vec<E> {
        let readers: Vec<(usize, Reader<'_>)> = pieces.lanes().into_iter().map(|l| (l, self.reader(l))).collect();
        self.item_sum(|k, scratch, out| {
            for (l, reader) in &readers {
                let (va, vb) = reader.item(k);
                scratch.a[*l] = va;
                scratch.d[*l] = vb - va;
            }
            if upper {
                pieces.add_upper(&scratch.a, &scratch.d, out);
            } else {
                pieces.add_at(&scratch.a, &scratch.d, out);
            }
        })
    }

    /// `(1 / scalar, l(1))` of the current round when its constant coefficient can be
    /// recovered from the claim: `claim / lift = scalar q(0) + l(1) (q(1) - q(0))`, with
    /// `scalar = l(0) + l(1)` the equality factor's running scalar (2 and 1 without one).
    fn recovery(&self) -> Option<(E, E)> {
        if !self.policy.recover {
            return None;
        }
        match &self.eq {
            Some(eq) => {
                let (l0, l1) = eq.linear_factor_evals();
                (l0 + l1).inverse().map(|inverse| (inverse, l1))
            }
            None => Some((inverse_power_of_two(1), E::one())),
        }
    }

    /// The constant coefficient of `q` from the claim and the other coefficients.
    fn recovered_constant(&self, q: &[E], (inverse, l1): (E, E)) -> E {
        let unscaled = self.claim * self.scale.inverse().expect("nonzero lift");
        let upper = q[1..].iter().fold(E::zero(), |s, &c| s + c);
        (unscaled - l1 * upper) * inverse
    }

    /// The second active round.
    fn level1(&mut self, upper: bool) -> Vec<E> {
        let width = self.coefficient_count();
        let mut q = vec![E::zero(); width];
        if !upper {
            q[0] += self.constant_term();
        }
        if let Some(quartet_masses) = self.masses.take() {
            // Every quartet pattern `left * P2 + right` once, with the pair tables of its two
            // pairs; tasks are ranges of `left` (no division per pattern).
            let components: Vec<Vec<Vec<usize>>> = self
                .segments
                .iter()
                .map(|segment| {
                    (0..segment.shape.columns.len())
                        .map(|position| {
                            (0..segment.shape.pair_space)
                                .map(|p| segment.shape.component(&self.columns, p, position))
                                .collect()
                        })
                        .collect()
                })
                .collect();
            let tasks: Vec<(usize, usize, usize)> = self
                .segments
                .iter()
                .enumerate()
                .flat_map(|(s, segment)| {
                    let space = segment.shape.pair_space;
                    let rows = (1_usize << 12).div_ceil(space).max(1);
                    (0..space.div_ceil(rows)).map(move |chunk| (s, chunk * rows, ((chunk + 1) * rows).min(space)))
                })
                .collect();
            let sums: Vec<Vec<Acc>> = tasks
                .par_iter()
                .map(|&(s, first, last)| {
                    let segment = &self.segments[s];
                    let space = segment.shape.pair_space;
                    let table = &quartet_masses[s];
                    let luts: Vec<(usize, &[usize], &[E])> = segment
                        .lanes
                        .iter()
                        .map(|&(l, position)| {
                            let State::Pair(lut) = &self.states[l] else {
                                unreachable!("level-1 coded lanes are pair tables")
                            };
                            (l, &components[s][position][..], &lut[..])
                        })
                        .collect();
                    let mut out = vec![Acc::zero(); width];
                    if segment.shape.columns.len() == 1 && segment.folded.bilinear.is_empty()
                        && segment.folded.weighted.is_empty() && segment.folded.generic.is_empty()
                    {
                        // Single-lane polynomials of one column: per `left` and lane, the
                        // moments `m_j = sum_right M[left, right] L[right]^j` (one unreduced
                        // product per pattern and power), then
                        // `sum_right M P(a + X (b - a)) = sum_i T_i(a) X^i sum_j C(i, j) (-a)^(i-j) m_j`.
                        for (l, kind) in &segment.folded.polys {
                            let State::Pair(lut) = &self.states[*l] else {
                                unreachable!("level-1 coded lanes are pair tables")
                            };
                            let degree = kind.degree();
                            let powers: Vec<[E; field::MAX_COEFFICIENTS]> = lut
                                .iter()
                                .map(|&v| {
                                    let mut p = [E::one(); field::MAX_COEFFICIENTS];
                                    for j in 1..=degree {
                                        p[j] = fm(p[j - 1], v);
                                    }
                                    p
                                })
                                .collect();
                            for left in first..last {
                                let row = &table[left * space..(left + 1) * space];
                                let mut moments = [Acc::zero(); field::MAX_COEFFICIENTS];
                                for (&mass, power) in row.iter().zip(&powers) {
                                    if mass.is_zero() {
                                        continue;
                                    }
                                    lift(&mut moments[0], mass);
                                    for j in 1..=degree {
                                        mad(&mut moments[j], mass, power[j]);
                                    }
                                }
                                let m: Vec<E> = moments[..=degree].iter().map(|&acc| E::reduce_product(acc)).collect();
                                let a = lut[left];
                                let taylor = kind.taylor(a);
                                let minus_a = -a;
                                let mut powers_of_minus_a = [E::one(); field::MAX_COEFFICIENTS];
                                for j in 1..=degree {
                                    powers_of_minus_a[j] = fm(powers_of_minus_a[j - 1], minus_a);
                                }
                                for i in usize::from(upper)..=degree {
                                    // sum_j C(i, j) (-a)^(i - j) m_j.
                                    let mut shifted = E::zero();
                                    for j in 0..=i {
                                        let term = fm(powers_of_minus_a[i - j], m[j]);
                                        shifted += match BINOMIAL[i][j] {
                                            1 => term,
                                            c => term.mul_i64(c),
                                        };
                                    }
                                    mad(&mut out[i], taylor[i], shifted);
                                }
                            }
                        }
                        return out;
                    }
                    let mut a = vec![E::zero(); self.sources.len()];
                    let mut d = vec![E::zero(); self.sources.len()];
                    let mut coefficients = vec![Acc::zero(); width];
                    for left in first..last {
                        for right in 0..space {
                            let mass = table[left * space + right];
                            if mass.is_zero() {
                                continue;
                            }
                            for &(l, component, lut) in &luts {
                                let va = lut[component[left]];
                                a[l] = va;
                                d[l] = lut[component[right]] - va;
                            }
                            coefficients.iter_mut().for_each(|c| *c = Acc::zero());
                            if upper {
                                segment.folded.add_upper(&a, &d, &mut coefficients);
                            } else {
                                segment.folded.add_at(&a, &d, &mut coefficients);
                            }
                            for (slot, &c) in out.iter_mut().zip(&coefficients) {
                                *slot += mass.mul_unreduced(E::reduce_product(c));
                            }
                        }
                    }
                    out
                })
                .collect();
            for sum in sums {
                for (slot, value) in q.iter_mut().zip(sum) {
                    *slot += E::reduce_product(value);
                }
            }
            if !self.direct_folded.is_empty() {
                let direct = self.per_item(&self.direct_folded, upper);
                for (slot, value) in q.iter_mut().zip(direct) {
                    *slot += value;
                }
            }
            return q;
        }
        for (slot, value) in q.iter_mut().zip(self.per_item(&self.plan.folded, upper)) {
            *slot += value;
        }
        q
    }

    /// The third active round (quartet tables, Taylor tables). (Evaluating it segment by
    /// segment over blocks of items, so that each segment's tables stay in cache, was
    /// measured slower.)
    fn level2(&self, upper: bool) -> Vec<E> {
        let width = self.coefficient_count();
        let mut q = vec![E::zero(); width];
        if !upper {
            q[0] += self.constant_term();
        }
        let pieces = &self.level2_pieces;
        let taylor_lanes: Vec<usize> = self.taylor.iter().map(|t| t.lane).collect();
        let readers: Vec<(usize, Reader<'_>)> = pieces
            .lanes()
            .into_iter()
            .filter(|l| !taylor_lanes.contains(l))
            .map(|l| (l, self.reader(l)))
            .collect();
        let taylors: Vec<(&Taylor, &[u8], &[E], usize)> = self
            .taylor
            .iter()
            .map(|taylor| {
                let State::Quartet(lut) = &self.states[taylor.lane] else {
                    unreachable!("Taylor lanes are quartet tables")
                };
                let column = &self.columns[self.lane_column[taylor.lane].expect("coded")];
                (taylor, &column.pairs[..], &lut[..], column.pair_space())
            })
            .collect();
        let sum = self.item_sum(|k, scratch, out| {
            // Taylor lanes first: their values also feed the other pieces reading them.
            for &(taylor, pairs, lut, space) in &taylors {
                let p = &pairs[4 * k..4 * k + 4];
                let left = p[0] as usize * space + p[1] as usize;
                let right = p[2] as usize * space + p[3] as usize;
                let (va, vb) = (lut[left], lut[right]);
                let delta = vb - va;
                scratch.a[taylor.lane] = va;
                scratch.d[taylor.lane] = delta;
                let t = &taylor.lines[left].0;
                if !upper {
                    lift(&mut out[0], taylor.kind.value(va));
                }
                mad(&mut out[1], t[0], delta);
                if taylor.degree >= 2 {
                    let d2 = fm(delta, delta);
                    mad(&mut out[2], t[1], d2);
                    if taylor.degree >= 3 {
                        let d3 = fm(d2, delta);
                        mad(&mut out[3], t[2], d3);
                        if taylor.degree == 4 {
                            mad(&mut out[4], t[3], fm(d2, d2));
                        }
                    }
                }
            }
            for (l, reader) in &readers {
                let (va, vb) = reader.item(k);
                scratch.a[*l] = va;
                scratch.d[*l] = vb - va;
            }
            if upper {
                pieces.add_upper(&scratch.a, &scratch.d, out);
            } else {
                pieces.add_at(&scratch.a, &scratch.d, out);
            }
        });
        for (slot, value) in q.iter_mut().zip(sum) {
            *slot += value;
        }
        q
    }

    /// The item sums of a round over the dense table (the fallback when the fold that
    /// produced the table did not evaluate them).
    fn dense_items(&self, upper: bool) -> Vec<E> {
        let (stride, dense) = (self.dense_stride, &self.dense);
        let pieces = &self.plan.dense;
        self.item_sum(|k, scratch, out| {
            for (l, (a, d)) in scratch.a.iter_mut().zip(&mut scratch.d).enumerate() {
                let x = dense[l * stride + 2 * k];
                *a = x;
                *d = dense[l * stride + 2 * k + 1] - x;
            }
            if upper {
                pieces.add_upper(&scratch.a, &scratch.d, out);
            } else {
                pieces.add_at(&scratch.a, &scratch.d, out);
            }
        })
    }

    /// The evaluation of the next round's item sums on a freshly folded block of the dense
    /// table: `(first, second)` the remaining equality tables after the bind (or ones).
    fn dense_evaluator(&self, upper: bool) -> DenseEval<'_> {
        let (first, second): (&[E], &[E]) = match &self.eq {
            Some(eq) => eq.remaining_eq_tables(),
            None => (&[], &[]),
        };
        DenseEval {
            pieces: &self.plan.dense,
            lanes: self.sources.len(),
            stride: self.dense_stride,
            first,
            second,
            weighted: self.eq.is_some(),
            upper,
            width: self.coefficient_count(),
        }
    }

    /// Bind the round's variable to `challenge`.
    pub fn bind(&mut self, round: usize, challenge: E) {
        assert_eq!(round, self.bound, "rounds in order");
        assert!(!self.pending.is_empty(), "round_message before bind");
        self.claim = evaluate_compressed(&self.pending, self.claim, challenge);
        self.pending.clear();
        if round < self.inactive {
            self.scale *= inverse_power_of_two(1);
            self.bound += 1;
            return;
        }
        // The equality factor first: the folds into the dense table evaluate the next
        // round's item sums with the next round's weights.
        if let Some(eq) = &mut self.eq {
            eq.bind(challenge);
        }
        self.challenges.push(challenge);
        self.bound += 1;
        match self.level() - 1 {
            0 => self.bind_first(challenge),
            1 => self.bind_second(challenge),
            2 => self.bind_third(challenge),
            _ => match &mut self.tail {
                Some(tail) => {
                    let last = self.bound == self.inactive + self.local_vars;
                    tail.bind(challenge, last);
                }
                None => self.bind_dense(challenge),
            },
        }
    }

    /// Bind 0: pair tables for coded lanes, folded tables for the others.
    fn bind_first(&mut self, challenge: E) {
        let states: Vec<State> = (0..self.sources.len())
            .into_par_iter()
            .map(|l| {
                let scale = self.plan.scales[l];
                match self.lane_column[l] {
                    Some(c) => {
                        let column = &self.columns[c];
                        let n = column.classes;
                        let lut = (0..n * n)
                            .map(|pattern| {
                                let (va, vd) = self.class_values(l, column, pattern / n, pattern % n);
                                let value = va + fm(challenge, vd);
                                if scale == E::one() { value } else { fm(value, scale) }
                            })
                            .collect();
                        State::Pair(lut)
                    }
                    None => {
                        let mut table = match &self.sources[l] {
                            Lane::Source(source) => source.fold(challenge),
                            Lane::Folded(cells) => {
                                let context = E::precompute(challenge);
                                cells.chunks_exact(2).map(|p| fold(&context, p[0], p[1])).collect()
                            }
                        };
                        if scale != E::one() {
                            table.par_iter_mut().for_each(|cell| *cell = fm(*cell, scale));
                        }
                        State::Table(table)
                    }
                }
            })
            .collect();
        self.states = states;
    }

    /// Bind 1: quartet tables (and Taylor tables) or the pair table with the challenge.
    fn bind_second(&mut self, challenge: E) {
        self.masses = None;
        let context = E::precompute(challenge);
        let octets = if self.local_vars >= 3 { 1_usize << (self.local_vars - 3) } else { 0 };
        let states: Vec<State> = std::mem::take(&mut self.states)
            .into_par_iter()
            .enumerate()
            .map(|(l, state)| match state {
                State::Pair(lut) => {
                    let column = &self.columns[self.lane_column[l].expect("coded")];
                    let space = column.pair_space();
                    let quartets = space * space;
                    let use_lut = match self.policy.quartet_lut {
                        Choice::Force => true,
                        Choice::Forbid => false,
                        Choice::Auto => quartets <= MAX_QUARTET_LUT && octets >= quartets,
                    };
                    if use_lut {
                        let mut quartet = Vec::with_capacity(quartets);
                        for &even in &lut {
                            for &odd in &lut {
                                quartet.push(fold(&context, even, odd));
                            }
                        }
                        State::Quartet(quartet)
                    } else {
                        State::PairFold(lut, challenge)
                    }
                }
                State::Table(table) => {
                    let folded = if table.len() / 2 >= MIN_PARALLEL_ITEMS {
                        table.par_chunks_exact(2).map(|p| fold(&context, p[0], p[1])).collect()
                    } else {
                        table.chunks_exact(2).map(|p| fold(&context, p[0], p[1])).collect()
                    };
                    State::Table(folded)
                }
                other => unreachable!("level-1 state {other:?}"),
            })
            .collect();
        self.states = states;
        // Taylor tables of the polynomials (degree 2 to 4) on quartet lanes.
        self.taylor.clear();
        if self.policy.taylor && self.local_vars >= 3 {
            for (l, kind) in &self.plan.folded.polys {
                let degree = kind.degree();
                if let (State::Quartet(lut), true) = (&self.states[*l], (2..=4).contains(&degree)) {
                    let lines: Vec<TaylorLine> = lut
                        .par_iter()
                        .map(|&v| {
                            let t = kind.taylor(v);
                            TaylorLine([t[1], t[2], t[3], t[4]])
                        })
                        .collect();
                    self.taylor.push(Taylor {
                        lane: *l,
                        degree,
                        kind: kind.clone(),
                        lines,
                    });
                }
            }
        }
        let taylor_lanes: Vec<usize> = self.taylor.iter().map(|t| t.lane).collect();
        let mut level2 = self.plan.folded.clone();
        level2.polys.retain(|(l, _)| !taylor_lanes.contains(l));
        self.level2_pieces = level2;
    }

    /// Bind 2: every lane into the dense lane-major table at an eighth of the domain, and
    /// the item sums of the first dense round on every block as it is written.
    #[allow(unsafe_code)]
    fn bind_third(&mut self, challenge: E) {
        self.taylor.clear();
        let context = E::precompute(challenge);
        let lanes = self.sources.len();
        let rows = 1 << (self.local_vars - 3);
        if let Some(terms) = self.tail_terms.take() {
            // The plane path has no lane shifts (checked when the tail was compiled).
            let readers: Vec<Reader<'_>> = (0..lanes).map(|l| self.reader(l)).collect();
            let table = Tail::new_filled(terms, lanes, rows, |l, row| {
                let (even, odd) = readers[l].item(row);
                let x = fold(&context, even, odd);
                (x.c0().to_canonical_u64(), x.c1().to_canonical_u64())
            });
            drop(readers);
            self.tail = Some(table);
            self.dense = Vec::new();
            self.dense_stride = 0;
            self.dense_rows = 0;
            self.states = vec![State::Dense; lanes];
            for column in &mut self.columns {
                column.pairs = Vec::new();
            }
            return;
        }
        self.dense_stride = rows;
        self.dense_rows = rows;
        let readers: Vec<Reader<'_>> = (0..lanes).map(|l| self.reader(l)).collect();
        let shifts = &self.plan.shifts;
        // The next round exists when the table has pairs; its constant coefficient comes
        // from the claim when recoverable.
        let evaluate = rows >= 2;
        let upper = evaluate && self.recovery().is_some();
        let eval = self.dense_evaluator(upper);
        let width = eval.width;
        let mut dense = match &self.arena {
            Some(arena) => lock_arena(arena).take(rows * lanes),
            None => buf::reuse(Vec::new(), rows * lanes),
        };
        let base = SendPtr(dense.as_mut_ptr().cast::<E>());
        // Row blocks filled lane by lane, so each lane's lookup table stays in cache while
        // the block (2^9 rows) is written; then the block's pairs are evaluated while they
        // are still in cache.
        let blocks = rows.div_ceil(DENSE_BLOCK_ROWS);
        let run = |block: usize| -> Vec<Acc> {
            let first = block * DENSE_BLOCK_ROWS;
            let count = DENSE_BLOCK_ROWS.min(rows - first);
            for (l, (reader, &shift)) in readers.iter().zip(shifts).enumerate() {
                // SAFETY: the plane segment of lane `l` and rows `first..first + count` is
                // written by this block only; every slot of the table is covered once. The
                // slots are uninitialized, so they are viewed as `MaybeUninit` and written.
                let out = unsafe {
                    std::slice::from_raw_parts_mut(base.get().add(l * rows + first).cast::<std::mem::MaybeUninit<E>>(), count)
                };
                for (r, slot) in out.iter_mut().enumerate() {
                    let (even, odd) = reader.item(first + r);
                    slot.write(fold(&context, even, odd) - shift);
                }
            }
            if evaluate {
                let mut scratch = Scratch::new(lanes);
                // SAFETY: the block's rows of every lane were written above and no other
                // block writes them.
                unsafe { eval.items(base.get(), first / 2, (first + count) / 2, &mut scratch) }
            } else {
                Vec::new()
            }
        };
        // Parallel once the table holds a few thousand cells (as for the other folds).
        let total = if blocks > 1 && rows * lanes >= MIN_PARALLEL_ITEMS {
            (0..blocks).into_par_iter().map(run).reduce(|| vec![Acc::zero(); width], add_acc)
        } else {
            (0..blocks).map(run).fold(vec![Acc::zero(); width], add_acc)
        };
        // SAFETY: every row of every lane was written by exactly one block.
        self.dense = unsafe { buf::finish(dense) };
        self.states = vec![State::Dense; lanes];
        if evaluate {
            self.cached_next = Some(total.into_iter().map(E::reduce_product).collect());
        }
        // The pair codes are not read again (one byte per pair of cells and column).
        for column in &mut self.columns {
            column.pairs = Vec::new();
        }
    }

    /// A dense bind: fold every plane (rows `2k, 2k + 1` into `k`) into a fresh table from
    /// the arena, block by block in parallel, and evaluate the next round's item sums on
    /// every block as it is written (the block is still in cache). The previous table
    /// returns to the arena, so the two buffers alternate round after round.
    #[allow(unsafe_code)]
    fn bind_dense(&mut self, challenge: E) {
        let lanes = self.sources.len();
        let (stride, rows) = (self.dense_stride, self.dense_rows);
        debug_assert!(rows >= 2 && rows.is_power_of_two());
        let next_rows = rows / 2;
        let context = E::precompute(challenge);
        let evaluate = next_rows >= 2;
        let upper = evaluate && self.recovery().is_some();
        let mut next = match &self.arena {
            Some(arena) => lock_arena(arena).take(next_rows * lanes),
            None => buf::reuse(Vec::new(), next_rows * lanes),
        };
        let out = SendPtr(next.as_mut_ptr().cast::<E>());
        let previous = std::mem::take(&mut self.dense);
        // The evaluator reads the new table (stride `next_rows`).
        self.dense_stride = next_rows;
        self.dense_rows = next_rows;
        let eval = self.dense_evaluator(upper);
        let width = eval.width;
        let source = &previous;
        let run = |block: usize| -> Vec<Acc> {
            let r0 = block * DENSE_BLOCK_ROWS;
            let r1 = (r0 + DENSE_BLOCK_ROWS).min(next_rows);
            for l in 0..lanes {
                let plane = &source[l * stride..l * stride + rows];
                // SAFETY: rows `r0..r1` of plane `l` of the new table are written by this
                // block only; every slot is covered once. The slots are uninitialized, so
                // they are viewed as `MaybeUninit` and written.
                let out = unsafe {
                    std::slice::from_raw_parts_mut(out.get().add(l * next_rows + r0).cast::<std::mem::MaybeUninit<E>>(), r1 - r0)
                };
                for (r, slot) in (r0..r1).zip(out.iter_mut()) {
                    slot.write(fold(&context, plane[2 * r], plane[2 * r + 1]));
                }
            }
            if evaluate {
                let mut scratch = Scratch::new(lanes);
                // SAFETY: rows `r0..r1` of every plane were written above by this block.
                unsafe { eval.items(out.get(), r0 / 2, r1 / 2, &mut scratch) }
            } else {
                Vec::new()
            }
        };
        let blocks = next_rows.div_ceil(DENSE_BLOCK_ROWS);
        let total = if blocks > 1 && next_rows * lanes >= MIN_PARALLEL_ITEMS {
            (0..blocks).into_par_iter().map(run).reduce(|| vec![Acc::zero(); width], add_acc)
        } else {
            (0..blocks).map(run).fold(vec![Acc::zero(); width], add_acc)
        };
        if let Some(arena) = &self.arena {
            lock_arena(arena).put(previous);
        }
        // SAFETY: every row of every plane of the new table was written by exactly one block.
        self.dense = unsafe { buf::finish(next) };
        if evaluate {
            self.cached_next = Some(total.into_iter().map(E::reduce_product).collect());
        }
    }

    /// The fully bound lane values, in lane order (unscaled).
    #[must_use]
    pub fn terminal_values(&self) -> Vec<E> {
        assert_eq!(self.bound, self.inactive + self.local_vars, "all rounds bound");
        let inverse = self.plan.inverse_scales();
        (0..self.sources.len())
            .map(|l| {
                let value = self.value_at(l, 0);
                if self.local_vars == 0 {
                    value
                } else {
                    value * inverse[l]
                }
            })
            .collect()
    }
}

/// Rows per block of the dense table: filled, folded and evaluated while in cache.
const DENSE_BLOCK_ROWS: usize = 1 << 9;

/// A raw table pointer shared by the parallel tasks of a fold (each writes its own rows).
#[derive(Clone, Copy)]
struct SendPtr(*mut E);
#[allow(unsafe_code)]
unsafe impl Send for SendPtr {}
#[allow(unsafe_code)]
unsafe impl Sync for SendPtr {}

impl SendPtr {
    /// The pointer (a method, so that closures capture the whole wrapper).
    fn get(self) -> *mut E {
        self.0
    }
}

fn add_acc(mut left: Vec<Acc>, right: Vec<Acc>) -> Vec<Acc> {
    for (l, r) in left.iter_mut().zip(right) {
        *l += r;
    }
    left
}

impl Scratch {
    fn new(lanes: usize) -> Self {
        Self {
            a: vec![E::zero(); lanes],
            d: vec![E::zero(); lanes],
        }
    }
}

/// The next round's item sums over a range of pairs of the dense table (lane-major, read
/// through a raw pointer right after the pairs were written).
struct DenseEval<'g> {
    pieces: &'g Pieces,
    lanes: usize,
    stride: usize,
    first: &'g [E],
    second: &'g [E],
    weighted: bool,
    upper: bool,
    width: usize,
}

impl DenseEval<'_> {
    /// `sum_k first[k & mask] second[k >> bits] c(k)` over the pairs `k0..k1` (or the plain
    /// sum without an equality factor), unreduced per coefficient: the block sums of
    /// `first` are reduced once and weighted by `second`, as `split_sum` does.
    ///
    /// # Safety
    /// Rows `2 k0..2 k1` of every plane at `base` must be initialized and not written
    /// concurrently.
    #[allow(unsafe_code)]
    unsafe fn items(&self, base: *const E, k0: usize, k1: usize, scratch: &mut Scratch) -> Vec<Acc> {
        let width = self.width;
        let mut coefficients = vec![Acc::zero(); width];
        let mut outer = vec![Acc::zero(); width];
        let mut inner = vec![Acc::zero(); width];
        let (first_bits, first_mask) = if self.weighted {
            (self.first.len().trailing_zeros(), self.first.len() - 1)
        } else {
            (0, 0)
        };
        let mut block = k0 >> first_bits;
        for k in k0..k1 {
            for l in 0..self.lanes {
                // SAFETY: the caller guarantees rows `2k`, `2k + 1` of plane `l` are initialized.
                let x = unsafe { *base.add(l * self.stride + 2 * k) };
                let y = unsafe { *base.add(l * self.stride + 2 * k + 1) };
                scratch.a[l] = x;
                scratch.d[l] = y - x;
            }
            if !self.weighted {
                if self.upper {
                    self.pieces.add_upper(&scratch.a, &scratch.d, &mut outer);
                } else {
                    self.pieces.add_at(&scratch.a, &scratch.d, &mut outer);
                }
                continue;
            }
            if (k >> first_bits) != block {
                let scale = self.second[block];
                for (o, i) in outer.iter_mut().zip(&mut inner) {
                    *o += scale.mul_unreduced(E::reduce_product(*i));
                    *i = Acc::zero();
                }
                block = k >> first_bits;
            }
            for c in &mut coefficients {
                *c = Acc::zero();
            }
            if self.upper {
                self.pieces.add_upper(&scratch.a, &scratch.d, &mut coefficients);
            } else {
                self.pieces.add_at(&scratch.a, &scratch.d, &mut coefficients);
            }
            let weight = self.first[k & first_mask];
            for (i, &c) in inner.iter_mut().zip(&coefficients) {
                *i += weight.mul_unreduced(E::reduce_product(c));
            }
        }
        if self.weighted && k0 < k1 {
            let scale = self.second[block];
            for (o, i) in outer.iter_mut().zip(&inner) {
                *o += scale.mul_unreduced(E::reduce_product(*i));
            }
        }
        outer
    }
}

impl Drop for Group<'_> {
    fn drop(&mut self) {
        if let Some(arena) = &self.arena {
            lock_arena(arena).put(std::mem::take(&mut self.dense));
        }
    }
}
