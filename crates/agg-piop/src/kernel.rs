//! Generic sum-check kernels: narrow source tables lifted lazily, lanes folded in place,
//! and groups of batched instances over one local domain (spec Section 7.1).
//!
//! Mirrors the shape of Aerie's kernels (`aerie-prover/src/protocol/kernels/`): every
//! committed table stays an `i8` (or `i32`) array until the first challenge binds its low
//! variable, and only the half-size folded table is lifted to `K`. A [`Group`] holds the
//! lanes of one local domain, an optional shared equality factor (Akita's `GruenSplitEq`,
//! so the equality table of the remaining variables is never materialized) and the
//! gamma-weighted [`Term`]s of the instances batched on it. The round message of a group
//! is the compressed ordinary message of Aerie 10.3: the coefficients of the degree-`d`
//! round polynomial without the linear one, which the verifier recovers from the running
//! claim (`g(0) + g(1) = claim`).
//!
//! Instances on a smaller domain than the stage are inactive in the leading rounds, where
//! the message is the constant `claim / 2`, and are lifted by `2^-inactive` (Aerie 7.1).
//!
//! Evaluation strategy (`docs/piop.md` Section 16.2): the terms of a group are compiled
//! once into a [`Plan`] (one polynomial per lane, weighted affine forms, bilinear
//! products, residual generic terms). In the first active round of a group whose lanes
//! are *coded* byte columns (cells in a span of at most 16 values), the equality weights
//! (or the weight lane's values) are bucketed by the pattern of the two cells of each pair
//! and every term is evaluated once per pattern (Aerie's `CodedSource` buckets); the first
//! fold of a coded lane is a 256-entry lookup. All paths compute the same polynomials.

use akita_algebra::split_eq::GruenSplitEq;
use jolt_field::{Field, Fold, One, Ring, Zero};
use jolt_poly::UnivariatePoly;
use rayon::prelude::*;

use crate::OpeningField as E;

/// Pairs per parallel task of the generic path.
const PAIR_CHUNK: usize = 1 << 10;
/// Pairs per parallel task of the bucketed path (the bucket arrays amortize over it).
const BUCKET_CHUNK: usize = 1 << 14;
/// Below this many pairs a round runs serially.
const MIN_PARALLEL_PAIRS: usize = 1 << 12;
/// Largest span of a coded column (one nibble per cell).
const MAX_SPAN: usize = 16;
/// Largest joint pattern space of a multi-column generic term in the bucketed path.
const MAX_JOINT_PATTERNS: usize = 1 << 12;
/// Round 1 is bucketed by quartet patterns only when the pairs outnumber the largest
/// pattern space by this factor (Aerie's `BUCKET_MIN_RATIO`).
const BUCKET_MIN_RATIO: usize = 4;
/// Pairs per parallel task of the round-1 bucketed path (65,536-entry buckets).
const BUCKET_CHUNK_L1: usize = 1 << 16;

/// A read-only integer or field table, the state of a lane before its first fold.
#[derive(Clone, Copy, Debug)]
pub enum Source<'a> {
    /// Committed `i8` cells seen through a per-cell integer map (`map[cell as u8]`).
    Bytes {
        /// The cells.
        cells: &'a [i8],
        /// The map applied to every cell (identity for a committed column).
        map: &'a [i64; 256],
    },
    /// A small-integer table (the virtual parities `y^(1)`).
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
        }
    }

    /// `(v(2k), v(2k + 1))` lifted.
    #[inline]
    fn pair(&self, k: usize) -> (E, E) {
        (self.value(2 * k), self.value(2 * k + 1))
    }

    /// The table folded by `challenge` on its low variable: `v(0, y) + r (v(1, y) - v(0, y))`.
    fn fold(&self, challenge: E) -> Vec<E> {
        let half = self.len() / 2;
        let context = E::precompute(challenge);
        let fold_pair = |k: usize| match self {
            Self::Bytes { cells, map } => {
                let low = map[cells[2 * k] as u8 as usize];
                let high = map[cells[2 * k + 1] as u8 as usize];
                E::from_i64(low) + challenge.mul_i64(high - low)
            }
            Self::Ints(cells) => {
                let (low, high) = (i64::from(cells[2 * k]), i64::from(cells[2 * k + 1]));
                E::from_i64(low) + challenge.mul_i64(high - low)
            }
            Self::Field(cells) => E::fold_one(&context, cells[2 * k], cells[2 * k + 1]),
        };
        if half >= MIN_PARALLEL_PAIRS {
            (0..half).into_par_iter().map(fold_pair).collect()
        } else {
            (0..half).map(fold_pair).collect()
        }
    }
}

/// One multilinear table being bound variable by variable.
#[derive(Clone, Debug)]
pub enum Lane<'a> {
    /// Not yet folded: the narrow source.
    Source(Source<'a>),
    /// A coded byte lane folded once: cell `k` is `lut[column.pattern(k)]` (no storage per
    /// cell). The second fold materializes a `K` table of a quarter of the domain: a
    /// quartet-pattern lookup (65,536 entries per digit column) was measured slower in the
    /// following round than the streamed table (random accesses into 1 MiB tables).
    Lookup {
        /// The coded column.
        column: Column<'a>,
        /// The folded value of every pair pattern (scaled).
        lut: Vec<E>,
    },
    /// Folded at least once and materialized.
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
            Self::Lookup { column, .. } => column.cells.len() / 2,
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
            Self::Lookup { column, lut } => lut[column.pattern(index)],
            Self::Folded(cells) => cells[index],
        }
    }

    #[inline]
    fn pair(&self, k: usize) -> (E, E) {
        match self {
            Self::Source(source) => source.pair(k),
            Self::Lookup { column, lut } => (lut[column.pattern(2 * k)], lut[column.pattern(2 * k + 1)]),
            Self::Folded(cells) => (cells[2 * k], cells[2 * k + 1]),
        }
    }

    /// Bind the low variable to `challenge`.
    pub fn fold(&mut self, challenge: E) {
        match self {
            Self::Source(source) => *self = Self::Folded(source.fold(challenge)),
            Self::Lookup { column, lut } => {
                let half = column.cells.len() / 4;
                let context = E::precompute(challenge);
                let fold_pair = |k: usize| {
                    E::fold_one(&context, lut[column.pattern(2 * k)], lut[column.pattern(2 * k + 1)])
                };
                let cells = if half >= MIN_PARALLEL_PAIRS {
                    (0..half).into_par_iter().map(fold_pair).collect()
                } else {
                    (0..half).map(fold_pair).collect()
                };
                *self = Self::Folded(cells);
            }
            Self::Folded(cells) => {
                let half = cells.len() / 2;
                let context = E::precompute(challenge);
                if half >= MIN_PARALLEL_PAIRS {
                    *cells = cells
                        .par_chunks_exact(2)
                        .map(|pair| E::fold_one(&context, pair[0], pair[1]))
                        .collect();
                } else {
                    for k in 0..half {
                        cells[k] = E::fold_one(&context, cells[2 * k], cells[2 * k + 1]);
                    }
                    cells.truncate(half);
                }
            }
        }
    }

    /// The fully bound value (the multilinear extension at the stage point).
    #[must_use]
    pub fn terminal(&self) -> E {
        assert_eq!(self.len(), 1, "lane fully bound");
        self.value(0)
    }
}

/// The leaf predicate of a digit alphabet (spec Section 4.3), as a polynomial in `P_Z`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafPoly {
    /// `Q_4(P) = P (P - 2)`.
    Q4,
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

    /// The leaf of a digit base: `Q_4`, `Q_8`, or both halves of `Q_16` (lo then hi).
    #[must_use]
    pub fn of_base(base: u32) -> Vec<Self> {
        match base {
            4 => vec![Self::Q4],
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
    /// `b_5 (b_4 + b_3 b_2)` (spec (7.4)), lanes in that order.
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

// ---------------------------------------------------------------------------------------
// Compiled evaluation plan.
// ---------------------------------------------------------------------------------------

/// The gamma-weighted polynomial in one lane's value.
#[derive(Clone, Debug, PartialEq, Eq)]
enum LanePoly {
    /// The lane is not read by a single-lane piece.
    None,
    /// The lane value itself (a scaled lane).
    Identity,
    /// Coefficients, constant term first (degree at least 1).
    Poly(Vec<E>),
}

impl LanePoly {
    fn add(&mut self, coefficients: &[E]) {
        let mut current = match std::mem::replace(self, Self::None) {
            Self::None => Vec::new(),
            Self::Identity => vec![E::zero(), E::one()],
            Self::Poly(c) => c,
        };
        if current.len() < coefficients.len() {
            current.resize(coefficients.len(), E::zero());
        }
        for (slot, &c) in current.iter_mut().zip(coefficients) {
            *slot += c;
        }
        while current.len() > 1 && current.last() == Some(&E::zero()) {
            current.pop();
        }
        *self = if current.len() <= 1 && current.iter().all(E::is_zero) {
            Self::None
        } else {
            Self::Poly(current)
        };
    }

    /// `(c)` when the polynomial is exactly `c v`.
    fn linear_coefficient(&self) -> Option<E> {
        match self {
            Self::Poly(c) if c.len() == 2 && c[0].is_zero() => Some(c[1]),
            _ => None,
        }
    }

    #[inline]
    fn evaluate(&self, v: E) -> E {
        match self {
            Self::None => E::zero(),
            Self::Identity => v,
            Self::Poly(c) => {
                let mut acc = c[c.len() - 1];
                for &coefficient in c[..c.len() - 1].iter().rev() {
                    acc = acc * v + coefficient;
                }
                acc
            }
        }
    }
}

/// `weight * (sum c_l v_l + constant)` with `K` coefficients.
#[derive(Clone, Debug)]
struct WeightedAffine {
    weight: usize,
    lanes: Vec<(usize, E)>,
    constant: E,
}

/// The pieces of a plan that depend on whether the lanes are still unscaled sources or
/// folded (and possibly scaled) tables.
#[derive(Clone, Debug)]
struct Phase {
    /// Per lane, the single-lane polynomial.
    polys: Vec<LanePoly>,
    /// `coefficient * v_a * v_b`.
    bilinear: Vec<(E, usize, usize)>,
}

/// The compiled terms of a group.
#[derive(Clone, Debug)]
struct Plan {
    /// Read from the unscaled source values (first active round).
    source: Phase,
    /// Read from the folded values (later rounds).
    folded: Phase,
    /// Per lane, the factor applied at the first fold (one for most lanes).
    scales: Vec<E>,
    /// The gamma-weighted constants of the weightless affine forms (per cell).
    constant: E,
    weighted: Vec<WeightedAffine>,
    /// `gamma * term` for everything else.
    generic: Vec<(E, Term)>,
}

impl Plan {
    fn compile(lane_count: usize, instances: &[GroupInstance]) -> Self {
        let mut source_polys = vec![LanePoly::None; lane_count];
        let mut constant = E::zero();
        let mut weighted = Vec::new();
        let mut bilinear = Vec::new();
        let mut generic = Vec::new();
        for instance in instances {
            let gamma = instance.gamma;
            for term in &instance.terms {
                match term {
                    Term::Product(lanes) if lanes.len() == 1 => {
                        source_polys[lanes[0]].add(&[E::zero(), gamma]);
                    }
                    Term::Product(lanes) if lanes.len() == 2 => {
                        bilinear.push((gamma, lanes[0], lanes[1]));
                    }
                    Term::Bool(l) => source_polys[*l].add(&[E::zero(), -gamma, gamma]),
                    Term::Link(l) => source_polys[*l].add(&[E::zero(), gamma, gamma]),
                    Term::Leaf(l, poly) => {
                        let coefficients: Vec<E> = poly
                            .coefficients()
                            .iter()
                            .map(|&c| gamma.mul_i64(c))
                            .collect();
                        source_polys[*l].add(&coefficients);
                    }
                    Term::Affine {
                        weight: None,
                        lanes,
                        constant: c,
                    } => {
                        constant += gamma.mul_i64(*c);
                        for &(l, coefficient) in lanes {
                            source_polys[l].add(&[E::zero(), gamma.mul_i64(coefficient)]);
                        }
                    }
                    Term::Affine {
                        weight: Some(w),
                        lanes,
                        constant: c,
                    } => weighted.push(WeightedAffine {
                        weight: *w,
                        lanes: lanes
                            .iter()
                            .map(|&(l, coefficient)| (l, gamma.mul_i64(coefficient)))
                            .collect(),
                        constant: gamma.mul_i64(*c),
                    }),
                    other => generic.push((gamma, other.clone())),
                }
            }
        }
        // Uses of every lane outside its own polynomial.
        let mut uses = vec![0_usize; lane_count];
        for form in &weighted {
            uses[form.weight] += 1;
            for &(l, _) in &form.lanes {
                uses[l] += 1;
            }
        }
        for &(_, a, b) in &bilinear {
            uses[a] += 1;
            uses[b] += 1;
        }
        for (_, term) in &generic {
            for l in term.lanes() {
                uses[l] += 1;
            }
        }
        let mut folded_polys = source_polys.clone();
        let mut folded_bilinear = bilinear.clone();
        let mut scales = vec![E::one(); lane_count];
        // A lane read only by its own linear polynomial is scaled at its first fold.
        for l in 0..lane_count {
            if uses[l] == 0 {
                if let Some(c) = source_polys[l].linear_coefficient() {
                    if !c.is_zero() {
                        scales[l] = c;
                        folded_polys[l] = LanePoly::Identity;
                    }
                }
            }
        }
        // A lane read by exactly one bilinear product and nothing else absorbs its
        // coefficient at the first fold.
        for (coefficient, a, b) in &mut folded_bilinear {
            for l in [*a, *b] {
                if uses[l] == 1 && source_polys[l] == LanePoly::None && !coefficient.is_zero() {
                    scales[l] = *coefficient;
                    *coefficient = E::one();
                    break;
                }
            }
        }
        Self {
            source: Phase {
                polys: source_polys,
                bilinear,
            },
            folded: Phase {
                polys: folded_polys,
                bilinear: folded_bilinear,
            },
            scales,
            constant,
            weighted,
            generic,
        }
    }

    /// The plan without its per-cell constant (added by the caller with the right weight).
    #[inline]
    fn evaluate(&self, phase: &Phase, values: &[E]) -> E {
        let mut acc = E::zero();
        for (l, poly) in phase.polys.iter().enumerate() {
            match poly {
                LanePoly::None => {}
                LanePoly::Identity => acc += values[l],
                LanePoly::Poly(_) => acc += poly.evaluate(values[l]),
            }
        }
        for form in &self.weighted {
            let mut affine = form.constant;
            for &(l, c) in &form.lanes {
                affine += c * values[l];
            }
            acc += values[form.weight] * affine;
        }
        for &(c, a, b) in &phase.bilinear {
            let product = values[a] * values[b];
            acc += if c == E::one() { product } else { c * product };
        }
        for (gamma, term) in &self.generic {
            acc += *gamma * term.evaluate(values);
        }
        acc
    }
}

// ---------------------------------------------------------------------------------------
// Coded columns and the bucketed first round.
// ---------------------------------------------------------------------------------------

/// A distinct `i8` column read by one or more lanes, with every cell in
/// `[min, min + span)` and `span <= 16`, so a pair of cells has one of `span^2` patterns
/// and a quartet (two consecutive pairs) one of `span^4`.
#[derive(Clone, Copy, Debug)]
pub struct Column<'a> {
    cells: &'a [i8],
    min: i8,
    span: usize,
    /// Bits of a pair pattern (`ceil(log2(span^2))`), so quartet patterns are shifts.
    pair_bits: u32,
}

impl<'a> Column<'a> {
    fn new(cells: &'a [i8], min: i8, span: usize) -> Self {
        let pair_bits = (span * span).next_power_of_two().trailing_zeros();
        Self {
            cells,
            min,
            span,
            pair_bits,
        }
    }

    /// The pattern of pair `k`: `(cell(2k) - min) * span + (cell(2k + 1) - min)`.
    #[inline]
    fn pattern(&self, k: usize) -> usize {
        let a = (i16::from(self.cells[2 * k]) - i16::from(self.min)) as usize;
        let b = (i16::from(self.cells[2 * k + 1]) - i16::from(self.min)) as usize;
        a * self.span + b
    }

    /// Pair patterns.
    fn patterns(&self) -> usize {
        self.span * self.span
    }

    /// The pattern of quartet `k` (pairs `2k`, `2k + 1`): `pattern(2k) << pair_bits | pattern(2k + 1)`.
    #[inline]
    fn quartet(&self, k: usize) -> usize {
        (self.pattern(2 * k) << self.pair_bits) | self.pattern(2 * k + 1)
    }

    /// Quartet patterns (a power of two).
    fn quartets(&self) -> usize {
        1 << (2 * self.pair_bits)
    }

    /// The two pair patterns of a quartet pattern.
    #[inline]
    fn pairs_of(&self, quartet: usize) -> (usize, usize) {
        (quartet >> self.pair_bits, quartet & ((1 << self.pair_bits) - 1))
    }

    /// The pattern space at `level` (0: pairs, 1: quartets).
    fn patterns_at(&self, level: usize) -> usize {
        if level == 0 {
            self.patterns()
        } else {
            self.quartets()
        }
    }

    /// The pattern of pair `k` at `level`.
    #[inline]
    fn pattern_at(&self, level: usize, k: usize) -> usize {
        if level == 0 {
            self.pattern(k)
        } else {
            self.quartet(k)
        }
    }

    /// The two cells of a pair pattern.
    fn cells_of(&self, pattern: usize) -> (i8, i8) {
        let a = (pattern / self.span) as i8;
        let b = (pattern % self.span) as i8;
        (self.min.wrapping_add(a), self.min.wrapping_add(b))
    }
}

/// `(min, max)` of a byte column.
fn byte_range(cells: &[i8]) -> (i8, i8) {
    let combine = |(a_min, a_max): (i8, i8), (b_min, b_max): (i8, i8)| {
        (a_min.min(b_min), a_max.max(b_max))
    };
    let of_chunk = |chunk: &[i8]| {
        chunk
            .iter()
            .fold((i8::MAX, i8::MIN), |(lo, hi), &c| (lo.min(c), hi.max(c)))
    };
    if cells.len() >= 1 << 16 {
        cells
            .par_chunks(1 << 16)
            .map(of_chunk)
            .reduce(|| (i8::MAX, i8::MIN), combine)
    } else {
        of_chunk(cells)
    }
}

/// How a lane's values arise in the bucketed round.
#[derive(Clone, Copy, Debug)]
enum Coded<'a> {
    /// A coded byte lane: its column and its map.
    Column(usize, &'a [i64; 256]),
    /// Anything else.
    No,
}

/// A piece of the plan rewritten for the bucketed round.
#[derive(Clone, Debug)]
enum Piece {
    /// `coefficient * v_a * v_b` with both lanes on one coded column.
    SameColumn(usize, E, usize, usize),
    /// A weight lane times a coded lane: `coefficient * f * v_l` (bucket key `(f, column)`).
    Weighted(usize, E, usize),
    /// A weight lane times a constant (bucket: any key of `f`).
    WeightConstant(usize, E),
    /// A generic term on coded lanes only, with its joint pattern space.
    Joint(E, Term, Vec<usize>),
}

/// The precomputed bucketed-round layout of a group.
#[derive(Clone, Debug)]
struct Buckets {
    /// Columns whose lanes are read by single-lane polynomials or same-column products.
    pure: Vec<usize>,
    /// Distinct `(weight lane, column)` keys.
    keys: Vec<(usize, usize)>,
    pieces: Vec<Piece>,
}

/// The lanes of one local domain and the instances batched on them.
pub struct Group<'a> {
    eq: Option<GruenSplitEq<E>>,
    lanes: Vec<Lane<'a>>,
    plan: Plan,
    columns: Vec<Column<'a>>,
    coded: Vec<Coded<'a>>,
    /// Bucket layouts for the source phase (round 0) and the lookup phase (round 1).
    buckets: Option<(Buckets, Buckets)>,
    bucket_min_ratio: usize,
    degree: usize,
    inactive: usize,
    local_vars: usize,
    bound: usize,
    scale: E,
    claim: E,
    pending: Vec<E>,
}

impl<'a> Group<'a> {
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
        let eq = match eq_point {
            Some(point) => {
                if point.len() != local_vars {
                    return Err("equality point of the wrong length".into());
                }
                Some(GruenSplitEq::new(point).map_err(|e| e.to_string())?)
            }
            None => None,
        };
        let plan = Plan::compile(lanes.len(), &instances);
        let (columns, coded) = code_columns(&lanes);
        let buckets = match (
            bucket_layout(&plan, &plan.source, &lanes, &columns, &coded),
            bucket_layout(&plan, &plan.folded, &lanes, &columns, &coded),
        ) {
            (Some(source), Some(folded)) => Some((source, folded)),
            _ => None,
        };
        let mut group = Self {
            eq,
            lanes,
            plan,
            columns,
            coded,
            buckets,
            bucket_min_ratio: BUCKET_MIN_RATIO,
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
    /// two-point inner evaluation (`eq(tau_0, 0) q(0) + eq(tau_0, 1) q(1)`).
    fn direct_claim(&self) -> E {
        let q = self.inner_evaluations(2);
        match &self.eq {
            Some(eq) => {
                let tau = eq.current_tau();
                (E::one() - tau) * q[0] + tau * q[1]
            }
            None => q[0] + q[1],
        }
    }

    /// The current gamma-weighted claim.
    #[must_use]
    pub fn claim(&self) -> E {
        self.claim
    }

    /// True when the first active round runs the bucketed path.
    #[must_use]
    pub fn is_bucketed(&self) -> bool {
        self.buckets.is_some()
    }

    /// The same group evaluated by the generic per-pair path only (tests).
    #[cfg(test)]
    fn without_buckets(mut self) -> Self {
        self.buckets = None;
        if self.inactive > 0 {
            self.claim = self.direct_claim();
        }
        self
    }

    /// Bucket round 1 whatever the pair count (tests on small domains).
    #[cfg(test)]
    fn with_bucket_ratio(mut self, ratio: usize) -> Self {
        self.bucket_min_ratio = ratio;
        self
    }

    /// The compact level of the next round: 0 while no lane has been folded, 1 after
    /// one fold (coded lanes are lookups), 2 and more after.
    fn level(&self) -> usize {
        (self.bound + 1).saturating_sub(self.inactive + 1)
    }

    /// Leading inactive rounds.
    #[must_use]
    pub fn inactive_rounds(&self) -> usize {
        self.inactive
    }

    /// True while no lane has been folded (the first active round).
    fn in_source_phase(&self) -> bool {
        self.bound <= self.inactive
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
        let points = if self.eq.is_some() {
            self.degree
        } else {
            self.degree + 1
        };
        let evaluations = self.inner_evaluations(points);
        let q = UnivariatePoly::from_evals(&evaluations);
        let full: Vec<E> = match &self.eq {
            Some(eq) => eq.gruen_mul(&q).into_coefficients(),
            None => q.into_coefficients(),
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

    /// `q(X) = sum_y eq'(y) sum_i gamma_i term_i(lanes(X, y))` at `X = 0..points`.
    fn inner_evaluations(&self, points: usize) -> Vec<E> {
        if let Some((source, folded)) = &self.buckets {
            match self.level() {
                0 => return self.bucketed_evaluations(source, &self.plan.source, 0, points),
                1 => {
                    let half = self.lanes[0].len() / 2;
                    let largest = self
                        .columns
                        .iter()
                        .map(Column::quartets)
                        .max()
                        .unwrap_or(1);
                    if half >= self.bucket_min_ratio.saturating_mul(largest) {
                        return self.bucketed_evaluations(folded, &self.plan.folded, 1, points);
                    }
                }
                _ => {}
            }
        }
        self.generic_evaluations(points)
    }

    /// The per-pair evaluation of the compiled plan.
    fn generic_evaluations(&self, points: usize) -> Vec<E> {
        let half = self.lanes[0].len() / 2;
        let (eq_first, eq_second) = match &self.eq {
            Some(eq) => eq.remaining_eq_tables(),
            None => (&[][..], &[][..]),
        };
        let first_mask = eq_first.len().saturating_sub(1);
        let first_bits = eq_first.len().trailing_zeros();
        let lanes = &self.lanes;
        let plan = &self.plan;
        let phase = if self.in_source_phase() {
            &plan.source
        } else {
            &plan.folded
        };
        let lane_count = lanes.len();
        let evaluate_chunk = |start: usize, end: usize| {
            let mut acc = vec![E::zero(); points];
            let mut values = vec![E::zero(); lane_count * points];
            for k in start..end {
                // values[x * lane_count + l] = lane l extrapolated to X = x.
                for (l, lane) in lanes.iter().enumerate() {
                    let (v0, v1) = lane.pair(k);
                    let delta = v1 - v0;
                    values[l] = v0;
                    let mut v = v0;
                    for x in 1..points {
                        v += delta;
                        values[x * lane_count + l] = v;
                    }
                }
                let weight = if eq_first.is_empty() {
                    E::one()
                } else {
                    eq_first[k & first_mask] * eq_second[k >> first_bits]
                };
                for (x, slot) in acc.iter_mut().enumerate() {
                    let at_x = &values[x * lane_count..(x + 1) * lane_count];
                    let local = plan.constant + plan.evaluate(phase, at_x);
                    *slot += weight * local;
                }
            }
            acc
        };
        let add = |mut left: Vec<E>, right: Vec<E>| {
            for (slot, value) in left.iter_mut().zip(right) {
                *slot += value;
            }
            left
        };
        if half >= MIN_PARALLEL_PAIRS {
            (0..half.div_ceil(PAIR_CHUNK))
                .into_par_iter()
                .map(|chunk| {
                    let start = chunk * PAIR_CHUNK;
                    evaluate_chunk(start, (start + PAIR_CHUNK).min(half))
                })
                .reduce(|| vec![E::zero(); points], add)
        } else {
            evaluate_chunk(0, half)
        }
    }

    /// The bucketed evaluation of a compact round: equality weights (or weight lane
    /// values) summed per cell pattern of each coded column, terms evaluated once per
    /// pattern. Level 0 reads the source cells (pair patterns), level 1 the lookup lanes
    /// (quartet patterns).
    fn bucketed_evaluations(&self, buckets: &Buckets, phase: &Phase, level: usize, points: usize) -> Vec<E> {
        let half = self.lanes[0].len() / 2;
        // Level-1 buckets are large (65,536 entries per digit column), so one task per
        // thread amortizes their zeroing and reduction.
        let chunk_size = if level == 0 {
            BUCKET_CHUNK
        } else {
            BUCKET_CHUNK_L1.max(half.div_ceil(rayon::current_num_threads()))
        };
        let (eq_first, eq_second) = match &self.eq {
            Some(eq) => eq.remaining_eq_tables(),
            None => (&[][..], &[][..]),
        };
        let has_eq = self.eq.is_some();
        let first_mask = eq_first.len().saturating_sub(1);
        let first_bits = eq_first.len().trailing_zeros();
        let columns = &self.columns;
        let lanes = &self.lanes;
        let joint_terms: Vec<(&Term, &[usize])> = buckets
            .pieces
            .iter()
            .filter_map(|piece| match piece {
                Piece::Joint(_, term, cols) => Some((term, cols.as_slice())),
                _ => None,
            })
            .collect();
        let joint_sizes: Vec<usize> = joint_terms
            .iter()
            .map(|(_, cols)| cols.iter().map(|&c| columns[c].patterns_at(level)).product())
            .collect();

        // Accumulators of one chunk: total weight, per pure column the weight per
        // pattern, per key the weight lane's values per pattern and point, per joint
        // term the weight per joint pattern.
        struct Sums {
            total: E,
            pure: Vec<Vec<E>>,
            keyed: Vec<Vec<E>>,
            joint: Vec<Vec<E>>,
        }
        let zeroed = || Sums {
            total: E::zero(),
            pure: buckets
                .pure
                .iter()
                .map(|&c| vec![E::zero(); columns[c].patterns_at(level)])
                .collect(),
            keyed: buckets
                .keys
                .iter()
                .map(|&(_, c)| vec![E::zero(); columns[c].patterns_at(level) * points])
                .collect(),
            joint: joint_sizes.iter().map(|&n| vec![E::zero(); n]).collect(),
        };
        let add = |mut left: Sums, right: Sums| {
            left.total += right.total;
            for (a, b) in left.pure.iter_mut().zip(right.pure) {
                for (slot, value) in a.iter_mut().zip(b) {
                    *slot += value;
                }
            }
            for (a, b) in left.keyed.iter_mut().zip(right.keyed) {
                for (slot, value) in a.iter_mut().zip(b) {
                    *slot += value;
                }
            }
            for (a, b) in left.joint.iter_mut().zip(right.joint) {
                for (slot, value) in a.iter_mut().zip(b) {
                    *slot += value;
                }
            }
            left
        };
        // Distinct weight lanes of the keys, each with the keys reading it.
        let mut weight_lanes: Vec<(usize, Vec<usize>)> = Vec::new();
        for (key_index, &(f, _)) in buckets.keys.iter().enumerate() {
            match weight_lanes.iter_mut().find(|(lane, _)| *lane == f) {
                Some((_, keys)) => keys.push(key_index),
                None => weight_lanes.push((f, vec![key_index])),
            }
        }
        let weight_lanes = &weight_lanes;
        let accumulate = |start: usize, end: usize| {
            let mut sums = zeroed();
            let mut patterns = vec![0_usize; columns.len()];
            let mut lane_values = vec![E::zero(); points];
            for k in start..end {
                for (slot, column) in patterns.iter_mut().zip(columns) {
                    *slot = column.pattern_at(level, k);
                }
                let weight = if has_eq {
                    eq_first[k & first_mask] * eq_second[k >> first_bits]
                } else {
                    E::one()
                };
                sums.total += weight;
                for (bucket, &c) in sums.pure.iter_mut().zip(&buckets.pure) {
                    bucket[patterns[c]] += weight;
                }
                for (f, key_indices) in weight_lanes {
                    let (v0, v1) = lanes[*f].pair(k);
                    let delta = v1 - v0;
                    let mut v = v0;
                    for slot in lane_values.iter_mut() {
                        *slot = if has_eq { weight * v } else { v };
                        v += delta;
                    }
                    for &key_index in key_indices {
                        let c = buckets.keys[key_index].1;
                        let base = patterns[c] * points;
                        for (slot, value) in sums.keyed[key_index][base..base + points]
                            .iter_mut()
                            .zip(&lane_values)
                        {
                            *slot += *value;
                        }
                    }
                }
                for (bucket, (_, cols)) in sums.joint.iter_mut().zip(&joint_terms) {
                    let mut index = 0;
                    for &c in cols.iter() {
                        index = index * columns[c].patterns_at(level) + patterns[c];
                    }
                    bucket[index] += weight;
                }
            }
            sums
        };
        let sums = if half >= MIN_PARALLEL_PAIRS {
            (0..half.div_ceil(chunk_size))
                .into_par_iter()
                .map(|chunk| {
                    let start = chunk * chunk_size;
                    accumulate(start, (start + chunk_size).min(half))
                })
                .reduce(zeroed, add)
        } else {
            accumulate(0, half)
        };

        // The value of coded lane `l` at a pattern of its column as `(v(0), v(1) - v(0))`,
        // so the extrapolation to `X = x` is `x` additions: from the map at level 0, from
        // the lane's lookup table at level 1.
        let coded_pair = |l: usize, pattern: usize| -> (E, E) {
            let Coded::Column(c, map) = self.coded[l] else {
                unreachable!("bucketed lanes are coded")
            };
            if level == 0 {
                let (a, b) = columns[c].cells_of(pattern);
                let va = map[a as u8 as usize];
                let vb = map[b as u8 as usize];
                (E::from_i64(va), E::from_i64(vb - va))
            } else {
                let Lane::Lookup { lut, .. } = &lanes[l] else {
                    unreachable!("level-1 lanes are pair lookups")
                };
                let (lo, hi) = columns[c].pairs_of(pattern);
                if lo >= lut.len() || hi >= lut.len() {
                    // A padded quartet pattern: never occurs, zero weight.
                    return (E::zero(), E::zero());
                }
                let va = lut[lo];
                (va, lut[hi] - va)
            }
        };
        let coded_value = |l: usize, pattern: usize, x: usize| -> E {
            let (v0, delta) = coded_pair(l, pattern);
            (0..x).fold(v0, |v, _| v + delta)
        };
        let add_vec = |mut left: Vec<E>, right: Vec<E>| {
            for (slot, value) in left.iter_mut().zip(right) {
                *slot += value;
            }
            left
        };
        let polys = &phase.polys;
        // Evaluate every piece once per pattern, in parallel over the pure columns and
        // the keys (the pattern spaces are large at level 1).
        let pure_part = sums
            .pure
            .par_iter()
            .zip(&buckets.pure)
            .map(|(bucket, &c)| {
                let mut q = vec![E::zero(); points];
                let column_lanes: Vec<usize> = (0..lanes.len())
                    .filter(|&l| matches!(self.coded[l], Coded::Column(cc, _) if cc == c))
                    .filter(|&l| polys[l] != LanePoly::None)
                    .collect();
                let products: Vec<(E, usize, usize)> = buckets
                    .pieces
                    .iter()
                    .filter_map(|piece| match piece {
                        Piece::SameColumn(cc, coefficient, a, b) if *cc == c => {
                            Some((*coefficient, *a, *b))
                        }
                        _ => None,
                    })
                    .collect();
                let mut lane_values = vec![(E::zero(), E::zero()); column_lanes.len()];
                let mut product_values = vec![((E::zero(), E::zero()), (E::zero(), E::zero())); products.len()];
                for (pattern, &weight) in bucket.iter().enumerate() {
                    if weight.is_zero() {
                        continue;
                    }
                    for (slot, &l) in lane_values.iter_mut().zip(&column_lanes) {
                        *slot = coded_pair(l, pattern);
                    }
                    for (slot, &(_, a, b)) in product_values.iter_mut().zip(&products) {
                        *slot = (coded_pair(a, pattern), coded_pair(b, pattern));
                    }
                    for slot in q.iter_mut() {
                        let mut value = E::zero();
                        for ((v, delta), &l) in lane_values.iter_mut().zip(&column_lanes) {
                            value += polys[l].evaluate(*v);
                            *v += *delta;
                        }
                        for (((va, da), (vb, db)), &(coefficient, _, _)) in
                            product_values.iter_mut().zip(&products)
                        {
                            let product = *va * *vb;
                            value += if coefficient == E::one() { product } else { coefficient * product };
                            *va += *da;
                            *vb += *db;
                        }
                        *slot += weight * value;
                    }
                }
                q
            })
            .reduce(|| vec![E::zero(); points], add_vec);
        let keyed_part = sums
            .keyed
            .par_iter()
            .enumerate()
            .zip(&buckets.keys)
            .map(|((key_index, bucket), &(f, c))| {
                let mut q = vec![E::zero(); points];
                let pieces: Vec<(E, usize)> = buckets
                    .pieces
                    .iter()
                    .filter_map(|piece| match piece {
                        Piece::Weighted(ff, coefficient, l)
                            if *ff == f && matches!(self.coded[*l], Coded::Column(cc, _) if cc == c) =>
                        {
                            Some((*coefficient, *l))
                        }
                        _ => None,
                    })
                    .collect();
                let constants: Vec<E> = buckets
                    .pieces
                    .iter()
                    .filter_map(|piece| match piece {
                        Piece::WeightConstant(ff, coefficient) if *ff == f => Some(*coefficient),
                        _ => None,
                    })
                    .collect();
                // The constant piece of `f` is charged to its first key only.
                let first_key_of_f =
                    buckets.keys.iter().position(|&(ff, _)| ff == f) == Some(key_index);
                let constant: E = if first_key_of_f {
                    constants.iter().fold(E::zero(), |acc, &c| acc + c)
                } else {
                    E::zero()
                };
                let mut piece_values = vec![(E::zero(), E::zero()); pieces.len()];
                for pattern in 0..columns[c].patterns_at(level) {
                    let weights = &bucket[pattern * points..(pattern + 1) * points];
                    if weights.iter().all(E::is_zero) {
                        continue;
                    }
                    for (slot, &(_, l)) in piece_values.iter_mut().zip(&pieces) {
                        *slot = coded_pair(l, pattern);
                    }
                    for (slot, &weight) in q.iter_mut().zip(weights) {
                        let mut value = constant;
                        for ((v, delta), &(coefficient, _)) in piece_values.iter_mut().zip(&pieces) {
                            value += coefficient * *v;
                            *v += *delta;
                        }
                        *slot += weight * value;
                    }
                }
                q
            })
            .reduce(|| vec![E::zero(); points], add_vec);
        let mut q = add_vec(pure_part, keyed_part);
        for (slot, _) in q.iter_mut().zip(0..points) {
            *slot += self.plan.constant * sums.total;
        }
        for ((bucket, (term, cols)), piece) in sums
            .joint
            .iter()
            .zip(&joint_terms)
            .zip(buckets.pieces.iter().filter(|p| matches!(p, Piece::Joint(..))))
        {
            let Piece::Joint(gamma, _, _) = piece else { unreachable!() };
            let term_lanes = term.lanes();
            let mut values = vec![E::zero(); lanes.len()];
            for (index, &weight) in bucket.iter().enumerate() {
                if weight.is_zero() {
                    continue;
                }
                // Decode the joint pattern into per-column patterns (last column fastest).
                let mut rest = index;
                let mut per_column = vec![0_usize; cols.len()];
                for (slot, &c) in per_column.iter_mut().zip(cols.iter()).rev() {
                    *slot = rest % columns[c].patterns_at(level);
                    rest /= columns[c].patterns_at(level);
                }
                for (x, slot) in q.iter_mut().enumerate() {
                    for &l in &term_lanes {
                        let Coded::Column(c, _) = self.coded[l] else { unreachable!() };
                        let position = cols.iter().position(|&cc| cc == c).expect("column listed");
                        values[l] = coded_value(l, per_column[position], x);
                    }
                    *slot += weight * *gamma * term.evaluate(&values);
                }
            }
        }
        q
    }

    /// Bind the round's variable to `challenge`.
    pub fn bind(&mut self, round: usize, challenge: E) {
        assert_eq!(round, self.bound, "rounds in order");
        assert!(!self.pending.is_empty(), "round_message before bind");
        self.claim = evaluate_compressed(&self.pending, self.claim, challenge);
        self.pending.clear();
        if round < self.inactive {
            self.scale *= inverse_power_of_two(1);
        } else {
            let first = self.in_source_phase();
            for (l, lane) in self.lanes.iter_mut().enumerate() {
                let scale = self.plan.scales[l];
                if first {
                    if let (Coded::Column(c, map), Lane::Source(_)) = (self.coded[l], &*lane) {
                        let column = self.columns[c];
                        *lane = Lane::Lookup {
                            column,
                            lut: lookup_table(&column, map, challenge, scale),
                        };
                        continue;
                    }
                }
                lane.fold(challenge);
                if first && scale != E::one() {
                    if let Lane::Folded(cells) = lane {
                        cells.par_iter_mut().for_each(|cell| *cell *= scale);
                    }
                }
            }
            if let Some(eq) = &mut self.eq {
                eq.bind(challenge);
            }
        }
        self.bound += 1;
    }

    /// The fully bound lane values, in lane order (unscaled).
    #[must_use]
    pub fn terminal_values(&self) -> Vec<E> {
        assert_eq!(self.bound, self.inactive + self.local_vars, "all rounds bound");
        self.lanes
            .iter()
            .zip(&self.plan.scales)
            .map(|(lane, &scale)| {
                let value = lane.terminal();
                if scale == E::one() {
                    value
                } else {
                    value * scale.inverse().expect("nonzero scale")
                }
            })
            .collect()
    }
}

/// Deduplicate the byte sources of the lanes into coded columns (span at most 16).
fn code_columns<'a>(lanes: &[Lane<'a>]) -> (Vec<Column<'a>>, Vec<Coded<'a>>) {
    let mut columns: Vec<Column<'a>> = Vec::new();
    let mut coded = Vec::with_capacity(lanes.len());
    // Sources that were examined and found too wide, by (pointer, length).
    let mut wide: Vec<(*const i8, usize)> = Vec::new();
    for lane in lanes {
        let Lane::Source(Source::Bytes { cells, map }) = lane else {
            coded.push(Coded::No);
            continue;
        };
        let key = (cells.as_ptr(), cells.len());
        if let Some(index) = columns
            .iter()
            .position(|c| (c.cells.as_ptr(), c.cells.len()) == key)
        {
            coded.push(Coded::Column(index, map));
            continue;
        }
        if wide.contains(&key) {
            coded.push(Coded::No);
            continue;
        }
        let (min, max) = byte_range(cells);
        let span = (i16::from(max) - i16::from(min) + 1) as usize;
        if cells.is_empty() || span > MAX_SPAN {
            wide.push(key);
            coded.push(Coded::No);
            continue;
        }
        columns.push(Column::new(cells, min, span));
        coded.push(Coded::Column(columns.len() - 1, map));
    }
    (columns, coded)
}

/// The bucketed layout of a plan, or `None` when some piece cannot be bucketed.
fn bucket_layout(plan: &Plan, phase: &Phase, lanes: &[Lane<'_>], columns: &[Column<'_>], coded: &[Coded<'_>]) -> Option<Buckets> {
    let column_of = |l: usize| match coded[l] {
        Coded::Column(c, _) => Some(c),
        Coded::No => None,
    };
    let is_field = |l: usize| matches!(lanes[l], Lane::Source(Source::Field(_)));
    let mut pure = Vec::new();
    let mut keys: Vec<(usize, usize)> = Vec::new();
    let mut pieces = Vec::new();
    let mut add_pure = |c: usize| {
        if !pure.contains(&c) {
            pure.push(c);
        }
    };
    for (l, poly) in phase.polys.iter().enumerate() {
        if *poly != LanePoly::None {
            add_pure(column_of(l)?);
        }
    }
    let mut add_key = |f: usize, c: usize| {
        if !keys.contains(&(f, c)) {
            keys.push((f, c));
        }
    };
    for &(coefficient, a, b) in &phase.bilinear {
        match (column_of(a), column_of(b)) {
            (Some(ca), Some(cb)) if ca == cb => {
                add_pure(ca);
                pieces.push(Piece::SameColumn(ca, coefficient, a, b));
            }
            (Some(ca), None) if is_field(b) => {
                add_key(b, ca);
                pieces.push(Piece::Weighted(b, coefficient, a));
            }
            (None, Some(cb)) if is_field(a) => {
                add_key(a, cb);
                pieces.push(Piece::Weighted(a, coefficient, b));
            }
            _ => return None,
        }
    }
    for form in &plan.weighted {
        if !is_field(form.weight) || column_of(form.weight).is_some() {
            return None;
        }
        for &(l, coefficient) in &form.lanes {
            let c = column_of(l)?;
            add_key(form.weight, c);
            pieces.push(Piece::Weighted(form.weight, coefficient, l));
        }
        if !form.constant.is_zero() {
            if form.lanes.is_empty() {
                return None;
            }
            pieces.push(Piece::WeightConstant(form.weight, form.constant));
        }
    }
    for (gamma, term) in &plan.generic {
        let mut cols: Vec<usize> = Vec::new();
        for l in term.lanes() {
            let c = column_of(l)?;
            if !cols.contains(&c) {
                cols.push(c);
            }
        }
        let size: usize = cols.iter().map(|&c| columns[c].patterns()).product();
        if size > MAX_JOINT_PATTERNS {
            return None;
        }
        pieces.push(Piece::Joint(*gamma, term.clone(), cols));
    }
    Some(Buckets { pure, keys, pieces })
}

/// The first fold of a coded lane as a table over pair patterns:
/// `scale * (v(2k) + r (v(2k + 1) - v(2k)))` depends only on the pattern of pair `k`.
fn lookup_table(column: &Column<'_>, map: &[i64; 256], challenge: E, scale: E) -> Vec<E> {
    (0..column.patterns())
        .map(|pattern| {
            let (a, b) = column.cells_of(pattern);
            let va = map[a as u8 as usize];
            let vb = map[b as u8 as usize];
            (E::from_i64(va) + challenge.mul_i64(vb - va)) * scale
        })
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{field, point};
    use aerie_falcon::protocol::algebra::{eq_evaluation, eq_table, evaluate_mle};
    use agg_witness::gen::SplitMix64;

    fn digits(rng: &mut SplitMix64, len: usize) -> Vec<i8> {
        (0..len).map(|_| rng.below(16) as i8 - 8).collect()
    }

    fn bits(rng: &mut SplitMix64, len: usize) -> Vec<i8> {
        (0..len).map(|_| rng.below(2) as i8).collect()
    }

    /// The direct sum of a weighted group over the Boolean domain.
    fn direct_sum(
        lanes: &[Lane<'_>],
        instances: &[GroupInstance],
        weights: Option<&[E]>,
    ) -> E {
        let len = lanes[0].len();
        (0..len).fold(E::zero(), |acc, x| {
            let values: Vec<E> = lanes.iter().map(|lane| lane.value(x)).collect();
            let summand = instances.iter().fold(E::zero(), |s, instance| {
                s + instance.gamma
                    * instance
                        .terms
                        .iter()
                        .fold(E::zero(), |t, term| t + term.evaluate(&values))
            });
            acc + summand * weights.map_or(E::one(), |w| w[x])
        })
    }

    /// The round-0 polynomial `g(X)` of a group at the integer `X`, from the definition:
    /// `sum_k eq(tau_0, X) eq(tau_rest, k) sum_i gamma_i term_i(v(0, k) + X (v(1, k) - v(0, k)))`.
    fn reference_round0(
        lanes: &[Lane<'_>],
        instances: &[GroupInstance],
        eq_point: Option<&[E]>,
        x: u64,
    ) -> E {
        let half = lanes[0].len() / 2;
        let xf = E::from_u64(x);
        let rest = eq_point.map(|tau| eq_table(&tau[1..]));
        let mut sum = E::zero();
        for k in 0..half {
            let values: Vec<E> = lanes
                .iter()
                .map(|lane| {
                    let (v0, v1) = lane.pair(k);
                    v0 + xf * (v1 - v0)
                })
                .collect();
            let summand = instances.iter().fold(E::zero(), |s, instance| {
                s + instance.gamma
                    * instance
                        .terms
                        .iter()
                        .fold(E::zero(), |t, term| t + term.evaluate(&values))
            });
            sum += summand * rest.as_ref().map_or(E::one(), |w| w[k]);
        }
        match eq_point {
            Some(tau) => sum * ((E::one() - tau[0]) * (E::one() - xf) + tau[0] * xf),
            None => sum,
        }
    }

    /// Run a group through a stage of `rounds` rounds with random challenges and check
    /// the sum-check invariants and the terminal equation. Returns the terminal values.
    fn run(
        mut group: Group<'_>,
        instances: &[GroupInstance],
        rounds: usize,
        degree: usize,
        eq_point: Option<&[E]>,
        mut claim: E,
        rng: &mut SplitMix64,
    ) -> (Vec<E>, Vec<E>, Vec<Vec<E>>) {
        if group.inactive_rounds() > 0 {
            assert_eq!(group.claim(), claim, "direct claim of a lifted group");
        }
        let mut challenges = Vec::new();
        let mut messages = Vec::new();
        for round in 0..rounds {
            let message = group.round_message(round);
            assert_eq!(message.len(), degree);
            assert_eq!(group.claim(), claim, "round {round}");
            let r = field(rng);
            claim = evaluate_compressed(&message, claim, r);
            group.bind(round, r);
            assert_eq!(group.claim(), claim);
            challenges.push(r);
            messages.push(message);
        }
        let values = group.terminal_values();
        let local = &challenges[rounds - group.local_vars..];
        let eq_factor = eq_point.map_or(E::one(), |p| eq_evaluation(p, local));
        let expected = expected_terminal(instances, &values, eq_factor, group.inactive_rounds());
        assert_eq!(claim, expected);
        (values, challenges, messages)
    }

    #[test]
    fn eq_group_matches_the_direct_sum_and_terminal_equation() {
        let mut rng = SplitMix64::new(1);
        let vars = 6;
        let cells: Vec<Vec<i8>> = (0..3).map(|_| digits(&mut rng, 1 << vars)).collect();
        let square = digit_map(|z| z * (z + 1));
        let lanes = vec![
            Lane::column(&cells[0]),
            Lane::column(&cells[1]),
            Lane::mapped(&cells[2], &square),
        ];
        let instances = vec![
            GroupInstance {
                gamma: field(&mut rng),
                terms: vec![Term::Product(vec![0, 1])],
            },
            GroupInstance {
                gamma: field(&mut rng),
                terms: vec![Term::Leaf(2, LeafPoly::Q8)],
            },
            GroupInstance {
                gamma: field(&mut rng),
                terms: vec![Term::Bool(0), Term::Link(1)],
            },
        ];
        let tau = point(&mut rng, vars);
        let weights = eq_table(&tau);
        let claim = direct_sum(&lanes, &instances, Some(&weights));
        for rounds in [vars, vars + 3] {
            let group =
                Group::new(lanes.clone(), instances.clone(), Some(&tau), rounds, 5).unwrap();
            assert_eq!(group.inactive_rounds(), rounds - vars);
            // Two coded columns read by a product: not bucketable, generic path.
            assert!(!group.is_bucketed());
            run(group, &instances, rounds, 5, Some(&tau), claim, &mut rng);
        }
    }

    #[test]
    fn weight_group_matches_the_direct_sum() {
        let mut rng = SplitMix64::new(2);
        let vars = 7;
        let weight = point(&mut rng, 1 << vars);
        let ints: Vec<i32> = (0..1 << vars).map(|_| rng.below(5) as i32 - 2).collect();
        let bits: Vec<i8> = (0..1 << vars).map(|_| rng.below(2) as i8).collect();
        let lanes = vec![
            Lane::Source(Source::Field(&weight)),
            Lane::Source(Source::Ints(&ints)),
            Lane::column(&bits),
        ];
        let instances = vec![
            GroupInstance {
                gamma: field(&mut rng),
                terms: vec![Term::Product(vec![0, 1])],
            },
            GroupInstance {
                gamma: field(&mut rng),
                terms: vec![Term::Affine {
                    weight: Some(0),
                    lanes: vec![(1, 3), (2, -7)],
                    constant: 11,
                }],
            },
        ];
        let claim = direct_sum(&lanes, &instances, None);
        let group = Group::new(lanes, instances.clone(), None, vars + 2, 2).unwrap();
        assert!(!group.is_bucketed(), "an Ints lane is not coded");
        run(group, &instances, vars + 2, 2, None, claim, &mut rng);
    }

    #[test]
    fn compressed_messages_round_trip_and_rejections() {
        let mut rng = SplitMix64::new(3);
        let full = point(&mut rng, 5);
        let claim = full.iter().fold(E::zero(), |s, &c| s + c) + full[0];
        let message = compress(&full, 4);
        assert_eq!(message.len(), 4);
        let x = field(&mut rng);
        let direct = full.iter().rev().fold(E::zero(), |acc, &c| acc * x + c);
        assert_eq!(evaluate_compressed(&message, claim, x), direct);
        assert_eq!(
            inactive_message(claim, 3),
            vec![claim * inverse_power_of_two(1), E::zero(), E::zero()]
        );
        assert_eq!(inverse_power_of_two(3) * E::from_u64(8), E::one());
        assert!(Group::new(vec![], vec![], None, 3, 2).is_err());
        let cells = vec![0_i8; 8];
        let lanes = vec![Lane::column(&cells)];
        let too_high = vec![GroupInstance {
            gamma: E::one(),
            terms: vec![Term::Bool(0)],
        }];
        assert!(Group::new(lanes.clone(), too_high.clone(), None, 3, 1).is_err());
        assert!(Group::new(lanes.clone(), too_high, None, 2, 3).is_err());
        let missing = vec![GroupInstance {
            gamma: E::one(),
            terms: vec![Term::Product(vec![0, 1])],
        }];
        assert!(Group::new(lanes, missing, None, 3, 3).is_err());
    }

    #[test]
    fn leaf_polynomials_vanish_exactly_on_the_alphabets() {
        for z in -128_i64..=127 {
            let p = E::from_i64(z * (z + 1));
            let in4 = (-2..=1).contains(&z);
            let in8 = (-4..=3).contains(&z);
            let in16 = (-8..=7).contains(&z);
            assert_eq!(LeafPoly::Q4.evaluate(p) == E::zero(), in4);
            assert_eq!(LeafPoly::Q8.evaluate(p) == E::zero(), in8);
            let q16 = LeafPoly::Q8.evaluate(p) * LeafPoly::Q16Hi.evaluate(p);
            assert_eq!(q16 == E::zero(), in16);
        }
        assert_eq!(LeafPoly::of_base(16), [LeafPoly::Q8, LeafPoly::Q16Hi]);
        assert_eq!(Term::Leaf(0, LeafPoly::Q16Hi).degree(), 4);
        assert_eq!(IDENTITY_MAP[0xFF], -1);
        assert_eq!(IDENTITY_MAP[7], 7);
    }

    #[test]
    fn leaf_coefficients_expand_the_root_products() {
        assert_eq!(LeafPoly::Q4.coefficients(), vec![0, -2, 1]);
        assert_eq!(LeafPoly::Q8.coefficients(), vec![0, -144, 108, -20, 1]);
        for poly in [LeafPoly::Q4, LeafPoly::Q8, LeafPoly::Q16Hi] {
            let coefficients = poly.coefficients();
            assert_eq!(coefficients.len(), poly.degree() + 1);
            for p in [-3_i64, 0, 2, 7, 56, 1000] {
                let pf = E::from_i64(p);
                let horner = coefficients
                    .iter()
                    .rev()
                    .fold(E::zero(), |acc, &c| acc * pf + E::from_i64(c));
                assert_eq!(horner, poly.evaluate(pf));
            }
        }
        let mut lane = LanePoly::None;
        lane.add(&[E::zero(), E::from_u64(3)]);
        assert_eq!(lane.linear_coefficient(), Some(E::from_u64(3)));
        lane.add(&[E::zero(), -E::from_u64(3)]);
        assert_eq!(lane, LanePoly::None);
        lane.add(&[E::one(), E::zero(), E::from_u64(2)]);
        assert_eq!(lane.evaluate(E::from_u64(5)), E::from_u64(51));
    }

    /// An equality group with every bucketable term kind: Booleans and the bit-range
    /// identity on four bit columns, leaves on two maps of one digit column, a link, a
    /// same-column product (`A^lo A^hi`), reductions and a weightless affine form.
    fn eq_fixture(rng: &mut SplitMix64, vars: usize) -> (Vec<Vec<i8>>, [i64; 256], [i64; 256], [i64; 256]) {
        let mut cells: Vec<Vec<i8>> = (0..4).map(|_| bits(rng, 1 << vars)).collect();
        cells.push(digits(rng, 1 << vars));
        cells.push(digits(rng, 1 << vars));
        // Values in [-2, 2]: a span of 5, whose pattern spaces are not powers of two.
        cells.push((0..1 << vars).map(|_| rng.below(5) as i8 - 2).collect());
        let square = digit_map(|z| z * (z + 1));
        let lo = digit_map(|z| LeafPoly::Q8.roots().iter().fold(1, |acc, &r| acc * (z * (z + 1) - r)));
        let hi = digit_map(|z| LeafPoly::Q16Hi.roots().iter().fold(1, |acc, &r| acc * (z * (z + 1) - r)));
        (cells, square, lo, hi)
    }

    fn eq_lanes<'a>(cells: &'a [Vec<i8>], square: &'a [i64; 256], lo: &'a [i64; 256], hi: &'a [i64; 256]) -> Vec<Lane<'a>> {
        vec![
            Lane::column(&cells[0]),      // 0: b_2
            Lane::column(&cells[1]),      // 1: b_3
            Lane::column(&cells[2]),      // 2: b_4
            Lane::column(&cells[3]),      // 3: b_5
            Lane::mapped(&cells[4], square), // 4: P of digit A
            Lane::mapped(&cells[4], lo),  // 5: A^lo of digit A
            Lane::mapped(&cells[4], hi),  // 6: A^hi of digit A
            Lane::column(&cells[4]),      // 7: digit A
            Lane::column(&cells[5]),      // 8: digit B
            Lane::mapped(&cells[6], square), // 9: P of a base-4 digit
        ]
    }

    fn eq_instances(rng: &mut SplitMix64) -> Vec<GroupInstance> {
        let mut instance = |terms: Vec<Term>| GroupInstance {
            gamma: field(rng),
            terms,
        };
        vec![
            instance(vec![Term::Bool(0)]),
            instance(vec![Term::Bool(1)]),
            instance(vec![Term::Bool(2), Term::Bool(3)]),
            instance(vec![Term::RangeW1([3, 2, 1, 0])]),
            instance(vec![Term::Leaf(4, LeafPoly::Q8)]),
            instance(vec![Term::Leaf(4, LeafPoly::Q16Hi)]),
            instance(vec![Term::Product(vec![5, 6])]),
            instance(vec![Term::Link(7)]),
            instance(vec![Term::Link(8)]),
            instance(vec![Term::Product(vec![8])]),
            instance(vec![Term::Leaf(9, LeafPoly::Q4)]),
            instance(vec![Term::Affine {
                weight: None,
                lanes: vec![(0, -190464), (7, 16), (8, 4096)],
                constant: 34952,
            }]),
            instance(vec![Term::Product(vec![1])]),
        ]
    }

    #[test]
    fn bucketed_first_round_matches_the_definition_and_the_generic_path() {
        let mut rng = SplitMix64::new(11);
        let vars = 9;
        let (cells, square, lo, hi) = eq_fixture(&mut rng, vars);
        let lanes = eq_lanes(&cells, &square, &lo, &hi);
        let instances = eq_instances(&mut rng);
        let degree = 5;
        for rounds in [vars, vars + 2] {
            let tau = point(&mut rng, vars);
            let claim = direct_sum(&lanes, &instances, Some(&eq_table(&tau)));
            let bucketed = Group::new(lanes.clone(), instances.clone(), Some(&tau), rounds, degree).unwrap();
            assert!(bucketed.is_bucketed());
            let span5 = bucketed.columns.iter().find(|c| c.span == 5).expect("a span-5 column");
            assert_eq!((span5.patterns(), span5.pair_bits, span5.quartets()), (25, 5, 1024));
            let generic = Group::new(lanes.clone(), instances.clone(), Some(&tau), rounds, degree).unwrap().without_buckets();
            assert!(!generic.is_bucketed());
            // Round 1 bucketed by quartet patterns whatever the pair count.
            let quartets = Group::new(lanes.clone(), instances.clone(), Some(&tau), rounds, degree).unwrap().with_bucket_ratio(0);
            // All paths send the same messages for the same challenges.
            let mut rng_a = SplitMix64::new(99);
            let mut rng_b = SplitMix64::new(99);
            let mut rng_c = SplitMix64::new(99);
            let (values_a, challenges, messages_a) = run(bucketed, &instances, rounds, degree, Some(&tau), claim, &mut rng_a);
            let (values_b, _, messages_b) = run(generic, &instances, rounds, degree, Some(&tau), claim, &mut rng_b);
            let (values_c, _, messages_c) = run(quartets, &instances, rounds, degree, Some(&tau), claim, &mut rng_c);
            assert_eq!(messages_a, messages_b);
            assert_eq!(values_a, values_b);
            assert_eq!(messages_a, messages_c);
            assert_eq!(values_a, values_c);
            // The first active message is the polynomial of the definition.
            let first = rounds - vars;
            let message = &messages_a[first];
            let lift = inverse_power_of_two(first);
            let claim_at_first = challenges[..first].iter().fold(claim, |c, &r| {
                evaluate_compressed(&inactive_message(c, degree), c, r)
            });
            for x in 0..=degree as u64 {
                let expected = reference_round0(&lanes, &instances, Some(&tau), x) * lift;
                assert_eq!(evaluate_compressed(message, claim_at_first, E::from_u64(x)), expected, "X = {x}");
            }
            // Terminal values are the extensions of the lanes (scaled lanes unscaled).
            let local = &challenges[first..];
            for (lane, value) in lanes.iter().zip(&values_a) {
                let table: Vec<E> = (0..1 << vars).map(|i| lane.value(i)).collect();
                assert_eq!(*value, evaluate_mle(&table, local));
            }
        }
    }

    #[test]
    fn weight_group_buckets_by_the_coded_columns() {
        let mut rng = SplitMix64::new(12);
        let vars = 8;
        let weight_a = point(&mut rng, 1 << vars);
        let weight_b = point(&mut rng, 1 << vars);
        let bit_cells: Vec<Vec<i8>> = (0..2).map(|_| bits(&mut rng, 1 << vars)).collect();
        let digit_cells: Vec<Vec<i8>> = (0..3).map(|_| digits(&mut rng, 1 << vars)).collect();
        let lanes = vec![
            Lane::Source(Source::Field(&weight_a)),
            Lane::Source(Source::Field(&weight_b)),
            Lane::column(&bit_cells[0]),
            Lane::column(&bit_cells[1]),
            Lane::column(&digit_cells[0]),
            Lane::column(&digit_cells[1]),
            Lane::column(&digit_cells[2]),
        ];
        let mut instance = |terms: Vec<Term>| GroupInstance {
            gamma: field(&mut rng),
            terms,
        };
        let instances = vec![
            instance(vec![Term::Product(vec![0, 2])]),
            instance(vec![Term::Product(vec![0, 3])]),
            instance(vec![Term::Product(vec![3, 1])]),
            instance(vec![Term::Affine {
                weight: Some(0),
                lanes: vec![(4, 1), (5, 16), (6, 256)],
                constant: -2056,
            }]),
            instance(vec![Term::Bool(2)]),
        ];
        let degree = 2;
        for rounds in [vars, vars + 1] {
            let claim = direct_sum(&lanes, &instances, None);
            let bucketed = Group::new(lanes.clone(), instances.clone(), None, rounds, degree).unwrap();
            assert!(bucketed.is_bucketed());
            let generic = Group::new(lanes.clone(), instances.clone(), None, rounds, degree).unwrap().without_buckets();
            let quartets = Group::new(lanes.clone(), instances.clone(), None, rounds, degree).unwrap().with_bucket_ratio(0);
            let mut rng_a = SplitMix64::new(7);
            let mut rng_b = SplitMix64::new(7);
            let mut rng_c = SplitMix64::new(7);
            let (values_a, challenges, messages_a) = run(bucketed, &instances, rounds, degree, None, claim, &mut rng_a);
            let (values_b, _, messages_b) = run(generic, &instances, rounds, degree, None, claim, &mut rng_b);
            let (values_c, _, messages_c) = run(quartets, &instances, rounds, degree, None, claim, &mut rng_c);
            assert_eq!(messages_a, messages_b);
            assert_eq!(values_a, values_b);
            assert_eq!(messages_a, messages_c);
            assert_eq!(values_a, values_c);
            let first = rounds - vars;
            let claim_at_first = challenges[..first].iter().fold(claim, |c, &r| {
                evaluate_compressed(&inactive_message(c, degree), c, r)
            });
            for x in 0..=degree as u64 {
                let expected = reference_round0(&lanes, &instances, None, x) * inverse_power_of_two(first);
                assert_eq!(evaluate_compressed(&messages_a[first], claim_at_first, E::from_u64(x)), expected);
            }
        }
    }

    #[test]
    fn wide_cells_and_mixed_lanes_fall_back_to_the_generic_path() {
        let mut rng = SplitMix64::new(13);
        let vars = 6;
        let mut wide = digits(&mut rng, 1 << vars);
        wide[5] = 9; // a tampered digit outside [-8, 7] widens the span to 18
        let narrow = digits(&mut rng, 1 << vars);
        let ints: Vec<i32> = (0..1 << vars).map(|_| rng.below(2000) as i32).collect();
        let square = digit_map(|z| z * (z + 1));
        let tau = point(&mut rng, vars);
        // Wide column: its lanes are not coded, the group is generic, results still hold.
        let lanes = vec![Lane::mapped(&wide, &square), Lane::column(&narrow)];
        let instances = vec![
            GroupInstance { gamma: field(&mut rng), terms: vec![Term::Leaf(0, LeafPoly::Q8)] },
            GroupInstance { gamma: field(&mut rng), terms: vec![Term::Link(1)] },
        ];
        let claim = direct_sum(&lanes, &instances, Some(&eq_table(&tau)));
        let group = Group::new(lanes.clone(), instances.clone(), Some(&tau), vars, 5).unwrap();
        assert!(!group.is_bucketed());
        assert!(matches!(group.coded[0], Coded::No));
        assert!(matches!(group.coded[1], Coded::Column(0, _)));
        run(group, &instances, vars, 5, Some(&tau), claim, &mut rng);
        // An Ints lane read by a Boolean term: generic as well.
        let lanes = vec![Lane::Source(Source::Ints(&ints)), Lane::column(&narrow)];
        let instances = vec![
            GroupInstance { gamma: field(&mut rng), terms: vec![Term::Bool(0)] },
            GroupInstance { gamma: field(&mut rng), terms: vec![Term::Link(1)] },
        ];
        let claim = direct_sum(&lanes, &instances, Some(&eq_table(&tau)));
        let group = Group::new(lanes, instances.clone(), Some(&tau), vars + 1, 3).unwrap();
        assert!(!group.is_bucketed());
        run(group, &instances, vars + 1, 3, Some(&tau), claim, &mut rng);
        // A product of two different coded columns is not bucketed either.
        let other = digits(&mut rng, 1 << vars);
        let lanes = vec![Lane::column(&narrow), Lane::column(&other)];
        let instances = vec![GroupInstance { gamma: field(&mut rng), terms: vec![Term::Product(vec![0, 1])] }];
        let group = Group::new(lanes, instances, Some(&tau), vars, 3).unwrap();
        assert!(!group.is_bucketed());
    }

    #[test]
    fn scaled_lanes_fold_by_lookup_and_report_unscaled_terminals() {
        let mut rng = SplitMix64::new(14);
        let vars = 7;
        let cells: Vec<Vec<i8>> = (0..3).map(|_| digits(&mut rng, 1 << vars)).collect();
        let lo = digit_map(|z| LeafPoly::Q8.roots().iter().fold(1, |acc, &r| acc * (z * (z + 1) - r)));
        let hi = digit_map(|z| LeafPoly::Q16Hi.roots().iter().fold(1, |acc, &r| acc * (z * (z + 1) - r)));
        let lanes = vec![
            Lane::column(&cells[0]),
            Lane::column(&cells[1]),
            Lane::mapped(&cells[2], &lo),
            Lane::mapped(&cells[2], &hi),
        ];
        let instances = vec![
            GroupInstance { gamma: field(&mut rng), terms: vec![Term::Product(vec![0])] },
            GroupInstance { gamma: field(&mut rng), terms: vec![Term::Product(vec![1])] },
            GroupInstance { gamma: field(&mut rng), terms: vec![Term::Product(vec![2, 3])] },
        ];
        let tau = point(&mut rng, vars);
        let group = Group::new(lanes.clone(), instances.clone(), Some(&tau), vars, 3).unwrap();
        // The two reductions and the first product factor are scaled at the fold.
        assert_eq!(group.plan.scales[0], instances[0].gamma);
        assert_eq!(group.plan.scales[1], instances[1].gamma);
        assert_eq!(group.plan.scales[2], instances[2].gamma);
        assert_eq!(group.plan.scales[3], E::one());
        assert_eq!(group.plan.folded.polys[0], LanePoly::Identity);
        assert_eq!(group.plan.folded.bilinear, vec![(E::one(), 2, 3)]);
        assert_eq!(group.plan.source.bilinear, vec![(instances[2].gamma, 2, 3)]);
        let claim = direct_sum(&lanes, &instances, Some(&eq_table(&tau)));
        let (values, challenges, _) = run(group, &instances, vars, 3, Some(&tau), claim, &mut rng);
        for (lane, value) in lanes.iter().zip(&values) {
            let table: Vec<E> = (0..1 << vars).map(|i| lane.value(i)).collect();
            assert_eq!(*value, evaluate_mle(&table, &challenges));
        }
        // After one fold a coded lane is a lookup lane with the folded table's values.
        let mut group = Group::new(lanes.clone(), instances.clone(), Some(&tau), vars, 3).unwrap();
        group.round_message(0);
        let r = field(&mut rng);
        group.bind(0, r);
        let Lane::Lookup { lut, column } = &group.lanes[3] else {
            panic!("a coded lane folds to a pair-pattern lookup lane")
        };
        assert_eq!(lut.len(), 256);
        assert_eq!(group.lanes[3].len(), 1 << (vars - 1));
        let mut once = Vec::new();
        for k in 0..1 << (vars - 1) {
            let (v0, v1) = lanes[3].pair(k);
            assert_eq!(lut[column.pattern(k)], v0 + r * (v1 - v0));
            assert_eq!(group.lanes[3].value(k), v0 + r * (v1 - v0));
            once.push(v0 + r * (v1 - v0));
        }
        assert_eq!(group.level(), 1);
        group.round_message(1);
        let r2 = field(&mut rng);
        group.bind(1, r2);
        let Lane::Folded(cells) = &group.lanes[3] else {
            panic!("the second fold materializes the lane")
        };
        assert_eq!(cells.len(), 1 << (vars - 2));
        for k in 0..1 << (vars - 2) {
            assert_eq!(cells[k], once[2 * k] + r2 * (once[2 * k + 1] - once[2 * k]));
        }
        assert_eq!(group.level(), 2);
    }
}
