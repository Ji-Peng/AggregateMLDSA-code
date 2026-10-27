//! Lookup-based range argument (prototype, `docs/lookup.md`): `LogUp` (Haboeck, ePrint
//! 2022/1530, Lemma 5) with the GKR fractional sum-check of Papini and Haboeck (ePrint
//! 2023/1284, Sections 3-4) over `F_p`, challenges in `K = F_{p^2}`.
//!
//! A *stack* is a list of committed digit columns of one domain that are looked up in one
//! table `T_w = [0, 2^w)`. Its columns are stacked into one lane of `2^depth` cells
//! (column index in the top variables, padded with zero columns to a power of two; zero
//! lies in every table) and every cell `g` becomes the leaf fraction `1 / (X - g)`. The
//! table side has one leaf `m(t) / (X - t)` per entry, `m` the multiplicity column. For
//! every table and every challenge `X`, the verifier checks
//!
//! `sum_{stacks into T} p_root / q_root = p_T / q_T`, every `q_root != 0`,
//!
//! where `(p_root, q_root)` is the projective root of a binary fraction tree: node `y` of
//! layer `k` has children `y` and `y + 2^k` in layer `k + 1` (the child bit is the top
//! variable, so the two halves of a layer are the two child lanes), with
//! `p_k = p_0 q_1 + p_1 q_0` and `q_k = q_0 q_1`. The trees are top-aligned and share every
//! challenge: session 1 sends layer 1 of every tree; session `s >= 2` is one batched
//! degree-3 sum-check of `s - 1` rounds reducing the layer-`(s - 1)` claims at the common
//! point `r` to layer `s` at `(rho, mu_s)`. A tree whose depth is `s` reads its leaves in
//! session `s`; its final claim is one evaluation of the stacked lane (or of `m`) at a
//! point of `K^depth`, an Akita opening claim in the integrated protocol (here checked by
//! the caller). One rational identity is proven per table: tables are never merged by
//! integer offsets, because a committed value is an arbitrary `F_p` element and an offset
//! tag does not stop a column from matching another table's entries (`docs/lookup.md`
//! Section 1).
//!
//! Every sum-check runs on [`crate::kernel::Group`] through its public constructors only:
//! layer lanes are `Source::Field` halves of the stored layers, leaf lanes are
//! `Source::Ints` halves of the stacked column (or of `m` and the public index), and the
//! `X`-dependence of the leaf terms is folded into the instance coefficients, so one
//! bottom group serves every challenge `X` at the cost of one.
//!
//! Fiat-Shamir runs through [`LookupTranscript`]: the integrated protocol drives the PIOP
//! transcript (`transcript.rs`, the lookup events of its registry, [`prove_in`] and
//! [`verify_in`]); the standalone [`prove`] and [`verify`] use a local SHAKE256 sponge
//! (`Fs`) bound to the layout and a caller binding.
//!
//! A stack may store its cells shifted: the looked-up value is `f = stored + shift` (the
//! L16 columns are stored balanced, `shift = 2^15`). The leaves are then
//! `1 / ((X - shift) - stored)`, the multiplicities count `f`, and the final claim is on the
//! stored lane (so it is a claim on the committed columns as they are).

use std::borrow::Cow;
use std::time::Instant;

use aerie_falcon::protocol::algebra::eq_evaluation;
use jolt_field::{CanonicalEncoding, ExtField, Field, One, Ring, Zero};
use rayon::prelude::*;
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::Shake256;

use crate::kernel::{
    evaluate_compressed, expected_terminal, fm, fmi, Group, GroupInstance, IntPairGroup, Lane, LayerGroup, LayerTree, Source,
    Term,
};
use crate::transcript::encode_extensions;
use crate::{BaseField, OpeningField as E};

/// Message degree of every session (equality factor times a bilinear summand).
pub const DEGREE: usize = 3;
/// Largest number of independent challenges `X` (the prototype measures 1 and 2).
pub const MAX_CHALLENGES: usize = 4;
/// Balanced base-16 digits of one multiplicity (`m < 7 (16^7 - 1) / 15 ~ 2^26.9`).
pub const MULTIPLICITY_DIGITS: usize = 7;
/// Below this many nodes a tree layer is built serially.
const PARALLEL_NODES: usize = 1 << 12;
/// Domain separator of the prototype transcript.
const DOMAIN: &[u8] = b"agg-mldsa/v1/lookup-prototype";

/// A lookup failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LookupError {
    /// The layout or a proof message has the wrong shape.
    Shape(String),
    /// An honest prover found a looked-up value outside its table.
    OutOfTable {
        /// The stack.
        stack: usize,
        /// The cell of the stacked lane.
        index: usize,
        /// The value.
        value: i32,
    },
    /// The root identity of a table failed or a root denominator is zero.
    Identity {
        /// The challenge index.
        challenge: usize,
        /// The table.
        table: usize,
    },
    /// The terminal equation of a session failed (verifier) or the prover's own claimed
    /// sum disagrees with its tables (prover self-check).
    Terminal(usize),
    /// A kernel error.
    Kernel(String),
    /// The transcript rejected an event.
    Transcript(String),
}

impl std::fmt::Display for LookupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shape(what) => write!(f, "malformed lookup: {what}"),
            Self::OutOfTable { stack, index, value } => {
                write!(f, "stack {stack} cell {index} holds {value}, outside its table")
            }
            Self::Identity { challenge, table } => {
                write!(f, "root identity of table {table} fails at challenge {challenge}")
            }
            Self::Terminal(session) => write!(f, "session {session} terminal check failed"),
            Self::Kernel(what) => write!(f, "kernel: {what}"),
            Self::Transcript(what) => write!(f, "transcript: {what}"),
        }
    }
}

impl std::error::Error for LookupError {}

// ---------------------------------------------------------------------------------------
// Tables, stacks and multiplicities.
// ---------------------------------------------------------------------------------------

/// The range table `T_w = [0, 2^w)`, entry `t` at index `t`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Table {
    /// `w`, between 2 and 30.
    pub width: u32,
}

impl Table {
    /// The 16-bit table of the L16 layout.
    pub const T16: Self = Self { width: 16 };

    /// `T_width`.
    #[must_use]
    pub const fn new(width: u32) -> Self {
        Self { width }
    }

    /// Number of entries.
    #[must_use]
    pub const fn len(self) -> usize {
        1 << self.width
    }

    /// Never empty (the width is at least 2).
    #[must_use]
    pub const fn is_empty(self) -> bool {
        false
    }

    /// True when `value` is an entry.
    #[must_use]
    pub fn contains(self, value: i32) -> bool {
        value >= 0 && i64::from(value) < 1_i64 << self.width
    }

    /// The two halves `t_0(y) = y`, `t_1(y) = y + 2^(w-1)` of the public index lane.
    ///
    /// # Panics
    /// If the width exceeds 30.
    #[must_use]
    pub fn index_halves(self) -> (Vec<i32>, Vec<i32>) {
        let half = i32::try_from(self.len() / 2).expect("width at most 30");
        ((0..half).collect(), (half..2 * half).collect())
    }

    /// The multilinear extension of the index table `t(x) = x` at `point`:
    /// `sum_k 2^k point_k`.
    #[must_use]
    pub fn index_evaluation(point: &[E]) -> E {
        point
            .iter()
            .enumerate()
            .fold(E::zero(), |acc, (k, &coordinate)| acc + coordinate.mul_u64(1 << k))
    }
}

/// The public shape of one stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StackShape {
    /// Real (committed) columns.
    pub columns: usize,
    /// Variables of every column.
    pub vars: usize,
    /// The table the stack is looked up in.
    pub table: usize,
    /// The storage shift: the looked-up value of a stored cell `s` is `s + shift`.
    pub shift: i32,
}

impl StackShape {
    /// Column slots: the column count rounded up to a power of two.
    #[must_use]
    pub const fn slots(self) -> usize {
        self.columns.next_power_of_two()
    }

    /// Variables of the stacked lane (the tree depth).
    #[must_use]
    pub const fn depth(self) -> usize {
        self.vars + self.slots().trailing_zeros() as usize
    }

    /// Leaves (looked-up cells, padding included).
    #[must_use]
    pub const fn leaves(self) -> usize {
        1 << self.depth()
    }

    /// The stored value of a padding slot: `-shift`, so that its looked-up value is 0.
    #[must_use]
    pub const fn padding(self) -> i32 {
        -self.shift
    }

    /// The weights `eq(point_top, c)` of the real columns in a stacked evaluation:
    /// `g(point) = sum_c weight_c f_c(point_low) + padding sum_{c >= columns} eq(point_top, c)`
    /// (zero padding contributes nothing when the shift is zero).
    #[must_use]
    pub fn column_weights(self, point: &[E]) -> Vec<E> {
        self.slot_weights(point)[..self.columns].to_vec()
    }

    /// The column claim of a stacked claim `(point, value)` once the padding slots are
    /// removed: `sum_{c < columns} weight_c f_c(point_low) = value - padding sum_{c >=
    /// columns} eq(point_top, c)`.
    #[must_use]
    pub fn column_claim(self, point: &[E], value: E) -> E {
        let padding: E = self.slot_weights(point)[self.columns..]
            .iter()
            .fold(E::zero(), |acc, &w| acc + w);
        value - padding.mul_i64(i64::from(self.padding()))
    }

    /// `eq(point_top, c)` for every slot `c`.
    fn slot_weights(self, point: &[E]) -> Vec<E> {
        let top = &point[self.vars..];
        (0..self.slots())
            .map(|c| {
                top.iter().enumerate().fold(E::one(), |acc, (bit, &coordinate)| {
                    if (c >> bit) & 1 == 1 {
                        acc * coordinate
                    } else {
                        acc * (E::one() - coordinate)
                    }
                })
            })
            .collect()
    }
}

/// Committed columns of one domain looked up in one table, stacked.
#[derive(Clone, Debug)]
pub struct LookupStack<'a> {
    shape: StackShape,
    stacked: Cow<'a, [i32]>,
}

impl<'a> LookupStack<'a> {
    /// Stack `columns` (each of `2^vars` looked-up values) for lookups in table `table`;
    /// missing column slots are zero.
    ///
    /// # Errors
    /// On an empty stack or columns of different or non-power-of-two lengths.
    pub fn new(columns: &[&[i32]], table: usize) -> Result<LookupStack<'static>, LookupError> {
        LookupStack::with_shift(columns, table, 0)
    }

    /// Stack stored `columns` whose looked-up values are `stored + shift`; missing column
    /// slots hold `-shift` (looked-up value 0).
    ///
    /// # Errors
    /// On an empty stack or columns of different or non-power-of-two lengths.
    pub fn with_shift(
        columns: &[&[i32]],
        table: usize,
        shift: i32,
    ) -> Result<LookupStack<'static>, LookupError> {
        let first = columns
            .first()
            .ok_or_else(|| LookupError::Shape("a stack needs a column".into()))?;
        let cells = first.len();
        if !cells.is_power_of_two() || columns.iter().any(|c| c.len() != cells) {
            return Err(LookupError::Shape("stack columns of unequal or odd length".into()));
        }
        let shape = StackShape {
            columns: columns.len(),
            vars: cells.trailing_zeros() as usize,
            table,
            shift,
        };
        let mut stacked = vec![shape.padding(); shape.leaves()];
        stacked
            .par_chunks_mut(cells)
            .zip(columns.par_iter())
            .for_each(|(slot, column)| slot.copy_from_slice(column));
        Ok(LookupStack {
            shape,
            stacked: Cow::Owned(stacked),
        })
    }

    /// A stack over an already stacked lane (column `c` at cells `c 2^vars ..`, the
    /// padding slots holding `shape.padding()`), borrowed.
    ///
    /// # Errors
    /// When the lane length is not `shape.leaves()`.
    pub fn borrowed(shape: StackShape, stacked: &'a [i32]) -> Result<Self, LookupError> {
        if stacked.len() != shape.leaves() {
            return Err(LookupError::Shape("stacked lane of the wrong length".into()));
        }
        Ok(Self {
            shape,
            stacked: Cow::Borrowed(stacked),
        })
    }

    /// The public shape.
    #[must_use]
    pub const fn shape(&self) -> StackShape {
        self.shape
    }

    /// The stacked lane (column `c` at cells `c 2^vars ..`).
    #[must_use]
    pub fn stacked(&self) -> &[i32] {
        &self.stacked
    }

    /// Mutable stacked lane (tests: tampering).
    #[doc(hidden)]
    pub fn stacked_mut(&mut self) -> &mut [i32] {
        self.stacked.to_mut()
    }
}

/// The multiplicity column of every table: `m_T(t)` = number of looked-up cells (padding
/// included) equal to `t` over all stacks into `T`.
///
/// # Errors
/// [`LookupError::OutOfTable`] when a cell is not an entry of its table (honest abort),
/// [`LookupError::Shape`] on a missing table or a count above `i32::MAX`.
pub fn multiplicities(stacks: &[LookupStack<'_>], tables: &[Table]) -> Result<Vec<Vec<i32>>, LookupError> {
    const CHUNK: usize = 1 << 16;
    let mut totals: Vec<Vec<u64>> = tables.iter().map(|t| vec![0; t.len()]).collect();
    for (s, stack) in stacks.iter().enumerate() {
        let table = *tables
            .get(stack.shape.table)
            .ok_or_else(|| LookupError::Shape(format!("stack {s} names a missing table")))?;
        let len = table.len();
        let shift = stack.shape.shift;
        let counts = stack
            .stacked
            .par_chunks(CHUNK)
            .enumerate()
            .try_fold(
                || vec![0_u64; len],
                |mut acc, (chunk, cells)| {
                    for (offset, &stored) in cells.iter().enumerate() {
                        let value = stored.wrapping_add(shift);
                        if !table.contains(value) || stored.checked_add(shift).is_none() {
                            return Err(LookupError::OutOfTable {
                                stack: s,
                                index: chunk * CHUNK + offset,
                                value,
                            });
                        }
                        acc[value.unsigned_abs() as usize] += 1;
                    }
                    Ok(acc)
                },
            )
            .try_reduce(
                || vec![0_u64; len],
                |mut left, right| {
                    for (slot, value) in left.iter_mut().zip(right) {
                        *slot += value;
                    }
                    Ok(left)
                },
            )?;
        for (slot, value) in totals[stack.shape.table].iter_mut().zip(counts) {
            *slot += value;
        }
    }
    totals
        .into_iter()
        .map(|column| {
            column
                .into_iter()
                .map(|m| i32::try_from(m).map_err(|_| LookupError::Shape("multiplicity above i32".into())))
                .collect()
        })
        .collect()
}

/// The balanced base-16 digits `m = sum_k 16^k d_k`, `d_k in [-8, 7]`, of a multiplicity
/// column: the seven width-4 Akita polynomials of group `G_m`. No digit check is needed
/// (Lemma 5 of `LogUp` holds for any `m in F_p`); the digits only let the honest values fit
/// the width-4 preset.
///
/// # Panics
/// If a multiplicity is negative or above `7 (16^7 - 1) / 15`.
#[must_use]
pub fn multiplicity_digits(m: &[i32]) -> Vec<Vec<i8>> {
    let mut digits = vec![vec![0_i8; m.len()]; MULTIPLICITY_DIGITS];
    for (index, &value) in m.iter().enumerate() {
        let mut rest = i64::from(value);
        assert!(rest >= 0, "a multiplicity is a count");
        for column in &mut digits {
            let digit = (rest + 8).rem_euclid(16) - 8;
            column[index] = i8::try_from(digit).expect("balanced digit");
            rest = (rest - digit) / 16;
        }
        assert_eq!(rest, 0, "multiplicity {value} exceeds seven balanced base-16 digits");
    }
    digits
}

/// `sum_k 16^k d_k` (the inverse of [`multiplicity_digits`]).
#[must_use]
pub fn compose_multiplicity(digits: &[Vec<i8>]) -> Vec<i64> {
    let len = digits.first().map_or(0, Vec::len);
    (0..len)
        .map(|index| {
            digits
                .iter()
                .rev()
                .fold(0_i64, |acc, column| 16 * acc + i64::from(column[index]))
        })
        .collect()
}

// ---------------------------------------------------------------------------------------
// The public layout.
// ---------------------------------------------------------------------------------------

/// What a tree sums.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TreeKind {
    /// Leaves `1 / (X - g)` of stack `i`.
    Stack(usize),
    /// Leaves `m(t) / (X - t)` of table `i`.
    Table(usize),
}

/// The public layout of a lookup: stacks, tables and the number of challenges `X`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupShape {
    /// The stacks.
    pub stacks: Vec<StackShape>,
    /// The tables.
    pub tables: Vec<Table>,
    /// Independent challenges `X` (one full tree set each).
    pub challenges: usize,
}

impl LookupShape {
    /// The shape of `stacks` and `tables` under `challenges` challenges.
    #[must_use]
    pub fn of(stacks: &[LookupStack<'_>], tables: &[Table], challenges: usize) -> Self {
        Self {
            stacks: stacks.iter().map(LookupStack::shape).collect(),
            tables: tables.to_vec(),
            challenges,
        }
    }

    /// Check the layout.
    ///
    /// # Errors
    /// On zero or too many challenges, a missing table, a table width outside `2..=30`
    /// or a tree of depth below 2.
    pub fn validate(&self) -> Result<(), LookupError> {
        if self.challenges == 0 || self.challenges > MAX_CHALLENGES {
            return Err(LookupError::Shape(format!("{} challenges", self.challenges)));
        }
        if self.stacks.is_empty() {
            return Err(LookupError::Shape("no stack".into()));
        }
        for table in &self.tables {
            if !(2..=30).contains(&table.width) {
                return Err(LookupError::Shape(format!("table width {}", table.width)));
            }
        }
        for (s, stack) in self.stacks.iter().enumerate() {
            if stack.table >= self.tables.len() {
                return Err(LookupError::Shape(format!("stack {s} names a missing table")));
            }
            if stack.columns == 0 || stack.depth() < 2 {
                return Err(LookupError::Shape(format!("stack {s} has depth below 2")));
            }
        }
        Ok(())
    }

    fn per_challenge(&self) -> usize {
        self.stacks.len() + self.tables.len()
    }

    /// Trees: one per stack and table, per challenge.
    #[must_use]
    pub fn tree_count(&self) -> usize {
        self.challenges * self.per_challenge()
    }

    /// The event counts of session `s >= 2`: trees of depth `>= s` (two batching
    /// coefficients each), trees of depth `> s` (four terminal values each) and the stacks
    /// and tables of depth `s` (two leaf values each).
    #[must_use]
    pub fn session_counts(&self, s: usize) -> (usize, usize, usize) {
        let ids = 0..self.tree_count();
        let active = ids.clone().filter(|&id| self.tree_depth(id) >= s).count();
        let continuing = ids.filter(|&id| self.tree_depth(id) > s).count();
        let ending = self.stacks.iter().filter(|st| st.depth() == s).count()
            + self.tables.iter().filter(|t| t.width as usize == s).count();
        (active, continuing, ending)
    }

    /// Tree `id`: its challenge index and kind (challenge-major, stacks then tables).
    fn tree(&self, id: usize) -> (usize, TreeKind) {
        let (x, local) = (id / self.per_challenge(), id % self.per_challenge());
        if local < self.stacks.len() {
            (x, TreeKind::Stack(local))
        } else {
            (x, TreeKind::Table(local - self.stacks.len()))
        }
    }

    fn depth(&self, kind: TreeKind) -> usize {
        match kind {
            TreeKind::Stack(s) => self.stacks[s].depth(),
            TreeKind::Table(t) => self.tables[t].width as usize,
        }
    }

    fn tree_depth(&self, id: usize) -> usize {
        self.depth(self.tree(id).1)
    }

    /// The deepest tree, i.e. the last session.
    #[must_use]
    pub fn max_depth(&self) -> usize {
        (0..self.tree_count()).map(|id| self.tree_depth(id)).max().unwrap_or(0)
    }

    /// Total looked-up cells per table (padding included).
    #[must_use]
    pub fn lookups(&self, table: usize) -> usize {
        self.stacks
            .iter()
            .filter(|s| s.table == table)
            .map(|s| s.leaves())
            .sum()
    }

    /// The bad-set probability of one challenge `X` for table `table` before grinding:
    /// `(L + |T| - 1) / (|K| - |T|)` (LogUp-GKR Proposition 2), as `log2`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn identity_error_log2(&self, table: usize) -> f64 {
        let numerator = (self.lookups(table) + self.tables[table].len() - 1) as f64;
        numerator.log2() - 2.0 * (2.0_f64.powi(64) - 59.0).log2()
    }

    /// Proof elements of `K` (every element is 16 bytes).
    #[must_use]
    pub fn proof_elements(&self) -> usize {
        let mut count = 4 * self.tree_count();
        for s in 2..=self.max_depth() {
            count += DEGREE * (s - 1);
            count += 4 * (0..self.tree_count()).filter(|&id| self.tree_depth(id) > s).count();
            count += 2 * self.stacks.iter().filter(|st| st.depth() == s).count();
            count += 2 * self.tables.iter().filter(|t| t.width as usize == s).count();
        }
        count
    }

    fn binding_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(self.challenges as u64).to_le_bytes());
        out.extend_from_slice(&(self.stacks.len() as u64).to_le_bytes());
        for stack in &self.stacks {
            for value in [stack.columns, stack.vars, stack.table] {
                out.extend_from_slice(&(value as u64).to_le_bytes());
            }
            out.extend_from_slice(&i64::from(stack.shift).to_le_bytes());
        }
        out.extend_from_slice(&(self.tables.len() as u64).to_le_bytes());
        for table in &self.tables {
            out.extend_from_slice(&u64::from(table.width).to_le_bytes());
        }
        out
    }
}

// ---------------------------------------------------------------------------------------
// Fiat-Shamir.
// ---------------------------------------------------------------------------------------

/// The events of a lookup in transcript order (`docs/piop.md` Section 9).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LookupSite {
    /// The challenges `X_1 .. X_c` (one vector challenge).
    X,
    /// The layer-1 values of every tree.
    Layer1,
    /// `mu_s` after session `s` (`mu_1` after the layer-1 values).
    Mu(usize),
    /// The batching coefficients of session `s`.
    Gamma(usize),
    /// Round `round` of session `s`: the message.
    Round(usize, usize),
    /// Round `round` of session `s`: the challenge.
    RoundChallenge(usize, usize),
    /// The layer terminals of session `s`.
    Terminals(usize),
    /// The leaf values of session `s`.
    Leaves(usize),
}

impl LookupSite {
    /// The label of the site (also the PIOP transcript site under `agg-mldsa/v1/lookup/`).
    #[must_use]
    pub fn label(self) -> String {
        match self {
            Self::X => "x".to_owned(),
            Self::Layer1 => "layer1".to_owned(),
            Self::Mu(s) => format!("mu/{s}"),
            Self::Gamma(s) => format!("gamma/{s}"),
            Self::Round(s, round) => format!("round/{s}/{round}"),
            Self::RoundChallenge(s, round) => format!("r/{s}/{round}"),
            Self::Terminals(s) => format!("terminals/{s}"),
            Self::Leaves(s) => format!("leaves/{s}"),
        }
    }
}

/// The Fiat-Shamir interface of a lookup: absorb proof values, sample challenges in `K`.
pub trait LookupTranscript {
    /// Absorb `values` at `site`.
    ///
    /// # Errors
    /// When the transcript refuses the event.
    fn absorb(&mut self, site: LookupSite, values: &[E]) -> Result<(), LookupError>;

    /// Sample `count` challenges at `site`.
    ///
    /// # Errors
    /// When the transcript refuses the event.
    fn challenges(&mut self, site: LookupSite, count: usize) -> Result<Vec<E>, LookupError>;
}

/// The standalone transcript: SHAKE256 over framed, labelled absorptions; a challenge
/// squeezes from a clone of the state and absorbs what it squeezed.
#[derive(Clone)]
struct Fs {
    state: Shake256,
}

impl Fs {
    fn new(shape: &LookupShape, binding: &[u8]) -> Self {
        let mut fs = Self {
            state: Shake256::default(),
        };
        fs.frame(b"domain", DOMAIN);
        fs.frame(b"shape", &shape.binding_bytes());
        fs.frame(b"binding", binding);
        fs
    }

    fn frame(&mut self, label: &[u8], bytes: &[u8]) {
        self.state.update(&(label.len() as u64).to_le_bytes());
        self.state.update(label);
        self.state.update(&(bytes.len() as u64).to_le_bytes());
        self.state.update(bytes);
    }
}

impl LookupTranscript for Fs {
    fn absorb(&mut self, site: LookupSite, values: &[E]) -> Result<(), LookupError> {
        self.frame(site.label().as_bytes(), &encode_extensions(values));
        Ok(())
    }

    /// `count` elements of `K`, each limb rejection-sampled from eight bytes as in
    /// `Transcript::sample_base`.
    fn challenges(&mut self, site: LookupSite, count: usize) -> Result<Vec<E>, LookupError> {
        self.frame(site.label().as_bytes(), &(count as u64).to_le_bytes());
        let mut reader = self.state.clone().finalize_xof();
        let mut base = || loop {
            let mut candidate = [0_u8; 8];
            reader.read(&mut candidate);
            if let Some(value) = BaseField::from_bytes_le_checked(&candidate) {
                return value;
            }
        };
        let out: Vec<E> = (0..count)
            .map(|_| {
                let c0 = base();
                let c1 = base();
                E::from_base_slice(&[c0, c1])
            })
            .collect();
        self.frame(b"squeezed", &encode_extensions(&out));
        Ok(out)
    }
}

fn one(transcript: &mut impl LookupTranscript, site: LookupSite) -> Result<E, LookupError> {
    Ok(transcript.challenges(site, 1)?[0])
}

// ---------------------------------------------------------------------------------------
// Fraction trees.
// ---------------------------------------------------------------------------------------

/// One layer of projective fractions.
#[derive(Clone, Debug, Default)]
struct Layer {
    p: Vec<E>,
    q: Vec<E>,
}

impl Layer {
    fn len(&self) -> usize {
        self.p.len()
    }

    fn bytes(&self) -> usize {
        32 * self.len()
    }
}

fn build_nodes(half: usize, node: impl Fn(usize) -> (E, E) + Sync + Send) -> Layer {
    if half < PARALLEL_NODES {
        let (p, q) = (0..half).map(node).unzip();
        return Layer { p, q };
    }
    // A parallel first touch, then blocks of both lanes filled in parallel (a parallel
    // `unzip` was measured slower).
    const BLOCK: usize = 1 << 10;
    let mut p: Vec<E> = rayon::iter::repeat_n(E::zero(), half).collect();
    let mut q: Vec<E> = rayon::iter::repeat_n(E::zero(), half).collect();
    p.par_chunks_mut(BLOCK)
        .zip(q.par_chunks_mut(BLOCK))
        .enumerate()
        .for_each(|(b, (p, q))| {
            for (i, (p, q)) in p.iter_mut().zip(q.iter_mut()).enumerate() {
                (*p, *q) = node(b * BLOCK + i);
            }
        });
    Layer { p, q }
}

/// The parent layer: `p = p_0 q_1 + p_1 q_0`, `q = q_0 q_1` (3 `K`-multiplications per node).
fn parent(child: &Layer) -> Layer {
    let half = child.len() / 2;
    let (p0, p1) = child.p.split_at(half);
    let (q0, q1) = child.q.split_at(half);
    build_nodes(half, |y| (fm(p0[y], q1[y]) + fm(p1[y], q0[y]), fm(q0[y], q1[y])))
}

/// The layer above the leaves `(1, X - g)` of a stacked lane: `p = (X - g_0) + (X - g_1)`,
/// `q = (X - g_0)(X - g_1)` (`x` is the challenge minus the stack's storage shift).
fn stack_bottom(stacked: &[i32], x: E) -> Layer {
    let half = stacked.len() / 2;
    build_nodes(half, |y| {
        let a = x - E::from_i64(i64::from(stacked[y]));
        let b = x - E::from_i64(i64::from(stacked[y + half]));
        (a + b, fm(a, b))
    })
}

/// The layer above the leaves `(m(t), X - t)` of a table: `p = m_0 (X - t_1) + m_1 (X - t_0)`.
fn table_bottom(m: &[i32], x: E) -> Layer {
    let half = m.len() / 2;
    build_nodes(half, |y| {
        let a = x - E::from_u64(y as u64);
        let b = x - E::from_u64((y + half) as u64);
        (b.mul_i64(i64::from(m[y])) + a.mul_i64(i64::from(m[y + half])), a * b)
    })
}

/// The largest spread `max - min` of a stack's stored values for which it runs the integer
/// path of session `depth - 1` ([`SymLanes`]): recentered to `|v - c| <= 2^15`, the lanes
/// `|a + b| <= 2^16` and `|a b| <= 2^30` fit `i32` and `e_4 <= 2^60` fits `i64`.
const SYM_SPREAD: i64 = 1 << 16;

/// Session `depth - 1` of a stack from its leaf integers (prover only).
///
/// Node `w` of layer `depth - 1` has the leaves `a = v(w) - c`, `b = v(w + 2^(depth-1)) - c`
/// (recentered by the stack's centre `c`), so with `X' = X - shift - c`, `p = 2 X' - S` and
/// `q = X'^2 - X' S + M` for `S = a + b`, `M = a b`. The session's lanes `P_c, Q_c` (`c`
/// the top bit of the node) are affine in the integer lanes `S_c, M_c`, which are shared by
/// every challenge, so one group of four `Ints` lanes replaces the four field lanes per
/// challenge, and the layer `depth - 1` is never stored (the trees start at layer
/// `depth - 2`, built from four leaves per node).
struct SymLanes {
    /// The centre `c` subtracted from every stored value.
    center: i32,
    /// `S_0, S_1, M_0, M_1`, each of `2^(depth - 2)` cells.
    lanes: [Vec<i32>; 4],
}

impl SymLanes {
    /// `None` when the stack is shallower than 3 or its values spread over more than
    /// [`SYM_SPREAD`] (only a dishonest witness; it then takes the stored-layer path).
    fn of(stacked: &[i32]) -> Option<Self> {
        let depth = stacked.len().trailing_zeros() as usize;
        if depth < 3 {
            return None;
        }
        let (min, max) = stacked
            .par_iter()
            .fold(|| (i32::MAX, i32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)))
            .reduce(|| (i32::MAX, i32::MIN), |a, b| (a.0.min(b.0), a.1.max(b.1)));
        if i64::from(max) - i64::from(min) > SYM_SPREAD {
            return None;
        }
        let center = i32::try_from((i64::from(min) + i64::from(max)) / 2).expect("between two i32");
        let quarter = stacked.len() / 4;
        let lane = |c: usize, product: bool| -> Vec<i32> {
            let cell = |y: usize| {
                let a = stacked[y + c * quarter] - center;
                let b = stacked[y + c * quarter + 2 * quarter] - center;
                if product { a * b } else { a + b }
            };
            if quarter >= PARALLEL_NODES {
                (0..quarter).into_par_iter().map(cell).collect()
            } else {
                (0..quarter).map(cell).collect()
            }
        };
        Some(Self {
            center,
            lanes: [lane(0, false), lane(1, false), lane(0, true), lane(1, true)],
        })
    }

    /// The coefficients `[c, c_s, c_m, c_k, c_mm]` of the group of four lanes (`S_0 = 0,
    /// S_1 = 1, M_0 = 2, M_1 = 3`, `kernel::IntPairGroup`) for
    /// the trees `(X', gamma_p, gamma_q)` of the stack: `sum_X gamma_p (P_0 Q_1 + P_1 Q_0) +
    /// gamma_q Q_0 Q_1` with `P_0 Q_1 + P_1 Q_0 = 4X^3 - 3X^2 (S_0 + S_1) + 2X (M_0 + M_1) +
    /// 2X S_0 S_1 - (S_0 M_1 + S_1 M_0)` and `Q_0 Q_1 = X^4 - X^3 (S_0 + S_1) + X^2 (M_0 +
    /// M_1) + X^2 S_0 S_1 - X (S_0 M_1 + S_1 M_0) + M_0 M_1`.
    fn coefficients(trees: &[(E, E, E)]) -> [E; 5] {
        let (mut constant, mut sum_s, mut sum_m, mut cross, mut mm) =
            (E::zero(), E::zero(), E::zero(), E::zero(), E::zero());
        for &(x, gp, gq) in trees {
            let x2 = x * x;
            let x3 = x2 * x;
            constant += (gp + gp + gp + gp) * x3 + gq * x3 * x;
            sum_s -= (gp + gp + gp) * x2 + gq * x3;
            sum_m += (gp + gp) * x + gq * x2;
            cross -= gp + gq * x;
            mm += gq;
        }
        [constant, sum_s, sum_m, cross, mm]
    }

    /// [`Self::coefficients`] as kernel instances (the generic fallback).
    fn instances([constant, sum_s, sum_m, cross, mm]: [E; 5]) -> Vec<GroupInstance> {
        vec![
            GroupInstance {
                gamma: constant,
                terms: vec![affine(&[], 1)],
            },
            GroupInstance {
                gamma: sum_s,
                terms: vec![affine(&[0, 1], 0)],
            },
            // The coefficient of M_0 + M_1 and of S_0 S_1 is the same.
            GroupInstance {
                gamma: sum_m,
                terms: vec![affine(&[2, 3], 0), Term::Product(vec![0, 1])],
            },
            GroupInstance {
                gamma: cross,
                terms: vec![Term::Product(vec![0, 3]), Term::Product(vec![1, 2])],
            },
            GroupInstance {
                gamma: mm,
                terms: vec![Term::Product(vec![2, 3])],
            },
        ]
    }

    /// `P_0, P_1, Q_0, Q_1` at a point from `S_0, S_1, M_0, M_1` there.
    fn terminal(values: &[E], x: E) -> [E; 4] {
        let (s0, s1, m0, m1) = (values[0], values[1], values[2], values[3]);
        let two_x = x + x;
        [two_x - s0, two_x - s1, x * (x - s0) + m0, x * (x - s1) + m1]
    }
}

/// Layer `depth - 2` of a stack for every challenge in one pass over the leaves: node `z`
/// has the leaves `v(z + j 2^(depth-2)) - c`, `j = 0..4` (`c` the centre of [`SymLanes`]);
/// with their elementary symmetric polynomials `e_1 .. e_4` (exact in `i64` within
/// [`SYM_SPREAD`]), `q = X'^4 - e_1 X'^3 + e_2 X'^2 - e_3 X' + e_4` and `p = dq/dX' =
/// 4 X'^3 - 3 e_1 X'^2 + 2 e_2 X' - e_3` (`X' = X - shift - c`), the same values as two
/// levels of [`stack_bottom`] and [`parent`].
fn stack_upper_bottoms(stacked: &[i32], center: i32, xs: &[E]) -> Vec<Layer> {
    let quarter = stacked.len() / 4;
    let powers: Vec<[E; 4]> = xs
        .iter()
        .map(|&x| {
            let x2 = x * x;
            let x3 = x2 * x;
            [x, x2, x3, x3 * x]
        })
        .collect();
    let node = |z: usize, out: &mut [(E, E)]| {
        let v = |j: usize| i64::from(stacked[z + j * quarter]) - i64::from(center);
        // Children A = (v0, v2), B = (v1, v3) of layer depth - 1.
        let (sa, ma) = (v(0) + v(2), v(0) * v(2));
        let (sb, mb) = (v(1) + v(3), v(1) * v(3));
        let e1 = sa + sb;
        let e2 = ma + mb + sa * sb;
        let e3 = sa * mb + sb * ma;
        let e4 = ma * mb;
        for (slot, &[x, x2, x3, x4]) in out.iter_mut().zip(&powers) {
            let q = x4 - fmi(x3, e1) + fmi(x2, e2) - fmi(x, e3) + E::from_i64(e4);
            let p = (x3 + x3 + x3 + x3) - fmi(x2, 3 * e1) + fmi(x, 2 * e2) - E::from_i64(e3);
            *slot = (p, q);
        }
    };
    let challenges = xs.len();
    let zeros = || -> Vec<E> {
        if quarter >= PARALLEL_NODES {
            rayon::iter::repeat_n(E::zero(), quarter).collect()
        } else {
            vec![E::zero(); quarter]
        }
    };
    let mut layers: Vec<Layer> = (0..challenges).map(|_| Layer { p: zeros(), q: zeros() }).collect();
    // Blocks of nodes: every output lane split at the same boundaries.
    const BLOCK: usize = 1 << 10;
    let mut blocks: Vec<Vec<&mut [E]>> = Vec::new();
    for layer in &mut layers {
        for lane in [&mut layer.p, &mut layer.q] {
            for (b, chunk) in lane.chunks_mut(BLOCK).enumerate() {
                if blocks.len() <= b {
                    blocks.push(Vec::with_capacity(2 * challenges));
                }
                blocks[b].push(chunk);
            }
        }
    }
    let fill = |(b, mut lanes): (usize, Vec<&mut [E]>)| {
        let mut values = vec![(E::zero(), E::zero()); challenges];
        let count = lanes[0].len();
        for i in 0..count {
            node(b * BLOCK + i, &mut values);
            for (pair, &(p, q)) in lanes.as_chunks_mut::<2>().0.iter_mut().zip(&values) {
                pair[0][i] = p;
                pair[1][i] = q;
            }
        }
    };
    if quarter >= PARALLEL_NODES {
        blocks.into_par_iter().enumerate().for_each(fill);
    } else {
        blocks.into_iter().enumerate().for_each(fill);
    }
    layers
}

/// The stored layers `0 .. depth - 1` of one tree (the leaves are read from the columns).
#[derive(Debug, Default)]
struct FractionTree {
    layers: Vec<Layer>,
}

impl FractionTree {
    fn from_bottom(bottom: Layer) -> Self {
        let depth = bottom.len().trailing_zeros() as usize + 1;
        let mut layers = vec![Layer::default(); depth];
        layers[depth - 1] = bottom;
        for k in (0..depth - 1).rev() {
            layers[k] = parent(&layers[k + 1]);
        }
        Self { layers }
    }

    /// A tree of `depth` layers whose stored bottom is `bottom` (layer `log2 |bottom|`,
    /// above the leaves by more than one layer): the layers below it are not stored.
    fn from_upper(bottom: Layer, depth: usize) -> Self {
        let top = bottom.len().trailing_zeros() as usize;
        let mut layers = vec![Layer::default(); depth];
        layers[top] = bottom;
        for k in (0..top).rev() {
            layers[k] = parent(&layers[k + 1]);
        }
        Self { layers }
    }

    fn root(&self) -> (E, E) {
        (self.layers[0].p[0], self.layers[0].q[0])
    }

    fn layer1(&self) -> [E; 4] {
        let layer = &self.layers[1];
        [layer.p[0], layer.p[1], layer.q[0], layer.q[1]]
    }

    fn bytes(&self) -> usize {
        self.layers.iter().map(Layer::bytes).sum()
    }
}

/// `(p, q)` of the root from the layer-1 values `[p(0), p(1), q(0), q(1)]`.
fn root_of(layer1: &[E; 4]) -> (E, E) {
    let [p0, p1, q0, q1] = *layer1;
    (p0 * q1 + p1 * q0, q0 * q1)
}

/// The root identity of every table and challenge.
fn check_roots(shape: &LookupShape, layer1: &[[E; 4]]) -> Result<(), LookupError> {
    let roots: Vec<(E, E)> = layer1.iter().map(root_of).collect();
    for x in 0..shape.challenges {
        for table in 0..shape.tables.len() {
            let fail = LookupError::Identity { challenge: x, table };
            let mut lhs = E::zero();
            let mut rhs = E::zero();
            for (id, &(p, q)) in roots.iter().enumerate() {
                let (tree_x, kind) = shape.tree(id);
                if tree_x != x {
                    continue;
                }
                let inverse = q.inverse().ok_or_else(|| fail.clone())?;
                match kind {
                    TreeKind::Stack(s) if shape.stacks[s].table == table => lhs += p * inverse,
                    TreeKind::Table(t) if t == table => rhs = p * inverse,
                    _ => {}
                }
            }
            if lhs != rhs {
                return Err(fail);
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------
// Sessions: the instance table shared by prover and verifier.
// ---------------------------------------------------------------------------------------

/// One group of a session.
#[derive(Clone, Debug)]
enum GroupKind {
    /// Lanes `P_0, P_1, Q_0, Q_1` of every listed tree (stored layer `s`).
    Layer(Vec<usize>),
    /// Lanes `v_0, v_1` (halves of the stacked lane) of a stack whose depth is `s`.
    Stack(usize),
    /// Lanes `m_0, m_1, t_0, t_1` of a table whose width is `s`.
    Table(usize),
}

/// The groups of session `s` and their gamma-weighted instances.
struct SessionPlan {
    groups: Vec<(GroupKind, Vec<GroupInstance>)>,
    /// Trees with depth `> s`, in id order (the layer group's lane order).
    continuing: Vec<usize>,
    ending_stacks: Vec<usize>,
    ending_tables: Vec<usize>,
}

fn affine(lanes: &[usize], constant: i64) -> Term {
    Term::Affine {
        weight: None,
        lanes: lanes.iter().map(|&l| (l, 1)).collect(),
        constant,
    }
}

/// `active` lists the trees of depth `>= s` in id order; `gammas[2 i], gammas[2 i + 1]`
/// batch the `p`- and `q`-claims of `active[i]`.
#[allow(clippy::too_many_lines)]
fn session_plan(shape: &LookupShape, s: usize, xs: &[E], active: &[usize], gammas: &[E]) -> SessionPlan {
    let gamma_of = |id: usize| {
        let i = active.binary_search(&id).expect("an active tree");
        (gammas[2 * i], gammas[2 * i + 1])
    };
    let continuing: Vec<usize> = active.iter().copied().filter(|&id| shape.tree_depth(id) > s).collect();
    let ending_stacks: Vec<usize> = (0..shape.stacks.len()).filter(|&j| shape.stacks[j].depth() == s).collect();
    let ending_tables: Vec<usize> = (0..shape.tables.len())
        .filter(|&t| shape.tables[t].width as usize == s)
        .collect();
    let mut groups = Vec::new();
    if !continuing.is_empty() {
        let mut instances = Vec::with_capacity(2 * continuing.len());
        for (i, &id) in continuing.iter().enumerate() {
            let (gp, gq) = gamma_of(id);
            let (p0, p1, q0, q1) = (4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3);
            instances.push(GroupInstance {
                gamma: gp,
                terms: vec![Term::Product(vec![p0, q1]), Term::Product(vec![p1, q0])],
            });
            instances.push(GroupInstance {
                gamma: gq,
                terms: vec![Term::Product(vec![q0, q1])],
            });
        }
        groups.push((GroupKind::Layer(continuing.clone()), instances));
    }
    let per_x = |kind: TreeKind| -> Vec<(E, E, E)> {
        (0..shape.challenges)
            .map(|x| {
                let (local, shift) = match kind {
                    TreeKind::Stack(j) => (j, shape.stacks[j].shift),
                    TreeKind::Table(t) => (shape.stacks.len() + t, 0),
                };
                let (gp, gq) = gamma_of(x * shape.per_challenge() + local);
                (xs[x] - E::from_i64(i64::from(shift)), gp, gq)
            })
            .collect()
    };
    for &j in &ending_stacks {
        // With X' = X - shift on the stored lanes v:
        // q = (X' - v0)(X' - v1) = X'^2 - X' (v0 + v1) + v0 v1; p = 2 X' - v0 - v1.
        let (mut constant, mut linear, mut bilinear) = (E::zero(), E::zero(), E::zero());
        for (x, gp, gq) in per_x(TreeKind::Stack(j)) {
            constant += gq * x * x + (gp + gp) * x;
            linear -= gq * x + gp;
            bilinear += gq;
        }
        let instances = vec![
            GroupInstance {
                gamma: constant,
                terms: vec![affine(&[], 1)],
            },
            GroupInstance {
                gamma: linear,
                terms: vec![affine(&[0, 1], 0)],
            },
            GroupInstance {
                gamma: bilinear,
                terms: vec![Term::Product(vec![0, 1])],
            },
        ];
        groups.push((GroupKind::Stack(j), instances));
    }
    for &t in &ending_tables {
        // Lanes m0 = 0, m1 = 1, t0 = 2, t1 = 3.
        // q = X^2 - X (t0 + t1) + t0 t1; p = X (m0 + m1) - m0 t1 - m1 t0.
        let (mut constant, mut index, mut index2, mut mult, mut cross) =
            (E::zero(), E::zero(), E::zero(), E::zero(), E::zero());
        for (x, gp, gq) in per_x(TreeKind::Table(t)) {
            constant += gq * x * x;
            index -= gq * x;
            index2 += gq;
            mult += gp * x;
            cross -= gp;
        }
        let instances = vec![
            GroupInstance {
                gamma: constant,
                terms: vec![affine(&[], 1)],
            },
            GroupInstance {
                gamma: index,
                terms: vec![affine(&[2, 3], 0)],
            },
            GroupInstance {
                gamma: index2,
                terms: vec![Term::Product(vec![2, 3])],
            },
            GroupInstance {
                gamma: mult,
                terms: vec![affine(&[0, 1], 0)],
            },
            GroupInstance {
                gamma: cross,
                terms: vec![Term::Product(vec![0, 3]), Term::Product(vec![1, 2])],
            },
        ];
        groups.push((GroupKind::Table(t), instances));
    }
    SessionPlan {
        groups,
        continuing,
        ending_stacks,
        ending_tables,
    }
}

/// The layer instances of the `i`-th tree of a stored-layer group (lanes `4 i ..`).
fn layer_instances(i: usize, (gp, gq): (E, E)) -> [GroupInstance; 2] {
    let (p0, p1, q0, q1) = (4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3);
    [
        GroupInstance {
            gamma: gp,
            terms: vec![Term::Product(vec![p0, q1]), Term::Product(vec![p1, q0])],
        },
        GroupInstance {
            gamma: gq,
            terms: vec![Term::Product(vec![q0, q1])],
        },
    ]
}

/// What the prover's group of a session yields at its terminal (prover only: the
/// verifier's groups are those of [`session_plan`]; the round messages are sums over
/// groups, so any split of the instances gives the same messages).
enum Role {
    /// `P_0, P_1, Q_0, Q_1` of these trees, in order.
    Trees(Vec<usize>),
    /// `S_0, S_1, M_0, M_1` of a stack shared by these trees `(id, X')`.
    Sym(Vec<(usize, E)>),
    /// The two leaf-lane values of an ending stack or table.
    Leaves,
}

/// A prover group: the generic kernel or a dedicated path (`kernel::layer`).
#[allow(clippy::large_enum_variant)]
enum SessionGroup<'a> {
    Generic(Group<'a>),
    Layer(LayerGroup<'a>),
    Pair(IntPairGroup<'a>),
}

impl SessionGroup<'_> {
    fn round_message(&mut self, round: usize) -> Vec<E> {
        match self {
            Self::Generic(g) => g.round_message(round),
            Self::Layer(g) => g.round_message(round),
            Self::Pair(g) => g.round_message(round),
        }
    }

    fn bind(&mut self, round: usize, challenge: E) {
        match self {
            Self::Generic(g) => g.bind(round, challenge),
            Self::Layer(g) => g.bind(round, challenge),
            Self::Pair(g) => g.bind(round, challenge),
        }
    }

    fn claim(&self) -> E {
        match self {
            Self::Generic(g) => g.claim(),
            Self::Layer(g) => g.claim(),
            Self::Pair(g) => g.claim(),
        }
    }

    fn consistent(&self) -> bool {
        match self {
            Self::Generic(_) => true,
            Self::Layer(g) => g.terminal_consistent(),
            Self::Pair(g) => g.terminal_consistent(),
        }
    }

    fn terminal_values(&self) -> Vec<E> {
        match self {
            Self::Generic(g) => g.terminal_values(),
            Self::Layer(g) => g.terminal_values(),
            Self::Pair(g) => g.terminal_values(),
        }
    }
}

fn line(at_zero: E, at_one: E, mu: E) -> E {
    at_zero + mu * (at_one - at_zero)
}

// ---------------------------------------------------------------------------------------
// Proof, claims and the prover.
// ---------------------------------------------------------------------------------------

/// One session `s >= 2`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionProof {
    /// `s - 1` compressed degree-3 messages (3 elements each).
    pub rounds: Vec<Vec<E>>,
    /// `P_0, P_1, Q_0, Q_1` at `rho` per tree of depth `> s`, in tree order.
    pub terminals: Vec<[E; 4]>,
    /// The two leaf-lane values at `rho` per stack of depth `s` (`v_0, v_1`), then per
    /// table of width `s` (`m_0, m_1`); shared by every challenge `X`.
    pub leaves: Vec<[E; 2]>,
}

/// The lookup proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupProof {
    /// `p_1(0), p_1(1), q_1(0), q_1(1)` per tree (challenge-major, stacks then tables).
    pub layer1: Vec<[E; 4]>,
    /// Sessions `2 ..= max_depth`.
    pub sessions: Vec<SessionProof>,
}

impl LookupProof {
    /// Elements of `K`.
    #[must_use]
    pub fn elements(&self) -> usize {
        4 * self.layer1.len()
            + self
                .sessions
                .iter()
                .map(|s| s.rounds.iter().map(Vec::len).sum::<usize>() + 4 * s.terminals.len() + 2 * s.leaves.len())
                .sum::<usize>()
    }

    /// Bytes at 16 per element.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        16 * self.elements()
    }
}

/// The evaluation claims the lookup leaves on committed columns: one per stack (its
/// stacked lane at a point of `K^depth`) and one per table (its multiplicity column at a
/// point of `K^width`). [`StackShape::column_weights`] splits a stack claim into column
/// claims at the point's low coordinates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupClaims {
    /// `(point, g(point))` per stack.
    pub stacks: Vec<(Vec<E>, E)>,
    /// `(point, m(point))` per table.
    pub multiplicities: Vec<(Vec<E>, E)>,
}

/// Prover wall times (seconds) and memory.
#[derive(Clone, Debug, Default)]
pub struct ProveStats {
    /// Building every fraction tree.
    pub trees: f64,
    /// Per session `2 ..= max_depth`.
    pub sessions: Vec<f64>,
    /// Everything (trees, sessions, transcript).
    pub total: f64,
    /// Bytes of the stored layers right after the build.
    pub layer_bytes: usize,
}

/// A proof, its claims and the prover's statistics.
#[derive(Clone, Debug)]
pub struct LookupOutput {
    /// The proof.
    pub proof: LookupProof,
    /// The claims on the committed columns.
    pub claims: LookupClaims,
    /// Timings.
    pub stats: ProveStats,
}

/// Prove that every cell of every stack lies in its table.
///
/// `multiplicities[t]` is the multiplicity column of table `t` ([`multiplicities`]);
/// `binding` stands for the commitments of the looked-up columns and of the
/// multiplicities, which must precede the challenges `X`.
///
/// # Errors
/// On a bad layout, a kernel error, or when the prover's own tables fail the root
/// identity or a session self-check (a dishonest witness).
pub fn prove(
    stacks: &[LookupStack<'_>],
    tables: &[Table],
    multiplicities: &[Vec<i32>],
    challenges: usize,
    binding: &[u8],
) -> Result<LookupOutput, LookupError> {
    let mut fs = Fs::new(&LookupShape::of(stacks, tables, challenges), binding);
    prove_with(stacks, tables, multiplicities, challenges, &mut fs, true, true)
}

/// [`prove`] inside a caller's transcript (the PIOP transcript of the integrated
/// protocol, which has bound the commitments before the challenges `X`). `check` runs
/// the prover's self-checks (root identity, session claims); tests pass `false` to hand a
/// dishonest witness to the verifier.
///
/// # Errors
/// As [`prove`], or when the transcript refuses an event.
pub fn prove_in(
    stacks: &[LookupStack<'_>],
    tables: &[Table],
    multiplicities: &[Vec<i32>],
    challenges: usize,
    transcript: &mut impl LookupTranscript,
    check: bool,
) -> Result<LookupOutput, LookupError> {
    prove_with(stacks, tables, multiplicities, challenges, transcript, check, true)
}

/// [`prove`] without the prover's self-checks, so tests can hand a dishonest witness to
/// the verifier.
#[doc(hidden)]
pub fn prove_unchecked(
    stacks: &[LookupStack<'_>],
    tables: &[Table],
    multiplicities: &[Vec<i32>],
    challenges: usize,
    binding: &[u8],
) -> Result<LookupOutput, LookupError> {
    let mut fs = Fs::new(&LookupShape::of(stacks, tables, challenges), binding);
    prove_with(stacks, tables, multiplicities, challenges, &mut fs, false, true)
}

#[allow(clippy::too_many_lines)]
fn prove_with(
    stacks: &[LookupStack<'_>],
    tables: &[Table],
    multiplicities: &[Vec<i32>],
    challenges: usize,
    fs: &mut impl LookupTranscript,
    check: bool,
    dedicated: bool,
) -> Result<LookupOutput, LookupError> {
    let start = Instant::now();
    let shape = LookupShape::of(stacks, tables, challenges);
    shape.validate()?;
    if multiplicities.len() != tables.len()
        || multiplicities.iter().zip(tables).any(|(m, t)| m.len() != t.len())
    {
        return Err(LookupError::Shape("one multiplicity column per table entry".into()));
    }
    let index_halves: Vec<(Vec<i32>, Vec<i32>)> = tables.iter().map(|t| t.index_halves()).collect();
    let xs = fs.challenges(LookupSite::X, challenges)?;

    let build = Instant::now();
    // The integer path of session `depth - 1` per stack (prover only, `SymLanes`).
    let sym: Vec<Option<SymLanes>> = if dedicated {
        stacks.iter().map(|stack| SymLanes::of(&stack.stacked)).collect()
    } else {
        stacks.iter().map(|_| None).collect()
    };
    let shifted = |j: usize, x: usize| xs[x] - E::from_i64(i64::from(stacks[j].shape.shift));
    // `X - shift - c` for an integer-path stack of centre `c`.
    let centred = |j: usize, x: usize| {
        shifted(j, x) - E::from_i64(i64::from(sym[j].as_ref().map_or(0, |lanes| lanes.center)))
    };
    // Layer `depth - 2` of every challenge in one pass over the leaves of a stack.
    let mut uppers: Vec<Vec<Layer>> = (0..stacks.len())
        .map(|j| {
            if let Some(lanes) = &sym[j] {
                let xs_j: Vec<E> = (0..challenges).map(|x| centred(j, x)).collect();
                stack_upper_bottoms(&stacks[j].stacked, lanes.center, &xs_j)
            } else {
                Vec::new()
            }
        })
        .collect();
    let mut trees: Vec<FractionTree> = (0..shape.tree_count())
        .map(|id| {
            let (x, kind) = shape.tree(id);
            match kind {
                TreeKind::Stack(j) if sym[j].is_some() => {
                    FractionTree::from_upper(std::mem::take(&mut uppers[j][x]), stacks[j].shape.depth())
                }
                TreeKind::Stack(j) => FractionTree::from_bottom(stack_bottom(&stacks[j].stacked, shifted(j, x))),
                TreeKind::Table(t) => FractionTree::from_bottom(table_bottom(&multiplicities[t], xs[x])),
            }
        })
        .collect();
    drop(uppers);
    let mut stats = ProveStats {
        trees: build.elapsed().as_secs_f64(),
        layer_bytes: trees.iter().map(FractionTree::bytes).sum(),
        ..ProveStats::default()
    };
    debug_assert!(trees.iter().all(|t| t.root().1 == root_of(&t.layer1()).1));

    let layer1: Vec<[E; 4]> = trees.iter().map(FractionTree::layer1).collect();
    if check {
        check_roots(&shape, &layer1)?;
    }
    fs.absorb(LookupSite::Layer1, &layer1.concat())?;
    let mu = one(fs, LookupSite::Mu(1))?;
    let mut point = vec![mu];
    // Claims (p, q) on layer `s - 1` at `point`, per tree with depth >= s.
    let mut claims: Vec<(E, E)> = layer1
        .iter()
        .map(|&[p0, p1, q0, q1]| (line(p0, p1, mu), line(q0, q1, mu)))
        .collect();
    let mut stack_claims: Vec<Option<(Vec<E>, E)>> = vec![None; stacks.len()];
    let mut table_claims: Vec<Option<(Vec<E>, E)>> = vec![None; tables.len()];
    let mut sessions = Vec::new();

    for s in 2..=shape.max_depth() {
        let session_start = Instant::now();
        let active: Vec<usize> = (0..shape.tree_count()).filter(|&id| shape.tree_depth(id) >= s).collect();
        let gammas = fs.challenges(LookupSite::Gamma(s), 2 * active.len())?;
        let mut claim = E::zero();
        for (i, &id) in active.iter().enumerate() {
            claim += gammas[2 * i] * claims[id].0 + gammas[2 * i + 1] * claims[id].1;
        }
        let plan = session_plan(&shape, s, &xs, &active, &gammas);
        let half = 1_usize << (s - 1);
        let gamma_of = |id: usize| {
            let i = active.binary_search(&id).expect("an active tree");
            (gammas[2 * i], gammas[2 * i + 1])
        };
        let mut groups: Vec<(Role, SessionGroup<'_>)> = Vec::with_capacity(plan.groups.len());
        for (kind, instances) in &plan.groups {
            match kind {
                GroupKind::Layer(ids) => {
                    // Trees whose layer `s` is the one above their leaves run on the stack's
                    // integer lanes; the others on their stored layer.
                    let mut stored = Vec::new();
                    let mut by_stack: Vec<Vec<(usize, E, E, E)>> = vec![Vec::new(); stacks.len()];
                    for &id in ids {
                        let (x, tree_kind) = shape.tree(id);
                        match tree_kind {
                            TreeKind::Stack(j) if sym[j].is_some() && stacks[j].shape.depth() == s + 1 => {
                                let (gp, gq) = gamma_of(id);
                                by_stack[j].push((id, centred(j, x), gp, gq));
                            }
                            _ => stored.push(id),
                        }
                    }
                    for (j, list) in by_stack.iter().enumerate().filter(|(_, l)| !l.is_empty()) {
                        let sym_lanes = &sym[j].as_ref().expect("an integer-path stack").lanes;
                        let trees_x: Vec<(E, E, E)> = list.iter().map(|&(_, x, gp, gq)| (x, gp, gq)).collect();
                        let coefficients = SymLanes::coefficients(&trees_x);
                        let [s0, s1, m0, m1] = sym_lanes.each_ref().map(Vec::as_slice);
                        let group = match IntPairGroup::new([s0, s1], Some([m0, m1]), coefficients, &point) {
                            Some(group) => SessionGroup::Pair(group),
                            None => {
                                let lanes = sym_lanes.iter().map(|lane| Lane::Source(Source::Ints(lane))).collect();
                                SessionGroup::Generic(
                                    Group::new(lanes, SymLanes::instances(coefficients), Some(&point), s - 1, DEGREE)
                                        .map_err(LookupError::Kernel)?,
                                )
                            }
                        };
                        groups.push((Role::Sym(list.iter().map(|&(id, x, _, _)| (id, x)).collect()), group));
                    }
                    if stored.is_empty() {
                        continue;
                    }
                    let dedicated_group = if dedicated {
                        let layer_trees = stored
                            .iter()
                            .map(|&id| {
                                let layer = &trees[id].layers[s];
                                let (gamma_p, gamma_q) = gamma_of(id);
                                LayerTree {
                                    p: &layer.p,
                                    q: &layer.q,
                                    gamma_p,
                                    gamma_q,
                                }
                            })
                            .collect();
                        LayerGroup::new(layer_trees, &point)
                    } else {
                        None
                    };
                    let group = match dedicated_group {
                        Some(group) => SessionGroup::Layer(group),
                        None => {
                            let lanes: Vec<Lane<'_>> = stored
                                .iter()
                                .flat_map(|&id| {
                                    let layer = &trees[id].layers[s];
                                    let (p0, p1) = layer.p.split_at(half);
                                    let (q0, q1) = layer.q.split_at(half);
                                    [p0, p1, q0, q1].map(|cells| Lane::Source(Source::Field(cells)))
                                })
                                .collect();
                            let instances = stored
                                .iter()
                                .enumerate()
                                .flat_map(|(i, &id)| layer_instances(i, gamma_of(id)))
                                .collect();
                            SessionGroup::Generic(
                                Group::new(lanes, instances, Some(&point), s - 1, DEGREE)
                                    .map_err(LookupError::Kernel)?,
                            )
                        }
                    };
                    groups.push((Role::Trees(stored), group));
                }
                GroupKind::Stack(j) => {
                    let (v0, v1) = stacks[*j].stacked.split_at(half);
                    // The plan's instances: constant, linear on v_0 + v_1, bilinear v_0 v_1.
                    let pair = if dedicated && instances.len() == 3 {
                        let c = [instances[0].gamma, instances[1].gamma, instances[2].gamma, E::zero(), E::zero()];
                        IntPairGroup::new([v0, v1], None, c, &point)
                    } else {
                        None
                    };
                    let group = match pair {
                        Some(group) => SessionGroup::Pair(group),
                        None => {
                            let lanes = vec![Lane::Source(Source::Ints(v0)), Lane::Source(Source::Ints(v1))];
                            SessionGroup::Generic(
                                Group::new(lanes, instances.clone(), Some(&point), s - 1, DEGREE)
                                    .map_err(LookupError::Kernel)?,
                            )
                        }
                    };
                    groups.push((Role::Leaves, group));
                }
                GroupKind::Table(t) => {
                    let (m0, m1) = multiplicities[*t].split_at(half);
                    let (t0, t1) = &index_halves[*t];
                    let lanes = vec![
                        Lane::Source(Source::Ints(m0)),
                        Lane::Source(Source::Ints(m1)),
                        Lane::Source(Source::Ints(t0)),
                        Lane::Source(Source::Ints(t1)),
                    ];
                    let group = Group::new(lanes, instances.clone(), Some(&point), s - 1, DEGREE)
                        .map_err(LookupError::Kernel)?;
                    groups.push((Role::Leaves, SessionGroup::Generic(group)));
                }
            }
        }
        let mut rounds = Vec::with_capacity(s - 1);
        let mut rho = Vec::with_capacity(s - 1);
        for round in 0..s - 1 {
            let mut message = vec![E::zero(); DEGREE];
            for (_, group) in &mut groups {
                for (slot, value) in message.iter_mut().zip(group.round_message(round)) {
                    *slot += value;
                }
            }
            fs.absorb(LookupSite::Round(s, round), &message)?;
            let challenge = one(fs, LookupSite::RoundChallenge(s, round))?;
            claim = evaluate_compressed(&message, claim, challenge);
            for (_, group) in &mut groups {
                group.bind(round, challenge);
            }
            rounds.push(message);
            rho.push(challenge);
        }
        if check
            && (groups.iter().fold(E::zero(), |acc, (_, g)| acc + g.claim()) != claim
                || groups.iter().any(|(_, g)| !g.consistent()))
        {
            return Err(LookupError::Terminal(s));
        }
        let mut by_tree: Vec<Option<[E; 4]>> = vec![None; shape.tree_count()];
        let mut leaves = Vec::new();
        for (role, group) in &groups {
            let values = group.terminal_values();
            match role {
                Role::Trees(ids) => {
                    for (&id, chunk) in ids.iter().zip(values.as_chunks::<4>().0) {
                        by_tree[id] = Some(*chunk);
                    }
                }
                Role::Sym(list) => {
                    for &(id, x) in list {
                        by_tree[id] = Some(SymLanes::terminal(&values, x));
                    }
                }
                Role::Leaves => leaves.push([values[0], values[1]]),
            }
        }
        drop(groups);
        let terminals: Vec<[E; 4]> = plan
            .continuing
            .iter()
            .map(|&id| by_tree[id].expect("every continuing tree has terminal values"))
            .collect();
        fs.absorb(LookupSite::Terminals(s), &terminals.concat())?;
        fs.absorb(LookupSite::Leaves(s), &leaves.concat())?;
        let mu = one(fs, LookupSite::Mu(s))?;
        for (&id, &[p0, p1, q0, q1]) in plan.continuing.iter().zip(&terminals) {
            claims[id] = (line(p0, p1, mu), line(q0, q1, mu));
        }
        rho.push(mu);
        for (&j, leaf) in plan.ending_stacks.iter().zip(&leaves) {
            stack_claims[j] = Some((rho.clone(), line(leaf[0], leaf[1], mu)));
        }
        for (&t, leaf) in plan.ending_tables.iter().zip(&leaves[plan.ending_stacks.len()..]) {
            table_claims[t] = Some((rho.clone(), line(leaf[0], leaf[1], mu)));
        }
        point = rho;
        // Layer `s` is no longer read: unmapped on a detached thread, off the PIOP branch's
        // critical path (lever A, docs/perf.md Section 11).
        let done: Vec<Layer> = trees
            .iter_mut()
            .filter_map(|tree| tree.layers.get_mut(s).map(std::mem::take))
            .collect();
        agg_pcs_config::sched::drop_in_background("lookup layer", done);
        sessions.push(SessionProof {
            rounds,
            terminals,
            leaves,
        });
        stats.sessions.push(session_start.elapsed().as_secs_f64());
    }
    let claims = LookupClaims {
        stacks: stack_claims.into_iter().map(|c| c.expect("every stack ends")).collect(),
        multiplicities: table_claims.into_iter().map(|c| c.expect("every table ends")).collect(),
    };
    stats.total = start.elapsed().as_secs_f64();
    Ok(LookupOutput {
        proof: LookupProof { layer1, sessions },
        claims,
        stats,
    })
}

// ---------------------------------------------------------------------------------------
// The verifier.
// ---------------------------------------------------------------------------------------

/// Verify a lookup proof against its public layout and return the claims on the committed
/// columns, which the caller must check (Akita openings in the integrated protocol).
///
/// # Errors
/// On a malformed proof, a failed root identity or a failed session terminal.
///
/// # Panics
/// Never: the leaf counts are checked against the layout before they are read.
pub fn verify(shape: &LookupShape, proof: &LookupProof, binding: &[u8]) -> Result<LookupClaims, LookupError> {
    let mut fs = Fs::new(shape, binding);
    verify_in(shape, proof, &mut fs)
}

/// [`verify`] inside a caller's transcript (the PIOP transcript of the integrated
/// protocol).
///
/// # Errors
/// As [`verify`], or when the transcript refuses an event.
pub fn verify_in(
    shape: &LookupShape,
    proof: &LookupProof,
    fs: &mut impl LookupTranscript,
) -> Result<LookupClaims, LookupError> {
    shape.validate()?;
    let max_depth = shape.max_depth();
    if proof.layer1.len() != shape.tree_count() || proof.sessions.len() + 1 != max_depth {
        return Err(LookupError::Shape("tree or session count".into()));
    }
    let xs = fs.challenges(LookupSite::X, shape.challenges)?;
    check_roots(shape, &proof.layer1)?;
    fs.absorb(LookupSite::Layer1, &proof.layer1.concat())?;
    let mu = one(fs, LookupSite::Mu(1))?;
    let mut point = vec![mu];
    let mut claims: Vec<(E, E)> = proof
        .layer1
        .iter()
        .map(|&[p0, p1, q0, q1]| (line(p0, p1, mu), line(q0, q1, mu)))
        .collect();
    let mut stack_claims: Vec<Option<(Vec<E>, E)>> = vec![None; shape.stacks.len()];
    let mut table_claims: Vec<Option<(Vec<E>, E)>> = vec![None; shape.tables.len()];
    for (s, session) in (2..=max_depth).zip(&proof.sessions) {
        let active: Vec<usize> = (0..shape.tree_count()).filter(|&id| shape.tree_depth(id) >= s).collect();
        let gammas = fs.challenges(LookupSite::Gamma(s), 2 * active.len())?;
        let plan = session_plan(shape, s, &xs, &active, &gammas);
        if session.rounds.len() != s - 1
            || session.rounds.iter().any(|m| m.len() != DEGREE)
            || session.terminals.len() != plan.continuing.len()
            || session.leaves.len() != plan.ending_stacks.len() + plan.ending_tables.len()
        {
            return Err(LookupError::Shape(format!("session {s}")));
        }
        let mut claim = E::zero();
        for (i, &id) in active.iter().enumerate() {
            claim += gammas[2 * i] * claims[id].0 + gammas[2 * i + 1] * claims[id].1;
        }
        let mut rho = Vec::with_capacity(s);
        for (round, message) in session.rounds.iter().enumerate() {
            fs.absorb(LookupSite::Round(s, round), message)?;
            let challenge = one(fs, LookupSite::RoundChallenge(s, round))?;
            claim = evaluate_compressed(message, claim, challenge);
            rho.push(challenge);
        }
        let eq_factor = eq_evaluation(&point, &rho);
        let mut expected = E::zero();
        let mut leaves = session.leaves.iter();
        for (kind, instances) in &plan.groups {
            let values: Vec<E> = match kind {
                GroupKind::Layer(_) => session.terminals.concat(),
                GroupKind::Stack(_) => leaves.next().expect("counted").to_vec(),
                GroupKind::Table(_) => {
                    let [m0, m1] = *leaves.next().expect("counted");
                    let t0 = Table::index_evaluation(&rho);
                    let t1 = t0 + E::from_u64(1 << (s - 1));
                    vec![m0, m1, t0, t1]
                }
            };
            expected += expected_terminal(instances, &values, eq_factor, 0);
        }
        if expected != claim {
            return Err(LookupError::Terminal(s));
        }
        fs.absorb(LookupSite::Terminals(s), &session.terminals.concat())?;
        fs.absorb(LookupSite::Leaves(s), &session.leaves.concat())?;
        let mu = one(fs, LookupSite::Mu(s))?;
        for (&id, &[p0, p1, q0, q1]) in plan.continuing.iter().zip(&session.terminals) {
            claims[id] = (line(p0, p1, mu), line(q0, q1, mu));
        }
        rho.push(mu);
        for (&j, leaf) in plan.ending_stacks.iter().zip(&session.leaves) {
            stack_claims[j] = Some((rho.clone(), line(leaf[0], leaf[1], mu)));
        }
        for (&t, leaf) in plan
            .ending_tables
            .iter()
            .zip(&session.leaves[plan.ending_stacks.len()..])
        {
            table_claims[t] = Some((rho.clone(), line(leaf[0], leaf[1], mu)));
        }
        point = rho;
    }
    Ok(LookupClaims {
        stacks: stack_claims.into_iter().map(|c| c.expect("every stack ends")).collect(),
        multiplicities: table_claims.into_iter().map(|c| c.expect("every table ends")).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use aerie_falcon::protocol::algebra::evaluate_mle;
    use agg_witness::gen::SplitMix64;

    fn lift(cells: &[i32]) -> Vec<E> {
        cells.iter().map(|&c| E::from_i64(i64::from(c))).collect()
    }

    fn column(rng: &mut SplitMix64, vars: usize, bound: u64) -> Vec<i32> {
        (0..1 << vars)
            .map(|_| i32::try_from(rng.below(bound)).expect("small"))
            .collect()
    }

    /// The L16 main-domain stack `[z_lo | u_lo | v_lo | v_hi]` with honest-looking values
    /// (the encoder of `docs/lookup.md` Section 1: `z + 2^17`, `v + 2^31`; relation H-u's
    /// `B_ulo = u + 190,465 - 63,079 H_u`).
    fn main_stack(rng: &mut SplitMix64, vars: usize) -> LookupStack<'static> {
        let cells = 1 << vars;
        let mut columns: Vec<Vec<i32>> = (0..4).map(|_| Vec::with_capacity(cells)).collect();
        for _ in 0..cells {
            let z = rng.below(2 * 130_993 + 1).cast_signed() - 130_993;
            let u = rng.below(2 * 190_465 + 1).cast_signed() - 190_465;
            let v = rng.below(2 * 67_068_447 + 1).cast_signed() - 67_068_447;
            let (zs, vs) = (z + (1 << 17), v + (1 << 31));
            let ulo = crate::manifest::U16_DIGITS.decompose(u)[0] + (1 << 15);
            assert!((zs >> 16) < 4 && (0..1 << 16).contains(&ulo) && (vs >> 16) < 1 << 16);
            for (c, value) in [zs & 0xffff, ulo, vs & 0xffff, vs >> 16].into_iter().enumerate() {
                columns[c].push(i32::try_from(value).expect("16 bits"));
            }
        }
        let refs: Vec<&[i32]> = columns.iter().map(Vec::as_slice).collect();
        LookupStack::new(&refs, 0).expect("stack")
    }

    /// Six carry-like columns (padded to 8 slots) and one single-column stack.
    fn carry_stacks(rng: &mut SplitMix64, vars: usize) -> Vec<LookupStack<'static>> {
        let six: Vec<Vec<i32>> = (0..6).map(|_| column(rng, vars, 1024)).collect();
        let refs: Vec<&[i32]> = six.iter().map(Vec::as_slice).collect();
        let one = column(rng, vars - 2, 256);
        vec![
            LookupStack::new(&refs, 0).expect("stack"),
            LookupStack::new(&[&one], 0).expect("stack"),
        ]
    }

    fn check_claims(stacks: &[LookupStack], m: &[Vec<i32>], claims: &LookupClaims) {
        for (stack, (point, value)) in stacks.iter().zip(&claims.stacks) {
            assert_eq!(evaluate_mle(&lift(stack.stacked()), point), *value);
            // The same claim split into column claims at the low coordinates.
            let shape = stack.shape();
            let cells = 1 << shape.vars;
            let low = &point[..shape.vars];
            let combined = shape
                .column_weights(point)
                .iter()
                .enumerate()
                .fold(E::zero(), |acc, (c, &w)| {
                    acc + w * evaluate_mle(&lift(&stack.stacked()[c * cells..(c + 1) * cells]), low)
                });
            assert_eq!(combined, *value);
        }
        for (column, (point, value)) in m.iter().zip(&claims.multiplicities) {
            assert_eq!(evaluate_mle(&lift(column), point), *value);
        }
    }

    fn round_trip(stacks: &[LookupStack], tables: &[Table], challenges: usize) -> LookupOutput {
        let m = multiplicities(stacks, tables).expect("honest");
        let output = prove(stacks, tables, &m, challenges, b"test").expect("prove");
        let shape = LookupShape::of(stacks, tables, challenges);
        let claims = verify(&shape, &output.proof, b"test").expect("verify");
        assert_eq!(claims, output.claims);
        check_claims(stacks, &m, &claims);
        assert_eq!(output.proof.elements(), shape.proof_elements());
        output
    }

    /// The dedicated prover paths (the integer lanes of session `depth - 1`, the layer
    /// group of `kernel::layer`, the one-pass layer `depth - 2`) give the same proof and
    /// claims as the generic kernel on the stored layers, byte for byte.
    #[test]
    fn dedicated_paths_match_the_generic_kernel() {
        let compare = |stacks: &[LookupStack], tables: &[Table], challenges: usize| {
            let m = multiplicities(stacks, tables).expect("honest");
            let shape = LookupShape::of(stacks, tables, challenges);
            let run = |dedicated: bool| {
                let mut fs = Fs::new(&shape, b"compare");
                prove_with(stacks, tables, &m, challenges, &mut fs, true, dedicated).expect("prove")
            };
            let (fast, reference) = (run(true), run(false));
            assert_eq!(fast.proof, reference.proof, "proof, {challenges} challenges");
            assert_eq!(fast.claims, reference.claims);
            assert!(fast.stats.layer_bytes < reference.stats.layer_bytes);
            assert_eq!(verify(&shape, &fast.proof, b"compare").expect("verify"), fast.claims);
        };
        let mut rng = SplitMix64::new(4242);
        for vars in [1, 2, 6, 9] {
            let mut stacks = vec![main_stack(&mut rng, vars)];
            stacks.extend(carry_stacks(&mut rng, vars + 3));
            for challenges in [1, 2, 3] {
                compare(&stacks, &[Table::T16], challenges);
            }
        }
        // Balanced storage (shift 2^15) as in the integrated protocol, a stack of depth 3
        // and one of depth 2 (no integer path), a small table.
        let balanced: Vec<Vec<i32>> = (0..3)
            .map(|_| column(&mut rng, 8, 1 << 16).iter().map(|v| v - (1 << 15)).collect())
            .collect();
        let refs: Vec<&[i32]> = balanced.iter().map(Vec::as_slice).collect();
        let tiny = column(&mut rng, 2, 256);
        let small = column(&mut rng, 2, 256);
        let stacks = vec![
            LookupStack::with_shift(&refs, 0, 1 << 15).expect("stack"),
            LookupStack::new(&[&tiny, &small], 1).expect("stack"),
            LookupStack::new(&[&tiny], 1).expect("stack"),
        ];
        compare(&stacks, &[Table::T16, Table::new(8)], 2);
    }

    #[test]
    fn tree_root_is_the_fractional_sum() {
        let mut rng = SplitMix64::new(1);
        let values = column(&mut rng, 6, 1 << 16);
        let x = crate::test_support::field(&mut rng);
        let tree = FractionTree::from_bottom(stack_bottom(&values, x));
        let (p, q) = tree.root();
        let direct = values.iter().fold(E::zero(), |acc, &g| {
            acc + (x - E::from_i64(i64::from(g))).inverse().expect("nonzero")
        });
        assert_eq!(p * q.inverse().expect("nonzero"), direct);
        let m: Vec<i32> = (0..64).map(|i| (i * 7) % 5).collect();
        let tree = FractionTree::from_bottom(table_bottom(&m, x));
        let (p, q) = tree.root();
        let direct = m.iter().enumerate().fold(E::zero(), |acc, (t, &mt)| {
            acc + (x - E::from_u64(t as u64)).inverse().expect("nonzero").mul_i64(i64::from(mt))
        });
        assert_eq!(p * q.inverse().expect("nonzero"), direct);
    }

    #[test]
    fn multiplicity_digits_round_trip() {
        let m = vec![0, 1, 7, 8, 15, 16, 255, 100_663_296, 125_269_879];
        let digits = multiplicity_digits(&m);
        assert_eq!(digits.len(), MULTIPLICITY_DIGITS);
        assert!(digits.iter().flatten().all(|&d| (-8..=7).contains(&d)));
        let composed = compose_multiplicity(&digits);
        assert_eq!(composed, m.iter().map(|&v| i64::from(v)).collect::<Vec<_>>());
    }

    #[test]
    fn honest_main_stack_verifies_at_several_sizes() {
        let mut rng = SplitMix64::new(2);
        // d_main 10 (tree depth 12 < 16: the table tree is the deeper one), 13 (15), 14 (16, equal depths).
        for vars in [10, 13, 14] {
            let stacks = vec![main_stack(&mut rng, vars)];
            let output = round_trip(&stacks, &[Table::T16], 1);
            assert_eq!(output.proof.sessions.len(), 15.max(vars + 1));
        }
    }

    #[test]
    fn full_layout_with_carry_stacks_and_two_challenges_verifies() {
        let mut rng = SplitMix64::new(3);
        // The main stack is deeper than the table (depth 17 > 16).
        let mut stacks = vec![main_stack(&mut rng, 15)];
        stacks.extend(carry_stacks(&mut rng, 9));
        for challenges in [1, 2] {
            round_trip(&stacks, &[Table::T16], challenges);
        }
    }

    #[test]
    fn two_tables_verify_and_a_cross_table_value_is_rejected() {
        let mut rng = SplitMix64::new(4);
        let tables = [Table::T16, Table::new(8)];
        let main = main_stack(&mut rng, 9);
        let small = column(&mut rng, 8, 256);
        let mut stacks = vec![main, LookupStack::new(&[&small], 1).expect("stack")];
        round_trip(&stacks, &tables, 1);
        // 300 is an entry of T_16 but not of T_8: the honest prover aborts, and a prover
        // that books it under T_16 (what an offset-tagged shared identity would accept)
        // fails the identity of T_8.
        let honest = multiplicities(&stacks, &tables).expect("honest");
        stacks[1].stacked_mut()[5] = 300;
        assert!(matches!(
            multiplicities(&stacks, &tables),
            Err(LookupError::OutOfTable { stack: 1, index: 5, value: 300 })
        ));
        let mut m = honest;
        m[1][usize::try_from(small[5]).expect("entry")] -= 1;
        m[0][300] += 1;
        let output = prove_unchecked(&stacks, &tables, &m, 1, b"test").expect("unchecked");
        let shape = LookupShape::of(&stacks, &tables, 1);
        assert!(matches!(
            verify(&shape, &output.proof, b"test"),
            Err(LookupError::Identity { challenge: 0, .. })
        ));
    }

    #[test]
    fn out_of_range_digit_and_wrong_multiplicity_are_rejected() {
        let mut rng = SplitMix64::new(5);
        let tables = [Table::T16];
        let mut stacks = vec![main_stack(&mut rng, 10)];
        let honest = multiplicities(&stacks, &tables).expect("honest");
        let shape = LookupShape::of(&stacks, &tables, 1);
        // A wrong multiplicity: move one count to a neighbouring entry.
        let mut m = honest.clone();
        let entry = usize::try_from(stacks[0].stacked()[0]).expect("entry");
        m[0][entry] -= 1;
        m[0][(entry + 1) % (1 << 16)] += 1;
        assert!(matches!(prove(&stacks, &tables, &m, 1, b"test"), Err(LookupError::Identity { .. })));
        let output = prove_unchecked(&stacks, &tables, &m, 1, b"test").expect("unchecked");
        assert!(matches!(verify(&shape, &output.proof, b"test"), Err(LookupError::Identity { .. })));
        // v_hi = 2^16 (the cell of column 3 at position 7) with the honest multiplicities
        // of the replaced value: the identity fails for every multiplicity column.
        let cells = 1 << 10;
        let index = 3 * cells + 7;
        let old = usize::try_from(stacks[0].stacked()[index]).expect("entry");
        stacks[0].stacked_mut()[index] = 1 << 16;
        assert!(matches!(multiplicities(&stacks, &tables), Err(LookupError::OutOfTable { .. })));
        // Cheat 1: count only the in-table cells. Cheat 2: book 2^16 at its residue 0.
        let mut ignoring = honest;
        ignoring[0][old] -= 1;
        let mut wrapped = ignoring.clone();
        wrapped[0][0] += 1;
        for cheat in [ignoring, wrapped] {
            let output = prove_unchecked(&stacks, &tables, &cheat, 1, b"test").expect("unchecked");
            assert!(matches!(verify(&shape, &output.proof, b"test"), Err(LookupError::Identity { .. })));
        }
    }

    fn median(values: &mut [f64]) -> f64 {
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    }

    fn peak_rss_mib() -> f64 {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|status| {
                status
                    .lines()
                    .find(|l| l.starts_with("VmHWM:"))
                    .and_then(|l| l.split_whitespace().nth(1).and_then(|kb| kb.parse::<f64>().ok()))
            })
            .map_or(0.0, |kb| kb / 1024.0)
    }

    /// The L16 lookups of `n` signatures: the main stack `[z_lo | u_lo | v_lo | v_hi]` on
    /// `d_main = 10 + d_sig` variables and the carry stacks `kappa^(1)` (6 columns on
    /// `D_1`), `kappa^(2)` (6 on `D_2`), `kappa^(3)` (1 on `D_3`), all in `T_16`.
    fn layout(rng: &mut SplitMix64, n: usize) -> Vec<LookupStack<'static>> {
        let d_sig = n.next_power_of_two().max(8).trailing_zeros() as usize;
        let d1 = 13 + d_sig.saturating_sub(9);
        let d2 = 13 + d_sig.saturating_sub(13);
        let d3 = 12 + d_sig.saturating_sub(13);
        let carries = |rng: &mut SplitMix64, count: usize, vars: usize, bound: u64| {
            let columns: Vec<Vec<i32>> = (0..count).map(|_| column(rng, vars, bound)).collect();
            let refs: Vec<&[i32]> = columns.iter().map(Vec::as_slice).collect();
            LookupStack::new(&refs, 0).expect("stack")
        };
        vec![
            main_stack(rng, 10 + d_sig),
            carries(rng, 6, d1, 1024),
            carries(rng, 6, d2, 1024),
            carries(rng, 1, d3, 256),
        ]
    }

    /// `AGG_LOOKUP_SIGS=1024,8192 RUSTUP_TOOLCHAIN=stable taskset -c 0,1,3-15 cargo test -p
    /// agg-piop --release --lib lookup::tests::timing -- --ignored --nocapture`
    /// (default sizes 1,024 and 8,192; `RAYON_NUM_THREADS` threads, default 7; challenge counts
    /// `AGG_LOOKUP_CHALLENGES`, default `1,2`; warmup plus five timed runs, medians).
    #[test]
    #[ignore = "timing: run with --release --ignored --nocapture"]
    #[allow(clippy::cast_precision_loss)]
    fn timing_at_1024_and_8192_signatures() {
        let sizes: Vec<usize> = std::env::var("AGG_LOOKUP_SIGS").map_or_else(
            |_| vec![1024, 8192],
            |list| list.split(',').map(|s| s.trim().parse().expect("a size")).collect(),
        );
        let threads = std::env::var("RAYON_NUM_THREADS").ok().and_then(|t| t.parse().ok()).unwrap_or(7);
        let counts: Vec<usize> = std::env::var("AGG_LOOKUP_CHALLENGES").map_or_else(
            |_| vec![1, 2],
            |list| list.split(',').map(|s| s.trim().parse().expect("a count")).collect(),
        );
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().expect("pool");
        pool.install(|| {
            for n in sizes {
                let mut rng = SplitMix64::new(n as u64);
                let stacks = layout(&mut rng, n);
                let tables = [Table::T16];
                let start = Instant::now();
                let m = multiplicities(&stacks, &tables).expect("honest");
                let m_time = start.elapsed().as_secs_f64();
                let lookups: usize = stacks.iter().map(|s| s.shape().leaves()).sum();
                let main_depth = stacks[0].shape().depth();
                println!(
                    "N = {n}: {lookups} lookups (main stack depth {main_depth}), multiplicities {:.1} ms, identity error 2^{:.1} per X",
                    m_time * 1e3,
                    LookupShape::of(&stacks, &tables, 1).identity_error_log2(0)
                );
                // AGG_LOOKUP_AB=1: the generic-kernel reference and the dedicated paths,
                // alternating run by run (the same load for both).
                let variants: &[bool] = if std::env::var("AGG_LOOKUP_AB").is_ok() { &[false, true] } else { &[true] };
                for &challenges in &counts {
                    let shape = LookupShape::of(&stacks, &tables, challenges);
                    // Per variant: trees, sessions, bottom, next, total, verify.
                    let mut series: Vec<[Vec<f64>; 6]> = variants.iter().map(|_| Default::default()).collect();
                    let mut last: Vec<Option<LookupOutput>> = variants.iter().map(|_| None).collect();
                    for run in 0..6 {
                        for (v, &dedicated) in variants.iter().enumerate() {
                            let mut fs = Fs::new(&shape, b"timing");
                            let output = prove_with(&stacks, &tables, &m, challenges, &mut fs, true, dedicated)
                                .expect("prove");
                            let start = Instant::now();
                            let claims = verify(&shape, &output.proof, b"timing").expect("verify");
                            let verified = start.elapsed().as_secs_f64();
                            assert_eq!(claims, output.claims);
                            if run > 0 {
                                let stats = &output.stats;
                                // Session s is stats.sessions[s - 2]; the main stack ends at its depth.
                                for (slot, value) in series[v].iter_mut().zip([
                                    stats.trees,
                                    stats.sessions.iter().sum::<f64>(),
                                    stats.sessions[main_depth - 2],
                                    stats.sessions[main_depth - 3],
                                    stats.total,
                                    verified,
                                ]) {
                                    slot.push(value);
                                }
                            }
                            last[v] = Some(output);
                        }
                    }
                    for (v, &dedicated) in variants.iter().enumerate() {
                        let output = last[v].take().expect("runs");
                        let [trees, sessions, bottom, upper, total, verify_time] = &mut series[v];
                        let per_sig = |t: f64| t * 1e6 / n as f64;
                        let total_ms = median(total) * 1e3;
                        println!(
                            "  X x{challenges} {}: prove {total_ms:.1} ms ({:.1} us/sig): trees {:.1}, sessions {:.1} (bottom {:.1}, next {:.1}); verify {:.2} ms; proof {} B; layers {:.0} MiB; peak RSS {:.0} MiB",
                            if dedicated { "dedicated" } else { "generic" },
                            per_sig(total_ms / 1e3),
                            median(trees) * 1e3,
                            median(sessions) * 1e3,
                            median(bottom) * 1e3,
                            median(upper) * 1e3,
                            median(verify_time) * 1e3,
                            output.proof.byte_len(),
                            output.stats.layer_bytes as f64 / f64::from(1 << 20),
                            peak_rss_mib(),
                        );
                        let per_session: Vec<String> =
                            output.stats.sessions.iter().map(|t| format!("{:.1}", t * 1e3)).collect();
                        println!("    sessions 2.. (ms, last run): {}", per_session.join(" "));
                    }
                }
            }
        });
    }

    /// The cost of a work nonce at the challenge site of `X` (option (i) of
    /// `docs/lookup.md` Section 2): Akita's `preview_grinding_predicate` per candidate
    /// (clone the Blake2b512 sponge, absorb the 4-byte nonce, squeeze 32 bytes), serial as
    /// in Akita's `search_grinding_nonce`, and on 7 threads.
    #[test]
    #[ignore = "timing: run with --release --ignored --nocapture"]
    fn grinding_rate() {
        use akita_transcript::TranscriptSponge;
        use spongefish::DuplexSpongeInterface;
        let mut base = TranscriptSponge::default();
        base.absorb(&[7_u8; 256]);
        let mut warm = [0_u8; 32];
        base.clone().squeeze(&mut warm);
        let attempt = |nonce: u32| {
            let mut sponge = base.clone();
            sponge.absorb(&nonce.to_le_bytes());
            let mut predicate = [0_u8; 32];
            sponge.squeeze(&mut predicate);
            predicate[0] == 0 && predicate[1] == 0
        };
        let serial_attempts = 1_u32 << 20;
        let start = Instant::now();
        let hits = (0..serial_attempts).filter(|&n| attempt(n)).count();
        let serial = start.elapsed().as_secs_f64() / f64::from(serial_attempts);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(7).build().expect("pool");
        let parallel_attempts = 1_u32 << 23;
        let start = Instant::now();
        let parallel_hits = pool.install(|| (0..parallel_attempts).into_par_iter().filter(|&n| attempt(n)).count());
        let parallel = start.elapsed().as_secs_f64() / f64::from(parallel_attempts);
        println!(
            "grinding: {:.0} ns per attempt serial, {:.1} ns on 7 threads (hits {hits}, {parallel_hits})",
            serial * 1e9,
            parallel * 1e9
        );
        for bits in [23_i32, 25, 26, 28] {
            let attempts = 2.0_f64.powi(bits);
            println!(
                "  {bits} bits: expected {:.2} s serial, {:.2} s on 7 threads",
                attempts * serial,
                attempts * parallel
            );
        }
    }

    #[test]
    fn tampered_messages_and_claims_are_rejected() {
        let mut rng = SplitMix64::new(6);
        let tables = [Table::T16];
        let mut stacks = vec![main_stack(&mut rng, 10)];
        stacks.extend(carry_stacks(&mut rng, 8));
        let m = multiplicities(&stacks, &tables).expect("honest");
        let shape = LookupShape::of(&stacks, &tables, 2);
        let output = prove(&stacks, &tables, &m, 2, b"test").expect("prove");
        let proof = &output.proof;
        assert!(verify(&shape, proof, b"test").is_ok());
        // A different binding (other commitments) changes every challenge.
        assert!(verify(&shape, proof, b"other").is_err());
        // A round message.
        let mut bad = proof.clone();
        bad.sessions[5].rounds[2][1] += E::one();
        assert_eq!(verify(&shape, &bad, b"test"), Err(LookupError::Terminal(7)));
        // Swapped terminal values P_0 <-> P_1 of the first continuing tree.
        let mut bad = proof.clone();
        bad.sessions[3].terminals[0].swap(0, 1);
        assert!(verify(&shape, &bad, b"test").is_err());
        // A swapped leaf pair (v_0 <-> v_1) of the main stack, which ends at session 12:
        // the leaf terms are symmetric in v_0, v_1, so session 12 passes, but the changed
        // transcript breaks session 13 (the table tree continues).
        let mut bad = proof.clone();
        bad.sessions[10].leaves[0].swap(0, 1);
        assert_eq!(verify(&shape, &bad, b"test"), Err(LookupError::Terminal(13)));
        // When the stack ends in the last session, only its opening claim catches the swap.
        let small: Vec<Vec<i32>> = (0..4).map(|_| column(&mut rng, 8, 256)).collect();
        let refs: Vec<&[i32]> = small.iter().map(Vec::as_slice).collect();
        let deep = [LookupStack::new(&refs, 0).expect("stack")];
        let t8 = [Table::new(8)];
        let m8 = multiplicities(&deep, &t8).expect("honest");
        let mut bad = prove(&deep, &t8, &m8, 1, b"test").expect("prove").proof;
        let last = bad.sessions.len() - 1;
        bad.sessions[last].leaves[0].swap(0, 1);
        let claims = verify(&LookupShape::of(&deep, &t8, 1), &bad, b"test").expect("symmetric");
        let (point, value) = &claims.stacks[0];
        assert_ne!(evaluate_mle(&lift(deep[0].stacked()), point), *value);
        // A swapped multiplicity pair (m_0 <-> m_1) of the table, which ends at session 16.
        let mut bad = proof.clone();
        let last = bad.sessions.len() - 1;
        bad.sessions[last].leaves[0].swap(0, 1);
        assert_eq!(verify(&shape, &bad, b"test"), Err(LookupError::Terminal(16)));
        // A changed layer-1 value moves the root.
        let mut bad = proof.clone();
        bad.layer1[0][0] += E::one();
        assert!(matches!(verify(&shape, &bad, b"test"), Err(LookupError::Identity { .. })));
        // A truncated proof and a wrong layout.
        let mut bad = proof.clone();
        bad.sessions.pop();
        assert!(matches!(verify(&shape, &bad, b"test"), Err(LookupError::Shape(_))));
        let one = LookupShape::of(&stacks, &tables, 1);
        assert!(verify(&one, proof, b"test").is_err());
    }
}
