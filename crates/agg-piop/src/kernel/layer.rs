//! Dedicated sum-check groups of the LogUp-GKR sessions and of stage 3 (`docs/lookup.md`
//! Section 4.6, `docs/piop.md` Section 20). Prover only.
//!
//! The polynomial of a round is unique, so any exact computation of it gives the messages
//! of the generic [`super::Group`] byte for byte. These groups compute it from fewer
//! products: every group is a *bilinear form* `f = sum_l L_l + sum_(i,j) L_i L_j` on a
//! row-major table of transformed lanes (no coefficient left in the summand), produced
//! from the group's own level-0 data by a linear (affine) map that commutes with the folds
//! and is inverted once at the terminal. A round reads the previous table once: it folds
//! four rows into two by the previous challenge and accumulates `q(0)` and the leading
//! coefficient `q_2` of the inner polynomial on the new pair; the linear coefficient
//! follows from the running claim (Gruen's factorisation of the equality factor, as in
//! [`super::Group`]). The first round reads the level-0 data directly and computes `q(1)`
//! as well, so a group's claim comes from its tables, never from the caller.
//!
//! - [`LayerGroup`]: the layer group, per tree `gamma_p (P_0 Q_1 + P_1 Q_0) + gamma_q Q_0 Q_1`
//!   on the stored halves, as `L_1 L_2 + L_3 L_4` with `L_1 = gamma_p P_0`, `L_2 = Q_1`,
//!   `L_3 = gamma_p Q_0`, `L_4 = P_1 + (gamma_q / gamma_p) Q_1`.
//! - [`IntPairGroup`]: integer lanes `S_0, S_1` (and optionally `M_0, M_1`) with the summand
//!   `c + c_s (S_0 + S_1) + c_m (M_0 + M_1 + S_0 S_1) + c_k (S_0 M_1 + S_1 M_0) + c_mm M_0 M_1`
//!   (the stack bottom: `S_c = v_c`, no `M`; the integer lanes of session `depth - 1`). The
//!   first round runs on exact `i64` products; the table holds `S_0 (, M_0)` and
//!   `W = C (1, S_1, M_1)`, so `f = W_0 + S_0 W_1 + M_0 W_2`.
//!
//! A constructor returns `None` when its transform is not invertible (a zero coefficient,
//! probability about `2^-128`) or the integers exceed the exact range; the caller then uses
//! [`super::Group`].

use akita_algebra::split_eq::GruenSplitEq;
use jolt_field::{Field, Fold, One, Ring, Unreduced, Zero};
use jolt_poly::UnivariatePoly;
use rayon::prelude::*;

use super::field::{fm, fmi, fold, lift, mad, mad_int, Acc, ITEM_CHUNK, MIN_PARALLEL_ITEMS};
use super::{compress, evaluate_compressed};
use crate::OpeningField as E;

/// Message degree of a session.
const DEGREE: usize = 3;

/// The level-0 data of a dedicated group and its table form.
trait Level0: Sync {
    /// Lanes per table row.
    fn width(&self) -> usize;
    /// Rows at level 0.
    fn rows(&self) -> usize;
    /// Item `k` of the first round (level-0 rows `2k`, `2k + 1`): `sums[0] += f(row 2k)`,
    /// `sums[1] += f(row 2k + 1)` when `full`, `sums[2] +=` the quadratic part of
    /// `f` at the difference.
    fn first_item(&self, k: usize, full: bool, sums: &mut [Acc; 3]);
    /// Row `i` of level 1 (level-0 rows `2i`, `2i + 1` folded by `r`), transformed.
    fn folded_row(&self, r: E, context: &<E as Fold>::Ctx, i: usize, out: &mut [E]);
    /// The lanes of the summand `sum_l L_l + sum L_i L_j` on a table row.
    fn linear(&self) -> &[usize];
    /// The products of the summand.
    fn products(&self) -> &[(usize, usize)];
    /// The group's terminal lane values from the fully folded row.
    fn terminal(&self, row: &[E]) -> Vec<E>;
    /// A materialized level-0 table (then the rounds run on it from the first, and
    /// [`Self::first_item`] and [`Self::folded_row`] are never called).
    fn initial_table(&self) -> Option<Vec<E>> {
        None
    }
    /// `[q(0), q(1), q_2]` of the first round with the equality weights `(first, second)`
    /// (`None`: unweighted) when the source has a faster path than [`Self::first_item`].
    fn first_round(&self, _weights: Option<(&[E], &[E])>) -> Option<[E; 3]> {
        None
    }
}

/// `sum_k first[k mod |first|] second[k / |first|] values(k)` for `N` integer quantities per
/// item, unreduced within a block of `first` (two products per term, [`mad_int`]).
fn weighted_int_sums<const N: usize>(
    items: usize,
    first: &[E],
    second: &[E],
    values: impl Fn(usize) -> [i64; N] + Sync,
) -> [E; N] {
    let first_bits = first.len().trailing_zeros();
    let first_mask = first.len() - 1;
    let run = |task: usize| -> [Acc; N] {
        let start = task * ITEM_CHUNK;
        let end = ((task + 1) * ITEM_CHUNK).min(items);
        let mut outer = [Acc::zero(); N];
        let mut inner = [Acc::zero(); N];
        let mut block = start >> first_bits;
        for k in start..end {
            if (k >> first_bits) != block {
                for (o, i) in outer.iter_mut().zip(&mut inner) {
                    mad(o, second[block], E::reduce_product(*i));
                    *i = Acc::zero();
                }
                block = k >> first_bits;
            }
            let weight = first[k & first_mask];
            for (i, v) in inner.iter_mut().zip(values(k)) {
                mad_int(i, weight, v);
            }
        }
        if start < end {
            for (o, i) in outer.iter_mut().zip(&inner) {
                mad(o, second[block], E::reduce_product(*i));
            }
        }
        outer
    };
    let tasks = items.div_ceil(ITEM_CHUNK);
    let add = |mut left: [Acc; N], right: [Acc; N]| {
        for (l, r) in left.iter_mut().zip(right) {
            *l += r;
        }
        left
    };
    let total = if items >= MIN_PARALLEL_ITEMS && tasks > 1 {
        (0..tasks).into_par_iter().map(run).reduce(|| [Acc::zero(); N], add)
    } else {
        run(0)
    };
    total.map(E::reduce_product)
}

/// The round driver shared by the dedicated groups.
struct Engine<S: Level0> {
    source: S,
    /// The equality factor, or `None` for a plain sum over the domain.
    eq: Option<GruenSplitEq<E>>,
    /// A coefficient applied to every round polynomial (the batching gamma of an eq-free
    /// group; one otherwise).
    scale: E,
    vars: usize,
    bound: usize,
    claim: E,
    pending: Vec<E>,
    /// The challenge whose fold the next pass performs.
    unfolded: Option<E>,
    /// The table from level 1 on (empty before the first fold).
    table: Vec<E>,
}

impl<S: Level0> Engine<S> {
    fn new(source: S, eq_point: &[E]) -> Option<Self> {
        let eq = GruenSplitEq::new(eq_point).ok()?;
        Self::with_eq(source, Some(eq), eq_point.len(), E::one())
    }

    fn with_eq(source: S, eq: Option<GruenSplitEq<E>>, vars: usize, scale: E) -> Option<Self> {
        if vars == 0 || source.rows() != 1 << vars {
            return None;
        }
        let table = source.initial_table().unwrap_or_default();
        Some(Self {
            source,
            eq,
            scale,
            vars,
            bound: 0,
            claim: E::zero(),
            pending: Vec::new(),
            unfolded: None,
            table,
        })
    }

    /// `f` on rows `a` (into `sums[0]`) and `b` (into `sums[1]`, when `full`) and its
    /// quadratic part at `b - a` (into `sums[2]`).
    #[inline]
    fn table_item(&self, a: &[E], b: &[E], full: bool, sums: &mut [Acc; 3]) {
        for &l in self.source.linear() {
            lift(&mut sums[0], a[l]);
            if full {
                lift(&mut sums[1], b[l]);
            }
        }
        for &(i, j) in self.source.products() {
            mad(&mut sums[0], a[i], a[j]);
            mad(&mut sums[2], b[i] - a[i], b[j] - a[j]);
            if full {
                mad(&mut sums[1], b[i], b[j]);
            }
        }
    }

    /// `inner += weight * sums` for the sums the round needs (`q(1)` only when `full`,
    /// `q_2` only with products).
    #[inline]
    fn add_weighted(&self, sums: [Acc; 3], weight: E, full: bool, inner: &mut [Acc; 3]) {
        mad(&mut inner[0], weight, E::reduce_product(sums[0]));
        if full {
            mad(&mut inner[1], weight, E::reduce_product(sums[1]));
        }
        if !self.source.products().is_empty() {
            mad(&mut inner[2], weight, E::reduce_product(sums[2]));
        }
    }

    /// The weighted item of rows `a`, `b`; a purely linear summand is weighted directly.
    #[inline]
    fn weighted_item(&self, a: &[E], b: &[E], weight: E, full: bool, inner: &mut [Acc; 3]) {
        if self.source.products().is_empty() {
            for &l in self.source.linear() {
                mad(&mut inner[0], weight, a[l]);
                if full {
                    mad(&mut inner[1], weight, b[l]);
                }
            }
            return;
        }
        let mut sums = [Acc::zero(); 3];
        self.table_item(a, b, full, &mut sums);
        self.add_weighted(sums, weight, full, inner);
    }

    /// One pass: fold by the pending challenge (if any) into the next table, then (when
    /// `evaluate`) `[q(0), q(1), q_2]` of the round on the new level, with the remaining
    /// equality weights (`q(1)` only when `full`).
    #[allow(clippy::too_many_lines)]
    fn pass(&mut self, evaluate: bool, full: bool) -> [E; 3] {
        let width = self.source.width();
        let challenge = self.unfolded.take();
        let from_source = self.table.is_empty();
        let rows_in = if from_source { self.source.rows() } else { self.table.len() / width };
        let rows_out = if challenge.is_some() { rows_in / 2 } else { rows_in };
        let context = challenge.map(E::precompute);
        let previous = std::mem::take(&mut self.table);
        let unweighted = self.eq.is_none();
        let (first, second): (Vec<E>, Vec<E>) = match &self.eq {
            Some(eq) if evaluate => {
                let (f, s) = eq.remaining_eq_tables();
                (f.to_vec(), s.to_vec())
            }
            _ => (vec![E::one()], vec![E::one()]),
        };
        if evaluate && from_source && challenge.is_none() {
            let weights = (!unweighted).then_some((first.as_slice(), second.as_slice()));
            if let Some(sums) = self.source.first_round(weights) {
                return sums;
            }
        }
        let items = (rows_out / 2).max(1);
        debug_assert!(!evaluate || unweighted || first.len() * second.len() == items);
        let first_bits = first.len().trailing_zeros();
        let first_mask = first.len() - 1;
        let keep = challenge.is_some();
        let this = &*self;
        // Output rows 2k, 2k + 1 (row 0 alone at the final fold) into `rows`.
        let fill = |k: usize, rows: &mut [E]| {
            let (r, context) = (challenge.expect("a fold"), context.as_ref().expect("a fold"));
            for (t, out) in rows.chunks_exact_mut(width).enumerate() {
                let row = 2 * k + t;
                if from_source {
                    this.source.folded_row(r, context, row, out);
                } else {
                    let (even, odd) = (2 * row * width, (2 * row + 1) * width);
                    for (j, slot) in out.iter_mut().enumerate() {
                        *slot = fold(context, previous[even + j], previous[odd + j]);
                    }
                }
            }
        };
        let run = |task: usize, out: &mut [E]| -> [Acc; 3] {
            let start = task * ITEM_CHUNK;
            let end = ((task + 1) * ITEM_CHUNK).min(items);
            let mut outer = [Acc::zero(); 3];
            let mut inner = [Acc::zero(); 3];
            let mut block = start >> first_bits;
            for k in start..end {
                if unweighted {
                    // A plain sum: the item sums accumulate unreduced, without weights.
                    if keep {
                        let offset = (k - start) * 2 * width;
                        let len = (2 * width).min(out.len() - offset);
                        let rows = &mut out[offset..offset + len];
                        fill(k, rows);
                        if evaluate {
                            let (a, b) = rows.split_at(width);
                            this.table_item(a, b, full, &mut outer);
                        }
                    } else if from_source {
                        this.source.first_item(k, full, &mut outer);
                    } else {
                        let a = &previous[2 * k * width..(2 * k + 1) * width];
                        let b = &previous[(2 * k + 1) * width..(2 * k + 2) * width];
                        this.table_item(a, b, full, &mut outer);
                    }
                    continue;
                }
                if evaluate && (k >> first_bits) != block {
                    for (o, i) in outer.iter_mut().zip(&mut inner) {
                        mad(o, second[block], E::reduce_product(*i));
                        *i = Acc::zero();
                    }
                    block = k >> first_bits;
                }
                let weight = first[k & first_mask];
                if keep {
                    let offset = (k - start) * 2 * width;
                    let len = (2 * width).min(out.len() - offset);
                    let rows = &mut out[offset..offset + len];
                    fill(k, rows);
                    if !evaluate {
                        continue;
                    }
                    let (a, b) = rows.split_at(width);
                    this.weighted_item(a, b, weight, full, &mut inner);
                } else if from_source {
                    let mut sums = [Acc::zero(); 3];
                    this.source.first_item(k, full, &mut sums);
                    this.add_weighted(sums, weight, full, &mut inner);
                } else {
                    let a = &previous[2 * k * width..(2 * k + 1) * width];
                    let b = &previous[(2 * k + 1) * width..(2 * k + 2) * width];
                    this.weighted_item(a, b, weight, full, &mut inner);
                }
            }
            if evaluate && !unweighted && start < end {
                for (o, i) in outer.iter_mut().zip(&inner) {
                    mad(o, second[block], E::reduce_product(*i));
                }
            }
            outer
        };
        let tasks = items.div_ceil(ITEM_CHUNK);
        let parallel = items >= MIN_PARALLEL_ITEMS && tasks > 1;
        // A parallel first touch (a serial `vec!` would write the table on one thread).
        let mut table: Vec<E> = if !keep {
            Vec::new()
        } else if parallel {
            rayon::iter::repeat_n(E::zero(), rows_out * width).collect()
        } else {
            vec![E::zero(); rows_out * width]
        };
        let add = |mut left: [Acc; 3], right: [Acc; 3]| {
            for (l, r) in left.iter_mut().zip(right) {
                *l += r;
            }
            left
        };
        let total = match (parallel, keep) {
            (true, true) => table
                .par_chunks_mut(ITEM_CHUNK * 2 * width)
                .enumerate()
                .map(|(task, out)| run(task, out))
                .reduce(|| [Acc::zero(); 3], add),
            (true, false) => (0..tasks)
                .into_par_iter()
                .map(|task| run(task, &mut []))
                .reduce(|| [Acc::zero(); 3], add),
            (false, true) => run(0, &mut table),
            (false, false) => run(0, &mut []),
        };
        if !keep {
            // A round over the current table without a fold (never happens after the first
            // round, which reads the source) keeps the table.
            self.table = previous;
        } else {
            drop(previous);
            self.table = table;
        }
        total.map(E::reduce_product)
    }

    fn round_message(&mut self, round: usize) -> Vec<E> {
        assert_eq!(round, self.bound, "rounds in order");
        let full = round == 0
            || self
                .eq
                .as_ref()
                .is_some_and(|eq| !eq.can_recover_linear_q_term_from_claim());
        let [q0, at_one, q2] = self.pass(true, full).map(|q| q * self.scale);
        let coefficients = match &self.eq {
            Some(eq) => {
                let q1 = if full {
                    at_one - q0 - q2
                } else {
                    let (l0, l1) = eq.linear_factor_evals();
                    let inverse = l1.inverse().expect("recoverable round");
                    (self.claim - l0 * q0) * inverse - q0 - q2
                };
                eq.gruen_mul(&UnivariatePoly::new(vec![q0, q1, q2])).into_coefficients()
            }
            // s = q: s(0) + s(1) = 2 q_0 + q_1 + q_2 = claim.
            None => {
                let q1 = if full { at_one - q0 - q2 } else { self.claim - q0 - q0 - q2 };
                vec![q0, q1, q2]
            }
        };
        if round == 0 {
            self.claim = coefficients.iter().fold(E::zero(), |s, &c| s + c) + coefficients[0];
        }
        let message = compress(&coefficients, DEGREE);
        self.pending = message.clone();
        message
    }

    fn bind(&mut self, round: usize, challenge: E) {
        assert_eq!(round, self.bound, "rounds in order");
        assert!(!self.pending.is_empty(), "round_message before bind");
        self.claim = evaluate_compressed(&self.pending, self.claim, challenge);
        self.pending.clear();
        if let Some(eq) = &mut self.eq {
            eq.bind(challenge);
        }
        self.unfolded = Some(challenge);
        self.bound += 1;
        if self.bound == self.vars {
            self.pass(false, false);
        }
    }

    fn terminal_values(&self) -> Vec<E> {
        assert_eq!(self.bound, self.vars, "all rounds bound");
        self.source.terminal(&self.table)
    }

    fn terminal_consistent(&self) -> bool {
        if self.bound != self.vars {
            return false;
        }
        let row = &self.table;
        let linear = self.source.linear().iter().fold(E::zero(), |s, &l| s + row[l]);
        let sum = self
            .source
            .products()
            .iter()
            .fold(linear, |s, &(i, j)| s + row[i] * row[j]);
        let factor = self.eq.as_ref().map_or(E::one(), GruenSplitEq::current_scalar);
        factor * self.scale * sum == self.claim
    }
}

// ---------------------------------------------------------------------------------------
// The layer group.
// ---------------------------------------------------------------------------------------

/// One tree of a layer group: the stored layer `s` (`2^s` cells of `p` and `q`, whose
/// halves are `P_0, P_1` and `Q_0, Q_1`) and its batching coefficients.
#[derive(Clone, Copy, Debug)]
pub struct LayerTree<'a> {
    /// The numerators of layer `s`.
    pub p: &'a [E],
    /// The denominators of layer `s`.
    pub q: &'a [E],
    /// The coefficient of the `p`-claim.
    pub gamma_p: E,
    /// The coefficient of the `q`-claim.
    pub gamma_q: E,
}

struct LayerSource<'a> {
    trees: Vec<LayerTree<'a>>,
    /// `gamma_q / gamma_p` and `1 / gamma_p` per tree.
    ratio: Vec<(E, E)>,
    products: Vec<(usize, usize)>,
    half: usize,
}

impl LayerSource<'_> {
    /// The transformed lanes of one tree from `P_0, P_1, Q_0, Q_1`.
    #[inline(always)]
    fn transform(tree: &LayerTree<'_>, ratio: E, [p0, p1, q0, q1]: [E; 4]) -> [E; 4] {
        [fm(tree.gamma_p, p0), q1, fm(tree.gamma_p, q0), p1 + fm(ratio, q1)]
    }

    #[inline(always)]
    fn cell(&self, t: usize, i: usize) -> [E; 4] {
        let tree = &self.trees[t];
        Self::transform(
            tree,
            self.ratio[t].0,
            [tree.p[i], tree.p[self.half + i], tree.q[i], tree.q[self.half + i]],
        )
    }
}

impl Level0 for LayerSource<'_> {
    fn width(&self) -> usize {
        4 * self.trees.len()
    }

    fn rows(&self) -> usize {
        self.half
    }

    fn first_item(&self, k: usize, full: bool, sums: &mut [Acc; 3]) {
        for t in 0..self.trees.len() {
            let a = self.cell(t, 2 * k);
            let b = self.cell(t, 2 * k + 1);
            mad(&mut sums[0], a[0], a[1]);
            mad(&mut sums[0], a[2], a[3]);
            mad(&mut sums[2], b[0] - a[0], b[1] - a[1]);
            mad(&mut sums[2], b[2] - a[2], b[3] - a[3]);
            if full {
                mad(&mut sums[1], b[0], b[1]);
                mad(&mut sums[1], b[2], b[3]);
            }
        }
    }

    fn folded_row(&self, _: E, context: &<E as Fold>::Ctx, i: usize, out: &mut [E]) {
        let (even, odd, half) = (2 * i, 2 * i + 1, self.half);
        for ((tree, &(ratio, _)), lanes) in self.trees.iter().zip(&self.ratio).zip(out.as_chunks_mut::<4>().0) {
            let raw = [
                fold(context, tree.p[even], tree.p[odd]),
                fold(context, tree.p[half + even], tree.p[half + odd]),
                fold(context, tree.q[even], tree.q[odd]),
                fold(context, tree.q[half + even], tree.q[half + odd]),
            ];
            *lanes = Self::transform(tree, ratio, raw);
        }
    }

    fn linear(&self) -> &[usize] {
        &[]
    }

    fn products(&self) -> &[(usize, usize)] {
        &self.products
    }

    fn terminal(&self, row: &[E]) -> Vec<E> {
        let mut out = Vec::with_capacity(row.len());
        for (&(ratio, inverse), lanes) in self.ratio.iter().zip(row.as_chunks::<4>().0) {
            let q1 = lanes[1];
            out.extend([lanes[0] * inverse, lanes[3] - ratio * q1, lanes[2] * inverse, q1]);
        }
        out
    }
}

/// The layer group of one session (see the module documentation).
pub struct LayerGroup<'a>(Engine<LayerSource<'a>>);

impl<'a> LayerGroup<'a> {
    /// A group over `trees` (every layer of `2^(vars + 1)` cells, `vars = eq_point.len()`
    /// rounds). `None` when some `gamma_p` is zero, the layers have different or wrong
    /// lengths, or there are no trees.
    #[must_use]
    pub fn new(trees: Vec<LayerTree<'a>>, eq_point: &[E]) -> Option<Self> {
        let half = 1_usize << eq_point.len();
        if trees.is_empty() || trees.iter().any(|t| t.p.len() != 2 * half || t.q.len() != 2 * half) {
            return None;
        }
        let ratio = trees
            .iter()
            .map(|t| t.gamma_p.inverse().map(|inverse| (t.gamma_q * inverse, inverse)))
            .collect::<Option<Vec<(E, E)>>>()?;
        let products = (0..trees.len()).flat_map(|t| [(4 * t, 4 * t + 1), (4 * t + 2, 4 * t + 3)]).collect();
        let source = LayerSource {
            trees,
            ratio,
            products,
            half,
        };
        Engine::new(source, eq_point).map(Self)
    }

    /// The current claim.
    #[must_use]
    pub fn claim(&self) -> E {
        self.0.claim
    }

    /// The compressed message of `round` (`3` elements).
    ///
    /// # Panics
    /// When rounds are called out of order.
    pub fn round_message(&mut self, round: usize) -> Vec<E> {
        self.0.round_message(round)
    }

    /// Bind the round's variable to `challenge` (the fold runs in the next pass).
    ///
    /// # Panics
    /// When rounds are called out of order.
    pub fn bind(&mut self, round: usize, challenge: E) {
        self.0.bind(round, challenge);
    }

    /// `P_0, P_1, Q_0, Q_1` at the bound point, per tree in order.
    ///
    /// # Panics
    /// Before every round is bound.
    #[must_use]
    pub fn terminal_values(&self) -> Vec<E> {
        self.0.terminal_values()
    }

    /// True when the running claim equals the equality factor times the summand at the
    /// terminal values (a self-check of the claim-based recovery).
    #[must_use]
    pub fn terminal_consistent(&self) -> bool {
        self.0.terminal_consistent()
    }
}

// ---------------------------------------------------------------------------------------
// Integer lanes.
// ---------------------------------------------------------------------------------------

/// Largest magnitude of an `S` lane without `M` lanes (`|dS_0 dS_1| < 2^62`).
const PAIR_BOUND: i64 = 1 << 30;

struct IntSource<'a> {
    s: [&'a [i32]; 2],
    m: Option<[&'a [i32]; 2]>,
    /// `c, c_s, c_m, c_k, c_mm`.
    c: [E; 5],
    /// Without `M`: `1 / c_m`; with `M`: `1 / (c_m c_mm - c_k^2)`.
    inverse: E,
    linear: [usize; 1],
    products: Vec<(usize, usize)>,
}

impl IntSource<'_> {
    /// `f` minus `c` at integer lanes (exact products in `i64`).
    #[inline(always)]
    fn value(&self, s0: i64, s1: i64, m: Option<(i64, i64)>) -> E {
        let [_, cs, cm, ck, cmm] = self.c;
        match m {
            None => fmi(cs, s0 + s1) + fmi(cm, s0 * s1),
            Some((m0, m1)) => {
                fmi(cs, s0 + s1) + fmi(cm, m0 + m1 + s0 * s1) + fmi(ck, s0 * m1 + s1 * m0) + fmi(cmm, m0 * m1)
            }
        }
    }

    /// `S_1` (and `M_1`) folded into the `W` lanes.
    #[inline(always)]
    fn w_lanes(&self, s1: E, m1: E, out: &mut [E]) {
        let [c, cs, cm, ck, cmm] = self.c;
        match self.m {
            None => {
                out[0] = c + fm(cs, s1);
                out[1] = cs + fm(cm, s1);
            }
            Some(_) => {
                out[0] = c + fm(cs, s1) + fm(cm, m1);
                out[1] = cs + fm(cm, s1) + fm(ck, m1);
                out[2] = cm + fm(ck, s1) + fm(cmm, m1);
            }
        }
    }
}

#[inline(always)]
fn int_fold(r: E, lane: &[i32], i: usize) -> E {
    let (low, high) = (i64::from(lane[2 * i]), i64::from(lane[2 * i + 1]));
    E::from_i64(low) + fmi(r, high - low)
}

impl Level0 for IntSource<'_> {
    fn width(&self) -> usize {
        if self.m.is_some() { 5 } else { 3 }
    }

    fn rows(&self) -> usize {
        self.s[0].len()
    }

    fn first_item(&self, k: usize, full: bool, sums: &mut [Acc; 3]) {
        let cell = |lane: &[i32], i: usize| i64::from(lane[i]);
        let (a, b) = (2 * k, 2 * k + 1);
        let [s0, s1] = self.s;
        let c = self.c[0];
        let [_, _, cm, ck, cmm] = self.c;
        let (sa0, sa1, sb0, sb1) = (cell(s0, a), cell(s1, a), cell(s0, b), cell(s1, b));
        let (ds0, ds1) = (sb0 - sa0, sb1 - sa1);
        match self.m {
            None => {
                lift(&mut sums[0], c + self.value(sa0, sa1, None));
                if full {
                    lift(&mut sums[1], c + self.value(sb0, sb1, None));
                }
                lift(&mut sums[2], fmi(cm, ds0 * ds1));
            }
            Some([m0, m1]) => {
                let (ma0, ma1, mb0, mb1) = (cell(m0, a), cell(m1, a), cell(m0, b), cell(m1, b));
                let (dm0, dm1) = (mb0 - ma0, mb1 - ma1);
                lift(&mut sums[0], c + self.value(sa0, sa1, Some((ma0, ma1))));
                if full {
                    lift(&mut sums[1], c + self.value(sb0, sb1, Some((mb0, mb1))));
                }
                lift(
                    &mut sums[2],
                    fmi(cm, ds0 * ds1) + fmi(ck, ds0 * dm1 + ds1 * dm0) + fmi(cmm, dm0 * dm1),
                );
            }
        }
    }

    fn folded_row(&self, r: E, _: &<E as Fold>::Ctx, i: usize, out: &mut [E]) {
        let [s0, s1] = self.s;
        out[0] = int_fold(r, s0, i);
        let s1 = int_fold(r, s1, i);
        match self.m {
            None => self.w_lanes(s1, E::zero(), &mut out[1..3]),
            Some([m0, m1]) => {
                out[1] = int_fold(r, m0, i);
                let m1 = int_fold(r, m1, i);
                self.w_lanes(s1, m1, &mut out[2..5]);
            }
        }
    }

    fn first_round(&self, weights: Option<(&[E], &[E])>) -> Option<[E; 3]> {
        let (first, second) = weights?;
        let cell = |lane: &[i32], i: usize| i64::from(lane[i]);
        let [s0, s1] = self.s;
        let [c, cs, cm, ck, cmm] = self.c;
        let items = self.rows() / 2;
        // The weights sum to `(sum first)(sum second)` (one in exact arithmetic).
        let weight = first.iter().fold(E::zero(), |a, &w| a + w) * second.iter().fold(E::zero(), |a, &w| a + w);
        let constant = c * weight;
        Some(match self.m {
            None => {
                let sums = weighted_int_sums(items, first, second, |k| {
                    let (a, b) = (2 * k, 2 * k + 1);
                    let (sa0, sa1, sb0, sb1) = (cell(s0, a), cell(s1, a), cell(s0, b), cell(s1, b));
                    [sa0 + sa1, sa0 * sa1, sb0 + sb1, sb0 * sb1, (sb0 - sa0) * (sb1 - sa1)]
                });
                [
                    constant + cs * sums[0] + cm * sums[1],
                    constant + cs * sums[2] + cm * sums[3],
                    cm * sums[4],
                ]
            }
            Some([m0, m1]) => {
                let sums = weighted_int_sums(items, first, second, |k| {
                    let (a, b) = (2 * k, 2 * k + 1);
                    let (sa0, sa1, sb0, sb1) = (cell(s0, a), cell(s1, a), cell(s0, b), cell(s1, b));
                    let (ma0, ma1, mb0, mb1) = (cell(m0, a), cell(m1, a), cell(m0, b), cell(m1, b));
                    let (ds0, ds1, dm0, dm1) = (sb0 - sa0, sb1 - sa1, mb0 - ma0, mb1 - ma1);
                    [
                        sa0 + sa1,
                        ma0 + ma1 + sa0 * sa1,
                        sa0 * ma1 + sa1 * ma0,
                        ma0 * ma1,
                        sb0 + sb1,
                        mb0 + mb1 + sb0 * sb1,
                        sb0 * mb1 + sb1 * mb0,
                        mb0 * mb1,
                        ds0 * ds1,
                        ds0 * dm1 + ds1 * dm0,
                        dm0 * dm1,
                    ]
                });
                [
                    constant + cs * sums[0] + cm * sums[1] + ck * sums[2] + cmm * sums[3],
                    constant + cs * sums[4] + cm * sums[5] + ck * sums[6] + cmm * sums[7],
                    cm * sums[8] + ck * sums[9] + cmm * sums[10],
                ]
            }
        })
    }

    fn linear(&self) -> &[usize] {
        &self.linear
    }

    fn products(&self) -> &[(usize, usize)] {
        &self.products
    }

    fn terminal(&self, row: &[E]) -> Vec<E> {
        let [_, cs, cm, ck, cmm] = self.c;
        match self.m {
            None => vec![row[0], (row[2] - cs) * self.inverse],
            Some(_) => {
                // [W_1 - c_s; W_2 - c_m] = [[c_m, c_k], [c_k, c_mm]] [S_1; M_1].
                let (u, v) = (row[3] - cs, row[4] - cm);
                let s1 = (cmm * u - ck * v) * self.inverse;
                let m1 = (cm * v - ck * u) * self.inverse;
                vec![row[0], s1, row[1], m1]
            }
        }
    }
}

/// Integer lanes with a bipartite quadratic summand (see the module documentation).
pub struct IntPairGroup<'a>(Engine<IntSource<'a>>);

impl<'a> IntPairGroup<'a> {
    /// The group over `S_0, S_1` (and `M_0, M_1`), every lane of `2^vars` cells
    /// (`vars = eq_point.len()` rounds), with `coefficients = [c, c_s, c_m, c_k, c_mm]`
    /// (`c_k`, `c_mm` unused without `M`). With `M` lanes the caller guarantees
    /// `|S| <= 2^17` and `|M| <= 2^31` (exact `i64` products); without them the lanes are
    /// checked against `2^30`. `None` when the transform is singular, a lane has the wrong
    /// length or a value is out of range.
    #[must_use]
    pub fn new(s: [&'a [i32]; 2], m: Option<[&'a [i32]; 2]>, coefficients: [E; 5], eq_point: &[E]) -> Option<Self> {
        let rows = 1_usize << eq_point.len();
        let lanes = s.iter().chain(m.iter().flatten());
        if lanes.clone().any(|lane| lane.len() != rows) {
            return None;
        }
        let [_, _, cm, ck, cmm] = coefficients;
        let (inverse, products) = match m {
            None => {
                let out_of_range = s
                    .iter()
                    .any(|lane| lane.par_iter().any(|&v| i64::from(v).abs() > PAIR_BOUND));
                if out_of_range {
                    return None;
                }
                (cm.inverse()?, vec![(0, 2)])
            }
            Some(_) => ((cm * cmm - ck * ck).inverse()?, vec![(0, 3), (1, 4)]),
        };
        let linear = [if m.is_some() { 2 } else { 1 }];
        let source = IntSource {
            s,
            m,
            c: coefficients,
            inverse,
            linear,
            products,
        };
        Engine::new(source, eq_point).map(Self)
    }

    /// The current claim.
    #[must_use]
    pub fn claim(&self) -> E {
        self.0.claim
    }

    /// The compressed message of `round` (`3` elements).
    ///
    /// # Panics
    /// When rounds are called out of order.
    pub fn round_message(&mut self, round: usize) -> Vec<E> {
        self.0.round_message(round)
    }

    /// Bind the round's variable to `challenge`.
    ///
    /// # Panics
    /// When rounds are called out of order.
    pub fn bind(&mut self, round: usize, challenge: E) {
        self.0.bind(round, challenge);
    }

    /// `S_0, S_1` (then `M_0, M_1` interleaved as `S_0, S_1, M_0, M_1`) at the bound point.
    ///
    /// # Panics
    /// Before every round is bound.
    #[must_use]
    pub fn terminal_values(&self) -> Vec<E> {
        self.0.terminal_values()
    }

    /// The self-check of [`LayerGroup::terminal_consistent`].
    #[must_use]
    pub fn terminal_consistent(&self) -> bool {
        self.0.terminal_consistent()
    }
}

// ---------------------------------------------------------------------------------------
// A linear combination of integer lanes.
// ---------------------------------------------------------------------------------------

struct LinearSource<'a> {
    /// Integer or mapped byte lanes (never field lanes).
    lanes: Vec<super::Source<'a>>,
    coefficients: Vec<E>,
}

/// The integer value of cell `i` of an integer or byte lane.
#[inline(always)]
fn int_cell(lane: &super::Source<'_>, i: usize) -> i64 {
    match lane {
        super::Source::Ints(cells) => i64::from(cells[i]),
        super::Source::Bytes { cells, map } => map[cells[i] as u8 as usize],
        super::Source::Field(_) => unreachable!("integer lanes only"),
    }
}

impl Level0 for LinearSource<'_> {
    fn width(&self) -> usize {
        1
    }

    fn rows(&self) -> usize {
        self.lanes[0].len()
    }

    /// `g = sum_l c_l f_l` on every cell (four products per cell at the L16 stacks, half
    /// of what the first round and the first fold would cost on the integer lanes).
    fn initial_table(&self) -> Option<Vec<E>> {
        // Unreduced products, one reduction per cell.
        let cell = |i: usize| {
            let mut acc = Acc::zero();
            for (lane, &c) in self.lanes.iter().zip(&self.coefficients) {
                mad_int(&mut acc, c, int_cell(lane, i));
            }
            E::reduce_product(acc)
        };
        let rows = self.rows();
        Some(if rows >= MIN_PARALLEL_ITEMS {
            (0..rows).into_par_iter().with_min_len(1 << 10).map(cell).collect()
        } else {
            (0..rows).map(cell).collect()
        })
    }

    fn first_item(&self, _: usize, _: bool, _: &mut [Acc; 3]) {
        unreachable!("the rounds run on the materialized table")
    }

    fn folded_row(&self, _: E, _: &<E as Fold>::Ctx, _: usize, _: &mut [E]) {
        unreachable!("the rounds run on the materialized table")
    }

    fn linear(&self) -> &[usize] {
        &[0]
    }

    fn products(&self) -> &[(usize, usize)] {
        &[]
    }

    fn terminal(&self, _: &[E]) -> Vec<E> {
        Vec::new()
    }
}

/// `sum_x eq(point, x) sum_l c_l f_l(x)` over integer or byte lanes `f_l` (`|f_l| < 2^62`),
/// as the single field lane `g = sum_l c_l f_l` built once. The group yields no terminal
/// lane values (the caller reads the lanes' values from another group at the same point).
pub struct LinearIntGroup<'a>(Engine<LinearSource<'a>>);

impl<'a> LinearIntGroup<'a> {
    /// `None` without lanes, with a field lane, or with lanes of other lengths than
    /// `2^eq_point.len()`.
    #[must_use]
    pub fn new(lanes: Vec<super::Source<'a>>, coefficients: Vec<E>, eq_point: &[E]) -> Option<Self> {
        let rows = 1_usize << eq_point.len();
        if lanes.is_empty()
            || lanes.len() != coefficients.len()
            || lanes
                .iter()
                .any(|l| l.len() != rows || matches!(l, super::Source::Field(_)))
        {
            return None;
        }
        Engine::new(LinearSource { lanes, coefficients }, eq_point).map(Self)
    }

    /// The current claim.
    #[must_use]
    pub fn claim(&self) -> E {
        self.0.claim
    }

    /// The compressed message of `round`.
    ///
    /// # Panics
    /// When rounds are called out of order.
    pub fn round_message(&mut self, round: usize) -> Vec<E> {
        self.0.round_message(round)
    }

    /// Bind the round's variable to `challenge`.
    ///
    /// # Panics
    /// When rounds are called out of order.
    pub fn bind(&mut self, round: usize, challenge: E) {
        self.0.bind(round, challenge);
    }

    /// The self-check of [`LayerGroup::terminal_consistent`].
    #[must_use]
    pub fn terminal_consistent(&self) -> bool {
        self.0.terminal_consistent()
    }
}

// ---------------------------------------------------------------------------------------
// A field weight times an integer combination, without an equality factor.
// ---------------------------------------------------------------------------------------

struct WeightIntSource<'a> {
    /// The weight lane (a public field table).
    weight: &'a [E],
    /// The integer lanes in the group's lane order (all also carried in the table for their
    /// terminal values) with their coefficients in the combination.
    lanes: Vec<(super::Source<'a>, i64)>,
    constant: i64,
    /// The group lane of the weight (the others are the integer lanes in order).
    weight_lane: usize,
}

impl WeightIntSource<'_> {
    /// `g(i) = sum_l c_l f_l(i) + constant` (exact: coefficients and cells are small).
    #[inline(always)]
    fn combination(&self, i: usize) -> i64 {
        self.lanes
            .iter()
            .fold(self.constant, |acc, (lane, c)| acc + c * int_cell(lane, i))
    }
}

impl Level0 for WeightIntSource<'_> {
    fn width(&self) -> usize {
        2 + self.lanes.len()
    }

    fn rows(&self) -> usize {
        self.weight.len()
    }

    fn first_item(&self, k: usize, full: bool, sums: &mut [Acc; 3]) {
        let (a, b) = (2 * k, 2 * k + 1);
        let (wa, wb) = (self.weight[a], self.weight[b]);
        let (ga, gb) = (self.combination(a), self.combination(b));
        mad_int(&mut sums[0], wa, ga);
        if full {
            mad_int(&mut sums[1], wb, gb);
        }
        mad_int(&mut sums[2], wb - wa, gb - ga);
    }

    fn first_round(&self, weights: Option<(&[E], &[E])>) -> Option<[E; 3]> {
        if weights.is_some() {
            return None;
        }
        let items = self.rows() / 2;
        let run = |range: std::ops::Range<usize>| {
            let mut sums = [Acc::zero(); 3];
            for k in range {
                self.first_item(k, true, &mut sums);
            }
            sums
        };
        let total = if items >= MIN_PARALLEL_ITEMS {
            (0..items.div_ceil(ITEM_CHUNK))
                .into_par_iter()
                .map(|task| run(task * ITEM_CHUNK..((task + 1) * ITEM_CHUNK).min(items)))
                .reduce(
                    || [Acc::zero(); 3],
                    |mut left, right| {
                        for (l, r) in left.iter_mut().zip(right) {
                            *l += r;
                        }
                        left
                    },
                )
        } else {
            run(0..items)
        };
        Some(total.map(E::reduce_product))
    }

    fn folded_row(&self, r: E, context: &<E as Fold>::Ctx, i: usize, out: &mut [E]) {
        let (a, b) = (2 * i, 2 * i + 1);
        out[0] = fold(context, self.weight[a], self.weight[b]);
        let (ga, gb) = (self.combination(a), self.combination(b));
        out[1] = E::from_i64(ga) + fmi(r, gb - ga);
        for (slot, (lane, _)) in out[2..].iter_mut().zip(&self.lanes) {
            let (low, high) = (int_cell(lane, a), int_cell(lane, b));
            *slot = E::from_i64(low) + fmi(r, high - low);
        }
    }

    fn linear(&self) -> &[usize] {
        &[]
    }

    fn products(&self) -> &[(usize, usize)] {
        &[(0, 1)]
    }

    fn terminal(&self, row: &[E]) -> Vec<E> {
        let mut out = row[2..].to_vec();
        out.insert(self.weight_lane, row[0]);
        out
    }
}

/// `gamma sum_x w(x) g(x)` for a field weight lane `w` and the integer combination
/// `g = sum_l c_l f_l + c` of integer or byte lanes (no equality factor; stage 3's `w_A z`).
/// The first round sums `w g` with unreduced integer products; the table holds `w`, `g` and
/// the integer lanes (folded for their terminal values).
pub struct WeightIntGroup<'a>(Engine<WeightIntSource<'a>>);

impl<'a> WeightIntGroup<'a> {
    /// `lanes` are the group's lanes in order: the weight at `weight_lane` (a field lane) and
    /// integer or byte lanes, with `coefficients[l]` the coefficient of lane `l` in `g`
    /// (ignored at the weight). `None` on lanes of other lengths than `2^rounds`, a field
    /// lane besides the weight, or coefficients above `2^20` (exactness of `g`).
    #[must_use]
    pub fn new(
        lanes: &[super::Source<'a>],
        weight_lane: usize,
        coefficients: &[i64],
        constant: i64,
        gamma: E,
        rounds: usize,
    ) -> Option<Self> {
        let rows = 1_usize << rounds;
        let super::Source::Field(weight) = *lanes.get(weight_lane)? else {
            return None;
        };
        if lanes.len() != coefficients.len()
            || lanes.iter().any(|l| l.len() != rows)
            || coefficients.iter().any(|c| c.abs() > 1 << 20)
            || constant.abs() > 1 << 40
        {
            return None;
        }
        let mut ints = Vec::with_capacity(lanes.len() - 1);
        for (l, (lane, &c)) in lanes.iter().zip(coefficients).enumerate() {
            if l == weight_lane {
                continue;
            }
            match lane {
                super::Source::Field(_) => return None,
                other => ints.push((*other, c)),
            }
        }
        let source = WeightIntSource {
            weight,
            lanes: ints,
            constant,
            weight_lane,
        };
        Engine::with_eq(source, None, rounds, gamma).map(Self)
    }

    /// The current claim.
    #[must_use]
    pub fn claim(&self) -> E {
        self.0.claim
    }

    /// The compressed message of `round`.
    ///
    /// # Panics
    /// When rounds are called out of order.
    pub fn round_message(&mut self, round: usize) -> Vec<E> {
        self.0.round_message(round)
    }

    /// Bind the round's variable to `challenge`.
    ///
    /// # Panics
    /// When rounds are called out of order.
    pub fn bind(&mut self, round: usize, challenge: E) {
        self.0.bind(round, challenge);
    }

    /// The lane values at the bound point, in the group's lane order.
    ///
    /// # Panics
    /// Before every round is bound.
    #[must_use]
    pub fn terminal_values(&self) -> Vec<E> {
        self.0.terminal_values()
    }

    /// The self-check of [`LayerGroup::terminal_consistent`].
    #[must_use]
    pub fn terminal_consistent(&self) -> bool {
        self.0.terminal_consistent()
    }
}
