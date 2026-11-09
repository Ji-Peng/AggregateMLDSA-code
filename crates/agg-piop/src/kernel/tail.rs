//! The dense rounds of a generic [`super::Group`] on lane-major `K` planes.
//!
//! When all of a generic group's pieces are at most quadratic in the lanes, the summand is
//! the bilinear form with coefficient slots of `agg_simd::fused`; otherwise, when every
//! piece is a sum of monomials of degree at most four (cubic and quartic lane polynomials,
//! products of three or four lanes, the bit-range identity), it is the monomial form of
//! `agg_simd::high`. Either way the dense rounds run on lane-major `K` planes: a round is
//! one fused pass that folds every lane by the previous challenge into the spare buffer and
//! evaluates the round on the just-written pairs. This replaces the row-major table with its
//! separate evaluation and fold passes. The two rounds before the table (levels 2 and 3) run
//! on the same kernels without a stored table: the lanes' compact level-2 states are
//! gathered block by block and evaluated (level 2), or gathered, folded by the challenge of
//! bind 2 and evaluated (level 3). Bind 3 gathers and folds them twice into the first table,
//! a sixteenth of the domain, while it evaluates the next round. The round polynomial is
//! unique, so the messages equal those of the scalar dense rounds byte for byte.

use agg_simd::buf::{AlignedVec, KPlanes};
use agg_simd::fused::{self, EvalAny, Fill, RowSource, Summand, Terms, MAX_LANES, MAX_SLOTS};
use agg_simd::high::{HighTerms, Monomial, MAX_DEGREE, MAX_FACTORS};
use agg_simd::k::Planes;
use agg_simd::pattern::Columns;
use akita_algebra::split_eq::GruenSplitEq;
use jolt_field::{Field, One, Ring, Zero};

use super::arena::{give_planes, take_planes, SharedArena};
use super::plan::{Pieces, PolyKind};
use super::Term;
use crate::OpeningField as E;

/// Products folded into one coefficient slot before the fused kernel's accumulator can wrap.
/// Each product is one `kmac7` of a limb below `6 · 2^52`, and the exact bound is 682.
const MAX_PRODUCTS_PER_SLOT: usize = 682;

/// The lane and slot caps of the plane tail: the fused kernels' limits, lowered by
/// `AGG_TAIL_MAX_LANES` and `AGG_TAIL_MAX_SLOTS` (an A/B switch: a group over a cap runs the
/// row-major dense rounds, whose messages are the same).
fn caps() -> (usize, usize) {
    static CAPS: std::sync::OnceLock<(usize, usize)> = std::sync::OnceLock::new();
    *CAPS.get_or_init(|| {
        let cap = |name: &str, limit: usize| {
            std::env::var(name).ok().and_then(|v| v.trim().parse::<usize>().ok()).map_or(limit, |v| v.min(limit))
        };
        (cap("AGG_TAIL_MAX_LANES", MAX_LANES), cap("AGG_TAIL_MAX_SLOTS", MAX_SLOTS))
    })
}

/// Whether summands above degree two may run on the plane tail (`AGG_TAIL_HIGH=0` or `off`
/// keeps them on the row-major dense rounds, an A/B switch with the same messages).
fn high_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("AGG_TAIL_HIGH").as_deref(), Ok("0" | "off")))
}

/// The dense pieces of a group as the term lists of the fused kernels.
pub(super) struct TailTerms {
    linear: Vec<(usize, usize)>,
    products: Vec<(usize, usize, usize)>,
    /// The monomials of a summand above degree two (the two lists above are then empty).
    high: Option<HighTerms>,
    /// The coefficient of every slot.
    slots: Vec<E>,
    /// Products accumulated into each slot.
    products_per_slot: Vec<usize>,
}

impl TailTerms {
    fn empty() -> Self {
        Self {
            linear: Vec::new(),
            products: Vec::new(),
            high: None,
            slots: Vec::new(),
            products_per_slot: Vec::new(),
        }
    }

    /// The coefficient slots of the summand.
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// The summand as the fused kernels read it.
    fn summand(&self) -> Summand<'_> {
        match &self.high {
            Some(high) => Summand::High(high),
            None => Summand::Bilinear(Terms {
                linear: &self.linear,
                products: &self.products,
                slots: self.slots.len(),
            }),
        }
    }

    /// The round's values from the flat slot sums of a pass and the cell constant: the
    /// inner polynomial at `0..D` (at one only when the pass computed it) and its
    /// coefficient of `X^D`, `D` the summand's degree (two for the bilinear form).
    fn message(&self, total: &[E], constant: E) -> Vec<E> {
        let comps = self.summand().components();
        // The equality weights sum to one over the pairs, so the constant enters every value
        // once and the leading coefficient not at all.
        let mut q = vec![constant; comps];
        q[comps - 1] = E::zero();
        for (&c, s) in self.slots.iter().zip(total.chunks_exact(comps)) {
            for (q, &s) in q.iter_mut().zip(s) {
                *q += c * s;
            }
        }
        q
    }

    /// [`Self::message`] of a round over the pairs of `source`'s rows, which are
    /// evaluated block by block as they are computed (no table is written); the remaining
    /// equality tables of `eq` weight the pairs, as in [`Tail::message`].
    pub fn rows_message(&self, source: &dyn RowSource, eq: &GruenSplitEq<E>, full: bool, constant: E) -> Vec<E> {
        let (first, second) = eq.remaining_eq_tables();
        let first_planes = KPlanes::from_fn(first.len(), |i| first[i]);
        let eval = EvalAny {
            summand: self.summand(),
            first: first_planes.planes(),
            second,
            full,
        };
        let total = fused::fused_pass_with::<Columns>(None, &[], Some(Fill::Computed(None, source)), Some(eval));
        self.message(&total, constant)
    }
}

/// The coefficients of the inner polynomial of degree `D` from its values `values[t] = q(t)`
/// at `t = 0..D` (with `values[1]` replaced by `at_one`) and its leading coefficient
/// `values[D]`: the values less the leading term, interpolated by forward differences.
pub(super) fn coefficients(values: &[E], at_one: E) -> Vec<E> {
    let degree = values.len() - 1;
    let leading = values[degree];
    let mut r: Vec<E> = (0..degree)
        .map(|t| {
            let y = if t == 1 { at_one } else { values[t] };
            y - leading * E::from_u64((t as u64).pow(degree as u32))
        })
        .collect();
    let mut q = vec![E::zero(); degree + 1];
    q[degree] = leading;
    // `r = sum_k (Delta^k r)(0) C(X, k)`, with the falling factorial `X (X - 1) ... (X - k + 1)`
    // expanded in `basis` and `k!` in `factorial`.
    let mut basis = vec![E::one()];
    let mut factorial = E::one();
    for k in 0..degree {
        if k > 0 {
            factorial *= E::from_u64(k as u64);
            let root = E::from_u64(k as u64 - 1);
            let mut next = vec![E::zero(); basis.len() + 1];
            for (i, &b) in basis.iter().enumerate() {
                next[i + 1] += b;
                next[i] -= root * b;
            }
            basis = next;
        }
        let c = r[0] * factorial.inverse().expect("nonzero factorial");
        for (q, &b) in q.iter_mut().zip(&basis) {
            *q += c * b;
        }
        for i in 0..degree - 1 - k {
            r[i] = r[i + 1] - r[i];
        }
    }
    q
}

impl TailTerms {
    /// The term lists of `pieces` over `lanes` lanes, or `None` when the lanes or distinct
    /// coefficients exceed the kernels' limits, or a piece fits neither form.
    pub fn compile(pieces: &Pieces, lanes: usize) -> Option<Self> {
        if lanes > caps().0 {
            return None;
        }
        Self::bilinear(pieces).or_else(|| if high_on() { Self::monomials(pieces) } else { None })
    }

    /// The bilinear form, when every piece is at most quadratic (and no residual generic term).
    fn bilinear(pieces: &Pieces) -> Option<Self> {
        if !pieces.generic.is_empty() {
            return None;
        }
        let mut terms = Self::empty();
        for (l, kind) in &pieces.polys {
            match kind {
                PolyKind::Unit => terms.add_linear(*l, E::one())?,
                PolyKind::Linear(c) => terms.add_linear(*l, *c)?,
                PolyKind::Quadratic { c1, c2 } => {
                    terms.add_linear(*l, *c1)?;
                    terms.add_product(*l, *l, *c2)?;
                }
                _ => return None,
            }
        }
        for b in &pieces.bilinear {
            terms.add_product(b.a, b.b, b.c)?;
        }
        for w in &pieces.weighted {
            for &(l, c) in &w.lanes {
                terms.add_product(w.weight, l, c)?;
            }
            terms.add_linear(w.weight, w.constant)?;
        }
        Some(terms)
    }

    /// The monomial form, when every piece expands into monomials of at most
    /// [`MAX_FACTORS`] lanes and degree [`MAX_DEGREE`].
    fn monomials(pieces: &Pieces) -> Option<Self> {
        let mut terms = Self::empty();
        let mut monomials: Vec<Monomial> = Vec::new();
        let mut push = |terms: &mut Self, lanes: &[usize], c: E| -> Option<()> {
            if c.is_zero() {
                return Some(());
            }
            // Repeated lanes become powers.
            let mut factors: Vec<(usize, usize)> = Vec::new();
            for &l in lanes {
                match factors.iter_mut().find(|(lane, _)| *lane == l) {
                    Some((_, e)) => *e += 1,
                    None => factors.push((l, 1)),
                }
            }
            if lanes.is_empty() || lanes.len() > MAX_DEGREE || factors.len() > MAX_FACTORS {
                return None;
            }
            let slot = terms.slot(c)?;
            monomials.push(Monomial { factors, coefficient: None, slot });
            Some(())
        };
        for (l, kind) in &pieces.polys {
            if matches!(kind, PolyKind::Depressed { .. }) {
                return None;
            }
            for (e, &c) in kind.coefficients().iter().enumerate().skip(1) {
                push(&mut terms, &vec![*l; e], c)?;
            }
        }
        for b in &pieces.bilinear {
            push(&mut terms, &[b.a, b.b], b.c)?;
        }
        for w in &pieces.weighted {
            for &(l, c) in &w.lanes {
                push(&mut terms, &[w.weight, l], c)?;
            }
            push(&mut terms, &[w.weight], w.constant)?;
        }
        for (gamma, term) in &pieces.generic {
            match term {
                Term::RangeW1([b5, b4, b3, b2]) => {
                    push(&mut terms, &[*b5, *b4], *gamma)?;
                    push(&mut terms, &[*b5, *b3, *b2], *gamma)?;
                }
                Term::Product(lanes) => push(&mut terms, lanes, *gamma)?,
                _ => return None,
            }
        }
        if monomials.is_empty() {
            return None;
        }
        // Slots that hold single lane powers only (the coefficients of the lane polynomials)
        // merge into one slot of coefficient one, their coefficients moving inside: one
        // product per power and point instead of one weighted reduction per slot and point.
        let single: Vec<bool> = (0..terms.slots.len())
            .map(|s| monomials.iter().filter(|m| m.slot == s).all(|m| m.factors.len() == 1))
            .collect();
        if single.iter().filter(|&&s| s).count() >= 2 {
            let mut kept = Vec::new();
            let mut index = vec![0; terms.slots.len()];
            for (s, &c) in terms.slots.iter().enumerate() {
                if !single[s] {
                    index[s] = kept.len();
                    kept.push(c);
                }
            }
            let merged = kept.len();
            kept.push(E::one());
            for m in &mut monomials {
                if single[m.slot] {
                    let c = terms.slots[m.slot];
                    m.coefficient = (c != E::one()).then_some(c);
                    m.slot = merged;
                } else {
                    m.slot = index[m.slot];
                }
            }
            terms.products_per_slot = vec![0; kept.len()];
            terms.slots = kept;
        }
        terms.high = Some(HighTerms {
            monomials,
            slots: terms.slots.len(),
        });
        Some(terms)
    }

    /// The slot of coefficient `c`.
    fn slot(&mut self, c: E) -> Option<usize> {
        if let Some(slot) = self.slots.iter().position(|&s| s == c) {
            return Some(slot);
        }
        (self.slots.len() < caps().1).then(|| {
            self.slots.push(c);
            self.products_per_slot.push(0);
            self.slots.len() - 1
        })
    }

    fn add_linear(&mut self, lane: usize, c: E) -> Option<()> {
        if !c.is_zero() {
            let slot = self.slot(c)?;
            self.linear.push((lane, slot));
        }
        Some(())
    }

    fn add_product(&mut self, a: usize, b: usize, c: E) -> Option<()> {
        if !c.is_zero() {
            let slot = self.slot(c)?;
            let count = &mut self.products_per_slot[slot];
            *count += 1;
            if *count > MAX_PRODUCTS_PER_SLOT {
                return None;
            }
            self.products.push((a, b, slot));
        }
        Some(())
    }
}

/// The dense table of a group in lane-major planes (`rows` words per coordinate plane, the
/// two planes of lane `l` at word offsets `2 l rows` and `(2 l + 1) rows`). Its buffers come
/// from the plane pool of the group's arena and return there when a fold replaces them, so
/// the groups of a stage reuse mapped pages instead of faulting fresh ones.
pub(super) struct Tail {
    terms: TailTerms,
    width: usize,
    rows: usize,
    table: AlignedVec<u64>,
    arena: SharedArena,
    /// The challenge whose fold the next pass performs.
    pending: Option<E>,
    /// The flat slot sums of the next round, when the pass that built the table evaluated
    /// them, with the `full` flag they were computed under.
    cached: Option<(bool, Vec<E>)>,
}

impl Tail {
    /// Build the table by folding the rows of `source` (twice the table's rows) by
    /// `challenge`. With `eval = Some((eq, full))`, the next round is evaluated on the new
    /// pairs in the same pass, block by block while they are in cache, and
    /// [`Self::message`] then returns the cached sums (it must be called with the same
    /// `full`); `None` builds the table of the last variable, after which no round follows.
    pub fn from_rows(
        terms: TailTerms,
        challenge: E,
        source: &dyn RowSource,
        eval: Option<(&GruenSplitEq<E>, bool)>,
        arena: SharedArena,
    ) -> Self {
        let (width, rows) = (source.width(), source.rows() / 2);
        let buffer = take_planes(&arena, 2 * width * rows);
        let fill = Fill::Computed(Some(challenge), source);
        let input: [Planes<'_>; 0] = [];
        let (table, cached) = match eval {
            Some((eq, full)) => {
                let (first, second) = eq.remaining_eq_tables();
                let first_planes = KPlanes::from_fn(first.len(), |i| first[i]);
                let eval = EvalAny {
                    summand: terms.summand(),
                    first: first_planes.planes(),
                    second,
                    full,
                };
                let (table, total) = fused::fused_pass_into_with::<Columns>(buffer, width, rows, &input, fill, Some(eval));
                (table, Some((full, total)))
            }
            None => (fused::fused_pass_into_with::<Columns>(buffer, width, rows, &input, fill, None).0, None),
        };
        Self { terms, width, rows, table, arena, pending: None, cached }
    }

    fn lanes(&self) -> Vec<Planes<'_>> {
        (0..self.width)
            .map(|l| {
                let (c0, c1) = (2 * l * self.rows, (2 * l + 1) * self.rows);
                (&self.table[c0..c0 + self.rows], &self.table[c1..c1 + self.rows])
            })
            .collect()
    }

    /// The value of lane `l` at row `i`.
    pub fn get(&self, l: usize, i: usize) -> E {
        agg_simd::scalar::get(self.lanes()[l], i)
    }

    /// Fold by the pending challenge (if any) and evaluate the slot sums of the pairs on the
    /// new level with `eval`.
    fn pass(&mut self, eval: Option<EvalAny<'_>>) -> Vec<E> {
        let challenge = self.pending.take();
        let Some(r) = challenge else {
            let input = self.lanes();
            return fused::fused_pass_with(None, &input, None::<Fill<'_, Columns>>, eval);
        };
        let rows_out = self.rows / 2;
        // The fold writes every word of the new table before reading it, so a pooled buffer
        // of any earlier contents serves; the replaced table returns to the pool.
        let buffer = take_planes(&self.arena, 2 * self.width * rows_out);
        let (out, total) = {
            let input = self.lanes();
            fused::fused_pass_into_with(buffer, self.width, rows_out, &input, Fill::<Columns>::Fold(r, &input), eval)
        };
        give_planes(&self.arena, std::mem::replace(&mut self.table, out));
        self.rows = rows_out;
        total
    }

    /// [`TailTerms::message`] of the next round: the inner polynomial of the summand over
    /// the pairs of the current table, weighted by the remaining equality tables of `eq` and
    /// shifted by the cell constant `constant`. `q(1)` is computed only when `full`.
    pub fn message(&mut self, eq: &GruenSplitEq<E>, full: bool, constant: E) -> Vec<E> {
        let total = match self.cached.take() {
            Some((cached_full, total)) => {
                // Sums cached under another `full` would lack (or misplace) `q(1)` and give a
                // wrong message, so both checks hold in release builds too (they are free).
                assert!(self.pending.is_none(), "cached sums belong to the current table");
                assert_eq!(cached_full, full, "cached sums of the same evaluation");
                total
            }
            None => {
                let (first, second) = eq.remaining_eq_tables();
                let first_planes = KPlanes::from_fn(first.len(), |i| first[i]);
                // The pass borrows the buffers mutably: the small term lists move out for it.
                let terms = std::mem::replace(&mut self.terms, TailTerms::empty());
                let eval = EvalAny {
                    summand: terms.summand(),
                    first: first_planes.planes(),
                    second,
                    full,
                };
                let total = self.pass(Some(eval));
                self.terms = terms;
                total
            }
        };
        self.terms.message(&total, constant)
    }

    /// Bind a variable to `challenge`: the fold happens in the next [`Self::message`], or
    /// right away for the last variable.
    pub fn bind(&mut self, challenge: E, last: bool) {
        debug_assert!(self.pending.is_none(), "one bind per round");
        self.pending = Some(challenge);
        if last {
            self.pass(None);
        }
    }
}

impl Drop for Tail {
    /// The last table returns to the pool for the groups that are still running.
    fn drop(&mut self) {
        let table = std::mem::replace(&mut self.table, AlignedVec::fill_par(0, 1, |_| 0));
        give_planes(&self.arena, table);
    }
}
