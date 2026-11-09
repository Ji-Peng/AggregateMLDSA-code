//! The dense rounds of a generic [`super::Group`] on lane-major `K` planes.
//!
//! From the third bind on, a generic group holds every lane as a table of `K` values. When
//! all of the group's pieces are at most quadratic in the lanes, the summand is the bilinear
//! form with coefficient slots of `agg_simd::fused`, and a round is one fused pass: fold
//! every lane by the previous challenge into the spare buffer and evaluate the round on the
//! just-written pairs. This replaces the row-major table with its separate evaluation and
//! fold passes. The round before the table (level 2) runs on the same kernels: the lanes'
//! compact states are gathered block by block and evaluated, and the third bind folds the
//! gathered rows into the first table while it evaluates the next round. The round
//! polynomial is unique, so the messages equal those of the scalar dense rounds byte for
//! byte.

use agg_simd::buf::{AlignedVec, KPlanes};
use agg_simd::fused::{self, Eval, Fill, RowSource, Terms, MAX_LANES, MAX_SLOTS};
use agg_simd::k::Planes;
use agg_simd::pattern::Columns;
use akita_algebra::split_eq::GruenSplitEq;
use jolt_field::{One, Zero};

use super::plan::{Pieces, PolyKind};
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

/// The dense pieces of a group as the term lists of the fused kernels.
pub(super) struct TailTerms {
    linear: Vec<(usize, usize)>,
    products: Vec<(usize, usize, usize)>,
    /// The coefficient of every slot.
    slots: Vec<E>,
    /// Products accumulated into each slot.
    products_per_slot: Vec<usize>,
}

impl TailTerms {
    /// The coefficient slots of the summand.
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// `[q(0), q(1), q_2]` from the slot sums of a pass and the cell constant.
    fn message(&self, total: &[[E; 3]], constant: E) -> [E; 3] {
        let mut q = [constant, constant, E::zero()];
        for (&c, s) in self.slots.iter().zip(total) {
            for k in 0..3 {
                q[k] += c * s[k];
            }
        }
        q
    }

    /// `[q(0), q(1), q_2]` of a round over the pairs of `source`'s rows, which are
    /// evaluated block by block as they are computed (no table is written); the remaining
    /// equality tables of `eq` weight the pairs, as in [`Tail::message`].
    pub fn rows_message(&self, source: &dyn RowSource, eq: &GruenSplitEq<E>, full: bool, constant: E) -> [E; 3] {
        let (first, second) = eq.remaining_eq_tables();
        let first_planes = KPlanes::from_fn(first.len(), |i| first[i]);
        let eval = Eval {
            terms: self.as_terms(),
            first: first_planes.planes(),
            second,
            full,
        };
        let total = fused::fused_pass::<Columns>(None, &[], Some(Fill::Computed(None, source)), Some(eval));
        self.message(&total, constant)
    }

    /// The term lists as the fused kernels read them.
    fn as_terms(&self) -> Terms<'_> {
        Terms {
            linear: &self.linear,
            products: &self.products,
            slots: self.slots.len(),
        }
    }
}

impl TailTerms {
    /// The term lists of `pieces` over `lanes` lanes, or `None` when a piece is not at most
    /// quadratic (or is a residual generic term), or when the lanes or distinct coefficients
    /// exceed the kernels' limits.
    pub fn compile(pieces: &Pieces, lanes: usize) -> Option<Self> {
        if lanes > caps().0 || !pieces.generic.is_empty() {
            return None;
        }
        let mut terms = Self {
            linear: Vec::new(),
            products: Vec::new(),
            slots: Vec::new(),
            products_per_slot: Vec::new(),
        };
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
/// two planes of lane `l` at word offsets `2 l rows` and `(2 l + 1) rows`), with the buffer
/// the next fold writes into.
pub(super) struct Tail {
    terms: TailTerms,
    width: usize,
    rows: usize,
    table: AlignedVec<u64>,
    spare: Option<AlignedVec<u64>>,
    /// The challenge whose fold the next pass performs.
    pending: Option<E>,
    /// The slot sums of the next round, when the pass that built the table evaluated them,
    /// with the `full` flag they were computed under.
    cached: Option<(bool, Vec<[E; 3]>)>,
}

impl Tail {
    /// Build the table by folding the rows of `source` (twice the table's rows) by
    /// `challenge`, and evaluate the next round on the new pairs in the same pass, block by
    /// block while they are in cache. [`Self::message`] then returns the cached sums when
    /// it is called with the same `full`.
    pub fn from_rows(terms: TailTerms, challenge: E, source: &dyn RowSource, eq: &GruenSplitEq<E>, full: bool) -> Self {
        let (width, rows) = (source.width(), source.rows() / 2);
        let buffer = AlignedVec::<u64>::uninit(2 * width * rows);
        let (first, second) = eq.remaining_eq_tables();
        let first_planes = KPlanes::from_fn(first.len(), |i| first[i]);
        let eval = Eval {
            terms: terms.as_terms(),
            first: first_planes.planes(),
            second,
            full,
        };
        let input: [Planes<'_>; 0] = [];
        let (table, total) =
            fused::fused_pass_into::<Columns>(buffer, width, rows, &input, Fill::Computed(Some(challenge), source), Some(eval));
        Self { terms, width, rows, table, spare: None, pending: None, cached: Some((full, total)) }
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
    fn pass(&mut self, eval: Option<Eval<'_>>) -> Vec<[E; 3]> {
        let challenge = self.pending.take();
        let Some(r) = challenge else {
            let input = self.lanes();
            return fused::fused_pass(None, &input, None::<Fill<'_, Columns>>, eval);
        };
        let rows_out = self.rows / 2;
        // The first fold allocates only its output size and writes every word before
        // reading it; subsequent folds reuse an initialized buffer.
        let (out, total) = if let Some(mut out) = self.spare.take() {
            let input = self.lanes();
            let total = fused::fused_pass(Some((&mut out[..], self.width, rows_out)), &input, Some(Fill::<Columns>::Fold(r, &input)), eval);
            (out, total)
        } else {
            let input = self.lanes();
            fused::fold_new(r, &input, rows_out, eval)
        };
        self.spare = Some(std::mem::replace(&mut self.table, out));
        self.rows = rows_out;
        total
    }

    /// `[q(0), q(1), q_2]` of the next round: the inner polynomial of the summand over the
    /// pairs of the current table, weighted by the remaining equality tables of `eq` and
    /// shifted by the cell constant `constant`. `q(1)` is computed only when `full`.
    pub fn message(&mut self, eq: &GruenSplitEq<E>, full: bool, constant: E) -> [E; 3] {
        let total = match self.cached.take() {
            Some((cached_full, total)) => {
                debug_assert!(self.pending.is_none(), "cached sums belong to the current table");
                debug_assert_eq!(cached_full, full, "cached sums of the same evaluation");
                total
            }
            None => {
                let (first, second) = eq.remaining_eq_tables();
                let first_planes = KPlanes::from_fn(first.len(), |i| first[i]);
                // The pass borrows the buffers mutably: copy the small term lists out.
                let (linear, products) = (self.terms.linear.clone(), self.terms.products.clone());
                let eval = Eval {
                    terms: Terms {
                        linear: &linear,
                        products: &products,
                        slots: self.terms.slots.len(),
                    },
                    first: first_planes.planes(),
                    second,
                    full,
                };
                self.pass(Some(eval))
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
