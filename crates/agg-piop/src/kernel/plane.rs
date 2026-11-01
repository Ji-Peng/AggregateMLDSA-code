//! Dedicated sum-check groups of the LogUp-GKR sessions on lane-major `K` planes
//! (`docs/lookup.md`, `docs/piop.md`). Prover only.
//!
//! The round polynomial is unique, so any exact computation of it gives the messages of
//! the generic [`super::Group`] byte for byte. A group's summand is a bilinear form with
//! *coefficient slots*, `f = c_0 + sum_slot c_slot (sum_(l, slot) L_l + sum_(i, j, slot)
//! L_i L_j)`, over lanes that live as canonical coordinate planes (`agg_simd::fused`):
//!
//! - [`LayerGroup`]: the stored layer of every active tree is already a plane table (its
//!   halves `P_0, P_1, Q_0, Q_1` are plane slices), so round 0 evaluates the layer in place
//!   and the first fold writes the group's own table.
//! - [`LeafGroup`]: integer leaves (the stack bottom `v_0, v_1`, or `S_c, M_c` of the layer
//!   above it computed on the fly from four quarters of the stacked column). Rounds 0 and 1
//!   come from one Boolean-pattern pass over the leaves (`agg_simd::pattern`); the round-2
//!   pass materialises the lanes folded by the two bound challenges as planes and evaluates
//!   round 2 on them.
//!
//! - [`WeightIntGroup`] (stage 3's `w_A z`, no equality factor): round 0 sums `w g` with
//!   unreduced integer products from the weight planes and the integer combination `g`; the
//!   first fold writes `w` and `g` as a width-2 table. The integer lanes' terminal values
//!   come from the multilinear evaluation of their columns at the bound point
//!   (`agg_simd::eval::column_at`), so they are never carried as `K` lanes.
//! - [`LinearIntGroup`] (stage 3's lookup-stack and bit reductions): up to four integer columns read
//!   linearly with `K` coefficients. The prefix pass sums each column by pattern; the round-2
//!   pass materialises the single combined lane `g = sum_l c_l f_l(rho_0, rho_1, .)`.
//!
//! From the first plane table on, a round is one fused pass in blocks of 512 output rows:
//! fold every lane by the bound challenge into the spare buffer, then evaluate the next
//! round on the just-written pairs with the split equality weights (inner sums over `first`
//! unreduced, times `second` per block). The last rounds run the scalar paths of the same
//! kernels. Table buffers come from a [`Pool`] shared by the sessions of one prover run.

use akita_algebra::split_eq::GruenSplitEq;
use jolt_field::{Field, One, Zero};
use jolt_poly::UnivariatePoly;
use rayon::prelude::*;
use std::mem::MaybeUninit;

use agg_simd::buf::{AlignedVec, KPlanes, Pool};
use agg_simd::eval::{column_at, split_eq_tables, Mapped};
use agg_simd::fused::{self, Eval, Fill, MixedLane, RowSource, Terms};
use agg_simd::lookup::UpperRows;
use agg_simd::k::Planes;
use agg_simd::pattern::{self, Bottom, ByteColumns, Columns, Leaves, PrefixSums, Symmetric};

use super::{compress, evaluate_compressed};
use crate::OpeningField as E;

/// Message degree of a session.
const DEGREE: usize = 3;
/// Variables bound by the Boolean-pattern prefix of a leaf group.
const PREFIX: usize = 2;

/// The level-0 data of a plane group.
trait PlaneSource: Sync {
    /// The integer leaves of a prefix group.
    type Leaves: Leaves;
    /// Lanes of the plane table.
    fn width(&self) -> usize;
    /// `log2` of the level-0 rows.
    fn vars(&self) -> usize;
    /// Rounds served by the pattern prefix (0: the level-0 lanes are planes).
    fn prefix(&self) -> usize;
    /// The constant term `c_0`.
    fn constant(&self) -> E;
    /// The slot coefficients.
    fn slots(&self) -> &[E];
    /// `(lane, slot)`.
    fn linear(&self) -> &[(usize, usize)];
    /// `(lane_i, lane_j, slot)`.
    fn products(&self) -> &[(usize, usize, usize)];
    /// The level-0 lanes as planes (prefix 0).
    fn level0(&self) -> Vec<Planes<'_>>;
    /// The pattern sums with the prefix weights, and the slot of every linear and product
    /// form of the shape (prefix 2).
    fn prefix_sums(&self, first: &[E], second: &[E]) -> (PrefixSums, Vec<usize>, Vec<usize>);
    /// The leaves the round-`prefix` pass materialises (prefix 2).
    fn leaves(&self) -> Option<&Self::Leaves>;
    /// The fill of the round-`prefix` pass: every lane folded by the pattern weights `e`,
    /// unless the source combines its lanes into one (`scratch` holds the combined weights).
    fn pattern_fill<'s>(&'s self, e: [E; 4], scratch: &'s mut Vec<[E; 4]>) -> Fill<'s, Self::Leaves> {
        let _ = scratch;
        Fill::Pattern(e, self.leaves().expect("prefix leaves"))
    }
    /// The level-0 lanes of mixed kinds for the first fold, when they are not all planes.
    fn mixed_lanes(&self) -> Option<Vec<MixedLane<'_>>> {
        None
    }
    /// `[q(0), q(1), q_2]` of round 0 from the level-0 data when it is not all planes.
    fn first_round(&self, full: bool) -> Option<[E; 3]> {
        let _ = full;
        None
    }
    /// The level-0 lanes when they are computed block by block instead of stored.
    fn computed(&self) -> Option<&dyn RowSource> {
        None
    }
    /// The group's terminal lane values from the fully folded row at the bound `point`.
    fn terminal(&self, row: &[E], point: &[E]) -> Vec<E>;
}

/// A lane-major plane table in a pooled buffer.
struct Table {
    buf: AlignedVec<u64>,
    width: usize,
    rows: usize,
}

impl Table {
    fn lane(&self, l: usize) -> Planes<'_> {
        let (c0, c1) = (2 * l * self.rows, (2 * l + 1) * self.rows);
        (&self.buf[c0..c0 + self.rows], &self.buf[c1..c1 + self.rows])
    }

    fn lanes(&self) -> Vec<Planes<'_>> {
        (0..self.width).map(|l| self.lane(l)).collect()
    }

    fn get(&self, l: usize, i: usize) -> E {
        agg_simd::scalar::get(self.lane(l), i)
    }
}

/// Where the next pass reads its rows.
enum Stage {
    /// Rounds `< prefix`: messages from the pattern sums.
    Prefix,
    /// The level-0 planes of the source.
    Level0,
    /// The group's own plane table.
    Table,
}

/// The round driver shared by the plane groups.
struct Engine<S: PlaneSource> {
    source: S,
    /// The equality factor, or `None` for a plain sum over the domain.
    eq: Option<GruenSplitEq<E>>,
    /// The equality point (the prefix rounds read its free coordinates).
    eq_point: Vec<E>,
    vars: usize,
    bound: usize,
    claim: E,
    pending: Vec<E>,
    /// The bound challenges.
    rho: Vec<E>,
    /// The challenge whose fold the next pass performs.
    unfolded: Option<E>,
    stage: Stage,
    /// The pattern sums (prefix groups, computed at round 0) with the forms' slots.
    sums: Option<(PrefixSums, Vec<usize>, Vec<usize>)>,
    table: Option<Table>,
    /// Buffers for the next table (the full-size one and the half-size one, swapped with the
    /// table's buffer round by round). Their contents are unspecified until a pass fills
    /// them.
    spare: Vec<AlignedVec<MaybeUninit<u64>>>,
}

impl<S: PlaneSource> Engine<S> {
    /// Words of the plane table at its first materialisation (the first fold for a
    /// prefix-free source).
    fn table_words(source: &S) -> usize {
        (2 * source.width()) << (source.vars() - source.prefix().max(1))
    }

    fn new(source: S, eq_point: Option<&[E]>, vars: usize, pool: &mut Pool) -> Option<Self> {
        if vars == 0 || source.vars() != vars || vars < source.prefix() || eq_point.is_some_and(|p| p.len() != vars) {
            return None;
        }
        if eq_point.is_none() && source.prefix() > 0 {
            return None;
        }
        let eq = match eq_point {
            Some(point) => Some(GruenSplitEq::new(point).ok()?),
            None => None,
        };
        let words = Self::table_words(&source);
        let stage = if source.prefix() == 0 { Stage::Level0 } else { Stage::Prefix };
        let spare = vec![pool.take(words), pool.take((words / 2).max(2 * source.width()))];
        Some(Self {
            source,
            eq,
            eq_point: eq_point.map_or_else(Vec::new, <[E]>::to_vec),
            vars,
            bound: 0,
            claim: E::zero(),
            pending: Vec::new(),
            rho: Vec::new(),
            unfolded: None,
            stage,
            sums: None,
            table: None,
            spare,
        })
    }

    /// Return the table buffers to the pool.
    fn recycle(self, pool: &mut Pool) {
        if let Some(table) = self.table {
            pool.give(table.buf.into_uninit());
        }
        for buffer in self.spare {
            pool.give(buffer);
        }
    }

    /// The inner polynomial `[q(0), q(1), q_2]` of a prefix round from the pattern sums:
    /// with `u_i(a; X) = prod_{k < i} eq(rho_k, a_k) eq(a_i, X)` and the free prefix
    /// variables' equality weights `m_i(a) = prod_{k > i} eq(r_k, a_k)`,
    /// `q_i(X) = c_0 sum w + sum_lin c sum_a u_i(a; X) m_i(a) A[a] + sum_prod c sum_{a, b :
    /// a_{>i} = b_{>i}} u_i(a; X) u_i(b; X) m_i(a) A[a, b]`.
    fn prefix_message(&self, round: usize) -> [E; 3] {
        let (sums, lin_slots, prod_slots) = self.sums.as_ref().expect("pattern sums");
        let slots = self.source.slots();
        let eq_point = &self.eq_point;
        // u(a; X) = alpha + beta X, and m(a).
        let line = |a: usize| -> (E, E) {
            let bit = |k: usize| (a >> k) & 1;
            let mut scale = E::one();
            for k in 0..round {
                let r = self.rho[k];
                scale *= if bit(k) == 1 { r } else { E::one() - r };
            }
            let (alpha, beta) = if bit(round) == 0 { (scale, -scale) } else { (E::zero(), scale) };
            (alpha, beta)
        };
        let m = |a: usize| -> E {
            let mut w = E::one();
            for k in round + 1..PREFIX {
                let r = eq_point[k];
                w *= if (a >> k) & 1 == 1 { r } else { E::one() - r };
            }
            w
        };
        let same_free = |a: usize, b: usize| (a >> (round + 1)) == (b >> (round + 1));
        let total = self.source.constant() * sums.total;
        let mut q = [total, total, E::zero()];
        let lines: [(E, E); 4] = std::array::from_fn(line);
        let ms: [E; 4] = std::array::from_fn(m);
        for (form, &slot) in lin_slots.iter().enumerate() {
            let c = slots[slot];
            for a in 0..4 {
                let (alpha, beta) = lines[a];
                let v = c * ms[a] * sums.linear[form][a];
                q[0] += alpha * v;
                q[1] += (alpha + beta) * v;
            }
        }
        for (form, &slot) in prod_slots.iter().enumerate() {
            let c = slots[slot];
            for a in 0..4 {
                for b in 0..4 {
                    if !same_free(a, b) {
                        continue;
                    }
                    let (aa, ba) = lines[a];
                    let (ab, bb) = lines[b];
                    let v = c * ms[a] * sums.products[form][4 * a + b];
                    q[0] += aa * ab * v;
                    q[1] += (aa + ba) * (ab + bb) * v;
                    q[2] += ba * bb * v;
                }
            }
        }
        q
    }

    /// The pattern weights `e[a] = eq(rho_0, a_0) eq(rho_1, a_1)`.
    fn pattern_weights(&self) -> [E; 4] {
        std::array::from_fn(|a| {
            (0..PREFIX).fold(E::one(), |w, k| {
                let r = self.rho[k];
                w * if (a >> k) & 1 == 1 { r } else { E::one() - r }
            })
        })
    }

    /// One pass: fold (or materialise) into the spare buffer when a challenge is pending,
    /// then evaluate `[q(0), q(1), q_2]` of the round on the new level (`q(1)` only when
    /// `full`).
    #[allow(clippy::too_many_lines)]
    fn pass(&mut self, evaluate: bool, full: bool) -> [E; 3] {
        let width = self.source.width();
        let challenge = self.unfolded.take();
        let materialize = matches!(self.stage, Stage::Prefix);
        let rows_in = match self.stage {
            Stage::Prefix => 1_usize << self.vars,
            Stage::Level0 => 1_usize << self.vars,
            Stage::Table => self.table.as_ref().expect("a table").rows,
        };
        let rows_out = if materialize {
            rows_in >> PREFIX
        } else if challenge.is_some() {
            rows_in / 2
        } else {
            rows_in
        };
        let fill = materialize || challenge.is_some();
        if evaluate && !fill && matches!(self.stage, Stage::Level0) {
            if let Some(sums) = self.source.first_round(full) {
                return self.finish(vec![sums]);
            }
        }
        let (first, second): (Vec<E>, Vec<E>) = match &self.eq {
            Some(eq) if evaluate => {
                let (f, s) = eq.remaining_eq_tables();
                (f.to_vec(), s.to_vec())
            }
            _ => (Vec::new(), Vec::new()),
        };
        let first_planes = KPlanes::from_fn(first.len(), |i| first[i]);
        debug_assert!(!evaluate || first.is_empty() || first.len() * second.len() == (rows_out / 2).max(1));
        let terms = Terms {
            linear: self.source.linear(),
            products: self.source.products(),
            slots: self.source.slots().len(),
        };
        let level0 = matches!(self.stage, Stage::Level0).then(|| self.source.level0());
        let input: Vec<Planes<'_>> = match (&level0, &self.table) {
            (Some(lanes), _) => lanes.clone(),
            (None, Some(table)) if !materialize => table.lanes(),
            _ => Vec::new(),
        };
        // A pooled buffer holds unspecified contents. The pass below writes every word of
        // the output rows before reading them back for evaluation, and returns the buffer
        // initialized.
        let out_buf = fill.then(|| {
            let needed = 2 * width * rows_out;
            let best = self
                .spare
                .iter()
                .enumerate()
                .filter(|(_, b)| b.len() >= needed)
                .min_by_key(|(_, b)| b.len())
                .map(|(i, _)| i)
                .expect("a spare buffer of the table's size");
            self.spare.swap_remove(best)
        });
        let mixed = if level0.is_some() && challenge.is_some() { self.source.mixed_lanes() } else { None };
        let computed = if level0.is_some() { self.source.computed() } else { None };
        let mut scratch = Vec::new();
        let fill_op = if materialize {
            Some(self.source.pattern_fill(self.pattern_weights(), &mut scratch))
        } else if let Some(rows) = computed {
            // Round 0 evaluates the computed rows; round 1 folds them into the first table.
            Some(Fill::Computed(challenge, rows))
        } else {
            match (challenge, &mixed) {
                (Some(r), Some(lanes)) => Some(Fill::Mixed(r, lanes.as_slice())),
                (Some(r), None) => Some(Fill::Fold(r, input.as_slice())),
                (None, _) => None,
            }
        };
        let eval = evaluate.then(|| Eval {
            terms,
            first: first_planes.planes(),
            second: &second,
            full,
        });
        let (out_buf, total) = match out_buf {
            Some(buf) => {
                let fill_op = fill_op.expect("a fill for the output table");
                let (buf, total) = fused::fused_pass_into(buf, width, rows_out, &input, fill_op, eval);
                (Some(buf), total)
            }
            None => (None, fused::fused_pass(None, &input, fill_op, eval)),
        };
        drop(input);
        drop(level0);
        drop(mixed);
        if let Some(buf) = out_buf {
            if let Some(previous) = self.table.take() {
                self.spare.push(previous.buf.into_uninit());
            }
            self.table = Some(Table {
                buf,
                width,
                rows: rows_out,
            });
            self.stage = Stage::Table;
        }
        self.finish(total)
    }

    /// The inner polynomial from the slot sums: `c_0 + sum_slot c_slot S_slot` on the
    /// constant components, the slot sums alone on the quadratic one.
    fn finish(&self, total: Vec<[E; 3]>) -> [E; 3] {
        let constant = self.source.constant();
        let mut q = [constant, constant, E::zero()];
        for (&c, s) in self.source.slots().iter().zip(&total) {
            for k in 0..3 {
                q[k] += c * s[k];
            }
        }
        q
    }

    fn round_message(&mut self, round: usize) -> Vec<E> {
        assert_eq!(round, self.bound, "rounds in order");
        let full = round == 0 || self.eq.as_ref().is_some_and(|eq| !eq.can_recover_linear_q_term_from_claim());
        let [q0, at_one, q2] = if round < self.source.prefix() {
            if round == 0 {
                let eq = self.eq.as_ref().expect("a prefix group has an equality factor");
                let (first, second) = eq.remaining_eq_tables_after(PREFIX - 1).expect("prefix weights");
                self.sums = Some(self.source.prefix_sums(first, second));
            }
            self.prefix_message(round)
        } else {
            self.pass(true, full)
        };
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
        self.rho.push(challenge);
        self.bound += 1;
        // The prefix challenges are consumed by the materialisation; later ones by a fold.
        if self.bound > self.source.prefix() {
            self.unfolded = Some(challenge);
        }
        if self.bound == self.vars {
            self.pass(false, false);
        }
    }

    fn terminal_row(&self) -> Vec<E> {
        assert_eq!(self.bound, self.vars, "all rounds bound");
        let table = self.table.as_ref().expect("the folded table");
        assert_eq!(table.rows, 1);
        (0..table.width).map(|l| table.get(l, 0)).collect()
    }

    fn terminal_values(&self) -> Vec<E> {
        self.source.terminal(&self.terminal_row(), &self.rho)
    }

    fn terminal_consistent(&self) -> bool {
        if self.bound != self.vars {
            return false;
        }
        let row = self.terminal_row();
        let slots = self.source.slots();
        let mut sum = self.source.constant();
        for &(l, s) in self.source.linear() {
            sum += slots[s] * row[l];
        }
        for &(i, j, s) in self.source.products() {
            sum += slots[s] * row[i] * row[j];
        }
        self.eq.as_ref().map_or(E::one(), GruenSplitEq::current_scalar) * sum == self.claim
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
    pub p: &'a KPlanes,
    /// The denominators of layer `s`.
    pub q: &'a KPlanes,
    /// The coefficient of the `p`-claim.
    pub gamma_p: E,
    /// The coefficient of the `q`-claim.
    pub gamma_q: E,
}

/// The level-0 lanes of a layer group.
enum LayerRows<'a> {
    /// Stored layers.
    Stored(Vec<LayerTree<'a>>),
    /// Layer `depth - 2` of one stack computed from its leaves, per challenge.
    Leaves(UpperRows<'a>),
}

struct LayerSource<'a> {
    rows: LayerRows<'a>,
    /// `gamma_p, gamma_q` per tree.
    slots: Vec<E>,
    products: Vec<(usize, usize, usize)>,
    vars: usize,
}

impl<'a> PlaneSource for LayerSource<'a> {
    type Leaves = Bottom<'a>;
    fn leaves(&self) -> Option<&Bottom<'a>> {
        None
    }
    fn width(&self) -> usize {
        match &self.rows {
            LayerRows::Stored(trees) => 4 * trees.len(),
            LayerRows::Leaves(rows) => rows.width(),
        }
    }
    fn vars(&self) -> usize {
        self.vars
    }
    fn prefix(&self) -> usize {
        0
    }
    fn computed(&self) -> Option<&dyn RowSource> {
        match &self.rows {
            LayerRows::Stored(_) => None,
            LayerRows::Leaves(rows) => Some(rows),
        }
    }
    fn constant(&self) -> E {
        E::zero()
    }
    fn slots(&self) -> &[E] {
        &self.slots
    }
    fn linear(&self) -> &[(usize, usize)] {
        &[]
    }
    fn products(&self) -> &[(usize, usize, usize)] {
        &self.products
    }
    fn level0(&self) -> Vec<Planes<'_>> {
        let half = 1_usize << self.vars;
        let LayerRows::Stored(trees) = &self.rows else {
            return Vec::new();
        };
        trees
            .iter()
            .flat_map(|t| {
                let (p0, p1) = t.p.planes();
                let (q0, q1) = t.q.planes();
                [
                    (&p0[..half], &p1[..half]),
                    (&p0[half..], &p1[half..]),
                    (&q0[..half], &q1[..half]),
                    (&q0[half..], &q1[half..]),
                ]
            })
            .collect()
    }
    fn prefix_sums(&self, _: &[E], _: &[E]) -> (PrefixSums, Vec<usize>, Vec<usize>) {
        unreachable!("a layer group has no prefix")
    }
    fn terminal(&self, row: &[E], _: &[E]) -> Vec<E> {
        row.to_vec()
    }
}

/// The layer group of one session (see the module documentation).
pub struct LayerGroup<'a>(Engine<LayerSource<'a>>);

impl<'a> LayerGroup<'a> {
    /// A group over `trees` (every layer of `2^(vars + 1)` cells, `vars = eq_point.len()`
    /// rounds) with table buffers from `pool`. `None` when the layers have different or
    /// wrong lengths or there are no trees.
    #[must_use]
    pub fn new(trees: Vec<LayerTree<'a>>, eq_point: &[E], pool: &mut Pool) -> Option<Self> {
        let vars = eq_point.len();
        let half = 1_usize << vars;
        if trees.is_empty() || trees.iter().any(|t| t.p.len() != 2 * half || t.q.len() != 2 * half) {
            return None;
        }
        let slots = trees.iter().flat_map(|t| [t.gamma_p, t.gamma_q]).collect();
        let products = Self::products(trees.len());
        let source = LayerSource {
            rows: LayerRows::Stored(trees),
            slots,
            products,
            vars,
        };
        Engine::new(source, Some(eq_point), vars, pool).map(Self)
    }

    /// The group of layer `depth - 2` of one stack whose layer is not stored: its lanes are
    /// computed from the leaves (`agg_simd::lookup::UpperRows`) in round 0 and in the first
    /// fold. `gammas[t] = (gamma_p, gamma_q)` of the tree of `rows`' challenge `t`. `None`
    /// when the rows do not match `eq_point` or the coefficient count is wrong.
    #[must_use]
    pub fn from_leaves(rows: UpperRows<'a>, gammas: &[(E, E)], eq_point: &[E], pool: &mut Pool) -> Option<Self> {
        let vars = eq_point.len();
        if gammas.len() != rows.trees() || rows.rows() != 1 << vars {
            return None;
        }
        let slots = gammas.iter().flat_map(|&(gp, gq)| [gp, gq]).collect();
        let source = LayerSource {
            products: Self::products(gammas.len()),
            rows: LayerRows::Leaves(rows),
            slots,
            vars,
        };
        Engine::new(source, Some(eq_point), vars, pool).map(Self)
    }

    /// `gamma_p (P_0 Q_1 + P_1 Q_0) + gamma_q Q_0 Q_1` per tree.
    fn products(trees: usize) -> Vec<(usize, usize, usize)> {
        (0..trees)
            .flat_map(|t| {
                let (p0, p1, q0, q1) = (4 * t, 4 * t + 1, 4 * t + 2, 4 * t + 3);
                [(p0, q1, 2 * t), (p1, q0, 2 * t), (q0, q1, 2 * t + 1)]
            })
            .collect()
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

    /// Return the table buffers to the pool.
    pub fn recycle(self, pool: &mut Pool) {
        self.0.recycle(pool);
    }
}

// ---------------------------------------------------------------------------------------
// The leaf groups.
// ---------------------------------------------------------------------------------------

/// Largest magnitude of a bottom leaf (`|v_0 + v_1| < 2^21`, `|v_0 v_1| < 2^40`, one limb).
pub const BOTTOM_BOUND: i64 = 1 << 20;

struct LeafSource<L: Leaves> {
    leaves: L,
    vars: usize,
    constant: E,
    slots: Vec<E>,
    linear: Vec<(usize, usize)>,
    products: Vec<(usize, usize, usize)>,
    /// The slot of every linear form of the shape, then of every product form.
    lin_forms: Vec<usize>,
    prod_forms: Vec<usize>,
}

impl<L: Leaves> PlaneSource for LeafSource<L> {
    type Leaves = L;
    fn leaves(&self) -> Option<&L> {
        Some(&self.leaves)
    }
    fn width(&self) -> usize {
        L::LANES
    }
    fn vars(&self) -> usize {
        self.vars
    }
    fn prefix(&self) -> usize {
        PREFIX
    }
    fn constant(&self) -> E {
        self.constant
    }
    fn slots(&self) -> &[E] {
        &self.slots
    }
    fn linear(&self) -> &[(usize, usize)] {
        &self.linear
    }
    fn products(&self) -> &[(usize, usize, usize)] {
        &self.products
    }
    fn level0(&self) -> Vec<Planes<'_>> {
        unreachable!("leaf lanes are integers")
    }
    fn prefix_sums(&self, first: &[E], second: &[E]) -> (PrefixSums, Vec<usize>, Vec<usize>) {
        (
            pattern::prefix_sums(&self.leaves, first, second),
            self.lin_forms.clone(),
            self.prod_forms.clone(),
        )
    }
    fn terminal(&self, row: &[E], _: &[E]) -> Vec<E> {
        row.to_vec()
    }
}

enum LeafEngine<'a> {
    Bottom(Engine<LeafSource<Bottom<'a>>>),
    Symmetric(Engine<LeafSource<Symmetric<'a>>>),
}

/// A session whose lanes are integer leaves (see the module documentation).
pub struct LeafGroup<'a>(LeafEngine<'a>);

impl<'a> LeafGroup<'a> {
    /// The stack bottom: lanes `v_0, v_1` of `2^vars` cells (`vars = eq_point.len()`),
    /// summand `c + c_s (v_0 + v_1) + c_m v_0 v_1`. `None` when `vars < 2`, a lane has the
    /// wrong length or a value reaches [`BOTTOM_BOUND`] (the caller then uses the generic
    /// group).
    #[must_use]
    pub fn bottom(v0: &'a [i32], v1: &'a [i32], [c, c_s, c_m]: [E; 3], eq_point: &[E], pool: &mut Pool) -> Option<Self> {
        let vars = eq_point.len();
        if vars < PREFIX || v0.len() != 1 << vars || v1.len() != 1 << vars {
            return None;
        }
        let out_of_range = [v0, v1]
            .iter()
            .any(|lane| lane.par_iter().with_min_len(1 << 16).any(|&v| i64::from(v).abs() >= BOTTOM_BOUND));
        if out_of_range {
            return None;
        }
        let source = LeafSource {
            leaves: Bottom { v0, v1 },
            vars,
            constant: c,
            slots: vec![c_s, c_m],
            linear: vec![(0, 0), (1, 0)],
            products: vec![(0, 1, 1)],
            lin_forms: vec![0],
            prod_forms: vec![1],
        };
        Engine::new(source, Some(eq_point), vars, pool).map(|e| Self(LeafEngine::Bottom(e)))
    }

    /// The layer above the bottom of a stack whose centred leaves satisfy `|v - centre| <=
    /// 2^15` (the caller checks the spread): lanes `S_0, S_1, M_0, M_1` of `2^vars` cells
    /// (`stacked` has `2^(vars + 2)`), summand `c + c_s (S_0 + S_1) + c_m (M_0 + M_1 + S_0
    /// S_1) + c_k (S_0 M_1 + S_1 M_0) + c_mm M_0 M_1`. `None` when `vars < 2` or the column
    /// has the wrong length.
    #[must_use]
    pub fn symmetric(
        stacked: &'a [i32],
        center: i32,
        [c, c_s, c_m, c_k, c_mm]: [E; 5],
        eq_point: &[E],
        pool: &mut Pool,
    ) -> Option<Self> {
        let vars = eq_point.len();
        if vars < PREFIX || stacked.len() != 1 << (vars + 2) {
            return None;
        }
        let source = LeafSource {
            leaves: Symmetric { stacked, center },
            vars,
            constant: c,
            slots: vec![c_s, c_m, c_k, c_mm],
            linear: vec![(0, 0), (1, 0), (2, 1), (3, 1)],
            products: vec![(0, 1, 1), (0, 3, 2), (1, 2, 2), (2, 3, 3)],
            lin_forms: vec![0, 1],
            prod_forms: vec![1, 2, 3],
        };
        Engine::new(source, Some(eq_point), vars, pool).map(|e| Self(LeafEngine::Symmetric(e)))
    }

    /// The current claim.
    #[must_use]
    pub fn claim(&self) -> E {
        match &self.0 {
            LeafEngine::Bottom(e) => e.claim,
            LeafEngine::Symmetric(e) => e.claim,
        }
    }

    /// The compressed message of `round` (`3` elements).
    ///
    /// # Panics
    /// When rounds are called out of order.
    pub fn round_message(&mut self, round: usize) -> Vec<E> {
        match &mut self.0 {
            LeafEngine::Bottom(e) => e.round_message(round),
            LeafEngine::Symmetric(e) => e.round_message(round),
        }
    }

    /// Bind the round's variable to `challenge`.
    ///
    /// # Panics
    /// When rounds are called out of order.
    pub fn bind(&mut self, round: usize, challenge: E) {
        match &mut self.0 {
            LeafEngine::Bottom(e) => e.bind(round, challenge),
            LeafEngine::Symmetric(e) => e.bind(round, challenge),
        }
    }

    /// The lane values at the bound point: `v_0, v_1`, or `S_0, S_1, M_0, M_1`.
    ///
    /// # Panics
    /// Before every round is bound.
    #[must_use]
    pub fn terminal_values(&self) -> Vec<E> {
        match &self.0 {
            LeafEngine::Bottom(e) => e.terminal_values(),
            LeafEngine::Symmetric(e) => e.terminal_values(),
        }
    }

    /// The self-check of [`LayerGroup::terminal_consistent`].
    #[must_use]
    pub fn terminal_consistent(&self) -> bool {
        match &self.0 {
            LeafEngine::Bottom(e) => e.terminal_consistent(),
            LeafEngine::Symmetric(e) => e.terminal_consistent(),
        }
    }

    /// Return the table buffers to the pool.
    pub fn recycle(self, pool: &mut Pool) {
        match self.0 {
            LeafEngine::Bottom(e) => e.recycle(pool),
            LeafEngine::Symmetric(e) => e.recycle(pool),
        }
    }
}

// ---------------------------------------------------------------------------------------
// Stage 3: a field weight times an integer combination, without an equality factor.
// ---------------------------------------------------------------------------------------

/// The multilinear evaluation of an integer or byte lane at `point` (low variable first).
fn int_lane_at(lane: &super::Source<'_>, point: &[E]) -> E {
    let (first, second) = split_eq_tables(point);
    match lane {
        super::Source::Ints(cells) => column_at(cells, &first, &second),
        super::Source::Bytes { cells, map } => column_at(&Mapped { cells, map }, &first, &second),
        super::Source::Field(_) | super::Source::Planes(_) | super::Source::PlaneSlice(_) => {
            unreachable!("integer lanes only")
        }
    }
}

struct WeightSource<'a> {
    /// The weight planes.
    weight: Planes<'a>,
    /// The integer lanes in the group's lane order (without the weight).
    lanes: Vec<super::Source<'a>>,
    /// `g(i) = sum_l c_l f_l(i) + constant` (exact: coefficients and cells are small),
    /// written for a block of cells (`fill(first, cells)`).
    combination: Box<dyn Fn(usize, &mut [i64]) + Sync + Send + 'a>,
    /// The group lane of the weight.
    weight_lane: usize,
    vars: usize,
    slots: [E; 1],
}

impl<'a> PlaneSource for WeightSource<'a> {
    type Leaves = Bottom<'a>;
    fn leaves(&self) -> Option<&Bottom<'a>> {
        None
    }
    fn width(&self) -> usize {
        2
    }
    fn vars(&self) -> usize {
        self.vars
    }
    fn prefix(&self) -> usize {
        0
    }
    fn constant(&self) -> E {
        E::zero()
    }
    fn slots(&self) -> &[E] {
        &self.slots
    }
    fn linear(&self) -> &[(usize, usize)] {
        &[]
    }
    fn products(&self) -> &[(usize, usize, usize)] {
        &[(0, 1, 0)]
    }
    fn level0(&self) -> Vec<Planes<'_>> {
        Vec::new()
    }
    fn prefix_sums(&self, _: &[E], _: &[E]) -> (PrefixSums, Vec<usize>, Vec<usize>) {
        unreachable!("a weight group has no prefix")
    }
    fn mixed_lanes(&self) -> Option<Vec<MixedLane<'_>>> {
        Some(vec![MixedLane::Planes(self.weight), MixedLane::Ints(&*self.combination)])
    }
    /// `sum_k w(2k) g(2k)`, `sum_k w(2k + 1) g(2k + 1)` and the sum of the products of the
    /// differences, in parallel chunks of the paired cells.
    fn first_round(&self, _: bool) -> Option<[E; 3]> {
        let rows = 1_usize << self.vars;
        let chunk = 1_usize << 14;
        let run = |start: usize| -> [E; 3] {
            let end = (start + chunk).min(rows);
            let mut g = vec![0_i64; end - start];
            (self.combination)(start, &mut g);
            agg_simd::dot::weight_round((&self.weight.0[start..end], &self.weight.1[start..end]), &g)
        };
        let add = |mut a: [E; 3], b: [E; 3]| {
            for (x, y) in a.iter_mut().zip(b) {
                *x += y;
            }
            a
        };
        Some(if rows > chunk {
            (0..rows).step_by(chunk).par_bridge().map(run).reduce(|| [E::zero(); 3], add)
        } else {
            run(0)
        })
    }
    fn terminal(&self, row: &[E], point: &[E]) -> Vec<E> {
        let mut out: Vec<E> = self.lanes.iter().map(|lane| int_lane_at(lane, point)).collect();
        out.insert(self.weight_lane, row[0]);
        out
    }
}

/// `gamma sum_x w(x) g(x)` for a weight lane `w` held as planes and the integer combination
/// `g = sum_l c_l f_l + c` of integer or byte lanes (no equality factor; stage 3's `w_A z`).
pub struct WeightIntGroup<'a>(Engine<WeightSource<'a>>);

impl<'a> WeightIntGroup<'a> {
    /// `lanes` are the group's lanes in order: the weight at `weight_lane` (planes) and
    /// integer or byte lanes, with `coefficients[l]` the coefficient of lane `l` in `g`
    /// (ignored at the weight). `None` on lanes of other lengths than `2^rounds`, a weight
    /// that is not held as planes, a field lane besides the weight, or coefficients above
    /// `2^20` (exactness of `g`).
    #[must_use]
    pub fn new(
        lanes: &[super::Source<'a>],
        weight_lane: usize,
        coefficients: &[i64],
        constant: i64,
        gamma: E,
        rounds: usize,
        pool: &mut Pool,
    ) -> Option<Self> {
        let rows = 1_usize << rounds;
        let weight = match lanes.get(weight_lane)? {
            super::Source::Planes(p) => p.planes(),
            super::Source::PlaneSlice(p) => *p,
            _ => return None,
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
                super::Source::Field(_) | super::Source::Planes(_) | super::Source::PlaneSlice(_) => return None,
                other => ints.push((*other, c)),
            }
        }
        let lanes_only: Vec<super::Source<'a>> = ints.iter().map(|(lane, _)| *lane).collect();
        // Per lane one pass over the block: a byte lane through its map scaled by the
        // coefficient, an `i32` lane times the coefficient.
        let scaled: Vec<Option<Box<[i64; 256]>>> = ints
            .iter()
            .map(|(lane, c)| match lane {
                super::Source::Bytes { map, .. } => Some(Box::new(std::array::from_fn(|b| c * map[b]))),
                _ => None,
            })
            .collect();
        let combination = Box::new(move |first: usize, cells: &mut [i64]| {
            cells.fill(constant);
            let range = first..first + cells.len();
            for ((lane, c), scaled) in ints.iter().zip(&scaled) {
                match (lane, scaled) {
                    (super::Source::Ints(column), _) => {
                        for (g, &x) in cells.iter_mut().zip(&column[range.clone()]) {
                            *g += c * i64::from(x);
                        }
                    }
                    (super::Source::Bytes { cells: column, .. }, Some(map)) => {
                        for (g, &x) in cells.iter_mut().zip(&column[range.clone()]) {
                            *g += map[x as u8 as usize];
                        }
                    }
                    _ => unreachable!("integer lanes only"),
                }
            }
        });
        let source = WeightSource {
            weight,
            lanes: lanes_only,
            combination,
            weight_lane,
            vars: rounds,
            slots: [gamma],
        };
        Engine::new(source, None, rounds, pool).map(Self)
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

    /// Return the table buffers to the pool.
    pub fn recycle(self, pool: &mut Pool) {
        self.0.recycle(pool);
    }
}

// ---------------------------------------------------------------------------------------
// Stage 3: a linear combination of integer columns.
// ---------------------------------------------------------------------------------------

struct LinearSource<L: Leaves> {
    columns: L,
    vars: usize,
    /// The coefficient of every column, then one for the combined lane.
    slots: Vec<E>,
    lin_forms: Vec<usize>,
}

impl<L: Leaves> PlaneSource for LinearSource<L> {
    type Leaves = L;
    fn leaves(&self) -> Option<&L> {
        Some(&self.columns)
    }
    fn width(&self) -> usize {
        1
    }
    fn vars(&self) -> usize {
        self.vars
    }
    fn prefix(&self) -> usize {
        PREFIX
    }
    fn constant(&self) -> E {
        E::zero()
    }
    fn slots(&self) -> &[E] {
        &self.slots
    }
    fn linear(&self) -> &[(usize, usize)] {
        &[(0, 4)]
    }
    fn products(&self) -> &[(usize, usize, usize)] {
        &[]
    }
    fn level0(&self) -> Vec<Planes<'_>> {
        unreachable!("the columns are integers")
    }
    fn prefix_sums(&self, first: &[E], second: &[E]) -> (PrefixSums, Vec<usize>, Vec<usize>) {
        (pattern::prefix_sums(&self.columns, first, second), self.lin_forms.clone(), Vec::new())
    }
    fn pattern_fill<'s>(&'s self, e: [E; 4], scratch: &'s mut Vec<[E; 4]>) -> Fill<'s, L> {
        // weights[l][a] = c_l eq(rho_0, a_0) eq(rho_1, a_1).
        scratch.clear();
        scratch.extend((0..4).map(|l| std::array::from_fn(|a| self.slots[l] * e[a])));
        Fill::PatternCombined(scratch.as_slice(), &self.columns)
    }
    fn terminal(&self, _: &[E], _: &[E]) -> Vec<E> {
        Vec::new()
    }
}

/// `sum_x eq(point, x) sum_l c_l f_l(x)` over up to four integer columns `f_l` (`i32` cells
/// with `|f_l| < 2^31`, or `i8` cells read through the identity map): rounds 0 and 1 from the
/// pattern sums of the columns, then the single combined lane `g = sum_l c_l f_l`. The group
/// yields no terminal lane values (the caller reads the lanes' values from another group at
/// the same point).
pub struct LinearIntGroup<'a>(LinearEngine<'a>);

/// The engine over `i32` columns or over `i8` columns.
enum LinearEngine<'a> {
    Ints(Engine<LinearSource<Columns<'a>>>),
    Bytes(Engine<LinearSource<ByteColumns<'a>>>),
}

impl<'a> LinearIntGroup<'a> {
    /// `None` without lanes, with more than four, with a lane that is not an `i32` column
    /// or an identity-mapped `i8` column, with lanes of mixed kinds, with lanes of other
    /// lengths than `2^eq_point.len()`, or with fewer than 2 variables.
    #[must_use]
    pub fn new(lanes: Vec<super::Source<'a>>, coefficients: Vec<E>, eq_point: &[E], pool: &mut Pool) -> Option<Self> {
        let vars = eq_point.len();
        let rows = 1_usize << vars;
        if lanes.is_empty() || lanes.len() > 4 || lanes.len() != coefficients.len() || vars < PREFIX {
            return None;
        }
        let mut ints = Vec::with_capacity(4);
        let mut bytes = Vec::with_capacity(4);
        for lane in &lanes {
            match lane {
                super::Source::Ints(cells) if cells.len() == rows => ints.push(*cells),
                super::Source::Bytes { cells, map } if cells.len() == rows && **map == super::IDENTITY_MAP => {
                    bytes.push(*cells);
                }
                _ => return None,
            }
        }
        if !ints.is_empty() && !bytes.is_empty() {
            return None;
        }
        let mut slots = coefficients;
        slots.resize(4, E::zero());
        slots.push(E::one());
        fn source<L: Leaves>(columns: L, vars: usize, slots: Vec<E>) -> LinearSource<L> {
            LinearSource {
                columns,
                vars,
                slots,
                lin_forms: vec![0, 1, 2, 3],
            }
        }
        if bytes.is_empty() {
            Engine::new(source(Columns { lanes: ints }, vars, slots), Some(eq_point), vars, pool)
                .map(|e| Self(LinearEngine::Ints(e)))
        } else {
            Engine::new(source(ByteColumns { lanes: bytes }, vars, slots), Some(eq_point), vars, pool)
                .map(|e| Self(LinearEngine::Bytes(e)))
        }
    }

    /// The current claim.
    #[must_use]
    pub fn claim(&self) -> E {
        match &self.0 {
            LinearEngine::Ints(e) => e.claim,
            LinearEngine::Bytes(e) => e.claim,
        }
    }

    /// The compressed message of `round`.
    ///
    /// # Panics
    /// When rounds are called out of order.
    pub fn round_message(&mut self, round: usize) -> Vec<E> {
        match &mut self.0 {
            LinearEngine::Ints(e) => e.round_message(round),
            LinearEngine::Bytes(e) => e.round_message(round),
        }
    }

    /// Bind the round's variable to `challenge`.
    ///
    /// # Panics
    /// When rounds are called out of order.
    pub fn bind(&mut self, round: usize, challenge: E) {
        match &mut self.0 {
            LinearEngine::Ints(e) => e.bind(round, challenge),
            LinearEngine::Bytes(e) => e.bind(round, challenge),
        }
    }

    /// The self-check of [`LayerGroup::terminal_consistent`].
    #[must_use]
    pub fn terminal_consistent(&self) -> bool {
        match &self.0 {
            LinearEngine::Ints(e) => e.terminal_consistent(),
            LinearEngine::Bytes(e) => e.terminal_consistent(),
        }
    }

    /// Return the table buffers to the pool.
    pub fn recycle(self, pool: &mut Pool) {
        match self.0 {
            LinearEngine::Ints(e) => e.recycle(pool),
            LinearEngine::Bytes(e) => e.recycle(pool),
        }
    }
}
