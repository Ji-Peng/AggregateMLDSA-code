//! The compiled evaluation plan of a group: the gamma-weighted terms rewritten as
//! single-lane polynomials, bilinear products, weighted sums `w (sum c_l v_l + c)` and
//! residual generic terms, in a source phase (unscaled lanes, the first active round) and
//! a folded phase (lanes scaled at their first fold), and their evaluation in coefficient
//! form at `v_l = a_l + X d_l` (docs/piop.md Section 18).

use jolt_field::{Field, One, Ring, Unreduced, Zero};

use super::field::{fm, lift, mad, taylor, Acc, Poly, MAX_COEFFICIENTS};
use super::{GroupInstance, Term};
use crate::OpeningField as E;

/// A single-lane polynomial without constant term, `sum_{m >= 1} c_m v^m`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum PolyKind {
    /// `v` (a lane scaled by its linear coefficient).
    Unit,
    /// `c1 v`.
    Linear(E),
    /// `c1 v + c2 v^2`.
    Quadratic { c1: E, c2: E },
    /// `c1 v + c2 v^2 + c3 v^3 + c4 v^4` with `three_c3 = 3 c3`.
    Quartic { c: [E; 5], three_c3: E },
    /// Coefficients `c_0 = 0, c_1, ..., c_D` (any degree up to 5).
    General(Vec<E>),
    /// A quartic read through the translated lane `u = v - s` with `s = -c3 / (4 c4)`, so
    /// that `P(u + s) = c4 (u^4 + q u^2 + l u) + k` has no cubic term (Aerie's depressed
    /// quartic, `weighted_affine_coefficients_shifted`); dense rounds only.
    Depressed { c4: E, q: E, l: E, k: E },
}

impl PolyKind {
    fn from_coefficients(c: &[E]) -> Option<Self> {
        let mut c = c.to_vec();
        while c.len() > 1 && c.last().is_some_and(E::is_zero) {
            c.pop();
        }
        if c.len() <= 1 {
            return None;
        }
        debug_assert!(c[0].is_zero(), "constants are split off");
        Some(match c.len() {
            2 if c[1] == E::one() => Self::Unit,
            2 => Self::Linear(c[1]),
            3 => Self::Quadratic { c1: c[1], c2: c[2] },
            4 | 5 => {
                let mut q = [E::zero(); 5];
                q[..c.len()].copy_from_slice(&c);
                Self::Quartic {
                    c: q,
                    three_c3: q[3] + q[3] + q[3],
                }
            }
            _ => Self::General(c),
        })
    }

    /// Coefficients `c_0 = 0, c_1, ...` (trailing zeros trimmed).
    pub fn coefficients(&self) -> Vec<E> {
        let mut c = match self {
            Self::Unit => vec![E::zero(), E::one()],
            Self::Linear(c1) => vec![E::zero(), *c1],
            Self::Quadratic { c1, c2 } => vec![E::zero(), *c1, *c2],
            Self::Quartic { c, .. } => c.to_vec(),
            Self::General(c) => c.clone(),
            Self::Depressed { .. } => unreachable!("dense pieces only"),
        };
        while c.len() > 2 && c.last().is_some_and(E::is_zero) {
            c.pop();
        }
        c
    }

    pub fn degree(&self) -> usize {
        self.coefficients().len() - 1
    }

    /// `P(v)` (Horner).
    pub fn value(&self, v: E) -> E {
        self.coefficients().iter().rev().fold(E::zero(), |acc, &c| fm(acc, v) + c)
    }

    /// The Taylor coefficients `P^(i)(a) / i!` (`i = 0..=degree`).
    pub fn taylor(&self, a: E) -> [E; MAX_COEFFICIENTS] {
        taylor(&self.coefficients(), a)
    }

    /// Add the coefficients of `P(a + X d)` to `out` (unreduced); with `UPPER`, only
    /// coefficients 1 and up (the constant one is recovered from the running claim). The
    /// last product of every coefficient is accumulated without reduction.
    #[inline(always)]
    pub fn add_at<const UPPER: bool>(&self, a: E, d: E, out: &mut [Acc]) {
        match self {
            Self::Unit => {
                if !UPPER {
                    lift(&mut out[0], a);
                }
                lift(&mut out[1], d);
            }
            Self::Linear(c1) => {
                if !UPPER {
                    mad(&mut out[0], *c1, a);
                }
                mad(&mut out[1], *c1, d);
            }
            Self::Quadratic { c1, c2 } => {
                let c2a = fm(*c2, a);
                if !UPPER {
                    mad(&mut out[0], c2a + *c1, a);
                }
                mad(&mut out[1], c2a + c2a + *c1, d);
                mad(&mut out[2], *c2, fm(d, d));
            }
            Self::Quartic { c, three_c3 } => {
                // Aerie's `PreparedWeights::taylor` with the constant term by Horner.
                let c4a = fm(c[4], a);
                let twice = c4a + c4a;
                let four = twice + twice;
                let t3 = four + c[3];
                let t2 = fm(four + twice + *three_c3, a) + c[2];
                let t1 = fm(fm(four + *three_c3, a) + c[2] + c[2], a) + c[1];
                if !UPPER {
                    mad(&mut out[0], fm(fm(c4a + c[3], a) + c[2], a) + c[1], a);
                }
                let d2 = fm(d, d);
                mad(&mut out[1], t1, d);
                mad(&mut out[2], t2, d2);
                mad(&mut out[3], t3, fm(d2, d));
                if out.len() > 4 {
                    mad(&mut out[4], c[4], fm(d2, d2));
                }
            }
            Self::General(c) => {
                let t = taylor(c, a);
                let mut power = E::one();
                for (i, slot) in out.iter_mut().enumerate().take(c.len()) {
                    if !UPPER || i > 0 {
                        mad(slot, t[i], power);
                    }
                    power = fm(power, d);
                }
            }
            Self::Depressed { c4, q, l, k } => {
                // `a = u`; nine products for coefficients 1 to 4, four of them unreduced.
                let u2 = fm(a, a);
                if !UPPER {
                    mad(&mut out[0], *c4, fm(u2 + *q, u2) + fm(*l, a));
                    lift(&mut out[0], *k);
                }
                let d1 = fm(*c4, d);
                let d2 = fm(d1, d);
                let d3 = fm(d2, d);
                let four_u = a + a + a + a;
                let four_u2 = u2 + u2 + u2 + u2;
                mad(&mut out[1], d1, fm(a, four_u2 + *q + *q) + *l);
                mad(&mut out[2], d2, four_u2 + u2 + u2 + *q);
                mad(&mut out[3], d3, four_u);
                mad(&mut out[4], d3, d);
            }
        }
    }

    /// The depressed form of a quartic (`c4 != 0`) and its translation `s`.
    pub fn depressed(&self) -> Option<(Self, E)> {
        let Self::Quartic { c, .. } = self else {
            return None;
        };
        if c[4].is_zero() {
            return None;
        }
        let four_c4 = c[4] + c[4] + c[4] + c[4];
        let shift = -c[3] * four_c4.inverse().expect("nonzero");
        let t = taylor(c, shift);
        debug_assert!(t[3].is_zero(), "no cubic term after the translation");
        let inverse = c[4].inverse().expect("nonzero");
        Some((
            Self::Depressed {
                c4: c[4],
                q: t[2] * inverse,
                l: t[1] * inverse,
                k: t[0],
            },
            shift,
        ))
    }
}

/// `c v_a v_b`; `unit` when `c = 1`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Bilinear {
    pub c: E,
    pub a: usize,
    pub b: usize,
}

/// `w (sum_l c_l v_l + constant)`: every piece linear in the weight lane `w`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Weighted {
    pub weight: usize,
    pub lanes: Vec<(usize, E)>,
    pub constant: E,
}

/// The pieces of a plan in one phase.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct Pieces {
    /// At most one polynomial per lane.
    pub polys: Vec<(usize, PolyKind)>,
    pub bilinear: Vec<Bilinear>,
    pub weighted: Vec<Weighted>,
    /// `gamma * term` for the terms without a compiled form (`RangeW1`, products of three
    /// or more lanes).
    pub generic: Vec<(E, Term)>,
}

impl Pieces {
    pub fn is_empty(&self) -> bool {
        self.polys.is_empty()
            && self.bilinear.is_empty()
            && self.weighted.is_empty()
            && self.generic.is_empty()
    }

    /// Every lane read by a piece, without repetition.
    pub fn lanes(&self) -> Vec<usize> {
        let mut out = Vec::new();
        let mut push = |l: usize| {
            if !out.contains(&l) {
                out.push(l);
            }
        };
        for (l, _) in &self.polys {
            push(*l);
        }
        for b in &self.bilinear {
            push(b.a);
            push(b.b);
        }
        for w in &self.weighted {
            push(w.weight);
            for &(l, _) in &w.lanes {
                push(l);
            }
        }
        for (_, term) in &self.generic {
            for l in term.lanes() {
                push(l);
            }
        }
        out
    }

    /// Add the coefficients of every piece at `v_l = a[l] + X d[l]` to `out` (unreduced).
    #[inline]
    pub fn add_at(&self, a: &[E], d: &[E], out: &mut [Acc]) {
        self.add::<false>(a, d, out);
    }

    /// [`Self::add_at`] for the coefficients 1 and up only.
    #[inline]
    pub fn add_upper(&self, a: &[E], d: &[E], out: &mut [Acc]) {
        self.add::<true>(a, d, out);
    }

    /// The reduced coefficients of every piece at `v_l = a[l] + X d[l]` (`width` of them).
    pub fn evaluate(&self, a: &[E], d: &[E], width: usize, upper: bool) -> Vec<E> {
        let mut out = vec![Acc::zero(); width];
        if upper {
            self.add_upper(a, d, &mut out);
        } else {
            self.add_at(a, d, &mut out);
        }
        out.into_iter().map(E::reduce_product).collect()
    }

    #[inline(always)]
    fn add<const UPPER: bool>(&self, a: &[E], d: &[E], out: &mut [Acc]) {
        for (l, poly) in &self.polys {
            poly.add_at::<UPPER>(a[*l], d[*l], out);
        }
        for b in &self.bilinear {
            add_bilinear_impl::<UPPER>(b.c, a[b.a], d[b.a], a[b.b], d[b.b], out);
        }
        for w in &self.weighted {
            let (mut sa, mut sd) = (w.constant, E::zero());
            for &(l, c) in &w.lanes {
                if c == E::one() {
                    sa += a[l];
                    sd += d[l];
                } else {
                    sa += fm(c, a[l]);
                    sd += fm(c, d[l]);
                }
            }
            add_bilinear_impl::<UPPER>(E::one(), a[w.weight], d[w.weight], sa, sd, out);
        }
        for (gamma, term) in &self.generic {
            match term {
                Term::RangeW1([b5, b4, b3, b2]) => {
                    // b5 (b4 + b3 b2): one Karatsuba product, one affine times quadratic.
                    let (a3, d3, a2, d2) = (a[*b3], d[*b3], a[*b2], d[*b2]);
                    let p0 = fm(a3, a2);
                    let p2 = fm(d3, d2);
                    let p1 = fm(a3 + d3, a2 + d2) - p0 - p2;
                    let (i0, i1, i2) = (p0 + a[*b4], p1 + d[*b4], p2);
                    let (a5, d5) = (a[*b5], d[*b5]);
                    if !UPPER {
                        mad(&mut out[0], *gamma, fm(a5, i0));
                    }
                    mad(&mut out[1], *gamma, fm(a5, i1) + fm(d5, i0));
                    mad(&mut out[2], *gamma, fm(a5, i2) + fm(d5, i1));
                    mad(&mut out[3], *gamma, fm(d5, i2));
                }
                _ => {
                    let poly = term_poly(term, a, d).scale(*gamma);
                    let start = usize::from(UPPER);
                    for (slot, &c) in out.iter_mut().zip(&poly.c[..poly.len]).skip(start) {
                        lift(slot, c);
                    }
                }
            }
        }
    }
}

/// `c (a1 + X d1)(a2 + X d2)` (Karatsuba middle coefficient).
#[inline(always)]
pub(super) fn add_bilinear(c: E, a1: E, d1: E, a2: E, d2: E, out: &mut [Acc]) {
    add_bilinear_impl::<false>(c, a1, d1, a2, d2, out);
}

#[inline(always)]
fn add_bilinear_impl<const UPPER: bool>(c: E, a1: E, d1: E, a2: E, d2: E, out: &mut [Acc]) {
    if c == E::one() {
        if UPPER {
            mad(&mut out[1], a1, d2);
            mad(&mut out[1], d1, a2);
        } else {
            let p0 = fm(a1, a2);
            let p2 = fm(d1, d2);
            lift(&mut out[0], p0);
            lift(&mut out[1], fm(a1 + d1, a2 + d2) - p0 - p2);
            lift(&mut out[2], p2);
            return;
        }
        mad(&mut out[2], d1, d2);
    } else {
        let p2 = fm(d1, d2);
        let p1 = if UPPER {
            fm(a1, d2) + fm(d1, a2)
        } else {
            let p0 = fm(a1, a2);
            mad(&mut out[0], c, p0);
            fm(a1 + d1, a2 + d2) - p0 - p2
        };
        mad(&mut out[1], c, p1);
        mad(&mut out[2], c, p2);
    }
}

/// A term as a polynomial in `X` at `v_l = a[l] + X d[l]`.
pub(super) fn term_poly(term: &Term, a: &[E], d: &[E]) -> Poly {
    let lane = |l: usize| Poly::affine(a[l], d[l]);
    match term {
        Term::Product(lanes) => lanes
            .iter()
            .fold(Poly::constant(E::one()), |acc, &l| acc.mul(&lane(l))),
        Term::Bool(l) => lane(*l).mul(&lane(*l).add_constant(-E::one())),
        Term::Link(l) => lane(*l).mul(&lane(*l).add_constant(E::one())),
        Term::RangeW1([b5, b4, b3, b2]) => lane(*b5).mul(&lane(*b4).add(&lane(*b3).mul(&lane(*b2)))),
        Term::Leaf(l, poly) => poly
            .roots()
            .iter()
            .fold(Poly::constant(E::one()), |acc, &root| {
                acc.mul(&lane(*l).add_constant(-E::from_i64(root)))
            }),
        Term::Affine {
            weight,
            lanes,
            constant,
        } => {
            let affine = lanes.iter().fold(Poly::constant(E::from_i64(*constant)), |acc, &(l, c)| {
                acc.add(&lane(l).scale(E::from_i64(c)))
            });
            match weight {
                Some(w) => lane(*w).mul(&affine),
                None => affine,
            }
        }
        Term::Monomials(monomials) => monomials.iter().fold(Poly::constant(E::zero()), |acc, (c, m)| {
            let product = m
                .iter()
                .fold(Poly::constant(E::one()), |p, &l| p.mul(&lane(l)));
            acc.add(&product.scale(E::from_i64(*c)))
        }),
    }
}

/// The compiled terms of a group.
#[derive(Clone, Debug)]
pub(super) struct Plan {
    /// Read from the unscaled source values (first active round).
    pub source: Pieces,
    /// Read from the folded, scaled values (later rounds).
    pub folded: Pieces,
    /// Read from the dense table (level 3 on): the folded pieces with depressed quartics.
    pub dense: Pieces,
    /// Per lane, the translation `s` of its dense column (`u = v - s`; zero for most lanes).
    pub shifts: Vec<E>,
    /// Per lane, the factor applied at the first fold (one for most lanes).
    pub scales: Vec<E>,
    /// The gamma-weighted constant of every cell (affine constants, polynomial constants).
    pub constant: E,
}

impl Plan {
    /// Compile the instances; `is_field[l]` marks the lanes over a field table (weights),
    /// whose bilinear products join the weighted sum of that lane.
    pub fn compile(lane_count: usize, instances: &[GroupInstance], is_field: &[bool]) -> Self {
        let mut polys: Vec<Vec<E>> = vec![Vec::new(); lane_count];
        let add_poly = |polys: &mut Vec<Vec<E>>, l: usize, c: &[E]| {
            let slot = &mut polys[l];
            if slot.len() < c.len() {
                slot.resize(c.len(), E::zero());
            }
            for (s, &v) in slot.iter_mut().zip(c) {
                *s += v;
            }
        };
        let mut constant = E::zero();
        let mut bilinear: Vec<Bilinear> = Vec::new();
        let mut weighted: Vec<Weighted> = Vec::new();
        let mut generic = Vec::new();
        let weighted_of = |weighted: &mut Vec<Weighted>, w: usize| -> usize {
            if let Some(i) = weighted.iter().position(|form| form.weight == w) {
                i
            } else {
                weighted.push(Weighted {
                    weight: w,
                    lanes: Vec::new(),
                    constant: E::zero(),
                });
                weighted.len() - 1
            }
        };
        let add_weighted_lane = |form: &mut Weighted, l: usize, c: E| {
            match form.lanes.iter_mut().find(|(lane, _)| *lane == l) {
                Some((_, slot)) => *slot += c,
                None => form.lanes.push((l, c)),
            }
        };
        for instance in instances {
            let gamma = instance.gamma;
            for term in &instance.terms {
                match term {
                    Term::Product(lanes) if lanes.is_empty() => constant += gamma,
                    Term::Product(lanes) if lanes.len() == 1 => {
                        add_poly(&mut polys, lanes[0], &[E::zero(), gamma]);
                    }
                    Term::Product(lanes) if lanes.len() == 2 && lanes[0] == lanes[1] => {
                        add_poly(&mut polys, lanes[0], &[E::zero(), E::zero(), gamma]);
                    }
                    Term::Product(lanes) if lanes.len() == 2 => {
                        let (a, b) = (lanes[0], lanes[1]);
                        if is_field[a] != is_field[b] {
                            let (w, l) = if is_field[a] { (a, b) } else { (b, a) };
                            let i = weighted_of(&mut weighted, w);
                            add_weighted_lane(&mut weighted[i], l, gamma);
                        } else {
                            bilinear.push(Bilinear { c: gamma, a, b });
                        }
                    }
                    Term::Bool(l) => add_poly(&mut polys, *l, &[E::zero(), -gamma, gamma]),
                    Term::Link(l) => add_poly(&mut polys, *l, &[E::zero(), gamma, gamma]),
                    Term::Leaf(l, poly) => {
                        let c: Vec<E> = poly.coefficients().iter().map(|&c| gamma.mul_i64(c)).collect();
                        constant += c[0];
                        let mut rest = c;
                        rest[0] = E::zero();
                        add_poly(&mut polys, *l, &rest);
                    }
                    Term::Affine {
                        weight: None,
                        lanes,
                        constant: c,
                    } => {
                        constant += gamma.mul_i64(*c);
                        for &(l, coefficient) in lanes {
                            add_poly(&mut polys, l, &[E::zero(), gamma.mul_i64(coefficient)]);
                        }
                    }
                    Term::Affine {
                        weight: Some(w),
                        lanes,
                        constant: c,
                    } => {
                        let i = weighted_of(&mut weighted, *w);
                        weighted[i].constant += gamma.mul_i64(*c);
                        for &(l, coefficient) in lanes {
                            add_weighted_lane(&mut weighted[i], l, gamma.mul_i64(coefficient));
                        }
                    }
                    Term::Monomials(monomials) => {
                        // Expand into the cheap pieces: constants, single-lane powers (one
                        // polynomial per lane), products of two lanes (bilinear or weighted),
                        // and only the mixed products of three or more lanes on the generic
                        // path.
                        for (c, m) in monomials {
                            let coefficient = gamma.mul_i64(*c);
                            match m.as_slice() {
                                [] => constant += coefficient,
                                [first, rest @ ..] if rest.iter().all(|l| l == first) => {
                                    let mut poly = vec![E::zero(); m.len() + 1];
                                    poly[m.len()] = coefficient;
                                    add_poly(&mut polys, *first, &poly);
                                }
                                [a, b] => {
                                    if is_field[*a] != is_field[*b] {
                                        let (w, l) = if is_field[*a] { (*a, *b) } else { (*b, *a) };
                                        let i = weighted_of(&mut weighted, w);
                                        add_weighted_lane(&mut weighted[i], l, coefficient);
                                    } else {
                                        bilinear.push(Bilinear { c: coefficient, a: *a, b: *b });
                                    }
                                }
                                _ => generic.push((coefficient, Term::Product(m.clone()))),
                            }
                        }
                    }
                    other => generic.push((gamma, other.clone())),
                }
            }
        }
        for form in &mut weighted {
            form.lanes.retain(|(_, c)| !c.is_zero());
        }
        let source_polys: Vec<(usize, PolyKind)> = polys
            .iter()
            .enumerate()
            .filter_map(|(l, c)| PolyKind::from_coefficients(c).map(|kind| (l, kind)))
            .collect();
        let source = Pieces {
            polys: source_polys,
            bilinear,
            weighted,
            generic,
        };

        // Uses of every lane outside its own polynomial.
        let mut uses = vec![0_usize; lane_count];
        for b in &source.bilinear {
            uses[b.a] += 1;
            uses[b.b] += 1;
        }
        for form in &source.weighted {
            uses[form.weight] += 1;
            for &(l, _) in &form.lanes {
                uses[l] += 1;
            }
        }
        for (_, term) in &source.generic {
            for l in term.lanes() {
                uses[l] += 1;
            }
        }
        let has_poly = |l: usize| source.polys.iter().any(|(lane, _)| *lane == l);
        let mut folded = source.clone();
        let mut scales = vec![E::one(); lane_count];
        // A lane read only by its own linear polynomial is scaled at its first fold.
        for (l, kind) in &mut folded.polys {
            if uses[*l] == 0 {
                if let PolyKind::Linear(c) = kind {
                    scales[*l] = *c;
                    *kind = PolyKind::Unit;
                }
            }
        }
        // A lane read by exactly one bilinear product and nothing else absorbs its
        // coefficient (one factor per product).
        for b in &mut folded.bilinear {
            if b.c.is_zero() || b.a == b.b {
                continue;
            }
            for l in [b.a, b.b] {
                if uses[l] == 1 && !has_poly(l) && scales[l] == E::one() {
                    scales[l] = b.c;
                    b.c = E::one();
                    break;
                }
            }
        }
        // A lane read once as a summand of a weighted sum absorbs its coefficient.
        for form in &mut folded.weighted {
            for (l, c) in &mut form.lanes {
                if uses[*l] == 1 && !has_poly(*l) && scales[*l] == E::one() && !c.is_zero() {
                    scales[*l] = *c;
                    *c = E::one();
                }
            }
        }
        debug_assert!(scales.iter().all(|s| !s.is_zero()));
        // A quartic lane read only by its polynomial is stored translated in the dense table.
        let mut dense = folded.clone();
        let mut shifts = vec![E::zero(); lane_count];
        for (l, kind) in &mut dense.polys {
            if uses[*l] == 0 {
                if let Some((depressed, shift)) = kind.depressed() {
                    *kind = depressed;
                    shifts[*l] = shift;
                }
            }
        }
        Self {
            source,
            folded,
            dense,
            shifts,
            scales,
            constant,
        }
    }

    /// The inverse scales (for the terminal values).
    pub fn inverse_scales(&self) -> Vec<E> {
        self.scales
            .iter()
            .map(|&s| if s == E::one() { s } else { s.inverse().expect("nonzero scale") })
            .collect()
    }
}
