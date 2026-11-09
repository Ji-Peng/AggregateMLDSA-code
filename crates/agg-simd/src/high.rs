//! The evaluation of summands of degree three and four on the pairs of a fused pass.
//!
//! A summand is a sum of monomials, each a product of at most four lane powers of total
//! degree at most [`MAX_DEGREE`] and an optional inner coefficient `k_m`, grouped by
//! coefficient slot as in [`crate::fused`]:
//! `f = sum_slot c_slot sum_{m in slot} k_m prod_{(l, e) in m} L_l^e`. On the pair of rows
//! `(2k, 2k + 1)` with `L_l = a_l + X d_l` the summand is a polynomial of degree `D` in `X`
//! (`D` the largest monomial degree). The kernels return per slot its values at
//! `X = 0, 1, ..., D - 1` and its coefficient of `X^D` (the products of the differences
//! over the monomials of degree `D`), which determine the polynomial; the value at one is
//! computed only when `full`. Every lane power is computed once per point and item and
//! shared by the monomials that read it. Field sums are exact, so the vector kernels and the
//! scalar one return the same elements.
use crate::{
    detect::{backend, Backend},
    k::Planes,
    scalar, K,
};
use jolt_field::{Ring, Zero};

/// Largest total degree of a monomial.
pub const MAX_DEGREE: usize = 4;
/// Most lane factors of a monomial.
pub const MAX_FACTORS: usize = 4;

/// One monomial of a summand: `(lane, exponent)` factors (exponents at least one, lanes
/// distinct), times `coefficient` when there is one, added to coefficient slot `slot`. An
/// inner coefficient costs one product per monomial and point; a slot costs one weighted
/// reduction per point, so slots of single lane powers (the coefficients of lane
/// polynomials) are cheaper merged into one slot with inner coefficients.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Monomial {
    pub factors: Vec<(usize, usize)>,
    pub coefficient: Option<K>,
    pub slot: usize,
}

impl Monomial {
    /// The total degree.
    #[must_use]
    pub fn degree(&self) -> usize {
        self.factors.iter().map(|&(_, e)| e).sum()
    }
}

/// The monomials of a summand over `slots` coefficient slots.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HighTerms {
    pub monomials: Vec<Monomial>,
    pub slots: usize,
}

impl HighTerms {
    /// The degree `D` of the summand in `X` (the largest monomial degree).
    #[must_use]
    pub fn degree(&self) -> usize {
        self.monomials.iter().map(Monomial::degree).max().unwrap_or(0)
    }

    /// Components per slot: the values at `0..D` and the coefficient of `X^D`.
    #[must_use]
    pub fn components(&self) -> usize {
        self.degree() + 1
    }
}

/// A lane power of the evaluation: a lane read at the point (index into `Plan::lanes`), or
/// the product of two earlier powers.
#[derive(Clone, Copy, Debug)]
enum Node {
    Base(usize),
    Product(usize, usize),
}

/// A compiled monomial: its lane powers (nodes), its inner coefficient (an index into
/// `Plan::constants`), and whether it has degree `D` (only those enter the leading
/// coefficient).
#[derive(Clone, Debug)]
struct Compiled {
    powers: Vec<usize>,
    constant: Option<usize>,
    top: bool,
}

/// The compiled form of [`HighTerms`], built once per pass: the lanes read, the lane powers
/// in dependency order, the inner coefficients, and the monomials by slot.
pub(crate) struct Plan {
    degree: usize,
    slots: usize,
    /// The lanes read, without repetition.
    lanes: Vec<usize>,
    nodes: Vec<Node>,
    constants: Vec<K>,
    /// Monomials of slot `s`: `monomials[offsets[s]..offsets[s + 1]]`.
    offsets: Vec<usize>,
    monomials: Vec<Compiled>,
    /// Per slot: whether some monomial has degree `D`.
    has_top: Vec<bool>,
    /// The most accumulator units a slot takes in one component of one step (a lifted power
    /// takes one, a product two).
    units: usize,
}

impl Plan {
    /// # Panics
    /// When a monomial has no factor, more than [`MAX_FACTORS`] factors, a repeated lane, a
    /// zero exponent, a degree above [`MAX_DEGREE`], or names a lane at or above `width` or a
    /// slot at or above `terms.slots`.
    pub(crate) fn new(terms: &HighTerms, width: usize) -> Self {
        let degree = terms.degree();
        assert!((1..=MAX_DEGREE).contains(&degree), "a summand of degree 1 to MAX_DEGREE");
        let mut lanes: Vec<usize> = Vec::new();
        let mut nodes: Vec<Node> = Vec::new();
        // The node of (lane, exponent), built from the square-and-multiply chain.
        let mut memo: Vec<((usize, usize), usize)> = Vec::new();
        fn power(
            lane: usize,
            e: usize,
            lanes: &mut Vec<usize>,
            nodes: &mut Vec<Node>,
            memo: &mut Vec<((usize, usize), usize)>,
        ) -> usize {
            if let Some(&(_, n)) = memo.iter().find(|(key, _)| *key == (lane, e)) {
                return n;
            }
            let node = if e == 1 {
                let index = lanes.iter().position(|&l| l == lane).unwrap_or_else(|| {
                    lanes.push(lane);
                    lanes.len() - 1
                });
                Node::Base(index)
            } else {
                let half = power(lane, e / 2, lanes, nodes, memo);
                let rest = power(lane, e - e / 2, lanes, nodes, memo);
                Node::Product(half, rest)
            };
            nodes.push(node);
            memo.push(((lane, e), nodes.len() - 1));
            nodes.len() - 1
        }
        let mut constants: Vec<K> = Vec::new();
        let mut by_slot: Vec<Vec<Compiled>> = vec![Vec::new(); terms.slots];
        for m in &terms.monomials {
            assert!(m.slot < terms.slots, "monomials name existing slots");
            assert!((1..=MAX_FACTORS).contains(&m.factors.len()), "one to MAX_FACTORS factors");
            for (i, &(lane, e)) in m.factors.iter().enumerate() {
                assert!(lane < width && e >= 1, "factors name existing lanes with positive exponents");
                assert!(m.factors[..i].iter().all(|&(l, _)| l != lane), "distinct lanes per monomial");
            }
            let powers: Vec<usize> =
                m.factors.iter().map(|&(lane, e)| power(lane, e, &mut lanes, &mut nodes, &mut memo)).collect();
            let constant = m.coefficient.map(|c| {
                constants.iter().position(|&k| k == c).unwrap_or_else(|| {
                    constants.push(c);
                    constants.len() - 1
                })
            });
            by_slot[m.slot].push(Compiled { powers, constant, top: m.degree() == degree });
        }
        let mut offsets = vec![0];
        let mut monomials = Vec::new();
        let mut has_top = Vec::new();
        let mut units = 1;
        for slot in by_slot {
            has_top.push(slot.iter().any(|m| m.top));
            units = units.max(slot.iter().map(|m| if m.powers.len() == 1 && m.constant.is_none() { 1 } else { 2 }).sum());
            monomials.extend(slot);
            offsets.push(monomials.len());
        }
        Self { degree, slots: terms.slots, lanes, nodes, constants, offsets, monomials, has_top, units }
    }

    fn components(&self) -> usize {
        self.degree + 1
    }

    /// Whether component `c` of slot `s` is computed (and may be nonzero).
    fn active(&self, s: usize, c: usize, full: bool) -> bool {
        self.offsets[s] < self.offsets[s + 1] && (c != 1 || full) && (c < self.degree || self.has_top[s])
    }
}

/// The per-task scratch of the vector kernels (empty on the portable backend).
pub(crate) enum Scratch {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    W8(WideScratch<std::arch::x86_64::__m512i>),
    #[cfg(target_arch = "x86_64")]
    W4(WideScratch<std::arch::x86_64::__m256i>),
}

impl Scratch {
    pub(crate) fn new(plan: &Plan) -> Self {
        #[cfg(target_arch = "x86_64")]
        match backend() {
            // SAFETY: `__m512i` is an integer vector (every bit pattern is valid).
            Backend::Avx512 | Backend::Avx512F => return Self::W8(unsafe { WideScratch::zeros(plan) }),
            // SAFETY: `__m256i` is an integer vector.
            Backend::Avx2 => return Self::W4(unsafe { WideScratch::zeros(plan) }),
            Backend::Portable => {}
        }
        let _ = plan;
        Self::Scalar
    }
}

/// The lanes of the current step (even rows, odd rows, differences), the powers at the
/// current point (canonical and split), and per slot and component the unreduced
/// accumulators and the reduced running sums.
#[cfg(target_arch = "x86_64")]
pub(crate) struct WideScratch<V> {
    base: Vec<[(V, V); 3]>,
    raw: Vec<(V, V)>,
    split: Vec<crate::wide::KS<V>>,
    /// The inner coefficients, broadcast and split once per call.
    constants: Vec<crate::wide::KS<V>>,
    acc: Vec<(crate::wide::Acc<V>, crate::wide::Acc<V>)>,
    running: Vec<(V, V)>,
}

#[cfg(target_arch = "x86_64")]
impl<V: Copy> WideScratch<V> {
    /// # Safety
    /// `V` must be a SIMD integer vector (the all-zero bit pattern is valid).
    unsafe fn zeros(plan: &Plan) -> Self {
        // SAFETY: the all-zero bit pattern is a valid vector, split value and accumulator.
        let (v, ks, acc): (V, crate::wide::KS<V>, crate::wide::Acc<V>) =
            unsafe { (std::mem::zeroed(), std::mem::zeroed(), std::mem::zeroed()) };
        let cells = plan.slots * plan.components();
        Self {
            base: vec![[(v, v); 3]; plan.lanes.len()],
            raw: vec![(v, v); plan.nodes.len()],
            split: vec![ks; plan.nodes.len()],
            constants: vec![ks; plan.constants.len()],
            acc: vec![(acc, acc); cells],
            running: vec![(v, v); cells],
        }
    }
}

/// The slot sums of the summand over the items `k0..k1` with the item weights
/// `weights[k - k0]` (one without weights), added to `out[slot * (D + 1) + c]`: component
/// `c < D` is `sum w f(a + c d)` (component 1 only when `full`), component `D` is
/// `sum w` of the coefficient of `X^D`.
///
/// # Panics
/// When the terms are malformed ([`Plan::new`]), a lane holds fewer than `2 k1` rows, the
/// weights fewer than `k1 - k0`, or `out` fewer than `slots (D + 1)` entries.
pub fn eval_items(
    lanes: &[Planes<'_>],
    terms: &HighTerms,
    weights: Option<Planes<'_>>,
    k0: usize,
    k1: usize,
    full: bool,
    out: &mut [K],
) {
    let plan = Plan::new(terms, lanes.len());
    let mut scratch = Scratch::new(&plan);
    eval_items_with(lanes, &plan, &mut scratch, weights, k0, k1, full, out);
}

/// [`eval_items`] with the plan and the scratch of the enclosing pass.
#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_items_with(
    lanes: &[Planes<'_>],
    plan: &Plan,
    scratch: &mut Scratch,
    weights: Option<Planes<'_>>,
    k0: usize,
    k1: usize,
    full: bool,
    out: &mut [K],
) {
    assert!(k0 <= k1);
    assert!(out.len() >= plan.slots * plan.components());
    // The vector kernels read rows `2k .. 2k + 2 LANES` of every lane the plan reads (all
    // below `lanes.len()`, checked by `Plan::new`) through raw pointers.
    assert!(
        plan.lanes.iter().all(|&l| lanes[l].0.len() >= 2 * k1 && lanes[l].1.len() >= 2 * k1),
        "lanes cover the items"
    );
    if let Some(w) = weights {
        assert!(w.0.len() >= k1 - k0 && w.1.len() >= k1 - k0);
    }
    #[cfg(target_arch = "x86_64")]
    if k1 - k0 >= 8 {
        match scratch {
            // SAFETY: the scratch kind follows the backend, which guarantees AVX-512F; the
            // assertions above cover every read.
            Scratch::W8(s) => return unsafe { eval_w8(lanes, plan, s, weights, k0, k1, full, out) },
            // SAFETY: as above, for AVX2.
            Scratch::W4(s) => return unsafe { eval_w4(lanes, plan, s, weights, k0, k1, full, out) },
            Scratch::Scalar => {}
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = scratch;
    eval_scalar(lanes, plan, weights, k0, k1, full, out);
}

fn eval_scalar(
    lanes: &[Planes<'_>],
    plan: &Plan,
    weights: Option<Planes<'_>>,
    k0: usize,
    k1: usize,
    full: bool,
    out: &mut [K],
) {
    let comps = plan.components();
    let mut base = vec![(K::zero(), K::zero()); plan.lanes.len()];
    let mut power = vec![K::zero(); plan.nodes.len()];
    for k in k0..k1 {
        for (b, &l) in base.iter_mut().zip(&plan.lanes) {
            let (even, odd) = (scalar::get(lanes[l], 2 * k), scalar::get(lanes[l], 2 * k + 1));
            *b = (even, odd - even);
        }
        let w = weights.map(|w| scalar::get(w, k - k0));
        for c in 0..comps {
            if c == 1 && !full {
                continue;
            }
            let top = c == plan.degree;
            for i in 0..plan.nodes.len() {
                power[i] = match plan.nodes[i] {
                    Node::Base(b) => {
                        let (even, d) = base[b];
                        if top {
                            d
                        } else {
                            even + d * K::from_u64(c as u64)
                        }
                    }
                    Node::Product(x, y) => power[x] * power[y],
                };
            }
            for s in 0..plan.slots {
                if !plan.active(s, c, full) {
                    continue;
                }
                let mut sum = K::zero();
                for m in &plan.monomials[plan.offsets[s]..plan.offsets[s + 1]] {
                    if top && !m.top {
                        continue;
                    }
                    let product = m.powers.iter().fold(K::from_u64(1), |p, &n| p * power[n]);
                    sum += match m.constant {
                        Some(i) => product * plan.constants[i],
                        None => product,
                    };
                }
                out[s * comps + c] += match w {
                    Some(w) => w * sum,
                    None => sum,
                };
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn eval_w8(
    lanes: &[Planes<'_>],
    plan: &Plan,
    scratch: &mut WideScratch<std::arch::x86_64::__m512i>,
    weights: Option<Planes<'_>>,
    k0: usize,
    k1: usize,
    full: bool,
    out: &mut [K],
) {
    // SAFETY: AVX-512F is enabled; the caller checked the reads.
    unsafe { eval_wide::<crate::wide::W8>(lanes, plan, scratch, weights, k0, k1, full, out) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn eval_w4(
    lanes: &[Planes<'_>],
    plan: &Plan,
    scratch: &mut WideScratch<std::arch::x86_64::__m256i>,
    weights: Option<Planes<'_>>,
    k0: usize,
    k1: usize,
    full: bool,
    out: &mut [K],
) {
    // SAFETY: AVX2 is enabled; the caller checked the reads.
    unsafe { eval_wide::<crate::wide::W4>(lanes, plan, scratch, weights, k0, k1, full, out) }
}

/// [`eval_items`] on the vector backends, step by step over `LANES` items. Per point the
/// lane powers are computed reduced (canonical) and split once; per slot the monomials are
/// summed unreduced. Without weights they go straight into the slot's accumulator; with
/// weights the slot's value is reduced to a multiplier operand (below 2^64) and multiplied
/// by the weight.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn eval_wide<B: crate::wide::Simd>(
    lanes: &[Planes<'_>],
    plan: &Plan,
    scratch: &mut WideScratch<B::V>,
    weights: Option<Planes<'_>>,
    k0: usize,
    k1: usize,
    full: bool,
    out: &mut [K],
) {
    use crate::wide::*;
    // SAFETY: the caller guarantees the backend's features, that every lane of the plan holds
    // the rows `2 k0 .. 2 k1` and the weights `k1 - k0` rows.
    unsafe {
        let l = B::LANES;
        let comps = plan.components();
        let stop = k0 + (k1 - k0) / l * l;
        let WideScratch { base, raw, split: ks, constants, acc, running } = scratch;
        for (split, &c) in constants.iter_mut().zip(&plan.constants) {
            *split = split_k::<B>(bck::<B>(c));
        }
        // A weighted slot takes one product (two units) per step and component; unweighted,
        // its monomials. Reduce before the units reach the accumulator's limit.
        let units = if weights.is_some() { 2 } else { plan.units };
        let flush_steps = (MAC_LIMIT / units).max(1);
        let zk = (zero::<B>(), zero::<B>());
        running.fill(zk);
        acc.fill(acck::<B>());
        let mut steps = 0_usize;
        for k in (k0..stop).step_by(l) {
            for (b, &li) in base.iter_mut().zip(&plan.lanes) {
                let (c0, c1) = lanes[li];
                // Items in the backend's lane order (the weights are put in the same order).
                let (e0, o0) = B::deinterleave_any(B::load(c0.as_ptr().add(2 * k)), B::load(c0.as_ptr().add(2 * k + l)));
                let (e1, o1) = B::deinterleave_any(B::load(c1.as_ptr().add(2 * k)), B::load(c1.as_ptr().add(2 * k + l)));
                // Canonical even and odd rows and their canonical difference.
                *b = [(e0, e1), (o0, o1), subk::<B>((o0, o1), (e0, e1))];
            }
            let w = match weights {
                Some(w) => {
                    let (w0, w1) = loadk::<B>(w, k - k0);
                    Some(split_k::<B>((B::reorder(w0), B::reorder(w1))))
                }
                None => None,
            };
            for c in 0..comps {
                if c == 1 && !full {
                    continue;
                }
                let top = c == plan.degree;
                for i in 0..plan.nodes.len() {
                    // Every power is canonical: the products reduce fully.
                    let value = match plan.nodes[i] {
                        Node::Base(b) => {
                            let [even, odd, d] = base[b];
                            match c {
                                0 => even,
                                1 => odd,
                                _ if top => d,
                                // `a + c d = odd + (c - 1) d`.
                                _ => {
                                    let mut v = addk::<B>(odd, d);
                                    for _ in 2..c {
                                        v = addk::<B>(v, d);
                                    }
                                    v
                                }
                            }
                        }
                        Node::Product(x, y) => kmul_s::<B>(raw[x], ks[y]),
                    };
                    raw[i] = value;
                    ks[i] = split_k::<B>(value);
                }
                for s in 0..plan.slots {
                    if !plan.active(s, c, full) {
                        continue;
                    }
                    let i = s * comps + c;
                    let mut value = if w.is_some() { acck::<B>() } else { acc[i] };
                    for m in &plan.monomials[plan.offsets[s]..plan.offsets[s + 1]] {
                        if top && !m.top {
                            continue;
                        }
                        match (m.powers.as_slice(), m.constant) {
                            ([p], None) => {
                                lift::<B>(&mut value.0, raw[*p].0);
                                lift::<B>(&mut value.1, raw[*p].1);
                            }
                            ([p], Some(i)) => kmac::<B>(&mut value, (ks[*p].c0, ks[*p].c1), constants[i]),
                            ([first, rest @ ..], constant) => {
                                // The canonical product of all powers but the last factor, which
                                // is the inner coefficient when there is one.
                                let (middle, last) = match constant {
                                    Some(i) => (rest, constants[i]),
                                    None => (&rest[..rest.len() - 1], ks[rest[rest.len() - 1]]),
                                };
                                let mut t = raw[*first];
                                for &p in middle {
                                    t = kmul_s::<B>(t, ks[p]);
                                }
                                kmac::<B>(&mut value, (split::<B>(t.0), split::<B>(t.1)), last);
                            }
                            ([], _) => unreachable!("monomials have factors"),
                        }
                    }
                    match w {
                        Some(ws) => {
                            // A multiplier operand below 2^64 per coordinate.
                            let (vr, vi) = (red_lazy::<B>(value.0), red_lazy::<B>(value.1));
                            kmac::<B>(&mut acc[i], (split::<B>(vr), split::<B>(vi)), ws);
                        }
                        None => acc[i] = value,
                    }
                }
            }
            steps += 1;
            if steps == flush_steps {
                steps = 0;
                flush::<B>(plan, full, acc, running);
            }
        }
        flush::<B>(plan, full, acc, running);
        for s in 0..plan.slots {
            for c in 0..comps {
                if plan.active(s, c, full) {
                    out[s * comps + c] += hsum_k::<B>(running[s * comps + c]);
                }
            }
        }
        if stop < k1 {
            let tail = weights.map(|w| (&w.0[stop - k0..], &w.1[stop - k0..]));
            eval_scalar(lanes, plan, tail, stop, k1, full, out);
        }
    }
}

/// Reduce the accumulators in use into the running sums (a function, not a closure:
/// closures do not inherit the entry point's target features).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn flush<B: crate::wide::Simd>(
    plan: &Plan,
    full: bool,
    acc: &mut [(crate::wide::Acc<B::V>, crate::wide::Acc<B::V>)],
    running: &mut [(B::V, B::V)],
) {
    use crate::wide::*;
    let comps = plan.components();
    // SAFETY: the caller guarantees the backend's features.
    unsafe {
        for s in 0..plan.slots {
            for c in 0..comps {
                if plan.active(s, c, full) {
                    let i = s * comps + c;
                    running[i] = addk::<B>(running[i], redk::<B>(acc[i]));
                    acc[i] = acck::<B>();
                }
            }
        }
    }
}
