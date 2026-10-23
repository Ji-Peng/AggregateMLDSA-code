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

use akita_algebra::split_eq::GruenSplitEq;
use jolt_field::{Field, Fold, One, Ring, Zero};
use jolt_poly::UnivariatePoly;
use rayon::prelude::*;

use crate::OpeningField as E;

/// Pairs per parallel task.
const PAIR_CHUNK: usize = 1 << 10;
/// Below this many pairs a round runs serially.
const MIN_PARALLEL_PAIRS: usize = 1 << 12;

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
    /// Folded at least once.
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

    #[inline]
    fn pair(&self, k: usize) -> (E, E) {
        match self {
            Self::Source(source) => source.pair(k),
            Self::Folded(cells) => (cells[2 * k], cells[2 * k + 1]),
        }
    }

    /// Bind the low variable to `challenge`.
    pub fn fold(&mut self, challenge: E) {
        match self {
            Self::Source(source) => *self = Self::Folded(source.fold(challenge)),
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

/// The lanes of one local domain and the instances batched on them.
pub struct Group<'a> {
    eq: Option<GruenSplitEq<E>>,
    lanes: Vec<Lane<'a>>,
    instances: Vec<GroupInstance>,
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
        let mut group = Self {
            eq,
            lanes,
            instances,
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

    /// Leading inactive rounds.
    #[must_use]
    pub fn inactive_rounds(&self) -> usize {
        self.inactive
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
        let half = self.lanes[0].len() / 2;
        let (eq_first, eq_second) = match &self.eq {
            Some(eq) => eq.remaining_eq_tables(),
            None => (&[][..], &[][..]),
        };
        let first_mask = eq_first.len().saturating_sub(1);
        let first_bits = eq_first.len().trailing_zeros();
        let lanes = &self.lanes;
        let instances = &self.instances;
        let lane_count = lanes.len();
        let evaluate_chunk = |start: usize, end: usize| {
            let mut acc = vec![E::zero(); points];
            let mut values = vec![E::zero(); lane_count * points];
            let mut local = vec![E::zero(); points];
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
                for slot in local.iter_mut() {
                    *slot = E::zero();
                }
                for instance in instances {
                    for term in &instance.terms {
                        for (x, slot) in local.iter_mut().enumerate() {
                            let at_x = &values[x * lane_count..(x + 1) * lane_count];
                            *slot += instance.gamma * term.evaluate(at_x);
                        }
                    }
                }
                let weight = if eq_first.is_empty() {
                    E::one()
                } else {
                    eq_first[k & first_mask] * eq_second[k >> first_bits]
                };
                for (slot, value) in acc.iter_mut().zip(&local) {
                    *slot += weight * *value;
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

    /// Bind the round's variable to `challenge`.
    pub fn bind(&mut self, round: usize, challenge: E) {
        assert_eq!(round, self.bound, "rounds in order");
        assert!(!self.pending.is_empty(), "round_message before bind");
        self.claim = evaluate_compressed(&self.pending, self.claim, challenge);
        self.pending.clear();
        if round < self.inactive {
            self.scale *= inverse_power_of_two(1);
        } else {
            for lane in &mut self.lanes {
                lane.fold(challenge);
            }
            if let Some(eq) = &mut self.eq {
                eq.bind(challenge);
            }
        }
        self.bound += 1;
    }

    /// The fully bound lane values, in lane order.
    #[must_use]
    pub fn terminal_values(&self) -> Vec<E> {
        assert_eq!(self.bound, self.inactive + self.local_vars, "all rounds bound");
        self.lanes.iter().map(Lane::terminal).collect()
    }
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
    use aerie_falcon::protocol::algebra::{eq_evaluation, eq_table};
    use agg_witness::gen::SplitMix64;

    fn digits(rng: &mut SplitMix64, len: usize) -> Vec<i8> {
        (0..len).map(|_| rng.below(16) as i8 - 8).collect()
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

    /// Run a group through a stage of `rounds` rounds with random challenges and check
    /// the sum-check invariants and the terminal equation.
    fn run(
        mut group: Group<'_>,
        instances: &[GroupInstance],
        rounds: usize,
        degree: usize,
        eq_point: Option<&[E]>,
        mut claim: E,
        rng: &mut SplitMix64,
    ) {
        if group.inactive_rounds() > 0 {
            assert_eq!(group.claim(), claim, "direct claim of a lifted group");
        }
        let mut challenges = Vec::new();
        for round in 0..rounds {
            let message = group.round_message(round);
            assert_eq!(message.len(), degree);
            assert_eq!(group.claim(), claim, "round {round}");
            let r = field(rng);
            claim = evaluate_compressed(&message, claim, r);
            group.bind(round, r);
            assert_eq!(group.claim(), claim);
            challenges.push(r);
        }
        let values = group.terminal_values();
        let local = &challenges[rounds - group.local_vars..];
        let eq_factor = eq_point.map_or(E::one(), |p| eq_evaluation(p, local));
        let expected = expected_terminal(instances, &values, eq_factor, group.inactive_rounds());
        assert_eq!(claim, expected);
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
}
