//! Field helpers of the kernel: a branch-free product in `K = F_p[u]/(u^2 - 2)`, the
//! unreduced split-equality contraction of Aerie's `sum_split_coefficients`, and small
//! polynomials in the round variable (coefficient form).

use jolt_field::{Fold, Ring, Unreduced, Zero};
use rayon::prelude::*;

use crate::{BaseField as F, OpeningField as E};

/// The unreduced product accumulator of `K` (exact sums of more than `2^62` products).
pub(crate) type Acc = <E as Unreduced>::Product;

/// `p = 2^64 - 59`.
const P: u64 = 0_u64.wrapping_sub(59);

/// `carry * 2^128 + hi * 2^64 + lo` modulo `p` (`carry <= 5`), canonical. `2^64 = 59` and
/// `2^128 = 3481` modulo `p`; the first fold is below `2^71`, the second below `2^64 + 2^13`.
#[inline(always)]
fn reduce3(lo: u64, hi: u64, carry: u64) -> u64 {
    let t = u128::from(hi) * 59 + u128::from(lo) + u128::from(carry) * 3481;
    let (t_lo, t_hi) = (t as u64, (t >> 64) as u64);
    let (s, overflow) = t_lo.overflowing_add(t_hi * 59);
    let s = s.wrapping_add(u64::from(overflow) * 59);
    s.min(s.wrapping_sub(P))
}

/// The product in `K` (non-residue 2) from four `64 x 64` products and two branch-free
/// reductions; equal to `lhs * rhs` (test `fast_product_matches_the_field_product`) and
/// faster (Aerie's `opening_field_ops::mul` is the model).
#[inline(always)]
pub(crate) fn fm(lhs: E, rhs: E) -> E {
    let a = lhs.c0().to_canonical_u64();
    let b = lhs.c1().to_canonical_u64();
    let c = rhs.c0().to_canonical_u64();
    let d = rhs.c1().to_canonical_u64();
    let ac = u128::from(a) * u128::from(c);
    let bd = u128::from(b) * u128::from(d);
    let ad = u128::from(a) * u128::from(d);
    let bc = u128::from(b) * u128::from(c);
    let (twice_bd, carry0) = bd.overflowing_add(bd);
    let (real, carry1) = ac.overflowing_add(twice_bd);
    let (imaginary, carry2) = ad.overflowing_add(bc);
    E::new(
        F::from_u64(reduce3(
            real as u64,
            (real >> 64) as u64,
            u64::from(carry0) + u64::from(carry1),
        )),
        F::from_u64(reduce3(
            imaginary as u64,
            (imaginary >> 64) as u64,
            u64::from(carry2),
        )),
    )
}

/// `a b + c d` in `K` from eight `64 x 64` products and one reduction per coordinate
/// (against two for [`fm`] twice plus an addition); equal to `a * b + c * d` (test
/// `fast_product_matches_the_field_product`).
#[inline(always)]
pub(crate) fn fm2(a: E, b: E, c: E, d: E) -> E {
    let wide = |x: E, y: E| {
        let (x0, x1) = (u128::from(x.c0().to_canonical_u64()), u128::from(x.c1().to_canonical_u64()));
        let (y0, y1) = (u128::from(y.c0().to_canonical_u64()), u128::from(y.c1().to_canonical_u64()));
        (x0 * y0, x1 * y1, x0 * y1, x1 * y0)
    };
    let (ac, bd, ad, bc) = wide(a, b);
    let (ac2, bd2, ad2, bc2) = wide(c, d);
    // Real part: ac + 2 bd + ac' + 2 bd', six terms below 2^128 each, so the carry out of
    // the low 128 bits is at most 5. Imaginary part: four terms, carry at most 3.
    let mut carry_real = 0_u64;
    let mut real = 0_u128;
    for term in [ac, bd, bd, ac2, bd2, bd2] {
        let (sum, overflow) = real.overflowing_add(term);
        real = sum;
        carry_real += u64::from(overflow);
    }
    let mut carry_imaginary = 0_u64;
    let mut imaginary = 0_u128;
    for term in [ad, bc, ad2, bc2] {
        let (sum, overflow) = imaginary.overflowing_add(term);
        imaginary = sum;
        carry_imaginary += u64::from(overflow);
    }
    E::new(
        F::from_u64(reduce3(real as u64, (real >> 64) as u64, carry_real)),
        F::from_u64(reduce3(imaginary as u64, (imaginary >> 64) as u64, carry_imaginary)),
    )
}

/// The canonical residue of a signed integer `|k| < p` (branch free).
#[inline(always)]
pub(crate) fn residue(k: i64) -> u64 {
    (k as u64).wrapping_add(((k >> 63) as u64) & P)
}

/// `w_0 k_0 + w_1 k_1 + w_2 k_2 + a` for canonical residues `k_t < p` and an addend `a =
/// (a_0, a_1)` of two coordinates below `2^66`: three `64 x 64` products per coordinate and
/// one reduction each (four terms below `2^128`, so the carry out of 128 bits is at most 3).
#[inline(always)]
pub(crate) fn dot3(w: [E; 3], k: [u64; 3], a: [u128; 2]) -> E {
    debug_assert!(a[0] < 1 << 66 && a[1] < 1 << 66);
    let coordinate = |c: usize| {
        let mut sum = a[c];
        let mut carry = 0_u64;
        for (w, &k) in w.iter().zip(&k) {
            let x = if c == 0 { w.c0() } else { w.c1() }.to_canonical_u64();
            let (next, overflow) = sum.overflowing_add(u128::from(x) * u128::from(k));
            sum = next;
            carry += u64::from(overflow);
        }
        F::from_u64(reduce3(sum as u64, (sum >> 64) as u64, carry))
    };
    E::new(coordinate(0), coordinate(1))
}

/// `x k` for a signed integer `|k| < p`: `k` is taken modulo `p` as a `u64` (branch free:
/// the signs of balanced digits are random), two `64 x 64` products and two reductions (half
/// of [`fm`]); equal to `x * E::from_i64(k)` (test `integer_product_matches`).
#[inline(always)]
pub(crate) fn fmi(x: E, k: i64) -> E {
    let u = u128::from((k as u64).wrapping_add(((k >> 63) as u64) & P));
    let a = u128::from(x.c0().to_canonical_u64()) * u;
    let b = u128::from(x.c1().to_canonical_u64()) * u;
    E::new(
        F::from_u64(reduce3(a as u64, (a >> 64) as u64, 0)),
        F::from_u64(reduce3(b as u64, (b >> 64) as u64, 0)),
    )
}

/// `acc += w k` without reduction for a signed integer `|k| < p`: `k` is taken modulo `p`
/// as a `u64` and each coordinate product (below `2^128`) is split into its two 64-bit
/// halves, so an accumulator absorbs `2^64` terms (two products, no reduction).
#[inline(always)]
pub(crate) fn mad_int(acc: &mut Acc, w: E, k: i64) {
    let u = u128::from((k as u64).wrapping_add(((k >> 63) as u64) & P));
    let a = u128::from(w.c0().to_canonical_u64()) * u;
    let b = u128::from(w.c1().to_canonical_u64()) * u;
    acc.0[0] += u128::from(a as u64);
    acc.0[1] += a >> 64;
    acc.0[2] += u128::from(b as u64);
    acc.0[3] += b >> 64;
}

/// `acc += w u` without reduction for a canonical `u < p` (two products, as [`mad_int`]).
#[inline(always)]
pub(super) fn mad_u64(acc: &mut Acc, w: E, u: u64) {
    let u = u128::from(u);
    let a = u128::from(w.c0().to_canonical_u64()) * u;
    let b = u128::from(w.c1().to_canonical_u64()) * u;
    acc.0[0] += u128::from(a as u64);
    acc.0[1] += a >> 64;
    acc.0[2] += u128::from(b as u64);
    acc.0[3] += b >> 64;
}

/// Add a signed integer `|k| < p` to an unreduced accumulator (its residue goes to the
/// low slot of the first coordinate).
#[inline(always)]
pub(super) fn lift_int(acc: &mut Acc, k: i64) {
    acc.0[0] += u128::from((k as u64).wrapping_add(((k >> 63) as u64) & P));
}

/// The canonical residue of a signed `|v| <= 2^64 + 2^63` (an exact `i128` product sum
/// that overflows `i64`; `carry` of [`reduce3`] stays zero for `|v| < 2^128`).
#[inline(always)]
pub(super) fn reduce_i128(v: i128) -> u64 {
    let magnitude = v.unsigned_abs();
    let residue = reduce3(magnitude as u64, (magnitude >> 64) as u64, 0);
    if v < 0 && residue != 0 {
        P - residue
    } else {
        residue
    }
}

/// `even + r (odd - even)`.
#[inline(always)]
pub(super) fn fold(context: &<E as Fold>::Ctx, even: E, odd: E) -> E {
    E::fold_one(context, even, odd)
}

/// Items per parallel task of the per-item passes.
pub(super) const ITEM_CHUNK: usize = 1 << 11;
/// Below this many items a pass runs serially.
pub(super) const MIN_PARALLEL_ITEMS: usize = 1 << 12;

/// Add a reduced element to an unreduced accumulator (its canonical coordinates go to the
/// low slots; the accumulator value is `lo + hi 2^64` per coordinate).
#[inline(always)]
pub(super) fn lift(acc: &mut Acc, value: E) {
    acc.0[0] += u128::from(value.c0().to_canonical_u64());
    acc.0[2] += u128::from(value.c1().to_canonical_u64());
}

/// `acc += x y` without reduction.
#[inline(always)]
pub(super) fn mad(acc: &mut Acc, x: E, y: E) {
    *acc += x.mul_unreduced(y);
}

/// `sum_k first[k & (|first| - 1)] second[k >> log|first|] c(k)` for the coefficient
/// vectors `c(k)` (`width` entries) that `item(k, scratch, c)` accumulates unreduced into
/// a zeroed slice. Each `c(k)` is reduced once, the inner sums over `first` are unreduced
/// (one `mul_unreduced` per coefficient and item), each block's sum is reduced once and
/// multiplied by `second[block]` unreduced. Field sums are exact, so the result equals
/// the plain sum.
pub(super) fn split_sum<S, I, G>(first: &[E], second: &[E], width: usize, init: I, item: G) -> Vec<E>
where
    I: Fn() -> S + Sync,
    G: Fn(usize, &mut S, &mut [Acc]) + Sync,
{
    let first_len = first.len();
    let blocks = second.len();
    let run = |range: std::ops::Range<usize>| {
        let mut scratch = init();
        let mut coefficients = vec![Acc::zero(); width];
        let mut outer = vec![Acc::zero(); width];
        let mut inner = vec![Acc::zero(); width];
        for block in range {
            for slot in &mut inner {
                *slot = Acc::zero();
            }
            let base = block * first_len;
            for (low, &weight) in first.iter().enumerate() {
                for slot in &mut coefficients {
                    *slot = Acc::zero();
                }
                item(base + low, &mut scratch, &mut coefficients);
                for (slot, &c) in inner.iter_mut().zip(&coefficients) {
                    *slot += weight.mul_unreduced(E::reduce_product(c));
                }
            }
            let scale = second[block];
            for (slot, &sum) in outer.iter_mut().zip(&inner) {
                *slot += scale.mul_unreduced(E::reduce_product(sum));
            }
        }
        outer
    };
    let add = |mut left: Vec<Acc>, right: Vec<Acc>| {
        for (slot, value) in left.iter_mut().zip(right) {
            *slot += value;
        }
        left
    };
    let total = if first_len * blocks >= MIN_PARALLEL_ITEMS && blocks > 1 {
        let per_task = ITEM_CHUNK.div_ceil(first_len).max(1);
        (0..blocks.div_ceil(per_task))
            .into_par_iter()
            .map(|task| run(task * per_task..((task + 1) * per_task).min(blocks)))
            .reduce(|| vec![Acc::zero(); width], add)
    } else {
        run(0..blocks)
    };
    total.into_iter().map(E::reduce_product).collect()
}

/// `sum_k c(k)` over `items` coefficient vectors (no weights), accumulated unreduced.
pub(super) fn plain_sum<S, I, G>(items: usize, width: usize, init: I, item: G) -> Vec<E>
where
    I: Fn() -> S + Sync,
    G: Fn(usize, &mut S, &mut [Acc]) + Sync,
{
    let run = |range: std::ops::Range<usize>| {
        let mut scratch = init();
        let mut sum = vec![Acc::zero(); width];
        for k in range {
            item(k, &mut scratch, &mut sum);
        }
        sum
    };
    let total = if items >= MIN_PARALLEL_ITEMS {
        (0..items.div_ceil(ITEM_CHUNK))
            .into_par_iter()
            .map(|task| run(task * ITEM_CHUNK..((task + 1) * ITEM_CHUNK).min(items)))
            .reduce(
                || vec![Acc::zero(); width],
                |mut left, right| {
                    for (slot, value) in left.iter_mut().zip(right) {
                        *slot += value;
                    }
                    left
                },
            )
    } else {
        run(0..items)
    };
    total.into_iter().map(E::reduce_product).collect()
}

/// Largest coefficient count of a round polynomial (degree 5 without an equality factor).
pub(super) const MAX_COEFFICIENTS: usize = 6;

/// A polynomial in the round variable `X` of degree below [`MAX_COEFFICIENTS`]
/// (coefficients, constant first).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Poly {
    pub c: [E; MAX_COEFFICIENTS],
    pub len: usize,
}

impl Poly {
    pub fn constant(value: E) -> Self {
        let mut c = [E::zero(); MAX_COEFFICIENTS];
        c[0] = value;
        Self { c, len: 1 }
    }

    /// `a + X d`.
    pub fn affine(a: E, d: E) -> Self {
        let mut c = [E::zero(); MAX_COEFFICIENTS];
        c[0] = a;
        c[1] = d;
        Self { c, len: 2 }
    }

    pub fn add(mut self, other: &Self) -> Self {
        for k in 0..other.len {
            self.c[k] += other.c[k];
        }
        self.len = self.len.max(other.len);
        self
    }

    pub fn add_constant(mut self, value: E) -> Self {
        self.c[0] += value;
        self
    }

    pub fn scale(mut self, factor: E) -> Self {
        for k in 0..self.len {
            self.c[k] = fm(self.c[k], factor);
        }
        self
    }

    pub fn mul(&self, other: &Self) -> Self {
        let mut c = [E::zero(); MAX_COEFFICIENTS];
        let len = self.len + other.len - 1;
        assert!(len <= MAX_COEFFICIENTS, "round polynomial degree overflow");
        for i in 0..self.len {
            for j in 0..other.len {
                c[i + j] += fm(self.c[i], other.c[j]);
            }
        }
        Self { c, len }
    }
}

/// The Taylor coefficients `P^(i)(a) / i!`, `i = 0..=D`, of `P(Y) = sum_m c[m] Y^m` at
/// `a` (synthetic division, `D (D + 1) / 2` products).
pub(super) fn taylor(c: &[E], a: E) -> [E; MAX_COEFFICIENTS] {
    let mut work = [E::zero(); MAX_COEFFICIENTS];
    work[..c.len()].copy_from_slice(c);
    let degree = c.len() - 1;
    for i in 0..degree {
        for m in (i..degree).rev() {
            work[m] += fm(work[m + 1], a);
        }
    }
    work
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::field;
    use agg_witness::gen::SplitMix64;
    use jolt_field::One;

    #[test]
    fn fast_product_matches_the_field_product() {
        let boundary = [0, 1, 2, 58, 59, 60, P / 2, P - 2, P - 1];
        for a in boundary {
            for b in boundary {
                for c in boundary {
                    for d in boundary {
                        let lhs = E::new(F::from_u64(a), F::from_u64(b));
                        let rhs = E::new(F::from_u64(c), F::from_u64(d));
                        assert_eq!(fm(lhs, rhs), lhs * rhs);
                        // All-maximal coordinates reach the largest carries of `fm2`.
                        assert_eq!(fm2(lhs, rhs, lhs, rhs), lhs * rhs + lhs * rhs);
                        assert_eq!(fm2(lhs, rhs, rhs, lhs), lhs * rhs + rhs * lhs);
                    }
                }
            }
        }
        let mut rng = SplitMix64::new(77);
        for _ in 0..100_000 {
            let a = field(&mut rng);
            let b = field(&mut rng);
            assert_eq!(fm(a, b), a * b);
            assert_eq!(fm(a, a), a * a);
            let (c, d) = (field(&mut rng), field(&mut rng));
            assert_eq!(fm2(a, b, c, d), a * b + c * d);
            let k = [rng.next_u64() % P, rng.next_u64() % P, P - 1];
            let addend = [rng.next_u64() % P, P - 1];
            let expected = a * E::from_u64(k[0]) + b * E::from_u64(k[1]) + c * E::from_u64(k[2])
                + E::new(F::from_u64(addend[0]), F::from_u64(addend[1]));
            assert_eq!(dot3([a, b, c], k, addend.map(u128::from)), expected);
            let big = [u128::from(P - 1) * 3, u128::from(P - 1) + 5];
            let expected = a * E::from_u64(P - 1) + b * E::from_u64(P - 1) + c * E::from_u64(P - 1)
                + E::new(F::from_u64(P - 1) * F::from_u64(3), F::from_u64(P - 1) + F::from_u64(5));
            assert_eq!(dot3([a, b, c], [P - 1; 3], big), expected);
            let i = (rng.next_u64() >> 2) as i64 - (1 << 61);
            assert_eq!(E::from_u64(residue(i)), E::from_i64(i));
        }
    }

    #[test]
    fn unreduced_integer_products_sum_exactly() {
        let mut rng = SplitMix64::new(92);
        let mut acc = Acc::zero();
        let mut expected = E::zero();
        for i in 0..2000 {
            let w = field(&mut rng);
            let k = match i % 4 {
                0 => -(rng.next_u64() as i64 >> 1),
                1 => rng.next_u64() as i64 >> 2,
                2 => -1,
                _ => 0,
            };
            mad_int(&mut acc, w, k);
            expected += w * E::from_i64(k);
        }
        assert_eq!(E::reduce_product(acc), expected);
    }

    #[test]
    fn integer_product_matches() {
        let mut rng = SplitMix64::new(91);
        for k in [0_i64, 1, -1, 59, -59, i64::MAX, -i64::MAX + 60, 1 << 62, -(1 << 62), 123_456_789_012_345] {
            let x = field(&mut rng);
            assert_eq!(fmi(x, k), x * E::from_i64(k), "k = {k}");
        }
        for _ in 0..1000 {
            let x = field(&mut rng);
            let k = rng.next_u64() as i64 >> (rng.next_u64() % 64);
            if k == i64::MIN {
                continue;
            }
            assert_eq!(fmi(x, k), x * E::from_i64(k));
        }
    }

    #[test]
    fn split_sums_match_the_nested_sum() {
        let mut rng = SplitMix64::new(78);
        for (first_len, second_len, width) in [(1, 1, 3), (1, 64, 4), (8, 1, 5), (16, 512, 6), (2048, 4, 2)] {
            let first: Vec<E> = (0..first_len).map(|_| field(&mut rng)).collect();
            let second: Vec<E> = (0..second_len).map(|_| field(&mut rng)).collect();
            let values: Vec<E> = (0..first_len * second_len * width).map(|_| field(&mut rng)).collect();
            let item = |k: usize, _: &mut (), out: &mut [Acc]| {
                for (i, slot) in out.iter_mut().enumerate() {
                    lift(slot, values[k * width + i]);
                    mad(slot, values[k * width + i], E::from_u64(2));
                }
            };
            let mut expected = vec![E::zero(); width];
            for k in 0..first_len * second_len {
                let w = first[k % first_len] * second[k / first_len];
                for (i, slot) in expected.iter_mut().enumerate() {
                    *slot += w * values[k * width + i] * E::from_u64(3);
                }
            }
            assert_eq!(split_sum(&first, &second, width, || (), item), expected);
            let plain: Vec<E> = (0..width)
                .map(|i| (0..first_len * second_len).fold(E::zero(), |s, k| s + values[k * width + i] * E::from_u64(3)))
                .collect();
            assert_eq!(plain_sum(first_len * second_len, width, || (), item), plain);

        }
    }

    #[test]
    fn taylor_coefficients_shift_the_polynomial() {
        let mut rng = SplitMix64::new(79);
        for degree in 1..=5 {
            let c: Vec<E> = (0..=degree).map(|_| field(&mut rng)).collect();
            let (a, d, x) = (field(&mut rng), field(&mut rng), field(&mut rng));
            let t = taylor(&c, a);
            let direct = c.iter().rev().fold(E::zero(), |acc, &k| acc * (a + x * d) + k);
            let shifted = (0..=degree).rev().fold(E::zero(), |acc, i| acc * (x * d) + t[i]);
            assert_eq!(direct, shifted, "degree {degree}");
        }
        let p = Poly::affine(E::one(), E::from_u64(2)).mul(&Poly::affine(E::from_u64(3), E::one()));
        assert_eq!(p.len, 3);
        assert_eq!(&p.c[..3], &[E::from_u64(3), E::from_u64(7), E::from_u64(2)]);
    }
}
