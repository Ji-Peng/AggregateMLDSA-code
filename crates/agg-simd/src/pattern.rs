//! The Boolean-pattern prefix of a sum-check over integer leaves: one pass over the leaf
//! streams gives the sums that serve the first two rounds, and a second pass materialises
//! the lanes folded by the two bound challenges as `K` planes.
//!
//! Cells are indexed `y = a + 4 hi` with the pattern `a` the two low variables. With the
//! equality weights `w(hi) = first[hi mod |first|] second[hi / |first|]` of the remaining
//! variables, the pass accumulates `total = sum w`, per linear form `A[a] = sum_hi w L(a, hi)`
//! and per product form `A[a, b] = sum_hi w L_i(a, hi) L_j(b, hi)`, index `4a + b`. Only raw
//! integer lane values are multiplied, exactly in `i64`.
//!
//! Two leaf shapes (`docs/lookup.md`): the stack *bottom* with the lanes `v_0, v_1` (the two
//! halves of the stacked column; linear form `v_0 + v_1`, product form `v_0 v_1`) and the
//! *symmetric* layer above it with `S_c = a_c + b_c`, `M_c = a_c b_c` of the centred leaves
//! `a_c = v(y + c q) - centre`, `b_c = v(y + c q + 2q) - centre`, `q` a quarter of the column
//! (linear forms `S_0 + S_1`, `M_0 + M_1`; product forms `S_0 S_1`, `S_0 M_1 + S_1 M_0`,
//! `M_0 M_1`). Ranges: bottom `|v| < 2^20`; symmetric `|a|, |b| <= 2^15`, so `|S| <= 2^16`,
//! `|M| <= 2^30`, `|S_0 S_1| <= 2^32`, `|S M| <= 2^46`, `|M_0 M_1| <= 2^60` (two limbs).
use crate::{
    detect::{backend, Backend},
    fused::OutLane,
    K,
};
use jolt_field::{Ring, Zero};
use rayon::prelude::*;

/// The sums of one pass (the forms in the shape's order).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefixSums {
    /// `sum_hi w(hi)`.
    pub total: K,
    /// Per linear form, by pattern.
    pub linear: Vec<[K; 4]>,
    /// Per product form, by pattern pair `4a + b`.
    pub products: Vec<[K; 16]>,
}

impl PrefixSums {
    fn zero(linear: usize, products: usize) -> Self {
        Self {
            total: K::zero(),
            linear: vec![[K::zero(); 4]; linear],
            products: vec![[K::zero(); 16]; products],
        }
    }

    fn add(mut self, other: &Self) -> Self {
        self.total += other.total;
        for (a, b) in self.linear.iter_mut().zip(&other.linear) {
            for (x, y) in a.iter_mut().zip(b) {
                *x += *y;
            }
        }
        for (a, b) in self.products.iter_mut().zip(&other.products) {
            for (x, y) in a.iter_mut().zip(b) {
                *x += *y;
            }
        }
        self
    }
}

/// The integer lanes of a leaf shape on one cell.
pub trait Leaves: Sync {
    /// Lanes per cell.
    const LANES: usize;
    /// Linear forms: the lanes summed.
    const LINEAR: &'static [&'static [usize]];
    /// Product forms: the lane pairs summed, and whether the sum needs two limbs
    /// (`|value| >= 2^51`).
    const PRODUCTS: &'static [(&'static [(usize, usize)], bool)];
    /// Cells of the shape.
    fn rows(&self) -> usize;
    /// The lane values of cell `y`.
    fn values(&self, y: usize) -> [i64; 4];
    /// The lane values of the eight cells `4 (row + l) + a`, `l = 0..8`, as `i64` lanes,
    /// indexed `[lane][pattern a]`.
    ///
    /// # Safety
    /// Requires AVX-512F/DQ/BW/VL/IFMA.
    #[cfg(target_arch = "x86_64")]
    unsafe fn values8(&self, row: usize) -> [[std::arch::x86_64::__m512i; 4]; 4];
}

/// The stack bottom: `v_0, v_1`.
pub struct Bottom<'a> {
    pub v0: &'a [i32],
    pub v1: &'a [i32],
}

impl Leaves for Bottom<'_> {
    const LANES: usize = 2;
    const LINEAR: &'static [&'static [usize]] = &[&[0, 1]];
    const PRODUCTS: &'static [(&'static [(usize, usize)], bool)] = &[(&[(0, 1)], false)];
    fn rows(&self) -> usize {
        self.v0.len()
    }
    #[inline(always)]
    fn values(&self, y: usize) -> [i64; 4] {
        [i64::from(self.v0[y]), i64::from(self.v1[y]), 0, 0]
    }
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
    unsafe fn values8(&self, row: usize) -> [[std::arch::x86_64::__m512i; 4]; 4] {
        // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
        unsafe {
            use crate::x86::*;
            let v0 = self.v0.as_ptr().add(4 * row);
            let v1 = self.v1.as_ptr().add(4 * row);
            let mut out = [[z(); 4]; 4];
            for a in 0..4 {
                out[0][a] = strided4_i32(v0, a as i32);
                out[1][a] = strided4_i32(v1, a as i32);
            }
            out
        }
    }
}

/// The symmetric layer: `S_0, S_1, M_0, M_1` from the four quarters of the stacked column.
pub struct Symmetric<'a> {
    pub stacked: &'a [i32],
    pub center: i32,
}

impl Leaves for Symmetric<'_> {
    const LANES: usize = 4;
    const LINEAR: &'static [&'static [usize]] = &[&[0, 1], &[2, 3]];
    const PRODUCTS: &'static [(&'static [(usize, usize)], bool)] =
        &[(&[(0, 1)], false), (&[(0, 3), (1, 2)], false), (&[(2, 3)], true)];
    fn rows(&self) -> usize {
        self.stacked.len() / 4
    }
    #[inline(always)]
    fn values(&self, y: usize) -> [i64; 4] {
        let q = self.stacked.len() / 4;
        let v = |j: usize| i64::from(self.stacked[y + j * q]) - i64::from(self.center);
        let (a0, b0, a1, b1) = (v(0), v(2), v(1), v(3));
        [a0 + b0, a1 + b1, a0 * b0, a1 * b1]
    }
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
    unsafe fn values8(&self, row: usize) -> [[std::arch::x86_64::__m512i; 4]; 4] {
        // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
        unsafe {
            use crate::x86::*;
            use std::arch::x86_64::*;
            let q = self.stacked.len() / 4;
            let c = bc(i64::from(self.center) as u64);
            let mut out = [[z(); 4]; 4];
            for a in 0..4 {
                let v = |j: usize| sub(strided4_i32(self.stacked.as_ptr().add(4 * row + j * q), a as i32), c);
                let (a0, b0, a1, b1) = (v(0), v(2), v(1), v(3));
                // |a|, |b| <= 2^15: sums below 2^17, products below 2^31, exact in i64.
                out[0][a] = add(a0, b0);
                out[1][a] = add(a1, b1);
                out[2][a] = _mm512_mullo_epi64(a0, b0);
                out[3][a] = _mm512_mullo_epi64(a1, b1);
            }
            out
        }
    }
}

/// The prefix sums of `leaves` with the weights `first`, `second` (`4 |first| |second|`
/// cells).
pub fn prefix_sums<L: Leaves>(leaves: &L, first: &[K], second: &[K]) -> PrefixSums {
    assert_eq!(leaves.rows(), 4 * first.len() * second.len(), "weights cover the cells");
    assert!(first.len().is_power_of_two());
    let blocks = second.len();
    let run = |range: std::ops::Range<usize>| -> PrefixSums {
        #[cfg(target_arch = "x86_64")]
        if first.len() >= 8 && backend() == Backend::Avx512 {
            let planes = crate::buf::KPlanes::from_fn(first.len(), |i| first[i]);
            // SAFETY: the call is guarded by the backend check and the length assertions above.
            return unsafe { sums_vector(leaves, planes.planes(), second, range) };
        }
        sums_scalar(leaves, first, second, range)
    };
    if blocks >= 2 && leaves.rows() >= 1 << 12 {
        let tasks = blocks.min(rayon::current_num_threads() * 4);
        let per = blocks.div_ceil(tasks);
        (0..tasks)
            .into_par_iter()
            .map(|t| run(t * per..((t + 1) * per).min(blocks)))
            .reduce(
                || PrefixSums::zero(L::LINEAR.len(), L::PRODUCTS.len()),
                |a, b| a.add(&b),
            )
    } else {
        run(0..blocks)
    }
}

fn sums_scalar<L: Leaves>(leaves: &L, first: &[K], second: &[K], blocks: std::ops::Range<usize>) -> PrefixSums {
    let f = first.len();
    let mut out = PrefixSums::zero(L::LINEAR.len(), L::PRODUCTS.len());
    for block in blocks {
        let mut sums = PrefixSums::zero(L::LINEAR.len(), L::PRODUCTS.len());
        for i in 0..f {
            let hi = block * f + i;
            let w = first[i];
            sums.total += w;
            let cells: [[i64; 4]; 4] = std::array::from_fn(|a| leaves.values(4 * hi + a));
            for (form, lanes) in L::LINEAR.iter().enumerate() {
                for a in 0..4 {
                    let value = lanes.iter().map(|&l| cells[a][l]).sum::<i64>();
                    sums.linear[form][a] += w * K::from_i64(value);
                }
            }
            for (form, (pairs, _)) in L::PRODUCTS.iter().enumerate() {
                for a in 0..4 {
                    for b in 0..4 {
                        let value = pairs.iter().map(|&(i, j)| cells[a][i] * cells[b][j]).sum::<i64>();
                        sums.products[form][4 * a + b] += w * K::from_i64(value);
                    }
                }
            }
        }
        let s = second[block];
        sums.total *= s;
        for a in sums.linear.iter_mut().flatten() {
            *a *= s;
        }
        for a in sums.products.iter_mut().flatten() {
            *a *= s;
        }
        out = out.add(&sums);
    }
    out
}

/// `out[l][hi] = sum_a e[a] L_l(a, hi)` for `hi` in `row0..row1` (the lanes folded by the
/// two bound challenges, `e[a] = eq(rho_0, a_0) eq(rho_1, a_1)`).
///
/// # Safety
/// Every output address must hold `row1` writable `u64` rows.
pub unsafe fn fill_rows<L: Leaves>(leaves: &L, e: [K; 4], out: &[OutLane], row0: usize, row1: usize) {
    assert_eq!(out.len(), L::LANES);
    #[cfg(target_arch = "x86_64")]
    if row1 - row0 >= 8 && backend() == Backend::Avx512 {
        // SAFETY: the caller checked the IFMA backend, and every slice or pointer is valid for the length asserted above.
        unsafe {
            return fill_vector(leaves, e, out, row0, row1);
        }
    }
    // SAFETY: the caller checked the IFMA backend, and every slice or pointer is valid for the length asserted above.
    unsafe { fill_scalar(leaves, e, out, row0, row1) }
}

unsafe fn fill_scalar<L: Leaves>(leaves: &L, e: [K; 4], out: &[OutLane], row0: usize, row1: usize) {
    for hi in row0..row1 {
        let cells: [[i64; 4]; 4] = std::array::from_fn(|a| leaves.values(4 * hi + a));
        for (l, &o) in out.iter().enumerate() {
            let value = (0..4).fold(K::zero(), |acc, a| acc + e[a] * K::from_i64(cells[a][l]));
            // SAFETY: the caller checked the IFMA backend, and every slice or pointer is valid for the length asserted above.
            unsafe {
                (o.0 as *mut u64).add(hi).write(value.c0().to_canonical_u64());
                (o.1 as *mut u64).add(hi).write(value.c1().to_canonical_u64());
            }
        }
    }
}

/// `out[row] = sum_l sum_a weights[l][a] L_l(a, row)` for `row` in `row0..row1`: one lane
/// combining every lane of the leaves folded by the two bound challenges with its
/// coefficient (`weights[l][a] = c_l eq(rho_0, a_0) eq(rho_1, a_1)`).
///
/// # Safety
/// The output address must hold `row1` writable `u64` rows.
pub unsafe fn fill_combined<L: Leaves>(leaves: &L, weights: &[[K; 4]], out: OutLane, row0: usize, row1: usize) {
    assert_eq!(weights.len(), L::LANES);
    #[cfg(target_arch = "x86_64")]
    if row1 - row0 >= 8 && backend() == Backend::Avx512 {
        // SAFETY: the caller checked the IFMA backend, and every slice or pointer is valid for the length asserted above.
        unsafe {
            return fill_combined_vector(leaves, weights, out, row0, row1);
        }
    }
    // SAFETY: the caller checked the IFMA backend, and every slice or pointer is valid for the length asserted above.
    unsafe { fill_combined_scalar(leaves, weights, out, row0, row1) }
}

unsafe fn fill_combined_scalar<L: Leaves>(leaves: &L, weights: &[[K; 4]], out: OutLane, row0: usize, row1: usize) {
    for row in row0..row1 {
        let mut value = K::zero();
        for a in 0..4 {
            let cells = leaves.values(4 * row + a);
            for (l, w) in weights.iter().enumerate() {
                value += w[a] * K::from_i64(cells[l]);
            }
        }
        // SAFETY: the caller checked the IFMA backend, and every slice or pointer is valid for the length asserted above.
        unsafe {
            (out.0 as *mut u64).add(row).write(value.c0().to_canonical_u64());
            (out.1 as *mut u64).add(row).write(value.c1().to_canonical_u64());
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
unsafe fn fill_combined_vector<L: Leaves>(leaves: &L, weights: &[[K; 4]], out: OutLane, row0: usize, row1: usize) {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
    unsafe {
        use crate::x86::*;
        let stop = row0 + (row1 - row0) / 8 * 8;
        let split: Vec<[(L52, L52); 4]> = weights.iter().map(|w| std::array::from_fn(|a| splitk(bck(w[a])))).collect();
        // Every one-limb term is shifted by 2^51; the shift is undone with the sum of the
        // weights (`4 LANES` terms per output, far below the bound).
        let total = weights.iter().flatten().fold(K::zero(), |s, &x| s + x);
        let correction = bck(K::from_u64(ONE_LIMB_SHIFT) * total);
        let shift = bc(ONE_LIMB_SHIFT);
        for row in (row0..stop).step_by(8) {
            let cells = leaves.values8(row);
            let mut acc = a3k();
            for (l, w) in split.iter().enumerate() {
                for a in 0..4 {
                    let k = add(cells[l][a], shift);
                    mac_int(&mut acc.0, w[a].0, k);
                    mac_int(&mut acc.1, w[a].1, k);
                }
            }
            let v = redk(acc);
            store((out.0 as *mut u64).add(row), sub_mod(v.0, correction.0));
            store((out.1 as *mut u64).add(row), sub_mod(v.1, correction.1));
        }
        fill_combined_scalar(leaves, weights, out, stop, row1);
    }
}

/// Up to four integer columns read linearly (an affine combination with `K` coefficients):
/// the linear forms are the lanes themselves and there are no products; missing lanes are
/// zero.
pub struct Columns<'a> {
    pub lanes: Vec<&'a [i32]>,
}

impl Leaves for Columns<'_> {
    const LANES: usize = 4;
    const LINEAR: &'static [&'static [usize]] = &[&[0], &[1], &[2], &[3]];
    const PRODUCTS: &'static [(&'static [(usize, usize)], bool)] = &[];
    fn rows(&self) -> usize {
        self.lanes[0].len()
    }
    #[inline(always)]
    fn values(&self, y: usize) -> [i64; 4] {
        std::array::from_fn(|l| self.lanes.get(l).map_or(0, |lane| i64::from(lane[y])))
    }
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
    unsafe fn values8(&self, row: usize) -> [[std::arch::x86_64::__m512i; 4]; 4] {
        // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
        unsafe {
            use crate::x86::*;
            let mut out = [[z(); 4]; 4];
            for (l, lane) in self.lanes.iter().enumerate() {
                for a in 0..4 {
                    out[l][a] = strided4_i32(lane.as_ptr().add(4 * row), a as i32);
                }
            }
            out
        }
    }
}

/// [`Columns`] over `i8` cells (committed digit columns read through the identity map).
pub struct ByteColumns<'a> {
    pub lanes: Vec<&'a [i8]>,
}

impl Leaves for ByteColumns<'_> {
    const LANES: usize = 4;
    const LINEAR: &'static [&'static [usize]] = &[&[0], &[1], &[2], &[3]];
    const PRODUCTS: &'static [(&'static [(usize, usize)], bool)] = &[];
    fn rows(&self) -> usize {
        self.lanes[0].len()
    }
    #[inline(always)]
    fn values(&self, y: usize) -> [i64; 4] {
        std::array::from_fn(|l| self.lanes.get(l).map_or(0, |lane| i64::from(lane[y])))
    }
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
    unsafe fn values8(&self, row: usize) -> [[std::arch::x86_64::__m512i; 4]; 4] {
        // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
        unsafe {
            use crate::x86::*;
            let mut out = [[z(); 4]; 4];
            for (l, lane) in self.lanes.iter().enumerate() {
                for a in 0..4 {
                    out[l][a] = strided4_i8(lane.as_ptr().add(4 * row), a as i32);
                }
            }
            out
        }
    }
}

/// Unreduced sums of one shape, in memory: `total`, linear forms by pattern, product forms
/// by pattern pair (one-limb and, for two-limb forms, the high limb).
#[cfg(target_arch = "x86_64")]
struct Accs {
    total: (crate::x86::A3, crate::x86::A3),
    linear: [[(crate::x86::A3, crate::x86::A3); 4]; 4],
    products: [[(crate::x86::A3, crate::x86::A3); 16]; 4],
    high: [[(crate::x86::A3, crate::x86::A3); 16]; 4],
}

/// One-limb integer terms are shifted by `2^51` into `[0, 2^52)`; the high limb of a two-limb
/// term (`k >> 52`, `|k| < 2^63`) by `2^11`. Each shift is undone once with `total`.
const ONE_LIMB_SHIFT: u64 = 1 << 51;
const HIGH_SHIFT: u64 = 1 << 11;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
#[allow(clippy::too_many_lines)]
unsafe fn sums_vector<L: Leaves>(
    leaves: &L,
    first: (&[u64], &[u64]),
    second: &[K],
    blocks: std::ops::Range<usize>,
) -> PrefixSums {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
    unsafe {
        use crate::x86::*;
        use std::arch::x86_64::*;
        let f = first.0.len();
        let (nl, np) = (L::LINEAR.len(), L::PRODUCTS.len());
        let mut out = PrefixSums::zero(nl, np);
        let one_shift = bc(ONE_LIMB_SHIFT);
        let high_shift = bc(HIGH_SHIFT);
        let mask52 = bc((1 << 52) - 1);
        // Per block: unreduced accumulators flushed into canonical lane sums every 512
        // steps (the `mac_int` bound), then the lane sums times `second[block]`.
        let mut acc = Accs {
            total: a3k(),
            linear: [[a3k(); 4]; 4],
            products: [[a3k(); 16]; 4],
            high: [[a3k(); 16]; 4],
        };
        let mut run_total = (z(), z());
        let mut run_linear = [[(z(), z()); 4]; 4];
        let mut run_products = [[(z(), z()); 16]; 4];
        let mut run_high = [[(z(), z()); 16]; 4];
        for block in blocks {
            let mut steps = 0_usize;
            let flush = |acc: &mut Accs,
                             run_total: &mut (V, V),
                             run_linear: &mut [[(V, V); 4]; 4],
                             run_products: &mut [[(V, V); 16]; 4],
                             run_high: &mut [[(V, V); 16]; 4]| {
                *run_total = addk(*run_total, redk(acc.total));
                acc.total = a3k();
                for form in 0..nl {
                    for a in 0..4 {
                        run_linear[form][a] = addk(run_linear[form][a], redk(acc.linear[form][a]));
                        acc.linear[form][a] = a3k();
                    }
                }
                for (form, &(_, two)) in L::PRODUCTS.iter().enumerate() {
                    for ab in 0..16 {
                        run_products[form][ab] = addk(run_products[form][ab], redk(acc.products[form][ab]));
                        acc.products[form][ab] = a3k();
                        if two {
                            run_high[form][ab] = addk(run_high[form][ab], redk(acc.high[form][ab]));
                            acc.high[form][ab] = a3k();
                        }
                    }
                }
            };
            for i in (0..f).step_by(8) {
                let row = block * f + i;
                let w = splitk(loadk(first, i));
                lift52(&mut acc.total.0, w.0);
                lift52(&mut acc.total.1, w.1);
                let cells = leaves.values8(row);
                for (form, lanes) in L::LINEAR.iter().enumerate() {
                    for a in 0..4 {
                        let mut k = z();
                        for &l in lanes.iter() {
                            k = add(k, cells[l][a]);
                        }
                        // |k| < 2^51: shifted into [0, 2^52).
                        let k = add(k, one_shift);
                        mac_int(&mut acc.linear[form][a].0, w.0, k);
                        mac_int(&mut acc.linear[form][a].1, w.1, k);
                    }
                }
                for (form, &(pairs, two)) in L::PRODUCTS.iter().enumerate() {
                    for a in 0..4 {
                        for b in 0..4 {
                            let mut k = z();
                            for &(l, m) in pairs.iter() {
                                k = add(k, _mm512_mullo_epi64(cells[l][a], cells[m][b]));
                            }
                            let slot = &mut acc.products[form][4 * a + b];
                            if two {
                                // k = lo + 2^52 hi, lo in [0, 2^52), |hi| < 2^11 shifted by 2^11.
                                let lo = and(k, mask52);
                                let high_limb = add(_mm512_srai_epi64::<52>(k), high_shift);
                                mac_int(&mut slot.0, w.0, lo);
                                mac_int(&mut slot.1, w.1, lo);
                                let high = &mut acc.high[form][4 * a + b];
                                mac_int(&mut high.0, w.0, high_limb);
                                mac_int(&mut high.1, w.1, high_limb);
                            } else {
                                let k = add(k, one_shift);
                                mac_int(&mut slot.0, w.0, k);
                                mac_int(&mut slot.1, w.1, k);
                            }
                        }
                    }
                }
                steps += 1;
                if steps == 512 {
                    steps = 0;
                    flush(&mut acc, &mut run_total, &mut run_linear, &mut run_products, &mut run_high);
                }
            }
            flush(&mut acc, &mut run_total, &mut run_linear, &mut run_products, &mut run_high);
            // The block's lane sums times `second[block]`, then the lanes summed.
            let s = bck(second[block]);
            let total = hsum_k(mulk(run_total, s));
            out.total += total;
            run_total = (z(), z());
            let one = K::from_u64(ONE_LIMB_SHIFT) * total;
            let high = K::from_u64(HIGH_SHIFT) * total;
            let limb = K::from_u64(1 << 52);
            for form in 0..nl {
                for a in 0..4 {
                    out.linear[form][a] += hsum_k(mulk(run_linear[form][a], s)) - one;
                    run_linear[form][a] = (z(), z());
                }
            }
            for (form, &(_, two)) in L::PRODUCTS.iter().enumerate() {
                for ab in 0..16 {
                    let low = hsum_k(mulk(run_products[form][ab], s));
                    run_products[form][ab] = (z(), z());
                    out.products[form][ab] += if two {
                        let h = hsum_k(mulk(run_high[form][ab], s)) - high;
                        run_high[form][ab] = (z(), z());
                        low + limb * h
                    } else {
                        low - one
                    };
                }
            }
        }
        out
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
unsafe fn fill_vector<L: Leaves>(leaves: &L, e: [K; 4], out: &[OutLane], row0: usize, row1: usize) {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
    unsafe {
        use crate::x86::*;
        let stop = row0 + (row1 - row0) / 8 * 8;
        let weights: [(L52, L52); 4] = std::array::from_fn(|a| splitk(bck(e[a])));
        // sum_a e_a (k_a + 2^51) = L + 2^51 sum_a e_a: the shift is undone with the sum of
        // the pattern weights (one in exact arithmetic, computed here regardless).
        let correction = bck(K::from_u64(ONE_LIMB_SHIFT) * e.iter().fold(K::zero(), |s, &x| s + x));
        let shift = bc(ONE_LIMB_SHIFT);
        for row in (row0..stop).step_by(8) {
            let cells = leaves.values8(row);
            for (l, &o) in out.iter().enumerate() {
                let mut acc = a3k();
                for a in 0..4 {
                    // Four one-limb terms per output, far below the bound.
                    let k = add(cells[l][a], shift);
                    mac_int(&mut acc.0, weights[a].0, k);
                    mac_int(&mut acc.1, weights[a].1, k);
                }
                let v = redk(acc);
                store((o.0 as *mut u64).add(row), sub_mod(v.0, correction.0));
                store((o.1 as *mut u64).add(row), sub_mod(v.1, correction.1));
            }
        }
        fill_scalar(leaves, e, out, stop, row1);
    }
}
