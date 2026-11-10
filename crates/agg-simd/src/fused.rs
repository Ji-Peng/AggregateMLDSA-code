//! Sum-check rounds on lane-major `K` planes: the fold of a table by the bound challenge and
//! the evaluation of the next round's inner polynomial on the folded pairs.
//!
//! A table is a list of lanes, each a pair of canonical coordinate planes over the same
//! rows. The summand is a bilinear form with coefficient slots,
//! `f = sum_slot c_slot (sum_{(l, slot)} L_l + sum_{(i, j, slot)} L_i L_j)`; the kernels
//! return the slot sums unweighted by `c_slot`, so the caller applies the coefficients once
//! per call. Field sums are exact, so every grouping of the terms gives the same element.
use crate::{
    buf::AlignedVec,
    detect::{backend, Backend},
    k::Planes,
    scalar, K,
};
use jolt_field::{Ring, Zero};

/// The term lists of a summand: `(lane, slot)` and `(lane_i, lane_j, slot)`.
#[derive(Clone, Copy, Debug)]
pub struct Terms<'a> {
    pub linear: &'a [(usize, usize)],
    pub products: &'a [(usize, usize, usize)],
    pub slots: usize,
}

/// The two coordinate planes of an output lane as raw addresses: row `i` of coordinate `c`
/// is the `u64` at `address_c + 8 i`. Tasks write disjoint row ranges of the same planes.
pub type OutLane = (usize, usize);

/// Most lanes a table may hold (the lane masks of the evaluation kernels).
pub const MAX_LANES: usize = 128;
/// Most coefficient slots of a summand (the stage-3 main group under H-FIPS of ML-DSA-87
/// takes 151).
pub const MAX_SLOTS: usize = 192;
/// Most terms (linear and product together) of one slot. The IFMA evaluation sums a slot's
/// terms on one pair of rows in one unreduced accumulator: each term adds below `2^53` to
/// its low limb and below `6 * 2^52` to its middle limb, so 512 terms keep them below `2^62`
/// and `3 * 2^62`.
pub const MAX_SLOT_TERMS: usize = 512;

/// # Safety
/// Both planes of `out` must hold a writable row `row`.
#[inline]
unsafe fn write(out: OutLane, row: usize, x: K) {
    // SAFETY: the caller guarantees row `row` of both planes is writable.
    unsafe {
        (out.0 as *mut u64).add(row).write(x.c0().to_canonical_u64());
        (out.1 as *mut u64).add(row).write(x.c1().to_canonical_u64());
    }
}

/// `out[l][row] = in[l][2 row] + r (in[l][2 row + 1] - in[l][2 row])` for every lane and
/// `row` in `row0..row1`.
///
/// # Safety
/// Every output address must hold `row1` writable `u64` rows not aliased by `lanes`.
///
/// # Panics
/// When the lane counts differ, `row0 > row1`, or an input plane holds fewer than `2 row1`
/// rows (the vector kernels read them through raw pointers).
pub unsafe fn fold_rows(r: K, lanes: &[Planes<'_>], out: &[OutLane], row0: usize, row1: usize) {
    assert_eq!(lanes.len(), out.len());
    assert!(row0 <= row1);
    assert!(
        lanes.iter().all(|lane| lane.0.len() >= 2 * row1 && lane.1.len() >= 2 * row1),
        "input lanes cover the folded rows"
    );
    #[cfg(target_arch = "x86_64")]
    if row1 - row0 >= 8 && backend() == Backend::Avx512 {
        // SAFETY: the IFMA backend is active; the inputs hold `2 row1` rows (asserted) and the
        // caller guarantees `row1` writable output rows.
        unsafe {
            return fold_vector(r, lanes, out, row0, row1);
        }
    }
    #[cfg(target_arch = "x86_64")]
    if row1 - row0 >= 8 {
        if let Some(()) = crate::wide::dispatch!(fold_w8(r, lanes, out, row0, row1), fold_w4(r, lanes, out, row0, row1)) {
            return;
        }
    }
    // SAFETY: the caller guarantees `row1` writable output rows; the inputs are read through
    // checked indexing.
    unsafe { fold_scalar(r, lanes, out, row0, row1) }
}

/// [`fold_rows`] on the backends without IFMA: one reduced `K` product per row.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn fold_wide<B: crate::wide::Simd>(r: K, lanes: &[Planes<'_>], out: &[OutLane], row0: usize, row1: usize) {
    use crate::wide::*;
    let l = B::LANES;
    let stop = row0 + (row1 - row0) / l * l;
    // SAFETY: the caller guarantees the backend's features, `2 row1` readable rows per input
    // lane and `row1` writable rows per output lane.
    unsafe {
        let rs = split_k::<B>(bck::<B>(r));
        for (lane, &o) in lanes.iter().zip(out) {
            for row in (row0..stop).step_by(l) {
                // Rows in the backend's item order; the two outputs are put back in order.
                let (e0, o0) = B::deinterleave_any(B::load(lane.0.as_ptr().add(2 * row)), B::load(lane.0.as_ptr().add(2 * row + l)));
                let (e1, o1) = B::deinterleave_any(B::load(lane.1.as_ptr().add(2 * row)), B::load(lane.1.as_ptr().add(2 * row + l)));
                let folded = kfold::<B>((e0, e1), (o0, o1), rs);
                B::store((o.0 as *mut u64).add(row), B::reorder(folded.0));
                B::store((o.1 as *mut u64).add(row), B::reorder(folded.1));
            }
        }
        fold_scalar(r, lanes, out, stop, row1);
    }
}
#[cfg(target_arch = "x86_64")]
crate::wide::entry_points!(fold_wide => fold_w8, fold_w4; (r: K, lanes: &[Planes<'_>], out: &[OutLane], row0: usize, row1: usize));

/// # Safety
/// Every output address must hold `row1` writable `u64` rows.
unsafe fn fold_scalar(r: K, lanes: &[Planes<'_>], out: &[OutLane], row0: usize, row1: usize) {
    for (lane, &o) in lanes.iter().zip(out) {
        for row in row0..row1 {
            let a = scalar::get(*lane, 2 * row);
            let b = scalar::get(*lane, 2 * row + 1);
            // SAFETY: `row < row1` and the caller guarantees `row1` writable output rows.
            unsafe { write(o, row, a + r * (b - a)) };
        }
    }
}

/// The terms bucketed by slot: bucket `s` of the linear terms is
/// `lin_order[lin_off[s]..lin_off[s + 1]]` (indices into `Terms::linear`), and likewise for
/// the products. The buckets depend only on the terms, so a pass builds them once and every
/// evaluation call of that pass reuses them.
struct SlotOrder {
    lin_off: Vec<usize>,
    lin_order: Vec<u32>,
    pro_off: Vec<usize>,
    pro_order: Vec<u32>,
    /// Bit `l`: lane `l` is a factor of a product (its row difference is read).
    in_product: u128,
    /// Bit `l`: lane `l` is the second factor of a product or squared (its doubled second
    /// coordinate is read).
    doubled: u128,
}

impl SlotOrder {
    /// # Panics
    /// When a slot holds more than [`MAX_SLOT_TERMS`] terms.
    fn new(terms: Terms<'_>) -> Self {
        let slots = terms.slots;
        let mut lin_off = vec![0_usize; slots + 1];
        for &(_, s) in terms.linear {
            lin_off[s + 1] += 1;
        }
        for s in 0..slots {
            lin_off[s + 1] += lin_off[s];
        }
        let mut lin_order = vec![0_u32; terms.linear.len()];
        {
            let mut next = lin_off.clone();
            for (index, &(_, s)) in terms.linear.iter().enumerate() {
                lin_order[next[s]] = index as u32;
                next[s] += 1;
            }
        }
        let mut pro_off = vec![0_usize; slots + 1];
        for &(_, _, s) in terms.products {
            pro_off[s + 1] += 1;
        }
        for s in 0..slots {
            pro_off[s + 1] += pro_off[s];
        }
        let mut pro_order = vec![0_u32; terms.products.len()];
        {
            let mut next = pro_off.clone();
            for (index, &(_, _, s)) in terms.products.iter().enumerate() {
                pro_order[next[s]] = index as u32;
                next[s] += 1;
            }
        }
        assert!(
            (0..slots).all(|s| lin_off[s + 1] - lin_off[s] + pro_off[s + 1] - pro_off[s] <= MAX_SLOT_TERMS),
            "at most MAX_SLOT_TERMS terms per slot"
        );
        let (mut in_product, mut doubled) = (0_u128, 0_u128);
        for &(i, j, _) in terms.products {
            in_product |= (1 << i) | (1 << j);
            doubled |= 1 << j;
        }
        Self { lin_off, lin_order, pro_off, pro_order, in_product, doubled }
    }

    /// Whether slot `s` has no terms (its sums stay zero).
    fn empty(&self, s: usize) -> bool {
        self.lin_off[s] == self.lin_off[s + 1] && self.pro_off[s] == self.pro_off[s + 1]
    }

    /// Whether slot `s` has products (component 2, the products of the differences, is
    /// zero without them).
    fn has_products(&self, s: usize) -> bool {
        self.pro_off[s] < self.pro_off[s + 1]
    }
}

/// The vector kernel's scratch, allocated once per task: the lane limbs of the current step
/// and the per-slot accumulators. It is empty unless the AVX-512 backend is active. The
/// unweighted accumulators (`inner`) and the weighted ones (`wac`, `wbd`, `wsum`) are
/// allocated by the first call that reads them: a task's calls are usually all weighted, and
/// at 151 slots the unused kind would cost 174 KB or more of initialisation per task.
#[cfg(target_arch = "x86_64")]
struct EvalScratch {
    even: Vec<(crate::x86::L52, crate::x86::L52)>,
    odd: Vec<(crate::x86::L52, crate::x86::L52)>,
    diff: Vec<(crate::x86::L52, crate::x86::L52)>,
    inner: Vec<[(crate::x86::A3, crate::x86::A3); 3]>,
    wac: Vec<[crate::x86::A5; 3]>,
    wbd: Vec<[crate::x86::A5; 3]>,
    wsum: Vec<[crate::x86::A5; 3]>,
    running: Vec<[(crate::x86::V, crate::x86::V); 3]>,
    /// The scratch of the AVX-512F kernel (empty on other backends).
    w8: WideScratch<std::arch::x86_64::__m512i>,
    /// The scratch of the AVX2 kernel (empty on other backends).
    w4: WideScratch<std::arch::x86_64::__m256i>,
}

/// The scratch of the evaluation on a backend without IFMA: the split lanes of the current
/// step, one unreduced accumulator pair per slot and component, and the reduced running sums.
#[cfg(target_arch = "x86_64")]
struct WideScratch<V> {
    even: Vec<crate::wide::KS<V>>,
    odd: Vec<crate::wide::KS<V>>,
    diff: Vec<crate::wide::KS<V>>,
    acc: Vec<[(crate::wide::Acc<V>, crate::wide::Acc<V>); 3]>,
    running: Vec<[(V, V); 3]>,
}

#[cfg(target_arch = "x86_64")]
impl<V: Copy> WideScratch<V> {
    fn empty() -> Self {
        Self { even: Vec::new(), odd: Vec::new(), diff: Vec::new(), acc: Vec::new(), running: Vec::new() }
    }

    /// # Safety
    /// `V` must be a SIMD integer vector (every bit pattern, the zero one included, is valid).
    unsafe fn zeros(width: usize, slots: usize) -> Self {
        // SAFETY: the all-zero bit pattern is a valid vector of every lane type.
        let (ks, acc, v) = unsafe { (std::mem::zeroed(), std::mem::zeroed(), std::mem::zeroed()) };
        Self {
            even: vec![ks; width],
            odd: vec![ks; width],
            diff: vec![ks; width],
            acc: vec![[acc; 3]; slots],
            running: vec![[(v, v); 3]; slots],
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl EvalScratch {
    fn new(width: usize, slots: usize) -> Self {
        if backend() == Backend::Avx512 {
            // SAFETY: the backend check guarantees the AVX-512 features of the initialisers.
            return unsafe { Self::zeros(width, slots) };
        }
        let (w8, w4) = match backend() {
            // SAFETY: both are integer vector types.
            Backend::Avx512F => (unsafe { WideScratch::zeros(width, slots) }, WideScratch::empty()),
            // SAFETY: both are integer vector types.
            Backend::Avx2 => (WideScratch::empty(), unsafe { WideScratch::zeros(width, slots) }),
            _ => (WideScratch::empty(), WideScratch::empty()),
        };
        Self {
            w8,
            w4,
            even: Vec::new(),
            odd: Vec::new(),
            diff: Vec::new(),
            inner: Vec::new(),
            wac: Vec::new(),
            wbd: Vec::new(),
            wsum: Vec::new(),
            running: Vec::new(),
        }
    }

    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
    unsafe fn zeros(width: usize, slots: usize) -> Self {
        use crate::x86::*;
        // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
        unsafe {
            Self {
                even: vec![(split52(z()), split52(z())); width],
                odd: vec![(split52(z()), split52(z())); width],
                diff: vec![(split52(z()), split52(z())); width],
                inner: Vec::new(),
                wac: Vec::new(),
                wbd: Vec::new(),
                wsum: Vec::new(),
                running: vec![[(z(), z()); 3]; slots],
                w8: WideScratch::empty(),
                w4: WideScratch::empty(),
            }
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
struct EvalScratch;

#[cfg(not(target_arch = "x86_64"))]
impl EvalScratch {
    fn new(width: usize, slots: usize) -> Self {
        let _ = (width, slots);
        Self
    }
}

/// The slot sums over the items `k0..k1` (rows `2k`, `2k + 1`) with the item weights
/// `weights[k - k0]` (one without weights), added to `out[slot]`: component 0 is
/// `sum w f(row 2k)`, component 1 `sum w f(row 2k + 1)` (only when `full`), component 2
/// `sum w` of the quadratic part of `f` at the row difference (the products of the
/// differences).
pub fn eval_items(
    lanes: &[Planes<'_>],
    terms: Terms<'_>,
    weights: Option<Planes<'_>>,
    k0: usize,
    k1: usize,
    full: bool,
    out: &mut [[K; 3]],
) {
    let order = SlotOrder::new(terms);
    let mut scratch = EvalScratch::new(lanes.len(), terms.slots);
    eval_items_with(lanes, terms, &order, &mut scratch, weights, k0, k1, full, out);
}

/// [`eval_items`] with the slot order and the vector scratch of the enclosing pass.
fn eval_items_with(
    lanes: &[Planes<'_>],
    terms: Terms<'_>,
    order: &SlotOrder,
    scratch: &mut EvalScratch,
    weights: Option<Planes<'_>>,
    k0: usize,
    k1: usize,
    full: bool,
    out: &mut [[K; 3]],
) {
    assert!(terms.slots <= MAX_SLOTS && out.len() >= terms.slots);
    assert!(lanes.len() <= MAX_LANES);
    assert!(k0 <= k1);
    // The vector path reads rows `2k .. 2k + 16` through raw pointers: every lane must hold
    // the rows of all items, and every term must name a lane and a slot that exist.
    assert!(
        lanes.iter().all(|lane| lane.0.len() >= 2 * k1 && lane.1.len() >= 2 * k1),
        "lanes cover the items"
    );
    assert!(
        terms.linear.iter().all(|&(l, s)| l < lanes.len() && s < terms.slots)
            && terms.products.iter().all(|&(i, j, s)| i < lanes.len() && j < lanes.len() && s < terms.slots),
        "terms name existing lanes and slots"
    );
    if let Some(w) = weights {
        assert!(w.0.len() >= k1 - k0 && w.1.len() >= k1 - k0);
    }
    #[cfg(target_arch = "x86_64")]
    if k1 - k0 >= 8 && backend() == Backend::Avx512 {
        // SAFETY: the IFMA backend is active, and the assertions above cover every raw read
        // (lane rows `2 k0 .. 2 k1`, weights `k1 - k0`, term indices).
        unsafe {
            return eval_vector(lanes, terms, order, scratch, weights, k0, k1, full, out);
        }
    }
    #[cfg(target_arch = "x86_64")]
    if k1 - k0 >= 8 {
        match backend() {
            // SAFETY: the backend check guarantees AVX-512F; the assertions above cover the reads.
            Backend::Avx512F => return unsafe { eval_w8(lanes, terms, order, &mut scratch.w8, weights, k0, k1, full, out) },
            // SAFETY: the backend check guarantees AVX2; the assertions above cover the reads.
            Backend::Avx2 => return unsafe { eval_w4(lanes, terms, order, &mut scratch.w4, weights, k0, k1, full, out) },
            _ => {}
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = (order, scratch);
    eval_scalar(lanes, terms, weights, k0, k1, full, out);
}

fn eval_scalar(
    lanes: &[Planes<'_>],
    terms: Terms<'_>,
    weights: Option<Planes<'_>>,
    k0: usize,
    k1: usize,
    full: bool,
    out: &mut [[K; 3]],
) {
    let slots = terms.slots;
    // Only the slots in use are cleared and summed.
    let mut sums = [[K::zero(); 3]; MAX_SLOTS];
    for k in k0..k1 {
        for s in sums[..slots].iter_mut() {
            *s = [K::zero(); 3];
        }
        let a = |l: usize| scalar::get(lanes[l], 2 * k);
        let b = |l: usize| scalar::get(lanes[l], 2 * k + 1);
        for &(l, s) in terms.linear {
            sums[s][0] += a(l);
            if full {
                sums[s][1] += b(l);
            }
        }
        for &(i, j, s) in terms.products {
            let (ai, aj) = (a(i), a(j));
            let (bi, bj) = (b(i), b(j));
            sums[s][0] += ai * aj;
            sums[s][2] += (bi - ai) * (bj - aj);
            if full {
                sums[s][1] += bi * bj;
            }
        }
        let w = weights.map(|w| scalar::get(w, k - k0));
        for (o, s) in out.iter_mut().zip(&sums[..slots]) {
            for c in 0..3 {
                if c == 1 && !full {
                    continue;
                }
                o[c] += match w {
                    Some(w) => w * s[c],
                    None => s[c],
                };
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
unsafe fn fold_vector(r: K, lanes: &[Planes<'_>], out: &[OutLane], row0: usize, row1: usize) {
    // SAFETY: the target features are enabled here and the caller checked them at run time.
    // `fold_rows` asserts that every input plane holds `2 row1` words, so the loads at
    // `2 row .. 2 row + 16` with `row + 8 <= stop <= row1` stay inside; the caller of
    // `fold_rows` guarantees `row1` writable rows at every output address.
    unsafe {
        use crate::x86::*;
        let rk = bck(r);
        let stop = row0 + (row1 - row0) / 8 * 8;
        for (lane, &o) in lanes.iter().zip(out) {
            for row in (row0..stop).step_by(8) {
                let (e0, o0) = deinterleave(load(lane.0.as_ptr().add(2 * row)), load(lane.0.as_ptr().add(2 * row + 8)));
                let (e1, o1) = deinterleave(load(lane.1.as_ptr().add(2 * row)), load(lane.1.as_ptr().add(2 * row + 8)));
                // Canonical difference, one reduced product, canonical sum.
                let d = (sub_mod(o0, e0), sub_mod(o1, e1));
                let folded = addk((e0, e1), mulk(d, rk));
                store((o.0 as *mut u64).add(row), folded.0);
                store((o.1 as *mut u64).add(row), folded.1);
            }
        }
        fold_scalar(r, lanes, out, stop, row1);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
#[allow(clippy::too_many_lines)]
unsafe fn eval_vector(
    lanes: &[Planes<'_>],
    terms: Terms<'_>,
    order: &SlotOrder,
    scratch: &mut EvalScratch,
    weights: Option<Planes<'_>>,
    k0: usize,
    k1: usize,
    full: bool,
    out: &mut [[K; 3]],
) {
    // SAFETY: the target features are enabled here and the caller checked them at run time.
    // `eval_items_with` asserts that every lane holds `2 k1` words per plane and the
    // weights `k1 - k0`, so the loads at `2 k .. 2 k + 16` and `k - k0` with
    // `k + 8 <= stop <= k1` stay inside; scratch and term indices are bounds-checked.
    unsafe {
        use crate::x86::*;
        let width = lanes.len();
        let slots = terms.slots;
        debug_assert_eq!(scratch.even.len(), width);
        debug_assert_eq!(scratch.running.len(), slots);
        let stop = k0 + (k1 - k0) / 8 * 8;
        let SlotOrder { lin_off, lin_order, pro_off, pro_order, in_product, .. } = order;
        let EvalScratch { even, odd, diff, inner, wac, wbd, wsum, running, .. } = scratch;
        let weighted = weights.is_some();
        // Reset the accumulators this call reads; the lane limbs are written before they are
        // read on every step.
        running.fill([(z(), z()); 3]);
        if weighted {
            if wac.is_empty() {
                *wac = vec![[a5(); 3]; slots];
                *wbd = vec![[a5(); 3]; slots];
                *wsum = vec![[a5(); 3]; slots];
            } else {
                wac.fill([a5(); 3]);
                wbd.fill([a5(); 3]);
                wsum.fill([a5(); 3]);
            }
        } else if inner.is_empty() {
            *inner = vec![[a3k(); 3]; slots];
        } else {
            inner.fill([a3k(); 3]);
        }
        let mut steps = 0_usize;
        for k in (k0..stop).step_by(8) {
            for l in 0..width {
                let (c0, c1) = lanes[l];
                let (e0, o0) = deinterleave(load(c0.as_ptr().add(2 * k)), load(c0.as_ptr().add(2 * k + 8)));
                let (e1, o1) = deinterleave(load(c1.as_ptr().add(2 * k)), load(c1.as_ptr().add(2 * k + 8)));
                even[l] = splitk((e0, e1));
                if full {
                    odd[l] = splitk((o0, o1));
                }
                // Only the factors of a product read the difference.
                if in_product & (1_u128 << l) != 0 {
                    // Canonical difference: limbs below 2^52 and 2^12.
                    diff[l] = splitk((sub_mod(o0, e0), sub_mod(o1, e1)));
                }
            }
            let w = weights.map(|w| {
                let (w0, w1) = loadk(w, k - k0);
                let (wr, wi) = (split51(w0), split51(w1));
                // The Karatsuba factor is the same for every slot and component.
                (wr, wi, add51(wr, wi))
            });
            for s in 0..slots {
                // A slot without terms keeps zero sums.
                if order.empty(s) {
                    continue;
                }
                let products_in_slot = order.has_products(s);
                // The slot's value on the pair. Its accumulator lives for one step and takes
                // at most `MAX_SLOT_TERMS` terms (checked by `SlotOrder::new`), within the
                // 512-term bound of `kmac7` and `lift52`, so it needs no flush.
                let mut value = [a3k(); 3];
                for &index in &lin_order[lin_off[s]..lin_off[s + 1]] {
                    let (l, _) = terms.linear[index as usize];
                    let e = even[l];
                    lift52(&mut value[0].0, e.0);
                    lift52(&mut value[0].1, e.1);
                    if full {
                        let o = odd[l];
                        lift52(&mut value[1].0, o.0);
                        lift52(&mut value[1].1, o.1);
                    }
                }
                for &index in &pro_order[pro_off[s]..pro_off[s + 1]] {
                    let (i, j, _) = terms.products[index as usize];
                    if i == j {
                        sqk(&mut value[0], even[i]);
                        sqk(&mut value[2], diff[i]);
                        if full {
                            sqk(&mut value[1], odd[i]);
                        }
                    } else {
                        kmac7(&mut value[0], even[i], even[j]);
                        kmac7(&mut value[2], diff[i], diff[j]);
                        if full {
                            kmac7(&mut value[1], odd[i], odd[j]);
                        }
                    }
                }
                for c in 0..3 {
                    if (c == 1 && !full) || (c == 2 && !products_in_slot) {
                        continue;
                    }
                    // A canonical value: limbs below 2^52 and 2^12 before the next multiply.
                    let (vr, vi) = redk(value[c]);
                    match w {
                        Some((wr, wi, wr_wi)) => {
                            // Radix-2^51 limbs of two canonical values, below 2^51 and 2^13.
                            let vr = split51(vr);
                            let vi = split51(vi);
                            mac5(&mut wac[s][c], vr, wr);
                            mac5(&mut wbd[s][c], vi, wi);
                            // The Karatsuba sum: `(a + b)(c + d)`, limbs below 2^52 and 2^14.
                            mac5(&mut wsum[s][c], add51(vr, vi), wr_wi);
                        }
                        None => {
                            let v = split52(vr);
                            lift52(&mut inner[s][c].0, v);
                            let v = split52(vi);
                            lift52(&mut inner[s][c].1, v);
                        }
                    }
                }
            }
            steps += 1;
            if steps == 512 {
                steps = 0;
                for s in 0..slots {
                    for c in 0..3 {
                        if c == 1 && !full {
                            continue;
                        }
                        if weighted {
                            let ac = wac[s][c];
                            let bd = wbd[s][c];
                            let sum = wsum[s][c];
                            // `ac + 2 bd` is the real part, `sum - ac - bd` the imaginary
                            // part; `red5` reduces both to canonical lanes.
                            running[s][c] = (
                                add_mod(running[s][c].0, red5(add5(ac, add5(bd, bd)))),
                                add_mod(running[s][c].1, red5(sub5(sub5(sum, ac), bd))),
                            );
                            wac[s][c] = a5();
                            wbd[s][c] = a5();
                            wsum[s][c] = a5();
                        } else {
                            running[s][c] = addk(running[s][c], redk(inner[s][c]));
                            inner[s][c] = a3k();
                        }
                    }
                }
            }
        }
        for s in 0..slots {
            for c in 0..3 {
                if c == 1 && !full {
                    continue;
                }
                if weighted {
                    let ac = wac[s][c];
                    let bd = wbd[s][c];
                    let sum = wsum[s][c];
                    running[s][c] = (
                        add_mod(running[s][c].0, red5(add5(ac, add5(bd, bd)))),
                        add_mod(running[s][c].1, red5(sub5(sub5(sum, ac), bd))),
                    );
                } else {
                    running[s][c] = addk(running[s][c], redk(inner[s][c]));
                }
                out[s][c] += hsum_k(running[s][c]);
            }
        }
        if stop < k1 {
            let tail = weights.map(|w| (&w.0[stop - k0..], &w.1[stop - k0..]));
            eval_scalar(lanes, terms, tail, stop, k1, full, out);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn eval_w8(
    lanes: &[Planes<'_>],
    terms: Terms<'_>,
    order: &SlotOrder,
    scratch: &mut WideScratch<std::arch::x86_64::__m512i>,
    weights: Option<Planes<'_>>,
    k0: usize,
    k1: usize,
    full: bool,
    out: &mut [[K; 3]],
) {
    // SAFETY: AVX-512F is enabled; the caller checked the reads.
    unsafe { eval_wide::<crate::wide::W8>(lanes, terms, order, scratch, weights, k0, k1, full, out) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn eval_w4(
    lanes: &[Planes<'_>],
    terms: Terms<'_>,
    order: &SlotOrder,
    scratch: &mut WideScratch<std::arch::x86_64::__m256i>,
    weights: Option<Planes<'_>>,
    k0: usize,
    k1: usize,
    full: bool,
    out: &mut [[K; 3]],
) {
    // SAFETY: AVX2 is enabled; the caller checked the reads.
    unsafe { eval_wide::<crate::wide::W4>(lanes, terms, order, scratch, weights, k0, k1, full, out) }
}

/// [`eval_items`] on the backends without IFMA, step by step over `LANES` items. Per slot the
/// value on the pair is summed unreduced; without weights it is added to the slot's
/// accumulator as it stands, with weights it is reduced to a multiplier operand (below 2^64,
/// not canonical) and multiplied by the weight. Slots without terms, and the difference
/// component of slots without products, are skipped (their sums are zero).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(clippy::too_many_lines)]
unsafe fn eval_wide<B: crate::wide::Simd>(
    lanes: &[Planes<'_>],
    terms: Terms<'_>,
    order: &SlotOrder,
    scratch: &mut WideScratch<B::V>,
    weights: Option<Planes<'_>>,
    k0: usize,
    k1: usize,
    full: bool,
    out: &mut [[K; 3]],
) {
    use crate::wide::*;
    // SAFETY: the caller guarantees the backend's features and that every lane holds the rows
    // `2 k0 .. 2 k1` and the weights `k1 - k0` rows.
    unsafe {
        let l = B::LANES;
        let width = lanes.len();
        let slots = terms.slots;
        debug_assert_eq!(scratch.even.len(), width);
        debug_assert_eq!(scratch.acc.len(), slots);
        let stop = k0 + (k1 - k0) / l * l;
        let SlotOrder { lin_off, lin_order, pro_off, pro_order, in_product, doubled } = order;
        let WideScratch { even, odd, diff, acc, running } = scratch;
        let weighted = weights.is_some();
        // Mac units a slot's accumulator takes per step: one per linear term (a lift) and two
        // per product and coordinate unweighted, two per coordinate weighted (one reduced
        // value times the weight). Reduce before the units reach the accumulator's limit.
        let units = if weighted {
            2
        } else {
            (0..slots).map(|s| (lin_off[s + 1] - lin_off[s]) + 2 * (pro_off[s + 1] - pro_off[s])).max().unwrap_or(0).max(1)
        };
        let flush_steps = (MAC_LIMIT / units).max(1);
        let zk = (zero::<B>(), zero::<B>());
        running[..slots].fill([zk; 3]);
        acc[..slots].fill([acck::<B>(); 3]);
        let mut steps = 0_usize;
        for k in (k0..stop).step_by(l) {
            for li in 0..width {
                let (c0, c1) = lanes[li];
                // Items in the backend's lane order (the sums do not depend on it; the
                // weights are put in the same order).
                let (e0, o0) = B::deinterleave_any(B::load(c0.as_ptr().add(2 * k)), B::load(c0.as_ptr().add(2 * k + l)));
                let (e1, o1) = B::deinterleave_any(B::load(c1.as_ptr().add(2 * k)), B::load(c1.as_ptr().add(2 * k + l)));
                // Every term reads the split coordinates; only the second factor of a product
                // reads the doubled second coordinate, and only products read the difference.
                let bit = 1_u128 << li;
                let x2 = doubled & bit != 0;
                even[li].c0 = split::<B>(e0);
                even[li].c1 = split::<B>(e1);
                if x2 {
                    even[li].c1x2 = split::<B>(B::add_mod(e1, e1));
                }
                if full {
                    odd[li].c0 = split::<B>(o0);
                    odd[li].c1 = split::<B>(o1);
                    if x2 {
                        odd[li].c1x2 = split::<B>(B::add_mod(o1, o1));
                    }
                }
                if in_product & bit != 0 {
                    let (d0, d1) = subk::<B>((o0, o1), (e0, e1));
                    diff[li].c0 = split::<B>(d0);
                    diff[li].c1 = split::<B>(d1);
                    if x2 {
                        diff[li].c1x2 = split::<B>(B::add_mod(d1, d1));
                    }
                }
            }
            let w = match weights {
                Some(w) => {
                    let (w0, w1) = loadk::<B>(w, k - k0);
                    Some(split_k::<B>((B::reorder(w0), B::reorder(w1))))
                }
                None => None,
            };
            for s in 0..slots {
                if order.empty(s) {
                    continue;
                }
                let lin = &lin_order[lin_off[s]..lin_off[s + 1]];
                let pro = &pro_order[pro_off[s]..pro_off[s + 1]];
                // One linear term and no product, weighted: the slot's value is the lane's
                // canonical value, multiplied by the weight as it stands.
                if let (Some(ws), [index], []) = (w, lin, pro) {
                    let (li, _) = terms.linear[*index as usize];
                    kmac::<B>(&mut acc[s][0], (even[li].c0, even[li].c1), ws);
                    if full {
                        kmac::<B>(&mut acc[s][1], (odd[li].c0, odd[li].c1), ws);
                    }
                    continue;
                }
                // Component by component (0: even rows, 1: odd rows, 2: row differences), so
                // that one value accumulator is live at a time.
                for c in 0..3 {
                    if (c == 1 && !full) || (c == 2 && pro.is_empty()) {
                        continue;
                    }
                    let rows: &[KS<B::V>] = match c {
                        0 => even,
                        1 => odd,
                        _ => diff,
                    };
                    // Unweighted, the terms go straight into the slot's accumulator.
                    let mut value = if w.is_some() { acck::<B>() } else { acc[s][c] };
                    if c < 2 {
                        for &index in lin {
                            let (li, _) = terms.linear[index as usize];
                            lift::<B>(&mut value.0, rows[li].c0.l);
                            lift::<B>(&mut value.1, rows[li].c1.l);
                        }
                    }
                    for &index in pro {
                        let (i, j, _) = terms.products[index as usize];
                        let x = rows[i];
                        if i == j {
                            // x^2: re += a^2 + 2 b^2, im += 2ab (three products).
                            mac::<B>(&mut value.0, x.c0, x.c0);
                            mac::<B>(&mut value.0, x.c1, x.c1x2);
                            mac::<B>(&mut value.1, x.c0, x.c1x2);
                        } else {
                            kmac::<B>(&mut value, (x.c0, x.c1), rows[j]);
                        }
                    }
                    match w {
                        Some(ws) => {
                            // A multiplier operand below 2^64 per coordinate.
                            let (vr, vi) = (red_lazy::<B>(value.0), red_lazy::<B>(value.1));
                            kmac::<B>(&mut acc[s][c], (split::<B>(vr), split::<B>(vi)), ws);
                        }
                        None => acc[s][c] = value,
                    }
                }
            }
            steps += 1;
            if steps == flush_steps {
                steps = 0;
                wide_flush::<B>(order, &mut acc[..slots], &mut running[..slots], full);
            }
        }
        wide_flush::<B>(order, &mut acc[..slots], &mut running[..slots], full);
        for s in 0..slots {
            if order.empty(s) {
                continue;
            }
            for c in 0..3 {
                if (c == 1 && !full) || (c == 2 && !order.has_products(s)) {
                    continue;
                }
                out[s][c] += hsum_k::<B>(running[s][c]);
            }
        }
        if stop < k1 {
            let tail = weights.map(|w| (&w.0[stop - k0..], &w.1[stop - k0..]));
            eval_scalar(lanes, terms, tail, stop, k1, full, out);
        }
    }
}

/// Reduce the slot accumulators in use into the running sums (a function, not a closure:
/// closures do not inherit the entry point's target features).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn wide_flush<B: crate::wide::Simd>(
    order: &SlotOrder,
    sums: &mut [[(crate::wide::Acc<B::V>, crate::wide::Acc<B::V>); 3]],
    running: &mut [[(B::V, B::V); 3]],
    full: bool,
) {
    use crate::wide::*;
    // SAFETY: the caller guarantees the backend's features.
    unsafe {
        for (s, (a, r)) in sums.iter_mut().zip(running.iter_mut()).enumerate() {
            if order.empty(s) {
                continue;
            }
            for c in 0..3 {
                if (c == 1 && !full) || (c == 2 && !order.has_products(s)) {
                    continue;
                }
                r[c] = addk::<B>(r[c], redk::<B>(a[c]));
                a[c] = acck::<B>();
            }
        }
    }
}

use crate::pattern::Leaves;
use rayon::prelude::*;

/// A level-0 lane folded by the first challenge into a plane lane.
pub enum MixedLane<'a> {
    /// A `K` lane already held as planes.
    Planes(Planes<'a>),
    /// An integer lane written block by block: `fill(first, cells)` writes the values of
    /// cells `first .. first + cells.len()` (`|value| < 2^62`).
    Ints(&'a (dyn Fn(usize, &mut [i64]) + Sync)),
}

/// Level-0 lanes computed block by block instead of being stored (the rows of a fraction
/// layer computed from the leaves).
pub trait RowSource: Sync {
    /// Lanes per row.
    fn width(&self) -> usize;
    /// Level-0 rows.
    fn rows(&self) -> usize;
    /// Write rows `row0..row1` of every lane, canonical, to rows `0..row1 - row0` of `out`.
    ///
    /// # Safety
    /// Every output address must hold `row1 - row0` writable `u64` rows.
    unsafe fn fill(&self, row0: usize, row1: usize, out: &[OutLane]);
}

/// How a pass writes the rows of its output table.
pub enum Fill<'a, L: Leaves> {
    /// Fold the input lanes by the challenge: `out[row] = in[2 row] + r (in[2 row + 1] - in[2 row])`.
    Fold(K, &'a [Planes<'a>]),
    /// The first fold of level-0 lanes of mixed kinds by the challenge.
    Mixed(K, &'a [MixedLane<'a>]),
    /// Materialise the leaves folded by the two prefix challenges (`agg_simd::pattern`).
    Pattern([K; 4], &'a L),
    /// Materialise one lane `sum_l sum_a weights[l][a] L_l(a, row)` of the leaves.
    PatternCombined(&'a [[K; 4]], &'a L),
    /// Level-0 lanes computed per block into a task-local scratch: folded by the challenge
    /// into the output table, or (`None`, no output table) only evaluated.
    Computed(Option<K>, &'a dyn RowSource),
}

/// `out[l][row]` for every lane of `lanes` and `row` in `row0..row1`: planes folded by `r`
/// as [`fold_rows`], integer lanes as `g(2 row) + r (g(2 row + 1) - g(2 row))`.
///
/// # Safety
/// Every output address must hold `row1` writable `u64` rows.
///
/// # Panics
/// As [`fold_rows`] for a plane lane holding fewer than `2 row1` rows.
pub unsafe fn fold_mixed(r: K, lanes: &[MixedLane<'_>], out: &[OutLane], row0: usize, row1: usize) {
    assert_eq!(lanes.len(), out.len());
    for (lane, &o) in lanes.iter().zip(out) {
        match lane {
            // SAFETY: the caller guarantees `row1` writable rows at `o`; `fold_rows` checks
            // the input rows.
            MixedLane::Planes(p) => unsafe { fold_rows(r, &[*p], &[o], row0, row1) },
            MixedLane::Ints(fill) => {
                const CHUNK: usize = 256;
                let mut cells = [0_i64; 2 * CHUNK];
                let mut lo = [0_i64; CHUNK];
                let mut hi = [0_i64; CHUNK];
                let mut row = row0;
                while row < row1 {
                    let n = (row1 - row).min(CHUNK);
                    fill(2 * row, &mut cells[..2 * n]);
                    for i in 0..n {
                        lo[i] = cells[2 * i];
                        hi[i] = cells[2 * i + 1];
                    }
                    // SAFETY: `row + n <= row1`, and the caller guarantees `row1` writable rows
                    // of this lane's planes, which no other view borrows.
                    let c0 = unsafe { std::slice::from_raw_parts_mut((o.0 as *mut u64).add(row), n) };
                    // SAFETY: as above, for the second plane.
                    let c1 = unsafe { std::slice::from_raw_parts_mut((o.1 as *mut u64).add(row), n) };
                    crate::fold::int_fold(r, &lo[..n], &hi[..n], (c0, c1));
                    row += n;
                }
            }
        }
    }
}

/// The evaluation a pass performs on the pairs `(2k, 2k + 1)` of its table: the slot sums
/// with the split equality weights `first[k mod |first|] second[k / |first|]` (no weights
/// when `first` is empty).
pub struct Eval<'a> {
    pub terms: Terms<'a>,
    pub first: Planes<'a>,
    pub second: &'a [K],
    pub full: bool,
}

/// The summand of an evaluation: a bilinear form (three components per slot) or a summand
/// of higher degree `D` ([`crate::high`], `D + 1` components per slot).
#[derive(Clone, Copy, Debug)]
pub enum Summand<'a> {
    Bilinear(Terms<'a>),
    High(&'a crate::high::HighTerms),
}

impl Summand<'_> {
    fn slots(&self) -> usize {
        match self {
            Self::Bilinear(terms) => terms.slots,
            Self::High(terms) => terms.slots,
        }
    }

    /// Components per slot.
    pub fn components(&self) -> usize {
        match self {
            Self::Bilinear(_) => 3,
            Self::High(terms) => terms.components(),
        }
    }
}

/// [`Eval`] of either summand. The pass returns the slot sums flat, slot-major:
/// `components()` entries per slot.
pub struct EvalAny<'a> {
    pub summand: Summand<'a>,
    pub first: Planes<'a>,
    pub second: &'a [K],
    pub full: bool,
}

impl<'a> From<Eval<'a>> for EvalAny<'a> {
    fn from(e: Eval<'a>) -> Self {
        Self { summand: Summand::Bilinear(e.terms), first: e.first, second: e.second, full: e.full }
    }
}

/// Flat slot sums of a bilinear summand as `[q(0), q(1), q_2]` per slot.
fn triples(flat: Vec<K>) -> Vec<[K; 3]> {
    flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

/// The term order of a pass's summand, built once per pass.
enum Order {
    Bilinear(SlotOrder),
    High(crate::high::Plan),
}

/// The evaluation scratch of one task of a pass.
enum TaskScratch {
    Bilinear(EvalScratch),
    High(crate::high::Scratch),
}

/// The slot sums of `e`'s summand over the items `k0..k1`, added to the flat `out`.
#[allow(clippy::too_many_arguments)]
fn eval_call(
    lanes: &[Planes<'_>],
    e: &EvalAny<'_>,
    order: &Order,
    scratch: &mut TaskScratch,
    weights: Option<Planes<'_>>,
    k0: usize,
    k1: usize,
    out: &mut [K],
) {
    match (e.summand, order, scratch) {
        (Summand::Bilinear(terms), Order::Bilinear(order), TaskScratch::Bilinear(scratch)) => {
            eval_items_with(lanes, terms, order, scratch, weights, k0, k1, e.full, out.as_chunks_mut::<3>().0);
        }
        (Summand::High(_), Order::High(plan), TaskScratch::High(scratch)) => {
            crate::high::eval_items_with(lanes, plan, scratch, weights, k0, k1, e.full, out);
        }
        _ => unreachable!("the order and the scratch follow the summand"),
    }
}

/// Output rows per fused sub-block (the written rows stay in L1/L2 for the evaluation).
const BLOCK_ROWS: usize = 512;
/// Output rows per parallel task.
const TASK_ROWS: usize = 4096;
/// Below this many entries in the short equality table, `eval_items` would run on fewer than
/// eight items per call (occasionally on a single item), so the pass builds one weight plane
/// per block instead and weights every item there.
const WEIGHT_PLANE_MAX_FIRST: usize = 64;

/// One sum-check pass: write `rows` output rows of `width` lanes into `out` (lane-major,
/// coordinate planes of `rows` words each) by `fill`, and evaluate `eval` on the written
/// pairs (on the `input` lanes when there is no fill), in sub-blocks that stay in cache.
/// Returns the slot sums (`[q(0), q(1), q_2]` per slot, before the slot coefficients).
///
/// # Panics
/// When `out` is too small for the table or the weights do not cover the items.
#[allow(clippy::too_many_lines)]
pub fn fused_pass<L: Leaves>(
    out: Option<(&mut [u64], usize, usize)>,
    input: &[Planes<'_>],
    fill: Option<Fill<'_, L>>,
    eval: Option<Eval<'_>>,
) -> Vec<[K; 3]> {
    // SAFETY: every word of `out` is initialized, and `MaybeUninit<u64>` has the same
    // layout, so the reinterpretation is valid. Words the pass leaves unwritten keep their
    // initialized values.
    let out = out.map(|(buf, width, rows)| unsafe {
        (std::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<std::mem::MaybeUninit<u64>>(), buf.len()), width, rows)
    });
    // SAFETY: every output word is initialized, which satisfies the precondition over
    // whatever `fill` writes.
    triples(unsafe { fused_pass_uninit(out, input, fill, eval.map(EvalAny::from)) })
}

/// [`fused_pass`] of either summand, with the slot sums flat ([`EvalAny`]).
///
/// # Panics
/// As [`fused_pass`].
pub fn fused_pass_with<L: Leaves>(
    out: Option<(&mut [u64], usize, usize)>,
    input: &[Planes<'_>],
    fill: Option<Fill<'_, L>>,
    eval: Option<EvalAny<'_>>,
) -> Vec<K> {
    // SAFETY: as in `fused_pass`.
    let out = out.map(|(buf, width, rows)| unsafe {
        (std::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<std::mem::MaybeUninit<u64>>(), buf.len()), width, rows)
    });
    // SAFETY: every output word is initialized.
    unsafe { fused_pass_uninit(out, input, fill, eval) }
}

/// Run a pass into a pooled buffer and return the buffer initialized.
///
/// `out` must hold at least `2 * width * rows` words. The returned buffer has length
/// `2 * width * rows`, and every one of those words is defined: the fill writes rows
/// `0..rows` of every lane before the evaluation reads them.
///
/// # Panics
/// When the buffer is too small or `fill` does not write every output lane
/// ([`Fill::PatternCombined`] writes one lane, [`Fill::Computed`] with no challenge writes
/// none).
pub fn fused_pass_into<L: Leaves>(
    out: AlignedVec<std::mem::MaybeUninit<u64>>,
    width: usize,
    rows: usize,
    input: &[Planes<'_>],
    fill: Fill<'_, L>,
    eval: Option<Eval<'_>>,
) -> (AlignedVec<u64>, Vec<[K; 3]>) {
    let (table, total) = fused_pass_into_with(out, width, rows, input, fill, eval.map(EvalAny::from));
    (table, triples(total))
}

/// [`fused_pass_into`] of either summand, with the slot sums flat ([`EvalAny`]).
///
/// # Panics
/// As [`fused_pass_into`].
pub fn fused_pass_into_with<L: Leaves>(
    mut out: AlignedVec<std::mem::MaybeUninit<u64>>,
    width: usize,
    rows: usize,
    input: &[Planes<'_>],
    fill: Fill<'_, L>,
    eval: Option<EvalAny<'_>>,
) -> (AlignedVec<u64>, Vec<K>) {
    let words = 2 * width * rows;
    assert!(out.len() >= words, "table buffer");
    assert!(
        !matches!(&fill, Fill::PatternCombined(..) if width != 1)
            && !matches!(&fill, Fill::Computed(None, _)),
        "the fill must write every output lane"
    );
    // SAFETY: the fill writes rows `0..rows` of every lane before any of them is read, and
    // the buffer covers those 2 * width * rows words.
    let total = unsafe { fused_pass_uninit(Some((&mut out[..], width, rows)), input, Some(fill), eval) };
    out.truncate(words);
    // SAFETY: every word of the truncated length was written by the fill.
    (unsafe { out.assume_init() }, total)
}

/// Run a pass with output slots that may be uninitialized. Each output block is written by
/// the fill before it is read for evaluation.
///
/// # Safety
/// Every output slot the pass reads must be initialized. `Fold`, `Mixed`, `Pattern` and
/// `Computed` with a challenge write rows `0..rows` of every output lane before the
/// evaluation reads them; `PatternCombined` writes lane 0 only, and `Computed` without a
/// challenge writes nothing, so the remaining slots must already be initialized.
unsafe fn fused_pass_uninit<L: Leaves>(
    mut out: Option<(&mut [std::mem::MaybeUninit<u64>], usize, usize)>,
    input: &[Planes<'_>],
    fill: Option<Fill<'_, L>>,
    eval: Option<EvalAny<'_>>,
) -> Vec<K> {
    let slots = eval.as_ref().map_or(0, |e| e.summand.slots());
    // Entries per slot of the flat sums.
    let comps = eval.as_ref().map_or(3, |e| e.summand.components());
    // The computed source of the pass and whether its rows are folded into the output.
    let computed = match &fill {
        Some(Fill::Computed(r, source)) => Some((*r, *source)),
        _ => None,
    };
    // The output addresses derive from the unique borrow of the buffer, so the pass may
    // write through them.
    let (addresses, rows_out): (Vec<OutLane>, usize) = match (&mut out, computed) {
        (Some((buf, width, rows)), _) => {
            let (width, rows) = (*width, *rows);
            assert!(buf.len() >= 2 * width * rows, "table buffer");
            let base = buf.as_mut_ptr() as usize;
            ((0..width).map(|l| (base + 16 * l * rows, base + 8 * (2 * l + 1) * rows)).collect(), rows)
        }
        (None, Some((None, source))) => (Vec::new(), source.rows()),
        (None, _) => (Vec::new(), input.first().map_or(0, |l| l.0.len())),
    };
    assert!(
        fill.is_none() || out.is_some() || matches!(computed, Some((None, _))),
        "a fill needs an output table"
    );
    assert!(
        !matches!(computed, Some((Some(_), source)) if 2 * rows_out != source.rows()),
        "a computed fold halves the source rows"
    );
    // Input rows per output row of a computed pass, and the scratch words per block.
    let computed_width = computed.map_or(0, |(_, source)| source.width());
    let scale = if matches!(computed, Some((Some(_), _))) { 2 } else { 1 };
    let items = if eval.is_some() { rows_out / 2 } else { 0 };
    let (first_bits, first_mask) = match &eval {
        Some(e) if !e.first.0.is_empty() => {
            assert!(e.first.0.len().is_power_of_two());
            assert_eq!(e.first.0.len() * e.second.len(), items.max(1), "weights cover the items");
            (e.first.0.len().trailing_zeros(), e.first.0.len() - 1)
        }
        _ => (0, 0),
    };
    // The lane count of the tables the evaluation reads and the term order it reuses. Both
    // depend only on the pass, not on the call or the task.
    let eval_width = match (computed, fill.is_some()) {
        (Some((None, _)), _) => computed_width,
        (_, true) => addresses.len(),
        (_, false) => input.len(),
    };
    let eval_order = eval.as_ref().map(|e| match e.summand {
        Summand::Bilinear(terms) => Order::Bilinear(SlotOrder::new(terms)),
        Summand::High(terms) => Order::High(crate::high::Plan::new(terms, eval_width)),
    });
    let run = |row0: usize, row1: usize| -> Vec<K> {
        let mut sums = vec![K::zero(); slots * comps];
        let mut partial = vec![K::zero(); slots * comps];
        // The vector kernel's scratch, reused by every evaluation call of this task.
        let mut eval_scratch = match &eval_order {
            Some(Order::High(plan)) => TaskScratch::High(crate::high::Scratch::new(plan)),
            _ => TaskScratch::Bilinear(EvalScratch::new(eval_width, slots)),
        };
        // A computed source's rows of one block: lane-major planes of `scale * BLOCK_ROWS`
        // rows, written by the source and read back by the fold or the evaluation (L2).
        let block_in = scale * BLOCK_ROWS;
        let mut scratch = vec![0_u64; 2 * computed_width * block_in];
        let scratch_lanes: Vec<OutLane> = {
            let base = scratch.as_mut_ptr() as usize;
            (0..computed_width)
                .map(|l| (base + 16 * l * block_in, base + 8 * (2 * l + 1) * block_in))
                .collect()
        };
        // Read-back views of the rows this task has just written (one block at a time).
        let mut written: Vec<Planes<'_>> = Vec::with_capacity(addresses.len());
        // Block-local weight plane `first[k & mask] second[k >> bits]` (only when the short
        // table would split the items into sub-vector calls).
        let block_items = BLOCK_ROWS / 2;
        let weight_plane = eval.as_ref().is_some_and(|e| {
            !e.first.0.is_empty() && e.first.0.len() < WEIGHT_PLANE_MAX_FIRST
        });
        // The plane buffers exist only when the short table needs a plane.
        let (mut first_lo, mut first_hi, mut second_lo, mut second_hi, mut weight_lo, mut weight_hi) = if weight_plane {
            (
                vec![0_u64; block_items],
                vec![0_u64; block_items],
                vec![0_u64; block_items],
                vec![0_u64; block_items],
                vec![0_u64; block_items],
                vec![0_u64; block_items],
            )
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };
        if weight_plane {
            let e = eval.as_ref().expect("a weight plane needs an evaluation");
            debug_assert_eq!(block_items % e.first.0.len(), 0, "blocks cover whole first tables");
            for k in 0..block_items {
                first_lo[k] = e.first.0[k & first_mask];
                first_hi[k] = e.first.1[k & first_mask];
            }
        }
        let mut block_start = row0;
        while block_start < row1 {
            let block_end = (block_start + BLOCK_ROWS).min(row1);
            // The source's rows of this block as planes (computed passes only).
            let mut local: Vec<Planes<'_>> = Vec::new();
            if let Some((r, source)) = computed {
                let rows = scale * (block_end - block_start);
                // SAFETY: every scratch lane holds `block_in >= rows` rows per plane, owned by
                // this task.
                unsafe { source.fill(scale * block_start, scale * block_end, &scratch_lanes) };
                local = scratch_lanes
                    .iter()
                    // SAFETY: the source has just written rows `0..rows` of every scratch plane
                    // (each holds `block_in >= rows` words); nothing writes them while the views
                    // live.
                    .map(|&(c0, c1)| unsafe {
                        (
                            std::slice::from_raw_parts(c0 as *const u64, rows),
                            std::slice::from_raw_parts(c1 as *const u64, rows),
                        )
                    })
                    .collect();
                if let Some(r) = r {
                    // Output rows from `block_start` on: the addresses stay inside the table.
                    let shifted: Vec<OutLane> =
                        addresses.iter().map(|&(c0, c1)| (c0 + 8 * block_start, c1 + 8 * block_start)).collect();
                    // SAFETY: the shifted addresses hold rows `0..block_end - block_start` inside
                    // the table (`block_end <= rows_out`), written by this task only.
                    unsafe { fold_rows(r, &local, &shifted, 0, block_end - block_start) };
                }
            }
            match &fill {
                // SAFETY (all four fills): every output lane addresses `rows_out >= block_end`
                // rows of the table (the buffer size is asserted above), and this task alone
                // writes rows `block_start..block_end`; the fills check their inputs.
                Some(Fill::Fold(r, lanes)) => unsafe { fold_rows(*r, lanes, &addresses, block_start, block_end) },
                // SAFETY: as above.
                Some(Fill::Mixed(r, lanes)) => unsafe { fold_mixed(*r, lanes, &addresses, block_start, block_end) },
                // SAFETY: as above.
                Some(Fill::Pattern(e, leaves)) => unsafe {
                    crate::pattern::fill_rows(*leaves, *e, &addresses, block_start, block_end);
                },
                // SAFETY: as above.
                Some(Fill::PatternCombined(weights, leaves)) => unsafe {
                    crate::pattern::fill_combined(*leaves, weights, addresses[0], block_start, block_end);
                },
                Some(Fill::Computed(..)) | None => {}
            }
            if let Some(e) = &eval {
                let order = eval_order.as_ref().expect("an evaluation has a term order");
                // Item `k` reads rows `2 (k - base)` of `lanes`: the block-local scratch of an
                // evaluate-only computed pass starts at the block's first row.
                let (lanes, base): (&[Planes<'_>], usize) = match computed {
                    Some((None, _)) => (&local, block_start / 2),
                    _ if fill.is_some() => {
                        // Rows `block_start..block_end` of the table, written above by this
                        // task; the views are created after the writes and cover no row of
                        // another task (`block_start` is even: tasks and blocks start at
                        // multiples of 512 rows).
                        let rows = block_end - block_start;
                        written.clear();
                        // SAFETY: rows `block_start..block_end` lie inside the table, were written
                        // by the fill above, and no task writes them while the views live.
                        written.extend(addresses.iter().map(|&(c0, c1)| unsafe {
                            (
                                std::slice::from_raw_parts((c0 as *const u64).add(block_start), rows),
                                std::slice::from_raw_parts((c1 as *const u64).add(block_start), rows),
                            )
                        }));
                        (&written, block_start / 2)
                    }
                    _ => (input, 0),
                };
                let (k0, k1) = (block_start / 2, block_end / 2);
                if e.first.0.is_empty() {
                    // No equality weights: one call over the whole block.
                    partial.fill(K::zero());
                    eval_call(lanes, e, order, &mut eval_scratch, None, k0 - base, k1 - base, &mut partial);
                    for (acc, p) in sums.iter_mut().zip(&partial) {
                        *acc += *p;
                    }
                } else if weight_plane {
                    // One weight per item and one vector call per block. The short table
                    // repeats with period `first.len()`, so only `second` changes inside it.
                    debug_assert_eq!(k0 & first_mask, 0, "blocks start at a whole period of the first table");
                    let count = k1 - k0;
                    let mut k = k0;
                    while k < k1 {
                        let block = k >> first_bits;
                        let end = ((block + 1) << first_bits).min(k1);
                        let s = e.second.get(block).copied().unwrap_or_else(|| K::from_u64(1));
                        second_lo[k - k0..end - k0].fill(s.c0().to_canonical_u64());
                        second_hi[k - k0..end - k0].fill(s.c1().to_canonical_u64());
                        k = end;
                    }
                    let weights = if e.first.0.len() == 1 && e.first.0[0] == 1 && e.first.1[0] == 0 {
                        // The short table is the single entry one; its factor drops out.
                        (&second_lo[..count], &second_hi[..count])
                    } else {
                        crate::k::k_mul(
                            (&first_lo[..count], &first_hi[..count]),
                            (&second_lo[..count], &second_hi[..count]),
                            (&mut weight_lo[..count], &mut weight_hi[..count]),
                        );
                        (&weight_lo[..count], &weight_hi[..count])
                    };
                    partial.fill(K::zero());
                    eval_call(lanes, e, order, &mut eval_scratch, Some(weights), k0 - base, k1 - base, &mut partial);
                    for (acc, p) in sums.iter_mut().zip(&partial) {
                        *acc += *p;
                    }
                } else {
                    // The short table has at least 64 entries: each block covers a vector
                    // step, and its `second` factor applies once to the block sums.
                    let mut k = k0;
                    while k < k1 {
                        let block = k >> first_bits;
                        let end = ((block + 1) << first_bits).min(k1);
                        let offset = k & first_mask;
                        partial.fill(K::zero());
                        eval_call(
                            lanes,
                            e,
                            order,
                            &mut eval_scratch,
                            Some((&e.first.0[offset..], &e.first.1[offset..])),
                            k - base,
                            end - base,
                            &mut partial,
                        );
                        let s = e.second.get(block).copied().unwrap_or_else(|| K::from_u64(1));
                        // Slots and components without terms have zero sums.
                        for (acc, p) in sums.iter_mut().zip(&partial) {
                            if !p.is_zero() {
                                *acc += s * *p;
                            }
                        }
                        k = end;
                    }
                }
            }
            block_start = block_end;
        }
        drop(scratch);
        sums
    };
    let add = |mut left: Vec<K>, right: Vec<K>| {
        for (l, r) in left.iter_mut().zip(right) {
            *l += r;
        }
        left
    };
    let tasks = rows_out.div_ceil(TASK_ROWS).max(1);
    if rows_out >= 2 * TASK_ROWS && tasks > 1 {
        (0..tasks)
            .into_par_iter()
            .map(|t| run(t * TASK_ROWS, ((t + 1) * TASK_ROWS).min(rows_out)))
            .reduce(|| vec![K::zero(); slots * comps], add)
    } else {
        run(0, rows_out)
    }
}
