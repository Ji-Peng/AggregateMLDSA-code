//! Sum-check round kernels over pairs of [`F128`] tables (agg-mldsa patch, PATCH.md): the
//! message `(q(1), q(inf))` of a product of two tables over their lowest variable, and the
//! fold of both tables by a challenge fused with the next round's message, read from the
//! folded values while they are still in registers. Two table shapes: PAIR tables, whose
//! variable pairs adjacent entries `(2k, 2k + 1)`, and BLOCK tables `[block][row]`, whose
//! variable pairs adjacent blocks of `rows` entries.
//!
//! Every function is safe and value-identical on every path: the AVX-512 kernels run when
//! the CPU has `avx512f` and `vpclmulqdq` (checked once), the portable loops otherwise. The
//! messages are accumulated as unreduced 256-bit carry-less products XORed together (exact:
//! GF(2) addition carries nothing) and reduced once per call, so a caller that splits a
//! table over threads adds the per-chunk messages.

use crate::{F128, F256Unreduced};

/// `(q(1), q(inf))` of the product of the pair tables `a` and `b` over their lowest
/// variable: `(sum_k a[2k+1] b[2k+1], sum_k (a[2k] + a[2k+1]) (b[2k] + b[2k+1]))`.
pub fn pair_message(a: &[F128], b: &[F128]) -> (F128, F128) {
    assert!(a.len() == b.len() && a.len() % 2 == 0);
    #[cfg(target_arch = "x86_64")]
    if x86::enabled() {
        // SAFETY: the CPU check passed; the lengths were asserted.
        return unsafe { x86::pair_message(a, b) };
    }
    portable::pair_message(a, b)
}

/// Fold the pair tables `src_a`, `src_b` by `r` into `dst_a`, `dst_b` (`dst[k] = src[2k] +
/// r (src[2k] + src[2k+1])`, `dst.len() * 2 == src.len()`) and return the message of the
/// folded tables over their lowest variable ([`pair_message`] of the destinations; an
/// unpaired last entry of an odd-length destination is not part of it).
pub fn fold_pairs_two_and_message(
    src_a: &[F128],
    src_b: &[F128],
    dst_a: &mut [F128],
    dst_b: &mut [F128],
    r: F128,
) -> (F128, F128) {
    assert!(src_a.len() == src_b.len() && dst_a.len() == dst_b.len() && 2 * dst_a.len() == src_a.len());
    #[cfg(target_arch = "x86_64")]
    if x86::enabled() {
        // SAFETY: the CPU check passed; the lengths were asserted.
        return unsafe { x86::fold_pairs_two_and_message(src_a, src_b, dst_a, dst_b, r) };
    }
    portable::fold_pairs_two_and_message(src_a, src_b, dst_a, dst_b, r)
}

/// `(q(1), q(inf))` of the product of the block tables `g` and `y` (two blocks of `rows`
/// entries each) over the block variable: `(sum_i g1[i] y1[i], sum_i (g0[i] + g1[i]) (y0[i] +
/// y1[i]))`.
pub fn block_message(g: &[F128], y: &[F128], rows: usize) -> (F128, F128) {
    assert!(g.len() == 2 * rows && y.len() == 2 * rows);
    #[cfg(target_arch = "x86_64")]
    if x86::enabled() {
        // SAFETY: the CPU check passed; the lengths were asserted.
        return unsafe { x86::block_message(g, y, rows) };
    }
    portable::block_message(g, y, rows)
}

/// Fold four blocks of `rows` entries of `g_src` and `y_src` by `r` into two blocks each
/// (`dst[k][i] = src[2k][i] + r (src[2k][i] + src[2k+1][i])`) and return the message of the
/// two folded blocks ([`block_message`] of the destinations).
pub fn fold_blocks_two_and_message(
    g_src: &[F128],
    y_src: &[F128],
    g_dst: &mut [F128],
    y_dst: &mut [F128],
    rows: usize,
    r: F128,
) -> (F128, F128) {
    assert!(g_src.len() == 4 * rows && y_src.len() == 4 * rows && g_dst.len() == 2 * rows && y_dst.len() == 2 * rows);
    #[cfg(target_arch = "x86_64")]
    if x86::enabled() {
        // SAFETY: the CPU check passed; the lengths were asserted.
        return unsafe { x86::fold_blocks_two_and_message(g_src, y_src, g_dst, y_dst, rows, r) };
    }
    portable::fold_blocks_two_and_message(g_src, y_src, g_dst, y_dst, rows, r)
}

/// Fold two blocks of `rows` entries of `g_src` and `y_src` by `r` into one block each (the
/// last round, which has no next message).
pub fn fold_blocks_two(g_src: &[F128], y_src: &[F128], g_dst: &mut [F128], y_dst: &mut [F128], rows: usize, r: F128) {
    assert!(g_src.len() == 2 * rows && y_src.len() == 2 * rows && g_dst.len() == rows && y_dst.len() == rows);
    #[cfg(target_arch = "x86_64")]
    if x86::enabled() {
        // SAFETY: the CPU check passed; the lengths were asserted.
        unsafe { x86::fold_blocks_two(g_src, y_src, g_dst, y_dst, rows, r) };
        return;
    }
    portable::fold_blocks_two(g_src, y_src, g_dst, y_dst, rows, r);
}

mod portable {
    use super::{F128, F256Unreduced};

    pub(super) fn pair_message(a: &[F128], b: &[F128]) -> (F128, F128) {
        // Unreduced carry-less sums, exact, reduced once per call.
        let (mut q1, mut qi) = (F256Unreduced::ZERO, F256Unreduced::ZERO);
        for (x, y) in a.chunks_exact(2).zip(b.chunks_exact(2)) {
            q1 = q1 ^ x[1].mul_unreduced(y[1]);
            qi = qi ^ (x[0] + x[1]).mul_unreduced(y[0] + y[1]);
        }
        (q1.reduce(), qi.reduce())
    }

    pub(super) fn fold_pairs_two_and_message(
        src_a: &[F128],
        src_b: &[F128],
        dst_a: &mut [F128],
        dst_b: &mut [F128],
        r: F128,
    ) -> (F128, F128) {
        for (d, s) in dst_a.iter_mut().zip(src_a.chunks_exact(2)) {
            *d = s[0] + r * (s[0] + s[1]);
        }
        for (d, s) in dst_b.iter_mut().zip(src_b.chunks_exact(2)) {
            *d = s[0] + r * (s[0] + s[1]);
        }
        pair_message(dst_a, dst_b)
    }

    pub(super) fn block_message(g: &[F128], y: &[F128], rows: usize) -> (F128, F128) {
        let (g0, g1) = g.split_at(rows);
        let (y0, y1) = y.split_at(rows);
        // Unreduced carry-less sums, exact, reduced once per call.
        let (mut q1, mut qi) = (F256Unreduced::ZERO, F256Unreduced::ZERO);
        for i in 0..rows {
            q1 = q1 ^ g1[i].mul_unreduced(y1[i]);
            qi = qi ^ (g0[i] + g1[i]).mul_unreduced(y0[i] + y1[i]);
        }
        (q1.reduce(), qi.reduce())
    }

    pub(super) fn fold_blocks_two(g_src: &[F128], y_src: &[F128], g_dst: &mut [F128], y_dst: &mut [F128], rows: usize, r: F128) {
        for i in 0..rows {
            g_dst[i] = g_src[i] + r * (g_src[i] + g_src[rows + i]);
            y_dst[i] = y_src[i] + r * (y_src[i] + y_src[rows + i]);
        }
    }

    pub(super) fn fold_blocks_two_and_message(
        g_src: &[F128],
        y_src: &[F128],
        g_dst: &mut [F128],
        y_dst: &mut [F128],
        rows: usize,
        r: F128,
    ) -> (F128, F128) {
        let (ga, gb) = g_src.split_at(2 * rows);
        let (ya, yb) = y_src.split_at(2 * rows);
        let (gd0, gd1) = g_dst.split_at_mut(rows);
        let (yd0, yd1) = y_dst.split_at_mut(rows);
        fold_blocks_two(ga, ya, gd0, yd0, rows, r);
        fold_blocks_two(gb, yb, gd1, yd1, rows, r);
        block_message(g_dst, y_dst, rows)
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use core::arch::x86_64::{
        __m128i, __m512i, _mm_extract_epi64, _mm_set_epi64x, _mm_xor_si128, _mm512_broadcast_i32x4,
        _mm512_clmulepi64_epi128, _mm512_extracti32x4_epi32, _mm512_loadu_si512, _mm512_permutex2var_epi64,
        _mm512_set_epi64, _mm512_setzero_si512, _mm512_storeu_si512, _mm512_xor_si512,
    };
    use std::sync::OnceLock;

    use super::{F128, F256Unreduced, portable};

    /// Whether the AVX-512 kernels run (`avx512f` and `vpclmulqdq`, detected once).
    pub(super) fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("vpclmulqdq")
        })
    }

    #[inline]
    #[target_feature(enable = "avx512f,vpclmulqdq")]
    fn reduce_x4(t0: __m512i, t1: __m512i) -> __m512i {
        let poly = _mm512_set_epi64(0, 0x87, 0, 0x87, 0, 0x87, 0, 0x87);
        // SAFETY: avx512f is enabled for this function.
        let t0 = _mm512_xor_si512(t0, unsafe { core::arch::x86_64::_mm512_bslli_epi128::<8>(t1) });
        _mm512_xor_si512(t0, _mm512_clmulepi64_epi128::<0x01>(t1, poly))
    }

    /// Four lane-wise GF(2^128) products.
    #[inline]
    #[target_feature(enable = "avx512f,vpclmulqdq")]
    fn mul_x4(x: __m512i, y: __m512i) -> __m512i {
        let t1 = _mm512_xor_si512(_mm512_clmulepi64_epi128::<0x01>(x, y), _mm512_clmulepi64_epi128::<0x10>(x, y));
        let t1 = reduce_x4(t1, _mm512_clmulepi64_epi128::<0x11>(x, y));
        reduce_x4(_mm512_clmulepi64_epi128::<0x00>(x, y), t1)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    fn splat(v: F128) -> __m512i {
        _mm512_broadcast_i32x4(_mm_set_epi64x(v.hi as i64, v.lo as i64))
    }

    /// Four unreduced products XORed together per lane (`lo + mid x^64 + hi x^128`); exact,
    /// reduced once by [`Self::fold`].
    #[derive(Clone, Copy)]
    struct Acc {
        lo: __m512i,
        mid: __m512i,
        hi: __m512i,
    }

    impl Acc {
        #[inline]
        #[target_feature(enable = "avx512f")]
        fn zero() -> Self {
            let z = _mm512_setzero_si512();
            Self { lo: z, mid: z, hi: z }
        }

        #[inline]
        #[target_feature(enable = "avx512f,vpclmulqdq")]
        fn mul_acc(&mut self, x: __m512i, y: __m512i) {
            self.lo = _mm512_xor_si512(self.lo, _mm512_clmulepi64_epi128::<0x00>(x, y));
            self.hi = _mm512_xor_si512(self.hi, _mm512_clmulepi64_epi128::<0x11>(x, y));
            let m = _mm512_xor_si512(_mm512_clmulepi64_epi128::<0x01>(x, y), _mm512_clmulepi64_epi128::<0x10>(x, y));
            self.mid = _mm512_xor_si512(self.mid, m);
        }

        /// XOR of the four lanes as one unreduced product.
        #[inline]
        #[target_feature(enable = "avx512f")]
        fn fold(self) -> F256Unreduced {
            #[inline]
            #[target_feature(enable = "avx512f")]
            fn xor4(v: __m512i) -> __m128i {
                let a = _mm_xor_si128(_mm512_extracti32x4_epi32::<0>(v), _mm512_extracti32x4_epi32::<1>(v));
                let b = _mm_xor_si128(_mm512_extracti32x4_epi32::<2>(v), _mm512_extracti32x4_epi32::<3>(v));
                _mm_xor_si128(a, b)
            }
            let (lo, mid, hi) = (xor4(self.lo), xor4(self.mid), xor4(self.hi));
            F256Unreduced {
                r0: _mm_extract_epi64::<0>(lo) as u64,
                r1: (_mm_extract_epi64::<1>(lo) ^ _mm_extract_epi64::<0>(mid)) as u64,
                r2: (_mm_extract_epi64::<0>(hi) ^ _mm_extract_epi64::<1>(mid)) as u64,
                r3: _mm_extract_epi64::<1>(hi) as u64,
            }
        }
    }

    /// Entries `(0, 2, 4, 6)` and `(1, 3, 5, 7)` of eight consecutive entries held in `a`
    /// (`0..4`) and `b` (`4..8`).
    #[inline]
    #[target_feature(enable = "avx512f")]
    fn deinterleave(a: __m512i, b: __m512i) -> (__m512i, __m512i) {
        let even = _mm512_set_epi64(13, 12, 9, 8, 5, 4, 1, 0);
        let odd = _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2);
        (_mm512_permutex2var_epi64(a, even, b), _mm512_permutex2var_epi64(a, odd, b))
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn load(p: &[F128], i: usize) -> __m512i {
        // SAFETY: the caller keeps i + 4 <= p.len().
        unsafe { _mm512_loadu_si512(p.as_ptr().add(i) as *const __m512i) }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn store(p: &mut [F128], i: usize, v: __m512i) {
        // SAFETY: the caller keeps i + 4 <= p.len().
        unsafe { _mm512_storeu_si512(p.as_mut_ptr().add(i) as *mut __m512i, v) }
    }

    /// Message of eight consecutive entries of each table, already deinterleaved.
    #[inline]
    #[target_feature(enable = "avx512f,vpclmulqdq")]
    fn message_acc(q1: &mut Acc, qi: &mut Acc, (ae, ao): (__m512i, __m512i), (be, bo): (__m512i, __m512i)) {
        q1.mul_acc(ao, bo);
        qi.mul_acc(_mm512_xor_si512(ae, ao), _mm512_xor_si512(be, bo));
    }

    /// # Safety
    /// `avx512f` and `vpclmulqdq` available; equal, even lengths.
    #[target_feature(enable = "avx512f,vpclmulqdq")]
    pub(super) unsafe fn pair_message(a: &[F128], b: &[F128]) -> (F128, F128) {
        let n = a.len() & !7;
        let (mut q1, mut qi) = (Acc::zero(), Acc::zero());
        // SAFETY: i + 8 <= n <= len.
        unsafe {
            for i in (0..n).step_by(8) {
                let da = deinterleave(load(a, i), load(a, i + 4));
                let db = deinterleave(load(b, i), load(b, i + 4));
                message_acc(&mut q1, &mut qi, da, db);
            }
        }
        let tail = portable::pair_message(&a[n..], &b[n..]);
        (q1.fold().reduce() + tail.0, qi.fold().reduce() + tail.1)
    }

    /// Fold eight consecutive entries at `i` of `src` into four at `i / 2` of `dst`,
    /// returning the folded values.
    #[inline]
    #[target_feature(enable = "avx512f,vpclmulqdq")]
    unsafe fn fold8(src: &[F128], dst: &mut [F128], i: usize, r: __m512i) -> __m512i {
        // SAFETY: the caller keeps i + 8 <= src.len() and i / 2 + 4 <= dst.len().
        unsafe {
            let (e, o) = deinterleave(load(src, i), load(src, i + 4));
            let f = _mm512_xor_si512(e, mul_x4(r, _mm512_xor_si512(e, o)));
            store(dst, i / 2, f);
            f
        }
    }

    /// # Safety
    /// `avx512f` and `vpclmulqdq` available; `2 dst.len() == src.len()` for both tables.
    #[target_feature(enable = "avx512f,vpclmulqdq")]
    pub(super) unsafe fn fold_pairs_two_and_message(
        src_a: &[F128],
        src_b: &[F128],
        dst_a: &mut [F128],
        dst_b: &mut [F128],
        r: F128,
    ) -> (F128, F128) {
        let n = src_a.len() & !15;
        let rr = splat(r);
        let (mut q1, mut qi) = (Acc::zero(), Acc::zero());
        // SAFETY: i + 16 <= n <= src.len(), i / 2 + 8 <= dst.len().
        unsafe {
            for i in (0..n).step_by(16) {
                let fa0 = fold8(src_a, dst_a, i, rr);
                let fa1 = fold8(src_a, dst_a, i + 8, rr);
                let fb0 = fold8(src_b, dst_b, i, rr);
                let fb1 = fold8(src_b, dst_b, i + 8, rr);
                message_acc(&mut q1, &mut qi, deinterleave(fa0, fa1), deinterleave(fb0, fb1));
            }
        }
        let tail = portable::fold_pairs_two_and_message(&src_a[n..], &src_b[n..], &mut dst_a[n / 2..], &mut dst_b[n / 2..], r);
        (q1.fold().reduce() + tail.0, qi.fold().reduce() + tail.1)
    }

    /// # Safety
    /// `avx512f` and `vpclmulqdq` available; both tables hold `2 rows` entries.
    #[target_feature(enable = "avx512f,vpclmulqdq")]
    pub(super) unsafe fn block_message(g: &[F128], y: &[F128], rows: usize) -> (F128, F128) {
        let n = rows & !3;
        let (mut q1, mut qi) = (Acc::zero(), Acc::zero());
        // SAFETY: rows + i + 4 <= 2 rows = len.
        unsafe {
            for i in (0..n).step_by(4) {
                let (g0, g1) = (load(g, i), load(g, rows + i));
                let (y0, y1) = (load(y, i), load(y, rows + i));
                message_acc(&mut q1, &mut qi, (g0, g1), (y0, y1));
            }
        }
        let (mut t1, mut ti) = (F256Unreduced::ZERO, F256Unreduced::ZERO);
        for i in n..rows {
            t1 = t1 ^ g[rows + i].mul_unreduced(y[rows + i]);
            ti = ti ^ (g[i] + g[rows + i]).mul_unreduced(y[i] + y[rows + i]);
        }
        ((q1.fold() ^ t1).reduce(), (qi.fold() ^ ti).reduce())
    }

    /// Fold entries `i .. i + 4` of blocks `2k`, `2k + 1` of `src` into block `k` of `dst`.
    #[inline]
    #[target_feature(enable = "avx512f,vpclmulqdq")]
    unsafe fn fold_block4(src: &[F128], dst: &mut [F128], k: usize, rows: usize, i: usize, r: __m512i) -> __m512i {
        // SAFETY: the caller keeps (2k + 1) rows + i + 4 <= src.len() and k rows + i + 4 <= dst.len().
        unsafe {
            let a = load(src, 2 * k * rows + i);
            let b = load(src, (2 * k + 1) * rows + i);
            let f = _mm512_xor_si512(a, mul_x4(r, _mm512_xor_si512(a, b)));
            store(dst, k * rows + i, f);
            f
        }
    }

    /// # Safety
    /// `avx512f` and `vpclmulqdq` available; four source and two destination blocks.
    #[target_feature(enable = "avx512f,vpclmulqdq")]
    pub(super) unsafe fn fold_blocks_two_and_message(
        g_src: &[F128],
        y_src: &[F128],
        g_dst: &mut [F128],
        y_dst: &mut [F128],
        rows: usize,
        r: F128,
    ) -> (F128, F128) {
        let n = rows & !3;
        let rr = splat(r);
        let (mut q1, mut qi) = (Acc::zero(), Acc::zero());
        // SAFETY: i + 4 <= n <= rows; block indices 0 .. 4 of the sources, 0 .. 2 of the destinations.
        unsafe {
            for i in (0..n).step_by(4) {
                let fg0 = fold_block4(g_src, g_dst, 0, rows, i, rr);
                let fg1 = fold_block4(g_src, g_dst, 1, rows, i, rr);
                let fy0 = fold_block4(y_src, y_dst, 0, rows, i, rr);
                let fy1 = fold_block4(y_src, y_dst, 1, rows, i, rr);
                message_acc(&mut q1, &mut qi, (fg0, fg1), (fy0, fy1));
            }
        }
        let (mut t1, mut ti) = (F256Unreduced::ZERO, F256Unreduced::ZERO);
        for i in n..rows {
            for k in 0..2 {
                g_dst[k * rows + i] = g_src[2 * k * rows + i] + r * (g_src[2 * k * rows + i] + g_src[(2 * k + 1) * rows + i]);
                y_dst[k * rows + i] = y_src[2 * k * rows + i] + r * (y_src[2 * k * rows + i] + y_src[(2 * k + 1) * rows + i]);
            }
            t1 = t1 ^ g_dst[rows + i].mul_unreduced(y_dst[rows + i]);
            ti = ti ^ (g_dst[i] + g_dst[rows + i]).mul_unreduced(y_dst[i] + y_dst[rows + i]);
        }
        ((q1.fold() ^ t1).reduce(), (qi.fold() ^ ti).reduce())
    }

    /// # Safety
    /// `avx512f` and `vpclmulqdq` available; two source blocks and one destination block.
    #[target_feature(enable = "avx512f,vpclmulqdq")]
    pub(super) unsafe fn fold_blocks_two(g_src: &[F128], y_src: &[F128], g_dst: &mut [F128], y_dst: &mut [F128], rows: usize, r: F128) {
        let n = rows & !3;
        let rr = splat(r);
        // SAFETY: i + 4 <= n <= rows.
        unsafe {
            for i in (0..n).step_by(4) {
                fold_block4(g_src, g_dst, 0, rows, i, rr);
                fold_block4(y_src, y_dst, 0, rows, i, rr);
            }
        }
        for i in n..rows {
            g_dst[i] = g_src[i] + r * (g_src[i] + g_src[rows + i]);
            y_dst[i] = y_src[i] + r * (y_src[i] + y_src[rows + i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_rng::Rng;

    fn reference_pair_message(a: &[F128], b: &[F128]) -> (F128, F128) {
        let mut q = (F128::ZERO, F128::ZERO);
        for (x, y) in a.chunks_exact(2).zip(b.chunks_exact(2)) {
            q.0 += x[1] * y[1];
            q.1 += (x[0] + x[1]) * (y[0] + y[1]);
        }
        q
    }

    fn reference_fold(t: &[F128], r: F128) -> Vec<F128> {
        t.chunks(2).map(|p| p[0] + r * (p[0] + p[1])).collect()
    }

    /// The pair kernels equal the scalar definitions at lengths around the vector widths.
    #[test]
    fn pair_kernels_match_reference() {
        let mut rng = Rng::new(0x9A1A);
        for &n in &[2usize, 4, 6, 8, 14, 16, 18, 30, 32, 34, 64, 130, 1024] {
            let a = rng.f128_vec(n);
            let b = rng.f128_vec(n);
            let r = rng.f128();
            assert_eq!(pair_message(&a, &b), reference_pair_message(&a, &b), "message n={n}");
            let (mut da, mut db) = (vec![F128::ZERO; n / 2], vec![F128::ZERO; n / 2]);
            let msg = fold_pairs_two_and_message(&a, &b, &mut da, &mut db, r);
            assert_eq!(da, reference_fold(&a, r), "fold a n={n}");
            assert_eq!(db, reference_fold(&b, r), "fold b n={n}");
            assert_eq!(msg, reference_pair_message(&da, &db), "next message n={n}");
        }
    }

    /// The block kernels equal the scalar definitions at row counts around the vector width.
    #[test]
    fn block_kernels_match_reference() {
        let mut rng = Rng::new(0xB10C);
        for &rows in &[1usize, 3, 4, 5, 8, 13, 64, 100] {
            let g = rng.f128_vec(4 * rows);
            let y = rng.f128_vec(4 * rows);
            let r = rng.f128();
            let want_msg = |g: &[F128], y: &[F128]| {
                let mut q = (F128::ZERO, F128::ZERO);
                for i in 0..rows {
                    q.0 += g[rows + i] * y[rows + i];
                    q.1 += (g[i] + g[rows + i]) * (y[i] + y[rows + i]);
                }
                q
            };
            assert_eq!(block_message(&g[..2 * rows], &y[..2 * rows], rows), want_msg(&g[..2 * rows], &y[..2 * rows]));
            let fold = |t: &[F128]| -> Vec<F128> {
                (0..t.len() / 2).map(|j| {
                    let (k, i) = (j / rows, j % rows);
                    t[2 * k * rows + i] + r * (t[2 * k * rows + i] + t[(2 * k + 1) * rows + i])
                })
                .collect()
            };
            let (mut gd, mut yd) = (vec![F128::ZERO; 2 * rows], vec![F128::ZERO; 2 * rows]);
            let msg = fold_blocks_two_and_message(&g, &y, &mut gd, &mut yd, rows, r);
            assert_eq!(gd, fold(&g), "fold g rows={rows}");
            assert_eq!(yd, fold(&y), "fold y rows={rows}");
            assert_eq!(msg, want_msg(&gd, &yd), "next message rows={rows}");
            let (mut g1, mut y1) = (vec![F128::ZERO; rows], vec![F128::ZERO; rows]);
            fold_blocks_two(&g[..2 * rows], &y[..2 * rows], &mut g1, &mut y1, rows, r);
            assert_eq!(g1, fold(&g[..2 * rows]), "fold two g rows={rows}");
            assert_eq!(y1, fold(&y[..2 * rows]), "fold two y rows={rows}");
        }
    }
}
