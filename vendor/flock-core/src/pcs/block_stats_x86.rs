//! AVX-512 and AVX2 kernels for the block statistics of the merged opening's
//! block-first phase (agg-mldsa patch, PATCH.md).
//!
//! The portable code computes, per term `j` and block `e`,
//! `S_j[e] = frob(kappa * sum_b pw[b][kinv_j] * bank[e][b], j)` with one scalar
//! multiply per `(j, e, b)` and `j` scalar squarings, and per column `c` the dot
//! product `f[c] = sum_r q[c][r] * eq_row[r]`. The kernels below compute the same
//! field elements: the sums with deferred reduction four lanes at a time (reduction is
//! F_2-linear), the products and squarings with [`ghash_mul_x4`], which is
//! field-identical to the scalar multiply. Callers check `simd::has(AVX512_CLMUL)`.

use core::arch::x86_64::__m512i;

use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::{
    field::{F128, F256Unreduced},
    simd::x86::{LaneAccX4, WideGhashX4, f128x4_loadu, f128x4_splat, f128x4_storeu, ghash_mul_x4},
};

/// Blocks per parallel task.
const E_CHUNK: usize = 64;
/// Four-lane accumulators per pass over a bank row (32 terms; 24 ZMM registers).
const GROUP: usize = 8;

/// Two-lane accumulators per pass over a bank row (8 terms; the remaining
/// registers hold the coefficient vector and the broadcast row element).
const GROUP_AVX2: usize = 4;

/// Output rows of [`s_vectors_into`]: row `jx` is the `n_e` entries at
/// `ptr + jx * stride` (rows disjoint, `stride >= n_e`).
pub(super) struct Rows {
    pub(super) ptr: *mut F128,
    pub(super) stride: usize,
}
// SAFETY: the parallel chunks write disjoint block ranges of the rows.
unsafe impl Sync for Rows {}

/// `out[jx][e] = frob(kappa * sum_b coef[jx][b] * bank[(2e + c0) * nb + b], js[jx])` for
/// `e < n_e`, where `coef[jx][b] = pw[b][(nb - js[jx]) % nb]`.
///
/// # Safety
/// The CPU has `avx512f`, `vpclmulqdq` and `sse4.1`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn s_vectors(
    bank: &[F128],
    nb: usize,
    n_e: usize,
    c0: usize,
    kappa: F128,
    js: &[usize],
    pw: &[Vec<F128>],
) -> Vec<Vec<F128>> {
    let mut flat = vec![F128::ZERO; js.len() * n_e];
    let out = Rows {
        ptr: flat.as_mut_ptr(),
        stride: n_e,
    };
    // SAFETY: forwarded features; `flat` holds every row.
    unsafe { s_vectors_into(bank, nb, n_e, c0, kappa, js, pw, &out) };
    flat.chunks(n_e.max(1))
        .take(js.len())
        .map(<[F128]>::to_vec)
        .collect()
}

/// [`s_vectors`] written into caller rows (the block-first term arena).
///
/// # Safety
/// The CPU has `avx512f`, `vpclmulqdq` and `sse4.1`; `out` has `js.len()` writable
/// rows of `n_e` entries.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn s_vectors_into(
    bank: &[F128],
    nb: usize,
    n_e: usize,
    c0: usize,
    kappa: F128,
    js: &[usize],
    pw: &[Vec<F128>],
    out: &Rows,
) {
    assert!(bank.len() >= 2 * n_e * nb, "bank covers every block");
    assert!(out.stride >= n_e, "output rows overlap");
    let nj = js.len();
    // Coefficients b-major, the term index padded to a multiple of 4 lanes:
    // coef[b * njp + jx].
    let njp = nj.next_multiple_of(4);
    let mut coef = vec![F128::ZERO; nb * njp];
    for b in 0..nb {
        for (jx, &j) in js.iter().enumerate() {
            coef[b * njp + jx] = pw[b][(nb - j) % nb];
        }
    }
    (0..n_e.div_ceil(E_CHUNK)).into_par_iter().for_each(|ch| {
        let e0 = ch * E_CHUNK;
        let ne = E_CHUNK.min(n_e - e0);
        // SAFETY: forwarded features; the bank assert covers every row read;
        // chunk `ch` writes entries [e0, e0 + ne) of each row only.
        unsafe { chunk(bank, nb, e0, ne, c0, kappa, js, &coef, njp, out) }
    });
}

/// Two-lane counterpart of [`s_vectors`] for the AVX2 tier: the inner products
/// run two terms per 256-bit VPCLMULQDQ vector, and the Frobenius step keeps
/// the portable byte tables, which do one table lookup per byte instead of
/// `j` dependent squarings.
///
/// # Safety
/// The CPU has `avx2` and `vpclmulqdq`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn s_vectors_avx2(
    bank: &[F128],
    nb: usize,
    n_e: usize,
    c0: usize,
    kappa: F128,
    js: &[usize],
    pw: &[Vec<F128>],
) -> Vec<Vec<F128>> {
    assert!(bank.len() >= 2 * n_e * nb, "bank covers every block");
    let nj = js.len();
    // Coefficients b-major, the term index padded to a multiple of two lanes:
    // coef[b * njp + jx].
    let njp = nj.next_multiple_of(2);
    let mut coef = vec![F128::ZERO; nb * njp];
    for b in 0..nb {
        for (jx, &j) in js.iter().enumerate() {
            coef[b * njp + jx] = pw[b][(nb - j) % nb];
        }
    }
    let chunks: Vec<Vec<F128>> = (0..n_e.div_ceil(E_CHUNK))
        .into_par_iter()
        .map(|ch| {
            let e0 = ch * E_CHUNK;
            let ne = E_CHUNK.min(n_e - e0);
            // SAFETY: forwarded features; the bank assert covers every row read.
            unsafe { chunk_avx2(bank, nb, e0, ne, c0, kappa, js, &coef, njp) }
        })
        .collect();
    let mut out: Vec<Vec<F128>> = (0..nj).map(|_| Vec::with_capacity(n_e)).collect();
    for (ch, buf) in chunks.iter().enumerate() {
        let ne = E_CHUNK.min(n_e - ch * E_CHUNK);
        for (jx, o) in out.iter_mut().enumerate() {
            o.extend_from_slice(&buf[jx * ne..(jx + 1) * ne]);
        }
    }
    out
}

/// One chunk of `ne` blocks starting at `e0`, into `out[jx][e0..e0 + ne]`.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f,vpclmulqdq,sse4.1")]
unsafe fn chunk(
    bank: &[F128],
    nb: usize,
    e0: usize,
    ne: usize,
    c0: usize,
    kappa: F128,
    js: &[usize],
    coef: &[F128],
    njp: usize,
    out: &Rows,
) {
    let nj = js.len();
    // SAFETY (row): rows jx < nj of `out`, entries [e0, e0 + ne) (caller).
    let row_of = |jx: usize| unsafe {
        core::slice::from_raw_parts_mut(out.ptr.add(jx * out.stride + e0), ne)
    };
    let mut lanes = [F128::ZERO; 4];
    // Pass 1: the inner products, four terms per accumulator, GROUP accumulators
    // per sweep of the bank row.
    for el in 0..ne {
        let row = &bank[(2 * (e0 + el) + c0) * nb..(2 * (e0 + el) + c0 + 1) * nb];
        let mut g0 = 0;
        while g0 < njp {
            let ng = GROUP.min((njp - g0) / 4);
            let mut acc = [LaneAccX4::zero(); GROUP];
            for (b, &x) in row.iter().enumerate() {
                let xb = f128x4_splat(x);
                let base = b * njp + g0;
                for (g, a) in acc.iter_mut().enumerate().take(ng) {
                    // SAFETY: base + 4g + 4 <= (b + 1) * njp <= coef.len().
                    let c = unsafe { f128x4_loadu(coef.as_ptr().add(base + 4 * g)) };
                    a.mul_acc(c, xb);
                }
            }
            for (g, a) in acc.iter().enumerate().take(ng) {
                // SAFETY: four writable F128 in `lanes`.
                unsafe { f128x4_storeu(lanes.as_mut_ptr(), a.reduce()) };
                for (l, &v) in lanes.iter().enumerate() {
                    let jx = g0 + 4 * g + l;
                    if jx < nj {
                        row_of(jx)[el] = v;
                    }
                }
            }
            g0 += 4 * ng;
        }
    }
    // Pass 2: times kappa, then j squarings, four blocks per vector and four
    // independent chains in flight.
    let kap = f128x4_splat(kappa);
    for (jx, &j) in js.iter().enumerate() {
        let row = row_of(jx);
        let mut e = 0;
        while e + 16 <= ne {
            // SAFETY: e + 16 <= ne = row.len().
            unsafe {
                let p = row.as_mut_ptr().add(e);
                let mut v: [__m512i; 4] =
                    core::array::from_fn(|k| ghash_mul_x4(f128x4_loadu(p.add(4 * k)), kap));
                for _ in 0..j {
                    for x in v.iter_mut() {
                        *x = ghash_mul_x4(*x, *x);
                    }
                }
                for (k, x) in v.iter().enumerate() {
                    f128x4_storeu(p.add(4 * k), *x);
                }
            }
            e += 16;
        }
        while e < ne {
            let mut x = kappa * row[e];
            for _ in 0..j {
                x = x.square();
            }
            row[e] = x;
            e += 1;
        }
    }
}

/// One chunk of `ne` blocks starting at `e0` for the AVX2 tier; returns
/// `[jx][e - e0]`.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,vpclmulqdq")]
unsafe fn chunk_avx2(
    bank: &[F128],
    nb: usize,
    e0: usize,
    ne: usize,
    c0: usize,
    kappa: F128,
    js: &[usize],
    coef: &[F128],
    njp: usize,
) -> Vec<F128> {
    use crate::simd::ghash2::{LANES, loadu, splat, storeu, wide_acc, wide_reduce, wide_zero};

    let nj = js.len();
    let mut buf = vec![F128::ZERO; nj * ne];
    let mut lanes = [F128::ZERO; LANES];
    // Pass 1: the inner products, two terms per accumulator, GROUP_AVX2
    // accumulators per sweep of the bank row.
    for el in 0..ne {
        let row = &bank[(2 * (e0 + el) + c0) * nb..(2 * (e0 + el) + c0 + 1) * nb];
        let mut g0 = 0;
        while g0 < njp {
            let ng = GROUP_AVX2.min((njp - g0) / LANES);
            let mut acc = [wide_zero(); GROUP_AVX2];
            for (b, &x) in row.iter().enumerate() {
                let xb = splat(x);
                let base = b * njp + g0;
                for (g, a) in acc.iter_mut().enumerate().take(ng) {
                    // SAFETY: base + LANES*g + LANES <= (b + 1) * njp <= coef.len().
                    let c = unsafe { loadu(coef.as_ptr().add(base + LANES * g)) };
                    wide_acc(a, c, xb);
                }
            }
            for (g, a) in acc.iter().enumerate().take(ng) {
                // SAFETY: two writable F128 in `lanes`.
                unsafe { storeu(lanes.as_mut_ptr(), wide_reduce(*a)) };
                for (l, &v) in lanes.iter().enumerate() {
                    let jx = g0 + LANES * g + l;
                    if jx < nj {
                        buf[jx * ne + el] = v;
                    }
                }
            }
            g0 += LANES * ng;
        }
    }
    // Pass 2: times kappa, then x^(2^j) through the byte tables.
    let tables = super::block_stats_portable::frob_tables();
    for (jx, &j) in js.iter().enumerate() {
        let t = &tables[j];
        for v in &mut buf[jx * ne..(jx + 1) * ne] {
            *v = super::block_stats_portable::frob_by_table(t, kappa * *v);
        }
    }
    buf
}

/// Qword indices of the even (`EVEN`) and odd (`ODD`) F128 entries of two zmm
/// `[x0..x3]`, `[x4..x7]`: `[x0, x2, x4, x6]` and `[x1, x3, x5, x7]`.
const EVEN: [i64; 8] = [0, 1, 4, 5, 8, 9, 12, 13];
const ODD: [i64; 8] = [2, 3, 6, 7, 10, 11, 14, 15];

/// One block round on one rank-1 term (`block_round_all` in `pcs.rs`): with `r`,
/// `x[i] ← x[2i] + r·(x[2i] + x[2i+1])` for `a` and `s` in place (`i < len/2`),
/// four pairs per zmm; with `msg`, the message `(Σ a1·s1, Σ (a0+a1)(s0+s1))` of
/// the (folded) pairs, accumulated from the values just computed (no second
/// pass). Accumulators: unreduced 256-bit carry-less sums of at most `len/2`
/// products per lane (exact in F_2[x] up to degree 254), reduced once.
///
/// # Safety
/// The CPU has `avx512f`, `vpclmulqdq` and `sse4.1`; `a.len() == s.len()`, a
/// multiple of 16.
#[target_feature(enable = "avx512f,vpclmulqdq,sse4.1")]
pub(super) unsafe fn round_term(
    a: &mut [F128],
    s: &mut [F128],
    r: Option<F128>,
    msg: bool,
) -> (F128, F128) {
    use core::arch::x86_64::{_mm512_loadu_si512, _mm512_permutex2var_epi64, _mm512_xor_si512};
    let len = a.len();
    debug_assert!(len == s.len() && len.is_multiple_of(16));
    // SAFETY: EVEN, ODD are 64 readable bytes.
    let (ie, io) = unsafe {
        (
            _mm512_loadu_si512(EVEN.as_ptr().cast()),
            _mm512_loadu_si512(ODD.as_ptr().cast()),
        )
    };
    let mut g1 = WideGhashX4::zero();
    let mut gi = WideGhashX4::zero();
    let mut acc_msg = |a0: __m512i, a1: __m512i, s0: __m512i, s1: __m512i| {
        let (ea, oa) = (
            _mm512_permutex2var_epi64(a0, ie, a1),
            _mm512_permutex2var_epi64(a0, io, a1),
        );
        let (es, os) = (
            _mm512_permutex2var_epi64(s0, ie, s1),
            _mm512_permutex2var_epi64(s0, io, s1),
        );
        g1.mul_acc(oa, os);
        gi.mul_acc(_mm512_xor_si512(ea, oa), _mm512_xor_si512(es, os));
    };
    match r {
        None => {
            if msg {
                for i in (0..len).step_by(8) {
                    // SAFETY: i + 8 <= len for both parts.
                    unsafe {
                        acc_msg(
                            f128x4_loadu(a.as_ptr().add(i)),
                            f128x4_loadu(a.as_ptr().add(i + 4)),
                            f128x4_loadu(s.as_ptr().add(i)),
                            f128x4_loadu(s.as_ptr().add(i + 4)),
                        );
                    }
                }
            }
        }
        Some(r) => {
            let rv = f128x4_splat(r);
            // x[2i] + r (x[2i] + x[2i+1]) for 8 outputs from 16 inputs at `p`.
            let fold8 = |p: *const F128| -> (__m512i, __m512i) {
                // SAFETY: 16 readable F128 at p (callers).
                unsafe {
                    let v0 = f128x4_loadu(p);
                    let v1 = f128x4_loadu(p.add(4));
                    let v2 = f128x4_loadu(p.add(8));
                    let v3 = f128x4_loadu(p.add(12));
                    let (e0, o0) = (
                        _mm512_permutex2var_epi64(v0, ie, v1),
                        _mm512_permutex2var_epi64(v0, io, v1),
                    );
                    let (e1, o1) = (
                        _mm512_permutex2var_epi64(v2, ie, v3),
                        _mm512_permutex2var_epi64(v2, io, v3),
                    );
                    (
                        _mm512_xor_si512(e0, ghash_mul_x4(rv, _mm512_xor_si512(e0, o0))),
                        _mm512_xor_si512(e1, ghash_mul_x4(rv, _mm512_xor_si512(e1, o1))),
                    )
                }
            };
            for i in (0..len).step_by(16) {
                // In place: outputs [i/2, i/2 + 8) are written after inputs
                // [i, i + 16) are read, and i/2 + 8 <= i + 16.
                // SAFETY: i + 16 <= len; i/2 + 8 <= len/2.
                unsafe {
                    let (fa0, fa1) = fold8(a.as_ptr().add(i));
                    let (fs0, fs1) = fold8(s.as_ptr().add(i));
                    f128x4_storeu(a.as_mut_ptr().add(i / 2), fa0);
                    f128x4_storeu(a.as_mut_ptr().add(i / 2 + 4), fa1);
                    f128x4_storeu(s.as_mut_ptr().add(i / 2), fs0);
                    f128x4_storeu(s.as_mut_ptr().add(i / 2 + 4), fs1);
                    if msg {
                        acc_msg(fa0, fa1, fs0, fs1);
                    }
                }
            }
        }
    }
    if !msg {
        return (F128::ZERO, F128::ZERO);
    }
    (g1.fold().reduce(), gi.fold().reduce())
}

/// [`round_term`] at the 256-bit and 128-bit GF(2^128) widths: the same
/// folds and message sums (deferred reduction lane-wise, reduced once per
/// term), so the returned pair and the folded parts are field-identical.
macro_rules! round_term_lanes {
    ($name:ident, $lanes:ident, $feat:literal) => {
        /// # Safety
        /// The CPU has the features of this width; `a` and `s` have equal
        /// lengths, a multiple of `4 * LANES`.
        #[target_feature(enable = $feat)]
        pub(super) unsafe fn $name(
            a: &mut [F128],
            s: &mut [F128],
            r: Option<F128>,
            msg: bool,
        ) -> (F128, F128) {
            use crate::simd::$lanes;
            const N: usize = $lanes::LANES;
            let len = a.len();
            debug_assert!(len == s.len() && len.is_multiple_of(4 * N));
            // SAFETY: every index below stays under `len` for both parts.
            unsafe {
                let mut g1 = $lanes::wide_zero();
                let mut gi = $lanes::wide_zero();
                match r {
                    None => {
                        if msg {
                            for i in (0..len).step_by(2 * N) {
                                let (ea, oa) = $lanes::even_odd(
                                    $lanes::loadu(a.as_ptr().add(i)),
                                    $lanes::loadu(a.as_ptr().add(i + N)),
                                );
                                let (es, os) = $lanes::even_odd(
                                    $lanes::loadu(s.as_ptr().add(i)),
                                    $lanes::loadu(s.as_ptr().add(i + N)),
                                );
                                $lanes::wide_acc(&mut g1, oa, os);
                                $lanes::wide_acc(&mut gi, $lanes::xor(ea, oa), $lanes::xor(es, os));
                            }
                        }
                    }
                    Some(r) => {
                        let rv = $lanes::splat(r);
                        for i in (0..len).step_by(4 * N) {
                            // In place: outputs [i/2, i/2 + 2N) are written
                            // after inputs [i, i + 4N) are read.
                            let mut out = [$lanes::zero(); 4];
                            for (k, x) in [a.as_ptr(), s.as_ptr()].into_iter().enumerate() {
                                for h in 0..2 {
                                    let p = x.add(i + 2 * N * h);
                                    let (e, o) = $lanes::even_odd($lanes::loadu(p), $lanes::loadu(p.add(N)));
                                    out[2 * k + h] = $lanes::xor(e, $lanes::mul(rv, $lanes::xor(e, o)));
                                }
                            }
                            $lanes::storeu(a.as_mut_ptr().add(i / 2), out[0]);
                            $lanes::storeu(a.as_mut_ptr().add(i / 2 + N), out[1]);
                            $lanes::storeu(s.as_mut_ptr().add(i / 2), out[2]);
                            $lanes::storeu(s.as_mut_ptr().add(i / 2 + N), out[3]);
                            if msg {
                                let (ea, oa) = $lanes::even_odd(out[0], out[1]);
                                let (es, os) = $lanes::even_odd(out[2], out[3]);
                                $lanes::wide_acc(&mut g1, oa, os);
                                $lanes::wide_acc(&mut gi, $lanes::xor(ea, oa), $lanes::xor(es, os));
                            }
                        }
                    }
                }
                if !msg {
                    return (F128::ZERO, F128::ZERO);
                }
                ($lanes::wide_fold(g1).reduce(), $lanes::wide_fold(gi).reduce())
            }
        }
    };
}

round_term_lanes!(round_term_avx2, ghash2, "avx2,vpclmulqdq");
round_term_lanes!(round_term_pclmul, ghash1, "pclmulqdq,sse4.1");

/// `sum_r q[r] * eq[r]` (equal lengths).
///
/// # Safety
/// The CPU has `avx512f`, `vpclmulqdq` and `sse4.1`.
#[target_feature(enable = "avx512f,vpclmulqdq,sse4.1")]
pub(super) unsafe fn dot(q: &[F128], eq: &[F128]) -> F128 {
    assert_eq!(q.len(), eq.len());
    let n4 = q.len() & !15;
    let mut acc = [WideGhashX4::zero(); 4];
    let mut i = 0;
    while i < n4 {
        for (k, a) in acc.iter_mut().enumerate() {
            // SAFETY: i + 4k + 4 <= n4 <= len.
            unsafe {
                a.mul_acc(
                    f128x4_loadu(q.as_ptr().add(i + 4 * k)),
                    f128x4_loadu(eq.as_ptr().add(i + 4 * k)),
                );
            }
        }
        i += 16;
    }
    let mut tail = F256Unreduced::ZERO;
    while i < q.len() {
        tail ^= q[i].mul_unreduced(eq[i]);
        i += 1;
    }
    for a in acc {
        tail ^= a.fold();
    }
    tail.reduce()
}

/// Two-lane counterpart of [`dot`] for the AVX2 tier.
///
/// # Safety
/// The CPU has `avx2` and `vpclmulqdq`.
#[target_feature(enable = "avx2,vpclmulqdq")]
pub(super) unsafe fn dot_avx2(q: &[F128], eq: &[F128]) -> F128 {
    use crate::simd::ghash2::{LANES, loadu, wide_acc, wide_fold, wide_zero};

    assert_eq!(q.len(), eq.len());
    const ACC: usize = 4;
    let n8 = q.len() & !(ACC * LANES - 1);
    let mut acc = [wide_zero(); ACC];
    let mut i = 0;
    while i < n8 {
        for (k, a) in acc.iter_mut().enumerate() {
            // SAFETY: i + LANES*(ACC-1) + LANES <= n8 <= len.
            unsafe {
                wide_acc(
                    a,
                    loadu(q.as_ptr().add(i + LANES * k)),
                    loadu(eq.as_ptr().add(i + LANES * k)),
                );
            }
        }
        i += ACC * LANES;
    }
    let mut tail = F256Unreduced::ZERO;
    while i < q.len() {
        tail ^= q[i].mul_unreduced(eq[i]);
        i += 1;
    }
    for a in acc {
        tail ^= wide_fold(a);
    }
    tail.reduce()
}

/// `out[t][i] = Σ_{e < rows} ws[t][e]·src[e·stride + i]` for `i < out[t].len()` (all
/// outputs the same length): a weighted sum of `rows` strided rows, `T` weight vectors
/// at once. Per register block of `4·G` outputs the rows are streamed once; each output
/// lane accumulates unreduced 256-bit carry-less sums of at most `rows` products (exact
/// in F_2[x] up to degree 254), reduced once at the end of the block. `T·G·3 ≤ 24`
/// accumulator registers.
///
/// # Safety
/// The CPU has `avx512f`, `vpclmulqdq`; `src` holds `(rows - 1)·stride + len` entries.
#[target_feature(enable = "avx512f,vpclmulqdq,sse4.1")]
pub(crate) unsafe fn fold_rows<const T: usize, const G: usize>(
    src: &[F128],
    stride: usize,
    rows: usize,
    ws: [&[F128]; T],
    out: [&mut [F128]; T],
) {
    let len = out[0].len();
    debug_assert!(out.iter().all(|o| o.len() == len));
    debug_assert!(ws.iter().all(|w| w.len() >= rows));
    assert!(
        rows == 0 || (rows - 1) * stride + len <= src.len(),
        "rows inside src"
    );
    let blk = 4 * G;
    let mut i0 = 0;
    while i0 + blk <= len {
        let mut acc = [[LaneAccX4::zero(); G]; T];
        for e in 0..rows {
            // SAFETY: e·stride + i0 + blk <= (rows-1)·stride + len (asserted).
            let p = unsafe { src.as_ptr().add(e * stride + i0) };
            let x: [__m512i; G] = core::array::from_fn(|g| unsafe { f128x4_loadu(p.add(4 * g)) });
            for t in 0..T {
                let w = f128x4_splat(ws[t][e]);
                for g in 0..G {
                    acc[t][g].mul_acc(x[g], w);
                }
            }
        }
        for t in 0..T {
            for g in 0..G {
                // SAFETY: i0 + 4g + 4 <= len = out[t].len().
                unsafe { f128x4_storeu(out[t].as_mut_ptr().add(i0 + 4 * g), acc[t][g].reduce()) };
            }
        }
        i0 += blk;
    }
    // Tail: fewer than 4·G outputs, scalar with unreduced sums.
    for i in i0..len {
        let mut acc = [F256Unreduced::ZERO; T];
        for e in 0..rows {
            let x = src[e * stride + i];
            for t in 0..T {
                acc[t] ^= ws[t][e].mul_unreduced(x);
            }
        }
        for t in 0..T {
            out[t][i] = acc[t].reduce();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{dot, dot_avx2, fold_rows, round_term, s_vectors, s_vectors_avx2};
    use crate::{field::F128, simd, test_rng::Rng};

    fn frob(mut x: F128, j: usize) -> F128 {
        for _ in 0..j {
            x = x.square();
        }
        x
    }

    /// The kernels give the portable block statistics bit for bit (odd term counts,
    /// chunk tails, blocks below and above one chunk).
    #[test]
    fn kernels_match_portable() {
        if !simd::has(simd::AVX512_CLMUL) {
            eprintln!("skipped: no avx512f+vpclmulqdq on this CPU");
            return;
        }
        let nb = 128;
        let mut rng = Rng::new(0xB10C);
        let pw: Vec<Vec<F128>> = (0..nb).map(|_| rng.f128_vec(nb)).collect();
        for (n_e, js) in [
            (1usize, vec![0usize, 5, 127]),
            (8, (0..128).collect::<Vec<_>>()),
            (100, vec![1, 2, 3, 64, 90]),
            (128, (0..128).step_by(3).collect()),
        ] {
            let bank = rng.f128_vec(2 * n_e * nb);
            for c0 in 0..2 {
                let kappa = rng.f128();
                // SAFETY: features checked above.
                let got = unsafe { s_vectors(&bank, nb, n_e, c0, kappa, &js, &pw) };
                for (jx, &j) in js.iter().enumerate() {
                    let kinv = (nb - j) % nb;
                    let want: Vec<F128> = (0..n_e)
                        .map(|e| {
                            let row = &bank[(2 * e + c0) * nb..(2 * e + c0 + 1) * nb];
                            let mut acc = F128::ZERO;
                            for b in 0..nb {
                                acc += pw[b][kinv] * row[b];
                            }
                            frob(kappa * acc, j)
                        })
                        .collect();
                    assert_eq!(got[jx], want, "n_e={n_e} c0={c0} j={j}");
                }
            }
        }
        for n in [0usize, 1, 15, 16, 17, 1000] {
            let q = rng.f128_vec(n);
            let eq = rng.f128_vec(n);
            let want = q.iter().zip(&eq).fold(F128::ZERO, |a, (&x, &y)| a + x * y);
            // SAFETY: features checked above.
            assert_eq!(unsafe { dot(&q, &eq) }, want, "n={n}");
        }
        // Strided row folds (one and two weight vectors, register and scalar tails).
        for (rows, stride, len) in [
            (0usize, 8usize, 8usize),
            (1, 40, 37),
            (5, 64, 64),
            (33, 100, 70),
        ] {
            let src = rng.f128_vec(rows.max(1) * stride + len);
            let w0 = rng.f128_vec(rows);
            let w1 = rng.f128_vec(rows);
            let want = |w: &[F128]| -> Vec<F128> {
                (0..len)
                    .map(|i| (0..rows).fold(F128::ZERO, |a, e| a + w[e] * src[e * stride + i]))
                    .collect()
            };
            let (mut o0, mut o1) = (vec![F128::ZERO; len], vec![F128::ZERO; len]);
            // SAFETY: features checked above; src covers every row.
            unsafe { fold_rows::<2, 4>(&src, stride, rows, [&w0, &w1], [&mut o0, &mut o1]) };
            assert_eq!(
                (o0, o1),
                (want(&w0), want(&w1)),
                "fold_rows T=2 rows={rows}"
            );
            let mut o = vec![F128::ZERO; len];
            // SAFETY: as above.
            unsafe { fold_rows::<1, 8>(&src, stride, rows, [&w0], [&mut o]) };
            assert_eq!(o, want(&w0), "fold_rows T=1 rows={rows}");
        }
        // One block round: fold plus the message from the folded values.
        for len in [16usize, 32, 64, 256] {
            let a0 = rng.f128_vec(len);
            let s0 = rng.f128_vec(len);
            let r = rng.f128();
            let mut a = a0.clone();
            let mut s = s0.clone();
            let (g1, gi) = unsafe { round_term(&mut a, &mut s, None, true) };
            let (want1, wanti) = a0
                .chunks_exact(2)
                .zip(s0.chunks_exact(2))
                .fold((F128::ZERO, F128::ZERO), |(x, y), (ap, sp)| {
                    (x + ap[1] * sp[1], y + (ap[0] + ap[1]) * (sp[0] + sp[1]))
                });
            assert_eq!((g1, gi), (want1, wanti), "round message len={len}");
            let mut a = a0.clone();
            let mut s = s0.clone();
            unsafe { round_term(&mut a, &mut s, Some(r), false) };
            let want: Vec<F128> = (0..len / 2)
                .map(|i| a0[2 * i] + r * (a0[2 * i] + a0[2 * i + 1]))
                .collect();
            assert_eq!(a[..len / 2], want, "round fold len={len}");
            let want: Vec<F128> = (0..len / 2)
                .map(|i| s0[2 * i] + r * (s0[2 * i] + s0[2 * i + 1]))
                .collect();
            assert_eq!(s[..len / 2], want, "round fold s len={len}");
            // Fused fold and message of the next round.
            let fa: Vec<F128> = a[..len / 2].to_vec();
            let fs: Vec<F128> = s[..len / 2].to_vec();
            let mut a = a0.clone();
            let mut s = s0.clone();
            let (g1, gi) = unsafe { round_term(&mut a, &mut s, Some(r), true) };
            assert_eq!(
                (&a[..len / 2], &s[..len / 2]),
                (&fa[..], &fs[..]),
                "fused fold len={len}"
            );
            let (want1, wanti) = fa
                .iter()
                .step_by(2)
                .zip(fs.iter().step_by(2))
                .zip(fa[1..].iter().step_by(2).zip(fs[1..].iter().step_by(2)))
                .fold((F128::ZERO, F128::ZERO), |(x, y), ((a0, s0), (a1, s1))| {
                    (x + *a1 * *s1, y + (*a0 + *a1) * (*s0 + *s1))
                });
            assert_eq!((g1, gi), (want1, wanti), "fused message len={len}");
        }
    }

    /// The 256-bit and 128-bit block-round kernels fold and sum exactly as the
    /// portable term round (message only, fold only, fused fold and message).
    #[test]
    fn narrow_round_term_kernels_match_portable() {
        use super::{round_term_avx2, round_term_pclmul};
        let mut rng = Rng::new(0xB10F);
        for len in [4usize, 8, 16, 64, 256] {
            let a0 = rng.f128_vec(len);
            let s0 = rng.f128_vec(len);
            let r = rng.f128();
            for (rr, msg) in [(None, true), (Some(r), false), (Some(r), true)] {
                let (mut a, mut s) = (a0.clone(), s0.clone());
                let want = crate::pcs::block_round_term_portable(&mut a, &mut s, rr, msg);
                let n = if rr.is_some() { len / 2 } else { len };
                if simd::has(simd::VPCLMUL256) && len >= 8 {
                    let (mut x, mut y) = (a0.clone(), s0.clone());
                    // SAFETY: features checked; equal lengths, a multiple of 8.
                    let got = unsafe { round_term_avx2(&mut x, &mut y, rr, msg) };
                    assert_eq!(got, want, "avx2 len={len} r={} msg={msg}", rr.is_some());
                    assert_eq!((&x[..n], &y[..n]), (&a[..n], &s[..n]), "avx2 fold len={len}");
                }
                if simd::has(simd::AVX2) {
                    let (mut x, mut y) = (a0.clone(), s0.clone());
                    // SAFETY: features checked; equal lengths, a multiple of 4.
                    let got = unsafe { round_term_pclmul(&mut x, &mut y, rr, msg) };
                    assert_eq!(got, want, "pclmul len={len} r={} msg={msg}", rr.is_some());
                    assert_eq!((&x[..n], &y[..n]), (&a[..n], &s[..n]), "pclmul fold len={len}");
                }
            }
        }
    }

    /// The AVX2 kernels give the portable block statistics bit for bit, for odd
    /// term counts, chunk tails and blocks below and above one chunk.
    #[test]
    fn avx2_kernels_match_portable() {
        if !simd::has(simd::VPCLMUL256) {
            eprintln!("skipped: no avx2+vpclmulqdq on this CPU");
            return;
        }
        let nb = 128;
        let mut rng = Rng::new(0xB10E);
        let pw: Vec<Vec<F128>> = (0..nb).map(|_| rng.f128_vec(nb)).collect();
        for (n_e, js) in [
            (1usize, vec![0usize]),
            (1, vec![0usize, 5, 127]),
            (3, vec![1usize, 2, 3, 64, 90]),
            (8, (0..128).collect::<Vec<_>>()),
            (100, vec![1, 2, 3, 64, 90]),
            (128, (0..128).step_by(3).collect()),
        ] {
            let bank = rng.f128_vec(2 * n_e * nb);
            for c0 in 0..2 {
                let kappa = rng.f128();
                // SAFETY: features checked above.
                let got = unsafe { s_vectors_avx2(&bank, nb, n_e, c0, kappa, &js, &pw) };
                for (jx, &j) in js.iter().enumerate() {
                    let kinv = (nb - j) % nb;
                    let want: Vec<F128> = (0..n_e)
                        .map(|e| {
                            let row = &bank[(2 * e + c0) * nb..(2 * e + c0 + 1) * nb];
                            let mut acc = F128::ZERO;
                            for b in 0..nb {
                                acc += pw[b][kinv] * row[b];
                            }
                            frob(kappa * acc, j)
                        })
                        .collect();
                    assert_eq!(got[jx], want, "n_e={n_e} c0={c0} j={j}");
                }
            }
        }
        for n in [0usize, 1, 7, 8, 9, 15, 16, 17, 1000] {
            let q = rng.f128_vec(n);
            let eq = rng.f128_vec(n);
            let want = q.iter().zip(&eq).fold(F128::ZERO, |a, (&x, &y)| a + x * y);
            // SAFETY: features checked above.
            assert_eq!(unsafe { dot_avx2(&q, &eq) }, want, "n={n}");
        }
    }

    /// Single-thread microbenchmark of the block-statistics kernel at the
    /// merged opening's shape: 128 terms and 2,048 blocks.
    #[test]
    #[ignore = "microbenchmark"]
    fn block_stats_speed() {
        use std::time::Instant;

        let nb = 128;
        let n_e = 2048;
        let js: Vec<usize> = (0..128).collect();
        let mut rng = Rng::new(0xB10F);
        let pw: Vec<Vec<F128>> = (0..nb).map(|_| rng.f128_vec(nb)).collect();
        let bank = rng.f128_vec(2 * n_e * nb);
        let kappa = rng.f128();
        for _ in 0..3 {
            let t = Instant::now();
            let out = if simd::has(simd::AVX512_CLMUL) {
                // SAFETY: features checked above.
                unsafe { s_vectors(&bank, nb, n_e, 0, kappa, &js, &pw) }
            } else {
                super::super::block_stats_portable::s_vectors(&bank, nb, n_e, 0, kappa, &js, &pw)
            };
            std::hint::black_box(&out);
            eprintln!("wide:  {:.2} ms", t.elapsed().as_secs_f64() * 1e3);
        }
        for _ in 0..3 {
            let t = Instant::now();
            // SAFETY: features checked above.
            let out = unsafe { s_vectors_avx2(&bank, nb, n_e, 0, kappa, &js, &pw) };
            std::hint::black_box(&out);
            eprintln!("avx2:  {:.2} ms", t.elapsed().as_secs_f64() * 1e3);
        }
    }
}
