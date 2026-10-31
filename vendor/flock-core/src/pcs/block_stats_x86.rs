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
    assert!(bank.len() >= 2 * n_e * nb, "bank covers every block");
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
    let chunks: Vec<Vec<F128>> = (0..n_e.div_ceil(E_CHUNK))
        .into_par_iter()
        .map(|ch| {
            let e0 = ch * E_CHUNK;
            let ne = E_CHUNK.min(n_e - e0);
            // SAFETY: forwarded features; the bank assert covers every row read.
            unsafe { chunk(bank, nb, e0, ne, c0, kappa, js, &coef, njp) }
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

/// One chunk of `ne` blocks starting at `e0`; returns `[jx][e - e0]`.
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
) -> Vec<F128> {
    let nj = js.len();
    let mut buf = vec![F128::ZERO; nj * ne];
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
                        buf[jx * ne + el] = v;
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
        let row = &mut buf[jx * ne..(jx + 1) * ne];
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
    buf
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

#[cfg(test)]
mod tests {
    use super::{dot, dot_avx2, s_vectors, s_vectors_avx2};
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
