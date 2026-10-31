//! Portable block statistics of the merged opening's block-first phase.
//!
//! Same field elements as the reference loop in `block_first_phase` (and as
//! the AVX-512 kernels of `block_stats_x86`), computed per term `j` and block
//! `e` as `S_j[e] = frob(kappa * sum_b pw[b][kinv_j] * bank[e][b], j)`, with
//! three changes of evaluation order only:
//! - blocks outer, terms inner: each 2 KiB bank row is read once from memory
//!   and stays in L1 for all terms (the term-outer loop streamed the whole
//!   bank once per term);
//! - the 128-term inner products accumulate 256-bit carry-less products and
//!   reduce once per `(j, e)` (exact in `F_2[x]`: degree <= 254, no carries,
//!   no bound on the number of terms);
//! - `frob(x, j) = x^(2^j)` is F_2-linear, so it is the XOR of 16 byte-table
//!   entries (one per byte of `x`) instead of `j` squarings.

use std::sync::OnceLock;

use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::field::{F128, F256Unreduced};

/// Blocks per parallel task.
const E_CHUNK: usize = 64;
/// Terms per sweep of a bank row: `GROUP` accumulators of 32 bytes each stay
/// in registers or L1 while the row streams once.
const GROUP: usize = 8;

/// The 16 byte tables of one Frobenius power: entry `[i][v]` is
/// `(v * x^(8 i))^(2^j)`.
pub(super) type FrobTable = [[F128; 256]; 16];

/// Byte tables of `x -> x^(2^j)` for `j in 0..128` (8 MiB), built once per
/// process. Universal: they depend only on the field.
///
/// Built serially: this runs inside `OnceLock::get_or_init`, and a parallel
/// build there deadlocks when several workers enter it at once, because the
/// waiting workers block the pool that the initializer's `par_iter` needs.
pub(super) fn frob_tables() -> &'static [FrobTable] {
    static TABLES: OnceLock<Vec<FrobTable>> = OnceLock::new();
    TABLES.get_or_init(|| {
        (0..128usize)
            .map(|j| {
                // Images of the monomial basis x^k, k < 128.
                let basis: Vec<F128> = (0..128)
                    .map(|k| {
                        let mut x = if k < 64 {
                            F128::new(1u64 << k, 0)
                        } else {
                            F128::new(0, 1u64 << (k - 64))
                        };
                        for _ in 0..j {
                            x = x.square();
                        }
                        x
                    })
                    .collect();
                let mut t: FrobTable = [[F128::ZERO; 256]; 16];
                for (i, tab) in t.iter_mut().enumerate() {
                    for v in 1usize..256 {
                        let low = v.trailing_zeros() as usize;
                        tab[v] = tab[v & (v - 1)] + basis[8 * i + low];
                    }
                }
                t
            })
            .collect()
    })
}

/// `x^(2^j)` from the byte tables of `j`.
#[inline]
pub(super) fn frob_by_table(t: &FrobTable, x: F128) -> F128 {
    let lo = x.lo.to_le_bytes();
    let hi = x.hi.to_le_bytes();
    let mut acc = F128::ZERO;
    for i in 0..8 {
        acc += t[i][lo[i] as usize];
        acc += t[8 + i][hi[i] as usize];
    }
    acc
}

/// `out[jx][e] = frob(kappa * sum_b coef[jx][b] * bank[(2e + c0) * nb + b], js[jx])`
/// for `e < n_e`, where `coef[jx][b] = pw[b][(nb - js[jx]) % nb]`. Same layout
/// and values as `block_stats_x86::s_vectors`.
pub(super) fn s_vectors(
    bank: &[F128],
    nb: usize,
    n_e: usize,
    c0: usize,
    kappa: F128,
    js: &[usize],
    pw: &[Vec<F128>],
) -> Vec<Vec<F128>> {
    assert!(bank.len() >= 2 * n_e * nb, "bank covers every block");
    assert!(js.iter().all(|&j| j < 128) && nb <= 128);
    let nj = js.len();
    // Coefficients b-major (coef[b * nj + jx]) so a sweep of the row reads
    // them contiguously.
    let mut coef = vec![F128::ZERO; nb * nj];
    for b in 0..nb {
        for (jx, &j) in js.iter().enumerate() {
            coef[b * nj + jx] = pw[b][(nb - j) % nb];
        }
    }
    let tables = frob_tables();
    let chunks: Vec<Vec<F128>> = (0..n_e.div_ceil(E_CHUNK))
        .into_par_iter()
        .map(|ch| {
            let e0 = ch * E_CHUNK;
            let ne = E_CHUNK.min(n_e - e0);
            let mut buf = vec![F128::ZERO; nj * ne];
            for el in 0..ne {
                let row = &bank[(2 * (e0 + el) + c0) * nb..(2 * (e0 + el) + c0 + 1) * nb];
                let mut g0 = 0;
                while g0 < nj {
                    let ng = GROUP.min(nj - g0);
                    let mut acc = [F256Unreduced::ZERO; GROUP];
                    for (b, &x) in row.iter().enumerate() {
                        let c = &coef[b * nj + g0..b * nj + g0 + ng];
                        for (a, &cv) in acc.iter_mut().zip(c) {
                            *a ^= cv.mul_unreduced(x);
                        }
                    }
                    for (g, a) in acc.iter().enumerate().take(ng) {
                        buf[(g0 + g) * ne + el] = a.reduce();
                    }
                    g0 += ng;
                }
            }
            for (jx, &j) in js.iter().enumerate() {
                let t = &tables[j];
                for v in &mut buf[jx * ne..(jx + 1) * ne] {
                    *v = frob_by_table(t, kappa * *v);
                }
            }
            buf
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

/// `sum_r q[r] * eq[r]` (equal lengths) with one reduction.
pub(super) fn dot(q: &[F128], eq: &[F128]) -> F128 {
    assert_eq!(q.len(), eq.len());
    // XOR of exact 256-bit products (see the module docs).
    let mut acc = [F256Unreduced::ZERO; 4];
    let mut qc = q.chunks_exact(4);
    let mut ec = eq.chunks_exact(4);
    for (x, y) in (&mut qc).zip(&mut ec) {
        for k in 0..4 {
            acc[k] ^= x[k].mul_unreduced(y[k]);
        }
    }
    for (&x, &y) in qc.remainder().iter().zip(ec.remainder()) {
        acc[0] ^= x.mul_unreduced(y);
    }
    (acc[0] ^ acc[1] ^ acc[2] ^ acc[3]).reduce()
}

#[cfg(test)]
mod tests {
    use super::{dot, frob_by_table, frob_tables, s_vectors};
    use crate::{field::F128, test_rng::Rng};

    fn frob(mut x: F128, j: usize) -> F128 {
        for _ in 0..j {
            x = x.square();
        }
        x
    }

    #[test]
    fn frob_tables_match_squaring() {
        let mut rng = Rng::new(0xF20B);
        let t = frob_tables();
        for j in 0..128 {
            for _ in 0..8 {
                let x = rng.f128();
                assert_eq!(frob_by_table(&t[j], x), frob(x, j), "j={j}");
            }
        }
    }

    /// Bit for bit the reference loop (odd term counts, chunk tails, blocks
    /// below and above one chunk).
    #[test]
    fn kernels_match_reference() {
        let nb = 128;
        let mut rng = Rng::new(0xB10D);
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
                let got = s_vectors(&bank, nb, n_e, c0, kappa, &js, &pw);
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
        for n in [0usize, 1, 3, 4, 5, 1000] {
            let q = rng.f128_vec(n);
            let eq = rng.f128_vec(n);
            let want = q.iter().zip(&eq).fold(F128::ZERO, |a, (&x, &y)| a + x * y);
            assert_eq!(dot(&q, &eq), want, "n={n}");
        }
    }
}
