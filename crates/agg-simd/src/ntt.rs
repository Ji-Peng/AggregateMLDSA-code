//! 16-lane AVX-512 negacyclic NTT, inverse NTT, pointwise products and CRT reconstruction
//! for the witness primes (`q = 8,380,417` and `P_2 = 2,013,265,921`, or any odd NTT prime
//! `p < 2^31` with a primitive 512th root of unity).
//!
//! Arithmetic uses unsigned 32-bit lanes with Montgomery multiplication (`R = 2^32`). Every
//! twiddle `zeta` is pre-multiplied by `R mod p` (`w = zeta * 2^32 mod p`, with quotient
//! factor `w_pinv = w * p^{-1} mod 2^32`), so `MontMul(x, w) = x * zeta mod p` keeps standard
//! (non-Montgomery) representation at every stage and matches the FIPS 204 Algorithms 41/42
//! butterfly order bit-for-bit. Eight levels are fused two per pass: `(128, 64)` and `(32, 16)`
//! across four 16-lane vectors, and `(8, 4, 2, 1)` in registers per 32-coefficient block via
//! `vpermt2d`.

use crate::detect::{backend, Backend};

/// Degree of the ML-DSA negacyclic ring `Z[X]/(X^256 + 1)`.
pub const N: usize = 256;

/// FIPS 204 prime `q = 2^23 - 2^13 + 1`.
pub const Q: u64 = 8_380_417;

/// Second CRT prime `P_2 = 15 * 2^27 + 1`.
pub const P2: u64 = 2_013_265_921;

const fn pow_mod_const(mut x: u64, mut e: u64, p: u64) -> u64 {
    let mut r = 1u64;
    x %= p;
    while e > 0 {
        if e & 1 == 1 {
            r = (r as u128 * x as u128 % p as u128) as u64;
        }
        x = (x as u128 * x as u128 % p as u128) as u64;
        e >>= 1;
    }
    r
}

/// `Q^{-1} mod P_2`.
pub const Q_INV_MOD_P2: u64 = pow_mod_const(Q % P2, P2 - 2, P2);

/// Shoup quotient `floor((Q^{-1} mod P_2) * 2^32 / P_2)` for 64-bit lane CRT reduction.
pub const Q_INV_SHOUP_P2: u64 = ((Q_INV_MOD_P2 as u128) << 32)
    .div_euclid(P2 as u128) as u64;

/// Precomputed scalar Montgomery multiplier `w = (z * 2^32) mod p` and `w_pinv = w * p^{-1} mod 2^32`.
#[derive(Clone, Copy, Debug)]
pub struct MontConst {
    pub w: u32,
    pub w_pinv: u32,
}

impl MontConst {
    #[inline]
    pub const fn new(z: u32, p: u32, pinv: u32) -> Self {
        let w = (((z as u64) << 32) % (p as u64)) as u32;
        let w_pinv = w.wrapping_mul(pinv);
        Self { w, w_pinv }
    }
}

/// 64-byte-aligned 16-lane Montgomery twiddle vector for in-register stages (`len = 8, 4, 2, 1`).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct MontVec16 {
    pub w: [u32; 16],
    pub w_pinv: [u32; 16],
}

impl MontVec16 {
    #[inline]
    pub fn from_zetas(zetas: [u32; 16], p: u32, pinv: u32) -> Self {
        let mut w = [0u32; 16];
        let mut w_pinv = [0u32; 16];
        for i in 0..16 {
            let mc = MontConst::new(zetas[i], p, pinv);
            w[i] = mc.w;
            w_pinv[i] = mc.w_pinv;
        }
        Self { w, w_pinv }
    }
}

/// Per-level Montgomery and Shoup twiddle tables precomputed from an `NttPrime`.
#[derive(Clone, Debug)]
pub struct NttTable {
    /// Prime modulus `p < 2^31`.
    pub p: u32,
    /// `p^{-1} mod 2^32`.
    pub pinv: u32,
    /// Barrett constant `floor(2^64 / p)` for scalar fallback.
    pub barrett: u64,
    /// Scalar Shoup forward table `(z, floor(z * 2^32 / p))` from `NttPrime`.
    pub forward_shoup: [(u32, u32); N],
    /// Scalar Shoup inverse table `(p - z, floor((p - z) * 2^32 / p))` from `NttPrime`.
    pub inverse_shoup: [(u32, u32); N],
    /// Scalar Shoup `(256^{-1} mod p, floor(256^{-1} * 2^32 / p))`.
    pub n_inv_shoup: (u32, u32),
    /// Montgomery multiplier for `z = 2^32 mod p` (`w = 2^64 mod p`), used to finish variable products.
    pub r_mont: MontConst,
    /// Forward scalar twiddles for `len = 128, 64, 32, 16` (`m = 1..16`).
    pub fwd_scalar: [MontConst; 16],
    /// Forward 16-lane twiddles for `len = 8` (one per 32-coeff block `b = 0..8`).
    pub fwd_len8: [MontVec16; 8],
    /// Forward 16-lane twiddles for `len = 4` (one per 32-coeff block `b = 0..8`).
    pub fwd_len4: [MontVec16; 8],
    /// Forward 16-lane twiddles for `len = 2` (one per 32-coeff block `b = 0..8`).
    pub fwd_len2: [MontVec16; 8],
    /// Forward 16-lane twiddles for `len = 1` (one per 32-coeff block `b = 0..8`).
    pub fwd_len1: [MontVec16; 8],
    /// Inverse 16-lane twiddles for `len = 1` (one per 32-coeff block `b = 0..8`).
    pub inv_len1: [MontVec16; 8],
    /// Inverse 16-lane twiddles for `len = 2` (one per 32-coeff block `b = 0..8`).
    pub inv_len2: [MontVec16; 8],
    /// Inverse 16-lane twiddles for `len = 4` (one per 32-coeff block `b = 0..8`).
    pub inv_len4: [MontVec16; 8],
    /// Inverse 16-lane twiddles for `len = 8` (one per 32-coeff block `b = 0..8`).
    pub inv_len8: [MontVec16; 8],
    /// Inverse scalar twiddles for `len = 16, 32, 64, 128` (`m = 1..16`).
    pub inv_scalar: [MontConst; 16],
    /// Montgomery multiplier for `256^{-1} mod p` (fused into the `len = 128` inverse stage).
    pub n_inv_mont: MontConst,
    /// Montgomery multiplier for `(inverse[1] * 256^{-1}) mod p` (fused into `len = 128`).
    pub inv1_ninv_mont: MontConst,
}

#[inline]
const fn inv_mod_2_32(p: u32) -> u32 {
    let mut inv = 1u32;
    let mut i = 0;
    while i < 5 {
        inv = inv.wrapping_mul(2u32.wrapping_sub(p.wrapping_mul(inv)));
        i += 1;
    }
    inv
}

impl NttTable {
    /// Precompute per-level AVX-512 Montgomery twiddle tables from `NttPrime`'s forward and
    /// inverse Shoup arrays.
    pub fn from_shoup(
        p_u64: u64,
        barrett: u64,
        forward_shoup: [(u32, u32); N],
        inverse_shoup: [(u32, u32); N],
        n_inv_shoup: (u32, u32),
    ) -> Self {
        assert!(p_u64 < (1 << 31) && p_u64 % 2 == 1, "NttTable requires odd p < 2^31");
        let p = p_u64 as u32;
        let pinv = inv_mod_2_32(p);
        debug_assert_eq!(p.wrapping_mul(pinv), 1);

        let r_mod_p = ((1u64 << 32) % p_u64) as u32;
        let r_mont = MontConst::new(r_mod_p, p, pinv);

        let fwd_scalar = std::array::from_fn(|m| MontConst::new(forward_shoup[m].0, p, pinv));
        let fwd_len8 = std::array::from_fn(|b| {
            let z = std::array::from_fn(|i| forward_shoup[16 + 2 * b + (i / 8)].0);
            MontVec16::from_zetas(z, p, pinv)
        });
        let fwd_len4 = std::array::from_fn(|b| {
            let z = std::array::from_fn(|i| forward_shoup[32 + 4 * b + (i / 4)].0);
            MontVec16::from_zetas(z, p, pinv)
        });
        let fwd_len2 = std::array::from_fn(|b| {
            let z = std::array::from_fn(|i| forward_shoup[64 + 8 * b + (i / 2)].0);
            MontVec16::from_zetas(z, p, pinv)
        });
        let fwd_len1 = std::array::from_fn(|b| {
            let z = std::array::from_fn(|i| forward_shoup[128 + 16 * b + i].0);
            MontVec16::from_zetas(z, p, pinv)
        });

        let inv_len1 = std::array::from_fn(|b| {
            let z = std::array::from_fn(|i| inverse_shoup[255 - (16 * b + i)].0);
            MontVec16::from_zetas(z, p, pinv)
        });
        let inv_len2 = std::array::from_fn(|b| {
            let z = std::array::from_fn(|i| inverse_shoup[127 - (8 * b + (i / 2))].0);
            MontVec16::from_zetas(z, p, pinv)
        });
        let inv_len4 = std::array::from_fn(|b| {
            let z = std::array::from_fn(|i| inverse_shoup[63 - (4 * b + (i / 4))].0);
            MontVec16::from_zetas(z, p, pinv)
        });
        let inv_len8 = std::array::from_fn(|b| {
            let z = std::array::from_fn(|i| inverse_shoup[31 - (2 * b + (i / 8))].0);
            MontVec16::from_zetas(z, p, pinv)
        });
        let inv_scalar = std::array::from_fn(|m| MontConst::new(inverse_shoup[m].0, p, pinv));

        let n_inv_mont = MontConst::new(n_inv_shoup.0, p, pinv);
        let inv1_ninv = ((u64::from(inverse_shoup[1].0) * u64::from(n_inv_shoup.0)) % p_u64) as u32;
        let inv1_ninv_mont = MontConst::new(inv1_ninv, p, pinv);

        Self {
            p,
            pinv,
            barrett,
            forward_shoup,
            inverse_shoup,
            n_inv_shoup,
            r_mont,
            fwd_scalar,
            fwd_len8,
            fwd_len4,
            fwd_len2,
            fwd_len1,
            inv_len1,
            inv_len2,
            inv_len4,
            inv_len8,
            inv_scalar,
            n_inv_mont,
            inv1_ninv_mont,
        }
    }

    #[inline(always)]
    fn shoup_mul(&self, x: u32, (z, zq): (u32, u32)) -> u32 {
        let p = self.p;
        let q = ((u64::from(zq) * u64::from(x)) >> 32) as u32;
        let r = z.wrapping_mul(x).wrapping_sub(q.wrapping_mul(p));
        r.min(r.wrapping_sub(p))
    }

    #[inline(always)]
    fn reduce_u64(&self, x: u64) -> u64 {
        let p = u64::from(self.p);
        let quotient = ((u128::from(x) * u128::from(self.barrett)) >> 64) as u64;
        let r = x - quotient * p;
        r.min(r.wrapping_sub(p))
    }

    #[inline]
    fn reduce_i64(&self, x: i64) -> u64 {
        let p = u64::from(self.p);
        let r = self.reduce_u64(x.unsigned_abs());
        if x >= 0 || r == 0 {
            r
        } else {
            p - r
        }
    }

    /// Scalar reference forward NTT (FIPS 204 Algorithm 41).
    pub fn ntt_scalar(&self, w: &[u64; N]) -> [u64; N] {
        let p = self.p;
        let mut a: [u32; N] = std::array::from_fn(|i| w[i] as u32);
        let mut m = 0usize;
        let mut len = 128usize;
        while len >= 1 {
            let mut start = 0usize;
            while start < N {
                m += 1;
                let z = self.forward_shoup[m];
                let (low, high) = a[start..start + 2 * len].split_at_mut(len);
                for (x, y) in low.iter_mut().zip(high.iter_mut()) {
                    let t = self.shoup_mul(*y, z);
                    let d = x.wrapping_sub(t);
                    *y = d.min(d.wrapping_add(p));
                    let s = *x + t;
                    *x = s.min(s.wrapping_sub(p));
                }
                start += 2 * len;
            }
            len /= 2;
        }
        a.map(u64::from)
    }

    /// Scalar reference inverse NTT (FIPS 204 Algorithm 42).
    pub fn intt_scalar(&self, w_hat: &[u64; N]) -> [u64; N] {
        let p = self.p;
        let mut a: [u32; N] = std::array::from_fn(|i| w_hat[i] as u32);
        let mut m = 256usize;
        let mut len = 1usize;
        while len < N {
            let mut start = 0usize;
            while start < N {
                m -= 1;
                let z = self.inverse_shoup[m];
                let (low, high) = a[start..start + 2 * len].split_at_mut(len);
                for (x, y) in low.iter_mut().zip(high.iter_mut()) {
                    let t = *x;
                    let s = t + *y;
                    *x = s.min(s.wrapping_sub(p));
                    let d = t.wrapping_sub(*y);
                    *y = self.shoup_mul(d.min(d.wrapping_add(p)), z);
                }
                start += 2 * len;
            }
            len *= 2;
        }
        a.map(|x| u64::from(self.shoup_mul(x, self.n_inv_shoup)))
    }

    /// Scalar reference pointwise multiplication.
    pub fn pointwise_scalar(&self, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        std::array::from_fn(|i| self.reduce_u64(a[i] * b[i]))
    }

    /// Scalar reference signed NTT.
    pub fn ntt_signed_scalar(&self, w: &[i64; N]) -> [u64; N] {
        let a = std::array::from_fn(|i| self.reduce_i64(w[i]));
        self.ntt_scalar(&a)
    }
}

/// 16-lane forward NTT (FIPS 204 Algorithm 41); dispatches to AVX-512 when available.
#[inline]
pub fn ntt16(table: &NttTable, w: &[u64; N]) -> [u64; N] {
    #[cfg(target_arch = "x86_64")]
    if backend() == Backend::Avx512 {
        return unsafe { ntt16_avx512(table, w) };
    }
    table.ntt_scalar(w)
}

/// 16-lane inverse NTT (FIPS 204 Algorithm 42); dispatches to AVX-512 when available.
#[inline]
pub fn intt16(table: &NttTable, w_hat: &[u64; N]) -> [u64; N] {
    #[cfg(target_arch = "x86_64")]
    if backend() == Backend::Avx512 {
        return unsafe { intt16_avx512(table, w_hat) };
    }
    table.intt_scalar(w_hat)
}

/// 16-lane forward NTT of a signed polynomial; dispatches to AVX-512 when available.
#[inline]
pub fn ntt_signed16(table: &NttTable, w: &[i64; N]) -> [u64; N] {
    #[cfg(target_arch = "x86_64")]
    if backend() == Backend::Avx512 {
        return unsafe { ntt_signed16_avx512(table, w) };
    }
    table.ntt_signed_scalar(w)
}

/// 16-lane coefficient-wise product modulo `table.p`; dispatches to AVX-512 when available.
#[inline]
pub fn pointwise16(table: &NttTable, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
    #[cfg(target_arch = "x86_64")]
    if backend() == Backend::Avx512 {
        return unsafe { pointwise16_avx512(table, a, b) };
    }
    table.pointwise_scalar(a, b)
}

/// 16-lane dot product of `L = 4` NTT-domain polynomials `sum_{s=0..4} a[s][j] * b[s][j] mod p`.
///
/// Lazy range: each product `a[s][j] * b[s][j] < p^2 < 2^62` (since `p < 2^31`), so the sum of
/// four unreduced products in `u64` is `< 4 * p^2 < 2^64` and never overflows `u64` before a
/// single reduction per coefficient.
#[inline]
pub fn pointwise_dot4(table: &NttTable, a: &[[u64; N]; 4], b: &[[u64; N]; 4]) -> [u64; N] {
    #[cfg(target_arch = "x86_64")]
    if backend() == Backend::Avx512 {
        return unsafe { pointwise_dot4_avx512(table, a, b) };
    }
    std::array::from_fn(|j| {
        // Each product < p^2 < 2^62; four terms sum to < 2^64 without overflow.
        let sum = a[0][j] * b[0][j] + a[1][j] * b[1][j] + a[2][j] * b[2][j] + a[3][j] * b[3][j];
        table.reduce_u64(sum)
    })
}

/// Scalar CRT reconstruction: `x in (-M/2, M/2]` with `x = x1 mod Q`, `x = x2 mod P2`, `M = Q * P2`.
#[inline]
pub fn crt_centered_scalar(x1: u64, x2: u64) -> i64 {
    let diff = (x2 + P2 - x1 % P2) % P2;
    let k = ((diff as u128 * Q_INV_MOD_P2 as u128) % P2 as u128) as u64;
    let m = Q as i128 * P2 as i128;
    let mut x = x1 as i128 + Q as i128 * k as i128;
    if x > m / 2 {
        x -= m;
    }
    x as i64
}

/// CRT reconstruction across all 256 coefficients on 8 x 64-bit lanes.
#[inline]
pub fn crt16(x1: &[u64; N], x2: &[u64; N], out: &mut [i64; N]) {
    #[cfg(target_arch = "x86_64")]
    if backend() == Backend::Avx512 {
        unsafe { crt16_avx512(x1, x2, out) };
        return;
    }
    for i in 0..N {
        out[i] = crt_centered_scalar(x1[i], x2[i]);
    }
}

/// Compute `T[rho] = 2^d (c * t1[rho])` over `Z[X]/(X^256 + 1)` from the sparse challenge
/// positions of `c` into a flat `4 * 256 = 1024`-element `i32` slice `out`.
///
/// Value range: `0 <= t1 < 2^10`, so `|t1 << d| < 2^23`. Across the `<= 39` nonzero positions
/// of `c`, the accumulated sum stays within `|T| <= 39 * (2^10 - 1) * 2^13 = 326,836,224 < 2^29`,
/// which never overflows `i32`.
#[inline]
pub fn public_t16_into(
    coeffs: &[i8; N],
    positions: &[u8],
    t1: &[[u16; N]; 4],
    d: u32,
    out: &mut [i32],
) {
    assert_eq!(out.len(), 4 * N);
    #[cfg(target_arch = "x86_64")]
    if backend() == Backend::Avx512 {
        unsafe { avx512::public_t16_avx512(coeffs, positions, t1, d, out) };
        return;
    }
    out.fill(0);
    for r in 0..4 {
        let row_out = &mut out[r * N..(r + 1) * N];
        for &pos in positions {
            let p = usize::from(pos);
            let sign = i32::from(coeffs[p]);
            let (low, high) = t1[r].split_at(N - p);
            for (slot, &x) in row_out[p..].iter_mut().zip(low) {
                *slot += sign * i32::from(x);
            }
            for (slot, &x) in row_out[..p].iter_mut().zip(high) {
                *slot -= sign * i32::from(x);
            }
        }
        for x in row_out.iter_mut() {
            *x <<= d;
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::*;
    use std::arch::x86_64::*;

    // Permutation indices for the 4 in-register butterfly levels on a 32-element block `(u, v)`.
    // Each `IDX*_LO` / `IDX*_HI` pair is an involution swapping the needed sub-blocks between `u` and `v`.
    const IDX8_LO: [i32; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23];
    const IDX8_HI: [i32; 16] = [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31];
    const IDX4_LO: [i32; 16] = [0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27];
    const IDX4_HI: [i32; 16] = [4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31];
    const IDX2_LO: [i32; 16] = [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29];
    const IDX2_HI: [i32; 16] = [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31];
    const IDX1_LO: [i32; 16] = [0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30];
    const IDX1_HI: [i32; 16] = [1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31];
    const OUT_LO: [i32; 16] = [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23];
    const OUT_HI: [i32; 16] = [8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31];
    const EVEN: [i32; 16] = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30];
    const ODD: [i32; 16] = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31];

    #[inline(always)]
    unsafe fn load_idx(idx: &[i32; 16]) -> __m512i {
        unsafe { _mm512_loadu_si512(idx.as_ptr().cast()) }
    }

    /// `a + b mod p` for `a, b in [0, p)` (`p < 2^31`, so `a + b < 2p < 2^32`).
    #[inline(always)]
    unsafe fn add_mod(a: __m512i, b: __m512i, p: __m512i) -> __m512i {
        unsafe {
            let s = _mm512_add_epi32(a, b);
            _mm512_min_epu32(s, _mm512_sub_epi32(s, p))
        }
    }

    /// `a - b mod p` for `a, b in [0, p)` (`p < 2^31`).
    #[inline(always)]
    unsafe fn sub_mod(a: __m512i, b: __m512i, p: __m512i) -> __m512i {
        unsafe {
            let d = _mm512_sub_epi32(a, b);
            _mm512_min_epu32(d, _mm512_add_epi32(d, p))
        }
    }

    /// Unreduced `a + p - b` in `[1, 2p - 1] subset [0, 2^32)` for `a, b in [0, p)`.
    #[inline(always)]
    unsafe fn sub_unreduced(a: __m512i, b: __m512i, p: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi32(_mm512_add_epi32(a, p), b) }
    }

    /// Unsigned Montgomery product `x * w * 2^{-32} mod p` on 16 lanes for any `x in [0, 2^32)`
    /// and constant `w in [0, p)` with `wp = w * p^{-1} mod 2^32`.
    ///
    /// Value range: `0 <= x * w < 2^32 * p` and `0 <= q * p < 2^32 * p` where `q = x * wp mod 2^32`.
    /// Because `x * w == q * p (mod 2^32)`, their low 32-bit unsigned halves are identical, so
    /// 32-bit lane subtraction leaves the exact quotient `u = (x * w - q * p) / 2^32` in the odd
    /// 32-bit lanes with `-(p - 1) <= u <= p - 1`. Since `2p < 2^32`, `min_epu32(u, u + p)` maps
    /// `u` to `[0, p)`.
    #[inline(always)]
    unsafe fn mont_mul_vec(
        x: __m512i,
        w: __m512i,
        w_odd: __m512i,
        wp: __m512i,
        wp_odd: __m512i,
        p: __m512i,
    ) -> __m512i {
        unsafe {
            let x_odd = _mm512_shuffle_epi32::<0xF5>(x);
            let even = _mm512_mul_epu32(x, w);
            let odd = _mm512_mul_epu32(x_odd, w_odd);
            let even_t = _mm512_mul_epu32(x, wp);
            let odd_t = _mm512_mul_epu32(x_odd, wp_odd);
            let even_diff = _mm512_sub_epi32(even, _mm512_mul_epu32(even_t, p));
            let odd_diff = _mm512_sub_epi32(odd, _mm512_mul_epu32(odd_t, p));
            let u = _mm512_mask_shuffle_epi32::<0xF5>(odd_diff, 0x5555, even_diff);
            _mm512_min_epu32(u, _mm512_add_epi32(u, p))
        }
    }

    #[inline(always)]
    unsafe fn mont_mul_const(x: __m512i, c: MontConst, p: __m512i) -> __m512i {
        unsafe {
            let w = _mm512_set1_epi32(c.w as i32);
            let wp = _mm512_set1_epi32(c.w_pinv as i32);
            mont_mul_vec(x, w, w, wp, wp, p)
        }
    }

    #[inline(always)]
    unsafe fn mont_mul_tw16(x: __m512i, tw: &MontVec16, p: __m512i) -> __m512i {
        unsafe {
            let w = _mm512_load_si512(tw.w.as_ptr().cast());
            let wp = _mm512_load_si512(tw.w_pinv.as_ptr().cast());
            let w_odd = _mm512_shuffle_epi32::<0xF5>(w);
            let wp_odd = _mm512_shuffle_epi32::<0xF5>(wp);
            mont_mul_vec(x, w, w_odd, wp, wp_odd, p)
        }
    }

    /// Variable-by-variable Montgomery product `a * b * 2^{-32} mod p` in `[0, p)` for `a, b in [0, p)`.
    #[inline(always)]
    unsafe fn mont_mul_var(a: __m512i, b: __m512i, p: __m512i, pinv: __m512i) -> __m512i {
        unsafe {
            let a_odd = _mm512_shuffle_epi32::<0xF5>(a);
            let b_odd = _mm512_shuffle_epi32::<0xF5>(b);
            let even = _mm512_mul_epu32(a, b);
            let odd = _mm512_mul_epu32(a_odd, b_odd);
            let even_t = _mm512_mul_epu32(even, pinv);
            let odd_t = _mm512_mul_epu32(odd, pinv);
            let even_diff = _mm512_sub_epi32(even, _mm512_mul_epu32(even_t, p));
            let odd_diff = _mm512_sub_epi32(odd, _mm512_mul_epu32(odd_t, p));
            let u = _mm512_mask_shuffle_epi32::<0xF5>(odd_diff, 0x5555, even_diff);
            _mm512_min_epu32(u, _mm512_add_epi32(u, p))
        }
    }

    #[inline(always)]
    unsafe fn load_u64_as_u32_16(ptr: *const u64) -> __m512i {
        unsafe {
            let lo = _mm512_cvtepi64_epi32(_mm512_loadu_si512(ptr.cast()));
            let hi = _mm512_cvtepi64_epi32(_mm512_loadu_si512(ptr.add(8).cast()));
            _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi)
        }
    }

    #[inline(always)]
    unsafe fn store_u32_16_as_u64(ptr: *mut u64, v: __m512i) {
        unsafe {
            let lo = _mm512_castsi512_si256(v);
            let hi = _mm512_extracti64x4_epi64::<1>(v);
            _mm512_storeu_si512(ptr.cast(), _mm512_cvtepu32_epi64(lo));
            _mm512_storeu_si512(ptr.add(8).cast(), _mm512_cvtepu32_epi64(hi));
        }
    }

    #[inline(always)]
    unsafe fn ntt16_core_to_u64(table: &NttTable, a: &mut [u32; N], out: &mut [u64; N]) {
        unsafe {
            let p = _mm512_set1_epi32(table.p as i32);
            let ptr = a.as_mut_ptr();
            let out_ptr = out.as_mut_ptr();

            // Pass 0: fused levels len = 128 and len = 64.
            let z1 = table.fwd_scalar[1];
            let z2 = table.fwd_scalar[2];
            let z3 = table.fwd_scalar[3];
            for j in (0..64).step_by(16) {
                let q0 = _mm512_loadu_si512(ptr.add(j).cast());
                let q1 = _mm512_loadu_si512(ptr.add(64 + j).cast());
                let q2 = _mm512_loadu_si512(ptr.add(128 + j).cast());
                let q3 = _mm512_loadu_si512(ptr.add(192 + j).cast());

                let t2 = mont_mul_const(q2, z1, p);
                let b0 = add_mod(q0, t2, p);
                let b2 = sub_mod(q0, t2, p);

                let t3 = mont_mul_const(q3, z1, p);
                let b1 = add_mod(q1, t3, p);
                let b3 = sub_mod(q1, t3, p);

                let u1 = mont_mul_const(b1, z2, p);
                _mm512_storeu_si512(ptr.add(j).cast(), add_mod(b0, u1, p));
                _mm512_storeu_si512(ptr.add(64 + j).cast(), sub_mod(b0, u1, p));

                let u3 = mont_mul_const(b3, z3, p);
                _mm512_storeu_si512(ptr.add(128 + j).cast(), add_mod(b2, u3, p));
                _mm512_storeu_si512(ptr.add(192 + j).cast(), sub_mod(b2, u3, p));
            }

            // Pass 1: fused levels len = 32 and len = 16.
            for g in 0..4 {
                let base = g * 64;
                let z0 = table.fwd_scalar[4 + g];
                let z1 = table.fwd_scalar[8 + 2 * g];
                let z2 = table.fwd_scalar[8 + 2 * g + 1];

                let q0 = _mm512_loadu_si512(ptr.add(base).cast());
                let q1 = _mm512_loadu_si512(ptr.add(base + 16).cast());
                let q2 = _mm512_loadu_si512(ptr.add(base + 32).cast());
                let q3 = _mm512_loadu_si512(ptr.add(base + 48).cast());

                let t2 = mont_mul_const(q2, z0, p);
                let b0 = add_mod(q0, t2, p);
                let b2 = sub_mod(q0, t2, p);

                let t3 = mont_mul_const(q3, z0, p);
                let b1 = add_mod(q1, t3, p);
                let b3 = sub_mod(q1, t3, p);

                let u1 = mont_mul_const(b1, z1, p);
                _mm512_storeu_si512(ptr.add(base).cast(), add_mod(b0, u1, p));
                _mm512_storeu_si512(ptr.add(base + 16).cast(), sub_mod(b0, u1, p));

                let u3 = mont_mul_const(b3, z2, p);
                _mm512_storeu_si512(ptr.add(base + 32).cast(), add_mod(b2, u3, p));
                _mm512_storeu_si512(ptr.add(base + 48).cast(), sub_mod(b2, u3, p));
            }

            // Passes 2 & 3 in registers per 32-element block: levels len = 8, 4, 2, 1, widened to u64.
            let idx8_lo = load_idx(&IDX8_LO);
            let idx8_hi = load_idx(&IDX8_HI);
            let idx4_lo = load_idx(&IDX4_LO);
            let idx4_hi = load_idx(&IDX4_HI);
            let idx2_lo = load_idx(&IDX2_LO);
            let idx2_hi = load_idx(&IDX2_HI);
            let idx1_lo = load_idx(&IDX1_LO);
            let idx1_hi = load_idx(&IDX1_HI);
            let out_lo = load_idx(&OUT_LO);
            let out_hi = load_idx(&OUT_HI);

            for b in 0..8 {
                let base = b * 32;
                let x = _mm512_loadu_si512(ptr.add(base).cast());
                let y = _mm512_loadu_si512(ptr.add(base + 16).cast());

                // len = 8
                let u8 = _mm512_permutex2var_epi32(x, idx8_lo, y);
                let v8 = _mm512_permutex2var_epi32(x, idx8_hi, y);
                let t8 = mont_mul_tw16(v8, &table.fwd_len8[b], p);
                let b_lo = add_mod(u8, t8, p);
                let b_hi = sub_mod(u8, t8, p);

                // len = 4
                let u4 = _mm512_permutex2var_epi32(b_lo, idx4_lo, b_hi);
                let v4 = _mm512_permutex2var_epi32(b_lo, idx4_hi, b_hi);
                let t4 = mont_mul_tw16(v4, &table.fwd_len4[b], p);
                let c_lo = add_mod(u4, t4, p);
                let c_hi = sub_mod(u4, t4, p);

                // len = 2
                let u2 = _mm512_permutex2var_epi32(c_lo, idx2_lo, c_hi);
                let v2 = _mm512_permutex2var_epi32(c_lo, idx2_hi, c_hi);
                let t2 = mont_mul_tw16(v2, &table.fwd_len2[b], p);
                let d_lo = add_mod(u2, t2, p);
                let d_hi = sub_mod(u2, t2, p);

                // len = 1
                let u1 = _mm512_permutex2var_epi32(d_lo, idx1_lo, d_hi);
                let v1 = _mm512_permutex2var_epi32(d_lo, idx1_hi, d_hi);
                let t1 = mont_mul_tw16(v1, &table.fwd_len1[b], p);
                let e_lo = add_mod(u1, t1, p);
                let e_hi = sub_mod(u1, t1, p);

                let r0 = _mm512_permutex2var_epi32(e_lo, out_lo, e_hi);
                let r1 = _mm512_permutex2var_epi32(e_lo, out_hi, e_hi);
                store_u32_16_as_u64(out_ptr.add(base), r0);
                store_u32_16_as_u64(out_ptr.add(base + 16), r1);
            }
        }
    }

    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
    pub unsafe fn ntt16_avx512(table: &NttTable, w: &[u64; N]) -> [u64; N] {
        unsafe {
            let mut a = [0u32; N];
            for j in (0..N).step_by(16) {
                let v = load_u64_as_u32_16(w.as_ptr().add(j));
                _mm512_storeu_si512(a.as_mut_ptr().add(j).cast(), v);
            }
            let mut out = [0u64; N];
            ntt16_core_to_u64(table, &mut a, &mut out);
            out
        }
    }

    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
    pub unsafe fn ntt_signed16_avx512(table: &NttTable, w: &[i64; N]) -> [u64; N] {
        unsafe {
            let p_i64 = i64::from(table.p);
            let p_u64_vec = _mm512_set1_epi64((table.p - 1) as i64);
            let p_u32_vec = _mm512_set1_epi32(table.p as i32);
            let zero32 = _mm512_setzero_si512();
            let mut a = [0u32; N];
            for j in (0..N).step_by(16) {
                let v0 = _mm512_loadu_si512(w.as_ptr().add(j).cast());
                let v1 = _mm512_loadu_si512(w.as_ptr().add(j + 8).cast());
                let m0 = _mm512_cmpgt_epu64_mask(_mm512_abs_epi64(v0), p_u64_vec);
                let m1 = _mm512_cmpgt_epu64_mask(_mm512_abs_epi64(v1), p_u64_vec);
                if (m0 | m1) == 0 {
                    let lo = _mm512_cvtepi64_epi32(v0);
                    let hi = _mm512_cvtepi64_epi32(v1);
                    let x = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi);
                    let neg = _mm512_cmplt_epi32_mask(x, zero32);
                    let red = _mm512_mask_add_epi32(x, neg, x, p_u32_vec);
                    _mm512_storeu_si512(a.as_mut_ptr().add(j).cast(), red);
                } else {
                    for k in 0..16 {
                        let x = w[j + k];
                        a[j + k] = if -p_i64 < x && x < p_i64 {
                            if x < 0 { (x + p_i64) as u32 } else { x as u32 }
                        } else {
                            table.reduce_i64(x) as u32
                        };
                    }
                }
            }
            let mut out = [0u64; N];
            ntt16_core_to_u64(table, &mut a, &mut out);
            out
        }
    }

    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
    pub unsafe fn intt16_avx512(table: &NttTable, w_hat: &[u64; N]) -> [u64; N] {
        unsafe {
            let p = _mm512_set1_epi32(table.p as i32);
            let mut a = [0u32; N];
            let ptr = a.as_mut_ptr();

            let even = load_idx(&EVEN);
            let odd = load_idx(&ODD);
            let idx1_lo = load_idx(&IDX1_LO);
            let idx1_hi = load_idx(&IDX1_HI);
            let idx2_lo = load_idx(&IDX2_LO);
            let idx2_hi = load_idx(&IDX2_HI);
            let idx4_lo = load_idx(&IDX4_LO);
            let idx4_hi = load_idx(&IDX4_HI);
            let idx8_lo = load_idx(&IDX8_LO);
            let idx8_hi = load_idx(&IDX8_HI);

            // Passes 0 & 1 in registers per 32-element block: levels len = 1, 2, 4, 8.
            for b in 0..8 {
                let base = b * 32;
                let x = load_u64_as_u32_16(w_hat.as_ptr().add(base));
                let y = load_u64_as_u32_16(w_hat.as_ptr().add(base + 16));

                // len = 1
                let u1 = _mm512_permutex2var_epi32(x, even, y);
                let v1 = _mm512_permutex2var_epi32(x, odd, y);
                let b_lo = add_mod(u1, v1, p);
                let b_hi = mont_mul_tw16(sub_unreduced(u1, v1, p), &table.inv_len1[b], p);

                // len = 2
                let u2 = _mm512_permutex2var_epi32(b_lo, idx1_lo, b_hi);
                let v2 = _mm512_permutex2var_epi32(b_lo, idx1_hi, b_hi);
                let c_lo = add_mod(u2, v2, p);
                let c_hi = mont_mul_tw16(sub_unreduced(u2, v2, p), &table.inv_len2[b], p);

                // len = 4
                let u4 = _mm512_permutex2var_epi32(c_lo, idx2_lo, c_hi);
                let v4 = _mm512_permutex2var_epi32(c_lo, idx2_hi, c_hi);
                let d_lo = add_mod(u4, v4, p);
                let d_hi = mont_mul_tw16(sub_unreduced(u4, v4, p), &table.inv_len4[b], p);

                // len = 8
                let u8 = _mm512_permutex2var_epi32(d_lo, idx4_lo, d_hi);
                let v8 = _mm512_permutex2var_epi32(d_lo, idx4_hi, d_hi);
                let e_lo = add_mod(u8, v8, p);
                let e_hi = mont_mul_tw16(sub_unreduced(u8, v8, p), &table.inv_len8[b], p);

                let r0 = _mm512_permutex2var_epi32(e_lo, idx8_lo, e_hi);
                let r1 = _mm512_permutex2var_epi32(e_lo, idx8_hi, e_hi);
                _mm512_storeu_si512(ptr.add(base).cast(), r0);
                _mm512_storeu_si512(ptr.add(base + 16).cast(), r1);
            }

            // Pass 2: fused levels len = 16 and len = 32.
            for g in 0..4 {
                let base = g * 64;
                let z1 = table.inv_scalar[15 - 2 * g];
                let z2 = table.inv_scalar[15 - (2 * g + 1)];
                let z0 = table.inv_scalar[7 - g];

                let q0 = _mm512_loadu_si512(ptr.add(base).cast());
                let q1 = _mm512_loadu_si512(ptr.add(base + 16).cast());
                let q2 = _mm512_loadu_si512(ptr.add(base + 32).cast());
                let q3 = _mm512_loadu_si512(ptr.add(base + 48).cast());

                let b0 = add_mod(q0, q1, p);
                let b1 = mont_mul_const(sub_unreduced(q0, q1, p), z1, p);
                let b2 = add_mod(q2, q3, p);
                let b3 = mont_mul_const(sub_unreduced(q2, q3, p), z2, p);

                _mm512_storeu_si512(ptr.add(base).cast(), add_mod(b0, b2, p));
                _mm512_storeu_si512(
                    ptr.add(base + 32).cast(),
                    mont_mul_const(sub_unreduced(b0, b2, p), z0, p),
                );
                _mm512_storeu_si512(ptr.add(base + 16).cast(), add_mod(b1, b3, p));
                _mm512_storeu_si512(
                    ptr.add(base + 48).cast(),
                    mont_mul_const(sub_unreduced(b1, b3, p), z0, p),
                );
            }

            // Pass 3: fused levels len = 64 and len = 128 + 256^{-1} scaling, widened to u64.
            let z_64_0 = table.inv_scalar[3];
            let z_64_1 = table.inv_scalar[2];
            let z_ninv = table.n_inv_mont;
            let z_inv1_ninv = table.inv1_ninv_mont;
            let mut out = [0u64; N];
            let out_ptr = out.as_mut_ptr();

            for j in (0..64).step_by(16) {
                let q0 = _mm512_loadu_si512(ptr.add(j).cast());
                let q1 = _mm512_loadu_si512(ptr.add(64 + j).cast());
                let q2 = _mm512_loadu_si512(ptr.add(128 + j).cast());
                let q3 = _mm512_loadu_si512(ptr.add(192 + j).cast());

                let b0 = add_mod(q0, q1, p);
                let b1 = mont_mul_const(sub_unreduced(q0, q1, p), z_64_0, p);
                let b2 = add_mod(q2, q3, p);
                let b3 = mont_mul_const(sub_unreduced(q2, q3, p), z_64_1, p);

                // Unreduced sum b0 + b2 is in [0, 2p) subset [0, 2^32), which mont_mul_const reduces.
                let r0 = mont_mul_const(_mm512_add_epi32(b0, b2), z_ninv, p);
                let r2 = mont_mul_const(sub_unreduced(b0, b2, p), z_inv1_ninv, p);
                let r1 = mont_mul_const(_mm512_add_epi32(b1, b3), z_ninv, p);
                let r3 = mont_mul_const(sub_unreduced(b1, b3, p), z_inv1_ninv, p);

                store_u32_16_as_u64(out_ptr.add(j), r0);
                store_u32_16_as_u64(out_ptr.add(64 + j), r1);
                store_u32_16_as_u64(out_ptr.add(128 + j), r2);
                store_u32_16_as_u64(out_ptr.add(192 + j), r3);
            }
            out
        }
    }

    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
    pub unsafe fn pointwise16_avx512(table: &NttTable, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        unsafe {
            let p = _mm512_set1_epi32(table.p as i32);
            let pinv = _mm512_set1_epi32(table.pinv as i32);
            let r_mont = table.r_mont;
            let mut out = [0u64; N];
            for j in (0..N).step_by(16) {
                let va = load_u64_as_u32_16(a.as_ptr().add(j));
                let vb = load_u64_as_u32_16(b.as_ptr().add(j));
                let t = mont_mul_var(va, vb, p, pinv);
                let c = mont_mul_const(t, r_mont, p);
                store_u32_16_as_u64(out.as_mut_ptr().add(j), c);
            }
            out
        }
    }

    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
    pub unsafe fn pointwise_dot4_avx512(
        table: &NttTable,
        a: &[[u64; N]; 4],
        b: &[[u64; N]; 4],
    ) -> [u64; N] {
        unsafe {
            let p = _mm512_set1_epi32(table.p as i32);
            let pinv = _mm512_set1_epi32(table.pinv as i32);
            let r_mont = table.r_mont;
            let mut out = [0u64; N];
            for j in (0..N).step_by(16) {
                // Lazy 2-product 64-bit accumulation per half: for a, b in [0, p) with p < 2^31,
                // a0*b0 + a1*b1 <= 2*(p - 1)^2 < 2^32 * (p - 1) < 2^32 * p, which stays within
                // the exact unsigned Montgomery reduction bound (-(p - 1) <= u <= p - 1).
                let reduce_pair = |s0: usize, s1: usize| {
                    let a0 = load_u64_as_u32_16(a[s0].as_ptr().add(j));
                    let b0 = load_u64_as_u32_16(b[s0].as_ptr().add(j));
                    let a1 = load_u64_as_u32_16(a[s1].as_ptr().add(j));
                    let b1 = load_u64_as_u32_16(b[s1].as_ptr().add(j));

                    let even = _mm512_add_epi64(
                        _mm512_mul_epu32(a0, b0),
                        _mm512_mul_epu32(a1, b1),
                    );
                    let odd = _mm512_add_epi64(
                        _mm512_mul_epu32(
                            _mm512_shuffle_epi32::<0xF5>(a0),
                            _mm512_shuffle_epi32::<0xF5>(b0),
                        ),
                        _mm512_mul_epu32(
                            _mm512_shuffle_epi32::<0xF5>(a1),
                            _mm512_shuffle_epi32::<0xF5>(b1),
                        ),
                    );
                    let even_t = _mm512_mul_epu32(even, pinv);
                    let odd_t = _mm512_mul_epu32(odd, pinv);
                    let even_diff = _mm512_sub_epi32(even, _mm512_mul_epu32(even_t, p));
                    let odd_diff = _mm512_sub_epi32(odd, _mm512_mul_epu32(odd_t, p));
                    let u = _mm512_mask_shuffle_epi32::<0xF5>(odd_diff, 0x5555, even_diff);
                    _mm512_min_epu32(u, _mm512_add_epi32(u, p))
                };
                let sum01 = reduce_pair(0, 1);
                let sum23 = reduce_pair(2, 3);
                let sum = add_mod(sum01, sum23, p);
                let c = mont_mul_const(sum, r_mont, p);
                store_u32_16_as_u64(out.as_mut_ptr().add(j), c);
            }
            out
        }
    }

    /// CRT reconstruction on 8 x 64-bit lanes:
    /// `k = (x2 - x1) * q^{-1} mod P_2`, `x = x1 + q * k` centered to `|x| <= Q * P_2 / 2`.
    ///
    /// Value ranges in each 64-bit lane:
    /// - `x1 in [0, Q)` (`Q = 8,380,417 < P_2`) and `x2 in [0, P_2)` (`P_2 = 2,013,265,921 < 2^31`).
    /// - `diff = (x2 - x1) mod P_2 in [0, P_2)` fits in 31 bits (`_mm512_mul_epu32` is exact 64-bit).
    /// - Shoup quotient `quot = (diff * Q_INV_SHOUP_P2) >> 32` is short by at most 1, so
    ///   `rem = diff * Q_INV_MOD_P2 - quot * P_2 in [0, 2 * P_2)`, reduced to `k in [0, P_2)`.
    /// - `x = x1 + Q * k in [0, Q * P_2)` (`Q * P_2 < 2^54`), centered by subtracting `Q * P_2`
    ///   when `x > (Q * P_2) / 2`.
    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
    pub unsafe fn crt16_avx512(x1: &[u64; N], x2: &[u64; N], out: &mut [i64; N]) {
        unsafe {
            const M: i64 = (Q as i64) * (P2 as i64);
            const HALF_M: i64 = M / 2;
            let p2_v = _mm512_set1_epi64(P2 as i64);
            let q_v = _mm512_set1_epi64(Q as i64);
            let q_inv_v = _mm512_set1_epi64(Q_INV_MOD_P2 as i64);
            let q_inv_shoup_v = _mm512_set1_epi64(Q_INV_SHOUP_P2 as i64);
            let m_v = _mm512_set1_epi64(M);
            let half_m_v = _mm512_set1_epi64(HALF_M);

            for i in (0..N).step_by(8) {
                let v1 = _mm512_loadu_si512(x1.as_ptr().add(i).cast());
                let v2 = _mm512_loadu_si512(x2.as_ptr().add(i).cast());
                // If x1 >= P2 (only in synthetic tests), reduce x1 mod P2 first.
                let v1_ge = _mm512_cmpge_epu64_mask(v1, p2_v);
                let v1_mod = if v1_ge == 0 {
                    v1
                } else {
                    let mut tmp = [0u64; 8];
                    _mm512_storeu_si512(tmp.as_mut_ptr().cast(), v1);
                    for x in &mut tmp {
                        *x %= P2;
                    }
                    _mm512_loadu_si512(tmp.as_ptr().cast())
                };

                let lt = _mm512_cmplt_epu64_mask(v2, v1_mod);
                let raw_diff = _mm512_sub_epi64(v2, v1_mod);
                let diff = _mm512_mask_add_epi64(raw_diff, lt, raw_diff, p2_v);

                let quot = _mm512_srli_epi64::<32>(_mm512_mul_epu32(diff, q_inv_shoup_v));
                let rem = _mm512_sub_epi64(
                    _mm512_mul_epu32(diff, q_inv_v),
                    _mm512_mul_epu32(quot, p2_v),
                );
                let k = _mm512_min_epu64(rem, _mm512_sub_epi64(rem, p2_v));

                let x = _mm512_add_epi64(v1, _mm512_mul_epu32(k, q_v));
                let gt = _mm512_cmpgt_epi64_mask(x, half_m_v);
                let centered = _mm512_mask_sub_epi64(x, gt, x, m_v);
                _mm512_storeu_si512(out.as_mut_ptr().add(i).cast(), centered);
            }
        }
    }

    /// 16-lane AVX-512 rotate-and-add for `public_t`:
    /// For each polynomial `r in 0..4`, builds the 512-entry negacyclic rotation table
    /// `[-t1[r] << d, +t1[r] << d]` in L1 and accumulates `sign * rot(t1[r] << d, p)` across
    /// all nonzero positions `p` in 16 ZMM registers before storing once to `out`.
    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
    pub unsafe fn public_t16_avx512(
        coeffs: &[i8; N],
        positions: &[u8],
        t1: &[[u16; N]; 4],
        d: u32,
        out: &mut [i32],
    ) {
        unsafe {
            let zero = _mm512_setzero_si512();
            let shift = _mm_cvtsi32_si128(d as i32);
            let mut rot = [0i32; 2 * N];
            let rot_ptr = rot.as_mut_ptr();

            for r in 0..4 {
                let t1_ptr = t1[r].as_ptr();
                for k in 0..16 {
                    let u = _mm256_loadu_si256(t1_ptr.add(16 * k).cast());
                    let pos_v = _mm512_sll_epi32(_mm512_cvtepu16_epi32(u), shift);
                    let neg_v = _mm512_sub_epi32(zero, pos_v);
                    _mm512_storeu_si512(rot_ptr.add(16 * k).cast(), neg_v);
                    _mm512_storeu_si512(rot_ptr.add(N + 16 * k).cast(), pos_v);
                }

                let mut acc = [zero; 16];
                for &pos in positions {
                    let p = usize::from(pos);
                    let src = rot_ptr.add(N - p);
                    if coeffs[p] > 0 {
                        for k in 0..16 {
                            acc[k] = _mm512_add_epi32(
                                acc[k],
                                _mm512_loadu_si512(src.add(16 * k).cast()),
                            );
                        }
                    } else {
                        for k in 0..16 {
                            acc[k] = _mm512_sub_epi32(
                                acc[k],
                                _mm512_loadu_si512(src.add(16 * k).cast()),
                            );
                        }
                    }
                }

                let dst = out.as_mut_ptr().add(r * N);
                for k in 0..16 {
                    _mm512_storeu_si512(dst.add(16 * k).cast(), acc[k]);
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub use avx512::{
    crt16_avx512, intt16_avx512, ntt16_avx512, ntt_signed16_avx512, pointwise16_avx512,
    pointwise_dot4_avx512,
};
