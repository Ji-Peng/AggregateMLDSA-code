//! F_p and K arithmetic for the tiers without IFMA: eight lanes on AVX-512F ([`W8`]) and four
//! lanes on AVX2 ([`W4`]).
//!
//! A 64 x 64-bit product is four `vpmuludq` partials of the 32-bit halves. Partials are summed
//! lazily in four radix-2^32 limbs ([`Acc`], weights 2^0, 2^32, 2^64, 2^96) and reduced with
//! 2^64 = 59 (mod p). Storage stays canonical `u64`: a value is split into its halves only in
//! registers ([`S`]). Every kernel is exact, so its results are bit-identical to the scalar
//! oracle and to the IFMA kernels.
//!
//! The generic kernels are `#[inline(always)]` bodies over [`Simd`]; each is instantiated by a
//! thin `#[target_feature]` entry point per backend (`avx512f` for [`W8`], `avx2` for [`W4`]),
//! into which the body and the backend's primitives inline.
#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

/// The 32 low bits of a lane.
const LOW32: u64 = 0xffff_ffff;

/// The lane primitives of one backend. Every function is `unsafe`: the caller guarantees the
/// backend's CPU features (checked once by [`crate::detect::backend`]) and that every pointer
/// addresses the elements the function reads or writes.
pub trait Simd: Copy + Send + Sync + 'static {
    /// A vector of `u64` lanes.
    type V: Copy;
    /// Lanes per vector.
    const LANES: usize;
    unsafe fn splat(x: u64) -> Self::V;
    unsafe fn load(p: *const u64) -> Self::V;
    unsafe fn store(p: *mut u64, x: Self::V);
    unsafe fn add(a: Self::V, b: Self::V) -> Self::V;
    unsafe fn sub(a: Self::V, b: Self::V) -> Self::V;
    unsafe fn and(a: Self::V, b: Self::V) -> Self::V;
    unsafe fn shr32(a: Self::V) -> Self::V;
    unsafe fn shl32(a: Self::V) -> Self::V;
    unsafe fn shl1(a: Self::V) -> Self::V;
    unsafe fn shl2(a: Self::V) -> Self::V;
    unsafe fn shl6(a: Self::V) -> Self::V;
    /// Products of the low 32-bit halves, unsigned.
    unsafe fn mul32(a: Self::V, b: Self::V) -> Self::V;
    /// Products of the low 32-bit halves read as signed `i32`.
    unsafe fn mul32s(a: Self::V, b: Self::V) -> Self::V;
    /// Any `u64` lane minus `p` when it is at least `p` (canonical out).
    unsafe fn canon(x: Self::V) -> Self::V;
    /// `a + b mod p` for canonical lanes.
    unsafe fn add_mod(a: Self::V, b: Self::V) -> Self::V;
    /// `a - b mod p` for canonical lanes.
    unsafe fn sub_mod(a: Self::V, b: Self::V) -> Self::V;
    /// `a + b` with a carry out of 64 bits folded back as `+ 59`. The caller guarantees that
    /// a carrying sum wraps to below `2^64 - 59`.
    unsafe fn add_fold(a: Self::V, b: Self::V) -> Self::V;
    /// [`Self::add_fold`] for a second operand below 2^63 (any first operand: a carrying sum
    /// wraps to below the second operand).
    unsafe fn add_fold_small(a: Self::V, small: Self::V) -> Self::V;
    /// The canonical residue of a signed `i64` lane (`k + p` when `k < 0`).
    unsafe fn residue(k: Self::V) -> Self::V;
    /// Rows `2k` and `2k + 1` of `2 LANES` consecutive rows (`x0` the first `LANES`).
    unsafe fn deinterleave(x0: Self::V, x1: Self::V) -> (Self::V, Self::V);
    /// [`Self::deinterleave`] with the items in the backend's lane order, which [`Self::reorder`]
    /// applies to `LANES` consecutive per-item values (cheaper where the natural order costs
    /// cross-lane permutes).
    unsafe fn deinterleave_any(x0: Self::V, x1: Self::V) -> (Self::V, Self::V);
    /// Puts `LANES` consecutive per-item values into the lane order of
    /// [`Self::deinterleave_any`]; the order is an involution, so this also undoes it.
    unsafe fn reorder(x: Self::V) -> Self::V;
    /// `LANES` consecutive `i32` values sign-extended to `i64` lanes.
    unsafe fn load_i32(p: *const i32) -> Self::V;
    /// The four patterns `a` of `LANES` cell quartets (`p[4 l + a]`, `l < LANES`) as
    /// sign-extended `i64` lanes. Reads exactly `4 LANES` values.
    unsafe fn strided4_i32(p: *const i32) -> [Self::V; 4];
    /// [`Self::strided4_i32`] over `i8` cells.
    unsafe fn strided4_i8(p: *const i8) -> [Self::V; 4];
    /// The lanes in order (the first `LANES` entries).
    unsafe fn lanes(x: Self::V) -> [u64; 8];
    /// Whether any lane exceeds `bound` as an unsigned integer.
    unsafe fn any_above(x: Self::V, bound: u64) -> bool;
}

/// AVX-512F, eight lanes.
#[derive(Clone, Copy, Debug)]
pub struct W8;

/// AVX2, four lanes.
#[derive(Clone, Copy, Debug)]
pub struct W4;

impl Simd for W8 {
    type V = __m512i;
    const LANES: usize = 8;
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn splat(x: u64) -> __m512i {
        _mm512_set1_epi64(x as i64)
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn load(p: *const u64) -> __m512i {
        // SAFETY: the caller guarantees eight readable words at `p`.
        unsafe { _mm512_loadu_si512(p.cast()) }
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn store(p: *mut u64, x: __m512i) {
        // SAFETY: the caller guarantees eight writable words at `p`.
        unsafe { _mm512_storeu_si512(p.cast(), x) }
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn add(a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi64(a, b)
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn sub(a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi64(a, b)
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn and(a: __m512i, b: __m512i) -> __m512i {
        _mm512_and_si512(a, b)
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn shr32(a: __m512i) -> __m512i {
        _mm512_srli_epi64::<32>(a)
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn shl32(a: __m512i) -> __m512i {
        _mm512_slli_epi64::<32>(a)
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn shl1(a: __m512i) -> __m512i {
        _mm512_slli_epi64::<1>(a)
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn shl2(a: __m512i) -> __m512i {
        _mm512_slli_epi64::<2>(a)
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn shl6(a: __m512i) -> __m512i {
        _mm512_slli_epi64::<6>(a)
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn mul32(a: __m512i, b: __m512i) -> __m512i {
        _mm512_mul_epu32(a, b)
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn mul32s(a: __m512i, b: __m512i) -> __m512i {
        _mm512_mul_epi32(a, b)
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn canon(x: __m512i) -> __m512i {
        _mm512_min_epu64(x, _mm512_sub_epi64(x, _mm512_set1_epi64(crate::P as i64)))
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn add_mod(a: __m512i, b: __m512i) -> __m512i {
        // SAFETY: AVX-512F is enabled.
        unsafe { Self::canon(Self::add_fold(a, b)) }
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn sub_mod(a: __m512i, b: __m512i) -> __m512i {
        let s = _mm512_sub_epi64(a, b);
        // A borrow wrapped by 2^64; subtracting 59 more leaves a - b + p.
        _mm512_mask_sub_epi64(s, _mm512_cmplt_epu64_mask(a, b), s, _mm512_set1_epi64(59))
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn add_fold(a: __m512i, b: __m512i) -> __m512i {
        let s = _mm512_add_epi64(a, b);
        _mm512_mask_add_epi64(s, _mm512_cmplt_epu64_mask(s, a), s, _mm512_set1_epi64(59))
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn add_fold_small(a: __m512i, small: __m512i) -> __m512i {
        // SAFETY: AVX-512F is enabled.
        unsafe { Self::add_fold(a, small) }
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn residue(k: __m512i) -> __m512i {
        let negative = _mm512_cmplt_epi64_mask(k, _mm512_setzero_si512());
        _mm512_mask_add_epi64(k, negative, k, _mm512_set1_epi64(crate::P as i64))
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn deinterleave(x0: __m512i, x1: __m512i) -> (__m512i, __m512i) {
        let even = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
        let odd = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
        (_mm512_permutex2var_epi64(x0, even, x1), _mm512_permutex2var_epi64(x0, odd, x1))
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn deinterleave_any(x0: __m512i, x1: __m512i) -> (__m512i, __m512i) {
        // SAFETY: AVX-512F is enabled.
        unsafe { Self::deinterleave(x0, x1) }
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn reorder(x: __m512i) -> __m512i {
        x
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn load_i32(p: *const i32) -> __m512i {
        // SAFETY: the caller guarantees eight readable values at `p`.
        unsafe { _mm512_cvtepi32_epi64(_mm256_loadu_si256(p.cast())) }
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn strided4_i32(p: *const i32) -> [__m512i; 4] {
        // SAFETY: the caller guarantees 32 readable values at `p`.
        unsafe {
            let x0 = _mm512_loadu_si512(p.cast());
            let x1 = _mm512_loadu_si512(p.add(16).cast());
            let base = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 28, 24, 20, 16, 12, 8, 4, 0);
            std::array::from_fn(|a| {
                let idx = _mm512_add_epi32(base, _mm512_set1_epi32(a as i32));
                _mm512_cvtepi32_epi64(_mm512_castsi512_si256(_mm512_permutex2var_epi32(x0, idx, x1)))
            })
        }
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn strided4_i8(p: *const i8) -> [__m512i; 4] {
        // SAFETY: the caller guarantees 32 readable bytes at `p`.
        unsafe {
            // Each 64-bit lane holds the four pattern bytes of one quartet; the pattern's byte
            // moves to the top of the lane and an arithmetic shift sign-extends it.
            let words = _mm512_cvtepu32_epi64(_mm256_loadu_si256(p.cast()));
            std::array::from_fn(|a| {
                let up = _mm512_sllv_epi64(words, _mm512_set1_epi64(56 - 8 * a as i64));
                _mm512_srai_epi64::<56>(up)
            })
        }
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn lanes(x: __m512i) -> [u64; 8] {
        let mut out = [0_u64; 8];
        // SAFETY: `out` holds eight words.
        unsafe { _mm512_storeu_si512(out.as_mut_ptr().cast(), x) };
        out
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn any_above(x: __m512i, bound: u64) -> bool {
        _mm512_cmpgt_epu64_mask(x, _mm512_set1_epi64(bound as i64)) != 0
    }
}

/// The sign bit: AVX2 compares 64-bit lanes as signed only, so unsigned order is the signed
/// order of the lanes with this bit flipped.
const SIGN: i64 = i64::MIN;

impl W4 {
    /// `b` in the lanes where bit 63 of `m` is set, else `a` (the blend reads the sign bit
    /// only, so `m` needs no full-lane mask).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn select(a: __m256i, b: __m256i, m: __m256i) -> __m256i {
        _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(a), _mm256_castsi256_pd(b), _mm256_castsi256_pd(m)))
    }
    /// All-ones lanes where `a < b` as unsigned integers.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn ltu(a: __m256i, b: __m256i) -> __m256i {
        let s = _mm256_set1_epi64x(SIGN);
        _mm256_cmpgt_epi64(_mm256_xor_si256(b, s), _mm256_xor_si256(a, s))
    }
}

impl Simd for W4 {
    type V = __m256i;
    const LANES: usize = 4;
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn splat(x: u64) -> __m256i {
        _mm256_set1_epi64x(x as i64)
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn load(p: *const u64) -> __m256i {
        // SAFETY: the caller guarantees four readable words at `p`.
        unsafe { _mm256_loadu_si256(p.cast()) }
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn store(p: *mut u64, x: __m256i) {
        // SAFETY: the caller guarantees four writable words at `p`.
        unsafe { _mm256_storeu_si256(p.cast(), x) }
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn add(a: __m256i, b: __m256i) -> __m256i {
        _mm256_add_epi64(a, b)
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn sub(a: __m256i, b: __m256i) -> __m256i {
        _mm256_sub_epi64(a, b)
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn and(a: __m256i, b: __m256i) -> __m256i {
        _mm256_and_si256(a, b)
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn shr32(a: __m256i) -> __m256i {
        _mm256_srli_epi64::<32>(a)
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn shl32(a: __m256i) -> __m256i {
        _mm256_slli_epi64::<32>(a)
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn shl1(a: __m256i) -> __m256i {
        _mm256_slli_epi64::<1>(a)
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn shl2(a: __m256i) -> __m256i {
        _mm256_slli_epi64::<2>(a)
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn shl6(a: __m256i) -> __m256i {
        _mm256_slli_epi64::<6>(a)
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn mul32(a: __m256i, b: __m256i) -> __m256i {
        _mm256_mul_epu32(a, b)
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn mul32s(a: __m256i, b: __m256i) -> __m256i {
        _mm256_mul_epi32(a, b)
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn canon(x: __m256i) -> __m256i {
        // x >= p exactly when t = x + 59 carries out of 64 bits, and then t = x - p. Adding 59
        // carries only from x >= 2^63 to t < 59, so bit 63 of x AND NOT t is the carry.
        let t = _mm256_add_epi64(x, _mm256_set1_epi64x(59));
        // SAFETY: AVX2 is enabled.
        unsafe { Self::select(x, t, _mm256_andnot_si256(t, x)) }
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn add_mod(a: __m256i, b: __m256i) -> __m256i {
        // a + b - p = a - (p - b): exact when a >= p - b, else add p back. With the sign
        // bits flipped (x ^ 2^63 = x + 2^63 mod 2^64) the unsigned order is the signed one:
        // (p - b) ^ 2^63 = (p ^ 2^63) - b, and the difference of two flipped values is
        // the difference of the values.
        let a_flipped = _mm256_xor_si256(a, _mm256_set1_epi64x(SIGN));
        let nb_flipped = _mm256_sub_epi64(_mm256_set1_epi64x((crate::P as i64) ^ SIGN), b);
        let t = _mm256_sub_epi64(a_flipped, nb_flipped);
        let borrow = _mm256_cmpgt_epi64(nb_flipped, a_flipped);
        _mm256_add_epi64(t, _mm256_and_si256(borrow, _mm256_set1_epi64x(crate::P as i64)))
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn sub_mod(a: __m256i, b: __m256i) -> __m256i {
        // SAFETY: AVX2 is enabled.
        unsafe {
            let s = _mm256_sub_epi64(a, b);
            // A borrow wrapped by 2^64; subtracting 59 more leaves a - b + p.
            _mm256_sub_epi64(s, _mm256_and_si256(Self::ltu(a, b), _mm256_set1_epi64x(59)))
        }
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn add_fold(a: __m256i, b: __m256i) -> __m256i {
        // SAFETY: AVX2 is enabled.
        unsafe {
            let s = _mm256_add_epi64(a, b);
            _mm256_add_epi64(s, _mm256_and_si256(Self::ltu(s, a), _mm256_set1_epi64x(59)))
        }
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn add_fold_small(a: __m256i, small: __m256i) -> __m256i {
        // With small < 2^63 the sum carries only from a >= 2^63, to a sum below 2^63: bit 63
        // of a AND NOT s is the carry.
        let s = _mm256_add_epi64(a, small);
        let t = _mm256_add_epi64(s, _mm256_set1_epi64x(59));
        // SAFETY: AVX2 is enabled.
        unsafe { Self::select(s, t, _mm256_andnot_si256(s, a)) }
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn residue(k: __m256i) -> __m256i {
        let negative = _mm256_cmpgt_epi64(_mm256_setzero_si256(), k);
        _mm256_add_epi64(k, _mm256_and_si256(negative, _mm256_set1_epi64x(crate::P as i64)))
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn deinterleave(x0: __m256i, x1: __m256i) -> (__m256i, __m256i) {
        // [r0 r4 r2 r6] and [r1 r5 r3 r7], then the middle 64-bit lanes swapped.
        let lo = _mm256_unpacklo_epi64(x0, x1);
        let hi = _mm256_unpackhi_epi64(x0, x1);
        (_mm256_permute4x64_epi64::<0b11_01_10_00>(lo), _mm256_permute4x64_epi64::<0b11_01_10_00>(hi))
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn deinterleave_any(x0: __m256i, x1: __m256i) -> (__m256i, __m256i) {
        // Items in the order 0, 2, 1, 3: [r0 r4 r2 r6] and [r1 r5 r3 r7], no permute.
        (_mm256_unpacklo_epi64(x0, x1), _mm256_unpackhi_epi64(x0, x1))
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn reorder(x: __m256i) -> __m256i {
        _mm256_permute4x64_epi64::<0b11_01_10_00>(x)
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn load_i32(p: *const i32) -> __m256i {
        // SAFETY: the caller guarantees four readable values at `p`.
        unsafe { _mm256_cvtepi32_epi64(_mm_loadu_si128(p.cast())) }
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn strided4_i32(p: *const i32) -> [__m256i; 4] {
        // SAFETY: the caller guarantees 16 readable values at `p`.
        unsafe {
            // A 4 x 4 transpose of the quartets (rows) into patterns (columns).
            let r0 = _mm_loadu_si128(p.cast());
            let r1 = _mm_loadu_si128(p.add(4).cast());
            let r2 = _mm_loadu_si128(p.add(8).cast());
            let r3 = _mm_loadu_si128(p.add(12).cast());
            let t0 = _mm_unpacklo_epi32(r0, r1);
            let t1 = _mm_unpacklo_epi32(r2, r3);
            let t2 = _mm_unpackhi_epi32(r0, r1);
            let t3 = _mm_unpackhi_epi32(r2, r3);
            [
                _mm256_cvtepi32_epi64(_mm_unpacklo_epi64(t0, t1)),
                _mm256_cvtepi32_epi64(_mm_unpackhi_epi64(t0, t1)),
                _mm256_cvtepi32_epi64(_mm_unpacklo_epi64(t2, t3)),
                _mm256_cvtepi32_epi64(_mm_unpackhi_epi64(t2, t3)),
            ]
        }
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn strided4_i8(p: *const i8) -> [__m256i; 4] {
        // SAFETY: the caller guarantees 16 readable bytes at `p`.
        unsafe {
            // Pattern-major byte order, then four bytes per pattern sign-extended.
            let x = _mm_loadu_si128(p.cast());
            let t = _mm_shuffle_epi8(x, _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15));
            [
                _mm256_cvtepi8_epi64(t),
                _mm256_cvtepi8_epi64(_mm_srli_si128::<4>(t)),
                _mm256_cvtepi8_epi64(_mm_srli_si128::<8>(t)),
                _mm256_cvtepi8_epi64(_mm_srli_si128::<12>(t)),
            ]
        }
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn lanes(x: __m256i) -> [u64; 8] {
        let mut out = [0_u64; 8];
        // SAFETY: `out` holds eight words, four are written.
        unsafe { _mm256_storeu_si256(out.as_mut_ptr().cast(), x) };
        out
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn any_above(x: __m256i, bound: u64) -> bool {
        // SAFETY: AVX2 is enabled.
        unsafe { _mm256_movemask_pd(_mm256_castsi256_pd(Self::ltu(_mm256_set1_epi64x(bound as i64), x))) != 0 }
    }
}

/// A lane value and its high half: `l` holds the value (the multiplier reads its low 32 bits),
/// `h = value >> 32`.
#[derive(Clone, Copy)]
pub(crate) struct S<V> {
    pub l: V,
    pub h: V,
}

/// Lazy sum of 32 x 32-bit partial products, by weight class. A class whose terms are below
/// 2^64 is kept as two sums: `w*`, the terms added in full and wrapping mod 2^64, and `h*`,
/// the exact sum of their high halves. The exact sum of the low halves is then
/// `w - 2^32 h mod 2^64` (it is below 2^64), so a term costs one add to `w` and a shift and
/// an add to `h`, where splitting it costs two masks and two adds.
///
/// The classes: `w0` (weight 2^0) with `h1`; `w1` (weight 2^32) with `h2`; `w2` (weight
/// 2^64) with `h3`. The value is `L0 + 2^32 (L1 + h1) + 2^64 (L2 + h2) + 2^96 h3` with
/// `Lk = wk - 2^32 h(k + 1) mod 2^64` ([`Acc::limbs`]). One [`mac`] of two values below 2^64
/// adds below `2^32` to `L0, h1, L2, h3` and below `2 * 2^32` to `L1, h2`; [`red`] needs the
/// limbs `L0 < 2^62`, `L1 + h1 < 2^62`, `L2 + h2 < 2^56`, `h3 < 2^56`, so an accumulator
/// takes up to [`MAC_LIMIT`] products between reductions.
#[derive(Clone, Copy)]
pub(crate) struct Acc<V> {
    pub w0: V,
    pub h1: V,
    pub w1: V,
    pub h2: V,
    pub w2: V,
    pub h3: V,
}

/// Products an accumulator may take before [`red`]: `3 * 2^32 * 2^22 < 2^56`.
pub(crate) const MAC_LIMIT: usize = 1 << 22;

/// Unsigned limbs `d0 + 2^32 d1 + 2^64 d2 + 2^96 d3` (the input of [`red4`]).
#[derive(Clone, Copy)]
pub(crate) struct Limbs<V> {
    pub d0: V,
    pub d1: V,
    pub d2: V,
    pub d3: V,
}

impl<V: Copy> Acc<V> {
    /// The limbs of the accumulated value (exact: each low-half sum is below 2^64).
    #[inline(always)]
    unsafe fn limbs<B: Simd<V = V>>(self) -> Limbs<V> {
        // SAFETY: the caller guarantees the backend's features.
        unsafe {
            Limbs {
                d0: B::sub(self.w0, B::shl32(self.h1)),
                d1: B::add(B::sub(self.w1, B::shl32(self.h2)), self.h1),
                d2: B::add(B::sub(self.w2, B::shl32(self.h3)), self.h2),
                d3: self.h3,
            }
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn split<B: Simd>(x: B::V) -> S<B::V> {
    // SAFETY: the caller guarantees the backend's features.
    unsafe { S { l: x, h: B::shr32(x) } }
}

#[inline(always)]
pub(crate) unsafe fn zero<B: Simd>() -> B::V {
    // SAFETY: the caller guarantees the backend's features.
    unsafe { B::splat(0) }
}

#[inline(always)]
pub(crate) unsafe fn acc0<B: Simd>() -> Acc<B::V> {
    // SAFETY: the caller guarantees the backend's features.
    unsafe {
        let z = zero::<B>();
        Acc { w0: z, h1: z, w1: z, h2: z, w2: z, h3: z }
    }
}

/// `acc += a b` for values below 2^64 (the limb gains of [`Acc`]).
#[inline(always)]
pub(crate) unsafe fn mac<B: Simd>(acc: &mut Acc<B::V>, a: S<B::V>, b: S<B::V>) {
    // SAFETY: the caller guarantees the backend's features.
    unsafe {
        let ll = B::mul32(a.l, b.l);
        let lh = B::mul32(a.l, b.h);
        let hl = B::mul32(a.h, b.l);
        let hh = B::mul32(a.h, b.h);
        acc.w0 = B::add(acc.w0, ll);
        acc.h1 = B::add(acc.h1, B::shr32(ll));
        acc.w1 = B::add(acc.w1, B::add(lh, hl));
        acc.h2 = B::add(acc.h2, B::add(B::shr32(lh), B::shr32(hl)));
        acc.w2 = B::add(acc.w2, hh);
        acc.h3 = B::add(acc.h3, B::shr32(hh));
    }
}

/// `acc += w k` for a value `w < 2^64` and `k < 2^32` (two partials): `L0`, `h1`, `L1`, `h2`
/// gain below 2^32, within the gains of one [`mac`].
#[inline(always)]
pub(crate) unsafe fn mac32<B: Simd>(acc: &mut Acc<B::V>, w: S<B::V>, k: B::V) {
    // SAFETY: the caller guarantees the backend's features.
    unsafe {
        let lo = B::mul32(w.l, k);
        let hi = B::mul32(w.h, k);
        acc.w0 = B::add(acc.w0, lo);
        acc.h1 = B::add(acc.h1, B::shr32(lo));
        acc.w1 = B::add(acc.w1, hi);
        acc.h2 = B::add(acc.h2, B::shr32(hi));
    }
}

/// Shift of a signed value in `[-2^31, 2^31)` into the multiplier range `[0, 2^32)` of
/// [`mac32`]; the caller subtracts `2^31` times the weight sum once.
pub(crate) const I32_SHIFT: u64 = 1 << 31;

/// `acc += x` for a value below 2^64 (`L0` and `h1` gain below 2^32).
#[inline(always)]
pub(crate) unsafe fn lift<B: Simd>(acc: &mut Acc<B::V>, x: B::V) {
    // SAFETY: the caller guarantees the backend's features.
    unsafe {
        acc.w0 = B::add(acc.w0, x);
        acc.h1 = B::add(acc.h1, B::shr32(x));
    }
}

/// Component-wise sum (the limb bounds add; the wrapping sums stay exact mod 2^64).
#[inline(always)]
pub(crate) unsafe fn add_acc<B: Simd>(a: Acc<B::V>, b: Acc<B::V>) -> Acc<B::V> {
    // SAFETY: the caller guarantees the backend's features.
    unsafe {
        Acc {
            w0: B::add(a.w0, b.w0),
            h1: B::add(a.h1, b.h1),
            w1: B::add(a.w1, b.w1),
            h2: B::add(a.h2, b.h2),
            w2: B::add(a.w2, b.w2),
            h3: B::add(a.h3, b.h3),
        }
    }
}

/// `2 a` component-wise (the limb bounds double).
#[inline(always)]
pub(crate) unsafe fn dbl_acc<B: Simd>(a: Acc<B::V>) -> Acc<B::V> {
    // SAFETY: the caller guarantees the backend's features.
    unsafe {
        Acc {
            w0: B::shl1(a.w0),
            h1: B::shl1(a.h1),
            w1: B::shl1(a.w1),
            h2: B::shl1(a.h2),
            w2: B::shl1(a.w2),
            h3: B::shl1(a.h3),
        }
    }
}

/// `59 x` for `x < 2^58`, as `64 x - 4 x - x`.
#[inline(always)]
unsafe fn mul59<B: Simd>(x: B::V) -> B::V {
    // SAFETY: the caller guarantees the backend's features.
    unsafe { B::sub(B::sub(B::shl6(x), B::shl2(x)), x) }
}

/// A value below 2^64 congruent to limbs with `d0, d1 < 2^62` and `d2, d3 < 2^56` (not
/// necessarily canonical: a multiplier operand needs no more).
#[inline(always)]
pub(crate) unsafe fn red4_lazy<B: Simd>(acc: Limbs<B::V>) -> B::V {
    // SAFETY: the caller guarantees the backend's features.
    unsafe {
        // 2^64 = 59 and 2^96 = 59 * 2^32: value = e0 + 2^32 e1 with e0, e1 < 2^63.
        let e0 = B::add(acc.d0, mul59::<B>(acc.d2));
        let e1 = B::add(acc.d1, mul59::<B>(acc.d3));
        // Carry the high half of e0: e1 < 2^63 + 2^31, e0 < 2^32.
        let e1 = B::add(e1, B::shr32(e0));
        let e0 = B::and(e0, B::splat(LOW32));
        // value = e0 + 2^32 (e1 mod 2^32) + 2^64 (e1 >> 32), and e1 >> 32 < 2^31 + 1, so its
        // fold is one 32-bit product: g < 2^32 + 59 * 2^32 < 2^38.
        let g = B::add(e0, B::mul32(B::shr32(e1), B::splat(59)));
        // A carry out of 2^32 (e1 mod 2^32) + g wraps to below g < 2^38; fold it as 59.
        B::add_fold_small(B::shl32(e1), g)
    }
}

/// The canonical value of limbs with `d0, d1 < 2^62` and `d2, d3 < 2^56`.
#[inline(always)]
pub(crate) unsafe fn red4<B: Simd>(acc: Limbs<B::V>) -> B::V {
    // SAFETY: the caller guarantees the backend's features.
    unsafe { B::canon(red4_lazy::<B>(acc)) }
}

/// The canonical value of an accumulator within the bounds of [`Acc`].
#[inline(always)]
pub(crate) unsafe fn red<B: Simd>(acc: Acc<B::V>) -> B::V {
    // SAFETY: the caller guarantees the backend's features.
    unsafe { red4::<B>(acc.limbs::<B>()) }
}

/// [`red`] without the final canonical step: a value below 2^64 for a multiplier operand.
#[inline(always)]
pub(crate) unsafe fn red_lazy<B: Simd>(acc: Acc<B::V>) -> B::V {
    // SAFETY: the caller guarantees the backend's features.
    unsafe { red4_lazy::<B>(acc.limbs::<B>()) }
}

/// The limbs of one product `a b` of values below 2^64: `d0, d3 < 2^32`, `d1, d2 < 3 * 2^32`.
/// A single product splits its partials directly (no wrapping sums to undo).
#[inline(always)]
unsafe fn product<B: Simd>(a: S<B::V>, b: S<B::V>) -> Limbs<B::V> {
    // SAFETY: the caller guarantees the backend's features.
    unsafe {
        let m = B::splat(LOW32);
        let ll = B::mul32(a.l, b.l);
        let lh = B::mul32(a.l, b.h);
        let hl = B::mul32(a.h, b.l);
        let hh = B::mul32(a.h, b.h);
        Limbs {
            d0: B::and(ll, m),
            d1: B::add(B::shr32(ll), B::add(B::and(lh, m), B::and(hl, m))),
            d2: B::add(B::add(B::shr32(lh), B::shr32(hl)), B::and(hh, m)),
            d3: B::shr32(hh),
        }
    }
}

/// The limbs of `a b + c d` for values below 2^64 (the bounds of two [`product`]s add).
#[inline(always)]
unsafe fn product2<B: Simd>(a: S<B::V>, b: S<B::V>, c: S<B::V>, d: S<B::V>) -> Limbs<B::V> {
    // SAFETY: the caller guarantees the backend's features.
    unsafe {
        let x = product::<B>(a, b);
        let y = product::<B>(c, d);
        Limbs { d0: B::add(x.d0, y.d0), d1: B::add(x.d1, y.d1), d2: B::add(x.d2, y.d2), d3: B::add(x.d3, y.d3) }
    }
}

/// `x` added to limbs: `d0` and `d1` gain below 2^32.
#[inline(always)]
unsafe fn lift_limbs<B: Simd>(l: Limbs<B::V>, x: B::V) -> Limbs<B::V> {
    // SAFETY: the caller guarantees the backend's features.
    unsafe { Limbs { d0: B::add(l.d0, B::and(x, B::splat(LOW32))), d1: B::add(l.d1, B::shr32(x)), ..l } }
}

/// `e + r (o - e)` lane-wise in `K` for canonical lanes (canonical out): the canonical
/// difference, two products per coordinate with `e` added to their limbs (`d0, d3 < 3 *
/// 2^32`, `d1, d2 < 7 * 2^32`), one reduction per coordinate.
#[inline(always)]
pub(crate) unsafe fn kfold<B: Simd>(e: (B::V, B::V), o: (B::V, B::V), r: KS<B::V>) -> (B::V, B::V) {
    // SAFETY: the caller guarantees the backend's features.
    unsafe {
        let d = (B::sub_mod(o.0, e.0), B::sub_mod(o.1, e.1));
        let (x0, x1) = (split::<B>(d.0), split::<B>(d.1));
        (
            red4::<B>(lift_limbs::<B>(product2::<B>(x0, r.c0, x1, r.c1x2), e.0)),
            red4::<B>(lift_limbs::<B>(product2::<B>(x0, r.c1, x1, r.c0), e.1)),
        )
    }
}

/// `l + d r` for canonical lanes `l`, `d` and a split `r` (canonical out): one product with
/// `l` added to its limbs, one reduction.
#[inline(always)]
pub(crate) unsafe fn mul_add<B: Simd>(l: B::V, d: B::V, r: S<B::V>) -> B::V {
    // SAFETY: the caller guarantees the backend's features.
    unsafe { red4::<B>(lift_limbs::<B>(product::<B>(split::<B>(d), r), l)) }
}

/// `a b` for canonical lanes.
#[inline(always)]
pub(crate) unsafe fn mul<B: Simd>(a: B::V, b: B::V) -> B::V {
    // SAFETY: the caller guarantees the backend's features.
    unsafe { red4::<B>(product::<B>(split::<B>(a), split::<B>(b))) }
}

/// `a b` for a canonical lane and a split operand.
#[inline(always)]
pub(crate) unsafe fn mul_s<B: Simd>(a: B::V, b: S<B::V>) -> B::V {
    // SAFETY: the caller guarantees the backend's features.
    unsafe { red4::<B>(product::<B>(split::<B>(a), b)) }
}

/// A `K` operand split for repeated products: both coordinates and twice the second (the
/// real part is `ac + 2 bd`).
#[derive(Clone, Copy)]
pub(crate) struct KS<V> {
    pub c0: S<V>,
    pub c1: S<V>,
    pub c1x2: S<V>,
}

#[inline(always)]
pub(crate) unsafe fn split_k<B: Simd>(x: (B::V, B::V)) -> KS<B::V> {
    // SAFETY: the caller guarantees the backend's features.
    unsafe { KS { c0: split::<B>(x.0), c1: split::<B>(x.1), c1x2: split::<B>(B::add_mod(x.1, x.1)) } }
}

/// `acc += x y` in `K` without reduction: `re += ac + 2bd`, `im += ad + bc` (two products per
/// coordinate).
#[inline(always)]
pub(crate) unsafe fn kmac<B: Simd>(acc: &mut (Acc<B::V>, Acc<B::V>), x: (S<B::V>, S<B::V>), y: KS<B::V>) {
    // SAFETY: the caller guarantees the backend's features.
    unsafe {
        mac::<B>(&mut acc.0, x.0, y.c0);
        mac::<B>(&mut acc.0, x.1, y.c1x2);
        mac::<B>(&mut acc.1, x.0, y.c1);
        mac::<B>(&mut acc.1, x.1, y.c0);
    }
}

/// `acc += x y` in `K` for a base-field `y` (one product per coordinate).
#[inline(always)]
pub(crate) unsafe fn kmac_base<B: Simd>(acc: &mut (Acc<B::V>, Acc<B::V>), x: (S<B::V>, S<B::V>), y: S<B::V>) {
    // SAFETY: the caller guarantees the backend's features.
    unsafe {
        mac::<B>(&mut acc.0, x.0, y);
        mac::<B>(&mut acc.1, x.1, y);
    }
}

#[inline(always)]
pub(crate) unsafe fn acck<B: Simd>() -> (Acc<B::V>, Acc<B::V>) {
    // SAFETY: the caller guarantees the backend's features.
    unsafe { (acc0::<B>(), acc0::<B>()) }
}

#[inline(always)]
pub(crate) unsafe fn redk<B: Simd>(acc: (Acc<B::V>, Acc<B::V>)) -> (B::V, B::V) {
    // SAFETY: the caller guarantees the backend's features.
    unsafe { (red::<B>(acc.0), red::<B>(acc.1)) }
}

/// `a b` lane-wise in `K` with a pre-split second operand (canonical out): each coordinate
/// is two products split directly and one reduction.
#[inline(always)]
pub(crate) unsafe fn kmul_s<B: Simd>(a: (B::V, B::V), b: KS<B::V>) -> (B::V, B::V) {
    // SAFETY: the caller guarantees the backend's features.
    unsafe {
        let (x0, x1) = (split::<B>(a.0), split::<B>(a.1));
        (red4::<B>(product2::<B>(x0, b.c0, x1, b.c1x2)), red4::<B>(product2::<B>(x0, b.c1, x1, b.c0)))
    }
}

/// `a b` lane-wise in `K` (canonical in, canonical out).
#[inline(always)]
pub(crate) unsafe fn kmul<B: Simd>(a: (B::V, B::V), b: (B::V, B::V)) -> (B::V, B::V) {
    // SAFETY: the caller guarantees the backend's features.
    unsafe { kmul_s::<B>(a, split_k::<B>(b)) }
}

#[inline(always)]
pub(crate) unsafe fn addk<B: Simd>(a: (B::V, B::V), b: (B::V, B::V)) -> (B::V, B::V) {
    // SAFETY: the caller guarantees the backend's features.
    unsafe { (B::add_mod(a.0, b.0), B::add_mod(a.1, b.1)) }
}

#[inline(always)]
pub(crate) unsafe fn subk<B: Simd>(a: (B::V, B::V), b: (B::V, B::V)) -> (B::V, B::V) {
    // SAFETY: the caller guarantees the backend's features.
    unsafe { (B::sub_mod(a.0, b.0), B::sub_mod(a.1, b.1)) }
}

/// A `K` element broadcast to every lane.
#[inline(always)]
pub(crate) unsafe fn bck<B: Simd>(x: crate::K) -> (B::V, B::V) {
    // SAFETY: the caller guarantees the backend's features.
    unsafe { (B::splat(x.c0().to_canonical_u64()), B::splat(x.c1().to_canonical_u64())) }
}

#[inline(always)]
pub(crate) unsafe fn loadk<B: Simd>(planes: (&[u64], &[u64]), i: usize) -> (B::V, B::V) {
    // SAFETY: the caller guarantees the backend's features and `LANES` rows from `i`.
    unsafe { (B::load(planes.0.as_ptr().add(i)), B::load(planes.1.as_ptr().add(i))) }
}

/// The lanes of a canonical vector summed mod p.
#[inline(always)]
pub(crate) unsafe fn hsum<B: Simd>(x: B::V) -> u64 {
    // SAFETY: the caller guarantees the backend's features.
    let lanes = unsafe { B::lanes(x) };
    lanes[..B::LANES].iter().fold(0_u64, |s, &x| {
        let (r, c) = s.overflowing_add(x);
        let r = r.wrapping_add(u64::from(c) * 59);
        r.min(r.wrapping_sub(crate::P))
    })
}

#[inline(always)]
pub(crate) unsafe fn hsum_k<B: Simd>(x: (B::V, B::V)) -> crate::K {
    use jolt_field::Ring;
    // SAFETY: the caller guarantees the backend's features.
    unsafe { crate::K::new(crate::F::from_u64(hsum::<B>(x.0)), crate::F::from_u64(hsum::<B>(x.1))) }
}

/// Instantiates a generic kernel body per backend: `$avx512` with AVX-512F enabled over
/// [`W8`] and `$avx2` with AVX2 enabled over [`W4`].
macro_rules! entry_points {
    ($body:ident => $avx512:ident, $avx2:ident; ($($arg:ident : $ty:ty),*) $(-> $ret:ty)?) => {
        #[target_feature(enable = "avx512f")]
        pub(crate) unsafe fn $avx512($($arg: $ty),*) $(-> $ret)? {
            // SAFETY: the caller guarantees AVX-512F and the body's slice preconditions.
            unsafe { $body::<$crate::wide::W8>($($arg),*) }
        }
        #[target_feature(enable = "avx2")]
        pub(crate) unsafe fn $avx2($($arg: $ty),*) $(-> $ret)? {
            // SAFETY: the caller guarantees AVX2 and the body's slice preconditions.
            unsafe { $body::<$crate::wide::W4>($($arg),*) }
        }
    };
}
pub(crate) use entry_points;

/// Selects the entry point of the active backend without IFMA, or returns `None`.
macro_rules! dispatch {
    ($avx512:expr, $avx2:expr) => {
        match $crate::detect::backend() {
            // SAFETY: the backend check guarantees the entry point's CPU features.
            $crate::detect::Backend::Avx512F => Some(unsafe { $avx512 }),
            // SAFETY: the backend check guarantees the entry point's CPU features.
            $crate::detect::Backend::Avx2 => Some(unsafe { $avx2 }),
            _ => None,
        }
    };
}
pub(crate) use dispatch;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{scalar, F, K, P};
    use jolt_field::Ring;

    fn random(s: &mut u64) -> u64 {
        *s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Canonical values with the edge cases of the 32-bit halves, of p and of the sign bit (the
    /// AVX2 canonical step and add mod p decide by bit 63).
    fn values(n: usize, seed: &mut u64) -> Vec<u64> {
        const EDGES: [u64; 15] = [
            0,
            1,
            58,
            59,
            0xffff_ffff,
            1 << 32,
            (1 << 32) + 1,
            P - 1,
            P - 2,
            P - (1 << 32),
            (1 << 63) - 59,
            (1 << 63) - 1,
            1 << 63,
            (1 << 63) + 1,
            (1 << 63) + 58,
        ];
        (0..n).map(|i| if i % 3 == 0 { EDGES[(i / 3) % EDGES.len()] } else { random(seed) % P }).collect()
    }

    /// The exact value of an accumulator mod p.
    fn exact(d: [u64; 4]) -> u64 {
        let x = |v: u64| F::from_u64(v % P);
        let two32 = F::from_u64(1 << 32);
        let two64 = F::from_u64(59);
        (x(d[0]) + two32 * x(d[1]) + two64 * x(d[2]) + two64 * two32 * x(d[3])).to_canonical_u64()
    }

    unsafe fn check_backend<B: Simd>() {
        let mut seed = 0x5eed;
        let l = B::LANES;
        // The reduction at and below its limb bounds.
        for round in 0..200 {
            let limb = |s: &mut u64, bits: u32| if round % 2 == 0 { (1 << bits) - 1 - (random(s) % 3) } else { random(s) >> (64 - bits) };
            let lanes: Vec<[u64; 4]> = (0..l).map(|_| [limb(&mut seed, 62), limb(&mut seed, 62), limb(&mut seed, 56), limb(&mut seed, 56)]).collect();
            let get = |k: usize| unsafe { B::load(lanes.iter().map(|d| d[k]).collect::<Vec<_>>().as_ptr()) };
            let limbs = Limbs { d0: get(0), d1: get(1), d2: get(2), d3: get(3) };
            let got = unsafe { B::lanes(red4::<B>(limbs)) };
            let lazy = unsafe { B::lanes(red4_lazy::<B>(limbs)) };
            for (j, d) in lanes.iter().enumerate() {
                assert_eq!(got[j], exact(*d), "red lane {j} limbs {d:?}");
                assert_eq!(lazy[j] % P, exact(*d), "lazy red lane {j} limbs {d:?}");
            }
        }
        // Products, K products, canonical sums and differences, residues.
        let n = 4096;
        let (a, b, c, d) = (values(n, &mut seed), values(n, &mut seed), values(n, &mut seed), values(n, &mut seed));
        let ints: Vec<i64> = (0..n).map(|i| [i64::MIN, i64::MAX, -1, 0, 1, -(P as i64 >> 1)][i % 6].wrapping_add((random(&mut seed) % 3) as i64)).collect();
        for i in (0..n).step_by(l) {
            let (x, y) = unsafe { (B::load(a.as_ptr().add(i)), B::load(b.as_ptr().add(i))) };
            let (z, w) = unsafe { (B::load(c.as_ptr().add(i)), B::load(d.as_ptr().add(i))) };
            let k = unsafe { B::load(ints.as_ptr().add(i).cast()) };
            let prod = unsafe { B::lanes(mul::<B>(x, y)) };
            let sum = unsafe { B::lanes(B::add_mod(x, y)) };
            let diff = unsafe { B::lanes(B::sub_mod(x, y)) };
            let res = unsafe { B::lanes(B::residue(k)) };
            let (re, im) = unsafe { kmul::<B>((x, y), (z, w)) };
            let (re, im) = unsafe { (B::lanes(re), B::lanes(im)) };
            for j in 0..l {
                let (fa, fb) = (F::from_u64(a[i + j]), F::from_u64(b[i + j]));
                assert_eq!(prod[j], (fa * fb).to_canonical_u64(), "mul {} {}", a[i + j], b[i + j]);
                assert_eq!(sum[j], (fa + fb).to_canonical_u64(), "add_mod");
                assert_eq!(diff[j], (fa - fb).to_canonical_u64(), "sub_mod");
                assert_eq!(res[j], K::from_i64(ints[i + j]).c0().to_canonical_u64(), "residue {}", ints[i + j]);
                let e = scalar::get((&a, &b), i + j) * scalar::get((&c, &d), i + j);
                assert_eq!((re[j], im[j]), (e.c0().to_canonical_u64(), e.c1().to_canonical_u64()), "kmul");
            }
        }
        // `canon` on every residue class boundary of a raw u64.
        let raw = [0, 1, P - 1, P, P + 1, u64::MAX - 1, u64::MAX, 1 << 63];
        for chunk in raw.chunks(l) {
            let mut lanes = [0_u64; 8];
            lanes[..chunk.len()].copy_from_slice(chunk);
            let got = unsafe { B::lanes(B::canon(B::load(lanes.as_ptr()))) };
            for j in 0..chunk.len() {
                assert_eq!(got[j], chunk[j] % P, "canon {}", chunk[j]);
            }
        }
        // Strided cell loads against the scalar index map.
        let cells: Vec<i32> = (0..4 * l as i32).map(|i| i * 0x0101_0101 - (1 << 30) * (i % 3)).collect();
        let bytes: Vec<i8> = (0..4 * l).map(|i| (i as i32 * 37 - 128) as i8).collect();
        let (wide, narrow) = unsafe { (B::strided4_i32(cells.as_ptr()), B::strided4_i8(bytes.as_ptr())) };
        for a in 0..4 {
            let (wide, narrow) = unsafe { (B::lanes(wide[a]), B::lanes(narrow[a])) };
            for j in 0..l {
                assert_eq!(wide[j] as i64, i64::from(cells[4 * j + a]), "strided i32 pattern {a} lane {j}");
                assert_eq!(narrow[j] as i64, i64::from(bytes[4 * j + a]), "strided i8 pattern {a} lane {j}");
            }
        }
        let rows: Vec<u64> = (0..2 * l as u64).collect();
        let (even, odd) = unsafe { B::deinterleave(B::load(rows.as_ptr()), B::load(rows.as_ptr().add(l))) };
        let (even, odd) = unsafe { (B::lanes(even), B::lanes(odd)) };
        for j in 0..l {
            assert_eq!((even[j], odd[j]), (2 * j as u64, 2 * j as u64 + 1), "deinterleave");
        }
        // The unordered split puts item k where `reorder` puts per-item value k.
        let items: Vec<u64> = (0..l as u64).collect();
        let (even, odd) = unsafe { B::deinterleave_any(B::load(rows.as_ptr()), B::load(rows.as_ptr().add(l))) };
        let (even, odd, order) = unsafe { (B::lanes(even), B::lanes(odd), B::lanes(B::reorder(B::load(items.as_ptr())))) };
        let back = unsafe { B::lanes(B::reorder(B::reorder(B::load(items.as_ptr())))) };
        for j in 0..l {
            assert_eq!((even[j], odd[j]), (2 * order[j], 2 * order[j] + 1), "deinterleave_any");
            assert_eq!(back[j], j as u64, "reorder is an involution");
        }
    }

    #[test]
    fn avx512f_lanes_match_field_arithmetic() {
        if crate::detect::avx512f_available() {
            // SAFETY: AVX-512F was detected.
            unsafe { check_backend::<W8>() }
        }
    }

    #[test]
    fn avx2_lanes_match_field_arithmetic() {
        if crate::detect::avx2_available() {
            // SAFETY: AVX2 was detected.
            unsafe { check_backend::<W4>() }
        }
    }

    /// A dot over more products than one accumulator takes ([`MAC_LIMIT`] per lane in the
    /// `bd`-doubled real part) at the worst-case operand `p - 1`.
    #[test]
    fn long_dots_cross_the_reduction_chunks() {
        let n = crate::dot::wide_chunk() * 4 + 4 * 8 + 3;
        let a = vec![P - 1; n];
        let b: Vec<u64> = (0..n as u64).map(|i| if i % 2 == 0 { P - 1 } else { i }).collect();
        if crate::detect::avx2_available() {
            // SAFETY: AVX2 was detected.
            unsafe {
                assert_eq!(crate::dot::fp_w4(&a, &b), scalar::fp_dot(&a, &b), "fp_dot");
                assert_eq!(crate::dot::k_w4((&a, &b), (&b, &a)), scalar::k_dot((&a, &b), (&b, &a)), "k_dot");
            }
        }
        if crate::detect::avx512f_available() {
            // SAFETY: AVX-512F was detected.
            unsafe {
                assert_eq!(crate::dot::fp_w8(&a, &b), scalar::fp_dot(&a, &b), "fp_dot");
                assert_eq!(crate::dot::k_w8((&a, &b), (&b, &a)), scalar::k_dot((&a, &b), (&b, &a)), "k_dot");
            }
        }
    }
}
