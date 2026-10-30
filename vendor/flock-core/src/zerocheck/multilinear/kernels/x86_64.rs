use core::{
    arch::x86_64::{
        __m128i, __m256i, __m512i, _mm_load_si128, _mm_set_epi64x, _mm_setzero_si128,
        _mm_xor_si128, _mm256_stream_si256, _mm512_broadcast_i32x4, _mm512_castsi512_si256,
        _mm512_extracti64x4_epi64, _mm512_loadu_si512, _mm512_permutex2var_epi64, _mm512_set_epi64,
        _mm512_storeu_si512, _mm512_stream_si512, _mm512_xor_si512,
    },
    mem::transmute,
};

// agg-mldsa patch (PATCH.md): the GHASH primitives come from
// `crate::simd::x86` (no `cfg(target_feature)` gate) and the AVX-512 kernel
// carries its own `#[target_feature]`, so the caller can choose it at run time.
use crate::{
    field::{F128, F256Unreduced},
    simd::x86::{WideGhashX4, f128x4_loadu, ghash_mul_x4},
};

/// How [`fold_and_message_x86_avx512`] writes the folded `a`/`b` outputs.
///
/// The caller picks one mode for a whole output slice; every store then uses
/// the same instruction. `Stream512` and `Stream256` are the Z2 non-temporal
/// stores: the outputs are too large to stay cached until the next round reads
/// them, so the read-for-ownership traffic of an ordinary store is wasted.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum FoldStore {
    /// Ordinary cached stores.
    Cached,
    /// `_mm512_stream_si512`; every destination address is 64-byte aligned.
    Stream512,
    /// `_mm256_stream_si256`; every destination address is 32-byte aligned
    /// but may sit 32 bytes off a cache line.
    Stream256,
}

/// Pick the store mode for the two output slices starting at `a_dst` and
/// `b_dst`; `nt` is the caller's decision (buffer size and `FLOCK_ZC_NT`).
///
/// Both fold kernels write whole 4-value groups at 32-byte-multiple offsets
/// from the slice base, so a 64-byte-aligned base supports 64-byte stores and
/// a 32-byte-aligned base supports 32-byte stores. A base that is not even
/// 32-byte aligned falls back to cached stores (the large allocator never
/// produces one, but a test may). Both outputs are written with the same store
/// width, so the less aligned of the two decides it.
#[inline]
pub(crate) fn fold_store_for(a_dst: *const F128, b_dst: *const F128, nt: bool) -> FoldStore {
    if !nt {
        return FoldStore::Cached;
    }
    match (a_dst as usize | b_dst as usize) & 63 {
        0 => FoldStore::Stream512,
        32 => FoldStore::Stream256,
        _ => FoldStore::Cached,
    }
}

/// Store one ZMM of folded values under `STORE` (`0` cached, `1` 64-byte
/// non-temporal, `2` 32-byte non-temporal).
///
/// # Safety
/// `dst` is valid for four `F128` writes and meets `STORE`'s alignment: 64
/// bytes for `1`, 32 bytes for `2`.
#[inline]
#[target_feature(enable = "avx512f,avx2,avx")]
unsafe fn store_fold_x4<const STORE: u8>(dst: *mut F128, v: __m512i) {
    // SAFETY: the caller supplies the bounds and alignment stated above.
    unsafe {
        match STORE {
            1 => _mm512_stream_si512(dst.cast::<__m512i>(), v),
            2 => {
                let lo = _mm512_castsi512_si256(v);
                let hi = _mm512_extracti64x4_epi64(v, 1);
                _mm256_stream_si256(dst.cast::<__m256i>(), lo);
                _mm256_stream_si256(dst.add(2).cast::<__m256i>(), hi);
            }
            _ => _mm512_storeu_si512(dst.cast::<__m512i>(), v),
        }
    }
}

/// Fold the four rows for one round-2 pair in parallel x86 SIMD registers.
/// Returns `[a0, a1, b0, b1]`.
///
/// The table lookups are data-dependent, so they remain four independent
/// aligned 128-bit loads per chunk. Keeping four XOR chains in flight exposes
/// their load-level parallelism; the caller then batches four returned pairs
/// into the AVX-512 GHASH message kernel.
///
/// # Safety
/// `table_data` must point to an 8 × 256 `F128` table and every row pointer
/// must expose 8 readable bytes.
#[inline(always)]
pub(crate) unsafe fn fold_round2_pair_x86_unchecked_8(
    table_data: *const F128,
    a0_bytes: *const u8,
    a1_bytes: *const u8,
    b0_bytes: *const u8,
    b1_bytes: *const u8,
) -> [F128; 4] {
    // SAFETY: the caller guarantees all table and row bounds. Every table
    // entry is 16-byte aligned because F128 has align(16).
    unsafe {
        let rows = [a0_bytes, a1_bytes, b0_bytes, b1_bytes];
        let mut acc = [_mm_setzero_si128(); 4];
        for chunk in 0..8 {
            let table_chunk = table_data.add(chunk * 256);
            for lane in 0..4 {
                let entry = table_chunk.add(*rows[lane].add(chunk) as usize);
                acc[lane] = _mm_xor_si128(acc[lane], _mm_load_si128(entry.cast::<__m128i>()));
            }
        }
        // F128 is exactly two u64 words and accepts every bit pattern.
        acc.map(|value| transmute::<__m128i, F128>(value))
    }
}

/// x86 fused fold plus next-round message for one worker chunk.
///
/// Each four-message iteration folds eight `a` and `b` outputs, stores them
/// for the next round, and consumes the same ZMM registers for the current
/// message before they leave registers. This removes the immediate output
/// readback performed by the portable two-pass path.
///
/// # Safety
/// Input/output lengths must satisfy `input.len() == 2 * output.len()` and
/// `output.len() == 2 * eq_lo.len()`, with `eq_lo.len()` even; `mode` meets
/// its alignment contract for every store address (see [`store_fold_x4`]).
/// The CPU has AVX-512F, VPCLMULQDQ, AVX2 and SSE4.1.
#[target_feature(enable = "avx512f,vpclmulqdq,sse4.1,avx2,avx")]
pub(crate) unsafe fn fold_and_message_x86_avx512(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    r_fold: F128,
    eq_lo: &[F128],
    mode: FoldStore,
) -> (F128, F128) {
    // SAFETY: the caller checked AVX-512; the const arms differ only in the
    // store instruction, and the wrapper has verified `mode`'s alignment.
    unsafe {
        match mode {
            FoldStore::Cached => {
                fold_and_message_impl::<0>(a_in, b_in, a_out, b_out, r_fold, eq_lo)
            }
            FoldStore::Stream512 => {
                fold_and_message_impl::<1>(a_in, b_in, a_out, b_out, r_fold, eq_lo)
            }
            FoldStore::Stream256 => {
                fold_and_message_impl::<2>(a_in, b_in, a_out, b_out, r_fold, eq_lo)
            }
        }
    }
}

/// Body of [`fold_and_message_x86_avx512`]; `STORE` selects the output store
/// (see [`store_fold_x4`]).
///
/// # Safety
/// As [`fold_and_message_x86_avx512`], with the store mode fixed by `STORE`.
#[target_feature(enable = "avx512f,vpclmulqdq,sse4.1,avx2,avx")]
unsafe fn fold_and_message_impl<const STORE: u8>(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    r_fold: F128,
    eq_lo: &[F128],
) -> (F128, F128) {
    debug_assert_eq!(a_in.len(), 2 * a_out.len());
    debug_assert_eq!(b_in.len(), 2 * b_out.len());
    debug_assert_eq!(a_out.len(), 2 * eq_lo.len());

    // Fold four adjacent output elements and return them in one ZMM.
    #[inline]
    #[target_feature(enable = "avx512f,vpclmulqdq")]
    unsafe fn fold_x4(
        src: *const F128,
        r: __m512i,
        even_idx: __m512i,
        odd_idx: __m512i,
    ) -> __m512i {
        // SAFETY: caller supplies eight readable F128 values at src.
        unsafe {
            let lo = _mm512_loadu_si512(src.cast::<__m512i>());
            let hi = _mm512_loadu_si512(src.add(4).cast::<__m512i>());
            let even = _mm512_permutex2var_epi64(lo, even_idx, hi);
            let odd = _mm512_permutex2var_epi64(lo, odd_idx, hi);
            _mm512_xor_si512(even, ghash_mul_x4(r, _mm512_xor_si512(even, odd)))
        }
    }

    // SAFETY: the function's length invariants bound all loads/stores and the
    // caller checked every intrinsic feature at run time.
    unsafe {
        let r = _mm512_broadcast_i32x4(_mm_set_epi64x(r_fold.hi as i64, r_fold.lo as i64));
        // Select even/odd F128 lanes from two concatenated ZMM inputs. The same
        // selectors deinterleave fold inputs and gather message a0/a1 lanes.
        let even_idx = _mm512_set_epi64(13, 12, 9, 8, 5, 4, 1, 0);
        let odd_idx = _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2);
        let mut p1_wide = WideGhashX4::zero();
        let mut pinf_wide = WideGhashX4::zero();
        let mut p1_tail = F256Unreduced::ZERO;
        let mut pinf_tail = F256Unreduced::ZERO;
        let mut x_lo = 0;

        while x_lo + 4 <= eq_lo.len() {
            let output = 2 * x_lo;
            let a_lo = fold_x4(a_in.as_ptr().add(2 * output), r, even_idx, odd_idx);
            let a_hi = fold_x4(a_in.as_ptr().add(2 * (output + 4)), r, even_idx, odd_idx);
            let b_lo = fold_x4(b_in.as_ptr().add(2 * output), r, even_idx, odd_idx);
            let b_hi = fold_x4(b_in.as_ptr().add(2 * (output + 4)), r, even_idx, odd_idx);

            store_fold_x4::<STORE>(a_out.as_mut_ptr().add(output), a_lo);
            store_fold_x4::<STORE>(a_out.as_mut_ptr().add(output + 4), a_hi);
            store_fold_x4::<STORE>(b_out.as_mut_ptr().add(output), b_lo);
            store_fold_x4::<STORE>(b_out.as_mut_ptr().add(output + 4), b_hi);

            let a0 = _mm512_permutex2var_epi64(a_lo, even_idx, a_hi);
            let a1 = _mm512_permutex2var_epi64(a_lo, odd_idx, a_hi);
            let b0 = _mm512_permutex2var_epi64(b_lo, even_idx, b_hi);
            let b1 = _mm512_permutex2var_epi64(b_lo, odd_idx, b_hi);
            let g1 = ghash_mul_x4(a1, b1);
            let g_inf = ghash_mul_x4(_mm512_xor_si512(a0, a1), _mm512_xor_si512(b0, b1));
            let eq = f128x4_loadu(eq_lo.as_ptr().add(x_lo));
            p1_wide.mul_acc(eq, g1);
            pinf_wide.mul_acc(eq, g_inf);
            x_lo += 4;
        }

        // Power-of-two eq blocks leave either no tail or exactly two pairs.
        if x_lo < eq_lo.len() {
            debug_assert_eq!(eq_lo.len() - x_lo, 2);
            let output = 2 * x_lo;
            let a_folded = fold_x4(a_in.as_ptr().add(2 * output), r, even_idx, odd_idx);
            let b_folded = fold_x4(b_in.as_ptr().add(2 * output), r, even_idx, odd_idx);
            store_fold_x4::<STORE>(a_out.as_mut_ptr().add(output), a_folded);
            store_fold_x4::<STORE>(b_out.as_mut_ptr().add(output), b_folded);

            // Read the two pairs out of the registers, not back from the
            // stores: a non-temporal store may not be forwarded to a load.
            // SAFETY: `__m512i` and `[F128; 4]` are both 64 bytes, and F128
            // accepts every bit pattern.
            let a_vals: [F128; 4] = transmute(a_folded);
            let b_vals: [F128; 4] = transmute(b_folded);
            for lane in 0..2 {
                let a0 = a_vals[2 * lane];
                let a1 = a_vals[2 * lane + 1];
                let b0 = b_vals[2 * lane];
                let b1 = b_vals[2 * lane + 1];
                let eq = eq_lo[x_lo + lane];
                p1_tail ^= eq.mul_unreduced(a1 * b1);
                pinf_tail ^= eq.mul_unreduced((a0 + a1) * (b0 + b1));
            }
        }

        p1_tail ^= p1_wide.fold();
        pinf_tail ^= pinf_wide.fold();
        (p1_tail.reduce(), pinf_tail.reduce())
    }
}
