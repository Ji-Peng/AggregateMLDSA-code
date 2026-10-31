use core::{
    arch::x86_64::{__m128i, _mm_load_si128, _mm_setzero_si128, _mm_xor_si128},
    mem::transmute,
};

// agg-mldsa patch (PATCH.md): the GHASH primitives come from `crate::simd`
// (no `cfg(target_feature)` gate), and every kernel carries its own
// `#[target_feature]`, so the caller chooses the vector width at run time.
use crate::field::{F128, F256Unreduced};

#[cfg(target_arch = "x86_64")]
use crate::simd::{ghash1, ghash2, ghash4};

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

/// Fused fold and next-round message for one worker chunk.
///
/// Each message iteration folds `2 * LANES` `a` and `b` outputs, stores them
/// for the next round, and consumes the same vector registers for the current
/// message before they leave registers. This removes the immediate output
/// readback of the portable two-pass path.
///
/// The body is generated for 512-bit VPCLMULQDQ, 256-bit VPCLMULQDQ and
/// 128-bit PCLMULQDQ; every width computes the same folded values and the same
/// unreduced message sums.
macro_rules! fold_and_message_kernels {
    ($name:ident, $lanes:path, $v:ty, $feat:literal) => {
        /// See [`fold_and_message_kernels`].
        ///
        /// # Safety
        /// `a_in.len() == 2 * a_out.len()`, `b_in.len() == 2 * b_out.len()`,
        /// `a_out.len() == 2 * eq_lo.len()`, and the CPU has the features of
        /// this width.
        #[target_feature(enable = $feat)]
        pub(crate) unsafe fn $name(
            a_in: &[F128],
            b_in: &[F128],
            a_out: &mut [F128],
            b_out: &mut [F128],
            r_fold: F128,
            eq_lo: &[F128],
        ) -> (F128, F128) {
            use $lanes as lanes;

            debug_assert_eq!(a_in.len(), 2 * a_out.len());
            debug_assert_eq!(b_in.len(), 2 * b_out.len());
            debug_assert_eq!(a_out.len(), 2 * eq_lo.len());
            let lanes = lanes::LANES;

            // Fold `2 * LANES` inputs into `LANES` outputs and return them in
            // one vector.
            //
            // # Safety
            // Eight readable `F128` at `src`; the width's features are enabled.
            #[inline]
            #[target_feature(enable = $feat)]
            unsafe fn fold(src: *const F128, r: $v) -> $v {
                use $lanes as lanes;
                // SAFETY: the caller supplies `2 * LANES` readable inputs.
                unsafe {
                    let (even, odd) = lanes::even_odd(
                        lanes::loadu(src),
                        lanes::loadu(src.add(lanes::LANES)),
                    );
                    lanes::xor(even, lanes::mul(r, lanes::xor(even, odd)))
                }
            }

            // SAFETY: the length invariants bound every load and store.
            unsafe {
                let r = lanes::splat(r_fold);
                let mut p1_wide = lanes::wide_zero();
                let mut pinf_wide = lanes::wide_zero();
                let mut p1_tail = F256Unreduced::ZERO;
                let mut pinf_tail = F256Unreduced::ZERO;
                let mut x_lo = 0;

                while x_lo + lanes <= eq_lo.len() {
                    let output = 2 * x_lo;
                    let a_lo = fold(a_in.as_ptr().add(2 * output), r);
                    let a_hi = fold(a_in.as_ptr().add(2 * (output + lanes)), r);
                    let b_lo = fold(b_in.as_ptr().add(2 * output), r);
                    let b_hi = fold(b_in.as_ptr().add(2 * (output + lanes)), r);

                    lanes::storeu(a_out.as_mut_ptr().add(output), a_lo);
                    lanes::storeu(a_out.as_mut_ptr().add(output + lanes), a_hi);
                    lanes::storeu(b_out.as_mut_ptr().add(output), b_lo);
                    lanes::storeu(b_out.as_mut_ptr().add(output + lanes), b_hi);

                    let (a0, a1) = lanes::even_odd(a_lo, a_hi);
                    let (b0, b1) = lanes::even_odd(b_lo, b_hi);
                    let g1 = lanes::mul(a1, b1);
                    let g_inf = lanes::mul(lanes::xor(a0, a1), lanes::xor(b0, b1));
                    let eq = lanes::loadu(eq_lo.as_ptr().add(x_lo));
                    lanes::wide_acc(&mut p1_wide, eq, g1);
                    lanes::wide_acc(&mut pinf_wide, eq, g_inf);
                    x_lo += lanes;
                }

                // Fold the remaining outputs one at a time; a power-of-two eq
                // block leaves fewer than `LANES` entries here.
                while x_lo < eq_lo.len() {
                    let output = 2 * x_lo;
                    for k in 0..2 {
                        let o = output + k;
                        let a0 = a_in[2 * o];
                        let a1 = a_in[2 * o + 1];
                        let b0 = b_in[2 * o];
                        let b1 = b_in[2 * o + 1];
                        a_out[o] = a0 + r_fold * (a1 + a0);
                        b_out[o] = b0 + r_fold * (b1 + b0);
                    }
                    let o = 2 * x_lo;
                    let a0 = a_out[o];
                    let a1 = a_out[o + 1];
                    let b0 = b_out[o];
                    let b1 = b_out[o + 1];
                    let eq = eq_lo[x_lo];
                    p1_tail ^= eq.mul_unreduced(a1 * b1);
                    pinf_tail ^= eq.mul_unreduced((a0 + a1) * (b0 + b1));
                    x_lo += 1;
                }

                p1_tail ^= lanes::wide_fold(p1_wide);
                pinf_tail ^= lanes::wide_fold(pinf_wide);
                (p1_tail.reduce(), pinf_tail.reduce())
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
fold_and_message_kernels!(
    fold_and_message_x86_avx512,
    ghash4,
    core::arch::x86_64::__m512i,
    "avx512f,vpclmulqdq,sse4.1"
);
#[cfg(target_arch = "x86_64")]
fold_and_message_kernels!(
    fold_and_message_x86_avx2,
    ghash2,
    core::arch::x86_64::__m256i,
    "avx2,vpclmulqdq"
);
#[cfg(target_arch = "x86_64")]
fold_and_message_kernels!(
    fold_and_message_x86_pclmul,
    ghash1,
    core::arch::x86_64::__m128i,
    "pclmulqdq,sse4.1"
);
