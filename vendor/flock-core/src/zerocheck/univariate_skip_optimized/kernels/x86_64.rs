// agg-mldsa patch (PATCH.md): every kernel here carries its own
// `#[target_feature(enable = ...)]` and is chosen at run time by the caller
// (`crate::simd`), so no item is gated on `cfg(target_feature)`.
use core::arch::x86_64::{
    __m128i, __m512i, _mm_gf2p8mul_epi8, _mm_loadu_si128, _mm_set1_epi8, _mm_setzero_si128,
    _mm_storeu_si128, _mm_xor_si128, _mm512_and_si512, _mm512_gf2p8mul_epi8,
    _mm512_loadu_si512, _mm512_permutexvar_epi8, _mm512_set1_epi8, _mm512_set1_epi64,
    _mm512_setzero_si512, _mm512_slli_epi64, _mm512_srli_epi64, _mm512_storeu_si512,
    _mm512_xor_si512,
};

use crate::{
    zerocheck::univariate_skip_optimized::{
        ELL, F8, F128, InvNttTableByteSingleGf8, N_CHUNKS, N_MEDIUM,
    },
};

/// AVX-512 (VBMI) 64-byte bit-transpose — direct port of the NEON two-stage
/// algorithm. `_mm512_permutexvar_epi8` does the byte-gather (NEON `vqtbl4q`)
/// in one instruction; the three masked bit-swap rounds (distances 7/14/28)
/// are identical to the NEON version, applied to all eight 64-bit lanes at once.
///
/// Replaces `bit_transpose_64bytes_scalar` (512 branchy bit ops/call) — which
/// profiling showed was ~85% of round1's time on x86.
#[target_feature(enable = "avx512vbmi,avx512bw,avx512f")]
pub(crate) unsafe fn bit_transpose_64bytes_avx512(input: &[u8; 64], output: &mut [u8; 64]) {
    // Gather index = NEON IDX0 ++ IDX1 ++ IDX2 ++ IDX3 (the 8×8 byte transpose).
    #[rustfmt::skip]
    const IDX: [i8; 64] = [
        0, 8, 16, 24, 32, 40, 48, 56,  1, 9, 17, 25, 33, 41, 49, 57,
        2, 10, 18, 26, 34, 42, 50, 58,  3, 11, 19, 27, 35, 43, 51, 59,
        4, 12, 20, 28, 36, 44, 52, 60,  5, 13, 21, 29, 37, 45, 53, 61,
        6, 14, 22, 30, 38, 46, 54, 62,  7, 15, 23, 31, 39, 47, 55, 63,
    ];
    unsafe {
        let inp = _mm512_loadu_si512(input.as_ptr() as *const __m512i);
        let idx = _mm512_loadu_si512(IDX.as_ptr() as *const __m512i);
        let mut y = _mm512_permutexvar_epi8(idx, inp); // y[i] = input[IDX[i]]

        let mask1 = _mm512_set1_epi64(0x00AA00AA00AA00AAu64 as i64);
        let mask2 = _mm512_set1_epi64(0x0000CCCC0000CCCCu64 as i64);
        let mask3 = _mm512_set1_epi64(0x00000000F0F0F0F0u64 as i64);

        let t = _mm512_and_si512(_mm512_xor_si512(y, _mm512_srli_epi64::<7>(y)), mask1);
        y = _mm512_xor_si512(y, _mm512_xor_si512(t, _mm512_slli_epi64::<7>(t)));
        let t = _mm512_and_si512(_mm512_xor_si512(y, _mm512_srli_epi64::<14>(y)), mask2);
        y = _mm512_xor_si512(y, _mm512_xor_si512(t, _mm512_slli_epi64::<14>(t)));
        let t = _mm512_and_si512(_mm512_xor_si512(y, _mm512_srli_epi64::<28>(y)), mask3);
        y = _mm512_xor_si512(y, _mm512_xor_si512(t, _mm512_slli_epi64::<28>(t)));

        _mm512_storeu_si512(output.as_mut_ptr() as *mut __m512i, y);
    }
}

/// 64-byte bit transpose with SSE2: an 8x8 byte transpose (`punpck` ladder)
/// followed by the three mask/shift rounds on each 64-bit lane.
///
/// One 64-bit word per output column, the same layout as
/// [`bit_transpose_64bytes_words`]. SSE2 is baseline on x86-64, so this runs on
/// the AVX2 tier even when the CPU has no AVX-512 VBMI.
///
/// # Safety
/// x86-64; the input and output are 64 bytes and do not overlap.
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn bit_transpose_64bytes_sse2(input: &[u8; 64], output: &mut [u8; 64]) {
    // SAFETY: [u8; 64] and [u64; 8] have the same size, alignment 1 vs 8 is
    // satisfied by reading through a u64 pointer only if the source is
    // 8-aligned; the transpose copies the bytes into registers first, so read
    // the rows through an unaligned copy instead.
    let mut words = [0u64; 8];
    for (w, chunk) in words.iter_mut().zip(input.chunks_exact(8)) {
        *w = u64::from_le_bytes(chunk.try_into().expect("eight bytes"));
    }
    // SAFETY: SSE2 is baseline on x86-64 and both arrays hold 64 bytes.
    unsafe { crate::simd::x86::bit_transpose_8x64_sse2(&words, output) };
}

/// SSE/GFNI x86 kernel. The inverse-NTT apply uses its best available x86 path,
/// writes two 64-byte columns, and this kernel multiplies them four XMM chunks
/// at a time. Kept as the fallback for GFNI CPUs without AVX-512.
#[inline]
#[allow(dead_code)] // the fallback for GFNI CPUs without AVX-512
#[target_feature(enable = "gfni,sse2")]
pub(crate) unsafe fn shift_reduce_inner_ab_x86_sse(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
    a_col: &mut [F8],
    b_col: &mut [F8],
) {
    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;

    // SAFETY: function carries gfni+sse2; raw loads/stores stay within the
    // validated `a_col`/`b_col` (len ELL) and `out` ([u8; 64]) buffers.
    unsafe {
        // 4 byte-accumulators × 16 lanes = ELL = 64 lanes, reduced F_8 values.
        let mut acc = [_mm_setzero_si128(); 4];
        for k in 0..8usize {
            let chunk_off = byte_base_b + k * N_CHUNKS;
            inv_table.apply(&a_packed[chunk_off..chunk_off + N_CHUNKS], a_col);
            inv_table.apply(&b_packed[chunk_off..chunk_off + N_CHUNKS], b_col);
            let a_ptr = a_col.as_ptr() as *const u8;
            let b_ptr = b_col.as_ptr() as *const u8;
            let xk = _mm_set1_epi8((1u8 << k) as i8); // x^k as an F_8 byte; k=0 ⇒ 1
            for c in 0..4usize {
                let av = _mm_loadu_si128(a_ptr.add(c * 16) as *const __m128i);
                let bv = _mm_loadu_si128(b_ptr.add(c * 16) as *const __m128i);
                // y = (a·b) · x^k in F_8. For k=0, xk=1 ⇒ second mul is identity.
                let y = _mm_gf2p8mul_epi8(_mm_gf2p8mul_epi8(av, bv), xk);
                acc[c] = _mm_xor_si128(acc[c], y);
            }
        }
        let out_ptr = out.as_mut_ptr();
        for c in 0..4usize {
            _mm_storeu_si128(out_ptr.add(c * 16) as *mut __m128i, acc[c]);
        }
    }
}

/// Fused AVX-512/GFNI x86 kernel. Each inverse-NTT apply returns all 64 F_8
/// evaluations in one ZMM register; the product and x^k scaling stay 64-wide
/// and register-resident through the final XOR accumulation.
#[inline]
#[target_feature(enable = "gfni,avx512f,avx512bw")]
pub(crate) unsafe fn shift_reduce_inner_ab_x86_avx512(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
) {
    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;

    // SAFETY: the caller's packed-input bounds guarantee 8 readable bytes at
    // every K-row offset. The table has the protocol-fixed ell=64/chunks=8
    // shape, and `out` is exactly one writable ZMM register.
    unsafe {
        let mut acc = _mm512_setzero_si512();
        for k in 0..8usize {
            let off = byte_base_b + k * N_CHUNKS;
            let av = inv_table.apply_x86_avx512_register_unchecked(a_packed.as_ptr().add(off));
            let bv = inv_table.apply_x86_avx512_register_unchecked(b_packed.as_ptr().add(off));
            let product = _mm512_gf2p8mul_epi8(av, bv);
            // x^0 is the multiplicative identity, so avoid one GFNI operation
            // for the first row.
            let scaled = if k == 0 {
                product
            } else {
                _mm512_gf2p8mul_epi8(product, _mm512_set1_epi8((1u8 << k) as i8))
            };
            acc = _mm512_xor_si512(acc, scaled);
        }
        _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, acc);
    }
}
/// x86 AVX-512 convert-table fold for the two-bank C path. Table lookups stay
/// scalar because their byte-selected addresses are irregular, while four
/// lanes of each resulting F128 accumulator are multiplied by `eq_lo_val` in
/// Generates the window accumulation for one vector width.
macro_rules! accumulate_convert_kernel {
    ($name:ident, $lanes:path, $feat:literal) => {
        /// Window accumulation: per lane, the convert-table sums times
        /// `eq_lo_val`, XORed into the reduced partials.
        ///
        /// # Safety
        /// `n_b_med <= 16` and `convert.len() >= n_b_med * 256`; the partial
        /// arrays are `ELL` long and writable; the CPU has the features of
        /// this width.
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = $feat)]
        pub(crate) unsafe fn $name(
            chunk_ab_bytes: &[[u8; ELL]; 1 << N_MEDIUM],
            chunk_c_bytes: &[[u8; ELL]; 1 << N_MEDIUM],
            n_b_med: usize,
            convert: &[F128],
            eq_lo_val: F128,
            partial_ab: &mut [F128; ELL],
            partial_c_0: &mut [F128; ELL],
            partial_c_1: &mut [F128; ELL],
        ) {
            use $lanes as lanes;
            debug_assert!(n_b_med <= 1 << N_MEDIUM);
            debug_assert_eq!(ELL % lanes::LANES, 0);

            // SAFETY: the fixed-size arrays contain every lane load and store;
            // convert indices are `b_med * 256 + u8`, inside the caller's range.
            unsafe {
                let eq = lanes::splat(eq_lo_val);
                for lane in (0..ELL).step_by(lanes::LANES) {
                    let mut cf_ab = [F128::ZERO; lanes::LANES];
                    let mut cf_c_0 = [F128::ZERO; lanes::LANES];
                    let mut cf_c_1 = [F128::ZERO; lanes::LANES];
                    for b_med in 0..n_b_med {
                        let table_base = b_med * 256;
                        for j in 0..lanes::LANES {
                            let v_ab = chunk_ab_bytes[b_med][lane + j] as usize;
                            let v_c = chunk_c_bytes[b_med][lane + j] as usize;
                            cf_ab[j] += convert[table_base + v_ab];
                            cf_c_0[j] += convert[table_base + (v_c & 0x55)];
                            cf_c_1[j] += convert[table_base + (v_c & 0xAA)];
                        }
                    }

                    let scaled_ab = lanes::mul(lanes::set_from(&cf_ab), eq);
                    let scaled_c_0 = lanes::mul(lanes::set_from(&cf_c_0), eq);
                    let scaled_c_1 = lanes::mul(lanes::set_from(&cf_c_1), eq);

                    let ab_ptr = partial_ab.as_mut_ptr().add(lane);
                    let c0_ptr = partial_c_0.as_mut_ptr().add(lane);
                    let c1_ptr = partial_c_1.as_mut_ptr().add(lane);
                    lanes::storeu(ab_ptr, lanes::xor(lanes::loadu(ab_ptr), scaled_ab));
                    lanes::storeu(c0_ptr, lanes::xor(lanes::loadu(c0_ptr), scaled_c_0));
                    lanes::storeu(c1_ptr, lanes::xor(lanes::loadu(c1_ptr), scaled_c_1));
                }
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
accumulate_convert_kernel!(
    accumulate_convert_with_s_hat_v_x86_avx512,
    crate::simd::ghash4,
    "avx512f,vpclmulqdq,sse4.1"
);
#[cfg(target_arch = "x86_64")]
accumulate_convert_kernel!(
    accumulate_convert_with_s_hat_v_x86_avx2,
    crate::simd::ghash2,
    "avx2,vpclmulqdq"
);
#[cfg(target_arch = "x86_64")]
accumulate_convert_kernel!(
    accumulate_convert_with_s_hat_v_x86_pclmul,
    crate::simd::ghash1,
    "pclmulqdq,sse4.1"
);
