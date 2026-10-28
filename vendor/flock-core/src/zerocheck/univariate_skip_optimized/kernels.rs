#[cfg(all(test, target_arch = "aarch64"))]
pub(super) use portable::bit_transpose_64bytes_scalar;
#[cfg(all(test, target_arch = "aarch64"))]
pub(super) use portable::shift_reduce_inner_ab_scalar;
#[cfg(all(test, target_arch = "x86_64"))]
pub(super) use portable::shift_reduce_inner_ab_scalar as shift_reduce_inner_ab_scalar_oracle;

#[cfg(target_arch = "aarch64")]
use crate::zerocheck::univariate_skip_optimized::kernels::aarch64::{
    accumulate_convert as accumulate_convert_aarch64,
    accumulate_convert_with_s_hat_v as accumulate_convert_with_s_hat_v_aarch64,
    bit_transpose_64bytes_neon, shift_reduce_inner_ab_fused_neon,
};
#[cfg(not(target_arch = "aarch64"))]
use crate::zerocheck::univariate_skip_optimized::kernels::portable::{
    accumulate_convert as accumulate_convert_portable,
    accumulate_convert_with_s_hat_v as accumulate_convert_with_s_hat_v_portable,
    bit_transpose_64bytes_scalar, shift_reduce_inner_ab_scalar,
};
// agg-mldsa patch (PATCH.md): the x86 kernels below are chosen at run
// time (`crate::simd`) instead of by `cfg(target_feature)`.
#[cfg(target_arch = "x86_64")]
use crate::{
    simd::{AVX512_CLMUL, AVX512_GFNI, AVX512_VBMI, GFNI, has},
    zerocheck::univariate_skip_optimized::kernels::x86_64::{
        accumulate_convert_with_s_hat_v_x86_avx512, bit_transpose_64bytes_avx512,
        shift_reduce_inner_ab_x86_avx512, shift_reduce_inner_ab_x86_sse,
    },
};
use crate::zerocheck::univariate_skip_optimized::{F8, F128, InvNttTableByteSingleGf8};
mod portable;

#[cfg(target_arch = "aarch64")]
pub(super) mod aarch64;

#[cfg(target_arch = "x86_64")]
pub(super) mod x86_64;

#[inline]
pub(super) fn bit_transpose_64bytes(input: &[u8; 64], output: &mut [u8; 64]) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: aarch64 statically guarantees NEON.
    unsafe {
        bit_transpose_64bytes_neon(input, output);
    }

    #[cfg(target_arch = "x86_64")]
    if has(AVX512_VBMI) {
        // SAFETY: the runtime check guarantees the kernel's target features.
        unsafe { bit_transpose_64bytes_avx512(input, output) };
        return;
    }

    #[cfg(not(target_arch = "aarch64"))]
    bit_transpose_64bytes_scalar(input, output);
}

#[allow(clippy::too_many_arguments)]
pub(super) fn shift_reduce_inner_ab(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
    a_col: &mut [F8],
    b_col: &mut [F8],
) {
    #[cfg(target_arch = "aarch64")]
    {
        let _ = (a_col, b_col);
        shift_reduce_inner_ab_fused_neon(
            a_packed,
            b_packed,
            inv_table,
            chunk_byte_base,
            b_med,
            out,
        );
    }

    #[cfg(target_arch = "x86_64")]
    if has(AVX512_GFNI) && inv_table.ell == 64 && inv_table.n_chunks == 8 {
        // The fused kernel reads eight bytes at every K-row offset; check what the
        // slice-based paths would have checked.
        let end = chunk_byte_base + (b_med + 1) * 64;
        assert!(end <= a_packed.len() && end <= b_packed.len());
        // SAFETY: the runtime check guarantees gfni+avx512f+avx512bw; bounds above.
        unsafe {
            shift_reduce_inner_ab_x86_avx512(
                a_packed,
                b_packed,
                inv_table,
                chunk_byte_base,
                b_med,
                out,
            );
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if has(GFNI) {
        // SAFETY: the runtime check guarantees gfni; SSE2 is baseline on x86_64.
        unsafe {
            shift_reduce_inner_ab_x86_sse(
                a_packed,
                b_packed,
                inv_table,
                chunk_byte_base,
                b_med,
                out,
                a_col,
                b_col,
            );
        }
        return;
    }

    #[cfg(not(target_arch = "aarch64"))]
    shift_reduce_inner_ab_scalar(
        a_packed,
        b_packed,
        inv_table,
        chunk_byte_base,
        b_med,
        out,
        a_col,
        b_col,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn accumulate_convert(
    chunk_ab_bytes: &[[u8; 64]; 16],
    chunk_c_bytes: &[[u8; 64]; 16],
    n_b_med: usize,
    convert: &[F128],
    eq_lo_val: F128,
    partial_ab: &mut [F128; 64],
    partial_c: &mut [F128; 64],
) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: aarch64 statically guarantees NEON and the fixed arrays cover
    // all table-selected loads.
    unsafe {
        accumulate_convert_aarch64(
            chunk_ab_bytes,
            chunk_c_bytes,
            n_b_med,
            convert,
            eq_lo_val,
            partial_ab,
            partial_c,
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    accumulate_convert_portable(
        chunk_ab_bytes,
        chunk_c_bytes,
        n_b_med,
        convert,
        eq_lo_val,
        partial_ab,
        partial_c,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn accumulate_convert_with_s_hat_v(
    chunk_ab_bytes: &[[u8; 64]; 16],
    chunk_c_bytes: &[[u8; 64]; 16],
    n_b_med: usize,
    convert: &[F128],
    eq_lo_val: F128,
    partial_ab: &mut [F128; 64],
    partial_c_0: &mut [F128; 64],
    partial_c_1: &mut [F128; 64],
) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: aarch64 statically guarantees NEON and the fixed arrays cover
    // all table-selected loads.
    unsafe {
        accumulate_convert_with_s_hat_v_aarch64(
            chunk_ab_bytes,
            chunk_c_bytes,
            n_b_med,
            convert,
            eq_lo_val,
            partial_ab,
            partial_c_0,
            partial_c_1,
        );
    }

    #[cfg(target_arch = "x86_64")]
    if has(AVX512_CLMUL) {
        // The kernel indexes `convert` without bounds checks in release builds of
        // upstream; keep the table-size precondition explicit.
        assert!(n_b_med <= 16 && convert.len() >= n_b_med * 256);
        // SAFETY: the runtime check guarantees the SIMD features and the fixed
        // arrays cover every four-lane load/store.
        unsafe {
            accumulate_convert_with_s_hat_v_x86_avx512(
                chunk_ab_bytes,
                chunk_c_bytes,
                n_b_med,
                convert,
                eq_lo_val,
                partial_ab,
                partial_c_0,
                partial_c_1,
            );
        }
        return;
    }

    #[cfg(not(target_arch = "aarch64"))]
    accumulate_convert_with_s_hat_v_portable(
        chunk_ab_bytes,
        chunk_c_bytes,
        n_b_med,
        convert,
        eq_lo_val,
        partial_ab,
        partial_c_0,
        partial_c_1,
    );
}

/// agg-mldsa patch (PATCH.md): the round-1 x86 kernels available on
/// this CPU match the portable ones bit for bit; the dispatching wrappers match
/// the portable kernels under every feature mask.
#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::{
        accumulate_convert_with_s_hat_v, accumulate_convert_with_s_hat_v_portable,
        bit_transpose_64bytes, bit_transpose_64bytes_scalar,
    };
    use crate::{
        field::F128,
        simd::{test_masks, with_mask},
        test_rng::Rng,
    };

    #[test]
    fn bit_transpose_every_path_matches_scalar() {
        let mut rng = Rng::new(0x7A5);
        for _ in 0..64 {
            let mut input = [0u8; 64];
            rng.fill_bytes(&mut input);
            let mut want = [0u8; 64];
            bit_transpose_64bytes_scalar(&input, &mut want);
            for mask in test_masks() {
                let mut got = [0u8; 64];
                with_mask(mask, || bit_transpose_64bytes(&input, &mut got));
                assert_eq!(got, want, "mask {mask:#x}");
            }
        }
    }

    #[test]
    fn accumulate_convert_every_path_matches_portable() {
        let mut rng = Rng::new(0xACC);
        let convert = rng.f128_vec(16 * 256);
        for n_b_med in [1usize, 5, 16] {
            let mut ab = [[0u8; 64]; 16];
            let mut c = [[0u8; 64]; 16];
            for row in ab.iter_mut().chain(c.iter_mut()) {
                rng.fill_bytes(row);
            }
            let eq = rng.f128();
            let init: [[F128; 64]; 3] = core::array::from_fn(|_| {
                let v = rng.f128_vec(64);
                core::array::from_fn(|i| v[i])
            });
            let mut want = init;
            {
                let [w0, w1, w2] = &mut want;
                accumulate_convert_with_s_hat_v_portable(&ab, &c, n_b_med, &convert, eq, w0, w1, w2);
            }
            for mask in test_masks() {
                let mut got = init;
                {
                    let [g0, g1, g2] = &mut got;
                    with_mask(mask, || {
                        accumulate_convert_with_s_hat_v(&ab, &c, n_b_med, &convert, eq, g0, g1, g2)
                    });
                }
                assert_eq!(got, want, "n_b_med={n_b_med} mask {mask:#x}");
            }
        }
    }
}
