//! Bit-matrix transposes shared by the Keccak witness and the parity chain.

/// Eight 64-bit rows to 64 bytes: bit `r` of byte `z` is bit `z` of `rows[r]` (the lane-mask
/// byte of position `z` for eight lanes). One `vpermb` and one `vgf2p8affineqb` where the
/// CPU has AVX-512 VBMI and GFNI, the scalar loop otherwise; the results are identical.
pub fn transpose_8x64(rows: &[u64; 8], out: &mut [u8; 64]) {
    #[cfg(target_arch = "x86_64")]
    if x86::enabled() {
        // SAFETY: the CPU check passed.
        unsafe { x86::transpose_8x64(rows, out) };
        return;
    }
    transpose_8x64_scalar(rows, out);
}

/// The specification of [`transpose_8x64`].
pub fn transpose_8x64_scalar(rows: &[u64; 8], out: &mut [u8; 64]) {
    for (z, o) in out.iter_mut().enumerate() {
        *o = rows.iter().enumerate().fold(0u8, |acc, (r, &x)| acc | ((((x >> z) & 1) as u8) << r));
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use core::arch::x86_64::{
        __m512i, _mm512_gf2p8affine_epi64_epi8, _mm512_loadu_si512, _mm512_permutexvar_epi8, _mm512_set1_epi64,
        _mm512_storeu_si512,
    };
    use std::sync::OnceLock;

    /// Byte permutation: destination byte `8 k + p` takes source byte `8 (7 - p) + k` (byte
    /// `k` of row `7 - p`), so qword `k` of the gathered vector holds byte `k` of every row.
    const GATHER: [u8; 64] = {
        let mut idx = [0u8; 64];
        let mut k = 0;
        while k < 8 {
            let mut p = 0;
            while p < 8 {
                idx[8 * k + p] = (8 * (7 - p) + k) as u8;
                p += 1;
            }
            k += 1;
        }
        idx
    };

    /// Bytes `1 << j` at byte `j` of every qword: the affine map's data operand, so that bit
    /// `i` of result byte `j` of qword `k` is bit `j` of gathered byte `7 - i`, i.e. bit
    /// `8 k + j` of row `i`.
    const IDENTITY: i64 = 0x8040_2010_0804_0201_u64.cast_signed();

    pub(super) fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("avx512vbmi")
                && std::arch::is_x86_feature_detected!("gfni")
        })
    }

    /// # Safety
    /// AVX-512 F, BW, VBMI and GFNI available.
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")]
    pub(super) unsafe fn transpose_8x64(rows: &[u64; 8], out: &mut [u8; 64]) {
        // SAFETY: `rows` and `out` are 64 readable and writable bytes.
        unsafe {
            let v: __m512i = _mm512_loadu_si512(rows.as_ptr().cast());
            let idx: __m512i = _mm512_loadu_si512(GATHER.as_ptr().cast());
            let g = _mm512_permutexvar_epi8(idx, v);
            let t = _mm512_gf2p8affine_epi64_epi8::<0>(_mm512_set1_epi64(IDENTITY), g);
            _mm512_storeu_si512(out.as_mut_ptr().cast(), t);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The dispatched transpose equals the scalar specification on random rows, sparse rows
    /// and the all-ones rows.
    #[test]
    fn transpose_matches_the_scalar_specification() {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for i in 0..300 {
            let rows: [u64; 8] = core::array::from_fn(|_| match i % 3 {
                0 => next(),
                1 => next() & next() & next(),
                _ => u64::MAX,
            });
            let (mut want, mut got) = ([0u8; 64], [0u8; 64]);
            transpose_8x64_scalar(&rows, &mut want);
            transpose_8x64(&rows, &mut got);
            assert_eq!(got, want);
        }
    }
}
