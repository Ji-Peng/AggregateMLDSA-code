//! 8 x 64 bit-matrix transpose of one zmm.
//!
//! Input: eight qwords `v[0..8]` (eight rows of 64 bits). Output: 64 bytes, byte `z` holding
//! bit `z` of every row (bit `r` of byte `z` is bit `z` of `v[r]`), the byte stripe of
//! Flock's lincheck for eight instances. One `vpermb` gathers byte `k` of every row into
//! qword `k` (row `r` at byte `7 - r`); one `vgf2p8affineqb` with the identity bytes
//! `[0x01, 0x02, .., 0x80]` as the data and the gathered qwords as the matrices transposes
//! each 8 x 8 block: result byte `j` of qword `k`, bit `i`, is `parity(A.byte[7 - i] & (1 <<
//! j))` = bit `j` of matrix byte `7 - i`, i.e. bit `8 k + j` of row `i`.

use core::arch::x86_64::{
    __m512i, _mm512_gf2p8affine_epi64_epi8, _mm512_loadu_si512, _mm512_permutexvar_epi8,
    _mm512_set1_epi64,
};

/// Byte permutation of [`transpose_8x64_zmm`]: destination byte `8 k + p` takes source byte
/// `8 (7 - p) + k` (byte `k` of row `7 - p`).
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

/// Bytes `1 << j` at byte `j` of every qword.
const IDENTITY: i64 = 0x8040_2010_0804_0201_u64.cast_signed();

/// The byte-gather index vector of [`transpose_8x64_zmm`], loaded once per kernel.
#[inline]
#[target_feature(enable = "avx512f")]
#[must_use]
pub fn gather_index() -> __m512i {
    // SAFETY: `GATHER` is 64 readable bytes.
    unsafe { _mm512_loadu_si512(GATHER.as_ptr().cast()) }
}

/// [`transpose_8x64`](crate::witness) on one zmm with a preloaded [`gather_index`].
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")]
#[must_use]
pub fn transpose_8x64_with(idx: __m512i, v: __m512i) -> __m512i {
    let g = _mm512_permutexvar_epi8(idx, v);
    _mm512_gf2p8affine_epi64_epi8::<0>(_mm512_set1_epi64(IDENTITY), g)
}

/// Eight qwords (rows) to 64 bytes, byte `z` bit `r` = bit `z` of qword `r`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")]
#[must_use]
pub fn transpose_8x64_zmm(v: __m512i) -> __m512i {
    transpose_8x64_with(gather_index(), v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::arch::x86_64::_mm512_storeu_si512;
    use flock_core::test_rng::Rng;

    #[test]
    fn transpose_matches_the_scalar_oracle() {
        if !crate::simd::detected() {
            eprintln!("no AVX-512 on this CPU: vector transpose not exercised");
            return;
        }
        let mut rng = Rng::new(0x7A);
        for _ in 0..200 {
            let v: [u64; 8] = core::array::from_fn(|_| rng.next_u64());
            let mut want = [0u8; 64];
            crate::witness::transpose_8x64(&v, &mut want);
            let mut got = [0u8; 64];
            // SAFETY: AVX-512 F/BW/VBMI and GFNI detected above; `v` and `got` are 64 bytes.
            unsafe {
                let x = _mm512_loadu_si512(v.as_ptr().cast());
                _mm512_storeu_si512(got.as_mut_ptr().cast(), transpose_8x64_zmm(x));
            }
            assert_eq!(got, want);
        }
    }
}
