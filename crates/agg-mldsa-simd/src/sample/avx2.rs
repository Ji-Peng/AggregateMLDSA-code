//! AVX2 implementation of samplers and packing routines.

use super::RejState;
use crate::params::{GAMMA1, Q};
use crate::poly::{Block, Poly32};

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Lookup table mapping an 8-bit rejection mask to 8 permutation indices.
///
/// For an 8-bit mask where bit `k` indicates whether candidate `k` is valid (< Q),
/// the table yields the indices of valid candidates packed into the lowest lanes,
/// followed by zeros for unused lanes.
const IDXLUT: [[u8; 8]; 256] = {
    let mut table = [[0u8; 8]; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut idx = 0usize;
        let mut bit = 0u8;
        while bit < 8 {
            if (i & (1 << bit)) != 0 {
                table[i][idx] = bit;
                idx += 1;
            }
            bit += 1;
        }
        i += 1;
    }
    table
};

/// Rejection sampling from a 168-byte rate block using AVX2.
///
/// Processes 56 candidates (8 per step across 7 steps). Each step loads 32 bytes,
/// unpacks 8 24-bit candidates, compares each against `Q`, and compresses the valid
/// coefficients into `st.coeffs`. Stops and returns `true` once 256 coefficients
/// have been collected.
#[inline]
#[target_feature(enable = "avx2,popcnt")]
pub(crate) unsafe fn rej_uniform(block: &Block, st: &mut RejState) -> bool {
    let mut ctr = st.count;
    if ctr >= 256 {
        st.count = 256;
        return true;
    }

    let bound = _mm256_set1_epi32(Q);
    let mask = _mm256_set1_epi32(0x7FFFFF);
    let idx8 = _mm256_set_epi8(
        -1, 15, 14, 13, -1, 12, 11, 10,
        -1, 9, 8, 7, -1, 6, 5, 4,
        -1, 11, 10, 9, -1, 8, 7, 6,
        -1, 5, 4, 3, -1, 2, 1, 0,
    );

    for step in 0..7 {
        let pos = step * 24;
        // SAFETY: Block has 256 initialized bytes. The maximum read position is
        // 6 * 24 + 32 = 176 bytes, which is strictly within the 256-byte buffer.
        let mut d = unsafe {
            _mm256_loadu_si256(block.bytes.as_ptr().add(pos) as *const __m256i)
        };
        d = _mm256_permute4x64_epi64(d, 0x94);
        d = _mm256_shuffle_epi8(d, idx8);
        d = _mm256_and_si256(d, mask);

        let tmp = _mm256_sub_epi32(d, bound);
        let good = _mm256_movemask_ps(_mm256_castsi256_ps(tmp)) as usize;

        // SAFETY: IDXLUT has 256 entries; good is an 8-bit movemask in 0..256.
        let lut_val = unsafe {
            _mm_loadl_epi64(IDXLUT.as_ptr().add(good) as *const __m128i)
        };
        let perm_indices = _mm256_cvtepu8_epi32(lut_val);
        d = _mm256_permutevar8x32_epi32(d, perm_indices);

        // SAFETY: ctr < 256 and RejState has 272 entries. Storing 8 entries
        // writes at most up to index 255 + 8 = 263 < 272.
        unsafe {
            _mm256_storeu_si256(st.coeffs.as_mut_ptr().add(ctr) as *mut __m256i, d);
        }
        ctr += _popcnt32(good as i32) as usize;

        if ctr >= 256 {
            st.count = 256;
            return true;
        }
    }

    st.count = ctr;
    ctr >= 256
}

/// Rejection sampling of `nblocks` consecutive 168-byte rate blocks straight
/// into a polynomial.
///
/// Continues from `ctr` accepted coefficients and returns the new count (at
/// most 256) and the number of blocks read up to the candidate that completed
/// the polynomial (`nblocks` if it is still open). As in pq-crystals
/// `rej_uniform_avx`, the 8-wide steps run while a full 8-lane store still
/// fits (`ctr <= 248`); the remaining candidates are taken one at a time, so no
/// store passes index 255. Steps of 24 bytes never straddle a block, so the
/// accepted coefficients and block counts equal those of [`rej_uniform`]
/// applied block by block.
///
/// # Panics
/// If `bytes` is shorter than `168 nblocks + 8` (the last step loads 32 bytes).
#[inline]
#[target_feature(enable = "avx2,popcnt")]
pub(crate) fn rej_uniform_into(
    bytes: &[u8],
    nblocks: usize,
    out: &mut [i32; 256],
    mut ctr: usize,
) -> (usize, usize) {
    let len = 168 * nblocks;
    assert!(bytes.len() >= len + 8);
    let bound = _mm256_set1_epi32(Q);
    let mask = _mm256_set1_epi32(0x7FFFFF);
    #[rustfmt::skip]
    let idx8 = _mm256_set_epi8(
        -1, 15, 14, 13, -1, 12, 11, 10,
        -1, 9, 8, 7, -1, 6, 5, 4,
        -1, 11, 10, 9, -1, 8, 7, 6,
        -1, 5, 4, 3, -1, 2, 1, 0,
    );
    let mut pos = 0;
    while pos < len && ctr <= 248 {
        // SAFETY: pos <= len - 24, so the 32-byte load ends at len + 8 <= bytes.len().
        let mut d = unsafe { _mm256_loadu_si256(bytes.as_ptr().add(pos).cast()) };
        d = _mm256_permute4x64_epi64(d, 0x94);
        d = _mm256_shuffle_epi8(d, idx8);
        d = _mm256_and_si256(d, mask);
        let good = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_sub_epi32(d, bound))) as usize;
        // SAFETY: good < 256 indexes the 256-entry table of 8-byte rows.
        let lut = unsafe { _mm_loadl_epi64(IDXLUT.as_ptr().add(good).cast()) };
        d = _mm256_permutevar8x32_epi32(d, _mm256_cvtepu8_epi32(lut));
        // SAFETY: ctr <= 248, so the 8 stored lanes end at index 255 or before.
        unsafe { _mm256_storeu_si256(out.as_mut_ptr().add(ctr).cast(), d) };
        ctr += good.count_ones() as usize;
        pos += 24;
    }
    while pos < len && ctr < 256 {
        let b = &bytes[pos..pos + 3];
        let x = (i32::from(b[0]) | i32::from(b[1]) << 8 | i32::from(b[2]) << 16) & 0x7FFFFF;
        if x < Q {
            out[ctr] = x;
            ctr += 1;
        }
        pos += 3;
    }
    let used = if ctr == 256 {
        pos.div_ceil(168)
    } else {
        nblocks
    };
    (ctr, used)
}

/// Decodes 4 polynomials of response vector z from 2,304 packed bytes in AVX2.
///
/// Unpacks 18-bit signed coefficients into standard representation `GAMMA1 - raw_bits`.
/// Computes and returns the maximum absolute value `||z||_inf` via interleaved max/min
/// tracking across SIMD registers without a secondary memory pass.
#[inline]
#[target_feature(enable = "avx2,popcnt")]
pub(crate) unsafe fn unpack_z(z: &[u8; 2304], out: &mut [Poly32; 4]) -> u32 {
    let shufbidx = _mm256_set_epi8(
        -1, 9, 8, 7, -1, 7, 6, 5, -1, 5, 4, 3, -1, 3, 2, 1,
        -1, 8, 7, 6, -1, 6, 5, 4, -1, 4, 3, 2, -1, 2, 1, 0,
    );
    let srlvdidx = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
    let mask = _mm256_set1_epi32(0x3FFFF);
    let gamma1 = _mm256_set1_epi32(GAMMA1);

    let mut max_val0 = _mm256_setzero_si256();
    let mut min_val0 = _mm256_setzero_si256();
    let mut max_val1 = _mm256_setzero_si256();
    let mut min_val1 = _mm256_setzero_si256();

    let out_ptr = out.as_mut_ptr() as *mut i32;

    // Single linear loop across 124 chunks (31 iterations of 4 chunks = 32 coefficients each).
    // Offsets: up to 123 * 18 + 32 = 2246 <= 2304 are fully within buffer bounds.
    for s in (0..124).step_by(4) {
        let off0 = 18 * s;
        let off1 = off0 + 18;
        let off2 = off1 + 18;
        let off3 = off2 + 18;

        // SAFETY: All offsets plus 32 bytes are within the 2304-byte array z.
        let raw0 = unsafe { _mm256_loadu_si256(z.as_ptr().add(off0) as *const __m256i) };
        // SAFETY: off1 + 32 <= 2304.
        let raw1 = unsafe { _mm256_loadu_si256(z.as_ptr().add(off1) as *const __m256i) };
        // SAFETY: off2 + 32 <= 2304.
        let raw2 = unsafe { _mm256_loadu_si256(z.as_ptr().add(off2) as *const __m256i) };
        // SAFETY: off3 + 32 <= 2304.
        let raw3 = unsafe { _mm256_loadu_si256(z.as_ptr().add(off3) as *const __m256i) };

        let mut f0 = _mm256_permute4x64_epi64(raw0, 0x94);
        let mut f1 = _mm256_permute4x64_epi64(raw1, 0x94);
        let mut f2 = _mm256_permute4x64_epi64(raw2, 0x94);
        let mut f3 = _mm256_permute4x64_epi64(raw3, 0x94);

        f0 = _mm256_shuffle_epi8(f0, shufbidx);
        f1 = _mm256_shuffle_epi8(f1, shufbidx);
        f2 = _mm256_shuffle_epi8(f2, shufbidx);
        f3 = _mm256_shuffle_epi8(f3, shufbidx);

        f0 = _mm256_srlv_epi32(f0, srlvdidx);
        f1 = _mm256_srlv_epi32(f1, srlvdidx);
        f2 = _mm256_srlv_epi32(f2, srlvdidx);
        f3 = _mm256_srlv_epi32(f3, srlvdidx);

        f0 = _mm256_and_si256(f0, mask);
        f1 = _mm256_and_si256(f1, mask);
        f2 = _mm256_and_si256(f2, mask);
        f3 = _mm256_and_si256(f3, mask);

        f0 = _mm256_sub_epi32(gamma1, f0);
        f1 = _mm256_sub_epi32(gamma1, f1);
        f2 = _mm256_sub_epi32(gamma1, f2);
        f3 = _mm256_sub_epi32(gamma1, f3);

        // SAFETY: Poly32 arrays are 64-byte aligned; out_ptr + 8 * s is aligned to 32 bytes.
        unsafe {
            _mm256_store_si256(out_ptr.add(8 * s) as *mut __m256i, f0);
            _mm256_store_si256(out_ptr.add(8 * (s + 1)) as *mut __m256i, f1);
            _mm256_store_si256(out_ptr.add(8 * (s + 2)) as *mut __m256i, f2);
            _mm256_store_si256(out_ptr.add(8 * (s + 3)) as *mut __m256i, f3);
        }

        // Interleaved min/max tracking across ports 0 and 1 while port 5 processes shuffles.
        max_val0 = _mm256_max_epi32(max_val0, _mm256_max_epi32(f0, f1));
        min_val0 = _mm256_min_epi32(min_val0, _mm256_min_epi32(f0, f1));
        max_val1 = _mm256_max_epi32(max_val1, _mm256_max_epi32(f2, f3));
        min_val1 = _mm256_min_epi32(min_val1, _mm256_min_epi32(f2, f3));
    }

    // Chunks 124, 125, 126 (offsets 2232, 2250, 2268): 2268 + 32 = 2300 <= 2304
    for s in 124..127 {
        // SAFETY: 18 * s + 32 <= 2304 for s in 124..127.
        let raw = unsafe { _mm256_loadu_si256(z.as_ptr().add(18 * s) as *const __m256i) };
        let mut f = _mm256_permute4x64_epi64(raw, 0x94);
        f = _mm256_shuffle_epi8(f, shufbidx);
        f = _mm256_srlv_epi32(f, srlvdidx);
        f = _mm256_and_si256(f, mask);
        f = _mm256_sub_epi32(gamma1, f);
        // SAFETY: out_ptr + 8 * s is aligned to 32 bytes and within 1024-entry out buffer.
        unsafe {
            _mm256_store_si256(out_ptr.add(8 * s) as *mut __m256i, f);
        }
        max_val0 = _mm256_max_epi32(max_val0, f);
        min_val0 = _mm256_min_epi32(min_val0, f);
    }

    // Chunk 127 (offset 2286): only 18 bytes remain in z. Bounded stack buffer.
    {
        let mut buf = [0u8; 32];
        buf[..18].copy_from_slice(&z[2286..2304]);
        // SAFETY: buf is 32 bytes on the stack, safely aligned for 32-byte load.
        let raw = unsafe { _mm256_loadu_si256(buf.as_ptr() as *const __m256i) };
        let mut f = _mm256_permute4x64_epi64(raw, 0x94);
        f = _mm256_shuffle_epi8(f, shufbidx);
        f = _mm256_srlv_epi32(f, srlvdidx);
        f = _mm256_and_si256(f, mask);
        f = _mm256_sub_epi32(gamma1, f);
        // SAFETY: out_ptr + 8 * 127 is aligned to 32 bytes and within 1024-entry out buffer.
        unsafe {
            _mm256_store_si256(out_ptr.add(8 * 127) as *mut __m256i, f);
        }
        max_val1 = _mm256_max_epi32(max_val1, f);
        min_val1 = _mm256_min_epi32(min_val1, f);
    }

    // Combine max and min: ||z||_inf = max(max(z), -min(z))
    let max_all = _mm256_max_epi32(max_val0, max_val1);
    let min_all = _mm256_min_epi32(min_val0, min_val1);
    let neg_min = _mm256_sub_epi32(_mm256_setzero_si256(), min_all);
    let max_vec = _mm256_max_epi32(max_all, neg_min);

    let hi128 = _mm256_extracti128_si256(max_vec, 1);
    let lo128 = _mm256_castsi256_si128(max_vec);
    let m128 = _mm_max_epi32(lo128, hi128);
    let m64 = _mm_max_epi32(m128, _mm_shuffle_epi32(m128, 0b01_00_11_10));
    let m32 = _mm_max_epi32(m64, _mm_shuffle_epi32(m64, 0b00_00_00_01));
    _mm_cvtsi128_si32(m32) as u32
}

/// Encodes 4 polynomials of high bits w1 into 768 bytes using AVX2.
///
/// Packs 6-bit values into 3 bytes per 4 coefficients using multiply-add and byte shuffles.
#[inline]
#[target_feature(enable = "avx2,popcnt")]
pub(crate) unsafe fn pack_w1(w1: &[[u8; 256]; 4], out: &mut [u8; 768]) {
    let shift1 = _mm256_set1_epi16((64 << 8) + 1);
    let shift2 = _mm256_set1_epi32((4096 << 16) + 1);
    let shufbidx = _mm256_set_epi8(
        -1, -1, -1, -1, 14, 13, 12, 10, 9, 8, 6, 5, 4, 2, 1, 0,
        -1, -1, -1, -1, 14, 13, 12, 10, 9, 8, 6, 5, 4, 2, 1, 0,
    );
    let shufdidx2 = _mm256_set_epi32(-1, -1, 6, 5, 4, 2, 1, 0);

    for (p, row) in w1.iter().enumerate() {
        for i in 0..8 {
            // SAFETY: 32 * i + 32 <= 256 for all i in 0..8.
            let raw = unsafe { _mm256_loadu_si256(row.as_ptr().add(32 * i) as *const __m256i) };
            let mut f0 = _mm256_maddubs_epi16(raw, shift1);
            f0 = _mm256_madd_epi16(f0, shift2);
            f0 = _mm256_shuffle_epi8(f0, shufbidx);
            f0 = _mm256_permutevar8x32_epi32(f0, shufdidx2);

            let out_offset = p * 192 + 24 * i;
            if out_offset + 32 <= 768 {
                // SAFETY: out_offset + 32 <= 768 is safely within the 768-byte buffer.
                // The trailing 8 bytes will be overwritten by the next chunk or belong to out.
                unsafe {
                    _mm256_storeu_si256(out.as_mut_ptr().add(out_offset) as *mut __m256i, f0);
                }
            } else {
                // For the last chunk (out_offset = 744), storing 32 bytes would write 8 bytes
                // beyond the end of out. Store through a local stack buffer instead.
                let mut tmp = [0u8; 32];
                // SAFETY: tmp is 32 bytes on the stack, safely aligned for 32-byte vector store.
                unsafe {
                    _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, f0);
                }
                out[744..768].copy_from_slice(&tmp[..24]);
            }
        }
    }
}
