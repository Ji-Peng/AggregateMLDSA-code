//! AVX-512 implementation of samplers and packing routines.

use super::RejState;
use crate::params::Q;
use crate::poly::{Block, Poly32};

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Flag controlling whether rejection sampling writes with compress-to-memory
/// (`_mm512_mask_compressstoreu_epi32`) or compress-in-register followed by full
/// 64-byte vector store (`_mm512_maskz_compress_epi32` + `_mm512_storeu_si512`).
pub const USE_COMPRESS_STORE: bool = false;

/// Permutation byte indices for extracting 16 24-bit candidates from 48 bytes.
///
/// In each 4-byte chunk: bytes 0..2 contain 3 successive input bytes; byte 3 is set
/// to 0x80 so that `vpermb` (`_mm512_permutexvar_epi8`) clears the highest byte.
#[repr(align(64))]
struct Align64<T>(T);

static IDX_U: Align64<[u8; 64]> = Align64([
    0, 1, 2, 0x80, 3, 4, 5, 0x80, 6, 7, 8, 0x80, 9, 10, 11, 0x80,
    12, 13, 14, 0x80, 15, 16, 17, 0x80, 18, 19, 20, 0x80, 21, 22, 23, 0x80,
    24, 25, 26, 0x80, 27, 28, 29, 0x80, 30, 31, 32, 0x80, 33, 34, 35, 0x80,
    36, 37, 38, 0x80, 39, 40, 41, 0x80, 42, 43, 44, 0x80, 45, 46, 47, 0x80,
]);

#[inline(always)]
fn host_has_vbmi() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static CACHED: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);
        let val = CACHED.load(core::sync::atomic::Ordering::Relaxed);
        if val != 0 {
            return val == 2;
        }
        let detected = is_x86_feature_detected!("avx512vbmi");
        CACHED.store(if detected { 2 } else { 1 }, core::sync::atomic::Ordering::Relaxed);
        detected
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

/// Generic VBMI rejection sampler parameterized by store strategy.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt,avx512vbmi")]
unsafe fn rej_uniform_vbmi_impl<const COMPRESS_STORE: bool>(
    block: &Block,
    st: &mut RejState,
) -> bool {
    let mut ctr = st.count;
    if ctr >= 256 {
        st.count = 256;
        return true;
    }

    let bound = _mm512_set1_epi32(Q);
    let mask = _mm512_set1_epi32(0x7FFFFF);
    // SAFETY: IDX_U is 64-byte aligned and 64 bytes in length.
    let idx_u = unsafe { _mm512_load_si512(IDX_U.0.as_ptr() as *const __m512i) };

    // Load and decode 56 candidates across 4 ZMM registers in parallel.
    // SAFETY: Block is 64-byte aligned and 256 bytes long.
    let mut d0 = unsafe { _mm512_load_si512(block.bytes.as_ptr() as *const __m512i) };
    // SAFETY: Offset 48 + 64 = 112 <= 256.
    let mut d1 = unsafe { _mm512_loadu_si512(block.bytes.as_ptr().add(48) as *const __m512i) };
    // SAFETY: Offset 96 + 64 = 160 <= 256.
    let mut d2 = unsafe { _mm512_loadu_si512(block.bytes.as_ptr().add(96) as *const __m512i) };
    // SAFETY: Offset 144 + 64 = 208 <= 256.
    let mut d3 = unsafe { _mm512_loadu_si512(block.bytes.as_ptr().add(144) as *const __m512i) };

    d0 = _mm512_permutexvar_epi8(idx_u, d0);
    d1 = _mm512_permutexvar_epi8(idx_u, d1);
    d2 = _mm512_permutexvar_epi8(idx_u, d2);
    d3 = _mm512_permutexvar_epi8(idx_u, d3);

    d0 = _mm512_and_si512(d0, mask);
    d1 = _mm512_and_si512(d1, mask);
    d2 = _mm512_and_si512(d2, mask);
    d3 = _mm512_and_si512(d3, mask);

    let g0 = _mm512_cmplt_epi32_mask(d0, bound);
    let g1 = _mm512_cmplt_epi32_mask(d1, bound);
    let g2 = _mm512_cmplt_epi32_mask(d2, bound);
    let g3 = _mm512_cmplt_epi32_mask(d3, bound) & 0x00FF;

    // Fast path for blocks where starting count < 200:
    // With 56 candidates per block, ctr + 56 <= 199 + 56 = 255 < 256.
    if ctr < 200 {
        if g0 == 0xFFFF {
            // SAFETY: ctr < 200, RejState has 272 entries. Storing 16 entries writes <= 215 < 272.
            unsafe {
                _mm512_storeu_si512(st.coeffs.as_mut_ptr().add(ctr) as *mut __m512i, d0);
            }
            ctr += 16;
        } else if COMPRESS_STORE {
            // SAFETY: RejState has 272 entries. Storing <= 16 entries writes <= 271.
            unsafe {
                _mm512_mask_compressstoreu_epi32(st.coeffs.as_mut_ptr().add(ctr), g0, d0);
            }
            ctr += g0.count_ones() as usize;
        } else {
            let c0 = _mm512_maskz_compress_epi32(g0, d0);
            // SAFETY: Storing full 64-byte vector writes 16 entries into slack area.
            unsafe {
                _mm512_storeu_si512(st.coeffs.as_mut_ptr().add(ctr) as *mut __m512i, c0);
            }
            ctr += g0.count_ones() as usize;
        }

        if g1 == 0xFFFF {
            // SAFETY: Storing 16 entries writes <= 231 < 272.
            unsafe {
                _mm512_storeu_si512(st.coeffs.as_mut_ptr().add(ctr) as *mut __m512i, d1);
            }
            ctr += 16;
        } else if COMPRESS_STORE {
            // SAFETY: Storing <= 16 entries writes <= 271.
            unsafe {
                _mm512_mask_compressstoreu_epi32(st.coeffs.as_mut_ptr().add(ctr), g1, d1);
            }
            ctr += g1.count_ones() as usize;
        } else {
            let c1 = _mm512_maskz_compress_epi32(g1, d1);
            // SAFETY: Storing full 64-byte vector writes 16 entries into slack area.
            unsafe {
                _mm512_storeu_si512(st.coeffs.as_mut_ptr().add(ctr) as *mut __m512i, c1);
            }
            ctr += g1.count_ones() as usize;
        }

        if g2 == 0xFFFF {
            // SAFETY: Storing 16 entries writes <= 247 < 272.
            unsafe {
                _mm512_storeu_si512(st.coeffs.as_mut_ptr().add(ctr) as *mut __m512i, d2);
            }
            ctr += 16;
        } else if COMPRESS_STORE {
            // SAFETY: Storing <= 16 entries writes <= 271.
            unsafe {
                _mm512_mask_compressstoreu_epi32(st.coeffs.as_mut_ptr().add(ctr), g2, d2);
            }
            ctr += g2.count_ones() as usize;
        } else {
            let c2 = _mm512_maskz_compress_epi32(g2, d2);
            // SAFETY: Storing full 64-byte vector writes 16 entries into slack area.
            unsafe {
                _mm512_storeu_si512(st.coeffs.as_mut_ptr().add(ctr) as *mut __m512i, c2);
            }
            ctr += g2.count_ones() as usize;
        }

        if g3 == 0x00FF {
            // SAFETY: Storing 8 entries writes <= 255 < 272.
            unsafe {
                _mm256_storeu_si256(
                    st.coeffs.as_mut_ptr().add(ctr) as *mut __m256i,
                    _mm512_castsi512_si256(d3),
                );
            }
            ctr += 8;
        } else if COMPRESS_STORE {
            // SAFETY: Storing <= 8 entries writes <= 271.
            unsafe {
                _mm512_mask_compressstoreu_epi32(st.coeffs.as_mut_ptr().add(ctr), g3, d3);
            }
            ctr += g3.count_ones() as usize;
        } else {
            let c3 = _mm512_maskz_compress_epi32(g3, d3);
            // SAFETY: Storing full 64-byte vector writes 16 entries into slack area.
            unsafe {
                _mm512_storeu_si512(st.coeffs.as_mut_ptr().add(ctr) as *mut __m512i, c3);
            }
            ctr += g3.count_ones() as usize;
        }

        st.count = ctr;
        return false;
    }

    // Checked path: for trailing blocks where 256 may be reached mid-block.
    if g0 == 0xFFFF {
        // SAFETY: ctr < 256 and RejState has 272 entries. Storing 16 entries writes <= 271.
        unsafe {
            _mm512_storeu_si512(st.coeffs.as_mut_ptr().add(ctr) as *mut __m512i, d0);
        }
        ctr += 16;
    } else if COMPRESS_STORE {
        // SAFETY: Storing <= 16 entries writes <= 271.
        unsafe {
            _mm512_mask_compressstoreu_epi32(st.coeffs.as_mut_ptr().add(ctr), g0, d0);
        }
        ctr += g0.count_ones() as usize;
    } else {
        let c0 = _mm512_maskz_compress_epi32(g0, d0);
        // SAFETY: Storing 16 entries into slack area.
        unsafe {
            _mm512_storeu_si512(st.coeffs.as_mut_ptr().add(ctr) as *mut __m512i, c0);
        }
        ctr += g0.count_ones() as usize;
    }
    if ctr >= 256 {
        st.count = 256;
        return true;
    }

    if g1 == 0xFFFF {
        // SAFETY: Same bounds: at most 16 entries stored into slack area.
        unsafe {
            _mm512_storeu_si512(st.coeffs.as_mut_ptr().add(ctr) as *mut __m512i, d1);
        }
        ctr += 16;
    } else if COMPRESS_STORE {
        // SAFETY: Storing <= 16 entries writes <= 271.
        unsafe {
            _mm512_mask_compressstoreu_epi32(st.coeffs.as_mut_ptr().add(ctr), g1, d1);
        }
        ctr += g1.count_ones() as usize;
    } else {
        let c1 = _mm512_maskz_compress_epi32(g1, d1);
        // SAFETY: Storing 16 entries into slack area.
        unsafe {
            _mm512_storeu_si512(st.coeffs.as_mut_ptr().add(ctr) as *mut __m512i, c1);
        }
        ctr += g1.count_ones() as usize;
    }
    if ctr >= 256 {
        st.count = 256;
        return true;
    }

    if g2 == 0xFFFF {
        // SAFETY: Same bounds: at most 16 entries stored into slack area.
        unsafe {
            _mm512_storeu_si512(st.coeffs.as_mut_ptr().add(ctr) as *mut __m512i, d2);
        }
        ctr += 16;
    } else if COMPRESS_STORE {
        // SAFETY: Storing <= 16 entries writes <= 271.
        unsafe {
            _mm512_mask_compressstoreu_epi32(st.coeffs.as_mut_ptr().add(ctr), g2, d2);
        }
        ctr += g2.count_ones() as usize;
    } else {
        let c2 = _mm512_maskz_compress_epi32(g2, d2);
        // SAFETY: Storing 16 entries into slack area.
        unsafe {
            _mm512_storeu_si512(st.coeffs.as_mut_ptr().add(ctr) as *mut __m512i, c2);
        }
        ctr += g2.count_ones() as usize;
    }
    if ctr >= 256 {
        st.count = 256;
        return true;
    }

    if g3 == 0x00FF {
        // SAFETY: Storing 8 entries writes <= 271.
        unsafe {
            _mm256_storeu_si256(
                st.coeffs.as_mut_ptr().add(ctr) as *mut __m256i,
                _mm512_castsi512_si256(d3),
            );
        }
        ctr += 8;
    } else if COMPRESS_STORE {
        // SAFETY: Storing <= 8 entries writes <= 271.
        unsafe {
            _mm512_mask_compressstoreu_epi32(st.coeffs.as_mut_ptr().add(ctr), g3, d3);
        }
        ctr += g3.count_ones() as usize;
    } else {
        let c3 = _mm512_maskz_compress_epi32(g3, d3);
        // SAFETY: Storing 16 entries into slack area.
        unsafe {
            _mm512_storeu_si512(st.coeffs.as_mut_ptr().add(ctr) as *mut __m512i, c3);
        }
        ctr += g3.count_ones() as usize;
    }

    st.count = ctr.min(256);
    ctr >= 256
}

/// Uniform rejection sampling from a 168-byte rate block using AVX-512 VBMI.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt,avx512vbmi")]
pub(crate) unsafe fn rej_uniform_vbmi(block: &Block, st: &mut RejState) -> bool {
    // SAFETY: caller guarantees target features.
    unsafe { rej_uniform_vbmi_impl::<USE_COMPRESS_STORE>(block, st) }
}

/// Variant of VBMI rejection sampling using `_mm512_mask_compressstoreu_epi32`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt,avx512vbmi")]
pub(crate) unsafe fn rej_uniform_vbmi_compressstore(block: &Block, st: &mut RejState) -> bool {
    // SAFETY: caller guarantees target features.
    unsafe { rej_uniform_vbmi_impl::<true>(block, st) }
}

/// Variant of VBMI rejection sampling using `_mm512_maskz_compress_epi32` + `_mm512_storeu_si512`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt,avx512vbmi")]
pub(crate) unsafe fn rej_uniform_vbmi_compress_reg(block: &Block, st: &mut RejState) -> bool {
    // SAFETY: caller guarantees target features.
    unsafe { rej_uniform_vbmi_impl::<false>(block, st) }
}

/// Uniform rejection sampling from a 168-byte rate block using AVX-512.
///
/// Uses the VBMI vectorized sampler if VBMI is detected on the host; otherwise,
/// falls back to the AVX2 implementation.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
pub(crate) unsafe fn rej_uniform(block: &Block, st: &mut RejState) -> bool {
    if host_has_vbmi() {
        // SAFETY: avx512vbmi feature is dynamically detected.
        return unsafe { rej_uniform_vbmi(block, st) };
    }
    // SAFETY: AVX2 and POPCNT are guaranteed by the target_feature attribute.
    unsafe { super::avx2::rej_uniform(block, st) }
}

/// Decodes 4 polynomials of response vector z from 2,304 packed bytes.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
pub(crate) unsafe fn unpack_z(z: &[u8; 2304], out: &mut [Poly32; 4]) -> u32 {
    // SAFETY: AVX2 and POPCNT are guaranteed by target_feature.
    unsafe { super::avx2::unpack_z(z, out) }
}

/// Encodes 4 polynomials of high bits w1 into 768 bytes.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
pub(crate) unsafe fn pack_w1(w1: &[[u8; 256]; 4], out: &mut [u8; 768]) {
    // SAFETY: AVX2 and POPCNT are guaranteed by target_feature.
    unsafe { super::avx2::pack_w1(w1, out) }
}
