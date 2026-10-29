//! 4-way Keccak-f[1600] and SHAKE extraction.
//!
//! Based on the CC0 XKCP permutation and pq-crystals' CC0 f1600x4,
//! with the rotating register names and row schedule used by Xurong Qi's
//! Hybrid-SHA-3 implementation (translated with the authors' permission).

use super::{
    lanes::{self, Kernel, State},
    Sink, XofJob,
};
use crate::poly::Block;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
fn load(word: &[u64; 4]) -> __m256i {
    // SAFETY: The complete vector is contained in the supplied fixed-size array.
    unsafe { _mm256_loadu_si256(word.as_ptr().cast()) }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
fn store(word: &mut [u64; 4], value: __m256i) {
    // SAFETY: The complete vector is contained in the supplied fixed-size array.
    unsafe { _mm256_storeu_si256(word.as_mut_ptr().cast(), value) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn extract(state: &State<4>, blocks: &mut [Block; 4], rate: usize) {
    if rate == 168 {
        extract_rate::<168>(state, blocks);
    } else {
        extract_rate::<136>(state, blocks);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
fn extract_rate<const RATE: usize>(state: &State<4>, blocks: &mut [Block; 4]) {
    let rate = RATE;
    // Transpose whole register tiles; handle the last one to five words separately.
    let words = rate / 8;
    let tiled = words / 4 * 4;
    for base in (0..tiled).step_by(4) {
        let a = load(&state.0[base]);
        let b = load(&state.0[base + 1]);
        let c = load(&state.0[base + 2]);
        let d = load(&state.0[base + 3]);
        let t0 = _mm256_unpacklo_epi64(a, b);
        let t1 = _mm256_unpackhi_epi64(a, b);
        let t2 = _mm256_unpacklo_epi64(c, d);
        let t3 = _mm256_unpackhi_epi64(c, d);
        let dst: &mut [u8; 32] = (&mut blocks[0].bytes[base * 8..(base + 4) * 8])
            .try_into()
            .unwrap();
        // SAFETY: The store fits exactly in the checked output subarray.
        unsafe {
            _mm256_storeu_si256(
                dst.as_mut_ptr().cast(),
                _mm256_permute2x128_si256::<0x20>(t0, t2),
            )
        };
        let dst: &mut [u8; 32] = (&mut blocks[1].bytes[base * 8..(base + 4) * 8])
            .try_into()
            .unwrap();
        // SAFETY: The store fits exactly in the checked output subarray.
        unsafe {
            _mm256_storeu_si256(
                dst.as_mut_ptr().cast(),
                _mm256_permute2x128_si256::<0x20>(t1, t3),
            )
        };
        let dst: &mut [u8; 32] = (&mut blocks[2].bytes[base * 8..(base + 4) * 8])
            .try_into()
            .unwrap();
        // SAFETY: The store fits exactly in the checked output subarray.
        unsafe {
            _mm256_storeu_si256(
                dst.as_mut_ptr().cast(),
                _mm256_permute2x128_si256::<0x31>(t0, t2),
            )
        };
        let dst: &mut [u8; 32] = (&mut blocks[3].bytes[base * 8..(base + 4) * 8])
            .try_into()
            .unwrap();
        // SAFETY: The store fits exactly in the checked output subarray.
        unsafe {
            _mm256_storeu_si256(
                dst.as_mut_ptr().cast(),
                _mm256_permute2x128_si256::<0x31>(t1, t3),
            )
        };
    }
    for (lane, block) in blocks.iter_mut().enumerate() {
        for word in tiled..words {
            block.bytes[word * 8..word * 8 + 8].copy_from_slice(&state.0[word][lane].to_le_bytes());
        }
        block.bytes[rate..].fill(0);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn run_xof<S: Sink>(jobs: &[XofJob<'_>], sink: &mut S) {
    lanes::run(
        jobs,
        sink,
        Kernel {
            // Closures inherit the enclosing target features.
            permute: |state: &mut State<4>| permute(state),
            // The same target features cover extraction.
            extract: |state: &State<4>, blocks: &mut [Block; 4], rate| extract(state, blocks, rate),
            clear: |state: &mut State<4>, mask| clear(state, mask),
        },
    );
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) unsafe fn run_xof<S: Sink>(_: &[XofJob<'_>], _: &mut S) {
    unreachable!("SIMD backend cannot be constructed on this architecture");
}

// Keep the state in its final interleaved storage between rows. The explicit
// load/store schedule bounds the live vector set to sixteen YMM registers.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) fn permute(state: &mut State<4>) {
    const RC: [u64; 24] = [
        0x1,
        0x8082,
        0x800000000000808a,
        0x8000000080008000,
        0x808b,
        0x80000001,
        0x8000000080008081,
        0x8000000000008009,
        0x8a,
        0x88,
        0x80008009,
        0x8000000a,
        0x8000808b,
        0x800000000000008b,
        0x8000000000008089,
        0x8000000000008003,
        0x8000000000008002,
        0x8000000000000080,
        0x800a,
        0x800000008000000a,
        0x8000000080008081,
        0x8000000000008080,
        0x80000001,
        0x8000000080008008,
    ];
    let rho8 = _mm256_set_epi64x(
        0x0e0d0c0b0a09080f,
        0x0605040302010007,
        0x0e0d0c0b0a09080f,
        0x0605040302010007,
    );
    let rho56 = _mm256_set_epi64x(
        0x080f0e0d0c0b0a09,
        0x0007060504030201,
        0x080f0e0d0c0b0a09,
        0x0007060504030201,
    );
    let y0 = rho8;
    let (
        mut y1,
        mut y2,
        mut y3,
        mut y4,
        mut y5,
        mut y6,
        mut y7,
        mut y8,
        mut y9,
        mut y10,
        mut y11,
        mut y12,
        mut y13,
        mut y14,
        mut y15,
    );
    for round in (0..24).step_by(4) {
        y8 = load(&state.0[0]);
        y9 = load(&state.0[1]);
        y10 = load(&state.0[2]);
        y11 = load(&state.0[3]);
        y12 = load(&state.0[4]);
        y8 = _mm256_xor_si256(y8, load(&state.0[5]));
        y9 = _mm256_xor_si256(y9, load(&state.0[6]));
        y10 = _mm256_xor_si256(y10, load(&state.0[7]));
        y11 = _mm256_xor_si256(y11, load(&state.0[8]));
        y12 = _mm256_xor_si256(y12, load(&state.0[9]));
        y8 = _mm256_xor_si256(y8, load(&state.0[10]));
        y9 = _mm256_xor_si256(y9, load(&state.0[11]));
        y10 = _mm256_xor_si256(y10, load(&state.0[12]));
        y11 = _mm256_xor_si256(y11, load(&state.0[13]));
        y12 = _mm256_xor_si256(y12, load(&state.0[14]));
        y8 = _mm256_xor_si256(y8, load(&state.0[15]));
        y9 = _mm256_xor_si256(y9, load(&state.0[16]));
        y10 = _mm256_xor_si256(y10, load(&state.0[17]));
        y11 = _mm256_xor_si256(y11, load(&state.0[18]));
        y12 = _mm256_xor_si256(y12, load(&state.0[19]));
        y8 = _mm256_xor_si256(y8, load(&state.0[20]));
        y9 = _mm256_xor_si256(y9, load(&state.0[21]));
        y10 = _mm256_xor_si256(y10, load(&state.0[22]));
        y11 = _mm256_xor_si256(y11, load(&state.0[23]));
        y12 = _mm256_xor_si256(y12, load(&state.0[24]));
        y13 = _mm256_slli_epi64::<1>(y9);
        y14 = _mm256_slli_epi64::<1>(y10);
        y15 = _mm256_slli_epi64::<1>(y11);
        y7 = _mm256_slli_epi64::<1>(y12);
        y6 = _mm256_slli_epi64::<1>(y8);
        y5 = _mm256_srli_epi64::<63>(y9);
        y4 = _mm256_srli_epi64::<63>(y10);
        y3 = _mm256_srli_epi64::<63>(y11);
        y2 = _mm256_srli_epi64::<63>(y12);
        y1 = _mm256_srli_epi64::<63>(y8);
        y5 = _mm256_or_si256(y5, y13);
        y4 = _mm256_or_si256(y4, y14);
        y3 = _mm256_or_si256(y3, y15);
        y2 = _mm256_or_si256(y2, y7);
        y1 = _mm256_or_si256(y1, y6);
        y5 = _mm256_xor_si256(y12, y5);
        y4 = _mm256_xor_si256(y8, y4);
        y3 = _mm256_xor_si256(y9, y3);
        y2 = _mm256_xor_si256(y10, y2);
        y1 = _mm256_xor_si256(y11, y1);
        y8 = _mm256_xor_si256(y5, load(&state.0[0]));
        y9 = _mm256_xor_si256(y4, load(&state.0[6]));
        y10 = _mm256_xor_si256(y3, load(&state.0[12]));
        y11 = _mm256_xor_si256(y2, load(&state.0[18]));
        y12 = _mm256_xor_si256(y1, load(&state.0[24]));
        y14 = _mm256_slli_epi64::<44>(y9);
        y15 = _mm256_slli_epi64::<43>(y10);
        y7 = _mm256_slli_epi64::<21>(y11);
        y6 = _mm256_slli_epi64::<14>(y12);
        y9 = _mm256_srli_epi64::<20>(y9);
        y10 = _mm256_srli_epi64::<21>(y10);
        y11 = _mm256_srli_epi64::<43>(y11);
        y12 = _mm256_srli_epi64::<50>(y12);
        y9 = _mm256_or_si256(y9, y14);
        y10 = _mm256_or_si256(y10, y15);
        y11 = _mm256_or_si256(y11, y7);
        y12 = _mm256_or_si256(y12, y6);
        y13 = _mm256_andnot_si256(y9, y10);
        y14 = _mm256_andnot_si256(y10, y11);
        y15 = _mm256_andnot_si256(y11, y12);
        y7 = _mm256_andnot_si256(y12, y8);
        y6 = _mm256_andnot_si256(y8, y9);
        y13 = _mm256_xor_si256(y13, y8);
        y14 = _mm256_xor_si256(y14, y9);
        y15 = _mm256_xor_si256(y15, y10);
        y7 = _mm256_xor_si256(y7, y11);
        y6 = _mm256_xor_si256(y6, y12);
        y8 = _mm256_set1_epi64x(RC[round + 0] as i64);
        y13 = _mm256_xor_si256(y13, y8);
        store(&mut state.0[0], y13);
        store(&mut state.0[6], y14);
        store(&mut state.0[12], y15);
        store(&mut state.0[18], y7);
        store(&mut state.0[24], y6);
        y8 = _mm256_xor_si256(y2, load(&state.0[3]));
        y9 = _mm256_xor_si256(y1, load(&state.0[9]));
        y10 = _mm256_xor_si256(y5, load(&state.0[10]));
        y11 = _mm256_xor_si256(y4, load(&state.0[16]));
        y12 = _mm256_xor_si256(y3, load(&state.0[22]));
        y13 = _mm256_slli_epi64::<28>(y8);
        y14 = _mm256_slli_epi64::<20>(y9);
        y15 = _mm256_slli_epi64::<3>(y10);
        y7 = _mm256_slli_epi64::<45>(y11);
        y6 = _mm256_slli_epi64::<61>(y12);
        y8 = _mm256_srli_epi64::<36>(y8);
        y9 = _mm256_srli_epi64::<44>(y9);
        y10 = _mm256_srli_epi64::<61>(y10);
        y11 = _mm256_srli_epi64::<19>(y11);
        y12 = _mm256_srli_epi64::<3>(y12);
        y8 = _mm256_or_si256(y8, y13);
        y9 = _mm256_or_si256(y9, y14);
        y10 = _mm256_or_si256(y10, y15);
        y11 = _mm256_or_si256(y11, y7);
        y12 = _mm256_or_si256(y12, y6);
        y13 = _mm256_andnot_si256(y9, y10);
        y14 = _mm256_andnot_si256(y10, y11);
        y15 = _mm256_andnot_si256(y11, y12);
        y7 = _mm256_andnot_si256(y12, y8);
        y6 = _mm256_andnot_si256(y8, y9);
        y13 = _mm256_xor_si256(y13, y8);
        y14 = _mm256_xor_si256(y14, y9);
        y15 = _mm256_xor_si256(y15, y10);
        y7 = _mm256_xor_si256(y7, y11);
        y6 = _mm256_xor_si256(y6, y12);
        store(&mut state.0[10], y13);
        store(&mut state.0[16], y14);
        store(&mut state.0[22], y15);
        store(&mut state.0[3], y7);
        store(&mut state.0[9], y6);
        y8 = _mm256_xor_si256(y4, load(&state.0[1]));
        y9 = _mm256_xor_si256(y3, load(&state.0[7]));
        y10 = _mm256_xor_si256(y2, load(&state.0[13]));
        y11 = _mm256_xor_si256(y1, load(&state.0[19]));
        y12 = _mm256_xor_si256(y5, load(&state.0[20]));
        y13 = _mm256_slli_epi64::<1>(y8);
        y14 = _mm256_slli_epi64::<6>(y9);
        y15 = _mm256_slli_epi64::<25>(y10);
        y6 = _mm256_slli_epi64::<18>(y12);
        y8 = _mm256_srli_epi64::<63>(y8);
        y9 = _mm256_srli_epi64::<58>(y9);
        y10 = _mm256_srli_epi64::<39>(y10);
        y12 = _mm256_srli_epi64::<46>(y12);
        y8 = _mm256_or_si256(y8, y13);
        y9 = _mm256_or_si256(y9, y14);
        y10 = _mm256_or_si256(y10, y15);
        y11 = _mm256_shuffle_epi8(y11, y0);
        y12 = _mm256_or_si256(y12, y6);
        y13 = _mm256_andnot_si256(y9, y10);
        y14 = _mm256_andnot_si256(y10, y11);
        y15 = _mm256_andnot_si256(y11, y12);
        y7 = _mm256_andnot_si256(y12, y8);
        y6 = _mm256_andnot_si256(y8, y9);
        y13 = _mm256_xor_si256(y13, y8);
        y14 = _mm256_xor_si256(y14, y9);
        y15 = _mm256_xor_si256(y15, y10);
        y7 = _mm256_xor_si256(y7, y11);
        y6 = _mm256_xor_si256(y6, y12);
        store(&mut state.0[20], y13);
        store(&mut state.0[1], y14);
        store(&mut state.0[7], y15);
        store(&mut state.0[13], y7);
        store(&mut state.0[19], y6);
        y8 = _mm256_xor_si256(y1, load(&state.0[4]));
        y9 = _mm256_xor_si256(y5, load(&state.0[5]));
        y10 = _mm256_xor_si256(y4, load(&state.0[11]));
        y11 = _mm256_xor_si256(y3, load(&state.0[17]));
        y12 = _mm256_xor_si256(y2, load(&state.0[23]));
        y13 = _mm256_slli_epi64::<27>(y8);
        y14 = _mm256_slli_epi64::<36>(y9);
        y15 = _mm256_slli_epi64::<10>(y10);
        y7 = _mm256_slli_epi64::<15>(y11);
        y8 = _mm256_srli_epi64::<37>(y8);
        y9 = _mm256_srli_epi64::<28>(y9);
        y10 = _mm256_srli_epi64::<54>(y10);
        y11 = _mm256_srli_epi64::<49>(y11);
        y8 = _mm256_or_si256(y8, y13);
        y9 = _mm256_or_si256(y9, y14);
        y10 = _mm256_or_si256(y10, y15);
        y11 = _mm256_or_si256(y11, y7);
        y12 = _mm256_shuffle_epi8(y12, rho56);
        y13 = _mm256_andnot_si256(y9, y10);
        y14 = _mm256_andnot_si256(y10, y11);
        y15 = _mm256_andnot_si256(y11, y12);
        y7 = _mm256_andnot_si256(y12, y8);
        y6 = _mm256_andnot_si256(y8, y9);
        y13 = _mm256_xor_si256(y13, y8);
        y14 = _mm256_xor_si256(y14, y9);
        y15 = _mm256_xor_si256(y15, y10);
        y7 = _mm256_xor_si256(y7, y11);
        y6 = _mm256_xor_si256(y6, y12);
        store(&mut state.0[5], y13);
        store(&mut state.0[11], y14);
        store(&mut state.0[17], y15);
        store(&mut state.0[23], y7);
        store(&mut state.0[4], y6);
        y8 = _mm256_xor_si256(y3, load(&state.0[2]));
        y9 = _mm256_xor_si256(y2, load(&state.0[8]));
        y10 = _mm256_xor_si256(y1, load(&state.0[14]));
        y11 = _mm256_xor_si256(y5, load(&state.0[15]));
        y12 = _mm256_xor_si256(y4, load(&state.0[21]));
        y13 = _mm256_slli_epi64::<62>(y8);
        y14 = _mm256_slli_epi64::<55>(y9);
        y15 = _mm256_slli_epi64::<39>(y10);
        y7 = _mm256_slli_epi64::<41>(y11);
        y6 = _mm256_slli_epi64::<2>(y12);
        y8 = _mm256_srli_epi64::<2>(y8);
        y9 = _mm256_srli_epi64::<9>(y9);
        y10 = _mm256_srli_epi64::<25>(y10);
        y11 = _mm256_srli_epi64::<23>(y11);
        y12 = _mm256_srli_epi64::<62>(y12);
        y8 = _mm256_or_si256(y8, y13);
        y9 = _mm256_or_si256(y9, y14);
        y10 = _mm256_or_si256(y10, y15);
        y11 = _mm256_or_si256(y11, y7);
        y12 = _mm256_or_si256(y12, y6);
        y13 = _mm256_andnot_si256(y9, y10);
        y14 = _mm256_andnot_si256(y10, y11);
        y15 = _mm256_andnot_si256(y11, y12);
        y7 = _mm256_andnot_si256(y12, y8);
        y6 = _mm256_andnot_si256(y8, y9);
        y13 = _mm256_xor_si256(y13, y8);
        y14 = _mm256_xor_si256(y14, y9);
        y15 = _mm256_xor_si256(y15, y10);
        y7 = _mm256_xor_si256(y7, y11);
        y6 = _mm256_xor_si256(y6, y12);
        store(&mut state.0[15], y13);
        store(&mut state.0[21], y14);
        store(&mut state.0[2], y15);
        store(&mut state.0[8], y7);
        store(&mut state.0[14], y6);
        y8 = load(&state.0[0]);
        y9 = load(&state.0[1]);
        y10 = load(&state.0[2]);
        y11 = load(&state.0[3]);
        y12 = load(&state.0[4]);
        y8 = _mm256_xor_si256(y8, load(&state.0[5]));
        y9 = _mm256_xor_si256(y9, load(&state.0[6]));
        y10 = _mm256_xor_si256(y10, load(&state.0[7]));
        y11 = _mm256_xor_si256(y11, load(&state.0[8]));
        y12 = _mm256_xor_si256(y12, load(&state.0[9]));
        y8 = _mm256_xor_si256(y8, load(&state.0[10]));
        y9 = _mm256_xor_si256(y9, load(&state.0[11]));
        y10 = _mm256_xor_si256(y10, load(&state.0[12]));
        y11 = _mm256_xor_si256(y11, load(&state.0[13]));
        y12 = _mm256_xor_si256(y12, load(&state.0[14]));
        y8 = _mm256_xor_si256(y8, load(&state.0[15]));
        y9 = _mm256_xor_si256(y9, load(&state.0[16]));
        y10 = _mm256_xor_si256(y10, load(&state.0[17]));
        y11 = _mm256_xor_si256(y11, load(&state.0[18]));
        y12 = _mm256_xor_si256(y12, load(&state.0[19]));
        y8 = _mm256_xor_si256(y8, load(&state.0[20]));
        y9 = _mm256_xor_si256(y9, load(&state.0[21]));
        y10 = _mm256_xor_si256(y10, load(&state.0[22]));
        y11 = _mm256_xor_si256(y11, load(&state.0[23]));
        y12 = _mm256_xor_si256(y12, load(&state.0[24]));
        y13 = _mm256_slli_epi64::<1>(y9);
        y14 = _mm256_slli_epi64::<1>(y10);
        y15 = _mm256_slli_epi64::<1>(y11);
        y7 = _mm256_slli_epi64::<1>(y12);
        y6 = _mm256_slli_epi64::<1>(y8);
        y5 = _mm256_srli_epi64::<63>(y9);
        y4 = _mm256_srli_epi64::<63>(y10);
        y3 = _mm256_srli_epi64::<63>(y11);
        y2 = _mm256_srli_epi64::<63>(y12);
        y1 = _mm256_srli_epi64::<63>(y8);
        y5 = _mm256_or_si256(y5, y13);
        y4 = _mm256_or_si256(y4, y14);
        y3 = _mm256_or_si256(y3, y15);
        y2 = _mm256_or_si256(y2, y7);
        y1 = _mm256_or_si256(y1, y6);
        y5 = _mm256_xor_si256(y12, y5);
        y4 = _mm256_xor_si256(y8, y4);
        y3 = _mm256_xor_si256(y9, y3);
        y2 = _mm256_xor_si256(y10, y2);
        y1 = _mm256_xor_si256(y11, y1);
        y8 = _mm256_xor_si256(y5, load(&state.0[0]));
        y9 = _mm256_xor_si256(y4, load(&state.0[16]));
        y10 = _mm256_xor_si256(y3, load(&state.0[7]));
        y11 = _mm256_xor_si256(y2, load(&state.0[23]));
        y12 = _mm256_xor_si256(y1, load(&state.0[14]));
        y14 = _mm256_slli_epi64::<44>(y9);
        y15 = _mm256_slli_epi64::<43>(y10);
        y7 = _mm256_slli_epi64::<21>(y11);
        y6 = _mm256_slli_epi64::<14>(y12);
        y9 = _mm256_srli_epi64::<20>(y9);
        y10 = _mm256_srli_epi64::<21>(y10);
        y11 = _mm256_srli_epi64::<43>(y11);
        y12 = _mm256_srli_epi64::<50>(y12);
        y9 = _mm256_or_si256(y9, y14);
        y10 = _mm256_or_si256(y10, y15);
        y11 = _mm256_or_si256(y11, y7);
        y12 = _mm256_or_si256(y12, y6);
        y13 = _mm256_andnot_si256(y9, y10);
        y14 = _mm256_andnot_si256(y10, y11);
        y15 = _mm256_andnot_si256(y11, y12);
        y7 = _mm256_andnot_si256(y12, y8);
        y6 = _mm256_andnot_si256(y8, y9);
        y13 = _mm256_xor_si256(y13, y8);
        y14 = _mm256_xor_si256(y14, y9);
        y15 = _mm256_xor_si256(y15, y10);
        y7 = _mm256_xor_si256(y7, y11);
        y6 = _mm256_xor_si256(y6, y12);
        y8 = _mm256_set1_epi64x(RC[round + 1] as i64);
        y13 = _mm256_xor_si256(y13, y8);
        store(&mut state.0[0], y13);
        store(&mut state.0[16], y14);
        store(&mut state.0[7], y15);
        store(&mut state.0[23], y7);
        store(&mut state.0[14], y6);
        y8 = _mm256_xor_si256(y2, load(&state.0[18]));
        y9 = _mm256_xor_si256(y1, load(&state.0[9]));
        y10 = _mm256_xor_si256(y5, load(&state.0[20]));
        y11 = _mm256_xor_si256(y4, load(&state.0[11]));
        y12 = _mm256_xor_si256(y3, load(&state.0[2]));
        y13 = _mm256_slli_epi64::<28>(y8);
        y14 = _mm256_slli_epi64::<20>(y9);
        y15 = _mm256_slli_epi64::<3>(y10);
        y7 = _mm256_slli_epi64::<45>(y11);
        y6 = _mm256_slli_epi64::<61>(y12);
        y8 = _mm256_srli_epi64::<36>(y8);
        y9 = _mm256_srli_epi64::<44>(y9);
        y10 = _mm256_srli_epi64::<61>(y10);
        y11 = _mm256_srli_epi64::<19>(y11);
        y12 = _mm256_srli_epi64::<3>(y12);
        y8 = _mm256_or_si256(y8, y13);
        y9 = _mm256_or_si256(y9, y14);
        y10 = _mm256_or_si256(y10, y15);
        y11 = _mm256_or_si256(y11, y7);
        y12 = _mm256_or_si256(y12, y6);
        y13 = _mm256_andnot_si256(y9, y10);
        y14 = _mm256_andnot_si256(y10, y11);
        y15 = _mm256_andnot_si256(y11, y12);
        y7 = _mm256_andnot_si256(y12, y8);
        y6 = _mm256_andnot_si256(y8, y9);
        y13 = _mm256_xor_si256(y13, y8);
        y14 = _mm256_xor_si256(y14, y9);
        y15 = _mm256_xor_si256(y15, y10);
        y7 = _mm256_xor_si256(y7, y11);
        y6 = _mm256_xor_si256(y6, y12);
        store(&mut state.0[20], y13);
        store(&mut state.0[11], y14);
        store(&mut state.0[2], y15);
        store(&mut state.0[18], y7);
        store(&mut state.0[9], y6);
        y8 = _mm256_xor_si256(y4, load(&state.0[6]));
        y9 = _mm256_xor_si256(y3, load(&state.0[22]));
        y10 = _mm256_xor_si256(y2, load(&state.0[13]));
        y11 = _mm256_xor_si256(y1, load(&state.0[4]));
        y12 = _mm256_xor_si256(y5, load(&state.0[15]));
        y13 = _mm256_slli_epi64::<1>(y8);
        y14 = _mm256_slli_epi64::<6>(y9);
        y15 = _mm256_slli_epi64::<25>(y10);
        y6 = _mm256_slli_epi64::<18>(y12);
        y8 = _mm256_srli_epi64::<63>(y8);
        y9 = _mm256_srli_epi64::<58>(y9);
        y10 = _mm256_srli_epi64::<39>(y10);
        y12 = _mm256_srli_epi64::<46>(y12);
        y8 = _mm256_or_si256(y8, y13);
        y9 = _mm256_or_si256(y9, y14);
        y10 = _mm256_or_si256(y10, y15);
        y11 = _mm256_shuffle_epi8(y11, y0);
        y12 = _mm256_or_si256(y12, y6);
        y13 = _mm256_andnot_si256(y9, y10);
        y14 = _mm256_andnot_si256(y10, y11);
        y15 = _mm256_andnot_si256(y11, y12);
        y7 = _mm256_andnot_si256(y12, y8);
        y6 = _mm256_andnot_si256(y8, y9);
        y13 = _mm256_xor_si256(y13, y8);
        y14 = _mm256_xor_si256(y14, y9);
        y15 = _mm256_xor_si256(y15, y10);
        y7 = _mm256_xor_si256(y7, y11);
        y6 = _mm256_xor_si256(y6, y12);
        store(&mut state.0[15], y13);
        store(&mut state.0[6], y14);
        store(&mut state.0[22], y15);
        store(&mut state.0[13], y7);
        store(&mut state.0[4], y6);
        y8 = _mm256_xor_si256(y1, load(&state.0[24]));
        y9 = _mm256_xor_si256(y5, load(&state.0[10]));
        y10 = _mm256_xor_si256(y4, load(&state.0[1]));
        y11 = _mm256_xor_si256(y3, load(&state.0[17]));
        y12 = _mm256_xor_si256(y2, load(&state.0[8]));
        y13 = _mm256_slli_epi64::<27>(y8);
        y14 = _mm256_slli_epi64::<36>(y9);
        y15 = _mm256_slli_epi64::<10>(y10);
        y7 = _mm256_slli_epi64::<15>(y11);
        y8 = _mm256_srli_epi64::<37>(y8);
        y9 = _mm256_srli_epi64::<28>(y9);
        y10 = _mm256_srli_epi64::<54>(y10);
        y11 = _mm256_srli_epi64::<49>(y11);
        y8 = _mm256_or_si256(y8, y13);
        y9 = _mm256_or_si256(y9, y14);
        y10 = _mm256_or_si256(y10, y15);
        y11 = _mm256_or_si256(y11, y7);
        y12 = _mm256_shuffle_epi8(y12, rho56);
        y13 = _mm256_andnot_si256(y9, y10);
        y14 = _mm256_andnot_si256(y10, y11);
        y15 = _mm256_andnot_si256(y11, y12);
        y7 = _mm256_andnot_si256(y12, y8);
        y6 = _mm256_andnot_si256(y8, y9);
        y13 = _mm256_xor_si256(y13, y8);
        y14 = _mm256_xor_si256(y14, y9);
        y15 = _mm256_xor_si256(y15, y10);
        y7 = _mm256_xor_si256(y7, y11);
        y6 = _mm256_xor_si256(y6, y12);
        store(&mut state.0[10], y13);
        store(&mut state.0[1], y14);
        store(&mut state.0[17], y15);
        store(&mut state.0[8], y7);
        store(&mut state.0[24], y6);
        y8 = _mm256_xor_si256(y3, load(&state.0[12]));
        y9 = _mm256_xor_si256(y2, load(&state.0[3]));
        y10 = _mm256_xor_si256(y1, load(&state.0[19]));
        y11 = _mm256_xor_si256(y5, load(&state.0[5]));
        y12 = _mm256_xor_si256(y4, load(&state.0[21]));
        y13 = _mm256_slli_epi64::<62>(y8);
        y14 = _mm256_slli_epi64::<55>(y9);
        y15 = _mm256_slli_epi64::<39>(y10);
        y7 = _mm256_slli_epi64::<41>(y11);
        y6 = _mm256_slli_epi64::<2>(y12);
        y8 = _mm256_srli_epi64::<2>(y8);
        y9 = _mm256_srli_epi64::<9>(y9);
        y10 = _mm256_srli_epi64::<25>(y10);
        y11 = _mm256_srli_epi64::<23>(y11);
        y12 = _mm256_srli_epi64::<62>(y12);
        y8 = _mm256_or_si256(y8, y13);
        y9 = _mm256_or_si256(y9, y14);
        y10 = _mm256_or_si256(y10, y15);
        y11 = _mm256_or_si256(y11, y7);
        y12 = _mm256_or_si256(y12, y6);
        y13 = _mm256_andnot_si256(y9, y10);
        y14 = _mm256_andnot_si256(y10, y11);
        y15 = _mm256_andnot_si256(y11, y12);
        y7 = _mm256_andnot_si256(y12, y8);
        y6 = _mm256_andnot_si256(y8, y9);
        y13 = _mm256_xor_si256(y13, y8);
        y14 = _mm256_xor_si256(y14, y9);
        y15 = _mm256_xor_si256(y15, y10);
        y7 = _mm256_xor_si256(y7, y11);
        y6 = _mm256_xor_si256(y6, y12);
        store(&mut state.0[5], y13);
        store(&mut state.0[21], y14);
        store(&mut state.0[12], y15);
        store(&mut state.0[3], y7);
        store(&mut state.0[19], y6);
        y8 = load(&state.0[0]);
        y9 = load(&state.0[1]);
        y10 = load(&state.0[2]);
        y11 = load(&state.0[3]);
        y12 = load(&state.0[4]);
        y8 = _mm256_xor_si256(y8, load(&state.0[5]));
        y9 = _mm256_xor_si256(y9, load(&state.0[6]));
        y10 = _mm256_xor_si256(y10, load(&state.0[7]));
        y11 = _mm256_xor_si256(y11, load(&state.0[8]));
        y12 = _mm256_xor_si256(y12, load(&state.0[9]));
        y8 = _mm256_xor_si256(y8, load(&state.0[10]));
        y9 = _mm256_xor_si256(y9, load(&state.0[11]));
        y10 = _mm256_xor_si256(y10, load(&state.0[12]));
        y11 = _mm256_xor_si256(y11, load(&state.0[13]));
        y12 = _mm256_xor_si256(y12, load(&state.0[14]));
        y8 = _mm256_xor_si256(y8, load(&state.0[15]));
        y9 = _mm256_xor_si256(y9, load(&state.0[16]));
        y10 = _mm256_xor_si256(y10, load(&state.0[17]));
        y11 = _mm256_xor_si256(y11, load(&state.0[18]));
        y12 = _mm256_xor_si256(y12, load(&state.0[19]));
        y8 = _mm256_xor_si256(y8, load(&state.0[20]));
        y9 = _mm256_xor_si256(y9, load(&state.0[21]));
        y10 = _mm256_xor_si256(y10, load(&state.0[22]));
        y11 = _mm256_xor_si256(y11, load(&state.0[23]));
        y12 = _mm256_xor_si256(y12, load(&state.0[24]));
        y13 = _mm256_slli_epi64::<1>(y9);
        y14 = _mm256_slli_epi64::<1>(y10);
        y15 = _mm256_slli_epi64::<1>(y11);
        y7 = _mm256_slli_epi64::<1>(y12);
        y6 = _mm256_slli_epi64::<1>(y8);
        y5 = _mm256_srli_epi64::<63>(y9);
        y4 = _mm256_srli_epi64::<63>(y10);
        y3 = _mm256_srli_epi64::<63>(y11);
        y2 = _mm256_srli_epi64::<63>(y12);
        y1 = _mm256_srli_epi64::<63>(y8);
        y5 = _mm256_or_si256(y5, y13);
        y4 = _mm256_or_si256(y4, y14);
        y3 = _mm256_or_si256(y3, y15);
        y2 = _mm256_or_si256(y2, y7);
        y1 = _mm256_or_si256(y1, y6);
        y5 = _mm256_xor_si256(y12, y5);
        y4 = _mm256_xor_si256(y8, y4);
        y3 = _mm256_xor_si256(y9, y3);
        y2 = _mm256_xor_si256(y10, y2);
        y1 = _mm256_xor_si256(y11, y1);
        y8 = _mm256_xor_si256(y5, load(&state.0[0]));
        y9 = _mm256_xor_si256(y4, load(&state.0[11]));
        y10 = _mm256_xor_si256(y3, load(&state.0[22]));
        y11 = _mm256_xor_si256(y2, load(&state.0[8]));
        y12 = _mm256_xor_si256(y1, load(&state.0[19]));
        y14 = _mm256_slli_epi64::<44>(y9);
        y15 = _mm256_slli_epi64::<43>(y10);
        y7 = _mm256_slli_epi64::<21>(y11);
        y6 = _mm256_slli_epi64::<14>(y12);
        y9 = _mm256_srli_epi64::<20>(y9);
        y10 = _mm256_srli_epi64::<21>(y10);
        y11 = _mm256_srli_epi64::<43>(y11);
        y12 = _mm256_srli_epi64::<50>(y12);
        y9 = _mm256_or_si256(y9, y14);
        y10 = _mm256_or_si256(y10, y15);
        y11 = _mm256_or_si256(y11, y7);
        y12 = _mm256_or_si256(y12, y6);
        y13 = _mm256_andnot_si256(y9, y10);
        y14 = _mm256_andnot_si256(y10, y11);
        y15 = _mm256_andnot_si256(y11, y12);
        y7 = _mm256_andnot_si256(y12, y8);
        y6 = _mm256_andnot_si256(y8, y9);
        y13 = _mm256_xor_si256(y13, y8);
        y14 = _mm256_xor_si256(y14, y9);
        y15 = _mm256_xor_si256(y15, y10);
        y7 = _mm256_xor_si256(y7, y11);
        y6 = _mm256_xor_si256(y6, y12);
        y8 = _mm256_set1_epi64x(RC[round + 2] as i64);
        y13 = _mm256_xor_si256(y13, y8);
        store(&mut state.0[0], y13);
        store(&mut state.0[11], y14);
        store(&mut state.0[22], y15);
        store(&mut state.0[8], y7);
        store(&mut state.0[19], y6);
        y8 = _mm256_xor_si256(y2, load(&state.0[23]));
        y9 = _mm256_xor_si256(y1, load(&state.0[9]));
        y10 = _mm256_xor_si256(y5, load(&state.0[15]));
        y11 = _mm256_xor_si256(y4, load(&state.0[1]));
        y12 = _mm256_xor_si256(y3, load(&state.0[12]));
        y13 = _mm256_slli_epi64::<28>(y8);
        y14 = _mm256_slli_epi64::<20>(y9);
        y15 = _mm256_slli_epi64::<3>(y10);
        y7 = _mm256_slli_epi64::<45>(y11);
        y6 = _mm256_slli_epi64::<61>(y12);
        y8 = _mm256_srli_epi64::<36>(y8);
        y9 = _mm256_srli_epi64::<44>(y9);
        y10 = _mm256_srli_epi64::<61>(y10);
        y11 = _mm256_srli_epi64::<19>(y11);
        y12 = _mm256_srli_epi64::<3>(y12);
        y8 = _mm256_or_si256(y8, y13);
        y9 = _mm256_or_si256(y9, y14);
        y10 = _mm256_or_si256(y10, y15);
        y11 = _mm256_or_si256(y11, y7);
        y12 = _mm256_or_si256(y12, y6);
        y13 = _mm256_andnot_si256(y9, y10);
        y14 = _mm256_andnot_si256(y10, y11);
        y15 = _mm256_andnot_si256(y11, y12);
        y7 = _mm256_andnot_si256(y12, y8);
        y6 = _mm256_andnot_si256(y8, y9);
        y13 = _mm256_xor_si256(y13, y8);
        y14 = _mm256_xor_si256(y14, y9);
        y15 = _mm256_xor_si256(y15, y10);
        y7 = _mm256_xor_si256(y7, y11);
        y6 = _mm256_xor_si256(y6, y12);
        store(&mut state.0[15], y13);
        store(&mut state.0[1], y14);
        store(&mut state.0[12], y15);
        store(&mut state.0[23], y7);
        store(&mut state.0[9], y6);
        y8 = _mm256_xor_si256(y4, load(&state.0[16]));
        y9 = _mm256_xor_si256(y3, load(&state.0[2]));
        y10 = _mm256_xor_si256(y2, load(&state.0[13]));
        y11 = _mm256_xor_si256(y1, load(&state.0[24]));
        y12 = _mm256_xor_si256(y5, load(&state.0[5]));
        y13 = _mm256_slli_epi64::<1>(y8);
        y14 = _mm256_slli_epi64::<6>(y9);
        y15 = _mm256_slli_epi64::<25>(y10);
        y6 = _mm256_slli_epi64::<18>(y12);
        y8 = _mm256_srli_epi64::<63>(y8);
        y9 = _mm256_srli_epi64::<58>(y9);
        y10 = _mm256_srli_epi64::<39>(y10);
        y12 = _mm256_srli_epi64::<46>(y12);
        y8 = _mm256_or_si256(y8, y13);
        y9 = _mm256_or_si256(y9, y14);
        y10 = _mm256_or_si256(y10, y15);
        y11 = _mm256_shuffle_epi8(y11, y0);
        y12 = _mm256_or_si256(y12, y6);
        y13 = _mm256_andnot_si256(y9, y10);
        y14 = _mm256_andnot_si256(y10, y11);
        y15 = _mm256_andnot_si256(y11, y12);
        y7 = _mm256_andnot_si256(y12, y8);
        y6 = _mm256_andnot_si256(y8, y9);
        y13 = _mm256_xor_si256(y13, y8);
        y14 = _mm256_xor_si256(y14, y9);
        y15 = _mm256_xor_si256(y15, y10);
        y7 = _mm256_xor_si256(y7, y11);
        y6 = _mm256_xor_si256(y6, y12);
        store(&mut state.0[5], y13);
        store(&mut state.0[16], y14);
        store(&mut state.0[2], y15);
        store(&mut state.0[13], y7);
        store(&mut state.0[24], y6);
        y8 = _mm256_xor_si256(y1, load(&state.0[14]));
        y9 = _mm256_xor_si256(y5, load(&state.0[20]));
        y10 = _mm256_xor_si256(y4, load(&state.0[6]));
        y11 = _mm256_xor_si256(y3, load(&state.0[17]));
        y12 = _mm256_xor_si256(y2, load(&state.0[3]));
        y13 = _mm256_slli_epi64::<27>(y8);
        y14 = _mm256_slli_epi64::<36>(y9);
        y15 = _mm256_slli_epi64::<10>(y10);
        y7 = _mm256_slli_epi64::<15>(y11);
        y8 = _mm256_srli_epi64::<37>(y8);
        y9 = _mm256_srli_epi64::<28>(y9);
        y10 = _mm256_srli_epi64::<54>(y10);
        y11 = _mm256_srli_epi64::<49>(y11);
        y8 = _mm256_or_si256(y8, y13);
        y9 = _mm256_or_si256(y9, y14);
        y10 = _mm256_or_si256(y10, y15);
        y11 = _mm256_or_si256(y11, y7);
        y12 = _mm256_shuffle_epi8(y12, rho56);
        y13 = _mm256_andnot_si256(y9, y10);
        y14 = _mm256_andnot_si256(y10, y11);
        y15 = _mm256_andnot_si256(y11, y12);
        y7 = _mm256_andnot_si256(y12, y8);
        y6 = _mm256_andnot_si256(y8, y9);
        y13 = _mm256_xor_si256(y13, y8);
        y14 = _mm256_xor_si256(y14, y9);
        y15 = _mm256_xor_si256(y15, y10);
        y7 = _mm256_xor_si256(y7, y11);
        y6 = _mm256_xor_si256(y6, y12);
        store(&mut state.0[20], y13);
        store(&mut state.0[6], y14);
        store(&mut state.0[17], y15);
        store(&mut state.0[3], y7);
        store(&mut state.0[14], y6);
        y8 = _mm256_xor_si256(y3, load(&state.0[7]));
        y9 = _mm256_xor_si256(y2, load(&state.0[18]));
        y10 = _mm256_xor_si256(y1, load(&state.0[4]));
        y11 = _mm256_xor_si256(y5, load(&state.0[10]));
        y12 = _mm256_xor_si256(y4, load(&state.0[21]));
        y13 = _mm256_slli_epi64::<62>(y8);
        y14 = _mm256_slli_epi64::<55>(y9);
        y15 = _mm256_slli_epi64::<39>(y10);
        y7 = _mm256_slli_epi64::<41>(y11);
        y6 = _mm256_slli_epi64::<2>(y12);
        y8 = _mm256_srli_epi64::<2>(y8);
        y9 = _mm256_srli_epi64::<9>(y9);
        y10 = _mm256_srli_epi64::<25>(y10);
        y11 = _mm256_srli_epi64::<23>(y11);
        y12 = _mm256_srli_epi64::<62>(y12);
        y8 = _mm256_or_si256(y8, y13);
        y9 = _mm256_or_si256(y9, y14);
        y10 = _mm256_or_si256(y10, y15);
        y11 = _mm256_or_si256(y11, y7);
        y12 = _mm256_or_si256(y12, y6);
        y13 = _mm256_andnot_si256(y9, y10);
        y14 = _mm256_andnot_si256(y10, y11);
        y15 = _mm256_andnot_si256(y11, y12);
        y7 = _mm256_andnot_si256(y12, y8);
        y6 = _mm256_andnot_si256(y8, y9);
        y13 = _mm256_xor_si256(y13, y8);
        y14 = _mm256_xor_si256(y14, y9);
        y15 = _mm256_xor_si256(y15, y10);
        y7 = _mm256_xor_si256(y7, y11);
        y6 = _mm256_xor_si256(y6, y12);
        store(&mut state.0[10], y13);
        store(&mut state.0[21], y14);
        store(&mut state.0[7], y15);
        store(&mut state.0[18], y7);
        store(&mut state.0[4], y6);
        y8 = load(&state.0[0]);
        y9 = load(&state.0[1]);
        y10 = load(&state.0[2]);
        y11 = load(&state.0[3]);
        y12 = load(&state.0[4]);
        y8 = _mm256_xor_si256(y8, load(&state.0[5]));
        y9 = _mm256_xor_si256(y9, load(&state.0[6]));
        y10 = _mm256_xor_si256(y10, load(&state.0[7]));
        y11 = _mm256_xor_si256(y11, load(&state.0[8]));
        y12 = _mm256_xor_si256(y12, load(&state.0[9]));
        y8 = _mm256_xor_si256(y8, load(&state.0[10]));
        y9 = _mm256_xor_si256(y9, load(&state.0[11]));
        y10 = _mm256_xor_si256(y10, load(&state.0[12]));
        y11 = _mm256_xor_si256(y11, load(&state.0[13]));
        y12 = _mm256_xor_si256(y12, load(&state.0[14]));
        y8 = _mm256_xor_si256(y8, load(&state.0[15]));
        y9 = _mm256_xor_si256(y9, load(&state.0[16]));
        y10 = _mm256_xor_si256(y10, load(&state.0[17]));
        y11 = _mm256_xor_si256(y11, load(&state.0[18]));
        y12 = _mm256_xor_si256(y12, load(&state.0[19]));
        y8 = _mm256_xor_si256(y8, load(&state.0[20]));
        y9 = _mm256_xor_si256(y9, load(&state.0[21]));
        y10 = _mm256_xor_si256(y10, load(&state.0[22]));
        y11 = _mm256_xor_si256(y11, load(&state.0[23]));
        y12 = _mm256_xor_si256(y12, load(&state.0[24]));
        y13 = _mm256_slli_epi64::<1>(y9);
        y14 = _mm256_slli_epi64::<1>(y10);
        y15 = _mm256_slli_epi64::<1>(y11);
        y7 = _mm256_slli_epi64::<1>(y12);
        y6 = _mm256_slli_epi64::<1>(y8);
        y5 = _mm256_srli_epi64::<63>(y9);
        y4 = _mm256_srli_epi64::<63>(y10);
        y3 = _mm256_srli_epi64::<63>(y11);
        y2 = _mm256_srli_epi64::<63>(y12);
        y1 = _mm256_srli_epi64::<63>(y8);
        y5 = _mm256_or_si256(y5, y13);
        y4 = _mm256_or_si256(y4, y14);
        y3 = _mm256_or_si256(y3, y15);
        y2 = _mm256_or_si256(y2, y7);
        y1 = _mm256_or_si256(y1, y6);
        y5 = _mm256_xor_si256(y12, y5);
        y4 = _mm256_xor_si256(y8, y4);
        y3 = _mm256_xor_si256(y9, y3);
        y2 = _mm256_xor_si256(y10, y2);
        y1 = _mm256_xor_si256(y11, y1);
        y8 = _mm256_xor_si256(y5, load(&state.0[0]));
        y9 = _mm256_xor_si256(y4, load(&state.0[1]));
        y10 = _mm256_xor_si256(y3, load(&state.0[2]));
        y11 = _mm256_xor_si256(y2, load(&state.0[3]));
        y12 = _mm256_xor_si256(y1, load(&state.0[4]));
        y14 = _mm256_slli_epi64::<44>(y9);
        y15 = _mm256_slli_epi64::<43>(y10);
        y7 = _mm256_slli_epi64::<21>(y11);
        y6 = _mm256_slli_epi64::<14>(y12);
        y9 = _mm256_srli_epi64::<20>(y9);
        y10 = _mm256_srli_epi64::<21>(y10);
        y11 = _mm256_srli_epi64::<43>(y11);
        y12 = _mm256_srli_epi64::<50>(y12);
        y9 = _mm256_or_si256(y9, y14);
        y10 = _mm256_or_si256(y10, y15);
        y11 = _mm256_or_si256(y11, y7);
        y12 = _mm256_or_si256(y12, y6);
        y13 = _mm256_andnot_si256(y9, y10);
        y14 = _mm256_andnot_si256(y10, y11);
        y15 = _mm256_andnot_si256(y11, y12);
        y7 = _mm256_andnot_si256(y12, y8);
        y6 = _mm256_andnot_si256(y8, y9);
        y13 = _mm256_xor_si256(y13, y8);
        y14 = _mm256_xor_si256(y14, y9);
        y15 = _mm256_xor_si256(y15, y10);
        y7 = _mm256_xor_si256(y7, y11);
        y6 = _mm256_xor_si256(y6, y12);
        y8 = _mm256_set1_epi64x(RC[round + 3] as i64);
        y13 = _mm256_xor_si256(y13, y8);
        store(&mut state.0[0], y13);
        store(&mut state.0[1], y14);
        store(&mut state.0[2], y15);
        store(&mut state.0[3], y7);
        store(&mut state.0[4], y6);
        y8 = _mm256_xor_si256(y2, load(&state.0[8]));
        y9 = _mm256_xor_si256(y1, load(&state.0[9]));
        y10 = _mm256_xor_si256(y5, load(&state.0[5]));
        y11 = _mm256_xor_si256(y4, load(&state.0[6]));
        y12 = _mm256_xor_si256(y3, load(&state.0[7]));
        y13 = _mm256_slli_epi64::<28>(y8);
        y14 = _mm256_slli_epi64::<20>(y9);
        y15 = _mm256_slli_epi64::<3>(y10);
        y7 = _mm256_slli_epi64::<45>(y11);
        y6 = _mm256_slli_epi64::<61>(y12);
        y8 = _mm256_srli_epi64::<36>(y8);
        y9 = _mm256_srli_epi64::<44>(y9);
        y10 = _mm256_srli_epi64::<61>(y10);
        y11 = _mm256_srli_epi64::<19>(y11);
        y12 = _mm256_srli_epi64::<3>(y12);
        y8 = _mm256_or_si256(y8, y13);
        y9 = _mm256_or_si256(y9, y14);
        y10 = _mm256_or_si256(y10, y15);
        y11 = _mm256_or_si256(y11, y7);
        y12 = _mm256_or_si256(y12, y6);
        y13 = _mm256_andnot_si256(y9, y10);
        y14 = _mm256_andnot_si256(y10, y11);
        y15 = _mm256_andnot_si256(y11, y12);
        y7 = _mm256_andnot_si256(y12, y8);
        y6 = _mm256_andnot_si256(y8, y9);
        y13 = _mm256_xor_si256(y13, y8);
        y14 = _mm256_xor_si256(y14, y9);
        y15 = _mm256_xor_si256(y15, y10);
        y7 = _mm256_xor_si256(y7, y11);
        y6 = _mm256_xor_si256(y6, y12);
        store(&mut state.0[5], y13);
        store(&mut state.0[6], y14);
        store(&mut state.0[7], y15);
        store(&mut state.0[8], y7);
        store(&mut state.0[9], y6);
        y8 = _mm256_xor_si256(y4, load(&state.0[11]));
        y9 = _mm256_xor_si256(y3, load(&state.0[12]));
        y10 = _mm256_xor_si256(y2, load(&state.0[13]));
        y11 = _mm256_xor_si256(y1, load(&state.0[14]));
        y12 = _mm256_xor_si256(y5, load(&state.0[10]));
        y13 = _mm256_slli_epi64::<1>(y8);
        y14 = _mm256_slli_epi64::<6>(y9);
        y15 = _mm256_slli_epi64::<25>(y10);
        y6 = _mm256_slli_epi64::<18>(y12);
        y8 = _mm256_srli_epi64::<63>(y8);
        y9 = _mm256_srli_epi64::<58>(y9);
        y10 = _mm256_srli_epi64::<39>(y10);
        y12 = _mm256_srli_epi64::<46>(y12);
        y8 = _mm256_or_si256(y8, y13);
        y9 = _mm256_or_si256(y9, y14);
        y10 = _mm256_or_si256(y10, y15);
        y11 = _mm256_shuffle_epi8(y11, y0);
        y12 = _mm256_or_si256(y12, y6);
        y13 = _mm256_andnot_si256(y9, y10);
        y14 = _mm256_andnot_si256(y10, y11);
        y15 = _mm256_andnot_si256(y11, y12);
        y7 = _mm256_andnot_si256(y12, y8);
        y6 = _mm256_andnot_si256(y8, y9);
        y13 = _mm256_xor_si256(y13, y8);
        y14 = _mm256_xor_si256(y14, y9);
        y15 = _mm256_xor_si256(y15, y10);
        y7 = _mm256_xor_si256(y7, y11);
        y6 = _mm256_xor_si256(y6, y12);
        store(&mut state.0[10], y13);
        store(&mut state.0[11], y14);
        store(&mut state.0[12], y15);
        store(&mut state.0[13], y7);
        store(&mut state.0[14], y6);
        y8 = _mm256_xor_si256(y1, load(&state.0[19]));
        y9 = _mm256_xor_si256(y5, load(&state.0[15]));
        y10 = _mm256_xor_si256(y4, load(&state.0[16]));
        y11 = _mm256_xor_si256(y3, load(&state.0[17]));
        y12 = _mm256_xor_si256(y2, load(&state.0[18]));
        y13 = _mm256_slli_epi64::<27>(y8);
        y14 = _mm256_slli_epi64::<36>(y9);
        y15 = _mm256_slli_epi64::<10>(y10);
        y7 = _mm256_slli_epi64::<15>(y11);
        y8 = _mm256_srli_epi64::<37>(y8);
        y9 = _mm256_srli_epi64::<28>(y9);
        y10 = _mm256_srli_epi64::<54>(y10);
        y11 = _mm256_srli_epi64::<49>(y11);
        y8 = _mm256_or_si256(y8, y13);
        y9 = _mm256_or_si256(y9, y14);
        y10 = _mm256_or_si256(y10, y15);
        y11 = _mm256_or_si256(y11, y7);
        y12 = _mm256_shuffle_epi8(y12, rho56);
        y13 = _mm256_andnot_si256(y9, y10);
        y14 = _mm256_andnot_si256(y10, y11);
        y15 = _mm256_andnot_si256(y11, y12);
        y7 = _mm256_andnot_si256(y12, y8);
        y6 = _mm256_andnot_si256(y8, y9);
        y13 = _mm256_xor_si256(y13, y8);
        y14 = _mm256_xor_si256(y14, y9);
        y15 = _mm256_xor_si256(y15, y10);
        y7 = _mm256_xor_si256(y7, y11);
        y6 = _mm256_xor_si256(y6, y12);
        store(&mut state.0[15], y13);
        store(&mut state.0[16], y14);
        store(&mut state.0[17], y15);
        store(&mut state.0[18], y7);
        store(&mut state.0[19], y6);
        y8 = _mm256_xor_si256(y3, load(&state.0[22]));
        y9 = _mm256_xor_si256(y2, load(&state.0[23]));
        y10 = _mm256_xor_si256(y1, load(&state.0[24]));
        y11 = _mm256_xor_si256(y5, load(&state.0[20]));
        y12 = _mm256_xor_si256(y4, load(&state.0[21]));
        y13 = _mm256_slli_epi64::<62>(y8);
        y14 = _mm256_slli_epi64::<55>(y9);
        y15 = _mm256_slli_epi64::<39>(y10);
        y7 = _mm256_slli_epi64::<41>(y11);
        y6 = _mm256_slli_epi64::<2>(y12);
        y8 = _mm256_srli_epi64::<2>(y8);
        y9 = _mm256_srli_epi64::<9>(y9);
        y10 = _mm256_srli_epi64::<25>(y10);
        y11 = _mm256_srli_epi64::<23>(y11);
        y12 = _mm256_srli_epi64::<62>(y12);
        y8 = _mm256_or_si256(y8, y13);
        y9 = _mm256_or_si256(y9, y14);
        y10 = _mm256_or_si256(y10, y15);
        y11 = _mm256_or_si256(y11, y7);
        y12 = _mm256_or_si256(y12, y6);
        y13 = _mm256_andnot_si256(y9, y10);
        y14 = _mm256_andnot_si256(y10, y11);
        y15 = _mm256_andnot_si256(y11, y12);
        y7 = _mm256_andnot_si256(y12, y8);
        y6 = _mm256_andnot_si256(y8, y9);
        y13 = _mm256_xor_si256(y13, y8);
        y14 = _mm256_xor_si256(y14, y9);
        y15 = _mm256_xor_si256(y15, y10);
        y7 = _mm256_xor_si256(y7, y11);
        y6 = _mm256_xor_si256(y6, y12);
        store(&mut state.0[20], y13);
        store(&mut state.0[21], y14);
        store(&mut state.0[22], y15);
        store(&mut state.0[23], y7);
        store(&mut state.0[24], y6);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
fn clear(state: &mut State<4>, mask: u8) {
    let keep = _mm256_set_epi64x(
        if mask & 8 == 0 { -1 } else { 0 },
        if mask & 4 == 0 { -1 } else { 0 },
        if mask & 2 == 0 { -1 } else { 0 },
        if mask & 1 == 0 { -1 } else { 0 },
    );
    for word in &mut state.0 {
        store(word, _mm256_and_si256(load(word), keep));
    }
}
