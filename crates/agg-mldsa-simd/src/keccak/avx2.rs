//! 4-way Keccak-f[1600] and block extraction (AVX2).
//!
//! The production permutation is pq-crystals' CC0 assembly `f1600x4.S`
//! (instruction for instruction; only labels and symbol visibility differ),
//! which keeps its register and instruction schedule. The retained review
//! implementation is an instruction-by-instruction intrinsics translation of
//! the same file (CC0; derived from Bas Westerbaan's 4-way SHAKE and
//! XKCP, Keccak designed by Guido Bertoni, Joan Daemen, Michael Peeters and
//! Gilles Van Assche): the variables `y0..y15` are its YMM registers, the state
//! stays in memory between rows, rotations are shift/shift/or with `vpshufb`
//! for the byte rotations by 8 and 56, and chi uses `vpandn`. The extraction
//! transposes 4x4 word tiles with `unpack` and `permute2x128`.

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

/// Expands bits 0..4 of `mask` to all-ones 64-bit lanes.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
fn lane_mask(mask: u8) -> __m256i {
    let bits = _mm256_setr_epi64x(1, 2, 4, 8);
    _mm256_cmpeq_epi64(
        _mm256_and_si256(_mm256_set1_epi64x(mask as i64), bits),
        bits,
    )
}

/// Transposes four state words into one 32-byte row of each of the four blocks.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
fn transpose_store<const TILE: usize>(
    a: __m256i,
    b: __m256i,
    c: __m256i,
    d: __m256i,
    blocks: &mut [Block; 4],
) {
    let t0 = _mm256_unpacklo_epi64(a, b);
    let t1 = _mm256_unpackhi_epi64(a, b);
    let t2 = _mm256_unpacklo_epi64(c, d);
    let t3 = _mm256_unpackhi_epi64(c, d);
    // SAFETY: TILE * 32 + 32 <= 160 <= 256 bytes of each block.
    unsafe {
        _mm256_storeu_si256(
            blocks[0].bytes.as_mut_ptr().add(TILE * 32).cast(),
            _mm256_permute2x128_si256::<0x20>(t0, t2),
        );
        _mm256_storeu_si256(
            blocks[1].bytes.as_mut_ptr().add(TILE * 32).cast(),
            _mm256_permute2x128_si256::<0x20>(t1, t3),
        );
        _mm256_storeu_si256(
            blocks[2].bytes.as_mut_ptr().add(TILE * 32).cast(),
            _mm256_permute2x128_si256::<0x31>(t0, t2),
        );
        _mm256_storeu_si256(
            blocks[3].bytes.as_mut_ptr().add(TILE * 32).cast(),
            _mm256_permute2x128_si256::<0x31>(t1, t3),
        );
    }
}

/// Writes bytes 0..192 of every block, zeroing bytes 168..192; words 17..20 only for the `wide` lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) fn extract(state: &State<4>, blocks: &mut [Block; 4], wide: u8) {
    let s = &state.0;
    transpose_store::<0>(load(&s[0]), load(&s[1]), load(&s[2]), load(&s[3]), blocks);
    transpose_store::<1>(load(&s[4]), load(&s[5]), load(&s[6]), load(&s[7]), blocks);
    transpose_store::<2>(load(&s[8]), load(&s[9]), load(&s[10]), load(&s[11]), blocks);
    transpose_store::<3>(
        load(&s[12]),
        load(&s[13]),
        load(&s[14]),
        load(&s[15]),
        blocks,
    );
    let (w17, w18, w19, w20) = if wide == 0b1111 {
        (load(&s[17]), load(&s[18]), load(&s[19]), load(&s[20]))
    } else {
        let keep = lane_mask(wide);
        (
            _mm256_and_si256(load(&s[17]), keep),
            _mm256_and_si256(load(&s[18]), keep),
            _mm256_and_si256(load(&s[19]), keep),
            _mm256_and_si256(load(&s[20]), keep),
        )
    };
    transpose_store::<4>(load(&s[16]), w17, w18, w19, blocks);
    // Word 20 lane by lane: one 64-bit store per block instead of a tile.
    let lo = _mm256_castsi256_si128(w20);
    let hi = _mm256_extracti128_si256::<1>(w20);
    let lo_pd = _mm_castsi128_pd(lo);
    let hi_pd = _mm_castsi128_pd(hi);
    // SAFETY: Stores bytes 160..168 within each 256-byte block.
    unsafe {
        _mm_storel_pd(blocks[0].bytes.as_mut_ptr().add(160).cast(), lo_pd);
        _mm_storeh_pd(blocks[1].bytes.as_mut_ptr().add(160).cast(), lo_pd);
        _mm_storel_pd(blocks[2].bytes.as_mut_ptr().add(160).cast(), hi_pd);
        _mm_storeh_pd(blocks[3].bytes.as_mut_ptr().add(160).cast(), hi_pd);
    }
}

/// Writes the 168 rate bytes of lane `i` to `out[i][off..off + 168]` (all
/// four lanes SHAKE128), so consecutive blocks of a lane end up contiguous.
///
/// # Panics
/// If `off + 168 > N`.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn extract_at<const N: usize>(state: &State<4>, out: &mut [[u8; N]; 4], off: usize) {
    assert!(off + 168 <= N);
    let s = &state.0;
    let [o0, o1, o2, o3] = out;
    let rows = [
        o0.as_mut_ptr(),
        o1.as_mut_ptr(),
        o2.as_mut_ptr(),
        o3.as_mut_ptr(),
    ];
    for t in 0..5 {
        let (a, b, c, d) = (
            load(&s[4 * t]),
            load(&s[4 * t + 1]),
            load(&s[4 * t + 2]),
            load(&s[4 * t + 3]),
        );
        let t0 = _mm256_unpacklo_epi64(a, b);
        let t1 = _mm256_unpackhi_epi64(a, b);
        let t2 = _mm256_unpacklo_epi64(c, d);
        let t3 = _mm256_unpackhi_epi64(c, d);
        let at = off + 32 * t;
        // SAFETY: at + 32 <= off + 160 <= N bytes of each row.
        unsafe {
            _mm256_storeu_si256(
                rows[0].add(at).cast(),
                _mm256_permute2x128_si256::<0x20>(t0, t2),
            );
            _mm256_storeu_si256(
                rows[1].add(at).cast(),
                _mm256_permute2x128_si256::<0x20>(t1, t3),
            );
            _mm256_storeu_si256(
                rows[2].add(at).cast(),
                _mm256_permute2x128_si256::<0x31>(t0, t2),
            );
            _mm256_storeu_si256(
                rows[3].add(at).cast(),
                _mm256_permute2x128_si256::<0x31>(t1, t3),
            );
        }
    }
    let w20 = load(&s[20]);
    let lo = _mm_castsi128_pd(_mm256_castsi256_si128(w20));
    let hi = _mm_castsi128_pd(_mm256_extracti128_si256::<1>(w20));
    // SAFETY: bytes off + 160 .. off + 168 <= N of each row.
    unsafe {
        _mm_storel_pd(rows[0].add(off + 160).cast(), lo);
        _mm_storeh_pd(rows[1].add(off + 160).cast(), lo);
        _mm_storel_pd(rows[2].add(off + 160).cast(), hi);
        _mm_storeh_pd(rows[3].add(off + 160).cast(), hi);
    }
}

/// Zeroes the columns of `lanes`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) fn clear(state: &mut State<4>, lanes: u8) {
    let keep = core::hint::black_box(lane_mask(!lanes));
    for word in &mut state.0 {
        store(word, _mm256_and_si256(load(word), keep));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) fn absorb(
    lanes: &mut [lanes::Lane; 4],
    jobs: &[XofJob<'_>],
    state: &mut State<4>,
    mask: u8,
) -> u8 {
    lanes::absorb_with(lanes, jobs, mask, |i, w, value| {
        state.0[w][i] ^= value;
    })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn run_xof<S: Sink>(jobs: &[XofJob<'_>], sink: &mut S) {
    lanes::run::<4, false, _, _>(
        jobs,
        sink,
        Kernel {
            // Closures inherit the enclosing target features.
            permute: |state: &mut State<4>| permute(state),
            extract: |state: &State<4>, blocks: &mut [Block; 4], wide| extract(state, blocks, wide),
            clear: |state: &mut State<4>, lanes| clear(state, lanes),
            absorb: |lanes: &mut [lanes::Lane; 4],
                     jobs: &[XofJob<'_>],
                     state: &mut State<4>,
                     mask| absorb(lanes, jobs, state, mask),
        },
    );
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) unsafe fn run_xof<S: Sink>(_: &[XofJob<'_>], _: &mut S) {
    unreachable!("SIMD backend cannot be constructed on this architecture");
}

#[cfg(target_arch = "x86_64")]
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

/// Compiler barrier on the state: the stores before it reach memory and the
/// loads after it re-read memory, so the row schedule stays the in-memory one
/// of the assembly original instead of a register-promoted copy with spills.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn fence(state: &mut State<4>) {
    core::hint::black_box(state);
}

// Keep the state in its final interleaved storage between rows. The explicit
// load/store schedule bounds the live vector set to sixteen YMM registers.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(dead_code)]
pub(crate) fn permute_intrinsics(state: &mut State<4>) {
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
        y8 = _mm256_set1_epi64x(RC[round] as i64);
        y13 = _mm256_xor_si256(y13, y8);
        store(&mut state.0[0], y13);
        store(&mut state.0[6], y14);
        store(&mut state.0[12], y15);
        store(&mut state.0[18], y7);
        store(&mut state.0[24], y6);
        fence(state);
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
        fence(state);
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
        fence(state);
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
        fence(state);
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
        fence(state);
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
        fence(state);
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
        fence(state);
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
        fence(state);
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
        fence(state);
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
        fence(state);
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
        fence(state);
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
        fence(state);
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
        fence(state);
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
        fence(state);
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
        fence(state);
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
        fence(state);
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
        fence(state);
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
        fence(state);
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
        fence(state);
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
        fence(state);
    }
}

// The intrinsics translation above remains available for review and differential
// tests. The CC0 assembly bounds register pressure and fixes instruction order.
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
core::arch::global_asm!(include_str!("avx2_permute.S"), options(att_syntax));

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
unsafe extern "C" {
    fn agg_mldsa_keccak_f1600x4(state: *mut u64, rc: *const u64);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) fn permute(state: &mut State<4>) {
    // SAFETY: State contains exactly 100 contiguous u64 words, aligned to 64
    // bytes; the kernel reads all 24 constants and accesses only those words.
    // The caller has detected AVX2.
    #[cfg(target_os = "linux")]
    unsafe {
        agg_mldsa_keccak_f1600x4(state.0.as_mut_ptr().cast(), RC.as_ptr())
    }
    #[cfg(not(target_os = "linux"))]
    permute_intrinsics(state);
}
