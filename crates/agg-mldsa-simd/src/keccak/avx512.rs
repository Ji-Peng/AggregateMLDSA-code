//! 8-way Keccak-f[1600] and block extraction (AVX-512F).
//!
//! The permutation uses ZMM vectors for the 25 words and follows the
//! structure of XKCP's `KeccakP-1600-times8-SIMD512.c` (CC0; implementer
//! Ronny Van Keer, Keccak designed by Guido Bertoni, Joan Daemen, Michael
//! Peeters and Gilles Van Assche): `vprolq` rotations, `vpternlogq` 0x96 for
//! the column parities and 0xD2 for chi, and the in-place lane renaming that
//! repeats every four rounds. Theta folds `D[x] = C[x-1] ^ rol(C[x+1], 1)` into
//! each lane with one three-input XOR, as in Xurong Qi's Hybrid-SHA-3
//! `keccak8x.c`. The project owner authorized direct Rust translation on
//! 2026-10-29, confirming permission from the Hybrid-SHA-3 authors. The scope
//! includes the permutation and squeeze techniques attributed here; the
//! original permission correspondence is not reproduced here. Columns 2 to 4 apply
//! it just before their lanes enter chi, which interleaves the rotations with
//! the logic operations. Extraction uses 8x8 word transposes built from
//! unpack and two-input shuffles, with zero vectors for the last tile's padding.

use super::{
    lanes::{self, Kernel, State},
    Sink, XofJob,
};
use crate::poly::Block;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
fn load(word: &[u64; 8]) -> __m512i {
    // SAFETY: The complete vector is contained in the supplied fixed-size array.
    unsafe { _mm512_loadu_si512(word.as_ptr().cast()) }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
fn store(word: &mut [u64; 8], value: __m512i) {
    // SAFETY: The complete vector is contained in the supplied fixed-size array.
    unsafe { _mm512_storeu_si512(word.as_mut_ptr().cast(), value) }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
fn rotate<const N: i32>(x: __m512i) -> __m512i {
    _mm512_rol_epi64::<N>(x)
}

/// Keccak-f[1600] on eight word-major states. Caller must have detected `avx512f`.
///
/// Per round: 10 three-input XORs for the column parities, 25 for theta, 29
/// rotations, 25 chi operations and the iota XOR. The rounds run as a loop of
/// four (the lane renaming repeats every four rounds), which keeps the body in
/// the decoded-instruction cache. A fully unrolled body does not fit there; on
/// Rocket Lake it measured 7-12% slower than this loop.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) fn permute(state: &mut State<8>) {
    // A 64-byte aligned local makes LLVM realign the frame, which aligns the
    // loop's register spill slots: an unaligned 64-byte slot that straddles a
    // 4 KiB page boundary makes every spill a page split.
    #[repr(align(64))]
    struct Line(u64);
    let mut line = Line(0);
    core::hint::black_box(&mut line.0);
    let mut a0 = load(&state.0[0]);
    let mut a1 = load(&state.0[1]);
    let mut a2 = load(&state.0[2]);
    let mut a3 = load(&state.0[3]);
    let mut a4 = load(&state.0[4]);
    let mut a5 = load(&state.0[5]);
    let mut a6 = load(&state.0[6]);
    let mut a7 = load(&state.0[7]);
    let mut a8 = load(&state.0[8]);
    let mut a9 = load(&state.0[9]);
    let mut a10 = load(&state.0[10]);
    let mut a11 = load(&state.0[11]);
    let mut a12 = load(&state.0[12]);
    let mut a13 = load(&state.0[13]);
    let mut a14 = load(&state.0[14]);
    let mut a15 = load(&state.0[15]);
    let mut a16 = load(&state.0[16]);
    let mut a17 = load(&state.0[17]);
    let mut a18 = load(&state.0[18]);
    let mut a19 = load(&state.0[19]);
    let mut a20 = load(&state.0[20]);
    let mut a21 = load(&state.0[21]);
    let mut a22 = load(&state.0[22]);
    let mut a23 = load(&state.0[23]);
    let mut a24 = load(&state.0[24]);
    macro_rules! round {
        ($rc:expr; $a0:ident, $a1:ident, $a2:ident, $a3:ident, $a4:ident, $a5:ident, $a6:ident, $a7:ident, $a8:ident, $a9:ident, $a10:ident, $a11:ident, $a12:ident, $a13:ident, $a14:ident, $a15:ident, $a16:ident, $a17:ident, $a18:ident, $a19:ident, $a20:ident, $a21:ident, $a22:ident, $a23:ident, $a24:ident) => {{
            let c0 = _mm512_ternarylogic_epi64::<0x96>(
                _mm512_ternarylogic_epi64::<0x96>($a0, $a5, $a10),
                $a15,
                $a20,
            );
            let c1 = _mm512_ternarylogic_epi64::<0x96>(
                _mm512_ternarylogic_epi64::<0x96>($a1, $a6, $a11),
                $a16,
                $a21,
            );
            let c2 = _mm512_ternarylogic_epi64::<0x96>(
                _mm512_ternarylogic_epi64::<0x96>($a2, $a7, $a12),
                $a17,
                $a22,
            );
            let c3 = _mm512_ternarylogic_epi64::<0x96>(
                _mm512_ternarylogic_epi64::<0x96>($a3, $a8, $a13),
                $a18,
                $a23,
            );
            let c4 = _mm512_ternarylogic_epi64::<0x96>(
                _mm512_ternarylogic_epi64::<0x96>($a4, $a9, $a14),
                $a19,
                $a24,
            );
            // Theta for columns 0 and 1; columns 2 to 4 are deferred to their rows so
            // rotations and three-input operations interleave (D[x] = C[x-1] ^ rol(C[x+1], 1)).
            let r1 = rotate::<1>(c1);
            $a0 = _mm512_ternarylogic_epi64::<0x96>($a0, c4, r1);
            $a5 = _mm512_ternarylogic_epi64::<0x96>($a5, c4, r1);
            $a10 = _mm512_ternarylogic_epi64::<0x96>($a10, c4, r1);
            $a15 = _mm512_ternarylogic_epi64::<0x96>($a15, c4, r1);
            $a20 = _mm512_ternarylogic_epi64::<0x96>($a20, c4, r1);
            let r2 = rotate::<1>(c2);
            $a1 = _mm512_ternarylogic_epi64::<0x96>($a1, c0, r2);
            $a6 = _mm512_ternarylogic_epi64::<0x96>($a6, c0, r2);
            $a11 = _mm512_ternarylogic_epi64::<0x96>($a11, c0, r2);
            $a16 = _mm512_ternarylogic_epi64::<0x96>($a16, c0, r2);
            $a21 = _mm512_ternarylogic_epi64::<0x96>($a21, c0, r2);
            // Output row 0: rho, pi, chi (and iota on lane 0).
            let b0 = $a0;
            let b1 = rotate::<44>($a6);
            let r3 = rotate::<1>(c3);
            $a12 = _mm512_ternarylogic_epi64::<0x96>($a12, c1, r3);
            let b2 = rotate::<43>($a12);
            let r4 = rotate::<1>(c4);
            $a18 = _mm512_ternarylogic_epi64::<0x96>($a18, c2, r4);
            let b3 = rotate::<21>($a18);
            let r0 = rotate::<1>(c0);
            $a24 = _mm512_ternarylogic_epi64::<0x96>($a24, c3, r0);
            let b4 = rotate::<14>($a24);
            $a0 = _mm512_ternarylogic_epi64::<0xd2>(b0, b1, b2);
            $a0 = _mm512_xor_si512($a0, _mm512_set1_epi64($rc as i64));
            $a6 = _mm512_ternarylogic_epi64::<0xd2>(b1, b2, b3);
            $a12 = _mm512_ternarylogic_epi64::<0xd2>(b2, b3, b4);
            $a18 = _mm512_ternarylogic_epi64::<0xd2>(b3, b4, b0);
            $a24 = _mm512_ternarylogic_epi64::<0xd2>(b4, b0, b1);
            // Output row 1: rho, pi, chi (and iota on lane 0).
            $a3 = _mm512_ternarylogic_epi64::<0x96>($a3, c2, r4);
            let b0 = rotate::<28>($a3);
            $a9 = _mm512_ternarylogic_epi64::<0x96>($a9, c3, r0);
            let b1 = rotate::<20>($a9);
            let b2 = rotate::<3>($a10);
            let b3 = rotate::<45>($a16);
            $a22 = _mm512_ternarylogic_epi64::<0x96>($a22, c1, r3);
            let b4 = rotate::<61>($a22);
            $a3 = _mm512_ternarylogic_epi64::<0xd2>(b0, b1, b2);
            $a9 = _mm512_ternarylogic_epi64::<0xd2>(b1, b2, b3);
            $a10 = _mm512_ternarylogic_epi64::<0xd2>(b2, b3, b4);
            $a16 = _mm512_ternarylogic_epi64::<0xd2>(b3, b4, b0);
            $a22 = _mm512_ternarylogic_epi64::<0xd2>(b4, b0, b1);
            // Output row 2: rho, pi, chi (and iota on lane 0).
            let b0 = rotate::<1>($a1);
            $a7 = _mm512_ternarylogic_epi64::<0x96>($a7, c1, r3);
            let b1 = rotate::<6>($a7);
            $a13 = _mm512_ternarylogic_epi64::<0x96>($a13, c2, r4);
            let b2 = rotate::<25>($a13);
            $a19 = _mm512_ternarylogic_epi64::<0x96>($a19, c3, r0);
            let b3 = rotate::<8>($a19);
            let b4 = rotate::<18>($a20);
            $a1 = _mm512_ternarylogic_epi64::<0xd2>(b0, b1, b2);
            $a7 = _mm512_ternarylogic_epi64::<0xd2>(b1, b2, b3);
            $a13 = _mm512_ternarylogic_epi64::<0xd2>(b2, b3, b4);
            $a19 = _mm512_ternarylogic_epi64::<0xd2>(b3, b4, b0);
            $a20 = _mm512_ternarylogic_epi64::<0xd2>(b4, b0, b1);
            // Output row 3: rho, pi, chi (and iota on lane 0).
            $a4 = _mm512_ternarylogic_epi64::<0x96>($a4, c3, r0);
            let b0 = rotate::<27>($a4);
            let b1 = rotate::<36>($a5);
            let b2 = rotate::<10>($a11);
            $a17 = _mm512_ternarylogic_epi64::<0x96>($a17, c1, r3);
            let b3 = rotate::<15>($a17);
            $a23 = _mm512_ternarylogic_epi64::<0x96>($a23, c2, r4);
            let b4 = rotate::<56>($a23);
            $a4 = _mm512_ternarylogic_epi64::<0xd2>(b0, b1, b2);
            $a5 = _mm512_ternarylogic_epi64::<0xd2>(b1, b2, b3);
            $a11 = _mm512_ternarylogic_epi64::<0xd2>(b2, b3, b4);
            $a17 = _mm512_ternarylogic_epi64::<0xd2>(b3, b4, b0);
            $a23 = _mm512_ternarylogic_epi64::<0xd2>(b4, b0, b1);
            // Output row 4: rho, pi, chi (and iota on lane 0).
            $a2 = _mm512_ternarylogic_epi64::<0x96>($a2, c1, r3);
            let b0 = rotate::<62>($a2);
            $a8 = _mm512_ternarylogic_epi64::<0x96>($a8, c2, r4);
            let b1 = rotate::<55>($a8);
            $a14 = _mm512_ternarylogic_epi64::<0x96>($a14, c3, r0);
            let b2 = rotate::<39>($a14);
            let b3 = rotate::<41>($a15);
            let b4 = rotate::<2>($a21);
            $a2 = _mm512_ternarylogic_epi64::<0xd2>(b0, b1, b2);
            $a8 = _mm512_ternarylogic_epi64::<0xd2>(b1, b2, b3);
            $a14 = _mm512_ternarylogic_epi64::<0xd2>(b2, b3, b4);
            $a15 = _mm512_ternarylogic_epi64::<0xd2>(b3, b4, b0);
            $a21 = _mm512_ternarylogic_epi64::<0xd2>(b4, b0, b1);
        }};
    }
    const RC: [u64; 24] = [
        0x0000000000000001u64,
        0x0000000000008082u64,
        0x800000000000808au64,
        0x8000000080008000u64,
        0x000000000000808bu64,
        0x0000000080000001u64,
        0x8000000080008081u64,
        0x8000000000008009u64,
        0x000000000000008au64,
        0x0000000000000088u64,
        0x0000000080008009u64,
        0x000000008000000au64,
        0x000000008000808bu64,
        0x800000000000008bu64,
        0x8000000000008089u64,
        0x8000000000008003u64,
        0x8000000000008002u64,
        0x8000000000000080u64,
        0x000000000000800au64,
        0x800000008000000au64,
        0x8000000080008081u64,
        0x8000000000008080u64,
        0x0000000080000001u64,
        0x8000000080008008u64,
    ];
    // After four rounds every lane sits at a fixed new position; the tuple
    // assignment at the end of the body restores the names (a few register
    // moves per round, far cheaper than decoding a fully unrolled body).
    macro_rules! four_rounds {
        ($r:expr) => {{
            let r = $r;

            round!(RC[r + 0]; a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24);
            round!(RC[r + 1]; a0, a6, a12, a18, a24, a3, a9, a10, a16, a22, a1, a7, a13, a19, a20, a4, a5, a11, a17, a23, a2, a8, a14, a15, a21);
            round!(RC[r + 2]; a0, a9, a13, a17, a21, a18, a22, a1, a5, a14, a6, a10, a19, a23, a2, a24, a3, a7, a11, a15, a12, a16, a20, a4, a8);
            round!(RC[r + 3]; a0, a22, a19, a11, a8, a17, a14, a6, a3, a20, a9, a1, a23, a15, a12, a21, a18, a10, a7, a4, a13, a5, a2, a24, a16);
            (
                a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18,
                a19, a20, a21, a22, a23, a24,
            ) = (
                a0, a14, a23, a7, a16, a11, a20, a9, a18, a2, a22, a6, a15, a4, a13, a8, a17, a1, a10,
                a24, a19, a3, a12, a21, a5,
            );
        }};
    }
    // An opaque bound keeps the loop rolled.
    let rounds = core::hint::black_box(24usize);
    let mut r = 0;
    while r < rounds && r <= 20 {
        four_rounds!(r);
        r += 4;
    }
    store(&mut state.0[0], a0);
    store(&mut state.0[1], a1);
    store(&mut state.0[2], a2);
    store(&mut state.0[3], a3);
    store(&mut state.0[4], a4);
    store(&mut state.0[5], a5);
    store(&mut state.0[6], a6);
    store(&mut state.0[7], a7);
    store(&mut state.0[8], a8);
    store(&mut state.0[9], a9);
    store(&mut state.0[10], a10);
    store(&mut state.0[11], a11);
    store(&mut state.0[12], a12);
    store(&mut state.0[13], a13);
    store(&mut state.0[14], a14);
    store(&mut state.0[15], a15);
    store(&mut state.0[16], a16);
    store(&mut state.0[17], a17);
    store(&mut state.0[18], a18);
    store(&mut state.0[19], a19);
    store(&mut state.0[20], a20);
    store(&mut state.0[21], a21);
    store(&mut state.0[22], a22);
    store(&mut state.0[23], a23);
    store(&mut state.0[24], a24);
}

/// Transpose eight words of eight lanes using only two-input shuffles.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
fn oct_store(state: &State<8>, blocks: &mut [Block; 8], tile: usize, wide: u8) {
    let w = tile * 8;
    let s = &state.0;
    let a = load(&s[w]);
    let tail = tile == 2;
    let mask = if tail { wide } else { 0xff };
    let b = _mm512_maskz_mov_epi64(mask, load(&s[w + 1]));
    let c = _mm512_maskz_mov_epi64(mask, load(&s[w + 2]));
    let d = _mm512_maskz_mov_epi64(mask, load(&s[w + 3]));
    let e = _mm512_maskz_mov_epi64(mask, load(&s[w + 4]));
    let f = if tail {
        _mm512_setzero_si512()
    } else {
        load(&s[w + 5])
    };
    let g = if tail {
        _mm512_setzero_si512()
    } else {
        load(&s[w + 6])
    };
    let h = if tail {
        _mm512_setzero_si512()
    } else {
        load(&s[w + 7])
    };
    let u0 = _mm512_unpacklo_epi64(a, b);
    let u1 = _mm512_unpackhi_epi64(a, b);
    let u2 = _mm512_unpacklo_epi64(c, d);
    let u3 = _mm512_unpackhi_epi64(c, d);
    let u4 = _mm512_unpacklo_epi64(e, f);
    let u5 = _mm512_unpackhi_epi64(e, f);
    let u6 = _mm512_unpacklo_epi64(g, h);
    let u7 = _mm512_unpackhi_epi64(g, h);
    let v0 = _mm512_shuffle_i64x2::<0x44>(u0, u2);
    let v1 = _mm512_shuffle_i64x2::<0xee>(u0, u2);
    let v2 = _mm512_shuffle_i64x2::<0x44>(u1, u3);
    let v3 = _mm512_shuffle_i64x2::<0xee>(u1, u3);
    let v4 = _mm512_shuffle_i64x2::<0x44>(u4, u6);
    let v5 = _mm512_shuffle_i64x2::<0xee>(u4, u6);
    let v6 = _mm512_shuffle_i64x2::<0x44>(u5, u7);
    let v7 = _mm512_shuffle_i64x2::<0xee>(u5, u7);
    let rows = [
        _mm512_shuffle_i64x2::<0x88>(v0, v4),
        _mm512_shuffle_i64x2::<0x88>(v2, v6),
        _mm512_shuffle_i64x2::<0xdd>(v0, v4),
        _mm512_shuffle_i64x2::<0xdd>(v2, v6),
        _mm512_shuffle_i64x2::<0x88>(v1, v5),
        _mm512_shuffle_i64x2::<0x88>(v3, v7),
        _mm512_shuffle_i64x2::<0xdd>(v1, v5),
        _mm512_shuffle_i64x2::<0xdd>(v3, v7),
    ];
    for (block, row) in blocks.iter_mut().zip(rows) {
        let dst: &mut [u8; 64] = (&mut block.bytes[tile * 64..tile * 64 + 64])
            .try_into()
            .unwrap();
        // SAFETY: The complete store is contained in the checked subarray.
        unsafe { _mm512_storeu_si512(dst.as_mut_ptr().cast(), row) };
    }
}

/// Writes bytes 0..192 of every block, zeroing bytes 168..192; words 17..20 only for the `wide` lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) fn extract(state: &State<8>, blocks: &mut [Block; 8], wide: u8) {
    oct_store(state, blocks, 0, wide);
    oct_store(state, blocks, 1, wide);
    oct_store(state, blocks, 2, wide);
}

/// Zeroes the columns of `lanes`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) fn clear(state: &mut State<8>, lanes: u8) {
    let keep = core::hint::black_box(_mm512_maskz_set1_epi64(!lanes, -1));
    for word in &mut state.0 {
        store(word, _mm512_and_si512(load(word), keep));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) fn absorb(
    lanes: &mut [lanes::Lane; 8],
    jobs: &[XofJob<'_>],
    state: &mut State<8>,
    mask: u8,
) -> u8 {
    lanes::absorb_with(lanes, jobs, mask, |i, w, value| {
        let mask = 1u8 << i;
        let word = &mut state.0[w];
        let old = load(word);
        let value = _mm512_set1_epi64(value as i64);
        store(word, _mm512_mask_xor_epi64(old, mask, old, value));
    })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn run_xof<S: Sink>(jobs: &[XofJob<'_>], sink: &mut S) {
    lanes::run::<8, false, _, _>(
        jobs,
        sink,
        Kernel {
            // Closures inherit the enclosing target features.
            permute: |state: &mut State<8>| permute(state),
            extract: |state: &State<8>, blocks: &mut [Block; 8], wide| extract(state, blocks, wide),
            clear: |state: &mut State<8>, lanes| clear(state, lanes),
            absorb: |lanes: &mut [lanes::Lane; 8],
                     jobs: &[XofJob<'_>],
                     state: &mut State<8>,
                     mask| absorb(lanes, jobs, state, mask),
        },
    );
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) unsafe fn run_xof<S: Sink>(_: &[XofJob<'_>], _: &mut S) {
    unreachable!("SIMD backend cannot be constructed on this architecture");
}
