//! 8-way Keccak-f[1600] and SHAKE extraction.
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

/// Permutes an interleaved state. Caller must have detected `avx512f`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) fn permute(state: &mut State<8>) {
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
            let d0 = _mm512_xor_si512(c4, rotate::<1>(c1));
            let d1 = _mm512_xor_si512(c0, rotate::<1>(c2));
            let d2 = _mm512_xor_si512(c1, rotate::<1>(c3));
            let d3 = _mm512_xor_si512(c2, rotate::<1>(c4));
            let d4 = _mm512_xor_si512(c3, rotate::<1>(c0));
            $a0 = _mm512_xor_si512($a0, d0);
            $a6 = _mm512_xor_si512($a6, d1);
            $a12 = _mm512_xor_si512($a12, d2);
            $a18 = _mm512_xor_si512($a18, d3);
            $a24 = _mm512_xor_si512($a24, d4);
            let t1 = rotate::<44>($a6);
            let t2 = rotate::<43>($a12);
            $a18 = rotate::<21>($a18);
            $a24 = rotate::<14>($a24);
            $a6 = _mm512_ternarylogic_epi64::<0xd2>(t1, t2, $a18);
            $a12 = _mm512_ternarylogic_epi64::<0xd2>(t2, $a18, $a24);
            $a18 = _mm512_ternarylogic_epi64::<0xd2>($a18, $a24, $a0);
            $a24 = _mm512_ternarylogic_epi64::<0xd2>($a24, $a0, t1);
            $a0 = _mm512_ternarylogic_epi64::<0xd2>($a0, t1, t2);
            $a0 = _mm512_xor_si512($a0, _mm512_set1_epi64($rc as i64));
            $a3 = _mm512_xor_si512($a3, d3);
            $a9 = _mm512_xor_si512($a9, d4);
            $a10 = _mm512_xor_si512($a10, d0);
            $a16 = _mm512_xor_si512($a16, d1);
            $a22 = _mm512_xor_si512($a22, d2);
            let t1 = rotate::<20>($a9);
            let t2 = rotate::<3>($a10);
            $a16 = rotate::<45>($a16);
            $a22 = rotate::<61>($a22);
            $a3 = rotate::<28>($a3);
            $a9 = _mm512_ternarylogic_epi64::<0xd2>(t1, t2, $a16);
            $a10 = _mm512_ternarylogic_epi64::<0xd2>(t2, $a16, $a22);
            $a16 = _mm512_ternarylogic_epi64::<0xd2>($a16, $a22, $a3);
            $a22 = _mm512_ternarylogic_epi64::<0xd2>($a22, $a3, t1);
            $a3 = _mm512_ternarylogic_epi64::<0xd2>($a3, t1, t2);
            $a1 = _mm512_xor_si512($a1, d1);
            $a7 = _mm512_xor_si512($a7, d2);
            $a13 = _mm512_xor_si512($a13, d3);
            $a19 = _mm512_xor_si512($a19, d4);
            $a20 = _mm512_xor_si512($a20, d0);
            let t1 = rotate::<6>($a7);
            let t2 = rotate::<25>($a13);
            $a19 = rotate::<8>($a19);
            $a20 = rotate::<18>($a20);
            $a1 = rotate::<1>($a1);
            $a7 = _mm512_ternarylogic_epi64::<0xd2>(t1, t2, $a19);
            $a13 = _mm512_ternarylogic_epi64::<0xd2>(t2, $a19, $a20);
            $a19 = _mm512_ternarylogic_epi64::<0xd2>($a19, $a20, $a1);
            $a20 = _mm512_ternarylogic_epi64::<0xd2>($a20, $a1, t1);
            $a1 = _mm512_ternarylogic_epi64::<0xd2>($a1, t1, t2);
            $a4 = _mm512_xor_si512($a4, d4);
            $a5 = _mm512_xor_si512($a5, d0);
            $a11 = _mm512_xor_si512($a11, d1);
            $a17 = _mm512_xor_si512($a17, d2);
            $a23 = _mm512_xor_si512($a23, d3);
            let t1 = rotate::<36>($a5);
            let t2 = rotate::<10>($a11);
            $a17 = rotate::<15>($a17);
            $a23 = rotate::<56>($a23);
            $a4 = rotate::<27>($a4);
            $a5 = _mm512_ternarylogic_epi64::<0xd2>(t1, t2, $a17);
            $a11 = _mm512_ternarylogic_epi64::<0xd2>(t2, $a17, $a23);
            $a17 = _mm512_ternarylogic_epi64::<0xd2>($a17, $a23, $a4);
            $a23 = _mm512_ternarylogic_epi64::<0xd2>($a23, $a4, t1);
            $a4 = _mm512_ternarylogic_epi64::<0xd2>($a4, t1, t2);
            $a2 = _mm512_xor_si512($a2, d2);
            $a8 = _mm512_xor_si512($a8, d3);
            $a14 = _mm512_xor_si512($a14, d4);
            $a15 = _mm512_xor_si512($a15, d0);
            $a21 = _mm512_xor_si512($a21, d1);
            let t1 = rotate::<55>($a8);
            let t2 = rotate::<39>($a14);
            $a15 = rotate::<41>($a15);
            $a21 = rotate::<2>($a21);
            $a2 = rotate::<62>($a2);
            $a8 = _mm512_ternarylogic_epi64::<0xd2>(t1, t2, $a15);
            $a14 = _mm512_ternarylogic_epi64::<0xd2>(t2, $a15, $a21);
            $a15 = _mm512_ternarylogic_epi64::<0xd2>($a15, $a21, $a2);
            $a21 = _mm512_ternarylogic_epi64::<0xd2>($a21, $a2, t1);
            $a2 = _mm512_ternarylogic_epi64::<0xd2>($a2, t1, t2);
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
    for r in (0..24).step_by(4) {
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
fn extract(state: &State<8>, blocks: &mut [Block; 8], rate: usize) {
    if rate == 168 {
        extract_rate::<168>(state, blocks);
    } else {
        extract_rate::<136>(state, blocks);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
fn extract_rate<const RATE: usize>(state: &State<8>, blocks: &mut [Block; 8]) {
    let rate = RATE;
    // Transpose whole register tiles; handle the last one to five words separately.
    let words = rate / 8;
    let tiled = words / 8 * 8;
    for base in (0..tiled).step_by(8) {
        let a0 = load(&state.0[base + 0]);
        let a1 = load(&state.0[base + 1]);
        let a2 = load(&state.0[base + 2]);
        let a3 = load(&state.0[base + 3]);
        let a4 = load(&state.0[base + 4]);
        let a5 = load(&state.0[base + 5]);
        let a6 = load(&state.0[base + 6]);
        let a7 = load(&state.0[base + 7]);
        let t0 = _mm512_unpacklo_epi64(a0, a1);
        let t1 = _mm512_unpackhi_epi64(a0, a1);
        let t2 = _mm512_unpacklo_epi64(a2, a3);
        let t3 = _mm512_unpackhi_epi64(a2, a3);
        let t4 = _mm512_unpacklo_epi64(a4, a5);
        let t5 = _mm512_unpackhi_epi64(a4, a5);
        let t6 = _mm512_unpacklo_epi64(a6, a7);
        let t7 = _mm512_unpackhi_epi64(a6, a7);
        let f0 = _mm512_shuffle_i64x2::<0x44>(t0, t2);
        let f1 = _mm512_shuffle_i64x2::<0x44>(t1, t3);
        let f2 = _mm512_shuffle_i64x2::<0xee>(t0, t2);
        let f3 = _mm512_shuffle_i64x2::<0xee>(t1, t3);
        let f4 = _mm512_shuffle_i64x2::<0x44>(t4, t6);
        let f5 = _mm512_shuffle_i64x2::<0x44>(t5, t7);
        let f6 = _mm512_shuffle_i64x2::<0xee>(t4, t6);
        let f7 = _mm512_shuffle_i64x2::<0xee>(t5, t7);
        let dst: &mut [u8; 64] = (&mut blocks[0].bytes[base * 8..(base + 8) * 8])
            .try_into()
            .unwrap();
        // SAFETY: The store fits exactly in the checked output subarray.
        unsafe {
            _mm512_storeu_si512(
                dst.as_mut_ptr().cast(),
                _mm512_shuffle_i64x2::<0x88>(f0, f4),
            )
        };
        let dst: &mut [u8; 64] = (&mut blocks[1].bytes[base * 8..(base + 8) * 8])
            .try_into()
            .unwrap();
        // SAFETY: The store fits exactly in the checked output subarray.
        unsafe {
            _mm512_storeu_si512(
                dst.as_mut_ptr().cast(),
                _mm512_shuffle_i64x2::<0x88>(f1, f5),
            )
        };
        let dst: &mut [u8; 64] = (&mut blocks[2].bytes[base * 8..(base + 8) * 8])
            .try_into()
            .unwrap();
        // SAFETY: The store fits exactly in the checked output subarray.
        unsafe {
            _mm512_storeu_si512(
                dst.as_mut_ptr().cast(),
                _mm512_shuffle_i64x2::<0xdd>(f0, f4),
            )
        };
        let dst: &mut [u8; 64] = (&mut blocks[3].bytes[base * 8..(base + 8) * 8])
            .try_into()
            .unwrap();
        // SAFETY: The store fits exactly in the checked output subarray.
        unsafe {
            _mm512_storeu_si512(
                dst.as_mut_ptr().cast(),
                _mm512_shuffle_i64x2::<0xdd>(f1, f5),
            )
        };
        let dst: &mut [u8; 64] = (&mut blocks[4].bytes[base * 8..(base + 8) * 8])
            .try_into()
            .unwrap();
        // SAFETY: The store fits exactly in the checked output subarray.
        unsafe {
            _mm512_storeu_si512(
                dst.as_mut_ptr().cast(),
                _mm512_shuffle_i64x2::<0x88>(f2, f6),
            )
        };
        let dst: &mut [u8; 64] = (&mut blocks[5].bytes[base * 8..(base + 8) * 8])
            .try_into()
            .unwrap();
        // SAFETY: The store fits exactly in the checked output subarray.
        unsafe {
            _mm512_storeu_si512(
                dst.as_mut_ptr().cast(),
                _mm512_shuffle_i64x2::<0x88>(f3, f7),
            )
        };
        let dst: &mut [u8; 64] = (&mut blocks[6].bytes[base * 8..(base + 8) * 8])
            .try_into()
            .unwrap();
        // SAFETY: The store fits exactly in the checked output subarray.
        unsafe {
            _mm512_storeu_si512(
                dst.as_mut_ptr().cast(),
                _mm512_shuffle_i64x2::<0xdd>(f2, f6),
            )
        };
        let dst: &mut [u8; 64] = (&mut blocks[7].bytes[base * 8..(base + 8) * 8])
            .try_into()
            .unwrap();
        // SAFETY: The store fits exactly in the checked output subarray.
        unsafe {
            _mm512_storeu_si512(
                dst.as_mut_ptr().cast(),
                _mm512_shuffle_i64x2::<0xdd>(f3, f7),
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
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn run_xof<S: Sink>(jobs: &[XofJob<'_>], sink: &mut S) {
    lanes::run(
        jobs,
        sink,
        Kernel {
            // Closures inherit the enclosing target features.
            permute: |state: &mut State<8>| permute(state),
            // The same target features cover extraction.
            extract: |state: &State<8>, blocks: &mut [Block; 8], rate| extract(state, blocks, rate),
            clear: |state: &mut State<8>, mask| clear(state, mask),
        },
    );
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) unsafe fn run_xof<S: Sink>(_: &[XofJob<'_>], _: &mut S) {
    unreachable!("SIMD backend cannot be constructed on this architecture");
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
fn clear(state: &mut State<8>, mask: u8) {
    for word in &mut state.0 {
        store(word, _mm512_maskz_mov_epi64(!mask, load(word)));
    }
}
