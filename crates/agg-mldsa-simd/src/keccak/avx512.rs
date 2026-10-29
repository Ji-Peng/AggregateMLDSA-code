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
            let t = rotate::<1>(c1);
            $a0 = _mm512_ternarylogic_epi64::<0x96>($a0, c4, t);
            $a5 = _mm512_ternarylogic_epi64::<0x96>($a5, c4, t);
            $a10 = _mm512_ternarylogic_epi64::<0x96>($a10, c4, t);
            $a15 = _mm512_ternarylogic_epi64::<0x96>($a15, c4, t);
            $a20 = _mm512_ternarylogic_epi64::<0x96>($a20, c4, t);
            let t = rotate::<1>(c2);
            $a1 = _mm512_ternarylogic_epi64::<0x96>($a1, c0, t);
            $a6 = _mm512_ternarylogic_epi64::<0x96>($a6, c0, t);
            $a11 = _mm512_ternarylogic_epi64::<0x96>($a11, c0, t);
            $a16 = _mm512_ternarylogic_epi64::<0x96>($a16, c0, t);
            $a21 = _mm512_ternarylogic_epi64::<0x96>($a21, c0, t);
            let t = rotate::<1>(c3);
            $a2 = _mm512_ternarylogic_epi64::<0x96>($a2, c1, t);
            $a7 = _mm512_ternarylogic_epi64::<0x96>($a7, c1, t);
            $a12 = _mm512_ternarylogic_epi64::<0x96>($a12, c1, t);
            $a17 = _mm512_ternarylogic_epi64::<0x96>($a17, c1, t);
            $a22 = _mm512_ternarylogic_epi64::<0x96>($a22, c1, t);
            let t = rotate::<1>(c4);
            $a3 = _mm512_ternarylogic_epi64::<0x96>($a3, c2, t);
            $a8 = _mm512_ternarylogic_epi64::<0x96>($a8, c2, t);
            $a13 = _mm512_ternarylogic_epi64::<0x96>($a13, c2, t);
            $a18 = _mm512_ternarylogic_epi64::<0x96>($a18, c2, t);
            $a23 = _mm512_ternarylogic_epi64::<0x96>($a23, c2, t);
            let t = rotate::<1>(c0);
            $a4 = _mm512_ternarylogic_epi64::<0x96>($a4, c3, t);
            $a9 = _mm512_ternarylogic_epi64::<0x96>($a9, c3, t);
            $a14 = _mm512_ternarylogic_epi64::<0x96>($a14, c3, t);
            $a19 = _mm512_ternarylogic_epi64::<0x96>($a19, c3, t);
            $a24 = _mm512_ternarylogic_epi64::<0x96>($a24, c3, t);
            $a6 = rotate::<44>($a6);
            $a12 = rotate::<43>($a12);
            $a18 = rotate::<21>($a18);
            $a24 = rotate::<14>($a24);
            let t0 = $a0;
            let t1 = $a6;
            $a0 = _mm512_ternarylogic_epi64::<0xd2>($a0, $a6, $a12);
            $a6 = _mm512_ternarylogic_epi64::<0xd2>($a6, $a12, $a18);
            $a12 = _mm512_ternarylogic_epi64::<0xd2>($a12, $a18, $a24);
            $a18 = _mm512_ternarylogic_epi64::<0xd2>($a18, $a24, t0);
            $a24 = _mm512_ternarylogic_epi64::<0xd2>($a24, t0, t1);
            $a0 = _mm512_xor_si512($a0, _mm512_set1_epi64($rc as i64));
            $a3 = rotate::<28>($a3);
            $a9 = rotate::<20>($a9);
            $a10 = rotate::<3>($a10);
            $a16 = rotate::<45>($a16);
            $a22 = rotate::<61>($a22);
            let t0 = $a3;
            let t1 = $a9;
            $a3 = _mm512_ternarylogic_epi64::<0xd2>($a3, $a9, $a10);
            $a9 = _mm512_ternarylogic_epi64::<0xd2>($a9, $a10, $a16);
            $a10 = _mm512_ternarylogic_epi64::<0xd2>($a10, $a16, $a22);
            $a16 = _mm512_ternarylogic_epi64::<0xd2>($a16, $a22, t0);
            $a22 = _mm512_ternarylogic_epi64::<0xd2>($a22, t0, t1);
            $a1 = rotate::<1>($a1);
            $a7 = rotate::<6>($a7);
            $a13 = rotate::<25>($a13);
            $a19 = rotate::<8>($a19);
            $a20 = rotate::<18>($a20);
            let t0 = $a1;
            let t1 = $a7;
            $a1 = _mm512_ternarylogic_epi64::<0xd2>($a1, $a7, $a13);
            $a7 = _mm512_ternarylogic_epi64::<0xd2>($a7, $a13, $a19);
            $a13 = _mm512_ternarylogic_epi64::<0xd2>($a13, $a19, $a20);
            $a19 = _mm512_ternarylogic_epi64::<0xd2>($a19, $a20, t0);
            $a20 = _mm512_ternarylogic_epi64::<0xd2>($a20, t0, t1);
            $a4 = rotate::<27>($a4);
            $a5 = rotate::<36>($a5);
            $a11 = rotate::<10>($a11);
            $a17 = rotate::<15>($a17);
            $a23 = rotate::<56>($a23);
            let t0 = $a4;
            let t1 = $a5;
            $a4 = _mm512_ternarylogic_epi64::<0xd2>($a4, $a5, $a11);
            $a5 = _mm512_ternarylogic_epi64::<0xd2>($a5, $a11, $a17);
            $a11 = _mm512_ternarylogic_epi64::<0xd2>($a11, $a17, $a23);
            $a17 = _mm512_ternarylogic_epi64::<0xd2>($a17, $a23, t0);
            $a23 = _mm512_ternarylogic_epi64::<0xd2>($a23, t0, t1);
            $a2 = rotate::<62>($a2);
            $a8 = rotate::<55>($a8);
            $a14 = rotate::<39>($a14);
            $a15 = rotate::<41>($a15);
            $a21 = rotate::<2>($a21);
            let t0 = $a2;
            let t1 = $a8;
            $a2 = _mm512_ternarylogic_epi64::<0xd2>($a2, $a8, $a14);
            $a8 = _mm512_ternarylogic_epi64::<0xd2>($a8, $a14, $a15);
            $a14 = _mm512_ternarylogic_epi64::<0xd2>($a14, $a15, $a21);
            $a15 = _mm512_ternarylogic_epi64::<0xd2>($a15, $a21, t0);
            $a21 = _mm512_ternarylogic_epi64::<0xd2>($a21, t0, t1);
        }};
    }
    round!(0x0000000000000001u64; a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24);
    round!(0x0000000000008082u64; a0, a6, a12, a18, a24, a3, a9, a10, a16, a22, a1, a7, a13, a19, a20, a4, a5, a11, a17, a23, a2, a8, a14, a15, a21);
    round!(0x800000000000808au64; a0, a9, a13, a17, a21, a18, a22, a1, a5, a14, a6, a10, a19, a23, a2, a24, a3, a7, a11, a15, a12, a16, a20, a4, a8);
    round!(0x8000000080008000u64; a0, a22, a19, a11, a8, a17, a14, a6, a3, a20, a9, a1, a23, a15, a12, a21, a18, a10, a7, a4, a13, a5, a2, a24, a16);
    round!(0x000000000000808bu64; a0, a14, a23, a7, a16, a11, a20, a9, a18, a2, a22, a6, a15, a4, a13, a8, a17, a1, a10, a24, a19, a3, a12, a21, a5);
    round!(0x0000000080000001u64; a0, a20, a15, a10, a5, a7, a2, a22, a17, a12, a14, a9, a4, a24, a19, a16, a11, a6, a1, a21, a23, a18, a13, a8, a3);
    round!(0x8000000080008081u64; a0, a2, a4, a1, a3, a10, a12, a14, a11, a13, a20, a22, a24, a21, a23, a5, a7, a9, a6, a8, a15, a17, a19, a16, a18);
    round!(0x8000000000008009u64; a0, a12, a24, a6, a18, a1, a13, a20, a7, a19, a2, a14, a21, a8, a15, a3, a10, a22, a9, a16, a4, a11, a23, a5, a17);
    round!(0x000000000000008au64; a0, a13, a21, a9, a17, a6, a19, a2, a10, a23, a12, a20, a8, a16, a4, a18, a1, a14, a22, a5, a24, a7, a15, a3, a11);
    round!(0x0000000000000088u64; a0, a19, a8, a22, a11, a9, a23, a12, a1, a15, a13, a2, a16, a5, a24, a17, a6, a20, a14, a3, a21, a10, a4, a18, a7);
    round!(0x0000000080008009u64; a0, a23, a16, a14, a7, a22, a15, a13, a6, a4, a19, a12, a5, a3, a21, a11, a9, a2, a20, a18, a8, a1, a24, a17, a10);
    round!(0x000000008000000au64; a0, a15, a5, a20, a10, a14, a4, a19, a9, a24, a23, a13, a3, a18, a8, a7, a22, a12, a2, a17, a16, a6, a21, a11, a1);
    round!(0x000000008000808bu64; a0, a4, a3, a2, a1, a20, a24, a23, a22, a21, a15, a19, a18, a17, a16, a10, a14, a13, a12, a11, a5, a9, a8, a7, a6);
    round!(0x800000000000008bu64; a0, a24, a18, a12, a6, a2, a21, a15, a14, a8, a4, a23, a17, a11, a5, a1, a20, a19, a13, a7, a3, a22, a16, a10, a9);
    round!(0x8000000000008089u64; a0, a21, a17, a13, a9, a12, a8, a4, a20, a16, a24, a15, a11, a7, a3, a6, a2, a23, a19, a10, a18, a14, a5, a1, a22);
    round!(0x8000000000008003u64; a0, a8, a11, a19, a22, a13, a16, a24, a2, a5, a21, a4, a7, a10, a18, a9, a12, a15, a23, a1, a17, a20, a3, a6, a14);
    round!(0x8000000000008002u64; a0, a16, a7, a23, a14, a19, a5, a21, a12, a3, a8, a24, a10, a1, a17, a22, a13, a4, a15, a6, a11, a2, a18, a9, a20);
    round!(0x8000000000000080u64; a0, a5, a10, a15, a20, a23, a3, a8, a13, a18, a16, a21, a1, a6, a11, a14, a19, a24, a4, a9, a7, a12, a17, a22, a2);
    round!(0x000000000000800au64; a0, a3, a1, a4, a2, a15, a18, a16, a19, a17, a5, a8, a6, a9, a7, a20, a23, a21, a24, a22, a10, a13, a11, a14, a12);
    round!(0x800000008000000au64; a0, a18, a6, a24, a12, a4, a17, a5, a23, a11, a3, a16, a9, a22, a10, a2, a15, a8, a21, a14, a1, a19, a7, a20, a13);
    round!(0x8000000080008081u64; a0, a17, a9, a21, a13, a24, a11, a3, a15, a7, a18, a5, a22, a14, a1, a12, a4, a16, a8, a20, a6, a23, a10, a2, a19);
    round!(0x8000000000008080u64; a0, a11, a22, a8, a19, a21, a7, a18, a4, a10, a17, a3, a14, a20, a6, a13, a24, a5, a16, a2, a9, a15, a1, a12, a23);
    round!(0x0000000080000001u64; a0, a7, a14, a16, a23, a8, a10, a17, a24, a1, a11, a18, a20, a2, a9, a19, a21, a3, a5, a12, a22, a4, a6, a13, a15);
    round!(0x8000000080008008u64; a0, a10, a20, a5, a15, a16, a1, a11, a21, a6, a7, a17, a2, a12, a22, a23, a8, a18, a3, a13, a14, a24, a9, a19, a4);
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
        },
    );
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) unsafe fn run_xof<S: Sink>(_: &[XofJob<'_>], _: &mut S) {
    unreachable!("SIMD backend cannot be constructed on this architecture");
}
