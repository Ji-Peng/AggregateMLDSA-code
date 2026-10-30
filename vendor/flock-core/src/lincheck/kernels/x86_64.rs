use core::arch::x86_64::{__m128i, _mm_loadu_si128, _mm_storeu_si128, _mm_xor_si128};

use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSlice};

use crate::lincheck::{F128, build_sum_table};

/// x86 single-matrix inner kernel — SSE2 mirror of
/// [`process_block_neon_single`]. Sweeps `TILE_T = 8` stripes for one
/// `BLOCK_K = 8` block of i_inner positions, keeping all 8 F128 accumulators in
/// xmm registers so the per-tile output is read/written once (vs once per
/// stripe in the untiled [`partial_fold_packed_z_fast_padded`] path).
///
/// # Safety
/// - `tile_bytes_ptr` must point to at least `TILE_T * k` bytes.
/// - `tables_ptr` must point to at least `TILE_T * 256 * 16` bytes.
/// - `out_ptr` must point to at least 8 F128 (128 bytes) of mutable storage.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn process_block_x86(
    tile_bytes_ptr: *const u8,
    k: usize,
    bs: usize,
    tables_ptr: *const u8,
    out_ptr: *mut F128,
) {
    const TILE_T: usize = 8;
    // SAFETY: caller upholds the pointer/length contract documented above; SSE2
    // is baseline on x86_64.
    unsafe {
        let o = out_ptr as *mut u8;
        let mut a0 = _mm_loadu_si128(o as *const __m128i);
        let mut a1 = _mm_loadu_si128(o.add(16) as *const __m128i);
        let mut a2 = _mm_loadu_si128(o.add(32) as *const __m128i);
        let mut a3 = _mm_loadu_si128(o.add(48) as *const __m128i);
        let mut a4 = _mm_loadu_si128(o.add(64) as *const __m128i);
        let mut a5 = _mm_loadu_si128(o.add(80) as *const __m128i);
        let mut a6 = _mm_loadu_si128(o.add(96) as *const __m128i);
        let mut a7 = _mm_loadu_si128(o.add(112) as *const __m128i);
        for t in 0..TILE_T {
            let stripe_ptr = tile_bytes_ptr.add(t * k + bs);
            let ta = tables_ptr.add(t * 256 * 16);
            let i0 = *stripe_ptr as usize;
            let i1 = *stripe_ptr.add(1) as usize;
            let i2 = *stripe_ptr.add(2) as usize;
            let i3 = *stripe_ptr.add(3) as usize;
            let i4 = *stripe_ptr.add(4) as usize;
            let i5 = *stripe_ptr.add(5) as usize;
            let i6 = *stripe_ptr.add(6) as usize;
            let i7 = *stripe_ptr.add(7) as usize;
            a0 = _mm_xor_si128(a0, _mm_loadu_si128(ta.add(i0 * 16) as *const __m128i));
            a1 = _mm_xor_si128(a1, _mm_loadu_si128(ta.add(i1 * 16) as *const __m128i));
            a2 = _mm_xor_si128(a2, _mm_loadu_si128(ta.add(i2 * 16) as *const __m128i));
            a3 = _mm_xor_si128(a3, _mm_loadu_si128(ta.add(i3 * 16) as *const __m128i));
            a4 = _mm_xor_si128(a4, _mm_loadu_si128(ta.add(i4 * 16) as *const __m128i));
            a5 = _mm_xor_si128(a5, _mm_loadu_si128(ta.add(i5 * 16) as *const __m128i));
            a6 = _mm_xor_si128(a6, _mm_loadu_si128(ta.add(i6 * 16) as *const __m128i));
            a7 = _mm_xor_si128(a7, _mm_loadu_si128(ta.add(i7 * 16) as *const __m128i));
        }
        _mm_storeu_si128(o as *mut __m128i, a0);
        _mm_storeu_si128(o.add(16) as *mut __m128i, a1);
        _mm_storeu_si128(o.add(32) as *mut __m128i, a2);
        _mm_storeu_si128(o.add(48) as *mut __m128i, a3);
        _mm_storeu_si128(o.add(64) as *mut __m128i, a4);
        _mm_storeu_si128(o.add(80) as *mut __m128i, a5);
        _mm_storeu_si128(o.add(96) as *mut __m128i, a6);
        _mm_storeu_si128(o.add(112) as *mut __m128i, a7);
    }
}

/// x86 tiled single-matrix partial fold — `TILE_T × BLOCK_K` register tile
/// mirror of `partial_fold_packed_z_neon_single_padded`. Replaces the untiled
/// full-k accumulator (streamed once per stripe ≈ `n_stripes × k` of acc
/// traffic) with per-tile register accumulators, cutting acc traffic ≈ `TILE_T`×
/// — the dominant cost of this bandwidth-bound fold on many-core hosts.
#[cfg(target_arch = "x86_64")]
pub fn partial_fold_packed_z_x86_tiled_padded(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    const TILE_T: usize = 8;
    const BLOCK_K: usize = 8;

    let n_log = m - k_log;
    let k = 1usize << k_log;
    let n_outer = 1usize << n_log;
    assert_eq!(z_packed.len(), (1usize << m) / 8);
    assert_eq!(eq_outer.len(), n_outer);
    assert!(n_log >= 6, "need n_outer ≥ 64 for tile of 8 stripes");
    assert!(k_log >= 3, "need k ≥ 8");
    assert!(useful_bits <= k);
    let n_stripes = n_outer / 8;
    assert_eq!(n_stripes % TILE_T, 0);
    assert_eq!(k % BLOCK_K, 0);
    let n_blocks_full = k / BLOCK_K;
    // Boundary block past useful_bits holds 0 padding → table[0] = 0 contributes
    // nothing, so cover only blocks that touch useful bits.
    let n_blocks = useful_bits.div_ceil(BLOCK_K).min(n_blocks_full);

    let n_tiles = n_stripes / TILE_T;
    let tiles_per_chunk = (n_tiles / 256).max(1);
    let bytes_per_chunk = tiles_per_chunk * TILE_T * k;

    z_packed
        .par_chunks(bytes_per_chunk)
        .enumerate()
        .fold(
            || (vec![F128::ZERO; k], vec![F128::ZERO; TILE_T * 256].into_boxed_slice()),
            |(mut out, mut tables), (chunk_idx, chunk_bytes)| {
                let tile_start = chunk_idx * tiles_per_chunk;
                // TILE_T × 256 F128 = 32 KB tables. L1 resident.
                let n_tiles_in_chunk = chunk_bytes.len() / (TILE_T * k);
                for tile_rel in 0..n_tiles_in_chunk {
                    let tile_idx = tile_start + tile_rel;
                    let stripe_base = tile_idx * TILE_T;
                    // SAFETY: tile_rel < n_tiles_in_chunk so the offset stays in bounds.
                    let tile_bytes_ptr = unsafe { chunk_bytes.as_ptr().add(tile_rel * TILE_T * k) };
                    for t in 0..TILE_T {
                        let byte_idx = stripe_base + t;
                        let eq_off = 8 * byte_idx;
                        build_sum_table(
                            &eq_outer[eq_off..eq_off + 8],
                            &mut tables[t * 256..(t + 1) * 256],
                        );
                    }
                    let tables_ptr = tables.as_ptr() as *const u8;
                    for block_idx in 0..n_blocks {
                        let bs = block_idx * BLOCK_K;
                        // SAFETY: bs + BLOCK_K ≤ k ≤ out.len(); tile_bytes_ptr
                        // covers TILE_T*k bytes; tables_ptr covers TILE_T*256 F128.
                        unsafe {
                            process_block_x86(
                                tile_bytes_ptr,
                                k,
                                bs,
                                tables_ptr,
                                out.as_mut_ptr().add(bs),
                            );
                        }
                    }
                }
                (out, tables)
            },
        )
        .map(|(out, _)| out)
        .reduce(
            || vec![F128::ZERO; k],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x += *y;
                }
                a
            },
        )
}

/// Byte permutation of the 8×64 bit transpose: destination byte `8 k + p` takes source
/// byte `8 (7 - p) + k` (byte `k` of row `7 - p`).
const BANK_GATHER: [u8; 64] = {
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

/// Bit transpose of eight-word runs on AVX-512 (the union bit-bank fold): run `g` of
/// `words` (8 F128 = 2 zmm) is split into its lo and hi qword planes by two
/// `vpermt2q`; each plane (eight rows of 64 bits) is transposed by one `vpermb`
/// (byte `k` of every row into qword `k`, row `r` at byte `7 - r`) and one
/// `vgf2p8affineqb` with the identity bytes as data, so byte `b` of the result holds bit
/// `b` of every row (bit `r` = row `r`). Output `out[128 g ..][..64]` from the lo plane,
/// `[64..128]` from the hi plane.
///
/// # Safety
/// AVX-512 F/BW/VBMI and GFNI available; `out.len() == 16 * words.len()`, `words.len()`
/// a multiple of 8.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")]
pub(crate) unsafe fn bank_transpose_avx512(words: &[F128], out: &mut [u8]) {
    use core::arch::x86_64::{
        _mm512_gf2p8affine_epi64_epi8, _mm512_loadu_si512, _mm512_permutex2var_epi64,
        _mm512_permutexvar_epi8, _mm512_set1_epi64, _mm512_set_epi64, _mm512_storeu_si512,
    };
    debug_assert_eq!(out.len(), 16 * words.len());
    debug_assert!(words.len().is_multiple_of(8));
    // SAFETY: BANK_GATHER is 64 readable bytes.
    let gidx = unsafe { _mm512_loadu_si512(BANK_GATHER.as_ptr().cast()) };
    let ident = _mm512_set1_epi64(0x8040_2010_0804_0201_u64 as i64);
    let lo_idx = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
    let hi_idx = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
    for g in 0..words.len() / 8 {
        // SAFETY: 8 F128 (128 bytes) readable at run g, 128 bytes writable at out[128 g].
        unsafe {
            let p = words.as_ptr().add(8 * g).cast::<u8>();
            let v0 = _mm512_loadu_si512(p.cast());
            let v1 = _mm512_loadu_si512(p.add(64).cast());
            let lo = _mm512_permutex2var_epi64(v0, lo_idx, v1);
            let hi = _mm512_permutex2var_epi64(v0, hi_idx, v1);
            let tl = _mm512_gf2p8affine_epi64_epi8::<0>(ident, _mm512_permutexvar_epi8(gidx, lo));
            let th = _mm512_gf2p8affine_epi64_epi8::<0>(ident, _mm512_permutexvar_epi8(gidx, hi));
            let o = out.as_mut_ptr().add(128 * g);
            _mm512_storeu_si512(o.cast(), tl);
            _mm512_storeu_si512(o.add(64).cast(), th);
        }
    }
}

/// Bit-bank table lookups on AVX-512 for points interleaved four to a 64-byte line.
///
/// `tables` holds, for each row group `g < gn` and each byte value `s < 256`, an entry of
/// `stride` consecutive F128 (`stride` a multiple of four): entry `(g, s)` starts at
/// `tables[(256 g + s) * stride]`, and F128 `p` of it is the subset-sum for point `p`
/// (padding lanes `np ..` are never stored to the output). `bytes[128 g + b]` is the
/// transposed byte of group `g` and output bit `b`. For every point `p < np` and
/// `b < nb`, `outs[p][base + b] ^= XOR_g tables[(256 g + bytes[128 g + b]) * stride + p]`.
///
/// Each point quad `q` (points `4 q ..`) is one zmm per lookup: a byte selects one 64-byte
/// line, loaded and XORed into one of `B` independent accumulators. Accumulators start at
/// zero and are folded into the output once per call, so the output sees `gn` XORs per
/// bit in one read-modify-write.
///
/// # Safety
/// AVX-512F available; `stride % 4 == 0`; `tables` readable for `gn * 256 * stride` F128,
/// 64-byte aligned (an unaligned base is correct but loses the single-line loads);
/// `bytes` readable for `128 * gn` bytes; `nb <= 128`; `outs.len() >= np`, and
/// `outs[p].add(base)` writable for `nb` F128 for each `p < np`, with no other thread
/// touching them.
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn bank_lookup_avx512(
    tables: *const F128,
    stride: usize,
    np: usize,
    bytes: *const u8,
    gn: usize,
    nb: usize,
    outs: &[*mut F128],
    base: usize,
) {
    debug_assert!(stride.is_multiple_of(4) && np <= stride && nb <= 128 && outs.len() >= np);
    for q in 0..stride / 4 {
        if 4 * q >= np {
            break;
        }
        let cnt = (np - 4 * q).min(4);
        let mut b = 0;
        while b + 8 <= nb {
            // SAFETY: forwarded from this function's contract (b + 8 <= nb).
            unsafe { bank_lookup_quad::<8>(tables, stride, q, cnt, bytes, gn, b, outs, base) };
            b += 8;
        }
        while b < nb {
            // SAFETY: forwarded from this function's contract (b < nb).
            unsafe { bank_lookup_quad::<1>(tables, stride, q, cnt, bytes, gn, b, outs, base) };
            b += 1;
        }
    }
}

/// `B` consecutive output bits `b ..` of point quad `q` (`cnt <= 4` live points): see
/// [`bank_lookup_avx512`].
///
/// # Safety
/// The contract of [`bank_lookup_avx512`] with `b + B <= nb`.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn bank_lookup_quad<const B: usize>(
    tables: *const F128,
    stride: usize,
    q: usize,
    cnt: usize,
    bytes: *const u8,
    gn: usize,
    b: usize,
    outs: &[*mut F128],
    base: usize,
) {
    use core::arch::x86_64::{
        __m512i, _mm512_loadu_si512, _mm512_setzero_si512, _mm512_storeu_si512, _mm512_xor_si512,
    };
    // Bytes per table entry and per group of 256 entries.
    let entry = 16 * stride;
    let mut acc = [_mm512_setzero_si512(); B];
    for g in 0..gn {
        // SAFETY: group g's 256 entries and its 128 bytes lie inside the caller's ranges;
        // a byte value is < 256, so the entry is inside the group; the quad offset
        // 64 q + 64 <= entry.
        unsafe {
            let tg = tables.cast::<u8>().add(g * 256 * entry + 64 * q);
            let bp = bytes.add(128 * g + b);
            for (j, a) in acc.iter_mut().enumerate() {
                let s = *bp.add(j) as usize;
                *a = _mm512_xor_si512(*a, _mm512_loadu_si512(tg.add(s * entry).cast()));
            }
        }
    }
    for (j, a) in acc.iter().enumerate() {
        let mut lanes = [F128::ZERO; 4];
        // SAFETY: lanes is 64 writable bytes; outs[4 q + i].add(base + b + j) is writable
        // by the caller's contract for i < cnt.
        unsafe {
            _mm512_storeu_si512(lanes.as_mut_ptr().cast::<__m512i>(), *a);
            for (i, l) in lanes.iter().enumerate().take(cnt) {
                let o = outs[4 * q + i].add(base + b + j);
                *o += *l;
            }
        }
    }
}
