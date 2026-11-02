//! Bit packing of Boolean witness columns: six `{0, 1}` lanes into one byte per cell (the
//! `w_1'` bits) and four lanes into one nibble, plus the strip accumulators that consume them.

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use std::arch::x86_64::*;

    /// Pack 6 Boolean `i8` slices of length `len` into one byte per cell (`sum_{t=0..6} b_t << t`)
    /// using 64-lane AVX-512BW operations. Returns `false` if any input byte is outside `{0, 1}`.
    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
    pub unsafe fn pack_six_bits_chunk_avx512(
        lanes: &[&[i8]; 6],
        start: usize,
        out: &mut [u8],
    ) -> bool {
        // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
        unsafe {
            let one = _mm512_set1_epi8(1);
            let len = out.len();
            let vec_len = len / 64 * 64;
            let mut bad = 0u64;
            for offset in (0..vec_len).step_by(64) {
                let idx = start + offset;
                let b0 = _mm512_loadu_si512(lanes[0].as_ptr().add(idx).cast());
                let b1 = _mm512_loadu_si512(lanes[1].as_ptr().add(idx).cast());
                let b2 = _mm512_loadu_si512(lanes[2].as_ptr().add(idx).cast());
                let b3 = _mm512_loadu_si512(lanes[3].as_ptr().add(idx).cast());
                let b4 = _mm512_loadu_si512(lanes[4].as_ptr().add(idx).cast());
                let b5 = _mm512_loadu_si512(lanes[5].as_ptr().add(idx).cast());

                bad |= _mm512_cmpgt_epu8_mask(b0, one)
                    | _mm512_cmpgt_epu8_mask(b1, one)
                    | _mm512_cmpgt_epu8_mask(b2, one)
                    | _mm512_cmpgt_epu8_mask(b3, one)
                    | _mm512_cmpgt_epu8_mask(b4, one)
                    | _mm512_cmpgt_epu8_mask(b5, one);

                // Since each b_t is in {0, 1}, 16-bit shifts never cross byte boundaries.
                let p01 = _mm512_or_si512(b0, _mm512_slli_epi16::<1>(b1));
                let p23 = _mm512_or_si512(
                    _mm512_slli_epi16::<2>(b2),
                    _mm512_slli_epi16::<3>(b3),
                );
                let p45 = _mm512_or_si512(
                    _mm512_slli_epi16::<4>(b4),
                    _mm512_slli_epi16::<5>(b5),
                );
                let packed = _mm512_or_si512(p01, _mm512_or_si512(p23, p45));
                _mm512_storeu_si512(out.as_mut_ptr().add(offset).cast(), packed);
            }
            if bad != 0 {
                return false;
            }
            for offset in vec_len..len {
                let idx = start + offset;
                let mut byte = 0u8;
                for t in 0..6 {
                    let b = lanes[t][idx] as u8;
                    if b > 1 {
                        return false;
                    }
                    byte |= b << t;
                }
                out[offset] = byte;
            }
            true
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    /// [`super::avx512::pack_six_bits_chunk_avx512`] with 32-lane AVX2 operations.
    #[target_feature(enable = "avx2")]
    pub unsafe fn pack_six_bits_chunk_avx2(lanes: &[&[i8]; 6], start: usize, out: &mut [u8]) -> bool {
        // SAFETY: the enclosing function enables AVX2; the caller checked it at run time and
        // every lane holds at least `start + out.len()` bytes.
        unsafe {
            let len = out.len();
            let vec_len = len / 32 * 32;
            // A byte outside {0, 1} has a bit set in 0xFE.
            let high = _mm256_set1_epi8(0xFE_u8 as i8);
            let mut bad = _mm256_setzero_si256();
            for offset in (0..vec_len).step_by(32) {
                let idx = start + offset;
                let load = |t: usize| _mm256_loadu_si256(lanes[t].as_ptr().add(idx).cast());
                let (b0, b1, b2, b3, b4, b5) = (load(0), load(1), load(2), load(3), load(4), load(5));
                let any = _mm256_or_si256(
                    _mm256_or_si256(_mm256_or_si256(b0, b1), _mm256_or_si256(b2, b3)),
                    _mm256_or_si256(b4, b5),
                );
                bad = _mm256_or_si256(bad, _mm256_and_si256(any, high));
                // Each b_t is in {0, 1} (checked above), so 16-bit shifts never cross a byte.
                let p01 = _mm256_or_si256(b0, _mm256_slli_epi16::<1>(b1));
                let p23 = _mm256_or_si256(_mm256_slli_epi16::<2>(b2), _mm256_slli_epi16::<3>(b3));
                let p45 = _mm256_or_si256(_mm256_slli_epi16::<4>(b4), _mm256_slli_epi16::<5>(b5));
                let packed = _mm256_or_si256(p01, _mm256_or_si256(p23, p45));
                _mm256_storeu_si256(out.as_mut_ptr().add(offset).cast(), packed);
            }
            if _mm256_testz_si256(bad, bad) == 0 {
                return false;
            }
            for offset in vec_len..len {
                let idx = start + offset;
                let mut byte = 0u8;
                for t in 0..6 {
                    let b = lanes[t][idx] as u8;
                    if b > 1 {
                        return false;
                    }
                    byte |= b << t;
                }
                out[offset] = byte;
            }
            true
        }
    }
}

/// Pack 6 Boolean `i8` columns of equal length into a single `Vec<u8>` (`six_bits[i] = sum_t b_t[i] << t`).
/// Returns `None` if any entry is outside `{0, 1}`.
pub fn pack_six_bits(lanes: &[&[i8]; 6]) -> Option<Vec<u8>> {
    use std::sync::atomic::{AtomicBool, Ordering};

    let len = lanes[0].len();
    assert!(lanes.iter().all(|l| l.len() == len));
    let ok = AtomicBool::new(true);
    // The kernels use byte compares and shifts only: AVX-512BW without IFMA, or AVX2.
    let avx512 = crate::detect::avx512_int();
    let avx2 = crate::detect::avx2();
    let out = crate::buf::fill_vec_par_chunks(len, 1 << 16, |chunk_idx, dst| {
        if !ok.load(Ordering::Relaxed) {
            dst.fill(0);
            return;
        }
        let start = chunk_idx * (1 << 16);
        #[cfg(target_arch = "x86_64")]
        if avx512 || avx2 {
            // SAFETY: the detector checked the CPU features of the kernel, and every lane
            // holds `len >= start + dst.len()` bytes (asserted above).
            let chunk_ok = unsafe {
                if avx512 {
                    avx512::pack_six_bits_chunk_avx512(lanes, start, dst)
                } else {
                    avx2::pack_six_bits_chunk_avx2(lanes, start, dst)
                }
            };
            if !chunk_ok {
                ok.store(false, Ordering::Relaxed);
            }
            return;
        }
        for (offset, slot) in dst.iter_mut().enumerate() {
            let idx = start + offset;
            let mut byte = 0u8;
            for t in 0..6 {
                let b = lanes[t][idx] as u8;
                if b > 1 {
                    ok.store(false, Ordering::Relaxed);
                    dst.fill(0);
                    return;
                }
                byte |= b << t;
            }
            *slot = byte;
        }
    });
    ok.into_inner().then_some(out)
}

/// Lookup table mapping a 5-bit mask `b & 0x1F` to five 12-bit lane increments at bit offsets
/// `0, 12, 24, 36, 48` inside a `u64` accumulator (each field supports sums up to `4,095`).
const EXPAND_5X12: [u64; 32] = {
    let mut table = [0u64; 32];
    let mut b = 0usize;
    while b < 32 {
        table[b] = (b as u64 & 1)
            | (((b as u64 >> 1) & 1) << 12)
            | (((b as u64 >> 2) & 1) << 24)
            | (((b as u64 >> 3) & 1) << 36)
            | (((b as u64 >> 4) & 1) << 48);
        b += 1;
    }
    table
};

/// Accumulate packed 6-bit inputs `six_bits[zeta]` over the column range `col_start..col_end`
/// of a sparse matrix (`entries` has `k` row indices per column) into `acc0` (lanes 0 through 4 in
/// 12-bit fields) and `acc5` (lane 5 in `u16`).
///
/// Lazy range: each 12-bit field in `acc0` holds `[0, 4095]`; the production bridge row weight
/// cap is `W_1 = W_2 = 2,047 < 4,095`, so no carry ever crosses a 12-bit boundary.
#[inline]
pub fn accumulate_six_lanes_strip(
    six_bits: &[u8],
    entries: &[u32],
    k: usize,
    col_start: usize,
    col_end: usize,
    acc0: &mut [u64],
    acc5: &mut [u16],
) {
    for (zeta, &b) in six_bits[col_start..col_end].iter().enumerate() {
        if b != 0 {
            let w0 = EXPAND_5X12[(b & 0x1F) as usize];
            let w5 = u16::from(b >> 5);
            let col_off = (col_start + zeta) * k;
            for &row in &entries[col_off..col_off + k] {
                let r = row as usize;
                acc0[r] += w0;
                acc5[r] += w5;
            }
        }
    }
}

/// Pack 4 Boolean `i8` columns of equal length into one nibble per byte
/// (`four_bits[i] = sum_t b_t[i] << t`), in parallel chunks of [`pack_four_bits_into`].
/// Returns `None` if any entry is outside `{0, 1}`.
pub fn pack_four_bits(lanes: &[&[i8]; 4]) -> Option<Vec<u8>> {
    use std::sync::atomic::{AtomicBool, Ordering};

    const CHUNK: usize = 1 << 16;
    let len = lanes[0].len();
    assert!(lanes.iter().all(|l| l.len() == len));
    let ok = AtomicBool::new(true);
    let out = crate::buf::fill_vec_par_chunks(len, CHUNK, |chunk_idx, dst: &mut [u8]| {
        let start = chunk_idx * CHUNK;
        let src: [&[i8]; 4] = std::array::from_fn(|t| &lanes[t][start..start + dst.len()]);
        if !pack_four_bits_into(src, dst) {
            ok.store(false, Ordering::Relaxed);
        }
    });
    ok.into_inner().then_some(out)
}

/// `dst[i] = sum_t (b_t[i] & 1) << t` for four `i8` lanes of `dst.len()` cells; returns
/// whether every entry is in `{0, 1}` (an `i8` of -1 reads as `0xFF` and is caught). The
/// lanes are sliced to `dst.len()` first, so the loop has no bounds checks and
/// auto-vectorises (32 cells per AVX2 operation on the x86-64-v3 baseline, the same code on
/// every `AGG_ISA` tier).
#[inline]
pub fn pack_four_bits_into(lanes: [&[i8]; 4], dst: &mut [u8]) -> bool {
    let n = dst.len();
    let (l0, l1, l2, l3) = (&lanes[0][..n], &lanes[1][..n], &lanes[2][..n], &lanes[3][..n]);
    let mut bad = 0u8;
    for i in 0..n {
        let (b0, b1, b2, b3) = (l0[i] as u8, l1[i] as u8, l2[i] as u8, l3[i] as u8);
        bad |= (b0 | b1 | b2 | b3) & 0xFE;
        dst[i] = (b0 & 1) | (b1 & 1) << 1 | (b2 & 1) << 2 | (b3 & 1) << 3;
    }
    bad == 0
}

/// Lookup table mapping a nibble `b & 0xF` to four 16-bit lane increments at bit offsets
/// `0, 16, 32, 48` of a `u64` accumulator.
const EXPAND_4X16: [u64; 16] = {
    let mut table = [0u64; 16];
    let mut b = 0usize;
    while b < 16 {
        table[b] = (b as u64 & 1)
            | (((b as u64 >> 1) & 1) << 16)
            | (((b as u64 >> 2) & 1) << 32)
            | (((b as u64 >> 3) & 1) << 48);
        b += 1;
    }
    table
};

/// Accumulate packed 4-bit inputs `four_bits[zeta]` over the column range `col_start..col_end`
/// of a sparse matrix (`entries` has `k` row indices per column) into `acc` (lane `t` in the
/// 16-bit field at bit `16 t`).
///
/// Lazy range: each field holds `[0, 65535]`; the caller's row-weight cap (2,047 for the
/// production bridge levels 1 and 2) keeps every row sum below it, so no carry crosses a field.
#[inline]
pub fn accumulate_four_lanes_strip(
    four_bits: &[u8],
    entries: &[u32],
    k: usize,
    col_start: usize,
    col_end: usize,
    acc: &mut [u64],
) {
    for (zeta, &b) in four_bits[col_start..col_end].iter().enumerate() {
        if b != 0 {
            let w = EXPAND_4X16[(b & 0xF) as usize];
            let col_off = (col_start + zeta) * k;
            for &row in &entries[col_off..col_off + k] {
                acc[row as usize] += w;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_pack(lanes: &[&[i8]; 6]) -> Option<Vec<u8>> {
        (0..lanes[0].len())
            .map(|i| {
                (0..6).try_fold(0_u8, |byte, t| {
                    let b = lanes[t][i] as u8;
                    (b <= 1).then_some(byte | b << t)
                })
            })
            .collect()
    }

    #[test]
    fn six_bit_packing_matches_the_scalar_definition_on_every_kernel() {
        // A length that is neither a multiple of 32 nor of 64 exercises the tails.
        let len = 3 * 64 + 37;
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let columns: Vec<Vec<i8>> = (0..6)
            .map(|_| {
                (0..len)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        (state & 1) as i8
                    })
                    .collect()
            })
            .collect();
        for bad in [None, Some((2, 5)), Some((4, len - 1)), Some((0, 100))] {
            let mut columns = columns.clone();
            if let Some((t, i)) = bad {
                columns[t][i] = if i % 2 == 0 { 2 } else { -1 };
            }
            let lanes: [&[i8]; 6] = std::array::from_fn(|t| columns[t].as_slice());
            let want = scalar_pack(&lanes);
            assert_eq!(want.is_none(), bad.is_some());
            assert_eq!(pack_six_bits(&lanes), want, "dispatched, {bad:?}");
            #[cfg(target_arch = "x86_64")]
            {
                let mut out = vec![0_u8; len];
                if std::is_x86_feature_detected!("avx2") {
                    // SAFETY: the CPU has AVX2 and every lane has `len` bytes.
                    let ok = unsafe { avx2::pack_six_bits_chunk_avx2(&lanes, 0, &mut out) };
                    assert_eq!(ok, want.is_some(), "AVX2, {bad:?}");
                    if let Some(want) = &want {
                        assert_eq!(&out, want, "AVX2");
                    }
                }
                if std::is_x86_feature_detected!("avx512f")
                    && std::is_x86_feature_detected!("avx512dq")
                    && std::is_x86_feature_detected!("avx512bw")
                    && std::is_x86_feature_detected!("avx512vl")
                {
                    // SAFETY: the CPU has AVX-512F/DQ/BW/VL and every lane has `len` bytes.
                    let ok = unsafe { avx512::pack_six_bits_chunk_avx512(&lanes, 0, &mut out) };
                    assert_eq!(ok, want.is_some(), "AVX-512, {bad:?}");
                    if let Some(want) = &want {
                        assert_eq!(&out, want, "AVX-512");
                    }
                }
            }
        }
    }

    #[test]
    fn four_bit_packing_and_accumulation_match_the_definition() {
        let len = 3 * 64 + 37;
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let columns: Vec<Vec<i8>> = (0..4)
            .map(|_| {
                (0..len)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        (state & 1) as i8
                    })
                    .collect()
            })
            .collect();
        let lanes: [&[i8]; 4] = std::array::from_fn(|t| columns[t].as_slice());
        let packed = pack_four_bits(&lanes).expect("bits");
        for i in 0..len {
            let want: u8 = (0..4).map(|t| (columns[t][i] as u8) << t).sum();
            assert_eq!(packed[i], want);
        }
        for bad_at in [5, len - 1] {
            let mut tampered = columns[2].clone();
            tampered[bad_at] = 2;
            let mut bad = lanes;
            bad[2] = &tampered;
            assert!(pack_four_bits(&bad).is_none());
            // The slice kernel flags the non-bit and keeps its low bit.
            let mut dst = vec![0u8; len];
            assert!(!pack_four_bits_into(bad, &mut dst));
            assert_eq!(dst[bad_at], packed[bad_at] & !4);
            assert!(pack_four_bits_into(lanes, &mut dst) && dst == packed);
        }
        // Two rows per column: rows (c mod 7, (c + 3) mod 7).
        let k = 2;
        let entries: Vec<u32> = (0..len).flat_map(|c| [(c % 7) as u32, ((c + 3) % 7) as u32]).collect();
        let mut acc = vec![0u64; 7];
        accumulate_four_lanes_strip(&packed, &entries, k, 0, len, &mut acc);
        for row in 0..7 {
            for t in 0..4 {
                let want: u64 = (0..len)
                    .filter(|&c| entries[2 * c] as usize == row || entries[2 * c + 1] as usize == row)
                    .map(|c| columns[t][c] as u64)
                    .sum();
                assert_eq!((acc[row] >> (16 * t)) & 0xFFFF, want, "row {row}, lane {t}");
            }
        }
    }
}
