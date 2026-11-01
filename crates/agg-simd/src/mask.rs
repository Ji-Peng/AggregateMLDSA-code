//! Masked sums of integer values: one mask byte selects eight consecutive positions.
//!
//! `psi` holds `rows` mask bytes per row block (`row_bytes` bytes per row, shared by every
//! record); bit `pos % 8` of byte `pos / 8` selects position `pos`. The scalar path builds
//! the 256 subset sums of each byte column once and reads them per row; the AVX-512 path
//! uses each mask byte as an eight-lane predicate and adds the column's eight values with
//! one masked add per row; the AVX2 path selects each half of the eight values with a
//! four-lane mask looked up from the byte's nibble.
use rayon::prelude::*;

/// `out[i * rows + o] = sum_pos bit(psi[o * row_bytes + pos / 8], pos % 8) ? v(i, pos) : 0`
/// for every record `i` and row `o`.
///
/// Every partial sum must fit `i64`; the values stay below `2^27` in the projection use.
///
/// # Panics
/// If `psi` does not hold `rows * row_bytes` bytes.
pub fn masked_row_sums(
    psi: &[u8],
    records: usize,
    rows: usize,
    row_bytes: usize,
    v: impl Fn(usize, usize) -> i64 + Sync,
) -> Vec<i64> {
    assert_eq!(psi.len(), rows * row_bytes, "one mask byte block per row");
    let mut out = vec![0_i64; records * rows];
    #[cfg(target_arch = "x86_64")]
    if crate::detect::avx512_int() {
        // SAFETY: the detector checked AVX-512F, and the kernel indexes only the value
        // buffer and mask bytes that the loop bounds cover.
        unsafe { vector(psi, rows, row_bytes, &v, &mut out) };
        return out;
    }
    #[cfg(target_arch = "x86_64")]
    if crate::detect::avx2() {
        // SAFETY: the detector checked AVX2, and the kernel indexes only the value buffer
        // and mask bytes that the loop bounds cover.
        unsafe { vector_avx2(psi, rows, row_bytes, &v, &mut out) };
        return out;
    }
    scalar(psi, rows, row_bytes, &v, &mut out);
    out
}

fn scalar(
    psi: &[u8],
    rows: usize,
    row_bytes: usize,
    v: &(impl Fn(usize, usize) -> i64 + Sync),
    out: &mut [i64],
) {
    out.par_chunks_mut(rows).enumerate().for_each(|(i, sums_out)| {
        let mut sums = [0_i64; 256];
        sums_out.fill(0);
        for byte in 0..row_bytes {
            let values: [i64; 8] = std::array::from_fn(|bit| v(i, 8 * byte + bit));
            for mask in 1..256_usize {
                // The subset without its lowest position, plus that position.
                sums[mask] = sums[mask & (mask - 1)] + values[mask.trailing_zeros() as usize];
            }
            for (o, acc) in sums_out.iter_mut().enumerate() {
                *acc += sums[usize::from(psi[o * row_bytes + byte])];
            }
        }
    });
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn vector(
    psi: &[u8],
    rows: usize,
    row_bytes: usize,
    v: &(dyn Fn(usize, usize) -> i64 + Sync),
    out: &mut [i64],
) {
    out.par_chunks_mut(rows).enumerate().map_init(
        || vec![0_i64; 8 * row_bytes],
        |values, (i, sums_out)| {
            // The record's values are read once and then kept in L1 across the rows.
            for (pos, slot) in values.iter_mut().enumerate() {
                *slot = v(i, pos);
            }
            // SAFETY: `values` holds `8 * row_bytes` elements and the row loop stays inside
            // the masks and the record's output slice.
            unsafe { rows_into(psi, row_bytes, values, sums_out) };
        },
    )
    .for_each(|()| {});
}

/// The masked sums of one record. Every lane accumulates at most `row_bytes` values, so the
/// eight-lane total stays inside `i64` whenever the per-lane partial sums do.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn rows_into(psi: &[u8], row_bytes: usize, values: &[i64], out: &mut [i64]) {
    use std::arch::x86_64::{
        __mmask8, _mm512_loadu_si512, _mm512_mask_add_epi64, _mm512_setzero_si512,
        _mm512_storeu_si512,
    };
    for (o, slot) in out.iter_mut().enumerate() {
        let bytes = &psi[o * row_bytes..(o + 1) * row_bytes];
        let mut acc = _mm512_setzero_si512();
        for (byte, &mask) in bytes.iter().enumerate() {
            // SAFETY: byte < row_bytes, so `8 * byte + 8 <= values.len()`.
            let x = unsafe { _mm512_loadu_si512(values.as_ptr().add(8 * byte).cast()) };
            acc = _mm512_mask_add_epi64(acc, mask as __mmask8, acc, x);
        }
        let mut lanes = [0_i64; 8];
        // SAFETY: `lanes` holds the eight output lanes of one ZMM register.
        unsafe { _mm512_storeu_si512(lanes.as_mut_ptr().cast(), acc) };
        *slot = lanes.iter().sum();
    }
}

/// [`vector`] with four-lane AVX2 masks; any integer AVX-512 level takes [`vector`] instead.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vector_avx2(
    psi: &[u8],
    rows: usize,
    row_bytes: usize,
    v: &(dyn Fn(usize, usize) -> i64 + Sync),
    out: &mut [i64],
) {
    out.par_chunks_mut(rows).enumerate().map_init(
        || vec![0_i64; 8 * row_bytes],
        |values, (i, sums_out)| {
            for (pos, slot) in values.iter_mut().enumerate() {
                *slot = v(i, pos);
            }
            // SAFETY: `values` holds `8 * row_bytes` elements and the row loop stays inside
            // the masks and the record's output slice.
            unsafe { rows_into_avx2(psi, row_bytes, values, sums_out) };
        },
    )
    .for_each(|()| {});
}

/// Lane `l` of entry `n` is all ones exactly when bit `l` of the nibble `n` is set.
#[cfg(target_arch = "x86_64")]
static NIBBLE_MASKS: [[i64; 4]; 16] = {
    let mut table = [[0_i64; 4]; 16];
    let mut n = 0;
    while n < 16 {
        let mut l = 0;
        while l < 4 {
            if (n >> l) & 1 == 1 {
                table[n][l] = -1;
            }
            l += 1;
        }
        n += 1;
    }
    table
};

/// The masked sums of one record with AVX2. Lane `l` of the low (high) accumulator sums
/// position `8 byte + l` (`8 byte + 4 + l`) over the bytes, at most `row_bytes` values each,
/// so the total stays inside `i64` whenever the partial sums do, as in [`rows_into`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rows_into_avx2(psi: &[u8], row_bytes: usize, values: &[i64], out: &mut [i64]) {
    use std::arch::x86_64::{
        __m256i, _mm256_add_epi64, _mm256_and_si256, _mm256_loadu_si256, _mm256_setzero_si256,
        _mm256_storeu_si256,
    };
    let masks = NIBBLE_MASKS.as_ptr().cast::<__m256i>();
    for (o, slot) in out.iter_mut().enumerate() {
        let bytes = &psi[o * row_bytes..(o + 1) * row_bytes];
        let mut low = _mm256_setzero_si256();
        let mut high = _mm256_setzero_si256();
        for (byte, &mask) in bytes.iter().enumerate() {
            // SAFETY: byte < row_bytes, so `8 * byte + 8 <= values.len()`; a nibble indexes
            // one of the sixteen table entries.
            unsafe {
                let x = _mm256_loadu_si256(values.as_ptr().add(8 * byte).cast());
                let y = _mm256_loadu_si256(values.as_ptr().add(8 * byte + 4).cast());
                let mx = _mm256_loadu_si256(masks.add(usize::from(mask & 15)));
                let my = _mm256_loadu_si256(masks.add(usize::from(mask >> 4)));
                low = _mm256_add_epi64(low, _mm256_and_si256(x, mx));
                high = _mm256_add_epi64(high, _mm256_and_si256(y, my));
            }
        }
        let mut lanes = [0_i64; 8];
        // SAFETY: `lanes` holds the eight output lanes of the two YMM registers.
        unsafe {
            _mm256_storeu_si256(lanes.as_mut_ptr().cast(), low);
            _mm256_storeu_si256(lanes.as_mut_ptr().add(4).cast(), high);
        }
        *slot = lanes.iter().sum();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kernel_matches_the_subset_sums() {
        let (records, rows, row_bytes) = (5, 7, 16);
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let psi: Vec<u8> = (0..rows * row_bytes).map(|_| next() as u8).collect();
        let values: Vec<i64> =
            (0..records * 8 * row_bytes).map(|_| (next() % (1 << 27)) as i64 - (1 << 26)).collect();
        let v = |i: usize, pos: usize| values[i * 8 * row_bytes + pos];
        let direct: Vec<i64> = (0..records * rows)
            .map(|cell| {
                let (i, o) = (cell / rows, cell % rows);
                (0..8 * row_bytes)
                    .filter(|&pos| (psi[o * row_bytes + pos / 8] >> (pos % 8)) & 1 == 1)
                    .map(|pos| v(i, pos))
                    .sum()
            })
            .collect();
        let mut out = vec![0_i64; records * rows];
        scalar(&psi, rows, row_bytes, &v, &mut out);
        assert_eq!(out, direct, "scalar");
        assert_eq!(masked_row_sums(&psi, records, rows, row_bytes, v), direct, "dispatched");
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f") {
                out.fill(0);
                // SAFETY: the CPU has AVX-512F.
                unsafe { vector(&psi, rows, row_bytes, &v, &mut out) };
                assert_eq!(out, direct, "AVX-512F");
            }
            if std::is_x86_feature_detected!("avx2") {
                out.fill(0);
                // SAFETY: the CPU has AVX2.
                unsafe { vector_avx2(&psi, rows, row_bytes, &v, &mut out) };
                assert_eq!(out, direct, "AVX2");
            }
        }
    }
}
