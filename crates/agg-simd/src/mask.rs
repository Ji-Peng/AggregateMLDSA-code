//! Masked sums of integer values: one mask byte selects eight consecutive positions.
//!
//! `psi` holds `rows` mask bytes per row block (`row_bytes` bytes per row, shared by every
//! record); bit `pos % 8` of byte `pos / 8` selects position `pos`. The scalar path builds
//! the 256 subset sums of each byte column once and reads them per row; the AVX-512 path
//! uses each mask byte as an eight-lane predicate and adds the column's eight values with
//! one masked add per row.
use crate::detect::{backend, Backend};
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
    if backend() == Backend::Avx512 {
        // SAFETY: the backend check provides the AVX-512 features, and the kernel indexes
        // only the value buffer and mask bytes that the loop bounds cover.
        unsafe { vector(psi, rows, row_bytes, &v, &mut out) };
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
