//! Leaf kernels for the F128 additive NTT.
//!
//! Transform scheduling and cache-blocking policy stay in the parent module;
//! this module owns the architecture-specific operations on blocks of data.
//! agg-mldsa patch (PATCH.md): the x86 AVX-512/VPCLMULQDQ kernels are
//! chosen at run time (`crate::simd`) instead of by `cfg(target_feature)`.

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
use crate::ntt::additive_ntt_f128::kernels::aarch64::{butterfly_block, butterfly_block_pair};
use crate::ntt::additive_ntt_f128::kernels::portable::{
    butterfly_fused_2layer as butterfly_fused_2layer_portable,
    blocked4_fused_3layer as blocked4_fused_3layer_portable,
    butterfly_fused_3layer_row as butterfly_fused_3layer_row_portable,
    butterfly_fused_4layer_row as butterfly_fused_4layer_row_portable,
    butterfly_row_pair as butterfly_row_pair_portable,
};
#[cfg(target_arch = "x86_64")]
use crate::ntt::additive_ntt_f128::kernels::x86_64::{
    butterfly_fused_2layer as butterfly_fused_2layer_x86_64,
    blocked4_fused_3layer as blocked4_fused_3layer_x86_64,
    butterfly_fused_3layer_row as butterfly_fused_3layer_row_x86_64,
    butterfly_fused_4layer_row as butterfly_fused_4layer_row_x86_64,
    butterfly_row_pair as butterfly_row_pair_x86_64,
};
use crate::{
    field::F128,
    ntt::additive_ntt_f128::kernels::portable::butterfly_fused_3layer as butterfly_fused_3layer_portable,
};
mod portable;

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
mod aarch64;

#[cfg(target_arch = "x86_64")]
mod x86_64;

/// True when the x86 AVX-512 fused kernels run (the parent's schedule fuses four
/// layers per pass only then).
#[inline]
pub(super) fn x86_avx512_clmul() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        crate::simd::has(crate::simd::AVX512_CLMUL)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline]
pub(super) fn butterfly_row_pair(top: &mut [F128], bot: &mut [F128], twiddle: F128) {
    debug_assert_eq!(top.len(), bot.len());

    #[cfg(target_arch = "x86_64")]
    if x86_avx512_clmul() {
        // SAFETY: the runtime check guarantees the required target features.
        unsafe { butterfly_row_pair_x86_64(top, bot, twiddle) };
        return;
    }
    butterfly_row_pair_portable(top, bot, twiddle);
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn butterfly_fused_2layer(
    a: &mut [F128],
    b: &mut [F128],
    c: &mut [F128],
    d: &mut [F128],
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), c.len());
    debug_assert_eq!(a.len(), d.len());

    #[cfg(target_arch = "x86_64")]
    if x86_avx512_clmul() {
        // SAFETY: the runtime check guarantees the required target features.
        unsafe { butterfly_fused_2layer_x86_64(a, b, c, d, t_outer, t_inner_a, t_inner_b) };
        return;
    }
    butterfly_fused_2layer_portable(a, b, c, d, t_outer, t_inner_a, t_inner_b);
}

/// One layer over a lane-blocked block (`block.len() == 8 half` words, position `i` =
/// words `4i .. 4i + 4`): positions `i` and `i + half` butterflied with `twiddle` (the
/// row-pair kernel on the two halves: word `w` of the top half pairs word `w` of the
/// bottom half).
#[inline]
pub(super) fn blocked4_layer(block: &mut [F128], twiddle: F128) {
    debug_assert!(block.len().is_multiple_of(8));
    let (top, bot) = block.split_at_mut(block.len() / 2);
    butterfly_row_pair(top, bot, twiddle);
}

/// Two layers over a lane-blocked block of `4 quarter` positions: layer L pairs
/// quarters (1, 3) and (2, 4) with `t_outer`, layer L+1 pairs (1, 2) with `t_inner_a`
/// and (3, 4) with `t_inner_b`.
#[inline]
pub(super) fn blocked4_fused_2layer(block: &mut [F128], t_outer: F128, t_inner_a: F128, t_inner_b: F128) {
    debug_assert!(block.len().is_multiple_of(16));
    let quarter = block.len() / 4;
    let (h1, h2) = block.split_at_mut(2 * quarter);
    let (a, b) = h1.split_at_mut(quarter);
    let (c, d) = h2.split_at_mut(quarter);
    butterfly_fused_2layer(a, b, c, d, t_outer, t_inner_a, t_inner_b);
}

/// Fused three-layer (8-point) butterfly over one row group. Portable on
/// every target: the kernel is deliberately scalar-per-lane (compiler ILP —
/// see the fused-2 lane-batching regression note) and the x86 AVX-512 path
/// prefers fused-4, so this branch fires there only on remnant block sizes.
#[inline]
pub(super) fn butterfly_fused_3layer(
    rows: [&mut [F128]; 8],
    t0: F128,
    t1: &[F128; 2],
    t2: &[F128; 4],
) {
    butterfly_fused_3layer_portable(rows, t0, t1, t2);
}

/// Process one fused-three-layer row group (8 rows) across every interleaved NTT lane:
/// the remainder kernel for transforms whose layer count is not a multiple of four.
///
/// # Safety
/// The caller must ensure the 8 row slices selected by `r` are valid and
/// disjoint from any row group being processed concurrently.
#[allow(dead_code)]
#[inline]
pub(super) unsafe fn butterfly_fused_3layer_row(
    ptr: *mut F128,
    eighth: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 7],
) {
    #[cfg(target_arch = "x86_64")]
    if x86_avx512_clmul() {
        // SAFETY: target features are checked at run time; the caller owns the row
        // geometry and disjointness contract.
        unsafe { butterfly_fused_3layer_row_x86_64(ptr, eighth, num_ntts, r, twiddles) };
        return;
    }
    // SAFETY: forwarded caller contract.
    unsafe {
        butterfly_fused_3layer_row_portable(ptr, eighth, num_ntts, r, twiddles);
    }
}

/// Three layers over a lane-blocked block of `8 eighth` words (position `i` = words
/// `4i .. 4i + 4`): layer L pairs eighths (k, k+4) with `twiddles[0]`, layer L+1 pairs
/// (k, k+2) inside each half with `twiddles[1]`, `twiddles[2]`, layer L+2 pairs
/// (k, k+1) inside each quarter with `twiddles[3..7]`.
#[allow(dead_code)]
#[inline]
pub(super) fn blocked4_fused_3layer(block: &mut [F128], twiddles: &[F128; 7]) {
    debug_assert!(block.len().is_multiple_of(32));
    #[cfg(target_arch = "x86_64")]
    if x86_avx512_clmul() {
        // SAFETY: the runtime check guarantees the required target features.
        unsafe { blocked4_fused_3layer_x86_64(block, twiddles) };
        return;
    }
    blocked4_fused_3layer_portable(block, twiddles);
}

/// Process one fused-four-layer row group across every interleaved NTT lane.
///
/// # Safety
/// The caller must ensure the 16 row slices selected by `r` are valid and
/// disjoint from any row group being processed concurrently.
#[inline]
pub(super) unsafe fn butterfly_fused_4layer_row(
    ptr: *mut F128,
    sixteenth: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 15],
) {
    #[cfg(target_arch = "x86_64")]
    if x86_avx512_clmul() {
        // SAFETY: target features are checked at run time; the caller owns the row
        // geometry and disjointness contract.
        unsafe { butterfly_fused_4layer_row_x86_64(ptr, sixteenth, num_ntts, r, twiddles) };
        return;
    }
    // SAFETY: forwarded caller contract.
    unsafe {
        butterfly_fused_4layer_row_portable(ptr, sixteenth, num_ntts, r, twiddles);
    }
}

/// [`butterfly_fused_4layer_row`] reading the sixteen rows from `src` and writing the
/// results to the same rows of `dst` (the pass-2 first layer of the lane-blocked encode,
/// which then needs no separate copy of the block).
///
/// # Safety
/// The caller must ensure the 16 row slices selected by `r` are valid in both buffers,
/// that `dst` does not overlap `src`, and that no concurrent call touches `dst`.
#[inline]
pub(super) unsafe fn butterfly_fused_4layer_row_from(
    src: *const F128,
    dst: *mut F128,
    sixteenth: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 15],
) {
    #[cfg(target_arch = "x86_64")]
    if x86_avx512_clmul() {
        // SAFETY: target features are checked at run time; the caller owns the row
        // geometry and disjointness contract.
        unsafe { x86_64::butterfly_fused_4layer_row_from(src, dst, sixteenth, num_ntts, r, twiddles) };
        return;
    }
    // SAFETY: forwarded caller contract.
    unsafe {
        portable::butterfly_fused_4layer_row_from(src, dst, sixteenth, num_ntts, r, twiddles);
    }
}

/// In-place radix-8 (fused-3-layer) rows of one block; `zero_root` selects
/// the block-0 XOR-only spine. NEON-only entry (the ranked top pass).
///
/// # Safety
/// Row geometry valid; concurrent calls own disjoint row ranges.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) unsafe fn butterfly_fused_3layer_rows(
    ptr: *mut F128,
    eighth: usize,
    num_ntts: usize,
    row_start: usize,
    row_end: usize,
    twiddles: &[F128; 7],
    zero_root: bool,
) {
    // SAFETY: cfg gate guarantees PMULL; caller contract forwarded.
    unsafe {
        aarch64::butterfly_fused_3layer_rows(
            ptr, eighth, num_ntts, row_start, row_end, twiddles, zero_root,
        )
    }
}

/// Rate-1/2 dual-destination from-source radix-8 row (see the aarch64
/// module comment: staged L1 tiles + non-temporal row bursts). NEON-only.
///
/// # Safety
/// Geometry valid for all three pointers; disjoint row groups; `num_ntts`
/// even and ≤ 64; `t_zero[0]==t_zero[1]==t_zero[3]==0`.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) unsafe fn butterfly_fused_3layer_dual_from_src_row(
    src: *const F128,
    dst0: *mut F128,
    dst1: *mut F128,
    eighth: usize,
    num_ntts: usize,
    r: usize,
    t_zero: &[F128; 7],
    t_gen: &[F128; 7],
) {
    // SAFETY: cfg gate guarantees PMULL; caller contract forwarded.
    unsafe {
        aarch64::butterfly_fused_3layer_dual_from_src_row(
            src, dst0, dst1, eighth, num_ntts, r, t_zero, t_gen,
        )
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
pub(super) unsafe fn butterfly_neon_block(chunk: &mut [F128], twiddle: F128, half: usize) {
    // SAFETY: the cfg gate guarantees PMULL through the aes feature.
    unsafe { butterfly_block(chunk, twiddle, half) }
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
pub(super) unsafe fn butterfly_neon_block_pair(
    data: &mut [F128],
    base: usize,
    t_a: F128,
    t_b: F128,
) {
    // SAFETY: the cfg gate guarantees PMULL through the aes feature.
    unsafe { butterfly_block_pair(&mut data[base..base + 4], t_a, t_b) }
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
pub(super) unsafe fn butterfly_neon_block_pair_chunk(chunk: &mut [F128], t_a: F128, t_b: F128) {
    // SAFETY: the cfg gate guarantees PMULL through the aes feature.
    unsafe { butterfly_block_pair(chunk, t_a, t_b) }
}

/// agg-mldsa patch (PATCH.md): each x86 kernel available on this CPU
/// matches its portable counterpart bit for bit (lengths with and without a
/// four-lane remainder).
#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::{portable, x86_64, x86_avx512_clmul};
    use crate::{field::F128, test_rng::Rng};

    #[test]
    fn x86_kernels_match_portable() {
        if !x86_avx512_clmul() {
            eprintln!("skipped: no avx512f+vpclmulqdq on this CPU");
            return;
        }
        let mut rng = Rng::new(0xB07);
        for len in [1usize, 3, 4, 8, 13, 64] {
            let v: Vec<Vec<F128>> = (0..4).map(|_| rng.f128_vec(len)).collect();
            let t = [rng.f128(), rng.f128(), rng.f128()];

            let (mut p0, mut p1) = (v[0].clone(), v[1].clone());
            let (mut x0, mut x1) = (v[0].clone(), v[1].clone());
            portable::butterfly_row_pair(&mut p0, &mut p1, t[0]);
            // SAFETY: features checked above; equal lengths.
            unsafe { x86_64::butterfly_row_pair(&mut x0, &mut x1, t[0]) };
            assert_eq!((p0, p1), (x0, x1), "row_pair len={len}");

            let mut p = v.clone();
            let mut x = v.clone();
            let [pa, pb, pc, pd] = &mut p[..] else { unreachable!() };
            portable::butterfly_fused_2layer(pa, pb, pc, pd, t[0], t[1], t[2]);
            let [xa, xb, xc, xd] = &mut x[..] else { unreachable!() };
            // SAFETY: features checked above; equal lengths.
            unsafe { x86_64::butterfly_fused_2layer(xa, xb, xc, xd, t[0], t[1], t[2]) };
            assert_eq!(p, x, "fused_2layer len={len}");
        }
        // Fused-4 rows: a 16 x sixteenth x num_ntts block, every row group r.
        let tw: [F128; 15] = core::array::from_fn(|_| rng.f128());
        for (sixteenth, num_ntts) in [(1usize, 4usize), (2, 7), (4, 16)] {
            let block = rng.f128_vec(16 * sixteenth * num_ntts);
            let mut p = block.clone();
            let mut x = block.clone();
            for r in 0..sixteenth {
                // SAFETY: the buffers hold 16 * sixteenth rows of num_ntts lanes;
                // calls are sequential.
                unsafe {
                    portable::butterfly_fused_4layer_row(p.as_mut_ptr(), sixteenth, num_ntts, r, &tw);
                    x86_64::butterfly_fused_4layer_row(x.as_mut_ptr(), sixteenth, num_ntts, r, &tw);
                }
            }
            assert_eq!(p, x, "fused_4layer_row sixteenth={sixteenth} num_ntts={num_ntts}");
        }
        // Fused-3 rows and the blocked variant against the portable kernels.
        let tw3: [F128; 7] = core::array::from_fn(|_| rng.f128());
        for (eighth, num_ntts) in [(1usize, 4usize), (2, 7), (4, 16), (3, 5)] {
            let block = rng.f128_vec(8 * eighth * num_ntts);
            let mut p = block.clone();
            let mut x = block.clone();
            for r in 0..eighth {
                // SAFETY: the buffers hold 8 * eighth rows of num_ntts lanes;
                // calls are sequential.
                unsafe {
                    portable::butterfly_fused_3layer_row(p.as_mut_ptr(), eighth, num_ntts, r, &tw3);
                    x86_64::butterfly_fused_3layer_row(x.as_mut_ptr(), eighth, num_ntts, r, &tw3);
                }
            }
            assert_eq!(p, x, "fused_3layer_row eighth={eighth} num_ntts={num_ntts}");
        }
        // The row kernel equals the slice-based fused-3 kernel on 8 equal slices.
        for n in [1usize, 5, 16] {
            let block = rng.f128_vec(8 * n);
            let mut want = block.clone();
            {
                let mut it = want.chunks_mut(n);
                let rows: [&mut [F128]; 8] = core::array::from_fn(|_| it.next().unwrap());
                portable::butterfly_fused_3layer(
                    rows,
                    tw3[0],
                    &[tw3[1], tw3[2]],
                    &[tw3[3], tw3[4], tw3[5], tw3[6]],
                );
            }
            let mut got = block.clone();
            for r in 0..n {
                // SAFETY: 8 rows of n words, one lane; sequential calls.
                unsafe { portable::butterfly_fused_3layer_row(got.as_mut_ptr(), n, 1, r, &tw3) };
            }
            assert_eq!(want, got, "fused_3layer_row vs slices n={n}");
        }
        for words in [32usize, 64, 96, 256] {
            let block = rng.f128_vec(words);
            let mut p = block.clone();
            let mut x = block.clone();
            portable::blocked4_fused_3layer(&mut p, &tw3);
            // SAFETY: features checked above; length is a multiple of 32.
            unsafe { x86_64::blocked4_fused_3layer(&mut x, &tw3) };
            assert_eq!(p, x, "blocked4_fused_3layer words={words}");
        }
    }
}
