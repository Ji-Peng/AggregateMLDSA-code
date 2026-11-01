//! Two-pass encoding of interleaved messages, retaining pass-1 blocks for sparse openings.
use super::{AdditiveNttF128, kernels, lane_blocked_ok, log2_pow2};
use crate::field::F128;
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, ParallelIterator, ParallelSliceMut,
};
use std::cell::RefCell;
const TILE_LOG: usize = 14;
const MAX_TOP_LAYERS: usize = 8;
const MIN_ROWS_LOG: usize = 2;
const LANES_PER_GROUP: usize = 4;
thread_local! { static TILE: RefCell<Vec<F128>> = const { RefCell::new(Vec::new()) }; }
impl AdditiveNttF128 {
    /// Encode independent native interleaved transforms in two cache-sized passes.
    /// The gather callback fills consecutive message rows. When leaves are supplied,
    /// hash the final blocks in cache; discard retains the pass-1 state for openings.
    pub fn encode_blocked<G>(
        &self,
        codeword: &mut [F128],
        width: usize,
        groups: usize,
        log_d: usize,
        log_inv_rate: usize,
        gather: G,
        leaves: Option<&mut [crate::merkle::Hash]>,
        discard: bool,
    ) -> usize
    where
        G: Fn(usize, usize, &mut [F128]) + Sync,
    {
        let n = 1usize << log_d;
        let d = n >> log_inv_rate;
        assert!(
            width >= 4 && width.is_power_of_two(),
            "width must be a power of two >= 4"
        );
        assert!(log_d <= self.log_domain_size());
        assert!(
            lane_blocked_ok(log_d, log_inv_rate),
            "shape too small for the blocked encode"
        );
        assert_eq!(codeword.len(), groups * width * n);
        assert!(
            !discard || leaves.is_some(),
            "discard requires fused hashing"
        );
        if let Some(hashes) = &leaves {
            assert_eq!(groups, 1);
            assert_eq!(hashes.len(), n);
            assert!(width <= 64);
        }
        let group_len = width * n;
        let r = log_inv_rate;
        let avail = log_d - r;
        let (tile_log, max_top) = (TILE_LOG, MAX_TOP_LAYERS);
        // Positions per tile at this width (the tile budget is 2^TILE_LOG lines of 64 B).
        let tile_log = (tile_log + 2)
            .saturating_sub(log2_pow2(width))
            .max(MIN_ROWS_LOG + 1);
        let b = max_top
            .min(avail - MIN_ROWS_LOG)
            .min(tile_log - MIN_ROWS_LOG)
            .max(1);
        let low = avail - b;
        let a = low.min(tile_log - b);
        let (rows, blocks) = (1usize << a, 1usize << b);
        let tile_width = rows * width;
        let tiles_per_group = 1usize << (r + low - a);
        let tiles_per_rep = 1usize << (low - a);
        let timing = std::env::var_os("FLOCK_COMMIT_TIMING").is_some();
        let t_pass = std::time::Instant::now();

        // Pass 1: gather, layers r .. r + b, store.
        let base = Base(codeword.as_mut_ptr());
        (0..groups * tiles_per_group)
            .into_par_iter()
            .for_each(|idx| {
                let g = idx / tiles_per_group;
                let rem = idx % tiles_per_group;
                let rep = rem / tiles_per_rep;
                let row0 = (rem % tiles_per_rep) << a;
                TILE.with_borrow_mut(|tile| {
                    if tile.len() < blocks * tile_width {
                        tile.resize(blocks * tile_width, F128::ZERO);
                    }
                    let tile = &mut tile[..blocks * tile_width];
                    for j in 0..blocks {
                        gather(
                            g,
                            (j << low) + row0,
                            &mut tile[j * tile_width..(j + 1) * tile_width],
                        );
                    }
                    self.blocked_tile_layers(tile, b, tile_width, r, rep);
                    for j in 0..blocks {
                        let p = rep * d + (j << low) + row0;
                        // SAFETY: positions [p, p + rows) of group g belong to this tile only
                        // (tiles partition the groups by (rep, row0) and blocks by j) and lie
                        // inside `codeword` (length asserted above).
                        unsafe {
                            stream_copy(
                                base.at(g * group_len + p * width),
                                &tile[j * tile_width..(j + 1) * tile_width],
                            );
                        }
                    }
                });
                // Publish this worker's non-temporal writes before pass 2 reads them.
                sfence();
            });
        if timing {
            eprintln!(
                "[commit-timing] blocked encode width {width}: pass 1 (b={b} a={a}) {:.2} ms",
                t_pass.elapsed().as_secs_f64() * 1e3
            );
        }
        let t_pass = std::time::Instant::now();

        // Pass 2: layers r + b .. log_d, one layer-(r + b) block at a time.
        let sub = 1usize << low;
        let subs_per_group = n / sub;
        if let Some(leaves) = leaves {
            codeword
                .par_chunks_mut(sub * width)
                .zip(leaves.par_chunks_mut(sub))
                .enumerate()
                .for_each(|(s, (block, hashes))| {
                    let mut scratch = if discard { block.to_vec() } else { Vec::new() };
                    let output = if discard {
                        scratch.as_mut_slice()
                    } else {
                        block
                    };
                    self.blocked_deep_layers(output, width, s * sub, r + b, log_d);
                    crate::merkle::GroupLeafHasher::new(width, crate::merkle::HashKind::Blake3)
                        .unwrap()
                        .hash_interleaved(output, hashes);
                });
        } else {
            codeword
                .par_chunks_mut(sub * width)
                .enumerate()
                .for_each(|(s, block)| {
                    let p0 = (s % subs_per_group) * sub;
                    self.blocked_deep_layers(block, width, p0, r + b, log_d);
                });
        }
        if timing {
            eprintln!(
                "[commit-timing] blocked encode width {width}: pass 2 (low={low}) {:.2} ms",
                t_pass.elapsed().as_secs_f64() * 1e3
            );
        }
        low
    }

    fn blocked_fused4_twiddles_prefix(&self, layer: usize, block: usize) -> [F128; 7] {
        let mut tw = [F128::ZERO; 7];
        for k in 0..3 {
            for j in 0..1usize << k {
                tw[(1 << k) - 1 + j] = self.twiddle(layer + k, (block << k) + j);
            }
        }
        tw
    }

    /// The fifteen twiddles of the fused four-layer butterfly at `(layer, block)`.
    fn blocked_fused4_twiddles(&self, layer: usize, block: usize) -> [F128; 15] {
        let mut tw = [F128::ZERO; 15];
        tw[0] = self.twiddle(layer, block);
        for s in 0..2 {
            tw[1 + s] = self.twiddle(layer + 1, 2 * block + s);
        }
        for s in 0..4 {
            tw[3 + s] = self.twiddle(layer + 2, 4 * block + s);
        }
        for s in 0..8 {
            tw[7 + s] = self.twiddle(layer + 3, 8 * block + s);
        }
        tw
    }

    /// Layers `r .. r + b` on a tile of `2^b` virtual positions of `width` words: virtual
    /// block `vb` at virtual layer `k` is global block `(rep << k) + vb` of layer `r + k`.
    fn blocked_tile_layers(&self, tile: &mut [F128], b: usize, width: usize, r: usize, rep: usize) {
        let mut k = 0;
        while k < b {
            let vblocks = 1usize << k;
            let vsize = (1usize << (b - k)) * width;
            if b - k >= 4 {
                let sixteenth = 1usize << (b - k - 4);
                for vb in 0..vblocks {
                    let tw = self.blocked_fused4_twiddles(r + k, (rep << k) + vb);
                    let ptr = tile[vb * vsize..].as_mut_ptr();
                    for rr in 0..sixteenth {
                        // SAFETY: the sixteen rows of row group rr lie in this block.
                        unsafe {
                            kernels::butterfly_fused_4layer_row(ptr, sixteenth, width, rr, &tw)
                        };
                    }
                }
                k += 4;
            } else if b - k == 3 {
                let eighth = 1usize << (b - k - 3);
                for vb in 0..vblocks {
                    let tw = self.blocked_fused4_twiddles_prefix(r + k, (rep << k) + vb);
                    let ptr = tile[vb * vsize..].as_mut_ptr();
                    for rr in 0..eighth {
                        // SAFETY: all eight rows lie in this tile block.
                        unsafe { kernels::butterfly_fused_3layer_row(ptr, eighth, width, rr, &tw) };
                    }
                }
                k += 3;
            } else if b - k >= 2 {
                let quarter = (1usize << (b - k - 2)) * width;
                for vb in 0..vblocks {
                    let gb = (rep << k) + vb;
                    let (t0, ta, tb) = (
                        self.twiddle(r + k, gb),
                        self.twiddle(r + k + 1, 2 * gb),
                        self.twiddle(r + k + 1, 2 * gb + 1),
                    );
                    let block = &mut tile[vb * vsize..(vb + 1) * vsize];
                    let (h1, h2) = block.split_at_mut(2 * quarter);
                    let (q1, q2) = h1.split_at_mut(quarter);
                    let (q3, q4) = h2.split_at_mut(quarter);
                    for o in (0..quarter).step_by(width) {
                        kernels::butterfly_fused_2layer(
                            &mut q1[o..o + width],
                            &mut q2[o..o + width],
                            &mut q3[o..o + width],
                            &mut q4[o..o + width],
                            t0,
                            ta,
                            tb,
                        );
                    }
                }
                k += 2;
            } else {
                let half = (1usize << (b - k - 1)) * width;
                for vb in 0..vblocks {
                    let tw = self.twiddle(r + k, (rep << k) + vb);
                    let block = &mut tile[vb * vsize..(vb + 1) * vsize];
                    let (top, bot) = block.split_at_mut(half);
                    for o in (0..half).step_by(width) {
                        kernels::butterfly_row_pair(
                            &mut top[o..o + width],
                            &mut bot[o..o + width],
                            tw,
                        );
                    }
                }
                k += 1;
            }
        }
    }

    /// Layers `first .. log_d` on one layer-`first` block of positions of `width` words
    /// starting at position `p0` of its transform.
    pub(crate) fn blocked_deep_layers(
        &self,
        block: &mut [F128],
        width: usize,
        p0: usize,
        first: usize,
        log_d: usize,
    ) {
        let sub = block.len() / width;
        let mut l = first;
        while l < log_d {
            let bs = 1usize << (log_d - l);
            let nb = sub / bs;
            let gb0 = p0 >> (log_d - l);
            let len = bs * width;
            if log_d - l >= 4 {
                let sixteenth = bs >> 4;
                for lb in 0..nb {
                    let tw = self.blocked_fused4_twiddles(l, gb0 + lb);
                    let ptr = block[lb * len..].as_mut_ptr();
                    for rr in 0..sixteenth {
                        // SAFETY: the sixteen rows of row group rr lie in this block.
                        unsafe {
                            kernels::butterfly_fused_4layer_row(ptr, sixteenth, width, rr, &tw)
                        };
                    }
                }
                l += 4;
            } else if log_d - l == 3 {
                let eighth = bs >> 3;
                for lb in 0..nb {
                    let tw = self.blocked_fused4_twiddles_prefix(l, gb0 + lb);
                    let ptr = block[lb * len..].as_mut_ptr();
                    for rr in 0..eighth {
                        // SAFETY: all eight rows lie in this deep block.
                        unsafe { kernels::butterfly_fused_3layer_row(ptr, eighth, width, rr, &tw) };
                    }
                }
                l += 3;
            } else if log_d - l >= 2 {
                for lb in 0..nb {
                    let gb = gb0 + lb;
                    let (t0, ta, tb) = (
                        self.twiddle(l, gb),
                        self.twiddle(l + 1, 2 * gb),
                        self.twiddle(l + 1, 2 * gb + 1),
                    );
                    let blk = &mut block[lb * len..(lb + 1) * len];
                    if width == LANES_PER_GROUP {
                        kernels::blocked4_fused_2layer(blk, t0, ta, tb);
                    } else {
                        let quarter = len / 4;
                        let (h1, h2) = blk.split_at_mut(2 * quarter);
                        let (q1, q2) = h1.split_at_mut(quarter);
                        let (q3, q4) = h2.split_at_mut(quarter);
                        for o in (0..quarter).step_by(width) {
                            kernels::butterfly_fused_2layer(
                                &mut q1[o..o + width],
                                &mut q2[o..o + width],
                                &mut q3[o..o + width],
                                &mut q4[o..o + width],
                                t0,
                                ta,
                                tb,
                            );
                        }
                    }
                }
                l += 2;
            } else {
                for lb in 0..nb {
                    let tw = self.twiddle(l, gb0 + lb);
                    let blk = &mut block[lb * len..(lb + 1) * len];
                    if width == LANES_PER_GROUP {
                        kernels::blocked4_layer(blk, tw);
                    } else {
                        let (top, bot) = blk.split_at_mut(len / 2);
                        for o in (0..len / 2).step_by(width) {
                            kernels::butterfly_row_pair(
                                &mut top[o..o + width],
                                &mut bot[o..o + width],
                                tw,
                            );
                        }
                    }
                }
                l += 1;
            }
        }
    }
}

struct Base(*mut F128);
// SAFETY: pass-1 tasks write disjoint ranges, partitioned by group, replica and row.
unsafe impl Sync for Base {}
impl Base {
    fn at(&self, off: usize) -> *mut F128 {
        unsafe { self.0.add(off) }
    }
}
unsafe fn stream_copy(dst: *mut F128, src: &[F128]) {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::simd::has(crate::simd::AVX512F) {
            // SAFETY: forwarded contract.
            unsafe { x86::stream_copy(dst, src) };
            return;
        }
        // Below AVX-512: 16-byte streaming stores (SSE2, every x86-64 CPU), so
        // the codeword written here costs no read-for-ownership either.
        use core::arch::x86_64::{__m128i, _mm_loadu_si128, _mm_stream_si128};
        for (i, v) in src.iter().enumerate() {
            // SAFETY: element i of both lies inside the ranges of the contract;
            // F128 is 16 bytes and 16-byte aligned.
            unsafe {
                let x = _mm_loadu_si128(core::ptr::from_ref(v).cast::<__m128i>());
                _mm_stream_si128(dst.add(i).cast::<__m128i>(), x);
            }
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    // SAFETY: forwarded contract.
    unsafe {
        core::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len())
    };
}

/// Order this thread's streaming stores before later reads.
fn sfence() {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: a plain store fence (SSE, every x86-64 CPU).
    unsafe {
        core::arch::x86_64::_mm_sfence()
    };
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use crate::field::F128;
    use core::arch::x86_64::{_mm512_loadu_si512, _mm512_stream_si512};
    /// Copy `src` to `dst` with 64-byte streaming stores between the aligned boundaries
    /// and plain copies of the unaligned head and tail.
    ///
    /// # Safety
    /// `dst .. dst + src.len()` is writable and disjoint from `src`; `avx512f` available.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn stream_copy(dst: *mut F128, src: &[F128]) {
        let n = src.len();
        // F128 is 16-byte aligned: the head is 0..3 elements.
        let mis = (dst as usize / 16) % 4;
        let head = ((4 - mis) % 4).min(n);
        // SAFETY: caller contract; every access is inside `[0, n)` of both.
        unsafe {
            core::ptr::copy_nonoverlapping(src.as_ptr(), dst, head);
            let mut i = head;
            while i + 4 <= n {
                let v = _mm512_loadu_si512(src.as_ptr().add(i).cast());
                _mm512_stream_si512(dst.add(i).cast(), v);
                i += 4;
            }
            core::ptr::copy_nonoverlapping(src.as_ptr().add(i), dst.add(i), n - i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        merkle::{HashKind, hash_leaf},
        simd::{test_masks, with_mask},
        test_rng::Rng,
    };

    #[test]
    fn blocked_hash_discard_matches_interleaved() {
        let mut rng = Rng::new(0xb10c);
        for (log_d, r, width) in [(10, 2, 8), (13, 4, 16), (15, 0, 16), (18, 0, 16)] {
            let n = 1usize << log_d;
            let poly = rng.f128_vec((n >> r) * width);
            let ntt = AdditiveNttF128::standard(log_d);
            let mut want = poly.repeat(1 << r);
            ntt.forward_transform_interleaved_from_layer(&mut want, width, r);
            for mask in test_masks() {
                with_mask(mask, || {
                    for discard in [false, true] {
                        let mut got = rng.f128_vec(n * width);
                        let mut hashes = vec![[0; 32]; n];
                        let low = ntt.encode_blocked(
                            &mut got,
                            width,
                            1,
                            log_d,
                            r,
                            |_, p, dst| {
                                dst.copy_from_slice(&poly[p * width..p * width + dst.len()])
                            },
                            Some(&mut hashes),
                            discard,
                        );
                        if discard {
                            let sub = 1usize << low;
                            for (s, block) in got.chunks_mut(sub * width).enumerate() {
                                ntt.blocked_deep_layers(block, width, s * sub, log_d - low, log_d);
                            }
                        }
                        assert_eq!(got, want);
                        for (row, hash) in got.chunks_exact(width).zip(hashes) {
                            let bytes = unsafe {
                                std::slice::from_raw_parts(
                                    row.as_ptr().cast::<u8>(),
                                    row.len() * 16,
                                )
                            };
                            assert_eq!(hash, hash_leaf(bytes, HashKind::Blake3));
                        }
                    }
                });
            }
        }
    }
}
