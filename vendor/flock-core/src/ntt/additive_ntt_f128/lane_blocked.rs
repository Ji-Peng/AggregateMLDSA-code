//! Lane-blocked encode of a lane-major message (agg-mldsa patch, PATCH.md section 8).
//!
//! The level-0 codeword of the lane-major commit is stored in groups of four lanes:
//! group `g` holds lanes `4g .. 4g + 4` as `codeword[4 g n .. 4 (g + 1) n]`, position
//! `p` of the group being the four consecutive words `[4 (g n + p), 4 (g n + p) + 4)`,
//! one 64-byte line. Lanes past the committed count in the last group are
//! zero and ride along. Every butterfly of the interleaved NTT then acts on whole,
//! aligned lines, and the transform runs in two passes over each group instead of one
//! full pass per three or four layers:
//!
//! 1. Pass 1 reads the four message lanes once. The first `r = log_inv_rate` layers of
//!    the zero-padded message are copies (after them the state is `2^r` replicas of the
//!    message), so the pass starts at layer `r`. A tile takes `2^B` blocks of layer `r`
//!    at `2^A` consecutive rows (the positions `rep d + (j << low) + row0 + i`, `low =
//!    log_d - r - B`), gathers their words from the lanes, applies layers `r .. r + B`
//!    inside the tile (the partner of position `p` at layer `l` is `p ^ 2^(log_d - 1 -
//!    l)`, so these layers pair blocks `j` at equal rows: the tile is an interleaved
//!    transform of `2^B` positions whose width is `2^A` rows of four lanes) and writes
//!    each block's rows to the codeword with streaming stores.
//! 2. Pass 2 applies the remaining layers `r + B .. log_d` to one block of layer `r + B`
//!    at a time: `2^low` consecutive positions that stay in cache.
//!
//! Traffic per group: the message lanes once, the group written once and read and
//! written once more, against the interleaved transform's one read and write per fused
//! pass plus the replicate fill. Output values are those of the interleaved transform of
//! the same lanes (the Merkle rows are gathered from the groups, see
//! `merkle::merkle_tree_lane_blocked`).
//!
//! The hashed variant ([`AdditiveNttF128::encode_lane_blocked_hashed`]) takes two more
//! layers in pass 1 and runs pass 2 position-major: one task takes a position range of
//! every group and absorbs each group's block into the BLAKE3 leaves of the range right
//! after transforming it (a row's lane group is one message block of its leaf,
//! `merkle::GroupLeafHasher`), so the Merkle pass no longer reads the codeword. It can also
//! transform in a per-thread copy and write nothing back: the codeword then keeps the
//! pass-1 state, and the opening recomputes the few queried positions from it.

use std::cell::RefCell;

use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator, ParallelSliceMut};

use super::{AdditiveNttF128, kernels, log2_pow2};
use crate::{
    field::F128,
    merkle::{GroupLeafHasher, Hash},
};

/// Lanes per group: one 64-byte line of F128 words.
pub const LANES_PER_GROUP: usize = 4;
/// Positions per pass-1 tile (`2^14` lines = 1 MiB).
const TILE_LOG: usize = 14;
/// Layers applied in pass 1 at most (two fused-four steps).
const MAX_TOP_LAYERS: usize = 8;
/// Pass-1 layers when pass 2 hashes the leaves: two more layers in the tile leave pass-2
/// blocks of a quarter the size, so one position range of every group fits in L2.
const FUSED_TOP_LAYERS: usize = 10;
/// Rows per tile block at least: the row gather transposes four positions at a time.
const MIN_ROWS_LOG: usize = 2;

/// Whether [`AdditiveNttF128::encode_lane_blocked`] supports a transform of `2^log_d`
/// positions at rate `1 / 2^log_inv_rate` (pass 1 needs four layers over at least four
/// rows).
pub fn lane_blocked_ok(log_d: usize, log_inv_rate: usize) -> bool {
    log_d >= log_inv_rate + MIN_ROWS_LOG + 4
}

thread_local! {
    static TILE: RefCell<Vec<F128>> = const { RefCell::new(Vec::new()) };
    /// Pass 2 with hashing: the block copy (when the final codeword is not kept) and the
    /// leaf chaining state of one position range.
    static PASS2: RefCell<(Vec<F128>, Vec<u32>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}

/// The pass split chosen by pass 1.
struct Pass1 {
    /// Layers applied in pass 1.
    b: usize,
    /// Layers left for pass 2 (log2 of its block length in positions).
    low: usize,
    /// Groups holding a live lane.
    live_groups: usize,
}

impl AdditiveNttF128 {
    /// Encode the lane-major message `q` (lane `l` is `q[l d .. (l + 1) d]`, lanes `live
    /// .. t` are zero, `t` lanes committed) into `codeword` in the lane-blocked layout
    /// with `2^log_inv_rate` positions per message word. `codeword` must hold `4 ceil(t /
    /// 4) d 2^log_inv_rate` words; its contents may be uninitialised.
    pub fn encode_lane_blocked(
        &self,
        codeword: &mut [F128],
        q: &[F128],
        t: usize,
        live: usize,
        d: usize,
        log_inv_rate: usize,
    ) {
        let n = d << log_inv_rate;
        let log_d = log2_pow2(n);
        let Pass1 { b, low, live_groups } = self.lane_blocked_pass1(codeword, q, t, live, d, log_inv_rate, MAX_TOP_LAYERS);
        if live_groups == 0 {
            return;
        }
        let r = log_inv_rate;
        let timing = std::env::var_os("FLOCK_COMMIT_TIMING").is_some();
        let t_pass = std::time::Instant::now();
        let group_len = LANES_PER_GROUP * n;

        // Pass 2: layers r + b .. log_d, one layer-(r + b) block at a time.
        let sub = 1usize << low;
        let subs_per_group = n / sub;
        codeword[..live_groups * group_len]
            .par_chunks_mut(sub * LANES_PER_GROUP)
            .enumerate()
            .for_each(|(s, block)| {
                let p0 = (s % subs_per_group) * sub;
                self.deep_layers(block, p0, r + b, log_d, false);
            });
        if timing {
            eprintln!("[commit-timing] lane-blocked pass 2 (low={low}): {:.2} ms", t_pass.elapsed().as_secs_f64() * 1e3);
        }
    }

    /// [`Self::encode_lane_blocked`] with the Merkle leaves hashed inside pass 2: one task
    /// takes a position range of every group, transforms each group's block through the
    /// remaining layers while it is in cache and absorbs it into the range's leaves
    /// (`leaves[p]` = the chaining value of row `p`, `leaves.len()` = the positions per
    /// group).
    ///
    /// With `keep_final` the transformed blocks are written back, so `codeword` ends as
    /// [`Self::encode_lane_blocked`] leaves it. Without it they are transformed in a
    /// per-thread copy and dropped after hashing: `codeword` keeps the pass-1 state, in
    /// which layers from the returned index on are still to be applied
    /// ([`Self::lane_blocked_finish_position`] recomputes any position from it,
    /// [`Self::lane_blocked_finish`] the whole codeword).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_lane_blocked_hashed(
        &self,
        codeword: &mut [F128],
        q: &[F128],
        t: usize,
        live: usize,
        d: usize,
        log_inv_rate: usize,
        hasher: &GroupLeafHasher,
        leaves: &mut [Hash],
        keep_final: bool,
    ) -> usize {
        let n = d << log_inv_rate;
        let log_d = log2_pow2(n);
        let groups = t.div_ceil(LANES_PER_GROUP);
        assert_eq!(hasher.groups(), groups);
        assert_eq!(leaves.len(), n);
        let Pass1 { b, low, live_groups } =
            self.lane_blocked_pass1(codeword, q, t, live, d, log_inv_rate, FUSED_TOP_LAYERS);
        let first = log_inv_rate + b;
        let timing = std::env::var_os("FLOCK_COMMIT_TIMING").is_some();
        let t_pass = std::time::Instant::now();
        let sub = 1usize << low;
        let words = sub * LANES_PER_GROUP;
        let state_len = GroupLeafHasher::state_len(sub);
        let base = Base(codeword.as_mut_ptr());
        leaves.par_chunks_mut(sub).enumerate().for_each(|(s, out)| {
            let p0 = s * sub;
            PASS2.with_borrow_mut(|(scratch, state)| {
                if state.len() < state_len {
                    state.resize(state_len, 0);
                }
                if !keep_final && scratch.len() < words {
                    scratch.resize(words, F128::ZERO);
                }
                for g in 0..groups {
                    let at = base.at((g * n + p0) * LANES_PER_GROUP);
                    if g >= live_groups {
                        // SAFETY: a dead group is zero (pass 1) and no task writes it; the
                        // range lies inside `codeword`.
                        let lines = unsafe { core::slice::from_raw_parts(at, words) };
                        hasher.absorb(state, g, lines);
                    } else if keep_final {
                        // SAFETY: positions [p0, p0 + sub) of group g belong to this task only
                        // (tasks partition the positions) and lie inside `codeword`.
                        let block = unsafe { core::slice::from_raw_parts_mut(at, words) };
                        self.deep_layers(block, p0, first, log_d, true);
                        hasher.absorb(state, g, block);
                    } else {
                        // SAFETY: no task writes `codeword` in this mode; the range lies
                        // inside it.
                        let src = unsafe { core::slice::from_raw_parts(at, words) };
                        let block = &mut scratch[..words];
                        block.copy_from_slice(src);
                        self.deep_layers(block, p0, first, log_d, true);
                        hasher.absorb(state, g, block);
                    }
                }
                hasher.finish(state, out);
            });
        });
        if timing {
            eprintln!(
                "[commit-timing] lane-blocked pass 2 + leaves (low={low}, keep={keep_final}): {:.2} ms",
                t_pass.elapsed().as_secs_f64() * 1e3
            );
        }
        first
    }

    /// Position `pos` (its four words) after layers `first .. log_d` of the lane-blocked
    /// block `block` (positions `p0 .. p0 + 2^(log_d - first)` of one group, state after
    /// layer `first - 1`, as [`Self::encode_lane_blocked_hashed`] leaves the codeword
    /// without `keep_final`); `work` is scratch. Equal to that position after the full
    /// layers ([`Self::lane_blocked_finish`]), at a fraction of the work: a layer pairs the
    /// halves of each block and the later layers stay inside one half, so only the half
    /// holding `pos` is computed at each layer (one product per pair, `2^(log_d - first)
    /// - 1` products per word in all).
    pub fn lane_blocked_finish_position(
        &self,
        block: &[F128],
        p0: usize,
        first: usize,
        log_d: usize,
        pos: usize,
        work: &mut Vec<F128>,
    ) -> [F128; LANES_PER_GROUP] {
        let sub = 1usize << (log_d - first);
        assert_eq!(block.len(), sub * LANES_PER_GROUP);
        assert!(p0.is_multiple_of(sub) && (p0..p0 + sub).contains(&pos));
        work.clear();
        work.extend_from_slice(block);
        // `work[..len * 4]` holds positions `start .. start + len` after layer `l - 1`.
        let (mut start, mut len) = (p0, sub);
        for l in first..log_d {
            let half = len / 2;
            let tw = self.twiddle(l, start >> (log_d - l));
            let top = pos < start + half;
            for i in 0..half * LANES_PER_GROUP {
                let (u, v) = (work[i], work[i + half * LANES_PER_GROUP]);
                // The butterfly: top u + v t, bottom v + (u + v t).
                let new_u = u + v * tw;
                work[i] = if top { new_u } else { v + new_u };
            }
            if !top {
                start += half;
            }
            len = half;
        }
        debug_assert_eq!(start, pos);
        core::array::from_fn(|k| work[k])
    }

    /// Layers `first .. log_d` on every block of a lane-blocked codeword of `n = 2^log_d`
    /// positions per group, in parallel.
    pub fn lane_blocked_finish(&self, codeword: &mut [F128], n: usize, first: usize) {
        let log_d = log2_pow2(n);
        let sub = 1usize << (log_d - first);
        assert!(codeword.len().is_multiple_of(LANES_PER_GROUP * n));
        codeword.par_chunks_mut(sub * LANES_PER_GROUP).enumerate().for_each(|(s, block)| {
            self.deep_layers(block, (s % (n / sub)) * sub, first, log_d, true);
        });
    }

    /// Pass 1 (gather, layers `r .. r + b`, streaming store; `b` at most `max_top` unless
    /// `FLOCK_LB_TOP` sets it) and the zero fill of the dead groups.
    #[allow(clippy::too_many_arguments)]
    fn lane_blocked_pass1(
        &self,
        codeword: &mut [F128],
        q: &[F128],
        t: usize,
        live: usize,
        d: usize,
        log_inv_rate: usize,
        max_top: usize,
    ) -> Pass1 {
        let n = d << log_inv_rate;
        let log_d = log2_pow2(n);
        let groups = t.div_ceil(LANES_PER_GROUP);
        assert!(log_d <= self.log_domain_size());
        assert!(lane_blocked_ok(log_d, log_inv_rate), "shape too small for the lane-blocked encode");
        assert_eq!(codeword.len(), groups * LANES_PER_GROUP * n);
        assert!(live <= t && q.len() >= live * d);
        let group_len = LANES_PER_GROUP * n;
        let live_groups = live.div_ceil(LANES_PER_GROUP);
        let r = log_inv_rate;
        let avail = log_d - r;
        let env = |k: &str, dflt: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(dflt);
        let (tile_log, max_top) = (env("FLOCK_LB_TILE_LOG", TILE_LOG), env("FLOCK_LB_TOP", max_top));
        let b = max_top.min(avail - MIN_ROWS_LOG);
        let low = avail - b;
        // Groups of dead lanes: their codeword is zero.
        codeword[live_groups * group_len..]
            .par_chunks_mut(1 << 16)
            .for_each(|c| c.fill(F128::ZERO));
        if live_groups == 0 {
            return Pass1 { b, low, live_groups };
        }
        let a = low.min(tile_log - b);
        let timing = std::env::var_os("FLOCK_COMMIT_TIMING").is_some();
        let t_pass = std::time::Instant::now();
        let (rows, blocks) = (1usize << a, 1usize << b);
        let width = rows * LANES_PER_GROUP;
        let tiles_per_group = 1usize << (r + low - a);
        let tiles_per_rep = 1usize << (low - a);

        // Pass 1: gather, layers r .. r + b, store.
        let base = Base(codeword.as_mut_ptr());
        (0..live_groups * tiles_per_group).into_par_iter().for_each(|idx| {
            let g = idx / tiles_per_group;
            let rem = idx % tiles_per_group;
            let rep = rem / tiles_per_rep;
            let row0 = (rem % tiles_per_rep) << a;
            TILE.with_borrow_mut(|tile| {
                if tile.len() < blocks * width {
                    tile.resize(blocks * width, F128::ZERO);
                }
                let tile = &mut tile[..blocks * width];
                for j in 0..blocks {
                    let m0 = (j << low) + row0;
                    let lane = |k: usize| {
                        let l = g * LANES_PER_GROUP + k;
                        (l < live).then(|| &q[l * d + m0..l * d + m0 + rows])
                    };
                    gather_rows(&mut tile[j * width..(j + 1) * width], [lane(0), lane(1), lane(2), lane(3)]);
                }
                self.tile_layers(tile, b, width, r, rep);
                for j in 0..blocks {
                    let p = rep * d + (j << low) + row0;
                    // SAFETY: positions [p, p + rows) of group g belong to this tile only
                    // (tiles partition the groups by (rep, row0) and blocks by j) and lie
                    // inside `codeword` (length asserted above).
                    unsafe {
                        stream_copy(base.at((g * n + p) * LANES_PER_GROUP), &tile[j * width..(j + 1) * width]);
                    }
                }
                // This worker's streaming stores reach memory before the task completes.
                sfence();
            });
        });
        sfence();
        if timing {
            eprintln!("[commit-timing] lane-blocked pass 1 (b={b} a={a}): {:.2} ms", t_pass.elapsed().as_secs_f64() * 1e3);
        }
        Pass1 { b, low, live_groups }
    }

    /// The seven twiddles of the fused three-layer butterfly at `(layer, block)`.
    fn fused3_twiddles(&self, layer: usize, block: usize) -> [F128; 7] {
        let mut tw = [F128::ZERO; 7];
        tw[0] = self.twiddle(layer, block);
        for s in 0..2 {
            tw[1 + s] = self.twiddle(layer + 1, 2 * block + s);
        }
        for s in 0..4 {
            tw[3 + s] = self.twiddle(layer + 2, 4 * block + s);
        }
        tw
    }

    /// The fifteen twiddles of the fused four-layer butterfly at `(layer, block)`.
    fn fused4_twiddles(&self, layer: usize, block: usize) -> [F128; 15] {
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
    fn tile_layers(&self, tile: &mut [F128], b: usize, width: usize, r: usize, rep: usize) {
        let mut k = 0;
        while k < b {
            let vblocks = 1usize << k;
            let vsize = (1usize << (b - k)) * width;
            if b - k >= 4 {
                let sixteenth = 1usize << (b - k - 4);
                for vb in 0..vblocks {
                    let tw = self.fused4_twiddles(r + k, (rep << k) + vb);
                    let ptr = tile[vb * vsize..].as_mut_ptr();
                    for rr in 0..sixteenth {
                        // SAFETY: the sixteen rows of row group rr lie in this block.
                        unsafe { kernels::butterfly_fused_4layer_row(ptr, sixteenth, width, rr, &tw) };
                    }
                }
                k += 4;
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
                        kernels::butterfly_row_pair(&mut top[o..o + width], &mut bot[o..o + width], tw);
                    }
                }
                k += 1;
            }
        }
    }

    /// Layers `first .. log_d` on one layer-`first` block of lane-blocked positions
    /// starting at position `p0` of its group. With `fused3` a remainder of three layers
    /// takes one fused-three pass instead of a fused-two and a single-layer pass (the same
    /// values).
    fn deep_layers(&self, block: &mut [F128], p0: usize, first: usize, log_d: usize, fused3: bool) {
        let sub = block.len() / LANES_PER_GROUP;
        let mut l = first;
        while l < log_d {
            let bs = 1usize << (log_d - l);
            let nb = sub / bs;
            let gb0 = p0 >> (log_d - l);
            let len = bs * LANES_PER_GROUP;
            if log_d - l >= 4 {
                let sixteenth = bs >> 4;
                for lb in 0..nb {
                    let tw = self.fused4_twiddles(l, gb0 + lb);
                    let ptr = block[lb * len..].as_mut_ptr();
                    for rr in 0..sixteenth {
                        // SAFETY: the sixteen rows of row group rr lie in this block.
                        unsafe { kernels::butterfly_fused_4layer_row(ptr, sixteenth, LANES_PER_GROUP, rr, &tw) };
                    }
                }
                l += 4;
            } else if fused3 && log_d - l == 3 {
                for lb in 0..nb {
                    let tw = self.fused3_twiddles(l, gb0 + lb);
                    kernels::blocked4_fused_3layer(&mut block[lb * len..(lb + 1) * len], &tw);
                }
                l += 3;
            } else if log_d - l >= 2 {
                for lb in 0..nb {
                    let gb = gb0 + lb;
                    kernels::blocked4_fused_2layer(
                        &mut block[lb * len..(lb + 1) * len],
                        self.twiddle(l, gb),
                        self.twiddle(l + 1, 2 * gb),
                        self.twiddle(l + 1, 2 * gb + 1),
                    );
                }
                l += 2;
            } else {
                for lb in 0..nb {
                    kernels::blocked4_layer(&mut block[lb * len..(lb + 1) * len], self.twiddle(l, gb0 + lb));
                }
                l += 1;
            }
        }
    }
}

/// The codeword base pointer shared by the pass-1 tasks (each writes its own positions).
struct Base(*mut F128);
// SAFETY: the tasks write disjoint position ranges (see `encode_lane_blocked`).
unsafe impl Sync for Base {}
impl Base {
    fn at(&self, off: usize) -> *mut F128 {
        self.0.wrapping_add(off)
    }
}

/// `dst[4 i + k] = lanes[k][i]` (zero for an absent lane) for `i < dst.len() / 4`.
fn gather_rows(dst: &mut [F128], lanes: [Option<&[F128]>; 4]) {
    let rows = dst.len() / LANES_PER_GROUP;
    debug_assert!(rows % 4 == 0 && lanes.iter().flatten().all(|l| l.len() == rows));
    for (k, lane) in lanes.iter().enumerate() {
        match lane {
            Some(src) => {
                for (i, &v) in src.iter().enumerate() {
                    dst[i * LANES_PER_GROUP + k] = v;
                }
            }
            None => {
                for i in 0..rows {
                    dst[i * LANES_PER_GROUP + k] = F128::ZERO;
                }
            }
        }
    }
}

/// Copy `src` to `dst`, with 16-byte streaming stores on x86-64 (SSE2, every x86-64 CPU):
/// the codeword is written once here and read again only in pass 2, after other tiles.
///
/// # Safety
/// `dst .. dst + src.len()` is writable and disjoint from `src`; `dst` is 16-byte aligned
/// (an `F128`).
unsafe fn stream_copy(dst: *mut F128, src: &[F128]) {
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::{__m128i, _mm_loadu_si128, _mm_stream_si128};
        for (i, v) in src.iter().enumerate() {
            // SAFETY: element i of both lies inside the ranges of the contract; F128 is 16
            // bytes, 16-byte aligned, two little-endian u64 words.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        simd::{test_masks, with_mask},
        test_rng::Rng,
    };

    /// The lane-blocked encode holds, per lane, the interleaved transform of the
    /// zero-padded message: every (log_d, rate, t, live) shape including partial last
    /// groups, dead groups and the smallest supported transform, on every SIMD mask.
    #[test]
    fn matches_interleaved_transform() {
        let mut rng = Rng::new(0x1A4E_B10C);
        for &(log_d, r, t, live) in &[
            (6usize, 0usize, 1usize, 1usize),
            (7, 1, 4, 4),
            (8, 1, 5, 5),
            (9, 2, 7, 3),
            (10, 1, 9, 9),
            (11, 1, 17, 17),
            (12, 2, 6, 4),
            (13, 1, 3, 0),
            (15, 1, 5, 5),
            (16, 2, 17, 13),
        ] {
            let d = 1usize << (log_d - r);
            let n = 1usize << log_d;
            let mut q = vec![F128::ZERO; t * d];
            for w in q[..live * d].iter_mut() {
                *w = rng.f128();
            }
            let ntt = AdditiveNttF128::standard(log_d);
            // Oracle: interleaved transform of the zero-padded lanes.
            let mut want = vec![F128::ZERO; n * t];
            for l in 0..t {
                for p in 0..d {
                    want[p * t + l] = q[l * d + p];
                }
            }
            ntt.forward_transform_interleaved(&mut want, t);
            let groups = t.div_ceil(4);
            for mask in test_masks() {
                let got = with_mask(mask, || {
                    let mut cw = rng.f128_vec(groups * 4 * n);
                    ntt.encode_lane_blocked(&mut cw, &q, t, live, d, r);
                    cw
                });
                for p in 0..n {
                    for l in 0..groups * 4 {
                        let v = got[(l / 4 * n + p) * 4 + l % 4];
                        let w = if l < t { want[p * t + l] } else { F128::ZERO };
                        assert_eq!(v, w, "log_d={log_d} r={r} t={t} live={live} p={p} l={l} mask={mask:#x}");
                    }
                }
            }
        }
    }

    /// The hashed encode leaves the codeword of the stored encode (`keep_final`) or a
    /// state that the finishing pass turns into it, and its leaves are the BLAKE3 leaf
    /// hashes of the rows: every shape of the transform test, every SIMD mask.
    #[test]
    fn hashed_encode_matches_encode_and_leaves() {
        use crate::merkle::{HashKind, hash_leaf};
        let mut rng = Rng::new(0x1A4E_4A54);
        for &(log_d, r, t, live) in &[
            (6usize, 0usize, 1usize, 1usize),
            (7, 1, 4, 4),
            (8, 1, 5, 5),
            (9, 2, 7, 3),
            (10, 1, 9, 9),
            (11, 1, 17, 17),
            (12, 2, 6, 4),
            (13, 1, 3, 0),
            (15, 1, 5, 5),
            (16, 2, 17, 13),
            (17, 2, 34, 34),
        ] {
            let d = 1usize << (log_d - r);
            let n = 1usize << log_d;
            let groups = t.div_ceil(4);
            let mut q = vec![F128::ZERO; t * d];
            for w in q[..live * d].iter_mut() {
                *w = rng.f128();
            }
            let ntt = AdditiveNttF128::standard(log_d);
            let mut want = rng.f128_vec(groups * 4 * n);
            ntt.encode_lane_blocked(&mut want, &q, t, live, d, r);
            let want_leaves: Vec<Hash> = (0..n)
                .map(|p| {
                    let mut bytes = Vec::with_capacity(16 * t);
                    for l in 0..t {
                        let w = want[(l / 4 * n + p) * 4 + l % 4];
                        bytes.extend_from_slice(&w.lo.to_le_bytes());
                        bytes.extend_from_slice(&w.hi.to_le_bytes());
                    }
                    hash_leaf(&bytes, HashKind::Blake3)
                })
                .collect();
            let hasher = GroupLeafHasher::new(t, HashKind::Blake3).expect("one chunk");
            for mask in test_masks() {
                for keep in [true, false] {
                    let (mut cw, leaves, first) = with_mask(mask, || {
                        let mut cw = rng.f128_vec(groups * 4 * n);
                        let mut leaves = vec![[0u8; 32]; n];
                        let first = ntt.encode_lane_blocked_hashed(&mut cw, &q, t, live, d, r, &hasher, &mut leaves, keep);
                        (cw, leaves, first)
                    });
                    let what = format!("log_d={log_d} r={r} t={t} live={live} keep={keep} mask={mask:#x}");
                    assert!(leaves == want_leaves, "leaves ({what})");
                    if !keep {
                        let sub = 1usize << (log_d - first);
                        let mut work = Vec::new();
                        for &p in &[0usize, 1, sub - 1, n / 2 + 3, n - 1] {
                            let p0 = p / sub * sub;
                            for g in 0..groups {
                                let block = &cw[(g * n + p0) * 4..][..sub * 4];
                                let line = ntt.lane_blocked_finish_position(block, p0, first, log_d, p, &mut work);
                                assert_eq!(&line[..], &want[(g * n + p) * 4..][..4], "position {p} group {g} ({what})");
                            }
                        }
                        ntt.lane_blocked_finish(&mut cw, n, first);
                    }
                    assert!(cw == want, "codeword ({what})");
                }
            }
        }
    }

    /// Single-thread, cache-resident cost of the pass-2 layers (cycles per line butterfly
    /// are what bound the encode where the carry-less multiply is slow):
    /// `cargo test --release --lib deep_layers_cost -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing"]
    fn deep_layers_cost() {
        let log_d = 21usize;
        let ntt = AdditiveNttF128::standard(log_d);
        let mut rng = Rng::new(7);
        for &low in &[10usize, 11, 12] {
            let sub = 1usize << low;
            let mut block = rng.f128_vec(sub * 4);
            let reps = 1usize << (24 - low);
            let t0 = std::time::Instant::now();
            for i in 0..reps {
                ntt.deep_layers(&mut block, (i % 8) * sub, log_d - low, log_d, true);
            }
            let ns = t0.elapsed().as_secs_f64() * 1e9;
            let butterflies = (reps * sub / 2 * low) as f64;
            eprintln!("low={low}: {:.2} ns per line butterfly ({:.1} ms total)", ns / butterflies, ns / 1e6);
        }
    }

    /// Direction-only timing of the encode at the 4,096-record sponge shape:
    /// `cargo test --release --lib lane_blocked_encode_timing -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing"]
    fn lane_blocked_encode_timing() {
        let (log_d, r, t) = (21usize, 2usize, 17usize);
        let d = 1usize << (log_d - r);
        let mut rng = Rng::new(0x71AE);
        let q: Vec<F128> = (0..t * d).map(|_| rng.f128()).collect();
        let ntt = AdditiveNttF128::standard(log_d);
        let groups = t.div_ceil(4);
        let ms = |t0: std::time::Instant| t0.elapsed().as_secs_f64() * 1e3;
        for rep in 0..3 {
            let mut cw = crate::scratch::take_f128(groups * 4 << log_d);
            cw.par_chunks_mut(256).for_each(|c| c[0] = F128::ZERO);
            let t0 = std::time::Instant::now();
            ntt.encode_lane_blocked(&mut cw, &q, t, t, d, r);
            let blocked = ms(t0);
            let mut iw = crate::scratch::take_f128(t << log_d);
            iw.par_chunks_mut(256).for_each(|c| c[0] = F128::ZERO);
            let t0 = std::time::Instant::now();
            for rep_i in 0..1usize << r {
                for p in 0..d {
                    for l in 0..t {
                        iw[(rep_i * d + p) * t + l] = q[l * d + p];
                    }
                }
            }
            let fill = ms(t0);
            let t0 = std::time::Instant::now();
            ntt.forward_transform_interleaved_from_layer(&mut iw, t, r);
            let interleaved = ms(t0);
            eprintln!("rep {rep}: lane-blocked encode {blocked:.1} ms | interleaved fill {fill:.1} + ntt {interleaved:.1} ms");
            crate::scratch::give_f128(cw);
            crate::scratch::give_f128(iw);
        }
    }

    /// Direction-only timing of the stored encode plus the gathered Merkle leaves against
    /// the hashed encode (final codeword kept and dropped), at the 4,096-record level-0
    /// shape or `FLOCK_LB_TIMING_SHAPE=log_d,r,t`:
    /// `cargo test --release --lib hashed_encode_timing -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing"]
    fn hashed_encode_timing() {
        use crate::merkle::{HashKind, merkle_tree_lane_blocked};
        let shape: Vec<usize> = std::env::var("FLOCK_LB_TIMING_SHAPE")
            .ok()
            .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
            .unwrap_or_else(|| vec![20, 2, 17]);
        let (log_d, r, t) = (shape[0], shape[1], shape[2]);
        let d = 1usize << (log_d - r);
        let n = 1usize << log_d;
        let mut rng = Rng::new(0x71AF);
        let q: Vec<F128> = (0..t * d).map(|_| rng.f128()).collect();
        let ntt = AdditiveNttF128::standard(log_d);
        let groups = t.div_ceil(4);
        let hasher = GroupLeafHasher::new(t, HashKind::Blake3).expect("one chunk");
        let ms = |t0: std::time::Instant| t0.elapsed().as_secs_f64() * 1e3;
        let mut cw = crate::scratch::take_f128(groups * 4 * n);
        cw.par_chunks_mut(256).for_each(|c| c[0] = F128::ZERO);
        for rep in 0..3 {
            let t0 = std::time::Instant::now();
            ntt.encode_lane_blocked(&mut cw, &q, t, t, d, r);
            let enc = ms(t0);
            let t0 = std::time::Instant::now();
            let tree = merkle_tree_lane_blocked(&cw, t, n, HashKind::Blake3);
            let leaves = ms(t0);
            drop(tree);
            let mut tree = vec![[0u8; 32]; n];
            let t0 = std::time::Instant::now();
            ntt.encode_lane_blocked_hashed(&mut cw, &q, t, t, d, r, &hasher, &mut tree, true);
            let keep = ms(t0);
            let t0 = std::time::Instant::now();
            ntt.encode_lane_blocked_hashed(&mut cw, &q, t, t, d, r, &hasher, &mut tree, false);
            let discard = ms(t0);
            eprintln!(
                "rep {rep}: stored encode {enc:.1} + tree {leaves:.1} = {:.1} ms | hashed keep {keep:.1} ms | hashed discard {discard:.1} ms",
                enc + leaves
            );
        }
        crate::scratch::give_f128(cw);
    }
}
