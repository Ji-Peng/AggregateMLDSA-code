//! Mixed-rate SHAKE scheduling on a word-major SIMD state.
//!
//! Every lane of a `W`-way Keccak state runs its own job: SHAKE128 and SHAKE256
//! lanes, and lanes that absorb next to lanes that squeeze, share one
//! permutation. This generalises the HSH/SSA technique of the Hybrid-SHA-3
//! authors to batches. A lane whose job ends is refilled from the queue at the
//! next permutation boundary, so no job ever has two blocks in flight.
//!
//! Absorption updates each selected column through an ISA-specific word writer.
//! SIMD stores preserve full-width store forwarding to the next permutation.
//! Callbacks use fixed lane indices; their completion flags are collected before
//! lanes are refilled at the same permutation boundary.

use super::{Sink, XofJob};
use crate::poly::Block;

/// Word-major state: `0[w][lane]` is word `w` of lane `lane`.
#[repr(C, align(64))]
pub(crate) struct State<const W: usize>(pub [[u64; W]; 25]);

/// The ISA kernels behind the scheduler.
pub(crate) trait Permutation<const W: usize> {
    fn permute(&mut self, state: &mut State<W>);
    fn absorb(
        &self,
        lanes: &mut [Lane; W],
        jobs: &[XofJob<'_>],
        state: &mut State<W>,
        mask: u8,
    ) -> u8;
    /// Writes bytes `0..168` of every block: 168 rate bytes for the lanes in
    /// `wide`, 136 rate bytes and 32 zero bytes for the others. Bytes
    /// `168..256` remain zero (a kernel may overwrite them with zero).
    fn extract(&self, state: &State<W>, blocks: &mut [Block; W], wide: u8);
    /// Zeroes the columns of `lanes` with full-width loads and stores, which
    /// the next permutation's loads can forward from (scalar stores into a
    /// vector word cannot be forwarded, so its loads would wait for them).
    fn clear(&self, state: &mut State<W>, lanes: u8);
}

/// Closure-based kernel set; closures defined inside a `#[target_feature]`
/// function inherit its features.
pub(crate) struct Kernel<P, E, C, A> {
    pub permute: P,
    pub extract: E,
    pub clear: C,
    pub absorb: A,
}

impl<const W: usize, P, E, C, A> Permutation<W> for Kernel<P, E, C, A>
where
    P: FnMut(&mut State<W>),
    E: Fn(&State<W>, &mut [Block; W], u8),
    C: Fn(&mut State<W>, u8),
    A: Fn(&mut [Lane; W], &[XofJob<'_>], &mut State<W>, u8) -> u8,
{
    #[inline(always)]
    fn absorb(
        &self,
        lanes: &mut [Lane; W],
        jobs: &[XofJob<'_>],
        state: &mut State<W>,
        mask: u8,
    ) -> u8 {
        (self.absorb)(lanes, jobs, state, mask)
    }
    #[inline(always)]
    fn clear(&self, state: &mut State<W>, lanes: u8) {
        (self.clear)(state, lanes);
    }
    #[inline(always)]
    fn permute(&mut self, state: &mut State<W>) {
        (self.permute)(state);
    }
    #[inline(always)]
    fn extract(&self, state: &State<W>, blocks: &mut [Block; W], wide: u8) {
        (self.extract)(state, blocks, wide);
    }
}

/// The `len < 8` bytes of `segment` ending at `end`, little-endian.
#[inline(always)]
fn partial(segment: &[u8], end: usize, len: usize) -> u64 {
    if len == 0 {
        return 0;
    }
    if end >= 8 {
        // The 8 bytes ending at `end`; keep the top `len` of them.
        let word = u64::from_le_bytes(segment[end - 8..end].try_into().unwrap());
        return word >> (64 - 8 * len);
    }
    let mut word = 0;
    for (i, &byte) in segment[end - len..end].iter().enumerate() {
        word |= u64::from(byte) << (8 * i);
    }
    word
}

/// Absorb cursor of one lane.
#[derive(Clone, Copy, Default)]
pub(crate) struct Lane {
    job: usize,
    segment: usize,
    offset: usize,
    rate: usize,
}

impl Lane {
    #[inline]
    fn new(job: usize, jobs: &[XofJob<'_>]) -> Self {
        Self {
            job,
            segment: 0,
            offset: 0,
            rate: jobs[job].xof.rate(),
        }
    }

    /// Emits the next block's XOR updates and returns whether it is padded.
    #[inline(always)]
    fn absorb(&mut self, job: &XofJob<'_>, mut write: impl FnMut(usize, u64)) -> bool {
        let rate = self.rate;
        while self.segment < 4 && self.offset == job.input[self.segment].len() {
            self.segment += 1;
            self.offset = 0;
        }
        if self.segment < 4 {
            let segment = job.input[self.segment];
            if segment.len() - self.offset >= rate {
                // A whole block inside one segment.
                let block = &segment[self.offset..self.offset + rate];
                self.offset += rate;
                for (w, chunk) in block.as_chunks::<8>().0.iter().enumerate() {
                    write(w, u64::from_le_bytes(*chunk));
                }
                return false;
            }
            if job.input[self.segment + 1..].iter().all(|s| s.is_empty()) {
                // The final block inside one segment: whole words in place, the
                // partial word and the suffix byte in a register.
                let rest = &segment[self.offset..];
                self.segment = 4;
                let (whole, tail) = rest.as_chunks::<8>();
                for (w, chunk) in whole.iter().enumerate() {
                    write(w, u64::from_le_bytes(*chunk));
                }
                write(
                    whole.len(),
                    partial(segment, segment.len(), tail.len()) | 0x1f << (8 * tail.len()),
                );
                write(rate / 8 - 1, 1 << 63);
                return true;
            }
        }
        self.stream(job, write)
    }

    /// A block that spans segments, or the padding-only block after an input
    /// of a whole number of blocks: bytes stream from the
    /// segments into whole words, and a word that straddles a segment
    /// boundary is assembled in a register.
    #[inline(always)]
    fn stream(&mut self, job: &XofJob<'_>, mut write: impl FnMut(usize, u64)) -> bool {
        let rate = self.rate;
        let mut w = 0;
        let mut emit = |value: u64| {
            write(w, value);
            w += 1;
        };
        // Bytes of a word in progress (fewer than 8).
        let (mut acc, mut pending) = (0u64, 0usize);
        let mut room = rate;
        while room > 0 && self.segment < 4 {
            let segment = job.input[self.segment];
            let start = self.offset;
            let end = segment.len().min(start + room);
            let bytes = &segment[start..end];
            room -= bytes.len();
            if end == segment.len() {
                self.segment += 1;
                self.offset = 0;
            } else {
                self.offset = end;
            }
            if pending == 0 {
                let (whole, rest) = bytes.as_chunks::<8>();
                for chunk in whole {
                    emit(u64::from_le_bytes(*chunk));
                }
                pending = rest.len();
                acc = partial(segment, end, pending);
            } else {
                for &byte in bytes {
                    acc |= u64::from(byte) << (8 * pending);
                    pending += 1;
                    if pending == 8 {
                        emit(acc);
                        (acc, pending) = (0, 0);
                    }
                }
            }
        }
        // A full block leaves no pending bytes; otherwise the input ended here.
        let last = room > 0;
        if last {
            emit(acc | 0x1f << (8 * pending));
        }
        if last {
            write(rate / 8 - 1, 1 << 63);
        }
        last
    }
}

/// Absorb with an ISA-specific word update, without changing the state layout.
#[inline(always)]
pub(crate) fn absorb_with<const W: usize>(
    lanes: &mut [Lane; W],
    jobs: &[XofJob<'_>],
    mask: u8,
    mut write: impl FnMut(usize, usize, u64),
) -> u8 {
    let mut last = 0;
    for i in bits(mask) {
        let lane = &mut lanes[i];
        last |= u8::from(lane.absorb(&jobs[lane.job], |w, value| write(i, w, value))) << i;
    }
    last
}

// Keep this strided copy outside the SIMD caller: LLVM otherwise turns it into
// gathers, which cost more than scalar loads and stores for one or two lanes.
/// Copies the rate words of the lanes in `mask` into their blocks (and zeros
/// bytes 136..168 for a SHAKE256 lane, which a previous SHAKE128 job may have
/// written).
#[inline(never)]
fn copy_lanes<const W: usize>(
    state: &State<W>,
    blocks: &mut [Block; W],
    lanes: &[Lane; W],
    mask: u8,
) {
    for i in bits(mask) {
        let words = lanes[i].rate / 8;
        let (out, _) = blocks[i].bytes[..168].as_chunks_mut::<8>();
        for (w, bytes) in out.iter_mut().enumerate() {
            let word = if w < words { state.0[w][i] } else { 0 };
            *bytes = word.to_le_bytes();
        }
    }
}

#[inline(always)]
pub(crate) fn bits(mut mask: u8) -> impl Iterator<Item = usize> {
    core::iter::from_fn(move || {
        if mask == 0 {
            return None;
        }
        let i = mask.trailing_zeros() as usize;
        mask &= mask - 1;
        Some(i)
    })
}

/// Runs every job to completion. Returns (occupied lane-permutations, total
/// lane-permutations) when `STATS` is enabled; production omits the counters.
#[inline(never)]
pub(crate) fn run<const W: usize, const STATS: bool, S: Sink, P: Permutation<W>>(
    jobs: &[XofJob<'_>],
    sink: &mut S,
    mut kernel: P,
) -> (usize, usize) {
    debug_assert!(W <= 8);
    // One or two squeezing lanes copy their words individually; wider
    // outputs use the kernel's transposes.
    let mut state = State([[0; W]; 25]);
    let mut blocks = [Block::zero(); W];
    let mut lanes = [Lane::default(); W];
    // Lane masks: running a job, squeezing (absorbed its final block), rate 168.
    let mut active = 0u8;
    let mut squeezing = 0u8;
    let mut wide = 0u8;
    let mut next = 0;
    for (i, lane) in lanes.iter_mut().enumerate() {
        if next == jobs.len() {
            break;
        }
        *lane = Lane::new(next, jobs);
        active |= 1 << i;
        wide |= u8::from(lane.rate == 168) << i;
        next += 1;
    }
    let (mut occupied, mut total) = (0, 0);
    if active != 0 {
        squeezing = kernel.absorb(&mut lanes, jobs, &mut state, active);
    }
    while active != 0 {
        kernel.permute(&mut state);
        if STATS {
            occupied += active.count_ones() as usize;
            total += W;
        }
        let out = squeezing;
        if out != 0 {
            let rest = out & out.wrapping_sub(1);
            let sparse = if W == 4 {
                rest == 0
            } else {
                rest & rest.wrapping_sub(1) == 0
            };
            if sparse {
                copy_lanes(&state, &mut blocks, &lanes, out);
            } else {
                kernel.extract(&state, &mut blocks, wide);
            }
        }
        let absorbing = active & !out;
        // Fixed indices avoid variable callback addresses and lane-array bounds
        // checks. Collect completion flags before refilling any lane.
        let mut done = 0u8;
        macro_rules! deliver {
            ($i:expr) => {
                if $i < W && out & (1 << $i) != 0 {
                    done |= u8::from(sink.block(lanes[$i].job, &blocks[$i], lanes[$i].rate)) << $i;
                }
            };
        }
        deliver!(0);
        deliver!(1);
        deliver!(2);
        deliver!(3);
        deliver!(4);
        deliver!(5);
        deliver!(6);
        deliver!(7);
        squeezing &= !done;
        if done == 0 && absorbing == 0 {
            continue;
        }
        let mut fresh = 0u8;
        for i in bits(done) {
            if next < jobs.len() {
                let lane = Lane::new(next, jobs);
                next += 1;
                wide = (wide & !(1 << i)) | (u8::from(lane.rate == 168) << i);
                lanes[i] = lane;
                fresh |= 1 << i;
            } else {
                active &= !(1 << i);
            }
        }
        if fresh != 0 {
            kernel.clear(&mut state, fresh);
        }
        let pending = absorbing | fresh;
        if pending != 0 {
            squeezing |= kernel.absorb(&mut lanes, jobs, &mut state, pending);
        }
    }
    (occupied, total)
}
