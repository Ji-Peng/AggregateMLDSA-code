//! Mixed-rate SHAKE scheduling on a word-major SIMD state.
//!
//! Every lane of a `W`-way Keccak state runs its own job: SHAKE128 and SHAKE256
//! lanes, and lanes that absorb next to lanes that squeeze, share one
//! permutation. This generalises the HSH/SSA technique of the Hybrid-SHA-3
//! authors to batches. A lane whose job ends is refilled from the queue at the
//! next permutation boundary, so no job ever has two blocks in flight.
//!
//! Absorption XORs each lane's input words straight into its column of the
//! state with scalar operations, so an absorbing lane touches only its own 17
//! to 21 words. Lanes that keep absorbing take their next block right after
//! the extraction, before the sinks run, which leaves their stores time to
//! retire before the next permutation loads the state; lanes that start a new
//! job are first cleared with full-width vector stores.

use super::{Sink, XofJob};
use crate::poly::Block;

/// Word-major state: `0[w][lane]` is word `w` of lane `lane`.
#[repr(C, align(64))]
pub(crate) struct State<const W: usize>(pub [[u64; W]; 25]);

/// The ISA kernels behind the scheduler.
pub(crate) trait Permutation<const W: usize> {
    fn permute(&mut self, state: &mut State<W>);
    /// Writes bytes `0..168` of every block: 168 rate bytes for the lanes in
    /// `wide`, 136 rate bytes and 32 zero bytes for the others. Bytes
    /// `168..256` are never written.
    fn extract(&self, state: &State<W>, blocks: &mut [Block; W], wide: u8);
    /// Zeroes the columns of `lanes` with full-width loads and stores, which
    /// the next permutation's loads can forward from (scalar stores into a
    /// vector word cannot be forwarded, so its loads would wait for them).
    fn clear(&self, state: &mut State<W>, lanes: u8);
}


/// Closure-based kernel set; closures defined inside a `#[target_feature]`
/// function inherit its features.
pub(crate) struct Kernel<P, E, C> {
    pub permute: P,
    pub extract: E,
    pub clear: C,
}

impl<const W: usize, P, E, C> Permutation<W> for Kernel<P, E, C>
where
    P: FnMut(&mut State<W>),
    E: Fn(&State<W>, &mut [Block; W], u8),
    C: Fn(&mut State<W>, u8),
{
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
struct Lane {
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

    /// XORs the lane's next block into column `lane` and returns whether it
    /// was the final (padded) block.
    #[inline(always)]
    fn absorb<const W: usize>(
        &mut self,
        job: &XofJob<'_>,
        state: &mut State<W>,
        lane: usize,
    ) -> bool {
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
                for (chunk, word) in block.as_chunks::<8>().0.iter().zip(&mut state.0) {
                    word[lane] ^= u64::from_le_bytes(*chunk);
                }
                return false;
            }
            if job.input[self.segment + 1..].iter().all(|s| s.is_empty()) {
                // The final block inside one segment: whole words in place, the
                // partial word and the suffix byte in a register.
                let rest = &segment[self.offset..];
                self.segment = 4;
                let (whole, tail) = rest.as_chunks::<8>();
                for (chunk, word) in whole.iter().zip(&mut state.0) {
                    word[lane] ^= u64::from_le_bytes(*chunk);
                }
                state.0[whole.len()][lane] ^=
                    partial(segment, segment.len(), tail.len()) | 0x1f << (8 * tail.len());
                state.0[rate / 8 - 1][lane] ^= 1 << 63;
                return true;
            }
        }
        self.stream(job, state, lane)
    }

    /// A block that spans segments, or the padding-only block after an input
    /// of a whole number of blocks: bytes stream from the
    /// segments into whole words, and a word that straddles a segment
    /// boundary is assembled in a register.
    #[inline(always)]
    fn stream<const W: usize>(
        &mut self,
        job: &XofJob<'_>,
        state: &mut State<W>,
        lane: usize,
    ) -> bool {
        let rate = self.rate;
        let mut words = state.0.iter_mut();
        let mut emit = |value: u64| {
            // At most `rate / 8 <= 21` words are emitted per block.
            if let Some(word) = words.next() {
                word[lane] ^= value;
            }
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
            state.0[rate / 8 - 1][lane] ^= 1 << 63;
        }
        last
    }
}

// The two column loops below are separate functions on purpose: they are
// compiled without the caller's vector features, so their strided accesses stay
// scalar loads and stores (inlined into an AVX-512 caller, LLVM turns them into
// `vpscatterqq`/`vpgatherqq`, which cost more than the scalar copies here).

/// XORs the next block of every lane in `absorbing` into its column and
/// returns the lanes whose final block that was.
#[inline(never)]
fn absorb<const W: usize>(
    lanes: &mut [Lane; W],
    jobs: &[XofJob<'_>],
    state: &mut State<W>,
    absorbing: u8,
) -> u8 {
    let mut last = 0;
    for i in bits(absorbing) {
        let lane = &mut lanes[i];
        last |= u8::from(lane.absorb(&jobs[lane.job], state, i)) << i;
    }
    last
}

/// Copies the rate words of the lanes in `mask` into their blocks (and zeros
/// bytes 136..168 for a SHAKE256 lane, which a previous SHAKE128 job may have
/// written).
#[inline(never)]
fn copy_lanes<const W: usize>(state: &State<W>, blocks: &mut [Block; W], lanes: &[Lane; W], mask: u8) {
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
fn bits(mut mask: u8) -> impl Iterator<Item = usize> {
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
/// lane-permutations) for diagnostics.
#[inline(always)]
pub(crate) fn run<const W: usize, S: Sink, P: Permutation<W>>(
    jobs: &[XofJob<'_>],
    sink: &mut S,
    mut kernel: P,
) -> (usize, usize) {
    debug_assert!(W <= 8);
    // Up to this many squeezing lanes copy their words one by one; more go
    // through the kernel's transposes.
    let copy_max = W / 4;
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
        squeezing = absorb(&mut lanes, jobs, &mut state, active);
    }
    while active != 0 {
        kernel.permute(&mut state);
        occupied += active.count_ones() as usize;
        total += W;
        let out = squeezing;
        if out != 0 {
            if out.count_ones() as usize <= copy_max {
                copy_lanes(&state, &mut blocks, &lanes, out);
            } else {
                kernel.extract(&state, &mut blocks, wide);
            }
        }
        // Lanes still absorbing take their next block now, so these stores
        // retire while the sinks run rather than just before the permutation.
        let absorbing = active & !out;
        if absorbing != 0 {
            squeezing |= absorb(&mut lanes, jobs, &mut state, absorbing);
        }
        let mut fresh = 0u8;
        for i in bits(out) {
            let lane = &lanes[i];
            if !sink.block(lane.job, &blocks[i], lane.rate) {
                continue;
            }
            // The job is done: refill the lane at this permutation boundary.
            squeezing &= !(1 << i);
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
            squeezing |= absorb(&mut lanes, jobs, &mut state, fresh);
        }
    }
    (occupied, total)
}
