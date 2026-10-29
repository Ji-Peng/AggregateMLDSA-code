//! Mixed-rate SHAKE scheduling on a word-major SIMD state.
//!
//! Every lane of a `W`-way Keccak state runs its own job: SHAKE128 and SHAKE256
//! lanes, and lanes that absorb next to lanes that squeeze, share one
//! permutation. This generalises the HSH/SSA technique of the Hybrid-SHA-3
//! authors to batches. A lane whose job ends is refilled from the queue at the
//! next permutation boundary, so no job ever has two blocks in flight.

use super::{Sink, XofJob};
use crate::poly::Block;

/// Word-major state: `0[w][lane]` is word `w` of lane `lane`.
#[repr(C, align(64))]
pub(crate) struct State<const W: usize>(pub [[u64; W]; 25]);

/// One lane's contribution to an absorb pass: whole little-endian words, then
/// one assembled word.
#[derive(Clone, Copy)]
pub(crate) struct Input<'a> {
    pub lane: usize,
    /// Length is a multiple of 8 and at most 168.
    pub words: &'a [u8],
    /// XORed into word `words.len() / 8` when nonzero (a partial final word
    /// with its 0x1F suffix byte).
    pub tail: u64,
}

/// The ISA kernels behind the scheduler.
pub(crate) trait Permutation<const W: usize> {
    fn permute(&mut self, state: &mut State<W>);
    /// Zeroes the lanes in `fresh`, XORs each input's words into words
    /// `0..len / 8` of its lane (and its nonzero tail into word `len / 8`), and XORs the final padding bit `1 << 63` into
    /// word 16 of the lanes in `pad16` (SHAKE256) and word 20 of the lanes in
    /// `pad20` (SHAKE128).
    fn absorb(&self, state: &mut State<W>, fresh: u8, inputs: &[Input<'_>], pad16: u8, pad20: u8);
    /// Writes bytes `0..192` of every block (the largest rate rounded up to a
    /// whole 64-byte vector): 168 rate bytes for the lanes in `wide`, 136 for
    /// the others, zeros after them. Bytes `192..256` are never written.
    fn extract(&self, state: &State<W>, blocks: &mut [Block; W], wide: u8);
}

/// Closure-based kernel set; closures defined inside a `#[target_feature]`
/// function inherit its features.
pub(crate) struct Kernel<P, A, E> {
    pub permute: P,
    pub absorb: A,
    pub extract: E,
}

impl<const W: usize, P, A, E> Permutation<W> for Kernel<P, A, E>
where
    P: FnMut(&mut State<W>),
    A: Fn(&mut State<W>, u8, &[Input<'_>], u8, u8),
    E: Fn(&State<W>, &mut [Block; W], u8),
{
    #[inline(always)]
    fn permute(&mut self, state: &mut State<W>) {
        (self.permute)(state);
    }
    #[inline(always)]
    fn absorb(&self, state: &mut State<W>, fresh: u8, inputs: &[Input<'_>], pad16: u8, pad20: u8) {
        (self.absorb)(state, fresh, inputs, pad16, pad20);
    }
    #[inline(always)]
    fn extract(&self, state: &State<W>, blocks: &mut [Block; W], wide: u8) {
        (self.extract)(state, blocks, wide);
    }
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

    /// Takes the lane's next block for lane index `lane` when it lies in one
    /// segment, or ends the input inside one. Returns the input and whether
    /// it is the final (padded) block, or `None` when the block spans
    /// segments (see `gather`). The 0x80 padding bit is left to the kernel.
    #[inline]
    fn next_block<'b>(&mut self, job: &XofJob<'b>, lane: usize) -> Option<(Input<'b>, bool)> {
        let rate = self.rate;
        while self.segment < 4 && self.offset == job.input[self.segment].len() {
            self.segment += 1;
            self.offset = 0;
        }
        if self.segment == 4 {
            // Exact multiple of the rate: a block of padding only.
            return Some((
                Input {
                    lane,
                    words: &[],
                    tail: 0x1f,
                },
                true,
            ));
        }
        let segment = job.input[self.segment];
        let rest = &segment[self.offset..];
        if rest.len() >= rate {
            self.offset += rate;
            return Some((
                Input {
                    lane,
                    words: &rest[..rate],
                    tail: 0,
                },
                false,
            ));
        }
        if job.input[self.segment + 1..].iter().any(|s| !s.is_empty()) {
            return None;
        }
        // Whole words in place; the partial word and the suffix byte in a register.
        self.segment = 4;
        let (words, partial) = rest.split_at(rest.len() & !7);
        let mut tail = 0x1fu64 << (8 * partial.len());
        for (i, &byte) in partial.iter().enumerate() {
            tail |= u64::from(byte) << (8 * i);
        }
        Some((Input { lane, words, tail }, true))
    }

    /// Copies a block that spans segments into `buf` (zeroed by the caller) and
    /// returns its length in bytes (whole words) and whether it is final.
    #[cold]
    #[inline(never)]
    fn gather(&mut self, job: &XofJob<'_>, buf: &mut [u8; 168]) -> (usize, bool) {
        let rate = self.rate;
        let mut used = 0;
        while used < rate && self.segment < 4 {
            let segment = job.input[self.segment];
            let n = (rate - used).min(segment.len() - self.offset);
            buf[used..used + n].copy_from_slice(&segment[self.offset..self.offset + n]);
            used += n;
            self.offset += n;
            if self.offset == segment.len() {
                self.segment += 1;
                self.offset = 0;
            }
        }
        if used == rate {
            return (rate, false);
        }
        buf[used] = 0x1f;
        ((used + 8) & !7, true)
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
    let mut state = State([[0; W]; 25]);
    let mut blocks = [Block::zero(); W];
    let mut lanes = [Lane::default(); W];
    // Lane masks: running a job, squeezing, rate 168.
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
    // The state starts at zero, so the first jobs need no clearing.
    let mut fresh = 0u8;
    let (mut occupied, mut total) = (0, 0);
    while active != 0 {
        let absorbing = active & !squeezing;
        if absorbing != 0 || fresh != 0 {
            let mut inputs = [Input {
                lane: 0,
                words: &[],
                tail: 0,
            }; W];
            let mut n = 0;
            let (mut pad16, mut pad20, mut spanning) = (0u8, 0u8, 0u8);
            for (i, lane) in lanes.iter_mut().enumerate() {
                if absorbing >> i & 1 == 0 {
                    continue;
                }
                let Some((input, last)) = lane.next_block(&jobs[lane.job], i) else {
                    spanning |= 1 << i;
                    continue;
                };
                inputs[n] = input;
                n += 1;
                if last {
                    squeezing |= 1 << i;
                    *(if lane.rate == 168 {
                        &mut pad20
                    } else {
                        &mut pad16
                    }) |= 1 << i;
                }
            }
            kernel.absorb(&mut state, fresh, &inputs[..n], pad16, pad20);
            fresh = 0;
            // Blocks spanning segments go through a staging copy after the clear.
            for i in bits(spanning) {
                let lane = &mut lanes[i];
                let mut buf = [0u8; 168];
                let (len, last) = lane.gather(&jobs[lane.job], &mut buf);
                let (mut pad16, mut pad20) = (0u8, 0u8);
                if last {
                    squeezing |= 1 << i;
                    *(if lane.rate == 168 {
                        &mut pad20
                    } else {
                        &mut pad16
                    }) |= 1 << i;
                }
                let input = Input {
                    lane: i,
                    words: &buf[..len],
                    tail: 0,
                };
                kernel.absorb(&mut state, 0, &[input], pad16, pad20);
            }
        }
        kernel.permute(&mut state);
        occupied += active.count_ones() as usize;
        total += W;
        if squeezing == 0 {
            continue;
        }
        kernel.extract(&state, &mut blocks, wide);
        for i in bits(squeezing) {
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
    }
    (occupied, total)
}
