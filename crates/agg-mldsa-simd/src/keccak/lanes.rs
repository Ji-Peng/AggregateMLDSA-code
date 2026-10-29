//! Mixed-rate SHAKE scheduling on a word-major SIMD state.
//!
//! The independent absorb/squeeze lanes follow the HSH/SSA technique of
//! Xurong Qi et al., Hybrid-SHA-3. Jobs move only at permutation boundaries.

use super::{Sink, XofJob};
use crate::poly::Block;

#[repr(C, align(64))]
pub(crate) struct State<const W: usize>(pub [[u64; W]; 25]);

/// Callbacks carry the ISA proof from the dispatch function into safe scheduling.
pub(crate) trait Permutation<const W: usize> {
    fn permute(&mut self, state: &mut State<W>);
    fn extract(&self, state: &State<W>, blocks: &mut [Block; W], rate: usize);
}

pub(crate) struct Kernel<P, E> {
    pub permute: P,
    pub extract: E,
}

impl<const W: usize, P, E> Permutation<W> for Kernel<P, E>
where
    P: FnMut(&mut State<W>),
    E: Fn(&State<W>, &mut [Block; W], usize),
{
    #[inline]
    fn permute(&mut self, state: &mut State<W>) {
        (self.permute)(state);
    }
    #[inline]
    fn extract(&self, state: &State<W>, blocks: &mut [Block; W], rate: usize) {
        (self.extract)(state, blocks, rate);
    }
}

#[derive(Clone, Copy)]
struct Lane {
    job: usize,
    segment: usize,
    offset: usize,
    squeeze: bool,
    rate: usize,
}

impl Lane {
    fn new(job: usize, jobs: &[XofJob<'_>]) -> Self {
        Self {
            job,
            segment: 0,
            offset: 0,
            squeeze: false,
            rate: jobs[job].xof.rate(),
        }
    }

    /// XOR one complete block or the final padded block into this lane.
    #[inline]
    fn absorb<const W: usize>(&mut self, jobs: &[XofJob<'_>], state: &mut State<W>, lane: usize) {
        let mut block = [0u8; 168];
        let mut used = 0;
        while used < self.rate && self.segment < 4 {
            let segment = jobs[self.job].input[self.segment];
            let n = (self.rate - used).min(segment.len() - self.offset);
            block[used..used + n].copy_from_slice(&segment[self.offset..self.offset + n]);
            used += n;
            self.offset += n;
            if self.offset == segment.len() {
                self.segment += 1;
                self.offset = 0;
            }
        }
        if used < self.rate {
            block[used] = 0x1f;
            block[self.rate - 1] |= 0x80;
            self.squeeze = true;
        }
        for (word, bytes) in state.0.iter_mut().zip(block[..self.rate].chunks_exact(8)) {
            word[lane] ^= u64::from_le_bytes(bytes.try_into().unwrap());
        }
    }
}

/// Returns (occupied lane-permutations, total lane-permutations) for diagnostics.
pub(crate) fn run<const W: usize, S: Sink, P: Permutation<W>>(
    jobs: &[XofJob<'_>],
    sink: &mut S,
    mut kernel: P,
) -> (usize, usize) {
    let mut state = State([[0; W]; 25]);
    let mut blocks = [Block::zero(); W];
    let mut next = 0;
    let mut lanes: [Option<Lane>; W] = std::array::from_fn(|_| {
        if next == jobs.len() {
            return None;
        }
        let lane = Lane::new(next, jobs);
        next += 1;
        Some(lane)
    });
    let mut occupied = 0;
    let mut total = 0;
    loop {
        let active = lanes.iter().filter(|lane| lane.is_some()).count();
        if active == 0 {
            break;
        }
        occupied += active;
        total += W;
        for (i, lane) in lanes.iter_mut().enumerate() {
            if let Some(lane) = lane {
                if !lane.squeeze {
                    lane.absorb(jobs, &mut state, i);
                }
            }
        }
        kernel.permute(&mut state);
        let rate = lanes.iter().flatten().next().unwrap().rate;
        let uniform = lanes
            .iter()
            .flatten()
            .all(|lane| lane.squeeze && lane.rate == rate);
        if uniform {
            kernel.extract(&state, &mut blocks, rate);
        }
        for (i, slot) in lanes.iter_mut().enumerate() {
            let Some(lane) = slot else {
                continue;
            };
            if !lane.squeeze {
                continue;
            }
            if !uniform {
                for (bytes, word) in blocks[i].bytes[..lane.rate]
                    .chunks_exact_mut(8)
                    .zip(&state.0)
                {
                    bytes.copy_from_slice(&word[i].to_le_bytes());
                }
                blocks[i].bytes[lane.rate..].fill(0);
            }
            if sink.block(lane.job, &blocks[i], lane.rate) {
                *slot = if next < jobs.len() {
                    let new = Lane::new(next, jobs);
                    next += 1;
                    // A recycled lane must not retain capacity or rate words of its old job.
                    for word in &mut state.0 {
                        word[i] = 0;
                    }
                    Some(new)
                } else {
                    None
                };
            }
        }
    }
    (occupied, total)
}
