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
    fn clear(&self, state: &mut State<W>, mask: u8);
}

pub(crate) struct Kernel<P, E, Z> {
    pub permute: P,
    pub extract: E,
    pub clear: Z,
}

impl<const W: usize, P, E, Z> Permutation<W> for Kernel<P, E, Z>
where
    P: FnMut(&mut State<W>),
    E: Fn(&State<W>, &mut [Block; W], usize),
    Z: Fn(&mut State<W>, u8),
{
    #[inline]
    fn permute(&mut self, state: &mut State<W>) {
        (self.permute)(state);
    }
    #[inline]
    fn extract(&self, state: &State<W>, blocks: &mut [Block; W], rate: usize) {
        (self.extract)(state, blocks, rate);
    }
    #[inline]
    fn clear(&self, state: &mut State<W>, mask: u8) {
        (self.clear)(state, mask);
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
            state.0[self.rate / 8 - 1][lane] ^= 1u64 << 63;
            self.squeeze = true;
        }
        let words = if self.squeeze {
            (used + 8) / 8
        } else {
            self.rate / 8
        };
        for (word, bytes) in state.0.iter_mut().zip(block[..words * 8].chunks_exact(8)) {
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
        let Some(rate) = lanes
            .iter()
            .flatten()
            .find(|lane| lane.squeeze)
            .map(|lane| lane.rate)
        else {
            continue;
        };
        if lanes
            .iter()
            .all(|lane| lane.is_some_and(|lane| lane.squeeze && lane.rate == rate))
        {
            loop {
                kernel.extract(&state, &mut blocks, rate);
                let mut finished = false;
                let mut clear_mask = 0u8;
                for (i, slot) in lanes.iter_mut().enumerate() {
                    let job = slot.as_ref().unwrap().job;
                    if sink.block(job, &blocks[i], rate) {
                        finished = true;
                        *slot = if next < jobs.len() {
                            let new = Lane::new(next, jobs);
                            next += 1;
                            clear_mask |= 1 << i;
                            Some(new)
                        } else {
                            None
                        };
                    }
                }
                if finished {
                    if clear_mask != 0 {
                        kernel.clear(&mut state, clear_mask);
                    }
                    break;
                }
                occupied += W;
                total += W;
                kernel.permute(&mut state);
            }
            continue;
        }
        // Absorbing lanes have no output yet; their transposed words are ignored.
        let uniform = lanes
            .iter()
            .flatten()
            .filter(|lane| lane.squeeze)
            .all(|lane| lane.rate == rate);
        if uniform {
            kernel.extract(&state, &mut blocks, rate);
        }
        let mut clear_mask = 0u8;
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
                    clear_mask |= 1 << i;
                    Some(new)
                } else {
                    None
                };
            }
        }
        // Clear all recycled lanes in one vector pass, including their capacity words.
        if clear_mask != 0 {
            kernel.clear(&mut state, clear_mask);
        }
    }
    (occupied, total)
}
