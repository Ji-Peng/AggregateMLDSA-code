//! Sequential segmented SHAKE using the keccak crate's scalar permutation.
use super::{Sink, XofJob};
use crate::poly::Block;
pub(crate) fn run_xof<S: Sink>(jobs: &[XofJob<'_>], sink: &mut S) {
    for (job, desc) in jobs.iter().enumerate() {
        let rate = desc.xof.rate();
        let mut state = [0u64; 25];
        let mut pos = 0;
        for segment in desc.input {
            for &byte in segment {
                state[pos / 8] ^= (byte as u64) << (8 * (pos % 8));
                pos += 1;
                if pos == rate {
                    keccak::f1600(&mut state);
                    pos = 0;
                }
            }
        }
        state[pos / 8] ^= 0x1f << (8 * (pos % 8));
        state[(rate - 1) / 8] ^= 0x80 << 56;
        let mut block = Block::zero();
        loop {
            keccak::f1600(&mut state);
            for (bytes, word) in block.bytes[..rate].chunks_exact_mut(8).zip(state.iter()) {
                bytes.copy_from_slice(&word.to_le_bytes());
            }
            if sink.block(job, &block, rate) {
                break;
            }
        }
    }
}
