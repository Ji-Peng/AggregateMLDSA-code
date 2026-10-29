//! Sequential segmented SHAKE using the keccak crate's scalar permutation.
use super::{Sink, XofJob};
use crate::poly::Block;
pub(crate) fn run_xof<S: Sink>(jobs: &[XofJob<'_>], sink: &mut S) {
    for (job, desc) in jobs.iter().enumerate() {
        let rate = desc.xof.rate();
        let mut state = [0u64; 25];
        let mut pos = 0;
        for mut segment in desc.input {
            while !segment.is_empty() {
                if pos % 8 == 0 && segment.len() >= 8 {
                    let words = ((rate - pos) / 8).min(segment.len() / 8);
                    for (word, bytes) in state[pos / 8..pos / 8 + words]
                        .iter_mut()
                        .zip(segment[..words * 8].as_chunks::<8>().0.iter())
                    {
                        *word ^= u64::from_le_bytes(*bytes);
                    }
                    pos += words * 8;
                    segment = &segment[words * 8..];
                } else {
                    state[pos / 8] ^= (segment[0] as u64) << (8 * (pos % 8));
                    pos += 1;
                    segment = &segment[1..];
                }
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
            for (bytes, word) in block.bytes[..rate]
                .as_chunks_mut::<8>()
                .0
                .iter_mut()
                .zip(state.iter())
            {
                bytes.copy_from_slice(&word.to_le_bytes());
            }
            if sink.block(job, &block, rate) {
                break;
            }
        }
    }
}
