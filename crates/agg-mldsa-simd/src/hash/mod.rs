//! Backend-independent batching over the engine's transform and sponge kernels.
pub use crate::poly::ExpandedA;
use crate::poly::{Block, Challenge39, Poly32, RejState};
use crate::{
    dispatch::Engine,
    keccak::{Sink, Xof, XofJob},
    ntt::{Form, Scale},
    params::Prime,
};

struct ExpandSink<'a> {
    engine: &'a Engine,
    out: &'a mut [ExpandedA],
    state: Vec<RejState>,
}
impl Sink for ExpandSink<'_> {
    fn block(&mut self, job: usize, block: &Block, _: usize) -> bool {
        let (key, r, s) = (job / 16, job % 16 / 4, job % 4);
        self.out[key].blocks[r][s] = self.out[key].blocks[r][s]
            .checked_add(1)
            .expect("ExpandA block counter overflow");
        if !self.engine.rej_uniform(block, &mut self.state[job]) {
            return false;
        }
        self.out[key].a_hat[r][s]
            .0
            .copy_from_slice(&self.state[job].coeffs[..256]);
        self.engine.order_from_fips(&mut self.out[key].a_hat[r][s]);
        true
    }
}
struct DigestSink<'a> {
    out: &'a mut [u8],
    len: usize,
}
impl Sink for DigestSink<'_> {
    fn block(&mut self, job: usize, block: &Block, _: usize) -> bool {
        self.out[job * self.len..(job + 1) * self.len].copy_from_slice(&block.bytes[..self.len]);
        true
    }
}
struct BallState {
    c: [i8; 256],
    signs: u64,
    next: usize,
}
impl Default for BallState {
    fn default() -> Self {
        Self {
            c: [0; 256],
            signs: 0,
            next: 0,
        }
    }
}
struct BallSink<'a> {
    states: Vec<BallState>,
    out: &'a mut [Challenge39],
}
impl Sink for BallSink<'_> {
    fn block(&mut self, job: usize, block: &Block, rate: usize) -> bool {
        let st = &mut self.states[job];
        let mut offset = 0;
        if st.next == 0 {
            st.signs = u64::from_le_bytes(block.bytes[..8].try_into().unwrap());
            st.next = 217;
            offset = 8;
        }
        for &b in &block.bytes[offset..rate] {
            let j = b as usize;
            if j > st.next {
                continue;
            }
            st.c[st.next] = st.c[j];
            st.c[j] = 1 - 2 * (st.signs & 1) as i8;
            st.signs >>= 1;
            st.next += 1;
            if st.next == 256 {
                let mut k = 0;
                let out = &mut self.out[job];
                out.neg = 0;
                for (i, &c) in st.c.iter().enumerate() {
                    if c != 0 {
                        out.pos[k] = i as u8;
                        out.neg |= ((c < 0) as u64) << k;
                        k += 1;
                    }
                }
                return true;
            }
        }
        false
    }
}
impl Engine {
    /// Expands seeds into sampled matrix entries and their consumed block counts.
    pub fn expand_a_many(&self, rhos: &[[u8; 32]], out: &mut [ExpandedA]) {
        assert_eq!(rhos.len(), out.len());
        // Bound scratch memory independently of the caller's batch size.
        for (rhos, out) in rhos.chunks(8).zip(out.chunks_mut(8)) {
            let suffix: [[u8; 2]; 16] = std::array::from_fn(|i| [(i % 4) as u8, (i / 4) as u8]);
            let mut jobs = Vec::with_capacity(16 * rhos.len());
            for rho in rhos {
                for tail in &suffix {
                    jobs.push(XofJob {
                        xof: Xof::Shake128,
                        input: [rho, tail, &[], &[]],
                    });
                }
            }
            for key in out.iter_mut() {
                key.blocks = [[0; 4]; 4];
            }
            let state = (0..jobs.len()).map(|_| RejState::new()).collect();
            self.run_xof(
                &jobs,
                &mut ExpandSink {
                    engine: self,
                    out,
                    state,
                },
            );
        }
    }
    /// Prepares natural centered A and centered P2 transforms in backend order.
    pub fn prepare_matrices(
        &self,
        a_hat: &[[Poly32; 4]; 4],
        a: &mut [[[i32; 256]; 4]; 4],
        a_hat_p2: &mut [[Poly32; 4]; 4],
    ) {
        for r in 0..4 {
            for s in 0..4 {
                let mut poly = a_hat[r][s];
                self.intt(Prime::Q, &mut poly, Scale::Plain, Form::Centered);
                a[r][s] = poly.0;
                self.ntt(Prime::P2, &mut poly);
                self.reduce(Prime::P2, &mut poly, Form::Centered);
                a_hat_p2[r][s] = poly;
            }
        }
    }
    /// Hashes segmented inputs into fixed-size SHAKE256 digests of at most 136 bytes.
    pub fn shake256_many(&self, inputs: &[[&[u8]; 4]], out_len: usize, out: &mut [u8]) {
        assert!(out_len <= 136);
        assert_eq!(out.len(), inputs.len().checked_mul(out_len).unwrap());
        for (i, chunk) in inputs.chunks(64).enumerate() {
            let jobs: Vec<_> = chunk
                .iter()
                .map(|&input| XofJob {
                    xof: Xof::Shake256,
                    input,
                })
                .collect();
            self.run_xof(
                &jobs,
                &mut DigestSink {
                    out: &mut out[i * 64 * out_len..][..chunk.len() * out_len],
                    len: out_len,
                },
            );
        }
    }
    /// Samples sorted sparse challenges, preserving Fisher-Yates state across blocks.
    pub fn sample_in_ball_many(&self, c_tildes: &[[u8; 32]], out: &mut [Challenge39]) {
        assert_eq!(c_tildes.len(), out.len());
        for (seeds, out) in c_tildes.chunks(64).zip(out.chunks_mut(64)) {
            let jobs: Vec<_> = seeds
                .iter()
                .map(|seed| XofJob {
                    xof: Xof::Shake256,
                    input: [seed, &[], &[], &[]],
                })
                .collect();
            let states = (0..seeds.len()).map(|_| BallState::default()).collect();
            self.run_xof(&jobs, &mut BallSink { states, out });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ball_continues_after_rejections() {
        let mut out = [Challenge39::new()];
        let mut sink = BallSink {
            states: vec![BallState::default()],
            out: &mut out,
        };
        let mut block = Block::zero();
        block.bytes.fill(255);
        assert!(!sink.block(0, &block, 136));
        assert_eq!(sink.states[0].next, 217);
        block.bytes.fill(0);
        assert!(sink.block(0, &block, 136));
        assert_eq!(out[0].pos[0], 0);
        assert_eq!(out[0].pos[38], 255);
        assert_eq!(out[0].neg, (1u64 << 39) - 1);
    }
}
