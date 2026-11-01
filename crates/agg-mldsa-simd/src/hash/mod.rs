//! Backend-independent batching over the engine's transform and sponge kernels.
#[cfg(target_arch = "x86_64")]
mod avx2;
pub use crate::poly::ExpandedA;
use crate::poly::{Block, Challenge, Poly32, RejState};
use crate::{
    dispatch::Engine,
    keccak::{Sink, Xof, XofJob},
    ntt::{Form, Scale},
    params::{ParamSet, Prime},
};

struct ExpandSink<'a, P: ParamSet, const K: usize, const L: usize> {
    engine: &'a Engine<P>,
    out: &'a mut [ExpandedA<K, L>],
    state: Vec<RejState>,
}
impl<P: ParamSet, const K: usize, const L: usize> Sink for ExpandSink<'_, P, K, L> {
    fn block(&mut self, job: usize, block: &Block, _: usize) -> bool {
        let entries_per_key = K * L;
        let (key, r, s) = (job / entries_per_key, (job % entries_per_key) / L, job % L);
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
struct BallSink<'a, const TAU: usize> {
    states: Vec<BallState>,
    out: &'a mut [Challenge<TAU>],
}
impl<const TAU: usize> Sink for BallSink<'_, TAU> {
    fn block(&mut self, job: usize, block: &Block, rate: usize) -> bool {
        let st = &mut self.states[job];
        let mut offset = 0;
        if st.next == 0 {
            st.signs = u64::from_le_bytes(block.bytes[..8].try_into().unwrap());
            st.next = 256 - TAU;
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
impl<P: ParamSet> Engine<P> {
    /// Expands seeds into 4x4 matrix entries and block counts (ML-DSA-44).
    pub fn expand_a_many(&self, rhos: &[[u8; 32]], out: &mut [ExpandedA<4, 4>]) {
        self.expand_a_many_kl::<4, 4>(rhos, out);
    }

    /// Expands seeds into KxL matrix entries and block counts for any dimension.
    pub fn expand_a_many_kl<const K: usize, const L: usize>(
        &self,
        rhos: &[[u8; 32]],
        out: &mut [ExpandedA<K, L>],
    ) {
        assert_eq!(rhos.len(), out.len());
        #[cfg(target_arch = "x86_64")]
        if K == 4 && L == 4 && self.backend() == crate::dispatch::Backend::Avx2 {
            let out4: &mut [ExpandedA<4, 4>] = unsafe {
                core::slice::from_raw_parts_mut(
                    out.as_mut_ptr() as *mut ExpandedA<4, 4>,
                    out.len(),
                )
            };
            for (rho, out_entry) in rhos.iter().zip(out4.iter_mut()) {
                // SAFETY: an AVX2 engine exists only if AVX2 and POPCNT were detected.
                unsafe { avx2::expand_a(rho, out_entry) };
            }
            return;
        }
        // Bound scratch memory independently of the caller's batch size.
        for (rhos, out) in rhos.chunks(8).zip(out.chunks_mut(8)) {
            let total_entries = K * L;
            let mut suffixes = Vec::with_capacity(total_entries);
            for r in 0..K {
                for s in 0..L {
                    suffixes.push([s as u8, r as u8]);
                }
            }
            let mut jobs = Vec::with_capacity(total_entries * rhos.len());
            for rho in rhos {
                for tail in &suffixes {
                    jobs.push(XofJob {
                        xof: Xof::Shake128,
                        input: [rho, tail, &[], &[]],
                    });
                }
            }
            for key in out.iter_mut() {
                key.blocks = [[0; L]; K];
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

    /// Prepares natural centered A and centered P2 transforms in backend order (4x4).
    pub fn prepare_matrices(
        &self,
        a_hat: &[[Poly32; 4]; 4],
        a: &mut [[[i32; 256]; 4]; 4],
        a_hat_p2: &mut [[Poly32; 4]; 4],
    ) {
        self.prepare_matrices_kl::<4, 4>(a_hat, a, a_hat_p2);
    }

    /// Prepares natural centered A and centered P2 transforms in backend order for any dimension.
    pub fn prepare_matrices_kl<const K: usize, const L: usize>(
        &self,
        a_hat: &[[Poly32; L]; K],
        a: &mut [[[i32; 256]; L]; K],
        a_hat_p2: &mut [[Poly32; L]; K],
    ) {
        for r in 0..K {
            for s in 0..L {
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
    pub fn sample_in_ball_many<const TAU: usize, T: AsRef<[u8]>>(
        &self,
        c_tildes: &[T],
        out: &mut [Challenge<TAU>],
    ) {
        assert_eq!(c_tildes.len(), out.len());
        for (seeds, out) in c_tildes.chunks(64).zip(out.chunks_mut(64)) {
            let jobs: Vec<_> = seeds
                .iter()
                .map(|seed| XofJob {
                    xof: Xof::Shake256,
                    input: [seed.as_ref(), &[], &[], &[]],
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
    use crate::poly::Challenge39;
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
