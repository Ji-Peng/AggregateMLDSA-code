//! High-level batch orchestration over SIMD kernels.

use crate::dispatch::Engine;
pub use crate::poly::ExpandedA;
use crate::poly::{Challenge39, Poly32};

impl Engine {
    /// Expands public seed `rho` into matrix `A` for multiple keys in parallel.
    pub fn expand_a_many(&self, rhos: &[[u8; 32]], out: &mut [ExpandedA]) {
        let _ = (rhos, out);
        todo!("T2")
    }

    /// Precomputes centered coefficient-form matrix A and NTT-domain matrix A modulo P2.
    ///
    /// Performs 16 inverse NTTs (Q, Plain, Centered) and 16 forward NTTs (P2) + reduction.
    pub fn prepare_matrices(
        &self,
        a_hat: &[[Poly32; 4]; 4],
        a: &mut [[[i32; 256]; 4]; 4],
        a_hat_p2: &mut [[Poly32; 4]; 4],
    ) {
        let _ = (a_hat, a, a_hat_p2);
        todo!("T2")
    }

    /// Computes batched SHAKE256 digests for multiple inputs (out_len <= 136).
    pub fn shake256_many(&self, inputs: &[[&[u8]; 4]], out_len: usize, out: &mut [u8]) {
        let _ = (inputs, out_len, out);
        todo!("T2")
    }

    /// Evaluates SampleInBall on multiple commitment hashes in parallel.
    pub fn sample_in_ball_many(&self, c_tildes: &[[u8; 32]], out: &mut [Challenge39]) {
        let _ = (c_tildes, out);
        todo!("T2")
    }
}
