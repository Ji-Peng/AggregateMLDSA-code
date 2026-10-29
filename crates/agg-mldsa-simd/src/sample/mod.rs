//! Samplers and packing routines (rejection sampling, unpack z, pack w1).

pub(crate) mod avx2;
pub(crate) mod avx512;
pub(crate) mod scalar;

use crate::dispatch::{Backend, Engine};
use crate::poly::{Block, Poly32};
pub use crate::poly::RejState;

impl Engine {
    /// Uniform rejection sampling from a 168-byte SHAKE128 rate block.
    ///
    /// Appends valid coefficients to `st`. Returns `true` when 256 coefficients
    /// have been collected.
    pub fn rej_uniform(&self, block: &Block, st: &mut RejState) -> bool {
        match self.backend() {
            Backend::Scalar => scalar::rej_uniform(block, st),
            Backend::Avx2 => {
                // SAFETY: The engine guarantees AVX2 and POPCNT features are available.
                unsafe { avx2::rej_uniform(block, st) }
            }
            Backend::Avx512 => {
                // SAFETY: The engine guarantees AVX-512 features are available.
                unsafe { avx512::rej_uniform(block, st) }
            }
        }
    }

    /// Decodes 4 polynomials of response vector z from 2,304 packed bytes.
    ///
    /// BitUnpack(gamma1 - 1, gamma1); returns the maximum absolute value ||z||_inf.
    pub fn unpack_z(&self, z: &[u8; 2304], out: &mut [Poly32; 4]) -> u32 {
        match self.backend() {
            Backend::Scalar => scalar::unpack_z(z, out),
            Backend::Avx2 => {
                // SAFETY: The engine guarantees AVX2 and POPCNT features are available.
                unsafe { avx2::unpack_z(z, out) }
            }
            Backend::Avx512 => {
                // SAFETY: The engine guarantees AVX-512 features are available.
                unsafe { avx512::unpack_z(z, out) }
            }
        }
    }

    /// Encodes 4 polynomials of high bits w1 into 768 bytes (w1Encode).
    pub fn pack_w1(&self, w1: &[[u8; 256]; 4], out: &mut [u8; 768]) {
        match self.backend() {
            Backend::Scalar => scalar::pack_w1(w1, out),
            Backend::Avx2 => {
                // SAFETY: The engine guarantees AVX2 and POPCNT features are available.
                unsafe { avx2::pack_w1(w1, out) }
            }
            Backend::Avx512 => {
                // SAFETY: The engine guarantees AVX-512 features are available.
                unsafe { avx512::pack_w1(w1, out) }
            }
        }
    }
}
