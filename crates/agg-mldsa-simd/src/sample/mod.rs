//! Samplers and packing routines (rejection sampling, unpack z, pack w1).

pub(crate) mod avx2;
pub(crate) mod avx512;
pub(crate) mod scalar;

use crate::dispatch::{Backend, Engine};
use crate::params::ParamSet;
use crate::poly::{Block, Poly32};
pub use crate::poly::RejState;

impl<P: ParamSet> Engine<P> {
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

    /// Decodes response vector z from packed bytes (18 bits or 20 bits per coefficient).
    ///
    /// BitUnpack(gamma1 - 1, gamma1); returns the maximum absolute value ||z||_inf.
    pub fn unpack_z(&self, z: &[u8], out: &mut [Poly32]) -> u32 {
        if P::Z_BITS == 18 && z.len() == 2304 && out.len() == 4 {
            let z4: &[u8; 2304] = z.try_into().unwrap();
            let out4: &mut [Poly32; 4] = out.try_into().unwrap();
            match self.backend() {
                Backend::Scalar => scalar::unpack_z(z4, out4),
                Backend::Avx2 => unsafe { avx2::unpack_z(z4, out4) },
                Backend::Avx512 => unsafe { avx512::unpack_z(z4, out4) },
            }
        } else {
            scalar::unpack_z_20(z, out)
        }
    }

    /// Decodes 4 polynomials of response vector z from 2,304 packed bytes (ML-DSA-44).
    pub fn unpack_z_44(&self, z: &[u8; 2304], out: &mut [Poly32; 4]) -> u32 {
        self.unpack_z(z, out)
    }

    /// Encodes polynomials of high bits w1 into packed bytes (w1Encode).
    pub fn pack_w1(&self, w1: &[[u8; 256]], out: &mut [u8]) {
        if P::W1_BITS == 6 && w1.len() == 4 && out.len() == 768 {
            let w4: &[[u8; 256]; 4] = w1.try_into().unwrap();
            let out4: &mut [u8; 768] = out.try_into().unwrap();
            match self.backend() {
                Backend::Scalar => scalar::pack_w1(w4, out4),
                Backend::Avx2 => unsafe { avx2::pack_w1(w4, out4) },
                Backend::Avx512 => unsafe { avx512::pack_w1(w4, out4) },
            }
        } else {
            scalar::pack_w1_4(w1, out);
        }
    }

    /// Encodes 4 polynomials of high bits w1 into 768 bytes (ML-DSA-44).
    pub fn pack_w1_44(&self, w1: &[[u8; 256]; 4], out: &mut [u8; 768]) {
        self.pack_w1(w1, out);
    }
}
