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
    ///
    /// # Panics
    /// If `out` does not hold l polynomials or `z` is not their packed length
    /// (l * 32 * `Z_BITS` bytes).
    pub fn unpack_z(&self, z: &[u8], out: &mut [Poly32]) -> u32 {
        assert_eq!(out.len(), P::L, "z has l polynomials");
        assert_eq!(z.len(), P::L * 32 * P::Z_BITS, "packed z length");
        match P::Z_BITS {
            18 => {
                // The 18-bit kernels decode exactly four polynomials (ML-DSA-44).
                let z4: &[u8; 2304] = z.try_into().expect("ML-DSA-44 packs z in 2304 bytes");
                let out4: &mut [Poly32; 4] = out.try_into().expect("ML-DSA-44 has l = 4");
                match self.backend() {
                    Backend::Scalar => scalar::unpack_z(z4, out4),
                    // SAFETY: the engine exists only if AVX2 and POPCNT were detected.
                    Backend::Avx2 => unsafe { avx2::unpack_z(z4, out4) },
                    // SAFETY: the engine exists only if the AVX-512 features were detected.
                    Backend::Avx512 => unsafe { avx512::unpack_z(z4, out4) },
                }
            }
            20 => scalar::unpack_z_20(z, out),
            bits => unreachable!("no FIPS 204 set packs z in {bits} bits"),
        }
    }

    /// Encodes polynomials of high bits w1 into packed bytes (w1Encode).
    ///
    /// # Panics
    /// If `w1` does not hold k polynomials or `out` is not their encoded length
    /// (`W1_ENCODE_BYTES`).
    pub fn pack_w1(&self, w1: &[[u8; 256]], out: &mut [u8]) {
        assert_eq!(w1.len(), P::K, "w1 has k polynomials");
        assert_eq!(out.len(), P::W1_ENCODE_BYTES, "w1Encode length");
        match P::W1_BITS {
            6 => {
                // The 6-bit kernels encode exactly four polynomials (ML-DSA-44).
                let w4: &[[u8; 256]; 4] = w1.try_into().expect("ML-DSA-44 has k = 4");
                let out4: &mut [u8; 768] = out.try_into().expect("ML-DSA-44 w1Encode is 768 bytes");
                match self.backend() {
                    Backend::Scalar => scalar::pack_w1(w4, out4),
                    // SAFETY: the engine exists only if AVX2 and POPCNT were detected.
                    Backend::Avx2 => unsafe { avx2::pack_w1(w4, out4) },
                    // SAFETY: the engine exists only if the AVX-512 features were detected.
                    Backend::Avx512 => unsafe { avx512::pack_w1(w4, out4) },
                }
            }
            4 => scalar::pack_w1_4(w1, out),
            bits => unreachable!("no FIPS 204 set encodes w1 in {bits} bits"),
        }
    }
}
