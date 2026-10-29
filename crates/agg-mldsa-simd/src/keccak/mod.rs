//! Keccak, SHAKE128, and SHAKE256 sponge operations.

pub(crate) mod avx2;
pub(crate) mod avx512;
pub(crate) mod lanes;
pub(crate) mod scalar;

use crate::dispatch::{Backend, Engine};
use crate::poly::Block;

/// Extendable output function type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Xof {
    /// SHAKE128 (rate 168 bytes, domain separation suffix 0x1F).
    Shake128,
    /// SHAKE256 (rate 136 bytes, domain separation suffix 0x1F).
    Shake256,
}

impl Xof {
    /// Returns the absorption and squeeze rate in bytes.
    #[inline(always)]
    pub const fn rate(self) -> usize {
        match self {
            Xof::Shake128 => 168,
            Xof::Shake256 => 136,
        }
    }
}

/// A batched XOF job taking up to 4 input segments.
#[derive(Clone, Copy, Debug)]
pub struct XofJob<'a> {
    /// Function type (SHAKE128 or SHAKE256).
    pub xof: Xof,
    /// Concatenation of up to 4 input segments.
    pub input: [&'a [u8]; 4],
}

/// Squeeze destination callback for streaming rate blocks.
pub trait Sink {
    /// Delivers one squeezed rate block for `job`.
    ///
    /// Returns `true` when the job requires no further blocks.
    fn block(&mut self, job: usize, block: &Block, rate: usize) -> bool;
}

impl Engine {
    /// Executes a batch of XOF jobs, delivering output blocks to `sink`.
    pub fn run_xof<S: Sink>(&self, jobs: &[XofJob<'_>], sink: &mut S) {
        match self.backend() {
            Backend::Scalar => scalar::run_xof(jobs, sink),
            Backend::Avx2 => {
                // SAFETY: The engine guarantees AVX2 and POPCNT features are available.
                unsafe { avx2::run_xof(jobs, sink) }
            }
            Backend::Avx512 => {
                // SAFETY: The engine guarantees AVX-512 features are available.
                unsafe { avx512::run_xof(jobs, sink) }
            }
        }
    }
}
