//! Keccak, SHAKE128, and SHAKE256 sponge operations.

pub(crate) mod avx2;
pub(crate) mod avx512;
pub(crate) mod lanes;
pub mod rounds;
pub(crate) mod scalar;

use crate::dispatch::{Backend, Engine};
use crate::params::ParamSet;
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

impl<P: ParamSet> Engine<P> {
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

pub use rounds::Words8;

impl<P: ParamSet> Engine<P> {
    /// Rounds `first .. first + n` of Keccak-f[1600] on eight word-major states
    /// (`state[w][i]` is lane `w` of state `i`), `n = z.len() / 25`. Round `k`
    /// writes, for lane `l = x + 5 y` and `phi = pi o rho o theta` of its input
    /// state, `a[25 k + l] = !phi[x + 1, y]`, `b[25 k + l] = phi[x + 2, y]` and
    /// `z[25 k + l] = a & b`; the state becomes `phi ^ z` with the round constant
    /// on lane 0. The scalar backend walks one state at a time, AVX2 four and
    /// AVX-512 eight; the outputs are equal.
    ///
    /// # Panics
    /// If the outputs differ in length, are not whole rounds, or `first + n > 24`.
    pub fn chi_rounds(
        &self,
        state: &mut [Words8; 25],
        first: usize,
        z: &mut [Words8],
        a: &mut [Words8],
        b: &mut [Words8],
    ) {
        match self.backend() {
            Backend::Scalar => rounds::scalar::chi_rounds(state, first, z, a, b),
            // SAFETY: The engine guarantees the AVX2 feature is available.
            Backend::Avx2 => unsafe { rounds::avx2::chi_rounds(state, first, z, a, b) },
            // SAFETY: The engine guarantees the AVX-512F feature is available.
            Backend::Avx512 => unsafe { rounds::avx512::chi_rounds(state, first, z, a, b) },
        }
    }

    /// [`Engine::chi_rounds`] with `phi` itself as the output: round `k` writes
    /// `phi[25 k + l]` for lane `l`, from which `a = !phi[x + 1, y]` and
    /// `b = phi[x + 2, y]` follow, and the chi AND outputs `t` of the round into the
    /// 25 words of `z`, then calls `each(k, z)` (a tile that stays in L1, for
    /// consumers of `t` such as a lincheck stripe).
    ///
    /// # Panics
    /// If `phi` is not whole rounds, `z.len() != 25` or `first + n > 24`.
    pub fn phi_rounds(
        &self,
        state: &mut [Words8; 25],
        first: usize,
        phi: &mut [Words8],
        z: &mut [Words8],
        each: impl FnMut(usize, &[Words8; 25]),
    ) {
        match self.backend() {
            Backend::Scalar => rounds::scalar::phi_rounds(state, first, phi, z, each),
            // SAFETY: The engine guarantees the AVX2 feature is available.
            Backend::Avx2 => unsafe { rounds::avx2::phi_rounds(state, first, phi, z, each) },
            // SAFETY: The engine guarantees the AVX-512F feature is available.
            Backend::Avx512 => unsafe { rounds::avx512::phi_rounds(state, first, phi, z, each) },
        }
    }

    /// Keccak-f[1600] on eight word-major states (see [`Engine::chi_rounds`]).
    pub fn keccak_f8(&self, state: &mut [Words8; 25]) {
        match self.backend() {
            Backend::Scalar => rounds::scalar::keccak_f(state),
            // SAFETY: The engine guarantees the AVX2 feature is available.
            Backend::Avx2 => unsafe { rounds::avx2::keccak_f(state) },
            // SAFETY: The engine guarantees the AVX-512F feature is available.
            Backend::Avx512 => unsafe { rounds::avx512::keccak_f(state) },
        }
    }

    /// The bit transpose of a lincheck stripe: for word `u` (one lane of each of
    /// eight states), `out[64 u + bit]` holds bit `bit` of state `r` at bit `r`.
    /// Words that are zero in all eight states leave their 64 bytes unchanged.
    ///
    /// # Panics
    /// If `out.len() != 64 * words.len()`.
    pub fn transpose_bits8(&self, words: &[Words8], out: &mut [u8]) {
        assert_eq!(out.len(), 64 * words.len(), "64 bytes per word");
        match self.backend() {
            Backend::Scalar => rounds::scalar::transpose_bits8(words, out),
            // SAFETY: The engine guarantees the AVX2 feature is available.
            Backend::Avx2 => unsafe { rounds::avx2::transpose_bits8(words, out) },
            // SAFETY: The engine guarantees the AVX-512F and AVX-512BW features are available.
            Backend::Avx512 => unsafe { rounds::avx512::transpose_bits8(words, out) },
        }
    }
}
