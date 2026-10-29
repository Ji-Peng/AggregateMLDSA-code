//! Polynomial, buffer, state and output types for SIMD kernels.

use std::ops::{Deref, DerefMut};

/// Ring element of 256 signed 32-bit coefficients, aligned to 64 bytes.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Poly32(pub [i32; 256]);

impl Poly32 {
    /// Creates a polynomial with all coefficients set to zero.
    #[inline(always)]
    pub const fn zero() -> Self {
        Self([0; 256])
    }

    /// Creates a polynomial with all coefficients set to zero.
    #[inline(always)]
    pub const fn new() -> Self {
        Self::zero()
    }
}

impl Default for Poly32 {
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

impl Deref for Poly32 {
    type Target = [i32; 256];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Poly32 {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl AsRef<[i32; 256]> for Poly32 {
    #[inline(always)]
    fn as_ref(&self) -> &[i32; 256] {
        &self.0
    }
}

impl AsMut<[i32; 256]> for Poly32 {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [i32; 256] {
        &mut self.0
    }
}

/// A 256-byte aligned buffer used for Keccak rate blocks and I/O.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Block {
    /// 256 aligned bytes.
    pub bytes: [u8; 256],
}

impl Block {
    /// Creates a zeroed block.
    #[inline(always)]
    pub const fn zero() -> Self {
        Self { bytes: [0; 256] }
    }

    /// Creates a zeroed block.
    #[inline(always)]
    pub const fn new() -> Self {
        Self::zero()
    }
}

impl Default for Block {
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

impl Deref for Block {
    type Target = [u8; 256];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.bytes
    }
}

impl DerefMut for Block {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.bytes
    }
}

impl AsRef<[u8; 256]> for Block {
    #[inline(always)]
    fn as_ref(&self) -> &[u8; 256] {
        &self.bytes
    }
}

impl AsMut<[u8; 256]> for Block {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [u8; 256] {
        &mut self.bytes
    }
}

/// Accumulator state for uniform rejection sampling, aligned to 64 bytes.
/// Holds up to 256 valid coefficients with 16 slack lanes for vector writes.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RejState {
    /// Accumulated coefficients (256 + 16 slack lanes).
    pub coeffs: [i32; 272],
    /// Current count of valid accepted coefficients.
    pub count: usize,
}

impl RejState {
    /// Creates a zeroed rejection accumulator state.
    #[inline(always)]
    pub const fn zero() -> Self {
        Self {
            coeffs: [0; 272],
            count: 0,
        }
    }

    /// Creates a zeroed rejection accumulator state.
    #[inline(always)]
    pub const fn new() -> Self {
        Self::zero()
    }
}

impl Default for RejState {
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

/// The sparse challenge c with tau = 39 nonzero coefficients.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Challenge39 {
    /// Increasing positions of the 39 nonzero coefficients.
    pub pos: [u8; 39],
    /// Bit k indicates sign: 1 if c[pos[k]] = -1, 0 if +1.
    pub neg: u64,
}

impl Challenge39 {
    /// Creates a zeroed challenge.
    #[inline(always)]
    pub const fn zero() -> Self {
        Self {
            pos: [0; 39],
            neg: 0,
        }
    }

    /// Creates a zeroed challenge.
    #[inline(always)]
    pub const fn new() -> Self {
        Self::zero()
    }
}

impl Default for Challenge39 {
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

/// Output of finishing one row: w1 values and the u, v polynomials.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FinishOut {
    /// High bits w1 for 256 coefficients.
    pub w1: [u8; 256],
    /// Residual polynomial u.
    pub u: Poly32,
    /// Quotient polynomial v.
    pub v: Poly32,
}

impl FinishOut {
    /// Creates a zeroed finish output.
    #[inline(always)]
    pub const fn zero() -> Self {
        Self {
            w1: [0; 256],
            u: Poly32::zero(),
            v: Poly32::zero(),
        }
    }

    /// Creates a zeroed finish output.
    #[inline(always)]
    pub const fn new() -> Self {
        Self::zero()
    }
}

impl Default for FinishOut {
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

/// Expanded matrix A in backend order and number of blocks consumed per entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpandedA {
    /// 4x4 matrix of polynomials in NTT backend order, in [0, q).
    pub a_hat: [[Poly32; 4]; 4],
    /// Number of SHAKE128 blocks consumed per matrix entry.
    pub blocks: [[u8; 4]; 4],
}

impl ExpandedA {
    /// Creates a zeroed ExpandedA instance.
    #[inline(always)]
    pub const fn zero() -> Self {
        Self {
            a_hat: [[Poly32::zero(); 4]; 4],
            blocks: [[0; 4]; 4],
        }
    }

    /// Creates a zeroed ExpandedA instance.
    #[inline(always)]
    pub const fn new() -> Self {
        Self::zero()
    }
}

impl Default for ExpandedA {
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_types_zero_and_alignment() {
        let p = Poly32::default();
        assert_eq!(std::mem::align_of::<Poly32>(), 64);
        assert_eq!(std::mem::size_of::<Poly32>(), 1024);
        assert_eq!(p.0[0], 0);

        let b = Block::default();
        assert_eq!(std::mem::align_of::<Block>(), 64);
        assert_eq!(std::mem::size_of::<Block>(), 256);
        assert_eq!(b.bytes[0], 0);

        let r = RejState::default();
        assert_eq!(std::mem::align_of::<RejState>(), 64);
        assert_eq!(r.count, 0);

        let c = Challenge39::default();
        assert_eq!(c.neg, 0);
        assert_eq!(c.pos[0], 0);

        let f = FinishOut::default();
        assert_eq!(f.w1[0], 0);
        assert_eq!(f.u[0], 0);

        let exp = ExpandedA::default();
        assert_eq!(exp.blocks[0][0], 0);
    }
}
