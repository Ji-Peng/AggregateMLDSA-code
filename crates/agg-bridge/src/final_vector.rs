//! The final public vector `Y`.
//!
//! `Y_sigma` is `m_3` bits per segment (4,096 in production, 512 bytes), bit `rho_3` at byte
//! `rho_3 >> 3`, bit position `rho_3 mod 8` (least significant first). The
//! verifier reads exactly `S m_3 / 8` bytes ([`parse`]), lays the segments out as the public
//! table `Y` on `D_3` (`Y_sigma[rho_3]` at `rho_3 + m_3 sigma`), and evaluates its multilinear
//! extension at `tau_3` ([`evaluate`]) for the right-hand side of the level-3 identity,
//! `Y(tau_3) + 2 kappa^(3)(tau_3)` ([`level3_claimed_sum`]).

use jolt_field::Ring;

use crate::shape::Shape;
use crate::weights::eq_table;

/// Bytes of one segment's final vector.
#[must_use]
pub const fn segment_bytes(shape: &Shape) -> usize {
    shape.output_bits() / 8
}

/// A final-vector item of the wrong length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FinalVectorLength {
    /// The expected length `S m_3 / 8`.
    pub expected: usize,
    /// The length received.
    pub actual: usize,
}

/// Pack bits (one byte each, values 0 or 1, a multiple of 8 of them), least significant first.
#[must_use]
pub fn pack(bits: &[u8]) -> Vec<u8> {
    assert_eq!(bits.len() % 8, 0, "whole bytes");
    bits.as_chunks::<8>()
        .0
        .iter()
        .map(|byte| {
            byte.iter().enumerate().fold(0_u8, |acc, (i, &bit)| {
                assert!(bit <= 1, "a bit");
                acc | (bit << i)
            })
        })
        .collect()
}

/// Unpack bytes into bits (one byte each), least significant first.
#[must_use]
pub fn unpack(bytes: &[u8]) -> Vec<u8> {
    bytes
        .iter()
        .flat_map(|&byte| (0..8).map(move |i| (byte >> i) & 1))
        .collect()
}

/// The public table `Y` on `D_3` from the final-vector bytes of `segments` segments; rejects any
/// length other than `segments m_3 / 8`.
pub fn parse(shape: &Shape, bytes: &[u8], segments: usize) -> Result<Vec<u8>, FinalVectorLength> {
    let expected = segments * segment_bytes(shape);
    if bytes.len() != expected {
        return Err(FinalVectorLength {
            expected,
            actual: bytes.len(),
        });
    }
    Ok(unpack(bytes))
}

/// `Y(point) = sum_{cells with Y = 1} eq(point, cell)`: the multilinear extension of a 0/1
/// table of `2^|point|` cells.
#[must_use]
pub fn evaluate<F: Ring>(y: &[u8], point: &[F]) -> F {
    assert_eq!(y.len(), 1 << point.len(), "one bit per cell");
    eq_table(point)
        .iter()
        .zip(y)
        .filter(|(_, &bit)| bit != 0)
        .fold(F::zero(), |acc, (&w, &bit)| {
            debug_assert_eq!(bit, 1);
            acc + w
        })
}

/// The level-3 claimed sum `Y(tau_3) + 2 kappa^(3)(tau_3)`.
#[must_use]
pub fn level3_claimed_sum<F: Ring>(y_at_tau3: F, kappa3_at_tau3: F) -> F {
    y_at_tau3 + kappa3_at_tau3 + kappa3_at_tau3
}
