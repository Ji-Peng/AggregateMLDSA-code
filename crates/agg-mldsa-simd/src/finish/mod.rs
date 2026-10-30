//! Finish operations: CRT combination, rounding, UseHint, quotient check, and column decomposition.

pub mod avx2;
pub(crate) mod avx512;
pub(crate) mod scalar;

use std::mem::MaybeUninit;

use crate::dispatch::{Backend, Engine};
pub use crate::poly::{Challenge39, FinishOut, Poly32};

/// Input views for finishing one polynomial row of the verification relation.
#[derive(Clone, Copy, Debug)]
pub struct FinishRow<'a> {
    /// Inverse NTT output modulo q.
    pub xq: &'a Poly32,
    /// Inverse NTT output modulo P2.
    pub xp: &'a Poly32,
    /// Shifted public key polynomial t (e.g. 2^d * c * t1).
    pub t: &'a [i32; 256],
    /// Hint bits (0 or 1).
    pub h: &'a [u8; 256],
}

/// Destination slices for decomposed relation columns.
pub struct HuCols<'a> {
    /// 16-bit low digits / wide columns: Z_lo, U_lo, V_lo, V_hi (each len 256).
    pub wide: [&'a mut [MaybeUninit<i32>]; 4],
    /// Narrow columns: Z_hi, U_hi, b0..b5 (each len 256).
    pub narrow: [&'a mut [MaybeUninit<i8>]; 8],
}

/// Errors occurring during row finishing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishError {
    /// Non-zero remainder or invalid quotient in v derivation.
    Quotient,
    /// Bound check failed for residual u (|u| > B_u).
    UBound,
    /// Bound check failed for quotient v (|v| > B_v).
    VBound,
}

impl Engine {
    /// Computes T = (c * t1) << shift for 4 rows using sparse challenge representation.
    pub fn sparse_ct(
        &self,
        c: &Challenge39,
        t1: &[[u16; 256]; 4],
        shift: u32,
        out: &mut [[i32; 256]; 4],
    ) {
        match self.backend() {
            Backend::Scalar => scalar::sparse_ct(c, t1, shift, out),
            Backend::Avx2 => {
                // SAFETY: The engine guarantees AVX2 and POPCNT features are available.
                unsafe { avx2::sparse_ct(c, t1, shift, out) }
            }
            Backend::Avx512 => {
                // SAFETY: The engine guarantees AVX-512 features are available.
                unsafe { avx512::sparse_ct(c, t1, shift, out) }
            }
        }
    }

    /// Reconstructs A*z over Z via CRT, computes w1 via UseHint, checks quotient divisibility and bounds.
    pub fn finish_row(&self, row: FinishRow<'_>, out: &mut FinishOut) -> Result<(), FinishError> {
        match self.backend() {
            Backend::Scalar => scalar::finish_row(row, out),
            Backend::Avx2 => {
                // SAFETY: The engine guarantees AVX2 and POPCNT features are available.
                unsafe { avx2::finish_row(row, out) }
            }
            Backend::Avx512 => {
                // SAFETY: The engine guarantees AVX-512 features are available.
                unsafe { avx512::finish_row(row, out) }
            }
        }
    }

    /// Decomposes (z, u, v, w1) into the H-u relation columns.
    pub fn hu_columns(&self, z: &Poly32, out: &FinishOut, cols: HuCols<'_>) {
        match self.backend() {
            Backend::Scalar => scalar::hu_columns(z, out, cols),
            Backend::Avx2 => {
                // SAFETY: The engine guarantees AVX2 and POPCNT features are available.
                unsafe { avx2::hu_columns(z, out, cols) }
            }
            Backend::Avx512 => {
                // SAFETY: The engine guarantees AVX-512 features are available.
                unsafe { avx512::hu_columns(z, out, cols) }
            }
        }
    }
}
