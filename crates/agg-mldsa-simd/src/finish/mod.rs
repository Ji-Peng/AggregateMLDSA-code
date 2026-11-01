//! Finish operations: CRT combination, rounding, UseHint, quotient check, and column decomposition.

pub mod avx2;
pub(crate) mod avx512;
pub(crate) mod scalar;

use std::mem::MaybeUninit;

use crate::dispatch::{Backend, Engine};
use crate::params::ParamSet;
pub use crate::poly::{Challenge, Challenge39, FinishOut, Poly32};

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

/// Three-prime input views for finishing one polynomial row with |A z| <= 2^51.8.
#[derive(Clone, Copy, Debug)]
pub struct FinishRow3P<'a> {
    /// Inverse NTT output modulo q.
    pub xq: &'a Poly32,
    /// Inverse NTT output modulo P2.
    pub xp2: &'a Poly32,
    /// Inverse NTT output modulo P3.
    pub xp3: &'a Poly32,
    /// Shifted public key polynomial t.
    pub t: &'a [i32; 256],
    /// Hint bits (0 or 1).
    pub h: &'a [u8; 256],
}

/// Exact integer ring-product input views for finishing one polynomial row.
#[derive(Clone, Copy, Debug)]
pub struct FinishRowExact<'a> {
    /// Exact ring product A z over Z.
    pub az: &'a [i128; 256],
    /// Inverse NTT output modulo q.
    pub xq: &'a Poly32,
    /// Shifted public key polynomial t.
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

impl<P: ParamSet> Engine<P> {
    /// Computes T = (c * t1) << shift for k rows using sparse challenge representation.
    pub fn sparse_ct<const K: usize, const TAU: usize>(
        &self,
        c: &Challenge<TAU>,
        t1: &[[u16; 256]; K],
        shift: u32,
        out: &mut [[i32; 256]; K],
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

    /// Computes T = (c * t1) << shift for variable row slices using sparse challenge representation.
    pub fn sparse_ct_slice<const TAU: usize>(
        &self,
        c: &Challenge<TAU>,
        t1: &[[u16; 256]],
        shift: u32,
        out: &mut [[i32; 256]],
    ) {
        match self.backend() {
            Backend::Scalar => scalar::sparse_ct_slice(c, t1, shift, out),
            Backend::Avx2 => {
                // SAFETY: The engine guarantees AVX2 and POPCNT features are available.
                unsafe { avx2::sparse_ct_slice(c, t1, shift, out) }
            }
            Backend::Avx512 => {
                // SAFETY: The engine guarantees AVX-512 features are available.
                unsafe { avx512::sparse_ct_slice(c, t1, shift, out) }
            }
        }
    }

    /// Reconstructs A*z over Z via CRT, computes w1 via UseHint, checks quotient divisibility and bounds.
    pub fn finish_row(&self, row: FinishRow<'_>, out: &mut FinishOut) -> Result<(), FinishError> {
        match self.backend() {
            Backend::Scalar => scalar::finish_row::<P>(row, out),
            Backend::Avx2 => {
                // SAFETY: The engine guarantees AVX2 and POPCNT features are available.
                unsafe { avx2::finish_row::<P>(row, out) }
            }
            Backend::Avx512 => {
                // SAFETY: The engine guarantees AVX-512 features are available.
                unsafe { avx512::finish_row::<P>(row, out) }
            }
        }
    }

    /// Reconstructs A*z over Z via 3-prime CRT for |A z| <= 2^51.8, computes w1 via UseHint, checks quotient divisibility and bounds.
    pub fn finish_row_3p(&self, row: FinishRow3P<'_>, out: &mut FinishOut) -> Result<(), FinishError> {
        match self.backend() {
            Backend::Scalar => scalar::finish_row_3p::<P>(row, out),
            Backend::Avx2 => {
                // SAFETY: The engine guarantees AVX2 and POPCNT features are available.
                unsafe { avx2::finish_row_3p::<P>(row, out) }
            }
            Backend::Avx512 => {
                // SAFETY: The engine guarantees AVX-512 features are available.
                unsafe { avx512::finish_row_3p::<P>(row, out) }
            }
        }
    }

    /// Reconstructs A*z from exact integer ring-product coefficients, computes w1 via UseHint, checks quotient divisibility and bounds.
    pub fn finish_row_exact(&self, row: FinishRowExact<'_>, out: &mut FinishOut) -> Result<(), FinishError> {
        match self.backend() {
            Backend::Scalar => scalar::finish_row_exact::<P>(row, out),
            Backend::Avx2 => {
                // SAFETY: The engine guarantees AVX2 and POPCNT features are available.
                unsafe { avx2::finish_row_exact::<P>(row, out) }
            }
            Backend::Avx512 => {
                // SAFETY: The engine guarantees AVX-512 features are available.
                unsafe { avx512::finish_row_exact::<P>(row, out) }
            }
        }
    }

    /// Decomposes (z, u, v, w1) into the H-u relation columns.
    pub fn hu_columns(&self, z: &Poly32, out: &FinishOut, cols: HuCols<'_>) {
        match self.backend() {
            Backend::Scalar => scalar::hu_columns::<P>(z, out, cols),
            Backend::Avx2 => {
                // SAFETY: The engine guarantees AVX2 and POPCNT features are available.
                unsafe { avx2::hu_columns::<P>(z, out, cols) }
            }
            Backend::Avx512 => {
                // SAFETY: The engine guarantees AVX-512 features are available.
                unsafe { avx512::hu_columns::<P>(z, out, cols) }
            }
        }
    }
}
