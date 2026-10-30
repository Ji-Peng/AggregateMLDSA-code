//! Canonical expansion for the fixed 64-bit commitment field.
use jolt_field::{Prime64Offset59 as F, Ring};

/// Canonical field plane of the centered cells.
///
/// This build has no SIMD encoder, so the portable `Field::from_i64` path is
/// used for every input; the parallel writers that fill the output are added by
/// the chunked-constructor change.
pub(super) fn expand(cells: &[i8]) -> Vec<F> {
    cells.iter().map(|&x| F::from_i64(i64::from(x))).collect()
}
