//! AVX-512 implementation of finish operations.

use super::{FinishError, FinishOut, FinishRow, HuCols};
use crate::poly::{Challenge39, Poly32};

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
pub(crate) unsafe fn sparse_ct(
    c: &Challenge39,
    t1: &[[u16; 256]; 4],
    shift: u32,
    out: &mut [[i32; 256]; 4],
) {
    super::scalar::sparse_ct(c, t1, shift, out)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
pub(crate) unsafe fn finish_row(
    row: FinishRow<'_>,
    out: &mut FinishOut,
) -> Result<(), FinishError> {
    super::scalar::finish_row(row, out)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
pub(crate) unsafe fn hu_columns(z: &Poly32, out: &FinishOut, cols: HuCols<'_>) {
    super::scalar::hu_columns(z, out, cols)
}
