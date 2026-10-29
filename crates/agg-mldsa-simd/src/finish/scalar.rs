//! Scalar implementation of finish operations.

use super::{FinishError, FinishOut, FinishRow, HuCols};
use crate::poly::{Challenge39, Poly32};

pub(crate) fn sparse_ct(
    c: &Challenge39,
    t1: &[[u16; 256]; 4],
    shift: u32,
    out: &mut [[i32; 256]; 4],
) {
    let _ = (c, t1, shift, out);
    todo!("T2")
}

pub(crate) fn finish_row(row: FinishRow<'_>, out: &mut FinishOut) -> Result<(), FinishError> {
    let _ = (row, out);
    todo!("T2")
}

pub(crate) fn hu_columns(z: &Poly32, out: &FinishOut, cols: HuCols<'_>) {
    let _ = (z, out, cols);
    todo!("T2")
}
