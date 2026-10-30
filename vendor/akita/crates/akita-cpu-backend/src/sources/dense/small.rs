//! Direct canonical expansion for the fixed 64-bit commitment field.
use jolt_field::solinas::parallel::*;
use jolt_field::{Prime64Offset59 as F, Ring};
use std::mem::MaybeUninit;

/// Elements per chunk of the parallel expansion. The dense constructor's
/// min/max reduction uses the same chunk size, and tables of at most one
/// chunk stay serial so small groups do not pay a rayon dispatch.
const EXPAND_CHUNK: usize = 1 << 15;

/// Canonical field plane of the centered cells, written into one allocation.
///
/// The output is allocated once at the full length; its first touch happens in
/// the writers, which run on the caller's pool for tables above one chunk.
pub(super) fn expand(cells: &[i8]) -> Vec<F> {
    let mut out: Vec<F> = Vec::with_capacity(cells.len());
    {
        let spare = &mut out.spare_capacity_mut()[..cells.len()];
        if cells.len() <= EXPAND_CHUNK {
            expand_chunk(cells, spare);
        } else {
            cfg_chunks_mut!(spare, EXPAND_CHUNK)
                .zip(cfg_chunks!(cells, EXPAND_CHUNK))
                .for_each(|(dst, src)| expand_chunk(src, dst));
        }
    }
    // SAFETY: `expand_chunk` writes every slot of the output slice it receives,
    // and the disjoint chunks above cover exactly the `cells.len()` slots of the
    // allocation that `set_len` exposes.
    unsafe { out.set_len(cells.len()) };
    out
}

/// Expand one chunk into `out`, which has one slot per input byte.
fn expand_chunk(cells: &[i8], out: &mut [MaybeUninit<F>]) {
    debug_assert_eq!(cells.len(), out.len());
    for (slot, &x) in out.iter_mut().zip(cells) {
        slot.write(F::from_i64(i64::from(x)));
    }
}
