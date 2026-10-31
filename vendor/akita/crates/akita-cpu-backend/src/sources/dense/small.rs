//! Direct canonical expansion of signed cells into the commitment field.
use jolt_field::solinas::parallel::*;
use jolt_field::Field;
use std::mem::MaybeUninit;

/// Elements per chunk of the parallel expansion. The dense constructor's
/// min/max reduction uses the same chunk size, and tables of at most one
/// chunk stay serial so small groups do not pay a rayon dispatch.
const EXPAND_CHUNK: usize = 1 << 15;

/// Canonical field plane of centered `i8` cells, written into one allocation.
///
/// The output is allocated once at the full length; its first touch happens in
/// the writers, which run on the caller's pool for tables above one chunk.
pub(super) fn expand_i8<F: Field>(cells: &[i8]) -> Vec<F> {
    expand_cells(cells, |cell| F::from_i8(cell))
}

/// Canonical field plane of signed `i16` cells, written into one allocation.
///
/// The output is allocated once at the full length; its first touch happens in
/// the writers, which run on the caller's pool for tables above one chunk.
pub(super) fn expand_i16<F: Field>(cells: &[i16]) -> Vec<F> {
    expand_cells(cells, |cell| F::from_i16(cell))
}

/// Convert every cell into one output slot, in chunks on the caller's pool.
fn expand_cells<T: Copy + Send + Sync, F: Field>(
    cells: &[T],
    convert: impl Fn(T) -> F + Sync,
) -> Vec<F> {
    let mut out: Vec<F> = Vec::with_capacity(cells.len());
    {
        let spare = &mut out.spare_capacity_mut()[..cells.len()];
        if cells.len() <= EXPAND_CHUNK {
            expand_chunk(cells, spare, &convert);
        } else {
            cfg_chunks_mut!(spare, EXPAND_CHUNK)
                .zip(cfg_chunks!(cells, EXPAND_CHUNK))
                .for_each(|(dst, src)| expand_chunk(src, dst, &convert));
        }
    }
    // SAFETY: `expand_chunk` writes every slot of the output slice it receives,
    // and the disjoint chunks above cover exactly the `cells.len()` slots of the
    // allocation that `set_len` exposes.
    unsafe { out.set_len(cells.len()) };
    out
}

/// Expand one chunk into `out`, which has one slot per input cell.
fn expand_chunk<T: Copy, F: Field>(
    cells: &[T],
    out: &mut [MaybeUninit<F>],
    convert: &(impl Fn(T) -> F + Sync),
) {
    debug_assert_eq!(cells.len(), out.len());
    for (slot, &cell) in out.iter_mut().zip(cells) {
        slot.write(convert(cell));
    }
}
