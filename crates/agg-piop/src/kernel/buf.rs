//! Parallel first-touch allocation without an initial zeroing pass.

use std::mem::MaybeUninit;

/// Allocate slots without making assumptions about the representation of `T`.
pub(crate) fn uninit<T: Copy>(len: usize) -> Box<[MaybeUninit<T>]> {
    Box::<[T]>::new_uninit_slice(len)
}

/// Reuse an allocation as writable uninitialized slots, preserving spare capacity.
pub(crate) fn reuse<T: Copy>(mut buf: Vec<T>, len: usize) -> Vec<MaybeUninit<T>> {
    buf.clear();
    buf.reserve_exact(len);
    let mut buf = std::mem::ManuallyDrop::new(buf);
    // SAFETY: MaybeUninit<T> has T's layout, all slots may be uninitialized, and the
    // allocation is transferred exactly once. No assumption about T's representation.
    unsafe { Vec::from_raw_parts(buf.as_mut_ptr().cast(), len, buf.capacity()) }
}

/// Transfer a fully initialized vector without shrinking its allocation.
///
/// # Safety
/// Every slot of buf must contain a valid T.
pub(crate) unsafe fn finish<T: Copy>(buf: Vec<MaybeUninit<T>>) -> Vec<T> {
    let mut buf = std::mem::ManuallyDrop::new(buf);
    // SAFETY: the caller initialized every slot; MaybeUninit<T> has T's layout.
    unsafe { Vec::from_raw_parts(buf.as_mut_ptr().cast(), buf.len(), buf.capacity()) }
}
