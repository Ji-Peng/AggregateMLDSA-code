//! Parallel first-touch allocation without an initial zeroing pass.

use std::mem::MaybeUninit;
use rayon::prelude::*;

/// Allocate slots without making assumptions about the representation of `T`.
pub(crate) fn uninit<T: Copy>(len: usize) -> Box<[MaybeUninit<T>]> {
    Box::<[T]>::new_uninit_slice(len)
}

/// Fill independent blocks, then transfer ownership without copying.
///
/// # Safety
/// `fill` must initialize every slot in its slice before returning normally. A panic
/// is safe: partially initialized slots have no destructor because `T: Copy`.
pub(crate) unsafe fn fill_par<T: Copy + Send>(
    mut buf: Box<[MaybeUninit<T>]>,
    block: usize,
    fill: impl Fn(usize, &mut [MaybeUninit<T>]) + Sync + Send,
) -> Vec<T> {
    buf.par_chunks_mut(block).enumerate().for_each(|(i, slots)| fill(i, slots));
    // SAFETY: the caller guarantees that every slot was initialized by `fill`.
    unsafe { buf.assume_init().into_vec() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filled_buffer_matches_collect() {
        for len in [0, 1, 17, 4097] {
            // SAFETY: every slot in every block is written once.
            let actual = unsafe {
                fill_par(uninit::<(usize, usize)>(len), 16, |block, slots| {
                    for (i, slot) in slots.iter_mut().enumerate() {
                        let index = block * 16 + i;
                        slot.write((index, index * index));
                    }
                })
            };
            let expected: Vec<_> = (0..len).map(|i| (i, i * i)).collect();
            assert_eq!(actual, expected);
        }
    }
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
