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
