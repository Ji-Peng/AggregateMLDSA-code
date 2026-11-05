//! Read-only views of Jolt's transparent canonical storage; ownership never changes.
use crate::{F, K};

const _: () = {
    assert!(std::mem::size_of::<F>() == 8);
    assert!(std::mem::align_of::<F>() == std::mem::align_of::<u64>());
    assert!(std::mem::size_of::<K>() == 16);
    assert!(std::mem::align_of::<K>() == std::mem::align_of::<u64>());
};
/// Fp64 is repr(transparent) over u64 in solinas/word.rs; all bit patterns are readable.
pub fn base_words(x: &[F]) -> &[u64] {
    // SAFETY: `F` is one `u64` word with `u64`'s alignment (asserted above), so `x.len()`
    // elements are `x.len()` readable words, borrowed for the lifetime of `x`.
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), x.len()) }
}
/// FpExt2 is repr(transparent) over [F; 2] in solinas/ext.rs, in c0,c1 order.
pub fn interleaved_words(x: &[K]) -> &[u64] {
    // SAFETY: `K` is two `u64` words with `u64`'s alignment (asserted above), so `x.len()`
    // elements are `2 x.len()` readable words, borrowed for the lifetime of `x`.
    unsafe {
        std::slice::from_raw_parts(
            x.as_ptr().cast(),
            x.len().checked_mul(2).expect("slice length"),
        )
    }
}
