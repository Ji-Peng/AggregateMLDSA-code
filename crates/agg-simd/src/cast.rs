//! Read-only views of Jolt's transparent canonical storage; ownership never changes.
use crate::{F, K};
use rayon::prelude::*;

/// Elements per chunk of the parallel encoder. Tables of at most one chunk
/// stay serial so small inputs do not pay a rayon dispatch.
const ENCODE_CHUNK: usize = 1 << 15;

const _: () = {
    assert!(std::mem::size_of::<F>() == 8);
    assert!(std::mem::align_of::<F>() == std::mem::align_of::<u64>());
    assert!(std::mem::size_of::<K>() == 16);
    assert!(std::mem::align_of::<K>() == std::mem::align_of::<u64>());
};
/// Fp64 is repr(transparent) over u64 in solinas/word.rs; all bit patterns are readable.
pub fn base_words(x: &[F]) -> &[u64] {
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), x.len()) }
}
/// FpExt2 is repr(transparent) over [F; 2] in solinas/ext.rs, in c0,c1 order.
pub fn interleaved_words(x: &[K]) -> &[u64] {
    unsafe {
        std::slice::from_raw_parts(
            x.as_ptr().cast(),
            x.len().checked_mul(2).expect("slice length"),
        )
    }
}

/// Construct Akita's owned canonical evaluations directly, including trailing zero padding.
/// The allocation has Vec<F>'s native alignment throughout; no aligned-owner cast occurs.
///
/// The output is allocated once at the full length and filled in chunks on the
/// caller's pool, so the first touch is spread over the pool. Tables of at most
/// [`ENCODE_CHUNK`] elements stay serial.
pub fn encode_i32(input: &[i32], len: usize) -> Vec<F> {
    encode(len, input.len(), Some(input), None)
}
pub fn encode_i8(input: &[i8], len: usize) -> Vec<F> {
    encode(len, input.len(), None, Some(input))
}
fn encode(len: usize, live: usize, wide: Option<&[i32]>, narrow: Option<&[i8]>) -> Vec<F> {
    assert!(live <= len);
    let mut out: Vec<F> = Vec::with_capacity(len);
    {
        let spare = &mut out.spare_capacity_mut()[..len];
        let (head, tail) = spare.split_at_mut(live);
        if live <= ENCODE_CHUNK {
            encode_chunk(wide, narrow, head);
        } else if let Some(input) = wide {
            head.par_chunks_mut(ENCODE_CHUNK)
                .zip(input.par_chunks(ENCODE_CHUNK))
                .for_each(|(dst, src)| encode_chunk(Some(src), None, dst));
        } else if let Some(input) = narrow {
            head.par_chunks_mut(ENCODE_CHUNK)
                .zip(input.par_chunks(ENCODE_CHUNK))
                .for_each(|(dst, src)| encode_chunk(None, Some(src), dst));
        }
        // The live prefix above is exactly `live` long; the rest is zero padding, which can
        // cover half of a large table when the record count is not a power of two.
        if tail.len() > ENCODE_CHUNK {
            tail.par_chunks_mut(ENCODE_CHUNK).for_each(|chunk| {
                // SAFETY: the chunk is writable, `F` is a plain word without padding, and
                // the all-zero bit pattern is the field's zero.
                unsafe { std::ptr::write_bytes(chunk.as_mut_ptr(), 0, chunk.len()) };
            });
        } else if !tail.is_empty() {
            // SAFETY: as above.
            unsafe { std::ptr::write_bytes(tail.as_mut_ptr(), 0, tail.len()) };
        }
    }
    // SAFETY: the writers above initialize all `len` slots; every value is a
    // canonical field element.
    unsafe { out.set_len(len) };
    out
}

/// Encode one chunk, where `out` has exactly one slot per input element.
fn encode_chunk(wide: Option<&[i32]>, narrow: Option<&[i8]>, out: &mut [std::mem::MaybeUninit<F>]) {
    let len = wide.map_or_else(|| narrow.map_or(0, |x| x.len()), |x| x.len());
    debug_assert_eq!(len, out.len());
    let p = out.as_mut_ptr().cast::<u64>();
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if len >= 64 && crate::detect::backend() == crate::detect::Backend::Avx512 {
        // SAFETY: `p` covers `len` initialized slots of `F` and the vector kernel
        // writes exactly `len` canonical elements from the matching input slice.
        unsafe {
            done = encode_vector(wide, narrow, p);
        }
    }
    for i in done..len {
        let x = match (wide, narrow) {
            (Some(input), _) => i64::from(input[i]),
            (None, Some(input)) => i64::from(input[i]),
            (None, None) => 0,
        };
        unsafe {
            p.add(i)
                .write((x as u64).wrapping_add(if x < 0 { crate::P } else { 0 }));
        }
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
unsafe fn encode_vector(wide: Option<&[i32]>, narrow: Option<&[i8]>, out: *mut u64) -> usize {
    unsafe {
        use std::arch::x86_64::*;
        let len = wide.map_or_else(|| narrow.unwrap().len(), |x| x.len());
        let n = len / 16 * 16;
        for i in (0..n).step_by(16) {
            for j in [i, i + 8] {
                let x = if let Some(w) = wide {
                    _mm512_cvtepi32_epi64(_mm256_loadu_si256(w.as_ptr().add(j).cast()))
                } else {
                    _mm512_cvtepi8_epi64(_mm_loadl_epi64(narrow.unwrap().as_ptr().add(j).cast()))
                };
                let x = _mm512_mask_add_epi64(
                    x,
                    _mm512_cmplt_epi64_mask(x, _mm512_setzero_si512()),
                    x,
                    _mm512_set1_epi64(crate::P as i64),
                );
                _mm512_storeu_si512(out.add(j).cast(), x);
            }
        }
        n
    }
}
