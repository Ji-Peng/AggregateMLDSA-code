//! Architecture-selected kernels over contiguous [`F128`] slices.

pub mod rounds;

use crate::F128;
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
use crate::f128_slice::aarch64::fold_pairs as fold_pairs_aarch64;
#[cfg(any(test, not(all(target_arch = "aarch64", target_feature = "aes"))))]
use crate::f128_slice::portable::fold_pairs as fold_pairs_portable;
#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
use crate::f128_slice::x86_64::fold_pairs as fold_pairs_x86_64;

#[cfg(any(test, not(all(target_arch = "aarch64", target_feature = "aes"))))]
mod portable;
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
mod aarch64;
// The kernel uses the CLMUL product of `gf2_128::x86_64`, compiled only with PCLMULQDQ.
#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
mod x86_64;

/// Fold adjacent pairs from `src` into `dst`, starting at pair `base`.
///
/// Computes `dst[t] = src[2j] * (1 + r) + src[2j + 1] * r`, where
/// `j = base + t`. The x86 kernel is selected at run time when the CPU and ISA cap allow it.
#[inline]
pub fn fold_pairs(src: &[F128], base: usize, dst: &mut [F128], r: F128) {
    assert!(
        base <= src.len() / 2 && dst.len() <= src.len() / 2 - base,
        "fold source must contain both elements for every destination pair"
    );

    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    if rounds::x86::enabled() {
        // SAFETY: runtime feature detection and the bounds check above passed.
        unsafe { fold_pairs_x86_64(src, base, dst, r) };
        return;
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        // SAFETY: the target enables PMULL and the bounds check above passed.
        unsafe { fold_pairs_aarch64(src, base, dst, r) };
        return;
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    fold_pairs_portable(src, base, dst, r);
}

#[cfg(test)]
mod tests {
    use crate::f128_slice::{F128, fold_pairs, fold_pairs_portable};

    #[test]
    fn selected_fold_matches_portable_with_offset_and_tail() {
        let mut state = 0x243f_6a88_85a3_08d3_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let src: Vec<F128> = (0..30)
            .map(|_| F128 {
                lo: next(),
                hi: next(),
            })
            .collect();
        let r = F128 {
            lo: next(),
            hi: next(),
        };
        let mut expected = vec![F128::ZERO; 9];
        let mut actual = vec![F128::ZERO; 9];

        fold_pairs_portable(&src, 3, &mut expected, r);
        fold_pairs(&src, 3, &mut actual, r);

        assert_eq!(actual, expected);
        for (t, &value) in expected.iter().enumerate() {
            let s = 2 * (3 + t);
            assert_eq!(value, src[s] * (F128::ONE + r) + src[s + 1] * r);
        }
    }
}
