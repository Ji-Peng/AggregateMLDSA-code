//! Runtime CPU-feature dispatch for the x86-64 kernels (agg-mldsa patch, PATCH.md
//! Section 3).
//!
//! Upstream Flock selects its x86 kernels with `cfg(target_feature = ...)`, so a build
//! without `-C target-cpu=native` always runs the portable kernels. This copy checks the
//! CPU once at run time (`is_x86_feature_detected!`) and calls the same
//! `#[target_feature(enable = ...)]` kernels when the CPU has the features. Every kernel
//! computes the same field elements as its portable counterpart (exact GF(2^8) and
//! GF(2^128) arithmetic), so proofs are byte-identical whichever path runs.
//!
//! `FLOCK_SIMD` in the environment caps the level, read once per process:
//! `portable` (or `off`) disables every x86 kernel, `gfni` allows only the SSE GFNI
//! kernels, anything else (or unset) allows all detected features.

use std::sync::atomic::{AtomicU8, Ordering};

/// Feature bits.
pub(crate) const AVX512_CLMUL: u8 = 1 << 0; // avx512f + vpclmulqdq
pub(crate) const AVX512_GFNI: u8 = 1 << 1; // gfni + avx512f + avx512bw
pub(crate) const GFNI: u8 = 1 << 2; // gfni (SSE)
pub(crate) const AVX512_VBMI: u8 = 1 << 3; // avx512f + avx512bw + avx512vbmi
pub(crate) const AVX512F: u8 = 1 << 4; // avx512f
const INIT: u8 = 1 << 7;

/// Detected features after the `FLOCK_SIMD` cap.
static DETECTED: AtomicU8 = AtomicU8::new(0);
/// Test-only override mask, ANDed with the detected features (`0xff` = no override).
static MASK: AtomicU8 = AtomicU8::new(0xff);

#[cold]
fn detect() -> u8 {
    #[allow(unused_mut)]
    let mut bits = 0u8;
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::is_x86_feature_detected as has;
        let avx512f = has!("avx512f");
        if avx512f {
            bits |= AVX512F;
        }
        if avx512f && has!("vpclmulqdq") {
            bits |= AVX512_CLMUL;
        }
        if has!("gfni") {
            bits |= GFNI;
            if avx512f && has!("avx512bw") {
                bits |= AVX512_GFNI;
            }
        }
        if avx512f && has!("avx512bw") && has!("avx512vbmi") {
            bits |= AVX512_VBMI;
        }
    }
    match std::env::var("FLOCK_SIMD").as_deref() {
        Ok("portable") | Ok("off") => bits = 0,
        Ok("gfni") => bits &= GFNI,
        _ => {}
    }
    DETECTED.store(bits | INIT, Ordering::Relaxed);
    bits
}

/// The features the kernels may use (detected, capped by `FLOCK_SIMD` and the test mask).
#[inline]
pub(crate) fn features() -> u8 {
    let d = DETECTED.load(Ordering::Relaxed);
    let d = if d & INIT == 0 { detect() } else { d & !INIT };
    d & MASK.load(Ordering::Relaxed)
}

/// True when every bit of `want` is available.
#[inline]
pub(crate) fn has(want: u8) -> bool {
    features() & want == want
}

/// Run `f` with the kernels limited to `mask` (tests: compare each path with the
/// portable one). Serialized by a lock; other threads see the mask while it is set,
/// which only changes which (equal-valued) kernel they run.
#[cfg(test)]
pub(crate) fn with_mask<T>(mask: u8, f: impl FnOnce() -> T) -> T {
    use std::sync::Mutex;
    static LOCK: Mutex<()> = Mutex::new(());
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let old = MASK.swap(mask, Ordering::SeqCst);
    let out = f();
    MASK.store(old, Ordering::SeqCst);
    out
}

/// The masks the machine supports, from portable (0) up to all detected features;
/// used by tests to run every available path.
#[cfg(test)]
pub(crate) fn test_masks() -> Vec<u8> {
    let all = {
        let d = DETECTED.load(Ordering::Relaxed);
        if d & INIT == 0 { detect() } else { d & !INIT }
    };
    let mut out = vec![0u8];
    for m in [GFNI, GFNI | AVX512F, AVX512F | AVX512_CLMUL, all] {
        let m = m & all;
        if !out.contains(&m) {
            out.push(m);
        }
    }
    out
}

/// AVX-512 GF(2^128) primitives, copied from `flock-field` 83f963e
/// (`gf2_128/x86_64.rs`, `f128_slice/x86_64.rs`) without its
/// `cfg(target_feature)` gates, so runtime-dispatched kernels can use them in any build.
/// Each function carries `#[target_feature(enable = ...)]`; callers must have checked
/// the features (`has(AVX512_CLMUL)`).
#[cfg(target_arch = "x86_64")]
pub(crate) mod x86 {
    use core::arch::x86_64::{
        __m128i, __m512i, _mm_extract_epi64, _mm_xor_si128,
        _mm512_bslli_epi128, _mm512_clmulepi64_epi128,
        _mm512_extracti32x4_epi32, _mm512_loadu_si512, _mm512_set_epi64, _mm512_setzero_si512, _mm512_xor_si512,
    };

    use crate::field::{F128, F256Unreduced};

    #[inline]
    #[target_feature(enable = "avx512f,vpclmulqdq")]
    fn ghash_poly_x4() -> __m512i {
        _mm512_set_epi64(0, 0x87, 0, 0x87, 0, 0x87, 0, 0x87)
    }

    #[inline]
    #[target_feature(enable = "avx512f,vpclmulqdq")]
    fn gf2_128_reduce_x4(mut t0: __m512i, t1: __m512i) -> __m512i {
        let poly = ghash_poly_x4();
        // SAFETY: avx512f is enabled for this function (the intrinsic is still
        // declared unsafe).
        t0 = _mm512_xor_si512(t0, unsafe { _mm512_bslli_epi128::<8>(t1) });
        _mm512_xor_si512(t0, _mm512_clmulepi64_epi128::<0x01>(t1, poly))
    }

    /// Four independent GF(2^128) products (lane-wise), as `flock-field`'s `ghash_mul_x4`.
    #[inline]
    #[target_feature(enable = "avx512f,vpclmulqdq")]
    pub(crate) fn ghash_mul_x4(x: __m512i, y: __m512i) -> __m512i {
        let t1a = _mm512_clmulepi64_epi128::<0x01>(x, y);
        let t1b = _mm512_clmulepi64_epi128::<0x10>(x, y);
        let mut t1 = _mm512_xor_si512(t1a, t1b);
        let t2 = _mm512_clmulepi64_epi128::<0x11>(x, y);
        t1 = gf2_128_reduce_x4(t1, t2);
        let t0 = _mm512_clmulepi64_epi128::<0x00>(x, y);
        gf2_128_reduce_x4(t0, t1)
    }

    /// Load four contiguous `F128`.
    ///
    /// # Safety
    /// `p` points to four readable `F128`; `avx512f` available.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn f128x4_loadu(p: *const F128) -> __m512i {
        // SAFETY: caller guarantees four readable F128 at p.
        unsafe { _mm512_loadu_si512(p as *const __m512i) }
    }

    /// Pack four `F128` (lane 0 = `a`).
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(crate) fn f128x4_set(a: F128, b: F128, c: F128, d: F128) -> __m512i {
        _mm512_set_epi64(
            d.hi as i64,
            d.lo as i64,
            c.hi as i64,
            c.lo as i64,
            b.hi as i64,
            b.lo as i64,
            a.hi as i64,
            a.lo as i64,
        )
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    fn xor4_lanes(v: __m512i) -> __m128i {
        let l0 = _mm512_extracti32x4_epi32::<0>(v);
        let l1 = _mm512_extracti32x4_epi32::<1>(v);
        let l2 = _mm512_extracti32x4_epi32::<2>(v);
        let l3 = _mm512_extracti32x4_epi32::<3>(v);
        _mm_xor_si128(_mm_xor_si128(l0, l1), _mm_xor_si128(l2, l3))
    }

    /// Four-lane unreduced GF(2^128) product accumulator, as `flock-field`'s `WideGhashX4`.
    #[derive(Clone, Copy)]
    pub(crate) struct WideGhashX4 {
        lo: __m512i,
        hi: __m512i,
        mid: __m512i,
    }

    impl WideGhashX4 {
        #[inline]
        #[target_feature(enable = "avx512f")]
        pub(crate) fn zero() -> Self {
            let z = _mm512_setzero_si512();
            Self {
                lo: z,
                hi: z,
                mid: z,
            }
        }

        #[inline]
        #[target_feature(enable = "avx512f,vpclmulqdq")]
        pub(crate) fn mul_acc(&mut self, x: __m512i, y: __m512i) {
            self.lo = _mm512_xor_si512(self.lo, _mm512_clmulepi64_epi128::<0x00>(x, y));
            self.hi = _mm512_xor_si512(self.hi, _mm512_clmulepi64_epi128::<0x11>(x, y));
            let m = _mm512_xor_si512(
                _mm512_clmulepi64_epi128::<0x01>(x, y),
                _mm512_clmulepi64_epi128::<0x10>(x, y),
            );
            self.mid = _mm512_xor_si512(self.mid, m);
        }

        #[inline]
        #[target_feature(enable = "avx512f,sse4.1")]
        pub(crate) fn fold(self) -> F256Unreduced {
            let lo = xor4_lanes(self.lo);
            let hi = xor4_lanes(self.hi);
            let mid = xor4_lanes(self.mid);
            let lo_lo = _mm_extract_epi64::<0>(lo) as u64;
            let lo_hi = _mm_extract_epi64::<1>(lo) as u64;
            let hi_lo = _mm_extract_epi64::<0>(hi) as u64;
            let hi_hi = _mm_extract_epi64::<1>(hi) as u64;
            let mid_lo = _mm_extract_epi64::<0>(mid) as u64;
            let mid_hi = _mm_extract_epi64::<1>(mid) as u64;
            F256Unreduced {
                r0: lo_lo,
                r1: lo_hi ^ mid_lo,
                r2: hi_lo ^ mid_hi,
                r3: hi_hi,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{features, test_masks, with_mask};

    #[test]
    fn masks_cap_the_features() {
        let masks = test_masks();
        assert_eq!(masks[0], 0);
        for m in masks {
            assert_eq!(with_mask(m, features) & !m, 0);
        }
        eprintln!("detected x86 kernel features: {:#07b}", features());
    }
}
