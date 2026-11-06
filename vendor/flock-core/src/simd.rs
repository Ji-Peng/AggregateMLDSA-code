//! Runtime CPU-feature dispatch for the x86-64 kernels (agg-mldsa patch, PATCH.md).
//!
//! Upstream Flock selects its x86 kernels with `cfg(target_feature = ...)`, so a build
//! without `-C target-cpu=native` always runs the portable kernels. This copy checks the
//! CPU once at run time (`is_x86_feature_detected!`) and calls the same
//! `#[target_feature(enable = ...)]` kernels when the CPU has the features. Every kernel
//! computes the same field elements as its portable counterpart (exact GF(2^8) and
//! GF(2^128) arithmetic), so proofs are byte-identical whichever path runs.
//!
//! Two environment variables cap the level, both read once per process:
//!
//! * `FLOCK_SIMD`: `portable` (or `off`) disables every x86 kernel and `gfni` allows only
//!   the SSE GFNI kernels; anything else (or unset) allows all detected features.
//! * `AGG_ISA`: the process-wide instruction-set cap. `scalar` behaves like
//!   `FLOCK_SIMD=portable`; `avx2` allows the AVX2 tier (x86-64-v3 plus PCLMULQDQ; the
//!   256-bit VPCLMULQDQ and GFNI extensions are used when the CPU has them); `avx512`
//!   allows AVX-512 without IFMA; `avx512ifma` (or unset) allows every detected feature.
//!   A cap only lowers the level.

use std::sync::atomic::{AtomicU16, Ordering};

/// Feature bits.
pub(crate) const AVX512_CLMUL: u16 = 1 << 0; // avx512f + vpclmulqdq
pub(crate) const AVX512_GFNI: u16 = 1 << 1; // gfni + avx512f + avx512bw
pub(crate) const GFNI: u16 = 1 << 2; // gfni (SSE)
pub(crate) const AVX512_VBMI: u16 = 1 << 3; // avx512f + avx512bw + avx512vbmi
pub(crate) const AVX512F: u16 = 1 << 4; // avx512f
pub(crate) const AVX2: u16 = 1 << 5; // avx2 + pclmulqdq
pub(crate) const VPCLMUL256: u16 = 1 << 6; // avx2 + vpclmulqdq (256-bit vectors)
pub(crate) const AVX512BW: u16 = 1 << 7; // avx512f + avx512bw (16-way BLAKE3 leaves)
const INIT: u16 = 1 << 15;

/// Instruction-set levels of `AGG_ISA`, ordered upwards.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Isa {
    Scalar,
    Avx2,
    Avx512,
    Avx512Ifma,
}

impl Isa {
    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "scalar" => Some(Self::Scalar),
            "avx2" => Some(Self::Avx2),
            "avx512" | "avx-512" => Some(Self::Avx512),
            "avx512ifma" | "avx512-ifma" | "ifma" => Some(Self::Avx512Ifma),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2",
            Self::Avx512 => "avx512",
            Self::Avx512Ifma => "avx512ifma",
        }
    }
}

/// The `AGG_ISA` cap, parsed once.
fn isa_cap() -> Isa {
    static CAP: std::sync::OnceLock<Isa> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        // An empty value is treated like an unset variable.
        match std::env::var("AGG_ISA")
            .ok()
            .filter(|value| !value.trim().is_empty())
        {
            Some(value) => Isa::parse(&value).unwrap_or_else(|| {
                panic!(
                    "invalid AGG_ISA '{value}': expected avx512ifma, avx512, avx2 or scalar"
                )
            }),
            None => Isa::Avx512Ifma,
        }
    })
}

/// The AGG_ISA value this process was given, for reports.
fn isa_requested() -> Option<&'static str> {
    // An empty value is treated like an unset variable.
    let value = std::env::var("AGG_ISA")
        .ok()
        .filter(|value| !value.trim().is_empty())?;
    Some(Isa::parse(&value)?.name())
}

/// Detected features after the `FLOCK_SIMD` cap.
static DETECTED: AtomicU16 = AtomicU16::new(0);
/// Test-only override mask, ANDed with the detected features (`!0` = no override).
static MASK: AtomicU16 = AtomicU16::new(!0);

#[cold]
fn detect() -> u16 {
    #[allow(unused_mut)]
    let mut bits = 0u16;
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::is_x86_feature_detected as has;
        let cap = isa_cap();
        if cap >= Isa::Avx2 {
            if has!("avx2") && has!("pclmulqdq") {
                bits |= AVX2;
                if has!("vpclmulqdq") {
                    bits |= VPCLMUL256;
                }
            }
            if has!("gfni") {
                bits |= GFNI;
            }
        }
        if cap >= Isa::Avx512 {
            let avx512f = has!("avx512f");
            if avx512f {
                bits |= AVX512F;
                if has!("vpclmulqdq") {
                    bits |= AVX512_CLMUL;
                }
            }
            if has!("gfni") && avx512f && has!("avx512bw") {
                bits |= AVX512_GFNI;
            }
            if avx512f && has!("avx512bw") && has!("avx512vbmi") {
                bits |= AVX512_VBMI;
            }
            if avx512f && has!("avx512bw") {
                bits |= AVX512BW;
            }
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
pub(crate) fn features() -> u16 {
    let d = DETECTED.load(Ordering::Relaxed);
    let d = if d & INIT == 0 { detect() } else { d & !INIT };
    d & MASK.load(Ordering::Relaxed)
}

/// True when every bit of `want` is available.
#[inline]
pub(crate) fn has(want: u16) -> bool {
    features() & want == want
}

/// Run `f` with the kernels limited to `mask` (tests: compare each path with the
/// portable one). Serialized by a lock; other threads see the mask while it is set,
/// which only changes which (equal-valued) kernel they run.
#[cfg(test)]
pub(crate) fn with_mask<T>(mask: u16, f: impl FnOnce() -> T) -> T {
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
pub(crate) fn test_masks() -> Vec<u16> {
    let all = {
        let d = DETECTED.load(Ordering::Relaxed);
        if d & INIT == 0 { detect() } else { d & !INIT }
    };
    let mut out = vec![0u16];
    for m in [
        GFNI,
        AVX2,
        AVX2 | VPCLMUL256,
        GFNI | AVX2 | VPCLMUL256,
        GFNI | AVX2 | AVX512F,
        AVX512F | AVX512_CLMUL | AVX2 | VPCLMUL256,
        AVX512F | AVX512BW,
        all,
    ] {
        let m = m & all;
        if !out.contains(&m) {
            out.push(m);
        }
    }
    out
}

/// The GF(2^128) vector width the kernels use, best first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GhashWidth {
    /// AVX-512 with 512-bit VPCLMULQDQ.
    Avx512,
    /// AVX2 with 256-bit VPCLMULQDQ.
    Vpclmul256,
    /// AVX2 with the 128-bit PCLMULQDQ.
    Pclmul,
    /// Portable.
    Portable,
}

/// The best GF(2^128) width the cap and the CPU allow.
#[inline]
pub(crate) fn ghash_width() -> GhashWidth {
    if has(AVX512_CLMUL) {
        GhashWidth::Avx512
    } else if has(VPCLMUL256) {
        GhashWidth::Vpclmul256
    } else if has(AVX2) {
        GhashWidth::Pclmul
    } else {
        GhashWidth::Portable
    }
}

/// Lanes per vector of the selected GF(2^128) width (1 on the portable path).
#[inline]
pub(crate) fn ghash_lanes() -> usize {
    match ghash_width() {
        GhashWidth::Avx512 => 4,
        GhashWidth::Vpclmul256 => 2,
        GhashWidth::Pclmul => 1,
        GhashWidth::Portable => 1,
    }
}

/// One line describing the cap and the selected x86 kernels.
pub(crate) fn report() -> String {
    let bits = features();
    let cap = isa_cap();
    format!(
        "flock-core: cap={} ({}) avx2={} vpclmul256={} gfni={} avx512f={} avx512clmul={} avx512gfni={} avx512vbmi={} avx512bw={} gf2-width={:?}",
        cap.name(),
        isa_requested().unwrap_or("detected"),
        bits & AVX2 == AVX2,
        bits & VPCLMUL256 == VPCLMUL256,
        bits & GFNI == GFNI,
        bits & AVX512F == AVX512F,
        bits & AVX512_CLMUL == AVX512_CLMUL,
        bits & AVX512_GFNI == AVX512_GFNI,
        bits & AVX512_VBMI == AVX512_VBMI,
        bits & AVX512BW == AVX512BW,
        ghash_width(),
    )
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
        _mm512_clmulepi64_epi128,
        _mm512_extracti32x4_epi32, _mm512_loadu_si512, _mm512_set_epi64, _mm512_setzero_si512, _mm512_storeu_si512, _mm512_unpacklo_epi64, _mm512_xor_si512,
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
        // `t1 << 64` per 128-bit lane: {0, t1.lo}. The AVX-512F unpack keeps the
        // shift inline; the byte shift is AVX512BW and would be an out-of-line call.
        t0 = _mm512_xor_si512(t0, _mm512_unpacklo_epi64(_mm512_setzero_si512(), t1));
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

    /// Broadcast one `F128` to the four lanes.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(crate) fn f128x4_splat(a: F128) -> __m512i {
        _mm512_set_epi64(
            a.hi as i64,
            a.lo as i64,
            a.hi as i64,
            a.lo as i64,
            a.hi as i64,
            a.lo as i64,
            a.hi as i64,
            a.lo as i64,
        )
    }

    /// Store four `F128`.
    ///
    /// # Safety
    /// `p` points to four writable `F128`; `avx512f` available.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn f128x4_storeu(p: *mut F128, v: __m512i) {
        // SAFETY: caller guarantees four writable F128 at p.
        unsafe { _mm512_storeu_si512(p as *mut __m512i, v) }
    }

    /// Lane-wise deferred-reduction accumulator: lane `i` holds the unreduced sum of
    /// the products accumulated in lane `i`; [`Self::reduce`] reduces each lane (the
    /// same operation sequence as [`ghash_mul_x4`], applied to the sums; reduction is
    /// F_2-linear, so this equals the sum of the reduced products).
    #[derive(Clone, Copy)]
    pub(crate) struct LaneAccX4 {
        lo: __m512i,
        mid: __m512i,
        hi: __m512i,
    }

    impl LaneAccX4 {
        #[inline]
        #[target_feature(enable = "avx512f")]
        pub(crate) fn zero() -> Self {
            let z = _mm512_setzero_si512();
            Self {
                lo: z,
                mid: z,
                hi: z,
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
        #[target_feature(enable = "avx512f,vpclmulqdq")]
        pub(crate) fn reduce(self) -> __m512i {
            gf2_128_reduce_x4(self.lo, gf2_128_reduce_x4(self.mid, self.hi))
        }
    }

    /// Bit transpose of eight 64-bit words: `out[8 k + j]` bit `r` is bit
    /// `8 k + j` of `v[r]` (an 8x8 byte transpose followed by the three
    /// mask/shift rounds on each 64-bit lane).
    ///
    /// # Safety
    /// x86-64; SSE2 is baseline.
    #[inline]
    #[target_feature(enable = "sse2")]
    pub(crate) unsafe fn bit_transpose_8x64_sse2(v: &[u64; 8], out: &mut [u8; 64]) {
        use core::arch::x86_64::{
            __m128i, _mm_and_si128, _mm_loadl_epi64, _mm_set1_epi64x, _mm_slli_epi64,
            _mm_srli_epi64, _mm_storeu_si128, _mm_unpackhi_epi16, _mm_unpackhi_epi32,
            _mm_unpacklo_epi16, _mm_unpacklo_epi32, _mm_unpacklo_epi8, _mm_xor_si128,
        };
        // SAFETY: `v` and `out` are 64 bytes; the row loads stay inside `v`
        // and the stores inside `out`.
        unsafe {
            let row = |i: usize| _mm_loadl_epi64(v.as_ptr().add(i).cast::<__m128i>());
            let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
            let (r4, r5, r6, r7) = (row(4), row(5), row(6), row(7));
            let t0 = _mm_unpacklo_epi8(r0, r1);
            let t1 = _mm_unpacklo_epi8(r2, r3);
            let t2 = _mm_unpacklo_epi8(r4, r5);
            let t3 = _mm_unpacklo_epi8(r6, r7);
            let u0 = _mm_unpacklo_epi16(t0, t1);
            let u1 = _mm_unpackhi_epi16(t0, t1);
            let u2 = _mm_unpacklo_epi16(t2, t3);
            let u3 = _mm_unpackhi_epi16(t2, t3);
            let columns = [
                _mm_unpacklo_epi32(u0, u2),
                _mm_unpackhi_epi32(u0, u2),
                _mm_unpacklo_epi32(u1, u3),
                _mm_unpackhi_epi32(u1, u3),
            ];

            let mask1 = _mm_set1_epi64x(0x00AA_00AA_00AA_00AAu64 as i64);
            let mask2 = _mm_set1_epi64x(0x0000_CCCC_0000_CCCCu64 as i64);
            let mask3 = _mm_set1_epi64x(0x0000_0000_F0F0_F0F0u64 as i64);
            for (k, mut g) in columns.into_iter().enumerate() {
                let t = _mm_and_si128(_mm_xor_si128(g, _mm_srli_epi64::<7>(g)), mask1);
                g = _mm_xor_si128(g, _mm_xor_si128(t, _mm_slli_epi64::<7>(t)));
                let t = _mm_and_si128(_mm_xor_si128(g, _mm_srli_epi64::<14>(g)), mask2);
                g = _mm_xor_si128(g, _mm_xor_si128(t, _mm_slli_epi64::<14>(t)));
                let t = _mm_and_si128(_mm_xor_si128(g, _mm_srli_epi64::<28>(g)), mask3);
                g = _mm_xor_si128(g, _mm_xor_si128(t, _mm_slli_epi64::<28>(t)));
                _mm_storeu_si128(out.as_mut_ptr().add(16 * k).cast::<__m128i>(), g);
            }
        }
    }

    /// [`bit_transpose_8x64_sse2`] with the byte transpose kept in two
    /// `__m256i`: the three mask/shift rounds run on four 64-bit lanes at a
    /// time instead of two, and only the final two stores leave registers.
    ///
    /// # Safety
    /// x86-64 with `avx2`; `v` and `out` are 64 bytes.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn bit_transpose_8x64_avx2(v: &[u64; 8], out: &mut [u8; 64]) {
        use core::arch::x86_64::{
            __m128i, __m256i, _mm_loadl_epi64, _mm_unpackhi_epi16, _mm_unpackhi_epi32,
            _mm_unpacklo_epi16, _mm_unpacklo_epi32, _mm_unpacklo_epi8, _mm256_and_si256,
            _mm256_set1_epi64x, _mm256_set_m128i, _mm256_slli_epi64, _mm256_srli_epi64,
            _mm256_storeu_si256, _mm256_xor_si256,
        };
        // SAFETY: `v` and `out` are 64 bytes; the row loads stay inside `v`
        // and the two 32-byte stores inside `out`.
        unsafe {
            let row = |i: usize| _mm_loadl_epi64(v.as_ptr().add(i).cast::<__m128i>());
            let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
            let (r4, r5, r6, r7) = (row(4), row(5), row(6), row(7));
            let t0 = _mm_unpacklo_epi8(r0, r1);
            let t1 = _mm_unpacklo_epi8(r2, r3);
            let t2 = _mm_unpacklo_epi8(r4, r5);
            let t3 = _mm_unpacklo_epi8(r6, r7);
            let u0 = _mm_unpacklo_epi16(t0, t1);
            let u1 = _mm_unpackhi_epi16(t0, t1);
            let u2 = _mm_unpacklo_epi16(t2, t3);
            let u3 = _mm_unpackhi_epi16(t2, t3);
            let y0 = _mm256_set_m128i(
                _mm_unpackhi_epi32(u0, u2),
                _mm_unpacklo_epi32(u0, u2),
            );
            let y1 = _mm256_set_m128i(
                _mm_unpackhi_epi32(u1, u3),
                _mm_unpacklo_epi32(u1, u3),
            );

            let mask1 = _mm256_set1_epi64x(0x00AA_00AA_00AA_00AAu64 as i64);
            let mask2 = _mm256_set1_epi64x(0x0000_CCCC_0000_CCCCu64 as i64);
            let mask3 = _mm256_set1_epi64x(0x0000_0000_F0F0_F0F0u64 as i64);
            for (k, mut g) in [y0, y1].into_iter().enumerate() {
                let t = _mm256_and_si256(_mm256_xor_si256(g, _mm256_srli_epi64::<7>(g)), mask1);
                g = _mm256_xor_si256(g, _mm256_xor_si256(t, _mm256_slli_epi64::<7>(t)));
                let t = _mm256_and_si256(_mm256_xor_si256(g, _mm256_srli_epi64::<14>(g)), mask2);
                g = _mm256_xor_si256(g, _mm256_xor_si256(t, _mm256_slli_epi64::<14>(t)));
                let t = _mm256_and_si256(_mm256_xor_si256(g, _mm256_srli_epi64::<28>(g)), mask3);
                g = _mm256_xor_si256(g, _mm256_xor_si256(t, _mm256_slli_epi64::<28>(t)));
                _mm256_storeu_si256(out.as_mut_ptr().add(32 * k).cast::<__m256i>(), g);
            }
        }
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

/// Two-lane GF(2^128) primitives in 256-bit registers (VPCLMULQDQ).
///
/// The function set mirrors [`x86`] so a kernel body can be instantiated for
/// either width by a macro. Every function carries its own
/// `#[target_feature]`; callers check [`VPCLMUL256`].
#[cfg(target_arch = "x86_64")]
pub(crate) mod ghash2 {
    use core::arch::x86_64::{
        __m256i, _mm_extract_epi64, _mm_xor_si128, _mm256_bslli_epi128,
        _mm256_clmulepi64_epi128, _mm256_extracti128_si256, _mm256_loadu_si256,
        _mm256_permute2x128_si256, _mm256_set_epi64x, _mm256_setzero_si256,
        _mm256_storeu_si256, _mm256_xor_si256,
    };

    use crate::field::{F128, F256Unreduced};

    /// The vector type of this width.
    pub(crate) type V = __m256i;
    /// Lanes per vector.
    pub(crate) const LANES: usize = 2;

    #[inline]
    #[target_feature(enable = "avx2,vpclmulqdq")]
    fn ghash_poly() -> __m256i {
        _mm256_set_epi64x(0, 0x87, 0, 0x87)
    }

    #[inline]
    #[target_feature(enable = "avx2,vpclmulqdq")]
    fn reduce(t0: __m256i, t1: __m256i) -> __m256i {
        let poly = ghash_poly();
        let shifted = _mm256_bslli_epi128::<8>(t1);
        let folded = _mm256_clmulepi64_epi128::<0x01>(t1, poly);
        _mm256_xor_si256(_mm256_xor_si256(t0, shifted), folded)
    }

    /// Two independent GF(2^128) products, lane-wise.
    #[inline]
    #[target_feature(enable = "avx2,vpclmulqdq")]
    pub(crate) fn mul(x: __m256i, y: __m256i) -> __m256i {
        let t1 = _mm256_xor_si256(
            _mm256_clmulepi64_epi128::<0x01>(x, y),
            _mm256_clmulepi64_epi128::<0x10>(x, y),
        );
        let t2 = _mm256_clmulepi64_epi128::<0x11>(x, y);
        let high = reduce(t1, t2);
        reduce(_mm256_clmulepi64_epi128::<0x00>(x, y), high)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn xor(a: __m256i, b: __m256i) -> __m256i {
        _mm256_xor_si256(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn zero() -> __m256i {
        _mm256_setzero_si256()
    }

    /// Broadcast one `F128` to both lanes.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn splat(a: F128) -> __m256i {
        _mm256_set_epi64x(a.hi as i64, a.lo as i64, a.hi as i64, a.lo as i64)
    }

    /// Pack two `F128` (lane 0 = `a`).
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn set(a: F128, b: F128) -> __m256i {
        _mm256_set_epi64x(b.hi as i64, b.lo as i64, a.hi as i64, a.lo as i64)
    }

    /// Pack the lanes of `a` in order.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn set_from(a: &[F128]) -> __m256i {
        debug_assert_eq!(a.len(), LANES);
        set(a[0], a[1])
    }

    /// Split `2 * LANES` consecutive `F128` (two vectors) into its
    /// even-indexed and odd-indexed halves, each in lane order.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn even_odd(v0: __m256i, v1: __m256i) -> (__m256i, __m256i) {
        (
            _mm256_permute2x128_si256::<0x20>(v0, v1),
            _mm256_permute2x128_si256::<0x31>(v0, v1),
        )
    }

    /// Load two contiguous `F128`.
    ///
    /// # Safety
    /// Two readable `F128` at `p`; the caller checked [`super::VPCLMUL256`].
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn loadu(p: *const F128) -> __m256i {
        // SAFETY: caller guarantees two readable F128 at p.
        unsafe { _mm256_loadu_si256(p as *const __m256i) }
    }

    /// Store two `F128`.
    ///
    /// # Safety
    /// Two writable `F128` at `p`.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn storeu(p: *mut F128, v: __m256i) {
        // SAFETY: caller guarantees two writable F128 at p.
        unsafe { _mm256_storeu_si256(p as *mut __m256i, v) }
    }

    /// Lane-wise deferred-reduction accumulator over 256-bit products.
    #[derive(Clone, Copy)]
    pub(crate) struct Wide {
        lo: __m256i,
        mid: __m256i,
        hi: __m256i,
    }

    #[inline]
    #[target_feature(enable = "avx2,vpclmulqdq")]
    pub(crate) fn wide_zero() -> Wide {
        Wide {
            lo: zero(),
            mid: zero(),
            hi: zero(),
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,vpclmulqdq")]
    pub(crate) fn wide_acc(acc: &mut Wide, x: __m256i, y: __m256i) {
        acc.lo = _mm256_xor_si256(acc.lo, _mm256_clmulepi64_epi128::<0x00>(x, y));
        acc.hi = _mm256_xor_si256(acc.hi, _mm256_clmulepi64_epi128::<0x11>(x, y));
        let m = _mm256_xor_si256(
            _mm256_clmulepi64_epi128::<0x01>(x, y),
            _mm256_clmulepi64_epi128::<0x10>(x, y),
        );
        acc.mid = _mm256_xor_si256(acc.mid, m);
    }

    /// Reduce each lane of the accumulator, keeping the two lanes separate.
    #[inline]
    #[target_feature(enable = "avx2,vpclmulqdq")]
    pub(crate) fn wide_reduce(acc: Wide) -> __m256i {
        reduce(acc.lo, reduce(acc.mid, acc.hi))
    }

    /// Fold both lanes of the accumulator into one unreduced product.
    #[inline]
    #[target_feature(enable = "avx2,vpclmulqdq")]
    pub(crate) fn wide_fold(acc: Wide) -> F256Unreduced {
        let lane = |v: __m256i| {
            _mm_xor_si128(
                _mm256_extracti128_si256::<0>(v),
                _mm256_extracti128_si256::<1>(v),
            )
        };
        let lo = lane(acc.lo);
        let mid = lane(acc.mid);
        let hi = lane(acc.hi);
        let (lo_lo, lo_hi, mid_lo, mid_hi, hi_lo, hi_hi) = (
            _mm_extract_epi64::<0>(lo) as u64,
            _mm_extract_epi64::<1>(lo) as u64,
            _mm_extract_epi64::<0>(mid) as u64,
            _mm_extract_epi64::<1>(mid) as u64,
            _mm_extract_epi64::<0>(hi) as u64,
            _mm_extract_epi64::<1>(hi) as u64,
        );
        F256Unreduced {
            r0: lo_lo,
            r1: lo_hi ^ mid_lo,
            r2: hi_lo ^ mid_hi,
            r3: hi_hi,
        }
    }
}

/// One-lane GF(2^128) primitives in 128-bit registers (PCLMULQDQ).
///
/// The function set mirrors [`x86`] and [`ghash2`]. Callers check [`AVX2`].
#[cfg(target_arch = "x86_64")]
pub(crate) mod ghash1 {
    use core::arch::x86_64::{
        __m128i, _mm_clmulepi64_si128, _mm_extract_epi64, _mm_loadu_si128, _mm_set_epi64x,
        _mm_setzero_si128, _mm_slli_si128, _mm_storeu_si128, _mm_xor_si128,
    };

    use crate::field::{F128, F256Unreduced};

    /// The vector type of this width.
    pub(crate) type V = __m128i;
    /// Lanes per vector.
    pub(crate) const LANES: usize = 1;

    #[inline]
    #[target_feature(enable = "pclmulqdq")]
    fn ghash_poly() -> __m128i {
        _mm_set_epi64x(0, 0x87)
    }

    #[inline]
    #[target_feature(enable = "pclmulqdq")]
    fn reduce(t0: __m128i, t1: __m128i) -> __m128i {
        let poly = ghash_poly();
        let folded = _mm_clmulepi64_si128::<0x01>(t1, poly);
        _mm_xor_si128(_mm_xor_si128(t0, _mm_slli_si128::<8>(t1)), folded)
    }

    /// One GF(2^128) product.
    #[inline]
    #[target_feature(enable = "pclmulqdq")]
    pub(crate) fn mul(x: __m128i, y: __m128i) -> __m128i {
        let t1 = _mm_xor_si128(
            _mm_clmulepi64_si128::<0x01>(x, y),
            _mm_clmulepi64_si128::<0x10>(x, y),
        );
        let t2 = _mm_clmulepi64_si128::<0x11>(x, y);
        let high = reduce(t1, t2);
        reduce(_mm_clmulepi64_si128::<0x00>(x, y), high)
    }

    #[inline]
    #[target_feature(enable = "pclmulqdq")]
    pub(crate) fn xor(a: __m128i, b: __m128i) -> __m128i {
        _mm_xor_si128(a, b)
    }

    #[inline]
    #[target_feature(enable = "pclmulqdq")]
    pub(crate) fn zero() -> __m128i {
        _mm_setzero_si128()
    }

    #[inline]
    #[target_feature(enable = "pclmulqdq")]
    pub(crate) fn splat(a: F128) -> __m128i {
        _mm_set_epi64x(a.hi as i64, a.lo as i64)
    }

    #[inline]
    #[target_feature(enable = "pclmulqdq")]
    pub(crate) fn set(a: F128) -> __m128i {
        _mm_set_epi64x(a.hi as i64, a.lo as i64)
    }

    /// Pack the lanes of `a` in order.
    #[inline]
    #[target_feature(enable = "pclmulqdq")]
    pub(crate) fn set_from(a: &[F128]) -> __m128i {
        debug_assert_eq!(a.len(), LANES);
        set(a[0])
    }

    /// Split two consecutive `F128` into its even and odd element; with one
    /// lane per vector the two inputs already are the halves.
    #[inline]
    #[target_feature(enable = "pclmulqdq")]
    pub(crate) fn even_odd(v0: __m128i, v1: __m128i) -> (__m128i, __m128i) {
        (v0, v1)
    }

    /// Load one `F128`.
    ///
    /// # Safety
    /// One readable `F128` at `p`; the caller checked [`super::AVX2`].
    #[inline]
    #[target_feature(enable = "pclmulqdq")]
    pub(crate) unsafe fn loadu(p: *const F128) -> __m128i {
        // SAFETY: caller guarantees one readable F128 at p.
        unsafe { _mm_loadu_si128(p as *const __m128i) }
    }

    /// Store one `F128`.
    ///
    /// # Safety
    /// One writable `F128` at `p`.
    #[inline]
    #[target_feature(enable = "pclmulqdq")]
    pub(crate) unsafe fn storeu(p: *mut F128, v: __m128i) {
        // SAFETY: caller guarantees one writable F128 at p.
        unsafe { _mm_storeu_si128(p as *mut __m128i, v) }
    }

    /// Deferred-reduction accumulator over one lane.
    #[derive(Clone, Copy)]
    pub(crate) struct Wide {
        lo: __m128i,
        mid: __m128i,
        hi: __m128i,
    }

    #[inline]
    #[target_feature(enable = "pclmulqdq")]
    pub(crate) fn wide_zero() -> Wide {
        Wide {
            lo: zero(),
            mid: zero(),
            hi: zero(),
        }
    }

    #[inline]
    #[target_feature(enable = "pclmulqdq")]
    pub(crate) fn wide_acc(acc: &mut Wide, x: __m128i, y: __m128i) {
        acc.lo = _mm_xor_si128(acc.lo, _mm_clmulepi64_si128::<0x00>(x, y));
        acc.hi = _mm_xor_si128(acc.hi, _mm_clmulepi64_si128::<0x11>(x, y));
        let m = _mm_xor_si128(
            _mm_clmulepi64_si128::<0x01>(x, y),
            _mm_clmulepi64_si128::<0x10>(x, y),
        );
        acc.mid = _mm_xor_si128(acc.mid, m);
    }

    /// Fold the accumulator into one unreduced product.
    ///
    /// # Safety
    /// The four 64-bit words of each accumulator member are readable.
    #[inline]
    #[target_feature(enable = "pclmulqdq,sse4.1")]
    pub(crate) fn wide_fold(acc: Wide) -> F256Unreduced {
        let (lo_lo, lo_hi, mid_lo, mid_hi, hi_lo, hi_hi) = (
            _mm_extract_epi64::<0>(acc.lo) as u64,
            _mm_extract_epi64::<1>(acc.lo) as u64,
            _mm_extract_epi64::<0>(acc.mid) as u64,
            _mm_extract_epi64::<1>(acc.mid) as u64,
            _mm_extract_epi64::<0>(acc.hi) as u64,
            _mm_extract_epi64::<1>(acc.hi) as u64,
        );
        F256Unreduced {
            r0: lo_lo,
            r1: lo_hi ^ mid_lo,
            r2: hi_lo ^ mid_hi,
            r3: hi_hi,
        }
    }
}

/// Four-lane GF(2^128) primitives with the uniform interface of [`ghash2`].
///
/// Re-exports [`x86`] behind the names a width-generic kernel body uses.
#[cfg(target_arch = "x86_64")]
pub(crate) mod ghash4 {
    use core::arch::x86_64::{
        __m512i, _mm512_permutex2var_epi64, _mm512_set_epi64, _mm512_setzero_si512,
        _mm512_xor_si512,
    };

    use crate::field::{F128, F256Unreduced};

    #[allow(unused_imports)]
    pub(crate) use super::x86::WideGhashX4 as Wide;
    #[allow(unused_imports)]
    pub(crate) use super::x86::{f128x4_loadu as loadu, f128x4_splat as splat};
    #[allow(unused_imports)]
    pub(crate) use super::x86::{f128x4_set, f128x4_storeu as storeu, ghash_mul_x4 as mul};

    /// The vector type of this width.
    pub(crate) type V = __m512i;
    /// Lanes per vector.
    pub(crate) const LANES: usize = 4;

    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn xor(a: __m512i, b: __m512i) -> __m512i {
        _mm512_xor_si512(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn zero() -> __m512i {
        _mm512_setzero_si512()
    }

    /// Pack the lanes of `a` in order.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn set_from(a: &[F128]) -> __m512i {
        debug_assert_eq!(a.len(), LANES);
        f128x4_set(a[0], a[1], a[2], a[3])
    }

    /// Split `2 * LANES` consecutive `F128` (two vectors) into its
    /// even-indexed and odd-indexed halves, each in lane order.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn even_odd(v0: __m512i, v1: __m512i) -> (__m512i, __m512i) {
        let even_idx = _mm512_set_epi64(13, 12, 9, 8, 5, 4, 1, 0);
        let odd_idx = _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2);
        (
            _mm512_permutex2var_epi64(v0, even_idx, v1),
            _mm512_permutex2var_epi64(v0, odd_idx, v1),
        )
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn wide_zero() -> Wide {
        Wide::zero()
    }

    #[inline]
    #[target_feature(enable = "avx512f,vpclmulqdq")]
    pub(crate) unsafe fn wide_acc(acc: &mut Wide, x: __m512i, y: __m512i) {
        acc.mul_acc(x, y)
    }

    #[inline]
    #[target_feature(enable = "avx512f,vpclmulqdq,sse4.1")]
    pub(crate) unsafe fn wide_fold(acc: Wide) -> F256Unreduced {
        acc.fold()
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
