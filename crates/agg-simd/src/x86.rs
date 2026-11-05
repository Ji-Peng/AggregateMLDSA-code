//! Eight-lane primitives. Canonical storage is split only in registers.
//!
//! Every function here is `unsafe` because the caller must guarantee two
//! things: the CPU has AVX-512 F/DQ/BW/VL/IFMA (checked once by
//! [`crate::detect::backend`], which gates every call), and every pointer
//! argument addresses the number of `u64` elements the caller's slice has.
//! Register-only functions need only the feature guarantee.
#![allow(unused_unsafe)]
use std::arch::x86_64::*;
pub(crate) type V = __m512i;
#[derive(Clone, Copy)]
pub(crate) struct L52 {
    pub l: V,
    pub h: V,
}
#[derive(Clone, Copy)]
pub(crate) struct L51 {
    pub l: V,
    pub h: V,
    pub s: V,
}
#[derive(Clone, Copy)]
pub(crate) struct A3 {
    pub a: V,
    pub b: V,
    pub c: V,
}
#[derive(Clone, Copy)]
pub(crate) struct A5 {
    pub a: V,
    pub b: V,
    pub m: V,
    pub n: V,
    pub q: V,
}
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn bc(x: u64) -> V {
    _mm512_set1_epi64(x as i64)
}
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn z() -> V {
    _mm512_setzero_si512()
}
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn add(a: V, b: V) -> V {
    _mm512_add_epi64(a, b)
}
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn sub(a: V, b: V) -> V {
    _mm512_sub_epi64(a, b)
}
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn and(a: V, b: V) -> V {
    _mm512_and_si512(a, b)
}
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn lo(c: V, a: V, b: V) -> V {
    _mm512_madd52lo_epu64(c, a, b)
}
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn hi(c: V, a: V, b: V) -> V {
    _mm512_madd52hi_epu64(c, a, b)
}
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn load(p: *const u64) -> V {
    // SAFETY: the enclosing function enables AVX-512F, which the caller checked at run time,
    // and the caller guarantees eight readable words at `p` (unaligned access is allowed).
    unsafe { _mm512_loadu_si512(p.cast()) }
}
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn store(p: *mut u64, x: V) {
    // SAFETY: the enclosing function enables AVX-512F, which the caller checked at run time,
    // and the caller guarantees eight writable words at `p` (unaligned access is allowed).
    unsafe { _mm512_storeu_si512(p.cast(), x) }
}
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn canon(x: V) -> V {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
    unsafe { _mm512_min_epu64(x, sub(x, bc(crate::P))) }
}
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn add_mod(a: V, b: V) -> V {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
    unsafe {
        let s = add(a, b);
        canon(_mm512_mask_add_epi64(
            s,
            _mm512_cmplt_epu64_mask(s, a),
            s,
            bc(59),
        ))
    }
}
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn sub_mod(a: V, b: V) -> V {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
    unsafe {
        let s = sub(a, b);
        _mm512_mask_sub_epi64(s, _mm512_cmplt_epu64_mask(a, b), s, bc(59))
    }
}
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn split52(a: V) -> L52 {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
    unsafe {
        L52 {
            l: and(a, bc((1 << 52) - 1)),
            h: _mm512_srli_epi64::<52>(a),
        }
    }
}
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn split51(a: V) -> L51 {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
    unsafe {
        let l = and(a, bc((1 << 51) - 1));
        let h = _mm512_srli_epi64::<51>(a);
        L51 { l, h, s: add(l, h) }
    }
}
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn double52(a: L52) -> L52 {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
    unsafe {
        let t = add(a.l, a.l);
        L52 {
            l: and(t, bc((1 << 52) - 1)),
            h: add(add(a.h, a.h), _mm512_srli_epi64::<52>(t)),
        }
    }
}
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn add51(a: L51, b: L51) -> L51 {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
    unsafe {
        let t = add(a.l, b.l);
        let l = and(t, bc((1 << 51) - 1));
        let h = add(add(a.h, b.h), _mm512_srli_epi64::<51>(t));
        L51 { l, h, s: add(l, h) }
    }
}
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn a3() -> A3 {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
    unsafe {
        A3 {
            a: z(),
            b: z(),
            c: z(),
        }
    }
}
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn a5() -> A5 {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
    unsafe {
        A5 {
            a: z(),
            b: z(),
            m: z(),
            n: z(),
            q: z(),
        }
    }
}
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn add5(a: A5, b: A5) -> A5 {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
    unsafe {
        A5 {
            a: add(a.a, b.a),
            b: add(a.b, b.b),
            m: add(a.m, b.m),
            n: add(a.n, b.n),
            q: add(a.q, b.q),
        }
    }
}
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn sub5(a: A5, b: A5) -> A5 {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
    unsafe {
        A5 {
            a: sub(a.a, b.a),
            b: sub(a.b, b.b),
            m: sub(a.m, b.m),
            n: sub(a.n, b.n),
            q: sub(a.q, b.q),
        }
    }
}
/// Per term a gains <2^52, b <3*2^52, c <2^24+2^13 (doubled operand: <2^25+2^14).
/// Element products use <=2 terms per reduction; each limb stays far below 2^64.
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn mac7(c: &mut A3, a: L52, b: L52) {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
    unsafe {
        c.a = lo(c.a, a.l, b.l);
        c.b = hi(c.b, a.l, b.l);
        c.b = lo(c.b, a.l, b.h);
        c.b = lo(c.b, a.h, b.l);
        c.c = hi(c.c, a.l, b.h);
        c.c = hi(c.c, a.h, b.l);
        c.c = lo(c.c, a.h, b.h);
    }
}
/// Each IFMA digit gains <2^52 per term (q <2^28 for extension pre-sums).
/// Flush at 1024 base or 512 extension terms; signed intermediates remain in (-2^63,2^63).
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn mac5(c: &mut A5, a: L51, b: L51) {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
    unsafe {
        c.a = lo(c.a, a.l, b.l);
        c.b = hi(c.b, a.l, b.l);
        c.m = lo(c.m, a.s, b.s);
        c.n = hi(c.n, a.s, b.s);
        c.q = lo(c.q, a.h, b.h);
    }
}
/// c.c + 2^12 <2^46. Both folded carries are bounded before IFMA truncates at 52 bits.
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn red3(c: A3) -> V {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
    unsafe {
        let mask = bc((1 << 52) - 1);
        let m12 = bc((1 << 12) - 1);
        let c59 = bc(59);
        let d2 = add(c.c, _mm512_srli_epi64::<52>(c.b));
        let d1 = add(and(c.b, mask), _mm512_srli_epi64::<52>(c.a));
        let d0 = and(c.a, mask);
        let f = lo(z(), d2, c59);
        let e0 = add(d0, _mm512_slli_epi64::<40>(and(f, m12)));
        let e1 = add(d1, _mm512_srli_epi64::<12>(f));
        let g = lo(e0, _mm512_srli_epi64::<12>(e1), c59);
        let w = add(g, _mm512_slli_epi64::<52>(and(e1, m12)));
        canon(_mm512_mask_add_epi64(
            w,
            _mm512_cmplt_epu64_mask(w, g),
            w,
            c59,
        ))
    }
}
/// Signed digits and all arithmetic-shift intermediates stay in (-2^63,2^63).
/// The represented total is nonnegative. Flush bounds: 1024 base / 512 K terms.
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn red5(c: A5) -> V {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
    unsafe {
        let mask = bc((1 << 52) - 1);
        let c59 = bc(59);
        let m12 = bc((1 << 12) - 1);
        let mut d0 = sub(sub(c.m, c.a), c.q);
        let mut d1 = sub(c.n, c.b);
        d1 = add(d1, _mm512_srai_epi64::<52>(d0));
        d0 = and(d0, mask);
        let mut e0 = add(c.a, _mm512_slli_epi64::<51>(and(d0, bc(1))));
        let mut e1 = add(c.b, _mm512_srli_epi64::<1>(d0));
        let mut e2 = add(c.q, add(d1, d1));
        e1 = add(e1, _mm512_srai_epi64::<52>(e0));
        e0 = and(e0, mask);
        e2 = add(e2, _mm512_srai_epi64::<50>(e1));
        e1 = and(e1, bc((1 << 50) - 1));
        let f = lo(z(), e2, c59);
        e0 = add(e0, _mm512_slli_epi64::<38>(and(f, bc((1 << 14) - 1))));
        e1 = add(e1, _mm512_srli_epi64::<14>(f));
        let g = lo(e0, _mm512_srli_epi64::<12>(e1), c59);
        let w = add(g, _mm512_slli_epi64::<52>(and(e1, m12)));
        canon(_mm512_mask_add_epi64(
            w,
            _mm512_cmplt_epu64_mask(w, g),
            w,
            c59,
        ))
    }
}
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn mul(a: V, b: V) -> V {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
    unsafe {
        let mut c = a3();
        mac7(&mut c, split52(a), split52(b));
        red3(c)
    }
}
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn kmul(a: V, b: V, c: V, d: V) -> (V, V) {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
    unsafe {
        let a = split52(a);
        let b = split52(b);
        let c = split52(c);
        let d = split52(d);
        let mut re = a3();
        let mut im = a3();
        mac7(&mut re, a, c);
        mac7(&mut re, b, double52(d));
        mac7(&mut im, a, d);
        mac7(&mut im, b, c);
        (red3(re), red3(im))
    }
}
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn hsum(a: V) -> u64 {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
    unsafe {
        let mut lanes = [0u64; 8];
        store(lanes.as_mut_ptr(), a);
        lanes.into_iter().fold(0u64, |s, x| {
            let (r, c) = s.overflowing_add(x);
            let r = r.wrapping_add(u64::from(c) * 59);
            r.min(r.wrapping_sub(crate::P))
        })
    }
}
/// Rows `2k` and `2k + 1` of sixteen consecutive rows (`x0` rows `0..8`, `x1` rows `8..16`)
/// as the even and odd eight-lane vectors.
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn deinterleave(x0: V, x1: V) -> (V, V) {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
    unsafe {
        let even = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
        let odd = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
        (_mm512_permutex2var_epi64(x0, even, x1), _mm512_permutex2var_epi64(x0, odd, x1))
    }
}
/// `acc += w k` for a split weight and an integer lane `k` in `[0, 2^52)`: limb `a` gains
/// `< 2^52`, limb `b` `< 2^53`, limb `c` `< 2^12` per term, so after 512 terms every limb is
/// `< 2^62` (unsigned `u64` limbs; [`red3`] shifts logically, so the hard limit is `2^64`).
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn mac_int(acc: &mut A3, w: L52, k: V) {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
    unsafe {
        acc.a = lo(acc.a, w.l, k);
        acc.b = hi(acc.b, w.l, k);
        acc.b = lo(acc.b, w.h, k);
        acc.c = hi(acc.c, w.h, k);
    }
}
/// `acc += x` for a canonical value split into limbs (limb `a` gains `< 2^52`, `b` `< 2^12`).
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn lift52(acc: &mut A3, x: L52) {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
    unsafe {
        acc.a = add(acc.a, x.l);
        acc.b = add(acc.b, x.h);
    }
}
/// Both coordinates of a `K` lane vector split into limbs.
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn splitk(x: (V, V)) -> (L52, L52) {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
    unsafe { (split52(x.0), split52(x.1)) }
}
/// `acc += x y` in `K` without reduction (`re += ac + 2bd`, `im += ad + bc`): per term limb
/// `a` gains `< 2^53`, limb `b` `< 6 * 2^52`, limb `c` `< 2^26`; after 512 terms limb `b` is
/// `< 3 * 2^62 < 2^64` (unsigned `u64` limbs; [`red3`] shifts logically; the exact bound is
/// 682 terms, `682 * 6 * 2^52 < 2^64`).
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn kmac7(acc: &mut (A3, A3), x: (L52, L52), y: (L52, L52)) {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
    unsafe {
        mac7(&mut acc.0, x.0, y.0);
        mac7(&mut acc.0, x.1, double52(y.1));
        mac7(&mut acc.1, x.0, y.1);
        mac7(&mut acc.1, x.1, y.0);
    }
}
/// `acc += x^2` in `K` without reduction (`re += a^2 + 2b^2`, `im += 2ab`): three schoolbook
/// products instead of the four of [`kmac7`], since the two imaginary products coincide.
/// Limb bounds and flush limits are those of [`kmac7`]; the doubled operand keeps its high
/// limb below `2^13`.
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn sqk(acc: &mut (A3, A3), x: (L52, L52)) {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
    unsafe {
        mac7(&mut acc.0, x.0, x.0);
        mac7(&mut acc.0, x.1, double52(x.1));
        mac7(&mut acc.1, double52(x.0), x.1);
    }
}
/// Both coordinates reduced to canonical lanes.
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn redk(acc: (A3, A3)) -> (V, V) {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
    unsafe { (red3(acc.0), red3(acc.1)) }
}
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn a3k() -> (A3, A3) {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
    unsafe { (a3(), a3()) }
}
/// The eight lanes of a canonical `K` vector summed into one element.
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn hsum_k(x: (V, V)) -> crate::K {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
    unsafe {
        use jolt_field::Ring;
        crate::K::new(crate::F::from_u64(hsum(x.0)), crate::F::from_u64(hsum(x.1)))
    }
}
/// A `K` element broadcast to eight canonical lanes.
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn bck(x: crate::K) -> (V, V) {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
    unsafe { (bc(x.c0().to_canonical_u64()), bc(x.c1().to_canonical_u64())) }
}
/// `a + b` lane-wise in `K` (canonical in, canonical out).
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn addk(a: (V, V), b: (V, V)) -> (V, V) {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
    unsafe { (add_mod(a.0, b.0), add_mod(a.1, b.1)) }
}
/// `a b` lane-wise in `K` (canonical in, canonical out).
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn mulk(a: (V, V), b: (V, V)) -> (V, V) {
    // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
    unsafe { kmul(a.0, a.1, b.0, b.1) }
}
/// Eight consecutive `K` lanes loaded from coordinate planes at `i`.
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn loadk(planes: (&[u64], &[u64]), i: usize) -> (V, V) {
    // SAFETY: the enclosing function enables AVX-512F, which the caller checked at run time,
    // and the caller guarantees that both planes hold at least `i + 8` words.
    unsafe { (load(planes.0.as_ptr().add(i)), load(planes.1.as_ptr().add(i))) }
}
/// Eight `i8` lanes at stride four (`pattern + 4 l`, `l = 0..8`) of thirty-two consecutive
/// cells as sign-extended `i64` lanes. Reads exactly the 32 bytes at `cells`.
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn strided4_i8(cells: *const i8, pattern: i32) -> V {
    // SAFETY: the enclosing function enables AVX-512F, which the caller checked at run time,
    // and the caller guarantees 32 readable bytes at `cells`.
    unsafe {
        // Each 32-bit word holds the four patterns of one cell quartet; the pattern's byte is
        // moved to the top of its 64-bit lane and sign-extended by an arithmetic shift.
        let words = _mm512_cvtepu32_epi64(_mm256_loadu_si256(cells.cast()));
        let up = _mm512_sllv_epi64(words, _mm512_set1_epi64(56 - 8 * i64::from(pattern)));
        _mm512_srai_epi64::<56>(up)
    }
}
/// Eight `i32` lanes at stride four (`pattern + 4 l`, `l = 0..8`) of thirty-two consecutive
/// cells as sign-extended `i64` lanes.
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub(crate) unsafe fn strided4_i32(cells: *const i32, pattern: i32) -> V {
    // SAFETY: the enclosing function enables AVX-512F, which the caller checked at run time,
    // and the caller guarantees 32 readable `i32` cells at `cells`.
    unsafe {
        let x0 = _mm512_loadu_si512(cells.cast());
        let x1 = _mm512_loadu_si512(cells.add(16).cast());
        let idx = _mm512_add_epi32(
            _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 28, 24, 20, 16, 12, 8, 4, 0),
            _mm512_set1_epi32(pattern),
        );
        _mm512_cvtepi32_epi64(_mm512_castsi512_si256(_mm512_permutex2var_epi32(x0, idx, x1)))
    }
}
