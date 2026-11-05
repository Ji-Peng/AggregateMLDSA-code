//! AVX-512 NTT for q and P2 (F/BW/VL instructions only); P3 transforms run the
//! scalar code, and the dot product covers all three primes.
//!
//! The transforms reimplement the register schedule published with
//! the Hybrid-SHA-3 C code (`ntt_so_avx512` / `intt_so_avx512`):
//! the whole polynomial stays in sixteen ZMM registers, layers 0-3 are
//! butterflies between registers, layers 4-7 butterflies inside registers
//! after lane exchanges, and the forward output keeps that schedule's
//! coefficient order (`order_from_fips` maps FIPS order to it). No source
//! file or literal table is taken from that project: every twiddle is
//! generated from the prime parameters in `avx512_tables.rs`.

#[cfg(target_arch = "x86_64")]
use super::{Form, Scale};
#[cfg(target_arch = "x86_64")]
use crate::{
    params::{consts, Prime},
    poly::Poly32,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
#[path = "avx512_tables.rs"]
mod tables;

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
fn load(a: &Poly32, reg: usize) -> __m512i {
    let chunk: &[i32; 16] = a.0[reg * 16..reg * 16 + 16].try_into().unwrap();
    // SAFETY: The fixed-size chunk contains all sixteen initialized lanes.
    unsafe { _mm512_loadu_si512(chunk.as_ptr().cast()) }
}
#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
fn store(a: &mut Poly32, reg: usize, v: __m512i) {
    let chunk: &mut [i32; 16] = (&mut a.0[reg * 16..reg * 16 + 16]).try_into().unwrap();
    // SAFETY: The fixed-size chunk has room for all sixteen lanes.
    unsafe { _mm512_storeu_si512(chunk.as_mut_ptr().cast(), v) }
}
#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
fn lanes(a: &tables::Lanes) -> __m512i {
    // SAFETY: Lanes is 64-byte aligned and holds sixteen initialized i32 values.
    unsafe { _mm512_load_si512(a.0.as_ptr().cast()) }
}
/// Returns the twiddle tables behind an optimisation barrier. When LLVM sees
/// the constant values it rewrites the signed products into unsigned products
/// plus two extra shifts per product; opaque loads keep one multiply each.
#[inline]
#[cfg(target_arch = "x86_64")]
fn opaque(t: &'static tables::Tables) -> &'static tables::Tables {
    core::hint::black_box(t)
}
/// Signed Montgomery product per lane: (x*z - t*p) / 2^32 with
/// t = x*z*p^-1 mod 2^32, so |result| <= |x*z| / 2^32 + p/2.
/// `zo`/`zio` hold the odd-lane twiddles in the even dword of each qword;
/// layers whose twiddles agree within a qword pass `z`/`zi` again.
#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
fn mont(x: __m512i, z: __m512i, zi: __m512i, zo: __m512i, zio: __m512i, p: __m512i) -> __m512i {
    // Odd lanes moved to the even dwords, where vpmuldq reads its operands. A
    // qword shift instead of a dword shuffle: vpmuldq ignores the high dwords, and
    // shifts and blends issue on more ports than shuffles on Zen 5.
    let h = _mm512_srli_epi64::<32>(x);
    let e = _mm512_mul_epi32(x, z);
    let o = _mm512_mul_epi32(h, zo);
    let te = _mm512_mul_epi32(_mm512_mul_epi32(x, zi), p);
    let to = _mm512_mul_epi32(_mm512_mul_epi32(h, zio), p);
    // The low dwords cancel exactly, so a 32-bit subtraction gives the exact
    // high dwords without a borrow.
    let e = _mm512_sub_epi32(e, te);
    let o = _mm512_sub_epi32(o, to);
    // Even results come from the high dwords of `e`; odd ones are in place in `o`.
    _mm512_mask_blend_epi32(0x5555, o, _mm512_srli_epi64::<32>(e))
}
// These exchanges are involutions; they preserve the coefficient storage layout.
#[cfg(target_arch = "x86_64")]
macro_rules! exchange {
    ($a:ident,$b:ident,8) => {{
        let t = $a;
        $a = _mm512_shuffle_i32x4::<0x44>($a, $b);
        $b = _mm512_shuffle_i32x4::<0xee>(t, $b);
    }};
    ($a:ident,$b:ident,4) => {{
        let t = $a;
        $a = _mm512_mask_blend_epi64(0xcc, $a, _mm512_permutex_epi64::<0x4e>($b));
        $b = _mm512_mask_blend_epi64(0x33, $b, _mm512_permutex_epi64::<0x4e>(t));
    }};
    ($a:ident,$b:ident,2) => {{
        let t = $a;
        $a = _mm512_mask_shuffle_epi32::<0x44>($a, 0xcccc, $b);
        $b = _mm512_mask_shuffle_epi32::<0xee>($b, 0x3333, t);
    }};
    ($a:ident,$b:ident,1) => {{
        let t = $a;
        $a = _mm512_mask_shuffle_epi32::<0xa0>($a, 0xaaaa, $b);
        $b = _mm512_mask_shuffle_epi32::<0xf5>($b, 0x5555, t);
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
pub(crate) unsafe fn ntt(prime: Prime, a: &mut Poly32) {
    match prime {
        Prime::Q => ntt_impl::<false>(a),
        Prime::P2 => ntt_impl::<true>(a),
        Prime::P3 => {
            super::scalar::ntt(prime, a);
            // SAFETY: target features avx512f, bw, vl are enabled.
            unsafe { order_from_fips(a) };
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
#[inline(never)]
fn ntt_impl<const SECOND: bool>(a: &mut Poly32) {
    let prime = if SECOND { Prime::P2 } else { Prime::Q };
    let t = opaque(if SECOND { &tables::P2 } else { &tables::Q });
    let p = _mm512_set1_epi32(consts(prime).p);
    // Lazy CT sums: starting at 2^22, every layer adds at most
    // p/2 + |z*x|/2^32; generated bounds give |x|<4.52q or |x|<4.25P2 < 2^31.
    macro_rules! bf {
        ($a:ident, $b:ident, $i:expr) => {{
            let (z, zi) = (lanes(&t.z[$i]), lanes(&t.zi[$i]));
            let v = mont($b, z, zi, z, zi, p);
            $b = _mm512_sub_epi32($a, v);
            $a = _mm512_add_epi32($a, v);
        }};
        ($a:ident, $b:ident, $i:expr, odd) => {{
            let (z, zi) = (lanes(&t.z[$i]), lanes(&t.zi[$i]));
            let (zo, zio) = (lanes(&t.zo[$i - 56]), lanes(&t.zio[$i - 56]));
            let v = mont($b, z, zi, zo, zio, p);
            $b = _mm512_sub_epi32($a, v);
            $a = _mm512_add_epi32($a, v);
        }};
    }
    let mut b0 = load(a, 0);
    let mut b1 = load(a, 1);
    let mut b2 = load(a, 2);
    let mut b3 = load(a, 3);
    let mut b4 = load(a, 4);
    let mut b5 = load(a, 5);
    let mut b6 = load(a, 6);
    let mut b7 = load(a, 7);
    let mut b8 = load(a, 8);
    let mut b9 = load(a, 9);
    let mut b10 = load(a, 10);
    let mut b11 = load(a, 11);
    let mut b12 = load(a, 12);
    let mut b13 = load(a, 13);
    let mut b14 = load(a, 14);
    let mut b15 = load(a, 15);
    bf!(b0, b8, 0);
    bf!(b1, b9, 1);
    bf!(b2, b10, 2);
    bf!(b3, b11, 3);
    bf!(b4, b12, 4);
    bf!(b5, b13, 5);
    bf!(b6, b14, 6);
    bf!(b7, b15, 7);
    bf!(b0, b4, 8);
    bf!(b1, b5, 9);
    bf!(b2, b6, 10);
    bf!(b3, b7, 11);
    bf!(b8, b12, 12);
    bf!(b9, b13, 13);
    bf!(b10, b14, 14);
    bf!(b11, b15, 15);
    bf!(b0, b2, 16);
    bf!(b1, b3, 17);
    bf!(b4, b6, 18);
    bf!(b5, b7, 19);
    bf!(b8, b10, 20);
    bf!(b9, b11, 21);
    bf!(b12, b14, 22);
    bf!(b13, b15, 23);
    bf!(b0, b1, 24);
    bf!(b2, b3, 25);
    bf!(b4, b5, 26);
    bf!(b6, b7, 27);
    bf!(b8, b9, 28);
    bf!(b10, b11, 29);
    bf!(b12, b13, 30);
    bf!(b14, b15, 31);
    exchange!(b0, b1, 8);
    bf!(b0, b1, 32);
    exchange!(b2, b3, 8);
    bf!(b2, b3, 33);
    exchange!(b4, b5, 8);
    bf!(b4, b5, 34);
    exchange!(b6, b7, 8);
    bf!(b6, b7, 35);
    exchange!(b8, b9, 8);
    bf!(b8, b9, 36);
    exchange!(b10, b11, 8);
    bf!(b10, b11, 37);
    exchange!(b12, b13, 8);
    bf!(b12, b13, 38);
    exchange!(b14, b15, 8);
    bf!(b14, b15, 39);
    exchange!(b0, b1, 4);
    bf!(b0, b1, 40);
    exchange!(b2, b3, 4);
    bf!(b2, b3, 41);
    exchange!(b4, b5, 4);
    bf!(b4, b5, 42);
    exchange!(b6, b7, 4);
    bf!(b6, b7, 43);
    exchange!(b8, b9, 4);
    bf!(b8, b9, 44);
    exchange!(b10, b11, 4);
    bf!(b10, b11, 45);
    exchange!(b12, b13, 4);
    bf!(b12, b13, 46);
    exchange!(b14, b15, 4);
    bf!(b14, b15, 47);
    exchange!(b0, b1, 2);
    bf!(b0, b1, 48);
    exchange!(b2, b3, 2);
    bf!(b2, b3, 49);
    exchange!(b4, b5, 2);
    bf!(b4, b5, 50);
    exchange!(b6, b7, 2);
    bf!(b6, b7, 51);
    exchange!(b8, b9, 2);
    bf!(b8, b9, 52);
    exchange!(b10, b11, 2);
    bf!(b10, b11, 53);
    exchange!(b12, b13, 2);
    bf!(b12, b13, 54);
    exchange!(b14, b15, 2);
    bf!(b14, b15, 55);
    exchange!(b0, b1, 1);
    bf!(b0, b1, 56, odd);
    exchange!(b2, b3, 1);
    bf!(b2, b3, 57, odd);
    exchange!(b4, b5, 1);
    bf!(b4, b5, 58, odd);
    exchange!(b6, b7, 1);
    bf!(b6, b7, 59, odd);
    exchange!(b8, b9, 1);
    bf!(b8, b9, 60, odd);
    exchange!(b10, b11, 1);
    bf!(b10, b11, 61, odd);
    exchange!(b12, b13, 1);
    bf!(b12, b13, 62, odd);
    exchange!(b14, b15, 1);
    bf!(b14, b15, 63, odd);
    store(a, 0, b0);
    store(a, 1, b1);
    store(a, 2, b2);
    store(a, 3, b3);
    store(a, 4, b4);
    store(a, 5, b5);
    store(a, 6, b6);
    store(a, 7, b7);
    store(a, 8, b8);
    store(a, 9, b9);
    store(a, 10, b10);
    store(a, 11, b11);
    store(a, 12, b12);
    store(a, 13, b13);
    store(a, 14, b14);
    store(a, 15, b15);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
pub(crate) unsafe fn intt(prime: Prime, a: &mut Poly32, scale: Scale, form: Form) {
    if prime == Prime::P3 {
        // SAFETY: target features avx512f, bw, vl are enabled.
        unsafe { order_to_fips(a) };
        super::scalar::intt(prime, a, scale, form);
        return;
    }
    match (prime, scale, form) {
        (Prime::Q, Scale::Plain, Form::Canonical) => intt_impl::<false, false, false>(a),
        (Prime::Q, Scale::Plain, Form::Centered) => intt_impl::<false, false, true>(a),
        (Prime::Q, Scale::Mont, Form::Canonical) => intt_impl::<false, true, false>(a),
        (Prime::Q, Scale::Mont, Form::Centered) => intt_impl::<false, true, true>(a),
        (Prime::P2, Scale::Plain, Form::Canonical) => intt_impl::<true, false, false>(a),
        (Prime::P2, Scale::Plain, Form::Centered) => intt_impl::<true, false, true>(a),
        (Prime::P2, Scale::Mont, Form::Canonical) => intt_impl::<true, true, false>(a),
        (Prime::P2, Scale::Mont, Form::Centered) => intt_impl::<true, true, true>(a),
        (Prime::P3, _, _) => unreachable!(),
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
#[inline(never)]
fn intt_impl<const SECOND: bool, const MONT: bool, const CENTERED: bool>(a: &mut Poly32) {
    let prime = if SECOND { Prime::P2 } else { Prime::Q };
    let c = consts(prime);
    let t = opaque(if SECOND { &tables::P2 } else { &tables::Q });
    let s = MONT as usize;
    let p = _mm512_set1_epi32(c.p);
    // GS sum chains double bounds; P2's selected long chains are reduced
    // after four layers. The generated bound table proves every lane < 2^31.
    macro_rules! bf {
        ($a:ident, $b:ident, $i:expr) => {{
            let v = _mm512_sub_epi32($a, $b);
            $a = _mm512_add_epi32($a, $b);
            let (z, zi) = (lanes(&t.iz[$i]), lanes(&t.izi[$i]));
            $b = mont(v, z, zi, z, zi, p);
        }};
        ($a:ident, $b:ident, $i:expr, odd) => {{
            let v = _mm512_sub_epi32($a, $b);
            $a = _mm512_add_epi32($a, $b);
            let (z, zi) = (lanes(&t.iz[$i]), lanes(&t.izi[$i]));
            let (zo, zio) = (lanes(&t.izo[$i - 56]), lanes(&t.izio[$i - 56]));
            $b = mont(v, z, zi, zo, zio, p);
        }};
    }
    // The last GS layer fuses the normalization: sums are scaled by f and
    // differences by root*f/R, so both outputs are Montgomery products
    // of inputs < 2^31 with twiddles <= p/2 and lie in (-p, p).
    macro_rules! last {
        ($a:ident, $b:ident) => {{
            let v = _mm512_sub_epi32($a, $b);
            let (f, fi) = (lanes(&t.scale[s]), lanes(&t.scale_i[s]));
            $a = mont(_mm512_add_epi32($a, $b), f, fi, f, fi, p);
            let (z, zi) = (lanes(&t.last[s]), lanes(&t.last_i[s]));
            $b = mont(v, z, zi, z, zi, p);
        }};
    }
    let mut b0 = load(a, 0);
    let mut b1 = load(a, 1);
    let mut b2 = load(a, 2);
    let mut b3 = load(a, 3);
    let mut b4 = load(a, 4);
    let mut b5 = load(a, 5);
    let mut b6 = load(a, 6);
    let mut b7 = load(a, 7);
    let mut b8 = load(a, 8);
    let mut b9 = load(a, 9);
    let mut b10 = load(a, 10);
    let mut b11 = load(a, 11);
    let mut b12 = load(a, 12);
    let mut b13 = load(a, 13);
    let mut b14 = load(a, 14);
    let mut b15 = load(a, 15);
    bf!(b0, b1, 56, odd);
    exchange!(b0, b1, 1);
    bf!(b2, b3, 57, odd);
    exchange!(b2, b3, 1);
    bf!(b4, b5, 58, odd);
    exchange!(b4, b5, 1);
    bf!(b6, b7, 59, odd);
    exchange!(b6, b7, 1);
    bf!(b8, b9, 60, odd);
    exchange!(b8, b9, 1);
    bf!(b10, b11, 61, odd);
    exchange!(b10, b11, 1);
    bf!(b12, b13, 62, odd);
    exchange!(b12, b13, 1);
    bf!(b14, b15, 63, odd);
    exchange!(b14, b15, 1);
    bf!(b0, b1, 48);
    exchange!(b0, b1, 2);
    bf!(b2, b3, 49);
    exchange!(b2, b3, 2);
    bf!(b4, b5, 50);
    exchange!(b4, b5, 2);
    bf!(b6, b7, 51);
    exchange!(b6, b7, 2);
    bf!(b8, b9, 52);
    exchange!(b8, b9, 2);
    bf!(b10, b11, 53);
    exchange!(b10, b11, 2);
    bf!(b12, b13, 54);
    exchange!(b12, b13, 2);
    bf!(b14, b15, 55);
    exchange!(b14, b15, 2);
    bf!(b0, b1, 40);
    exchange!(b0, b1, 4);
    bf!(b2, b3, 41);
    exchange!(b2, b3, 4);
    bf!(b4, b5, 42);
    exchange!(b4, b5, 4);
    bf!(b6, b7, 43);
    exchange!(b6, b7, 4);
    bf!(b8, b9, 44);
    exchange!(b8, b9, 4);
    bf!(b10, b11, 45);
    exchange!(b10, b11, 4);
    bf!(b12, b13, 46);
    exchange!(b12, b13, 4);
    bf!(b14, b15, 47);
    exchange!(b14, b15, 4);
    bf!(b0, b1, 32);
    exchange!(b0, b1, 8);
    bf!(b2, b3, 33);
    exchange!(b2, b3, 8);
    bf!(b4, b5, 34);
    exchange!(b4, b5, 8);
    bf!(b6, b7, 35);
    exchange!(b6, b7, 8);
    bf!(b8, b9, 36);
    exchange!(b8, b9, 8);
    bf!(b10, b11, 37);
    exchange!(b10, b11, 8);
    bf!(b12, b13, 38);
    exchange!(b12, b13, 8);
    bf!(b14, b15, 39);
    exchange!(b14, b15, 8);
    if SECOND {
        // |x|<=8p: rounded division by 2^25 and then by five leaves |r|<0.61p.
        let bias = _mm512_set1_epi32(1 << 24);
        let fifth = _mm512_set1_epi16(6554);
        let red = |x| {
            let n = _mm512_srai_epi32::<25>(_mm512_add_epi32(x, bias));
            let q = _mm512_mulhrs_epi16(n, fifth);
            let q = _mm512_srai_epi32::<16>(_mm512_slli_epi32::<16>(q));
            _mm512_sub_epi32(x, _mm512_mullo_epi32(q, p))
        };
        b0 = _mm512_mask_blend_epi32(tables::reduction_mask(0), b0, red(b0));
        b1 = _mm512_mask_blend_epi32(tables::reduction_mask(1), b1, red(b1));
        b2 = _mm512_mask_blend_epi32(tables::reduction_mask(2), b2, red(b2));
        b3 = _mm512_mask_blend_epi32(tables::reduction_mask(3), b3, red(b3));
        b4 = _mm512_mask_blend_epi32(tables::reduction_mask(4), b4, red(b4));
        b5 = _mm512_mask_blend_epi32(tables::reduction_mask(5), b5, red(b5));
        b6 = _mm512_mask_blend_epi32(tables::reduction_mask(6), b6, red(b6));
        b7 = _mm512_mask_blend_epi32(tables::reduction_mask(7), b7, red(b7));
        b8 = _mm512_mask_blend_epi32(tables::reduction_mask(8), b8, red(b8));
        b9 = _mm512_mask_blend_epi32(tables::reduction_mask(9), b9, red(b9));
        b10 = _mm512_mask_blend_epi32(tables::reduction_mask(10), b10, red(b10));
        b11 = _mm512_mask_blend_epi32(tables::reduction_mask(11), b11, red(b11));
        b12 = _mm512_mask_blend_epi32(tables::reduction_mask(12), b12, red(b12));
        b13 = _mm512_mask_blend_epi32(tables::reduction_mask(13), b13, red(b13));
        b14 = _mm512_mask_blend_epi32(tables::reduction_mask(14), b14, red(b14));
        b15 = _mm512_mask_blend_epi32(tables::reduction_mask(15), b15, red(b15));
    }
    bf!(b0, b1, 24);
    bf!(b2, b3, 25);
    bf!(b4, b5, 26);
    bf!(b6, b7, 27);
    bf!(b8, b9, 28);
    bf!(b10, b11, 29);
    bf!(b12, b13, 30);
    bf!(b14, b15, 31);
    bf!(b0, b2, 16);
    bf!(b1, b3, 17);
    bf!(b4, b6, 18);
    bf!(b5, b7, 19);
    bf!(b8, b10, 20);
    bf!(b9, b11, 21);
    bf!(b12, b14, 22);
    bf!(b13, b15, 23);
    bf!(b0, b4, 8);
    bf!(b1, b5, 9);
    bf!(b2, b6, 10);
    bf!(b3, b7, 11);
    bf!(b8, b12, 12);
    bf!(b9, b13, 13);
    bf!(b10, b14, 14);
    bf!(b11, b15, 15);
    last!(b0, b8);
    last!(b1, b9);
    last!(b2, b10);
    last!(b3, b11);
    last!(b4, b12);
    last!(b5, b13);
    last!(b6, b14);
    last!(b7, b15);
    store(a, 0, finish::<CENTERED>(b0, p));
    store(a, 1, finish::<CENTERED>(b1, p));
    store(a, 2, finish::<CENTERED>(b2, p));
    store(a, 3, finish::<CENTERED>(b3, p));
    store(a, 4, finish::<CENTERED>(b4, p));
    store(a, 5, finish::<CENTERED>(b5, p));
    store(a, 6, finish::<CENTERED>(b6, p));
    store(a, 7, finish::<CENTERED>(b7, p));
    store(a, 8, finish::<CENTERED>(b8, p));
    store(a, 9, finish::<CENTERED>(b9, p));
    store(a, 10, finish::<CENTERED>(b10, p));
    store(a, 11, finish::<CENTERED>(b11, p));
    store(a, 12, finish::<CENTERED>(b12, p));
    store(a, 13, finish::<CENTERED>(b13, p));
    store(a, 14, finish::<CENTERED>(b14, p));
    store(a, 15, finish::<CENTERED>(b15, p));
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
fn finish<const CENTERED: bool>(x: __m512i, p: __m512i) -> __m512i {
    // Input lies in (-1.5p, 1.5p) for the centered form and in (-p, p) for the
    // canonical one; at most one correction reaches the requested range.
    if CENTERED {
        let h = _mm512_srli_epi32::<1>(p);
        let x = _mm512_mask_add_epi32(
            x,
            _mm512_cmplt_epi32_mask(x, _mm512_sub_epi32(_mm512_setzero_si512(), h)),
            x,
            p,
        );
        _mm512_mask_sub_epi32(x, _mm512_cmpgt_epi32_mask(x, h), x, p)
    } else {
        _mm512_mask_add_epi32(x, _mm512_cmplt_epi32_mask(x, _mm512_setzero_si512()), x, p)
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
pub(crate) unsafe fn dot_l<const L: usize>(
    prime: Prime,
    a: &[Poly32; L],
    b: &[Poly32; L],
    out: &mut Poly32,
) {
    let c = consts(prime);
    let p = _mm512_set1_epi32(c.p);
    let pinv = _mm512_set1_epi32(c.pinv);
    for reg in 0..16 {
        let mut e = _mm512_setzero_si512();
        let mut o = e;
        for k in 0..L {
            let x = load(&a[k], reg);
            let y = load(&b[k], reg);
            // L unreduced signed products of a centered (canonical for q) matrix and
            // a lazy NTT(z), L <= 7: below 2^58.6 for P2 and 2^61 for P3.
            e = _mm512_add_epi64(e, _mm512_mul_epi32(x, y));
            o = _mm512_add_epi64(
                o,
                _mm512_mul_epi32(_mm512_srli_epi64::<32>(x), _mm512_srli_epi64::<32>(y)),
            );
        }
        e = _mm512_sub_epi64(e, _mm512_mul_epi32(_mm512_mul_epi32(e, pinv), p));
        o = _mm512_sub_epi64(o, _mm512_mul_epi32(_mm512_mul_epi32(o, pinv), p));
        let v = _mm512_mask_blend_epi32(0xaaaa, _mm512_srli_epi64::<32>(e), o);
        // |v| < 1.08p for P2 needs one centering pass; |v| < 1.9p for P3 needs two.
        let v = match prime {
            Prime::Q => v,
            Prime::P2 => finish::<true>(v, p),
            Prime::P3 => finish::<true>(finish::<true>(v, p), p),
        };
        store(out, reg, v);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
#[allow(dead_code)]
pub(crate) unsafe fn dot4(prime: Prime, a: &[Poly32; 4], b: &[Poly32; 4], out: &mut Poly32) {
    // SAFETY: AVX-512 target features are enabled on this function.
    unsafe { dot_l::<4>(prime, a, b, out) }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
pub(crate) unsafe fn reduce(prime: Prime, a: &mut Poly32, form: Form) {
    let c = consts(prime);
    let p = _mm512_set1_epi32(c.p);
    // Rounded reciprocal quotient for any i32: residual is within (-p,p).
    // Widening the subtraction avoids signed overflow near i32::MIN/MAX.
    let reciprocal = ((1u64 << 32) + (c.p as u64) / 2) / (c.p as u64);
    let r = _mm512_set1_epi32(reciprocal as i32);
    for reg in 0..16 {
        let x = load(a, reg);
        let e = _mm512_srai_epi64::<32>(_mm512_mul_epi32(x, r));
        let o = _mm512_srai_epi64::<32>(_mm512_mul_epi32(_mm512_srli_epi64::<32>(x), r));
        let q = _mm512_mask_blend_epi32(0xaaaa, e, _mm512_slli_epi64::<32>(o));
        let v = _mm512_sub_epi32(x, _mm512_mullo_epi32(q, p));
        let v = _mm512_mask_sub_epi32(v, _mm512_cmpge_epi32_mask(v, p), v, p);
        store(
            a,
            reg,
            match form {
                Form::Canonical => finish::<false>(v, p),
                Form::Centered => finish::<true>(v, p),
            },
        );
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
pub(crate) unsafe fn order_from_fips(a: &mut Poly32) {
    let even = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    let odd = _mm512_add_epi32(even, _mm512_set1_epi32(1));
    for reg in (0..16).step_by(2) {
        let x = load(a, reg);
        let y = load(a, reg + 1);
        store(a, reg, _mm512_permutex2var_epi32(x, even, y));
        store(a, reg + 1, _mm512_permutex2var_epi32(x, odd, y));
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
pub(crate) unsafe fn order_to_fips(a: &mut Poly32) {
    let lo = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
    let hi = _mm512_add_epi32(lo, _mm512_set1_epi32(8));
    for reg in (0..16).step_by(2) {
        let x = load(a, reg);
        let y = load(a, reg + 1);
        store(a, reg, _mm512_permutex2var_epi32(x, lo, y));
        store(a, reg + 1, _mm512_permutex2var_epi32(x, hi, y));
    }
}

// These names remain resolvable on scalar-only targets; dispatch never selects
// the AVX-512 backend there.
#[cfg(not(target_arch = "x86_64"))]
pub(crate) use super::scalar::{dot4, dot_l, intt, ntt, order_from_fips, order_to_fips, reduce};
