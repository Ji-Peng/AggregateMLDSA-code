//! AVX-512 NTT using the register schedule of Hybrid-SHA-3 by Xurong Xiao et al.
//! Translated with the authors' permission. Twiddles are generated from the roots
//! and moduli; no external literal tables are embedded.

use super::{Form, Scale};
use crate::{
    params::{consts, Prime},
    poly::Poly32,
};
use core::arch::x86_64::*;
#[path = "avx512_tables.rs"]
mod tables;

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
fn load(a: &Poly32, reg: usize) -> __m512i {
    let chunk: &[i32; 16] = a.0[reg * 16..reg * 16 + 16].try_into().unwrap();
    // SAFETY: The fixed-size chunk contains all sixteen initialized lanes.
    unsafe { _mm512_loadu_si512(chunk.as_ptr().cast()) }
}
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
fn store(a: &mut Poly32, reg: usize, v: __m512i) {
    let chunk: &mut [i32; 16] = (&mut a.0[reg * 16..reg * 16 + 16]).try_into().unwrap();
    // SAFETY: The fixed-size chunk has room for all sixteen lanes.
    unsafe { _mm512_storeu_si512(chunk.as_mut_ptr().cast(), v) }
}
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
fn lanes(a: &tables::Lanes) -> __m512i {
    // SAFETY: Lanes contains sixteen initialized i32 values.
    unsafe { _mm512_load_si512(a.0.as_ptr().cast()) }
}
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
fn mont(x: __m512i, z: __m512i, zi: __m512i, p: __m512i) -> __m512i {
    let odd = _mm512_srli_epi64::<32>(x);
    let e = _mm512_mul_epi32(x, z);
    let o = _mm512_mul_epi32(odd, _mm512_srli_epi64::<32>(z));
    let te = _mm512_mul_epi32(x, zi);
    let to = _mm512_mul_epi32(odd, _mm512_srli_epi64::<32>(zi));
    let e = _mm512_sub_epi64(e, _mm512_mul_epi32(te, p));
    let o = _mm512_sub_epi64(o, _mm512_mul_epi32(to, p));
    _mm512_mask_blend_epi32(0xaaaa, _mm512_srli_epi64::<32>(e), o)
}
// These exchanges are involutions; they preserve the coefficient storage layout.
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

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
pub(crate) unsafe fn ntt(prime: Prime, a: &mut Poly32) {
    let t = match prime {
        Prime::Q => &tables::Q,
        Prime::P2 => &tables::P2,
    };
    let p = _mm512_set1_epi32(consts(prime).p);
    // Lazy CT sums: starting at 2^22, every layer adds at most
    // p/2 + |z*x|/2^32; generated twiddles keep all outputs below 4.4p.
    macro_rules! bf { ($a:ident,$b:ident,$i:expr) => {{
        let v=mont($b,lanes(&t.z[$i]),lanes(&t.zi[$i]),p);
        $b=_mm512_sub_epi32($a,v); $a=_mm512_add_epi32($a,v);
    }}; }
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
    bf!(b0, b1, 56);
    exchange!(b2, b3, 1);
    bf!(b2, b3, 57);
    exchange!(b4, b5, 1);
    bf!(b4, b5, 58);
    exchange!(b6, b7, 1);
    bf!(b6, b7, 59);
    exchange!(b8, b9, 1);
    bf!(b8, b9, 60);
    exchange!(b10, b11, 1);
    bf!(b10, b11, 61);
    exchange!(b12, b13, 1);
    bf!(b12, b13, 62);
    exchange!(b14, b15, 1);
    bf!(b14, b15, 63);
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

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
pub(crate) unsafe fn intt(prime: Prime, a: &mut Poly32, scale: Scale, form: Form) {
    let c = consts(prime);
    let t = match prime {
        Prime::Q => &tables::Q,
        Prime::P2 => &tables::P2,
    };
    let p = _mm512_set1_epi32(c.p);
    // GS sum chains double bounds; P2's selected long chains are reduced
    // after four layers. The generated bound table proves every lane < 2^31.
    macro_rules! bf { ($a:ident,$b:ident,$i:expr) => {{
        let v=_mm512_sub_epi32($a,$b); $a=_mm512_add_epi32($a,$b);
        $b=mont(v,lanes(&t.iz[$i]),lanes(&t.izi[$i]),p);
    }}; }
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
    bf!(b0, b1, 56);
    exchange!(b0, b1, 1);
    bf!(b2, b3, 57);
    exchange!(b2, b3, 1);
    bf!(b4, b5, 58);
    exchange!(b4, b5, 1);
    bf!(b6, b7, 59);
    exchange!(b6, b7, 1);
    bf!(b8, b9, 60);
    exchange!(b8, b9, 1);
    bf!(b10, b11, 61);
    exchange!(b10, b11, 1);
    bf!(b12, b13, 62);
    exchange!(b12, b13, 1);
    bf!(b14, b15, 63);
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
    if prime == Prime::P2 {
        let z = _mm512_set1_epi32(c.mont);
        let zi = _mm512_set1_epi32(c.mont.wrapping_mul(c.pinv));
        b0 = _mm512_mask_blend_epi32(tables::reduction_mask(0), b0, mont(b0, z, zi, p));
        b1 = _mm512_mask_blend_epi32(tables::reduction_mask(1), b1, mont(b1, z, zi, p));
        b2 = _mm512_mask_blend_epi32(tables::reduction_mask(2), b2, mont(b2, z, zi, p));
        b3 = _mm512_mask_blend_epi32(tables::reduction_mask(3), b3, mont(b3, z, zi, p));
        b4 = _mm512_mask_blend_epi32(tables::reduction_mask(4), b4, mont(b4, z, zi, p));
        b5 = _mm512_mask_blend_epi32(tables::reduction_mask(5), b5, mont(b5, z, zi, p));
        b6 = _mm512_mask_blend_epi32(tables::reduction_mask(6), b6, mont(b6, z, zi, p));
        b7 = _mm512_mask_blend_epi32(tables::reduction_mask(7), b7, mont(b7, z, zi, p));
        b8 = _mm512_mask_blend_epi32(tables::reduction_mask(8), b8, mont(b8, z, zi, p));
        b9 = _mm512_mask_blend_epi32(tables::reduction_mask(9), b9, mont(b9, z, zi, p));
        b10 = _mm512_mask_blend_epi32(tables::reduction_mask(10), b10, mont(b10, z, zi, p));
        b11 = _mm512_mask_blend_epi32(tables::reduction_mask(11), b11, mont(b11, z, zi, p));
        b12 = _mm512_mask_blend_epi32(tables::reduction_mask(12), b12, mont(b12, z, zi, p));
        b13 = _mm512_mask_blend_epi32(tables::reduction_mask(13), b13, mont(b13, z, zi, p));
        b14 = _mm512_mask_blend_epi32(tables::reduction_mask(14), b14, mont(b14, z, zi, p));
        b15 = _mm512_mask_blend_epi32(tables::reduction_mask(15), b15, mont(b15, z, zi, p));
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
    bf!(b0, b8, 0);
    bf!(b1, b9, 1);
    bf!(b2, b10, 2);
    bf!(b3, b11, 3);
    bf!(b4, b12, 4);
    bf!(b5, b13, 5);
    bf!(b6, b14, 6);
    bf!(b7, b15, 7);
    let f = match scale {
        Scale::Plain => c.inv_plain,
        Scale::Mont => c.inv_mont,
    };
    let z = _mm512_set1_epi32(f);
    let zi = _mm512_set1_epi32(f.wrapping_mul(c.pinv));
    store(a, 0, finish(mont(b0, z, zi, p), p, form));
    store(a, 1, finish(mont(b1, z, zi, p), p, form));
    store(a, 2, finish(mont(b2, z, zi, p), p, form));
    store(a, 3, finish(mont(b3, z, zi, p), p, form));
    store(a, 4, finish(mont(b4, z, zi, p), p, form));
    store(a, 5, finish(mont(b5, z, zi, p), p, form));
    store(a, 6, finish(mont(b6, z, zi, p), p, form));
    store(a, 7, finish(mont(b7, z, zi, p), p, form));
    store(a, 8, finish(mont(b8, z, zi, p), p, form));
    store(a, 9, finish(mont(b9, z, zi, p), p, form));
    store(a, 10, finish(mont(b10, z, zi, p), p, form));
    store(a, 11, finish(mont(b11, z, zi, p), p, form));
    store(a, 12, finish(mont(b12, z, zi, p), p, form));
    store(a, 13, finish(mont(b13, z, zi, p), p, form));
    store(a, 14, finish(mont(b14, z, zi, p), p, form));
    store(a, 15, finish(mont(b15, z, zi, p), p, form));
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
fn finish(x: __m512i, p: __m512i, form: Form) -> __m512i {
    // Input lies in (-p,p); one correction makes it canonical.
    let x = _mm512_add_epi32(x, _mm512_and_si512(_mm512_srai_epi32::<31>(x), p));
    if form == Form::Centered {
        _mm512_mask_sub_epi32(
            x,
            _mm512_cmpgt_epi32_mask(x, _mm512_srli_epi32::<1>(p)),
            x,
            p,
        )
    } else {
        x
    }
}
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
pub(crate) unsafe fn dot4(prime: Prime, a: &[Poly32; 4], b: &[Poly32; 4], out: &mut Poly32) {
    let c = consts(prime);
    let p = _mm512_set1_epi32(c.p);
    let pinv = _mm512_set1_epi32(c.pinv);
    for reg in 0..16 {
        let mut e = _mm512_setzero_si512();
        let mut o = e;
        for k in 0..4 {
            let x = load(&a[k], reg);
            let y = load(&b[k], reg);
            // Four unreduced signed products: q < 18q^2, P2 < 8.8P2^2 < 2^58.
            e = _mm512_add_epi64(e, _mm512_mul_epi32(x, y));
            o = _mm512_add_epi64(
                o,
                _mm512_mul_epi32(_mm512_srli_epi64::<32>(x), _mm512_srli_epi64::<32>(y)),
            );
        }
        e = _mm512_sub_epi64(e, _mm512_mul_epi32(_mm512_mul_epi32(e, pinv), p));
        o = _mm512_sub_epi64(o, _mm512_mul_epi32(_mm512_mul_epi32(o, pinv), p));
        let v = _mm512_mask_blend_epi32(0xaaaa, _mm512_srli_epi64::<32>(e), o);
        store(
            out,
            reg,
            if prime == Prime::P2 {
                finish(v, p, Form::Centered)
            } else {
                v
            },
        );
    }
}
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
        store(a, reg, finish(v, p, form));
    }
}
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
