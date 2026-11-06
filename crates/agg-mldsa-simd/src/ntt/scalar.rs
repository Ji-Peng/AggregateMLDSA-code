//! Signed Montgomery NTT in FIPS 204 order, following pq-crystals Dilithium.
//!
//! Forward inputs have magnitude at most 2^22. Inverse inputs have magnitude
//! below q for the primary prime and at most (p-1)/2 for the CRT primes P2 and
//! P3; callers normalize forward outputs before a standalone inverse round trip.
use super::{Form, Scale};
use crate::{
    params::{Prime, PrimeConsts, CONSTS_P2, CONSTS_P3, CONSTS_Q},
    poly::Poly32,
};

const fn twiddles(c: PrimeConsts) -> [(i32, i32); 256] {
    let mut out = [(0, 0); 256];
    let mut k = 0;
    while k < 256 {
        let mut e = (k as u8).reverse_bits();
        let mut b = c.zeta as i64;
        let mut x = 1i64;
        while e != 0 {
            if e & 1 != 0 {
                x = x * b % c.p as i64;
            }
            b = b * b % c.p as i64;
            e >>= 1;
        }
        x = x * ((1i64 << 32) % c.p as i64) % c.p as i64;
        if x > c.p as i64 / 2 {
            x -= c.p as i64;
        }
        out[k] = (x as i32, (x as i32).wrapping_mul(c.pinv));
        k += 1;
    }
    out
}
const ZQ: [(i32, i32); 256] = twiddles(CONSTS_Q);
const ZP: [(i32, i32); 256] = twiddles(CONSTS_P2);
const ZP3: [(i32, i32); 256] = twiddles(CONSTS_P3);

#[inline(always)]
pub(crate) fn mont(a: i64, c: PrimeConsts) -> i32 {
    let t = (a as i32).wrapping_mul(c.pinv);
    ((a >> 32) - ((t as i64 * c.p as i64) >> 32)) as i32
}
#[inline]
fn mul(a: i32, z: (i32, i32), p: i32) -> i32 {
    let t = a.wrapping_mul(z.1);
    (((a as i64 * z.0 as i64) >> 32) - ((t as i64 * p as i64) >> 32)) as i32
}
#[inline]
fn centered(x: i32, c: PrimeConsts) -> i32 {
    // Montgomery multiplication by R preserves the residue, giving |r| < p.
    let r = mont(x as i64 * c.mont as i64, c);
    let r = r + ((r >> 31) & c.p);
    r - (((c.p / 2 - r) >> 31) & c.p)
}
pub(crate) fn ntt(p: Prime, a: &mut Poly32) {
    match p {
        Prime::Q => forward::<0>(a),
        Prime::P2 => forward::<1>(a),
        Prime::P3 => forward::<2>(a),
    }
}
fn forward<const P_IDX: u8>(a: &mut Poly32) {
    let (c, zs) = match P_IDX {
        0 => (CONSTS_Q, &ZQ),
        1 => (CONSTS_P2, &ZP),
        _ => (CONSTS_P3, &ZP3),
    };
    let mut k = 0;
    let mut len = 128;
    while len > 0 {
        for start in (0..256).step_by(2 * len) {
            k += 1;
            let (low, high) = a.0[start..start + 2 * len].split_at_mut(len);
            for (x, y) in low.iter_mut().zip(high) {
                let t = mul(*y, zs[k], c.p);
                // The recurrence b <- b*(1+p/(2R))+p/2 bounds all
                // eight lazy layers by 5p < 2^31 (R = 2^32).
                *y = *x - t;
                *x += t;
            }
        }
        len >>= 1;
    }
}
pub(crate) fn intt(p: Prime, a: &mut Poly32, s: Scale, f: Form) {
    match p {
        Prime::Q => inverse::<0>(a, s, f),
        Prime::P2 => inverse::<1>(a, s, f),
        Prime::P3 => inverse::<2>(a, s, f),
    }
}
fn inverse<const P_IDX: u8>(a: &mut Poly32, s: Scale, f: Form) {
    let (c, zs) = match P_IDX {
        0 => (CONSTS_Q, &ZQ),
        1 => (CONSTS_P2, &ZP),
        _ => (CONSTS_P3, &ZP3),
    };
    let mut k = 256;
    let mut len = 1;
    while len < 256 {
        for start in (0..256).step_by(2 * len) {
            k -= 1;
            let z = (-zs[k].0, zs[k].1.wrapping_neg());
            let (low, high) = a.0[start..start + 2 * len].split_at_mut(len);
            for (x, y) in low.iter_mut().zip(high) {
                let t = *x;
                // q input |x| < q gives sums below 256q < 2^31.
                // Centered P2 inputs give sums <= 8p through layer four.
                // Centered P3 inputs would reach 8p > 2^31 there, so P3
                // reduces its sums twice, at len = 4 and len = 16.
                *x = t + *y;
                *y = mul(t - *y, z, c.p);
                if (P_IDX == 1 && len == 8) || (P_IDX == 2 && (len == 4 || len == 16)) {
                    // P2: |sum| <= 8p becomes |r| < 0.626p. Four remaining
                    // doublings stay below 10.016p < 2^31; the difference
                    // outputs of this layer are already below 0.657p.
                    // P3: |sum| <= 4p at len = 4 and < 2.56p at len = 16
                    // become |r| < 0.67p; every later sum stays below
                    // 4.68p < 2^31 (5.69p).
                    *x = mont(*x as i64 * c.mont as i64, c);
                }
            }
        }
        len <<= 1;
    }
    let scale = match s {
        Scale::Plain => c.inv_plain,
        Scale::Mont => c.inv_mont,
    };
    for x in a.iter_mut() {
        let r = mont(*x as i64 * scale as i64, c);
        // Final scaled Montgomery output has |r| < p; normalize directly.
        let r = r + ((r >> 31) & c.p);
        *x = match f {
            Form::Canonical => r,
            Form::Centered => r - (((c.p / 2 - r) >> 31) & c.p),
        };
    }
}
pub(crate) fn dot_l<const L: usize>(p: Prime, a: &[Poly32; L], b: &[Poly32; L], out: &mut Poly32) {
    let c = p.consts();
    for i in 0..256 {
        // Canonical q / centered CRT prime matrix times lazy NTT(z), L <= 7:
        // |sum| < 2^58.6 for P2 and < 2^61 for P3. The Montgomery output then
        // lies below 1.9p, and the centering accepts any i32.
        let mut sum = 0i64;
        for j in 0..L {
            sum += a[j][i] as i64 * b[j][i] as i64;
        }
        let r = mont(sum, c);
        out[i] = if p != Prime::Q { centered(r, c) } else { r };
    }
}
#[allow(dead_code)]
pub(crate) fn dot4(p: Prime, a: &[Poly32; 4], b: &[Poly32; 4], out: &mut Poly32) {
    dot_l::<4>(p, a, b, out)
}
pub(crate) fn reduce(p: Prime, a: &mut Poly32, f: Form) {
    let c = p.consts();
    for x in a.iter_mut() {
        let r = centered(*x, c);
        *x = match f {
            Form::Centered => r,
            Form::Canonical => r + ((r >> 31) & c.p),
        };
    }
}
pub(crate) fn order_from_fips(_: &mut Poly32) {}
pub(crate) fn order_to_fips(_: &mut Poly32) {}
