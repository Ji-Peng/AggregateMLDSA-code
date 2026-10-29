//! Signed Montgomery NTT in FIPS 204 order, following pq-crystals Dilithium.
use super::{Form, Scale};
use crate::{
    params::{Prime, PrimeConsts, CONSTS_P2, CONSTS_Q},
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

#[inline]
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
    let c = p.consts();
    let zs = if p == Prime::Q { &ZQ } else { &ZP };
    let mut k = 0;
    let mut len = 128;
    while len > 0 {
        for start in (0..256).step_by(2 * len) {
            k += 1;
            for j in start..start + len {
                let t = mul(a[j + len], zs[k], c.p);
                // Centered twiddles bound eight lazy layers by 4.4p < 2^31.
                a[j + len] = a[j] - t;
                a[j] += t;
            }
        }
        len >>= 1;
    }
}
pub(crate) fn intt(p: Prime, a: &mut Poly32, s: Scale, f: Form) {
    let c = p.consts();
    let zs = if p == Prime::Q { &ZQ } else { &ZP };
    let mut k = 256;
    let mut len = 1;
    while len < 256 {
        for start in (0..256).step_by(2 * len) {
            k -= 1;
            let z = (-zs[k].0, zs[k].1.wrapping_neg());
            for j in start..start + len {
                let t = a[j];
                // q starts in (-q,q); its sum chains stay below 256q < 2^31.
                // P2 is centered each layer, so sums/differences stay below p.
                a[j] = t + a[j + len];
                a[j + len] = mul(t - a[j + len], z, c.p);
                if p == Prime::P2 {
                    a[j] = centered(a[j], c);
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
        *x = mont(*x as i64 * scale as i64, c);
    }
    reduce(p, a, f);
}
pub(crate) fn dot4(p: Prime, a: &[Poly32; 4], b: &[Poly32; 4], out: &mut Poly32) {
    let c = p.consts();
    for i in 0..256 {
        // Canonical q / centered P2 matrix times lazy NTT(z): |sum| < 2^58.
        let mut sum = 0i64;
        for j in 0..4 {
            sum += a[j][i] as i64 * b[j][i] as i64;
        }
        let r = mont(sum, c);
        out[i] = if p == Prime::P2 { centered(r, c) } else { r };
    }
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
