//! Generated-at-compile-time pq-crystals AVX2 qdata layout (CC0/Apache-2.0).
use super::Scale;
use crate::params::{consts, Prime};
#[repr(C, align(32))]
pub(super) struct Table(pub [i32; 624]);
const fn pow(mut a: i64, mut n: u32, p: i64) -> i64 {
    let mut r = 1;
    while n > 0 {
        if n & 1 != 0 {
            r = r * a % p;
        }
        a = a * a % p;
        n >>= 1;
    }
    r
}
const fn centered(x: i64, p: i64) -> i32 {
    let x = x.rem_euclid(p);
    if x > p / 2 {
        (x - p) as i32
    } else {
        x as i32
    }
}
const fn generate(prime: Prime, scale: Scale) -> Table {
    let c = consts(prime);
    let p = c.p as i64;
    let mut t = [0; 624];
    let f = match scale {
        Scale::Plain => c.inv_plain,
        Scale::Mont => c.inv_mont,
    };
    let mut i = 0;
    while i < 8 {
        t[i] = c.p;
        t[8 + i] = c.pinv;
        t[16 + i] = f.wrapping_mul(c.pinv);
        t[24 + i] = f;
        i += 1;
    }
    i = 0;
    while i < 296 {
        let k = if i < 8 {
            i
        } else if i < 40 {
            8 + (i - 8) / 4
        } else if i < 72 {
            16 + (i - 40) / 2
        } else if i < 104 {
            32 + i - 72
        } else if i < 168 {
            64 + 2 * ((i - 104) % 32) + (i - 104) / 32
        } else {
            128 + 4 * ((i - 168) % 32) + (i - 168) / 32
        };
        let z = if i == 0 {
            centered(pow(c.zeta as i64, 128, p) * f as i64, p)
        } else {
            let rev = (k as u8).reverse_bits() as u32;
            centered(pow(c.zeta as i64, rev, p) * (c.mont as i64), p)
        };
        t[328 + i] = z;
        t[32 + i] = z.wrapping_mul(c.pinv);
        i += 1;
    }
    Table(t)
}
pub(super) static Q: Table = generate(Prime::Q, Scale::Mont);
pub(super) static Q_PLAIN: Table = generate(Prime::Q, Scale::Plain);
pub(super) static P2: Table = generate(Prime::P2, Scale::Mont);
pub(super) static P2_PLAIN: Table = generate(Prime::P2, Scale::Plain);
pub(super) fn table(p: Prime, s: Scale) -> &'static Table {
    match (p, s) {
        (Prime::Q, Scale::Mont) => &Q,
        (Prime::Q, Scale::Plain) => &Q_PLAIN,
        (Prime::P2, Scale::Mont) => &P2,
        (Prime::P2, Scale::Plain) => &P2_PLAIN,
    }
}
