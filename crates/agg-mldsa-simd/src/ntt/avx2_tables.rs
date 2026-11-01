//! Generated-at-compile-time pq-crystals AVX2 qdata layout (CC0/Apache-2.0),
//! followed by twiddle vectors that the kernels would otherwise shuffle per call.
//!
//! The second field holds, for every eight-word twiddle group starting at
//! word `i` of the first field's `_ZETAS_QINV`/`_ZETAS` regions:
//! - forward groups `i = 72..=288`: the odd words duplicated into both lanes
//!   of each pair (`vmovshdup`), at vector `2 * (i / 8 - 9) + z`;
//! - inverse groups `i = 8..=288`: the qword-reversed group (`vpermq 0x1B`)
//!   and its odd-word duplicate, at vector `FWD_VECS + 4 * (i / 8 - 1) + 2 * z + odd`;
//!
//! where `z` is 0 for `_ZETAS_QINV` and 1 for `_ZETAS`. The kernels only read the
//! even word of each pair, so every vector reproduces the shuffled operand exactly.
use super::Scale;
use crate::params::{consts, Prime};
/// Number of forward odd-word vectors (28 groups, two regions).
pub(super) const FWD_VECS: usize = 56;
/// Number of inverse vectors (36 groups, two regions, two shuffles).
pub(super) const INV_VECS: usize = 144;
/// Words after the 624-word reference layout.
pub(super) const EXTRA: usize = 8 * (FWD_VECS + INV_VECS);
// Page alignment gives every table the same placement modulo 4 KiB, so 4K
// store-to-load aliasing with the polynomial does not depend on prime or scale.
#[repr(C, align(4096))]
pub(super) struct Table(pub [i32; 624], pub [i32; EXTRA]);
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
    let mut x = [0; EXTRA];
    // Forward groups: odd words duplicated.
    let mut g = 72;
    while g <= 288 {
        let mut z = 0;
        while z < 2 {
            let base = if z == 0 { 32 } else { 328 } + g;
            let v = 8 * (2 * (g / 8 - 9) + z);
            let mut j = 0;
            while j < 8 {
                x[v + j] = t[base + (j | 1)];
                j += 1;
            }
            z += 1;
        }
        g += 8;
    }
    // Inverse groups: qword-reversed, then its odd words duplicated.
    g = 8;
    while g <= 288 {
        let mut z = 0;
        while z < 2 {
            let base = if z == 0 { 32 } else { 328 } + g;
            let v = 8 * (FWD_VECS + 4 * (g / 8 - 1) + 2 * z);
            let mut j = 0;
            while j < 8 {
                // Lane j of vpermq 0x1B reads qword 3 - j / 2, same word parity.
                let r = 2 * (3 - j / 2) + (j & 1);
                x[v + j] = t[base + r];
                x[v + 8 + j] = t[base + (r | 1)];
                j += 1;
            }
            z += 1;
        }
        g += 8;
    }
    Table(t, x)
}
pub(super) static Q: Table = generate(Prime::Q, Scale::Mont);
pub(super) static Q_PLAIN: Table = generate(Prime::Q, Scale::Plain);
pub(super) static P2: Table = generate(Prime::P2, Scale::Mont);
pub(super) static P2_PLAIN: Table = generate(Prime::P2, Scale::Plain);
pub(super) static P3: Table = generate(Prime::P3, Scale::Mont);
pub(super) static P3_PLAIN: Table = generate(Prime::P3, Scale::Plain);
pub(super) fn table(p: Prime, s: Scale) -> &'static Table {
    match (p, s) {
        (Prime::Q, Scale::Mont) => &Q,
        (Prime::Q, Scale::Plain) => &Q_PLAIN,
        (Prime::P2, Scale::Mont) => &P2,
        (Prime::P2, Scale::Plain) => &P2_PLAIN,
        (Prime::P3, Scale::Mont) => &P3,
        (Prime::P3, Scale::Plain) => &P3_PLAIN,
    }
}
