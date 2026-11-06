//! Parameter-derived twiddles for the register shuffle network. Every value is
//! computed at compile time from the prime, its 512th root of unity and its
//! Montgomery constants.
use crate::params::{PrimeConsts, CONSTS_P2, CONSTS_P3, CONSTS_Q};

#[repr(C, align(64))]
pub(super) struct Lanes(pub [i32; 16]);
#[repr(C)]
pub(super) struct Tables {
    pub z: [Lanes; 64],
    pub zi: [Lanes; 64],
    pub iz: [Lanes; 64],
    pub izi: [Lanes; 64],
    // Odd-lane twiddles of the last forward (first inverse) layer, moved into
    // the even dword of each qword where the signed 32x32 multiply reads them.
    // Earlier layers share one twiddle per qword and need no second table.
    pub zo: [Lanes; 8],
    pub zio: [Lanes; 8],
    pub izo: [Lanes; 8],
    pub izio: [Lanes; 8],
    pub scale: [Lanes; 2],
    pub scale_i: [Lanes; 2],
    pub last: [Lanes; 2],
    pub last_i: [Lanes; 2],
}
const fn pow(mut a: i64, mut n: u32, p: i64) -> i64 {
    a %= p;
    let mut r = 1;
    while n != 0 {
        if n & 1 != 0 {
            r = r * a % p;
        }
        a = a * a % p;
        n >>= 1;
    }
    r
}
const fn centered(x: i64, p: i64) -> i32 {
    let x = x % p;
    (if x > p / 2 { x - p } else { x }) as i32
}
const fn reverse8(mut x: usize) -> u32 {
    let mut y = 0;
    let mut i = 0;
    while i < 8 {
        y = (y << 1) | (x & 1) as u32;
        x >>= 1;
        i += 1;
    }
    y
}
// Exchange the upper half of each 2*width group with the lower half
// of its partner register. The same exchange is its own inverse.
pub(super) const fn exchange(
    a: [usize; 16],
    b: [usize; 16],
    width: usize,
) -> ([usize; 16], [usize; 16]) {
    let mut x = a;
    let mut y = b;
    let mut i = 0;
    while i < 16 {
        if i & width != 0 {
            x[i] = b[i - width];
        } else {
            y[i] = a[i + width];
        }
        i += 1;
    }
    (x, y)
}
const fn make(c: PrimeConsts) -> Tables {
    let mut t = Tables {
        scale: [const { Lanes([0; 16]) }; 2],
        scale_i: [const { Lanes([0; 16]) }; 2],
        last: [const { Lanes([0; 16]) }; 2],
        last_i: [const { Lanes([0; 16]) }; 2],
        iz: [const { Lanes([0; 16]) }; 64],
        izi: [const { Lanes([0; 16]) }; 64],
        z: [const { Lanes([0; 16]) }; 64],
        zi: [const { Lanes([0; 16]) }; 64],
        zo: [const { Lanes([0; 16]) }; 8],
        zio: [const { Lanes([0; 16]) }; 8],
        izo: [const { Lanes([0; 16]) }; 8],
        izio: [const { Lanes([0; 16]) }; 8],
    };
    let mut labels = [[0usize; 16]; 16];
    let mut i = 0;
    while i < 256 {
        labels[i / 16][i % 16] = i;
        i += 1;
    }
    let mut stage = 0;
    while stage < 8 {
        let len = 128 >> stage;
        let mut pair = 0;
        let mut base = 0;
        while base < 16 {
            let stride = if stage < 4 { 8 >> stage } else { 1 };
            let mut j = 0;
            while j < stride {
                let a = base + j;
                let b = a + stride;
                if stage >= 4 {
                    let (x, y) = exchange(labels[a], labels[b], len);
                    labels[a] = x;
                    labels[b] = y;
                }
                let mut lane = 0;
                while lane < 16 {
                    let index = (1 << stage) + labels[a][lane] / (2 * len);
                    let z = centered(
                        pow(c.zeta as i64, reverse8(index), c.p as i64) * (1i64 << 32)
                            % (c.p as i64),
                        c.p as i64,
                    );
                    t.z[stage * 8 + pair].0[lane] = z;
                    t.zi[stage * 8 + pair].0[lane] = z.wrapping_mul(c.pinv);
                    let index = (1 << (stage + 1)) - 1 - labels[a][lane] / (2 * len);
                    let iz = -centered(
                        pow(c.zeta as i64, reverse8(index), c.p as i64) * (1i64 << 32)
                            % (c.p as i64),
                        c.p as i64,
                    );
                    t.iz[stage * 8 + pair].0[lane] = iz;
                    t.izi[stage * 8 + pair].0[lane] = iz.wrapping_mul(c.pinv);
                    lane += 1;
                }
                pair += 1;
                j += 1;
            }
            base += 2 * stride;
        }
        stage += 1;
    }
    let mut k = 0;
    while k < 8 {
        let mut lane = 0;
        while lane < 16 {
            t.zo[k].0[lane] = t.z[56 + k].0[lane | 1];
            t.zio[k].0[lane] = t.zi[56 + k].0[lane | 1];
            t.izo[k].0[lane] = t.iz[56 + k].0[lane | 1];
            t.izio[k].0[lane] = t.izi[56 + k].0[lane | 1];
            lane += 1;
        }
        k += 1;
    }
    let mut i = 0;
    while i < 2 {
        let f = if i == 0 { c.inv_plain } else { c.inv_mont };
        // Fuse the last inverse root with the requested normalization.
        let root = (t.iz[0].0[0] as i64).rem_euclid(c.p as i64);
        let r_inverse = pow(1i64 << 32, c.p as u32 - 2, c.p as i64);
        let last = centered(
            root * f as i64 % (c.p as i64) * r_inverse % (c.p as i64),
            c.p as i64,
        );
        t.scale[i] = Lanes([f; 16]);
        t.scale_i[i] = Lanes([f.wrapping_mul(c.pinv); 16]);
        t.last[i] = Lanes([last; 16]);
        t.last_i[i] = Lanes([last.wrapping_mul(c.pinv); 16]);
        i += 1;
    }
    t
}
pub(super) static Q: Tables = make(CONSTS_Q);
pub(super) static P2: Tables = make(CONSTS_P2);
pub(super) static P3: Tables = make(CONSTS_P3);

// Conservative integer bounds in natural coefficient order after each GS layer.
// Each multiply uses |Mont(x*z)| <= ceil(|x*z|/2^32) + floor(p/2).
// The short reduction of selected lanes leaves |x| < 0.61p.
pub(super) const fn inverse_bounds() -> ([[i64; 256]; 8], [bool; 256]) {
    let p = CONSTS_P2.p as i64;
    let mut b = [p / 2; 256];
    let mut layers = [[0; 256]; 8];
    let mut reduce = [false; 256];
    let mut stage = 0;
    while stage < 8 {
        let len = 1 << stage;
        let mut base = 0;
        while base < 256 {
            let index = (512 >> (stage + 1)) - 1 - base / (2 * len);
            let z = centered(
                pow(CONSTS_P2.zeta as i64, reverse8(index), p) * (1i64 << 32) % p,
                p,
            ) as i64;
            let z = if z < 0 { -z } else { z };
            let mut j = base;
            while j < base + len {
                let sum = b[j] + b[j + len];
                b[j] = sum;
                b[j + len] = (sum * z + ((1i64 << 32) - 1)) / (1i64 << 32) + p / 2;
                j += 1;
            }
            base += 2 * len;
        }
        if stage == 3 {
            let mut j = 0;
            while j < 256 {
                if b[j] > p {
                    reduce[j] = true;
                    // Rounded x/2^25 followed by rounded division by five.
                    // |n|<=41 and 6554/32768 differs from 1/5 by < 1/65536:
                    // |x-q*p| < (0.5/5 + 0.5 + 41/65536)*p < 0.61p.
                    b[j] = (61 * p + 99) / 100;
                }
                j += 1;
            }
        }
        layers[stage] = b;
        stage += 1;
    }
    (layers, reduce)
}
pub(super) const INVERSE_BOUNDS: ([[i64; 256]; 8], [bool; 256]) = inverse_bounds();
pub(super) const fn reduction_mask(reg: usize) -> u16 {
    let mut m = 0;
    let mut i = 0;
    while i < 16 {
        if INVERSE_BOUNDS.1[reg * 16 + i] {
            m |= 1 << i;
        }
        i += 1;
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn forward_bounds() {
        for c in [CONSTS_Q, CONSTS_P2, CONSTS_P3] {
            let p = c.p as i64;
            let mut bounds = [1i64 << 22; 256];
            for stage in 0..8 {
                let len = 128 >> stage;
                for base in (0..256).step_by(2 * len) {
                    let k = (1 << stage) + base / (2 * len);
                    let z = (centered(pow(c.zeta as i64, reverse8(k), p) * (1i64 << 32) % p, p)
                        as i64)
                        .abs();
                    for j in base..base + len {
                        let product =
                            (z * bounds[j + len] + ((1i64 << 32) - 1)) / (1i64 << 32) + p / 2;
                        let bound = bounds[j] + product;
                        assert!(bound < 1i64 << 31);
                        bounds[j] = bound;
                        bounds[j + len] = bound;
                    }
                }
            }
            let hundredths = match c.p {
                p if p == CONSTS_Q.p => 452,
                p if p == CONSTS_P2.p => 425,
                _ => 455,
            };
            assert!(bounds.into_iter().all(|b| b * 100 < hundredths * p));
        }
    }
    #[test]
    fn short_reduction_bound() {
        let p = CONSTS_P2.p as i64;
        // Each rounded n corresponds to an interval of width 2^25. The
        // residual is linear in x there, so checking its endpoints suffices.
        for n in -41i64..=41 {
            let q = (n * 6554 + (1 << 14)) >> 15;
            for x in [n * (1 << 25) - (1 << 24), n * (1 << 25) + (1 << 24) - 1] {
                if x.abs() <= 8 * p {
                    assert!((x - q * p).abs() * 100 < 61 * p);
                }
            }
        }
    }
    #[test]
    fn inverse_bound_trace() {
        let p = CONSTS_P2.p as i64;
        let mut seed = 1u64;
        for trial in 0..128 {
            let mut a = [0i64; 256];
            for (i, x) in a.iter_mut().enumerate() {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                *x = match trial {
                    0 => p / 2,
                    1 => -p / 2,
                    2 => {
                        if i % 2 == 0 {
                            p / 2
                        } else {
                            -p / 2
                        }
                    }
                    _ => seed as i64 % (p / 2),
                };
            }
            for stage in 0..8 {
                let len = 1 << stage;
                for base in (0..256).step_by(2 * len) {
                    let k = (512 >> (stage + 1)) - 1 - base / (2 * len);
                    let z = -centered(
                        pow(CONSTS_P2.zeta as i64, reverse8(k), p) * (1i64 << 32) % p,
                        p,
                    ) as i64;
                    for j in base..base + len {
                        let x = a[j];
                        let y = a[j + len];
                        assert!((x + y).abs() < 1i64 << 31);
                        assert!((x - y).abs() < 1i64 << 31);
                        a[j] = x + y;
                        let product = (x - y) * z;
                        let t = (product as i32).wrapping_mul(CONSTS_P2.pinv) as i64;
                        a[j + len] = (product - t * p) >> 32;
                    }
                }
                if stage == 3 {
                    for i in 0..256 {
                        if INVERSE_BOUNDS.1[i] {
                            assert!(a[i].abs() <= 8 * p);
                            let n = (a[i] + (1 << 24)) >> 25;
                            let q = (n * 6554 + (1 << 14)) >> 15;
                            a[i] -= q * p;
                        }
                    }
                }
                for i in 0..256 {
                    assert!(a[i].abs() <= INVERSE_BOUNDS.0[stage][i]);
                }
            }
        }
    }
    #[test]
    fn shared_qword_twiddles() {
        // The butterflies feed the even-dword twiddle to both lanes of a qword
        // in every layer but the last, so those layers must agree per qword.
        for t in [&Q, &P2] {
            for table in [&t.z, &t.zi, &t.iz, &t.izi] {
                for v in &table[..56] {
                    assert!((0..16).step_by(2).all(|l| v.0[l] == v.0[l + 1]));
                }
            }
            for (odd, full) in [
                (&t.zo, &t.z),
                (&t.zio, &t.zi),
                (&t.izo, &t.iz),
                (&t.izio, &t.izi),
            ] {
                for k in 0..8 {
                    assert!((0..16).all(|l| odd[k].0[l] == full[56 + k].0[l | 1]));
                }
            }
        }
    }
    #[test]
    fn inverse_headroom() {
        for layer in INVERSE_BOUNDS.0 {
            for bound in layer {
                assert!(bound < 1i64 << 31);
            }
        }
        assert!(INVERSE_BOUNDS.1.iter().any(|&x| x));
    }
}

const _: () = assert!(core::mem::size_of::<Lanes>() == 64);
