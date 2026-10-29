//! Parameter-derived twiddles for the register shuffle network.
use crate::params::{PrimeConsts, CONSTS_P2, CONSTS_Q};

#[repr(C, align(64))]
pub(super) struct Lanes(pub [i32; 16]);
pub(super) struct Tables {
    pub z: [Lanes; 64],
    pub zi: [Lanes; 64],
    pub iz: [Lanes; 64],
    pub izi: [Lanes; 64],
}
const fn pow(mut a: i64, mut n: u32, p: i64) -> i64 {
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
        iz: [const { Lanes([0; 16]) }; 64],
        izi: [const { Lanes([0; 16]) }; 64],
        z: [const { Lanes([0; 16]) }; 64],
        zi: [const { Lanes([0; 16]) }; 64],
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
    t
}
pub(super) static Q: Tables = make(CONSTS_Q);
pub(super) static P2: Tables = make(CONSTS_P2);

// Conservative integer bounds in natural coefficient order after each GS layer.
// Each multiply uses |Mont(x*z)| <= ceil(|x*z|/2^32) + floor(p/2).
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
                    b[j] = (b[j] * 67_108_890 + ((1i64 << 32) - 1)) / (1i64 << 32) + p / 2;
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
    fn inverse_headroom() {
        for layer in INVERSE_BOUNDS.0 {
            for bound in layer {
                assert!(bound < 1i64 << 31);
            }
        }
        assert!(INVERSE_BOUNDS.1.iter().any(|&x| x));
    }
}
