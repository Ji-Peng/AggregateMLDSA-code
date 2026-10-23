//! Keccak-f[1600] primitives on u64 lanes, the linear layer
//! phi = pi o rho o theta, and its transpose over GF(2^128).
//!
//! State bits are indexed lane-linearly: bit `s = 64 L + z` is bit `z` of lane
//! `L = x + 5 y`. This is FIPS 202's byte order (state byte `b` holds bits
//! `8b .. 8b + 7`), so a sponge block byte `b` XORs into bits `8b .. 8b + 7`.

use flock_field::F128;

/// Bits in a Keccak-f[1600] state.
pub const STATE_BITS: usize = 1600;
/// Lanes in a state.
pub const N_LANES: usize = 25;
/// Rounds of Keccak-f[1600].
pub const N_ROUNDS: usize = 24;

/// Lanes of one state.
pub type Lanes = [u64; N_LANES];

/// rho rotation offsets `r[x][y]` (FIPS 202 Table 2), indexed `[x][y]`.
pub const RHO_OFFSETS: [[u32; 5]; 5] = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
];

/// iota round constants.
pub const ROUND_CONSTANTS: [u64; N_ROUNDS] = [
    0x0000_0000_0000_0001,
    0x0000_0000_0000_8082,
    0x8000_0000_0000_808A,
    0x8000_0000_8000_8000,
    0x0000_0000_0000_808B,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8009,
    0x0000_0000_0000_008A,
    0x0000_0000_0000_0088,
    0x0000_0000_8000_8009,
    0x0000_0000_8000_000A,
    0x0000_0000_8000_808B,
    0x8000_0000_0000_008B,
    0x8000_0000_0000_8089,
    0x8000_0000_0000_8003,
    0x8000_0000_0000_8002,
    0x8000_0000_0000_0080,
    0x0000_0000_0000_800A,
    0x8000_0000_8000_000A,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8080,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8008,
];

/// Lane index of `(x, y)`.
#[inline]
#[must_use]
pub const fn lane(x: usize, y: usize) -> usize {
    x + 5 * y
}

/// Lane-linear bit index of `(x, y, z)`.
#[inline]
#[must_use]
pub const fn bit(x: usize, y: usize, z: usize) -> usize {
    64 * lane(x, y) + z
}

/// theta on lanes (in place).
#[inline]
pub fn theta(s: &mut Lanes) {
    let c: [u64; 5] = std::array::from_fn(|x| s[x] ^ s[x + 5] ^ s[x + 10] ^ s[x + 15] ^ s[x + 20]);
    let d: [u64; 5] = std::array::from_fn(|x| c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1));
    for y in 0..5 {
        for x in 0..5 {
            s[lane(x, y)] ^= d[x];
        }
    }
}

/// rho then pi on lanes: `B[x, y] = rotl(A[(x + 3y) mod 5, x], r[(x + 3y) mod 5][x])`.
#[inline]
#[must_use]
pub fn rho_pi(s: &Lanes) -> Lanes {
    let mut out = [0u64; N_LANES];
    for y in 0..5 {
        for x in 0..5 {
            let a = (x + 3 * y) % 5;
            out[lane(x, y)] = s[lane(a, x)].rotate_left(RHO_OFFSETS[a][x]);
        }
    }
    out
}

/// The linear layer `phi = pi o rho o theta`.
#[inline]
#[must_use]
pub fn phi(s: &Lanes) -> Lanes {
    let mut t = *s;
    theta(&mut t);
    rho_pi(&t)
}

/// The chi AND outputs of a phi output `b`: `t[x, y] = !b[x + 1, y] & b[x + 2, y]`.
/// chi itself is `b ^ t`.
#[inline]
#[must_use]
pub fn chi_and(b: &Lanes) -> Lanes {
    let mut t = [0u64; N_LANES];
    for y in 0..5 {
        for x in 0..5 {
            t[lane(x, y)] = !b[lane((x + 1) % 5, y)] & b[lane((x + 2) % 5, y)];
        }
    }
    t
}

/// One round with round index `r`: returns the chi AND outputs and updates the state.
#[inline]
pub fn round(s: &mut Lanes, r: usize) -> Lanes {
    let b = phi(s);
    let t = chi_and(&b);
    for i in 0..N_LANES {
        s[i] = b[i] ^ t[i];
    }
    s[0] ^= ROUND_CONSTANTS[r];
    t
}

/// Keccak-f[1600] reduced to its first `rounds` rounds (all 24 for the real permutation).
pub fn keccak_f_rounds(s: &mut Lanes, rounds: usize) {
    for r in 0..rounds {
        round(s, r);
    }
}

/// Keccak-f[1600].
pub fn keccak_f(s: &mut Lanes) {
    keccak_f_rounds(s, N_ROUNDS);
}

/// Lanes to lane-linear bits.
#[must_use]
pub fn lanes_to_bits(s: &Lanes) -> Vec<bool> {
    (0..STATE_BITS)
        .map(|i| (s[i / 64] >> (i % 64)) & 1 == 1)
        .collect()
}

/// Lane-linear bits to lanes.
#[must_use]
pub fn bits_to_lanes(bits: &[bool]) -> Lanes {
    assert_eq!(bits.len(), STATE_BITS);
    let mut s = [0u64; N_LANES];
    for (i, &b) in bits.iter().enumerate() {
        s[i / 64] |= u64::from(b) << (i % 64);
    }
    s
}

/// Transpose of [`phi`] over `F_{2^128}`: for weights `g` on the bits of `phi(s)`,
/// returns the weights `k` on the bits of `s` with `<g, phi(s)> = <k, s>` for every
/// state `s` (both inner products taken in characteristic 2).
#[must_use]
pub fn phi_transpose(g: &[F128]) -> Vec<F128> {
    assert_eq!(g.len(), STATE_BITS);
    // (rho pi)^T: weight on A1[a, b, c] = g[B position (x = b, y = 2(a - b), z = c + r[a][b])].
    let mut g1 = vec![F128::ZERO; STATE_BITS];
    for y in 0..5 {
        for x in 0..5 {
            let a = (x + 3 * y) % 5;
            let r = RHO_OFFSETS[a][x] as usize;
            let src = 64 * lane(x, y);
            let dst = 64 * lane(a, x);
            for c in 0..64 {
                g1[dst + c] = g[src + (c + r) % 64];
            }
        }
    }
    // theta^T: k[x, y, z] = g1[x, y, z] + S[x + 1, z] + S[x - 1, z + 1], S[x, z] = sum_y g1[x, y, z].
    let mut col = [[F128::ZERO; 64]; 5];
    for (x, cx) in col.iter_mut().enumerate() {
        for y in 0..5 {
            let base = 64 * lane(x, y);
            for z in 0..64 {
                cx[z] += g1[base + z];
            }
        }
    }
    let mut k = g1;
    for x in 0..5 {
        let xp = (x + 1) % 5;
        let xm = (x + 4) % 5;
        let mut d = [F128::ZERO; 64];
        for z in 0..64 {
            d[z] = col[xp][z] + col[xm][(z + 1) % 64];
        }
        for y in 0..5 {
            let base = 64 * lane(x, y);
            for z in 0..64 {
                k[base + z] += d[z];
            }
        }
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;
    use flock_core::test_rng::Rng;

    fn random_lanes(rng: &mut Rng) -> Lanes {
        std::array::from_fn(|_| rng.next_u64())
    }

    #[test]
    fn keccak_f_matches_keccak_crate() {
        let mut rng = Rng::new(0xC0FFEE);
        for _ in 0..16 {
            let s0 = random_lanes(&mut rng);
            let mut ours = s0;
            keccak_f(&mut ours);
            let mut theirs = s0;
            keccak::f1600(&mut theirs);
            assert_eq!(ours, theirs);
        }
    }

    #[test]
    fn phi_transpose_is_adjoint() {
        let mut rng = Rng::new(7);
        for _ in 0..8 {
            let s = random_lanes(&mut rng);
            let g: Vec<F128> = (0..STATE_BITS)
                .map(|_| F128::new(rng.next_u64(), rng.next_u64()))
                .collect();
            let ps = lanes_to_bits(&phi(&s));
            let lhs = g
                .iter()
                .zip(&ps)
                .filter(|(_, &b)| b)
                .fold(F128::ZERO, |acc, (&w, _)| acc + w);
            let k = phi_transpose(&g);
            let sb = lanes_to_bits(&s);
            let rhs = k
                .iter()
                .zip(&sb)
                .filter(|(_, &b)| b)
                .fold(F128::ZERO, |acc, (&w, _)| acc + w);
            assert_eq!(lhs, rhs);
        }
    }

    #[test]
    fn bit_conversions_round_trip() {
        let mut rng = Rng::new(3);
        let s = random_lanes(&mut rng);
        assert_eq!(bits_to_lanes(&lanes_to_bits(&s)), s);
        assert_eq!(bit(1, 2, 3), 64 * 11 + 3);
    }
}
