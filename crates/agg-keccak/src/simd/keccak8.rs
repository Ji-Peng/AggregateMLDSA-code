//! Keccak-f[1600] on eight lane-sliced states.
//!
//! A [`State8`] holds eight states: vector `L` is lane `L = x + 5 y` of every state, qword
//! `r` of each vector belongs to state `r`. One round is theta (column parities with
//! `vpternlogq`, `D[x] = C[x - 1] ^ rol(C[x + 1], 1)`), rho (`vprolq`), pi (register
//! renaming), chi and iota, exactly as [`crate::keccak::round`] on each qword.

use core::arch::x86_64::{
    __m512i, _mm512_andnot_si512, _mm512_rolv_epi64, _mm512_set1_epi64, _mm512_setzero_si512,
    _mm512_ternarylogic_epi64, _mm512_xor_si512,
};

use crate::keccak::{lane, N_LANES, RHO_OFFSETS, ROUND_CONSTANTS};

/// Eight lane-sliced Keccak states.
pub type State8 = [__m512i; N_LANES];

/// The all-zero state.
#[inline]
#[target_feature(enable = "avx512f")]
#[must_use]
pub fn zero8() -> State8 {
    [_mm512_setzero_si512(); N_LANES]
}

/// `rol(v, n)` on every qword.
#[inline]
#[target_feature(enable = "avx512f")]
fn rol(v: __m512i, n: u32) -> __m512i {
    _mm512_rolv_epi64(v, _mm512_set1_epi64(i64::from(n)))
}

/// `a ^ b ^ c`.
#[inline]
#[target_feature(enable = "avx512f")]
fn xor3(a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    _mm512_ternarylogic_epi64::<0x96>(a, b, c)
}

/// `phi = pi o rho o theta` on eight states: the vector `B` that chi reads,
/// `B[x, y] = rol(A'[(x + 3 y) mod 5, x], r[(x + 3 y) mod 5][x])` with `A' = theta(A)`.
#[inline]
#[target_feature(enable = "avx512f")]
#[must_use]
pub fn phi8(s: &State8) -> State8 {
    let mut c = [_mm512_setzero_si512(); 5];
    for x in 0..5 {
        c[x] = xor3(xor3(s[x], s[x + 5], s[x + 10]), s[x + 15], s[x + 20]);
    }
    let mut d = [_mm512_setzero_si512(); 5];
    for x in 0..5 {
        d[x] = _mm512_xor_si512(c[(x + 4) % 5], rol(c[(x + 1) % 5], 1));
    }
    let mut b = [_mm512_setzero_si512(); N_LANES];
    for y in 0..5 {
        for x in 0..5 {
            let a = (x + 3 * y) % 5;
            b[lane(x, y)] = rol(_mm512_xor_si512(s[lane(a, x)], d[a]), RHO_OFFSETS[a][x]);
        }
    }
    b
}

/// The chi AND output of lane `(x, y)` of a phi output: `!b[x + 1, y] & b[x + 2, y]`.
#[inline]
#[target_feature(enable = "avx512f")]
#[must_use]
pub fn chi_and_lane(b: &State8, x: usize, y: usize) -> __m512i {
    _mm512_andnot_si512(b[lane((x + 1) % 5, y)], b[lane((x + 2) % 5, y)])
}

/// One round with round index `r` on eight states: updates `s` and returns the chi AND
/// outputs `t` (the committed chi lanes), as [`crate::keccak::round`] per qword.
#[inline]
#[target_feature(enable = "avx512f")]
pub fn round8(s: &mut State8, r: usize) -> State8 {
    let b = phi8(s);
    let mut t = [_mm512_setzero_si512(); N_LANES];
    for y in 0..5 {
        for x in 0..5 {
            let l = lane(x, y);
            t[l] = chi_and_lane(&b, x, y);
            s[l] = _mm512_xor_si512(b[l], t[l]);
        }
    }
    s[0] = _mm512_xor_si512(s[0], _mm512_set1_epi64(ROUND_CONSTANTS[r] as i64));
    t
}

/// XOR the first `planes.len()` lanes of eight absorbed blocks into the states.
#[inline]
#[target_feature(enable = "avx512f")]
pub fn absorb8(s: &mut State8, planes: &[__m512i]) {
    debug_assert!(planes.len() <= N_LANES);
    for (l, &p) in planes.iter().enumerate() {
        s[l] = _mm512_xor_si512(s[l], p);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keccak::{self, Lanes};
    use core::arch::x86_64::{_mm512_loadu_si512, _mm512_storeu_si512};
    use flock_core::test_rng::Rng;

    /// Eight states to lane-sliced vectors.
    fn load(states: &[Lanes; 8]) -> State8 {
        let mut out = [[0u64; 8]; N_LANES];
        for (r, st) in states.iter().enumerate() {
            for l in 0..N_LANES {
                out[l][r] = st[l];
            }
        }
        // SAFETY: each `out[l]` is 64 readable bytes; the caller checked avx512f.
        core::array::from_fn(|l| unsafe { _mm512_loadu_si512(out[l].as_ptr().cast()) })
    }

    fn store(v: &State8) -> [Lanes; 8] {
        let mut out = [[0u64; N_LANES]; 8];
        for l in 0..N_LANES {
            let mut q = [0u64; 8];
            // SAFETY: `q` is 64 writable bytes; the caller checked avx512f.
            unsafe { _mm512_storeu_si512(q.as_mut_ptr().cast(), v[l]) };
            for r in 0..8 {
                out[r][l] = q[r];
            }
        }
        out
    }

    /// All 24 rounds and their chi outputs equal the scalar round on each of eight random
    /// states.
    #[test]
    fn round8_matches_scalar_round() {
        if !crate::simd::detected() {
            eprintln!("no AVX-512 on this CPU: vector Keccak not exercised");
            return;
        }
        let mut rng = Rng::new(0x8EC);
        for _ in 0..4 {
            let mut states: [Lanes; 8] =
                core::array::from_fn(|_| core::array::from_fn(|_| rng.next_u64()));
            let mut v = load(&states);
            for r in 0..keccak::N_ROUNDS {
                let want: [Lanes; 8] = core::array::from_fn(|i| keccak::round(&mut states[i], r));
                // SAFETY: AVX-512 F detected above.
                let t = unsafe { round8(&mut v, r) };
                assert_eq!(store(&t), want, "chi outputs, round {r}");
                assert_eq!(store(&v), states, "state after round {r}");
            }
        }
    }
}
