//! Keccak-f[1600] rounds that expose the chi row values of a gate witness, and
//! the bit transpose of a lincheck stripe, on groups of eight states.
//!
//! A chi-only Keccak gate commits, per round, the chi AND outputs and needs
//! the chi row values of the round. With `phi = pi o rho o theta` of the state,
//! lane `(x, y)` of a round gives
//!
//! ```text
//! a = !phi[x + 1, y],  b = phi[x + 2, y],  t = a & b,  next = phi[x, y] ^ t (^ RC on lane 0).
//! ```
//!
//! `b` is `phi` with its lanes permuted within each row, so the outputs expose
//! `phi` too. States are word-major: `state[w][i]` is lane `w` of state `i`,
//! the layout of the witness tiles that consume the outputs. The scalar
//! backend walks the eight states one at a time, AVX2 in two halves of four,
//! AVX-512 all eight in lockstep; outputs are equal on every backend.
//!
//! The stripe transpose maps eight words (one per state) to 64 bytes:
//! byte `bit` holds bit `bit` of state `r` at bit `r`. Scalar uses three delta
//! swaps per 8x8 bit matrix; the vector versions first transpose the 8x8 byte
//! matrix with byte and word shuffles, then run the same delta swaps on all
//! eight matrices of a word at once.

/// One Keccak lane of each of eight states.
pub type Words8 = [u64; 8];

/// iota round constants (FIPS 202).
const RC: [u64; 24] = [
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

/// One round on 25 words of any width. `$rol!(v, n)` rotates left by the
/// literal `n`; `$emit(l, a, b, t)` receives the chi row values of lane `l`.
/// rho and pi: lane `x + 5 y` of `phi` is lane `(x + 3 y) mod 5 + 5 x` of the
/// theta output, rotated by its rho offset (FIPS 202 Table 2).
macro_rules! keccak_round {
    ($s:expr, $rc:expr, $xor:ident, $andn:ident, $not:ident, $rol:ident, $emit:ident) => {{
        let s: &mut [_; 25] = $s;
        let c = [
            $xor($xor($xor($xor(s[0], s[5]), s[10]), s[15]), s[20]),
            $xor($xor($xor($xor(s[1], s[6]), s[11]), s[16]), s[21]),
            $xor($xor($xor($xor(s[2], s[7]), s[12]), s[17]), s[22]),
            $xor($xor($xor($xor(s[3], s[8]), s[13]), s[18]), s[23]),
            $xor($xor($xor($xor(s[4], s[9]), s[14]), s[19]), s[24]),
        ];
        let d = [
            $xor(c[4], $rol!(c[1], 1)),
            $xor(c[0], $rol!(c[2], 1)),
            $xor(c[1], $rol!(c[3], 1)),
            $xor(c[2], $rol!(c[4], 1)),
            $xor(c[3], $rol!(c[0], 1)),
        ];
        let p = [
            $xor(s[0], d[0]),
            $rol!($xor(s[6], d[1]), 44),
            $rol!($xor(s[12], d[2]), 43),
            $rol!($xor(s[18], d[3]), 21),
            $rol!($xor(s[24], d[4]), 14),
            $rol!($xor(s[3], d[3]), 28),
            $rol!($xor(s[9], d[4]), 20),
            $rol!($xor(s[10], d[0]), 3),
            $rol!($xor(s[16], d[1]), 45),
            $rol!($xor(s[22], d[2]), 61),
            $rol!($xor(s[1], d[1]), 1),
            $rol!($xor(s[7], d[2]), 6),
            $rol!($xor(s[13], d[3]), 25),
            $rol!($xor(s[19], d[4]), 8),
            $rol!($xor(s[20], d[0]), 18),
            $rol!($xor(s[4], d[4]), 27),
            $rol!($xor(s[5], d[0]), 36),
            $rol!($xor(s[11], d[1]), 10),
            $rol!($xor(s[17], d[2]), 15),
            $rol!($xor(s[23], d[3]), 56),
            $rol!($xor(s[2], d[2]), 62),
            $rol!($xor(s[8], d[3]), 55),
            $rol!($xor(s[14], d[4]), 39),
            $rol!($xor(s[15], d[0]), 41),
            $rol!($xor(s[21], d[1]), 2),
        ];
        for y in 0..5 {
            for x in 0..5 {
                let l = x + 5 * y;
                let p1 = p[(x + 1) % 5 + 5 * y];
                let p2 = p[(x + 2) % 5 + 5 * y];
                let t = $andn(p1, p2);
                $emit(l, $not(p1), p2, t);
                s[l] = $xor(p[l], t);
            }
        }
        s[0] = $xor(s[0], $rc);
    }};
}

/// The `phi` lane that `b` takes at lane `l = x + 5 y`: `(x + 2) mod 5 + 5 y`.
#[inline(always)]
const fn b_source(l: usize) -> usize {
    (l % 5 + 2) % 5 + 5 * (l / 5)
}

/// Checks the shapes of a `phi_rounds` call; returns the number of rounds.
pub(crate) fn check_phi(first: usize, phi: &[Words8], z: &[Words8]) -> usize {
    assert!(phi.len().is_multiple_of(25), "whole rounds of 25 words");
    assert!(z.len() == 25, "one round of chi outputs");
    let n = phi.len() / 25;
    assert!(first + n <= 24, "rounds within Keccak-f[1600]");
    n
}

/// Checks the shapes of a `chi_rounds` call; returns the number of rounds.
pub(crate) fn check_rounds(first: usize, z: &[Words8], a: &[Words8], b: &[Words8]) -> usize {
    assert!(z.len().is_multiple_of(25), "whole rounds of 25 words");
    assert!(
        a.len() == z.len() && b.len() == z.len(),
        "equal output lengths"
    );
    let n = z.len() / 25;
    assert!(first + n <= 24, "rounds within Keccak-f[1600]");
    n
}

/// Views round `k` of an output as 25 words.
#[inline(always)]
fn round_rows(out: &mut [Words8], k: usize) -> &mut [Words8; 25] {
    (&mut out[25 * k..25 * k + 25])
        .try_into()
        .expect("25 words")
}

pub(crate) mod scalar {
    use super::{b_source, check_phi, check_rounds, round_rows, Words8, RC};

    #[inline(always)]
    fn xor(x: u64, y: u64) -> u64 {
        x ^ y
    }
    #[inline(always)]
    fn andn(x: u64, y: u64) -> u64 {
        !x & y
    }
    #[inline(always)]
    fn not(x: u64) -> u64 {
        !x
    }
    macro_rules! rol64 {
        ($v:expr, $n:literal) => {
            $v.rotate_left($n)
        };
    }

    /// One state at a time (W = 1), round by round over the eight states, so a
    /// round's 4.8 KiB of outputs stay in L1 until all eight states have written
    /// their words (state-major order would write each output line eight times
    /// from L2).
    pub(crate) fn chi_rounds(
        state: &mut [Words8; 25],
        first: usize,
        z: &mut [Words8],
        a: &mut [Words8],
        b: &mut [Words8],
    ) {
        let n = check_rounds(first, z, a, b);
        let mut states: [[u64; 25]; 8] =
            core::array::from_fn(|i| core::array::from_fn(|w| state[w][i]));
        for k in 0..n {
            let (zr, ar, br) = (round_rows(z, k), round_rows(a, k), round_rows(b, k));
            for (i, s) in states.iter_mut().enumerate() {
                let mut emit = |l: usize, av: u64, bv: u64, tv: u64| {
                    zr[l][i] = tv;
                    ar[l][i] = av;
                    br[l][i] = bv;
                };
                keccak_round!(s, RC[first + k], xor, andn, not, rol64, emit);
            }
        }
        for (i, s) in states.iter().enumerate() {
            for (w, word) in s.iter().enumerate() {
                state[w][i] = *word;
            }
        }
    }

    /// See [`crate::Engine::phi_rounds`]; one state at a time, round-major.
    pub(crate) fn phi_rounds(
        state: &mut [Words8; 25],
        first: usize,
        phi: &mut [Words8],
        z: &mut [Words8],
        mut each: impl FnMut(usize, &[Words8; 25]),
    ) {
        let n = check_phi(first, phi, z);
        let zr: &mut [Words8; 25] = z.try_into().expect("25 words");
        let mut states: [[u64; 25]; 8] =
            core::array::from_fn(|i| core::array::from_fn(|w| state[w][i]));
        for k in 0..n {
            let pr = round_rows(phi, k);
            for (i, s) in states.iter_mut().enumerate() {
                let mut emit = |l: usize, _: u64, bv: u64, tv: u64| {
                    zr[l][i] = tv;
                    pr[b_source(l)][i] = bv;
                };
                keccak_round!(s, RC[first + k], xor, andn, not, rol64, emit);
            }
            each(k, zr);
        }
        for (i, s) in states.iter().enumerate() {
            for (w, word) in s.iter().enumerate() {
                state[w][i] = *word;
            }
        }
    }

    /// Keccak-f[1600] on each of eight word-major states.
    pub(crate) fn keccak_f(state: &mut [Words8; 25]) {
        let mut states: [[u64; 25]; 8] =
            core::array::from_fn(|i| core::array::from_fn(|w| state[w][i]));
        for s in &mut states {
            for rc in RC {
                let emit = |_: usize, _: u64, _: u64, _: u64| {};
                keccak_round!(s, rc, xor, andn, not, rol64, emit);
            }
        }
        for (i, s) in states.iter().enumerate() {
            for (w, word) in s.iter().enumerate() {
                state[w][i] = *word;
            }
        }
    }

    /// `out[64 u + bit] = sum_r bit(words[u][r], bit) << r`; all-zero words leave their
    /// 64 bytes unchanged.
    pub(crate) fn transpose_bits8(words: &[Words8], out: &mut [u8]) {
        for (v, o) in words.iter().zip(out.as_chunks_mut::<64>().0) {
            if *v == [0u64; 8] {
                continue;
            }
            for k in 0..8 {
                // Byte r of x is byte k of v[r].
                let mut x = 0u64;
                for (r, &w) in v.iter().enumerate() {
                    x |= ((w >> (8 * k)) & 0xff) << (8 * r);
                }
                let t = (x ^ (x >> 7)) & 0x00AA_00AA_00AA_00AA;
                x ^= t ^ (t << 7);
                let t = (x ^ (x >> 14)) & 0x0000_CCCC_0000_CCCC;
                x ^= t ^ (t << 14);
                let t = (x ^ (x >> 28)) & 0x0000_0000_F0F0_F0F0;
                x ^= t ^ (t << 28);
                // Byte c of x: bit r = bit 8k + c of v[r].
                o[8 * k..8 * k + 8].copy_from_slice(&x.to_le_bytes());
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2 {
    use super::{b_source, check_phi, check_rounds, round_rows, Words8, RC};
    use core::arch::x86_64::*;

    /// Half `h` (states `4 h .. 4 h + 4`) of a word.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn load_half(word: &Words8, h: usize) -> __m256i {
        let half: &[u64; 4] = word[4 * h..4 * h + 4].try_into().expect("four words");
        // SAFETY: The complete vector is contained in the four-word array.
        unsafe { _mm256_loadu_si256(half.as_ptr().cast()) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn store_half(word: &mut Words8, h: usize, v: __m256i) {
        let half: &mut [u64; 4] = (&mut word[4 * h..4 * h + 4])
            .try_into()
            .expect("four words");
        // SAFETY: The complete vector is contained in the four-word array.
        unsafe { _mm256_storeu_si256(half.as_mut_ptr().cast(), v) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn xor(x: __m256i, y: __m256i) -> __m256i {
        _mm256_xor_si256(x, y)
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    fn andn(x: __m256i, y: __m256i) -> __m256i {
        _mm256_andnot_si256(x, y)
    }
    macro_rules! rol256 {
        ($v:expr, $n:literal) => {{
            let v = $v;
            _mm256_or_si256(
                _mm256_slli_epi64::<$n>(v),
                _mm256_srli_epi64::<{ 64 - $n }>(v),
            )
        }};
    }

    /// Two halves of four states (W = 4), alternating per round so both halves of
    /// an output line are written while it is in L1. Caller must have detected `avx2`.
    #[target_feature(enable = "avx2")]
    pub(crate) fn chi_rounds(
        state: &mut [Words8; 25],
        first: usize,
        z: &mut [Words8],
        a: &mut [Words8],
        b: &mut [Words8],
    ) {
        let n = check_rounds(first, z, a, b);
        let ones = _mm256_set1_epi64x(-1);
        let not = |x: __m256i| _mm256_xor_si256(x, ones);
        let mut halves: [[__m256i; 25]; 2] =
            core::array::from_fn(|h| core::array::from_fn(|w| load_half(&state[w], h)));
        for k in 0..n {
            let (zr, ar, br) = (round_rows(z, k), round_rows(a, k), round_rows(b, k));
            let rc = _mm256_set1_epi64x(RC[first + k] as i64);
            for (h, s) in halves.iter_mut().enumerate() {
                let mut emit = |l: usize, av: __m256i, bv: __m256i, tv: __m256i| {
                    store_half(&mut zr[l], h, tv);
                    store_half(&mut ar[l], h, av);
                    store_half(&mut br[l], h, bv);
                };
                keccak_round!(s, rc, xor, andn, not, rol256, emit);
            }
        }
        for (h, s) in halves.iter().enumerate() {
            for (w, v) in s.iter().enumerate() {
                store_half(&mut state[w], h, *v);
            }
        }
    }

    /// See [`crate::Engine::phi_rounds`]; halves alternate per round.
    #[target_feature(enable = "avx2")]
    pub(crate) fn phi_rounds(
        state: &mut [Words8; 25],
        first: usize,
        phi: &mut [Words8],
        z: &mut [Words8],
        mut each: impl FnMut(usize, &[Words8; 25]),
    ) {
        let n = check_phi(first, phi, z);
        let zr: &mut [Words8; 25] = z.try_into().expect("25 words");
        let ones = _mm256_set1_epi64x(-1);
        let not = |x: __m256i| _mm256_xor_si256(x, ones);
        let mut halves: [[__m256i; 25]; 2] =
            core::array::from_fn(|h| core::array::from_fn(|w| load_half(&state[w], h)));
        for k in 0..n {
            let pr = round_rows(phi, k);
            let rc = _mm256_set1_epi64x(RC[first + k] as i64);
            for (h, s) in halves.iter_mut().enumerate() {
                let mut emit = |l: usize, _: __m256i, bv: __m256i, tv: __m256i| {
                    store_half(&mut zr[l], h, tv);
                    store_half(&mut pr[b_source(l)], h, bv);
                };
                keccak_round!(s, rc, xor, andn, not, rol256, emit);
            }
            each(k, zr);
        }
        for (h, s) in halves.iter().enumerate() {
            for (w, v) in s.iter().enumerate() {
                store_half(&mut state[w], h, *v);
            }
        }
    }

    /// Keccak-f[1600] on eight word-major states, four at a time.
    #[target_feature(enable = "avx2")]
    pub(crate) fn keccak_f(state: &mut [Words8; 25]) {
        let ones = _mm256_set1_epi64x(-1);
        let not = |x: __m256i| _mm256_xor_si256(x, ones);
        for h in 0..2 {
            let mut s: [__m256i; 25] = core::array::from_fn(|w| load_half(&state[w], h));
            for r in RC {
                let rc = _mm256_set1_epi64x(r as i64);
                let emit = |_: usize, _: __m256i, _: __m256i, _: __m256i| {};
                keccak_round!(&mut s, rc, xor, andn, not, rol256, emit);
            }
            for (w, v) in s.iter().enumerate() {
                store_half(&mut state[w], h, *v);
            }
        }
    }

    /// One delta swap `x ^= t ^ (t << S)`, `t = (x ^ (x >> S)) & mask`, per word.
    macro_rules! delta256 {
        ($x:ident, $s:literal, $m:expr) => {{
            let t = _mm256_and_si256(_mm256_xor_si256($x, _mm256_srli_epi64::<$s>($x)), $m);
            $x = _mm256_xor_si256($x, _mm256_xor_si256(t, _mm256_slli_epi64::<$s>(t)));
        }};
    }

    /// See the scalar version. Caller must have detected `avx2`.
    #[target_feature(enable = "avx2")]
    pub(crate) fn transpose_bits8(words: &[Words8], out: &mut [u8]) {
        // Within each 128-bit lane: bytes (r, k), (r + 1, k) side by side for k = 0..8.
        let pair = _mm256_setr_epi8(
            0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15, 0, 8, 1, 9, 2, 10, 3, 11, 4, 12,
            5, 13, 6, 14, 7, 15,
        );
        let m7 = _mm256_set1_epi64x(0x00AA_00AA_00AA_00AA);
        let m14 = _mm256_set1_epi64x(0x0000_CCCC_0000_CCCC);
        let m28 = _mm256_set1_epi64x(0x0000_0000_F0F0_F0F0);
        for (v, o) in words.iter().zip(out.as_chunks_mut::<64>().0) {
            // SAFETY: Each 16-byte load lies inside the eight-word array.
            let (r01, r23, r45, r67) = unsafe {
                (
                    _mm_loadu_si128(v.as_ptr().cast()),
                    _mm_loadu_si128(v.as_ptr().add(2).cast()),
                    _mm_loadu_si128(v.as_ptr().add(4).cast()),
                    _mm_loadu_si128(v.as_ptr().add(6).cast()),
                )
            };
            let any = _mm_or_si128(_mm_or_si128(r01, r23), _mm_or_si128(r45, r67));
            if _mm_testz_si128(any, any) == 1 {
                continue;
            }
            // [rows 0, 1 | rows 4, 5] and [rows 2, 3 | rows 6, 7] as 16-bit pairs per byte k.
            let a = _mm256_shuffle_epi8(_mm256_set_m128i(r45, r01), pair);
            let b = _mm256_shuffle_epi8(_mm256_set_m128i(r67, r23), pair);
            // 32-bit units of rows 0..4 (low lane) and rows 4..8 (high lane): k = 0..4, 4..8.
            let lo = _mm256_unpacklo_epi16(a, b);
            let hi = _mm256_unpackhi_epi16(a, b);
            let e = _mm256_permute2x128_si256::<0x20>(lo, hi);
            let f = _mm256_permute2x128_si256::<0x31>(lo, hi);
            // Bytes r = 0..8 of byte k: g = [X0 X1 X4 X5], h = [X2 X3 X6 X7].
            let mut g = _mm256_unpacklo_epi32(e, f);
            let mut h = _mm256_unpackhi_epi32(e, f);
            delta256!(g, 7, m7);
            delta256!(h, 7, m7);
            delta256!(g, 14, m14);
            delta256!(h, 14, m14);
            delta256!(g, 28, m28);
            delta256!(h, 28, m28);
            // SAFETY: Each 16-byte store lies inside the 64-byte array.
            unsafe {
                let p = o.as_mut_ptr();
                _mm_storeu_si128(p.cast(), _mm256_castsi256_si128(g));
                _mm_storeu_si128(p.add(16).cast(), _mm256_castsi256_si128(h));
                _mm_storeu_si128(p.add(32).cast(), _mm256_extracti128_si256::<1>(g));
                _mm_storeu_si128(p.add(48).cast(), _mm256_extracti128_si256::<1>(h));
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx512 {
    use super::{b_source, check_phi, check_rounds, round_rows, Words8, RC};
    use core::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx512f")]
    fn load(word: &Words8) -> __m512i {
        // SAFETY: The complete vector is contained in the eight-word array.
        unsafe { _mm512_loadu_si512(word.as_ptr().cast()) }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    fn store(word: &mut Words8, v: __m512i) {
        // SAFETY: The complete vector is contained in the eight-word array.
        unsafe { _mm512_storeu_si512(word.as_mut_ptr().cast(), v) }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    fn xor(x: __m512i, y: __m512i) -> __m512i {
        _mm512_xor_si512(x, y)
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    fn andn(x: __m512i, y: __m512i) -> __m512i {
        _mm512_andnot_si512(x, y)
    }
    #[inline]
    #[target_feature(enable = "avx512f")]
    fn not(x: __m512i) -> __m512i {
        _mm512_ternarylogic_epi64::<0x55>(x, x, x)
    }
    macro_rules! rol512 {
        ($v:expr, $n:literal) => {
            _mm512_rol_epi64::<$n>($v)
        };
    }

    /// All eight states in lockstep (W = 8). Caller must have detected `avx512f`.
    #[target_feature(enable = "avx512f")]
    pub(crate) fn chi_rounds(
        state: &mut [Words8; 25],
        first: usize,
        z: &mut [Words8],
        a: &mut [Words8],
        b: &mut [Words8],
    ) {
        let n = check_rounds(first, z, a, b);
        let mut s: [__m512i; 25] = core::array::from_fn(|w| load(&state[w]));
        for k in 0..n {
            let (zr, ar, br) = (round_rows(z, k), round_rows(a, k), round_rows(b, k));
            let mut emit = |l: usize, av: __m512i, bv: __m512i, tv: __m512i| {
                store(&mut zr[l], tv);
                store(&mut ar[l], av);
                store(&mut br[l], bv);
            };
            let rc = _mm512_set1_epi64(RC[first + k] as i64);
            keccak_round!(&mut s, rc, xor, andn, not, rol512, emit);
        }
        for (w, v) in s.iter().enumerate() {
            store(&mut state[w], *v);
        }
    }

    /// See [`crate::Engine::phi_rounds`]; all eight states in lockstep.
    #[target_feature(enable = "avx512f")]
    pub(crate) fn phi_rounds(
        state: &mut [Words8; 25],
        first: usize,
        phi: &mut [Words8],
        z: &mut [Words8],
        mut each: impl FnMut(usize, &[Words8; 25]),
    ) {
        let n = check_phi(first, phi, z);
        let zr: &mut [Words8; 25] = z.try_into().expect("25 words");
        let mut s: [__m512i; 25] = core::array::from_fn(|w| load(&state[w]));
        for k in 0..n {
            let pr = round_rows(phi, k);
            let mut emit = |l: usize, _: __m512i, bv: __m512i, tv: __m512i| {
                store(&mut zr[l], tv);
                store(&mut pr[b_source(l)], bv);
            };
            let rc = _mm512_set1_epi64(RC[first + k] as i64);
            keccak_round!(&mut s, rc, xor, andn, not, rol512, emit);
            each(k, zr);
        }
        for (w, v) in s.iter().enumerate() {
            store(&mut state[w], *v);
        }
    }

    /// Keccak-f[1600] on eight word-major states in lockstep.
    #[target_feature(enable = "avx512f")]
    pub(crate) fn keccak_f(state: &mut [Words8; 25]) {
        let mut s: [__m512i; 25] = core::array::from_fn(|w| load(&state[w]));
        for r in RC {
            let rc = _mm512_set1_epi64(r as i64);
            let emit = |_: usize, _: __m512i, _: __m512i, _: __m512i| {};
            keccak_round!(&mut s, rc, xor, andn, not, rol512, emit);
        }
        for (w, v) in s.iter().enumerate() {
            store(&mut state[w], *v);
        }
    }

    /// One delta swap per word: `t = (x ^ (x >> S)) & mask` (ternary 0x28), then
    /// `x ^ t ^ (t << S)` (ternary 0x96).
    macro_rules! delta512 {
        ($x:ident, $s:literal, $m:expr) => {{
            let t = _mm512_ternarylogic_epi64::<0x28>($x, _mm512_srli_epi64::<$s>($x), $m);
            $x = _mm512_ternarylogic_epi64::<0x96>($x, t, _mm512_slli_epi64::<$s>(t));
        }};
    }

    /// See the scalar version. Caller must have detected `avx512f` and `avx512bw`.
    #[target_feature(enable = "avx512f,avx512bw")]
    pub(crate) fn transpose_bits8(words: &[Words8], out: &mut [u8]) {
        // Within each 128-bit lane: bytes (r, k), (r + 1, k) side by side for k = 0..8.
        let pair = _mm512_broadcast_i32x4(_mm_setr_epi8(
            0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15,
        ));
        // 16-bit unit (pair p, byte k) sits at 8 p + k; word k collects pairs 0..4.
        let gather = _mm512_set_epi16(
            31, 23, 15, 7, 30, 22, 14, 6, 29, 21, 13, 5, 28, 20, 12, 4, 27, 19, 11, 3, 26, 18, 10,
            2, 25, 17, 9, 1, 24, 16, 8, 0,
        );
        let m7 = _mm512_set1_epi64(0x00AA_00AA_00AA_00AA);
        let m14 = _mm512_set1_epi64(0x0000_CCCC_0000_CCCC);
        let m28 = _mm512_set1_epi64(0x0000_0000_F0F0_F0F0);
        for (v, o) in words.iter().zip(out.as_chunks_mut::<64>().0) {
            let x = load(v);
            if _mm512_test_epi64_mask(x, x) == 0 {
                continue;
            }
            let mut x = _mm512_permutexvar_epi16(gather, _mm512_shuffle_epi8(x, pair));
            delta512!(x, 7, m7);
            delta512!(x, 14, m14);
            delta512!(x, 28, m28);
            // SAFETY: The 64-byte store lies inside the 64-byte array.
            unsafe { _mm512_storeu_si512(o.as_mut_ptr().cast(), x) }
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) mod avx2 {
    use super::Words8;
    pub(crate) fn chi_rounds(
        _: &mut [Words8; 25],
        _: usize,
        _: &mut [Words8],
        _: &mut [Words8],
        _: &mut [Words8],
    ) {
        unreachable!("AVX2 is x86-64 only")
    }
    pub(crate) fn keccak_f(_: &mut [Words8; 25]) {
        unreachable!("AVX2 is x86-64 only")
    }
    pub(crate) fn phi_rounds(
        _: &mut [Words8; 25],
        _: usize,
        _: &mut [Words8],
        _: &mut [Words8],
        _: impl FnMut(usize, &[Words8; 25]),
    ) {
        unreachable!("AVX2 is x86-64 only")
    }
    pub(crate) fn transpose_bits8(_: &[Words8], _: &mut [u8]) {
        unreachable!("AVX2 is x86-64 only")
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) mod avx512 {
    pub(crate) use super::avx2::{chi_rounds, keccak_f, phi_rounds, transpose_bits8};
}

#[cfg(test)]
mod tests {
    use super::{Words8, RC};
    use crate::dispatch::{Backend, Engine};

    /// FIPS 202 Table 2, indexed `x + 5 y`.
    const ROT: [u32; 25] = [
        0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56,
        14,
    ];

    fn engines() -> Vec<Engine> {
        [Backend::Scalar, Backend::Avx2, Backend::Avx512]
            .into_iter()
            .filter_map(Engine::new)
            .collect()
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    /// Textbook phi = pi o rho o theta: `B[y, 2 x + 3 y] = rot(A'[x, y], r[x, y])`.
    fn phi(s: &[u64; 25]) -> [u64; 25] {
        let mut a = *s;
        let c: [u64; 5] = core::array::from_fn(|x| (0..5).fold(0, |acc, y| acc ^ a[x + 5 * y]));
        for x in 0..5 {
            let d = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
            for y in 0..5 {
                a[x + 5 * y] ^= d;
            }
        }
        let mut b = [0u64; 25];
        for x in 0..5 {
            for y in 0..5 {
                b[y + 5 * ((2 * x + 3 * y) % 5)] = a[x + 5 * y].rotate_left(ROT[x + 5 * y]);
            }
        }
        b
    }

    fn random_state(rng: &mut Rng) -> [Words8; 25] {
        core::array::from_fn(|_| core::array::from_fn(|_| rng.next()))
    }

    #[test]
    fn chi_rounds_match_definition_on_every_backend() {
        let mut rng = Rng(7);
        for (first, n) in [(0usize, 24usize), (0, 1), (5, 3), (23, 1), (10, 0)] {
            let start = random_state(&mut rng);
            for e in engines() {
                let mut st = start;
                let (mut z, mut a, mut b) = (
                    vec![[0u64; 8]; 25 * n],
                    vec![[0u64; 8]; 25 * n],
                    vec![[0u64; 8]; 25 * n],
                );
                e.chi_rounds(&mut st, first, &mut z, &mut a, &mut b);
                for i in 0..8 {
                    let mut s: [u64; 25] = core::array::from_fn(|w| start[w][i]);
                    for k in 0..n {
                        let p = phi(&s);
                        for y in 0..5 {
                            for x in 0..5 {
                                let l = x + 5 * y;
                                let av = !p[(x + 1) % 5 + 5 * y];
                                let bv = p[(x + 2) % 5 + 5 * y];
                                assert_eq!(a[25 * k + l][i], av, "{:?} a", e.backend());
                                assert_eq!(b[25 * k + l][i], bv, "{:?} b", e.backend());
                                assert_eq!(z[25 * k + l][i], av & bv, "{:?} t", e.backend());
                                s[l] = p[l] ^ (av & bv);
                            }
                        }
                        s[0] ^= RC[first + k];
                    }
                    for w in 0..25 {
                        assert_eq!(st[w][i], s[w], "{:?} state", e.backend());
                    }
                }
            }
        }
    }

    #[test]
    fn phi_rounds_match_chi_rounds_on_every_backend() {
        let mut rng = Rng(9);
        for (first, n) in [(0usize, 24usize), (3, 2), (23, 1)] {
            let start = random_state(&mut rng);
            let scalar = Engine::new(Backend::Scalar).expect("scalar");
            let mut want_st = start;
            let (mut z, mut a, mut b) = (
                vec![[0u64; 8]; 25 * n],
                vec![[0u64; 8]; 25 * n],
                vec![[0u64; 8]; 25 * n],
            );
            scalar.chi_rounds(&mut want_st, first, &mut z, &mut a, &mut b);
            for e in engines() {
                let mut st = start;
                let mut phi = vec![[0u64; 8]; 25 * n];
                let mut zt = vec![[0u64; 8]; 25];
                let mut seen = Vec::new();
                e.phi_rounds(&mut st, first, &mut phi, &mut zt, |k, zr| {
                    seen.push(k);
                    assert_eq!(&zr[..], &z[25 * k..25 * k + 25], "{:?} z", e.backend());
                });
                assert_eq!(seen, (0..n).collect::<Vec<_>>());
                assert_eq!(st, want_st, "{:?} state", e.backend());
                for k in 0..n {
                    for y in 0..5 {
                        for x in 0..5 {
                            let l = 25 * k + x + 5 * y;
                            let p1 = 25 * k + (x + 1) % 5 + 5 * y;
                            let p2 = 25 * k + (x + 2) % 5 + 5 * y;
                            for i in 0..8 {
                                assert_eq!(!phi[p1][i], a[l][i], "{:?} a", e.backend());
                                assert_eq!(phi[p2][i], b[l][i], "{:?} b", e.backend());
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn keccak_f8_matches_keccak_crate() {
        let mut rng = Rng(11);
        let start = random_state(&mut rng);
        for e in engines() {
            let mut st = start;
            e.keccak_f8(&mut st);
            for i in 0..8 {
                let mut s: [u64; 25] = core::array::from_fn(|w| start[w][i]);
                keccak::f1600(&mut s);
                for w in 0..25 {
                    assert_eq!(st[w][i], s[w], "{:?}", e.backend());
                }
            }
        }
    }

    #[test]
    fn transpose_bits8_matches_bitwise_on_every_backend() {
        let mut rng = Rng(13);
        let mut words: Vec<Words8> = (0..40)
            .map(|_| core::array::from_fn(|_| rng.next()))
            .collect();
        words[3] = [0; 8];
        words[7] = [0, 0, 0, 0, 0, 0, 0, 1 << 63];
        words[9] = [1, 0, 0, 0, 0, 0, 0, 0];
        for e in engines() {
            let mut out = vec![0xA5u8; 64 * words.len()];
            e.transpose_bits8(&words, &mut out);
            for (u, v) in words.iter().enumerate() {
                for bit in 0..64 {
                    let want = if *v == [0; 8] {
                        0xA5
                    } else {
                        (0..8).fold(0u8, |acc, r| acc | ((((v[r] >> bit) & 1) as u8) << r))
                    };
                    assert_eq!(
                        out[64 * u + bit],
                        want,
                        "{:?} word {u} bit {bit}",
                        e.backend()
                    );
                }
            }
        }
    }
}
