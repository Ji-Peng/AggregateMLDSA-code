//! Proof-of-work arithmetic of the 128-bit schedule (`docs/soundness.md` Sections 1, 3.6
//! and 6), shared by the PIOP plan (`agg-piop`) and the tie site of a tied part
//! (`agg-keccak`).
//!
//! A site with assigned bound `e` gets the difficulty `g = min{g >= 0 : 2^128 e <= 2^g}`
//! (Aerie Eq. (17)) computed with the exact field size, and a nonce of `g + 7` bits when
//! `g > 0` (none when `g = 0`). [`grind_bits_k`] is the rule for the PIOP field
//! `K = F_{p^2}`, `p = 2^64 - 59` (so even `1/|K|` needs one bit, `p^2 < 2^128`);
//! [`grind_bits_f128`] the rule for `GF(2^128)` (so `1/2^128` needs none). The tie site's
//! assigned numerator carries the Johnson-list sizes of the two Ligerito commitments
//! ([`list_size`], [`tie_numerator`]).

/// Security target in bits.
pub const TARGET_BITS: u32 = 128;
/// Nonce bits beyond the difficulty (Aerie C.1, Akita Eq. (146)): honest search fails with
/// probability at most `exp(-2^7)` per site.
pub const NONCE_SLACK_BITS: u8 = 7;
/// Largest difficulty a site may carry (Akita's transition cap; the 32-bit nonce must hold
/// `g + 7` bits).
pub const MAX_GRIND_BITS: u8 = 25;
/// The PIOP base-field modulus `p = 2^64 - 59`.
pub const P: u64 = u64::MAX - 58;

/// `|K| = p^2` (fits: `p^2 < 2^128`).
const K_ORDER: u128 = (P as u128) * (P as u128);

/// Least `g >= 0` with `2^128 n <= 2^g |F|`, given `floor_shift(k) = floor(|F| / 2^k)` for
/// `0 <= k <= 128`: `n 2^(128 - g) <= |F|` iff `n <= floor(|F| / 2^(128 - g))`.
fn least_bits(n: u64, floor_shift: impl Fn(u32) -> u128) -> u8 {
    if n == 0 {
        return 0;
    }
    (0..=TARGET_BITS)
        .find(|&g| u128::from(n) <= floor_shift(TARGET_BITS - g))
        .map(|g| u8::try_from(g).expect("g <= 128"))
        .unwrap_or(u8::MAX)
}

/// Difficulty of a site of `K` with assigned bound `n / |K|` (exact; `bit_length(n)` for
/// `1 <= n < 2^57`, as Aerie's `grind_bits`).
#[must_use]
pub fn grind_bits_k(n: u64) -> u8 {
    // floor(p^2 / 2^128) = 0: even n = 1 needs one bit.
    least_bits(n, |k| if k >= TARGET_BITS { 0 } else { K_ORDER >> k })
}

/// Difficulty of a site of `GF(2^128)` with assigned bound `n / 2^128`: `ceil(log2 n)`.
#[must_use]
pub fn grind_bits_f128(n: u64) -> u8 {
    // floor(2^128 / 2^k) = 2^(128 - k); for k = 0 the value 2^128 exceeds every u64 n.
    least_bits(n, |k| {
        if k == 0 {
            u128::MAX
        } else {
            1_u128 << (TARGET_BITS - k)
        }
    })
}

/// Nonce width of a site of difficulty `g`: `g + 7`, or 0 for `g = 0` (no nonce).
#[must_use]
pub const fn nonce_bits(g: u8) -> u8 {
    if g == 0 {
        0
    } else {
        g + NONCE_SLACK_BITS
    }
}

/// `floor(L_max)`, `L_max = 1 / (2 eta sqrt(rho_0))` (Flock Eq. (10)) at `eta = 1/50` and
/// level-0 rate `rho_0 = 2^-rate_log`: `floor(25 * 2^(rate_log / 2))` = 35, 50, 70 for
/// `rate_log` = 1, 2, 3 (Ligerito `Fast`, `Slim`, `Slim8`).
#[must_use]
pub fn list_size(rate_log: u32) -> u64 {
    (625_u64 << rate_log).isqrt()
}

/// The public geometry of one tied part that its tie bound depends on
/// (`docs/soundness.md` Section 3.6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TieGeometry {
    /// `log2` of the record positions per segment of the part.
    pub d: usize,
    /// `ceil(log2 S)` for a span of `S` segments (0 for one segment).
    pub span_log: usize,
    /// `log2` of the level-1 blocks per lane (`max(0, d - 9)`).
    pub blocks_log: usize,
    /// Row capacity `log2` of the sponge instance.
    pub nu_s: usize,
}

/// Output words of the levels-2-3 row (`log2(m_3 / 128)`).
const OUTPUT_WORDS_LOG2: usize = 5;

/// The assigned numerator of the tie site (over `2^128`) with list size `list` for both
/// Ligerito commitments: `2 L^2 (d + span_log + 5) + 2 L (blocks_log + span_log + 9) +
/// L (5 + span_log) + L (nu_s + 3)` (tie 1 and tie 2 for two lane groups each, the output
/// claim, the public words).
#[must_use]
pub fn tie_numerator(list: u64, geo: &TieGeometry) -> u64 {
    let w = |x: usize| x as u64;
    let tie1 = w(geo.d + geo.span_log + 5);
    let tie2 = w(geo.blocks_log + geo.span_log + 9);
    let out = w(OUTPUT_WORDS_LOG2 + geo.span_log);
    let public = w(geo.nu_s + 3);
    2 * list * list * tie1 + 2 * list * tie2 + list * out + list * public
}

/// The tie site's difficulty: [`grind_bits_f128`] of [`tie_numerator`].
#[must_use]
pub fn tie_bits(list: u64, geo: &TieGeometry) -> u8 {
    grind_bits_f128(tie_numerator(list, geo))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bit_length(n: u64) -> u8 {
        (64 - n.leading_zeros()) as u8
    }

    #[test]
    fn k_rule_is_the_bit_length_and_matches_aerie() {
        for (n, want) in [(1, 1), (3, 2), (8, 4), (16, 5), (25, 5), (44, 6), (0, 0)] {
            assert_eq!(grind_bits_k(n), want, "n = {n}");
        }
        for n in 1..5000_u64 {
            assert_eq!(grind_bits_k(n), bit_length(n));
        }
        // Up to 2^57 the bit length is exact; at 2^64 - 1 the deficit of p^2 shows.
        for n in [(1_u64 << 56) - 1, 1 << 56, (1 << 57) - 1] {
            assert_eq!(grind_bits_k(n), bit_length(n));
        }
        assert_eq!(grind_bits_k(u64::MAX), 65);
        assert_eq!(K_ORDER, u128::MAX - (118_u128 << 64) + 3482);
    }

    #[test]
    fn f128_rule_is_the_ceiling_log() {
        assert_eq!(grind_bits_f128(1), 0);
        assert_eq!(grind_bits_f128(2), 1);
        assert_eq!(grind_bits_f128(95), 7);
        assert_eq!(grind_bits_f128(128), 7);
        assert_eq!(grind_bits_f128(129), 8);
        assert_eq!(grind_bits_f128(102_750), 17);
        assert_eq!(nonce_bits(0), 0);
        assert_eq!(nonce_bits(17), 24);
    }

    #[test]
    fn tie_bounds_match_the_soundness_doc() {
        // docs/soundness.md Sections 3.6 and 10 (script output).
        assert_eq!([1, 2, 3].map(list_size), [35, 50, 70]);
        let four = TieGeometry {
            d: 13,
            span_log: 2,
            blocks_log: 4,
            nu_s: 15,
        };
        // Without the lists: docs/bridge.md 5c's 95.
        assert_eq!(tie_numerator(1, &four), 95);
        assert_eq!(tie_numerator(35, &four), 50_925);
        assert_eq!(tie_numerator(50, &four), 102_750);
        assert_eq!(tie_numerator(70, &four), 199_850);
        assert_eq!([35, 50, 70].map(|l| tie_bits(l, &four)), [16, 17, 18]);
        let one = TieGeometry {
            d: 13,
            span_log: 0,
            blocks_log: 4,
            nu_s: 13,
        };
        assert_eq!(
            [35, 50, 70].map(|l| tie_numerator(l, &one)),
            [45_745, 92_350, 179_690]
        );
        let small = TieGeometry {
            d: 3,
            span_log: 0,
            blocks_log: 0,
            nu_s: 4,
        };
        assert_eq!(tie_numerator(35, &small), 20_650);
        assert_eq!(tie_numerator(50, &small), 41_500);
        assert_eq!(tie_bits(50, &small), 16);
    }
}
