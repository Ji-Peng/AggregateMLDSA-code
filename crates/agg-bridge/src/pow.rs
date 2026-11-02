//! Proof-of-work arithmetic of the 128-bit schedule, shared by the PIOP
//! plan (`agg-piop`) and the tie site of a tied part (`agg-keccak`).
//!
//! A site with assigned bound `e` gets the difficulty `g = min{g >= 0 : 2^128 e <= 2^g}`
//! (Aerie) computed with the exact field size, and a nonce of `g + 7` bits when
//! `g > 0` (none when `g = 0`). [`grind_bits_k`] is the rule for the PIOP field
//! `K = F_{p^2}`, `p = 2^64 - 59` (so even `1/|K|` needs one bit, `p^2 < 2^128`);
//! [`grind_bits_f128`] the rule for `GF(2^128)` (so `1/2^128` needs none). The sites of a
//! tied part (tied-part proof version 5: the tie site of
//! the public claim, the `eq_Y` site and one site per parity-chain round) carry the
//! Johnson-list size of the sponge commitment ([`list_size`], [`part_work`]).

/// Security target in bits.
pub const TARGET_BITS: u32 = 128;
/// Nonce bits beyond the difficulty (Aerie, Akita): honest search fails with
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

/// `floor(L_max)`, `L_max = 1 / (2 eta sqrt(rho_0))` (Flock) at `eta = 1/50` and
/// level-0 rate `rho_0 = 2^-rate_log`: `floor(25 * 2^(rate_log / 2))` = 35, 50, 70 for
/// `rate_log` = 1, 2, 3 (Ligerito `Fast`, `Slim`, `Slim8`).
#[must_use]
pub fn list_size(rate_log: u32) -> u64 {
    (625_u64 << rate_log).isqrt()
}

/// The public geometry of one tied part that its tie bound depends on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TieGeometry {
    /// `log2` of the record positions per segment of the part.
    pub d: usize,
    /// `ceil(log2 S)` for a span of `S` segments (0 for one segment).
    pub span_log: usize,
    /// `log2` of the level-1 blocks per lane (`max(0, d - log2 B_1)`: `B_1 = 2^19 / P_pad`
    /// records per level-1 block, 512 for ML-DSA-44 and 256 for ML-DSA-65 and ML-DSA-87).
    pub blocks_log: usize,
    /// Row capacity `log2` of the sponge instance.
    pub nu_s: usize,
    /// Word coordinates of the sponge's public region (`log2` of its width in words): 3 for
    /// ML-DSA-44 and ML-DSA-65, 4 for ML-DSA-87 (mu, a 64-byte `c~` and the constant).
    pub public_vars: usize,
}

/// `log2 m_3`: the output-row coordinates of `eq_Y`.
const OUT_VARS: usize = agg_witness::bridge::M3.trailing_zeros() as usize;
/// `log2` of the chain's input bits per record: `P_pad` positions times the padded lanes
/// `2^lb`, `1024 x 8` for ML-DSA-44 (six lanes, two of weight zero) and `2048 x 4` for
/// ML-DSA-65 and ML-DSA-87, the same `2^13` for every parameter set.
const RECORD_CHAIN_VARS: usize = 13;
/// The assigned numerator of the tie site (over `2^128`) with list size `list` for the
/// sponge commitment: `L (nu_s + c)`, the public claim's free coordinates (`c` =
/// [`TieGeometry::public_vars`]; tied-part proof version 5).
#[must_use]
pub fn tie_numerator(list: u64, geo: &TieGeometry) -> u64 {
    list * (geo.nu_s + geo.public_vars) as u64
}

/// The tie site's difficulty: [`grind_bits_f128`] of [`tie_numerator`].
#[must_use]
pub fn tie_bits(list: u64, geo: &TieGeometry) -> u8 {
    grind_bits_f128(tie_numerator(list, geo))
}

/// The assigned numerator of the `eq_Y` site: `L (log2 m_3 + span_log)` (two distinct
/// multilinear tables of the final vectors' variables agree at a random point with
/// probability at most that over `2^128`, for every sponge list element).
#[must_use]
pub fn eq_numerator(list: u64, geo: &TieGeometry) -> u64 {
    list * (OUT_VARS + geo.span_log) as u64
}

/// The assigned numerator of one round of the parity chain: `2 L` (degree 2).
#[must_use]
pub const fn round_numerator(list: u64) -> u64 {
    2 * list
}

/// Rounds of the parity chain of a production part: the in-block positions (`log2 P_pad + d`
/// up to `log2 J_1 = 19`), the `blocks_log` block coordinates above them, the lane
/// coordinates and `span_log` segment coordinates. The in-block positions and the block
/// coordinates together are the `log2 P_pad + d` position bits of the part's lane, so for
/// every parameter set the count is `13 + d + span_log`.
#[must_use]
pub fn chain_rounds(geo: &TieGeometry) -> usize {
    RECORD_CHAIN_VARS + geo.d + geo.span_log
}

/// The work of one tied part (tied-part proof version 5): the tie site, the `eq_Y` site and
/// the `rounds` chain rounds, each with its difficulty.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PartWork {
    /// Difficulty of the tie site (the public claim's point).
    pub tie_bits: u8,
    /// Difficulty of the `eq_Y` site.
    pub eq_bits: u8,
    /// Difficulty of every chain round.
    pub round_bits: u8,
    /// Chain rounds.
    pub rounds: u16,
}

/// The work of a tied part of geometry `geo` with list size `list`.
#[must_use]
pub fn part_work(list: u64, geo: &TieGeometry) -> PartWork {
    PartWork {
        tie_bits: tie_bits(list, geo),
        eq_bits: grind_bits_f128(eq_numerator(list, geo)),
        round_bits: grind_bits_f128(round_numerator(list)),
        rounds: u16::try_from(chain_rounds(geo)).expect("few rounds"),
    }
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
        // The pinned list sizes, tied-part proof version 5.
        assert_eq!([1, 2, 3].map(list_size), [35, 50, 70]);
        let four = TieGeometry {
            d: 13,
            span_log: 2,
            blocks_log: 4,
            nu_s: 15,
            public_vars: 3,
        };
        assert_eq!(tie_numerator(1, &four), 18);
        assert_eq!(
            [35, 50, 70].map(|l| tie_numerator(l, &four)),
            [630, 900, 1_260]
        );
        assert_eq!([35, 50, 70].map(|l| tie_bits(l, &four)), [10, 10, 11]);
        assert_eq!(
            [35, 50, 70].map(|l| eq_numerator(l, &four)),
            [490, 700, 980]
        );
        assert_eq!(chain_rounds(&four), 28);
        assert_eq!(
            part_work(50, &four),
            PartWork {
                tie_bits: 10,
                eq_bits: 10,
                round_bits: 7,
                rounds: 28
            }
        );
        let one = TieGeometry {
            d: 13,
            span_log: 0,
            blocks_log: 4,
            nu_s: 13,
            public_vars: 3,
        };
        assert_eq!(chain_rounds(&one), 26);
        assert_eq!(
            [35, 50, 70].map(|l| part_work(l, &one)),
            [
                PartWork {
                    tie_bits: 10,
                    eq_bits: 9,
                    round_bits: 7,
                    rounds: 26
                },
                PartWork {
                    tie_bits: 10,
                    eq_bits: 10,
                    round_bits: 7,
                    rounds: 26
                },
                PartWork {
                    tie_bits: 11,
                    eq_bits: 10,
                    round_bits: 8,
                    rounds: 26
                },
            ]
        );
        let small = TieGeometry {
            d: 3,
            span_log: 0,
            blocks_log: 0,
            nu_s: 4,
            public_vars: 3,
        };
        assert_eq!(chain_rounds(&small), 16);
        assert_eq!(tie_numerator(50, &small), 350);
        assert_eq!(part_work(50, &small).tie_bits, 9);
    }
}
