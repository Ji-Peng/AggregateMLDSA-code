//! Digit decompositions in balanced alphabets (spec Section 4.3, (4.1)) and the slack forms of
//! the feature-flagged layouts (spec Appendix A), with the leaf predicates Q_4, Q_8, Q_16.
//!
//! A digit of base b lies in the balanced alphabet [-b/2, b/2 - 1]. A value with digits
//! Z_0..Z_{m-1} (least significant first) is X = sum_k W_k Z_k + C_X (4.1). Decomposition is the
//! inverse: the unsigned mixed-radix digits of X + eps B/2, each shifted by -b_k/2. Outside the
//! proven range the top digit absorbs the excess and leaves its alphabet, which is exactly what
//! the leaf checks of Section 7 detect.

/// Alphabet of a committed column.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Alphabet {
    /// Balanced digit of the given base (2, 4, 8 or 16): [-b/2, b/2 - 1].
    Balanced(u32),
    /// Boolean {0, 1}.
    Bool,
    /// Sign {-1, 1} (S1+S2 layout).
    Sign,
    /// Ternary {-1, 0, 1} (S1+S2 layout).
    Ternary,
}

impl Alphabet {
    /// Smallest admissible value.
    pub fn min(self) -> i64 {
        match self {
            Alphabet::Balanced(b) => -i64::from(b / 2),
            Alphabet::Bool => 0,
            Alphabet::Sign | Alphabet::Ternary => -1,
        }
    }

    /// Largest admissible value.
    pub fn max(self) -> i64 {
        match self {
            Alphabet::Balanced(b) => i64::from(b / 2) - 1,
            Alphabet::Bool | Alphabet::Sign | Alphabet::Ternary => 1,
        }
    }

    /// Membership.
    pub fn contains(self, v: i64) -> bool {
        match self {
            Alphabet::Sign => v == -1 || v == 1,
            _ => self.min() <= v && v <= self.max(),
        }
    }
}

/// Mixed-radix layout of (4.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DigitLayout {
    /// Short name used in column names and reports.
    pub name: &'static str,
    /// Bases, least significant first.
    pub bases: &'static [u32],
    /// Signed (eps = 1) or unsigned (eps = 0).
    pub signed: bool,
}

impl DigitLayout {
    /// Number of digits.
    pub const fn len(&self) -> usize {
        self.bases.len()
    }

    /// Always false (layouts have at least one digit).
    pub const fn is_empty(&self) -> bool {
        self.bases.is_empty()
    }

    /// Weights W_k = prod_{l<k} b_l.
    pub fn weights(&self) -> Vec<i64> {
        let mut w = Vec::with_capacity(self.len());
        let mut acc = 1i64;
        for &b in self.bases {
            w.push(acc);
            acc *= i64::from(b);
        }
        w
    }

    /// B = prod b_k.
    pub fn product(&self) -> i64 {
        self.bases.iter().map(|&b| i64::from(b)).product()
    }

    /// C_X = sum W_k b_k/2 - eps B/2.
    pub fn constant(&self) -> i64 {
        let s: i64 = self
            .weights()
            .iter()
            .zip(self.bases)
            .map(|(w, &b)| w * i64::from(b / 2))
            .sum();
        s - if self.signed { self.product() / 2 } else { 0 }
    }

    /// Proven range [lo, hi]: [-B/2, B/2 - 1] or [0, B - 1].
    pub fn range(&self) -> (i64, i64) {
        let b = self.product();
        if self.signed {
            (-b / 2, b / 2 - 1)
        } else {
            (0, b - 1)
        }
    }

    /// Alphabet of digit k.
    pub fn alphabet(&self, k: usize) -> Alphabet {
        Alphabet::Balanced(self.bases[k])
    }

    /// Digits of x; the top digit absorbs any excess (then it leaves its alphabet).
    pub fn decompose(&self, x: i64) -> Vec<i64> {
        let mut y = x + if self.signed { self.product() / 2 } else { 0 };
        let m = self.len();
        let mut out = Vec::with_capacity(m);
        for (k, &b) in self.bases.iter().enumerate() {
            let b = i64::from(b);
            if k + 1 < m {
                let d = y.rem_euclid(b);
                out.push(d - b / 2);
                y = (y - d) / b;
            } else {
                out.push(y - b / 2);
            }
        }
        out
    }

    /// X = sum W_k Z_k + C_X (4.1), for any digit values.
    pub fn compose(&self, digits: &[i64]) -> i64 {
        assert_eq!(digits.len(), self.len());
        self.weights()
            .iter()
            .zip(digits)
            .map(|(w, z)| w * z)
            .sum::<i64>()
            + self.constant()
    }

    /// True when every digit lies in its alphabet.
    pub fn in_alphabets(&self, digits: &[i64]) -> bool {
        digits
            .iter()
            .enumerate()
            .all(|(k, &d)| self.alphabet(k).contains(d))
    }
}

/// z in relation H: bases 16,16,16,16,4, signed, range [-2^17, 2^17 - 1].
pub const Z_DIGITS: DigitLayout = DigitLayout {
    name: "z",
    bases: &[16, 16, 16, 16, 4],
    signed: true,
};
/// u in relation H: bases 16,16,16,16,8, signed, range [-2^18, 2^18 - 1].
pub const U_DIGITS: DigitLayout = DigitLayout {
    name: "u",
    bases: &[16, 16, 16, 16, 8],
    signed: true,
};
/// The ring quotient v: bases 16^6, 8, signed, range [-2^26, 2^26 - 1].
pub const V_DIGITS: DigitLayout = DigitLayout {
    name: "v",
    bases: &[16, 16, 16, 16, 16, 16, 8],
    signed: true,
};
/// Level-1 and level-2 carries: bases 16,16,4, unsigned, range [0, 1023].
pub const KAPPA12_DIGITS: DigitLayout = DigitLayout {
    name: "kappa",
    bases: &[16, 16, 4],
    signed: false,
};
/// Level-3 carries: bases 16,16, unsigned, range [0, 255].
pub const KAPPA3_DIGITS: DigitLayout = DigitLayout {
    name: "kappa3",
    bases: &[16, 16],
    signed: false,
};
/// H-exact: b_z in [0, 2^17): bases 16,16,16,16,2, unsigned (C = 100,488).
pub const BZ_DIGITS: DigitLayout = DigitLayout {
    name: "bz",
    bases: &[16, 16, 16, 16, 2],
    signed: false,
};
/// H-exact: b_u in [0, 2^18): bases 16,16,16,16,4, unsigned (C = 166,024).
pub const BU_DIGITS: DigitLayout = DigitLayout {
    name: "bu",
    bases: &[16, 16, 16, 16, 4],
    signed: false,
};
/// S1+S2: b_v in [0, 2^16): bases 16^4, unsigned.
pub const BV_DIGITS: DigitLayout = DigitLayout {
    name: "bvr",
    bases: &[16, 16, 16, 16],
    signed: false,
};
/// S1+S2: b_h in [0, 64): bases 8,8, unsigned.
pub const BH_DIGITS: DigitLayout = DigitLayout {
    name: "bh",
    bases: &[8, 8],
    signed: false,
};

/// Slack form X = w a + b - offset with a in {0, 1}, b in [0, 2^j) given by unsigned digits
/// (spec Appendix A; `improvements-tables.md` Section 2 slack-digit lemma).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlackLayout {
    /// Weight w of the Boolean slack a.
    pub slack_weight: i64,
    /// Digits of b (unsigned, range [0, 2^j)).
    pub digits: DigitLayout,
    /// Offset subtracted at the end.
    pub offset: i64,
}

impl SlackLayout {
    /// The exact range [-offset, w + 2^j - 1 - offset].
    pub fn range(&self) -> (i64, i64) {
        (
            -self.offset,
            self.slack_weight + self.digits.product() - 1 - self.offset,
        )
    }

    /// Canonical decomposition: a = [x + offset >= 2^j], b = x + offset - w a.
    pub fn decompose(&self, x: i64) -> (i64, Vec<i64>) {
        let y = x + self.offset;
        let a = i64::from(y >= self.digits.product());
        (a, self.digits.decompose(y - self.slack_weight * a))
    }

    /// X = w a + b - offset.
    pub fn compose(&self, a: i64, digits: &[i64]) -> i64 {
        self.slack_weight * a + self.digits.compose(digits) - self.offset
    }
}

/// H-exact z: z = 130,915 a_z + b_z - 130,993, exact range [-B_z, B_z].
pub const Z_EXACT: SlackLayout = SlackLayout {
    slack_weight: 130_915,
    digits: BZ_DIGITS,
    offset: 130_993,
};
/// H-exact u: u = 118,787 a_u + b_u - 190,465, exact range [-B_u, B_u].
pub const U_EXACT: SlackLayout = SlackLayout {
    slack_weight: 118_787,
    digits: BU_DIGITS,
    offset: 190_465,
};
/// S1+S2 magnitude v_r = 29,696 a_v + b_v, exact range [0, gamma2 - 1].
pub const VR_EXACT: SlackLayout = SlackLayout {
    slack_weight: 29_696,
    digits: BV_DIGITS,
    offset: 0,
};
/// S1+S2 hint slack u_h = 17 a_h + b_h, exact range [0, omega].
pub const UH_EXACT: SlackLayout = SlackLayout {
    slack_weight: 17,
    digits: BH_DIGITS,
    offset: 0,
};

/// S1+S2 r1 = 4 a + b with a in [0, 8), b in [0, 16): committed as the balanced digits
/// (A, B) = (a - 4, b - 8), so r1 = 4 A + B + 24. Canonical a = min(r1 / 4, 7).
pub fn r1_decompose(r1: i64) -> (i64, i64) {
    let a = (r1.div_euclid(4)).clamp(0, 7);
    let b = r1 - 4 * a;
    (a - 4, b - 8)
}

/// r1 = 4 A + B + 24.
pub fn r1_compose(a_digit: i64, b_digit: i64) -> i64 {
    4 * a_digit + b_digit + 24
}

/// P_Z = Z (Z + 1) (spec Section 4.3).
#[inline]
pub fn p_of(z: i64) -> i128 {
    i128::from(z) * (i128::from(z) + 1)
}

fn product_of_shifts(p: i128, roots: &[i128]) -> Option<i128> {
    roots
        .iter()
        .try_fold(1i128, |acc, &t| acc.checked_mul(p - t))
}

/// Q_4(P) = P (P - 2); `None` on i128 overflow (then the value is certainly nonzero).
pub fn q4(p: i128) -> Option<i128> {
    product_of_shifts(p, &[0, 2])
}

/// Q_8(P) = P (P - 2)(P - 6)(P - 12).
pub fn q8(p: i128) -> Option<i128> {
    product_of_shifts(p, &[0, 2, 6, 12])
}

/// Q_16^lo = Q_8.
pub fn q16_lo(p: i128) -> Option<i128> {
    q8(p)
}

/// Q_16^hi(P) = (P - 20)(P - 30)(P - 42)(P - 56).
pub fn q16_hi(p: i128) -> Option<i128> {
    product_of_shifts(p, &[20, 30, 42, 56])
}

/// Q_2(P) = P: Z (Z + 1) = 0 exactly on the base-2 alphabet {-1, 0}.
pub fn q2(p: i128) -> Option<i128> {
    Some(p)
}

/// True when the leaf predicate of `base` vanishes at P (exactly the balanced alphabet).
pub fn leaf_vanishes(base: u32, p: i128) -> bool {
    let v = match base {
        2 => q2(p),
        4 => q4(p),
        8 => q8(p),
        16 => match (q16_lo(p), q16_hi(p)) {
            (Some(a), Some(b)) => a.checked_mul(b),
            _ => None,
        },
        _ => panic!("unsupported digit base {base}"),
    };
    v == Some(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::*;

    fn enumerate(layout: &DigitLayout) -> Vec<i64> {
        let mut vals = vec![0i64];
        let mut digits: Vec<Vec<i64>> = vec![vec![]];
        for k in 0..layout.len() {
            let a = layout.alphabet(k);
            let mut next = Vec::new();
            for d in &digits {
                for z in a.min()..=a.max() {
                    let mut e = d.clone();
                    e.push(z);
                    next.push(e);
                }
            }
            digits = next;
        }
        vals.clear();
        for d in &digits {
            vals.push(layout.compose(d));
        }
        vals.sort_unstable();
        vals
    }

    #[test]
    fn constants_of_section_4_3() {
        assert_eq!(Z_DIGITS.constant(), 34_952);
        assert_eq!(U_DIGITS.constant(), 34_952);
        assert_eq!(V_DIGITS.constant(), 8_947_848);
        assert_eq!(KAPPA12_DIGITS.constant(), 648);
        assert_eq!(KAPPA3_DIGITS.constant(), 136);
        assert_eq!(BZ_DIGITS.constant(), 100_488);
        assert_eq!(BU_DIGITS.constant(), 166_024);
        assert_eq!(Z_DIGITS.range(), (-BAR_B_Z, BAR_B_Z - 1));
        assert_eq!(U_DIGITS.range(), (-BAR_B_U, BAR_B_U - 1));
        assert_eq!(V_DIGITS.range(), (-V_HALF_RANGE, V_HALF_RANGE - 1));
        assert_eq!(KAPPA12_DIGITS.range(), (0, 1023));
        assert_eq!(KAPPA3_DIGITS.range(), (0, 255));
        assert_eq!(Z_EXACT.range(), (-B_Z, B_Z));
        assert_eq!(U_EXACT.range(), (-B_U, B_U));
        assert_eq!(VR_EXACT.range(), (0, GAMMA2 - 1));
        assert_eq!(UH_EXACT.range(), (0, OMEGA as i64));
    }

    #[test]
    fn bijections_by_enumeration() {
        // (4.1) is a bijection from the digit alphabets onto the proven range (spec_checks.py item 8).
        for layout in [
            Z_DIGITS,
            U_DIGITS,
            KAPPA12_DIGITS,
            KAPPA3_DIGITS,
            BZ_DIGITS,
            BU_DIGITS,
            BV_DIGITS,
            BH_DIGITS,
        ] {
            let vals = enumerate(&layout);
            let (lo, hi) = layout.range();
            assert_eq!(vals.len() as i64, hi - lo + 1, "{}", layout.name);
            assert!(vals.iter().copied().eq(lo..=hi), "{}", layout.name);
        }
    }

    #[test]
    fn decompose_compose_roundtrip_and_canonical_zero() {
        for layout in [
            Z_DIGITS,
            U_DIGITS,
            V_DIGITS,
            KAPPA12_DIGITS,
            KAPPA3_DIGITS,
            BZ_DIGITS,
            BU_DIGITS,
        ] {
            let (lo, hi) = layout.range();
            let step = ((hi - lo) / 5000).max(1);
            let mut x = lo;
            while x <= hi {
                let d = layout.decompose(x);
                assert!(layout.in_alphabets(&d), "{} {x}", layout.name);
                assert_eq!(layout.compose(&d), x);
                x += step;
            }
            for x in [lo, hi] {
                assert!(layout.in_alphabets(&layout.decompose(x)));
            }
            // Out of range: the top digit leaves its alphabet but composition still inverts.
            for x in [lo - 1, hi + 1, hi + 12345] {
                let d = layout.decompose(x);
                assert!(!layout.in_alphabets(&d));
                assert_eq!(layout.compose(&d), x);
            }
            // Digits of zero: -b_k/2, except 0 for the top digit of a signed value (spec 4.2).
            let z = layout.decompose(0);
            for (k, &b) in layout.bases.iter().enumerate() {
                let expect = if layout.signed && k + 1 == layout.len() {
                    0
                } else {
                    -i64::from(b / 2)
                };
                assert_eq!(z[k], expect);
            }
        }
    }

    #[test]
    fn slack_forms_exhaustive() {
        for s in [Z_EXACT, U_EXACT, VR_EXACT, UH_EXACT] {
            let (lo, hi) = s.range();
            for x in lo..=hi {
                let (a, d) = s.decompose(x);
                assert!(a == 0 || a == 1);
                assert!(s.digits.in_alphabets(&d), "{x}");
                assert_eq!(s.compose(a, &d), x);
            }
            for x in [lo - 1, hi + 1] {
                let (a, d) = s.decompose(x);
                assert!(!s.digits.in_alphabets(&d) || !(a == 0 || a == 1));
                assert_eq!(s.compose(a, &d), x);
            }
        }
        // The slack-digit lemma: {w a + b} covers exactly the range (every value reachable).
        for r1 in 0..M_W {
            let (a, b) = r1_decompose(r1);
            assert!(Alphabet::Balanced(8).contains(a) && Alphabet::Balanced(16).contains(b));
            assert_eq!(r1_compose(a, b), r1);
        }
        let mut image: Vec<i64> = Vec::new();
        for a in -4..4 {
            for b in -8..8 {
                image.push(r1_compose(a, b));
            }
        }
        image.sort_unstable();
        image.dedup();
        assert!(image.iter().copied().eq(0..M_W));
    }

    #[test]
    fn leaf_predicates_exact_on_balanced_alphabets() {
        // spec_checks.py item 9, extended to base 2.
        for base in [2u32, 4, 8, 16] {
            let a = Alphabet::Balanced(base);
            for z in -4 * i64::from(base)..4 * i64::from(base) {
                assert_eq!(
                    leaf_vanishes(base, p_of(z)),
                    a.contains(z),
                    "base {base} z {z}"
                );
            }
            for z in [i64::from(i8::MIN), i64::from(i8::MAX), 1 << 40, -(1 << 40)] {
                assert_eq!(leaf_vanishes(base, p_of(z)), a.contains(z));
            }
        }
        // b16_prod: one of Q16_lo, Q16_hi vanishes exactly on [-8, 7].
        for z in -8..8 {
            let p = p_of(z);
            assert!(q16_lo(p) == Some(0) || q16_hi(p) == Some(0));
        }
    }
}
