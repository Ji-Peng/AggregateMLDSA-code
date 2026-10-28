//! Digit decompositions in balanced alphabets, the value forms the PIOP commits (the step form
//! of `u` in relations H-u and H-exact, the bit-slack form of `z` in relations H-exact and
//! H-FIPS, the slack form of H-FIPS's magnitude `L`), and the leaf predicates Q_2, Q_4, Q_6,
//! Q_8, Q_16.
//!
//! A digit of base b lies in the balanced alphabet [-b/2, b/2 - 1]. A value with digits
//! Z_0..Z_{m-1} (least significant first) is X = sum_k W_k Z_k + C_X. Decomposition is the
//! inverse: the unsigned mixed-radix digits of X + eps B/2, each shifted by -b_k/2. Outside the
//! proven range the top digit absorbs the excess and leaves its alphabet, which is exactly what
//! the leaf checks detect.

use crate::params::{B_Z, SLACK_Z};

/// Alphabet of a committed column.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Alphabet {
    /// Balanced digit of the given base (2, 4, 6, 8 or 16): [-b/2, b/2 - 1].
    Balanced(u32),
    /// Boolean {0, 1}.
    Bool,
    /// Ternary {-1, 0, 1} (relation H-FIPS).
    Ternary,
}

impl Alphabet {
    /// Smallest admissible value.
    pub fn min(self) -> i64 {
        match self {
            Alphabet::Balanced(b) => -i64::from(b / 2),
            Alphabet::Bool => 0,
            Alphabet::Ternary => -1,
        }
    }

    /// Largest admissible value.
    pub fn max(self) -> i64 {
        match self {
            Alphabet::Balanced(b) => i64::from(b / 2) - 1,
            Alphabet::Bool | Alphabet::Ternary => 1,
        }
    }

    /// Membership.
    pub fn contains(self, v: i64) -> bool {
        self.min() <= v && v <= self.max()
    }
}

/// Mixed-radix layout X = sum W_k Z_k + C_X.
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
        let mut out = vec![0; self.len()];
        self.decompose_into(x, &mut out);
        out
    }

    /// [`Self::decompose`] into `out[..self.len()]` (no allocation).
    ///
    /// # Panics
    /// If `out` is shorter than the layout.
    pub fn decompose_into(&self, x: i64, out: &mut [i64]) {
        let mut y = x + if self.signed { self.product() / 2 } else { 0 };
        let m = self.len();
        for (k, (&b, slot)) in self.bases.iter().zip(&mut out[..m]).enumerate() {
            let b = i64::from(b);
            if k + 1 < m {
                let d = y.rem_euclid(b);
                *slot = d - b / 2;
                y = (y - d) / b;
            } else {
                *slot = y - b / 2;
            }
        }
    }

    /// X = sum W_k Z_k + C_X, for any digit values.
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

/// Slack form X = w a + b - offset with a in {0, 1}, b in [0, 2^j) given by unsigned digits.
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
        let mut digits = vec![0; self.digits.len()];
        let a = self.decompose_into(x, &mut digits);
        (a, digits)
    }

    /// [`Self::decompose`] with the digits of b written into `out[..digits.len()]`;
    /// returns a.
    pub fn decompose_into(&self, x: i64, out: &mut [i64]) -> i64 {
        let y = x + self.offset;
        let a = i64::from(y >= self.digits.product());
        self.digits.decompose_into(y - self.slack_weight * a, out);
        a
    }

    /// X = w a + b - offset.
    pub fn compose(&self, a: i64, digits: &[i64]) -> i64 {
        self.slack_weight * a + self.digits.compose(digits) - self.offset
    }
}

/// Step form X = step H + b - offset with one top digit H in [0, c) and b in [0, 2^j) given by
/// unsigned digits (relation H-u, `docs/h-exact.md`). It covers [-offset, step (c - 1) +
/// 2^j - 1 - offset] without gap when step <= 2^j. Committed digits: the low digits of b, then
/// the top digit stored balanced, D = H - c/2 in the base-c alphabet [-c/2, c/2 - 1].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepLayout {
    /// Digits of b (unsigned, range [0, 2^j)).
    pub low: DigitLayout,
    /// Weight of the top digit.
    pub step: i64,
    /// Base c of the top digit (even, so the leaf halving trick applies).
    pub top_base: u32,
    /// Offset subtracted at the end.
    pub offset: i64,
}

impl StepLayout {
    /// Number of committed digits (the low digits and the top digit).
    pub const fn len(&self) -> usize {
        self.low.len() + 1
    }

    /// Always false.
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// The exact range [-offset, step (c - 1) + 2^j - 1 - offset].
    pub fn range(&self) -> (i64, i64) {
        (
            -self.offset,
            self.step * (i64::from(self.top_base) - 1) + self.low.product() - 1 - self.offset,
        )
    }

    /// Alphabet of committed digit k (low digits first, then the top digit).
    pub fn alphabet(&self, k: usize) -> Alphabet {
        if k < self.low.len() {
            self.low.alphabet(k)
        } else {
            Alphabet::Balanced(self.top_base)
        }
    }

    /// Weights of the committed digits in X = sum W_k Z_k + C_X.
    pub fn weights(&self) -> Vec<i64> {
        let mut w = self.low.weights();
        w.push(self.step);
        w
    }

    /// C_X = C_b + step c/2 - offset.
    pub fn constant(&self) -> i64 {
        self.low.constant() + self.step * i64::from(self.top_base / 2) - self.offset
    }

    /// Canonical digits: H = min(floor(y / step), c - 1) clamped at 0, b = y - step H with
    /// y = x + offset. Outside the range a digit leaves its alphabet (the top digit of b when
    /// y is too large or negative) and composition still inverts.
    pub fn decompose(&self, x: i64) -> Vec<i64> {
        let mut out = vec![0; self.len()];
        self.decompose_into(x, &mut out);
        out
    }

    /// [`Self::decompose`] into `out[..self.len()]` (no allocation).
    pub fn decompose_into(&self, x: i64, out: &mut [i64]) {
        let y = x + self.offset;
        let c = i64::from(self.top_base);
        let h = y.div_euclid(self.step).clamp(0, c - 1);
        let m = self.low.len();
        self.low.decompose_into(y - self.step * h, &mut out[..m]);
        out[m] = h - c / 2;
    }

    /// X = sum W_k Z_k + C_X for any digit values.
    pub fn compose(&self, digits: &[i64]) -> i64 {
        assert_eq!(digits.len(), self.len());
        let m = self.low.len();
        self.low.compose(&digits[..m]) + self.step * (digits[m] + i64::from(self.top_base / 2))
            - self.offset
    }

    /// True when every digit lies in its alphabet.
    pub fn in_alphabets(&self, digits: &[i64]) -> bool {
        digits
            .iter()
            .enumerate()
            .all(|(k, &d)| self.alphabet(k).contains(d))
    }
}

/// Bit-slack form X = b + 2^j B_hi + w a - offset with b in [0, 2^j) given by unsigned
/// digits and two Boolean digits B_hi, a (relation H-exact, `docs/h-exact.md`). With
/// w <= 2^(j+1) the four blocks [0, 2^j), [2^j, 2^(j+1)), [w, w + 2^j), [w + 2^j, w + 2^(j+1))
/// cover [0, w + 2^(j+1) - 1] without gap. Committed digits: the low digits of b, then B_hi,
/// then a, both stored as bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitSlackLayout {
    /// Digits of b (unsigned, range [0, 2^j)).
    pub low: DigitLayout,
    /// Weight of the Boolean slack a.
    pub slack_weight: i64,
    /// Offset subtracted at the end.
    pub offset: i64,
}

impl BitSlackLayout {
    /// Number of committed digits (the low digits, B_hi and a).
    pub const fn len(&self) -> usize {
        self.low.len() + 2
    }

    /// Always false.
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// The exact range [-offset, w + 2^(j+1) - 1 - offset].
    pub fn range(&self) -> (i64, i64) {
        (
            -self.offset,
            self.slack_weight + 2 * self.low.product() - 1 - self.offset,
        )
    }

    /// Alphabet of committed digit k (low digits first, then the two bits).
    pub fn alphabet(&self, k: usize) -> Alphabet {
        if k < self.low.len() {
            self.low.alphabet(k)
        } else {
            Alphabet::Bool
        }
    }

    /// Weights of the committed digits in X = sum W_k Z_k + C_X.
    pub fn weights(&self) -> Vec<i64> {
        let mut w = self.low.weights();
        w.push(self.low.product());
        w.push(self.slack_weight);
        w
    }

    /// C_X = C_b - offset.
    pub fn constant(&self) -> i64 {
        self.low.constant() - self.offset
    }

    /// Canonical digits: a = [y >= 2^(j+1)], then B_hi = [y - w a >= 2^j] and b the rest, with
    /// y = x + offset. Outside the range the top digit of b leaves its
    /// alphabet and composition still inverts.
    pub fn decompose(&self, x: i64) -> Vec<i64> {
        let mut out = vec![0; self.len()];
        self.decompose_into(x, &mut out);
        out
    }

    /// [`Self::decompose`] into `out[..self.len()]` (no allocation).
    pub fn decompose_into(&self, x: i64, out: &mut [i64]) {
        let y = x + self.offset;
        let lo = self.low.product();
        let a = i64::from(y >= 2 * lo);
        let rest = y - self.slack_weight * a;
        let hi = rest.div_euclid(lo).clamp(0, 1);
        let m = self.low.len();
        self.low.decompose_into(rest - lo * hi, &mut out[..m]);
        out[m] = hi;
        out[m + 1] = a;
    }

    /// X = sum W_k Z_k + C_X for any digit values.
    pub fn compose(&self, digits: &[i64]) -> i64 {
        assert_eq!(digits.len(), self.len());
        let m = self.low.len();
        self.low.compose(&digits[..m]) + self.low.product() * digits[m]
            + self.slack_weight * digits[m + 1]
            - self.offset
    }

    /// True when every digit lies in its alphabet.
    pub fn in_alphabets(&self, digits: &[i64]) -> bool {
        digits
            .iter()
            .enumerate()
            .all(|(k, &d)| self.alphabet(k).contains(d))
    }
}

/// A value committed through digits: a mixed-radix layout, a step form, a bit-slack
/// form or a slack form. The PIOP (`agg_piop::manifest::DigitValue::layout`) uses it to treat
/// all alike: the value is always `sum W_k Z_k + C_X` over the committed digits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueForm {
    /// Mixed-radix digits.
    Radix(DigitLayout),
    /// Low digits plus one stepped top digit.
    Step(StepLayout),
    /// Low digits plus two Boolean digits.
    BitSlack(BitSlackLayout),
    /// Low digits plus one Boolean slack (committed in that order: the low digits, then `a`;
    /// relation H-FIPS's `L = L_lo + 29,696 L_hi`, [`L_SLACK`]).
    Slack(SlackLayout),
}

impl ValueForm {
    /// Number of committed digits.
    pub const fn len(&self) -> usize {
        match self {
            Self::Radix(l) => l.len(),
            Self::Step(l) => l.len(),
            Self::BitSlack(l) => l.len(),
            Self::Slack(l) => l.digits.len() + 1,
        }
    }

    /// Always false.
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Base of committed digit k (its balanced alphabet is [-b/2, b/2 - 1]); `None` for a
    /// Boolean digit of a bit-slack form.
    pub fn base(&self, k: usize) -> Option<u32> {
        match self {
            Self::Radix(l) => Some(l.bases[k]),
            Self::Step(l) if k < l.low.len() => Some(l.low.bases[k]),
            Self::Step(l) => Some(l.top_base),
            Self::BitSlack(l) if k < l.low.len() => Some(l.low.bases[k]),
            Self::BitSlack(_) => None,
            Self::Slack(l) if k < l.digits.len() => Some(l.digits.bases[k]),
            Self::Slack(_) => None,
        }
    }

    /// Alphabet of committed digit k.
    pub fn alphabet(&self, k: usize) -> Alphabet {
        self.base(k).map_or(Alphabet::Bool, Alphabet::Balanced)
    }

    /// Weights W_k.
    pub fn weights(&self) -> Vec<i64> {
        match self {
            Self::Radix(l) => l.weights(),
            Self::Step(l) => l.weights(),
            Self::BitSlack(l) => l.weights(),
            Self::Slack(l) => {
                let mut w = l.digits.weights();
                w.push(l.slack_weight);
                w
            }
        }
    }

    /// C_X.
    pub fn constant(&self) -> i64 {
        match self {
            Self::Radix(l) => l.constant(),
            Self::Step(l) => l.constant(),
            Self::BitSlack(l) => l.constant(),
            Self::Slack(l) => l.digits.constant() - l.offset,
        }
    }

    /// Proven range [lo, hi].
    pub fn range(&self) -> (i64, i64) {
        match self {
            Self::Radix(l) => l.range(),
            Self::Step(l) => l.range(),
            Self::BitSlack(l) => l.range(),
            Self::Slack(l) => l.range(),
        }
    }

    /// Canonical digits (a digit leaves its alphabet outside the range).
    pub fn decompose(&self, x: i64) -> Vec<i64> {
        let mut out = vec![0; self.len()];
        self.decompose_into(x, &mut out);
        out
    }

    /// [`Self::decompose`] into `out[..self.len()]`, without an allocation (the encoder
    /// calls it for every value of every coefficient).
    ///
    /// # Panics
    /// If `out` is shorter than [`Self::len`].
    pub fn decompose_into(&self, x: i64, out: &mut [i64]) {
        match self {
            Self::Radix(l) => l.decompose_into(x, out),
            Self::Step(l) => l.decompose_into(x, out),
            Self::BitSlack(l) => l.decompose_into(x, out),
            Self::Slack(l) => {
                let m = l.digits.len();
                out[m] = l.decompose_into(x, &mut out[..m]);
            }
        }
    }

    /// X = sum W_k Z_k + C_X.
    pub fn compose(&self, digits: &[i64]) -> i64 {
        match self {
            Self::Radix(l) => l.compose(digits),
            Self::Step(l) => l.compose(digits),
            Self::BitSlack(l) => l.compose(digits),
            Self::Slack(l) => {
                assert_eq!(digits.len(), l.digits.len() + 1);
                let m = l.digits.len();
                l.compose(digits[m], &digits[..m])
            }
        }
    }

    /// True when every digit lies in its alphabet.
    pub fn in_alphabets(&self, digits: &[i64]) -> bool {
        match self {
            Self::Radix(l) => l.in_alphabets(digits),
            Self::Step(l) => l.in_alphabets(digits),
            Self::BitSlack(l) => l.in_alphabets(digits),
            Self::Slack(l) => {
                let m = l.digits.len();
                digits.len() == m + 1
                    && l.digits.in_alphabets(&digits[..m])
                    && Alphabet::Bool.contains(digits[m])
            }
        }
    }
}

impl From<DigitLayout> for ValueForm {
    fn from(l: DigitLayout) -> Self {
        Self::Radix(l)
    }
}

impl From<StepLayout> for ValueForm {
    fn from(l: StepLayout) -> Self {
        Self::Step(l)
    }
}

impl From<BitSlackLayout> for ValueForm {
    fn from(l: BitSlackLayout) -> Self {
        Self::BitSlack(l)
    }
}

impl From<SlackLayout> for ValueForm {
    fn from(l: SlackLayout) -> Self {
        Self::Slack(l)
    }
}

/// Relation H-FIPS's magnitude `L = L_lo + 29,696 L_hi` in `[0, gamma2 - 1]` with `L_lo` in
/// `[0, 2^16)` (one unsigned base-`2^16` digit, stored `L_lo - 2^15`, looked up in `T_16`) and
/// the bit `L_hi`: `29,696 + 2^16 = gamma2` (`docs/h-exact.md`). As a
/// [`ValueForm`]: `L = S_lo + 29,696 L_hi + 2^15`.
pub const L_SLACK: SlackLayout = SlackLayout {
    slack_weight: crate::hfips::L_STEP,
    digits: DigitLayout {
        name: "l",
        bases: &[1 << 16],
        signed: false,
    },
    offset: 0,
};

/// z in relation H-exact with a 16-bit low part: z = B_zlo + 2^16 B_zhi + 130,915 a_z - B_z
/// with B_zlo in [0, 2^16) (one unsigned base-2^16 digit, stored B_zlo - 2^15), B_zhi, a_z in
/// {0, 1}; exact range [-B_z, B_z] since 130,915 + 2^17 = 2 B_z + 1.
pub const Z_BIT_SLACK: BitSlackLayout = BitSlackLayout {
    low: DigitLayout {
        name: "bz",
        bases: &[1 << 16],
        signed: false,
    },
    slack_weight: SLACK_Z,
    offset: B_Z,
};

/// P_Z = Z (Z + 1).
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

/// Q_6(P) = P (P - 2)(P - 6): zero exactly on the base-6 alphabet [-3, 2].
pub fn q6(p: i128) -> Option<i128> {
    product_of_shifts(p, &[0, 2, 6])
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
        6 => q6(p),
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

    /// Unsigned layouts of ranges [0, 1023] and [0, 255].
    const UNSIGNED: DigitLayout = DigitLayout {
        name: "unsigned",
        bases: &[16, 16, 4],
        signed: false,
    };
    const UNSIGNED2: DigitLayout = DigitLayout {
        name: "unsigned2",
        bases: &[16, 16],
        signed: false,
    };
    /// A signed layout (the canonical digits of zero differ in the top digit).
    const SIGNED: DigitLayout = DigitLayout {
        name: "signed",
        bases: &[16, 16, 4],
        signed: true,
    };
    /// Relation H-u's step form of u as the PIOP commits it: one 16-bit low digit.
    const U_STEP16: StepLayout = StepLayout {
        low: DigitLayout {
            name: "u",
            bases: &[1 << 16],
            signed: false,
        },
        step: STEP_U,
        top_base: U_TOP_BASE,
        offset: B_U,
    };

    #[test]
    fn constants_and_ranges() {
        assert_eq!(UNSIGNED.constant(), 648);
        assert_eq!(UNSIGNED2.constant(), 136);
        assert_eq!(SIGNED.constant(), 136);
        assert_eq!(U_STEP16.constant(), 31_540);
        assert_eq!(UNSIGNED.range(), (0, 1023));
        assert_eq!(UNSIGNED2.range(), (0, 255));
        assert_eq!(SIGNED.range(), (-512, 511));
        assert_eq!(U_STEP16.range(), (-BAR_B_U, BAR_B_U));
        assert_eq!(U_STEP16.range(), (-B_U, B_U));
    }

    #[test]
    fn bijections_by_enumeration() {
        // The mixed-radix map is a bijection from the digit alphabets onto the proven range
        // (spec_checks.py item 8).
        for layout in [UNSIGNED, UNSIGNED2, SIGNED] {
            let vals = enumerate(&layout);
            let (lo, hi) = layout.range();
            assert_eq!(vals.len() as i64, hi - lo + 1, "{}", layout.name);
            assert!(vals.iter().copied().eq(lo..=hi), "{}", layout.name);
        }
    }

    #[test]
    fn decompose_compose_roundtrip_and_canonical_zero() {
        let z16 = DigitLayout {
            name: "z16",
            bases: &[1 << 16, 4],
            signed: true,
        };
        for layout in [UNSIGNED, UNSIGNED2, SIGNED, z16] {
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
            // Digits of zero: -b_k/2, except 0 for the top digit of a signed value.
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
    fn u_step_form_is_exact_at_the_slack_boundaries() {
        // 5 * 63,079 + 2^16 = 2 B_u + 1 and 63,079 <= 2^16: the six blocks tile [-B_u, B_u].
        // (Also compile-time checks in params.rs.)
        assert_eq!(5 * STEP_U + (1 << 16), 2 * B_U + 1);
        assert_eq!(U_STEP16.len(), 2);
        assert_eq!(U_STEP16.weights(), vec![1, STEP_U]);
        // Exhaustive: every u in range has canonical digits in their alphabets and recomposes.
        for u in -B_U..=B_U {
            let d = U_STEP16.decompose(u);
            assert!(U_STEP16.in_alphabets(&d), "u = {u}: {d:?}");
            assert_eq!(U_STEP16.compose(&d), u);
        }
        // Boundaries: +-190,465 encode, +-190,466 have no in-alphabet digits.
        assert_eq!(U_STEP16.decompose(B_U), vec![(1 << 15) - 1, 2]);
        assert_eq!(U_STEP16.decompose(-B_U), vec![-(1 << 15), -3]);
        for u in [B_U + 1, -B_U - 1, B_U + 12_345, -(1 << 18)] {
            let d = U_STEP16.decompose(u);
            assert!(!U_STEP16.in_alphabets(&d), "u = {u}");
            assert_eq!(U_STEP16.compose(&d), u);
        }
        // Reachable set of all in-alphabet digit tuples: exactly [-B_u, B_u].
        let mut hit = vec![false; (2 * B_U + 1) as usize];
        for top in -3..=2i64 {
            for b in 0..(1i64 << 16) {
                let u = b + STEP_U * (top + 3) - B_U;
                assert!((-B_U..=B_U).contains(&u));
                hit[(u + B_U) as usize] = true;
            }
        }
        assert!(hit.iter().all(|&h| h));
        // Zero: H_u = 3 (stored 0), B_ulo = 1,228 (stored 1,228 - 2^15).
        assert_eq!(U_STEP16.decompose(0), vec![-31_540, 0]);
    }

    #[test]
    fn z_bit_slack_form_is_exact_at_the_fips_bound() {
        // docs/h-exact.md: z = S_zlo + 65,536 B_zhi + 130,915 a_z - 98,225.
        let form = ValueForm::from(Z_BIT_SLACK);
        assert_eq!(form.len(), 3);
        assert_eq!(form.weights(), vec![1, 1 << 16, SLACK_Z]);
        assert_eq!(form.constant(), -98_225);
        assert_eq!(form.range(), (-B_Z, B_Z));
        assert_eq!(
            (0..3).map(|k| form.alphabet(k)).collect::<Vec<_>>(),
            [Alphabet::Balanced(1 << 16), Alphabet::Bool, Alphabet::Bool]
        );
        assert_eq!((form.base(0), form.base(1), form.base(2)), (Some(1 << 16), None, None));
        // Exhaustive: every z in range has canonical digits in their alphabets and recomposes.
        for z in -B_Z..=B_Z {
            let d = form.decompose(z);
            assert!(form.in_alphabets(&d), "z = {z}: {d:?}");
            assert_eq!(form.compose(&d), z);
            // a_z = [z >= 79] (the rule a = [x + offset >= 2^j]).
            assert_eq!(d[2], i64::from(z >= 79), "z = {z}");
        }
        // Edges and padding of docs/h-exact.md, as (S_zlo, B_zhi, a_z).
        assert_eq!(form.decompose(B_Z), vec![32_767, 1, 1]);
        assert_eq!(form.decompose(-B_Z), vec![-32_768, 0, 0]);
        assert_eq!(form.decompose(0), vec![32_689, 1, 0]);
        // +-130,994 have no in-alphabet canonical digits; 130,994 lands on S_zlo = 2^15.
        assert_eq!(form.decompose(B_Z + 1), vec![32_768, 1, 1]);
        assert_eq!(form.decompose(-B_Z - 1), vec![-32_769, 0, 0]);
        for z in [B_Z + 1, -B_Z - 1, 131_000, 1 << 17, -(1 << 17)] {
            let d = form.decompose(z);
            assert!(!form.in_alphabets(&d), "z = {z}");
            assert_eq!(form.compose(&d), z);
        }
        // Reachable set of all in-alphabet digit tuples: exactly [-B_z, B_z].
        let mut hit = vec![false; (2 * B_Z + 1) as usize];
        for a in 0..2 {
            for hi in 0..2 {
                for s in -(1i64 << 15)..(1 << 15) {
                    let z = form.compose(&[s, hi, a]);
                    assert!((-B_Z..=B_Z).contains(&z), "{s} {hi} {a}");
                    hit[(z + B_Z) as usize] = true;
                }
            }
        }
        assert!(hit.iter().all(|&h| h));
    }

    #[test]
    fn l_slack_form_covers_exactly_zero_to_gamma2_minus_one() {
        let form = ValueForm::from(L_SLACK);
        assert_eq!(form.len(), 2);
        assert_eq!(form.weights(), vec![1, crate::hfips::L_STEP]);
        assert_eq!(form.constant(), 1 << 15);
        assert_eq!(form.range(), (0, GAMMA2 - 1));
        assert_eq!(
            (form.alphabet(0), form.alphabet(1)),
            (Alphabet::Balanced(1 << 16), Alphabet::Bool)
        );
        for l in 0..GAMMA2 {
            let d = form.decompose(l);
            assert!(form.in_alphabets(&d), "L = {l}");
            assert_eq!(form.compose(&d), l);
            let (lo, hi) = crate::hfips::Coeff::split_l(l);
            assert_eq!((d[0] + (1 << 15), d[1]), (lo, hi));
        }
        for l in [-1, GAMMA2, GAMMA2 + 7] {
            let d = form.decompose(l);
            assert!(!form.in_alphabets(&d), "L = {l}");
            assert_eq!(form.compose(&d), l);
        }
        assert_eq!(form.decompose(0), vec![-(1 << 15), 0]);
    }

    #[test]
    fn leaf_predicates_exact_on_balanced_alphabets() {
        // spec_checks.py item 9, extended to base 2.
        for base in [2u32, 4, 6, 8, 16] {
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
        // The base-16 leaf factors as Q16_lo Q16_hi: one factor vanishes on each digit of [-8, 7].
        for z in -8..8 {
            let p = p_of(z);
            assert!(q16_lo(p) == Some(0) || q16_hi(p) == Some(0));
        }
    }
}
