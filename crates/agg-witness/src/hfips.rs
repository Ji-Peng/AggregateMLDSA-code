//! Relation H-FIPS (relation byte 2): the
//! exact FIPS 204 verification relation, hint included.
//!
//! Per coefficient of `w'` the witness is `(L_lo, L_hi, sigma, delta, e)` with
//!
//! ```text
//! L  = L_lo + 29,696 L_hi in [0, gamma2 - 1]     L_lo in [0, 2^16) (looked up), L_hi in {0, 1}
//! s  = (1 + sigma)/2 in {0, 1}
//! r0 = gamma2 s + L + 1 - gamma2 in (-gamma2, gamma2]
//! u  = r0 - 2 gamma2 delta - e
//! ```
//!
//! `(s, L) -> r0` is a bijection onto `(-gamma2, gamma2]` with `s = 1` iff `r0 > 0`
//! (`L = (r0 - 1) mod gamma2`), so `sigma` is the sign test of FIPS 204 Algorithm 40 and `u` is
//! affine in the committed values. (The alternative `r0 = sigma L + (1 + sigma)/2` is the same
//! set of pairs `(sigma, r0)` under another labelling of `L`, but has the product `sigma L`.)
//!
//! and the six zero-checks `sigma^2 - 1`, `delta (delta - sigma)`, `e (e - 1)(e - delta)`,
//! `e (e + 1) w1'`, `e (e - 1)(w1' - 43)`, `L_hi (L_hi - 1)`; per signature the hint-weight
//! identity `sum_p delta_p^2 + u_h = omega = 80` with `u_h in [0, 2^16)` (looked up); `z` by
//! H-exact's bit-slack form (`digits::Z_BIT_SLACK`). This `u` takes the place of H-u's step
//! form in the ring identity, so `v` and the six bits are those of relations H-u and
//! H-exact (the value `u = ctr_q(w' - 2 gamma2 w1')` is the same integer).
//!
//! The constraint `e (e - 1)(e - delta) = 0` (not `e (e - delta) = 0`) is what makes the honest
//! pair `w' = q - gamma2, h = 0` representable with `delta = 0`: its only witness under
//! `e (e - delta) = 0` has `delta = 1`, so a signature of hint weight 80 with such a coefficient
//! would fail the weight identity (`docs/scripts/spec_checks.py`, the exhaustive witness check).
//!
//! Soundness (exhaustive over all 17,236,992 witnesses, `spec_checks.py` and the unit
//! test `every_witness_is_sound`): with `w' = 2 gamma2 w1' + u mod q` and
//! `h* = [HighBits(w') != w1']`, `UseHint(h*, w') = w1'`, `h* <= delta^2` and `|u| <= B_u`. So
//! the extracted `(c~, z, h*)` with the canonical HintBitPack is a FIPS 204 Algorithm 8
//! signature: `||z||_inf <= B_z` (line 13), `wt(h*) <= sum delta^2 <= omega` (HintBitUnpack,
//! Algorithm 21, line 3 of Algorithm 8), and `c~ = H(mu || w1Encode(UseHint(h*, w')))` (line 12)
//! by the proven hash identity. [`HFipsRecord::extract_signature`] does it.

use crate::check::{CheckReport, Identity};
use crate::crt::az_integer;
use crate::derive::{reencode_h, ReencodeError, SigWitness};
use crate::digits::{ValueForm, Z_BIT_SLACK};
use crate::instance::{PublicInstance, PublicRecord};
use crate::keccak::{sponge_trace, w1_bit_index};
use crate::native::{decompose, high_bits, mod_q, use_hint, SmallK, VecK, VecL};
use crate::padding::main_index;
use crate::params::*;

/// Relation byte of H-FIPS (the proof header's flags byte and byte 3 of the instance record).
pub const RELATION_H_FIPS: u8 = 2;
/// Weight of the bit `L_hi`: `L = L_lo + 29,696 L_hi` with `L_lo in [0, 2^16)`. The two blocks
/// `[0, 2^16)` and `[29,696, 29,696 + 2^16)` cover `[0, gamma2 - 1]` exactly.
pub const L_STEP: i64 = 29_696;
/// Table width of the looked-up cells (`T_16`).
pub const TABLE: i64 = 1 << 16;
/// `w1'` of the top value (`m_w - 1 = 43`), the target of the `e = -1` wrap.
pub const W1_TOP: i64 = M_W - 1;

const _: () = assert!(L_STEP + TABLE == GAMMA2);
const _: () = assert!(OMEGA < 1 << 16);

/// The per-coefficient H-FIPS witness. `l_lo` is the looked-up value itself (the PIOP stores
/// `L_lo - 2^15`, the storage shift of the `T_16` lookup).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Coeff {
    /// `L_lo in [0, 2^16)`.
    pub l_lo: i64,
    /// `L_hi in {0, 1}`.
    pub l_hi: i64,
    /// `sigma in {-1, 1}`: the sign of `r0` (`sigma = 1` iff `r0 > 0`).
    pub sigma: i64,
    /// `delta in {0, sigma}`: `delta^2` is the hint bit of an honest witness.
    pub delta: i64,
    /// `e in {-1, 0, 1}`: the wrap correction (`1` at `w1' = 0`, `-1` at `w1' = 43`).
    pub e: i64,
}

impl Coeff {
    /// `L = L_lo + 29,696 L_hi`.
    pub fn l(&self) -> i64 {
        self.l_lo + L_STEP * self.l_hi
    }

    /// The sign bit `s = (1 + sigma)/2`.
    pub fn s(&self) -> i64 {
        (1 + self.sigma).div_euclid(2)
    }

    /// `r0 = gamma2 s + L + 1 - gamma2`.
    pub fn r0(&self) -> i64 {
        GAMMA2 * self.s() + self.l() + 1 - GAMMA2
    }

    /// `u = r0 - 2 gamma2 delta - e`, the value that enters the ring identity.
    pub fn u(&self) -> i64 {
        self.r0() - 2 * GAMMA2 * self.delta - self.e
    }

    /// The contribution `delta^2` to the hint-weight identity.
    pub fn weight(&self) -> i64 {
        self.delta * self.delta
    }

    /// The six zero-check polynomials at this coefficient, in the order `sigma^2 - 1`,
    /// `delta (delta - sigma)`, `e (e - 1)(e - delta)`, `e (e + 1) w1'`, `e (e - 1)(w1' - 43)`,
    /// `L_hi (L_hi - 1)`. A witness is valid iff all six vanish and `L_lo in [0, 2^16)`.
    pub fn zero_checks(&self, w1: i64) -> [i128; 6] {
        let (s, d, e, h) = (
            i128::from(self.sigma),
            i128::from(self.delta),
            i128::from(self.e),
            i128::from(self.l_hi),
        );
        let w1 = i128::from(w1);
        [
            s * s - 1,
            d * (d - s),
            e * (e - 1) * (e - d),
            e * (e + 1) * w1,
            e * (e - 1) * (w1 - i128::from(W1_TOP)),
            h * (h - 1),
        ]
    }

    /// True when the six zero-checks vanish and `L_lo` lies in the table.
    pub fn is_valid(&self, w1: i64) -> bool {
        (0..TABLE).contains(&self.l_lo) && self.zero_checks(w1).iter().all(|&c| c == 0)
    }

    /// The canonical split of `L in [0, gamma2)`: `L_hi = [L >= 2^16]`, `L_lo = L - 29,696 L_hi`.
    pub fn split_l(l: i64) -> (i64, i64) {
        let hi = i64::from(l >= TABLE);
        (l - L_STEP * hi, hi)
    }

    /// The zero coefficient of a padded row (`w' = 0`, `h = 0`): Decompose(0) = (0, 0), so
    /// `sigma = -1`, `L = gamma2 - 1` (`L_lo = 2^16 - 1`, `L_hi = 1`), `delta = e = 0`.
    pub const PADDED: Coeff = Coeff {
        l_lo: TABLE - 1,
        l_hi: 1,
        sigma: -1,
        delta: 0,
        e: 0,
    };
}

/// The canonical (honest) witness of an accepted coefficient: `w'` (any representative) and the
/// hint bit `h` of the signature, with `w1' = UseHint(h, w')` (FIPS 204 Algorithm 40). It has
/// `delta^2 = h`, and `u` equals `ctr_q(w' - 2 gamma2 w1')`.
///
/// * Decompose wrap (Algorithm 36 line 4, `w' mod q in [q - gamma2, q - 1]`, `r1 = 0`): read as
///   `r1 = 44`, `r0' = w' - (q - 1) in (-gamma2, 0]`, so `sigma = -1`,
///   `L = r0' - 1 + gamma2 = w' - (q - gamma2)`; `h = 0` gives `w1' = 0` and `u = r0' - 1`:
///   `(delta, e) = (0, 1)` (the corner `w' = q - gamma2` has `L = 0`); `h = 1` gives
///   `w1' = 43` and `u = r0' + 2 gamma2`: `(delta, e) = (-1, 0)`.
/// * Otherwise `(r1, r0) = Decompose(w')`, `sigma = 1` iff `r0 > 0`,
///   `L = (r0 - 1) mod gamma2` (`r0 - 1` or `r0 - 1 + gamma2`); `h = 0`: `delta = e = 0`;
///   `h = 1`: `delta = sigma`, and `e = sigma` when `w1'` wraps (`r1 = 43, sigma = 1` gives
///   `w1' = 0`; `r1 = 0, sigma = -1` gives `w1' = 43`), else 0.
pub fn honest_coeff(w: i64, h: bool) -> Coeff {
    let wq = mod_q(w);
    let hv = i64::from(h);
    let (sigma, l, delta, e) = if wq >= Q - GAMMA2 {
        (-1, wq - (Q - GAMMA2), -hv, 1 - hv)
    } else {
        let (r1, r0) = decompose(wq);
        let sigma = if r0 > 0 { 1 } else { -1 };
        let l = (r0 - 1).rem_euclid(GAMMA2);
        let wraps = (sigma == 1 && r1 == W1_TOP) || (sigma == -1 && r1 == 0);
        (sigma, l, hv * sigma, if h && wraps { sigma } else { 0 })
    };
    let (l_lo, l_hi) = Coeff::split_l(l);
    Coeff {
        l_lo,
        l_hi,
        sigma,
        delta,
        e,
    }
}

/// Why an H-FIPS witness cannot be derived.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HFipsError {
    /// Internal invariant broken (a bug): the honest rule disagrees with the derived `u`, `w1'`
    /// or the hint, or a constraint fails.
    Invariant(&'static str),
    /// The hint weight exceeds omega (impossible for a decoded signature, Algorithm 21).
    HintWeight,
}

/// The committed values of one record under relation H-FIPS, as integers before the PIOP's
/// storage shifts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HFipsRecord {
    /// `z` by H-exact's bit-slack form: the stored digits `(S_zlo, B_zhi, a_z)` of
    /// `digits::Z_BIT_SLACK` per coefficient (`z = S_zlo + 2^16 B_zhi + 130,915 a_z - 98,225`).
    pub z_digits: Box<[[[i64; 3]; N]; L]>,
    /// The per-coefficient witness, `[rho][j]`.
    pub coeffs: Box<[[Coeff; N]; K]>,
    /// The quotient `v` of the ring identity (as in relation H-u).
    pub v: VecK,
    /// The six w1Encode bits, `bits[t][rho][j]`.
    pub bits: Box<[[[i64; N]; K]; W1_BITS]>,
    /// The hint slack `u_h = omega - sum delta^2`.
    pub u_h: i64,
}

impl HFipsRecord {
    /// The canonical witness of an accepted signature (from [`crate::derive::derive`]).
    pub fn from_witness(w: &SigWitness) -> Result<Self, HFipsError> {
        let form = ValueForm::from(Z_BIT_SLACK);
        let mut z_digits = Box::new([[[0i64; 3]; N]; L]);
        for s in 0..L {
            for j in 0..N {
                let d = form.decompose(w.z[s][j]);
                if !form.in_alphabets(&d) {
                    return Err(HFipsError::Invariant("|z| > B_z"));
                }
                z_digits[s][j] = [d[0], d[1], d[2]];
            }
        }
        let mut coeffs = Box::new([[Coeff::default(); N]; K]);
        let mut bits = Box::new([[[0i64; N]; K]; W1_BITS]);
        let mut weight = 0i64;
        for rho in 0..K {
            for j in 0..N {
                let h = w.h[rho][j] != 0;
                let w1 = i64::from(w.w1[rho][j]);
                let c = honest_coeff(w.w[rho][j], h);
                if use_hint(h, w.w[rho][j]) != w1 {
                    return Err(HFipsError::Invariant("w1' != UseHint(h, w')"));
                }
                if c.u() != w.u[rho][j] {
                    return Err(HFipsError::Invariant("u of the H-FIPS form != ctr_q(w' - 2 gamma2 w1')"));
                }
                if !c.is_valid(w1) {
                    return Err(HFipsError::Invariant("honest coefficient violates a constraint"));
                }
                if c.weight() != i64::from(h) {
                    return Err(HFipsError::Invariant("delta^2 != h"));
                }
                weight += c.weight();
                coeffs[rho][j] = c;
                for t in 0..W1_BITS {
                    bits[t][rho][j] = (w1 >> t) & 1;
                }
            }
        }
        if weight > OMEGA as i64 {
            return Err(HFipsError::HintWeight);
        }
        Ok(Self {
            z_digits,
            coeffs,
            v: w.v,
            bits,
            u_h: OMEGA as i64 - weight,
        })
    }

    /// The canonical encoding of the all-zero witness of a padded row: `z = 0`
    /// (digits `(32,689, 1, 0)`), `w' = 0` with `h = 0` ([`Coeff::PADDED`]), `v = 0`, bits 0,
    /// `u_h = omega`.
    pub fn padded() -> Self {
        let zd = ValueForm::from(Z_BIT_SLACK).decompose(0);
        Self {
            z_digits: Box::new([[[zd[0], zd[1], zd[2]]; N]; L]),
            coeffs: Box::new([[Coeff::PADDED; N]; K]),
            v: [[0; N]; K],
            bits: Box::new([[[0; N]; K]; W1_BITS]),
            u_h: OMEGA as i64,
        }
    }

    /// `z` from its digits.
    pub fn z(&self) -> VecL {
        let form = ValueForm::from(Z_BIT_SLACK);
        let mut z = [[0i64; N]; L];
        for s in 0..L {
            for j in 0..N {
                z[s][j] = form.compose(&self.z_digits[s][j]);
            }
        }
        z
    }

    /// `u` of the H-FIPS form at every coefficient.
    pub fn u(&self) -> VecK {
        let mut u = [[0i64; N]; K];
        for rho in 0..K {
            for j in 0..N {
                u[rho][j] = self.coeffs[rho][j].u();
            }
        }
        u
    }

    /// `w1' = sum_t 2^t b_t` (any bit values).
    pub fn w1(&self) -> VecK {
        let mut w1 = [[0i64; N]; K];
        for rho in 0..K {
            for j in 0..N {
                w1[rho][j] = (0..W1_BITS).map(|t| self.bits[t][rho][j] << t).sum();
            }
        }
        w1
    }

    /// `sum_p delta_p^2`.
    pub fn hint_weight(&self) -> i64 {
        self.coeffs.iter().flatten().map(Coeff::weight).sum()
    }

    /// The extracted hint `h*_p = [HighBits(w'_p) != w1'_p]` with `w' = 2 gamma2 w1' + u mod q`.
    pub fn extracted_hint(&self) -> SmallK {
        let (u, w1) = (self.u(), self.w1());
        let mut h = [[0u8; N]; K];
        for rho in 0..K {
            for j in 0..N {
                let wp = mod_q(2 * GAMMA2 * w1[rho][j] + u[rho][j]);
                h[rho][j] = u8::from(high_bits(wp) != w1[rho][j]);
            }
        }
        h
    }

    /// The extracted FIPS 204 signature `(c~, z, h*)`, sigEncode (Algorithm 26) with the
    /// canonical HintBitPack (Algorithm 20). For a record that passes [`check_record`] it
    /// verifies under Algorithm 8 (`tests/all/hfips.rs`); `reencode_h` re-checks reachability and
    /// the weight.
    pub fn extract_signature(&self, c_tilde: &[u8; C_TILDE_BYTES]) -> Result<Vec<u8>, ReencodeError> {
        let w1 = self.w1();
        if w1.iter().flatten().any(|&x| !(0..M_W).contains(&x)) {
            return Err(ReencodeError::Domain);
        }
        let mut w1s = [[0u8; N]; K];
        for rho in 0..K {
            for j in 0..N {
                w1s[rho][j] = w1[rho][j] as u8;
            }
        }
        reencode_h(c_tilde, &self.z(), &self.u(), &w1s)
    }
}

/// Every identity of relation H-FIPS on one record, over the integers: the lookups (`B_zlo`,
/// `L_lo`, `u_h`, the two 16-bit halves of `v`), the Booleans (`B_zhi`, `a_z`, `L_hi`, `b_t`),
/// `range_w1`, the six zero-checks, the hint-weight identity, the ring identity with the
/// H-FIPS `u`, and, for a live record, the hash identity through the seven-permutation sponge.
/// `public` is `None` for a padded row (`A = T = 0`, no hash). `i` is the record index (reports).
pub fn check_record(rep: &mut CheckReport, public: Option<&PublicRecord>, rec: &HFipsRecord, i: usize) {
    for s in 0..L {
        for j in 0..N {
            let x = main_index(j, s, i);
            let d = rec.z_digits[s][j];
            if !(-(TABLE / 2)..TABLE / 2).contains(&d[0]) {
                rep.fail(Identity::Range16, "bz_lo", x, format!("S_zlo = {}", d[0]));
            }
            for (k, name) in [(1, "bz_hi"), (2, "a_z")] {
                if d[k] * (d[k] - 1) != 0 {
                    rep.fail(Identity::BoolSlack, name, x, format!("value {}", d[k]));
                }
            }
        }
    }
    let w1 = rec.w1();
    let mut weight = 0i64;
    for rho in 0..K {
        for j in 0..N {
            let x = main_index(j, rho, i);
            for t in 0..W1_BITS {
                let b = rec.bits[t][rho][j];
                if b * (b - 1) != 0 {
                    rep.fail(Identity::BoolB, &format!("b_{t}"), x, format!("value {b}"));
                }
            }
            let b = |t: usize| rec.bits[t][rho][j];
            if b(5) * (b(4) + b(3) * b(2)) != 0 {
                rep.fail(Identity::RangeW1, "b", x, format!("w1' = {}", w1[rho][j]));
            }
            let c = rec.coeffs[rho][j];
            if !(0..TABLE).contains(&c.l_lo) {
                rep.fail(Identity::Range16, "l_lo", x, format!("L_lo = {}", c.l_lo));
            }
            let zc = c.zero_checks(w1[rho][j]);
            let ids = [
                (Identity::SignSquare, "sigma"),
                (Identity::DeltaSign, "delta"),
                (Identity::EdgeE, "e"),
                (Identity::WrapLow, "e_wrap0"),
                (Identity::WrapHigh, "e_wrap43"),
                (Identity::BoolSlack, "l_hi"),
            ];
            for (value, (id, name)) in zc.iter().zip(ids) {
                if *value != 0 {
                    rep.fail(id, name, x, format!("{c:?}, w1' = {}", w1[rho][j]));
                }
            }
            let v = rec.v[rho][j];
            if !(-(1i64 << 31)..1i64 << 31).contains(&v) {
                rep.fail(Identity::Range16, "v", x, format!("v = {v} outside [-2^31, 2^31)"));
            }
            weight += c.weight();
        }
    }
    if !(0..TABLE).contains(&rec.u_h) {
        rep.fail(Identity::Range16, "u_h", i, format!("u_h = {}", rec.u_h));
    }
    if weight + rec.u_h != OMEGA as i64 {
        rep.fail(
            Identity::HintWeight,
            "u_h",
            i,
            format!("sum delta^2 = {weight}, u_h = {}", rec.u_h),
        );
    }
    // Ring identity with the H-FIPS u; A = T = 0 on padded rows.
    let z = rec.z();
    let az = public.map(|p| az_integer(&p.key, &z));
    for rho in 0..K {
        for j in 0..N {
            let (azv, t) = match (&az, public) {
                (Some(a), Some(p)) => (a[rho][j], i128::from(p.t[rho][j])),
                _ => (0, 0),
            };
            let delta = azv
                - t
                - i128::from(2 * GAMMA2 * w1[rho][j])
                - i128::from(rec.coeffs[rho][j].u())
                - i128::from(Q) * i128::from(rec.v[rho][j]);
            if delta != 0 {
                rep.fail(
                    Identity::Ring,
                    "ring",
                    main_index(j, rho, i),
                    format!("Delta = {delta} (record {i})"),
                );
            }
        }
    }
    if let Some(p) = public {
        let mut bytes = [0u8; W1_ENCODE_BYTES];
        for rho in 0..K {
            for j in 0..N {
                for t in 0..W1_BITS {
                    let k = w1_bit_index(t, j, rho);
                    bytes[k / 8] |= ((rec.bits[t][rho][j] & 1) as u8) << (k % 8);
                }
            }
        }
        if sponge_trace(&p.mu, &bytes).c_tilde != p.c_tilde {
            rep.fail(
                Identity::CommitmentHash,
                "c_tilde",
                i,
                "SHAKE256(mu || w1Encode(w1')) != c~".into(),
            );
        }
    }
}

/// [`check_record`] over a batch: `records[i]` for every live `i`, then the canonical padded
/// rows up to `N*` (which must satisfy the relation with `A = T = 0`).
pub fn check_batch(inst: &PublicInstance, records: &[HFipsRecord]) -> CheckReport {
    let mut rep = CheckReport::default();
    if records.len() != inst.records.len() {
        rep.fail(
            Identity::Shape,
            "records",
            0,
            format!("{} records for {} live ones", records.len(), inst.records.len()),
        );
        return rep;
    }
    for (i, (p, r)) in inst.records.iter().zip(records).enumerate() {
        check_record(&mut rep, Some(p), r, i);
    }
    let padded = HFipsRecord::padded();
    for i in inst.records.len()..inst.geom.n_star {
        check_record(&mut rep, None, &padded, i);
    }
    rep
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::ctr_q;

    /// All (w1', sigma, delta, e) that satisfy the zero-checks independent of L.
    fn combos() -> Vec<(i64, i64, i64, i64)> {
        let mut out = Vec::new();
        for w1 in 0..M_W {
            for sigma in [-1, 1] {
                for delta in [-1, 0, 1] {
                    for e in [-1, 0, 1] {
                        let c = Coeff {
                            l_lo: 0,
                            l_hi: 0,
                            sigma,
                            delta,
                            e,
                        };
                        if c.is_valid(w1) {
                            out.push((w1, sigma, delta, e));
                        }
                    }
                }
            }
        }
        out
    }

    #[test]
    fn every_witness_is_sound() {
        // The exhaustive check: 181 combinations times gamma2 values of L = 17,236,992
        // witnesses; for each, h* = [HighBits(w') != w1'] reaches w1' and h* <= delta^2.
        let cs = combos();
        assert_eq!(cs.len(), 181);
        let mut count = 0u64;
        for &(w1, sigma, delta, e) in &cs {
            for l in 0..GAMMA2 {
                let (l_lo, l_hi) = Coeff::split_l(l);
                let c = Coeff {
                    l_lo,
                    l_hi,
                    sigma,
                    delta,
                    e,
                };
                assert!(c.is_valid(w1));
                let u = c.u();
                assert!(u.abs() <= B_U, "{c:?}");
                let wp = mod_q(2 * GAMMA2 * w1 + u);
                let hs = high_bits(wp) != w1;
                assert_eq!(use_hint(hs, wp), w1, "{c:?} w1' = {w1}");
                assert!(i64::from(hs) <= c.weight(), "{c:?} w1' = {w1}");
                count += 1;
            }
        }
        assert_eq!(count, 17_236_992);
    }

    #[test]
    fn honest_rule_is_complete_with_weight_h() {
        // Every (w', h) in Z_q x {0, 1}: the canonical witness is valid, has delta^2 = h and
        // reproduces u = ctr_q(w' - 2 gamma2 UseHint(h, w')).
        for w in 0..Q {
            for h in [false, true] {
                let w1 = use_hint(h, w);
                let c = honest_coeff(w, h);
                assert!(c.is_valid(w1), "w' = {w}, h = {h}: {c:?}");
                assert_eq!(c.weight(), i64::from(h), "w' = {w}, h = {h}");
                assert_eq!(c.u(), ctr_q(w - 2 * GAMMA2 * w1), "w' = {w}, h = {h}");
                assert!((0..GAMMA2).contains(&c.l()));
            }
        }
        // The corner: w' = q - gamma2, h = 0 (r0' = -gamma2 + 1, so L = 0).
        assert_eq!(
            honest_coeff(Q - GAMMA2, false),
            Coeff {
                l_lo: 0,
                l_hi: 0,
                sigma: -1,
                delta: 0,
                e: 1
            }
        );
        // (s, L) -> r0 is a bijection onto (-gamma2, gamma2] with s = [r0 > 0].
        let mut seen = vec![false; 2 * GAMMA2 as usize];
        for sigma in [-1, 1] {
            for l in 0..GAMMA2 {
                let (l_lo, l_hi) = Coeff::split_l(l);
                let r0 = Coeff { l_lo, l_hi, sigma, delta: 0, e: 0 }.r0();
                assert!(-GAMMA2 < r0 && r0 <= GAMMA2);
                assert_eq!(sigma == 1, r0 > 0);
                let slot = &mut seen[(r0 + GAMMA2 - 1) as usize];
                assert!(!*slot);
                *slot = true;
            }
        }
        assert!(seen.iter().all(|&s| s));
        assert_eq!(honest_coeff(0, false), Coeff::PADDED);
    }

    #[test]
    fn ideator_constraint_is_incomplete_at_the_corner() {
        // Under e (e - delta) = 0 the pair (q - gamma2, h = 0) (w1' = 0, u = -gamma2) has only
        // witnesses with delta^2 = 1.
        let (w, w1) = (Q - GAMMA2, 0);
        let u = ctr_q(w);
        assert_eq!(u, -GAMMA2);
        let mut best = i64::MAX;
        for sigma in [-1i64, 1] {
            for delta in [-1i64, 0, 1] {
                for e in [-1i64, 0, 1] {
                    if delta * (delta - sigma) != 0 || e * (e - delta) != 0 {
                        continue;
                    }
                    if e * (e + 1) * w1 != 0 || e * (e - 1) * (w1 - W1_TOP) != 0 {
                        continue;
                    }
                    let l = u + 2 * GAMMA2 * delta + e + GAMMA2 - 1 - GAMMA2 * (1 + sigma).div_euclid(2);
                    if (0..GAMMA2).contains(&l) {
                        best = best.min(delta * delta);
                    }
                }
            }
        }
        assert_eq!(best, 1);
    }

    #[test]
    fn l_split_covers_the_range() {
        for l in 0..GAMMA2 {
            let (lo, hi) = Coeff::split_l(l);
            assert!((0..TABLE).contains(&lo) && (hi == 0 || hi == 1));
            assert_eq!(lo + L_STEP * hi, l);
        }
        assert_eq!(Coeff::split_l(GAMMA2 - 1), (TABLE - 1, 1));
    }
}
