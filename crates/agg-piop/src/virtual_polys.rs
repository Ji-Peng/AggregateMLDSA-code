//! Virtual polynomials (spec Section 5, lookup layout L16).
//!
//! Every virtual table is defined on Boolean points and extended multilinearly as a table.
//! Two kinds of evaluator live here:
//!
//! * affine evaluators, valid at any point because the table is an affine combination of
//!   committed tables: `w_1' = sum_t 2^t b_t`, the compositions (4.1) of `z, u, v` from
//!   their two L16 digits (`X = X_lo + 2^16 X_hi + 2^15` for `z, v`; relation H-u's step form
//!   `u = U_lo + 63,079 U_hi + 31,540`) and of the carries
//!   (`kappa = K + 2^15`), and the high part `2 gamma_2 w_1' + u + q v` of the ring residual.
//!   The verifier forms them from opened values ([`MainValues`], [`CarryValues`]);
//! * Boolean-domain materializations used by the prover and the tests: `w_1'`, the
//!   residual `Delta = (A z) - T - 2 gamma_2 w_1' - u - q v`, the level-1 parities
//!   `y^(1)_t = sum_zeta M_1[rho_1, zeta] b_t - 2 kappa^(1)_t` (spec (7.5); the sums are
//!   `agg_bridge::parity::level1_sums` at the production shape, for any [`SparseMatrix`] of
//!   that shape), and the leaf tables `P_Z = Z (Z + 1)` of the two small digits.
//!
//! `y^(1)_t` is not affine in committed evaluations at a non-Boolean point: its value there
//! is a claim reduced by the `def_y1` sum-check, whose claimed sum is
//! [`parity1_claimed_sum`].

use agg_bridge::{parity::level1_sums, Shape};
use agg_witness::bridge::SparseMatrix;
use agg_witness::crt::az_integer;
use agg_witness::digits::ValueForm;
use agg_witness::instance::PublicInstance;
use agg_witness::padding::main_index;
use agg_witness::params::{GAMMA2, L, N, POSITIONS, Q, W1_BITS};
use jolt_field::{Ring, Zero};
use rayon::prelude::*;

use crate::manifest::{Column, CommitmentManifest, DigitValue, GroupRole, CARRY16_DIGITS};
use crate::source_tables::{Cells, MainTables, SourceTables};
use crate::OpeningField as E;

/// `w_1' = sum_{t < 6} 2^t b_t` from the six bit evaluations (any point).
#[must_use]
pub fn w1_from_bits(bits: &[E; W1_BITS]) -> E {
    bits.iter()
        .rev()
        .fold(E::zero(), |acc, &bit| acc + acc + bit)
}

/// `w_1'` on the Boolean main domain.
#[must_use]
pub fn w1_table(lanes: &[&[i8]; W1_BITS]) -> Vec<i64> {
    (0..lanes[0].len())
        .into_par_iter()
        .map(|x| (0..W1_BITS).map(|t| i64::from(lanes[t][x]) << t).sum())
        .collect()
}

/// (4.1): `X = sum_k W_k Z_k + C_X` from the digit evaluations (any point).
#[must_use]
pub fn compose_digits(layout: &ValueForm, digits: &[E]) -> E {
    assert_eq!(digits.len(), layout.len(), "one evaluation per digit");
    layout
        .weights()
        .into_iter()
        .zip(digits)
        .fold(E::from_i64(layout.constant()), |sum, (weight, &digit)| {
            sum + digit.mul_i64(weight)
        })
}

/// (4.1) on the Boolean domain, over the integers.
#[must_use]
pub fn compose_table(layout: &ValueForm, digits: &[Cells<'_>]) -> Vec<i64> {
    assert_eq!(digits.len(), layout.len(), "one table per digit");
    let weights = layout.weights();
    let constant = layout.constant();
    (0..digits[0].len())
        .into_par_iter()
        .map(|x| {
            digits
                .iter()
                .zip(&weights)
                .fold(constant, |sum, (column, &weight)| sum + weight * column.value(x))
        })
        .collect()
}

/// The high part `2 gamma_2 w_1' + u + q v` of the ring residual (any point): the
/// eq-weighted second sum of the ring instance (7.6).
#[must_use]
pub fn residual_high(w1: E, u: E, v: E) -> E {
    w1.mul_i64(2 * GAMMA2) + u + v.mul_i64(Q)
}

/// `Delta = (A z) - T - 2 gamma_2 w_1' - u - q v` (spec Section 5) from its parts.
#[must_use]
pub fn residual(az: E, t: E, w1: E, u: E, v: E) -> E {
    az - t - residual_high(w1, u, v)
}

/// `Delta` on the Boolean main domain, as a field table. `(A z)` is the exact integer
/// product of `agg-witness` (CRT) for live records, and `A = T = 0` on padded rows; `z, u,
/// v` are recomposed from the committed digits by (4.1), `w_1'` from the bits. Diagnostic and
/// test use only: the prover never materializes `Delta` (stage 3 pulls `(A z)` back through
/// the ring weight).
#[must_use]
pub fn residual_table(
    manifest: &CommitmentManifest,
    instance: &PublicInstance,
    main: &MainTables,
) -> Vec<E> {
    let compose = |value: DigitValue| {
        let digits = [0, 1].map(|digit| main.column(manifest, Column::Digit { value, digit }));
        compose_table(&value.layout(), &digits)
    };
    let (z, u, v) = (compose(DigitValue::Z), compose(DigitValue::U), compose(DigitValue::V));
    let w1 = w1_table(&main.bit_lanes(manifest));
    let geom = main.geometry();
    let mut out = vec![E::zero(); geom.main_len()];
    out.par_chunks_mut(POSITIONS)
        .enumerate()
        .for_each(|(i, chunk)| {
            let base = i * POSITIONS;
            let product = (i < geom.n).then(|| {
                let mut z_record = [[0_i64; N]; L];
                for (s, row) in z_record.iter_mut().enumerate() {
                    for (j, value) in row.iter_mut().enumerate() {
                        *value = z[main_index(j, s, i)];
                    }
                }
                az_integer(&instance.records[i].key, &z_record)
            });
            for (offset, cell) in chunk.iter_mut().enumerate() {
                let (j, rho) = (offset % N, offset / N);
                let (az, t) = match &product {
                    Some(az) => (
                        E::from_i128(az[rho][j]),
                        E::from_i64(instance.records[i].t[rho][j]),
                    ),
                    None => (E::zero(), E::zero()),
                };
                let x = base + offset;
                *cell = residual(
                    az,
                    t,
                    E::from_i64(w1[x]),
                    E::from_i64(u[x]),
                    E::from_i64(v[x]),
                );
            }
        });
    out
}

/// `y^(1)_t = sum_zeta M_1[rho_1, zeta] b_t(zeta, beta) - 2 kappa^(1)_t(rho_1, beta)` on the
/// Boolean domain `D_1`, over the integers, per lane (spec Section 5, (7.5)), with
/// `kappa^(1)_t = K^(1)_t + 2^15` from the committed balanced carries. The honest value is
/// the GF(2) parity of the level-1 row, `{0, 1}`.
#[must_use]
pub fn parity1_tables(tables: SourceTables<'_>, m1: &SparseMatrix) -> [Vec<i64>; W1_BITS] {
    let sums = level1_sums(&Shape::PRODUCTION, &tables.main.bit_lanes(tables.manifest), m1);
    let mut out: [Vec<i64>; W1_BITS] = Default::default();
    for (t, (sum, slot)) in sums.into_iter().zip(out.iter_mut()).enumerate() {
        let kappa = compose_table(&ValueForm::Radix(CARRY16_DIGITS), &[tables.column(Column::carry(1, t as u8))]);
        *slot = sum.iter().zip(&kappa).map(|(s, k)| s - 2 * k).collect();
    }
    out
}

/// The claimed sum of the `def_y1[t]` instance (7.5): `y^(1)_t(r) + 2 kappa^(1)_t(r)`.
#[must_use]
pub fn parity1_claimed_sum(y1: E, kappa1: E) -> E {
    y1 + kappa1 + kappa1
}

/// `P_Z = Z (Z + 1)` at a point value.
#[must_use]
pub fn digit_square(z: E) -> E {
    z * (z + E::from_i64(1))
}

fn shifted_product(p: E, roots: &[i64]) -> E {
    roots
        .iter()
        .fold(E::from_i64(1), |acc, &root| acc * (p - E::from_i64(root)))
}

/// The leaf predicate `Q_b(P)` of a base-4 or base-6 digit (spec Section 4.3): zero on
/// `P = Z (Z + 1)` exactly when `Z` is in the balanced alphabet.
///
/// # Panics
/// For another base.
#[must_use]
pub fn leaf(base: u32, p: E) -> E {
    match base {
        4 => shifted_product(p, &[0, 2]),
        6 => shifted_product(p, &[0, 2, 6]),
        _ => panic!("leaf-checked digits have base 4 or 6, not {base}"),
    }
}

/// `z, u, v, w_1'` at a main-domain point from the stage-3 evaluations of the twelve
/// main-domain polynomials (`G_main16` then `G_main4`, manifest order).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MainValues {
    /// `z` by (4.1).
    pub z: E,
    /// `u` by (4.1).
    pub u: E,
    /// `v` by (4.1).
    pub v: E,
    /// `w_1'`.
    pub w1: E,
}

impl MainValues {
    /// From the claims `value(column)` of the main-domain columns.
    #[must_use]
    pub fn from_claims(value: impl Fn(Column) -> E) -> Self {
        let compose = |digit: DigitValue| {
            compose_digits(&digit.layout(), &[0, 1].map(|d| value(Column::Digit { value: digit, digit: d })))
        };
        let bits: [E; W1_BITS] = std::array::from_fn(|t| value(Column::W1Bit(t as u8)));
        Self {
            z: compose(DigitValue::Z),
            u: compose(DigitValue::U),
            v: compose(DigitValue::V),
            w1: w1_from_bits(&bits),
        }
    }

    /// `2 gamma_2 w_1' + u + q v`.
    #[must_use]
    pub fn residual_high(&self) -> E {
        residual_high(self.w1, self.u, self.v)
    }
}

/// The carries (and the level-2 parities) of a round-2 group from its evaluations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CarryValues {
    /// `kappa^(l)_t` by (4.1) (`K + 2^15`), per lane (one for level 3).
    pub kappa: Vec<E>,
    /// `y^(2)_t` (level 2 only).
    pub parity: Vec<E>,
}

impl CarryValues {
    /// From the evaluations of the polynomials of `role` (manifest order).
    ///
    /// # Panics
    /// For a main-domain or multiplicity group, or a wrong value count.
    #[must_use]
    pub fn from_evaluations(manifest: &CommitmentManifest, role: GroupRole, values: &[E]) -> Self {
        let group = manifest.group(role);
        assert_eq!(values.len(), group.polynomial_count(), "one evaluation per polynomial");
        let carry = |level: u8, t: u8| {
            compose_digits(&ValueForm::Radix(CARRY16_DIGITS), &[values[group.position(Column::carry(level, t)).unwrap()]])
        };
        let lanes = 0..W1_BITS as u8;
        match role {
            GroupRole::Level1 => Self {
                kappa: lanes.map(|t| carry(1, t)).collect(),
                parity: Vec::new(),
            },
            GroupRole::Level2 => Self {
                kappa: lanes.clone().map(|t| carry(2, t)).collect(),
                parity: lanes
                    .map(|t| values[group.position(Column::Parity2(t)).unwrap()])
                    .collect(),
            },
            GroupRole::Level3 => Self {
                kappa: vec![carry(3, 0)],
                parity: Vec::new(),
            },
            other => panic!("{other:?} holds no carries"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agg_witness::digits::{leaf_vanishes, p_of, Alphabet};
    use agg_witness::gen::SplitMix64;

    fn lift(value: i64) -> E {
        E::from_i64(value)
    }

    #[test]
    fn w1_from_bits_is_the_binary_value() {
        for w in 0..64_i64 {
            let bits: [E; W1_BITS] = std::array::from_fn(|t| lift((w >> t) & 1));
            assert_eq!(w1_from_bits(&bits), lift(w));
        }
        let lanes: Vec<Vec<i8>> = (0..W1_BITS)
            .map(|t| (0..64).map(|w| ((w >> t) & 1) as i8).collect())
            .collect();
        let lanes: [&[i8]; W1_BITS] = std::array::from_fn(|t| lanes[t].as_slice());
        assert_eq!(w1_table(&lanes), (0..64).collect::<Vec<i64>>());
    }

    #[test]
    fn digit_composition_matches_the_l16_layouts() {
        let mut rng = SplitMix64::new(4);
        for value in [DigitValue::Z, DigitValue::U, DigitValue::V] {
            let layout = value.layout();
            let (lo, hi): (Vec<i32>, Vec<i8>) = (0..64)
                .map(|_| {
                    let a = layout.alphabet(0);
                    let b = layout.alphabet(1);
                    let pick = |rng: &mut SplitMix64, min: i64, max: i64| min + rng.below((max - min + 1) as u64) as i64;
                    let lo = pick(&mut rng, a.min(), a.max());
                    let hi = pick(&mut rng, b.min(), b.max().min(127)).max(-128);
                    (lo as i32, hi as i8)
                })
                .unzip();
            let hi_wide: Vec<i32> = hi.iter().map(|&h| i32::from(h)).collect();
            let table = compose_table(&layout, &[Cells::Wide(&lo), Cells::Wide(&hi_wide)]);
            for x in 0..64 {
                let digits = [i64::from(lo[x]), i64::from(hi[x])];
                assert_eq!(table[x], layout.compose(&digits));
                assert_eq!(compose_digits(&layout, &digits.map(lift)), lift(table[x]));
                let (min, max) = layout.range();
                assert!((min..=max).contains(&table[x]));
            }
            // The digits of zero (padded rows) compose to zero.
            let zero: Vec<E> = layout.decompose(0).into_iter().map(lift).collect();
            assert_eq!(compose_digits(&layout, &zero), E::zero());
        }
        // A carry: kappa = K + 2^15.
        let carry = ValueForm::Radix(CARRY16_DIGITS);
        assert_eq!(compose_digits(&carry, &[lift(-(1 << 15))]), E::zero());
        assert_eq!(compose_digits(&carry, &[lift(1000 - (1 << 15))]), lift(1000));
    }

    #[test]
    fn leaves_vanish_exactly_on_the_balanced_alphabets() {
        for z in i64::from(i8::MIN)..=i64::from(i8::MAX) {
            let p = digit_square(lift(z));
            assert_eq!(p, E::from_i128(p_of(z)));
            for base in [4_u32, 6] {
                let inside = Alphabet::Balanced(base).contains(z);
                assert_eq!(leaf(base, p) == E::zero(), inside, "base {base} digit {z}");
                assert_eq!(leaf_vanishes(base, p_of(z)), inside);
            }
        }
    }

    #[test]
    fn residual_parts_are_the_ring_identity() {
        // (3.1) for integers: az - t - 2 gamma2 w1 - u - q v.
        let (az, t, w1, u, v) = (562_061_561_757_i64, -326_836_224, 43, -190_465, 67_068);
        let expected = az - t - 2 * GAMMA2 * w1 - u - Q * v;
        assert_eq!(
            residual(lift(az), lift(t), lift(w1), lift(u), lift(v)),
            lift(expected)
        );
        assert_eq!(parity1_claimed_sum(lift(1), lift(700)), lift(1401));
    }
}
