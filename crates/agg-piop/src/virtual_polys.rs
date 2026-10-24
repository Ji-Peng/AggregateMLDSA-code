//! Virtual polynomials (spec Section 5).
//!
//! Every virtual table is defined on Boolean points and extended multilinearly as a table.
//! Two kinds of evaluator live here:
//!
//! * affine evaluators, valid at any point because the table is an affine combination of
//!   committed tables: `w_1' = sum_t 2^t b_t`, the digit compositions (4.1) of
//!   `z, u, v, kappa`, and the high part `2 gamma_2 w_1' + u + q v` of the ring residual.
//!   The verifier forms them from opened values ([`MainValues`], [`CarryValues`]);
//! * Boolean-domain materializations used by the prover and the tests: `w_1'`, the
//!   residual `Delta = (A z) - T - 2 gamma_2 w_1' - u - q v`, the level-1 parities
//!   `y^(1)_t = sum_zeta M_1[rho_1, zeta] b_t - 2 kappa^(1)_t` (spec (7.5); the sums are
//!   `agg_bridge::parity::level1_sums` at the production shape, for any [`SparseMatrix`] of
//!   that shape), and the digit tables `P_Z = Z (Z + 1)`, `A_Z^lo = Q_16^lo(P_Z)`,
//!   `A_Z^hi = Q_16^hi(P_Z)` with the leaf predicates of spec Section 4.3.
//!
//! `y^(1)_t` is not affine in committed evaluations at a non-Boolean point: its value there
//! is a claim reduced by the `def_y1` sum-check, whose claimed sum is
//! [`parity1_claimed_sum`].

use agg_bridge::{parity::level1_sums, Shape};
use agg_witness::bridge::SparseMatrix;
use agg_witness::crt::az_integer;
use agg_witness::digits::DigitLayout;
use agg_witness::instance::PublicInstance;
use agg_witness::padding::main_index;
use agg_witness::params::{GAMMA2, L, N, POSITIONS, Q, W1_BITS};
use jolt_field::{Ring, Zero};
use rayon::prelude::*;

use crate::manifest::{Column, CommitmentManifest, DigitValue, GroupRole};
use crate::source_tables::{MainTables, SourceTables};
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
pub fn compose_digits(layout: &DigitLayout, digits: &[E]) -> E {
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
pub fn compose_table(layout: &DigitLayout, digits: &[&[i8]]) -> Vec<i64> {
    assert_eq!(digits.len(), layout.len(), "one table per digit");
    let weights = layout.weights();
    let constant = layout.constant();
    (0..digits[0].len())
        .into_par_iter()
        .map(|x| {
            digits
                .iter()
                .zip(&weights)
                .fold(constant, |sum, (column, &weight)| {
                    sum + weight * i64::from(column[x])
                })
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
    let z = compose_table(
        &DigitValue::Z.layout(),
        &main.digits(manifest, DigitValue::Z),
    );
    let u = compose_table(
        &DigitValue::U.layout(),
        &main.digits(manifest, DigitValue::U),
    );
    let v = compose_table(
        &DigitValue::V.layout(),
        &main.digits(manifest, DigitValue::V),
    );
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
/// Boolean domain `D_1`, over the integers, per lane (spec Section 5, (7.5)). The
/// honest value is the GF(2) parity of the level-1 row, `{0, 1}`.
#[must_use]
pub fn parity1_tables(tables: SourceTables<'_>, m1: &SparseMatrix) -> [Vec<i64>; W1_BITS] {
    let sums = level1_sums(&Shape::PRODUCTION, &tables.main.bit_lanes(tables.manifest), m1);
    let layout = DigitValue::Kappa1(0).layout();
    let mut out: [Vec<i64>; W1_BITS] = Default::default();
    for (t, (sum, slot)) in sums.into_iter().zip(out.iter_mut()).enumerate() {
        let kappa = compose_table(&layout, &tables.digits(DigitValue::Kappa1(t as u8)));
        *slot = sum.iter().zip(&kappa).map(|(s, k)| s - 2 * k).collect();
    }
    out
}

/// The claimed sum of the `def_y1[t]` instance (7.5): `y^(1)_t(r) + 2 kappa^(1)_t(r)`.
#[must_use]
pub fn parity1_claimed_sum(y1: E, kappa1: E) -> E {
    y1 + kappa1 + kappa1
}

/// `P_Z = Z (Z + 1)`.
#[must_use]
pub fn digit_square(z: E) -> E {
    z * (z + E::from_i64(1))
}

fn shifted_product(p: E, roots: &[i64]) -> E {
    roots
        .iter()
        .fold(E::from_i64(1), |acc, &root| acc * (p - E::from_i64(root)))
}

/// `Q_16^lo(P) = Q_8(P) = P (P - 2)(P - 6)(P - 12)`.
#[must_use]
pub fn q16_lo(p: E) -> E {
    shifted_product(p, &[0, 2, 6, 12])
}

/// `Q_16^hi(P) = (P - 20)(P - 30)(P - 42)(P - 56)`.
#[must_use]
pub fn q16_hi(p: E) -> E {
    shifted_product(p, &[20, 30, 42, 56])
}

/// The leaf predicate `Q_b` of a digit of base `b` (spec Section 4.3): `Q_4(P) = P (P - 2)`,
/// `Q_8(P) = Q_16^lo(P)`, `Q_16 = Q_16^lo Q_16^hi`. It vanishes at `P_Z` exactly when `Z` lies in
/// the balanced alphabet.
#[must_use]
pub fn leaf(base: u32, p: E) -> E {
    match base {
        4 => shifted_product(p, &[0, 2]),
        8 => q16_lo(p),
        16 => q16_lo(p) * q16_hi(p),
        _ => panic!("unsupported digit base {base}"),
    }
}

/// The virtual tables of one committed digit on its Boolean domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DigitTables {
    /// `P_Z = Z (Z + 1)`.
    pub p: Vec<E>,
    /// `A_Z^lo = Q_16^lo(P_Z)` (base 16 only, else empty).
    pub a_lo: Vec<E>,
    /// `A_Z^hi = Q_16^hi(P_Z)` (base 16 only, else empty).
    pub a_hi: Vec<E>,
}

/// Materialize `P_Z` and, for base 16, `A_Z^lo, A_Z^hi` (stages 1 and 2).
#[must_use]
pub fn digit_tables(cells: &[i8], base: u32) -> DigitTables {
    let p: Vec<E> = cells
        .par_iter()
        .map(|&z| digit_square(E::from_i64(i64::from(z))))
        .collect();
    let (a_lo, a_hi) = if base == 16 {
        (
            p.par_iter().map(|&value| q16_lo(value)).collect(),
            p.par_iter().map(|&value| q16_hi(value)).collect(),
        )
    } else {
        (Vec::new(), Vec::new())
    };
    DigitTables { p, a_lo, a_hi }
}

/// The values the verifier forms from the 23 opened `G_main` evaluations (any point).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MainValues {
    /// `z` by (4.1).
    pub z: E,
    /// `u` by (4.1).
    pub u: E,
    /// `v` by (4.1).
    pub v: E,
    /// `w_1'` from the bits.
    pub w1: E,
}

impl MainValues {
    /// From the evaluations of `G_main` in manifest order.
    #[must_use]
    pub fn from_evaluations(manifest: &CommitmentManifest, values: &[E]) -> Self {
        let group = manifest.group(GroupRole::Main);
        assert_eq!(
            values.len(),
            group.polynomial_count(),
            "23 G_main evaluations"
        );
        let compose = |value: DigitValue| {
            let digits: Vec<E> = (0..value.layout().len() as u8)
                .map(|digit| values[group.position(Column::Digit { value, digit }).unwrap()])
                .collect();
            compose_digits(&value.layout(), &digits)
        };
        let bits: [E; W1_BITS] =
            std::array::from_fn(|t| values[group.position(Column::W1Bit(t as u8)).unwrap()]);
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

/// The carries the verifier forms from opened (or pre-stage) digit evaluations of one
/// round-2 group (any point).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CarryValues {
    /// `kappa^(1)_t`, `kappa^(2)_t` per lane, or the single `kappa^(3)`.
    pub kappa: Vec<E>,
    /// `y^(2)_t` (level 2 only, else empty).
    pub parity: Vec<E>,
}

impl CarryValues {
    /// From the evaluations of `role` (a round-2 group) in manifest order.
    #[must_use]
    pub fn from_evaluations(manifest: &CommitmentManifest, role: GroupRole, values: &[E]) -> Self {
        let group = manifest.group(role);
        assert_eq!(
            values.len(),
            group.polynomial_count(),
            "one evaluation per polynomial"
        );
        let compose = |value: DigitValue| {
            let digits: Vec<E> = (0..value.layout().len() as u8)
                .map(|digit| values[group.position(Column::Digit { value, digit }).unwrap()])
                .collect();
            compose_digits(&value.layout(), &digits)
        };
        let lanes = 0..W1_BITS as u8;
        match role {
            GroupRole::Level1 => Self {
                kappa: lanes.map(|t| compose(DigitValue::Kappa1(t))).collect(),
                parity: Vec::new(),
            },
            GroupRole::Level2 => Self {
                kappa: lanes
                    .clone()
                    .map(|t| compose(DigitValue::Kappa2(t)))
                    .collect(),
                parity: lanes
                    .map(|t| values[group.position(Column::Parity2(t)).unwrap()])
                    .collect(),
            },
            GroupRole::Level3 => Self {
                kappa: vec![compose(DigitValue::Kappa3)],
                parity: Vec::new(),
            },
            GroupRole::Main => panic!("G_main holds no carries"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agg_witness::check::digit_virtual;
    use agg_witness::digits::{leaf_vanishes, p_of, Alphabet};
    use agg_witness::gen::SplitMix64;

    const LAYOUTS: [DigitValue; 5] = [
        DigitValue::Z,
        DigitValue::U,
        DigitValue::V,
        DigitValue::Kappa1(0),
        DigitValue::Kappa3,
    ];

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
    fn digit_composition_matches_the_witness_layouts() {
        let mut rng = SplitMix64::new(4);
        for value in LAYOUTS {
            let layout = value.layout();
            let columns: Vec<Vec<i8>> = (0..layout.len())
                .map(|k| {
                    let alphabet = layout.alphabet(k);
                    (0..64)
                        .map(|_| {
                            let width = (alphabet.max() - alphabet.min() + 1) as u64;
                            (alphabet.min() + rng.below(width) as i64) as i8
                        })
                        .collect()
                })
                .collect();
            let refs: Vec<&[i8]> = columns.iter().map(Vec::as_slice).collect();
            let table = compose_table(&layout, &refs);
            for x in 0..64 {
                let digits: Vec<i64> = columns.iter().map(|c| i64::from(c[x])).collect();
                assert_eq!(table[x], layout.compose(&digits));
                let lifted: Vec<E> = digits.iter().map(|&d| lift(d)).collect();
                assert_eq!(compose_digits(&layout, &lifted), lift(table[x]));
                let (min, max) = layout.range();
                assert!((min..=max).contains(&table[x]));
            }
        }
        // The digits of zero (padded rows) compose to zero.
        for value in [DigitValue::Z, DigitValue::U, DigitValue::V] {
            let layout = value.layout();
            let zero: Vec<E> = layout.decompose(0).into_iter().map(lift).collect();
            assert_eq!(compose_digits(&layout, &zero), E::zero());
        }
    }

    #[test]
    fn leaves_vanish_exactly_on_the_balanced_alphabets() {
        for z in i64::from(i8::MIN)..=i64::from(i8::MAX) {
            let p = digit_square(lift(z));
            assert_eq!(p, E::from_i128(p_of(z)));
            for base in [4_u32, 8, 16] {
                let inside = Alphabet::Balanced(base).contains(z);
                assert_eq!(leaf(base, p) == E::zero(), inside, "base {base} digit {z}");
                assert_eq!(leaf_vanishes(base, p_of(z)), inside);
            }
            let product_zero = q16_lo(p) * q16_hi(p) == E::zero();
            assert_eq!(product_zero, Alphabet::Balanced(16).contains(z));
        }
    }

    #[test]
    fn digit_tables_match_the_checker_materialization() {
        let cells: Vec<i8> = (i8::MIN..=i8::MAX).collect();
        for base in [4_u32, 8, 16] {
            let ours = digit_tables(&cells, base);
            let reference = digit_virtual(&cells, base);
            for x in 0..cells.len() {
                assert_eq!(ours.p[x], E::from_i128(reference.p[x]));
                if base == 16 {
                    assert_eq!(ours.a_lo[x], E::from_i128(reference.a_lo[x].unwrap()));
                    assert_eq!(ours.a_hi[x], E::from_i128(reference.a_hi[x].unwrap()));
                }
            }
            assert_eq!(ours.a_lo.is_empty(), base != 16);
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
