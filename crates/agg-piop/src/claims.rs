//! The claim vectors exchanged between the stages (spec Sections 7.2 to 7.5) and their
//! canonical wire order.

use agg_witness::digits::{KAPPA12_DIGITS, KAPPA3_DIGITS};
use agg_witness::params::W1_BITS;

use crate::manifest::{Column, CommitmentManifest, DigitValue, GroupRole};
use crate::pcs::evaluate_column;
use crate::source_tables::SourceTables;
use crate::stage::suffix;
use crate::virtual_polys::compose_digits;
use crate::OpeningField as E;

/// Digits per level-1 and level-2 carry.
pub const CARRY12_DIGITS: usize = 3;
/// Digits per level-3 carry.
pub const CARRY3_DIGITS: usize = 2;

fn take<const M: usize>(values: &[E], offset: &mut usize) -> [E; M] {
    let out: [E; M] = values[*offset..*offset + M].try_into().expect("length");
    *offset += M;
    out
}

/// The 26 pre-stage claims (spec Section 7.2): `y^(2)_t(tau_2)`, `kappa^(2)_{t,k}(tau_2)`,
/// `kappa^(3)_k(tau_3)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreStageClaims {
    /// `y^(2)_t(tau_2)`.
    pub y2: [E; W1_BITS],
    /// `kappa^(2)_{t,k}(tau_2)`, `t`-major.
    pub kappa2: [[E; CARRY12_DIGITS]; W1_BITS],
    /// `kappa^(3)_k(tau_3)`.
    pub kappa3: [E; CARRY3_DIGITS],
}

impl PreStageClaims {
    /// Wire length.
    pub const LEN: usize = W1_BITS + W1_BITS * CARRY12_DIGITS + CARRY3_DIGITS;

    /// The honest claims: the committed columns at the suffixes of `tau`.
    #[must_use]
    pub fn evaluate(manifest: &CommitmentManifest, tables: SourceTables<'_>, tau: &[E]) -> Self {
        let tau2 = suffix(tau, manifest.group(GroupRole::Level2).logical_vars);
        let tau3 = suffix(tau, manifest.group(GroupRole::Level3).logical_vars);
        let at = |column: Column, point: &[E]| evaluate_column(tables.column(column), point);
        Self {
            y2: std::array::from_fn(|t| at(Column::Parity2(t as u8), &tau2)),
            kappa2: std::array::from_fn(|t| {
                std::array::from_fn(|k| {
                    at(
                        Column::Digit {
                            value: DigitValue::Kappa2(t as u8),
                            digit: k as u8,
                        },
                        &tau2,
                    )
                })
            }),
            kappa3: std::array::from_fn(|k| {
                at(
                    Column::Digit {
                        value: DigitValue::Kappa3,
                        digit: k as u8,
                    },
                    &tau3,
                )
            }),
        }
    }

    /// `kappa^(2)_t(tau_2)` by (4.1).
    #[must_use]
    pub fn kappa2_value(&self, t: usize) -> E {
        compose_digits(&KAPPA12_DIGITS, &self.kappa2[t])
    }

    /// `kappa^(3)(tau_3)` by (4.1).
    #[must_use]
    pub fn kappa3_value(&self) -> E {
        compose_digits(&KAPPA3_DIGITS, &self.kappa3)
    }

    /// Wire order.
    #[must_use]
    pub fn to_vec(&self) -> Vec<E> {
        let mut out = Vec::with_capacity(Self::LEN);
        out.extend_from_slice(&self.y2);
        out.extend(self.kappa2.iter().flatten());
        out.extend_from_slice(&self.kappa3);
        out
    }

    /// Parse the wire order.
    #[must_use]
    pub fn from_slice(values: &[E]) -> Option<Self> {
        if values.len() != Self::LEN {
            return None;
        }
        let mut offset = 0;
        let y2 = take(values, &mut offset);
        let kappa2 = std::array::from_fn(|_| take(values, &mut offset));
        let kappa3 = take(values, &mut offset);
        Some(Self { y2, kappa2, kappa3 })
    }
}

/// The 18 claims `kappa^(1)_{t,k}(r^(1)_1)` sent before `gamma^(2)` (spec Section 7.4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Kappa1Claims(pub [[E; CARRY12_DIGITS]; W1_BITS]);

impl Kappa1Claims {
    /// Wire length.
    pub const LEN: usize = W1_BITS * CARRY12_DIGITS;

    /// The honest claims at the `D_1` suffix of `r1`.
    #[must_use]
    pub fn evaluate(manifest: &CommitmentManifest, tables: SourceTables<'_>, r1: &[E]) -> Self {
        let r1_1 = suffix(r1, manifest.group(GroupRole::Level1).logical_vars);
        Self(std::array::from_fn(|t| {
            std::array::from_fn(|k| {
                evaluate_column(
                    tables.column(Column::Digit {
                        value: DigitValue::Kappa1(t as u8),
                        digit: k as u8,
                    }),
                    &r1_1,
                )
            })
        }))
    }

    /// `kappa^(1)_t(r^(1)_1)` by (4.1).
    #[must_use]
    pub fn value(&self, t: usize) -> E {
        compose_digits(&KAPPA12_DIGITS, &self.0[t])
    }

    /// Wire order (`t`-major).
    #[must_use]
    pub fn to_vec(&self) -> Vec<E> {
        self.0.iter().flatten().copied().collect()
    }

    /// Parse the wire order.
    #[must_use]
    pub fn from_slice(values: &[E]) -> Option<Self> {
        if values.len() != Self::LEN {
            return None;
        }
        let mut offset = 0;
        Some(Self(std::array::from_fn(|_| take(values, &mut offset))))
    }
}

/// The 98 stage-1 terminal claims (spec Section 7.3): `y^(1)_t(r^(1)_1)`, `y^(2)_t(r^(1)_2)`,
/// `b_t(r^(1))` and `(A_Z^lo, A_Z^hi)(r^(1)_Z)` per base-16 digit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1Claims {
    /// `y^(1)_t(r^(1)_1)`.
    pub y1: [E; W1_BITS],
    /// `y^(2)_t(r^(1)_2)`.
    pub y2: [E; W1_BITS],
    /// `b_t(r^(1))`.
    pub b: [E; W1_BITS],
    /// `(A_Z^lo, A_Z^hi)` per base-16 digit, in digit order.
    pub products: Vec<[E; 2]>,
}

impl Stage1Claims {
    /// Wire length.
    pub const LEN: usize = 3 * W1_BITS + 2 * 40;

    /// Wire order.
    #[must_use]
    pub fn to_vec(&self) -> Vec<E> {
        let mut out = Vec::with_capacity(Self::LEN);
        out.extend_from_slice(&self.y1);
        out.extend_from_slice(&self.y2);
        out.extend_from_slice(&self.b);
        out.extend(self.products.iter().flatten());
        out
    }

    /// Parse the wire order.
    #[must_use]
    pub fn from_slice(values: &[E]) -> Option<Self> {
        if values.len() != Self::LEN {
            return None;
        }
        let mut offset = 0;
        let y1 = take(values, &mut offset);
        let y2 = take(values, &mut offset);
        let b = take(values, &mut offset);
        let products = (0..40).map(|_| take(values, &mut offset)).collect();
        Some(Self {
            y1,
            y2,
            b,
            products,
        })
    }
}

/// The 61 stage-2 terminal claims (spec Section 7.4): `b_t(r^(2))` and `P_Z(r^(2)_Z)` per
/// digit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2Claims {
    /// `b_t(r^(2))`.
    pub b: [E; W1_BITS],
    /// `P_Z(r^(2)_Z)` in digit order.
    pub squares: Vec<E>,
}

impl Stage2Claims {
    /// Wire length.
    pub const LEN: usize = W1_BITS + 55;

    /// Wire order.
    #[must_use]
    pub fn to_vec(&self) -> Vec<E> {
        let mut out = Vec::with_capacity(Self::LEN);
        out.extend_from_slice(&self.b);
        out.extend_from_slice(&self.squares);
        out
    }

    /// Parse the wire order.
    #[must_use]
    pub fn from_slice(values: &[E]) -> Option<Self> {
        if values.len() != Self::LEN {
            return None;
        }
        let mut offset = 0;
        let b = take(values, &mut offset);
        Some(Self {
            b,
            squares: values[offset..].to_vec(),
        })
    }
}

/// The 67 stage-3 terminal claims: every committed polynomial at `r^(3)` restricted to its
/// group, in manifest order (group by group).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3Claims(pub Vec<E>);

impl Stage3Claims {
    /// Wire length.
    pub const LEN: usize = 67;

    /// Offset of a group's values.
    #[must_use]
    pub fn group_offset(manifest: &CommitmentManifest, role: GroupRole) -> usize {
        manifest.groups()[..role.index()]
            .iter()
            .map(|g| g.polynomial_count())
            .sum()
    }

    /// The claimed value of a column.
    #[must_use]
    pub fn column(&self, manifest: &CommitmentManifest, column: Column) -> E {
        let (role, position) = manifest.locate(column);
        self.0[Self::group_offset(manifest, role) + position]
    }

    /// The values of a group.
    #[must_use]
    pub fn group(&self, manifest: &CommitmentManifest, role: GroupRole) -> &[E] {
        let offset = Self::group_offset(manifest, role);
        &self.0[offset..offset + manifest.group(role).polynomial_count()]
    }

    /// Parse the wire order.
    #[must_use]
    pub fn from_slice(values: &[E]) -> Option<Self> {
        (values.len() == Self::LEN).then(|| Self(values.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::point;
    use agg_witness::gen::SplitMix64;

    #[test]
    fn claim_vectors_round_trip() {
        let mut rng = SplitMix64::new(9);
        let pre = PreStageClaims::from_slice(&point(&mut rng, PreStageClaims::LEN)).unwrap();
        assert_eq!(PreStageClaims::from_slice(&pre.to_vec()), Some(pre.clone()));
        assert_eq!(pre.to_vec()[6], pre.kappa2[0][0]);
        assert_eq!(pre.to_vec()[24], pre.kappa3[0]);
        assert!(PreStageClaims::from_slice(&pre.to_vec()[1..]).is_none());
        let k1 = Kappa1Claims::from_slice(&point(&mut rng, Kappa1Claims::LEN)).unwrap();
        assert_eq!(Kappa1Claims::from_slice(&k1.to_vec()), Some(k1.clone()));
        assert_eq!(k1.to_vec()[4], k1.0[1][1]);
        let s1 = Stage1Claims::from_slice(&point(&mut rng, Stage1Claims::LEN)).unwrap();
        assert_eq!(Stage1Claims::from_slice(&s1.to_vec()), Some(s1.clone()));
        assert_eq!(s1.to_vec()[18], s1.products[0][0]);
        let s2 = Stage2Claims::from_slice(&point(&mut rng, Stage2Claims::LEN)).unwrap();
        assert_eq!(Stage2Claims::from_slice(&s2.to_vec()), Some(s2.clone()));
        let manifest = CommitmentManifest::for_batch(8).unwrap();
        let s3 = Stage3Claims::from_slice(&point(&mut rng, 67)).unwrap();
        assert_eq!(Stage3Claims::group_offset(&manifest, GroupRole::Level2), 41);
        assert_eq!(s3.group(&manifest, GroupRole::Level3), &s3.0[65..]);
        assert_eq!(s3.column(&manifest, Column::Parity2(1)), s3.0[42]);
        assert!(Stage3Claims::from_slice(&s3.0[..66]).is_none());
    }
}
