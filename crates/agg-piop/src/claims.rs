//! The claim vectors exchanged between the stages (spec Sections 7.2 to 7.5, lookup layout
//! L16) and their canonical wire order.

use agg_pcs_config::catalogs::MULTIPLICITY_DIGITS;
use agg_witness::params::W1_BITS;
use jolt_field::{Ring, Zero};

use crate::manifest::{Column, CommitmentManifest, GroupRole, CARRY16_DIGITS, STAGE3_POLYNOMIALS};
use crate::source_tables::SourceTables;
use crate::stage::suffix;
use crate::virtual_polys::compose_digits;
use crate::OpeningField as E;

fn take<const M: usize>(values: &[E], offset: &mut usize) -> [E; M] {
    let out: [E; M] = values[*offset..*offset + M].try_into().expect("length");
    *offset += M;
    out
}

/// A carry `kappa = K + 2^15` from the claim on its balanced column `K`.
#[must_use]
pub fn carry_value(stored: E) -> E {
    compose_digits(&CARRY16_DIGITS, &[stored])
}

/// The 13 pre-stage claims (spec Section 7.2): `y^(2)_t(tau_2)`, `K^(2)_t(tau_2)`,
/// `K^(3)(tau_3)` (the balanced carry columns).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreStageClaims {
    /// `y^(2)_t(tau_2)`.
    pub y2: [E; W1_BITS],
    /// `K^(2)_t(tau_2)`.
    pub kappa2: [E; W1_BITS],
    /// `K^(3)(tau_3)`.
    pub kappa3: E,
}

impl PreStageClaims {
    /// Wire length.
    pub const LEN: usize = 2 * W1_BITS + 1;

    /// The honest claims: the committed columns at the suffixes of `tau`.
    #[must_use]
    pub fn evaluate(manifest: &CommitmentManifest, tables: SourceTables<'_>, tau: &[E]) -> Self {
        let tau2 = suffix(tau, manifest.group(GroupRole::Level2).logical_vars);
        let tau3 = suffix(tau, manifest.group(GroupRole::Level3).logical_vars);
        let at = |column: Column, point: &[E]| tables.column(column).evaluate(point);
        Self {
            y2: std::array::from_fn(|t| at(Column::Parity2(t as u8), &tau2)),
            kappa2: std::array::from_fn(|t| at(Column::carry(2, t as u8), &tau2)),
            kappa3: at(Column::carry(3, 0), &tau3),
        }
    }

    /// `kappa^(2)_t(tau_2)` by (4.1).
    #[must_use]
    pub fn kappa2_value(&self, t: usize) -> E {
        carry_value(self.kappa2[t])
    }

    /// `kappa^(3)(tau_3)` by (4.1).
    #[must_use]
    pub fn kappa3_value(&self) -> E {
        carry_value(self.kappa3)
    }

    /// Wire order.
    #[must_use]
    pub fn to_vec(&self) -> Vec<E> {
        let mut out = Vec::with_capacity(Self::LEN);
        out.extend_from_slice(&self.y2);
        out.extend_from_slice(&self.kappa2);
        out.push(self.kappa3);
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
        let kappa2 = take(values, &mut offset);
        Some(Self {
            y2,
            kappa2,
            kappa3: values[offset],
        })
    }
}

/// The 6 claims `K^(1)_t(r^(1)_1)` sent before `gamma^(2)` (spec Section 7.4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Kappa1Claims(pub [E; W1_BITS]);

impl Kappa1Claims {
    /// Wire length.
    pub const LEN: usize = W1_BITS;

    /// The honest claims at the `D_1` suffix of `r1`.
    #[must_use]
    pub fn evaluate(manifest: &CommitmentManifest, tables: SourceTables<'_>, r1: &[E]) -> Self {
        let r1_1 = suffix(r1, manifest.group(GroupRole::Level1).logical_vars);
        Self(std::array::from_fn(|t| {
            tables.column(Column::carry(1, t as u8)).evaluate(&r1_1)
        }))
    }

    /// `kappa^(1)_t(r^(1)_1)` by (4.1).
    #[must_use]
    pub fn value(&self, t: usize) -> E {
        carry_value(self.0[t])
    }

    /// Wire order.
    #[must_use]
    pub fn to_vec(&self) -> Vec<E> {
        self.0.to_vec()
    }

    /// Parse the wire order.
    #[must_use]
    pub fn from_slice(values: &[E]) -> Option<Self> {
        Some(Self(values.try_into().ok()?))
    }
}

/// The 18 stage-1 terminal claims (spec Section 7.3): `y^(1)_t(r^(1)_1)`, `y^(2)_t(r^(1)_2)`,
/// `b_t(r^(1))`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1Claims {
    /// `y^(1)_t(r^(1)_1)`.
    pub y1: [E; W1_BITS],
    /// `y^(2)_t(r^(1)_2)`.
    pub y2: [E; W1_BITS],
    /// `b_t(r^(1))`.
    pub b: [E; W1_BITS],
}

impl Stage1Claims {
    /// Wire length.
    pub const LEN: usize = 3 * W1_BITS;

    /// Wire order.
    #[must_use]
    pub fn to_vec(&self) -> Vec<E> {
        let mut out = Vec::with_capacity(Self::LEN);
        out.extend_from_slice(&self.y1);
        out.extend_from_slice(&self.y2);
        out.extend_from_slice(&self.b);
        out
    }

    /// Parse the wire order.
    #[must_use]
    pub fn from_slice(values: &[E]) -> Option<Self> {
        if values.len() != Self::LEN {
            return None;
        }
        let mut offset = 0;
        Some(Self {
            y1: take(values, &mut offset),
            y2: take(values, &mut offset),
            b: take(values, &mut offset),
        })
    }
}

/// The 8 stage-2 terminal claims (spec Section 7.4): `b_t(r^(2))` and `P_Z(r^(2))` for
/// `Z_hi`, `U_hi`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2Claims {
    /// `b_t(r^(2))`.
    pub b: [E; W1_BITS],
    /// `P_Z(r^(2))` for `Z_hi`, `U_hi`.
    pub squares: [E; 2],
}

impl Stage2Claims {
    /// Wire length.
    pub const LEN: usize = W1_BITS + 2;

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
        Some(Self {
            b: take(values, &mut offset),
            squares: take(values, &mut offset),
        })
    }
}

/// The 31 stage-3 terminal claims: every polynomial of `G_main16, G_main4, G_1, G_2, G_3`
/// at `r^(3)` restricted to its group, in manifest order (group by group). `G_m` is opened
/// at the lookup's table point instead ([`MultiplicityClaims`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3Claims(pub Vec<E>);

impl Stage3Claims {
    /// Wire length.
    pub const LEN: usize = STAGE3_POLYNOMIALS;

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

/// The seven claims `m_k(pi_m)` on the digits of the multiplicity column at the table
/// tree's final point (`docs/lookup.md` Section 1): the verifier checks
/// `sum_k 16^k m_k(pi_m) = m(pi_m)` against the lookup's claim; the Akita opening of `G_m`
/// authenticates them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiplicityClaims(pub [E; MULTIPLICITY_DIGITS]);

impl MultiplicityClaims {
    /// Wire length.
    pub const LEN: usize = MULTIPLICITY_DIGITS;

    /// `m = sum_k 16^k m_k`.
    #[must_use]
    pub fn compose(&self) -> E {
        self.0.iter().rev().fold(E::zero(), |acc, &digit| acc.mul_u64(16) + digit)
    }

    /// Parse the wire order.
    #[must_use]
    pub fn from_slice(values: &[E]) -> Option<Self> {
        Some(Self(values.try_into().ok()?))
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
        assert_eq!(pre.to_vec()[6], pre.kappa2[0]);
        assert_eq!(pre.to_vec()[12], pre.kappa3);
        assert!(PreStageClaims::from_slice(&pre.to_vec()[1..]).is_none());
        let k1 = Kappa1Claims::from_slice(&point(&mut rng, Kappa1Claims::LEN)).unwrap();
        assert_eq!(Kappa1Claims::from_slice(&k1.to_vec()), Some(k1.clone()));
        let s1 = Stage1Claims::from_slice(&point(&mut rng, Stage1Claims::LEN)).unwrap();
        assert_eq!(Stage1Claims::from_slice(&s1.to_vec()), Some(s1.clone()));
        let s2 = Stage2Claims::from_slice(&point(&mut rng, Stage2Claims::LEN)).unwrap();
        assert_eq!(Stage2Claims::from_slice(&s2.to_vec()), Some(s2.clone()));
        let manifest = CommitmentManifest::for_batch(8).unwrap();
        let s3 = Stage3Claims::from_slice(&point(&mut rng, 31)).unwrap();
        assert_eq!(Stage3Claims::group_offset(&manifest, GroupRole::Level2), 18);
        assert_eq!(s3.group(&manifest, GroupRole::Level3), &s3.0[30..]);
        assert_eq!(s3.column(&manifest, Column::Parity2(1)), s3.0[19]);
        assert_eq!(s3.column(&manifest, Column::W1Bit(0)), s3.0[6]);
        assert!(Stage3Claims::from_slice(&s3.0[..30]).is_none());
        let digits = MultiplicityClaims([3, -2, 0, 1, 0, 0, 7].map(E::from_i64));
        assert_eq!(digits.compose(), E::from_i64(3 - 32 + 4096 + 7 * (1 << 24)));
        assert_eq!(carry_value(E::from_i64(-(1 << 15))), E::from_i64(0));
    }
}
