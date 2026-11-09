//! The claim vectors exchanged between the stages and their canonical wire order.

use agg_pcs_config::catalogs::MULTIPLICITY_DIGITS;
use agg_witness::params::W1_BITS;
use jolt_field::{Ring, Zero};

use crate::manifest::{Column, CommitmentManifest, GroupRole, HashField, CARRY16_DIGITS};
use crate::source_tables::SourceTables;
use crate::stage::suffix;
use crate::virtual_polys::compose_digits;
use crate::OpeningField as E;

fn take(values: &[E], offset: &mut usize, len: usize) -> Vec<E> {
    let out = values[*offset..*offset + len].to_vec();
    *offset += len;
    out
}

/// Bits of `w_1'` of the manifest's parameter set (6 for ML-DSA-44, 4 otherwise).
fn w1_bits(manifest: &CommitmentManifest) -> usize {
    manifest.shape().w1_bits
}

/// The bit columns `b_t^(rho)` in claim order: bit-major (`t` outer, the polynomial lane
/// `rho` inner), one lane per bit for ML-DSA-44.
#[must_use]
pub fn bit_columns(manifest: &CommitmentManifest) -> Vec<Column> {
    let lanes = manifest.shape().lanes_k as u8;
    (0..w1_bits(manifest) as u8)
        .flat_map(|t| (0..lanes).map(move |lane| Column::W1Bit(t, lane)))
        .collect()
}

/// A carry `kappa = K + 2^15` from the claim on its balanced column `K`.
#[must_use]
pub fn carry_value(stored: E) -> E {
    compose_digits(&agg_witness::digits::ValueForm::Radix(CARRY16_DIGITS), &[stored])
}

/// The pre-stage claims (13 for ML-DSA-44, 9 otherwise): `y^(2)_t(tau_2)`,
/// `K^(2)_t(tau_2)`, `K^(3)(tau_3)` (the balanced carry columns).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreStageClaims {
    /// `y^(2)_t(tau_2)`.
    pub y2: Vec<E>,
    /// `K^(2)_t(tau_2)`.
    pub kappa2: Vec<E>,
    /// `K^(3)(tau_3)`.
    pub kappa3: E,
}

impl PreStageClaims {
    /// Wire length of ML-DSA-44.
    pub const LEN: usize = 2 * W1_BITS + 1;

    /// Wire length of the manifest's parameter set (`2 w + 1` with `w` bits of `w_1'`).
    #[must_use]
    pub fn len(manifest: &CommitmentManifest) -> usize {
        2 * w1_bits(manifest) + 1
    }

    /// The honest claims: the committed columns at the suffixes of `tau`.
    #[must_use]
    pub fn evaluate(manifest: &CommitmentManifest, tables: SourceTables<'_>, tau: &[E]) -> Self {
        let tau2 = suffix(tau, manifest.group(GroupRole::Level2).logical_vars);
        let tau3 = suffix(tau, manifest.group(GroupRole::Level3).logical_vars);
        let at = |column: Column, point: &[E]| tables.column(column).evaluate(point);
        let bits = 0..w1_bits(manifest) as u8;
        Self {
            y2: bits.clone().map(|t| at(Column::Parity2(t), &tau2)).collect(),
            kappa2: bits.map(|t| at(Column::carry(2, t), &tau2)).collect(),
            kappa3: at(Column::carry(3, 0), &tau3),
        }
    }

    /// `kappa^(2)_t(tau_2)` by `kappa = K + 2^15`.
    #[must_use]
    pub fn kappa2_value(&self, t: usize) -> E {
        carry_value(self.kappa2[t])
    }

    /// `kappa^(3)(tau_3)` by `kappa = K + 2^15`.
    #[must_use]
    pub fn kappa3_value(&self) -> E {
        carry_value(self.kappa3)
    }

    /// Wire order.
    #[must_use]
    pub fn to_vec(&self) -> Vec<E> {
        let mut out = Vec::with_capacity(2 * self.y2.len() + 1);
        out.extend_from_slice(&self.y2);
        out.extend_from_slice(&self.kappa2);
        out.push(self.kappa3);
        out
    }

    /// Parse the wire order of ML-DSA-44.
    #[must_use]
    pub fn from_slice(values: &[E]) -> Option<Self> {
        Self::from_slice_bits(values, W1_BITS)
    }

    /// Parse the wire order of the manifest's parameter set.
    #[must_use]
    pub fn from_slice_for(manifest: &CommitmentManifest, values: &[E]) -> Option<Self> {
        Self::from_slice_bits(values, w1_bits(manifest))
    }

    fn from_slice_bits(values: &[E], bits: usize) -> Option<Self> {
        if values.len() != 2 * bits + 1 {
            return None;
        }
        let mut offset = 0;
        let y2 = take(values, &mut offset, bits);
        let kappa2 = take(values, &mut offset, bits);
        Some(Self {
            y2,
            kappa2,
            kappa3: values[offset],
        })
    }
}

/// The claims `K^(1)_t(r^(1)_1)` sent before `gamma^(2)` (6 for ML-DSA-44, 4 otherwise).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Kappa1Claims(pub Vec<E>);

impl Kappa1Claims {
    /// Wire length of ML-DSA-44.
    pub const LEN: usize = W1_BITS;

    /// Wire length of the manifest's parameter set (`w` under `Gf2`, `0` under `Fp`).
    #[must_use]
    pub fn len(manifest: &CommitmentManifest) -> usize {
        match manifest.hash_field() {
            HashField::Gf2 => w1_bits(manifest),
            HashField::Fp => 0,
        }
    }

    /// The honest claims at the `D_1` suffix of `r1` (empty under `HashField::Fp`).
    #[must_use]
    pub fn evaluate(manifest: &CommitmentManifest, tables: SourceTables<'_>, r1: &[E]) -> Self {
        if manifest.hash_field() == HashField::Fp {
            return Self(Vec::new());
        }
        let r1_1 = suffix(r1, manifest.group(GroupRole::Level1).logical_vars);
        Self(
            (0..w1_bits(manifest) as u8)
                .map(|t| tables.column(Column::carry(1, t)).evaluate(&r1_1))
                .collect(),
        )
    }

    /// `kappa^(1)_t(r^(1)_1)` by `kappa = K + 2^15`.
    #[must_use]
    pub fn value(&self, t: usize) -> E {
        carry_value(self.0[t])
    }

    /// Wire order.
    #[must_use]
    pub fn to_vec(&self) -> Vec<E> {
        self.0.clone()
    }

    /// Parse the wire order of ML-DSA-44.
    #[must_use]
    pub fn from_slice(values: &[E]) -> Option<Self> {
        (values.len() == Self::LEN).then(|| Self(values.to_vec()))
    }

    /// Parse the wire order of the manifest's parameter set.
    #[must_use]
    pub fn from_slice_for(manifest: &CommitmentManifest, values: &[E]) -> Option<Self> {
        (values.len() == Self::len(manifest)).then(|| Self(values.to_vec()))
    }
}

/// The stage-1 terminal claims (18 for ML-DSA-44 under `Gf2`; `2 w + w k` with polynomial
/// lanes; under `Fp`, the `w k` bit claims followed by the non-public Keccak lane claims):
/// `y^(1)_t(r^(1)_1)`, `y^(2)_t(r^(1)_2)`, `b_t^(rho)(r^(1))` (bit-major, [`bit_columns`]),
/// and `keccak`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1Claims {
    /// `y^(1)_t(r^(1)_1)` (empty under `HashField::Fp`).
    pub y1: Vec<E>,
    /// `y^(2)_t(r^(1)_2)` (empty under `HashField::Fp`).
    pub y2: Vec<E>,
    /// `b_t^(rho)(r^(1))` in the order of [`bit_columns`].
    pub b: Vec<E>,
    /// Terminal evaluations of every non-public Keccak lane of stage 1 (`Column`, `Rot`,
    /// `Prev`, `PrevIota`, `Blk`) in `spec.lane_ids()` order (empty under `HashField::Gf2`).
    pub keccak: Vec<E>,
}

impl Stage1Claims {
    /// Wire length of ML-DSA-44 under `HashField::Gf2`.
    pub const LEN: usize = 3 * W1_BITS;

    /// Parity claims per level (`w` under `Gf2`, `0` under `Fp`).
    #[must_use]
    pub fn y_len(manifest: &CommitmentManifest) -> usize {
        match manifest.hash_field() {
            HashField::Gf2 => w1_bits(manifest),
            HashField::Fp => 0,
        }
    }

    /// Non-public Keccak lane claims (`0` under `Gf2`; under `Fp`, `66 Q + |t0_w1| + |out_rot|`
    /// from [`agg_fpkeccak::KeccakLayout`]).
    #[must_use]
    pub fn keccak_len(manifest: &CommitmentManifest) -> usize {
        match manifest.hash_field() {
            HashField::Gf2 => 0,
            HashField::Fp => {
                let layout = agg_fpkeccak::KeccakLayout::for_set(manifest.set());
                let per_slot_group = 2 * agg_fpkeccak::LANES + 3 * agg_fpkeccak::COLS + 1;
                per_slot_group * layout.slot_groups
                    + crate::keccak_fp::masks::t0_w1_lanes(&layout).len()
                    + crate::keccak_fp::masks::output_rot_columns(&layout).len()
            }
        }
    }

    /// Wire length of the manifest's parameter set and hash field.
    #[must_use]
    pub fn len(manifest: &CommitmentManifest) -> usize {
        2 * Self::y_len(manifest) + bit_columns(manifest).len() + Self::keccak_len(manifest)
    }

    /// Wire order.
    #[must_use]
    pub fn to_vec(&self) -> Vec<E> {
        let mut out = Vec::with_capacity(self.y1.len() + self.y2.len() + self.b.len() + self.keccak.len());
        out.extend_from_slice(&self.y1);
        out.extend_from_slice(&self.y2);
        out.extend_from_slice(&self.b);
        out.extend_from_slice(&self.keccak);
        out
    }

    /// Parse the wire order of ML-DSA-44 under `HashField::Gf2`.
    #[must_use]
    pub fn from_slice(values: &[E]) -> Option<Self> {
        Self::from_slice_shape(values, W1_BITS, W1_BITS, 0)
    }

    /// Parse the wire order of the manifest's parameter set and hash field.
    #[must_use]
    pub fn from_slice_for(manifest: &CommitmentManifest, values: &[E]) -> Option<Self> {
        Self::from_slice_shape(
            values,
            Self::y_len(manifest),
            bit_columns(manifest).len(),
            Self::keccak_len(manifest),
        )
    }

    fn from_slice_shape(values: &[E], bits: usize, bit_lanes: usize, keccak_lanes: usize) -> Option<Self> {
        if values.len() != 2 * bits + bit_lanes + keccak_lanes {
            return None;
        }
        let mut offset = 0;
        Some(Self {
            y1: take(values, &mut offset, bits),
            y2: take(values, &mut offset, bits),
            b: take(values, &mut offset, bit_lanes),
            keccak: take(values, &mut offset, keccak_lanes),
        })
    }
}

/// The stage-2 terminal claims: `b_t^(rho)(r^(2))` ([`bit_columns`], empty under
/// `HashField::Fp`), `P_Z(r^(2))` for each leaf-checked digit, H-FIPS's `delta(r^(2))` and
/// `e(r^(2))` (`stage2::FIPS_TERMINALS`), and the non-public Keccak lane claims (empty under
/// `HashField::Gf2`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2Claims {
    /// `b_t^(rho)(r^(2))` (empty under `HashField::Fp`).
    pub b: Vec<E>,
    /// `P_Z(r^(2))` per leaf-checked digit, in the order of `manifest.leaf_digits()`.
    pub squares: Vec<E>,
    /// H-FIPS: `delta(r^(2))`, `e(r^(2))` (empty otherwise).
    pub fips: Vec<E>,
    /// Terminal evaluations of every non-public Keccak lane of stage 2 in `spec.lane_ids()`
    /// order (empty under `HashField::Gf2`).
    pub keccak: Vec<E>,
}

impl Stage2Claims {
    /// Bit claims (`w k` under `Gf2`, `0` under `Fp`).
    #[must_use]
    pub fn b_len(manifest: &CommitmentManifest) -> usize {
        match manifest.hash_field() {
            HashField::Gf2 => bit_columns(manifest).len(),
            HashField::Fp => 0,
        }
    }

    /// Non-public Keccak lane claims (`0` under `Gf2`; under `Fp`, `86 (P - 1)` from
    /// [`agg_fpkeccak::KeccakLayout`]).
    #[must_use]
    pub fn keccak_len(manifest: &CommitmentManifest) -> usize {
        match manifest.hash_field() {
            HashField::Gf2 => 0,
            HashField::Fp => {
                let layout = agg_fpkeccak::KeccakLayout::for_set(manifest.set());
                let per_boundary_group = 3 * agg_fpkeccak::LANES + 2 * agg_fpkeccak::COLS + 1;
                per_boundary_group * crate::keccak_fp::masks::boundary_slot_groups(&layout).len()
            }
        }
    }

    /// Wire length under `manifest`'s parameter set, relation and hash field.
    #[must_use]
    pub fn len(manifest: &CommitmentManifest) -> usize {
        Self::b_len(manifest)
            + manifest.leaf_digits().len()
            + Self::fips_len(manifest)
            + Self::keccak_len(manifest)
    }

    /// H-FIPS claims (`2 LANES_K` under H-FIPS: 2 for ML-DSA-44, 12 and 16 for ML-DSA-65 and
    /// ML-DSA-87; else 0).
    #[must_use]
    pub fn fips_len(manifest: &CommitmentManifest) -> usize {
        crate::stage2::fips_terminals(manifest).len()
    }

    /// Wire order.
    #[must_use]
    pub fn to_vec(&self) -> Vec<E> {
        let mut out =
            Vec::with_capacity(self.b.len() + self.squares.len() + self.fips.len() + self.keccak.len());
        out.extend_from_slice(&self.b);
        out.extend_from_slice(&self.squares);
        out.extend_from_slice(&self.fips);
        out.extend_from_slice(&self.keccak);
        out
    }

    /// Parse the wire order (length [`Stage2Claims::len`]).
    #[must_use]
    pub fn from_slice(manifest: &CommitmentManifest, values: &[E]) -> Option<Self> {
        if values.len() != Self::len(manifest) {
            return None;
        }
        let mut offset = 0;
        let b = take(values, &mut offset, Self::b_len(manifest));
        let squares = take(values, &mut offset, manifest.leaf_digits().len());
        let fips = take(values, &mut offset, Self::fips_len(manifest));
        let keccak = take(values, &mut offset, Self::keccak_len(manifest));
        Some(Self {
            b,
            squares,
            fips,
            keccak,
        })
    }

    /// Parse the wire order of `manifest` (alias for [`Self::from_slice`]).
    #[must_use]
    pub fn from_slice_for(manifest: &CommitmentManifest, values: &[E]) -> Option<Self> {
        Self::from_slice(manifest, values)
    }
}

/// The stage-3 terminal claims: every polynomial of `G_main16, G_main4, G_1, G_2, G_3, G_k` at
/// `r^(3)` restricted to its group, in manifest order (group by group): 35 under H-u, 36
/// under H-exact (40 under H-FIPS). Under `HashField::Fp` the groups are `G_main16, G_main4`,
/// the seven Keccak groups (at the `d_K` suffix) and `G_k`: 401 under H-u. `G_m` is opened at
/// the lookup's table point instead ([`MultiplicityClaims`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3Claims(pub Vec<E>);

impl Stage3Claims {
    /// Wire length under `manifest`'s relation.
    #[must_use]
    pub fn len(manifest: &CommitmentManifest) -> usize {
        manifest.stage3_polynomials()
    }

    /// Offset of a group's values.
    #[must_use]
    pub fn group_offset(manifest: &CommitmentManifest, role: GroupRole) -> usize {
        manifest.groups()[..role.index_in(manifest.hash_field())]
            .iter()
            .map(|g| g.claimed_count())
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
        &self.0[offset..offset + manifest.group(role).claimed_count()]
    }

    /// Parse the wire order (length [`Stage3Claims::len`]).
    #[must_use]
    pub fn from_slice(manifest: &CommitmentManifest, values: &[E]) -> Option<Self> {
        (values.len() == Self::len(manifest)).then(|| Self(values.to_vec()))
    }
}

/// The seven claims `m_k(pi_m)` on the digits of the multiplicity column at the table
/// tree's final point: the verifier checks
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
        let manifest = CommitmentManifest::for_batch(8).unwrap();
        let exact = CommitmentManifest::for_relation(8, crate::manifest::Relation::HExact).unwrap();
        assert_eq!((Stage2Claims::len(&manifest), Stage2Claims::len(&exact)), (10, 9));
        assert_eq!((Stage3Claims::len(&manifest), Stage3Claims::len(&exact)), (35, 36));
        for m in [&manifest, &exact] {
            let s2 = Stage2Claims::from_slice(m, &point(&mut rng, Stage2Claims::len(m))).unwrap();
            assert_eq!(Stage2Claims::from_slice(m, &s2.to_vec()), Some(s2.clone()));
        }
        assert!(Stage2Claims::from_slice(&exact, &point(&mut rng, 8)).is_none());
        let s3 = Stage3Claims::from_slice(&manifest, &point(&mut rng, 35)).unwrap();
        assert_eq!(Stage3Claims::group_offset(&manifest, GroupRole::Level2), 18);
        // G_3 has one claim (its zero polynomials have none), then G_k's four.
        assert_eq!(s3.group(&manifest, GroupRole::Level3), &s3.0[30..31]);
        assert_eq!(s3.group(&manifest, GroupRole::Proj), &s3.0[31..]);
        assert_eq!(s3.column(&manifest, Column::K_2), s3.0[33]);
        assert_eq!(s3.column(&manifest, Column::K_3), s3.0[34]);
        assert_eq!(s3.column(&manifest, Column::Parity2(1)), s3.0[19]);
        assert_eq!(s3.column(&manifest, Column::W1Bit(0, 0)), s3.0[6]);
        assert!(Stage3Claims::from_slice(&manifest, &s3.0[..34]).is_none());
        // H-exact: a_z is the last main-domain claim, every later group moves by one.
        let x3 = Stage3Claims::from_slice(&exact, &point(&mut rng, 36)).unwrap();
        assert_eq!(Stage3Claims::group_offset(&exact, GroupRole::Level2), 19);
        assert_eq!(x3.column(&exact, Column::A_Z), x3.0[12]);
        assert_eq!(x3.column(&exact, Column::W1Bit(0, 0)), x3.0[6]);
        assert!(Stage3Claims::from_slice(&exact, &x3.0[..34]).is_none());
        let digits = MultiplicityClaims([3, -2, 0, 1, 0, 0, 7].map(E::from_i64));
        assert_eq!(digits.compose(), E::from_i64(3 - 32 + 4096 + 7 * (1 << 24)));
        assert_eq!(carry_value(E::from_i64(-(1 << 15))), E::from_i64(0));

        // Under HashField::Fp: y1 and y2 are empty in Stage1Claims, b is empty in Stage2Claims,
        // and keccak holds the non-public Keccak lane claims in spec.lane_ids() order.
        let fp = CommitmentManifest::for_batch_fp(8).unwrap();
        assert_eq!(Kappa1Claims::len(&fp), 0);
        assert_eq!(Stage1Claims::y_len(&fp), 0);
        assert_eq!(Stage1Claims::keccak_len(&fp), 740);
        assert_eq!(Stage1Claims::len(&fp), 6 + 740);
        let fp_s1 = Stage1Claims::from_slice_for(&fp, &point(&mut rng, Stage1Claims::len(&fp))).unwrap();
        assert!(fp_s1.y1.is_empty() && fp_s1.y2.is_empty());
        assert_eq!(fp_s1.b.len(), 6);
        assert_eq!(fp_s1.keccak.len(), 740);
        assert_eq!(Stage1Claims::from_slice_for(&fp, &fp_s1.to_vec()), Some(fp_s1));

        assert_eq!(Stage2Claims::b_len(&fp), 0);
        assert_eq!(Stage2Claims::keccak_len(&fp), 516);
        assert_eq!(Stage2Claims::len(&fp), 4 + 516);
        let fp_s2 = Stage2Claims::from_slice_for(&fp, &point(&mut rng, Stage2Claims::len(&fp))).unwrap();
        assert!(fp_s2.b.is_empty() && fp_s2.fips.is_empty());
        assert_eq!(fp_s2.squares.len(), 4);
        assert_eq!(fp_s2.keccak.len(), 516);
        assert_eq!(Stage2Claims::from_slice_for(&fp, &fp_s2.to_vec()), Some(fp_s2));

        // Closed-form Keccak claim counts match the deduplicated stage spec lanes for every set
        // and relation.
        use crate::manifest::Relation;
        use agg_witness::params::Set;
        for (set, relation) in [Set::S44, Set::S65, Set::S87]
            .into_iter()
            .flat_map(|s| [Relation::Hu, Relation::HExact, Relation::HFips].map(|r| (s, r)))
        {
            let Ok(m) = CommitmentManifest::for_set_fp(set, 256, relation) else {
                continue;
            };
            assert_eq!(
                Stage1Claims::keccak_len(&m),
                crate::stage1::keccak_claim_lanes(&crate::stage1::spec(&m)).len()
            );
            assert_eq!(
                Stage2Claims::keccak_len(&m),
                crate::stage1::keccak_claim_lanes(&crate::stage2::spec(&m)).len()
            );
        }
    }
}
