//! Range proofs by digit decomposition (spec Section 4.3): the per-cell maps of the
//! virtual digit tables `P_Z = Z (Z + 1)`, `A_Z^lo = Q_16^lo(P_Z)`, `A_Z^hi = Q_16^hi(P_Z)`,
//! and the canonical digit orders of the `b16_prod`, leaf and link instances.
//!
//! The maps are exact over the integers for every `i8` cell (`|Q(P(z))| < 2^57`), so the
//! lifted table equals the polynomial of the lifted digit; the leaf predicates are the
//! [`LeafPoly`]s of Aerie (3.7), (3.8).

use std::sync::OnceLock;

use crate::kernel::{digit_map, LeafPoly};
use crate::manifest::{Column, CommitmentManifest, GroupRole};

fn poly(roots: &[i64], p: i64) -> i64 {
    roots.iter().fold(1, |acc, &root| acc * (p - root))
}

/// `z -> z (z + 1)`.
pub fn square_map() -> &'static [i64; 256] {
    static MAP: OnceLock<[i64; 256]> = OnceLock::new();
    MAP.get_or_init(|| digit_map(|z| z * (z + 1)))
}

/// `z -> Q_16^lo(z (z + 1))`.
pub fn lo_map() -> &'static [i64; 256] {
    static MAP: OnceLock<[i64; 256]> = OnceLock::new();
    MAP.get_or_init(|| digit_map(|z| poly(LeafPoly::Q8.roots(), z * (z + 1))))
}

/// `z -> Q_16^hi(z (z + 1))`.
pub fn hi_map() -> &'static [i64; 256] {
    static MAP: OnceLock<[i64; 256]> = OnceLock::new();
    MAP.get_or_init(|| digit_map(|z| poly(LeafPoly::Q16Hi.roots(), z * (z + 1))))
}

/// A committed digit with its position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Digit {
    /// The group.
    pub group: GroupRole,
    /// Polynomial position in the group.
    pub position: usize,
    /// The column.
    pub column: Column,
    /// The base.
    pub base: u32,
}

/// The 55 digits in the order of spec Section 4.1 (the order of the `b16_prod`, leaf and
/// link instances).
#[must_use]
pub fn digits(manifest: &CommitmentManifest) -> Vec<Digit> {
    manifest
        .digit_columns()
        .into_iter()
        .map(|(group, position, column)| Digit {
            group,
            position,
            column,
            base: column.base().expect("a digit"),
        })
        .collect()
}

/// The 40 base-16 digits in order.
#[must_use]
pub fn base16_digits(manifest: &CommitmentManifest) -> Vec<Digit> {
    digits(manifest)
        .into_iter()
        .filter(|d| d.base == 16)
        .collect()
}

/// The digits of `base` in order.
#[must_use]
pub fn digits_of_base(manifest: &CommitmentManifest, base: u32) -> Vec<Digit> {
    digits(manifest)
        .into_iter()
        .filter(|d| d.base == base)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::virtual_polys::{digit_square, q16_hi, q16_lo};
    use crate::OpeningField as E;
    use jolt_field::Ring;

    #[test]
    fn maps_agree_with_the_field_polynomials_for_every_cell() {
        for byte in 0..256_usize {
            let z = E::from_i64(i64::from(byte as u8 as i8));
            let p = digit_square(z);
            assert_eq!(E::from_i64(square_map()[byte]), p);
            assert_eq!(E::from_i64(lo_map()[byte]), q16_lo(p));
            assert_eq!(E::from_i64(hi_map()[byte]), q16_hi(p));
        }
    }

    #[test]
    fn digit_orders_match_the_manifest() {
        let manifest = CommitmentManifest::for_batch(8).unwrap();
        let all = digits(&manifest);
        assert_eq!(all.len(), 55);
        assert_eq!(base16_digits(&manifest).len(), 40);
        assert_eq!(digits_of_base(&manifest, 8).len(), 2);
        assert_eq!(digits_of_base(&manifest, 4).len(), 13);
        assert_eq!(all[0].column.name(), "z_0");
        assert_eq!(all[4].column.name(), "z_4");
        assert_eq!(all[4].base, 4);
        assert_eq!(all[17].column.name(), "kappa1_0_0");
        assert_eq!(all[54].column.name(), "kappa3_1");
        let names: Vec<String> = digits_of_base(&manifest, 4)
            .iter()
            .map(|d| d.column.name())
            .collect();
        assert_eq!(names[0], "z_4");
        assert_eq!(names[1], "kappa1_0_2");
        assert_eq!(names[12], "kappa2_5_2");
    }
}
