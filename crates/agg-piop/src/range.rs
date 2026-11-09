//! Range proofs of the small digits: the per-cell
//! map of the virtual table `P_Z = Z (Z + 1)` and the order of the leaf and link instances
//! of `U_hi` (base 6, relation H-exact), the middle digits of ML-DSA-65 and ML-DSA-87 (base
//! 8) and the projection digits `K_2`, `K_3` (base 8, and `K_4`, base 4). Every other range is the lookup into `T_16`
//! (`lookup`).
//!
//! The map is exact over the integers for every `i8` cell, so the lifted table equals the
//! polynomial of the lifted digit; the leaf predicates are the [`crate::kernel::LeafPoly`]s
//! `Q_4`, `Q_6` of Aerie (`Q_6 = Q_8` without its last root).

use std::sync::OnceLock;

use crate::kernel::digit_map;
use crate::manifest::{Column, CommitmentManifest, GroupRole};

/// `z -> z (z + 1)`.
pub fn square_map() -> &'static [i64; 256] {
    static MAP: OnceLock<[i64; 256]> = OnceLock::new();
    MAP.get_or_init(|| digit_map(|z| z * (z + 1)))
}

#[cfg(test)]
fn poly(roots: &[i64], p: i64) -> i64 {
    roots.iter().fold(1, |acc, &root| acc * (p - root))
}

/// `z -> Q_16^lo(z (z + 1))` (kernel tests: an 8-class map).
#[cfg(test)]
pub(crate) fn lo_map() -> &'static [i64; 256] {
    static MAP: OnceLock<[i64; 256]> = OnceLock::new();
    MAP.get_or_init(|| digit_map(|z| poly(crate::kernel::LeafPoly::Q8.roots(), z * (z + 1))))
}

/// `z -> Q_16^hi(z (z + 1))` (kernel tests: an 8-class map).
#[cfg(test)]
pub(crate) fn hi_map() -> &'static [i64; 256] {
    static MAP: OnceLock<[i64; 256]> = OnceLock::new();
    MAP.get_or_init(|| digit_map(|z| poly(crate::kernel::LeafPoly::Q16Hi.roots(), z * (z + 1))))
}

/// A leaf-checked digit with its position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Digit {
    /// The group (`G_main4`, or `G_k` for `K_2`).
    pub group: GroupRole,
    /// Polynomial position in the group.
    pub position: usize,
    /// The column.
    pub column: Column,
    /// The base (4 or 6).
    pub base: u32,
}

/// The leaf-checked digits (`U_hi`, `K_2`, `K_3` under H-exact, `K_2`, `K_3` under H-FIPS) in
/// the order of the leaf and link instances.
#[must_use]
pub fn digits(manifest: &CommitmentManifest) -> Vec<Digit> {
    manifest
        .leaf_digits()
        .into_iter()
        .map(|column| {
            let (group, position) = manifest.locate(column);
            Digit {
                group,
                position,
                column,
                base: column.base().expect("a digit"),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::virtual_polys::digit_square;
    use crate::OpeningField as E;
    use jolt_field::Ring;

    #[test]
    fn square_map_agrees_with_the_field_polynomial_for_every_cell() {
        for byte in 0..256_usize {
            let z = E::from_i64(i64::from(byte as u8 as i8));
            assert_eq!(E::from_i64(square_map()[byte]), digit_square(z));
        }
    }

    #[test]
    fn leaf_digits_are_u_hi_and_the_projection_digits() {
        let shape = |relation| -> Vec<(String, u32, GroupRole, usize)> {
            let manifest = CommitmentManifest::for_relation(8, relation).unwrap();
            digits(&manifest)
                .iter()
                .map(|d| (d.column.name(), d.base, d.group, d.position))
                .collect()
        };
        let k = [("k_2".to_owned(), 8, GroupRole::Proj, 2), ("k_3".to_owned(), 8, GroupRole::Proj, 3)];
        let mut exact = vec![("u_hi".to_owned(), 6, GroupRole::Main, 1)];
        exact.extend(k.clone());
        assert_eq!(shape(crate::manifest::Relation::HExact), exact);
        assert_eq!(shape(crate::manifest::Relation::HFips), k);
    }
}
