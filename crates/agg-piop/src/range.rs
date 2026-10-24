//! Range proofs of the two small digits (spec Section 4.3, lookup layout L16): the per-cell
//! map of the virtual table `P_Z = Z (Z + 1)` and the order of the leaf and link instances
//! of `Z_hi` (base 4) and `U_hi` (base 8). Every other range is the lookup into `T_16`
//! (`lookup`, `docs/lookup.md`).
//!
//! The map is exact over the integers for every `i8` cell, so the lifted table equals the
//! polynomial of the lifted digit; the leaf predicates are the [`crate::kernel::LeafPoly`]s
//! `Q_4`, `Q_8` of Aerie (3.7).

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
    /// The group (`G_main4`).
    pub group: GroupRole,
    /// Polynomial position in the group.
    pub position: usize,
    /// The column.
    pub column: Column,
    /// The base (4 or 8).
    pub base: u32,
}

/// The two leaf-checked digits `Z_hi`, `U_hi` in the order of the leaf and link instances.
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
    fn leaf_digits_are_z_hi_and_u_hi() {
        let manifest = CommitmentManifest::for_batch(8).unwrap();
        let all = digits(&manifest);
        let shape: Vec<(String, u32, GroupRole, usize)> = all
            .iter()
            .map(|d| (d.column.name(), d.base, d.group, d.position))
            .collect();
        assert_eq!(
            shape,
            [
                ("z_hi".to_owned(), 4, GroupRole::Main, 0),
                ("u_hi".to_owned(), 8, GroupRole::Main, 1)
            ]
        );
    }
}
