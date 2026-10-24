//! Dummy group tables and claims shared by `tests/commit_open.rs` and
//! `benches/commit_open.rs` (included through `#[path]`).
#![allow(dead_code)]

use agg_pcs_config::catalogs::Group;
use agg_pcs_config::commit::{evaluate_signed_table, GroupClaim};
use agg_pcs_config::preset::{BaseField, OpeningField};
use agg_pcs_config::work_plan::WorkPlan;
use jolt_field::{ExtField, Ring};

/// SplitMix64, a deterministic test source.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A value of the group's honest alphabet: Booleans for the bit and
    /// parity columns, balanced digits otherwise (every one inside `[-8, 7]`).
    pub fn digit(&mut self, base: i64) -> i8 {
        let value = (self.next() % base as u64) as i64 - base / 2;
        i8::try_from(value).expect("small digit")
    }

    pub fn field(&mut self) -> OpeningField {
        OpeningField::from_base_slice(&[
            BaseField::from_u64(self.next()),
            BaseField::from_u64(self.next()),
        ])
    }
}

/// Digit bases of each group's polynomials, spec Section 4.1 (`0` marks a
/// Boolean column).
pub fn bases(group: Group) -> Vec<i64> {
    match group {
        Group::Main => [
            16, 16, 16, 16, 4, 16, 16, 16, 16, 8, 16, 16, 16, 16, 16, 16, 8, 0, 0, 0, 0, 0, 0,
        ]
        .to_vec(),
        Group::Level1 => [16, 16, 4].repeat(6),
        Group::Level2 => {
            let mut bases = vec![0; 6];
            bases.extend([16, 16, 4].repeat(6));
            bases
        }
        Group::Level3 => vec![16, 16],
    }
}

/// Dummy tables of one group; the rows of padded signatures (`i >= live`) are
/// the canonical digits of zero, `-b/2`, and zero bits (spec Section 4.2).
pub fn dummy_tables(plan: &WorkPlan, group: Group, rng: &mut Rng) -> Vec<Vec<i8>> {
    let cells = plan.logical_cells(group);
    let bases = bases(group);
    assert_eq!(bases.len(), group.num_polynomials());
    bases
        .iter()
        .map(|&base| {
            (0..cells)
                .map(|cell| {
                    // Main-domain index order is (j, rho or s, i): the
                    // signature index is the top d_sig bits.
                    let padded = group == Group::Main && (cell >> 10) >= plan.records;
                    match (base, padded) {
                        (0, true) => 0,
                        (0, false) => (rng.next() & 1) as i8,
                        (_, true) => i8::try_from(-base / 2).unwrap(),
                        (_, false) => rng.digit(base),
                    }
                })
                .collect()
        })
        .collect()
}

pub fn claims(plan: &WorkPlan, tables: &[Vec<Vec<i8>>; 4], rng: &mut Rng) -> [GroupClaim; 4] {
    Group::ALL.map(|group| {
        let point: Vec<OpeningField> = (0..plan.logical_vars(group)).map(|_| rng.field()).collect();
        let values = tables[group.index()]
            .iter()
            .map(|table| evaluate_signed_table(table, &point))
            .collect();
        GroupClaim { point, values }
    })
}
