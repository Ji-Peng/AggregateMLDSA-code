//! Dummy group tables and claims shared by `tests/commit_open.rs` and
//! `benches/commit_open.rs` (included through `#[path]`).
#![allow(dead_code)]

use agg_pcs_config::catalogs::{Group, Relation};
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

    pub fn field(&mut self) -> OpeningField {
        OpeningField::from_base_slice(&[
            BaseField::from_u64(self.next()),
            BaseField::from_u64(self.next()),
        ])
    }
}

/// The honest alphabet of each group's polynomials (spec Section 4.1, lookup layout L16):
/// `0` marks a Boolean column, `b > 0` a balanced base-`b` digit, so a 16-bit lookup column
/// is base `2^16`, a multiplicity digit base 16. `G_main4` is `Z_hi` (base 4), `U_hi`
/// (base 6) and six bits under H-u, and `B_zhi` (bit), `D_u` (base 6), six bits and `a_z`
/// (bit) under H-exact.
pub fn bases(group: Group, relation: Relation) -> Vec<i64> {
    let wide = 1 << 16;
    match group {
        Group::Main16 => vec![wide; 4],
        Group::Main => match relation {
            Relation::Hu => [4, 6, 0, 0, 0, 0, 0, 0].to_vec(),
            Relation::HExact => [0, 6, 0, 0, 0, 0, 0, 0, 0].to_vec(),
            // bz_hi, L_hi, six bits, a_z, the sign bit (Booleans); delta, e in [-1, 1]
            // (dummy values from the base-4 alphabet, inside the width-4 admission range).
            Relation::HFips => [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4].to_vec(),
        },
        Group::Level1 => vec![wide; 6],
        Group::Level2 => {
            let mut bases = vec![0; 6];
            bases.extend([wide; 6]);
            if relation == Relation::HFips {
                bases.push(wide); // u_h on D_2
            }
            bases
        }
        Group::Level3 => vec![wide],
        Group::Mult => vec![16; 7],
    }
}

/// Dummy tables of one group; the rows of padded signatures (`i >= live`) are the
/// canonical digits of zero (`-2^15` for the 16-bit columns, `0` for the top digits and
/// bits; spec Section 4.2).
pub fn dummy_tables(plan: &WorkPlan, group: Group, rng: &mut Rng) -> Vec<Vec<i32>> {
    let cells = plan.logical_cells(group);
    let bases = bases(group, plan.relation());
    assert_eq!(bases.len(), plan.num_polynomials(group));
    let main = matches!(group, Group::Main16 | Group::Main);
    bases
        .iter()
        .map(|&base| {
            (0..cells)
                .map(|cell| {
                    // Main-domain index order is (j, rho or s, i): the
                    // signature index is the top d_sig bits.
                    let padded = main && (cell >> 10) >= plan.records;
                    match (base, padded) {
                        (0, true) => 0,
                        (0, false) => (rng.next() & 1) as i32,
                        (65_536, true) => -32_768,
                        (_, true) => 0,
                        (_, false) => (rng.next() % base as u64) as i32 - (base / 2) as i32,
                    }
                })
                .collect()
        })
        .collect()
}

pub fn claims(plan: &WorkPlan, tables: &[Vec<Vec<i32>>; 6], rng: &mut Rng) -> [GroupClaim; 6] {
    // Both main groups share one point (the stage-3 terminal), as in the protocol.
    let main_point: Vec<OpeningField> = (0..plan.logical_vars(Group::Main)).map(|_| rng.field()).collect();
    Group::ALL.map(|group| {
        let point: Vec<OpeningField> = if matches!(group, Group::Main16 | Group::Main) {
            main_point.clone()
        } else {
            (0..plan.logical_vars(group)).map(|_| rng.field()).collect()
        };
        let values = tables[group.index()]
            .iter()
            .map(|table| evaluate_signed_table(table, &point))
            .collect();
        GroupClaim { point, values }
    })
}
