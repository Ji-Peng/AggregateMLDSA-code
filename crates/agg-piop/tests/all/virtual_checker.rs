//! The committed tables and the virtual-table evaluators agree with the integer relation on
//! the Boolean domain: the
//! recomposed `z, u, v` equal the derived witnesses, the ring residual vanishes, the bridge
//! identities hold over the integers with the balanced carries, and single-cell mutations
//! move exactly the expected entries.

use crate::common;

use agg_bridge::parity::{check, ParityWitness};
use agg_bridge::Shape;
use agg_piop::manifest::{Column, DigitValue, GroupRole, LOOKUP_SHIFT};
use agg_piop::source_tables::{BridgeGroupTables, Cells, MainTables, SourceTables};
use agg_piop::virtual_polys::{compose_table, parity1_tables, residual_table, w1_table};
use agg_piop::OpeningField;
use agg_witness::derive::derive;
use agg_witness::padding::main_index;
use agg_witness::params::{K, L, N};
use crate::common::Batch;
use jolt_field::Zero;

fn small_batch() -> Batch {
    // 50 live records pad to N* = 64.
    Batch::build(common::records(0x51, 40, 10), 0x51)
}

fn nonzero(table: &[OpeningField]) -> Vec<usize> {
    (0..table.len())
        .filter(|&x| table[x] != OpeningField::zero())
        .collect()
}

fn parity_witness(batch: &Batch, main: &MainTables, bridge: &BridgeGroupTables) -> ParityWitness {
    let tables = SourceTables {
        manifest: &batch.manifest,
        main,
        bridge,
    };
    let widen = |cells: &[i8]| cells.iter().map(|&c| i64::from(c)).collect::<Vec<i64>>();
    ParityWitness {
        y1: parity1_tables(tables, &batch.matrices.m1).to_vec(),
        kappa1: bridge.carry_values(&batch.manifest, 1),
        y2: bridge.parities().iter().map(|p| widen(p)).collect(),
        kappa2: bridge.carry_values(&batch.manifest, 2),
        y: bridge.public_y().iter().map(|&b| i64::from(b)).collect(),
        kappa3: bridge.carry_values(&batch.manifest, 3).remove(0),
    }
}

#[test]
fn honest_l16_tables_satisfy_the_integer_relation() {
    let batch = small_batch();
    let manifest = &batch.manifest;
    // z, u, v recomposed from their two digits equal the derived witnesses; padded rows
    // recompose to zero.
    for i in 0..manifest.padded_records() {
        let (z, u, v) = batch.main.record_values(manifest, i);
        if i < manifest.records() {
            let record = &batch.instance.records[i];
            let witness = derive(&record.key, &record.mu, &batch.records[i].sig).expect("valid");
            assert_eq!(z, witness.z, "record {i}");
            assert_eq!(u, witness.u);
            assert_eq!(v, witness.v);
        } else {
            assert_eq!(z, [[0; N]; L]);
            assert_eq!((u, v), ([[0; N]; K], [[0; N]; K]));
        }
    }
    // The ring residual vanishes on the whole domain.
    let residual = residual_table(manifest, &batch.instance, &batch.main);
    assert!(nonzero(&residual).is_empty());
    // The tables of z and the stored digits: z = bz_lo + 2^16 bz_hi + 130,915 a_z - 98,225.
    let tables = batch.tables();
    let z = compose_table(&DigitValue::Zx.layout(), &tables.digits(DigitValue::Zx));
    let (lo, hi, a) = (tables.column(Column::ZX_LO), tables.column(Column::ZX_HI), tables.column(Column::A_Z));
    for x in [0, 1, 1023, 4096 + 17] {
        assert_eq!(z[x], lo.value(x) + (hi.value(x) << 16) + 130_915 * a.value(x) - 98_225);
    }
    // The bridge identities over the integers, with kappa = K + 2^15.
    let witness = parity_witness(&batch, &batch.main, &batch.bridge);
    let lanes = batch.main.bit_lanes(manifest);
    check(&Shape::PRODUCTION, &lanes, &batch.matrices, &witness).expect("bridge identities");
    assert!(witness.y1.iter().flatten().all(|&y| y == 0 || y == 1));
    // Every carry is small (honest range [0, 1023]) and the padding slots look up 0.
    for level in 1..=3 {
        assert!(batch
            .bridge
            .carry_values(manifest, level)
            .iter()
            .flatten()
            .all(|&k| (0..1024).contains(&k)));
        let cells = manifest.group(agg_piop::manifest::STACK_DOMAINS[level]).logical_cells();
        let stacked = batch.bridge.carries(level);
        // Level 2 under H-FIPS also holds the hint slack u_h in slot 6.
        let real = match level {
            3 => 1,
            2 if manifest.is_fips() => 7,
            _ => 6,
        };
        assert!(stacked[real * cells..].iter().all(|&k| k == -LOOKUP_SHIFT));
    }
    // w_1' from the bits is in [0, 44).
    assert!(w1_table(&lanes).iter().all(|&w| (0..44).contains(&w)));
}

#[test]
fn single_cell_mutations_move_the_expected_entries() {
    let batch = small_batch();
    let manifest = &batch.manifest;
    let x = main_index(5, 1, 3);
    // V_lo + 1 moves the residual by -q at x only.
    let mut main = batch.main.clone();
    *main.wide_cell_mut(manifest, Column::V_LO, x) += 1;
    let residual = residual_table(manifest, &batch.instance, &main);
    assert_eq!(nonzero(&residual), vec![x]);
    // bz_lo + 1 moves (A z) on the record's whole row block (a negacyclic product).
    let mut main = batch.main.clone();
    let x = main_index(9, 2, 4);
    *main.wide_cell_mut(manifest, Column::ZX_LO, x) += 1;
    let moved = nonzero(&residual_table(manifest, &batch.instance, &main));
    assert!(!moved.is_empty());
    assert!(moved.iter().all(|&y| y >> 10 == 4), "only record 4 changes");
    // A flipped bit breaks the level-1 identity of the stale round-2 tables.
    let mut main = batch.main.clone();
    let (role, b1) = manifest.locate(Column::W1Bit(1, 0));
    assert_eq!(role, GroupRole::Main);
    main.columns_mut()[b1][x] ^= 1;
    let stale = parity_witness(&batch, &main, &batch.bridge);
    let lanes = main.bit_lanes(manifest);
    let y1 = &stale.y1;
    assert!(
        y1.iter().flatten().any(|&y| y != 0 && y != 1),
        "a level-1 parity leaves {{0, 1}}"
    );
    let _ = check(&Shape::PRODUCTION, &lanes, &batch.matrices, &stale);
    // A carry + 1 moves y1 by -2 in that cell.
    let mut bridge = batch.bridge.clone();
    bridge.carries_mut(1)[7] += 1;
    let moved = parity_witness(&batch, &batch.main, &bridge);
    let honest = parity_witness(&batch, &batch.main, &batch.bridge);
    assert_eq!(moved.y1[0][7], honest.y1[0][7] - 2);
    // The wide cells read back as committed.
    assert!(matches!(batch.tables().column(Column::V_HI), Cells::Wide(_)));
}
