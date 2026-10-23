//! Source-table construction checks shapes and honest alphabets against the manifest, the
//! honest bridge builder aborts on heavy rows, and the committer rejects malformed tables
//! before Akita.

mod common;

use agg_pcs_config::commit::CommitError;
use agg_piop::manifest::{Column, CommitmentManifest, DigitValue, GroupRole};
use agg_piop::pcs::{PcsError, TableCommitter};
use agg_piop::source_tables::{BridgeGroupTables, MainTables, SourceTableError};
use agg_witness::bridge::{BridgeError, BridgeMatrices, BridgeTables, SparseMatrix};
use agg_witness::instance::PublicInstance;
use agg_witness::tables::{BatchTables, Layout};

fn batch_tables(records: &[agg_witness::gen::Record]) -> (PublicInstance, BatchTables) {
    let instance = PublicInstance::from_records(records).unwrap();
    let signatures: Vec<&[u8]> = records.iter().map(|r| r.sig.as_slice()).collect();
    let tables = BatchTables::build(Layout::H, &instance, &signatures).unwrap();
    (instance, tables)
}

#[test]
fn main_tables_reject_wrong_shapes_and_alphabets() {
    let records = common::records(0x77, 15, 5);
    let (_, tables) = batch_tables(&records);
    let manifest = CommitmentManifest::for_batch(20).unwrap();
    let main = MainTables::from_batch(&manifest, tables.clone()).expect("honest tables");
    assert_eq!(main.columns().len(), 23);
    assert_eq!(main.geometry().n_star, 32);

    // Same N* but another live count.
    let other = CommitmentManifest::for_batch(21).unwrap();
    assert_eq!(
        MainTables::from_batch(&other, tables.clone()).unwrap_err(),
        SourceTableError::Records {
            expected: 21,
            actual: 20
        }
    );
    // A z digit outside [-8, 7], a non-Boolean bit, a base-4 top digit outside [-2, 1].
    let z0 = manifest
        .locate(Column::Digit {
            value: DigitValue::Z,
            digit: 0,
        })
        .1;
    let b3 = manifest.locate(Column::W1Bit(3)).1;
    let z4 = manifest
        .locate(Column::Digit {
            value: DigitValue::Z,
            digit: 4,
        })
        .1;
    for (column, cell, value) in [(z0, 1234, 8_i8), (b3, 77, 2), (z4, 5, 2)] {
        let mut bad = tables.clone();
        bad.columns[column][cell] = value;
        assert_eq!(
            MainTables::from_batch(&manifest, bad).unwrap_err(),
            SourceTableError::Alphabet {
                group: GroupRole::Main,
                column,
                cell,
                value
            }
        );
    }
    let mut short = tables.clone();
    short.columns.pop();
    assert!(matches!(
        MainTables::from_batch(&manifest, short),
        Err(SourceTableError::ColumnCount {
            expected: 23,
            actual: 22,
            ..
        })
    ));
    let mut truncated = tables;
    truncated.columns[4].pop();
    assert!(matches!(
        MainTables::from_batch(&manifest, truncated),
        Err(SourceTableError::ColumnLength { column: 4, .. })
    ));
}

#[test]
fn bridge_tables_check_shapes_and_abort_on_heavy_rows() {
    let records = common::records(0x78, 12, 4);
    let (_, tables) = batch_tables(&records);
    let manifest = CommitmentManifest::for_batch(16).unwrap();
    let main = MainTables::from_batch(&manifest, tables).unwrap();
    let mut matrices = BridgeMatrices::random_for_tests(0x78);
    let honest = BridgeGroupTables::build(&manifest, &main, &matrices).expect("honest bridge");
    assert_eq!(honest.public_y().len(), 1 << 12);
    assert_eq!(honest.segment_y(0), honest.public_y());
    let witness: BridgeTables = honest.to_witness_tables();
    assert_eq!(
        BridgeGroupTables::from_tables(&manifest, witness.clone()).unwrap(),
        honest
    );

    let mut bad_y = witness.clone();
    bad_y.y[9] = 2;
    assert_eq!(
        BridgeGroupTables::from_tables(&manifest, bad_y).unwrap_err(),
        SourceTableError::PublicY
    );
    let mut bad_g3 = witness.clone();
    bad_g3.g3.pop();
    assert!(matches!(
        BridgeGroupTables::from_tables(&manifest, bad_g3),
        Err(SourceTableError::ColumnCount {
            group: GroupRole::Level3,
            ..
        })
    ));
    let mut bad_parity = witness;
    bad_parity.g2[0][3] = -1;
    assert!(matches!(
        BridgeGroupTables::from_tables(&manifest, bad_parity),
        Err(SourceTableError::Alphabet {
            group: GroupRole::Level2,
            column: 0,
            cell: 3,
            value: -1
        })
    ));

    // Row 0 of M_3 in 600 columns: weight above 511, the honest prover aborts (Section 9.2).
    let m3 = &matrices.m3;
    let mut entries = m3.entries.clone();
    for column in 0..600 {
        let rows = &mut entries[column * m3.k..(column + 1) * m3.k];
        if !rows.contains(&0) {
            rows[0] = 0;
        }
    }
    matrices.m3 = SparseMatrix::new(m3.rows, m3.cols, m3.k, entries);
    assert_eq!(
        BridgeGroupTables::build(&manifest, &main, &matrices).unwrap_err(),
        SourceTableError::Bridge(BridgeError::RowWeight)
    );
}

#[test]
fn committer_rejects_malformed_tables_before_akita() {
    let records = common::records(0x79, 8, 0);
    let (_, tables) = batch_tables(&records);
    let manifest = CommitmentManifest::for_batch(8).unwrap();
    let committer = TableCommitter::new(manifest.clone()).expect("setup");
    let mut main = MainTables::from_batch(&manifest, tables).unwrap();
    main.columns_mut()[7].truncate(100);
    assert!(matches!(
        committer.commit_round1(&main),
        Err(PcsError::Commit(CommitError::TableLength {
            group: GroupRole::Main,
            actual: 100,
            ..
        }))
    ));
}
