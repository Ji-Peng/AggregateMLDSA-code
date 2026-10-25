//! Source-table construction checks shapes and honest alphabets against the manifest
//! (lookup layout L16), the honest bridge builder aborts on heavy rows and counts the
//! multiplicities of `T_16`, and the committer rejects malformed tables before Akita.

mod common;

use agg_piop::lookup::compose_multiplicity;
use agg_piop::manifest::{Column, CommitmentManifest, GroupRole, LOOKUP_SHIFT};
use agg_piop::pcs::{PcsError, TableCommitter};
use agg_piop::source_tables::{BridgeGroupTables, Cells, MainTables, RecordValues, SourceTableError};
use agg_witness::bridge::{BridgeError, BridgeMatrices, SparseMatrix};
use agg_witness::derive::derive;
use agg_witness::instance::PublicInstance;

fn instance_and_main(records: &[agg_witness::gen::Record], manifest: &CommitmentManifest) -> (PublicInstance, MainTables) {
    let instance = PublicInstance::from_records(records).unwrap();
    let signatures: Vec<&[u8]> = records.iter().map(|r| r.sig.as_slice()).collect();
    let main = MainTables::build(manifest, &instance, &signatures).unwrap();
    (instance, main)
}

#[test]
fn main_tables_encode_l16_and_reject_wrong_shapes_and_alphabets() {
    let records = common::records(0x77, 15, 5);
    let manifest = CommitmentManifest::for_batch(20).unwrap();
    let (instance, main) = instance_and_main(&records, &manifest);
    assert_eq!(main.columns().len(), 8);
    assert_eq!(main.wide().len(), 4 * (32 << 10));
    assert_eq!(main.geometry().n_star, 32);
    // Padded rows hold the digits of zero: Z_lo = V_lo = -2^15, U_lo = -31,540 (relation
    // H-u: B_ulo = 1,228, H_u = 3), the rest 0.
    let padded = 25 << 10;
    for column in [Column::Z_LO, Column::V_LO] {
        assert_eq!(main.column(&manifest, column).value(padded), -i64::from(LOOKUP_SHIFT));
    }
    assert_eq!(main.column(&manifest, Column::U_LO).value(padded), -31_540);
    for column in [Column::V_HI, Column::Z_HI, Column::U_HI, Column::W1Bit(0)] {
        assert_eq!(main.column(&manifest, column).value(padded), 0);
    }
    // The same tables from the derived values.
    let values: Vec<RecordValues> = instance
        .records
        .iter()
        .zip(&records)
        .map(|(record, signed)| {
            let w = derive(&record.key, &record.mu, &signed.sig).unwrap();
            RecordValues {
                z: w.z,
                u: w.u,
                v: w.v,
                w1: w.w1,
            }
        })
        .collect();
    assert_eq!(MainTables::from_values(&manifest, &values).unwrap(), main);
    // Same N* but another live count.
    let other = CommitmentManifest::for_batch(21).unwrap();
    assert_eq!(
        MainTables::from_values(&other, &values).unwrap_err(),
        SourceTableError::Records {
            expected: 21,
            actual: 20
        }
    );
    // A value outside its L16 range is an honest abort.
    let mut wide = values.clone();
    wide[3].z[1][7] = 1 << 17;
    assert!(matches!(
        MainTables::from_values(&manifest, &wide),
        Err(SourceTableError::Encode(_))
    ));
    // Relation H-u: u = +-190,465 encodes, +-190,466 is an honest abort.
    for (u, ok) in [(190_465, true), (-190_465, true), (190_466, false), (-190_466, false)] {
        let mut edge = values.clone();
        edge[2].u[3][9] = u;
        let built = MainTables::from_values(&manifest, &edge);
        assert_eq!(built.is_ok(), ok, "u = {u}");
        if let Ok(tables) = built {
            assert_eq!(tables.record_values(&manifest, 2).1[3][9], u);
        }
    }
    let mut huge = values;
    huge[0].v[0][0] = -(1 << 31) - 1;
    assert!(matches!(
        MainTables::from_values(&manifest, &huge),
        Err(SourceTableError::Encode(_))
    ));
    // A non-Boolean bit, a base-4 top digit outside [-2, 1], a base-6 digit outside [-3, 2],
    // a 16-bit column outside [-2^15, 2^15): the self-check names the group, column and cell.
    let b3 = manifest.locate(Column::W1Bit(3)).1;
    let z_hi = manifest.locate(Column::Z_HI).1;
    let u_hi = manifest.locate(Column::U_HI).1;
    for (column, cell, value) in [(b3, 77, 2_i8), (z_hi, 5, 2), (u_hi, 6, 3), (u_hi, 7, -4)] {
        let mut bad = main.clone();
        bad.columns_mut()[column][cell] = value;
        assert_eq!(
            bad.validate(&manifest).unwrap_err(),
            SourceTableError::Alphabet {
                group: GroupRole::Main,
                column,
                cell,
                value: i64::from(value)
            }
        );
    }
    let mut bad = main;
    *bad.wide_cell_mut(&manifest, Column::U_LO, 99) = LOOKUP_SHIFT;
    assert_eq!(
        bad.validate(&manifest).unwrap_err(),
        SourceTableError::Alphabet {
            group: GroupRole::Main16,
            column: 1,
            cell: 99,
            value: i64::from(LOOKUP_SHIFT)
        }
    );
}

#[test]
fn bridge_tables_count_the_multiplicities_and_abort_on_heavy_rows() {
    let records = common::records(0x78, 12, 4);
    let manifest = CommitmentManifest::for_batch(16).unwrap();
    let (_, main) = instance_and_main(&records, &manifest);
    let mut matrices = BridgeMatrices::random_for_tests(0x78);
    let honest = BridgeGroupTables::build(&manifest, &main, &matrices).expect("honest bridge");
    assert_eq!(honest.public_y().len(), 1 << 12);
    assert_eq!(honest.segment_y(0), honest.public_y());
    // m counts every looked-up cell, padding slots included, and its digits compose to it.
    let shape = manifest.lookup_shape();
    let m = honest.multiplicity();
    assert_eq!(m.len(), 1 << 16);
    assert_eq!(m.iter().map(|&c| i64::from(c)).sum::<i64>(), shape.lookups(0) as i64);
    let digits: Vec<Vec<i8>> = honest
        .group(&manifest, GroupRole::Mult)
        .iter()
        .map(|cells: &Cells<'_>| cells.narrow().to_vec())
        .collect();
    assert_eq!(digits.len(), 7);
    assert_eq!(
        compose_multiplicity(&digits),
        m.iter().map(|&c| i64::from(c)).collect::<Vec<_>>()
    );
    // The 16-bit carry groups and the parities.
    assert_eq!(honest.group(&manifest, GroupRole::Level1).len(), 6);
    assert_eq!(honest.group(&manifest, GroupRole::Level2).len(), 12);
    assert_eq!(honest.group(&manifest, GroupRole::Level3).len(), 1);
    honest.validate(&manifest).unwrap();
    let mut bad_y = honest.clone();
    bad_y.public_y_mut()[9] = 2;
    assert_eq!(bad_y.validate(&manifest).unwrap_err(), SourceTableError::PublicY);
    let mut bad_parity = honest.clone();
    bad_parity.parities_mut()[0][3] = -1;
    assert_eq!(
        bad_parity.validate(&manifest).unwrap_err(),
        SourceTableError::Alphabet {
            group: GroupRole::Level2,
            column: 0,
            cell: 3,
            value: -1
        }
    );
    // An out-of-table carry: the honest recount aborts.
    let mut bad_carry = honest;
    bad_carry.carries_mut(2)[0] = LOOKUP_SHIFT;
    assert!(matches!(
        bad_carry.recount(&manifest, &main),
        Err(SourceTableError::Lookup(_))
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
    let manifest = CommitmentManifest::for_batch(8).unwrap();
    let committer = TableCommitter::new(manifest.clone()).expect("setup");
    let (_, mut main) = instance_and_main(&records, &manifest);
    main.columns_mut()[7].truncate(100);
    assert!(matches!(
        committer.commit_round1(&main),
        Err(PcsError::Commit(agg_pcs_config::commit::CommitError::TableLength {
            group: GroupRole::Main,
            actual: 100,
            ..
        }))
    ));
}
