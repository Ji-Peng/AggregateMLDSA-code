//! Witness derivation, tables, checker and re-encoding on random signatures, for every layout.

#![allow(clippy::needless_range_loop)]

mod common;

use agg_witness::bridge::{self, BridgeMatrices, BridgeTables};
use agg_witness::check::{check, CheckOptions};
use agg_witness::crt::az_schoolbook;
use agg_witness::digits::*;
use agg_witness::gen::crate_verify;
use agg_witness::native::{ctr_q, inf_norm, use_hint};
use agg_witness::padding::main_index;
use agg_witness::params::*;
use agg_witness::tables::{zero_record_cells, BatchTables, Layout};

fn run_layout(layout: Layout, seed: u64) {
    let records = common::mixed_records(seed, 100, 28);
    assert!(records.len() >= 100);
    let (inst, tables) = common::batch(layout, &records);
    assert_eq!(tables.geom.n_star, 128);
    let map = tables.map();
    assert_eq!(tables.columns.len(), map.len());
    let rep = common::report(&inst, &tables);
    assert!(rep.is_ok(), "{}: {:?}", layout.name(), rep.failures);
    // Re-encoding every live record reproduces the original signature, which the crate accepts.
    for (i, r) in records.iter().enumerate() {
        let bytes = tables
            .reencode(i, &inst.records[i].c_tilde)
            .expect("re-encodable");
        assert_eq!(bytes, r.sig, "{} record {i}", layout.name());
        assert!(crate_verify(&r.pk, &r.msg, &bytes));
    }
    // Serialization roundtrip.
    let ser = tables.to_bytes();
    assert_eq!(BatchTables::from_bytes(&ser).unwrap(), tables);
    assert!(BatchTables::from_bytes(&ser[..ser.len() - 1]).is_err());
    let mut bad = ser.clone();
    bad[0] ^= 1;
    assert!(BatchTables::from_bytes(&bad).is_err());
    let mut extra = ser;
    extra.push(0);
    assert!(BatchTables::from_bytes(&extra).is_err());
}

#[test]
fn layout_h_random_signatures() {
    run_layout(Layout::H, 200);
}

#[cfg(feature = "h-exact")]
#[test]
fn layout_h_exact_random_signatures() {
    run_layout(Layout::HExact, 201);
}

#[cfg(feature = "fips")]
#[test]
fn layout_fips_random_signatures() {
    run_layout(Layout::Fips, 202);
}

#[test]
fn derived_values_satisfy_relation_h() {
    let records = common::mixed_records(210, 100, 28);
    let inst = agg_witness::instance::PublicInstance::from_records(&records).unwrap();
    let wits = common::witnesses(&inst, &records);
    let mut both_signs = [0usize; 2];
    let mut edge = 0usize;
    let mut edge_hint = 0usize;
    for (i, w) in wits.iter().enumerate() {
        let key = &inst.records[i].key;
        assert!(w.accepted);
        assert!(inf_norm(&w.z) <= B_Z);
        assert!(inf_norm(&w.u) <= B_U);
        assert!(inf_norm(&w.v) <= B_V);
        assert_eq!(w.t, inst.records[i].t);
        // (A z) by CRT equals the schoolbook product.
        if i % 16 == 0 {
            let sb = az_schoolbook(&key.a, &w.z);
            for r in 0..K {
                for j in 0..N {
                    assert_eq!(i128::from(w.az[r][j]), sb[r][j]);
                }
            }
        }
        for r in 0..K {
            for j in 0..N {
                let w1 = i64::from(w.w1[r][j]);
                assert!((0..M_W).contains(&w1));
                // The ring identity over Z with the spec's centering.
                assert_eq!(
                    w.az[r][j] - w.t[r][j] - 2 * GAMMA2 * w1 - w.u[r][j] - Q * w.v[r][j],
                    0
                );
                assert_eq!(w.w[r][j], ctr_q(w.az[r][j] - w.t[r][j]));
                assert_eq!(w.u[r][j], ctr_q(w.w[r][j] - 2 * GAMMA2 * w1));
                assert_eq!(use_hint(w.h[r][j] == 1, w.w[r][j]), w1);
                // Bits.
                assert_eq!(
                    (0..6).map(|t| i64::from(w.bit(t, j, r)) << t).sum::<i64>(),
                    w1
                );
                if w.h[r][j] == 1 {
                    let (_, r0) = agg_witness::native::decompose(w.w[r][j]);
                    both_signs[usize::from(r0 > 0)] += 1;
                }
                if agg_witness::native::mod_q(w.w[r][j]) >= Q - GAMMA2 {
                    edge += 1;
                    edge_hint += usize::from(w.h[r][j]);
                }
            }
        }
        // Digits of every value lie in the spec alphabets and recompose.
        for p in w.z.iter().flatten() {
            let d = Z_DIGITS.decompose(*p);
            assert!(Z_DIGITS.in_alphabets(&d));
            assert_eq!(Z_DIGITS.compose(&d), *p);
        }
        for p in w.u.iter().flatten() {
            let d = U_STEP.decompose(*p);
            assert!(U_STEP.in_alphabets(&d));
            assert_eq!(U_STEP.compose(&d), *p);
        }
        for p in w.v.iter().flatten() {
            assert!(V_DIGITS.in_alphabets(&V_DIGITS.decompose(*p)));
        }
    }
    // Hints at both signs of r0 and Decompose edge positions occur in the batch.
    assert!(both_signs[0] > 0 && both_signs[1] > 0, "{both_signs:?}");
    assert!(
        edge > 0 && edge_hint > 0,
        "edge {edge}, edge with hint {edge_hint}"
    );
}

#[test]
fn padded_rows_hold_canonical_digits_of_zero() {
    let records = common::mixed_records(220, 5, 0);
    let (inst, tables) = common::batch(Layout::H, &records);
    assert_eq!((tables.geom.n, tables.geom.n_star), (5, 8));
    let map = tables.map();
    for i in 5..8 {
        for pos in [0usize, 511, 1023] {
            let x = i * POSITIONS + pos;
            let z: Vec<i8> = (0..5).map(|k| tables.columns[map.z + k][x]).collect();
            assert_eq!(z, vec![-8, -8, -8, -8, 0]);
            let u: Vec<i8> = (0..5)
                .map(|k| tables.columns[map.u.unwrap() + k][x])
                .collect();
            // H-u: u = 0 is B_ulo = 1,228 (0x4CC), H_u = 3 (stored 0).
            assert_eq!(u, vec![4, 4, -4, -8, 0]);
            let v: Vec<i8> = (0..7).map(|k| tables.columns[map.v + k][x]).collect();
            assert_eq!(v, vec![-8, -8, -8, -8, -8, -8, 0]);
            assert!((0..6).all(|t| tables.columns[map.b + t][x] == 0));
        }
    }
    assert!(inst.t_table()[main_index(0, 0, 5)..]
        .iter()
        .all(|&t| t == 0));
    for layout in Layout::all() {
        let (main, _) = zero_record_cells(layout);
        assert_eq!(main.len(), layout.columns().n_main * POSITIONS);
    }
}

#[test]
fn bridge_identities_hold_on_honest_witness() {
    let records = common::mixed_records(230, 40, 10);
    let (inst, tables) = common::batch(Layout::H, &records);
    let mats = BridgeMatrices::random_for_tests(0xB41D6E);
    let bw = bridge::compute(&tables.geom, &tables.bit_lanes(), &mats).unwrap();
    let bt = BridgeTables::from_witness(&bw).unwrap();
    assert_eq!(bt.g1.len(), 18);
    assert_eq!(bt.g2.len(), 24);
    assert_eq!(bt.g3.len(), 2);
    let rep = check(&inst, &tables, Some((&mats, &bt)), CheckOptions::default());
    assert!(rep.is_ok(), "{:?}", rep.failures);
}

#[test]
fn records_sharing_keys_reuse_expanded_keys() {
    let records = agg_witness::gen::records_shared_keys(240, 12, 3);
    let inst = agg_witness::instance::PublicInstance::from_records(&records).unwrap();
    assert!(std::sync::Arc::ptr_eq(
        &inst.records[0].key,
        &inst.records[3].key
    ));
    assert!(!std::sync::Arc::ptr_eq(
        &inst.records[0].key,
        &inst.records[1].key
    ));
}
