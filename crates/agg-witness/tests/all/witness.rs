//! Witness derivation and re-encoding on random signatures.

#![allow(clippy::needless_range_loop)]

use crate::common;

use agg_witness::crt::az_schoolbook;
use agg_witness::derive::reencode_h;
use agg_witness::gen::crate_verify;
use agg_witness::native::{ctr_q, inf_norm, use_hint};
use agg_witness::padding::main_index;
use agg_witness::params::*;

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
        // Re-encoding reproduces the original signature, which the crate accepts.
        let bytes = reencode_h(&inst.records[i].c_tilde, &w.z, &w.u, &w.w1).expect("re-encodable");
        assert_eq!(bytes, records[i].sig, "record {i}");
        assert!(crate_verify(&records[i].pk, &records[i].msg, &bytes));
    }
    // Hints at both signs of r0 and Decompose edge positions occur in the batch.
    assert!(both_signs[0] > 0 && both_signs[1] > 0, "{both_signs:?}");
    assert!(
        edge > 0 && edge_hint > 0,
        "edge {edge}, edge with hint {edge_hint}"
    );
}

#[test]
fn padded_rows_of_the_instance_have_zero_t() {
    let records = common::mixed_records(220, 5, 0);
    let inst = agg_witness::instance::PublicInstance::from_records(&records).unwrap();
    assert_eq!((inst.geom.n, inst.geom.n_star), (5, 8));
    assert!(inst.t_table()[main_index(0, 0, 5)..]
        .iter()
        .all(|&t| t == 0));
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
