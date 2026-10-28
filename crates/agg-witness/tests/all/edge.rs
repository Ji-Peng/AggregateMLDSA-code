//! Edge cases: the Decompose edge (exhaustively and on pinned signatures), |z| = B_z.

#![allow(clippy::needless_range_loop)]

use crate::common;

use agg_witness::derive::{derive, reencode_h, usehint_residual_max};
use agg_witness::digits::{ValueForm, Z_BIT_SLACK};
use agg_witness::gen::{self, crate_verify};
use agg_witness::hfips::{check_batch, HFipsRecord};
use agg_witness::instance::PublicInstance;
use agg_witness::native::{decompose, inf_norm, mod_q, mu};
use agg_witness::params::*;

#[test]
fn usehint_residual_lemma_is_exact() {
    // max over Z_q x {0, 1} of |ctr_q(r - 2 gamma2 UseHint(h, r))| is exactly B_u.
    assert_eq!(usehint_residual_max(), B_U);
}

fn edge_record(t: (u64, u64, u64)) -> gen::Record {
    gen::record_hedged(t.0, t.1, t.2)
}

/// Every record derives, its H-FIPS witness passes the checker, and re-encoding reproduces the
/// signature, which the crate accepts.
fn check_records(records: &[gen::Record]) {
    let inst = PublicInstance::from_records(records).unwrap();
    let wits = common::witnesses(&inst, records);
    let hf: Vec<HFipsRecord> = wits
        .iter()
        .map(|w| HFipsRecord::from_witness(w).expect("H-FIPS witness"))
        .collect();
    let rep = check_batch(&inst, &hf);
    assert!(rep.is_ok(), "{:?}", rep.failures);
    for (i, (r, w)) in records.iter().zip(&wits).enumerate() {
        let bytes = reencode_h(&inst.records[i].c_tilde, &w.z, &w.u, &w.w1).unwrap();
        assert_eq!(bytes, r.sig);
        assert!(crate_verify(&r.pk, &r.msg, &bytes));
    }
}

#[test]
fn pinned_decompose_edge_residue() {
    let rec = edge_record(common::EDGE_RESIDUE);
    assert!(crate_verify(&rec.pk, &rec.msg, &rec.sig));
    let inst = PublicInstance::from_records(std::slice::from_ref(&rec)).unwrap();
    let key = &inst.records[0].key;
    let w = derive(key, &mu(&key.tr, &rec.msg, &[]), &rec.sig).unwrap();
    let mut found = None;
    for r in 0..K {
        for j in 0..N {
            if mod_q(w.w[r][j]) == Q - GAMMA2 {
                found = Some((r, j));
            }
        }
    }
    let (r, j) = found.expect("pinned record hits the residue -gamma2");
    assert_eq!(decompose(w.w[r][j]), (0, -GAMMA2));
    // The ordinary edge range r+ in [q - gamma2, q - 1] is common: many positions per signature.
    assert!(common::edge_positions(&w) > 1);
    let mut batch = gen::records(400, 3);
    batch.insert(1, rec);
    check_records(&batch);
}

#[test]
fn pinned_z_at_the_honest_bound() {
    let rec = edge_record(common::EDGE_ZMAX);
    let inst = PublicInstance::from_records(std::slice::from_ref(&rec)).unwrap();
    let key = &inst.records[0].key;
    let w = derive(key, &mu(&key.tr, &rec.msg, &[]), &rec.sig).unwrap();
    assert_eq!(inf_norm(&w.z), B_Z);
    let mut batch = gen::records(401, 2);
    batch.push(rec);
    check_records(&batch);
    // The bit-slack form at the bound: z = +-B_z is representable exactly, B_z + 1 is not.
    let form = ValueForm::from(Z_BIT_SLACK);
    for z in [-B_Z, B_Z] {
        assert!(form.in_alphabets(&form.decompose(z)));
    }
    assert!(!form.in_alphabets(&form.decompose(B_Z + 1)));
}
