//! Edge cases: the Decompose edge (exhaustively and on pinned signatures), |z| = B_z.

#![allow(clippy::needless_range_loop)]

mod common;

use agg_witness::derive::{derive, usehint_residual_max};
use agg_witness::gen::{self, crate_verify};
use agg_witness::instance::PublicInstance;
use agg_witness::native::{decompose, inf_norm, mod_q, mu};
use agg_witness::params::*;
use agg_witness::tables::Layout;

#[test]
fn usehint_residual_lemma_is_exact() {
    // max over Z_q x {0, 1} of |ctr_q(r - 2 gamma2 UseHint(h, r))| is exactly B_u (spec 3.2).
    assert_eq!(usehint_residual_max(), B_U);
}

#[cfg(feature = "fips")]
#[test]
fn s1_encoding_is_exact_on_all_of_zq() {
    use agg_witness::derive::fips_position;
    use agg_witness::digits::VR_EXACT;
    use agg_witness::native::use_hint;
    let mut edges = 0;
    for r in 0..Q {
        let (r1, r0, s, vr, e) = fips_position(r);
        assert_eq!(mod_q(2 * GAMMA2 * r1 + r0), r);
        assert_eq!(r0, s * vr - GAMMA2 * e + (1 + s) / 2);
        assert!(e * e == e && s * s == 1);
        assert!(e * (1 + s) == 0 && e * vr == 0 && e * r1 == 0);
        assert!((0..M_W).contains(&r1));
        let (lo, hi) = VR_EXACT.range();
        assert!(lo <= vr && vr <= hi);
        assert_eq!(
            e == 1,
            r == Q - GAMMA2,
            "the edge flag marks exactly the residue -gamma2"
        );
        edges += e;
        for h in [0i64, 1] {
            let g = h * s;
            assert_eq!(g * (s - g), 0);
            let x = r1 + g - use_hint(h == 1, r);
            assert_eq!(x * (x - M_W) * (x + M_W), 0);
        }
    }
    assert_eq!(edges, 1);
}

fn edge_record(t: (u64, u64, u64)) -> gen::Record {
    gen::record_hedged(t.0, t.1, t.2)
}

fn check_all_layouts(records: &[gen::Record]) {
    for layout in Layout::all() {
        let (inst, tables) = common::batch(layout, records);
        let rep = common::report(&inst, &tables);
        assert!(rep.is_ok(), "{}: {:?}", layout.name(), rep.failures);
        for (i, r) in records.iter().enumerate() {
            let bytes = tables.reencode(i, &inst.records[i].c_tilde).unwrap();
            assert_eq!(bytes, r.sig);
            assert!(crate_verify(&r.pk, &r.msg, &bytes));
        }
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
    check_all_layouts(&batch);
    #[cfg(feature = "fips")]
    {
        let (_, tables) = common::batch(Layout::Fips, &batch);
        let vals = tables.record_values(1);
        let idx = r * N + j;
        assert_eq!(
            (vals.fips.e[idx], vals.fips.s[idx], vals.fips.vr[idx]),
            (1, -1, 0)
        );
        assert_eq!((vals.fips.r1[idx], vals.fips.r0[idx]), (0, -GAMMA2));
        assert_eq!(vals.fips.e.iter().sum::<i64>(), 1);
    }
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
    check_all_layouts(&batch);
    // The slack form at the bound: z = +-B_z is representable exactly, B_z + 1 is not.
    use agg_witness::digits::Z_EXACT;
    for z in [-B_Z, B_Z] {
        let (a, d) = Z_EXACT.decompose(z);
        assert!(Z_EXACT.digits.in_alphabets(&d) && (a == 0 || a == 1));
    }
    let (_, d) = Z_EXACT.decompose(B_Z + 1);
    assert!(!Z_EXACT.digits.in_alphabets(&d));
}
