//! Negative tests: every mutation is rejected by the checker at the stated stage.

#![allow(clippy::needless_range_loop)]

mod common;

use agg_witness::bridge::{self, BridgeMatrices, BridgeTables};
use agg_witness::check::{
    check, check_digit_column, digit_virtual, CheckOptions, CheckReport, Identity, Stage,
};
use agg_witness::derive::{derive_unchecked, SigWitness};
use agg_witness::gen::{self, Record};
use agg_witness::instance::PublicInstance;
use agg_witness::native::{sig_decode, sig_encode};
use agg_witness::padding::main_index;
use agg_witness::params::*;
use agg_witness::tables::{BatchTables, Layout};

/// Builds a batch where record `k` is replaced by `bad_sig`, derived without verification.
fn batch_with(
    layout: Layout,
    records: &[Record],
    k: usize,
    bad_sig: &[u8],
) -> Option<(PublicInstance, BatchTables)> {
    let mut recs = records.to_vec();
    recs[k].sig = bad_sig.to_vec();
    let inst = PublicInstance::from_records(&recs).unwrap();
    let wits: Option<Vec<SigWitness>> = inst
        .records
        .iter()
        .zip(&recs)
        .map(|(p, r)| derive_unchecked(&p.key, &p.mu, &r.sig).ok())
        .collect();
    let tables = BatchTables::from_witnesses(layout, &wits?).ok()?;
    Some((inst, tables))
}

fn assert_rejected(rep: &CheckReport, id: Identity) {
    assert!(!rep.relation_ok(), "accepted");
    assert!(rep.failed(id), "expected {id:?}, got {:?}", rep.counts);
}

#[test]
fn flipped_signature_bits_are_rejected_by_the_checker() {
    let records = gen::records(300, 8);
    let mut through_checker = [0usize; 3];
    for layout in Layout::all() {
        for k in [0usize, 5] {
            let sig = &records[k].sig;
            // c~ bits, z bits (various coefficients and bit positions), hint bytes.
            let mut flips: Vec<(usize, usize)> = vec![(0, 0), (17, 3), (31, 7)];
            for c in [0usize, 100, 511, 1023] {
                for b in [0usize, 9, 16, 17] {
                    let bit = 32 * 8 + c * Z_BITS + b;
                    flips.push((bit / 8, bit % 8));
                }
            }
            for pos in [
                SIG_BYTES - OMEGA - K,
                SIG_BYTES - OMEGA - K + 3,
                SIG_BYTES - 1,
            ] {
                for b in 0..8 {
                    flips.push((pos, b));
                }
            }
            for &(byte, bit) in &flips {
                let mut bad = sig.clone();
                bad[byte] ^= 1 << bit;
                let region = if byte < 32 {
                    0
                } else if byte < 32 + L * 576 {
                    1
                } else {
                    2
                };
                match batch_with(layout, &records, k, &bad) {
                    None => assert_eq!(region, 2, "only a malformed hint may fail to derive"),
                    Some((inst, tables)) => {
                        let rep = check(&inst, &tables, None, CheckOptions::default());
                        assert_rejected(&rep, Identity::CommitmentHash);
                        through_checker[region] += 1;
                    }
                }
            }
        }
    }
    assert!(
        through_checker.iter().all(|&c| c > 0),
        "{through_checker:?}"
    );
}

fn honest(layout: Layout) -> (PublicInstance, BatchTables) {
    common::batch(layout, &gen::records(310, 6))
}

fn rep_of(inst: &PublicInstance, t: &BatchTables) -> CheckReport {
    check(inst, t, None, CheckOptions::default())
}

#[test]
fn table_mutations_fail_at_the_stated_stage() {
    let (inst, base) = honest(Layout::H);
    let map = base.map();
    let x = main_index(77, 2, 3);
    // Non-Boolean bit: bool_b, stage 1.
    let mut t = base.clone();
    t.columns[map.b + 1][x] = 2;
    let rep = rep_of(&inst, &t);
    assert_rejected(&rep, Identity::BoolB);
    assert_eq!(rep.first_stage(), Some(Stage::One));
    // w1' = 44 (bits 2, 3, 5): range_w1, stage 1.
    let mut t = base.clone();
    for (b, v) in [0, 0, 1, 1, 0, 1].into_iter().enumerate() {
        t.columns[map.b + b][x] = v;
    }
    let rep = rep_of(&inst, &t);
    assert_rejected(&rep, Identity::RangeW1);
    assert_eq!(rep.first_stage(), Some(Stage::One));
    // Base-16 digit outside its alphabet: b16_prod, stage 1.
    let mut t = base.clone();
    t.columns[map.z + 1][x] = 8;
    let rep = rep_of(&inst, &t);
    assert_rejected(&rep, Identity::B16Prod);
    assert_eq!(rep.first_stage(), Some(Stage::One));
    // Base-6 digit (u_4, H-u) outside [-3, 2]: leaf6, stage 2 (3 and -4 are in the old
    // base-8 alphabet).
    for value in [3, -4] {
        let mut t = base.clone();
        t.columns[map.u.unwrap() + 4][x] = value;
        let rep = rep_of(&inst, &t);
        assert_rejected(&rep, Identity::Leaf6);
        assert_eq!(rep.first_stage(), Some(Stage::Two));
    }
    // Base-8 digit (v_6) outside [-4, 3]: leaf8, stage 2.
    let mut t = base.clone();
    t.columns[map.v + 6][x] = 4;
    let rep = rep_of(&inst, &t);
    assert_rejected(&rep, Identity::Leaf8);
    assert_eq!(rep.first_stage(), Some(Stage::Two));
    // Base-4 digit (z_4) outside [-2, 1]: leaf4, stage 2.
    let mut t = base.clone();
    t.columns[map.z + 4][x] = 2;
    assert_rejected(&rep_of(&inst, &t), Identity::Leaf4);
    // In-alphabet change of a z digit: only the ring identity (stage 3) fails.
    let mut t = base.clone();
    let d = t.columns[map.z][x];
    t.columns[map.z][x] = if d < 7 { d + 1 } else { d - 1 };
    let rep = rep_of(&inst, &t);
    assert_rejected(&rep, Identity::Ring);
    assert_eq!(rep.first_stage(), Some(Stage::Three));
    assert!(!rep.failed(Identity::CommitmentHash));
    // In-alphabet change of a v digit or a u digit: ring.
    for c in [map.v + 3, map.u.unwrap() + 2] {
        let mut t = base.clone();
        let d = t.columns[c][x];
        t.columns[c][x] = if d < 7 { d + 1 } else { d - 1 };
        let rep = rep_of(&inst, &t);
        assert_rejected(&rep, Identity::Ring);
        assert_eq!(rep.counts.len(), 1, "{:?}", rep.counts);
    }
    // A valid bit flipped: w1' changes, so the ring identity and the commitment hash fail.
    let mut t = base.clone();
    t.columns[map.b][x] ^= 1;
    let rep = rep_of(&inst, &t);
    assert!(rep.failed(Identity::Ring) && rep.failed(Identity::CommitmentHash));
    // A padded row with other in-alphabet digits violates only the honest padding form: the
    // relation does not constrain padded z (spec Section 4.2: padded digits MUST NOT be required zero).
    let mut t = base.clone();
    t.columns[map.z][main_index(5, 1, 7)] = 3;
    let rep = rep_of(&inst, &t);
    assert!(rep.relation_ok());
    assert!(rep.failed(Identity::PaddingCanonical));
    // Shape: a missing column.
    let mut t = base.clone();
    t.columns.pop();
    assert_rejected(&rep_of(&inst, &t), Identity::Shape);
}

/// Relation H-u makes u exact: u = +-190,465 = +-B_u has digits in their alphabets, while
/// +-190,466 cannot be encoded without a digit outside its alphabet (a base-16 digit of B_ulo,
/// caught by b16_prod, or the base-6 top digit, caught by leaf6).
#[test]
fn u_slack_boundaries_of_relation_h_u() {
    use agg_witness::digits::U_STEP;
    use agg_witness::tables::RecordValues;
    let records = gen::records(311, 2);
    let inst = PublicInstance::from_records(&records).unwrap();
    let wits = common::witnesses(&inst, &records);
    let (rho, j) = (2usize, 5usize);
    let x = main_index(j, rho, 1);
    let with_u = |u: i64| {
        let mut vals: Vec<RecordValues> = wits
            .iter()
            .map(|w| RecordValues::from_witness(Layout::H, w))
            .collect();
        vals[1].u[rho][j] = u;
        BatchTables::from_values(Layout::H, &vals).unwrap()
    };
    let range_ids = [Identity::B16Prod, Identity::Leaf6];
    for u in [B_U, -B_U] {
        let t = with_u(u);
        assert_eq!(t.record_values(1).u[rho][j], u);
        let rep = rep_of(&inst, &t);
        // The value changed, so the ring identity fails, but no range identity does.
        assert!(range_ids.iter().all(|&id| !rep.failed(id)), "{u}: {:?}", rep.counts);
        assert!(!rep.failed(Identity::HonestBound));
    }
    for (u, id) in [(B_U + 1, Identity::B16Prod), (-B_U - 1, Identity::B16Prod)] {
        let t = with_u(u);
        assert_eq!(t.record_values(1).u[rho][j], u);
        assert_rejected(&rep_of(&inst, &t), id);
    }
    // The other encodings of +-190,466: top digit out of [-3, 2] with B_ulo in [0, 2^16).
    let map = with_u(0).map();
    let u0 = map.u.unwrap();
    for (u, top) in [(B_U + 1, 3i64), (-B_U - 1, -4)] {
        let low = u + B_U - STEP_U * (top + 3);
        assert!((0..1 << 16).contains(&low), "{low}");
        let mut t = with_u(0);
        let digits = U_STEP.low.decompose(low);
        for (k, d) in digits.iter().enumerate() {
            t.columns[u0 + k][x] = *d as i8;
        }
        t.columns[u0 + 4][x] = top as i8;
        assert_eq!(t.record_values(1).u[rho][j], u);
        let rep = rep_of(&inst, &t);
        assert_rejected(&rep, Identity::Leaf6);
        assert!(!rep.failed(Identity::B16Prod));
    }
}

#[test]
fn tampered_virtual_tables_fail_link_and_leaf16() {
    let (_, base) = honest(Layout::H);
    let col = &base.columns[0];
    let mut rep = CheckReport::default();
    let mut v = digit_virtual(col, 16);
    check_digit_column(&mut rep, "z_0", col, 16, &v);
    assert!(rep.is_ok());
    v.p[10] += 2;
    check_digit_column(&mut rep, "z_0", col, 16, &v);
    assert!(rep.failed(Identity::Link));
    let mut rep = CheckReport::default();
    let mut v = digit_virtual(col, 16);
    v.a_lo[3] = Some(1);
    check_digit_column(&mut rep, "z_0", col, 16, &v);
    assert!(rep.failed(Identity::Leaf16) && !rep.failed(Identity::Link));
}

#[test]
fn bridge_mutations_are_rejected() {
    let (inst, tables) = honest(Layout::H);
    let mats = BridgeMatrices::random_for_tests(0xB41D6E);
    let bw = bridge::compute(&tables.geom, &tables.bit_lanes(), &mats).unwrap();
    let base = BridgeTables::from_witness(&bw).unwrap();
    let run = |bt: &BridgeTables| check(&inst, &tables, Some((&mats, bt)), CheckOptions::default());
    assert!(run(&base).is_ok());
    let bump = |c: &mut Vec<i8>, x: usize| c[x] = if c[x] < 7 { c[x] + 1 } else { c[x] - 1 };
    // Level-1 carry digit changed: y1 = sum - 2 kappa1 leaves {0, 1}.
    let mut bt = base.clone();
    let x = bw.kappa1[2].iter().position(|&k| k > 3).unwrap();
    bump(&mut bt.g1[3 * 2], x);
    assert_rejected(&run(&bt), Identity::BoolY1);
    // Parity of the wrong value (y2 flipped, carry kept): level2.
    let mut bt = base.clone();
    bt.g2[4][100] ^= 1;
    let rep = run(&bt);
    assert_rejected(&rep, Identity::Level2);
    assert!(!rep.failed(Identity::BoolY2));
    // Non-Boolean y2: bool_y2.
    let mut bt = base.clone();
    bt.g2[0][5] = 2;
    assert_rejected(&run(&bt), Identity::BoolY2);
    // Level-2 carry digit changed: level2.
    let mut bt = base.clone();
    bump(&mut bt.g2[6 + 3 * 5 + 1], 77);
    assert_rejected(&run(&bt), Identity::Level2);
    // Changed public Y bit: level3 on the F_p side and the GF(2) side.
    let mut bt = base.clone();
    bt.y[9] ^= 1;
    let rep = run(&bt);
    assert!(rep.failed(Identity::Level3) && rep.failed(Identity::Gf2Bridge));
    // Level-3 carry digit changed: level3 only.
    let mut bt = base.clone();
    bump(&mut bt.g3[0], 4000);
    let rep = run(&bt);
    assert_rejected(&rep, Identity::Level3);
    assert!(!rep.failed(Identity::Gf2Bridge));
    // A carry digit outside its alphabet: b16_prod (base 16) or leaf4 (base 4).
    let mut bt = base.clone();
    bt.g1[2][0] = 2;
    assert_rejected(&run(&bt), Identity::Leaf4);
    let mut bt = base;
    bt.g3[1][1] = -9;
    assert_rejected(&run(&bt), Identity::B16Prod);
}

/// The F_p cheat of spec Section 6: for a wrong z, v = q^{-1} Delta_0 mod p satisfies the field
/// identity, but no v in [-2^26, 2^26) does, so the digits of v reject it.
#[test]
fn unbounded_quotient_cheat_needs_out_of_range_v() {
    let records = gen::records(320, 1);
    let inst = PublicInstance::from_records(&records).unwrap();
    let w = common::witnesses(&inst, &records).remove(0);
    let key = &inst.records[0].key;
    let mut z = w.z;
    z[2][5] += 1;
    let az = agg_witness::crt::az_crt(key, &z);
    let p = u128::from(P_FIELD);
    let mul = |a: u128, b: u128| a * b % p;
    let pow = |mut b: u128, mut e: u128| {
        let mut r = 1u128;
        while e > 0 {
            if e & 1 == 1 {
                r = mul(r, b);
            }
            b = mul(b, b);
            e >>= 1;
        }
        r
    };
    let to_fp = |x: i128| x.rem_euclid(p as i128) as u128;
    let q_inv = pow(Q as u128, p - 2);
    assert_eq!(mul(q_inv, Q as u128), 1);
    let mut nonzero = 0;
    for r in 0..K {
        for j in 0..N {
            let d0 =
                i128::from(az[r][j] - w.t[r][j] - 2 * GAMMA2 * i64::from(w.w1[r][j]) - w.u[r][j]);
            let v = mul(to_fp(d0), q_inv);
            // Field identity Delta = Delta_0 - q v = 0 holds.
            assert_eq!((to_fp(d0) + p - mul(Q as u128, v)) % p, 0);
            if d0.rem_euclid(i128::from(Q)) != 0 {
                nonzero += 1;
                let centered = if v > p / 2 {
                    v as i128 - p as i128
                } else {
                    v as i128
                };
                assert!(
                    centered.abs() >= 1 << 26,
                    "cheat quotient inside the proven range"
                );
                let digits = agg_witness::digits::V_DIGITS
                    .decompose(i64::try_from(centered).unwrap_or(i64::MAX / 2));
                assert!(!agg_witness::digits::V_DIGITS.in_alphabets(&digits));
            }
        }
    }
    assert!(nonzero > 1000, "the wrong z changes every row");
}

#[test]
fn hint_reencoding_rejects_unreachable_w1() {
    // Relation H is relaxed: a witness whose w1' differs from HighBits(w') by 2 is valid for the
    // ring identity but has no FIPS hint.
    let records = gen::records(330, 1);
    let inst = PublicInstance::from_records(&records).unwrap();
    let w = common::witnesses(&inst, &records).remove(0);
    let mut w1 = w.w1;
    let mut u = w.u;
    let (r, j) = (0usize, 3usize);
    let new = (i64::from(w1[r][j]) + 2).rem_euclid(M_W);
    u[r][j] -= 2 * GAMMA2 * (new - i64::from(w1[r][j]));
    w1[r][j] = new as u8;
    assert_eq!(
        agg_witness::derive::reencode_h(&w.c_tilde, &w.z, &u, &w1),
        Err(agg_witness::derive::ReencodeError::NotReachable)
    );
    // Unchanged witness re-encodes to the signature.
    assert_eq!(
        agg_witness::derive::reencode_h(&w.c_tilde, &w.z, &w.u, &w.w1).unwrap(),
        records[0].sig
    );
    let d = sig_decode(&records[0].sig).unwrap();
    assert_eq!(sig_encode(&d.c_tilde, &d.z, &d.h), records[0].sig);
}

#[cfg(feature = "h-exact")]
#[test]
fn h_exact_mutations() {
    let (inst, base) = honest(Layout::HExact);
    let map = base.map();
    let x = main_index(9, 1, 2);
    let mut t = base.clone();
    t.columns[map.az.unwrap()][x] = 2;
    assert_rejected(&rep_of(&inst, &t), Identity::BoolSlack);
    // Flipping a valid slack bit moves z by 130,915: ring fails (and b_z may leave its range).
    let mut t = base.clone();
    t.columns[map.au.unwrap()][x] ^= 1;
    assert_rejected(&rep_of(&inst, &t), Identity::Ring);
    // Base-2 top digit of b_z outside {-1, 0}: leaf2, stage 2.
    let mut t = base.clone();
    t.columns[map.z + 4][x] = 1;
    let rep = rep_of(&inst, &t);
    assert_rejected(&rep, Identity::Leaf2);
    assert_eq!(rep.first_stage(), Some(Stage::Two));
}

#[cfg(feature = "fips")]
#[test]
fn fips_mutations() {
    let (inst, base) = honest(Layout::Fips);
    let map = base.map();
    let vals = base.record_values(1);
    let hint_pos = vals.fips.g.iter().position(|&g| g != 0).expect("a hint");
    let plain_pos = (0..K * N)
        .find(|&p| vals.fips.s[p] == 1 && vals.fips.g[p] == 0)
        .unwrap();
    let at = |p: usize| main_index(p % N, p / N, 1);
    let (s, g, e) = (map.s.unwrap(), map.g.unwrap(), map.e.unwrap());
    let mut t = base.clone();
    t.columns[s][at(plain_pos)] = 0;
    assert_rejected(&rep_of(&inst, &t), Identity::SignSquare);
    let mut t = base.clone();
    t.columns[e][at(plain_pos)] = 1;
    assert_rejected(&rep_of(&inst, &t), Identity::EdgeProducts);
    let mut t = base.clone();
    t.columns[g][at(hint_pos)] = -t.columns[g][at(hint_pos)];
    let rep = rep_of(&inst, &t);
    assert!(rep.failed(Identity::GSign) && rep.failed(Identity::UseHintCubic));
    let mut t = base.clone();
    t.columns[g][at(plain_pos)] = 2;
    assert_rejected(&rep_of(&inst, &t), Identity::TernaryG);
    // Dropping a hint (g -> 0) breaks UseHint and the hint weight.
    let mut t = base.clone();
    t.columns[g][at(hint_pos)] = 0;
    let rep = rep_of(&inst, &t);
    assert!(rep.failed(Identity::UseHintCubic) && rep.failed(Identity::HintWeight));
    // Hint slack digit changed: hint weight.
    let mut t = base.clone();
    let bh = map.bh.unwrap();
    t.columns[bh][1] = if t.columns[bh][1] < 3 {
        t.columns[bh][1] + 1
    } else {
        t.columns[bh][1] - 1
    };
    assert_rejected(&rep_of(&inst, &t), Identity::HintWeight);
    // r1 digit changed: ring and UseHint.
    let mut t = base.clone();
    let r1b = map.r1.unwrap() + 1;
    let d = t.columns[r1b][at(plain_pos)];
    t.columns[r1b][at(plain_pos)] = if d < 7 { d + 1 } else { d - 1 };
    let rep = rep_of(&inst, &t);
    assert!(rep.failed(Identity::Ring) && rep.failed(Identity::UseHintCubic));
    // Padded rows carry s = -1 and u_h = omega; s = 0 there is rejected.
    let mut t = base;
    t.columns[s][main_index(0, 0, 7)] = 0;
    assert_rejected(&rep_of(&inst, &t), Identity::SignSquare);
}
