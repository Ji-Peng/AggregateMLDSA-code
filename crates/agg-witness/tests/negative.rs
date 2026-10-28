//! Negative tests: a mutated signature is rejected by the checker of relation H-FIPS, the
//! quotient cheat needs an out-of-range v, and an unreachable w1' has no FIPS hint.

#![allow(clippy::needless_range_loop)]

mod common;

use agg_witness::check::Identity;
use agg_witness::derive::derive_unchecked;
use agg_witness::gen::{self, Record};
use agg_witness::hfips::{check_batch, HFipsRecord};
use agg_witness::instance::PublicInstance;
use agg_witness::native::{sig_decode, sig_encode};
use agg_witness::params::*;

/// Outcome of a batch where record `k` is replaced by `bad_sig`, derived without verification.
enum Outcome {
    /// The signature does not decode (only a malformed hint).
    Undecodable,
    /// The H-FIPS witness does not exist (|z| > B_z).
    NoWitness,
    /// The checker's report has these failing identities.
    Checked(Vec<Identity>),
}

fn batch_with(records: &[Record], k: usize, bad_sig: &[u8]) -> Outcome {
    let mut recs = records.to_vec();
    recs[k].sig = bad_sig.to_vec();
    let inst = PublicInstance::from_records(&recs).unwrap();
    let mut hf = Vec::new();
    for (p, r) in inst.records.iter().zip(&recs) {
        let Ok(w) = derive_unchecked(&p.key, &p.mu, &r.sig) else {
            return Outcome::Undecodable;
        };
        let Ok(h) = HFipsRecord::from_witness(&w) else {
            return Outcome::NoWitness;
        };
        hf.push(h);
    }
    Outcome::Checked(check_batch(&inst, &hf).counts.keys().copied().collect())
}

#[test]
fn flipped_signature_bits_are_rejected_by_the_checker() {
    let records = gen::records(300, 8);
    let mut through_checker = [0usize; 3];
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
            match batch_with(&records, k, &bad) {
                Outcome::Undecodable => {
                    assert_eq!(region, 2, "only a malformed hint may fail to decode")
                }
                Outcome::NoWitness => assert_eq!(region, 1, "only a flipped z may leave B_z"),
                Outcome::Checked(failed) => {
                    assert!(failed.contains(&Identity::CommitmentHash), "{failed:?}");
                    through_checker[region] += 1;
                }
            }
        }
    }
    assert!(
        through_checker.iter().all(|&c| c > 0),
        "{through_checker:?}"
    );
}

/// The F_p cheat: for a wrong z, v = q^{-1} Delta_0 mod p satisfies the field
/// identity, but no v in the proven range [-2^31, 2^31) of the two 16-bit halves does.
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
                    centered.abs() >= 1 << 31,
                    "cheat quotient inside the proven range"
                );
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
