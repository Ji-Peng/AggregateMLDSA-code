//! The reference verifier (own FIPS 204 arithmetic) against the `ml-dsa` crate.

#![allow(clippy::needless_range_loop)]

use crate::common;

use agg_witness::gen::{self, crate_verify, SplitMix64};
use agg_witness::native::{sig_decode, verify, verify_with_context};
use agg_witness::params::*;
use rayon::prelude::*;

#[test]
fn reference_verifier_accepts_random_signatures() {
    let records = common::mixed_records(100, 128, 32);
    assert_eq!(records.len(), 160);
    records.par_iter().for_each(|r| {
        assert!(crate_verify(&r.pk, &r.msg, &r.sig));
        assert!(verify(&r.pk, &r.msg, &r.sig));
    });
}

#[test]
fn reference_verifier_agrees_with_crate_on_mutations() {
    let records = gen::records(101, 128);
    let other = gen::record(102, 0);
    let cases: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = records
        .iter()
        .enumerate()
        .flat_map(|(i, r)| {
            let mut rng = SplitMix64::new(1000 + i as u64);
            let mut out = Vec::new();
            // One bit flip in each region: c~, z, hint indices, hint counters.
            for (lo, hi) in [
                (0, 32),
                (32, 32 + L * 576),
                (32 + L * 576, SIG_BYTES - K),
                (SIG_BYTES - K, SIG_BYTES),
            ] {
                let mut s = r.sig.clone();
                let pos = lo + rng.below((hi - lo) as u64) as usize;
                s[pos] ^= 1 << rng.below(8);
                out.push((r.pk.clone(), r.msg.clone(), s));
            }
            // Wrong message, wrong key, truncated signature, flipped public-key bit.
            let mut m = r.msg.clone();
            m.push(0x55);
            out.push((r.pk.clone(), m, r.sig.clone()));
            out.push((other.pk.clone(), r.msg.clone(), r.sig.clone()));
            out.push((r.pk.clone(), r.msg.clone(), r.sig[..SIG_BYTES - 1].to_vec()));
            let mut pk = r.pk.clone();
            pk[32 + rng.below((PK_BYTES - 32) as u64) as usize] ^= 1;
            out.push((pk, r.msg.clone(), r.sig.clone()));
            out
        })
        .collect();
    let results: Vec<(bool, bool)> = cases
        .par_iter()
        .map(|(pk, m, s)| (verify(pk, m, s), crate_verify(pk, m, s)))
        .collect();
    let mut rejected = 0;
    for (k, (ours, theirs)) in results.iter().enumerate() {
        assert_eq!(ours, theirs, "case {k}");
        rejected += usize::from(!ours);
    }
    // Every mutation of an honest signature is rejected (overwhelming probability).
    assert_eq!(rejected, cases.len());
}

#[test]
fn reference_verifier_rejects_large_z_like_crate() {
    // Set one z coefficient to gamma1 - beta (= B_z + 1): decodes, fails line 13 of Algorithm 8.
    let r = gen::record(103, 0);
    let d = sig_decode(&r.sig).unwrap();
    let mut z = d.z;
    z[1][7] = B_Z + 1;
    let s = agg_witness::native::sig_encode(&d.c_tilde, &z, &d.h);
    assert!(!verify(&r.pk, &r.msg, &s));
    assert!(!crate_verify(&r.pk, &r.msg, &s));
}

#[test]
fn reference_verifier_with_context_strings() {
    use ml_dsa::{EncodedVerifyingKey, MlDsa44, VerifyingKey};
    let (sk, pk) = gen::keypair(&gen::key_seed(104, 0));
    for ctx_len in [0usize, 1, 17, 255] {
        let ctx = vec![0xC7u8; ctx_len];
        let sig = sk.sign_deterministic(b"context test", &ctx).unwrap();
        let bytes = sig.encode().as_slice().to_vec();
        assert!(verify_with_context(&pk, b"context test", &ctx, &bytes));
        let vk = VerifyingKey::<MlDsa44>::decode(
            &EncodedVerifyingKey::<MlDsa44>::try_from(pk.as_slice()).unwrap(),
        );
        assert!(vk.verify_with_context(b"context test", &ctx, &sig));
        assert_eq!(verify(&pk, b"context test", &bytes), ctx_len == 0);
    }
    assert!(!verify_with_context(
        &pk,
        b"m",
        &[0u8; 256],
        &gen::sign(&sk, b"m", None)
    ));
}
