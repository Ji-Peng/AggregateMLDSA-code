//! The batched key expansion, public instance and witness derivation on every available SIMD
//! backend equal the reference arithmetic of this crate bit for bit, errors included.

use agg_mldsa_simd::{Backend, Engine};
use agg_witness::derive::{derive_with_t, DeriveError};
use agg_witness::expand::{expand_a, public_t, ExpandedKey};
use agg_witness::fast::{derive_tile_with, fast_path_only_with, FastWitness};
use agg_witness::gen::{self, Record};
use agg_witness::instance::{InstanceError, PublicInstance};
use agg_witness::native::{ctr_q, mu, pk_decode, sample_in_ball, tr, SigDecodeError};
use agg_witness::ntt::NttPrime;
use agg_witness::params::*;
use rayon::prelude::*;

fn engines() -> Vec<Engine> {
    [Backend::Scalar, Backend::Avx2, Backend::Avx512]
        .into_iter()
        .filter_map(Engine::new)
        .collect()
}

fn inputs(records: &[Record]) -> Vec<(&[u8], &[u8], [u8; C_TILDE_BYTES])> {
    records
        .iter()
        .map(|r| {
            let mut c = [0u8; C_TILDE_BYTES];
            let n = r.sig.len().min(C_TILDE_BYTES);
            c[..n].copy_from_slice(&r.sig[..n]);
            (r.pk.as_slice(), r.msg.as_slice(), c)
        })
        .collect()
}

/// Every field and reference form of `key` against the reference expansion of `pk`.
fn assert_key_matches_reference(key: &ExpandedKey, pk: &[u8]) {
    let decoded = pk_decode(pk).expect("decodes");
    let (a_hat, blocks) = expand_a(&decoded.rho);
    let fips = NttPrime::fips();
    assert_eq!(key.pk, pk);
    assert_eq!(key.rho, decoded.rho);
    assert_eq!(key.t1, decoded.t1);
    assert_eq!(key.tr, tr(pk));
    assert_eq!(key.expand_blocks, blocks);
    assert_eq!(*key.a_hat(), *a_hat);
    let p2 = agg_witness::crt::prime2();
    let p2_hat = key.a_hat_p2_reference();
    for r in 0..K {
        for s in 0..L {
            let coeff = fips.intt(&a_hat[r][s]);
            let centered: [i64; N] = coeff.map(|x| ctr_q(x as i64));
            assert_eq!(key.a[r][s], centered.map(|x| x as i32), "A[{r}][{s}]");
            assert_eq!(p2_hat[r][s], p2.ntt_signed(&centered));
        }
        let t: [i64; N] = decoded.t1[r].map(|x| i64::from(x) << D);
        assert_eq!(key.t1_2d_hat()[r], fips.ntt_signed(&t));
    }
}

#[test]
fn expanded_keys_match_the_reference_expansion() {
    // 56 own keys, 8 keys sharing the matrix seed of key 0 (and of each other), one key of
    // the wrong length in the middle of a tile.
    let mut pks: Vec<Vec<u8>> = (0..56u64)
        .map(|i| gen::keypair(&gen::key_seed(0xE8, i)).1)
        .collect();
    let rho: [u8; 32] = pks[0][..32].try_into().unwrap();
    pks.extend((0..8u64).map(|i| gen::keypair_with_rho(&gen::key_seed(0xE9, i), &rho).1));
    let mut short = pks[3].clone();
    short.pop();
    pks.insert(13, short);
    let views: Vec<&[u8]> = pks.iter().map(Vec::as_slice).collect();
    for engine in engines() {
        let keys = ExpandedKey::many_with(&engine, &views);
        assert_eq!(keys.len(), views.len());
        for (i, (key, pk)) in keys.iter().zip(&views).enumerate() {
            match key {
                None => assert_eq!(i, 13, "{:?}: only the short key fails", engine.backend()),
                Some(key) => {
                    assert_eq!(key.prepared.backend, engine.backend());
                    assert_key_matches_reference(key, pk);
                }
            }
        }
    }
    let one = ExpandedKey::from_bytes(views[0]).unwrap();
    assert_key_matches_reference(&one, views[0]);
    assert!(ExpandedKey::from_bytes(views[13]).is_none());
    // A 1,312-byte key lives in about 53 KiB (A, its two NTT-domain copies, t1, pk).
    let bytes = std::mem::size_of::<ExpandedKey>() + 3 * K * L * N * 4 + one.pk.capacity();
    assert!(bytes < 54 << 10, "{bytes} bytes per key");
}

fn assert_instance_matches_reference(engine: &Engine, records: &[Record]) {
    let inputs = inputs(records);
    let inst = PublicInstance::new_with(engine, &inputs).unwrap();
    assert_eq!(inst.records.len(), records.len());
    inst.records.par_iter().zip(records).for_each(|(p, r)| {
        assert_eq!(p.key.pk, r.pk);
        assert_eq!(p.key.prepared.backend, engine.backend());
        assert_eq!(p.msg, r.msg);
        assert_eq!(p.c_tilde[..], r.sig[..C_TILDE_BYTES]);
        assert_eq!(p.mu, mu(&tr(&r.pk), &r.msg, &[]));
        let c = sample_in_ball(&r.sig[..C_TILDE_BYTES]);
        let t = public_t(&c, &p.key.t1);
        assert_eq!(p.c, c);
        assert_eq!(p.t.map(|row| row.map(i64::from)), t);
    });
    // Records with the same key share one expansion.
    for (i, p) in inst.records.iter().enumerate() {
        let first = records.iter().position(|r| r.pk == records[i].pk).unwrap();
        assert!(std::sync::Arc::ptr_eq(&p.key, &inst.records[first].key));
    }
    assert_key_matches_reference(&inst.records[0].key, &records[0].pk);
}

#[test]
fn public_instance_matches_the_reference_build() {
    let shared = gen::records_shared_keys(0xE10, 256, 16);
    let distinct = gen::records(0xE11, 256);
    for engine in engines() {
        assert_instance_matches_reference(&engine, &shared);
        assert_instance_matches_reference(&engine, &distinct);
        // Odd counts across tile boundaries.
        assert_instance_matches_reference(&engine, &shared[..65]);
        assert_instance_matches_reference(&engine, &distinct[..9]);
    }
    // The first record whose key does not decode, as before.
    let mut bad = shared[..40].to_vec();
    bad[17].pk.pop();
    bad[33].pk = bad[17].pk.clone();
    let inputs = inputs(&bad);
    for engine in engines() {
        assert_eq!(
            PublicInstance::new_with(&engine, &inputs).unwrap_err(),
            InstanceError::PublicKey(17)
        );
    }
}

fn reference_derivation(inst: &PublicInstance, sigs: &[&[u8]]) -> Vec<Result<FastWitness, DeriveError>> {
    inst.records
        .par_iter()
        .zip(sigs)
        .map(|(p, sig)| {
            derive_with_t(&p.key, &p.mu, sig, &p.c_tilde, &p.t).map(|w| {
                let mut out = FastWitness::zero();
                for s in 0..L {
                    out.z[s].0 = w.z[s].map(|x| x as i32);
                }
                for r in 0..K {
                    out.rows[r].w1 = w.w1[r];
                    out.rows[r].u.0 = w.u[r].map(|x| x as i32);
                    out.rows[r].v.0 = w.v[r].map(|x| x as i32);
                }
                out
            })
        })
        .collect()
}

fn fast(
    engine: &Engine,
    inst: &PublicInstance,
    sigs: &[&[u8]],
) -> Result<Vec<FastWitness>, (usize, DeriveError)> {
    let records: Vec<_> = inst.records.iter().collect();
    let mut out = vec![FastWitness::zero(); records.len()];
    derive_tile_with(engine, &records, sigs, &mut out)?;
    Ok(out)
}

#[test]
fn fast_derivation_matches_the_reference_witness() {
    let mut records = gen::records_shared_keys(0xE12, 40, 5);
    records.extend(gen::records(0xE13, 13));
    records.push(gen::hint_corner_record(0xE14));
    let sigs: Vec<&[u8]> = records.iter().map(|r| r.sig.as_slice()).collect();
    for engine in engines() {
        let inst = PublicInstance::new_with(&engine, &inputs(&records)).unwrap();
        let expected: Vec<FastWitness> = reference_derivation(&inst, &sigs)
            .into_iter()
            .map(Result::unwrap)
            .collect();
        assert_eq!(
            fast(&engine, &inst, &sigs).unwrap(),
            expected,
            "{:?}",
            engine.backend()
        );
        // Honest records never take the reference fallback.
        for ((p, sig), want) in inst.records.iter().zip(&sigs).zip(&expected) {
            assert_eq!(fast_path_only_with(&engine, p, sig).as_ref(), Some(want));
        }
        // Values agree with the widened accessors.
        let w = derive_with_t(
            &inst.records[0].key,
            &inst.records[0].mu,
            sigs[0],
            &inst.records[0].c_tilde,
            &inst.records[0].t,
        )
        .unwrap();
        assert_eq!(expected[0].z_wide(), w.z);
        assert_eq!(expected[0].u_wide(), w.u);
        assert_eq!(expected[0].v_wide(), w.v);
        assert_eq!(expected[0].w1(), w.w1);
    }
}

/// The signature bytes of `r` with z coefficient 0 of row 0 set to gamma1 (packed as 0).
fn z_at_gamma1(sig: &[u8]) -> Vec<u8> {
    let mut s = sig.to_vec();
    s[C_TILDE_BYTES] = 0;
    s[C_TILDE_BYTES + 1] = 0;
    s[C_TILDE_BYTES + 2] &= !0x03;
    s
}

#[test]
fn a_bad_record_fails_with_the_reference_error_and_index() {
    let records = gen::records_shared_keys(0xE15, 21, 4);
    let other = gen::record(0xE16, 0);
    type Mutation = (&'static str, fn(&mut Record, &Record), Option<DeriveError>);
    let cases: [Mutation; 7] = [
        (
            "short",
            |r, _| {
                r.sig.pop();
            },
            Some(DeriveError::Decode(SigDecodeError::WrongLength)),
        ),
        (
            "long",
            |r, _| r.sig.push(0),
            Some(DeriveError::Decode(SigDecodeError::WrongLength)),
        ),
        (
            "hint",
            |r, _| r.sig[C_TILDE_BYTES + L * 32 * Z_BITS + OMEGA] = 81,
            Some(DeriveError::Decode(SigDecodeError::MalformedHint)),
        ),
        (
            "z norm",
            |r, _| r.sig = z_at_gamma1(&r.sig),
            Some(DeriveError::ZNorm),
        ),
        (
            "z byte",
            |r, _| r.sig[C_TILDE_BYTES + 100] ^= 4,
            Some(DeriveError::CommitmentMismatch),
        ),
        (
            "c~",
            |r, _| r.sig[5] ^= 1,
            Some(DeriveError::CommitmentMismatch),
        ),
        (
            "other key",
            |r, o| r.sig = o.sig.clone(),
            Some(DeriveError::CommitmentMismatch),
        ),
    ];
    for engine in engines() {
        for (name, mutate, expected) in &cases {
            for k in [0usize, 7, 8, 20] {
                // Mutated before the instance (the instance's c~ follows the signature) and
                // after it (the instance keeps the honest c~ and T).
                for before in [true, false] {
                    let mut bad = records.clone();
                    if before {
                        mutate(&mut bad[k], &other);
                    }
                    let inst = PublicInstance::new_with(&engine, &inputs(&bad)).unwrap();
                    if !before {
                        mutate(&mut bad[k], &other);
                    }
                    let sigs: Vec<&[u8]> = bad.iter().map(|r| r.sig.as_slice()).collect();
                    let reference = reference_derivation(&inst, &sigs);
                    let first = reference
                        .iter()
                        .position(Result::is_err)
                        .map(|i| (i, reference[i].clone().unwrap_err()));
                    assert_eq!(
                        first,
                        expected.map(|e| (k, e)),
                        "{name} at {k}, before {before}"
                    );
                    assert_eq!(
                        fast(&engine, &inst, &sigs).unwrap_err(),
                        first.unwrap(),
                        "{name} at {k}"
                    );
                    assert!(fast_path_only_with(&engine, &inst.records[k], sigs[k]).is_none());
                    assert!(fast_path_only_with(
                        &engine,
                        &inst.records[(k + 1) % bad.len()],
                        sigs[(k + 1) % bad.len()]
                    )
                    .is_some());
                }
            }
        }
    }
}
