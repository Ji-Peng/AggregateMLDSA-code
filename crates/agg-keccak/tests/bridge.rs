//! Flock side of the parity bridge (docs/bridge.md Sections 5 and 9): the bridge slot proven
//! on its own at the toy shape (`L = 2^12`) and the bridged sponge segment (sponge gates and
//! bridge slot in one union proof) at the production shape. The bridge seed is derived from
//! the Flock commitment, so the witness is fixed before the seed exists (spec Section 10,
//! events 2 and 3). Run with `--release`.

#![allow(clippy::needless_range_loop)]

use std::sync::Arc;
use std::time::Instant;

use agg_bridge::{gf2_walker, seed::derive_matrices_for, shape::Shape};
use agg_keccak::prover::Profile;
use agg_keccak::{
    bridge_slot::{pack_y, SlotLayout},
    prover::KeccakCommitment,
    segment::{BridgeBatch, BridgedSegment},
    sponge::{self, SpongeRecord},
};
use agg_witness::bridge::{gf2_composite, BridgeMatrices};
use agg_witness::padding::Geometry;
use flock_core::test_rng::Rng;
use sha3::{Digest, Sha3_256};

/// The bridge seed of a test transcript: a hash of the absorbed commitment (event 2 then
/// event 3), so it cannot be known before the witness is committed.
fn seed_after(commitment: &KeccakCommitment) -> [u8; 32] {
    let mut h = Sha3_256::new();
    h.update(b"test/sd");
    h.update(commitment.as_bytes());
    h.finalize().into()
}

/// The Flock challenger seed `fs_sigma` of the test transcript (after `sd`).
fn fs_after(commitment: &KeccakCommitment, sd: &[u8; 32]) -> [u8; 32] {
    let mut h = Sha3_256::new();
    h.update(b"test/fs");
    h.update(commitment.as_bytes());
    h.update(sd);
    h.finalize().into()
}

fn random_record(rng: &mut Rng) -> SpongeRecord {
    let mut mu = [0u8; 64];
    let mut w1 = [0u8; 768];
    mu.iter_mut().for_each(|b| *b = rng.next_u32() as u8);
    w1.iter_mut().for_each(|b| *b = rng.next_u32() as u8);
    SpongeRecord { mu, w1_encode: w1 }
}

/// `Y = M b` computed directly from dense 0/1 matrices, row by row (the toy shape).
fn direct_toy_product(mats: &BridgeMatrices, bits: &[u64]) -> Vec<u8> {
    let dense = |m: &agg_witness::bridge::SparseMatrix| {
        let mut d = vec![vec![0u8; m.cols]; m.rows];
        for c in 0..m.cols {
            for &r in m.column(c) {
                d[r as usize][c] = 1;
            }
        }
        d
    };
    let (d1, d2, d3) = (dense(&mats.m1), dense(&mats.m2), dense(&mats.m3));
    let b = |c: usize| ((bits[c / 64] >> (c % 64)) & 1) as u8;
    // Level 1 per block of 1,024, level 2 over the 4 x 64 parities, level 3 over 32.
    let y1: Vec<u8> = (0..4 * 64)
        .map(|i| {
            let (beta, r) = (i / 64, i % 64);
            (0..1024).fold(0, |acc, z| acc ^ (d1[r][z] & b(1024 * beta + z)))
        })
        .collect();
    let y2: Vec<u8> = (0..32)
        .map(|r| (0..256).fold(0, |acc, z| acc ^ (d2[r][z] & y1[z])))
        .collect();
    (0..16)
        .map(|r| (0..32).fold(0, |acc, z| acc ^ (d3[r][z] & y2[z])))
        .collect()
}

/// docs/bridge.md Section 9 items 2 and 3 on the Flock side: the toy slot (`L = 2^12`, one
/// lane) proves `Y = M_3 M_2 M_1 b'` against a direct GF(2) computation; a flipped `Y` bit,
/// a flipped committed bit with the stale `Y`, and another seed are rejected.
#[test]
#[cfg_attr(debug_assertions, ignore = "Flock proofs; run with --release")]
fn toy_bridge_slot_round_trip_and_tampering() {
    let layout = SlotLayout::new(gf2_walker::SegmentLayout::new(Shape::TOY, 1 << 12), 1 << 12);
    let batch = BridgeBatch::new(layout, Profile::Fast);
    let mut rng = Rng::new(21);
    let bits: Vec<u64> = (0..64).map(|_| rng.next_u64()).collect();

    // Event 2: commit; event 3: sd from the transcript; then Y and fs.
    let committed = batch.commit_bits(&bits);
    let commitment = committed.commitment().clone();
    let sd = seed_after(&commitment);
    let mats = Arc::new(derive_matrices_for(&Shape::TOY, &sd));
    let y = batch.final_vector(&committed, &mats);
    assert_eq!(
        y,
        direct_toy_product(&mats, &bits),
        "Y against the direct product"
    );
    let fs = fs_after(&commitment, &sd);
    let proof = batch.prove(committed, sd, mats.clone(), y.clone(), &fs);
    batch
        .verify(&commitment, sd, mats.clone(), &y, &fs, &proof)
        .expect("honest toy slot verifies");
    eprintln!(
        "toy bridge slot: k_log {}, proof {} bytes",
        layout.k_log(),
        proof.bytes.len()
    );

    // A flipped Y bit.
    for o in [0, 7, 15] {
        let mut bad = y.clone();
        bad[o] ^= 1;
        assert!(batch
            .verify(&commitment, sd, mats.clone(), &bad, &fs, &proof)
            .is_err());
    }
    // Another seed: other matrices (and digest), same Y.
    let other = [0x55u8; 32];
    let other_mats = Arc::new(derive_matrices_for(&Shape::TOY, &other));
    assert!(batch
        .verify(&commitment, other, other_mats, &y, &fs, &proof)
        .is_err());
    // The right matrices under a different seed in the digest.
    assert!(batch
        .verify(&commitment, other, mats.clone(), &y, &fs, &proof)
        .is_err());
    // Malformed Y.
    assert!(batch
        .verify(&commitment, sd, mats.clone(), &y[..15], &fs, &proof)
        .is_err());

    // One committed bit flipped (the GF(2) copy differs from the bits Y was computed from):
    // the prover can only prove the stale Y, which the verifier rejects.
    let mut flipped = bits.clone();
    flipped[1] ^= 1 << 9;
    let committed = batch.commit_bits(&flipped);
    let commitment = committed.commitment().clone();
    let sd2 = seed_after(&commitment);
    let mats2 = Arc::new(derive_matrices_for(&Shape::TOY, &sd2));
    let honest_y = direct_toy_product(&mats2, &bits);
    let tampered_y = batch.final_vector(&committed, &mats2);
    assert_ne!(
        honest_y, tampered_y,
        "a single flip changes Y (column 73 does not vanish)"
    );
    let fs2 = fs_after(&commitment, &sd2);
    let stale = batch.prove(committed, sd2, mats2.clone(), honest_y.clone(), &fs2);
    assert!(batch
        .verify(&commitment, sd2, mats2.clone(), &honest_y, &fs2, &stale)
        .is_err());
}

/// Lane `t` of the main domain of `records`: `b_t[j + 256 rho + 1024 i]` is stream bit
/// `1536 rho + 6 j + t` of record `i`'s w1Encode (FIPS 204 Algorithms 16 and 28).
fn main_domain_lanes(records: &[SpongeRecord], len: usize) -> Vec<Vec<i8>> {
    (0..6)
        .map(|t| {
            (0..len)
                .map(|x| {
                    let (i, pos) = (x / 1024, x % 1024);
                    if i >= records.len() {
                        return 0;
                    }
                    let (rho, j) = (pos / 256, pos % 256);
                    let q = 1536 * rho + 6 * j + t;
                    ((records[i].w1_encode[q / 8] >> (q % 8)) & 1) as i8
                })
                .collect()
        })
        .collect()
}

fn bridged_round_trip(n: usize, rng_seed: u64, check_tampering: bool) {
    let prover = agg_keccak::prover::SpongeProver::new(Default::default());
    let gate = prover.gate();
    let d_seg = BridgedSegment::d_seg_for(n);
    let seg = BridgedSegment::new(gate, n, d_seg, Profile::Fast);
    let mut rng = Rng::new(rng_seed);
    let records: Vec<SpongeRecord> = (0..n).map(|_| random_record(&mut rng)).collect();

    let t0 = Instant::now();
    let committed = seg.commit(&records);
    let t_commit = t0.elapsed();
    let commitment = committed.commitment().clone();
    let sd = seed_after(&commitment);
    let t1 = Instant::now();
    let mats = Arc::new(derive_matrices_for(&Shape::PRODUCTION, &sd));
    let t_mats = t1.elapsed();
    let fs = fs_after(&commitment, &sd);
    let c_tilde = committed.c_tilde().to_vec();
    let mu: Vec<[u8; 64]> = records.iter().map(|r| r.mu).collect();
    let t2 = Instant::now();
    let out = seg.prove(committed, sd, mats.clone(), &fs);
    let t_prove = t2.elapsed();
    let t3 = Instant::now();
    seg.verify(
        &mu,
        &c_tilde,
        &commitment,
        sd,
        mats.clone(),
        &out.final_vector,
        &fs,
        &out.proof,
    )
    .expect("honest bridged segment verifies");
    let t_verify = t3.elapsed();

    // Y against the F_p side's main-domain walk on the same bits (agg-witness).
    let geom = Geometry::new(n).unwrap();
    let lanes = main_domain_lanes(&records, geom.main_len());
    let b: [&[i8]; 6] = std::array::from_fn(|t| lanes[t].as_slice());
    assert_eq!(
        out.final_vector,
        gf2_composite(&geom, &b, &mats),
        "Y of both sides"
    );
    for (r, c) in records.iter().zip(&c_tilde) {
        assert_eq!(&r.c_tilde(), c);
    }
    assert_eq!(pack_y(&out.final_vector).len(), 512);

    // The nonzeros the bridge adds to the sponge relation.
    let y_weight = out.final_vector.iter().filter(|&&b| b == 1).count();
    let cost = seg.cost(y_weight);
    let flattened = gf2_walker::composite_nonzeros(&seg.layout().segment, &mats);
    eprintln!(
        "bridged segment n = {n} (d_seg {d_seg}, nu {}, M {}, dense m {}): sponge nonzeros {} \
         ({} per record); bridge adds factored {} + identity {} = {} nonzeros ({:.2}% of the \
         sponge), flattened composite A entries {} ({:.1}x the sponge); {} copy wires; bridge \
         k_log {}, committed words sponge {} + bridge {}; commit {:.2?}, matrices {:.2?}, \
         prove {:.2?}, verify {:.2?}, proof {} bytes, commitment {} bytes",
        seg.nu(),
        seg.m_total(),
        seg.dense_m(),
        cost.sponge_nonzeros,
        cost.sponge_nonzeros / n as u64,
        cost.bridge_factored_nonzeros,
        cost.bridge_identity_nonzeros,
        cost.bridge_factored_nonzeros + cost.bridge_identity_nonzeros,
        100.0 * (cost.bridge_factored_nonzeros + cost.bridge_identity_nonzeros) as f64
            / cost.sponge_nonzeros as f64,
        flattened,
        flattened as f64 / cost.sponge_nonzeros as f64,
        cost.copy_wires,
        cost.bridge_k_log,
        cost.sponge_committed_words,
        cost.bridge_committed_words,
        t_commit,
        t_mats,
        t_prove,
        t_verify,
        out.proof.bytes.len(),
        commitment.as_bytes().len(),
    );
    let lanes_len = seg.layout().segment.lane_len;
    assert_eq!(
        cost.bridge_factored_nonzeros,
        (14 * 6 * lanes_len + 14 * 6 * 8192 * lanes_len.div_ceil(1 << 19) + 18 * 49_152) as u64
    );
    assert!(flattened > 0);

    // A flipped Y bit; another seed; a wrong c~.
    let mut bad = out.final_vector.clone();
    bad[4095] ^= 1;
    assert!(seg
        .verify(
            &mu,
            &c_tilde,
            &commitment,
            sd,
            mats.clone(),
            &bad,
            &fs,
            &out.proof
        )
        .is_err());
    let other = [0xA5u8; 32];
    let other_mats = Arc::new(derive_matrices_for(&Shape::PRODUCTION, &other));
    assert!(seg
        .verify(
            &mu,
            &c_tilde,
            &commitment,
            other,
            other_mats,
            &out.final_vector,
            &fs,
            &out.proof
        )
        .is_err());
    let mut bad_c = c_tilde.clone();
    bad_c[n - 1][0] ^= 1;
    assert!(seg
        .verify(
            &mu,
            &bad_c,
            &commitment,
            sd,
            mats.clone(),
            &out.final_vector,
            &fs,
            &out.proof
        )
        .is_err());

    if !check_tampering {
        return;
    }
    // The same records without the bridge slot (the sponge sub-proof of docs/keccak.md), for
    // the added prover and verifier time and proof bytes.
    let t4 = Instant::now();
    let plain = prover.prove_batch(&records, &fs);
    let t_plain = t4.elapsed();
    let t5 = Instant::now();
    prover
        .verify_batch(&mu, &c_tilde, &plain.commitment, &plain.proof, &fs)
        .expect("plain sponge sub-proof verifies");
    eprintln!(
        "same records without the bridge slot: prove {t_plain:.2?} (commit included), verify {:.2?}, proof {} bytes",
        t5.elapsed(),
        plain.proof.bytes.len()
    );
    // A bit flipped in the bridge copy only, with Y recomputed from the tampered copy (so
    // the bridge rows hold): the copy constraint to the sponge rows fails.
    let (sponge_buf, mut bridge_z) = seg.honest_parts(&records);
    let c = seg.layout().segment.column(3, 1024 * (n - 1) + 77);
    bridge_z[c / 64] ^= 1 << (c % 64);
    let committed = seg.commit_parts(sponge_buf, bridge_z, &records);
    let commitment = committed.commitment().clone();
    let sd = seed_after(&commitment);
    let mats = Arc::new(derive_matrices_for(&Shape::PRODUCTION, &sd));
    let fs = fs_after(&commitment, &sd);
    let out = seg.prove(committed, sd, mats.clone(), &fs);
    assert!(seg
        .verify(
            &mu,
            &c_tilde,
            &commitment,
            sd,
            mats.clone(),
            &out.final_vector,
            &fs,
            &out.proof
        )
        .is_err());

    // A bit flipped in both copies (the sponge recomputed on the tampered record), Y
    // consistent with it, the published c~ stale: the sponge rows fail.
    let mut tampered = records.clone();
    tampered[0].w1_encode[100] ^= 0x10;
    let (sponge_buf, bridge_z) = seg.honest_parts(&tampered);
    let committed = seg.commit_parts(sponge_buf, bridge_z, &records);
    let commitment = committed.commitment().clone();
    let sd = seed_after(&commitment);
    let mats = Arc::new(derive_matrices_for(&Shape::PRODUCTION, &sd));
    let fs = fs_after(&commitment, &sd);
    let out = seg.prove(committed, sd, mats.clone(), &fs);
    assert!(seg
        .verify(
            &mu,
            &c_tilde,
            &commitment,
            sd,
            mats.clone(),
            &out.final_vector,
            &fs,
            &out.proof
        )
        .is_err());
}

/// The sponge proof at 64 signatures with the bridge rows (`d_seg = 6`, `k_log = 19` for
/// both types): round trip, `Y` equal to the F_p side's GF(2) walk, the added nonzeros
/// reported, and the tampering cases of docs/bridge.md Section 9 item 6.
#[test]
#[cfg_attr(debug_assertions, ignore = "heavy Flock proof; run with --release")]
fn sponge_with_bridge_rows_round_trip_64() {
    bridged_round_trip(64, 0xB1D6E, true);
}

/// A small segment with padded record positions (`n = 5`, `d_seg = 3`, `k_log` 16 for the
/// bridge type): the padded positions are forced zero and still enter the composite rows.
#[test]
#[cfg_attr(debug_assertions, ignore = "heavy Flock proof; run with --release")]
fn bridged_segment_with_padded_records_round_trip() {
    bridged_round_trip(5, 0x5EED5, false);
}

/// Statement checks that need no proof.
#[test]
fn bridged_segment_shapes() {
    let gate = sponge::sponge_gate();
    assert_eq!(BridgedSegment::d_seg_for(1), 3);
    assert_eq!(BridgedSegment::d_seg_for(64), 6);
    assert_eq!(BridgedSegment::d_seg_for(100), 7);
    let seg = BridgedSegment::new(&gate, 100, 7, Profile::Fast);
    assert_eq!(seg.layout().k_log(), 20);
    assert_eq!(seg.layout().live_records(), 100);
    let cost = seg.cost(0);
    assert_eq!(cost.copy_wires, 4_800);
    assert_eq!(cost.bridge_committed_words, (6 << 17) / 128 + 1 + 32);
}
