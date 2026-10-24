//! The tied part (docs/keccak.md Section 9, docs/bridge.md Section 5a): a sponge instance and
//! a bridge instance per part, tied by evaluation claims. Round trips at 5 records (padded
//! positions, `d = 3`) and 64 records (`d = 6`), the tampering cases of docs/bridge.md
//! Section 9 item 6, and a segment split into parts whose shares XOR to the segment's `Y`
//! (deviation K10). The bridge seed is derived from the commitments, so the witnesses are
//! fixed before the seed exists (spec Section 10, events 2 and 3). Run with `--release`.

#![allow(clippy::needless_range_loop)]

use std::sync::Arc;
use std::time::Instant;

use agg_bridge::{seed::derive_matrices_for, shape::Shape};
use agg_keccak::prover::{KeccakError, KeccakProof, Profile};
use agg_keccak::{
    bridge_slot::pack_y,
    sponge::{self, SpongeRecord, W1_INPUT_BASE},
    tied::{TiedCommitment, TiedPart},
    witness::pack_slot,
};
use agg_witness::bridge::gf2_composite;
use agg_witness::padding::Geometry;
use flock_core::test_rng::Rng;
use sha3::{Digest, Sha3_256};

/// The bridge seed of a test transcript: a hash of the absorbed commitments (event 2 then
/// event 3), so it cannot be known before the witnesses are committed.
fn seed_after(commitment: &TiedCommitment) -> [u8; 32] {
    let mut h = Sha3_256::new();
    h.update(b"test/sd");
    h.update(commitment.to_bytes());
    h.finalize().into()
}

/// The Flock seed `fs` of the test transcript (after `sd`).
fn fs_after(commitment: &TiedCommitment, sd: &[u8; 32]) -> [u8; 32] {
    let mut h = Sha3_256::new();
    h.update(b"test/fs");
    h.update(commitment.to_bytes());
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

/// The main-domain lanes `b_t[x]` of the records (the F_p side's view, `x = j + 256 rho +
/// 1024 i`), padded to `len`.
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

fn statement(records: &[SpongeRecord]) -> (Vec<[u8; 64]>, Vec<[u8; 32]>) {
    (
        records.iter().map(|r| r.mu).collect(),
        records.iter().map(SpongeRecord::c_tilde).collect(),
    )
}

fn tied_round_trip(n: usize, rng_seed: u64, check_tampering: bool) {
    let gate = sponge::sponge_gate();
    let d = TiedPart::d_seg_for(n);
    let part = TiedPart::new(&gate, n, d, Profile::Fast);
    let mut rng = Rng::new(rng_seed);
    let records: Vec<SpongeRecord> = (0..n).map(|_| random_record(&mut rng)).collect();
    let (mu, c_tilde) = statement(&records);

    let t0 = Instant::now();
    let committed = part.commit(&records);
    let t_commit = t0.elapsed();
    let commitment = committed.commitment().clone();
    let sd = seed_after(&commitment);
    let t1 = Instant::now();
    let mats = Arc::new(derive_matrices_for(&Shape::PRODUCTION, &sd));
    let t_mats = t1.elapsed();
    let fs = fs_after(&commitment, &sd);
    assert_eq!(committed.c_tilde(), &c_tilde[..]);
    let t2 = Instant::now();
    let out = part.prove(committed, sd, mats.clone(), &fs);
    let t_prove = t2.elapsed();
    let verify = |mu: &[[u8; 64]],
                  c: &[[u8; 32]],
                  com: &TiedCommitment,
                  sd,
                  mats,
                  y: &[u8],
                  fs: &[u8; 32],
                  proof| { part.verify(mu, c, com, sd, mats, y, fs, proof) };
    let t3 = Instant::now();
    verify(
        &mu,
        &c_tilde,
        &commitment,
        sd,
        mats.clone(),
        &out.final_vector,
        &fs,
        &out.proof,
    )
    .expect("honest tied part verifies");
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
    assert_eq!(pack_y(&out.final_vector).len(), 512);

    let cost = part.cost();
    eprintln!(
        "tied part n = {n} (d {d}): sponge nu {} M {} dense m {} ({} committed words); bridge \
         k_log {} M {} dense m {} ({} committed words, {} factored nonzeros); commit {:.2?}, \
         matrices {:.2?}, prove {:.2?}, verify {:.2?}, proof {} bytes, commitments {} + {} bytes",
        cost.sponge_nu,
        cost.sponge_m,
        cost.sponge_dense_m,
        cost.sponge_committed_words,
        cost.bridge_k_log,
        cost.bridge_m,
        cost.bridge_dense_m,
        cost.bridge_committed_words,
        cost.bridge_factored_nonzeros,
        t_commit,
        t_mats,
        t_prove,
        t_verify,
        out.proof.bytes.len(),
        commitment.sponge.as_bytes().len(),
        commitment.bridge.as_bytes().len(),
    );

    // A flipped Y bit; another seed; a wrong c~; a wrong mu; another fs; a flipped proof byte.
    let mut bad = out.final_vector.clone();
    bad[4095] ^= 1;
    assert!(verify(
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
    assert!(verify(
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
    // A wrong c~ or mu changes the public digest the sponge challenger absorbs, so Flock's
    // transcript already rejects (the public claim would, too).
    let mut bad_c = c_tilde.clone();
    bad_c[n - 1][0] ^= 1;
    assert!(verify(
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
    let mut bad_mu = mu.clone();
    bad_mu[0][63] ^= 0x80;
    assert!(verify(
        &bad_mu,
        &c_tilde,
        &commitment,
        sd,
        mats.clone(),
        &out.final_vector,
        &fs,
        &out.proof
    )
    .is_err());
    let other_fs = [0x3Cu8; 32];
    assert!(verify(
        &mu,
        &c_tilde,
        &commitment,
        sd,
        mats.clone(),
        &out.final_vector,
        &other_fs,
        &out.proof
    )
    .is_err());
    let mut flipped = out.proof.clone();
    let k = flipped.bytes.len() / 2;
    flipped.bytes[k] ^= 1;
    assert!(verify(
        &mu,
        &c_tilde,
        &commitment,
        sd,
        mats.clone(),
        &out.final_vector,
        &fs,
        &flipped
    )
    .is_err());
    // Swapped caps (same bytes, other roles): malformed lengths or a rejection.
    let swapped = TiedCommitment {
        sponge: commitment.bridge.clone(),
        bridge: commitment.sponge.clone(),
    };
    assert!(verify(
        &mu,
        &c_tilde,
        &swapped,
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
    // A bit flipped in the bridge copy only, with Y recomputed from the tampered copy (so the
    // bridge rows hold): the tie to the sponge instance fails.
    let (sponge_buf, mut bridge_z) = part.honest_parts(&records);
    let c = part.layout().segment.column(3, 1024 * (n - 1) + 77);
    bridge_z[c / 64] ^= 1 << (c % 64);
    let committed = part.commit_parts(sponge_buf, bridge_z, &records);
    let commitment = committed.commitment().clone();
    let sd = seed_after(&commitment);
    let mats = Arc::new(derive_matrices_for(&Shape::PRODUCTION, &sd));
    let fs = fs_after(&commitment, &sd);
    let out = part.prove(committed, sd, mats.clone(), &fs);
    assert!(matches!(
        verify(
            &mu,
            &c_tilde,
            &commitment,
            sd,
            mats.clone(),
            &out.final_vector,
            &fs,
            &out.proof
        ),
        Err(KeccakError::Tie("bridge copy of lanes 0..3"))
    ));

    // A bit flipped in both copies (the sponge recomputed on the tampered record), Y
    // consistent with it, the published c~ stale: the sponge instance's public claim fails.
    let mut tampered = records.clone();
    tampered[0].w1_encode[100] ^= 0x10;
    let (sponge_buf, bridge_z) = part.honest_parts(&tampered);
    let committed = part.commit_parts(sponge_buf, bridge_z, &records);
    let commitment = committed.commitment().clone();
    let sd = seed_after(&commitment);
    let mats = Arc::new(derive_matrices_for(&Shape::PRODUCTION, &sd));
    let fs = fs_after(&commitment, &sd);
    let out = part.prove(committed, sd, mats.clone(), &fs);
    assert!(matches!(
        verify(
            &mu,
            &c_tilde,
            &commitment,
            sd,
            mats.clone(),
            &out.final_vector,
            &fs,
            &out.proof
        ),
        Err(KeccakError::Tie("public words"))
    ));

    // A sponge input bit flipped with the stale chi witness (the bridge copy honest, Y of the
    // honest bits): the sponge rows fail.
    let inputs: Vec<Vec<u64>> = records.iter().map(SpongeRecord::input_bits).collect();
    let col = gate.layout().input_col(W1_INPUT_BASE + 4000);
    let sponge_buf = pack_slot(&gate, part.nu(), n, |r, zz, a, b| {
        gate.honest_z(&inputs[r], zz);
        if r == n - 1 {
            zz[col / 64] ^= 1 << (col % 64);
        }
        gate.apply_ab(zz, a, b);
    });
    let (_, bridge_z) = part.honest_parts(&records);
    let committed = part.commit_parts(sponge_buf, bridge_z, &records);
    let commitment = committed.commitment().clone();
    let sd = seed_after(&commitment);
    let mats = Arc::new(derive_matrices_for(&Shape::PRODUCTION, &sd));
    let fs = fs_after(&commitment, &sd);
    let out = part.prove(committed, sd, mats.clone(), &fs);
    assert!(matches!(
        verify(
            &mu,
            &c_tilde,
            &commitment,
            sd,
            mats.clone(),
            &out.final_vector,
            &fs,
            &out.proof
        ),
        Err(KeccakError::Flock(_))
    ));
}

/// 64 records (`d = 6`): round trip, `Y` equal to the F_p side's GF(2) walk, and the
/// tampering cases of docs/bridge.md Section 9 item 6 with the tie's error names.
#[test]
#[cfg_attr(debug_assertions, ignore = "heavy Flock proof; run with --release")]
fn tied_part_round_trip_64() {
    tied_round_trip(64, 0xB1D6E, true);
}

/// A small part with padded record positions (`n = 5`, `d = 3`, sponge `nu = 3`).
#[test]
#[cfg_attr(debug_assertions, ignore = "heavy Flock proof; run with --release")]
fn tied_part_with_padded_records_round_trip() {
    tied_round_trip(5, 0x5EED5, false);
}

/// Statement checks that need no proof.
#[test]
fn tied_part_shapes() {
    let gate = sponge::sponge_gate();
    assert_eq!(TiedPart::d_seg_for(1), 3);
    assert_eq!(TiedPart::d_seg_for(64), 6);
    assert_eq!(TiedPart::d_seg_for(100), 7);
    let part = TiedPart::new(&gate, 100, 7, Profile::Fast);
    assert_eq!(part.layout().k_log(), 20);
    assert_eq!(part.layout().live_records(), 100);
    let cost = part.cost();
    assert_eq!((cost.sponge_nu, cost.sponge_m), (7, 26));
    assert_eq!((cost.bridge_k_log, cost.bridge_m), (20, 23));
    assert_eq!(cost.bridge_committed_words, (6 << 17) / 128 + 1 + 32);
    assert_eq!(cost.sponge_committed_words, 100 * 2_156);
    // A full segment: sponge M = 32, bridge k_log 26 and M = 29 (docs/keccak.md Section 9).
    let full = TiedPart::new(&gate, 1 << 13, 13, Profile::Fast);
    let cost = full.cost();
    assert_eq!(
        (cost.sponge_nu, cost.sponge_m, cost.sponge_dense_m),
        (13, 32, 32)
    );
    assert_eq!(
        (cost.bridge_k_log, cost.bridge_m, cost.bridge_dense_m),
        (26, 29, 26)
    );
    // A part proof must carry its part's count and d.
    let other = TiedPart::new_part(&gate, 100, 7, 1, Profile::Fast);
    assert_eq!(other.layout().segment.offset, 1 << 17);
    assert!(matches!(
        part.verify(
            &[[0; 64]; 100],
            &[[0; 32]; 100],
            &TiedCommitment {
                sponge: agg_keccak::prover::KeccakCommitment::from_bytes(Vec::new()),
                bridge: agg_keccak::prover::KeccakCommitment::from_bytes(Vec::new())
            },
            [0; 32],
            Arc::new(derive_matrices_for(&Shape::PRODUCTION, &[0; 32])),
            &[0; 4096],
            &[0; 32],
            &KeccakProof {
                bytes: b"AGGKCTxx".to_vec()
            }
        ),
        Err(KeccakError::Malformed(_))
    ));
}

/// A segment split into parts (deviation K10): 20 live records (`d_seg = 5`) in parts of 8
/// record positions (`d = 3`; 8, 8 and 4 live records). Each part commits before the seed,
/// proves its share, and the shares XOR to the F_p side's `Y_sigma` of the whole segment.
/// The buffers are dropped after the commitment and rebuilt for the proof (`reattach`). A
/// share proven under another part's index, swapped shares, and a part proven against the
/// whole segment's `Y` are rejected.
#[test]
#[cfg_attr(debug_assertions, ignore = "Flock proofs; run with --release")]
fn tied_parts_prove_shares_that_xor_to_the_segment_vector() {
    let gate = sponge::sponge_gate();
    let (n, d_part) = (20_usize, 3_u32);
    let mut rng = Rng::new(0xA875);
    let records: Vec<SpongeRecord> = (0..n).map(|_| random_record(&mut rng)).collect();
    let parts: Vec<(usize, std::ops::Range<usize>)> = (0..n.div_ceil(8))
        .map(|s| (s, 8 * s..(8 * s + 8).min(n)))
        .collect();
    let segs: Vec<TiedPart<'_>> = parts
        .iter()
        .map(|(s, r)| TiedPart::new_part(&gate, r.len(), d_part, *s, Profile::Fast))
        .collect();
    assert_eq!(segs[2].layout().live_records(), 4);
    assert_eq!(segs[1].layout().segment.offset, 1 << 13);
    // Event 2: commit every part, keep only the commitments.
    let commitments: Vec<TiedCommitment> = segs
        .iter()
        .zip(&parts)
        .map(|(seg, (_, r))| seg.commit(&records[r.clone()]).commitment().clone())
        .collect();
    let mut h = Sha3_256::new();
    for c in &commitments {
        h.update(c.to_bytes());
    }
    let sd: [u8; 32] = h.finalize().into();
    let mats = Arc::new(derive_matrices_for(&Shape::PRODUCTION, &sd));
    // The F_p side's Y of the whole segment (main-domain walk of agg-witness).
    let geom = Geometry::new(n).unwrap();
    let lanes = main_domain_lanes(&records, geom.main_len());
    let b: [&[i8]; 6] = std::array::from_fn(|t| lanes[t].as_slice());
    let y_segment = gf2_composite(&geom, &b, &mats);
    // Event 6: rebuild each part, prove its share; the last share is Y XOR the others.
    let rebuild = |k: usize| {
        let r = parts[k].1.clone();
        let (sponge, bridge_z) = segs[k].honest_parts(&records[r.clone()]);
        segs[k].reattach(sponge, bridge_z, &records[r], commitments[k].clone())
    };
    let fs = |k: usize| -> [u8; 32] { std::array::from_fn(|i| (i + 7 * k) as u8) };
    let mut shares = Vec::new();
    let mut acc = vec![0_u8; 4096];
    for k in 0..parts.len() {
        let committed = rebuild(k);
        let own = segs[k].final_vector(&committed, &mats);
        let share = if k + 1 < parts.len() {
            own.clone()
        } else {
            acc.iter().zip(&y_segment).map(|(a, y)| a ^ y).collect()
        };
        assert_eq!(share, own, "honest shares XOR to the segment's Y");
        acc.iter_mut().zip(&share).for_each(|(a, s)| *a ^= s);
        let proof =
            segs[k].prove_with_final_vector(committed, sd, mats.clone(), share.clone(), &fs(k));
        shares.push((share, proof));
    }
    assert_eq!(acc, y_segment);
    let verify = |seg: &TiedPart<'_>, k: usize, y: &[u8], proof| {
        let (mu, c) = statement(&records[parts[k].1.clone()]);
        seg.verify(&mu, &c, &commitments[k], sd, mats.clone(), y, &fs(k), proof)
    };
    for k in 0..parts.len() {
        verify(&segs[k], k, &shares[k].0, &shares[k].1).expect("honest part");
    }
    // Part 1's proof checked as part 0 (same live count, another offset): rejected.
    assert!(verify(&segs[0], 1, &shares[1].0, &shares[1].1).is_err());
    // Swapped shares of parts 0 and 1 (their XOR is unchanged): both rejected.
    assert!(verify(&segs[0], 0, &shares[1].0, &shares[0].1).is_err());
    assert!(verify(&segs[1], 1, &shares[0].0, &shares[1].1).is_err());
    // A part checked against the whole segment's Y: rejected.
    assert!(verify(&segs[2], 2, &y_segment, &shares[2].1).is_err());
}
