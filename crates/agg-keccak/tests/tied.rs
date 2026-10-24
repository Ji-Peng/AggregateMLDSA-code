//! The tied part (docs/keccak.md Sections 9 to 11, docs/bridge.md Sections 5a and 5b): a
//! sponge instance and a two-type bridge union per part, tied by evaluation claims. Round
//! trips at 5 records (padded positions, `d = 3`), 64 (`d = 6`), 256 (`d = 8`), 512 (`d = 9`,
//! one full level-1 block) and 1,024 (`d = 10`, two level-1 blocks per lane), the tampering
//! cases of docs/bridge.md Section 9 item 6 with the tie's error names, the precomputed
//! commitment (K14), and a segment split into parts whose shares XOR to the segment's `Y`
//! (deviation K10). The bridge seed is derived from the event-2 commitments, so the sponge
//! witnesses and the part's bits are fixed before the seed exists (spec Section 10, events 2
//! and 3). Run with `--release`.

#![allow(clippy::needless_range_loop)]

use std::sync::Arc;
use std::time::Instant;

use agg_bridge::{seed::derive_matrices_for, shape::Shape};
use agg_keccak::prover::{KeccakCommitment, KeccakError, KeccakProof, Profile};
use agg_keccak::{
    bridge_slot::pack_y,
    sponge::{self, SpongeRecord, W1_INPUT_BASE},
    tied::{TiedCommitment, TiedPart},
    witness::pack_slot,
};
use agg_witness::bridge::{gf2_composite, BridgeMatrices};
use agg_witness::padding::Geometry;
use flock_core::test_rng::Rng;
use sha3::{Digest, Sha3_256};

/// The bridge seed of a test transcript: a hash of the absorbed commitment (event 2 then
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

/// Which share a tampered run proves.
#[derive(Clone, Copy)]
enum Share {
    /// The share of the proven (tampered) blocks.
    OfProven,
    /// The share of the honest blocks.
    Stale,
    /// The share of the proven blocks with bit `o` flipped.
    Flipped(usize),
}

/// One proof of a part and its verification inputs.
struct Run {
    commitment: TiedCommitment,
    sd: [u8; 32],
    mats: Arc<BridgeMatrices>,
    fs: [u8; 32],
    y: Vec<u8>,
    proof: KeccakProof,
}

/// The test context of one part.
struct Ctx<'g> {
    part: TiedPart<'g>,
    records: Vec<SpongeRecord>,
    mu: Vec<[u8; 64]>,
    c_tilde: Vec<[u8; 32]>,
}

impl Ctx<'_> {
    fn verify(&self, r: &Run) -> Result<(), KeccakError> {
        self.verify_with(r, &self.mu, &self.c_tilde, &r.commitment, r.sd, &r.y, &r.fs)
    }

    #[allow(clippy::too_many_arguments)]
    fn verify_with(
        &self,
        r: &Run,
        mu: &[[u8; 64]],
        c: &[[u8; 32]],
        com: &TiedCommitment,
        sd: [u8; 32],
        y: &[u8],
        fs: &[u8; 32],
    ) -> Result<(), KeccakError> {
        let mats = if sd == r.sd {
            r.mats.clone()
        } else {
            Arc::new(derive_matrices_for(&Shape::PRODUCTION, &sd))
        };
        self.part.verify(mu, c, com, sd, &mats, y, fs, &r.proof)
    }

    /// Commit `sponge` and `bits` (event 2), derive `sd` and `fs` from the commitment, and
    /// prove with the honest blocks changed by `tamper` and the chosen share.
    fn run(
        &self,
        sponge: agg_keccak::witness::SlotBuffers,
        bits: Vec<u64>,
        tamper: &dyn Fn(&mut agg_keccak::bridge_union::BridgeBlocks),
        share: Share,
    ) -> Run {
        let committed = self.part.commit_parts(sponge, bits, &self.records);
        let commitment = committed.commitment().clone();
        let sd = seed_after(&commitment);
        let mats = Arc::new(derive_matrices_for(&Shape::PRODUCTION, &sd));
        let fs = fs_after(&commitment, &sd);
        let honest = self.part.honest_blocks(&committed, &mats);
        let mut blocks = honest.clone();
        tamper(&mut blocks);
        let source = if matches!(share, Share::Stale) {
            &honest
        } else {
            &blocks
        };
        let mut y = self.part.bridge().share(&mats, source);
        if let Share::Flipped(o) = share {
            y[o] ^= 1;
        }
        let proof = self
            .part
            .prove_with_blocks(committed, sd, &mats, blocks, y.clone(), &fs);
        Run {
            commitment,
            sd,
            mats,
            fs,
            y,
            proof,
        }
    }
}

/// How much tampering a round trip checks.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tamper {
    /// None (round trip and the cheap rejections only).
    None,
    /// The bridge-union cases that depend on the block geometry: a level-1 input copy (tie
    /// 1) and a parity copy (tie 2) in the last level-1 block, and the output columns.
    Blocks,
    /// Every case, plus the precomputed commitment (K14).
    Full,
}

fn tied_round_trip(n: usize, rng_seed: u64, tamper: Tamper) {
    let gate = sponge::sponge_gate();
    let d = TiedPart::d_seg_for(n);
    let part = TiedPart::new(&gate, n, d, Profile::Fast);
    let mut rng = Rng::new(rng_seed);
    let records: Vec<SpongeRecord> = (0..n).map(|_| random_record(&mut rng)).collect();
    let (mu, c_tilde) = statement(&records);
    let ctx = Ctx {
        part,
        records,
        mu,
        c_tilde,
    };
    let part = &ctx.part;

    let t0 = Instant::now();
    let committed = part.commit(&ctx.records);
    let t_commit = t0.elapsed();
    let commitment = committed.commitment().clone();
    let sd = seed_after(&commitment);
    let t1 = Instant::now();
    let mats = Arc::new(derive_matrices_for(&Shape::PRODUCTION, &sd));
    let t_mats = t1.elapsed();
    let fs = fs_after(&commitment, &sd);
    assert_eq!(committed.c_tilde(), &ctx.c_tilde[..]);
    let t2 = Instant::now();
    let out = part.prove(committed, sd, &mats, &fs);
    let t_prove = t2.elapsed();
    let run = Run {
        commitment: commitment.clone(),
        sd,
        mats: mats.clone(),
        fs,
        y: out.final_vector.clone(),
        proof: out.proof.clone(),
    };
    let t3 = Instant::now();
    ctx.verify(&run).expect("honest tied part verifies");
    let t_verify = t3.elapsed();

    // Y against the F_p side's main-domain walk on the same bits (agg-witness).
    let geom = Geometry::new(n).unwrap();
    let lanes = main_domain_lanes(&ctx.records, geom.main_len());
    let b: [&[i8]; 6] = std::array::from_fn(|t| lanes[t].as_slice());
    assert_eq!(
        out.final_vector,
        gf2_composite(&geom, &b, &mats),
        "Y of both sides"
    );
    assert_eq!(pack_y(&out.final_vector).len(), 512);

    let cost = part.cost();
    // Proof layout: header (13), then framed bridge cap, sponge proof, bridge union proof.
    let framed =
        |at: usize| u32::from_le_bytes(out.proof.bytes[at..at + 4].try_into().unwrap()) as usize;
    let cap_len = framed(13);
    let sponge_len = framed(17 + cap_len);
    let bridge_len = framed(21 + cap_len + sponge_len);
    let parse = |at: usize, len: usize, claims: usize| {
        agg_keccak::flock_union::UnionProof::from_bytes(&out.proof.bytes[at..at + len], claims)
            .unwrap()
            .size_breakdown()
    };
    eprintln!(
        "  proof bytes n = {n}: bridge cap {cap_len}, sponge proof {sponge_len} (zerocheck, \
         lincheck, opening, values {:?}), bridge union proof {bridge_len} ({:?})",
        parse(21 + cap_len, sponge_len, 3),
        parse(25 + cap_len + sponge_len, bridge_len, 7)
    );
    eprintln!(
        "tied part n = {n} (d {d}): sponge nu {} M {} dense m {} ({} committed words); bridge \
         union nu {} k_log {:?} M {} dense m {} ({} committed words, walker {} entries, \
         one-row slot {}); commit {:.2?}, matrices {:.2?}, prove {:.2?}, verify {:.2?}, proof \
         {} bytes, commitment {} bytes",
        cost.sponge_nu,
        cost.sponge_m,
        cost.sponge_dense_m,
        cost.sponge_committed_words,
        cost.bridge.nu,
        cost.bridge.k_log,
        cost.bridge.m,
        cost.bridge.dense_m,
        cost.bridge.committed_words,
        cost.bridge.walker_entries,
        cost.slot_walker_entries,
        t_commit,
        t_mats,
        t_prove,
        t_verify,
        out.proof.bytes.len(),
        commitment.sponge.as_bytes().len(),
    );

    // A flipped Y bit; another seed; a wrong c~; a wrong mu; another fs; a flipped proof
    // byte (in the bridge cap, in the sponge proof, at the end of the bridge proof).
    let (mu, c) = (&ctx.mu, &ctx.c_tilde);
    let mut bad = out.final_vector.clone();
    bad[4095] ^= 1;
    assert!(ctx
        .verify_with(&run, mu, c, &commitment, sd, &bad, &fs)
        .is_err());
    assert!(ctx
        .verify_with(&run, mu, c, &commitment, [0xA5; 32], &run.y, &fs)
        .is_err());
    // A wrong c~ or mu changes the public digest the sponge challenger absorbs, so Flock's
    // transcript already rejects (the public claim would, too).
    let mut bad_c = c.clone();
    bad_c[n - 1][0] ^= 1;
    assert!(ctx
        .verify_with(&run, mu, &bad_c, &commitment, sd, &run.y, &fs)
        .is_err());
    let mut bad_mu = mu.clone();
    bad_mu[0][63] ^= 0x80;
    assert!(ctx
        .verify_with(&run, &bad_mu, c, &commitment, sd, &run.y, &fs)
        .is_err());
    assert!(ctx
        .verify_with(&run, mu, c, &commitment, sd, &run.y, &[0x3C; 32])
        .is_err());
    for at in [
        13 + 4 + 100,
        run.proof.bytes.len() / 2,
        run.proof.bytes.len() - 20,
    ] {
        let mut flipped = run.proof.clone();
        flipped.bytes[at] ^= 1;
        let r = Run {
            proof: flipped,
            ..run_clone(&run)
        };
        assert!(ctx.verify(&r).is_err(), "flipped proof byte {at}");
    }
    // The bridge union's cap (from the proof) in place of the event-2 sponge cap.
    let cap_len = part.bridge().geometry().commitment_len();
    let swapped = TiedCommitment {
        sponge: KeccakCommitment::from_bytes(run.proof.bytes[17..17 + cap_len].to_vec()),
    };
    assert_ne!(swapped, commitment);
    assert!(ctx
        .verify_with(&run, mu, c, &swapped, sd, &run.y, &fs)
        .is_err());

    if tamper == Tamper::None {
        return;
    }
    let full = tamper == Tamper::Full;
    if full {
        // The precomputed commitment (K14) gives the same proof as committing again (K7).
        let mut committed = part.commit(&ctx.records);
        let data = committed.take_commit_data();
        assert!(data.is_some());
        let again = part.prove(committed, sd, &mats, &fs);
        assert_eq!(again, out, "proof without the kept commitment");
        let (sponge_buf, bits) = part.honest_parts(&ctx.records);
        let reattached = part.reattach(sponge_buf, bits, &ctx.records, commitment.clone(), data);
        assert_eq!(part.prove(reattached, sd, &mats, &fs), out);
    }

    let honest = |_: &mut agg_keccak::bridge_union::BridgeBlocks| {};
    let expect = |r: &Run, label: &str, ok: &dyn Fn(&KeccakError) -> bool| match ctx.verify(r) {
        Err(e) if ok(&e) => eprintln!("  {label}: rejected ({e})"),
        other => panic!("{label}: {other:?}"),
    };
    // A bit flipped in the level-1 input copy only (lane 3 of the last record), the parities
    // and Y of the tampered copy (so every bridge row holds): tie 1 of lanes 0..3 fails.
    let (sponge_buf, mut bits) = part.honest_parts(&ctx.records);
    let col = part.segment().column(3, 1024 * (n - 1) + 77);
    bits[col / 64] ^= 1 << (col % 64);
    let r = ctx.run(sponge_buf, bits, &honest, Share::OfProven);
    expect(&r, "level-1 input copy", &|e| {
        matches!(e, KeccakError::Tie("bridge copy of lanes 0..3"))
    });
    // A parity copy flipped in the levels-2-3 row only (lane 5, the last block), Y of the
    // tampered copies: tie 2 of lanes 4, 5 fails.
    let top = *part.bridge().top();
    let blocks = part.bridge().level1().blocks();
    let flip_copy = move |b: &mut agg_keccak::bridge_union::BridgeBlocks| {
        let c = top.input_col(5, blocks - 1, 1234);
        b.top[c / 64] ^= 1 << (c % 64);
    };
    let (sponge_buf, bits) = part.honest_parts(&ctx.records);
    let r = ctx.run(sponge_buf, bits, &flip_copy, Share::OfProven);
    expect(&r, "parity copy", &|e| {
        matches!(e, KeccakError::Tie("level-1 parities of lanes 4, 5"))
    });
    if full {
        // A level-1 parity flipped in its row only (the copies and Y honest): Flock rejects
        // the level-1 rows.
        let level1 = *part.bridge().level1();
        let flip_parity = move |b: &mut agg_keccak::bridge_union::BridgeBlocks| {
            let c = level1.y1_col(4321);
            b.rows[level1.row(2, 0)][c / 64] ^= 1 << (c % 64);
        };
        let (sponge_buf, bits) = part.honest_parts(&ctx.records);
        let r = ctx.run(sponge_buf, bits, &flip_parity, Share::OfProven);
        expect(&r, "level-1 parity", &|e| {
            matches!(e, KeccakError::Flock(_))
        });
        // A share of the honest blocks against tampered copies: the levels-2-3 rows fail.
        let (sponge_buf, bits) = part.honest_parts(&ctx.records);
        let r = ctx.run(sponge_buf, bits, &flip_copy, Share::Stale);
        expect(&r, "stale share", &|e| matches!(e, KeccakError::Flock(_)));
    }
    // A wrong share whose difference is absorbed by the committed output columns (the
    // levels-2-3 rows only define them): every row holds, the output claim fails.
    let flip_out = move |b: &mut agg_keccak::bridge_union::BridgeBlocks| {
        let c = top.y_col(777);
        b.top[c / 64] ^= 1 << (c % 64);
    };
    let (sponge_buf, bits) = part.honest_parts(&ctx.records);
    let r = ctx.run(sponge_buf, bits, &flip_out, Share::Flipped(777));
    expect(&r, "wrong share, output columns set", &|e| {
        matches!(e, KeccakError::Tie("levels-2-3 output columns"))
    });
    if !full {
        return;
    }

    // A bit flipped in both copies (the sponge recomputed on the tampered record), Y
    // consistent with it, the published c~ stale: the sponge instance's public claim fails.
    let mut tampered = ctx.records.clone();
    tampered[0].w1_encode[100] ^= 0x10;
    let (sponge_buf, bits) = part.honest_parts(&tampered);
    let r = ctx.run(sponge_buf, bits, &honest, Share::OfProven);
    expect(&r, "both copies, stale c~", &|e| {
        matches!(e, KeccakError::Tie("public words"))
    });

    // A sponge input bit flipped with the stale chi witness (the bridge copy honest, Y of the
    // honest bits): the sponge rows fail.
    let inputs: Vec<Vec<u64>> = ctx.records.iter().map(SpongeRecord::input_bits).collect();
    let col = gate.layout().input_col(W1_INPUT_BASE + 4000);
    let sponge_buf = pack_slot(&gate, part.nu(), n, |r, zz, a, b| {
        gate.honest_z(&inputs[r], zz);
        if r == n - 1 {
            zz[col / 64] ^= 1 << (col % 64);
        }
        gate.apply_ab(zz, a, b);
    });
    let (_, bits) = part.honest_parts(&ctx.records);
    let r = ctx.run(sponge_buf, bits, &honest, Share::OfProven);
    expect(&r, "sponge input, stale chi", &|e| {
        matches!(e, KeccakError::Flock(_))
    });
}

fn run_clone(r: &Run) -> Run {
    Run {
        commitment: r.commitment.clone(),
        sd: r.sd,
        mats: r.mats.clone(),
        fs: r.fs,
        y: r.y.clone(),
        proof: r.proof.clone(),
    }
}

/// 64 records (`d = 6`): round trip, `Y` equal to the F_p side's GF(2) walk, the precomputed
/// commitment, and the tampering cases with the tie's error names.
#[test]
#[cfg_attr(debug_assertions, ignore = "heavy Flock proof; run with --release")]
fn tied_part_round_trip_64() {
    tied_round_trip(64, 0xB1D6E, Tamper::Full);
}

/// A small part with padded record positions (`n = 5`, `d = 3`, sponge `nu = 3`).
#[test]
#[cfg_attr(debug_assertions, ignore = "heavy Flock proof; run with --release")]
fn tied_part_with_padded_records_round_trip() {
    tied_round_trip(5, 0x5EED5, Tamper::None);
}

/// 256 records (`d = 8`) and 512 records (`d = 9`: exactly one level-1 block per lane).
#[test]
#[cfg_attr(debug_assertions, ignore = "heavy Flock proof; run with --release")]
fn tied_part_round_trips_256_and_512() {
    tied_round_trip(256, 0x256, Tamper::None);
    tied_round_trip(512, 0x512, Tamper::None);
}

/// 1,024 records (`d = 10`): two level-1 blocks per lane, the first part whose level-1 rows
/// carry a block coordinate; with the tampering cases that depend on the blocks (the
/// tampered input and parity copy sit in block 1).
#[test]
#[cfg_attr(debug_assertions, ignore = "heavy Flock proof; run with --release")]
fn tied_part_round_trip_1024_two_blocks() {
    tied_round_trip(1024, 0x1024, Tamper::Blocks);
}

/// Statement checks that need no proof.
#[test]
fn tied_part_shapes() {
    let gate = sponge::sponge_gate();
    assert_eq!(TiedPart::d_seg_for(1), 3);
    assert_eq!(TiedPart::d_seg_for(64), 6);
    assert_eq!(TiedPart::d_seg_for(100), 7);
    let part = TiedPart::new(&gate, 100, 7, Profile::Fast);
    assert_eq!(part.segment().lane_len, 1 << 17);
    let cost = part.cost();
    assert_eq!((cost.sponge_nu, cost.sponge_m), (7, 26));
    // Level 1: 6 rows of 2^17 inputs + 2^14 (k_log 18); levels 2-3: 6 x 8,192 + 4,224 (16).
    assert_eq!(cost.bridge.k_log, [18, 16]);
    assert_eq!((cost.bridge.nu, cost.bridge.m), (3, 22));
    assert_eq!(
        cost.bridge.committed_words,
        6 * ((1 << 17) + (1 << 14)) / 128 + 448
    );
    assert_eq!(cost.sponge_committed_words, 100 * 2_156);
    // A full segment: sponge M = 32; bridge union M = 28 (k_log 20 and 20, nu 7, 96 + 1 live
    // rows), dense stack 2^19 words, walker 1.9 x 10^7 entries (docs/keccak.md Section 10).
    let full = TiedPart::new(&gate, 1 << 13, 13, Profile::Fast);
    let cost = full.cost();
    assert_eq!(
        (cost.sponge_nu, cost.sponge_m, cost.sponge_dense_m),
        (13, 32, 32)
    );
    assert_eq!(
        (
            cost.bridge.nu,
            cost.bridge.k_log,
            cost.bridge.m,
            cost.bridge.dense_m
        ),
        (7, [20, 20], 28, 26)
    );
    assert_eq!(cost.bridge.rows, [96, 1]);
    assert_eq!(cost.bridge.walker_entries, 19_234_816);
    assert_eq!(cost.slot_walker_entries, 716_537_856);
    // A part proof must carry its part's count and d.
    let other = TiedPart::new_part(&gate, 100, 7, 1, Profile::Fast);
    assert_eq!(other.segment().offset, 1 << 17);
    assert!(matches!(
        part.verify(
            &[[0; 64]; 100],
            &[[0; 32]; 100],
            &TiedCommitment {
                sponge: KeccakCommitment::from_bytes(Vec::new()),
            },
            [0; 32],
            &derive_matrices_for(&Shape::PRODUCTION, &[0; 32]),
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
/// keeps only its commitment and commit data, proves its share, and the shares XOR to the
/// F_p side's `Y_sigma` of the whole segment. The buffers are rebuilt for the proof
/// (`reattach`). A share proven under another part's index, swapped shares, and a part
/// proven against the whole segment's `Y` are rejected.
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
    assert_eq!(segs[2].len(), 4);
    assert_eq!(segs[1].segment().offset, 1 << 13);
    // Event 2: commit every part, keep only the commitments and the commit data.
    let mut kept = Vec::new();
    let commitments: Vec<TiedCommitment> = segs
        .iter()
        .zip(&parts)
        .map(|(seg, (_, r))| {
            let mut committed = seg.commit(&records[r.clone()]);
            kept.push(committed.take_commit_data());
            committed.commitment().clone()
        })
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
    let fs = |k: usize| -> [u8; 32] { std::array::from_fn(|i| (i + 7 * k) as u8) };
    let mut shares = Vec::new();
    let mut acc = vec![0_u8; 4096];
    for k in 0..parts.len() {
        let r = parts[k].1.clone();
        let (sponge, bits) = segs[k].honest_parts(&records[r.clone()]);
        let committed = segs[k].reattach(
            sponge,
            bits,
            &records[r],
            commitments[k].clone(),
            kept[k].take(),
        );
        let own = segs[k].final_vector(&committed, &mats);
        let share = if k + 1 < parts.len() {
            own.clone()
        } else {
            acc.iter().zip(&y_segment).map(|(a, y)| a ^ y).collect()
        };
        assert_eq!(share, own, "honest shares XOR to the segment's Y");
        acc.iter_mut().zip(&share).for_each(|(a, s)| *a ^= s);
        let proof = segs[k].prove_with_final_vector(committed, sd, &mats, share.clone(), &fs(k));
        shares.push((share, proof));
    }
    assert_eq!(acc, y_segment);
    let verify = |seg: &TiedPart<'_>, k: usize, y: &[u8], proof| {
        let (mu, c) = statement(&records[parts[k].1.clone()]);
        seg.verify(&mu, &c, &commitments[k], sd, &mats, y, &fs(k), proof)
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

/// Measurement (ignored; `--ignored --nocapture`, CLMUL build): one tied part per size in
/// `AGG_TIED_SIZES` (default `256,512,1024,2048`; an entry `n@d` is a part of `2^d` record
/// positions with `n` live records, e.g. `8@13` for a full-segment bridge union with a
/// small sponge instance) and per Ligerito profile in
/// `AGG_TIED_PROFILES` (default `fast,slim,secure`): event-2 commit, prove (the K14 path:
/// the sponge commitment's prover data kept), verify, median of `AGG_TIED_REPS` (default 3)
/// runs after one warmup; the proof bytes split into the bridge cap, the sponge proof and the
/// bridge union's proof; the kept commit data's heap bytes. Matrices are derived outside
/// the timed region.
#[test]
#[ignore = "measurement; run with --release --ignored --nocapture"]
fn tied_part_profiles_bytes_and_times() {
    let list = |var: &str, default: &str| -> Vec<String> {
        std::env::var(var)
            .unwrap_or_else(|_| default.to_string())
            .split(',')
            .map(|s| s.trim().to_string())
            .collect()
    };
    let sizes: Vec<(usize, u32)> = list("AGG_TIED_SIZES", "256,512,1024,2048")
        .iter()
        .map(|s| match s.split_once('@') {
            Some((n, d)) => (n.parse().unwrap(), d.parse().unwrap()),
            None => {
                let n: usize = s.parse().unwrap();
                (n, TiedPart::d_seg_for(n))
            }
        })
        .collect();
    let reps: usize = std::env::var("AGG_TIED_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let gate = sponge::sponge_gate();
    let median = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    eprintln!(
        "| Records (d) | Profile | Commit (ms) | Prove (ms) | Verify (ms) | Proof (B) | of which \
         bridge cap / sponge / bridge union | Commitment (B) | Kept commit data (MiB) |"
    );
    for &(n, d) in &sizes {
        let mut rng = Rng::new(n as u64);
        let records: Vec<SpongeRecord> = (0..n).map(|_| random_record(&mut rng)).collect();
        let (mu, c_tilde) = statement(&records);
        for name in list("AGG_TIED_PROFILES", "fast,slim,secure") {
            let profile = match name.as_str() {
                "fast" => Profile::Fast,
                "slim" => Profile::Slim,
                "secure" => Profile::Secure,
                other => panic!("unknown profile {other}"),
            };
            let part = TiedPart::new(&gate, n, d, profile);
            let (mut t_commit, mut t_prove, mut t_verify) = (vec![], vec![], vec![]);
            let mut last = None;
            for rep in 0..=reps {
                let t0 = Instant::now();
                let committed = part.commit(&records);
                let commit = t0.elapsed().as_secs_f64() * 1e3;
                let commitment = committed.commitment().clone();
                let sd = seed_after(&commitment);
                let mats = Arc::new(derive_matrices_for(&Shape::PRODUCTION, &sd));
                let fs = fs_after(&commitment, &sd);
                let t1 = Instant::now();
                let out = part.prove(committed, sd, &mats, &fs);
                let prove = t1.elapsed().as_secs_f64() * 1e3;
                let t2 = Instant::now();
                part.verify(
                    &mu,
                    &c_tilde,
                    &commitment,
                    sd,
                    &mats,
                    &out.final_vector,
                    &fs,
                    &out.proof,
                )
                .expect("honest part");
                let verify = t2.elapsed().as_secs_f64() * 1e3;
                if rep > 0 {
                    t_commit.push(commit);
                    t_prove.push(prove);
                    t_verify.push(verify);
                }
                last = Some((out, commitment));
            }
            let (out, commitment) = last.unwrap();
            let kept = part
                .commit(&records)
                .take_commit_data()
                .map_or(0, |d| d.heap_bytes());
            let b = &out.proof.bytes;
            let framed = |at: usize| u32::from_le_bytes(b[at..at + 4].try_into().unwrap()) as usize;
            let cap = framed(13);
            let sponge_len = framed(17 + cap);
            let bridge_len = framed(21 + cap + sponge_len);
            eprintln!(
                "| {n} ({d}) | {name} | {:.1} | {:.1} | {:.1} | {} | {} / {} / {} | {} | {:.1} |",
                median(t_commit),
                median(t_prove),
                median(t_verify),
                b.len(),
                cap,
                sponge_len,
                bridge_len,
                commitment.to_bytes().len(),
                kept as f64 / f64::from(1 << 20)
            );
        }
    }
}
