//! The tied part (docs/parity-chain.md): the sponge
//! instance with the virtual parity chain (one sum-check from the public final vectors to a
//! bit-level claim on the sponge commitment). Round trips at 5 records (padded positions, `d
//! = 3`), 64 (`d = 6`), 256 (`d = 8`), 512 (`d = 9`, one full level-1 block) and 1,024 (`d =
//! 10`, two level-1 blocks per lane), spans, the tampering cases of docs/parity-chain.md
//! with their error names, the precomputed commitment, and a segment split into parts whose
//! shares XOR to the segment's `Y`. The bridge seed is
//! derived from the event-2 commitments, so the sponge witnesses and the part's bits are
//! fixed before the seed exists (events 2 and 3). Run with `--release`.

#![allow(clippy::needless_range_loop)]

use std::sync::Arc;
use std::time::Instant;

use agg_bridge::{seed::derive_matrices_for, shape::Shape};
use agg_keccak::prover::{KeccakCommitment, KeccakError, KeccakProof, Profile};
use agg_keccak::{
    sponge::{self, SpongeRecord, W1_INPUT_BASE},
    tied::{pack_y, TiedCommitment, TiedPart, TiedProofParts, ROUND_BYTES},
    witness::pack_slot,
};
use agg_witness::bridge::{gf2_composite, BridgeMatrices};
use agg_witness::padding::Geometry;
use flock_core::test_rng::Rng;
use flock_field::F128;
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

/// Which final vectors a tampered run proves.
#[derive(Clone, Copy)]
enum Share {
    /// The final vectors of the proven (possibly tampered) bits.
    OfProven,
    /// The final vectors of the honest records.
    Stale,
}

/// One proof of a part and its verification inputs.
struct Run {
    commitment: TiedCommitment,
    sd: [u8; 32],
    mats: Arc<BridgeMatrices>,
    fs: [u8; 32],
    ys: Vec<Vec<u8>>,
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
        self.verify_with(
            r,
            &self.mu,
            &self.c_tilde,
            &r.commitment,
            r.sd,
            &r.ys,
            &r.fs,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn verify_with(
        &self,
        r: &Run,
        mu: &[[u8; 64]],
        c: &[[u8; 32]],
        com: &TiedCommitment,
        sd: [u8; 32],
        ys: &[Vec<u8>],
        fs: &[u8; 32],
    ) -> Result<(), KeccakError> {
        let mats = if sd == r.sd {
            r.mats.clone()
        } else {
            Arc::new(derive_matrices_for(&Shape::PRODUCTION, &sd))
        };
        self.part.verify(mu, c, com, sd, &mats, ys, fs, &r.proof)
    }

    /// Commit `sponge` and `bits` (event 2) with the statement of the honest records,
    /// derive `sd` and `fs` from the commitment, and prove against the chosen final vectors.
    fn run(&self, sponge: agg_keccak::witness::SlotBuffers, bits: Vec<u64>, share: Share) -> Run {
        let committed = self.part.commit_parts(sponge, bits, &self.records);
        let commitment = committed.commitment().clone();
        let sd = seed_after(&commitment);
        let mats = Arc::new(derive_matrices_for(&Shape::PRODUCTION, &sd));
        let fs = fs_after(&commitment, &sd);
        let ys = match share {
            Share::OfProven => self.part.final_vectors(&committed, &mats),
            Share::Stale => {
                let honest = self.part.commit(&self.records);
                self.part.final_vectors(&honest, &mats)
            }
        };
        let proof = self
            .part
            .prove_with_final_vectors(committed, sd, &mats, ys.clone(), &fs)
            .expect("prove");
        Run {
            commitment,
            sd,
            mats,
            fs,
            ys,
            proof,
        }
    }

    /// `r` with its proof's parts changed by `f`.
    fn tampered(&self, r: &Run, f: &dyn Fn(&mut TiedProofParts)) -> Run {
        let mut parts = self.part.decode_parts(&r.proof).expect("own proof parses");
        f(&mut parts);
        Run {
            proof: self.part.encode_parts(&parts),
            ..run_clone(r)
        }
    }
}

/// How much tampering a round trip checks.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tamper {
    /// None (round trip and the cheap rejections only).
    None,
    /// The case that depends on the block geometry: a bit of the chain's copy flipped in the
    /// last level-1 block.
    Blocks,
    /// Every case, plus the precomputed commitment.
    Full,
}

fn expect(ctx: &Ctx<'_>, r: &Run, label: &str, ok: &dyn Fn(&KeccakError) -> bool) {
    match ctx.verify(r) {
        Err(e) if ok(&e) => eprintln!("  {label}: rejected ({e})"),
        other => panic!("{label}: {other:?}"),
    }
}

#[allow(clippy::too_many_lines)]
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
    let out = part.prove(committed, sd, &mats, &fs).expect("prove");
    let t_prove = t2.elapsed();
    let run = Run {
        commitment: commitment.clone(),
        sd,
        mats: mats.clone(),
        fs,
        ys: out.final_vectors.clone(),
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
        out.final_vectors,
        [gf2_composite(&geom, &b, &mats)],
        "Y of both sides"
    );
    assert_eq!(pack_y(&out.final_vectors[0]).len(), 512);

    let cost = part.cost();
    let parts = part.decode_parts(&out.proof).expect("parse");
    let chain_bytes = 4 + ROUND_BYTES * cost.chain_rounds + 16;
    let [zc, lc, open, values] = parts.sponge.size_breakdown();
    assert_eq!(parts.rounds.len(), cost.chain_rounds);
    assert_eq!(out.proof.bytes.len(), 18 + chain_bytes + 4 + zc + lc + open + values + 4);
    eprintln!(
        "tied part n = {n} (d {d}): sponge nu {} M {} dense m {} ({} committed words); chain \
         {} rounds over {} bits; commit {:.2?}, matrices {:.2?}, prove {:.2?}, verify {:.2?}; \
         proof {} B = 18 header + {chain_bytes} chain + sponge proof (zerocheck {zc}, \
         lincheck {lc}, opening {open} of which ring switches {} B, values {values}) + 8 \
         framing; commitment {} B",
        cost.sponge_nu,
        cost.sponge_m,
        cost.sponge_dense_m,
        cost.sponge_committed_words,
        cost.chain_rounds,
        cost.chain_input_bits,
        t_commit,
        t_mats,
        t_prove,
        t_verify,
        out.proof.bytes.len(),
        parts.sponge.flock.pcs_open.ring_switches.len() * (16 * 128 + 8),
        commitment.sponge.as_bytes().len(),
    );
    assert_eq!(parts.sponge.flock.pcs_open.ring_switches.len(), 3);

    // A flipped Y bit (the final vectors' digest enters the tie challenger before the tie
    // nonce, so that nonce fails first); the same flipped Y proven by the honest prover (the
    // chain fails); another seed (other matrices); a wrong c~; a wrong mu; another fs.
    let (mu, c) = (&ctx.mu, &ctx.c_tilde);
    let mut bad = out.final_vectors.clone();
    bad[0][4095] ^= 1;
    assert!(matches!(
        ctx.verify_with(&run, mu, c, &commitment, sd, &bad, &fs),
        Err(KeccakError::Tie("tie nonce" | "parity chain"))
    ));
    let wrong = part
        .prove_with_final_vectors(part.commit(&ctx.records), sd, &mats, bad.clone(), &fs)
        .expect("prove");
    let r = Run {
        ys: bad.clone(),
        proof: wrong,
        ..run_clone(&run)
    };
    expect(&ctx, &r, "flipped Y bit", &|e| {
        matches!(e, KeccakError::Tie("parity chain"))
    });
    assert!(matches!(
        ctx.verify_with(&run, mu, c, &commitment, [0xA5; 32], &run.ys, &fs),
        Err(KeccakError::Tie("parity chain"))
    ));
    // A wrong c~ or mu changes the public digest the sponge challenger absorbs, so Flock's
    // transcript already rejects (the public claim would, too).
    let mut bad_c = c.clone();
    bad_c[n - 1][0] ^= 1;
    assert!(ctx
        .verify_with(&run, mu, &bad_c, &commitment, sd, &run.ys, &fs)
        .is_err());
    let mut bad_mu = mu.clone();
    bad_mu[0][63] ^= 0x80;
    assert!(ctx
        .verify_with(&run, &bad_mu, c, &commitment, sd, &run.ys, &fs)
        .is_err());
    assert!(ctx
        .verify_with(&run, mu, c, &commitment, sd, &run.ys, &[0x3C; 32])
        .is_err());
    // Flipped proof bytes: the tie nonce, the eq_Y nonce, a round message, the bit value,
    // the sponge proof.
    let len = run.proof.bytes.len();
    for at in [14, 17, 18, 22 + 3 * ROUND_BYTES + 5, 18 + chain_bytes - 3, len / 2, len - 20] {
        let mut flipped = run.proof.clone();
        flipped.bytes[at] ^= 1;
        let r = Run {
            proof: flipped,
            ..run_clone(&run)
        };
        assert!(ctx.verify(&r).is_err(), "flipped proof byte {at}");
    }
    // Another sponge cap in place of the event-2 one.
    let mut other_cap = commitment.sponge.as_bytes().to_vec();
    other_cap[0] ^= 1;
    let swapped = TiedCommitment {
        sponge: KeccakCommitment::from_bytes(other_cap),
    };
    assert!(ctx
        .verify_with(&run, mu, c, &swapped, sd, &run.ys, &fs)
        .is_err());

    if tamper == Tamper::None {
        return;
    }
    let full = tamper == Tamper::Full;
    if full {
        // The precomputed commitment gives the same proof as committing again.
        let mut committed = part.commit(&ctx.records);
        let data = committed.take_commit_data();
        assert!(data.is_some());
        let again = part.prove(committed, sd, &mats, &fs).expect("prove");
        assert_eq!(again, out, "proof without the kept commitment");
        let (sponge_buf, bits) = part.honest_parts(&ctx.records);
        let reattached = part.reattach(sponge_buf, bits, &ctx.records, commitment.clone(), data);
        assert_eq!(part.prove(reattached, sd, &mats, &fs).expect("prove"), out);
    }

    // A bit flipped in the chain's copy of b' only (lane 3 of the last record, in the last
    // level-1 block), the final vectors of the copy (so the chain's claim and messages are
    // consistent): its final value is not the sponge's.
    let (sponge_buf, mut bits) = part.honest_parts(&ctx.records);
    let col = part.segment().column(3, 1024 * (n - 1) + 77);
    bits[col / 64] ^= 1 << (col % 64);
    let r = ctx.run(sponge_buf, bits, Share::OfProven);
    expect(&ctx, &r, "chain copy of b' only", &|e| {
        matches!(e, KeccakError::Tie("parity chain"))
    });
    if !full {
        return;
    }
    // A b' bit flipped in the committed record (sponge and chain copy recomputed, c~ of
    // the tampered record), the stale final vectors of the honest records: the chain's claim
    // is not the committed bits' inner product.
    let mut tampered = ctx.records.clone();
    tampered[0].w1_encode[100] ^= 0x10;
    let (sponge_buf, bits) = part.honest_parts(&tampered);
    let tctx = Ctx {
        part: TiedPart::new(&gate, n, d, Profile::Fast),
        records: tampered.clone(),
        mu: ctx.mu.clone(),
        c_tilde: statement(&tampered).1,
    };
    let committed = tctx.part.commit_parts(sponge_buf, bits, &tampered);
    let com_t = committed.commitment().clone();
    let sd_t = seed_after(&com_t);
    let mats_t = Arc::new(derive_matrices_for(&Shape::PRODUCTION, &sd_t));
    let fs_t = fs_after(&com_t, &sd_t);
    let honest_ys = part.final_vectors(&part.commit(&ctx.records), &mats_t);
    let proof = tctx
        .part
        .prove_with_final_vectors(committed, sd_t, &mats_t, honest_ys.clone(), &fs_t)
        .expect("prove");
    let r = Run {
        commitment: com_t,
        sd: sd_t,
        mats: mats_t,
        fs: fs_t,
        ys: honest_ys,
        proof,
    };
    expect(&tctx, &r, "flipped b' bit, stale Y", &|e| {
        matches!(e, KeccakError::Tie("parity chain"))
    });
    // Tampered proof parts: a round message (its own nonce then fails with probability
    // 1 - 2^-g, else the final check), the last round's leading coefficient, the claimed bit
    // value, and the bit claim's s_hat_v in the sponge opening (the third ring switch).
    let r = ctx.tampered(&run, &|p| p.rounds[5].q1 += F128::ONE);
    expect(&ctx, &r, "round message", &|e| {
        matches!(
            e,
            KeccakError::Tie("parity chain" | "chain round nonce")
        )
    });
    let r = ctx.tampered(&run, &|p| {
        let last = p.rounds.len() - 1;
        p.rounds[last].qinf += F128::ONE;
    });
    expect(&ctx, &r, "last round message", &|e| {
        matches!(
            e,
            KeccakError::Tie("parity chain" | "chain round nonce")
        )
    });
    let r = ctx.tampered(&run, &|p| p.value += F128::ONE);
    expect(&ctx, &r, "bit value", &|e| {
        matches!(e, KeccakError::Tie("parity chain"))
    });
    let r = ctx.tampered(&run, &|p| {
        p.sponge.flock.pcs_open.ring_switches[2].s_hat_v[17] += F128::ONE;
    });
    expect(&ctx, &r, "bit claim's s_hat_v", &|e| {
        matches!(e, KeccakError::Flock(_))
    });
    // A bit flipped in both copies (the sponge recomputed on the tampered record), Y
    // consistent with it, the published c~ stale: the sponge instance's public claim fails.
    let (sponge_buf, bits) = part.honest_parts(&tampered);
    let r = ctx.run(sponge_buf, bits, Share::OfProven);
    expect(&ctx, &r, "both copies, stale c~", &|e| {
        matches!(e, KeccakError::Tie("public words"))
    });
    // The honest chain copy against stale Y (sponge and copy tampered, Y of the honest
    // records) as above but with the honest statement: the chain fails first.
    let (sponge_buf, bits) = part.honest_parts(&tampered);
    let r = ctx.run(sponge_buf, bits, Share::Stale);
    expect(&ctx, &r, "both copies, stale Y and c~", &|e| {
        matches!(e, KeccakError::Tie("parity chain"))
    });

    // A sponge input bit flipped with the stale chi witness (the chain's copy honest, Y of
    // the honest bits): the sponge rows fail (or the chain, whose bit value now differs).
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
    let r = ctx.run(sponge_buf, bits, Share::OfProven);
    expect(&ctx, &r, "sponge input, stale chi", &|e| {
        matches!(e, KeccakError::Flock(_) | KeccakError::Tie("parity chain"))
    });
}

fn run_clone(r: &Run) -> Run {
    Run {
        commitment: r.commitment.clone(),
        sd: r.sd,
        mats: r.mats.clone(),
        fs: r.fs,
        ys: r.ys.clone(),
        proof: r.proof.clone(),
    }
}

/// 64 records (`d = 6`): round trip, `Y` equal to the F_p side's GF(2) walk, the precomputed
/// commitment, and the tampering cases with their error names.
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

/// 512 records (`d = 9`: exactly one level-1 block per lane).
#[test]
#[cfg_attr(debug_assertions, ignore = "heavy Flock proof; run with --release")]
fn tied_part_round_trip_512() {
    tied_round_trip(512, 0x512, Tamper::None);
}

/// 2,048 records (`d = 11`): a full-height sponge instance (`n = 2^nu_s`), so Flock's opening
/// takes the block-first transport with the chain's column bit-bank (the path of every
/// power-of-two batch from 2,048 records on, 32K included).
#[test]
#[cfg_attr(debug_assertions, ignore = "heavy Flock proof; run with --release")]
fn tied_part_round_trip_2048_full_height() {
    tied_round_trip(2048, 0x2048, Tamper::None);
}

/// 1,024 records (`d = 10`): two level-1 blocks per lane, the first part whose chain has a
/// block coordinate; with the tampering case that depends on the blocks (the flipped copy
/// sits in block 1).
#[test]
#[cfg_attr(debug_assertions, ignore = "heavy Flock proof; run with --release")]
fn tied_part_round_trip_1024_two_blocks() {
    tied_round_trip(1024, 0x1024, Tamper::Blocks);
}

/// A span of `segments` segments of `2^d` record positions filled in order by `n` records,
/// proven as one tied part: each `Y_sigma` equals the F_p side's
/// walk of its segment's records (zero for an empty segment); swapped or flipped final
/// vectors, a wrong final vector of segment 1 proven by the honest prover, a chain copy
/// flipped in segment 1 only, another segment count and another record count are rejected.
fn tied_span_round_trip(n: usize, d: u32, segments: usize, rng_seed: u64) {
    let gate = sponge::sponge_gate();
    let part = TiedPart::new_span(&gate, n, d, 0, segments, Profile::Fast);
    let mut rng = Rng::new(rng_seed);
    let records: Vec<SpongeRecord> = (0..n).map(|_| random_record(&mut rng)).collect();
    let (mu, c_tilde) = statement(&records);
    let committed = part.commit(&records);
    let commitment = committed.commitment().clone();
    let sd = seed_after(&commitment);
    let mats = Arc::new(derive_matrices_for(&Shape::PRODUCTION, &sd));
    let fs = fs_after(&commitment, &sd);
    let t0 = Instant::now();
    let out = part.prove(committed, sd, &mats, &fs).expect("prove");
    let t_prove = t0.elapsed();
    let t1 = Instant::now();
    let verify = |ys: &[Vec<u8>], proof: &KeccakProof| {
        part.verify(&mu, &c_tilde, &commitment, sd, &mats, ys, &fs, proof)
    };
    verify(&out.final_vectors, &out.proof).expect("honest span");
    eprintln!(
        "span of {segments} segments (d {d}, n {n}): prove {t_prove:.2?}, verify {:.2?}, \
         proof {} B, sponge nu {}, chain rounds {}",
        t1.elapsed(),
        out.proof.bytes.len(),
        part.nu(),
        part.cost().chain_rounds
    );
    // Y of every segment against the F_p side's main-domain walk of its records.
    let geom = Geometry::new(1 << d).unwrap();
    assert_eq!(geom.d_seg(), d);
    assert_eq!(out.final_vectors.len(), segments);
    for (sigma, y) in out.final_vectors.iter().enumerate() {
        let r = (sigma << d).min(n)..((sigma + 1) << d).min(n);
        assert_eq!(part.live(sigma), r.len());
        let lanes = main_domain_lanes(&records[r.clone()], geom.main_len());
        let b: [&[i8]; 6] = std::array::from_fn(|t| lanes[t].as_slice());
        assert_eq!(*y, gf2_composite(&geom, &b, &mats), "Y of segment {sigma}");
        if r.is_empty() {
            assert!(y.iter().all(|&x| x == 0));
        }
    }
    // Final vectors: a flipped bit of segment 1 and the vectors of segments 0, 1 swapped.
    let mut bad = out.final_vectors.clone();
    bad[1][100] ^= 1;
    assert!(verify(&bad, &out.proof).is_err(), "flipped Y_1 bit");
    let mut swapped = out.final_vectors.clone();
    swapped.swap(0, 1);
    assert_ne!(swapped, out.final_vectors);
    assert!(verify(&swapped, &out.proof).is_err(), "swapped Y_0, Y_1");
    assert!(matches!(
        verify(&out.final_vectors[..segments - 1], &out.proof),
        Err(KeccakError::Statement("final vector"))
    ));
    // A wrong final vector of segment 1 proven by the honest prover: the chain fails.
    let committed = part.commit(&records);
    let mut wrong = out.final_vectors.clone();
    wrong[1][2049] ^= 1;
    let proof = part
        .prove_with_final_vectors(committed, sd, &mats, wrong.clone(), &fs)
        .expect("prove");
    assert!(matches!(
        verify(&wrong, &proof),
        Err(KeccakError::Tie("parity chain"))
    ));
    // The chain's copy flipped in segment 1 only (lane 1, its first record), the final
    // vectors of the tampered copy: the chain's final value is not the sponge's.
    let (sponge_buf, mut bits) = part.honest_parts(&records);
    let words = part.segment().input_bits() / 64;
    let col = 64 * words + part.segment().column(1, 33);
    bits[col / 64] ^= 1 << (col % 64);
    let committed = part.commit_parts(sponge_buf, bits, &records);
    let commitment2 = committed.commitment().clone();
    assert_eq!(commitment2, commitment, "the sponge is honest");
    let ys = part.final_vectors(&committed, &mats);
    assert_ne!(ys[1], out.final_vectors[1]);
    let proof = part
        .prove_with_final_vectors(committed, sd, &mats, ys.clone(), &fs)
        .expect("prove");
    assert!(matches!(
        verify(&ys, &proof),
        Err(KeccakError::Tie("parity chain"))
    ));
    // The proof checked as a span of another segment count or record count.
    let fewer = TiedPart::new_span(&gate, n, d, 0, segments + 1, Profile::Fast);
    let mut more_ys = out.final_vectors.clone();
    more_ys.push(vec![0; 4096]);
    assert!(fewer
        .verify(
            &mu,
            &c_tilde,
            &commitment,
            sd,
            &mats,
            &more_ys,
            &fs,
            &out.proof
        )
        .is_err());
    let short = TiedPart::new_span(&gate, n - 1, d, 0, segments, Profile::Fast);
    assert!(matches!(
        short.verify(
            &mu,
            &c_tilde,
            &commitment,
            sd,
            &mats,
            &out.final_vectors,
            &fs,
            &out.proof
        ),
        Err(KeccakError::Statement("record count"))
    ));
}

/// Spans: three segments of 8 positions (8, 8, 3 records), three with the last empty (8, 8,
/// 0; a fourth segment of the chain is always empty), two segments of 64 (64, 6).
#[test]
#[cfg_attr(debug_assertions, ignore = "Flock proofs; run with --release")]
fn tied_spans_round_trip() {
    tied_span_round_trip(19, 3, 3, 0x5A19);
    tied_span_round_trip(16, 3, 3, 0x5A16);
    tied_span_round_trip(70, 6, 2, 0x5A70);
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
    // The chain: 17 in-block coordinates (2^17 positions), 3 lane coordinates; its input is
    // eight lanes of 2^17 bits.
    assert_eq!((cost.chain_rounds, cost.chain_input_bits), (20, 8 << 17));
    assert_eq!(cost.sponge_committed_words, 100 * 2_156);
    // A full segment: sponge M = 32; the chain has 19 + 4 + 3 = 26 rounds.
    let full = TiedPart::new(&gate, 1 << 13, 13, Profile::Fast);
    let cost = full.cost();
    assert_eq!(
        (cost.sponge_nu, cost.sponge_m, cost.sponge_dense_m),
        (13, 32, 32)
    );
    assert_eq!(cost.chain_rounds, 26);
    // A span of four full segments (a 32K batch): 28 rounds over 2^28 input bits.
    let span = TiedPart::new_span(&gate, 1 << 15, 13, 0, 4, Profile::Slim);
    assert_eq!(
        (span.cost().chain_rounds, span.cost().chain_input_bits),
        (28, 1 << 28)
    );
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
            &[vec![0; 4096]],
            &[0; 32],
            &KeccakProof {
                bytes: b"AGGKCTxx".to_vec()
            }
        ),
        Err(KeccakError::Malformed(_))
    ));
}

/// A segment split into parts: 20 live records (`d_seg = 5`) in parts of 8
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
        let own = segs[k].final_vectors(&committed, &mats).remove(0);
        let share = if k + 1 < parts.len() {
            own.clone()
        } else {
            acc.iter().zip(&y_segment).map(|(a, y)| a ^ y).collect()
        };
        assert_eq!(share, own, "honest shares XOR to the segment's Y");
        acc.iter_mut().zip(&share).for_each(|(a, s)| *a ^= s);
        let proof = segs[k]
            .prove_with_final_vectors(committed, sd, &mats, vec![share.clone()], &fs(k))
            .expect("prove");
        shares.push((share, proof));
    }
    assert_eq!(acc, y_segment);
    let verify = |seg: &TiedPart<'_>, k: usize, y: &[u8], proof| {
        let (mu, c) = statement(&records[parts[k].1.clone()]);
        seg.verify(
            &mu,
            &c,
            &commitments[k],
            sd,
            &mats,
            &[y.to_vec()],
            &fs(k),
            proof,
        )
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
/// positions with `n` live records, e.g. `8@13` for a full segment with a
/// small sponge instance) and per Ligerito profile in
/// `AGG_TIED_PROFILES` (default `fast,slim,secure`): event-2 commit, prove (with
/// the sponge commitment's prover data kept), verify, median of `AGG_TIED_REPS` (default 3)
/// runs after one warmup; the proof bytes split into the header and chain and
/// the sponge proof; the kept commit data's heap bytes. Matrices are derived outside the
/// timed region.
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
    // `n`, `n@d` or `n@d/S` (a span of S segments of 2^d positions filled in order by n
    // records; `8@13/4` is a 32K batch's span with a small sponge instance).
    let sizes: Vec<(usize, u32, usize)> = list("AGG_TIED_SIZES", "256,512,1024,2048")
        .iter()
        .map(|s| {
            let (s, segments) = s
                .split_once('/')
                .map_or((s.as_str(), 1), |(a, b)| (a, b.parse().unwrap()));
            match s.split_once('@') {
                Some((n, d)) => (n.parse().unwrap(), d.parse().unwrap(), segments),
                None => {
                    let n: usize = s.parse().unwrap();
                    (n, TiedPart::d_seg_for(n), segments)
                }
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
         header and chain / sponge proof (with framing) | Commitment (B) | Kept commit data \
         (MiB) |"
    );
    for &(n, d, segments) in &sizes {
        let mut rng = Rng::new(n as u64);
        let records: Vec<SpongeRecord> = (0..n).map(|_| random_record(&mut rng)).collect();
        let (mu, c_tilde) = statement(&records);
        for name in list("AGG_TIED_PROFILES", "fast,slim,secure") {
            let profile = match name.as_str() {
                "fast" => Profile::Fast,
                "slim" => Profile::Slim,
                "slim8" => Profile::Slim8,
                "secure" => Profile::Secure,
                other => panic!("unknown profile {other}"),
            };
            let part = TiedPart::new_span(&gate, n, d, 0, segments, profile);
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
                let out = part.prove(committed, sd, &mats, &fs).expect("prove");
                let prove = t1.elapsed().as_secs_f64() * 1e3;
                let t2 = Instant::now();
                part.verify(
                    &mu,
                    &c_tilde,
                    &commitment,
                    sd,
                    &mats,
                    &out.final_vectors,
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
            let chain = 18 + 4 + ROUND_BYTES * part.cost().chain_rounds + 16;
            eprintln!(
                "| {n} ({d}{}) | {name} | {:.1} | {:.1} | {:.1} | {} | {} / {} | {} | {:.1} |",
                if segments > 1 {
                    format!(", {segments} segments")
                } else {
                    String::new()
                },
                median(t_commit),
                median(t_prove),
                median(t_verify),
                b.len(),
                chain,
                b.len() - chain,
                commitment.to_bytes().len(),
                kept as f64 / f64::from(1 << 20)
            );
            if std::env::var_os("AGG_TIED_BREAKDOWN").is_some() {
                let up = part.decode_parts(&out.proof).expect("parse").sponge;
                let [zc, lc, open, values] = up.size_breakdown();
                let o = &up.flock.pcs_open;
                eprintln!(
                    "  sponge: zerocheck {zc} B, lincheck {lc} B, opening {open} B ({} ring \
                     switches, {} multipoint dual values), values {values} B; Ligerito inner \
                     {} B:",
                    o.ring_switches.len(),
                    o.frobenius.values.iter().map(Vec::len).sum::<usize>(),
                    o.inner.ligerito.size_bytes()
                );
                o.inner.ligerito.print_size_breakdown();
            }
        }
    }
}

/// Measurement (ignored; `--ignored --nocapture`): the prover cost of Ligerito's query
/// proof of work at `AGG_GRIND_BITS` (default `16,20,22,24`) bits with Flock's challenger
/// (the fused BLAKE3 PoW and squeeze its Ligerito levels use, parallel on the rayon pool),
/// median of `AGG_TIED_REPS` (default 5) grinds from distinct transcript states, for the
/// grinding-versus-queries trade of docs/keccak.md.
#[test]
#[ignore = "measurement; run with --release --ignored --nocapture"]
fn ligerito_query_grinding_cost() {
    use flock_core::challenger::{Challenger, FsChallenger};
    let bits: Vec<u32> = std::env::var("AGG_GRIND_BITS")
        .unwrap_or_else(|_| "16,20,22,24".to_string())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let reps: usize = std::env::var("AGG_TIED_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    for b in bits {
        let mut times: Vec<f64> = (0..reps)
            .map(|rep| {
                let mut ch = FsChallenger::new(b"agg-keccak/test/grind");
                ch.observe_bytes(&[b as u8, rep as u8]);
                let t = Instant::now();
                let (nonce, words) = ch.grind_pow_and_sample_f128_vec(b, 119);
                let ms = t.elapsed().as_secs_f64() * 1e3;
                assert_eq!(words.len(), 119);
                std::hint::black_box(nonce);
                ms
            })
            .collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!(
            "query PoW {b} bits: median {:.2} ms (min {:.2}, max {:.2}) over {reps} grinds",
            times[reps / 2],
            times[0],
            times[reps - 1]
        );
    }
}
