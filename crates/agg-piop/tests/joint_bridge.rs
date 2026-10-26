//! The joint parity-bridge test (docs/bridge.md Section 9 items 3 and 7, mode A): for 64 and
//! 256 signatures the witness is derived from real ML-DSA-44 records, the F_p side
//! (`agg-piop` with the bridge tables) and the Flock side (`agg-keccak`, sponge gates plus
//! the bridge slot) run in one transcript, so both use the same bridge seed `sd` (event 3,
//! drawn after the Akita round-1 commitment and the Flock commitment). Checked:
//!
//! - both final vectors (the F_p side's published `Y_sigma` and the Flock side's
//!   `M_3 M_2 M_1 b'` of its committed bridge copy) agree with each other and with a
//!   direct computation from the derived `w_1'` (sparse columns, index conventions of
//!   docs/bridge.md Section 2, written here independently of both crates);
//! - the bit orders agree: the derived `w1Encode` bytes, the committed `b_t` columns and
//!   the Flock records carry the same bits;
//! - the honest aggregate verifies with the Flock sub-proof checked inside the PIOP
//!   verifier (Flock profile `Fast`, seed-derived matrices);
//! - flipping one `w1Encode` bit on either side makes exactly that side fail: on the F_p
//!   side (a committed `b_0` bit flipped, the published `Y` the agreed one) stage 1
//!   rejects while the Flock sub-proof is accepted; on the Flock side (the record's bit
//!   flipped in both Flock copies, the published `Y` the F_p side's) the Flock sub-proof
//!   is rejected while the F_p side accepts, and the Flock rejection is the bridge rows
//!   alone (the same tampered witness proves its own `Y`).
//!
//! Run with `--release`.

use std::borrow::Cow;
use std::sync::{Arc, Mutex};

use agg_bridge::seed::derive_matrices;
use agg_keccak::prover::{KeccakCommitment, KeccakProof, Profile, SpongeProver};
use agg_keccak::segment::{BridgedCommitted, BridgedSegment};
use agg_keccak::sponge::SpongeRecord;
use agg_piop::binding::FlockProfile;
use agg_piop::manifest::{Column, CommitmentManifest, Relation};
use agg_piop::prover::{
    main_tables, prove, prove_with_tables, BridgeDerivation, BridgeTamper, KeccakSide, ProveError,
    ProveHooks,
    ProverInputs, ProverOutput, SeededMatrices,
};
use agg_piop::source_tables::{BridgeGroupTables, MainTables};
use agg_piop::stage::StageError;
use agg_piop::verifier::{verify, KeccakVerifier, SegmentContext, VerifyError};
use agg_piop::wire::{unpack_final_vector, AggregateProof};
use agg_witness::bridge::BridgeMatrices;
use agg_witness::derive::{derive, SigWitness};
use agg_witness::gen::{self, Record};
use agg_witness::instance::PublicInstance;
use agg_witness::padding::Geometry;
use agg_witness::params::C_TILDE_BYTES;

fn on_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("test thread");
}

/// The derivation of spec 9.1 (`SeededMatrices`) that also records the seed it was asked
/// for, so that a test hook can rebuild tables under the transcript's matrices.
#[derive(Default)]
struct RecordingSeed(Mutex<Option<([u8; 32], Arc<BridgeMatrices>)>>);

impl RecordingSeed {
    fn last(&self) -> ([u8; 32], Arc<BridgeMatrices>) {
        self.0.lock().unwrap().clone().expect("a seed was derived")
    }
}

impl BridgeDerivation for RecordingSeed {
    fn matrices(&self, seed: &[u8; 32]) -> Cow<'_, BridgeMatrices> {
        let matrices = derive_matrices(seed);
        *self.0.lock().unwrap() = Some((*seed, Arc::new(matrices.clone())));
        Cow::Owned(matrices)
    }

    fn seed_derived(&self) -> bool {
        true
    }
}

/// What the Flock prover saw and produced at event 6.
#[derive(Clone)]
struct FlockRun {
    /// `sd` handed to the Flock side.
    bridge_seed: [u8; 32],
    /// `M_3 M_2 M_1 b'` of the committed bridge copy (the Flock side's own final vector).
    own_y: Vec<u8>,
    /// `c~_i` of the committed sponge records (the Flock statement it is consistent with).
    c_tilde: Vec<[u8; C_TILDE_BYTES]>,
}

/// The Flock side of mode A for one segment: commits the bridged segment of `records`
/// at event 2 and proves it at event 6 against the F_p side's published `Y_sigma`.
struct FlockSide<'a, 'g> {
    seg: &'a BridgedSegment<'g>,
    committed: Mutex<Option<BridgedCommitted>>,
    run: Mutex<Option<FlockRun>>,
}

impl<'a, 'g> FlockSide<'a, 'g> {
    fn new(seg: &'a BridgedSegment<'g>, records: &[SpongeRecord]) -> Self {
        Self {
            seg,
            committed: Mutex::new(Some(seg.commit(records))),
            run: Mutex::new(None),
        }
    }

    fn run(&self) -> FlockRun {
        self.run
            .lock()
            .unwrap()
            .clone()
            .expect("the Flock side proved")
    }
}

impl KeccakSide for FlockSide<'_, '_> {
    fn commitments(&self, segments: usize) -> Vec<Vec<u8>> {
        assert_eq!(segments, 1, "one segment");
        let guard = self.committed.lock().unwrap();
        vec![guard
            .as_ref()
            .expect("committed")
            .commitment()
            .as_bytes()
            .to_vec()]
    }

    fn prove(
        &self,
        bridge_seed: &[u8; 32],
        matrices: &BridgeMatrices,
        seeds: &[[u8; 32]],
        final_vectors: &[Vec<u8>],
        _record: &[u8; 32],
    ) -> Result<Vec<Vec<u8>>, ProveError> {
        assert_eq!((seeds.len(), final_vectors.len()), (1, 1), "one segment");
        let committed = self
            .committed
            .lock()
            .unwrap()
            .take()
            .expect("committed once");
        let matrices = Arc::new(matrices.clone());
        let own_y = self.seg.final_vector(&committed, &matrices);
        let c_tilde = committed.c_tilde().to_vec();
        let proof = self.seg.prove_with_final_vector(
            committed,
            *bridge_seed,
            matrices,
            final_vectors[0].clone(),
            &seeds[0],
        );
        *self.run.lock().unwrap() = Some(FlockRun {
            bridge_seed: *bridge_seed,
            own_y,
            c_tilde,
        });
        Ok(vec![proof.bytes])
    }
}

/// The Flock verifier of one segment, recording every outcome; `accept_anyway` returns
/// true whatever Flock says, so that the F_p side can be observed on its own.
struct FlockVerifier<'a, 'g> {
    seg: &'a BridgedSegment<'g>,
    accept_anyway: bool,
    outcomes: Mutex<Vec<bool>>,
}

impl<'a, 'g> FlockVerifier<'a, 'g> {
    fn new(seg: &'a BridgedSegment<'g>, accept_anyway: bool) -> Self {
        Self {
            seg,
            accept_anyway,
            outcomes: Mutex::new(Vec::new()),
        }
    }

    fn outcomes(&self) -> Vec<bool> {
        self.outcomes.lock().unwrap().clone()
    }
}

impl KeccakVerifier for FlockVerifier<'_, '_> {
    fn profile(&self) -> FlockProfile {
        FlockProfile::Fast
    }

    fn verify(&self, segment: &SegmentContext<'_>) -> bool {
        let ok = self
            .seg
            .verify(
                segment.mu,
                segment.c_tilde,
                &KeccakCommitment::from_bytes(segment.commitment.to_vec()),
                *segment.bridge_seed,
                Arc::new(segment.matrices.clone()),
                segment.final_vector,
                segment.seed,
                &KeccakProof {
                    bytes: segment.proof.to_vec(),
                },
            )
            .is_ok();
        self.outcomes.lock().unwrap().push(ok);
        ok || self.accept_anyway
    }
}

/// Real ML-DSA-44 records (own and shared keys) and their strictly derived witnesses.
struct Fixture {
    records: Vec<Record>,
    instance: PublicInstance,
    witnesses: Vec<SigWitness>,
}

impl Fixture {
    fn new(n: usize, seed: u64) -> Self {
        let shared = n / 4;
        let mut records = gen::records(seed, n - shared);
        records.extend(gen::records_shared_keys(seed + 1, shared, 3));
        let instance = PublicInstance::from_records(&records).expect("instance");
        let witnesses = instance
            .records
            .iter()
            .zip(&records)
            .map(|(public, record)| derive(&public.key, &public.mu, &record.sig).expect("derive"))
            .collect();
        Self {
            records,
            instance,
            witnesses,
        }
    }

    fn inputs(&self) -> Vec<(&[u8], &[u8], [u8; C_TILDE_BYTES])> {
        self.records
            .iter()
            .map(|r| {
                let mut ct = [0_u8; C_TILDE_BYTES];
                ct.copy_from_slice(&r.sig[..C_TILDE_BYTES]);
                (r.pk.as_slice(), r.msg.as_slice(), ct)
            })
            .collect()
    }

    fn signatures(&self) -> Vec<&[u8]> {
        self.records.iter().map(|r| r.sig.as_slice()).collect()
    }

    /// The Flock side's records: `mu_i` and `w1Encode(w_1')` of the derived witness.
    fn sponge_records(&self) -> Vec<SpongeRecord> {
        self.witnesses
            .iter()
            .map(|w| SpongeRecord {
                mu: w.mu,
                w1_encode: w.w1_bytes,
            })
            .collect()
    }

    /// The mode-A verifier (Flock profile `Fast`, the matrices of spec 9.1).
    fn verify(&self, bytes: &[u8], keccak: &FlockVerifier<'_, '_>) -> Result<(), VerifyError> {
        let public: Vec<(&[u8], &[u8])> = self
            .records
            .iter()
            .map(|r| (r.pk.as_slice(), r.msg.as_slice()))
            .collect();
        verify(
            &public,
            bytes,
            &SeededMatrices,
            keccak,
            FlockProfile::Fast,
            Relation::Hu,
            None,
        )
        .map(|_| ())
    }
}

/// Main-domain cell of `b_t(j, rho, i)` (docs/bridge.md Section 2): `x = j + 256 rho + 1024 i`.
const fn cell(j: usize, rho: usize, i: usize) -> usize {
    j + 256 * rho + 1024 * i
}

/// Index of `b_t(j, rho)` in the `w1Encode` bit string (FIPS 204 Algorithms 16 and 28,
/// least significant bit of each byte first): `1536 rho + 6 j + t`.
const fn stream_bit(t: usize, j: usize, rho: usize) -> usize {
    1536 * rho + 6 * j + t
}

/// The three bit orders agree: the derived `w_1'`, its `w1Encode` bytes (the Flock
/// records) and the committed columns `b_t` (padded cells zero).
fn check_bit_orders(witnesses: &[SigWitness], lanes: &[&[i8]; 6]) {
    for (t, lane) in lanes.iter().enumerate() {
        for (x, &bit) in lane.iter().enumerate() {
            let (i, rho, j) = (x / 1024, (x % 1024) / 256, x % 256);
            let expected = witnesses.get(i).map_or(0, |w| {
                let q = stream_bit(t, j, rho);
                let encoded = (w.w1_bytes[q / 8] >> (q % 8)) & 1;
                assert_eq!(encoded, w.bit(t, j, rho), "w1Encode bit {q} of record {i}");
                encoded
            });
            assert_eq!(bit, expected as i8, "b_{t} at cell {x}");
        }
    }
}

/// `Y = M_3 M_2 M_1 b` over GF(2) for one segment, straight from the derived `w_1'` and
/// the sparse columns: level 1 per `2^19`-cell block (`zeta_1 = x mod 2^19`, output cell
/// `rho_1 + 2^13 (x >> 19)`), level 2 over the level-1 cells (`zeta_2 = rho_1 + 2^13
/// i_mid`), level 3 over `zeta_3 = t + 6 rho_2` (docs/bridge.md Section 2). Padded cells
/// are zero and contribute nothing.
fn direct_final_vector(witnesses: &[SigWitness], m: &BridgeMatrices) -> Vec<u8> {
    let blocks = (1024 * witnesses.len()).div_ceil(1 << 19);
    let mut y1 = vec![vec![0_u8; blocks << 13]; 6];
    for (i, w) in witnesses.iter().enumerate() {
        for rho in 0..4 {
            for j in 0..256 {
                let x = cell(j, rho, i);
                for (t, lane) in y1.iter_mut().enumerate() {
                    if w.bit(t, j, rho) == 1 {
                        for &r in m.m1.column(x % (1 << 19)) {
                            lane[((x >> 19) << 13) + r as usize] ^= 1;
                        }
                    }
                }
            }
        }
    }
    let mut y2 = vec![vec![0_u8; 1 << 13]; 6];
    for (lane1, lane2) in y1.iter().zip(&mut y2) {
        for (zeta2, _) in lane1.iter().enumerate().filter(|(_, &bit)| bit == 1) {
            for &r in m.m2.column(zeta2) {
                lane2[r as usize] ^= 1;
            }
        }
    }
    let mut y = vec![0_u8; 4096];
    for (t, lane2) in y2.iter().enumerate() {
        for (rho2, _) in lane2.iter().enumerate().filter(|(_, &bit)| bit == 1) {
            for &r in m.m3.column(t + 6 * rho2) {
                y[r as usize] ^= 1;
            }
        }
    }
    y
}

fn published_y(output: &ProverOutput) -> Vec<u8> {
    let bytes = output.proof.to_bytes();
    let proof = AggregateProof::from_bytes(&bytes).expect("wire round trip");
    assert_eq!(proof.final_vectors.len(), 1, "one segment");
    unpack_final_vector(&proof.final_vectors[0])
}

/// The flipped bit of the tampering cases: `b_0(j, rho, i)`; bit 0 keeps `w_1'` in
/// `[0, 44)`, so the F_p side's range identity still holds and only the bridge can object.
const FLIP: (usize, usize, usize) = (0, 77, 2);

#[allow(clippy::too_many_lines)]
fn joint_bridge(n: usize, seed: u64) {
    let f = Fixture::new(n, seed);
    let inputs = f.inputs();
    let signatures = f.signatures();
    let prover_inputs = ProverInputs {
        instance: &f.instance,
        records: &inputs,
        signatures: &signatures,
        relation: Relation::Hu,
    };
    let (manifest, main): (CommitmentManifest, MainTables) =
        main_tables(&prover_inputs).expect("round-1 tables");
    assert_eq!(manifest.segments(), 1);
    let geom = Geometry::new(n).unwrap();
    check_bit_orders(&f.witnesses, &main.bit_lanes(&manifest));
    let honest_records = f.sponge_records();
    for (record, w) in honest_records.iter().zip(&f.witnesses) {
        assert_eq!(
            record.c_tilde(),
            w.c_tilde,
            "the Flock record hashes to the published c~"
        );
    }

    let flock = SpongeProver::new(Default::default());
    let d_seg = BridgedSegment::d_seg_for(n);
    assert_eq!(
        d_seg,
        geom.d_seg(),
        "both sides use the segment of spec Section 1"
    );
    let seg = BridgedSegment::new(flock.gate(), n, d_seg, Profile::Fast);

    // Honest: one transcript, the same sd on both sides, one final vector.
    let bridge = RecordingSeed::default();
    let side = FlockSide::new(&seg, &honest_records);
    let output = prove(&prover_inputs, &bridge, &side, FlockProfile::Fast).expect("prove");
    let (sd, matrices) = bridge.last();
    let run = side.run();
    assert_eq!(output.bridge_seed, sd);
    assert_eq!(
        run.bridge_seed, sd,
        "the Flock side proves under the transcript's sd"
    );
    let direct = direct_final_vector(&f.witnesses, &matrices);
    assert!(direct.contains(&1) && direct.contains(&0));
    let y_fp = published_y(&output);
    assert_eq!(y_fp, direct, "F_p side's Y against the direct computation");
    assert_eq!(
        run.own_y, direct,
        "Flock side's Y against the direct computation"
    );
    assert_eq!(run.c_tilde, output.proof.c_tilde);
    let bytes = output.proof.to_bytes();
    let strict = FlockVerifier::new(&seg, false);
    f.verify(&bytes, &strict)
        .expect("honest mode-A aggregate verifies");
    assert_eq!(strict.outcomes(), [true]);

    let (t, j, rho) = FLIP;
    let i = n - 3;

    // F_p side: b_0 flipped in G_main, the round-2 tables (and so the published Y) those
    // of the honest bits under the transcript's matrices. Only the F_p side fails.
    let (_, position) = manifest.locate(Column::W1Bit(t as u8));
    let mut flipped = main.clone();
    flipped.columns_mut()[position][cell(j, rho, i)] ^= 1;
    let bridge = RecordingSeed::default();
    let stale: BridgeTamper<'_> = &|manifest, tables| {
        let (_, m) = bridge.last();
        *tables = BridgeGroupTables::build(manifest, &main, &m).expect("honest tables");
    };
    let side = FlockSide::new(&seg, &honest_records);
    let hooks = ProveHooks {
        unchecked: true,
        tamper_bridge: Some(stale),
    };
    let output = prove_with_tables(
        &prover_inputs,
        &manifest,
        &flipped,
        &bridge,
        &side,
        FlockProfile::Fast,
        &hooks,
    )
    .expect("prove");
    let (sd, m) = bridge.last();
    assert_ne!(sd, run.bridge_seed, "the round-1 commitment changed");
    let agreed = direct_final_vector(&f.witnesses, &m);
    assert_eq!(published_y(&output), agreed);
    assert_eq!(side.run().own_y, agreed);
    let own_fp = BridgeGroupTables::build(&manifest, &flipped, &m).unwrap();
    assert_ne!(
        own_fp.public_y().to_vec(),
        agreed,
        "one flipped bit changes Y (Lemma 1)"
    );
    let strict = FlockVerifier::new(&seg, false);
    match f.verify(&output.proof.to_bytes(), &strict) {
        Err(VerifyError::Stage(StageError::Terminal(1))) => {}
        other => panic!("F_p-side flip: expected the stage-1 terminal check, got {other:?}"),
    }
    assert_eq!(strict.outcomes(), [true], "the Flock side still accepts");

    // Flock side: the record's bit flipped in both Flock copies (sponge input and bridge
    // slot), the F_p side honest. Only the Flock side fails.
    let mut tampered = honest_records.clone();
    let q = stream_bit(t, j, rho);
    tampered[i].w1_encode[q / 8] ^= 1 << (q % 8);
    let bridge = RecordingSeed::default();
    let side = FlockSide::new(&seg, &tampered);
    let output = prove(&prover_inputs, &bridge, &side, FlockProfile::Fast).expect("prove");
    let (sd, m) = bridge.last();
    let run = side.run();
    let y_fp = published_y(&output);
    assert_eq!(y_fp, direct_final_vector(&f.witnesses, &m));
    let mut flipped_witnesses = f.witnesses.clone();
    flipped_witnesses[i].w1[rho][j] ^= 1 << t;
    assert_eq!(run.own_y, direct_final_vector(&flipped_witnesses, &m));
    assert_ne!(run.own_y, y_fp, "one flipped bit changes Y (Lemma 1)");
    let bytes = output.proof.to_bytes();
    let strict = FlockVerifier::new(&seg, false);
    match f.verify(&bytes, &strict) {
        Err(VerifyError::Keccak(0)) => {}
        other => panic!("Flock-side flip: expected the Flock sub-proof to fail, got {other:?}"),
    }
    assert_eq!(strict.outcomes(), [false]);
    let lenient = FlockVerifier::new(&seg, true);
    f.verify(&bytes, &lenient)
        .expect("with the Flock verdict ignored the F_p side accepts");
    assert_eq!(lenient.outcomes(), [false]);

    // The Flock rejection is the bridge rows alone: under the tampered records' own
    // statement (their c~), the F_p side's Y is rejected and their own Y verifies.
    let commitment = KeccakCommitment::from_bytes(output.proof.flock_commitments[0].clone());
    let fs = output.flock_seeds[0];
    let mu: Vec<[u8; 64]> = tampered.iter().map(|r| r.mu).collect();
    let joint = KeccakProof {
        bytes: output.proof.flock_proofs[0].clone(),
    };
    assert_ne!(run.c_tilde[i], output.proof.c_tilde[i]);
    assert!(seg
        .verify(
            &mu,
            &run.c_tilde,
            &commitment,
            sd,
            m.clone(),
            &y_fp,
            &fs,
            &joint
        )
        .is_err());
    let recommitted = seg.commit(&tampered);
    assert_eq!(recommitted.commitment(), &commitment);
    let own = seg.prove_with_final_vector(recommitted, sd, m.clone(), run.own_y.clone(), &fs);
    seg.verify(&mu, &run.c_tilde, &commitment, sd, m, &run.own_y, &fs, &own)
        .expect("the tampered records prove their own Y");
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Flock and Akita proofs; run with --release"
)]
fn joint_bridge_64_signatures() {
    on_large_stack(|| joint_bridge(64, 0x7015_0064));
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Flock and Akita proofs; run with --release"
)]
fn joint_bridge_256_signatures() {
    on_large_stack(|| joint_bridge(256, 0x7015_0256));
}
