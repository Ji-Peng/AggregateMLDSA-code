//! Mode A end to end (spec Sections 8 to 11, milestone M3): the aggregator turns records
//! `(pk_i, M_i, signature_i)` into an aggregate, the list of `c~_i` and one proof holding the
//! algebraic PIOP (`agg-piop`), the Flock Keccak sub-proofs (`agg-keccak`) and the bridge
//! data; the verifier takes `(pk_i, M_i)` and the aggregate and returns accept or reject with
//! the cost of every step.
//!
//! Transcript (spec Section 10, one Spongefish stream run by `agg_piop::prover`): event 0
//! instance (record digest over `(pk_i, M_i, c~_i)`, Flock profile `Fast`), event 1 the
//! round-1 Akita commitment `G_main`, event 2 the Flock commitments, event 3 the bridge seed
//! `sd`, event 4 the round-2 Akita commitments, event 5 the Flock seeds `fs_sigma`, event 6
//! the final vectors `Y_sigma` and the Flock proofs, events 7 to 11 the three sum-check
//! stages, event 12 the batched Akita opening. There is no projection seed: the residual is
//! proven through the committed quotient (spec Section 6, `docs/deviations.md` P2).
//!
//! Keccak side ([`FlockProver`], [`FlockVerifier`]): segment `sigma` (`2^13` record positions,
//! spec Section 1) is split into parts of `2^{d_part}` record positions, `d_part =
//! min(part_log2, d_seg)` (default [`DEFAULT_PART_LOG2`]); each part with live records is
//! one Flock union proof of its sponges and of a bridge slot proving the part's share
//! `Y_{sigma, s} = M^{(sigma)} b'_s`. The shares XOR to `Y_sigma` (the last share is
//! implied), which the F_p side proves for the committed bits `b` (`docs/deviations.md`
//! K10). The event-2 and event-6 items are those of [`crate::wire`]. Part `s` seeds its
//! Flock challenger with [`part_seed`]`(fs_sigma, s)`.
//!
//! Memory: the parts are committed one after the other at event 2 and only their
//! commitments are kept; at event 6 each part's witness is rebuilt from the records (the
//! Flock prover commits again internally, K7), proven and dropped, so at most one part's
//! Flock buffers are alive. The PIOP tables are `i8` columns (`agg_piop::source_tables`).

use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use agg_keccak::gate::ChainGate;
use agg_keccak::prover::{KeccakCommitment, KeccakProof, Profile};
use agg_keccak::segment::BridgedSegment;
use agg_keccak::sponge::{sponge_gate, SpongeRecord};
use agg_keccak::witness::SlotBuffers;
use agg_piop::binding::FlockProfile;
use agg_piop::manifest::CommitmentManifest;
use agg_piop::prover::{
    main_tables, prove_with_tables, BridgeDerivation, BridgeTamper, KeccakSide, ProveError,
    ProveHooks, ProveTimings, ProverInputs, SeededMatrices,
};
use agg_piop::public_data::KeyCache;
use agg_piop::source_tables::{BridgeGroupTables, MainTables, SourceTableError};
use agg_piop::verifier::{verify_timed, KeccakVerifier, SegmentContext, VerifyReport};
use agg_piop::wire::{pack_final_vector, unpack_final_vector, AggregateProof, ProofSize};
use agg_witness::bridge::BridgeMatrices;
use agg_witness::instance::{InstanceError, PublicInstance};
use agg_witness::padding::{main_index, Geometry};
use agg_witness::params::{C_TILDE_BYTES, K, N, PK_BYTES, SIG_BYTES, W1_ENCODE_BYTES};
use agg_witness::tables::BuildError;
use rayon::prelude::*;
use sha3::{Digest, Sha3_256};

use crate::wire::{SegmentCommitments, SegmentProofs};

/// The Flock profile of mode A in this build (event 0).
pub const FLOCK_PROFILE: FlockProfile = FlockProfile::Fast;
/// The Ligerito profile of the Flock sub-proofs.
pub const KECCAK_PROFILE: Profile = Profile::Fast;
/// Default `log2` of the record positions per Flock part: 512 records, whose part (bridge
/// slot `k_log = 22`) measured 11.5 GiB peak RSS and 676 KB of proof locally; 256 records
/// take 5.5 GiB and 522 KB, 128 records 3.2 GiB and 408 KB (`docs/cli.md` Section 4;
/// deviation K10). The cloud run should calibrate 10.
pub const DEFAULT_PART_LOG2: u32 = 9;
/// Smallest admitted part (`2^3` records, the smallest segment).
pub const MIN_PART_LOG2: u32 = 3;
/// Record positions per segment (`log2`, spec Section 1).
pub const SEGMENT_LOG2: u32 = 13;
/// Label of the per-part Flock seed.
pub const PART_SEED_LABEL: &[u8] = b"agg-mldsa/v1/flock-part-seed";
/// Bits of a final vector or share.
const Y_BITS: usize = 4096;

/// The Flock challenger seed of part `part` of a segment whose seed is `fs` (event 5):
/// `SHA3-256(label || fs || LE64(part))`.
#[must_use]
pub fn part_seed(fs: &[u8; 32], part: usize) -> [u8; 32] {
    let mut h = Sha3_256::new();
    h.update((PART_SEED_LABEL.len() as u64).to_le_bytes());
    h.update(PART_SEED_LABEL);
    h.update(fs);
    h.update((part as u64).to_le_bytes());
    h.finalize().into()
}

/// The split of one segment's live records into Flock parts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentParts {
    /// Segment index.
    pub sigma: usize,
    /// Global index of the segment's first record.
    pub first: usize,
    /// Live records of the segment.
    pub live: usize,
    /// `log2` of the record positions per part.
    pub d_part: u32,
}

impl SegmentParts {
    /// The parts of segment `sigma` for a requested part size `2^{part_log2}` (clamped to
    /// `[MIN_PART_LOG2, d_seg]`).
    #[must_use]
    pub fn plan(geometry: &Geometry, sigma: usize, part_log2: u32) -> Self {
        Self {
            sigma,
            first: (sigma << SEGMENT_LOG2).min(geometry.n),
            live: geometry.live_in_segment(sigma),
            d_part: part_log2.clamp(MIN_PART_LOG2, geometry.d_seg()),
        }
    }

    /// Parts with live records.
    #[must_use]
    pub fn count(&self) -> usize {
        self.live.div_ceil(1 << self.d_part)
    }

    /// Record range of part `s` within the segment.
    #[must_use]
    pub fn local(&self, s: usize) -> Range<usize> {
        let start = s << self.d_part;
        start..((s + 1) << self.d_part).min(self.live)
    }

    /// Global record range of part `s`.
    #[must_use]
    pub fn global(&self, s: usize) -> Range<usize> {
        let r = self.local(s);
        self.first + r.start..self.first + r.end
    }
}

/// Builds the Flock witness (sponge slot buffers and bridge block) of one part from its
/// records; the honest builder is `BridgedSegment::honest_parts`. Called at event 2 and
/// again at event 6, so it must be deterministic. Arguments: the part's context, `sigma`,
/// the part index, the part's records.
pub type PartWitness<'a> = &'a (dyn Fn(&BridgedSegment<'_>, usize, usize, &[SpongeRecord]) -> (SlotBuffers, Vec<u64>)
         + Sync);

/// Wall time and size of one Flock part on the prover side.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PartProveLog {
    /// Segment.
    pub sigma: usize,
    /// Part index.
    pub part: usize,
    /// Live records.
    pub records: usize,
    /// Event 2: context, witness and commitment.
    pub commit: Duration,
    /// Event 6: witness rebuilt, share, Flock proof.
    pub prove: Duration,
    /// Proof bytes.
    pub proof_bytes: usize,
}

/// The prover's Keccak side of mode A (module docs).
pub struct FlockProver<'g, 'h> {
    gate: &'g ChainGate,
    geometry: Geometry,
    records: Vec<SpongeRecord>,
    part_log2: u32,
    witness: Option<PartWitness<'h>>,
    state: Mutex<ProverState<'g>>,
}

struct PartState<'g> {
    seg: BridgedSegment<'g>,
    range: Range<usize>,
    commitment: KeccakCommitment,
}

#[derive(Default)]
struct ProverState<'g> {
    segments: Vec<Vec<PartState<'g>>>,
    log: Vec<PartProveLog>,
}

impl<'g, 'h> FlockProver<'g, 'h> {
    /// The Keccak side for the live records `(mu_i, w1Encode(w_1'_i))` of a batch.
    ///
    /// # Panics
    /// If the record count does not match the geometry.
    #[must_use]
    pub fn new(
        gate: &'g ChainGate,
        geometry: Geometry,
        records: Vec<SpongeRecord>,
        part_log2: u32,
        witness: Option<PartWitness<'h>>,
    ) -> Self {
        assert_eq!(
            records.len(),
            geometry.n,
            "one sponge record per live record"
        );
        Self {
            gate,
            geometry,
            records,
            part_log2,
            witness,
            state: Mutex::new(ProverState::default()),
        }
    }

    /// Per-part wall times and proof sizes (after [`KeccakSide::prove`]).
    #[must_use]
    pub fn log(&self) -> Vec<PartProveLog> {
        self.state.lock().expect("state").log.clone()
    }

    fn build(
        &self,
        seg: &BridgedSegment<'_>,
        sigma: usize,
        part: usize,
        records: &[SpongeRecord],
    ) -> (SlotBuffers, Vec<u64>) {
        match self.witness {
            Some(f) => f(seg, sigma, part, records),
            None => seg.honest_parts(records),
        }
    }
}

impl KeccakSide for FlockProver<'_, '_> {
    fn commitments(&self, segments: usize) -> Vec<Vec<u8>> {
        assert_eq!(segments, self.geometry.segments(), "segments of the batch");
        let mut state = self.state.lock().expect("state");
        state.segments.clear();
        state.log.clear();
        let mut items = Vec::with_capacity(segments);
        for sigma in 0..segments {
            let plan = SegmentParts::plan(&self.geometry, sigma, self.part_log2);
            let mut parts = Vec::with_capacity(plan.count());
            for s in 0..plan.count() {
                let start = Instant::now();
                let range = plan.global(s);
                let records = &self.records[range.clone()];
                let seg = BridgedSegment::new_part(
                    self.gate,
                    records.len(),
                    plan.d_part,
                    s,
                    KECCAK_PROFILE,
                );
                let (sponge, bridge_z) = self.build(&seg, sigma, s, records);
                // The committed buffers are dropped here and rebuilt at event 6.
                let commitment = seg
                    .commit_parts(sponge, bridge_z, records)
                    .commitment()
                    .clone();
                state.log.push(PartProveLog {
                    sigma,
                    part: s,
                    records: records.len(),
                    commit: start.elapsed(),
                    ..PartProveLog::default()
                });
                parts.push(PartState {
                    seg,
                    range,
                    commitment,
                });
            }
            items.push(
                SegmentCommitments {
                    d_part: plan.d_part,
                    commitments: parts
                        .iter()
                        .map(|p| p.commitment.as_bytes().to_vec())
                        .collect(),
                }
                .to_bytes(),
            );
            state.segments.push(parts);
        }
        items
    }

    fn prove(
        &self,
        bridge_seed: &[u8; 32],
        matrices: &BridgeMatrices,
        seeds: &[[u8; 32]],
        final_vectors: &[Vec<u8>],
    ) -> Vec<Vec<u8>> {
        let mut state = self.state.lock().expect("state");
        let segments = std::mem::take(&mut state.segments);
        assert_eq!(segments.len(), seeds.len(), "one seed per segment");
        assert_eq!(segments.len(), final_vectors.len(), "one Y per segment");
        let matrices = Arc::new(matrices.clone());
        let mut entry = 0;
        let mut items = Vec::with_capacity(segments.len());
        for (sigma, parts) in segments.into_iter().enumerate() {
            let count = parts.len();
            let mut acc = vec![0_u8; Y_BITS];
            let mut item = SegmentProofs {
                shares: Vec::with_capacity(count.saturating_sub(1)),
                proofs: Vec::with_capacity(count),
            };
            for (s, part) in parts.into_iter().enumerate() {
                let start = Instant::now();
                let records = &self.records[part.range.clone()];
                let (sponge, bridge_z) = self.build(&part.seg, sigma, s, records);
                let committed = part
                    .seg
                    .reattach(sponge, bridge_z, records, part.commitment);
                let share = if s + 1 < count {
                    let own = part.seg.final_vector(&committed, &matrices);
                    item.shares.push(pack_final_vector(&own));
                    acc.iter_mut().zip(&own).for_each(|(a, b)| *a ^= b);
                    own
                } else {
                    acc.iter()
                        .zip(&final_vectors[sigma])
                        .map(|(a, y)| a ^ y)
                        .collect()
                };
                let proof = part.seg.prove_with_final_vector(
                    committed,
                    *bridge_seed,
                    matrices.clone(),
                    share,
                    &part_seed(&seeds[sigma], s),
                );
                let log = &mut state.log[entry];
                debug_assert_eq!((log.sigma, log.part), (sigma, s));
                log.prove = start.elapsed();
                log.proof_bytes = proof.bytes.len();
                entry += 1;
                item.proofs.push(proof.bytes);
            }
            items.push(item.to_bytes());
        }
        items
    }
}

/// Wall time and outcome of one Flock part on the verifier side.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PartVerifyLog {
    /// Segment.
    pub sigma: usize,
    /// Part index.
    pub part: usize,
    /// Live records.
    pub records: usize,
    /// Building the part's Flock context (registry, wiring).
    pub setup: Duration,
    /// The Flock verification.
    pub verify: Duration,
    /// Accepted.
    pub accepted: bool,
}

/// Why the Keccak side of a segment was rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SegmentRejection {
    /// A malformed segment item.
    Item(crate::wire::ItemError),
    /// An empty segment with a nonzero `Y_sigma`.
    EmptySegment,
    /// A Flock part was rejected.
    Part(usize, String),
}

/// The verifier's Keccak side of mode A (module docs).
pub struct FlockVerifier<'g> {
    gate: &'g ChainGate,
    matrices: Mutex<Option<([u8; 32], Arc<BridgeMatrices>)>>,
    log: Mutex<Vec<PartVerifyLog>>,
    rejection: Mutex<Option<(usize, SegmentRejection)>>,
}

impl<'g> FlockVerifier<'g> {
    /// The verifier side over the sponge gate `gate`.
    #[must_use]
    pub fn new(gate: &'g ChainGate) -> Self {
        Self {
            gate,
            matrices: Mutex::new(None),
            log: Mutex::new(Vec::new()),
            rejection: Mutex::new(None),
        }
    }

    /// Per-part wall times and outcomes of the last run.
    #[must_use]
    pub fn log(&self) -> Vec<PartVerifyLog> {
        self.log.lock().expect("log").clone()
    }

    /// The segment and reason of the last rejection, if any.
    #[must_use]
    pub fn rejection(&self) -> Option<(usize, SegmentRejection)> {
        self.rejection.lock().expect("rejection").clone()
    }

    fn matrices_for(&self, sd: &[u8; 32], matrices: &BridgeMatrices) -> Arc<BridgeMatrices> {
        let mut cached = self.matrices.lock().expect("matrices");
        match &*cached {
            Some((seed, m)) if seed == sd => m.clone(),
            _ => {
                let m = Arc::new(matrices.clone());
                *cached = Some((*sd, m.clone()));
                m
            }
        }
    }

    fn reject(&self, sigma: usize, why: SegmentRejection) -> bool {
        *self.rejection.lock().expect("rejection") = Some((sigma, why));
        false
    }
}

impl KeccakVerifier for FlockVerifier<'_> {
    fn profile(&self) -> FlockProfile {
        FLOCK_PROFILE
    }

    fn verify(&self, segment: &SegmentContext<'_>) -> bool {
        let sigma = segment.sigma;
        if sigma == 0 {
            self.log.lock().expect("log").clear();
            *self.rejection.lock().expect("rejection") = None;
        }
        let live = segment.mu.len();
        let commitments = match SegmentCommitments::from_bytes(segment.commitment) {
            Ok(c) => c,
            Err(e) => return self.reject(sigma, SegmentRejection::Item(e)),
        };
        let d_part = commitments.d_part;
        if d_part < MIN_PART_LOG2 || d_part > segment.geometry.d_seg() {
            return self.reject(
                sigma,
                SegmentRejection::Item(crate::wire::ItemError::PartSize(d_part as u8)),
            );
        }
        let plan = SegmentParts {
            sigma,
            first: 0,
            live,
            d_part,
        };
        let count = plan.count();
        if commitments.commitments.len() != count {
            return self.reject(
                sigma,
                SegmentRejection::Item(crate::wire::ItemError::Parts {
                    expected: count,
                    actual: commitments.commitments.len(),
                }),
            );
        }
        let proofs = match SegmentProofs::from_bytes(segment.proof, count) {
            Ok(p) => p,
            Err(e) => return self.reject(sigma, SegmentRejection::Item(e)),
        };
        if count == 0 {
            // No live record: the Flock side's bits are all zero, so Y_sigma must be zero.
            return segment.final_vector.iter().all(|&b| b == 0)
                || self.reject(sigma, SegmentRejection::EmptySegment);
        }
        let shares: Vec<Vec<u8>> = proofs.shares.iter().map(unpack_final_vector).collect();
        let mut last = segment.final_vector.to_vec();
        for share in &shares {
            last.iter_mut().zip(share).for_each(|(a, b)| *a ^= b);
        }
        let matrices = self.matrices_for(segment.bridge_seed, segment.matrices);
        for s in 0..count {
            let range = plan.local(s);
            let start = Instant::now();
            let seg = BridgedSegment::new_part(self.gate, range.len(), d_part, s, KECCAK_PROFILE);
            let setup = start.elapsed();
            // Parts 0 .. count - 1 carry their share; the last one's is implied.
            let share = shares.get(s).unwrap_or(&last);
            let outcome = seg.verify(
                &segment.mu[range.clone()],
                &segment.c_tilde[range.clone()],
                &KeccakCommitment::from_bytes(commitments.commitments[s].clone()),
                *segment.bridge_seed,
                matrices.clone(),
                share,
                &part_seed(segment.seed, s),
                &KeccakProof {
                    bytes: proofs.proofs[s].clone(),
                },
            );
            self.log.lock().expect("log").push(PartVerifyLog {
                sigma,
                part: s,
                records: range.len(),
                setup,
                verify: start.elapsed() - setup,
                accepted: outcome.is_ok(),
            });
            if let Err(e) = outcome {
                return self.reject(sigma, SegmentRejection::Part(s, e.to_string()));
            }
        }
        true
    }
}

/// A signed record `(pk, M, signature)` (pure ML-DSA-44, empty context).
pub type SignedRecord<'a> = (&'a [u8], &'a [u8], &'a [u8]);

/// Aggregator configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AggregatorConfig {
    /// Requested `log2` of the record positions per Flock part (clamped per segment to
    /// `[MIN_PART_LOG2, d_seg]`).
    pub part_log2: u32,
}

impl Default for AggregatorConfig {
    fn default() -> Self {
        Self {
            part_log2: DEFAULT_PART_LOG2,
        }
    }
}

/// An aggregator failure.
#[derive(Debug)]
pub enum AggregateError {
    /// No record, or more than `2^15`.
    Count(usize),
    /// Record `i` has a public key or signature of the wrong length.
    Malformed(usize, &'static str),
    /// Signature `i` does not verify (strict FIPS 204 verification during the derivation).
    Signature(usize, String),
    /// The prover failed.
    Prove(ProveError),
}

impl std::fmt::Display for AggregateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Count(n) => write!(f, "{n} records: an aggregate holds 1 to 2^15"),
            Self::Malformed(i, what) => write!(f, "record {i}: {what}"),
            Self::Signature(i, why) => write!(f, "record {i}: signature rejected ({why})"),
            Self::Prove(e) => write!(f, "prover: {e}"),
        }
    }
}

impl std::error::Error for AggregateError {}

impl From<ProveError> for AggregateError {
    fn from(e: ProveError) -> Self {
        match e {
            ProveError::Tables(SourceTableError::Build(BuildError::Derive(i, why))) => {
                Self::Signature(i, format!("{why:?}"))
            }
            other => Self::Prove(other),
        }
    }
}

/// What the aggregator measured.
#[derive(Clone, Debug)]
pub struct ProveReport {
    /// Live records `N`.
    pub records: usize,
    /// The public instance: key expansion (`ExpandA`, inverse NTTs), `tr`, `mu`, `c`, `T`.
    pub instance: Duration,
    /// The PIOP prover's steps (`tables` is the strict witness derivation and `G_main`;
    /// `keccak` covers every Flock part's commitment and proof).
    pub timings: ProveTimings,
    /// The Flock side's records (`w1Encode` of the committed bits).
    pub flock_records: Duration,
    /// Per Flock part.
    pub parts: Vec<PartProveLog>,
    /// Bytes per part of the aggregate.
    pub size: ProofSize,
    /// The bridge seed `sd`.
    pub bridge_seed: [u8; 32],
    /// End to end.
    pub total: Duration,
}

impl ProveReport {
    /// Aggregate bytes.
    #[must_use]
    pub const fn bytes(&self) -> usize {
        self.size.total()
    }

    /// Aggregate bytes per signature.
    #[must_use]
    pub fn bytes_per_signature(&self) -> f64 {
        self.bytes() as f64 / self.records as f64
    }
}

/// Lines `name value` of a [`ProofSize`].
#[must_use]
pub fn size_lines(size: &ProofSize, records: usize) -> String {
    let parts = [
        ("header", size.header),
        ("c~ list", size.c_tilde),
        ("Akita commitments", size.akita_commitments),
        ("Flock commitments", size.keccak_commitments),
        ("final vectors", size.final_vectors),
        ("Flock proofs and shares", size.keccak_proofs),
        ("PIOP rounds and claims", size.piop),
        ("Akita opening", size.opening),
    ];
    let mut out = String::new();
    for (name, bytes) in parts {
        out.push_str(&format!(
            "  {name:<26} {bytes:>10} B  {:>9.1} B/sig\n",
            bytes as f64 / records as f64
        ));
    }
    out.push_str(&format!(
        "  {:<26} {:>10} B  {:>9.1} B/sig",
        "total",
        size.total(),
        size.total() as f64 / records as f64
    ));
    out
}

impl std::fmt::Display for ProveReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        let t = &self.timings;
        writeln!(f, "prover, N = {}:", self.records)?;
        for (name, d) in [
            ("instance (ExpandA, mu, c, T)", self.instance),
            ("witness and G_main tables", t.tables),
            ("Flock records", self.flock_records),
            ("Akita setup", t.setup),
            ("event 1: G_main commitment", t.round1),
            ("events 2, 6: Flock parts", t.keccak),
            ("events 3-5: matrices, round 2", t.round2),
            ("events 7, 8: tau, pre-stage", t.prestage),
            ("stage 1", t.stage1),
            ("stage 2", t.stage2),
            ("stage 3", t.stage3),
            ("event 12: Akita opening", t.opening),
        ] {
            writeln!(f, "  {name:<32} {:>10.1} ms", ms(d))?;
        }
        writeln!(f, "  {:<32} {:>10.1} ms", "total", ms(self.total))?;
        for p in &self.parts {
            writeln!(
                f,
                "  Flock part ({}, {}): {} records, commit {:.1} ms, prove {:.1} ms, {} B",
                p.sigma,
                p.part,
                p.records,
                ms(p.commit),
                ms(p.prove),
                p.proof_bytes
            )?;
        }
        writeln!(f, "aggregate, N = {}:", self.records)?;
        write!(f, "{}", size_lines(&self.size, self.records))
    }
}

/// An aggregate: the proof (which starts with the `c~_i` list) and its bytes.
pub struct Aggregate {
    /// The parsed aggregate.
    pub proof: AggregateProof,
    /// The wire bytes (`agg_piop::wire`): header, `c~_i` list, then the proof blob.
    pub bytes: Vec<u8>,
    /// Timings and sizes.
    pub report: ProveReport,
}

impl Aggregate {
    /// The `c~_i` list.
    #[must_use]
    pub fn c_tilde(&self) -> &[[u8; C_TILDE_BYTES]] {
        &self.proof.c_tilde
    }
}

/// A change applied to the honest round-1 tables before they are committed.
pub type MainTamper<'a> = &'a dyn Fn(&CommitmentManifest, &mut MainTables);
/// A change applied to the Flock side's records.
pub type RecordTamper<'a> = &'a dyn Fn(&mut Vec<SpongeRecord>);

/// Adversarial-test hooks of [`Aggregator::aggregate_with`]: produce aggregates from tampered
/// witnesses (the PIOP stages then run unchecked).
#[doc(hidden)]
#[derive(Default)]
pub struct AggregateHooks<'a> {
    /// Change the round-1 tables before they are committed.
    pub tamper_main: Option<MainTamper<'a>>,
    /// Change the round-2 tables before they are committed.
    pub tamper_bridge: Option<BridgeTamper<'a>>,
    /// The Flock side reads the bits of the tampered round-1 tables (both copies tampered);
    /// otherwise it reads the honest bits.
    pub flock_from_tampered: bool,
    /// Change the Flock side's records.
    pub tamper_flock_records: Option<RecordTamper<'a>>,
    /// Build each Flock part's witness.
    pub flock_witness: Option<PartWitness<'a>>,
    /// Commit the round-2 tables (parities, carries, `Y_sigma`) of the honest bits under the
    /// transcript's matrices while `G_main` holds the tampered bits (the F_p-side flip of the
    /// joint bridge test). Replaces `tamper_bridge`.
    pub stale_bridge: bool,
}

/// The derivation of spec 9.1 that keeps the last matrices it derived (test hooks only).
#[derive(Default)]
struct RecordingMatrices(Mutex<Option<Arc<BridgeMatrices>>>);

impl BridgeDerivation for RecordingMatrices {
    fn matrices(&self, seed: &[u8; 32]) -> std::borrow::Cow<'_, BridgeMatrices> {
        let m = agg_bridge::seed::derive_matrices(seed);
        *self.0.lock().expect("matrices") = Some(Arc::new(m.clone()));
        std::borrow::Cow::Owned(m)
    }

    fn seed_derived(&self) -> bool {
        true
    }
}

/// The Flock side's records of a batch: `mu_i` and the committed bits `b_t(j, rho, i)` (low
/// bit of each cell) packed in `w1Encode` order, bit `1536 rho + 6 j + t` (FIPS 204
/// Algorithms 16 and 28). For honest tables this is `w1Encode(w_1'_i)`
/// (`agg_piop::mode_b::w1_encodings`); unlike `w1Encode` it also packs out-of-range values
/// (adversarial tests).
#[must_use]
pub fn sponge_records(
    instance: &PublicInstance,
    manifest: &CommitmentManifest,
    main: &MainTables,
) -> Vec<SpongeRecord> {
    let lanes = main.bit_lanes(manifest);
    instance
        .records
        .par_iter()
        .enumerate()
        .map(|(i, record)| {
            let mut w1_encode = [0_u8; W1_ENCODE_BYTES];
            for rho in 0..K {
                for j in 0..N {
                    let x = main_index(j, rho, i);
                    for (t, lane) in lanes.iter().enumerate() {
                        if lane[x] & 1 == 1 {
                            let q = 1536 * rho + 6 * j + t;
                            w1_encode[q / 8] |= 1 << (q % 8);
                        }
                    }
                }
            }
            SpongeRecord {
                mu: record.mu,
                w1_encode,
            }
        })
        .collect()
}

/// The mode-A aggregator (holds the sponge gate, built once).
pub struct Aggregator {
    gate: ChainGate,
    config: AggregatorConfig,
}

impl Aggregator {
    /// An aggregator with the given configuration.
    #[must_use]
    pub fn new(config: AggregatorConfig) -> Self {
        Self {
            gate: sponge_gate(),
            config,
        }
    }

    /// The configuration.
    #[must_use]
    pub const fn config(&self) -> AggregatorConfig {
        self.config
    }

    /// Aggregate signed records: every signature is verified strictly (FIPS 204) while its
    /// witness is derived, so an invalid one is refused with its index.
    ///
    /// # Errors
    /// On a malformed or invalid record, or a prover failure.
    pub fn aggregate(&self, records: &[SignedRecord<'_>]) -> Result<Aggregate, AggregateError> {
        self.aggregate_with(records, &AggregateHooks::default())
    }

    /// [`Self::aggregate`] with adversarial-test hooks.
    ///
    /// # Errors
    /// As [`Self::aggregate`].
    #[doc(hidden)]
    pub fn aggregate_with(
        &self,
        records: &[SignedRecord<'_>],
        hooks: &AggregateHooks<'_>,
    ) -> Result<Aggregate, AggregateError> {
        let start = Instant::now();
        let n = records.len();
        if n == 0 || n > 1 << 15 {
            return Err(AggregateError::Count(n));
        }
        for (i, (pk, _, sig)) in records.iter().enumerate() {
            if pk.len() != PK_BYTES {
                return Err(AggregateError::Malformed(i, "public key length"));
            }
            if sig.len() != SIG_BYTES {
                return Err(AggregateError::Malformed(i, "signature length"));
            }
        }
        let inputs: Vec<(&[u8], &[u8], [u8; C_TILDE_BYTES])> = records
            .iter()
            .map(|&(pk, msg, sig)| (pk, msg, sig[..C_TILDE_BYTES].try_into().expect("32 bytes")))
            .collect();
        let signatures: Vec<&[u8]> = records.iter().map(|r| r.2).collect();
        let instance = PublicInstance::new(&inputs).map_err(|e| match e {
            InstanceError::PublicKey(i) => AggregateError::Malformed(i, "public key"),
            InstanceError::Geometry(_) => AggregateError::Count(n),
        })?;
        let instance_time = start.elapsed();
        let prover_inputs = ProverInputs {
            instance: &instance,
            records: &inputs,
            signatures: &signatures,
        };
        let clock = Instant::now();
        let (manifest, honest) = main_tables(&prover_inputs)?;
        let tables_time = clock.elapsed();
        let clock = Instant::now();
        let tampered = hooks.tamper_main.map(|tamper| {
            let mut main = honest.clone();
            tamper(&manifest, &mut main);
            main
        });
        let main = tampered.as_ref().unwrap_or(&honest);
        let mut flock_records = sponge_records(
            &instance,
            &manifest,
            if hooks.flock_from_tampered {
                main
            } else {
                &honest
            },
        );
        if let Some(tamper) = hooks.tamper_flock_records {
            tamper(&mut flock_records);
        }
        let flock_time = clock.elapsed();
        let side = FlockProver::new(
            &self.gate,
            main.geometry(),
            flock_records,
            self.config.part_log2,
            hooks.flock_witness,
        );
        let recording = RecordingMatrices::default();
        let stale = |manifest: &CommitmentManifest, tables: &mut BridgeGroupTables| {
            let matrices = recording.0.lock().expect("matrices").clone();
            *tables = BridgeGroupTables::build(manifest, &honest, &matrices.expect("derived"))
                .expect("honest round-2 tables");
        };
        let prove_hooks = ProveHooks {
            unchecked: hooks.tamper_main.is_some()
                || hooks.tamper_bridge.is_some()
                || hooks.stale_bridge,
            tamper_bridge: if hooks.stale_bridge {
                Some(&stale)
            } else {
                hooks.tamper_bridge
            },
        };
        let mut output = if hooks.stale_bridge {
            prove_with_tables(
                &prover_inputs,
                &manifest,
                main,
                &recording,
                &side,
                FLOCK_PROFILE,
                &prove_hooks,
            )?
        } else {
            prove_with_tables(
                &prover_inputs,
                &manifest,
                main,
                &SeededMatrices,
                &side,
                FLOCK_PROFILE,
                &prove_hooks,
            )?
        };
        output.timings.tables = tables_time;
        output.timings.total += tables_time;
        let bytes = output.proof.to_bytes();
        let size = output.proof.size();
        debug_assert_eq!(size.total(), bytes.len());
        let report = ProveReport {
            records: n,
            instance: instance_time,
            timings: output.timings,
            flock_records: flock_time,
            parts: side.log(),
            size,
            bridge_seed: output.bridge_seed,
            total: start.elapsed(),
        };
        Ok(Aggregate {
            proof: output.proof,
            bytes,
            report,
        })
    }
}

/// The verifier's verdict with the cost of every step.
#[derive(Debug)]
pub struct VerifyCostReport {
    /// Accept (`Ok`) or the first failing check, with the PIOP verifier's step timings.
    pub report: VerifyReport,
    /// Per Flock part (in verification order; a rejected run stops at the failing part).
    pub parts: Vec<PartVerifyLog>,
    /// The segment and reason when the Keccak side rejected.
    pub rejection: Option<(usize, SegmentRejection)>,
    /// Records `(pk_i, M_i)` given.
    pub records: usize,
    /// Aggregate bytes.
    pub aggregate_bytes: usize,
    /// Whether a key cache was supplied.
    pub cached: bool,
}

impl VerifyCostReport {
    /// True when the aggregate was accepted.
    #[must_use]
    pub fn accepted(&self) -> bool {
        self.report.accepted()
    }

    /// Total verification time.
    #[must_use]
    pub fn total(&self) -> Duration {
        self.report.timings.total
    }
}

impl std::fmt::Display for VerifyCostReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        let verdict = match &self.report.outcome {
            Ok(_) => "accept".to_string(),
            Err(e) => format!("reject ({e})"),
        };
        writeln!(
            f,
            "verifier, N = {}, {} keys, aggregate {} B: {verdict}",
            self.records,
            if self.cached { "cached" } else { "uncached" },
            self.aggregate_bytes
        )?;
        let labels = [
            ("parse", "parse, manifest, PCS verifier"),
            ("public_data", "pkDecode, tr, mu, SampleInBall"),
            ("transcript", "events 0-6 replay"),
            ("matrices", "bridge matrices from sd"),
            ("keccak", "Flock parts"),
            ("prestage", "tau, Y(tau_3), pre-stage"),
            ("stage1", "stage 1"),
            ("stage2", "stage 2"),
            ("stage3_rounds", "stage 3 rounds"),
            ("public_t", "T_i and T(r2)"),
            ("public_a", "ExpandA/cache, w_A(r3)"),
            ("stage3_terminal", "stage 3 terminal"),
            ("bits", "bit check (mode B only)"),
            ("opening", "Akita opening"),
        ];
        for (name, d) in self.report.timings.steps() {
            let label = labels
                .iter()
                .find(|(n, _)| *n == name)
                .map_or(name, |(_, l)| l);
            writeln!(f, "  {name:<16} {label:<32} {:>10.2} ms", ms(d))?;
        }
        if let Ok(v) = &self.report.outcome {
            writeln!(f, "  keys expanded in this run: {}", v.expanded_keys)?;
        }
        for p in &self.parts {
            writeln!(
                f,
                "  Flock part ({}, {}): {} records, setup {:.2} ms, verify {:.2} ms, {}",
                p.sigma,
                p.part,
                p.records,
                ms(p.setup),
                ms(p.verify),
                if p.accepted { "accepted" } else { "rejected" }
            )?;
        }
        if let Some((sigma, why)) = &self.rejection {
            writeln!(f, "  Keccak side of segment {sigma} rejected: {why:?}")?;
        }
        write!(f, "  {:<49} {:>10.2} ms", "total", ms(self.total()))
    }
}

/// The mode-A verifier (holds the sponge gate, built once).
pub struct AggregateVerifier {
    gate: ChainGate,
}

impl Default for AggregateVerifier {
    fn default() -> Self {
        Self::new()
    }
}

impl AggregateVerifier {
    /// A verifier.
    #[must_use]
    pub fn new() -> Self {
        Self {
            gate: sponge_gate(),
        }
    }

    /// Verify `aggregate` for the records `(pk_i, M_i)`. The verifier computes `tr_i`, `mu_i`,
    /// `c_i = SampleInBall(c~_i)`, `T_i` and the `ExpandA` data itself; with `cache`, keys
    /// found there are not expanded again and new expansions are added.
    pub fn verify(
        &self,
        statements: &[(&[u8], &[u8])],
        aggregate: &[u8],
        cache: Option<&mut KeyCache>,
    ) -> VerifyCostReport {
        let keccak = FlockVerifier::new(&self.gate);
        let cached = cache.is_some();
        let report = verify_timed(
            statements,
            aggregate,
            &SeededMatrices,
            &keccak,
            FLOCK_PROFILE,
            cache,
        );
        VerifyCostReport {
            report,
            parts: keccak.log(),
            rejection: keccak.rejection(),
            records: statements.len(),
            aggregate_bytes: aggregate.len(),
            cached,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parts_tile_every_segment() {
        for (n, part_log2) in [
            (1, 10),
            (20, 3),
            (64, 10),
            (512, 7),
            (1000, 9),
            ((1 << 13) + 5, 10),
            ((1 << 14) + 1, 12),
        ] {
            let geometry = Geometry::new(n).unwrap();
            let mut covered = 0;
            for sigma in 0..geometry.segments() {
                let plan = SegmentParts::plan(&geometry, sigma, part_log2);
                assert!(plan.d_part >= MIN_PART_LOG2 && plan.d_part <= geometry.d_seg());
                assert_eq!(
                    plan.d_part,
                    part_log2.clamp(MIN_PART_LOG2, geometry.d_seg())
                );
                let mut next = plan.first;
                for s in 0..plan.count() {
                    let r = plan.global(s);
                    assert_eq!(r.start, next);
                    assert!(!r.is_empty() && r.len() <= 1 << plan.d_part);
                    next = r.end;
                }
                assert_eq!(next - plan.first, plan.live);
                covered += plan.live;
            }
            assert_eq!(covered, n, "n {n}");
        }
        // A trailing segment of padded records only has no part.
        let geometry = Geometry::new((1 << 14) + 1).unwrap();
        assert_eq!(geometry.segments(), 4);
        assert_eq!(SegmentParts::plan(&geometry, 3, 10).count(), 0);
        assert_eq!(SegmentParts::plan(&geometry, 2, 10).count(), 1);
    }

    /// A trailing segment of padded records only (`N = 2^14 + 1`, segment 3): no Flock part,
    /// empty items, and `Y_sigma` must be zero; malformed items are rejected with a reason.
    #[test]
    fn an_empty_segment_needs_a_zero_final_vector_and_empty_items() {
        let gate = sponge_gate();
        let verifier = FlockVerifier::new(&gate);
        let geometry = Geometry::new((1 << 14) + 1).unwrap();
        assert_eq!(geometry.live_in_segment(3), 0);
        let matrices = BridgeMatrices::random_for_tests(3);
        let item = |d_part: u32, commitments: Vec<Vec<u8>>| {
            SegmentCommitments {
                d_part,
                commitments,
            }
            .to_bytes()
        };
        let zero = vec![0_u8; Y_BITS];
        let mut one = zero.clone();
        one[5] = 1;
        let check = |commitment: &[u8], y: &[u8], proof: &[u8]| {
            verifier.verify(&SegmentContext {
                sigma: 3,
                geometry,
                commitment,
                seed: &[0; 32],
                bridge_seed: &[0; 32],
                mu: &[],
                c_tilde: &[],
                final_vector: y,
                proof,
                matrices: &matrices,
            })
        };
        let empty = item(10, Vec::new());
        assert!(check(&empty, &zero, &[]));
        assert!(verifier.rejection().is_none());
        assert!(!check(&empty, &one, &[]));
        assert_eq!(
            verifier.rejection(),
            Some((3, SegmentRejection::EmptySegment))
        );
        assert!(!check(&empty, &zero, &[0]));
        assert!(matches!(
            verifier.rejection(),
            Some((3, SegmentRejection::Item(crate::wire::ItemError::Trailing)))
        ));
        assert!(!check(&item(10, vec![vec![1]]), &zero, &[]));
        assert!(matches!(
            verifier.rejection(),
            Some((
                3,
                SegmentRejection::Item(crate::wire::ItemError::Parts {
                    expected: 0,
                    actual: 1
                })
            ))
        ));
        for d_part in [2, 14] {
            assert!(!check(&item(d_part, Vec::new()), &zero, &[]));
            assert!(matches!(
                verifier.rejection(),
                Some((
                    3,
                    SegmentRejection::Item(crate::wire::ItemError::PartSize(_))
                ))
            ));
        }
        assert!(!check(&[9, 10, 0, 0, 0, 0], &zero, &[]));
        assert!(verifier.log().is_empty(), "no Flock part was run");
    }

    /// The Flock records of honest tables are `mu_i` and `w1Encode(w_1'_i)`, which hash to the
    /// published `c~_i`.
    #[test]
    fn sponge_records_of_honest_tables_are_the_w1_encodings() {
        let records = agg_witness::gen::records_shared_keys(0x5e, 9, 2);
        let instance = PublicInstance::from_records(&records).unwrap();
        let manifest = CommitmentManifest::for_batch(records.len()).unwrap();
        let signatures: Vec<&[u8]> = records.iter().map(|r| r.sig.as_slice()).collect();
        let main = MainTables::build(&manifest, &instance, &signatures).unwrap();
        let flock = sponge_records(&instance, &manifest, &main);
        let encodings = agg_piop::mode_b::w1_encodings(&manifest, &main);
        assert_eq!(flock.len(), 9);
        for ((f, e), r) in flock.iter().zip(&encodings).zip(&instance.records) {
            assert_eq!(f.w1_encode, *e);
            assert_eq!(f.mu, r.mu);
            assert_eq!(f.c_tilde(), r.c_tilde);
        }
    }

    #[test]
    fn part_seeds_are_distinct_per_part_and_segment_seed() {
        let a = part_seed(&[1; 32], 0);
        assert_ne!(a, part_seed(&[1; 32], 1));
        assert_ne!(a, part_seed(&[2; 32], 0));
        assert_eq!(a, part_seed(&[1; 32], 0));
    }
}
