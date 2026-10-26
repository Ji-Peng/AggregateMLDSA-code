//! Mode A end to end (spec Sections 8 to 11, milestone M3): the aggregator turns records
//! `(pk_i, M_i, signature_i)` into an aggregate, the list of `c~_i` and one proof holding the
//! algebraic PIOP (`agg-piop`), the Flock Keccak sub-proofs (`agg-keccak`) and the bridge
//! data; the verifier takes `(pk_i, M_i)` and the aggregate and returns accept or reject with
//! the cost of every step.
//!
//! Transcript (spec Section 10, one Spongefish stream run by `agg_piop::prover`): event 0
//! instance (record digest over `(pk_i, M_i, c~_i)`, the Flock profile: `Fast`, `Slim` or
//! `Slim8`, [`AggregatorConfig::profile`]), event 1 the
//! round-1 Akita commitment `G_main`, event 2 the Flock commitments, event 3 the bridge seed
//! `sd`, event 4 the round-2 Akita commitments, event 5 the Flock seeds `fs_sigma`, event 6
//! the final vectors `Y_sigma` and the Flock proofs, events 7 to 11 the three sum-check
//! stages, event 12 the batched Akita opening. There is no projection seed: the residual is
//! proven through the committed quotient (spec Section 6, `docs/deviations.md` P2).
//!
//! Keccak side ([`FlockProver`], [`FlockVerifier`]): segment `sigma` (`2^13` record positions,
//! spec Section 1) is split into parts of `2^{d_part}` record positions, `d_part =
//! clamp(part_log2, min_part_log2, d_seg)` (default [`DEFAULT_PART_LOG2`]; at most two parts
//! per segment, [`MAX_PARTS_LOG2`]); each part with live records is
//! one tied part (`agg_keccak::tied`, tied-part proof version 5, deviation K21): a Flock
//! sponge instance of its records whose opening also proves the bit claim of the virtual
//! parity chain, a sum-check that reduces the part's share `Y_{sigma, s} = M^{(sigma)} b'_s`
//! to that claim (`docs/parity-chain.md`). The shares XOR to `Y_sigma` (the last share is
//! implied), which the F_p side proves for the committed bits `b` (`docs/deviations.md`
//! K10). The event-2 item carries one cap per part (the sponge instance's), the event-6 item
//! one proof per part ([`crate::wire`]). Part `s` seeds its Flock challengers with
//! [`part_seed`]`(fs_sigma, s)`.
//!
//! Span (deviation K16, [`AggregatorConfig::span`], the default): a batch of more than one
//! segment whose parts are whole segments is proven by ONE tied part over all its segments
//! (`agg_keccak::tied::TiedPart::new_span`): one sponge instance of all `N` records and one
//! parity chain over every segment's bits, whose claim `<eq_Y, Y>` covers every `Y_sigma`, so
//! the Flock opening is paid once per batch instead of once per segment. Segment 0's items carry the span's cap and
//! proof, the other segments' items say "covered by segment 0" ([`crate::wire`]); the
//! verifier checks the span when it reaches the last segment, seeded by
//! [`span_seed`]`(fs_0, .., fs_{S-1})`.
//!
//! Memory: the parts are committed one after the other at event 2; each keeps its
//! commitment and, by default, the sponge commitment's Flock prover data (its Reed-Solomon
//! codeword and Merkle tree, deviation K14), so that event 6 opens it without committing
//! again; [`Aggregator::keep_commit_data`]`(false)` drops it and Flock commits a second time
//! (K7). The witness buffers are dropped after event 2 and rebuilt from the records at event
//! 6, proven and dropped, so at most one part's Flock buffers are alive. The PIOP tables are
//! `i8` columns (`agg_piop::source_tables`).

use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use agg_keccak::flock_union::CommitData;
use agg_keccak::gate::ChainGate;
use agg_keccak::prover::{KeccakProof, Profile};
use agg_keccak::sponge::{sponge_gate, SpongeRecord};
use agg_keccak::tied::{TiedCommitment, TiedPart};
use agg_keccak::witness::SlotBuffers;
use agg_piop::binding::FlockProfile;
use agg_piop::manifest::{CommitmentManifest, Relation};
use agg_piop::pow::KeccakPlan;
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
use rayon::prelude::*;
use sha3::{Digest, Sha3_256};

use crate::wire::{SegmentCommitments, SegmentProofs};

/// The default Flock profile of mode A (event 0; [`AggregatorConfig::profile`]): Ligerito
/// `Slim`, 36% fewer Flock bytes than `Fast` for 3% more prover time at 2,048 records
/// (docs/keccak.md Section 12).
pub const FLOCK_PROFILE: FlockProfile = FlockProfile::Slim;
/// The Flock profiles of mode A: Ligerito `Fast`, `Slim` and `Slim8`, all 128 bits
/// work-normalized in Flock's accounting (docs/keccak.md Section 12).
pub const MODE_A_PROFILES: [FlockProfile; 3] =
    [FlockProfile::Fast, FlockProfile::Slim, FlockProfile::Slim8];
/// The mode-A Flock profiles accepted by a default verifier: Ligerito `Fast` and `Slim`.
/// `Slim8` is opt-in via [`AggregateVerifier::require`] or `--flock-profile slim8`.
pub const DEFAULT_ACCEPTED_PROFILES: [FlockProfile; 2] =
    [FlockProfile::Fast, FlockProfile::Slim];

/// The Ligerito profile of the Flock sub-proofs of a mode-A Flock profile.
#[must_use]
pub const fn keccak_profile(profile: FlockProfile) -> Option<Profile> {
    match profile {
        FlockProfile::Fast => Some(Profile::Fast),
        FlockProfile::Slim => Some(Profile::Slim),
        FlockProfile::Slim8 => Some(Profile::Slim8),
        _ => None,
    }
}

/// The Ligerito profile of a mode-A Flock profile (panics on another profile).
fn ligerito_of(profile: FlockProfile) -> Profile {
    keccak_profile(profile).expect("a mode-A Flock profile (Fast, Slim or Slim8)")
}
/// Default `log2` of the record positions per Flock part. With the tied part (K11) a part
/// of `2^d` records costs a sponge instance of `2^(d + 19)` bits and a bridge instance whose
/// block is `2^(13 + d)` bits; the whole segment (`d = 13`) is the target on the cloud
/// machine, and every local batch (at most 2,048 records) is one part under this default
/// (`docs/cli.md` Section 4, `ref/notes/bench-local.md`).
pub const DEFAULT_PART_LOG2: u32 = 13;
/// Smallest admitted part (`2^3` records, the smallest segment).
pub const MIN_PART_LOG2: u32 = 3;
/// `log2` of the most tied parts one segment may hold: at most two, so `d_part >= d_seg - 1`
/// (docs/soundness.md 3.5). Each part's sponge commitment is bound only to its Johnson list
/// when `sd` is drawn, so `sd`'s bad set is a union over the product of the segment's `P`
/// lists, `L^P theta`; `P <= 2` keeps it below `2^-128` for every mode-A profile.
pub const MAX_PARTS_LOG2: u32 = 1;

/// The smallest admitted `d_part` of a batch: `max(MIN_PART_LOG2, d_seg - MAX_PARTS_LOG2)`
/// (at most two tied parts per segment, [`MAX_PARTS_LOG2`]).
#[must_use]
pub fn min_part_log2(geometry: &Geometry) -> u32 {
    geometry.d_seg().saturating_sub(MAX_PARTS_LOG2).max(MIN_PART_LOG2)
}
/// Record positions per segment (`log2`, spec Section 1).
pub const SEGMENT_LOG2: u32 = 13;
/// Label of the per-part Flock seed.
pub const PART_SEED_LABEL: &[u8] = b"agg-mldsa/v1/flock-part-seed";
/// Label of the Flock seed of a span over all segments.
pub const SPAN_SEED_LABEL: &[u8] = b"agg-mldsa/v1/flock-span-seed";
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

/// The Flock challenger seed of a span over the segments whose seeds are `fs` (event 5):
/// `SHA3-256(label || LE64(S) || fs_0 || .. || fs_{S-1})`.
#[must_use]
pub fn span_seed(fs: &[[u8; 32]]) -> [u8; 32] {
    let mut h = Sha3_256::new();
    h.update((SPAN_SEED_LABEL.len() as u64).to_le_bytes());
    h.update(SPAN_SEED_LABEL);
    h.update((fs.len() as u64).to_le_bytes());
    for f in fs {
        h.update(f);
    }
    h.finalize().into()
}

/// Whether a batch is proven by one span over all its segments: more than one segment, whole
/// segments as parts, and the span option on.
#[must_use]
pub fn spans_segments(geometry: &Geometry, part_log2: u32, span: bool) -> bool {
    spans_with_floor(geometry, part_log2, span, min_part_log2(geometry))
}

fn spans_with_floor(geometry: &Geometry, part_log2: u32, span: bool, floor: u32) -> bool {
    span && geometry.segments() >= 2
        && SegmentParts::plan_with_floor(geometry, 0, part_log2, floor).d_part == geometry.d_seg()
}

/// The Keccak side's part of the proof-of-work plan of a mode-A batch (docs/soundness.md
/// 6.4): one tie site for a span (`S` = the segments), else one per part with live records
/// of every segment in order (`S` = 1), each with the difficulty of its public geometry
/// ([`TiedPart::plan_work`]: the tie site, the `eq_Y` site and the parity-chain rounds).
///
/// # Panics
/// If `profile` is not a mode-A profile.
#[must_use]
pub fn keccak_plan(
    gate: &ChainGate,
    geometry: &Geometry,
    part_log2: u32,
    span: bool,
    profile: FlockProfile,
) -> KeccakPlan {
    keccak_plan_with_floor(gate, geometry, part_log2, span, profile, min_part_log2(geometry))
}

fn keccak_plan_with_floor(
    gate: &ChainGate,
    geometry: &Geometry,
    part_log2: u32,
    span: bool,
    profile: FlockProfile,
    floor: u32,
) -> KeccakPlan {
    let ligerito = ligerito_of(profile);
    let segments = geometry.segments();
    if spans_with_floor(geometry, part_log2, span, floor) {
        let d = geometry.d_seg();
        return KeccakPlan {
            span: u8::try_from(segments).expect("at most 255 segments"),
            d_part: d as u8,
            parts: vec![TiedPart::plan_work(gate, geometry.n, d, segments, ligerito)],
        };
    }
    let d_part = SegmentParts::plan_with_floor(geometry, 0, part_log2, floor).d_part;
    let parts = (0..segments)
        .flat_map(|sigma| {
            let plan = SegmentParts::plan_with_floor(geometry, sigma, part_log2, floor);
            (0..plan.count()).map(move |s| plan.local(s).len())
        })
        .map(|n| TiedPart::plan_work(gate, n, d_part, 1, ligerito))
        .collect();
    KeccakPlan {
        span: 1,
        d_part: d_part as u8,
        parts,
    }
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
    /// `[min_part_log2, d_seg]`, so at most two parts per segment, [`MAX_PARTS_LOG2`]).
    #[must_use]
    pub fn plan(geometry: &Geometry, sigma: usize, part_log2: u32) -> Self {
        Self::plan_with_floor(geometry, sigma, part_log2, min_part_log2(geometry))
    }

    /// [`Self::plan`] clamped to `[floor, d_seg]` instead (test hook: lets a prover build an
    /// inadmissible split that the verifier must refuse).
    fn plan_with_floor(geometry: &Geometry, sigma: usize, part_log2: u32, floor: u32) -> Self {
        Self {
            sigma,
            first: (sigma << SEGMENT_LOG2).min(geometry.n),
            live: geometry.live_in_segment(sigma),
            d_part: part_log2.clamp(floor.max(MIN_PART_LOG2), geometry.d_seg()),
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

/// Builds the Flock witness (sponge slot buffers and the part's lane-major input bits, the
/// parity chain's copy of `b'`) of one part from its records; the honest builder is
/// `TiedPart::honest_parts`. Called at event 2 and again at event 6, so it must be
/// deterministic. Arguments: the part's context, `sigma`, the part index, the part's records.
pub type PartWitness<'a> =
    &'a (dyn Fn(&TiedPart<'_>, usize, usize, &[SpongeRecord]) -> (SlotBuffers, Vec<u64>) + Sync);

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
    /// Heap bytes of the kept commit data (codeword and Merkle tree; 0 when not kept).
    pub kept_bytes: usize,
    /// Event 6: witness rebuilt, share, Flock proof.
    pub prove: Duration,
    /// Proof bytes.
    pub proof_bytes: usize,
}

/// The prover's Keccak side of mode A (module docs).
pub struct FlockProver<'g, 'h> {
    gate: &'g ChainGate,
    profile: FlockProfile,
    span: bool,
    geometry: Geometry,
    records: Vec<SpongeRecord>,
    part_log2: u32,
    /// Smallest `d_part` the prover uses (`min_part_log2` unless a test lowers it).
    part_floor: u32,
    witness: Option<PartWitness<'h>>,
    keep_commit_data: bool,
    state: Mutex<ProverState<'g>>,
}

struct PartState<'g> {
    seg: TiedPart<'g>,
    range: Range<usize>,
    commitment: TiedCommitment,
    commit_data: Option<CommitData>,
}

#[derive(Default)]
struct ProverState<'g> {
    segments: Vec<Vec<PartState<'g>>>,
    /// The span over all segments, when the batch is proven as one (module docs).
    span: Option<PartState<'g>>,
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
            profile: FLOCK_PROFILE,
            span: true,
            geometry,
            records,
            part_log2,
            part_floor: min_part_log2(&geometry),
            witness,
            keep_commit_data: true,
            state: Mutex::new(ProverState::default()),
        }
    }

    /// Prove under the mode-A Flock profile `profile` ([`MODE_A_PROFILES`]; default
    /// [`FLOCK_PROFILE`]).
    ///
    /// # Panics
    /// If `profile` is not a mode-A profile.
    #[must_use]
    pub fn profile(mut self, profile: FlockProfile) -> Self {
        ligerito_of(profile);
        self.profile = profile;
        self
    }

    /// Prove a batch of more than one segment as one span when its parts are whole segments
    /// (default; module docs), or every segment on its own.
    #[must_use]
    pub fn span(mut self, span: bool) -> Self {
        self.span = span;
        self
    }

    /// Lower the smallest `d_part` the prover uses below the admitted `min_part_log2` (test
    /// hook: an aggregate with more than two parts per segment, which verifiers refuse).
    #[doc(hidden)]
    #[must_use]
    pub fn part_floor(mut self, floor: u32) -> Self {
        self.part_floor = floor;
        self
    }

    /// Keep each part's sponge commit data from event 2 to event 6 (default; K14), or drop it
    /// and let Flock commit again at event 6 (K7).
    #[must_use]
    pub fn keep_commit_data(mut self, keep: bool) -> Self {
        self.keep_commit_data = keep;
        self
    }

    /// Per-part wall times and proof sizes (after [`KeccakSide::prove`]).
    #[must_use]
    pub fn log(&self) -> Vec<PartProveLog> {
        self.state.lock().expect("state").log.clone()
    }

    fn build(
        &self,
        seg: &TiedPart<'_>,
        sigma: usize,
        part: usize,
        records: &[SpongeRecord],
    ) -> (SlotBuffers, Vec<u64>) {
        match self.witness {
            Some(f) => f(seg, sigma, part, records),
            None => seg.honest_parts(records),
        }
    }

    /// Event 2 of a span: one tied part over all `segments` (module docs); segment 0's item
    /// holds its cap, the others are covered.
    fn commit_span(&self, state: &mut ProverState<'g>, segments: usize) -> Vec<Vec<u8>> {
        let start = Instant::now();
        let d = self.geometry.d_seg();
        let n = self.records.len();
        let seg = TiedPart::new_span(self.gate, n, d, 0, segments, ligerito_of(self.profile));
        let (sponge, bits) = self.build(&seg, 0, 0, &self.records);
        let mut committed = seg.commit_parts(sponge, bits, &self.records);
        let commitment = committed.commitment().clone();
        let commit_data = committed
            .take_commit_data()
            .filter(|_| self.keep_commit_data);
        drop(committed);
        state.log.push(PartProveLog {
            sigma: 0,
            part: 0,
            records: n,
            commit: start.elapsed(),
            kept_bytes: commit_data.as_ref().map_or(0, CommitData::heap_bytes),
            ..PartProveLog::default()
        });
        let item = |span: u8, commitments: Vec<Vec<u8>>| {
            SegmentCommitments {
                profile: self.profile as u8,
                d_part: d,
                span,
                commitments,
            }
            .to_bytes()
        };
        let mut items = vec![item(
            u8::try_from(segments).expect("at most 255 segments"),
            vec![commitment.to_bytes()],
        )];
        items.extend((1..segments).map(|_| item(0, Vec::new())));
        state.span = Some(PartState {
            seg,
            range: 0..n,
            commitment,
            commit_data,
        });
        items
    }

    /// Event 6 of a span: the proof of every `Y_sigma` in segment 0's item, empty items for
    /// the covered segments.
    ///
    /// # Errors
    /// As [`TiedPart::prove_with_final_vectors`] (an exhausted tie nonce search).
    fn prove_span(
        &self,
        state: &mut ProverState<'g>,
        part: PartState<'g>,
        bridge_seed: &[u8; 32],
        matrices: &BridgeMatrices,
        seeds: &[[u8; 32]],
        final_vectors: &[Vec<u8>],
    ) -> Result<Vec<Vec<u8>>, ProveError> {
        let start = Instant::now();
        let segments = part.seg.segments();
        assert_eq!(seeds.len(), segments, "one seed per segment");
        assert_eq!(final_vectors.len(), segments, "one Y per segment");
        let (sponge, bits) = self.build(&part.seg, 0, 0, &self.records);
        let committed =
            part.seg
                .reattach(sponge, bits, &self.records, part.commitment, part.commit_data);
        let proof = part
            .seg
            .prove_with_final_vectors(
                committed,
                *bridge_seed,
                matrices,
                final_vectors.to_vec(),
                &span_seed(seeds),
            )
            .map_err(|e| ProveError::Keccak(e.to_string()))?;
        let log = &mut state.log[0];
        log.prove = start.elapsed();
        log.proof_bytes = proof.bytes.len();
        let empty = SegmentProofs {
            shares: Vec::new(),
            proofs: Vec::new(),
        };
        let mut items = vec![SegmentProofs {
            shares: Vec::new(),
            proofs: vec![proof.bytes],
        }
        .to_bytes()];
        items.extend((1..segments).map(|_| empty.to_bytes()));
        Ok(items)
    }
}

impl KeccakSide for FlockProver<'_, '_> {
    /// The sites of every tied part (docs/soundness.md 6.4, [`keccak_plan`]).
    fn plan(&self, segments: usize) -> KeccakPlan {
        assert_eq!(segments, self.geometry.segments(), "segments of the batch");
        keccak_plan_with_floor(
            self.gate,
            &self.geometry,
            self.part_log2,
            self.span,
            self.profile,
            self.part_floor,
        )
    }

    fn commitments(&self, segments: usize) -> Vec<Vec<u8>> {
        assert_eq!(segments, self.geometry.segments(), "segments of the batch");
        let mut state = self.state.lock().expect("state");
        state.segments.clear();
        state.span = None;
        state.log.clear();
        if spans_with_floor(&self.geometry, self.part_log2, self.span, self.part_floor) {
            return self.commit_span(&mut state, segments);
        }
        let mut items = Vec::with_capacity(segments);
        for sigma in 0..segments {
            let plan =
                SegmentParts::plan_with_floor(&self.geometry, sigma, self.part_log2, self.part_floor);
            let mut parts = Vec::with_capacity(plan.count());
            for s in 0..plan.count() {
                let start = Instant::now();
                let range = plan.global(s);
                let records = &self.records[range.clone()];
                let seg = TiedPart::new_part(
                    self.gate,
                    records.len(),
                    plan.d_part,
                    s,
                    ligerito_of(self.profile),
                );
                let (sponge, bits) = self.build(&seg, sigma, s, records);
                // The committed buffers are dropped here and rebuilt at event 6; the commit
                // data (codeword, Merkle tree) is kept unless configured otherwise (K14).
                let mut committed = seg.commit_parts(sponge, bits, records);
                let commitment = committed.commitment().clone();
                let commit_data = committed
                    .take_commit_data()
                    .filter(|_| self.keep_commit_data);
                drop(committed);
                state.log.push(PartProveLog {
                    sigma,
                    part: s,
                    records: records.len(),
                    commit: start.elapsed(),
                    kept_bytes: commit_data.as_ref().map_or(0, CommitData::heap_bytes),
                    ..PartProveLog::default()
                });
                parts.push(PartState {
                    seg,
                    range,
                    commitment,
                    commit_data,
                });
            }
            items.push(
                SegmentCommitments {
                    profile: self.profile as u8,
                    d_part: plan.d_part,
                    span: 1,
                    commitments: parts.iter().map(|p| p.commitment.to_bytes()).collect(),
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
    ) -> Result<Vec<Vec<u8>>, ProveError> {
        let mut state = self.state.lock().expect("state");
        if let Some(span) = state.span.take() {
            return self.prove_span(&mut state, span, bridge_seed, matrices, seeds, final_vectors);
        }
        let segments = std::mem::take(&mut state.segments);
        assert_eq!(segments.len(), seeds.len(), "one seed per segment");
        assert_eq!(segments.len(), final_vectors.len(), "one Y per segment");
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
                let (sponge, bits) = self.build(&part.seg, sigma, s, records);
                let committed =
                    part.seg
                        .reattach(sponge, bits, records, part.commitment, part.commit_data);
                let share = if s + 1 < count {
                    let own = part.seg.final_vectors(&committed, matrices).remove(0);
                    item.shares.push(pack_final_vector(&own));
                    acc.iter_mut().zip(&own).for_each(|(a, b)| *a ^= b);
                    own
                } else {
                    acc.iter()
                        .zip(&final_vectors[sigma])
                        .map(|(a, y)| a ^ y)
                        .collect()
                };
                let proof = part
                    .seg
                    .prove_with_final_vectors(
                        committed,
                        *bridge_seed,
                        matrices,
                        vec![share],
                        &part_seed(&seeds[sigma], s),
                    )
                    .map_err(|e| ProveError::Keccak(e.to_string()))?;
                let log = &mut state.log[entry];
                debug_assert_eq!((log.sigma, log.part), (sigma, s));
                log.prove = start.elapsed();
                log.proof_bytes = proof.bytes.len();
                entry += 1;
                item.proofs.push(proof.bytes);
            }
            items.push(item.to_bytes());
        }
        Ok(items)
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
    profile: FlockProfile,
    log: Mutex<Vec<PartVerifyLog>>,
    rejection: Mutex<Option<(usize, SegmentRejection)>>,
    span: Mutex<Option<SpanStash>>,
}

/// What a span's verification needs from the segments it covers, gathered segment by
/// segment until the last one (module docs).
struct SpanStash {
    segments: usize,
    d_part: u32,
    commitment: Vec<u8>,
    proof: Vec<u8>,
    mu: Vec<[u8; 64]>,
    c_tilde: Vec<[u8; C_TILDE_BYTES]>,
    final_vectors: Vec<Vec<u8>>,
    seeds: Vec<[u8; 32]>,
}

impl<'g> FlockVerifier<'g> {
    /// The verifier side over the sponge gate `gate` under the default profile
    /// [`FLOCK_PROFILE`].
    #[must_use]
    pub fn new(gate: &'g ChainGate) -> Self {
        Self::with_profile(gate, FLOCK_PROFILE)
    }

    /// The verifier side under the mode-A Flock profile `profile` ([`MODE_A_PROFILES`]).
    ///
    /// # Panics
    /// If `profile` is not a mode-A profile.
    #[must_use]
    pub fn with_profile(gate: &'g ChainGate, profile: FlockProfile) -> Self {
        ligerito_of(profile);
        Self {
            gate,
            profile,
            log: Mutex::new(Vec::new()),
            rejection: Mutex::new(None),
            span: Mutex::new(None),
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

    fn reject(&self, sigma: usize, why: SegmentRejection) -> bool {
        *self.rejection.lock().expect("rejection") = Some((sigma, why));
        false
    }

    /// Segment `sigma` of a span (`commitments.span != 1`): segment 0 opens it with the
    /// span's cap and proof, every later segment is covered (no commitment, an empty proof
    /// item) and adds its statement, `Y_sigma` and seed; the last one verifies the span.
    fn verify_span(&self, segment: &SegmentContext<'_>, commitments: &SegmentCommitments) -> bool {
        use crate::wire::ItemError;
        let sigma = segment.sigma;
        let total = segment.geometry.segments();
        let mut stash = self.span.lock().expect("span");
        if sigma == 0 {
            let span = usize::from(commitments.span);
            if span != total
                || total < 2
                || commitments.d_part != segment.geometry.d_seg()
                || commitments.commitments.len() != 1
            {
                return self.reject(sigma, SegmentRejection::Item(ItemError::Span(commitments.span)));
            }
            let proofs = match SegmentProofs::from_bytes(segment.proof, 1) {
                Ok(p) => p,
                Err(e) => return self.reject(sigma, SegmentRejection::Item(e)),
            };
            *stash = Some(SpanStash {
                segments: total,
                d_part: commitments.d_part,
                commitment: commitments.commitments[0].clone(),
                proof: proofs.proofs[0].clone(),
                mu: Vec::new(),
                c_tilde: Vec::new(),
                final_vectors: Vec::new(),
                seeds: Vec::new(),
            });
        } else {
            let covered = commitments.span == 0
                && commitments.commitments.is_empty()
                && stash
                    .as_ref()
                    .is_some_and(|st| st.d_part == commitments.d_part && st.seeds.len() == sigma);
            if !covered {
                return self.reject(sigma, SegmentRejection::Item(ItemError::Span(commitments.span)));
            }
            if let Err(e) = SegmentProofs::from_bytes(segment.proof, 0) {
                return self.reject(sigma, SegmentRejection::Item(e));
            }
        }
        let st = stash.as_mut().expect("an open span");
        st.mu.extend_from_slice(segment.mu);
        st.c_tilde.extend_from_slice(segment.c_tilde);
        st.final_vectors.push(segment.final_vector.to_vec());
        st.seeds.push(*segment.seed);
        if sigma + 1 < st.segments {
            return true;
        }
        let st = stash.take().expect("an open span");
        drop(stash);
        let start = Instant::now();
        let n = st.mu.len();
        if n == 0 {
            return self.reject(sigma, SegmentRejection::EmptySegment);
        }
        let seg = TiedPart::new_span(
            self.gate,
            n,
            st.d_part,
            0,
            st.segments,
            ligerito_of(self.profile),
        );
        let setup = start.elapsed();
        let outcome = TiedCommitment::from_bytes(&st.commitment).and_then(|commitment| {
            seg.verify(
                &st.mu,
                &st.c_tilde,
                &commitment,
                *segment.bridge_seed,
                segment.matrices,
                &st.final_vectors,
                &span_seed(&st.seeds),
                &KeccakProof { bytes: st.proof },
            )
        });
        self.log.lock().expect("log").push(PartVerifyLog {
            sigma: 0,
            part: 0,
            records: n,
            setup,
            verify: start.elapsed() - setup,
            accepted: outcome.is_ok(),
        });
        match outcome {
            Ok(()) => true,
            Err(e) => self.reject(sigma, SegmentRejection::Part(0, e.to_string())),
        }
    }
}

impl KeccakVerifier for FlockVerifier<'_> {
    fn profile(&self) -> FlockProfile {
        self.profile
    }

    /// The tie sites of the tied parts the event-2 items name (docs/soundness.md 6.4): one for
    /// a span (segment 0's item opens it), else one per part with live records of every
    /// segment; `None` on an item that does not parse or names an inadmissible geometry
    /// (the per-segment checks of [`Self::verify`] reject it as well).
    fn plan(&self, geometry: &Geometry, commitments: &[Vec<u8>]) -> Option<KeccakPlan> {
        let profile = ligerito_of(self.profile);
        let items: Vec<SegmentCommitments> = commitments
            .iter()
            .map(|c| SegmentCommitments::from_bytes(c).ok())
            .collect::<Option<_>>()?;
        let first = items.first()?;
        if first.span != 1 {
            let segments = usize::from(first.span);
            if segments != geometry.segments() || segments < 2 || first.d_part != geometry.d_seg() {
                return None;
            }
            return Some(KeccakPlan {
                span: first.span,
                d_part: first.d_part as u8,
                parts: vec![TiedPart::plan_work(self.gate, geometry.n, first.d_part, segments, profile)],
            });
        }
        let d_part = first.d_part;
        if d_part < min_part_log2(geometry) || d_part > geometry.d_seg() {
            return None;
        }
        let mut parts = Vec::new();
        for (sigma, item) in items.iter().enumerate() {
            if item.span != 1 || item.d_part != d_part {
                return None;
            }
            let plan = SegmentParts {
                sigma,
                first: 0,
                live: geometry.live_in_segment(sigma),
                d_part,
            };
            parts.extend(
                (0..plan.count()).map(|s| TiedPart::plan_work(self.gate, plan.local(s).len(), d_part, 1, profile)),
            );
        }
        Some(KeccakPlan {
            span: 1,
            d_part: d_part as u8,
            parts,
        })
    }

    fn verify(&self, segment: &SegmentContext<'_>) -> bool {
        let sigma = segment.sigma;
        if sigma == 0 {
            self.log.lock().expect("log").clear();
            *self.rejection.lock().expect("rejection") = None;
            *self.span.lock().expect("span") = None;
        }
        let live = segment.mu.len();
        let commitments = match SegmentCommitments::from_bytes(segment.commitment) {
            Ok(c) => c,
            Err(e) => return self.reject(sigma, SegmentRejection::Item(e)),
        };
        if commitments.profile != self.profile as u8 {
            return self.reject(
                sigma,
                SegmentRejection::Item(crate::wire::ItemError::Profile(commitments.profile)),
            );
        }
        if commitments.span != 1 || self.span.lock().expect("span").is_some() {
            return self.verify_span(segment, &commitments);
        }
        let d_part = commitments.d_part;
        if d_part < min_part_log2(&segment.geometry) || d_part > segment.geometry.d_seg() {
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
        for s in 0..count {
            let range = plan.local(s);
            let start = Instant::now();
            let seg = TiedPart::new_part(
                self.gate,
                range.len(),
                d_part,
                s,
                ligerito_of(self.profile),
            );
            let setup = start.elapsed();
            let commitment = match TiedCommitment::from_bytes(&commitments.commitments[s]) {
                Ok(c) => c,
                Err(e) => return self.reject(sigma, SegmentRejection::Part(s, e.to_string())),
            };
            // Parts 0 .. count - 1 carry their share; the last one's is implied.
            let share = shares.get(s).unwrap_or(&last);
            let outcome = seg.verify(
                &segment.mu[range.clone()],
                &segment.c_tilde[range.clone()],
                &commitment,
                *segment.bridge_seed,
                segment.matrices,
                std::slice::from_ref(share),
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
    /// `[min_part_log2, d_seg]`: at most two parts per segment, [`MAX_PARTS_LOG2`]).
    pub part_log2: u32,
    /// The Flock profile ([`MODE_A_PROFILES`], bound at event 0).
    pub profile: FlockProfile,
    /// Prove a batch of more than one segment as one span over all segments when the parts
    /// are whole segments (module docs, deviation K16; default on).
    pub span: bool,
    /// The relation to prove (`docs/h-exact.md` Section 1; default H-u). The Keccak side
    /// does not depend on it.
    pub relation: Relation,
}

impl Default for AggregatorConfig {
    fn default() -> Self {
        Self {
            part_log2: DEFAULT_PART_LOG2,
            profile: FLOCK_PROFILE,
            span: true,
            relation: Relation::Hu,
        }
    }
}

impl AggregatorConfig {
    /// Parts of `2^{part_log2}` record positions under the default profile.
    #[must_use]
    pub fn with_part_log2(part_log2: u32) -> Self {
        Self {
            part_log2,
            ..Self::default()
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
    /// The configured Flock profile is not a mode-A profile.
    Profile(FlockProfile),
}

impl std::fmt::Display for AggregateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Count(n) => write!(f, "{n} records: an aggregate holds 1 to 2^15"),
            Self::Malformed(i, what) => write!(f, "record {i}: {what}"),
            Self::Signature(i, why) => write!(f, "record {i}: signature rejected ({why})"),
            Self::Prove(e) => write!(f, "prover: {e}"),
            Self::Profile(p) => write!(f, "Flock profile {p:?} is not a mode-A profile"),
        }
    }
}

impl std::error::Error for AggregateError {}

impl From<ProveError> for AggregateError {
    fn from(e: ProveError) -> Self {
        match e {
            ProveError::Tables(SourceTableError::Derive(i, why)) => {
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
    /// The Flock profile (event 0).
    pub profile: FlockProfile,
    /// The relation proven (event 0 and the proof header).
    pub relation: Relation,
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
        ("lookup (T_16)", size.lookup),
        ("PIOP rounds and claims", size.piop),
        ("PIOP work nonces", size.work),
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
        writeln!(
            f,
            "prover, N = {}, Flock profile {:?}, relation {}:",
            self.records, self.profile, self.relation
        )?;
        for (name, d) in [
            ("instance (ExpandA, mu, c, T)", self.instance),
            ("witness and G_main tables", t.tables),
            ("Flock records", self.flock_records),
            ("Akita setup", t.setup),
            ("event 1: G_main16, G_main4 commits", t.round1),
            ("events 2, 6: Flock parts", t.keccak),
            ("events 3-5: matrices, round 2", t.round2),
            ("event 7: lookup into T_16", t.lookup),
            ("events 8, 9: tau, pre-stage", t.prestage),
            ("stage 1", t.stage1),
            ("stage 2", t.stage2),
            ("stage 3", t.stage3),
            ("event 13: Akita opening", t.opening),
        ] {
            writeln!(f, "  {name:<32} {:>10.1} ms", ms(d))?;
        }
        writeln!(f, "  {:<32} {:>10.1} ms", "total", ms(self.total))?;
        for p in &self.parts {
            writeln!(
                f,
                "  Flock part ({}, {}): {} records, commit {:.1} ms, prove {:.1} ms, {} B, \
                 kept commit data {:.1} MiB",
                p.sigma,
                p.part,
                p.records,
                ms(p.commit),
                ms(p.prove),
                p.proof_bytes,
                p.kept_bytes as f64 / (1 << 20) as f64
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
    /// Let the prover split segments into parts below the admitted `min_part_log2` (more
    /// than two parts per segment), which verifiers must refuse.
    pub part_floor: Option<u32>,
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
    keep_commit_data: bool,
}

impl Aggregator {
    /// An aggregator with the given configuration. It keeps each Flock part's sponge commit
    /// data from event 2 to event 6 (K14) unless the environment variable
    /// `AGG_FLOCK_RECOMMIT` is set (a measurement knob) or [`Self::keep_commit_data`] says
    /// otherwise.
    #[must_use]
    pub fn new(config: AggregatorConfig) -> Self {
        Self {
            gate: sponge_gate(),
            config,
            keep_commit_data: std::env::var_os("AGG_FLOCK_RECOMMIT").is_none(),
        }
    }

    /// Keep the Flock parts' sponge commit data between events 2 and 6 (`true`, K14: about
    /// the size of the sponge instance's codeword and Merkle tree per part, measured in
    /// `ref/notes/bench-local.md`) or commit again at event 6 (`false`, K7: one more Ligerito
    /// commit per part).
    #[must_use]
    pub fn keep_commit_data(mut self, keep: bool) -> Self {
        self.keep_commit_data = keep;
        self
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
        let profile = self.config.profile;
        if keccak_profile(profile).is_none() {
            return Err(AggregateError::Profile(profile));
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
            relation: self.config.relation,
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
        )
        .profile(profile)
        .span(self.config.span)
        .keep_commit_data(self.keep_commit_data);
        let side = match hooks.part_floor {
            Some(floor) => side.part_floor(floor),
            None => side,
        };
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
                profile,
                &prove_hooks,
            )?
        } else {
            prove_with_tables(
                &prover_inputs,
                &manifest,
                main,
                &SeededMatrices,
                &side,
                profile,
                &prove_hooks,
            )?
        };
        output.timings.tables = tables_time;
        output.timings.total += tables_time;
        let bytes = output.proof.to_bytes();
        let size = output.proof.size();
        debug_assert_eq!(size.total(), bytes.len());
        let report = ProveReport {
            relation: self.config.relation,
            records: n,
            profile,
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
    /// The Flock profile the aggregate was verified under.
    pub profile: FlockProfile,
    /// The relation the aggregate was verified under (the verifier's, never the proof's).
    pub relation: Relation,
    /// Reading the profile from the aggregate's event-2 items (a parse of the aggregate,
    /// outside the PIOP verifier's steps; zero when the profile was given).
    pub profile_probe: Duration,
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
            "verifier, N = {}, {} keys, aggregate {} B, Flock profile {:?}, relation {}: {verdict}",
            self.records,
            if self.cached { "cached" } else { "uncached" },
            self.aggregate_bytes,
            self.profile,
            self.relation
        )?;
        writeln!(
            f,
            "  {:<16} {:<32} {:>10.2} ms",
            "profile",
            "Flock profile from the items",
            ms(self.profile_probe)
        )?;
        let labels = [
            ("parse", "parse, manifest, PCS verifier"),
            ("public_data", "pkDecode, tr, mu, SampleInBall"),
            ("transcript", "events 0-6 replay"),
            ("matrices", "bridge matrices from sd"),
            ("keccak", "Flock parts"),
            ("lookup", "lookup into T_16"),
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

/// The mode-A verifier (holds the sponge gate, built once). It verifies an aggregate under
/// the Flock profile its event-2 items name when that is a default-accepted mode-A profile
/// ([`DEFAULT_ACCEPTED_PROFILES`]: `Fast` or `Slim`; the transcript binds the profile at
/// event 0, so the items cannot name another one than the prover bound), or under the profile
/// fixed by [`Self::require`] (which also admits `Slim8`).
pub struct AggregateVerifier {
    gate: ChainGate,
    require: Option<FlockProfile>,
    relation: Relation,
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
            require: None,
            relation: Relation::Hu,
        }
    }

    /// Verify under `relation` (default H-u). The relation is the verifier's choice: an
    /// aggregate whose header names another relation is rejected (`docs/h-exact.md` 1.3).
    #[must_use]
    pub const fn relation(mut self, relation: Relation) -> Self {
        self.relation = relation;
        self
    }

    /// Verify under `profile` only (an aggregate under another profile is rejected).
    ///
    /// # Panics
    /// If `profile` is not a mode-A profile.
    #[must_use]
    pub fn require(mut self, profile: FlockProfile) -> Self {
        ligerito_of(profile);
        self.require = Some(profile);
        self
    }

    /// The profile to verify `aggregate` under: the required one, else the default-accepted
    /// profile its event-2 items name, else the default (an unparsable aggregate is rejected later).
    fn profile_of(&self, aggregate: &[u8]) -> FlockProfile {
        if let Some(p) = self.require {
            return p;
        }
        AggregateProof::from_bytes(aggregate)
            .ok()
            .and_then(|proof| crate::wire::flock_profile(&proof.flock_commitments))
            .and_then(FlockProfile::from_id)
            .filter(|p| DEFAULT_ACCEPTED_PROFILES.contains(p))
            .unwrap_or(FLOCK_PROFILE)
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
        let probe = Instant::now();
        let profile = self.profile_of(aggregate);
        let profile_probe = if self.require.is_some() {
            Duration::ZERO
        } else {
            probe.elapsed()
        };
        let keccak = FlockVerifier::with_profile(&self.gate, profile);
        let cached = cache.is_some();
        let report = verify_timed(
            statements,
            aggregate,
            &SeededMatrices,
            &keccak,
            profile,
            self.relation,
            cache,
        );
        VerifyCostReport {
            report,
            profile,
            relation: self.relation,
            profile_probe,
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
                    part_log2.clamp(min_part_log2(&geometry), geometry.d_seg())
                );
                assert!(plan.count() <= 1 << MAX_PARTS_LOG2, "at most two parts");
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
                profile: FLOCK_PROFILE as u8,
                d_part,
                span: 1,
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
        let empty = item(12, Vec::new());
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
        assert!(!check(&item(12, vec![vec![1]]), &zero, &[]));
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
        // At most two parts per segment: `d_part >= d_seg - 1 = 12`.
        for d_part in [2, 3, 10, 11, 14] {
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

    /// The span glue of the verifier (deviation K16) on the geometry of `N = 2^14 + 1` (four
    /// segments, the last empty), without proofs: segment 0's item opens a span over all four
    /// segments, segments 1 to 3 are covered and add their statements, and the last one runs
    /// the span's verification (garbage proof bytes here: rejected as part 0 at segment 3,
    /// with all `N` records). A span over fewer segments, a covered item at segment 0, a
    /// covered item with a commitment, a per-segment item inside a span, a second span item
    /// and a covered segment with a proof are rejected.
    #[test]
    fn span_items_open_gather_and_close_at_the_last_segment() {
        let gate = sponge_gate();
        let verifier = FlockVerifier::new(&gate);
        let geometry = Geometry::new((1 << 14) + 1).unwrap();
        assert_eq!(geometry.segments(), 4);
        assert!(spans_segments(&geometry, DEFAULT_PART_LOG2, true));
        assert!(!spans_segments(&geometry, DEFAULT_PART_LOG2, false));
        assert!(!spans_segments(&geometry, 12, true));
        assert!(!spans_segments(&Geometry::new(8192).unwrap(), DEFAULT_PART_LOG2, true));
        let matrices = BridgeMatrices::random_for_tests(3);
        let item = |span: u8, commitments: Vec<Vec<u8>>| {
            SegmentCommitments {
                profile: FLOCK_PROFILE as u8,
                d_part: 13,
                span,
                commitments,
            }
            .to_bytes()
        };
        let open = item(4, vec![vec![0; 8]]);
        let open_proof = SegmentProofs {
            shares: Vec::new(),
            proofs: vec![vec![1, 2, 3]],
        }
        .to_bytes();
        let covered = item(0, Vec::new());
        let zero = vec![0_u8; Y_BITS];
        let mu = vec![[0_u8; 64]; 1 << 13];
        let c_tilde = vec![[0_u8; C_TILDE_BYTES]; 1 << 13];
        let live = [1 << 13, 1 << 13, 1, 0];
        let check = |sigma: usize, commitment: &[u8], proof: &[u8]| {
            verifier.verify(&SegmentContext {
                sigma,
                geometry,
                commitment,
                seed: &[sigma as u8; 32],
                bridge_seed: &[0; 32],
                mu: &mu[..live[sigma]],
                c_tilde: &c_tilde[..live[sigma]],
                final_vector: &zero,
                proof,
                matrices: &matrices,
            })
        };
        let span_rejection = |b: u8| {
            matches!(
                verifier.rejection(),
                Some((_, SegmentRejection::Item(crate::wire::ItemError::Span(x)))) if x == b
            )
        };
        assert!(check(0, &open, &open_proof));
        assert!(check(1, &covered, &[]));
        assert!(check(2, &covered, &[]));
        assert!(verifier.log().is_empty(), "nothing verified before the last segment");
        assert!(!check(3, &covered, &[]));
        assert!(matches!(
            verifier.rejection(),
            Some((3, SegmentRejection::Part(0, _)))
        ));
        let log = verifier.log();
        assert_eq!((log.len(), log[0].records), (1, (1 << 14) + 1));
        // A span over three of the four segments; a covered item at segment 0.
        assert!(!check(0, &item(3, vec![vec![0; 8]]), &open_proof));
        assert!(span_rejection(3));
        assert!(!check(0, &covered, &[]));
        assert!(span_rejection(0));
        // Inside an open span: a covered item with a commitment, a per-segment item, a second
        // span item, a covered segment with a proof.
        for (bad, proof, b) in [
            (item(0, vec![vec![1]]), Vec::new(), Some(0)),
            (item(1, Vec::new()), Vec::new(), Some(1)),
            (open.clone(), Vec::new(), Some(4)),
            (covered.clone(), vec![0, 0, 0, 0], None),
        ] {
            assert!(check(0, &open, &open_proof));
            assert!(!check(1, &bad, &proof));
            match b {
                Some(b) => assert!(span_rejection(b)),
                None => assert!(matches!(
                    verifier.rejection(),
                    Some((1, SegmentRejection::Item(crate::wire::ItemError::Trailing)))
                )),
            }
        }
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
    fn plan_digests_reproduce_the_soundness_doc_vectors() {
        // docs/soundness.md Section 10 (python3 -I docs/scripts/pow_plan.py), N = N*, default
        // part size: the plan of the code's own Keccak side and PIOP registry.
        use agg_piop::pow::PowPlan;
        let gate = sponge_gate();
        let hex = |d: [u8; 32]| d.iter().map(|b| format!("{b:02x}")).collect::<String>();
        for (d_sig, profile, span, want) in [
            (3, FlockProfile::Slim, true, "a56daef1067736a1465e21e1099da13ad3dc355f90594e4f401251071f845328"),
            (10, FlockProfile::Slim, true, "11b035262a5b998594b795e346241abec1b953a14cb4b8afa6d52a444634c960"),
            (13, FlockProfile::Slim, true, "ca232ca0195b87274440b1745808795c896ef4ceaa4e39d32c08f137f54d75db"),
            (15, FlockProfile::Fast, true, "98ed5d446d68fe2416580acc5469c29c90d53e1d88937a6c381773b89aa98d00"),
            (15, FlockProfile::Slim, true, "911303030ad4a06781f89ceff4f373ffdbfe6da1946ae54c30a6a965b3c28cb2"),
            (15, FlockProfile::Slim, false, "7aa378b470947b1f64a8ae049c4c3741f4ebb493583c305e552750f448f79896"),
            (15, FlockProfile::Slim8, true, "1467d650376c39e30bde4f7ad9d12ecee8c4f5b3f580ae81ea7364e840b8d8d8"),
        ] {
            let n = 1 << d_sig;
            let geometry = Geometry::new(n).unwrap();
            let keccak = keccak_plan(&gate, &geometry, DEFAULT_PART_LOG2, span, profile);
            let manifest = CommitmentManifest::for_batch(n).unwrap();
            let plan = PowPlan::new(&manifest, profile, &keccak);
            assert_eq!(hex(plan.digest()), want, "2^{d_sig} {profile:?} span {span}");
        }
        // Parts of 2^7 at N = 256: two tied parts (d = 7, nu_s = 8, Slim): tie 50 x 11 = 550
        // (10 bits), eq_Y 50 x 12 = 600 (10 bits), 20 rounds of 2 x 50 = 100 (7 bits).
        let keccak = keccak_plan(&gate, &Geometry::new(256).unwrap(), 7, true, FlockProfile::Slim);
        let work = agg_bridge::pow::PartWork {
            tie_bits: 10,
            eq_bits: 10,
            round_bits: 7,
            rounds: 20,
        };
        assert_eq!((keccak.span, keccak.d_part, keccak.parts), (1, 7, vec![work, work]));
    }

    #[test]
    fn part_seeds_are_distinct_per_part_and_segment_seed() {
        let a = part_seed(&[1; 32], 0);
        assert_ne!(a, part_seed(&[1; 32], 1));
        assert_ne!(a, part_seed(&[2; 32], 0));
        assert_eq!(a, part_seed(&[1; 32], 0));
        // The span seed depends on every segment seed and their order.
        let s = span_seed(&[[1; 32], [2; 32]]);
        assert_ne!(s, span_seed(&[[2; 32], [1; 32]]));
        assert_ne!(s, span_seed(&[[1; 32], [2; 32], [0; 32]]));
        assert_ne!(s, part_seed(&[1; 32], 0));
    }
}
