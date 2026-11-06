//! Mode A end to end: the aggregator turns records `(pk_i, M_i, signature_i)` into an
//! aggregate, the list of `c~_i` and one proof holding the algebraic PIOP (`agg-piop`), the
//! Flock Keccak sub-proofs (`agg-keccak`) and the bridge data; the verifier takes `(pk_i, M_i)`
//! and the aggregate and returns accept or reject with the cost of every step.
//!
//! Transcript (one Spongefish stream run by `agg_piop::prover`): the instance record (record
//! digest over `(pk_i, M_i, c~_i)`, the Flock profile: `Fast`, `Slim` or `Slim8`,
//! [`AggregatorConfig::profile`]), then the round-1 Akita commitment `G_main`, the Flock
//! commitments, the bridge seed `sd`, the round-2 Akita commitments, the Flock seeds
//! `fs_sigma`, the final vectors `Y_sigma` and the part shares, the lookup and the three
//! sum-check stages, then the batched Akita opening. The Flock proofs are never absorbed;
//! they are proven on their own thread pool while the prover runs the lookup, the stages and
//! the opening. There is no projection seed: the residual is proven through the committed
//! quotient.
//!
//! Keccak side ([`FlockProver`], [`FlockVerifier`]): segment `sigma` (`2^13` record positions)
//! is split into parts of `2^{d_part}` record positions, `d_part = clamp(part_log2,
//! min_part_log2, d_seg)` (default [`DEFAULT_PART_LOG2`]; at most two parts per segment,
//! [`MAX_PARTS_LOG2`]); each part with live records is one tied part (`agg_keccak::tied`,
//! tied-part proof version 5): a Flock sponge instance of its records whose opening also proves
//! the bit claim of the virtual parity chain, a sum-check that reduces the part's share
//! `Y_{sigma, s} = M^{(sigma)} b'_s` to that claim. The shares XOR to
//! `Y_sigma` (the last share is implied), which the F_p side proves for the committed bits `b`.
//! The commitment-phase item carries one cap per part (the sponge instance's), the share item the
//! shares, the join's item one proof per part ([`crate::wire`]). Part `s` seeds its Flock
//! challengers with [`part_seed`]`(fs_sigma, s, rec)`, `rec` the digest of the final-vector record
//! (`agg_piop::transcript::flock_record_digest`).
//!
//! Span ([`AggregatorConfig::span`], the default): a batch of more than one segment whose parts
//! are whole segments is proven by ONE tied part over all its segments
//! (`agg_keccak::tied::TiedPart::new_span`): one sponge instance of all `N` records and one
//! parity chain over every segment's bits, whose claim `<eq_Y, Y>` covers every `Y_sigma`, so
//! the Flock opening is paid once per batch instead of once per segment. Segment 0's items
//! carry the span's cap and proof, the other segments' items say "covered by segment 0"
//! ([`crate::wire`]); the verifier checks the span when it reaches the last segment, seeded by
//! [`span_seed`]`(fs_0, .., fs_{S-1}, rec)`.
//!
//! Memory: the parts are committed one after the other at the commitment phase; each keeps its commitment
//! and the sponge commitment's Flock prover data (its Reed-Solomon codeword and Merkle tree),
//! so that the proof opens it without committing again. When the batch has exactly one tied
//! part (a span, or one part in a one-segment batch), its committed witness (sponge buffers,
//! input bits, statement) is kept from the commitment phase until the join and proven without a rebuild.
//! With more than one part the witness buffers are dropped after the commitment phase and rebuilt
//! from the records for the proof, proven and dropped, so at most one part's Flock buffers are
//! alive (the input bits of a part whose share is sent are kept, `6 * 2^(d + 7)` bytes, to
//! compute the share at the final-vector step). The PIOP tables are `i8` columns (`agg_piop::source_tables`).

use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use agg_keccak::flock_union::CommitData;
use agg_keccak::gate::ChainGate;
use agg_keccak::prover::{KeccakProof, Profile};
use agg_keccak::prover::KeccakError;
use agg_keccak::sponge::{sponge_gate, sponge_gate_for, SpongeRecord};
use agg_keccak::tied::{TiedCommitment, TiedCommitted, TiedPart};
use agg_keccak::witness::SlotBuffers;
use agg_piop::binding::FlockProfile;
use agg_piop::manifest::{CommitmentManifest, Relation};
use agg_piop::pow::KeccakPlan;
use agg_piop::lane_public::{LaneKeyCache, LanePublicData};
use agg_piop::prover::{
    main_tables, main_tables_for, prove_lanes_with_tables_using, prove_with_tables_using,
    BridgeDerivation, BridgeTamper, KeccakSide, ProveError, ProveHooks, ProveTimings,
    ProverInputs, SeededMatrices,
};
use agg_piop::public_data::{KeyCache, PublicData};
use agg_piop::source_tables::{BridgeGroupTables, MainTables, SourceTableError};
use agg_piop::verifier::{
    verify_lanes_timed, verify_timed, KeccakVerifier, SegmentContext, VerifyError, VerifyReport,
    VerifyTimings,
};
use agg_piop::wire::{pack_final_vector, unpack_final_vector, AggregateProof, ProofSize};
use agg_witness::bridge::BridgeMatrices;
use agg_witness::instance::{InstanceError, PublicInstance};
use agg_witness::padding::{main_index, Geometry};
use agg_witness::params::{
    ParamSet, Set, Set44, Set65, Set87, C_TILDE_BYTES, K, N, PK_BYTES, SIG_BYTES, W1_ENCODE_BYTES,
};
use rayon::prelude::*;
use sha3::{Digest, Sha3_256};

use crate::wire::{SegmentCommitments, SegmentProofs, SegmentShares};

/// The default Flock profile of mode A (the instance record; [`AggregatorConfig::profile`]): Ligerito
/// `Fast`, the measured best at 32K (2.97 s on the general build, 2.76 s on the IFMA build
/// at 16C32T for 49.2 B/sig); `Slim` saves 36% of the Flock bytes for 3% more prover time
/// at 2,048 records.
pub const FLOCK_PROFILE: FlockProfile = FlockProfile::Fast;
/// The Flock profiles of mode A: Ligerito `Fast`, `Slim` and `Slim8`, all 128 bits
/// work-normalized in Flock's accounting.
pub const MODE_A_PROFILES: [FlockProfile; 3] =
    [FlockProfile::Fast, FlockProfile::Slim, FlockProfile::Slim8];
/// The mode-A Flock profiles accepted by a default verifier: Ligerito `Fast` only, the one
/// profile whose every site meets 2^-128. `Slim` (its 15 outer zero-check rounds reach
/// 2^-127.8 once the constant-column term is charged) and `Slim8` are opt-in via
/// [`AggregateVerifier::require`] or `--flock-profile`.
pub const DEFAULT_ACCEPTED_PROFILES: [FlockProfile; 1] = [FlockProfile::Fast];

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
/// Default `log2` of the record positions per Flock part. A part of `2^d` records costs a
/// sponge instance of `2^(d + 19)` bits; the whole segment (`d = 13`) is the target on the
/// cloud machine, and every local batch (at most 2,048 records) is one part under this
/// default.
pub const DEFAULT_PART_LOG2: u32 = 13;
/// Smallest admitted part (`2^3` records, the smallest segment).
pub const MIN_PART_LOG2: u32 = 3;
/// `log2` of the most tied parts one segment may hold: at most two, so `d_part >= d_seg - 1`.
/// Each part's sponge commitment is bound only to its Johnson list
/// when `sd` is drawn, so `sd`'s bad set is a union over the product of the segment's `P`
/// lists, `L^P theta`; `P <= 2` keeps it below `2^-128` for every mode-A profile.
pub const MAX_PARTS_LOG2: u32 = 1;

/// The smallest admitted `d_part` of a batch: `max(MIN_PART_LOG2, d_seg - MAX_PARTS_LOG2)`
/// (at most two tied parts per segment, [`MAX_PARTS_LOG2`]).
#[must_use]
pub fn min_part_log2(geometry: &Geometry) -> u32 {
    geometry.d_seg().saturating_sub(MAX_PARTS_LOG2).max(MIN_PART_LOG2)
}
/// Record positions per segment (`log2`).
pub const SEGMENT_LOG2: u32 = 13;
/// Label of the per-part Flock seed.
pub const PART_SEED_LABEL: &[u8] = b"agg-mldsa/v1/flock-part-seed";
/// Label of the Flock seed of a span over all segments.
pub const SPAN_SEED_LABEL: &[u8] = b"agg-mldsa/v1/flock-span-seed";
/// Bits of a final vector or share.
#[cfg(test)]
const Y_BITS: usize = 4096;

/// The Flock challenger seed of part `part` of a segment whose seed is `fs` (the Flock seed step), bound
/// to the digest `record` of the final-vector record (`agg_piop::transcript::flock_record_digest`):
/// `SHA3-256(label || fs || LE64(part) || record)`. Both challengers of the part (tie and
/// sponge) absorb it first, so every Flock challenge address encodes the input of the
/// lookup challenge `X`.
#[must_use]
pub fn part_seed(fs: &[u8; 32], part: usize, record: &[u8; 32]) -> [u8; 32] {
    let mut h = Sha3_256::new();
    h.update((PART_SEED_LABEL.len() as u64).to_le_bytes());
    h.update(PART_SEED_LABEL);
    h.update(fs);
    h.update((part as u64).to_le_bytes());
    h.update(record);
    h.finalize().into()
}

/// The Flock challenger seed of a span over the segments whose seeds are `fs` (the Flock seed step),
/// bound to the final-vector record digest `record` as [`part_seed`]:
/// `SHA3-256(label || LE64(S) || fs_0 || .. || fs_{S-1} || record)`.
#[must_use]
pub fn span_seed(fs: &[[u8; 32]], record: &[u8; 32]) -> [u8; 32] {
    let mut h = Sha3_256::new();
    h.update((SPAN_SEED_LABEL.len() as u64).to_le_bytes());
    h.update(SPAN_SEED_LABEL);
    h.update((fs.len() as u64).to_le_bytes());
    for f in fs {
        h.update(f);
    }
    h.update(record);
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

/// The Keccak side's part of the proof-of-work plan of a mode-A batch:
/// one tie site for a span (`S` = the segments), else one per part with live records
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
/// `TiedPart::honest_parts`. Called at the commitment phase and again for the proof (unless the batch has
/// one part, whose witness is kept), so it must be deterministic. Arguments: the part's context, `sigma`, the part index, the part's records.
pub type PartWitness<'a, P = Set44> =
    &'a (dyn Fn(&TiedPart<'_>, usize, usize, &[SpongeRecord<P>]) -> (SlotBuffers, Vec<u64>) + Sync);

/// Wall time and size of one Flock part on the prover side.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PartProveLog {
    /// Segment.
    pub sigma: usize,
    /// Part index.
    pub part: usize,
    /// Live records.
    pub records: usize,
    /// The commitment phase: context, witness and commitment.
    pub commit: Duration,
    /// Heap bytes of the kept commit data (codeword and Merkle tree; 0 when not kept).
    pub kept_bytes: usize,
    /// The join's proof (witness rebuilt unless kept, Flock proof; the share is computed at
    /// the final-vector step).
    pub prove: Duration,
    /// Proof bytes.
    pub proof_bytes: usize,
}

/// The prover's Keccak side of mode A (module docs) for the records of parameter set `P`.
pub struct FlockProver<'g, 'h, P: ParamSet = Set44> {
    gate: &'g ChainGate,
    profile: FlockProfile,
    span: bool,
    geometry: Geometry,
    records: Vec<SpongeRecord<P>>,
    part_log2: u32,
    /// Smallest `d_part` the prover uses (`min_part_log2` unless a test lowers it).
    part_floor: u32,
    witness: Option<PartWitness<'h, P>>,
    state: Mutex<ProverState<'g>>,
}

struct PartState<'g> {
    seg: TiedPart<'g>,
    range: Range<usize>,
    commitment: TiedCommitment,
    commit_data: Option<CommitData>,
    /// The committed witness, kept from the commitment phase for the join when the batch has exactly one
    /// part; `None`: rebuilt for the proof.
    committed: Option<TiedCommitted>,
    /// The part's input bits when its share is sent (every part of a segment but the last):
    /// the share is computed from them at the final-vector step, before the proofs.
    share_bits: Option<Vec<u64>>,
}

#[derive(Default)]
struct ProverState<'g> {
    segments: Vec<Vec<PartState<'g>>>,
    /// The span over all segments, when the batch is proven as one (module docs).
    span: Option<PartState<'g>>,
    /// Per segment, the shares of its parts but the last (one bit per byte), from
    /// [`KeccakSide::shares`].
    shares: Vec<Vec<Vec<u8>>>,
    log: Vec<PartProveLog>,
}

impl<'g, 'h, P: ParamSet> FlockProver<'g, 'h, P> {
    /// The Keccak side for the live records `(mu_i, w1Encode(w_1'_i))` of a batch.
    ///
    /// # Panics
    /// If the record count does not match the geometry.
    #[must_use]
    pub fn new(
        gate: &'g ChainGate,
        geometry: Geometry,
        records: Vec<SpongeRecord<P>>,
        part_log2: u32,
        witness: Option<PartWitness<'h, P>>,
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

    /// The committed witness and commit data to keep from the commitment phase (module docs): the whole
    /// witness if `keep` (one part in the batch), otherwise the commit data alone.
    fn split_committed(
        mut committed: TiedCommitted,
        keep: bool,
    ) -> (TiedCommitment, Option<CommitData>, Option<TiedCommitted>) {
        let commitment = committed.commitment().clone();
        if keep {
            (commitment, None, Some(committed))
        } else {
            let commit_data = committed.take_commit_data();
            // A one-thread unmap of the part's witness: off the critical path.
            agg_pcs_config::sched::drop_in_background("event-2 witness", committed);
            (commitment, commit_data, None)
        }
    }

    /// The part's committed witness for its proof: the one kept at the commitment phase, or rebuilt from
    /// the records with the commitment (and commit data) of the commitment phase.
    fn committed_for_proof(
        &self,
        seg: &TiedPart<'_>,
        (sigma, s): (usize, usize),
        records: &[SpongeRecord<P>],
        commitment: TiedCommitment,
        commit_data: Option<CommitData>,
        kept: Option<TiedCommitted>,
    ) -> TiedCommitted {
        if let Some(committed) = kept {
            debug_assert!(
                *committed.commitment() == commitment,
                "kept witness of another commitment"
            );
            return committed;
        }
        let (sponge, bits) = self.build(seg, sigma, s, records);
        seg.reattach(sponge, bits, records, commitment, commit_data)
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
        records: &[SpongeRecord<P>],
    ) -> (SlotBuffers, Vec<u64>) {
        match self.witness {
            Some(f) => f(seg, sigma, part, records),
            None => seg.honest_parts(records),
        }
    }

    /// The commitment phase of a span: one tied part over all `segments` (module docs); segment 0's item
    /// holds its cap, the others are covered.
    fn commit_span(&self, state: &mut ProverState<'g>, segments: usize) -> Vec<Vec<u8>> {
        let start = Instant::now();
        let d = self.geometry.d_seg();
        let n = self.records.len();
        let seg = TiedPart::new_span(self.gate, n, d, 0, segments, ligerito_of(self.profile));
        let (sponge, bits) = self.build(&seg, 0, 0, &self.records);
        // One part in the batch: its witness is kept for the join.
        let (commitment, commit_data, committed) =
            Self::split_committed(seg.commit_parts(sponge, bits, &self.records), true);
        let kept_bytes = committed
            .as_ref()
            .and_then(TiedCommitted::commit_data)
            .or(commit_data.as_ref())
            .map_or(0, CommitData::heap_bytes);
        state.log.push(PartProveLog {
            sigma: 0,
            part: 0,
            records: n,
            commit: start.elapsed(),
            kept_bytes,
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
            committed,
            share_bits: None,
        });
        items
    }

    /// The join of a span: the proof of every `Y_sigma` in segment 0's item, empty items for
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
        record: &[u8; 32],
    ) -> Result<Vec<Vec<u8>>, ProveError> {
        let start = Instant::now();
        let segments = part.seg.segments();
        assert_eq!(seeds.len(), segments, "one seed per segment");
        assert_eq!(final_vectors.len(), segments, "one Y per segment");
        let committed = self.committed_for_proof(
            &part.seg,
            (0, 0),
            &self.records,
            part.commitment,
            part.commit_data,
            part.committed,
        );
        let proof = part
            .seg
            .prove_with_final_vectors(
                committed,
                *bridge_seed,
                matrices,
                final_vectors.to_vec(),
                &span_seed(seeds, record),
            )
            .map_err(|e| ProveError::Keccak(e.to_string()))?;
        let log = &mut state.log[0];
        log.prove = start.elapsed();
        log.proof_bytes = proof.bytes.len();
        let empty = SegmentProofs { proofs: Vec::new() };
        let mut items = vec![SegmentProofs {
            proofs: vec![proof.bytes],
        }
        .to_bytes()];
        items.extend((1..segments).map(|_| empty.to_bytes()));
        Ok(items)
    }
}

impl<P: ParamSet> KeccakSide for FlockProver<'_, '_, P> {
    /// The sites of every tied part ([`keccak_plan`]).
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
        state.shares.clear();
        state.log.clear();
        if spans_with_floor(&self.geometry, self.part_log2, self.span, self.part_floor) {
            return self.commit_span(&mut state, segments);
        }
        // A batch of exactly one part keeps its witness for the join; with more
        // parts at most one part's buffers are alive (module docs).
        let total: usize = (0..segments)
            .map(|sigma| {
                let floor = self.part_floor;
                SegmentParts::plan_with_floor(&self.geometry, sigma, self.part_log2, floor).count()
            })
            .sum();
        let keep = total == 1;
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
                let share_bits = (s + 1 < plan.count()).then(|| bits.clone());
                // Unless kept (one part in the batch), the committed buffers are
                // dropped here and rebuilt for the proof; the commit data (codeword, Merkle
                // tree) is kept.
                let (commitment, commit_data, committed) =
                    Self::split_committed(seg.commit_parts(sponge, bits, records), keep);
                let kept_bytes = committed
                    .as_ref()
                    .and_then(TiedCommitted::commit_data)
                    .or(commit_data.as_ref())
                    .map_or(0, CommitData::heap_bytes);
                state.log.push(PartProveLog {
                    sigma,
                    part: s,
                    records: records.len(),
                    commit: start.elapsed(),
                    kept_bytes,
                    ..PartProveLog::default()
                });
                parts.push(PartState {
                    seg,
                    range,
                    commitment,
                    commit_data,
                    committed,
                    share_bits,
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

    /// The shares of every segment's parts but the last, under the matrices of `sd`,
    /// from the input bits kept at the commitment phase (empty items for a span and one-part segments).
    fn shares(
        &self,
        matrices: &BridgeMatrices,
        final_vectors: &[Vec<u8>],
    ) -> Result<Vec<Vec<u8>>, ProveError> {
        let mut state = self.state.lock().expect("state");
        let segments = final_vectors.len();
        if state.span.is_some() {
            return Ok(vec![Vec::new(); segments]);
        }
        assert_eq!(state.segments.len(), segments, "one Y per segment");
        let shares: Vec<Vec<Vec<u8>>> = state
            .segments
            .iter_mut()
            .map(|parts| {
                parts
                    .iter_mut()
                    .filter_map(|part| {
                        let bits = part.share_bits.take()?;
                        Some(part.seg.final_vectors_of_bits(&bits, matrices).remove(0))
                    })
                    .collect()
            })
            .collect();
        let items = shares
            .iter()
            .map(|own| {
                SegmentShares {
                    shares: own.iter().map(|bits| pack_final_vector(bits)).collect(),
                }
                .to_bytes()
            })
            .collect();
        state.shares = shares;
        Ok(items)
    }

    fn prove(
        &self,
        bridge_seed: &[u8; 32],
        matrices: &BridgeMatrices,
        seeds: &[[u8; 32]],
        final_vectors: &[Vec<u8>],
        record: &[u8; 32],
    ) -> Result<Vec<Vec<u8>>, ProveError> {
        let mut state = self.state.lock().expect("state");
        if let Some(span) = state.span.take() {
            return self.prove_span(
                &mut state,
                span,
                bridge_seed,
                matrices,
                seeds,
                final_vectors,
                record,
            );
        }
        let segments = std::mem::take(&mut state.segments);
        let shares = std::mem::take(&mut state.shares);
        assert_eq!(segments.len(), seeds.len(), "one seed per segment");
        assert_eq!(segments.len(), final_vectors.len(), "one Y per segment");
        assert_eq!(segments.len(), shares.len(), "the shares of event 6 first");
        let mut entry = 0;
        let mut items = Vec::with_capacity(segments.len());
        for (sigma, (parts, own)) in segments.into_iter().zip(shares).enumerate() {
            let count = parts.len();
            assert_eq!(own.len(), count.saturating_sub(1), "one share per part but the last");
            // The last part's share is Y_sigma XOR the others.
            let mut last = final_vectors[sigma].clone();
            for share in &own {
                last.iter_mut().zip(share).for_each(|(a, b)| *a ^= b);
            }
            let mut own = own.into_iter();
            let mut item = SegmentProofs {
                proofs: Vec::with_capacity(count),
            };
            for (s, part) in parts.into_iter().enumerate() {
                let start = Instant::now();
                let records = &self.records[part.range.clone()];
                let committed = self.committed_for_proof(
                    &part.seg,
                    (sigma, s),
                    records,
                    part.commitment,
                    part.commit_data,
                    part.committed,
                );
                let share = if s + 1 < count {
                    own.next().expect("a share per part but the last")
                } else {
                    std::mem::take(&mut last)
                };
                let proof = part
                    .seg
                    .prove_with_final_vectors(
                        committed,
                        *bridge_seed,
                        matrices,
                        vec![share],
                        &part_seed(&seeds[sigma], s, record),
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
    /// The parameter set of the gate (the width of `c~`).
    set: Set,
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
    record: [u8; 32],
    mu: Vec<[u8; 64]>,
    /// The `c~` of every covered record, concatenated.
    c_tilde: Vec<u8>,
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
        Self::for_set(gate, Set::S44, profile)
    }

    /// The verifier side of parameter set `set` (`gate` is that set's sponge gate,
    /// `agg_keccak::sponge::sponge_gate_for`) under the mode-A Flock profile `profile`.
    ///
    /// # Panics
    /// If `profile` is not a mode-A profile.
    #[must_use]
    pub fn for_set(gate: &'g ChainGate, set: Set, profile: FlockProfile) -> Self {
        ligerito_of(profile);
        Self {
            gate,
            set,
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
    /// span's cap and proof, every later segment is covered (no commitment, empty share and
    /// proof items) and adds its statement, `Y_sigma` and seed; the last one verifies the
    /// span. A span sends no share, so every share item is empty.
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
            if let Err(e) = SegmentShares::from_bytes(segment.shares, 1) {
                return self.reject(sigma, SegmentRejection::Item(e));
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
                record: *segment.record,
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
            if let Err(e) = SegmentShares::from_bytes(segment.shares, 0)
                .and_then(|_| SegmentProofs::from_bytes(segment.proof, 0))
            {
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
            verify_tied(
                &seg,
                self.set,
                &st.mu,
                &st.c_tilde,
                &commitment,
                *segment.bridge_seed,
                segment.matrices,
                &st.final_vectors,
                &span_seed(&st.seeds, &st.record),
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

/// [`TiedPart::verify`] with the `c~` of `set` given concatenated.
#[allow(clippy::too_many_arguments)]
fn verify_tied(
    part: &TiedPart<'_>,
    set: Set,
    mu: &[[u8; 64]],
    c_tilde: &[u8],
    commitment: &TiedCommitment,
    bridge_seed: [u8; 32],
    matrices: &BridgeMatrices,
    final_vectors: &[Vec<u8>],
    seed: &[u8; 32],
    proof: &KeccakProof,
) -> Result<(), KeccakError> {
    fn chunks<const C: usize>(bytes: &[u8]) -> &[[u8; C]] {
        let (chunks, rest) = bytes.as_chunks::<C>();
        debug_assert!(rest.is_empty(), "whole c~ values");
        chunks
    }
    match set {
        Set::S44 => part.verify(mu, chunks::<32>(c_tilde), commitment, bridge_seed, matrices, final_vectors, seed, proof),
        Set::S65 => part.verify(mu, chunks::<48>(c_tilde), commitment, bridge_seed, matrices, final_vectors, seed, proof),
        Set::S87 => part.verify(mu, chunks::<64>(c_tilde), commitment, bridge_seed, matrices, final_vectors, seed, proof),
    }
}

impl KeccakVerifier for FlockVerifier<'_> {
    fn profile(&self) -> FlockProfile {
        self.profile
    }

    /// The tie sites of the tied parts the commitment-phase items name: one for
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
        let shares = match SegmentShares::from_bytes(segment.shares, count) {
            Ok(s) => s,
            Err(e) => return self.reject(sigma, SegmentRejection::Item(e)),
        };
        let proofs = match SegmentProofs::from_bytes(segment.proof, count) {
            Ok(p) => p,
            Err(e) => return self.reject(sigma, SegmentRejection::Item(e)),
        };
        if count == 0 {
            // No live record: the Flock side's bits are all zero, so Y_sigma must be zero.
            return segment.final_vector.iter().all(|&b| b == 0)
                || self.reject(sigma, SegmentRejection::EmptySegment);
        }
        let shares: Vec<Vec<u8>> = shares.shares.iter().map(unpack_final_vector).collect();
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
            let width = agg_piop::wire::c_tilde_bytes(self.set);
            let outcome = verify_tied(
                &seg,
                self.set,
                &segment.mu[range.clone()],
                &segment.c_tilde[range.start * width..range.end * width],
                &commitment,
                *segment.bridge_seed,
                segment.matrices,
                std::slice::from_ref(share),
                &part_seed(segment.seed, s, segment.record),
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

/// A signed record `(pk, M, signature)` (pure ML-DSA, empty context).
pub type SignedRecord<'a> = (&'a [u8], &'a [u8], &'a [u8]);

/// Aggregator configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AggregatorConfig {
    /// Requested `log2` of the record positions per Flock part (clamped per segment to
    /// `[min_part_log2, d_seg]`: at most two parts per segment, [`MAX_PARTS_LOG2`]).
    pub part_log2: u32,
    /// The Flock profile ([`MODE_A_PROFILES`], bound at the instance record).
    pub profile: FlockProfile,
    /// Prove a batch of more than one segment as one span over all segments when the parts
    /// are whole segments (module docs; default on).
    pub span: bool,
    /// The relation to prove (default H-u). The Keccak side
    /// does not depend on it.
    pub relation: Relation,
    /// The parameter set of the records (default ML-DSA-44; bound in the instance record
    /// and the proof header).
    pub set: Set,
}

impl Default for AggregatorConfig {
    fn default() -> Self {
        Self {
            part_log2: DEFAULT_PART_LOG2,
            profile: FLOCK_PROFILE,
            span: true,
            relation: Relation::Hu,
            set: Set::S44,
        }
    }
}

/// The production sponge gate of `set` (ML-DSA-44's is [`sponge_gate`]).
fn gate_of(set: Set) -> ChainGate {
    match set {
        Set::S44 => sponge_gate(),
        other => sponge_gate_for(other),
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
    /// No record, or more than `2^15` (ML-DSA-65 and ML-DSA-87: fewer than
    /// `agg_piop::prover::LANE_MIN_RECORDS`).
    Count(usize),
    /// Record `i` has a public key or signature of the wrong length.
    Malformed(usize, &'static str),
    /// Signature `i` does not verify (strict FIPS 204 verification during the derivation).
    Signature(usize, String),
    /// The prover failed.
    Prove(ProveError),
    /// The configured Flock profile is not a mode-A profile.
    Profile(FlockProfile),
    /// The relation is not defined for the parameter set.
    Relation(Set, Relation),
}

impl std::fmt::Display for AggregateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Count(n) => write!(
                f,
                "{n} records: an aggregate holds 1 to 2^15 (ML-DSA-65 and ML-DSA-87: {} to 2^15)",
                agg_piop::prover::LANE_MIN_RECORDS
            ),
            Self::Malformed(i, what) => write!(f, "record {i}: {what}"),
            Self::Signature(i, why) => write!(f, "record {i}: signature rejected ({why})"),
            Self::Prove(e) => write!(f, "prover: {e}"),
            Self::Profile(p) => write!(f, "Flock profile {p:?} is not a mode-A profile"),
            Self::Relation(set, relation) => write!(
                f,
                "relation {relation} is not available for ML-DSA-{}",
                set.name()
            ),
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
    /// The Flock profile (bound in the instance record).
    pub profile: FlockProfile,
    /// The relation proven (bound in the instance record and the proof header).
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
            ("Flock parts (events 2, 6, join)", t.keccak),
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
        writeln!(f, "  opening pool: {} threads", t.opening_threads)?;
        if t.flock_threads > 0 {
            writeln!(
                f,
                "  fork-join: Flock proofs {:.1} ms on {} threads beside the PIOP; wall {:.1} ms",
                ms(t.flock_prove),
                t.flock_threads,
                ms(t.fork_join)
            )?;
        } else {
            writeln!(f, "  no fork-join: Flock proofs {:.1} ms, then the PIOP", ms(t.flock_prove))?;
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
    pub fn c_tilde(&self) -> &[Vec<u8>] {
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
    /// transcript's matrices while `G_main` holds the tampered bits (an F_p-side flip the
    /// Flock side does not see). Replaces `tamper_bridge`.
    pub stale_bridge: bool,
    /// Let the prover split segments into parts below the admitted `min_part_log2` (more
    /// than two parts per segment), which verifiers must refuse.
    pub part_floor: Option<u32>,
}

/// The bridge-matrix derivation from the seed that keeps the last matrices it derived (test
/// hooks only).
#[derive(Default)]
struct RecordingMatrices(Mutex<Option<Arc<BridgeMatrices>>>);

impl BridgeDerivation for RecordingMatrices {
    fn matrices(&self, seed: &[u8; 32]) -> std::borrow::Cow<'_, BridgeMatrices> {
        self.matrices_for(Set::S44, seed)
    }

    fn seed_derived(&self) -> bool {
        true
    }

    fn matrices_for(&self, set: Set, seed: &[u8; 32]) -> std::borrow::Cow<'_, BridgeMatrices> {
        let m = agg_bridge::seed::derive_matrices_for(&agg_bridge::Shape::production(set), seed);
        *self.0.lock().expect("matrices") = Some(Arc::new(m.clone()));
        std::borrow::Cow::Owned(m)
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

/// [`sponge_records`] for any parameter set: the committed bits `b_t` at bridge position
/// `j + 256 rho + P_pad i` packed in `w1Encode` order, bit `256 W rho + W j + t` for `W`-bit
/// coefficients (FIPS 204 Algorithms 16 and 28: `W = 4` for ML-DSA-65 and ML-DSA-87). Only
/// the low bit of a cell is read (`cell & 1`): an honest prover's cells are bits, and the
/// F_p side's Booleanity constraint rejects tables whose cells are not, so a non-bit cell
/// never reaches an accepted proof whatever bit is taken here.
///
/// For four-bit coefficients a record's positions are read as nibbles straight from the bit
/// columns (`MainTables::bridge_nibble_source`, one 2 KiB buffer per record), and byte `m`
/// of w1Encode is nibble `2 m` plus nibble `2 m + 1` shifted by four.
///
/// # Panics
/// If the manifest is of another parameter set than `P`.
#[must_use]
pub fn sponge_records_for<P: agg_witness::params::ParamSet>(
    instance: &PublicInstance<P>,
    manifest: &CommitmentManifest,
    main: &MainTables,
) -> Vec<SpongeRecord<P>> {
    assert_eq!(manifest.set(), P::SET, "tables of the instance's parameter set");
    if let Some(nibbles) = main.bridge_nibble_source(manifest) {
        assert_eq!(P::W1_BITS, 4, "nibbles of four-bit coefficients");
        return instance
            .records
            .par_iter()
            .enumerate()
            .map_init(
                || vec![0_u8; P::P_PAD],
                |cells, (i, record)| {
                    // Non-bit cells keep their low bit (see above), so the flag is unused.
                    let _ = nibbles.record_into(i, cells);
                    let mut w1_encode = P::zero_w1_encode();
                    let bytes = w1_encode.as_mut();
                    for (byte, pair) in bytes.iter_mut().zip(cells[..P::P_POSITIONS].chunks_exact(2)) {
                        *byte = pair[0] | pair[1] << 4;
                    }
                    SpongeRecord {
                        mu: record.mu,
                        w1_encode,
                    }
                },
            )
            .collect();
    }
    let lanes = main.bridge_lanes(manifest);
    let width = P::W1_BITS;
    instance
        .records
        .par_iter()
        .enumerate()
        .map(|(i, record)| {
            let mut w1_encode = P::zero_w1_encode();
            let bytes = w1_encode.as_mut();
            for (t, lane) in lanes.iter().enumerate() {
                let cells = &lane[i * P::P_PAD..i * P::P_PAD + P::P_POSITIONS];
                for (p, &cell) in cells.iter().enumerate() {
                    if cell & 1 == 1 {
                        let q = 256 * width * (p / 256) + width * (p % 256) + t;
                        bytes[q / 8] |= 1 << (q % 8);
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
    /// An aggregator with the given configuration. It keeps each Flock part's sponge commit
    /// data from the commitment phase until the part is proven for the join (about the size of the sponge
    /// instance's codeword and Merkle tree per part), and a batch with exactly one Flock part
    /// (a span, or a one-segment batch of one part) keeps its committed witness from the commitment phase
    /// until the join (about 4.2-4.5 GiB held from the bridge-seed step through the final-vector step at 32K, est.).
    #[must_use]
    pub fn new(config: AggregatorConfig) -> Self {
        Self {
            gate: gate_of(config.set),
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
        match self.config.set {
            Set::S44 => self.aggregate_44(records, hooks),
            Set::S65 => self.aggregate_lanes::<Set65>(records, hooks),
            Set::S87 => self.aggregate_lanes::<Set87>(records, hooks),
        }
    }

    /// The report and the aggregate of a prover run.
    fn finish(
        &self,
        start: Instant,
        n: usize,
        instance_time: Duration,
        tables_time: Duration,
        flock_time: Duration,
        mut output: agg_piop::prover::ProverOutput,
        parts: Vec<PartProveLog>,
    ) -> Aggregate {
        output.timings.tables = tables_time;
        output.timings.total += tables_time;
        let bytes = output.proof.to_bytes();
        let size = output.proof.size();
        debug_assert_eq!(size.total(), bytes.len());
        let report = ProveReport {
            relation: self.config.relation,
            records: n,
            profile: self.config.profile,
            instance: instance_time,
            timings: output.timings,
            flock_records: flock_time,
            parts,
            size,
            bridge_seed: output.bridge_seed,
            total: start.elapsed(),
        };
        agg_piop::prover::phase_stamp("end");
        Aggregate {
            proof: output.proof,
            bytes,
            report,
        }
    }

    /// [`Self::aggregate_with`] for a parameter set with polynomial lanes (ML-DSA-65,
    /// ML-DSA-87): the same steps on the set's instance, tables, sponge records and lane
    /// public data. The hooks that build the Flock witness from ML-DSA-44 records
    /// (`tamper_flock_records`, `flock_witness`) are not available.
    fn aggregate_lanes<P: ParamSet>(
        &self,
        records: &[SignedRecord<'_>],
        hooks: &AggregateHooks<'_>,
    ) -> Result<Aggregate, AggregateError>
    where
        P::CTilde: Sync,
    {
        assert!(
            hooks.tamper_flock_records.is_none() && hooks.flock_witness.is_none(),
            "the Flock-record hooks are ML-DSA-44 only"
        );
        let start = Instant::now();
        agg_piop::prover::phase_stamp("start");
        let n = records.len();
        if !(agg_piop::prover::LANE_MIN_RECORDS..=1 << 15).contains(&n) {
            return Err(AggregateError::Count(n));
        }
        let profile = self.config.profile;
        if keccak_profile(profile).is_none() {
            return Err(AggregateError::Profile(profile));
        }
        let relation = self.config.relation;
        if !relation.is_available(P::SET) {
            return Err(AggregateError::Relation(P::SET, relation));
        }
        // The Akita setup and the lane public data on their own thread, as for ML-DSA-44.
        let (public_tx, public_rx) = std::sync::mpsc::channel::<Arc<PublicInstance<P>>>();
        let early_setup = std::thread::Builder::new()
            .name("agg-setup".into())
            .spawn(move || {
                let committer = CommitmentManifest::for_set(P::SET, n, relation)
                    .ok()
                    .and_then(|m| agg_piop::pcs::TableCommitter::new(m).ok());
                if let Some(committer) = &committer {
                    let _ = committer.warm();
                }
                let public = public_rx
                    .recv()
                    .ok()
                    .map(|instance| LanePublicData::from_instance(&instance));
                (committer, public)
            })
            .ok();
        for (i, (pk, _, sig)) in records.iter().enumerate() {
            if pk.len() != P::PK_BYTES {
                return Err(AggregateError::Malformed(i, "public key length"));
            }
            if sig.len() != P::SIG_BYTES {
                return Err(AggregateError::Malformed(i, "signature length"));
            }
        }
        let inputs: Vec<(&[u8], &[u8], P::CTilde)> = records
            .iter()
            .map(|&(pk, msg, sig)| (pk, msg, P::c_tilde_from_slice(&sig[..P::C_TILDE_BYTES])))
            .collect();
        let signatures: Vec<&[u8]> = records.iter().map(|r| r.2).collect();
        let instance = Arc::new(PublicInstance::<P>::new_for(&inputs).map_err(|e| match e {
            InstanceError::PublicKey(i) => AggregateError::Malformed(i, "public key"),
            InstanceError::Geometry(_) => AggregateError::Count(n),
        })?);
        let _ = public_tx.send(Arc::clone(&instance));
        let instance_time = start.elapsed();
        agg_piop::prover::phase_stamp("instance");
        let prover_inputs = ProverInputs {
            instance: &instance,
            records: &inputs,
            signatures: &signatures,
            relation,
        };
        let clock = Instant::now();
        let (manifest, honest) = main_tables_for(&prover_inputs)?;
        let tables_time = clock.elapsed();
        agg_piop::prover::phase_stamp("tables");
        let clock = Instant::now();
        let tampered = hooks.tamper_main.map(|tamper| {
            let mut main = honest.clone();
            tamper(&manifest, &mut main);
            main
        });
        let main = tampered.as_ref().unwrap_or(&honest);
        let flock_records =
            sponge_records_for(&instance, &manifest, if hooks.flock_from_tampered { main } else { &honest });
        let flock_time = clock.elapsed();
        agg_piop::prover::phase_stamp("flock-records");
        let side = FlockProver::<P>::new(&self.gate, main.geometry(), flock_records, self.config.part_log2, None)
            .profile(profile)
            .span(self.config.span);
        let side = match hooks.part_floor {
            Some(floor) => side.part_floor(floor),
            None => side,
        };
        // The round-2 tables of the honest bits under the transcript's matrices, as for
        // ML-DSA-44.
        let recording = RecordingMatrices::default();
        let stale = |manifest: &CommitmentManifest, tables: &mut BridgeGroupTables| {
            let matrices = recording.0.lock().expect("matrices").clone();
            let psi = tables.psi().to_vec();
            *tables = BridgeGroupTables::build(manifest, &honest, &matrices.expect("derived"), &psi)
                .expect("honest round-2 tables");
        };
        let prove_hooks = ProveHooks {
            unchecked: hooks.tamper_main.is_some() || hooks.tamper_bridge.is_some() || hooks.stale_bridge,
            tamper_bridge: if hooks.stale_bridge { Some(&stale) } else { hooks.tamper_bridge },
            swap_join: false,
        };
        let (committer, public) = early_setup
            .and_then(|handle| handle.join().ok())
            .unwrap_or((None, None));
        let output = if hooks.stale_bridge {
            prove_lanes_with_tables_using(
                committer,
                public,
                &prover_inputs,
                &manifest,
                main,
                &recording,
                &side,
                profile,
                &prove_hooks,
            )?
        } else {
            prove_lanes_with_tables_using(
                committer,
                public,
                &prover_inputs,
                &manifest,
                main,
                &SeededMatrices,
                &side,
                profile,
                &prove_hooks,
            )?
        };
        Ok(self.finish(start, n, instance_time, tables_time, flock_time, output, side.log()))
    }

    /// [`Self::aggregate_with`] for ML-DSA-44.
    fn aggregate_44(
        &self,
        records: &[SignedRecord<'_>],
        hooks: &AggregateHooks<'_>,
    ) -> Result<Aggregate, AggregateError> {
        let start = Instant::now();
        agg_piop::prover::phase_stamp("start");
        let n = records.len();
        if n == 0 || n > 1 << 15 {
            return Err(AggregateError::Count(n));
        }
        let profile = self.config.profile;
        if keccak_profile(profile).is_none() {
            return Err(AggregateError::Profile(profile));
        }
        // The Akita prover setup and the public data depend only on the record count, the
        // relation and the instance: the setup is built on its own thread while the instance
        // is computed, then the same thread builds the public data while the tables are
        // computed. Any failure here leaves both to `prove_with_tables_using`.
        let relation = self.config.relation;
        let (public_tx, public_rx) = std::sync::mpsc::channel::<Arc<PublicInstance>>();
        let early_setup = std::thread::Builder::new()
            .name("agg-setup".into())
            .spawn(move || {
                let committer = CommitmentManifest::for_relation(n, relation)
                    .ok()
                    .and_then(|m| agg_piop::pcs::TableCommitter::new(m).ok());
                if let Some(committer) = &committer {
                    let _ = committer.warm();
                }
                let public = public_rx
                    .recv()
                    .ok()
                    .map(|instance| PublicData::from_instance(&instance));
                (committer, public)
            })
            .ok();
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
        let instance = Arc::new(PublicInstance::new(&inputs).map_err(|e| match e {
            InstanceError::PublicKey(i) => AggregateError::Malformed(i, "public key"),
            InstanceError::Geometry(_) => AggregateError::Count(n),
        })?);
        let _ = public_tx.send(Arc::clone(&instance));
        let instance_time = start.elapsed();
        agg_piop::prover::phase_stamp("instance");
        let prover_inputs = ProverInputs {
            instance: &instance,
            records: &inputs,
            signatures: &signatures,
            relation: self.config.relation,
        };
        let clock = Instant::now();
        let (manifest, honest) = main_tables(&prover_inputs)?;
        let tables_time = clock.elapsed();
        agg_piop::prover::phase_stamp("tables");
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
        agg_piop::prover::phase_stamp("flock-records");
        let side = FlockProver::new(
            &self.gate,
            main.geometry(),
            flock_records,
            self.config.part_log2,
            hooks.flock_witness,
        )
        .profile(profile)
        .span(self.config.span);
        let side = match hooks.part_floor {
            Some(floor) => side.part_floor(floor),
            None => side,
        };
        let recording = RecordingMatrices::default();
        let stale = |manifest: &CommitmentManifest, tables: &mut BridgeGroupTables| {
            let matrices = recording.0.lock().expect("matrices").clone();
            let psi = tables.psi().to_vec();
            *tables = BridgeGroupTables::build(manifest, &honest, &matrices.expect("derived"), &psi)
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
            swap_join: false,
        };
        let (committer, public) = early_setup
            .and_then(|handle| handle.join().ok())
            .unwrap_or((None, None));
        let output = if hooks.stale_bridge {
            prove_with_tables_using(
                committer,
                public,
                &prover_inputs,
                &manifest,
                main,
                &recording,
                &side,
                profile,
                &prove_hooks,
            )?
        } else {
            prove_with_tables_using(
                committer,
                public,
                &prover_inputs,
                &manifest,
                main,
                &SeededMatrices,
                &side,
                profile,
                &prove_hooks,
            )?
        };
        Ok(self.finish(start, n, instance_time, tables_time, flock_time, output, side.log()))
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
    /// Reading the profile from the aggregate's commitment-phase items (a scan of the header,
    /// the `c~` list and those items, outside the PIOP verifier's steps; zero when the profile
    /// was given).
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

    /// Total verification time, the profile probe included.
    #[must_use]
    pub fn total(&self) -> Duration {
        self.report.timings.total + self.profile_probe
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
            ("parse", "parse, manifest"),
            ("public_data", "pkDecode, tr, mu, SampleInBall, PCS"),
            ("transcript", "events 0-6 replay"),
            ("matrices", "bridge matrices from sd"),
            ("keccak", "Flock parts (beside the PIOP)"),
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
/// the Flock profile named by its commitment-phase items when that is a default-accepted mode-A profile
/// ([`DEFAULT_ACCEPTED_PROFILES`]: `Fast`; the transcript binds the profile at
/// the instance record, so the items cannot name another one than the prover bound), or under the profile
/// fixed by [`Self::require`] (which also admits `Slim` and `Slim8`).
pub struct AggregateVerifier {
    gate: ChainGate,
    require: Option<FlockProfile>,
    relation: Relation,
    set: Set,
}

/// The verifier's key cache of the verified parameter set.
pub enum KeyCacheMut<'c> {
    /// No cache: every key is expanded.
    None,
    /// ML-DSA-44 keys.
    Rows(&'c mut KeyCache),
    /// ML-DSA-65 or ML-DSA-87 keys.
    Lanes(&'c mut LaneKeyCache),
}

impl Default for AggregateVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// The Flock profile id named by the commitment-phase items of `aggregate`, read in place
/// without parsing the whole aggregate: the header (`u16` version, flags, `u64` `N`), the
/// `c~_i`, the two framed round-1 commitments, the `LE32` item count, then the framed items
/// up to the first that parses ([`crate::wire::flock_profile`]). `None` when the bytes end
/// early or the header names no known set; the full parse in the verifier rejects such
/// aggregates.
fn probe_profile_id(aggregate: &[u8]) -> Option<u8> {
    fn take<'a>(bytes: &'a [u8], at: &mut usize, len: usize) -> Option<&'a [u8]> {
        let end = at.checked_add(len)?;
        let out = bytes.get(*at..end)?;
        *at = end;
        Some(out)
    }
    fn le32(bytes: &[u8], at: &mut usize) -> Option<usize> {
        let word = take(bytes, at, 4)?;
        usize::try_from(u32::from_le_bytes(word.try_into().ok()?)).ok()
    }
    let at = &mut 0_usize;
    take(aggregate, at, 2)?;
    let set = agg_piop::wire::set_of_nibble(take(aggregate, at, 1)?[0] >> 4)?;
    let records = u64::from_le_bytes(take(aggregate, at, 8)?.try_into().ok()?);
    let c_tilde = usize::try_from(records).ok()?.checked_mul(agg_piop::wire::c_tilde_bytes(set))?;
    take(aggregate, at, c_tilde)?;
    for _ in 0..2 {
        let len = le32(aggregate, at)?;
        take(aggregate, at, len)?;
    }
    let count = le32(aggregate, at)?;
    for _ in 0..count {
        let len = le32(aggregate, at)?;
        if let Ok(item) = crate::wire::SegmentCommitments::from_bytes(take(aggregate, at, len)?) {
            return Some(item.profile);
        }
    }
    None
}

impl AggregateVerifier {
    /// A verifier.
    #[must_use]
    pub fn new() -> Self {
        Self::for_set(Set::S44)
    }

    /// A verifier of aggregates of parameter set `set`. The set is the verifier's choice,
    /// like the relation: an aggregate whose header names another set is rejected.
    #[must_use]
    pub fn for_set(set: Set) -> Self {
        Self {
            gate: gate_of(set),
            require: None,
            relation: Relation::Hu,
            set,
        }
    }

    /// The parameter set verified.
    #[must_use]
    pub const fn set(&self) -> Set {
        self.set
    }

    /// Verify under `relation` (default H-u). The relation is the verifier's choice: an
    /// aggregate whose header names another relation is rejected.
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

    /// The profile to verify `aggregate` under: the required one, else the mode-A profile its
    /// commitment-phase items name, else the default (an unparsable aggregate is rejected later). A
    /// named profile that is not accepted by default is returned as is, so that [`Self::verify`]
    /// can refuse it with a message naming `--flock-profile`.
    fn profile_of(&self, aggregate: &[u8]) -> FlockProfile {
        if let Some(p) = self.require {
            return p;
        }
        probe_profile_id(aggregate)
            .and_then(FlockProfile::from_id)
            .filter(|profile| profile.is_mode_a())
            .unwrap_or(FLOCK_PROFILE)
    }

    /// Verify `aggregate` for the records `(pk_i, M_i)`. The verifier computes `tr_i`, `mu_i`,
    /// `c_i = SampleInBall(c~_i)`, `T_i` and the `ExpandA` data itself; with `cache`, keys
    /// found there are not expanded again and new expansions are added.
    ///
    /// # Panics
    /// With a cache for a verifier of ML-DSA-65 or ML-DSA-87 (use [`Self::verify_with`]).
    pub fn verify(
        &self,
        statements: &[(&[u8], &[u8])],
        aggregate: &[u8],
        cache: Option<&mut KeyCache>,
    ) -> VerifyCostReport {
        self.verify_with(statements, aggregate, cache.map_or(KeyCacheMut::None, KeyCacheMut::Rows))
    }

    /// [`Self::verify`] with the key cache of any parameter set.
    ///
    /// # Panics
    /// If the cache is of another parameter set than the verifier's.
    pub fn verify_with(
        &self,
        statements: &[(&[u8], &[u8])],
        aggregate: &[u8],
        cache: KeyCacheMut<'_>,
    ) -> VerifyCostReport {
        let probe = Instant::now();
        let profile = self.profile_of(aggregate);
        let profile_probe = if self.require.is_some() {
            Duration::ZERO
        } else {
            probe.elapsed()
        };
        let keccak = FlockVerifier::for_set(&self.gate, self.set, profile);
        let cached = !matches!(cache, KeyCacheMut::None);
        let report = if self.require.is_some() || DEFAULT_ACCEPTED_PROFILES.contains(&profile) {
            match (self.set, cache) {
                (Set::S44, KeyCacheMut::None) => verify_timed(
                    statements,
                    aggregate,
                    &SeededMatrices,
                    &keccak,
                    profile,
                    self.relation,
                    None,
                ),
                (Set::S44, KeyCacheMut::Rows(cache)) => verify_timed(
                    statements,
                    aggregate,
                    &SeededMatrices,
                    &keccak,
                    profile,
                    self.relation,
                    Some(cache),
                ),
                (set, KeyCacheMut::None) if set != Set::S44 => verify_lanes_timed(
                    set,
                    statements,
                    aggregate,
                    &SeededMatrices,
                    &keccak,
                    profile,
                    self.relation,
                    None,
                ),
                (set, KeyCacheMut::Lanes(cache)) if set != Set::S44 && cache.set() == set => {
                    verify_lanes_timed(
                        set,
                        statements,
                        aggregate,
                        &SeededMatrices,
                        &keccak,
                        profile,
                        self.relation,
                        Some(cache),
                    )
                }
                (set, _) => panic!("a key cache of another parameter set than ML-DSA-{}", set.name()),
            }
        } else {
            VerifyReport {
                outcome: Err(VerifyError::ProfileNotAccepted(profile)),
                timings: VerifyTimings::default(),
            }
        };
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
        let check_with = |commitment: &[u8], y: &[u8], shares: &[u8], proof: &[u8]| {
            verifier.verify(&SegmentContext {
                sigma: 3,
                geometry,
                commitment,
                seed: &[0; 32],
                bridge_seed: &[0; 32],
                mu: &[],
                c_tilde: &[],
                final_vector: y,
                shares,
                record: &[0; 32],
                proof,
                matrices: &matrices,
            })
        };
        let check = |commitment: &[u8], y: &[u8], proof: &[u8]| check_with(commitment, y, &[], proof);
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
        assert!(!check_with(&empty, &zero, &[0; 512], &[]), "a share for an empty segment");
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

    /// The span glue of the verifier on the geometry of `N = 2^14 + 1` (four
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
            proofs: vec![vec![1, 2, 3]],
        }
        .to_bytes();
        let covered = item(0, Vec::new());
        let zero = vec![0_u8; Y_BITS];
        let mu = vec![[0_u8; 64]; 1 << 13];
        let c_tilde = vec![[0_u8; C_TILDE_BYTES]; 1 << 13];
        let live = [1 << 13, 1 << 13, 1, 0];
        let check_with = |sigma: usize, commitment: &[u8], shares: &[u8], proof: &[u8]| {
            verifier.verify(&SegmentContext {
                sigma,
                geometry,
                commitment,
                seed: &[sigma as u8; 32],
                bridge_seed: &[0; 32],
                mu: &mu[..live[sigma]],
                c_tilde: c_tilde[..live[sigma]].as_flattened(),
                final_vector: &zero,
                shares,
                record: &[7; 32],
                proof,
                matrices: &matrices,
            })
        };
        let check = |sigma: usize, commitment: &[u8], proof: &[u8]| {
            check_with(sigma, commitment, &[], proof)
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
        // A span sends no share: a share item at segment 0 or at a covered segment is
        // rejected (the record digest would bind it, but the item must be empty).
        assert!(!check_with(0, &open, &[0; 512], &open_proof));
        assert!(matches!(
            verifier.rejection(),
            Some((0, SegmentRejection::Item(crate::wire::ItemError::Trailing)))
        ));
        assert!(check(0, &open, &open_proof));
        assert!(!check_with(1, &covered, &[1], &[]));
        assert!(matches!(
            verifier.rejection(),
            Some((1, SegmentRejection::Item(crate::wire::ItemError::Trailing)))
        ));
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
        // The vectors of `python3 -I docs/scripts/pow_plan.py`, N = N*, default
        // part size: the plan of the code's own Keccak side and PIOP registry.
        use agg_piop::pow::PowPlan;
        let gate = sponge_gate();
        let hex = |d: [u8; 32]| d.iter().map(|b| format!("{b:02x}")).collect::<String>();
        for (d_sig, profile, span, want) in [
            (3, FlockProfile::Slim, true, "9757c063ae3b3a704330367ea2552dcfd41c9f20841abccce47a9a720cf69437"),
            (10, FlockProfile::Slim, true, "0aa41fd0c4c2a9bd0df0b63d86e57787cb31c6d7cedec0ac9edd626cd2fb3eea"),
            (13, FlockProfile::Slim, true, "aa72651e90b32c908db52ec832790d72e685c4493c7fd3e80bebe60cd506b5c9"),
            (15, FlockProfile::Fast, true, "deb090a48d8057c6aed064b95bf6fb8a1499207935005fc0d98b40ab5de3864c"),
            (15, FlockProfile::Slim, true, "ef9710959f30286f422e170d9112bca67e7526e17382660675061256ef3e3303"),
            (15, FlockProfile::Slim, false, "fe42263b92bff80e6c367e2b44d7673d274f454841e1784332f5971a03e136c8"),
            (15, FlockProfile::Slim8, true, "0ba687fbd3b201e24c3776d9ee689dfcfb81ea841de12aa7cbbb3157199ccad8"),
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
    fn plan_digests_of_65_and_87_reproduce_the_script() {
        // python3 -I docs/scripts/pow_plan.py --set 65|87: mode A at N = N*, default part size,
        // with the tied parts of the set's sponge gate (four chain lanes, P_pad = 2^11, a
        // public region of 3 or 4 word coordinates).
        use agg_keccak::sponge::sponge_gate_for;
        use agg_piop::pow::PowPlan;
        use agg_witness::params::Set;
        let hex = |d: [u8; 32]| d.iter().map(|b| format!("{b:02x}")).collect::<String>();
        let cases: [(Set, u32, FlockProfile, bool, &str); 12] = [
            (Set::S65, 3, FlockProfile::Slim, true, "09d0b47510834ab04d6b9eafc1b9afd61ff3ce538ea292b6ccc74f8384eaa3a1"),
            (Set::S65, 10, FlockProfile::Fast, true, "482fb6d4668fcf08df8e3ebd90ba49536a647f7249f2d00eacd4fa29086137a5"),
            (Set::S65, 13, FlockProfile::Slim, true, "6b31340ea7b8be4c7cdd2c8364f59a6b5c0c803bc557640cfba877f26dac1712"),
            (Set::S65, 15, FlockProfile::Fast, true, "fa3b618068aab74000424181bc16a5ad9bc686f27856af2b86834ecacc000a4e"),
            (Set::S65, 15, FlockProfile::Fast, false, "1298836050bd5679cb2fdda08aaad4e89503c96bbba35f5bfda99b06325b34fb"),
            (Set::S65, 15, FlockProfile::Slim8, true, "1c4a6b3718ad4cd0e2cb768dbf604a1acd1d3b6355fe591bd3ab74f4b5fe2e94"),
            (Set::S87, 3, FlockProfile::Slim, true, "f82fcfead3bf6d2bdbd26cc8a3f23ba4f178726626efa5204500e48573584754"),
            (Set::S87, 10, FlockProfile::Fast, true, "470e31983acbea1b02fe0a1ce92792eb4d58a8c22398d9a5d9b684f119a22660"),
            (Set::S87, 13, FlockProfile::Slim, true, "66dba96201af36606e5cc0715ff7addd2a1fb0b63d089cd6cbd540af3dfd2949"),
            (Set::S87, 15, FlockProfile::Fast, true, "ffe87bdf123c773fd206afc67c59c3b81c00b1e8e70d837bc7f90fef9a3dccad"),
            (Set::S87, 15, FlockProfile::Fast, false, "0052e6e8cad1587549889c3d0c81fffeb9b4b577ccf3bddaac25043c8eff1ed7"),
            (Set::S87, 15, FlockProfile::Slim8, true, "2fcb99afb477f65e296189d5918cc79532fbfcee4e1a3f9c54b22f9cc774dfcb"),
        ];
        for (set, d_sig, profile, span, want) in cases {
            let gate = sponge_gate_for(set);
            let n = 1 << d_sig;
            let keccak = keccak_plan(&gate, &Geometry::new(n).unwrap(), DEFAULT_PART_LOG2, span, profile);
            let manifest = CommitmentManifest::for_set(set, n, Relation::Hu).unwrap();
            let plan = PowPlan::new(&manifest, profile, &keccak);
            assert_eq!(hex(plan.digest()), want, "{set} 2^{d_sig} {profile:?} span {span}");
        }
    }

    #[test]
    fn part_seeds_are_distinct_per_part_segment_seed_and_record() {
        let rec = [9; 32];
        let a = part_seed(&[1; 32], 0, &rec);
        assert_ne!(a, part_seed(&[1; 32], 1, &rec));
        assert_ne!(a, part_seed(&[2; 32], 0, &rec));
        assert_eq!(a, part_seed(&[1; 32], 0, &rec));
        // The final-vector record digest changes every seed.
        assert_ne!(a, part_seed(&[1; 32], 0, &[8; 32]));
        // The span seed depends on every segment seed, their order and the record digest.
        let s = span_seed(&[[1; 32], [2; 32]], &rec);
        assert_ne!(s, span_seed(&[[2; 32], [1; 32]], &rec));
        assert_ne!(s, span_seed(&[[1; 32], [2; 32], [0; 32]], &rec));
        assert_ne!(s, span_seed(&[[1; 32], [2; 32]], &[8; 32]));
        assert_ne!(s, part_seed(&[1; 32], 0, &rec));
    }
}
