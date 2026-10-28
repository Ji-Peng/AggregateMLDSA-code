//! The tied part (`docs/parity-chain.md`): the Flock argument of one part of a segment (or
//! of a span of segments) as ONE Flock instance, the sponge instance (the `n` live records'
//! SHAKE256 gates, docs/keccak.md), and the virtual parity chain
//! ([`agg_bridge::parity_chain`]): a degree-2 sum-check that reduces the public final vectors
//! `Y_sigma = M_3 M_2 M_1 b'_sigma` to one bit-level evaluation of the sponge commitment,
//! which the sponge instance's Flock opening proves as a third ring-switched claim
//! ([`crate::flock_union::BitClaim`]).
//!
//! Regions. The sponge instance commits its words BatchMajor: word column `c` of row `j` is
//! the padded word `(c << nu_s) | j`. Lane `t` of a record is the eight words `8 t + w`, `w <
//! 8` ([`crate::sponge::W1_WORD_BASE`]), bit `p & 127` of word `p >> 7` for the lane position
//! `p = j + 256 rho`; the words `[48, 56)` are the public region (mu, `c~`, the constant, a
//! zero word). The chain runs over eight lanes, the word columns `[0, 64)`: lanes 6 and 7 are
//! the public region and the first chi words, with weight zero.
//!
//! Transcript. From the part's Flock seed `fs`, its geometry, the digest of
//! the final vectors and the sponge's event-2 cap, the tie challenger draws (each site after
//! its proof of work): the public claim's point `(rho_r, rho_c)`; `eq_Y = (r_o, r_y)`; one
//! challenge per chain round after absorbing its message `(q(1), q(inf))`; then it absorbs
//! the claimed bit value `v` and squeezes a 32-byte tie binding, which the sponge challenger
//! absorbs before Flock binds its statement (docs/soundness.md). The verifier recomputes
//! the claim `<eq_Y, Y>`, replays the rounds, checks the last claim against its own weight
//! `W~(r)` times `eq(r_rec[nu_s..], 0) v`, and Flock's opening proves `v` and the public value.
//!
//! Spans. A tied part may cover `S` segments that share its geometry (`d`,
//! part index): mode A proves all segments of a batch of more than `2^13` records as one tied
//! part. The sponge instance holds the `n` records of the span in order (record `i` of
//! segment `sigma` is row `2^d sigma + i`), and the chain covers `2^span_log >= S` segments,
//! the final vectors of the segments `S..2^span_log` being zero.
//!
//! Work (docs/soundness.md): the sponge commitment is bound only to its Johnson list
//! when these challenges are drawn, so the tie site carries `L_s (nu_s + 3)`, the `eq_Y` site
//! `L_s (log2 m_3 + span_log)` and every round `2 L_s` ([`agg_bridge::pow::part_work`]); the
//! nonces (`g + 7` bits, `LE32`) travel in the proof.

use agg_bridge::gf2_walker::{self, SegmentLayout};
use agg_bridge::parity_chain::{ChainLayout, ChainPoint, EqY};
use agg_bridge::pow::{nonce_bits, part_work, PartWork, TieGeometry};
use flock_core::{
    challenger::{Challenger, FsChallenger},
    lincheck::{build_eq_table, LincheckCircuit},
};
use flock_field::F128;
use rayon::prelude::*;
use sha3::{Digest, Sha3_256};

use agg_witness::bridge::BridgeMatrices;

use crate::{
    bridge_slot::pack_y,
    flock_union::{bit_value, BitClaim, CommitData, UnionGeometry, UnionProof},
    gate::ChainGate,
    prover::{
        capacity_log, sponge_public_words, KeccakCommitment, KeccakError, KeccakProof, Profile,
        DOMAIN,
    },
    sponge::{
        SpongeRecord, CHI_WORD_BASE, CONST_WORD, CTILDE_BYTES, CTILDE_WORD_BASE, MU_BYTES,
        MU_WORD_BASE, PUBLIC_WORDS_LOG2, PUBLIC_WORD_BASE, W1_INPUT_BASE, W1_WORD_BASE,
    },
    witness::{pack_slot, SlotBuffers},
};

/// Magic prefix of a tied-part proof.
pub const MAGIC: &[u8; 6] = b"AGGKCT";
/// Serialization version.
pub const VERSION: u8 = 5;
/// Header: magic, version, profile, `n` (LE32), `d`, segments, tie nonce (LE32).
const HEADER: usize = 6 + 1 + 1 + 4 + 1 + 1 + 4;
/// Offset of the tie nonce in the header.
pub const NONCE_OFFSET: usize = 14;
/// Bytes of one chain round in the proof: `q(1)`, `q(inf)`, `LE32(nonce)`.
pub const ROUND_BYTES: usize = 16 + 16 + 4;
/// Label under which the sponge challenger absorbs the tie binding.
const TIE_BINDING_LABEL: &[u8] = b"agg-keccak/v4/tie-binding";
/// Digest label of the public table of final vectors.
const Y_TABLE_LABEL: &[u8] = b"agg-keccak/v3/tied-final-vectors";
/// Domain of the tie challenger (version 5).
pub const TIE_DOMAIN: &[u8] = b"agg-mldsa/v3/tie";
/// Challenger label of the sponge instance (version 5).
const SPONGE_LABEL: &[u8] = b"agg-keccak/v5/tied-sponge";
/// Record positions per production level-1 block (`log2`): `2^19 / 1024`.
const BLOCK_RECORDS_LOG2: usize = 9;
/// Packed-direct claims of the sponge proof: the public region.
const SPONGE_CLAIMS: usize = 1;
/// Lanes of the chain (`2^lb`, junk lanes 6 and 7 included).
const CHAIN_LANES: usize = 8;

/// Peak resident set size (VmHWM, MiB) of this process, for the `AGG_TIED_TRACE` phase trace.
fn peak_rss_mib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib / 1024)
}

/// With `AGG_TIED_TRACE` set, print a phase of the tied prover with the peak RSS so far.
fn trace(phase: &str, start: &std::time::Instant) {
    if std::env::var_os("AGG_TIED_TRACE").is_some() {
        eprintln!(
            "  [tied] {phase:<28} {:8.1} ms  peak RSS {:>6} MiB",
            start.elapsed().as_secs_f64() * 1e3,
            peak_rss_mib().unwrap_or(0)
        );
    }
}

/// The sponge gate's block layout must be the one the regions assume (`crate::sponge`,
/// module docs).
fn assert_tied_layout(gate: &ChainGate) {
    let lay = gate.layout();
    let spec = gate.spec();
    assert!(
        spec.input_fields.len() == 2
            && !spec.input_fields[0].public
            && spec.input_fields[0].bits == 6 * 1024
            && spec.input_fields[1].public
            && spec.input_fields[1].bits == 8 * MU_BYTES
            && lay.field_base == [128 * W1_WORD_BASE, 128 * MU_WORD_BASE]
            && lay.output_base == 128 * CTILDE_WORD_BASE
            && lay.const_col == 128 * CONST_WORD
            && lay.t_base == 2 * CHI_WORD_BASE
            && PUBLIC_WORD_BASE + (1 << PUBLIC_WORDS_LOG2) == CHI_WORD_BASE
            && PUBLIC_WORD_BASE.is_multiple_of(1 << PUBLIC_WORDS_LOG2)
            && 8 * CHAIN_LANES <= CHI_WORD_BASE + 8
            && W1_WORD_BASE == 0,
        "the tied part needs the sponge layout of crate::sponge"
    );
}

fn put_framed(h: &mut impl Challenger, bytes: &[u8]) {
    h.observe_bytes(&(bytes.len() as u64).to_le_bytes());
    h.observe_bytes(bytes);
}

/// The work of a site: the prover grinds, the verifier checks a nonce from the proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TieWork {
    /// Search the smallest passing nonce (Flock's `grind_pow`).
    Grind,
    /// Check this nonce.
    Check(u32),
}

/// The prover's nonce if it lies below the allowance `limit = 2^(g + 7)`: Flock's `grind_pow`
/// returns the smallest passing nonce, so a larger one means that the first `2^(g + 7)`
/// candidates all failed (docs/soundness.md).
fn nonce_within(nonce: u64, limit: u64) -> Result<u32, KeccakError> {
    const EXHAUSTED: &str = "tie nonce search exhausted its 2^(g + 7) candidates";
    if nonce >= limit {
        return Err(KeccakError::Work(EXHAUSTED));
    }
    u32::try_from(nonce).map_err(|_| KeccakError::Work(EXHAUSTED))
}

/// The proof of work of one site of the tie challenger with difficulty `bits`: the prover
/// grinds (Flock's `grind_pow`, the nonce absorbed), the verifier checks the given nonce
/// (`g + 7` bits at most, the predicate) or rejects with `KeccakError::Tie(name)`.
fn site_work(
    ch: &mut FsChallenger,
    bits: u8,
    work: TieWork,
    name: &'static str,
) -> Result<u32, KeccakError> {
    let limit = 1_u64 << nonce_bits(bits);
    match (bits, work) {
        (0, TieWork::Grind) => Ok(0),
        (0, TieWork::Check(nonce)) => {
            if nonce == 0 {
                Ok(0)
            } else {
                Err(KeccakError::Tie(name))
            }
        }
        (_, TieWork::Grind) => nonce_within(ch.grind_pow(u32::from(bits)), limit),
        (_, TieWork::Check(nonce)) => {
            if u64::from(nonce) >= limit || !ch.verify_pow(u64::from(nonce), u32::from(bits)) {
                Err(KeccakError::Tie(name))
            } else {
                Ok(nonce)
            }
        }
    }
}

/// The 32-byte tie binding: two `F128` squeezes.
fn squeeze_binding(ch: &mut FsChallenger) -> [u8; 32] {
    let mut binding = [0_u8; 32];
    for (chunk, word) in binding
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(ch.sample_f128_vec(2))
    {
        chunk[..8].copy_from_slice(&word.lo.to_le_bytes());
        chunk[8..].copy_from_slice(&word.hi.to_le_bytes());
    }
    binding
}

/// The event-2 commitment of a tied part: the sponge instance's Ligerito cap.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TiedCommitment {
    /// The sponge instance's Ligerito cap.
    pub sponge: KeccakCommitment,
}

impl TiedCommitment {
    /// `LE32(len) || sponge`: the event-2 payload of the part.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let b = self.sponge.as_bytes();
        let mut out = Vec::with_capacity(4 + b.len());
        out.extend_from_slice(
            &u32::try_from(b.len())
                .expect("cap below 4 GiB")
                .to_le_bytes(),
        );
        out.extend_from_slice(b);
        out
    }

    /// Parse (the cap length is checked by [`TiedPart::verify`]).
    ///
    /// # Errors
    /// On a malformed byte string.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, KeccakError> {
        let malformed = || KeccakError::Malformed("tied commitment");
        let len = bytes.get(..4).ok_or_else(malformed)?;
        let len = u32::from_le_bytes(len.try_into().expect("4 bytes")) as usize;
        if bytes.len() != 4 + len {
            return Err(malformed());
        }
        Ok(Self {
            sponge: KeccakCommitment::from_bytes(bytes[4..].to_vec()),
        })
    }
}

/// A committed tied part: the sponge witness and its commitment data, the part's input
/// bits (the chain's lanes 0..5, the prover's copy of `b'`), the statement words and the
/// cap.
pub struct TiedCommitted {
    n: usize,
    sponge: SlotBuffers,
    sponge_data: Option<CommitData>,
    bits: Vec<u64>,
    public: Vec<F128>,
    c_tilde: Vec<[u8; CTILDE_BYTES]>,
    commitment: TiedCommitment,
}

impl TiedCommitted {
    /// The commitment to absorb at event 2.
    #[must_use]
    pub fn commitment(&self) -> &TiedCommitment {
        &self.commitment
    }

    /// The published `c~_i`.
    #[must_use]
    pub fn c_tilde(&self) -> &[[u8; CTILDE_BYTES]] {
        &self.c_tilde
    }

    /// The part's lane-major input bits.
    #[must_use]
    pub fn bits(&self) -> &[u64] {
        &self.bits
    }

    /// Take the sponge commitment's prover data (to keep it while the buffers are dropped);
    /// the proof then commits again unless the data is handed back by
    /// [`TiedPart::reattach`].
    pub fn take_commit_data(&mut self) -> Option<CommitData> {
        self.sponge_data.take()
    }

    /// The sponge commitment's prover data, if it is still held (the proof then opens it
    /// without committing again).
    #[must_use]
    pub fn commit_data(&self) -> Option<&CommitData> {
        self.sponge_data.as_ref()
    }

    /// Records in the part.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Always false (a part has at least one record).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }
}

/// A tied part's proof with the final vectors it proves (one per segment of the span).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TiedProof {
    /// `Y_sigma` (one bit per byte) of the span's segments, in order.
    pub final_vectors: Vec<Vec<u8>>,
    /// The proof bytes.
    pub proof: KeccakProof,
}

/// One round of the parity chain in a proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChainRound {
    /// `q(1)`.
    pub q1: F128,
    /// `q(inf)`, the leading coefficient.
    pub qinf: F128,
    /// The round's work nonce.
    pub nonce: u32,
}

/// The parsed parts of a tied-part proof: the nonces, the chain's rounds, the
/// claimed bit value and the sponge instance's proof. Tests use it to tamper single fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TiedProofParts {
    /// The tie site's nonce.
    pub nonce: u32,
    /// The `eq_Y` site's nonce.
    pub eq_nonce: u32,
    /// The chain's rounds.
    pub rounds: Vec<ChainRound>,
    /// The claimed value `v` of the sponge's bit polynomial at the chain's point.
    pub value: F128,
    /// The sponge instance's Flock proof (one packed-direct value: the public region).
    pub sponge: UnionProof,
}

/// Sizes of a tied part.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TiedCost {
    /// Records.
    pub records: usize,
    /// Segments of the span.
    pub segments: usize,
    /// Sponge instance: row capacity `log2`, address variables, dense (committed) variables.
    pub sponge_nu: usize,
    pub sponge_m: usize,
    pub sponge_dense_m: usize,
    /// Committed 128-bit words of the sponge instance.
    pub sponge_committed_words: usize,
    /// Rounds of the parity chain.
    pub chain_rounds: usize,
    /// Input bits of the parity chain (`2^span_log` segments of eight lanes).
    pub chain_input_bits: usize,
}

/// Proving and verifying context of one tied part: `n` live records in part `index` of a
/// segment split into parts of `2^d` record positions, or in that part of each of `S`
/// segments (a span, module docs).
pub struct TiedPart<'g> {
    gate: &'g ChainGate,
    n: usize,
    d: u32,
    index: usize,
    segments: usize,
    profile: Profile,
    sponge: UnionGeometry,
    chain: ChainLayout,
}

impl<'g> TiedPart<'g> {
    /// `d_seg` of a single-segment batch of `n <= 2^13` records: `log2 max(next_pow2(n), 8)`.
    #[must_use]
    pub fn d_seg_for(n: usize) -> u32 {
        assert!(
            (1..=1 << 13).contains(&n),
            "one segment holds 1 to 2^13 records"
        );
        n.next_power_of_two().max(8).trailing_zeros()
    }

    /// The whole segment as one part (`d_seg`, index 0).
    ///
    /// # Panics
    /// If `n == 0`, `n > 2^{d_seg}`, `d_seg < 3` or `d_seg > 13`.
    #[must_use]
    pub fn new(gate: &'g ChainGate, n: usize, d_seg: u32, profile: Profile) -> Self {
        Self::new_part(gate, n, d_seg, 0, profile)
    }

    /// Part `index` of a segment split into parts of `2^d` record positions, with `n` live
    /// records (the part's first `n` positions, segment records `2^d index ..`). It proves
    /// the part's share `Y_{sigma, index} = M^{(sigma)} b'_{index}`.
    ///
    /// # Panics
    /// If `n == 0`, `n > 2^d`, `d < 3`, `d > 13`, the part lies outside the segment, or the
    /// gate is not the sponge gate of `crate::sponge`.
    #[must_use]
    pub fn new_part(gate: &'g ChainGate, n: usize, d: u32, index: usize, profile: Profile) -> Self {
        Self::new_span(gate, n, d, index, 1, profile)
    }

    /// Part `index` of `segments` consecutive segments (each split into parts of `2^d`
    /// record positions) as one tied part with `n` live records in segment order: record
    /// `i` of segment `sigma` is record `2^d sigma + i` of the span, so the segments are
    /// filled in order (segments `>= ceil(n / 2^d)` are empty). It proves `Y_sigma =
    /// M^{(sigma)} b'_sigma` for every segment of the span (module docs).
    ///
    /// # Panics
    /// If `n == 0`, `n > segments 2^d`, `segments == 0`, `d < 3`, `d > 13`, the part lies
    /// outside the segment, or the gate is not the sponge gate of `crate::sponge`.
    #[must_use]
    pub fn new_span(
        gate: &'g ChainGate,
        n: usize,
        d: u32,
        index: usize,
        segments: usize,
        profile: Profile,
    ) -> Self {
        assert!(segments >= 1, "a span of at least one segment");
        assert!(n >= 1 && n <= segments << d, "1 <= n <= segments 2^d");
        assert!((3..=13).contains(&d), "3 <= d <= 13");
        assert_tied_layout(gate);
        let span_log = segments.next_power_of_two().trailing_zeros() as usize;
        let layout = SegmentLayout::production_part(d, index);
        let chain = ChainLayout::new(layout, span_log);
        assert_eq!(chain.lanes_padded(), CHAIN_LANES, "eight chain lanes");
        let nu_s = capacity_log(n, gate.layout().k_log);
        let sponge = UnionGeometry::new(gate.table_type(), n, nu_s, profile);
        assert!(
            sponge.word_vars() >= nu_s + 6,
            "the chain's lanes span 64 word columns"
        );
        let part = Self {
            gate,
            n,
            d,
            index,
            segments,
            profile,
            sponge,
            chain,
        };
        assert_eq!(
            part.tie_geometry(),
            Self::plan_geometry(gate, n, d, segments),
            "the plan's tie geometry is the part's"
        );
        assert_eq!(part.chain.rounds(), usize::from(part.work().rounds));
        part
    }

    /// The tie geometry of a part of `n` records, `2^d` positions per segment and `segments`
    /// segments, from the public numbers alone (the proof-of-work plan of event 0 needs it
    /// before the part exists; [`Self::new_span`] checks it against the part's own).
    #[must_use]
    pub fn plan_geometry(gate: &ChainGate, n: usize, d: u32, segments: usize) -> TieGeometry {
        TieGeometry {
            d: d as usize,
            span_log: segments.next_power_of_two().trailing_zeros() as usize,
            blocks_log: (d as usize).saturating_sub(BLOCK_RECORDS_LOG2),
            nu_s: capacity_log(n, gate.layout().k_log),
        }
    }

    /// The work of a part from the public numbers (as [`Self::plan_geometry`]) under
    /// `profile`: the tie, `eq_Y` and round difficulties and the round count.
    #[must_use]
    pub fn plan_work(
        gate: &ChainGate,
        n: usize,
        d: u32,
        segments: usize,
        profile: Profile,
    ) -> PartWork {
        part_work(
            profile.list_size(),
            &Self::plan_geometry(gate, n, d, segments),
        )
    }

    /// The tie site's difficulty from the public numbers.
    #[must_use]
    pub fn plan_tie_bits(
        gate: &ChainGate,
        n: usize,
        d: u32,
        segments: usize,
        profile: Profile,
    ) -> u8 {
        Self::plan_work(gate, n, d, segments, profile).tie_bits
    }

    /// The part's tie geometry.
    #[must_use]
    pub fn tie_geometry(&self) -> TieGeometry {
        TieGeometry {
            d: self.d as usize,
            span_log: self.chain.span_log,
            blocks_log: self.chain.mid_vars(),
            nu_s: self.sponge.nu(),
        }
    }

    /// The part's work (`docs/soundness.md`).
    #[must_use]
    pub fn work(&self) -> PartWork {
        part_work(self.profile.list_size(), &self.tie_geometry())
    }

    /// The tie site's difficulty `g_tie`.
    #[must_use]
    pub fn tie_bits(&self) -> u8 {
        self.work().tie_bits
    }

    /// Segments of the span (1 for a part of one segment).
    #[must_use]
    pub fn segments(&self) -> usize {
        self.segments
    }

    /// Live records of segment `sigma` of the span.
    #[must_use]
    pub fn live(&self, sigma: usize) -> usize {
        self.n.saturating_sub(sigma << self.d).min(1 << self.d)
    }

    /// u64 words of the lane-major input bits of one segment (six lanes).
    fn segment_words(&self) -> usize {
        self.segment().input_bits() / 64
    }

    /// Records in the part.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Always false.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// The part's input geometry (shape, lanes, `lane_len = 2^(10 + d)`, offset).
    #[must_use]
    pub fn segment(&self) -> &SegmentLayout {
        &self.chain.layout
    }

    /// The parity chain's geometry.
    #[must_use]
    pub fn chain(&self) -> &ChainLayout {
        &self.chain
    }

    /// `log2` of the part's record positions.
    #[must_use]
    pub fn d_part(&self) -> u32 {
        self.d
    }

    /// The sponge instance's row capacity `log2` (the `nu` of the sponge rows).
    #[must_use]
    pub fn nu(&self) -> usize {
        self.sponge.nu()
    }

    /// The sponge instance's geometry.
    #[must_use]
    pub fn sponge(&self) -> &UnionGeometry {
        &self.sponge
    }

    /// Sizes of the part.
    #[must_use]
    pub fn cost(&self) -> TiedCost {
        TiedCost {
            records: self.n,
            segments: self.segments,
            sponge_nu: self.sponge.nu(),
            sponge_m: self.sponge.m_total(),
            sponge_dense_m: self.sponge.dense_m(),
            sponge_committed_words: self.n * self.gate.layout().useful_bits.div_ceil(128),
            chain_rounds: self.chain.rounds(),
            chain_input_bits: 64 * self.chain.input_words(),
        }
    }

    /// SHA3-256 of the final vectors of the span (label, count, the packed vectors): bound
    /// before the tie challenges, so the chain's claim is fixed before `eq_Y`.
    #[must_use]
    pub fn y_digest(ys: &[Vec<u8>]) -> [u8; 32] {
        let mut h = Sha3_256::new();
        h.update((Y_TABLE_LABEL.len() as u64).to_le_bytes());
        h.update(Y_TABLE_LABEL);
        h.update((ys.len() as u64).to_le_bytes());
        for y in ys {
            h.update(pack_y(y));
        }
        h.finalize().into()
    }

    /// Digest of the part's geometry (bound into both challengers): `d`, the part index, the
    /// segments and the span, `n`, the sponge's row capacity and word coordinates, the
    /// chain's rounds and coordinate groups.
    fn tie_digest(&self) -> [u8; 32] {
        let mut h = Sha3_256::new();
        h.update(b"agg-keccak/v5/tie-geometry");
        for v in [
            self.d as usize,
            self.index,
            self.segments,
            self.chain.span_log,
            self.n,
            self.sponge.nu(),
            self.sponge.word_vars(),
            self.chain.rounds(),
            self.chain.zeta_vars(),
            self.chain.mid_vars(),
            self.chain.lane_vars(),
        ] {
            h.update((v as u64).to_le_bytes());
        }
        h.finalize().into()
    }

    /// The tie challenger before its first site: `fs`, the geometry, the final vectors'
    /// digest and the sponge cap.
    fn tie_start(&self, fs: &[u8; 32], y_digest: &[u8; 32], cap: &KeccakCommitment) -> FsChallenger {
        let mut ch = FsChallenger::new(TIE_DOMAIN);
        ch.observe_bytes(fs);
        ch.observe_bytes(&self.tie_digest());
        ch.observe_bytes(y_digest);
        put_framed(&mut ch, cap.as_bytes());
        ch
    }

    /// Digest of the sponge statement: `n` and the words `[mu_i, c~_i]`.
    fn public_digest(public: &[F128]) -> [u8; 32] {
        let mut h = Sha3_256::new();
        h.update(b"agg-keccak/v1/tied-public");
        h.update((public.len() as u64).to_le_bytes());
        for w in public {
            h.update(w.lo.to_le_bytes());
            h.update(w.hi.to_le_bytes());
        }
        h.finalize().into()
    }

    fn sponge_challenger(&self, fs: &[u8; 32], public: &[F128], binding: &[u8; 32]) -> FsChallenger {
        let mut ch = FsChallenger::new(DOMAIN);
        ch.observe_bytes(fs);
        ch.observe_label(SPONGE_LABEL);
        ch.observe_bytes(&self.gate.digest());
        ch.observe_bytes(&Self::public_digest(public));
        ch.observe_bytes(&self.tie_digest());
        ch.observe_label(TIE_BINDING_LABEL);
        put_framed(&mut ch, binding);
        ch
    }

    /// The public claim's point: rows `rho_r`, word columns `48 + c8` (`c8 = rho_c`).
    fn public_point(&self, rho_r: &[F128], rho_c: &[F128]) -> Vec<F128> {
        let nu_s = self.sponge.nu();
        let mut p = rho_r.to_vec();
        p.extend_from_slice(rho_c);
        for bit in PUBLIC_WORDS_LOG2..self.sponge.word_vars() - nu_s {
            p.push(if (PUBLIC_WORD_BASE >> bit) & 1 == 1 {
                F128::ONE
            } else {
                F128::ZERO
            });
        }
        p
    }

    /// The record coordinates of a chain point: `i_lo` (the in-block coordinates above the
    /// record position), `i_mid`, then `sigma` (`d + span_log` in all).
    fn record_coords(point: &ChainPoint) -> Vec<F128> {
        point.zeta[10..]
            .iter()
            .chain(&point.i_mid)
            .chain(&point.sigma)
            .copied()
            .collect()
    }

    /// `eq(r_rec[nu_s..], 0) = prod_k (1 + r_rec[k])` over the record coordinates beyond the
    /// sponge's rows: the factor between the chain's `b''~(r)` and the sponge's bit value
    /// when the sponge instance has `nu_s < d + span_log` row coordinates (its records `>=
    /// 2^nu_s` do not exist, and the chain's are zero).
    fn row_factor(&self, point: &ChainPoint) -> F128 {
        Self::record_coords(point)
            .iter()
            .skip(self.sponge.nu())
            .fold(F128::ONE, |acc, &x| acc * (F128::ONE + x))
    }

    /// The bit claim on the sponge commitment at a chain point: in-word bits
    /// 0..6 from `zeta[0..7]`, rows from the record coordinates (truncated or zero-padded to
    /// `nu_s`), word columns `[zeta[7..10] | t | 0 ..]` (word `8 t + w`).
    fn bit_claim(&self, point: &ChainPoint) -> BitClaim {
        let nu_s = self.sponge.nu();
        let rec = Self::record_coords(point);
        let mut x = Vec::with_capacity(self.sponge.word_vars() + 1);
        x.push(point.zeta[6]);
        x.extend((0..nu_s).map(|k| rec.get(k).copied().unwrap_or(F128::ZERO)));
        x.extend_from_slice(&point.zeta[7..10]);
        x.extend_from_slice(&point.t);
        x.resize(self.sponge.word_vars() + 1, F128::ZERO);
        BitClaim {
            skip: std::array::from_fn(|k| point.zeta[k]),
            x_full: x,
        }
    }

    /// The chain's input (`2^span_log` segments of eight lanes): lanes 0..5 from the part's
    /// input bits, lanes 6, 7 from the sponge's word columns 48..63 (module docs).
    fn chain_input(&self, bits: &[u64], z: &[F128]) -> Vec<u64> {
        let chain = &self.chain;
        let lane_words = self.segment().lane_len / 64;
        let seg_words = self.segment_words();
        let mut out = vec![0_u64; chain.input_words()];
        for sigma in 0..self.segments {
            for t in 0..self.segment().lanes() {
                let src = sigma * seg_words + t * lane_words;
                let dst = chain.bit_index(sigma, t, 0) / 64;
                out[dst..dst + lane_words].copy_from_slice(&bits[src..src + lane_words]);
            }
        }
        let nu = self.sponge.nu();
        for g in 0..self.n {
            let (sigma, i) = (g >> self.d, g & ((1 << self.d) - 1));
            for t in self.segment().lanes()..CHAIN_LANES {
                for w in 0..8 {
                    let word = z[((8 * t + w) << nu) | g];
                    let dst = chain.bit_index(sigma, t, 1024 * i + 128 * w) / 64;
                    out[dst] = word.lo;
                    out[dst + 1] = word.hi;
                }
            }
        }
        out
    }

    /// The honest sponge slot buffers of `records` (`n` of them) and the span's lane-major
    /// input bits, one block of `6 * 2^(10 + d) / 64` words per segment (segment-major; in a
    /// segment's block column `p + 2^(10 + d) t` is bit `t` of position `p = j + 256 rho +
    /// 1024 i`, padded records zero).
    #[must_use]
    pub fn honest_parts(&self, records: &[SpongeRecord]) -> (SlotBuffers, Vec<u64>) {
        assert_eq!(records.len(), self.n, "one record per live position");
        // Per record and independent: parallel (same values in the same order).
        let inputs: Vec<Vec<u64>> = records.par_iter().map(SpongeRecord::input_bits).collect();
        let sponge = pack_slot(self.gate, self.sponge.nu(), self.n, |j, z, a, b| {
            self.gate.instance_rows(&inputs[j], z, a, b);
        });
        let seg = self.segment();
        let words = self.segment_words();
        let mut bits = vec![0_u64; self.segments * words];
        let lane = seg.lane_len;
        for (g, input) in inputs.iter().enumerate() {
            let (sigma, i) = (g >> self.d, g & ((1 << self.d) - 1));
            // Lane t of a record is gate input bits W1_INPUT_BASE + 1024 t + (256 rho + j),
            // 16 aligned u64 words (`SpongeRecord::input_bits`).
            for t in 0..6 {
                let src = (W1_INPUT_BASE + 1024 * t) / 64;
                let dst = sigma * words + (lane * t + 1024 * i) / 64;
                bits[dst..dst + 16].copy_from_slice(&input[src..src + 16]);
            }
        }
        (sponge, bits)
    }

    fn statement(records: &[SpongeRecord]) -> (Vec<F128>, Vec<[u8; CTILDE_BYTES]>) {
        // One SHAKE256 per record: parallel, order preserved.
        let c_tilde: Vec<[u8; CTILDE_BYTES]> =
            records.par_iter().map(SpongeRecord::c_tilde).collect();
        let public = records
            .iter()
            .zip(&c_tilde)
            .flat_map(|(r, c)| sponge_public_words(&r.mu, c))
            .collect();
        (public, c_tilde)
    }

    /// Commit to given witnesses (tests commit tampered ones): the sponge buffers and the
    /// part's input bits; `records` give the statement words `[mu_i, c~_i]` (with `c~_i`
    /// computed natively). Keeps the sponge commitment's prover data.
    #[must_use]
    pub fn commit_parts(
        &self,
        sponge: SlotBuffers,
        bits: Vec<u64>,
        records: &[SpongeRecord],
    ) -> TiedCommitted {
        assert_eq!(records.len(), self.n, "one record per live position");
        assert_eq!(
            bits.len(),
            self.segments * self.segment_words(),
            "part input length"
        );
        let start = std::time::Instant::now();
        trace("commit: start", &start);
        let data = self.sponge.commit_data(&[&sponge.z]);
        trace("sponge committed", &start);
        let (public, c_tilde) = Self::statement(records);
        TiedCommitted {
            n: self.n,
            commitment: TiedCommitment {
                sponge: data.commitment(),
            },
            sponge,
            sponge_data: Some(data),
            bits,
            public,
            c_tilde,
        }
    }

    /// Commit to the honest witness of `records` (event 2: before `sd` and `fs`).
    #[must_use]
    pub fn commit(&self, records: &[SpongeRecord]) -> TiedCommitted {
        let (sponge, bits) = self.honest_parts(records);
        self.commit_parts(sponge, bits, records)
    }

    /// Rebuild a committed part from its witnesses and the commitment computed earlier by
    /// [`Self::commit_parts`] on the same witnesses (the driver drops the buffers between
    /// events 2 and 6), with the commitment's prover data if it was kept. Without
    /// the data the proof commits again and panics if Flock's cap differs from
    /// `commitment`.
    #[must_use]
    pub fn reattach(
        &self,
        sponge: SlotBuffers,
        bits: Vec<u64>,
        records: &[SpongeRecord],
        commitment: TiedCommitment,
        data: Option<CommitData>,
    ) -> TiedCommitted {
        assert_eq!(records.len(), self.n, "one record per live position");
        assert_eq!(
            bits.len(),
            self.segments * self.segment_words(),
            "part input length"
        );
        assert_eq!(
            commitment.sponge.as_bytes().len(),
            self.sponge.commitment_len()
        );
        if let Some(d) = &data {
            assert!(
                d.commitment() == commitment.sponge,
                "commit data of another commitment"
            );
        }
        let (public, c_tilde) = Self::statement(records);
        TiedCommitted {
            n: self.n,
            sponge,
            sponge_data: data,
            bits,
            public,
            c_tilde,
            commitment,
        }
    }

    /// Bytes of the event-2 commitment (the sponge cap).
    #[must_use]
    pub fn commitment_len(&self) -> usize {
        self.sponge.commitment_len()
    }

    /// The honest shares `M^{(sigma)} b'_sigma` of a committed part (one per segment of the
    /// span) under the matrices of `sd`, from the part's input bits.
    #[must_use]
    pub fn final_vectors(
        &self,
        committed: &TiedCommitted,
        matrices: &BridgeMatrices,
    ) -> Vec<Vec<u8>> {
        self.final_vectors_of_bits(&committed.bits, matrices)
    }

    /// [`Self::final_vectors`] from the part's lane-major input bits (as returned by
    /// [`Self::honest_parts`] and kept by [`TiedCommitted::bits`]), without the sponge
    /// witness: the event-6 shares of a segment's parts are computed before the Flock proofs
    /// (fork-join transcript: the Flock proofs run at a join after stage 3).
    ///
    /// # Panics
    /// If `bits` has another length than the part's input.
    #[must_use]
    pub fn final_vectors_of_bits(&self, bits: &[u64], matrices: &BridgeMatrices) -> Vec<Vec<u8>> {
        let words = self.segment_words();
        assert_eq!(bits.len(), self.segments * words, "part input length");
        (0..self.segments)
            .into_par_iter()
            .map(|sigma| {
                gf2_walker::composite(
                    self.segment(),
                    matrices,
                    &bits[sigma * words..(sigma + 1) * words],
                )
            })
            .collect()
    }

    /// Prove a committed part against the final vectors `ys` (one per segment of the span;
    /// tests pass wrong ones) under the matrices of `sd`, seeded by `fs` (the part's Flock
    /// seed): the tie site, `eq_Y`, the parity chain over the part's bits (lanes 0..5) and the
    /// sponge's word columns 48..63 (lanes 6, 7), the claimed bit value, then the sponge
    /// instance's Flock proof with the public claim and the bit claim (opening the event-2
    /// commitment). `sd` is bound through `fs` (event 5 follows event 3).
    ///
    /// # Errors
    /// [`KeccakError::Work`] if a nonce search exhausts its allowance of `2^(g + 7)`
    /// candidates (probability below `2^-184.6` per site, docs/soundness.md).
    ///
    /// # Panics
    /// If `committed` belongs to another part shape, `ys` has the wrong shape, or Flock
    /// commits to another sponge cap than the event-2 one.
    pub fn prove_with_final_vectors(
        &self,
        committed: TiedCommitted,
        _sd: [u8; 32],
        matrices: &BridgeMatrices,
        ys: Vec<Vec<u8>>,
        fs: &[u8; 32],
    ) -> Result<KeccakProof, KeccakError> {
        assert_eq!(committed.n, self.n, "committed part of another shape");
        assert_eq!(ys.len(), self.segments, "one final vector per segment");
        let m3 = self.segment().output_bits();
        assert!(
            ys.iter()
                .all(|y| y.len() == m3 && y.iter().all(|&b| b <= 1)),
            "final vectors are m_3-bit vectors"
        );
        let TiedCommitted {
            sponge,
            sponge_data,
            bits,
            public,
            commitment: absorbed,
            ..
        } = committed;
        let start = std::time::Instant::now();
        trace("prove: start", &start);
        let work = self.work();
        let nu_s = self.sponge.nu();
        let mut ch = self.tie_start(fs, &Self::y_digest(&ys), &absorbed.sponge);
        let nonce = site_work(&mut ch, work.tie_bits, TieWork::Grind, "tie nonce")?;
        let rho_r = ch.sample_f128_vec(nu_s);
        let rho_c = ch.sample_f128_vec(3);
        let eq_nonce = site_work(&mut ch, work.eq_bits, TieWork::Grind, "eq_Y nonce")?;
        let eq_y = EqY {
            r_o: ch.sample_f128_vec(self.chain.out_vars()),
            r_y: ch.sample_f128_vec(self.chain.span_log),
        };
        let claim = self.chain.claim(&ys, &eq_y);
        let input = self.chain_input(&bits, &sponge.z);
        drop(bits);
        trace("chain input", &start);
        let mut nonces = Vec::with_capacity(self.chain.rounds());
        let chain = self.chain.prove(matrices, &input, &eq_y, claim, |_, (q1, qinf)| {
            ch.observe_f128(q1);
            ch.observe_f128(qinf);
            nonces.push(site_work(
                &mut ch,
                work.round_bits,
                TieWork::Grind,
                "chain round nonce",
            )?);
            Ok(ch.sample_f128())
        })?;
        drop(input);
        trace("parity chain", &start);
        let bit_claim = self.bit_claim(&chain.point);
        let value = bit_value(&sponge.z, &bit_claim, nu_s);
        ch.observe_f128(value);
        let binding = squeeze_binding(&mut ch);
        trace("bit value", &start);
        let mut sch = self.sponge_challenger(fs, &public, &binding);
        let gate: [&dyn LincheckCircuit; 1] = [self.gate];
        let (sponge_proof, cap) = self.sponge.prove(
            vec![sponge],
            &gate,
            &[self.public_point(&rho_r, &rho_c)],
            &[bit_claim],
            &mut sch,
            sponge_data,
        );
        assert!(
            cap == absorbed.sponge,
            "Flock's prover committed to a different sponge cap than TiedPart::commit"
        );
        trace("sponge instance proven", &start);
        let rounds = chain
            .messages
            .iter()
            .zip(&nonces)
            .map(|(&(q1, qinf), &nonce)| ChainRound { q1, qinf, nonce })
            .collect();
        Ok(self.encode_parts(&TiedProofParts {
            nonce,
            eq_nonce,
            rounds,
            value,
            sponge: sponge_proof,
        }))
    }

    /// Compute the shares from the committed bits and prove (the honest prover).
    ///
    /// # Errors
    /// As [`Self::prove_with_final_vectors`].
    pub fn prove(
        &self,
        committed: TiedCommitted,
        sd: [u8; 32],
        matrices: &BridgeMatrices,
        fs: &[u8; 32],
    ) -> Result<TiedProof, KeccakError> {
        let ys = self.final_vectors(&committed, matrices);
        let proof = self.prove_with_final_vectors(committed, sd, matrices, ys.clone(), fs)?;
        Ok(TiedProof {
            final_vectors: ys,
            proof,
        })
    }

    /// The proof bytes of `parts`: `AGGKCT || 5 || profile || LE32(n) || d || S || LE32(tie
    /// nonce) || LE32(eq nonce) || rounds (q(1) || q(inf) || LE32(nonce)) || v || LE32(len) ||
    /// sponge proof`.
    #[must_use]
    pub fn encode_parts(&self, parts: &TiedProofParts) -> KeccakProof {
        let f128 = |out: &mut Vec<u8>, x: F128| {
            out.extend_from_slice(&x.lo.to_le_bytes());
            out.extend_from_slice(&x.hi.to_le_bytes());
        };
        let s = parts.sponge.to_bytes();
        let mut bytes =
            Vec::with_capacity(HEADER + 4 + ROUND_BYTES * parts.rounds.len() + 20 + s.len());
        bytes.extend_from_slice(MAGIC);
        bytes.push(VERSION);
        bytes.push(self.profile.id());
        bytes.extend_from_slice(&(self.n as u32).to_le_bytes());
        bytes.push(self.d as u8);
        bytes.push(u8::try_from(self.segments).expect("at most 255 segments"));
        bytes.extend_from_slice(&parts.nonce.to_le_bytes());
        bytes.extend_from_slice(&parts.eq_nonce.to_le_bytes());
        for r in &parts.rounds {
            f128(&mut bytes, r.q1);
            f128(&mut bytes, r.qinf);
            bytes.extend_from_slice(&r.nonce.to_le_bytes());
        }
        f128(&mut bytes, parts.value);
        bytes.extend_from_slice(
            &u32::try_from(s.len())
                .expect("proof below 4 GiB")
                .to_le_bytes(),
        );
        bytes.extend_from_slice(&s);
        KeccakProof { bytes }
    }

    /// Parse a proof of this part (the inverse of [`Self::encode_parts`]).
    ///
    /// # Errors
    /// On a malformed proof or one of another version, profile or part shape.
    pub fn decode_parts(&self, proof: &KeccakProof) -> Result<TiedProofParts, KeccakError> {
        let b = &proof.bytes;
        if b.len() < HEADER || &b[..6] != MAGIC {
            return Err(KeccakError::Malformed("magic"));
        }
        if b[6] != VERSION {
            return Err(KeccakError::Malformed("version"));
        }
        if b[7] != self.profile.id() {
            return Err(KeccakError::Malformed("profile"));
        }
        if u32::from_le_bytes(b[8..12].try_into().expect("4 bytes")) as usize != self.n
            || b[12] != self.d as u8
            || usize::from(b[13]) != self.segments
        {
            return Err(KeccakError::Malformed("instance count"));
        }
        let le32 = |at: usize| u32::from_le_bytes(b[at..at + 4].try_into().expect("4 bytes"));
        let f128 = |at: usize| {
            F128::new(
                u64::from_le_bytes(b[at..at + 8].try_into().expect("8 bytes")),
                u64::from_le_bytes(b[at + 8..at + 16].try_into().expect("8 bytes")),
            )
        };
        let rounds = self.chain.rounds();
        let chain_end = HEADER + 4 + ROUND_BYTES * rounds + 16;
        if b.len() < chain_end + 4 {
            return Err(KeccakError::Malformed("framing"));
        }
        let len = le32(chain_end) as usize;
        if b.len() != chain_end + 4 + len {
            return Err(KeccakError::Malformed("framing"));
        }
        let rounds = (0..rounds)
            .map(|k| {
                let at = HEADER + 4 + ROUND_BYTES * k;
                ChainRound {
                    q1: f128(at),
                    qinf: f128(at + 16),
                    nonce: le32(at + 32),
                }
            })
            .collect();
        Ok(TiedProofParts {
            nonce: le32(NONCE_OFFSET),
            eq_nonce: le32(HEADER),
            rounds,
            value: f128(chain_end - 16),
            sponge: UnionProof::from_bytes(&b[chain_end + 4..], SPONGE_CLAIMS)?,
        })
    }

    /// The verifier's value of the public claim: the multilinear extension of the public
    /// region's words (mu, `c~`, the constant 1, zero; zero on dummy rows) at the point.
    fn public_value(&self, public: &[F128], rho_r: &[F128], rho_c: &[F128]) -> F128 {
        let words = 6; // mu (4) then c~ (2) per record, as `sponge_public_words`
        assert_eq!(public.len(), words * self.n);
        let eq_rows = build_eq_table(rho_r);
        let eq_c = build_eq_table(rho_c);
        let mu_at = MU_WORD_BASE - PUBLIC_WORD_BASE;
        let ct_at = CTILDE_WORD_BASE - PUBLIC_WORD_BASE;
        let const_at = CONST_WORD - PUBLIC_WORD_BASE;
        let mut acc = F128::ZERO;
        for (j, row) in eq_rows.iter().enumerate().take(self.n) {
            let rec = &public[words * j..words * (j + 1)];
            let mut v = eq_c[const_at] * F128::ONE;
            for (k, &w) in rec[..4].iter().enumerate() {
                v += eq_c[mu_at + k] * w;
            }
            for (k, &w) in rec[4..].iter().enumerate() {
                v += eq_c[ct_at + k] * w;
            }
            acc += *row * v;
        }
        acc
    }

    /// Verify a part proof against `(mu_i, c~_i)` of the span's records, the event-2
    /// commitment, the bridge seed and its matrices, the final vectors `ys` (one per segment
    /// of the span) and the part's Flock seed `fs`.
    ///
    /// # Errors
    /// On a malformed statement, proof or commitment, a Flock rejection, a rejected nonce, or
    /// a failed chain or public check (`KeccakError::Tie` with `"parity chain"`, `"public
    /// words"`).
    #[allow(clippy::too_many_arguments)]
    pub fn verify(
        &self,
        mu: &[[u8; MU_BYTES]],
        c_tilde: &[[u8; CTILDE_BYTES]],
        commitment: &TiedCommitment,
        _sd: [u8; 32],
        matrices: &BridgeMatrices,
        ys: &[Vec<u8>],
        fs: &[u8; 32],
        proof: &KeccakProof,
    ) -> Result<(), KeccakError> {
        if mu.len() != self.n || c_tilde.len() != self.n {
            return Err(KeccakError::Statement("record count"));
        }
        let m3 = self.segment().output_bits();
        if ys.len() != self.segments || ys.iter().any(|y| y.len() != m3 || y.iter().any(|&b| b > 1))
        {
            return Err(KeccakError::Statement("final vector"));
        }
        if !self.segment().shape.matches(matrices) {
            return Err(KeccakError::Statement("bridge matrices"));
        }
        if commitment.sponge.as_bytes().len() != self.sponge.commitment_len() {
            return Err(KeccakError::Malformed("commitment length"));
        }
        let start = std::time::Instant::now();
        let parts = self.decode_parts(proof)?;
        let public: Vec<F128> = mu
            .iter()
            .zip(c_tilde)
            .flat_map(|(m, c)| sponge_public_words(m, c))
            .collect();
        let work = self.work();
        let nu_s = self.sponge.nu();
        let mut ch = self.tie_start(fs, &Self::y_digest(ys), &commitment.sponge);
        site_work(&mut ch, work.tie_bits, TieWork::Check(parts.nonce), "tie nonce")?;
        let rho_r = ch.sample_f128_vec(nu_s);
        let rho_c = ch.sample_f128_vec(3);
        site_work(
            &mut ch,
            work.eq_bits,
            TieWork::Check(parts.eq_nonce),
            "eq_Y nonce",
        )?;
        let eq_y = EqY {
            r_o: ch.sample_f128_vec(self.chain.out_vars()),
            r_y: ch.sample_f128_vec(self.chain.span_log),
        };
        let claim = self.chain.claim(ys, &eq_y);
        let messages: Vec<(F128, F128)> = parts.rounds.iter().map(|r| (r.q1, r.qinf)).collect();
        let (point, last) = self.chain.replay(claim, &messages, |k, (q1, qinf)| {
            ch.observe_f128(q1);
            ch.observe_f128(qinf);
            site_work(
                &mut ch,
                work.round_bits,
                TieWork::Check(parts.rounds[k].nonce),
                "chain round nonce",
            )?;
            Ok::<_, KeccakError>(ch.sample_f128())
        })?;
        let weight = self.chain.weight_at(matrices, &eq_y, &point);
        if last != weight * self.row_factor(&point) * parts.value {
            return Err(KeccakError::Tie("parity chain"));
        }
        ch.observe_f128(parts.value);
        let binding = squeeze_binding(&mut ch);
        trace("verify: parity chain", &start);
        let mut sch = self.sponge_challenger(fs, &public, &binding);
        let gate: [&dyn LincheckCircuit; 1] = [self.gate];
        let v = self.sponge.verify(
            &gate,
            &commitment.sponge,
            &parts.sponge,
            &[self.public_point(&rho_r, &rho_c)],
            &[(self.bit_claim(&point), parts.value)],
            &mut sch,
        )?;
        if v[0] != self.public_value(&public, &rho_r, &rho_c) {
            return Err(KeccakError::Tie("public words"));
        }
        trace("verify: sponge instance", &start);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flock_union::eval_packed;
    use crate::sponge::sponge_gate;
    use agg_bridge::seed::derive_matrices_for;
    use agg_bridge::shape::Shape;
    use flock_core::test_rng::Rng;

    /// An exhausted nonce search is an error, not a panic (docs/soundness.md): the
    /// smallest passing nonce at or above `2^(g + 7)` is refused, the largest admitted one is
    /// kept.
    #[test]
    fn exhausted_tie_nonce_search_is_an_error() {
        for g in [1_u8, 16, 18, 25] {
            let limit = 1_u64 << nonce_bits(g);
            assert_eq!(nonce_bits(g), g + 7);
            assert!(matches!(nonce_within(0, limit), Ok(0)));
            assert!(matches!(nonce_within(limit - 1, limit), Ok(n) if u64::from(n) == limit - 1));
            for nonce in [limit, limit + 1, u64::MAX] {
                assert!(matches!(
                    nonce_within(nonce, limit),
                    Err(KeccakError::Work(_))
                ));
            }
        }
    }

    fn random_record(rng: &mut Rng) -> SpongeRecord {
        let mut mu = [0u8; MU_BYTES];
        let mut w1 = [0u8; 768];
        mu.iter_mut().for_each(|b| *b = rng.next_u32() as u8);
        w1.iter_mut().for_each(|b| *b = rng.next_u32() as u8);
        SpongeRecord { mu, w1_encode: w1 }
    }

    fn rand_f(rng: &mut Rng) -> F128 {
        F128::new(rng.next_u64(), rng.next_u64())
    }

    /// The chain's input and its bit claim on the sponge commitment agree with the sponge's
    /// committed words: at random chain points `b''~(r)` of the chain input equals the row
    /// factor times the sponge's bit value at the claim, for parts whose sponge has more,
    /// fewer or exactly as many row coordinates as the part has record positions, a part of
    /// two level-1 blocks, parts with an offset and spans; the public value equals the
    /// verifier's table; the honest final vectors are the F_p-side walk's.
    #[test]
    fn the_bit_claim_is_the_chain_input_on_the_sponge() {
        let gate = sponge_gate();
        let mats = derive_matrices_for(&Shape::PRODUCTION, &[5; 32]);
        for (n, d, index, segments) in [
            (5usize, 3u32, 0usize, 1usize),
            (8, 3, 0, 1),
            (9, 6, 2, 1),
            (1, 6, 0, 1),
            (3, 10, 0, 1),
            (12, 3, 0, 2),
            (19, 3, 1, 3),
            (70, 6, 0, 2),
        ] {
            let part = TiedPart::new_span(&gate, n, d, index, segments, Profile::Fast);
            let mut rng = Rng::new(100 + n as u64 + u64::from(d));
            let records: Vec<SpongeRecord> = (0..n).map(|_| random_record(&mut rng)).collect();
            let (sponge, bits) = part.honest_parts(&records);
            let input = part.chain_input(&bits, &sponge.z);
            let chain = part.chain();
            for _ in 0..3 {
                let rs: Vec<F128> = (0..chain.rounds()).map(|_| rand_f(&mut rng)).collect();
                let point = chain.split_point(&rs);
                let claim = part.bit_claim(&point);
                assert_eq!(claim.x_full.len(), part.sponge().word_vars() + 1);
                assert_eq!(
                    chain.input_value(&input, &point),
                    part.row_factor(&point) * bit_value(&sponge.z, &claim, part.nu()),
                    "n {n}, d {d}, index {index}, segments {segments}"
                );
            }
            // The six lanes of the chain input are the bits, lanes 6, 7 the public and chi
            // words of the records.
            let (public, _) = TiedPart::statement(&records);
            let rho_r: Vec<F128> = (0..part.nu()).map(|_| rand_f(&mut rng)).collect();
            let rho_c: Vec<F128> = (0..3).map(|_| rand_f(&mut rng)).collect();
            let p = part.public_point(&rho_r, &rho_c);
            assert_eq!(
                eval_packed(&sponge.z, &p, part.nu()),
                part.public_value(&public, &rho_r, &rho_c)
            );
            let committed = part.commit_parts(sponge, bits, &records);
            let ys = part.final_vectors(&committed, &mats);
            assert_eq!(ys.len(), segments);
            for (sigma, y) in ys.iter().enumerate() {
                if part.live(sigma) == 0 {
                    assert!(y.iter().all(|&b| b == 0));
                }
            }
        }
    }

    /// A prover that re-grinds every nonce after changing a round message (so every work
    /// site of the tie challenger accepts) still fails the chain's final check: the last
    /// round's message, a middle round's message (the later challenges change, the later
    /// messages stay) and the claimed bit value are rejected with `"parity chain"`; the
    /// re-grind of the honest parts reproduces the honest proof (the smallest nonces).
    #[test]
    fn regrinding_after_a_tampered_round_still_fails_the_chain() {
        let gate = sponge_gate();
        let n = 5;
        let part = TiedPart::new_part(&gate, n, 3, 0, Profile::Fast);
        let mut rng = Rng::new(0x7e57);
        let records: Vec<SpongeRecord> = (0..n).map(|_| random_record(&mut rng)).collect();
        let committed = part.commit(&records);
        let commitment = committed.commitment().clone();
        let sd = [9; 32];
        let mats = derive_matrices_for(&Shape::PRODUCTION, &sd);
        let fs = [3; 32];
        let out = part.prove(committed, sd, &mats, &fs).expect("prove");
        let mu: Vec<[u8; MU_BYTES]> = records.iter().map(|r| r.mu).collect();
        let ct: Vec<[u8; CTILDE_BYTES]> = records.iter().map(SpongeRecord::c_tilde).collect();
        let ys = &out.final_vectors;
        let verify = |p: &KeccakProof| part.verify(&mu, &ct, &commitment, sd, &mats, ys, &fs, p);
        verify(&out.proof).expect("honest");
        let regrind = |parts: &mut TiedProofParts| {
            let work = part.work();
            let mut ch = part.tie_start(&fs, &TiedPart::y_digest(ys), &commitment.sponge);
            parts.nonce = site_work(&mut ch, work.tie_bits, TieWork::Grind, "tie").unwrap();
            let _ = ch.sample_f128_vec(part.nu());
            let _ = ch.sample_f128_vec(3);
            parts.eq_nonce = site_work(&mut ch, work.eq_bits, TieWork::Grind, "eq").unwrap();
            let _ = ch.sample_f128_vec(part.chain().out_vars());
            let _ = ch.sample_f128_vec(part.chain().span_log);
            for r in &mut parts.rounds {
                ch.observe_f128(r.q1);
                ch.observe_f128(r.qinf);
                r.nonce = site_work(&mut ch, work.round_bits, TieWork::Grind, "round").unwrap();
                let _ = ch.sample_f128();
            }
        };
        let honest = part.decode_parts(&out.proof).expect("parse");
        let mut again = honest.clone();
        regrind(&mut again);
        assert_eq!(again, honest, "the re-grind reproduces the honest nonces");
        let rounds = honest.rounds.len();
        for (label, k) in [("last round", rounds - 1), ("middle round", rounds / 2)] {
            let mut bad = honest.clone();
            bad.rounds[k].qinf += F128::ONE;
            regrind(&mut bad);
            assert!(
                matches!(
                    verify(&part.encode_parts(&bad)),
                    Err(KeccakError::Tie("parity chain"))
                ),
                "{label}"
            );
        }
        let mut bad = honest.clone();
        bad.value += F128::ONE;
        assert!(matches!(
            verify(&part.encode_parts(&bad)),
            Err(KeccakError::Tie("parity chain"))
        ));
        // An eq_Y nonce one below the honest (smallest passing) one fails its site.
        if honest.eq_nonce > 0 {
            let mut bad = honest.clone();
            bad.eq_nonce -= 1;
            assert!(matches!(
                verify(&part.encode_parts(&bad)),
                Err(KeccakError::Tie("eq_Y nonce"))
            ));
        }
    }

    /// The sponge instance at a full-height count (2,048 records: Flock's block-first
    /// transport) verifies with a bit claim whose column coordinates are zero above the
    /// chain's 64 word columns, and rejects it with a false value (ring-switch `ClaimMismatch`). Regression: the block rounds
    /// fold the column bit-bank's multilinear extension, so a bank computed only on the
    /// columns of nonzero weight (enough for `s_hat_v`) makes the honest proof fail.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "Flock proofs; run with --release")]
    fn bit_claims_verify_on_a_full_height_sponge() {
        let gate = sponge_gate();
        let n = 2048;
        let part = TiedPart::new_part(&gate, n, 11, 0, Profile::Fast);
        let mut rng = Rng::new(77);
        let records: Vec<SpongeRecord> = (0..n).map(|_| random_record(&mut rng)).collect();
        let nu = part.nu();
        let wv = part.sponge().word_vars();
        let rand_claim = |rng: &mut Rng, zero_cols_from: usize| {
            let mut x: Vec<F128> = (0..=wv).map(|_| rand_f(rng)).collect();
            for v in x.iter_mut().skip(1 + nu + zero_cols_from) {
                *v = F128::ZERO;
            }
            BitClaim {
                skip: std::array::from_fn(|_| rand_f(rng)),
                x_full: x,
            }
        };
        let label = "64 columns";
        let claims = vec![rand_claim(&mut rng, 6)];
        let (sponge, _) = part.honest_parts(&records);
        let data = part.sponge().commit_data(&[&sponge.z]);
        let cap = data.commitment();
        let values: Vec<F128> = claims.iter().map(|c| bit_value(&sponge.z, c, nu)).collect();
        let rho_r: Vec<F128> = (0..nu).map(|_| rand_f(&mut rng)).collect();
        let pubp = part.public_point(&rho_r, &[F128::ONE, F128::ZERO, F128::ONE]);
        let mut ch = FsChallenger::new(b"debug");
        let gates: [&dyn LincheckCircuit; 1] = [part.gate];
        let (proof, _) = part.sponge().prove(
            vec![sponge],
            &gates,
            std::slice::from_ref(&pubp),
            &claims,
            &mut ch,
            Some(data),
        );
        let mut ch = FsChallenger::new(b"debug");
        let mut with: Vec<(BitClaim, F128)> = claims.iter().cloned().zip(values).collect();
        let pubs = std::slice::from_ref(&pubp);
        let r = part.sponge().verify(&gates, &cap, &proof, pubs, &with, &mut ch);
        assert!(r.is_ok(), "{label}: {r:?}");
        // A false bit value: the ring-switch claim check of the block-first opening
        // rejects it.
        if let Some(first) = with.first_mut() {
            first.1 += F128::ONE;
            let mut ch = FsChallenger::new(b"debug");
            let r = part
                .sponge()
                .verify(&gates, &cap, &proof, &[pubp], &with, &mut ch);
            assert!(claim_mismatch(&r), "{label}, false value: {r:?}");
        }
    }

    /// Flock's ring-switch claim check rejected (a false value of a ring-switched claim).
    fn claim_mismatch<T>(r: &Result<T, KeccakError>) -> bool {
        matches!(r, Err(KeccakError::Flock(e)) if format!("{e:?}").contains("ClaimMismatch"))
    }

    /// [`TiedPart::prove_with_final_vectors`] with one change: the bit value sent is the
    /// chain input's `b''~(r)` divided by the row factor, not the sponge's bit value. The two
    /// agree for an honest prover; for a chain copy of `b'` that differs from the sponge they
    /// do not, and the chain's final check then passes while the bit claim is false.
    fn prove_with_the_chains_bit_value(
        part: &TiedPart<'_>,
        committed: TiedCommitted,
        matrices: &BridgeMatrices,
        ys: &[Vec<u8>],
        fs: &[u8; 32],
    ) -> KeccakProof {
        let TiedCommitted {
            sponge,
            sponge_data,
            bits,
            public,
            commitment: absorbed,
            ..
        } = committed;
        let work = part.work();
        let nu_s = part.sponge.nu();
        let mut ch = part.tie_start(fs, &TiedPart::y_digest(ys), &absorbed.sponge);
        let nonce = site_work(&mut ch, work.tie_bits, TieWork::Grind, "tie").unwrap();
        let rho_r = ch.sample_f128_vec(nu_s);
        let rho_c = ch.sample_f128_vec(3);
        let eq_nonce = site_work(&mut ch, work.eq_bits, TieWork::Grind, "eq_Y").unwrap();
        let eq_y = EqY {
            r_o: ch.sample_f128_vec(part.chain.out_vars()),
            r_y: ch.sample_f128_vec(part.chain.span_log),
        };
        let claim = part.chain.claim(ys, &eq_y);
        let input = part.chain_input(&bits, &sponge.z);
        let mut nonces = Vec::new();
        let chain = part
            .chain
            .prove(matrices, &input, &eq_y, claim, |_, (q1, qinf)| {
                ch.observe_f128(q1);
                ch.observe_f128(qinf);
                nonces.push(site_work(&mut ch, work.round_bits, TieWork::Grind, "round")?);
                Ok::<F128, KeccakError>(ch.sample_f128())
            })
            .unwrap();
        let bit_claim = part.bit_claim(&chain.point);
        let value = chain.value * part.row_factor(&chain.point).inv();
        ch.observe_f128(value);
        let binding = squeeze_binding(&mut ch);
        let mut sch = part.sponge_challenger(fs, &public, &binding);
        let gate: [&dyn LincheckCircuit; 1] = [part.gate];
        let (sponge_proof, _) = part.sponge.prove(
            vec![sponge],
            &gate,
            &[part.public_point(&rho_r, &rho_c)],
            &[bit_claim],
            &mut sch,
            sponge_data,
        );
        let rounds = chain
            .messages
            .iter()
            .zip(&nonces)
            .map(|(&(q1, qinf), &nonce)| ChainRound { q1, qinf, nonce })
            .collect();
        part.encode_parts(&TiedProofParts {
            nonce,
            eq_nonce,
            rounds,
            value,
            sponge: sponge_proof,
        })
    }

    /// A false bit value that is consistent with the chain is rejected by Flock's ring-switch
    /// claim check: the chain's copy of `b'` has one bit flipped (in lane 3 of the last
    /// record), `Y` is the copy's, and the prover sends the copy's bit value, so every chain
    /// round and the chain's final check pass and only the sponge opening's third
    /// ring-switched claim (the bit claim) can fail. The honest variant of the same prover
    /// reproduces the honest proof (counts below 2,048).
    fn chain_consistent_false_bit_value(n: usize) {
        let gate = sponge_gate();
        {
            let part = TiedPart::new(&gate, n, TiedPart::d_seg_for(n), Profile::Fast);
            let mut rng = Rng::new(0xFA15E + n as u64);
            let records: Vec<SpongeRecord> = (0..n).map(|_| random_record(&mut rng)).collect();
            let sd = [11; 32];
            let mats = derive_matrices_for(&Shape::PRODUCTION, &sd);
            let fs = [12; 32];
            let mu: Vec<[u8; MU_BYTES]> = records.iter().map(|r| r.mu).collect();
            let ct: Vec<[u8; CTILDE_BYTES]> =
                records.iter().map(SpongeRecord::c_tilde).collect();
            if n < 2048 {
                let honest = part.prove(part.commit(&records), sd, &mats, &fs).expect("prove");
                let committed = part.commit(&records);
                let again = prove_with_the_chains_bit_value(
                    &part,
                    committed,
                    &mats,
                    &honest.final_vectors,
                    &fs,
                );
                assert_eq!(again, honest.proof, "n {n}: the replica is the honest prover");
            }
            let (sponge, mut bits) = part.honest_parts(&records);
            let col = part.segment().column(3, 1024 * (n - 1) + 77);
            bits[col / 64] ^= 1 << (col % 64);
            let committed = part.commit_parts(sponge, bits, &records);
            let commitment = committed.commitment().clone();
            let ys = part.final_vectors(&committed, &mats);
            let proof = prove_with_the_chains_bit_value(&part, committed, &mats, &ys, &fs);
            let r = part.verify(&mu, &ct, &commitment, sd, &mats, &ys, &fs, &proof);
            assert!(claim_mismatch(&r), "n {n}: {r:?}");
        }
    }

    /// Counts 5 and 64: the plain opening.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "Flock proofs; run with --release")]
    fn a_chain_consistent_false_bit_value_fails_the_bit_claim() {
        let t = std::time::Instant::now();
        chain_consistent_false_bit_value(5);
        chain_consistent_false_bit_value(64);
        eprintln!("5 and 64: {:.2?}", t.elapsed());
    }

    /// Count 2,048 (`n = 2^nu_s`, a full-height sponge): Flock's block-first opening. Slow;
    /// the suite pins the block-first rejection of a false bit value in
    /// `bit_claims_verify_on_a_full_height_sponge`.
    #[test]
    #[ignore = "slow; run with --release --ignored"]
    fn a_chain_consistent_false_bit_value_fails_the_block_first_bit_claim() {
        let t = std::time::Instant::now();
        chain_consistent_false_bit_value(2048);
        eprintln!("2048: {:.2?}", t.elapsed());
    }

    #[test]
    fn work_follows_the_profile_lists() {
        // docs/soundness.md: tie L_s (nu_s + 3), eq_Y
        // L_s (12 + span_log), rounds 2 L_s.
        let gate = sponge_gate();
        let profiles = [Profile::Fast, Profile::Slim, Profile::Slim8];
        assert_eq!(profiles.map(Profile::list_size), [35, 50, 70]);
        assert_eq!(Profile::Secure.list_size(), 1);
        let four = profiles.map(|p| TiedPart::plan_work(&gate, 1 << 15, 13, 4, p));
        assert_eq!(four.map(|w| w.tie_bits), [10, 10, 11]);
        assert_eq!(four.map(|w| w.eq_bits), [9, 10, 10]);
        assert_eq!(four.map(|w| w.round_bits), [7, 7, 8]);
        assert_eq!(four.map(|w| w.rounds), [28; 3]);
        let span = TiedPart::new_span(&gate, 3 << 13, 13, 0, 3, Profile::Slim);
        assert_eq!(span.tie_geometry(), TiedPart::plan_geometry(&gate, 3 << 13, 13, 3));
        assert_eq!(span.work(), four[1]);
        assert_eq!(span.chain().rounds(), 28);
        let small = TiedPart::new_part(&gate, 5, 3, 0, Profile::Fast);
        assert_eq!((small.nu(), small.chain().rounds()), (3, 16));
        // n = 256 = 2^8 below 2^11 rows: nu_s = 9 > d = 8 (capacity_log), so the claim's
        // ninth row coordinate is zero.
        let wide = TiedPart::new_part(&gate, 256, 8, 0, Profile::Fast);
        assert_eq!(wide.nu(), 9);
        assert_eq!(wide.chain().rounds(), 21);
    }

    #[test]
    fn tied_commitment_bytes_round_trip() {
        let c = TiedCommitment {
            sponge: KeccakCommitment::from_bytes(vec![1; 64]),
        };
        let bytes = c.to_bytes();
        assert_eq!(bytes.len(), 4 + 64);
        assert_eq!(TiedCommitment::from_bytes(&bytes).unwrap(), c);
        assert!(TiedCommitment::from_bytes(&bytes[..bytes.len() - 1]).is_err());
        let mut trailing = bytes;
        trailing.push(0);
        assert!(TiedCommitment::from_bytes(&trailing).is_err());
        assert!(TiedCommitment::from_bytes(&[1, 0]).is_err());
    }
}
