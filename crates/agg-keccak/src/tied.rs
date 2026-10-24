//! The tied part: the Flock argument of one part of a segment as two Flock instances, the
//! sponge instance (the `n` live records' SHAKE256 gates, docs/keccak.md Sections 1 to 4)
//! and the bridge union (the level-1 rows and the levels-2-3 row of
//! [`crate::bridge_union`], docs/bridge.md Section 5b, deviation K13), tied to each other and
//! to the public words by evaluation claims on their committed polynomials
//! ([`crate::flock_union`]; docs/keccak.md Sections 9 to 11, docs/bridge.md Sections 5a and
//! 5b, deviations K11, K13, K14). It replaces the two-type union of [`crate::segment`]
//! (docs/keccak.md Section 8) and, since proof version 2, the one-row bridge slot of
//! [`crate::bridge_slot`] (docs/keccak.md Section 10).
//!
//! Regions. Both instances commit their words BatchMajor: word column `c` of row `j` of a
//! slot is the padded word `(prefix << (nu + k_log - 7)) | (c << nu) | j`. In the sponge
//! block (`nu_s` rows, one slot) lane `t` of record `j` is the eight words `8 t + w`, `w < 8`
//! ([`crate::sponge::W1_WORD_BASE`]), so lanes 0..3 are the word columns `[0, 32)` and lanes
//! 4, 5 the columns `[32, 48)`: two aligned subcubes of the column index `[w (3 bits) | t |
//! high bits]`. The public region of the sponge block is the word columns `[48, 56)`: mu
//! (4 words), `c~` (2), the constant (1, the word with bit 0 set), one zero word. In the
//! bridge union, level-1 row `j = beta + blocks t` holds lane `t`, level-1 block `beta` of
//! the part: record `i` (`i = i_lo + 2^9 beta`) is its input words `8 i_lo + w`, so a lane
//! group is the subcube rows `[beta | t (3 bits)]`, columns `[w (3) | i_lo]`; the row's
//! level-1 parities are its 64 words from `y1_base / 128`, and their copies in the
//! levels-2-3 row are the words `y + 64 beta + 64 blocks t`.
//!
//! Claims. From the part's Flock seed `fs` and the two commitments both sides derive `rho =
//! (rho_w, rho_i, rho_t, rho_t', rho_r, rho_c, rho_beta, rho_y)` ([`TieChallenges`]) and the
//! points of [`TiePoints`]. The sponge proof carries `v_pub = z_s(P_pub)` and the lane-group
//! values `v_s_h = z_s(P_s_h)`; the bridge union's proof carries the level-1 input values
//! `v_in_h`, the level-1 parity values `v_y1_h`, their copies' values `v_top_h` and the
//! value `v_out` of the levels-2-3 output columns, each the committed multilinear (over
//! words) at the point, bound by Flock's merged opening. The verifier checks `v_pub` against
//! the table it builds from `(mu_i, c~_i)`, `v_in_h = eq(rho_i[nu_s..d], 0) v_s_h` (tie 1:
//! the level-1 inputs are the sponge inputs; the factor is 1 unless the sponge instance has
//! fewer row coordinates than the part has record positions), `v_top_h = v_y1_h` (tie 2:
//! the levels-2-3 inputs are the level-1 parities) for both lane groups `h`, and `v_out = 0`
//! (the output columns are zero, so the composite rows say `M_3 M_2 y1 = Y`). A differing
//! (nonzero) word makes the restricted multilinears differ as polynomials in their free
//! coordinates (`d + 5` for tie 1, `log2(blocks) + 9` for tie 2, 5 for the outputs), so they
//! agree at `rho` with probability at most that number over `2^128`.
//!
//! Order (deviations K7, K13, K14). [`TiedPart::commit`] fixes the sponge witness and the
//! part's lane-major input bits and returns the sponge cap (event 2), keeping Flock's
//! prover data. After `sd` (event 3) the level-1 parities exist; after `fs` (event 5)
//! [`TiedPart::prove`] builds the bridge union's blocks, commits them (its cap is part of the
//! proof), derives the tie challenges from `fs` and both caps, proves the sponge instance
//! (opening the event-2 commitment without recommitting) and then the bridge union.
//! [`TiedPart::verify`] replays both with its own points, matrices and share.

use agg_bridge::gf2_walker::SegmentLayout;
use flock_core::{
    challenger::{Challenger, FsChallenger},
    lincheck::{build_eq_table, LincheckCircuit},
};
use flock_field::F128;
use sha3::{Digest, Sha3_256};

use agg_witness::bridge::BridgeMatrices;

use crate::{
    bridge_union::{BridgeBlocks, BridgeUnion, BridgeUnionCost, Level1Layout},
    flock_union::{CommitData, UnionGeometry, UnionProof},
    gate::ChainGate,
    prover::{
        capacity_log, sponge_public_words, KeccakCommitment, KeccakError, KeccakProof, Profile,
        DOMAIN,
    },
    segment::BridgedProof,
    sponge::{
        SpongeRecord, CHI_WORD_BASE, CONST_WORD, CTILDE_BYTES, CTILDE_WORD_BASE, MU_BYTES,
        MU_WORD_BASE, PUBLIC_WORDS_LOG2, PUBLIC_WORD_BASE, W1_INPUT_BASE, W1_WORD_BASE,
    },
    witness::{pack_slot, SlotBuffers},
};

/// Magic prefix of a tied-part proof.
pub const MAGIC: &[u8; 6] = b"AGGKCT";
/// Serialization version (2: the bridge union of K13, its cap inside the proof).
const VERSION: u8 = 2;
/// Header: magic, version, profile, `n` (LE32), `d`.
const HEADER: usize = 6 + 1 + 1 + 4 + 1;
/// Domain of the tie challenges.
pub const TIE_DOMAIN: &[u8] = b"agg-mldsa/v2/tie";
/// Challenger labels of the two instances.
const SPONGE_LABEL: &[u8] = b"agg-keccak/v2/tied-sponge";
const BRIDGE_LABEL: &[u8] = b"agg-keccak/v2/tied-bridge-union";
/// Record positions per production level-1 block (`log2`): `2^19 / 1024`.
const BLOCK_RECORDS_LOG2: usize = 9;
/// Lane groups tied by one claim each: lanes 0..3 (two `t` coordinates) and lanes 4, 5 (one).
const LANE_GROUPS: usize = 2;
/// Claims of the sponge proof: the public region, then the two lane groups.
const SPONGE_CLAIMS: usize = 1 + LANE_GROUPS;
/// Claims of the bridge union's proof: level-1 inputs, level-1 parities, their copies, each
/// per lane group, then the levels-2-3 output columns.
const BRIDGE_CLAIMS: usize = 3 * LANE_GROUPS + 1;
/// Output words of the levels-2-3 row (`log2(m_3 / 128)`).
const OUTPUT_WORDS_LOG2: usize = 5;

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

/// The sponge gate's block layout must be the one the tie regions assume
/// (`crate::sponge`, module docs).
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
            && W1_WORD_BASE == 0,
        "the tied part needs the sponge layout of crate::sponge"
    );
}

fn put_framed(h: &mut impl Challenger, bytes: &[u8]) {
    h.observe_bytes(&(bytes.len() as u64).to_le_bytes());
    h.observe_bytes(bytes);
}

/// The random coordinates of the tie, derived from the part's Flock seed and the two
/// commitments.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TieChallenges {
    /// The word within a lane (3 coordinates).
    pub rho_w: [F128; 3],
    /// The record within the part (`d` coordinates).
    pub rho_i: Vec<F128>,
    /// The lane within lanes 0..3 (2 coordinates).
    pub rho_t: [F128; 2],
    /// The lane within lanes 4, 5 (1 coordinate).
    pub rho_t1: F128,
    /// The row of the public claim (`nu_s` coordinates).
    pub rho_r: Vec<F128>,
    /// The word within the public region (3 coordinates).
    pub rho_c: [F128; 3],
    /// The level-1 block of the parity tie (`log2(blocks)` coordinates).
    pub rho_beta: Vec<F128>,
    /// The parity word within a level-1 row (6 coordinates).
    pub rho_y: [F128; 6],
    /// The output word of the levels-2-3 row (5 coordinates).
    pub rho_o: [F128; OUTPUT_WORDS_LOG2],
}

impl TieChallenges {
    /// Derive from `fs` for a part of `2^d` record positions and `2^blocks_log` level-1
    /// blocks per lane, a sponge instance of `2^nu_s` rows, its event-2 cap and the bridge
    /// union's cap.
    #[must_use]
    pub fn derive(
        fs: &[u8; 32],
        d: u32,
        nu_s: usize,
        blocks_log: usize,
        sponge_cap: &KeccakCommitment,
        bridge_cap: &KeccakCommitment,
    ) -> Self {
        let mut ch = FsChallenger::new(TIE_DOMAIN);
        ch.observe_bytes(fs);
        ch.observe_bytes(&u64::from(d).to_le_bytes());
        ch.observe_bytes(&(nu_s as u64).to_le_bytes());
        ch.observe_bytes(&(blocks_log as u64).to_le_bytes());
        put_framed(&mut ch, sponge_cap.as_bytes());
        put_framed(&mut ch, bridge_cap.as_bytes());
        let w = ch.sample_f128_vec(3);
        let rho_i = ch.sample_f128_vec(d as usize);
        let t = ch.sample_f128_vec(2);
        let rho_t1 = ch.sample_f128();
        let rho_r = ch.sample_f128_vec(nu_s);
        let c = ch.sample_f128_vec(3);
        let rho_beta = ch.sample_f128_vec(blocks_log);
        let y = ch.sample_f128_vec(6);
        let o = ch.sample_f128_vec(OUTPUT_WORDS_LOG2);
        Self {
            rho_w: [w[0], w[1], w[2]],
            rho_i,
            rho_t: [t[0], t[1]],
            rho_t1,
            rho_r,
            rho_c: [c[0], c[1], c[2]],
            rho_beta,
            rho_y: std::array::from_fn(|k| y[k]),
            rho_o: std::array::from_fn(|k| o[k]),
        }
    }

    /// `eq(rho_i[nu_s..], 0) = prod_k (1 + rho_i[k])`: the factor between a level-1 input
    /// value and the sponge value of the same lane group when the sponge instance has
    /// `nu_s < d` row coordinates (its records `>= 2^nu_s` do not exist and the bridge's are
    /// padded zeros).
    #[must_use]
    pub fn row_factor(&self, nu_s: usize) -> F128 {
        self.rho_i
            .iter()
            .skip(nu_s)
            .fold(F128::ONE, |acc, &x| acc * (F128::ONE + x))
    }

    /// Lane coordinates of group `h` (lane-index bits 0..2): lanes 0..3 are `[t0, t1, 0]`,
    /// lanes 4, 5 are `[t', 0, 1]`.
    #[must_use]
    pub fn lane_bits(&self, h: usize) -> [F128; 3] {
        if h == 0 {
            [self.rho_t[0], self.rho_t[1], F128::ZERO]
        } else {
            [self.rho_t1, F128::ZERO, F128::ONE]
        }
    }
}

/// The claim points of both instances (module docs), `M - 7` coordinates each, rows low.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TiePoints {
    /// Sponge: the public region (all rows, word columns `[48, 56)`).
    pub sponge_public: Vec<F128>,
    /// Sponge: lanes 0..3 and lanes 4, 5 (rows `< 2^d`).
    pub sponge_lanes: [Vec<F128>; LANE_GROUPS],
    /// Bridge union, level 1: the input words of lanes 0..3 and of lanes 4, 5.
    pub bridge_inputs: [Vec<F128>; LANE_GROUPS],
    /// Bridge union, level 1: the parity words of the two lane groups.
    pub bridge_parities: [Vec<F128>; LANE_GROUPS],
    /// Bridge union, levels 2-3: the parity copies of the two lane groups.
    pub bridge_copies: [Vec<F128>; LANE_GROUPS],
    /// Bridge union, levels 2-3: the output columns (value 0).
    pub bridge_outputs: Vec<F128>,
}

impl TiePoints {
    /// The points for a part of `2^d` record positions, a sponge instance of `2^nu_s` rows
    /// and `sponge_vars` word coordinates, and the bridge union `bridge`.
    #[must_use]
    pub fn new(
        ch: &TieChallenges,
        d: u32,
        nu_s: usize,
        sponge_vars: usize,
        bridge: &BridgeUnion,
    ) -> Self {
        let d = d as usize;
        assert_eq!(ch.rho_i.len(), d);
        assert_eq!(ch.rho_r.len(), nu_s);
        assert_eq!(ch.rho_beta.len(), bridge.level1().blocks_log());
        assert!(sponge_vars >= nu_s + 6, "sponge block of at least 64 words");
        let zero = F128::ZERO;
        // Sponge rows: rho_i on the first min(nu_s, d) coordinates, zero above (rows
        // `>= 2^d` hold no record of this part).
        let sponge_rows: Vec<F128> = (0..nu_s)
            .map(|k| if k < d { ch.rho_i[k] } else { zero })
            .collect();
        let sponge_lane = |h: usize| {
            let mut p = sponge_rows.clone();
            p.extend_from_slice(&ch.rho_w);
            p.extend_from_slice(&ch.lane_bits(h));
            p.resize(sponge_vars, zero);
            p
        };
        // Level 1: record i = i_lo + 2^9 beta; columns [w | i_lo], rows [beta | t].
        let lo = d.min(BLOCK_RECORDS_LOG2);
        let cols: Vec<F128> = ch.rho_w.iter().chain(&ch.rho_i[..lo]).copied().collect();
        let input = |h: usize| bridge.input_point(&ch.rho_i[lo..], &ch.lane_bits(h), &cols);
        let parity = |h: usize| bridge.y1_point(&ch.rho_beta, &ch.lane_bits(h), &ch.rho_y);
        let copy = |h: usize| bridge.top_point(&ch.rho_y, &ch.rho_beta, &ch.lane_bits(h));
        // Public region: word columns 48 + c8 = 0b110000 | c8.
        let mut sponge_public = ch.rho_r.clone();
        sponge_public.extend_from_slice(&ch.rho_c);
        for bit in PUBLIC_WORDS_LOG2..sponge_vars - nu_s {
            sponge_public.push(if (PUBLIC_WORD_BASE >> bit) & 1 == 1 {
                F128::ONE
            } else {
                zero
            });
        }
        Self {
            sponge_public,
            sponge_lanes: [sponge_lane(0), sponge_lane(1)],
            bridge_inputs: [input(0), input(1)],
            bridge_parities: [parity(0), parity(1)],
            bridge_copies: [copy(0), copy(1)],
            bridge_outputs: bridge.output_point(&ch.rho_o),
        }
    }

    /// The bridge union's claim points in proof order: inputs, parities, copies, outputs.
    #[must_use]
    pub fn bridge_points(&self) -> Vec<Vec<F128>> {
        self.bridge_inputs
            .iter()
            .chain(&self.bridge_parities)
            .chain(&self.bridge_copies)
            .chain(std::iter::once(&self.bridge_outputs))
            .cloned()
            .collect()
    }

    /// The sponge's claim points in proof order: public region, lanes.
    #[must_use]
    pub fn sponge_points(&self) -> Vec<Vec<F128>> {
        std::iter::once(&self.sponge_public)
            .chain(&self.sponge_lanes)
            .cloned()
            .collect()
    }
}

/// The event-2 commitment of a tied part: the sponge instance's Ligerito cap (the bridge
/// union's cap depends on `sd` and travels in the proof, K13).
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
        let malformed = KeccakError::Malformed("tied commitment");
        let len = bytes.get(..4).ok_or(malformed.clone_static())?;
        let len = u32::from_le_bytes(len.try_into().expect("4 bytes")) as usize;
        if bytes.len() != 4 + len {
            return Err(malformed);
        }
        Ok(Self {
            sponge: KeccakCommitment::from_bytes(bytes[4..].to_vec()),
        })
    }
}

impl KeccakError {
    /// A copy of a `Malformed` error (the enum does not derive `Clone`).
    fn clone_static(&self) -> Self {
        match self {
            Self::Malformed(m) => Self::Malformed(m),
            Self::Statement(m) => Self::Statement(m),
            Self::Tie(m) => Self::Tie(m),
            Self::Flock(_) => Self::Malformed("flock"),
        }
    }
}

/// A committed tied part: the sponge witness and its commitment data, the part's input
/// bits (the bridge union's source, independent of `sd`), the statement words and the cap.
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

    /// The part's lane-major input bits (the bridge union's inputs).
    #[must_use]
    pub fn bits(&self) -> &[u64] {
        &self.bits
    }

    /// Take the sponge commitment's prover data (to keep it while the buffers are dropped,
    /// K14); the proof then commits again unless the data is handed back by
    /// [`TiedPart::reattach`].
    pub fn take_commit_data(&mut self) -> Option<CommitData> {
        self.sponge_data.take()
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

/// Sizes of a tied part's two instances.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TiedCost {
    /// Records.
    pub records: usize,
    /// Sponge instance: row capacity `log2`, address variables, dense (committed) variables.
    pub sponge_nu: usize,
    pub sponge_m: usize,
    pub sponge_dense_m: usize,
    /// Committed 128-bit words of the sponge instance.
    pub sponge_committed_words: usize,
    /// The bridge union.
    pub bridge: BridgeUnionCost,
    /// Entries the one-row bridge slot's walker read per fold (proof version 1), for
    /// comparison.
    pub slot_walker_entries: u64,
}

/// Proving and verifying context of one tied part: `n` live records in part `index` of a
/// segment split into parts of `2^d` record positions.
pub struct TiedPart<'g> {
    gate: &'g ChainGate,
    n: usize,
    d: u32,
    index: usize,
    profile: Profile,
    sponge: UnionGeometry,
    bridge: BridgeUnion,
}

impl<'g> TiedPart<'g> {
    /// `d_seg` of a single-segment batch of `n <= 2^13` records: `log2 max(next_pow2(n), 8)`
    /// (spec Section 1).
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
    /// records (the part's first `n` positions, segment records `2^d index ..`). Its bridge
    /// union proves the part's share `Y_{sigma, index} = M^{(sigma)} b'_{index}` (deviation
    /// K10).
    ///
    /// # Panics
    /// If `n == 0`, `n > 2^d`, `d < 3`, `d > 13`, the part lies outside the segment, or the
    /// gate is not the sponge gate of `crate::sponge`.
    #[must_use]
    pub fn new_part(gate: &'g ChainGate, n: usize, d: u32, index: usize, profile: Profile) -> Self {
        assert!(n >= 1 && n <= 1 << d, "1 <= n <= 2^d");
        assert_tied_layout(gate);
        let bridge = BridgeUnion::production_part(d, index, profile);
        assert_eq!(
            bridge.segment().lane_len,
            1 << (10 + d),
            "1,024 positions per record"
        );
        let nu_s = capacity_log(n, gate.layout().k_log);
        let sponge = UnionGeometry::new(gate.table_type(), n, nu_s, profile);
        Self {
            gate,
            n,
            d,
            index,
            profile,
            sponge,
            bridge,
        }
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

    /// The bridge union's level-1 layout (its `segment` is the part's input geometry).
    #[must_use]
    pub fn layout(&self) -> &Level1Layout {
        self.bridge.level1()
    }

    /// The part's input geometry (shape, lanes, `lane_len = 2^(10 + d)`, offset).
    #[must_use]
    pub fn segment(&self) -> &SegmentLayout {
        self.bridge.segment()
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

    /// The bridge union.
    #[must_use]
    pub fn bridge(&self) -> &BridgeUnion {
        &self.bridge
    }

    /// Sizes of the two instances.
    #[must_use]
    pub fn cost(&self) -> TiedCost {
        TiedCost {
            records: self.n,
            sponge_nu: self.sponge.nu(),
            sponge_m: self.sponge.m_total(),
            sponge_dense_m: self.sponge.dense_m(),
            sponge_committed_words: self.n * self.gate.layout().useful_bits.div_ceil(128),
            bridge: self.bridge.cost(),
            slot_walker_entries: self.segment().factored_nonzeros(),
        }
    }

    /// The tie challenges and points of this part for the Flock seed `fs` and the two
    /// commitments.
    #[must_use]
    pub fn tie(
        &self,
        fs: &[u8; 32],
        sponge_cap: &KeccakCommitment,
        bridge_cap: &KeccakCommitment,
    ) -> (TieChallenges, TiePoints) {
        let ch = TieChallenges::derive(
            fs,
            self.d,
            self.sponge.nu(),
            self.bridge.level1().blocks_log(),
            sponge_cap,
            bridge_cap,
        );
        let points = TiePoints::new(
            &ch,
            self.d,
            self.sponge.nu(),
            self.sponge.word_vars(),
            &self.bridge,
        );
        (ch, points)
    }

    /// Digest of the tie geometry (bound into both challengers): `d`, the part index, `n`,
    /// the two instances' row capacities, word coordinates and block widths, the blocks.
    fn tie_digest(&self) -> [u8; 32] {
        let mut h = Sha3_256::new();
        h.update(b"agg-keccak/v2/tie-geometry");
        let g = self.bridge.geometry();
        for v in [
            self.d as usize,
            self.index,
            self.n,
            self.sponge.nu(),
            self.sponge.word_vars(),
            g.nu(),
            g.word_vars(),
            g.k_log(BridgeUnion::LEVEL1),
            g.k_log(BridgeUnion::TOP),
            self.bridge.level1().blocks(),
        ] {
            h.update((v as u64).to_le_bytes());
        }
        h.finalize().into()
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

    fn sponge_challenger(&self, fs: &[u8; 32], public: &[F128]) -> FsChallenger {
        let mut ch = FsChallenger::new(DOMAIN);
        ch.observe_bytes(fs);
        ch.observe_label(SPONGE_LABEL);
        ch.observe_bytes(&self.gate.digest());
        ch.observe_bytes(&Self::public_digest(public));
        ch.observe_bytes(&self.tie_digest());
        ch
    }

    fn bridge_challenger(&self, fs: &[u8; 32], digests: [[u8; 32]; 2]) -> FsChallenger {
        let mut ch = FsChallenger::new(DOMAIN);
        ch.observe_bytes(fs);
        ch.observe_label(BRIDGE_LABEL);
        ch.observe_bytes(&digests[0]);
        ch.observe_bytes(&digests[1]);
        ch.observe_bytes(&self.tie_digest());
        ch
    }

    /// The honest sponge slot buffers of `records` (`n` of them) and the part's lane-major
    /// input bits (`6 * 2^(10 + d) / 64` words; column `p + 2^(10 + d) t` is bit `t` of
    /// position `p = j + 256 rho + 1024 i`, padded records zero).
    #[must_use]
    pub fn honest_parts(&self, records: &[SpongeRecord]) -> (SlotBuffers, Vec<u64>) {
        assert_eq!(records.len(), self.n, "one record per live position");
        let inputs: Vec<Vec<u64>> = records.iter().map(SpongeRecord::input_bits).collect();
        let sponge = pack_slot(self.gate, self.sponge.nu(), self.n, |j, z, a, b| {
            self.gate.instance_rows(&inputs[j], z, a, b);
        });
        let seg = self.segment();
        let mut bits = vec![0_u64; seg.input_bits() / 64];
        let lane = seg.lane_len;
        for (i, input) in inputs.iter().enumerate() {
            // Lane t of a record is gate input bits W1_INPUT_BASE + 1024 t + (256 rho + j),
            // 16 aligned u64 words (`SpongeRecord::input_bits`).
            for t in 0..6 {
                let src = (W1_INPUT_BASE + 1024 * t) / 64;
                let dst = (lane * t + 1024 * i) / 64;
                bits[dst..dst + 16].copy_from_slice(&input[src..src + 16]);
            }
        }
        (sponge, bits)
    }

    fn statement(records: &[SpongeRecord]) -> (Vec<F128>, Vec<[u8; CTILDE_BYTES]>) {
        let c_tilde: Vec<[u8; CTILDE_BYTES]> = records.iter().map(SpongeRecord::c_tilde).collect();
        let public = records
            .iter()
            .zip(&c_tilde)
            .flat_map(|(r, c)| sponge_public_words(&r.mu, c))
            .collect();
        (public, c_tilde)
    }

    /// Commit to given witnesses (tests commit tampered ones): the sponge buffers and the
    /// part's input bits; `records` give the statement words `[mu_i, c~_i]` (with `c~_i`
    /// computed natively). Keeps the sponge commitment's prover data (K14).
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
            self.segment().input_bits() / 64,
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
    /// events 2 and 6), with the commitment's prover data if it was kept (K14). Without
    /// the data the proof commits again and panics if Flock's cap differs from
    /// `commitment` (K7).
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
            self.segment().input_bits() / 64,
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

    /// The honest bridge-union blocks of a committed part under the matrices of `sd`.
    #[must_use]
    pub fn honest_blocks(
        &self,
        committed: &TiedCommitted,
        matrices: &BridgeMatrices,
    ) -> BridgeBlocks {
        self.bridge.honest_blocks(matrices, &committed.bits)
    }

    /// The honest share `M^{(sigma)} b'` of a committed part under the matrices of `sd`.
    #[must_use]
    pub fn final_vector(&self, committed: &TiedCommitted, matrices: &BridgeMatrices) -> Vec<u8> {
        let blocks = self.honest_blocks(committed, matrices);
        self.bridge.share(matrices, &blocks)
    }

    /// Prove a committed part with the honest bridge-union blocks of its bits, the
    /// relations of `sd` and the given share `y` (tests pass a wrong one), seeded by `fs`
    /// (the part's Flock seed).
    ///
    /// # Panics
    /// As [`Self::prove_with_blocks`].
    #[must_use]
    pub fn prove_with_final_vector(
        &self,
        committed: TiedCommitted,
        sd: [u8; 32],
        matrices: &BridgeMatrices,
        y: Vec<u8>,
        fs: &[u8; 32],
    ) -> KeccakProof {
        let start = std::time::Instant::now();
        let blocks = self.honest_blocks(&committed, matrices);
        trace("bridge blocks", &start);
        self.prove_with_blocks(committed, sd, matrices, blocks, y, fs)
    }

    /// Prove a committed part with the given bridge-union blocks (tests pass tampered ones)
    /// and share `y`: commit the bridge union, derive the tie, prove the sponge instance
    /// (opening its event-2 commitment) and the bridge union.
    ///
    /// # Panics
    /// If `committed` or `blocks` belongs to another part shape, or Flock commits to another
    /// sponge cap than the event-2 one.
    #[must_use]
    pub fn prove_with_blocks(
        &self,
        committed: TiedCommitted,
        sd: [u8; 32],
        matrices: &BridgeMatrices,
        blocks: BridgeBlocks,
        y: Vec<u8>,
        fs: &[u8; 32],
    ) -> KeccakProof {
        assert_eq!(committed.n, self.n, "committed part of another shape");
        let TiedCommitted {
            sponge,
            sponge_data,
            public,
            commitment: absorbed,
            ..
        } = committed;
        let start = std::time::Instant::now();
        trace("prove: start", &start);
        // The bridge union: its rows under the relations of sd and y, then its commitment.
        let (s1, s2) = self.bridge.slots(sd, matrices, y);
        let buffers = self.bridge.buffers(&blocks, Some((&s1, &s2)));
        drop(blocks);
        let g = self.bridge.geometry();
        let bridge_data = g.commit_data(&[&buffers[0].z, &buffers[1].z]);
        let bridge_cap = bridge_data.commitment();
        trace("bridge rows and commitment", &start);
        let (_, points) = self.tie(fs, &absorbed.sponge, &bridge_cap);
        // The sponge instance, with its three claims (event-2 commitment reused, K14).
        let mut ch = self.sponge_challenger(fs, &public);
        let gate: [&dyn LincheckCircuit; 1] = [self.gate];
        let (sponge_proof, cap) = self.sponge.prove(
            vec![sponge],
            &gate,
            &points.sponge_points(),
            &mut ch,
            sponge_data,
        );
        assert!(
            cap == absorbed.sponge,
            "Flock's prover committed to a different sponge cap than TiedPart::commit"
        );
        trace("sponge instance proven", &start);
        // The bridge union, with its six claims.
        let mut ch = self.bridge_challenger(fs, [s1.digest(), s2.digest()]);
        let circuits: [&dyn LincheckCircuit; 2] = [&s1, &s2];
        let (bridge_proof, cap) = g.prove(
            buffers,
            &circuits,
            &points.bridge_points(),
            &mut ch,
            Some(bridge_data),
        );
        debug_assert!(cap == bridge_cap);
        trace("bridge union proven", &start);
        self.encode(&bridge_cap, &sponge_proof, &bridge_proof)
    }

    /// Compute the share from the committed bits and prove (the honest prover).
    #[must_use]
    pub fn prove(
        &self,
        committed: TiedCommitted,
        sd: [u8; 32],
        matrices: &BridgeMatrices,
        fs: &[u8; 32],
    ) -> BridgedProof {
        let blocks = self.honest_blocks(&committed, matrices);
        let y = self.bridge.share(matrices, &blocks);
        let proof = self.prove_with_blocks(committed, sd, matrices, blocks, y.clone(), fs);
        BridgedProof {
            final_vector: y,
            proof,
        }
    }

    fn encode(
        &self,
        bridge_cap: &KeccakCommitment,
        sponge: &UnionProof,
        bridge: &UnionProof,
    ) -> KeccakProof {
        let (s, b) = (sponge.to_bytes(), bridge.to_bytes());
        let cap = bridge_cap.as_bytes();
        let mut bytes = Vec::with_capacity(HEADER + 12 + cap.len() + s.len() + b.len());
        bytes.extend_from_slice(MAGIC);
        bytes.push(VERSION);
        bytes.push(self.profile.id());
        bytes.extend_from_slice(&(self.n as u32).to_le_bytes());
        bytes.push(self.d as u8);
        for part in [cap, &s, &b] {
            bytes.extend_from_slice(
                &u32::try_from(part.len())
                    .expect("proof below 4 GiB")
                    .to_le_bytes(),
            );
            bytes.extend_from_slice(part);
        }
        KeccakProof { bytes }
    }

    fn decode(
        &self,
        proof: &KeccakProof,
    ) -> Result<(KeccakCommitment, UnionProof, UnionProof), KeccakError> {
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
        {
            return Err(KeccakError::Malformed("instance count"));
        }
        let mut at = HEADER;
        let mut framed = || -> Result<&[u8], KeccakError> {
            let len = b.get(at..at + 4).ok_or(KeccakError::Malformed("framing"))?;
            let len = u32::from_le_bytes(len.try_into().expect("4 bytes")) as usize;
            at += 4;
            let out = b
                .get(at..at + len)
                .ok_or(KeccakError::Malformed("framing"))?;
            at += len;
            Ok(out)
        };
        let cap = framed()?;
        let s = framed()?;
        let br = framed()?;
        if at != b.len() {
            return Err(KeccakError::Malformed("trailing bytes"));
        }
        if cap.len() != self.bridge.geometry().commitment_len() {
            return Err(KeccakError::Malformed("bridge commitment length"));
        }
        Ok((
            KeccakCommitment::from_bytes(cap.to_vec()),
            UnionProof::from_bytes(s, SPONGE_CLAIMS)?,
            UnionProof::from_bytes(br, BRIDGE_CLAIMS)?,
        ))
    }

    /// The verifier's value of the public claim: the multilinear extension of the public
    /// region's words (mu, `c~`, the constant 1, zero; zero on dummy rows) at the point.
    fn public_value(&self, public: &[F128], ch: &TieChallenges) -> F128 {
        let words = 6; // mu (4) then c~ (2) per record, as `sponge_public_words`
        assert_eq!(public.len(), words * self.n);
        let eq_rows = build_eq_table(&ch.rho_r);
        let eq_c = build_eq_table(&ch.rho_c);
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

    /// Verify a part proof against `(mu_i, c~_i)`, the event-2 commitment, the bridge seed
    /// and its matrices, the part's share `y` and the part's Flock seed `fs`.
    ///
    /// # Errors
    /// On a malformed statement, proof or commitment, a Flock rejection, or a tie mismatch.
    #[allow(clippy::too_many_arguments)]
    pub fn verify(
        &self,
        mu: &[[u8; MU_BYTES]],
        c_tilde: &[[u8; CTILDE_BYTES]],
        commitment: &TiedCommitment,
        sd: [u8; 32],
        matrices: &BridgeMatrices,
        y: &[u8],
        fs: &[u8; 32],
        proof: &KeccakProof,
    ) -> Result<(), KeccakError> {
        if mu.len() != self.n || c_tilde.len() != self.n {
            return Err(KeccakError::Statement("record count"));
        }
        if y.len() != self.segment().output_bits() || y.iter().any(|&b| b > 1) {
            return Err(KeccakError::Statement("final vector"));
        }
        if !self.segment().shape.matches(matrices) {
            return Err(KeccakError::Statement("bridge matrices"));
        }
        let start = std::time::Instant::now();
        let (bridge_cap, sponge_proof, bridge_proof) = self.decode(proof)?;
        let public: Vec<F128> = mu
            .iter()
            .zip(c_tilde)
            .flat_map(|(m, c)| sponge_public_words(m, c))
            .collect();
        let (chal, points) = self.tie(fs, &commitment.sponge, &bridge_cap);
        // The sponge instance.
        let mut ch = self.sponge_challenger(fs, &public);
        let gate: [&dyn LincheckCircuit; 1] = [self.gate];
        let v_s = self.sponge.verify(
            &gate,
            &commitment.sponge,
            &sponge_proof,
            &points.sponge_points(),
            &mut ch,
        )?;
        if v_s[0] != self.public_value(&public, &chal) {
            return Err(KeccakError::Tie("public words"));
        }
        trace("verify: sponge instance", &start);
        // The bridge union.
        let (s1, s2) = self.bridge.slots(sd, matrices, y.to_vec());
        let mut ch = self.bridge_challenger(fs, [s1.digest(), s2.digest()]);
        let circuits: [&dyn LincheckCircuit; 2] = [&s1, &s2];
        let v_b = self.bridge.geometry().verify(
            &circuits,
            &bridge_cap,
            &bridge_proof,
            &points.bridge_points(),
            &mut ch,
        )?;
        let factor = chal.row_factor(self.sponge.nu());
        for h in 0..LANE_GROUPS {
            if v_b[h] != factor * v_s[1 + h] {
                return Err(KeccakError::Tie(if h == 0 {
                    "bridge copy of lanes 0..3"
                } else {
                    "bridge copy of lanes 4, 5"
                }));
            }
        }
        for h in 0..LANE_GROUPS {
            if v_b[2 * LANE_GROUPS + h] != v_b[LANE_GROUPS + h] {
                return Err(KeccakError::Tie(if h == 0 {
                    "level-1 parities of lanes 0..3"
                } else {
                    "level-1 parities of lanes 4, 5"
                }));
            }
        }
        if v_b[3 * LANE_GROUPS] != F128::ZERO {
            return Err(KeccakError::Tie("levels-2-3 output columns"));
        }
        trace("verify: bridge union", &start);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flock_union::eval_packed;
    use crate::sponge::sponge_gate;
    use agg_bridge::{seed::derive_matrices_for, shape::Shape};
    use flock_core::test_rng::Rng;

    fn random_record(rng: &mut Rng) -> SpongeRecord {
        let mut mu = [0u8; MU_BYTES];
        let mut w1 = [0u8; 768];
        mu.iter_mut().for_each(|b| *b = rng.next_u32() as u8);
        w1.iter_mut().for_each(|b| *b = rng.next_u32() as u8);
        SpongeRecord { mu, w1_encode: w1 }
    }

    /// The bridge union's padded `z` of honest blocks.
    fn bridge_z(part: &TiedPart<'_>, blocks: &BridgeBlocks) -> Vec<F128> {
        let g = part.bridge().geometry();
        let bufs = part.bridge().buffers(blocks, None);
        let mut out = vec![F128::ZERO; 1 << (g.m_total() - 7)];
        for (i, b) in bufs.iter().enumerate() {
            let start = g.slot(i).offset >> 7;
            out[start..start + b.z.len()].copy_from_slice(&b.z);
        }
        out
    }

    /// On honest witnesses the region evaluations satisfy the tie identities exactly
    /// (`v_in = eq(rho_i[nu_s..], 0) v_s` and `v_top = v_y1` per lane group, the public value
    /// equals the verifier's table), including parts whose sponge instance has fewer row
    /// coordinates than the part has record positions and a part of two level-1 blocks
    /// (`d = 10`); a flipped input bit breaks exactly its lane group's tie 1, a flipped
    /// parity copy its group's tie 2.
    #[test]
    fn honest_regions_satisfy_the_tie_identities() {
        let gate = sponge_gate();
        let mats = derive_matrices_for(&Shape::PRODUCTION, &[5; 32]);
        for (n, d, index) in [(5usize, 3u32, 0usize), (9, 6, 2), (3, 5, 1), (3, 10, 0)] {
            let part = TiedPart::new_part(&gate, n, d, index, Profile::Fast);
            let mut rng = Rng::new(100 + n as u64 + u64::from(d));
            let records: Vec<SpongeRecord> = (0..n).map(|_| random_record(&mut rng)).collect();
            let (sponge, bits) = part.honest_parts(&records);
            let (public, _) = TiedPart::statement(&records);
            let caps = (
                KeccakCommitment::from_bytes(vec![1; 32]),
                KeccakCommitment::from_bytes(vec![2; 32]),
            );
            let fs = [n as u8; 32];
            let (chal, points) = part.tie(&fs, &caps.0, &caps.1);
            let nu_s = part.nu();
            let blocks = part.bridge().honest_blocks(&mats, &bits);
            let zb = bridge_z(&part, &blocks);
            let nu_b = part.bridge().geometry().nu();
            let v_pub = eval_packed(&sponge.z, &points.sponge_public, nu_s);
            assert_eq!(v_pub, part.public_value(&public, &chal), "public (n {n})");
            let factor = chal.row_factor(nu_s);
            for h in 0..LANE_GROUPS {
                let v_s = eval_packed(&sponge.z, &points.sponge_lanes[h], nu_s);
                let v_in = eval_packed(&zb, &points.bridge_inputs[h], nu_b);
                assert_eq!(v_in, factor * v_s, "lane group {h} (n {n}, d {d})");
                assert_ne!(v_s, F128::ZERO);
                let v_y1 = eval_packed(&zb, &points.bridge_parities[h], nu_b);
                let v_top = eval_packed(&zb, &points.bridge_copies[h], nu_b);
                assert_eq!(v_y1, v_top, "parities of group {h} (d {d})");
                assert_ne!(v_y1, F128::ZERO);
            }
            assert_eq!(eval_packed(&zb, &points.bridge_outputs, nu_b), F128::ZERO);
            // An output column set to 1 (rows kept: the share changes with it) is seen.
            let mut bad = blocks.clone();
            let c = part.bridge().top().y_col(4095);
            bad.top[c / 64] ^= 1 << (c % 64);
            let zb_bad = bridge_z(&part, &bad);
            assert_ne!(
                eval_packed(&zb_bad, &points.bridge_outputs, nu_b),
                F128::ZERO
            );
            // A bit flipped in the input copy (lane 4, the last record) breaks tie 1 of
            // lanes 4, 5 only.
            let mut bad = bits.clone();
            let c = part.segment().column(4, 1024 * (n - 1) + 300);
            bad[c / 64] ^= 1 << (c % 64);
            let bad_blocks = part.bridge().honest_blocks(&mats, &bad);
            let zb_bad = bridge_z(&part, &bad_blocks);
            let v_s1 = eval_packed(&sponge.z, &points.sponge_lanes[1], nu_s);
            assert_ne!(
                eval_packed(&zb_bad, &points.bridge_inputs[1], nu_b),
                factor * v_s1
            );
            let v_s0 = eval_packed(&sponge.z, &points.sponge_lanes[0], nu_s);
            assert_eq!(
                eval_packed(&zb_bad, &points.bridge_inputs[0], nu_b),
                factor * v_s0
            );
            // A parity copy of lane 1 flipped: tie 2 of lanes 0..3 only.
            let mut bad = blocks.clone();
            let c = part.bridge().top().input_col(1, 0, 4000);
            bad.top[c / 64] ^= 1 << (c % 64);
            let zb_bad = bridge_z(&part, &bad);
            for h in 0..LANE_GROUPS {
                let v_y1 = eval_packed(&zb_bad, &points.bridge_parities[h], nu_b);
                let v_top = eval_packed(&zb_bad, &points.bridge_copies[h], nu_b);
                assert_eq!(v_y1 == v_top, h == 1, "group {h} (d {d})");
            }
        }
    }

    #[test]
    fn tie_points_have_the_documented_shape() {
        let gate = sponge_gate();
        let part = TiedPart::new_part(&gate, 5, 3, 0, Profile::Fast);
        let caps = (
            KeccakCommitment::from_bytes(vec![1; 32]),
            KeccakCommitment::from_bytes(vec![2; 32]),
        );
        let (chal, points) = part.tie(&[1; 32], &caps.0, &caps.1);
        assert_eq!(part.nu(), 3);
        assert_eq!(part.sponge().word_vars(), 3 + 12);
        // d = 3: level 1 k_log 15, levels 2-3 k_log 16 (slot 0), nu 5, M = 22.
        let g = part.bridge().geometry();
        assert_eq!((g.nu(), g.word_vars()), (5, 15));
        assert_eq!(
            (
                g.slot(BridgeUnion::TOP).offset,
                g.slot(BridgeUnion::LEVEL1).prefix
            ),
            (0, 2)
        );
        assert_eq!(chal.rho_i.len(), 3);
        assert!(chal.rho_beta.is_empty());
        assert_eq!(chal.row_factor(3), F128::ONE);
        assert_eq!(points.sponge_public.len(), 15);
        // Public region: rows, c8, then bits 3..5 of 48 = 0b110000.
        assert_eq!(&points.sponge_public[..3], &chal.rho_r[..]);
        assert_eq!(&points.sponge_public[3..6], &chal.rho_c[..]);
        assert_eq!(
            &points.sponge_public[6..9],
            &[F128::ZERO, F128::ONE, F128::ONE]
        );
        assert!(points.sponge_public[9..].iter().all(|&x| x == F128::ZERO));
        // Level-1 inputs of lanes 4, 5: rows [t', 0, 1, 0, 0], columns [w, i (3), 0 ...],
        // the level-1 slot's prefix (0b10, one bit: slot 1 of two equal halves).
        let p = &points.bridge_inputs[1];
        assert_eq!(p.len(), 15);
        assert_eq!(
            &p[..5],
            &[chal.rho_t1, F128::ZERO, F128::ONE, F128::ZERO, F128::ZERO]
        );
        assert_eq!(&p[5..8], &chal.rho_w[..]);
        assert_eq!(&p[8..11], &chal.rho_i[..]);
        assert!(p[11..13].iter().all(|&x| x == F128::ZERO));
        assert_eq!(&p[13..], &[F128::ZERO, F128::ONE]);
        // Parity words of lanes 0..3: columns [y (6) | (2^13 + 2^13) / 2^13 = 2 = 0b10].
        let p = &points.bridge_parities[0];
        assert_eq!(
            &p[..5],
            &[
                chal.rho_t[0],
                chal.rho_t[1],
                F128::ZERO,
                F128::ZERO,
                F128::ZERO
            ]
        );
        assert_eq!(&p[5..11], &chal.rho_y[..]);
        assert_eq!(&p[11..13], &[F128::ZERO, F128::ONE]);
        // Their copies: row 0, columns [y | t (3)], slot 0 (prefix 0).
        let p = &points.bridge_copies[0];
        assert!(p[..5].iter().all(|&x| x == F128::ZERO));
        assert_eq!(&p[5..11], &chal.rho_y[..]);
        assert_eq!(&p[11..14], &chal.lane_bits(0)[..]);
        assert_eq!(p[14], F128::ZERO);
        // The output columns: row 0, words 6 x 64 + 32 + o = 0b1_1010_0000 + o (o < 32), so
        // columns [o (5) | 1, 0, 1, 1] then the slot prefix 0.
        let p = &points.bridge_outputs;
        assert!(p[..5].iter().all(|&x| x == F128::ZERO));
        assert_eq!(&p[5..10], &chal.rho_o[..]);
        assert_eq!(&p[10..14], &[F128::ONE, F128::ZERO, F128::ONE, F128::ONE]);
        assert_eq!(p[14], F128::ZERO);
        // Another seed or another cap gives other challenges.
        let other = KeccakCommitment::from_bytes(vec![3; 32]);
        assert_ne!(
            TieChallenges::derive(&[2; 32], 3, 3, 0, &caps.0, &caps.1),
            chal
        );
        assert_ne!(
            TieChallenges::derive(&[1; 32], 3, 3, 0, &caps.0, &other),
            chal
        );
        assert_ne!(
            TieChallenges::derive(&[1; 32], 3, 3, 0, &other, &caps.1),
            chal
        );
        // A part of 1 record with d = 6: nu_s = 3 < d, so the factor is not 1.
        let wide = TiedPart::new_part(&gate, 1, 6, 0, Profile::Fast);
        assert_eq!(wide.nu(), 3);
        let (c2, p2) = wide.tie(&[1; 32], &caps.0, &caps.1);
        assert_ne!(c2.row_factor(3), F128::ONE);
        assert_eq!(p2.sponge_lanes[0][..3], c2.rho_i[..3]);
        // d = 13: 16 blocks, 4 block coordinates in the rows of level 1.
        let full = TiedPart::new_part(&gate, 1 << 13, 13, 0, Profile::Fast);
        let (c3, p3) = full.tie(&[1; 32], &caps.0, &caps.1);
        assert_eq!(c3.rho_beta.len(), 4);
        assert_eq!(&p3.bridge_inputs[0][..4], &c3.rho_i[9..]);
        assert_eq!(&p3.bridge_parities[0][..4], &c3.rho_beta[..]);
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
