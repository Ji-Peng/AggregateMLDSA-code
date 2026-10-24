//! The tied part: the Flock argument of one part of a segment as two single-type Flock
//! instances, the sponge instance (the `n` live records' SHAKE256 gates, docs/keccak.md
//! Sections 1 to 4) and the bridge instance (one bridge slot, docs/bridge.md Section 5,
//! deviation K8), tied to each other and to the public words by evaluation claims on their
//! committed polynomials ([`crate::flock_union`]; docs/keccak.md Section 9, docs/bridge.md
//! Section 5a, deviation K11). It replaces the two-type union of [`crate::segment`], whose
//! word-copy wiring and uniform row capacity dominated time, memory and proof bytes
//! (docs/keccak.md Section 8).
//!
//! Regions. Both instances commit their words BatchMajor: word column `c` of row `j` is the
//! padded word `(c << nu) + j`. In the sponge block (`nu_s` rows) lane `t` of record `j` is
//! the eight words `8 t + w`, `w < 8` ([`crate::sponge::W1_WORD_BASE`]), so lanes 0..3 are
//! the word columns `[0, 32)` and lanes 4, 5 the columns `[32, 48)`: two aligned subcubes
//! of the column index `[w (3 bits) | t | high bits]`. In the bridge block (one live row of
//! 8) lane `t` of record `i` is the eight words `t 2^(3 + d) + 8 i + w`, so the same two lane
//! groups are the subcubes `[w (3) | i (d) | t | high]` (the bridge instance has one live
//! row of `2^nu_b`, `nu_b = max(3, 22 - k_b)`: Flock's lincheck minimum, raised to reach its
//! dense floor for small blocks). The public region of the sponge
//! block is the word columns `[48, 56)`: mu (4 words), `c~` (2), the constant (1, the word
//! with bit 0 set), one zero word.
//!
//! Claims. From the part's Flock seed `fs` both sides derive `rho = (rho_w, rho_i, rho_t,
//! rho_t', rho_r, rho_c)` ([`TieChallenges`]) and the five points of [`TiePoints`]. The
//! sponge proof carries `v_pub = z_s(P_pub)`, `v_s0 = z_s(P_s0)`, `v_s1 = z_s(P_s1)` and the
//! bridge proof `v_b0 = z_b(P_b0)`, `v_b1 = z_b(P_b1)`, each the committed multilinear (over
//! words) at the point, bound by Flock's merged opening. The verifier checks `v_pub` against
//! the table it builds from `(mu_i, c~_i)` and `v_bh = eq(rho_i[nu_s..d], 0) v_sh` for both
//! lane groups `h` (the factor is 1 unless the sponge instance has fewer row coordinates
//! than the part has record positions). If the bridge copy of any bit differed from the
//! sponge input (the sponge's dummy rows and the bridge's padded records included, both
//! zero), the two restricted multilinears would differ as polynomials in the `d + 5` free
//! coordinates and agree at `rho` with probability at most `(d + 5) / 2^128`: `rho` is fixed
//! only after both commitments were absorbed (events 2 and 5 of spec Section 10). The same
//! bound applies to the public claim (`nu_s + 3` coordinates).
//!
//! Order (deviation K7): [`TiedPart::commit`] fixes both witnesses and returns the two caps
//! (event 2); after `sd` (event 3) the prover computes the part's share of `Y_sigma`; after
//! `fs` (event 5) [`TiedPart::prove`] proves the sponge instance, then the bridge instance,
//! and [`TiedPart::verify`] replays both with its own points, matrices and share.

use std::sync::Arc;

use flock_core::{
    challenger::{Challenger, FsChallenger},
    lincheck::build_eq_table,
};
use flock_field::F128;
use sha3::{Digest, Sha3_256};

use agg_witness::bridge::BridgeMatrices;

use crate::{
    bridge_slot::{slot_buffers, BridgeSlot, SlotLayout},
    flock_union::{eval_packed, UnionGeometry, UnionProof},
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
/// Serialization version.
const VERSION: u8 = 1;
/// Header: magic, version, profile, `n` (LE32), `d`.
const HEADER: usize = 6 + 1 + 1 + 4 + 1;
/// Domain of the tie challenges.
pub const TIE_DOMAIN: &[u8] = b"agg-mldsa/v1/tie";
/// Challenger labels of the two instances.
const SPONGE_LABEL: &[u8] = b"agg-keccak/v1/tied-sponge";
const BRIDGE_LABEL: &[u8] = b"agg-keccak/v1/tied-bridge";
/// Word columns per record and lane (`log2`).
const LANE_WORDS_LOG2: usize = 3;
/// Lane groups tied by one claim each: lanes 0..3 (two `t` coordinates) and lanes 4, 5 (one).
const LANE_GROUPS: usize = 2;
/// Claims of the sponge proof: the public region, then the two lane groups.
const SPONGE_CLAIMS: usize = 1 + LANE_GROUPS;

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

/// The random coordinates of the tie, derived from the part's Flock seed.
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
}

impl TieChallenges {
    /// Derive from `fs` for a part of `2^d` record positions and a sponge instance of
    /// `2^nu_s` rows.
    #[must_use]
    pub fn derive(fs: &[u8; 32], d: u32, nu_s: usize) -> Self {
        let mut ch = FsChallenger::new(TIE_DOMAIN);
        ch.observe_bytes(fs);
        ch.observe_bytes(&u64::from(d).to_le_bytes());
        ch.observe_bytes(&(nu_s as u64).to_le_bytes());
        let w = ch.sample_f128_vec(3);
        let rho_i = ch.sample_f128_vec(d as usize);
        let t = ch.sample_f128_vec(2);
        let rho_t1 = ch.sample_f128();
        let rho_r = ch.sample_f128_vec(nu_s);
        let c = ch.sample_f128_vec(3);
        Self {
            rho_w: [w[0], w[1], w[2]],
            rho_i,
            rho_t: [t[0], t[1]],
            rho_t1,
            rho_r,
            rho_c: [c[0], c[1], c[2]],
        }
    }

    /// `eq(rho_i[nu_s..], 0) = prod_k (1 + rho_i[k])`: the factor between a bridge value and
    /// the sponge value of the same lane group when the sponge instance has `nu_s < d` row
    /// coordinates (its records `>= 2^nu_s` do not exist and the bridge's are padded zeros).
    #[must_use]
    pub fn row_factor(&self, nu_s: usize) -> F128 {
        self.rho_i
            .iter()
            .skip(nu_s)
            .fold(F128::ONE, |acc, &x| acc * (F128::ONE + x))
    }
}

/// The claim points of both instances (module docs), `M - 7` coordinates each, rows low.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TiePoints {
    /// Sponge: the public region (all rows, word columns `[48, 56)`).
    pub sponge_public: Vec<F128>,
    /// Sponge: lanes 0..3 and lanes 4, 5 (rows `< 2^d`).
    pub sponge_lanes: [Vec<F128>; LANE_GROUPS],
    /// Bridge: lanes 0..3 and lanes 4, 5 (row 0).
    pub bridge_lanes: [Vec<F128>; LANE_GROUPS],
}

impl TiePoints {
    /// The points for a part of `2^d` record positions, a sponge instance of `2^nu_s` rows
    /// and `sponge_vars` word coordinates, and a bridge instance of `2^nu_b` rows and
    /// `bridge_vars = nu_b + d + 6` word coordinates.
    #[must_use]
    pub fn new(
        ch: &TieChallenges,
        d: u32,
        nu_s: usize,
        sponge_vars: usize,
        nu_b: usize,
        bridge_vars: usize,
    ) -> Self {
        let d = d as usize;
        assert_eq!(ch.rho_i.len(), d);
        assert_eq!(ch.rho_r.len(), nu_s);
        assert!(sponge_vars >= nu_s + 6, "sponge block of at least 64 words");
        assert_eq!(
            bridge_vars,
            nu_b + LANE_WORDS_LOG2 + d + 3,
            "bridge word coordinates"
        );
        let (zero, one) = (F128::ZERO, F128::ONE);
        // Lane coordinates after `w` (and, for the bridge, after `i`): lanes 0..3 are
        // `[t0, t1, 0]`, lanes 4 and 5 are `[t', 0, 1]` (lane index bits 0..2 of 4 + t').
        let lane_bits = |h: usize| -> [F128; 3] {
            if h == 0 {
                [ch.rho_t[0], ch.rho_t[1], zero]
            } else {
                [ch.rho_t1, zero, one]
            }
        };
        // Sponge rows: rho_i on the first min(nu_s, d) coordinates, zero above (rows
        // `>= 2^d` hold no record of this part).
        let sponge_rows: Vec<F128> = (0..nu_s)
            .map(|k| if k < d { ch.rho_i[k] } else { zero })
            .collect();
        let sponge_lane = |h: usize| {
            let mut p = sponge_rows.clone();
            p.extend_from_slice(&ch.rho_w);
            p.extend_from_slice(&lane_bits(h));
            p.resize(sponge_vars, zero);
            p
        };
        let bridge_lane = |h: usize| {
            let mut p = vec![zero; nu_b];
            p.extend_from_slice(&ch.rho_w);
            p.extend_from_slice(&ch.rho_i);
            p.extend_from_slice(&lane_bits(h));
            p
        };
        // Public region: word columns 48 + c8 = 0b110000 | c8.
        let mut sponge_public = ch.rho_r.clone();
        sponge_public.extend_from_slice(&ch.rho_c);
        for bit in PUBLIC_WORDS_LOG2..sponge_vars - nu_s {
            sponge_public.push(if (PUBLIC_WORD_BASE >> bit) & 1 == 1 {
                one
            } else {
                zero
            });
        }
        Self {
            sponge_public,
            sponge_lanes: [sponge_lane(0), sponge_lane(1)],
            bridge_lanes: [bridge_lane(0), bridge_lane(1)],
        }
    }
}

/// The two event-2 commitments of a tied part.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TiedCommitment {
    /// The sponge instance's Ligerito cap.
    pub sponge: KeccakCommitment,
    /// The bridge instance's Ligerito cap.
    pub bridge: KeccakCommitment,
}

impl TiedCommitment {
    /// `LE32(len) || sponge || LE32(len) || bridge`: the event-2 payload of the part.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for c in [&self.sponge, &self.bridge] {
            let b = c.as_bytes();
            out.extend_from_slice(
                &u32::try_from(b.len())
                    .expect("cap below 4 GiB")
                    .to_le_bytes(),
            );
            out.extend_from_slice(b);
        }
        out
    }

    /// Parse (the cap lengths are checked by [`TiedPart::verify`]).
    ///
    /// # Errors
    /// On a malformed byte string.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, KeccakError> {
        let malformed = KeccakError::Malformed("tied commitment");
        let mut at = 0;
        let mut take = |bytes: &[u8]| -> Result<Vec<u8>, KeccakError> {
            let len = bytes.get(at..at + 4).ok_or(malformed.clone_static())?;
            let len = u32::from_le_bytes(len.try_into().expect("4 bytes")) as usize;
            at += 4;
            let out = bytes
                .get(at..at + len)
                .ok_or(malformed.clone_static())?
                .to_vec();
            at += len;
            Ok(out)
        };
        let sponge = take(bytes)?;
        let bridge = take(bytes)?;
        if at != bytes.len() {
            return Err(malformed);
        }
        Ok(Self {
            sponge: KeccakCommitment::from_bytes(sponge),
            bridge: KeccakCommitment::from_bytes(bridge),
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

/// A committed tied part: both witnesses (independent of `sd`), the statement words and
/// the two caps.
pub struct TiedCommitted {
    n: usize,
    sponge: SlotBuffers,
    bridge_z: Vec<u64>,
    public: Vec<F128>,
    c_tilde: Vec<[u8; CTILDE_BYTES]>,
    commitment: TiedCommitment,
}

impl TiedCommitted {
    /// The commitments to absorb at event 2.
    #[must_use]
    pub fn commitment(&self) -> &TiedCommitment {
        &self.commitment
    }

    /// The published `c~_i`.
    #[must_use]
    pub fn c_tilde(&self) -> &[[u8; CTILDE_BYTES]] {
        &self.c_tilde
    }

    /// The committed bridge block.
    #[must_use]
    pub fn bridge_block(&self) -> &[u64] {
        &self.bridge_z
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
    /// Bridge instance: block width `log2`, address variables, dense variables.
    pub bridge_k_log: usize,
    pub bridge_m: usize,
    pub bridge_dense_m: usize,
    /// Nonzeros of the factored bridge rows the walker touches.
    pub bridge_factored_nonzeros: u64,
    /// Committed 128-bit words of each instance.
    pub sponge_committed_words: usize,
    pub bridge_committed_words: usize,
}

/// Proving and verifying context of one tied part: `n` live records in part `index` of a
/// segment split into parts of `2^d` record positions.
pub struct TiedPart<'g> {
    gate: &'g ChainGate,
    n: usize,
    d: u32,
    index: usize,
    layout: SlotLayout,
    profile: Profile,
    sponge: UnionGeometry,
    bridge: UnionGeometry,
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
    /// If `n == 0`, `n > 2^{d_seg}` or `d_seg > 13`.
    #[must_use]
    pub fn new(gate: &'g ChainGate, n: usize, d_seg: u32, profile: Profile) -> Self {
        Self::new_part(gate, n, d_seg, 0, profile)
    }

    /// Part `index` of a segment split into parts of `2^d` record positions, with `n` live
    /// records (the part's first `n` positions, segment records `2^d index ..`). Its bridge
    /// instance proves the part's share `Y_{sigma, index} = M^{(sigma)} b'_{index}`
    /// (deviation K10).
    ///
    /// # Panics
    /// If `n == 0`, `n > 2^d`, `d > 13`, the part lies outside the segment, or the gate is not
    /// the sponge gate of `crate::sponge`.
    #[must_use]
    pub fn new_part(gate: &'g ChainGate, n: usize, d: u32, index: usize, profile: Profile) -> Self {
        assert!(n >= 1 && n <= 1 << d, "1 <= n <= 2^d");
        assert_tied_layout(gate);
        let layout = SlotLayout::production_part(d, index, n);
        let nu_s = capacity_log(n, gate.layout().k_log);
        let sponge = UnionGeometry::new(gate.table_type(), n, nu_s, profile);
        let bridge = UnionGeometry::new(
            layout.table_type(),
            1,
            capacity_log(1, layout.k_log()),
            profile,
        );
        assert_eq!(layout.k_log(), 13 + d as usize, "bridge block width");
        Self {
            gate,
            n,
            d,
            index,
            layout,
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

    /// The bridge slot layout.
    #[must_use]
    pub fn layout(&self) -> &SlotLayout {
        &self.layout
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

    /// The bridge instance's geometry.
    #[must_use]
    pub fn bridge(&self) -> &UnionGeometry {
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
            bridge_k_log: self.bridge.k_log(),
            bridge_m: self.bridge.m_total(),
            bridge_dense_m: self.bridge.dense_m(),
            bridge_factored_nonzeros: self.layout.segment.factored_nonzeros(),
            sponge_committed_words: self.n * self.gate.layout().useful_bits.div_ceil(128),
            bridge_committed_words: self.layout.useful_bits().div_ceil(128),
        }
    }

    /// The tie challenges and points of this part for the Flock seed `fs`.
    #[must_use]
    pub fn tie(&self, fs: &[u8; 32]) -> (TieChallenges, TiePoints) {
        let ch = TieChallenges::derive(fs, self.d, self.sponge.nu());
        let points = TiePoints::new(
            &ch,
            self.d,
            self.sponge.nu(),
            self.sponge.word_vars(),
            self.bridge.nu(),
            self.bridge.word_vars(),
        );
        (ch, points)
    }

    /// Digest of the tie geometry (bound into both challengers): `d`, the part index, `n`,
    /// the two instances' row capacities and word coordinates.
    fn tie_digest(&self) -> [u8; 32] {
        let mut h = Sha3_256::new();
        h.update(b"agg-keccak/v1/tie-geometry");
        for v in [
            self.d as usize,
            self.index,
            self.n,
            self.sponge.nu(),
            self.sponge.word_vars(),
            self.bridge.nu(),
            self.bridge.word_vars(),
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

    fn bridge_challenger(&self, fs: &[u8; 32], slot: &BridgeSlot) -> FsChallenger {
        let mut ch = FsChallenger::new(DOMAIN);
        ch.observe_bytes(fs);
        ch.observe_label(BRIDGE_LABEL);
        ch.observe_bytes(&slot.digest());
        ch.observe_bytes(&self.tie_digest());
        ch
    }

    /// The honest sponge slot buffers and bridge block of `records` (`n` of them).
    #[must_use]
    pub fn honest_parts(&self, records: &[SpongeRecord]) -> (SlotBuffers, Vec<u64>) {
        assert_eq!(records.len(), self.n, "one record per live position");
        let inputs: Vec<Vec<u64>> = records.iter().map(SpongeRecord::input_bits).collect();
        let sponge = pack_slot(self.gate, self.sponge.nu(), self.n, |j, z, a, b| {
            self.gate.instance_rows(&inputs[j], z, a, b);
        });
        let mut bits = vec![0_u64; self.layout.input_bits() / 64];
        let lane = self.layout.segment.lane_len;
        for (i, input) in inputs.iter().enumerate() {
            // Lane t of a record is gate input bits W1_INPUT_BASE + 1024 t + (256 rho + j),
            // 16 aligned u64 words (`SpongeRecord::input_bits`).
            for t in 0..6 {
                let src = (W1_INPUT_BASE + 1024 * t) / 64;
                let dst = (lane * t + 1024 * i) / 64;
                bits[dst..dst + 16].copy_from_slice(&input[src..src + 16]);
            }
        }
        (sponge, self.layout.honest_z(&bits))
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

    /// Commit to given witnesses (tests commit tampered ones); `records` give the statement
    /// words `[mu_i, c~_i]` (with `c~_i` computed natively).
    #[must_use]
    pub fn commit_parts(
        &self,
        sponge: SlotBuffers,
        bridge_z: Vec<u64>,
        records: &[SpongeRecord],
    ) -> TiedCommitted {
        assert_eq!(records.len(), self.n, "one record per live position");
        assert_eq!(
            bridge_z.len(),
            self.layout.useful_words(),
            "bridge block length"
        );
        let start = std::time::Instant::now();
        trace("commit: start", &start);
        let sponge_cap = self.sponge.commit(&sponge.z);
        trace("sponge committed", &start);
        let bridge_buf = slot_buffers(&self.layout, self.bridge.nu(), &bridge_z, None);
        let bridge_cap = self.bridge.commit(&bridge_buf.z);
        trace("bridge committed", &start);
        let (public, c_tilde) = Self::statement(records);
        TiedCommitted {
            n: self.n,
            sponge,
            bridge_z,
            public,
            c_tilde,
            commitment: TiedCommitment {
                sponge: sponge_cap,
                bridge: bridge_cap,
            },
        }
    }

    /// Commit to the honest witness of `records` (event 2: before `sd` and `fs`).
    #[must_use]
    pub fn commit(&self, records: &[SpongeRecord]) -> TiedCommitted {
        let (sponge, bridge_z) = self.honest_parts(records);
        self.commit_parts(sponge, bridge_z, records)
    }

    /// Rebuild a committed part from its witnesses and the commitment computed earlier by
    /// [`Self::commit_parts`] on the same witnesses, without committing again (the driver
    /// drops the buffers between events 2 and 6); [`Self::prove_with_final_vector`] panics
    /// if Flock's caps differ from `commitment`.
    #[must_use]
    pub fn reattach(
        &self,
        sponge: SlotBuffers,
        bridge_z: Vec<u64>,
        records: &[SpongeRecord],
        commitment: TiedCommitment,
    ) -> TiedCommitted {
        assert_eq!(records.len(), self.n, "one record per live position");
        assert_eq!(
            bridge_z.len(),
            self.layout.useful_words(),
            "bridge block length"
        );
        assert_eq!(
            commitment.sponge.as_bytes().len(),
            self.sponge.commitment_len()
        );
        assert_eq!(
            commitment.bridge.as_bytes().len(),
            self.bridge.commitment_len()
        );
        let (public, c_tilde) = Self::statement(records);
        TiedCommitted {
            n: self.n,
            sponge,
            bridge_z,
            public,
            c_tilde,
            commitment,
        }
    }

    /// Bytes of the two event-2 commitments.
    #[must_use]
    pub fn commitment_lens(&self) -> (usize, usize) {
        (self.sponge.commitment_len(), self.bridge.commitment_len())
    }

    /// The honest share `M^{(sigma)} b'` of a committed part under the matrices of `sd`.
    #[must_use]
    pub fn final_vector(&self, committed: &TiedCommitted, matrices: &BridgeMatrices) -> Vec<u8> {
        BridgeSlot::parities(&self.layout, matrices, &committed.bridge_z)
    }

    /// Prove a committed part with the bridge rows of `sd` and the given share `y` (tests
    /// pass a wrong one), seeded by `fs` (the part's Flock seed).
    ///
    /// # Panics
    /// If `committed` belongs to another part shape, or Flock commits to another cap.
    #[must_use]
    pub fn prove_with_final_vector(
        &self,
        committed: TiedCommitted,
        sd: [u8; 32],
        matrices: Arc<BridgeMatrices>,
        y: Vec<u8>,
        fs: &[u8; 32],
    ) -> KeccakProof {
        assert_eq!(committed.n, self.n, "committed part of another shape");
        let TiedCommitted {
            sponge,
            bridge_z,
            public,
            commitment: absorbed,
            ..
        } = committed;
        let start = std::time::Instant::now();
        trace("prove: start", &start);
        let (_, points) = self.tie(fs);
        // The sponge instance, with its three claims.
        let nu_s = self.sponge.nu();
        let mut claims: Vec<(Vec<F128>, F128)> = Vec::with_capacity(SPONGE_CLAIMS);
        for p in std::iter::once(&points.sponge_public).chain(&points.sponge_lanes) {
            claims.push((p.clone(), eval_packed(&sponge.z, p, nu_s)));
        }
        trace("sponge claims", &start);
        let mut ch = self.sponge_challenger(fs, &public);
        let (sponge_proof, cap) = self.sponge.prove(sponge, self.gate, &claims, &mut ch);
        assert!(
            cap == absorbed.sponge,
            "Flock's prover committed to a different sponge cap than TiedPart::commit"
        );
        trace("sponge instance proven", &start);
        // The bridge instance, with its two claims.
        let slot = BridgeSlot::new(self.layout, sd, matrices, y);
        let nu_b = self.bridge.nu();
        let buffers = slot_buffers(&self.layout, nu_b, &bridge_z, Some(&slot));
        let claims: Vec<(Vec<F128>, F128)> = points
            .bridge_lanes
            .iter()
            .map(|p| (p.clone(), eval_packed(&buffers.z, p, nu_b)))
            .collect();
        trace("bridge rows and claims", &start);
        let mut ch = self.bridge_challenger(fs, &slot);
        let (bridge_proof, cap) = self.bridge.prove(buffers, &slot, &claims, &mut ch);
        assert!(
            cap == absorbed.bridge,
            "Flock's prover committed to a different bridge cap than TiedPart::commit"
        );
        trace("bridge instance proven", &start);
        self.encode(&sponge_proof, &bridge_proof)
    }

    /// Compute the share from the committed bridge block and prove (the honest prover).
    #[must_use]
    pub fn prove(
        &self,
        committed: TiedCommitted,
        sd: [u8; 32],
        matrices: Arc<BridgeMatrices>,
        fs: &[u8; 32],
    ) -> BridgedProof {
        let y = self.final_vector(&committed, &matrices);
        let proof = self.prove_with_final_vector(committed, sd, matrices, y.clone(), fs);
        BridgedProof {
            final_vector: y,
            proof,
        }
    }

    fn encode(&self, sponge: &UnionProof, bridge: &UnionProof) -> KeccakProof {
        let (s, b) = (sponge.to_bytes(), bridge.to_bytes());
        let mut bytes = Vec::with_capacity(HEADER + 8 + s.len() + b.len());
        bytes.extend_from_slice(MAGIC);
        bytes.push(VERSION);
        bytes.push(self.profile.id());
        bytes.extend_from_slice(&(self.n as u32).to_le_bytes());
        bytes.push(self.d as u8);
        for part in [&s, &b] {
            bytes.extend_from_slice(
                &u32::try_from(part.len())
                    .expect("proof below 4 GiB")
                    .to_le_bytes(),
            );
            bytes.extend_from_slice(part);
        }
        KeccakProof { bytes }
    }

    fn decode(&self, proof: &KeccakProof) -> Result<(UnionProof, UnionProof), KeccakError> {
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
        let s = framed()?;
        let br = framed()?;
        if at != b.len() {
            return Err(KeccakError::Malformed("trailing bytes"));
        }
        Ok((
            UnionProof::from_bytes(s, SPONGE_CLAIMS)?,
            UnionProof::from_bytes(br, LANE_GROUPS)?,
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

    /// Verify a part proof against `(mu_i, c~_i)`, the two event-2 commitments, the bridge
    /// seed and its matrices, the part's share `y` and the part's Flock seed `fs`.
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
        matrices: Arc<BridgeMatrices>,
        y: &[u8],
        fs: &[u8; 32],
        proof: &KeccakProof,
    ) -> Result<(), KeccakError> {
        if mu.len() != self.n || c_tilde.len() != self.n {
            return Err(KeccakError::Statement("record count"));
        }
        if y.len() != self.layout.segment.output_bits() || y.iter().any(|&b| b > 1) {
            return Err(KeccakError::Statement("final vector"));
        }
        if !self.layout.shape().matches(&matrices) {
            return Err(KeccakError::Statement("bridge matrices"));
        }
        let (sponge_proof, bridge_proof) = self.decode(proof)?;
        let public: Vec<F128> = mu
            .iter()
            .zip(c_tilde)
            .flat_map(|(m, c)| sponge_public_words(m, c))
            .collect();
        let (chal, points) = self.tie(fs);
        // The sponge instance.
        let mut sponge_points = vec![points.sponge_public.clone()];
        sponge_points.extend(points.sponge_lanes.iter().cloned());
        let mut ch = self.sponge_challenger(fs, &public);
        let v_s = self.sponge.verify(
            self.gate,
            &commitment.sponge,
            &sponge_proof,
            &sponge_points,
            &mut ch,
        )?;
        if v_s[0] != self.public_value(&public, &chal) {
            return Err(KeccakError::Tie("public words"));
        }
        // The bridge instance.
        let slot = BridgeSlot::new(self.layout, sd, matrices, y.to_vec());
        let mut ch = self.bridge_challenger(fs, &slot);
        let v_b = self.bridge.verify(
            &slot,
            &commitment.bridge,
            &bridge_proof,
            &points.bridge_lanes,
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
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sponge::sponge_gate;
    use flock_core::test_rng::Rng;

    fn random_record(rng: &mut Rng) -> SpongeRecord {
        let mut mu = [0u8; MU_BYTES];
        let mut w1 = [0u8; 768];
        mu.iter_mut().for_each(|b| *b = rng.next_u32() as u8);
        w1.iter_mut().for_each(|b| *b = rng.next_u32() as u8);
        SpongeRecord { mu, w1_encode: w1 }
    }

    /// On honest witnesses the region evaluations satisfy the tie identities exactly
    /// (`v_b = eq(rho_i[nu_s..], 0) v_s` per lane group, the public value equals the
    /// verifier's table), for a part whose sponge instance has more row coordinates than the
    /// part has record positions (`n = 5`, `d = 3`, `nu_s = 3`... and `d = 6`, `nu_s = 4`).
    #[test]
    fn honest_regions_satisfy_the_tie_identities() {
        let gate = sponge_gate();
        for (n, d, index) in [(5usize, 3u32, 0usize), (9, 6, 2), (3, 5, 1)] {
            let part = TiedPart::new_part(&gate, n, d, index, Profile::Fast);
            let mut rng = Rng::new(100 + n as u64);
            let records: Vec<SpongeRecord> = (0..n).map(|_| random_record(&mut rng)).collect();
            let (sponge, bridge_z) = part.honest_parts(&records);
            let (public, _) = TiedPart::statement(&records);
            let fs = [n as u8; 32];
            let (chal, points) = part.tie(&fs);
            let (nu_s, nu_b) = (part.nu(), part.bridge().nu());
            let bridge = slot_buffers(part.layout(), nu_b, &bridge_z, None);
            let v_pub = eval_packed(&sponge.z, &points.sponge_public, nu_s);
            assert_eq!(
                v_pub,
                part.public_value(&public, &chal),
                "public claim (n {n})"
            );
            let factor = chal.row_factor(nu_s);
            for h in 0..LANE_GROUPS {
                let v_s = eval_packed(&sponge.z, &points.sponge_lanes[h], nu_s);
                let v_b = eval_packed(&bridge.z, &points.bridge_lanes[h], nu_b);
                assert_eq!(v_b, factor * v_s, "lane group {h} (n {n}, d {d})");
                assert_ne!(v_s, F128::ZERO);
            }
            // A bit flipped in the bridge copy breaks exactly its lane group's identity.
            let mut bad = bridge_z.clone();
            let c = part.layout().segment.column(4, 1024 * (n - 1) + 300);
            bad[c / 64] ^= 1 << (c % 64);
            let bad = slot_buffers(part.layout(), nu_b, &bad, None);
            let v_s1 = eval_packed(&sponge.z, &points.sponge_lanes[1], nu_s);
            assert_ne!(
                eval_packed(&bad.z, &points.bridge_lanes[1], nu_b),
                factor * v_s1
            );
            let v_s0 = eval_packed(&sponge.z, &points.sponge_lanes[0], nu_s);
            assert_eq!(
                eval_packed(&bad.z, &points.bridge_lanes[0], nu_b),
                factor * v_s0
            );
        }
    }

    #[test]
    fn tie_points_have_the_documented_shape() {
        let gate = sponge_gate();
        let part = TiedPart::new_part(&gate, 5, 3, 0, Profile::Fast);
        let (chal, points) = part.tie(&[1; 32]);
        assert_eq!(part.nu(), 3);
        assert_eq!(part.sponge().word_vars(), 3 + 12);
        // d = 3: the bridge block is 2^16 bits, so Flock's dense floor raises nu_b to 6.
        let nu_b = part.bridge().nu();
        assert_eq!(nu_b, 6);
        assert_eq!(part.bridge().word_vars(), nu_b + 9);
        assert_eq!(chal.rho_i.len(), 3);
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
        // Lanes 4, 5 on the bridge: rows 0, w, i, t', 0, 1.
        let p = &points.bridge_lanes[1];
        assert_eq!(p.len(), nu_b + 9);
        assert!(p[..nu_b].iter().all(|&x| x == F128::ZERO));
        assert_eq!(&p[nu_b..nu_b + 3], &chal.rho_w[..]);
        assert_eq!(&p[nu_b + 3..nu_b + 6], &chal.rho_i[..]);
        assert_eq!(&p[nu_b + 6..], &[chal.rho_t1, F128::ZERO, F128::ONE]);
        // d >= 6: the bridge instance has Flock's minimum of 8 rows.
        assert_eq!(
            TiedPart::new_part(&gate, 1, 6, 0, Profile::Fast)
                .bridge()
                .nu(),
            3
        );
        // Another seed or another geometry gives other challenges.
        assert_ne!(TieChallenges::derive(&[2; 32], 3, 3), chal);
        assert_ne!(TieChallenges::derive(&[1; 32], 4, 3).rho_w, chal.rho_w);
        let wide = TiedPart::new_part(&gate, 1, 6, 0, Profile::Fast);
        assert_eq!(wide.nu(), 3);
        let (c2, p2) = wide.tie(&[1; 32]);
        assert_ne!(c2.row_factor(3), F128::ONE);
        assert_eq!(p2.sponge_lanes[0][..3], c2.rho_i[..3]);
    }

    #[test]
    fn tied_commitment_bytes_round_trip() {
        let c = TiedCommitment {
            sponge: KeccakCommitment::from_bytes(vec![1; 64]),
            bridge: KeccakCommitment::from_bytes(vec![2; 32]),
        };
        let bytes = c.to_bytes();
        assert_eq!(bytes.len(), 8 + 96);
        assert_eq!(TiedCommitment::from_bytes(&bytes).unwrap(), c);
        assert!(TiedCommitment::from_bytes(&bytes[..bytes.len() - 1]).is_err());
        let mut trailing = bytes;
        trailing.push(0);
        assert!(TiedCommitment::from_bytes(&trailing).is_err());
    }
}
