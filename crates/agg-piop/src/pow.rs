//! The proof-of-work plan of a proof and the packed PIOP nonce stream.
//!
//! The plan is the ordered list of the proof's protected challenge sites: the sites of every
//! tied part of the Keccak side ([`KeccakPlan`], in segment and part order: the tie site, the
//! `eq_Y` site and the parity-chain rounds of tied-part proof version 5), then the PIOP sites
//! in registry order ([`crate::transcript::Registry`]: the projection matrix `psi`, the lookup
//! vector `X` included with difficulty 0). Every site carries its class, two indices and its
//! difficulty `g` ([`agg_bridge::pow`]); the canonical encoding is hashed with
//! SHA3-256 and the digest completes the instance record
//! ([`crate::binding::instance_record`]), so the transcript binds the plan before its first
//! site.
//!
//! The PIOP nonces travel in one byte string per proof ([`crate::wire::AggregateProof::work`]):
//! nonce `i` occupies `g_i + 7` bits at bit offset `sum_{j < i} (g_j + 7)`, little-endian bit
//! order, the padding bits zero ([`WorkWriter`], [`WorkReader`]). The tie nonces travel in the
//! tied-part proofs and Akita's and Flock's nonces in their own proofs.

use agg_bridge::pow::{nonce_bits, PartWork};
use sha3::{Digest, Sha3_256};

use crate::binding::FlockProfile;
use crate::manifest::CommitmentManifest;
use crate::transcript::Registry;

/// Domain label of the plan digest.
pub const PLAN_LABEL: &[u8] = b"agg-mldsa/v2/pow-plan";

/// Class of a protected site (the id is its wire byte).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SiteClass {
    /// The lookup challenge vector `X = (X_1, X_2)`: one site, joint bound, no work.
    LookupX = 1,
    /// A line merge `mu_s` of the lookup.
    LookupMu = 2,
    /// The batching vector `gamma_s` of GKR session `s`.
    LookupGamma = 3,
    /// A round challenge of a GKR session (degree 3).
    LookupRound = 4,
    /// The zero-check point `tau` (a vector of `d_main` coordinates).
    Tau = 5,
    /// The batching vector of a stage.
    StageGamma = 6,
    /// A round challenge of a stage (degree 4, 5, 3).
    StageRound = 7,
    /// The tie challenges of a tied part (GF(2^128), `agg_keccak::tied`): the public claim's
    /// point.
    Tie = 8,
    /// The `eq_Y` challenges of a tied part's parity chain (GF(2^128), version 5).
    ChainEq = 9,
    /// A round challenge of a tied part's parity chain (degree 2).
    ChainRound = 10,
    /// The projection matrix `Psi` that bounds `v`: a 16 KiB byte challenge drawn right
    /// after the bridge seed. Its bad set has mass at most `2^-128`, and one bit of work
    /// lowers each query to `2^-129`.
    ProjPsi = 11,
}

impl SiteClass {
    /// Sites of the Keccak side (their nonces travel in the tied-part proofs, not in the
    /// PIOP nonce stream).
    #[must_use]
    pub const fn is_keccak(self) -> bool {
        matches!(self, Self::Tie | Self::ChainEq | Self::ChainRound)
    }
}

/// One protected site of the plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkSite {
    /// Class.
    pub class: SiteClass,
    /// First index: `s` (lookup), the stage, or the tied part.
    pub i: u16,
    /// Second index: the round (rounds), else 0.
    pub j: u16,
    /// Difficulty `g`.
    pub bits: u8,
}

impl WorkSite {
    /// Nonce width (`g + 7`, or 0 without work).
    #[must_use]
    pub const fn nonce_bits(&self) -> u8 {
        nonce_bits(self.bits)
    }
}

/// The Keccak side's part of the plan: the span length `S` of a tied part (1 for parts of
/// one segment, 0 without tied parts), the part size `d_part` (0 without tied parts) and the
/// work of each tied part (its tie, `eq_Y` and chain-round difficulties and its round count)
/// in segment and part order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KeccakPlan {
    /// `S`, the segments of one tied part (0: no tied parts).
    pub span: u8,
    /// `log2` of the record positions per part (0: no tied parts).
    pub d_part: u8,
    /// The work of each tied part.
    pub parts: Vec<PartWork>,
}

/// The plan of one proof (module docs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PowPlan {
    padded: u64,
    profile: u8,
    span: u8,
    d_part: u8,
    sites: Vec<WorkSite>,
}

impl PowPlan {
    /// The plan of a batch with `manifest` under the Flock profile `profile` with the Keccak
    /// side's tied parts `keccak`.
    #[must_use]
    pub fn new(manifest: &CommitmentManifest, profile: FlockProfile, keccak: &KeccakPlan) -> Self {
        let mut sites = Vec::new();
        for (part, work) in keccak.parts.iter().enumerate() {
            let i = u16::try_from(part).expect("at most 2^16 tied parts");
            sites.push(WorkSite {
                class: SiteClass::Tie,
                i,
                j: 0,
                bits: work.tie_bits,
            });
            sites.push(WorkSite {
                class: SiteClass::ChainEq,
                i,
                j: 0,
                bits: work.eq_bits,
            });
            sites.extend((0..work.rounds).map(|j| WorkSite {
                class: SiteClass::ChainRound,
                i,
                j,
                bits: work.round_bits,
            }));
        }
        sites.extend(Registry::new(manifest).work_sites());
        Self {
            padded: manifest.padded_records() as u64,
            profile: profile as u8,
            span: keccak.span,
            d_part: keccak.d_part,
            sites,
        }
    }

    /// The sites in plan order.
    #[must_use]
    pub fn sites(&self) -> &[WorkSite] {
        &self.sites
    }

    /// Canonical encoding: `LE64(N*) || u8(profile) ||
    /// u8(S) || u8(d_part) || LE32(count) || (u8(class) || LE16(i) || LE16(j) || u8(g) ||
    /// u8(nonce bits))*`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut body = Vec::with_capacity(15 + 7 * self.sites.len());
        body.extend_from_slice(&self.padded.to_le_bytes());
        body.extend_from_slice(&[self.profile, self.span, self.d_part]);
        body.extend_from_slice(&(self.sites.len() as u32).to_le_bytes());
        for site in &self.sites {
            body.push(site.class as u8);
            body.extend_from_slice(&site.i.to_le_bytes());
            body.extend_from_slice(&site.j.to_le_bytes());
            body.extend_from_slice(&[site.bits, site.nonce_bits()]);
        }
        body
    }

    /// `SHA3-256(label || LE64(len) || body)`.
    #[must_use]
    pub fn digest(&self) -> [u8; 32] {
        let body = self.encode();
        let mut h = Sha3_256::new();
        h.update(PLAN_LABEL);
        h.update((body.len() as u64).to_le_bytes());
        h.update(&body);
        h.finalize().into()
    }

    /// Bits of the packed PIOP nonce stream.
    #[must_use]
    pub fn work_bits(&self) -> usize {
        self.sites
            .iter()
            .filter(|s| !s.class.is_keccak())
            .map(|s| usize::from(s.nonce_bits()))
            .sum()
    }

    /// Expected honest work `sum 2^g` over the PIOP sites (predicate evaluations).
    #[must_use]
    pub fn expected_work(&self) -> u64 {
        self.sites
            .iter()
            .filter(|s| !s.class.is_keccak() && s.bits > 0)
            .map(|s| 1_u64 << s.bits)
            .sum()
    }
}

/// Writer of the packed nonce stream (prover).
#[derive(Clone, Debug, Default)]
pub struct WorkWriter {
    bytes: Vec<u8>,
    bits: usize,
}

impl WorkWriter {
    /// Append the low `width` bits of `value`.
    pub fn push(&mut self, value: u32, width: u8) {
        debug_assert!(width == 32 || value >> width == 0, "value fits its width");
        for k in 0..usize::from(width) {
            if self.bits.is_multiple_of(8) {
                self.bytes.push(0);
            }
            let bit = ((value >> k) & 1) as u8;
            *self.bytes.last_mut().expect("a byte") |= bit << (self.bits % 8);
            self.bits += 1;
        }
    }

    /// Bits written.
    #[must_use]
    pub const fn bits(&self) -> usize {
        self.bits
    }

    /// Append every bit of `other` (a branch's nonces after this stream's).
    pub fn append(&mut self, other: &Self) {
        for at in 0..other.bits {
            self.push(u32::from((other.bytes[at / 8] >> (at % 8)) & 1), 1);
        }
    }

    /// The stream (padding bits zero).
    #[must_use]
    pub fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

/// Reader of the packed nonce stream (verifier).
#[derive(Clone, Debug)]
pub struct WorkReader {
    bytes: Vec<u8>,
    bits: usize,
    total: usize,
}

/// A malformed nonce stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkStreamError {
    /// The stream does not have `ceil(total / 8)` bytes.
    Length,
    /// A padding bit is set.
    Padding,
    /// More nonces were read than the stream holds.
    Exhausted,
    /// Nonces are left unread.
    Unread,
}

impl WorkReader {
    /// A reader of a stream of `total` bits: exact length and zero padding are checked here.
    ///
    /// # Errors
    /// On a wrong length or a set padding bit.
    pub fn new(bytes: &[u8], total: usize) -> Result<Self, WorkStreamError> {
        if bytes.len() != total.div_ceil(8) {
            return Err(WorkStreamError::Length);
        }
        if !total.is_multiple_of(8) {
            let last = bytes[bytes.len() - 1];
            if last >> (total % 8) != 0 {
                return Err(WorkStreamError::Padding);
            }
        }
        Ok(Self {
            bytes: bytes.to_vec(),
            bits: 0,
            total,
        })
    }

    /// Read the next `width` bits.
    ///
    /// # Errors
    /// When the stream is exhausted.
    pub fn read(&mut self, width: u8) -> Result<u32, WorkStreamError> {
        let width = usize::from(width);
        if self.bits + width > self.total {
            return Err(WorkStreamError::Exhausted);
        }
        let mut value = 0_u32;
        for k in 0..width {
            let at = self.bits + k;
            value |= u32::from((self.bytes[at / 8] >> (at % 8)) & 1) << k;
        }
        self.bits += width;
        Ok(value)
    }

    /// Split off the next `bits` bits as a reader of their own (a branch's nonces) and skip
    /// them here.
    ///
    /// # Errors
    /// When fewer than `bits` bits are left.
    pub fn split(&mut self, bits: usize) -> Result<Self, WorkStreamError> {
        if self.bits + bits > self.total {
            return Err(WorkStreamError::Exhausted);
        }
        let branch = Self {
            bytes: self.bytes.clone(),
            bits: self.bits,
            total: self.bits + bits,
        };
        self.bits += bits;
        Ok(branch)
    }

    /// Confirm every nonce was read.
    ///
    /// # Errors
    /// When bits are left.
    pub fn finish(&self) -> Result<(), WorkStreamError> {
        if self.bits == self.total {
            Ok(())
        } else {
            Err(WorkStreamError::Unread)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agg_bridge::pow::{list_size, part_work, TieGeometry};

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// The tie site of a full mode-A batch of `2^d_sig` records under the default part size
    /// (`docs/scripts/pow_plan.py`, `tie_parts`): one tied part (a span when `d_sig > 13`).
    fn mode_a(d_sig: usize, rate_log: u32, span: bool) -> KeccakPlan {
        let d = d_sig.min(13);
        let segments = 1_usize << d_sig.saturating_sub(13);
        // agg_keccak::prover::capacity_log at k_log = 19 for n = 2^d_sig (and 2^13 per part).
        let nu_s = |n_log: usize| match n_log {
            3 => 4,
            10 => 11,
            13..=15 => n_log,
            _ => unreachable!(),
        };
        let geo = |span_log: usize, n_log: usize| TieGeometry {
            d,
            span_log,
            blocks_log: d.saturating_sub(9),
            nu_s: nu_s(n_log),
            public_vars: 3,
        };
        let list = list_size(rate_log);
        if segments == 1 || span {
            let span_log = segments.next_power_of_two().trailing_zeros() as usize;
            KeccakPlan {
                span: segments as u8,
                d_part: d as u8,
                parts: vec![part_work(list, &geo(span_log, d_sig))],
            }
        } else {
            KeccakPlan {
                span: 1,
                d_part: d as u8,
                parts: vec![part_work(list, &geo(0, 13)); segments],
            }
        }
    }

    #[test]
    fn plan_digests_reproduce_the_soundness_doc_vectors() {
        // The vectors printed by `python3 -I docs/scripts/pow_plan.py`.
        let cases: [(usize, FlockProfile, Option<bool>, &str); 10] = [
            (
                3,
                FlockProfile::PublicBits,
                None,
                "c3dc3ec0258146974daae961a943ec7f8704748cc8683e257bcb02326859cf26",
            ),
            (
                3,
                FlockProfile::Slim,
                Some(true),
                "979d6e78ca49e082607c9912fbd1c0f49ed0a33ca03e0052cf4155848fe0ae31",
            ),
            (
                10,
                FlockProfile::PublicBits,
                None,
                "3f6d2aac995ed02826a9da18724a301660de37952febdbd67ad991ac2535700f",
            ),
            (
                10,
                FlockProfile::Slim,
                Some(true),
                "0a0afe3dd4fa65a6265d2d234b57ca868f793a62ec8d3d3c9949f9af81b2ceac",
            ),
            (
                13,
                FlockProfile::Slim,
                Some(true),
                "ecbce3b30764d9c0d5f19e5504568c799d0ac8086d02ed10904d12e9a84879b8",
            ),
            (
                15,
                FlockProfile::PublicBits,
                None,
                "329f278362ccd503ccbbe8ce1f8940b5b1babb2a5a6b0870f44965107cff8a79",
            ),
            (
                15,
                FlockProfile::Fast,
                Some(true),
                "1469c9f7b92f4a22dda5942757531c7a61682f0355a81d5d5159702462b822e5",
            ),
            (
                15,
                FlockProfile::Slim,
                Some(true),
                "bcb8ae3e6614861ec518c680603dca5e391faaf27d4fa6699eefb117f8ecb65e",
            ),
            (
                15,
                FlockProfile::Slim,
                Some(false),
                "60381b6cf0b713e6d8f3c230c019a5a544436f76dee101cf67fdf0e257ea5014",
            ),
            (
                15,
                FlockProfile::Slim8,
                Some(true),
                "4b9a149549f2998fec83bea28478bfc7c4bed26d17f4b8788b44c87c1e52c5be",
            ),
        ];
        for (d_sig, profile, span, want) in cases {
            let manifest = CommitmentManifest::for_batch(1 << d_sig).unwrap();
            let keccak = match (profile, span) {
                (FlockProfile::PublicBits, _) => KeccakPlan::default(),
                (FlockProfile::Fast, Some(s)) => mode_a(d_sig, 1, s),
                (FlockProfile::Slim, Some(s)) => mode_a(d_sig, 2, s),
                (FlockProfile::Slim8, Some(s)) => mode_a(d_sig, 3, s),
                _ => unreachable!(),
            };
            let plan = PowPlan::new(&manifest, profile, &keccak);
            assert_eq!(
                hex(&plan.digest()),
                want,
                "N* = 2^{d_sig} {profile:?} span {span:?}"
            );
        }
    }

    #[test]
    fn plan_totals_match_the_soundness_doc() {
        // Sites, nonces, bits, bytes and sum 2^g of the PIOP part.
        for (d_sig, sites, nonces, bits, work) in [
            (3, 196, 195, 1805, 1298),
            (10, 317, 316, 2906, 1884),
            (13, 398, 397, 3641, 2250),
            (15, 457, 456, 4178, 2546),
        ] {
            let manifest = CommitmentManifest::for_batch(1 << d_sig).unwrap();
            let plan = PowPlan::new(&manifest, FlockProfile::PublicBits, &KeccakPlan::default());
            assert_eq!(plan.sites().len(), sites, "2^{d_sig}");
            assert_eq!(plan.sites().iter().filter(|s| s.bits > 0).count(), nonces);
            assert_eq!(plan.work_bits(), bits);
            assert_eq!(plan.expected_work(), work);
            assert_eq!((plan.sites()[0].class, plan.sites()[0].bits), (SiteClass::ProjPsi, 1));
            assert_eq!((plan.sites()[1].class, plan.sites()[1].bits), (SiteClass::LookupX, 0));
            assert!(plan.sites().iter().all(|s| s.bits <= 6));
        }
        let manifest = CommitmentManifest::for_batch(1 << 15).unwrap();
        let plan = PowPlan::new(&manifest, FlockProfile::PublicBits, &KeccakPlan::default());
        assert_eq!(plan.work_bits().div_ceil(8), 523);
    }

    #[test]
    fn ml_dsa_65_and_87_plans_reproduce_the_script() {
        use crate::manifest::Relation;
        use crate::transcript::{stage_degrees, stage_gammas};
        use agg_witness::params::Set;
        // Stage batching vectors, degrees and gamma difficulties per set under H-exact: 65 has
        // 13, 18 and 105 coefficients (g = 4, 5, 7), 87 has 13, 22 and 132 (g = 4, 5, 8); stage 1
        // has degree 3 (bridge instances only).
        for (set, gammas, bits) in [(Set::S65, [13, 18, 105], [4, 5, 7]), (Set::S87, [13, 22, 132], [4, 5, 8])] {
            for d_sig in [3, 10, 15] {
                let manifest = CommitmentManifest::for_set(set, 1 << d_sig, Relation::HExact).unwrap();
                assert_eq!(stage_gammas(&manifest), gammas, "{set:?}");
                assert_eq!(stage_degrees(&manifest), [3, 5, 3], "{set:?}");
                let plan = PowPlan::new(&manifest, FlockProfile::PublicBits, &KeccakPlan::default());
                let found: Vec<u8> =
                    plan.sites().iter().filter(|s| s.class == SiteClass::StageGamma).map(|s| s.bits).collect();
                assert_eq!(found, bits, "{set:?}");
                let rounds = |stage: u16| {
                    plan.sites().iter().filter(|s| s.class == SiteClass::StageRound && s.i == stage).map(|s| s.bits).collect::<Vec<_>>()
                };
                assert_eq!(rounds(1), vec![2; 8 + d_sig]);
                assert_eq!(rounds(2), vec![3; 8 + d_sig]);
                assert_eq!(rounds(3), vec![2; 8 + d_sig]);
            }
        }
        // Totals of the PIOP part (sites, nonces, bits, sum 2^g) and the mode-B plan digests of
        // `python3 -I docs/scripts/pow_plan.py --set 65|87 --relation h-exact`.
        let cases: [(Set, usize, [usize; 3], u64, &str); 8] = [
            (Set::S65, 3, [190, 189, 1759], 1642, "d99a5de6eb56d98d1f9bff7055dd3caae19fe702e4c287542578698eb2db9ad2"),
            (Set::S65, 10, [311, 310, 2865], 2308, "27a88ced5405c551456dec5fbcc88b93a90a1a82abc0d47e59d34a0091f808c7"),
            (Set::S65, 13, [392, 391, 3603], 2722, "5a2418074a296891fccb781382d203e0a75d094c36013a73355b10555c0560aa"),
            (Set::S65, 15, [451, 450, 4142], 3082, "9f7de8c2876001420754c40e60503a1ddfe6f6e7a441c2cc01915384bdffbc3e"),
            (Set::S87, 3, [190, 189, 1761], 1802, "c4b764d4114adc861df05379ff360fe5f1c20f015a982c4d836e23bc8da41d8b"),
            (Set::S87, 10, [311, 310, 2871], 2556, "4004cf282c9614ac54babee0e5c4452defe20766cc46ea5350afe708276e9847"),
            (Set::S87, 13, [392, 391, 3609], 2970, "73e89f097aff209c49e1dbac41963a41be95ebd31de7cf7fd9fd312d03a7fe2b"),
            (Set::S87, 15, [451, 450, 4147], 3298, "40684a44bd7e84a188c72078fb00af361dfdedeec8a4e18f863e723a20fc0158"),
        ];
        for (set, d_sig, [sites, nonces, bits], work, want) in cases {
            let manifest = CommitmentManifest::for_set(set, 1 << d_sig, Relation::HExact).unwrap();
            let plan = PowPlan::new(&manifest, FlockProfile::PublicBits, &KeccakPlan::default());
            assert_eq!(plan.sites().len(), sites, "{set:?} 2^{d_sig}");
            assert_eq!(plan.sites().iter().filter(|s| s.bits > 0).count(), nonces);
            assert_eq!(plan.work_bits(), bits);
            assert_eq!(plan.expected_work(), work);
            assert_eq!(hex(&plan.digest()), want, "{set:?} 2^{d_sig}");
            assert!(plan.sites().iter().all(|s| s.bits <= 8));
        }
        // The 1K plans: 359 nonce bytes for both sets.
        for (set, bytes) in [(Set::S65, 359), (Set::S87, 359)] {
            let manifest = CommitmentManifest::for_set(set, 1 << 10, Relation::HExact).unwrap();
            let plan = PowPlan::new(&manifest, FlockProfile::PublicBits, &KeccakPlan::default());
            assert_eq!(plan.work_bits().div_ceil(8), bytes);
        }
    }

    #[test]
    fn h_exact_plans_match_the_documented_plans() {
        use crate::manifest::Relation;
        // Totals: (d_sig, sites, nonces, bits, bytes, sum 2^g). With the projection of v the
        // stage-2 gamma site has 10 coefficients under H-exact (def_y1, leaf6[U_hi], leaf8[K_2],
        // leaf8[K_3], range_w1), g = 4 (nonce 11 bits).
        for (d_sig, sites, nonces, bits, bytes, work) in [
            (3, 196, 195, 1805, 226, 1298),
            (10, 317, 316, 2906, 364, 1884),
            (13, 398, 397, 3641, 456, 2250),
            (15, 457, 456, 4178, 523, 2546),
        ] {
            let manifest = CommitmentManifest::for_relation(1 << d_sig, Relation::HExact).unwrap();
            let plan = PowPlan::new(&manifest, FlockProfile::PublicBits, &KeccakPlan::default());
            assert_eq!(plan.sites().len(), sites, "2^{d_sig}");
            assert_eq!(plan.sites().iter().filter(|s| s.bits > 0).count(), nonces);
            assert_eq!(plan.work_bits(), bits);
            assert_eq!(plan.work_bits().div_ceil(8), bytes);
            assert_eq!(plan.expected_work(), work);
            let gamma2 = plan
                .sites()
                .iter()
                .find(|s| s.class == SiteClass::StageGamma && s.i == 2)
                .unwrap();
            assert_eq!((gamma2.bits, gamma2.nonce_bits()), (4, 11));
        }
        // The pinned anchors (mode B = PublicBits; Slim, span on); the plan of the default
        // relation H-FIPS is the same.
        for (d_sig, profile, want) in [
            (3, FlockProfile::PublicBits, "c3dc3ec0258146974daae961a943ec7f8704748cc8683e257bcb02326859cf26"),
            (3, FlockProfile::Slim, "979d6e78ca49e082607c9912fbd1c0f49ed0a33ca03e0052cf4155848fe0ae31"),
            (10, FlockProfile::PublicBits, "3f6d2aac995ed02826a9da18724a301660de37952febdbd67ad991ac2535700f"),
            (10, FlockProfile::Slim, "0a0afe3dd4fa65a6265d2d234b57ca868f793a62ec8d3d3c9949f9af81b2ceac"),
            (15, FlockProfile::PublicBits, "329f278362ccd503ccbbe8ce1f8940b5b1babb2a5a6b0870f44965107cff8a79"),
            (15, FlockProfile::Slim, "bcb8ae3e6614861ec518c680603dca5e391faaf27d4fa6699eefb117f8ecb65e"),
        ] {
            let manifest = CommitmentManifest::for_relation(1 << d_sig, Relation::HExact).unwrap();
            let keccak = match profile {
                FlockProfile::Slim => mode_a(d_sig, 2, true),
                _ => KeccakPlan::default(),
            };
            let digest = hex(&PowPlan::new(&manifest, profile, &keccak).digest());
            assert_eq!(digest, want, "H-exact N* = 2^{d_sig} {profile:?}");
            let fips = CommitmentManifest::for_batch(1 << d_sig).unwrap();
            assert_eq!(fips.relation(), Relation::HFips);
            assert_eq!(hex(&PowPlan::new(&fips, profile, &keccak).digest()), want);
        }
    }

    #[test]
    fn h_fips_plans_equal_h_exact_plans() {
        use crate::manifest::Relation;
        // H-FIPS's stage batching vectors (19, 10, 56) give the same
        // difficulties as H-exact's (19, 10, 50): bit_length(56) = bit_length(50) = 6. The plan,
        // its stream and its digest are H-exact's; the relation is bound by the instance record's
        // manifest bytes (relation byte, G_main4 and G_2 descriptors, catalog row).
        for d_sig in [3, 10, 13, 15] {
            for profile in [FlockProfile::PublicBits, FlockProfile::Slim] {
                let keccak = match profile {
                    FlockProfile::Slim => mode_a(d_sig, 2, true),
                    _ => KeccakPlan::default(),
                };
                let plan = |relation| {
                    let manifest = CommitmentManifest::for_relation(1 << d_sig, relation).unwrap();
                    PowPlan::new(&manifest, profile, &keccak)
                };
                let (fips, exact) = (plan(Relation::HFips), plan(Relation::HExact));
                assert_eq!(fips.work_bits(), exact.work_bits());
                assert_eq!(fips.expected_work(), exact.expected_work());
                assert_eq!(fips.digest(), exact.digest(), "N* = 2^{d_sig} {profile:?}");
                let gamma3 = fips
                    .sites()
                    .iter()
                    .find(|s| s.class == SiteClass::StageGamma && s.i == 3)
                    .unwrap();
                assert_eq!(gamma3.bits, 6);
            }
        }
        let fips = CommitmentManifest::for_relation(1 << 15, Relation::HFips).unwrap();
        let exact = CommitmentManifest::for_relation(1 << 15, Relation::HExact).unwrap();
        assert_ne!(fips.binding_bytes(), exact.binding_bytes());
    }

    #[test]
    fn lane_h_fips_plans_follow_the_longer_stage_three_vector() {
        use crate::manifest::Relation;
        use crate::transcript::{stage_degrees, stage_gammas};
        use agg_pcs_config::catalogs::Set;
        // H-FIPS of ML-DSA-65 and ML-DSA-87: stage 2 trades leaf8[U_mid^(rho)] for F_e^(rho)
        // (same length), stage 3 gains 5 k + 1 coefficients (bool[l_mid^(rho)], F_s, F_delta,
        // F_w1 per row, F_omega, the reductions of delta^(rho), e^(rho); minus link[U_mid^(rho)]):
        // 136 (g = 8, one bit more than H-exact's 105) and 173 (g = 8, as H-exact's 132). Every
        // other site is H-exact's; the lookup keeps its sites (L grows by three padding slots of
        // D_2).
        for (set, gammas, bits) in [(Set::S65, [13, 18, 136], [4, 5, 8]), (Set::S87, [13, 22, 173], [4, 5, 8])] {
            for d_sig in [3, 10, 13, 15] {
                let manifest = |relation| CommitmentManifest::for_set(set, 1 << d_sig, relation).unwrap();
                let fips = manifest(Relation::HFips);
                assert_eq!(stage_gammas(&fips), gammas, "{set:?}");
                assert_eq!(stage_degrees(&fips), [3, 5, 3], "{set:?}");
                let plan = PowPlan::new(&fips, FlockProfile::PublicBits, &KeccakPlan::default());
                let exact = PowPlan::new(&manifest(Relation::HExact), FlockProfile::PublicBits, &KeccakPlan::default());
                let gamma = |p: &PowPlan| -> Vec<u8> {
                    p.sites().iter().filter(|s| s.class == SiteClass::StageGamma).map(|s| s.bits).collect()
                };
                assert_eq!(gamma(&plan), bits, "{set:?}");
                assert_eq!(plan.sites().len(), exact.sites().len());
                // Only batching sites differ, by one bit each: gamma^(3) of ML-DSA-65 and the
                // lookup's stack-batching site, whose vector grows with the padded level-2 stack.
                let differ: Vec<_> = plan.sites().iter().zip(exact.sites()).filter(|(a, b)| a != b).collect();
                for (a, b) in &differ {
                    assert!(matches!(a.class, SiteClass::StageGamma | SiteClass::LookupGamma), "{a:?}");
                    assert_eq!((a.class, a.i, a.j, a.bits), (b.class, b.i, b.j, b.bits + 1), "{set:?} 2^{d_sig}");
                }
                let stage = differ.iter().filter(|(a, _)| a.class == SiteClass::StageGamma).count();
                assert_eq!(stage, usize::from(set == Set::S65), "{set:?} 2^{d_sig}");
                assert_eq!(plan.work_bits(), exact.work_bits() + differ.len(), "{set:?} 2^{d_sig}");
                assert_eq!(plan.digest() == exact.digest(), differ.is_empty(), "{set:?} 2^{d_sig}");
                eprintln!(
                    "{set:?} H-FIPS 2^{d_sig}: sites {}, bits {}, bytes {}, work {}",
                    plan.sites().len(),
                    plan.work_bits(),
                    plan.work_bits().div_ceil(8),
                    plan.expected_work()
                );
            }
        }
    }

    #[test]
    fn plan_digest_depends_on_profile_span_and_ties() {
        let manifest = CommitmentManifest::for_batch(1000).unwrap();
        let work = PartWork {
            tie_bits: 10,
            eq_bits: 10,
            round_bits: 7,
            rounds: 23,
        };
        let base = KeccakPlan {
            span: 1,
            d_part: 10,
            parts: vec![work],
        };
        let d = |p: FlockProfile, k: &KeccakPlan| PowPlan::new(&manifest, p, k).digest();
        let reference = d(FlockProfile::Slim, &base);
        assert_ne!(reference, d(FlockProfile::Fast, &base));
        assert_ne!(
            reference,
            d(
                FlockProfile::Slim,
                &KeccakPlan {
                    span: 2,
                    ..base.clone()
                }
            )
        );
        assert_ne!(
            reference,
            d(
                FlockProfile::Slim,
                &KeccakPlan {
                    d_part: 9,
                    ..base.clone()
                }
            )
        );
        assert_ne!(
            reference,
            d(
                FlockProfile::Slim,
                &KeccakPlan {
                    parts: vec![PartWork {
                        tie_bits: 9,
                        ..work
                    }],
                    ..base.clone()
                }
            )
        );
        assert_ne!(
            reference,
            d(
                FlockProfile::Slim,
                &KeccakPlan {
                    parts: vec![PartWork {
                        round_bits: 8,
                        ..work
                    }],
                    ..base.clone()
                }
            )
        );
        assert_ne!(
            reference,
            d(
                FlockProfile::Slim,
                &KeccakPlan {
                    parts: vec![PartWork {
                        rounds: 22,
                        ..work
                    }],
                    ..base.clone()
                }
            )
        );
        assert_ne!(
            reference,
            d(
                FlockProfile::Slim,
                &KeccakPlan {
                    parts: vec![work, work],
                    ..base
                }
            )
        );
        // The plan depends on N* (N itself is bound by the manifest bytes of the instance record).
        let none = |n: usize| {
            let m = CommitmentManifest::for_batch(n).unwrap();
            PowPlan::new(&m, FlockProfile::None, &KeccakPlan::default()).digest()
        };
        assert_eq!(none(1000), none(1024));
        assert_ne!(none(1000), none(1025));
    }

    #[test]
    fn branch_streams_concatenate_and_split_back() {
        // A trunk, two branches written apart and appended in order, equals one writer.
        let parts: [&[(u32, u8)]; 3] = [&[(3, 8), (100, 9)], &[(7, 11)], &[(1, 8), (300, 10), (0, 8)]];
        let mut serial = WorkWriter::default();
        for part in parts {
            for &(v, w) in part {
                serial.push(v, w);
            }
        }
        let mut writers: Vec<WorkWriter> = parts
            .iter()
            .map(|part| {
                let mut w = WorkWriter::default();
                for &(v, width) in *part {
                    w.push(v, width);
                }
                w
            })
            .collect();
        let tail = writers.pop().unwrap();
        let middle = writers.pop().unwrap();
        let mut joined = writers.pop().unwrap();
        joined.append(&middle);
        joined.append(&tail);
        assert_eq!(joined.bits(), serial.bits());
        let total = serial.bits();
        let bytes = serial.finish();
        assert_eq!(joined.finish(), bytes);
        // The reader splits the middle part off and skips it.
        let mut trunk = WorkReader::new(&bytes, total).unwrap();
        assert_eq!(trunk.read(8).unwrap(), 3);
        assert_eq!(trunk.read(9).unwrap(), 100);
        let mut branch = trunk.split(11).unwrap();
        assert_eq!(trunk.read(8).unwrap(), 1);
        assert_eq!(branch.finish(), Err(WorkStreamError::Unread));
        assert_eq!(branch.read(11).unwrap(), 7);
        assert!(branch.finish().is_ok());
        assert_eq!(branch.read(1), Err(WorkStreamError::Exhausted));
        assert_eq!(trunk.split(19).err(), Some(WorkStreamError::Exhausted));
        assert_eq!(trunk.read(10).unwrap(), 300);
    }

    #[test]
    fn nonce_stream_round_trips_and_rejects_bad_framing() {
        let values = [
            (5_u32, 8_u8),
            (0, 9),
            (511, 9),
            (1, 13),
            (4095, 12),
            (77, 10),
        ];
        let mut w = WorkWriter::default();
        for &(v, width) in &values {
            w.push(v, width);
        }
        let total = w.bits();
        assert_eq!(total, 61);
        let bytes = w.finish();
        assert_eq!(bytes.len(), 8);
        let mut r = WorkReader::new(&bytes, total).unwrap();
        assert_eq!(r.finish(), Err(WorkStreamError::Unread));
        for &(v, width) in &values {
            assert_eq!(r.read(width).unwrap(), v);
        }
        assert!(r.finish().is_ok());
        assert_eq!(r.read(1), Err(WorkStreamError::Exhausted));
        // First nonce in the low bits of byte 0.
        assert_eq!(bytes[0], 5);
        assert_eq!(
            WorkReader::new(&bytes[..7], total).err(),
            Some(WorkStreamError::Length)
        );
        let mut long = bytes.clone();
        long.push(0);
        assert_eq!(
            WorkReader::new(&long, total).err(),
            Some(WorkStreamError::Length)
        );
        let mut padded = bytes.clone();
        padded[7] |= 0x80;
        assert_eq!(
            WorkReader::new(&padded, total).err(),
            Some(WorkStreamError::Padding)
        );
        assert!(WorkReader::new(&[], 0).unwrap().finish().is_ok());
    }
}
