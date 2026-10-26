//! Native Spongefish transcript and the version-0 PIOP event registry (spec Section 10).
//!
//! Exactly the construction of `aerie-falcon/src/protocol/transcript.rs` with our own
//! domain: backend Spongefish `Blake2b512` (`akita_transcript::TranscriptSponge`), protocol
//! tag `agg-mldsa/v1/piop/blake2b` zero-padded to 64 bytes, session `B("agg-mldsa/v1")`,
//! instance `B(digest)` with the cSHAKE256 digest of the canonical instance record, which
//! event 0 then absorbs. Every proof payload is absorbed as one public message (the proof
//! bytes travel in the wire format, not in a Spongefish argument string); challenges are
//! rejection-sampled per 64-bit limb from eight squeezed bytes, two limbs per element of
//! `K`. The ordered registry rejects a missing, extra or out-of-order event before it
//! reaches the sponge. Prover and verifier run the same state machine.
//!
//! Grinding (wire version 2, `docs/soundness.md` Section 6.1): every field-challenge event
//! with difficulty `g > 0` is preceded by a work nonce `v < 2^(g + 7)`, absorbed as the
//! public message `LE32(v)`; the 32-byte predicate squeezed next must have its low `g` bits
//! zero (Akita's `grinding_predicate_accepts`), and the challenge is squeezed after it from
//! the same stream. The prover searches `v = 0, 1, ..` and records the winner in the packed
//! nonce stream ([`crate::pow::WorkWriter`]); the verifier reads and checks it
//! ([`crate::pow::WorkReader`]). The seeds `sd`, `fs_sigma` and the PCS export carry no work
//! (`docs/soundness.md` Section 3.5).

use std::num::NonZeroU8;

use agg_bridge::pow::grind_bits_k;
use akita_transcript::{grinding_predicate_accepts, TranscriptSponge, GRINDING_PREDICATE_LEN};
use jolt_field::{CanonicalEncoding, ExtField};
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::{CShake256, CShake256Core};
use spongefish::{DomainSeparator, DuplexSpongeInterface, Encoding, VerifierState, WithoutInstance};

use crate::lookup::{LookupError, LookupSite, LookupTranscript, DEGREE as LOOKUP_DEGREE};
use crate::manifest::{CommitmentManifest, GroupRole};
use crate::pow::{SiteClass, WorkReader, WorkSite, WorkStreamError, WorkWriter};
use crate::{BaseField, OpeningField};

/// Spongefish protocol tag, zero-padded to 64 bytes.
pub const PIOP_PROTOCOL_TAG: [u8; 64] = protocol_tag();
/// The session label.
pub const PROTOCOL_DOMAIN: &[u8] = b"agg-mldsa/v1";
/// cSHAKE256 customization of the instance digest.
pub const INSTANCE_CUSTOMIZATION: &[u8] = b"agg-mldsa/v1/instance";
/// Stage message degrees (spec Section 7.1): the same in relations H-u and H-exact (stage 2
/// has degree 4 in both, `docs/h-exact.md` 3.5).
pub const STAGE_DEGREES: [usize; 3] = [4, 4, 3];

/// Batching coefficients per stage (spec Section 7, lookup layout L16): `(26, 8, 44)` under
/// H-u, `(26, 7, 45)` under H-exact (`docs/h-exact.md` 3.5).
#[must_use]
pub fn stage_gammas(manifest: &CommitmentManifest) -> [usize; 3] {
    [
        crate::stage1::GAMMA_COUNT,
        crate::stage2::gamma_count(manifest),
        crate::stage3::gamma_count(manifest),
    ]
}

/// Terminal claims per stage: `(18, 8, 31)` under H-u, `(18, 7, 32)` under H-exact.
#[must_use]
pub fn stage_terminals(manifest: &CommitmentManifest) -> [usize; 3] {
    [
        crate::claims::Stage1Claims::LEN,
        crate::claims::Stage2Claims::len(manifest),
        crate::claims::Stage3Claims::len(manifest),
    ]
}
/// Pre-stage claims (spec Section 7.2).
pub const PRESTAGE_CLAIMS: usize = 13;
/// Level-1 carry claims before stage 2 (spec Section 7.4).
pub const KAPPA1_CLAIMS: usize = 6;
/// Multiplicity-digit claims after the lookup.
pub const MULTIPLICITY_CLAIMS: usize = agg_pcs_config::catalogs::MULTIPLICITY_DIGITS;
/// Bytes of one packed final vector `Y_sigma` (512, `agg_bridge::final_vector`).
pub const FINAL_VECTOR_BYTES: usize =
    agg_bridge::final_vector::segment_bytes(&agg_bridge::Shape::PRODUCTION);

pub const INSTANCE_SITE: &str = "agg-mldsa/v1/instance";
pub const FLOCK_COMMIT_SITE: &str = "agg-mldsa/v1/commit/flock";
pub const BRIDGE_SEED_SITE: &str = "agg-mldsa/v1/bridge/seed";
pub const FLOCK_SEEDS_SITE: &str = "agg-mldsa/v1/flock/seeds";
pub const FINAL_VECTORS_SITE: &str = "agg-mldsa/v1/flock/final-vectors";
/// Event 6 (after the final vectors): the `S` framed share items of the Keccak side (empty
/// for a span or a segment of one tied part; mode B: empty).
pub const FLOCK_SHARES_SITE: &str = "agg-mldsa/v1/flock/shares";
/// The join event (fork-join transcript, deviation K22): the `S` framed proof items of the
/// Keccak side, absorbed after the stage-3 terminal claims and before the PCS session
/// export, so the Flock sub-proofs no longer precede the lookup challenge `X`.
pub const FLOCK_PROOFS_SITE: &str = "agg-mldsa/v1/flock/proofs";
/// cSHAKE256 customization of the digest of the event-6 record ([`flock_record_digest`]).
pub const FLOCK_RECORD_CUSTOMIZATION: &[u8] = b"agg-mldsa/v1/flock/record";
pub const TAU_SITE: &str = "agg-mldsa/v1/tau";
pub const PRESTAGE_SITE: &str = "agg-mldsa/v1/prestage/claims";
pub const KAPPA1_SITE: &str = "agg-mldsa/v1/stage2/kappa1";
pub const MULTIPLICITY_SITE: &str = "agg-mldsa/v1/lookup/multiplicities";
/// Prefix of the lookup sites (`agg-mldsa/v1/lookup/<LookupSite::label>`).
pub const LOOKUP_PREFIX: &str = "agg-mldsa/v1/lookup/";
pub const PCS_SESSION_SITE: &str = "agg-mldsa/v1/pcs/session";

const PUBLIC_KIND: u8 = 0;
const PROOF_KIND: u8 = 1;
const CHALLENGE_KIND: u8 = 2;
const EXPORT_KIND: u8 = 4;

const fn protocol_tag() -> [u8; 64] {
    let source = b"agg-mldsa/v1/piop/blake2b";
    let mut tag = [0; 64];
    let mut index = 0;
    while index < source.len() {
        tag[index] = source[index];
        index += 1;
    }
    tag
}

/// The commitment site of a group.
#[must_use]
pub const fn commitment_site(role: GroupRole) -> &'static str {
    match role {
        GroupRole::Main16 => "agg-mldsa/v1/commit/main16",
        GroupRole::Main => "agg-mldsa/v1/commit/main",
        GroupRole::Level1 => "agg-mldsa/v1/commit/level1",
        GroupRole::Level2 => "agg-mldsa/v1/commit/level2",
        GroupRole::Level3 => "agg-mldsa/v1/commit/level3",
        GroupRole::Mult => "agg-mldsa/v1/commit/mult",
    }
}

/// The PIOP transcript site of a lookup event.
#[must_use]
pub fn lookup_site(site: LookupSite) -> String {
    format!("{LOOKUP_PREFIX}{}", site.label())
}

/// The gamma site of a stage (`1..=3`).
#[must_use]
pub fn gamma_site(stage: usize) -> String {
    format!("agg-mldsa/v1/stage{stage}/gamma")
}

/// The terminal site of a stage.
#[must_use]
pub fn terminal_site(stage: usize) -> String {
    format!("agg-mldsa/v1/stage{stage}/terminal")
}

/// The message and challenge sites of one round.
#[must_use]
pub fn round_sites(stage: usize, round: usize) -> (String, String) {
    (
        format!("agg-mldsa/v1/stage{stage}/round/{round}/message"),
        format!("agg-mldsa/v1/stage{stage}/round/{round}/challenge"),
    )
}

/// A transcript replay fault.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TranscriptError {
    /// The next event does not match the registry.
    EventOrder {
        /// Registry index of the expected event.
        index: usize,
        /// The site that was requested.
        site: String,
    },
    /// The payload size or atom count is wrong for the event.
    EventShape {
        /// Registry index.
        index: usize,
    },
    /// The registry ended before the requested event.
    Incomplete {
        /// Registry index.
        index: usize,
    },
    /// A work nonce fails its predicate.
    Work {
        /// Registry index of the challenge event.
        index: usize,
    },
    /// The honest nonce search exhausted `2^(g + 7)` candidates (probability below
    /// `2^-184` per site).
    WorkExhausted {
        /// Registry index of the challenge event.
        index: usize,
    },
    /// The packed nonce stream is malformed.
    WorkStream(WorkStreamError),
}

impl std::fmt::Display for TranscriptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EventOrder { index, site } => {
                write!(f, "transcript event {index}: unexpected site {site}")
            }
            Self::EventShape { index } => write!(f, "transcript event {index}: wrong shape"),
            Self::Incomplete { index } => write!(f, "transcript ended before event {index}"),
            Self::Work { index } => write!(f, "transcript event {index}: work nonce rejected"),
            Self::WorkExhausted { index } => {
                write!(f, "transcript event {index}: nonce search exhausted")
            }
            Self::WorkStream(e) => write!(f, "nonce stream: {e:?}"),
        }
    }
}

impl std::error::Error for TranscriptError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PayloadSize {
    Exact(usize),
    /// Framed items of variable length (the Flock commitments and proofs).
    Variable,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RegistryEvent {
    site: String,
    kind: u8,
    atoms: usize,
    size: PayloadSize,
    /// The protected site of a field-challenge event (`docs/soundness.md` Section 3.1).
    work: Option<WorkSite>,
}

/// Event list under construction.
struct Builder(Vec<RegistryEvent>);

impl Builder {
    fn push(&mut self, site: &str, kind: u8, atoms: usize, size: PayloadSize) {
        self.0.push(RegistryEvent {
            site: site.to_owned(),
            kind,
            atoms,
            size,
            work: None,
        });
    }

    /// A field-challenge event of `atoms` elements of `K`, protected as site `(class, i, j)`
    /// with assigned bound `numerator / |K|` (`None`: the joint-bound site `X`, no work).
    fn challenge(&mut self, site: &str, atoms: usize, class: SiteClass, i: usize, j: usize, numerator: Option<u64>) {
        self.push(site, CHALLENGE_KIND, atoms, PayloadSize::Exact(16 * atoms));
        let index = |v: usize| u16::try_from(v).expect("site index below 2^16");
        self.0.last_mut().expect("pushed").work = Some(WorkSite {
            class,
            i: index(i),
            j: index(j),
            bits: numerator.map_or(0, grind_bits_k),
        });
    }
}

/// The ordered event list of one batch (spec Section 10).
#[derive(Clone, Debug)]
pub struct Registry {
    events: Vec<RegistryEvent>,
}

impl Registry {
    /// The registry of `manifest`, with the difficulty of every field-challenge site
    /// (`docs/soundness.md` Section 3.1: Aerie's assigned bounds, a vector of `l`
    /// coefficients `l / |K|`, a round its degree over `|K|`, a merge `1 / |K|`, the point
    /// `tau` `d_main / |K|`; `X` carries none, its joint bound is `((L + M - 1) / |K|)^2`).
    #[must_use]
    pub fn new(manifest: &CommitmentManifest) -> Self {
        let rounds = manifest.d_main();
        let segments = manifest.segments();
        let mut b = Builder(Vec::with_capacity(32 + 6 * rounds + 4 * rounds * rounds));
        let elements = |count: usize| PayloadSize::Exact(16 * count);
        let count = |n: usize| Some(n as u64);
        b.push(INSTANCE_SITE, PUBLIC_KIND, 1, PayloadSize::Variable);
        for role in [GroupRole::Main16, GroupRole::Main] {
            b.push(commitment_site(role), PROOF_KIND, 1, PayloadSize::Variable);
        }
        b.push(FLOCK_COMMIT_SITE, PROOF_KIND, segments, PayloadSize::Variable);
        b.push(BRIDGE_SEED_SITE, CHALLENGE_KIND, 1, PayloadSize::Exact(32));
        for role in [GroupRole::Level1, GroupRole::Level2, GroupRole::Level3, GroupRole::Mult] {
            b.push(commitment_site(role), PROOF_KIND, 1, PayloadSize::Variable);
        }
        b.push(
            FLOCK_SEEDS_SITE,
            CHALLENGE_KIND,
            segments,
            PayloadSize::Exact(32 * segments),
        );
        b.push(
            FINAL_VECTORS_SITE,
            PROOF_KIND,
            segments,
            PayloadSize::Exact(FINAL_VECTOR_BYTES * segments),
        );
        b.push(FLOCK_SHARES_SITE, PROOF_KIND, segments, PayloadSize::Variable);
        // The lookup (docs/lookup.md Section 1): X, layer 1, mu_1, the sessions, then the
        // multiplicity-digit claims.
        let shape = manifest.lookup_shape();
        let challenges = shape.challenges;
        let proof = |b: &mut Builder, site: LookupSite, atoms: usize| {
            b.push(&lookup_site(site), PROOF_KIND, atoms, elements(atoms));
        };
        b.challenge(&lookup_site(LookupSite::X), challenges, SiteClass::LookupX, 0, 0, None);
        proof(&mut b, LookupSite::Layer1, 4 * shape.tree_count());
        b.challenge(&lookup_site(LookupSite::Mu(1)), 1, SiteClass::LookupMu, 1, 0, count(1));
        for s in 2..=shape.max_depth() {
            let (active, continuing, ending) = shape.session_counts(s);
            b.challenge(
                &lookup_site(LookupSite::Gamma(s)),
                2 * active,
                SiteClass::LookupGamma,
                s,
                0,
                count(2 * active),
            );
            for round in 0..s - 1 {
                proof(&mut b, LookupSite::Round(s, round), LOOKUP_DEGREE);
                b.challenge(
                    &lookup_site(LookupSite::RoundChallenge(s, round)),
                    1,
                    SiteClass::LookupRound,
                    s,
                    round,
                    count(LOOKUP_DEGREE),
                );
            }
            proof(&mut b, LookupSite::Terminals(s), 4 * continuing);
            proof(&mut b, LookupSite::Leaves(s), 2 * ending);
            b.challenge(&lookup_site(LookupSite::Mu(s)), 1, SiteClass::LookupMu, s, 0, count(1));
        }
        b.push(MULTIPLICITY_SITE, PROOF_KIND, MULTIPLICITY_CLAIMS, elements(MULTIPLICITY_CLAIMS));
        b.challenge(TAU_SITE, rounds, SiteClass::Tau, 0, 0, count(rounds));
        b.push(
            PRESTAGE_SITE,
            PROOF_KIND,
            PRESTAGE_CLAIMS,
            PayloadSize::Exact(16 * PRESTAGE_CLAIMS),
        );
        let (stage_gammas, stage_terminals) = (stage_gammas(manifest), stage_terminals(manifest));
        for stage in 1..=3 {
            if stage == 2 {
                b.push(
                    KAPPA1_SITE,
                    PROOF_KIND,
                    KAPPA1_CLAIMS,
                    PayloadSize::Exact(16 * KAPPA1_CLAIMS),
                );
            }
            let gammas = stage_gammas[stage - 1];
            b.challenge(&gamma_site(stage), gammas, SiteClass::StageGamma, stage, 0, count(gammas));
            let width = STAGE_DEGREES[stage - 1];
            for round in 0..rounds {
                let (message, challenge) = round_sites(stage, round);
                b.push(&message, PROOF_KIND, width, PayloadSize::Exact(16 * width));
                b.challenge(&challenge, 1, SiteClass::StageRound, stage, round, count(width));
            }
            let terminals = stage_terminals[stage - 1];
            b.push(
                &terminal_site(stage),
                PROOF_KIND,
                terminals,
                PayloadSize::Exact(16 * terminals),
            );
        }
        // The join (deviation K22): the Keccak side's proofs, after every PIOP challenge and
        // before the PCS session digest, which therefore binds every Flock byte.
        b.push(FLOCK_PROOFS_SITE, PROOF_KIND, segments, PayloadSize::Variable);
        b.push(PCS_SESSION_SITE, EXPORT_KIND, 1, PayloadSize::Exact(32));
        Self { events: b.0 }
    }

    /// The protected field-challenge sites in event order (the PIOP part of the plan).
    pub fn work_sites(&self) -> impl Iterator<Item = WorkSite> + '_ {
        self.events.iter().filter_map(|e| e.work)
    }

    /// Bits of the packed PIOP nonce stream.
    #[must_use]
    pub fn work_bits(&self) -> usize {
        self.work_sites().map(|w| usize::from(w.nonce_bits())).sum()
    }

    /// Number of events.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// True when empty (never).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

/// cSHAKE256 digest with a customization string.
#[must_use]
pub fn cshake256(customization: &[u8], data: &[u8], out_len: usize) -> Vec<u8> {
    let mut hash = CShake256::from_core(CShake256Core::new(customization));
    hash.update(data);
    let mut out = vec![0_u8; out_len];
    hash.finalize_xof().read(&mut out);
    out
}

/// The digest of the event-6 record: `LE64(S)`, the packed final vectors `Y_0..Y_{S-1}` and
/// the `S` share items framed as absorbed (`LE64(len) || bytes`), cSHAKE256 with
/// customization [`FLOCK_RECORD_CUSTOMIZATION`]. The Keccak side binds it into every tied
/// part's challengers (mode A: through the part and span seeds), so that every Flock
/// challenge address encodes the input of the lookup challenge `X` (`docs/soundness.md`
/// 12.2, Lemma 12.3 (a); deviation K22).
#[must_use]
pub fn flock_record_digest(final_vectors: &[[u8; FINAL_VECTOR_BYTES]], shares: &[Vec<u8>]) -> [u8; 32] {
    let mut data = Vec::with_capacity(8 + FINAL_VECTOR_BYTES * final_vectors.len());
    data.extend_from_slice(&(final_vectors.len() as u64).to_le_bytes());
    for y in final_vectors {
        data.extend_from_slice(y);
    }
    data.extend_from_slice(&frame_items(shares));
    cshake256(FLOCK_RECORD_CUSTOMIZATION, &data, 32)
        .try_into()
        .expect("32 bytes")
}

/// The 32-byte instance digest of the canonical instance record.
#[must_use]
pub fn instance_digest(record: &[u8]) -> [u8; 32] {
    let mut framed = Vec::with_capacity(8 + record.len());
    framed.extend_from_slice(&(record.len() as u64).to_le_bytes());
    framed.extend_from_slice(record);
    cshake256(INSTANCE_CUSTOMIZATION, &framed, 32)
        .try_into()
        .expect("32 bytes")
}

#[derive(Clone, Copy)]
struct LengthPrefixedBytes<'a>(&'a [u8]);

impl Encoding<[u8]> for LengthPrefixedBytes<'_> {
    fn encode(&self) -> impl AsRef<[u8]> {
        let mut bytes = Vec::with_capacity(8 + self.0.len());
        bytes.extend_from_slice(&(self.0.len() as u64).to_le_bytes());
        bytes.extend_from_slice(self.0);
        bytes
    }
}

/// Canonical extension atom encoding: `LE64(c0) || LE64(c1)` per element.
#[must_use]
pub fn encode_extensions(values: &[OpeningField]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16 * values.len());
    for value in values {
        for coefficient in value.to_base_vec() {
            bytes.extend_from_slice(&coefficient.to_canonical_u64().to_le_bytes());
        }
    }
    bytes
}

/// Decode the canonical atom encoding; `None` on a non-canonical limb or a bad length.
#[must_use]
pub fn decode_extensions(bytes: &[u8]) -> Option<Vec<OpeningField>> {
    if !bytes.len().is_multiple_of(16) {
        return None;
    }
    bytes
        .as_chunks::<16>()
        .0
        .iter()
        .map(|chunk| {
            let c0 = BaseField::from_bytes_le_checked(&chunk[..8])?;
            let c1 = BaseField::from_bytes_le_checked(&chunk[8..])?;
            Some(OpeningField::from_base_slice(&[c0, c1]))
        })
        .collect()
}

/// Frame a list of byte strings: `LE64(len) || bytes` each.
#[must_use]
pub fn frame_items(items: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    for item in items {
        out.extend_from_slice(&(item.len() as u64).to_le_bytes());
        out.extend_from_slice(item);
    }
    out
}

/// The PIOP transcript, shared by prover and verifier (both replay the same public
/// messages; no Spongefish argument string is produced).
pub struct Transcript {
    state: VerifierState<'static, TranscriptSponge>,
    registry: Registry,
    next: usize,
    work: Work,
}

/// The nonce stream: written by the prover's search, read by the verifier.
enum Work {
    Search(WorkWriter),
    Replay(WorkReader),
}

/// The sponge after absorbing `LE32(nonce)` and the predicate squeezed from it.
fn preview(sponge: &TranscriptSponge, nonce: u32) -> (TranscriptSponge, [u8; GRINDING_PREDICATE_LEN]) {
    let mut sponge = sponge.clone();
    sponge.absorb(&nonce.to_le_bytes());
    let mut predicate = [0; GRINDING_PREDICATE_LEN];
    sponge.squeeze(&mut predicate);
    (sponge, predicate)
}

/// Bytes of the packed PIOP nonce stream of `manifest` (`docs/soundness.md` Section 6.2).
#[must_use]
pub fn work_bytes(manifest: &CommitmentManifest) -> usize {
    Registry::new(manifest).work_bits().div_ceil(8)
}

impl Transcript {
    /// The prover's stream: initialize from the canonical instance record and absorb
    /// event 0; the work nonces are searched and recorded ([`Self::work_stream`]).
    pub fn new(manifest: &CommitmentManifest, record: &[u8]) -> Result<Self, TranscriptError> {
        Self::with_work(manifest, record, Work::Search(WorkWriter::default()))
    }

    /// The verifier's stream: as [`Self::new`], the work nonces read from `work` (exact
    /// length and zero padding checked here, every nonce consumed at
    /// [`Self::ensure_complete`]).
    pub fn replay(
        manifest: &CommitmentManifest,
        record: &[u8],
        work: &[u8],
    ) -> Result<Self, TranscriptError> {
        let registry = Registry::new(manifest);
        let reader =
            WorkReader::new(work, registry.work_bits()).map_err(TranscriptError::WorkStream)?;
        Self::with_work(manifest, record, Work::Replay(reader))
    }

    fn with_work(
        manifest: &CommitmentManifest,
        record: &[u8],
        work: Work,
    ) -> Result<Self, TranscriptError> {
        let digest = instance_digest(record);
        let instance = LengthPrefixedBytes(&digest);
        let domain = DomainSeparator::<WithoutInstance>::new(PIOP_PROTOCOL_TAG)
            .session(LengthPrefixedBytes(PROTOCOL_DOMAIN))
            .instance(&instance);
        let state = domain.to_verifier(TranscriptSponge::default(), &[]);
        let mut this = Self {
            state,
            registry: Registry::new(manifest),
            next: 0,
            work,
        };
        this.absorb_kind(PUBLIC_KIND, INSTANCE_SITE, 1, record)?;
        Ok(this)
    }

    /// The prover's packed nonce stream so far (padding bits zero).
    ///
    /// # Panics
    /// On a verifier's stream.
    #[must_use]
    pub fn work_stream(&self) -> Vec<u8> {
        match &self.work {
            Work::Search(writer) => writer.clone().finish(),
            Work::Replay(_) => panic!("the verifier's transcript records no nonces"),
        }
    }

    /// The work of site `site` at registry event `index` (`docs/soundness.md` Section 6.1).
    fn grind(&mut self, index: usize, site: WorkSite) -> Result<(), TranscriptError> {
        let Some(bits) = NonZeroU8::new(site.bits) else {
            return Ok(());
        };
        let width = site.nonce_bits();
        let sponge = &self.state.duplex_sponge_state;
        let (next, nonce) = match &mut self.work {
            Work::Search(writer) => {
                let found = (0..1_u64 << width).find_map(|candidate| {
                    let nonce = u32::try_from(candidate).expect("nonce below 2^32");
                    let (next, predicate) = preview(sponge, nonce);
                    grinding_predicate_accepts(&predicate, bits).then_some((next, nonce))
                });
                let (next, nonce) = found.ok_or(TranscriptError::WorkExhausted { index })?;
                writer.push(nonce, width);
                (next, nonce)
            }
            Work::Replay(reader) => {
                let nonce = reader.read(width).map_err(TranscriptError::WorkStream)?;
                let (next, predicate) = preview(sponge, nonce);
                if !grinding_predicate_accepts(&predicate, bits) {
                    return Err(TranscriptError::Work { index });
                }
                (next, nonce)
            }
        };
        debug_assert!(u64::from(nonce) < 1 << width);
        self.state.duplex_sponge_state = next;
        Ok(())
    }

    fn take(
        &mut self,
        site: &str,
        kind: u8,
        atoms: usize,
        payload_bytes: usize,
    ) -> Result<usize, TranscriptError> {
        let index = self.next;
        let Some(expected) = self.registry.events.get(index) else {
            return Err(TranscriptError::Incomplete { index });
        };
        if expected.site != site || expected.kind != kind {
            return Err(TranscriptError::EventOrder {
                index,
                site: site.to_owned(),
            });
        }
        let size_ok = match expected.size {
            PayloadSize::Exact(bytes) => bytes == payload_bytes,
            PayloadSize::Variable => true,
        };
        if expected.atoms != atoms || !size_ok {
            return Err(TranscriptError::EventShape { index });
        }
        self.next += 1;
        Ok(index)
    }

    fn absorb_kind(
        &mut self,
        kind: u8,
        site: &str,
        atoms: usize,
        payload: &[u8],
    ) -> Result<(), TranscriptError> {
        self.take(site, kind, atoms, payload.len())?;
        self.state.public_message(payload);
        Ok(())
    }

    /// Absorb one proof payload (kind 1) as one message.
    pub fn absorb(&mut self, site: &str, atoms: usize, payload: &[u8]) -> Result<(), TranscriptError> {
        self.absorb_kind(PROOF_KIND, site, atoms, payload)
    }

    /// Absorb a vector of extension elements.
    pub fn absorb_extensions(
        &mut self,
        site: &str,
        values: &[OpeningField],
    ) -> Result<(), TranscriptError> {
        self.absorb(site, values.len(), &encode_extensions(values))
    }

    /// Absorb a list of framed byte strings as one event of `items.len()` atoms.
    pub fn absorb_items(&mut self, site: &str, items: &[Vec<u8>]) -> Result<(), TranscriptError> {
        self.absorb(site, items.len(), &frame_items(items))
    }

    fn squeeze(&mut self, bytes: &mut [u8]) {
        self.state.duplex_sponge_state.squeeze(bytes);
    }

    fn sample_base(&mut self) -> BaseField {
        loop {
            let mut candidate = [0; 8];
            self.squeeze(&mut candidate);
            if let Some(value) = BaseField::from_bytes_le_checked(&candidate) {
                return value;
            }
        }
    }

    /// Draw `count` independent elements of `K` under one challenge event.
    pub fn challenge_extensions(
        &mut self,
        site: &str,
        count: usize,
    ) -> Result<Vec<OpeningField>, TranscriptError> {
        let index = self.take(site, CHALLENGE_KIND, count, 16 * count)?;
        if let Some(work) = self.registry.events[index].work {
            self.grind(index, work)?;
        }
        Ok((0..count)
            .map(|_| {
                let c0 = self.sample_base();
                let c1 = self.sample_base();
                OpeningField::from_base_slice(&[c0, c1])
            })
            .collect())
    }

    /// Draw `atoms * 32` raw bytes under one challenge event (no field sampling).
    pub fn challenge_bytes(&mut self, site: &str, atoms: usize) -> Result<Vec<u8>, TranscriptError> {
        self.take(site, CHALLENGE_KIND, atoms, 32 * atoms)?;
        let mut out = vec![0; 32 * atoms];
        self.squeeze(&mut out);
        Ok(out)
    }

    /// Export the 32-byte PCS session digest after the stage-3 terminal (a clone of the
    /// sponge, so the native PCS stream is separately domain-separated).
    pub fn pcs_session(&mut self) -> Result<[u8; 32], TranscriptError> {
        self.take(PCS_SESSION_SITE, EXPORT_KIND, 1, 32)?;
        let mut sponge = self.state.duplex_sponge_state.clone();
        let mut session = [0; 32];
        sponge.squeeze(&mut session);
        Ok(session)
    }

    /// Confirm the registry index reached `index`.
    pub fn require_next(&self, index: usize) -> Result<(), TranscriptError> {
        if self.next != index {
            return Err(TranscriptError::Incomplete { index: self.next });
        }
        Ok(())
    }

    /// Confirm every registered event ran (and, on the verifier's stream, that every
    /// nonce was read).
    pub fn ensure_complete(&self) -> Result<(), TranscriptError> {
        self.require_next(self.registry.len())?;
        match &self.work {
            Work::Search(_) => Ok(()),
            Work::Replay(reader) => reader.finish().map_err(TranscriptError::WorkStream),
        }
    }
}

impl LookupTranscript for Transcript {
    fn absorb(&mut self, site: LookupSite, values: &[OpeningField]) -> Result<(), LookupError> {
        self.absorb_extensions(&lookup_site(site), values)
            .map_err(|e| LookupError::Transcript(e.to_string()))
    }

    fn challenges(&mut self, site: LookupSite, count: usize) -> Result<Vec<OpeningField>, LookupError> {
        self.challenge_extensions(&lookup_site(site), count)
            .map_err(|e| LookupError::Transcript(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Ring;

    fn manifest() -> CommitmentManifest {
        CommitmentManifest::for_batch(100).unwrap()
    }

    /// The events before the lookup: instance, two round-1 commitments, Flock commitments,
    /// seed, four round-2 commitments, Flock seeds, final vectors, Flock shares.
    const LEADING: usize = 12;

    fn lookup_events(manifest: &CommitmentManifest) -> usize {
        let shape = manifest.lookup_shape();
        3 + (2..=shape.max_depth()).map(|s| 4 + 2 * (s - 1)).sum::<usize>() + 1
    }

    #[test]
    fn registry_has_the_spec_event_order() {
        let manifest = manifest();
        let registry = Registry::new(&manifest);
        let rounds = 17;
        let lookup = lookup_events(&manifest);
        // Leading events, the lookup, tau and the pre-stage claims, the kappa1 absorb,
        // three stages of gamma + rounds + terminal, the join, the PCS export.
        assert_eq!(
            registry.len(),
            LEADING + lookup + 2 + 1 + 3 * (2 + 2 * rounds) + 1 + 1
        );
        let sites: Vec<&str> = registry.events.iter().map(|e| e.site.as_str()).collect();
        assert_eq!(
            &sites[..LEADING + 2],
            &[
                INSTANCE_SITE,
                "agg-mldsa/v1/commit/main16",
                "agg-mldsa/v1/commit/main",
                FLOCK_COMMIT_SITE,
                BRIDGE_SEED_SITE,
                "agg-mldsa/v1/commit/level1",
                "agg-mldsa/v1/commit/level2",
                "agg-mldsa/v1/commit/level3",
                "agg-mldsa/v1/commit/mult",
                FLOCK_SEEDS_SITE,
                FINAL_VECTORS_SITE,
                FLOCK_SHARES_SITE,
                "agg-mldsa/v1/lookup/x",
                "agg-mldsa/v1/lookup/layer1",
            ]
        );
        let after = LEADING + lookup;
        assert_eq!(sites[after - 1], MULTIPLICITY_SITE);
        assert_eq!(&sites[after..after + 2], &[TAU_SITE, PRESTAGE_SITE]);
        assert_eq!(sites[after + 2], "agg-mldsa/v1/stage1/gamma");
        assert_eq!(sites[after + 3], "agg-mldsa/v1/stage1/round/0/message");
        assert_eq!(sites[after + 3 + 2 * rounds], "agg-mldsa/v1/stage1/terminal");
        assert_eq!(sites[after + 4 + 2 * rounds], KAPPA1_SITE);
        // The join: the Flock proofs after the stage-3 terminal, right before the export.
        assert_eq!(sites[registry.len() - 3], "agg-mldsa/v1/stage3/terminal");
        assert_eq!(sites[registry.len() - 2], FLOCK_PROOFS_SITE);
        assert_eq!(registry.events[registry.len() - 2].atoms, 1, "one segment at N = 100");
        assert_eq!(sites[registry.len() - 1], PCS_SESSION_SITE);
        assert_eq!(sites.iter().filter(|s| **s == FLOCK_PROOFS_SITE).count(), 1);
        let kinds: Vec<u8> = registry.events.iter().map(|e| e.kind).collect();
        assert_eq!(&kinds[..5], &[0, 1, 1, 1, 2]);
        assert_eq!(registry.events[4].size, PayloadSize::Exact(32));
        assert_eq!(registry.events[9].atoms, 1, "one segment at N = 100");
        // Two lookup challenges, 4 layer-1 values per tree (2 x (4 stacks + 1 table)).
        assert_eq!(registry.events[LEADING].atoms, 2);
        assert_eq!(registry.events[LEADING + 1].atoms, 40);
        assert_eq!(registry.events[after].atoms, rounds);
        let full = Registry::new(&CommitmentManifest::for_batch(1 << 15).unwrap());
        assert_eq!(full.events[9].size, PayloadSize::Exact(128));
        assert_eq!(full.events[10].size, PayloadSize::Exact(4 * 512));
    }

    /// Absorb events 1 to 3 with dummy payloads.
    fn to_seed(t: &mut Transcript, commitment: &[u8]) -> Vec<u8> {
        t.absorb(commitment_site(GroupRole::Main16), 1, commitment).unwrap();
        t.absorb(commitment_site(GroupRole::Main), 1, &[1]).unwrap();
        t.absorb_items(FLOCK_COMMIT_SITE, &[vec![]]).unwrap();
        t.challenge_bytes(BRIDGE_SEED_SITE, 1).unwrap()
    }

    #[test]
    fn events_must_follow_the_registry() {
        let manifest = manifest();
        let record = manifest.binding_bytes();
        let mut t = Transcript::new(&manifest, &record).unwrap();
        // Skipping the main commitments is an order fault; a wrong seed size a shape fault.
        assert!(matches!(
            t.absorb_items(FLOCK_COMMIT_SITE, &[]),
            Err(TranscriptError::EventOrder { index: 1, .. })
        ));
        t.absorb(commitment_site(GroupRole::Main16), 1, &[1, 2, 3])
            .unwrap();
        assert!(matches!(
            t.absorb_items(FLOCK_COMMIT_SITE, &[]),
            Err(TranscriptError::EventOrder { index: 2, .. })
        ));
        t.absorb(commitment_site(GroupRole::Main), 1, &[4]).unwrap();
        t.absorb_items(FLOCK_COMMIT_SITE, &[vec![9; 4]]).unwrap();
        assert!(matches!(
            t.challenge_bytes(BRIDGE_SEED_SITE, 2),
            Err(TranscriptError::EventShape { index: 4 })
        ));
        let seed = t.challenge_bytes(BRIDGE_SEED_SITE, 1).unwrap();
        assert_eq!(seed.len(), 32);
        assert!(t.require_next(5).is_ok());
        assert!(t.ensure_complete().is_err());
    }

    #[test]
    fn replay_is_deterministic_and_sensitive_to_every_byte() {
        let manifest = manifest();
        let record = manifest.binding_bytes();
        let run = |record: &[u8], commitment: &[u8]| {
            let mut t = Transcript::new(&manifest, record).unwrap();
            to_seed(&mut t, commitment)
        };
        let base = run(&record, &[7; 40]);
        assert_eq!(base, run(&record, &[7; 40]));
        assert_ne!(base, run(&record, &[8; 40]));
        let mut other = record.clone();
        other[0] ^= 1;
        assert_ne!(base, run(&other, &[7; 40]));
    }

    /// Events 1 to 6 with dummy payloads.
    fn to_lookup(t: &mut Transcript) {
        to_seed(t, &[0]);
        for role in [GroupRole::Level1, GroupRole::Level2, GroupRole::Level3, GroupRole::Mult] {
            t.absorb(commitment_site(role), 1, &[0]).unwrap();
        }
        t.challenge_bytes(FLOCK_SEEDS_SITE, 1).unwrap();
        t.absorb(FINAL_VECTORS_SITE, 1, &[0; 512]).unwrap();
        t.absorb_items(FLOCK_SHARES_SITE, &[vec![]]).unwrap();
    }

    #[test]
    fn registry_difficulties_follow_the_soundness_doc() {
        // docs/soundness.md 3.1 at N = 100 (N* = 128): d_main = 17.
        let registry = Registry::new(&manifest());
        let sites: Vec<WorkSite> = registry.work_sites().collect();
        assert_eq!(sites.len(), 264);
        assert_eq!((sites[0].class, sites[0].bits), (SiteClass::LookupX, 0));
        assert_eq!((sites[1].class, sites[1].i, sites[1].bits), (SiteClass::LookupMu, 1, 1));
        // Session 2: all ten trees active, 20 coefficients, 5 bits; one round of 2 bits.
        assert_eq!((sites[2].class, sites[2].i, sites[2].bits), (SiteClass::LookupGamma, 2, 5));
        assert_eq!((sites[3].class, sites[3].i, sites[3].j, sites[3].bits), (SiteClass::LookupRound, 2, 0, 2));
        let tau = sites.iter().find(|s| s.class == SiteClass::Tau).unwrap();
        assert_eq!(tau.bits, 5);
        let gammas: Vec<u8> = sites.iter().filter(|s| s.class == SiteClass::StageGamma).map(|s| s.bits).collect();
        assert_eq!(gammas, [5, 4, 6]);
        for (stage, bits) in [(1, 3), (2, 3), (3, 2)] {
            let rounds: Vec<&WorkSite> = sites
                .iter()
                .filter(|s| s.class == SiteClass::StageRound && s.i == stage)
                .collect();
            assert_eq!(rounds.len(), 17);
            assert!(rounds.iter().all(|s| s.bits == bits));
            assert_eq!(rounds[16].j, 16);
        }
        // 2,442 stream bits at N* = 2^7 (docs/soundness.md Section 10).
        assert_eq!(registry.work_bits(), 2442);
        assert_eq!(work_bytes(&manifest()), 306);
        // H-exact (docs/h-exact.md 3.7): gamma^(2) has 7 coefficients (g = 3), gamma^(3) 45
        // (g = 6), the rounds keep their difficulties; one stream bit fewer.
        let exact = CommitmentManifest::for_relation(100, crate::manifest::Relation::HExact).unwrap();
        assert_eq!(stage_gammas(&exact), [26, 7, 45]);
        assert_eq!(stage_terminals(&exact), [18, 7, 32]);
        assert_eq!(stage_gammas(&manifest()), [26, 8, 44]);
        assert_eq!(stage_terminals(&manifest()), [18, 8, 31]);
        let exact_registry = Registry::new(&exact);
        let gammas: Vec<u8> = exact_registry
            .work_sites()
            .filter(|s| s.class == SiteClass::StageGamma)
            .map(|s| s.bits)
            .collect();
        assert_eq!(gammas, [5, 3, 6]);
        assert_eq!(exact_registry.work_bits(), 2441);
        assert_eq!(work_bytes(&exact), 306);
        // The challenge field is K = F_{p^2} with p = 2^64 - 59, as the difficulties assume.
        assert_eq!(BaseField::from_u64(agg_bridge::pow::P), BaseField::from_u64(0));
    }

    #[test]
    fn work_nonces_are_searched_recorded_and_checked() {
        let manifest = manifest();
        let record = manifest.binding_bytes();
        // The prover through X and mu_1 (1 bit of work, a nonce of 8 bits).
        let mut prover = Transcript::new(&manifest, &record).unwrap();
        to_lookup(&mut prover);
        let x = LookupTranscript::challenges(&mut prover, LookupSite::X, 2).unwrap();
        prover.absorb_extensions(&lookup_site(LookupSite::Layer1), &vec![OpeningField::from_u64(3); 40]).unwrap();
        let mu = LookupTranscript::challenges(&mut prover, LookupSite::Mu(1), 1).unwrap();
        let stream = prover.work_stream();
        assert_eq!(stream.len(), 1, "X carries no nonce, mu_1 one of 8 bits");
        let nonce = stream[0];
        // A full-length stream for the replay (the remaining nonces zero).
        let full = |first: u8| {
            let mut bytes = vec![0; work_bytes(&manifest)];
            bytes[0] = first;
            bytes
        };
        let replay = |first: u8| -> Result<(Vec<OpeningField>, Vec<OpeningField>), TranscriptError> {
            let mut t = Transcript::replay(&manifest, &record, &full(first))?;
            to_lookup(&mut t);
            let x = t.challenge_extensions(&lookup_site(LookupSite::X), 2)?;
            t.absorb_extensions(&lookup_site(LookupSite::Layer1), &vec![OpeningField::from_u64(3); 40])?;
            let mu = t.challenge_extensions(&lookup_site(LookupSite::Mu(1)), 1)?;
            Ok((x, mu))
        };
        assert_eq!(replay(nonce).unwrap(), (x.clone(), mu));
        // The prover's nonce is the smallest passing one: every smaller one fails.
        let mu_index = LEADING + 2;
        for wrong in 0..nonce {
            assert_eq!(replay(wrong), Err(TranscriptError::Work { index: mu_index }));
        }
        // The stream's length and padding are exact.
        let bytes = full(nonce);
        let short = &bytes[..bytes.len() - 1];
        assert!(matches!(
            Transcript::replay(&manifest, &record, short),
            Err(TranscriptError::WorkStream(WorkStreamError::Length))
        ));
        let mut long = bytes.clone();
        long.push(0);
        assert!(matches!(
            Transcript::replay(&manifest, &record, &long),
            Err(TranscriptError::WorkStream(WorkStreamError::Length))
        ));
        let mut padded = bytes.clone();
        *padded.last_mut().unwrap() |= 0x80; // 2,442 bits: the last 6 bits are padding
        assert!(matches!(
            Transcript::replay(&manifest, &record, &padded),
            Err(TranscriptError::WorkStream(WorkStreamError::Padding))
        ));
        // An unread nonce fails the completeness check.
        let t = Transcript::replay(&manifest, &record, &bytes).unwrap();
        assert!(t.ensure_complete().is_err());
    }

    #[test]
    fn the_flock_proofs_cannot_be_absorbed_at_event_6() {
        let manifest = manifest();
        let mut t = Transcript::new(&manifest, &manifest.binding_bytes()).unwrap();
        to_seed(&mut t, &[0]);
        for role in [GroupRole::Level1, GroupRole::Level2, GroupRole::Level3, GroupRole::Mult] {
            t.absorb(commitment_site(role), 1, &[0]).unwrap();
        }
        t.challenge_bytes(FLOCK_SEEDS_SITE, 1).unwrap();
        t.absorb(FINAL_VECTORS_SITE, 1, &[0; 512]).unwrap();
        assert!(matches!(
            t.absorb_items(FLOCK_PROOFS_SITE, &[vec![]]),
            Err(TranscriptError::EventOrder { index: 11, .. })
        ));
    }

    #[test]
    fn the_record_digest_binds_every_final_vector_and_share_item() {
        let ys = [[1_u8; 512], [2; 512]];
        let shares = vec![vec![], vec![7; 512]];
        let base = flock_record_digest(&ys, &shares);
        assert_eq!(base, flock_record_digest(&ys, &shares));
        let mut other = ys;
        other[1][511] ^= 1;
        assert_ne!(base, flock_record_digest(&other, &shares));
        let mut moved = shares.clone();
        moved.swap(0, 1);
        assert_ne!(base, flock_record_digest(&ys, &moved), "share items are framed per segment");
        assert_ne!(base, flock_record_digest(&ys, &[vec![], vec![7; 511]]));
        assert_ne!(
            flock_record_digest(&ys[..1], &[vec![]]),
            flock_record_digest(&ys[..1], &[vec![0]]),
            "a non-empty share item of a span changes the digest"
        );
    }

    #[test]
    fn extension_encoding_round_trips_and_rejects_non_canonical_limbs() {
        let values = vec![
            OpeningField::from_u64(5),
            OpeningField::from_base_slice(&[BaseField::from_u64(3), BaseField::from_u64(9)]),
        ];
        let bytes = encode_extensions(&values);
        assert_eq!(bytes.len(), 32);
        assert_eq!(decode_extensions(&bytes).unwrap(), values);
        assert!(decode_extensions(&bytes[..20]).is_none());
        let mut bad = bytes.clone();
        bad[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(decode_extensions(&bad).is_none());
        let mut t = Transcript::new(&manifest(), &[1, 2]).unwrap();
        to_seed(&mut t, &[0]);
        for role in [GroupRole::Level1, GroupRole::Level2, GroupRole::Level3, GroupRole::Mult] {
            t.absorb(commitment_site(role), 1, &[0]).unwrap();
        }
        t.challenge_bytes(FLOCK_SEEDS_SITE, 1).unwrap();
        t.absorb(FINAL_VECTORS_SITE, 1, &[0; 512]).unwrap();
        t.absorb_items(FLOCK_SHARES_SITE, &[vec![]]).unwrap();
        // The lookup's challenges come first: tau out of order is refused.
        assert!(t.challenge_extensions(TAU_SITE, 17).is_err());
        let x = LookupTranscript::challenges(&mut t, LookupSite::X, 2).unwrap();
        assert_eq!(x.len(), 2);
        assert!(x.iter().all(|c| *c != OpeningField::from_u64(0)));
    }
}
