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
//! Grinding: the spec adopts Aerie's scheduled grinding; this stage registers every
//! challenge at difficulty zero (recorded in `docs/deviations.md`).

use akita_transcript::TranscriptSponge;
use jolt_field::{CanonicalEncoding, ExtField};
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::{CShake256, CShake256Core};
use spongefish::{DomainSeparator, DuplexSpongeInterface, Encoding, VerifierState, WithoutInstance};

use crate::manifest::{CommitmentManifest, GroupRole};
use crate::{BaseField, OpeningField};

/// Spongefish protocol tag, zero-padded to 64 bytes.
pub const PIOP_PROTOCOL_TAG: [u8; 64] = protocol_tag();
/// The session label.
pub const PROTOCOL_DOMAIN: &[u8] = b"agg-mldsa/v1";
/// cSHAKE256 customization of the instance digest.
pub const INSTANCE_CUSTOMIZATION: &[u8] = b"agg-mldsa/v1/instance";
/// Stage message degrees (spec Section 7.1).
pub const STAGE_DEGREES: [usize; 3] = [4, 5, 3];
/// Batching coefficients per stage (spec Section 7).
pub const STAGE_GAMMAS: [usize; 3] = [66, 101, 118];
/// Terminal claims per stage.
pub const STAGE_TERMINALS: [usize; 3] = [98, 61, 67];
/// Pre-stage claims (spec Section 7.2).
pub const PRESTAGE_CLAIMS: usize = 26;
/// Level-1 carry digit claims before stage 2 (spec Section 7.4).
pub const KAPPA1_CLAIMS: usize = 18;
/// Bytes of one packed final vector `Y_sigma`.
pub const FINAL_VECTOR_BYTES: usize = 512;

pub const INSTANCE_SITE: &str = "agg-mldsa/v1/instance";
pub const FLOCK_COMMIT_SITE: &str = "agg-mldsa/v1/commit/flock";
pub const BRIDGE_SEED_SITE: &str = "agg-mldsa/v1/bridge/seed";
pub const FLOCK_SEEDS_SITE: &str = "agg-mldsa/v1/flock/seeds";
pub const FINAL_VECTORS_SITE: &str = "agg-mldsa/v1/flock/final-vectors";
pub const FLOCK_PROOFS_SITE: &str = "agg-mldsa/v1/flock/proofs";
pub const TAU_SITE: &str = "agg-mldsa/v1/tau";
pub const PRESTAGE_SITE: &str = "agg-mldsa/v1/prestage/claims";
pub const KAPPA1_SITE: &str = "agg-mldsa/v1/stage2/kappa1";
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
        GroupRole::Main => "agg-mldsa/v1/commit/main",
        GroupRole::Level1 => "agg-mldsa/v1/commit/level1",
        GroupRole::Level2 => "agg-mldsa/v1/commit/level2",
        GroupRole::Level3 => "agg-mldsa/v1/commit/level3",
    }
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
}

impl std::fmt::Display for TranscriptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EventOrder { index, site } => {
                write!(f, "transcript event {index}: unexpected site {site}")
            }
            Self::EventShape { index } => write!(f, "transcript event {index}: wrong shape"),
            Self::Incomplete { index } => write!(f, "transcript ended before event {index}"),
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
}

/// The ordered event list of one batch (spec Section 10).
#[derive(Clone, Debug)]
pub struct Registry {
    events: Vec<RegistryEvent>,
}

impl Registry {
    /// The registry of `manifest`.
    #[must_use]
    pub fn new(manifest: &CommitmentManifest) -> Self {
        let rounds = manifest.d_main();
        let segments = manifest.segments();
        let mut events = Vec::with_capacity(24 + 6 * rounds);
        let mut absorb = |site: &str, kind: u8, atoms: usize, size: PayloadSize| {
            events.push(RegistryEvent {
                site: site.to_owned(),
                kind,
                atoms,
                size,
            });
        };
        absorb(INSTANCE_SITE, PUBLIC_KIND, 1, PayloadSize::Variable);
        absorb(
            commitment_site(GroupRole::Main),
            PROOF_KIND,
            1,
            PayloadSize::Variable,
        );
        absorb(FLOCK_COMMIT_SITE, PROOF_KIND, segments, PayloadSize::Variable);
        absorb(BRIDGE_SEED_SITE, CHALLENGE_KIND, 1, PayloadSize::Exact(32));
        for role in [GroupRole::Level1, GroupRole::Level2, GroupRole::Level3] {
            absorb(commitment_site(role), PROOF_KIND, 1, PayloadSize::Variable);
        }
        absorb(
            FLOCK_SEEDS_SITE,
            CHALLENGE_KIND,
            segments,
            PayloadSize::Exact(32 * segments),
        );
        absorb(
            FINAL_VECTORS_SITE,
            PROOF_KIND,
            segments,
            PayloadSize::Exact(FINAL_VECTOR_BYTES * segments),
        );
        absorb(FLOCK_PROOFS_SITE, PROOF_KIND, segments, PayloadSize::Variable);
        absorb(TAU_SITE, CHALLENGE_KIND, rounds, PayloadSize::Exact(16 * rounds));
        absorb(
            PRESTAGE_SITE,
            PROOF_KIND,
            PRESTAGE_CLAIMS,
            PayloadSize::Exact(16 * PRESTAGE_CLAIMS),
        );
        for stage in 1..=3 {
            if stage == 2 {
                absorb(
                    KAPPA1_SITE,
                    PROOF_KIND,
                    KAPPA1_CLAIMS,
                    PayloadSize::Exact(16 * KAPPA1_CLAIMS),
                );
            }
            let gammas = STAGE_GAMMAS[stage - 1];
            absorb(
                &gamma_site(stage),
                CHALLENGE_KIND,
                gammas,
                PayloadSize::Exact(16 * gammas),
            );
            let width = STAGE_DEGREES[stage - 1];
            for round in 0..rounds {
                let (message, challenge) = round_sites(stage, round);
                absorb(&message, PROOF_KIND, width, PayloadSize::Exact(16 * width));
                absorb(&challenge, CHALLENGE_KIND, 1, PayloadSize::Exact(16));
            }
            let terminals = STAGE_TERMINALS[stage - 1];
            absorb(
                &terminal_site(stage),
                PROOF_KIND,
                terminals,
                PayloadSize::Exact(16 * terminals),
            );
        }
        absorb(PCS_SESSION_SITE, EXPORT_KIND, 1, PayloadSize::Exact(32));
        Self { events }
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
}

impl Transcript {
    /// Initialize the stream from the canonical instance record and absorb event 0.
    pub fn new(manifest: &CommitmentManifest, record: &[u8]) -> Result<Self, TranscriptError> {
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
        };
        this.absorb_kind(PUBLIC_KIND, INSTANCE_SITE, 1, record)?;
        Ok(this)
    }

    fn take(
        &mut self,
        site: &str,
        kind: u8,
        atoms: usize,
        payload_bytes: usize,
    ) -> Result<(), TranscriptError> {
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
        Ok(())
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
        self.take(site, CHALLENGE_KIND, count, 16 * count)?;
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

    /// Confirm every registered event ran.
    pub fn ensure_complete(&self) -> Result<(), TranscriptError> {
        self.require_next(self.registry.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Ring;

    fn manifest() -> CommitmentManifest {
        CommitmentManifest::for_batch(100).unwrap()
    }

    #[test]
    fn registry_has_the_spec_event_order() {
        let registry = Registry::new(&manifest());
        let rounds = 17;
        // 12 leading events, the kappa1 absorb, three stages of gamma + rounds + terminal, the PCS export.
        assert_eq!(registry.len(), 12 + 1 + 3 * (2 + 2 * rounds) + 1);
        let sites: Vec<&str> = registry.events.iter().map(|e| e.site.as_str()).collect();
        assert_eq!(
            &sites[..12],
            &[
                INSTANCE_SITE,
                "agg-mldsa/v1/commit/main",
                FLOCK_COMMIT_SITE,
                BRIDGE_SEED_SITE,
                "agg-mldsa/v1/commit/level1",
                "agg-mldsa/v1/commit/level2",
                "agg-mldsa/v1/commit/level3",
                FLOCK_SEEDS_SITE,
                FINAL_VECTORS_SITE,
                FLOCK_PROOFS_SITE,
                TAU_SITE,
                PRESTAGE_SITE,
            ]
        );
        assert_eq!(sites[12], "agg-mldsa/v1/stage1/gamma");
        assert_eq!(sites[13], "agg-mldsa/v1/stage1/round/0/message");
        assert_eq!(sites[12 + 1 + 2 * rounds], "agg-mldsa/v1/stage1/terminal");
        assert_eq!(sites[12 + 2 + 2 * rounds], KAPPA1_SITE);
        assert_eq!(sites[registry.len() - 1], PCS_SESSION_SITE);
        let kinds: Vec<u8> = registry.events.iter().map(|e| e.kind).collect();
        assert_eq!(&kinds[..4], &[0, 1, 1, 2]);
        assert_eq!(registry.events[3].size, PayloadSize::Exact(32));
        assert_eq!(registry.events[7].atoms, 1, "one segment at N = 100");
        assert_eq!(registry.events[10].atoms, rounds);
        let full = Registry::new(&CommitmentManifest::for_batch(1 << 15).unwrap());
        assert_eq!(full.events[7].size, PayloadSize::Exact(128));
        assert_eq!(full.events[8].size, PayloadSize::Exact(4 * 512));
    }

    #[test]
    fn events_must_follow_the_registry() {
        let manifest = manifest();
        let record = manifest.binding_bytes();
        let mut t = Transcript::new(&manifest, &record).unwrap();
        // Skipping the main commitment is an order fault; a wrong seed size a shape fault.
        assert!(matches!(
            t.absorb_items(FLOCK_COMMIT_SITE, &[]),
            Err(TranscriptError::EventOrder { index: 1, .. })
        ));
        t.absorb(commitment_site(GroupRole::Main), 1, &[1, 2, 3])
            .unwrap();
        t.absorb_items(FLOCK_COMMIT_SITE, &[vec![9; 4]]).unwrap();
        assert!(matches!(
            t.challenge_bytes(BRIDGE_SEED_SITE, 2),
            Err(TranscriptError::EventShape { index: 3 })
        ));
        let seed = t.challenge_bytes(BRIDGE_SEED_SITE, 1).unwrap();
        assert_eq!(seed.len(), 32);
        assert!(t.require_next(4).is_ok());
        assert!(t.ensure_complete().is_err());
    }

    #[test]
    fn replay_is_deterministic_and_sensitive_to_every_byte() {
        let manifest = manifest();
        let record = manifest.binding_bytes();
        let run = |record: &[u8], commitment: &[u8]| {
            let mut t = Transcript::new(&manifest, record).unwrap();
            t.absorb(commitment_site(GroupRole::Main), 1, commitment)
                .unwrap();
            t.absorb_items(FLOCK_COMMIT_SITE, &[vec![]]).unwrap();
            t.challenge_bytes(BRIDGE_SEED_SITE, 1).unwrap()
        };
        let base = run(&record, &[7; 40]);
        assert_eq!(base, run(&record, &[7; 40]));
        assert_ne!(base, run(&record, &[8; 40]));
        let mut other = record.clone();
        other[0] ^= 1;
        assert_ne!(base, run(&other, &[7; 40]));
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
        t.absorb(commitment_site(GroupRole::Main), 1, &[0]).unwrap();
        t.absorb_items(FLOCK_COMMIT_SITE, &[vec![]]).unwrap();
        t.challenge_bytes(BRIDGE_SEED_SITE, 1).unwrap();
        for role in [GroupRole::Level1, GroupRole::Level2, GroupRole::Level3] {
            t.absorb(commitment_site(role), 1, &[0]).unwrap();
        }
        t.challenge_bytes(FLOCK_SEEDS_SITE, 1).unwrap();
        t.absorb(FINAL_VECTORS_SITE, 1, &[0; 512]).unwrap();
        t.absorb_items(FLOCK_PROOFS_SITE, &[vec![]]).unwrap();
        let tau = t.challenge_extensions(TAU_SITE, 17).unwrap();
        assert_eq!(tau.len(), 17);
        assert!(tau.iter().all(|c| *c != OpeningField::from_u64(0)));
    }
}
