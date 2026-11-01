//! Proof wire format (wire version 4): header (version, flags, `N`), the `|c~| N` bytes of
//! `c~_i` (32, 48 or 64 bytes each), then the payloads of the commitments, the final vectors and the
//! Keccak side's share items, the lookup, the sum-check stages and the join
//! (the Keccak side's proof items, after stage 3) in transcript order, the packed PIOP nonce
//! stream (`docs/soundness.md`, its exact length a function of `N`), then the Akita opening.
//! Variable-length items carry a `LE32` length; field elements use the canonical atom
//! encoding of the transcript; the round and session counts follow from `N`, the stage-2 and
//! stage-3 claim counts from the relation (`docs/h-exact.md`). The flags byte holds the
//! relation id in its low nibble and the parameter set in its high nibble (0 = ML-DSA-44,
//! 1 = ML-DSA-65, 2 = ML-DSA-87), so an ML-DSA-44 header is unchanged. The parser reads the
//! relation and the set from the header; a verifier MUST compare both with the ones it
//! expects before using the proof.

use agg_witness::dispatch_set;
use agg_witness::params::{ParamSet, Set};

use crate::claims::{
    Kappa1Claims, MultiplicityClaims, PreStageClaims, Stage1Claims, Stage2Claims, Stage3Claims,
};
use crate::lookup::{LookupProof, LookupShape, SessionProof, DEGREE as LOOKUP_DEGREE};
use crate::manifest::{CommitmentManifest, Relation, WIRE_VERSION};
use crate::stage1::Stage1Proof;
use crate::stage2::Stage2Proof;
use crate::stage3::Stage3Proof;
use crate::transcript::{
    decode_extensions, encode_extensions, stage_degrees, work_bytes, FINAL_VECTOR_BYTES,
};
use crate::OpeningField as E;

/// A malformed proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WireError {
    /// Unknown wire version.
    Version(u16),
    /// Unknown feature flags (relation and parameter-set nibbles).
    Flags(u8),
    /// `N` outside `1..=2^15`.
    Records(u64),
    /// The buffer ended early or a length prefix is out of range.
    Truncated,
    /// Bytes after the opening.
    Trailing,
    /// A non-canonical field element.
    Encoding,
    /// A claim vector or round count of the wrong shape.
    Shape(&'static str),
}

impl std::fmt::Display for WireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for WireError {}

/// The aggregate proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AggregateProof {
    /// The parameter set the proof claims (the high nibble of the header's flags byte).
    pub set: Set,
    /// The relation the proof claims (the low nibble of the header's flags byte).
    pub relation: Relation,
    /// Live records `N`.
    pub records: usize,
    /// `c~_i` per live record ([`c_tilde_bytes`] each).
    pub c_tilde: Vec<Vec<u8>>,
    /// The round-1 Akita commitment of `G_main16`.
    pub main16_commitment: Vec<u8>,
    /// The round-1 Akita commitment of `G_main4`.
    pub main_commitment: Vec<u8>,
    /// `S` Flock commitments.
    pub flock_commitments: Vec<Vec<u8>>,
    /// The round-2 Akita commitments (`G_1, G_2, G_3, G_k, G_m`).
    pub bridge_commitments: [Vec<u8>; 5],
    /// `Y_sigma`, 4,096 bits packed least significant first.
    pub final_vectors: Vec<[u8; FINAL_VECTOR_BYTES]>,
    /// `S` share items of the Keccak side (mode A: the shares of a segment's parts
    /// but the last; empty for a span, a one-part segment and mode B).
    pub flock_shares: Vec<Vec<u8>>,
    /// The join (after stage 3): `S` Flock proof byte strings.
    pub flock_proofs: Vec<Vec<u8>>,
    /// The lookup (layer 1 and the sessions).
    pub lookup: LookupProof,
    /// The multiplicity-digit claims at the table point.
    pub multiplicities: MultiplicityClaims,
    /// The pre-stage claims.
    pub prestage: PreStageClaims,
    /// Stage 1.
    pub stage1: Stage1Proof,
    /// Stage 2.
    pub stage2: Stage2Proof,
    /// Stage 3.
    pub stage3: Stage3Proof,
    /// The packed PIOP work nonces in plan order (`docs/soundness.md`).
    pub work: Vec<u8>,
    /// The Akita opening.
    pub opening: Vec<u8>,
}

/// The high nibble of the flags byte for `set`.
#[must_use]
pub const fn set_nibble(set: Set) -> u8 {
    match set {
        Set::S44 => 0,
        Set::S65 => 1,
        Set::S87 => 2,
    }
}

/// The set of a flags nibble ([`set_nibble`]).
#[must_use]
pub const fn set_of_nibble(nibble: u8) -> Option<Set> {
    match nibble {
        0 => Some(Set::S44),
        1 => Some(Set::S65),
        2 => Some(Set::S87),
        _ => None,
    }
}

/// The header's flags byte: relation id in the low nibble, [`set_nibble`] in the high one.
#[must_use]
pub const fn flags(set: Set, relation: Relation) -> u8 {
    set_nibble(set) << 4 | relation.id()
}

/// Bytes of one `c~` of `set` (`lambda / 4`: 32, 48 or 64).
#[must_use]
pub const fn c_tilde_bytes(set: Set) -> usize {
    dispatch_set!(set, |P| P::C_TILDE_BYTES)
}

struct Writer(Vec<u8>);

impl Writer {
    fn bytes(&mut self, bytes: &[u8]) {
        self.0.extend_from_slice(bytes);
    }

    fn framed(&mut self, bytes: &[u8]) {
        self.0
            .extend_from_slice(&u32::try_from(bytes.len()).expect("item below 4 GiB").to_le_bytes());
        self.0.extend_from_slice(bytes);
    }

    fn elements(&mut self, values: &[E]) {
        self.0.extend_from_slice(&encode_extensions(values));
    }

    fn rounds(&mut self, rounds: &[Vec<E>]) {
        for round in rounds {
            self.elements(round);
        }
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], WireError> {
        let end = self.at.checked_add(len).ok_or(WireError::Truncated)?;
        let out = self.bytes.get(self.at..end).ok_or(WireError::Truncated)?;
        self.at = end;
        Ok(out)
    }

    fn array<const M: usize>(&mut self) -> Result<[u8; M], WireError> {
        Ok(self.take(M)?.try_into().expect("length"))
    }

    fn framed(&mut self) -> Result<Vec<u8>, WireError> {
        let len = u32::from_le_bytes(self.array()?) as usize;
        Ok(self.take(len)?.to_vec())
    }

    fn elements(&mut self, count: usize) -> Result<Vec<E>, WireError> {
        decode_extensions(self.take(16 * count)?).ok_or(WireError::Encoding)
    }

    fn rounds(&mut self, rounds: usize, degree: usize) -> Result<Vec<Vec<E>>, WireError> {
        (0..rounds).map(|_| self.elements(degree)).collect()
    }
}

impl AggregateProof {
    /// Serialize.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = Writer(Vec::new());
        w.bytes(&WIRE_VERSION.to_le_bytes());
        w.bytes(&[flags(self.set, self.relation)]);
        w.bytes(&(self.records as u64).to_le_bytes());
        let width = c_tilde_bytes(self.set);
        for c in &self.c_tilde {
            assert_eq!(c.len(), width, "c~ of the proof's parameter set");
            w.bytes(c);
        }
        w.framed(&self.main16_commitment);
        w.framed(&self.main_commitment);
        w.bytes(&(self.flock_commitments.len() as u32).to_le_bytes());
        for item in &self.flock_commitments {
            w.framed(item);
        }
        for item in &self.bridge_commitments {
            w.framed(item);
        }
        for y in &self.final_vectors {
            w.bytes(y);
        }
        for item in &self.flock_shares {
            w.framed(item);
        }
        for layer in &self.lookup.layer1 {
            w.elements(layer);
        }
        for session in &self.lookup.sessions {
            w.rounds(&session.rounds);
            w.elements(&session.terminals.concat());
            w.elements(&session.leaves.concat());
        }
        w.elements(&self.multiplicities.0);
        w.elements(&self.prestage.to_vec());
        w.rounds(&self.stage1.rounds);
        w.elements(&self.stage1.claims.to_vec());
        w.elements(&self.stage2.kappa1.to_vec());
        w.rounds(&self.stage2.rounds);
        w.elements(&self.stage2.claims.to_vec());
        w.elements(&[self.stage2.projection]);
        w.rounds(&self.stage3.rounds);
        w.elements(&self.stage3.claims.0);
        for item in &self.flock_proofs {
            w.framed(item);
        }
        w.bytes(&self.work);
        w.framed(&self.opening);
        w.0
    }

    /// Parse; the shapes follow from the header's `N`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, WireError> {
        let mut r = Reader { bytes, at: 0 };
        let version = u16::from_le_bytes(r.array()?);
        if version != WIRE_VERSION {
            return Err(WireError::Version(version));
        }
        let flags = r.array::<1>()?[0];
        let relation = Relation::from_id(flags & 0x0F).ok_or(WireError::Flags(flags))?;
        let set = set_of_nibble(flags >> 4).ok_or(WireError::Flags(flags))?;
        if set != Set::S44 && relation != Relation::Hu {
            // ML-DSA-65 and ML-DSA-87 have relation H-u only.
            return Err(WireError::Flags(flags));
        }
        let records = u64::from_le_bytes(r.array()?);
        let manifest = usize::try_from(records)
            .ok()
            .and_then(|n| CommitmentManifest::for_set(set, n, relation).ok())
            .ok_or(WireError::Records(records))?;
        let records = manifest.records();
        let segments = manifest.segments();
        let rounds = manifest.d_main();
        let width = c_tilde_bytes(set);
        let c_tilde = (0..records)
            .map(|_| r.take(width).map(<[u8]>::to_vec))
            .collect::<Result<Vec<_>, _>>()?;
        let main16_commitment = r.framed()?;
        let main_commitment = r.framed()?;
        let count = u32::from_le_bytes(r.array()?) as usize;
        if count != segments {
            return Err(WireError::Shape("flock commitment count"));
        }
        let flock_commitments = (0..segments)
            .map(|_| r.framed())
            .collect::<Result<Vec<_>, _>>()?;
        let bridge_commitments = [r.framed()?, r.framed()?, r.framed()?, r.framed()?, r.framed()?];
        let final_vectors = (0..segments)
            .map(|_| r.array::<FINAL_VECTOR_BYTES>())
            .collect::<Result<Vec<_>, _>>()?;
        let flock_shares = (0..segments)
            .map(|_| r.framed())
            .collect::<Result<Vec<_>, _>>()?;
        let lookup = read_lookup(&mut r, &manifest.lookup_shape())?;
        let multiplicities = MultiplicityClaims::from_slice(&r.elements(MultiplicityClaims::LEN)?)
            .ok_or(WireError::Shape("multiplicity claims"))?;
        let degrees = stage_degrees(&manifest);
        let prestage = PreStageClaims::from_slice_for(&manifest, &r.elements(PreStageClaims::len(&manifest))?)
            .ok_or(WireError::Shape("pre-stage claims"))?;
        let stage1 = Stage1Proof {
            rounds: r.rounds(rounds, degrees[0])?,
            claims: Stage1Claims::from_slice_for(&manifest, &r.elements(Stage1Claims::len(&manifest))?)
                .ok_or(WireError::Shape("stage 1 claims"))?,
        };
        let stage2 = Stage2Proof {
            kappa1: Kappa1Claims::from_slice_for(&manifest, &r.elements(Kappa1Claims::len(&manifest))?)
                .ok_or(WireError::Shape("kappa1 claims"))?,
            rounds: r.rounds(rounds, degrees[1])?,
            claims: Stage2Claims::from_slice(&manifest, &r.elements(Stage2Claims::len(&manifest))?)
                .ok_or(WireError::Shape("stage 2 claims"))?,
            projection: r.elements(1)?[0],
        };
        let stage3 = Stage3Proof {
            rounds: r.rounds(rounds, degrees[2])?,
            claims: Stage3Claims::from_slice(&manifest, &r.elements(Stage3Claims::len(&manifest))?)
                .ok_or(WireError::Shape("stage 3 claims"))?,
        };
        let flock_proofs = (0..segments)
            .map(|_| r.framed())
            .collect::<Result<Vec<_>, _>>()?;
        let work = r.take(work_bytes(&manifest))?.to_vec();
        let opening = r.framed()?;
        if r.at != bytes.len() {
            return Err(WireError::Trailing);
        }
        Ok(Self {
            set,
            relation,
            records,
            c_tilde,
            main16_commitment,
            main_commitment,
            flock_commitments,
            bridge_commitments,
            final_vectors,
            flock_shares,
            flock_proofs,
            lookup,
            multiplicities,
            prestage,
            stage1,
            stage2,
            stage3,
            work,
            opening,
        })
    }

    /// The lookup part of the proof in bytes (layer 1, the sessions and the
    /// multiplicity-digit claims).
    #[must_use]
    pub fn lookup_bytes(&self) -> usize {
        self.lookup.byte_len() + 16 * MultiplicityClaims::LEN
    }

    /// The PIOP part of the proof in bytes (the sum-check stages, without the lookup).
    #[must_use]
    pub fn piop_bytes(&self) -> usize {
        let width = |rounds: &[Vec<E>]| rounds.iter().map(Vec::len).sum::<usize>();
        16 * (self.prestage.to_vec().len()
            + self.stage1.claims.to_vec().len()
            + self.stage2.kappa1.0.len()
            + self.stage2.claims.to_vec().len()
            + 1
            + self.stage3.claims.0.len()
            + width(&self.stage1.rounds)
            + width(&self.stage2.rounds)
            + width(&self.stage3.rounds))
    }
}

/// Bytes of each part of a serialized proof (framing included).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProofSize {
    /// Version, flags, `N`.
    pub header: usize,
    /// The `32 N` bytes of `c~_i`.
    pub c_tilde: usize,
    /// The seven Akita commitments.
    pub akita_commitments: usize,
    /// The Keccak-side items (Flock commitments; mode B: the public bits).
    pub keccak_commitments: usize,
    /// The final vectors `Y_sigma`.
    pub final_vectors: usize,
    /// The final-vector step and the join: the Keccak side's share and proof items.
    pub keccak_proofs: usize,
    /// The lookup and the multiplicity claims.
    pub lookup: usize,
    /// Claims and round messages.
    pub piop: usize,
    /// The packed PIOP work nonces.
    pub work: usize,
    /// The Akita opening.
    pub opening: usize,
}

impl ProofSize {
    /// The serialized length.
    #[must_use]
    pub const fn total(&self) -> usize {
        self.header
            + self.c_tilde
            + self.akita_commitments
            + self.keccak_commitments
            + self.final_vectors
            + self.keccak_proofs
            + self.lookup
            + self.piop
            + self.work
            + self.opening
    }
}

impl AggregateProof {
    /// The size of each part of [`AggregateProof::to_bytes`].
    #[must_use]
    pub fn size(&self) -> ProofSize {
        let framed = |items: &[Vec<u8>]| items.iter().map(|item| 4 + item.len()).sum::<usize>();
        ProofSize {
            header: 2 + 1 + 8,
            c_tilde: c_tilde_bytes(self.set) * self.c_tilde.len(),
            akita_commitments: 8
                + self.main16_commitment.len()
                + self.main_commitment.len()
                + framed(&self.bridge_commitments),
            keccak_commitments: 4 + framed(&self.flock_commitments),
            final_vectors: FINAL_VECTOR_BYTES * self.final_vectors.len(),
            keccak_proofs: framed(&self.flock_shares) + framed(&self.flock_proofs),
            lookup: self.lookup_bytes(),
            piop: self.piop_bytes(),
            work: self.work.len(),
            opening: 4 + self.opening.len(),
        }
    }
}

fn chunks<const M: usize>(values: &[E]) -> Vec<[E; M]> {
    values.as_chunks::<M>().0.to_vec()
}

/// The lookup proof of `shape`: `4` layer-1 values per tree, then per session the rounds,
/// the terminals and the leaves.
fn read_lookup(r: &mut Reader<'_>, shape: &LookupShape) -> Result<LookupProof, WireError> {
    let layer1 = chunks::<4>(&r.elements(4 * shape.tree_count())?);
    let mut sessions = Vec::new();
    for s in 2..=shape.max_depth() {
        let (_, continuing, ending) = shape.session_counts(s);
        sessions.push(SessionProof {
            rounds: r.rounds(s - 1, LOOKUP_DEGREE)?,
            terminals: chunks::<4>(&r.elements(4 * continuing)?),
            leaves: chunks::<2>(&r.elements(2 * ending)?),
        });
    }
    Ok(LookupProof { layer1, sessions })
}

/// Pack 4,096 bits (one byte each) into 512 bytes, least significant bit first
/// (`agg_bridge::final_vector::pack`).
#[must_use]
pub fn pack_final_vector(bits: &[u8]) -> [u8; FINAL_VECTOR_BYTES] {
    assert_eq!(bits.len(), 8 * FINAL_VECTOR_BYTES, "4,096 bits");
    agg_bridge::final_vector::pack(bits)
        .try_into()
        .expect("512 bytes")
}

/// Unpack 512 bytes into 4,096 bits (one byte each; `agg_bridge::final_vector::unpack`).
#[must_use]
pub fn unpack_final_vector(packed: &[u8; FINAL_VECTOR_BYTES]) -> Vec<u8> {
    agg_bridge::final_vector::unpack(packed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::point;
    use agg_witness::gen::SplitMix64;

    fn sample_lookup(shape: &LookupShape, rng: &mut SplitMix64) -> LookupProof {
        let four = |rng: &mut SplitMix64, n: usize| chunks::<4>(&point(rng, 4 * n));
        LookupProof {
            layer1: four(rng, shape.tree_count()),
            sessions: (2..=shape.max_depth())
                .map(|s| {
                    let (_, continuing, ending) = shape.session_counts(s);
                    SessionProof {
                        rounds: (0..s - 1).map(|_| point(rng, LOOKUP_DEGREE)).collect(),
                        terminals: four(rng, continuing),
                        leaves: chunks::<2>(&point(rng, 2 * ending)),
                    }
                })
                .collect(),
        }
    }

    fn sample_proof(records: usize, rng: &mut SplitMix64) -> AggregateProof {
        sample_proof_of(Relation::Hu, records, rng)
    }

    fn sample_proof_of(relation: Relation, records: usize, rng: &mut SplitMix64) -> AggregateProof {
        sample_proof_in(Set::S44, relation, records, rng)
    }

    fn sample_proof_in(set: Set, relation: Relation, records: usize, rng: &mut SplitMix64) -> AggregateProof {
        let manifest = CommitmentManifest::for_set(set, records, relation).unwrap();
        let degrees = stage_degrees(&manifest);
        let rounds = manifest.d_main();
        let segments = manifest.segments();
        let rounds_of = |degree: usize, rng: &mut SplitMix64| {
            (0..rounds).map(|_| point(rng, degree)).collect::<Vec<_>>()
        };
        AggregateProof {
            set,
            relation,
            records,
            c_tilde: (0..records).map(|i| vec![i as u8; c_tilde_bytes(set)]).collect(),
            main16_commitment: vec![1, 2],
            main_commitment: vec![1, 2, 3],
            flock_commitments: vec![vec![4; 10]; segments],
            bridge_commitments: [vec![5], vec![], vec![6, 7], vec![10, 11, 12], vec![8]],
            final_vectors: vec![[8; 512]; segments],
            flock_shares: vec![vec![]; segments],
            flock_proofs: vec![vec![3; 5]; segments],
            lookup: sample_lookup(&manifest.lookup_shape(), rng),
            multiplicities: MultiplicityClaims::from_slice(&point(rng, MultiplicityClaims::LEN)).unwrap(),
            prestage: PreStageClaims::from_slice_for(&manifest, &point(rng, PreStageClaims::len(&manifest)))
                .unwrap(),
            stage1: Stage1Proof {
                rounds: rounds_of(degrees[0], rng),
                claims: Stage1Claims::from_slice_for(&manifest, &point(rng, Stage1Claims::len(&manifest))).unwrap(),
            },
            stage2: Stage2Proof {
                kappa1: Kappa1Claims::from_slice_for(&manifest, &point(rng, Kappa1Claims::len(&manifest))).unwrap(),
                rounds: rounds_of(degrees[1], rng),
                claims: Stage2Claims::from_slice(&manifest, &point(rng, Stage2Claims::len(&manifest)))
                    .unwrap(),
                projection: point(rng, 1)[0],
            },
            stage3: Stage3Proof {
                rounds: rounds_of(degrees[2], rng),
                claims: Stage3Claims(point(rng, Stage3Claims::len(&manifest))),
            },
            work: (0..work_bytes(&manifest)).map(|i| i as u8).collect(),
            opening: vec![9; 100],
        }
    }

    #[test]
    fn proofs_round_trip_and_malformed_buffers_are_rejected() {
        let mut rng = SplitMix64::new(77);
        for records in [1, 100, 1000] {
            let proof = sample_proof(records, &mut rng);
            let bytes = proof.to_bytes();
            assert_eq!(AggregateProof::from_bytes(&bytes).unwrap(), proof);
            let size = proof.size();
            assert_eq!(size.total(), bytes.len());
            assert_eq!((size.c_tilde, size.piop), (32 * records, proof.piop_bytes()));
            // The join: the framed proof items follow the stage-3 claims and
            // precede the nonce stream and the opening.
            let tail = size.work + 4 + proof.opening.len();
            let segments = proof.flock_proofs.len();
            let join = &bytes[bytes.len() - tail - 9 * segments..bytes.len() - tail];
            assert_eq!(join, [5, 0, 0, 0, 3, 3, 3, 3, 3].repeat(segments));
            // Relation H-u: stage degrees 4, 5, 3 (12 elements per round); the shared
            // projection claim after the stage-2 terminal.
            assert_eq!(proof.piop_bytes(), 16 * (13 + 18 + 6 + 10 + 1 + 35 + 12 * proof.stage1.rounds.len()));
            let shape = CommitmentManifest::for_batch(records).unwrap().lookup_shape();
            assert_eq!(size.lookup, 16 * (shape.proof_elements() + 7));
            assert_eq!(AggregateProof::from_bytes(&bytes[..bytes.len() - 1]), Err(WireError::Truncated));
            let mut trailing = bytes.clone();
            trailing.push(0);
            assert_eq!(AggregateProof::from_bytes(&trailing), Err(WireError::Trailing));
            let mut version = bytes.clone();
            version[0] = 9;
            assert_eq!(AggregateProof::from_bytes(&version), Err(WireError::Version(9)));
            // Wire versions 0 to 3 are refused.
            for old in [0, 1, 2, 3] {
                version[0] = old;
                assert_eq!(AggregateProof::from_bytes(&version), Err(WireError::Version(u16::from(old))));
            }
            // The nonce stream has the plan's exact length.
            assert_eq!(size.work, work_bytes(&CommitmentManifest::for_batch(records).unwrap()));
            assert!(size.work >= 227);
            // Relation byte: 0 (H-u), 1 (H-exact), 2 (H-FIPS); 3 and above are unknown. The
            // parser reads the relation from the header (H-u and H-exact proofs have equally
            // many stage claims, 9 + 34 = 8 + 35, so a rewritten flag may still parse; the
            // verifier compares it with the relation it expects).
            assert_eq!(bytes[2], 0);
            for (flag, relation) in [(1_u8, Relation::HExact), (2, Relation::HFips)] {
                let mut rewritten = bytes.clone();
                rewritten[2] = flag;
                if let Ok(parsed) = AggregateProof::from_bytes(&rewritten) {
                    assert_eq!(parsed.relation, relation);
                    assert_ne!(parsed, proof);
                }
            }
            for flag in [3_u8, 0xFF] {
                let mut flags = bytes.clone();
                flags[2] = flag;
                assert_eq!(AggregateProof::from_bytes(&flags), Err(WireError::Flags(flag)));
            }
            let mut n = bytes.clone();
            n[3..11].copy_from_slice(&0_u64.to_le_bytes());
            assert_eq!(AggregateProof::from_bytes(&n), Err(WireError::Records(0)));
            // A non-canonical limb in the first lookup element.
            let offset =
                11 + 32 * records + (4 + 2) + (4 + 3) + 4 + 4 + 10 + (4 + 1) + 4 + (4 + 2) + (4 + 3) + (4 + 1) + 512 + 4;
            let mut limb = bytes.clone();
            limb[offset..offset + 8].copy_from_slice(&u64::MAX.to_le_bytes());
            assert_eq!(AggregateProof::from_bytes(&limb), Err(WireError::Encoding));
        }
        let too_many = {
            let mut bytes = sample_proof(1, &mut rng).to_bytes();
            bytes[3..11].copy_from_slice(&((1_u64 << 15) + 1).to_le_bytes());
            bytes
        };
        assert_eq!(
            AggregateProof::from_bytes(&too_many),
            Err(WireError::Records((1 << 15) + 1))
        );
    }

    #[test]
    fn h_exact_proofs_round_trip_with_their_claim_counts() {
        let mut rng = SplitMix64::new(78);
        for records in [1, 100, 1000] {
            let proof = sample_proof_of(Relation::HExact, records, &mut rng);
            let bytes = proof.to_bytes();
            assert_eq!(bytes[2], 1);
            let parsed = AggregateProof::from_bytes(&bytes).unwrap();
            assert_eq!(parsed, proof);
            assert_eq!(parsed.relation, Relation::HExact);
            assert_eq!(parsed.stage2.claims.squares.len(), 3);
            assert_eq!(parsed.stage3.claims.0.len(), 36);
            // H-exact: 13 + 18 + 6 + 9 + 1 + 36 claims and 12 elements per round.
            let rounds = proof.stage1.rounds.len();
            assert_eq!(proof.piop_bytes(), 16 * (13 + 18 + 6 + 9 + 1 + 36 + 12 * rounds));
            assert_eq!(proof.size().total(), bytes.len());
            // A truncated proof is rejected.
            assert_eq!(AggregateProof::from_bytes(&bytes[..bytes.len() - 200]), Err(WireError::Truncated));
        }
        // At 2^15 the stage messages and claims take 6,128 bytes (25 rounds), as under H-u.
        assert_eq!(16 * (13 + 18 + 6 + 9 + 1 + 36 + 12 * 25), 6_128);
    }

    #[test]
    fn h_fips_proofs_round_trip_with_their_claim_counts() {
        let mut rng = SplitMix64::new(79);
        for records in [1, 100, 1000] {
            let proof = sample_proof_of(Relation::HFips, records, &mut rng);
            let bytes = proof.to_bytes();
            assert_eq!(bytes[2], 2);
            let parsed = AggregateProof::from_bytes(&bytes).unwrap();
            assert_eq!(parsed, proof);
            assert_eq!(parsed.relation, Relation::HFips);
            assert_eq!(parsed.stage2.claims.squares.len(), 2);
            assert_eq!(parsed.stage2.claims.fips.len(), 2);
            assert_eq!(parsed.stage3.claims.0.len(), 40);
            // H-FIPS: 13 + 18 + 6 + 10 + 1 + 40 claims and 12 elements per round.
            let rounds = proof.stage1.rounds.len();
            assert_eq!(proof.piop_bytes(), 16 * (13 + 18 + 6 + 10 + 1 + 40 + 12 * rounds));
            assert_eq!(proof.size().total(), bytes.len());
            assert_eq!(AggregateProof::from_bytes(&bytes[..bytes.len() - 200]), Err(WireError::Truncated));
        }
        // At 2^15 the stage messages and claims take 6,208 bytes (25 rounds).
        assert_eq!(16 * (13 + 18 + 6 + 10 + 1 + 40 + 12 * 25), 6_208);
    }

    #[test]
    fn ml_dsa_65_and_87_headers_carry_the_set_and_the_wider_c_tilde() {
        let mut rng = SplitMix64::new(80);
        for (set, nibble, width) in [(Set::S65, 0x10_u8, 48), (Set::S87, 0x20, 64)] {
            for records in [1, 100, 1000] {
                let proof = sample_proof_in(set, Relation::Hu, records, &mut rng);
                let bytes = proof.to_bytes();
                assert_eq!(bytes[2], nibble, "{set:?}");
                let parsed = AggregateProof::from_bytes(&bytes).unwrap();
                assert_eq!(parsed, proof);
                assert_eq!((parsed.set, parsed.c_tilde[0].len()), (set, width));
                let size = proof.size();
                assert_eq!(size.total(), bytes.len());
                assert_eq!(size.c_tilde, width * records);
                let manifest = CommitmentManifest::for_set(set, records, Relation::Hu).unwrap();
                assert_eq!(size.work, work_bytes(&manifest));
                assert_eq!(proof.stage1.rounds[0].len(), 3);
                // A rewritten set nibble changes every shape that follows, so the parse fails
                // or yields another proof.
                for other in [0x00_u8, 0x10, 0x20] {
                    if other == nibble {
                        continue;
                    }
                    let mut rewritten = bytes.clone();
                    rewritten[2] = other;
                    if let Ok(parsed) = AggregateProof::from_bytes(&rewritten) {
                        assert_ne!(parsed, proof);
                    }
                }
                // Unknown set nibbles and the H-exact and H-FIPS relations are refused.
                for bad in [0x30_u8, 0xF0, nibble | 1, nibble | 2] {
                    let mut flags = bytes.clone();
                    flags[2] = bad;
                    assert_eq!(AggregateProof::from_bytes(&flags), Err(WireError::Flags(bad)));
                }
                assert_eq!(AggregateProof::from_bytes(&bytes[..bytes.len() - 1]), Err(WireError::Truncated));
            }
        }
    }

    #[test]
    fn final_vectors_pack_least_significant_bit_first() {
        let bits: Vec<u8> = (0..4096).map(|i| ((i * 7) % 3 == 0) as u8).collect();
        let packed = pack_final_vector(&bits);
        assert_eq!(packed[0] & 1, bits[0]);
        assert_eq!((packed[0] >> 1) & 1, bits[1]);
        assert_eq!(unpack_final_vector(&packed), bits);
    }
}
