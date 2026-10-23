//! Proof wire format (spec Section 10): header (version, flags, `N`), the `32N` bytes of
//! `c~_i`, then the payloads of events 1, 2, 4, 6, 8 to 11 in order, then the Akita
//! opening. Variable-length items carry a `LE32` length; field elements use the
//! canonical atom encoding of the transcript; the round counts follow from `N`.

use agg_witness::params::C_TILDE_BYTES;

use crate::claims::{Kappa1Claims, PreStageClaims, Stage1Claims, Stage2Claims, Stage3Claims};
use crate::manifest::{CommitmentManifest, RELATION_H, WIRE_VERSION};
use crate::stage1::Stage1Proof;
use crate::stage2::Stage2Proof;
use crate::stage3::Stage3Proof;
use crate::transcript::{decode_extensions, encode_extensions, FINAL_VECTOR_BYTES, STAGE_DEGREES};
use crate::OpeningField as E;

/// A malformed proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WireError {
    /// Unknown wire version.
    Version(u16),
    /// Unknown feature flags.
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
    /// Live records `N`.
    pub records: usize,
    /// `c~_i` per live record.
    pub c_tilde: Vec<[u8; C_TILDE_BYTES]>,
    /// Event 1: the round-1 Akita commitment (`G_main`).
    pub main_commitment: Vec<u8>,
    /// Event 2: `S` Flock commitments.
    pub flock_commitments: Vec<Vec<u8>>,
    /// Event 4: the round-2 Akita commitments (`G_1, G_2, G_3`).
    pub bridge_commitments: [Vec<u8>; 3],
    /// Event 6: `Y_sigma`, 4,096 bits packed least significant first.
    pub final_vectors: Vec<[u8; FINAL_VECTOR_BYTES]>,
    /// Event 6: `S` Flock proof byte strings.
    pub flock_proofs: Vec<Vec<u8>>,
    /// Event 8.
    pub prestage: PreStageClaims,
    /// Event 9.
    pub stage1: Stage1Proof,
    /// Event 10.
    pub stage2: Stage2Proof,
    /// Event 11.
    pub stage3: Stage3Proof,
    /// Event 12: the Akita opening.
    pub opening: Vec<u8>,
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
        w.bytes(&[RELATION_H]);
        w.bytes(&(self.records as u64).to_le_bytes());
        for c in &self.c_tilde {
            w.bytes(c);
        }
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
        for item in &self.flock_proofs {
            w.framed(item);
        }
        w.elements(&self.prestage.to_vec());
        w.rounds(&self.stage1.rounds);
        w.elements(&self.stage1.claims.to_vec());
        w.elements(&self.stage2.kappa1.to_vec());
        w.rounds(&self.stage2.rounds);
        w.elements(&self.stage2.claims.to_vec());
        w.rounds(&self.stage3.rounds);
        w.elements(&self.stage3.claims.0);
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
        if flags != RELATION_H {
            return Err(WireError::Flags(flags));
        }
        let records = u64::from_le_bytes(r.array()?);
        let manifest = usize::try_from(records)
            .ok()
            .and_then(|n| CommitmentManifest::for_batch(n).ok())
            .ok_or(WireError::Records(records))?;
        let records = manifest.records();
        let segments = manifest.segments();
        let rounds = manifest.d_main();
        let c_tilde = (0..records)
            .map(|_| r.array::<C_TILDE_BYTES>())
            .collect::<Result<Vec<_>, _>>()?;
        let main_commitment = r.framed()?;
        let count = u32::from_le_bytes(r.array()?) as usize;
        if count != segments {
            return Err(WireError::Shape("flock commitment count"));
        }
        let flock_commitments = (0..segments)
            .map(|_| r.framed())
            .collect::<Result<Vec<_>, _>>()?;
        let bridge_commitments = [r.framed()?, r.framed()?, r.framed()?];
        let final_vectors = (0..segments)
            .map(|_| r.array::<FINAL_VECTOR_BYTES>())
            .collect::<Result<Vec<_>, _>>()?;
        let flock_proofs = (0..segments)
            .map(|_| r.framed())
            .collect::<Result<Vec<_>, _>>()?;
        let prestage = PreStageClaims::from_slice(&r.elements(PreStageClaims::LEN)?)
            .ok_or(WireError::Shape("pre-stage claims"))?;
        let stage1 = Stage1Proof {
            rounds: r.rounds(rounds, STAGE_DEGREES[0])?,
            claims: Stage1Claims::from_slice(&r.elements(Stage1Claims::LEN)?)
                .ok_or(WireError::Shape("stage 1 claims"))?,
        };
        let stage2 = Stage2Proof {
            kappa1: Kappa1Claims::from_slice(&r.elements(Kappa1Claims::LEN)?)
                .ok_or(WireError::Shape("kappa1 claims"))?,
            rounds: r.rounds(rounds, STAGE_DEGREES[1])?,
            claims: Stage2Claims::from_slice(&r.elements(Stage2Claims::LEN)?)
                .ok_or(WireError::Shape("stage 2 claims"))?,
        };
        let stage3 = Stage3Proof {
            rounds: r.rounds(rounds, STAGE_DEGREES[2])?,
            claims: Stage3Claims::from_slice(&r.elements(Stage3Claims::LEN)?)
                .ok_or(WireError::Shape("stage 3 claims"))?,
        };
        let opening = r.framed()?;
        if r.at != bytes.len() {
            return Err(WireError::Trailing);
        }
        Ok(Self {
            records,
            c_tilde,
            main_commitment,
            flock_commitments,
            bridge_commitments,
            final_vectors,
            flock_proofs,
            prestage,
            stage1,
            stage2,
            stage3,
            opening,
        })
    }

    /// The PIOP part of the proof in bytes (events 8 to 11).
    #[must_use]
    pub fn piop_bytes(&self) -> usize {
        let rounds = self.stage1.rounds.len();
        16 * (PreStageClaims::LEN
            + Stage1Claims::LEN
            + Kappa1Claims::LEN
            + Stage2Claims::LEN
            + Stage3Claims::LEN
            + rounds * (STAGE_DEGREES[0] + STAGE_DEGREES[1] + STAGE_DEGREES[2]))
    }
}

/// Bytes of each part of a serialized proof (framing included).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProofSize {
    /// Version, flags, `N`.
    pub header: usize,
    /// The `32 N` bytes of `c~_i`.
    pub c_tilde: usize,
    /// Events 1 and 4: the four Akita commitments.
    pub akita_commitments: usize,
    /// Event 2: the Keccak-side items (Flock commitments; mode B: the public bits).
    pub keccak_commitments: usize,
    /// Event 6: the final vectors `Y_sigma`.
    pub final_vectors: usize,
    /// Event 6: the Keccak-side proofs.
    pub keccak_proofs: usize,
    /// Events 8 to 11: claims and round messages.
    pub piop: usize,
    /// Event 12: the Akita opening.
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
            + self.piop
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
            c_tilde: C_TILDE_BYTES * self.c_tilde.len(),
            akita_commitments: 4 + self.main_commitment.len() + framed(&self.bridge_commitments),
            keccak_commitments: 4 + framed(&self.flock_commitments),
            final_vectors: FINAL_VECTOR_BYTES * self.final_vectors.len(),
            keccak_proofs: framed(&self.flock_proofs),
            piop: self.piop_bytes(),
            opening: 4 + self.opening.len(),
        }
    }
}

/// Pack 4,096 bits (one byte each) into 512 bytes, least significant bit first.
#[must_use]
pub fn pack_final_vector(bits: &[u8]) -> [u8; FINAL_VECTOR_BYTES] {
    assert_eq!(bits.len(), 8 * FINAL_VECTOR_BYTES, "4,096 bits");
    let mut out = [0_u8; FINAL_VECTOR_BYTES];
    for (i, &bit) in bits.iter().enumerate() {
        out[i / 8] |= (bit & 1) << (i % 8);
    }
    out
}

/// Unpack 512 bytes into 4,096 bits (one byte each).
#[must_use]
pub fn unpack_final_vector(packed: &[u8; FINAL_VECTOR_BYTES]) -> Vec<u8> {
    (0..8 * FINAL_VECTOR_BYTES)
        .map(|i| (packed[i / 8] >> (i % 8)) & 1)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::point;
    use agg_witness::gen::SplitMix64;

    fn sample_proof(records: usize, rng: &mut SplitMix64) -> AggregateProof {
        let manifest = CommitmentManifest::for_batch(records).unwrap();
        let rounds = manifest.d_main();
        let segments = manifest.segments();
        let rounds_of = |degree: usize, rng: &mut SplitMix64| {
            (0..rounds).map(|_| point(rng, degree)).collect::<Vec<_>>()
        };
        AggregateProof {
            records,
            c_tilde: (0..records).map(|i| [i as u8; 32]).collect(),
            main_commitment: vec![1, 2, 3],
            flock_commitments: vec![vec![4; 10]; segments],
            bridge_commitments: [vec![5], vec![], vec![6, 7]],
            final_vectors: vec![[8; 512]; segments],
            flock_proofs: vec![vec![]; segments],
            prestage: PreStageClaims::from_slice(&point(rng, PreStageClaims::LEN)).unwrap(),
            stage1: Stage1Proof {
                rounds: rounds_of(4, rng),
                claims: Stage1Claims::from_slice(&point(rng, Stage1Claims::LEN)).unwrap(),
            },
            stage2: Stage2Proof {
                kappa1: Kappa1Claims::from_slice(&point(rng, Kappa1Claims::LEN)).unwrap(),
                rounds: rounds_of(5, rng),
                claims: Stage2Claims::from_slice(&point(rng, Stage2Claims::LEN)).unwrap(),
            },
            stage3: Stage3Proof {
                rounds: rounds_of(3, rng),
                claims: Stage3Claims(point(rng, Stage3Claims::LEN)),
            },
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
            assert_eq!(proof.piop_bytes(), 16 * (26 + 98 + 18 + 61 + 67 + 12 * proof.stage1.rounds.len()));
            assert_eq!(AggregateProof::from_bytes(&bytes[..bytes.len() - 1]), Err(WireError::Truncated));
            let mut trailing = bytes.clone();
            trailing.push(0);
            assert_eq!(AggregateProof::from_bytes(&trailing), Err(WireError::Trailing));
            let mut version = bytes.clone();
            version[0] = 9;
            assert_eq!(AggregateProof::from_bytes(&version), Err(WireError::Version(9)));
            let mut flags = bytes.clone();
            flags[2] = 1;
            assert_eq!(AggregateProof::from_bytes(&flags), Err(WireError::Flags(1)));
            let mut n = bytes.clone();
            n[3..11].copy_from_slice(&0_u64.to_le_bytes());
            assert_eq!(AggregateProof::from_bytes(&n), Err(WireError::Records(0)));
            // A non-canonical limb in the pre-stage claims.
            let offset = 11 + 32 * records + 4 + 3 + 4 + 4 + 10 + 4 + 1 + 4 + 4 + 2 + 512 + 4;
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
    fn final_vectors_pack_least_significant_bit_first() {
        let bits: Vec<u8> = (0..4096).map(|i| ((i * 7) % 3 == 0) as u8).collect();
        let packed = pack_final_vector(&bits);
        assert_eq!(packed[0] & 1, bits[0]);
        assert_eq!((packed[0] >> 1) & 1, bits[1]);
        assert_eq!(unpack_final_vector(&packed), bits);
    }
}
