//! Mode-A segment items of the aggregate wire format (spec Section 10, events 2 and 6;
//! `docs/deviations.md` K10).
//!
//! The aggregate itself is `agg_piop::wire::AggregateProof`: header (version, flags, `N`),
//! the `32 N` bytes of `c~_i`, then the payloads of events 1, 2, 4, 6, 8 to 11 and the Akita
//! opening. In mode A the event-2 item and the event-6 proof item of segment `sigma` are the
//! byte strings defined here, because a segment's Flock argument is split into parts of
//! `2^{d_part}` records (sub-batches), each with its own Flock commitment and proof:
//!
//! - event 2, [`SegmentCommitments`]: `version (u8) || d_part (u8) || parts (LE32)`, then
//!   per part `LE32(len) || commitment`, the part's `agg_keccak::tied::TiedCommitment` bytes
//!   (the Ligerito level-0 caps of its sponge and bridge instances, K7, K11);
//! - event 6, [`SegmentProofs`]: the shares `Y_{sigma, s}` of parts `s < parts - 1`, 512
//!   bytes each (4,096 bits packed least significant first, like `Y_sigma`), then per part
//!   `LE32(len) || tied-part proof` (the sponge and bridge proofs). The last part's share is
//!   not sent: it is `Y_sigma` XOR the others, so the shares XOR to `Y_sigma` by construction.
//!
//! Both items are absorbed by the PIOP transcript (event 2 before the bridge seed `sd`,
//! event 6 before `tau`), so `d_part`, the commitments, the shares and the proofs are bound.

use agg_piop::transcript::FINAL_VECTOR_BYTES;

/// Version byte of the segment items (2: the tied parts of deviation K11, two Flock caps
/// and two Flock proofs per part).
pub const ITEM_VERSION: u8 = 2;

/// A malformed segment item.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ItemError {
    /// Unknown item version.
    Version(u8),
    /// The buffer ended early or a length is out of range.
    Truncated,
    /// Bytes after the last part.
    Trailing,
    /// The part count differs from the one the segment's live records imply.
    Parts {
        /// Parts implied by the live records and `d_part`.
        expected: usize,
        /// Parts in the item.
        actual: usize,
    },
    /// `d_part` outside the admitted range.
    PartSize(u8),
}

impl std::fmt::Display for ItemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ItemError {}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], ItemError> {
        let end = self.at.checked_add(len).ok_or(ItemError::Truncated)?;
        let out = self.bytes.get(self.at..end).ok_or(ItemError::Truncated)?;
        self.at = end;
        Ok(out)
    }

    fn u32(&mut self) -> Result<usize, ItemError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("4 bytes")) as usize)
    }

    fn framed(&mut self) -> Result<Vec<u8>, ItemError> {
        let len = self.u32()?;
        Ok(self.take(len)?.to_vec())
    }

    fn finish(&self) -> Result<(), ItemError> {
        if self.at == self.bytes.len() {
            Ok(())
        } else {
            Err(ItemError::Trailing)
        }
    }
}

fn put_framed(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(
        &u32::try_from(bytes.len())
            .expect("item below 4 GiB")
            .to_le_bytes(),
    );
    out.extend_from_slice(bytes);
}

/// The event-2 item of one segment: the part size and one Flock commitment per part.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SegmentCommitments {
    /// `log2` of the record positions per part.
    pub d_part: u32,
    /// The parts' Flock commitments, in part order.
    pub commitments: Vec<Vec<u8>>,
}

impl SegmentCommitments {
    /// Serialize.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = vec![
            ITEM_VERSION,
            u8::try_from(self.d_part).expect("d_part < 256"),
        ];
        out.extend_from_slice(
            &u32::try_from(self.commitments.len())
                .expect("parts below 2^32")
                .to_le_bytes(),
        );
        for c in &self.commitments {
            put_framed(&mut out, c);
        }
        out
    }

    /// Parse.
    ///
    /// # Errors
    /// On a malformed item (the part count is checked by the caller).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ItemError> {
        let mut r = Reader { bytes, at: 0 };
        let version = r.take(1)?[0];
        if version != ITEM_VERSION {
            return Err(ItemError::Version(version));
        }
        let d_part = u32::from(r.take(1)?[0]);
        let parts = r.u32()?;
        // Each part needs at least its 4-byte length: refuse absurd counts before allocating.
        if parts > bytes.len() / 4 {
            return Err(ItemError::Truncated);
        }
        let commitments = (0..parts)
            .map(|_| r.framed())
            .collect::<Result<Vec<_>, _>>()?;
        r.finish()?;
        Ok(Self {
            d_part,
            commitments,
        })
    }
}

/// The event-6 proof item of one segment: the shares of all parts but the last, and one
/// Flock proof per part.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SegmentProofs {
    /// `Y_{sigma, s}` for `s < parts - 1`, packed (512 bytes each).
    pub shares: Vec<[u8; FINAL_VECTOR_BYTES]>,
    /// The parts' Flock proofs, in part order.
    pub proofs: Vec<Vec<u8>>,
}

impl SegmentProofs {
    /// Serialize.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        assert_eq!(
            self.shares.len() + 1,
            self.proofs.len().max(1),
            "one share per part but the last"
        );
        let mut out = Vec::new();
        for share in &self.shares {
            out.extend_from_slice(share);
        }
        for proof in &self.proofs {
            put_framed(&mut out, proof);
        }
        out
    }

    /// Parse an item of `parts` parts.
    ///
    /// # Errors
    /// On a malformed item.
    pub fn from_bytes(bytes: &[u8], parts: usize) -> Result<Self, ItemError> {
        let mut r = Reader { bytes, at: 0 };
        let shares = (0..parts.saturating_sub(1))
            .map(|_| {
                r.take(FINAL_VECTOR_BYTES)
                    .map(|s| s.try_into().expect("512 bytes"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let proofs = (0..parts)
            .map(|_| r.framed())
            .collect::<Result<Vec<_>, _>>()?;
        r.finish()?;
        Ok(Self { shares, proofs })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_items_round_trip_and_malformed_items_are_rejected() {
        let c = SegmentCommitments {
            d_part: 10,
            commitments: vec![vec![1; 40], vec![], vec![2, 3]],
        };
        let bytes = c.to_bytes();
        assert_eq!(bytes.len(), 2 + 4 + 3 * 4 + 42);
        assert_eq!(SegmentCommitments::from_bytes(&bytes), Ok(c.clone()));
        assert_eq!(
            SegmentCommitments::from_bytes(&bytes[..bytes.len() - 1]),
            Err(ItemError::Truncated)
        );
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert_eq!(
            SegmentCommitments::from_bytes(&trailing),
            Err(ItemError::Trailing)
        );
        let mut version = bytes.clone();
        version[0] = 9;
        assert_eq!(
            SegmentCommitments::from_bytes(&version),
            Err(ItemError::Version(9))
        );
        let mut huge = bytes;
        huge[2..6].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(
            SegmentCommitments::from_bytes(&huge),
            Err(ItemError::Truncated)
        );
        assert_eq!(
            SegmentCommitments::from_bytes(&[]),
            Err(ItemError::Truncated)
        );
        // An empty segment: no parts.
        let empty = SegmentCommitments {
            d_part: 3,
            commitments: Vec::new(),
        };
        assert_eq!(SegmentCommitments::from_bytes(&empty.to_bytes()), Ok(empty));

        let p = SegmentProofs {
            shares: vec![[7; 512], [8; 512]],
            proofs: vec![vec![1], vec![2; 3], vec![]],
        };
        let bytes = p.to_bytes();
        assert_eq!(bytes.len(), 1024 + 12 + 4);
        assert_eq!(SegmentProofs::from_bytes(&bytes, 3), Ok(p));
        assert!(SegmentProofs::from_bytes(&bytes, 2).is_err());
        assert!(SegmentProofs::from_bytes(&bytes, 4).is_err());
        let none = SegmentProofs {
            shares: Vec::new(),
            proofs: Vec::new(),
        };
        assert!(none.to_bytes().is_empty());
        assert_eq!(SegmentProofs::from_bytes(&[], 0), Ok(none));
        assert_eq!(SegmentProofs::from_bytes(&[0], 0), Err(ItemError::Trailing));
    }
}
