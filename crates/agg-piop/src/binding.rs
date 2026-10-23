//! Public-record binding (spec Section 2) and the canonical instance record of event 0.
//!
//! The transcript binds `X = LE64(N) || ||_i (pk_i || LE64(|M_i|) || M_i || c~_i)` with a
//! 16-KiB-leaf cSHAKE256 tree: leaf `i` is `cSHAKE256(block_i, 32)` under
//! `agg-mldsa/v1/record-leaf`, the root `cSHAKE256(LE64(blocks) || leaves, 32)` under
//! `agg-mldsa/v1/public-records-binding`. The instance record is the manifest binding
//! bytes of event 0 (`docs/piop.md` Section 2) followed by this digest and the Flock
//! profile id.

use agg_witness::params::C_TILDE_BYTES;
use rayon::prelude::*;

use crate::manifest::CommitmentManifest;
use crate::transcript::cshake256;

/// Leaf block size.
pub const BINDING_BLOCK_BYTES: usize = 16 * 1024;
const LEAF_CUSTOMIZATION: &[u8] = b"agg-mldsa/v1/record-leaf";
const BINDING_CUSTOMIZATION: &[u8] = b"agg-mldsa/v1/public-records-binding";

/// Flock profile ids bound in event 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum FlockProfile {
    /// No Keccak argument (mode B: the bits are not tied to the commitment hashes).
    None = 0,
    /// Ligerito `Fast` (engineering baseline).
    Fast = 1,
    /// LaBinius (target commitment).
    LaBinius = 2,
    /// Mode B: the `w_1'` bits in the clear as the event-2 items (`crate::mode_b`).
    PublicBits = 3,
}

impl FlockProfile {
    /// From the wire byte.
    #[must_use]
    pub fn from_id(id: u8) -> Option<Self> {
        match id {
            0 => Some(Self::None),
            1 => Some(Self::Fast),
            2 => Some(Self::LaBinius),
            3 => Some(Self::PublicBits),
            _ => None,
        }
    }

    /// Mode A: the hash relation is proved by Flock sub-proofs (`Fast` or `LaBinius`), whose
    /// tie to the committed bits is the parity bridge with seed-derived matrices (spec 9.4).
    #[must_use]
    pub const fn is_mode_a(self) -> bool {
        matches!(self, Self::Fast | Self::LaBinius)
    }
}

/// The canonical record stream `X`.
#[must_use]
pub fn record_stream(records: &[(&[u8], &[u8], [u8; C_TILDE_BYTES])]) -> Vec<u8> {
    let mut stream = Vec::new();
    stream.extend_from_slice(&(records.len() as u64).to_le_bytes());
    for (pk, msg, c_tilde) in records {
        stream.extend_from_slice(pk);
        stream.extend_from_slice(&(msg.len() as u64).to_le_bytes());
        stream.extend_from_slice(msg);
        stream.extend_from_slice(c_tilde);
    }
    stream
}

/// The 32-byte public-record digest.
#[must_use]
pub fn public_records_binding(records: &[(&[u8], &[u8], [u8; C_TILDE_BYTES])]) -> [u8; 32] {
    let stream = record_stream(records);
    let leaves: Vec<Vec<u8>> = stream
        .par_chunks(BINDING_BLOCK_BYTES)
        .map(|block| cshake256(LEAF_CUSTOMIZATION, block, 32))
        .collect();
    let mut root_input = Vec::with_capacity(8 + 32 * leaves.len());
    root_input.extend_from_slice(&(leaves.len() as u64).to_le_bytes());
    for leaf in &leaves {
        root_input.extend_from_slice(leaf);
    }
    cshake256(BINDING_CUSTOMIZATION, &root_input, 32)
        .try_into()
        .expect("32 bytes")
}

/// The canonical instance record of event 0.
#[must_use]
pub fn instance_record(
    manifest: &CommitmentManifest,
    record_digest: &[u8; 32],
    flock: FlockProfile,
) -> Vec<u8> {
    let mut record = manifest.binding_bytes();
    record.extend_from_slice(record_digest);
    record.push(flock as u8);
    record
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binding_depends_on_every_field_and_spans_blocks() {
        let pk = vec![1_u8; 1312];
        let msg = vec![2_u8; 40];
        let ct = [3_u8; C_TILDE_BYTES];
        let one = public_records_binding(&[(&pk, &msg, ct)]);
        assert_eq!(one, public_records_binding(&[(&pk, &msg, ct)]));
        let mut other_pk = pk.clone();
        other_pk[5] ^= 1;
        assert_ne!(one, public_records_binding(&[(&other_pk, &msg, ct)]));
        assert_ne!(one, public_records_binding(&[(&pk, &msg[..39], ct)]));
        let mut other_ct = ct;
        other_ct[0] ^= 1;
        assert_ne!(one, public_records_binding(&[(&pk, &msg, other_ct)]));
        // Thirteen records exceed one 16-KiB block.
        let many: Vec<(&[u8], &[u8], [u8; 32])> = (0..13).map(|_| (&pk[..], &msg[..], ct)).collect();
        assert!(record_stream(&many).len() > BINDING_BLOCK_BYTES);
        let digest = public_records_binding(&many);
        assert_ne!(digest, one);
        let manifest = CommitmentManifest::for_batch(13).unwrap();
        let record = instance_record(&manifest, &digest, FlockProfile::None);
        assert_eq!(record.len(), manifest.binding_bytes().len() + 33);
        assert_eq!(*record.last().unwrap(), 0);
        assert_eq!(FlockProfile::from_id(2), Some(FlockProfile::LaBinius));
        assert_eq!(FlockProfile::from_id(3), Some(FlockProfile::PublicBits));
        assert_eq!(FlockProfile::from_id(4), None);
        let mode_a: Vec<u8> = (0..4)
            .filter(|&id| FlockProfile::from_id(id).unwrap().is_mode_a())
            .collect();
        assert_eq!(mode_a, [1, 2]);
    }
}
