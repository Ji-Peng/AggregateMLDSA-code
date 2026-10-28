//! Public-record binding and the canonical instance record of event 0.
//!
//! The transcript binds `X = LE64(N) || ||_i (pk_i || LE64(|M_i|) || M_i || c~_i)` with a
//! 16-KiB-leaf cSHAKE256 tree: leaf `i` is `cSHAKE256(block_i, 32)` under
//! `agg-mldsa/v1/record-leaf`, the root `cSHAKE256(LE64(blocks) || leaves, 32)` under
//! `agg-mldsa/v1/public-records-binding`. The instance record is the manifest binding
//! bytes of event 0 (`docs/piop.md`) followed by this digest, the Flock
//! profile id and the 32-byte proof-of-work plan digest
//! ([`crate::pow::PowPlan::digest`], `docs/soundness.md`).

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
    /// Mode B: the `w_1'` bits in the clear as the event-2 items (`crate::mode_b`).
    PublicBits = 3,
    /// Ligerito `Slim` (rate 1/4, Flock's 128-bit work-normalized accounting as `Fast`):
    /// about two thirds of the `Fast` Flock bytes for about twice the level-0 encoding work.
    Slim = 4,
    /// Ligerito `Slim8` (rate 1/8 at level 0 with `Slim`'s accounting, a parameter set the
    /// vendored Flock core derives and validates with its own analysis).
    Slim8 = 5,
}

impl FlockProfile {
    /// From the wire byte.
    #[must_use]
    pub fn from_id(id: u8) -> Option<Self> {
        match id {
            0 => Some(Self::None),
            1 => Some(Self::Fast),
            3 => Some(Self::PublicBits),
            4 => Some(Self::Slim),
            5 => Some(Self::Slim8),
            _ => None,
        }
    }

    /// Mode A: the hash relation is proved by Flock sub-proofs (`Fast`, `Slim` or `Slim8`),
    /// whose tie to the committed bits is the parity bridge with seed-derived matrices.
    #[must_use]
    pub const fn is_mode_a(self) -> bool {
        matches!(self, Self::Fast | Self::Slim | Self::Slim8)
    }
}

/// The canonical record stream `X`, written in parallel: the record offsets are a prefix sum
/// of the record lengths, and every record is copied into its own disjoint slice.
#[must_use]
pub fn record_stream(records: &[(&[u8], &[u8], [u8; C_TILDE_BYTES])]) -> Vec<u8> {
    let sizes: Vec<usize> = records
        .iter()
        .map(|(pk, msg, _)| pk.len() + 8 + msg.len() + C_TILDE_BYTES)
        .collect();
    let mut stream = vec![0_u8; 8 + sizes.iter().sum::<usize>()];
    let (count, mut rest) = stream.split_at_mut(8);
    count.copy_from_slice(&(records.len() as u64).to_le_bytes());
    let mut slots: Vec<&mut [u8]> = Vec::with_capacity(records.len());
    for &size in &sizes {
        let (slot, tail) = std::mem::take(&mut rest).split_at_mut(size);
        slots.push(slot);
        rest = tail;
    }
    slots
        .into_par_iter()
        .zip(records.par_iter())
        .for_each(|(slot, (pk, msg, c_tilde))| {
            let (pk_part, slot) = slot.split_at_mut(pk.len());
            pk_part.copy_from_slice(pk);
            let (len_part, slot) = slot.split_at_mut(8);
            len_part.copy_from_slice(&(msg.len() as u64).to_le_bytes());
            let (msg_part, c_part) = slot.split_at_mut(msg.len());
            msg_part.copy_from_slice(msg);
            c_part.copy_from_slice(c_tilde);
        });
    stream
}

/// The record stream `X` built serially (the reference of [`record_stream`]).
#[cfg(test)]
fn record_stream_serial(records: &[(&[u8], &[u8], [u8; C_TILDE_BYTES])]) -> Vec<u8> {
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

/// The canonical instance record of event 0: the manifest bytes, the public-record digest,
/// the Flock profile id and the plan digest.
#[must_use]
pub fn instance_record(
    manifest: &CommitmentManifest,
    record_digest: &[u8; 32],
    flock: FlockProfile,
    plan_digest: &[u8; 32],
) -> Vec<u8> {
    let mut record = manifest.binding_bytes();
    record.extend_from_slice(record_digest);
    record.push(flock as u8);
    record.extend_from_slice(plan_digest);
    record
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_stream_matches_the_serial_stream() {
        let keys: Vec<Vec<u8>> = (0..5_u8).map(|k| vec![k; 1312]).collect();
        let msgs: Vec<Vec<u8>> = (0..300_usize).map(|i| vec![i as u8; i % 97]).collect();
        let records: Vec<(&[u8], &[u8], [u8; C_TILDE_BYTES])> = msgs
            .iter()
            .enumerate()
            .map(|(i, m)| (keys[i % 5].as_slice(), m.as_slice(), [i as u8; C_TILDE_BYTES]))
            .collect();
        for n in [0, 1, 2, 13, 300] {
            assert_eq!(record_stream(&records[..n]), record_stream_serial(&records[..n]));
        }
    }

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
        let record = instance_record(&manifest, &digest, FlockProfile::None, &[7; 32]);
        assert_eq!(record.len(), manifest.binding_bytes().len() + 65);
        assert_eq!(record[record.len() - 33], 0);
        assert_eq!(record[record.len() - 32..], [7; 32]);
        assert_eq!(FlockProfile::from_id(2), None);
        assert_eq!(FlockProfile::from_id(3), Some(FlockProfile::PublicBits));
        assert_eq!(FlockProfile::from_id(4), Some(FlockProfile::Slim));
        assert_eq!(FlockProfile::from_id(5), Some(FlockProfile::Slim8));
        assert_eq!(FlockProfile::from_id(6), None);
        let mode_a: Vec<u8> = (0..6)
            .filter(|&id| FlockProfile::from_id(id).is_some_and(FlockProfile::is_mode_a))
            .collect();
        assert_eq!(mode_a, [1, 4, 5]);
        for id in [0, 1, 3, 4, 5] {
            assert_eq!(FlockProfile::from_id(id).unwrap() as u8, id);
        }
    }
}
