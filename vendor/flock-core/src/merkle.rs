//! Merkle commitments.

#[cfg(feature = "hash-count")]
pub use flock_merkle::hash_count;
pub use flock_merkle::{
    Blake3MerkleHash, Hash, HashKind, MerkleHash, Sha256MerkleHash, cap_depth, cap_layer,
    hash_leaf, hash_leaves_serial, hash_pair, hash_parents_serial, merkle_proof,
    merkle_proof_capped, merkle_root, merkle_root_with, merkle_tree,
    merkle_tree_from_prehashed_level, merkle_tree_sequential, merkle_tree_sequential_with,
    merkle_tree_with, verify_merkle_proof, verify_merkle_proof_capped,
    verify_merkle_proof_capped_with, verify_merkle_proof_with,
};

use core::{mem::size_of, slice::from_raw_parts};

use rayon::prelude::*;

use crate::field::F128;
use crate::ntt::additive_ntt_f128::LANES_PER_GROUP;

/// Leaves per rayon task (as upstream's batched leaf path).
const LEAF_GROUP: usize = 1024;

/// The Merkle tree whose leaf `p` is row `p` (the `lanes` words of position `p`, `16
/// lanes` bytes) of a lane-blocked codeword (`AdditiveNttF128::encode_lane_blocked`:
/// group `g` of four lanes at `codeword[4 g num_leaves ..]`, position `p` of the group at
/// `4 (g num_leaves + p)`). Identical to [`merkle_tree`] over the row-major codeword. The
/// rows are gathered sixteen at a time into a small buffer (one 64-byte line per group and
/// position, the last group's line cut to the committed lanes) and hashed from there.
pub fn merkle_tree_lane_blocked(codeword: &[F128], lanes: usize, num_leaves: usize, kind: HashKind) -> Vec<Hash> {
    let groups = lanes.div_ceil(LANES_PER_GROUP);
    assert_eq!(codeword.len(), groups * LANES_PER_GROUP * num_leaves);
    assert!(num_leaves.is_power_of_two() && lanes >= 1);
    let leaf = lanes * size_of::<F128>();
    let line = LANES_PER_GROUP * size_of::<F128>();
    // `vec!` of zero arrays takes the calloc path: lazy zero pages, no fill pass.
    let mut tree: Vec<Hash> = vec![[0u8; 32]; 2 * num_leaves - 1];
    tree[..num_leaves]
        .par_chunks_mut(LEAF_GROUP)
        .enumerate()
        .for_each(|(c, out)| {
            let mut buf = vec![0u8; 16 * leaf];
            for (k, o) in out.chunks_mut(16).enumerate() {
                let pos = c * LEAF_GROUP + 16 * k;
                let n = o.len();
                for g in 0..groups {
                    let bytes = line.min(leaf - g * line);
                    let src = &codeword[(g * num_leaves + pos) * LANES_PER_GROUP..][..n * LANES_PER_GROUP];
                    // SAFETY: F128 is two little-endian u64 words without padding.
                    let src: &[u8] = unsafe { from_raw_parts(src.as_ptr().cast(), n * line) };
                    for i in 0..n {
                        buf[i * leaf + g * line..i * leaf + g * line + bytes]
                            .copy_from_slice(&src[i * line..i * line + bytes]);
                    }
                }
                for (i, h) in o.iter_mut().enumerate() {
                    *h = hash_leaf(&buf[i * leaf..(i + 1) * leaf], kind);
                }
            }
        });
    merkle_tree_from_prehashed_level(tree, num_leaves, kind, 0)
}
