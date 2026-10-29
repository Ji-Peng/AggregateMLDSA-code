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

/// BLAKE3 leaves absorbed one lane group at a time, for the commit that hashes the
/// level-0 rows while the transformed blocks are in cache.
///
/// A row of `lanes` words is `16 lanes` bytes (at most 1,024: one BLAKE3 chunk), so lane
/// group `g` of a row is exactly message block `g` of the leaf: a full 64-byte block for
/// every group but the last, which carries `16 (lanes - 4 g)` bytes. A caller transforms
/// group `g` of a position range, passes its lines to [`GroupLeafHasher::absorb`] and
/// moves on; after the last group [`GroupLeafHasher::finish`] writes the chaining values,
/// which equal [`hash_leaf`] of the rows under [`HashKind::Blake3`] (the non-root chaining
/// value of chunk 0).
///
/// The state holds eight 32-bit chaining words per position, transposed per batch of
/// sixteen positions (`state[128 b + 16 w + i]` = word `w` of position `16 b + i`), so the
/// AVX-512 path keeps a batch in eight registers with no transpose between groups and the
/// portable path reads the same layout.
#[derive(Clone, Copy, Debug)]
pub struct GroupLeafHasher {
    lanes: usize,
    groups: usize,
}

impl GroupLeafHasher {
    /// The hasher for rows of `lanes` words, or `None` when the leaves are not single
    /// BLAKE3 chunks (another hash, or more than 64 lanes).
    pub fn new(lanes: usize, kind: HashKind) -> Option<Self> {
        (kind == HashKind::Blake3 && (1..=64).contains(&lanes)).then(|| Self {
            lanes,
            groups: lanes.div_ceil(LANES_PER_GROUP),
        })
    }

    /// Lane groups per row (message blocks per leaf).
    pub fn groups(&self) -> usize {
        self.groups
    }

    /// State words for `positions` leaves.
    pub fn state_len(positions: usize) -> usize {
        128 * positions.div_ceil(16)
    }

    /// Absorb group `g` of `positions = lines.len() / 4` consecutive leaves: `lines` holds
    /// the group's 64-byte line of each position (lanes past the committed count read as
    /// zero and are cut off by the block length). Group 0 starts from the IV and ignores
    /// what `state` held.
    pub fn absorb(&self, state: &mut [u32], g: usize, lines: &[F128]) {
        assert!(g < self.groups && lines.len().is_multiple_of(LANES_PER_GROUP));
        let positions = lines.len() / LANES_PER_GROUP;
        assert!(state.len() >= Self::state_len(positions));
        let words = if g + 1 == self.groups { 4 * (self.lanes - LANES_PER_GROUP * g) } else { 16 };
        let flags = if g == 0 { blake3_groups::CHUNK_START } else { 0 }
            | if g + 1 == self.groups { blake3_groups::CHUNK_END } else { 0 };
        let full = positions / 16;
        #[cfg(target_arch = "x86_64")]
        if blake3_groups::x16_available() {
            for b in 0..full {
                let st: &mut [u32; 128] = (&mut state[128 * b..128 * (b + 1)]).try_into().expect("128 words");
                // SAFETY: AVX-512F is available (checked above); the sixteen lines of batch
                // b are `lines[64 b .. 64 (b + 1)]`, inside the slice.
                unsafe { blake3_groups::absorb16(st, lines[64 * b..].as_ptr(), words, flags, g == 0, LANES_PER_GROUP) };
            }
            blake3_groups::absorb_portable(state, lines, 16 * full, positions, words, flags, g == 0);
            return;
        }
        let _ = full;
        blake3_groups::absorb_portable(state, lines, 0, positions, words, flags, g == 0);
    }

    /// Hash native interleaved rows without rearranging lane groups in memory.
    pub fn hash_interleaved(&self, rows: &[F128], out: &mut [Hash]) {
        assert_eq!(rows.len(), out.len() * self.lanes);
        if !blake3_groups::x16_available() {
            for (row, hash) in rows.chunks_exact(self.lanes).zip(out) {
                // SAFETY: F128 is sixteen initialized bytes with the canonical leaf layout.
                let bytes = unsafe { std::slice::from_raw_parts(row.as_ptr().cast::<u8>(), row.len() * 16) };
                *hash = hash_leaf(bytes, HashKind::Blake3);
            }
            return;
        }
        for (batch, hashes) in rows.chunks(16 * self.lanes).zip(out.chunks_mut(16)) {
            let mut state = [0u32; 128];
            for g in 0..self.groups {
                let words = 4 * (self.lanes - g * 4).min(4);
                let flags = if g == 0 { blake3_groups::CHUNK_START } else { 0 }
                    | if g + 1 == self.groups { blake3_groups::CHUNK_END } else { 0 };
                #[cfg(target_arch = "x86_64")]
                if hashes.len() == 16 && blake3_groups::x16_available() {
                    // SAFETY: feature detection passed; sixteen strided group lines are
                    // inside the batch. The final partial group is masked by words.
                    unsafe { blake3_groups::absorb16(&mut state, batch[g * 4..].as_ptr(), words, flags, g == 0, self.lanes) };
                    continue;
                }
                for i in 0..hashes.len() {
                    let row = &batch[i * self.lanes + g * 4..][..words / 4];
                    // A single leaf uses the same transposed state with lane zero.
                    let mut one = [0u32; 128];
                    for w in 0..8 { one[16 * w] = state[16 * w + i]; }
                    blake3_groups::absorb_portable(&mut one, row, 0, 1, words, flags, g == 0);
                    for w in 0..8 { state[16 * w + i] = one[16 * w]; }
                }
            }
            self.finish(&state, hashes);
        }
    }

    /// The chaining values of the `out.len()` leaves after the last group.
    pub fn finish(&self, state: &[u32], out: &mut [Hash]) {
        assert!(state.len() >= Self::state_len(out.len()));
        for (i, h) in out.iter_mut().enumerate() {
            let (b, k) = (i / 16, i % 16);
            for w in 0..8 {
                h[4 * w..4 * w + 4].copy_from_slice(&state[128 * b + 16 * w + k].to_le_bytes());
            }
        }
    }
}

/// BLAKE3 compression for [`GroupLeafHasher`]: every leaf is chunk 0 (counter zero) and
/// never the root.
mod blake3_groups {
    use crate::field::F128;
    use crate::ntt::additive_ntt_f128::LANES_PER_GROUP;

    pub(super) const IV: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    pub(super) const CHUNK_START: u32 = 1;
    pub(super) const CHUNK_END: u32 = 2;
    const MSG_PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

    /// Whether the sixteen-lane AVX-512 path may run (`FLOCK_SIMD` and test masks apply).
    pub(super) fn x16_available() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            crate::simd::has(crate::simd::AVX512F)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    #[inline(always)]
    fn g(v: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, mx: u32, my: u32) {
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(mx);
        v[d] = (v[d] ^ v[a]).rotate_right(16);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(12);
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(my);
        v[d] = (v[d] ^ v[a]).rotate_right(8);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(7);
    }

    /// The BLAKE3 compression function (counter zero), returning the next chaining value.
    pub(super) fn compress(cv: &[u32; 8], m: &[u32; 16], block_len: u32, flags: u32) -> [u32; 8] {
        let mut v = [0u32; 16];
        v[..8].copy_from_slice(cv);
        v[8..12].copy_from_slice(&IV[..4]);
        v[14] = block_len;
        v[15] = flags;
        let mut m = *m;
        for round in 0..7 {
            g(&mut v, 0, 4, 8, 12, m[0], m[1]);
            g(&mut v, 1, 5, 9, 13, m[2], m[3]);
            g(&mut v, 2, 6, 10, 14, m[4], m[5]);
            g(&mut v, 3, 7, 11, 15, m[6], m[7]);
            g(&mut v, 0, 5, 10, 15, m[8], m[9]);
            g(&mut v, 1, 6, 11, 12, m[10], m[11]);
            g(&mut v, 2, 7, 8, 13, m[12], m[13]);
            g(&mut v, 3, 4, 9, 14, m[14], m[15]);
            if round < 6 {
                let p = m;
                for i in 0..16 {
                    m[i] = p[MSG_PERMUTATION[i]];
                }
            }
        }
        core::array::from_fn(|i| v[i] ^ v[i + 8])
    }

    /// Positions `from .. to` through the scalar compression: `words` message words of
    /// each line (the rest zero), block length `4 words` bytes.
    pub(super) fn absorb_portable(
        state: &mut [u32],
        lines: &[F128],
        from: usize,
        to: usize,
        words: usize,
        flags: u32,
        first: bool,
    ) {
        for p in from..to {
            let (b, k) = (p / 16, p % 16);
            let cv: [u32; 8] = if first { IV } else { core::array::from_fn(|w| state[128 * b + 16 * w + k]) };
            let mut m = [0u32; 16];
            for (w, mw) in m.iter_mut().enumerate().take(words) {
                let x = lines[p * LANES_PER_GROUP + w / 4];
                let limb = if w % 4 < 2 { x.lo } else { x.hi };
                *mw = (limb >> (32 * (w % 2))) as u32;
            }
            let out = compress(&cv, &m, 4 * words as u32, flags);
            for w in 0..8 {
                state[128 * b + 16 * w + k] = out[w];
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        __m512i, _mm512_add_epi32, _mm512_loadu_si512, _mm512_maskz_loadu_epi32, _mm512_ror_epi32,
        _mm512_set1_epi32, _mm512_setzero_si512, _mm512_shuffle_i32x4, _mm512_storeu_si512,
        _mm512_unpackhi_epi32, _mm512_unpackhi_epi64, _mm512_unpacklo_epi32, _mm512_unpacklo_epi64,
        _mm512_xor_si512,
    };

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx512f")]
    fn g16(v: &mut [__m512i; 16], a: usize, b: usize, c: usize, d: usize, mx: __m512i, my: __m512i) {
        v[a] = _mm512_add_epi32(_mm512_add_epi32(v[a], v[b]), mx);
        v[d] = _mm512_ror_epi32::<16>(_mm512_xor_si512(v[d], v[a]));
        v[c] = _mm512_add_epi32(v[c], v[d]);
        v[b] = _mm512_ror_epi32::<12>(_mm512_xor_si512(v[b], v[c]));
        v[a] = _mm512_add_epi32(_mm512_add_epi32(v[a], v[b]), my);
        v[d] = _mm512_ror_epi32::<8>(_mm512_xor_si512(v[d], v[a]));
        v[c] = _mm512_add_epi32(v[c], v[d]);
        v[b] = _mm512_ror_epi32::<7>(_mm512_xor_si512(v[b], v[c]));
    }

    /// [`compress`] of sixteen states at once: word `i` of every state in `cv[i]`, word `w`
    /// of every message block in `m[w]`.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx512f")]
    fn compress16(cv: &[__m512i; 8], m: &[__m512i; 16], block_len: u32, flags: u32) -> [__m512i; 8] {
        let mut v = [_mm512_setzero_si512(); 16];
        v[..8].copy_from_slice(cv);
        for i in 0..4 {
            v[8 + i] = _mm512_set1_epi32(IV[i] as i32);
        }
        v[14] = _mm512_set1_epi32(block_len as i32);
        v[15] = _mm512_set1_epi32(flags as i32);
        let mut m = *m;
        for round in 0..7 {
            g16(&mut v, 0, 4, 8, 12, m[0], m[1]);
            g16(&mut v, 1, 5, 9, 13, m[2], m[3]);
            g16(&mut v, 2, 6, 10, 14, m[4], m[5]);
            g16(&mut v, 3, 7, 11, 15, m[6], m[7]);
            g16(&mut v, 0, 5, 10, 15, m[8], m[9]);
            g16(&mut v, 1, 6, 11, 12, m[10], m[11]);
            g16(&mut v, 2, 7, 8, 13, m[12], m[13]);
            g16(&mut v, 3, 4, 9, 14, m[14], m[15]);
            if round < 6 {
                let p = m;
                for i in 0..16 {
                    m[i] = p[MSG_PERMUTATION[i]];
                }
            }
        }
        core::array::from_fn(|i| _mm512_xor_si512(v[i], v[i + 8]))
    }

    /// 16 x 16 transpose of 32-bit words: on return `r[w]` lane `i` is input `r[i]` lane `w`
    /// (unpack 32, unpack 64, two 128-bit lane shuffles).
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx512f")]
    fn transpose16(r: &mut [__m512i; 16]) {
        let mut a = [_mm512_setzero_si512(); 16];
        for i in 0..8 {
            a[2 * i] = _mm512_unpacklo_epi32(r[2 * i], r[2 * i + 1]);
            a[2 * i + 1] = _mm512_unpackhi_epi32(r[2 * i], r[2 * i + 1]);
        }
        let mut b = [_mm512_setzero_si512(); 16];
        for q in 0..4 {
            let (ab0, ab2, cd0, cd2) = (a[4 * q], a[4 * q + 1], a[4 * q + 2], a[4 * q + 3]);
            b[4 * q] = _mm512_unpacklo_epi64(ab0, cd0);
            b[4 * q + 1] = _mm512_unpackhi_epi64(ab0, cd0);
            b[4 * q + 2] = _mm512_unpacklo_epi64(ab2, cd2);
            b[4 * q + 3] = _mm512_unpackhi_epi64(ab2, cd2);
        }
        let mut c = [_mm512_setzero_si512(); 16];
        for h in 0..2 {
            for k in 0..4 {
                let (x, y) = (b[8 * h + k], b[8 * h + 4 + k]);
                c[8 * h + k] = _mm512_shuffle_i32x4::<0x88>(x, y);
                c[8 * h + 4 + k] = _mm512_shuffle_i32x4::<0xdd>(x, y);
            }
        }
        for k in 0..8 {
            r[k] = _mm512_shuffle_i32x4::<0x88>(c[k], c[8 + k]);
            r[8 + k] = _mm512_shuffle_i32x4::<0xdd>(c[k], c[8 + k]);
        }
    }

    /// One group of sixteen consecutive leaves: `lines` points at their sixteen 64-byte
    /// lines, of which the first `words` 32-bit words are message (a masked load; the rest
    /// is BLAKE3's zero padding).
    ///
    /// # Safety
    /// AVX-512F is available; sixteen lines separated by `stride` words are readable.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn absorb16(state: &mut [u32; 128], lines: *const F128, words: usize, flags: u32, first: bool, stride: usize) {
        debug_assert!((1..=16).contains(&words));
        let mask = if words == 16 { u16::MAX } else { (1u16 << words) - 1 };
        let mut m = [_mm512_setzero_si512(); 16];
        for (i, mi) in m.iter_mut().enumerate() {
            // SAFETY: line i is four F128 words (64 bytes) inside the caller's range.
            *mi = unsafe {
                let p = lines.add(i * stride).cast::<i32>();
                if words == 16 { _mm512_loadu_si512(p.cast()) } else { _mm512_maskz_loadu_epi32(mask, p) }
            };
        }
        transpose16(&mut m);
        let mut cv = [_mm512_setzero_si512(); 8];
        for (w, c) in cv.iter_mut().enumerate() {
            // SAFETY: `state` holds 8 x 16 words.
            *c = if first {
                _mm512_set1_epi32(IV[w] as i32)
            } else {
                unsafe { _mm512_loadu_si512(state[16 * w..].as_ptr().cast()) }
            };
        }
        let out = compress16(&cv, &m, 4 * words as u32, flags);
        for (w, o) in out.iter().enumerate() {
            // SAFETY: as above.
            unsafe { _mm512_storeu_si512(state[16 * w..].as_mut_ptr().cast(), *o) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{GroupLeafHasher, Hash, HashKind, LANES_PER_GROUP, hash_leaf};
    use crate::{
        field::F128,
        simd::{test_masks, with_mask},
        test_rng::Rng,
    };

    /// Absorbing the lane groups one by one gives `hash_leaf` of the rows for every lane
    /// count mod 4 up to one full chunk, position counts with and without a partial batch
    /// of sixteen, and every SIMD mask.
    #[test]
    fn group_leaf_hasher_matches_hash_leaf() {
        assert!(GroupLeafHasher::new(65, HashKind::Blake3).is_none());
        assert!(GroupLeafHasher::new(8, HashKind::Sha256).is_none());
        let mut rng = Rng::new(0x6E0F);
        for &lanes in &[1usize, 2, 3, 4, 5, 13, 17, 34, 63, 64] {
            for &positions in &[1usize, 7, 16, 48, 77] {
                let groups = lanes.div_ceil(LANES_PER_GROUP);
                // Group-major lines; lanes past `lanes` in the last group are zero.
                let mut lines = vec![F128::ZERO; groups * positions * LANES_PER_GROUP];
                for g in 0..groups {
                    for p in 0..positions {
                        for k in 0..LANES_PER_GROUP.min(lanes - g * LANES_PER_GROUP) {
                            lines[(g * positions + p) * LANES_PER_GROUP + k] = rng.f128();
                        }
                    }
                }
                let want: Vec<Hash> = (0..positions)
                    .map(|p| {
                        let mut bytes = Vec::with_capacity(16 * lanes);
                        for l in 0..lanes {
                            let w = lines[((l / 4) * positions + p) * LANES_PER_GROUP + l % 4];
                            bytes.extend_from_slice(&w.lo.to_le_bytes());
                            bytes.extend_from_slice(&w.hi.to_le_bytes());
                        }
                        hash_leaf(&bytes, HashKind::Blake3)
                    })
                    .collect();
                let hasher = GroupLeafHasher::new(lanes, HashKind::Blake3).expect("one chunk");
                for mask in test_masks() {
                    let got = with_mask(mask, || {
                        let mut state = vec![0xA5A5_A5A5u32; GroupLeafHasher::state_len(positions)];
                        for g in 0..groups {
                            let w = positions * LANES_PER_GROUP;
                            hasher.absorb(&mut state, g, &lines[g * w..(g + 1) * w]);
                        }
                        let mut out = vec![[0u8; 32]; positions];
                        hasher.finish(&state, &mut out);
                        out
                    });
                    assert!(got == want, "lanes={lanes} positions={positions} mask={mask:#x}");
                }
            }
        }
    }
}
