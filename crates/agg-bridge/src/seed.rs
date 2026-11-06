//! Bridge seed to the three bridge matrices.
//!
//! Column `zeta` of `M_t` (`t` = 1, 2, 3, the level number) is a `k_t`-subset of the
//! `m_t = 2^{e_t}` rows (`e_t` = 13, 13, 12): read 16-bit little-endian words from
//! `cSHAKE256(sd || LE64(t) || LE64(zeta), *, "", "agg-mldsa/v1/bridge")`, keep the low
//! `e_t` bits of each word, skip a row already chosen for the column, and stop after `k_t`
//! distinct rows. Rows are stored in the order drawn. Prover and verifier both derive the
//! matrices from the transcript's `sd`; neither transmits them.
//!
//! A column depends on `(sd, t, zeta)` only, so a matrix may hold a prefix of its columns:
//! [`derive_matrices_for`] derives the `J_t^used` columns of each level that the bridge reads
//! (level 2 of the production shapes uses `2^17` or `2^18` of its `2^19` columns) and every
//! derived column equals the one of the full matrix. Columns are hashed eight at a time on
//! the process-wide SIMD engine (`agg_mldsa_simd::Engine::keccak_f8`), with the same output on
//! every backend.

use agg_mldsa_simd::Engine;
use agg_witness::bridge::{BridgeMatrices, SparseMatrix, M1, M2, M3};
use rayon::prelude::*;
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::{CShake256, CShake256Core};

use crate::shape::Shape;

/// Module name, used by smoke tests.
pub const NAME: &str = "agg-bridge seed";

/// cSHAKE256 customization string `S` of the derivation (the function name `N` is empty).
pub const BRIDGE_CUSTOMIZATION: &[u8] = b"agg-mldsa/v1/bridge";

/// Length of the bridge seed `sd` (drawn at the bridge-seed step).
pub const SEED_BYTES: usize = 32;

/// Shape of one bridge matrix: level number, rows `2^rows_log2`, columns, column weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Level {
    /// The level number `t` absorbed into the hash input (1, 2 or 3).
    pub t: u64,
    /// `log2 m_t`: the number of low bits kept from each 16-bit word.
    pub rows_log2: u32,
    /// `J_t`.
    pub cols: usize,
    /// `k_t`.
    pub k: usize,
}

/// The three production levels (`Shape::PRODUCTION`).
pub const LEVELS: [Level; 3] = [
    Shape::PRODUCTION.levels[0].seed_level(),
    Shape::PRODUCTION.levels[1].seed_level(),
    Shape::PRODUCTION.levels[2].seed_level(),
];

const _: () = {
    assert!(M1.is_power_of_two() && M2.is_power_of_two() && M3.is_power_of_two());
    assert!(M1 <= 1 << 16 && M2 <= 1 << 16 && M3 <= 1 << 16);
};

impl Level {
    /// `m_t`.
    #[must_use]
    pub const fn rows(&self) -> usize {
        1 << self.rows_log2
    }
}

/// The cSHAKE256 state after the `bytepad(encode_string(N) || encode_string(S))` prefix,
/// cloned once per column.
#[must_use]
pub fn prefix_state() -> CShake256 {
    CShake256::from_core(CShake256Core::new(BRIDGE_CUSTOMIZATION))
}

/// The hash input of column `zeta` of level `t`: `sd || LE64(t) || LE64(zeta)`.
#[must_use]
pub fn column_input(sd: &[u8; SEED_BYTES], t: u64, zeta: u64) -> [u8; SEED_BYTES + 16] {
    let mut input = [0_u8; SEED_BYTES + 16];
    input[..SEED_BYTES].copy_from_slice(sd);
    input[SEED_BYTES..SEED_BYTES + 8].copy_from_slice(&t.to_le_bytes());
    input[SEED_BYTES + 8..].copy_from_slice(&zeta.to_le_bytes());
    input
}

/// cSHAKE256's rate in bytes.
const RATE: usize = 136;

/// The column hash on the bare Keccak-f[1600] permutation: the state after
/// cSHAKE256's `bytepad(encode_string("") || encode_string(S), 136)` block is computed
/// once; a column XORs its 48-byte input with cSHAKE's padding (`0x04 .. 0x80`) into a copy
/// and squeezes. One permutation per column (two past 136 output bytes), with none of the
/// generic sponge's buffering; the output equals [`derive_column_into`]'s (test
/// `keccak_columns_match_the_cshake_columns`).
#[derive(Clone, Debug)]
pub struct ColumnHasher {
    prefix: [u64; 25],
}

impl Default for ColumnHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl ColumnHasher {
    /// The prefix state of [`BRIDGE_CUSTOMIZATION`].
    #[must_use]
    pub fn new() -> Self {
        // left_encode(136) || left_encode(0) || left_encode(8 |S|) || S, zero padded.
        let bits = 8 * BRIDGE_CUSTOMIZATION.len();
        assert!(bits < 256, "a one-byte left_encode");
        let mut block = [0_u8; RATE];
        let mut head = vec![1, RATE as u8, 1, 0, 1, bits as u8];
        head.extend_from_slice(BRIDGE_CUSTOMIZATION);
        block[..head.len()].copy_from_slice(&head);
        let mut prefix = [0_u64; 25];
        absorb(&mut prefix, &block);
        keccak::f1600(&mut prefix);
        Self { prefix }
    }

    /// Fill `out` (length `k_t`) with the rows of column `zeta` of `level` (the reading rule
    /// of [`derive_column_into`]).
    pub fn column_into(&self, sd: &[u8; SEED_BYTES], level: &Level, zeta: usize, out: &mut [u32]) {
        assert_eq!(out.len(), level.k, "one row per unit of column weight");
        assert!(level.k <= level.rows(), "column weight above the row count");
        let input = column_input(sd, level.t, zeta as u64);
        let mut block = [0_u8; RATE];
        block[..input.len()].copy_from_slice(&input);
        block[input.len()] ^= 0x04;
        block[RATE - 1] ^= 0x80;
        let mut state = self.prefix;
        absorb(&mut state, &block);
        let mut filled = 0;
        while filled < out.len() {
            keccak::f1600(&mut state);
            filled = read_rows(level, |w| state[w], filled, out);
        }
    }

    /// Columns `first .. first + out.len() / k_t` (at most eight) of `level` into `out`, one
    /// column of `k_t` rows after another: the eight first permutations run in lockstep on
    /// `engine`, and a column that needs a second output block (fewer than `k_t` distinct
    /// rows among its first 68 words) is finished by [`Self::column_into`].
    pub fn columns8_into(
        &self,
        engine: &Engine,
        sd: &[u8; SEED_BYTES],
        level: &Level,
        first: usize,
        out: &mut [u32],
    ) {
        let k = level.k;
        assert!(k > 0 && out.len().is_multiple_of(k) && out.len() <= 8 * k, "up to eight whole columns");
        assert!(k <= level.rows(), "column weight above the row count");
        // The padded input block of each column (`column_input` then cSHAKE's `0x04 .. 0x80`):
        // lanes 0-3 hold `sd`, lane 4 `t`, lane 5 `zeta`, lane 6 the `0x04` byte and lane 16
        // the final `0x80` byte. Unused states hash columns past the end and are discarded.
        let mut state: [[u64; 8]; 25] = std::array::from_fn(|w| [self.prefix[w]; 8]);
        for (w, chunk) in sd.as_chunks::<8>().0.iter().enumerate() {
            let lane = u64::from_le_bytes(*chunk);
            state[w].iter_mut().for_each(|x| *x ^= lane);
        }
        state[4].iter_mut().for_each(|x| *x ^= level.t);
        for (i, x) in state[5].iter_mut().enumerate() {
            *x ^= (first + i) as u64;
        }
        state[6].iter_mut().for_each(|x| *x ^= 0x04);
        state[16].iter_mut().for_each(|x| *x ^= 0x80 << 56);
        engine.keccak_f8(&mut state);
        for (i, column) in out.chunks_exact_mut(k).enumerate() {
            if read_rows(level, |w| state[w][i], 0, column) < k {
                self.column_into(sd, level, first + i, column);
            }
        }
    }
}

/// Append to `out[..filled]` the distinct rows read from one output block (lanes `0 .. 17`
/// of a state, given by `lane`): 16-bit little-endian words masked to `e_t` bits, a row
/// already in the column skipped, stopping at `k_t` rows. Returns the new fill.
#[inline]
fn read_rows(level: &Level, lane: impl Fn(usize) -> u64, mut filled: usize, out: &mut [u32]) -> usize {
    let mask = (1_u32 << level.rows_log2) - 1;
    for w in 0..RATE / 8 {
        let lane = lane(w);
        for q in 0..4 {
            let row = ((lane >> (16 * q)) as u32) & mask;
            if !out[..filled].contains(&row) {
                out[filled] = row;
                filled += 1;
                if filled == out.len() {
                    return filled;
                }
            }
        }
    }
    filled
}

/// XOR a rate block into the state (little-endian lanes).
fn absorb(state: &mut [u64; 25], block: &[u8; RATE]) {
    for (lane, chunk) in state.iter_mut().zip(block.as_chunks::<8>().0) {
        *lane ^= u64::from_le_bytes(*chunk);
    }
}

/// Fill `out` (length `k_t`) with the rows of column `zeta` of `level` (the generic cSHAKE256
/// sponge; [`ColumnHasher`] computes the same on the bare permutation).
pub fn derive_column_into(
    prefix: &CShake256,
    sd: &[u8; SEED_BYTES],
    level: &Level,
    zeta: usize,
    out: &mut [u32],
) {
    assert_eq!(out.len(), level.k, "one row per unit of column weight");
    assert!(level.k <= level.rows(), "column weight above the row count");
    let mut hash = prefix.clone();
    hash.update(&column_input(sd, level.t, zeta as u64));
    let mut reader = hash.finalize_xof();
    let mask = (1_u32 << level.rows_log2) - 1;
    // 32 words per squeeze: k_t <= 18 distinct rows almost always fit in the first one.
    let mut buffer = [0_u8; 64];
    let mut filled = 0;
    while filled < out.len() {
        reader.read(&mut buffer);
        for word in buffer.as_chunks::<2>().0 {
            let row = u32::from(u16::from_le_bytes(*word)) & mask;
            if !out[..filled].contains(&row) {
                out[filled] = row;
                filled += 1;
                if filled == out.len() {
                    break;
                }
            }
        }
    }
}

/// The rows of column `zeta` of `level` (a fresh prefix state; [`derive_matrix`] shares one).
#[must_use]
pub fn derive_column(sd: &[u8; SEED_BYTES], level: &Level, zeta: usize) -> Vec<u32> {
    let mut out = vec![0_u32; level.k];
    ColumnHasher::new().column_into(sd, level, zeta, &mut out);
    out
}

/// `M_t` of `sd`, every column (derived in parallel).
#[must_use]
pub fn derive_matrix(sd: &[u8; SEED_BYTES], level: &Level) -> SparseMatrix {
    derive_columns(sd, level, level.cols)
}

/// The first `cols` columns of `M_t` of `sd` (a `m_t x cols` matrix whose columns are those
/// of [`derive_matrix`]), eight columns per permutation batch, in parallel.
///
/// # Panics
/// If `cols` exceeds the level's column count.
#[must_use]
pub fn derive_columns(sd: &[u8; SEED_BYTES], level: &Level, cols: usize) -> SparseMatrix {
    assert!(cols <= level.cols, "columns within the level");
    let hasher = ColumnHasher::new();
    let engine = agg_mldsa_simd::engine();
    let k = level.k;
    // Every entry is written below.
    let mut entries = vec![0_u32; cols * k];
    entries
        .par_chunks_mut(8 * k)
        .enumerate()
        .with_min_len(1 << 7)
        .for_each(|(batch, columns)| hasher.columns8_into(engine, sd, level, 8 * batch, columns));
    SparseMatrix {
        rows: level.rows(),
        cols,
        k,
        entries,
    }
}

/// `M_1, M_2, M_3` of `sd`.
#[must_use]
pub fn derive_matrices(sd: &[u8; SEED_BYTES]) -> BridgeMatrices {
    derive_matrices_for(&Shape::PRODUCTION, sd)
}

/// `M_1, M_2, M_3` of `sd` for any shape (the toy shape in tests): the used columns
/// `zeta < J_t^used` of each level, the only ones the bridge reads (module docs). Level 2 of
/// the production shapes thus holds `2^17` (ML-DSA-44) or `2^18` (ML-DSA-65, ML-DSA-87) of
/// its `2^19` columns.
#[must_use]
pub fn derive_matrices_for(shape: &Shape, sd: &[u8; SEED_BYTES]) -> BridgeMatrices {
    let [m1, m2, m3] = shape
        .levels
        .map(|level| derive_columns(sd, &level.seed_level(), level.used));
    BridgeMatrices { m1, m2, m3 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keccak_columns_match_the_cshake_columns() {
        let hasher = ColumnHasher::new();
        let prefix = prefix_state();
        for (s, sd) in [[0_u8; SEED_BYTES], [0xa5; SEED_BYTES], std::array::from_fn(|i| i as u8)]
            .iter()
            .enumerate()
        {
            for level in &LEVELS {
                // Every column of a slice, plus columns that need more than 136 output bytes
                // when the rows are few (forced with a tiny level below).
                for zeta in (0..4096).chain([level.cols - 1]) {
                    let (mut fast, mut slow) = (vec![0; level.k], vec![0; level.k]);
                    hasher.column_into(sd, level, zeta, &mut fast);
                    derive_column_into(&prefix, sd, level, zeta, &mut slow);
                    assert_eq!(fast, slow, "seed {s}, level {}, column {zeta}", level.t);
                }
            }
            // 100 distinct rows of 128 per column: every column reads past the first block
            // of 68 words.
            let tiny = Level {
                t: 9,
                rows_log2: 7,
                cols: 64,
                k: 100,
            };
            for zeta in 0..tiny.cols {
                let (mut fast, mut slow) = (vec![0; 100], vec![0; 100]);
                hasher.column_into(sd, &tiny, zeta, &mut fast);
                derive_column_into(&prefix, sd, &tiny, zeta, &mut slow);
                assert_eq!(fast, slow, "tiny level, column {zeta}");
            }
        }
    }

    #[test]
    fn eight_way_columns_match_single_columns_on_every_backend() {
        let hasher = ColumnHasher::new();
        let tiny = Level {
            t: 9,
            rows_log2: 7,
            cols: 64,
            k: 100,
        };
        let backends = [
            agg_mldsa_simd::Backend::Scalar,
            agg_mldsa_simd::Backend::Avx2,
            agg_mldsa_simd::Backend::Avx512,
        ];
        for engine in backends.into_iter().filter_map(Engine::new) {
            for (s, sd) in [[0_u8; SEED_BYTES], std::array::from_fn(|i| (7 * i) as u8)].iter().enumerate() {
                // Whole and partial batches, and a level whose columns all need a second block.
                for (level, first, count) in [(&LEVELS[0], 0, 8), (&LEVELS[1], 4096, 8), (&LEVELS[2], 49_144, 8), (&LEVELS[2], 16, 3), (&tiny, 8, 8)] {
                    let mut fast = vec![0; count * level.k];
                    hasher.columns8_into(&engine, sd, level, first, &mut fast);
                    for (i, column) in fast.chunks_exact(level.k).enumerate() {
                        let mut slow = vec![0; level.k];
                        hasher.column_into(sd, level, first + i, &mut slow);
                        assert_eq!(column, slow.as_slice(), "{:?}, seed {s}, level {}, column {}", engine.backend(), level.t, first + i);
                    }
                }
            }
        }
        // A prefix of the columns is the prefix of the full matrix.
        let sd = seed(0x33);
        let full = derive_matrix(&sd, &LEVELS[2]);
        let prefix = derive_columns(&sd, &LEVELS[2], 1000);
        assert_eq!(prefix.cols, 1000);
        assert_eq!(prefix.entries, full.entries[..1000 * LEVELS[2].k]);
    }

    /// The reading rule written out independently: a fresh cSHAKE256 per
    /// column, the input assembled byte by byte, one 16-bit word per read.
    fn reference_column(sd: &[u8; SEED_BYTES], level: &Level, zeta: usize) -> (Vec<u32>, bool) {
        let mut hash = CShake256::from_core(CShake256Core::new(b"agg-mldsa/v1/bridge"));
        let mut input = sd.to_vec();
        input.extend_from_slice(&level.t.to_le_bytes());
        input.extend_from_slice(&(zeta as u64).to_le_bytes());
        hash.update(&input);
        let mut reader = hash.finalize_xof();
        let mut rows = Vec::new();
        let mut skipped = false;
        while rows.len() < level.k {
            let mut word = [0_u8; 2];
            reader.read(&mut word);
            let row = u32::from(u16::from_le_bytes(word)) % level.rows() as u32;
            if rows.contains(&row) {
                skipped = true;
            } else {
                rows.push(row);
            }
        }
        (rows, skipped)
    }

    fn seed(byte: u8) -> [u8; SEED_BYTES] {
        std::array::from_fn(|i| byte.wrapping_add(i as u8))
    }

    #[test]
    fn levels_have_the_production_shapes() {
        let shape = |l: &Level| (l.t, l.rows(), l.cols, l.k);
        assert_eq!(shape(&LEVELS[0]), (1, 1 << 13, 1 << 19, 14));
        assert_eq!(shape(&LEVELS[1]), (2, 1 << 13, 1 << 19, 14));
        assert_eq!(shape(&LEVELS[2]), (3, 1 << 12, 49_152, 18));
        assert_eq!(column_input(&[9; 32], 3, 0x0102).len(), 48);
        assert_eq!(
            &column_input(&[9; 32], 3, 0x0102)[32..42],
            &[3, 0, 0, 0, 0, 0, 0, 0, 2, 1]
        );
    }

    #[test]
    fn derived_matrices_have_spec_shapes_valid_columns_and_capped_row_weights() {
        let matrices = derive_matrices(&seed(7));
        assert!(matrices.has_spec_shapes());
        assert!(matrices.row_weights_ok());
        // Level 2 holds the used columns only.
        assert_eq!(matrices.m2.cols, agg_witness::bridge::ZETA2_USED);
        for m in [&matrices.m1, &matrices.m2, &matrices.m3] {
            // The validated constructor rejects an out-of-range or repeated row.
            let _ = SparseMatrix::new(m.rows, m.cols, m.k, m.entries.clone());
        }
        assert_eq!(matrices, derive_matrices(&seed(7)), "deterministic");
    }

    #[test]
    fn columns_follow_the_spec_reading_rule_including_skips() {
        let sd = seed(0x40);
        for level in &LEVELS {
            let matrix = derive_matrix(&sd, level);
            for zeta in [0, 1, 2, level.cols / 2, level.cols - 1] {
                let (rows, _) = reference_column(&sd, level, zeta);
                assert_eq!(
                    matrix.column(zeta),
                    rows.as_slice(),
                    "level {} column {zeta}",
                    level.t
                );
                assert_eq!(derive_column(&sd, level, zeta), rows);
            }
            // A column whose word stream repeats a row before k distinct rows are drawn.
            let zeta = (0..4096)
                .find(|&z| reference_column(&sd, level, z).1)
                .expect("a repeated row within 4,096 columns");
            assert_eq!(
                matrix.column(zeta),
                reference_column(&sd, level, zeta).0.as_slice()
            );
        }
    }

    #[test]
    fn matrices_depend_on_the_seed_the_level_and_the_column() {
        let (a, b) = (seed(1), seed(2));
        let level = &LEVELS[2];
        assert_ne!(derive_matrix(&a, level), derive_matrix(&b, level));
        let mut flipped = a;
        flipped[31] ^= 0x80;
        assert_ne!(
            derive_column(&a, level, 5),
            derive_column(&flipped, level, 5)
        );
        assert_ne!(
            derive_column(&a, &LEVELS[0], 5),
            derive_column(&a, &LEVELS[1], 5)
        );
        assert_ne!(derive_column(&a, level, 5), derive_column(&a, level, 6));
    }

    #[test]
    fn toy_matrices_follow_the_same_rule_with_the_toy_row_masks() {
        let sd = seed(0x11);
        let toy = derive_matrices_for(&Shape::TOY, &sd);
        assert!(Shape::TOY.matches(&toy));
        assert!(!Shape::PRODUCTION.matches(&toy));
        for (level, m) in Shape::TOY.levels.iter().zip([&toy.m1, &toy.m2, &toy.m3]) {
            let level = level.seed_level();
            let _ = SparseMatrix::new(m.rows, m.cols, m.k, m.entries.clone());
            for zeta in [0, 1, level.cols - 1] {
                assert_eq!(
                    m.column(zeta),
                    reference_column(&sd, &level, zeta).0.as_slice()
                );
            }
        }
        assert_eq!(toy, derive_matrices_for(&Shape::TOY, &sd), "deterministic");
        assert_eq!(LEVELS[1], Shape::PRODUCTION.levels[1].seed_level());
    }

    /// Column 0 of each level for `sd = 0^32`, cross-checked against an independent
    /// pure-Python cSHAKE256 (SP 800-185, its Keccak core checked against `hashlib`'s
    /// SHAKE256).
    #[test]
    fn known_answer_column_zero_of_the_zero_seed() {
        let expected: [&[u32]; 3] = [
            &[
                4516, 2580, 7365, 5576, 2775, 600, 2684, 2272, 932, 441, 2024, 3960, 2214, 3567,
            ],
            &[
                5512, 6607, 8140, 6871, 5828, 3468, 1342, 7690, 8055, 6394, 4124, 2618, 2409, 1437,
            ],
            &[
                3302, 2338, 1930, 1108, 2917, 2227, 2349, 1568, 1612, 2738, 677, 2458, 3679, 3815,
                1722, 2430, 2369, 143,
            ],
        ];
        for (level, rows) in LEVELS.iter().zip(expected) {
            assert_eq!(
                derive_column(&[0; SEED_BYTES], level, 0),
                rows,
                "level {}",
                level.t
            );
        }
    }
}
