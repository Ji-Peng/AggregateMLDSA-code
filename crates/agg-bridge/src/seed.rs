//! Bridge seed to the three bridge matrices.
//!
//! Column `zeta` of `M_t` (`t` = 1, 2, 3, the level number) is a `k_t`-subset of the
//! `m_t = 2^{e_t}` rows (`e_t` = 13, 13, 12): read 16-bit little-endian words from
//! `cSHAKE256(sd || LE64(t) || LE64(zeta), *, "", "agg-mldsa/v1/bridge")`, keep the low
//! `e_t` bits of each word, skip a row already chosen for the column, and stop after `k_t`
//! distinct rows. Rows are stored in the order drawn. Prover and verifier both derive the
//! matrices from the transcript's `sd`; neither transmits them.

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
        let mask = (1_u32 << level.rows_log2) - 1;
        let mut filled = 0;
        while filled < out.len() {
            keccak::f1600(&mut state);
            let mut bytes = [0_u8; RATE];
            for (chunk, lane) in bytes.as_chunks_mut::<8>().0.iter_mut().zip(&state) {
                *chunk = lane.to_le_bytes();
            }
            for word in bytes.as_chunks::<2>().0 {
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

/// `M_t` of `sd` (columns derived in parallel).
#[must_use]
pub fn derive_matrix(sd: &[u8; SEED_BYTES], level: &Level) -> SparseMatrix {
    let hasher = ColumnHasher::new();
    let mut entries = vec![0_u32; level.cols * level.k];
    entries
        .par_chunks_mut(level.k)
        .enumerate()
        .with_min_len(1 << 10)
        .for_each(|(zeta, column)| hasher.column_into(sd, level, zeta, column));
    SparseMatrix {
        rows: level.rows(),
        cols: level.cols,
        k: level.k,
        entries,
    }
}

/// `M_1, M_2, M_3` of `sd`.
#[must_use]
pub fn derive_matrices(sd: &[u8; SEED_BYTES]) -> BridgeMatrices {
    derive_matrices_for(&Shape::PRODUCTION, sd)
}

/// `M_1, M_2, M_3` of `sd` for any shape (the toy shape in tests). Every column of every
/// level is derived, including the unused level-2 columns of the production shape, so the
/// production matrices are exactly those of the derivation rule.
#[must_use]
pub fn derive_matrices_for(shape: &Shape, sd: &[u8; SEED_BYTES]) -> BridgeMatrices {
    let [l1, l2, l3] = shape.levels.map(|level| level.seed_level());
    BridgeMatrices {
        m1: derive_matrix(sd, &l1),
        m2: derive_matrix(sd, &l2),
        m3: derive_matrix(sd, &l3),
    }
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
    fn levels_are_those_of_spec_section_1() {
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
