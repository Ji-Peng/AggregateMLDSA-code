//! The GF(2) side of the bridge for one segment: the forward
//! composite `Y = M_3 M_2 M_1 b'` and, for tests, its transpose `M_1^T M_2^T M_3^T v` over
//! `GF(2^128)`.
//!
//! Generic over [`Shape`]. A segment holds `lanes` lanes of `lane_len` input positions each
//! (a power of two, at most `B_1 J_1`); input column `c = p + lane_len * t` is position `p`
//! of lane `t`. Position `p` lies in level-1 block `beta = p / J_1` at in-block index
//! `zeta_1 = p mod J_1`; when `lane_len < J_1` there is one block whose positions
//! `[lane_len, J_1)` are public zeros. The level-2 input of a lane is `zeta_2 = rho_1 + m_1
//! beta < blocks * m_1`, the level-3 input `zeta_3 = t + lanes rho_2`. For the production
//! shape and `lane_len = 2^{10 + d_seg}` this is the main-domain layout of
//! `agg_witness::bridge`.
//!
//! A layout may also be one part of a segment split into parts of `lane_len` positions
//! ([`SegmentLayout::part`], Flock sub-batches): local position `p` is then segment position
//! `offset + p`, so its level-1 block and in-block index, and its level-2 input, are those
//! of the segment position. The composite is linear, so the parts' outputs XOR to the
//! segment's.
//!
//! [`composite`] is the forward XOR walk (what the honest prover publishes as `Y_sigma`);
//! the parity chain (`crate::parity_chain`) proves the same map by sum-check.

use agg_witness::bridge::{BridgeMatrices, SparseMatrix};
#[cfg(test)]
use flock_field::F128;
use rayon::prelude::*;

use crate::shape::Shape;
use agg_witness::params::{ParamSet, Set};

/// Padded positions `P_pad` of one signature of `set` (1,024 for ML-DSA-44, 2,048 for
/// ML-DSA-65 and ML-DSA-87).
#[must_use]
pub const fn positions_pad(set: Set) -> usize {
    agg_witness::params::dispatch_set!(set, |P| P::P_PAD)
}

/// The input geometry of one segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentLayout {
    /// The bridge shape.
    pub shape: Shape,
    /// Input positions per lane (a power of two, `64 <= lane_len <= B_1 J_1`).
    pub lane_len: usize,
    /// Position of the layout's first input within the segment's lane: `0` for a whole
    /// segment, `index * lane_len` for part `index` of a segment split into parts of
    /// `lane_len` positions (sub-batches). Local position `p` is
    /// segment position `offset + p`.
    pub offset: usize,
}

impl SegmentLayout {
    /// A segment of `lane_len` positions per lane.
    ///
    /// # Panics
    /// If `lane_len` is not a power of two in `[64, B_1 J_1]` or the shape is invalid.
    #[must_use]
    pub fn new(shape: Shape, lane_len: usize) -> Self {
        shape.validate().expect("a valid bridge shape");
        assert!(
            lane_len.is_power_of_two() && lane_len >= 64,
            "lane length is a power of two of at least 64"
        );
        assert!(
            lane_len <= shape.segment_input_bits(),
            "a segment holds at most B_1 J_1 positions per lane"
        );
        Self {
            shape,
            lane_len,
            offset: 0,
        }
    }

    /// Part `index` of a segment split into parts of `lane_len` positions per lane: local
    /// position `p` is segment position `index * lane_len + p`. The parts of a segment tile
    /// it, and the composite is linear, so the segment's `Y` is the XOR of the parts' `Y`.
    ///
    /// # Panics
    /// As [`Self::new`], or if the part lies outside the segment.
    #[must_use]
    pub fn part(shape: Shape, lane_len: usize, index: usize) -> Self {
        let whole = Self::new(shape, lane_len);
        let offset = index
            .checked_mul(lane_len)
            .filter(|&o| o + lane_len <= shape.segment_input_bits())
            .expect("the part lies inside the segment");
        Self { offset, ..whole }
    }

    /// The production layout of part `index` of a segment split into parts of
    /// `2^{d_part}` records (`lane_len = 2^{10 + d_part}`).
    ///
    /// # Panics
    /// If `d_part > 13` or the part lies outside the segment.
    #[must_use]
    pub fn production_part(d_part: u32, index: usize) -> Self {
        Self::production_part_for(Set::S44, d_part, index)
    }

    /// The production layout of part `index` of a segment of parameter set `set` split into
    /// parts of `2^{d_part}` records (`lane_len = P_pad 2^{d_part}`, `P_pad` the padded
    /// positions of a signature: record `i` of the part starts at position `P_pad i`).
    ///
    /// # Panics
    /// If `d_part > 13` or the part lies outside the segment.
    #[must_use]
    pub fn production_part_for(set: Set, d_part: u32, index: usize) -> Self {
        assert!(d_part <= 13, "a segment has at most 2^13 records");
        Self::part(Shape::production(set), positions_pad(set) << d_part, index)
    }

    /// Segment-level index of the layout's first level-1 block.
    #[must_use]
    pub fn first_block(&self) -> usize {
        self.offset / self.shape.levels[0].cols
    }

    /// In-block index `zeta_1` of local position 0 (nonzero only for a part shorter than a
    /// level-1 block).
    #[must_use]
    pub fn zeta1_base(&self) -> usize {
        self.offset % self.shape.levels[0].cols
    }

    /// First level-2 input `zeta_2` of the layout (`m_1 * first_block`).
    #[must_use]
    pub fn zeta2_base(&self) -> usize {
        self.first_block() * self.shape.levels[0].rows
    }

    /// The production layout of a segment of `2^{d_seg}` records (`lane_len = 2^{10 + d_seg}`).
    ///
    /// # Panics
    /// If `d_seg > 13`.
    #[must_use]
    pub fn production(d_seg: u32) -> Self {
        assert!(d_seg <= 13, "a segment has at most 2^13 records");
        Self::new(Shape::PRODUCTION, 1 << (10 + d_seg))
    }

    /// Lanes.
    #[must_use]
    pub const fn lanes(&self) -> usize {
        self.shape.lanes
    }

    /// Input columns `lanes * lane_len`.
    #[must_use]
    pub const fn input_bits(&self) -> usize {
        self.shape.lanes * self.lane_len
    }

    /// Level-1 blocks per lane present in the segment.
    #[must_use]
    pub fn blocks(&self) -> usize {
        self.lane_len.div_ceil(self.shape.levels[0].cols)
    }

    /// Positions per level-1 block present (`min(lane_len, J_1)`).
    #[must_use]
    pub fn block_len(&self) -> usize {
        self.lane_len.min(self.shape.levels[0].cols)
    }

    /// Level-2 inputs per lane present (`blocks * m_1`).
    #[must_use]
    pub fn zeta2_len(&self) -> usize {
        self.blocks() * self.shape.levels[0].rows
    }

    /// Output bits `m_3`.
    #[must_use]
    pub const fn output_bits(&self) -> usize {
        self.shape.levels[2].rows
    }

    /// Input column of position `p` of lane `t`.
    #[must_use]
    pub const fn column(&self, t: usize, p: usize) -> usize {
        p + self.lane_len * t
    }

    /// The matrices fit this layout (shape match).
    fn check(&self, mats: &BridgeMatrices) {
        assert!(
            self.shape.matches(mats),
            "bridge matrices of another shape than the layout's"
        );
    }
}

#[cfg(test)]
#[inline]
fn bit(words: &[u64], c: usize) -> u8 {
    ((words[c / 64] >> (c % 64)) & 1) as u8
}

/// Level-1 parities of every block as lane masks: `y1[beta][rho_1]` bit `t`.
fn level1_masks(layout: &SegmentLayout, m1: &SparseMatrix, bits: &[u64]) -> Vec<Vec<u8>> {
    let lanes = layout.lanes();
    let (blen, j1, rows) = (layout.block_len(), m1.cols, m1.rows);
    let zeta1_base = layout.zeta1_base();
    (0..layout.blocks())
        .into_par_iter()
        .map(|beta| {
            let mut out = vec![0_u8; rows];
            let base = beta * j1;
            // 64 positions at a time (block lengths are multiples of 64).
            for w in 0..blen / 64 {
                let mut words = [0_u64; 8];
                for (t, word) in words.iter_mut().enumerate().take(lanes) {
                    *word = bits[(layout.column(t, base) >> 6) + w];
                }
                let any = words[..lanes].iter().fold(0, |acc, x| acc | x);
                if any == 0 {
                    continue;
                }
                let mut rest = any;
                while rest != 0 {
                    let z = rest.trailing_zeros() as usize;
                    rest &= rest - 1;
                    let mask = words[..lanes]
                        .iter()
                        .enumerate()
                        .fold(0_u8, |m, (t, x)| m | ((((x >> z) & 1) as u8) << t));
                    for &r in m1.column(zeta1_base + 64 * w + z) {
                        out[r as usize] ^= mask;
                    }
                }
            }
            out
        })
        .collect()
}

/// `Y = M_3 M_2 M_1 b'` over GF(2), one output bit (0 or 1) per byte, for the segment's input
/// bits `bits` (lane-major packed: column `c` is bit `c mod 64` of word `c / 64`; length
/// `input_bits / 64`).
///
/// # Panics
/// If the matrices or `bits` do not fit the layout.
#[must_use]
pub fn composite(layout: &SegmentLayout, mats: &BridgeMatrices, bits: &[u64]) -> Vec<u8> {
    layout.check(mats);
    assert_eq!(bits.len(), layout.input_bits() / 64, "packed input length");
    let lanes = layout.lanes();
    let rows1 = mats.m1.rows;
    let y1 = level1_masks(layout, &mats.m1, bits);
    let first = layout.first_block();
    let mut y2 = vec![0_u8; mats.m2.rows];
    for (beta, block) in y1.iter().enumerate() {
        for (rho1, &mask) in block.iter().enumerate() {
            if mask != 0 {
                for &r in mats.m2.column(rho1 + rows1 * (first + beta)) {
                    y2[r as usize] ^= mask;
                }
            }
        }
    }
    let mut y = vec![0_u8; mats.m3.rows];
    for (rho2, &mask) in y2.iter().enumerate() {
        for t in 0..lanes {
            if (mask >> t) & 1 == 1 {
                for &r in mats.m3.column(t + lanes * rho2) {
                    y[r as usize] ^= 1;
                }
            }
        }
    }
    y
}

/// `sum_{r in S(col)} v(r)` over the rows of column `col`.
#[cfg(test)]
fn gather(m: &SparseMatrix, col: usize, v: impl Fn(usize) -> F128) -> F128 {
    m.column(col)
        .iter()
        .fold(F128::ZERO, |acc, &r| acc + v(r as usize))
}

/// `(M^T v3)[c]` for every input column `c`, with `M = M_3 M_2 M_1` the segment's composite:
/// `v2 = M_3^T v3`, `v1_t = M_2^T v2(t, .)`, `out_t = M_1^T v1_t` block by block. `v3` has
/// `m_3` entries; the result has `input_bits` (characteristic 2: the GF(2) and `GF(2^128)`
/// products agree). The brute-force reference of the parity chain's tests.
///
/// # Panics
/// If the matrices or `v3` do not fit the layout.
#[cfg(test)]
pub(crate) fn transpose(layout: &SegmentLayout, mats: &BridgeMatrices, v3: &[F128]) -> Vec<F128> {
    layout.check(mats);
    assert_eq!(v3.len(), mats.m3.rows, "one weight per output row");
    let lanes = layout.lanes();
    // Level 3: v2[zeta_3], zeta_3 = t + lanes rho_2.
    let v2: Vec<F128> = (0..mats.m3.cols)
        .into_par_iter()
        .map(|zeta3| gather(&mats.m3, zeta3, |r| v3[r]))
        .collect();
    // Level 2: v1[t][zeta_2] over the zeta_2 present.
    let (z2, z2_base) = (layout.zeta2_len(), layout.zeta2_base());
    let v1: Vec<F128> = (0..lanes * z2)
        .into_par_iter()
        .map(|i| {
            let (t, zeta2) = (i / z2, i % z2);
            gather(&mats.m2, z2_base + zeta2, |r| v2[t + lanes * r])
        })
        .collect();
    // Level 1, one chunk per (lane, block).
    let (blen, blocks, rows1) = (layout.block_len(), layout.blocks(), mats.m1.rows);
    let zeta1_base = layout.zeta1_base();
    let mut out = vec![F128::ZERO; layout.input_bits()];
    out.par_chunks_mut(blen).enumerate().for_each(|(g, chunk)| {
        let (t, beta) = (g / blocks, g % blocks);
        let v = &v1[t * z2 + rows1 * beta..t * z2 + rows1 * (beta + 1)];
        for (zeta1, o) in chunk.iter_mut().enumerate() {
            *o = gather(&mats.m1, zeta1_base + zeta1, |r| v[r]);
        }
    });
    out
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::seed::derive_matrices_for;
    use agg_witness::gen::SplitMix64;

    fn random_f128(rng: &mut SplitMix64) -> F128 {
        F128::new(rng.next_u64(), rng.next_u64())
    }

    fn random_bits(rng: &mut SplitMix64, n: usize) -> Vec<u64> {
        (0..n / 64).map(|_| rng.next_u64()).collect()
    }

    fn seed(byte: u8) -> [u8; 32] {
        std::array::from_fn(|i| byte ^ (3 * i as u8))
    }

    /// Dense GF(2) matrices built directly from the column lists, multiplied densely.
    fn dense_product_reference(layout: &SegmentLayout, mats: &BridgeMatrices) -> Vec<Vec<u8>> {
        let lanes = layout.lanes();
        let dense = |m: &SparseMatrix| -> Vec<Vec<u8>> {
            let mut d = vec![vec![0_u8; m.cols]; m.rows];
            for c in 0..m.cols {
                for &r in m.column(c) {
                    d[r as usize][c] = 1;
                }
            }
            d
        };
        let (d1, d2, d3) = (dense(&mats.m1), dense(&mats.m2), dense(&mats.m3));
        let (m1, m2, m3) = (mats.m1.rows, mats.m2.rows, mats.m3.rows);
        let (j1, n) = (mats.m1.cols, layout.input_bits());
        // Composite column by column: e_c -> level 1 -> level 2 -> level 3.
        let mut rows = vec![vec![0_u8; n]; m3];
        for c in 0..n {
            let (t, p) = (c / layout.lane_len, c % layout.lane_len);
            let (beta, zeta1) = (p / j1, p % j1);
            let y1: Vec<u8> = (0..m1).map(|r| d1[r][zeta1]).collect();
            let y2: Vec<u8> = (0..m2)
                .map(|r| (0..m1).fold(0, |acc, q| acc ^ (d2[r][q + m1 * beta] & y1[q])))
                .collect();
            for (o, row) in rows.iter_mut().enumerate() {
                row[c] = (0..m2).fold(0, |acc, q| acc ^ (d3[o][t + lanes * q] & y2[q]));
            }
        }
        rows
    }

    /// The toy shape, `L = 2^12` input bits, against the
    /// dense composite (16 rows by 4,096 columns) and its transpose on a random vector.
    #[test]
    fn toy_composite_and_transpose_match_the_dense_product() {
        let layout = SegmentLayout::new(Shape::TOY, 1 << 12);
        assert_eq!(
            (layout.input_bits(), layout.blocks(), layout.zeta2_len()),
            (4096, 4, 256)
        );
        let mats = derive_matrices_for(&Shape::TOY, &seed(5));
        let reference = dense_product_reference(&layout, &mats);
        let mut rng = SplitMix64::new(9);
        for _ in 0..8 {
            let bits = random_bits(&mut rng, 4096);
            let want: Vec<u8> = reference
                .iter()
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .fold(0, |acc, (c, &m)| acc ^ (m & bit(&bits, c)))
                })
                .collect();
            assert_eq!(composite(&layout, &mats, &bits), want);
            let v3: Vec<F128> = (0..16).map(|_| random_f128(&mut rng)).collect();
            let got = transpose(&layout, &mats, &v3);
            for c in 0..4096 {
                let want = (0..16)
                    .filter(|&o| reference[o][c] == 1)
                    .fold(F128::ZERO, |acc, o| acc + v3[o]);
                assert_eq!(got[c], want, "column {c}");
            }
        }
    }

    /// The transpose is the adjoint of the forward walk on every layout: `<M^T v, b> =
    /// <v, M b>` with the GF(2) bits embedded in `GF(2^128)`.
    fn check_adjoint(layout: &SegmentLayout, mats: &BridgeMatrices, seed: u64) {
        let mut rng = SplitMix64::new(seed);
        let v3: Vec<F128> = (0..layout.output_bits())
            .map(|_| random_f128(&mut rng))
            .collect();
        let mt = transpose(layout, mats, &v3);
        for _ in 0..3 {
            let bits = random_bits(&mut rng, layout.input_bits());
            let y = composite(layout, mats, &bits);
            let lhs = (0..layout.input_bits())
                .filter(|&c| bit(&bits, c) == 1)
                .fold(F128::ZERO, |acc, c| acc + mt[c]);
            let rhs = (0..y.len())
                .filter(|&o| y[o] == 1)
                .fold(F128::ZERO, |acc, o| acc + v3[o]);
            assert_eq!(lhs, rhs);
        }
    }

    #[test]
    fn transpose_is_the_adjoint_of_the_composite() {
        let toy = derive_matrices_for(&Shape::TOY, &seed(1));
        for len in [64, 256, 1024, 4096] {
            check_adjoint(&SegmentLayout::new(Shape::TOY, len), &toy, len as u64);
        }
        let production = derive_matrices_for(&Shape::PRODUCTION, &seed(2));
        check_adjoint(&SegmentLayout::production(3), &production, 3);
    }

    /// A single flipped input bit changes `Y` exactly when its composite column is nonzero
    /// (linearity). Level 1 never kills a weight-one difference (column weight `k_1 >= 1`);
    /// levels 2 and 3 kill its weight-`k_1` image with probability at most
    /// `1 / C(m_2, k_2) + 1 / C(m_3, k_3)`, which is
    /// `2^-10.7` on the toy, so a handful of the 4,096 toy columns may vanish (expected 2.4).
    #[test]
    fn single_bit_flips_are_detected_unless_the_column_vanishes() {
        let layout = SegmentLayout::new(Shape::TOY, 1 << 12);
        let mats = derive_matrices_for(&Shape::TOY, &seed(3));
        let rows = dense_product_reference(&layout, &mats);
        let mut rng = SplitMix64::new(4);
        let bits = random_bits(&mut rng, 4096);
        let y = composite(&layout, &mats, &bits);
        let mut missed = 0;
        for c in 0..4096 {
            let mut flipped = bits.clone();
            flipped[c / 64] ^= 1 << (c % 64);
            let y2 = composite(&layout, &mats, &flipped);
            let column_weight = rows.iter().filter(|r| r[c] == 1).count();
            assert_eq!(y != y2, column_weight > 0, "column {c}");
            // Level 1 alone: the image M_1 e_zeta has exactly k_1 ones.
            assert_eq!(mats.m1.column(c % 1024).len(), 4);
            missed += usize::from(y == y2);
        }
        assert!(missed <= 16, "{missed} vanishing toy columns");
    }

    /// Pack lane `t`, positions `[from, from + len)` of a lane-major bit vector with lanes of
    /// `lane_len` positions into a lane-major vector with lanes of `len` positions.
    fn slice_lanes(
        bits: &[u64],
        lanes: usize,
        lane_len: usize,
        from: usize,
        len: usize,
    ) -> Vec<u64> {
        let mut out = vec![0_u64; lanes * len / 64];
        for t in 0..lanes {
            for p in 0..len {
                let c = t * lane_len + from + p;
                if bit(bits, c) == 1 {
                    let d = t * len + p;
                    out[d / 64] |= 1 << (d % 64);
                }
            }
        }
        out
    }

    /// Parts tile a segment: their composites XOR to the segment's composite and their
    /// transposes are the segment transpose's columns (toy shape at every part length, and
    /// production parts shorter than, equal to and longer than a level-1 block).
    #[test]
    fn parts_xor_to_the_segment_and_transpose_to_its_columns() {
        let check = |whole: SegmentLayout, part_len: usize, mats: &BridgeMatrices, seed: u64| {
            let mut rng = SplitMix64::new(seed);
            let lanes = whole.lanes();
            let bits = random_bits(&mut rng, whole.input_bits());
            let y = composite(&whole, mats, &bits);
            let v3: Vec<F128> = (0..whole.output_bits())
                .map(|_| random_f128(&mut rng))
                .collect();
            let full_t = transpose(&whole, mats, &v3);
            let parts = whole.lane_len / part_len;
            let mut acc = vec![0_u8; y.len()];
            for index in 0..parts {
                let part = SegmentLayout::part(whole.shape, part_len, index);
                assert_eq!(part.offset, index * part_len);
                let local = slice_lanes(&bits, lanes, whole.lane_len, index * part_len, part_len);
                for (a, b) in acc.iter_mut().zip(composite(&part, mats, &local)) {
                    *a ^= b;
                }
                let part_t = transpose(&part, mats, &v3);
                for t in 0..lanes {
                    for p in (0..part_len).step_by(7) {
                        assert_eq!(
                            part_t[part.column(t, p)],
                            full_t[whole.column(t, index * part_len + p)],
                            "part {index} lane {t} position {p}"
                        );
                    }
                }
            }
            assert_eq!(acc, y, "parts of {part_len} positions");
        };
        let toy = derive_matrices_for(&Shape::TOY, &seed(11));
        let whole = SegmentLayout::new(Shape::TOY, 1 << 12);
        for part_len in [1 << 12, 1 << 11, 1 << 10, 1 << 8, 1 << 6] {
            check(whole, part_len, &toy, part_len as u64);
        }
        let production = derive_matrices_for(&Shape::PRODUCTION, &seed(12));
        // d_seg = 6 in parts of 8 records (inside one level-1 block, zeta_1 offsets).
        check(SegmentLayout::production(6), 1 << 13, &production, 21);
        // d_seg = 10 in parts of 512 records (one level-1 block each, block offsets).
        check(SegmentLayout::production(10), 1 << 19, &production, 22);
        assert_eq!(
            SegmentLayout::production_part(9, 1),
            SegmentLayout::part(Shape::PRODUCTION, 1 << 19, 1)
        );
        let p = SegmentLayout::production_part(3, 70);
        assert_eq!(
            (p.first_block(), p.zeta1_base(), p.zeta2_base()),
            (1, 6 << 13, 1 << 13)
        );
        assert_eq!(SegmentLayout::production(5).offset, 0);
        assert!(std::panic::catch_unwind(|| SegmentLayout::production_part(13, 1)).is_err());
    }

    /// The production layout agrees with the main-domain walk of `agg-witness` at one
    /// segment (`d_seg = 6`, 64 records): column `p + 2^16 t` is `b_t[x = p]`.
    #[test]
    fn production_layout_agrees_with_the_main_domain_walk() {
        use agg_witness::bridge::gf2_composite;
        use agg_witness::padding::Geometry;
        let geom = Geometry::new(64).unwrap();
        let layout = SegmentLayout::production(6);
        assert_eq!(layout.lane_len, geom.main_len());
        let mats = derive_matrices_for(&Shape::PRODUCTION, &seed(7));
        let mut rng = SplitMix64::new(8);
        let bits = random_bits(&mut rng, layout.input_bits());
        let lanes: Vec<Vec<i8>> = (0..6)
            .map(|t| {
                (0..layout.lane_len)
                    .map(|p| bit(&bits, layout.column(t, p)) as i8)
                    .collect()
            })
            .collect();
        let b: [&[i8]; 6] = std::array::from_fn(|t| lanes[t].as_slice());
        assert_eq!(
            composite(&layout, &mats, &bits),
            gf2_composite(&geom, &b, &mats)
        );
    }
}
