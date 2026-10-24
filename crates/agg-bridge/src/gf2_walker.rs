//! The GF(2) side of the bridge for one segment (docs/bridge.md Sections 2, 5): the forward
//! composite `Y = M_3 M_2 M_1 b'` and its transpose `M_1^T M_2^T M_3^T v` over `GF(2^128)`.
//!
//! Generic over [`Shape`]. A segment holds `lanes` lanes of `lane_len` input positions each
//! (a power of two, at most `B_1 J_1`); input column `c = p + lane_len * t` is position `p`
//! of lane `t`. Position `p` lies in level-1 block `beta = p / J_1` at in-block index
//! `zeta_1 = p mod J_1`; when `lane_len < J_1` there is one block whose positions
//! `[lane_len, J_1)` are public zeros. The level-2 input of a lane is `zeta_2 = rho_1 + m_1
//! beta < blocks * m_1`, the level-3 input `zeta_3 = t + lanes rho_2`. For the production
//! shape and `lane_len = 2^{10 + d_seg}` this is the slot layout of docs/bridge.md Section 2
//! (`p = j + 256 rho + 1024 i_seg`) and the main-domain layout of `agg_witness::bridge`.
//!
//! [`composite`] is the forward XOR walk (what the honest prover publishes as `Y_sigma`),
//! [`transpose_into`] the factored transpose that the Flock bridge slot's `LincheckCircuit`
//! uses (characteristic 2: the GF(2) and `GF(2^128)` products agree), and
//! [`dense_rows`] / [`composite_nonzeros`] materialize the composite for tests and cost
//! reports only (it is dense: about half of its `m_3 * lanes * lane_len` entries are 1).

use agg_witness::bridge::{BridgeMatrices, SparseMatrix};
use flock_field::F128;
use rayon::prelude::*;

use crate::shape::Shape;

/// The input geometry of one segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentLayout {
    /// The bridge shape.
    pub shape: Shape,
    /// Input positions per lane (a power of two, `64 <= lane_len <= B_1 J_1`).
    pub lane_len: usize,
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
        Self { shape, lane_len }
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

    /// Nonzeros of the factored representation `M_1, M_2, M_3` restricted to the segment:
    /// `k_1` per input column, `k_2` per level-2 input and lane, `k_3` per level-3 input.
    #[must_use]
    pub fn factored_nonzeros(&self) -> u64 {
        let [l1, l2, l3] = &self.shape.levels;
        (l1.k * self.input_bits() + l2.k * self.lanes() * self.zeta2_len() + l3.k * l3.cols) as u64
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
                    for &r in m1.column(64 * w + z) {
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
    let mut y2 = vec![0_u8; mats.m2.rows];
    for (beta, block) in y1.iter().enumerate() {
        for (rho1, &mask) in block.iter().enumerate() {
            if mask != 0 {
                for &r in mats.m2.column(rho1 + rows1 * beta) {
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
#[inline]
fn gather(m: &SparseMatrix, col: usize, v: impl Fn(usize) -> F128) -> F128 {
    m.column(col)
        .iter()
        .fold(F128::ZERO, |acc, &r| acc + v(r as usize))
}

/// `out[c] = (M^T v3)[c]` for every input column `c`, with `M = M_3 M_2 M_1` the segment's
/// composite: `v2 = M_3^T v3`, `v1_t = M_2^T v2(t, .)`, `out_t = M_1^T v1_t` block by block
/// (docs/bridge.md Section 5, the walker). `v3` has `m_3` entries; `out` has `input_bits`
/// entries and is overwritten.
///
/// # Panics
/// If the matrices or the slice lengths do not fit the layout.
pub fn transpose_into(
    layout: &SegmentLayout,
    mats: &BridgeMatrices,
    v3: &[F128],
    out: &mut [F128],
) {
    layout.check(mats);
    assert_eq!(v3.len(), mats.m3.rows, "one weight per output row");
    assert_eq!(out.len(), layout.input_bits(), "one entry per input column");
    let lanes = layout.lanes();
    // Level 3: v2[zeta_3], zeta_3 = t + lanes rho_2.
    let v2: Vec<F128> = (0..mats.m3.cols)
        .into_par_iter()
        .map(|zeta3| gather(&mats.m3, zeta3, |r| v3[r]))
        .collect();
    // Level 2: v1[t][zeta_2] over the zeta_2 present.
    let z2 = layout.zeta2_len();
    let v1: Vec<F128> = (0..lanes * z2)
        .into_par_iter()
        .map(|i| {
            let (t, zeta2) = (i / z2, i % z2);
            gather(&mats.m2, zeta2, |r| v2[t + lanes * r])
        })
        .collect();
    // Level 1, one chunk per (lane, block).
    let (blen, blocks, rows1) = (layout.block_len(), layout.blocks(), mats.m1.rows);
    out.par_chunks_mut(blen).enumerate().for_each(|(g, chunk)| {
        let (t, beta) = (g / blocks, g % blocks);
        let v = &v1[t * z2 + rows1 * beta..t * z2 + rows1 * (beta + 1)];
        for (zeta1, o) in chunk.iter_mut().enumerate() {
            *o = gather(&mats.m1, zeta1, |r| v[r]);
        }
    });
}

/// [`transpose_into`] into a fresh vector.
#[must_use]
pub fn transpose(layout: &SegmentLayout, mats: &BridgeMatrices, v3: &[F128]) -> Vec<F128> {
    let mut out = vec![F128::ZERO; layout.input_bits()];
    transpose_into(layout, mats, v3, &mut out);
    out
}

/// Per lane `t` and level-2 input `zeta_2`, the composite image `M_3 M_2 e_{(t, zeta_2)}` as
/// an `m_3`-bit set (`ceil(m_3 / 64)` words).
fn level2_images(layout: &SegmentLayout, mats: &BridgeMatrices) -> Vec<Vec<u64>> {
    let (lanes, z2, w3) = (
        layout.lanes(),
        layout.zeta2_len(),
        mats.m3.rows.div_ceil(64),
    );
    (0..lanes * z2)
        .into_par_iter()
        .map(|i| {
            let (t, zeta2) = (i / z2, i % z2);
            let mut image = vec![0_u64; w3];
            for &rho2 in mats.m2.column(zeta2) {
                for &r in mats.m3.column(t + lanes * rho2 as usize) {
                    image[r as usize / 64] ^= 1 << (r % 64);
                }
            }
            image
        })
        .collect()
}

/// Column `c` of the composite (`(t, beta, zeta_1)`) from the level-2 images.
fn column_from_images(
    layout: &SegmentLayout,
    mats: &BridgeMatrices,
    images: &[Vec<u64>],
    c: usize,
    out: &mut [u64],
) {
    let (t, p) = (c / layout.lane_len, c % layout.lane_len);
    let (j1, rows1, z2) = (mats.m1.cols, mats.m1.rows, layout.zeta2_len());
    let (beta, zeta1) = (p / j1, p % j1);
    out.fill(0);
    for &rho1 in mats.m1.column(zeta1) {
        for (o, x) in out
            .iter_mut()
            .zip(&images[t * z2 + rho1 as usize + rows1 * beta])
        {
            *o ^= x;
        }
    }
}

/// The composite rows `M[o, .]`, `o < m_3`, as bit sets over the `input_bits` columns
/// (lane-major packed like the input of [`composite`]). Tests and small layouts only:
/// `m_3 * input_bits / 8` bytes.
#[must_use]
pub fn dense_rows(layout: &SegmentLayout, mats: &BridgeMatrices) -> Vec<Vec<u64>> {
    layout.check(mats);
    let images = level2_images(layout, mats);
    let (m3, n) = (mats.m3.rows, layout.input_bits());
    let mut rows = vec![vec![0_u64; n / 64]; m3];
    let mut column = vec![0_u64; m3.div_ceil(64)];
    for c in 0..n {
        column_from_images(layout, mats, &images, c, &mut column);
        for (o, row) in rows.iter_mut().enumerate() {
            if (column[o / 64] >> (o % 64)) & 1 == 1 {
                row[c / 64] |= 1 << (c % 64);
            }
        }
    }
    rows
}

/// Exact number of nonzeros of the composite `M_3 M_2 M_1` over the segment's input columns
/// (the entries a materialized Flock row matrix would hold), without materializing it.
#[must_use]
pub fn composite_nonzeros(layout: &SegmentLayout, mats: &BridgeMatrices) -> u64 {
    layout.check(mats);
    let images = level2_images(layout, mats);
    let w3 = mats.m3.rows.div_ceil(64);
    (0..layout.input_bits())
        .into_par_iter()
        .fold(
            || (vec![0_u64; w3], 0_u64),
            |(mut column, acc), c| {
                column_from_images(layout, mats, &images, c, &mut column);
                let ones: u64 = column.iter().map(|w| u64::from(w.count_ones())).sum();
                (column, acc + ones)
            },
        )
        .map(|(_, acc)| acc)
        .sum()
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

    /// docs/bridge.md Section 9 item 2(b): the toy shape, `L = 2^12` input bits, against the
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
        let rows = dense_rows(&layout, &mats);
        for (o, row) in reference.iter().enumerate() {
            for (c, &want) in row.iter().enumerate() {
                assert_eq!(
                    ((rows[o][c / 64] >> (c % 64)) & 1) as u8,
                    want,
                    "({o}, {c})"
                );
            }
        }
        let nonzeros: u64 = reference.iter().flatten().map(|&x| u64::from(x)).sum();
        assert_eq!(composite_nonzeros(&layout, &mats), nonzeros);
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
    /// (linearity). Level 1 never kills a weight-one difference (Lemma 1 of docs/bridge.md
    /// Section 7: column weight `k_1 >= 1`); levels 2 and 3 kill its weight-`k_1` image with
    /// probability at most `1 / C(m_2, k_2) + 1 / C(m_3, k_3)` (Lemma 2), which is
    /// `2^-10.7` on the toy, so a handful of the 4,096 toy columns may vanish (expected 2.4).
    #[test]
    fn single_bit_flips_are_detected_unless_the_column_vanishes() {
        let layout = SegmentLayout::new(Shape::TOY, 1 << 12);
        let mats = derive_matrices_for(&Shape::TOY, &seed(3));
        let rows = dense_rows(&layout, &mats);
        let mut rng = SplitMix64::new(4);
        let bits = random_bits(&mut rng, 4096);
        let y = composite(&layout, &mats, &bits);
        let mut missed = 0;
        for c in 0..4096 {
            let mut flipped = bits.clone();
            flipped[c / 64] ^= 1 << (c % 64);
            let y2 = composite(&layout, &mats, &flipped);
            let column_weight = rows
                .iter()
                .filter(|r| (r[c / 64] >> (c % 64)) & 1 == 1)
                .count();
            assert_eq!(y != y2, column_weight > 0, "column {c}");
            // Level 1 alone: the image M_1 e_zeta has exactly k_1 ones.
            assert_eq!(mats.m1.column(c % 1024).len(), 4);
            missed += usize::from(y == y2);
        }
        assert!(missed <= 16, "{missed} vanishing toy columns");
    }

    /// The slot layout agrees with the production main-domain walk of `agg-witness` at one
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
        assert_eq!(
            layout.factored_nonzeros(),
            (14 * 6 * (1 << 16) + 14 * 6 * (1 << 13) + 18 * 49_152) as u64
        );
    }
}
