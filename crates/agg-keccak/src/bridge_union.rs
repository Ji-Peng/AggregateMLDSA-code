//! The two-type bridge union: the bridge instance of a tied part factored the way the
//! composite `M = M_3 M_2 M_1` is factored (docs/bridge.md Section 5b, deviation K13;
//! docs/keccak.md Sections 10 and 11).
//!
//! The one-row bridge slot of [`crate::bridge_slot`] proves `Y = M b'` with one block of
//! `2^(13 + d)` columns, so its walker, its eq and comb vectors and the verifier's lincheck
//! discharge are all linear in the part (`7 x 10^8` walker entries and 1 GiB vectors at a
//! full segment). Here the same relation is split over two Flock types of one union:
//!
//! - **level 1** ([`Level1Layout`], [`Level1Slot`]): one row per (lane `t`, level-1 block
//!   `beta`) of the part, row index `j = beta + blocks * t` (rows of lanes 6, 7 are dummy).
//!   Columns `[0, L)` hold the block's input bits (`L = block_len`, local in-block index
//!   `zeta`, `M_1` column `zeta1_base + zeta`), the constant is column `L`, and the level-1
//!   parities `y1 = M_1 b'(t, beta)` are columns `y1_base + rho_1`, `y1_base = L + max(m_1,
//!   128)` (an aligned region of `m_1 / 128` words). `A_0` is `M_1` itself: rows `z_c * one
//!   = z_c` on the inputs (all `L` columns, the same in every row: padded positions are tied
//!   to the sponge's zeros, not forced by rows), `one * one = one`, `(sum_zeta M_1[rho_1,
//!   zeta1_base + zeta] z_zeta) * one = z_{y1(rho_1)}`, empty rows elsewhere. Flock's
//!   `A = I (x) A_0` is then exactly the per-block level 1.
//! - **levels 2-3** ([`TopLayout`], [`TopSlot`]): one live row whose inputs are the level-1
//!   parities of all rows concatenated in row order, column `c(t, beta, rho_1) = rho_1 + m_1
//!   beta + G t` (`G = blocks * m_1`), the constant at `lanes * G`, and the output columns
//!   `y_o = y_base + o`, `y_base = lanes G + max(m_3, 128)` (an aligned region of `m_3 / 128`
//!   words), `o < m_3`, with rows `(sum_c (M_3 M_2)[o, c] z_c + Y_o one) * one = z_{y_o}` (`Y`
//!   a coefficient of the relation as in K8), where `(M_3 M_2)[o, (t, beta, rho_1)] =
//!   sum_{rho_2} M_3[o, t + lanes rho_2] M_2[rho_2, rho_1 + m_1 (first_block + beta)]`.
//!
//! The parities are a function of `sd`, so the union's commitment is computed after `sd`
//! (inside the part's proof, event 6), not at event 2 like the sponge instance's; the tie
//! challenges are derived after both commitments ([`crate::tied`]). The level-1 inputs are
//! tied to the sponge's input words and the levels-2-3 inputs to the level-1 parities by
//! evaluation claims ([`BridgeUnion::input_point`], [`BridgeUnion::y1_point`],
//! [`BridgeUnion::top_point`]). A row with `C = I` only DEFINES its output column, so the
//! output columns `y_o` are pinned to zero by one more claim ([`BridgeUnion::output_point`],
//! value 0), which turns the definitions into `M_3 M_2 y1 = Y`. (The one-row slot of K8 did
//! not need it: its block, the `y_o` columns included, was committed at event 2, before
//! `sd` and `Y` existed; a block committed after `sd` could otherwise set `z_y = M_3 M_2 y1 +
//! Y` for any `Y`.)
//!
//! Layouts are generic over [`Shape`] (the toy shape has one lane and `m_1 = 64`); the tie
//! points need the production shape's six lanes and `m_1` a multiple of 128.
//!
//! **Spans** (tied-part proof version 3, docs/bridge.md Section 5c, deviation K16). A union
//! may cover `2^span_log` segments that share the part geometry (`d`, part index; in mode A
//! whole segments of `2^13` records): the level-1 row of segment `sigma`, lane `t`, block
//! `beta` is `sigma + 2^span_log (beta + blocks t)` and the levels-2-3 row of segment `sigma`
//! is row `sigma`. Every row of both types is live (`2^span_log * lanes * blocks` and
//! `2^span_log`); segments without records are all-zero rows (constant 1). The tied part
//! proves the relations with a zero coefficient (`Y = 0` in [`TopSlot`]), so the composite rows
//! say `z_{y_o} = (M_3 M_2 z_in)_o`, and pins the output columns of row `sigma` to the public
//! `Y_sigma` with the output claim (its value is the multilinear extension of the `Y` table,
//! computed by the verifier).

use agg_bridge::gf2_walker::SegmentLayout;
use agg_bridge::shape::Shape;
use agg_witness::bridge::{BridgeMatrices, SparseMatrix};
use flock_core::{
    lincheck::LincheckCircuit,
    r1cs::SparseBinaryMatrix,
    schedule::{TableClass, TableType},
};
use flock_field::F128;
use rayon::prelude::*;
use sha3::{Digest, Sha3_256};

use crate::{
    bridge_slot::pack_y,
    flock_union::UnionGeometry,
    prover::Profile,
    witness::{pack_rows, SlotBuffers},
};

/// Digest label of the level-1 relation.
pub const LEVEL1_LABEL: &[u8] = b"agg-keccak/v2/bridge-level1";
/// Digest label of the levels-2-3 relation.
pub const TOP_LABEL: &[u8] = b"agg-keccak/v2/bridge-top";
/// Smallest union address space Flock's Ligerito configs accept (`MIN_DENSE_M`).
const MIN_M: usize = 22;

fn log2_ceil(x: usize) -> usize {
    x.next_power_of_two().trailing_zeros() as usize
}

#[inline]
fn bit(v: &[u64], c: usize) -> u64 {
    (v[c / 64] >> (c % 64)) & 1
}

/// `sum_{r in S(col)} v(r)` over the rows of column `col`.
#[inline]
fn gather(m: &SparseMatrix, col: usize, v: impl Fn(usize) -> F128) -> F128 {
    m.column(col)
        .iter()
        .fold(F128::ZERO, |acc, &r| acc + v(r as usize))
}

fn empty_stub() -> SparseBinaryMatrix {
    SparseBinaryMatrix::new(0, 0, Vec::new())
}

fn put_shape(h: &mut Sha3_256, shape: &Shape) {
    let put = |h: &mut Sha3_256, v: usize| h.update((v as u64).to_le_bytes());
    for level in &shape.levels {
        put(h, level.t as usize);
        put(h, level.cols);
        put(h, level.used);
        put(h, level.rows);
        put(h, level.k);
    }
    put(h, shape.lanes);
    put(h, shape.blocks1);
}

/// Column layout of the level-1 type (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Level1Layout {
    /// The part's input geometry (shape, lanes, `lane_len`, offset).
    pub segment: SegmentLayout,
}

impl Level1Layout {
    /// The level-1 layout of a part.
    ///
    /// # Panics
    /// If `m_1` is not a multiple of 64 or the block length is not a multiple of
    /// `max(m_1, 128)` (the parity region must be an aligned word region).
    #[must_use]
    pub fn new(segment: SegmentLayout) -> Self {
        let lay = Self { segment };
        assert!(
            lay.m1().is_multiple_of(64) && lay.m1().is_power_of_two(),
            "m_1 is a power of two of at least 64"
        );
        assert!(
            lay.block_len().is_multiple_of(lay.m1().max(128)),
            "block length is a multiple of max(m_1, 128)"
        );
        assert!(lay.lanes() <= 8, "at most 8 lanes (one byte mask)");
        lay
    }

    /// The shape.
    #[must_use]
    pub fn shape(&self) -> &Shape {
        &self.segment.shape
    }

    /// Lanes.
    #[must_use]
    pub fn lanes(&self) -> usize {
        self.segment.lanes()
    }

    /// Level-1 blocks per lane in the part.
    #[must_use]
    pub fn blocks(&self) -> usize {
        self.segment.blocks()
    }

    /// `log2` of the blocks.
    #[must_use]
    pub fn blocks_log(&self) -> usize {
        log2_ceil(self.blocks())
    }

    /// Bits of the lane in a row index.
    #[must_use]
    pub fn lane_log(&self) -> usize {
        log2_ceil(self.lanes())
    }

    /// Live rows `lanes * blocks`.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.lanes() * self.blocks()
    }

    /// Row-index bits the rows need (`lane_log + blocks_log`).
    #[must_use]
    pub fn rows_log(&self) -> usize {
        self.lane_log() + self.blocks_log()
    }

    /// Row of lane `t`, block `beta`.
    #[must_use]
    pub fn row(&self, t: usize, beta: usize) -> usize {
        beta + self.blocks() * t
    }

    /// Positions per block `L = min(lane_len, J_1)`.
    #[must_use]
    pub fn block_len(&self) -> usize {
        self.segment.block_len()
    }

    /// `m_1`.
    #[must_use]
    pub fn m1(&self) -> usize {
        self.shape().levels[0].rows
    }

    /// The constant column `L`.
    #[must_use]
    pub fn const_col(&self) -> usize {
        self.block_len()
    }

    /// First parity column `L + max(m_1, 128)`.
    #[must_use]
    pub fn y1_base(&self) -> usize {
        self.block_len() + self.m1().max(128)
    }

    /// Column of parity `rho_1`.
    #[must_use]
    pub fn y1_col(&self, rho1: usize) -> usize {
        self.y1_base() + rho1
    }

    /// Columns `[0, useful_bits)` are committed.
    #[must_use]
    pub fn useful_bits(&self) -> usize {
        self.y1_base() + self.m1()
    }

    /// u64 words of a row's committed data.
    #[must_use]
    pub fn useful_words(&self) -> usize {
        self.useful_bits().div_ceil(64)
    }

    /// `log2` of the block width.
    #[must_use]
    pub fn k_log(&self) -> usize {
        log2_ceil(self.useful_bits()).max(7)
    }

    /// Lane-major word range of row `(t, beta)`'s inputs in the part's packed input bits
    /// (`input_bits / 64` words).
    #[must_use]
    pub fn input_words(&self, t: usize, beta: usize) -> std::ops::Range<usize> {
        let start = (self.segment.column(t, beta * self.block_len())) / 64;
        start..start + self.block_len() / 64
    }

    /// The Flock registry type: empty matrix stubs (walker convention, K2), the constant
    /// pin, no IO schema.
    #[must_use]
    pub fn table_type(&self) -> TableType {
        TableType {
            k_log: self.k_log(),
            useful_bits: self.useful_bits(),
            a_0: empty_stub(),
            b_0: empty_stub(),
            c_0: empty_stub(),
            const_pin: Some(self.const_col()),
            class: TableClass::Boolean,
            io_schema: Vec::new(),
        }
    }

    /// `y1 = M_1 x` of every row, `x` the row's `L` input bits given by `input(t, beta)`
    /// (`L / 64` words): one `m_1`-bit set per row (`m_1 / 64` words), rows in row order.
    /// All lanes of a block are walked together with byte masks (as
    /// `agg_bridge::gf2_walker::composite`), parallel over blocks and position chunks.
    #[must_use]
    pub fn parities<'a>(
        &self,
        m1: &SparseMatrix,
        input: &(dyn Fn(usize, usize) -> &'a [u64] + Sync),
    ) -> Vec<Vec<u64>> {
        const CHUNK_WORDS: usize = 256;
        let (lanes, blocks, rows1) = (self.lanes(), self.blocks(), self.m1());
        assert_eq!(m1.rows, rows1, "M_1 of this shape");
        let words = self.block_len() / 64;
        let base = self.segment.zeta1_base();
        let chunks = words.div_ceil(CHUNK_WORDS);
        let masks: Vec<Vec<u8>> = (0..blocks)
            .into_par_iter()
            .map(|beta| {
                let inputs: Vec<&[u64]> = (0..lanes).map(|t| input(t, beta)).collect();
                (0..chunks)
                    .into_par_iter()
                    .map(|ch| {
                        let mut out = vec![0_u8; rows1];
                        for w in ch * CHUNK_WORDS..((ch + 1) * CHUNK_WORDS).min(words) {
                            let mut lane_words = [0_u64; 8];
                            for (t, x) in inputs.iter().enumerate() {
                                lane_words[t] = x[w];
                            }
                            let mut rest = lane_words.iter().fold(0, |acc, x| acc | x);
                            while rest != 0 {
                                let z = rest.trailing_zeros() as usize;
                                rest &= rest - 1;
                                let mask = lane_words[..lanes]
                                    .iter()
                                    .enumerate()
                                    .fold(0_u8, |m, (t, x)| m | ((((x >> z) & 1) as u8) << t));
                                for &r in m1.column(base + 64 * w + z) {
                                    out[r as usize] ^= mask;
                                }
                            }
                        }
                        out
                    })
                    .reduce(
                        || vec![0_u8; rows1],
                        |mut a, b| {
                            a.iter_mut().zip(&b).for_each(|(x, y)| *x ^= y);
                            a
                        },
                    )
            })
            .collect();
        (0..self.rows())
            .into_par_iter()
            .map(|j| {
                let (t, beta) = (j / blocks, j % blocks);
                let mut v = vec![0_u64; rows1 / 64];
                for (r, &m) in masks[beta].iter().enumerate() {
                    v[r / 64] |= u64::from((m >> t) & 1) << (r % 64);
                }
                v
            })
            .collect()
    }

    /// The committed row: `inputs` (`L / 64` words), the constant, the parities.
    #[must_use]
    pub fn row_block(&self, inputs: &[u64], parities: &[u64]) -> Vec<u64> {
        let mut z = vec![0_u64; self.useful_words()];
        z[..inputs.len()].copy_from_slice(inputs);
        let c = self.const_col();
        z[c / 64] |= 1 << (c % 64);
        let at = self.y1_base() / 64;
        z[at..at + parities.len()].copy_from_slice(parities);
        z
    }
}

/// The level-1 relation (module docs): layout, `sd` and its matrices (borrowed).
#[derive(Clone)]
pub struct Level1Slot<'m> {
    layout: Level1Layout,
    sd: [u8; 32],
    matrices: &'m BridgeMatrices,
}

impl std::fmt::Debug for Level1Slot<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Level1Slot")
            .field("layout", &self.layout)
            .field("sd", &self.sd)
            .finish_non_exhaustive()
    }
}

impl<'m> Level1Slot<'m> {
    /// The relation for `sd`, whose matrices are `matrices`.
    ///
    /// # Panics
    /// If the matrices have another shape than the layout.
    #[must_use]
    pub fn new(layout: Level1Layout, sd: [u8; 32], matrices: &'m BridgeMatrices) -> Self {
        assert!(
            layout.shape().matches(matrices),
            "matrices of another shape"
        );
        Self {
            layout,
            sd,
            matrices,
        }
    }

    #[must_use]
    pub fn layout(&self) -> &Level1Layout {
        &self.layout
    }

    #[must_use]
    pub fn matrices(&self) -> &'m BridgeMatrices {
        self.matrices
    }

    /// Row values `a = A_0 z`, `b = B_0 z` of a row `z` whose inputs have the level-1
    /// parities `parity` (`M_1` of `z`'s input columns; `a`, `b` zeroed on entry). An honest
    /// row has `a = z`, `b = 1` on every live column.
    pub fn apply_ab_with(&self, z: &[u64], parity: &[u64], a: &mut [u64], b: &mut [u64]) {
        let lay = &self.layout;
        let w = lay.useful_words();
        assert!(z.len() >= w && a.len() >= w && b.len() >= w, "row length");
        let one = bit(z, lay.const_col());
        let fill = one.wrapping_neg();
        let n_in = lay.block_len() / 64;
        a[..n_in].copy_from_slice(&z[..n_in]);
        b[..n_in].fill(fill);
        let c = lay.const_col();
        a[c / 64] |= one << (c % 64);
        b[c / 64] |= one << (c % 64);
        let m1 = lay.m1();
        let at = lay.y1_base() / 64;
        a[at..at + m1 / 64].copy_from_slice(&parity[..m1 / 64]);
        b[at..at + m1 / 64].fill(fill);
    }

    /// [`Self::apply_ab_with`] computing the row's parities itself.
    pub fn apply_ab(&self, z: &[u64], a: &mut [u64], b: &mut [u64]) {
        let lay = self.layout;
        let n_in = lay.block_len() / 64;
        // One row: the layout's walk over a single "lane" row (t = 0, beta = 0 view).
        let parity = &lay_single(&lay).parities(&self.matrices.m1, &|_, _| &z[..n_in])[0];
        self.apply_ab_with(z, parity, a, b);
    }

    /// SHA3-256 of the relation: label, `sd`, the shape, the layout (`lane_len`, offset,
    /// blocks, block length, `k_log`, useful bits, constant column, parity base).
    #[must_use]
    pub fn digest(&self) -> [u8; 32] {
        let lay = &self.layout;
        let mut h = Sha3_256::new();
        let put = |h: &mut Sha3_256, v: usize| h.update((v as u64).to_le_bytes());
        put(&mut h, LEVEL1_LABEL.len());
        h.update(LEVEL1_LABEL);
        h.update(self.sd);
        put_shape(&mut h, lay.shape());
        for v in [
            lay.segment.lane_len,
            lay.segment.offset,
            lay.blocks(),
            lay.block_len(),
            lay.k_log(),
            lay.useful_bits(),
            lay.const_col(),
            lay.y1_base(),
        ] {
            put(&mut h, v);
        }
        h.finalize().into()
    }

    /// `xi_A` on the input columns (the identity rows plus `M_1^T` of the parity rows'
    /// weights) and the constant column's two entries `(xi_A[one], xi_B[one])`.
    fn marginals(&self, eq: &[F128], xi_a_inputs: &mut [F128]) -> (F128, F128) {
        let lay = &self.layout;
        assert_eq!(eq.len(), 1 << lay.k_log(), "eq length is the block width");
        let len = lay.block_len();
        assert_eq!(xi_a_inputs.len(), len);
        let m1 = &self.matrices.m1;
        let base = lay.segment.zeta1_base();
        let v: Vec<F128> = (0..lay.m1()).map(|r| eq[lay.y1_col(r)]).collect();
        let in_sum = xi_a_inputs
            .par_chunks_mut(1 << 12)
            .zip(eq[..len].par_chunks(1 << 12))
            .enumerate()
            .map(|(ci, (x, e))| {
                let mut s = F128::ZERO;
                for (k, (xc, &ec)) in x.iter_mut().zip(e).enumerate() {
                    *xc = ec + gather(m1, base + (ci << 12) + k, |r| v[r]);
                    s += ec;
                }
                s
            })
            .reduce(|| F128::ZERO, |p, q| p + q);
        let one = eq[lay.const_col()];
        let v_sum = v.iter().fold(F128::ZERO, |acc, &x| acc + x);
        (one, one + in_sum + v_sum)
    }
}

/// A one-lane, one-block view of `lay` (for computing one row's parities alone).
fn lay_single(lay: &Level1Layout) -> Level1Layout {
    let mut shape = *lay.shape();
    shape.lanes = 1;
    let segment = SegmentLayout {
        shape,
        lane_len: lay.block_len(),
        offset: lay.segment.zeta1_base(),
    };
    Level1Layout { segment }
}

impl LincheckCircuit for Level1Slot<'_> {
    fn n_cols(&self) -> usize {
        1 << self.layout.k_log()
    }

    fn const_pin_col(&self) -> Option<usize> {
        Some(self.layout.const_col())
    }

    /// `comb[c] = alpha xi_A(c) + xi_B(c)`: the input columns and the constant column.
    fn fold_alpha_batched(&self, alpha: F128, eq: &[F128]) -> Vec<F128> {
        let lay = &self.layout;
        let mut comb = vec![F128::ZERO; 1 << lay.k_log()];
        let len = lay.block_len();
        let (a_one, b_one) = self.marginals(eq, &mut comb[..len]);
        comb[..len].par_chunks_mut(1 << 12).for_each(|chunk| {
            for x in chunk {
                *x = alpha * *x;
            }
        });
        comb[lay.const_col()] = alpha * a_one + b_one;
        comb
    }

    /// Both marginals from one walk.
    fn fold_split(&self, eq: &[F128]) -> (Vec<F128>, Vec<F128>) {
        let lay = &self.layout;
        let n = 1 << lay.k_log();
        let mut xi_a = vec![F128::ZERO; n];
        let mut xi_b = vec![F128::ZERO; n];
        let (a_one, b_one) = self.marginals(eq, &mut xi_a[..lay.block_len()]);
        xi_a[lay.const_col()] = a_one;
        xi_b[lay.const_col()] = b_one;
        (xi_a, xi_b)
    }
}

/// Column layout of the levels-2-3 type (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TopLayout {
    /// The part's input geometry.
    pub segment: SegmentLayout,
}

impl TopLayout {
    /// The levels-2-3 layout of a part.
    ///
    /// # Panics
    /// If the output region would not be aligned (`lanes G` not a multiple of `max(m_3,
    /// 128)`).
    #[must_use]
    pub fn new(segment: SegmentLayout) -> Self {
        let lay = Self { segment };
        let m3 = segment.output_bits();
        assert!(m3.is_power_of_two(), "m_3 is a power of two");
        assert!(
            lay.const_col().is_multiple_of(m3.max(128)),
            "the output region is aligned"
        );
        lay
    }

    /// The shape.
    #[must_use]
    pub fn shape(&self) -> &Shape {
        &self.segment.shape
    }

    /// `m_1`.
    #[must_use]
    pub fn m1(&self) -> usize {
        self.shape().levels[0].rows
    }

    /// Parities per lane `G = blocks * m_1`.
    #[must_use]
    pub fn group(&self) -> usize {
        self.segment.blocks() * self.m1()
    }

    /// Input columns `lanes * G`.
    #[must_use]
    pub fn input_bits(&self) -> usize {
        self.segment.lanes() * self.group()
    }

    /// Input column of parity `rho_1` of lane `t`, block `beta` (= `m_1 j + rho_1` for the
    /// level-1 row `j = beta + blocks t`).
    #[must_use]
    pub fn input_col(&self, t: usize, beta: usize, rho1: usize) -> usize {
        rho1 + self.m1() * beta + self.group() * t
    }

    /// The constant column `lanes * G`.
    #[must_use]
    pub fn const_col(&self) -> usize {
        self.input_bits()
    }

    /// First composite row `y_0`: `lanes G + max(m_3, 128)`, so the output columns are an
    /// aligned region of `m_3 / 128` words (pinned to zero by a claim).
    #[must_use]
    pub fn y_base(&self) -> usize {
        self.const_col() + self.segment.output_bits().max(128)
    }

    /// Row (and committed zero column) of composite row `o`.
    #[must_use]
    pub fn y_col(&self, o: usize) -> usize {
        self.y_base() + o
    }

    /// Columns `[0, useful_bits)` are committed.
    #[must_use]
    pub fn useful_bits(&self) -> usize {
        self.y_base() + self.segment.output_bits()
    }

    /// u64 words of the block's committed data.
    #[must_use]
    pub fn useful_words(&self) -> usize {
        self.useful_bits().div_ceil(64)
    }

    /// `log2` of the block width.
    #[must_use]
    pub fn k_log(&self) -> usize {
        log2_ceil(self.useful_bits()).max(7)
    }

    /// The Flock registry type.
    #[must_use]
    pub fn table_type(&self) -> TableType {
        TableType {
            k_log: self.k_log(),
            useful_bits: self.useful_bits(),
            a_0: empty_stub(),
            b_0: empty_stub(),
            c_0: empty_stub(),
            const_pin: Some(self.const_col()),
            class: TableClass::Boolean,
            io_schema: Vec::new(),
        }
    }

    /// The committed block: the level-1 parities `rows` (row order) concatenated, the
    /// constant, zeros.
    #[must_use]
    pub fn block(&self, parities: &[&[u64]]) -> Vec<u64> {
        let mut z = vec![0_u64; self.useful_words()];
        let per = self.m1() / 64;
        assert_eq!(parities.len() * self.m1(), self.input_bits());
        for (j, p) in parities.iter().enumerate() {
            z[per * j..per * (j + 1)].copy_from_slice(&p[..per]);
        }
        let c = self.const_col();
        z[c / 64] |= 1 << (c % 64);
        z
    }

    /// `Y = M_3 M_2 x` over GF(2), one bit per byte, for the block's inputs `x` (the first
    /// `input_bits / 64` words of `z`).
    #[must_use]
    pub fn parities(&self, mats: &BridgeMatrices, z: &[u64]) -> Vec<u8> {
        assert!(self.shape().matches(mats), "matrices of another shape");
        let (lanes, blocks, m1, g) = (
            self.segment.lanes(),
            self.segment.blocks(),
            self.m1(),
            self.group(),
        );
        let first = self.segment.first_block();
        let y2: Vec<u8> = (0..blocks * m1)
            .into_par_iter()
            .fold(
                || vec![0_u8; mats.m2.rows],
                |mut acc, i| {
                    let (beta, rho1) = (i / m1, i % m1);
                    let mut mask = 0_u8;
                    for t in 0..lanes {
                        mask |= (bit(z, rho1 + m1 * beta + g * t) as u8) << t;
                    }
                    if mask != 0 {
                        for &r in mats.m2.column(rho1 + m1 * (first + beta)) {
                            acc[r as usize] ^= mask;
                        }
                    }
                    acc
                },
            )
            .reduce(
                || vec![0_u8; mats.m2.rows],
                |mut a, b| {
                    a.iter_mut().zip(&b).for_each(|(x, y)| *x ^= y);
                    a
                },
            );
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
}

/// The levels-2-3 relation: layout, the matrices of `sd` (borrowed), and the public share `Y`.
#[derive(Clone)]
pub struct TopSlot<'m> {
    layout: TopLayout,
    sd: [u8; 32],
    matrices: &'m BridgeMatrices,
    y: Vec<u8>,
}

impl std::fmt::Debug for TopSlot<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TopSlot")
            .field("layout", &self.layout)
            .field("sd", &self.sd)
            .finish_non_exhaustive()
    }
}

impl<'m> TopSlot<'m> {
    /// The relation for `sd` (whose matrices are `matrices`) and the share `Y` (one bit per
    /// byte, `m_3` entries).
    ///
    /// # Panics
    /// If the matrices have another shape than the layout or `Y` is malformed.
    #[must_use]
    pub fn new(layout: TopLayout, sd: [u8; 32], matrices: &'m BridgeMatrices, y: Vec<u8>) -> Self {
        assert!(
            layout.shape().matches(matrices),
            "matrices of another shape"
        );
        assert_eq!(y.len(), layout.segment.output_bits(), "Y has m_3 bits");
        assert!(y.iter().all(|&b| b <= 1), "Y is a bit vector");
        Self {
            layout,
            sd,
            matrices,
            y,
        }
    }

    #[must_use]
    pub fn layout(&self) -> &TopLayout {
        &self.layout
    }

    #[must_use]
    pub fn y(&self) -> &[u8] {
        &self.y
    }

    #[must_use]
    pub fn matrices(&self) -> &'m BridgeMatrices {
        self.matrices
    }

    /// Row values of the block `z` (`a`, `b` zeroed on entry). For an honest block every
    /// composite row has `a = 0`, `b = 1`.
    pub fn apply_ab(&self, z: &[u64], a: &mut [u64], b: &mut [u64]) {
        let lay = &self.layout;
        let w = lay.useful_words();
        assert!(z.len() >= w && a.len() >= w && b.len() >= w, "block length");
        let one = bit(z, lay.const_col());
        let n_in = lay.input_bits() / 64;
        a[..n_in].copy_from_slice(&z[..n_in]);
        b[..n_in].fill(one.wrapping_neg());
        let c = lay.const_col();
        a[c / 64] |= one << (c % 64);
        b[c / 64] |= one << (c % 64);
        let mz = lay.parities(self.matrices, z);
        for (o, (&m, &y)) in mz.iter().zip(&self.y).enumerate() {
            let col = lay.y_col(o);
            let row = u64::from(m) ^ (u64::from(y) & one);
            a[col / 64] |= row << (col % 64);
            b[col / 64] |= one << (col % 64);
        }
    }

    /// SHA3-256 of the relation: label, `sd`, the shape, the layout (`lane_len`, offset,
    /// blocks, `G`, `k_log`, useful bits, constant column, first composite row) and `Y`.
    #[must_use]
    pub fn digest(&self) -> [u8; 32] {
        let lay = &self.layout;
        let mut h = Sha3_256::new();
        let put = |h: &mut Sha3_256, v: usize| h.update((v as u64).to_le_bytes());
        put(&mut h, TOP_LABEL.len());
        h.update(TOP_LABEL);
        h.update(self.sd);
        put_shape(&mut h, lay.shape());
        for v in [
            lay.segment.lane_len,
            lay.segment.offset,
            lay.segment.blocks(),
            lay.group(),
            lay.k_log(),
            lay.useful_bits(),
            lay.const_col(),
            lay.y_base(),
            self.y.len(),
        ] {
            put(&mut h, v);
        }
        h.update(pack_y(&self.y));
        h.finalize().into()
    }

    /// `xi_A` on the input columns (identity rows plus `M_2^T M_3^T` of the composite rows'
    /// weights) and the constant column's two entries.
    fn marginals(&self, eq: &[F128], xi_a_inputs: &mut [F128]) -> (F128, F128) {
        let lay = &self.layout;
        assert_eq!(eq.len(), 1 << lay.k_log(), "eq length is the block width");
        let mats = self.matrices;
        let (lanes, m1, g) = (lay.segment.lanes(), lay.m1(), lay.group());
        let first = lay.segment.first_block();
        let m3 = lay.segment.output_bits();
        let v3: Vec<F128> = (0..m3).map(|o| eq[lay.y_col(o)]).collect();
        let v2: Vec<F128> = (0..mats.m3.cols)
            .into_par_iter()
            .map(|zeta3| gather(&mats.m3, zeta3, |r| v3[r]))
            .collect();
        let n_in = lay.input_bits();
        assert_eq!(xi_a_inputs.len(), n_in);
        let in_sum = xi_a_inputs
            .par_chunks_mut(m1)
            .zip(eq[..n_in].par_chunks(m1))
            .enumerate()
            .map(|(j, (x, e))| {
                // Row j of level 1: lane t, block beta.
                let (t, beta) = (j / (g / m1), j % (g / m1));
                let mut s = F128::ZERO;
                for (rho1, (xc, &ec)) in x.iter_mut().zip(e).enumerate() {
                    *xc = ec + gather(&mats.m2, rho1 + m1 * (first + beta), |r| v2[t + lanes * r]);
                    s += ec;
                }
                s
            })
            .reduce(|| F128::ZERO, |p, q| p + q);
        let one = eq[lay.const_col()];
        let (mut a_one, mut b_one) = (one, one + in_sum);
        for (o, &e) in v3.iter().enumerate() {
            b_one += e;
            if self.y[o] == 1 {
                a_one += e;
            }
        }
        (a_one, b_one)
    }
}

impl LincheckCircuit for TopSlot<'_> {
    fn n_cols(&self) -> usize {
        1 << self.layout.k_log()
    }

    fn const_pin_col(&self) -> Option<usize> {
        Some(self.layout.const_col())
    }

    fn fold_alpha_batched(&self, alpha: F128, eq: &[F128]) -> Vec<F128> {
        let lay = &self.layout;
        let mut comb = vec![F128::ZERO; 1 << lay.k_log()];
        let n_in = lay.input_bits();
        let (a_one, b_one) = self.marginals(eq, &mut comb[..n_in]);
        comb[..n_in].par_chunks_mut(1 << 12).for_each(|chunk| {
            for x in chunk {
                *x = alpha * *x;
            }
        });
        comb[lay.const_col()] = alpha * a_one + b_one;
        comb
    }

    fn fold_split(&self, eq: &[F128]) -> (Vec<F128>, Vec<F128>) {
        let lay = &self.layout;
        let n = 1 << lay.k_log();
        let mut xi_a = vec![F128::ZERO; n];
        let mut xi_b = vec![F128::ZERO; n];
        let (a_one, b_one) = self.marginals(eq, &mut xi_a[..lay.input_bits()]);
        xi_a[lay.const_col()] = a_one;
        xi_b[lay.const_col()] = b_one;
        (xi_a, xi_b)
    }
}

/// The committed blocks of a bridge union: the level-1 rows and the levels-2-3 blocks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BridgeBlocks {
    /// Level-1 rows in union row order ([`BridgeUnion::row`]; `Level1Layout::useful_words`
    /// u64 words each): the block's input bits, the constant, the level-1 parities.
    pub rows: Vec<Vec<u64>>,
    /// The levels-2-3 blocks, one per segment of the span (`TopLayout::useful_words` words
    /// each): the segment's rows' parities concatenated, the constant, zeros.
    pub tops: Vec<Vec<u64>>,
}

/// Sizes of a bridge union.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BridgeUnionCost {
    /// Row capacity `log2` (uniform).
    pub nu: usize,
    /// Block width `log2` of the level-1 and the levels-2-3 type.
    pub k_log: [usize; 2],
    /// Live rows of the two types.
    pub rows: [usize; 2],
    /// Address-space variables `M = nu + max k_log + 1`.
    pub m: usize,
    /// Committed polynomial variables (dense stack).
    pub dense_m: usize,
    /// Committed 128-bit words (dense stack before padding).
    pub committed_words: usize,
    /// Entries the two walkers read per fold: `k_1 L` (level 1, the same `M_1` block for
    /// every row) plus `k_2 lanes G + k_3 J_3` (levels 2-3).
    pub walker_entries: u64,
}

/// The bridge union of one part, or of the same part of `2^span_log` segments (module docs):
/// layouts, union geometry, tie points.
pub struct BridgeUnion {
    l1: Level1Layout,
    top: TopLayout,
    span_log: usize,
    geometry: UnionGeometry,
}

impl BridgeUnion {
    /// Caller index of the level-1 type in the union geometry.
    pub const LEVEL1: usize = 0;
    /// Caller index of the levels-2-3 type.
    pub const TOP: usize = 1;

    /// The bridge union of the part `segment` (one segment).
    #[must_use]
    pub fn new(segment: SegmentLayout, profile: Profile) -> Self {
        Self::new_span(segment, 0, profile)
    }

    /// The bridge union of the part `segment` in each of `2^span_log` segments: row capacity
    /// `2^nu`, `nu = max(span_log + rows_log, 3, 21 - max k_log)`, so that `M = nu + max k_log
    /// + 1 >= 22` (Flock's dense floor).
    #[must_use]
    pub fn new_span(segment: SegmentLayout, span_log: usize, profile: Profile) -> Self {
        let l1 = Level1Layout::new(segment);
        let top = TopLayout::new(segment);
        let k_max = l1.k_log().max(top.k_log());
        let nu = (span_log + l1.rows_log())
            .max(3)
            .max((MIN_M - 1).saturating_sub(k_max));
        let geometry = UnionGeometry::new_multi(
            vec![l1.table_type(), top.table_type()],
            vec![l1.rows() << span_log, 1 << span_log],
            nu,
            profile,
        );
        assert_eq!(geometry.m_total(), nu + k_max + 1, "two-slot address space");
        Self {
            l1,
            top,
            span_log,
            geometry,
        }
    }

    /// The production bridge union of part `index` of a segment split into parts of
    /// `2^d` record positions (`lane_len = 2^(10 + d)`).
    ///
    /// # Panics
    /// If `d < 3` (the parity region needs blocks of at least `m_1` positions) or the part
    /// lies outside the segment.
    #[must_use]
    pub fn production_part(d: u32, index: usize, profile: Profile) -> Self {
        Self::production_span(d, index, 0, profile)
    }

    /// [`Self::production_part`] in each of `2^span_log` segments.
    ///
    /// # Panics
    /// As [`Self::production_part`].
    #[must_use]
    pub fn production_span(d: u32, index: usize, span_log: usize, profile: Profile) -> Self {
        assert!(d >= 3, "a part of at least 8 record positions");
        Self::new_span(SegmentLayout::production_part(d, index), span_log, profile)
    }

    /// `log2` of the segments the union covers.
    #[must_use]
    pub fn span_log(&self) -> usize {
        self.span_log
    }

    /// Segments the union covers (`2^span_log`, rows of the levels-2-3 type).
    #[must_use]
    pub fn span(&self) -> usize {
        1 << self.span_log
    }

    /// Union row of segment `sigma`, lane `t`, block `beta` of the level-1 type.
    #[must_use]
    pub fn row(&self, sigma: usize, t: usize, beta: usize) -> usize {
        sigma + (self.l1.row(t, beta) << self.span_log)
    }

    #[must_use]
    pub fn level1(&self) -> &Level1Layout {
        &self.l1
    }

    #[must_use]
    pub fn top(&self) -> &TopLayout {
        &self.top
    }

    #[must_use]
    pub fn geometry(&self) -> &UnionGeometry {
        &self.geometry
    }

    /// The part's input geometry.
    #[must_use]
    pub fn segment(&self) -> &SegmentLayout {
        &self.l1.segment
    }

    /// Sizes.
    #[must_use]
    pub fn cost(&self) -> BridgeUnionCost {
        let shape = self.l1.shape();
        let [l1, l2, l3] = &shape.levels;
        BridgeUnionCost {
            nu: self.geometry.nu(),
            k_log: [self.l1.k_log(), self.top.k_log()],
            rows: [self.l1.rows() << self.span_log, self.span()],
            m: self.geometry.m_total(),
            dense_m: self.geometry.dense_m(),
            committed_words: self.geometry.dense_words(),
            walker_entries: (l1.k * self.l1.block_len()
                + l2.k * self.top.input_bits()
                + l3.k * l3.cols) as u64,
        }
    }

    /// The honest blocks of the lane-major packed input bits of `S <= 2^span_log` segments
    /// (`S * input_bits / 64` words, segment-major; segments `S..2^span_log` are zero): the
    /// level-1 rows with `y1 = M_1 b'(sigma, t, beta)` in union row order, one levels-2-3
    /// block of the parities per segment.
    ///
    /// # Panics
    /// If `bits` has the wrong length or the matrices another shape.
    #[must_use]
    pub fn honest_blocks(&self, matrices: &BridgeMatrices, bits: &[u64]) -> BridgeBlocks {
        let seg = self.segment();
        let seg_words = seg.input_bits() / 64;
        let segments = bits.len() / seg_words;
        assert!(
            bits.len() == segments * seg_words && (1..=self.span()).contains(&segments),
            "packed input length: 1 to 2^span_log segments"
        );
        assert!(seg.shape.matches(matrices), "matrices of another shape");
        let l1 = &self.l1;
        let zero_inputs = vec![0_u64; l1.block_len() / 64];
        let zero_parities = vec![0_u64; l1.m1() / 64];
        let mut rows = vec![Vec::new(); l1.rows() << self.span_log];
        let mut tops = Vec::with_capacity(self.span());
        for sigma in 0..self.span() {
            let b = (sigma < segments).then(|| &bits[sigma * seg_words..(sigma + 1) * seg_words]);
            let input = |t: usize, beta: usize| -> &[u64] {
                b.map_or(&zero_inputs[..], |b| &b[l1.input_words(t, beta)])
            };
            let parities = match b {
                Some(_) => l1.parities(&matrices.m1, &input),
                None => vec![zero_parities.clone(); l1.rows()],
            };
            let own: Vec<Vec<u64>> = parities
                .par_iter()
                .enumerate()
                .map(|(j, p)| {
                    let (t, beta) = (j / l1.blocks(), j % l1.blocks());
                    l1.row_block(input(t, beta), p)
                })
                .collect();
            for (j, row) in own.into_iter().enumerate() {
                rows[sigma + (j << self.span_log)] = row;
            }
            let refs: Vec<&[u64]> = parities.iter().map(Vec::as_slice).collect();
            tops.push(self.top.block(&refs));
        }
        BridgeBlocks { rows, tops }
    }

    /// Set every levels-2-3 block's output columns to the composite of its inputs, `z_{y_o}
    /// = (M_3 M_2 z_in)_o` (the segment's share): the honest output columns of the
    /// zero-coefficient relation of a span (module docs). [`Self::honest_blocks`] leaves them
    /// zero, the honest columns of the relation with `Y` as a coefficient.
    pub fn fill_outputs(&self, matrices: &BridgeMatrices, blocks: &mut BridgeBlocks) {
        let top = &self.top;
        for block in &mut blocks.tops {
            let y = top.parities(matrices, block);
            for (o, &b) in y.iter().enumerate() {
                let c = top.y_col(o);
                block[c / 64] = (block[c / 64] & !(1 << (c % 64))) | (u64::from(b) << (c % 64));
            }
        }
    }

    /// The share `Y = M_3 M_2 y1` of the blocks' levels-2-3 inputs (one segment).
    ///
    /// # Panics
    /// If the union covers more than one segment.
    #[must_use]
    pub fn share(&self, matrices: &BridgeMatrices, blocks: &BridgeBlocks) -> Vec<u8> {
        assert_eq!(self.span(), 1, "one segment");
        self.top.parities(matrices, &blocks.tops[0])
    }

    /// The final vectors `Y_sigma = M_3 M_2 y1_sigma` of every segment of the span
    /// (`2^span_log` of them; zero for segments without bits).
    #[must_use]
    pub fn shares(&self, matrices: &BridgeMatrices, blocks: &BridgeBlocks) -> Vec<Vec<u8>> {
        blocks
            .tops
            .iter()
            .map(|top| self.top.parities(matrices, top))
            .collect()
    }

    /// The two relations for `sd`, its matrices and the share `y`.
    #[must_use]
    pub fn slots<'m>(
        &self,
        sd: [u8; 32],
        matrices: &'m BridgeMatrices,
        y: Vec<u8>,
    ) -> (Level1Slot<'m>, TopSlot<'m>) {
        (
            Level1Slot::new(self.l1, sd, matrices),
            TopSlot::new(self.top, sd, matrices, y),
        )
    }

    /// The packed Flock buffers of the two types (caller order: level 1, levels 2-3) for
    /// `blocks`, with the row values under the given relations (`None`: zero `a`, `b`, for
    /// a commitment alone). The level-1 values use the parities of the committed rows'
    /// own inputs, so a tampered parity breaks its row.
    ///
    /// # Panics
    /// If the blocks have the wrong shape.
    #[must_use]
    pub fn buffers(
        &self,
        blocks: &BridgeBlocks,
        slots: Option<(&Level1Slot<'_>, &TopSlot<'_>)>,
    ) -> Vec<SlotBuffers> {
        let (l1, top) = (&self.l1, &self.top);
        let (span, span_log) = (self.span(), self.span_log);
        assert_eq!(
            blocks.rows.len(),
            l1.rows() << span_log,
            "one block per level-1 row"
        );
        assert!(blocks.rows.iter().all(|r| r.len() == l1.useful_words()));
        assert_eq!(blocks.tops.len(), span, "one levels-2-3 block per segment");
        assert!(
            blocks.tops.iter().all(|t| t.len() == top.useful_words()),
            "levels-2-3 block length"
        );
        let nu = self.geometry.nu();
        let n_in = l1.block_len() / 64;
        // Per segment, the parities of the committed rows' own inputs (row order of the
        // segment, `beta + blocks t`).
        let parities: Option<Vec<Vec<Vec<u64>>>> = slots.map(|(s1, _)| {
            (0..span)
                .map(|sigma| {
                    l1.parities(&s1.matrices().m1, &|t, beta| {
                        &blocks.rows[self.row(sigma, t, beta)][..n_in]
                    })
                })
                .collect()
        });
        let b1 = pack_rows(
            l1.k_log(),
            l1.useful_bits(),
            nu,
            l1.rows() << span_log,
            |j, z, a, b| {
                z.copy_from_slice(&blocks.rows[j]);
                if let (Some((s1, _)), Some(p)) = (slots, &parities) {
                    s1.apply_ab_with(z, &p[j % span][j >> span_log], a, b);
                }
            },
        );
        let b2 = pack_rows(
            top.k_log(),
            top.useful_bits(),
            nu,
            span,
            |sigma, z, a, b| {
                z.copy_from_slice(&blocks.tops[sigma]);
                if let Some((_, s2)) = slots {
                    s2.apply_ab(z, a, b);
                }
            },
        );
        vec![b1, b2]
    }

    fn lane_coords(&self, sigma: &[F128], lane: &[F128]) {
        assert_eq!(
            lane.len(),
            self.l1.lane_log(),
            "one coordinate per lane-index bit"
        );
        assert_eq!(
            sigma.len(),
            self.span_log,
            "one coordinate per segment-index bit"
        );
    }

    /// Claim point on the level-1 inputs: rows `[sigma (span_log) | beta (blocks_log) | lane
    /// (lane_log) | 0]`, word columns `cols` (`log2(L / 128)` coordinates over the block's
    /// input words; for the production layout `[w (3) | i mod 2^9 (min(d, 9))]`).
    #[must_use]
    pub fn input_point(
        &self,
        sigma: &[F128],
        beta: &[F128],
        lane: &[F128],
        cols: &[F128],
    ) -> Vec<F128> {
        assert_eq!(beta.len(), self.l1.blocks_log());
        self.lane_coords(sigma, lane);
        assert_eq!(
            cols.len(),
            log2_ceil(self.l1.block_len() / 128),
            "one coordinate per input-word bit"
        );
        let rows: Vec<F128> = sigma.iter().chain(beta).chain(lane).copied().collect();
        self.geometry.region_point(Self::LEVEL1, &rows, cols)
    }

    /// Claim point on the level-1 parities: rows `[sigma | beta | lane | 0]`, word columns
    /// `[y (log2(m_1 / 128)) | the parity region's word base above them, as 0/1 constants]`.
    ///
    /// # Panics
    /// If `m_1` is not a multiple of 128.
    #[must_use]
    pub fn y1_point(&self, sigma: &[F128], beta: &[F128], lane: &[F128], y: &[F128]) -> Vec<F128> {
        let m1 = self.l1.m1();
        assert!(m1.is_multiple_of(128), "parity words need m_1 % 128 == 0");
        let y_log = log2_ceil(m1 / 128);
        assert_eq!(y.len(), y_log);
        assert_eq!(beta.len(), self.l1.blocks_log());
        self.lane_coords(sigma, lane);
        let base = (self.l1.y1_base() / 128) >> y_log;
        let k = self.l1.k_log() - 7;
        let mut cols = y.to_vec();
        for b in 0..k - y_log {
            cols.push(if (base >> b) & 1 == 1 {
                F128::ONE
            } else {
                F128::ZERO
            });
        }
        let rows: Vec<F128> = sigma.iter().chain(beta).chain(lane).copied().collect();
        self.geometry.region_point(Self::LEVEL1, &rows, &cols)
    }

    /// Claim point on the levels-2-3 output columns `y_o` (value: the multilinear extension
    /// of the public `Y` table at the point): rows `[sigma]`, word columns `[o (log2(m_3 /
    /// 128)) | the region's word base above them]`.
    ///
    /// # Panics
    /// If `o` or `sigma` has the wrong length.
    #[must_use]
    pub fn output_point(&self, sigma: &[F128], o: &[F128]) -> Vec<F128> {
        let m3 = self.top.segment.output_bits();
        let o_log = log2_ceil(m3.div_ceil(128));
        assert_eq!(o.len(), o_log, "one coordinate per output-word bit");
        assert_eq!(
            sigma.len(),
            self.span_log,
            "one coordinate per segment-index bit"
        );
        let base = (self.top.y_base() / 128) >> o_log;
        let k = self.top.k_log() - 7;
        let mut cols = o.to_vec();
        for b in 0..k - o_log {
            cols.push(if (base >> b) & 1 == 1 {
                F128::ONE
            } else {
                F128::ZERO
            });
        }
        self.geometry.region_point(Self::TOP, sigma, &cols)
    }

    /// Claim point on the levels-2-3 inputs: rows `[sigma]`, word columns `[y | beta | lane |
    /// 0]` (input word `y + (m_1 / 128) beta + (G / 128) t`).
    #[must_use]
    pub fn top_point(&self, sigma: &[F128], y: &[F128], beta: &[F128], lane: &[F128]) -> Vec<F128> {
        let m1 = self.l1.m1();
        assert!(m1.is_multiple_of(128), "parity words need m_1 % 128 == 0");
        assert_eq!(y.len(), log2_ceil(m1 / 128));
        assert_eq!(beta.len(), self.l1.blocks_log());
        self.lane_coords(sigma, lane);
        let cols: Vec<F128> = y.iter().chain(beta).chain(lane).copied().collect();
        self.geometry.region_point(Self::TOP, sigma, &cols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flock_union::eval_packed;
    use agg_bridge::gf2_walker;
    use agg_bridge::seed::derive_matrices_for;
    use bincode::Options as _;
    use flock_core::challenger::{Challenger, FsChallenger};
    use flock_core::lincheck::SparseMatrixCircuit;
    use flock_core::test_rng::Rng;
    use std::sync::Arc;

    fn random_f128(rng: &mut Rng) -> F128 {
        F128::new(rng.next_u64(), rng.next_u64())
    }

    fn sd(byte: u8) -> [u8; 32] {
        std::array::from_fn(|i| byte.wrapping_mul(i as u8 + 1))
    }

    fn random_bits(seg: &SegmentLayout, rng: &mut Rng) -> Vec<u64> {
        (0..seg.input_bits() / 64).map(|_| rng.next_u64()).collect()
    }

    /// The toy segment (`L = 2^12`, one lane, four level-1 blocks of 1,024 positions).
    fn toy() -> (BridgeUnion, Arc<BridgeMatrices>, Vec<u64>) {
        let seg = SegmentLayout::new(Shape::TOY, 1 << 12);
        let bu = BridgeUnion::new(seg, Profile::Fast);
        let mats = Arc::new(derive_matrices_for(&Shape::TOY, &sd(3)));
        let mut rng = Rng::new(3);
        let bits = random_bits(&seg, &mut rng);
        (bu, mats, bits)
    }

    #[test]
    fn layouts_have_the_documented_columns() {
        let (bu, _, _) = toy();
        let (l1, top) = (bu.level1(), bu.top());
        assert_eq!((l1.blocks(), l1.block_len(), l1.rows()), (4, 1024, 4));
        assert_eq!(
            (l1.const_col(), l1.y1_base(), l1.useful_bits()),
            (1024, 1152, 1216)
        );
        assert_eq!(l1.k_log(), 11);
        assert_eq!(
            (top.group(), top.const_col(), top.y_base()),
            (256, 256, 384)
        );
        assert_eq!((top.useful_bits(), top.k_log()), (400, 9));
        assert_eq!(bu.geometry().m_total(), 22);
        // Production parts: d = 3 (one block of 2^13), 9 (one block of 2^19), 10 (two blocks),
        // 13 (a full segment: 16 blocks, M = 28, 97 live rows).
        for (d, k1, k2, nu, m) in [
            (3u32, 15, 16, 5, 22),
            (8, 19, 16, 3, 23),
            (9, 20, 16, 3, 24),
            (10, 20, 17, 4, 25),
            (13, 20, 20, 7, 28),
        ] {
            let bu = BridgeUnion::production_part(d, 0, Profile::Fast);
            let c = bu.cost();
            assert_eq!(c.k_log, [k1, k2], "d {d}");
            assert_eq!((c.nu, c.m), (nu, m), "d {d}");
            assert_eq!(c.rows, [6 << d.saturating_sub(9), 1]);
            let l1 = bu.level1();
            assert_eq!(l1.y1_base(), l1.block_len() + 8192);
            assert_eq!(bu.top().const_col(), 6 * 8192 * l1.blocks());
        }
        // The walker of a full segment: 14 x 2^19 + 14 x 6 x 2^17 + 18 x 49,152 entries
        // (the one-row bridge slot's: 7.2 x 10^8).
        let full = BridgeUnion::production_part(13, 0, Profile::Fast);
        assert_eq!(full.cost().walker_entries, 19_234_816);
        assert_eq!(full.segment().factored_nonzeros(), 716_537_856);
    }

    /// Honest rows satisfy `a & b = z`; a flipped parity, input, `Y` or zero-column bit breaks
    /// a row.
    #[test]
    fn honest_rows_satisfy_the_relation_and_tampering_breaks_a_row() {
        let (bu, mats, bits) = toy();
        let blocks = bu.honest_blocks(&mats, &bits);
        let y = bu.share(&mats, &blocks);
        assert_eq!(y, gf2_walker::composite(bu.segment(), &mats, &bits));
        let (s1, s2) = bu.slots(sd(3), &mats, y.clone());
        let ok1 = |z: &[u64]| {
            let w = bu.level1().useful_words();
            let (mut a, mut b) = (vec![0_u64; w], vec![0_u64; w]);
            s1.apply_ab(z, &mut a, &mut b);
            (0..w).all(|i| a[i] & b[i] == z[i])
        };
        let ok2 = |s: &TopSlot<'_>, z: &[u64]| {
            let w = bu.top().useful_words();
            let (mut a, mut b) = (vec![0_u64; w], vec![0_u64; w]);
            s.apply_ab(z, &mut a, &mut b);
            (0..w).all(|i| a[i] & b[i] == z[i])
        };
        for r in &blocks.rows {
            assert!(ok1(r));
        }
        assert!(ok2(&s2, &blocks.tops[0]));
        let l1 = *bu.level1();
        let flip = |v: &[u64], c: usize| {
            let mut v = v.to_vec();
            v[c / 64] ^= 1 << (c % 64);
            v
        };
        // A parity bit, an input bit (every M_1 column is nonzero), a zero column.
        assert!(!ok1(&flip(&blocks.rows[2], l1.y1_col(17))));
        assert!(!ok1(&flip(&blocks.rows[2], 300)));
        assert!(!ok1(&flip(&blocks.rows[1], l1.const_col() + 5)));
        // Levels 2-3: a flipped Y bit; a flipped input whose composite column is nonzero.
        let mut wrong = y.clone();
        wrong[3] ^= 1;
        let (_, bad) = bu.slots(sd(3), &mats, wrong);
        assert!(!ok2(&bad, &blocks.tops[0]));
        let top = *bu.top();
        let c = (0..top.input_bits())
            .find(|&c| top.parities(&mats, &flip(&blocks.tops[0], c)) != y)
            .expect("a visible column");
        assert!(!ok2(&s2, &flip(&blocks.tops[0], c)));
        assert!(!ok2(&s2, &flip(&blocks.tops[0], top.y_col(2))));
    }

    /// A row evaluator `(z, a, b)`.
    type RowEval<'a> = dyn Fn(&[u64], &mut [u64], &mut [u64]) + 'a;

    /// `<comb, z> = alpha <eq, A z> + <eq, B z>` for random blocks: the walkers are the
    /// transposes of the row evaluators.
    fn check_adjoint(
        circuit: &dyn LincheckCircuit,
        words: usize,
        useful: usize,
        const_col: usize,
        apply: &RowEval<'_>,
        seed: u64,
    ) {
        let k = circuit.n_cols();
        let mut rng = Rng::new(seed);
        let eq: Vec<F128> = (0..k).map(|_| random_f128(&mut rng)).collect();
        let alpha = random_f128(&mut rng);
        let comb = circuit.fold_alpha_batched(alpha, &eq);
        assert!(comb[useful..].iter().all(|&c| c == F128::ZERO));
        for trial in 0..3 {
            let mut z: Vec<u64> = (0..words).map(|_| rng.next_u64()).collect();
            if trial != 1 {
                z[const_col / 64] |= 1 << (const_col % 64);
            }
            if !useful.is_multiple_of(64) {
                z[words - 1] &= (1 << (useful % 64)) - 1;
            }
            let (mut a, mut b) = (vec![0_u64; words], vec![0_u64; words]);
            apply(&z, &mut a, &mut b);
            let (mut lhs, mut rhs) = (F128::ZERO, F128::ZERO);
            for col in 0..useful {
                if bit(&z, col) == 1 {
                    lhs += comb[col];
                }
                if bit(&a, col) == 1 {
                    rhs += alpha * eq[col];
                }
                if bit(&b, col) == 1 {
                    rhs += eq[col];
                }
            }
            assert_eq!(lhs, rhs, "trial {trial}");
        }
        // fold_split equals two alpha folds.
        let b0 = circuit.fold_alpha_batched(F128::ZERO, &eq);
        let mut a1 = circuit.fold_alpha_batched(F128::ONE, &eq);
        a1.iter_mut().zip(&b0).for_each(|(x, y)| *x += *y);
        assert_eq!(circuit.fold_split(&eq), (a1, b0));
    }

    #[test]
    fn walkers_are_the_transposes_of_the_row_evaluators() {
        let (bu, mats, _) = toy();
        let y: Vec<u8> = (0..16).map(|o| (o % 3 == 0) as u8).collect();
        let (s1, s2) = bu.slots(sd(3), &mats, y);
        let (l1, top) = (*bu.level1(), *bu.top());
        check_adjoint(
            &s1,
            l1.useful_words(),
            l1.useful_bits(),
            l1.const_col(),
            &|z, a, b| s1.apply_ab(z, a, b),
            1,
        );
        check_adjoint(
            &s2,
            top.useful_words(),
            top.useful_bits(),
            top.const_col(),
            &|z, a, b| s2.apply_ab(z, a, b),
            2,
        );
        // A production part of 2^10 records at index 1 (two blocks, offset): both types.
        let bu = BridgeUnion::production_part(10, 1, Profile::Fast);
        let mats = Arc::new(derive_matrices_for(&Shape::PRODUCTION, &sd(4)));
        let y: Vec<u8> = (0..4096).map(|o| (o % 5 == 1) as u8).collect();
        let (s1, s2) = bu.slots(sd(4), &mats, y);
        let (l1, top) = (*bu.level1(), *bu.top());
        check_adjoint(
            &s1,
            l1.useful_words(),
            l1.useful_bits(),
            l1.const_col(),
            &|z, a, b| s1.apply_ab(z, a, b),
            3,
        );
        check_adjoint(
            &s2,
            top.useful_words(),
            top.useful_bits(),
            top.const_col(),
            &|z, a, b| s2.apply_ab(z, a, b),
            4,
        );
    }

    /// Explicit `A_0`, `B_0` of the level-1 type.
    fn lowered_level1(s: &Level1Slot<'_>) -> (SparseBinaryMatrix, SparseBinaryMatrix) {
        let lay = *s.layout();
        let k = 1usize << lay.k_log();
        let one = lay.const_col();
        let mut a_rows = vec![Vec::new(); k];
        let mut b_rows = vec![Vec::new(); k];
        for c in 0..lay.block_len() {
            a_rows[c].push(c);
            b_rows[c].push(one);
            for &r in s.matrices().m1.column(lay.segment.zeta1_base() + c) {
                a_rows[lay.y1_col(r as usize)].push(c);
            }
        }
        a_rows[one].push(one);
        b_rows[one].push(one);
        for r in 0..lay.m1() {
            b_rows[lay.y1_col(r)].push(one);
            a_rows[lay.y1_col(r)].sort_unstable();
        }
        (
            SparseBinaryMatrix::new(k, k, a_rows),
            SparseBinaryMatrix::new(k, k, b_rows),
        )
    }

    /// Explicit `A_0`, `B_0` of the levels-2-3 type, from the dense `M_3 M_2` (toy only).
    fn lowered_top(s: &TopSlot<'_>) -> (SparseBinaryMatrix, SparseBinaryMatrix) {
        let lay = *s.layout();
        let k = 1usize << lay.k_log();
        let one = lay.const_col();
        let mut a_rows = vec![Vec::new(); k];
        let mut b_rows = vec![Vec::new(); k];
        for c in 0..lay.input_bits() {
            a_rows[c].push(c);
            b_rows[c].push(one);
            let mut e = vec![0_u64; lay.useful_words()];
            e[c / 64] |= 1 << (c % 64);
            for (o, &v) in lay.parities(s.matrices(), &e).iter().enumerate() {
                if v == 1 {
                    a_rows[lay.y_col(o)].push(c);
                }
            }
        }
        a_rows[one].push(one);
        b_rows[one].push(one);
        for o in 0..lay.segment.output_bits() {
            if s.y()[o] == 1 {
                a_rows[lay.y_col(o)].push(one);
            }
            b_rows[lay.y_col(o)].push(one);
        }
        (
            SparseBinaryMatrix::new(k, k, a_rows),
            SparseBinaryMatrix::new(k, k, b_rows),
        )
    }

    fn check_against_sparse(
        c: &dyn LincheckCircuit,
        lowered: (SparseBinaryMatrix, SparseBinaryMatrix),
        seed: u64,
    ) {
        let reference =
            SparseMatrixCircuit::new(&lowered.0, &lowered.1).with_const_pin(c.const_pin_col());
        let mut rng = Rng::new(seed);
        let eq: Vec<F128> = (0..c.n_cols()).map(|_| random_f128(&mut rng)).collect();
        let alpha = random_f128(&mut rng);
        assert_eq!(reference.n_cols(), c.n_cols());
        assert_eq!(
            c.fold_alpha_batched(alpha, &eq),
            reference.fold_alpha_batched(alpha, &eq)
        );
        assert_eq!(c.fold_split(&eq), reference.fold_split(&eq));
    }

    /// The walkers equal Flock's `SparseMatrixCircuit` fold on the explicitly lowered rows:
    /// both types on the toy shape, level 1 of a production part at `d = 3` (part 5, so the
    /// in-block offset `zeta1_base` is nonzero).
    #[test]
    fn walkers_equal_flock_sparse_fold_on_lowered_rows() {
        let (bu, mats, _) = toy();
        let y: Vec<u8> = (0..16).map(|o| (o % 4 == 1) as u8).collect();
        let (s1, s2) = bu.slots(sd(3), &mats, y);
        check_against_sparse(&s1, lowered_level1(&s1), 5);
        check_against_sparse(&s2, lowered_top(&s2), 6);
        let bu = BridgeUnion::production_part(3, 5, Profile::Fast);
        assert_ne!(bu.segment().zeta1_base(), 0);
        let mats = Arc::new(derive_matrices_for(&Shape::PRODUCTION, &sd(7)));
        let (s1, _) = bu.slots(sd(7), &mats, vec![0; 4096]);
        check_against_sparse(&s1, lowered_level1(&s1), 7);
    }

    #[test]
    fn digests_bind_seed_layout_and_y() {
        let (bu, mats, _) = toy();
        let y = vec![0_u8; 16];
        let (s1, s2) = bu.slots(sd(3), &mats, y.clone());
        let (o1, o2) = bu.slots(sd(4), &mats, y.clone());
        assert_ne!(s1.digest(), o1.digest());
        assert_ne!(s2.digest(), o2.digest());
        let mut y1 = y.clone();
        y1[0] = 1;
        let (_, t2) = bu.slots(sd(3), &mats, y1);
        assert_ne!(s2.digest(), t2.digest());
        let other = BridgeUnion::new(SegmentLayout::part(Shape::TOY, 1 << 11, 1), Profile::Fast);
        let (p1, p2) = other.slots(sd(3), &mats, y);
        assert_ne!(s1.digest(), p1.digest());
        assert_ne!(s2.digest(), p2.digest());
        assert_ne!(s1.digest(), s2.digest());
    }

    /// The honest share equals the segment walk of `agg_bridge` (production parts with one
    /// block, two blocks and an offset), and the region points evaluate the regions they
    /// name: the level-1 parities and their copies agree, the inputs are the bits.
    #[test]
    fn honest_share_and_region_points() {
        let mats = Arc::new(derive_matrices_for(&Shape::PRODUCTION, &sd(9)));
        let mut rng = Rng::new(9);
        for (d, index) in [(3u32, 2usize), (10, 1)] {
            let bu = BridgeUnion::production_part(d, index, Profile::Fast);
            let bits = random_bits(bu.segment(), &mut rng);
            let blocks = bu.honest_blocks(&mats, &bits);
            assert_eq!(
                bu.share(&mats, &blocks),
                gf2_walker::composite(bu.segment(), &mats, &bits),
                "d {d}"
            );
            let bufs = bu.buffers(&blocks, None);
            let zs: Vec<&[F128]> = bufs.iter().map(|b| b.z.as_slice()).collect();
            let union_z = scatter_for_test(bu.geometry(), &zs);
            let nu = bu.geometry().nu();
            let rand = |rng: &mut Rng, k: usize| -> Vec<F128> {
                (0..k).map(|_| random_f128(rng)).collect()
            };
            let beta = rand(&mut rng, bu.level1().blocks_log());
            // Lanes 0..3 (lanes 6, 7 have no level-1 row; their "copies" would be the
            // constant and composite columns, which is why the ties use lane groups).
            let mut lane = rand(&mut rng, 3);
            lane[2] = F128::ZERO;
            let y = rand(&mut rng, 6);
            let v_y1 = eval_packed(&union_z, &bu.y1_point(&[], &beta, &lane, &y), nu);
            let v_top = eval_packed(&union_z, &bu.top_point(&[], &y, &beta, &lane), nu);
            assert_eq!(v_y1, v_top, "parity copies (d {d})");
            assert_ne!(v_y1, F128::ZERO);
            // A flipped parity copy breaks the identity.
            let mut bad = blocks.clone();
            bad.tops[0][0] ^= 1;
            let bufs = bu.buffers(&bad, None);
            let zs: Vec<&[F128]> = bufs.iter().map(|b| b.z.as_slice()).collect();
            let bad_z = scatter_for_test(bu.geometry(), &zs);
            let lane0 = [F128::ZERO; 3];
            let y0 = [F128::ZERO; 6];
            let beta0 = vec![F128::ZERO; bu.level1().blocks_log()];
            assert_ne!(
                eval_packed(&bad_z, &bu.y1_point(&[], &beta0, &lane0, &y0), nu),
                eval_packed(&bad_z, &bu.top_point(&[], &y0, &beta0, &lane0), nu)
            );
            // The input region: word w of record 0, lane 0 is bits word 2 w, 2 w + 1.
            let cols = vec![F128::ZERO; bu.level1().block_len().trailing_zeros() as usize - 7];
            let v_in = eval_packed(&union_z, &bu.input_point(&[], &beta0, &lane0, &cols), nu);
            assert_eq!(v_in, F128::new(bits[0], bits[1]));
        }
    }

    /// A span of three segments (two coordinates, the fourth segment zero) at `d = 3` and of
    /// two at `d = 10` (two blocks): the rows of segment `sigma` are `sigma + 2^span_log j`,
    /// the shares are the segments' composites, and the region points with a segment
    /// coordinate evaluate the segment's regions (parity copies agree, inputs and outputs
    /// are the segment's own).
    #[test]
    fn spans_of_segments_share_the_layout_and_address_each_segment() {
        let mats = Arc::new(derive_matrices_for(&Shape::PRODUCTION, &sd(13)));
        let mut rng = Rng::new(13);
        for (d, segments, span_log) in [(3u32, 3usize, 2usize), (10, 2, 1)] {
            let bu = BridgeUnion::production_span(d, 0, span_log, Profile::Fast);
            let one = BridgeUnion::production_part(d, 0, Profile::Fast);
            assert_eq!(bu.span(), 1 << span_log);
            let c = bu.cost();
            assert_eq!(c.rows, [one.cost().rows[0] << span_log, 1 << span_log]);
            assert_eq!(
                c.nu,
                (one.level1().rows_log() + span_log).max(one.cost().nu)
            );
            let seg = *bu.segment();
            let per: Vec<Vec<u64>> = (0..segments).map(|_| random_bits(&seg, &mut rng)).collect();
            let bits: Vec<u64> = per.concat();
            let blocks = bu.honest_blocks(&mats, &bits);
            let shares = bu.shares(&mats, &blocks);
            assert_eq!(shares.len(), 1 << span_log);
            for (sigma, share) in shares.iter().enumerate() {
                let want = if sigma < segments {
                    gf2_walker::composite(&seg, &mats, &per[sigma])
                } else {
                    vec![0; 4096]
                };
                assert_eq!(*share, want, "d {d}, segment {sigma}");
            }
            // Rows of segment 1 are the single-segment rows of its bits.
            let single = one.honest_blocks(&mats, &per[1]);
            let l1 = bu.level1();
            for t in 0..6 {
                for beta in 0..l1.blocks() {
                    assert_eq!(
                        blocks.rows[bu.row(1, t, beta)],
                        single.rows[l1.row(t, beta)]
                    );
                }
            }
            assert_eq!(blocks.tops[1], single.tops[0]);
            // Region points at a Boolean segment coordinate: segment 1.
            let bufs = bu.buffers(&blocks, None);
            let zs: Vec<&[F128]> = bufs.iter().map(|b| b.z.as_slice()).collect();
            let union_z = scatter_for_test(bu.geometry(), &zs);
            let nu = bu.geometry().nu();
            let mut sigma = vec![F128::ZERO; span_log];
            sigma[0] = F128::ONE;
            let rand = |rng: &mut Rng, k: usize| -> Vec<F128> {
                (0..k).map(|_| random_f128(rng)).collect()
            };
            let beta = rand(&mut rng, l1.blocks_log());
            let lane = [random_f128(&mut rng), F128::ZERO, F128::ONE];
            let y = rand(&mut rng, 6);
            let v_y1 = eval_packed(&union_z, &bu.y1_point(&sigma, &beta, &lane, &y), nu);
            let v_top = eval_packed(&union_z, &bu.top_point(&sigma, &y, &beta, &lane), nu);
            assert_eq!(v_y1, v_top);
            let sbufs = one.buffers(&single, None);
            let szs: Vec<&[F128]> = sbufs.iter().map(|b| b.z.as_slice()).collect();
            let single_z = scatter_for_test(one.geometry(), &szs);
            let snu = one.geometry().nu();
            assert_eq!(
                v_y1,
                eval_packed(&single_z, &one.y1_point(&[], &beta, &lane, &y), snu)
            );
            let cols = rand(&mut rng, l1.block_len().trailing_zeros() as usize - 7);
            assert_eq!(
                eval_packed(&union_z, &bu.input_point(&sigma, &beta, &lane, &cols), nu),
                eval_packed(&single_z, &one.input_point(&[], &beta, &lane, &cols), snu)
            );
            // The output region of the honest blocks is zero (Y is the claim, not a column).
            let o = rand(&mut rng, 5);
            let rs = rand(&mut rng, span_log);
            assert_eq!(
                eval_packed(&union_z, &bu.output_point(&rs, &o), nu),
                F128::ZERO
            );
        }
    }

    /// The union `z` (test helper, as `UnionGeometry` scatters it).
    fn scatter_for_test(g: &UnionGeometry, zs: &[&[F128]]) -> Vec<F128> {
        let mut out = vec![F128::ZERO; 1 << (g.m_total() - 7)];
        for (i, z) in zs.iter().enumerate() {
            let start = g.slot(i).offset >> 7;
            out[start..start + z.len()].copy_from_slice(z);
        }
        out
    }

    /// The toy bridge union proven by Flock (two boolean types in one union, `M = 22`) with
    /// two claims; a tampered parity (its row broken) and a wrong share are rejected.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "Flock proof; run with --release")]
    fn toy_bridge_union_round_trip() {
        let (bu, mats, bits) = toy();
        let blocks = bu.honest_blocks(&mats, &bits);
        let y = bu.share(&mats, &blocks);
        let g = bu.geometry();
        let mut rng = Rng::new(12);
        let points: Vec<Vec<F128>> = (0..2)
            .map(|i| {
                let rows: Vec<F128> = (0..2).map(|_| random_f128(&mut rng)).collect();
                g.region_point(i, &rows, &[random_f128(&mut rng)])
            })
            .collect();
        let prove = |blocks: &BridgeBlocks, y: &[u8], pre: bool| {
            let (s1, s2) = bu.slots(sd(3), &mats, y.to_vec());
            let bufs = bu.buffers(blocks, Some((&s1, &s2)));
            let data = pre.then(|| {
                let zs: Vec<&[F128]> = bufs.iter().map(|b| b.z.as_slice()).collect();
                g.commit_data(&zs)
            });
            let mut ch = FsChallenger::new(b"test/bridge-union");
            ch.observe_bytes(&s1.digest());
            let circuits: [&dyn LincheckCircuit; 2] = [&s1, &s2];
            g.prove(bufs, &circuits, &points, &[], &mut ch, data)
        };
        let verify = |cap: &crate::prover::KeccakCommitment,
                      proof: &crate::flock_union::UnionProof,
                      y: &[u8]| {
            let (s1, s2) = bu.slots(sd(3), &mats, y.to_vec());
            let mut ch = FsChallenger::new(b"test/bridge-union");
            ch.observe_bytes(&s1.digest());
            let circuits: [&dyn LincheckCircuit; 2] = [&s1, &s2];
            g.verify(&circuits, cap, proof, &points, &[], &mut ch)
        };
        let (proof, cap) = prove(&blocks, &y, false);
        verify(&cap, &proof, &y).expect("honest toy union");
        // The serialized proof leaves out the level-0 cap (the commitment); the parsed proof
        // verifies against the commitment and not against another one.
        let bytes = proof.to_bytes();
        let parsed = crate::flock_union::UnionProof::from_bytes(&bytes, 2).expect("parse");
        assert!(parsed.flock.pcs_open.inner.ligerito.initial_cap.is_empty());
        assert!(!proof.flock.pcs_open.inner.ligerito.initial_cap.is_empty());
        assert_eq!(
            bytes.len() + cap.as_bytes().len(),
            4 + crate::prover::codec()
                .serialized_size(&proof.flock)
                .unwrap() as usize
                + 32,
            "the cap's bytes are saved"
        );
        // A proof that carries its level-0 cap (e.g. serialized with its cap intact) is
        // rejected on parse (K17).
        let raw_with_cap = crate::prover::codec().serialize(&proof.flock).unwrap();
        let mut malformed_with_cap = Vec::new();
        malformed_with_cap.extend_from_slice(&(raw_with_cap.len() as u32).to_le_bytes());
        malformed_with_cap.extend_from_slice(&raw_with_cap);
        for v in &proof.values {
            malformed_with_cap.extend_from_slice(&v.lo.to_le_bytes());
            malformed_with_cap.extend_from_slice(&v.hi.to_le_bytes());
        }
        assert!(matches!(
            crate::flock_union::UnionProof::from_bytes(&malformed_with_cap, 2),
            Err(crate::prover::KeccakError::Malformed("union proof carries its level-0 cap"))
        ));
        verify(&cap, &parsed, &y).expect("parsed toy union");
        let mut other = cap.as_bytes().to_vec();
        other[5] ^= 1;
        let other = crate::prover::KeccakCommitment::from_bytes(other);
        assert!(verify(&other, &parsed, &y).is_err());
        // The precomputed commitment gives the same proof (K14).
        let (again, cap2) = prove(&blocks, &y, true);
        assert_eq!((again, cap2.clone()), (proof.clone(), cap.clone()));
        let mut wrong = y.clone();
        wrong[0] ^= 1;
        assert!(verify(&cap, &proof, &wrong).is_err());
        let mut bad = blocks.clone();
        let c = bu.level1().y1_col(5);
        bad.rows[1][c / 64] ^= 1 << (c % 64);
        let (proof, cap) = prove(&bad, &y, false);
        assert!(verify(&cap, &proof, &y).is_err());
    }
}
