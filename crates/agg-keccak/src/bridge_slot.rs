//! The bridge slot: one Flock instance per segment whose rows prove `Y_sigma = M_3 M_2 M_1 b'`
//! over GF(2) (docs/bridge.md Section 5, spec Section 8, decision B.13).
//!
//! Block layout (`Lambda = lane_len`, `lanes * Lambda` input columns, deviation K8):
//!
//! | Columns | Committed | Row (`C = I`: `(A z)_c (B z)_c = z_c`) |
//! | --- | --- | --- |
//! | `c = p + Lambda t`, live (`p mod Lambda < live_len`) | the bit `b'_t(p)` | `z_c * 1 = z_c` (`A = e_c`, `B = e_one`) |
//! | `c = p + Lambda t`, padded | 0 | empty row (forces `z_c = 0`) |
//! | `one = lanes * Lambda` | 1 (Flock's constant pin) | `1 * 1 = 1` |
//! | `y_o = one + 128 + o`, `o < m_3` | 0 | `(sum_c M[o, c] z_c + Y_o z_one) * z_one = z_{y_o}` |
//! | all others | 0 | empty rows |
//!
//! `M = M_3 M_2 M_1` is the segment's composite ([`agg_bridge::gf2_walker`]); it depends
//! on the bridge seed `sd` and is never materialized. `Y_o` enters the relation as the
//! coefficient of the constant column, so the committed witness (bit copies, the constant,
//! zeros) is independent of `sd` and `Y` and can be committed at event 2, before `sd` exists.
//! The relation binds `Y`: a satisfying witness has `M b' = Y` (characteristic 2). The type
//! carries empty matrix stubs (Flock's walker convention, deviation K2); [`BridgeSlot`]
//! implements `LincheckCircuit` with the factored walker and its [`BridgeSlot::digest`]
//! (covering `sd`, the shape, the layout and `Y`) is absorbed into the Flock challenger.

use std::sync::Arc;

use agg_bridge::gf2_walker::{self, SegmentLayout};
use agg_bridge::shape::Shape;
use agg_witness::bridge::BridgeMatrices;
use flock_core::{
    lincheck::LincheckCircuit,
    r1cs::SparseBinaryMatrix,
    schedule::{IoWord, TableClass, TableType},
};
use flock_field::F128;
use rayon::prelude::*;
use sha3::{Digest, Sha3_256};

/// Digest label of the bridge relation.
pub const DIGEST_LABEL: &[u8] = b"agg-keccak/v1/bridge-slot";
/// Positions per record and lane (`256 j`-coefficients times 4 polynomials).
pub const RECORD_POSITIONS: usize = 1024;
/// 128-bit words per record and lane.
pub const RECORD_WORDS: usize = RECORD_POSITIONS / 128;

/// Column layout of the bridge type for one segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotLayout {
    /// The segment's input geometry (shape, lanes, `Lambda`).
    pub segment: SegmentLayout,
    /// Live positions per lane: columns `[Lambda t, Lambda t + live_len)` have identity
    /// rows, the rest of each lane is padded (forced zero).
    pub live_len: usize,
}

impl SlotLayout {
    /// A slot with `live_len` live positions per lane (a multiple of 128).
    ///
    /// # Panics
    /// If `live_len` exceeds the lane or is not word aligned.
    #[must_use]
    pub fn new(segment: SegmentLayout, live_len: usize) -> Self {
        assert!(
            live_len <= segment.lane_len && live_len.is_multiple_of(128),
            "live positions are a word-aligned prefix of each lane"
        );
        Self { segment, live_len }
    }

    /// The production slot of a segment of `2^{d_seg}` record positions with `live` live
    /// records (`Lambda = 2^{10 + d_seg}`, `live_len = 1024 live`).
    ///
    /// # Panics
    /// If `live > 2^{d_seg}` or `d_seg > 13`.
    #[must_use]
    pub fn production(d_seg: u32, live: usize) -> Self {
        assert!(
            live <= 1 << d_seg,
            "more live records than the segment holds"
        );
        Self::new(SegmentLayout::production(d_seg), RECORD_POSITIONS * live)
    }

    /// The shape.
    #[must_use]
    pub fn shape(&self) -> &Shape {
        &self.segment.shape
    }

    /// Input columns `lanes * Lambda`.
    #[must_use]
    pub fn input_bits(&self) -> usize {
        self.segment.input_bits()
    }

    /// The constant-one column.
    #[must_use]
    pub fn const_col(&self) -> usize {
        self.input_bits()
    }

    /// First row of the composite rows (`y_0`).
    #[must_use]
    pub fn y_base(&self) -> usize {
        self.const_col() + 128
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

    /// `log2` of the block width.
    #[must_use]
    pub fn k_log(&self) -> usize {
        (self.useful_bits().next_power_of_two().trailing_zeros() as usize).max(7)
    }

    /// u64 words of a block's committed data.
    #[must_use]
    pub fn useful_words(&self) -> usize {
        self.useful_bits().div_ceil(64)
    }

    /// Whether input column `c` is live (has an identity row).
    #[inline]
    #[must_use]
    pub fn is_live(&self, c: usize) -> bool {
        c < self.input_bits() && c % self.segment.lane_len < self.live_len
    }

    /// Live input columns.
    #[must_use]
    pub fn live_inputs(&self) -> usize {
        self.segment.lanes() * self.live_len
    }

    /// Word column of word `w < 8` of lane `t` of record `i` (production layout):
    /// `(Lambda t + 1024 i) / 128 + w`.
    #[must_use]
    pub fn record_word(&self, t: usize, i: usize, w: usize) -> usize {
        (self.segment.column(t, RECORD_POSITIONS * i)) / 128 + w
    }

    /// Live records (`live_len / 1024`; production layouts).
    #[must_use]
    pub fn live_records(&self) -> usize {
        self.live_len / RECORD_POSITIONS
    }

    /// The IO schema: every live input word, lane-major, then record, then word (`In`). The
    /// copy constraints to the sponge rows name these words; there are no outputs (`Y` is a
    /// coefficient of the relation, deviation K8).
    #[must_use]
    pub fn io_schema(&self) -> Vec<IoWord> {
        let lanes = self.segment.lanes();
        let words = self.live_len / 128;
        let lane_words = self.segment.lane_len / 128;
        (0..lanes)
            .flat_map(|t| (0..words).map(move |w| IoWord::input(lane_words * t + w)))
            .collect()
    }

    /// Schema index of word `w` of lane `t` of record `i` in [`Self::io_schema`].
    #[must_use]
    pub fn schema_index(&self, t: usize, i: usize, w: usize) -> usize {
        t * (self.live_len / 128) + RECORD_WORDS * i + w
    }

    /// The Flock registry type: empty matrix stubs, the constant pin and the IO schema.
    #[must_use]
    pub fn table_type(&self) -> TableType {
        let stub = || SparseBinaryMatrix::new(0, 0, Vec::new());
        TableType {
            k_log: self.k_log(),
            useful_bits: self.useful_bits(),
            a_0: stub(),
            b_0: stub(),
            c_0: stub(),
            const_pin: Some(self.const_col()),
            class: TableClass::Boolean,
            io_schema: self.io_schema(),
        }
    }

    /// The honest committed block for lane-major packed input bits (`input_bits / 64` words):
    /// the live bits, the constant, zeros elsewhere (padded positions of `bits` are dropped).
    ///
    /// # Panics
    /// If `bits` has the wrong length.
    #[must_use]
    pub fn honest_z(&self, bits: &[u64]) -> Vec<u64> {
        assert_eq!(bits.len(), self.input_bits() / 64, "packed input length");
        let mut z = vec![0_u64; self.useful_words()];
        let (lane_words, live_words) = (self.segment.lane_len / 64, self.live_len / 64);
        for t in 0..self.segment.lanes() {
            let at = lane_words * t;
            z[at..at + live_words].copy_from_slice(&bits[at..at + live_words]);
        }
        let one = self.const_col();
        z[one / 64] |= 1 << (one % 64);
        z
    }

    /// Structural nonzeros of `A_0 + B_0 + C_0` other than the composite rows' `A` part:
    /// identity rows (3 each), the constant row (3), and per composite row `B`, `C` and the
    /// constant coefficient when `Y_o = 1` (`y_weight` of them).
    #[must_use]
    pub fn identity_nonzeros(&self, y_weight: usize) -> u64 {
        (3 * self.live_inputs() + 3 + 2 * self.segment.output_bits() + y_weight) as u64
    }
}

/// The bridge relation of one segment: layout, the matrices of `sd`, and the public `Y`.
#[derive(Clone)]
pub struct BridgeSlot {
    layout: SlotLayout,
    sd: [u8; 32],
    matrices: Arc<BridgeMatrices>,
    y: Vec<u8>,
}

impl std::fmt::Debug for BridgeSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BridgeSlot")
            .field("layout", &self.layout)
            .field("sd", &self.sd)
            .finish_non_exhaustive()
    }
}

/// `Y` (one bit per byte) packed least significant first, bit `o` at byte `o / 8`.
#[must_use]
pub fn pack_y(y: &[u8]) -> Vec<u8> {
    let mut out = vec![0_u8; y.len().div_ceil(8)];
    for (o, &bit) in y.iter().enumerate() {
        out[o / 8] |= (bit & 1) << (o % 8);
    }
    out
}

impl BridgeSlot {
    /// The relation for `sd` (whose matrices are `matrices`) and the public `Y` (one bit per
    /// byte, `m_3` entries, each 0 or 1).
    ///
    /// # Panics
    /// If the matrices have another shape than the layout or `Y` is malformed.
    #[must_use]
    pub fn new(
        layout: SlotLayout,
        sd: [u8; 32],
        matrices: Arc<BridgeMatrices>,
        y: Vec<u8>,
    ) -> Self {
        assert!(
            layout.shape().matches(&matrices),
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

    /// The honest `Y = M b'` of a committed block `z` (its input columns).
    #[must_use]
    pub fn parities(layout: &SlotLayout, matrices: &BridgeMatrices, z: &[u64]) -> Vec<u8> {
        gf2_walker::composite(&layout.segment, matrices, &z[..layout.input_bits() / 64])
    }

    #[must_use]
    pub fn layout(&self) -> &SlotLayout {
        &self.layout
    }

    #[must_use]
    pub fn y(&self) -> &[u8] {
        &self.y
    }

    #[must_use]
    pub fn matrices(&self) -> &BridgeMatrices {
        &self.matrices
    }

    /// Row values `a = A_0 z`, `b = B_0 z` of a block `z` (`useful_words` u64 words; `a`, `b`
    /// zeroed on entry). For an honest block every composite row has `a = 0`, `b = 1`.
    pub fn apply_ab(&self, z: &[u64], a: &mut [u64], b: &mut [u64]) {
        let lay = &self.layout;
        let w = lay.useful_words();
        assert!(z.len() >= w && a.len() >= w && b.len() >= w, "block length");
        let get = |c: usize| (z[c / 64] >> (c % 64)) & 1;
        let one = get(lay.const_col());
        let (lane_words, live_words) = (lay.segment.lane_len / 64, lay.live_len / 64);
        for t in 0..lay.segment.lanes() {
            let at = lane_words * t;
            a[at..at + live_words].copy_from_slice(&z[at..at + live_words]);
            b[at..at + live_words].fill(one.wrapping_neg());
        }
        let c = lay.const_col();
        a[c / 64] |= one << (c % 64);
        b[c / 64] |= one << (c % 64);
        let mz = Self::parities(lay, &self.matrices, z);
        for (o, (&m, &y)) in mz.iter().zip(&self.y).enumerate() {
            let col = lay.y_col(o);
            let row = u64::from(m) ^ (u64::from(y) & one);
            a[col / 64] |= row << (col % 64);
            b[col / 64] |= one << (col % 64);
        }
    }

    /// SHA3-256 of the relation: label, `sd`, the shape (`(J_t, used, m_t, k_t)`, lanes,
    /// `B_1`), the layout (`Lambda`, live positions, `k_log`, useful bits, constant column,
    /// first composite row) and `Y` packed (docs/bridge.md Section 5, Binding).
    #[must_use]
    pub fn digest(&self) -> [u8; 32] {
        let lay = &self.layout;
        let shape = lay.shape();
        let mut h = Sha3_256::new();
        let put = |h: &mut Sha3_256, v: usize| h.update((v as u64).to_le_bytes());
        h.update((DIGEST_LABEL.len() as u64).to_le_bytes());
        h.update(DIGEST_LABEL);
        h.update(self.sd);
        for level in &shape.levels {
            put(&mut h, level.t as usize);
            put(&mut h, level.cols);
            put(&mut h, level.used);
            put(&mut h, level.rows);
            put(&mut h, level.k);
        }
        put(&mut h, shape.lanes);
        put(&mut h, shape.blocks1);
        put(&mut h, lay.segment.lane_len);
        put(&mut h, lay.live_len);
        put(&mut h, lay.k_log());
        put(&mut h, lay.useful_bits());
        put(&mut h, lay.const_col());
        put(&mut h, lay.y_base());
        put(&mut h, self.y.len());
        h.update(pack_y(&self.y));
        h.finalize().into()
    }

    /// `(xi_A, xi_B)` restricted to their nonzero parts: `xi_A` on the input columns (the
    /// walker plus the identity rows), and the constant column's two entries.
    fn marginals(&self, eq: &[F128], xi_a_inputs: &mut [F128]) -> (F128, F128) {
        let lay = &self.layout;
        assert_eq!(
            eq.len(),
            1 << lay.k_log(),
            "eq length must equal the block width"
        );
        let m3 = lay.segment.output_bits();
        let v3: Vec<F128> = (0..m3).map(|o| eq[lay.y_col(o)]).collect();
        gf2_walker::transpose_into(&lay.segment, &self.matrices, &v3, xi_a_inputs);
        let lane = lay.segment.lane_len;
        let live = lay.live_len;
        // Identity rows: A = e_c on live inputs; their B = e_one is summed below.
        let live_sum = xi_a_inputs
            .par_chunks_mut(lane)
            .zip(eq[..lay.input_bits()].par_chunks(lane))
            .map(|(x, e)| {
                let mut s = F128::ZERO;
                for (xc, &ec) in x[..live].iter_mut().zip(&e[..live]) {
                    *xc += ec;
                    s += ec;
                }
                s
            })
            .reduce(|| F128::ZERO, |p, q| p + q);
        let one = eq[lay.const_col()];
        let (mut a_one, mut b_one) = (one, one + live_sum);
        for (o, &e) in v3.iter().enumerate() {
            b_one += e;
            if self.y[o] == 1 {
                a_one += e;
            }
        }
        (a_one, b_one)
    }
}

impl LincheckCircuit for BridgeSlot {
    fn n_cols(&self) -> usize {
        1 << self.layout.k_log()
    }

    fn const_pin_col(&self) -> Option<usize> {
        Some(self.layout.const_col())
    }

    /// `comb[c] = alpha xi_A(c) + xi_B(c)`: the factored walker on the input columns, the
    /// identity rows, and the constant column (docs/bridge.md Section 5).
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

    /// Both marginals from one walk.
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

#[cfg(test)]
mod tests {
    use super::*;
    use agg_bridge::seed::derive_matrices_for;
    use flock_core::lincheck::SparseMatrixCircuit;
    use flock_core::test_rng::Rng;

    fn random_f128(rng: &mut Rng) -> F128 {
        F128::new(rng.next_u64(), rng.next_u64())
    }

    fn sd(byte: u8) -> [u8; 32] {
        std::array::from_fn(|i| byte.wrapping_mul(i as u8 + 1))
    }

    /// A toy slot (`L = 2^12`, one lane) with a live prefix, honest `Y` for random bits.
    fn toy_slot(live_len: usize, seed: u64) -> (BridgeSlot, Vec<u64>) {
        let layout = SlotLayout::new(SegmentLayout::new(Shape::TOY, 1 << 12), live_len);
        let mats = Arc::new(derive_matrices_for(&Shape::TOY, &sd(seed as u8)));
        let mut rng = Rng::new(seed);
        let bits: Vec<u64> = (0..64).map(|_| rng.next_u64()).collect();
        let z = layout.honest_z(&bits);
        let y = BridgeSlot::parities(&layout, &mats, &z);
        (BridgeSlot::new(layout, sd(seed as u8), mats, y), z)
    }

    fn bit(v: &[u64], c: usize) -> bool {
        (v[c / 64] >> (c % 64)) & 1 == 1
    }

    #[test]
    fn toy_layout_has_the_documented_columns() {
        let (slot, _) = toy_slot(4096, 1);
        let lay = slot.layout();
        assert_eq!(lay.const_col(), 4096);
        assert_eq!(lay.y_base(), 4224);
        assert_eq!(lay.useful_bits(), 4240);
        assert_eq!(lay.k_log(), 13);
        let p = SlotLayout::production(13, 1 << 13);
        assert_eq!(
            (p.const_col(), p.useful_bits(), p.k_log()),
            (6 << 23, 50_335_872, 26)
        );
        assert_eq!(p.io_schema().len(), 393_216);
        assert_eq!(p.record_word(2, 5, 3), (2 << 23) / 128 + 8 * 5 + 3);
        let q = SlotLayout::production(6, 40);
        assert_eq!(q.k_log(), 19);
        assert_eq!(
            q.io_schema()[q.schema_index(5, 39, 7)].word_col,
            q.record_word(5, 39, 7)
        );
    }

    /// The honest block satisfies every row: `a & b = z` word by word, and every composite
    /// row has `a = 0`, `b = 1`; a flipped committed bit or a flipped `Y` bit breaks a row.
    #[test]
    fn honest_rows_satisfy_the_relation_and_tampering_breaks_a_row() {
        for live in [4096, 2048] {
            let (slot, z) = toy_slot(live, 2);
            let lay = *slot.layout();
            let w = lay.useful_words();
            let rows_ok = |slot: &BridgeSlot, z: &[u64]| {
                let (mut a, mut b) = (vec![0_u64; w], vec![0_u64; w]);
                slot.apply_ab(z, &mut a, &mut b);
                (0..w).all(|i| a[i] & b[i] == z[i])
            };
            assert!(rows_ok(&slot, &z));
            // A flipped live bit: the composite rows see it (unless its column vanishes).
            let rows = gf2_walker::dense_rows(&lay.segment, slot.matrices());
            for c in [0, 17, live - 1] {
                let mut bad = z.clone();
                bad[c / 64] ^= 1 << (c % 64);
                let visible = rows.iter().any(|r| bit(r, c));
                assert_eq!(rows_ok(&slot, &bad), !visible, "column {c}");
            }
            // A one on a padded position or on a composite-row column breaks its row.
            let mut ones = vec![lay.y_col(3), lay.const_col() + 1];
            if live < 4096 {
                ones.extend([live, 4095]);
            }
            for c in ones {
                assert!(!lay.is_live(c) && c != lay.const_col());
                {
                    let mut bad = z.clone();
                    bad[c / 64] |= 1 << (c % 64);
                    let mut fixed = slot.clone();
                    // Keep Y consistent with the tampered block so only row c can fail.
                    fixed.y = BridgeSlot::parities(&lay, slot.matrices(), &bad);
                    assert!(!rows_ok(&fixed, &bad), "column {c}");
                }
            }
            // A flipped Y bit.
            let mut wrong = slot.clone();
            wrong.y[5] ^= 1;
            assert!(!rows_ok(&wrong, &z));
        }
    }

    /// `<comb, z> = alpha <eq, A z> + <eq, B z>` for random, not satisfying, blocks: the
    /// walker is the transpose of the row evaluator (toy and a production slot).
    fn check_walker_adjoint(slot: &BridgeSlot, seed: u64) {
        let lay = *slot.layout();
        let k = 1usize << lay.k_log();
        let w = lay.useful_words();
        let mut rng = Rng::new(seed);
        let eq: Vec<F128> = (0..k).map(|_| random_f128(&mut rng)).collect();
        let alpha = random_f128(&mut rng);
        let comb = slot.fold_alpha_batched(alpha, &eq);
        assert!(comb[lay.useful_bits()..].iter().all(|&c| c == F128::ZERO));
        for trial in 0..4 {
            let mut z: Vec<u64> = (0..w).map(|_| rng.next_u64()).collect();
            if trial % 2 == 0 {
                let c = lay.const_col();
                z[c / 64] |= 1 << (c % 64);
            }
            let tail = lay.useful_bits() % 64;
            if tail != 0 {
                z[w - 1] &= (1 << tail) - 1;
            }
            let (mut a, mut b) = (vec![0_u64; w], vec![0_u64; w]);
            slot.apply_ab(&z, &mut a, &mut b);
            let (mut lhs, mut rhs) = (F128::ZERO, F128::ZERO);
            for col in 0..lay.useful_bits() {
                if bit(&z, col) {
                    lhs += comb[col];
                }
                if bit(&a, col) {
                    rhs += alpha * eq[col];
                }
                if bit(&b, col) {
                    rhs += eq[col];
                }
            }
            assert_eq!(lhs, rhs, "trial {trial}");
        }
    }

    #[test]
    fn walker_is_the_transpose_of_the_row_evaluator() {
        let (toy, _) = toy_slot(2048, 3);
        check_walker_adjoint(&toy, 3);
        let layout = SlotLayout::production(3, 5);
        let mats = Arc::new(derive_matrices_for(&Shape::PRODUCTION, &sd(4)));
        let y: Vec<u8> = (0..4096).map(|o| (o % 3 == 0) as u8).collect();
        check_walker_adjoint(&BridgeSlot::new(layout, sd(4), mats, y), 4);
    }

    /// `fold_split` equals the two marginals that the trait's default derives from two
    /// `fold_alpha_batched` calls.
    #[test]
    fn fold_split_agrees_with_two_alpha_folds() {
        let (slot, _) = toy_slot(4096, 5);
        let mut rng = Rng::new(6);
        let eq: Vec<F128> = (0..slot.n_cols()).map(|_| random_f128(&mut rng)).collect();
        let b = slot.fold_alpha_batched(F128::ZERO, &eq);
        let mut a = slot.fold_alpha_batched(F128::ONE, &eq);
        for (x, y) in a.iter_mut().zip(&b) {
            *x += *y;
        }
        assert_eq!(slot.fold_split(&eq), (a, b));
    }

    /// Explicit `A_0`, `B_0` of a slot (rows indexed by their C column), from the dense
    /// composite: the matrices the walker never builds.
    fn lowered(slot: &BridgeSlot) -> (SparseBinaryMatrix, SparseBinaryMatrix) {
        let lay = *slot.layout();
        let k = 1usize << lay.k_log();
        let one = lay.const_col();
        let dense = gf2_walker::dense_rows(&lay.segment, slot.matrices());
        let mut a_rows = vec![Vec::new(); k];
        let mut b_rows = vec![Vec::new(); k];
        for c in (0..lay.input_bits()).filter(|&c| lay.is_live(c)) {
            a_rows[c].push(c);
            b_rows[c].push(one);
        }
        a_rows[one].push(one);
        b_rows[one].push(one);
        for (o, row) in dense.iter().enumerate() {
            let r = lay.y_col(o);
            a_rows[r] = (0..lay.input_bits()).filter(|&c| bit(row, c)).collect();
            if slot.y()[o] == 1 {
                a_rows[r].push(one);
            }
            b_rows[r].push(one);
        }
        (
            SparseBinaryMatrix::new(k, k, a_rows),
            SparseBinaryMatrix::new(k, k, b_rows),
        )
    }

    fn check_against_sparse(slot: &BridgeSlot, seed: u64) {
        let (a0, b0) = lowered(slot);
        let reference = SparseMatrixCircuit::new(&a0, &b0).with_const_pin(slot.const_pin_col());
        let mut rng = Rng::new(seed);
        let eq: Vec<F128> = (0..slot.n_cols()).map(|_| random_f128(&mut rng)).collect();
        let alpha = random_f128(&mut rng);
        assert_eq!(reference.n_cols(), slot.n_cols());
        assert_eq!(
            slot.fold_alpha_batched(alpha, &eq),
            reference.fold_alpha_batched(alpha, &eq)
        );
        assert_eq!(slot.fold_split(&eq), reference.fold_split(&eq));
    }

    /// The walker equals Flock's `SparseMatrixCircuit` fold on the explicitly lowered rows:
    /// the toy slot (`2^13` columns) and a production slot of one record (`2^14` columns,
    /// 4,096 dense composite rows over 6,144 columns).
    #[test]
    #[cfg_attr(debug_assertions, ignore = "dense lowering; run with --release")]
    fn walker_equals_flock_sparse_fold_on_lowered_rows() {
        let (toy, _) = toy_slot(3072, 7);
        check_against_sparse(&toy, 7);
        let layout = SlotLayout::production(0, 1);
        let mats = Arc::new(derive_matrices_for(&Shape::PRODUCTION, &sd(8)));
        let mut rng = Rng::new(8);
        let bits: Vec<u64> = (0..layout.input_bits() / 64)
            .map(|_| rng.next_u64())
            .collect();
        let z = layout.honest_z(&bits);
        let y = BridgeSlot::parities(&layout, &mats, &z);
        check_against_sparse(&BridgeSlot::new(layout, sd(8), mats, y), 9);
    }

    #[test]
    fn digest_binds_seed_layout_and_y() {
        let (slot, _) = toy_slot(4096, 9);
        let d = slot.digest();
        let mut other = slot.clone();
        other.sd[0] ^= 1;
        assert_ne!(other.digest(), d);
        let mut other = slot.clone();
        other.y[0] ^= 1;
        assert_ne!(other.digest(), d);
        let mut other = slot.clone();
        other.layout.live_len = 2048;
        assert_ne!(other.digest(), d);
        assert_eq!(slot.clone().digest(), d);
    }
}
