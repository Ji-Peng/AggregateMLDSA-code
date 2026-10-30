//! The virtual parity chain (`docs/parity-chain.md`): the GF(2) side of the
//! bridge as one degree-2 sum-check over the committed bits, without the bridge union.
//!
//! In characteristic 2 the composite `M = M_3 M_2 M_1` is `GF(2^128)`-linear, so for a random
//! `eq_Y = eq(r_o, rho_3) eq(r_y, sigma)` the value `<eq_Y, Y>` of the public final vectors
//! equals `sum_x W(x) b''(x)` with the public weight `W = M^T eq_Y` and the committed bits
//! `b''`. The prover proves this inner product by a sum-check over the variables of `b''` in
//! two phases (block variables first, then the in-block position), never materializing `W`:
//!
//! * block phase: for every block point, `sum_zeta W~(beta, zeta) b''~(beta, zeta) =
//!   sum_{rho_1} G~(beta, rho_1) y1~(beta, rho_1)` with `y1 = M_1 b''` the honest level-1
//!   parities and `G = M_1`-side weight (`M_2^T M_3^T eq_Y`, scaled by the segment factor);
//!   the round messages come from the tables `G` and `y1` (`2^(block vars) x m_1` entries);
//! * in-block phase: `w = M_1^T G~(r_block, .)` and `bf = b''~(r_block, .)` over the block's
//!   positions, an ordinary product sum-check.
//!
//! The final claim is `T = W~(r) b''~(r)`; the verifier evaluates `W~(r)` itself
//! ([`weight_at`]) and the caller proves `b''~(r)` (a bit-level evaluation: a ring-switched
//! claim on the sponge commitment in `agg-keccak`).
//!
//! Generic over [`crate::Shape`] and transcript-agnostic: the caller's closure absorbs each
//! round message and returns the round challenge.
//!
//! Indices. A chain covers `2^span_log` segments of one part layout ([`SegmentLayout`],
//! `lane_len` positions per lane, part `offset`), each with `2^lb` lanes (`lb =
//! ceil(log2 lanes)`; the lanes `lanes..2^lb` are junk with weight zero).
//! The input bits are `2^span_log` segment blocks of `2^lb * lane_len` bits, lane-major:
//! bit `p + lane_len t` of a segment block is position `p` of lane `t`
//! ([`ChainLayout::bit_index`]). A position `p = zeta + block_len i_mid` has the in-block
//! index `zeta < block_len = min(lane_len, J_1)` and the block `i_mid < blocks`. The block
//! index is `beta = i_mid + blocks (t + 2^lb sigma)`; the sum-check binds the bits of `beta`
//! (low first: `i_mid`, `t`, `sigma`), then those of `zeta` (low first).

use agg_witness::bridge::{BridgeMatrices, SparseMatrix};
use flock_field::F128;
use rayon::prelude::*;

use crate::gf2_walker::SegmentLayout;

/// Below this many entries the serial paths are used.
const MIN_PARALLEL: usize = 1 << 12;

/// A zero-filled vector from the allocator's zeroed path. Large buffers come back as fresh
/// zero pages, so the first touch happens in the parallel writer instead of a serial fill.
fn zeroed_f128(n: usize) -> Vec<F128> {
    flock_core::alloc_zeroed_vec::<F128>(n)
}

/// `eq(point, bits(i))` for every `i < 2^|point|` over `GF(2^128)`, coordinate `j` on index
/// bit `j`.
#[must_use]
pub fn eq_table(point: &[F128]) -> Vec<F128> {
    let mut table = Vec::with_capacity(1 << point.len());
    table.push(F128::ONE);
    for &x in point {
        let len = table.len();
        table.resize(2 * len, F128::ZERO);
        let (low, high) = table.split_at_mut(len);
        let fill = |(lo, hi): (&mut F128, &mut F128)| {
            let scaled = *lo * x;
            *hi = scaled;
            *lo += scaled;
        };
        if len >= MIN_PARALLEL {
            low.par_iter_mut().zip(high.par_iter_mut()).for_each(fill);
        } else {
            low.iter_mut().zip(high.iter_mut()).for_each(fill);
        }
    }
    table
}

/// `eq(a, b) = prod_j (a_j b_j + (1 + a_j)(1 + b_j))`.
#[must_use]
pub fn eq_eval(a: &[F128], b: &[F128]) -> F128 {
    assert_eq!(a.len(), b.len(), "points of equal length");
    a.iter().zip(b).fold(F128::ONE, |acc, (&x, &y)| {
        acc * (x * y + (F128::ONE + x) * (F128::ONE + y))
    })
}

/// `q(r)` for the degree-2 round polynomial with `q(0) + q(1) = claim`, given `q(1)` and the
/// leading coefficient `q(inf)` (characteristic 2: `q(0) = claim + q(1)`).
#[must_use]
pub fn round_value(claim: F128, q1: F128, qinf: F128, r: F128) -> F128 {
    let q0 = claim + q1;
    let b = q1 + q0 + qinf;
    q0 + r * (b + qinf * r)
}

/// The geometry of one chain (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChainLayout {
    /// The part's input layout (shape, `lane_len`, offset).
    pub layout: SegmentLayout,
    /// `log2` of the segments the chain covers.
    pub span_log: usize,
}

/// The challenge point of a chain, by coordinate group (each least significant first).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChainPoint {
    /// In-block position (`log2 block_len` coordinates).
    pub zeta: Vec<F128>,
    /// Level-1 block within the part (`log2 blocks`).
    pub i_mid: Vec<F128>,
    /// Lane (`lb`).
    pub t: Vec<F128>,
    /// Segment (`span_log`).
    pub sigma: Vec<F128>,
}

impl ChainPoint {
    /// The block coordinates `i_mid || t || sigma` (the order of the block index `beta`).
    #[must_use]
    pub fn block(&self) -> Vec<F128> {
        self.i_mid
            .iter()
            .chain(&self.t)
            .chain(&self.sigma)
            .copied()
            .collect()
    }
}

/// The `eq_Y` challenges: `r_o` over the `m_3` output rows, `r_y` over the segments.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EqY {
    /// Output-row coordinates (`log2 m_3`).
    pub r_o: Vec<F128>,
    /// Segment coordinates (`span_log`).
    pub r_y: Vec<F128>,
}

/// A chain's round messages `(q(1), q(inf))`, its point and the final value `b''~(r)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChainProof {
    /// One message per round, block rounds first.
    pub messages: Vec<(F128, F128)>,
    /// The challenge point.
    pub point: ChainPoint,
    /// `b''~(point)`.
    pub value: F128,
    /// The running claim after the last round (`W~(r) b''~(r)` for an honest prover).
    pub last: F128,
}

impl ChainLayout {
    /// A chain over `2^span_log` segments of `layout`.
    #[must_use]
    pub fn new(layout: SegmentLayout, span_log: usize) -> Self {
        Self { layout, span_log }
    }

    /// `lb = ceil(log2 lanes)`.
    #[must_use]
    pub fn lane_vars(&self) -> usize {
        self.layout.lanes().next_power_of_two().trailing_zeros() as usize
    }

    /// `2^lb`: lanes of the chain's input, junk lanes included.
    #[must_use]
    pub fn lanes_padded(&self) -> usize {
        1 << self.lane_vars()
    }

    /// `log2 block_len`.
    #[must_use]
    pub fn zeta_vars(&self) -> usize {
        self.layout.block_len().trailing_zeros() as usize
    }

    /// `log2 blocks`.
    #[must_use]
    pub fn mid_vars(&self) -> usize {
        self.layout.blocks().trailing_zeros() as usize
    }

    /// Block variables `log2 blocks + lb + span_log`.
    #[must_use]
    pub fn block_vars(&self) -> usize {
        self.mid_vars() + self.lane_vars() + self.span_log
    }

    /// Rounds of the sum-check.
    #[must_use]
    pub fn rounds(&self) -> usize {
        self.block_vars() + self.zeta_vars()
    }

    /// Coordinates of `eq_Y`: `log2 m_3 + span_log`.
    #[must_use]
    pub fn eq_vars(&self) -> usize {
        self.out_vars() + self.span_log
    }

    /// `log2 m_3`.
    #[must_use]
    pub fn out_vars(&self) -> usize {
        self.layout.output_bits().trailing_zeros() as usize
    }

    /// Bits of one segment block of the input (`2^lb lane_len`).
    #[must_use]
    pub fn segment_bits(&self) -> usize {
        self.lanes_padded() * self.layout.lane_len
    }

    /// `u64` words of the whole input (`2^span_log` segment blocks).
    #[must_use]
    pub fn input_words(&self) -> usize {
        (self.segment_bits() << self.span_log) / 64
    }

    /// Bit index of position `p` of lane `t` of segment `sigma` in the input.
    #[must_use]
    pub fn bit_index(&self, sigma: usize, t: usize, p: usize) -> usize {
        sigma * self.segment_bits() + t * self.layout.lane_len + p
    }

    /// Bit index of the first position of block `beta` (`i_mid + blocks (t + 2^lb sigma)`).
    fn block_start(&self, beta: usize) -> usize {
        let blocks = self.layout.blocks();
        let (i_mid, rest) = (beta % blocks, beta / blocks);
        let (t, sigma) = (rest % self.lanes_padded(), rest / self.lanes_padded());
        self.bit_index(sigma, t, i_mid * self.layout.block_len())
    }

    /// Split a point in round order (block variables, then in-block) into its groups.
    #[must_use]
    pub fn split_point(&self, rs: &[F128]) -> ChainPoint {
        assert_eq!(rs.len(), self.rounds(), "one coordinate per round");
        let (mid, lb) = (self.mid_vars(), self.lane_vars());
        let nb = self.block_vars();
        ChainPoint {
            i_mid: rs[..mid].to_vec(),
            t: rs[mid..mid + lb].to_vec(),
            sigma: rs[mid + lb..nb].to_vec(),
            zeta: rs[nb..].to_vec(),
        }
    }

    /// The verifier's claim `<eq_Y, Y>` for the final vectors `ys` (one `m_3`-bit vector,
    /// one bit per byte, per segment `sigma < ys.len() <= 2^span_log`; the others are zero).
    ///
    /// # Panics
    /// On a vector of the wrong length or more vectors than segments.
    #[must_use]
    pub fn claim(&self, ys: &[Vec<u8>], eq_y: &EqY) -> F128 {
        self.check_eq(eq_y);
        assert!(ys.len() <= 1 << self.span_log, "at most 2^span_log vectors");
        let e_o = eq_table(&eq_y.r_o);
        let e_s = eq_table(&eq_y.r_y);
        ys.iter().enumerate().fold(F128::ZERO, |acc, (sigma, y)| {
            assert_eq!(y.len(), e_o.len(), "m_3 bits per final vector");
            let v = y
                .iter()
                .zip(&e_o)
                .filter(|(b, _)| **b & 1 == 1)
                .fold(F128::ZERO, |a, (_, &e)| a + e);
            acc + e_s[sigma] * v
        })
    }

    fn check_eq(&self, eq_y: &EqY) {
        assert_eq!(
            eq_y.r_o.len(),
            self.out_vars(),
            "log2 m_3 output coordinates"
        );
        assert_eq!(
            eq_y.r_y.len(),
            self.span_log,
            "span_log segment coordinates"
        );
    }

    /// `v2[t + lanes rho_2] = sum_{rho_3 in S_3(.)} eq(r_o, rho_3)` (level 3 transposed).
    fn level3(&self, mats: &BridgeMatrices, r_o: &[F128]) -> Vec<F128> {
        let e3 = eq_table(r_o);
        (0..mats.m3.cols)
            .into_par_iter()
            .with_min_len(1024)
            .map(|zeta3| gather(&mats.m3, zeta3, |r| e3[r]))
            .collect()
    }

    /// `G'(t, i_mid, rho_1)` for `t < lanes` (`[t][i_mid][rho_1]`, `lanes x blocks x m_1`).
    #[must_use]
    pub fn level2_weights(&self, mats: &BridgeMatrices, r_o: &[F128]) -> Vec<F128> {
        let v2 = self.level3(mats, r_o);
        let lanes = self.layout.lanes();
        let (rows1, blocks) = (mats.m1.rows, self.layout.blocks());
        let z2_base = self.layout.zeta2_base();
        let per_lane = blocks * rows1;
        (0..lanes * per_lane)
            .into_par_iter()
            .with_min_len(1024)
            .map(|i| {
                let (t, z2) = (i / per_lane, i % per_lane);
                gather(&mats.m2, z2_base + z2, |r| v2[t + lanes * r])
            })
            .collect()
    }

    /// `W~(r)`: the multilinear extension of the weight at the chain point,
    /// computed right to left without the level-2 table.
    ///
    /// # Panics
    /// On points or matrices of another shape.
    #[must_use]
    pub fn weight_at(&self, mats: &BridgeMatrices, eq_y: &EqY, point: &ChainPoint) -> F128 {
        self.check_eq(eq_y);
        assert!(
            self.layout.shape.matches(mats),
            "matrices of the layout's shape"
        );
        assert_eq!(point.zeta.len(), self.zeta_vars());
        assert_eq!(point.i_mid.len(), self.mid_vars());
        assert_eq!(point.t.len(), self.lane_vars());
        assert_eq!(point.sigma.len(), self.span_log);
        let lanes = self.layout.lanes();
        let e3 = eq_table(&eq_y.r_o);
        let e_t = eq_table(&point.t);
        // u2[rho_2] = sum_t eq(r_t, t) sum_{rho_3 in S_3(t + lanes rho_2)} e3[rho_3].
        let u2: Vec<F128> = (0..mats.m2.rows)
            .into_par_iter()
            .with_min_len(256)
            .map(|rho2| {
                (0..lanes).fold(F128::ZERO, |acc, t| {
                    acc + e_t[t] * gather(&mats.m3, t + lanes * rho2, |r| e3[r])
                })
            })
            .collect();
        // g[rho_1] = sum_{i_mid} eq(r_mid, i_mid) sum_{rho_2 in S_2(..)} u2[rho_2].
        let e_mid = eq_table(&point.i_mid);
        let rows1 = mats.m1.rows;
        let z2_base = self.layout.zeta2_base();
        let g: Vec<F128> = (0..rows1)
            .into_par_iter()
            .with_min_len(256)
            .map(|rho1| {
                e_mid
                    .iter()
                    .enumerate()
                    .fold(F128::ZERO, |acc, (i_mid, &e)| {
                        acc + e * gather(&mats.m2, z2_base + rho1 + rows1 * i_mid, |r| u2[r])
                    })
            })
            .collect();
        // sum_zeta eq(r_zeta, zeta) sum_{rho_1 in S_1(zeta1_base + zeta)} g[rho_1].
        let e_z = eq_table(&point.zeta);
        let base = self.layout.zeta1_base();
        let inner = e_z
            .par_iter()
            .enumerate()
            .with_min_len(1024)
            .map(|(zeta, &e)| e * gather(&mats.m1, base + zeta, |r| g[r]))
            .reduce(|| F128::ZERO, |a, b| a + b);
        eq_eval(&point.sigma, &eq_y.r_y) * inner
    }

    /// The words of the `2^lb` lanes of block `(sigma, i_mid)` at word `w` of the block.
    #[inline]
    fn lane_words(&self, bits: &[u64], sigma: usize, i_mid: usize, w: usize) -> [u64; 8] {
        let start = self.layout.block_len() * i_mid;
        std::array::from_fn(|t| {
            if t < self.lanes_padded() {
                bits[self.bit_index(sigma, t, start) / 64 + w]
            } else {
                0
            }
        })
    }

    /// The level-1 parities `y1 = M_1 b''` of every block of the input as `GF(2^128)` 0/1
    /// entries, `[beta][rho_1]`. The `2^lb <= 8` lanes of a position travel as one byte mask
    /// (14 byte XORs per position with a set bit, as `Level1Layout::parities`), one task per
    /// `(sigma, i_mid)`.
    fn level1_table(&self, m1: &SparseMatrix, bits: &[u64]) -> Vec<F128> {
        let n_blocks = 1usize << self.block_vars();
        let (blen, rows1) = (self.layout.block_len(), m1.rows);
        let (blocks, lanes_p) = (self.layout.blocks(), self.lanes_padded());
        let base = self.layout.zeta1_base();
        let masks: Vec<Vec<u8>> = (0..blocks << self.span_log)
            .into_par_iter()
            .map(|g| {
                let (sigma, i_mid) = (g / blocks, g % blocks);
                let mut m = vec![0_u8; rows1];
                for w in 0..blen / 64 {
                    let words = self.lane_words(bits, sigma, i_mid, w);
                    let mut any = words.iter().fold(0, |acc, x| acc | x);
                    while any != 0 {
                        let z = any.trailing_zeros() as usize;
                        any &= any - 1;
                        let mask = words
                            .iter()
                            .enumerate()
                            .fold(0_u8, |acc, (t, x)| acc | ((((x >> z) & 1) as u8) << t));
                        for &r in m1.column(base + 64 * w + z) {
                            m[r as usize] ^= mask;
                        }
                    }
                }
                m
            })
            .collect();
        let mut out = zeroed_f128(n_blocks * rows1);
        out.par_chunks_mut(rows1)
            .enumerate()
            .for_each(|(beta, row)| {
                let (i_mid, rest) = (beta % blocks, beta / blocks);
                let (t, sigma) = (rest % lanes_p, rest / lanes_p);
                let m = &masks[sigma * blocks + i_mid];
                for (o, &x) in row.iter_mut().zip(m) {
                    if (x >> t) & 1 == 1 {
                        *o = F128::ONE;
                    }
                }
            });
        out
    }

    /// `bf(zeta) = sum_beta eq(r_block, beta) b''(beta, zeta)` for every in-block position:
    /// per `(sigma, i_mid)` a 256-entry table of the lane weights' subset sums, looked up
    /// with the byte of the position's lane bits.
    fn block_fold(&self, bits: &[u64], e_b: &[F128]) -> Vec<F128> {
        let (blocks, lanes_p) = (self.layout.blocks(), self.lanes_padded());
        let groups = blocks << self.span_log;
        let tables: Vec<Vec<F128>> = (0..groups)
            .map(|g| {
                let (sigma, i_mid) = (g / blocks, g % blocks);
                let lane_e: [F128; 8] = std::array::from_fn(|t| {
                    if t < lanes_p {
                        e_b[i_mid + blocks * (t + lanes_p * sigma)]
                    } else {
                        F128::ZERO
                    }
                });
                let mut tab = vec![F128::ZERO; 256];
                for byte in 1..256_usize {
                    let low = byte & (byte - 1);
                    tab[byte] = tab[low] + lane_e[(byte ^ low).trailing_zeros() as usize];
                }
                tab
            })
            .collect();
        let mut bf = vec![F128::ZERO; self.layout.block_len()];
        bf.par_chunks_mut(64).enumerate().for_each(|(w, acc)| {
            for (g, tab) in tables.iter().enumerate() {
                let words = self.lane_words(bits, g / blocks, g % blocks, w);
                let mut any = words.iter().fold(0, |a, x| a | x);
                while any != 0 {
                    let z = any.trailing_zeros() as usize;
                    any &= any - 1;
                    let byte = words
                        .iter()
                        .enumerate()
                        .fold(0_usize, |a, (t, x)| a | ((((x >> z) & 1) as usize) << t));
                    acc[z] += tab[byte];
                }
            }
        });
        bf
    }

    /// Prove `sum_x W(x) b''(x) = claim` for the input `bits` (module docs) under the matrices
    /// and `eq_Y`: `challenge(round, (q(1), q(inf)))` absorbs the message and returns the
    /// round's challenge (or the caller's error).
    ///
    /// # Errors
    /// The first error `challenge` returns.
    ///
    /// # Panics
    /// If the input or the matrices do not fit the layout.
    pub fn prove<E>(
        &self,
        mats: &BridgeMatrices,
        bits: &[u64],
        eq_y: &EqY,
        claim: F128,
        mut challenge: impl FnMut(usize, (F128, F128)) -> Result<F128, E>,
    ) -> Result<ChainProof, E> {
        self.check_eq(eq_y);
        assert!(
            self.layout.shape.matches(mats),
            "matrices of the layout's shape"
        );
        assert_eq!(bits.len(), self.input_words(), "chain input length");
        let rows1 = mats.m1.rows;
        let (blocks, lanes, lanes_p) = (
            self.layout.blocks(),
            self.layout.lanes(),
            self.lanes_padded(),
        );
        // The block-phase tables: G[beta][rho_1] = eq(r_y, sigma) G'(t, i_mid, rho_1) (zero on
        // the junk lanes) and y1[beta][rho_1].
        let g2 = self.level2_weights(mats, &eq_y.r_o);
        let e_s = eq_table(&eq_y.r_y);
        let n_blocks = 1usize << self.block_vars();
        let mut gt = zeroed_f128(n_blocks * rows1);
        gt.par_chunks_mut(rows1)
            .enumerate()
            .for_each(|(beta, row)| {
                let (i_mid, rest) = (beta % blocks, beta / blocks);
                let (t, sigma) = (rest % lanes_p, rest / lanes_p);
                if t < lanes {
                    let src = &g2[(t * blocks + i_mid) * rows1..(t * blocks + i_mid + 1) * rows1];
                    let f = e_s[sigma];
                    for (o, &x) in row.iter_mut().zip(src) {
                        *o = f * x;
                    }
                }
            });
        drop(g2);
        let mut yt = self.level1_table(&mats.m1, bits);
        let mut messages = Vec::with_capacity(self.rounds());
        let mut rs = Vec::with_capacity(self.rounds());
        let mut running = claim;
        let mut len = n_blocks;
        if self.block_vars() > 0 {
            // Ping-pong buffers of the first fold's size, allocated once.
            let half0 = n_blocks / 2 * rows1;
            let (mut ga, mut ya) = (zeroed_f128(half0), zeroed_f128(half0));
            let (mut gb, mut yb) = (zeroed_f128(half0 / 2), zeroed_f128(half0 / 2));
            let mut msg = block_message(&gt[..len * rows1], &yt[..len * rows1], rows1);
            for round in 0..self.block_vars() {
                let (q1, qinf) = msg;
                let r = challenge(round, (q1, qinf))?;
                running = round_value(running, q1, qinf, r);
                messages.push((q1, qinf));
                rs.push(r);
                let half = len / 2;
                // Fold into the (free) output pair; the next message comes from the same
                // pass. Round 0 reads the initial tables, later rounds alternate.
                let (gi, yi, go, yo): (&[F128], &[F128], &mut Vec<F128>, &mut Vec<F128>) =
                    if round == 0 {
                        (&gt, &yt, &mut ga, &mut ya)
                    } else if round % 2 == 1 {
                        (&ga, &ya, &mut gb, &mut yb)
                    } else {
                        (&gb, &yb, &mut ga, &mut ya)
                    };
                msg = fold_blocks_message(
                    &gi[..len * rows1],
                    &yi[..len * rows1],
                    &mut go[..half * rows1],
                    &mut yo[..half * rows1],
                    rows1,
                    r,
                );
                len = half;
                if round == 0 {
                    drop(std::mem::take(&mut gt));
                    drop(std::mem::take(&mut yt));
                }
            }
            // The last fold wrote `ga` (odd round count) or `gb`.
            gt = if self.block_vars() % 2 == 1 { ga } else { gb };
            yt = if self.block_vars() % 2 == 1 { ya } else { yb };
        }
        drop(yt);
        gt.truncate(rows1);
        let g = gt;
        // In-block phase: w = M_1^T g over the block's positions, bf = b''~(r_block, .).
        let blen = self.layout.block_len();
        let base = self.layout.zeta1_base();
        let mut w: Vec<F128> = (0..blen)
            .into_par_iter()
            .with_min_len(1024)
            .map(|zeta| gather(&mats.m1, base + zeta, |r| g[r]))
            .collect();
        let mut bf = self.block_fold(bits, &eq_table(&rs));
        let nb = self.block_vars();
        for round in 0..self.zeta_vars() {
            let (q1, qinf) = pair_message(&w, &bf);
            let r = challenge(nb + round, (q1, qinf))?;
            running = round_value(running, q1, qinf, r);
            messages.push((q1, qinf));
            rs.push(r);
            fold_pairs(&mut w, r);
            fold_pairs(&mut bf, r);
        }
        debug_assert_eq!(w.len(), 1);
        debug_assert_eq!(w[0] * bf[0], running, "honest final claim");
        Ok(ChainProof {
            messages,
            point: self.split_point(&rs),
            value: bf[0],
            last: running,
        })
    }

    /// Replay the rounds of `messages` from `claim`: `challenge(round, message)` absorbs and
    /// returns the challenge. Returns the point and the running claim after the last round,
    /// which the caller checks against `W~(r) b''~(r)`.
    ///
    /// # Errors
    /// The first error `challenge` returns.
    ///
    /// # Panics
    /// If the message count is not the round count.
    pub fn replay<E>(
        &self,
        claim: F128,
        messages: &[(F128, F128)],
        mut challenge: impl FnMut(usize, (F128, F128)) -> Result<F128, E>,
    ) -> Result<(ChainPoint, F128), E> {
        assert_eq!(messages.len(), self.rounds(), "one message per round");
        let mut running = claim;
        let mut rs = Vec::with_capacity(messages.len());
        for (round, &(q1, qinf)) in messages.iter().enumerate() {
            let r = challenge(round, (q1, qinf))?;
            running = round_value(running, q1, qinf, r);
            rs.push(r);
        }
        Ok((self.split_point(&rs), running))
    }

    /// `b''~(point)` evaluated directly from the input bits (tests and debug checks).
    #[must_use]
    pub fn input_value(&self, bits: &[u64], point: &ChainPoint) -> F128 {
        assert_eq!(bits.len(), self.input_words(), "chain input length");
        let e_b = eq_table(&point.block());
        let e_z = eq_table(&point.zeta);
        let blen = self.layout.block_len();
        e_b.par_iter()
            .enumerate()
            .map(|(beta, &eb)| {
                let start = self.block_start(beta);
                let mut acc = F128::ZERO;
                for (zeta, &ez) in e_z.iter().enumerate().take(blen) {
                    let c = start + zeta;
                    if (bits[c / 64] >> (c % 64)) & 1 == 1 {
                        acc += ez;
                    }
                }
                eb * acc
            })
            .reduce(|| F128::ZERO, |a, b| a + b)
    }
}

/// `sum_{r in S(col)} v(r)` over the rows of column `col`.
#[inline]
fn gather(m: &SparseMatrix, col: usize, v: impl Fn(usize) -> F128) -> F128 {
    m.column(col)
        .iter()
        .fold(F128::ZERO, |acc, &r| acc + v(r as usize))
}

/// `(q(1), q(inf))` of the product of two tables over their lowest variable (pairs
/// `(2k, 2k + 1)`).
fn pair_message(a: &[F128], b: &[F128]) -> (F128, F128) {
    let f = |(x, y): (&[F128], &[F128])| (x[1] * y[1], (x[0] + x[1]) * (y[0] + y[1]));
    let add = |p: (F128, F128), q: (F128, F128)| (p.0 + q.0, p.1 + q.1);
    if a.len() >= MIN_PARALLEL {
        a.par_chunks(2)
            .zip(b.par_chunks(2))
            .map(f)
            .reduce(|| (F128::ZERO, F128::ZERO), add)
    } else {
        a.chunks(2)
            .zip(b.chunks(2))
            .map(f)
            .fold((F128::ZERO, F128::ZERO), add)
    }
}

/// `t[k] = t[2k] + r (t[2k] + t[2k + 1])` in place, halving the length.
fn fold_pairs(t: &mut Vec<F128>, r: F128) {
    let half = t.len() / 2;
    let folded: Vec<F128> = t
        .par_chunks(2)
        .with_min_len(MIN_PARALLEL / 2)
        .map(|p| p[0] + r * (p[0] + p[1]))
        .collect();
    debug_assert_eq!(folded.len(), half);
    *t = folded;
}

/// `(q(1), q(inf))` of the block phase: the product of two `[beta][rho_1]` tables over the
/// lowest block variable, summed over `rho_1` and the other blocks.
fn block_message(g: &[F128], y: &[F128], rows: usize) -> (F128, F128) {
    g.par_chunks(2 * rows)
        .zip(y.par_chunks(2 * rows))
        .map(|(gp, yp)| {
            let (g0, g1) = gp.split_at(rows);
            let (y0, y1) = yp.split_at(rows);
            let mut q1 = F128::ZERO;
            let mut qi = F128::ZERO;
            for i in 0..rows {
                q1 += g1[i] * y1[i];
                qi += (g0[i] + g1[i]) * (y0[i] + y1[i]);
            }
            (q1, qi)
        })
        .reduce(|| (F128::ZERO, F128::ZERO), |p, q| (p.0 + q.0, p.1 + q.1))
}

/// Fold the lowest block variable of both `[beta][rho_1]` tables (`len` blocks, `len` even)
/// into `g_out`/`y_out` (`len / 2` blocks) with the challenge `r`, and return the next
/// round's message `(q(1), q(inf))` of the folded tables (zero when they are a single block,
/// where no further round follows). Same sums as a separate fold followed by
/// `block_message`; the field is characteristic 2, so the grouping of the sums does not
/// change the result.
fn fold_blocks_message(
    g: &[F128],
    y: &[F128],
    g_out: &mut [F128],
    y_out: &mut [F128],
    rows: usize,
    r: F128,
) -> (F128, F128) {
    let out_blocks = g_out.len() / rows;
    if out_blocks < 2 {
        g_out
            .par_chunks_mut(rows)
            .zip(y_out.par_chunks_mut(rows))
            .enumerate()
            .for_each(|(k, (gd, yd))| {
                for i in 0..rows {
                    let (a, b) = (g[2 * k * rows + i], g[(2 * k + 1) * rows + i]);
                    gd[i] = a + r * (a + b);
                    let (a, b) = (y[2 * k * rows + i], y[(2 * k + 1) * rows + i]);
                    yd[i] = a + r * (a + b);
                }
            });
        return (F128::ZERO, F128::ZERO);
    }
    // Output blocks come in pairs (2j, 2j + 1) built from the input blocks 4j..4j + 3.
    g_out
        .par_chunks_mut(2 * rows)
        .zip(y_out.par_chunks_mut(2 * rows))
        .enumerate()
        .map(|(j, (gd, yd))| {
            let (gd0, gd1) = gd.split_at_mut(rows);
            let (yd0, yd1) = yd.split_at_mut(rows);
            let base = 4 * j * rows;
            let mut q1 = F128::ZERO;
            let mut qi = F128::ZERO;
            for i in 0..rows {
                let at = |t: &[F128], b: usize| t[base + b * rows + i];
                let (a, b) = (at(g, 0), at(g, 1));
                let g0 = a + r * (a + b);
                let (a, b) = (at(g, 2), at(g, 3));
                let g1 = a + r * (a + b);
                let (a, b) = (at(y, 0), at(y, 1));
                let y0 = a + r * (a + b);
                let (a, b) = (at(y, 2), at(y, 3));
                let y1 = a + r * (a + b);
                gd0[i] = g0;
                gd1[i] = g1;
                yd0[i] = y0;
                yd1[i] = y1;
                q1 += g1 * y1;
                qi += (g0 + g1) * (y0 + y1);
            }
            (q1, qi)
        })
        .reduce(|| (F128::ZERO, F128::ZERO), |p, q| (p.0 + q.0, p.1 + q.1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gf2_walker;
    use crate::seed::derive_matrices_for;
    use crate::shape::{LevelShape, Shape};
    use agg_witness::gen::SplitMix64;

    fn rand_f(rng: &mut SplitMix64) -> F128 {
        F128::new(rng.next_u64(), rng.next_u64())
    }

    /// The toy shape with `lanes` lanes (level 3 takes `lanes m_2` inputs).
    fn toy_with_lanes(lanes: usize) -> Shape {
        let mut s = Shape::TOY;
        s.lanes = lanes;
        s.levels[2] = LevelShape {
            cols: lanes * s.levels[1].rows,
            used: lanes * s.levels[1].rows,
            ..s.levels[2]
        };
        s.validate().expect("valid toy variant");
        s
    }

    /// Dense `W(x)` and `b''(x)` over the chain's variables in ROUND order (block variables
    /// low, then in-block), for the brute-force sum-check.
    fn dense(
        chain: &ChainLayout,
        mats: &BridgeMatrices,
        bits: &[u64],
        eq_y: &EqY,
    ) -> (Vec<F128>, Vec<F128>) {
        let lay = chain.layout;
        let e3 = eq_table(&eq_y.r_o);
        let wt = gf2_walker::transpose(&lay, mats, &e3); // [t * lane_len + p]
        let e_s = eq_table(&eq_y.r_y);
        let nb = chain.block_vars();
        let n = 1usize << chain.rounds();
        let (blocks, lanes_p, blen) = (lay.blocks(), chain.lanes_padded(), lay.block_len());
        let mut w = vec![F128::ZERO; n];
        let mut b = vec![F128::ZERO; n];
        for x in 0..n {
            let (beta, zeta) = (x % (1 << nb), x >> nb);
            let (i_mid, rest) = (beta % blocks, beta / blocks);
            let (t, sigma) = (rest % lanes_p, rest / lanes_p);
            let p = i_mid * blen + zeta;
            if t < lay.lanes() {
                w[x] = e_s[sigma] * wt[lay.column(t, p)];
            }
            let c = chain.bit_index(sigma, t, p);
            if (bits[c / 64] >> (c % 64)) & 1 == 1 {
                b[x] = F128::ONE;
            }
        }
        (w, b)
    }

    fn mle(table: &[F128], point: &[F128]) -> F128 {
        let e = eq_table(point);
        table
            .iter()
            .zip(&e)
            .fold(F128::ZERO, |a, (&x, &y)| a + x * y)
    }

    /// The two-phase prover against a brute-force product sum-check on dense tables, for
    /// random inputs (junk lanes included) at several toy layouts: the claim, every round
    /// message (block-phase cross terms included), the final value and the verifier's weight.
    #[test]
    fn two_phase_sumcheck_matches_brute_force() {
        let mut rng = SplitMix64::new(0x9a71);
        // (lanes, lane_len, part index, span_log)
        let cases = [
            (1, 1 << 12, 0, 0),
            (1, 1 << 12, 0, 1),
            (3, 1 << 11, 1, 1),
            (3, 1 << 9, 1, 0),
            (3, 1 << 10, 2, 2),
            (6, 1 << 11, 0, 1),
        ];
        for (case, &(lanes, lane_len, index, span_log)) in cases.iter().enumerate() {
            let shape = toy_with_lanes(lanes);
            let sd: [u8; 32] = std::array::from_fn(|i| (i as u8) ^ (case as u8));
            let mats = derive_matrices_for(&shape, &sd);
            let layout = SegmentLayout::part(shape, lane_len, index);
            let chain = ChainLayout::new(layout, span_log);
            let bits: Vec<u64> = (0..chain.input_words()).map(|_| rng.next_u64()).collect();
            let eq_y = EqY {
                r_o: (0..chain.out_vars()).map(|_| rand_f(&mut rng)).collect(),
                r_y: (0..span_log).map(|_| rand_f(&mut rng)).collect(),
            };
            // The final vectors of the input (junk lanes do not enter Y).
            let ys: Vec<Vec<u8>> = (0..1usize << span_log)
                .map(|sigma| {
                    let mut seg = vec![0_u64; layout.input_bits() / 64];
                    for t in 0..lanes {
                        for p in 0..lane_len {
                            let c = chain.bit_index(sigma, t, p);
                            if (bits[c / 64] >> (c % 64)) & 1 == 1 {
                                let d = layout.column(t, p);
                                seg[d / 64] |= 1 << (d % 64);
                            }
                        }
                    }
                    gf2_walker::composite(&layout, &mats, &seg)
                })
                .collect();
            let claim = chain.claim(&ys, &eq_y);
            let (mut w, mut b) = dense(&chain, &mats, &bits, &eq_y);
            let brute: F128 = w.iter().zip(&b).fold(F128::ZERO, |a, (&x, &y)| a + x * y);
            assert_eq!(claim, brute, "case {case}: <eq_Y, Y> = sum W b''");
            // Fixed challenges, so the brute-force run sees the same ones.
            let rs: Vec<F128> = (0..chain.rounds()).map(|_| rand_f(&mut rng)).collect();
            let proof = chain
                .prove(&mats, &bits, &eq_y, claim, |k, _| Ok::<_, ()>(rs[k]))
                .expect("no transcript errors");
            let mut running = claim;
            for (k, &r) in rs.iter().enumerate() {
                let (q1, qi) = pair_message(&w, &b);
                let q0 = w
                    .chunks(2)
                    .zip(b.chunks(2))
                    .fold(F128::ZERO, |a, (x, y)| a + x[0] * y[0]);
                assert_eq!(q0 + q1, running, "case {case} round {k}: q(0) + q(1)");
                assert_eq!(
                    proof.messages[k],
                    (q1, qi),
                    "case {case} round {k}: message"
                );
                running = round_value(running, q1, qi, r);
                fold_pairs(&mut w, r);
                fold_pairs(&mut b, r);
            }
            assert_eq!(proof.last, running, "case {case}: last claim");
            assert_eq!(proof.value, b[0], "case {case}: b''~(r)");
            assert_eq!(proof.value, chain.input_value(&bits, &proof.point));
            let weight = chain.weight_at(&mats, &eq_y, &proof.point);
            assert_eq!(weight, w[0], "case {case}: W~(r)");
            assert_eq!(weight * proof.value, proof.last, "case {case}: final check");
            // The replay gives the same point and claim.
            let (point, last) = chain
                .replay(claim, &proof.messages, |k, _| Ok::<_, ()>(rs[k]))
                .expect("no transcript errors");
            assert_eq!((point, last), (proof.point.clone(), proof.last));
            // The dense MLEs at the point in round order.
            let order: Vec<F128> = rs.clone();
            let (w0, b0) = dense(&chain, &mats, &bits, &eq_y);
            assert_eq!(mle(&w0, &order), weight);
            assert_eq!(mle(&b0, &order), proof.value);
        }
    }

    /// A flipped input bit (in a used lane) with the stale final vectors breaks the first
    /// round's identity `q(0) + q(1) = claim` (the honest prover's messages of the new
    /// input), with overwhelming probability over `eq_Y`; a junk-lane flip does not change
    /// the claim.
    #[test]
    fn a_flipped_bit_with_stale_y_breaks_the_first_round() {
        let mut rng = SplitMix64::new(0x51a1e);
        let shape = toy_with_lanes(3);
        let mats = derive_matrices_for(&shape, &[7; 32]);
        let layout = SegmentLayout::part(shape, 1 << 11, 0);
        let chain = ChainLayout::new(layout, 1);
        let bits: Vec<u64> = (0..chain.input_words()).map(|_| rng.next_u64()).collect();
        let eq_y = EqY {
            r_o: (0..chain.out_vars()).map(|_| rand_f(&mut rng)).collect(),
            r_y: vec![rand_f(&mut rng)],
        };
        let first = |bits: &[u64]| {
            let mut msg = None;
            let _ = chain.prove(&mats, bits, &eq_y, F128::ZERO, |k, m| {
                if k == 0 {
                    msg = Some(m);
                }
                Err::<F128, ()>(())
            });
            msg.expect("first message")
        };
        let honest = first(&bits);
        let (w, b) = dense(&chain, &mats, &bits, &eq_y);
        let claim: F128 = w.iter().zip(&b).fold(F128::ZERO, |a, (&x, &y)| a + x * y);
        let q0 = |bits: &[u64]| {
            let (w, b) = dense(&chain, &mats, bits, &eq_y);
            w.chunks(2)
                .zip(b.chunks(2))
                .fold(F128::ZERO, |a, (x, y)| a + x[0] * y[0])
        };
        assert_eq!(q0(&bits) + honest.0, claim);
        // Lane 1, segment 1, position 77: a used lane.
        let mut flipped = bits.clone();
        let c = chain.bit_index(1, 1, 77);
        flipped[c / 64] ^= 1 << (c % 64);
        let m = first(&flipped);
        assert_ne!(
            q0(&flipped) + m.0,
            claim,
            "stale Y against a flipped used bit"
        );
        // Lane 3 (junk): the claim of the stale Y still holds.
        let mut junk = bits.clone();
        let c = chain.bit_index(0, 3, 5);
        junk[c / 64] ^= 1 << (c % 64);
        let m = first(&junk);
        assert_eq!(q0(&junk) + m.0, claim, "junk lanes carry no weight");
    }

    /// Measurement (ignored; `--release --ignored --nocapture`): the chain prover (`prove`,
    /// dense random input, the production shape) and the verifier's `claim` + `weight_at` at
    /// `AGG_CHAIN_SHAPES` (default `11/0,13/0,13/2`: `d/span_log`), median of
    /// `AGG_CHAIN_REPS` (default 3) runs after one warmup.
    #[test]
    #[ignore = "measurement; run with --release --ignored --nocapture"]
    fn chain_timing_at_production_shapes() {
        let shapes: Vec<(u32, usize)> = std::env::var("AGG_CHAIN_SHAPES")
            .unwrap_or_else(|_| "11/0,13/0,13/2".to_string())
            .split(',')
            .map(|x| {
                let (d, s) = x.split_once('/').expect("d/span_log");
                (d.trim().parse().unwrap(), s.trim().parse().unwrap())
            })
            .collect();
        let reps: usize = std::env::var("AGG_CHAIN_REPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);
        let shape = Shape::PRODUCTION;
        let t = std::time::Instant::now();
        let mats = derive_matrices_for(&shape, &[3; 32]);
        eprintln!("matrices: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
        let median = |mut v: Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        for (d, span_log) in shapes {
            let chain = ChainLayout::new(SegmentLayout::production_part(d, 0), span_log);
            let mut rng = SplitMix64::new(u64::from(d) + 100 * span_log as u64);
            let bits: Vec<u64> = (0..chain.input_words()).map(|_| rng.next_u64()).collect();
            let eq_y = EqY {
                r_o: (0..chain.out_vars()).map(|_| rand_f(&mut rng)).collect(),
                r_y: (0..span_log).map(|_| rand_f(&mut rng)).collect(),
            };
            let (mut tp, mut tv) = (Vec::new(), Vec::new());
            for rep in 0..=reps {
                let t = std::time::Instant::now();
                let proof = chain
                    .prove(&mats, &bits, &eq_y, F128::ZERO, |k, (q1, qi)| {
                        Ok::<_, ()>(q1 + qi + F128::new(k as u64 + 1, 7))
                    })
                    .unwrap();
                let p = t.elapsed().as_secs_f64() * 1e3;
                let t = std::time::Instant::now();
                let w = chain.weight_at(&mats, &eq_y, &proof.point);
                std::hint::black_box((w, chain.claim(&[vec![1; 4096]], &eq_y)));
                let v = t.elapsed().as_secs_f64() * 1e3;
                if rep > 0 {
                    tp.push(p);
                    tv.push(v);
                }
            }
            eprintln!(
                "d {d}, span_log {span_log}: {} rounds over {} input bits: prove {:.1} ms, \
                 verifier weight and claim {:.1} ms (medians of {reps})",
                chain.rounds(),
                64 * chain.input_words(),
                median(tp),
                median(tv)
            );
        }
    }

    #[test]
    fn eq_helpers_agree() {
        let mut rng = SplitMix64::new(3);
        let a: Vec<F128> = (0..5).map(|_| rand_f(&mut rng)).collect();
        let t = eq_table(&a);
        for (i, &e) in t.iter().enumerate() {
            let bits: Vec<F128> = (0..5)
                .map(|k| {
                    if (i >> k) & 1 == 1 {
                        F128::ONE
                    } else {
                        F128::ZERO
                    }
                })
                .collect();
            assert_eq!(e, eq_eval(&a, &bits));
        }
        // q(r) for q = 3 + 5X + 7X^2 (char 2 arithmetic).
        let (c0, c1, c2) = (F128::new(3, 0), F128::new(5, 0), F128::new(7, 0));
        let r = rand_f(&mut rng);
        let q = |x: F128| c0 + c1 * x + c2 * x * x;
        assert_eq!(
            round_value(q(F128::ZERO) + q(F128::ONE), q(F128::ONE), c2, r),
            q(r)
        );
    }
}
