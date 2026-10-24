//! Bridge shapes (docs/bridge.md Section 1): three levels `(J_t, m_t, k_t)`, the lanes and
//! the number `B_1` of level-1 blocks per lane and segment, with the carry caps `W_t` and the
//! carry-digit layouts.
//!
//! Level `t` maps every block of `J_t` input positions to `m_t` outputs with one matrix
//! `M_t` shared by all blocks, lanes and segments. Level 2 reads only the first
//! `J_2^used = B_1 m_1` columns of `M_2`; level 3 takes all lanes of a segment,
//! `J_3 = lanes m_2`, in-block index `zeta_3 = t + lanes rho_2`.
//!
//! Domains (least significant index first, docs/bridge.md Section 2) for an input of `len`
//! positions per lane (a power of two; the main domain in production):
//! level-1 block `beta = x / J_1` (one block, zero-padded, when `len < J_1`), `D_1` cell
//! `rho_1 + m_1 beta`; level-2 block (segment) `sigma = D_1 cell / J_2^used`, `D_2` cell
//! `rho_2 + m_2 sigma`; `D_3` cell `rho_3 + m_3 sigma`.

use agg_witness::bridge::{
    BridgeMatrices, SparseMatrix, J1, J2, J3, K1, K2, K3, M1, M2, M3, ROW_WEIGHT_MAX_12,
    ROW_WEIGHT_MAX_3, ZETA2_USED,
};
use agg_witness::digits::{DigitLayout, KAPPA12_DIGITS, KAPPA3_DIGITS};
use agg_witness::params::W1_BITS;

use crate::seed::Level;

/// One level of a bridge shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LevelShape {
    /// The level number `t` (1, 2, 3), absorbed into the seed derivation.
    pub t: u64,
    /// `J_t`: the columns of `M_t` (all derived from the seed).
    pub cols: usize,
    /// `J_t^used`: the columns actually read (`J_2^used = B_1 m_1`; `J_t` otherwise).
    pub used: usize,
    /// `m_t`, a power of two.
    pub rows: usize,
    /// `k_t`: the column weight.
    pub k: usize,
    /// `W_t`: the largest row weight (over the used columns) the honest prover accepts.
    pub weight_cap: u32,
    /// The carry digits (spec (4.1)); their range is `[0, floor(W_t / 2)]`.
    pub carry: DigitLayout,
}

impl LevelShape {
    /// `log2 m_t`.
    #[must_use]
    pub const fn rows_log2(&self) -> u32 {
        self.rows.trailing_zeros()
    }

    /// The largest honest carry `floor(W_t / 2)`.
    #[must_use]
    pub const fn carry_max(&self) -> i64 {
        (self.weight_cap / 2) as i64
    }

    /// The seed-derivation parameters of this level (`agg_bridge::seed`).
    #[must_use]
    pub const fn seed_level(&self) -> Level {
        Level {
            t: self.t,
            rows_log2: self.rows_log2(),
            cols: self.cols,
            k: self.k,
        }
    }

    /// True when `m` has the rows, columns and column weight of this level.
    #[must_use]
    pub fn matches(&self, m: &SparseMatrix) -> bool {
        (m.rows, m.cols, m.k) == (self.rows, self.cols, self.k) && m.entries.len() == m.cols * m.k
    }

    /// Row weights over the used columns.
    #[must_use]
    pub fn used_row_weights(&self, m: &SparseMatrix) -> Vec<u32> {
        let mut weights = vec![0_u32; m.rows];
        for &row in &m.entries[..self.used * m.k] {
            weights[row as usize] += 1;
        }
        weights
    }
}

/// Toy level-1 carries: bases 8, 8, range `[0, 63]`.
pub const TOY_CARRY1_DIGITS: DigitLayout = DigitLayout {
    name: "kappa-toy1",
    bases: &[8, 8],
    signed: false,
};
/// Toy level-2 carries: bases 8, 4, range `[0, 31]`.
pub const TOY_CARRY2_DIGITS: DigitLayout = DigitLayout {
    name: "kappa-toy2",
    bases: &[8, 4],
    signed: false,
};
/// Toy level-3 carries: bases 4, 4, range `[0, 15]`.
pub const TOY_CARRY3_DIGITS: DigitLayout = DigitLayout {
    name: "kappa-toy3",
    bases: &[4, 4],
    signed: false,
};

/// A bridge shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    /// Short name for reports.
    pub name: &'static str,
    /// Levels 1, 2, 3.
    pub levels: [LevelShape; 3],
    /// Lanes per segment (the six bit columns `b_t` in production).
    pub lanes: usize,
    /// `B_1`: level-1 blocks per lane and segment.
    pub blocks1: usize,
}

impl Shape {
    /// The production shape of spec Section 1 (wire version 0 admits no other).
    pub const PRODUCTION: Shape = Shape {
        name: "production",
        levels: [
            LevelShape {
                t: 1,
                cols: J1,
                used: J1,
                rows: M1,
                k: K1,
                weight_cap: ROW_WEIGHT_MAX_12,
                carry: KAPPA12_DIGITS,
            },
            LevelShape {
                t: 2,
                cols: J2,
                used: ZETA2_USED,
                rows: M2,
                k: K2,
                weight_cap: ROW_WEIGHT_MAX_12,
                carry: KAPPA12_DIGITS,
            },
            LevelShape {
                t: 3,
                cols: J3,
                used: J3,
                rows: M3,
                k: K3,
                weight_cap: ROW_WEIGHT_MAX_3,
                carry: KAPPA3_DIGITS,
            },
        ],
        lanes: W1_BITS,
        blocks1: 16,
    };

    /// The toy shape of docs/bridge.md Section 1 (tests only): `L = 2^12` input bits, one
    /// lane, one segment, 16 output bits.
    pub const TOY: Shape = Shape {
        name: "toy",
        levels: [
            LevelShape {
                t: 1,
                cols: 1 << 10,
                used: 1 << 10,
                rows: 1 << 6,
                k: 4,
                weight_cap: 127,
                carry: TOY_CARRY1_DIGITS,
            },
            LevelShape {
                t: 2,
                cols: 1 << 8,
                used: 1 << 8,
                rows: 1 << 5,
                k: 4,
                weight_cap: 63,
                carry: TOY_CARRY2_DIGITS,
            },
            LevelShape {
                t: 3,
                cols: 32,
                used: 32,
                rows: 16,
                k: 4,
                weight_cap: 31,
                carry: TOY_CARRY3_DIGITS,
            },
        ],
        lanes: 1,
        blocks1: 4,
    };

    /// Level `t` (1, 2 or 3).
    #[must_use]
    pub const fn level(&self, t: usize) -> &LevelShape {
        &self.levels[t - 1]
    }

    /// Input bits per lane and segment, `L = B_1 J_1`.
    #[must_use]
    pub const fn segment_input_bits(&self) -> usize {
        self.blocks1 * self.levels[0].cols
    }

    /// Output bits per segment, `m_3`.
    #[must_use]
    pub const fn output_bits(&self) -> usize {
        self.levels[2].rows
    }

    /// The invariants of docs/bridge.md Section 1; `Err` names the first one violated.
    pub fn validate(&self) -> Result<(), &'static str> {
        let [l1, l2, l3] = &self.levels;
        for (t, level) in self.levels.iter().enumerate() {
            if level.t != t as u64 + 1 {
                return Err("level numbers are 1, 2, 3");
            }
            if !level.rows.is_power_of_two() || level.rows > 1 << 16 {
                return Err("m_t is a power of two of at most 2^16");
            }
            if level.k == 0 || level.k > level.rows {
                return Err("1 <= k_t <= m_t");
            }
            if level.used > level.cols {
                return Err("J_t^used <= J_t");
            }
            if level.carry.signed || level.carry.range() != (0, level.carry_max()) {
                return Err("the carry digits cover [0, floor(W_t / 2)] exactly");
            }
            // Both sides of a level identity stay far below p (spec Section 9.2).
            if u64::from(level.weight_cap) >= 1 << 20 {
                return Err("W_t far below p");
            }
        }
        if !l1.cols.is_power_of_two() || l1.used != l1.cols {
            return Err("J_1 is a power of two, all used");
        }
        if !self.blocks1.is_power_of_two() || l2.used != self.blocks1 * l1.rows {
            return Err("J_2^used = B_1 m_1 with B_1 a power of two");
        }
        if !l2.cols.is_power_of_two() {
            return Err("J_2 is a power of two");
        }
        if self.lanes == 0
            || self.lanes > 8
            || l3.cols != self.lanes * l2.rows
            || l3.used != l3.cols
        {
            return Err("J_3 = lanes m_2 with 1 <= lanes <= 8");
        }
        Ok(())
    }

    /// True when the three matrices have this shape.
    #[must_use]
    pub fn matches(&self, matrices: &BridgeMatrices) -> bool {
        self.levels[0].matches(&matrices.m1)
            && self.levels[1].matches(&matrices.m2)
            && self.levels[2].matches(&matrices.m3)
    }

    /// The honest prover's abort condition (docs/bridge.md Section 3): every row weight over
    /// the used columns is at most `W_t`.
    #[must_use]
    pub fn row_weights_ok(&self, matrices: &BridgeMatrices) -> bool {
        [&matrices.m1, &matrices.m2, &matrices.m3]
            .into_iter()
            .zip(&self.levels)
            .all(|(m, level)| {
                level
                    .used_row_weights(m)
                    .into_iter()
                    .all(|w| w <= level.weight_cap)
            })
    }

    /// The bridge domains for `len` input positions per lane (a power of two).
    #[must_use]
    pub fn domains(&self, len: usize) -> Domains {
        assert!(len.is_power_of_two(), "the input domain is a power of two");
        let [l1, l2, l3] = &self.levels;
        let blocks1 = (len / l1.cols).max(1);
        let segments = blocks1.div_ceil(self.blocks1);
        Domains {
            input_len: len,
            blocks1,
            segments,
            d1_len: blocks1 * l1.rows,
            d2_len: segments * l2.rows,
            d3_len: segments * l3.rows,
        }
    }
}

/// The bridge domains of one input size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Domains {
    /// Input positions per lane (the main domain).
    pub input_len: usize,
    /// Level-1 blocks per lane.
    pub blocks1: usize,
    /// Segments (level-2 blocks).
    pub segments: usize,
    /// Cells of `D_1` per lane.
    pub d1_len: usize,
    /// Cells of `D_2` per lane.
    pub d2_len: usize,
    /// Cells of `D_3`.
    pub d3_len: usize,
}

impl Domains {
    /// Variables of the input domain.
    #[must_use]
    pub const fn input_vars(&self) -> usize {
        self.input_len.trailing_zeros() as usize
    }

    /// Variables of `D_1`.
    #[must_use]
    pub const fn d1_vars(&self) -> usize {
        self.d1_len.trailing_zeros() as usize
    }

    /// Variables of `D_2`.
    #[must_use]
    pub const fn d2_vars(&self) -> usize {
        self.d2_len.trailing_zeros() as usize
    }

    /// Variables of `D_3`.
    #[must_use]
    pub const fn d3_vars(&self) -> usize {
        self.d3_len.trailing_zeros() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agg_witness::padding::Geometry;

    #[test]
    fn both_shapes_satisfy_the_invariants_with_the_documented_numbers() {
        for shape in [Shape::PRODUCTION, Shape::TOY] {
            shape.validate().unwrap();
        }
        let p = Shape::PRODUCTION;
        assert_eq!(p.segment_input_bits(), 1 << 23);
        assert_eq!(p.output_bits(), 4096);
        assert_eq!(p.levels.map(|l| l.carry_max()), [1023, 1023, 255]);
        assert_eq!(p.levels.map(|l| l.carry.constant()), [648, 648, 136]);
        let t = Shape::TOY;
        assert_eq!(t.segment_input_bits(), 1 << 12);
        assert_eq!(t.output_bits(), 16);
        assert_eq!(t.levels.map(|l| l.carry_max()), [63, 31, 15]);
        assert_eq!(t.levels.map(|l| l.carry.range().1), [63, 31, 15]);
        assert_eq!(
            t.levels.map(|l| (l.cols, l.rows, l.k)),
            [(1 << 10, 64, 4), (1 << 8, 32, 4), (32, 16, 4)]
        );
        let mut bad = Shape::TOY;
        bad.levels[1].used = 128;
        assert!(bad.validate().is_err());
        let mut bad = Shape::TOY;
        bad.levels[2].carry = TOY_CARRY2_DIGITS;
        assert!(bad.validate().is_err());
    }

    #[test]
    fn production_domains_are_those_of_the_geometry() {
        for n in [
            1,
            5,
            300,
            600,
            1024,
            5000,
            8192,
            8193,
            1 << 14,
            3 << 13,
            1 << 15,
        ] {
            let geom = Geometry::new(n).unwrap();
            let d = Shape::PRODUCTION.domains(geom.main_len());
            assert_eq!(d.input_vars(), geom.d_main() as usize);
            assert_eq!(d.d1_vars(), geom.d1() as usize, "N = {n}");
            assert_eq!(d.d2_vars(), geom.d2() as usize);
            assert_eq!(d.d3_vars(), geom.d3() as usize);
            assert_eq!(d.blocks1, geom.blocks1());
            assert_eq!(d.segments, geom.segments());
        }
        let toy = Shape::TOY.domains(1 << 12);
        assert_eq!((toy.blocks1, toy.segments), (4, 1));
        assert_eq!((toy.d1_len, toy.d2_len, toy.d3_len), (256, 32, 16));
    }
}
