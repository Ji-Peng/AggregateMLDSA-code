//! Parity-bridge geometry and the honest integer bridge witness (spec Sections 1, 5, 9).
//!
//! The matrices M_1, M_2, M_3 are taken as input (`SparseMatrix`); their derivation from the
//! bridge seed sd (spec Section 9.1) belongs to `agg-bridge`. Given the w1Encode bit tables b_t
//! on the main domain, this module computes the honest parities and carries of (9.2):
//! level 1 per block beta = (i_mid, sigma) over in-block positions zeta = (j, rho, i_lo),
//! level 2 per segment over zeta = (rho_1, i_mid) zero-padded to 2^19, level 3 per segment over
//! the six lanes zeta = (t, rho_2). Index conventions (least significant first):
//! zeta_1 = x mod 2^19 and beta_1 = x >> 19 for the main index x; D_1 index rho_1 + 2^13 beta_1;
//! zeta_2 = D_1 index mod 2^17 and sigma = D_1 index >> 17; D_2 index rho_2 + 2^13 sigma;
//! zeta_3 = t + 6 rho_2; D_3 index rho_3 + 2^12 sigma.

use crate::gen::SplitMix64;
use crate::padding::Geometry;
use crate::params::W1_BITS;

/// Level-1 in-block size J_1 = 2^19.
pub const J1: usize = 1 << 19;
/// Level-1 output rows m_1 = 2^13.
pub const M1: usize = 1 << 13;
/// Level-1 column weight k_1.
pub const K1: usize = 14;
/// Level-2 in-block size J_2 = 2^19 (zero-padded).
pub const J2: usize = 1 << 19;
/// Level-2 output rows m_2 = 2^13.
pub const M2: usize = 1 << 13;
/// Level-2 column weight k_2.
pub const K2: usize = 14;
/// Level-3 in-block size J_3 = 6 2^13.
pub const J3: usize = W1_BITS * M2;
/// Level-3 output rows m_3 = 2^12.
pub const M3: usize = 1 << 12;
/// Level-3 column weight k_3.
pub const K3: usize = 18;
/// Row-weight cap of levels 1 and 2 (carries in [0, 1023]).
pub const ROW_WEIGHT_MAX_12: u32 = 2047;
/// Row-weight cap of level 3 (carries in [0, 255]).
pub const ROW_WEIGHT_MAX_3: u32 = 511;
/// Level-2 in-block positions actually used: (rho_1, i_mid) < 2^17.
pub const ZETA2_USED: usize = 1 << 17;

/// A 0/1 matrix stored by columns, each column a set of `k` distinct rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseMatrix {
    /// Number of rows m.
    pub rows: usize,
    /// Number of columns J.
    pub cols: usize,
    /// Column weight k.
    pub k: usize,
    /// Row indices, column-major (`cols * k` entries).
    pub entries: Vec<u32>,
}

impl SparseMatrix {
    /// Validated constructor.
    pub fn new(rows: usize, cols: usize, k: usize, entries: Vec<u32>) -> Self {
        assert_eq!(entries.len(), cols * k);
        for c in 0..cols {
            let col = &entries[c * k..(c + 1) * k];
            for (a, &r) in col.iter().enumerate() {
                assert!((r as usize) < rows, "row out of range");
                assert!(!col[..a].contains(&r), "repeated row in a column");
            }
        }
        Self {
            rows,
            cols,
            k,
            entries,
        }
    }

    /// Rows of column c.
    #[inline]
    pub fn column(&self, c: usize) -> &[u32] {
        &self.entries[c * self.k..(c + 1) * self.k]
    }

    /// Row weights.
    pub fn row_weights(&self) -> Vec<u32> {
        let mut w = vec![0u32; self.rows];
        for &r in &self.entries {
            w[r as usize] += 1;
        }
        w
    }

    /// Random matrix with uniform k-subset columns from a test PRNG (not the spec's seed derivation).
    pub fn random_for_tests(seed: u64, rows: usize, cols: usize, k: usize) -> Self {
        let mut rng = SplitMix64::new(seed);
        let mut entries = Vec::with_capacity(cols * k);
        for _ in 0..cols {
            let start = entries.len();
            while entries.len() - start < k {
                let r = rng.below(rows as u64) as u32;
                if !entries[start..].contains(&r) {
                    entries.push(r);
                }
            }
        }
        Self {
            rows,
            cols,
            k,
            entries,
        }
    }
}

/// The three bridge matrices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BridgeMatrices {
    /// M_1 (2^13 x 2^19, weight 14).
    pub m1: SparseMatrix,
    /// M_2 (2^13 x 2^19, weight 14).
    pub m2: SparseMatrix,
    /// M_3 (2^12 x 49,152, weight 18).
    pub m3: SparseMatrix,
}

impl BridgeMatrices {
    /// Test matrices of the spec shapes from a PRNG seed.
    pub fn random_for_tests(seed: u64) -> Self {
        Self {
            m1: SparseMatrix::random_for_tests(seed ^ 1, M1, J1, K1),
            m2: SparseMatrix::random_for_tests(seed ^ 2, M2, J2, K2),
            m3: SparseMatrix::random_for_tests(seed ^ 3, M3, J3, K3),
        }
    }

    /// True when the shapes are those of spec Section 1.
    pub fn has_spec_shapes(&self) -> bool {
        let s = |m: &SparseMatrix| (m.rows, m.cols, m.k);
        s(&self.m1) == (M1, J1, K1) && s(&self.m2) == (M2, J2, K2) && s(&self.m3) == (M3, J3, K3)
    }

    /// Honest-prover abort condition (spec Section 9.2): a row weight above the cap.
    pub fn row_weights_ok(&self) -> bool {
        let max = |m: &SparseMatrix| m.row_weights().into_iter().max().unwrap_or(0);
        max(&self.m1) <= ROW_WEIGHT_MAX_12
            && max(&self.m2) <= ROW_WEIGHT_MAX_12
            && max(&self.m3) <= ROW_WEIGHT_MAX_3
    }
}

/// zeta_3 = t + 6 rho_2.
#[inline]
pub fn zeta3(t: usize, rho2: usize) -> usize {
    t + W1_BITS * rho2
}

/// Integer sums sum_zeta M_1[rho_1, zeta] b_t(zeta, beta) on D_1, per lane (any integer b).
pub fn level1_sums(
    geom: &Geometry,
    b: &[&[i8]; W1_BITS],
    m1: &SparseMatrix,
) -> [Vec<i64>; W1_BITS] {
    let len = 1usize << geom.d1();
    let mut sums: [Vec<i64>; W1_BITS] = std::array::from_fn(|_| vec![0i64; len]);
    for x in 0..geom.main_len() {
        let zeta = x & (J1 - 1);
        let base = (x >> 19) * M1;
        for t in 0..W1_BITS {
            let bit = i64::from(b[t][x]);
            if bit != 0 {
                for &r in m1.column(zeta) {
                    sums[t][base + r as usize] += bit;
                }
            }
        }
    }
    sums
}

/// Integer sums sum_zeta M_2[rho_2, zeta] y_t(zeta, sigma) on D_2 from a D_1 table, per lane.
pub fn level2_sums(
    geom: &Geometry,
    y1: &[Vec<i64>; W1_BITS],
    m2: &SparseMatrix,
) -> [Vec<i64>; W1_BITS] {
    let len = 1usize << geom.d2();
    let mut sums: [Vec<i64>; W1_BITS] = std::array::from_fn(|_| vec![0i64; len]);
    for t in 0..W1_BITS {
        for (idx, &y) in y1[t].iter().enumerate() {
            if y != 0 {
                let zeta = idx & (ZETA2_USED - 1);
                let base = (idx >> 17) * M2;
                for &r in m2.column(zeta) {
                    sums[t][base + r as usize] += y;
                }
            }
        }
    }
    sums
}

/// Integer sums sum_{t, rho_2} M_3[rho_3, (t, rho_2)] y_t(rho_2, sigma) on D_3.
pub fn level3_sums(geom: &Geometry, y2: &[Vec<i64>; W1_BITS], m3: &SparseMatrix) -> Vec<i64> {
    let len = 1usize << geom.d3();
    let mut sums = vec![0i64; len];
    for t in 0..W1_BITS {
        for (idx, &y) in y2[t].iter().enumerate() {
            if y != 0 {
                let rho2 = idx & (M2 - 1);
                let base = (idx >> 13) * M3;
                for &r in m3.column(zeta3(t, rho2)) {
                    sums[base + r as usize] += y;
                }
            }
        }
    }
    sums
}

/// Honest bridge witness: parities and carries of the three levels (9.2) and the public Y.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BridgeWitness {
    /// y^(1)_t on D_1 (virtual in the PIOP; materialized here).
    pub y1: [Vec<i64>; W1_BITS],
    /// kappa^(1)_t on D_1.
    pub kappa1: [Vec<i64>; W1_BITS],
    /// y^(2)_t on D_2 (committed).
    pub y2: [Vec<i64>; W1_BITS],
    /// kappa^(2)_t on D_2.
    pub kappa2: [Vec<i64>; W1_BITS],
    /// Y on D_3: Y_sigma[rho_3] at rho_3 + 2^12 sigma (public).
    pub y: Vec<i64>,
    /// kappa^(3) on D_3.
    pub kappa3: Vec<i64>,
}

/// Bridge errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BridgeError {
    /// Matrix shapes differ from spec Section 1.
    Shape,
    /// A row weight exceeds the cap (the honest prover MUST abort, spec Section 9.2).
    RowWeight,
}

fn split_parity(sums: Vec<i64>) -> (Vec<i64>, Vec<i64>) {
    let y = sums.iter().map(|s| s & 1).collect();
    let k = sums.iter().map(|s| s >> 1).collect();
    (y, k)
}

/// Computes the honest bridge witness from the committed bits b_t (main domain, values 0/1).
pub fn compute(
    geom: &Geometry,
    b: &[&[i8]; W1_BITS],
    mats: &BridgeMatrices,
) -> Result<BridgeWitness, BridgeError> {
    if !mats.has_spec_shapes() {
        return Err(BridgeError::Shape);
    }
    if !mats.row_weights_ok() {
        return Err(BridgeError::RowWeight);
    }
    let s1 = level1_sums(geom, b, &mats.m1);
    let mut y1: [Vec<i64>; W1_BITS] = Default::default();
    let mut kappa1: [Vec<i64>; W1_BITS] = Default::default();
    for (t, s) in s1.into_iter().enumerate() {
        (y1[t], kappa1[t]) = split_parity(s);
    }
    let s2 = level2_sums(geom, &y1, &mats.m2);
    let mut y2: [Vec<i64>; W1_BITS] = Default::default();
    let mut kappa2: [Vec<i64>; W1_BITS] = Default::default();
    for (t, s) in s2.into_iter().enumerate() {
        (y2[t], kappa2[t]) = split_parity(s);
    }
    let (y, kappa3) = split_parity(level3_sums(geom, &y2, &mats.m3));
    Ok(BridgeWitness {
        y1,
        kappa1,
        y2,
        kappa2,
        y,
        kappa3,
    })
}

/// The GF(2) side (9.1): Y_sigma = M_3 M_2 M_1 b' computed with XOR only.
pub fn gf2_composite(geom: &Geometry, b: &[&[i8]; W1_BITS], mats: &BridgeMatrices) -> Vec<u8> {
    let len1 = 1usize << geom.d1();
    let mut y1 = vec![[0u8; W1_BITS]; len1];
    for x in 0..geom.main_len() {
        let zeta = x & (J1 - 1);
        let base = (x >> 19) * M1;
        for t in 0..W1_BITS {
            if b[t][x] & 1 == 1 {
                for &r in mats.m1.column(zeta) {
                    y1[base + r as usize][t] ^= 1;
                }
            }
        }
    }
    let mut y2 = vec![[0u8; W1_BITS]; 1usize << geom.d2()];
    for (idx, lanes) in y1.iter().enumerate() {
        let zeta = idx & (ZETA2_USED - 1);
        let base = (idx >> 17) * M2;
        for t in 0..W1_BITS {
            if lanes[t] == 1 {
                for &r in mats.m2.column(zeta) {
                    y2[base + r as usize][t] ^= 1;
                }
            }
        }
    }
    let mut y = vec![0u8; 1usize << geom.d3()];
    for (idx, lanes) in y2.iter().enumerate() {
        let rho2 = idx & (M2 - 1);
        let base = (idx >> 13) * M3;
        for t in 0..W1_BITS {
            if lanes[t] == 1 {
                for &r in mats.m3.column(zeta3(t, rho2)) {
                    y[base + r as usize] ^= 1;
                }
            }
        }
    }
    y
}

/// The committed bridge groups of spec Section 4.1 (round 2) and the public Y.
///
/// Column order: G_1 = kappa^(1)_{t,k} at index 3 t + k; G_2 = y^(2)_t at index t, then
/// kappa^(2)_{t,k} at 6 + 3 t + k; G_3 = kappa^(3)_k at index k (t < 6, digits least significant
/// first). Logical lengths 2^{d_1}, 2^{d_2}, 2^{d_3} (physical zero-extension to 2^13 is the PCS's).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BridgeTables {
    /// G_1: 18 carry-digit columns on D_1.
    pub g1: Vec<Vec<i8>>,
    /// G_2: 6 parity columns and 18 carry-digit columns on D_2.
    pub g2: Vec<Vec<i8>>,
    /// G_3: 2 carry-digit columns on D_3.
    pub g3: Vec<Vec<i8>>,
    /// Public Y on D_3 (Y_sigma[rho_3] at rho_3 + 2^12 sigma).
    pub y: Vec<u8>,
}

fn digit_columns(
    values: &[i64],
    layout: &crate::digits::DigitLayout,
) -> Result<Vec<Vec<i8>>, &'static str> {
    let mut cols = vec![Vec::with_capacity(values.len()); layout.len()];
    for &v in values {
        for (k, d) in layout.decompose(v).into_iter().enumerate() {
            cols[k].push(i8::try_from(d).map_err(|_| "carry digit does not fit i8")?);
        }
    }
    Ok(cols)
}

fn compose_columns(cols: &[Vec<i8>], layout: &crate::digits::DigitLayout) -> Vec<i64> {
    let len = cols.first().map_or(0, Vec::len);
    (0..len)
        .map(|x| layout.compose(&cols.iter().map(|c| i64::from(c[x])).collect::<Vec<_>>()))
        .collect()
}

impl BridgeTables {
    /// Commits the honest witness: carries by their digits (spec Section 4.3).
    pub fn from_witness(w: &BridgeWitness) -> Result<Self, &'static str> {
        use crate::digits::{KAPPA12_DIGITS, KAPPA3_DIGITS};
        let mut g1 = Vec::new();
        for t in 0..W1_BITS {
            g1.extend(digit_columns(&w.kappa1[t], &KAPPA12_DIGITS)?);
        }
        let mut g2: Vec<Vec<i8>> = (0..W1_BITS)
            .map(|t| w.y2[t].iter().map(|&y| y as i8).collect())
            .collect();
        for t in 0..W1_BITS {
            g2.extend(digit_columns(&w.kappa2[t], &KAPPA12_DIGITS)?);
        }
        let g3 = digit_columns(&w.kappa3, &KAPPA3_DIGITS)?;
        let y = w.y.iter().map(|&v| v as u8).collect();
        Ok(Self { g1, g2, g3, y })
    }

    /// kappa^(1)_t recomposed by (4.1).
    pub fn kappa1(&self, t: usize) -> Vec<i64> {
        compose_columns(&self.g1[3 * t..3 * t + 3], &crate::digits::KAPPA12_DIGITS)
    }

    /// y^(2)_t.
    pub fn y2(&self, t: usize) -> Vec<i64> {
        self.g2[t].iter().map(|&x| i64::from(x)).collect()
    }

    /// kappa^(2)_t recomposed by (4.1).
    pub fn kappa2(&self, t: usize) -> Vec<i64> {
        compose_columns(
            &self.g2[W1_BITS + 3 * t..W1_BITS + 3 * t + 3],
            &crate::digits::KAPPA12_DIGITS,
        )
    }

    /// kappa^(3) recomposed by (4.1).
    pub fn kappa3(&self) -> Vec<i64> {
        compose_columns(&self.g3, &crate::digits::KAPPA3_DIGITS)
    }

    /// Column names and alphabets of (G_1, G_2, G_3).
    pub fn specs() -> [Vec<(String, crate::digits::Alphabet)>; 3] {
        use crate::digits::{Alphabet, KAPPA12_DIGITS, KAPPA3_DIGITS};
        let mut g1 = Vec::new();
        let mut g2 = Vec::new();
        for t in 0..W1_BITS {
            for k in 0..3 {
                g1.push((format!("kappa1_{t}_{k}"), KAPPA12_DIGITS.alphabet(k)));
            }
        }
        for t in 0..W1_BITS {
            g2.push((format!("y2_{t}"), Alphabet::Bool));
        }
        for t in 0..W1_BITS {
            for k in 0..3 {
                g2.push((format!("kappa2_{t}_{k}"), KAPPA12_DIGITS.alphabet(k)));
            }
        }
        let g3 = (0..2)
            .map(|k| (format!("kappa3_{k}"), KAPPA3_DIGITS.alphabet(k)))
            .collect();
        [g1, g2, g3]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_matrices_have_spec_shapes_and_row_weights() {
        let m = BridgeMatrices::random_for_tests(42);
        assert!(m.has_spec_shapes());
        assert!(m.row_weights_ok());
        assert_eq!(J3, 49_152);
        // Column weight exactly k_t with distinct rows (SparseMatrix::new validates).
        let _ = SparseMatrix::new(M3, J3, K3, m.m3.entries.clone());
    }

    #[test]
    fn integer_and_gf2_sides_agree_on_random_bits() {
        let geom = Geometry::new(600).unwrap(); // d_sig = 10: two level-1 blocks
        let mats = BridgeMatrices::random_for_tests(7);
        let mut rng = SplitMix64::new(3);
        let lanes: Vec<Vec<i8>> = (0..W1_BITS)
            .map(|_| {
                (0..geom.main_len())
                    .map(|x| {
                        if x >> 10 < geom.n {
                            (rng.next_u64() & 1) as i8
                        } else {
                            0
                        }
                    })
                    .collect()
            })
            .collect();
        let b: [&[i8]; W1_BITS] = std::array::from_fn(|t| lanes[t].as_slice());
        let w = compute(&geom, &b, &mats).unwrap();
        assert_eq!(
            w.y.iter().map(|&v| v as u8).collect::<Vec<_>>(),
            gf2_composite(&geom, &b, &mats)
        );
        assert!(w.kappa1.iter().flatten().all(|&k| (0..=1023).contains(&k)));
        assert!(w.kappa3.iter().all(|&k| (0..=255).contains(&k)));
        // A dense reference for level 1 at a few rows.
        for &row in &[0usize, 1, M1 - 1, M1 + 5] {
            let block = row / M1;
            let mut s = 0i64;
            for zeta in 0..J1 {
                let x = zeta + (block << 19);
                if x < geom.main_len() && mats.m1.column(zeta).contains(&((row % M1) as u32)) {
                    s += i64::from(b[2][x]);
                }
            }
            assert_eq!(s, w.y1[2][row] + 2 * w.kappa1[2][row]);
        }
    }
}
