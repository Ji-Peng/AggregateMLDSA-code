//! Soundness constants of the bridge.
//!
//! For a nonzero difference `delta = b XOR b'` fixed before the seed, the three seeded levels
//! all miss it with probability at most `sum_t 1 / binom(m_t, k_t)` (a union bound over
//! the levels), independent of `J_t`, the row weights and the number of blocks.
//! Production: `2^-145.64 + 2^-145.64 + 2^-163.44 = 2^-144.64`; toy: `2^-10.75`.

use crate::shape::{LevelShape, Shape};

/// `log2 binom(n, k)`, exact up to `f64` rounding of the summed logarithms.
#[must_use]
pub fn log2_binomial(n: u64, k: u64) -> f64 {
    assert!(k <= n, "k <= n");
    let k = k.min(n - k);
    (0..k)
        .map(|i| ((n - i) as f64 / (i + 1) as f64).log2())
        .sum()
}

/// `log2` of the per-level miss probability `1 / binom(m_t, k_t)`.
#[must_use]
pub fn level_error_log2(level: &LevelShape) -> f64 {
    -log2_binomial(level.rows as u64, level.k as u64)
}

/// `log2 sum_t 1 / binom(m_t, k_t)`: the bridge term `Pr_sd[M_3 M_2 M_1 delta = 0]`.
#[must_use]
pub fn bridge_error_log2(shape: &Shape) -> f64 {
    let errors = shape.levels.map(|level| level_error_log2(&level));
    let max = errors.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    max + errors.iter().map(|e| (e - max).exp2()).sum::<f64>().log2()
}

/// Bits of margin of the bridge term below `2^-target`: the number of seed re-derivations
/// (`log2 Q`) an adversary may grind before the bound exceeds `2^-target`.
#[must_use]
pub fn grinding_margin_bits(shape: &Shape, target: f64) -> f64 {
    -bridge_error_log2(shape) - target
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 5e-3
    }

    #[test]
    fn level_and_bridge_errors_of_the_production_and_toy_shapes() {
        let p = Shape::PRODUCTION.levels.map(|l| level_error_log2(&l));
        assert!(
            close(p[0], -145.64) && close(p[1], -145.64) && close(p[2], -163.44),
            "{p:?}"
        );
        assert!(close(bridge_error_log2(&Shape::PRODUCTION), -144.64));
        assert!(grinding_margin_bits(&Shape::PRODUCTION, 128.0) > 16.6);
        let t = Shape::TOY.levels.map(|l| level_error_log2(&l));
        assert!(
            close(t[0], -19.277) && close(t[1], -15.134) && close(t[2], -10.830),
            "{t:?}"
        );
        assert!(close(bridge_error_log2(&Shape::TOY), -10.755));
        // Exact small binomials.
        assert!(close(log2_binomial(64, 3), (41_664_f64).log2()));
        assert!(close(log2_binomial(16, 4), (1820_f64).log2()));
        assert_eq!(log2_binomial(9, 0), 0.0);
    }
}
