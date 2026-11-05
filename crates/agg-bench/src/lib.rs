//! Benchmark harness: prover, verifier (cached and uncached) and memory.
//! Local runs stay at <= 1K signatures and run on
//! CPUs 0,1,3-15 (never CPU 2); 32K runs go to the cloud machine.

pub mod harness;
pub mod memory;

/// Median of a non-empty sample (benchmarks report medians).
/// For an even count it returns the mean of the two middle values.
#[must_use]
pub fn median(samples: &mut [f64]) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    samples.sort_by(f64::total_cmp);
    let n = samples.len();
    Some(if n % 2 == 1 {
        samples[n / 2]
    } else {
        (samples[n / 2 - 1] + samples[n / 2]) / 2.0
    })
}

/// Build description printed with every benchmark result: the `agg-simd` backend (named by
/// its `AGG_ISA` tier), the PIOP table layout and the Aerie SIMD backend.
#[must_use]
pub fn build_info() -> String {
    use agg_simd::detect::{backend, Backend};
    let simd = match backend() {
        Backend::Avx512 => "avx512ifma",
        Backend::Avx512F => "avx512",
        Backend::Avx2 => "avx2",
        Backend::Portable => "portable",
    };
    let layout = match agg_piop::kernel::layout() {
        agg_piop::kernel::Layout::Plane => "plane",
        agg_piop::kernel::Layout::Row => "row",
    };
    format!(
        "agg_simd_backend={simd} piop_layout={layout} aerie_simd_backend={}",
        agg_piop::aerie_simd_backend()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_odd_even_empty() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), Some(2.5));
        assert_eq!(median(&mut []), None);
        assert!(build_info().contains("piop_layout="));
        assert!(build_info().contains("aerie_simd_backend="));
    }
}
