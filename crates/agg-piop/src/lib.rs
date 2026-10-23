//! Sum-check PIOP for the ML-DSA-44 aggregation relation H
//! (docs/spec.md Sections 5-7, 10-11; Section 14, milestone M2).
//!
//! Aerie's projection, stage and transcript machinery is reached through
//! `aerie_prover` with its `internals` feature (re-exported below).

pub mod opening;
pub mod range;
pub mod stage1;
pub mod stage2;
pub mod stage3;
pub mod verifier;
pub mod virtual_polys;
pub mod weights;

pub use aerie_falcon;
pub use aerie_prover;

/// Base field of the PIOP: Akita's fp64 field, as used by Aerie.
pub type BaseField = aerie_prover::BaseField;
/// Extension field in which openings and challenges live.
pub type OpeningField = aerie_prover::OpeningField;

/// Whether this build was compiled with the opt-in `avx512` feature.
#[must_use]
pub const fn avx512_enabled() -> bool {
    cfg!(feature = "avx512")
}

/// The SIMD ceiling Aerie's runtime-dispatched kernels selected on this CPU
/// (reported by the benchmark harness next to every measurement).
#[must_use]
pub fn aerie_simd_backend() -> &'static str {
    aerie_simd::Backend::selected().name()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aerie_reexports_resolve() {
        assert_eq!(
            std::any::type_name::<BaseField>(),
            std::any::type_name::<aerie_falcon::BaseField>()
        );
        assert!(!aerie_prover::PROTOCOL_DOMAIN.is_empty());
        let backend = aerie_simd_backend();
        assert!(["scalar", "neon", "avx2", "avx512"].contains(&backend));
        assert_eq!(avx512_enabled(), cfg!(feature = "avx512"));
    }
}
