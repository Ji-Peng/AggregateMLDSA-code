//! Sum-check PIOP for the ML-DSA-44 aggregation relation H.
//!
//! Aerie's projection, stage and transcript machinery is reached through
//! `aerie_prover` with its `internals` feature (re-exported below).
//!
//! Prover-side tables and commitments: [`manifest`] fixes the four
//! Akita groups, [`source_tables`] materializes them from `agg-witness` batches, [`pcs`]
//! commits them through `agg-pcs-config` and opens them in one batch, [`public_data`] holds
//! the verifier's precomputation and the negacyclic `eq` helpers, and [`virtual_polys`] the
//! virtual-table evaluators.
//!
//! The sum-check PIOP: [`kernel`] folds narrow tables and
//! batches instances per local domain, [`stage`] runs a declarative stage table as prover
//! or verifier, [`stage1`], [`stage2`], [`stage3`] are the three stages, [`weights`] the
//! public weight tables, [`transcript`] the Spongefish stream and event registry,
//! [`wire`] the proof bytes, [`prover`] and [`verifier`] the drivers, [`public_stream`]
//! the verifier's streamed public data. [`mode_b`] is the Keccak-free aggregate
//! (`prove_mode_b`, `verify_mode_b`).

#[allow(unsafe_code)]
pub mod alloc;
pub mod binding;
pub mod claims;
pub mod kernel;
pub mod lane_public;
pub mod lookup;
pub mod manifest;
pub mod mode_b;
pub mod opening;
pub mod pcs;
pub mod pow;
pub mod projection;
pub mod prover;
pub mod public_data;
pub mod public_stream;
pub mod range;
pub mod source_tables;
pub mod split_weight;
pub mod stage;
pub mod stage1;
pub mod stage2;
pub mod stage3;
pub mod tables;
pub mod transcript;
pub mod verifier;
pub mod virtual_polys;
pub mod weights;
pub mod wire;

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

/// Whether the unsound test stand-ins are compiled in (unit tests, or the `test-hooks`
/// feature): `prover::FixedMatrices`, `prover::NoKeccak`, `verifier::NoKeccakVerifier`,
/// and verification under Flock profile 0. A prover, verifier or benchmark build has
/// `false`.
#[must_use]
pub const fn test_hooks_enabled() -> bool {
    cfg!(any(test, feature = "test-hooks"))
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
        assert!(test_hooks_enabled(), "unit tests always have the stand-ins");
    }
}

/// Deterministic field samplers shared by the unit tests.
#[cfg(test)]
pub(crate) mod test_support {
    use super::{BaseField, OpeningField};
    use agg_witness::gen::SplitMix64;
    use jolt_field::{ExtField, Ring};

    /// A pseudo-random element of `K`.
    pub fn field(rng: &mut SplitMix64) -> OpeningField {
        OpeningField::from_base_slice(&[
            BaseField::from_u64(rng.next_u64()),
            BaseField::from_u64(rng.next_u64()),
        ])
    }

    /// A pseudo-random point of `K^len`.
    pub fn point(rng: &mut SplitMix64, len: usize) -> Vec<OpeningField> {
        (0..len).map(|_| field(rng)).collect()
    }

    /// The Boolean point of `index` (least significant coordinate first).
    pub fn boolean_point(index: usize, len: usize) -> Vec<OpeningField> {
        (0..len)
            .map(|bit| OpeningField::from_u64(((index >> bit) & 1) as u64))
            .collect()
    }
}
