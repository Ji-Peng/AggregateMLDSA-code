//! Parity bridge between the Fp-committed bits b and the GF(2) Keccak argument
//! (`docs/bridge.md`).
//!
//! Shared by both sides and generic over the [`shape::Shape`] (production, and the toy shape
//! of the tests):
//!
//! * [`seed`]: the bridge seed `sd` to `M_1, M_2, M_3` (cSHAKE256);
//! * [`parity`]: the integer level sums, the parities and the quotient (carry) columns of the
//!   three levels, and an independent checker of the identities at every Boolean point;
//! * [`weights`]: the public weights `w_1, w_2, w_{3,t}` as prover tables and as the
//!   verifier's terminal evaluations, over any `jolt_field::Ring`;
//! * [`final_vector`]: the packed `Y_sigma`, its length check and `Y(tau_3)`;
//! * [`gf2_walker`]: the GF(2) side, the forward composite `M_3 M_2 M_1 b'` of a segment;
//! * [`parity_chain`]: the GF(2) side as one sum-check over the committed bits;
//! * [`soundness`]: `sum_t 1 / binom(m_t, k_t)` and the grinding margin;
//! * [`pow`]: the proof-of-work difficulties of the 128-bit schedule (`docs/soundness.md`);
//! * [`sched`]: the hook through which the prover chooses the pool of the Flock opening.
//!
//! `agg-piop` uses these modules for the production shape; `agg-witness::bridge` keeps its own
//! production-only computation as an independent reference.

pub mod final_vector;
pub mod gf2_walker;
pub mod parity;
pub mod parity_chain;
pub mod pow;
pub mod sched;
pub mod seed;
pub mod shape;
pub mod soundness;
pub mod weights;

pub use shape::{Domains, LevelShape, Shape};

#[cfg(test)]
mod tests {
    #[test]
    fn field_crates_link() {
        assert!(std::any::type_name::<flock_field::F128>().contains("F128"));
        // `Prime64Offset59` is an alias, so only check that it resolves.
        assert!(!std::any::type_name::<jolt_field::Prime64Offset59>().is_empty());
        let _ = std::any::type_name::<agg_witness::ml_dsa::MlDsa44>();
    }
}
