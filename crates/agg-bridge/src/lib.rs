//! Parity bridge between the Fp-committed bits b and the GF(2) Keccak argument
//! (docs/spec.md Section 9, docs/bridge.md; Section 14, milestone M2).
//!
//! Shared by both sides and generic over the [`shape::Shape`] (production, and the toy shape
//! of the tests):
//!
//! * [`seed`]: the bridge seed `sd` to `M_1, M_2, M_3` (cSHAKE256, spec 9.1);
//! * [`parity`]: the integer level sums, the parities and the quotient (carry) columns of the
//!   three levels (9.2), and an independent checker of the identities at every Boolean point;
//! * [`carry`]: the carry digits (4.1) and the committed round-2 groups `G_1, G_2, G_3`;
//! * [`weights`]: the public weights `w_1, w_2, w_{3,t}` as prover tables and as the
//!   verifier's terminal evaluations (spec 7.6), over any `jolt_field::Ring`;
//! * [`final_vector`]: the packed `Y_sigma`, its length check and `Y(tau_3)`;
//! * [`gf2_walker`]: the GF(2) side, `M_3 M_2 M_1 b'` and its transpose (Flock bridge slot);
//! * [`soundness`]: `sum_t 1 / binom(m_t, k_t)` and the grinding margin.
//!
//! `agg-piop` uses these modules for the production shape; `agg-witness::bridge` keeps its own
//! production-only computation as an independent reference.

pub mod carry;
pub mod final_vector;
pub mod gf2_walker;
pub mod parity;
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
