//! Parity bridge between the Fp-committed bits b and the GF(2) Keccak argument
//! (docs/spec.md Section 9; Section 14, milestone M2).

pub mod carry;
pub mod gf2_walker;
pub mod parity;
pub mod seed;
pub mod soundness;

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
