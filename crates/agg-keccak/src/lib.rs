//! Keccak argument for the witness-dependent SHAKE256 call of ML-DSA verification
//! (docs/spec.md Section 8; Section 14, milestone M3). Built on Flock HEAD.

pub mod bridge_slot;
pub mod gate;
pub mod segment;
pub mod sponge;

pub use aerie_keccak;
pub use flock_core;
pub use flock_field;
pub use flock_prover;

/// Whether this build was compiled with the opt-in `avx512` feature.
#[must_use]
pub const fn avx512_enabled() -> bool {
    cfg!(feature = "avx512")
}

#[cfg(test)]
mod tests {
    #[test]
    fn aerie_batched_shake_matches_reference() {
        use sha3::digest::{ExtendableOutput, Update, XofReader};
        let inputs: [&[u8]; 2] = [b"", b"agg-keccak"];
        let batch = aerie_keccak::shake256_64_batch(&inputs);
        for (input, got) in inputs.iter().zip(batch.iter()) {
            let mut h = sha3::Shake256::default();
            h.update(input);
            let mut want = [0u8; 64];
            h.finalize_xof().read(&mut want);
            assert_eq!(got, &want);
        }
    }

    #[test]
    fn flock_crates_link() {
        assert!(std::any::type_name::<flock_field::F128>().contains("F128"));
    }
}
