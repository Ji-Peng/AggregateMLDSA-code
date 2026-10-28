//! Keccak argument for the witness-dependent SHAKE256 call of ML-DSA verification
//! (`docs/spec.md`). Built on Flock.
//! Design, costs and measurements: `docs/keccak.md`.
//!
//! - [`keccak`]: Keccak-f[1600] on lanes, `phi = pi o rho o theta` and its transpose.
//! - [`gate`]: the chi-only gate (chains of permutations with absorptions), its rows,
//!   lincheck walker and Flock registry type.
//! - [`sponge`]: the SHAKE256 gate for `c~ = H(mu || w1Encode)`.
//! - [`witness`]: batch-major packing into a Flock union slot.
//! - [`prover`]: prove/verify of a sub-batch (Ligerito `Fast` or `Secure`), proof bytes.
//! - [`flock_union`]: boolean Flock union proofs (one or more types) with the caller's
//!   evaluation claims (packed-direct and ring-switched bit claims) and precomputed
//!   commitments.
//! - [`tied`]: the tied part (docs/parity-chain.md): the sponge instance and
//!   the virtual parity chain whose bit claim its opening proves, the mode-A Flock argument of
//!   one part.

// Index loops mirror the (x, y, z) bit arithmetic of FIPS 202.
#![allow(clippy::needless_range_loop)]

#[cfg(test)]
mod dsl_ref;
pub mod flock_union;
pub mod gate;
pub mod keccak;
pub mod prover;
pub mod sponge;
pub mod tied;
pub mod witness;

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
