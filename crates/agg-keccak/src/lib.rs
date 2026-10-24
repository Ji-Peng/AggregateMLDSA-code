//! Keccak argument for the witness-dependent SHAKE256 call of ML-DSA verification
//! (docs/spec.md Section 8; Section 14, milestone M3). Built on Flock HEAD.
//! Design, costs and measurements: `docs/keccak.md`.
//!
//! - [`keccak`]: Keccak-f[1600] on lanes, `phi = pi o rho o theta` and its transpose.
//! - [`gate`]: the chi-only gate (chains of permutations with absorptions), its rows,
//!   lincheck walker and Flock registry type.
//! - [`sponge`]: the SHAKE256 gate for `c~ = H(mu || w1Encode)`.
//! - [`witness`]: batch-major packing into a Flock union slot.
//! - [`prover`]: prove/verify of a sub-batch (Ligerito `Fast` or `Secure`), proof bytes.
//! - [`bridge_slot`]: the bridge type, whose seeded rows prove `Y_sigma = M_3 M_2 M_1 b'`
//!   with a factored `LincheckCircuit` walker (docs/bridge.md Section 5).
//! - [`segment`]: the bridged segment (sponge rows and bridge slot in one union proof,
//!   committed before the bridge seed) and the standalone bridge slot proof; superseded
//!   by [`tied`] for mode A (kept for the toy-shape slot tests and as the reference).
//! - [`flock_union`]: boolean Flock union proofs (one or more types) with the caller's
//!   evaluation claims and precomputed commitments.
//! - [`bridge_union`]: the two-type bridge union (level-1 rows with `A_0 = M_1`, one
//!   levels-2-3 row), the factored bridge instance of a tied part (docs/bridge.md Section 5b).
//! - [`tied`]: the tied part, a sponge instance and a bridge union tied by evaluation
//!   claims (docs/keccak.md Sections 9 to 11), the mode-A Flock argument of one part.

// Index loops mirror the (x, y, z) bit arithmetic of FIPS 202.
#![allow(clippy::needless_range_loop)]

pub mod bridge_slot;
pub mod bridge_union;
#[cfg(test)]
mod dsl_ref;
pub mod flock_union;
pub mod gate;
pub mod keccak;
pub mod prover;
pub mod segment;
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
