//! Fiat-Shamir challenger interfaces and implementations.

#[cfg(any(test, feature = "unsound-challenger"))]
pub use flock_transcript::challenger::RandomChallenger;
#[cfg(feature = "hash-count")]
pub use flock_transcript::challenger::fs_count;
pub use flock_transcript::challenger::{
    Challenger, FsChallenger, KIND_NONE, KIND_SCALAR, KIND_SLICE, OP_BYTES, OP_DOMAIN, OP_LABEL,
    OP_OBSERVE, OP_SQUEEZE, POW_SQUEEZE_COUNTER_TAG, grinding_bits_for_degree,
    has_leading_zero_bits, pow_has_leading_zero_bits, pow_squeeze_counter,
};

use flock_field::F128;

/// Slack of a proof-of-work nonce above its site's difficulty: a site of `g > 0` bits takes
/// nonces of at most `g + 7` bits, as Akita's `GRINDING_NONCE_SLACK_BITS` and Aerie's work
/// nonces (agg-mldsa patch, `docs/soundness.md`). The honest search
/// (`Challenger::grind_pow`, smallest passing nonce first) exceeds `2^(g + 7)` candidates with
/// probability `(1 - 2^-g)^(2^(g + 7)) <= e^-128`.
pub const POW_NONCE_SLACK_BITS: u32 = 7;

/// Whether `nonce` has the canonical width of a site of `bits` bits: zero at a site without
/// work, below `2^(bits + 7)` otherwise.
#[must_use]
pub const fn pow_nonce_is_canonical(nonce: u64, bits: u32) -> bool {
    if bits == 0 {
        nonce == 0
    } else {
        let width = bits.saturating_add(POW_NONCE_SLACK_BITS);
        width >= u64::BITS || nonce >> width == 0
    }
}

/// The verifier's proof-of-work checks with the canonical nonce width
/// ([`pow_nonce_is_canonical`]; agg-mldsa patch). Each method runs the challenger's
/// own check (which absorbs the nonce, keeping the transcript in lockstep with the prover)
/// and additionally rejects a nonce wider than `bits + 7` bits. Every verifier call site of
/// this crate uses these methods (`PATCH.md`).
pub trait CanonicalPow: Challenger {
    /// [`Challenger::verify_pow`] with the canonical width.
    fn verify_pow_canonical(&mut self, nonce: u64, bits: u32) -> bool {
        let ok = self.verify_pow(nonce, bits);
        ok && pow_nonce_is_canonical(nonce, bits)
    }

    /// [`Challenger::verify_pow_and_sample_f128`] with the canonical width.
    fn verify_pow_and_sample_f128_canonical(&mut self, nonce: u64, bits: u32) -> Option<F128> {
        let out = self.verify_pow_and_sample_f128(nonce, bits);
        out.filter(|_| pow_nonce_is_canonical(nonce, bits))
    }

    /// [`Challenger::verify_pow_and_sample_f128_vec`] with the canonical width.
    fn verify_pow_and_sample_f128_vec_canonical(
        &mut self,
        nonce: u64,
        bits: u32,
        n: usize,
    ) -> Option<Vec<F128>> {
        let out = self.verify_pow_and_sample_f128_vec(nonce, bits, n);
        out.filter(|_| pow_nonce_is_canonical(nonce, bits))
    }
}

impl<C: Challenger + ?Sized> CanonicalPow for C {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_width_is_bits_plus_seven() {
        assert!(pow_nonce_is_canonical(0, 0));
        assert!(!pow_nonce_is_canonical(1, 0));
        for bits in [1_u32, 6, 16, 25] {
            let limit = 1_u64 << (bits + POW_NONCE_SLACK_BITS);
            assert!(pow_nonce_is_canonical(limit - 1, bits));
            assert!(!pow_nonce_is_canonical(limit, bits));
            assert!(!pow_nonce_is_canonical(u64::MAX, bits));
        }
        assert!(pow_nonce_is_canonical(u64::MAX, 57));
        assert!(pow_nonce_is_canonical(u64::MAX, 128));
    }

    /// A nonce that passes the predicate but is wider than `bits + 7` bits is accepted by
    /// the challenger's own check and rejected by the canonical one; the honest nonce
    /// passes both, and both consume the transcript alike.
    #[test]
    fn wide_passing_nonces_are_rejected() {
        let bits = 2;
        let limit = 1_u64 << (bits + POW_NONCE_SLACK_BITS);
        let fresh = || {
            let mut ch = FsChallenger::new(b"agg-mldsa/canonical-pow-test");
            ch.observe_bytes(b"prefix");
            ch
        };
        let honest = fresh().grind_pow(bits);
        assert!(honest < limit);
        assert!(fresh().verify_pow_canonical(honest, bits));
        let wide = (limit..limit + 4096)
            .find(|&n| fresh().verify_pow(n, bits))
            .expect("a passing nonce above the window");
        assert!(!fresh().verify_pow_canonical(wide, bits));
        assert!(fresh().verify_pow_and_sample_f128_canonical(wide, bits).is_none());
        assert!(fresh()
            .verify_pow_and_sample_f128_vec_canonical(wide, bits, 3)
            .is_none());
        let (mut a, mut b) = (fresh(), fresh());
        assert_eq!(
            a.verify_pow_and_sample_f128_canonical(honest, bits),
            b.verify_pow_and_sample_f128(honest, bits)
        );
        assert_eq!(a.sample_f128(), b.sample_f128());
    }
}
