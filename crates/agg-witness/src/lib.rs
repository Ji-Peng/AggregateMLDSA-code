//! ML-DSA-44 witness generation for the aggregation PIOP (docs/spec.md Section 14, milestone M1).

pub mod crt;
pub mod derive;
pub mod expand;
pub mod native;
pub mod padding;
pub mod params;
pub mod tables;

/// Re-export of the RustCrypto ML-DSA crate used for key generation and signing.
pub use ml_dsa;

#[cfg(test)]
mod tests {
    use ml_dsa::signature::Verifier;
    use ml_dsa::{ExpandedSigningKey, KeyExport, MlDsa44, Seed};

    #[test]
    fn mldsa44_sign_verify_roundtrip_and_sizes() {
        let sk = ExpandedSigningKey::<MlDsa44>::from_seed(&Seed::default());
        let vk = sk.verifying_key();
        let msg = b"agg-witness smoke test";
        let sig = sk.sign_deterministic(msg, b"").expect("sign");
        vk.verify(msg, &sig).expect("verify");
        assert!(vk.verify(b"other message", &sig).is_err());
        // FIPS 204 Table 2: ML-DSA-44 public key 1312 bytes, signature 2420 bytes.
        assert_eq!(vk.to_bytes().len(), 1312);
        assert_eq!(sig.encode().len(), 2420);
    }

    #[test]
    fn reference_shake256_available() {
        use sha3::digest::{ExtendableOutput, Update, XofReader};
        let mut h = sha3::Shake256::default();
        h.update(b"");
        let mut out = [0u8; 4];
        h.finalize_xof().read(&mut out);
        // SHAKE256("") starts with 46 b9 dd 2b.
        assert_eq!(out, [0x46, 0xb9, 0xdd, 0x2b]);
    }
}
