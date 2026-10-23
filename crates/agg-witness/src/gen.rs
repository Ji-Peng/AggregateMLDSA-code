//! Deterministic ML-DSA-44 test records (key generation and signing through the `ml-dsa` crate).
//!
//! Only key generation and signing use the crate; everything the relation needs is recomputed by
//! this crate's own arithmetic (`native`, `expand`). The generators are deterministic in their
//! seeds so that tests and downstream fixtures are reproducible.

use ml_dsa::signature::Verifier;
use ml_dsa::{
    EncodedSignature, EncodedVerifyingKey, ExpandedSigningKey, MlDsa44, Signature, VerifyingKey,
    B32,
};
use rayon::prelude::*;
use sha3::digest::{ExtendableOutput, Update, XofReader};

/// SplitMix64 pseudo-random generator (deterministic test randomness, not cryptographic).
#[derive(Clone, Debug)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    /// New generator from a seed.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Next 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform value in [0, n) (n > 0), by rejection.
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0);
        let zone = u64::MAX - u64::MAX % n;
        loop {
            let x = self.next_u64();
            if x < zone {
                return x % n;
            }
        }
    }

    /// Fills a byte slice.
    pub fn fill(&mut self, out: &mut [u8]) {
        for chunk in out.chunks_mut(8) {
            let x = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&x[..chunk.len()]);
        }
    }
}

/// One public record with its signature: (pk, M, sigma), pure mode with empty context.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    /// 1,312-byte public key.
    pub pk: Vec<u8>,
    /// Message M (pure mode, empty context: M' = 0x00 || 0x00 || M).
    pub msg: Vec<u8>,
    /// 2,420-byte signature.
    pub sig: Vec<u8>,
}

/// Domain-separated SHAKE256 expansion used to derive seeds and messages.
pub fn derive_bytes(label: &[u8], parts: &[u64], out: &mut [u8]) {
    let mut h = sha3::Shake256::default();
    h.update(b"agg-witness/gen/v1/");
    h.update(label);
    for p in parts {
        h.update(&p.to_le_bytes());
    }
    h.finalize_xof().read(out);
}

/// Key pair from a 32-byte seed (FIPS 204 KeyGen_internal through the crate).
pub fn keypair(seed: &[u8; 32]) -> (ExpandedSigningKey<MlDsa44>, Vec<u8>) {
    let sk = ExpandedSigningKey::<MlDsa44>::from_seed(&B32::from(*seed));
    let pk = sk.verifying_key().encode().as_slice().to_vec();
    (sk, pk)
}

/// Signs `msg` in pure mode with empty context. `rnd = None` is the deterministic variant
/// (rnd = 0^32); `Some(r)` is the hedged variant with the given randomness.
pub fn sign(sk: &ExpandedSigningKey<MlDsa44>, msg: &[u8], rnd: Option<&[u8; 32]>) -> Vec<u8> {
    let sig = match rnd {
        None => sk.sign_deterministic(msg, b"").expect("empty context"),
        Some(r) => sk.sign_internal(&[&[0u8, 0u8], msg], &B32::from(*r)),
    };
    sig.encode().as_slice().to_vec()
}

/// Verifies with the crate (pure mode, empty context); false on any decoding failure.
pub fn crate_verify(pk: &[u8], msg: &[u8], sig: &[u8]) -> bool {
    let Ok(enc_pk) = EncodedVerifyingKey::<MlDsa44>::try_from(pk) else {
        return false;
    };
    let vk = VerifyingKey::<MlDsa44>::decode(&enc_pk);
    let Ok(enc_sig) = EncodedSignature::<MlDsa44>::try_from(sig) else {
        return false;
    };
    let Some(sigma) = Signature::<MlDsa44>::decode(&enc_sig) else {
        return false;
    };
    vk.verify(msg, &sigma).is_ok() && vk.verify_with_context(msg, &[], &sigma)
}

/// Key seed of record `index` under `seed` when every record has its own key.
pub fn key_seed(seed: u64, index: u64) -> [u8; 32] {
    let mut s = [0u8; 32];
    derive_bytes(b"key", &[seed, index], &mut s);
    s
}

/// Message of record `index` under `seed`: length in [0, 160), bytes from SHAKE256.
pub fn message(seed: u64, index: u64) -> Vec<u8> {
    let mut len = [0u8; 2];
    derive_bytes(b"msg-len", &[seed, index], &mut len);
    let n = usize::from(u16::from_le_bytes(len)) % 160;
    let mut m = vec![0u8; n];
    derive_bytes(b"msg", &[seed, index], &mut m);
    m
}

/// Deterministic record `index` under `seed` (own key, deterministic signing).
pub fn record(seed: u64, index: u64) -> Record {
    let (sk, pk) = keypair(&key_seed(seed, index));
    let msg = message(seed, index);
    let sig = sign(&sk, &msg, None);
    Record { pk, msg, sig }
}

/// Deterministic record with hedged signing: key `key_index`, message and rnd from `index`.
pub fn record_hedged(seed: u64, key_index: u64, index: u64) -> Record {
    let (sk, pk) = keypair(&key_seed(seed, key_index));
    let msg = message(seed, index);
    let mut rnd = [0u8; 32];
    derive_bytes(b"rnd", &[seed, key_index, index], &mut rnd);
    let sig = sign(&sk, &msg, Some(&rnd));
    Record { pk, msg, sig }
}

/// `n` deterministic records (own key each), generated in parallel.
pub fn records(seed: u64, n: usize) -> Vec<Record> {
    (0..n as u64)
        .into_par_iter()
        .map(|i| record(seed, i))
        .collect()
}

/// `n` deterministic hedged records over `n_keys` keys (records share keys round-robin).
pub fn records_shared_keys(seed: u64, n: usize, n_keys: usize) -> Vec<Record> {
    assert!(n_keys > 0);
    let keys: Vec<_> = (0..n_keys as u64)
        .into_par_iter()
        .map(|k| keypair(&key_seed(seed, k)))
        .collect();
    (0..n)
        .into_par_iter()
        .map(|i| {
            let (sk, pk) = &keys[i % n_keys];
            let msg = message(seed ^ 0x5eed, i as u64);
            let mut rnd = [0u8; 32];
            derive_bytes(b"rnd-shared", &[seed, i as u64], &mut rnd);
            Record {
                pk: pk.clone(),
                msg: msg.clone(),
                sig: sign(sk, &msg, Some(&rnd)),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{PK_BYTES, SIG_BYTES};

    #[test]
    fn records_are_deterministic_and_valid() {
        let a = records(1, 4);
        let b = records(1, 4);
        assert_eq!(a, b);
        for r in &a {
            assert_eq!(r.pk.len(), PK_BYTES);
            assert_eq!(r.sig.len(), SIG_BYTES);
            assert!(crate_verify(&r.pk, &r.msg, &r.sig));
            assert!(!crate_verify(&r.pk, b"wrong message", &r.sig) || r.msg == b"wrong message");
        }
        let h = record_hedged(1, 0, 5);
        assert!(crate_verify(&h.pk, &h.msg, &h.sig));
        let s = records_shared_keys(2, 6, 2);
        assert_eq!(s[0].pk, s[2].pk);
        assert!(s.iter().all(|r| crate_verify(&r.pk, &r.msg, &r.sig)));
    }

    #[test]
    fn splitmix_below_is_in_range() {
        let mut r = SplitMix64::new(3);
        for n in [1u64, 2, 3, 44, 1000] {
            for _ in 0..100 {
                assert!(r.below(n) < n);
            }
        }
    }
}
