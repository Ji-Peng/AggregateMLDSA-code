//! Deterministic ML-DSA-44 test records (key generation and signing through the `ml-dsa` crate).
//!
//! Only key generation and signing use the crate; everything the relation needs is recomputed by
//! this crate's own arithmetic (`native`, `expand`). The generators are deterministic in their
//! seeds so that tests and downstream fixtures are reproducible.

use crate::params::{ParamSet, Set44};
use ml_dsa::signature::Verifier;
use ml_dsa::{
    EncodedSignature, EncodedVerifyingKey, ExpandedSigningKey, MlDsa44,
    Signature, VerifyingKey, B32,
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

/// Key pair from a 32-byte seed for parameter set P.
pub fn keypair_for<P: ParamSet>(seed: &[u8; 32]) -> (ExpandedSigningKey<P::Signer>, Vec<u8>) {
    let sk = ExpandedSigningKey::<P::Signer>::from_seed(&B32::from(*seed));
    let pk = sk.verifying_key().encode().as_slice().to_vec();
    (sk, pk)
}

/// Key pair from a 32-byte seed (FIPS 204 KeyGen_internal through the crate).
pub fn keypair(seed: &[u8; 32]) -> (ExpandedSigningKey<MlDsa44>, Vec<u8>) {
    keypair_for::<Set44>(seed)
}

/// Signs `msg` in pure mode with empty context for parameter set P.
pub fn sign_for<P: ParamSet>(
    sk: &ExpandedSigningKey<P::Signer>,
    msg: &[u8],
    rnd: Option<&[u8; 32]>,
) -> Vec<u8> {
    let sig = match rnd {
        None => sk.sign_deterministic(msg, b"").expect("empty context"),
        Some(r) => sk.sign_internal(&[&[0u8, 0u8], msg], &B32::from(*r)),
    };
    sig.encode().as_slice().to_vec()
}

/// Signs `msg` in pure mode with empty context. `rnd = None` is the deterministic variant
/// (rnd = 0^32); `Some(r)` is the hedged variant with the given randomness.
pub fn sign(sk: &ExpandedSigningKey<MlDsa44>, msg: &[u8], rnd: Option<&[u8; 32]>) -> Vec<u8> {
    sign_for::<Set44>(sk, msg, rnd)
}

/// Verifies with the crate for parameter set P (pure mode, empty context).
pub fn crate_verify_for<P: ParamSet>(pk: &[u8], msg: &[u8], sig: &[u8]) -> bool {
    let Ok(enc_pk) = EncodedVerifyingKey::<P::Signer>::try_from(pk) else {
        return false;
    };
    let vk = VerifyingKey::<P::Signer>::decode(&enc_pk);
    let Ok(enc_sig) = EncodedSignature::<P::Signer>::try_from(sig) else {
        return false;
    };
    let Some(sigma) = Signature::<P::Signer>::decode(&enc_sig) else {
        return false;
    };
    vk.verify(msg, &sigma).is_ok() && vk.verify_with_context(msg, &[], &sigma)
}

/// Verifies with the crate (pure mode, empty context); false on any decoding failure.
pub fn crate_verify(pk: &[u8], msg: &[u8], sig: &[u8]) -> bool {
    crate_verify_for::<Set44>(pk, msg, sig)
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

/// Deterministic record `index` under `seed` for parameter set P.
pub fn record_for<P: ParamSet>(seed: u64, index: u64) -> Record {
    let (sk, pk) = keypair_for::<P>(&key_seed(seed, index));
    let msg = message(seed, index);
    let sig = sign_for::<P>(&sk, &msg, None);
    Record { pk, msg, sig }
}

/// Deterministic record `index` under `seed` (own key, deterministic signing).
pub fn record(seed: u64, index: u64) -> Record {
    record_for::<Set44>(seed, index)
}

/// Deterministic record with hedged signing for parameter set P.
pub fn record_hedged_for<P: ParamSet>(seed: u64, key_index: u64, index: u64) -> Record {
    let (sk, pk) = keypair_for::<P>(&key_seed(seed, key_index));
    let msg = message(seed, index);
    let mut rnd = [0u8; 32];
    derive_bytes(b"rnd", &[seed, key_index, index], &mut rnd);
    let sig = sign_for::<P>(&sk, &msg, Some(&rnd));
    Record { pk, msg, sig }
}

/// Deterministic record with hedged signing: key `key_index`, message and rnd from `index`.
pub fn record_hedged(seed: u64, key_index: u64, index: u64) -> Record {
    record_hedged_for::<Set44>(seed, key_index, index)
}

/// `n` deterministic records for parameter set P (own key each), generated in parallel.
pub fn records_for<P: ParamSet>(seed: u64, n: usize) -> Vec<Record> {
    (0..n as u64)
        .into_par_iter()
        .map(|i| record_for::<P>(seed, i))
        .collect()
}

/// `n` deterministic records (own key each), generated in parallel.
pub fn records(seed: u64, n: usize) -> Vec<Record> {
    records_for::<Set44>(seed, n)
}

/// `n` deterministic hedged records over `n_keys` keys for parameter set P.
pub fn records_shared_keys_for<P: ParamSet>(seed: u64, n: usize, n_keys: usize) -> Vec<Record> {
    assert!(n_keys > 0);
    let keys: Vec<_> = (0..n_keys as u64)
        .into_par_iter()
        .map(|k| keypair_for::<P>(&key_seed(seed, k)))
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
                sig: sign_for::<P>(sk, &msg, Some(&rnd)),
            }
        })
        .collect()
}

/// `n` deterministic hedged records over `n_keys` keys (records share keys round-robin).
pub fn records_shared_keys(seed: u64, n: usize, n_keys: usize) -> Vec<Record> {
    records_shared_keys_for::<Set44>(seed, n, n_keys)
}

/// A genuine key pair whose public key starts with the given `rho` (the matrix seed of
/// another key): the secrets `s1`, `s2` and `K` of the seed's own key, `t = A[rho] s1 + s2`
/// recomputed under `rho`, `(t1, t0) = Power2Round(t)` (FIPS 204 Algorithm 35) and
/// `tr = H(pk, 64)`, decoded by the crate through skDecode (Algorithm 25).
///
/// For tests of key handling: `rho` is chosen by the signer, so two honest-looking keys may
/// share it, and anything keyed by the 32-byte prefix of `pk` would confuse them.
pub fn keypair_with_rho(seed: &[u8; 32], rho: &[u8; 32]) -> (ExpandedSigningKey<MlDsa44>, Vec<u8>) {
    use crate::expand::expand_a;
    use crate::native::{bit_pack, bit_unpack, mod_q, pk_encode, tr, PublicKey};
    use crate::ntt::NttPrime;
    use crate::params::{D, ETA, K, L, N};
    use ml_dsa::ExpandedSigningKeyBytes;

    let (own, _) = keypair(seed);
    #[allow(deprecated)]
    let own_bytes = own.to_expanded();
    let own_bytes = own_bytes.as_slice();
    let poly_eta = 32 * 3; // bitlen(2 eta) = 3 bits per coefficient
    let s1_at = 32 + 32 + 64;
    let s2_at = s1_at + L * poly_eta;
    let t0_at = s2_at + K * poly_eta;
    let s1: Vec<[i64; N]> = (0..L)
        .map(|j| bit_unpack(&own_bytes[s1_at + j * poly_eta..][..poly_eta], ETA, ETA))
        .collect();
    let s2: Vec<[i64; N]> = (0..K)
        .map(|i| bit_unpack(&own_bytes[s2_at + i * poly_eta..][..poly_eta], ETA, ETA))
        .collect();
    let fips = NttPrime::fips();
    let (a_hat, _) = expand_a(rho);
    let s1_hat: Vec<[u64; N]> = s1.iter().map(|s| fips.ntt_signed(s)).collect();
    let mut t1 = [[0_u16; N]; K];
    let mut t0 = [[0_i64; N]; K];
    for i in 0..K {
        let mut acc = [0_u64; N];
        for j in 0..L {
            let product = fips.pointwise(&a_hat[i][j], &s1_hat[j]);
            for (x, y) in acc.iter_mut().zip(product) {
                *x = fips.add_mod(*x, y);
            }
        }
        let t = fips.intt(&acc);
        for m in 0..N {
            let r = mod_q(t[m] as i64 + s2[i][m]);
            let half = 1_i64 << (D - 1);
            let r0 = (r + half - 1).rem_euclid(1 << D) - (half - 1); // r mod+- 2^d
            t1[i][m] = ((r - r0) >> D) as u16;
            t0[i][m] = r0;
        }
    }
    let pk = pk_encode(&PublicKey { rho: *rho, t1 });
    let mut sk = Vec::with_capacity(own_bytes.len());
    sk.extend_from_slice(rho);
    sk.extend_from_slice(&own_bytes[32..64]);
    sk.extend_from_slice(&tr(&pk));
    sk.extend_from_slice(&own_bytes[s1_at..t0_at]);
    for row in &t0 {
        bit_pack(row, (1 << (D - 1)) - 1, 1 << (D - 1), &mut sk);
    }
    assert_eq!(sk.len(), own_bytes.len(), "skEncode length");
    let bytes = ExpandedSigningKeyBytes::<MlDsa44>::try_from(sk.as_slice()).expect("sk length");
    #[allow(deprecated)]
    let signer = ExpandedSigningKey::<MlDsa44>::from_expanded(&bytes);
    assert_eq!(
        signer.verifying_key().encode().as_slice(),
        pk.as_slice(),
        "the crate derives the same public key"
    );
    (signer, pk)
}

/// Position `(rho, j)` of the coefficient `w' = q - gamma2, h = 0` in [`hint_corner_record`].
pub const HINT_CORNER: (usize, usize) = (1, 17);
/// Position `(rho, j)` of the coefficient `w' = 0, h = 1` (the `e = -1` wrap of relation H-FIPS)
/// in [`hint_corner_record`].
pub const HINT_ZERO: (usize, usize) = (2, 101);

/// A synthetic but genuine FIPS 204 signature of hint weight exactly 80 with a coefficient
/// `w' = q - gamma2, h = 0` at [`HINT_CORNER`] and `w' = 0, h = 1` at [`HINT_ZERO`] (relation
/// H-FIPS). The public key has `t1 = 0` (a well-formed
/// 1,312-byte key; pkDecode, FIPS 204 Algorithm 23, accepts any bytes), so Algorithm 8 computes
/// `w'_Approx = A z` independently of `c` (line 9): `z` is drawn uniform in `[-B_z, B_z]` and two
/// of its coefficients are moved (a 2x2 solve modulo `q`) so that `A z` hits both targets, then
/// the hint gets 80 ones (at least one per `h = 1` case of the H-FIPS honest rule, none at the
/// corner), then `c~ = H(mu || w1Encode(UseHint(h, w')))` (line 12). Honest signing cannot place
/// a coefficient or a weight deliberately; random batches of 1K hit the corner with weight 80
/// with probability about 3e-4 (estimate).
///
/// # Panics
/// If no two-coefficient move is found or a hint case is missing (does not happen for the
/// seeds used in the tests).
#[allow(clippy::too_many_lines)]
pub fn hint_corner_record(seed: u64) -> Record {
    use crate::crt::az_integer;
    use crate::expand::ExpandedKey;
    use crate::native::{commitment_hash, ctr_q, decompose, mod_q, mu, sig_encode, use_hint, w1_encode, SmallK, VecL};
    use crate::ntt::pow_mod;
    use crate::params::{B_Z, GAMMA2, K, L, N, OMEGA, PK_BYTES, Q};
    const CORNER: (usize, usize) = HINT_CORNER;
    const ZERO: (usize, usize) = HINT_ZERO;

    let mut rho = [0u8; 32];
    derive_bytes(b"h-fips/rho", &[seed], &mut rho);
    let mut pk = rho.to_vec();
    pk.resize(PK_BYTES, 0);
    let key = ExpandedKey::from_bytes(&pk).unwrap();
    assert!(key.t1.iter().flatten().all(|&t| t == 0));
    let msg = message(seed ^ 0x4F1F5, 0);
    let mu = mu(&key.tr, &msg, &[]);

    // z uniform in [-B_z, B_z], then two coefficients moved so that (A z) = q - gamma2 at
    // CORNER and (A z) = 0 at ZERO (the only non-wrap residue with r1 = 0 and r0 <= 0, where
    // h = 1 gives the e = -1 wrap to w1' = 43).
    let mut rng = SplitMix64::new(0x4F1F5 ^ seed);
    let mut z: VecL = [[0; N]; L];
    for zi in z.iter_mut() {
        for x in zi.iter_mut() {
            *x = rng.below(2 * B_Z as u64 + 1) as i64 - B_Z;
        }
    }
    let targets = [(CORNER, Q - GAMMA2), (ZERO, 0)];
    let az0 = az_integer(&key, &z);
    let need: Vec<i64> = targets
        .iter()
        .map(|&((r, j), t)| mod_q(t - az0[r][j] as i64))
        .collect();
    // Coefficient of z[s][k] in (A z)[r][j] = sum_s (A[r][s] * z[s])_j (negacyclic).
    let coef = |(r, j): (usize, usize), s: usize, k: usize| -> i64 {
        mod_q(i64::from(key.a[r][s][(j + N - k) % N]) * if k <= j { 1 } else { -1 })
    };
    let inv = |a: i64| pow_mod(a as u64, (Q - 2) as u64, Q as u64) as i64;
    let mulq = |a: i64, b: i64| ((i128::from(a) * i128::from(b)).rem_euclid(i128::from(Q))) as i64;
    let mut moved = false;
    'search: for k1 in 0..N {
        for k2 in (k1 + 1)..N {
            let (s1, s2) = (k1 % L, k2 % L);
            let (a11, a12) = (coef(targets[0].0, s1, k1), coef(targets[0].0, s2, k2));
            let (a21, a22) = (coef(targets[1].0, s1, k1), coef(targets[1].0, s2, k2));
            let det = mod_q(mulq(a11, a22) - mulq(a12, a21));
            if det == 0 {
                continue;
            }
            let di = inv(det);
            let d1 = ctr_q(mulq(mod_q(mulq(need[0], a22) - mulq(a12, need[1])), di));
            let d2 = ctr_q(mulq(mod_q(mulq(a11, need[1]) - mulq(a21, need[0])), di));
            let (n1, n2) = (z[s1][k1] + d1, z[s2][k2] + d2);
            if n1.abs() <= B_Z && n2.abs() <= B_Z {
                z[s1][k1] = n1;
                z[s2][k2] = n2;
                moved = true;
                break 'search;
            }
        }
    }
    assert!(moved, "no two-coefficient move found");
    let az = az_integer(&key, &z);
    let w: [[i64; N]; K] = std::array::from_fn(|r| std::array::from_fn(|j| mod_q(az[r][j] as i64)));
    assert_eq!(w[CORNER.0][CORNER.1], Q - GAMMA2);
    assert_eq!(w[ZERO.0][ZERO.1], 0);

    // Hint: up to two coefficients of each class with h = 1 (wrap; w1' wraps 43 -> 0; w1' wraps
    // 0 -> 43, only w' = 0; plain sigma = 1; plain sigma = -1), then the first others up to
    // exactly 80.
    let class = |wq: i64| -> usize {
        if wq >= Q - GAMMA2 {
            return 0;
        }
        let (r1, r0) = decompose(wq);
        match (r0 > 0, r1) {
            (true, 43) => 1,
            (false, 0) => 2,
            (true, _) => 3,
            (false, _) => 4,
        }
    };
    let mut h: SmallK = [[0; N]; K];
    let mut per_class = [0usize; 5];
    let mut weight = 0usize;
    for r in 0..K {
        for j in 0..N {
            let c = class(w[r][j]);
            if (r, j) != CORNER && per_class[c] < 2 {
                per_class[c] += 1;
                h[r][j] = 1;
                weight += 1;
            }
        }
    }
    assert!(per_class.iter().all(|&c| c >= 1), "every class present: {per_class:?}");
    assert_eq!(h[ZERO.0][ZERO.1], 1);
    for r in 0..K {
        for j in 0..N {
            if weight < OMEGA && (r, j) != CORNER && h[r][j] == 0 {
                h[r][j] = 1;
                weight += 1;
            }
        }
    }
    assert_eq!(weight, OMEGA);
    let mut w1: SmallK = [[0; N]; K];
    for r in 0..K {
        for j in 0..N {
            w1[r][j] = use_hint(h[r][j] != 0, w[r][j]) as u8;
        }
    }
    let c_tilde = commitment_hash(&mu, &w1_encode(&w1));
    let sig = sig_encode(&c_tilde, &z, &h);
    Record { pk, msg, sig }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{Set65, Set87, PK_BYTES, SIG_BYTES};

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
    fn records_are_deterministic_and_valid_65() {
        let a = records_for::<Set65>(65, 4);
        let b = records_for::<Set65>(65, 4);
        assert_eq!(a, b);
        for r in &a {
            assert_eq!(r.pk.len(), Set65::PK_BYTES);
            assert_eq!(r.sig.len(), Set65::SIG_BYTES);
            assert!(crate_verify_for::<Set65>(&r.pk, &r.msg, &r.sig));
            assert!(crate::native::verify_for::<Set65>(&r.pk, &r.msg, &r.sig));
            assert!(!crate_verify_for::<Set65>(&r.pk, b"wrong message", &r.sig) || r.msg == b"wrong message");
            assert!(!crate::native::verify_for::<Set65>(&r.pk, b"wrong message", &r.sig) || r.msg == b"wrong message");
        }
        let h = record_hedged_for::<Set65>(65, 0, 5);
        assert!(crate_verify_for::<Set65>(&h.pk, &h.msg, &h.sig));
        assert!(crate::native::verify_for::<Set65>(&h.pk, &h.msg, &h.sig));
        let s = records_shared_keys_for::<Set65>(65, 4, 2);
        assert_eq!(s[0].pk, s[2].pk);
        assert!(s.iter().all(|r| crate_verify_for::<Set65>(&r.pk, &r.msg, &r.sig)));
        assert!(s.iter().all(|r| crate::native::verify_for::<Set65>(&r.pk, &r.msg, &r.sig)));
    }

    #[test]
    fn records_are_deterministic_and_valid_87() {
        let a = records_for::<Set87>(87, 4);
        let b = records_for::<Set87>(87, 4);
        assert_eq!(a, b);
        for r in &a {
            assert_eq!(r.pk.len(), Set87::PK_BYTES);
            assert_eq!(r.sig.len(), Set87::SIG_BYTES);
            assert!(crate_verify_for::<Set87>(&r.pk, &r.msg, &r.sig));
            assert!(crate::native::verify_for::<Set87>(&r.pk, &r.msg, &r.sig));
            assert!(!crate_verify_for::<Set87>(&r.pk, b"wrong message", &r.sig) || r.msg == b"wrong message");
            assert!(!crate::native::verify_for::<Set87>(&r.pk, b"wrong message", &r.sig) || r.msg == b"wrong message");
        }
        let h = record_hedged_for::<Set87>(87, 0, 5);
        assert!(crate_verify_for::<Set87>(&h.pk, &h.msg, &h.sig));
        assert!(crate::native::verify_for::<Set87>(&h.pk, &h.msg, &h.sig));
        let s = records_shared_keys_for::<Set87>(87, 4, 2);
        assert_eq!(s[0].pk, s[2].pk);
        assert!(s.iter().all(|r| crate_verify_for::<Set87>(&r.pk, &r.msg, &r.sig)));
        assert!(s.iter().all(|r| crate::native::verify_for::<Set87>(&r.pk, &r.msg, &r.sig)));
    }

    #[test]
    fn keys_sharing_rho_sign_and_verify() {
        let (_, victim) = keypair(&key_seed(9, 0));
        let mut rho = [0u8; 32];
        rho.copy_from_slice(&victim[..32]);
        let (sk, pk) = keypair_with_rho(&key_seed(9, 1), &rho);
        assert_eq!(pk[..32], victim[..32]);
        assert_ne!(pk, victim);
        let msg = message(9, 2);
        let sig = sign(&sk, &msg, None);
        assert!(crate_verify(&pk, &msg, &sig));
        assert!(crate::native::verify(&pk, &msg, &sig));
        assert!(!crate_verify(&victim, &msg, &sig));
        assert!(!crate::native::verify(&victim, &msg, &sig));
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
