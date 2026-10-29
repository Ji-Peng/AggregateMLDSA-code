//! Native ML-DSA-44 verification exposing the intermediate values the relation needs.
//!
//! Own implementation of the FIPS 204 verifier-side algorithms: encodings (Algorithms 16 to 23,
//! 26 to 28), hashes (tr, mu, SampleInBall, Algorithm 29), rounding (Algorithms 36 to 40) and
//! `ML-DSA.Verify_internal` (Algorithm 8). The `ml-dsa` crate is used only to generate keys and
//! signatures and as an oracle in tests; nothing here calls its internals.

use crate::expand::ExpandedKey;
use crate::ntt::NttPrime;
use crate::params::*;
use sha3::digest::{ExtendableOutput, Update, XofReader};

/// A polynomial with signed integer coefficients.
pub type Poly = [i64; N];
/// A k-vector of polynomials (rows of A).
pub type VecK = [Poly; K];
/// An l-vector of polynomials (columns of A).
pub type VecL = [Poly; L];
/// Per-coefficient small values (hint bits, w1 values) for the k rows.
pub type SmallK = [[u8; N]; K];

// ---------------------------------------------------------------------------------------------
// Modular helpers.
// ---------------------------------------------------------------------------------------------

/// Centered representative ctr_q(x) in [-(q-1)/2, (q-1)/2].
#[inline]
pub fn ctr_q(x: i64) -> i64 {
    let r = x.rem_euclid(Q);
    if r > (Q - 1) / 2 {
        r - Q
    } else {
        r
    }
}

/// x mod q in [0, q).
#[inline]
pub fn mod_q(x: i64) -> i64 {
    x.rem_euclid(Q)
}

/// x mod+- alpha for even alpha: the representative in (-alpha/2, alpha/2] (FIPS 204 Section 2.3).
#[inline]
pub fn mod_pm(x: i64, alpha: i64) -> i64 {
    debug_assert!(alpha % 2 == 0);
    let r = x.rem_euclid(alpha);
    if r > alpha / 2 {
        r - alpha
    } else {
        r
    }
}

// ---------------------------------------------------------------------------------------------
// Rounding (FIPS 204 Algorithms 36 to 40).
// ---------------------------------------------------------------------------------------------

/// Decompose(r) (Algorithm 36): (r1, r0) with r = r1 (2 gamma2) + r0 mod q.
pub fn decompose(r: i64) -> (i64, i64) {
    let r_plus = mod_q(r);
    let r0 = mod_pm(r_plus, 2 * GAMMA2);
    if r_plus - r0 == Q - 1 {
        (0, r0 - 1)
    } else {
        ((r_plus - r0) / (2 * GAMMA2), r0)
    }
}

/// HighBits(r) (Algorithm 37).
pub fn high_bits(r: i64) -> i64 {
    decompose(r).0
}

/// LowBits(r) (Algorithm 38).
pub fn low_bits(r: i64) -> i64 {
    decompose(r).1
}

/// MakeHint(z, r) (Algorithm 39).
pub fn make_hint(z: i64, r: i64) -> bool {
    high_bits(r) != high_bits(r + z)
}

/// UseHint(h, r) (Algorithm 40).
pub fn use_hint(h: bool, r: i64) -> i64 {
    let (r1, r0) = decompose(r);
    if h && r0 > 0 {
        (r1 + 1).rem_euclid(M_W)
    } else if h {
        (r1 - 1).rem_euclid(M_W)
    } else {
        r1
    }
}

// ---------------------------------------------------------------------------------------------
// Bit packing (FIPS 204 Algorithms 9 to 19; bit strings are little-endian within bytes).
// ---------------------------------------------------------------------------------------------

fn pack_bits(values: impl Iterator<Item = u64>, bits: usize, out: &mut Vec<u8>) {
    let mut acc: u128 = 0;
    let mut n = 0usize;
    for v in values {
        debug_assert!(v < 1 << bits);
        acc |= u128::from(v) << n;
        n += bits;
        while n >= 8 {
            out.push(acc as u8);
            acc >>= 8;
            n -= 8;
        }
    }
    debug_assert_eq!(n, 0, "256 coefficients always fill whole bytes");
}

fn unpack_bits(bytes: &[u8], bits: usize) -> [u64; N] {
    debug_assert_eq!(bytes.len(), 32 * bits);
    assert!((1..=56).contains(&bits), "a coefficient and a byte fit the 64-bit window");
    let mut out = [0u64; N];
    let mask = (1u64 << bits) - 1;
    // Little-endian bit stream: `acc` holds the `n` unread bits, n < bits + 8 <= 64.
    let (mut acc, mut n) = (0u64, 0usize);
    let mut next = bytes.iter();
    for o in out.iter_mut() {
        while n < bits {
            acc |= u64::from(*next.next().expect("32 bits bytes")) << n;
            n += 8;
        }
        *o = acc & mask;
        acc >>= bits;
        n -= bits;
    }
    out
}

/// SimpleBitPack(w, b) (Algorithm 16) with bitlen(b) = `bits`.
pub fn simple_bit_pack(w: &[u64; N], bits: usize, out: &mut Vec<u8>) {
    pack_bits(w.iter().copied(), bits, out);
}

/// SimpleBitUnpack(v, b) (Algorithm 18) with bitlen(b) = `bits`.
pub fn simple_bit_unpack(v: &[u8], bits: usize) -> [u64; N] {
    unpack_bits(v, bits)
}

/// BitPack(w, a, b) (Algorithm 17): packs b - w_i with bitlen(a + b) bits.
pub fn bit_pack(w: &Poly, a: i64, b: i64, out: &mut Vec<u8>) {
    let bits = 64 - ((a + b) as u64).leading_zeros() as usize;
    pack_bits(
        w.iter().map(|&x| {
            assert!(-a <= x && x <= b, "BitPack input out of range");
            (b - x) as u64
        }),
        bits,
        out,
    );
}

/// BitUnpack(v, a, b) (Algorithm 19).
pub fn bit_unpack(v: &[u8], a: i64, b: i64) -> Poly {
    let bits = 64 - ((a + b) as u64).leading_zeros() as usize;
    let raw = unpack_bits(v, bits);
    let mut w = [0i64; N];
    for i in 0..N {
        w[i] = b - raw[i] as i64;
    }
    w
}

/// HintBitPack(h) (Algorithm 20). Panics if the weight exceeds omega.
pub fn hint_bit_pack(h: &SmallK) -> [u8; OMEGA + K] {
    let mut y = [0u8; OMEGA + K];
    let mut index = 0usize;
    for i in 0..K {
        for j in 0..N {
            if h[i][j] != 0 {
                assert!(index < OMEGA, "hint weight exceeds omega");
                y[index] = j as u8;
                index += 1;
            }
        }
        y[OMEGA + i] = index as u8;
    }
    y
}

/// HintBitUnpack(y) (Algorithm 21); `None` is the FIPS symbol "bottom".
pub fn hint_bit_unpack(y: &[u8]) -> Option<SmallK> {
    assert_eq!(y.len(), OMEGA + K);
    let mut h = [[0u8; N]; K];
    let mut index = 0usize;
    for i in 0..K {
        let end = usize::from(y[OMEGA + i]);
        if end < index || end > OMEGA {
            return None;
        }
        let first = index;
        while index < end {
            if index > first && y[index - 1] >= y[index] {
                return None;
            }
            h[i][usize::from(y[index])] = 1;
            index += 1;
        }
    }
    if y[index..OMEGA].iter().any(|&b| b != 0) {
        return None;
    }
    Some(h)
}

/// Public key decoded by pkDecode (Algorithm 23).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicKey {
    /// Seed rho of A.
    pub rho: [u8; 32],
    /// t1 with coefficients in [0, 2^10 - 1].
    pub t1: [[u16; N]; K],
}

/// pkDecode (Algorithm 23); `None` on a wrong length.
pub fn pk_decode(pk: &[u8]) -> Option<PublicKey> {
    if pk.len() != PK_BYTES {
        return None;
    }
    let mut rho = [0u8; 32];
    rho.copy_from_slice(&pk[..32]);
    let mut t1 = [[0u16; N]; K];
    for (i, row) in t1.iter_mut().enumerate() {
        let off = 32 + i * 32 * T1_BITS;
        let w = simple_bit_unpack(&pk[off..off + 32 * T1_BITS], T1_BITS);
        for j in 0..N {
            row[j] = w[j] as u16;
        }
    }
    Some(PublicKey { rho, t1 })
}

/// pkEncode (Algorithm 22).
pub fn pk_encode(pk: &PublicKey) -> Vec<u8> {
    let mut out = Vec::with_capacity(PK_BYTES);
    out.extend_from_slice(&pk.rho);
    for row in &pk.t1 {
        let mut w = [0u64; N];
        for j in 0..N {
            w[j] = u64::from(row[j]);
        }
        simple_bit_pack(&w, T1_BITS, &mut out);
    }
    out
}

/// A decoded signature (c~, z, h) of sigDecode (Algorithm 27).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedSig {
    /// Commitment hash c~.
    pub c_tilde: [u8; C_TILDE_BYTES],
    /// Response z with coefficients in [-gamma1 + 1, gamma1].
    pub z: VecL,
    /// Hint h (0/1 per coefficient).
    pub h: SmallK,
}

/// Why a signature fails to decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SigDecodeError {
    /// Length differs from 2,420 bytes.
    WrongLength,
    /// HintBitUnpack returned bottom (Algorithm 8 line 3).
    MalformedHint,
}

/// sigDecode (Algorithm 27) with the hint check of Algorithm 8 line 3.
pub fn sig_decode(sig: &[u8]) -> Result<DecodedSig, SigDecodeError> {
    if sig.len() != SIG_BYTES {
        return Err(SigDecodeError::WrongLength);
    }
    let mut c_tilde = [0u8; C_TILDE_BYTES];
    c_tilde.copy_from_slice(&sig[..C_TILDE_BYTES]);
    let mut z = [[0i64; N]; L];
    for (i, zi) in z.iter_mut().enumerate() {
        let off = C_TILDE_BYTES + i * 32 * Z_BITS;
        *zi = bit_unpack(&sig[off..off + 32 * Z_BITS], GAMMA1 - 1, GAMMA1);
    }
    let off = C_TILDE_BYTES + L * 32 * Z_BITS;
    let h = hint_bit_unpack(&sig[off..]).ok_or(SigDecodeError::MalformedHint)?;
    Ok(DecodedSig { c_tilde, z, h })
}

/// sigEncode (Algorithm 26). Panics if z leaves [-gamma1 + 1, gamma1] or the hint weight exceeds omega.
pub fn sig_encode(c_tilde: &[u8; C_TILDE_BYTES], z: &VecL, h: &SmallK) -> Vec<u8> {
    let mut out = Vec::with_capacity(SIG_BYTES);
    out.extend_from_slice(c_tilde);
    for zi in z {
        bit_pack(zi, GAMMA1 - 1, GAMMA1, &mut out);
    }
    out.extend_from_slice(&hint_bit_pack(h));
    debug_assert_eq!(out.len(), SIG_BYTES);
    out
}

/// w1Encode (Algorithm 28): SimpleBitPack(w1[i], 43) for each row, 768 bytes.
pub fn w1_encode(w1: &SmallK) -> [u8; W1_ENCODE_BYTES] {
    let mut out = Vec::with_capacity(W1_ENCODE_BYTES);
    for row in w1 {
        let mut w = [0u64; N];
        for j in 0..N {
            assert!(i64::from(row[j]) < M_W, "w1 coefficient out of [0, 44)");
            w[j] = u64::from(row[j]);
        }
        simple_bit_pack(&w, W1_BITS, &mut out);
    }
    out.try_into().expect("768 bytes")
}

/// Inverse of w1Encode for any 768-byte string (values in [0, 64)).
pub fn w1_decode(bytes: &[u8; W1_ENCODE_BYTES]) -> SmallK {
    let mut w1 = [[0u8; N]; K];
    for (i, row) in w1.iter_mut().enumerate() {
        let off = i * 32 * W1_BITS;
        let w = simple_bit_unpack(&bytes[off..off + 32 * W1_BITS], W1_BITS);
        for j in 0..N {
            row[j] = w[j] as u8;
        }
    }
    w1
}

// ---------------------------------------------------------------------------------------------
// Hashes: H = SHAKE256 (FIPS 204 Section 3.7).
// ---------------------------------------------------------------------------------------------

/// H(parts..., out_len) = SHAKE256 of the concatenation.
pub fn shake256(parts: &[&[u8]], out: &mut [u8]) {
    let mut h = sha3::Shake256::default();
    for p in parts {
        h.update(p);
    }
    h.finalize_xof().read(out);
}

/// tr = H(pk, 64) (Algorithm 8 line 6).
pub fn tr(pk: &[u8]) -> [u8; TR_BYTES] {
    let mut out = [0u8; TR_BYTES];
    shake256(&[pk], &mut out);
    out
}

/// mu = H(tr || M', 64) with M' = 0x00 || |ctx| || ctx || M (Algorithms 3 and 8, pure mode).
pub fn mu(tr: &[u8; TR_BYTES], msg: &[u8], ctx: &[u8]) -> [u8; TR_BYTES] {
    assert!(ctx.len() <= 255);
    let mut out = [0u8; TR_BYTES];
    shake256(&[tr, &[0u8, ctx.len() as u8], ctx, msg], &mut out);
    out
}

/// c~' = H(mu || w1Encode(w1), lambda/4) (Algorithm 8 line 12).
pub fn commitment_hash(
    mu: &[u8; TR_BYTES],
    w1_bytes: &[u8; W1_ENCODE_BYTES],
) -> [u8; C_TILDE_BYTES] {
    let mut out = [0u8; C_TILDE_BYTES];
    shake256(&[mu, w1_bytes], &mut out);
    out
}

/// The challenge c = SampleInBall(c~) (Algorithm 29), as coefficients and as its tau positions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Challenge {
    /// Coefficients in {-1, 0, 1}, exactly tau nonzero.
    pub coeffs: [i8; N],
    /// Positions of the nonzero coefficients, increasing.
    pub positions: Vec<u8>,
}

/// SampleInBall(c~) (Algorithm 29) with H = SHAKE256.
pub fn sample_in_ball(c_tilde: &[u8]) -> Challenge {
    let mut h = sha3::Shake256::default();
    h.update(c_tilde);
    let mut reader = h.finalize_xof();
    let mut s = [0u8; 8];
    reader.read(&mut s);
    let signs = u64::from_le_bytes(s);
    let mut c = [0i8; N];
    for i in (N - TAU)..N {
        let mut j = [0u8; 1];
        reader.read(&mut j);
        while usize::from(j[0]) > i {
            reader.read(&mut j);
        }
        let j = usize::from(j[0]);
        c[i] = c[j];
        let bit = (signs >> (i + TAU - N)) & 1;
        c[j] = if bit == 1 { -1 } else { 1 };
    }
    let positions = (0..N).filter(|&k| c[k] != 0).map(|k| k as u8).collect();
    Challenge {
        coeffs: c,
        positions,
    }
}

// ---------------------------------------------------------------------------------------------
// Verification (Algorithms 3 and 8).
// ---------------------------------------------------------------------------------------------

/// Intermediate values of `ML-DSA.Verify_internal` (Algorithm 8).
#[derive(Clone, Debug)]
pub struct VerifyTrace {
    /// Decoded signature.
    pub sig: DecodedSig,
    /// mu.
    pub mu: [u8; TR_BYTES],
    /// c = SampleInBall(c~).
    pub c: Challenge,
    /// w'_Approx = NTT^{-1}(A_hat o NTT(z) - NTT(c) o NTT(t1 2^d)) in [0, q).
    pub w_approx: VecK,
    /// A_hat o NTT(z) in [0, q) (the NTT-domain residue of A z modulo q, which the
    /// integer product of [`crate::crt::az_crt_with`] reuses).
    pub az_hat: [[u64; N]; K],
    /// w1' = UseHint(h, w'_Approx).
    pub w1: SmallK,
    /// w1Encode(w1').
    pub w1_bytes: [u8; W1_ENCODE_BYTES],
    /// c~' = H(mu || w1Encode(w1'), 32).
    pub c_tilde_prime: [u8; C_TILDE_BYTES],
    /// [[ ||z||_inf < gamma1 - beta ]].
    pub z_norm_ok: bool,
    /// Final result of Algorithm 8.
    pub accept: bool,
}

/// ||z||_inf for a vector of polynomials.
pub fn inf_norm<const R: usize>(v: &[Poly; R]) -> i64 {
    v.iter()
        .flat_map(|p| p.iter())
        .map(|x| x.abs())
        .max()
        .unwrap_or(0)
}

/// Algorithm 8 lines 2 to 13 on an expanded key and a precomputed mu.
pub fn verify_internal_trace(
    key: &ExpandedKey,
    mu: &[u8; TR_BYTES],
    sig: &[u8],
) -> Result<VerifyTrace, SigDecodeError> {
    let sig = sig_decode(sig)?;
    let c = sample_in_ball(&sig.c_tilde);
    let fips = NttPrime::fips();
    let z_hat: Vec<[u64; N]> = sig.z.iter().map(|zi| fips.ntt_signed(zi)).collect();
    let mut c_poly = [0i64; N];
    for j in 0..N {
        c_poly[j] = i64::from(c.coeffs[j]);
    }
    let c_hat = fips.ntt_signed(&c_poly);
    let mut w_approx = [[0i64; N]; K];
    let mut az_hat = [[0u64; N]; K];
    let (a_hat, t1_2d_hat) = (key.a_hat(), key.t1_2d_hat());
    for r in 0..K {
        let mut acc = [0u64; N];
        for s in 0..L {
            let prod = fips.pointwise(&a_hat[r][s], &z_hat[s]);
            for j in 0..N {
                acc[j] = fips.add_mod(acc[j], prod[j]);
            }
        }
        az_hat[r] = acc;
        let ct = fips.pointwise(&c_hat, &t1_2d_hat[r]);
        for j in 0..N {
            acc[j] = fips.sub_mod(acc[j], ct[j]);
        }
        let w = fips.intt(&acc);
        for j in 0..N {
            w_approx[r][j] = w[j] as i64;
        }
    }
    let mut w1 = [[0u8; N]; K];
    for r in 0..K {
        for j in 0..N {
            w1[r][j] = use_hint(sig.h[r][j] != 0, w_approx[r][j]) as u8;
        }
    }
    let w1_bytes = w1_encode(&w1);
    let c_tilde_prime = commitment_hash(mu, &w1_bytes);
    let z_norm_ok = inf_norm(&sig.z) < GAMMA1 - BETA;
    let accept = z_norm_ok && c_tilde_prime == sig.c_tilde;
    Ok(VerifyTrace {
        sig,
        mu: *mu,
        c,
        w_approx,
        az_hat,
        w1,
        w1_bytes,
        c_tilde_prime,
        z_norm_ok,
        accept,
    })
}

/// ML-DSA.Verify (Algorithm 3) with context `ctx`, using only this crate's arithmetic.
pub fn verify_with_context(pk: &[u8], msg: &[u8], ctx: &[u8], sig: &[u8]) -> bool {
    if ctx.len() > 255 {
        return false;
    }
    let Some(key) = ExpandedKey::from_bytes(pk) else {
        return false;
    };
    let mu = mu(&key.tr, msg, ctx);
    match verify_internal_trace(&key, &mu, sig) {
        Ok(t) => t.accept,
        Err(_) => false,
    }
}

/// ML-DSA.Verify in pure mode with empty context (the aggregated statement).
pub fn verify(pk: &[u8], msg: &[u8], sig: &[u8]) -> bool {
    verify_with_context(pk, msg, &[], sig)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpack_bits_matches_a_bit_by_bit_reader() {
        let mut rng = crate::gen::SplitMix64::new(0xB175);
        for bits in 1..=24 {
            let mut bytes = vec![0u8; 32 * bits];
            rng.fill(&mut bytes);
            let got = unpack_bits(&bytes, bits);
            for (i, &v) in got.iter().enumerate() {
                let want = (0..bits).fold(0u64, |acc, b| {
                    let pos = i * bits + b;
                    acc | (u64::from((bytes[pos / 8] >> (pos % 8)) & 1) << b)
                });
                assert_eq!(v, want, "bits {bits}, coefficient {i}");
            }
            let mut packed = Vec::new();
            pack_bits(got.iter().copied(), bits, &mut packed);
            assert_eq!(packed, bytes);
        }
    }
    use crate::gen::{self, SplitMix64};

    #[test]
    fn decompose_matches_definition_exhaustively() {
        // r = r1 (2 gamma2) + r0 mod q, r1 in [0, 44), r0 in (-gamma2, gamma2] except the edge pair.
        let mut edge = 0usize;
        for r in 0..Q {
            let (r1, r0) = decompose(r);
            assert!((0..M_W).contains(&r1));
            assert_eq!(mod_q(r1 * 2 * GAMMA2 + r0), r);
            if r >= Q - GAMMA2 {
                // r+ in [q - gamma2, q - 1]: the FIPS edge case, r1 = 0, r0 in [-gamma2, -1].
                assert_eq!(r1, 0);
                assert!((-GAMMA2..=-1).contains(&r0));
                edge += 1;
            } else {
                assert!(-GAMMA2 < r0 && r0 <= GAMMA2);
            }
            if r0 == -GAMMA2 {
                assert_eq!(r, Q - GAMMA2);
            }
        }
        assert_eq!(edge as i64, GAMMA2);
    }

    #[test]
    fn use_hint_examples_and_make_hint() {
        // Edge residue -gamma2: Decompose gives (0, -gamma2); UseHint with h=1 wraps to 43.
        assert_eq!(decompose(Q - GAMMA2), (0, -GAMMA2));
        assert_eq!(use_hint(true, Q - GAMMA2), 43);
        assert_eq!(use_hint(true, Q - 1), 43);
        assert_eq!(use_hint(true, 1), 1);
        assert_eq!(use_hint(true, 0), 43);
        assert_eq!(use_hint(false, Q - 1), 0);
        // MakeHint/UseHint correctness (FIPS 204 Section 2.4) on random small z.
        let mut rng = SplitMix64::new(5);
        for _ in 0..20000 {
            let r = rng.below(Q as u64) as i64;
            let z = rng.below(2 * GAMMA2 as u64 + 1) as i64 - GAMMA2;
            let h = make_hint(z, r);
            assert_eq!(use_hint(h, r), high_bits(r + z));
        }
    }

    #[test]
    fn packing_roundtrips() {
        let mut rng = SplitMix64::new(9);
        let mut z = [[0i64; N]; L];
        for zi in z.iter_mut() {
            for x in zi.iter_mut() {
                *x = rng.below(2 * GAMMA1 as u64) as i64 - GAMMA1 + 1;
            }
        }
        let mut h = [[0u8; N]; K];
        for _ in 0..OMEGA {
            h[rng.below(K as u64) as usize][rng.below(N as u64) as usize] = 1;
        }
        let mut ct = [0u8; 32];
        rng.fill(&mut ct);
        let bytes = sig_encode(&ct, &z, &h);
        assert_eq!(bytes.len(), SIG_BYTES);
        let d = sig_decode(&bytes).unwrap();
        assert_eq!((d.c_tilde, d.z, d.h), (ct, z, h));
        let mut w1 = [[0u8; N]; K];
        for row in w1.iter_mut() {
            for x in row.iter_mut() {
                *x = rng.below(M_W as u64) as u8;
            }
        }
        assert_eq!(w1_decode(&w1_encode(&w1)), w1);
    }

    #[test]
    fn hint_unpack_rejects_malformed() {
        let mut y = [0u8; OMEGA + K];
        assert!(hint_bit_unpack(&y).is_some());
        y[OMEGA] = 2;
        y[0] = 5;
        y[1] = 5; // not strictly increasing
        y[OMEGA + 1] = 2;
        y[OMEGA + 2] = 2;
        y[OMEGA + 3] = 2;
        assert!(hint_bit_unpack(&y).is_none());
        y[1] = 6;
        assert!(hint_bit_unpack(&y).is_some());
        y[OMEGA + 2] = 1; // decreasing index
        assert!(hint_bit_unpack(&y).is_none());
        y[OMEGA + 2] = 2;
        y[10] = 1; // leftover nonzero byte
        assert!(hint_bit_unpack(&y).is_none());
        y[10] = 0;
        y[OMEGA + 3] = (OMEGA + 1) as u8; // beyond omega
        assert!(hint_bit_unpack(&y).is_none());
    }

    #[test]
    fn sample_in_ball_has_weight_tau() {
        let mut rng = SplitMix64::new(13);
        for _ in 0..50 {
            let mut ct = [0u8; 32];
            rng.fill(&mut ct);
            let c = sample_in_ball(&ct);
            assert_eq!(c.positions.len(), TAU);
            assert_eq!(c.coeffs.iter().filter(|&&x| x != 0).count(), TAU);
        }
    }

    #[test]
    fn pk_encode_decode_roundtrip() {
        let r = gen::record(21, 0);
        let pk = pk_decode(&r.pk).unwrap();
        assert_eq!(pk_encode(&pk), r.pk);
        assert!(pk_decode(&r.pk[1..]).is_none());
    }
}
