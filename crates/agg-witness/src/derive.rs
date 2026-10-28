//! Per-signature witness derivation of (z, u, v, b) and the committed quotient.
//!
//! For a signature accepted by Algorithm 8:
//! w' = ctr_q(A z - T) with A centered and T = 2^d (c * t1) over Z, w1' = UseHint(h, w'),
//! u = ctr_q(w' - 2 gamma2 w1') (|u| <= B_u by the UseHint lemma), and the exact quotient
//! v = (A z - T - 2 gamma2 w1' - u) / q over Z (|v| <= B_v). The six w1Encode bits are
//! b_t = bit t of w1'.

use crate::crt::az_integer_with;
use crate::expand::{public_t, ExpandedKey};
use crate::native::*;
use crate::params::*;

/// Why a witness cannot be derived.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeriveError {
    /// The signature does not decode (wrong length or malformed hint).
    Decode(SigDecodeError),
    /// Strict mode: c~ != H(mu || w1Encode(w1')).
    CommitmentMismatch,
    /// Strict mode: ||z||_inf >= gamma1 - beta.
    ZNorm,
    /// Internal invariant broken (a bug): the residual is not divisible by q, or a bound fails.
    Invariant(&'static str),
}

impl From<SigDecodeError> for DeriveError {
    fn from(e: SigDecodeError) -> Self {
        DeriveError::Decode(e)
    }
}

/// The witness of one signature.
#[derive(Clone, Debug)]
pub struct SigWitness {
    /// c~ (public in mode A).
    pub c_tilde: [u8; C_TILDE_BYTES],
    /// mu (public).
    pub mu: [u8; TR_BYTES],
    /// Public T = 2^d (c * t1) over Z.
    pub t: VecK,
    /// Response z.
    pub z: VecL,
    /// Hint h.
    pub h: SmallK,
    /// (A z) over Z.
    pub az: VecK,
    /// w' = ctr_q(A z - T).
    pub w: VecK,
    /// w1' = UseHint(h, w') in [0, 44).
    pub w1: SmallK,
    /// w1Encode(w1').
    pub w1_bytes: [u8; W1_ENCODE_BYTES],
    /// u = ctr_q(w' - 2 gamma2 w1').
    pub u: VecK,
    /// v = (A z - T - 2 gamma2 w1' - u) / q.
    pub v: VecK,
    /// Whether Algorithm 8 accepts (always true in strict mode).
    pub accepted: bool,
}

impl SigWitness {
    /// Bit t of w1'[rho][j] (the committed b_t(j, rho, i)).
    pub fn bit(&self, t: usize, j: usize, rho: usize) -> u8 {
        (self.w1[rho][j] >> t) & 1
    }

    /// The seven-permutation sponge of c~ = SHAKE256(mu || w1Encode(w1'), 32).
    pub fn sponge(&self) -> crate::keccak::SpongeTrace {
        crate::keccak::sponge_trace(&self.mu, &self.w1_bytes)
    }
}

/// Strict derivation: the signature must verify (Algorithm 8) under (key, mu).
pub fn derive(
    key: &ExpandedKey,
    mu: &[u8; TR_BYTES],
    sig: &[u8],
) -> Result<SigWitness, DeriveError> {
    derive_inner(key, mu, sig, true)
}

/// [`derive`] with the public `T = c t1 2^d` already computed for the challenge `c_tilde`
/// (the instance's); it is used when the signature's `c~` is that one, else recomputed.
pub fn derive_with_t(
    key: &ExpandedKey,
    mu: &[u8; TR_BYTES],
    sig: &[u8],
    c_tilde: &[u8; C_TILDE_BYTES],
    t: &VecK,
) -> Result<SigWitness, DeriveError> {
    derive_core(key, mu, sig, true, Some((c_tilde, t)))
}

/// Derivation without the hash and norm checks of Algorithm 8 (only decoding must succeed).
/// Used to show that the checker, not the derivation, rejects a bad signature.
pub fn derive_unchecked(
    key: &ExpandedKey,
    mu: &[u8; TR_BYTES],
    sig: &[u8],
) -> Result<SigWitness, DeriveError> {
    derive_inner(key, mu, sig, false)
}

fn derive_inner(
    key: &ExpandedKey,
    mu: &[u8; TR_BYTES],
    sig: &[u8],
    strict: bool,
) -> Result<SigWitness, DeriveError> {
    derive_core(key, mu, sig, strict, None)
}

fn derive_core(
    key: &ExpandedKey,
    mu: &[u8; TR_BYTES],
    sig: &[u8],
    strict: bool,
    known_t: Option<(&[u8; C_TILDE_BYTES], &VecK)>,
) -> Result<SigWitness, DeriveError> {
    let trace = verify_internal_trace(key, mu, sig)?;
    if strict {
        if !trace.z_norm_ok {
            return Err(DeriveError::ZNorm);
        }
        if trace.c_tilde_prime != trace.sig.c_tilde {
            return Err(DeriveError::CommitmentMismatch);
        }
    }
    let z = trace.sig.z;
    let t = match known_t {
        Some((c_tilde, t)) if *c_tilde == trace.sig.c_tilde => *t,
        _ => public_t(&trace.c, &key.t1),
    };
    // The residue of A z modulo q is the trace's, in the NTT domain.
    let az_wide = az_integer_with(key, &z, None);
    let mut az = [[0i64; N]; K];
    let mut w = [[0i64; N]; K];
    let mut u = [[0i64; N]; K];
    let mut v = [[0i64; N]; K];
    for r in 0..K {
        for j in 0..N {
            let a = i64::try_from(az_wide[r][j])
                .map_err(|_| DeriveError::Invariant("A z exceeds i64"))?;
            az[r][j] = a;
            let wp = ctr_q(a - t[r][j]);
            if mod_q(wp) != trace.w_approx[r][j] {
                return Err(DeriveError::Invariant(
                    "integer A z - T disagrees with the NTT path",
                ));
            }
            w[r][j] = wp;
            let w1 = i64::from(trace.w1[r][j]);
            let ur = ctr_q(wp - 2 * GAMMA2 * w1);
            u[r][j] = ur;
            let delta0 = a - t[r][j] - 2 * GAMMA2 * w1 - ur;
            if delta0.rem_euclid(Q) != 0 {
                return Err(DeriveError::Invariant("residual not divisible by q"));
            }
            v[r][j] = delta0 / Q;
        }
    }
    if strict {
        if inf_norm(&z) > B_Z {
            return Err(DeriveError::Invariant("|z| > B_z"));
        }
        if inf_norm(&u) > B_U {
            return Err(DeriveError::Invariant("|u| > B_u (UseHint lemma)"));
        }
        if inf_norm(&v) > B_V {
            return Err(DeriveError::Invariant("|v| > B_v (3.4')"));
        }
    }
    Ok(SigWitness {
        c_tilde: trace.sig.c_tilde,
        mu: *mu,
        t,
        z,
        h: trace.sig.h,
        az,
        w,
        w1: trace.w1,
        w1_bytes: trace.w1_bytes,
        u,
        v,
        accepted: trace.accept,
    })
}

/// Why witness values cannot be re-encoded to a FIPS 204 signature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReencodeError {
    /// z outside [-gamma1 + 1, gamma1] (not packable).
    ZRange,
    /// w1' is not reachable from w' = u + 2 gamma2 w1' by any hint bit (relation H is relaxed).
    NotReachable,
    /// The hint weight exceeds omega.
    HintWeight,
    /// A value is outside its domain (w1' not in [0, 44)).
    Domain,
}

/// Re-encodes witness values (z, u, w1') to signature bytes (c~, z, h) with
/// h = [HighBits(w') != w1'] and w' = u + 2 gamma2 w1' mod q.
pub fn reencode_h(
    c_tilde: &[u8; C_TILDE_BYTES],
    z: &VecL,
    u: &VecK,
    w1: &SmallK,
) -> Result<Vec<u8>, ReencodeError> {
    if z.iter()
        .flat_map(|p| p.iter())
        .any(|&x| x <= -GAMMA1 || x > GAMMA1)
    {
        return Err(ReencodeError::ZRange);
    }
    let mut h = [[0u8; N]; K];
    let mut weight = 0usize;
    for r in 0..K {
        for j in 0..N {
            let w1v = i64::from(w1[r][j]);
            if w1v >= M_W {
                return Err(ReencodeError::Domain);
            }
            let wp = u[r][j] + 2 * GAMMA2 * w1v;
            let hb = high_bits(wp) != w1v;
            if use_hint(hb, wp) != w1v {
                return Err(ReencodeError::NotReachable);
            }
            h[r][j] = u8::from(hb);
            weight += usize::from(hb);
        }
    }
    if weight > OMEGA {
        return Err(ReencodeError::HintWeight);
    }
    Ok(sig_encode(c_tilde, z, &h))
}

/// The UseHint residual lemma, exhaustively: max |ctr_q(r - 2 gamma2 UseHint(h, r))| over
/// r in Z_q and h in {0, 1} (expected exactly 190,465).
pub fn usehint_residual_max() -> i64 {
    let mut m = 0i64;
    for r in 0..Q {
        let (r1, r0) = decompose(r);
        for h in [false, true] {
            let w1 = if !h {
                r1
            } else if r0 > 0 {
                (r1 + 1).rem_euclid(M_W)
            } else {
                (r1 - 1).rem_euclid(M_W)
            };
            m = m.max(ctr_q(r - 2 * GAMMA2 * w1).abs());
        }
    }
    m
}
