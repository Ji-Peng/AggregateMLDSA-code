//! Per-signature witness derivation of (z, u, v, b) and the committed quotient.
//!
//! For a signature accepted by Algorithm 8:
//! w' = ctr_q(A z - T) with A centered and T = 2^d (c * t1) over Z, w1' = UseHint(h, w'),
//! u = ctr_q(w' - 2 gamma2 w1') (|u| <= B_u by the UseHint lemma), and the exact quotient
//! v = (A z - T - 2 gamma2 w1' - u) / q over Z (|v| <= B_v). The six w1Encode bits are
//! b_t = bit t of w1'.

use crate::expand::{public_t_for, ExpandedKey};
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

/// The witness of one signature for parameter set P.
#[derive(Clone, Debug)]
pub struct SigWitness<P: ParamSet = Set44> {
    /// c~ (public in mode A).
    pub c_tilde: P::CTilde,
    /// mu (public).
    pub mu: [u8; TR_BYTES],
    /// Public T = 2^d (c * t1) over Z.
    pub t: P::VecK,
    /// Response z.
    pub z: P::VecL,
    /// Hint h.
    pub h: P::SmallK,
    /// (A z) over Z.
    pub az: P::VecK,
    /// w' = ctr_q(A z - T).
    pub w: P::VecK,
    /// w1' = UseHint(h, w') in [0, m_w).
    pub w1: P::SmallK,
    /// w1Encode(w1').
    pub w1_bytes: P::W1Encode,
    /// u = ctr_q(w' - 2 gamma2 w1').
    pub u: P::VecK,
    /// v = (A z - T - 2 gamma2 w1' - u) / q.
    pub v: P::VecK,
    /// Whether Algorithm 8 accepts (always true in strict mode).
    pub accepted: bool,
}

impl<P: ParamSet> SigWitness<P> {
    /// Bit t of w1'[rho][j] (the committed b_t(j, rho, i)).
    pub fn bit(&self, t: usize, j: usize, rho: usize) -> u8 {
        (self.w1[rho][j] >> t) & 1
    }

    /// The sponge of c~ = SHAKE256(mu || w1Encode(w1'), lambda/4).
    pub fn sponge(&self) -> crate::keccak::SpongeTrace<P> {
        crate::keccak::sponge_trace_for::<P>(&self.mu, self.w1_bytes.as_ref())
    }
}

/// Strict derivation: the signature must verify (Algorithm 8) under (key, mu) for parameter set P.
pub fn derive_for<P: ParamSet>(
    key: &ExpandedKey<P>,
    mu: &[u8; TR_BYTES],
    sig: &[u8],
) -> Result<SigWitness<P>, DeriveError> {
    derive_inner_for::<P>(key, mu, sig, true)
}

/// Strict derivation: the signature must verify (Algorithm 8) under (key, mu).
pub fn derive(
    key: &ExpandedKey,
    mu: &[u8; TR_BYTES],
    sig: &[u8],
) -> Result<SigWitness<Set44>, DeriveError> {
    derive_for::<Set44>(key, mu, sig)
}

/// [`derive_for`] with the public `T = c t1 2^d` already computed for the challenge `c_tilde`.
pub fn derive_with_t_for<P: ParamSet>(
    key: &ExpandedKey<P>,
    mu: &[u8; TR_BYTES],
    sig: &[u8],
    c_tilde: &P::CTilde,
    t: &[[i32; N]],
) -> Result<SigWitness<P>, DeriveError> {
    derive_core_for::<P>(key, mu, sig, true, Some((c_tilde, t)))
}

/// [`derive`] with the public `T = c t1 2^d` already computed for the challenge `c_tilde`
/// (the instance's); it is used when the signature's `c~` is that one, else recomputed.
pub fn derive_with_t(
    key: &ExpandedKey,
    mu: &[u8; TR_BYTES],
    sig: &[u8],
    c_tilde: &[u8; C_TILDE_BYTES],
    t: &[[i32; N]; K],
) -> Result<SigWitness<Set44>, DeriveError> {
    derive_with_t_for::<Set44>(key, mu, sig, c_tilde, t)
}

/// Derivation without the hash and norm checks of Algorithm 8 for parameter set P.
pub fn derive_unchecked_for<P: ParamSet>(
    key: &ExpandedKey<P>,
    mu: &[u8; TR_BYTES],
    sig: &[u8],
) -> Result<SigWitness<P>, DeriveError> {
    derive_inner_for::<P>(key, mu, sig, false)
}

/// Derivation without the hash and norm checks of Algorithm 8 (only decoding must succeed).
/// Used to show that the checker, not the derivation, rejects a bad signature.
pub fn derive_unchecked(
    key: &ExpandedKey,
    mu: &[u8; TR_BYTES],
    sig: &[u8],
) -> Result<SigWitness<Set44>, DeriveError> {
    derive_unchecked_for::<Set44>(key, mu, sig)
}

fn derive_inner_for<P: ParamSet>(
    key: &ExpandedKey<P>,
    mu: &[u8; TR_BYTES],
    sig: &[u8],
    strict: bool,
) -> Result<SigWitness<P>, DeriveError> {
    derive_core_for::<P>(key, mu, sig, strict, None)
}

fn derive_core_for<P: ParamSet>(
    key: &ExpandedKey<P>,
    mu: &[u8; TR_BYTES],
    sig: &[u8],
    strict: bool,
    known_t: Option<(&P::CTilde, &[[i32; N]])>,
) -> Result<SigWitness<P>, DeriveError> {
    let trace = verify_internal_trace_for::<P>(key, mu, sig)?;
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
        Some((c_tilde, t_arr)) if *c_tilde == trace.sig.c_tilde => {
            let mut out = P::zero_vec_k();
            for r in 0..P::K {
                for j in 0..N {
                    out[r][j] = i64::from(t_arr[r][j]);
                }
            }
            out
        }
        _ => public_t_for::<P>(&trace.c, &key.t1),
    };
    let az_wide = crate::crt::az_integer_with_for::<P>(key, &z, Some(&trace.az_hat));
    let mut az = P::zero_vec_k();
    let mut w = P::zero_vec_k();
    let mut u = P::zero_vec_k();
    let mut v = P::zero_vec_k();
    for r in 0..P::K {
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
            let ur = ctr_q(wp - 2 * P::GAMMA2 * w1);
            u[r][j] = ur;
            let delta0 = a - t[r][j] - 2 * P::GAMMA2 * w1 - ur;
            if delta0.rem_euclid(Q) != 0 {
                return Err(DeriveError::Invariant("residual not divisible by q"));
            }
            v[r][j] = delta0 / Q;
        }
    }
    if strict {
        if inf_norm_slice(z.as_ref()) > P::B_Z {
            return Err(DeriveError::Invariant("|z| > B_z"));
        }
        if inf_norm_slice(u.as_ref()) > P::B_U {
            return Err(DeriveError::Invariant("|u| > B_u (UseHint lemma)"));
        }
        if inf_norm_slice(v.as_ref()) > P::B_V {
            return Err(DeriveError::Invariant("|v| > B_v (honest quotient bound)"));
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
    /// A value is outside its domain (w1' not in [0, m_w)).
    Domain,
}

/// Re-encodes witness values (z, u, w1') to signature bytes (c~, z, h) for parameter set P.
pub fn reencode_h_for<P: ParamSet>(
    c_tilde: &P::CTilde,
    z: &P::VecL,
    u: &P::VecK,
    w1: &P::SmallK,
) -> Result<Vec<u8>, ReencodeError> {
    if (0..P::L).any(|s| z[s].iter().any(|&x| x <= -P::GAMMA1 || x > P::GAMMA1)) {
        return Err(ReencodeError::ZRange);
    }
    let mut h = P::zero_small_k();
    let mut weight = 0usize;
    for r in 0..P::K {
        for j in 0..N {
            let w1v = i64::from(w1[r][j]);
            if w1v >= P::M_W {
                return Err(ReencodeError::Domain);
            }
            let wp = u[r][j] + 2 * P::GAMMA2 * w1v;
            let hb = high_bits_for::<P>(wp) != w1v;
            if use_hint_for::<P>(hb, wp) != w1v {
                return Err(ReencodeError::NotReachable);
            }
            h[r][j] = u8::from(hb);
            weight += usize::from(hb);
        }
    }
    if weight > P::OMEGA {
        return Err(ReencodeError::HintWeight);
    }
    Ok(sig_encode_for::<P>(c_tilde, z, &h))
}

/// Re-encodes witness values (z, u, w1') to signature bytes (c~, z, h) with
/// h = [HighBits(w') != w1'] and w' = u + 2 gamma2 w1' mod q.
pub fn reencode_h(
    c_tilde: &[u8; C_TILDE_BYTES],
    z: &VecL,
    u: &VecK,
    w1: &SmallK,
) -> Result<Vec<u8>, ReencodeError> {
    reencode_h_for::<Set44>(c_tilde, z, u, w1)
}

/// The UseHint residual lemma, exhaustively for parameter set P.
pub fn usehint_residual_max_for<P: ParamSet>() -> i64 {
    let mut m = 0i64;
    for r in 0..Q {
        let (r1, r0) = decompose_for::<P>(r);
        for h in [false, true] {
            let w1 = if !h {
                r1
            } else if r0 > 0 {
                (r1 + 1).rem_euclid(P::M_W)
            } else {
                (r1 - 1).rem_euclid(P::M_W)
            };
            m = m.max(ctr_q(r - 2 * P::GAMMA2 * w1).abs());
        }
    }
    m
}

/// The UseHint residual lemma, exhaustively: max |ctr_q(r - 2 gamma2 UseHint(h, r))| over
/// r in Z_q and h in {0, 1} (expected exactly 190,465 for Set44).
pub fn usehint_residual_max() -> i64 {
    usehint_residual_max_for::<Set44>()
}
