//! ML-DSA-44 parameters (FIPS 204 Table 1 and 2) and the spec's derived constants (spec Section 1).
//!
//! Every FIPS value is copied from `ref/FIPS-204-ML-DSA.pdf` Table 1 (ML-DSA-44 column);
//! every derived constant is recomputed from those values in the unit tests below.

/// Modulus q (FIPS 204 Table 1).
pub const Q: i64 = 8_380_417;
/// Ring degree n.
pub const N: usize = 256;
/// Rows of A (k).
pub const K: usize = 4;
/// Columns of A (l).
pub const L: usize = 4;
/// Dropped bits of t (d).
pub const D: u32 = 13;
/// Number of +-1 entries of c (tau).
pub const TAU: usize = 39;
/// Collision strength of c~ (lambda).
pub const LAMBDA: usize = 128;
/// Coefficient range of y (gamma1 = 2^17).
pub const GAMMA1: i64 = 1 << 17;
/// Low-order rounding range (gamma2 = (q-1)/88).
pub const GAMMA2: i64 = (Q - 1) / 88;
/// Secret key range (eta).
pub const ETA: i64 = 2;
/// beta = tau * eta.
pub const BETA: i64 = TAU as i64 * ETA;
/// Maximum number of ones in the hint (omega).
pub const OMEGA: usize = 80;
/// 512th root of unity modulo q used by the NTT (FIPS 204 Section 7.5).
pub const ZETA: i64 = 1753;
/// 256^{-1} mod q (FIPS 204 Algorithm 42, line 21).
pub const N_INV_Q: i64 = 8_347_681;

/// Bytes of c~ (lambda / 4).
pub const C_TILDE_BYTES: usize = LAMBDA / 4;
/// Bits per t1 coefficient: bitlen(q-1) - d.
pub const T1_BITS: usize = 10;
/// Bits per packed z coefficient: 1 + bitlen(gamma1 - 1).
pub const Z_BITS: usize = 18;
/// Bits per w1 coefficient: bitlen((q-1)/(2 gamma2) - 1).
pub const W1_BITS: usize = 6;
/// Public key bytes (FIPS 204 Table 2): 32 + 32 k (bitlen(q-1) - d).
pub const PK_BYTES: usize = 32 + 32 * K * T1_BITS;
/// Signature bytes (FIPS 204 Table 2): lambda/4 + l 32 (1 + bitlen(gamma1-1)) + omega + k.
pub const SIG_BYTES: usize = C_TILDE_BYTES + L * 32 * Z_BITS + OMEGA + K;
/// Bytes of tr and mu.
pub const TR_BYTES: usize = 64;
/// Bytes of w1Encode(w1).
pub const W1_ENCODE_BYTES: usize = 32 * K * W1_BITS;

// ---------------------------------------------------------------------------------------------
// Derived constants (spec Section 1).
// ---------------------------------------------------------------------------------------------

/// m_w = (q-1)/(2 gamma2): number of w1 values.
pub const M_W: i64 = (Q - 1) / (2 * GAMMA2);
/// B_z = gamma1 - beta - 1: honest bound on |z|_inf.
pub const B_Z: i64 = GAMMA1 - BETA - 1;
/// B_u = 2 gamma2 + 1: UseHint residual bound.
pub const B_U: i64 = 2 * GAMMA2 + 1;
/// B_T = 2^d tau (2^10 - 1): bound on |2^d c t1| over Z.
pub const B_T: i64 = (1 << D) * TAU as i64 * ((1 << T1_BITS) - 1);
/// Worst-case |Delta_0| of (3.4'): l 256 (q-1)/2 B_z + B_T + 2 gamma2 (m_w - 1) + B_u.
pub const DELTA0_MAX: i64 =
    (L * N) as i64 * ((Q - 1) / 2) * B_Z + B_T + 2 * GAMMA2 * (M_W - 1) + B_U;
/// B_v = floor(DELTA0_MAX / q): honest quotient bound.
pub const B_V: i64 = DELTA0_MAX / Q;
/// SelfTargetMSIS bound of FIPS 204: max(gamma1 - beta, 2 gamma2 + 1 + 2^(d-1) tau).
pub const ZETA_MSIS: i64 = 350_209;
/// w1Encode bits per signature: k 256 6.
pub const W1LEN: usize = K * N * W1_BITS;

/// Proven bound on |z| in relation H (Section 3.1): bar B_z = 2^17.
pub const BAR_B_Z: i64 = 1 << 17;
/// Proven bound on |u| in relation H (Section 3.1): bar B_u = 2^18.
pub const BAR_B_U: i64 = 1 << 18;
/// Proven range of the quotient v: [-2^26, 2^26 - 1] (Section 4.3).
pub const V_HALF_RANGE: i64 = 1 << 26;
/// zeta' of (3.5) for the relaxed bounds.
pub const ZETA_PRIME: i64 = if BAR_B_Z > BAR_B_U + (1 << (D - 1)) * TAU as i64 {
    BAR_B_Z
} else {
    BAR_B_U + (1 << (D - 1)) * TAU as i64
};

/// PIOP field modulus p = 2^64 - 59 (Akita `Prime64Offset59`).
pub const P_FIELD: u64 = 0xFFFF_FFFF_FFFF_FFC5;

// ---------------------------------------------------------------------------------------------
// Batch geometry constants (spec Section 1).
// ---------------------------------------------------------------------------------------------

/// Maximum number of live records in wire version 0.
pub const N_MAX: usize = 1 << 15;
/// Minimum padded batch size N*.
pub const N_STAR_MIN: usize = 8;
/// Coefficient variables.
pub const D_COEF: u32 = 8;
/// Vector-index variables (k = l = 4).
pub const D_VEC: u32 = 2;
/// Position variables d_pos = d_coef + 2.
pub const D_POS: u32 = D_COEF + D_VEC;
/// Positions per signature (k 256 = l 256 = 1024).
pub const POSITIONS: usize = 1 << D_POS;
/// Segment size exponent: a segment is 2^13 signatures.
pub const D_SEG_MAX: u32 = 13;
/// Akita minimum physical table size exponent (`PCS_MIN_CELLS_LOG2`).
pub const PCS_MIN_CELLS_LOG2: u32 = 13;

// Compile-time checks of spec Section 1 bounds.
const _: () = assert!(B_T < 1 << 29);
const _: () = assert!(DELTA0_MAX < 1 << 49);
const _: () = assert!(B_V < V_HALF_RANGE);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fips_values_and_sizes() {
        assert_eq!(GAMMA2, 95_232);
        assert_eq!(BETA, 78);
        assert_eq!(PK_BYTES, 1312);
        assert_eq!(SIG_BYTES, 2420);
        assert_eq!(W1_ENCODE_BYTES, 768);
        // bitlen(q - 1) = 23.
        assert_eq!(
            64 - ((Q - 1) as u64).leading_zeros() as usize - D as usize,
            T1_BITS
        );
        assert_eq!(
            1 + (64 - ((GAMMA1 - 1) as u64).leading_zeros() as usize),
            Z_BITS
        );
        assert_eq!(64 - ((M_W - 1) as u64).leading_zeros() as usize, W1_BITS);
        // 256 * 256^{-1} = 1 mod q.
        assert_eq!((256 * N_INV_Q) % Q, 1);
        // zeta is a primitive 512th root of unity: zeta^256 = -1.
        let mut x = 1i64;
        for _ in 0..256 {
            x = x * ZETA % Q;
        }
        assert_eq!(x, Q - 1);
    }

    #[test]
    fn derived_constants_match_spec_section_1() {
        assert_eq!(M_W, 44);
        assert_eq!(B_Z, 130_993);
        assert_eq!(B_U, 190_465);
        assert_eq!(B_T, 326_836_224);
        assert_eq!(DELTA0_MAX, 562_061_561_757_697);
        assert_eq!(B_V, 67_068_447);
        assert_eq!(W1LEN, 6144);
        let zeta = std::cmp::max(GAMMA1 - BETA, 2 * GAMMA2 + 1 + (1 << (D - 1)) * TAU as i64);
        assert_eq!(zeta, ZETA_MSIS);
        assert_eq!(ZETA_PRIME, 421_888);
        assert_eq!(P_FIELD, u64::MAX - 58);
    }

    #[test]
    fn lifting_bound_6_1() {
        // |Delta^Z| < 2^51 < p/2 under the proven ranges (Section 6).
        let az = (L * N) as i128 * ((Q - 1) / 2) as i128 * BAR_B_Z as i128;
        let d = az
            + B_T as i128
            + (2 * GAMMA2 * (M_W - 1)) as i128
            + BAR_B_U as i128
            + Q as i128 * V_HALF_RANGE as i128;
        assert!(d < 1i128 << 51);
        assert!(d < (P_FIELD / 2) as i128);
    }
}
