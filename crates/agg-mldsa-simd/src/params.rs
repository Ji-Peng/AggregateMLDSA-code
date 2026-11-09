//! ML-DSA parameters (top-level constants: ML-DSA-44; [`ParamSet`]: all three
//! FIPS 204 sets) and the constants of the transform primes q, P2 and P3.

/// Ring dimension n.
pub const N: usize = 256;
/// Matrix rows of A (k).
pub const K: usize = 4;
/// Matrix columns of A (l).
pub const L: usize = 4;
/// Dropped bits of t (d).
pub const D: u32 = 13;
/// Number of +-1 entries in challenge c (tau).
pub const TAU: usize = 39;
/// Collision strength lambda.
pub const LAMBDA: usize = 128;
/// Coefficient range gamma1 = 2^17.
pub const GAMMA1: i32 = 1 << 17;
/// Low-order rounding range gamma2 = (q - 1) / 88.
pub const GAMMA2: i32 = (Q - 1) / 88;
/// alpha = 2 * gamma2.
pub const ALPHA: i32 = 2 * GAMMA2;
/// Secret key range eta.
pub const ETA: i32 = 2;
/// beta = tau * eta.
pub const BETA: i32 = TAU as i32 * ETA;
/// Maximum number of ones in the hint (omega).
pub const OMEGA: usize = 80;

/// Bound on UseHint residual u: 2 * gamma2 + 1.
pub const B_U: i32 = 2 * GAMMA2 + 1;
/// Bound on |2^d * c * t1| over Z.
pub const B_T: i32 = (1 << D) * TAU as i32 * ((1 << 10) - 1);
/// Honest quotient bound B_v.
pub const B_V: i32 = 67_068_447;

/// Primary ML-DSA modulus q = 8,380,417 = 2^23 - 2^13 + 1.
pub const Q: i32 = 8_380_417;
/// Secondary CRT prime P2 = 167,772,161 = 5 * 2^25 + 1.
pub const P2: i32 = 167_772_161;

/// Centered CRT constant: q^-1 * 2^32 mod P2 = 158,916,606 = -8,855,555.
pub const CRT_C: i32 = -8_855_555;
/// Tertiary CRT prime P3 = 377,487,361 = 45 * 2^23 + 1 (29 bits, 2-adicity 23).
pub const P3: i32 = 377_487_361;
/// Centered CRT constant: q^-1 * 2^32 mod P3 = 35,430,533.
pub const CRT_C_P3: i32 = 35_430_533;
/// P2^-1 mod P3 = 75,497,474.
pub const P2_INV_MOD_P3: i32 = 75_497_474;
/// Multiplicative inverse of q mod 2^32.
pub const QINV32: i32 = 58_728_449;

/// Identifies the modulus prime (q, secondary CRT prime P2, or tertiary CRT prime P3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Prime {
    /// ML-DSA modulus q = 8,380,417.
    Q,
    /// CRT prime P2 = 167,772,161.
    P2,
    /// CRT prime P3 = 377,487,361.
    P3,
}

/// Precomputed Montgomery and NTT constants for a prime modulus.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrimeConsts {
    /// Modulus p.
    pub p: i32,
    /// p^-1 mod 2^32 (as signed i32).
    pub pinv: i32,
    /// R mod p centered, where R = 2^32.
    pub mont: i32,
    /// Primitive 512th root of unity modulo p.
    pub zeta: i32,
    /// Plain inverse scale factor: R / 256 mod p.
    pub inv_plain: i32,
    /// Montgomery inverse scale factor: R^2 / 256 mod p.
    pub inv_mont: i32,
}

/// Constants for prime q = 8,380,417.
pub const CONSTS_Q: PrimeConsts = PrimeConsts {
    p: 8_380_417,
    pinv: 58_728_449,
    mont: -4_186_625,
    zeta: 1_753,
    inv_plain: 16_382,
    inv_mont: 41_978,
};

/// Constants for prime P2 = 167,772,161.
pub const CONSTS_P2: PrimeConsts = PrimeConsts {
    p: 167_772_161,
    pinv: -167_772_159,
    mont: -67_108_890,
    zeta: 57_844_624,
    inv_plain: 16_777_216,
    inv_mont: 6_710_889,
};

/// Constants for prime P3 = 377,487,361.
pub const CONSTS_P3: PrimeConsts = PrimeConsts {
    p: 377_487_361,
    pinv: -377_487_359,
    mont: 142_606_325,
    zeta: 69_218_873,
    inv_plain: 16_777_216,
    inv_mont: -48_281_099,
};

/// Returns the precomputed constants for the given prime modulus.
#[inline(always)]
pub const fn consts(prime: Prime) -> PrimeConsts {
    match prime {
        Prime::Q => CONSTS_Q,
        Prime::P2 => CONSTS_P2,
        Prime::P3 => CONSTS_P3,
    }
}

impl Prime {
    /// Returns the precomputed constants for this prime modulus.
    #[inline(always)]
    pub const fn consts(self) -> PrimeConsts {
        consts(self)
    }
}

/// Compile-time parameter set specification for ML-DSA aggregation SIMD kernels.
pub trait ParamSet: Copy + Send + Sync + 'static {
    /// Identifier: 0x44, 0x65, or 0x87.
    const ID: u8;
    /// Matrix rows k.
    const K: usize;
    /// Matrix columns l.
    const L: usize;
    /// Challenge tau.
    const TAU: usize;
    /// Secret key range eta.
    const ETA: i32;
    /// Commitment range gamma1.
    const GAMMA1: i32;
    /// Low-order rounding range gamma2.
    const GAMMA2: i32;
    /// Maximum ones in hint (omega).
    const OMEGA: usize;
    /// Collision strength lambda in bits.
    const LAMBDA: usize;
    /// Bits per packed z coefficient: 1 + bitlen(gamma1 - 1).
    const Z_BITS: usize;
    /// Bits per w1 coefficient.
    const W1_BITS: usize;
    /// Number of w1 values m_w = (q - 1) / (2 * gamma2).
    const M_W: i32;
    /// Residual bound: 2 * gamma2 + 1.
    const B_U: i32;
    /// Honest quotient bound B_v.
    const B_V: i32;
    /// Bound on |2^d * c * t1| over Z.
    const B_T: i32;
    /// Bound on |ve| = |e / q|.
    const C_B_Q: u32;
    // The four digit-layout constants below are not read by this crate's kernels.
    // They mirror the witness crate's parameter trait, which derives its
    // implementation of this trait from its own constants and asserts at compile
    // time that both tables agree.
    /// Base of the top digit of u (6 for 44, 8 for 65/87).
    const U_TOP_BASE: u32;
    /// Step of the top digit of u for step layout (63,079 for 44; 0 for 65/87).
    const STEP_U: i32;
    /// Slack for u in digit-slack form (523,267 for 65/87; 0 for 44).
    const U_SLACK: i32;
    /// Reconstruction offset for u in relation H-exact (-190,465 for 44, -228,865 for 65/87).
    const U_OFFSET: i32;
    /// Bytes of commitment hash c~ (lambda / 4).
    const C_TILDE_BYTES: usize;
    /// Public key byte length.
    const PK_BYTES: usize;
    /// Signature byte length.
    const SIG_BYTES: usize;
    /// w1Encode bytes.
    const W1_ENCODE_BYTES: usize;
}

/// Zero-sized type selecting ML-DSA-44 parameters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Set44;

impl ParamSet for Set44 {
    const ID: u8 = 0x44;
    const K: usize = 4;
    const L: usize = 4;
    const TAU: usize = 39;
    const ETA: i32 = 2;
    const GAMMA1: i32 = 1 << 17;
    const GAMMA2: i32 = (Q - 1) / 88;
    const OMEGA: usize = 80;
    const LAMBDA: usize = 128;
    const Z_BITS: usize = 18;
    const W1_BITS: usize = 6;
    const M_W: i32 = 44;
    const B_U: i32 = 190_465;
    const B_V: i32 = 67_068_447;
    const B_T: i32 = (1 << 13) * 39 * ((1 << 10) - 1);
    const C_B_Q: u32 = 41;
    const U_TOP_BASE: u32 = 6;
    const STEP_U: i32 = 63_079;
    const U_SLACK: i32 = 0;
    const U_OFFSET: i32 = -190_465;
    const C_TILDE_BYTES: usize = 32;
    const PK_BYTES: usize = 1312;
    const SIG_BYTES: usize = 2420;
    const W1_ENCODE_BYTES: usize = 768;
}

/// Zero-sized type selecting ML-DSA-65 parameters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Set65;

impl ParamSet for Set65 {
    const ID: u8 = 0x65;
    const K: usize = 6;
    const L: usize = 5;
    const TAU: usize = 49;
    const ETA: i32 = 4;
    const GAMMA1: i32 = 1 << 19;
    const GAMMA2: i32 = (Q - 1) / 32;
    const OMEGA: usize = 55;
    const LAMBDA: usize = 192;
    const Z_BITS: usize = 20;
    const W1_BITS: usize = 4;
    const M_W: i32 = 16;
    const B_U: i32 = 523_777;
    const B_V: i32 = 335_418_249;
    const B_T: i32 = (1 << 13) * 49 * ((1 << 10) - 1);
    const C_B_Q: u32 = 51;
    const U_TOP_BASE: u32 = 8;
    const STEP_U: i32 = 0;
    const U_SLACK: i32 = 523_267;
    const U_OFFSET: i32 = -228_865;
    const C_TILDE_BYTES: usize = 48;
    const PK_BYTES: usize = 1952;
    const SIG_BYTES: usize = 3309;
    const W1_ENCODE_BYTES: usize = 768;
}

/// Zero-sized type selecting ML-DSA-87 parameters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Set87;

impl ParamSet for Set87 {
    const ID: u8 = 0x87;
    const K: usize = 8;
    const L: usize = 7;
    const TAU: usize = 60;
    const ETA: i32 = 2;
    const GAMMA1: i32 = 1 << 19;
    const GAMMA2: i32 = (Q - 1) / 32;
    const OMEGA: usize = 75;
    const LAMBDA: usize = 256;
    const Z_BITS: usize = 20;
    const W1_BITS: usize = 4;
    const M_W: i32 = 16;
    const B_U: i32 = 523_777;
    const B_V: i32 = 469_653_636;
    const B_T: i32 = (1 << 13) * 60 * ((1 << 10) - 1);
    const C_B_Q: u32 = 62;
    const U_TOP_BASE: u32 = 8;
    const STEP_U: i32 = 0;
    const U_SLACK: i32 = 523_267;
    const U_OFFSET: i32 = -228_865;
    const C_TILDE_BYTES: usize = 64;
    const PK_BYTES: usize = 2592;
    const SIG_BYTES: usize = 4627;
    const W1_ENCODE_BYTES: usize = 1024;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pow_mod(mut base: u128, mut exp: u128, modulus: u128) -> u128 {
        let mut res = 1u128;
        base %= modulus;
        while exp > 0 {
            if exp & 1 == 1 {
                res = (res * base) % modulus;
            }
            base = (base * base) % modulus;
            exp >>= 1;
        }
        res
    }

    #[test]
    fn test_constants_derivation() {
        // ML-DSA parameters check
        assert_eq!(GAMMA2, 95_232);
        assert_eq!(ALPHA, 190_464);
        assert_eq!(B_U, 190_465);
        assert_eq!(B_T, 326_836_224);

        // Prime Q checks
        let cq = consts(Prime::Q);
        assert_eq!(cq.p, 8_380_417);
        // zeta^256 = -1 mod q
        assert_eq!(
            pow_mod(cq.zeta as u128, 256, cq.p as u128),
            cq.p as u128 - 1
        );
        // pinv * p = 1 mod 2^32
        assert_eq!((cq.pinv as u32).wrapping_mul(cq.p as u32), 1);
        // mont = 2^32 mod q centered
        let r_mod_q = ((1u128 << 32) % cq.p as u128) as i64;
        let mont_q = if r_mod_q > (cq.p as i64 / 2) {
            r_mod_q - cq.p as i64
        } else {
            r_mod_q
        };
        assert_eq!(cq.mont, mont_q as i32);
        // inv_plain = (2^32 / 256) mod q = 2^24 mod q
        assert_eq!(
            cq.inv_plain as u128,
            pow_mod(2, 24, cq.p as u128)
        );
        // inv_mont = (2^32)^2 / 256 mod q = 2^56 mod q
        assert_eq!(
            cq.inv_mont as u128,
            pow_mod(2, 56, cq.p as u128)
        );

        // Prime P2 checks
        let cp2 = consts(Prime::P2);
        assert_eq!(cp2.p, 167_772_161);
        // zeta^256 = -1 mod P2
        assert_eq!(
            pow_mod(cp2.zeta as u128, 256, cp2.p as u128),
            cp2.p as u128 - 1
        );
        // pinv * p = 1 mod 2^32
        assert_eq!((cp2.pinv as u32).wrapping_mul(cp2.p as u32), 1);
        // mont = 2^32 mod P2 centered
        let r_mod_p2 = ((1u128 << 32) % cp2.p as u128) as i64;
        let mont_p2 = if r_mod_p2 > (cp2.p as i64 / 2) {
            r_mod_p2 - cp2.p as i64
        } else {
            r_mod_p2
        };
        assert_eq!(cp2.mont, mont_p2 as i32);
        // inv_plain = 2^24 mod P2
        assert_eq!(
            cp2.inv_plain as u128,
            pow_mod(2, 24, cp2.p as u128)
        );
        // inv_mont = 2^56 mod P2
        assert_eq!(
            cp2.inv_mont as u128,
            pow_mod(2, 56, cp2.p as u128)
        );

        // CRT_C = q^-1 * 2^32 mod P2 (centered)
        let q_inv_p2 = pow_mod(Q as u128, cp2.p as u128 - 2, cp2.p as u128);
        let crt_c_uncentered = (q_inv_p2 * ((1u128 << 32) % cp2.p as u128)) % cp2.p as u128;
        let crt_c_centered = if crt_c_uncentered > (cp2.p as u128 / 2) {
            (crt_c_uncentered as i64) - (cp2.p as i64)
        } else {
            crt_c_uncentered as i64
        };
        assert_eq!(CRT_C, crt_c_centered as i32);
        assert_eq!(CRT_C, -8_855_555);

        // QINV32 * q = 1 mod 2^32
        assert_eq!((QINV32 as u32).wrapping_mul(Q as u32), 1);

        // CRT exactness bound: 4 * 256 * ((q - 1) / 2) * 2^17 < q * P2 / 2
        let az_bound: u128 = 4 * 256 * ((Q as u128 - 1) / 2) * (1u128 << 17);
        let crt_capacity: u128 = (Q as u128 * P2 as u128) / 2;
        assert!(
            az_bound < crt_capacity,
            "CRT bound failed: {az_bound} >= {crt_capacity}"
        );

        // 32-bit headroom: 9 * P2 < 2^31
        let p2_headroom: u128 = 9 * P2 as u128;
        assert!(
            p2_headroom < (1u128 << 31),
            "P2 headroom failed: {p2_headroom} >= 2^31"
        );

        // Prime P3 checks
        let cp3 = consts(Prime::P3);
        assert_eq!(cp3.p, 377_487_361);
        assert_eq!(
            pow_mod(cp3.zeta as u128, 256, cp3.p as u128),
            cp3.p as u128 - 1
        );
        assert_eq!((cp3.pinv as u32).wrapping_mul(cp3.p as u32), 1);
        let p3_headroom: u128 = 5 * P3 as u128;
        assert!(
            p3_headroom < (1u128 << 31),
            "P3 headroom failed: {p3_headroom} >= 2^31"
        );

        // CRT exactness bound for 65 and 87: q * P2 * P3 / 2 exceeds worst-case |A z|
        let three_prime_capacity: u128 = (Q as u128 * P2 as u128 * P3 as u128) / 2;
        let az_max_87: u128 = 7 * 256 * ((Q as u128 - 1) / 2) * (1u128 << 19);
        assert!(
            az_max_87 < three_prime_capacity,
            "Three prime CRT capacity insufficient: {az_max_87} >= {three_prime_capacity}"
        );
    }
}
