//! Parameter sets for ML-DSA (FIPS 204) and derived aggregation constants.
//!
//! Provides the [`ParamSet`] trait with associated constants for ML-DSA-44,
//! ML-DSA-65, and ML-DSA-87, zero-sized types [`Set44`], [`Set65`], [`Set87`],
//! the run-time enum [`Set`], and the [`dispatch_set!`] macro.
//!
//! Top-level constants preserve the ML-DSA-44 configuration for existing uses.

/// Run-time identifier for ML-DSA parameter sets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Set {
    /// ML-DSA-44 (NIST Category 2).
    S44,
    /// ML-DSA-65 (NIST Category 3).
    S65,
    /// ML-DSA-87 (NIST Category 5).
    S87,
}

impl Set {
    /// All parameter set variants.
    pub const ALL: [Self; 3] = [Self::S44, Self::S65, Self::S87];

    /// Numeric identifier (0x44, 0x65, or 0x87).
    pub const fn id(self) -> u8 {
        match self {
            Self::S44 => 0x44,
            Self::S65 => 0x65,
            Self::S87 => 0x87,
        }
    }

    /// Decode from numeric identifier.
    pub const fn from_id(id: u8) -> Option<Self> {
        match id {
            0x44 => Some(Self::S44),
            0x65 => Some(Self::S65),
            0x87 => Some(Self::S87),
            _ => None,
        }
    }

    /// Short name ("44", "65", "87").
    pub const fn name(self) -> &'static str {
        match self {
            Self::S44 => "44",
            Self::S65 => "65",
            Self::S87 => "87",
        }
    }
}

impl TryFrom<u8> for Set {
    type Error = u8;

    fn try_from(id: u8) -> Result<Self, Self::Error> {
        Self::from_id(id).ok_or(id)
    }
}

impl core::fmt::Display for Set {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Monomorphise a generic computation over a runtime [`Set`] value.
#[macro_export]
macro_rules! dispatch_set {
    ($set:expr, |$P:ident| $body:expr) => {
        match $set {
            $crate::params::Set::S44 => {
                type $P = $crate::params::Set44;
                $body
            }
            $crate::params::Set::S65 => {
                type $P = $crate::params::Set65;
                $body
            }
            $crate::params::Set::S87 => {
                type $P = $crate::params::Set87;
                $body
            }
        }
    };
}

pub use crate::dispatch_set;

/// A polynomial with 256 signed integer coefficients.
pub type Poly = [i64; 256];

/// Compile-time parameter set specification for ML-DSA aggregation.
pub trait ParamSet: Copy + Send + Sync + 'static {
    /// Numeric identifier (0x44, 0x65, 0x87).
    const ID: u8;
    /// Full standard name.
    const NAME: &'static str;
    /// Associated run-time enum variant.
    const SET: Set;

    // -----------------------------------------------------------------------------------------
    // Primary parameters from FIPS 204 Table 1.
    // -----------------------------------------------------------------------------------------

    /// Modulus q = 8,380,417 (FIPS 204 Section 2.3).
    const Q: i64 = 8_380_417;
    /// Ring degree n = 256.
    const N: usize = 256;
    /// Dropped bits of t (d = 13).
    const D: u32 = 13;
    /// 512th root of unity modulo q used by NTT (FIPS 204 Section 7.5).
    const ZETA: i64 = 1753;
    /// 256^{-1} mod q (FIPS 204 Algorithm 42, line 21).
    const N_INV_Q: i64 = 8_347_681;
    /// Bytes of tr and mu.
    const TR_BYTES: usize = 64;

    /// Rows of matrix A (k).
    const K: usize;
    /// Columns of matrix A (l).
    const L: usize;
    /// Number of +-1 entries in challenge polynomial c (tau).
    const TAU: usize;
    /// Collision strength of c~ in bits (lambda).
    const LAMBDA: usize;
    /// Coefficient range of commitment y (gamma1).
    const GAMMA1: i64;
    /// Low-order rounding range (gamma2).
    const GAMMA2: i64;
    /// Secret key coefficient bound (eta).
    const ETA: i64;
    /// Bound on challenge product beta = tau * eta.
    const BETA: i64 = Self::TAU as i64 * Self::ETA;
    /// Maximum number of ones in hint vector h (omega).
    const OMEGA: usize;

    // -----------------------------------------------------------------------------------------
    // Bit widths and byte sizes (FIPS 204 Table 2).
    // -----------------------------------------------------------------------------------------

    /// Bits per t1 coefficient: bitlen(q - 1) - d = 23 - 13 = 10.
    const T1_BITS: usize = 10;
    /// Bits per packed z coefficient: 1 + bitlen(gamma1 - 1).
    const Z_BITS: usize = 1 + (64 - ((Self::GAMMA1 - 1) as u64).leading_zeros() as usize);
    /// Number of w1 values m_w = (q - 1) / (2 * gamma2).
    const M_W: i64 = (Self::Q - 1) / (2 * Self::GAMMA2);
    /// Bits per w1 coefficient: bitlen(m_w - 1).
    const W1_BITS: usize = 64 - ((Self::M_W - 1) as u64).leading_zeros() as usize;

    /// Bytes of commitment hash c~ (lambda / 4).
    const C_TILDE_BYTES: usize = Self::LAMBDA / 4;
    /// Public key byte length (FIPS 204 Table 2): 32 + 32 * k * t1_bits.
    const PK_BYTES: usize = 32 + 32 * Self::K * Self::T1_BITS;
    /// Signature byte length (FIPS 204 Table 2): c_tilde_bytes + l * 32 * z_bits + omega + k.
    const SIG_BYTES: usize =
        Self::C_TILDE_BYTES + Self::L * 32 * Self::Z_BITS + Self::OMEGA + Self::K;
    /// Bytes of w1Encode(w1): 32 * k * w1_bits.
    const W1_ENCODE_BYTES: usize = 32 * Self::K * Self::W1_BITS;
    /// w1Encode bit length per signature: k * 256 * w1_bits.
    const W1LEN: usize = Self::K * Self::N * Self::W1_BITS;

    // -----------------------------------------------------------------------------------------
    // Derived bounds.
    // -----------------------------------------------------------------------------------------

    /// Honest bound on |z|_inf: gamma1 - beta - 1.
    const B_Z: i64 = Self::GAMMA1 - Self::BETA - 1;
    /// UseHint residual bound: 2 * gamma2 + 1.
    const B_U: i64 = 2 * Self::GAMMA2 + 1;
    /// Bound on |2^d c t1| over Z: 2^d * tau * (2^t1_bits - 1).
    const B_T: i64 = (1 << Self::D) * Self::TAU as i64 * ((1 << Self::T1_BITS) - 1);
    /// Worst-case ring residual |Delta_0| before quotient:
    /// l * 256 * ((q - 1) / 2) * B_z + B_T + 2 * gamma2 * (m_w - 1) + B_u.
    const DELTA0_MAX: i64 = (Self::L * Self::N) as i64 * ((Self::Q - 1) / 2) * Self::B_Z
        + Self::B_T
        + 2 * Self::GAMMA2 * (Self::M_W - 1)
        + Self::B_U;
    /// Honest quotient bound B_v = floor(DELTA0_MAX / q).
    const B_V: i64 = Self::DELTA0_MAX / Self::Q;
    /// SelfTargetMSIS bound of FIPS 204: max(gamma1 - beta, 2 * gamma2 + 1 + 2^(d-1) * tau).
    const ZETA_MSIS: i64 = {
        let t = Self::B_U + (1 << (Self::D - 1)) * Self::TAU as i64;
        let b = Self::GAMMA1 - Self::BETA;
        if b > t {
            b
        } else {
            t
        }
    };

    /// Proven bound on |z| in relation H-u: bar B_z.
    const BAR_B_Z: i64;
    /// Proven bound on |u| in relation H-u: bar B_u = B_u.
    const BAR_B_U: i64 = Self::B_U;
    /// SelfTargetMSIS bound zeta' = max(bar B_z, bar B_u + 2^(d-1) * tau) for relation H-u.
    const ZETA_PRIME: i64 =
        if Self::BAR_B_Z > Self::BAR_B_U + (1 << (Self::D - 1)) * Self::TAU as i64 {
            Self::BAR_B_Z
        } else {
            Self::BAR_B_U + (1 << (Self::D - 1)) * Self::TAU as i64
        };

    /// Honest encoding range of the quotient v (two 16-bit halves): [-2^31, 2^31 - 1]. It is
    /// not proven: v carries no range check, and the projection bounds it.
    const V_HALF_RANGE: i64 = 1 << 31;

    // -----------------------------------------------------------------------------------------
    // Hypercube, polynomial lanes, and domain layout.
    // -----------------------------------------------------------------------------------------

    /// Vector-index variables in the hypercube (2 for Set44, 0 for Set65/Set87).
    const D_VEC: u32;
    /// Polynomial lanes for k-vectors (1 for Set44, 6 for Set65, 8 for Set87).
    const LANES_K: usize;
    /// Polynomial lanes for l-vectors (1 for Set44, 5 for Set65, 7 for Set87).
    const LANES_L: usize;
    /// Position variables d_pos = 8 + D_VEC.
    const D_POS: u32 = 8 + Self::D_VEC;
    /// Committed cells per signature column (1 << D_POS).
    const POSITIONS: usize = 1 << Self::D_POS;

    // -----------------------------------------------------------------------------------------
    // Bridge geometry constants.
    // -----------------------------------------------------------------------------------------

    /// Positions per signature P = 256 * k.
    const P_POSITIONS: usize = 256 * Self::K;
    /// Padded positions per signature P_pad = 2^{ceil(log2(P))}.
    const P_PAD: usize;
    /// log2(P_pad).
    const P_PAD_LOG2: u32;
    /// Signatures per level-1 block B_1 = 2^19 / P_pad.
    const B_1: usize = (1 << 19) / Self::P_PAD;
    /// log2(B_1): 9 for Set44, 8 for Set65/Set87.
    const BLOCK_RECORDS_LOG2: u32 = Self::B_1.trailing_zeros();
    /// Keccak permutations per signature (7 for Set44/Set65, 9 for Set87).
    const N_PERMS: usize;

    // -----------------------------------------------------------------------------------------
    // Projection and integer lift bounds.
    // -----------------------------------------------------------------------------------------

    /// Row blocks g: 1 for Set44 and Set65, 2 for Set87.
    const PROJ_BLOCKS: usize;
    /// Projection bound log2 B: 37 for Set44, 39 for Set65/Set87.
    const PROJ_BOUND_LOG2: u32;
    /// Projection matrix atoms (128 * P / 256).
    const PROJ_ATOMS: usize = 128 * Self::P_POSITIONS / 256;
    /// Projection matrix byte size (128 * P / 8).
    const PROJ_BYTES: usize = 128 * Self::P_POSITIONS / 8;

    // -----------------------------------------------------------------------------------------
    // H-u digit forms.
    // -----------------------------------------------------------------------------------------

    /// Step of the top digit of u for step layout (63,079 for Set44; 0 for Set65/Set87).
    const STEP_U: i64;
    /// Base of the top digit of u (6 for Set44, 8 for Set65/Set87).
    const U_TOP_BASE: u32;
    /// Slack for u in digit-slack form (523,267 for Set65/Set87; 0 for Set44).
    const U_SLACK: i64;
    /// Reconstruction offset for u in relation H-u.
    const U_OFFSET: i64;
    /// Weight of Boolean slack a_z for Set44 H-exact (130,915).
    const SLACK_Z: i64;
    /// Reconstruction offset for z in relation H-u (-229,376 for Set65/Set87; 0 for Set44).
    const Z_OFFSET: i64;

    // -----------------------------------------------------------------------------------------
    // Associated types and per-set vector/matrix storage.
    // -----------------------------------------------------------------------------------------

    /// Signer parameter set type in ml_dsa (MlDsa44, MlDsa65, MlDsa87).
    type Signer: ml_dsa::MlDsaParams;

    /// Vector of polynomials for the k rows.
    type VecK: core::ops::Index<usize, Output = Poly>
        + core::ops::IndexMut<usize, Output = Poly>
        + AsRef<[Poly]>
        + AsMut<[Poly]>
        + Clone
        + PartialEq
        + Eq
        + core::fmt::Debug
        + Send
        + Sync
        + 'static;

    /// Vector of polynomials for the l columns.
    type VecL: core::ops::Index<usize, Output = Poly>
        + core::ops::IndexMut<usize, Output = Poly>
        + AsRef<[Poly]>
        + AsMut<[Poly]>
        + Clone
        + PartialEq
        + Eq
        + core::fmt::Debug
        + Send
        + Sync
        + 'static;

    /// Small values per coefficient for the k rows (hints, w1).
    type SmallK: core::ops::Index<usize, Output = [u8; 256]>
        + core::ops::IndexMut<usize, Output = [u8; 256]>
        + AsRef<[[u8; 256]]>
        + AsMut<[[u8; 256]]>
        + Clone
        + PartialEq
        + Eq
        + core::fmt::Debug
        + Send
        + Sync
        + 'static;

    /// t1 values per coefficient for the k rows.
    type T1: core::ops::Index<usize, Output = [u16; 256]>
        + core::ops::IndexMut<usize, Output = [u16; 256]>
        + AsRef<[[u16; 256]]>
        + AsMut<[[u16; 256]]>
        + Clone
        + PartialEq
        + Eq
        + core::fmt::Debug
        + Send
        + Sync
        + 'static;

    /// Commitment hash c~ bytes (lambda / 4 bytes).
    type CTilde: AsRef<[u8]>
        + AsMut<[u8]>
        + Clone
        + Copy
        + PartialEq
        + Eq
        + core::fmt::Debug
        + Send
        + Sync
        + 'static;

    /// w1Encode bytes (32 * k * w1_bits bytes).
    type W1Encode: AsRef<[u8]>
        + AsMut<[u8]>
        + Clone
        + PartialEq
        + Eq
        + core::fmt::Debug
        + Send
        + Sync
        + 'static;

    /// k-vector of u64 coefficients (e.g. NTT domain residue modulo q).
    type VecKU64: core::ops::Index<usize, Output = [u64; 256]>
        + core::ops::IndexMut<usize, Output = [u64; 256]>
        + AsRef<[[u64; 256]]>
        + AsMut<[[u64; 256]]>
        + Clone
        + PartialEq
        + Eq
        + core::fmt::Debug
        + Send
        + Sync
        + 'static;

    /// Matrix of NTT polynomials (k rows x l columns).
    type MatrixNtt: core::fmt::Debug + Send + Sync + 'static;

    /// Matrix of centered coefficient polynomials (k rows x l columns).
    type MatrixCoeff: core::fmt::Debug + Send + Sync + 'static;

    /// Block counts for ExpandA (k rows x l columns).
    type ExpandBlocks: core::fmt::Debug + Send + Sync + 'static;

    /// SIMD prepared matrices or unit for reference expansion.
    type Prepared: Clone + core::fmt::Debug + Send + Sync + 'static;

    /// Target type for Deref on TView.
    type TTarget: core::ops::Index<usize, Output = [i32; 256]>
        + core::ops::IndexMut<usize, Output = [i32; 256]>
        + 'static;

    /// Creates a zeroed k-vector of polynomials.
    fn zero_vec_k() -> Self::VecK;
    /// Creates a zeroed l-vector of polynomials.
    fn zero_vec_l() -> Self::VecL;
    /// Creates zeroed small values for the k rows.
    fn zero_small_k() -> Self::SmallK;
    /// Creates zeroed t1 values for the k rows.
    fn zero_t1() -> Self::T1;
    /// Creates zeroed commitment hash c~ bytes.
    fn zero_c_tilde() -> Self::CTilde;
    /// Creates zeroed w1Encode bytes.
    fn zero_w1_encode() -> Self::W1Encode;
    /// Creates zeroed k-vector of u64 coefficients.
    fn zero_vec_k_u64() -> Self::VecKU64;
    /// Creates a boxed zeroed matrix of NTT polynomials.
    fn zero_matrix_ntt() -> Box<Self::MatrixNtt>;
    /// Creates a boxed zeroed matrix of centered coefficient polynomials.
    fn zero_matrix_coeff() -> Box<Self::MatrixCoeff>;
    /// Creates zeroed ExpandA block counts.
    fn zero_expand_blocks() -> Self::ExpandBlocks;
    /// Initializes prepared matrices for expansion.
    fn make_prepared(rho: &[u8; 32]) -> Self::Prepared;
    /// Expands an encoded public key for this parameter set.
    fn expand_key(pk: &[u8]) -> Option<crate::expand::ExpandedKey<Self>>;
    /// Strict witness derivation of a tile of records on the process-wide SIMD engine
    /// ([`crate::fast_lanes::derive_tile`]; ML-DSA-44, whose tables run [`crate::fast`],
    /// takes the legacy derivation here).
    ///
    /// # Errors
    /// The lowest index whose signature the legacy derivation rejects, with its error.
    fn derive_lane_tile(
        records: &[&crate::instance::PublicRecord<Self>],
        sigs: &[&[u8]],
        out: &mut [crate::fast_lanes::LaneWitness],
    ) -> Result<(), (usize, crate::derive::DeriveError)>;

    /// Reads entry (r, s) of the NTT matrix.
    fn get_ntt(m: &Self::MatrixNtt, r: usize, s: usize) -> &[u64; 256];
    /// Mutably reads entry (r, s) of the NTT matrix.
    fn get_ntt_mut(m: &mut Self::MatrixNtt, r: usize, s: usize) -> &mut [u64; 256];
    /// Reads entry (r, s) of the coefficient matrix.
    fn get_coeff(m: &Self::MatrixCoeff, r: usize, s: usize) -> &[i32; 256];
    /// Mutably reads entry (r, s) of the coefficient matrix.
    fn get_coeff_mut(m: &mut Self::MatrixCoeff, r: usize, s: usize) -> &mut [i32; 256];
    /// Reads entry (r, s) of ExpandA block counts.
    fn get_block(b: &Self::ExpandBlocks, r: usize, s: usize) -> usize;
    /// Writes entry (r, s) of ExpandA block counts.
    fn set_block(b: &mut Self::ExpandBlocks, r: usize, s: usize, val: usize);

    /// Converts a slice of 256-coefficient rows into the TView target.
    fn as_t_target(slice: &[[i32; 256]]) -> &Self::TTarget;
    /// Mutably converts a slice of 256-coefficient rows into the TView target.
    fn as_t_target_mut(slice: &mut [[i32; 256]]) -> &mut Self::TTarget;

    /// Decodes commitment hash c~ from the prefix of a slice, zero-padded when the slice is
    /// shorter than c~.
    fn c_tilde_from_slice(slice: &[u8]) -> Self::CTilde;
    /// Decodes w1Encode from the prefix of a slice.
    fn w1_encode_from_slice(slice: &[u8]) -> Self::W1Encode;
}

/// The first `M` bytes of `slice`, zero-padded when it is shorter.
fn zero_padded_prefix<const M: usize>(slice: &[u8]) -> [u8; M] {
    let mut out = [0u8; M];
    let n = slice.len().min(M);
    out[..n].copy_from_slice(&slice[..n]);
    out
}

/// Zero-sized type selecting ML-DSA-44 parameters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Set44;

impl ParamSet for Set44 {
    const ID: u8 = 0x44;
    const NAME: &'static str = "ML-DSA-44";
    const SET: Set = Set::S44;

    const K: usize = 4;
    const L: usize = 4;
    const TAU: usize = 39;
    const LAMBDA: usize = 128;
    const GAMMA1: i64 = 1 << 17;
    const GAMMA2: i64 = (Self::Q - 1) / 88;
    const ETA: i64 = 2;
    const OMEGA: usize = 80;

    const BAR_B_Z: i64 = 1 << 17;

    const D_VEC: u32 = 2;
    const LANES_K: usize = 1;
    const LANES_L: usize = 1;

    const P_PAD: usize = 1024;
    const P_PAD_LOG2: u32 = 10;
    const N_PERMS: usize = 7;

    const PROJ_BLOCKS: usize = 1;
    const PROJ_BOUND_LOG2: u32 = 37;

    const STEP_U: i64 = 63_079;
    const U_TOP_BASE: u32 = 6;
    const U_SLACK: i64 = 0;
    const U_OFFSET: i64 = -190_465;
    const SLACK_Z: i64 = 130_915;
    const Z_OFFSET: i64 = 0;

    type Signer = ml_dsa::MlDsa44;
    type VecK = [Poly; 4];
    type VecL = [Poly; 4];
    type SmallK = [[u8; 256]; 4];
    type T1 = [[u16; 256]; 4];
    type CTilde = [u8; 32];
    type W1Encode = [u8; 768];
    type VecKU64 = [[u64; 256]; 4];
    type MatrixNtt = [[[u64; 256]; 4]; 4];
    type MatrixCoeff = [[[i32; 256]; 4]; 4];
    type ExpandBlocks = [[usize; 4]; 4];
    type Prepared = crate::expand::PreparedMatrices;
    type TTarget = [[i32; 256]; 4];

    fn zero_vec_k() -> Self::VecK { [[0; 256]; 4] }
    fn zero_vec_l() -> Self::VecL { [[0; 256]; 4] }
    fn zero_small_k() -> Self::SmallK { [[0; 256]; 4] }
    fn zero_t1() -> Self::T1 { [[0; 256]; 4] }
    fn zero_c_tilde() -> Self::CTilde { [0; 32] }
    fn zero_w1_encode() -> Self::W1Encode { [0; 768] }
    fn zero_vec_k_u64() -> Self::VecKU64 { [[0; 256]; 4] }
    fn zero_matrix_ntt() -> Box<Self::MatrixNtt> { Box::new([[[0; 256]; 4]; 4]) }
    fn zero_matrix_coeff() -> Box<Self::MatrixCoeff> { Box::new([[[0; 256]; 4]; 4]) }
    fn zero_expand_blocks() -> Self::ExpandBlocks { [[0; 4]; 4] }
    fn make_prepared(rho: &[u8; 32]) -> Self::Prepared { crate::expand::make_prepared_44(rho) }
    fn expand_key(pk: &[u8]) -> Option<crate::expand::ExpandedKey<Self>> {
        crate::expand::ExpandedKey::<Set44>::many(&[pk]).pop().flatten()
    }
    fn derive_lane_tile(
        records: &[&crate::instance::PublicRecord<Self>],
        sigs: &[&[u8]],
        out: &mut [crate::fast_lanes::LaneWitness],
    ) -> Result<(), (usize, crate::derive::DeriveError)> {
        crate::fast_lanes::derive_tile_legacy(records, sigs, out)
    }

    fn get_ntt(m: &Self::MatrixNtt, r: usize, s: usize) -> &[u64; 256] { &m[r][s] }
    fn get_ntt_mut(m: &mut Self::MatrixNtt, r: usize, s: usize) -> &mut [u64; 256] { &mut m[r][s] }
    fn get_coeff(m: &Self::MatrixCoeff, r: usize, s: usize) -> &[i32; 256] { &m[r][s] }
    fn get_coeff_mut(m: &mut Self::MatrixCoeff, r: usize, s: usize) -> &mut [i32; 256] { &mut m[r][s] }
    fn get_block(b: &Self::ExpandBlocks, r: usize, s: usize) -> usize { b[r][s] }
    fn set_block(b: &mut Self::ExpandBlocks, r: usize, s: usize, val: usize) { b[r][s] = val; }

    fn as_t_target(slice: &[[i32; 256]]) -> &Self::TTarget { slice.try_into().expect("4 rows") }
    fn as_t_target_mut(slice: &mut [[i32; 256]]) -> &mut Self::TTarget { slice.try_into().expect("4 rows") }

    fn c_tilde_from_slice(slice: &[u8]) -> Self::CTilde { zero_padded_prefix(slice) }
    fn w1_encode_from_slice(slice: &[u8]) -> Self::W1Encode {
        let mut w = [0u8; 768];
        w.copy_from_slice(&slice[..768]);
        w
    }
}

/// Zero-sized type selecting ML-DSA-65 parameters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Set65;

impl ParamSet for Set65 {
    const ID: u8 = 0x65;
    const NAME: &'static str = "ML-DSA-65";
    const SET: Set = Set::S65;

    const K: usize = 6;
    const L: usize = 5;
    const TAU: usize = 49;
    const LAMBDA: usize = 192;
    const GAMMA1: i64 = 1 << 19;
    const GAMMA2: i64 = (Self::Q - 1) / 32;
    const ETA: i64 = 4;
    const OMEGA: usize = 55;

    const BAR_B_Z: i64 = 1 << 19;

    const D_VEC: u32 = 0;
    const LANES_K: usize = 6;
    const LANES_L: usize = 5;

    const P_PAD: usize = 2048;
    const P_PAD_LOG2: u32 = 11;
    const N_PERMS: usize = 7;

    const PROJ_BLOCKS: usize = 1;
    const PROJ_BOUND_LOG2: u32 = 39;

    const STEP_U: i64 = 0;
    const U_TOP_BASE: u32 = 8;
    const U_SLACK: i64 = 523_267;
    const U_OFFSET: i64 = -228_865;
    const SLACK_Z: i64 = 0;
    const Z_OFFSET: i64 = -229_376;

    type Signer = ml_dsa::MlDsa65;
    type VecK = [Poly; 6];
    type VecL = [Poly; 5];
    type SmallK = [[u8; 256]; 6];
    type T1 = [[u16; 256]; 6];
    type CTilde = [u8; 48];
    type W1Encode = [u8; 768];
    type VecKU64 = [[u64; 256]; 6];
    type MatrixNtt = [[[u64; 256]; 5]; 6];
    type MatrixCoeff = [[[i32; 256]; 5]; 6];
    type ExpandBlocks = [[usize; 5]; 6];
    type Prepared = ();
    type TTarget = [[i32; 256]; 6];

    fn zero_vec_k() -> Self::VecK { [[0; 256]; 6] }
    fn zero_vec_l() -> Self::VecL { [[0; 256]; 5] }
    fn zero_small_k() -> Self::SmallK { [[0; 256]; 6] }
    fn zero_t1() -> Self::T1 { [[0; 256]; 6] }
    fn zero_c_tilde() -> Self::CTilde { [0; 48] }
    fn zero_w1_encode() -> Self::W1Encode { [0; 768] }
    fn zero_vec_k_u64() -> Self::VecKU64 { [[0; 256]; 6] }
    fn zero_matrix_ntt() -> Box<Self::MatrixNtt> { Box::new([[[0; 256]; 5]; 6]) }
    fn zero_matrix_coeff() -> Box<Self::MatrixCoeff> { Box::new([[[0; 256]; 5]; 6]) }
    fn zero_expand_blocks() -> Self::ExpandBlocks { [[0; 5]; 6] }
    fn make_prepared(_rho: &[u8; 32]) -> Self::Prepared { () }
    fn expand_key(pk: &[u8]) -> Option<crate::expand::ExpandedKey<Self>> {
        crate::expand::expand_key_engine::<Set65, 6, 5>(pk)
    }
    fn derive_lane_tile(
        records: &[&crate::instance::PublicRecord<Self>],
        sigs: &[&[u8]],
        out: &mut [crate::fast_lanes::LaneWitness],
    ) -> Result<(), (usize, crate::derive::DeriveError)> {
        crate::fast_lanes::derive_tile(&agg_mldsa_simd::engine_for::<Self>(), records, sigs, out)
    }

    fn get_ntt(m: &Self::MatrixNtt, r: usize, s: usize) -> &[u64; 256] { &m[r][s] }
    fn get_ntt_mut(m: &mut Self::MatrixNtt, r: usize, s: usize) -> &mut [u64; 256] { &mut m[r][s] }
    fn get_coeff(m: &Self::MatrixCoeff, r: usize, s: usize) -> &[i32; 256] { &m[r][s] }
    fn get_coeff_mut(m: &mut Self::MatrixCoeff, r: usize, s: usize) -> &mut [i32; 256] { &mut m[r][s] }
    fn get_block(b: &Self::ExpandBlocks, r: usize, s: usize) -> usize { b[r][s] }
    fn set_block(b: &mut Self::ExpandBlocks, r: usize, s: usize, val: usize) { b[r][s] = val; }

    fn as_t_target(slice: &[[i32; 256]]) -> &Self::TTarget { slice.try_into().expect("6 rows") }
    fn as_t_target_mut(slice: &mut [[i32; 256]]) -> &mut Self::TTarget { slice.try_into().expect("6 rows") }

    fn c_tilde_from_slice(slice: &[u8]) -> Self::CTilde { zero_padded_prefix(slice) }
    fn w1_encode_from_slice(slice: &[u8]) -> Self::W1Encode {
        let mut w = [0u8; 768];
        w.copy_from_slice(&slice[..768]);
        w
    }
}

/// Zero-sized type selecting ML-DSA-87 parameters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Set87;

impl ParamSet for Set87 {
    const ID: u8 = 0x87;
    const NAME: &'static str = "ML-DSA-87";
    const SET: Set = Set::S87;

    const K: usize = 8;
    const L: usize = 7;
    const TAU: usize = 60;
    const LAMBDA: usize = 256;
    const GAMMA1: i64 = 1 << 19;
    const GAMMA2: i64 = (Self::Q - 1) / 32;
    const ETA: i64 = 2;
    const OMEGA: usize = 75;

    const BAR_B_Z: i64 = 1 << 19;

    const D_VEC: u32 = 0;
    const LANES_K: usize = 8;
    const LANES_L: usize = 7;

    const P_PAD: usize = 2048;
    const P_PAD_LOG2: u32 = 11;
    const N_PERMS: usize = 9;

    const PROJ_BLOCKS: usize = 2;
    const PROJ_BOUND_LOG2: u32 = 39;

    const STEP_U: i64 = 0;
    const U_TOP_BASE: u32 = 8;
    const U_SLACK: i64 = 523_267;
    const U_OFFSET: i64 = -228_865;
    const SLACK_Z: i64 = 0;
    const Z_OFFSET: i64 = -229_376;

    type Signer = ml_dsa::MlDsa87;
    type VecK = [Poly; 8];
    type VecL = [Poly; 7];
    type SmallK = [[u8; 256]; 8];
    type T1 = [[u16; 256]; 8];
    type CTilde = [u8; 64];
    type W1Encode = [u8; 1024];
    type VecKU64 = [[u64; 256]; 8];
    type MatrixNtt = [[[u64; 256]; 7]; 8];
    type MatrixCoeff = [[[i32; 256]; 7]; 8];
    type ExpandBlocks = [[usize; 7]; 8];
    type Prepared = ();
    type TTarget = [[i32; 256]; 8];

    fn zero_vec_k() -> Self::VecK { [[0; 256]; 8] }
    fn zero_vec_l() -> Self::VecL { [[0; 256]; 7] }
    fn zero_small_k() -> Self::SmallK { [[0; 256]; 8] }
    fn zero_t1() -> Self::T1 { [[0; 256]; 8] }
    fn zero_c_tilde() -> Self::CTilde { [0; 64] }
    fn zero_w1_encode() -> Self::W1Encode { [0; 1024] }
    fn zero_vec_k_u64() -> Self::VecKU64 { [[0; 256]; 8] }
    fn zero_matrix_ntt() -> Box<Self::MatrixNtt> { Box::new([[[0; 256]; 7]; 8]) }
    fn zero_matrix_coeff() -> Box<Self::MatrixCoeff> { Box::new([[[0; 256]; 7]; 8]) }
    fn zero_expand_blocks() -> Self::ExpandBlocks { [[0; 7]; 8] }
    fn make_prepared(_rho: &[u8; 32]) -> Self::Prepared { () }
    fn expand_key(pk: &[u8]) -> Option<crate::expand::ExpandedKey<Self>> {
        crate::expand::expand_key_engine::<Set87, 8, 7>(pk)
    }
    fn derive_lane_tile(
        records: &[&crate::instance::PublicRecord<Self>],
        sigs: &[&[u8]],
        out: &mut [crate::fast_lanes::LaneWitness],
    ) -> Result<(), (usize, crate::derive::DeriveError)> {
        crate::fast_lanes::derive_tile(&agg_mldsa_simd::engine_for::<Self>(), records, sigs, out)
    }

    fn get_ntt(m: &Self::MatrixNtt, r: usize, s: usize) -> &[u64; 256] { &m[r][s] }
    fn get_ntt_mut(m: &mut Self::MatrixNtt, r: usize, s: usize) -> &mut [u64; 256] { &mut m[r][s] }
    fn get_coeff(m: &Self::MatrixCoeff, r: usize, s: usize) -> &[i32; 256] { &m[r][s] }
    fn get_coeff_mut(m: &mut Self::MatrixCoeff, r: usize, s: usize) -> &mut [i32; 256] { &mut m[r][s] }
    fn get_block(b: &Self::ExpandBlocks, r: usize, s: usize) -> usize { b[r][s] }
    fn set_block(b: &mut Self::ExpandBlocks, r: usize, s: usize, val: usize) { b[r][s] = val; }

    fn as_t_target(slice: &[[i32; 256]]) -> &Self::TTarget { slice.try_into().expect("8 rows") }
    fn as_t_target_mut(slice: &mut [[i32; 256]]) -> &mut Self::TTarget { slice.try_into().expect("8 rows") }

    fn c_tilde_from_slice(slice: &[u8]) -> Self::CTilde { zero_padded_prefix(slice) }
    fn w1_encode_from_slice(slice: &[u8]) -> Self::W1Encode {
        let mut w = [0u8; 1024];
        w.copy_from_slice(&slice[..1024]);
        w
    }
}

// The SIMD crate's parameter trait is derived from the witness constants above, so the two
// cannot drift; the bound C_B_Q on |e / q| is not a witness constant and is copied.
// `simd_params_agree` then ties both to the SIMD crate's own Set44/Set65/Set87.
macro_rules! impl_simd_params {
    ($set:ty, $c_b_q:expr) => {
        impl agg_mldsa_simd::ParamSet for $set {
            const ID: u8 = <$set as ParamSet>::ID;
            const K: usize = <$set as ParamSet>::K;
            const L: usize = <$set as ParamSet>::L;
            const TAU: usize = <$set as ParamSet>::TAU;
            const ETA: i32 = <$set as ParamSet>::ETA as i32;
            const GAMMA1: i32 = <$set as ParamSet>::GAMMA1 as i32;
            const GAMMA2: i32 = <$set as ParamSet>::GAMMA2 as i32;
            const OMEGA: usize = <$set as ParamSet>::OMEGA;
            const LAMBDA: usize = <$set as ParamSet>::LAMBDA;
            const Z_BITS: usize = <$set as ParamSet>::Z_BITS;
            const W1_BITS: usize = <$set as ParamSet>::W1_BITS;
            const M_W: i32 = <$set as ParamSet>::M_W as i32;
            const B_U: i32 = <$set as ParamSet>::B_U as i32;
            const B_V: i32 = <$set as ParamSet>::B_V as i32;
            const B_T: i32 = <$set as ParamSet>::B_T as i32;
            const C_B_Q: u32 = $c_b_q;
            const U_TOP_BASE: u32 = <$set as ParamSet>::U_TOP_BASE;
            const STEP_U: i32 = <$set as ParamSet>::STEP_U as i32;
            const U_SLACK: i32 = <$set as ParamSet>::U_SLACK as i32;
            const U_OFFSET: i32 = <$set as ParamSet>::U_OFFSET as i32;
            const Z_OFFSET: i32 = <$set as ParamSet>::Z_OFFSET as i32;
            const C_TILDE_BYTES: usize = <$set as ParamSet>::C_TILDE_BYTES;
            const PK_BYTES: usize = <$set as ParamSet>::PK_BYTES;
            const SIG_BYTES: usize = <$set as ParamSet>::SIG_BYTES;
            const W1_ENCODE_BYTES: usize = <$set as ParamSet>::W1_ENCODE_BYTES;
        }
    };
}

impl_simd_params!(Set44, 41);
impl_simd_params!(Set65, 51);
impl_simd_params!(Set87, 62);

/// Compile-time check that the SIMD parameter trait of a witness set equals the SIMD crate's own
/// set of the same name (every constant, including the lossless i64 to i32 narrowing).
macro_rules! simd_params_agree {
    ($w:ty, $s:ty) => {
        const _: () = {
            use agg_mldsa_simd::ParamSet as S;
            assert!(<$w as S>::ID == <$s as S>::ID);
            assert!(<$w as S>::K == <$s as S>::K && <$w as S>::L == <$s as S>::L);
            assert!(<$w as S>::TAU == <$s as S>::TAU && <$w as S>::ETA == <$s as S>::ETA);
            assert!(<$w as S>::GAMMA1 == <$s as S>::GAMMA1);
            assert!(<$w as S>::GAMMA2 == <$s as S>::GAMMA2);
            assert!(<$w as S>::OMEGA == <$s as S>::OMEGA && <$w as S>::LAMBDA == <$s as S>::LAMBDA);
            assert!(<$w as S>::Z_BITS == <$s as S>::Z_BITS);
            assert!(<$w as S>::W1_BITS == <$s as S>::W1_BITS && <$w as S>::M_W == <$s as S>::M_W);
            assert!(<$w as S>::B_U == <$s as S>::B_U && <$w as S>::B_V == <$s as S>::B_V);
            assert!(<$w as S>::B_T == <$s as S>::B_T && <$w as S>::C_B_Q == <$s as S>::C_B_Q);
            assert!(<$w as S>::U_TOP_BASE == <$s as S>::U_TOP_BASE);
            assert!(<$w as S>::STEP_U == <$s as S>::STEP_U);
            assert!(<$w as S>::U_SLACK == <$s as S>::U_SLACK);
            assert!(<$w as S>::U_OFFSET == <$s as S>::U_OFFSET);
            assert!(<$w as S>::Z_OFFSET == <$s as S>::Z_OFFSET);
            assert!(<$w as S>::C_TILDE_BYTES == <$s as S>::C_TILDE_BYTES);
            assert!(<$w as S>::PK_BYTES == <$s as S>::PK_BYTES);
            assert!(<$w as S>::SIG_BYTES == <$s as S>::SIG_BYTES);
            assert!(<$w as S>::W1_ENCODE_BYTES == <$s as S>::W1_ENCODE_BYTES);
            // The narrowed i32 values equal the witness i64 values.
            assert!(<$w as S>::B_V as i64 == <$w as ParamSet>::B_V);
            assert!(<$w as S>::B_T as i64 == <$w as ParamSet>::B_T);
            assert!(<$w as S>::GAMMA1 as i64 == <$w as ParamSet>::GAMMA1);
            assert!(<$w as S>::U_OFFSET as i64 == <$w as ParamSet>::U_OFFSET);
            assert!(<$w as S>::Z_OFFSET as i64 == <$w as ParamSet>::Z_OFFSET);
            assert!(<$w as S>::U_SLACK as i64 == <$w as ParamSet>::U_SLACK);
            assert!(<$w as S>::STEP_U as i64 == <$w as ParamSet>::STEP_U);
        };
    };
}

simd_params_agree!(Set44, agg_mldsa_simd::Set44);
simd_params_agree!(Set65, agg_mldsa_simd::Set65);
simd_params_agree!(Set87, agg_mldsa_simd::Set87);

/// Module providing parameter set types and dispatch helpers.
pub mod set {
    pub use super::{dispatch_set, ParamSet, Set, Set44, Set65, Set87};
}

// ---------------------------------------------------------------------------------------------
// Top-level constants: preserved ML-DSA-44 instantiation for compatibility.
// ---------------------------------------------------------------------------------------------

/// Modulus q (FIPS 204 Table 1).
pub const Q: i64 = Set44::Q;
/// Ring degree n.
pub const N: usize = Set44::N;
/// Rows of A (k).
pub const K: usize = Set44::K;
/// Columns of A (l).
pub const L: usize = Set44::L;
/// Dropped bits of t (d).
pub const D: u32 = Set44::D;
/// Number of +-1 entries of c (tau).
pub const TAU: usize = Set44::TAU;
/// Collision strength of c~ (lambda).
pub const LAMBDA: usize = Set44::LAMBDA;
/// Coefficient range of y (gamma1 = 2^17).
pub const GAMMA1: i64 = Set44::GAMMA1;
/// Low-order rounding range (gamma2 = (q-1)/88).
pub const GAMMA2: i64 = Set44::GAMMA2;
/// Secret key range (eta).
pub const ETA: i64 = Set44::ETA;
/// beta = tau * eta.
pub const BETA: i64 = Set44::BETA;
/// Maximum number of ones in the hint (omega).
pub const OMEGA: usize = Set44::OMEGA;
/// 512th root of unity modulo q used by the NTT (FIPS 204 Section 7.5).
pub const ZETA: i64 = Set44::ZETA;
/// 256^{-1} mod q (FIPS 204 Algorithm 42, line 21).
pub const N_INV_Q: i64 = Set44::N_INV_Q;

/// Bytes of c~ (lambda / 4).
pub const C_TILDE_BYTES: usize = Set44::C_TILDE_BYTES;
/// Bits per t1 coefficient: bitlen(q-1) - d.
pub const T1_BITS: usize = Set44::T1_BITS;
/// Bits per packed z coefficient: 1 + bitlen(gamma1 - 1).
pub const Z_BITS: usize = Set44::Z_BITS;
/// Bits per w1 coefficient: bitlen((q-1)/(2 gamma2) - 1).
pub const W1_BITS: usize = Set44::W1_BITS;
/// Public key bytes (FIPS 204 Table 2): 32 + 32 k (bitlen(q-1) - d).
pub const PK_BYTES: usize = Set44::PK_BYTES;
/// Signature bytes (FIPS 204 Table 2): lambda/4 + l 32 (1 + bitlen(gamma1-1)) + omega + k.
pub const SIG_BYTES: usize = Set44::SIG_BYTES;
/// Bytes of tr and mu.
pub const TR_BYTES: usize = Set44::TR_BYTES;
/// Bytes of w1Encode(w1).
pub const W1_ENCODE_BYTES: usize = Set44::W1_ENCODE_BYTES;

// ---------------------------------------------------------------------------------------------
// Derived constants (ML-DSA-44).
// ---------------------------------------------------------------------------------------------

/// m_w = (q-1)/(2 gamma2): number of w1 values.
pub const M_W: i64 = Set44::M_W;
/// B_z = gamma1 - beta - 1: honest bound on |z|_inf.
pub const B_Z: i64 = Set44::B_Z;
/// B_u = 2 gamma2 + 1: UseHint residual bound.
pub const B_U: i64 = Set44::B_U;
/// B_T = 2^d tau (2^10 - 1): bound on |2^d c t1| over Z.
pub const B_T: i64 = Set44::B_T;
/// Worst-case |Delta_0| (the ring residual before the quotient).
pub const DELTA0_MAX: i64 = Set44::DELTA0_MAX;
/// B_v = floor(DELTA0_MAX / q): honest quotient bound.
pub const B_V: i64 = Set44::B_V;
/// SelfTargetMSIS bound of FIPS 204: max(gamma1 - beta, 2 gamma2 + 1 + 2^(d-1) tau).
pub const ZETA_MSIS: i64 = Set44::ZETA_MSIS;
/// w1Encode bits per signature: k 256 6.
pub const W1LEN: usize = Set44::W1LEN;

/// Proven bound on |z| in relation H-u: bar B_z = 2^17 (z in [-2^17, 2^17 - 1]).
pub const BAR_B_Z: i64 = Set44::BAR_B_Z;
/// Proven bound on |u| in relation H-u: bar B_u = B_u = 190,465, exact and symmetric.
pub const BAR_B_U: i64 = Set44::BAR_B_U;
/// Step of the six-valued top digit of u in relation H-u.
pub const STEP_U: i64 = Set44::STEP_U;
/// Base of the top digit H_u of u in relation H-u.
pub const U_TOP_BASE: u32 = Set44::U_TOP_BASE;
/// Weight of the Boolean slack a_z of z in relation H-exact.
pub const SLACK_Z: i64 = Set44::SLACK_Z;
/// Honest encoding range of the quotient v (two 16-bit halves): [-2^31, 2^31 - 1]; not proven,
/// the projection bounds v.
pub const V_HALF_RANGE: i64 = Set44::V_HALF_RANGE;
/// SelfTargetMSIS bound zeta' = max(bar B_z, bar B_u + 2^(d-1) tau) for relation H-u.
pub const ZETA_PRIME: i64 = Set44::ZETA_PRIME;

/// PIOP field modulus p = 2^64 - 59 (Akita `Prime64Offset59`).
pub const P_FIELD: u64 = 0xFFFF_FFFF_FFFF_FFC5;

// ---------------------------------------------------------------------------------------------
// Batch geometry constants (ML-DSA-44).
// ---------------------------------------------------------------------------------------------

/// Maximum number of live records.
pub const N_MAX: usize = 1 << 15;
/// Minimum padded batch size N*.
pub const N_STAR_MIN: usize = 8;
/// Coefficient variables.
pub const D_COEF: u32 = 8;
/// Vector-index variables (k = l = 4).
pub const D_VEC: u32 = Set44::D_VEC;
/// Position variables d_pos = d_coef + D_VEC.
pub const D_POS: u32 = Set44::D_POS;
/// Positions per signature (k 256 = l 256 = 1024).
pub const POSITIONS: usize = Set44::POSITIONS;
/// Segment size exponent: a segment is 2^13 signatures.
pub const D_SEG_MAX: u32 = 13;
/// Akita minimum physical table size exponent (`PCS_MIN_CELLS_LOG2`).
pub const PCS_MIN_CELLS_LOG2: u32 = 13;

// Compile-time checks of the derived bounds for all parameter sets.
const _: () = assert!(Set44::B_T < 1 << 29);
const _: () = assert!(Set44::DELTA0_MAX < 1 << 49);
const _: () = assert!(Set44::B_V < Set44::V_HALF_RANGE);
const _: () = assert!((Set44::U_TOP_BASE as i64 - 1) * Set44::STEP_U + (1 << 16) == 2 * Set44::B_U + 1);
const _: () = assert!(Set44::STEP_U <= 1 << 16);
const _: () = assert!(Set44::SLACK_Z + (1 << 17) == 2 * Set44::B_Z + 1);
const _: () = assert!(Set44::SLACK_Z <= 1 << 17);

const _: () = assert!(Set65::B_T < 1 << 29);
const _: () = assert!(Set65::DELTA0_MAX < 1 << 52);
const _: () = assert!(Set65::B_V < Set65::V_HALF_RANGE);
const _: () = assert!(Set65::U_SLACK + (1 << 19) == 2 * Set65::B_U + 1);
const _: () = assert!((Set65::P_POSITIONS / Set65::PROJ_BLOCKS) as i64 * Set65::B_V < (1i64 << Set65::PROJ_BOUND_LOG2));

const _: () = assert!(Set87::B_T < 1 << 29);
const _: () = assert!(Set87::DELTA0_MAX < 1 << 52);
const _: () = assert!(Set87::B_V < Set87::V_HALF_RANGE);
const _: () = assert!(Set87::U_SLACK + (1 << 19) == 2 * Set87::B_U + 1);
const _: () = assert!((Set87::P_POSITIONS / Set87::PROJ_BLOCKS) as i64 * Set87::B_V < (1i64 << Set87::PROJ_BOUND_LOG2));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fips_values_and_sizes_44() {
        assert_eq!(GAMMA2, 95_232);
        assert_eq!(BETA, 78);
        assert_eq!(PK_BYTES, 1312);
        assert_eq!(SIG_BYTES, 2420);
        assert_eq!(W1_ENCODE_BYTES, 768);
        assert_eq!(
            64 - ((Q - 1) as u64).leading_zeros() as usize - D as usize,
            T1_BITS
        );
        assert_eq!(
            1 + (64 - ((GAMMA1 - 1) as u64).leading_zeros() as usize),
            Z_BITS
        );
        assert_eq!(64 - ((M_W - 1) as u64).leading_zeros() as usize, W1_BITS);
        assert_eq!((256 * N_INV_Q) % Q, 1);
        let mut x = 1i64;
        for _ in 0..256 {
            x = x * ZETA % Q;
        }
        assert_eq!(x, Q - 1);
    }

    #[test]
    fn fips_values_and_sizes_65() {
        assert_eq!(Set65::K, 6);
        assert_eq!(Set65::L, 5);
        assert_eq!(Set65::TAU, 49);
        assert_eq!(Set65::LAMBDA, 192);
        assert_eq!(Set65::GAMMA1, 1 << 19);
        assert_eq!(Set65::GAMMA2, 261_888);
        assert_eq!(Set65::ETA, 4);
        assert_eq!(Set65::BETA, 196);
        assert_eq!(Set65::OMEGA, 55);
        assert_eq!(Set65::T1_BITS, 10);
        assert_eq!(Set65::Z_BITS, 20);
        assert_eq!(Set65::W1_BITS, 4);
        assert_eq!(Set65::M_W, 16);
        assert_eq!(Set65::C_TILDE_BYTES, 48);
        // FIPS 204 Table 2: ML-DSA-65 public key 1,952 B, signature 3,309 B.
        assert_eq!(Set65::PK_BYTES, 1952);
        assert_eq!(Set65::SIG_BYTES, 3309);
        assert_eq!(Set65::W1_ENCODE_BYTES, 768);
        assert_eq!(Set65::W1LEN, 6144);
    }

    #[test]
    fn fips_values_and_sizes_87() {
        assert_eq!(Set87::K, 8);
        assert_eq!(Set87::L, 7);
        assert_eq!(Set87::TAU, 60);
        assert_eq!(Set87::LAMBDA, 256);
        assert_eq!(Set87::GAMMA1, 1 << 19);
        assert_eq!(Set87::GAMMA2, 261_888);
        assert_eq!(Set87::ETA, 2);
        assert_eq!(Set87::BETA, 120);
        assert_eq!(Set87::OMEGA, 75);
        assert_eq!(Set87::T1_BITS, 10);
        assert_eq!(Set87::Z_BITS, 20);
        assert_eq!(Set87::W1_BITS, 4);
        assert_eq!(Set87::M_W, 16);
        assert_eq!(Set87::C_TILDE_BYTES, 64);
        // FIPS 204 Table 2: ML-DSA-87 public key 2,592 B, signature 4,627 B.
        assert_eq!(Set87::PK_BYTES, 2592);
        assert_eq!(Set87::SIG_BYTES, 4627);
        assert_eq!(Set87::W1_ENCODE_BYTES, 1024);
        assert_eq!(Set87::W1LEN, 8192);
    }

    #[test]
    fn derived_constants_44() {
        assert_eq!(M_W, 44);
        assert_eq!(B_Z, 130_993);
        assert_eq!(B_U, 190_465);
        assert_eq!(B_T, 326_836_224);
        assert_eq!(DELTA0_MAX, 562_061_561_757_697);
        assert_eq!(B_V, 67_068_447);
        assert_eq!(W1LEN, 6144);
        let zeta = std::cmp::max(GAMMA1 - BETA, 2 * GAMMA2 + 1 + (1 << (D - 1)) * TAU as i64);
        assert_eq!(zeta, ZETA_MSIS);
        assert_eq!(ZETA_PRIME, 350_209);
        assert_eq!(ZETA_PRIME, ZETA_MSIS);
        assert_eq!(BAR_B_U, 190_465);
        assert_eq!(5 * STEP_U + (1 << 16), 2 * B_U + 1);
        assert_eq!(SLACK_Z + (1 << 17), 2 * B_Z + 1);
        assert_eq!(P_FIELD, u64::MAX - 58);
    }

    #[test]
    fn derived_constants_65_and_87() {
        // ML-DSA-65
        assert_eq!(Set65::B_Z, 524_091);
        assert_eq!(Set65::B_U, 523_777);
        assert_eq!(Set65::B_T, 410_640_384);
        assert_eq!(Set65::DELTA0_MAX, 2_810_944_804_208_641);
        assert_eq!(Set65::B_V, 335_418_249);
        assert_eq!(Set65::ZETA_MSIS, 724_481);
        assert_eq!(Set65::ZETA_PRIME, 724_481);
        assert_eq!(Set65::BAR_B_Z, 1 << 19);
        assert_eq!(Set65::BAR_B_U, 523_777);

        // ML-DSA-87
        assert_eq!(Set87::B_Z, 524_167);
        assert_eq!(Set87::B_U, 523_777);
        assert_eq!(Set87::B_T, 502_824_960);
        assert_eq!(Set87::DELTA0_MAX, 3_935_893_323_276_289);
        assert_eq!(Set87::B_V, 469_653_636);
        assert_eq!(Set87::ZETA_MSIS, 769_537);
        assert_eq!(Set87::ZETA_PRIME, 769_537);
        assert_eq!(Set87::BAR_B_Z, 1 << 19);
        assert_eq!(Set87::BAR_B_U, 523_777);
    }

    #[test]
    fn hu_digit_forms_65_and_87() {
        // Step-slack exact tiling: U_SLACK + 2^19 = 2 * B_u + 1.
        assert_eq!(Set65::U_SLACK + (1 << 19), 2 * Set65::B_U + 1);
        assert_eq!(Set87::U_SLACK + (1 << 19), 2 * Set87::B_U + 1);

        // Reconstruction constants:
        // u = U_lo + 2^16 * U_mid + 523_267 * a_u - 228_865
        // where U_lo = B_ulo - 2^15, U_mid = H_u - 4.
        let u_const = 32_768i64 + 262_144 - Set65::B_U;
        assert_eq!(u_const, -228_865);
        assert_eq!(Set65::U_OFFSET, -228_865);
        assert_eq!(Set87::U_OFFSET, -228_865);

        // z relaxed form:
        // z = Z_lo + 2^16 * Z_mid + 2^19 * a_z - 229_376
        // where Z_lo = z_lo - 2^15, Z_mid = z_mid - 4.
        let z_const = 32_768i64 + 262_144 - (1 << 19);
        assert_eq!(z_const, -229_376);
        assert_eq!(Set65::Z_OFFSET, -229_376);
        assert_eq!(Set87::Z_OFFSET, -229_376);
    }

    #[test]
    fn projection_and_lift_bounds() {
        // Projection atoms and bytes
        assert_eq!(Set44::PROJ_ATOMS, 512);
        assert_eq!(Set44::PROJ_BYTES, 16_384);
        assert_eq!(Set65::PROJ_ATOMS, 768);
        assert_eq!(Set65::PROJ_BYTES, 24_576);
        assert_eq!(Set87::PROJ_ATOMS, 1024);
        assert_eq!(Set87::PROJ_BYTES, 32_768);

        // Honest projection bounds (P / g) * B_v < 2^B
        let p44 = (Set44::P_POSITIONS / Set44::PROJ_BLOCKS) as i64 * Set44::B_V;
        assert_eq!(p44, 68_678_089_728);
        assert!(p44 < 1i64 << Set44::PROJ_BOUND_LOG2);

        let p65 = (Set65::P_POSITIONS / Set65::PROJ_BLOCKS) as i64 * Set65::B_V;
        assert_eq!(p65, 515_202_430_464);
        assert!(p65 < 1i64 << Set65::PROJ_BOUND_LOG2);

        let p87 = (Set87::P_POSITIONS / Set87::PROJ_BLOCKS) as i64 * Set87::B_V;
        assert_eq!(p87, 480_925_323_264);
        assert!(p87 < 1i64 << Set87::PROJ_BOUND_LOG2);

        // Integer lift bound |Delta^Z| < p / 2
        let p_half = (P_FIELD / 2) as i128;

        for (l, bar_bz, bu, mw, g2, tau, proj_log2) in [
            (
                Set44::L,
                Set44::BAR_B_Z,
                Set44::BAR_B_U,
                Set44::M_W,
                Set44::GAMMA2,
                Set44::TAU,
                Set44::PROJ_BOUND_LOG2,
            ),
            (
                Set65::L,
                Set65::BAR_B_Z,
                Set65::BAR_B_U,
                Set65::M_W,
                Set65::GAMMA2,
                Set65::TAU,
                Set65::PROJ_BOUND_LOG2,
            ),
            (
                Set87::L,
                Set87::BAR_B_Z,
                Set87::BAR_B_U,
                Set87::M_W,
                Set87::GAMMA2,
                Set87::TAU,
                Set87::PROJ_BOUND_LOG2,
            ),
        ] {
            let az = (l * N) as i128 * ((Q - 1) / 2) as i128 * bar_bz as i128;
            let bt = (1i128 << D) * tau as i128 * ((1i128 << T1_BITS) - 1);
            let d_z = az
                + bt
                + (2 * g2 * (mw - 1)) as i128
                + bu as i128
                + Q as i128 * (2i128 << proj_log2);
            assert!(d_z < p_half);
        }
    }

    #[test]
    fn c_tilde_from_short_slices_is_zero_padded() {
        let sig: Vec<u8> = (1..=70).collect();
        for n in [0, 1, 31, 32, 47, 48, 63, 64, 70] {
            let s = &sig[..n];
            let pad = |m: usize| -> Vec<u8> { (0..m).map(|i| if i < n { sig[i] } else { 0 }).collect() };
            assert_eq!(Set44::c_tilde_from_slice(s).to_vec(), pad(32));
            assert_eq!(Set65::c_tilde_from_slice(s).to_vec(), pad(48));
            assert_eq!(Set87::c_tilde_from_slice(s).to_vec(), pad(64));
        }
    }

    #[test]
    fn dispatch_set_macro_and_enum() {
        for set in Set::ALL {
            let id = dispatch_set!(set, |P| P::ID);
            assert_eq!(id, set.id());
            assert_eq!(Set::from_id(id), Some(set));
            assert_eq!(Set::try_from(id), Ok(set));
        }
        assert_eq!(Set::from_id(0x00), None);
        assert_eq!(Set::from_id(0xFF), None);
    }
}
