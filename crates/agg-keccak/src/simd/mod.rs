//! AVX-512 kernels of the Keccak witness generator, selected once at run time.
//!
//! Every kernel is a `#[target_feature]` function; [`avx512`] says whether the CPU has the
//! features they need (AVX-512 F/BW/VBMI, GFNI, BMI2) and whether `FLOCK_SIMD` allows
//! them (`portable` or `off` disables them, as for the vendored Flock kernels). The
//! callers keep the portable code as the fallback and as the tests' oracle.
//!
//! - [`keccak8`]: Keccak-f[1600] rounds on eight lane-sliced states (`[__m512i; 25]`,
//!   qword `r` of every vector is instance `r`).
//! - [`transpose`]: 8 x 64 bit-matrix transpose of one zmm (`vpermb` + `vgf2p8affineqb`).

#![allow(unsafe_code)]

#[cfg(target_arch = "x86_64")]
pub mod keccak8;
#[cfg(target_arch = "x86_64")]
pub mod transpose;

use std::sync::OnceLock;

/// Whether the AVX-512 kernels may run: the CPU has AVX-512 F/BW/VBMI, GFNI and BMI2, and
/// `FLOCK_SIMD` is neither `portable` nor `off`. Detected once per process.
#[must_use]
pub fn avx512() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| detected() && !capped())
}

/// The CPU features alone (ignores `FLOCK_SIMD`; tests use it to decide whether the vector
/// path can be compared with the portable one).
#[must_use]
pub fn detected() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::is_x86_feature_detected as has;
        has!("avx512f") && has!("avx512bw") && has!("avx512vbmi") && has!("gfni") && has!("bmi2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn capped() -> bool {
    matches!(
        std::env::var("FLOCK_SIMD").as_deref(),
        Ok("portable") | Ok("off")
    )
}
