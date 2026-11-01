//! AVX-512 kernels of the Keccak witness generator, selected once at run time (the AVX2
//! generator in `crate::witness` is selected by [`avx2`]).
//!
//! Every kernel is a `#[target_feature]` function; [`avx512`] says whether the CPU has the
//! features they need (AVX-512 F/BW/VBMI, GFNI, BMI2) and whether `FLOCK_SIMD` allows
//! them (`portable` or `off` disables them, as for the vendored Flock kernels) and the
//! process-wide `AGG_ISA` cap admits the `avx512` level. The callers keep the portable
//! code as the fallback and as the tests' oracle.
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

/// Whether the AVX2 witness kernels may run (the x86-64-v3 baseline: AVX2 and BMI2): the CPU
/// has them, `FLOCK_SIMD` is neither `portable` nor `off`, and the `AGG_ISA` cap admits the
/// `avx2` level. Detected once per process.
#[must_use]
pub fn avx2() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        let has = std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("bmi2");
        #[cfg(not(target_arch = "x86_64"))]
        let has = false;
        let flock = matches!(
            std::env::var("FLOCK_SIMD").as_deref(),
            Ok("portable") | Ok("off")
        );
        has && !flock && agg_mldsa_simd::isa::allows(agg_mldsa_simd::isa::IsaLevel::Avx2)
    })
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
    let flock = matches!(
        std::env::var("FLOCK_SIMD").as_deref(),
        Ok("portable") | Ok("off")
    );
    flock || !agg_mldsa_simd::isa::allows(agg_mldsa_simd::isa::IsaLevel::Avx512)
}
