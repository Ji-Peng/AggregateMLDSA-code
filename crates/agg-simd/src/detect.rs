//! Select the backend once, before processing any slices.
//!
//! `AGG_ISA` is the process-wide instruction-set cap shared by every backend
//! of the prover; `AGG_SIMD` names a backend below that cap. The IFMA kernels
//! need the full `avx512ifma` level; `AGG_ISA=avx512` selects the AVX-512F
//! kernels without IFMA and `AGG_ISA=avx2` the AVX2 kernels, even on a CPU
//! that has IFMA. `AGG_SIMD=portable` turns every vector kernel off.

use std::sync::OnceLock;

/// The backends of this crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Portable arithmetic.
    Portable,
    /// AVX-512 IFMA (the `avx512ifma` tier).
    Avx512,
    /// AVX-512F without IFMA: eight lanes of 32 x 32-bit partial products (the `avx512` tier).
    Avx512F,
    /// AVX2: four lanes of 32 x 32-bit partial products (the `avx2` tier).
    Avx2,
}

/// Instruction-set levels of `AGG_ISA`, ordered upwards.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum IsaLevel {
    /// Portable kernels only.
    Scalar,
    /// x86-64-v3 plus PCLMULQDQ.
    Avx2,
    /// AVX-512 without IFMA.
    Avx512,
    /// Every detected extension, including IFMA.
    Avx512Ifma,
}

impl IsaLevel {
    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "scalar" => Some(Self::Scalar),
            "avx2" => Some(Self::Avx2),
            "avx512" | "avx-512" => Some(Self::Avx512),
            "avx512ifma" | "avx512-ifma" | "ifma" => Some(Self::Avx512Ifma),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2",
            Self::Avx512 => "avx512",
            Self::Avx512Ifma => "avx512ifma",
        }
    }
}

/// The `AGG_ISA` cap, parsed once. An empty value is treated as unset.
fn cap() -> IsaLevel {
    static CAP: OnceLock<IsaLevel> = OnceLock::new();
    *CAP.get_or_init(|| {
        let Some(value) = std::env::var("AGG_ISA")
            .ok()
            .filter(|value| !value.trim().is_empty())
        else {
            return IsaLevel::Avx512Ifma;
        };
        IsaLevel::parse(&value).unwrap_or_else(|| {
            panic!("invalid AGG_ISA '{value}': expected avx512ifma, avx512, avx2 or scalar")
        })
    })
}

/// Whether the CPU has every feature the AVX-512 kernels use.
pub fn available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512dq")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vl")
            && std::is_x86_feature_detected!("avx512ifma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Whether the integer AVX-512 kernels (F/DQ/BW/VL, no IFMA) may run: the CPU has those
/// features and the `AGG_ISA` cap admits the `avx512` level. Detected once per process.
pub fn avx512_int() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        let has = std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512dq")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vl");
        #[cfg(not(target_arch = "x86_64"))]
        let has = false;
        has && cap() >= IsaLevel::Avx512
    })
}

/// Whether the AVX2 integer kernels may run: the CPU has AVX2 and the `AGG_ISA` cap admits
/// the `avx2` level. Detected once per process.
pub fn avx2() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        let has = std::is_x86_feature_detected!("avx2");
        #[cfg(not(target_arch = "x86_64"))]
        let has = false;
        has && cap() >= IsaLevel::Avx2
    })
}

/// Whether the CPU has AVX-512F (the kernels of [`Backend::Avx512F`] use nothing else).
pub fn avx512f_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Whether the CPU has AVX2.
pub fn avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// The best backend without IFMA below the cap, or the portable one.
fn below_ifma() -> Backend {
    if cap() >= IsaLevel::Avx512 && avx512f_available() {
        Backend::Avx512F
    } else if cap() >= IsaLevel::Avx2 && avx2_available() {
        Backend::Avx2
    } else {
        Backend::Portable
    }
}

/// The backend this process may use.
pub fn backend() -> Backend {
    static BACKEND: OnceLock<Backend> = OnceLock::new();
    *BACKEND.get_or_init(|| {
        let allowed = cap() >= IsaLevel::Avx512Ifma && available();
        match std::env::var("AGG_SIMD").as_deref() {
            Ok("portable") => Backend::Portable,
            // A request above the cap runs at the cap.
            Ok("avx512") => {
                assert!(
                    available(),
                    "AGG_SIMD=avx512 requires AVX-512F/DQ/BW/VL/IFMA"
                );
                if allowed {
                    Backend::Avx512
                } else {
                    below_ifma()
                }
            }
            Err(std::env::VarError::NotPresent) => {
                if allowed {
                    Backend::Avx512
                } else {
                    below_ifma()
                }
            }
            _ => panic!("AGG_SIMD must be portable or avx512"),
        }
    })
}

/// One line for `AGG_SIMD_REPORT`.
#[must_use]
pub fn report() -> String {
    let requested = std::env::var("AGG_ISA")
        .ok()
        .filter(|value| !value.trim().is_empty());
    format!(
        "agg-simd: cap={} ({}) backend={:?} ifma-cpu={}",
        cap().name(),
        requested.as_deref().unwrap_or("detected"),
        backend(),
        available(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_every_level() {
        assert_eq!(IsaLevel::parse("scalar"), Some(IsaLevel::Scalar));
        assert_eq!(IsaLevel::parse("AVX2"), Some(IsaLevel::Avx2));
        assert_eq!(IsaLevel::parse(" avx512 "), Some(IsaLevel::Avx512));
        assert_eq!(IsaLevel::parse("ifma"), Some(IsaLevel::Avx512Ifma));
        assert_eq!(IsaLevel::parse("avx5"), None);
    }
}
