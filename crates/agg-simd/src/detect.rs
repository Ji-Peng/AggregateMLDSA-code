//! Select the backend once, before processing any slices.
//!
//! `AGG_ISA` is the process-wide instruction-set cap shared by every backend
//! of the prover; `AGG_SIMD` names a backend below that cap. The IFMA kernels
//! need the full `avx512ifma` level, so `AGG_ISA=avx512` (or lower) selects
//! the portable kernels here even on a CPU that has IFMA.

use std::sync::OnceLock;

/// The backends of this crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Portable arithmetic.
    Portable,
    /// AVX-512 IFMA.
    Avx512,
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
                    Backend::Portable
                }
            }
            Err(std::env::VarError::NotPresent) => {
                if allowed {
                    Backend::Avx512
                } else {
                    Backend::Portable
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
