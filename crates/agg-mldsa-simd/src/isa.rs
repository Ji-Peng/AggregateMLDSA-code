//! Process-wide x86 instruction-set cap from `AGG_ISA`.
//!
//! `AGG_ISA=avx512ifma|avx512|avx2|scalar` lowers the highest instruction-set
//! level every run-time detector in the process may choose. An unset variable
//! leaves the choice to feature detection. A cap never raises a level: a
//! request above the cap runs at the cap, and a request below it still needs
//! the hardware features of the level it names.
//!
//! `scalar` disables the vector kernels; `avx2` means x86-64-v3 plus
//! PCLMULQDQ, with the 256-bit VPCLMULQDQ and GFNI extensions used when the
//! CPU has them; `avx512` allows every AVX-512 feature except IFMA; and
//! `avx512ifma` allows all detected features.

use std::sync::OnceLock;

/// Instruction-set levels, ordered from the portable baseline upwards.
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
    /// Parses a level name; `None` for an unknown name.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "scalar" => Some(Self::Scalar),
            "avx2" => Some(Self::Avx2),
            "avx512" | "avx-512" => Some(Self::Avx512),
            "avx512ifma" | "avx512-ifma" | "ifma" => Some(Self::Avx512Ifma),
            _ => None,
        }
    }

    /// The name used in reports and error messages.
    pub fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2",
            Self::Avx512 => "avx512",
            Self::Avx512Ifma => "avx512ifma",
        }
    }
}

/// The `AGG_ISA` cap of this process.
#[derive(Clone, Copy, Debug)]
pub struct IsaCap {
    limit: IsaLevel,
    requested: Option<&'static str>,
}

impl IsaCap {
    /// The highest level this process may run.
    pub fn limit(self) -> IsaLevel {
        self.limit
    }

    /// The `AGG_ISA` value this cap came from, if it was set.
    pub fn requested(self) -> Option<&'static str> {
        self.requested
    }

    /// Whether the cap admits `level`.
    pub fn allows(self, level: IsaLevel) -> bool {
        self.limit >= level
    }
}

/// Returns the process-wide cap, parsed once from `AGG_ISA`.
///
/// # Panics
/// Panics when `AGG_ISA` names an unknown level: a misspelt cap would silently
/// run a different instruction set.
pub fn cap() -> IsaCap {
    static CAP: OnceLock<IsaCap> = OnceLock::new();
    *CAP.get_or_init(|| {
        // An empty value is treated like an unset variable.
        let Some(value) = std::env::var("AGG_ISA")
            .ok()
            .filter(|value| !value.trim().is_empty())
        else {
            return IsaCap {
                limit: IsaLevel::Avx512Ifma,
                requested: None,
            };
        };
        let level = IsaLevel::parse(&value).unwrap_or_else(|| {
            panic!(
                "invalid AGG_ISA '{value}': expected avx512ifma, avx512, avx2 or scalar"
            )
        });
        IsaCap {
            limit: level,
            requested: Some(level.name()),
        }
    })
}

/// Whether the cap admits `level` regardless of hardware support.
pub fn allows(level: IsaLevel) -> bool {
    cap().allows(level)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_every_level() {
        assert_eq!(IsaLevel::parse("scalar"), Some(IsaLevel::Scalar));
        assert_eq!(IsaLevel::parse("AVX2"), Some(IsaLevel::Avx2));
        assert_eq!(IsaLevel::parse(" avx512 "), Some(IsaLevel::Avx512));
        assert_eq!(IsaLevel::parse("avx512-ifma"), Some(IsaLevel::Avx512Ifma));
        assert_eq!(IsaLevel::parse("avx512ifma"), Some(IsaLevel::Avx512Ifma));
        assert_eq!(IsaLevel::parse("avx5"), None);
    }

    #[test]
    fn levels_are_ordered() {
        assert!(IsaLevel::Scalar < IsaLevel::Avx2);
        assert!(IsaLevel::Avx2 < IsaLevel::Avx512);
        assert!(IsaLevel::Avx512 < IsaLevel::Avx512Ifma);
    }

    #[test]
    fn cap_only_lowers() {
        let avx2 = IsaCap {
            limit: IsaLevel::Avx2,
            requested: Some("avx2"),
        };
        assert!(avx2.allows(IsaLevel::Scalar));
        assert!(avx2.allows(IsaLevel::Avx2));
        assert!(!avx2.allows(IsaLevel::Avx512));
        assert!(!avx2.allows(IsaLevel::Avx512Ifma));
    }
}
