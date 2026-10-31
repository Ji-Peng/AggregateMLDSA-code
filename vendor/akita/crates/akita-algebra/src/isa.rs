//! Cap on the x86 instruction set from `AGG_ISA`.
//!
//! The cap is an upper bound for every run-time feature check in this crate.
//! It is parsed once per process; an unknown value is an error, because a
//! misspelt cap would silently run a different instruction set.

use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Level {
    Scalar,
    Avx2,
    Avx512,
    Avx512Ifma,
}

impl Level {
    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "scalar" => Some(Self::Scalar),
            "avx2" => Some(Self::Avx2),
            "avx512" | "avx-512" => Some(Self::Avx512),
            "avx512ifma" | "avx512-ifma" | "ifma" => Some(Self::Avx512Ifma),
            _ => None,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2",
            Self::Avx512 => "avx512",
            Self::Avx512Ifma => "avx512ifma",
        }
    }
}

/// The process-wide cap; `Avx512Ifma` when `AGG_ISA` is unset.
pub(crate) fn cap() -> Level {
    static CAP: OnceLock<Level> = OnceLock::new();
    *CAP.get_or_init(|| match std::env::var("AGG_ISA") {
        Ok(value) => Level::parse(&value).unwrap_or_else(|| {
            panic!("invalid AGG_ISA '{value}': expected avx512ifma, avx512, avx2 or scalar")
        }),
        Err(_) => Level::Avx512Ifma,
    })
}

/// The `AGG_ISA` value, for reports.
pub(crate) fn requested() -> Option<&'static str> {
    let value = std::env::var("AGG_ISA").ok()?;
    Some(Level::parse(&value)?.name())
}

/// Whether the cap admits the AVX2 tier.
#[inline]
pub(crate) fn avx2() -> bool {
    cap() >= Level::Avx2
}

/// Whether the cap admits AVX-512.
#[inline]
pub(crate) fn avx512() -> bool {
    cap() >= Level::Avx512
}

/// Whether the cap admits AVX-512IFMA.
#[inline]
pub(crate) fn ifma() -> bool {
    cap() >= Level::Avx512Ifma
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_orders_levels() {
        assert_eq!(Level::parse("scalar"), Some(Level::Scalar));
        assert_eq!(Level::parse("AVX2"), Some(Level::Avx2));
        assert_eq!(Level::parse("avx512"), Some(Level::Avx512));
        assert_eq!(Level::parse("avx512ifma"), Some(Level::Avx512Ifma));
        assert_eq!(Level::parse("avx512-vbmi"), None);
        assert!(Level::Scalar < Level::Avx2);
        assert!(Level::Avx2 < Level::Avx512);
        assert!(Level::Avx512 < Level::Avx512Ifma);
    }
}
