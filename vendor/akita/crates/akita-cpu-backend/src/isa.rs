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
    *CAP.get_or_init(|| {
        // An empty value is treated like an unset variable.
        match std::env::var("AGG_ISA")
            .ok()
            .filter(|value| !value.trim().is_empty())
        {
            Some(value) => Level::parse(&value).unwrap_or_else(|| {
                panic!("invalid AGG_ISA '{value}': expected avx512ifma, avx512, avx2 or scalar")
            }),
            None => Level::Avx512Ifma,
        }
    })
}

/// The `AGG_ISA` value, for reports.
pub(crate) fn requested() -> Option<&'static str> {
    // An empty value is treated like an unset variable.
    let value = std::env::var("AGG_ISA")
        .ok()
        .filter(|value| !value.trim().is_empty())?;
    Some(Level::parse(&value)?.name())
}

/// Whether the cap admits the AVX2 tier.
#[inline]
pub(crate) fn avx2() -> bool {
    cap() >= Level::Avx2
}
