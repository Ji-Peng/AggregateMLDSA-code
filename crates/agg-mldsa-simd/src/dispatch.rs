//! Hardware backend dispatch and engine initialization.

use std::sync::OnceLock;

/// Available implementation backends.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Backend {
    /// Scalar reference implementation.
    Scalar,
    /// AVX2 implementation (requires AVX2 and POPCNT).
    Avx2,
    /// AVX-512 implementation (requires AVX-512F, BW, VL, AVX2, and POPCNT).
    Avx512,
}

/// Execution engine holding feature detection proofs.
///
/// Holding an `Engine` guarantees that the corresponding hardware features
/// are available on the current machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Engine {
    backend: Backend,
    vbmi: bool,
}

impl Engine {
    /// Creates a new engine instance for the requested backend if supported by hardware.
    pub fn new(b: Backend) -> Option<Engine> {
        match b {
            Backend::Scalar => Some(Engine {
                backend: Backend::Scalar,
                vbmi: false,
            }),
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2 => {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("popcnt") {
                    Some(Engine {
                        backend: Backend::Avx2,
                        vbmi: false,
                    })
                } else {
                    None
                }
            }
            #[cfg(target_arch = "x86_64")]
            Backend::Avx512 => {
                if is_x86_feature_detected!("avx512f")
                    && is_x86_feature_detected!("avx512bw")
                    && is_x86_feature_detected!("avx512vl")
                    && is_x86_feature_detected!("avx2")
                    && is_x86_feature_detected!("popcnt")
                {
                    let vbmi = is_x86_feature_detected!("avx512vbmi");
                    Some(Engine {
                        backend: Backend::Avx512,
                        vbmi,
                    })
                } else {
                    None
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            Backend::Avx2 | Backend::Avx512 => None,
        }
    }

    /// Returns the active backend for this engine.
    #[inline(always)]
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Returns whether AVX-512 VBMI instructions are supported.
    #[inline(always)]
    pub fn has_vbmi(&self) -> bool {
        self.vbmi
    }
}

/// Checks whether a given backend is supported on the current CPU.
pub fn available(b: Backend) -> bool {
    Engine::new(b).is_some()
}

/// Parses a backend name from a string.
///
/// Returns `Ok(None)` for "auto", `Ok(Some(backend))` for specific backends,
/// or an error message for unknown backends.
pub fn parse_backend(s: &str) -> Result<Option<Backend>, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(None),
        "scalar" => Ok(Some(Backend::Scalar)),
        "avx2" => Ok(Some(Backend::Avx2)),
        "avx512" | "avx-512" => Ok(Some(Backend::Avx512)),
        other => Err(format!(
            "invalid backend: expected auto, scalar, avx2, or avx512; got '{other}'"
        )),
    }
}

/// Returns the globally configured engine according to AGG_MLDSA_SIMD.
pub fn engine() -> &'static Engine {
    static ENGINE: OnceLock<Engine> = OnceLock::new();
    ENGINE.get_or_init(|| {
        let env_val = std::env::var("AGG_MLDSA_SIMD").unwrap_or_else(|_| "auto".to_string());
        let selection = parse_backend(&env_val).unwrap_or_else(|err| {
            panic!("Invalid AGG_MLDSA_SIMD environment variable: {err}");
        });
        match selection {
            None => {
                #[cfg(target_arch = "x86_64")]
                {
                    if let Some(e) = Engine::new(Backend::Avx512) {
                        return e;
                    }
                    if let Some(e) = Engine::new(Backend::Avx2) {
                        return e;
                    }
                }
                Engine::new(Backend::Scalar).expect("Scalar backend is always available")
            }
            Some(backend) => Engine::new(backend).unwrap_or_else(|| {
                panic!("AGG_MLDSA_SIMD requested backend {backend:?}, but required CPU features are not available");
            }),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_backend() {
        assert_eq!(parse_backend("auto"), Ok(None));
        assert_eq!(parse_backend("AUTO"), Ok(None));
        assert_eq!(parse_backend("  auto  "), Ok(None));

        assert_eq!(parse_backend("scalar"), Ok(Some(Backend::Scalar)));
        assert_eq!(parse_backend("SCALAR"), Ok(Some(Backend::Scalar)));

        assert_eq!(parse_backend("avx2"), Ok(Some(Backend::Avx2)));
        assert_eq!(parse_backend("AVX2"), Ok(Some(Backend::Avx2)));

        assert_eq!(parse_backend("avx512"), Ok(Some(Backend::Avx512)));
        assert_eq!(parse_backend("avx-512"), Ok(Some(Backend::Avx512)));

        assert!(parse_backend("unknown").is_err());
        assert!(parse_backend("neon").is_err());
        assert!(parse_backend("").is_err());
    }

    #[test]
    fn test_scalar_engine_always_available() {
        assert!(available(Backend::Scalar));
        let eng = Engine::new(Backend::Scalar).unwrap();
        assert_eq!(eng.backend(), Backend::Scalar);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_dispatch_engine() {
        let eng = engine();
        assert!(matches!(
            eng.backend(),
            Backend::Scalar | Backend::Avx2 | Backend::Avx512
        ));
    }
}
