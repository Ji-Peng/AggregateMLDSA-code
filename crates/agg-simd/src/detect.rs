//! Select the backend once, before processing any slices.
use std::sync::OnceLock;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Portable,
    Avx512,
}
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
pub fn backend() -> Backend {
    static BACKEND: OnceLock<Backend> = OnceLock::new();
    *BACKEND.get_or_init(|| match std::env::var("AGG_SIMD").as_deref() {
        Ok("portable") => Backend::Portable,
        Ok("avx512") => {
            assert!(
                available(),
                "AGG_SIMD=avx512 requires AVX-512F/DQ/BW/VL/IFMA"
            );
            Backend::Avx512
        }
        Err(std::env::VarError::NotPresent) => {
            if available() {
                Backend::Avx512
            } else {
                Backend::Portable
            }
        }
        _ => panic!("AGG_SIMD must be portable or avx512"),
    })
}
