//! AVX2 polynomial transforms, with scalar compilation on other architectures.
//! The x86-64 kernels derive from pq-crystals Dilithium (CC0/Apache-2.0).
use super::{Form, Scale};
#[cfg(target_arch = "x86_64")]
#[path = "avx2_x86.rs"]
mod implementation;
#[cfg(not(target_arch = "x86_64"))]
pub(crate) use super::scalar::{dot4, intt, ntt, order_from_fips, order_to_fips, reduce};
#[cfg(target_arch = "x86_64")]
pub(crate) use implementation::{dot4, intt, ntt, order_from_fips, order_to_fips, reduce};
