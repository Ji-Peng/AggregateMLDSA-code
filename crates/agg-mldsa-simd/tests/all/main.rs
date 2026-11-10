//! All integration tests of the crate in one test binary (one link step).

// The kernels included from the library source name `crate::params` and `crate::poly`.
#[allow(unused_imports)]
use agg_mldsa_simd::{params, poly};

mod common;

mod backends;
mod finish_simd;
mod keccak_simd;
mod ntt_avx2;
mod ntt_avx512;
mod sample_simd;
mod scalar;
