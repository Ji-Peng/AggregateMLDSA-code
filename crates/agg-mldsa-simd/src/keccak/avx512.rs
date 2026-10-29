//! AVX-512 implementation of Keccak sponge operations.

use super::{Sink, XofJob};

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
pub(crate) unsafe fn run_xof<S: Sink>(jobs: &[XofJob<'_>], sink: &mut S) {
    super::scalar::run_xof(jobs, sink)
}
