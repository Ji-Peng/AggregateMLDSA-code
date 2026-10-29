//! Scalar implementation of Keccak sponge operations.

use super::{Sink, XofJob};

pub(crate) fn run_xof<S: Sink>(jobs: &[XofJob<'_>], sink: &mut S) {
    let _ = (jobs, sink);
    todo!("T2")
}
