//! Integration tests of agg-simd in one binary: differential checks against jolt-field,
//! buffer storage, and the ignored kernel timing benchmarks.

mod differential;
mod storage;
mod timing;
