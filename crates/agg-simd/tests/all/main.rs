//! Integration tests of agg-simd in one binary: input-length checks of the safe entry points,
//! differential checks against jolt-field, buffer storage, and the ignored kernel timing
//! benchmarks.

mod bounds;
mod differential;
mod storage;
mod timing;
