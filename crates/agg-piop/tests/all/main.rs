//! Integration tests of `agg-piop`, one module per topic, in one test binary. The tests
//! that change process-wide state (`fork_join`, `single_thread`) stay separate binaries.

#[path = "../common/mod.rs"]
mod common;
mod bridge;
mod commit_open;
mod h_exact;
mod h_fips;
mod h_fips_lanes;
mod keccak_fp;
mod mode_b;
mod projection;
mod simd;
mod prove_verify;
mod source_tables;
mod stages;
mod virtual_checker;
