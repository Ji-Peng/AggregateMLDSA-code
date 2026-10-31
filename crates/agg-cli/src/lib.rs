//! Library half of the `agg` command-line tool: the mode-A aggregator and verifier
//! ([`mode_a`]), the mode-A segment items of the wire format ([`wire`]), the persistent key
//! cache ([`cache`]), the batch files ([`record`]), peak-memory readings ([`rss`]) and the
//! subcommands ([`commands`]).

pub mod cache;
pub mod commands;
pub mod mode_a;
pub mod record;
pub mod rss;
pub mod wire;

pub use commands::{parse_command, Command, USAGE};

/// One line per crate with the `AGG_ISA` cap and the kernels it selected.
///
/// The lines come from the same detectors the prover uses, so a run with
/// `AGG_SIMD_REPORT=1` states which instruction set every backend chose.
#[must_use]
pub fn simd_report() -> Vec<String> {
    vec![
        agg_mldsa_simd::dispatch::report(),
        flock_core::simd_report(),
        akita_algebra::simd_report(),
        akita_challenges::simd_report(),
        akita_cpu_backend::simd_report(),
    ]
}
