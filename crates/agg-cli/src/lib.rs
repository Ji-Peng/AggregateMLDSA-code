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

use std::io::{self, Read as _, Write as _};
use std::path::Path;

/// Largest aggregate file `verify` reads: 64 MiB.
///
/// The largest honest aggregate (ML-DSA-87, `N = 2^15`, any profile and relation) is a few
/// MiB, under 150 B per signature; at the 1K minimum of ML-DSA-87 it is under 0.5 MiB. The
/// cap allows 2 KiB per signature at `2^15` records, so it rejects no honest file and keeps
/// an oversized input from being read into memory. Record, key and cache files are the
/// verifier's own inputs and have no cap (record messages may be of any length).
pub const MAX_AGGREGATE_BYTES: u64 = 64 << 20;

/// Read a whole input file. Errors name the file; with `cap`, a file longer than `cap`
/// bytes is refused before more than `cap + 1` bytes are read.
///
/// # Errors
/// On an unreadable file (the error keeps the I/O kind, so a missing file stays
/// [`io::ErrorKind::NotFound`]) or a file longer than `cap`.
pub fn read_input(path: &Path, cap: Option<u64>) -> io::Result<Vec<u8>> {
    let named = |e: io::Error| io::Error::new(e.kind(), format!("{}: {e}", path.display()));
    let file = std::fs::File::open(path).map_err(named)?;
    let limit = cap.unwrap_or(u64::MAX);
    let hint = file.metadata().map_or(0, |m| m.len()).min(limit);
    let mut bytes = Vec::with_capacity(usize::try_from(hint).unwrap_or(0));
    file.take(limit.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(named)?;
    if bytes.len() as u64 > limit {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{}: longer than the {limit} B limit", path.display()),
        ));
    }
    Ok(bytes)
}

/// Write one line to standard output. A closed pipe (`agg verify ... | head -1`) is not an
/// error: the command goes on and returns its own exit status.
///
/// # Errors
/// On any other write failure.
pub fn write_stdout_line(args: std::fmt::Arguments<'_>) -> io::Result<()> {
    match writeln!(io::stdout().lock(), "{args}") {
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => Ok(()),
        result => result,
    }
}

/// One line per crate with the `AGG_ISA` cap and the kernels it selected.
///
/// The lines come from the same detectors the prover uses, so a run with
/// `AGG_SIMD_REPORT=1` states which instruction set every backend chose.
#[must_use]
pub fn simd_report() -> Vec<String> {
    vec![
        agg_mldsa_simd::dispatch::report(),
        agg_simd::detect::report(),
        flock_core::simd_report(),
        akita_algebra::simd_report(),
        akita_challenges::simd_report(),
        akita_cpu_backend::simd_report(),
        format!("aerie-simd: backend={}", agg_piop::aerie_simd_backend()),
    ]
}
