//! `agg`: ML-DSA-44 aggregate signatures, mode A (`docs/spec.md`).
//! Subcommands `keygen-batch`, `sign-batch`, `aggregate`, `verify` and `bench`; run
//! `agg help` for the options.

use std::process::ExitCode;

/// Large tables are backed by transparent huge pages and freed ones are reused
/// (`agg_piop::alloc`). `AGG_HUGEPAGE=0` turns the huge-page advice off,
/// `AGG_ALLOC_CACHE=<GiB>` sets the cache cap (0: no cache) and
/// `AGG_ALLOC_POISON=1` fills reused blocks with a pattern (a test mode).
#[global_allocator]
static ALLOC: agg_piop::alloc::HugePageAlloc = agg_piop::alloc::HugePageAlloc;

fn main() -> ExitCode {
    if std::env::var_os("AGG_HUGEPAGE").is_some_and(|v| v == "0") {
        agg_piop::alloc::set_enabled(false);
    }
    if let Some(gib) = std::env::var("AGG_ALLOC_CACHE").ok().and_then(|v| v.parse::<usize>().ok()) {
        agg_piop::alloc::set_cache_bytes(gib << 30);
    }
    if std::env::var_os("AGG_ALLOC_POISON").is_some_and(|v| v == "1") {
        agg_piop::alloc::set_poison(true);
    }
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(command) = agg_cli::parse_command(args.first().map(String::as_str)) else {
        eprintln!("{}", agg_cli::USAGE);
        return ExitCode::from(2);
    };
    match agg_cli::commands::run(command, &args[1..]) {
        Ok(status) => ExitCode::from(status),
        Err(e) => {
            eprintln!("agg: {e}");
            ExitCode::from(2)
        }
    }
}
