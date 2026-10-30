//! `agg`: ML-DSA-44 aggregate signatures, mode A (`docs/spec.md`).
//! Subcommands `keygen-batch`, `sign-batch`, `aggregate`, `verify` and `bench`; run
//! `agg help` for the options.

use std::process::ExitCode;

/// Large tables are backed by transparent huge pages; `AGG_HUGEPAGE=0` turns
/// this off.
#[global_allocator]
static ALLOC: agg_piop::alloc::HugePageAlloc = agg_piop::alloc::HugePageAlloc;

fn main() -> ExitCode {
    if std::env::var_os("AGG_HUGEPAGE").is_some_and(|v| v == "0") {
        agg_piop::alloc::set_enabled(false);
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
