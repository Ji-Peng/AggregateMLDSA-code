//! `agg`: ML-DSA aggregate signatures (ML-DSA-44, 65 and 87), mode A (`docs/spec.md`).
//! Subcommands `keygen-batch`, `sign-batch`, `aggregate`, `verify` and `bench`; run
//! `agg help` for the options.

use std::process::ExitCode;

/// Large tables are backed by transparent huge pages and freed ones are reused
/// (`agg_piop::alloc`). `AGG_HUGEPAGE=0` turns the huge-page advice off,
/// `AGG_ALLOC_CACHE=<GiB>` sets the cache cap (0: no cache),
/// `AGG_ALLOC_SPLIT=0` reuses only cached blocks of up to 125% of a request and
/// `AGG_ALLOC_POISON=1` fills reused blocks with a pattern (a test mode).
#[global_allocator]
static ALLOC: agg_piop::alloc::HugePageAlloc = agg_piop::alloc::HugePageAlloc;

/// Lowers Aerie's runtime-dispatched kernels to the `AGG_ISA` cap.
///
/// Aerie reads `AERIE_SIMD_BACKEND` once per process (its `backend-override`
/// feature, enabled by this workspace); the value can only lower the ceiling.
fn apply_isa_to_aerie() {
    if std::env::var_os("AERIE_SIMD_BACKEND").is_some() {
        return;
    }
    let cap = agg_mldsa_simd::isa::cap();
    if cap.requested().is_none() {
        return;
    }
    let backend = match cap.limit() {
        agg_mldsa_simd::isa::IsaLevel::Scalar => "scalar",
        agg_mldsa_simd::isa::IsaLevel::Avx2 => "avx2",
        agg_mldsa_simd::isa::IsaLevel::Avx512
        | agg_mldsa_simd::isa::IsaLevel::Avx512Ifma => "avx512",
    };
    std::env::set_var("AERIE_SIMD_BACKEND", backend);
}

fn main() -> ExitCode {
    apply_isa_to_aerie();
    if std::env::var_os("AGG_HUGEPAGE").is_some_and(|v| v == "0") {
        agg_piop::alloc::set_enabled(false);
    }
    if let Ok(value) = std::env::var("AGG_ALLOC_CACHE") {
        match value.trim().parse::<usize>() {
            Ok(gib) => agg_piop::alloc::set_cache_bytes(gib.saturating_mul(1 << 30)),
            Err(_) => eprintln!(
                "agg: AGG_ALLOC_CACHE={value:?} is not a whole number of GiB; using the default"
            ),
        }
    }
    if std::env::var_os("AGG_ALLOC_POISON").is_some_and(|v| v == "1") {
        agg_piop::alloc::set_poison(true);
    }
    if std::env::var_os("AGG_ALLOC_SPLIT").is_some_and(|v| v == "0") {
        agg_piop::alloc::set_split(false);
    }
    if std::env::var_os("AGG_SIMD_REPORT").is_some_and(|v| v == "1") {
        for line in agg_cli::simd_report() {
            eprintln!("agg: {line}");
        }
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
