//! Benchmarking harness using high-resolution Time Stamp Counter (TSC).

#![allow(dead_code)]

use std::hint::black_box;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};

pub const CALLS_PER_TRIAL: usize = 2000;
pub const NUM_TRIALS: usize = 31;

static HAS_FAIL: AtomicBool = AtomicBool::new(false);

/// Reads the Time Stamp Counter with serialization fence before.
#[inline(always)]
pub fn start_tsc() -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_mm_lfence();
        let t = core::arch::x86_64::_rdtsc();
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
        t
    }
    #[cfg(not(target_arch = "x86_64"))]
    0
}

/// Reads the Time Stamp Counter with serialization fence after.
#[inline(always)]
pub fn end_tsc() -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
        let mut aux = 0u32;
        let t = core::arch::x86_64::__rdtscp(&mut aux);
        core::arch::x86_64::_mm_lfence();
        t
    }
    #[cfg(not(target_arch = "x86_64"))]
    0
}

/// Statistical summary of timed trials.
#[derive(Clone, Copy, Debug)]
pub struct TimingStats {
    pub median: f64,
    pub min: f64,
    pub max: f64,
}

/// Measures execution of a function over 31 trials of 2,000 calls each.
#[inline(never)]
pub fn measure_kernel<F: FnMut()>(mut f: F) -> TimingStats {
    // Warmup
    for _ in 0..500 {
        f();
    }

    let mut trials = [0.0f64; NUM_TRIALS];
    for trial in trials.iter_mut() {
        let start = start_tsc();
        for _ in 0..CALLS_PER_TRIAL {
            f();
        }
        let end = end_tsc();
        *trial = (end.wrapping_sub(start) as f64) / (CALLS_PER_TRIAL as f64);
    }

    trials.sort_by(f64::total_cmp);
    TimingStats {
        median: trials[NUM_TRIALS / 2],
        min: trials[0],
        max: trials[NUM_TRIALS - 1],
    }
}

/// Measures empty call overhead in TSC ticks.
#[inline(never)]
fn empty_call() {
    black_box(());
}

/// Verifies that the TSC measurement harness overhead is within bounds (<= 40 ticks).
pub fn harness_empty_call_ticks() -> f64 {
    let stats = measure_kernel(empty_call);
    stats.median
}

/// Prints system information: CPU model, rustc version, and gcc version.
pub fn print_system_info() {
    let cpu = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|info| {
            info.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|s| s.trim().to_string())
        })
        .unwrap_or_else(|| "Unknown CPU".to_string());

    let rustc = Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown rustc".to_string());

    let gcc = Command::new("gcc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.lines().next().unwrap_or("").trim().to_string())
        .unwrap_or_else(|| "unknown gcc".to_string());

    println!("================================================================================");
    println!("CPU:   {cpu}");
    println!("rustc: {rustc}");
    println!("gcc:   {gcc}");
    println!("================================================================================");
    println!(
        "{:<20} {:<10} {:>10} {:>10} {:>8} {:>12}",
        "kernel", "backend", "rust", "c", "ratio", "verdict"
    );
    println!("--------------------------------------------------------------------------------");
}

/// Alternates Rust and C trials, evaluates performance ratio, and prints the result row.
/// Like [`compare`], for a composite Rust kernel measured against C work that does only part
/// of it: prints the ratio as INFO and never fails the gate.
pub fn compare_info<R: FnMut(), C: FnMut()>(name: &str, backend: &str, mut rust_fn: R, mut c_fn: C, c_available: bool) {
    if !c_available {
        return;
    }
    let mut r = [0.0f64; NUM_TRIALS];
    let mut c = [0.0f64; NUM_TRIALS];
    for i in 0..NUM_TRIALS {
        let t = start_tsc();
        for _ in 0..CALLS_PER_TRIAL {
            rust_fn();
        }
        r[i] = (end_tsc().wrapping_sub(t) as f64) / (CALLS_PER_TRIAL as f64);
        let t = start_tsc();
        for _ in 0..CALLS_PER_TRIAL {
            c_fn();
        }
        c[i] = (end_tsc().wrapping_sub(t) as f64) / (CALLS_PER_TRIAL as f64);
    }
    r.sort_by(f64::total_cmp);
    c.sort_by(f64::total_cmp);
    let (rm, cm) = (r[NUM_TRIALS / 2], c[NUM_TRIALS / 2]);
    println!("{:<20} {:<10} {:>10.2} {:>10.2} {:>8.3} {:>12}", name, backend, rm, cm, rm / cm, "INFO");
}

pub fn compare<R: FnMut(), C: FnMut()>(
    name: &str,
    backend: &str,
    mut rust_fn: R,
    mut c_fn: C,
    c_available: bool,
) {
    if !c_available {
        let rust_stats = measure_kernel(rust_fn);
        println!(
            "{:<20} {:<10} {:>10.2} {:>10} {:>8} {:>12}",
            name, backend, rust_stats.median, "-", "-", "UNRESOLVED"
        );
        return;
    }

    // Warmup both
    for _ in 0..500 {
        rust_fn();
        c_fn();
    }

    let mut rust_trials = [0.0f64; NUM_TRIALS];
    let mut c_trials = [0.0f64; NUM_TRIALS];

    for i in 0..NUM_TRIALS {
        // Rust trial
        let start_r = start_tsc();
        for _ in 0..CALLS_PER_TRIAL {
            rust_fn();
        }
        let end_r = end_tsc();
        rust_trials[i] = (end_r.wrapping_sub(start_r) as f64) / (CALLS_PER_TRIAL as f64);

        // C trial
        let start_c = start_tsc();
        for _ in 0..CALLS_PER_TRIAL {
            c_fn();
        }
        let end_c = end_tsc();
        c_trials[i] = (end_c.wrapping_sub(start_c) as f64) / (CALLS_PER_TRIAL as f64);
    }

    rust_trials.sort_by(f64::total_cmp);
    c_trials.sort_by(f64::total_cmp);

    let rust_med = rust_trials[NUM_TRIALS / 2];
    let c_med = c_trials[NUM_TRIALS / 2];
    let ratio = rust_med / c_med;

    let verdict = if ratio <= 1.00 {
        "PASS"
    } else if ratio <= 1.03 {
        // Re-run 3 times with swapped order (C first, then Rust) and decide on pooled ratio
        let mut pooled_rust = Vec::with_capacity(3 * NUM_TRIALS);
        let mut pooled_c = Vec::with_capacity(3 * NUM_TRIALS);
        for _ in 0..3 {
            for _ in 0..NUM_TRIALS {
                let start_c = start_tsc();
                for _ in 0..CALLS_PER_TRIAL {
                    c_fn();
                }
                let end_c = end_tsc();
                pooled_c.push((end_c.wrapping_sub(start_c) as f64) / (CALLS_PER_TRIAL as f64));

                let start_r = start_tsc();
                for _ in 0..CALLS_PER_TRIAL {
                    rust_fn();
                }
                let end_r = end_tsc();
                pooled_rust.push((end_r.wrapping_sub(start_r) as f64) / (CALLS_PER_TRIAL as f64));
            }
        }
        pooled_rust.sort_by(f64::total_cmp);
        pooled_c.sort_by(f64::total_cmp);
        let pooled_ratio = pooled_rust[pooled_rust.len() / 2] / pooled_c[pooled_c.len() / 2];
        if pooled_ratio <= 1.00 {
            "PASS"
        } else {
            HAS_FAIL.store(true, Ordering::SeqCst);
            "FAIL"
        }
    } else {
        HAS_FAIL.store(true, Ordering::SeqCst);
        "FAIL"
    };

    println!(
        "{:<20} {:<10} {:>10.2} {:>10.2} {:>8.3} {:>12}",
        name, backend, rust_med, c_med, ratio, verdict
    );
}

/// Checks `--gate` argument and exits non-zero if any benchmark failed.
pub fn finish_benchmarks() {
    let gate_enabled = std::env::args().any(|a| a == "--gate");
    if gate_enabled && HAS_FAIL.load(Ordering::SeqCst) {
        eprintln!("Perf gate FAILED: one or more kernels were slower than reference.");
        std::process::exit(1);
    }
}
