//! Prover-side scheduling helpers (lever A, docs/perf.md Section 11): take the
//! one-vCPU windows (single-threaded unmaps of buffers of several GiB, the serial
//! public-data copy) off the prover's critical path. Nothing here changes an output.

use std::sync::OnceLock;

/// `true` when `AGG_SCHED_SERIAL` is set (any value): every helper then behaves as
/// before lever A (drops in place, the serial setup step), for A/B runs in one binary.
/// Read once per process.
#[must_use]
pub fn serial_schedule() -> bool {
    static SERIAL: OnceLock<bool> = OnceLock::new();
    *SERIAL.get_or_init(|| std::env::var_os("AGG_SCHED_SERIAL").is_some())
}

/// Drop `value` on a detached thread, so that unmapping its pages (one thread in the
/// kernel for each GiB-sized buffer) runs beside the caller's next step instead of on
/// the critical path. Under [`serial_schedule`], or if no thread can be spawned, the
/// value is dropped here (a failed `spawn` drops the closure, and with it `value`).
/// With `AGG_STAGE_TRACE` set, the time each drop takes is printed under `label`.
pub fn drop_in_background<T: Send + 'static>(label: &'static str, value: T) {
    if serial_schedule() {
        timed_drop(label, "inline", value);
        return;
    }
    let _ = std::thread::Builder::new()
        .name("agg-drop".into())
        .spawn(move || timed_drop(label, "background", value));
}

fn timed_drop<T>(label: &str, place: &str, value: T) {
    static TRACE: OnceLock<bool> = OnceLock::new();
    if *TRACE.get_or_init(|| std::env::var_os("AGG_STAGE_TRACE").is_some()) {
        let start = std::time::Instant::now();
        drop(value);
        eprintln!(
            "drop {label} ({place}): {:.1} ms",
            start.elapsed().as_secs_f64() * 1e3
        );
    } else {
        drop(value);
    }
}
