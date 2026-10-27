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
pub fn drop_in_background<T: Send + 'static>(value: T) {
    if serial_schedule() {
        drop(value);
        return;
    }
    let _ = std::thread::Builder::new()
        .name("agg-drop".into())
        .spawn(move || drop(value));
}
