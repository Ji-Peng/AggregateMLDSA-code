//! Prover-side scheduling helpers: take the
//! one-vCPU windows (single-threaded unmaps of buffers of several GiB, the serial
//! public-data copy) off the prover's critical path. Nothing here changes an output.

use std::sync::{Condvar, Mutex, OnceLock};

/// Background drop threads currently running.
static ACTIVE: Mutex<usize> = Mutex::new(0);
/// Signalled when [`ACTIVE`] reaches zero.
static IDLE: Condvar = Condvar::new();

/// Counts a background drop for the lifetime of its thread; dropping it on a failed spawn
/// keeps the count exact.
struct Active;

impl Active {
    fn new() -> Self {
        *ACTIVE.lock().unwrap_or_else(|e| e.into_inner()) += 1;
        Self
    }
}

impl Drop for Active {
    fn drop(&mut self) {
        let mut active = ACTIVE.lock().unwrap_or_else(|e| e.into_inner());
        *active -= 1;
        if *active == 0 {
            IDLE.notify_all();
        }
    }
}

/// Drop `value` on a detached thread, so that unmapping its pages (one thread in the
/// kernel for each GiB-sized buffer) runs beside the caller's next step instead of on
/// the critical path. If no thread can be spawned, the value is dropped here (a failed
/// `spawn` drops the closure, and with it `value`). With `AGG_STAGE_TRACE` set, the time
/// each drop takes is printed under `label`.
pub fn drop_in_background<T: Send + 'static>(label: &'static str, value: T) {
    let active = Active::new();
    let _ = std::thread::Builder::new()
        .name("agg-drop".into())
        .spawn(move || {
            let _active = active;
            timed_drop(label, "background", value);
        });
}

/// Wait until every background drop started so far has finished. A benchmark calls this
/// before releasing the allocator cache, so the next timed run starts cold.
pub fn wait_background_drops() {
    let mut active = ACTIVE.lock().unwrap_or_else(|e| e.into_inner());
    while *active > 0 {
        active = IDLE.wait(active).unwrap_or_else(|e| e.into_inner());
    }
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
