//! Where the Flock opening runs: a per-thread hook between the Keccak side, which reaches
//! the opening (ring switch and Ligerito), and the aggregation prover, which owns the pools.
//!
//! The prover installs an [`OpeningRunner`] on the workers of the Flock pool
//! ([`set_opening_runner`] in the pool's start handler); `agg-keccak` passes each opening to
//! [`run_opening`], which hands it to the runner of the current thread, or runs it in place
//! when there is none. The runner may move the opening to another pool (thread donation at
//! the fork's join, `AGG_SCHED_DONATE`). The opening's parallel loops are exact GF(2^128)
//! arithmetic and its nonce searches return the smallest passing nonce, so the pool that runs
//! it never changes the proof bytes.

use std::cell::RefCell;
use std::sync::Arc;

/// Runs one opening job: called on the thread that reaches the opening, it must call the job
/// exactly once (in place or on another pool) and return after the job has finished.
pub type OpeningRunner = Arc<dyn Fn(&mut (dyn FnMut() + Send)) + Send + Sync>;

thread_local! {
    static RUNNER: RefCell<Option<OpeningRunner>> = const { RefCell::new(None) };
}

/// Install (or with `None` remove) the opening runner of the current thread.
pub fn set_opening_runner(runner: Option<OpeningRunner>) {
    RUNNER.with(|r| *r.borrow_mut() = runner);
}

/// Run the opening `f` through the current thread's runner, or in place without one.
///
/// # Panics
/// If the runner returns without running the job.
pub fn run_opening<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    let Some(runner) = RUNNER.with(|r| r.borrow().clone()) else {
        return f();
    };
    let mut f = Some(f);
    let mut out = None;
    runner(&mut || out = Some((f.take().expect("the opening runs once"))()));
    out.expect("the opening runner ran the opening")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn the_runner_of_the_current_thread_runs_the_opening() {
        assert_eq!(run_opening(|| 3), 3);
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&calls);
        set_opening_runner(Some(Arc::new(move |job: &mut (dyn FnMut() + Send)| {
            seen.fetch_add(1, Ordering::Relaxed);
            std::thread::scope(|s| {
                s.spawn(|| job());
            });
        })));
        assert_eq!(run_opening(|| 5), 5);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        // Another thread has no runner.
        std::thread::spawn(|| assert_eq!(run_opening(|| 7), 7)).join().unwrap();
        set_opening_runner(None);
        assert_eq!(run_opening(|| 9), 9);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }
}
