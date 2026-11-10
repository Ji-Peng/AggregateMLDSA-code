//! Integration tests of `agg-piop` that change process-wide state: environment knobs
//! (`AGG_FLOCK_THREADS`, `AGG_LATE_JOIN`, `AGG_PIOP_FORK`, `AGG_PIOP_S_THREADS`) and the
//! lazy-field-plane counter. They run one after another in one test function, so no other
//! test of this binary runs beside them.

#[path = "../common/mod.rs"]
mod common;
mod fork_join;
mod lazy_field_plane;
mod piop_fork;

fn on_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("test thread");
}

#[test]
fn process_state() {
    fork_join::run();
    lazy_field_plane::run();
    piop_fork::run();
}
