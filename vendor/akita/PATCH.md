# Akita vendor record

Upstream: https://github.com/LayerZero-Labs/akita at `e023b6a46143f7b23199ccecb2684192baf7b640`. The `third_party/lattice-estimator` submodule is copied at `c667a48546f140c3a5454c7503c3ca44a264cce2`.

## 1. Prewarm a root commit's NTT slots

`crates/akita-cpu-backend/src/commitment/executor.rs`: move the slot installation
out of `prepare_request_resources` into `ensure_requirement`, and add
`prewarm_plan_resources`, which installs the A slot of both dense representations
and the B slot of a root plan without materializing a source.
`crates/akita-cpu-backend/src/arithmetic/controls.rs`: add
`CpuBackend::prewarm_commit(profile)`, which builds the executor for
`CommitmentExecutionPlan::for_root` and calls that method. No arithmetic, wire
format or verifier check changes: the commit finds the same slots and skips only
their lazy `OnceLock` build.

Callers are outside this copy: `PcsProver::prewarm_groups` (`agg-pcs-config`) and
`TableCommitter::warm` (`agg-piop`) run it for all six groups on the `agg-setup`
thread. On the IFMA line, 4K local, adjacent alternating pairs: round 2
268/194 -> 167/142 ms, event 1 687/576 -> 478/417 ms, total 3772/3280 ->
3181/2667 ms (shared, noisy box: direction only). The 1K aggregate is
byte-identical under scalar and avx512 and under one and sixteen rayon threads.
