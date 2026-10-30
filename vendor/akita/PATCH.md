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

## 2. Commit from the small mirrors

`sources/dense/poly.rs`: `DensePoly` keeps the signed-16 mirror of a width-16
source (`attach_small_i16`) and the inclusive bounds of either mirror.
`prime_digit_cache` builds the ring-major digit planes at construction: for one
digit in the basis range the planes are a copy of the centered mirror, otherwise
the previous per-ring decomposition of the field coefficients.
`small_i16_ring_coeffs` views the mirror per ring. This build has no SIMD
encoder, so `sources/dense/small.rs` expands the centered bytes through the
portable `Field::from_i64` path.

`commitment/source.rs` and `commitment/source/builtins.rs`:
`DenseCoefficientSource` exposes the signed-16 mirror, and
`committed_centered_reach` answers from the mirror bounds when the admission
threshold is the centered half-modulus (the single-digit case), so admission no
longer scans the 8 B/cell plane.

`opaque/plans.rs`, `arithmetic/commitment.rs` and `arithmetic/exact_i16.rs`:
`CoeffBlocks` carries the per-block signed-16 mirrors; a 16-bit single-digit
commit takes `dense_commit_i16_mirror_rows`, which feeds the mirror blocks to the
same exact-i16 `mat_vec_i16` kernel (the portable CRT NTT on this build, the
IFMA52 specialization on the IFMA line) and zero-pads a ragged last block. Both
paths request the same A transform key. With a primed i8 cache the source
advertises `PredecomposedDigits` first, so the commit reads 32 MiB per column
instead of the 256 MiB field plane. The new unit test
`dense_i16_mirror_commit_matches_coefficient_commit` compares both paths on two
rings covering the full signed-16 interval.

On the IFMA line, 8K local, two adjacent alternating pairs: event 1 898/765 ->
620/692 ms, round 2 168/183 -> 100/155 ms, opening 1038/808 -> 736/796 ms,
total 4552/4043 -> 3452/4007 ms (shared noisy box: direction only). The 1K
aggregate is byte-identical under scalar and avx512 and under one and sixteen
rayon threads.
