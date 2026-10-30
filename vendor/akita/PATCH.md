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
digit in the basis range the cache records that the centered mirror itself
serves as the planes and resolves them through the owning polynomial, otherwise
it owns the previous per-ring decomposition of the field coefficients.
`small_i16_ring_coeffs` views the mirror per ring. This build has no SIMD
encoder, so the fixed-prime constructor expands the centered bytes through the
portable `Field::from_i64` path.

The mirror-backed form stores no bytes. With one digit the mirror is already in
ring-major order, so resolving the cache is a live-prefix slice of
`small_i8_coeffs`. This removes the 32 MiB per narrow column copy at 32K, and
therefore 256-384 MiB of resident memory from the pre-fork commit through the
end of the opening, with identical plane bytes. At 8K a narrow column is only
8 KiB, so the change is below the noise of a fresh-process VmHWM; the 1K
aggregate remains byte-identical under scalar and avx512 and under one and
sixteen rayon threads.

Bounds and the signed-16 mirror are now scanned with the conditional
parallelism helpers instead of the previous serial `Option`-state fold and
per-cell `push`. `chunked_min_max` splits a slice into 32K-element chunks,
reduces each with slice `min`/`max` (vectorised), and combines the partials on
the pool; slices of at most one chunk stay serial to avoid nesting overhead on
small groups. `attach_small_i16` first range-checks the cells through the same
reduction, then writes the mirror into `spare_capacity_mut` chunk by chunk, so
the 64 MiB first touch per wide column at 32K is spread over the pool. The
error path still names the first out-of-range cell. Three tests cover minima and
maxima at both ends of two chunks, the signed-16 mirror at both interval ends,
and the rejection of the first out-of-range cell. At 8K a column is 8K cells, so
event 1 is unchanged within noise (adjacent pairs 551/562 vs 566/552 ms); the
lever targets the 2^25 cells per column at 32K.

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

## 3. Concurrent opening passes and a signed-16 probe cache

`sources/dense/kernels.rs`: the packing pass iterates columns in parallel, and
the sparse decompose-fold batch primes every digit cache serially before it
decomposes the columns in parallel; both collect in column order.

`sources/poly_helpers/decompose_fold_partitioned.rs` and its `mod.rs`:
`CachedDigits16` and `cached_i16_decompose_fold_partitioned` run the existing
element-partitioned accumulation over the signed-16 mirror. The challenge plan
and the i16-to-i32 flush bounds are the same as for live rings.

`sources/dense/ops.rs`: the sparse and chunked fold probes read the signed-16
mirror when the source has one and the basis is wider than `i8`, instead of
decomposing the 8 B/cell field plane on every nonce.

`opaque/openings.rs`: `prepare_openings` runs the groups concurrently.

`akita-prover/src/protocol/fold_grind.rs`: challenges are still drawn in group
order on the preview transcript, then the probes of one nonce run concurrently.
A rejected nonce now probes every group instead of stopping at the first, which
is rare at the scheduled grind difficulty.

`akita-prover/src/backend/{opening,scope,traits}.rs`: `OpaqueOpeningKernel` is
`Sync`, and `ProofSessionHandle` and `PreparedOpeningHandle` are `Send + Sync`,
so the parallel tasks can share the backend, the session and the prepared
openings. No arithmetic or wire-format change.

On the IFMA line, 8K local, two adjacent pairs: opening 1161/889 -> 975/716 ms,
total 5208/4752 -> 4969/3767 ms (shared noisy box: direction only). The 1K
aggregate is byte-identical under scalar and avx512 and under one and sixteen
rayon threads.
