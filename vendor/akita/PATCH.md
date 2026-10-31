# Akita vendor record

Upstream: https://github.com/LayerZero-Labs/akita at
`e023b6a46143f7b23199ccecb2684192baf7b640`.

## 0. Pruned trees

This copy keeps only the files the workspace builds, tests or benches read. Removed:
`book/`, `third_party/` (including the `lattice-estimator` submodule copy),
`.github/`, `docs/`, `bench-data/`, `profile/`, `fuzz/`, `hooks/` and `.gitmodules`;
under `scripts/` everything except `sis_golden/`; under `artifacts/` the
`schedule-catalog*.tsv` snapshots; and under `specs/` everything except
`evidence/subring-coefficient-packing/`. The kept files are those read at build or
test time: `scripts/sis_golden/*.csv` by the `akita-sis-estimator` golden tests and
benches, `artifacts/schedules*/` by the trusted-catalog tests,
`fixtures/jolt-field-cutover/` by the serialization fixtures, and
`specs/evidence/subring-coefficient-packing/` by the planner's catalog snapshot
test. The `exclude` entries for `profile/akita-recursion` and `fuzz` are gone with
those trees. Every removed file is unchanged upstream content and is restorable
from https://github.com/LayerZero-Labs/akita at `e023b6a`.

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
`TableCommitter::warm` (`agg-piop`) run it for all seven groups on the `agg-setup`
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

`ops.rs` uses the mirror-backed fold only when `num_digits == 1` and
`log_basis == MAX_I16_LOG_BASIS`, and `dense_commit_i16_mirror_rows` asserts the same. The
mirror stores one balanced digit per coefficient, so it is exact only at the full signed-16
basis; a narrower basis inside the `i16` range would fold an unbalanced digit and disagree
with `balanced_ring_decompose_fold_chunked`, so it falls back to the balanced decomposition.

Bounds and the signed-16 mirror are now scanned with the conditional
parallelism helpers instead of the previous serial `Option`-state fold and
per-cell `push`. `chunked_min_max` splits a slice into 32K-element chunks,
reduces each in one pass with a copied two-accumulator fold (slice `min`/`max`
compile to a scalar select loop; the fold vectorises), and combines the partials on
the pool; slices of at most one chunk stay serial to avoid nesting overhead on
small groups. `attach_small_i16` first range-checks the cells through the same
reduction, then writes the mirror into `spare_capacity_mut` chunk by chunk, so
the 64 MiB first touch per wide column at 32K is spread over the pool. The
error path still names the first out-of-range cell. Three tests cover minima and
maxima at both ends of two chunks, the signed-16 mirror at both interval ends,
and the rejection of the first out-of-range cell (`sources/dense/tests.rs`;
`sources/dense/mod.rs` declares the test module). At 8K a column is 8K cells, so
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
rings covering the full signed-16 interval (`arithmetic/exact_i16_tests.rs`).

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
is rare at the scheduled grind difficulty. The parallel result is reconsidered
in group order: a group's rejection wins over another group's error, and an
error is reported only when no group rejected the nonce.

`akita-prover/src/backend/{opening,scope,traits}.rs`: `OpaqueOpeningKernel` is
`Sync`, and `ProofSessionHandle` and `PreparedOpeningHandle` are `Send + Sync`,
so the parallel tasks can share the backend, the session and the prepared
openings. No arithmetic or wire-format change.

On the IFMA line, 8K local, two adjacent pairs: opening 1161/889 -> 975/716 ms,
total 5208/4752 -> 4969/3767 ms (shared noisy box: direction only). The 1K
aggregate is byte-identical under scalar and avx512 and under one and sixteen
rayon threads.

## 4. Construct dense columns in chunked parallel writers

`sources/dense/small.rs`: `expand` allocates the field plane once at its final
length and fills it in chunks of 32K cells on the caller's pool; tables of at
most one chunk stay serial. This build has no SIMD encoder, so the scalar
`Field::from_i64` kernel runs per chunk.

`sources/dense/poly.rs`: `from_small_i8_cells` takes the logical centered bytes,
writes the physical-length mirror from them in the same chunked manner, and then
builds the plane through `expand`. It removes the caller's `to_vec` plus
`resize` copy, which reallocated because the copy's capacity equalled its
logical length, and spreads the 32 MiB i8 copy and its page faults over the
pool. `from_small_i8` keeps its exact-length contract and shares the common
finish.

`from_wide_field_evals` takes the already encoded field evaluations and the
width-16 cells together. The cells of this path are known to lie outside the
centered `i8` range, so the small-`i8` probe of `from_field_evals` is skipped
and the polynomial starts without an i8 mirror; the signed-16 mirror is then
attached as before. In both constructors the bounds are taken over exactly the
same cells as in the previous code. Outside this copy, `agg-piop`'s wide
column encoder (`kernel::buf::fill_par`) fills the field plane in chunks on
the pool as well, so the wide column's 256 MiB plane is first touched by the
writers.

The 1K aggregate is byte-identical under scalar and avx512 and under one and
sixteen rayon threads; the constructors are not otherwise timed on this box.

`attach_small_i16` checks the signed-16 range and writes the mirror in one pass
instead of a min/max reduction followed by a second write; each chunk reports
its range and its first offending cell, and the chunks combine in order, so the
error still names the first out-of-range cell.

## 5. Split a batched opening into prepare and finish

`akita-prover`: `prove_root` is now `prepare_root` plus `finish_root`.
`prepare_root` resolves the root claims, validates every commitment and builds
the point-only part of the root fold: the opening plans, `prepare_openings`
(coefficient packing), the per-group digit material and the public relation
payload. `finish_root` replays the public commitment, point, opening and
payload messages on the session transcript, draws the fold grind challenges,
finishes the recursive witness and runs `prove_fold`.

`prepare_fold` is split the same way: `prepare_fold_part` consumes the claims
and commitment material without a transcript, and `finish_fold_part` absorbs
the prepared messages and finishes the relation. `RingRelationProver` gains
`begin` and `finish` around the previous monolithic `prepare`;
`append_opening_messages` is the transcript-only part of
`ProverOpeningData::append_to`.

`BackendRegistry` can own its imported setup-prefix slots through an `Arc`
(`register_shared`) and exposes `take_session` / `adopt_session`, so an
admitted proof lifetime can move across a scheduling boundary and be resumed
by a new registry. Public prefix slots are stored by value rather than
borrowed.

`akita-pcs`: `AkitaCommitmentScheme::prepare_batched_prove` imports the
prefixes and returns a `PreparedBatchedProve`; `finish_batched_prove` takes the
transcript session and produces the proof. `batched_prove` keeps its previous
one-call contract.

Nothing absorbed depends on the new split: the prepared part reads only the
commitments and the opening points, and the finish replays the same messages
in the same order before drawing the same challenges. The 1K aggregate is
byte-identical under scalar and avx512, one and sixteen rayon threads, and
with the allocation poison enabled.
