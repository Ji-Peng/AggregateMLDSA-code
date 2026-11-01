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
encoder, so the fixed-prime path expands the centered bytes through the
portable `Field::from_i64` path on first use of the plane (section 6 makes
that expansion lazy).

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
same exact-i16 `mat_vec_i16` kernel (the portable CRT NTT, or the
IFMA52 specialization when the `avx512ifma` tier is selected at run time) and zero-pads a ragged last block. Both
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

Section 6 supersedes the eager-plane statements below: the dense constructors
now install only the signed mirror, and the canonical field plane is expanded
on first use. The chunked writer and mirror construction described here are
unchanged.

`sources/dense/small.rs`: `expand_i8` and `expand_i16` allocate the canonical
plane once at its final length and fill it in chunks of 32K cells on the
caller's pool; tables of at most one chunk stay serial. This build has no SIMD
encoder, so the scalar `Field::from_i8`/`from_i16` kernel runs per chunk.
Section 6 calls both functions only from the lazy expansion.

`sources/dense/poly.rs`: `from_small_i8_cells` takes the logical centered bytes,
writes the physical-length mirror from them in the same chunked manner, and
leaves the canonical plane to section 6. It removes the caller's `to_vec` plus
`resize` copy, which reallocated because the copy's capacity equalled its
logical length, and spreads the 32 MiB i8 copy and its page faults over the
pool. `from_small_i8` keeps its exact-length contract and shares the common
finish.

`from_wide_field_evals` takes the already encoded field evaluations and the
width-16 cells together. The cells of this path are known to lie outside the
centered `i8` range, so the small-`i8` probe of `from_field_evals` is skipped
and the polynomial starts without an i8 mirror; the signed-16 mirror is
attached as before. In both constructors the bounds are taken over exactly the
same cells as in the previous code. Section 6 routes production width-16
columns through `from_small_i16_cells` instead, so this constructor now has no
caller outside tests. `agg-piop`'s wide column encoder
(`kernel::buf::fill_par`) is unchanged and still writes other tables, but it no
longer feeds the wide dense column.

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
by a new registry. `register` keeps its signature and clones the imported
prefix registry into an `Arc` (`Arc::new(prefixes.clone())`); the clone is
cheap because the handles are `Arc`s and the public rows are small, and only
the one-call paths reach it. Public prefix slots are stored by value rather
than borrowed.

`ProofScopeConsumer::ProofSessionHandle` gains a `'static` bound, which a
session handle needs to travel as `Any` across the scope boundary.
`ProofScope::into_session` moves the admitted session out of the guard and
marks the scope finished, so the guard does not abort on drop. A session that
is then dropped without being rebuilt into a new scope aborts the proof
lifetime, exactly as dropping the original guard would.

`akita-pcs`: `AkitaCommitmentScheme::prepare_batched_prove` imports the
prefixes and returns a `PreparedBatchedProve`; `finish_batched_prove` takes the
transcript session and produces the proof. `batched_prove` keeps its previous
one-call contract.

Nothing absorbed depends on the new split: the prepared part reads only the
commitments and the opening points, and the finish replays the same messages
in the same order before drawing the same challenges. The 1K aggregate is
byte-identical under scalar and avx512, one and sixteen rayon threads, and
with the allocation poison enabled.

## 6. Read the signed mirrors in packing and opening evaluation

This section supersedes the eager-plane text of section 4.

`sources/dense/poly.rs`: `DensePoly` stores the canonical field plane in a
`OnceLock` beside its physical length. `from_small_i8_mirror` and the new
`from_small_i16_cells` install only the signed mirror; `field_coeffs` expands
that mirror on first use over the full physical length, including the zero
padding. The plane is expanded before the `OnceLock` is set, so a parallel
expansion cannot re-enter the lock; two concurrent first readers may both
expand and the loser's plane is dropped. `live_ring_count` validates a `D`-view
without touching the plane and is used by the opening views and by
`evaluate_root_polynomial`. `PartialEq` compares materialized planes first,
then same-kind mirrors, and materializes only when neither fast path applies,
so a lazy and a materialized copy of the same coefficients stay equal without
building a plane when both sides carry a mirror.

`sources/dense/small.rs`: the single `expand` became `expand_i8` and
`expand_i16`, both generic over the commitment field and converting with
`Field::from_i8` or `Field::from_i16`. Both share `expand_cells`, which
allocates the plane once at the physical length and writes it in 32K-cell
chunks on the caller's pool.

`sources/dense/kernels.rs`: coefficient packing reads the `i8` or `i16` mirror
through the same position callback and the same shared loop as the field plane,
converting each cell with `from_i8`/`from_i16` immediately before the weight is
applied. `sources/dense/ops.rs` does the same for base evaluation, reproducing
`scale_accumulate_into` with one `mul_add` per coefficient, and for subfield
evaluation, where the canonical ring is reconstructed before
`mul_accumulate_sparse_rhs_into`. Both fall back to the field plane only when
the source has no mirror. The decompose-fold paths likewise read a centered-`i8`
mirror directly at any single-digit signed-16 basis, where every cell is already
one exact balanced digit. `views.rs` and both decompose-fold paths validate
through `live_ring_count`, so constructing an opening view does not materialize
the plane.

`commitment/source.rs`, `commitment/source/builtins.rs`,
`commitment/source/compiler.rs` and `arithmetic/commitment.rs`: the source
descriptor, the representation-length check and the width-16 commit decision
use a length query that does not touch the plane. A single-digit width-16
commit takes the mirror rows before the generic path asks for coefficients;
one helper holds the dispatch rule and both commit entry points call it. An
`i8` mirror feeds the same signed-16 kernel and widens each cell per
coefficient, so an `i8` column in a signed-16-basis group also skips the plane.

`arithmetic/exact_i16_tests.rs` adds `dense_i16_mirror_commit_skips_the_field_plane`,
which commits a width-16 source and checks that its plane is still absent, and
`dense_i8_mirror_commit_at_the_i16_basis_skips_the_field_plane`, which compares
the mirror commit with the coefficient commit and checks the same absence.
`sources/dense/tests.rs` adds `small_mirror_fold_matches_the_field_plane`,
`small_mirror_packing_matches_the_field_plane`,
`small_mirror_fold_pads_the_last_ring`,
`small_mirror_decompose_fold_at_the_i16_basis_matches_the_field_plane` and
`narrow_commit_representation_does_not_materialize_the_field_plane`; the first
four compare each mirror kernel with a forced-plane copy, and the last checks
that representation selection stays lazy. `crates/agg-piop` adds
`tests/lazy_field_plane.rs`, which proves a default aggregate at 64 records and
asserts that the process-wide counter of lazily installed planes does not move.

The readers that still build the plane on demand are:

- `opaque/owned.rs` `SourceCoefficients::source_coefficients`, reached by EOR
  `extension`/`begin_eor` (`opaque/owned.rs`, `opaque/eor.rs`) and by the
  dense-polynomial import. Root EOR runs only under `run_eor`.
- `commitment/source/builtins.rs`, the centered-reach fallback used when the
  threshold is not the half modulus.
- `sources/dense/poly.rs` `digit_planes_for`, outside the mirror case: a
  multi-digit shape, an i16 poly at an i8 basis, or `shape = None`.
- `sources/dense/ops.rs`, the decompose-fold fallbacks that run without a
  digit cache and outside the single-digit mirror cases (multi-digit shapes or
  a basis below the mirror's exact range).
- `arithmetic/commitment.rs`, the generic `coefficients()` path taken when
  neither mirror early return nor `PredecomposedDigits` applies.
- `crates/agg-piop/src/pcs.rs`, a test only.

The verifier side builds no `DensePoly`, and `from_wide_field_evals` keeps its
plane because it is now test-only.

The vendored diff also carries rustfmt-only reformatting of
`commitment/source/builtins.rs` (the `negative` if/else),
`sources/dense/ops.rs` (the `poly_helpers` import list),
`arithmetic/commitment.rs` (the `exact_i16` import list) and
`sources/dense/poly.rs` (the `digit_planes_for` closure). No behavior changes.

Outside this copy, `agg-piop`'s `group_polynomials` calls
`DensePoly::from_small_i16_cells`, so a width-16 column no longer encodes its
field plane at construction. This removes the 8 B/cell plane write for every
narrow column at event 1; the plane remains available if a later reader needs
it.

The 1K aggregate is byte-identical under scalar and avx512, one and sixteen
rayon threads, and with the allocation poison enabled. On `agg aggregate` over
`p1p8/records.bin` with eight rayon threads, peak RSS fell from 1,970-1,979 MiB
to 1,883-1,898 MiB (two before and five after runs; direction only). No
arithmetic, wire format or verifier check changes.

## 8. Instruction-set cap from `AGG_ISA`

`akita-algebra/src/isa.rs`, `akita-challenges/src/isa.rs` and `akita-cpu-backend/src/isa.rs`
are one small module each (the vendored crates do not depend on each other for this): a
`Level { Scalar, Avx2, Avx512, Avx512Ifma }` parsed once from `AGG_ISA`, with `avx2()`,
`avx512()` and `ifma()` predicates. An unknown value panics; an unset value leaves every
check at feature detection. The module only ever lowers a level.

Every run-time detector in the three crates consults it:

| Site | Cap |
| --- | --- |
| `binary/host.rs`, `binary/product.rs`, `binary/packed.rs` PCLMUL kernels | AVX2 |
| the same files' `avx512f` kernels (4-lane equality, packed `x86_512`) | AVX-512 |
| `binary/field_switch/kernels.rs` (Gfni512 / Gfni256) | AVX-512 / AVX2 |
| `ntt/avx/runtime.rs` (`detect_cpu_features`, `use_avx2_transform_ntt`) | AVX2 and AVX-512 |
| `ntt/ifma52.rs` (`ifma52_available`) | AVX-512IFMA |
| `ntt/digit_validation.rs`, `ring/cyclotomic/decomposition.rs`, `ring/cyclotomic/shift_windows.rs` | AVX2 / AVX-512 |
| `akita-challenges/sampler/op_norm_accumulate.rs` | AVX2 |
| `akita-cpu-backend/sources/packed_digits/x86_64.rs` (BMI2 + AVX2) | AVX2 |

`AKITA_SCALAR_NTT` and `AKITA_AVX512_NTT` still apply on top. The CRTD NTT keeps its AVX2
kernels under `AGG_ISA=avx2`; `ifma52_enabled()` is false, so the AVX-512IFMA NTT cannot
run. `akita_algebra::simd_report()` (with the two sibling crates' `simd_report()`)
prints the cap and the selected tiers.

## 9. Runtime dispatch of the AVX2 poly helpers

`sources/poly_helpers/windowed_accum.rs`: the 32-coefficient AVX-512BW gather is
deleted. It changed `GATHER` from one 16-coefficient window to two, a table
layout change, and its self-check failed on both Rocket Lake and Zen 5. `GATHER`
is one window under every target, and the kernel always uses 256-bit vectors.

`kernels/decompose_fold_avx.rs`, `sources/poly_helpers/mod.rs` and
`sources/poly_helpers/windowed_accum.rs`: the AVX2 kernels compile under every
x86-64 build (each entry point carries `#[target_feature(enable = "avx2")]`)
and are selected by `AGG_ISA >= avx2` plus
`is_x86_feature_detected!("avx2")`. A global `x86-64-v3` build is no longer
needed for them. `AKITA_SCALAR_NTT=1` still selects the scalar paths.
