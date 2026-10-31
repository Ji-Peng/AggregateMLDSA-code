# Engineering techniques of the prover

This note describes the prover's engineering techniques: what each one does, where it
lives in the code, the knobs that control it, and its measured effect. The techniques
described here are prover-only; the vendored trees also carry the verifier-visible changes
recorded in their patch records. The protocol itself is specified in `docs/spec.md`; the
vendored changes are recorded in `vendor/akita/PATCH.md` and
`vendor/flock-core/PATCH.md`.

Unless a different condition is stated, timings refer to relation H-u, 32K signatures,
median of five cold runs, and the reference run takes 2.76 s on the AVX-512IFMA build and
2.98 s on the general build, at 49.2 B/sig. The phrase "measured on a Zen 5 node (EPYC 9355),
32K signatures, 16 cores / 32 threads" identifies that run. Numbers from another machine,
another size or another thread count carry that condition with them; estimates are marked
as estimates.

## 1. The fork-join schedule

Event 6 closes the part of the transcript that the Flock side depends on. The prover then
clones the sponge into branch L, which proves the lookup, and branch S, which draws `tau`
and proves the pre-stage claims and stages 1 and 2. The join absorbs the two branch digests
in order, and stage 3 runs on the trunk. The Flock proofs depend only on data absorbed up to
event 6, so they run on their own pool beside the PIOP and are absorbed at the join. No PIOP
challenge depends on them, and the transcript and the proof are identical with or without
the fork.

The schedule lives in `crates/agg-piop/src/prover.rs`: `prove` starts the fork,
`fork_join` runs the Flock side on a fresh Rayon pool on an OS thread, and `run_branches`
runs L and S. `Transcript::fork` and `Transcript::join` are in
`crates/agg-piop/src/transcript.rs`. Branch S gets its own pool when the fork is running and
the PIOP pool has at least four workers, so branch L keeps every worker of the main pool and
its round barriers do not wait behind stage jobs.

| Variable | Default | Effect |
| --- | --- | --- |
| `AGG_FLOCK_THREADS` | `ceil(3/8)` of the Rayon pool, 12 at 32 threads | Workers of the Flock pool; `0` disables the fork. |
| `AGG_PIOP_THREADS` | the Rayon pool | Workers of the PIOP pool; with the Flock pool it is capped at the logical CPUs. |
| `AGG_PIOP_S_THREADS` | 16 at 16C32T, 6 at 8C16T | Workers of branch S's own pool; `0` shares the PIOP pool. |
| `AGG_PIOP_FORK` | on | `0` or `off` runs L and S sequentially. |
| `AGG_OPENING_THREADS` | one per physical core | Workers of the pool that runs the opening's finish part; prepare shares the PIOP pool. |

Measured on a Zen 5 node (EPYC 9355), 32K signatures, 16 cores / 32 threads, the dedicated
stage-branch pool is worth about 100 ms. At 8C16T the sweep found no gain from a separate
pool; the default is still six workers. Thread-split sweeps at 16C32T show that the fork is
bound by the work of its branches, not by the split: moving threads between the pools
changes the wall time by much less than the length of the critical branch.

## 2. The huge-page global allocator with a FIFO block cache

The prover allocates many tables of tens to hundreds of MiB. The system allocator maps each
one freshly and unmaps it on free, so every table pays a fault per 4 KiB page on first touch
and every free pays an unmap with a TLB shootdown. `crates/agg-piop/src/alloc.rs` installs a
global allocator (`HugePageAlloc`) that handles blocks of at least 4 MiB directly:

- The block is a private anonymous mapping rounded up to 2 MiB and aligned to 2 MiB, so any
  requested alignment up to 2 MiB holds. The mapping is advised with
  `madvise(MADV_HUGEPAGE)`, and first-touch faults then take 2 MiB at a time.
- Freed blocks are kept in a 64-slot cache, oldest first, with FIFO eviction when the slots
  or the byte cap are exhausted. The default cap is 6 GiB, and a block larger than the cap
  is unmapped. A later uninitialised allocation takes the smallest cached block between the
  requested size and 125% of it, trims it, and needs neither a fault nor an unmap.
- A zeroed allocation always takes a fresh mapping. Its zero pages are then faulted in by
  whichever threads first write them, in parallel; zeroing a cached block would be one
  serial `memset`. A grow uses `mremap` in place when the address space allows it and
  otherwise copies into a new block.
- Outside poison mode, the cache and the advice change page backing and reuse only, never
  the bytes an allocation can observe. Non-Linux targets forward everything to the system
  allocator.

The allocator is installed in `crates/agg-cli/src/main.rs`. `set_enabled`, `set_cache_bytes`
and `set_poison` are the knobs behind the environment variables:

| Variable | Default | Effect |
| --- | --- | --- |
| `AGG_HUGEPAGE` | on | `0` turns the huge-page advice off. |
| `AGG_ALLOC_CACHE` | 6 GiB | Cap of the cache in GiB; `0` disables the cache and releases it. |
| `AGG_ALLOC_POISON` | off | `1` fills every reused block with `0xA5` before returning it. |

Poison mode is a test mode: it exposes a reader of uninitialised memory, including in
Flock's scratch pools, where `vendor/flock-core/src/scratch.rs` does the same for buffers
handed out by `take_f128`, `take_hashes` and `take_u8`. Proofs with reused blocks poisoned
are byte-identical to proofs with a cold cache.

Measured on a Zen 5 node (EPYC 9355), 32K signatures, 16 cores / 32 threads, page faults
fall from 17.2 M to 0.9 M per bench run, the prover gains about 360 ms, and peak RSS falls
by about 2 GiB. At 8C16T the gain is about 450 ms.

## 3. Parallel first touch

No large table is zeroed by a serial pass. A table that is fully overwritten is allocated
as uninitialised storage and written in parallel, so its pages are first touched by the
writers. The reusable primitives are `uninit`, `fill_par`, `reuse` and `finish` in
`crates/agg-piop/src/kernel/buf.rs`. `MainTables::encode_tiles`
(`crates/agg-piop/src/source_tables.rs`) allocates the final columns once and gives each
worker a disjoint tile of records; `build_nodes` (`crates/agg-piop/src/lookup.rs`) touches
both lanes of a layer in parallel; `expand` and `attach_small_i16`
(`vendor/akita/crates/akita-cpu-backend/src/sources/dense/`) fill the field plane and the
signed-16 mirror in chunks of 32K cells on the caller's pool. Flock's
`alloc_zeroed_vec` (`vendor/flock-core/src/lib.rs`) uses `alloc_zeroed`, whose zero pages are
likewise faulted in by the parallel writers rather than memset by one thread.

The pitfall is `vec![T::zero(); n]`. For a newtype such as a field element it does not take
the `calloc` path: it clones a zero value `n` times, which is a single-threaded fill with a
page fault per 4 KiB page. For a buffer that is fully overwritten, allocate
`Box::<[MaybeUninit<T>]>::new_uninit_slice(n)` and fill it in parallel, or allocate
`vec![0u64; n]` and reinterpret it with `Vec::from_raw_parts` when the representation
permits it. A `vec!` line at the top of a profile is the first place to look.

The gain of parallel first touch is part of the memory-system result in Section 2; the
notes do not record a separate A/B for it.

## 4. AVX-512IFMA kernels for F_p and F_{p^2}

On the `ifma` line, the base field and its quadratic extension have slice kernels that keep
canonical `u64` storage: `&[F]` is read as `&[u64]`, and limbs are split in registers only.
`crates/agg-simd/src/fp.rs` (`fp_mul`), `k.rs` (`k_mul`, `k_mul_base`), `dot.rs` (`fp_dot`,
`k_dot`, `k_base_dot`) and `x86.rs` (the IFMA primitives and the radix-2^52/2^51 split and
reduction) implement the kernels; `fused.rs` provides the fold-and-evaluate forms used by
the sum-check groups. `detect.rs` selects the backend once per process:
`AGG_SIMD=portable` forces the scalar path, `AGG_SIMD=avx512` requires
AVX-512F/DQ/BW/VL/IFMA, and the default uses IFMA when the CPU has it.

The kernels use four interleaved blocks of eight F_p elements, 7 IFMA multiply-adds for the
product and 2 for the fold by 59, and schoolbook products with two lazy reductions for K.
Dots accumulate unreduced radix-2^51 limbs and flush every 1,024 F_p or 512 K products.
Their scalar fallbacks are the oracle: `crates/agg-simd/tests/differential.rs` compares every
kernel with the scalar path at the flush boundaries, at the tail, and on worst-case values,
and the proofs are byte-identical whichever path runs.

Single-core, L1/L2-resident costs on Zen 5 are:

| Kernel | Cost | Against the best scalar code |
| --- | --- | --- |
| F_p multiply | 1.38 cycles/element | 3.9x |
| F_p^2 multiply | 3.92 cycles/element | 5.3x |
| F_p dot | 0.37 cycles/term | 4.8-5.9x |
| F_p^2 dot | 1.25 cycles/term | 4.8-5.9x |

Streaming beyond L3 the gain falls to about 1.0x (est.), because memory traffic decides
there. Measured on a Zen 5 node (EPYC 9355), 32K signatures, 16 cores / 32 threads, the IFMA
build is about 0.2 s faster at 16C32T and about 1.1 s at 8C16T (2.76 s against 2.98 s, and
4.43 s against 5.50 s).

## 5. Stage-3 plane groups and fused fold-and-round evaluation

Stage 3 proves the ring identity, the range links, the Booleanity checks, the reduction to
`r^(3)`, the lookup stacks and the two sides of the projection identity. Its two dedicated
groups are `WeightIntGroup` for `w_A z` and `LinearIntGroup` for the lookup-stack and bit
reductions. Both are bilinear forms with coefficient slots: the first round reads the
integer cells, and the first fold writes the group's own table. On the `ifma` line,
`crates/agg-piop/src/kernel/plane.rs` runs these groups on lane-major `K` planes, where
round `t + 1` is one fused pass over blocks of 512 output rows: every lane is folded by the
bound challenge into the spare buffer, and the next round is evaluated on the pairs just
written. `crates/agg-simd/src/fused.rs` holds the kernel. `crates/agg-piop/src/kernel/tail.rs`
does the same for the generic dense groups from the third bind on. The default build uses
the row-major dedicated groups in `crates/agg-piop/src/kernel/layer.rs`, which fold four
rows into two and accumulate `q(0)` and `q_2` in one read.

Before a lane is dense it stays a small integer. A committed `i8` column is coded when its
cells fall into at most 16 classes under the maps of the lanes that read it (8 for a digit
read through `z (z + 1)`, 16 under the identity, 2 for a bit). Pieces on coded lanes are
bucketed by pair or quartet patterns. When the quartets outnumber their largest pattern
space by `BUCKET_MIN_RATIO = 4`, one pass with the round-1 equality weights serves rounds 0
and 1; otherwise the pair masses do. The first bind turns coded lanes into pair lookup
tables, the second into quartet tables when the space is at most 4,096, and the third
materialises every lane in one dense `K` table at an eighth of the domain. The constants
and the coded-column builder are in `crates/agg-piop/src/kernel/mod.rs` and `coded.rs`.

Measured on a Zen 5 node (EPYC 9355), 32K signatures, 16 cores / 32 threads, stage 3 takes
about 0.48 s of the 2.76 s run. The notes do not record an isolated A/B for the plane-group
change, so this section quotes no separate saving for it.

## 6. Vendored Akita prover changes

`vendor/akita` is a verbatim copy of the pinned Akita revision with five prover-only
changes; arithmetic, wire format and verifier are untouched. The record is
`vendor/akita/PATCH.md`.

- **Setup-thread warm-up.** `CpuBackend::prewarm_commit`
  (`vendor/akita/crates/akita-cpu-backend/src/arithmetic/controls.rs`) builds the executor
  for a root plan and calls `prewarm_plan_resources` in
  `commitment/executor.rs`, which installs the A slots of both dense representations and
  the B slot without materialising a source. `PcsProver::prewarm_groups`
  (`crates/agg-pcs-config/src/commit.rs`) runs it for all seven groups on the setup thread;
  `TableCommitter::warm` (`crates/agg-piop/src/pcs.rs`) is the caller, invoked while the
  instance is computed.
- **Commits from the small mirrors.** `DensePoly` keeps the signed-16 mirror of a width-16
  source (`attach_small_i16`, `sources/dense/poly.rs`), and `prime_digit_cache` builds the
  ring-major digit planes from it. A single-digit commit takes
  `dense_commit_i16_mirror_rows` (`arithmetic/exact_i16.rs`) and reads 32 MiB per column
  instead of the 256 MiB field plane; with one digit the cache is a live-prefix slice of
  `small_i8_coeffs`, so the per-column copy disappears. This removes 256-384 MiB of
  resident memory at 32K from the pre-fork commit through the end of the opening.
- **Split opening, spread passes.** The opening is cut at the transcript session. The
  point-only prepare (`prepare_openings` in `opaque/openings.rs`, the digit material and the
  relation inputs) runs in the PIOP branch once stage 3 has fixed the opening points, and
  the transcript-bound finish runs after the join. `fold_grind.rs` draws the challenges in
  group order and then probes the groups of one nonce concurrently, and the finish keeps its
  own pool (`AGG_OPENING_THREADS`). The backend, session and prepared-opening handles are
  `Send + Sync` (`backend/{opening,scope,traits}.rs`), and the prepared state is `Send` so
  that it can cross the join.
- **Digit caches.** `CachedDigits16` and `cached_i16_decompose_fold_partitioned`
  (`sources/poly_helpers/decompose_fold_partitioned.rs`) fold from the signed-16 mirror,
  and the sparse and chunked probes read the mirror instead of decomposing the field plane
  on every nonce (`sources/dense/ops.rs`).

Measured on a Zen 5 node (EPYC 9355), 32K signatures, 16 cores / 32 threads, the Akita
changes are worth about 150 ms on the IFMA build. On the general build, with the allocator
in place, the same changes are worth about 600 ms. The 1K aggregate is byte-identical under
scalar and AVX-512 and under one and sixteen threads; the 32K aggregate is byte-identical
across the general and IFMA builds with reused blocks poisoned.

## 7. Flock changes

The vendored `flock-core` carries the following prover changes; all are described in
`vendor/flock-core/PATCH.md`. The round polynomials are unique, so the routes below
compute the same field elements and leave the proof bytes unchanged.

- **`a, b` rows on the PIOP pool.** On the `ifma` line the sponge witness generator writes
  `z` only. `KeccakSide::prepare_fork` (`crates/agg-piop/src/prover.rs`) returns a closure
  that the PIOP pool runs at fork start; it calls `pack_ab`
  (`crates/agg-keccak/src/witness.rs`) to build `a = A z` and `b = B z` eight rows per
  task, and hands them to the proof through a channel (`FlockProver::prepare_fork`,
  `crates/agg-cli/src/mode_a.rs`). The default build builds them in the tied-part witness
  pass (`pack_slot_groups`).
- **Round-23 pruning.** `GateLayout::new` and `round_lanes`
  (`crates/agg-keccak/src/gate.rs`) commit only the lanes of the last round that hold an
  output bit. For the SHAKE256 sponge that is four of 25 lanes, so the other chi AND bits
  are neither committed nor constrained.
- **Sum-check memory traffic.** The zerocheck folds and evaluates the next message in one
  pass (`fold_and_message_x86_avx512`, `zerocheck/multilinear.rs`), the packed round-3 path
  reduces wide products once per equality block (`round3_task_avx512`), the merged opening
  fuses fold and next message per term (`pcs.rs`, `pcs/block_stats_portable.rs`), and the
  union lincheck folds and builds its bit banks in one pass over `z`
  (`lincheck/union.rs`). The portable kernels accumulate unreduced products and reduce at
  the flush boundary (`kernels::LaneAcc`). These are pure data-layout and
  evaluation-order changes.
- **AVX-512, VPCLMULQDQ and GFNI kernels.** `vendor/flock-core/src/simd.rs` checks the CPU
  once and caches the answer; the additive NTT, zerocheck and Ligerito kernels are called
  through it. The kernels cover the butterflies, the round-1 shift-reduce and bit
  transpose, the round-2 fold and message, the packed round 3, the Ligerito induced basis
  and the inverse-NTT table. They are exact and prover-only; `FLOCK_SIMD=portable` (or
  `off`) forces the portable path for A/B runs, and `gfni` keeps only the SSE GFNI kernel.
- **Keeping the witness from commitment to proof.** When a batch has exactly one tied part
  (a span, or one part in a one-segment batch), `PartState` in
  `crates/agg-cli/src/mode_a.rs` keeps its `TiedCommitted` (`crates/agg-keccak/src/tied.rs`)
  from event 2 until the join. The Flock branch then proves the part without rebuilding its
  sponge buffers, input bits and statement. Batches with more than one part keep the
  drop-and-rebuild path, which bounds the live Flock buffers.

Measured on a Zen 5 node (EPYC 9355), 32K signatures, 16 cores / 32 threads, the `a, b` rows
on the PIOP pool save 50-85 ms. Round-23 pruning is small. The Flock critical path in that
run is zerocheck 800 ms, lincheck 120 ms and opening 625 ms inside a fork of 1.75-1.95 s;
the kernel and traffic changes are what hold those numbers, but the notes record no
isolated 32K A/B for them. Keeping the witness is measured separately at 32K on
c4d-highmem-16 (EPYC 9B45, the same Zen 5 core), 16 threads, where it saves about 0.66 s.

## 8. AVX2 and AVX-512 kernels for the ML-DSA instance and witness

`crates/agg-mldsa-simd` holds every ML-DSA step of the instance and the witness behind one
runtime-dispatched engine. `dispatch.rs` selects scalar, AVX2 or AVX-512 by
`is_x86_feature_detected!` and caches the result; `AGG_MLDSA_SIMD=auto|scalar|avx2|avx512`
overrides it, and a requested backend whose features are absent is refused. The kernels
are `ntt/{scalar,avx2,avx512}.rs`, `sample/`, `finish/`, `hash/` and `keccak/`. The
consumers are `PublicInstance::new` (`crates/agg-witness/src/instance.rs`), the batched
derivation in `crates/agg-witness/src/fast.rs`, and the streamed public pass in
`crates/agg-piop/src/public_stream.rs`.

Each polynomial is transformed under two primes: the ML-DSA prime `q = 8,380,417` and the
28-bit transform prime `P2 = 167,772,161 = 5 * 2^25 + 1`, both with a 512th root of unity.
`derive_one` (`crates/agg-witness/src/fast.rs`) transforms `z` under both primes, takes
`dot4` against the prepared key under both, inverse-transforms both and combines the
residues by centered CRT. The pair is exact because
`|A z| <= l 256 (q - 1)/2 gamma_1 < q P2 / 2` for every decodable `z`. The row then
finishes without division: centered CRT, UseHint on `A z - T` modulo `q`, and the exact
quotient `v` with its bounds. A record that fails any step falls back to the reference
derivation (`crates/agg-witness/src/derive.rs`), so its error and its accepted witness are
exactly the reference ones. The reference CRT path with the 31-bit prime
`2,013,265,921` is `crates/agg-witness/src/crt.rs`.

Measured on a Zen 5 node (EPYC 9355), 32K signatures, 16 cores / 32 threads, the instance
and the witness together take about 230 ms of the 2.76 s run. An independent A/B on
c4d-highmem-16 (EPYC 9B45, the same Zen 5 core), 32K, 16 threads, moved the instance from
447.7 ms to 158.4 ms and the witness and tables from 419.7 ms to 65.4 ms when the engine was
introduced. The backend equality gate is described in Section 9.

## 9. Byte-identity gates and cold-start benchmarking

Byte identity is the acceptance test for every prover change. Two gates cover the builds
and the dispatch:

- `crates/agg-cli/tests/all/simd_backends.rs` runs `keygen-batch`, `sign-batch`,
  `aggregate` and `verify` for scalar, AVX2 and AVX-512 and for one and sixteen threads. It
  covers H-u at 1,024 records over 64 and over 1,024 keys, H-exact at 256 records and
  H-FIPS at 256 records, and compares the aggregate and the key-cache bytes across every
  combination.
- `scripts/simd-ab.sh` builds `origin/main` as a baseline and compares a 1K aggregate of
  the working tree under scalar, AVX2 and AVX-512 and one and sixteen threads with `cmp`.
  Cross-build equality at 32K (general against IFMA) is checked the same way, and the
  32K gates run with `AGG_ALLOC_POISON=1` so a reader of a reused block is caught.

`agg bench` measures cold runs. `cold_start` in `crates/agg-cli/src/commands.rs` waits for
detached drops (`agg_pcs_config::sched::wait_background_drops`), empties the allocator cache
(`agg_piop::alloc::release_cache`) and clears Flock's scratch pools
(`agg_keccak::flock_core::scratch::clear`) before every timed prove and before the verifier
runs. The defaults are one warmup and five timed runs (`--warmup`, `--reps`), with the
median reported, so each quoted run starts like a new process.

The cold reset changes no proof byte and leaves the medians unchanged beyond noise:
measured on a Zen 5 node (EPYC 9355), 32K signatures, 16 cores / 32 threads, warm and cold
Flock pools differ by less than 20 ms (about 2.99 s on the general build and 2.75 s on the
IFMA build). Keeping the reset therefore does not inflate the quoted times.

## 10. Techniques measured and not adopted

Each of these was exact and byte-identical; the reason for leaving it out is performance.
The comparisons use the same EPYC 9355 node class at 32K signatures; each row states the
thread count.

| Technique | Measured reason it is not used |
| --- | --- |
| Thread donation from the PIOP pool to the Flock opening | The PIOP branch always finished before the opening began, so the donated workers were never available. |
| More Flock threads | 12, 14 and 16 workers all gave 3.45-3.55 s at 16C32T; alone, 32 workers gave 1.79 s against 2.07 s on 12, so the branch is bandwidth-bound. |
| Streaming stores in the zerocheck | On and off differed by less than 20 ms at 16C32T (2.87-2.91 s either way). |
| The level-0 opening kernel rewrite | 2.79-2.82 s against 2.76-2.79 s at 16C32T, about 30 ms slower. |
| The wide-A2 layout | 2.99 s either way at 16C32T and 5.47 s against 5.46 s at 8C16T; event 1 stays bounded by the adjacent Flock commit. |
| Pipelining the Flock witness with the table build | No change outside the fork (1.28-1.29 s) and totals within noise (2.98-3.01 s against 2.95-3.01 s at 16C32T). |
