# Prover performance: profile and speed levers (local, direction only)

| Field | Value |
| --- | --- |
| Status | informative; a local profile, **direction only**. No number here is a benchmark result: the development box was shared with other agents' builds (load average 2 to 10), and figures marked "est." are estimates. Benchmark results are in the top-level `ref/notes/bench-cloud.md` (cloud VM, medians, exact revisions). |
| Profiled revision | `be398d8` (2026-10-25, before the proof-of-work schedule and the speed branches) |
| Folded in | 2026-10-25, from the profiling report of plan Part 2, Sections 2.2-2.3 (`ref/notes/plan-128bit-speed.md`) |

## 0. What happened since the profile (merged into `main`)

The ranking of Section 6 drove the speed branches `flock-speed` and `piop-speed`, merged
into `main` before `1fef53d`. All changes are prover-only; aggregates stayed byte-identical
except for the proof-of-work nonces of wire version 2/3.

| Lever (Section 6) | Status | Commit(s) |
| --- | --- | --- |
| F1 runtime-dispatched Flock x86 SIMD kernels in the vendored `flock-core` | done | `48d89d8` (`vendor/flock-core/PATCH.md` Section 3) |
| F2 parallel Flock witness inputs, stripe transpose, `input_bits` | done | `efb80ce`, `4bba464` |
| F3 sponge instance and bridge union proven side by side | done | `efb80ce` |
| P5 stage parity tables built while the Flock span is proven | done | `7476508` |
| lookup tree build (fast product, parallel fills); stage-3 main-stack reduction on one lane | done (related to P2/P4, not the coded bottom session of P2) | `94bd37e`, `808b763` |
| instance `T` reused in the witness; vectorized `public_t` | done | `51c3aea` |
| P1 overlap the Flock span with the lookup and stages | open: needs a transcript change and a composition argument | - |
| P2 coded bottom lookup session, P2b, P3, P4, P6, P7, F4, F5 | open | - |

Outcome on the cloud VM (`ref/notes/bench-cloud.md`, Phase 8 check of `1fef53d`, 2 runs
at `N = 32,768`, so not a median of 5): prove 23.74 s (Phase 7, `be398d8`) to 14.62 s,
Flock parts 13.8 s to 7.57 s, lookup 3.86 s to 2.53 s, stage 3 1.80 s to 1.48 s. The 12 s
target of the plan is not reached yet; P1 and P2 are the largest remaining levers by the
estimates below.

Date 2026-10-25. Code `be398d8` (main checkout, read-only; no committed code changed).
Machine: i7-11700K (Rocket Lake, 8 cores, HT off; flags include `pclmulqdq`, `vpclmulqdq`,
`gfni`, `avx512f/bw/vl/vbmi`), runs on `taskset -c 0,1,3-15` (= cores 0,1,3-7, 7 cores),
`RAYON_NUM_THREADS=7`. **The box was shared with other agents' builds and Python jobs during
every run (load average 2 to 10, see each table); all local times are for direction only.**
Cloud figures marked "est." are estimates scaled from local measurements, not measurements.

Builds (all `RUSTUP_TOOLCHAIN=stable`, `CARGO_BUILD_JOBS=4`):

| Dir | RUSTFLAGS | Used for |
| --- | --- | --- |
| `target/clmul` | `-C target-feature=+pclmulqdq` | the baseline (as `scripts/bench.sh CLMUL=1`) |
| `target/prof-gfni` | `+pclmulqdq,+gfni` | A/B of Flock's GFNI (SSE) zerocheck kernel in full mode A |
| `target/prof-v4` | `-C target-cpu=x86-64-v4 -C target-feature=+vpclmulqdq,+pclmulqdq,+gfni,+avx512vbmi` | Keccak-only bench (`agg-keccak --bench keccak_sponge`, no `avx512` feature) to measure Flock's AVX-512 kernels; mode A cannot run in this build (Akita self-check, bench-cloud.md) |
| scratch `gfbench` | soft / `+pclmulqdq` / `+pclmulqdq,+avx512f,+vpclmulqdq` | GF(2^128) multiply microbench on `flock-field` 83f963e |

Scripts and raw logs stayed in the profiling session's scratch directory (not committed).

## 1. Commands

```sh
# 1K and 8K timed runs with traces and 100 ms CPU sampling of /proc/<pid>/stat
PCS_TRACE=1 LINCHECK_TRACE=1 AGG_TIED_TRACE=1 RAYON_NUM_THREADS=7 taskset -c 0,1,3-15 \
  target/clmul/release/agg bench --count 1024 --warmup 1 --reps 3     # and --count 8192
# perf profile of one 8K prove (agg aggregate = prover only), LBR call stacks, monotonic clock
#   (perf_event_paranoid = 1, user-space sampling works; stderr timestamped with CLOCK_MONOTONIC)
PCS_TRACE=1 LINCHECK_TRACE=1 AGG_TIED_TRACE=1 AGG_STAGE_TRACE=1 FLOCK_ZC_TIMING=1 \
FLOCK_COMMIT_TIMING=1 FLOCK_LIG_TIMING=1 RAYON_NUM_THREADS=7 taskset -c 0,1,3-15 \
  perf record -F 499 --call-graph lbr -k CLOCK_MONOTONIC agg aggregate --records r8k.bin --out a8k.bin
perf report --time <t0>,<t1> --no-children --sort symbol -g none --stdio    # per phase
# the same prove with RAYON_NUM_THREADS=1 (parallel speedup per phase)
# GFNI A/B: target/clmul vs target/prof-gfni, agg aggregate 8K, alternated twice
# Keccak A/B: AGG_KECCAK_N=8192 AGG_KECCAK_PROFILE=slim AGG_KECCAK_WARMUP=0 AGG_KECCAK_REPS=1, alternated twice
# lookup alone: AGG_LOOKUP_SIGS=8192 agg_piop-<hash> lookup::tests::timing --ignored --nocapture
```

## 2. Timed runs (`agg bench`, CLMUL, 7 threads)

| N | Load avg (start / end) | Prove median of 3 (min) | Flock part | Everything else | Verify uncached / cached | Prover RSS | Avg process CPU |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 2.79 / 3.66 | 2,479.1 ms (2,477.7) | 1,449.2 (commit 262.3, prove 1,186.5) | 1,029.9 | 292.0 / 260.6 ms | 1,787 MiB | 486% of 700% |
| 8,192 | 2.82 / 6.08 | 13,320.2 ms (13,307.4) | 7,453.4 (commit 2,206.5, prove 5,246.3) | 5,866.8 | 646.1 / 413.7 ms | 7,432 MiB | 502% of 700% |

Median 8K run: instance 553.0, witness and tables 610.0, Flock records 37.5, Akita setup 29.3,
event 1 `G_main` 405.5 (overlapped with the Flock commit), Flock 7,453.4, events 3-5 301.7,
lookup 2,087.8, tau/pre-stage 0.8, stages 296.9 / 266.8 / 988.1, Akita opening 681.8 ms.

## 3. Breakdown of one 8K prove (perf run, 14,208 ms; load 3.2 -> 5.2)

Times from the code's own timers (`FLOCK_*`, `PCS_TRACE`, `AGG_TIED_TRACE`, `AGG_STAGE_TRACE`);
"1T" is the same prove with `RAYON_NUM_THREADS=1` (64,259 ms, load 6.3 -> 3.5; aggregate
byte-identical); busy threads = perf samples / (499 Hz x window). Items without their own
timer are by difference of timestamps (marked "diff").

| Phase | 7T ms | 1T ms | 1T/7T | Busy thr. | Top symbols (perf, 7T) |
| --- | ---: | ---: | ---: | ---: | --- |
| instance (ExpandA, mu, c, T) | 578 | 3,246 | 5.6 | 5.4 (with witness) | `NttPrime::ntt_signed/intt`, `public_t`, SHAKE |
| witness and `G_main` tables | 777 | 3,451 | 4.4 | | `az_crt`, Akita NTT |
| event 2, Flock witness build `honest_parts` + `statement` (diff) | ~890 | ~2,464 | **2.8** | | `pack_rows`, **`SpongeRecord::input_bits` (serial)**, `ChainGate::apply_ab/honest_z` |
| event 2, sponge commit: lane fill / NTT (rate 1/4) / Merkle | 87 / **1,087** / 180 | 393 / 5,972 / 1,036 | 4.5 / 5.5 / 5.8 | 5.7 | **`kernels::portable::butterfly_fused_2layer` 32%**, `ghash_mul_karatsuba_barrett` 25%, BLAKE3 avx512 12% |
| events 3-5: matrices, round 2 | 302 | 1,496 | 5.0 | 4.1 (with rebuild) | software `keccak_p` in `agg_bridge::seed::derive_column_into`, `level1_sums` |
| event 6, witness rebuild + `reattach` (diff) | ~725 | ~2,410 | **3.3** | 1.0-2.1 for ~0.3 s | **`input_bits` 71% at ~1.3 threads**, `pack_rows` |
| bridge blocks / bridge rows and commitment | 18 / 116 | 100 / 318 | | | |
| sponge zerocheck: round 1 (URM) / round 2 / tail | **1,807** / 190 / 170 | 9,936 / 976 / 914 | 5.5 | 5.6 | **`kernels::portable::shift_reduce_inner_ab_scalar` 38%**, `round1_shift_reduce_extract...` 37%, portable `accumulate_convert_with_s_hat_v` |
| sponge lincheck: comb / union partial folds / sumcheck | 20 / 193 / 4 | 20 / 179 / 18 | **~1.0** (folds) | 4.9 | `partial_fold_packed_z_x86_tiled_padded` |
| sponge merged opening total | 1,604 | 7,513 | 4.7 | 5.0 | `ghash_mul_karatsuba_barrett` 26%, `block_first_phase`, portable NTT 13% |
|   block statistics / block rounds | 408 / 92 | 2,350 / 197 | 5.8 | | |
|   Ligerito inner open (of which ligero commits: encode + Merkle) | 1,067 (~460 + ~98) | 4,391 | 4.1 | | portable NTT in `ligero_commit` |
| bridge union proof (whole) | ~410 | ~1,040 | **2.5** | 3.3 | |
| lookup: trees / sessions | 436 / 1,850 (total 2,290) | 1,930 / 8,760 | 4.7 | 5.1 | `kernel::Group::level0`, `lookup::parent`, `Source::fold` |
| stages 1 / 2 / 3 | 297 / 290 / 1,021 | 1,404 / 1,413 / 4,960 | 4.7-4.9 | 5.2 | `Group::level0/level2/dense_round`, `eq_negacyclic_product`, `level1_sums` |
| Akita opening | 745 | 2,268 | **3.0** | 3.7 (tail 2.3) | Akita `decompose_fold`, `coefficient_packing`, `_int_free` |

Stage-3 groups at 8K (7T): `Main eq R2` 178 + 140 ms, `Main eq Lookup(0)` (four 16-bit
`Ints` columns) 126 + 119 ms, `Main eq None` 69 + 72, `Main eq R1` 54 + 41; weights `w_A`
153 ms, parity tables 88 ms, `w_1` 26 ms. Round 0 is 353 ms of stage 3.

Lookup alone at 8,192 (`timing` test, 2 challenges, load 2.0 -> 4.1): prove 2,178.6 ms,
trees 352.6, sessions 1,825.9; per session `s = 25` (bottom, `Ints` lanes, shared by
`X_1, X_2`) 434.4, **`s = 24` 681.3 (37% of the lookup)**, `s = 23` 362.0, `s = 22` 146.6,
`s <= 21` about 200; one challenge: 1,466.1 ms (`s = 24` 398.1). Stored layers 2,120 MiB.

Serial or under-parallel phases (1T/7T well below the 4.4-5.8 of the bulk, which is the
ceiling on this shared box): Flock witness build (2.8, Amdahl fit about 0.5 s serial per
build at 8K), event-6 rebuild (3.3), bridge union (2.5), Akita opening (3.0), lincheck union
partial fold (no scaling: 153 ms at 1K, 179 ms 1T vs 193 ms 7T at 8K). The two `input_bits`
dips are the only near-single-thread windows of the timeline (perf busy threads 1.0-2.1 at
`t = 066.2-066.4`, 2.9 at `063.6`). On the VM, process CPU was 575%-1,081% of 1,600%; 16
vCPUs are 8 cores x 2 SMT, so ~800% already means every physical core is busy and the
"idle" half is mostly SMT siblings, not free cores.

## 4. SIMD kernels: what the CLMUL build leaves on the table

Flock's x86 kernels are selected **at compile time** (`cfg(target_feature = ...)`), so the
CLMUL build runs portable code in its three hottest loops:

| Kernel (vendored flock-core / flock-field 83f963e) | Gate | CLMUL build uses |
| --- | --- | --- |
| additive NTT butterflies (`ntt/additive_ntt_f128/kernels/x86_64.rs`) | `avx512f + vpclmulqdq` | `kernels::portable` |
| zerocheck shift-reduce (`zerocheck/univariate_skip_optimized/kernels/x86_64.rs`) | `gfni` (SSE) or `gfni + avx512f + avx512bw` | `portable::shift_reduce_inner_ab_scalar` |
| zerocheck accumulate / bit transpose | `avx512f + vpclmulqdq` / `avx512vbmi` | portable |
| multilinear fold kernels, `f128_slice::fold_pairs` (flock-field) | `avx512f + vpclmulqdq` | portable / scalar CLMUL |
| scalar `F128 * F128` (flock-field) | `pclmulqdq` | Karatsuba + Barrett, CLMUL |

GF(2^128) microbench (one core, best of 15, L2-resident 2^14 vectors, load ~7):

| Build | elementwise mul | chained mul | `fold_pairs` (1 mul/output) | software |
| --- | ---: | ---: | ---: | ---: |
| `+pclmulqdq` | 7.05 ns (25.4 cyc) | 7.75 ns | 9.85 ns (35.5 cyc) | |
| `+avx512f,+vpclmulqdq` | 4.00 ns (14.4 cyc, autovectorized loop) | 7.40 ns | **1.73 ns (6.2 cyc), 5.7x** | |
| none (software) | 1,729 ns | 1,945 ns | 3,254 ns | ~250x slower than CLMUL |

Keccak-only bench (sponge instance of 8,192 records = 57,344 permutations, `Slim`, 7 threads,
same proof 286,742 B in both builds; load 2.1 -> 3.4; second pair shown, first pair agrees):

| Step | CLMUL | v4 + VPCLMULQDQ + GFNI | Speedup |
| --- | ---: | ---: | ---: |
| commit NTT (each of the two commits; K7 duplicate in this bench) | 1,049 / 1,035 ms | 238 / 253 ms | 4.2x |
| zerocheck round 1 / round 2 / tail | 1,700 / 174 / 229 ms | 121 / 115 / 194 ms | 14x / 1.5x / 1.2x (4.9x total) |
| zerocheck + lincheck | 2,441 ms | 787 ms | 3.1x |
| merged opening | 1,658 ms | 1,034 ms | 1.6x |
| prove incl. witness and commits | 7,215 ms (7,948 perm/s) | 3,247 ms (17,661 perm/s) | **2.22x** |

(The v4 build also autovectorizes our own code; the NTT and zerocheck gains are Flock's
explicit kernels.) Full mode A with `+gfni` only (no AVX-512, so Akita works; aggregate
byte-identical to CLMUL, 863,460 B): zerocheck round 1 2,256 -> 1,514/1,614 ms (-31%), Flock part
9,819/9,829 -> 9,141/9,192 ms (-6.8%), prove 18,256/18,394 -> 17,580/17,668 ms (two
alternated pairs, load 7.4-9.2).

### The 3x gap to Flock's own bench (49K perm/s at 8 threads vs our ~16.6K perm/s at 16)

At 32K our sponge commit is the same size as Flock's 262,144-permutation bench (m = 34, 2 GB;
we commit 39.4K bits per permutation against Flock's 65.5K, so the gate is not the cause).
The gap decomposes as follows (cloud 32K shares est. from the local 8K proportions x 1.78,
the measured local-8K -> cloud-32K ratio of the Flock part):

| Cause | Ours, 32K (est.) | Flock bench | Evidence |
| --- | ---: | ---: | --- |
| Build flags: Flock's workspace `.cargo/config.toml` sets `rustflags = ["-C", "target-cpu=native"]` (Zen 5: AVX-512, VPCLMULQDQ, GFNI kernels); ours runs the portable kernels | zerocheck ~3.9 s, NTT ~1.9 s, opening ~2.9 s | zerocheck 1.50 s, commit 0.94 s, open 1.91 s | local A/B above: 2.22x on the sponge instance |
| Profile: Flock bench `prove_fast` (rate 1/2) vs our `Slim` (rate 1/4: 2x encode, more queries) | | | P7 cloud: `Fast` 22.12 s vs `Slim` 23.74 s |
| Witness built twice, with a serial `input_bits` loop and serial `statement` (c~ by SHAKE per record) | ~2.9 s | witness + lincheck 0.74 s | Section 3: 2.8x / 3.3x scaling, perf dips |
| Bridge union, bridge rows/commit, tie (absent from Flock's bench) | ~1.0 s | 0 | Section 3 |
| Threads: Flock 8 (physical cores; its doc reports SMT hurts its CLMUL kernels), ours 16 on 8 cores x 2 SMT | | | not measured on the VM |

The flags alone explain about half of the gap (2.2x on the sponge instance), the duplicated
serial witness work and the bridge union most of the rest.

## 5. Overlap (plan 2.3 item 3): file/line evidence

- **Flock span prove vs lookup/stages: not overlappable without a transcript change.**
  `crates/agg-piop/src/prover.rs:614` proves the Flock span, `:618` absorbs the proofs
  (`FLOCK_PROOFS_SITE`) and only then `:627` calls `prove_piop`; the registry orders
  `FLOCK_PROOFS_SITE` (`transcript.rs:212`) before the lookup's `X` (`:220`). Every later
  challenge (lookup `X`, `mu_s`, `gamma`, round challenges, `tau`, stage rounds, the PCS
  session at `prover.rs:636`) is squeezed from the sponge that absorbed the Flock proof
  bytes. In data the two are independent: the Flock prove reads only `sd` (event 3,
  `:588`), the matrices, `fs_sigma` (event 5, `:607`) and `Y_sigma` (event 6, `:612`), and
  the PIOP reads none of the Flock proof. Moving the absorb after stage 3 (or into the
  event-12 export) is a wire/transcript change that needs a written composition argument
  (the Flock statement is fixed at event 6 and its challengers are seeded from `fs` only).
- **Prover-only overlap inside the Flock prove:** the bridge-union proof does not read the
  sponge proof: `crates/agg-keccak/src/tied.rs:987` derives the tie points before both,
  the sponge challenger (`:749`) observes `fs`, the gate, the public words and the tie
  geometry, the bridge challenger (`:759`) observes `fs` and the two slot digests; the
  sponge prove (`:989-1002`) and the bridge prove (`:1004-1014`) run in sequence today.
- **Event-2 Flock commit vs `G_main` commit: already concurrent** (`prover.rs:568-577`,
  `crate::pcs::overlap`, a scoped OS thread, `pcs.rs:238-261`); in the P7 32K breakdown the
  step times sum to ~570 ms more than the total, i.e. the 603 ms `G_main` commit is hidden.
- **Event-2 Flock commit vs witness work: not as written.** The Flock records are built
  from the main tables (`crates/agg-cli/src/mode_a.rs:1210-1232`; `sponge_records` reads
  `main.bit_lanes` at `:1095`), and the witness is rebuilt at event 6 (`mode_a.rs:388`,
  `self.build` then `reattach`, which recomputes `statement`, `tied.rs:878`). A prover-only
  restructuring could compute `w1'` first and run the Flock build/commit beside the rest of
  `MainTables::build`.
- Challenge-independent PIOP work that could run during the Flock prove (prover-only):
  `lookup_stacks`, table index halves (`lookup.rs:1133`), `ProverTables::new` parity tables
  (`prover.rs:465`, 88 ms at 8K).

## 6. Ranked levers (estimated savings at 32K on `c4d-highmem-16`; all est.)

Baseline P7: prove 23.74 s = Flock 13.82 (commit 4.40, prove 9.41) + lookup 3.86 + stages
2.75 + instance/witness 2.16 + Akita opening 0.74 + rest. Savings below are estimates from
the local measurements scaled by ~1.75 (local 8K -> cloud 32K ratio of N-linear phases);
they are not additive with each other where noted.

### Flock span

| # | Lever | Est. saving | Prover-only | Evidence |
| --- | --- | ---: | --- | --- |
| F1 | Runtime-dispatched AVX-512/VPCLMULQDQ/GFNI kernels in the vendored `flock-core` (NTT, zerocheck shift-reduce/accumulate/transpose, multilinear folds) and a vendored `flock-field` (`fold_pairs`, slice ops): replace the `cfg(target_feature)` call-site gates by `is_x86_feature_detected!` + the existing `#[target_feature(enable)]` kernels; default build stays portable (extends K12) | **4.5-5.6 s** (NTT -1.5, zerocheck -3.0, opening -1.1) | yes (exact field arithmetic; same proof size in the A/B; check pinned digests) | Section 4 A/B: sponge instance 7.22 -> 3.25 s; NTT 4.2x, zerocheck 4.9x, opening 1.6x |
| F2 | Flock witness: parallelize `input_bits` (`tied.rs:776`, serial `.iter()`), the bits loop (`:784-793`) and `statement` (`:797-805`, serial SHAKE c~); compute inputs/statement once and reuse at event 6; overlap the event-6 rebuild with round 2 (events 3-5) | **1.0-1.8 s** | yes | 2.8x / 3.3x scaling; ~0.28 s serial `input_bits` per call at 8K (2 calls); Amdahl ~0.5 s serial per build |
| F3 | Prove the bridge union concurrently with the sponge instance (`tied.rs:991-1014`) | 0.3-0.6 s | yes | bridge union ~0.41 s at 8K, 2.5x scaling, 3.3 busy threads |
| F4 | Pool sizing on SMT: run the Flock span on an 8-thread (physical-core) pool, the rest on 16 | 0-1 s (unmeasured; one VM A/B with `RAYON_NUM_THREADS=8` vs 16) | yes | Flock's own doc and bench use physical cores |
| F5 | Lincheck union partial fold does not scale with threads (`union_bitbank_fold`) | 0.1-0.3 s | yes | 179 ms 1T vs 193 ms 7T at 8K |
| F6 | Interim, no code change: add `+gfni` to the timing build (SSE GFNI zerocheck path; Akita unaffected) | ~0.9 s (subsumed by F1) | yes (build flag; conflicts with "default build without SIMD", so keep it to the CLMUL timing build or use F1) | Section 4: Flock part -6.8%, aggregate byte-identical |
| ref | `Fast` instead of `Slim` | 1.6 s (measured P7) | no (verifier-visible profile; +6.6 B/sig) | bench-cloud.md P7 |

F1 + F2 + F3 bring the Flock span from 13.8 s to about 6-7 s (est.), at the plan's <= 6 s
target with F4.

### PIOP, lookup and the rest

| # | Lever | Est. saving | Prover-only | Evidence |
| --- | --- | ---: | --- | --- |
| P1 | Overlap the Flock span prove with the lookup and stages: absorb the Flock proofs after stage 3 (wire version bump) | 1.5-3 s before F1, about 1-2 s after F1 (bounded by idle capacity and RSS: lookup layers 8.3 GiB at 32K would coexist with the Flock prove peak) | **no** (transcript change) | Section 5 |
| P2 | Lookup: evaluate the `s = D-1` session from the leaf integers (node `= (2X - a - b, (X-a)(X-b))`, shared by `X_1, X_2`) instead of stored field layers; drop layer `D-1` from the trees | 0.8-1.2 s (lookup 3.86 -> ~2.8 s) and -4 GiB stored layers | yes (same round messages) | `s = 24` is 681 ms = 37% of the 8K lookup; bottom `Ints` session 434 ms covers both challenges |
| P2b | Extend coded evaluation to `s = D-2` (four leaves per node) and trim tree build | 0.3-0.6 s | yes | `s = 23` 362 ms, trees 353 ms at 8K |
| P3 | Start the Flock records + event-2 build/commit beside the rest of `MainTables::build` (compute `w1'` first) | 0.3-0.5 s | yes | Section 5 |
| P4 | Stage 3: coded 16-bit lanes for the four `Ints` columns (`Main eq Lookup(0)`) and cheaper `w_A` weights | 0.2-0.4 s | yes | 245 ms + 153 ms of 1,021 ms at 8K |
| P5 | Challenge-independent PIOP precompute during the Flock prove | 0.1-0.3 s | yes | Section 5 |
| P6 | Multi-way (AVX-512 x8) Keccak for the bridge matrices from `sd` (`agg_bridge::seed::derive_column_into`, runtime dispatch) | ~0.1 s prover (also ~0.1-0.2 s off the verifier's `matrices` step) | yes | software `keccak_p` dominates events 3-5 and the verifier's 165-203 ms `matrices` at 1K |
| P7 | Akita opening (0.74 s, 3.0x scaling, serial tail with `_int_free`) | <= 0.2 s | yes (Akita-internal; upstream) | Section 3 |

Ranking for the 12 s target: F1 >> F2 > P1 (if a wire bump is acceptable) > P2 > F3 ~ P2b ~
P3 > F4 > P4 > rest. Prover-only levers alone (F1-F3, P2-P5): about 8-11 s off 23.7 s (est.),
i.e. roughly 13-16 s; reaching 12 s likely needs P1 or the F4 thread gain as well.
