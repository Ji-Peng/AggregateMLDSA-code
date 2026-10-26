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
| F3 sponge instance and bridge union proven side by side | done, opt-in (`AGG_TIED_CONCURRENT`); moot since tied-part version 5, which has no bridge union (K21) | `efb80ce` |
| P5 stage parity tables built while the Flock span is proven | done | `7476508` |
| lookup tree build (fast product, parallel fills); stage-3 main-stack reduction on one lane | done (related to P2/P4, not the coded bottom session of P2) | `94bd37e`, `808b763` |
| instance `T` reused in the witness; vectorized `public_t` | done | `51c3aea` |
| P1 overlap the Flock span with the lookup and stages | done on branch `fork-join-p1` (deviation K22), merged into `main` 2026-10-27 (`8adea56`): the Flock proofs at a join after stage 3, proven on their own pool (`AGG_FLOCK_THREADS`, default half the global pool); composition argument in `docs/soundness.md` 12. Decision rule of the memo (>= 1 s saved at 32K in the cloud dry run): met, -1.11 s (pool 16) and -1.15 s (pool 8), `ref/notes/bench-cloud.md` Phase 10 (6). Local 8K (measured, direction): -0.93 / -1.03 s, prover peak +0.8 to +1.1 GiB; pool sweep at 8K: 4 / 8 / 15 threads 6.03 / 6.37 / 6.76 s, rerun at `52c3d23`: 4 / 8 threads 6.34 / 6.35 s vs main 7.31 s; cloud sweep at 32K on `184374f` (Section 8.1): pools 0 / 4 / 6 / 8 / 12 give 14.26 / 13.85 / 12.78 / 12.75 / 13.15 s, so the default 8 stays | `f34d1bd`, `52c3d23` |
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
| F3 | Prove the bridge union concurrently with the sponge instance (`tied.rs:991-1014`; superseded: version 5 deletes the union, K21) | 0.3-0.6 s | yes | bridge union ~0.41 s at 8K, 2.5x scaling, 3.3 busy threads |
| F4 | Pool sizing on SMT: run the Flock span on an 8-thread (physical-core) pool, the rest on 16 | 0-1 s (unmeasured; one VM A/B with `RAYON_NUM_THREADS=8` vs 16) | yes | Flock's own doc and bench use physical cores |
| F5 | Lincheck union partial fold does not scale with threads (`union_bitbank_fold`; since tied-part version 5 it runs once more, for the bit claim's full bank, K21, `docs/keccak.md` 12.7) | 0.1-0.3 s | yes | 179 ms 1T vs 193 ms 7T at 8K |
| F6 | Interim, no code change: add `+gfni` to the timing build (SSE GFNI zerocheck path; Akita unaffected) | ~0.9 s (subsumed by F1) | yes (build flag; conflicts with "default build without SIMD", so keep it to the CLMUL timing build or use F1) | Section 4: Flock part -6.8%, aggregate byte-identical |
| ref | `Fast` instead of `Slim` | 1.6 s (measured P7) | no (verifier-visible profile; +6.6 B/sig) | bench-cloud.md P7 |

F1 + F2 + F3 bring the Flock span from 13.8 s to about 6-7 s (est.), at the plan's <= 6 s
target with F4.

### PIOP, lookup and the rest

| # | Lever | Est. saving | Prover-only | Evidence |
| --- | --- | ---: | --- | --- |
| P1 | Overlap the Flock span prove with the lookup and stages: absorb the Flock proofs after stage 3 (implemented on branch `fork-join-p1`, merged into `main` 2026-10-27, deviation K22; wire version 4 kept, nothing released since K21) | 1.5-3 s before F1, about 1-2 s after F1 (est.); measured: cloud dry run at 32K -1.11 s (Flock pool 16, +2.6 GiB) and -1.15 s (pool 8, +0.5 GiB), bench-cloud Phase 10 (6); local 8K -0.93 / -1.03 s with pool 8, +0.8 to +1.1 GiB | **no** (transcript change) | Section 5; `ref/notes/overnight-20261026/fork-join-p1.md` |
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

## 7. Verifier kernels (branch `verifier-kernels`, merged into `main`; memo 3.4 (a)-(c))

Scope: the cached verifier's public-data steps, which were 272 ms of the 424.0 ms cached
verify at 32K (public_data 106.3 + public_t 91.9 + public_a 73.8 ms; measured,
`ref/notes/bench-cloud.md` "Verifier at 32K", code `8da7538`). All changes are verifier-only
and exact: the transcript, the aggregate bytes and every accept/reject decision are
unchanged (checks below). Local numbers: i7-11700K under WSL2, CPUs 0,1,3-15, shared with
other agents' builds and provers (load average 4-14 during the runs), default portable build
(no CLMUL, so the Flock step is 3.5-6.6 s here and dominates the local totals);
**direction only**, not benchmarks.

### 7.1 Changes

| Step | Before (`6803ce5`) | After | Exactness argument |
| --- | --- | --- | --- |
| `sparse_t` (`public_data.rs`) | per entry `if p + m < N` branch in the inner loop | per nonzero `c_p`, two contiguous slice updates split at `N - p` (`straight += t1[..N-p]`, `wrapped -= t1[N-p..]`, signs swapped for `c_p = -1`); `sparse_ct` without the `2^d` factor | the same integer additions in another order; `|c t1| <= tau (2^10 - 1) < 2^16` |
| `T(r2)` inner product (`public_stream.rs`, 1,024 terms per record) | `E::mul_i64` per term, i.e. a full `Ext2` product (`jolt_field::Ring::mul_i64` is `self * from_i64(n)`) | `SplitWeights::dot::<16>` on `c t1`, times `2^d` once per pass | integer times `Ext2` element is coordinatewise; each coordinate sum is exact over the integers and reduced once |
| `key_weight` (cached `A`, 16 x 256 terms per key) | `E::mul_i64` per term, `eq_row[rho] eq_col[s]` per key | `SplitWeights::dot::<22>`, the 16 products `eq_row[rho] eq_col[s]` once per pass | as above; `|A| <= (q-1)/2 < 2^22` |
| `StreamedPublic::prepare` dedup | serial `HashMap<&[u8], usize>` (SipHash) over all `pk` | keyed SipHash (`RandomState`) per record in parallel, parallel sort of the indices by (hash, full `pk` bytes, index), one scan for runs, runs sorted by first record | equality on the **full** `pk` bytes (never the 32-byte `rho` prefix, which the signer chooses); same keys, same order (first appearance), same record lists; test against the old map |
| cached keys in `prepare` | per key `pkDecode`, a copy of `pk` (1,312 B) and `t1` (2 KiB) | a cached key holds only its cache entry (`t1`, `tr` of the entry were decoded from the same full `pk`; the cache file recomputes `t1` on load); uncached keys as before | same `t1`/`tr` values; the `pkDecode` length error is kept for cached keys |
| record stream `X` (`binding.rs`) | serial `Vec` growth | offsets by prefix sum, every record copied into its own disjoint slice in parallel | byte-equal to the serial stream (test) |

`SplitWeights::dot::<BITS>`: every weight is stored as the four 32-bit halves of its two
canonical `F_p` coordinates; with biased values `b_j = v_j + 2^BITS in [0, 2^(BITS+1))`
each product `half * b_j` is below `2^(BITS+33)`, so up to `2^(31-BITS)` terms fit four
`u64` lanes (`u32 x u32 -> u64`, which baseline SSE2 vectorizes with `pmuludq`); the
coordinate is `lo + 2^32 hi - 2^BITS sum_j w_j` in `i128` (`< 2^97`). If any value is out
of range (for example a corrupted cache entry) or there are too many terms, `dot` falls
back to `dot_wide`: per-coordinate `i128` accumulation, exact for any `i32` values (terms
`< 2^95`, at most `2^31` of them `< 2^126`; for `A` the bound is `256 x 2^64 x 2^22 = 2^94 <
2^95`). Portable code only; no `target_feature` gate. An AVX2 path was not added: calling a
`#[target_feature]` function needs `unsafe`, which `agg-piop` forbids (`unsafe_code =
"forbid"`).

### 7.2 Exactness checks (measured)

- Unit tests against the old code kept as references (`public_stream.rs` tests):
  `sparse_t` on 200 random and 12 extreme challenges (`tau` entries of one sign at the start,
  the end, spread; `t1 = 1023`); `dot`, `dot_lanes`, `dot_wide` against the `Ext2` sum on
  random weights and all-`(p-1)` weights, random/boundary/`i32::MIN`/`i32::MAX` values, and
  the fast-path selection at each boundary; `key_weight` on random and `+-(q-1)/2` matrices;
  dedup against the serial map on 500 records over 13 keys (3 sharing key 0's `rho`, one
  differing in the last byte, one of a wrong length).
- Two genuine keys sharing `rho` (`agg_witness::gen::keypair_with_rho` builds the second key
  under the first one's matrix seed and signs through `ml-dsa`'s skDecode): two distinct keys
  in `prepare`, each record's `T` from its own `t1`, cached or not
  (`keys_sharing_rho_stay_distinct`); end to end (`agg-cli/tests/mode_a.rs`
  `keys_sharing_rho_verify_only_against_their_own_t1`): the aggregate verifies uncached, with
  only the first key cached and fully cached; naming the other key for a record is rejected,
  cached or not.
- `agg aggregate` with the old (`6803ce5`) and new (`dc395fc`) binaries on the same records:
  byte-identical aggregates at 256 (64 shared keys) and 1,024 (own keys) records under both
  relations (H-u 430,061 / 522,164 B; H-exact 430,089 / 522,171 B); the new verifier accepts
  the old aggregates uncached and cached, the old verifier accepts the new ones.
- `cargo test --release -p agg-piop -p agg-witness`: 184 passed, 0 failed, 7 ignored;
  `cargo test --release -p agg-cli`: 29 passed, 0 failed (commands in Section 7.4).

### 7.3 Timings (local, direction only)

Predictions (written before these runs, from memo 3.4; est.): `sparse_t` 3.7-4.5x and the
`T` product 3.0-3.8x single-thread; cloud 32K cached public_t 92 -> 25-33 ms, public_a 73.8
-> 20-30 ms, public_data 106 -> 65-80 ms.

Single thread (`RAYON_NUM_THREADS=1 cargo test --release -p agg-piop --lib kernel_microbench
-- --ignored --nocapture`, two runs, ns per call, measured):

| Kernel | Before | After | Ratio |
| --- | ---: | ---: | ---: |
| `sparse_t` (`k tau 256` additions) | 22,657 / 23,667 | 5,897 / 6,689 | 3.5-3.8x |
| `T` inner product, 1,024 terms (`Ext2` per term -> `i128` per coordinate) | 5,073 / 5,290 | 1,167 / 1,237 | 4.3-4.4x |
| `T` inner product, `u32` lanes on `c t1` (final) | 5,073 / 5,290 | 601 / 636 | 8.3-8.4x |
| per record: `sparse_t` + inner product | 28,021 / 28,374 | 6,264 / 6,402 | 4.4-4.5x |
| `key_weight`, 512 matrices round robin (`i128` per coordinate) | 21,668 / 20,838 | 4,990 / 5,025 | 4.2-4.3x |
| `key_weight`, `u32` lanes (final) | 21,668 / 20,838 | 2,943 / 2,933 | 7.1-7.4x |

Step 2 on 8,192 synthetic distinct keys, all threads (`prepare_profile`, medians of 5 after a
warmup, measured): parallel dedup 1.5 ms vs the serial map 6.4-7.2 ms; record stream 1.5 ms;
`public_records_binding` 7.4-10.0 ms. The first (cold) `prepare` with a full cache took
92 ms against 15-19 ms warm (before the slim cached keys): the per-key copies are fresh
memory, and on this WSL2 kernel parallel first-touch page faults dominate. That is why
the cached `StreamKey` now holds only the cache entry.

`agg verify` at 8K (8,192 own keys, H-u, `Slim`, 863,597 B aggregate made by the old
binary; key cache filled by one earlier run; old and new binaries alternated; medians,
measured; Flock and `matrices` are untouched, their spread is load):

| Round, binary, runs (load average) | public_data | public_t | public_a | sum of the three | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1, cached, before, 5 (5-7) | 108.7 | 32.7 | 25.1 | 167.4 | 3,855 |
| 1, cached, `u32` lanes without the slim cached keys, 5 (5-7) | 98.9 | 8.0 | 5.5 | 118.2 | 4,285 |
| 2, cached, before, 5 (7-8, spikes) | 124.5 | 105.9 | 56.9 | 285.4 | 5,403 |
| 2, cached, after (`dc395fc`), 5 (7-8) | 23.5 | 16.4 | 7.8 | 46.9 | 4,316 |
| 3, cached, before, 7 (9-14) | 125.7 | 53.0 | 42.9 | 215.4 | 4,826 |
| 3, cached, after (`dc395fc`), 7 (9-14) | 30.2 | 17.6 | 12.6 | 61.9 | 4,962 |
| 3, uncached, before, 3 (9-14) | 103.9 | 31.8 | 93.3 | 226.4 | 4,089 |
| 3, uncached, after (`dc395fc`), 3 (9-14) | 106.8 | 7.7 | 75.2 | 206.5 | 3,969 |

Each cell is the median over the runs of that row ("sum" is the median of the per-run sums).
Round 1 isolates the kernels: public_t and public_a drop 4-5x, public_data only 10%; round
2 adds the slim cached keys, which take public_data from about 100-125 ms to 23-30 ms.

The cloud reference for 8K cached is 188.6 ms total (`bench-cloud.md`, H-u, `8da7538`); the
local totals are dominated by the portable Flock step and are not comparable. Cached, the
three public steps drop 3.5-6x locally (rounds 2 and 3). Uncached public_data does not move: it still
decodes and copies every key into fresh memory and hashes `tr`, and public_a is
ExpandA plus 16 inverse NTTs per key.

Prediction for the cloud at 32K cached (est., not measured): public_t 92 -> 20-30 ms,
public_a 74 -> 15-25 ms (floor: streaming 512 MiB of cached `A`), public_data 106 -> 30-60
ms (the cold-memory share is likely smaller on a native Linux VM than on WSL2), so cached
verify 424 -> about 280-320 ms. To be measured with the next cloud matrix.

### 7.4 Commands

```sh
W=/home/jipengzhang/code/aggmldsa-wt/verifier-kernels
RAYON_NUM_THREADS=1 taskset -c 0,1,3-15 cargo test --release -p agg-piop --lib kernel_microbench -- --ignored --nocapture
taskset -c 0,1,3-15 cargo test --release -p agg-piop --lib prepare_profile -- --ignored --nocapture
A=target/release/agg   # old binary: the same build at 6803ce5
taskset -c 0,1,3-15 $A keygen-batch --count 8192 --seed 1 --out keys8k.bin
taskset -c 0,1,3-15 $A sign-batch --keys keys8k.bin --count 8192 --seed 1 --out rec8k.bin
taskset -c 0,1,3-15 $A aggregate --records rec8k.bin --out agg8k-hu.bin
taskset -c 0,1,3-15 $A verify --records rec8k.bin --aggregate agg8k-hu.bin --cache k.cache   # twice: fill, then cached
taskset -c 0,1,3-15 cargo test --release -p agg-piop -p agg-witness
taskset -c 0,1,3-15 cargo test --release -p agg-cli
```

Open: an uncached public_data with fewer copies (keep the packed `t1` and unpack per record,
or borrow `pk` from the inputs); a packed 23-bit `A` cache (memo 3.4 (b)); AVX2 kernels
would need a crate outside the `unsafe_code = "forbid"` rule.

## 8. Cloud profile at 32K after the Stage B merges (`184374f`; bench-cloud.md Phase 10b)

| Field | Value |
| --- | --- |
| Status | informative; **measured on the cloud VM** (one session, 2026-10-26 23:10-23:45 UTC, VM `agg-bench-c4d-prof`, c4d-highmem-16, AMD EPYC 9B45, 8 cores x 2 SMT, rustc 1.99.1). Timed runs are `agg bench` 1 warmup + 3 (sweep: + 2; other relations: + 1), so direction-grade, not the paper's medians of 5. Lever savings in 8.4 are estimates. |
| Revisions | timed runs: code `main` `184374f` (tied-part v5, wire v4, fork-join K22, H-FIPS L10, verifier kernels); profile runs: branch `cloud-profile` `db4a38d` = `184374f` + opt-in `AGG_PHASE_STAMPS` (stderr wall-clock stamps at each prover step end; the aggregate is byte-identical, checked with `cmp`) |
| Build | CLMUL (`RUSTFLAGS='-C target-feature=+pclmulqdq'`, `cargo build --release --locked -p agg-cli`), 16 threads, Slim, span on, Flock pool 8 (default) unless stated |
| Raw logs | top-level `ref/notes/overnight-20261026/cloud-profile.md` (per-run numbers and predictions); raw files in that session's scratchpad (not committed) |

Method: `AGG_PHASE_STAMPS=1 agg aggregate` in a fresh process (32K: 5 runs, 3 of them with every
trace of Section 1 on; 8K: 2 runs at 16 threads, 1 at `RAYON_NUM_THREADS=1`) beside a sampler of
`/proc/stat` every 100 ms (all 16 vCPUs; `scripts/cpusample.py`; medians by
`scripts/phase_profile.py`). The stamps give the step boundaries in wall-clock time, so the
busy share of each step is the time-weighted mean of the samples inside it. "Busy" counts SMT
siblings: 100% busy is not twice the throughput of 50%.

### 8.1 Timed results at 32K (H-u unless stated; in-process `agg bench`)

| Run | Prove median (min) | Flock proofs / fork-join wall | Verify uncached / cached | Prover VmHWM | Aggregate |
| --- | ---: | ---: | ---: | ---: | ---: |
| pool 8 (default), 1+3 | **12,745.2 (12,713.5) ms** | 7,822.9 / 7,823.1 ms | 392.0 / 163.3 ms | 30,905 MiB | 1,469,874 B (44.9 B/sig) |
| pool 0 (no fork: Flock proofs on 16 threads, then the PIOP), 1+2 | 14,257.2 (14,216.7) ms | 4,310.7 alone / - | 390.0 / 162.8 ms | 27,067 MiB | same |
| pool 4, 1+2 | 13,852.3 (13,841.7) ms | 8,922.9 / 8,923.1 | 387.1 / 162.1 ms | 30,654 MiB | same |
| pool 6, 1+2 | 12,777.3 (12,773.5) ms | 7,842.0 / 7,842.2 | 389.4 / 162.5 ms | 30,183 MiB | same |
| pool 12, 1+2 | 13,149.1 (13,117.6) ms | 7,837.6 / 8,164.6 | 384.6 / 161.7 ms | 31,286 MiB | same |
| H-exact, 1+1 | 13,015.8 ms | 7,938.9 / 7,939.2 | 392.4 / 162.1 ms | 30,681 MiB | 1,469,861 B |
| H-FIPS, 1+1 | 13,714.3 ms | 8,285.1 / 8,288.1 | 391.4 / 171.9 ms | 31,542 MiB | 1,470,050 B |
| H-u reference right after H-FIPS, 1+1 | 12,778.6 ms | 7,851.5 / 7,851.7 | 389.1 / 161.7 ms | 30,744 MiB | 1,469,874 B |

Fresh processes (main): `agg aggregate` 13,098.4 ms, max RSS 25,187 MiB (`time -v`; the bench
tool's VmHWM of 30.9 GiB also holds the record set and allocator retention across runs);
`agg verify` uncached 449.9 ms (wall 0.53 s, 382 MiB), cached 220.2 ms (wall 0.98 s with the
583 MB cache file read, 1,350 MiB).

Derived: fork-join -1,512 ms against pool 0; tied-part v5 alone (pool 0) -470 ms against the v4
prover of Phase 10 m0 (14,727.4 ms, another VM). The Flock proofs take 7.82-7.84 s on 6, 8 or 12
threads beside the PIOP but 4.31 s alone on 16, and the PIOP (lookup + stages) 4.97 s alone but
6.1-8.0 s beside the Flock pool (pool 4 to 12): the fork section is bound by the shared machine, not by the
split. H-exact +237 ms and H-FIPS +936 ms over the back-to-back H-u run. Against the 12 s target:
0.75 s to go (H-u).

### 8.2 Per-phase wall time and CPU utilization at 32K (medians of 5 fresh-process runs)

| Step (between stamps) | Wall ms (range) | Busy % | Idle vCPU-s |
| --- | ---: | ---: | ---: |
| instance (ExpandA, mu, c, T) | 540.9 (539-545) | 82.6 | 1.51 |
| witness and G_main tables | 731.1 (729-732) | 96.8 | 0.37 |
| Flock records | 50.5 | 66.0 | 0.27 |
| Akita setup (fresh process; 98 ms warm in `agg bench`) | 260.6 (260-269) | 19.4 | 3.36 |
| event 1 G_main commits beside the Flock event 2 | 640.6 (612-664) | 90.9 | 0.94 |
| Flock event 2 after G_main: witness build (~630 ms, 89-90%) then sponge commit (~1,360 ms, 87-88%) | 1,993.0 (1,984-2,009) | 88.5 | 3.66 |
| events 3-5: matrices, round 2 | 215.1 | 78.0 | 0.76 |
| fork, PIOP side: parity tables | 137.9 | 90.6 | 0.21 |
| fork, PIOP side: lookup | 3,425.4 (3,391-3,460) | 85.3 | 8.08 |
| fork, PIOP side: tau, pre-stage, stage 1 | 608.7 | 81.4 | 1.81 |
| fork, PIOP side: stage 2 | 876.1 | 91.2 | 1.24 |
| fork, PIOP side: stage 3 | 2,377.5 | 90.2 | 3.72 |
| fork: Flock-only tail after stage 3 | 460.4 (423-572) | 62.6 | 2.76 |
| fork-join section (sum of the fork rows) | 7,878.7 (7,840-7,926) | 85.8 | 17.90 |
| event 13: Akita opening | 734.2 (728-749) | 62.5 | 4.40 |
| serialize | 12.5 | 6.2 | 0.19 |
| **total** | **13,057.6 (13,004-13,120)** | **84.2** | **33.09** |

The prove is throughput-bound: everything but the fresh-process Akita setup runs at 78-97% of
the 16 vCPUs. The only low-utilization windows are the Akita opening (0.73 s at 62%) and the
Flock-only tail of the fork (0.46 s at 63%, the PIOP branch ends first). Filling idle cores
can therefore win at most about 0.4-0.6 s (est.: the two windows at 100% busy); the larger levers
must remove work or memory traffic.

Flock branch at 32K (3 traced runs, ms): **witness rebuild 2,052-2,070** (`prove_span` calls
`build` + `reattach` again because the event-2 witness is dropped after the commit, K14), parity
chain 411-427, zerocheck rounds 1 / 2 / 3+ 390-458 / 394-412 / 691-734, union partial folds ~339,
bit-claim banks ~314, `open_merged` 2,649-2,688 (Ligerito recursive prover 2,035-2,248); sponge
instance proven 5,797-5,856 after its start. Event 2: witness build 1,257-1,264 + sponge commit
1,356-1,369 (lane fill 180-192, NTT 746-748, Merkle 428-429). Lookup: trees 589-634, sessions
2,803-2,810, stored layers 4,236.5 MiB.

### 8.3 Parallel speedup per step at 8K (16 threads, mean of 2, vs `RAYON_NUM_THREADS=1`)

| Step | 16 threads | 1 thread | Speedup |
| --- | ---: | ---: | ---: |
| instance | 135.2 | 1,123.3 | 8.3x |
| witness and G_main tables | 182.7 | 1,708.0 | 9.3x |
| event 1 G_main commits | 177.1 | 953.7 | 5.4x |
| Flock event 2: witness build / sponge commit | 322.8 / 308.6 | 957.7 / 3,278.3 | **3.0x** / 10.6x |
| events 3-5 round 2 | 100.3 | 714.4 | 7.1x |
| Flock proofs (16 thr: 8-thread pool beside the PIOP) | 2,261.3 | 6,852.1 | 3.0x |
| - of which witness rebuild | 347.5 | 924.2 | **2.7x** |
| lookup (beside the Flock pool) | 1,131.9 | 4,471.8 | 4.0x |
| stage 1 / 2 / 3 (beside the Flock pool) | 197.6 / 174.9 / 654.6 | 665.6 / 952.6 / 2,811.2 | 3.4x / 5.4x / 4.3x |
| Akita opening | 342.0 | 1,553.3 | 4.5x |
| total (ms) | 3,745.4 | 26,650.5 | 7.1x |

The fork section does 15.75 s of one-thread work (Flock proofs 6.85 + PIOP 8.90 s) in 2.26 s
(7.0x). The sponge witness build scales only 2.7-3.0x although it keeps 89-90% of the vCPUs busy
at 32K (8.2): it is memory-bound (est.), and it runs twice per prove.

### 8.4 Levers for the 12 s target (from this profile; savings at 32K are estimates)

| # | Lever | Est. saving | Prover-only | Evidence (8.1-8.3) |
| --- | --- | ---: | --- | --- |
| W1 | Keep the event-2 sponge witness (`SlotBuffers` and input bits) in the prover state and hand it to the proof, instead of `build` + `reattach` in `prove_span` (`crates/agg-cli/src/mode_a.rs`); amends K14's drop-and-rebuild for the span path | **0.4-1.0 s** (the Flock branch loses 2.05 s and becomes shorter than the PIOP branch, 7.4 s beside Flock / 4.97 s alone; the PIOP gets the freed bandwidth) | yes (same witness, byte-identical aggregate) | rebuild 2,052-2,070 ms of the 7.86-7.93 s Flock branch; 2.7x scaling; RSS est. +0 to +2 GiB (the rebuilt witness is alive at the prover peak anyway) |
| W2 | Akita opening: parallelize the serial part (P7) | 0.1-0.25 s | yes (upstream Akita) | 734 ms at 62.5% busy, 4.5x scaling |
| W3 | Rebalance the fork after W1: with a shorter Flock branch, the Flock pool can shrink (6) so the PIOP gets more threads | 0-0.3 s | yes | pool 6 = pool 8 today (12.78 vs 12.75 s); the PIOP branch becomes critical after W1 |
| W4 | Event-2 witness build: fewer passes over memory (build the bits in the commit's lane layout) | 0.2-0.5 s | yes | 1,257-1,264 ms on the critical path at 89% busy, 3.0x scaling |
| W5 | Lookup P2b / stage-3 P4 (Section 6) | 0.2-0.5 s under the fork (half of the alone saving, est.) | yes | lookup 3.43 s beside Flock, 2.63 s alone |
| W6 | Fresh-process Akita setup (one-time init 260 ms at 19% busy; 98 ms warm) | 0.1-0.17 s in a fresh process only | yes | not in `agg bench` numbers |

W1 alone is the best single bet for 12 s (12.75 - 0.4..1.0 = 11.75-12.35 s, est.); W1 + W2 + W3
reach 12 s by these estimates. P3 of Section 6 (overlap the Flock event 2 with the tables) is
worth little now: the tables already run at 96.8% busy. A global pool of 8 threads stays
refuted (Phase 10 (1)); the Flock pool sizes 6 and 8 are equivalent.

Plain default build (no `+pclmulqdq`) at 8K, one run: 752.1 s (Flock commit 548.8 s, Flock
proofs 202.5 s, 119% CPU), aggregate byte-identical to the CLMUL one; unchanged from Phase 10
(2b) (769.1 s). It stays unusable for Flock until `flock-field`'s scalar `F128` multiply
dispatches at run time.

Commands (on the VM; `R=rec32k.bin`):

```sh
RAYON_NUM_THREADS=16 [AGG_FLOCK_THREADS=k] agg bench --count 32768 --warmup 1 --reps 3 --relation h-u
python3 scripts/cpusample.py STOP 0.1 > run.cpu &      # /proc/stat busy share every 100 ms until STOP exists
PCS_TRACE=1 LINCHECK_TRACE=1 AGG_TIED_TRACE=1 AGG_STAGE_TRACE=1 FLOCK_ZC_TIMING=1 FLOCK_COMMIT_TIMING=1 \
FLOCK_LIG_TIMING=1 AGG_PHASE_STAMPS=1 RAYON_NUM_THREADS=16 agg aggregate --records $R --out a.bin 2> run.err
touch STOP; python3 scripts/phase_profile.py run   # run.cpu + run.err (several runs: medians); per-step wall time and busy share
```
