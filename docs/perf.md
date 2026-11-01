# Prover performance: profile and speed levers (local, direction only)

| Field | Value |
| --- | --- |
| Status | informative; a local profile, **direction only**. No number here is a benchmark result: the development box was shared with other agents' builds (load average 2 to 10), and figures marked "est." are estimates. |
| Profiled code | the code before the proof-of-work schedule and the speed levers of Section 0 |

## 0. What happened since the profile

The ranking of Section 6 drove the prover speed work below (`docs/piop.md` Sections 6 and 14 for
the PIOP side). All changes are prover-only; aggregates stayed byte-identical except for the
proof-of-work nonces of wire version 2/3.

| Lever (Section 6) | Status |
| --- | --- |
| F1 runtime-dispatched Flock x86 SIMD kernels in the vendored `flock-core` | done (`vendor/flock-core/PATCH.md` Section 3) |
| F2 parallel Flock witness inputs, stripe transpose, `input_bits` | done |
| F3 sponge instance and bridge union proven side by side | moot: tied-part version 5 has no bridge union (K21) |
| P5 stage parity tables built while the Flock span is proven | done |
| lookup tree build (fast product, parallel fills); stage-3 main-stack reduction on one lane | done (related to P2/P4, not the coded bottom session of P2) |
| instance `T` reused in the witness; vectorized `public_t` | done |
| P1 overlap the Flock span with the lookup and stages | done (deviation K22): the Flock proofs at a join after stage 3, proven on their own pool (`AGG_FLOCK_THREADS`, default ceil(3/8 of the global pool)); composition argument in `docs/soundness.md` 12. Cloud dry run at 32K: -1.11 s (pool 16) and -1.15 s (pool 8). Local 8K (measured, direction): -0.93 / -1.03 s, prover peak +0.8 to +1.1 GiB; pool sweep at 8K: 4 / 8 / 15 threads 6.03 / 6.37 / 6.76 s, a rerun: 4 / 8 threads 6.34 / 6.35 s against 7.31 s without the fork |
| W1 keep the event-2 Flock witness of a one-part batch until the join (Section 9) | done; cloud 32K: -0.66 s (Section 12) |
| B inline Flock's scalar GF(2^128) multiply via a vendored `flock-field` (Section 10) | done; cloud 32K: -0.26 s prover, Flock verifier 29.0 -> 22.6 ms (Section 12) |
| A move the one-vCPU windows off the serial path (Section 11) | done; cloud 32K: -0.21 s (Section 12) |
| P2 coded bottom lookup session, P2b, P3, P4, P6, P7, F4, F5 | open |

Outcome on the cloud VM after F1 to P5 (2 runs at `N = 32,768`, so not a median of 5): prove
23.74 s to 14.62 s, Flock parts 13.8 s to 7.57 s, lookup 3.86 s to 2.53 s, stage 3 1.80 s to
1.48 s. With P1, W1, B and A the 12 s target is met: H-u 32K proves in 11.34 s with a 6-thread Flock
pool and 11.69 s with the default pool of 8 (medians of 5, Section 12).

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
| `target/prof-v4` | `-C target-cpu=x86-64-v4 -C target-feature=+vpclmulqdq,+pclmulqdq,+gfni,+avx512vbmi` | Keccak-only bench (`agg-keccak --bench keccak_sponge`, no `avx512` feature) to measure Flock's AVX-512 kernels; mode A cannot run in this build (Akita self-check) |
| scratch `gfbench` | soft / `+pclmulqdq` / `+pclmulqdq,+avx512f,+vpclmulqdq` | GF(2^128) multiply microbench on `flock-field` 83f963e |


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
| event 2, Flock witness build `honest_parts` + `statement` (diff) | ~890 | ~2,464 | **2.8** | | `pack_slot`, **`SpongeRecord::input_bits` (serial)**, `ChainGate::apply_ab/honest_z` |
| event 2, sponge commit: lane fill / NTT (rate 1/4) / Merkle | 87 / **1,087** / 180 | 393 / 5,972 / 1,036 | 4.5 / 5.5 / 5.8 | 5.7 | **`kernels::portable::butterfly_fused_2layer` 32%**, `ghash_mul_karatsuba_barrett` 25%, BLAKE3 avx512 12% |
| events 3-5: matrices, round 2 | 302 | 1,496 | 5.0 | 4.1 (with rebuild) | software `keccak_p` in `agg_bridge::seed::derive_column_into`, `level1_sums` |
| event 6, witness rebuild + `reattach` (diff) | ~725 | ~2,410 | **3.3** | 1.0-2.1 for ~0.3 s | **`input_bits` 71% at ~1.3 threads**, `pack_slot` |
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
| Profile: Flock bench `prove_fast` (rate 1/2) vs our `Slim` (rate 1/4: 2x encode, more queries) | | | 32K cloud run: `Fast` 22.12 s vs `Slim` 23.74 s |
| Witness built twice, with a serial `input_bits` loop and serial `statement` (c~ by SHAKE per record) | ~2.9 s | witness + lincheck 0.74 s | Section 3: 2.8x / 3.3x scaling, perf dips |
| Bridge union, bridge rows/commit, tie (absent from Flock's bench) | ~1.0 s | 0 | Section 3 |
| Threads: Flock 8 (physical cores; its doc reports SMT hurts its CLMUL kernels), ours 16 on 8 cores x 2 SMT | | | not measured on the VM |

The flags alone explain about half of the gap (2.2x on the sponge instance), the duplicated
serial witness work and the bridge union most of the rest.

## 5. Overlap: file/line evidence

- **Flock span prove vs lookup/stages.** In data the two are independent: the Flock prove
  reads only `sd` (event 3), the matrices, `fs_sigma` (event 5) and `Y_sigma` (event 6), and
  the PIOP reads none of the Flock proof. The profiled transcript absorbed the Flock proofs
  (`FLOCK_PROOFS_SITE`) before the lookup's `X`, so every later challenge depended on them;
  the fork-join transcript (deviation K22, lever P1) absorbs them after stage 3.
- **Event-2 Flock commit vs `G_main` commit: already concurrent** (`prover.rs:568-577`,
  `crate::pcs::overlap`, a scoped OS thread, `pcs.rs:238-261`); in the 32K cloud breakdown the
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

Baseline (32K cloud run): prove 23.74 s = Flock 13.82 (commit 4.40, prove 9.41) + lookup 3.86
+ stages
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
| ref | `Fast` instead of `Slim` | 1.6 s (measured) | no (verifier-visible profile; +6.6 B/sig) | 32K cloud run |

F1 + F2 + F3 bring the Flock span from 13.8 s to about 6-7 s (est.), at the <= 6 s target
with F4.

### PIOP, lookup and the rest

| # | Lever | Est. saving | Prover-only | Evidence |
| --- | --- | ---: | --- | --- |
| P1 | Overlap the Flock span prove with the lookup and stages: absorb the Flock proofs after stage 3 (implemented, deviation K22; wire version 4 kept, nothing released since K21) | 1.5-3 s before F1, about 1-2 s after F1 (est.); measured: cloud dry run at 32K -1.11 s (Flock pool 16, +2.6 GiB) and -1.15 s (pool 8, +0.5 GiB); local 8K -0.93 / -1.03 s with pool 8, +0.8 to +1.1 GiB | **no** (transcript change) | Section 5 |
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

## 7. Verifier kernels

Scope: the cached verifier's public-data steps, which were 272 ms of the 424.0 ms cached
verify at 32K (public_data 106.3 + public_t 91.9 + public_a 73.8 ms; measured on the cloud
VM). All changes are verifier-only
and exact: the transcript, the aggregate bytes and every accept/reject decision are
unchanged (checks below). Local numbers: i7-11700K under WSL2, CPUs 0,1,3-15, shared with
other agents' builds and provers (load average 4-14 during the runs), default portable build
(no CLMUL, so the Flock step is 3.5-6.6 s here and dominates the local totals);
**direction only**, not benchmarks.

### 7.1 Changes

| Step | Before | After | Exactness argument |
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
  (`keys_sharing_rho_stay_distinct`); end to end (`agg-cli/tests/all/mode_a.rs`
  `keys_sharing_rho_verify_only_against_their_own_t1`): the aggregate verifies uncached, with
  only the first key cached and fully cached; naming the other key for a record is rejected,
  cached or not.
- `agg aggregate` with the old and new binaries on the same records:
  byte-identical aggregates at 256 (64 shared keys) and 1,024 (own keys) records under both
  relations (H-u 430,061 / 522,164 B; H-exact 430,089 / 522,171 B); the new verifier accepts
  the old aggregates uncached and cached, the old verifier accepts the new ones.
- `cargo test --release -p agg-piop -p agg-witness`: 184 passed, 0 failed, 7 ignored;
  `cargo test --release -p agg-cli`: 29 passed, 0 failed (commands in Section 7.4).

### 7.3 Timings (local, direction only)

Predictions (written before these runs; est.): `sparse_t` 3.7-4.5x and the
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
| 2, cached, after, 5 (7-8) | 23.5 | 16.4 | 7.8 | 46.9 | 4,316 |
| 3, cached, before, 7 (9-14) | 125.7 | 53.0 | 42.9 | 215.4 | 4,826 |
| 3, cached, after, 7 (9-14) | 30.2 | 17.6 | 12.6 | 61.9 | 4,962 |
| 3, uncached, before, 3 (9-14) | 103.9 | 31.8 | 93.3 | 226.4 | 4,089 |
| 3, uncached, after, 3 (9-14) | 106.8 | 7.7 | 75.2 | 206.5 | 3,969 |

Each cell is the median over the runs of that row ("sum" is the median of the per-run sums).
Round 1 isolates the kernels: public_t and public_a drop 4-5x, public_data only 10%; round
2 adds the slim cached keys, which take public_data from about 100-125 ms to 23-30 ms.

The cloud reference for 8K cached is 188.6 ms total (H-u); the
local totals are dominated by the portable Flock step and are not comparable. Cached, the
three public steps drop 3.5-6x locally (rounds 2 and 3). Uncached public_data does not move: it still
decodes and copies every key into fresh memory and hashes `tr`, and public_a is
ExpandA plus 16 inverse NTTs per key.

Prediction for the cloud at 32K cached (est., not measured): public_t 92 -> 20-30 ms,
public_a 74 -> 15-25 ms (floor: streaming 512 MiB of cached `A`), public_data 106 -> 30-60
ms (the cold-memory share is likely smaller on a native Linux VM than on WSL2), so cached
verify 424 -> about 280-320 ms.

### 7.4 Commands

```sh
RAYON_NUM_THREADS=1 taskset -c 0,1,3-15 cargo test --release -p agg-piop --lib kernel_microbench -- --ignored --nocapture
taskset -c 0,1,3-15 cargo test --release -p agg-piop --lib prepare_profile -- --ignored --nocapture
A=target/release/agg   # old binary: the same build of the code before 7.1
taskset -c 0,1,3-15 $A keygen-batch --count 8192 --seed 1 --out keys8k.bin
taskset -c 0,1,3-15 $A sign-batch --keys keys8k.bin --count 8192 --seed 1 --out rec8k.bin
taskset -c 0,1,3-15 $A aggregate --records rec8k.bin --out agg8k-hu.bin
taskset -c 0,1,3-15 $A verify --records rec8k.bin --aggregate agg8k-hu.bin --cache k.cache   # twice: fill, then cached
taskset -c 0,1,3-15 cargo test --release -p agg-piop -p agg-witness
taskset -c 0,1,3-15 cargo test --release -p agg-cli
```

Open: an uncached public_data with fewer copies (keep the packed `t1` and unpack per record,
or borrow `pk` from the inputs); a packed 23-bit `A` cache; AVX2 kernels
would need a crate outside the `unsafe_code = "forbid"` rule.

## 8. Per-step profile with phase stamps

`AGG_PHASE_STAMPS=1` makes the prover print `phase-stamp <step> <unix ns>` on stderr at the end
of every prover step. `scripts/cpusample.py` samples the busy share of all vCPUs from
`/proc/stat`, and `scripts/phase_profile.py` aligns both into per-step wall time and busy
share (medians over several runs). "Busy" counts SMT siblings: 100% busy is not twice the
throughput of 50%.

The `opening` step covers only the finish part of the Akita opening. Its prepare part runs on
the PIOP branch, so it appears in the `opening-prepare` to `join` interval and in the
fork-join wall when the fork runs; `AGG_OPENING_THREADS` sizes the finish pool only. Opening
times quoted from older profiles span both parts and are not directly comparable with a run
that splits them.

```sh
python3 scripts/cpusample.py STOP 0.1 > run.cpu &      # busy share every 100 ms until STOP exists
AGG_PHASE_STAMPS=1 agg aggregate --records rec32k.bin --out a.bin 2> run.err
touch STOP; python3 scripts/phase_profile.py run      # run.cpu + run.err: per-step wall time and busy share
```

## 9. Lever W1: keep the event-2 Flock witness until the join

### 9.1 Change

Without W1, every Flock part drops its sponge buffers, input bits and statement after event 2
and rebuilds them in the Flock branch of the fork-join (K14). At 32K on the cloud VM the
rebuild is 2,052-2,070 ms of the 7.86-7.93 s Flock branch (measured, 3 traced runs), and the
drop at the end of event 2 is a serial munmap of 166-197 ms on one vCPU (measured). With W1 a
batch with exactly one tied part keeps its whole
`TiedCommitted` in `mode_a::PartState` from event 2 to the join and proves it without a
rebuild. That covers the span (every batch of `N >= 2^14` with whole-segment parts) and the
one-part batch at `N <= 2^13` (default part size), the second added so that the change is
visible locally. Batches of more than one part keep the drop-and-rebuild. Prover-only; the aggregate is byte-identical by construction.

### 9.2 Checks (measured, local)

- `cmp` against the code without W1 (and against `AGG_FLOCK_REBUILD=1`): identical at 256 and 1K
  (H-u and H-exact), 2K, 8K, and 2K in two parts (`--part-log2 10`); `agg verify` accepts.

### 9.3 Local 8K A/B (measured, direction only)

`agg bench --count 8192 --warmup 1 --reps 2`, CLMUL build, `RAYON_NUM_THREADS=7`,
`taskset -c 0,1,3-15`, main / W1 / main / W1 interleaved, other agents active (load 1.8 to
6.6). Predictions written first: Flock proofs -0.6 to -1.2 s, prove
-0.2 to -0.8 s, peak RSS -0.3 to +0.5 GiB.

| Binary | Prove median of 2 | Flock proofs (pool of 4) | Fork-join wall | Outside the fork | Prover peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| without W1 | 5,957.6 / 6,018.7 ms | 3,051.8 / 3,047.9 | 3,052.5 / 3,112.3 | 2,782.0 / 2,819.4 | 8,657 / 8,739 MiB |
| W1 | 5,712.1 / 5,888.7 ms | 2,480.2 / 2,539.8 | 2,950.5 / 3,023.6 | 2,755.6 / 2,833.2 | 8,779 / 8,871 MiB |

Flock proofs -0.51 to -0.57 s (just below the predicted range). Prove -0.25 / -0.13 s (low
edge of the prediction). Peak +122 / +132 MiB (inside). The fork wall drops only 0.09-0.10 s:
at 8K on this box the PIOP branch (about 2.95 s) becomes the critical one as soon as the Flock
branch loses its rebuild. This is the "soft floor" case of the model, where only
the Flock-only tail disappears.

### 9.4 What it means at 32K (est.)

The local result supports the floor end of the estimated range. Estimate for
`c4d-highmem-16` at 32K: fork -0.46 s (Flock-only tail, measured 460.4 ms) to -0.99 s
(uniform-rate model), plus the event-2 munmap -0.16 to -0.19 s, so prove -0.6 to -1.2 s,
point -0.7 s (est.). From 12.75 s (32K bench median without W1) that is about 11.6-12.1 s
(est.). With W1 the PIOP branch is critical, so the Flock pool can shrink (`AGG_FLOCK_THREADS`
4 / 6 / 8 to compare), and the lookup and stage levers act directly on the critical path.
RSS at 32K: about 4.2-4.5 GiB of witness (est.) stays resident from event 2 to the join; the
bench VmHWM headroom to 32 GiB is 1.82 GiB.

## 10. Lever B: inline Flock's scalar GF(2^128) multiply

### 10.1 Change

`flock-field` 83f963e declares its x86 multiplies (`ghash_mul_karatsuba_barrett`,
`ghash_mul_unreduced_x86`, `ghash_square_x86`) `#[target_feature(enable = "pclmulqdq,sse4.1")]`
without `#[inline]`, so in our CLMUL build (`+pclmulqdq`, no LTO) every `F128` multiply in
flock-core was an out-of-line call (3.87% self time in a local 8K profile; microbench 5.5 ns per
multiply against 2.7 ns inlined). Thin LTO plus `+sse4.1` was the first candidate. Measured
locally,
a global `+sse4.1` slows our lookup and stage 3 by 6-7% (it alone, without LTO, gives the same
loss), which cancels the Flock gain once W1 makes the PIOP branch critical. The code therefore
vendors `flock-field` (`vendor/flock-field/PATCH.md`): `#[inline]` on the three functions,
`pclmulqdq` as their only target feature, SSE2 lane extracts. The build used here was
`-C target-feature=+pclmulqdq`; the shipped baseline is now
`-C target-cpu=x86-64-v3 -C target-feature=+pclmulqdq` (README "SIMD policy"). The
multiply inlines everywhere in the prover (pclmul instruction lines in `agg`: 461 ->
6,903; 18 out-of-line Karatsuba calls remain, all in verifier functions). Exact
arithmetic; the proof bytes are unchanged.

### 10.2 Checks (measured, local)

- Aggregates identical to main's CLMUL build at 256 and 1,024 records, H-u and H-exact; `agg
  verify` accepts with both binaries.
- Tests under `+pclmulqdq`, `--release`: workspace 367 passed, 0 failed, 19 ignored (5m27s);
  vendored `flock-core` lib with the vendored `flock-field` patched in (`--config
  'patch."https://github.com/succinctlabs/flock".flock-field.path=...'`) 460 passed, 0 failed,
  6 ignored; `flock-field` 29 passed. Default build (no flags): `cargo check --workspace
  --all-targets` clean, `flock-field` tests 28 passed.
- 8K, `agg bench --count 8192 --warmup 1 --reps 2`, 7 threads, base and new interleaved in
  rotated order (load 4.1-4.3, another job active; direction only):

| Mode | Line | base (median) | new (median) |
| --- | --- | ---: | ---: |
| fork-join (default), 4 runs each | prove | 5,715 ms | 5,698 ms |
| | Flock proofs (4-thread pool) | 2,474 ms | 2,191 ms (-11.4%) |
| | fork-join wall | 2,945 ms | 2,915 ms |
| | zerocheck rounds 3+ tail | 220 ms | 171 ms |
| | user CPU of the bench | 129.7 s | 126.1 s (-2.8%) |
| `AGG_FLOCK_THREADS=0`, 3 runs each | prove | 6,899 ms | 6,796 ms |
| | Flock parts (commit + proofs) | 2,897 ms | 2,746 ms (-5.2%) |
| | event 7 lookup / stage 3 | 1,236 / 684 ms | 1,225 / 692 ms |
| both | verify uncached | 275.7 ms | 273.5 ms |

- Rejected variants (same A/B method): thin LTO + `+sse4.1` (Flock
  proofs -0.26 to -0.30 s, but lookup +38 to +88 ms and stage 3 +42 ms, prove +0.23 s in one
  set, -0.04 s in another); `+sse4.1` alone (lookup +70 ms, stage 3 +50 ms, no Flock gain);
  vendored inline with `+sse4.1` (Flock gain, same PIOP loss).

### 10.3 What it means at 32K (est.)

At 8K on this box the PIOP branch is critical, so the Flock gain shows as freed CPU only.
On `c4d-highmem-16` at 32K, with W1, the model gives prove -0.14 to -0.36 s (est.) from 2-5
vCPU-s freed. An estimate of -0.3 to -0.8 s assumes the Flock branch critical, which does not
hold with W1. Bytes are identical, and the verifier changes by about 0 to -1%. Prediction for
a cloud A/B (1 warmup + 3, back to back, same flags): Flock proofs -0.3 to -0.7 s, prove -0.14
to -0.36 s, verify within +-2%, 1,469,874 B at 32K H-u.

## 11. Lever A: move the one-vCPU windows off the serial path

### 11.1 Change

The 32K sampler logs of the cloud VM (5 fresh runs, 100 ms `/proc/stat` samples) show three
windows on the serial path where one vCPU works, mostly in the kernel, while the other 15 wait
(measured on the VM): (a) the "Akita setup" step, 97.7 ms in the bench
and 260.6 ms fresh, mostly the serial `PublicData::from_instance` loop; (b) the event-2 witness
unmap, 166-197 ms; (c) the release of the six committed sources inside Akita's opening,
132-163 ms. Changes (prover-only, no transcript change):

- (a) `PublicData::from_instance` builds the distinct keys and the records in parallel (same
  keys, same first-appearance order; unit test against `PublicData::prepare`). `aggregate_with`
  builds the Akita `TableCommitter` on its own thread at entry (it depends on `n` and the
  relation only) and hands it to the new `prove_with_tables_using`.
- (b) The event-2 witness is dropped on a detached thread. Since W1 it is dropped there only
  in multi-part non-span batches; the span and one-part batches keep it until the join.
- (c) `PcsProver::open` holds `Arc` clones of the six `CommitmentHandle`s across
  `batched_prove` and releases them on a detached thread afterwards. Akita `e023b6a` shares the
  committed source by `Arc` inside the opening anyway (`opaque/openings.rs:90`,
  `opaque/eor.rs:175`), so the clones trigger no copy.
- On the PIOP branch, which W1 made critical: the stage-3 `ProverTables` buffers and each
  finished lookup layer are dropped on detached threads.
- Measured with `AGG_PARITY_EARLY`, since removed: build the stages' parity tables beside
  the round-2 commitment instead of in the fork.

`AGG_STAGE_TRACE` prints the time of each detached drop.

### 11.2 Checks (measured, local)

- `cmp` against the code without lever A (CLMUL build): identical at 256 and 1K (H-u and
  H-exact), 1K
  over 100 shared keys, 2K in two parts (`--part-log2 10`, the multi-part drop path) and 8K, for
  the default, `AGG_SCHED_SERIAL=1` and both parity placements; 16 fresh 8K runs of the A/B
  below are identical too. `agg verify` accepts.
- Tests under `+pclmulqdq`, `--release`, `-p agg-pcs-config -p agg-piop -p agg-cli`: 193
  passed, 0 failed, 12 ignored (5m35s).
- Detached drop times, `AGG_STAGE_TRACE=1`, 8K fresh, THP off for the process (`prctl` wrapper,
  as on the cloud VM), 2 runs: opening sources 15.6 / 15.9 ms, lookup layers 21.9 / 34.9 ms in
  24 drops, stage tables 3.0 / 3.1 ms. These are the windows taken off the critical path at 8K
  locally.

### 11.3 Local 8K A/B (measured, direction only)

CLMUL build, `RAYON_NUM_THREADS=7`, `taskset -c 0,1,3-15`, arms interleaved in rotated order,
no other job on the box (load is the runs themselves). Arms: main (without lever A); new
(parity beside round 2); ser (new with `AGG_SCHED_SERIAL=1`, the old schedule); pif (new with
the parity tables in the fork, the shipped default). Predictions were written first.

`agg bench --count 8192 --warmup 1 --reps 2`, 4 rounds, medians of the 4 per-run medians:

| Arm | Prove | Akita setup | Round 2 | Fork-join wall | Outside the fork | Opening |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| main | 5,731.4 ms | 27.6 | 185.5 | 2,970.1 | 2,731.1 | 793.3 |
| ser | 5,654.1 ms | 26.5 | 183.7 | 2,952.5 | 2,690.3 | 799.4 |
| new (parity early) | 5,713.6 ms | 11.6 | 252.6 | 2,909.4 | 2,787.5 | 786.5 |
| pif (shipped default) | 5,809.9 ms | 12.3 | 187.4 | 2,978.8 | 2,763.1 | 796.8 |

Fresh `agg aggregate` of the same 8K records, 4 runs per arm (THP on), medians: Akita setup
124.6 (main) / 141.5 (ser) -> 15.0 (new) / 14.5 (pif) ms; total minus the instance and table
steps 5,616.9 / 5,779.6 / 5,716.1 / 5,668.3 ms.

The final binary against main, `agg bench` as above, 3 more rounds in rotated order (the
box ran about 0.3 s slower in these rounds for both arms): prove 6,146.7 / 6,061.5 / 6,314.2 ms
(main) against 5,997.5 / 5,975.4 / 6,027.3 ms (lever A), paired median -149 ms, faster in 3 of 3
pairs; fork wall -81 to -143 ms, lookup -66 to -187 ms, opening -15 to -69 ms; prover peak RSS
8,781-8,883 MiB (main) against 8,746-8,782 MiB.

Reading: the setup line moves as predicted (fresh -110 to -125 ms, warm bench -15 ms; prediction
P2 5-20 ms: inside). In the first 4 rounds the totals do not resolve: main and ser run the same
schedule and differ by 77 ms (bench) and 163 ms (fresh), which is the noise floor; the -149 ms
of the last 3 pairs is of the same size, so it is direction only. On this box (THP on, 2 MiB pages) the
unmaps are short, so (c) and the PIOP drops are 15-50 ms at 8K, below the noise; on the VM they
are larger (4 KiB pages, 132-163 ms for (c) in the sampler logs). Parity tables beside round 2:
round 2 +65 ms, fork wall -38 to -61 ms; no net change locally, so it stays opt-in.

### 11.4 What it means at 32K (est.)

`agg bench` on `c4d-highmem-16`, from the sampler windows: (a) -0.07
to -0.08 s, (c) -0.12 to -0.15 s (partly a stopwatch-boundary gain: the release moves after the
prove, where serialization and the untimed drops sit), (b) about 0 (W1 keeps the witness at
32K), PIOP-branch drops 0 to -0.10 s (unmeasured at 32K; locally 25-38 ms at 8K, about 4x at
32K on THP-on pages). Total -0.19 to -0.33 s (est.); fresh process about -0.15 s more from (a).

## 12. Cloud measurement of W1, B and A at 32K

`c4d-highmem-16`, CLMUL build, 16 threads, H-u unless stated. All numbers in this section are
measured on the VM.

### 12.1 Results

- Matrix, `agg bench --warmup 1 --reps 5`: H-u 32K **11,336.3 ms** with `AGG_FLOCK_THREADS=6`
  and 11,688.6 ms with the default pool (8); H-exact 11,581.8 / 11,933.3 ms; H-FIPS 12,306.9 /
  12,625.0 ms; H-u 8K 3,285.3 / 3,423.9 ms, 1K 912.4 / 930.4 ms. `FLOCK_SIMD=portable` 32K:
  15,169.6 / 14,968.2 ms. Bytes unchanged (1,469,874 B = 44.9 B/sig at 32K); verify 380.9 /
  156.8 ms in process, 426.9 / 202.3 ms in a fresh process. Prover peak: tool VmHWM 31,177 MiB
  (H-u), 32,605 MiB (H-FIPS); fresh `agg aggregate` 28,669 MiB (H-u), up 3.4 GiB from
  the schedule without W1, B and A because W1 keeps the witness.
- Lever decomposition, same VM, back to back (1 warmup + 3, pool 8): without the three levers
  12,775.9 ms; + W1 12,112.9 ms; + B (main with `AGG_SCHED_SERIAL=1`) 11,855.8 ms; + A (main)
  11,680.4 / 11,604.4 ms. W1 -0.66 s, B -0.26 s, A -0.21 s (setup -66 ms, opening -123 ms);
  together -1.13 s (-8.9%). `AGG_FLOCK_REBUILD=1` on main: +0.65 s. All three inside the
  estimates of Sections 9.4, 10.3 and 11.4. `AGG_PARITY_EARLY=1` measured -47 ms, under the
  50 ms bar, but the switch has since been removed. Every arm's aggregate is byte-identical.
- Flock pool (H-u 32K): 4 threads 11,262.0 ms (1+5), 6 threads 11,336.3 ms (1+5), 8 threads
  11,688.6 ms (1+5), 10 threads 11,944.5 ms (1+2). With W1 the PIOP branch is critical at 6 or
  more Flock threads (fork wall 6,850.6 ms against Flock proofs 6,279.5 ms at 6) and the Flock
  branch at 4 (6,693.7 against 6,656.9 ms). Suggested default: about 3/8 of the rayon pool (6 at
  16 threads) instead of half; not changed yet (a code change with its own test run).

### 12.2 Where the 11.34 s go (H-u, pool 6, median run, ms)

Outside the fork 4,485.7: instance 495.7, witness and tables 697.4, Flock records 49.0, setup
35.6, Flock commit 2,376.7 (event 1, 578.0, runs beside it), round 2 190.3, opening 615.4.
Fork wall 6,850.6, PIOP branch critical: lookup 3,471.8, stages 1/2/3 508.2 / 768.1 / 1,977.4;
the Flock proofs take 6,279.5 on 6 threads beside it.

### 12.3 Toward the 10 s stretch target (est.)

1.34 s to go from 11.34 s. Since the PIOP branch is critical, its levers now count at full
value: P2 / P2b (lookup, Section 6: 0.8-1.2 s + 0.3-0.6 s, estimated against the old 3.86 s
lookup; against today's 3.47 s inside the fork, est. 0.5-1.0 s) and P4 (stage 3, 0.2-0.4 s,
est.), each until the Flock branch (6.28 s at 6 threads) becomes critical again, at which point
the pool shifts back towards 8. On the serial path, the Flock commit (2.38 s, beside event 1)
is the largest item; P3 (start the event-2 build earlier) is 0.3-0.5 s (est., Section 6).
The 11.34 s already use the 6-thread pool; with the default pool (8) the starting point is
11.69 s, so the default change of 12.1 is the first 0.35 s (measured).

## 13. Flock pool after the scalar kernel work (local 4K, direction only)

WSL on an i7-11700K (8 cores, 16 threads, turbo and HT on), CLMUL build, H-u 4K,
`agg bench --count 4096 --keys 16 --warmup 1 --reps 2` per pool size, sizes alternated
(4, 6, 8, 8, 6, 4) in two passes. The host was shared with another session (load 4-16), so
runs that the load visibly disturbed (prove above 5 s) are left out. All numbers are local
and measured; they set a direction only, not a 32K figure.

| `AGG_FLOCK_THREADS` | prove, median of 2 (ms) | Flock proofs (ms) | fork-join wall (ms) |
|---:|---:|---:|---:|
| 4 | 3,540.4 / 3,064.6 / 3,174.0 | 882.8 / 902.6 / 979.0 | 1,275.4 / 1,148.0 / 1,294.7 |
| 6 | 3,363.0 / 2,926.0 | 799.3 / 795.2 | 1,203.9 / 1,216.7 |
| 8 | 3,339.2 / 3,272.9 / 3,247.8 | 873.2 / 847.2 / 917.3 | 1,406.5 / 1,274.4 / 1,373.2 |

At 4K the PIOP branch is the longer one at every size (the fork wall exceeds the Flock
proofs), so a smaller Flock pool helps the wall until the Flock branch slows down. On this shared
host 4 and 6 threads do not separate: 4 had the shortest single fork wall (1,148.0 ms) and
the prove times overlap (3,064.6-3,540.4 against 2,926.0-3,363.0 ms); 8 threads took cores
from the PIOP. The default is now 3/8 of the global pool rounded up (6 of 16), the value
Section 12.1 suggested. In that 32K sweep on main (4 / 6 / 8 threads: 11,262.0 / 11,336.3 /
11,688.6 ms) 4 threads were 74 ms (0.7%) faster than 6, within the run-to-run spread, so the
32K re-sweep below decides the default.

For the next cloud run: the scalar kernel work shortens the PIOP branch (lookup and stages)
more than the Flock proofs, so at 32K the Flock branch may become the critical one; sweep
`AGG_FLOCK_THREADS` = 4 / 6 / 8 at 32K again (1 warmup + 5) and move the default to the
winner.

## 14. Scalar prover work against main (local 1K and 4K, direction only)

Same box as Section 13 (shared, load 8-10 throughout). Main `44c882d` against `scalar-opt`
`e95fe80`, both CLMUL builds with their default pools (Flock 8 on main, 6 on the branch;
Akita opening 8 on the branch). `agg bench --keys 16 --warmup 1 --reps 2`, main and the
branch alternated, two passes. Proofs are byte-identical: the 1K gate (H-u, H-exact, H-u with
`FLOCK_SIMD=portable`) and the 4K H-u aggregates of both binaries compare equal (498,885 B).

| ms (median of 2) | main pass 1 / 2 | scalar-opt pass 1 / 2 | ratio |
| --- | ---: | ---: | ---: |
| 1K prove | 2,094.8 / 2,254.8 | 1,282.6 / 1,361.4 | 1.63x / 1.66x |
| 1K fork-join wall | 1,003.1 / 1,039.7 | 552.0 / 596.6 | 1.82x / 1.74x |
| 4K prove | 4,190.8 / 3,939.0 | 2,643.1 / 2,719.1 | 1.59x / 1.45x |
| 4K fork-join wall | 2,253.0 / 2,053.6 | 1,324.9 / 1,323.7 | 1.70x / 1.55x |
| 4K outside the fork (total - wall) | 1,888.6 / 1,872.7 | 1,301.6 / 1,335.5 | 1.45x / 1.40x |
| 4K prover peak RSS (MiB) | 4,226 / 4,265 | 4,345 / 4,362 | +3% |

One cold 4K aggregate per binary (`agg aggregate`, H-u, ms):

| Phase | main | scalar-opt | ratio |
| --- | ---: | ---: | ---: |
| instance | 115.1 | 22.1 | 5.2x |
| witness and `G_main` tables | 237.4 | 79.2 | 3.0x |
| Flock commit (event 2) | 683.7 | 492.6 | 1.39x |
| event 1 (beside the Flock commit) | 242.1 | 501.5 | runs beside it |
| events 3-5 | 206.9 | 143.5 | 1.44x |
| lookup (event 7) | 1,480.3 | 750.1 | 1.97x |
| stage 1 | 123.7 | 214.7 | 0.58x (overlap with the Flock tail) |
| stage 2 | 132.3 | 123.4 | 1.07x |
| stage 3 | 383.9 | 262.4 | 1.46x |
| Flock proofs (beside the PIOP) | 1,368.1 (8 thr.) | 925.0 (6 thr.) | 1.48x |
| fork-join wall | 2,174.1 | 1,351.9 | 1.61x |
| Akita opening (event 13) | 933.7 | 609.6 | 1.53x |
| total | 4,370.6 | 2,737.0 | 1.60x |

Stage 1 code is not slower: with no fork (`AGG_FLOCK_THREADS=0`) the branch's stage 1 is at
least as fast as main's. Since the lookup shrank 2x and the Flock proofs only 1.5x, stage 1
now runs beside the last part of the Flock proofs and shares the cores with them. The round-0
quartet-mass pass, which ran its widest segment on one thread, now splits segments over item
ranges and derives nested segments as marginals (mass pass 12-18 -> 7.3-7.6 ms without the
fork). The rest of the stage-1 excess moves with the pool split, not with stage-1 code; the
fork wall is set by the PIOP branch either way. In the two bench passes stage 1 was 189.6 /
125.3 ms on main and 175.4 / 179.7 ms on the branch.

Outside the fork, the opening pool (one thread per physical core), the flat instance `T`, the
stack-buffer digit decomposition and the lane-blocked level-0 Flock codeword gave 1.4-1.45x
in the bench. Event 1 now takes about as long as the Flock commit beside it (502 against
493 ms), so the two share the window and only less total work in it shortens it.

At 32K (est., from the 32K phases of Section 12.2 and the 4K ratios above): outside the fork
about 2.6-2.8 s instead of 4.49 s, the PIOP branch about 3.8 s and the Flock proofs 3.7-4.2 s
on 6 threads, so the Flock branch may be the critical one, and a total of about 6.5-7.0 s
instead of 11.34 s (1.6-1.75x, est.). The cloud matrix and the 4 / 6 / 8 pool sweep of
Section 13 decide.

## 15. ML-DSA SIMD dispatch

`AGG_MLDSA_SIMD=auto|scalar|avx2|avx512` selects the ML-DSA engine once per process.
The default, `auto`, selects the best supported tier. An invalid or unsupported forced
value fails at startup. Scalar is portable; AVX2 requires AVX2 and POPCNT; AVX-512 requires
AVX512F/BW/VL, AVX2 and POPCNT. VBMI optionally accelerates AVX-512 rejection sampling;
it is reported as `avx512+vbmi` in the `agg bench` header.

The engine covers public-instance expansion and hashes, challenge sampling, sparse products,
and witness transforms, CRT reconstruction, rounding, encoding and commitment hashes.
The streamed verifier uses it for public data, public T and uncached matrix expansion.
The Flock event-2 lane integration also uses these Keccak kernels. This setting is independent
of the Aerie field backend and the compile-time `avx512` feature. Aggregate and key-cache
bytes are unchanged across engines.

Run `scripts/simd-ab.sh` for the local comparison against a detached `main` checkout under
`target/main-wt`. It builds both binaries with `-C target-feature=+pclmulqdq`, compares 1K
H-u aggregates at one and sixteen threads, then measures 4K records over 4096 and 16 keys
with one warmup and two repetitions. Runtime gates use the shared benchmark lock and CPU 3;
the thread count is left at its default. Logs, revisions and the table are retained under
`target/simd-ab-logs`. Ratios above 1.05 are flagged for review, including RSS. These shared-host
measurements indicate local direction only; they do not replace the 32K cloud measurements.

The longer backend equality test covers H-u at 1024 records over 64 and 1024 keys,
and H-exact/H-FIPS at 256 records. It verifies uncached, fills a fresh cache, verifies a cache
hit, and compares aggregate and cache bytes across every available engine at 1/16 threads:

```sh
CARGO_BUILD_JOBS=3 RUSTFLAGS='-C target-feature=+pclmulqdq' \
  cargo test --release -p agg-cli --test all simd_backends -- --ignored
```

Kernel gates (`cargo bench -p agg-mldsa-simd --bench <name> -- --gate` with
`RUSTFLAGS='--cfg cref'` and `AGG_CREF_SRC` holding `dilithium/` and `Hybrid-SHA-3/`, one pinned
core) compare each kernel with its C original: pass at a ratio of at most 1.00, a ratio in
(1.00, 1.03] reruns and decides on the pooled ratio. The C references are built for the ISA of the
backend they are compared with (`x86-64-v3` for AVX2, `x86-64-v4` for AVX-512). Rows where the
Rust kernel does more than the C function are reported as INFO (the finish composites
against `poly_use_hint`) or get the missing C work added (`unpack_z` adds `poly_chknorm`). The
per-step finish rows are the C criterion for the finish.

On the i7-11700K every row passes. On Zen 5 (c4d, 2026-10-30, two runs) these rows stay
above C: AVX2 `rej_uniform` 1.18, AVX2 `use_hint` step 1.23, AVX-512 inverse NTT mod q 1.08
(the Rust kernel also brings its output to the canonical or centred range), and the P2 inverse
NTT at 1.23 of its 1.20 budget. The AVX-512 forward NTT is at 1.01. The end-to-end effect is
small (instance and witness together take 224 ms of the 8.49 s prove at 32K), and c4d runs the
AVX-512 backend, where only the inverse NTT rows apply.

Local A/B results (4096 records, CLMUL, CPU 3, default threads;
one warmup and two repetitions per arm):

Local direction only: N=4096, K=4096, H-u, CLMUL, CPU 3, default threads. Times in ms.
| Metric | main | scalar | avx2 | avx512 |
|---|---:|---:|---:|---:|
| instance | 523.3 | 407.7 (0.779x) | 126.8 (0.242x) | 93.1 (0.178x) |
| witness and tables | 496.8 | 166.8 (0.336x) | 48.4 (0.097x) | 40.5 (0.082x) |
| Flock records | 85.5 | 78.8 (0.922x) | 81.6 (0.954x) | 111.8 (1.308x) FLAG >5% |
| prove total | 11669.3 | 11062.1 (0.948x) | 10515.9 (0.901x) | 10538.5 (0.903x) |
| verify uncached | 809.9 | 722.5 (0.892x) | 543.6 (0.671x) | 522.2 (0.645x) |
| verify cached | 502.3 | 498.0 (0.991x) | 476.5 (0.949x) | 512.4 (1.020x) |
| public_data | 57.2 | 42.9 (0.750x) | 33.6 (0.588x) | 34.4 (0.601x) |
| public_t | 34.4 | 26.9 (0.781x) | 10.0 (0.292x) | 8.1 (0.236x) |
| public_a | 270.8 | 223.5 (0.825x) | 64.5 (0.238x) | 37.6 (0.139x) |
| prover RSS MiB | 3740.0 | 3568.0 (0.954x) | 3568.0 (0.954x) | 3568.0 (0.954x) |
| verifier RSS MiB | 2952.0 | 2678.0 (0.907x) | 2873.0 (0.973x) | 2678.0 (0.907x) |

Speedups (main/current; values above 1 are faster):
| Backend | Instance | Witness/tables | Verify uncached | Verify cached |
|---|---:|---:|---:|---:|
| scalar | 1.28x | 2.98x | 1.12x | 1.01x |
| avx2 | 4.13x | 10.26x | 1.49x | 1.05x |
| avx512 | 5.62x | 12.27x | 1.55x | 0.98x |

Local direction only: N=4096, K=16, H-u, CLMUL, CPU 3, default threads. Times in ms.
| Metric | main | scalar | avx2 | avx512 |
|---|---:|---:|---:|---:|
| instance | 69.3 | 55.9 (0.807x) | 31.0 (0.447x) | 28.9 (0.417x) |
| witness and tables | 516.7 | 172.7 (0.334x) | 47.8 (0.093x) | 36.4 (0.070x) |
| Flock records | 79.2 | 82.6 (1.043x) | 86.2 (1.088x) FLAG >5% | 90.9 (1.148x) FLAG >5% |
| prove total | 11294.7 | 10808.7 (0.957x) | 10543.2 (0.933x) | 10489.0 (0.929x) |
| verify uncached | 498.1 | 482.1 (0.968x) | 478.6 (0.961x) | 459.8 (0.923x) |
| verify cached | 496.2 | 485.5 (0.978x) | 462.4 (0.932x) | 457.6 (0.922x) |
| public_data | 25.0 | 21.7 (0.867x) | 20.1 (0.806x) | 19.3 (0.773x) |
| public_t | 26.0 | 25.6 (0.986x) | 8.8 (0.339x) | 7.7 (0.297x) |
| public_a | 1.1 | 0.9 (0.822x) | 0.3 (0.290x) | 0.2 (0.215x) |
| prover RSS MiB | 3339.0 | 3257.0 (0.975x) | 3405.0 (1.020x) | 3354.0 (1.004x) |
| verifier RSS MiB | 2535.0 | 2325.0 (0.917x) | 2599.0 (1.025x) | 2520.0 (0.994x) |

Speedups (main/current; values above 1 are faster):
| Backend | Instance | Witness/tables | Verify uncached | Verify cached |
|---|---:|---:|---:|---:|
| scalar | 1.24x | 2.99x | 1.03x | 1.02x |
| avx2 | 2.24x | 10.81x | 1.04x | 1.07x |
| avx512 | 2.40x | 14.20x | 1.08x | 1.08x |

The verifier public-step columns are from the last uncached run. CPU affinity makes the
default Rayon/opening pool one thread here; Flock runs without a fork. Shared-host local
ratios cannot establish the design estimates for 32K on sixteen cloud threads.
The script prints the deferred cloud commands without running them.

All six aggregate byte comparisons pass. The Flock-record flags (+30.8% for
distinct-key AVX-512, +8.8%/+14.8% for shared-key AVX2/AVX-512) prevent a clean
no-regression verdict. All memory columns are within 5% of main.

The shared-key AVX2 Flock-record comparison was repeated with a fresh main arm
and the same one-warmup/two-repetition protocol. Main measures 81.8 ms and AVX2
87.3 ms, a 1.067 ratio; the >5% flag persists. The other repeated AVX2 timings
are instance 30.6 ms (main 67.0), witness/tables 41.1 ms (524.8), prove
10670.4 ms (11320.5), verify uncached 470.6 ms (504.9), and cached 490.2 ms
(482.3). These results do not establish a clean integrated performance gate.

### 15.1 Cloud A/B at 32K (Zen 5, measured)

`c4d-highmem-16` (AMD EPYC 9B45), H-u 32K, 16 threads, `AGG_FLOCK_THREADS=6`, `+pclmulqdq`;
`agg bench --count 32768 --warmup 1 --reps 2`, two rounds alternated between the builds, mean
of the two medians. The baseline `6d79063` runs the scalar ML-DSA path; `cfc566b` adds the
ML-DSA SIMD dispatch (`auto`, AVX-512 on this VM) and the Flock commit levers of
`vendor/flock-core/PATCH.md`, so the Flock commit line mostly reflects the latter.

| ms | `6d79063` | `cfc566b` |
| --- | ---: | ---: |
| instance | 447.7 | 158.4 |
| witness and tables | 419.7 | 65.4 |
| event 1 | 625.1 | 649.0 |
| Flock commit | 2,305.1 | 1,673.2 |
| fork wall / Flock proofs | 5,689.9 / 5,681.1 | 5,611.8 / 5,611.6 |
| lookup | 2,544.5 | 2,503.7 |
| stage 3 | 1,774.7 | 1,691.6 |
| Akita opening | 619.2 | 654.7 |
| prove | 9,809.5 | 8,494.2 |
| verify uncached / cached | 378.4 / 154.3 | 148.8 / 131.4 |
| prover peak RSS (round 1) | 31,405 MiB | 30,122 MiB |

One-round runs of `cfc566b` with the engine forced: `AGG_MLDSA_SIMD=avx2` prove 8,570.0 ms,
`scalar` 8,795.0 ms. Aggregates (1,469,874 B) are byte-identical across the builds, and every
verifier run accepts.

## Flock pool size

The Flock fork pool takes `ceil(3 Rayon / 8)` workers by default, capped at the logical
CPUs; `AGG_FLOCK_THREADS` overrides the count and `0` runs Flock before the PIOP. Flock
and PIOP pools remain additive.

On WSL (i7-11700K, 16 logical CPUs), the 4K comparison of four against six Flock workers
alternated four launches with one warmup and one measured run each. Mean fork wall was
1,427.0 / 1,389.0 ms, Flock proofs 1,155.1 / 989.7 ms, PIOP branch 1,426.8 / 1,388.9 ms,
and total prove 2,725.1 / 2,723.3 ms (0.999x). These 4K results are directional; a 32K
cloud sweep remains open. One-thread 1K fresh aggregates matched at 1, 4, 16 and 32 Rayon
threads.
