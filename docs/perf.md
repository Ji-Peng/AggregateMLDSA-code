# Performance measurements

Build and run instructions: [README](../README.md). CLI options and benchmark semantics:
[CLI](cli.md). Local measurements on the shared i7-11700K are direction only.
Cloud measurements identify their machine. Figures marked "est." are estimates.

## 7. Verifier kernels

Scope: the cached verifier's public-data steps. The kernels are exact: the transcript, the aggregate bytes and every accept/reject decision are
unchanged. Local numbers: i7-11700K under WSL2, CPUs 0,1,3-15, shared with
other agents' builds and provers (load average 4-14 during the runs), default portable build
(no CLMUL, so the Flock step is 3.5-6.6 s here and dominates the local totals);
**direction only**, not benchmarks.

`SplitWeights::dot::<BITS>`: every weight is stored as the four 32-bit halves of its two
canonical `F_p` coordinates; with biased values `b_j = v_j + 2^BITS in [0, 2^(BITS+1))`
each product `half * b_j` is below `2^(BITS+33)`, so up to `2^(31-BITS)` terms fit four
`u64` lanes (`u32 x u32 -> u64`, which baseline SSE2 vectorizes with `pmuludq`); the
coordinate is `lo + 2^32 hi - 2^BITS sum_j w_j` in `i128` (`< 2^97`). If any value is out
of range (for example a corrupted cache entry) or there are too many terms, `dot` falls
back to `dot_wide`: per-coordinate `i128` accumulation, exact for any `i32` values (terms
`< 2^95`, at most `2^31` of them `< 2^126`; for `A` the bound is `256 x 2^64 x 2^22 = 2^94 <
2^95`). Portable code only; no `target_feature` gate. Calling a
`#[target_feature]` function needs `unsafe`, which `agg-piop` forbids (`unsafe_code =
"forbid"`).

### 7.3 Timings (local, direction only)

Single thread (`RAYON_NUM_THREADS=1 cargo test --release -p agg-piop --lib kernel_microbench
-- --ignored --nocapture`, two runs, ns per call, measured):

| Kernel | Before | After | Ratio |
| --- | ---: | ---: | ---: |
| `sparse_t` (`k tau 256` additions) | 22,657 / 23,667 | 5,897 / 6,689 | 3.5-3.8x |
| `T` inner product, `u32` lanes on `c t1` (final) | 5,073 / 5,290 | 601 / 636 | 8.3-8.4x |
| per record: `sparse_t` + inner product | 28,021 / 28,374 | 6,264 / 6,402 | 4.4-4.5x |
| `key_weight`, `u32` lanes (final) | 21,668 / 20,838 | 2,943 / 2,933 | 7.1-7.4x |

Step 2 on 8,192 synthetic distinct keys, all threads (`prepare_profile`, medians of 5 after a
warmup, measured): parallel dedup 1.5 ms vs the serial map 6.4-7.2 ms; record stream 1.5 ms;
`public_records_binding` 7.4-10.0 ms. `agg verify` at 8K (8,192 own keys, H-u, `Slim`, 863,597 B aggregate; the cache is filled before timing. The binaries run alternately; times are
measured medians. Flock and `matrices` use the same code):

| Round, binary, runs (load average) | public_data | public_t | public_a | sum of the three | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3, cached, before, 7 (9-14) | 125.7 | 53.0 | 42.9 | 215.4 | 4,826 |
| 3, cached, after, 7 (9-14) | 30.2 | 17.6 | 12.6 | 61.9 | 4,962 |
| 3, uncached, before, 3 (9-14) | 103.9 | 31.8 | 93.3 | 226.4 | 4,089 |
| 3, uncached, after, 3 (9-14) | 106.8 | 7.7 | 75.2 | 206.5 | 3,969 |

Each cell is the median over the runs of that row ("sum" is the median of the per-run sums).
## 8. Per-step profile with phase stamps

`AGG_PHASE_STAMPS=1` makes the prover print `phase-stamp <step> <unix ns>` on stderr at the end
of every prover step. `scripts/cpusample.py` samples the busy share of all vCPUs from
`/proc/stat`, and `scripts/phase_profile.py` aligns both into per-step wall time and busy
share (medians over several runs). "Busy" counts SMT siblings: 100% busy is not twice the
throughput of 50%.

```sh
python3 scripts/cpusample.py STOP 0.1 > run.cpu &      # busy share every 100 ms until STOP exists
AGG_PHASE_STAMPS=1 agg aggregate --records rec32k.bin --out a.bin 2> run.err
touch STOP; python3 scripts/phase_profile.py run      # run.cpu + run.err: per-step wall time and busy share
```

## 9. Flock witness retention

A batch with exactly one tied part keeps its `TiedCommitted` in `mode_a::PartState`
from event 2 to the join. This covers a span and a one-part batch under the default
part size. Batches with more than one part drop and rebuild witness buffers.
The deterministic witness and statement give identical aggregate bytes.

## 10. Flock scalar arithmetic

The arithmetic and its exactness arguments are in
[flock-field/PATCH.md](../vendor/flock-field/PATCH.md).

## 11. Parallel setup and buffer release

`PublicData::from_instance` builds distinct keys and records in parallel, preserving
first-appearance order. `aggregate_with` builds the Akita `TableCommitter` on its own
thread and passes it to `prove_with_tables_using`.

Multi-part batches drop event-2 witness buffers on a detached thread. `PcsProver::open`
holds `Arc` clones of the six `CommitmentHandle`s across `batched_prove` and releases
them on a detached thread. The committed sources are shared, so this causes no copy.
Stage-3 buffers and finished lookup layers also drop on detached threads.
`AGG_STAGE_TRACE` reports drop times. Parity tables are built in the fork.

## 12. Cloud measurements at 32K

`c4d-highmem-16`, CLMUL build, 16 threads, H-u unless stated. All numbers in this section are
measured on the VM.

### 12.1 Results

- Matrix, `agg bench --warmup 1 --reps 5`: H-u 32K **11,336.3 ms** with `AGG_FLOCK_THREADS=6`
  and 11,688.6 ms with an 8-thread pool; H-exact 11,581.8 / 11,933.3 ms; H-FIPS 12,306.9 /
  12,625.0 ms; H-u 8K 3,285.3 / 3,423.9 ms, 1K 912.4 / 930.4 ms. `FLOCK_SIMD=portable` 32K:
  15,169.6 / 14,968.2 ms. Bytes unchanged (1,469,874 B = 44.9 B/sig at 32K); verify 380.9 /
  156.8 ms in process, 426.9 / 202.3 ms in a fresh process. Prover peak: tool VmHWM 31,177 MiB
  (H-u), 32,605 MiB (H-FIPS); fresh `agg aggregate` 28,669 MiB (H-u), because the witness stays live until the join.
- Flock pool (H-u 32K): 4 threads 11,262.0 ms (1+5), 6 threads 11,336.3 ms (1+5), 8 threads
  11,688.6 ms (1+5), 10 threads 11,944.5 ms (1+2). The PIOP branch is critical at 6 or
  more Flock threads (fork wall 6,850.6 ms against Flock proofs 6,279.5 ms at 6) and the Flock
  branch at 4 (6,693.7 against 6,656.9 ms). The default is 3/8 of the global pool, rounded up.

### 12.2 Where the 11.34 s go (H-u, pool 6, median run, ms)

Outside the fork 4,485.7: instance 495.7, witness and tables 697.4, Flock records 49.0, setup
35.6, Flock commit 2,376.7 (event 1, 578.0, runs beside it), round 2 190.3, opening 615.4.
Fork wall 6,850.6, PIOP branch critical: lookup 3,471.8, stages 1/2/3 508.2 / 768.1 / 1,977.4;
the Flock proofs take 6,279.5 on 6 threads beside it.

## 13. Flock pool (local 4K, direction only)

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
from the PIOP. The default is documented in [CLI](cli.md).

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

## 15. ML-DSA SIMD dispatch

Runtime backend selection is documented in [CLI](cli.md).

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
  taskset -c 0,1,3-15 cargo test --release -p agg-cli --test all simd_backends -- --ignored
```

The pipeline executable has independent AVX2 finish composite gates. It compares one finish call with three C hint calls, and
nine finish calls with two scalar calls (the existing 4.5x speedup budget).
These grouped comparisons retain the shared harness's strict 1.00 threshold
and repeat policy. Diagnostic per-step passes cannot override their failures.
Both finish budgets still fail: the dedicated benchmark reports normalized
ratios 1.071/1.069 against C and 1.210/1.197 against scalar. Composite ExpandA
is 1.107/1.085 times C on AVX2 and 0.460/0.456 on AVX-512. The performance gate remains blocked.

AVX-512 SHAKE128 passes twice (0.994/0.998), while the marginal AVX2 SHAKE128
result is 1.003 before the harness repeat and 0.981 after it.
The AVX2 NTT P2 Plain/Mont row passes at 0.998 against its 1.02 budget.
All six 1K aggregate comparisons against main pass. A serial
`scripts/test.sh --release` run with `RUST_TEST_THREADS=1` passes 419 workspace tests
and 466 vendored Flock tests, including the timing self-check.

Local A/B measurements (2026-10-30, one warmup and two repetitions per arm):

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

All six aggregate byte comparisons pass. The Flock-record flags (+30.8% for
distinct-key AVX-512, +8.8%/+14.8% for shared-key AVX2/AVX-512) prevent a clean
no-regression verdict. All memory columns are within 5% of main.

A separate shared-key AVX2 Flock-record comparison uses a fresh main arm
and the same one-warmup/two-repetition protocol. Main measures 81.8 ms and AVX2
87.3 ms, a 1.067 ratio; the >5% flag persists. The other repeated AVX2 timings
are instance 30.6 ms (main 67.0), witness/tables 41.1 ms (524.8), prove
10670.4 ms (11320.5), verify uncached 470.6 ms (504.9), and cached 490.2 ms
(482.3). These results do not establish a clean integrated performance gate.

## 16. Relation sanity measurements

Relation sanity runs (revision `f526097`, CLMUL build `RUSTFLAGS="-C
target-feature=+pclmulqdq"`, `agg bench --count 1024 --warmup 1 --reps 1 --flock-profile slim`,
span on, CPUs 0,1,3-15 with `RAYON_NUM_THREADS=3` while a catalog generation ran with 3
workers: single noisy runs, not benchmarks):

| Relation | Aggregate | PIOP rounds and claims | Work nonces | Akita opening | Prove | Stage 2 / stage 3 | Verify uncached / cached |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H-u | 522,138 B (509.9 B/sig) | 4,736 B | 391 B | 60,994 B | 2,901 ms | 67.5 / 181.1 ms | 352 / 296 ms |
| H-exact | 522,154 B (509.9 B/sig) | 4,736 B | 391 B | 61,010 B | 2,981 ms | 60.9 / 198.0 ms | 316 / 290 ms |

Every other part is equal (lookup 21,344 B, Flock proofs 398,518 B). The PIOP bytes agree
with `16 (76 + 11 d_main)` at `d_main = 20` for both relations (8 + 31 and 7 + 32 claims).

## 17. H-FIPS mode-B measurements

Measured, mode B (`AGG_RELATION=... scripts/bench.sh --bench mode_b -- 1024 8192 --warmup 1
--reps 2`, default build, CPUs 0,1,3-15 of this box, medians of 2 after 1 warmup;
the PIOP part only, no Flock. Mode A adds Flock's Keccak argument, the same for every relation,
but there the `G_main` commitment overlaps the Flock commitments, so the mode-A delta may
differ from the mode-B delta in either direction; it has not been measured):

| Code | Relation | Prove 1K (s) | Prove 8K (s) | Prover peak RSS 8K (MiB) | Aggregate without public bits 8K (B) |
| --- | --- | ---: | ---: | ---: | ---: |
| layout | H-u (two runs) | 1.564, 1.676 | 4.367, 4.335 | 3,587, 3,552 | 357,148 |
| layout | H-FIPS (affine `r0`, two runs) | 1.729, 1.760 | 4.885, 4.964 | 3,915, 3,909 | 357,309 |
| spot check | H-u | 1.658 | | | |
| spot check | H-FIPS | 1.696 | | | |

With this layout H-FIPS costs about +0.13 s (+8%) at 1K and +0.57 s (+13%) at 8K
over H-u in this box's mode B (means of the two interleaved runs; the run-to-run spread is about
0.1 s), +0.34 GiB of prover RSS (+9.5%) and +161 B of aggregate at 8K (0.02 B per signature:
5 more stage claims, Akita opening of 43 polynomials). The 1K delta is inside the run-to-run
noise: the spot check (1 warmup + 2 reps) measured +0.04 s, with
systematic step deltas of about +85 ms (stage 3 +47, round 1 +20, tables +12 ms) and an
opening that varied by about 90 ms.
Stage-3 trace at 8K (`AGG_STAGE_TRACE=1`, `tests/all/h_fips.rs` `h_fips_stage_trace`):
the main `eq(r^(2))` group's messages take 178 ms against 93 ms for H-u (the extra terms of
`F1`, `F2`, `F45` and the reductions), the `F7` main group 55 ms, the ring's `w_A` group 129
against 103 ms (H-exact's `a_z` lane); stage 2's `F3` group 75 ms against 39 ms for H-u's
leaves.

16K uses the `d_sig = 14` rows (the `l16f` grouped row for `2^14` and
the width-16 `G_2` row at (14, 13); the bench asserts that the verifier accepts). Measured,
mode B, same box and cores, `-- 16384 --warmup 1 --reps 1`, run four times interleaved
(H-u, H-FIPS, H-u, H-FIPS):

| Relation | Prove 16K (s) | Prover peak RSS (MiB) | Aggregate without public bits (B) | Committed elements per signature |
| --- | ---: | ---: | ---: | ---: |
| H-u | 6.826, 7.230 | 6,981, 6,945 | 621,859 | 12,424.5 |
| H-FIPS | 7.858, 8.459 | 7,657, 7,638 | 621,901 | 16,521.5 |

So +1.13 s (+16%, means of two runs each; pairwise +1.03 and +1.23 s), +0.67 GiB of prover
RSS (+9.8%) and +42 B of aggregate (PIOP +80 B, Akita opening -38 B). The step deltas (means)
are stage 3 +0.47 s, tables +0.25 s, opening +0.19 s, round 1 +0.18 s and stage 2 +0.07 s; the
lookup does not change.


## 18. Parity-chain tied parts

Machine: i7-11700K, CLMUL build `RUSTFLAGS="-C target-feature=+pclmulqdq"`,
7 threads, CPUs 0,1,3–15, `RAYON_NUM_THREADS=7`. The host is shared (load 6–18).
Times are direction only. Tied-part times are medians of 5 after 1 warmup
(8,192 records: 3). `8@13/4` denotes eight live records over four full-segment geometries.

| Part | Proof (header and chain / sponge) | Prove (ms) | Verify (ms) |
| --- | --- | --- | --- |
| `8@13/4` | 150,758 B (1,046 / 149,712) | 153.4 / 150.0 / 196.4 | 26.5 / 24.6 / 32.3 |
| 2,048 (`d` 11) | 248,230 B (902 / 247,328) | 802.9 / 752.5 / 1,034.2 | 28.4 / 29.7 / 39.9 |
| 8,192 (`d` 13) | 282,654 B (974 / 281,680) | 1,959.8 / 3,295.9 | 33.8 / 44.0 |

The chain timing test on dense random bits measures 32.8 / 59.8 / 178.5 ms at
`d` 11 / 13 / the 32K span `13/2`; verifier weight and claim evaluation takes 7 ms.
At full height, the bit claim's full column bit-bank costs 174–222 ms at 8,192
records under load; block statistics and the `W'` fold cost about 60 ms more.

Fresh-process RSS (`/usr/bin/time -v`, two runs, load 0.9–4.5):

| N | Prover RSS (MiB) | With `MALLOC_MMAP_THRESHOLD_=1048576` (MiB) | Verifier RSS (MiB) | Aggregate wall (s) |
| ---: | --- | --- | --- | --- |
| 2,048 | 2,011 / 2,007 | — | 117 / 116 | 3.29 / 3.27 |
| 8,192 | 6,542 / 6,484 | 5,764 / 5,761 | 143 / 144 | 7.84 / 8.02 |

The allocator retains freed heap during the lookup. glibc raises its mmap threshold
dynamically up to 32 MiB after large frees. A fixed threshold reduces retained heap;
its time cost is unmeasured.

## 19. Standalone sponge measurements

Bench `benches/keccak_sponge.rs` (harness-free; `AGG_KECCAK_N`, `AGG_KECCAK_PROFILE`,
`AGG_KECCAK_WARMUP` (default 1), `AGG_KECCAK_REPS` (default 5)): random `(mu, w1Encode)` records, one proof,
medians of 5 timed repetitions after 1 discarded warmup run.
"prove" includes witness generation and packing; peak heap is Flock's counting allocator
(the `PeakAlloc` of Flock's `blake3_proof`/`sha2_proof` benches) over one prove; VmHWM is
the process peak RSS. Machine: i7-11700K, 7 threads (`taskset -c 0,1,3-15`), turbo and
hyper-threading off, shared with other jobs (load average 5 to 8 during the runs, so
times are upper bounds).

Commit-then-prove split (same machine and threads, 3 repetitions at 1,170
records, Fast): `avx512` + `pclmulqdq` build, commit 0.046 s, prove
including witness and
both commits 0.94 s, verify 13.0 ms, peak heap 990 MiB, proof 316,446 B plus a 4,096 B
commitment; default build, commit 42.0 s of 120.8 s.

Commands:

```sh
taskset -c 0,1,3-15 cargo bench -p agg-keccak --bench keccak_sponge
AGG_KECCAK_PROFILE=secure taskset -c 0,1,3-15 cargo bench -p agg-keccak --bench keccak_sponge
# CLMUL/AVX-512 build (as scripts/bench.sh AVX512=1):
RUSTFLAGS="-C target-cpu=x86-64-v4 -C target-feature=+vpclmulqdq,+pclmulqdq" CARGO_TARGET_DIR=target/avx512 \
  taskset -c 0,1,3-15 cargo bench -p agg-keccak --features avx512 --bench keccak_sponge
taskset -c 0,1,3-15 cargo test --release -p agg-keccak
taskset -c 0,1,3-15 cargo test --release -p agg-keccak --lib print_gate_costs -- --ignored --nocapture
```


## 20. Lookup measurements

Standalone (`timing_at_1024_and_8192_signatures` with `AGG_LOOKUP_SIGS=1024,2048`, the
CLMUL build, 7 threads, warmup plus five runs, medians;
load average 6.6 to 6.8 including the run and another agent's tests):

| `N` | `X` | Prove | per signature | trees | sessions | bottom session | verify | proof | stored layers | peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 1 | 221.4 ms | 216 us | 26.0 | 196.3 | 65.1 | 0.51 ms | 16,240 B | 136 MiB | 255 MiB |
| 1,024 | 2 | 312.0 ms | 305 us | 44.5 | 265.3 | 59.0 | 0.57 ms | 21,232 B | 272 MiB | 464 MiB |
| 2,048 | 1 | 401.5 ms | 196 us | 48.6 | 354.2 | 118.6 | 0.55 ms | 17,424 B | 268 MiB | 576 MiB |
| 2,048 | 2 | 617.1 ms | 301 us | 97.6 | 519.4 | 122.8 | 0.61 ms | 22,544 B | 536 MiB | 756 MiB |

The second challenge costs 1.41x at 1,024 and 1.54x at 2,048, not 2x, because the bottom
session is shared.

Integrated (mode B): committed cells per signature 14,672, 12,932 and 12,658 at 256, 1,024
and 2,048 signatures. Stage 3 takes about 0.12 ms per signature locally (CLMUL build): its
four 16-bit columns are generic `Ints` lanes.

Measured (lookup timing test, 2 challenges, interleaved runs of the generic and the
dedicated paths in one process, `AGG_LOOKUP_AB=1`; local, shared machine, direction only):

| `N` | threads | load average | generic | dedicated | trees | `s = D` | `s = D - 1` | `s = D - 2` |
| ---: | ---: | --- | ---: | ---: | --- | --- | --- | --- |
| 8,192 | 3 | 7.6 to 8.3 | 4,519.4 ms | 2,823.4 ms | 798.9 -> 564.5 | 840.2 -> 593.3 | 1,275.0 -> 475.9 | 667.6 -> 555.9 |

In mode A at 8,192 signatures (CLMUL
build, 3 threads, `agg bench --warmup 1 --reps 3`, load average 4.6 to 5.7, revisions
`be398d8` and `9a53721`), event 7 measures 3,822.5 to 2,270.0 ms. The remaining cost is
close to the arithmetic of the protocol: the tree build (about 0.5 s at 8,192 on 3 threads),
the bottom session (one pass per round over `2^(D-1)` integer pairs) and the layer sessions
`D - 2, D - 3` (two trees each). Levers not taken, with their estimates: session `D - 2`
on the integer lanes `e_1 .. e_4` (a 5 x 5 bilinear form; the first round gets cheaper but
the first fold dearer, net about 15% of that session plus 1 GiB of stored layers at 8,192,
estimate); the small-value rounds 1 and 2 of the bottom session from the integers (about
20% of that session, estimate); multi-way SIMD for the folds.
