# Artifact evaluation package: aggregating ML-DSA-44, ML-DSA-65 and ML-DSA-87 signatures

This directory rebuilds every measurement and table of the evaluation section
from the pinned source tree on a plain x86-64 Linux machine. It does not need
Slurm, root, or network access after `./run.sh fetch`.

The package builds one source tree **once** and measures proving, verification,
peak resident set size and aggregate byte composition across all three FIPS 204
parameter sets (**ML-DSA-44**, **ML-DSA-65** and **ML-DSA-87**) under the
relation **H-FIPS** and three run-time ISA tiers (`ifma`, `avx512`, `avx2`) at
batch sizes up to 32,768. The reference relation **H-exact** is available as an
opt-in (see [Relations](#relations)). It also runs the
mode-B public-bits aggregate (`agg-bench --bench mode_b`), the control and
scheduling ablations (including `AGG_LATE_JOIN=0` and `AGG_FLOCK_THREADS=0`),
the traced timeline run (`AGG_STAGE_TRACE`, `AGG_SCHED_TRACE`,
`AGG_TIED_TRACE`, `PCS_TRACE`, `LINCHECK_TRACE`, `FLOCK_ZC_TIMING`,
`AGG_PHASE_STAMPS`), and two external baselines: Aerie v0.2.0 on Falcon-512
and Falcon-1024, and the pq-crystals ML-DSA AVX2 batch verifier (`natver`,
`natver65`, `natver87`) for all three parameter sets.

## Quick start

```sh
cd artifact
./run.sh check            # tools, CPU features, sources, binaries
./run.sh fetch            # clone Aerie and pq-crystals/dilithium; verify SHAs
./run.sh build            # release agg, mode_b, Aerie, natver (ML-DSA-44, 65, 87)
./run.sh gates            # byte identity, acceptance, self-reports at 1K, per set
./run.sh quick --dry-run  # print the planned command lines without running them
./run.sh quick --smoke    # about five minutes of measurements after the build
./run.sh tables           # write results/LATEST/paper-tables/*.tex
```

`./run.sh all` runs `fetch`, `build`, `gates`, `quick` and `tables` in that
order. Add `--smoke` for the smallest run and `--full` to use the full matrix
instead of `quick`. Pass `--dry-run` (or `-n`) to `quick` or `full` to inspect
the planned command lines without executing them.

## Relations

Every default run (`gates`, `quick`, `full`, `tables`, `all`) proves and
verifies **H-FIPS**: the aggregate certifies that every extracted
$(\tilde c, z, h)$ is a FIPS 204 signature. **H-exact** proves the FIPS 204
bounds on $z$ and on the UseHint residual but not the hint itself. It is a
reference relation, slightly faster than H-FIPS (9--13% on 16C32T at 32K in the
reference measurements), and runs only on request:

```sh
./run.sh gates --relations h-fips,h-exact   # 1K gates of both relations
./run.sh quick --relations h-fips,h-exact   # adds H-exact to the quick-ours groups
./run.sh full  --relations h-fips,h-exact   # adds the relations groups (H-exact at 32K)
```

`--relations h-exact` runs only the groups that list H-exact. `./run.sh tables`
needs no flag: `tab-relations.tex` holds H-exact against H-FIPS when the
results contain H-exact cells, and a comment otherwise.

## Hardware and software

- x86-64 Linux built for x86-64-v3: AVX2, BMI2, FMA, POPCNT and
  `pclmulqdq`. The release binary does not start on a CPU below that
  baseline; `./run.sh check` refuses with a message instead of running.
- AVX-512F and AVX-512IFMA are optional. On a CPU with AVX-512F the package
  runs the `avx512` and `avx2` tiers; with IFMA it runs all three. A tier the
  CPU cannot run is `n/a` in the tables and recorded in `skipped.json`.
- 16 hardware threads are enough for `quick`; 16 cores with SMT are
  recommended for `full`. The 8C16T rows need eight cores.
- RAM: 16 GiB for `quick` (up to 4K signatures), 32 GiB minimum for the 32K
  cells and 64 GiB recommended. On the reference machine, the 32K prover peaks
  at about 16 GiB for ML-DSA-44, 17 GiB for ML-DSA-65 and 24 GiB for
  ML-DSA-87 (`VmHWM`, estimate). The script refuses a 32K cell when
  `MemAvailable` is below 28 GiB and records the reason in `skipped.json`.
- Disk: about 20 GB during a full run (one cargo target directory for `agg`
  and `mode_b`, Aerie's target, the cloned sources and the results). `quick`
  needs about 10 GB.
- bash, python3 (standard library; `tomllib` on 3.11+, a bundled fallback on
  older versions), a C compiler, `make`, `objdump`, `taskset`, rustup with
  Rust 1.99.1, and Rust 1.95.0 for Aerie. `numactl`, `/usr/bin/time`,
  `shellcheck`, Docker and Intel SDE are optional.

The repository's root `rust-toolchain.toml` is not modified. The scripts run
cargo with `RUSTUP_TOOLCHAIN=1.99.1`; `artifact/rust-toolchain.toml` documents
the same pin for commands started inside this directory.

## Modes and time

| Mode | What it runs | Time on 16 cores |
| --- | --- | --- |
| `check` | tool and feature probe | seconds |
| `fetch` | two git clones and `cargo fetch` | 2--10 min, network |
| `build` | one release build of `agg` and `mode_b`, Aerie's bench, `natver{,65,87}` | 5--20 min; the first build is the long one |
| `gates` | per parameter set (`--params 44,65,87`) and relation (`--relations`, default `h-fips`): 1K aggregates in every tier, verification, negative check; per set the aggregate on one thread and natver; self-reports | about 5--10 min per relation (estimate) |
| `quick --smoke` | touches every run kind (`ours`, `mode_b`, `ablations`, `fresh`, `aerie`, `native`) and all three parameter sets at 1K, 1+1 runs | about 5 min after the build (estimate) |
| `quick` | 1K and 4K (ML-DSA-44) plus 1K (ML-DSA-65, -87), three tiers, 8C16T and 1T, mode B, traced ablations, fresh memory, Aerie (Falcon-512/1024) and native (44/65/87) at 1K | 20--40 min after the build (estimate) |
| `full` | the complete evaluation matrix for ML-DSA-44, -65 and -87 under H-FIPS: 32K headline on 16C32T, 8C16T and 1T; scaling; mode-B public bits; ablations (`AGG_LATE_JOIN=0`, `AGG_FLOCK_THREADS=0`, etc.) and traced timeline; fresh memory; Aerie and native (44/65/87) baselines, 1+5 warm runs | about 1.7 h (~100 min, estimate) after the build; H-exact adds about 6 min (estimate) |

### Estimated full-run time breakdown

Derived from the paper's per-run times on one 16-core EPYC 9355 (Zen 5) node
(all figures below are estimates):

- **Headline 32K on 16C32T and 8C16T** (`headline`, `headline-65`,
  `headline-87`, 3 tiers, 1 warmup + 5 runs): about **10 min** (estimate).
- **Headline 32K on 1T** (`headline-1t`, `headline-1t-65`, `headline-1t-87`,
  3 tiers, 2 processes = 7 prover runs per tier at 30--60 s each): about
  **48 min** (estimate; the longest part of `full`).
- **Scaling** (`scaling` at $2^{10}\text{--}2^{15}$ for 44; `scaling-65` and
  `scaling-87` at $2^{10}, 2^{13}, 2^{15}$): about **23 min** (estimate).
- **Mode B, control ablations, traced timeline and fresh memory** (`mode-b`,
  `ablation` with 7 arms including `AGG_LATE_JOIN=0`, `timeline` with traces,
  `prover-memory`, `verifier-memory` at 32K for 44): about **11 min**
  (estimate).
- **Aerie and native baselines** (`aerie`, `aerie-avx2`, `aerie-1024`,
  `aerie-1024-avx2` at 8K and 32K; `native`, `native-65`, `native-87` at 32K):
  about **10 min** (estimate).
- **H-exact at 32K, opt-in** (`relations`, `relations-65`, `relations-87`
  with `--relations h-fips,h-exact`, 3 tiers): about **6 min** (estimate).

Total: **about 100 minutes (~1.7 hours, estimate)** after the build, without
the opt-in H-exact groups.

## What is measured

One release build, three reported tiers. Every tier is a run-time cap of the
same binary; the baseline flags live in `.cargo/config.toml` and are not
repeated here.

| Tier | `AGG_ISA` | Selected kernels |
| --- | --- | --- |
| `ifma` | unset (detect) | IFMA field and NTT kernels; AVX-512 Flock, ML-DSA and Aerie |
| `avx512` | `avx512` | AVX-512 everywhere except the IFMA field and NTT kernels |
| `avx2` | `avx2` | executes no AVX-512 instruction; the x86-64-v3 baseline |

### Ablation and mode-B arms

- **Mode B (`kind = "mode_b"`, `agg-bench --bench mode_b`)**: publishes
  $\mathsf{w1Encode}(w_1')$ in the clear and proves no Keccak hash (ML-DSA-44,
  about 803 B/sig at 32K). Built into `artifact/build/bin/mode_b` by `./run.sh build`
  and run at $N=32{,}768$ with `--warmup 1 --reps 5`.
- **Traced timeline (`tag = "timeline"`, `traced = true`)**: runs `defaults`
  and `flock-serial` (`AGG_FLOCK_THREADS=0`, Flock after the PIOP) with
  `AGG_STAGE_TRACE=1`, `AGG_SCHED_TRACE=1`, `AGG_TIED_TRACE=1`, `PCS_TRACE=1`,
  `LINCHECK_TRACE=1`, `FLOCK_ZC_TIMING=1` and `AGG_PHASE_STAMPS=1`, feeding the
  `Flock after the PIOP` row and traced note of `tab:ablation`.
- **Control ablations (`tag = "ablation"`)**: alternates seven arms on 16C32T
  at 32K: `defaults`, `no-hugepage` (`AGG_HUGEPAGE=0`), `no-alloc-cache`
  (`AGG_ALLOC_CACHE=0`), `flock-serial` (`AGG_FLOCK_THREADS=0`), `late-join`
  (`AGG_LATE_JOIN=0`, running the Akita opening's finish after the Flock proofs
  join instead of beside them), `stage-branch-main-pool`
  (`AGG_PIOP_S_THREADS=0`), and `piop-row` (`AGG_PIOP_LAYOUT=row`).

`AGG_PIOP_LAYOUT=plane|row` selects the sum-check table layout. Unset selects
plane on every vector backend (avx512ifma, avx512, avx2) and row on the
portable (scalar) backend; the proof does not depend on the choice.

The `scalar` cap (`AGG_ISA=scalar`) is not a reported tier. `./run.sh gates`
runs it as an additional correctness check: its aggregate must match the three
reported tiers byte for byte.

### Absence of AVX-512 is checked at run time

One binary legitimately contains every kernel, so a static scan of it proves
nothing. The package uses two run-time checks instead:

1. `AGG_SIMD_REPORT=1` prints the level every dispatcher selected. `gates`
   requires every AVX-512 and IFMA flag to be false under `AGG_ISA=avx2` (and
   under `scalar`), and every IFMA flag to be false under `AGG_ISA=avx512`.
   `quick` and `full` repeat the check before measuring a tier.
2. If `sde64` is installed, `gates` runs the binary under Intel SDE at
   N = 256: `-hsw` (Haswell: AVX2, no AVX-512) for `avx2` and `-skx`
   (Skylake-X: AVX-512 without IFMA) for `avx512`. `--no-sde` skips it.

The static `objdump` check remains for the C baselines (`natver`, `natver65`,
`natver87`), which must contain no AVX-512 instruction (`zmm`, opmask, high
registers and EVEX-only mnemonics are counted in `build.json`).

### Expected times

The following medians were measured on an idle AMD EPYC 9355 node (Zen 5) at
$N = 32{,}768$ under H-FIPS on 16C32T, 8C16T and 1T, one warm-up and five runs (`tab:prove`,
`tab:bytes`, `tab:baselines`). Node-to-node spread is a few percent; `./run.sh full`
reproduces every cell.

| Set | Tier | 16C32T prove | 8C16T prove | 1T prove | 16C32T verify (unc. / cached) | Aggregate (B/sig) |
| --- | --- | --- | --- | --- | --- | --- |
| 44 | `ifma` | 2.34 s | 3.47 s | 24.69 s | 62 / 54 ms | 1,611,831 B (49.2) |
| 44 | `avx512` | 2.40 s | 3.52 s | 24.79 s | -- | 1,611,831 B (49.2) |
| 44 | `avx2` | 2.70 s | 4.27 s | 32.63 s | -- | 1,611,831 B (49.2) |
| 65 | `ifma` | 2.94 s | 4.07 s | 29.69 s | 82 / 70 ms | 2,143,007 B (65.4) |
| 65 | `avx512` | 2.96 s | 4.13 s | 29.83 s | -- | 2,143,007 B (65.4) |
| 65 | `avx2` | 3.33 s | 4.99 s | 39.08 s | -- | 2,143,007 B (65.4) |
| 87 | `ifma` | 3.72 s | 5.31 s | 38.92 s | 114 / 88 ms | 2,710,255 B (82.7) |
| 87 | `avx512` | 3.74 s | 5.44 s | 38.92 s | -- | 2,710,255 B (82.7) |
| 87 | `avx2` | 4.25 s | 6.64 s | 51.40 s | -- | 2,710,255 B (82.7) |

On 16C32T with IFMA, the opt-in reference relation H-exact proves 3.4% (44),
10.6% (65) and 6.6% (87) faster than H-FIPS. In `tab:ablation` (ML-DSA-44,
32K, 16C32T, AVX2), running Flock after the PIOP (`AGG_FLOCK_THREADS=0`) costs
0.43 s against the default, and the mode-B public-bits aggregate takes 2.08 s
and 803.1 B/sig. Absolute times on another microarchitecture
differ; the tier order and the byte identity are the portable parts of the
result.

Warm cells follow the paper protocol: one process per cell with
`--warmup 1 --reps N`, split into `processes` chunks so that the tiers
alternate (ablations use one process per cell and alternate the arms instead).
Each process pays its own warm-up, and every timed run is read from its
`--json` output.

Every individual run is kept in `results/<timestamp>/raw/*.json`, and
each cell records the JSON of every process that contributed to it.
`summary.csv` has the median, minimum, maximum, the order-statistic interval
of the median (97.9% for ten runs, 93.75% for five) and a `noisy` or
`co-tenant` flag.

The CPU sets come from sysfs: SMT siblings, L3 domains (`index3`) and NUMA
nodes. `16C32T` is 16 cores with both siblings, filling whole L3 domains
first; `8C16T` is one eight-core L3 domain; `1T` is one thread of a core whose
sibling is idle. A smaller machine uses what it has and labels the set with
its actual core and thread count. Commands are pinned with `taskset` and with
`numactl` when NUMA is available.

## Claims and the cell that checks each one

| Claim | Where to look | Command |
| --- | --- | --- |
| C1: the aggregate takes 49.2, 65.4 and 82.7 B per signature at 32K for ML-DSA-44, -65 and -87 | `tab-bytes.tex`, row `total` | `./run.sh full` |
| C2: every tier and the scalar gate give the same aggregate and verification accepts | `gates.json`, `all_configurations_identical` and `verify` | `./run.sh gates` |
| C3: prove and verify times on a Zen 5 machine are within 10% of the paper across all three sets | `tab-prove.tex`, `tab-scaling.tex` | `./run.sh full` |
| C4: the tier ladder is ordered IFMA, AVX-512F, AVX2 and the GF(2) ablation matches | `tab-prove.tex`, `tab-ablation.tex` | `./run.sh quick --smoke` |
| C5: ours against Aerie and native verification for ML-DSA-44, -65 and -87 | `tab-baselines.tex` | `./run.sh quick` |

`expected/claims.toml` carries the same table, and each results directory gets
a `claims-measured.csv` with the values that were actually measured.

## Reading the output

Everything lands in `artifact/results/<timestamp>/` (UTC; the host name is
recorded in `machine.json`, not in the path, so the generated tables do not
carry it):

- `machine.json`: CPU model and flags, kernel, governor, boost, THP, memory,
  load, foreign processes, compiler versions, and the tiers the CPU supports.
  The probe only reads; it never writes sysfs.
- `cpusets.json`: the three allocations, their CPU lists and L3 domains.
- `build.json`: the release flags, binary hashes (`agg` and `mode_b`), the
  run-time self-report of every tier, and the four AVX-512 counts for each
  `natver` binary.
- `gates.json`: per parameter set and relation, the 1K aggregate hashes,
  verify verdicts and the negative check; per set, the one-thread check and
  natver; the self-reports and the SDE result.
- `raw/<run-id>.json`: one cell, with every timed run, the aggregate byte
  breakdown and the selected kernels.
- `summary.csv`: one row per cell (keyed by parameter set) with the statistics
  and flags.
- `paper-tables/`: the generated LaTeX tables for the results directory (also
  copied to `artifact/paper-tables/`, which is ignored by git):
  - `tab-prove.tex` (`\label{tab:prove}`): 32K prover time across sets (44, 65,
    87), tiers (IFMA, AVX-512F, AVX2) and allocations (16C32T, 8C16T, 1T).
  - `tab-ablation.tex` (`\label{tab:ablation}`): GF(2)-side ablation (`full
    aggregate`, `Flock after the PIOP`, `public bits` mode B) and control
    ablations (including `AGG_LATE_JOIN=0`).
  - `tab-scaling.tex` (`\label{tab:scaling}`): batch-size scaling ($2^{10}$,
    $2^{13}$, $2^{15}$) with B/sig, prove, uncached/cached verify and peak RSS
    in GiB for 44, 65 and 87.
  - `tab-bytes.tex` (`\label{tab:bytes}`): aggregate byte composition per
    signature across components for 44, 65 and 87.
  - `tab-baselines.tex` (`\label{tab:baselines}`): native AVX2 verification
    (44, 65, 87), Aerie (Falcon-512, Falcon-1024) and ours (44, 65, 87) on
    16C32T and 1T.
  - `tab-relations.tex` (`\label{tab:relations}`): with the opt-in H-exact
    runs, H-FIPS and H-exact prover times, the change of H-exact against
    H-FIPS, verify times and aggregate bytes for 44, 65 and 87.
  - `numbers.tex`: `\newcommand` macros for in-text figures.

## Known variance sources

- Frequency governor and boost. The probe records both. A governor other
  than `performance` adds noise, and the summary says so.
- Transparent huge pages. The prover's own huge-page advice is independent of
  THP, but a `never` setting changes allocation behaviour. The
  `AGG_HUGEPAGE=0` ablation shows the size of the effect.
- Co-tenancy. The summary marks a cell `co-tenant` when more than 5% of the
  time on CPUs outside the pinned set was busy. Shared L3 and DRAM move the
  32K cells by up to a few percent.
- Turbo, SMT and the memory system differ across CPU models. Absolute times
  are only comparable within one machine; the tier ratios are the portable
  part of the result.
- Record generation is outside every timed region. It is deterministic
  (seed 1; the gates use keys from seed 42 and records from seed 43) and the
  1K aggregate bytes of every parameter set and relation are checked against
  `expected/hashes.json`, keyed by set and relation.

## Troubleshooting

- `check` says the CPU lacks AVX2: the release binary is built for
  x86-64-v3 and cannot run. There is no scalar-only build.
- `check` notes that a source is not fetched or a binary is not built yet: run
  `./run.sh fetch` and `./run.sh build`.
- `gates` fails on `expected_ok`: the aggregate bytes changed with the code.
  This is expected after a protocol change. Re-pin them with
  `./run.sh gates --write-expected` only after confirming that the change is
  intended. Maintainers pin the submission revision with `./run.sh freeze`,
  which builds the binary, rewrites `expected/hashes.json` for every set and
  relation (H-FIPS and H-exact) and sets the in-repo commit of `sources.lock`
  to `HEAD`.
- `gates` fails on a self-report: a dispatcher selected a kernel above
  `AGG_ISA`. The failing crate and flag are printed.
- `./run.sh gates --reference FILE` compares an external reference aggregate.
  A mismatch means the reference was produced by a different code state or a
  different record set, not that a tier is broken. Re-pin the reference after
  the final merge.
- `16C32T` is `n/a`: the machine has fewer than 16 cores. `quick` uses
  `8C16T`; the table labels the fallback.
- A 32K cell is skipped: `MemAvailable` is below 28 GiB. Close other
  processes or run `quick`.
- THP is `never`: results resemble the `AGG_HUGEPAGE=0` ablation. The run is
  still valid; the note in the tables records it.
- No IFMA: the `ifma` row is `n/a (CPU without AVX-512IFMA)`. No AVX-512F:
  the `ifma` and `avx512` rows are `n/a` and only `avx2` runs.

## Docker

`docker/Dockerfile` builds the package in a pinned image. The container
shares the host kernel, so THP, the governor and co-tenancy cannot be set
from inside; the probe reports them. Do not pass `--cpus`, because the cgroup
quota throttles CPU-bound loops. Expect native-like timings, but measure one
cell both ways if timing claims matter. Use native runs for the `Reproduced`
badge and Docker for `Functional`.

Nix packaging (a flake with both Rust toolchains) is future work.

## License

Our code is under the MIT licence (`../LICENSE`); vendored dependencies keep their upstream licences (`sources.lock`).
