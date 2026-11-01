# Artifact evaluation package: aggregating ML-DSA-44 signatures

This directory rebuilds every measurement of the evaluation section from the
pinned source tree on a plain x86-64 Linux machine. It does not need Slurm,
root, or network access after `./run.sh fetch`.

The package builds one source tree **once** and measures proving, verification
and aggregate size at batch sizes up to 32,768 in three run-time tiers. It also
runs two baselines: Aerie v0.2.0 on Falcon-512 and the pq-crystals ML-DSA-44
AVX2 implementation. The comparison baseline is a native batch verification,
not an aggregate.

## Quick start

```sh
cd artifact
./run.sh check            # tools, CPU features, sources, binaries
./run.sh fetch            # clone Aerie and pq-crystals/dilithium; verify SHAs
./run.sh build            # the release binary, Aerie, natver
./run.sh gates            # byte identity, acceptance, self-reports at 1K
./run.sh quick --smoke    # about five minutes of measurements after the build
./run.sh tables           # write paper-tables/*.tex
```

`./run.sh all` runs `fetch`, `build`, `gates`, `quick` and `tables` in that
order. Add `--smoke` for the smallest run and `--full` to use the full matrix
instead of `quick`.

## Hardware and software

- x86-64 Linux built for x86-64-v3: AVX2, BMI2, FMA, POPCNT and
  `pclmulqdq`. The release binary does not start on a CPU below that
  baseline; `./run.sh check` refuses with a message instead of running.
- AVX-512F and AVX-512IFMA are optional. On a CPU with AVX-512F the package
  runs the `avx512` and `avx2` tiers; with IFMA it runs all three. A tier the
  CPU cannot run is `n/a` in the tables and recorded in `skipped.json`.
- 16 hardware threads are enough for `quick`; 16 cores with SMT are
  recommended for `full`. The 8C16T rows need eight cores.
- RAM: 16 GiB for `quick` (4K signatures), 32 GiB minimum for the 32K cells
  and 64 GiB recommended. The 32K prover peaks around 19.5 GiB on the
  reference machine. The script refuses a 32K cell when `MemAvailable` is
  below 28 GiB and records the reason in `skipped.json`.
- Disk: about 20 GB during a full run (one cargo target directory for ours,
  Aerie's target, the cloned sources and the results). `quick` needs about
  10 GB.
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
| `build` | one release build of ours, Aerie's bench, natver | 5--20 min; the first build is the long one |
| `gates` | 1K aggregates in every tier, verification, negative check, self-reports | 2--5 min |
| `quick --smoke` | three tiers at 1K, one thread, Aerie and native at 1K, 1+1 runs | 5--15 min after the build |
| `quick` | 1K and 4K, three tiers, eight cores and one thread, Aerie and native at 1K, 1+3 runs | 20--40 min after the build |
| `full` | the evaluation matrix: 32K headline, 1T, relations, scaling, ablations, memory, baselines, 1+5 warm runs | about 1 h after the build |

`full` writes about 125 MB of raw JSON at 32K. Its longest cells are 1T
proving. The full matrix took 51 minutes on one EPYC 9355 node (`xcni`) after
the build (2026-11-01).

## What is measured

One release build, three reported tiers. Every tier is a run-time cap of the
same binary; the baseline flags live in `.cargo/config.toml` and are not
repeated here.

| Tier | `AGG_ISA` | Selected kernels |
| --- | --- | --- |
| `ifma` | unset (detect) | IFMA field and NTT kernels; AVX-512 Flock, ML-DSA and Aerie |
| `avx512` | `avx512` | AVX-512 everywhere except the IFMA field and NTT kernels |
| `avx2` | `avx2` | executes no AVX-512 instruction; the x86-64-v3 baseline |

`AGG_PIOP_LAYOUT=plane|row` selects the sum-check table layout. Unset selects
plane exactly when the IFMA backend is selected and row otherwise; the proof
does not depend on the choice. The `full` matrix runs one `row` ablation.

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

The static `objdump` check remains for the C baseline `natver`, which must
contain no AVX-512 instruction (`zmm`, opmask, high registers and EVEX-only
mnemonics are counted in `build.json`).

### Expected times

The following were measured on one AMD EPYC 9355 node (`xcni`, Zen 5) at
N = 32,768 under H-u, one warm-up and three runs, unified `main`
`7b9c67a`. Node-to-node spread is a few percent.

| Tier | 16C32T | 8C16T |
| --- | --- | --- |
| `ifma` | 2.60--2.69 s | 4.21 s |
| `avx512` | 2.77 s | not measured |
| `avx2` | 2.84--2.94 s | 5.51 s |

The aggregate is 1,612,232 bytes at 32K, or 49.2 bytes per signature, in every
tier. Absolute times on another microarchitecture differ; the tier order and
the byte identity are the portable parts of the result.

Warm cells follow the paper protocol: one process per cell with
`--warmup 1 --reps N`, split into two processes of about N/2 so that the
tiers still alternate (ablations use one process per cell and alternate the
arms instead). Each process pays its own warm-up, and every timed run is read
from its `--json` output.

Every individual run is kept in `results/<host>-<timestamp>/raw/*.json`, and
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
| C1: the aggregate takes 49.2 B per signature at 32K | `tab-bytes.tex`, row `2^15` | `quick`, or `full` for 32K |
| C2: every tier and the scalar gate give the same aggregate and verification accepts | `gates.json`, `all_configurations_identical` and `verify` | `./run.sh gates` |
| C3: prove and verify times on a Zen 5 machine are within 10% of the paper | `tab-e2e.tex` | `./run.sh full` |
| C4: the tier ladder is ordered IFMA, AVX-512, AVX2 | `tab-ablation.tex`, ISA ladder rows | `./run.sh quick --smoke` |
| C5: ours against Aerie and native verification | `tab-baselines.tex` | `quick` |

`expected/claims.toml` carries the same table, and each results directory gets
a `claims-measured.csv` with the values that were actually measured.

## Reading the output

Everything lands in `artifact/results/<host>-<timestamp>/`:

- `machine.json`: CPU model and flags, kernel, governor, boost, THP, memory,
  load, foreign processes, compiler versions, and the tiers the CPU supports.
  The probe only reads; it never writes sysfs.
- `cpusets.json`: the three allocations, their CPU lists and L3 domains.
- `build.json`: the release flags, binary hashes, the run-time self-report of
  every tier, and the four AVX-512 counts for `natver`.
- `gates.json`: the 1K aggregate hashes, verify verdicts, the negative check,
  the self-reports and the SDE result.
- `raw/<run-id>.json`: one cell, with every timed run and the selected
  kernels.
- `summary.csv`: one row per cell with the statistics and flags.
- `paper-tables/` and `../../paper-tables/`: the generated LaTeX tables.

The tables are complete `talltblr` environments with labels `tab:ae-*`, so
they can be `\input` without clashing with the paper's current labels. The
generator also writes `numbers.tex` with one macro per in-text number and
rounds seconds to two decimals, milliseconds to integers (one decimal below
100 ms for baselines), bytes per signature to one decimal and ratios to one
decimal.

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
  (seed 1) and the 1K aggregate bytes are checked against
  `expected/hashes.json`.

## Troubleshooting

- `check` says the CPU lacks AVX2: the release binary is built for
  x86-64-v3 and cannot run. There is no scalar-only build.
- `check` says a binary is missing: run `./run.sh build`.
- `gates` fails on `expected_ok`: the aggregate bytes changed with the code.
  This is expected after a protocol change. Re-pin them with
  `./run.sh gates --write-expected` only after confirming that the change is
  intended.
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
