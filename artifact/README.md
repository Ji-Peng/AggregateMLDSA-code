# Artifact evaluation package: aggregating ML-DSA-44 signatures

This directory rebuilds every measurement of the evaluation section from the
pinned source tree on a plain x86-64 Linux machine. It does not need Slurm,
root, or network access after `./run.sh fetch`.

The package builds one source tree in four kernel configurations and measures
proving, verification and aggregate size at batch sizes up to 32,768. It also
runs two baselines: Aerie v0.2.0 on Falcon-512 and the pq-crystals ML-DSA-44
AVX2 implementation. The comparison baseline is a native batch verification,
not an aggregate.

## Quick start

```sh
cd artifact
./run.sh check            # tools, CPU features, sources, binaries
./run.sh fetch            # clone Aerie and pq-crystals/dilithium; verify SHAs
./run.sh build            # ours (three configurations), Aerie, natver
./run.sh gates            # byte identity, acceptance, negative check at 1K
./run.sh quick --smoke    # about five minutes of measurements after the builds
./run.sh tables           # write paper-tables/*.tex
```

`./run.sh all` runs `fetch`, `build`, `gates`, `quick` and `tables` in that
order. Add `--smoke` for the smallest run and `--full` to use the full matrix
instead of `quick`.

## Hardware and software

- x86-64 Linux with AVX2 and `pclmulqdq`; the prover requires CLMUL. AVX-512F
  is optional, AVX-512IFMA is optional. The script skips the rows the CPU
  cannot run and labels them `n/a` in the tables.
- 16 hardware threads are enough for `quick`; 16 cores with SMT are
  recommended for `full`. The 8C16T rows need eight cores.
- RAM: 16 GiB for `quick` (4K signatures), 32 GiB minimum for the 32K cells
  and 64 GiB recommended. The 32K prover peaks around 19.5 GiB on the
  reference machine. The script refuses a 32K cell when `MemAvailable` is
  below 28 GiB and records the reason in `skipped.json`.
- Disk: about 60 GB during a full run (three cargo target directories for
  ours, Aerie's target, the cloned sources and the results). `quick` needs
  about 25 GB.
- bash, python3 (standard library; `tomllib` on 3.11+, a bundled fallback on
  older versions), a C compiler, `make`, `objdump`, `taskset`, rustup with
  Rust 1.99.1, and Rust 1.95.0 for Aerie. `numactl`, `/usr/bin/time`,
  `shellcheck` and Docker are optional.

The repository's root `rust-toolchain.toml` is not modified. The scripts run
cargo with `RUSTUP_TOOLCHAIN=1.99.1`; `artifact/rust-toolchain.toml` documents
the same pin for commands started inside this directory.

## Modes and time

| Mode | What it runs | Time on 16 cores |
| --- | --- | --- |
| `check` | tool and feature probe | seconds |
| `fetch` | two git clones and `cargo fetch` | 2--10 min, network |
| `build` | one release build of ours, Aerie's bench, natver | 10--25 min; the first build is the long one |
| `gates` | five 1K aggregates, verification, negative check | 2--5 min |
| `quick --smoke` | four configurations at 1K, one thread, Aerie and native at 1K, 1+1 runs | 5--15 min after builds |
| `quick` | 1K and 4K, four configurations, eight cores and one thread, Aerie and native at 1K, 1+3 runs | 20--40 min after builds |
| `full` | the evaluation matrix: 32K headline, 1T, relations, scaling, ablations, memory, baselines, 1+10 runs | 3--5 h after builds |

`full` writes about 125 MB of raw JSON at 32K. Its longest cells are scalar
proving on one thread.

## What is measured

Four configurations of one source tree:

| Row | Build | Environment |
| --- | --- | --- |
| `ifma` | `agg-v3` (x86-64-v3 + CLMUL) | none |
| `avx512` | `agg-v3` | `AGG_SIMD=portable` |
| `avx2` | `agg-noavx512` when the crate declares the feature, else `agg-v3` | `AGG_SIMD=portable AGG_MLDSA_SIMD=avx2 FLOCK_SIMD=portable` |
| `scalar` | `agg-v1` when the crate declares the feature, else `agg-v3` | additionally `AGG_MLDSA_SIMD=scalar AKITA_SCALAR_NTT=1` |

The knobs are in `config/configs.toml` only. When the workspace does not
declare the `no-avx512` feature (the current state of the pinned commit), the
build records that fact and uses the v3 binary for the two lower rows. The
static AVX-512 check then records a warning instead of failing; `natver` is
always checked and must contain no AVX-512 instruction.

Each measurement is one warm-up run followed by `reps` timed runs.
Configurations are interleaved by round, and every individual run is kept in
`results/<host>-<timestamp>/raw/*.json`. `summary.csv` has the median, minimum,
maximum, the order-statistic interval of the median (97.9% for ten runs,
93.75% for five) and a `noisy` or `co-tenant` flag.

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
| C2: every configuration and thread count gives the same aggregate and verification accepts | `gates.json`, `all_configurations_identical` and `verify` | `./run.sh gates` |
| C3: prove and verify times on a Zen 5 machine are within 10% of the paper | `tab-e2e.tex` | `./run.sh full` |
| C4: the ISA ladder is ordered IFMA, AVX-512, AVX2, scalar | `tab-ablation.tex`, ISA ladder rows | `./run.sh quick --smoke` |
| C5: ours against Aerie and native verification | `tab-baselines.tex` | `quick` |

`expected/claims.toml` carries the same table, and each results directory gets
a `claims-measured.csv` with the values that were actually measured.

## Reading the output

Everything lands in `artifact/results/<host>-<timestamp>/`:

- `machine.json`: CPU model and flags, kernel, governor, boost, THP, memory,
  load, foreign processes, compiler versions. The probe only reads; it never
  writes sysfs.
- `cpusets.json`: the three allocations, their CPU lists and L3 domains.
- `build.json`: flags, binary hashes, the `no-avx512` feature state and the
  four AVX-512 counts for every binary.
- `gates.json`: the 1K aggregate hashes, verify verdicts and the negative
  check.
- `raw/<run-id>.json`: one cell, with every timed run.
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
  are only comparable within one machine; the ISA-ladder ratios are the
  portable part of the result.
- Record generation is outside every timed region. It is deterministic
  (seed 1) and the 1K aggregate bytes are checked against
  `expected/hashes.json`.

## Troubleshooting

- `check` says a binary is missing: run `./run.sh build`.
- `gates` fails on `expected_ok`: the aggregate bytes changed with the code.
  This is expected after a protocol change. Re-pin them with
  `./run.sh gates --write-expected` only after confirming that the change is
  intended.
- `./run.sh gates --reference FILE` compares an external reference aggregate.
  A mismatch means the reference was produced by a different code state, not
  that a configuration is broken; the two tips measured here (the pinned main
  commit and the `ifma` branch tip) both produce 426,895 bytes at 1K. Re-pin
  the reference after the final merge.
- `16C32T` is `n/a`: the machine has fewer than 16 cores. `quick` uses
  `8C16T`; the table labels the fallback.
- A 32K cell is skipped: `MemAvailable` is below 28 GiB. Close other
  processes or run `quick`.
- THP is `never`: results resemble the `AGG_HUGEPAGE=0` ablation. The run is
  still valid; the note in the tables records it.
- No IFMA: the `ifma` row is skipped and the tables show
  `n/a (CPU without AVX-512IFMA)`.

## Docker

`docker/Dockerfile` builds the package in a pinned image. The container
shares the host kernel, so THP, the governor and co-tenancy cannot be set
from inside; the probe reports them. Do not pass `--cpus`, because the cgroup
quota throttles CPU-bound loops. Expect native-like timings, but measure one
cell both ways if timing claims matter. Use native runs for the `Reproduced`
badge and Docker for `Functional`.

Nix packaging (a flake with both Rust toolchains) is future work.
