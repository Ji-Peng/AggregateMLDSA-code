# AggregateMLDSA

Aggregation of ML-DSA (FIPS 204) signatures into one succinct proof per batch, for
ML-DSA-44, ML-DSA-65 and ML-DSA-87. The proof system follows Aerie (a sum-check PIOP with
the Akita lattice polynomial commitment, no recursion). The SHAKE256 call that ML-DSA
verification makes on a witness-dependent value, `c~ = H(mu || w1Encode(w1'))`, is proven
with Flock over GF(2) and tied to the prime-field side by a parity bridge.

At 32,768 signatures the aggregate takes 49.2, 65.4 and 82.7 bytes per signature for
ML-DSA-44, -65 and -87 (public `c~` included).

## Requirements

- x86-64 Linux with AVX2 (x86-64-v3) and PCLMULQDQ. AVX-512F and AVX-512IFMA are
  detected at run time and used when present.
- rustup; the toolchain (1.99.1) is pinned by `rust-toolchain.toml` and installed on the
  first build.
- 16 GiB RAM for batches up to 4K signatures; 32 GiB or more for 32K.

## Build and quick check

```sh
cargo build --release -p agg-cli        # first build takes 5-20 min
scripts/gate.sh                         # 1K round trips and tamper checks, all sets and relations
scripts/test.sh                         # unit and integration tests (add --release for speed)
```

## Usage

```sh
agg=target/release/agg
$agg keygen-batch --params 44 --count 1024 --out keys.bin
$agg sign-batch   --params 44 --keys keys.bin --count 1024 --out records.bin
$agg aggregate    --params 44 --records records.bin --out agg.bin
$agg verify       --params 44 --records records.bin --aggregate agg.bin   # exit 0 = accept
$agg bench        --params 44 --count 1024 --warmup 1 --reps 5            # timings and size
```

`--relation h-u|h-exact|h-fips` selects the proven relation (default `h-u`; `h-fips` makes
the extracted witness a FIPS 204 signature). Running `agg` without arguments lists every
option; [`docs/cli.md`](docs/cli.md) describes the tool and the aggregate format.

## Reproducing the paper

[`artifact/`](artifact/README.md) is the reproduction package. It fetches and verifies
the pinned external sources (Aerie, pq-crystals Dilithium), builds every binary once,
checks byte identity across ISA tiers, runs the evaluation matrix and regenerates the
paper's tables:

```sh
cd artifact
./run.sh check && ./run.sh fetch && ./run.sh build
./run.sh gates            # byte identity and acceptance at 1K, about 15-30 min
./run.sh quick --smoke    # every experiment kind at 1K, about 5 min
./run.sh full             # the complete matrix, about 2 h on 16 cores
./run.sh tables           # results/LATEST/paper-tables/*.tex
```

[`artifact/README.md`](artifact/README.md) lists the claims, the expected numbers and
their reference machine (AMD EPYC 9355, Zen 5).

## Run-time knobs

| Variable | Effect |
| --- | --- |
| `AGG_ISA=avx512ifma\|avx512\|avx2\|scalar` | upper bound on the SIMD level of every kernel; proofs are byte-identical across levels |
| `AGG_SIMD_REPORT=1` | print the selected kernels at startup |
| `AGG_CPUS=<list>` | CPU list for `taskset` in `scripts/test.sh` and `scripts/bench.sh` (default: no pinning) |

[`docs/engineering.md`](docs/engineering.md) documents the remaining knobs.

## Layout

| Path | Content |
| --- | --- |
| `crates/agg-witness` | native ML-DSA verification and the witness (`z`, `u`, `v`, `b`) derivation |
| `crates/agg-pcs-config` | Akita commitment presets and parameter catalogs |
| `crates/agg-piop` | the PIOP: range lookup (LogUp-GKR), sum-check stages, proof-of-work schedule, verifier |
| `crates/agg-bridge` | the parity bridge between the prime field and GF(2) |
| `crates/agg-keccak` | the Flock Keccak circuit and its tie to the bridge |
| `crates/agg-cli` | the `agg` binary: key generation, signing, aggregation, verification, benchmarks |
| `crates/agg-simd`, `crates/agg-mldsa-simd` | run-time dispatched AVX-512IFMA, AVX-512F, AVX2 and portable kernels |
| `crates/agg-bench`, `crates/agg-mldsa-cref` | benchmark harness and C reference harness |
| `vendor/` | Akita and Flock with prover-side patches ([`PATCH.md`](PATCH.md)) |
| `docs/` | specification and proofs |
| `artifact/` | reproduction package |

Main documents: [`docs/spec.md`](docs/spec.md) (protocol), [`docs/soundness.md`](docs/soundness.md)
(128-bit soundness and the proof-of-work schedule), [`docs/h-exact.md`](docs/h-exact.md)
(the three relations), [`docs/deviations.md`](docs/deviations.md) (differences from Aerie).

## Dependencies

All revisions are pinned in `Cargo.toml` and `Cargo.lock`; `artifact/sources.lock` lists
them with their licenses.

| Dependency | Source | Revision |
| --- | --- | --- |
| Aerie | https://github.com/a16z/aerie | `49b6c21` (v0.2.0) |
| Akita | https://github.com/LayerZero-Labs/akita | `e023b6a`, vendored in `vendor/akita` |
| Flock | https://github.com/succinctlabs/flock | `83f963e`, vendored in `vendor/flock-core`, `vendor/flock-field` |
| Jolt (`jolt-field`, `jolt-poly`) | https://github.com/a16z/jolt | `c7c2d08` |
| Spongefish | https://github.com/arkworks-rs/spongefish | `ef97413` |
| `ml-dsa`, `sha3` | crates.io | `=0.1.1`, `0.10.9` |

The Flock patches change the verifier (`vendor/flock-core/PATCH.md`), so our Flock proofs
do not verify with upstream Flock. The Akita patches touch the prover only.

## License

MIT OR Apache-2.0.
