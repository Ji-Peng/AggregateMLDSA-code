# AggregateMLDSA-code

Implementation of the AggregateMLDSA project: aggregation of ML-DSA (FIPS 204)
signatures following Aerie (sum-check PIOP + Akita lattice PCS, no recursion),
with a Flock-based argument for the witness-dependent SHAKE256 call.
The protocol source of truth is [`docs/spec.md`](docs/spec.md); deviations from it are
recorded in `docs/deviations.md`.

## Layout

| Crate | Kind | Content (spec Section 14) | Milestone |
| --- | --- | --- | --- |
| `crates/agg-witness` | lib | ML-DSA-44 native verify, key expansion, (z,u,v,b) derivation, CRT product, table builders, padding | M1 |
| `crates/agg-pcs-config` | lib | Aerie's `Fp64SignedWidth4` preset, four-group catalogs, two-round commit driver, work plan | M1 |
| `crates/agg-piop` | lib | stages 1-3, range leaves and links, weight tables, verifier, Akita openings | M2 |
| `crates/agg-bridge` | lib | seed to matrices, Fp parity and carry-digit tables, GF(2) composite walker, soundness constants | M2 |
| `crates/agg-keccak` | lib | Flock HEAD Keccak gate port, sponge union circuit, bridge slot, segment driver | M3 |
| `crates/agg-cli` | bin `agg` + lib | prove/verify, wire format, record binding, caches | M2/M3 |
| `crates/agg-bench` | lib + benches | prover, verifier (cached/uncached), memory | M2/M3 |

Other files: `docs/` (spec and spec checks), `scripts/test.sh`, `scripts/bench.sh`,
`rust-toolchain.toml` (stable), `.cargo/config.toml` (no `target-cpu=native`).

## Pinned revisions

All pins live in `[workspace.dependencies]` of `Cargo.toml`; `Cargo.lock` is committed
and was seeded from Aerie's lockfile so shared transitive crates match Aerie's.

| Dependency | Source | Revision |
| --- | --- | --- |
| Aerie (`aerie-falcon`, `aerie-prover`, `aerie-pcs-config`, `aerie-keccak`, `aerie-simd`) | https://github.com/a16z/aerie | `49b6c212c209286c536196f53618823ffdf0e899` |
| Akita (`akita-*`) | https://github.com/LayerZero-Labs/akita | `e023b6a46143f7b23199ccecb2684192baf7b640` (the rev Aerie 49b6c21 pins; `akita-config` with `dev-protocol`, transcripts with `transcript-blake2b`) |
| Jolt (`jolt-field` with `solinas`, `jolt-poly`) | https://github.com/a16z/jolt | `c7c2d08d5bb0c7c80501f29f723e595af7fcff3c` (as Aerie) |
| Spongefish | https://github.com/arkworks-rs/spongefish | `ef9741346a150039427d9e0c02d6b2d73e93ec81` (as Aerie) |
| Flock (`flock-core`, `flock-prover`, `flock-field`) | https://github.com/succinctlabs/flock | `83f963e8ede6d9a85ec4762b0c018943b6f6542b` (HEAD circuit language) |
| `ml-dsa` (RustCrypto) | crates.io | `=0.1.1` (key generation and signing) |
| `sha3` | crates.io | `0.10.9` (reference SHAKE) |

### Which Aerie crates are importable as git dependencies

Checked by building this workspace (2026-10-24): all five Aerie crates resolve and
compile as git dependencies of a workspace member. Cargo resolves Aerie's internal path
dependencies and `workspace = true` inheritance inside the git checkout, so the
transitive `aerie-falcon-native`, `aerie-falcon-statement` and `aerie-falcon-witness`
come along too (and can be added directly if needed).

| Crate | Importable | Notes |
| --- | --- | --- |
| `aerie-falcon` | yes | with feature `internals` (exports `BaseField`, `OpeningField`, `PublicInputs`, openings); `default-features = false`, `parallel` forwarded from `agg-piop` |
| `aerie-prover` | yes | with `internals` it re-exports the protocol machinery we reuse: `projection`, `stage1`..`stage3`, `transcript`, `pcs`, `openings`, `ring`, `wire`, `manifest` |
| `aerie-pcs-config` | yes | default feature `schedules-fp64-aerie` (generated catalogs); `configs::Fp64SignedWidth4` |
| `aerie-keccak` | yes | runtime-dispatched batched SHAKE256 (`shake256_64_batch`, etc.) |
| `aerie-simd` | yes | runtime ISA ceiling `Backend::selected()`; `backend-override` is test-only upstream |

Aerie's test-only features (`honest-vectors`, `test-fixtures`, `backend-override`) are
exist upstream but are not enabled here (not yet tried as git features).

## Build and test

```sh
# first build is long (Aerie + Akita + Jolt + Flock); reuse target/, never `cargo clean`
taskset -c 0,1,3-15 cargo build --release --workspace
scripts/test.sh                 # = taskset -c 0,1,3-15 cargo test --workspace
scripts/test.sh --release       # for heavy tests
scripts/bench.sh                # local calibration benches (agg-bench)
AVX512=1 scripts/bench.sh       # opt-in AVX-512 build in target/avx512
./target/release/agg            # CLI skeleton
```

If the stable toolchain from `rust-toolchain.toml` is not what rustup resolves, set
`RUSTUP_TOOLCHAIN=stable` (the scripts do). Aerie's own MSRV is 1.95.

### SIMD policy

The default build has no `target-cpu`/`target-feature` flags. The `avx512` cargo
feature (in `agg-piop`, `agg-keccak`, forwarded by `agg-bench`) is opt-in:

- Aerie's kernels (`aerie-simd`, `aerie-keccak`) are runtime-dispatched with a scalar
  fallback, independent of our flags; the selected ceiling is reported by
  `agg_piop::aerie_simd_backend()` and printed by every benchmark.
- Flock's GF(2^128) AVX-512/VPCLMULQDQ kernels are compile-time (`target_feature`);
  `AVX512=1 scripts/bench.sh` sets `-C target-cpu=x86-64-v4 -C target-feature=+vpclmulqdq,+pclmulqdq`
  in a separate target dir (`target/avx512`). A bare `+avx512f,...` feature list made rustc
  1.99.1 crash (SIGILL in LLVM DAG combine) while compiling `aerie-prover`.
- Our own AVX-512 kernels (none yet) go behind `#[cfg(feature = "avx512")]`.

## Machine rules (see ../AGENTS.md)

- Local box (i7-11700K, 30 GB): tests and small calibrations only, never on CPU 2.
  Prefix every cargo/test/bench command with `taskset -c 0,1,3-15` (the scripts do).
- Local batch sizes: at most 1K signatures and 2^14 Keccak permutations.
- Never change machine settings. Large runs (32K signatures, 64-128 GB) run on a Google
  Cloud C4D (AMD Zen 5) VM, deleted as soon as the run is done.
- Benchmarks report medians and the exact git revision; never invent numbers.
