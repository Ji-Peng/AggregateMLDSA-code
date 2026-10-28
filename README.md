# AggregateMLDSA-code

Implementation of the AggregateMLDSA project: aggregation of ML-DSA (FIPS 204)
signatures following Aerie (sum-check PIOP + Akita lattice PCS, no recursion),
with a Flock-based argument for the witness-dependent SHAKE256 call.
The protocol source of truth is [`docs/spec.md`](docs/spec.md); deviations from it are
recorded in `docs/deviations.md`.

Implemented: ML-DSA-44, mode A (public `c~`) and mode B (public bits), PIOP wire version 4
(tied-part proof version 5: the GF(2) side of the bridge is the virtual parity chain,
[`docs/parity-chain.md`](docs/parity-chain.md); fork-join transcript: the Flock proofs are
absorbed at a join after stage 3 and proven on their own thread pool beside the PIOP,
`AGG_FLOCK_THREADS`). Three relations, selected by `--relation` and bound in event 0:
**H-u** (default; `z` in `[-2^17, 2^17)`, `|u| <= 190,465`, SelfTargetMSIS bound
`zeta' = 350,209`) and **H-exact** (also `|z| <= 130,993`, FIPS 204's bounds; one more
committed column) and **H-FIPS** (relation byte 2, `--relation h-fips`: also the hint, so the
extracted witness is a FIPS 204 Algorithm 8 signature; four more columns and one value per
signature, `docs/h-exact.md`). Every Fiat-Shamir field site carries Aerie's per-site
proof-of-work schedule ([`docs/soundness.md`](docs/soundness.md): census, schedule, the
128-bit theorem and its composition proof, conditional on the stated assumptions (F) and
(K)). Aggregate unforgeability reduces to SUF-CMA of the relaxed scheme, with no open step
for signing queries asked only in rewound runs (`docs/soundness.md`); under H-FIPS it reduces
to SUF-CMA of ML-DSA-44 itself, an assumption (FIPS 204 Section 3.1 states it as a design
goal; `docs/h-exact.md`).

## Layout

| Crate | Kind | Content |
| --- | --- | --- |
| `crates/agg-witness` | lib | ML-DSA-44 native verify, key expansion, (z,u,v,b) derivation, digit and value forms, CRT product, padding geometry, H-FIPS witness and checker |
| `crates/agg-pcs-config` | lib | Aerie's `Fp64SignedWidth4` preset and the width-16 preset `AggFp64SignedWidth16`, six-group catalogs in two families for both relations (`Relation`), two-round commit driver, work plan |
| `crates/agg-piop` | lib | the LogUp-GKR lookup into `T_16` (`docs/lookup.md`), stages 1-3, the two remaining range leaves and links, weight tables, the proof-of-work schedule (`pow`), verifier, Akita openings |
| `crates/agg-bridge` | lib | bridge shapes (production, toy), seed to matrices, Fp parities, quotient (carry) columns and carry digits, public weights and their verifier evaluation, final vector, GF(2) composite walker, the virtual parity chain (`parity_chain`), soundness constants and work |
| `crates/agg-keccak` | lib | Flock HEAD Keccak gate port, sponge union circuit, tied parts (version 5: the sponge instance whose opening proves the parity chain's bit claim, `flock_union`, `tied`; spans over several segments); `examples/gen_slim8_configs.rs` regenerates the `Slim8` configs |
| `crates/agg-cli` | bin `agg` + lib | mode-A aggregator and verifier (`mode_a`), Flock parts and segment items (`wire`), persistent key cache, batch files, the `agg` tool (`keygen-batch`, `sign-batch`, `aggregate`, `verify`, `bench`); `docs/cli.md` |
| `crates/agg-bench` | lib + benches | prover, verifier (cached/uncached), memory |

Other files: `docs/` (`spec.md` and `scripts/spec_checks.py`; `soundness.md` with
`scripts/pow_plan.py` and `scripts/fold_support.py` for the proof-of-work schedule and the
theorem; `h-exact.md` for relations H-u and H-exact; `witness.md`; `pcs-config.md` for the
Akita presets, catalog status and regeneration; `piop.md` for the PIOP tables and
commitments; `lookup.md` for the range lookup; `bridge.md` and `keccak.md` for the parity
bridge and the Flock side; `cli.md` for mode A end to end and the `agg` tool;
`perf.md` for the local prover profile, the speed levers and the verifier kernels
(direction only);
`deviations.md`), `scripts/test.sh`, `scripts/bench.sh`,
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
| Flock (`flock-core`, `flock-prover`, `flock-field`) | https://github.com/succinctlabs/flock | `83f963e8ede6d9a85ec4762b0c018943b6f6542b` (HEAD circuit language); `flock-core` is the vendored copy `vendor/flock-core` of that revision with the prover-only opening patch, runtime-dispatched x86 SIMD kernels, the derived rate-1/8 Ligerito parameter set `Slim8`, the verifier-visible list-aware work at the outer sites and the canonical work-nonce width of `vendor/flock-core/PATCH.md` (`docs/deviations.md`; our Flock proofs do not verify with upstream), applied through `[patch]` in `Cargo.toml`; `flock-field` is the vendored copy `vendor/flock-field` with `#[inline]` on the x86 scalar GF(2^128) multiply (exact, speed only, `vendor/flock-field/PATCH.md`) |
| `ml-dsa` (RustCrypto) | crates.io | `=0.1.1` (key generation and signing) |
| `sha3` | crates.io | `0.10.9` (reference SHAKE) |

### Which Aerie crates are importable as git dependencies

All five Aerie crates resolve and compile as git dependencies of a workspace member. Cargo resolves Aerie's internal path
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

Aerie's test-only features (`honest-vectors`, `test-fixtures`, `backend-override`)
exist upstream but are not enabled here (not yet tried as git features).

## Build and test

```sh
# first build is long (Aerie + Akita + Jolt + Flock); reuse target/, never `cargo clean`
taskset -c 0,1,3-15 cargo build --release --workspace
scripts/test.sh                 # = taskset -c 0,1,3-15 cargo test --workspace, then the vendored flock-core lib tests (target/flock-core; skipped with -p or SKIP_FLOCK_CORE=1)
scripts/test.sh --release       # for heavy tests
scripts/bench.sh                # local calibration benches (agg-bench)
scripts/bench.sh --bench mode_b -- 256 512 1024 [--warmup 1] [--reps 5]   # mode-B prove/verify/RSS/size (docs/piop.md)
AGG_KECCAK_WARMUP=1 AGG_KECCAK_REPS=5 taskset -c 0,1,3-15 cargo bench -p agg-keccak --bench keccak_sponge   # Keccak sponge proof
AVX512=1 scripts/bench.sh       # opt-in AVX-512 build in target/avx512
./target/release/agg            # mode-A tool: keygen-batch, sign-batch, aggregate, verify, bench (docs/cli.md)
./target/release/agg bench --count 1024 [--warmup 1] [--reps 5]   # mode A: median of 5 after 1 warmup
```

If the stable toolchain from `rust-toolchain.toml` is not what rustup resolves, set
`RUSTUP_TOOLCHAIN=stable` (the scripts do). Aerie's own MSRV is 1.95.

### SIMD policy

The default build has no `target-cpu`/`target-feature` flags. The `avx512` cargo
feature (in `agg-piop`, `agg-keccak`, forwarded by `agg-bench`) is opt-in:

- Aerie's kernels (`aerie-simd`, `aerie-keccak`) are runtime-dispatched with a scalar
  fallback, independent of our flags; the selected ceiling is reported by
  `agg_piop::aerie_simd_backend()` and printed by every benchmark.
- Flock's scalar GF(2^128) multiply is compile-time (`target_feature`); its x86 SIMD
  kernels (AVX-512/VPCLMULQDQ NTT and zerocheck, GFNI) are runtime-dispatched in the
  vendored `flock-core` (`vendor/flock-core/PATCH.md`; `FLOCK_SIMD=portable|gfni`
  caps them for A/B runs). The recommended opt-in
  build is `CLMUL=1 scripts/bench.sh` (`-C target-feature=+pclmulqdq`, target dir
  `target/clmul`): only Flock's carry-less path changes; Akita, Jolt and Aerie keep the
  baseline codegen. Do not add `+sse4.1`: it slows our lookup and stage 3 by 6-7% locally
  (`docs/perf.md`); the vendored `flock-field` needs only `pclmulqdq` to inline its scalar
  GF(2^128) multiply (`vendor/flock-field/PATCH.md`). Locally, mode A proves and verifies
  about an order of magnitude faster than in the default build at N = 1,024, and the Keccak
  bench alone reaches most of the throughput of the full AVX-512 build.
- `AVX512=1 scripts/bench.sh` (`-C target-cpu=x86-64-v4 -C target-feature=+vpclmulqdq,+pclmulqdq`,
  `target/avx512`) is broken for everything that commits with Akita: with x86-64-v4 Akita's
  compile-time `avx512bw` path (`akita-cpu-backend/src/sources/poly_helpers/windowed_accum.rs`)
  makes its prover self-check fail ("relation session final claim disagrees with its folded
  oracle"), on this i7-11700K and on a Zen 5 VM; `AKITA_SCALAR_NTT=1` does not help,
  `x86-64-v3` passes. Use it only for the Keccak-only bench. A bare `+avx512f,...` feature
  list makes rustc 1.99.1 crash (SIGILL in LLVM DAG combine) while compiling `aerie-prover`.
- Our own AVX-512 kernels (none yet) go behind `#[cfg(feature = "avx512")]`.

## Machine rules

- Local box (i7-11700K, 30 GB): tests and small calibrations only, never on CPU 2.
  Prefix every cargo/test/bench command with `taskset -c 0,1,3-15` (the scripts do).
- Local batch sizes: at most 1K signatures and 2^14 Keccak permutations.
- Never change machine settings. Large runs (32K signatures, 64-128 GB) run on a Google
  Cloud C4D (AMD Zen 5) VM, deleted as soon as the run is done.
- Benchmarks report medians and the exact git revision; never invent numbers.
