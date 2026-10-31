# Patch index

The normative changes of each vendored tree are recorded in its own file. This
index links them and records the changes that belong to the unified build rather
than to one tree.

| Tree | Upstream | Record |
| --- | --- | --- |
| `vendor/akita` | `LayerZero-Labs/akita` at `e023b6a` | [`vendor/akita/PATCH.md`](vendor/akita/PATCH.md) |
| `vendor/flock-core` | `succinctlabs/flock` at `83f963e` | [`vendor/flock-core/PATCH.md`](vendor/flock-core/PATCH.md) |
| `vendor/flock-field` | `succinctlabs/flock` at `83f963e` | [`vendor/flock-field/PATCH.md`](vendor/flock-field/PATCH.md) |

## Unified build changes

- **One SIMD baseline.** `.cargo/config.toml` carries
  `-C target-feature=+pclmulqdq`; every higher instruction set is selected at
  run time under `AGG_ISA` (see README "SIMD policy"). The retired
  `x86-64-v3` global target is replaced by run-time dispatch in
  `vendor/akita/crates/akita-cpu-backend/src/sources/poly_helpers/`.
- **Akita window gather.** The compile-time `avx512bw` branch of
  `windowed_accum.rs` is deleted. It changed `GATHER` from one window to two,
  which failed the prover self-check; the gather width is now the same under
  every target.
- **Akita AVX2 decompose-fold.** `decompose_fold_avx.rs` compiles under every
  x86-64 build and is chosen by `is_x86_feature_detected!("avx2")` plus the
  `AGG_ISA` cap. `AKITA_SCALAR_NTT=1` still disables it.
- **Flock Frobenius tables.** `frob_tables` builds serially inside its
  `OnceLock`. A parallel build there deadlocks when several workers enter it at
  once: the waiting workers occupy the Rayon pool that the initializer's
  `par_iter` needs.
- **Flock 16-way BLAKE3 leaves.** The `AVX512BW` feature bit is part of the
  runtime mask; the leaf hasher itself is not adopted on this branch, because
  main's Merkle path already hashes whole groups with AVX-512F.
- **Plane PIOP layout.** `AGG_PIOP_LAYOUT=plane|row` selects the sum-check
  table layout independently of `AGG_ISA`; `plane` is the default only with the
  IFMA backend. Both layouts send identical messages.
