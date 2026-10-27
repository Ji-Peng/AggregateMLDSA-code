# Patch against succinctlabs/flock 83f963e8ede6d9a85ec4762b0c018943b6f6542b

One change; `Cargo.toml` inlines the workspace-inherited fields (package metadata, the
`serde` dependency, the clippy lints) and allows dead code and unused imports (a path
dependency shows warnings that Cargo hides for a git dependency). Everything else is a
verbatim copy of `crates/flock-field/src` at that revision.

## 1. Inline the x86 scalar GF(2^128) multiply (prover and verifier speed; agg-mldsa `docs/perf.md` Section 10)

`src/gf2_128/x86_64.rs` only:

- `#[inline]` on `ghash_mul_karatsuba_barrett` (the x86 `F128 * F128`),
  `ghash_mul_unreduced_x86` (the 256-bit unreduced product of the deferred-reduction dots
  and folds) and `ghash_square_x86`;
- their `#[target_feature]` and that of the helper `pmull` is `pclmulqdq` instead of
  `pclmulqdq,sse4.1`;
- the helpers `lane0`/`lane1` extract a 64-bit lane with the SSE2 intrinsics
  `_mm_cvtsi128_si64` / `_mm_unpackhi_epi64` instead of SSE4.1's `_mm_extract_epi64`, and
  carry no `#[target_feature]`. In LLVM IR both forms are the same `extractelement`; LLVM
  emits `pextrq` wherever SSE4.1 is enabled (the AVX-512 kernels) and `movq`/`punpckhqdq`
  elsewhere.

Nothing else changes; the other SSE4.1 users of the file (`ghash_mul_schoolbook`,
`ghash_mul_binius`, `ghash_mul_karatsuba`, the AVX-512 accumulator fold) keep their features
and call the helpers as before.

Why: upstream declares the three functions `#[target_feature(enable = "pclmulqdq,sse4.1")]`
without `#[inline]`. The `#[inline]` wrappers in `gf2_128.rs` (`impl Mul for F128`,
`ghash_mul_unreduced`, `ghash_square`) inline into flock-core, but the bodies stay in
flock-field's object code: a non-generic, non-`#[inline]` function of another crate inlines
only under LTO, and LLVM never inlines a callee whose target features exceed the caller's.
Upstream benches with `lto = "thin"` and `-C target-cpu=native` (its `[profile.bench]` and
`.cargo/config.toml`), so it does not see this. Our CLMUL build has no LTO and only
`+pclmulqdq`, so every `F128` multiply in flock-core was an out-of-line call (3.87% self time
in a local 8K profile). Thin LTO plus a global `+sse4.1` would inline them too, but a global
`+sse4.1` slows our lookup and stage 3 by 6-7% locally (measured, perf.md Section 10). With
this patch the bodies inline into every caller of the `+pclmulqdq` build: 6,903 pclmul
instruction lines in `agg` instead of 461; the unreduced multiply and the square have no
out-of-line copy left; the Karatsuba multiply keeps 18 out-of-line calls, all in verifier
functions.

Exact: the same carry-less products and XORs compute the same field elements, so proofs,
transcripts and the verifier are unchanged and interoperate with upstream Flock (aggregates
byte-identical to main, perf.md Section 10). The default build (no `pclmulqdq`) does not
reach these functions and is unchanged. The crate's own tests pass under `+pclmulqdq`
(`cargo test --release --manifest-path vendor/flock-field/Cargo.toml`: 29 passed).

To regenerate: copy `crates/flock-field/src` of the revision above here, write this
`Cargo.toml`, and apply the edits listed above (each carries an `agg-mldsa patch` comment).
