# Patch against succinctlabs/flock 83f963e8ede6d9a85ec4762b0c018943b6f6542b

One change; `Cargo.toml` inlines the workspace-inherited fields (package metadata, the
`serde` dependency, the clippy lints) and allows dead code (a path dependency shows warnings
that Cargo hides for a git dependency). Everything else is a verbatim copy of
`crates/flock-field/src` at that revision.

## 1. Prover-only (and verifier): inline the x86 scalar GF(2^128) multiply (agg-mldsa `docs/perf.md` Section 10)

`src/gf2_128/x86_64.rs`: `#[inline]` on `ghash_mul_karatsuba_barrett` (the x86 `F128 * F128`),
`ghash_mul_unreduced_x86` (the 256-bit unreduced product of the deferred-reduction dots and
folds) and `ghash_square_x86`. Nothing else changes.

Why: upstream declares the three functions `#[target_feature(enable = "pclmulqdq,sse4.1")]`
without `#[inline]`. The `#[inline]` wrappers in `gf2_128.rs` (`impl Mul for F128`,
`ghash_mul_unreduced`, `ghash_square`) inline into flock-core, but the bodies stay in
flock-field's object code: a non-generic, non-`#[inline]` function of another crate is only
inlined under LTO, and LLVM never inlines a callee whose target features exceed the caller's.
Upstream benches with `lto = "thin"` and `-C target-cpu=native` (its `[profile.bench]` and
`.cargo/config.toml`), so it does not see this. Our CLMUL timing build has no LTO and only
`+pclmulqdq`, so every `F128` multiply in flock-core was an out-of-line call (3.87% self time
in a local 8K profile). With `#[inline]` and `-C target-feature=+pclmulqdq,+sse4.1` (every
PCLMULQDQ CPU has SSE4.1) the bodies inline into every caller. Thin LTO plus `+sse4.1` also
inlines them, but slows our PIOP lines by 3-6% locally (perf.md Section 10), hence this
patch instead.

Exact: the same instructions compute the same field elements, so proofs, transcripts and the
verifier are unchanged and interoperate with upstream Flock. The default build (no
`pclmulqdq`) does not compile `x86_64.rs`'s multiplies into any call path and is unchanged.

To regenerate: copy `crates/flock-field/src` of the revision above here, write this
`Cargo.toml`, and add `#[inline]` above the `#[target_feature]` attribute of the three
functions.
