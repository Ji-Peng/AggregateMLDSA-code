# Patch against succinctlabs/flock 83f963e8ede6d9a85ec4762b0c018943b6f6542b

Three changes; `Cargo.toml` inlines the workspace-inherited fields (package metadata, the
`serde` dependency, the clippy lints) and allows dead code and unused imports (a path
dependency shows warnings that Cargo hides for a git dependency). Everything else is a
verbatim copy of `crates/flock-field/src` at that revision, plus the new module of section 3.

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
byte-identical to main, perf.md Section 10). A build without `pclmulqdq` does not
reach these functions and is unchanged. The crate's own tests pass under `+pclmulqdq`
(`cargo test --release --manifest-path vendor/flock-field/Cargo.toml`: 29 passed).

To regenerate: copy `crates/flock-field/src` of the revision above here, write this
`Cargo.toml`, and apply the edits listed above (each carries an `agg-mldsa patch` comment).

## 2. Register-resident x86 scalar arithmetic (prover and verifier speed)

`src/gf2_128/x86_64.rs` gains `to_xmm`/`from_xmm`, `clmul_256_xmm`, `reduce_xmm`,
`ghash_mul_xmm`, `ghash_mul_unreduced_xmm`, `ghash_reduce_xmm` and `ghash_square_xmm`;
`src/gf2_128.rs` maps `F128 * F128`, `mul_unreduced`, `square` and, under `+pclmulqdq`,
`F256Unreduced::reduce` to them, and gives `F256Unreduced` `#[repr(C, align(16))]` so that its
`{r0, r1}` and `{r2, r3}` halves are two XMM values.

Why: the forms of section 1 move every operand qword into an XMM register for `pclmulqdq`
and every partial product back into general-purpose registers for the XORs and the
shift-based reduction, about ten cross-domain moves per multiply. The new forms reinterpret
`F128` (`repr(C, align(16))`, `lo` first) as an `__m128i`, select the qwords with the CLMUL
immediate (schoolbook, four products) and reduce with two CLMULs by `0x87`. Measured on the
i7-11700K, one core: butterfly 3.2 -> 1.7 ns, unreduced dot product 2.1 -> 1.0 ns per term.

Exact: a canonical residue modulo `x^128 + x^7 + x^2 + x + 1` is unique, so every reduction
route yields the same bits (test `xmm_forms_match_software`, including operands with the top
bits set). Proofs and transcripts are unchanged (aggregates byte-identical to main).

## 3. Sum-check round kernels for the parity chain (`src/f128_slice/rounds.rs`, prover speed)

A new module, declared by `pub mod rounds;` in `src/f128_slice.rs`: the round message
`(q(1), q(inf))` of a product of two `F128` tables over their lowest variable, and the fold
of both tables by a challenge fused with the next round's message, for pair tables
(`pair_message`, `fold_pairs_two_and_message`) and block tables `[block][row]`
(`block_message`, `fold_blocks_two_and_message`, `fold_blocks_two`). The parity chain's
prover (`crates/agg-bridge/src/parity_chain.rs`) calls them; Flock's own code does not.

Every function is safe and has two paths: AVX-512 kernels (`avx512f` and `vpclmulqdq`,
detected once at run time, allowed by the `AGG_ISA` cap and turned off by
`FLOCK_SIMD=portable|off`) and portable loops. Messages accumulate unreduced 256-bit
carry-less products and reduce once per call (exact: GF(2) addition has no carry), so both
paths return the same bits (tests `pair_kernels_match_reference`,
`block_kernels_match_reference`). Prover-only: proofs, transcripts and the verifier are
unchanged.

`f128_slice::fold_pairs` also selects its four-lane VPCLMULQDQ kernel at
run time using the same CPU detection, `AGG_ISA` cap and `FLOCK_SIMD` override.
The portable and vector paths compute the same field elements.
