# Patch against succinctlabs/flock 83f963e8ede6d9a85ec4762b0c018943b6f6542b

Only `src/pcs/jagged.rs`, function `build_merged_weight_and_prime` (the merged opening's
weight vector `W`), is changed; `Cargo.toml` inlines the workspace-inherited fields and
points the sibling Flock crates at the pinned git revision. Everything else is a verbatim
copy of `crates/flock-core` at that revision (`src/`, `configs/`).

Why: Flock bakes, for every ring-switched claim, a 16 x 256-entry fold table (64 KB,
4,096 fold evaluations) per used word column of the committed stack, whatever the column's
height. A one-row table type with `2^16` word columns (the bridge instance of
`agg-keccak`, `docs/keccak.md` Section 9) then spends 3.4 s and 6.4 GiB in this step for a
stack of `2^16` words, and a full segment would need 51 GiB. The patch evaluates such
claims per slot instead, `φ(eq_col[c]·r)` with one multiply and one fold-table lookup,
whenever the stack averages fewer than `BAKE_MIN_HEIGHT = 64` live rows per used column
(`ColSide::Short`). Flock's own shapes (hundreds of rows per column) keep the bake.

Prover-only and exact: `x ↦ φ(eq_col[c]·x)` is F₂-linear, so the baked byte decomposition
and the direct evaluation compute the same `W` bit for bit; the proof bytes, the transcript
and the verifier are unchanged, and proofs made with this copy verify with upstream Flock.

To regenerate: copy `crates/flock-core/{src,configs}` of the revision above here, write the
`Cargo.toml` of this directory, and re-apply the `ColSide::Short` hunk (three edits in
`build_merged_weight_and_prime`: the enum variant and the `bake_columns` decision, the
`Folded` claim arm, and the segment sweep arm).
