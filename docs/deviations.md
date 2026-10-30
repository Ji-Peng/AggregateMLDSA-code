# Deviations from `docs/spec.md`

Implementation choices that differ from, or make concrete, the wording of the
specification. Each entry names the spec section, what the code does, and why.

## D1. Own catalog family for the width-4 preset (spec Sections 4.1, 14)

Spec: every committed table uses "Aerie's `Fp64SignedWidth4` preset".

Code: `agg_pcs_config::preset::AggFp64SignedWidth4`, a separate type whose Akita
parameters are identical to Aerie's (decomposition `log_basis 3`,
`log_commit_bound 4`, `log_open_bound 64`; opening basis search 3..=6; inner basis
exactly 4, one balanced base-16 digit; source class `BalancedSignedDigit`; SIS profile
`Q64Offset59`; Dense ring-dimension schedule and challenge policy). A unit test checks
the equality field by field against `aerie_pcs_config::configs::Fp64SignedWidth4`.

Why: Akita binds a generated schedule catalog to the configuration type and its family
name. Aerie's catalog holds Falcon layouts only, so our
six-group layouts need their own catalog, `agg_mldsa_fp64_signed_width4`
(`crates/agg-pcs-config/src/schedules/`). The family name enters the catalog digest
that event 0 binds; the protocol is otherwise unchanged. See `docs/pcs-config.md`.
Under lookup layout L16 (entry L1) this catalog holds the scalar rows of
`G_main4` and the grouped rows of the six-group batches; the width-16 groups have their
own family (D2).

## D2. Second catalog family for the width-16 preset (spec Section 4.1, L16)

Spec (Section 4.1, L16): the 16-bit lookup columns, the carries and the level-2 parities are
committed under a signed width-16 preset.

Code: `agg_pcs_config::preset::AggFp64SignedWidth16`: the width-4 preset's parameters
except `log_commit_bound 16` (`[-2^15, 2^15 - 1]`), the inner (A/source) basis range
`(16, 16)` (one balanced base-`2^16` digit, entry L3) and the family name
`agg_mldsa_fp64_signed_width16`, whose catalog holds the scalar rows of `G_main16`, `G_1`,
`G_2`, `G_3`. A unit test checks that the certified interval (Akita's accepted bounds of the
digit geometry) is exactly the declared one. The grouped rows stay in the width-4 catalog,
the family of the final group `G_m`: Akita freezes the source bound into each group's own
profile, so a mixed-bound batch is keyed by the precommitted profiles of either family
(Akita's fp128 test `bounded_dense_precommit_with_onehot_final_group` exercises the same
path), and the width-4 catalog's rows also size the setup for the width-16 commits.

## W1. S1+S2 column count (spec Appendix A)

Spec: "Sixteen columns" for the `relation = fips` layout (a count that predates the
committed quotient and the base-16 range digits).

Code: no layout implements S1+S2; relation H-FIPS (L10) takes its place. Why: H-FIPS commits
no `r1` (the `w1'` bits carry it) and replaces the cubic UseHint check by zero-checks of
degree at most 3.

## W2. Padded rows of relations H-u, H-exact and H-FIPS (spec Section 4.2, Appendix A)

Spec: "for i >= N every value is zero", with the digits of zero at -b_k/2.

Code: padded rows hold the canonical encoding of the all-zero witness (z = 0, w' = 0, h = 0).
For z and v under H-u this is exactly the spec (digits -b_k/2, top signed digit 0); u = 0
takes the step form (B_ulo = 1,228, H_u = 3; spec 4.2). For H-exact, z = 0 is encoded
through the bit-slack form (a_z = 0, B_zhi = 1, B_zlo = 65,457; spec 4.3). For H-FIPS,
w' = 0 with h = 0 is the coefficient `hfips::Coeff::PADDED` and the hint slack is
u_h = omega = 80. Why: sum delta^2 + u_h = omega then holds on every row, so no live
selector is needed.

## W3. Orders and canonical choices left open by the spec (Sections 1, 4.1, Appendix A)

Fixed in `agg-witness` and documented in `docs/witness.md`: level-3 in-block index
zeta_3 = t + 6 rho_2 (least significant first); G_2 column order y2_0..y2_5 then
kappa2_{t,k} t-major; G_1 kappa1_{t,k} t-major; slack choice a = [x + offset >= 2^j].

## K1. One gate instance per signature sponge (spec Section 8)

Spec: "One Keccak gate type (one A_0 block, A = I (x) A_0); per record a 7-slot union
circuit with copy constraints between consecutive permutations", sub-batches of "2^16
slots", committed bits per record 6,144 + 7 * 38,400 = 274,944.

Code (`agg_keccak::sponge`): one gate instance (one A_0 block, `k_log = 19`) holds a
record's whole sponge, the 7 chi-only permutations chained by virtual XOR expressions;
the Flock union has one row per record (capacity 2^13 at a full segment, 57,344
permutations). Committed per record: 1 constant + 512 `mu` + 6,144 w1Encode + 256 `c~` +
268,800 chi bits = 275,713 (useful 275,968 of 2^19 with K11; the dense-stack
commitment stores only the 2,156 used words per row). Dummy rows are identically zero (Flock's count-derived
constant pin), not "a fixed all-zero permutation instance". Copy constraints are used for
the public words only.

Why: Flock's linear rows (lincheck) are per block, and copy constraints equate committed
128-bit words. Chaining permutations across slots would need committed 1,600-bit states
on both sides of every boundary (26 words per permutation) and still could not express
"output XOR absorbed block" as a word copy, so the chi-only witness with XOR chaining and
the `(t, j, rho)` input columns of Section 8 exist only when the 7 permutations share a
block. Per record this commits 275,776 bits against 7 * 41,792 = 292,544 for 7
single-permutation slots with committed states.

## K2. Walker-based gate instead of a DSL-lowered relation (spec Section 8)

Spec: "Flock HEAD circuit language", theta/rho/pi/iota as virtual XOR expressions.

Code: the relation is a Flock registry type with empty matrix stubs and its own
`LincheckCircuit` (`ChainGate`, the retired `c2d0c24` transpose walk extended across
permutations), the convention Flock uses for walker-based gates. The same relation is
authored in Flock's Boolean DSL (`xor`, `define_and`, `define_linear`) in
`src/dsl_ref.rs`, and tests check equality with Flock's lowering for reduced rounds.

Why: the DSL builder stores every XOR node's normalized support eagerly. With implicit
states the supports approach half of all committed columns: measured 67,214, 726,322,
4,747,519 and 13,152,431 support terms for 1 to 4 rounds of one permutation (estimate:
about 5e8 for 24 rounds and 2e10 for the 7-permutation sponge). The flattened matrices
have 6.7e8 nonzeros per permutation and 3.6e10 per sponge (exact, `docs/keccak.md`).
Binding: the registry digest covers only the stubs, so `ChainGate::digest` (rounds,
layout, absorptions, outputs) is absorbed into the Flock challenger; the PIOP's event 0
should also carry it.

## K3. Public inputs and constants of the sponge (spec Section 8)

Spec: "`public_input`: mu; `fixed`: capacity zeros and the padding bytes; `publish`: c~".

Code: `mu` (4 words) and `c~` (2 words) are committed cells of each record, wired to the
public segment (`[mu_i, c~_i]` per record, record order). The capacity zeros and the
padding bits are constants of the gate's linear expressions (through the constant
column), not public words.

## K4. Flock challenger seeding (spec Section 10, event 5)

Spec: the Flock sub-proof's challenger "is initialized with fs_sigma".

Code: `FsChallenger::new("agg-mldsa/v1/keccak")`, then absorb `fs_sigma` (32 bytes) and the
gate digest, before Flock's own statement binding. Why: Flock's challenger takes a domain
label at construction; absorbing the seed first is equivalent.

## K5. Row capacity of a sub-proof

Not in the spec. `nu = max(ceil(log2 n), 3, 22 - k_log)`, plus one when `n = 2^nu < 2^11`:
Flock needs `nu >= 3` and a dense stack of at least 2^22 bits, and a full-height stack
selects its block-first opening, whose cost is fixed by the 2,155 used columns (about 0.5 s
with CLMUL at any `n`, 26 s without). A function of `n`, so prover and verifier agree;
at a full 2^13-record segment the block-first path is kept.

## K7. Commit, then prove; the Flock commitment travels at event 2 only (spec Sections 8, 10)

Spec: event 2 absorbs the "S Flock commitments (Ligerito roots ...)", then `sd` (event 3)
and `fs_sigma` (event 5, the Flock challenger's seed) are drawn, and event 6 absorbs the
Flock proof byte strings; the Section 8 driver commits the union witness and hands the
commitment to the PIOP before it receives `sd` and `fs_sigma`.

Code (`agg_keccak::prover`):

- `GateBatch::commit` (`SpongeProver::commit_batch`, `commit_all`) computes the
  commitment before any seed exists and returns a `CommittedBatch` that owns the slot
  buffers; `prove_committed` (`prove_all`) takes `fs_sigma` later. Flock 83f963e's
  prover has no entry that accepts a precomputed commitment, so `commit` applies the same
  map as `prove_fast_ligerito_union_circuit` (single-slot passthrough, `compact_witness`
  unless the compaction is the identity, `commit_lane_major` or `commit`); Flock then
  commits again inside its prover (the Ligerito commit is deterministic) and
  `prove_committed` asserts that the two caps are equal. A unit test compares both caps on
  all four arms (identity or gathered compaction, lane-major or plain commit).
- The commitment is the Ligerito level-0 Merkle cap, `2^c` nodes of 32 bytes (`c = 7`,
  4,096 bytes, at every size tested), not one root: Flock binds and opens against the cap.
  `KeccakCommitment::as_bytes` is the event-2 payload.
- The proof bytes (format version 2) do not contain the commitment. `GateBatch::verify`
  and `SpongeProver::verify_batch` take the commitment absorbed at event 2, check its
  length (`Malformed`, not a panic) and pass it, with the verifier's own `PcsParams`, to
  Flock's verifier. With the commitment carried only at event 2, there is no second copy
  to compare: the equality holds by construction, and a different absorbed
  commitment fails Flock's opening (test
  `commitments_are_absorbed_before_the_seeds_and_bound`). On the prover side the accessor
  is `CommittedBatch::commitment` (`SpongeCommitted::commitment`).

Cost: one extra Ligerito commit per sub-proof (local calibration at 1,170 records: 0.046 s
of 0.94 s with CLMUL, 42 s of 121 s in the default build without CLMUL; `docs/keccak.md`
Section 6). The slot buffers of all `S` sub-batches stay in memory from event 2 to event 5
(estimate: about 2 GiB per full segment, `z`, `a`, `b` and the stripe at 512 MiB each, so
8 GiB at `N = 2^15`; a driver may keep only `z` and rebuild the rest). Removing the second
commit needs a Flock entry that takes the commitment and its `ProverData`.

For the tied part of the mode-A driver K14 applies instead: the commitment is computed once and
its prover data kept from event 2 to the join; the paragraphs above remain true of the
standalone sponge path (`GateBatch`, `SpongeProver`).

## P1. Manifest part of event 0 (spec Section 10)

Spec: event 0 carries "the four group descriptors with catalog row ids", the digit layout
id and the bridge parameters, without a byte encoding.

Code (`agg_piop::manifest::CommitmentManifest::binding_bytes`, `docs/piop.md` Section 2):
a fixed little-endian encoding of the wire version (1), parameter set, relation flag, `N`,
`N*`, digit layout id (1 = L16), then per group (six) its role, round, PCS family (0 width 4,
1 width 16), logical and physical variable counts, polynomial count and one alphabet byte per
polynomial (4 or 8 for a leaf-checked digit, `0x90` for a column looked up in `T_16`,
`0x50` for a multiplicity digit, 0 for a Boolean), then one catalog row id, then
`(J_t, m_t, k_t)`, then the lookup layout (challenge count, table widths, per stack the
column count, variables, table and storage shift). Only the grouped row id
(`mldsa-44-l16-batch-<N*>-mult-final`) is bound, not one id per group: Akita keys the
precommitted groups' scalar rows by their layouts, which the per-group fields already bind,
and the grouped row determines the final profile and the opening schedule. The public-record
digest and the Flock profile id are added by the transcript stage.

## P2. No divisibility projection (spec Section 6, Appendix B.1)

The implementation task asked for Aerie's projection `Pi_res` on the virtual residual
with its stage-1 linkage. The spec (Section 6, decision B.1) replaces it by the committed
quotient `v` and the field identity `Delta = 0` tested inside the stage-3 ring instance at
`r^(2)`, because an ML-DSA residual projection wraps modulo `p`. The code follows the spec:
there is no projection seed, Rice section, reverse map or stage-1 projection instance;
Aerie's `projection/` is not used. The quotient's range (`v` digits, Section 4.3) is what
lifts `Delta = 0` to (3.1).

## P3. Grinding ( spec Sections 10 and 12.1)

Spec: "the scheduled grinding [is] Aerie's; the work plan per admitted shape is an
obligation of `agg-pcs-config`."

Code (`docs/soundness.md`, normative): every PIOP field-challenge site
has a difficulty from Aerie's assigned bounds (`agg_piop::transcript::Registry`,
`agg_bridge::pow::grind_bits_k`); the transcript absorbs `LE32(nonce)`, checks the low `g`
bits of a 32-byte predicate with Akita's `grinding_predicate_accepts` and squeezes the
challenge after it; the nonces travel in one packed stream (`AggregateProof::work`, 227 to
556 bytes); the plan (tie sites, then PIOP sites) is bound in event 0 by its SHA3-256 digest
(`agg_piop::pow::PowPlan`). The sites of a tied part grind on the tie challenger
(tied-part proof version 5, `docs/soundness.md` 3.6). Akita's opening already ran its own plan
(Akita's instance descriptor, pinned by
`agg-pcs-config/tests/all/akita_grinding.rs`). Remaining differences from Aerie: (1) the nonce is a fixed-width field of a packed stream, not a LEB128 atom in a
Spongefish argument string (our PIOP has no argument string; the stream's width per site is
public, so the encoding is canonical); (2) the plan is a function of `N*`, the Flock profile
and the tied-part geometry rather than a per-shape table in `agg-pcs-config`; (3) the seeds
`sd`, `fs_sigma` and the PCS export carry no work (their bad sets are below `2^-128`,
`docs/soundness.md` Section 3.5).

## P4. Bridge matrices and the Keccak side are hooks (spec Sections 8, 9.1, 10, 11)

Spec: the verifier derives `M_1, M_2, M_3` from `sd` (Section 9.1) and verifies the Flock
sub-proofs against `mu_i, c~_i, Y_sigma` and the circuit digest (Section 11 step 3).

Code: `prover::BridgeDerivation::matrices(sd)` and `prover::KeccakSide` /
`verifier::KeccakVerifier` are trait hooks, so the wire format and the event order are the
same with and without `agg-keccak` (Flock sub-proofs); the transcript
events 2, 3, 5 and 6 are always present with `S` atoms.

- Matrices: `prover::SeededMatrices` is the derivation of Section 9.1
  (`agg_bridge::seed::derive_matrices`; `BridgeDerivation::seed_derived` is true for it
  only). Choices the spec leaves open: the `LE64(t)` of the hash input is the level number
  `t = 1, 2, 3`; a column's rows are stored in the order drawn; the XOF is read in 64-byte
  chunks (the word stream is the same). A known-answer test (column 0 of each level for
  `sd = 0^32`) was cross-checked against an independent pure-Python cSHAKE256. Cost: the
  2 x 2^19 + 49,152 columns take one Keccak-f each after a shared prefix state, and the
  matrices take 62 MB in each of prover and verifier. Measured on this machine (release,
  one test process, not a benchmark): median 150 ms over 7 derivations on 7
  threads (134 to 215 ms), 0.94 s on one thread. This is a fixed verifier cost (step
  `matrices`) that the estimates of spec Section 11 do not list; at small `N` it dominates
  mode-B verification. Reducing it needs a spec change (fewer XOF calls per column), not
  made here.
- Test stand-ins: `prover::FixedMatrices` (caller-supplied matrices whatever the seed),
  `prover::NoKeccak` and `verifier::NoKeccakVerifier` (empty Flock items, Flock profile 0)
  exist only in unit tests and with the crate feature `test-hooks`, which the integration
  tests enable through a self dev-dependency (`agg_piop::test_hooks_enabled()`). The
  prover, verifier and benchmark builds do not have them.
- Configuration policy (`verifier::check_configuration`, run before the proof is read;
  `VerifyError::Configuration`): the `KeccakVerifier` declares its Flock profile
  (`KeccakVerifier::profile`) and must match the requested one; mode A (profiles `Fast`,
  `Slim`, `Slim8`) requires `seed_derived()`, in every build, because with matrices known
  before `G_main` and the Flock commitments are fixed the bound of Section 9.4 does not
  apply; profile 0 (no Keccak argument, `c~_i` not tied to the committed bits) is refused
  unless the test hooks are compiled in. `prover::prove` refuses mode A over unseeded
  matrices as well (completeness only).

The production Flock `KeccakVerifier` is the mode-A driver's (`agg_cli::mode_a`, A1). Mode B
(`mode_b::verify_mode_b` with `SeededMatrices`) is sound through its own checks (P9). The
policy refuses mode A over any bridge other than `SeededMatrices`.

## P5. Public-record binding tree and instance digest (spec Sections 2, 10)

Spec: Aerie's 16-KiB-leaf cSHAKE256 tree under `agg-mldsa/v1/public-records-binding`.

Code (`agg_piop::binding`): the stream `LE64(N) || (pk || LE64(|M|) || M || c~)*` is cut into
16-KiB blocks, leaf `i = cSHAKE256(block_i, 32)` under `agg-mldsa/v1/record-leaf`, root
`cSHAKE256(LE64(blocks) || leaves, 32)` under the binding customization. Aerie's own
implementation hashes FN-DSA `hpk` values and salts and is not reusable for ML-DSA
records; this tree keeps its block size and customization scheme. The instance digest of
the Spongefish domain separator is `cSHAKE256(LE64(len) || record, 32)` under
`agg-mldsa/v1/instance`, as Aerie's `instance::digest`.

## P6. Transcript event framing (spec Section 10)

Spec: one event each for the round-2 commitments (event 4), the Flock seeds (event 5) and
the final vectors with the Flock proofs (event 6).

Code: event 1 is two registry events (`commit/main16`, `commit/main`) and event 4 four
(`commit/level1`, `commit/level2`, `commit/level3`, `commit/mult`), one per group, as
Aerie absorbs its groups; event 5 is one squeeze of
`32 S` bytes; event 6 is two events, the `512 S` bytes of `Y_sigma` (bits packed least
significant first) and the `S` Flock proofs each framed by `LE64` length; event 2 frames
the `S` Flock commitments the same way. Every challenge still follows the previous absorb
and the order of Section 10 is unchanged; only the registry granularity differs.

## P7. Instance orders left open by the spec (Sections 7.3 to 7.5)

Fixed in `docs/piop.md` Section 7 (lookup layout L16): stage 2 lists `def_y1[t]`, then
`leaf4[Z_hi]`, `leaf6[U_hi]`; stage 3 lists the ring, the two links, the 37 evaluation
reductions in the order of spec 7.5 item 3, then the four lookup-stack reductions (stack
order of Section 4.3); stage 1 as listed in 7.3. The pre-stage, terminal, `kappa1` and
multiplicity claim vectors are in the orders of `agg_piop::claims`.

## P8. Stage-3 ring instance as two prover groups (spec Section 7.5)

Spec: the ring instance (7.6) has two sums, one weighted by `w_A` in order `(j', s, i)`
and one by `eq(r^(2), .)` in order `(j, rho, i)`.

Code: both are on the main domain with the same variable order and the same `gamma^(3)_0`;
the prover evaluates them as two kernel groups (a weight-lane group and the `eq(r^(2))`
group that also holds the links and the `b_t` reductions) and the verifier as the sum of
the two terminal terms. The round messages are the sum of the two groups' messages, so
the wire format is that of one instance.

## P9. Mode B: the public bits as the Keccak side (spec Sections 10, 11, 14)

Spec: milestone M2 runs "the algebraic PIOP end to end with the bridge tables but `b`
published in the clear (mode B)"; the wire format, the transcript and the verifier of
Sections 10 and 11 describe mode A (Flock commitments and sub-proofs at events 2 and 6).

Code (`agg_piop::mode_b`, `docs/piop.md` Section 12):

- The event-2 item of segment `sigma` is the concatenation of `w1Encode(w_1'_i)` (768 bytes,
  FIPS 204 Algorithm 28) of the segment's live records; the event-6 proof items are empty;
  event 0 carries Flock profile id 3 (`FlockProfile::PublicBits`). The public bits are thus
  absorbed before `sd` and before every sum-check challenge, like the Flock commitment they
  replace, and the wire format and the event order are those of mode A.
- The verifier side (`PublicBitsVerifier`) replaces the Flock verification of Section 11
  step 3 by native checks: 768 bytes per live record; every 6-bit value below 44 (3.2);
  `c~_i = SHAKE256(mu_i || w1Encode(w_1'_i), 32)` (3.3, 7 Keccak-f per record); and the
  projection predicate `Y_sigma = M_3 M_2 M_1 b'` over GF(2) for the public bits `b'` (the
  equation the Flock bridge slot proves in mode A).
- After stage 3 it also checks `b_t(r^(3)) = b'_t(r^(3))` for `t < 6`: the committed bits'
  terminal claims, which the Akita opening authenticates, against the multilinear
  extensions of the public bit tables (`KeccakVerifier::check_bits`, 1,024 additions in `K`
  per record). The spec has no such step.

Why the extra check: the projection binds the committed bits to the public ones only
through seed-derived matrices (Section 9.4). With matrices the prover knows in advance
(the test stand-in `FixedMatrices`, P4), a prover can commit bits other than the public
ones with the same parities and absorb the change of `w_1'` in `u` (test
`committed_bits_other_than_the_public_ones_are_caught_by_the_bit_check`: with two equal
`M_1` columns every PIOP check and the projection pass). The direct check binds the bits
with error `d_main / |K|` (Schwartz-Zippel; the public bits are fixed before `r^(3)`)
whatever the matrices. It costs about 1 ms at `N = 1,024`. With `SeededMatrices`, the
derivation of mode-B prover and verifier builds (P4), the projection binds on its own up to
`2^-144.6`; the direct check is kept as a second, matrix-independent binding.

## P10. Verifier step 6 as two streamed loops (spec Section 11)

Spec: "Stream `A_i` ..., inverse-NTT, and accumulate `T(r^(2))` and `w_A(r^(3))` in one
pass over `A_i` by absorbing the stage-3 rounds first and deferring the terminal checks."

Code (`agg_piop::public_stream::StreamedPublic::pass`, `stage3::verify_streamed`): the
stage-3 rounds are replayed with the ring claim `T(r^(2))` set to zero; since compressed
rounds make the final claim affine in the initial claim with slope `prod_k r^(3)_k`, the
deferred term `gamma_ring T(r^(2)) prod_k r^(3)_k` is added once `r^(3)` is known. The pass
is two parallel loops after that point, one over the records (`T_i` recomputed from the
sparse `c_i`, `T(r^(2))`) and one over the distinct keys (`A` expanded or taken from the
cache, its `k l` products with `Lambda`, weighted by `sum_i eq(r2_sig, i) eq(r3_sig, i)`
over the key's records). Neither `A_i` nor `T_i` is stored for the batch. Why two loops:
`T_i` depends on the record, `A` only on the key, so the key loop expands each shared key
once and the record loop stays balanced when many records share a key.

## B1. Fp side of the bridge: choices left open by docs/bridge.md (Sections 1, 3, 4)

Code (`agg_bridge::{shape, parity, carry, weights, final_vector, soundness}`, used by
`agg_piop` at `Shape::PRODUCTION`):

- Toy-shape carry digits (tests only; `docs/bridge.md` gives the ranges, not the digits):
  unsigned bases `(8, 8)`, `(8, 4)`, `(4, 4)` for `[0, 63]`, `[0, 31]`, `[0, 15]`
  (`shape::TOY_CARRY{1,2,3}_DIGITS`). `Shape::validate` requires every level's digit range to
  be exactly `[0, floor(W_t / 2)]`.
- Honest-abort row weights (`Shape::row_weights_ok`) count the used columns only
  (`J_2^used = 2^17` at level 2), the weights that bound the level sums.
  `agg_witness::bridge::BridgeMatrices::row_weights_ok` counts all `2^19` columns of `M_2`;
  both accept every matrix whose all-column weights are within the caps (production means
  896, 224, 216 against 2,047, 2,047, 511), so no honest run differs.
- Carries are the quotient columns `floor(s / 2)` of the integer row sums `s`
  (`parity::split`), parities `s mod 2`; `carry::group_tables` refuses a carry outside its
  range instead of committing an out-of-alphabet top digit (the honest prover never has one).
- The verifier evaluates the stage-1 bridge weights `w_2(r^(1)_1)` and `w_{3,t}(r^(1)_2)`
  once per stage (`stage1::BridgeWeightsAt`), not once per lane; values unchanged.
- `agg_witness::bridge::compute` and `gf2_composite` stay as the independent production
  reference; the tests compare `agg-bridge` against them.

## P11. The Keccak hooks carry the bridge seed (spec Sections 8, 10, 11)

Spec: the bridge slot's circuit digest MUST cover `sd`, and the Flock rows are generated
from `M_1, M_2, M_3` of `sd` (Section 8 driver: "receive `sd` and `fs_sigma`, derive the
bridge rows, bind the statement").

Mismatch found by the joint bridge test: `prover::KeccakSide::prove` received only the
Flock seeds `fs_sigma` and the final vectors, and `verifier::SegmentContext` only `fs_sigma`
and the matrices, so a Flock side plugged into the PIOP could not bind the transcript's
`sd` into the bridge relation's digest and would have had to re-derive the matrices.

Code: `KeccakSide::prove(bridge_seed, matrices, seeds, final_vectors)` and
`SegmentContext::bridge_seed` pass the transcript's `sd` (event 3) and its matrices to the
Keccak side. Wire format, transcript and the mode-B and profile-0 sides are unchanged
(they ignore both). The mode-A driver (A1) runs the F_p side and the Flock side in one
transcript through these hooks; its end-to-end tests (`agg-cli/tests/all/mode_a.rs`) check that
the aggregate verifies with the Flock sub-proof checked inside the PIOP verifier and that one
flipped `w1Encode` bit fails exactly the side that holds it.

## K10. Flock parts within a segment (spec Sections 1, 8, 10)

A segment's Flock argument may use at most two tied parts. Each part proves its share
of the segment's final vector; the shares XOR to `Y_sigma` by linearity over GF(2).
The part layout and share identity are specified in [bridge.md](bridge.md), Section 5a.
The current share framing, seeds and transcript order are specified in K22.
Part-size options and defaults are in [cli.md](cli.md).

## K11. Tied parts (spec Section 8; docs/bridge.md Section 5a)

A part is one sponge instance and a virtual parity chain. The public evaluation claim,
regions and soundness argument are specified in [bridge.md](bridge.md), Section 5a.
The chain and its opening claim are specified in [parity-chain.md](parity-chain.md).

## K12. Vendored Flock patches (spec Section 8; pinned dependencies)

The workspace patches `flock-core` and `flock-field` through `Cargo.toml`.
The pinned revision, patch details, exactness arguments and regeneration instructions
are in [flock-core/PATCH.md](../vendor/flock-core/PATCH.md) and
[flock-field/PATCH.md](../vendor/flock-field/PATCH.md).
The list-aware work schedule and canonical nonce checks are verifier-visible.
Proofs do not interoperate with upstream Flock.

## K14. The Flock commitment computed once

Code: `flock_union::UnionGeometry::commit_data` returns Flock's `Commitment` with its
`ProverData` (the Reed-Solomon codeword and the Merkle tree); `UnionGeometry::prove(...,
pre: Some(data))` opens that commitment without committing again (the Ligerito commit is
deterministic, so the data is what the prover would recompute; tests show byte-identical
proofs with and without it). `TiedCommitted` keeps the sponge instance's data;
`agg_cli::mode_a` keeps it per part from event 2 to the join, while the witness buffers are
dropped after event 2 and rebuilt for the join only in a batch of
more than one part. The standalone sponge path (`prover::GateBatch`) still goes through Flock's own prover and commits twice.

Cost (measured locally): the kept data is the sponge instance's codeword and Merkle
tree, 19 / 52 / 84 / 152 MiB at 256 / 512 / 1,024 / 2,048 records with `Fast` (twice that
with `Slim`; extrapolated about 0.6 GiB per full segment, 2.4 GiB at `N = 2^15`, not
measured); it is the data the sponge proof would rebuild at event 6, so the prover's peak is
unchanged (1,942 MiB kept, 1,959 MiB recommitted at 2,048 records). Saved: one Ligerito commit
per part, 217 ms of the Flock part at 2,048 records (1,901 against 2,118 ms) and 58 ms at
1,024.

Lever W1 (`docs/perf.md` Section 9): a batch with exactly one tied
part (the span at `N >= 2^14`, one part at `N <= 2^13` under the default part size) keeps the
whole `TiedCommitted` (sponge buffers, input bits, statement and commit data) in `mode_a`'s
`PartState` from event 2 until its proof at the join, so the witness is built once. At most
one part's buffers are then alive trivially; a batch of more than one part still drops and
rebuilds them as above. Prover-only: the witness and statement are the ones the deterministic
builder would rebuild, so the aggregate is byte-identical (`cmp` of 256 / 1K / 2K / 8K
aggregates against `184374f`).

## K15. Ligerito `Slim` in mode A; the profile in the segment item (spec Sections 8, 10)

Spec Section 10: event 0 binds the Flock profile; Section 8: Ligerito `Fast` as the
engineering baseline.

Code: `FlockProfile::Slim = 4` (a mode-A profile: `is_mode_a`, so the configuration policy
of `agg_piop::verifier` requires the seed-derived matrices for it as for `Fast`);
`agg_cli::mode_a::AggregatorConfig::profile` and the CLI option `--flock-profile fast|slim`
choose it, `agg_cli::mode_a::keccak_profile` maps it to `agg_keccak::prover::Profile::Slim`
(the proof header's profile byte 3). The event-2 segment item carries the profile id
(`agg_cli::wire::SegmentCommitments::profile`, item version 4); the verifier
(`AggregateVerifier`) runs under the mode-A profile the items name (or the one fixed by
`require`, CLI `verify --flock-profile`), the transcript binds it at event 0, and an item
naming another profile than the verifier's is rejected before any Flock work
(`ItemError::Profile`).

Why: Flock documents `Slim` (rate 1/4) at the same level as `Fast`: 128 bits
work-normalized in the Johnson regime with out-of-domain binding and a 16-bit query proof of
work, every component of its accounting strictly above 128 bits
(`vendor/flock-core/src/pcs/ligerito.rs`, `LigeritoSecurityConfig::validate`); it halves the
level-0 queries (119 instead of 244), so a tied part's proof is 33 to 37% smaller
(`docs/keccak.md` Section 12) for about twice the level-0 encoding and twice the kept commit
data (K14). Measurements and the default: `docs/keccak.md` Section 12. By default the verifier
accepts either `Fast` or `Slim` because both meet the same documented level in upstream Flock;
`Slim8` (K18) is an opt-in mode-A profile; `Secure` (120 bits) is not a mode-A profile.

## K16. Spans (spec Sections 8, 9, 10; docs/bridge.md Section 5c)

`AggregatorConfig::span` defaults on. A batch of more than one segment with
whole-segment parts uses one tied part over all segments: one sponge instance and
one parity chain. The layout, seeds and wire items are specified in
[bridge.md](bridge.md), Section 5c. Sharing one opening limits proof size growth.

## K17. The Ligerito level-0 cap is not serialized (docs/keccak.md Sections 4, 12)

Flock's `LigeritoProof` carries `initial_cap`, the level-0 Merkle cap, which is the
commitment itself (the event-2 cap of the sponge instance); Flock's verifier rejects a proof whose `initial_cap` differs from the commitment's.

Code: `agg_keccak::flock_union::UnionProof::to_bytes` serializes the proof with an empty
`initial_cap`, `from_bytes` refuses a proof that carries one, and `UnionGeometry::verify` puts
the commitment's cap in its place before Flock's verifier runs. Prover, transcript and
verification are otherwise unchanged: an accepted proof is exactly one whose cap equals the
commitment's, so substituting it accepts the same proofs.

Why: sending the cap twice costs 4,096 bytes (`Fast`) or 2,048 bytes (`Slim`) per Flock
instance.

## K18. A rate-1/8 Ligerito parameter set, `Slim8`, in the vendored Flock core (spec Section 8)

K12: the vendored `flock-core` differs from Flock `83f963e` by a prover-only patch; Flock
ships the Ligerito profiles `Fast` (rate 1/2), `Slim` (rate 1/4) and `Secure`, plus two
100-bit compatibility profiles.

Code (`vendor/flock-core/PATCH.md` Section 2; `agg_keccak::prover::Profile::Slim8`, header
byte 4; `agg_piop::binding::FlockProfile::Slim8 = 5`, a mode-A profile; CLI `--flock-profile
slim8`): a sixth Ligerito profile with `Slim`'s accounting at level-0 rate 1/8, its ladder
3, 5, 7, ... (+2 per level as `Fast` and `Slim`), 16-bit query PoW, the same `initial_k`
rule. Its configs are produced by Flock's own `derive_profile` and checked by `validate()`
(every Johnson component strictly above 128 bits: query term `Q log2(1/(1 - gamma)) + 16`,
proximity gap over F256, two-point OOD binding, the algebraic terms of the Flock paper's
Appendix C.3); the generator asserts that the same derivation reproduces the shipped `Slim`
configs byte for byte. The five shipped profiles are unchanged.

Why: the level-0 queries dominate the opening (`docs/keccak.md` Section 12): rate 1/8 needs
79 instead of `Slim`'s 119 (per-query `log2(1/(1 - gamma))` 1.42 instead of 0.94 bits at
`gamma = 1 - sqrt(rho) - 0.02`), and the next levels 48, 35, ... instead of 60, 41, ..., for
twice `Slim`'s level-0 encoding (four times `Fast`'s) and twice its kept commit data (K14).
It is a parameter set of the documented analysis, not a profile Flock ships: proofs under it
verify with this vendored copy (or upstream with the same configs). Measurements and the
choice of the default: `docs/keccak.md` Section 12.

## K19. List-aware work at Flock's outer sites in the vendored core (spec 12.1; docs/soundness.md 4.3, 4.4)

Flock `83f963e`: under `Fast`, `Slim` (and our `Slim8`) every outer challenge of a Flock
proof (zerocheck initial and skip points and rounds, lincheck `alpha`, `beta`, rounds and
skip point, the ring-switch point, the transport claim batch, the merged rounds, the
multipoint batch and rounds) carries a nonce sized for its numerator alone, except the
transport claim batch, which prices the level-0 Johnson list with 6 bits. These challenges
are drawn before Ligerito's level-0 two-point binding, and the Flock paper's own accounting
(Appendix A "Security", Remark 11) multiplies each by the list size `L_max`, which leaves them
at about 122 to 124 work-normalized bits (`docs/soundness.md` Section 4.3); `Slim8`'s
claim batch reached 127.86 bits (4.4).

Code (`vendor/flock-core/PATCH.md` Section 5): `LigeritoProfile::l0_list_factor` =
`ceil(L_max)` = 36, 50, 71 for `Fast`, `Slim`, `Slim8` (`eta = 0.02`); the policies
`ZerocheckGrinding`, `LincheckGrinding`, `OpeningGrinding` and `MultipointGrinding` gain
`per_challenge_128_listed(list)`, which grinds `floor(log2(numerator * list)) + 1` bits at
every outer site (claim batch at least the former 6), and `PcsParams::{zerocheck,
lincheck, opening}_grinding` select them for the three Johnson profiles (`Secure`, unique
decoding, keeps upstream's schedule). Per profile `Fast / Slim / Slim8`: rounds 7 / 7 / 8,
`alpha`, `beta` and the claim batch 6 / 6 / 7, ring switch 8 / 9 / 9, zerocheck skip 13 / 13
/ 14, lincheck skip `k + 6 / k + 6 / k + 7` bits. The AG-basis skip points (not on our code
path) keep upstream's bits. No byte changes (Flock nonces are 8-byte words).

Consequence: the transcript validity rule of the Flock verifier changes, so the vendored
core is no longer prover-only: our Flock proofs do not verify with upstream `83f963e` and
upstream's do not verify here (mode A uses tied-part proof version
5, K21). The finding should be raised with the Flock authors, who may have an argument for
the outer sites that their audit document does not state; if so this patch can be dropped.

Canonical nonce width (`vendor/flock-core/PATCH.md` Section 6): the vendored
verifier also rejects a Flock work nonce of `g + 7` or more bits at a site of `g > 0` bits (and
a nonzero nonce at a site without work), as Akita and Aerie do and as our PIOP and tie nonces
do (`docs/soundness.md` 6.1 to 6.3). Upstream accepts any passing `u64`. Honest proofs are
unchanged except when Flock's smallest passing nonce is `2^(g + 7)` or more (probability at
most `2^-184.66` per site), which is a completeness failure (`docs/soundness.md` 6.5).

## K20. Tie work and the tie binding (tied-part proof version 5; docs/soundness.md 3.6, 4.5, 6.3)

The tie challenger grinds before drawing coordinates and squeezes a 32-byte binding
that the sponge challenger absorbs before Flock binds its statement.
The current site numerators and nonce format are in [parity-chain.md](parity-chain.md),
Sections 6–8. The label is `agg-keccak/v4/tie-binding`.

Parts per segment (mode A, `agg_cli::mode_a::MAX_PARTS_LOG2`): at most two tied parts per
segment, `d_part >= max(3, d_seg - 1)`. The bridge seed `sd` is drawn while each part's
sponge commitment is bound only to its Johnson list, so with `P` parts in a segment its bad
set is a union over `L^P` list tuples (`docs/soundness.md` 3.5): `L^2 theta <= 2^-132.4` is
fine, `L^3 theta = 2^-127.7` (`Slim`) is not. The prover clamps `part_log2` into the admitted
range, and the verifier refuses a smaller `d_part` while deriving the plan of event 0
(`VerifyError::Plan`).

Exhausted tie search: if Flock's smallest passing tie nonce is `2^(g + 7)` or more
(probability below `2^-184.6` per part), the nonce check returns
`KeccakError::Work` instead of panicking; `TiedPart::prove` and `prove_with_final_vectors` return it and mode A's Keccak side maps it to
`ProveError::Keccak` (`agg_piop::prover::KeccakSide::prove` returns a `Result`).

## K21. The virtual parity chain (tied-part proof version 5; spec Section 8; docs/bridge.md Section 5d; `docs/parity-chain.md`)

Code (`agg_keccak::tied`, version 5; `agg_bridge::parity_chain`): no bridge union. The tie
challenger draws `eq_Y = eq(r_o, rho_3) eq(r_y, sigma)` (12 + `span_log` coordinates) after the
sponge cap, the geometry and the final-vector digest, and the prover runs a degree-2 sum-check
of `<eq_Y, Y> = sum_x W(x) b''(x)` over the `13 + d + span_log` variables of the committed
bits (eight lanes: lanes 6, 7 are the sponge's word columns 48..63 with weight zero), with the
public weight `W = M^T eq_Y` never materialized: a block phase over `(i_mid, t, sigma)` from
the tables `G = M_2^T M_3^T eq_Y` and the honest level-1 parities `y1`, then an in-block phase
over the position. Each round absorbs `(q(1), q(inf))`, grinds and draws its challenge. The
final claim is the sponge's bit polynomial at the chain's point; its value `v` is absorbed,
the 32-byte tie binding squeezed, and the sponge instance's Flock opening proves `v` as a
third ring-switched claim (`flock_union::BitClaim`, `SkipPoint::Eq` of vendored Flock
PATCH.md Section 7, a multilinear skip basis). The verifier recomputes the claim from `Y`,
replays the rounds, checks `T_R = W~(r) eq(r_rec[nu_s..], 0) v` with its own `W~(r)` (about
`10^7` additions) and verifies the sponge proof with the public claim and the bit claim.
Proof bytes: `AGGKCT || 5 || profile || LE32(n) || d || S || LE32(tie nonce) || LE32(eq_Y
nonce) || rounds (q(1) || q(inf) || LE32(nonce)) || v || LE32(len) || sponge proof`.

Work (docs/soundness.md 3.6): the tie site carries `L_s (nu_s + 3)` (the public claim only),
the `eq_Y` site `L_s (12 + span_log)`, every round `2 L_s` (`agg_bridge::pow::part_work`);
the plan of event 0 names all of them per tied part (site classes 9 and 10,
`agg_piop::pow::KeccakPlan::parts`). The chain uses about 1 KB of rounds and one ring-switched claim (about 4 KB).
Its bit claim uses the full sponge column bit-bank. The vendored core skips
zero-weight columns when baking weights (PATCH.md Section 1).

Wire versions: PIOP 4, tied-part proof 5.

## K22. Fork-join transcript: the Flock proofs at a join after stage 3 (spec Sections 10, 11; docs/soundness.md 12)

Code (`agg_piop::{transcript, prover, verifier, wire}`, `agg_cli::{mode_a, wire}`):

- Event 6 absorbs `Y_0..Y_{S-1}` (site `flock/final-vectors`) and the `S` share items (new
  site `flock/shares`; empty for a span, a one-part segment and mode B). The `S` proof items
  are absorbed at the join (site `flock/proofs`), after the stage-3 terminal claims and
  before the PCS session export (event 13), so the session digest, which seeds Akita's
  stream, binds every Flock byte. The work plan and its digest are unchanged (no site moves
  or is added; the join and the share event carry no work).
- The part and span seeds bind `rec = cSHAKE256("agg-mldsa/v1/flock/record", LE64(S) ||
  Y_0 || .. || Y_{S-1} || framed share items)` (`transcript::flock_record_digest`):
  `part_seed = SHA3-256(label || fs_sigma || LE64(part) || rec)`, `span_seed =
  SHA3-256(label || LE64(S) || fs_0 .. fs_{S-1} || rec)`. Both challengers of a tied part
  absorb the seed first. Why: with the parts' stages after `X`, Lemma 12.3 (a) of
  docs/soundness.md needs every part address to encode the input of `X`, the whole event-6
  record. A span's table is all of `Y`, so a span alone would satisfy it without `rec`; the
  alternative is to admit only span layouts under this wire version. Binding `rec`
  keeps every layout (span off, two parts per segment, `--part-log2`) at the cost of one hash
  and no bytes.
- The shares are computed at event 6 from the part's input bits kept since event 2 (6 MiB
  per non-last part of a full segment), not from the rebuilt witness
  (`TiedPart::final_vectors_of_bits`).
- Prover schedule: after event 6, `KeccakSide::prove` runs on its own rayon `ThreadPool` (on
  its own OS thread, 64 MiB worker stacks) while the parity tables, the lookup, the
  pre-stage claims and the three stages run on the global pool; the join then absorbs the
  proofs (`prover::fork_join`). The pool has `AGG_FLOCK_THREADS` threads if set (`0`: no
  fork, serial execution), else 3/8 of the global pool rounded up: 6 of 16 on the cloud
  machine.
  On a rayon worker or with a one-thread pool there is no fork (the P12 pattern), nor for
  a side whose proof items are empty (`KeccakSide::proves_concurrently`: mode B, profile
  0). A non-numeric `AGG_FLOCK_THREADS` is reported once on stderr and replaced by the
  default; larger values are clamped to the larger of the logical CPUs and the global pool
  (`prover::max_flock_threads`).
  Both schedules give the same proof bytes (test `agg-piop/tests/fork_join.rs`).
- Verifier: it checks the Keccak side right after replaying event 6 (the verdict
  does not depend on the order of the checks, and the early check keeps precise rejection
  reasons), and absorbs the proof items at the join.
- Wire: the share items follow the final vectors, the proof items follow the stage-3 claims
  (before the nonce stream); `+4 S` bytes of framing for the share items. Item version 4,
  tied-part proof version 5 and PIOP wire version 4 identify the format.

Soundness: docs/soundness.md Section 12 (intro, 12.1, 12.2, the proof of Lemma 12.3, 12.3,
12.4, Corollary 12.5, 12.7 item 9): the tied parts' stages move from the segment of `sd` to
the segment of `X`, after the stage-3 rounds; the checkpoint test at `X` becomes `R_PIOP and
(for all p) C4_p`, `C4_p` decided by listing the Johnson list of the part's sponge word;
masses and the work plan are unchanged. The extractor runs that list decoder (Akita's
checkpoint node evaluates its test once per invocation), so its polynomial cost enters the
cost of one run, `T_run`, and the Module-SIS reductions, not the run count or the error
(soundness.md 12.6); the decoder's fit to Ligerito's level-0 code is cited, not checked
(12.7 item 9). Measurements are in [perf.md](perf.md).

## A1. Mode-A driver: inputs, checks and caches (spec Sections 2, 10, 11, 13)

Code (`agg_cli::mode_a::{Aggregator, AggregateVerifier}`):

- The aggregator takes `(pk_i, M_i, signature_i)`; `c~_i` is the first 32 bytes of the
  signature, and every signature is verified strictly while its witness is derived
  (`agg_witness::derive`, FIPS 204 Algorithm 8), so an invalid one is refused with its
  index before anything is committed. The Flock side's records are `mu_i` and the
  `w1Encode` of the committed bits (`agg_piop::mode_b::w1_encodings`), so the two bit
  copies agree by construction.
- The aggregate is the `agg_piop::wire::AggregateProof` byte string: header, the `c~_i` list,
  then the proof blob (Akita commitments, the mode-A segment items of K10, `Y_sigma`, PIOP
  rounds and claims, the Akita opening). There is no projection seed in the transcript
  (P2); the ordering is that of spec Section 10.
- The verifier takes `(pk_i, M_i)` and the aggregate and recomputes `tr_i`, `mu_i`,
  `c_i = SampleInBall(c~_i)`, `T_i` and the `ExpandA` data (`agg_piop::verifier`, P10). The
  optional key cache is `agg_piop::public_data::KeyCache` in memory (a hash map keyed by the
  encoded key) and, on disk, `agg_cli::cache` (entries addressed by `SHA3-256(pk)`, holding
  `pk`, `tr` and `A`; `t1` is re-decoded from `pk` on load). The spec's "cached" verifier
  number of Section 11 is the run with every key in the cache.
- Cost report: the PIOP verifier's 14 steps (`VerifyTimings`), per Flock part the context
  setup and verification time, and the keys expanded in the run.
- Empty segments (possible for `N` in `(2^14, 2^14 + 2^13]`) have no Flock part and require
  `Y_sigma = 0`; this path is unit-tested on the item level only, since batches above
  `2^13` records do not run on the development machine.

## P12. Prover kernel: compiled plans, coded rounds, dense tables from the third fold (spec Section 7)

Spec: the stages are ordinary batched sum-checks whose round polynomials are defined by
the instance tables of Sections 7.3 to 7.5; nothing is said about how the prover
evaluates them.

Code (`agg_piop::kernel`, `docs/piop.md` Section 6): the prover compiles the
gamma-weighted terms of a group into per-lane polynomials, bilinear products, weighted sums
and residual terms, evaluated in coefficient form. An `i8` column whose present cells fall
into at most 16 classes under the maps of its lanes is coded (one byte per pair of cells);
pieces on coded lanes are bucketed by cell patterns. In the first active round one pass
accumulates quartet masses with the round-1 equality weights and serves rounds 0 and 1
(Aerie's bivariate prefix) when the quartets outnumber the pattern space four times, else
pair masses; coded lanes are pair tables after the first fold and quartet tables (with
Taylor tables of their polynomials) or pair tables with the second challenge after the
second; every lane is materialized in one row-major `K` table after the third fold. From
the second active round on, the constant coefficient of the inner polynomial is recovered
from the running claim, which is the group's own exact sum. The parities `y^(1)` are `i8`
tables when every value fits. The opening takes its claims from the stage-3 terminal values
instead of evaluating the tables again; the round-1 commitment runs concurrently with
the Keccak side's commitments, and the four round-2 commitments concurrently with each
other (all transcript-free, absorbed in event order). With the lookup layout (L4) the two
round-1 commits and the four round-2 commits run concurrently on scoped OS threads
(`pcs::concurrent_commits`), not as rayon jobs: Akita builds its NTT slots lazily inside
`OnceLock::get_or_init` with rayon-parallel loops, and two commits joined on the pool can
deadlock when the initializing worker steals the other commit, which re-enters the same
`OnceLock` (observed once, in the default-build `agg-cli` tampering test). No rayon worker
blocks on those OS threads: with the round-1 commits inside a `rayon::join` with the Keccak
side's commitments, a worker waits for its commit threads while their Akita loops need a
worker, and with a one-thread pool (`taskset -c 2`, `RAYON_NUM_THREADS=1`) the prover hangs
in round 1. So `pcs::overlap` runs the round-1 commits on the calling thread and
the Keccak side on a scoped OS thread, and `pcs::concurrent_commits` commits one group after
another on the calling thread when that thread is a rayon worker or the pool has one thread
(one commit at a time cannot re-enter a `OnceLock`); `tests/single_thread.rs` proves and
verifies a batch with a one-thread pool, from a plain thread and from a pool worker.

Why it is sound and the proof unchanged: every round message is the same polynomial as
before, computed in a different order over `K` (bucketing and moments are distributivity of
the weights over equal summands; Taylor coefficients, the depressed quartic and the claim
recovery are exact identities in the field; unreduced sums are exact), so the messages, the
terminal values, the opening claims and therefore every proof byte are identical
(aggregates compared byte for byte against `7ed4a32` at 256, 512, 1,024 and 2,048 records
in mode B and at 256 and 1,024 records in mode A). The verifier is untouched. The kernel
tests compare every strategy (generic, coded per item, pair or quartet masses, quartet or
pair-fold tables, with or without Taylor tables and claim recovery) bit for bit against an
independent reference (the round polynomial from its definition over full equality tables),
on random tables with every `Term` kind, wide cells, `i32` lanes and domains of 0 to 4
variables. The claim recovery uses only the prover's own running claim (computed from the
group's tables in its first active round, or directly for a lifted group), never a claimed
sum, so a tampered witness under the test-only unchecked provers still yields the true
polynomials of its tables. Why: stages 1-3 were 1.82 s of the 2.73 s mode-B prover at
`N = 2,048` (`7ed4a32`, same session) and 0.65 s with this kernel.

## L1. Lookup layout L16 (spec Sections 4–7, 10–12)

The lookup layout is specified in [lookup.md](lookup.md). It uses 12 committed columns
per position. Every 16-bit value and carry is looked up in `T_16 = [0, 2^16)` by
LogUp-GKR with two independent challenges. `Z_hi` and `U_hi` use small leaves
(base 4 and base 6). A single layout avoids duplicating the manifest, stage tables,
claim vectors, transcript registry, wire format, verifier and catalogs.

## L2. Carries stored balanced like every looked-up column (docs/lookup.md Section 1)

every looked-up column, carries included, stores `K = f - 2^15` and is
looked up as `f = K + 2^15` (`manifest::LOOKUP_SHIFT`; the carry layout is one unsigned
base-`2^16` digit, `kappa = K + 2^15`); the padding slots of the carry stacks hold `-2^15`
(looked-up value 0). Why: one storage shift for all four stacks gives one leaf formula
`1 / ((X - 2^15) - K)` and one stage-3 stack claim, and makes the width-16 admission interval
`[-2^15, 2^15)` coincide with `T_16` for every looked-up column (as-is storage would admit
negative carries that only the lookup rejects). Soundness is unchanged: the lookup proves
`K + 2^15 in [0, 2^16)` for every cell, the range of `docs/lookup.md` Section 2.

## L3. One forced balanced base-`2^16` inner digit for the width-16 preset (docs/pcs-config.md Section 2)

Planner probe: at `(18, 4)` and `(20, 4)` Dense's search (`3..=16`)
selects two base-`2^9` digits, a forced `(16, 16)` one digit and a forced `(8, 8)` three,
with the same estimated opening proof (60,877 B and 61,707 B). Code: inner range `(16, 16)`,
the fewest inner digits (12 per position in all), hence the least commit work for the same
estimated proof bytes.

## L4. Six groups, two families, `G_m` as Akita's final group (spec Section 4.1)

`G_main16` (width 16), `G_main4` (width 4), `G_1`, `G_2` (parities and carries in one
width-16 group), `G_3` (width 16), `G_m` (width 4) in this wire order; `G_m` is the final
group (committed against the five precommitted profiles), so the grouped rows
(`mldsa-44-l16-batch-<N*>-mult-final`) live in the width-4 catalog (D2). The multiplicity
column is committed in round 2: it counts the carries, which exist only after the bridge
seed. Both main groups are opened at `r^(3)`, `G_m` at the lookup's table point `pi_m`.

## L5. Lookup events between the Keccak side and `tau`; the PIOP split (spec Section 10)

Code: the lookup is event 7, after the last commitment (round 2, event 4) and the Keccak
side (events 5, 6) and before `tau`: sites `agg-mldsa/v1/lookup/x` (two challenges, one
vector), `.../layer1`, `.../mu/1`, then per session `s = 2 .. D` `.../gamma/s`,
`.../round/s/k` and `.../r/s/k`, `.../terminals/s`, `.../leaves/s`, `.../mu/s`, and
`.../multiplicities` (the seven digit claims). Every count follows from `N` through the
manifest's lookup shape, so the registry stays fixed per batch. The PIOP part (events 7 to
12) is `prover::prove_piop` / `verifier::verify_piop`, which the full drivers and the
PIOP-level tests share. Why the lookup before `tau`: it depends only on the commitments and
`X`, and its claims are reduced in stage 3 like every other claim.

## L6. Stage-3 stack reductions as scaled instance parts (spec Section 7.5)

The stack reduction of stack `j` has `K` coefficients `w_{j,c} = eq(pi_top, c)`. Code: one
gamma index per stack and one instance part per column with the public scale
`stage::InstanceSpec::scale = w_{j,c}` (the batched coefficient is `gamma_j w_{j,c}`), each
a linear term on its column, in an equality group at the stack's low point
(`stage::EqPoint::Lookup(j)`). The claimed sum is the stacked claim minus the public
padding (`lookup::StackShape::column_claim`). The kernel absorbs the scale in the linear
lane polynomial; the verifier evaluates the same parts.

## L7. Multiplicity digits claimed after the lookup (docs/lookup.md Section 1)

The lookup leaves one claim `m(pi_m)` on the multiplicity column, which is committed as
seven balanced base-16 digits. Code: the prover sends `m_k(pi_m)`, `k < 7`, right after the
lookup (`claims::MultiplicityClaims`, 112 bytes); the verifier checks
`sum_k 16^k m_k(pi_m) = m(pi_m)` (`VerifyError::Lookup`) and the opening authenticates the
seven claims of `G_m` at `pi_m`. No range check of the digits: Lemma 5 of LogUp holds for any
`m` in `F_p` (`docs/lookup.md` Section 2).

## L8. Relation H-u (spec Sections 3, 4.3, 7.4, 10, 12; Appendix A)

H-u is the default relation. Its value forms and six-valued `u` digit are specified
in [h-exact.md](h-exact.md), Section 8. It proves the exact `u` bound without an
extra committed column and uses ML-DSA-44's SelfTargetMSIS instance.

## L9. Relation H-exact as a runtime relation: exact `z` by a bit-slack form, one six-valued `u` digit, Booleanity at `r^(2)` (spec Sections 3.1, 4.1, 4.3, 7.4, 7.5, 10, 12; Appendix A; `docs/h-exact.md` Section 9)

H-exact is a runtime relation
(`manifest::Relation::HExact`, relation byte 1, bound in the proof header and byte 3 of
event 0) next to the default H-u, so one binary proves and verifies both results:

* `z = S_zlo + 2^16 B_zhi + 130,915 a_z - 98,225` (`DigitValue::Zx`, columns `bz_lo`, `bz_hi`,
  `a_z`): `S_zlo` takes the slot of `Z_lo` in `G_main16` and in the main lookup stack, `B_zhi`
  the slot of `Z_hi`, `a_z` is appended to `G_main4` after `b_5` (9 polynomials; the bits keep
  positions 2..7); `u` keeps H-u's one six-valued digit (L8) instead of `a_u` plus a base-4
  digit, so H-exact costs one column (+1,024 elements per signature), not two.
* The Booleanity of `bz_hi` and `a_z` is proven by two zero-checks at `r^(2)` in stage 3
  (`Term::Bool` in the `eq(r^(2))` main group, claim 0) instead of stage 1 at `tau`: both
  columns are opened at `r^(3)` anyway, so no evaluation reduction is added. Sound because
  both are committed in round 1, before `r^(2)` (the argument of the links and the ring
  identity at `r^(2)`, `docs/soundness.md` 3.4).
* Counts follow the relation (`stage2::gamma_count`, `stage3::Gammas`, `Stage2Claims::len`,
  `Stage3Claims::len`, `transcript::stage_gammas`): batching vectors (26, 7, 45), terminal
  claims (18, 7, 32), stage degrees (4, 4, 3) as for H-u. The proof-of-work plan differs only
  at the stage-2 batching site (`g = 3`), so the plan encoding needs no relation field; the
  H-exact digests differ from H-u's anyway (`docs/soundness.md` 10.1).
* The catalogs are keyed by `(d_sig, relation)`: 13 scalar `G_main4` rows at `(13..25, 9)` and
  13 grouped rows `mldsa-44-l16x-batch-<N*>-mult-final` in the width-4 artifact; H-u's rows
  and the width-16 artifact are byte-identical (`docs/pcs-config.md`).
* The verifier takes the relation from its caller (`verifier::verify(..., relation, ...)`,
  `AggregateVerifier::relation`, `agg verify --relation`) and rejects a header that names
  another relation before event 0 (`VerifyError::Relation`), as `docs/h-exact.md` 1.3
  requires.

Why: the project reports two results (H-u and H-exact); a runtime relation
lets one build benchmark both back to back, and the layout uses one extra committed column.

## L10. Relation H-FIPS: relation byte 2, `u_h` in `G_2`, the sign as a bit, five zero-checks (spec Sections 3.1, 4.1, 7.4, 7.5, 10, 12; Appendix A; `docs/h-exact.md` Section 10)

Design proposal:
the exact FIPS 204 relation with the hint as an optional third relation, relation byte 2;
per coefficient `(L_lo, L_hi, sigma, delta, e)` with `L = L_lo + 29,696 L_hi`,
`r0 = sigma L + (1 + sigma)/2`, `u = r0 - 2 gamma_2 delta - e`; six zero-checks
(`sigma^2 = 1`, `delta (delta - sigma) = 0`, `e (e - 1)(e - delta) = 0`, `e (e + 1) w1' = 0`,
`e (e - 1)(w1' - 43) = 0`, `L_hi` Boolean) placed in stage 1 (batching vector 26 -> 33) or in
stage 3 at `r^(2)`; per signature `sum delta^2 + u_h = 80` with `u_h` looked up "on the
signature domain" in "a seventh Akita group". Code (`docs/h-exact.md` 10.6, 10.7):

* `u_h` is the thirteenth polynomial of `G_2` (stored `u_h - 2^15`, width 16, round 2) and
  takes the free slot 6 of lookup stack 2. `D_2`'s last `d_sig` variables are the signature
  index; below `d_sig = 13` the honest table holds `u_h(i)` at index `i 2^(d_2 - d_sig)` and
  0 elsewhere, and the weight identity sums the cells of a signature. Same committed cost as
  a seventh group; the six-group contract and the lookup shape (`L`, sites) are unchanged.
* The PIOP commits the bit `s = (1 + sigma)/2` instead of `sigma`, and uses
  `r0 = gamma_2 s + L + 1 - gamma_2` (`L = (r0 - 1) mod gamma_2`) instead of
  `r0 = sigma L + (1 + sigma)/2`: the same pairs `(sigma, r0)` under another labelling of `L`,
  but `u` is affine in the committed values, so the ring identity needs no product `s L`.
* `e (e + 1) w1' = 0` and `e (e - 1)(w1' - 43) = 0` become one degree-2 check
  `2 e w1' + 43 e^2 - 43 e = 0`, equivalent once `e in {-1, 0, 1}`.
* Placement: the degree-3 `e (e - 1)(e - delta)` in stage 2 at `r^(1)` (degree 4 with `eq`);
  `s (s - 1)`, `delta (delta - 2 s + 1)`, the merged wrap check, `L_hi (L_hi - 1)` and the
  weight identity in stage 3 at `r^(2)` (the weight identity at `(1/2, ..., r^(2)_sig)`, two
  parts on the main domain and `D_2`). Stage vectors (26, 7, 51), terminals (18, 8, 36).
* The proof-of-work plan equals H-exact's (`g` 5, 3, 6 at the stage gamma sites); the
  relation is bound by event 0's manifest bytes, so the plan encoding stays as it is.
* Catalog rows `mldsa-44-l16f-batch-<N*>-mult-final` (13 grouped, 13 width-4 scalar at
  `(13..25, 12)`, 3 width-16 scalar for `G_2` at `(13|14|15, 13)`).

Why: a seventh group would turn every fixed six-group array of the PCS driver, the verifier
and Akita's grouped rows into a per-relation shape for no saving in committed cells; the
affine `r0` keeps the 16-bit `l_lo` lane on the kernel's linear fast path; the merged wrap check and the stage-3 placement keep stage 3 at degree 3 and need two cheap
reductions instead of new lanes; the stage-1 vector and the `tau` site stay as they are. No
wire-version bump: H-u and H-exact proofs are byte-identical, and an H-FIPS proof differs in
the relation byte and in counts the verifier derives from the manifest.
