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
name. Aerie's catalog `aerie_fp64_signed_width4` holds Falcon layouts only, so our
four-group layouts need their own catalog, `agg_mldsa_fp64_signed_width4`
(`crates/agg-pcs-config/src/schedules/`). The family name enters the catalog digest
that event 0 binds; the protocol is otherwise unchanged. See `docs/pcs-config.md`.
Since wire version 1 (lookup layout L16, entry L1) this catalog holds the scalar rows of
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

Spec: "Sixteen columns" for the `relation = fips` layout (the count of
`improvements-tables.md`, which predates the committed quotient and the base-16 range digits).

Code (`agg-witness`, feature `fips`): with the committed quotient of Section 6 and every
range expressed in the balanced digits of Section 4.3, the layout has 29 main-domain columns
(b_z 5 digits + a_z, r1 2 digits, b_v 4 digits + a_v, s, g, e, v 7 digits, b 6 bits) and 3
signature-domain columns (b_h 2 base-8 digits + a_h). Column order and constants are in
`docs/witness.md`. Why: the spec's count does not include the quotient digits or the
digits of the 16- and 17-bit slack parts.

## W2. Padded rows of the feature-flagged layouts (spec Section 4.2, Appendix A)

Spec: "for i >= N every value is zero", with the digits of zero at -b_k/2.

Code: padded rows hold the canonical encoding of the all-zero witness (z = 0, w' = 0, h = 0).
For H this is exactly the spec (all values zero, digits -b_k/2, top signed digit 0). For
H-exact, z = 0 and u = 0 are encoded through the slack forms (a_z = 0, b_z = 130,993;
a_u = 0, b_u = 190,465). For S1+S2, Decompose(0) = (0, 0) gives s = -1 (r0 <= 0) and the
hint slack is u_h = omega = 80 (a_h = 1, b_h = 63). Why: s in {-1, 1} and
sum g^2 + u_h = omega then hold on every row, so no live selector is needed.

## W3. Orders and canonical choices left open by the spec (Sections 1, 4.1, Appendix A)

Fixed in `agg-witness` and documented in `docs/witness.md`: level-3 in-block index
zeta_3 = t + 6 rho_2 (least significant first); G_2 column order y2_0..y2_5 then
kappa2_{t,k} t-major; G_1 kappa1_{t,k} t-major; H-exact order (b_z digits, a_z, b_u
digits, a_u, v, b); S1+S2 order as in W1; r1 = 4 A + B + 24 with A base 8 first;
slack choice a = [x + offset >= 2^j]. The S1+S2 identities have no stage in the spec; the
checker reports them under "layout".

## K1. One gate instance per signature sponge (spec Section 8)

Spec: "One Keccak gate type (one A_0 block, A = I (x) A_0); per record a 7-slot union
circuit with copy constraints between consecutive permutations", sub-batches of "2^16
slots", committed bits per record 6,144 + 7 * 38,400 = 274,944.

Code (`agg_keccak::sponge`): one gate instance (one A_0 block, `k_log = 19`) holds a
record's whole sponge, the 7 chi-only permutations chained by virtual XOR expressions;
the Flock union has one row per record (capacity 2^13 at a full segment, 57,344
permutations). Committed per record: 1 constant + 512 `mu` + 6,144 w1Encode + 256 `c~` +
268,800 chi bits = 275,713 (useful 275,968 of 2^19 since K11, 275,776 before; the dense-stack
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

## K6. Bridge words wired only in the bridged segment (legacy path; see K11)

Spec Section 8: the bridge slot copies "every live record's 48 w1Encode words".

Code: the plain sponge sub-proof (`prover::SpongeProver`, `GateBatch`) keeps the IO schema
`mu`, `c~` and its digests. The bridged segment (`segment::BridgedSegment`) registers the
sponge type with the 48 w1Encode words (aligned word columns 4..51) added as `In` schema
words after `mu`, schema `[mu (4), w1Encode (48), c~ (2)]` (digest-visible through Flock's
registry digest), and wires word `4 + 8t + w` of record `i` to the bridge slot's word
`(Lambda t + 1024 i) / 128 + w`: `48 n` wire classes of two `In` cells each, no bit
permutation (docs/bridge.md Section 2).

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
- The proof bytes (format version 2) no longer contain the commitment. `GateBatch::verify`
  and `SpongeProver::verify_batch` take the commitment absorbed at event 2, check its
  length (`Malformed`, not a panic) and pass it, with the verifier's own `PcsParams`, to
  Flock's verifier. The review asked for a commitment accessor on `KeccakProof` and an
  equality check in `verify`. With the commitment carried only at event 2, there is no
  second copy to compare: the equality holds by construction, and a different absorbed
  commitment fails Flock's opening (test
  `commitments_are_absorbed_before_the_seeds_and_bound`). On the prover side the accessor
  is `CommittedBatch::commitment` (`SpongeCommitted::commitment`).

Cost: one extra Ligerito commit per sub-proof (local calibration at 1,170 records: 0.046 s
of 0.94 s with CLMUL, 42 s of 121 s in the default build without CLMUL; `docs/keccak.md`
Section 6). The slot buffers of all `S` sub-batches stay in memory from event 2 to event 5
(estimate: about 2 GiB per full segment, `z`, `a`, `b` and the stripe at 512 MiB each, so
8 GiB at `N = 2^15`; a driver may keep only `z` and rebuild the rest). Removing the second
commit needs a Flock entry that takes the commitment and its `ProverData`.

Superseded for the tied part of the mode-A driver by K14: the commitment is computed once and
its prover data kept from event 2 to event 6; the paragraphs above remain true of the
standalone sponge path (`GateBatch`, `SpongeProver`).

The bridge slot (`bridge_slot.rs`) and the segment driver (`segment.rs`) follow the same
split: `BridgedSegment::commit` commits the sponge rows and the bridge copies (neither
depends on `sd`) before event 2; `prove` takes `sd`, its matrices and `fs_sigma`. The
bridge type's relation depends on `sd` and `Y_sigma`, so its digest (`BridgeSlot::digest`:
`sd`, the shape, the layout, `Y_sigma`) is absorbed into the Flock challenger after
`fs_sigma` and the sponge gate digest (the registry digest covers only the matrix stubs,
K2). `Y_sigma` is a coefficient of the relation, not a wired public word (K8). Because
`fs_sigma` is drawn after `sd`, the Flock transcript already depends on `sd` through its
seed; the digest is what binds the relation itself.

## K8. `Y_sigma` as a coefficient of the bridge rows, not a committed word (docs/bridge.md Section 5)

docs/bridge.md Section 5: columns `6 Lambda + 128 + rho_3` hold `Y_sigma[rho_3]` as committed
`Out` schema words wired to public words, with rows `(sum_c M[rho_3, c] z_c) * 1 = z_Y`; and
the slot's "witness is copies of the bits, so it is committed before sd".

Code (`agg_keccak::bridge_slot`): the two requirements contradict each other, since
`Y_sigma` is a function of `sd` and a committed `Y` column would make the event-2
commitment depend on `sd`. The columns `y_o = lanes Lambda + 128 + o` are committed zeros,
and row `y_o` is `(sum_c M[o, c] z_c + Y_o z_one) * z_one = z_{y_o}`: `Y_o` enters `A_0` as
the coefficient of the constant column (pinned to 1 by Flock). On an honest block these
rows have `a = 0`, `b = 1`, so the bridge witness (the bit copies, the constant, zeros) and
even its `a`, `b` buffers are independent of `sd` and `Y`. The type has no `Out` words; the
Flock public words are `[mu_i, c~_i]` only. Binding: the verifier rebuilds the relation from
its own `sd`, matrices and `Y_sigma` (another `Y` changes `xi_A` at the constant column by
`sum_o (Y_o + Y'_o) eq[y_o]`, so the lincheck fails), and `BridgeSlot::digest` covers `Y`.
The walker of docs/bridge.md Section 5 is unchanged except `xi_A[one] += sum_o Y_o eq[y_o]`.
The Y rows lie inside `useful_bits` (committed zero columns), so Flock's dead-region
gating never skips them. Tests (`tests/bridge.rs`): a flipped `Y` bit, another seed, a
flipped committed bit with the stale `Y`, a bit flipped in the slot copy only, a bit flipped
in both copies with the stale `c~` are rejected, at the toy and the production shape.

Test-plan differences (docs/bridge.md Section 9 item 6): the comparison with Flock's
`SparseMatrixCircuit` on explicitly lowered rows runs on the toy slot and on a production
slot of one record (`d_seg = 0`: 4,096 dense composite rows over 6,144 columns), not at
`d_seg = 3` (about `10^8` entries, 800 MB as Flock's `Vec<Vec<usize>>`); the bridged proofs
run at `d_seg = 3` (5 live records, 3 padded) and `d_seg = 6` (64 records), not 7; "a stale
parity row" has no counterpart because no parity is committed (the nearest case, a stale
`Y` for a flipped committed bit, is tested).

## K9. Full-segment bridged union under Flock's uniform row capacity (risk, open)

Not in the spec. A Flock registry gives every type the same `2^nu` rows. At a full segment
(`n = 2^13`, `nu = 13`) the bridge type's `k_log = 26` makes its slot `2^39` bits, so the
union has `M = 40` variables: Flock's prover allocates the padded buffers `z`, `a`, `b` with
`2^33` words each (128 GiB each, zero-filled on the prebuilt-slot path this code uses, and
the bridge slot's own buffers are `2^32` words; only the dense commitment is
count-proportional), and the
wiring argument's cell space is `2^nu` cells per schema word, `393,216 + 54` words, so
`mu = 13 + 19 = 32` (`2^32` cells, and the verifier's `s_sigma` discharge is `O(2^mu)`).
The design of docs/bridge.md Section 5 (B.13) therefore does not fit 64 to 128 GB at a full
segment with Flock `83f963e` (estimate from the formulas above; not run). Local sizes are
unaffected: at 64 records `M = 27`, `mu = 19`. `UnionContext::commit` in `segment.rs` also
materializes the padded `z` (fine at test sizes; a production commit must gather the
dense stack directly).

Status: the remedy below was implemented as K10 (Flock parts of `2^{d_part}` records, then
default `d_part = 9`); K11 removes the cause (the bridge slot has its own instance) and the
default part is the whole segment again.

Remedy (proposal at the time): split a segment into Flock sub-batches of `2^{d_f}`
records, each with its own bridge slot proving the partial vector
`Y_{sigma,s} = M^{(sigma)} b'_s` over the sub-batch's columns, and let the verifier check
`XOR_s Y_{sigma,s} = Y_sigma` (the map is linear and the matrices are the same seeded ones,
so the bound of docs/bridge.md Section 7 is unchanged; the walker only needs an offset of
the level-1 block index). At `d_f = 10` the bridge slot is `2^33` bits (1 GiB per buffer)
and `mu` about 26, at the price of 8 Flock proofs per segment instead of one (estimate:
proof bytes times 8 for the Keccak side). The alternative is a Flock change (per-type row
capacity). A decision is needed before the cloud run.

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

## P3. Grinding (resolved in wire version 2, spec Sections 10 and 12.1)

Spec: "the scheduled grinding [is] Aerie's; the work plan per admitted shape is an
obligation of `agg-pcs-config`."

Code since wire version 2 (`docs/soundness.md`, normative): every PIOP field-challenge site
has a difficulty from Aerie's assigned bounds (`agg_piop::transcript::Registry`,
`agg_bridge::pow::grind_bits_k`); the transcript absorbs `LE32(nonce)`, checks the low `g`
bits of a 32-byte predicate with Akita's `grinding_predicate_accepts` and squeezes the
challenge after it; the nonces travel in one packed stream (`AggregateProof::work`, 227 to
556 bytes); the plan (tie sites, then PIOP sites) is bound in event 0 by its SHA3-256 digest
(`agg_piop::pow::PowPlan`). The tie challenges of a tied part grind 16 to 18 bits on the tie
challenger (tied-part proof version 4). Akita's opening already ran its own plan
(`akita_config::transcript_instance_descriptor`, pinned by
`agg-pcs-config/tests/akita_grinding.rs`). Remaining differences from Aerie, all narrower
than before: (1) the nonce is a fixed-width field of a packed stream, not a LEB128 atom in a
Spongefish argument string (our PIOP has no argument string; the stream's width per site is
public, so the encoding is canonical); (2) the plan is a function of `N*`, the Flock profile
and the tied-part geometry rather than a per-shape table in `agg-pcs-config`; (3) the seeds
`sd`, `fs_sigma` and the PCS export carry no work (their bad sets are below `2^-128`,
`docs/soundness.md` Section 3.5).

## P4. Bridge matrices and the Keccak side are hooks (spec Sections 8, 9.1, 10, 11)

Spec: the verifier derives `M_1, M_2, M_3` from `sd` (Section 9.1) and verifies the Flock
sub-proofs against `mu_i, c~_i, Y_sigma` and the circuit digest (Section 11 step 3).

Code: `prover::BridgeDerivation::matrices(sd)` and `prover::KeccakSide` /
`verifier::KeccakVerifier` are trait hooks, so the wire format and the event order do not
change when `agg-keccak` (Flock sub-proofs, milestone M3) is plugged in; the transcript
events 2, 3, 5 and 6 are always present with `S` atoms.

- Matrices: `prover::SeededMatrices` is the derivation of Section 9.1
  (`agg_bridge::seed::derive_matrices`; `BridgeDerivation::seed_derived` is true for it
  only). Choices the spec leaves open: the `LE64(t)` of the hash input is the level number
  `t = 1, 2, 3`; a column's rows are stored in the order drawn; the XOF is read in 64-byte
  chunks (the word stream is the same). A known-answer test (column 0 of each level for
  `sd = 0^32`) was cross-checked against an independent pure-Python cSHAKE256. Cost: the
  2 x 2^19 + 49,152 columns take one Keccak-f each after a shared prefix state, and the
  matrices take 62 MB in each of prover and verifier. Measured on this machine (release,
  2026-10-24, one test process, not a benchmark): median 150 ms over 7 derivations on 7
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
  `LaBinius`) requires `seed_derived()`, in every build, because with matrices known
  before `G_main` and the Flock commitments are fixed the bound of Section 9.4 does not
  apply; profile 0 (no Keccak argument, `c~_i` not tied to the committed bits) is refused
  unless the test hooks are compiled in. `prover::prove` refuses mode A over unseeded
  matrices as well (completeness only).

What remains open is planned, not a defect of the PIOP: there is no Flock
`KeccakVerifier` in a library crate yet (milestone M3, `agg-keccak` bridge slot and segment
driver, K6, K7); the joint test `agg-piop/tests/joint_bridge.rs` wires one up for a single
segment (P11).
Until it exists the only accepting production configuration is mode B
(`mode_b::verify_mode_b` with `SeededMatrices`), which is sound through its own checks
(P9). Mode A becomes available by passing the Flock verifier (profile `Fast` or
`LaBinius`) with `SeededMatrices`; the policy refuses any other bridge.

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
`leaf4[Z_hi]`, `leaf8[U_hi]`; stage 3 lists the ring, the two links, the 37 evaluation
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

## P11. The Keccak hooks carry the bridge seed (spec Sections 8, 10, 11; docs/bridge.md Section 5)

Spec: the bridge slot's circuit digest MUST cover `sd`, and the Flock rows are generated
from `M_1, M_2, M_3` of `sd` (Section 8 driver: "receive `sd` and `fs_sigma`, derive the
bridge rows, bind the statement").

Mismatch found by the joint bridge test: `prover::KeccakSide::prove` received only the
Flock seeds `fs_sigma` and the final vectors, and `verifier::SegmentContext` only `fs_sigma`
and the matrices, so a Flock side plugged into the PIOP could not bind the transcript's
`sd` into `BridgeSlot::digest` (`agg-keccak` takes `sd` explicitly in
`BridgedSegment::prove`/`verify`) and would have had to re-derive the matrices.

Code: `KeccakSide::prove(bridge_seed, matrices, seeds, final_vectors)` and
`SegmentContext::bridge_seed` pass the transcript's `sd` (event 3) and its matrices to the
Keccak side. Wire format, transcript and the mode-B and profile-0 sides are unchanged
(they ignore both). Test: `agg-piop/tests/joint_bridge.rs` (release, 64 and 256 real
ML-DSA-44 signatures, one segment) runs the F_p side and the Flock side (`BridgedSegment`,
profile `Fast`) in one transcript through test adapters of these hooks; both final vectors
equal a direct sparse GF(2) computation from the derived `w_1'`, the bit orders of `w_1'`,
`w1Encode` and the committed `b_t` agree (no ordering fix was needed), the aggregate verifies
with the Flock sub-proof checked inside the PIOP verifier, and one flipped `w1Encode` bit
fails exactly the side that holds it (F_p side: stage-1 terminal with the Flock sub-proof
accepted; Flock side: the sub-proof rejected with the F_p side accepting, the rejection due
to the bridge rows alone). The adapters stay in the test until the mode-A driver exists.

## K10. Flock parts within a segment (spec Sections 1, 8, 10; resolves K9)

Spec: "A segment is both one Keccak sub-batch ... and one parity-bridge chain"; one Flock
commitment (event 2) and one Flock proof (event 6) per segment, the bridge slot proving
`Y_sigma = M_3 M_2 M_1 b'` over the whole segment.

Code (`agg_cli::mode_a`, `agg_keccak::segment::BridgedSegment::new_part`,
`agg_bridge::gf2_walker::SegmentLayout::part`): the parity-bridge chain stays one per segment
(the F_p side, `Y_sigma` and the matrices are unchanged), but the segment's Flock argument is
split into parts of `2^{d_part}` record positions, `d_part = min(part_log2, d_seg)` with
`part_log2` chosen by the prover (default 9, `docs/cli.md` Section 4) and `3 <= d_part`. Part `s` holds the
segment's records `[2^{d_part} s, 2^{d_part} (s + 1))` that are live; parts without live
records do not exist (an all-padded segment has none, and the verifier then requires
`Y_sigma = 0`). Each part is one Flock union proof of its sponges and of a bridge slot whose
layout is the segment's restricted to the part (`SegmentLayout::part`: local position `p` is
segment position `2^{10 + d_part} s + p`, so its level-1 block, in-block index and level-2
input are the segment's). The slot proves the part's share
`Y_{sigma, s} = M^{(sigma)} b'_s`; by linearity over GF(2) the shares XOR to
`M^{(sigma)} b' = Y_sigma`, so the bound of spec 9.4 is unchanged (the difference `b XOR b'`
and the matrices are those of the whole segment).

Wire (`agg_cli::wire`): the event-2 item of a segment is `version || d_part || LE32(parts)`
followed by the framed part commitments; the event-6 proof item is the packed shares of all
parts but the last (512 bytes each) followed by the framed part proofs. The last share is
`Y_sigma` XOR the others, so the XOR identity holds by construction. Both items are
absorbed where the spec absorbs the single Flock commitment and proof, so `d_part`, the
commitments, the shares and the proofs are bound in the PIOP transcript. Part `s` seeds its
challenger with `SHA3-256(len || "agg-mldsa/v1/flock-part-seed" || fs_sigma || LE64(s))`, and
`BridgeSlot::digest` now covers the part offset (and the share), so a part proof does not
verify at another index. The verifier rejects `d_part` outside `[3, d_seg]` and a part count
other than `ceil(live / 2^{d_part})`. (Since the proof-of-work schedule the admitted range is
`[max(3, d_seg - 1), d_seg]`, at most two parts per segment: `docs/soundness.md` 3.5, K20.) Its Flock work grows with the number of parts (and so
with the proof size); a floor on `d_part` per `N` is a possible later policy.

Why: K9. A Flock registry gives every type the same `2^nu` rows, so the bridged union of
`2^d` records has about `2^{13 + 2d}` bits; at a full segment that is `2^40` bits (128 GiB
per buffer). Parts of `2^{d_part}` records cost `2^{13 - d_part}` Flock proofs per full
segment. Measured locally (`docs/cli.md` Section 4, default build): a part of 128, 256 and
512 records takes 3.2, 5.5 and 11.5 GiB prover peak RSS and 408, 522 and 676 KB of proof,
at about the same time per record. At `N = 2^15` that is (estimate from these sizes) 43 MB
of Flock proofs at `d_part = 9` and 67 MB at `d_part = 8`; `d_part = 10` (about 23 GiB per
part, extrapolated) is for the cloud machine to calibrate. With `d_part = d_seg`
(every local batch at the default) a segment has exactly one part, as in the spec, except
for the part seed and the item framing.

Memory: the driver commits the parts one after the other at event 2 and keeps only their
commitments; at event 6 it rebuilds each part's witness from the records
(`BridgedSegment::reattach`, no second commitment: Flock's prover commits internally, K7,
and the cap is checked against the absorbed one), proves, and drops it. This removes the
K7 cost of keeping all slot buffers from event 2 to event 6, for one witness rebuild per
part (cheap next to the proof).

## K11. Tied parts: two Flock instances per part, tied by evaluation claims (spec Section 8; docs/bridge.md Sections 5, 5a)

Spec and docs/bridge.md Section 5: one Flock union proof per segment holding the sponge
slots and the bridge slot, the bridge slot's bit copies "tied to the records' `Input` words
by Flock's word-level copy constraints".

Code (`agg_keccak::tied`, `agg_keccak::flock_union`; the mode-A driver `agg_cli::mode_a`): a
part is two single-type Flock proofs, the sponge instance (its records' gates, row capacity
`capacity_log(n, 19)`, K5) and the bridge instance (the bridge slot of K8 alone, one live row
of `2^max(3, 22 - k_b)`), and the copies are tied by evaluation claims on the two committed
polynomials at a point derived from the part's Flock seed (five packed-direct claims carried
by Flock's merged opening: the public region of the sponge block against the verifier's
table of `(mu_i, c~_i, 1, 0)`, and the two lane groups of the w1Encode words against the same
groups of the bridge block). Neither proof has a circuit, a wiring argument or public words;
the statement words are bound by a digest in the challenger. docs/bridge.md Section 5a has
the layout, the points, the checks and the soundness argument (`(d + 5) / 2^128` per claim,
the point fixed after both commitments).

To make the lane groups aligned word subcubes, the sponge gate's block puts the w1Encode
field first (words 0..47, lane `t` at `8 t`), then mu (48..51), `c~` (52..53), the constant
(word 54) and the chi words from word 56 (`GateLayout` aligns the chi words to an 8-word
boundary): 275,968 useful bits per record instead of 275,776 (2,156 committed words instead
of 2,155), and 42,496 instead of 41,792 for the single-permutation gate. `W1_INPUT_BASE` is 0
and `MU_INPUT_BASE` 6,144 in the packed gate inputs.

Why: the profile of docs/keccak.md Section 8. Flock's copy constraints cost one cell-slot,
one gather claim and 16 proof bytes per schema word of a type, and a one-row bridge slot
has one schema word per copied word: 768 proof bytes and about 1.4 ms of opening time per
signature at 512 records, 25 MB of gather values at `N = 2^15`; and the uniform row capacity
gave the bridge slot `2^(k_b + nu)` bits of address space, `2^(2 d + 14)` per part, which is
what capped parts at `2^9` records (K9, K10). Neither can be reduced inside the wiring
argument (a cross-record linear relation needs its inputs in one block). With the tie, the
bridge instance's cost no longer depends on the sponge rows and a part may be the whole
segment (`DEFAULT_PART_LOG2 = 13`; every local batch is one part).

Flock entry: `flock_union::UnionGeometry::{prove, verify}` compose Flock's public building
blocks (`bind_statement`, `zerocheck::prove_packed_padded_capture_s_hat_v_c_with_grinding`,
`lincheck::prove_union_capture_z_vec_with_grinding`, `pcs::open_batch_merged` and their
verifiers) in the order of `flock_prover::prover::prove_fast_ligerito_union`, for one
boolean type, plus the caller's claims; Flock 83f963e is unmodified. The one `unsafe` of
`agg-keccak` (a byte view of the `F128` buffers for the zerocheck, as Flock's own prover
does) lives there. The two-type union of `segment::BridgedSegment` is kept as the reference
and for the toy-shape slot tests; the mode-A driver and its wire items (version 2) use the
tied part.

Test-plan differences (docs/bridge.md Section 9 item 6): the tampering cases run on the tied
part (`tests/tied.rs`) with the tie's error names: a bit flipped in the bridge copy only
fails the lane-group tie, a bit flipped in both copies with the stale `c~` fails the public
claim, a sponge input bit flipped with the stale chi witness fails Flock's sponge rows; a
flipped share, another seed, another `fs`, a flipped proof byte and swapped caps are
rejected; parts' shares XOR to the segment's `Y` as in K10.

Since tied-part proof version 2 the bridge instance is the two-type bridge union of K13
(level-1 rows and one levels-2-3 row, committed at event 6), the event-2 item of a part is
the sponge cap alone, and the sponge commitment is computed once (K14); the sponge instance,
its claims, the public check, the lane-group tie and its error names are unchanged.

## K12. Vendored `flock-core` with a prover-only opening patch (spec Section 8; AGENTS.md dependency rule; see K19 for the verifier-visible list-aware schedule)

AGENTS.md: dependencies are pinned git revisions (Flock `83f963e`, recorded in `README.md`).

Code: `vendor/flock-core` is a verbatim copy of `crates/flock-core` at that revision with
one change in `pcs/jagged.rs::build_merged_weight_and_prime` (`vendor/flock-core/PATCH.md`),
wired in by `[patch."https://github.com/succinctlabs/flock"]` in the root `Cargo.toml`; the
other Flock crates stay the pinned git dependencies. The change: the merged opening bakes a
16 x 256-entry fold table (64 KB, 4,096 fold evaluations) per used word column for every
ring-switched claim, whatever the column's height; the patch evaluates the claim per slot
(`phi(eq_col[c] r)`, one multiply and one lookup) when the stack averages fewer than 64 live
rows per used column. Flock's own shapes (hundreds of rows per column) are unaffected.

Why: the bridge instance of K11 is one row of `48 x 2^d + 33` word columns (49,185 at
`d = 10`, 393,249 at a full segment). Measured (`ref/notes/bench-local.md`): the bake took
3.4 s and 6.4 GiB of the bridge instance's 4.2 s and 7.8 GiB at 1,024 records, and would
take 51 GiB at a full segment; Flock's block-first transport (about `128^2` field
operations per column) is no alternative for a wide one-row stack. The patch is exact and
prover-only: `x -> phi(eq_col[c] x)` is F_2-linear, so the baked byte decomposition and the
direct evaluation give the same weight vector bit for bit, the transcript and the proof
bytes are unchanged, and proofs made with the vendored crate verify with upstream Flock
(and the other way round). The vendored copy is a project decision to confirm (upstreaming
the patch, or a fork, would remove it); until then `vendor/flock-core/PATCH.md` is the
record of the revision and the diff.

Second prover-only change (`PATCH.md` Section 3, 2026-10-25): Flock's x86 kernels (additive
NTT butterflies with the fused-4 schedule, zerocheck rounds 1 and 2, the inverse-NTT table
apply) are chosen at run time (`is_x86_feature_detected!`: AVX-512F, VPCLMULQDQ, GFNI,
AVX512-VBMI) instead of by `cfg(target_feature)`, so the CLMUL build uses them on a CPU that
has them while the default build stays portable and runs everywhere. Why: upstream only
enables them under `target-cpu=native`, which our builds cannot use (AGENTS.md; a global
`x86-64-v4` breaks Akita), and the portable kernels were the top of the Flock profile
(`ref/notes/plan-128bit-speed.md` 2.3). Exact: same field values on every path, so the proof
bytes are unchanged (byte-identical aggregates; per-path tests listed in `PATCH.md`).
`FLOCK_SIMD=portable|gfni` caps the level for A/B runs. The merged opening's block
statistics have an AVX-512 path of the same kind (`PATCH.md` Section 4).

## K13. Two-type bridge union as the bridge instance of a tied part (spec Section 8; docs/bridge.md Sections 5, 5b)

Spec and docs/bridge.md Section 5: one bridge slot per segment, one block whose rows prove
`Y_sigma = M_3 M_2 M_1 b'`, intermediate parities not materialized; K11: that slot as its own
one-row Flock instance of the tied part, committed at event 2.

Code (`agg_keccak::bridge_union`, `tied`, tied-part proof version 2; mode-A item version 3):
the bridge instance is a Flock union of two boolean types. Level 1: one row per (lane,
level-1 block) of the part (`6 x 2^max(0, d - 9)` live rows) whose `A_0` is `M_1` itself,
committing the block's input bits and its level-1 parities `y1 = M_1 b'(t, beta)`. Levels 2-3:
one row whose inputs are the parities of all level-1 rows and whose rows are `(sum_c (M_3
M_2)[o, c] z_c + Y_o one) * one = z_{y_o}` (`Y` a coefficient, K8). Ties, as packed-direct
claims in the union's merged opening: the level-1 inputs against the sponge's lane groups
(tie 1, the error names of K11), the parities against their copies (tie 2), and the output
columns `z_y` against zero. docs/bridge.md Section 5b has the layouts, points, checks and the
soundness argument.

Two changes against the design note that proposed this (`docs/keccak.md` Section 10 and its
handoff): (1) the parities depend on `sd`, so the union cannot be committed at event 2; it is
committed at event 6, its cap travels in the part's proof, the event-2 item of a part is the
sponge cap alone, and the tie challenges are derived from `fs` and both caps (the union's
content is forced by the sponge's `b'`, fixed at event 2, so the bridge bound (9.3) is
unchanged); (2) a row with `C = I` only defines its output column, so a union committed after
`sd` needs the claim `z_y = 0` (the output region moved to the aligned base `6 G + 4096`); the
one-row slot of K8 did without it only because its `z_y` was fixed before `sd` (docs/bridge.md
Section 5b gives the argument for both).

Why: the profile of `docs/keccak.md` Section 10. Every cost of the one-row slot was linear in
its `2^(13 + d)` columns: `7.2 x 10^8` walker entries per fold and 1 GiB vectors at a full
segment, and a verifier discharge of 0.26 / 0.50 / 1.43 s at `d = 10 / 11 / 13`. The union
reads `1.9 x 10^7` entries over `2^20`-entry tables at a full segment (`M = 28`, 97 live rows,
411,712 committed words). Measured (`docs/keccak.md` Section 11, CLMUL build, 7 threads,
medians of 5): at 2,048 records the mode-A prover takes 4.80 s instead of 6.59 s (Flock part
1.90 s instead of 3.70 s, the bridge instance about 0.23 s instead of 1.73 s), the verifier
0.30 s instead of 0.75 s uncached (Flock part 63 ms instead of 505 ms), the prover's peak RSS
1.94 GiB instead of 3.10 GiB; the aggregate is 7,781 bytes (0.9%) larger (the union's cap
moved from event 2 into the proof, and a slightly larger opening). A part with the
full-segment bridge geometry (`d = 13`, 8 live records) proves in 0.45 s instead of 6.77 s
and verifies in 83 ms instead of 1.87 s, with 0.8 GiB instead of 6.1 GiB peak.

## K14. The Flock commitment computed once (updates K7)

K7: Flock 83f963e's prover has no entry that takes a precomputed commitment, so the event-2
commitment was computed twice, before event 2 and again inside Flock's prover.

Code: `flock_union::UnionGeometry::commit_data` returns Flock's `Commitment` with its
`ProverData` (the Reed-Solomon codeword and the Merkle tree); `UnionGeometry::prove(...,
pre: Some(data))` opens that commitment without committing again (the Ligerito commit is
deterministic, so the data is what the prover would recompute; tests show byte-identical
proofs with and without it). `TiedCommitted` keeps the sponge instance's data;
`agg_cli::mode_a` keeps it per part from event 2 to event 6 (default;
`Aggregator::keep_commit_data(false)` or the environment variable `AGG_FLOCK_RECOMMIT`
restores the K7 behaviour), while the witness buffers are still dropped after event 2 and
rebuilt at event 6. The bridge union is committed once, at event 6 (K13). The standalone
sponge path (`prover::GateBatch`) still goes through Flock's own prover and commits twice.

Cost (`docs/keccak.md` Section 11): the kept data is the sponge instance's codeword and Merkle
tree, 19 / 52 / 84 / 152 MiB at 256 / 512 / 1,024 / 2,048 records with `Fast` (twice that
with `Slim`; extrapolated about 0.6 GiB per full segment, 2.4 GiB at `N = 2^15`, not
measured); it is the data the sponge proof would rebuild at event 6, so the prover's peak is
unchanged (1,942 MiB kept, 1,959 MiB recommitted at 2,048 records). Saved: one Ligerito commit
per part, 217 ms of the Flock part at 2,048 records (1,901 against 2,118 ms) and 58 ms at
1,024.

## K15. Ligerito `Slim` in mode A; the profile in the segment item (spec Sections 8, 10)

Spec Section 10: event 0 binds the Flock profile; Section 8: Ligerito `Fast` as the
engineering baseline. K11 to K14 ran mode A under `Fast` only
(`agg_piop::binding::FlockProfile` had `Fast = 1`).

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
(`docs/keccak.md` Section 11) for about twice the level-0 encoding and twice the kept commit
data (K14). Measurements and the default: `docs/keccak.md` Section 12. By default the verifier
accepts either `Fast` or `Slim` because both meet the same documented level in upstream Flock;
`Slim8` (K18) is an opt-in mode-A profile; `Secure` (120 bits) is not a mode-A profile.

## K16. Spans: one tied part over all segments; the final vectors as the output claim (spec Sections 8, 9, 10; docs/bridge.md Section 5c)

Spec and K10, K11, K13: one Flock argument per segment (at least one tied part per segment),
the bridge union's levels-2-3 row proving the part's share with `Y` a coefficient of its
composite rows (K8) and its output columns pinned to zero by a claim.

Code (`agg_keccak::bridge_union::BridgeUnion::new_span`, `agg_keccak::tied::TiedPart::new_span`,
tied-part proof version 3; `agg_cli::mode_a`, `AggregatorConfig::span`, CLI `--flock-span
on|off`, default on): a batch of `S >= 2` segments whose parts are whole segments
(`d_part = d_seg = 13`) is proven by ONE tied part over all segments: one sponge instance of
all `N` records (record `i` of segment `sigma` is row `2^13 sigma + i`) and one bridge union
covering `2^ceil(log2 S)` segments (level-1 row `sigma + 2^span_log (beta + blocks t)`,
levels-2-3 row `sigma`; segments without records are zero rows). Every tie point gains the
segment coordinates `rho_sigma`. In every tied part (also `S = 1`) the levels-2-3 rows are
proven with a zero coefficient, so their output columns hold `M_3 M_2 y1`, and the output
claim's value is the multilinear extension of the public table of the final vectors (the
share for a part of one segment, the `Y_sigma` for a span), computed by the verifier; the
table's digest is absorbed before the tie challenges. Wire: segment 0's event-2 item has
span byte `S` and the span's cap, its event-6 item the span's proof (no shares); the items
of segments `1..S` have span byte 0, no commitment and an empty proof item; the verifier
gathers the segments' statements, `Y_sigma` and seeds and checks the span at the last
segment, seeded by `span_seed(fs_0, .., fs_{S-1})`. docs/bridge.md Section 5c has the
layout, points, checks and the soundness argument.

Why: every tied part pays two Ligerito openings (about 300 KB each with `Fast`, 200 KB with
`Slim`) whatever its size, so at `N = 2^15` four parts pay eight; a span pays two, of
polynomials four times larger, whose openings grow by a few levels only (`docs/keccak.md`
Section 12: Flock bytes per signature at `N = 2^15` from 94.6 measured with `Fast` parts to
about 16 with a `Slim` span, an estimate). The output claim replaces the `Y` coefficient
because the rows of one levels-2-3 type share one `A_0`, so different `Y_sigma` per row
cannot be coefficients. Prover and verifier work are the sums of the parts' (the instances
are linear in their rows); the prover's peak memory is that of one instance of all `N`
records (to be measured on the cloud machine). Local tests cover spans at the tied-part level
(`d = 3, 6`, two and three segments, an empty segment); mode A has more than one segment only
above `2^13` records, so the span path of `agg_cli::mode_a` is covered locally by the item
logic without proofs (`span_items_open_gather_and_close_at_the_last_segment`) and awaits the
cloud run.

## K17. The Ligerito level-0 cap is not serialized (docs/keccak.md Sections 4, 12)

Flock's `LigeritoProof` carries `initial_cap`, the level-0 Merkle cap, which is the
commitment itself (the event-2 cap of the sponge instance, the framed cap of the bridge
union); Flock's verifier rejects a proof whose `initial_cap` differs from the commitment's.

Code: `agg_keccak::flock_union::UnionProof::to_bytes` serializes the proof with an empty
`initial_cap`, `from_bytes` refuses a proof that carries one, and `UnionGeometry::verify` puts
the commitment's cap in its place before Flock's verifier runs. Prover, transcript and
verification are otherwise unchanged: an accepted proof is exactly one whose cap equals the
commitment's, so substituting it accepts the same proofs.

Why: the cap was sent twice, 4,096 bytes (`Fast`) or 2,048 bytes (`Slim`) per Flock
instance, two instances per tied part.

## K18. A rate-1/8 Ligerito parameter set, `Slim8`, in the vendored Flock core (spec Section 8; updates K12)

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

## K19. List-aware work at Flock's outer sites in the vendored core (spec 12.1; docs/soundness.md 4.3, 4.4; updates K12, K18)

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
upstream's do not verify here (mode A rejects the old format anyway: tied-part proof version
4, K20). The finding should be raised with the Flock authors, who may have an argument for
the outer sites that their audit document does not state; if so this patch can be dropped.

Known malleability: Flock's `verify_pow` (`flock-transcript` `83f963e`, `challenger.rs`)
accepts any `u64` nonce that passes the predicate, with no `g + 7` width bound, so the Flock
nonces inside a proof are not canonical (unlike the PIOP stream and the tie nonce, which are
bounded to `2^(g + 7)`). Another passing nonce gives another valid proof of the same
statement. Akita Proposition 10.3 does not need the bound, so soundness is unaffected; we do
not patch it (a bound in the vendored verifier would make the proofs canonical).

## K20. Tie work and the tie binding (tied-part proof version 4; docs/soundness.md 3.6, 4.5, 6.3)

docs/bridge.md 5c (version 3): the tie challenges `rho` were drawn without work, with the
bound `95 / 2^128` at a span of four segments, and the two instance challengers absorbed
neither the union's cap (the sponge instance) nor the final-vector digest.

Code (`agg_keccak::tied`, version 4): the tie bound carries the Johnson-list sizes of both
Ligerito commitments (the level-0 binding happens inside the openings, after `rho`):
`[2 L^2 (d + s + 5) + 2 L (log2 blocks + s + 9) + L (5 + s) + L (nu_s + 3)] / 2^128`
(`agg_bridge::pow::tie_numerator`), so the tie challenger grinds 16 / 17 / 18 bits (`Fast` /
`Slim` / `Slim8`) at a full segment or span with Flock's `grind_pow` before the first
coordinate; the nonce (below `2^(g + 7)`) travels as `LE32` in the proof header. After the
coordinates the tie challenger squeezes a 32-byte tie binding that both instance
challengers absorb (label `agg-keccak/v4/tie-binding`) before Flock binds its statement.
The difficulty is a function of the part's public geometry (`TiedPart::plan_tie_bits`), so
the plan of event 0 names it before the part exists. Cost: about `2^17` BLAKE3 evaluations
and 4 bytes per tied part.

Parts per segment (mode A, `agg_cli::mode_a::MAX_PARTS_LOG2`): at most two tied parts per
segment, `d_part >= max(3, d_seg - 1)`. The bridge seed `sd` is drawn while each part's
sponge commitment is bound only to its Johnson list, so with `P` parts in a segment its bad
set is a union over `L^P` list tuples (`docs/soundness.md` 3.5): `L^2 theta <= 2^-132.4` is
fine, `L^3 theta = 2^-127.7` (`Slim`) is not. The prover clamps `part_log2` into the admitted
range, and the verifier refuses a smaller `d_part` while deriving the plan of event 0
(`VerifyError::Plan`). Before this rule `agg aggregate --part-log2 3` on 48 records built six
parts in one segment and verified.

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

Code (`agg_piop::kernel`, `docs/piop.md` Sections 6, 16.2 and 18): the prover compiles the
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
instead of evaluating the 67 tables again; the round-1 commitment runs concurrently with
the Keccak side's commitments, and the three round-2 commitments concurrently with each
other (all transcript-free, absorbed in event order). Since the lookup layout (L4) the two
round-1 commits and the four round-2 commits run concurrently on scoped OS threads
(`pcs::concurrent_commits`), not as rayon jobs: Akita builds its NTT slots lazily inside
`OnceLock::get_or_init` with rayon-parallel loops, and two commits joined on the pool can
deadlock when the initializing worker steals the other commit, which re-enters the same
`OnceLock` (observed once, in the default-build `agg-cli` tampering test). No rayon worker
blocks on those OS threads: the round-1 commits used to run inside a `rayon::join` with the
Keccak side's commitments, so a worker waited for its commit threads while their Akita loops
needed a worker, and with a one-thread pool (`taskset -c 2`, `RAYON_NUM_THREADS=1`) the
prover hung in round 1. Now `pcs::overlap` runs the round-1 commits on the calling thread and
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
`N = 2,048` (`7ed4a32`, same session); they are 0.65 s now (`docs/piop.md` Section 18.3). The committed layout
is not changed: `docs/piop.md` Section 16.1 records why the quotient `v`, wider Akita
digits and merged Boolean lanes do not reduce the element count without weakening relation
H or adding a lookup argument.

## L1. Lookup layout L16 adopted; the base-16 digit path removed (spec Sections 4 to 7, 10 to 12; was Appendix C)

Spec (wire version 0): 23 committed columns per position (`z`, `u`, `v` in base-16 digits,
the six bits) and 40 base-16 digits on the bridge domains, ranged by `b16_prod` (stage 1),
95 leaves (stage 2) and 55 links (stage 3).

Code (wire version 1, `docs/lookup.md`): the lookup layout L16 of former Appendix C, now
normative in spec Sections 4, 7, 10, 11 and 12: 12 committed columns per position, every
16-bit value and every carry looked up in `T_16 = [0, 2^16)` by LogUp-GKR with two
independent challenges, `Z_hi` and `U_hi` kept on a base-4 and a base-8 leaf. The base-16
digit layout is removed, not kept behind a configuration switch. Why: a switch would have
duplicated the manifest, the three stage tables, the claim vectors, the transcript
registry, the wire format, the verifier and the catalogs (the four-group width-4 rows next
to the six-group two-family rows), so every protocol file would carry two layouts; the
measurements (`ref/notes/bench-local.md`, lookup integration) show no batch size at which
the base-16 layout is cheaper. The old layout stays reproducible at code `4247f61` (the
"before" of every comparison); its wire version 0 is refused by the parser.

## L2. Carries stored balanced like every looked-up column (docs/lookup.md Section 1)

`docs/lookup.md` Section 1 (proposal) stored the carries as they are (`0 <= kappa <= 1023`
honestly). Code: every looked-up column, carries included, stores `K = f - 2^15` and is
looked up as `f = K + 2^15` (`manifest::LOOKUP_SHIFT`; the carry layout is one unsigned
base-`2^16` digit, `kappa = K + 2^15`); the padding slots of the carry stacks hold `-2^15`
(looked-up value 0). Why: one storage shift for all four stacks gives one leaf formula
`1 / ((X - 2^15) - K)` and one stage-3 stack claim, and makes the width-16 admission interval
`[-2^15, 2^15)` coincide with `T_16` for every looked-up column (as-is storage would admit
negative carries that only the lookup rejects). Soundness is unchanged: the lookup proves
`K + 2^15 in [0, 2^16)` for every cell, the range of `docs/lookup.md` Section 2.

## L3. One forced balanced base-`2^16` inner digit for the width-16 preset (docs/pcs-config.md Section 2)

`docs/lookup.md` Section 6 left open whether the width-16 preset should take Dense's
cost-optimal inner digits or one forced base-`2^16` digit. Planner probe
(`width16_inner_probe`, 2026-10-24): at `(18, 4)` and `(20, 4)` Dense's search (`3..=16`)
selects two base-`2^9` digits, a forced `(16, 16)` one digit and a forced `(8, 8)` three,
with the same estimated opening proof (60,877 B and 61,707 B). Code: inner range `(16, 16)`,
the fewest inner digits (12 per position in all), hence the least commit work for the same
estimated proof bytes.

## L4. Six groups, two families, `G_m` as Akita's final group (spec Section 4.1)

`docs/lookup.md` Section 6 item 1 left the group split of `D_2` and the final group open.
Code: `G_main16` (width 16), `G_main4` (width 4), `G_1`, `G_2` (parities and carries in one
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

## L8. Relation H-u replaces relaxed H; `u` by one six-valued digit (spec Sections 3, 4.3, 7.4, 10, 12; Appendix A; `docs/h-exact.md` step A)

Spec (revision 4, wire version 2): the relaxed relation H with `(bar B_z, bar B_u) = (2^17,
2^18)`, `u = U_lo + 2^16 U_hi + 2^15` with a base-8 `U_hi` on `leaf8` (degree 5), so
`zeta' = 421,888`; Appendix A's H-exact `u` form `118,787 a_u + b_u - 190,465` needs two small
columns. Code (wire version 3, decision 2026-10-25): relation H-u, `z` unchanged and `u =
U_lo + 63,079 U_hi + 31,540` with `U_lo` looked up (`B_ulo = U_lo + 2^15`) and `U_hi = H_u - 3
in [-3, 2]` on `leaf6` (`Q_6 = P (P - 2)(P - 6)`, degree 4), which covers exactly `[-190,465,
190,465]` because `5 * 63,079 + 2^16 = 2 B_u + 1`; hence `zeta' = zeta = 350,209`. The relaxed
`u` path is removed, not kept behind a flag (as for L1: a flag would duplicate the stage-2
degree, the leaf and the encoder for a dominated variant). Same columns, groups, alphabets'
widths and catalog rows; event 0 changes only in the version (3), the relation byte (0 =
H-u, `manifest::Relation`; 1 is reserved for H-exact, step B) and the alphabet code of
`U_hi` (8 -> 6). Stage 2 drops from degree 5 to 4 (16 B per round less: 5,616 B of stage
messages and claims at `2^15` instead of 6,016); every proof-of-work difficulty and plan
digest is unchanged (`bit_length(4) = bit_length(5)`), the round-by-round ledger falls from
`1,434/|K|` to `1,409/|K|` at `2^15` (`docs/soundness.md` Section 10). The witness crate's
default layout (`Layout::H`, id 0) follows: `u_0..u_3` base 16 (`B_ulo`), `u_4` base 6
(`docs/witness.md`). Why the default changed: H-u dominates relaxed H in security (ML-DSA-44's
own SelfTargetMSIS instance, core-SVP 123.5 instead of 119.7, `ref/notes/msis-estimate.md`)
and in cost (no extra column, cheaper stage 2).

## L9. Relation H-exact as a runtime relation: exact `z` by a bit-slack form, one six-valued `u` digit, Booleanity at `r^(2)` (spec Sections 3.1, 4.1, 4.3, 7.4, 7.5, 10, 12; Appendix A; `docs/h-exact.md` step B)

Spec (Appendix A before revision 6): H-exact behind a feature flag with `z = 130,915 a_z +
b_z - 130,993` and `u = 118,787 a_u + b_u - 190,465`, i.e. two Boolean columns `a_z, a_u`
checked in stage 1 and a base-4 digit of `b_u` on `leaf4` (+2,048 committed elements per
signature). Code (decision 2026-10-25, design X1 of `docs/h-exact.md`): a runtime relation
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
  requires (there called `RelationPolicy::Exactly`; a plain `Relation` argument is the same
  policy). No wire-version bump beyond L8's version 3: H-u proofs are byte-identical.

Why: the 2026-10-25 decision record names two results (H-u and H-exact); a runtime relation
lets one build benchmark both back to back, and X1 is the cheaper layout where the prover
spends its time (`docs/h-exact.md` Appendix X). The witness crate's base-16 H-exact layout
(feature `h-exact`, `Layout::HExact`) is unchanged and serves only as a cross-check.
