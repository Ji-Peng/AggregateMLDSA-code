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
268,800 chi bits = 275,713 (useful 275,776 of 2^19; the dense-stack commitment stores only
the 2,155 used words per row). Dummy rows are identically zero (Flock's count-derived
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

## K6. Bridge words not yet wireable

Spec Section 8: the bridge slot copies "every live record's 48 w1Encode words".

Code: the 48 words sit at aligned word columns 4..51 of the sponge gate, but the IO
schema lists only `mu` and `c~`. The bridge-slot stage adds the 48 words as `In` schema
words (after `mu`); the schema is digest-visible, so that changes the circuit digest.

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

Still open, and planned rather than a defect of this stage: the bridge slot
(`bridge_slot.rs`) and the segment driver (`segment.rs`) are skeletons for the bridge-slot
stage of M3 (spec Section 14; K6). The Section 8 requirements stand: the bridge type's
relation depends on `sd`, so its digest (covering `sd` and `(J_t, m_t, k_t)`) is absorbed
into the Flock challenger next to the gate digest (the registry digest covers only the
matrix stubs, K2), and `Y_sigma` is a public input wired like `mu` and `c~`. The split API
is what allows that order: the bridge slot's witness (copies of the w1Encode words) does
not depend on `sd`, so it is committed with the Keccak slot before event 2. Because
`fs_sigma` is drawn after `sd`, the Flock transcript already depends on `sd` through its
seed; the digest is what binds the relation itself.
