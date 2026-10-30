# PIOP (`agg-piop`): tables, commitments, sum-check stages, transcript and wire format

The range mechanism is the lookup into `T_16` (lookup layout L16, digit layout id 1, wire
version 4): `docs/lookup.md` has the argument, the soundness and the remedies for the
identity term; `docs/deviations.md` P1 to P12, L1 to L10 and D2 the departures from the spec;
`docs/spec.md` Sections 4, 7, 10 to 12 the normative text. The relations are H-u (relation
byte 0, the default), H-exact (byte 1) and H-FIPS (byte 2), `docs/h-exact.md`.

Part 1 (prover-side tables, commitments, public data, virtual evaluators; spec Sections
2, 4, 5, 7.6), part 2 (the lookup, the three sum-check stages, the weight tables, the
transcript, the proof wire format, the prover and verifier drivers; spec Sections 6, 7, 10,
11) and part 3 (the timed, streamed verifier and the mode-B aggregate, `prove_mode_b`
/ `verify_mode_b`; spec Sections 11, 14).

## 1. Modules

| Module | Content | Spec | Aerie counterpart |
| --- | --- | --- | --- |
| `manifest` | `CommitmentManifest`: the seven groups (role, round, family, logical and physical variables, polynomial order and alphabets) under the batch's relation, the columns and their value forms, the lookup shape, event-0 binding bytes | 4.1, 4.3, 10 | `aerie-falcon/src/protocol/manifest.rs` |
| `source_tables` | `MainTables` (round 1, `G_main16` and `G_main4`), `BridgeGroupTables` (round 2, `G_1, G_2, G_3, G_m` and public `Y`, from `agg_bridge::parity` at `Shape::PRODUCTION`, with the multiplicities of `T_16`), `Cells`, `SourceTables` view | 4.1, 4.2, 9.2 | `aerie-prover/src/protocol/source_tables.rs` |
| `pcs` | `group_polynomials`, `TableCommitter` (two rounds, the groups of a round committed concurrently, batched opening from given claims `open_claims` or table evaluations `open`), `opening_claims`, `group_points`, `verifier_for` | 4.1, 7.5, 10 | `aerie-prover/src/protocol/pcs/group.rs` |
| `public_data` | `PublicData` (expanded keys, `mu`, `c~`, `T`), `KeyCache`, `t_evaluation`, `a_row_combination`, `negacyclic_kernel` and the allocation-free `negacyclic_kernel_into` (7.8), `eq_correlation` (Aerie (8.8)-(8.10)) | 2, 7.6 | `aerie-falcon/src/protocol/ring/functional.rs` |
| `virtual_polys` | `w_1'`, the compositions of `z, u, v` and the carries from their columns, ring residual, `y^(1)` parities, `P_Z`, `MainValues`/`CarryValues` | 4.3, 5 | |
| `kernel` (`kernel/`: `mod.rs`, `plan.rs`, `coded.rs`, `field.rs`, `layer.rs`, `tests.rs`) | `Source`/`Lane` (the narrow input tables), `Term` (summand shapes), `Group` (one local domain, optional split equality factor, gamma-weighted instances; coded columns, pattern masses, pair/quartet/Taylor tables, one dense table from the third fold), the dedicated groups of `layer`, compressed messages, lifted inactive rounds | 7.1 | `aerie-prover/src/protocol/kernels/`, `algebra.rs` |
| `lookup` | LogUp-GKR range argument: stacks, multiplicities and their digits, fraction trees, sessions on the kernel, `prove_in`/`verify_in` on a `LookupTranscript` (Section 7.1) | 4.3, 7.0 | |
| `range` | the digit map `z -> z(z+1)`; the leaf-checked digits (`Z_hi` base 4, `U_hi` base 6) and the order of their leaf and link instances | 4.3 | `aerie-falcon/src/protocol/relations/quartic.rs` |
| `stage` | `StageSpec` (declarative instance table: groups, lanes, equality points, gamma indices), `prove_stage`, `prove_stage_with`, `split_linear`, `replay_stage`, `check_terminal` | 7.1 | `aerie-falcon/src/protocol/stage.rs` |
| `stage1`, `stage2`, `stage3` | the instance tables of spec Sections 7.3 to 7.5, claimed sums, terminal claims, `prove`/`verify` | 7.3-7.5 | `stage1.rs`, `stage2.rs`, `stage3.rs` of both Aerie crates |
| `weights` | `w_1, w_2, w_{3,t}` tables and verifier evaluations (wrappers of `agg_bridge::weights` at the production shape); `w_A` table and (7.10) | 7.6 | `ring/functional.rs`, `stage1/projection_link.rs` |
| `claims` | the claim vectors between stages and their wire order | 7.2-7.5 | |
| `tables` | `ProverTables`: borrowed `i8` columns and `i32` lookup columns, `y^(1)` as `Parity1` (`i8` when every value fits, else `i32`), the weight tables when known; `LaneId -> Lane` | 7 | |
| `transcript` | Spongefish `Blake2b512` stream, event registry, samplers, work nonces, `pcs_session` | 10 | `aerie-falcon/src/protocol/transcript.rs` |
| `pow` | the proof-of-work plan (`PowPlan`, its digest) and the packed nonce stream (`WorkWriter`, `WorkReader`) | 10, 12.1 | |
| `binding` | public-record digest (16-KiB-leaf cSHAKE256 tree), instance record of event 0, `FlockProfile` | 2, 10 | `public_records_binding.rs` |
| `wire` | `AggregateProof`, byte format, packed final vectors, `ProofSize` | 10 | `wire.rs` |
| `opening` | opening claims at `r^(3)` and its suffixes and at the lookup's table point, commitment parsing | 7.5, 10 | `pcs/verifier.rs` |
| `prover`, `verifier` | the drivers in event order with per-step timings (`ProveTimings`, `VerifyTimings`); hooks `BridgeDerivation`, `KeccakSide`/`KeccakVerifier`; `prove_piop`/`verify_piop`, the PIOP part shared with the stage tests | 10, 11 | `transport.rs` |
| `public_stream` | `StreamedPublic`: the verifier's light per-record data (step 2) and the one pass of step 6 (`T(r^(2))`, `w_A(r^(3))`) | 2, 11 | |
| `mode_b` | mode B: public `w_1'` bits as the Keccak side (`PublicBits`, `PublicBitsVerifier`), the GF(2) projection, the bit check, `prove_mode_b`, `verify_mode_b` | 11, 14 | |

`agg-pcs-config`'s `PcsProver::commit_polynomials` commits already materialized
polynomials; `commit_group` delegates to it.

## 2. Manifest and the event-0 bytes

Seven groups (spec Section 4.1), in wire order, under relation H-u:

| Group | Columns (manifest order) | Width | Storage | Round |
| --- | --- | ---: | --- | ---: |
| `G_main16` | `Z_lo, U_lo, V_lo, V_hi` (`V_lo, V_hi` not looked up, deviation K24) | 16 | one stacked `i32` lane (`MainTables::wide`), column `c` at `c 2^d_main` | 1 |
| `G_main4` | `Z_hi, U_hi, b_0..b_5` | 4 | `i8` columns (`MainTables::columns`) | 1 |
| `G_1` | `kappa1_0..5` (stored `K = kappa - 2^15`) | 16 | stacked `i32`, 8 slots, slots 6, 7 hold `-2^15` | 2 |
| `G_2` | `y2_0..5`, `kappa2_0..5` | 16 | `i8` parities, stacked `i32` carries | 2 |
| `G_3` | `kappa3`, `zero_1..zero_11` (`G_2`'s Akita layout, K24) | 16 | `i32`; one shared `i8` zero column | 2 |
| `G_k` | `k_0, k_1, k_2`: the digits of the projection outputs `k_{o,i} = <Psi_o, v_i>` on `D_k` (cell `o + 128 i`) | 16 | stacked `i32` `k_0 | k_1`, `i8` `k_2` (`projection::ProjectionTables`) | 2 |
| `G_m` | `m_0..m_6` (digits of the multiplicities of `T_16`) | 4 | `i8` | 2 |

`G_m` is Akita's final group. The relation changes the columns (`manifest::group_columns`):

- H-exact: `G_main16` = `bz_lo, U_lo, V_lo, V_hi` (`bz_lo` = `S_zlo` in the slot of `Z_lo`,
  so the main lookup stack keeps its shape), `G_main4` = `bz_hi, U_hi, b_0..b_5, a_z` (9
  polynomials), `z = S_zlo + 2^16 B_zhi + 130,915 a_z - 98,225` (`DigitValue::Zx`,
  `ValueForm::BitSlack`).
- H-FIPS: `G_main16` = `bz_lo, l_lo, v_lo, v_hi`; `G_main4` = `bz_hi, l_hi, b_0..b_5, a_z, s,
  delta, e` (12 polynomials); `G_2` gains `u_h` (13 polynomials) in slot 6 of lookup stack 2.
  `z` as H-exact's, `L = l_lo + 29,696 l_hi + 2^15` (`DigitValue::L`, `ValueForm::Slack`),
  `u = gamma_2 s + L + 1 - gamma_2 - 2 gamma_2 delta - e` (affine).

`source_tables::Cells` reads a column of either width (`Narrow(&[i8])`, `Wide(&[i32])`);
`Cells::lane` gives a coded byte lane or an `Ints` lane. `MainTables::build` derives every
witness (strict) and encodes `z + 2^17 = z_lo + 2^16 z_hi` (stored `Z_lo = z_lo - 2^15`,
`Z_hi = z_hi - 2`), `u` by relation H-u's step form (`U_lo = B_ulo - 2^15`, `U_hi = H_u - 3`
in base 6, spec (4.2)) and `v` (two 16-bit halves); padded rows hold the digits of zero. The
encoder resolves the slot of every digit of `z, u, v` and of the bits from the manifest;
under H-exact `a_z = [z >= 79]` and the padded rows are `(bz_lo, bz_hi, a_z) = (32,689, 1,
0)`; under H-FIPS the coefficient witness comes from `(u, w_1')` (`fips_coefficient`) and the
hint slack per record is kept in `MainTables` for round 2 (`hint_slack_table`).
`BridgeGroupTables::build` adds the balanced carries, the parities and `Y` of
`agg_bridge::parity::compute` and the multiplicities of every looked-up cell of both rounds
(`recount`, `lookup::multiplicities`), with their digits.

`CommitmentManifest::binding_bytes` (all integers little-endian):

| Field | Encoding |
| --- | --- |
| wire version | `u16` = 4 |
| parameter set | `u8` = `0x44` |
| relation byte | `u8` (0 H-u, 1 H-exact, 2 H-FIPS) |
| `N`, `N*` | `u64`, `u64` |
| digit layout id | `u8` = 1 (the lookup layout) |
| group count | `u8` = 6 |
| per group, wire order | role `u8`, round `u8`, family `u8` (0 = width 4, 1 = width 16), logical vars `u8`, physical vars `u8`, polynomial count `u8`, then one alphabet code per polynomial: the base of a leaf digit (4, 6), `0x90` for a looked-up column, `0x50` for a multiplicity digit, `0x23` for H-FIPS's `delta` and `e`, 0 for a Boolean |
| catalog row id | `u16` length, ASCII (`mldsa-44-l16-batch-<N*>-mult-final`, `l16x` under H-exact, `l16f` under H-FIPS) |
| bridge parameters | for levels 1..3: `J_t` `u64`, `m_t` `u64`, `k_t` `u8` |
| lookup layout | challenges `u8` (2), table count `u8` and widths `u8` (16), stack count `u8`, per stack column count `u8`, variables `u8`, table `u8`, shift `i32` |

The instance record of event 0 (`binding::instance_record`) is these bytes followed by the
32-byte public-record digest, the Flock profile id (`u8`: 0 none, 1 `Fast`, 3 public bits,
4 `Slim`, 5 `Slim8`; other bytes are rejected) and the digest of the proof-of-work plan
(Section 9). The record digest: the stream `LE64(N) || (pk || LE64(|M|) || M || c~)*` is cut
into 16-KiB blocks, each hashed by cSHAKE256 under `agg-mldsa/v1/record-leaf`, and the root
is cSHAKE256 of `LE64(blocks) || leaves` under `agg-mldsa/v1/public-records-binding`.

## 3. Commit flow

1. `MainTables::build(manifest, instance, sigs)` (strict derivation) or `from_values`; both
   check shape and honest alphabets (encoder self-check, not a soundness check).
2. `TableCommitter::new(manifest)`: work plan from the catalog, `manifest.matches_plan`,
   seed-derived Akita setup. `commit_round1` commits `G_main16` and `G_main4` concurrently
   (event 1); `pcs::overlap` runs the pair on the calling thread while the Keccak side's
   commitments run on a scoped OS thread.
3. After the bridge seed: `BridgeGroupTables::build(manifest, main, matrices)` (honest abort
   on a row weight above the cap or a cell outside `T_16`), `commit_round2` commits the four
   round-2 groups concurrently (event 4), each commit on its own scoped OS thread
   (`pcs::concurrent_commits`): Akita's lazily built NTT slots sit behind a `OnceLock` whose
   initializer runs rayon loops, so commits joined as rayon jobs can deadlock (deviation
   P12). No rayon worker waits on those threads: on a worker, or with a one-thread pool, the
   commits run one after another on the calling thread (`tests/single_thread.rs`). Each
   commit is transcript-free and the final group is committed against the plan's
   precommitted profiles; the prover absorbs the commitments in wire order.
4. After stage 3: `open_claims(commitments, claims, session)` proves one batched opening of
   the stage-3 claims (31 under H-u, 32 under H-exact, 36 under H-FIPS) at the suffixes of
   `r^(3)` and of the 7 multiplicity-digit claims at the lookup's table point `pi_m`
   (`pcs::group_points`). The claims are the stage-3 terminal values (the multilinear
   extensions of the committed tables) and the digit claims of the lookup; a debug assertion
   and `tests/all/prove_verify.rs` compare them with `opening_claims`, which evaluates every
   table again (`open` does that). `verifier_for(manifest)` verifies from the setup seed
   alone; `opening::parse_commitment` reads a commitment with the coefficient count of the
   plan's profile, never from the proof.

## 4. Public data

`PublicData::prepare(inputs, cache)` stores everything: `pkDecode`, ExpandA and 16
inverse NTTs per distinct key (skipped for keys in the `KeyCache`), `mu`, `SampleInBall`
and `T` per record. `PublicData::from_instance` shares the prover's expanded keys (prover
and tests). The verifier uses `public_stream::StreamedPublic` instead (Section 11): step 2
keeps per record only `mu`, `c~` and the sparse `c`, per distinct key `rho`, `t1` and `tr`;
`A` and `T_i` are formed inside the pass of step 6 and dropped, and new expansions go to the
`KeyCache` when one is given.
`t_evaluation(r)` streams `T(r)`; `negacyclic_kernel(a, r)` is `K_{a,r}` of (7.8) by
Aerie's recurrence (8.7); `eq_correlation(l, n)` is Aerie's `Lambda` by the carry trie.

## 5. Virtual evaluators

Affine in committed evaluations: `w1_from_bits`, `compose_digits` (the value forms of `z, u,
v` and the carries), `residual_high`, `MainValues`, `CarryValues`. Boolean-domain
materializations: `w1_table`, `compose_table`, `u_table`, `residual_table` (diagnostic),
`parity1_tables` (`y^(1)_t = sum M_1 b_t - 2 kappa^(1)_t`), and for the leaf digits
`digit_square` (`P_Z = Z (Z + 1)`) and `leaf` (`Q_4`, `Q_6`).

## 6. Sum-check kernel

Every stage is one batched ordinary sum-check of `d_main` rounds (Aerie 10.3): a round
message is the degree-`d` round polynomial without its linear coefficient (`d` elements;
`d = 4, 4, 3`), recovered by the verifier from `g(0) + g(1) = claim`. Instances on a smaller
domain `D_t` (always a suffix of the main variables, spec 7.1) are inactive in the
leading `d_main - d_t` rounds, where their message is the constant `claim / 2`, and are
lifted by `2^-(d_main - d_t)` at the terminal.

### 6.1 Generic groups (`kernel::Group`)

`kernel::Group` holds the lanes of one local domain. Every lane starts as a narrow
`Source` (`i8` cells with a per-cell integer map, `i32` cells, or a materialized `K`
weight). Within a group the instances share one equality factor, kept as Akita's
`GruenSplitEq` (two tables of about `2^(d/2)` entries; the full `eq` table is never
materialized), and their gamma-weighted summands (`Term`s: products, `v(v-1)`, the bit-range
expression, leaf predicates `Q(P)`, links `Z(Z+1)`, affine combinations with an optional
weight lane, monomial sums) are evaluated at `d` or `d + 1` interpolation points per pair of
cells and interpolated to coefficients (Newton), then multiplied by the equality line of the
current variable (`gruen_mul`) and the lift. Groups without an equality factor carry their
weight as a lane. A group's claim is computed from its own tables (directly for lifted
groups, from `g(0) + g(1)` of round 0 otherwise); the stage driver checks the sum of the
group claims against the gamma-weighted claimed sums at the end (prover self-check; the
test-only unchecked provers skip it so tampered witnesses reach the verifier).

How a round is evaluated (all paths compute the same polynomial, so the proof bytes do not
depend on them; `kernel/mod.rs` holds the schedule, `kernel/plan.rs` the compiled pieces,
`kernel/coded.rs` the codes and pattern masses, `kernel/field.rs` the field helpers):

- *Compiled plan.* `Group::new` compiles the gamma-weighted terms into one polynomial per
  lane (every single-lane term: `Bool`, `Link`, `Leaf`, one-lane `Product`, the lanes of a
  weightless `Affine`; constants go to one per-cell constant), bilinear products
  `c v_a v_b`, weighted sums `w (sum c_l v_l + c)` (every `Affine` with a weight and every
  product of a field lane with another lane, merged per weight lane), and the residual
  generic terms (`RangeW1`, products of three lanes; `Term::Monomials` is expanded into
  polynomial, bilinear and generic pieces). Everything is evaluated in coefficient form at
  `v = a + X d` (Taylor coefficients `P^(i)(a) / i!` times `d^i`). A lane read only by its
  linear polynomial, by one bilinear product or once in one weighted sum absorbs its
  coefficient at its first fold (the terminal values are unscaled before they are
  reported).
- *Coded columns.* The `i8` lanes over one column (by pointer) code the column by the
  classes of its present cells under the tuple of their maps, if there are at most 16 (a
  width-4 digit has at most its base of classes, fewer when it is read only through
  `z (z + 1)`, which identifies `z` and `-1 - z`; a bit has 2), with one byte of pair code
  per pair of cells (`class(2k) n + class(2k + 1)`). A column with more classes (a tampered
  cell can cause that), the `i32` lanes (the looked-up columns, the carries) and the `i32`
  parity fallback stay uncoded. Pieces on coded lanes are *bucketed*: per column
  (polynomials, same-column products), or per set of columns for a generic term whose joint
  pair space is at most `2^12` (`RangeW1` on four bit columns: 256); the other pieces
  (weighted sums, uncoded lanes, cross-column products) are *direct* and evaluated per item.
- *Level 0* (first active round). If the group has at least two active rounds, the largest
  quartet space is at most `2^16` and the quartets outnumber it by `BUCKET_MIN_RATIO = 4`,
  one pass accumulates per segment the *quartet masses* (the round-1 equality weight
  `eq(tau_{2..}, q)`, or one without an equality factor, per quartet pattern); round 0 is
  `sum_p (l_0 H_0[p] + l_1 H_1[p]) F(p)` over the pair patterns with the two marginals
  `H_0, H_1` and `l_0 = 1 - tau_1`, `l_1 = tau_1` (Aerie's bivariate prefix), and the
  masses are kept for round 1. Otherwise pair masses with the round-0 weights. Narrow
  segment sets (at most `2^16` entries in all) accumulate in one table per item chunk
  (Aerie's `chunked_bucket_sums`); wide ones one table per segment, each task owning whole
  segments with the item weights computed once. Direct pieces run one pass over the pairs;
  weighted sums read per-byte tables `c_l map_l[byte]` of their byte lanes (additions, no
  products).
- *Bind 0.* Coded lanes become pair tables (`n^2` entries, scaled; no storage per cell);
  uncoded lanes are folded into `K` tables.
- *Level 1.* With quartet masses, every quartet pattern once with the pair tables of its
  two pairs; for a column read only by single-lane polynomials, by moments
  `m_j[left] = sum_right M[left, right] L[right]^j` and the Taylor coefficients at
  `L[left]` (one unreduced product per pattern and power). Direct pieces, and every piece
  without masses, per item.
- *Bind 1.* Coded lanes of columns with at most 4,096 quartet patterns become quartet tables
  (when the octets outnumber the patterns), with Taylor tables of their polynomials of
  degree 2 to 4 (coefficients 1 to 4 per entry, one 64-byte line); the others keep the pair
  table and the second challenge.
- *Level 2.* Per item: quartet tables (two reads per lane), Taylor tables (seven products
  for a quartic), or the pair table folded on the fly.
- *Bind 2.* Every lane is written into one row-major `K` table at an eighth of the domain
  (row blocks filled lane by lane, after a parallel first touch); a quartic lane read only
  by its polynomial is stored translated by `s = -c_3 / (4 c_4)` (Aerie's depressed
  quartic: nine products per lane and pair instead of fourteen).
- *Dense rounds* (level 3 on): per pair over the table; folds in place (Aerie's
  `fold_rows_in_place`).
- *Arithmetic.* From level 1 on, the constant coefficient of the inner polynomial is not
  evaluated but recovered from the running claim, which is the group's own exact sum
  (`claim / lift = scalar q(0) + l(1) (q(1) - q(0))`; `scalar` the equality factor's running
  scalar, 2 without one). The last product of every coefficient is accumulated unreduced
  (`mul_unreduced`), each item's coefficients are reduced once and contracted with the split
  equality tables unreduced (Aerie's `sum_split_coefficients`). Products use a branch-free
  `K` multiplication (`kernel::field::fm`: four `64 x 64` products and two reductions with
  `2^64 = 59`, `2^128 = 3481` modulo `p`); `fmi(x, k)` multiplies by a signed integer (two
  `64 x 64` products, branch free: the sign of a balanced digit is random) and folds the
  integer and byte lanes.
- `AGG_STAGE_TRACE=1` prints per group the construction time, the message and fold time
  summed over the rounds, the per-round wall time, and the weight-table build times
  (`stage::prove_stage_with`, `prover::prove_with_tables`); split groups are attributed to
  their spec group.

The weight tables `w_1`, `w_A` are `512 MB` each at `N = 2^15` (16 bytes per `K` element on
the `2^25`-cell main domain); the bridge domains are at most `2^19` cells.

### 6.2 Dedicated groups (`kernel::layer`)

The round polynomial of a group is unique, so a group may compute it any exact way.
`kernel::layer` has a round driver (`Engine`) for groups whose summand is a coefficient-free
bilinear form `sum_l L_l + sum L_i L_j` on a row-major table of transformed lanes: one pass
per round folds four rows into two by the previous challenge and accumulates `q(0)` and the
leading coefficient of the inner polynomial on the new pair; the linear coefficient comes
from the running claim (Gruen's factorisation of `eq`, as in `Group`); the first round reads
the group's level-0 data and computes `q(1)` too (the group's claim comes from its tables).
A self-check compares the final claim with `eq` times the summand at the terminal row.

- `LayerGroup`: the layer group of a lookup session (`docs/lookup.md` Section 4.6 item 3).
- `IntPairGroup`: integer lanes `S_0, S_1 (, M_0, M_1)` with the summand `c + c_s (S_0 + S_1)
  + c_m (M_0 + M_1 + S_0 S_1) + c_k (S_0 M_1 + S_1 M_0) + c_mm M_0 M_1`; the first round on
  exact `i64` products weighted without reduction (`field::mad_int`), the table holds
  `S_0 (, M_0)` and `W = C (1, S_1, M_1)`. Used by the lookup's bottom session and session
  `D - 1` (`docs/lookup.md` Section 4.6 items 1, 2).
- `LinearIntGroup`: `sum_x eq(point, x) sum_l c_l f_l(x)` over integer lanes, as the single
  field lane `g = sum_l c_l f_l` built once (unreduced products, one reduction per cell). It
  yields no terminal lane values; the stage driver reads them from the generic group that
  also reads the lanes at the same point.
- `WeightIntGroup`: `gamma sum_x w(x) g(x)` without an equality factor, for a field weight
  `w` and the exact integer combination `g = sum_l c_l f_l + c` (stage 3's `w_A z`); the
  engine's eq-free mode (`s = q`, the linear coefficient from `s(0) + s(1) = claim`) and a
  coefficient on every round polynomial (`gamma`). The table carries `w`, `g` and the integer
  lanes (folded for their terminal values).

### 6.3 Stage driver (`stage::prove_stage_with`)

- A full-domain group without an equality factor whose single instance is one weighted
  affine term with a field weight and integer or byte lanes runs on `WeightIntGroup` (stage
  3's `w_A` group).
- A full-domain equality group whose lanes are all integer tables read only by single-lane
  products, each lane also read by a generic group of the stage, runs on `LinearIntGroup`
  (stage 3: the reduction of the main lookup stack, four 16-bit columns).
- `split_linear`: in any other full-domain equality group, the single-lane products and
  unweighted affine terms on integer lanes read nowhere else move to one `LinearIntGroup` at
  the same point; the lanes stay in the generic group, unread, so it still folds them and
  yields their terminal values (stage 3's `eq(r^(2))` group: the ring identity's terms on
  `U_lo, V_lo, V_hi`). The two groups' messages sum to the original group's.
- Byte lanes stay on the generic coded path (the six bits of stage 3's `eq(r^(1))` group).

## 7. Lookup and stage tables

Gamma indices are positions in the stage's `gamma^(s)` vector; "lift" is the number of
inactive leading rounds (`d_main - d_t`). The ring instance is one gamma on two groups (the
`w_A` lane group and the `eq(r2)` group); `z, u, v` enter as the affine combinations of
their columns and `w_1'` as `sum 2^t b_t`.

### 7.1 The lookup (event 7)

Shape (`CommitmentManifest::lookup_shape`): stacks `[Z_lo | U_lo]` (depth
`d_main + 1`, the first half of the `G_main16` lane), `kappa1` (8 slots, `d_1 + 3`), `kappa2`
(8 slots, `d_2 + 3`), `kappa3` (`d_3`), `[k_0 | k_1]` (`d_k + 1`); table `T_16`; challenges `X_1, X_2`; storage shift `2^15` (leaves
`1 / ((X - 2^15) - stored)`). `prover::prove_piop` runs `lookup::prove_in` on the PIOP
transcript, sends the seven digit claims `m_k(pi_m)`, then `tau`, the pre-stage claims and
the stages; `verifier::verify_piop` mirrors it (`VerifyError::Lookup` for the lookup and
the digit check). Lookup proof elements: 1,394 at `N = 1,024` with two
challenges (22,304 B), 1,891 at `2^15` (30,256 B), plus 112 B of digit claims.

### 7.2 Stages under relation H-u

Stage 1 (degree 4, `gamma^(1) in K^26`, point `tau`): `bool_y1[t]` (0..6), `level2[t]`
(6..12), `level3` (12), `bool_y2[t]` (13..19), `bool_b[t]` (19..25), `range_w1` (25); groups:
main `eq(tau)` {6 bits}, `D_1` `eq(tau_1)` {6 `y1`}, `D_2` `eq(tau_2)` {6 `y2`}, `D_1`
{`w_2`, 6 `y1`}, `D_2` {6 `w_3`, 6 `y2`}. Terminal (18): `y1_t(r1_1)`, `y2_t(r1_2)`,
`b_t(r1)`. Pre-stage (13): `y2_t(tau_2)`, `K2_t(tau_2)`, `K3(tau_3)`.

Stage 2 (degree 4, `gamma^(2) in K^9`, point `r^(1)`; before `gamma^(2)` the 6 claims
`K1_t(r1_1)`): `def_y1[t]` (0..6, main, lane `w_1`), `leaf4[Z_hi]` (6), `leaf6[U_hi]` (7)
(main `eq(r1)` {`P_{Z_hi}`, `P_{U_hi}`}), `leaf4[K_2]` (8; `D_k` `eq(r1_k)` {`P_{K_2}`}).
Terminal (9): `b_t(r2)`, `P_Z(r2)`; then the projection claim `c = k~(r')` (`r'` the `D_k`
suffix of `r^(2)`, site `proj/claim`).

Stage 3 (degree 3, `gamma^(3) in K^48`, point `r^(2)`):

| Instance | Gamma | Domain | Equality / weight | Summand |
| --- | ---: | --- | --- | --- |
| `ring` | 0 | main | lane `w_A`; and `eq(r2, .)` | `w_A (Z_lo + 2^16 Z_hi + 2^15) - eq (2 gamma_2 w_1' + u + q v)` |
| `link[Z_hi]`, `link[U_hi]`, `link[K_2]` | 1, 2, 3 | main; `D_k` for `K_2` | `eq(r2, .)`; `eq(r', .)` | `Z (Z + 1)` |
| `b_t` from `r1`, from `r2` | 4..10, 10..16 | main | `eq(r1)`, `eq(r2)` | `b_t` |
| `y2_t` from `tau_2`, from `r1_2` | 16..22, 22..28 | `D_2` | `eq(tau_2)`, `eq(r1_2)` | `y2_t` |
| `K2_t` from `tau_2` | 28..34 | `D_2` | `eq(tau_2)` | `K2_t` |
| `K3` from `tau_3` | 34 | `D_3` | `eq(tau_3)` | `K3` |
| `K1_t` from `r1_1` | 35..41 | `D_1` | `eq(r1_1)` | `K1_t` |
| stack `j` | 41 + j | stack domain | `eq(pi_j,low)` | `sum_c w_{j,c} f_{j,c}` (one part per column, scale `w_{j,c}`) |
| `proj_v` | 46 | main | lane `W` | `W (V_lo + 2^16 V_hi + 2^15)`, claim `c` |
| `proj_k` | 47 | `D_k` | `eq(r', .)` | `K_0 + 2^16 K_1 + 2^32 K_2 + 2^31 + 2^15`, claim `c` |

Groups: main `eq(r2)` {`U_lo, V_lo, V_hi, Z_hi, U_hi`, 6 bits}, main {`w_A`, `Z_lo`,
`Z_hi`}, main `eq(r1)` {6 bits}, `D_1` `eq(r1_1)` {6 `K1`}, `D_2` `eq(tau_2)` {6 `y2`, 6
`K2`}, `D_2` `eq(r1_2)` {6 `y2`}, `D_3` `eq(tau_3)` {`K3`}, main {`W`, `V_lo`, `V_hi`},
`D_k` `eq(r')` {`k_0, k_1, k_2`}, and one `eq(pi_j,low)` group per stack. Terminal (34,
`Stage3Claims`): every column of the six non-multiplicity groups at `r^(3)` but the zero
polynomials of `G_3`, manifest order; these are the opening claims together with zero for
the zero polynomials and the multiplicity digits at `pi_m`. The verifier evaluates
`W(r^(3)) = P~(r3_pos) eq(r'_sig, r3_sig)` from `Psi` (`projection::weight_at`).

### 7.3 Relation H-exact (`docs/h-exact.md` step B)

The relation (`manifest::Relation`) is a field of the manifest and selects, through it,
everything below; the verifier takes it from its caller (`verifier::verify(..., relation,
...)`, `VerifyError::Relation` on a header mismatch). Columns: Section 2. Stage 2:
`def_y1[t]` (0..6), `leaf6[U_hi]` (6); `gamma^(2) in K^7`, degree 4, terminal 7. Stage 3:
`ring` (0, the `w_A` group reads `bz_lo, bz_hi, a_z` with weights `1, 2^16, 130,915` and
constant `-98,225`), `link[U_hi]` (1), `bool[bz_hi]` (2), `bool[a_z]` (3) (`Term::Bool` in
the main `eq(r2)` group, claim 0), then the reductions from 4 (`stage3::Gammas`);
`gamma^(3) in K^45`, terminal 32. Registry: stage 2 (7, rounds of 4, 7), stage 3 (45,
rounds of 3, 32); the proof-of-work plan follows (stage-2 batching `g = 3`).

### 7.4 Relation H-FIPS (`docs/h-exact.md` Section 10)

Relation byte 2 (`Relation::HFips`); columns and values: Section 2. Stage 2: `def_y1[t]`
(0..6), `F3 = e (e - 1)(e - delta)` (6, `Term::Monomials`, main `eq(r^(1))` group);
`gamma^(2) in K^7`, terminal 8 (`b_t`, `delta`, `e`). Stage 3: `ring` (0), `bool[bz_hi]`,
`bool[l_hi]`, `bool[a_z]` (1..3), `F1 = s (s - 1)`, `F2`, `F45`, `F7` (4..7), 37 reductions
(8..44), `delta`, `e` from `r^(2)` (45, 46), the stacks (47..50); `gamma^(3) in K^51`,
terminal 36. `F7` lives in two groups at `EqPoint::HalfR2` (main `delta^2`, `D_2` `u_h`). The
proof-of-work plan equals H-exact's.

## 8. Weight tables (`weights`)

Point conventions (least significant coordinate first): a main point splits as
`(coef 8, row/column 2, sig d_sig)`; the suffix of length `d_1` as `(rho_1: 13, beta =
(i_mid, sigma))`; of `d_2` as `(rho_2: 13, sigma)`; of `d_3` as `(rho_3: 12, sigma)`.

- `w_1` on the main domain for `p1 = r1_1`: `c_1[zeta] = sum_{rho in column zeta of M_1}
  eq(p1_rho, rho)` for `zeta < min(2^19, 2^d_main)`, `w_1[x] = c_1[x mod 2^19]
  eq(p1_beta, x >> 19)`. Verifier: `eq(p1_beta, r_beta) sum_zeta eq(r_zeta, zeta) c_1[zeta]`
  (`k_1 2^19` additions for `c_1`, one `2^19` equality table).
- `w_2` on `D_1` for `p2 = tau_2`: `c_2` over the used `zeta_2 < 2^17` with `M_2` and
  `eq(p2_rho, .)`, times `eq(p2_sigma, sigma)`; zero-padded positions have zero weight.
- `w_{3,t}` on `D_2` for `p3 = tau_3`: `c_3[t + 6 rho_2]` with `M_3` and `eq(p3_rho, .)`,
  times `eq(p3_sigma, sigma)`.
- `w_A(j', s, i) = eq(r2_sig, i) K_{A~_{i,s}, r2_coef}[j']`, `A~_{i,s} = sum_rho
  eq(r2_row, rho) A_i[rho, s]`, one `negacyclic_kernel` per `(i, s)` (`O(256 log 256)`);
  zero on padded rows. Verifier (7.10): `Lambda = eq_correlation(r2_coef, r3_coef)`, then
  per live record `eq(r2_sig, i) eq(r3_sig, i) sum_{rho, s} eq(r2_row, rho) eq(r3_col, s)
  <A_i[rho, s], Lambda>` (`k l 256` multiply-adds per record, streamed over the keys).
- `ring_weight_table` builds `w_A` with the allocation-free negacyclic transform (one fold
  product per butterfly output); the row weights are scaled by `eq(r2_sig, i)` once per
  record and combined with the integer rows of `A` by base-field products.

## 9. Transcript (`transcript`)

Spongefish `Blake2b512` (`akita_transcript::TranscriptSponge`), tag
`agg-mldsa/v1/piop/blake2b` zero-padded to 64 bytes, session `B("agg-mldsa/v1")`, instance
`B(cSHAKE256(LE64(len) || record))` under `agg-mldsa/v1/instance`. Prover and verifier run
the same state machine (every payload is a public message; no Spongefish argument string).
Elements of `K` are rejection-sampled per 64-bit limb from eight squeezed bytes. The ordered
registry (`transcript::Registry`, spec Section 10) rejects a missing, extra or out-of-order
event before it reaches the sponge. Counts below are those of H-u (Section 7.3 and 7.4 for
the other relations):

| # | Site (`agg-mldsa/v1/...`) | Kind | Atoms | Payload |
| ---: | --- | --- | ---: | --- |
| 0 | `instance` | public | 1 | the instance record |
| 1 | `commit/main16`, `commit/main` | proof | 1 each | Akita native encodings of `G_main16`, `G_main4` |
| 2 | `commit/flock` | proof | `S` | `S` framed (`LE64` length) Flock commitments |
| 3 | `bridge/seed` | challenge | 1 | 32 squeezed bytes `sd` |
| 4 | `commit/level1`, `level2`, `level3`, `mult` | proof | 1 each | the round-2 commitments |
| 5 | `flock/seeds` | challenge | `S` | `32 S` squeezed bytes `fs_sigma` |
| 6 | `flock/final-vectors`; `flock/shares` | proof | `S`; `S` | `512 S` bytes `Y_sigma`; `S` framed share items (deviation K22) |
| 7 | `lookup/x`; `lookup/layer1`; `lookup/mu/1`; per session `s = 2 .. D`: `lookup/gamma/s`, `lookup/round/s/k` and `lookup/r/s/k`, `lookup/terminals/s`, `lookup/leaves/s`, `lookup/mu/s`; `lookup/multiplicities` | challenge; proof; challenge; (challenge, proof and challenge, proof, proof, challenge); proof | 2; 40; 1; (2 per active tree, 3 and 1, 4 per continuing tree, 2 per ending stack or table, 1); 7 | the lookup, then the digit claims `m_k(pi_m)` |
| 8 | `tau` | challenge | `d_main` | `tau` |
| 9 | `prestage/claims` | proof | 13 | pre-stage claims |
| 10 | `stage1/gamma`; `.../round/{r}/message`, `.../challenge`; `.../terminal` | challenge; proof, challenge; proof | 26; 4, 1; 18 | |
| 11 | `stage2/kappa1`; `stage2/gamma`; rounds; terminal | proof; challenge; (4, 1); proof | 6; 8; ; 8 | |
| 12 | `stage3/gamma`; rounds; terminal | challenge; (3, 1); proof | 44; ; 31 | |
| join | `flock/proofs` | proof | `S` | `S` framed Flock proofs, after the stage-3 terminal claims (deviation K22) |
| 13 | `pcs/session` | export | 1 | 32 bytes squeezed from a sponge clone, the seed of the Akita opening |

Grinding (`docs/soundness.md` Section 6): every field-challenge event carries a difficulty
from Aerie's assigned bounds (`Registry::work_sites`: `X` 0, a merge 1, a GKR batching
vector `bit_length(2 active)`, a GKR round 2, `tau` `bit_length(d_main)`, the stage batching
vectors 5, 4, 6 (H-exact and H-FIPS: 5, 3, 6), the stage rounds 3, 3, 2). With `g > 0` the
transcript absorbs `LE32(nonce)` (`nonce < 2^(g + 7)`), squeezes a 32-byte predicate whose
low `g` bits must be zero and then the challenge; the prover searches the smallest passing
nonce and packs it into the work stream (`Transcript::new`, `work_stream`), the verifier reads
and checks it (`Transcript::replay`, `TranscriptError::Work`), the stream's exact length and
zero padding checked up front and its exhaustion at `ensure_complete`. The seeds and the PCS
export carry no work. The plan (`pow::PowPlan`: the Keccak side's tied-part sites (tie,
`eq_Y`, chain rounds), then these sites) is hashed into the instance record of event 0
(`binding::instance_record`).

## 10. Wire format (`wire::AggregateProof`)

`LE16` version 4 (`manifest::WIRE_VERSION`), `u8` relation byte (0 = H-u, 1 = H-exact,
2 = H-FIPS), `LE64 N`; `32 N` bytes `c~`; the `G_main16` and `G_main4` commitments framed
(`LE32` length); event 2: `LE32 S` then `S` framed items; event 4: four framed items; event
6: `512 S` bytes then `S` framed share items (deviation K22); the lookup (layer-1 values,
then per session the rounds, the terminals and the leaves) and the 7 digit claims; events 9
to 12 as raw atom encodings of 16 bytes per element (13 pre-stage claims; `4 d_main` stage-1
round elements and 18 claims; 6 `kappa1` claims, `4 d_main` round elements and 8 claims;
`3 d_main` round elements and 31 claims); the join: `S` framed Flock proof items (K22); the
packed PIOP work nonces (`transcript::work_bytes(manifest)` bytes, 227 at `N* = 8` to 556 at
`2^15`); the opening framed; no trailing bytes. `from_bytes` derives every count from `N` and
the relation byte and rejects non-canonical limbs. PIOP bytes (events 9 to 12):
`16 (76 + 11 d_main)` under H-u and H-exact (8 + 31 and 7 + 32 stage-2 and stage-3 claims),
`16 (81 + 11 d_main)` under H-FIPS; 5,616 B at `N = 2^15` under H-u. `AggregateProof::size()`
(`ProofSize`) splits the serialized length into header, `c~`, Akita commitments, Keccak-side
items (event 2), final vectors, Keccak-side shares and proofs (event 6 and the join), lookup,
PIOP, work nonces and opening, framing included.

## 11. Drivers and hooks

`prover::prove(inputs, bridge, keccak, flock)` derives every witness (strict FIPS 204
verification, `main_tables`) and calls `prove_with_tables`, which runs events 0 to 13: the
round-1 commitments and the Keccak commitments (`KeccakSide::commitments`) concurrently
(`pcs::overlap`, Section 3; absorbed in event order, and each step's time is its own
closure's, so the two overlap in wall time), the bridge seed, the matrices
(`BridgeDerivation::matrices(sd)`, a `Cow` so fixed matrices are not copied), round 2, the
Flock seeds, the final vectors and the share items. Then the fork (deviation K22): the Flock
proofs (`KeccakSide::prove`) run on their own rayon pool (`flock_pool_threads`,
`AGG_FLOCK_THREADS`, default 3/8 of the global pool, rounded up) while the parity tables `y^(1)`, the
lookup, `tau`, the pre-stage claims and the three stages run on the global pool, with the
weight tables built between the stages (`ProverTables::set_tau/set_r1/set_r2`, each dropping
the previous weights); `AGG_FLOCK_THREADS=0` (also on a rayon worker or with a one-thread
pool) proves the Flock side first, with the parity tables built beside it, then the PIOP.
The join absorbs the Flock proofs; the PCS session and the opening follow.
`prove_with_tables_using` takes the Akita prover setup built beforehand (it depends on the
record count and the relation only). `ProverOutput::timings` (`ProveTimings`) has the wall
time of each step. `ProveHooks` (adversarial tests only) selects the stages' unchecked
provers and may change the honest round-2 tables before they are committed.

`verifier::verify_timed(records, bytes, bridge, keccak, flock, relation, cache)` returns a
`VerifyReport`: the outcome (`VerifiedAggregate`, or the first failing check as a
`VerifyError`) and `VerifyTimings`, one entry per step of spec Section 11;
`verifier::verify` returns the outcome only. Steps:

| Timing | Spec 11 | Work |
| --- | --- | --- |
| `parse` | 1 | wire parse (all shapes from `N` and the relation), the header's relation against the caller's, manifest, PCS verifier from the plan's setup seed |
| `public_data` | 2 | `StreamedPublic::prepare`: `pkDecode`, `tr` per distinct key (cached keys reuse theirs), `mu`, `SampleInBall`; the record digest |
| `transcript` | 3 | events 0 to 6: commitments parsed (coefficient counts from the plan) and absorbed, `sd`, `fs_sigma`, final vectors and share items |
| `matrices` | 3 | `BridgeDerivation::matrices(sd)`; `SeededMatrices` derives them as in spec 9.1 (`agg_bridge::seed`) |
| `keccak` | 3 | `KeccakVerifier::verify` per segment (`SegmentContext`: commitment item, `fs_sigma`, `sd`, `mu_i`, `c~_i`, `Y_sigma`, share item, the event-6 record digest, proof item, matrices; deviation P11); the proof items are absorbed at the join |
| `lookup` | 4 | the lookup (event 7) and the multiplicity-digit claims |
| `prestage` | 4 | `tau`, pre-stage claims, `Y(tau_3)` |
| `stage1`, `stage2` | 5, 6 | rounds and terminal checks; `w_2, w_3` at `r^(1)`, `w_1` at `r^(2)` (bridge weights, Section 8), the leaf predicates |
| `stage3_rounds` | 6 | `stage3::verify_streamed`: gamma, rounds, terminal claims, with the ring claim `T(r^(2))` deferred (deviation P10) |
| `public_t`, `public_a` | 6 | `StreamedPublic::pass`: `T(r^(2))` over the records (`T_i` recomputed from the sparse `c_i`), `w_A(r^(3))` over the distinct keys (`A` expanded and inverse-NTT'd, or cached; `Lambda` products) |
| `stage3_terminal` | 7 | the batched stage-3 terminal with `gamma_ring T(r^(2)) prod_k r^(3)_k` added, the links and the stack reductions |
| `bits` | 7 | `KeccakVerifier::check_bits` (`BitsContext`: event-2 items, `r^(3)`, the six claims `b_t(r^(3))`); a no-op except in mode B |
| `opening` | 7 | the batched Akita opening of the 38 claims (H-u) at the six group points |

A rejected run stops at its failing step (later timings stay zero); a trailing segment that
holds padded records only (`N` between `2^14` and `3 2^13`) has an empty record range.
Before step `parse` the verifier refuses an unsound configuration
(`verifier::check_configuration`, `VerifyError::Configuration`): a `KeccakVerifier` whose
`profile()` differs from the requested Flock profile, mode A (`Fast`, `Slim`, `Slim8`) without
`BridgeDerivation::seed_derived()` (only `SeededMatrices`), and profile 0 outside builds
with the test hooks. The test stand-ins `FixedMatrices` (matrices that ignore `sd`) and
`NoKeccak`/`NoKeccakVerifier` (profile 0) exist only in unit tests and with the
`test-hooks` feature (deviation P4).

## 12. Mode B (`mode_b`)

Mode B is the Keccak-free aggregate (spec Section 14): the aggregate is
the proof bytes, which carry `c~_i` and, as the event-2 item of segment `sigma`, the
`w1Encode(w_1'_i)` strings (768 bytes each, record order) of the segment's live records;
the event-6 share items and the join's proof items are empty; event 0 binds Flock profile
id 3 (`PublicBits`). Layout and the extra bit check: deviation P9.

- Prover: `prove_mode_b(inputs, bridge)` builds the round-1 tables, publishes the committed
  bits (`PublicBits::from_tables`, i.e. `w_1' = sum_t 2^t b_t` per position, `w1Encode` per
  record), and runs `prove_with_tables` with `PublicBits` as the Keccak side.
- Verifier: `verify_mode_b(records, aggregate, bridge, relation, cache)` is `verify_timed`
  with `PublicBitsVerifier`. Step `keccak` checks per segment: 768 bytes per live record and
  an empty proof item; every 6-bit value below 44; `c~_i = SHAKE256(mu_i || w1Encode)` (7
  Keccak-f per record); and the projection predicate `Y_sigma = M_3 M_2 M_1 b'` over GF(2)
  (`segment_projection`: the six lanes share the matrices, so the 6-bit value is XORed as a
  lane mask, 14 byte XORs per nonzero position at level 1, parallel over level-1 blocks).
  Step `bits` checks `b_t(r^(3))` against `public_bit_evaluations` (per record the 1,024
  equality weights of `r^(3)_pos` bucketed by the 6-bit value, then 32 buckets per lane,
  times `eq(r^(3)_sig, i)`).
- `public_bits(proof, i)` returns record `i`'s 768 bytes.

What an accepted mode-B aggregate establishes: for every live record, (3.3) natively for
the published `w_1'` and (3.1), (3.2) by the PIOP for committed bits equal to the published
ones, under the caller's relation, with the hash relation checked in the clear.

## 13. Tests

`taskset -c 0,1,3-15 cargo test -p agg-piop --release` (101 unit tests; 40 integration tests
in three binaries: `tests/all/` with one module per topic, and `tests/fork_join.rs`,
`tests/single_thread.rs`, which change process-wide state; one diagnostic test is ignored, and
the 1,024-signature round trip is ignored in debug builds):

- unit: manifest shapes, digit layouts, lookup shape and binding bytes under the three
  relations, the manifest against the work plan, committed elements per signature; kernel
  and `Lambda` vs their definitions; `sparse_t`, inverse NTT, `prepare` vs `from_instance`,
  `T(r)` vs the MLE; the square map, the leaves and the digit compositions of the three
  relations; the ring residual; the narrow and wide parity tables; weight tables vs dense
  definitions and vs their verifier evaluations at Boolean and random points, `w_A` vs
  `(A z)` for random `z` and vs its extension at `r3`; transcript registry order, shape
  faults, determinism, the difficulties, work nonces searched, recorded and checked, the
  Flock proofs refused at event 6, the record digest, encoding round trips; the plan
  digests, totals and nonce stream of `pow` against `docs/soundness.md` and
  `docs/h-exact.md`; claim vectors; the three stage specs of every relation (instance
  counts, gamma order, degrees) and the H-FIPS checks against their polynomials; the
  stage-3 deferral (compressed rounds are affine in the claim with slope `prod_k r_k`); wire
  round trips with the claim counts of every relation, the packed final vectors and every
  malformed-buffer rejection; record binding; the lookup (tree roots, multiplicity digits,
  honest stacks at several sizes, the full layout with two challenges, two tables and a
  cross-table value, an out-of-range digit, a wrong multiplicity, tampered messages and
  claims, the dedicated paths against the generic kernel); `StreamedPublic` (`T_i`, the
  split products and key weights against references, deduplication, keys sharing `rho`, and
  the pass against `PublicData::t_evaluation` and `weights::ring_weight_at`, without a
  cache, filling a cache, and from it with no expansion); the seeded and fixed matrices; the
  verifier's configuration policy; mode B: `segment_projection` against the GF(2) side of
  the bridge, `public_bit_evaluations` against the extensions of the bit lanes,
  `PublicBitsVerifier` on shape, range, hash and projection faults.
- kernel (`src/kernel/tests.rs`, `coded.rs`, `field.rs`): every evaluation strategy
  (automatic; no coding; coded per item; quartet masses forced; pair masses only; quartet
  tables forced, with and without Taylor tables; pair fold at level 2; without the claim
  recovery, also with quartet tables) against an independent reference (full `K` tables,
  the round polynomial from its definition at `d + 1` points over full equality tables,
  interpolated), bit for bit in every message and terminal value and with the terminal
  equation, on an equality group with every `Term` kind, a weight group (field, `i32` and
  materialized lanes, weighted sums, a field-field product), a group with wide cells, an
  `i32` lane and cross-column products, the two-round prefix, and domains of 0 to 4
  variables with and without lifted rounds; the classes of digit columns under the square,
  leaf and identity maps; pattern masses against the serial sums in both accumulation
  layouts; quartet and Taylor tables against the folded lanes and the shifted polynomial;
  the compiled pieces against the terms; compressed-message round trips and rejections; the
  leaf polynomials on their alphabets; the dedicated layer groups against `Group`; the fast
  and integer products against `*`; split and plain unreduced sums against nested sums;
  Taylor shifts.
- `tests/all/stages.rs` (the lookup and the stages without commitments): at 256 signatures
  (200 own keys, 56 on 3 shared keys) the claims equal the extensions of the committed and
  virtual tables, and a changed round message, terminal claim or pre-stage claim is rejected
  at its stage. Tampering (50 records, unchecked prover): main cells looked up outside
  `T_16` (`Z_lo` moved by `2^16` against `Z_hi`), carries outside `T_16` and
  a wrong multiplicity at the lookup, a padding slot of the level-1 stack at stage 3;
  tampered lookup messages, layer terminals, table leaves and digit claims at the lookup,
  a swapped stack leaf pair; swapped column claims at stage 3; values valid only for
  another range (`Z_hi = 2` and `U_hi = 3, -4` at stage 2, an unshifted carry at stage 1,
  `U_lo + 63,079` with `U_hi - 1` at the lookup; `V_hi = 2^16` at stage 3 and `V_lo + 2^16`
  with `V_hi - 1` accepted, since `V_lo, V_hi` are not looked up); a
  16-bit column or `Z_hi` changed inside its range and a flipped bit at stage 3; a
  non-Boolean bit and `w_1' = 44` at stage 1; changed carries of every level, a level-2
  parity and a `Y` bit at stage 1; a padded batch of 10 records round-trips.
- `tests/all/h_exact.rs`, `tests/all/h_fips.rs` (PIOP level): honest round trips with every
  claim equal to its table and the pointwise identities; each relation's tamperings at the
  stated place; for H-exact the value boundaries `z = +-130,994`, `u = +-190,466` with
  synthetic statements (the public `T` moved so that the ring identity holds over the
  integers), also under H-u; for H-FIPS the weight-80 corner records and the encoding round
  trip of the main tables.
- `tests/all/prove_verify.rs`: a full prover run at 256 signatures; the opening claims taken
  from the stage-3 terminal values equal the evaluations of the committed tables at the
  group points; the blob round-trips and verifies from `(pk, M)` with and without cached
  keys; mode A over fixed matrices is refused; a shorter or reordered record list, a changed
  `c~`, other bridge matrices, a changed lookup round or layer-1 value, compensated digit
  claims, a mode-A profile under the no-Keccak stand-in, a changed round message, stage-3
  claim, pre-stage claim or opening byte, a truncated commitment, a non-empty Keccak payload
  in mode B and a truncated blob are rejected; a padded batch of 10 records round-trips.
- `tests/all/mode_b.rs`: `prove_mode_b` / `verify_mode_b` round trips at 64, 256 (200 own keys,
  56 on 3 shared keys) and 1,024 signatures, H-exact at 1, 64 and 256 and H-FIPS round
  trips with the cross-relation rejections; the proof size by part, the prover and verifier
  step timings (uncached, cached) printed; the published bits satisfy
  `c~ = H(mu || w1Encode)`; a cached run expands no key. Rejections at 256: a flipped public
  bit, a public value of 63, a short bit item, a changed `c~`, a changed final-vector bit
  and a non-empty proof item at the Keccak side of segment 0; a changed lookup round at the
  lookup; a changed pre-stage claim, round messages of stages 2 and 3 and a stage-3 claim at
  the stage or the next work nonce; a changed nonce at one site of every class, bad nonce
  padding and length; a plan with a tie site; an opening byte; a reordered or shorter record
  list, another Flock profile, other matrices and a truncated aggregate. Every tampering
  through the full prover (`prove_with_tables`, unchecked stages, honest public bits, 50
  records): a 16-bit `Z_lo` or `V_lo` cell changed inside `T_16` at stage 3; a flipped
  committed bit and `w_1' = 44` in the committed bits at the Keccak side (the final vector no
  longer matches the projection of the public bits); a non-Boolean bit at stage 1; a stored
  `U_lo = 2^15` cannot be committed (Akita: "committed source exceeds the scheduled
  coefficient bound"); base-4 and base-6 digits outside their alphabets at stage 2; a
  level-1 carry and a level-2 parity at stage 1; a final-vector bit at the Keccak side; a
  wrong multiplicity at the lookup. Bit check: with two equal `M_1` columns a prover flips
  `b_0` at two positions and moves `u` by `-/+ 2 gamma_2`; every PIOP check, the projection
  and the opening pass, `verify_mode_b` rejects with `VerifyError::Bits`, and the same
  aggregate is accepted by a verifier without the bit check (deviation P9).
- `tests/all/bridge.rs` (the Fp side of the bridge, `docs/bridge.md` Section 9 item 7): a full
  prover run at 256 signatures with `SeededMatrices` verifies; the round-2 tables built
  through `agg_bridge` equal the `agg-witness` reference, satisfy `agg_bridge::parity::check`,
  and the published `Y` equals both GF(2) composites; the verifier's bridge weights at
  `r^(1)` (`w_2`, `w_{3,t}`, `stage1::BridgeWeightsAt`) and at `r^(2)` (`w_1`) equal the
  extensions of the prover's weight tables at the transcript's points. Tampering through the
  full prover (`FixedMatrices`, unchecked stages): a flipped final-vector bit, a flipped
  level-2 parity, a level-1 and a level-3 carry moved inside `T_16`, and a flipped committed
  bit under the stale round-2 tables are each rejected at the stage-1 terminal.
- `tests/fork_join.rs`: the Flock proofs are absorbed at the join (no PIOP challenge depends
  on them, the PCS session binds them, the side's challengers receive the event-6 record
  digest), and both prover schedules give the same proof.
- `tests/single_thread.rs`: a prover and verifier run with a one-thread rayon pool, from a
  plain thread and from a pool worker.
- `tests/all/projection.rs` (deviation K24): the unbounded-quotient attack (`Delta = 0` in
  `F_p` with `|v| ~ 2^41` at one position) rejected by the lookup of `k_1`; `k_0 + 1` and `v`
  moved with `k` kept at stage 3; `k_2 = 2` at stage 2; a changed claim `c`; the honest abort
  on `|k| >= 2^33`.
- `tests/all/commit_open.rs` (the seven groups commit and open in one batch, also for a padded
  batch at independent points), `tests/all/source_tables.rs` (shapes, alphabets,
  multiplicities, heavy rows, malformed tables refused before Akita, the H-exact `z` form),
  `tests/all/virtual_checker.rs` (the integer relation on the Boolean domain, single-cell
  mutations).

`agg-bench` (`taskset -c 0,1,3-15 cargo test -p agg-bench --release`): medians, batch
inputs, `VmHWM`/`VmRSS` parsing, the peak following a 64 MiB allocation and its reset.

## 14. Instance, witness and bridge on the prover's path

Parts of other crates that the prover's speed depends on:

- `agg_witness::ntt::NttPrime`: the 256-point NTTs (about 66 per record in the key
  expansion, `T`, `A z` by CRT and the strict FIPS verification) run on 32-bit lanes with
  Shoup products by the constant zetas (the quotient `floor(zeta 2^32 / p)` is precomputed;
  both primes are below `2^31`, so the remainder lies in `[0, 2p)`), the sums with
  branch-free conditional subtractions, and the pointwise products with a Barrett reduction
  (`floor(2^64 / p)`).
- `agg_witness::derive::derive_with_t`: the witness reuses the instance's `T = c t1 2^d` when
  the signature's `c~` is the instance's (else it recomputes); `public_t` adds `x^p t1` in
  two straight (vectorized) loops.
- `agg_bridge::seed::ColumnHasher`: the bridge matrices (about a million columns, prover and
  verifier) on the bare Keccak-f[1600] from the precomputed cSHAKE256 prefix state, one
  permutation per column; equal to the cSHAKE256 path (test
  `keccak_columns_match_the_cshake_columns`).
