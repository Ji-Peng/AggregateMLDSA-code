# PIOP (`agg-piop`): tables, commitments, sum-check stages, transcript and wire format

**Wire version 1 (lookup layout L16, 2026-10-24):** the range mechanism is the lookup into
`T_16` (`docs/lookup.md`). Section 19 is the current description of the groups, the event-0
bytes, the commit flow, the lookup, the stage tables, the transcript and the wire format;
Sections 2, 3, 7, 9 and 10 below describe wire version 0 (the base-16 digit layout, code
`4247f61`) where Section 19 does not replace them, and Sections 13 to 18 are measurements and
kernel work on that layout.

Part 1 (prover-side tables, commitments, public data, virtual evaluators; spec Sections
2, 4, 5, 7.6), part 2 (the three sum-check stages, the weight tables, the transcript,
the proof wire format, the prover and verifier drivers; spec Sections 6, 7, 10, 11) and
part 3 (the timed, streamed verifier and the mode-B aggregate of milestone M2, `prove_mode_b`
/ `verify_mode_b`; spec Sections 11, 14). Deviations: `docs/deviations.md` entries P1 to P12.

## 1. Modules

| Module | Content | Spec | Aerie counterpart |
| --- | --- | --- | --- |
| `manifest` | `CommitmentManifest`: the four groups (role, round, family, logical and physical variables, polynomial order and alphabets), digit columns, event-0 binding bytes | 4.1, 4.3, 10 | `aerie-falcon/src/protocol/manifest.rs` |
| `source_tables` | `MainTables` (round 1, `G_main`), `BridgeGroupTables` (round 2, `G_1, G_2, G_3` and public `Y`, from `agg_bridge::parity` and `agg_bridge::carry` at `Shape::PRODUCTION`), `SourceTables` view | 4.1, 4.2, 9.2 | `aerie-prover/src/protocol/source_tables.rs` |
| `pcs` | `group_polynomials`, `TableCommitter` (two rounds, round 2 committed concurrently, batched opening from given claims `open_claims` or table evaluations `open`), `opening_claims`, `suffix_points`, `verifier_for` | 4.1, 7.5, 10 | `aerie-prover/src/protocol/pcs/group.rs` |
| `public_data` | `PublicData` (expanded keys, `mu`, `c~`, `T`), `KeyCache`, `t_evaluation`, `a_row_combination`, `negacyclic_kernel` and the allocation-free `negacyclic_kernel_into` (7.8), `eq_correlation` (Aerie (8.8)-(8.10)) | 2, 7.6 | `aerie-falcon/src/protocol/ring/functional.rs` |
| `virtual_polys` | `w_1'`, (4.1) compositions, ring residual, `y^(1)` parities, `P_Z, A^lo, A^hi`, `MainValues`/`CarryValues` | 4.3, 5 | |
| `kernel` (`kernel/`: `mod.rs`, `plan.rs`, `coded.rs`, `field.rs`, `tests.rs`) | `Source`/`Lane` (the narrow input tables), `Term` (summand shapes), `Group` (one local domain, optional split equality factor, gamma-weighted instances; coded columns, pattern masses, pair/quartet/Taylor tables, one dense table from the third fold), compressed messages, lifted inactive rounds | 7.1 | `aerie-prover/src/protocol/kernels/`, `algebra.rs` |
| `lookup` | LogUp-GKR range argument: stacks, multiplicities and their digits, fraction trees, sessions on the kernel, `prove_in`/`verify_in` on a `LookupTranscript` (Section 19) | 4.3, 7.0 | |
| `range` | the digit map `z -> z(z+1)`; the two leaf-checked digits `Z_hi`, `U_hi` | 4.3 | `aerie-falcon/src/protocol/relations/quartic.rs` |
| `stage` | `StageSpec` (declarative instance table: groups, lanes, equality points, gamma indices), `prove_stage`, `replay_stage`, `check_terminal` | 7.1 | `aerie-falcon/src/protocol/stage.rs` |
| `stage1`, `stage2`, `stage3` | the instance tables of Section 7.3 to 7.5, claimed sums, terminal claims, `prove`/`verify` | 7.3-7.5 | `stage1.rs`, `stage2.rs`, `stage3.rs` of both Aerie crates |
| `weights` | `w_1, w_2, w_{3,t}` tables and verifier evaluations (wrappers of `agg_bridge::weights` at the production shape); `w_A` table and (7.10) | 7.6 | `ring/functional.rs`, `stage1/projection_link.rs` |
| `claims` | the claim vectors between stages and their wire order | 7.2-7.5 | |
| `tables` | `ProverTables`: borrowed `i8` columns, `y^(1)` as `Parity1` (`i8` when every value fits, else `i32`), the weight tables when known; `LaneId -> Lane` | 7 | |
| `transcript` | Spongefish `Blake2b512` stream, event registry, samplers, `pcs_session` | 10 | `aerie-falcon/src/protocol/transcript.rs` |
| `binding` | public-record digest (16-KiB-leaf cSHAKE256 tree), instance record of event 0 | 2, 10 | `public_records_binding.rs` |
| `wire` | `AggregateProof`, byte format, packed final vectors | 10 | `wire.rs` |
| `opening` | opening claims at `r^(3)` and its suffixes, commitment parsing | 7.5, 10 | `pcs/verifier.rs` |
| `prover`, `verifier` | the drivers in event order with per-step timings (`ProveTimings`, `VerifyTimings`); hooks `BridgeDerivation`, `KeccakSide`/`KeccakVerifier` | 10, 11 | `transport.rs` |
| `public_stream` | `StreamedPublic`: the verifier's light per-record data (step 2) and the one pass of step 6 (`T(r^(2))`, `w_A(r^(3))`) | 2, 11 | |
| `mode_b` | mode B: public `w_1'` bits as the Keccak side (`PublicBits`, `PublicBitsVerifier`), the GF(2) projection, the bit check, `prove_mode_b`, `verify_mode_b` | 11, 14 | |

`agg-pcs-config` gained `PcsProver::commit_polynomials` (commit already materialized
polynomials); `commit_group` now delegates to it.

## 2. Manifest and the event-0 bytes

Group and polynomial order are the table order of spec Section 4.1 and `docs/witness.md`
W3: `G_main = z_0..z_4, u_0..u_4, v_0..v_6, b_0..b_5`; `G_1 = kappa1_{t,k}` (t-major);
`G_2 = y2_0..y2_5, kappa2_{t,k}`; `G_3 = kappa3_0, kappa3_1`. Counts: 67 polynomials, 55
digits (40 base 16, 2 base 8, 13 base 4), 23,865 committed elements per signature at
`N = 2^15`.

`CommitmentManifest::binding_bytes` (all integers little-endian):

| Field | Encoding |
| --- | --- |
| wire version | `u16` = 0 |
| parameter set | `u8` = `0x44` |
| relation flag | `u8` = 0 (H) |
| `N`, `N*` | `u64`, `u64` |
| digit layout id | `u8` = 0 (spec Section 4.3) |
| group count | `u8` = 4 |
| per group, wire order | role `u8` (0..3), round `u8`, family `u8` (0 = width 4), logical vars `u8`, physical vars `u8`, polynomial count `u8`, then one alphabet byte per polynomial (digit base, 0 for a Boolean) |
| catalog row id | `u16` length, ASCII (`mldsa-44-mode-a-batch-<N*>-level3-final`) |
| bridge parameters | for levels 1..3: `J_t` `u64`, `m_t` `u64`, `k_t` `u8` |

The instance record of event 0 (`binding::instance_record`) is these bytes followed by the
32-byte public-record digest and the Flock profile id (`u8`: 0 none, 1 `Fast`, 2 LaBinius).
The record digest: the stream `LE64(N) || (pk || LE64(|M|) || M || c~)*` is cut into
16-KiB blocks, each hashed by cSHAKE256 under `agg-mldsa/v1/record-leaf`, and the root is
cSHAKE256 of `LE64(blocks) || leaves` under `agg-mldsa/v1/public-records-binding`.

## 3. Commit flow

1. `MainTables::build(manifest, instance, sigs)` (strict derivation) or `from_batch`;
   both check shape and honest alphabets (encoder self-check, not a soundness check).
2. `TableCommitter::new(manifest)`: work plan from the catalog, `manifest.matches_plan`,
   seed-derived Akita setup. `commit_round1` gives the event-1 commitment.
3. After the bridge seed: `BridgeGroupTables::build(manifest, main, matrices)` (honest abort
   on a row weight above the cap), `commit_round2` gives `G_1, G_2, G_3` (event 4); the three
   commits run concurrently (each is transcript-free and the final group is committed
   against the plan's precommitted profiles), the prover absorbs them in wire order.
4. After stage 3: `open_claims(commitments, claims, session)` proves one batched opening of
   the 67 claims at the four suffix points of `r^(3)`, with the claims taken from the
   stage-3 terminal values (the multilinear extensions of the committed tables; a debug
   assertion and `tests/prove_verify.rs` compare them with `opening_claims`, which evaluates
   every table again; `open` still does that). `verifier_for(manifest)` verifies from the
   setup seed alone; `opening::parse_commitment` reads a commitment with the coefficient
   count of the plan's profile, never from the proof.

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

Affine in committed evaluations: `w1_from_bits`, `compose_digits`, `residual_high`,
`MainValues`, `CarryValues`. Boolean-domain materializations: `w1_table`, `compose_table`,
`residual_table` (diagnostic), `parity1_tables` (`y^(1)_t = sum M_1 b_t - 2 kappa^(1)_t`),
`digit_tables`, `leaf`.

## 6. Sum-check kernel

Every stage is one batched ordinary sum-check of `d_main` rounds (Aerie 10.3): a round
message is the degree-`d` round polynomial without its linear coefficient (`d` elements;
`d = 4, 5, 3`), recovered by the verifier from `g(0) + g(1) = claim`. Instances on a smaller
domain `D_t` (always a suffix of the main variables, spec 7.1) are inactive in the
leading `d_main - d_t` rounds, where their message is the constant `claim / 2`, and are
lifted by `2^-(d_main - d_t)` at the terminal.

`kernel::Group` holds the lanes of one local domain. Every lane starts as a narrow
`Source` (`i8` cells with a per-cell integer map, `i32` parities, or a materialized `K`
weight). Within a group the instances share one equality factor, kept as Akita's
`GruenSplitEq` (two tables of about `2^(d/2)` entries; the full `eq` table is never
materialized), and their gamma-weighted summands (`Term`s: products, `v(v-1)`, the bit-range
expression, leaf predicates `Q(P)`, links `Z(Z+1)`, affine combinations with an optional
weight lane) are evaluated at `d` or `d + 1` interpolation points per pair of cells and
interpolated to coefficients (Newton), then multiplied by the equality line of the current
variable (`gruen_mul`) and the lift. Groups without an equality factor carry their weight
as a lane. A group's claim is computed from its own tables (directly for lifted groups,
from `g(0) + g(1)` of round 0 otherwise); the stage driver checks the sum of the group
claims against the gamma-weighted claimed sums at the end (prover self-check; the
test-only `prove_unchecked` paths skip it so tampered witnesses reach the verifier).

How a round is evaluated (Sections 16.2 and 18; all paths compute the same polynomial, so
the proof bytes do not depend on them; `kernel/mod.rs` holds the schedule, `kernel/plan.rs`
the compiled pieces, `kernel/coded.rs` the codes and pattern masses, `kernel/field.rs` the
field helpers):

- *Compiled plan.* `Group::new` compiles the gamma-weighted terms into one polynomial per
  lane (every single-lane term: `Bool`, `Link`, `Leaf`, one-lane `Product`, the lanes of a
  weightless `Affine`; constants go to one per-cell constant), bilinear products
  `c v_a v_b`, weighted sums `w (sum c_l v_l + c)` (every `Affine` with a weight and every
  product of a field lane with another lane, merged per weight lane), and the residual
  generic terms (`RangeW1`, products of three lanes). Everything is evaluated in coefficient
  form at `v = a + X d` (Taylor coefficients `P^(i)(a) / i!` times `d^i`). A lane read only
  by its linear polynomial, by one bilinear product or once in one weighted sum absorbs its
  coefficient at its first fold (the terminal values are unscaled before they are
  reported).
- *Coded columns.* The `i8` lanes over one column (by pointer) code the column by the
  classes of its present cells under the tuple of their maps, if there are at most 16: 8 for a
  digit read through `z (z + 1)` (`P_Z`, `A^lo`, `A^hi`), 16 for a digit under the identity
  (the stage-3 links), 2 for a bit, and one byte of pair code per pair of cells
  (`class(2k) n + class(2k + 1)`). A tampered cell (for example a digit 9 under the identity:
  17 classes) and the `i32` parity fallback leave a lane uncoded. Pieces on coded lanes are
  *bucketed*: per column (polynomials, same-column products), or per set of columns for a
  generic term whose joint pair space is at most `2^12` (`RangeW1` on four bit columns: 256);
  the other pieces (weighted sums, uncoded lanes, cross-column digit products) are *direct*
  and evaluated per item.
- *Level 0* (first active round). If the group has at least two active rounds and the
  quartets outnumber the largest quartet space by `BUCKET_MIN_RATIO = 4` (`4 * 8^4` for an
  8-class column: from `N = 1,024` on the main domain; `4 * 16^4`: from 1,024 for the
  stage-3 links), one pass accumulates per segment the *quartet masses* (the round-1
  equality weight `eq(tau_{2..}, q)`, or one without an equality factor, per quartet
  pattern); round 0 is `sum_p (l_0 H_0[p] + l_1 H_1[p]) F(p)` over the pair patterns with the
  two marginals `H_0, H_1` and `l_0 = 1 - tau_1`, `l_1 = tau_1` (Aerie's bivariate prefix),
  and the masses are kept for round 1. Otherwise pair masses with the round-0 weights.
  Narrow segment sets (at most `2^16` entries in all) accumulate in one table per item chunk
  (Aerie's `chunked_bucket_sums`); wide ones (the 16-class stage-3 columns, 17 MiB) one table
  per segment, each task owning whole segments with the item weights computed once. Direct
  pieces run one pass over the pairs; weighted sums read per-byte tables `c_l map_l[byte]`
  of their byte lanes (additions, no products).
- *Bind 0.* Coded lanes become pair tables (`n^2` entries, scaled; no storage per cell);
  uncoded lanes are folded into `K` tables.
- *Level 1.* With quartet masses, every quartet pattern once with the pair tables of its
  two pairs; for a column read only by single-lane polynomials, by moments
  `m_j[left] = sum_right M[left, right] L[right]^j` and the Taylor coefficients at
  `L[left]` (one unreduced product per pattern and power). Direct pieces, and every piece
  without masses, per item.
- *Bind 1.* Coded lanes of columns with at most 4,096 quartet patterns (8 classes) become
  quartet tables (when the octets outnumber the patterns), with Taylor tables of their
  polynomials of degree 2 to 4 (coefficients 1 to 4 per entry, one 64-byte line); the
  others keep the pair table and the second challenge.
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
  `K` multiplication (`kernel::field::fm`, about 20% faster than `*` here).
- `AGG_STAGE_TRACE=1` prints per group the message and fold time summed over the rounds,
  the per-round wall time, and the weight-table build times (`stage::prove_stage`,
  `prover::prove_with_tables`).

Memory at `N = 2^15` (estimates, 16 bytes per `K` element): the first `K` tables appear
after the third fold at an eighth of the domain, so stage 1's main-domain group holds
`34 * 2^22 * 16 B = 2.3 GB` (one row-major table), stage 2 `17 * 2^22 * 16 B = 1.1 GB`,
stage 3 `23 * 2^22 * 16 B = 1.5 GB` (half the figures of `50fc538`); before that the coded
lanes cost one byte of pair code per pair of cells and group (`2^24` bytes per column at
32K), the quartet masses at most `17 * 1 MiB` (stage 3) and the quartet and Taylor tables
`80 KiB` per lane. The weight tables `w_1`, `w_A` are `512 MB` each. The bridge domains are at
most `2^19` cells.

## 7. Stage table

Gamma indices are positions in the stage's `gamma^(s)` vector; "lift" is the number of
inactive leading rounds (`d_main - d_t`). Digit order is that of spec Section 4.1 (`range::digits`):
`z_0..z_4, u_0..u_4, v_0..v_6` (main), `kappa1_{t,k}` (`D_1`), `kappa2_{t,k}` (`D_2`),
`kappa3_k` (`D_3`); the 40 base-16 digits in that order are `z_0..z_3, u_0..u_3,
v_0..v_5, kappa1_{t,0}, kappa1_{t,1}, kappa2_{t,0}, kappa2_{t,1}, kappa3_0, kappa3_1`.

Stage 1 (degree 4, `gamma^(1) in K^66`, point `tau`):

| Instance | Gamma | Domain | Equality / weight | Summand | Degree | Claimed sum |
| --- | ---: | --- | --- | --- | ---: | --- |
| `bool_y1[t]` | `t` | `D_1` | `eq(tau_1, .)` | `y1_t (y1_t - 1)` | 3 | 0 |
| `level2[t]` | `6 + t` | `D_1` | lane `w_2` | `w_2 y1_t` | 2 | `y2_t(tau_2) + 2 kappa2_t(tau_2)` |
| `level3` | 12 | `D_2` | lanes `w_{3,t}` | `sum_t w_{3,t} y2_t` | 2 | `Y(tau_3) + 2 kappa3(tau_3)` |
| `bool_y2[t]` | `13 + t` | `D_2` | `eq(tau_2, .)` | `y2_t (y2_t - 1)` | 3 | 0 |
| `bool_b[t]` | `19 + t` | main | `eq(tau, .)` | `b_t (b_t - 1)` | 3 | 0 |
| `range_w1` | 25 | main | `eq(tau, .)` | `b_5 (b_4 + b_3 b_2)` | 4 | 0 |
| `b16_prod[Z]` | `26 + k` | `D_Z` | `eq(tau_Z, .)` | `A_Z^lo A_Z^hi` | 3 | 0 |

Terminal (98, `claims::Stage1Claims`): `y1_t(r1_1)`, `y2_t(r1_2)`, `b_t(r1)`, then
`(A_Z^lo, A_Z^hi)(r1_Z)` per base-16 digit. Pre-stage (event 8, 26 values): `y2_t(tau_2)`,
`kappa2_{t,k}(tau_2)` (t-major), `kappa3_k(tau_3)`.

Stage 2 (degree 5, `gamma^(2) in K^101`, point `r^(1)`; before `gamma^(2)` the 18 claims
`kappa1_{t,k}(r1_1)`):

| Instance | Gamma | Domain | Equality / weight | Summand | Degree | Claimed sum |
| --- | ---: | --- | --- | --- | ---: | --- |
| `def_y1[t]` | `t` | main | lane `w_1` | `w_1 b_t` | 2 | `y1_t(r1_1) + 2 kappa1_t(r1_1)` |
| `leaf16[Z, lo]` | `6 + 2k` | `D_Z` | `eq(r1_Z, .)` | `Q_8(P_Z)` | 5 | `A_Z^lo(r1_Z)` |
| `leaf16[Z, hi]` | `7 + 2k` | `D_Z` | `eq(r1_Z, .)` | `Q_16^hi(P_Z)` | 5 | `A_Z^hi(r1_Z)` |
| `leaf8[Z]` (`u_4, v_6`) | `86, 87` | main | `eq(r1, .)` | `Q_8(P_Z)` | 5 | 0 |
| `leaf4[Z]` (13 base-4 digits) | `88 + m` | `D_Z` | `eq(r1_Z, .)` | `Q_4(P_Z)` | 3 | 0 |

Terminal (61, `Stage2Claims`): `b_t(r2)`, then `P_Z(r2_Z)` per digit.

Stage 3 (degree 3, `gamma^(3) in K^118`, point `r^(2)`):

| Instance | Gamma | Domain | Equality / weight | Summand | Degree | Claimed sum |
| --- | ---: | --- | --- | --- | ---: | --- |
| `ring` | 0 | main | lane `w_A`; and `eq(r2, .)` | `w_A z - eq (2 gamma_2 w_1' + u + q v)` | 2 | `T(r2)` |
| `link[Z]` | `1 + k` | `D_Z` | `eq(r2_Z, .)` | `Z (Z + 1)` | 3 | `P_Z(r2_Z)` |
| `b_t` from `r1` | `56 + t` | main | `eq(r1, .)` | `b_t` | 2 | `b_t(r1)` |
| `b_t` from `r2` | `62 + t` | main | `eq(r2, .)` | `b_t` | 2 | `b_t(r2)` |
| `y2_t` from `tau_2` | `68 + t` | `D_2` | `eq(tau_2, .)` | `y2_t` | 2 | `y2_t(tau_2)` |
| `y2_t` from `r1_2` | `74 + t` | `D_2` | `eq(r1_2, .)` | `y2_t` | 2 | `y2_t(r1_2)` |
| `kappa2_{t,k}` from `tau_2` | `80 + 3t + k` | `D_2` | `eq(tau_2, .)` | `kappa2_{t,k}` | 2 | `kappa2_{t,k}(tau_2)` |
| `kappa3_k` from `tau_3` | `98 + k` | `D_3` | `eq(tau_3, .)` | `kappa3_k` | 2 | `kappa3_k(tau_3)` |
| `kappa1_{t,k}` from `r1_1` | `100 + 3t + k` | `D_1` | `eq(r1_1, .)` | `kappa1_{t,k}` | 2 | `kappa1_{t,k}(r1_1)` |

The ring instance is one gamma on two groups (the `w_A` lane group and the `eq(r2)` group);
`z, u, v` enter as the affine (4.1) combinations of their digit lanes and `w_1'` as
`sum 2^t b_t`. Terminal (67, `Stage3Claims`): every committed polynomial at `r^(3)`
restricted to its group, in manifest order; these are the opening claims.

Prover groups per stage (lanes): stage 1: main `eq(tau)` {6 bits, 14 x (`A^lo`, `A^hi`)},
`D_1` `eq(tau_1)` {6 `y1`, 12 x (`A^lo`, `A^hi`)}, `D_1` {`w_2`, 6 `y1`}, `D_2` `eq(tau_2)`
{6 `y2`, 12 x (`A^lo`, `A^hi`)}, `D_2` {6 `w_3`, 6 `y2`}, `D_3` `eq(tau_3)` {2 x (`A^lo`,
`A^hi`)}; stage 2: main {`w_1`, 6 bits}, one `eq(r1_Z)` group per domain with the `P_Z`
lanes (17, 18, 18, 2); stage 3: main `eq(r2)` {23 columns}, main {`w_A`, 5 `z` digits},
main `eq(r1)` {6 bits}, `D_1` `eq(r2_1)` and `eq(r1_1)` {18 `kappa1` digits each}, `D_2`
`eq(r2_2)` {18 `kappa2`}, `eq(tau_2)` {6 `y2`, 18 `kappa2`}, `eq(r1_2)` {6 `y2`}, `D_3`
`eq(r2_3)` and `eq(tau_3)` {2 `kappa3` digits each}.

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

## 9. Transcript (`transcript`)

Spongefish `Blake2b512` (`akita_transcript::TranscriptSponge`), tag
`agg-mldsa/v1/piop/blake2b` zero-padded to 64 bytes, session `B("agg-mldsa/v1")`, instance
`B(cSHAKE256(LE64(len) || record))` under `agg-mldsa/v1/instance`. Prover and verifier run
the same state machine (every payload is a public message; no Spongefish argument string).
Elements of `K` are rejection-sampled per 64-bit limb from eight squeezed bytes.

| # | Site | Kind | Atoms | Payload |
| ---: | --- | --- | ---: | --- |
| 0 | `agg-mldsa/v1/instance` | public | 1 | the instance record |
| 1 | `agg-mldsa/v1/commit/main` | proof | 1 | Akita native encoding of `G_main` |
| 2 | `agg-mldsa/v1/commit/flock` | proof | `S` | `S` framed (`LE64` length) Flock commitments |
| 3 | `agg-mldsa/v1/bridge/seed` | challenge | 1 | 32 squeezed bytes `sd` |
| 4 | `agg-mldsa/v1/commit/level1`, `level2`, `level3` | proof | 1 each | the round-2 commitments |
| 5 | `agg-mldsa/v1/flock/seeds` | challenge | `S` | `32 S` squeezed bytes `fs_sigma` |
| 6 | `agg-mldsa/v1/flock/final-vectors`; `agg-mldsa/v1/flock/proofs` | proof | `S`; `S` | `512 S` bytes `Y_sigma`; `S` framed Flock proofs |
| 7 | `agg-mldsa/v1/tau` | challenge | `d_main` | `tau` |
| 8 | `agg-mldsa/v1/prestage/claims` | proof | 26 | pre-stage claims |
| 9 | `agg-mldsa/v1/stage1/gamma`; `.../round/{r}/message`, `.../challenge`; `.../terminal` | challenge; proof, challenge; proof | 66; 4, 1; 98 | |
| 10 | `agg-mldsa/v1/stage2/kappa1`; `stage2/gamma`; rounds; terminal | proof; challenge; (5, 1); proof | 18; 101; ; 61 | |
| 11 | `agg-mldsa/v1/stage3/gamma`; rounds; terminal | challenge; (3, 1); proof | 118; ; 67 | |
| 12 | `agg-mldsa/v1/pcs/session` | export | 1 | 32 bytes squeezed from a sponge clone |

Grinding (wire version 2, `docs/soundness.md` Section 6): every field-challenge event
carries a difficulty from Aerie's assigned bounds (`Registry::work_sites`: `X` 0, a merge 1, a
GKR batching vector `bit_length(2 active)`, a GKR round 2, `tau` `bit_length(d_main)`, the
stage batching vectors 5, 4, 6, the stage rounds 3, 3, 2). With `g > 0` the transcript absorbs
`LE32(nonce)` (`nonce < 2^(g + 7)`), squeezes a 32-byte predicate whose low `g` bits must be
zero and then the challenge; the prover searches the smallest passing nonce and packs it into
the work stream (`Transcript::new`, `work_stream`), the verifier reads and checks it
(`Transcript::replay`, `TranscriptError::Work`), the stream's exact length and zero padding
checked up front and its exhaustion at `ensure_complete`. The seeds and the PCS export carry
no work. The plan (`pow::PowPlan`: the Keccak side's tie sites, then these sites) is hashed
into the instance record of event 0 (`binding::instance_record`).

## 10. Wire format (`wire::AggregateProof`)

`LE16` version 2, `u8` flags (0 = H), `LE64 N`; `32 N` bytes `c~`; event 1 framed
(`LE32` length); event 2: `LE32 S` then `S` framed items; event 4: three framed items;
event 6: `512 S` bytes then `S` framed items; events 8 to 11 as raw atom encodings of
16 bytes per element (26 pre-stage claims; `4 d_main` stage-1 round elements and 98
claims; 18 `kappa1` claims, `5 d_main` round elements and 61 claims; `3 d_main` round
elements and 67 claims); the packed PIOP work nonces (`transcript::work_bytes(manifest)` bytes, 227 at `N* = 8` to 556 at `2^15`); the opening framed; no trailing bytes. `from_bytes` derives every count from `N` and rejects
non-canonical limbs. PIOP bytes: `16 (270 + 12 d_main)` = 7,776 at `N = 256`, 8,640 at
`N = 2^15`. `AggregateProof::size()` (`ProofSize`) splits the serialized length into header,
`c~`, Akita commitments, Keccak-side items (event 2), final vectors, Keccak-side proofs,
PIOP and opening, framing included.

## 11. Drivers and hooks

`prover::prove(inputs, bridge, keccak, flock)` derives every witness (strict FIPS 204
verification, `main_tables`) and calls `prove_with_tables`, which runs events 0 to 12:
round-1 commitment and the Keccak commitments (`KeccakSide::commitments`) concurrently
(`rayon::join`; `keccak` is `KeccakSide + Sync`; absorbed in event order, and each step's time
is its own closure's, so the two overlap in wall time), the bridge seed, the
matrices (`BridgeDerivation::matrices(sd)`, a `Cow` so fixed matrices are not copied),
round 2, the Flock seeds, the final vectors and Flock proofs (`KeccakSide::prove`), `tau`,
the pre-stage claims, the three stages with the weight tables built between them
(`ProverTables::set_tau/set_r1/set_r2`, each dropping the previous weights), the PCS
session and the opening. `ProverOutput::timings` (`ProveTimings`) has the wall time of each
step. `ProveHooks` (adversarial tests only) selects the stages' unchecked provers and may
change the honest round-2 tables before they are committed.

`verifier::verify_timed(records, bytes, bridge, keccak, flock, cache)` returns a
`VerifyReport`: the outcome (`VerifiedAggregate`, or the first failing check as a
`VerifyError`) and `VerifyTimings`, one entry per step of spec Section 11;
`verifier::verify` returns the outcome only. Steps:

| Timing | Spec 11 | Work |
| --- | --- | --- |
| `parse` | 1 | wire parse (all shapes from `N`), manifest, PCS verifier from the plan's setup seed |
| `public_data` | 2 | `StreamedPublic::prepare`: `pkDecode`, `tr` per distinct key (cached keys reuse theirs), `mu`, `SampleInBall`; the record digest |
| `transcript` | 3 | events 0 to 6: commitments parsed (coefficient counts from the plan) and absorbed, `sd`, `fs_sigma`, final vectors, Keccak-side proofs |
| `matrices` | 3 | `BridgeDerivation::matrices(sd)`; `SeededMatrices` derives them as in spec 9.1 (`agg_bridge::seed`) |
| `keccak` | 3 | `KeccakVerifier::verify` per segment (`SegmentContext`: commitment item, `fs_sigma`, `sd`, `mu_i`, `c~_i`, `Y_sigma`, proof item, matrices; deviation P11) |
| `prestage` | 4 | `tau`, pre-stage claims, `Y(tau_3)` |
| `stage1`, `stage2` | 5, 6 | rounds and terminal checks; `w_2, w_3` at `r^(1)`, `w_1` at `r^(2)` (bridge weights, Section 8) |
| `stage3_rounds` | 6 | `stage3::verify_streamed`: gamma, rounds, terminal claims, with the ring claim `T(r^(2))` deferred (deviation P10) |
| `public_t`, `public_a` | 6 | `StreamedPublic::pass`: `T(r^(2))` over the records (`T_i` recomputed from the sparse `c_i`), `w_A(r^(3))` over the distinct keys (`A` expanded and inverse-NTT'd, or cached; `Lambda` products) |
| `stage3_terminal` | 7 | the batched stage-3 terminal with `gamma_ring T(r^(2)) prod_k r^(3)_k` added |
| `bits` | 7 | `KeccakVerifier::check_bits` (`BitsContext`: event-2 items, `r^(3)`, the six claims `b_t(r^(3))`); a no-op except in mode B |
| `opening` | 7 | the batched Akita opening of the 67 claims at the four group points |

A rejected run stops at its failing step (later timings stay zero); a trailing segment that
holds padded records only (`N` between `2^14` and `3 2^13`) has an empty record range.
Before step `parse` the verifier refuses an unsound configuration
(`verifier::check_configuration`, `VerifyError::Configuration`): a `KeccakVerifier` whose
`profile()` differs from the requested Flock profile, mode A (`Fast`, `LaBinius`) without
`BridgeDerivation::seed_derived()` (only `SeededMatrices`), and profile 0 outside builds
with the test hooks. The test stand-ins `FixedMatrices` (matrices that ignore `sd`) and
`NoKeccak`/`NoKeccakVerifier` (profile 0) exist only in unit tests and with the
`test-hooks` feature (deviation P4).

## 12. Mode B (`mode_b`)

Mode B is the Keccak-free aggregate of milestone M2 (spec Section 14): the aggregate is
the proof bytes, which carry `c~_i` and, as the event-2 item of segment `sigma`, the
`w1Encode(w_1'_i)` strings (768 bytes each, record order) of the segment's live records;
the event-6 proof items are empty; event 0 binds Flock profile id 3 (`PublicBits`). Layout
and the extra bit check: deviation P9.

- Prover: `prove_mode_b(inputs, bridge)` builds `G_main`, publishes the committed bits
  (`PublicBits::from_tables`, i.e. `w_1' = sum_t 2^t b_t` per position, `w1Encode` per
  record), and runs `prove_with_tables` with `PublicBits` as the Keccak side.
- Verifier: `verify_mode_b(records, aggregate, bridge, cache)` is `verify_timed` with
  `PublicBitsVerifier`. Step `keccak` checks per segment: 768 bytes per live record and an
  empty proof item; every 6-bit value below 44; `c~_i = SHAKE256(mu_i || w1Encode)` (7
  Keccak-f per record); and the projection predicate `Y_sigma = M_3 M_2 M_1 b'` over GF(2)
  (`segment_projection`: the six lanes share the matrices, so the 6-bit value is XORed as a
  lane mask, 14 byte XORs per nonzero position at level 1, parallel over level-1 blocks).
  Step `bits` checks `b_t(r^(3))` against `public_bit_evaluations` (per record the 1,024
  equality weights of `r^(3)_pos` bucketed by the 6-bit value, then 32 buckets per lane,
  times `eq(r^(3)_sig, i)`).
- `public_bits(proof, i)` returns record `i`'s 768 bytes.

What an accepted mode-B aggregate establishes: for every live record, (3.3) natively for
the published `w_1'` and (3.1), (3.2) by the PIOP for committed bits equal to the published
ones (relation `H(2^17, 2^18)` with the hash relation checked in the clear).

## 13. Measurements

This machine (i7-11700K, turbo boost and hyper-threading off), 2026-10-24,
`taskset -c 0,1,3-15` (7 cores; CPU 2 busy with another task, load average about 6 at the
start), release build without the `avx512` feature (Aerie's runtime dispatch selected
AVX-512), working tree on top of `6434fef` (dirty: this stage, uncommitted).
`scripts/bench.sh --bench mode_b` (`crates/agg-bench/benches/mode_b.rs`): per size one
prover process (3 runs) and one verifier process (7 runs uncached, then 7 with every key
cached); `N - N/32` records under their own keys, `N/32` under 3 shared keys; fixed test
matrices (`FixedMatrices`, built outside the timed region; deviation P4). The benchmark now
derives the matrices from `sd` (`SeededMatrices`), which adds the derivation (about
150 ms on 7 threads, 62 MB, deviation P4) to the prover and to every verification, so the
table below predates that change. Times are medians; peak RSS is `VmHWM` of the
role's process after resetting it once the inputs are built (`/proc/self/clear_refs`).

| `N` | `d_main` | Prove | Verify uncached | Verify cached keys | Prover peak RSS | Verifier peak RSS | Aggregate | Public bits | Aggregate without bits |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 18 | 1.38 s | 34.3 ms | 31.6 ms | 310 MiB | 100 MiB | 273,069 B | 196,616 B | 76,453 B |
| 512 | 19 | 2.20 s | 55.0 ms | 41.5 ms | 453 MiB | 117 MiB | 478,058 B | 393,224 B | 84,834 B |
| 1,024 | 20 | 3.90 s | 78.3 ms | 50.8 ms | 782 MiB | 123 MiB | 889,459 B | 786,440 B | 103,019 B |

The RSS before proving was 93, 120 and 173 MiB (records, instance, expanded keys, test
matrices); before verifying 68, 69 and 71 MiB, of which about 62 MB are the fixed test
matrices the verifier process builds. The aggregate is `32 N` bytes of `c~` plus 768 bytes of
public bits per signature (`786,440 = 1,024 * 768 + 8` bytes of framing at one segment)
plus 59 to 61 KB of opening, 7.8 to 8.2 KB of PIOP messages, 528 bytes of Akita
commitments and 512 bytes of final vector. Without the public bits (mode A replaces them
by a Flock commitment and proof) it is 76 to 103 KB, of which `32 N` are the hashes.

Verifier steps at `N = 1,024` (ms, medians; uncached / cached keys): parse 5.4 / 5.0
(wire parse, manifest and the PCS verifier from the plan's setup seed; the parts are not
timed separately, and at 256 the step still takes 3.9 ms, so the setup dominates); public data 5.0 / 4.4 (`pkDecode`, `tr`, `mu`,
`SampleInBall`, record digest); transcript 1.0 / 1.0; matrices 0 (borrowed); Keccak side
7.2 / 7.3 (1,024 SHAKE256 hashes of 832 bytes, range checks, GF(2) projection); prestage
0.04; stage 1 5.0 / 4.7 (`w_2, w_3`); stage 2 10.8 / 10.2 (`c_1`: `14 2^19` additions and a
`2^19` equality table); stage-3 rounds 0.07; public pass `T` 6.5 / 5.6 and `A` 27.8 / 4.0
(992 + 3 ExpandA and 16 inverse NTTs each when uncached); stage-3 terminal 0.04; bit check
1.1 / 1.0; opening 6.8 / 6.7. At 256 the same steps total 34.3 / 31.6 ms.

Prover steps at `N = 1,024` (last run, ms): witness tables 76, Akita setup 7, round 1 143,
round 2 (matrices, parities, carries, commitments) 84, stage 1 691, stage 2 1,827, stage 3
700, opening 338; stage 2 (degree 5, 101 instances) dominates.

`tests/mode_b.rs` prints the same breakdowns for one run each (`--nocapture`); there the
round trips run concurrently with the other tests, so its timings are noisier.

## 14. Tests

`taskset -c 0,1,3-15 cargo test -p agg-piop --release` (61 tests, about 25 s after the
build; the 1,024-signature round trip is ignored in debug builds):

- unit: manifest shapes and binding bytes; field conversion; kernel and `Lambda` vs their
  definitions; `sparse_t`, inverse NTT, `prepare` vs `from_instance`, `T(r)` vs the MLE;
  leaves and digit tables; kernel groups (equality and weight groups) vs direct sums with
  the terminal equation at random challenges, compressed-message round trips, lift and
  rejections; weight tables vs dense definitions and vs their verifier evaluations at
  Boolean and random points, `w_A` vs `(A z)` for random `z` and vs its extension at `r3`;
  transcript registry order, shape faults, determinism, encoding round trips; claim
  vectors; the three stage specs (instance counts, gamma order, degrees, every column read
  at `r3`); wire round trips, the `ProofSize` parts summing to the length, and every
  malformed-buffer rejection; record binding; the stage-3 deferral (compressed rounds
  are affine in the claim with slope `prod_k r_k`); `StreamedPublic` (`mu`, `T_i`, and
  the pass against `PublicData::t_evaluation` and `weights::ring_weight_at`, without a
  cache, filling a cache, and from it with no expansion; a malformed key); mode B:
  `segment_projection` against `agg_witness::bridge::gf2_composite` on a real batch and
  against a sparse per-lane reference at 600, `2^13 + 700` (two segments) and `2^14 + 3`
  (four segments, the last one padded) records; `public_bit_evaluations` against the
  extensions of the bit lanes and on malformed items; `PublicBitsVerifier` accepting the
  honest segment and rejecting a non-empty proof, a short item, a flipped bit (hash),
  `w_1' >= 44`, a changed `Y` bit (projection) and a changed `c~`; `check_bits` with a
  changed claim.
- kernel (`src/kernel/tests.rs`, `coded.rs`, `field.rs`; Section 18): every evaluation
  strategy (automatic; no coding; coded per item; quartet masses forced; pair masses only;
  quartet tables forced, with and without Taylor tables; pair fold at level 2; without the
  claim recovery, also with quartet tables) against an independent reference (full `K`
  tables, the round polynomial from its definition at `d + 1` points over full equality
  tables, interpolated), bit for bit in every message and terminal value and with the
  terminal equation, on an equality group with every `Term` kind (Booleans, `RangeW1`, both
  leaves of a digit, `A^lo A^hi`, links, reductions, a base-4 leaf on a five-class column, a
  weightless affine form), a weight group (field, `i32` and materialized lanes, weighted
  sums, a field-field product), a group with a tampered wide column, an `i32` lane and
  cross-column products, the two-round prefix at `2^16` cells, and domains of 0 to 4
  variables with and without lifted rounds; classes of digit columns under the square, leaf
  and identity maps (8, 8, 16, and 17 for a tampered 9: not coded); pattern masses against
  the serial sums in both accumulation layouts; quartet and Taylor tables against the folded
  lanes and the shifted polynomial; the compiled pieces (source, folded, depressed, upper
  coefficients) against the terms; the fast product against `*` on boundary and random
  inputs; split and plain unreduced sums against nested sums; Taylor shifts; narrow and wide
  parity tables (`tables.rs`). `profile_stage_groups` (ignored) prints per-round times of
  stage-like groups (`AGG_PROFILE_VARS`, `AGG_PROFILE_POLICY`).
- `tests/stages.rs` (256 signatures, 200 own keys, 56 on 3 shared keys): every stage-1 and
  stage-2 zero-check sums to zero and every level, `def_y1` and link claimed sum equals an
  independent dense evaluation (equality tables materialized, weights from their table
  definitions, `(A z)` by the integer CRT product); the ring sum equals `T(r2)`; the
  terminal claims of each stage equal the multilinear extensions of the virtual tables at
  the stage point; prover and verifier outputs agree; a changed terminal claim or round
  message is rejected at the stage. Tampering (50 and 24 records, unchecked prover): a
  changed `z` digit, a changed `v` digit (residual off by `q`) and a flipped bit fail at
  stage 3 only; a non-Boolean bit, `w_1' >= 44`, a base-16 digit outside its alphabet, a
  changed level-1 carry digit, a flipped level-2 parity and a flipped `Y` bit fail at
  stage 1; base-4 and base-8 digits outside their alphabets pass stage 1 and fail at
  stage 2.
- `tests/prove_verify.rs`: a full prover run at 256 signatures; the opening claims taken
  from the stage-3 terminal values equal the evaluations of the committed tables at the
  group points; the blob round-trips and verifies from `(pk, M)` with and without cached
  keys; a shorter or reordered record
  list, a changed `c~`, other bridge matrices, another Flock profile, a changed round
  message (rejected at stage 2), a changed opening claim (stage 3), a changed pre-stage
  claim, a corrupted opening byte, a truncated commitment, a non-empty Keccak payload in
  mode B and a truncated blob are rejected; a padded batch of 10 records round-trips.
- `tests/mode_b.rs`: `prove_mode_b` / `verify_mode_b` round trips at 256 (200 own keys,
  56 on 3 shared keys) and 1,024 signatures (1,000 and 24; about 4.5 s): the proof size by
  part, the prover and verifier step timings (uncached, cached) printed; the published
  bits satisfy `c~ = H(mu || w1Encode)`; a cached run expands no key. Rejections at 256: a
  flipped public bit, a public value of 63, a short bit item, a changed `c~`, a changed
  final-vector bit, other matrices and a non-empty proof item at the Keccak side of
  segment 0; a changed pre-stage claim at stage 1, round messages at stages 2 and 3, a
  changed stage-3 claim at stage 3; an opening byte at the opening; a reordered or shorter
  record list, another Flock profile and a truncated aggregate. Every tampering of
  `tests/stages.rs` through the full prover (`prove_with_tables`, unchecked stages, honest
  public bits, 50 records): `z` and `v` digits at stage 3; a non-Boolean bit at stage 1;
  base-4 and base-8 digits at stage 2; a level-1 carry digit and a level-2 parity at
  stage 1; a flipped committed bit, `w_1' = 44` in the committed bits and a flipped `Y`
  bit at the Keccak side (the final vector no longer matches the projection of the public
  bits); a base-16 digit of 9 cannot be committed (Akita: "committed source exceeds the
  scheduled coefficient bound"). Bit check: with two equal `M_1` columns a prover flips
  `b_0` at two positions and moves `u` by `-/+ 2 gamma_2`; every PIOP check, the projection
  and the opening pass, `verify_mode_b` rejects with `VerifyError::Bits`, and the same
  aggregate is accepted by a verifier without the bit check (deviation P9).
- `tests/bridge.rs` (the Fp side of the bridge, `docs/bridge.md` Section 9 item 7): a full
  prover run at 256 signatures with `SeededMatrices` verifies; the round-2 tables built through
  `agg_bridge` equal the `agg-witness` reference, satisfy `agg_bridge::parity::check`, and the
  published `Y` equals both GF(2) composites; the verifier's bridge weights at `r^(1)` (`w_2`,
  `w_{3,t}`, `stage1::BridgeWeightsAt`) and at `r^(2)` (`w_1`) equal the extensions of the
  prover's weight tables at the transcript's points. Tampering through the full prover (48
  records, `FixedMatrices`, unchecked stages): a flipped `Y` bit, a flipped level-2 parity, a
  level-1 and a level-3 carry digit, and a flipped committed bit under the stale round-2
  tables are each rejected at the stage-1 terminal.
- `tests/commit_open.rs`, `tests/virtual_checker.rs`, `tests/source_tables.rs` as in part 1.

`agg-bench` (`taskset -c 0,1,3-15 cargo test -p agg-bench --release`, 3 tests): medians,
batch inputs, `VmHWM`/`VmRSS` parsing, the peak following a 64 MiB allocation and its reset.

## 15. Profile of stages 1 to 3 (2026-10-24, before the kernel rework)

Setting: this machine, 7 threads (`taskset -c 0,1,3-15`, `RAYON_NUM_THREADS=7`), CLMUL
build (`RUSTFLAGS="-C target-feature=+pclmulqdq"`, `CARGO_TARGET_DIR=target/clmul`), tree
`a735dff` plus the `AGG_STAGE_TRACE` instrumentation of `stage::prove_stage` (per group:
message and fold time summed over the rounds; per round: wall time), `N = 1,024`
(`d_main = 20`), one run each, load average below 1. Mode A: `agg bench --count 1024
--warmup 0 --reps 1`; mode B: `cargo bench -p agg-bench --bench mode_b -- 1024 --warmup 0
--reps 1`. The stages cost the same in both modes (mode A 667 / 1,829 / 683 ms, mode B
656 / 1,842 / 682 ms for stages 1 / 2 / 3); the non-Flock prover is 4.03 s (mode A) and
3.95 s (mode B), of which stages 1-3 are 3.2 s, the `G_main` commitment 0.15 s, the opening
0.34 s, round 2 (matrices, parities, carries, commitments) 0.17-0.20 s.

Per group (message = `Group::round_message` over all rounds, bind = `Group::bind`), ms:

| Stage | Group | Lanes | Instances | Message | Bind | Share of the stage |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | main, `eq(tau)`: 6 bits, 14 x (`A^lo`, `A^hi`) | 34 | 21 | 478 | 88 | 89% |
| 1 | `D_1`, `eq(tau_1)`: 6 `y^(1)`, 12 x (`A^lo`, `A^hi`) | 30 | 18 | 19 | 3 | 3% |
| 1 | `D_2`, `eq(tau_2)`: 6 `y^(2)`, 12 x (`A^lo`, `A^hi`) | 30 | 18 | 13 | 2 | 2% |
| 1 | `D_3`, `eq(tau_3)` | 4 | 2 | 2 | 0 | |
| 1 | `D_1`, weight `w_2` | 7 | 6 | 8 | 1 | 1% |
| 1 | `D_2`, weights `w_{3,t}` | 12 | 1 | 6 | 1 | 1% |
| 2 | main, weight `w_1`: 6 bits (`def_y1`) | 7 | 6 | 216 | 17 | 13% |
| 2 | main, `eq(r^(1))`: 17 `P_Z` (leaves) | 17 | 31 | 1,444 | 45 | 81% |
| 2 | `D_1`, `eq(r^(1)_1)`: 18 `P_Z` | 18 | 30 | 58 | 2 | 3% |
| 2 | `D_2`, `eq(r^(1)_2)`: 18 `P_Z` | 18 | 30 | 41 | 1 | 2% |
| 2 | `D_3`, `eq(r^(1)_3)`: 2 `P_Z` | 2 | 4 | 5 | 0 | |
| 3 | main, `eq(r^(2))`: 23 columns (ring residual, 17 links, 6 reductions) | 23 | 24 | 369 | 64 | 61% |
| 3 | main, weight `w_A`: 5 `z` digits | 6 | 1 | 60 | 17 | 11% |
| 3 | main, `eq(r^(1))`: 6 bits (reductions) | 6 | 6 | 88 | 20 | 15% |
| 3 | the seven bridge-domain groups | 2-24 | 2-24 | 42 | 6 | 7% |

Per round (ms, round 0 first): stage 1 `272 130 66 36 18 11 18 21 24 12 6 3 2 1 0 ...`;
stage 2 `829 396 203 115 64 35 43 30 57 28 14 7 4 2 1 0 ...`; stage 3 `292 146 72 38 20 14
17 19 23 12 6 3 1 1 0 ...`. Round `r` has `2^(d_main - 1 - r)` pairs of cells, so round 0
is 43-45% of a stage, round 1 20-21%, round 2 10-11%; the bump at rounds 6 to 8 is the
bridge-domain groups becoming active (`d_1 = 14`, `d_2 = d_3 = 13` at this `N`). The
weight tables are not the cost: `y^(1)` parities 9.5 ms, `w_2, w_3` 0.8 ms, `w_1` 5.2 ms,
`w_A` 32.5 ms.

Where the time goes (`kernel::Group::inner_evaluations`): for every pair of cells and
every interpolation point `X` (`degree` points with an equality factor, `degree + 1`
without) every term of every instance is evaluated in `K` and multiplied by its gamma. The
stage-2 main group evaluates 31 leaves `Q(P_Z)` of degree 4 at 5 points: `31 * 5 * (4 + 1)
= 775` `K` multiplications per pair (about 11 ns each on this machine, 8.7 us per pair in
round 0), the same in every round although the first round reads `i8` cells. The stage-1
main group does 21 terms x 4 points x 2 multiplications (term and gamma) plus 34 lane
extrapolations; the stage-3 main group an affine form in 17 lanes (17 `mul_i64`, each a
full `K` multiplication: `jolt_field::Ring::mul_i64` is `self * from_i64(n)`) plus 17
links and 6 reductions at 3 points. Folding (`bind`) is 10-15% of a stage; the first fold
lifts every cell with one `K` multiplication (`challenge.mul_i64(high - low)`).

What this suggests, measured against Aerie's kernels (`aerie-prover/src/protocol/kernels/`,
`quartic_kernel.rs`, `range_source_kernel.rs`): Aerie never evaluates a term per cell in
the first rounds. Its sources are coded (`CodedSource`: cells in `[min, max]` with
`max - min < 16`), round 0 buckets the equality weights by the pair pattern of the two
cells (256 patterns per source) and evaluates each term once per pattern, round 1 does the
same on quartet patterns when the domain is large enough (`BUCKET_MIN_RATIO = 4`), and
the lifted lanes appear only after the compact rounds. Its terms are also pre-aggregated
per source (`aggregate_weights`: one gamma-weighted polynomial per digit instead of two
leaves). Neither is done here. The committed layout itself (23 main-domain columns) is not
what separates us from Aerie's cost; see Section 16 for the element-count analysis.

Reference figures of this profile (prove, ms, one run): mode A 6,272 total (Flock part
2,238, stages 3,178); mode B 3,945 (stages 3,180).

## 16. Design of the kernel rework (2026-10-24)

### 16.1 Committed elements: what can and cannot be cut

The 23 main-domain columns (spec Section 4.1) hold 70 bits of range-checked value per
position (`z` 18, `u` 19, `v` 27, `w_1'` 6) in base-16 digits (4 bits each) plus the six
bits; 5 + 5 + 7 + 6 = 23 is the minimum for that argument. The 8,192 elements of the H
layout in `ref/notes/improvements-tables.md` assumed that an Akita width certifies a
range; the review withdrew that (Akita Remark 10.41, spec B.3). The candidates of the
task, with numbers:

- **Divisibility projection instead of `v` (P2).** A single-level residual projection
  with coefficients in `[-12, 12]` over the `2^25` entries of `Delta_0 = (A z) - T -
  2 gamma_2 w_1' - u` has worst-case output `2^49 * 12 * 2^25 = 2^77.6` (honest RMS
  `2^57.75`; Aerie's three composed levels give about `2^63`), against `p/2 = 2^63`: the
  integer output cannot be recovered from its residue, so `q | y` cannot be checked. The
  worst case is reachable by a signer who holds the key (`z` sign-aligned with a row of
  `A` gives `|(A z)_j| ~ 2^48`), so a completeness cap below `B_v` would make valid
  signatures non-aggregatable. A coarse quotient `v = 2^m v_hi + v_lo` with only `v_hi`
  committed needs `2^(22 + m) * 12 * 2^25 < 2^63`, so `m <= 12` and `v_hi` keeps 15 bits
  (4 base-16 columns): 3 columns saved for the whole projection machinery (seeded cores,
  weight pull-back through `A`, Rice codec, one stage-1 instance). Sending the output as a
  clear integer with slack `y = Y - m p` is unsound (a residual entry `== p (mod q)` lets
  every output be shifted to a multiple of `q`). Not adopted.
- **One wide `v` column with Aerie's norm projection as an approximate range proof.**
  (6.1) needs `|v| < 2^39.9`, i.e. `||v||_2^2 < 2^79.8` over the batch. Aerie's Theorem 5.1
  certifies `E* = (Theta + 1) / zeta` with `zeta = 1024^-5 (33/32)^1470 ~ 2^15.3`, a slack
  of about `2^15.8` over the honest batch norm; the honest worst case for `v` is
  `||v||_2^2 <= 2^25 * 2^52 = 2^77`, so the certified bound would be `2^92.8`. Not usable
  without a new projection analysis (Aerie's is parameter-specific and Lean-checked).
- **Wider Akita digits.** A width-27 column needs two Akita inner digits instead of seven
  (docs/pcs-config.md Section 2), but gives no range, and the leaf argument needs the
  digits themselves committed (`P_Z = Z (Z + 1)` is a table of a committed `Z`). No gain.
- **Base-32 digits.** Save 3 columns (`z` 4, `u` 4, `v` 6) but need 4 virtual factor lanes
  per digit in stage 1 and 4 degree-5 leaves per digit in stage 2 instead of 2 each; the
  sum-check, not the PCS (0.5 s of 4 s at 1K), is the cost. Not adopted.
- **Merging the six Boolean lanes.** The bridge takes six separate Boolean lanes
  (parities per bit) and the bits are not low-degree in `w_1'`. Not possible in relation H.
- **Lookups** (Shout, LogUp) would allow 16-bit digits and about 12 columns; none exists
  for fp64/Akita (risks R5). Open.

Conclusion: the element count stays at 23,552 main-domain elements per signature; the
wire format, the verifier and the soundness ledger of spec Section 12 are unchanged by
this work. The cost gap to Aerie is in the prover's kernel, not in the layout.

### 16.2 Kernel rework (prover only, bit-identical proofs)

Three changes in `kernel::Group`, all of which compute the same round polynomials:

1. **Compiled evaluation plan.** At `Group::new`, the gamma-weighted terms are compiled
   into: one polynomial per lane in `K[v]` (the sum of every single-lane term on that
   lane: `Bool`, `Link`, `Leaf`, one-lane `Product`, the lanes of a weightless `Affine`,
   the constant of an `Affine`), evaluated by Horner at each interpolation point; weighted
   affine forms `w (sum_l c_l v_l + c)` with `K` coefficients (gamma folded in);
   bilinear products `c * v_a * v_b`; and the residual generic terms (`RangeW1`). The
   stage-2 leaves of one digit cost 4 multiplications per point instead of 10, the stage-3
   links and residual 2 per lane instead of 3.4.
2. **Compact round 0 (pattern buckets).** A `Source::Bytes` lane whose cells lie in a
   span of at most 16 values is *coded* (`min`, `span`); a lane pair `(cell(2k),
   cell(2k+1))` has one of at most 256 patterns. In the first active round of a group
   whose terms read only coded lanes (and, for weight groups, the one weight lane), the
   prover sums the equality weights (or the weight lane's values at every interpolation
   point) into one bucket per pattern per source column (one `K` addition per pair per
   column), then evaluates every term once per pattern. Multi-column terms (`RangeW1` on
   four bit columns) use a joint pattern of at most `2^12` entries. Groups with an `Ints`
   lane (the `y^(1)` parities on `D_1`) or a non-coded cell (tampered tables in tests)
   fall back to the generic path. Aerie's `CodedSource` and `chunked_bucket_sums` do the
   same.
3. **Fold by lookup.** The first fold of a coded lane is a 256-entry table lookup
   (`lut[pattern] = v(2k) + r (v(2k+1) - v(2k))`) instead of one `K` multiplication per
   cell; a lane read by exactly one linear term is scaled by its coefficient at this
   fold, so that term costs no multiplication in later rounds.

Why this is sound: every message is the same polynomial `g_j(Y)` as before, evaluated in
a different order over `K` (bucketing is distributivity of the eq weights over equal
summands; Horner and the folded coefficients are exact). The verifier is untouched, so the
existing stage tests (prover/verifier agreement, tampering at the stated stage) check the
rework end to end; `kernel` tests compare the compact and generic paths on random tables
with every `Term` kind. Memory after the first fold is unchanged (one `K` per cell per
lane); a coded `u8` representation through round 1 (Aerie's quartet LUT) would divide
it by 16 and is the next step if measurements justify it (Section 17).

## 17. Measurements of the kernel rework

(First step, `50fc538`. The second step, codes through three rounds and a dense table from
the third fold, is Section 18; its measurements are in Section 18.3.)

Same machine and build as Section 15 (7 threads, CLMUL build), tree `a735dff` plus the
rework; full tables in `ref/notes/bench-local.md` (last section). Every aggregate is
byte-identical to the one before the rework and verifies with the unchanged verifier.

Stages at `N = 1,024` through the steps of Section 16.2 (ms, one traced run each):

| Step | Stage 1 | Stage 2 | Stage 3 | Rounds 0, 1, 2 of stage 2 |
| --- | ---: | ---: | ---: | --- |
| before | 637 | 1,848 | 706 | 829, 396, 203 |
| compiled plan, bucketed round 0, lookup fold | 295 | 579 | 360 | 32, 220, 112 |
| plus round 1 bucketed by quartet patterns | 257 | 441 | 348 | 18, 104, 112 |
| (quartet lookup after the second fold, rejected) | 305 | 516 | 383 | 17, 96, 172 |

Medians of 5 runs after 1 warmup (`agg bench`, mode A, and the mode-B bench), ms:

| `N` | Stage 1 | Stage 2 | Stage 3 | Stages per signature | Non-Flock prover (mode A) | Mode-B prove |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 104 | 203 | 122 | 1.68 ms | 917 | 879 |
| 512 | 161 | 317 | 192 | 1.31 ms | 1,283 | 1,233 |
| 1,024 | 258 | 452 | 353 | 1.04 ms | 1,852 | 1,825 |
| 2,048 | 446 | 783 | 608 | 0.90 ms | 2,931 | 2,789 |

Before the rework the stages cost 1,032 / 1,723 / 3,245 / 6,070 ms at these sizes (2.4x
to 3.3x), the non-Flock prover 1,510 / 2,346 / 4,103 / 7,171 ms. The per-signature cost
falls with `N` because the bucketed rounds amortize over more pairs (the quartet space
is fixed). What remains: round 2, the first generic round over the materialized `K`
lanes, is now the largest term of every stage (249 ms of stage 2 at `N = 2,048`), then
the geometric tail of the later rounds; a third compact round would need Aerie's
quartet-LUT approach with a cache-resident layout (the straightforward 1 MiB-per-lane
lookup was slower, see the table). The `G_main` commitment (138 ms) and the opening
(325 ms) are now a third of the non-Flock prover at `N = 1,024`.

## 18. Kernel rework, second step: codes through three rounds (2026-10-24)

Prover only: every message, terminal value and proof byte is unchanged (aggregates
byte-identical to `7ed4a32` at 256 to 2,048 records, Section 18.3), so the transcript, the
wire format (version 0) and the verifier are untouched.

**Public API changes** (for code that uses `kernel` or the prover; all inside `agg-piop`):

- `src/kernel.rs` is now the directory `src/kernel/` (`mod.rs` with the public items,
  `plan.rs`, `coded.rs`, `field.rs`, `tests.rs`). Unchanged public items: `Source`,
  `Lane::{column, mapped, len, is_empty, value, fold, terminal}`, `Lane::Source`,
  `Lane::Folded`, `Term`, `LeafPoly`, `GroupInstance`, `Group::{new, claim, round_message,
  bind, terminal_values, inactive_rounds, is_bucketed}`, `compress`, `evaluate_compressed`,
  `inactive_message`, `inverse_power_of_two`, `expected_terminal`, `digit_map`,
  `IDENTITY_MAP`. Removed: `Lane::Lookup` (it held a crate-private `Column` and could not be
  built outside the kernel); `Lane` is now the input type only, the compact states live in
  `Group`.
- `tables::ProverTables::parity1` is a `tables::Parity1` (`Narrow([Vec<i8>; 6])` when every
  parity fits an `i8`, as for an honest witness, `Wide([Vec<i32>; 6])` otherwise, with
  `lane(t)` and `value(t, index)`) instead of `[Vec<i32>; 6]`.
- `prover::prove` and `prover::prove_with_tables` take `keccak: &(impl KeccakSide + Sync)`
  (every implementation in the workspace already is: `PublicBits`, `NoKeccak`,
  `agg-cli`'s `FlockProver`, which holds its state in a `Mutex`).
- `pcs::TableCommitter::open_claims(commitments, claims, session)`; `open` is kept.
- `public_data::negacyclic_kernel_into(a, r, out, scratch)` (no allocation).

### 18.1 What was implemented (handoff items)

1. *Coding by lane maps* (item 1) as designed: classes of the present cells under the tuple
   of the maps of the lanes on the column, at most 16; one byte of pair code per pair.
2. *Pieces* (item 2): bucketed per column or joint column set; direct otherwise. Weighted
   sums replace the keyed buckets (`Piece::Weighted`, `WeightConstant` are gone): every
   `Affine` with a weight and every product of a field lane with another lane is one form
   `w (sum c_l v_l + c)` per weight lane; in the first round its byte lanes are read through
   per-byte tables `c_l map_l[byte]` (additions only), later its lanes absorb their
   coefficients at the first fold. The virtual `Int` lane of the design was not needed: with
   the per-byte tables and the absorbed coefficients the `w_A` and `w_1` groups cost about
   seven products per pair at every level (synthetic profile below: 33 and 36 ms at
   `2^21`, against 110 and 128 ms of message and fold time in the `50fc538` trace at
   2,048).
3. *Lane states* (item 3): `Source -> Pair -> Quartet | PairFold -> Dense`, plus `Table` for
   uncoded lanes; `terminal_values` reads any state (domains of 0 to 3 variables end in a
   compact state).
4. *Rounds* (item 4): quartet masses with `remaining_eq_tables_after(1)` serve rounds 0 and 1;
   level 2 reads quartet tables with Taylor tables. Departures: see 18.2.
5. *Coefficient form* (item 5) everywhere, and two further reductions of the arithmetic: the
   constant coefficient is recovered from the running claim from the second active round
   on, and the last product of each coefficient is accumulated unreduced.
6. *Narrow parities* (item 6) as designed (`tables::Parity1`); the stage-1 `D_1` group is now
   bucketed.
7. *Faster `K` product* (item 7): `kernel::field::fm`, four `64 x 64` products and two
   branch-free reductions (`2^64 = 59`, `2^128 = 3481` modulo `p`); 7.4 ns against 9.4 ns for
   `*` in a throughput loop on this machine (Aerie's `opening_field_ops::mul`: 7.7 ns), equal
   to `*` on boundary and random inputs (`fast_product_matches_the_field_product`).
8. *Opening and commitments*: (2a) the opening takes its claims from the stage-3 terminal
   values (`TableCommitter::open_claims`; the recomputation is a debug assertion and a check
   in `tests/prove_verify.rs`); (2b) `rayon::join` of the round-1 commitment and
   `KeccakSide::commitments`, no change to `agg-cli` needed; (2c) the round-2 groups are
   committed concurrently; (2d) no planner change (Section 16 of the handoff).
9. *Weight table `w_A`* (not in the handoff; it was 73 ms of stage 3 at 2,048 after the
   kernel work): the negacyclic transform of `ring_weight_table` is allocation-free (it
   allocated two vectors per recursion node, about 2,000 per record) with one fold product
   per butterfly output, the row weights are scaled by `eq(r2_sig, i)` once per record and
   combined with the integer rows of `A` by base-field products: 73 to 36 ms at 2,048
   (single traced runs).

### 18.2 Where the implementation departs from the design, with the measured alternatives

Synthetic stage-like groups at `2^21` cells (the main domain at `N = 2,048`;
`kernel::tests::profile_stage_groups`, 7 threads, minimum of 3 runs, load average 1 to 3),
total ms for the stage-1, stage-2 and stage-3 main equality groups:

| Variant | Stage 1 | Stage 2 | Stage 3 |
| --- | ---: | ---: | ---: |
| first complete version (dense table filled by a serial `vec!`, Taylor tables with the constant) | 275 | 353 | 261 |
| dense table filled in parallel row by row | 170 | 199 | 179 |
| Taylor tables without the constant, claim recovery, dense table filled in lane-major row blocks | 145 | 175 | 165 |
| unreduced accumulation of the last products | 132 | 156 | 154 |
| level-1 patterns without divisions, wide masses with shared item weights, moments for polynomial-only columns | 128 | 149 | 132 |

Measured and rejected (kept out of the code):

- *Quartet tables for 16-class columns*: the stage-3 links (identity map) have `16^4`
  quartet patterns, 1 MiB per lane; level 2 keeps the pair table and folds by the second
  challenge on the fly. For 8-class columns the quartet table wins (one profile run, stage-2
  level 2: 58 ms with quartet and Taylor tables, 86 ms with quartet tables only, 96 ms with
  the pair fold).
- *Dense table by one indexed parallel collect* (row-major, multiply-shift division by the
  lane count, no zero fill): 49.5 ms for the stage-1 materialization against 31.6 ms for
  lane-major row blocks after a parallel first touch, because each row touches every lane's
  quartet table.
- *Level 2 segment by segment over blocks of 256 items* (so each segment's tables stay in
  cache): stage 1, 2, 3 level 2 at 33, 42, 42 ms against 25, 40, 36 ms item by item.
- *Deferring the materialization to the fourth fold*: estimated from the single-thread
  profile (bind 2 at 23 ns per cell, quartet reads plus two folds per lane and pair in
  round 3) to save about 10 ms on stage 1 and to cost on stage 3 (pair-fold lanes); not
  implemented.

The bivariate prefix does apply to the 16-class stage-3 group from `N = 1,024` on (`2^18`
quartets against `4 * 16^4`), contrary to the handoff's estimate (from 2,048); its masses are
17 tables of 1 MiB accumulated one table per task (no replication), and round 1 reads them
by moments (2.5 ms instead of 12 ms at `2^21`).

### 18.3 Measurements

This machine, 7 threads (`taskset -c 0,1,3-15`, `RAYON_NUM_THREADS=7`), CLMUL build
(`RUSTFLAGS="-C target-feature=+pclmulqdq" CARGO_TARGET_DIR=target/clmul`; the CLI with
`--features agg-keccak/avx512,agg-piop/avx512`), 2026-10-24 16:53 to 16:56. "Before" is the
binary of `7ed4a32` built at the start of this work (PIOP code of `50fc538`), "after" the
tree with this section's changes; the two ran back to back, before first, medians of 5 timed
runs after 1 warmup. Load average 2.8 at the start and 5.1 to 5.6 during the runs (the runs'
own 7 threads included; no other job was visible). Every mode-B aggregate (256 to 2,048
records) and the mode-A aggregates of 256 and 1,024 records (`agg aggregate`, same records)
are byte-identical before and after.

Mode B (`cargo bench -p agg-bench --bench mode_b -- 256 512 1024 2048 --warmup 1 --reps 5`),
prover steps of the median run, ms:

| `N` | Prove before / after | Stage 1 | Stage 2 | Stage 3 | Stages per signature | Opening | Round 1 (`G_main`) | Round 2 | Prover peak RSS (MiB) |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | 0.880 / 0.601 s | 104.0 / 44.5 | 213.1 / 60.7 | 120.5 / 56.7 | 1.71 / 0.63 ms | 218.6 / 220.2 | 36.5 / 36.1 | 161.4 / 155.8 | 294 / 302 |
| 512 | 1.245 / 0.761 s | 177.9 / 72.5 | 319.7 / 82.5 | 208.4 / 90.9 | 1.38 / 0.48 ms | 255.2 / 242.2 | 67.8 / 67.5 | 170.0 / 155.9 | 437 / 449 |
| 1,024 | 1.770 / 1.038 s | 276.6 / 99.6 | 440.6 / 127.0 | 340.9 / 152.1 | 1.03 / 0.37 ms | 306.8 / 272.8 | 138.2 / 135.2 | 181.5 / 163.3 | 809 / 668 |
| 2,048 | 2.734 / 1.495 s | 442.1 / 176.9 | 776.3 / 229.1 | 601.0 / 244.5 | 0.89 / 0.32 ms | 383.0 / 311.3 | 182.4 / 192.1 | 180.1 / 173.2 | 1,343 / 1,125 |

Stages 1-3 are 2.7x, 2.9x, 2.8x and 2.8x faster (256 to 2,048); the prover 1.46x to 1.83x. The
opening is 19% faster at 2,048 (the 67 table evaluations are gone, 2a) and unchanged at 256,
where Akita's `batched_prove` dominates. Round 2 is mostly the seeded matrices (P4); its
concurrent commits save 6 to 18 ms. Verification is unchanged (174 / 183 / 213 / 260 ms
uncached before, 173 / 196 / 201 / 256 ms after). The prover's peak falls at 1,024 and 2,048
because the first `K` tables are an eighth of the domain instead of a quarter.

Per round, one traced run after one warmup (`AGG_STAGE_TRACE=1`, prover child), ms, round 0
first:

| `N`, stage | Before (`50fc538` profile, Handoff) | After |
| --- | --- | --- |
| 1,024, stage 1 | `10 92 59 25 16 9 13 8 15 8 4 2 1 1` | `7 1 31 11 6 4 3 4 5 3 2 1 0` |
| 1,024, stage 2 | `17 116 118 61 37 19 13 12 31 16 8 4 2 1 1` | `16 5 32 17 11 7 4 7 8 5 2 1 1` |
| 1,024, stage 3 | `22 143 57 32 16 9 10 9 17 8 4 2 1 1` | `15 5 40 12 7 5 3 5 6 4 2 1 0` |
| 2,048, stage 1 | `21 140 113 48 28 17 19 11 8 15 8 4 2 1 1` | `13 1 58 19 10 6 4 4 5 5 3 2 1` |
| 2,048, stage 2 | `34 162 226 115 62 30 22 22 15 31 16 8 4 2 1 1` | `20 7 61 34 20 10 7 8 6 9 5 2 1 1` |
| 2,048, stage 3 | `42 198 112 66 36 17 13 13 8 17 9 4 2 1 1` | `28 8 83 24 12 8 6 5 6 6 4 2 1` |

Round 1 now costs 1 to 8 ms (quartet masses from round 0); round 2 (level 2 and the
materialization at bind 2) is the largest round of every stage, then the first dense round.
Per group at 2,048 (message / bind summed over the rounds, before -> after, ms): stage 1
main `eq(tau)` 272 / 124 -> 66 / 56, `D_1 eq(tau)` (now coded parities) 19 / 3 -> 4 / 2;
stage 2 main `eq(r^(1))` 516 / 49 -> 112 / 30, main `w_1` 78 / 32 -> 18 / 15; stage 3 main
`eq(r^(2))` 244 / 64 -> 76 / 44, main `w_A` 110 / 18 -> 20 / 18, main `eq(r^(1))` 23 / 29 ->
11 / 10. Weight tables at 2,048: `w_A` 77 -> 33 ms (item 9), `w_1` 9 -> 8 ms, parities
(`agg_bridge::parity`, not this crate) 19 -> 20 ms.

Mode A (`agg bench --count 1024 --warmup 1 --reps 5`): prove 3.140 -> 2.374 s (min 3.115 ->
2.361 s); stages 258 / 449 / 361 -> 100 / 129 / 141 ms; opening 309 -> 297 ms; the `G_main`
commitment (137 -> 183 ms) now runs concurrently with the Flock commitment (183 -> 249 ms,
both slower when they share the cores), and the listed steps add up to 171 ms more than the
total, the overlap; Flock parts 1.300 -> 1.375 s (commit included), everything else 1.840 ->
0.999 s; verify 272 / 243 -> 283 / 263 ms (verifier unchanged; noise); prover peak RSS 1,699
-> 1,672 MiB; aggregate 717,641 B before and after.

Against the goals: stages 1-3 are 2.8x faster at 1,024 and 2,048 (target about 2x), 0.32 ms
per signature at 2,048 on 7 threads. At 32K this extrapolates to about 10 s of stages on this
machine and, with the cloud-to-local ratio of the `50fc538` runs (13.4 s measured against
29 s extrapolated), about 5 s on the cloud VM (estimates, not measured; Aerie's stages take
0.60 s there). Per committed element that is about 6.5 ns on the cloud VM at 32K
(estimate) against Aerie's 12 ns.

## Handoff (2026-10-24)

State of the working tree: clean at `50fc538`; nothing was edited yet (the baseline
build in `target/clmul` is current, the workspace tests of `50fc538` pass as recorded in
Section 14). Everything below is the profile taken for the next kernel step, the design
decided from it, and the implementation order. Raw logs are in the session scratchpad
only (`bench/baseline-*.log`, `bench/trace-baseline.log`).

### Done: profile of `50fc538` (CLMUL build, 7 threads, load average 0.7 to 3.7)

Mode B, `cargo bench -p agg-bench --bench mode_b -- N --warmup 1 --reps 5`
(`RUSTFLAGS="-C target-feature=+pclmulqdq" CARGO_TARGET_DIR=target/clmul`,
`RAYON_NUM_THREADS=7`), medians of 5, prover steps in ms:

| `N` | Prove | tables | round 1 (`G_main`) | round 2 (matrices etc.) | stage 1 | stage 2 | stage 3 | opening |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 0.903 s | 20 | 36 | 158 | 105 | 204 | 125 | 249 |
| 512 | 1.272 s | 45 | 72 | 199 | 165 | 337 | 188 | 257 |
| 1,024 | 1.833 s | 77 | 137 | 189 | 257 | 450 | 359 | 356 |
| 2,048 | 2.824 s | 159 | 189 | 186 | 445 | 753 | 665 | 413 |

Round 2 is the seeded matrix derivation (P4, about 150 ms, not a PIOP cost). The
`AGG_STAGE_TRACE=1` output of the mode-B bench is swallowed by the driver (`Command::output`
captures the prover child's stderr); run the prover child directly:
`target/clmul/release/deps/mode_b-<hash> --prover-child N --warmup 0 --prove-runs 1
--verify-runs 1 --shared N/32`. Per round at `N = 2,048` (ms, round 0 first):
stage 1 `21 140 113 48 28 17 19 11 8 15 8 4 2 1 1`, stage 2 `34 162 226 115 62 30 22 22 15
31 16 8 4 2 1 1`, stage 3 `42 198 112 66 36 17 13 13 8 17 9 4 2 1 1`; at 1,024: stage 1
`10 92 59 25 16 9 13 8 15 8 4 2 1 1`, stage 2 `17 116 118 61 37 19 13 12 31 16 8 4 2 1 1`,
stage 3 `22 143 57 32 16 9 10 9 17 8 4 2 1 1`. Per group at 2,048 (message / bind, ms):
stage 1 main `eq(tau)` 268 / 124, `D_1 eq(tau)` (generic: `Ints` parity lanes) 19 / 3;
stage 2 main `eq(r1)` 536 / 45, main `w_1` 80 / 32; stage 3 main `eq(r2)` 246 / 71, main
`w_A` 108 / 18, main `eq(r1)` 28 / 30; bridge-domain groups 1 to 15 ms each.

Reading: round 1 (quartet buckets, 65,536 patterns per digit column under the raw
coding) is the largest or second-largest round in every stage. Its cost is not the
arithmetic (about 15 ms of per-pattern work) but the bucket tables: `bucketed_evaluations`
builds one `Sums` of 14 x 65,536 x 16 B = 14.7 MB per rayon split and `reduce(zeroed,
add)` zeroes and adds a fresh table per split (about 16 tables = 235 MB memset + 118 MB
adds at 2,048). Round 2 (first generic round over the materialized `K` lanes at a quarter
of the domain) costs 17 lanes x 5 points x 4 Horner multiplications = 340 `K`
multiplications per pair in stage 2 (3.5 us per pair), plus the materializing fold at the
end of round 1 (34 lanes x 2^19 x 16 B = 285 MB written in stage 1). The dense tail from
round 3 on is 160 / 330 / 190 ms per stage at 2,048, i.e. as large as rounds 0-2 together
after the prefix below is in place. Per committed element the stages cost 17 ns at 32K
(13.4 s / (32,768 x 23,552)) against Aerie's 12 ns (0.60 s / (32,768 x 1,536)): the
remaining gap to Aerie's absolute figure is mostly the 15x element count of the layout
(Section 16.1), so the realistic kernel target is about 2x on stages 1-3.

### Design (prover only; every path computes the same polynomials, proof bytes unchanged)

Aerie's kernels (`aerie-prover/src/protocol/kernels/{quartic,product,range_source}_kernel.rs`,
`kernel_common.rs`) keep the `i8` sources through three active rounds: round 0 buckets
quartet masses with the round-1 equality weights and serves rounds 0 and 1 from them
("bivariate prefix"), round 2 reads a quartet lookup table (folded values of every
quartet pattern) and expands the terms with Taylor coefficients, and only the third bind
materializes `K` lanes at an eighth of the domain. Pattern spaces use the range image
`z(z + 1)` (8 classes for a base-16 digit). The plan for `kernel::Group`:

1. *Column coding by lane maps.* A column's code is the class of its cell under the tuple
   of the maps of the bucketable lanes reading it (`lo_map`, `hi_map`, `square_map` all
   factor through `z(z+1)`, so stages 1 and 2 get 8 codes per digit column, bits 2; stage
   3's links read the identity map, 16 codes). `Column { cells, code: [u8; 256], reps:
   Vec<i8>, pair_bits }`; a column with more than 16 classes is not coded. Coding must be
   by lane *map value*, never by the composed polynomial: `v(v+1)` at the pair `(3, -4)`
   and at `(3, 3)` are different polynomials in the round variable.
2. *Pieces.* Compile as now into per-lane polynomials, same-column bilinears, joint generic
   terms (bucketable when all their lanes are coded) and *direct* pieces evaluated per
   item at every level: every `Term::Affine` becomes `gamma [* w] *` a virtual integer
   lane `Lane::Int { parts: (column, map, i64 coefficient), constant, challenges }` (value
   at level `a` = fold of `2^a` consecutive `i64` cell values by the stored challenges,
   `E::mul_base` for the first level); bilinears with a `Field` weight lane or across two
   columns, and polynomials on non-coded lanes, are direct as well. Keyed buckets
   (`Piece::Weighted`, `WeightConstant`) disappear: the `w_A * z` and `w_1 * b_t` pieces
   cost about 7 `K` multiplications per pair directly and no bucket memory. A group is
   never "all generic" any more; only its direct pieces take the per-item pass.
3. *Lane states.* `Source -> Lookup {pair LUT, n^2} -> Quartet {quartet LUT, 2^(2 pair_bits)}
   -> Folded` (materialized at 1/8 of the domain). LUTs carry the first-fold scale, as now;
   `Int` lanes are not scaled. `terminal()` works in any state (domains of 1 or 2
   variables end as `Lookup`/`Quartet`).
4. *Rounds.* Level 0: if `quartets >= BUCKET_MIN_RATIO * max pattern space` (and at least
   two active rounds) accumulate quartet masses per pure column and joint term with the
   round-1 equality tables (`GruenSplitEq::remaining_eq_tables_after(1)`, exists at Akita
   `e023b6a`), using Aerie's `chunked_bucket_sums` shape (exactly `min(threads, items /
   width)` contiguous chunks, one table each, block-parallel reduction), and derive the
   round-0 polynomial as `l0 * sum_q M_q F(pair p0 of q) + l1 * sum_q M_q F(pair p1 of q)`
   with `l0 = 1 - tau_1, l1 = tau_1` (`tau_1` = the local eq point's second coordinate;
   both 1 for weight groups); keep the masses. Level 1: from the kept masses with the pair
   LUT values (`F(lut[p0] + X (lut[p1] - lut[p0]))`), no pass over the data. Otherwise
   pair masses at level 0 (as today) and per-item at level 1. Level 2: one pass over
   octets; each coded lane's `(left, right)` are two quartet-LUT reads; lane polynomials
   of degree >= 2 on columns with at most 4,096 quartet patterns use a Taylor table built
   at the second bind (`(d + 1)` coefficients per LUT entry, so a lane costs `2d - 1`
   multiplications per octet instead of `d * points`), larger ones the shift below.
   Levels >= 3: per pair over `Folded` lanes.
5. *Coefficient form everywhere.* Pieces return the coefficient vector of the inner
   polynomial `q(X)` (a small fixed-size `Poly`) instead of evaluations at `points`
   points; a lane polynomial of degree `d` at `u + delta X` by Taylor shift (synthetic
   division, `d(d+1)/2` multiplications, then `delta` powers): 14-17 for `d = 4` against
   20, 5-6 for `d = 2` against 6; bilinears 3-4; joint terms by `Poly` arithmetic. The
   message is `UnivariatePoly::new(coefficients)` then `gruen_mul` as now (exact, so
   identical to `from_evals`). `direct_claim` of lifted groups uses `q(0) + q(1)` from the
   same coefficients and caches the level-0 result so the first active round does not
   repeat the pass.
6. *Parity lanes as `i8`.* `ProverTables::parity1` becomes narrow (`Vec<i8>`) when every
   value fits, else the `i32` fallback (`Source::Ints`, uncoded, direct). Then the stage-1
   `D_1 eq(tau)` group (30 lanes, generic today) is bucketable; it matters at 32K
   (`D_1 = 2^19`). Public type change of `tables.rs`; note it here when done.
7. Optional: a faster `K` multiplication (Aerie's `opening_field_ops::mul`, 4 `u128`
   products and two `reduce130` steps; `pub(crate)` in Aerie, so copy it and test against
   `*` on random inputs; `Prime64Offset59::solinas_reduce` and `to_limbs` are public in
   jolt-field `c7c2d08`).

Memory: quartet masses per chunk = sum over pure columns of `2^(2 pair_bits) * 16 B`
(stage 2 main: 17 x 64 KB; stage 3 main: 17 x 1 MiB, 125 MB over 7 chunks; the bivariate
threshold `quartets >= 4 * 65,536` holds from `N = 2,048` on for the stage-3 group), the
quartet LUTs 1 MiB per 16-code lane, and the materialized lanes at 1/8 of the domain
(stage 1 at 32K: 2.3 GB instead of 4.6).

Expected (estimates from the per-round costs above, at `N = 2,048`): stage 1 about 200 ms
(from 445), stage 2 about 350 (from 753), stage 3 about 250 (from 665); the dense tail
then dominates, where only fewer multiplications per lane (item 5, Aerie's unreduced
`sum_split_coefficients` accumulation) remain.

Goal (2), commit and opening (`G_main` 189 ms, opening 413 ms at 2,048): (a) the opening
recomputes the 67 claims with `evaluate_signed_table` (one streaming pass per column,
about 50-70 ms at 2,048 on 7 threads) although they equal the stage-3 terminal values the
prover already has and checks against; take the claims from `stage3.claims` and keep the
recomputation as a debug assertion or test; (b) in mode A, run `commit_round1` and
`keccak.commitments(segments)` under `rayon::join` (both precede `sd`; absorb in event
order afterwards; Aerie commits its groups in a parallel fan-out because an Akita commit
is memory-bound and does not scale across threads); (c) `commit_round2` can commit `G_1,
G_2, G_3` in parallel (the final group needs only the plan's profiles); (d) the catalog
rows are planned under `policy_of::<Cfg>()` (proof-size objective, `akita-config`); there
is no prover-time objective to switch to, and the opening's 200 ms fixed part at 256
records is Akita's `batched_prove`, so no planner change is proposed. None of (a)-(c)
changes a byte of the proof.

### Next steps, in order

1. Rewrite the internals of `crates/agg-piop/src/kernel.rs` per the design (keep the
   public items `Source`, `Lane` constructors, `Term`, `LeafPoly`, `GroupInstance`,
   `Group::{new, claim, round_message, bind, terminal_values, inactive_rounds,
   is_bucketed}`, `compress`, `evaluate_compressed`, `inactive_message`,
   `inverse_power_of_two`, `expected_terminal`, `digit_map`, `IDENTITY_MAP`; `Lane::Lookup`
   changes shape and gains `Quartet`/`Int` variants, used only inside the kernel). Adapt
   the kernel tests and add: coding classes (8 per digit column under `lo`/`hi`/`square`
   maps, 16 under the identity), bivariate masses vs generic path vs definition on random
   tables with every `Term` kind (bit-exact messages and terminals), quartet-LUT and
   octet rounds vs generic, `Int` lanes vs the affine `Term`, Taylor tables vs the shift,
   domains of 1, 2 and 3 variables, the `i32` parity fallback, the wide-cell fallback.
2. `tables.rs`: narrow parity tables (item 6); `prover.rs`: opening claims from stage 3
   (2a), `rayon::join` of round 1 and the Keccak commitments (2b); `pcs.rs`/`commit.rs`:
   parallel round 2 (2c).
3. `taskset -c 0,1,3-15 cargo test --release --workspace` (RUSTUP_TOOLCHAIN=stable), then
   confirm byte-identical aggregates against `50fc538` (the mode-B bench prints the sizes;
   `tests/mode_b.rs` and `tests/prove_verify.rs` verify with the unchanged verifier).
4. Measure: mode B at 256, 512, 1,024, 2,048 (`--warmup 1 --reps 5`) and the prover child
   with `AGG_STAGE_TRACE=1` at 1,024 and 2,048; mode A at 1,024 (`target/clmul/release/agg
   bench --count 1024 --warmup 1 --reps 5`, built with `--features
   agg-keccak/avx512,agg-piop/avx512` as in bench-local.md); report the load average.
5. Docs: update Sections 6 and 17 of this file (round schedule, memory, measured per-round
   table), `docs/deviations.md` P12 (extend, no new wire change), `docs/pcs-config.md`
   Section 5 if the commit driver changes, append a section to `ref/notes/bench-local.md`
   and update the PIOP items of `ref/notes/status.md` (Section 3 row and open problem 4).

### Progress (kernel rework, second agent; updated after each step)

- Step 1 (kernel): done; `src/kernel/` per Section 18 (API notes there). Kernel tests compare
  every strategy against an independent reference bit for bit.
- Step 2: done; `tables::Parity1` (item 6), opening claims from stage 3 (2a), `rayon::join` of
  the round-1 and Keccak commitments (2b, no `agg-cli` edit needed), concurrent round-2
  commits (2c), no planner change (2d); plus the allocation-free `w_A` table (Section 18.1
  item 9).
- Step 3: done. `taskset -c 0,1,3-15 cargo test --release --workspace`: 261 passed, 0 failed,
  10 ignored (`agg-piop` 88 and 3); mode-B aggregates at 256 to 2,048 and mode-A aggregates
  at 256 and 1,024 records byte-identical to `7ed4a32`.
- Step 4: done (Section 18.3; `ref/notes/bench-local.md`, last section).
- Step 5: done (Sections 1, 3, 6, 11, 14, 17, 18; deviations P12; pcs-config Section 5;
  bench-local.md; the PIOP items of status.md).
- Open: the 32K cloud run of this step; the third active round and the materialization at
  the third fold are the largest rounds (deferring the materialization one fold is the
  next candidate, Section 18.2); `agg_bridge::parity` (20 ms at 2,048) and the matrices
  (P4, about 150 ms) are outside this crate.

## 19. Lookup layout L16 (wire version 1, 2026-10-24)

Branch `lookup-integration` (base `4247f61`). `docs/lookup.md` has the argument, the
soundness and the remedies for the identity term; `docs/deviations.md` L1 to L7 and D2 the
departures; `docs/spec.md` Sections 4, 7, 10 to 12 the normative text.

### 19.1 Groups and tables

| Group | Columns (manifest order) | Width | Storage | Round |
| --- | --- | ---: | --- | ---: |
| `G_main16` | `Z_lo, U_lo, V_lo, V_hi` | 16 | one stacked `i32` lane (`MainTables::wide`), column `c` at `c 2^d_main` | 1 |
| `G_main4` | `Z_hi, U_hi, b_0..b_5` | 4 | `i8` columns (`MainTables::columns`) | 1 |
| `G_1` | `kappa1_0..5` (stored `K = kappa - 2^15`) | 16 | stacked `i32`, 8 slots, slots 6, 7 hold `-2^15` | 2 |
| `G_2` | `y2_0..5`, `kappa2_0..5` | 16 | `i8` parities, stacked `i32` carries | 2 |
| `G_3` | `kappa3` | 16 | `i32` | 2 |
| `G_m` | `m_0..m_6` (digits of the multiplicities of `T_16`) | 4 | `i8` | 2 |

`source_tables::Cells` reads a column of either width (`Narrow(&[i8])`, `Wide(&[i32])`);
`Cells::lane` gives a coded byte lane or an `Ints` lane. `MainTables::build` derives every
witness (strict) and encodes `z + 2^17 = z_lo + 2^16 z_hi` (stored `Z_lo = z_lo - 2^15`,
`Z_hi = z_hi - 2`), `u` by relation H-u's step form (`U_lo = B_ulo - 2^15`, `U_hi = H_u - 3`
in base 6, spec (4.2)) and `v` (two 16-bit halves); padded rows hold
the digits of zero. `BridgeGroupTables::build` adds the balanced carries, the parities and
`Y` of `agg_bridge::parity::compute` and the multiplicities of every looked-up cell of both
rounds (`recount`, `lookup::multiplicities`), with their digits. `agg-witness` and
`agg-bridge` are unchanged.

### 19.2 Event-0 bytes and commit flow

`CommitmentManifest::binding_bytes`: wire version 1, digit layout id 1, six groups (role,
round, family byte 0/1, variables, polynomial count, alphabet codes: 4 and 8 for the leaf
digits, `0x90` for a looked-up column, `0x50` for a multiplicity digit, 0 for a Boolean),
the grouped row id `mldsa-44-l16-batch-<N*>-mult-final`, the bridge shape, then the lookup
layout (2 challenges; table width 16; per stack column count, variables, table and shift).
Commit flow: `TableCommitter::commit_round1` commits `G_main16` and `G_main4` concurrently
(the pair runs on the calling thread while the Keccak side's commitments run on a scoped OS
thread, `pcs::overlap`), `commit_round2` the four round-2 groups concurrently, each commit on
its own scoped OS thread (`pcs::concurrent_commits`: Akita's lazily built NTT slots sit
behind a `OnceLock` whose initializer runs rayon loops, so commits joined as rayon jobs can
deadlock; deviation P12). No rayon worker waits on those threads: on a worker, or with a
one-thread pool, the commits run one after another on the calling thread (a worker blocked
on its commit threads hung the one-thread prover in round 1; `tests/single_thread.rs`);
`open_claims` opens the 31 stage-3 claims at the suffixes of
`r^(3)` and the 7 digit claims at `pi_m` (`pcs::group_points`).

### 19.3 The lookup (event 7) and the stages

Shape (`CommitmentManifest::lookup_shape`): stacks `[Z_lo | U_lo | V_lo | V_hi]` (depth
`d_main + 2`), `kappa1` (8 slots, `d_1 + 3`), `kappa2` (8 slots, `d_2 + 3`), `kappa3`
(`d_3`); table `T_16`; challenges `X_1, X_2`; storage shift `2^15` (leaves
`1 / ((X - 2^15) - stored)`). `prover::prove_piop` runs `lookup::prove_in` on the PIOP
transcript, sends the seven digit claims `m_k(pi_m)`, then `tau`, the pre-stage claims and
the stages; `verifier::verify_piop` mirrors it (`VerifyError::Lookup` for the lookup and
the digit check). Lookup proof elements: 1,015 + 312 = 1,327 at `N = 1,024` with two
challenges (21,232 B), 1,799 at `2^15` (28,784 B), plus 112 B of digit claims.

Stage 1 (degree 4, `gamma^(1) in K^26`, point `tau`): `bool_y1[t]` (0..6), `level2[t]`
(6..12), `level3` (12), `bool_y2[t]` (13..19), `bool_b[t]` (19..25), `range_w1` (25); groups:
main `eq(tau)` {6 bits}, `D_1` `eq(tau_1)` {6 `y1`}, `D_2` `eq(tau_2)` {6 `y2`}, `D_1`
{`w_2`, 6 `y1`}, `D_2` {6 `w_3`, 6 `y2`}. Terminal (18): `y1_t(r1_1)`, `y2_t(r1_2)`,
`b_t(r1)`. Pre-stage (13): `y2_t(tau_2)`, `K2_t(tau_2)`, `K3(tau_3)`.

Stage 2 (degree 4 in relation H-u, 5 in relaxed H; `gamma^(2) in K^8`, point `r^(1)`; before `gamma^(2)` the 6 claims
`K1_t(r1_1)`): `def_y1[t]` (0..6, main, lane `w_1`), `leaf4[Z_hi]` (6), `leaf6[U_hi]` (7)
(main `eq(r1)` {`P_{Z_hi}`, `P_{U_hi}`}). Terminal (8): `b_t(r2)`, `P_Z(r2)`.

Stage 3 (degree 3, `gamma^(3) in K^44`, point `r^(2)`):

| Instance | Gamma | Domain | Equality / weight | Summand |
| --- | ---: | --- | --- | --- |
| `ring` | 0 | main | lane `w_A`; and `eq(r2, .)` | `w_A (Z_lo + 2^16 Z_hi + 2^15) - eq (2 gamma_2 w_1' + u + q v)` |
| `link[Z_hi]`, `link[U_hi]` | 1, 2 | main | `eq(r2, .)` | `Z (Z + 1)` |
| `b_t` from `r1`, from `r2` | 3..9, 9..15 | main | `eq(r1)`, `eq(r2)` | `b_t` |
| `y2_t` from `tau_2`, from `r1_2` | 15..21, 21..27 | `D_2` | `eq(tau_2)`, `eq(r1_2)` | `y2_t` |
| `K2_t` from `tau_2` | 27..33 | `D_2` | `eq(tau_2)` | `K2_t` |
| `K3` from `tau_3` | 33 | `D_3` | `eq(tau_3)` | `K3` |
| `K1_t` from `r1_1` | 34..40 | `D_1` | `eq(r1_1)` | `K1_t` |
| stack `j` | 40 + j | stack domain | `eq(pi_j,low)` | `sum_c w_{j,c} f_{j,c}` (one part per column, scale `w_{j,c}`) |

Groups: main `eq(r2)` {`U_lo, V_lo, V_hi, Z_hi, U_hi`, 6 bits}, main {`w_A`, `Z_lo`,
`Z_hi`}, main `eq(r1)` {6 bits}, `D_1` `eq(r1_1)` {6 `K1`}, `D_2` `eq(tau_2)` {6 `y2`, 6
`K2`}, `D_2` `eq(r1_2)` {6 `y2`}, `D_3` `eq(tau_3)` {`K3`}, and one `eq(pi_j,low)` group per
stack. Terminal (31, `Stage3Claims`): every column of the five non-multiplicity groups at
`r^(3)`, manifest order.

### 19.4 Transcript and wire

Registry (`transcript::Registry`): instance; `commit/main16`, `commit/main`; Flock
commitments; seed; `commit/level1..3`, `commit/mult`; Flock seeds; final vectors; Flock
proofs; `lookup/x` (2), `lookup/layer1` (40), `lookup/mu/1`; per session `s = 2 .. D`:
`lookup/gamma/s` (2 per active tree), `lookup/round/s/k` (3) and `lookup/r/s/k`,
`lookup/terminals/s` (4 per continuing tree), `lookup/leaves/s` (2 per ending stack or
table), `lookup/mu/s`; `lookup/multiplicities` (7); `tau`; pre-stage (13); stage 1
(`gamma` 26, rounds, terminal 18); `stage2/kappa1` (6), stage 2 (8, rounds, 8); stage 3 (44,
rounds, 31); PCS session. Wire (`wire::AggregateProof`, version 1): header, `c~`, the
`G_main16` and `G_main4` commitments (framed), the Flock items, the four round-2 commitments,
`Y_sigma`, the Flock proofs, the lookup (layer-1 values, then per session the rounds, the
terminals and the leaves), the 7 digit claims, the pre-stage claims, the stages, the
opening. `ProofSize::lookup` is the lookup part. Every count follows from `N`.

### 19.5 Tests

`cargo test -p agg-piop --release` (67 unit tests, 28 integration tests in 8 files): the
integrated tests of `docs/lookup.md` Section 6 (lookup tampering at the PIOP level in
`tests/stages.rs`; mode-B round trips at 64, 256 and 1,024; end-to-end tampering in
`tests/mode_b.rs`, `tests/prove_verify.rs`, `tests/bridge.rs`). Measurements:
`ref/notes/bench-local.md` (lookup integration) and `docs/lookup.md` Section 4.5.

## 20. Prover speed of the PIOP side (branch `piop-speed`, 2026-10-25)

Plan Part 2, Section 2.2 (`ref/notes/plan-128bit-speed.md`): lookup, stage 3, instance and
witness. Every change is prover only; the aggregates are byte-identical to `main` at 256,
1,024 and 8,192 signatures (sha256 of `agg aggregate` on the same records, `be398d8` against
this branch) and the full workspace suite passes.

### 20.1 Dedicated groups (`kernel::layer`)

The round polynomial of a group is unique, so a group may compute it any exact way.
`kernel::layer` adds a round driver (`Engine`) for groups whose summand is a coefficient-free
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

Field helpers: `fmi(x, k)` (field times a signed integer, two `64 x 64` products, branch
free: the sign of a balanced digit is random) and `mad_int`; the generic kernel folds integer
and byte lanes with `fmi` instead of a full product. Tests:
`kernel::tests::dedicated_layer_groups_match_the_generic_group` (message by message against
`Group` at 1, 2, 5 and 13 variables), `field::tests::{integer_product_matches,
unreduced_integer_products_sum_exactly}`.

### 20.2 Stage driver (`stage::prove_stage_with`)

- A full-domain group without an equality factor whose single instance is one weighted
  affine term with a field weight and integer or byte lanes runs on `WeightIntGroup`
  (`Main eq None`: 76.1 -> 62.9 ms at 1,024, one thread).
- A full-domain equality group whose lanes are all integer tables read only by single-lane
  products, each lane also read by a generic group of the stage, runs on `LinearIntGroup`
  (stage 3: the reduction of the main lookup stack, `Main eq Lookup(0)`, four 16-bit
  columns).
- `split_linear`: in any other full-domain equality group, the single-lane products and
  unweighted affine terms on integer lanes read nowhere else move to one `LinearIntGroup` at
  the same point; the lanes stay in the generic group, unread, so it still folds them and
  yields their terminal values (stage 3 `Main eq R2`: the ring identity's terms on `U_lo,
  V_lo, V_hi`). The two groups' messages sum to the original group's.
- Byte lanes stay on the generic coded path: a linear group over the six bits (`Main eq R1`)
  measured slower than the bucketed generic group (67 against 49 ms at 1,024, one thread).
- `AGG_STAGE_TRACE` prints the construction time per group and attributes split groups to
  their spec group.

### 20.3 Instance, witness, events 3 to 5, scheduling

- `agg_witness::ntt::NttPrime`: the 256-point NTTs (about 66 per record in the key
  expansion, `T`, `A z` by CRT and the strict FIPS verification) used `%` by a run-time
  modulus (a hardware division) in every butterfly. The butterflies now run on 32-bit lanes
  with Shoup products by the constant zetas (the quotient `floor(zeta 2^32 / p)` is
  precomputed; both primes are below `2^31`, so the remainder lies in `[0, 2p)`), the sums
  with branch-free conditional subtractions, and the pointwise products with a Barrett
  reduction (`floor(2^64 / p)`). `ntt_timing` (one core): 6.3 us for the forward transform
  with `%`, 3.7 us with Barrett, 1.8 us with Shoup.
- `agg_witness::derive::derive_with_t`: the witness reuses the instance's `T = c t1 2^d` when
  the signature's `c~` is the instance's (else it recomputes); `public_t` adds `x^p t1` in
  two straight (vectorized) loops instead of a branch per coefficient.
- `agg_bridge::seed::ColumnHasher`: the bridge matrices (about a million columns, prover and
  verifier) on the bare Keccak-f[1600] from the precomputed cSHAKE256 prefix state, one
  permutation per column; equal to the cSHAKE256 path (test
  `keccak_columns_match_the_cshake_columns`).
- `prover::prove_with_tables`: the parity tables `y^(1)` of the stages (no challenge) are
  built on the overlap thread beside the Flock prove of event 6.
- Not done, by design: proving the Flock span concurrently with the lookup and the stages.
  The data are independent, but the transcript is not: `FLOCK_PROOFS_SITE` is absorbed before
  the lookup's `X` (`transcript.rs` registry), and every later challenge is squeezed from the
  sponge that absorbed the Flock proof bytes, so the overlap needs a transcript change (wire
  version, composition argument; profile lever P1), outside a prover-only branch.

### 20.4 Measurements (local, shared machine, direction only)

`agg bench --count N --warmup 1 --reps 3`, CLMUL build (`RUSTFLAGS="-C
target-feature=+pclmulqdq"`), 3 threads on CPUs 0, 1, 3-15, the base `be398d8` and this
branch back to back. Two runs at 8,192 signatures: branch at `9a53721` (load average 5.7 ->
4.6) and at `74d5b69` (with the Shoup NTT and `WeightIntGroup`; load average 4.5 -> 5.4):

| Step | base (`9a53721` run) | `9a53721` | base (`74d5b69` run) | `74d5b69` |
| --- | ---: | ---: | ---: | ---: |
| instance (ExpandA, mu, c, T) | 1,023.1 ms | 643.0 | 1,035.2 | 494.4 |
| witness and `G_main` tables | 1,204.8 | 729.4 | 1,153.0 | 626.9 |
| events 3-5: matrices, round 2 | 518.6 | 343.1 | 557.5 | 380.5 |
| event 7: lookup | 3,822.5 | 2,270.0 | 3,827.2 | 2,551.7 |
| stage 1 / 2 / 3 | 507.5 / 535.4 / 1,835.3 | 351.7 / 506.9 / 1,467.1 | 509.5 / 499.8 / 1,773.0 | 384.4 / 513.7 / 1,502.0 |
| Flock parts (unchanged code) | 11,348.3 | 11,086.4 | 11,803.6 | 11,666.8 |
| prove, median of 3 | 21,853.0 | 18,447.5 | 22,240.9 | 19,256.8 |
| everything but the Flock parts | 10,504.7 | 7,361.0 | 10,437.3 | 7,590.0 |
| verify uncached / cached | 1,162.5 / 684.5 | 913.5 / 538.8 | 1,156.2 / 703.6 | 773.7 / 532.4 |
| prover peak RSS | 7,210 MiB | 7,053 MiB | 7,304 MiB | 6,998 MiB |

The lookup and stage-3 lines of the two runs differ by the load (the code of those steps is
the same apart from `WeightIntGroup`). At 1,024 signatures, `74d5b69` against the base, two
alternated pairs (load average 3.6 to 4.4): prove 4,058 / 4,220 -> 3,537 / 3,709 ms,
everything but the Flock parts 1,946 / 2,021 -> 1,422 / 1,486 ms, instance 126 / 131 -> 58 /
65 ms, witness 142 / 143 -> 79 / 77 ms, lookup 498 / 520 -> 322 / 331 ms, stage 3 232 / 228
-> 194 / 189 ms, verify uncached 491 / 513 -> 314 / 315 ms, cached 435 / 454 -> 280 / 283 ms.

Against the plan's targets at `2^15` on the VM (lookup <= 1.5 s from 3.86 s, stage 3 <=
1.0 s from 1.80 s, instance plus witness <= 1.0 s from 2.16 s), scaling the cloud figures by
the local ratios at 8,192 gives about 2.3 to 2.6 s, 1.5 s and 1.1 s (estimates, not
measured): the directions hold; the instance and witness are close to their target, the
lookup and stage 3 are not. What is left, with estimates: the stage-3 weight `w_A` (about
0.3 s at 8,192 locally: four 256-point negacyclic kernels per record) and the passenger
lanes of the split groups; the lookup items listed at the end of `docs/lookup.md` Section
4.6; and the overlap of the Flock records and commit with the witness (lever P3, 0.3 to 0.5 s
at `2^15`, estimate). Proving the Flock span beside the lookup and the stages (lever P1)
needs a transcript change (Section 20.3).
