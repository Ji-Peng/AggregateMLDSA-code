# PIOP (`agg-piop`): tables, commitments, sum-check stages, transcript and wire format

Part 1 (prover-side tables, commitments, public data, virtual evaluators; spec Sections
2, 4, 5, 7.6), part 2 (the three sum-check stages, the weight tables, the transcript,
the proof wire format, the prover and verifier drivers; spec Sections 6, 7, 10, 11) and
part 3 (the timed, streamed verifier and the mode-B aggregate of milestone M2, `prove_mode_b`
/ `verify_mode_b`; spec Sections 11, 14). Deviations: `docs/deviations.md` entries P1 to P10.

## 1. Modules

| Module | Content | Spec | Aerie counterpart |
| --- | --- | --- | --- |
| `manifest` | `CommitmentManifest`: the four groups (role, round, family, logical and physical variables, polynomial order and alphabets), digit columns, event-0 binding bytes | 4.1, 4.3, 10 | `aerie-falcon/src/protocol/manifest.rs` |
| `source_tables` | `MainTables` (round 1, `G_main`), `BridgeGroupTables` (round 2, `G_1, G_2, G_3` and public `Y`, from `agg_bridge::parity` and `agg_bridge::carry` at `Shape::PRODUCTION`), `SourceTables` view | 4.1, 4.2, 9.2 | `aerie-prover/src/protocol/source_tables.rs` |
| `pcs` | `group_polynomials`, `TableCommitter` (two rounds, batched opening), `opening_claims`, `suffix_points`, `verifier_for` | 4.1, 7.5, 10 | `aerie-prover/src/protocol/pcs/group.rs` |
| `public_data` | `PublicData` (expanded keys, `mu`, `c~`, `T`), `KeyCache`, `t_evaluation`, `a_row_combination`, `negacyclic_kernel` (7.8), `eq_correlation` (Aerie (8.8)-(8.10)) | 2, 7.6 | `aerie-falcon/src/protocol/ring/functional.rs` |
| `virtual_polys` | `w_1'`, (4.1) compositions, ring residual, `y^(1)` parities, `P_Z, A^lo, A^hi`, `MainValues`/`CarryValues` | 4.3, 5 | |
| `kernel` | `Source`/`Lane` (narrow tables lifted at the first fold), `Term` (summand shapes), `Group` (one local domain, optional split equality factor, gamma-weighted instances), compressed messages, lifted inactive rounds | 7.1 | `aerie-prover/src/protocol/kernels/`, `algebra.rs` |
| `range` | digit maps `z -> z(z+1)`, `Q_16^lo(P)`, `Q_16^hi(P)`; the digit orders | 4.3 | `aerie-falcon/src/protocol/relations/quartic.rs` |
| `stage` | `StageSpec` (declarative instance table: groups, lanes, equality points, gamma indices), `prove_stage`, `replay_stage`, `check_terminal` | 7.1 | `aerie-falcon/src/protocol/stage.rs` |
| `stage1`, `stage2`, `stage3` | the instance tables of Section 7.3 to 7.5, claimed sums, terminal claims, `prove`/`verify` | 7.3-7.5 | `stage1.rs`, `stage2.rs`, `stage3.rs` of both Aerie crates |
| `weights` | `w_1, w_2, w_{3,t}` tables and verifier evaluations (wrappers of `agg_bridge::weights` at the production shape); `w_A` table and (7.10) | 7.6 | `ring/functional.rs`, `stage1/projection_link.rs` |
| `claims` | the claim vectors between stages and their wire order | 7.2-7.5 | |
| `tables` | `ProverTables`: borrowed `i8` columns, `y^(1)` as `i32`, the weight tables when known; `LaneId -> Lane` | 7 | |
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
   on a row weight above the cap), `commit_round2` gives `G_1, G_2, G_3` (event 4).
4. After stage 3: `open(commitments, tables, points, session)` evaluates the 67 claims at
   the four suffix points of `r^(3)` and proves one batched opening; the prover checks that
   these equal its stage-3 terminal values. `verifier_for(manifest)` verifies from the
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
weight) and is lifted to `K` only by its first fold, so the peak field memory of a lane is
half its domain. Within a group the instances share one equality factor, kept as Akita's
`GruenSplitEq` (two tables of about `2^(d/2)` entries; the full `eq` table is never
materialized), and their gamma-weighted summands (`Term`s: products, `v(v-1)`, the bit-range
expression, leaf predicates `Q(P)`, links `Z(Z+1)`, affine combinations with an optional
weight lane) are evaluated together at `d` or `d + 1` interpolation points per pair of
cells and interpolated to coefficients (Newton), then multiplied by the equality line of the
current variable (`gruen_mul`) and the lift. Groups without an equality factor carry their
weight as a lane. A group's claim is computed from its own tables (directly for lifted
groups, from `g(0) + g(1)` of round 0 otherwise); the stage driver checks the sum of the
group claims against the gamma-weighted claimed sums at the end (prover self-check; the
test-only `prove_unchecked` paths skip it so tampered witnesses reach the verifier).

Memory at `N = 2^15` (estimates, 16 bytes per `K` element, after the first fold): stage 1
main-domain group 6 bit lanes and 28 `A^lo/A^hi` lanes, `34 * 2^24 * 16 B = 9.1 GB`;
stage 2, 17 `P_Z` lanes plus `w_1` and 6 bit lanes, `6.4 GB`; stage 3, the 23 main lanes
plus `w_A` and 5 digit lanes plus 6 bit lanes, `9.1 GB`; the weight tables `w_1`, `w_A`
are `512 MB` each. The bridge domains are at most `2^19` cells. Compacting the first two
or three rounds through nibble-pair lookups (Akita's `CompactPairFoldLut`, as Aerie's
kernels do) would divide these by four to eight; not done yet (risks R4).

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

Every challenge is at grinding difficulty zero in this stage (deviation P3).

## 10. Wire format (`wire::AggregateProof`)

`LE16` version 0, `u8` flags (0 = H), `LE64 N`; `32 N` bytes `c~`; event 1 framed
(`LE32` length); event 2: `LE32 S` then `S` framed items; event 4: three framed items;
event 6: `512 S` bytes then `S` framed items; events 8 to 11 as raw atom encodings of
16 bytes per element (26 pre-stage claims; `4 d_main` stage-1 round elements and 98
claims; 18 `kappa1` claims, `5 d_main` round elements and 61 claims; `3 d_main` round
elements and 67 claims); the opening framed; no trailing bytes. `from_bytes` derives every count from `N` and rejects
non-canonical limbs. PIOP bytes: `16 (270 + 12 d_main)` = 7,776 at `N = 256`, 8,640 at
`N = 2^15`. `AggregateProof::size()` (`ProofSize`) splits the serialized length into header,
`c~`, Akita commitments, Keccak-side items (event 2), final vectors, Keccak-side proofs,
PIOP and opening, framing included.

## 11. Drivers and hooks

`prover::prove(inputs, bridge, keccak, flock)` derives every witness (strict FIPS 204
verification, `main_tables`) and calls `prove_with_tables`, which runs events 0 to 12:
round-1 commitment, the Keccak commitments (`KeccakSide::commitments`), the bridge seed, the
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
- `tests/prove_verify.rs`: a full prover run at 256 signatures; the blob round-trips and
  verifies from `(pk, M)` with and without cached keys; a shorter or reordered record
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
