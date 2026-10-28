# H-exact: the exact FIPS 204 bounds in the PIOP and the catalogs (design)

| Field | Value |
| --- | --- |
| Applies to | ML-DSA-44, lookup layout L16, modes A and B; relations H-u (byte 0, deviation L8, Section 8), H-exact (byte 1, deviation L9, Section 9) and H-FIPS (byte 2, deviation L10, Section 10) |
| Replaces | spec Appendix A, paragraph "H-exact" |

The uppercase terms MUST, SHOULD and MAY are normative as in BCP 14. Numbers not marked
"estimate" were computed for this document with exact integer arithmetic (Section 4.4 lists
the checks) or read from the pinned code and docs; every cost figure in Section 5 is an
estimate scaled from a measured 32K cloud run of relation H.

## 0. Results in one page

1. **Relation.** H-exact is relation H of spec Section 3 with the honest bounds as the proven
   bounds: `|z|_inf <= B_z = gamma_1 - beta - 1 = 130,993` and `|u|_inf <= B_u = 2 gamma_2 + 1
   = 190,465`, everything else ((3.1), (3.3), `w_1' in [0, 44)`, the `v` range) unchanged. Its
   SelfTargetMSIS bound is `zeta' = max(130,993, 190,465 + 2^12 * 39) = 350,209 = zeta` of
   FIPS 204, so aggregate unforgeability rests on ML-DSA-44's own instance (core-SVP 123.5
   classical).
2. **Flag.** Runtime relation id, byte `1` (H is `0`; byte `2` is H-FIPS, Section 10),
   carried in the proof header's flags byte and in the event-0 manifest bytes; the verifier
   accepts exactly the relation its caller configures (no downgrade to H).
3. **Layout (recommended, "X1").** `z = B_zlo + 2^16 B_zhi + 130,915 a_z - 130,993` with
   `B_zlo in [0, 2^16)` looked up in `T_16` and `B_zhi, a_z` Boolean (spec Appendix A's `z`
   form); `u = B_ulo + 63,079 H_u - 190,465` with `B_ulo` looked up and **one six-valued digit**
   `H_u in [0, 6)` (new: `5 * 63,079 + 2^16 = 2 B_u + 1`), checked by Aerie's leaf/link pair
   with `Q_6(P) = P (P - 2)(P - 6)`. This replaces Appendix A's `u = 118,787 a_u + b_u -
   190,465` (two small columns) by one, so H-exact costs **one** extra main-domain column
   (`a_z`), **+1,024** committed elements per signature, not +2,048.
4. **Groups and catalogs.** `G_main16` keeps 4 polynomials (the two low parts take the slots of
   `Z_lo, U_lo`), so the width-16 catalog is reused unchanged. `G_main4` grows from 8 to 9
   polynomials (`B_zhi, D_u, b_0..b_5, a_z`), so the width-4 catalog needs 13 new scalar rows
   (`G_main4` at `(13..25, 9)`) and 13 new grouped rows (`mldsa-44-l16x-batch-<N*>-mult-final`);
   regenerating only these is an estimated 20 to 35 minutes, not the full 44.
5. **PIOP.** Lookup unchanged (same four stacks, same `L = 138,690,560` at `2^15`). Stage 1
   unchanged. Stage 2: `def_y1[t]` and one `leaf6[D_u]`: 7 instances, **degree 4** (was 8,
   degree 5). Stage 3: `ring` with the affine `z, u`, `link[D_u]`, two Booleanity zero-checks
   `bool[B_zhi]`, `bool[a_z]` at `r^(2)` (degree 3, no new evaluation reductions), the 37
   reductions, 4 stack reductions: 45 instances (was 44). Opening: 32 claims at `r^(3)` plus
   7 multiplicity claims (was 31 + 7).
6. **Proof of work (`docs/soundness.md`).** The `tau` site is unchanged (`nu = d_main` whatever
   the number of zero-checks, round-by-round), the new zero-checks sit at `r^(2)` inside the
   stage-2 rounds (no new site). Changes: `gamma^(2)` has 7 coefficients, `g` 4 -> 3; stage-2
   rounds have degree 4, `g` stays 3; `gamma^(3)` has 45, `g` stays 6. At `2^15`: 483 nonces,
   4,443 bits (was 4,444), 556 bytes (unchanged); round-by-round ledger before work
   `1,409/|K| = 2^-117.5` (was 1,434). The plan digest differs from H's at every `N*`.
7. **Costs at `N = 2^15` (estimates).** 13,434.5 committed elements per signature (+8.3%);
   prove about 23.8 to 24.0 s against the measured 23.74 s (+0.1 to +0.25 s); PIOP bytes
   5,616 (was 6,016), Akita opening +0 to +1.5 KB, so 51.0 B/sig unchanged at one decimal;
   verifier unchanged within noise.
8. **A free intermediate (open question 1).** The SelfTargetMSIS bound does not depend on the
   `z` bound at all (`2^17 < 350,209`). Making only `u` exact ("H-u": H's `z` digits, the
   six-valued `u` digit) gives `zeta' = zeta = 350,209` with **no** extra column, the same
   catalog rows as H, and a cheaper stage 2 (degree 4). The relaxed result could therefore be
   reported at core-SVP 123.5 instead of 119.7 at no cost; H-exact then adds only the exact
   `z` bound (the literal FIPS 204 line-13 bound) for +1,024 elements per signature.

## 1. The relation flag and its binding

### 1.1 Values

| Relation | Id byte | Proven bounds `(B_z bar, B_u bar)` | `zeta'` | Status |
| --- | ---: | --- | ---: | --- |
| H-u (default) | 0 | `(2^17, 190,465)` (inclusive `[-2^17, 2^17 - 1]`, `[-190,465, 190,465]`) | 350,209 | implemented (step A, deviation L8); replaced relaxed H `(2^17, 2^18)`, `zeta' = 421,888`, which used id 0 before wire version 3 |
| H-exact | 1 | `(130,993, 190,465)`, both symmetric and exact | 350,209 | implemented (step B, deviation L9) |
| H-FIPS | 2 | `(130,993, 190,465)` and the hint: `sum h <= omega` and `UseHint(h, w') = w1'` (Section 10) | 350,209 (not needed: the extracted witness is a FIPS 204 signature) | implemented (Sections 10.5 to 10.7) |
| S1+S2 (`fips`, retired sketch) | none (witness-crate layout id 3) | as H-FIPS | 350,209 | witness only, superseded by H-FIPS |

The id equals `agg_witness::tables::Layout::id()` (0 H-u, 1 H-exact; the base-16 S1+S2
sketch serializes as 3) and the `BatchTables` serialization layout byte, so one number names
the relation everywhere; H-FIPS (2) has no base-16 layout (`agg_witness::hfips`).

### 1.2 Runtime parameter, not a cargo feature

The relation MUST be a runtime parameter of `agg_piop::manifest::CommitmentManifest`
(`Relation::{H, HExact}`), so one binary proves and benchmarks both results and the checked-in
catalogs hold the rows of both. The `agg-witness` feature `h-exact` stays as it is (it gates
the wire-version-0 base-16 H-exact layout and its checker, used only by the cross-check of
Section 6.3).

### 1.3 Where it is bound

1. **Proof header** (`wire::AggregateProof`): the flags byte after the version is the relation
   id. `from_bytes` accepts `0` and `1` and rejects any other value (`WireError::Flags`).
2. **Event 0** (`CommitmentManifest::binding_bytes`, absorbed through
   `binding::instance_record` before any commitment):
   * byte 3, the relation flag (today the constant `RELATION_H = 0`), becomes the id;
   * the `G_main4` descriptor: polynomial count 9 instead of 8, alphabet codes
     `[0, 6, 0, 0, 0, 0, 0, 0, 0]` (`B_zhi` Boolean, `D_u` six-valued leaf digit, six bits,
     `a_z` Boolean) instead of `[4, 8, 0, 0, 0, 0, 0, 0]`; the `G_main16` codes stay
     `[0x90; 4]`;
   * the catalog row id `mldsa-44-l16x-batch-<N*>-mult-final` instead of
     `mldsa-44-l16-batch-<N*>-mult-final`;
   * the PIOP plan digest of `docs/soundness.md` 6.4 (the proof-of-work plan), which differs
     because the plan's stage-2 batching site has `g = 3` (Section 3.7).
   The digit layout id stays `1` (it names the lookup layout L16, which is unchanged), and the
   lookup layout bytes are unchanged (Section 3.1). The affine value forms of Section 2.3 are
   a function of the relation id (normative constants), not separately encoded.
3. **Verifier policy.** An H proof is a valid proof of a weaker statement, so the relation
   MUST NOT be read from the proof and trusted: `verifier::verify` takes the expected relation
   (`RelationPolicy::Exactly(Relation)`), rejects a header flag that differs from it before
   replaying event 0, and builds the manifest (hence event 0, the catalog row, the plan) from
   the policy, not from the header. `agg verify --relation {h,h-exact}` (default `h`) prints the
   relation it verified; `agg prove --relation ...` and `agg bench --relation ...` likewise.

No wire-version bump is needed beyond version 2 (the proof-of-work schedule): H proofs are
byte-identical, and
H-exact proofs differ only in the flags byte, in counts the verifier derives from the manifest
(stage-2 message width and claim count, stage-3 claim count) and in the opening.

## 2. Column layout and catalogs

### 2.1 Value forms (verified, Section 4.4)

Let `w = z + 130,993 in [0, 261,986]` and `w_u = u + 190,465 in [0, 380,930]`.

* `z`: the slack lemma with `261,987 =
  130,915 (2^1 - 1) + 2^17`, and `2^17 = 2^16 + 2^16`: `w = 130,915 a_z + 2^16 B_zhi + B_zlo`
  with `a_z, B_zhi in {0, 1}`, `B_zlo in [0, 2^16)`. The four blocks `[0, 65,535]`,
  `[65,536, 131,071]`, `[130,915, 196,450]`, `[196,451, 261,986]` cover `[0, 261,986]`
  without gap; nothing outside is reachable.
* `u`: one digit `H_u in [0, 6)` with step `63,079`: `5 * 63,079 + 2^16 = 380,931 = 2 B_u + 1`
  and `63,079 <= 2^16`, so the six blocks `[63,079 H, 63,079 H + 65,535]` cover `[0, 380,930]`
  exactly: `w_u = 63,079 H_u + B_ulo`, `B_ulo in [0, 2^16)`.
* No single small digit works for `z`: it would need `(c - 1) A = 261,987 - 2^16 = 196,451 =
  151 * 1,301` with `A <= 2^16`, i.e. `c = 152`. For `u` the only solution with `c <= 16` is
  `c = 6`. So `z` needs two small columns and `u` one.

Stored values (what Akita commits): `S_zlo = B_zlo - 2^15`, `S_ulo = B_ulo - 2^15` (the L16
storage shift, deviation L2), `B_zhi`, `a_z` as bits, `D_u = H_u - 3 in [-3, 2]` (balanced, as
every leaf digit). Then

```text
z = S_zlo + 65,536 B_zhi + 130,915 a_z - 98,225        (C_z = 2^15 - 130,993)
u = S_ulo + 63,079 D_u + 31,540                        (C_u = 3 * 63,079 + 2^15 - 190,465)
v = V_lo + 65,536 V_hi + 32,768                        (unchanged)
```

Canonical honest decomposition (the encoder's choice; digits are not unique in the overlaps,
which is harmless because only the value enters an identity): `a_z = [w >= 2^17]` (i.e.
`z >= 79`, the rule `a = [x + offset >= 2^j]` of `docs/witness.md`), `b = w - 130,915 a_z`,
`B_zhi = b >> 16`, `B_zlo = b mod 2^16`; `H_u = min(floor(w_u / 63,079), 5)`,
`B_ulo = w_u - 63,079 H_u`. Padded rows (`z = u = 0`): `(a_z, B_zhi, S_zlo) = (0, 1, 32,689)`,
`(D_u, S_ulo) = (0, -31,540)`, `V_lo = -2^15`, `V_hi = 0`, bits 0. Edge values:
`z = 130,993 -> (1, 1, 32,767)`, `z = -130,993 -> (0, 0, -32,768)`, `u = 190,465 -> (2, 32,767)`,
`u = -190,465 -> (-3, -32,768)`.

### 2.2 Groups

| Group | Family (preset) | H (L16, today) | H-exact | Polys |
| --- | --- | --- | --- | ---: |
| `G_main16` | width 16 (`AggFp64SignedWidth16`) | `Z_lo, U_lo, V_lo, V_hi` | `S_zlo, S_ulo, V_lo, V_hi` (same slots, same alphabet `[-2^15, 2^15)`, all looked up) | 4 |
| `G_main4` | width 4 (`AggFp64SignedWidth4`) | `Z_hi` (base 4, `[-2, 1]`), `U_hi` (base 8, `[-4, 3]`), `b_0..b_5` | `B_zhi` (`{0, 1}`), `D_u` (base 6, `[-3, 2]`), `b_0..b_5`, `a_z` (`{0, 1}`) | 8 -> 9 |
| `G_1, G_2, G_3, G_m` | unchanged | unchanged | unchanged | 6, 12, 1, 7 |

`a_z` is appended after the bits so that `b_t` keeps positions 2..7 of `G_main4` (code that
locates bits by position keeps working; everything else locates through `manifest.locate`).
Every new alphabet lies inside the width-4 admission interval `[-8, 7]`; as always the width
is a producer bound, not a range proof (Akita Remark 10.41): Section 3 proves every range.
Total 39 polynomials (was 38); 32 opened at `r^(3)`, 7 at the table point.

Committed elements per signature: main domain `13 * 1,024 = 13,312` (was 12,288), bridge and
multiplicities unchanged (96 + 12 + 0.5 + 14 at `2^15`): **13,434.5** at `N = 2^15` (was
12,410.5, +8.25%); 13,960 physical at `N = 1,024` (was 12,936).

Rejected alternatives: `a_z` in `G_main16` (5 width-16 polynomials) needs new width-16 rows
instead and commits a bit at width 16; a fifth main-stack column would double the main
stack to 8 slots (`L` from `138.7M` to `272.9M` cells at `2^15`) and cost far more than a
bit; Appendix A's `u` form (`a_u` plus a base-4 digit) costs a second column (Appendix X
compares).

### 2.3 Catalogs

A catalog row is keyed by the ordered physical layouts (`agg_pcs_config::catalogs`
module doc), so:

* width-16 family: no new row (`G_main16` stays `(10 + d_sig, 4)`; `G_1, G_2, G_3` unchanged);
* width-4 family: 13 new scalar producer rows, `G_main4` at `(13..=25, 9)`, and 13 new grouped
  rows, one per `N* = 2^3 .. 2^15`, keyed by `G_m = (16, 7)` and the five precommitted
  profiles `(G_main16 (v, 4), G_main4 (v, 9), G_1, G_2, G_3)`, named
  `mldsa-44-l16x-batch-<N*>-mult-final`. H's 26 width-4 rows and 25 width-16 rows stay
  byte-identical (H's plan digests and pinned proofs do not move).

Generation: `update_catalog_rows` plans only missing rows but **prunes rows of layouts no
longer requested**, so `CATALOG_D_SIG` must become a list of `(d_sig, relation)` pairs (or
`scalar_layouts`/`BatchLayout` must iterate both relations) before it is run, or H's rows are
deleted. Estimated cost: 20 to 35 minutes with 3 scalar workers off CPU 2 (estimate: the
run of all rows of both families took 2,666 s; the width-16 rows, about a third to
a half of the scalar work, are reused). The width-4 artifact grows from 331,178 B by about
the same amount again (estimate). Akita derives its transcript proof-of-work plan from the
row, so the 13 new rows get their own Akita plan digests (the pin test of `docs/soundness.md`
4.1 gains 13 entries); the evaluation batch has 39 claims, `g = 6` as for 38.

The planner's opening estimate barely depends on the polynomial count (`docs/pcs-config.md`
Section 4: the 67-polynomial base-16 layout was within 2.4% of the 38-polynomial L16 layout),
so one more polynomial should cost between 0 and about 1.5 KB of opening (estimate; the
planner may switch schedule at a size, as at `d_sig` 9 and 11).

## 3. PIOP changes

### 3.1 Lookup (event 7): unchanged shape

The four stacks keep their column counts and domains: `[S_zlo | S_ulo | V_lo | V_hi]` on the
main domain (depth `d_main + 2`), the carries on `D_1, D_2, D_3`; one table `T_16`, two
challenges, storage shift `2^15`. Hence `L = 4 2^d_main + 8 2^d_1 + 8 2^d_2 + 2^d_3` is
unchanged (138,690,560 at `2^15`, 4,395,008 at 1,024), `D = max(d_main + 2, 16)` is unchanged,
the lookup proof (1,799 elements plus 7 digit claims, 28,896 B at `2^15`) and all lookup sites
are unchanged. Only the honest multiplicities move (padded cells now look up 65,457 and
1,228 instead of 0); the seven multiplicity digits still suffice because `L` is unchanged. The
lookup now proves `B_zlo, B_ulo in [0, 2^16)` instead of `z_lo, u_lo in [0, 2^16)`: the same
cells, other meaning (bound by the relation id).

### 3.2 Stage 1 (point `tau`): unchanged

26 instances, degree 4, the 13 pre-stage and 18 terminal claims unchanged. The two new
Booleanity zero-checks could go here (`bool[B_zhi]`, `bool[a_z]` at `tau`, degree 3, 28
instances, `g` of `gamma^(1)` still 5), but then each leaves a claim at `r^(1)` that needs an
evaluation reduction in stage 3; placing them in stage 3 at `r^(2)` (3.4) needs none.

### 3.3 Stage 2 (point `r^(1)`): one leaf of degree 4

Instances, in `gamma^(2)` order: `def_y1[t]` for `t < 6` (gammas 0..5, unchanged, degree 2)
and `leaf6[D_u]` (gamma 6):

```text
0 = sum_x eq(r^(1), x) Q_6(P_D(x)),   P_D = D_u (D_u + 1) (virtual, extended separately),
Q_6(P) = P (P - 2)(P - 6).                                             degree 1 + 3 = 4
```

`leaf4[Z_hi]` and `leaf8[U_hi]` are removed (`B_zhi` is Boolean, checked in stage 3). Stage
degree 4 (was 5), 7 batching coefficients (was 8), terminal claims `b_t(r^(2))` (6) and
`P_D(r^(2))` (1): 7 (was 8). `Q_6` is exact over `F_p`: `Z (Z + 1) = t` has at most two roots,
and for `t in {0, 2, 6}` the roots are `{0, -1}, {1, -2}, {2, -3}`, so `Q_6(Z (Z + 1)) = 0`
iff `Z in [-3, 2]` (the halving trick of Aerie (3.7) works for every even alphabet symmetric
about `-1/2`). `Q_6` is `Q_8` without its last root.

### 3.4 Stage 3 (point `r^(2)`): affine ring, one link, two Booleanity checks

Instances, in `gamma^(3)` order:

| Gamma | Instance | Domain, group | Summand | Degree |
| ---: | --- | --- | --- | ---: |
| 0 | `ring` | main; lane `w_A` and `eq(r^(2))` | (7.6) with the forms of 2.1 (below) | 2 |
| 1 | `link[D_u]` | main, `eq(r^(2))` | `D_u (D_u + 1)`, input claim `P_D(r^(2))` | 3 |
| 2 | `bool[B_zhi]` | main, `eq(r^(2))` | `B_zhi (B_zhi - 1)`, claim 0 | 3 |
| 3 | `bool[a_z]` | main, `eq(r^(2))` | `a_z (a_z - 1)`, claim 0 | 3 |
| 4..40 | the 37 evaluation reductions | as today | as today | 2 |
| 41..44 | the 4 lookup-stack reductions (7.7) | as today | as today | 2 |

`GAMMA_COUNT = 45` (was 44: one link fewer, two zero-checks more). The ring identity reads

```text
T(r^(2)) = sum_{j',s,i} w_A(j',s,i) * (S_zlo + 65,536 B_zhi + 130,915 a_z - 98,225)(j',s,i)
         - sum_{j,rho,i} eq(r^(2),(j,rho,i)) * (2 gamma_2 w_1' + (S_ulo + 63,079 D_u + 31,540)
                                               + q (V_lo + 65,536 V_hi + 32,768))(j,rho,i)
```

still degree 2 (a weight times an affine combination; `Term::Affine` with coefficients
`(1, 65,536, 130,915)` and constant `-98,225` for `z`, `(1, 63,079)` and `31,540` for `u`).
The constants are exact because the multilinear extension of an affine combination of tables
plus a constant is the same combination of the extensions plus the constant. The
linear split (`stage::split_linear`) takes only `Ints` lanes, i.e. `S_zlo, S_ulo, V_lo,
V_hi`; the narrow lanes `B_zhi, a_z, D_u` stay in the generic groups exactly as `Z_hi, U_hi`
do today.

Groups: main `eq(r^(2))` `{S_ulo, V_lo, V_hi, B_zhi, D_u, a_z, b_0..b_5}` (12 lanes, was 11);
main `{w_A, S_zlo, B_zhi, a_z}` (4 lanes, was 3); the others unchanged. Terminal: the 32
committed values at `r^(3)` and its suffixes (manifest order); the verifier forms `z, u, v`
by 2.1, `w_1'`, `w_A(r^(3))`, `D_u (D_u + 1)` for the link and `B (B - 1)` for the two
zero-checks, and checks the batched terminal. The Booleanity zero-checks need no new
reduction: `B_zhi` and `a_z` are opened at `r^(3)` anyway.

Soundness of the placement: `B_zhi, a_z` are committed in round 1, long before `r^(2)` is
drawn by the stage-2 rounds, so `sum_x eq(r^(2), x) B(x)(B(x) - 1)` is, as a function of
`r^(2)`, a fixed multilinear polynomial that is nonzero whenever some cell is not a bit; this
is the argument for the links and the ring identity at the same point (spec 12.1 "point
reuse", `docs/soundness.md` 3.4 "identities tested at a point of the preceding stage").

### 3.5 Stage degrees and transcript counts

| | H | H-exact |
| --- | --- | --- |
| Stage degrees (`STAGE_DEGREES`) | (4, 5, 3) | (4, 4, 3) |
| Batching coefficients (`STAGE_GAMMAS`) | (26, 8, 44) | (26, 7, 45) |
| Pre-stage, stage-1, `K^(1)`, stage-2, stage-3 claims | 13, 18, 6, 8, 31 (76) | 13, 18, 6, 7, 32 (76) |
| Opening claims | 31 at `r^(3)` + 7 at `pi_m` | 32 + 7 |
| Stage messages at `2^15` | `(4 + 5 + 3) * 25` = 300 elements | `(4 + 4 + 3) * 25` = 275 |
| PIOP stage bytes at `2^15` | 6,016 | 5,616 |

Events of spec Section 10 for H-exact: event 11 `gamma^(2) in K^7`, stage-2 rounds of 4
elements, 7 claims; event 12 `gamma^(3) in K^45`, rounds of 3 elements, 32 claims. Every count
follows from `N` and the relation; `transcript::Registry` MUST take both from the manifest
(today `STAGE_DEGREES` and the claim counts are constants).

### 3.6 Instances that change, by mechanism

| Value | Mechanism in H | Mechanism in H-exact |
| --- | --- | --- |
| `z` low part | `Z_lo` looked up | `S_zlo` looked up (same stack slot) |
| `z` high part | `Z_hi in [-2, 1]`: `leaf4` (stage 2), `link` (stage 3) | `B_zhi in {0, 1}`: `bool` at `r^(2)` (stage 3) |
| `z` slack | none | `a_z in {0, 1}`: `bool` at `r^(2)` (stage 3) |
| `u` low part | `U_lo` looked up | `S_ulo` looked up (same stack slot) |
| `u` high part | `U_hi in [-4, 3]`: `leaf8` (degree 5), `link` | `D_u in [-3, 2]`: `leaf6` (degree 4), `link` |
| `v`, bits, carries, parities | unchanged | unchanged |

### 3.7 Proof-of-work schedule

Under the normative assigned-bound policy of `docs/soundness.md` Section 1 a new zero-check
or range instance can change a difficulty only through (i) the `tau` site, (ii) a batching
vector's length `l` (`e = l/|K|`, `g = bit_length(l)`), (iii) a stage's round degree
(`e = deg/|K|`), or (iv) a new site. For H-exact:

| Site class | H | H-exact | Why |
| --- | --- | --- | --- |
| `tau` (`nu = d_main`) | `g = 5` at `2^15` | unchanged | one site for all zero-checks at `tau` (round-by-round); H-exact adds none there, and would not change `nu` if it did |
| `stage.gamma` stage 1 | `l = 26`, `g = 5` | unchanged | |
| `stage.gamma` stage 2 | `l = 8`, `g = 4`, nonce 11 bits | `l = 7`, **`g = 3`**, nonce 10 bits | `leaf4[Z_hi]`, `leaf8[U_hi]` -> `leaf6[D_u]` |
| `stage.gamma` stage 3 | `l = 44`, `g = 6` | `l = 45`, `g = 6` | below 64 |
| `stage.round` stage 2 (`d_main` sites) | degree 5, `g = 3` | degree 4, `g = 3` | `bit_length(4) = bit_length(5)` |
| `stage.round` stages 1, 3 | degrees 4, 3; `g` = 3, 2 | unchanged | |
| new zero-checks `bool[B_zhi]`, `bool[a_z]` at `r^(2)` | | no site | ride on the stage-2 rounds' `nu >= 1` and on `gamma^(3)` (`docs/soundness.md` 3.4) |
| lookup (`X`, `mu`, `gamma_s`, rounds) | | unchanged | same trees |
| Akita (inside the opening) | per H row | per new row | the plan is derived from the selected catalog row |
| tie site, Flock | | unchanged | the Keccak side does not see the relation |

Totals (`docs/scripts/pow_plan.py`, run with `STAGE_DEGREES = (4, 4, 3)` and
`STAGE_GAMMAS = (26, 7, 45)`):

| `N*` | H: bits / bytes / work / ledger | H-exact: bits / bytes / work / ledger |
| ---: | --- | --- |
| `2^3` | 1,811 / 227 / 1,364 / `563/|K|` | 1,810 / 227 / 1,356 / `550/|K|` |
| `2^10` | 3,124 / 391 / 2,048 / `999/|K|` | 3,123 / 391 / 2,040 / `979/|K|` |
| `2^13` | 3,889 / 487 / 2,438 / `1,251/|K|` | 3,888 / 486 / 2,430 / `1,228/|K|` |
| `2^15` | 4,444 / 556 / 2,718 / `1,434/|K|` | 4,443 / 556 / 2,710 / `1,409/|K|` |

(`ledger` = round-by-round sum before work; nonce counts are equal, 194 to 483.) The plan
encoding of `docs/soundness.md` 6.4 needs no new field: the relation is bound by the manifest
bytes in the same event, and the stage-2 entry already makes the digests differ. Anchors
(unchanged encoding, `N = N*`, default part size; `pow_plan.py --relation h-exact` prints
them):

```text
H-exact  N* = 2^3   modeB          c9dd30a002089fff68b1653faddb2ac860e093f771eb3eeaaf03e60152e54c85
H-exact  N* = 2^3   slim span on   395afaec30fb8143597d9cd965d9348f8aba6367d5ea3c931338a519f3538f92
H-exact  N* = 2^10  modeB          750a40c19877ba6bda8c7ddee36866239b6338f7caa84109880662a33d64a1d8
H-exact  N* = 2^10  slim span on   50f288e92a2e1f8832ee42535ca855bf2883786f79fb1629313f5a7951a36187
H-exact  N* = 2^15  modeB          e4b1dffc21351ca615ace27500406a1990609ccb6cc07e13c6da50c18a416046
H-exact  N* = 2^15  slim span on   1f8e7aba1af1db61d3be14fccd16252008b8cdc71799c61d5711d923df38db16
```

(The same computation reproduces the H anchors of `docs/soundness.md` Section 10, e.g.
`df9161a5...e2f38c` at `2^15` Slim span on.)

## 4. Knowledge soundness and the exact bounds

### 4.1 Ledger delta at `N = 2^15`

Spec 12.1 (per-instance union bound):

| Component | H | H-exact |
| --- | ---: | ---: |
| LogUp identity, two challenges | `2^-201.9` | `2^-201.9` (same `L`, `M`) |
| GKR sessions | `1,106/|K|` | `1,106/|K|` |
| Stage sum-checks, degrees (4, 5, 3) -> (4, 4, 3) | `300/|K|` | `275/|K|` |
| Zero-checks at `tau` (19) | `475/|K|` | `475/|K|` |
| Leaves at `r^(1)`, links and new zero-checks at `r^(2)` | 2 + 2 instances, `100/|K|` | 1 + 1 + 2 instances, `100/|K|` |
| Linear identities at a point (62) | `1,550/|K|` | `1,550/|K|` |
| Batching coefficients | `3/|K|` | `3/|K|` |
| **Sum of field terms** | `3,534/|K| = 2^-116.2` | `3,509/|K| = 2^-116.2` |
| Bridge | `2^-144.6` | `2^-144.6` |
| Akita, Flock | by their ledgers | Akita by the new rows' ledger; Flock unchanged |

`docs/soundness.md` Section 5 (round-by-round, the ledger that replaces 12.1):
`(sum of stage degrees * d_main + d_main + 3 + GKR)/|K|` = `(275 + 25 + 3 + 1,106)/|K| =
1,409/|K| = 2^-117.5` (H: 1,434). With the schedule of 3.7 every site weighs at most
`2^-128` per address (the maximum site weight is the script's `2^-128.36` at `2^15`, as for H),
so the theorem of `docs/soundness.md` Section 7 holds verbatim with "relation H" replaced by
"relation H-exact" and the H-exact rows' Akita plan.

### 4.2 Extraction

From an accepting transcript the Akita extractor yields the 39 committed tables as `F_p`
values with no range guarantee (Akita Remark 10.41). Except with the terms of 4.1:

1. Lookup: every looked-up cell is an integer entry of `T_16` after the shift, so `S_zlo,
   S_ulo, V_lo, V_hi in [-2^15, 2^15)` and every carry in `[0, 2^16)` (unchanged).
2. `bool[B_zhi]`, `bool[a_z]`: `B(x)(B(x) - 1) = 0` on the main domain, so `B_zhi, a_z in
   {0, 1}` (a field element with `b^2 = b` is 0 or 1).
3. `leaf6[D_u]` and `link[D_u]`: the link ties the leaf's terminal claim to the multilinear
   extension of the table `D_u (D_u + 1)` (Aerie (7.8)), the leaf then gives
   `Q_6(D_u (D_u + 1)) = 0` on the main domain, so `D_u in [-3, 2]` by 3.3.
4. By 2.1 the field values `z = S_zlo + 65,536 B_zhi + 130,915 a_z - 98,225` and
   `u = S_ulo + 63,079 D_u + 31,540` are integers with `z in [-130,993, 130,993]` and
   `u in [-190,465, 190,465]` (the union of the blocks is exactly these intervals, and all are
   far below `p/2`, so the field element is the integer).
5. Unchanged: `v in [-2^31, 2^31)`, `b_t, y^(2)_t in {0, 1}`, (3.4) on every position, (9.2)
   over `Z`, `Delta = 0` in `F_p`; the lifting bound (6.1) holds a fortiori (`|Delta^Z| <
   2^54.05 < p/2` with the H-exact bounds), so (3.1) holds over `Z`; the Flock extractor and the
   bridge give (3.3).

Hence every live record satisfies `H(130,993, 190,465)`, i.e. H-exact.

### 4.3 The bounds equal FIPS 204's, and the reduction uses `zeta = 350,209`

* `z`: FIPS 204 Algorithm 8 line 13 accepts iff `||z||_inf < gamma_1 - beta`; with `gamma_1 =
  2^17`, `beta = tau eta = 39 * 2 = 78` (FIPS 204 Table 1, read with `pdftotext -layout`) the
  accepted set is exactly `|z| <= 130,993`, the H-exact bound. No smaller bound is complete:
  the pinned record `gen::record_hedged(0x2A4A, 0, 290)` has `|z|_inf = 130,993`.
* `u`: not a FIPS 204 quantity but the residual its verification implies, `u =
  ctr_q(w' - 2 gamma_2 UseHint(h, w'))`; the exhaustive check over `Z_q x {0, 1}`
  (`docs/scripts/spec_checks.py` item 2) gives maximum exactly `2 gamma_2 + 1 = 190,465`
  (`gamma_2 = (q - 1)/88 = 95,232`), the H-exact bound, attained, so no smaller bound is
  complete either.
* `w_1' in [0, 44)` (3.4) and the hash (3.3): unchanged, exact in both relations.

So H-exact is the image of FIPS 204 verification on the quantities that enter the security
proof: FIPS verification implies H-exact (spec 3.2, unchanged), and H-exact certifies no
larger `z` or `u`. It is still hint-free: it does not certify `sum h <= omega` and does not
re-encode to a FIPS signature (that is S1+S2's black-box reduction).

The SelfTargetMSIS solution of spec 3.3, `(z, -c, c t_0 - u)` for `[A | t | I]`, has
`|c t_0|_inf <= tau 2^(d-1) = 39 * 4,096 = 159,744`, so

```text
zeta'(H-exact) = max(B_z, B_u + 2^(d-1) tau) = max(130,993, 190,465 + 159,744) = 350,209
zeta (FIPS 204) = max(gamma_1 - beta, 2 gamma_2 + 1 + 2^(d-1) tau) = max(130,994, 350,209) = 350,209
```

and the reduction of spec 12.3 runs with ML-DSA-44's own instance: core-SVP 123.5 classical
(lattice-estimator rough model, block size 423; 112 quantum by security-estimates), the
published Dilithium2 figures; the "+20.5%" sentence of spec 12.3 and
its estimator obligation do not apply to H-exact. (FIPS 204 Table 1 claims Category 2 for
ML-DSA-44; spec 12.3's "category 1" is H's.)

### 4.4 Checks run for this document

Exact integer checks (`docs/scripts/spec_checks.py` items 12 and 13, `python3 -I`):

* the reachable sets of both forms of 2.1 equal `[-130,993, 130,993]` and `[-190,465,
  190,465]` (full enumeration of all digit tuples), and `130,994`, `190,466` are unreachable;
  Appendix A's `u` form also gives exactly `[-190,465, 190,465]`;
* `C_z = -98,225`, `C_u = 31,540`; the canonical decomposition round-trips for every `z` and
  `u` in range with every digit in its alphabet; padded and edge encodings as in 2.1;
* `Q_6(Z (Z + 1)) = 0` iff `Z in [-3, 2]` on `[-200, 200)`;
* lifting: `|Delta^Z| < 2^54.05 < 2^63`;
* `zeta', zeta` as above; `B_z, B_u, gamma_2, beta` from FIPS 204 Table 1;
* the proof-of-work totals and digests of 3.7 with `docs/scripts/pow_plan.py`.

## 5. Costs at `N = 2^15` (estimates)

Base: a measured run of relation H at `N = 2^15` on `c4d-highmem-16` (`Slim`, span on,
medians of 5), before the prover speed work of `docs/perf.md` and `docs/piop.md` Section 20.
The deltas are estimates from the structure of each step (which lanes and polynomials it
touches); they are not measurements. They apply in absolute terms on top of that speed work
(which does not touch the parts H-exact changes, except stage 3, where it makes the `Ints`
lanes cheaper and leaves the narrow lanes as they are).

### 5.1 Committed elements and memory

| | H (measured layout) | H-exact |
| --- | ---: | ---: |
| Main-domain columns | 12 (4 width 16 + 8 width 4) | 13 (4 + 9) |
| Committed elements per signature | 12,410.5 | 13,434.5 (+8.25%) |
| Committed elements, total | 406.7M | 440.2M |
| Prover RSS | 28.9 GiB | +0.1 to +0.3 GiB (one `i8` column of `2^25` cells, its Akita commit data; estimate) |

### 5.2 Prover time

| Step | H (ms) | H-exact (estimate, ms) | Reason |
| --- | ---: | ---: | --- |
| witness and tables | 1,103.9 | +10 to +30 | one more narrow column to encode and validate |
| event 1 `G_main` commits | 602.8 | +35 to +50 | 13 instead of 12 columns, one width-4 digit each |
| Flock span (commit + prove) | 13,816.4 | 0 | Keccak side unchanged |
| LogUp-GKR lookup | 3,861.7 | 0 | same stacks and `L` |
| stage 1 | 466.4 | 0 | unchanged |
| stage 2 | 483.6 | -60 to -120 | stage degree 5 -> 4 for every group (one evaluation point fewer per round), one mapped leaf lane instead of two |
| stage 3 | 1,795.6 | +60 to +150 | `a_z` lane in two main groups, two `Bool` terms instead of one `Link`, a third lane in the ring's `w_A` group |
| Akita opening | 737.6 | +10 to +20 | 39 instead of 38 polynomials |
| **prove total** | **23,742.5** | **+55 to +130 (about +0.1 s; range -0.05 to +0.25 s)** | |

So prove about 23.8 to 24.0 s at this base, +0.5% to +1%; against a 12-s prove the same
absolute delta is +1% to +2%.

### 5.3 Proof bytes

| Part | H (B) | H-exact (B) |
| --- | ---: | ---: |
| `c~` list | 1,048,576 | 1,048,576 |
| Flock proofs, commitments, final vectors | 522,458 | 522,458 |
| lookup `T_16` | 28,896 | 28,896 |
| PIOP rounds and claims | 6,016 | 5,616 (exact count, 3.5) |
| proof-of-work stream | 556 | 556 |
| Akita commitments | 792 | about 792 (estimate) |
| Akita opening | 62,974 | 63.0 to 64.5 K (estimate, planner) |
| **total** | **1,669,723 (51.0 B/sig)**, plus 560 with the proof-of-work stream | **about 1,669.9 to 1,671.4 K with the proof-of-work stream (51.0 B/sig)** |

### 5.4 Verifier

Unchanged within noise (estimate): three integer coefficients more at the stage-3 terminal,
one claim more in the Akita opening (9.1 ms today), the same lookup, public data and Flock
work. 0.92 s uncached, 0.46 s cached as measured for H.

## 6. Test plan

Two steps: step A (value forms, `Q_6`, `u` exact: relation H-u, Section 8) and step B (the `z`
slack, `G_main4` count 9, catalogs: Section 9).

### 6.2 Honest round trips

* `agg-piop` mode B: H-exact at 64, 256, 1,024 records (`tests/mode_b.rs`), also with
  `N` not a power of two (padding) and `N = 1`; H unchanged at the same sizes (pinned digests
  of H MUST NOT move).
* `agg-cli` mode A: H-exact at 256 and 1,024 locally (`tests/mode_a.rs`, `tests/cli.rs`),
  cached and uncached verifier paths agree; 32,768 on the cloud VM (both relations back to
  back, same VM, medians of 5).
* Edge records: the pinned `|z| = B_z` record and the Decompose edge record prove and verify
  under H-exact; extend `edge_search` to find a record with a coefficient `|u| = B_u`
  (rare; if none is found within a bounded search, the synthetic statements of 6.4 cover it).

### 6.3 Cross-check with the witness checker

* For 128 random signatures (as `agg-witness/tests/witness.rs`): `agg_witness::check` with
  `Layout::HExact` (feature `h-exact`, dev-dependency of `agg-piop`) passes, and the `(z, u, v,
  b)` decoded from the PIOP's H-exact tables equal those of `derive` and those composed from
  the witness crate's base-16 H-exact digits (different digits, equal values).
* `tests/virtual_checker.rs`: the pointwise H-exact identities on the Boolean domain (both
  `bool`, `Q_6(P_D) = 0`, `P_D = D_u (D_u + 1)`, the ring with the forms) on the honest
  tables, and each tampering of 6.4 caught pointwise by the identity it targets.

### 6.4 Negative tests (each MUST be rejected, at the stated place)

Instance level (`tests/stages.rs`, tables tampered after commitment, stage prover run
unchecked; the stage verifier rejects):

| Tamper | Rejected by |
| --- | --- |
| `a_z = 2` at one live cell | stage 3 (`bool[a_z]`) |
| `B_zhi = 2`, or `B_zhi = -1` | stage 3 (`bool[B_zhi]`) |
| `D_u = 3` or `D_u = -4` (`P_D = 12`) | stage 2 (`leaf6`) |
| `P_D` claim inconsistent with `D_u` | stage 3 (`link[D_u]`) |
| `S_zlo = 2^15` or `S_ulo = -2^15 - 1` | lookup (event 7) |

Value boundaries, where the tamper keeps the ring identity true so only the range mechanism
can catch it (synthetic mode-B statements through a test-only `PublicData` constructor that
takes `A, T` and the bits directly, `#[cfg(test)]` or a `test-hooks` feature; `T` chosen as
`A z - 2 gamma_2 w_1' - u - q v` for the chosen witness):

| Witness | H | H-exact |
| --- | --- | --- |
| `z = +-130,993`, `u = +-190,465` (canonical digits) | accept | accept |
| `z = 130,994` encoded as `(a_z, B_zhi, S_zlo) = (1, 1, 32,768)` | accept (`|z| < 2^17`) | reject: lookup |
| `z = 130,994` encoded with `B_zhi = 2` or `a_z = 2` (`S` in range) | | reject: `bool` |
| `z = -130,994` encoded with `S_zlo = -32,769` | | reject: lookup |
| `u = 190,466` as `(D_u, S_ulo) = (2, 32,768)` | accept (`|u| < 2^18`) | reject: lookup |
| `u = 190,466` as `D_u = 3` (`S_ulo = -30,311`) | | reject: `leaf6` |
| `z = 131,000` (FIPS rejects, H accepts) | accept | reject |

Protocol level: an H proof verified with `RelationPolicy::Exactly(HExact)` and an H-exact
proof verified as H are rejected before event 0 (flag mismatch); an H-exact proof whose
header flag is rewritten to 0 is rejected (and vice versa); an H-exact proof whose flag is
rewritten to 2 (H-FIPS, Section 10) is not accepted under either relation, and flag byte 3 is
rejected as malformed;
an H-exact proof replayed with H's catalog row or H's plan digest fails event 0 (different
transcript); a proof-of-work nonce wrong at the stage-2 `gamma` site (10-bit nonce) is
rejected; a truncated stage-2 message (H-exact width 4) is rejected.

Catalog: `checked_in_catalog_holds_exactly_the_requested_rows` with both relations;
`manifest_matches_the_work_plan` for both; the Akita plan-digest pin for the 13 new rows.

## 7. Open questions

1. **Exact `u` for free (H-u).** `zeta'` depends only on the `u` bound (`B_z bar = 2^17 <
   350,209`), and the six-valued digit makes `u` exact with the same two columns as H. H with
   that `u` digit (relation H-u: `z in [-2^17, 2^17)`, `u` exact) has `zeta' = 350,209`, the
   same physical layouts `(v, 4)`, `(v, 8)` as H (no new catalog rows; only the alphabet code
   of `U_hi` 8 -> 6 and the relation byte change), stage-2 degree 4 instead of 5 (about
   -60 to -120 ms, estimate), PIOP bytes 5,616, and the same proof-of-work totals as H (`gamma^(2)`
   keeps 8 coefficients, degree-4 rounds keep `g = 3`). Options: (a) replace H's `u` digit (H's `zeta'` becomes
   350,209 and the relaxed result is reported at core-SVP 123.5; H's pinned digests move);
   (b) add H-u as relation id 3 and report three rows; (c) keep H as is. Recommendation:
   (a), since it dominates H in security and cost; then H-exact differs from the default only
   by the literal `z` bound (+1,024 elements per signature), which matters for a "the exact
   FIPS 204 relation" statement but not for MSIS hardness. Answered (a): Section 8.
2. **Appendix A vs X1.** This design departs from spec Appendix A for `u` (one six-valued digit
   instead of `a_u` plus a base-4 digit; Appendix X). If Appendix A must be kept literally,
   Appendix X gives the exact deltas (+2,048 elements, `G_main4` at 10 polynomials, stage-2
   degree 3).
3. **Booleanity placement.** Stage 3 at `r^(2)` (chosen) vs stage 1 at `tau` (two more
   reductions). Both are sound; the choice is a cost question only.
4. **Catalog regeneration placement.** Done locally off CPU 2 (1,617 s, Section 9).
5. **Plan encoding.** Keep `docs/soundness.md` 6.4 unchanged (chosen; the relation is bound by
   the manifest bytes in the same event) or add a relation byte to the plan body (moves H's
   anchors).
6. **`v` exactness.** H-exact keeps `v in [-2^31, 2^31)` (the honest bound is `|v| <=
   67,068,447 < 2^26`). `v` is not part of any FIPS 204 check or of `zeta`; a tighter `v` only
   matters for (6.1), which has 9 bits of slack. No change proposed.
7. **ML-DSA-65/87.** The same construction applies with their `gamma_1, gamma_2, beta`: for
   each set, search the single-digit `u` form (`(c - 1) A = 2 B_u + 1 - 2^16`, `A <= 2^16`,
   even `c`) and the `z` slack; not done here.

## 8. Step A as implemented (relation H-u)

Open question 1 is answered with option (a): H-u replaces H as the default relation.

* Relation byte 0 means H-u (`agg_piop::manifest::Relation::Hu`, `RELATION_HU`). It is bound
  in the proof
  header's flags byte and in byte 3 of the event-0 manifest bytes as Section 1.3 says; the
  manifest carries the relation (`CommitmentManifest::for_relation`), and `for_batch` is the
  H-u default. Unlike Section 1.3's "no wire-version bump" (written for adding H-exact next to
  an unchanged H), the meaning of relation byte 0 changed, so `WIRE_VERSION` is 3 and a
  version-2 (relaxed H) proof is refused by its header.
* `u = U_lo + 63,079 U_hi + 31,540` (the form of 2.1 with H's slot names: `U_lo` is `S_ulo`,
  `U_hi` is `D_u`), `leaf6` with `Q_6`, `G_main4` alphabet codes `[4, 6, 0, 0, 0, 0, 0, 0]`,
  catalog rows unchanged (no new rows, as 7.1 predicted), stage degrees (4, 4, 3), gammas
  (26, 8, 44), stage-2 rounds of 4 elements, PIOP stage bytes 5,616 at `2^15` (4,736 at
  `N = 1,024`, measured in the mode-A sanity run).
* Proof of work, checked against `docs/soundness.md` and `pow_plan.py`: with
  `STAGE_DEGREES = (4, 4, 3)` and `STAGE_GAMMAS = (26, 8, 44)` every difficulty, nonce width,
  total (4,444 bits / 556 bytes / 2,718 work at `2^15`) and plan digest equals relaxed H's
  (`bit_length(4) = bit_length(5)`; the script asserts it), and the round-by-round ledger is
  `1,409/|K|` at `2^15` (`979` at `2^10`). The H-exact figures of 3.7 (4,443 bits, 2,710 work,
  `1,409/|K|`, the six anchors) reproduce exactly with the script and
  `STAGE_GAMMAS = (26, 7, 45)`, so step B's schedule stands as written.
## 9. Step B as implemented (relation H-exact)

Layout X1 as designed (Sections 1 to 4), on top of step A; the decisions below fill in or
depart from the design.

* **Relation type.** `Relation { Hu = 0, HExact = 1 }` lives in `agg_pcs_config::catalogs`
  (the lowest crate both use) and is re-exported by `agg_piop::manifest`; it
  also parses the command-line spelling (`h-u`, `h-exact`). Step A's name `Hu` is kept for
  relation byte 0 (the design's `H`).
* **Columns.** H-exact's `z` is a value of its own, `DigitValue::Zx` with the columns `bz_lo`
  (`S_zlo`), `bz_hi` (`B_zhi`), `a_z` and the value form `ValueForm::BitSlack` (agg-witness
  `BitSlackLayout`, `Z_BIT_SLACK`), instead of new column kinds `BzLo, BzHi, Az`; `u` keeps
  step A's `U_lo`/`U_hi` (the design's `S_ulo`/`D_u`). So `Column::alphabet` stays a function
  of the column, and only `group_columns`, `stack_columns`, `leaf_digits` and the new
  `bool_digits` depend on the relation. `a_z` follows the bits (positions 2..7 unchanged).
* **Counts.** `STAGE_DEGREES` stays a constant ((4, 4, 3) for both relations); the gamma and
  claim counts come from the manifest (`stage2::gamma_count`, `stage3::Gammas`,
  `Stage2Claims::len`, `Stage3Claims::len`, `transcript::stage_gammas`/`stage_terminals`),
  and the wire reads the claim vectors with the manifest's lengths. The proof-of-work plan
  follows the registry, so it needed no change; `pow_plan.py --relation h-exact` prints the
  H-exact tables and its self-check asserts the totals and two anchors of 3.7.
* **Verifier policy.** `verifier::verify` and `verify_timed` take the expected relation as
  an argument (the design's `RelationPolicy::Exactly`) and return `VerifyError::Relation`
  for a header that names another one, before event 0; the manifest, hence event 0, the
  catalog row and the plan, comes from the argument. `ProverInputs::relation` selects the
  prover's relation; `agg aggregate|verify|bench --relation h-u|h-exact` (default `h-u`;
  the design's `h`) set them in the CLI, whose reports print the relation.
* **Catalogs.** As 2.3: 26 new width-4 rows, generated in 1,617 s (`docs/pcs-config.md`
  Section 4); the H-u rows and the width-16 artifact are unchanged; the harness keeps every
  row of `catalog_batches()` (both relations) and takes `AGG_CATALOG_RELATION`. The planner's
  opening estimates of the H-exact rows equal H-u's; measured at `N = 1,024` the H-exact
  opening is 16 B longer (61,010 against 60,994 B).
* **Tests** (6.2 to 6.4). Mode B: H-exact round trips at 1, 64 and 256 records with the
  protocol-level negatives (cross-relation verification, a rewritten or unknown relation
  byte, a wrong stage-2 gamma nonce of 10 bits). Mode A: H-exact at 64 and 256 (Slim, one
  and two Flock parts) and the CLI round trip. PIOP level (`tests/h_exact.rs`): honest
  round trip with every claim equal to its table, the pointwise identities, the tamperings
  of the 6.4 table (stage 3 for `a_z = 2`, `B_zhi = 2, -1`; stage 2 for `D_u = 3, -4`; the
  lookup for `S_zlo = 2^15`, `S_ulo = -2^15 - 1`), non-Boolean digits that keep both `z` and
  the looked-up cell (`(a_z, B_zhi)` from `(0, 0)` to `(-1, 2)`, `S_zlo - 157`), which only the
  Booleanity zero-checks reject, and the value-boundary table with synthetic statements:
  instead of a test-only `PublicData` constructor, the test moves the public `T` of the
  record so that (3.1) holds over the integers (`T' = T + A (z' - z)` from the exact CRT
  product, or `T' = T - (u' - u)`), at the PIOP level where the verifier takes `T` from the
  instance. All rows of the table behave as stated (H-u accepts `z = 130,994` and `131,000`).
  Cross-check with the witness crate (6.3): the values decoded from the H-exact tables equal
  the derived ones, the H-u tables' values and the base-16 `Z_EXACT` form's (`a_z` agrees);
  the feature-gated `agg_witness::check` with `Layout::HExact` is exercised by the witness
  crate's own tests (`--features h-exact`), not from `agg-piop`.
* **Measured at `N = 1,024`** (CLMUL build, Slim, 1 warmup + 1 rep, a shared machine, sanity
  only, `docs/cli.md` Section 4): aggregate 522,154 B against H-u's 522,138 B; PIOP bytes
  4,736 B and work nonces 391 B for both; prove 2.98 s against 2.90 s.
* **Bench harness.** `agg-bench`'s mode-B harness takes `AGG_RELATION=h-exact` and prints
  the relation in its header and in every row.

## 10. H-FIPS: the exact FIPS 204 relation, hint included (relation byte 2)

Step 1 is the witness side (10.1 to 10.5), step 2 the PIOP (10.6, 10.7). FIPS 204 citations
are by algorithm and line.

H-u and H-exact are hint-free: they certify `z`, `u = ctr_q(w' - 2 gamma2 w1')` and `w1'`, but
not that some hint of weight at most `omega` maps `w'` to `w1'`, so the extracted values are a
signature of a relaxed scheme, and unforgeability goes through the SelfTargetMSIS argument of
spec 12.3. H-FIPS adds the hint: the extracted `(c~, z, h)` is an Algorithm 8 signature.

### 10.1 Relation

For every live record the witness is H-exact's (`z` by (4.3), the six bits `b_t`, the quotient
`v`) with `u` replaced by a per-coefficient tuple `(L_lo, L_hi, sigma, delta, e)` and one
per-signature slack `u_h`:

```text
L  = L_lo + 29,696 L_hi                     L_lo in [0, 2^16) (looked up in T_16)
s  = (1 + sigma)/2
r0 = gamma2 s + L + 1 - gamma2
u  = r0 - 2 gamma2 delta - e                (enters (3.1) in place of H-u's (4.2))

(F1) sigma^2 = 1                            (F4) e (e + 1) w1' = 0
(F2) delta (delta - sigma) = 0              (F5) e (e - 1)(w1' - 43) = 0
(F3) e (e - 1)(e - delta) = 0               (F6) L_hi (L_hi - 1) = 0
(F7) sum over the 1,024 coefficients of delta^2, plus u_h, equals omega = 80;  u_h in [0, 2^16)
```

with (3.1), (3.3), `w1' in [0, 44)` and `|z| <= B_z` as in H-exact. `29,696 + 2^16 = gamma2`, so
the two blocks `[0, 2^16)` and `[29,696, gamma2)` give `L in [0, gamma2 - 1]` exactly (the
spec_checks item 14 check). (F1) to (F3) give `sigma in {-1, 1}`, `delta in {0, sigma}` and
`e in {0, 1, delta}`, hence `e in {-1, 0, 1}` with `e = -1` only when `delta = -1`; (F4), (F5)
allow `e = 1` only at `w1' = 0` and `e = -1` only at `w1' = 43`. Committed per position: three
columns more than H-exact (`sigma`, `delta`, `e`; `L_lo`, `L_hi` take the slots of `U_lo`,
`U_hi`); per signature one more value (`u_h`).

### 10.2 What the pieces encode (FIPS 204 Algorithms 36 and 40)

Decompose (Algorithm 36) writes `r+ = w' mod q` (line 1) as `r1 (2 gamma2) + r0` with
`r0 = r+ mod+- 2 gamma2 in (-gamma2, gamma2]` (line 2), except when `r+ - r0 = q - 1` (line 3),
i.e. `r+ in [q - gamma2, q - 1]`, where it sets `r1 = 0` and `r0 = r0 - 1` (lines 4, 5). UseHint
(Algorithm 40) returns `r1` for `h = 0` (line 5) and `(r1 +- 1) mod 44` for `h = 1`, `+` iff
`r0 > 0` (lines 3, 4). In H-FIPS:

* `(s, L) -> r0 = gamma2 s + L + 1 - gamma2` is a bijection onto `(-gamma2, gamma2]` with
  `s = 1` (`sigma = 1`) iff `r0 > 0`, the sign test of Algorithm 40 lines 3, 4, and
  `L = (r0 - 1) mod gamma2` (`r0 - 1` for `sigma = 1`, `r0 - 1 + gamma2` for `sigma = -1`).
  The alternative `r0 = sigma L + (1 + sigma)/2` is the same set of pairs `(sigma, r0)` under
  another labelling of `L` (`L = -r0` for `sigma = -1`); the form here keeps `u` affine in the
  committed values (10.6 item 2), and every result of 10.3 holds for both;
* `delta in {0, sigma}` is the move of UseHint, `w1' = r1 + delta` before reduction mod 44;
* `e` is the wrap: `e = 1` where `w1' = 0` was reached from `r1 = 43` (`(r1 + 1) mod 44`, line
  3) or where Decompose took line 3 with `h = 0` (read as `r1 = 44`, `r0' = r+ - (q - 1) in
  (-gamma2, 0]`, and `q - 1 = 88 gamma2`, so `u = r0' - 1`); `e = -1` where `w1' = 43` was
  reached from `r1 = 0` (`(r1 - 1) mod 44`, line 4). Each wrap moves `u` by one because
  `44 (2 gamma2) = q - 1`.

The honest rule (`agg_witness::hfips::honest_coeff`, from `w'` and the signature's `h`):

| Case (`r+ = w' mod q`) | `h` | `w1'` | `sigma` | `L` | `delta` | `e` |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| `r+ >= q - gamma2` (Decompose line 3) | 0 | 0 | -1 | `r+ - (q - gamma2)` | 0 | 1 |
| `r+ >= q - gamma2` | 1 | 43 | -1 | `r+ - (q - gamma2)` | -1 | 0 |
| otherwise, `(r1, r0) = Decompose(r+)` | 0 | `r1` | sign of `r0` | `(r0 - 1) mod gamma2` | 0 | 0 |
| otherwise, `r0 > 0`, `r1 < 43` | 1 | `r1 + 1` | 1 | `r0 - 1` | 1 | 0 |
| otherwise, `r0 > 0`, `r1 = 43` | 1 | 0 | 1 | `r0 - 1` | 1 | 1 |
| otherwise, `r0 <= 0`, `r1 > 0` | 1 | `r1 - 1` | -1 | `r0 - 1 + gamma2` | -1 | 0 |
| otherwise, `r0 <= 0`, `r1 = 0` (only `r+ = 0`) | 1 | 43 | -1 | `gamma2 - 1` | -1 | -1 |

It has `delta^2 = h` and `u = ctr_q(w' - 2 gamma2 w1')`, the H-u value, so `v` and the bits are
unchanged. The corner `w' = q - gamma2, h = 0` (`r0' = 1 - gamma2`) gets `(sigma, L, delta, e)
= (-1, 0, 0, 1)`. A padded row (`w' = 0`, `h = 0`) gets `(-1, gamma2 - 1, 0, 0)` and
`u_h = 80`.

### 10.3 Soundness and weight-aware completeness (exhaustive)

**Soundness.** For every `(w1', L, sigma, delta, e)` with `w1' in [0, 44)`, `L in [0, gamma2)`
and (F1) to (F6), let `w' = 2 gamma2 w1' + u mod q` and `h* = [HighBits(w') != w1']`
(Algorithm 37). Then `|u| <= B_u`, `UseHint(h*, w') = w1'` and `h* <= delta^2`. Checked over all
181 admissible `(w1', sigma, delta, e)` times 95,232 values of `L`, 17,236,992 witnesses: 0
violations (`docs/scripts/spec_checks.py` item 14 with numpy; `agg_witness::hfips` unit test
`every_witness_is_sound` in Rust). `h*` is below `delta^2`, not equal: e.g. `(w1', sigma, L,
delta, e) = (0, 1, gamma2 - 1, 1, 0)` gives `w' = q - gamma2`, where `HighBits = 0 = w1'`.

**Weight-aware completeness.** For every honest pair `(w', h) in Z_q x {0, 1}`, with
`w1' = UseHint(h, w')` and `u = ctr_q(w' - 2 gamma2 w1')`, the least `delta^2` over the witnesses
of `(w1', u)` equals `h`: 0 failures over the 16,760,834 pairs (item 14), and the honest rule of
10.2 attains it (item 14 and unit test `honest_rule_is_complete_with_weight_h`). Hence an
honest signature of hint weight `wt(h) <= 80` has `sum delta^2 = wt(h)` and `u_h = 80 - wt(h)`.

**The completeness fix.** The variant `e (e - delta) = 0` in place of (F3) forbids `e = 1` at
`delta = 0`. Item 14 then finds exactly one failing pair, `(w', h) = (8,285,185, 0)` =
`(q - gamma2, 0)`, whose only witnesses have `delta^2 = 1` (`(sigma, L, delta, e) = (1,
gamma2 - 1, 1, 0)`); an honest signature of hint weight 80 with such a coefficient would need
`u_h = -1`. A model puts this at about `3.2e-7` per signature (estimate), about `3e-4` per
batch of 1,024 (estimate). Soundness is the same under both (item 14: 0 violations over the
16,951,296 witnesses of the variant's set).

### 10.4 Extraction: the witness is a FIPS 204 signature

Let a record satisfy H-FIPS. Set `w' = 2 gamma2 w1' + u mod q`, `h*_p = [HighBits(w'_p) !=
w1'_p]` and `sigma_FIPS = sigEncode(c~, z, h*)` (Algorithm 26, with HintBitPack, Algorithm 20).
Algorithm 8 on `(pk, M', sigma_FIPS)`:

* line 2 (sigDecode, Algorithm 27): `|z|_inf <= 130,993 < gamma1`, so BitPack/BitUnpack round-trip
  `z` (Algorithm 27 line 3); `wt(h*) <= sum delta^2 = 80 - u_h <= 80` (10.3; the sum is at most
  1,024 and `u_h < 2^16`, so (F7) over `F_p` is the integer identity), so HintBitPack is defined,
  and HintBitUnpack (Algorithm 21, called at Algorithm 27 line 5) returns `h*`: the counts are
  nondecreasing and at most `omega` (line 4), the indices within a row increase (line 9), and
  the unused bytes are zero (line 17). So line 3 does not return false;
* line 9: `w'_Approx = A z - c t1 2^d mod q`, which is `2 gamma2 w1' + u mod q = w'` by (3.1)
  (`T = 2^d c t1` over the integers);
* line 10: `UseHint(h*, w') = w1'` coefficient-wise (10.3);
* line 12: `c~' = H(mu || w1Encode(w1'), 32) = c~` by (3.3);
* line 13: `||z||_inf <= 130,993 < gamma1 - beta = 131,072 - 78 = 130,994` (FIPS 204 Table 1:
  `gamma1 = 2^17`, `beta = tau eta = 78`; `B_z = gamma1 - beta - 1`).

So Algorithm 8 accepts, and the aggregate argument for H-FIPS is a proof of knowledge of `N`
FIPS 204 signatures (on the records' keys and messages, with the published `c~`). The aggregate
unforgeability game then reduces, as a black box, to the SUF-CMA security of ML-DSA-44 itself:
step 1' of spec 12.3 runs with `Sigma_H` replaced by ML-DSA-44 and `phi` by the identity, and
`B` outputs `(M_i*, sigEncode(c~_i*, z_i*, h*_i*))`. EUF-CMA is not enough: the extractor's
reruns may ask signing queries on `M_i*` (the rerun-only queries of step 1'), so `B`'s
challenger may have signed `M_i*`, and `B` wins only because its output differs from every
answer (the `c~` collision bound of step 1'). That is the obligation spec revision 7 closed
with SUF. SUF-CMA of ML-DSA-44 is an assumption, not a theorem: FIPS 204 Section 3.1 says that
ML-DSA "is designed to be" SUF-CMA, and the published proofs do not cover FIPS 204's exact
algorithms (spec 12.3, "What is assumed", (b)). The relaxed scheme and step 2' are not needed
for this relation. Compared with H-u, H-FIPS replaces the proved SUF-CMA of the relaxed scheme
(Lemma 12.8 of `docs/soundness.md`: MLWE, SelfTargetMSIS at `zeta'`, MSIS at 380,930, plus the
CMA-to-NMA assumption) by the standard assumption on ML-DSA-44; it does not strengthen the
concrete bound, since `zeta' = zeta` already holds for H-u and H-exact (spec 12.3). For an
honest signature the extracted `h*` is the signature's own hint (UseHint is injective in `h`),
so the extracted bytes equal the signature, byte for byte (tests below).

### 10.5 Witness side (step 1) as implemented

`agg-witness` module `hfips` (always compiled; the PIOP of step 2 calls it):

* `Coeff` (`l_lo, l_hi, sigma, delta, e`), with `l()`, `r0()`, `u()`, `weight()`,
  `zero_checks(w1)` (F1) to (F6) in that order, `is_valid`, `split_l` (canonical
  `L_hi = [L >= 2^16]`), `Coeff::PADDED`; `honest_coeff(w', h)` (10.2); constants
  `RELATION_H_FIPS = 2`, `L_STEP = 29,696`.
* `HFipsRecord` (the committed values of one record before storage shifts: the three
  `Z_BIT_SLACK` digits of `z`, the coefficients, `v`, the bits, `u_h`): `from_witness` (from
  `derive::derive`; checks `u`, `w1'`, `delta^2 = h` and every constraint coefficient by
  coefficient), `padded()`, `extracted_hint()`, `extract_signature(c~)` (through
  `derive::reencode_h`).
* `check_record` / `check_batch`: every H-FIPS identity over the integers (the five looked-up
  kinds `B_zlo, L_lo, u_h` and the halves of `v`, the Booleans, `range_w1`, (F1) to (F7), the
  ring identity with the H-FIPS `u`, (3.3) through the sponge) on live and padded records. New
  checker identities `Range16`, `DeltaSign`, `EdgeE`, `WrapLow`, `WrapHigh` (with `SignSquare`,
  `BoolSlack`, `HintWeight` reused).
* The retired S1+S2 sketch (feature `fips`, base-16 `tables::Layout::Fips`) moves from layout
  id 2 to 3, so relation byte 2 names H-FIPS only. H-FIPS supersedes S1+S2: it commits no `r1`
  (the `w1'` bits already carry it) and replaces the cubic UseHint check by (F3) to (F5).

Tests (`crates/agg-witness/tests/hfips.rs`, 6 tests; unit tests in `hfips.rs`, 4 tests; all
pass, `cargo test --release -p agg-witness`, also with `--all-features`):

* 64 honest signatures (40 own-key deterministic, 24 shared-key hedged): derivation, the
  checker, the extracted signature equals the original bytes and verifies; measured hint
  weights min 48, median 62, max 76 (test output).
* The pinned edge record `EDGE_RESIDUE` (a coefficient `w' = q - gamma2`): the corner witness.
* **Synthetic weight-80 signature with the corner** (deterministic). A public key with
  `t1 = 0` (`rho` from a seed, 1,280 zero bytes: a well-formed key, pkDecode accepts any bytes,
  Algorithm 23) makes Algorithm 8 line 9 compute `w'_Approx = A z` independently of `c`. The
  test draws `z` uniform in `[-B_z, B_z]`, moves two coefficients of `z` (solving a 2x2 system
  modulo `q`) so that `A z` equals `q - gamma2` at `(rho, j) = (1, 17)` and `0` at `(2, 101)`,
  sets 80 hint bits (at least one per `h = 1` case of 10.2, including `w' = 0, h = 1`, the only
  `e = -1` case; none at the corner), and computes `c~ = H(mu || w1Encode(UseHint(h, w')))`.
  This crate's Algorithm 8 and RustCrypto's `ml-dsa` verifier both accept it. Derivation gives
  `sum delta^2 = 80`, `u_h = 0` and the corner witness; every `(sigma, delta, e)` class occurs;
  the checker passes; the extracted signature equals the constructed bytes and verifies. Honest
  signing cannot place such a coefficient deliberately (the hash fixes `c` after `w1'`).
* The same record under the variant constraint: the corner's alternative witness keeps `u`
  but makes the weight 81, rejected by the weight identity (`u_h = 0`) or the lookup (`u_h = -1`).
* Tamperings, each rejected by its identity: `sigma = 0`, `delta = -sigma`, `e = -1` at
  `delta = 0`, `e = 1` at `w1' != 0`, `e = -1` at `w1' != 43`, `L_hi = 2`, `L_lo = 2^16`,
  `L_hi = -1` with `L_lo + 29,696` (keeps `L` and the ring: only (F6)), `u_h + 1` (weight), a
  dropped hint (`delta -> 0`, `u_h + 1`: only the ring), a flipped `w1'` bit (ring and (3.3)),
  `B_zhi = 2`; the other split of an `L` in the overlap `[29,696, 2^16)` is accepted.
* Padded rows of a batch of 3 (`N* = 8`).

### 10.6 Step 2 decisions

An earlier sketch had a seventh Akita group on the signature domain, a `u_h` lookup stack on
that domain, six zero-checks (`sigma^2 = 1`,
`delta (delta - sigma) = 0`, `e (e - 1)(e - delta) = 0`, `e (e + 1) w1' = 0`,
`e (e - 1)(w1' - 43) = 0`, `L_hi` Boolean) and a stage-1 batching vector of 33. The PIOP
departs from it as follows:

1. **No seventh group.** `u_h` is a thirteenth polynomial of `G_2` (domain `D_2`, width 16,
   round 2) and takes the free slot 6 of lookup stack 2 (6 carries in 8 slots). `D_2` has
   `13 + max(0, d_sig - 13)` variables, the last `d_sig` of which are the signature index
   (spec Section 1); at `d_sig >= 13` it is the signature domain. Below, the honest table holds
   `u_h(i)` at `D_2` index `i 2^(d_2 - d_sig)` and 0 elsewhere, and the weight identity sums
   the `D_2` cells of a signature (all in `[0, 2^16)` by the lookup, so `sum delta^2 <= 80`
   still follows). The six-group contract (`GROUP_COUNT` arrays in the PCS driver, the
   verifier, Akita's grouped rows with five precommitted profiles) and the lookup shape (four
   stacks, `L` unchanged) stay; the committed cost is the same as a seventh group's (one
   element per signature at `2^15`, the `2^13`-cell floor below). Round 2 is early enough:
   `u_h` is fixed before `X`, `tau` and `r^(2)`, and the extracted signature depends on
   round-1 values only (10.4; `u_h` only certifies `sum delta^2 <= 80`).
2. **The sign as a bit, `u` affine.** The PIOP commits `s = (1 + sigma)/2 in {0, 1}`
   (`sigma^2 = 1` iff `s^2 = s`) and uses `r0 = gamma2 s + L + 1 - gamma2` instead of the
   sketch's `r0 = sigma L + (1 + sigma)/2 = (2 s - 1) L + s` (10.2: the same pairs `(sigma, r0)`,
   another labelling of `L`). The sketch's form puts the products `s l_lo`, `s l_hi` into the
   stage-3 ring identity; in a first implementation they kept the 16-bit `l_lo` lane off the
   kernel's linear fast path and cost about 170 ms of stage 3 at `N = 8,192` (measured with
   `AGG_STAGE_TRACE`, main `eq(r^(2))` group message 264 ms against 178 ms after the change,
   local, direction only). With the affine form the ring identity is a weight times an affine
   combination, as in H-u and H-exact.
3. **Five zero-checks, one of degree 3.** `F4 = e (e + 1) w1'` and `F5 = e (e - 1)(w1' - 43)`
   are replaced by the single degree-2 `F45 = 2 e w1' + 43 e^2 - 43 e`: once `F1`, `F2`, `F3`
   give `e in {-1, 0, 1}`, `F45` is `2 w1'` at `e = 1`, `86 - 2 w1'` at `e = -1` and 0 at
   `e = 0` (small integers, so zero in `F_p` iff zero over `Z`), i.e. exactly `F4` and `F5`.
   The exhaustive soundness check of 10.3 holds unchanged (the admissible sets are equal).
4. **Stage placement.** Stage 3 has degree 3, so only the
   degree-2 checks can sit there with `eq`: `F1 = s (s - 1)`, `F2 = delta (delta - 2 s + 1)`,
   `F45` and `F6 = L_hi (L_hi - 1)` go to stage 3 at `r^(2)`, as H-exact's Booleanity checks
   (their lanes `s, delta, e, L_hi, b_t` are read there by the ring identity anyway, so they add
   terms, not lanes or reductions). The weight identity `F7` also goes to stage 3 (degree 3 with
   `eq`, 10.7). The one degree-3 check `F3 = e (e - 1)(e - delta)` needs degree 4 with `eq`:
   stage 2 at `r^(1)` (degree 4, like `leaf6`; H-FIPS has no leaf digit) rather than stage 1
   at `tau`, because in stage 2 its terminal claims `delta(r^(2))`, `e(r^(2))` land where the
   ring already reads both columns (two cheap reductions in an existing group), while in stage
   1 they would land at `r^(1)` (two reductions in another group) and grow the `tau`-batched
   vector. Both placements are sound by the same point-reuse argument (`delta`, `e` are
   committed in round 1, before `r^(1)`; `docs/soundness.md` 3.4).

So the stage-1 vector stays at 26 (the sketch's 33 does not apply), stage 2 has 7
coefficients and stage 3 has 51.

### 10.7 PIOP layout as implemented (relation byte 2)

Columns (`agg_piop::manifest`, relation `Relation::HFips`):

| Group | H-FIPS polynomials | Count |
| --- | --- | ---: |
| `G_main16` | `bz_lo` (`S_zlo`), `l_lo` (`L_lo - 2^15`, the slot of `U_lo`), `v_lo`, `v_hi` | 4 |
| `G_main4` | `bz_hi`, `l_hi` (the slot of `U_hi`), `b_0..b_5`, `a_z`, `s`, `delta`, `e` | 12 |
| `G_1`, `G_3`, `G_m` | unchanged | 6, 1, 7 |
| `G_2` | `y2_0..5`, `kappa2_0..5`, `u_h` (stored `u_h - 2^15`) | 13 |

43 polynomials (H-u 38, H-exact 39), 36 opened at `r^(3)`; 16 main-domain columns, so
16,507.5 committed elements per signature at `N = 2^15` (`manifest` test; H-u 12,410.5,
+33.0%). Alphabet codes in event 0: `s` 0 (Boolean), `delta` and `e` `0x23` (ternary,
`TERNARY_ALPHABET_CODE`), `u_h` `0x90` (looked up). Catalog rows `mldsa-44-l16f-batch-<N*>-mult-final`.

Lookup: stack 0 is `[bz_lo | l_lo | v_lo | v_hi]`, stack 2 `[kappa2_0..5 | u_h]` (7 columns, 8
slots), so `L` and every lookup site are those of H-u; the stack-2 reduction (7.7) reads `u_h`.

Value forms: `z` by (4.3); `L = l_lo + 29,696 l_hi + 2^15` (`agg_witness::digits::L_SLACK`, a
new `ValueForm::Slack`); `u = gamma_2 s + L + 1 - gamma_2 - 2 gamma_2 delta - e`.

Stage 2 (degree 4, 7 coefficients): `def_y1[t]` (gammas 0..5) and `F3` (gamma 6) on the main
domain at `r^(1)` (a `Term::Monomials` of four monomials in `delta, e`). Terminal claims: `b_t`
(6), `delta(r^(2))`, `e(r^(2))` (8 in all).

Stage 3 (degree 3, 51 coefficients; `stage3::Gammas`): `ring` (0); `bool` for `bz_hi`, `l_hi`,
`a_z` (1..3); `F1`, `F2`, `F45`, `F7` (4..7); the 37 reductions (8..44); the reductions of
stage 2's `delta`, `e` (45, 46); the four stack reductions (47..50). The ring reads `-u` as an
affine combination (`l_lo` -1, `l_hi` -29,696, `s` `-gamma_2`, `delta` `2 gamma_2`, `e` 1,
constant `gamma_2 - 2^15 - 1`). `F7` is one instance with two parts at the
point `(1/2, ..., 1/2, r^(2)_sig)` (`EqPoint::HalfR2`; `eq` at a coordinate `1/2` is `1/2` for
both bit values, so these are sums over positions): `delta^2` on the main domain with scale
`2^(d_main - d_sig) = 2^10` and `u_h = U + 2^15` on `D_2` with scale `2^(d_2 - d_sig)`, claim
`omega = 80`. Its terminal values are `delta(r^(3))` and `U(r^(3)_2)`, opened anyway, so `F7`
needs no reduction. Every stage-3 term has degree at most 3 with `eq` (test
`h_fips_stage_three_has_fifty_one_instances`).

Counts: stage gammas (26, 7, 51), stage terminals (18, 8, 36), stage degrees (4, 4, 3); PIOP
bytes `16 (13 + 18 + 6 + 8 + 36 + 11 d_main)`: 5,696 B at `2^15` (H-u and H-exact: 5,616 B).

Proof of work: `bit_length(7) = 3` and `bit_length(51) = bit_length(45) = 6`, so every
difficulty, the nonce stream (4,443 bits, 556 bytes, 2,710 units of work at `2^15`) and the
plan digest equal H-exact's (`pow.rs` test `h_fips_plans_equal_h_exact_plans`,
`pow_plan.py --relation h-fips` self-check). The relation is bound by event 0's manifest bytes
(relation byte 2, the `G_main4` and `G_2` descriptors, the catalog row), as 3.7 argues for
H-exact; the round-by-round ledger before work is H-exact's `1,409/|K|` at `2^15`.

Header and verifier: relation byte 2 in the proof header and byte 3 of event 0;
`Relation::from_id(2)` is H-FIPS (byte 3 and above are unknown); the verifier takes the relation
from its caller and rejects another relation's proof before event 0 (`VerifyError::Relation`);
`agg aggregate|verify|bench --relation h-fips`, `AGG_RELATION=h-fips` for the mode-B bench.

Catalogs (`agg-pcs-config`, `CATALOG_D_SIG_H_FIPS = 3..=15`): 13 width-4 scalar rows
(`G_main4` at `(13..25, 12)`), 13 grouped rows, 3 width-16 scalar rows (`G_2` at
`(13|14|15, 13)`), planned by `update_catalog_rows` with `AGG_CATALOG_RELATION=h-fips` in
1,105.6 s (measured, 3 workers, CPUs 0,1,3-15). The planner's opening estimate
equals H-u's at every size except `2^15` (65,398 B against 65,282 B; estimate of the planner).
Every grouped row keeps at least 128.0881 bits of accepted fold support (2^128.5633 at
`d_sig` 13 and 15); the Akita grinding plans are pinned (`tests/akita_grinding.rs`) and
differ from the H-u and H-exact rows'.

Tests (all pass, `cargo test --release`): `agg-piop` `tests/h_fips.rs` (PIOP round trips
with every claim equal to its table's extension, the pointwise identities, the synthetic
weight-80 corner records of `agg_witness::gen::hint_corner_record`, the variant's corner
witness rejected by `F7` (stage 3) or, with `u_h = -1`, by the lookup, and tamperings each
rejected at its place: `s = 2`, `delta = -sigma` (stage 3); `e = -1` at `delta = 0` with `L`
moved to keep `u` (stage 2, `F3`); `e = 1` at `w1' != 0` and `e = -1` at `w1' != 43` with `L`
moved to keep `u` (stage 3, `F45` alone); `l_hi = -1` with `l_lo + 29,696` (stage 3, `F6`
alone); `l_lo = 2^16` and a `u_h` cell of `2^16` (lookup); `u_h +- 1` and a dropped hint
(stage 3)); `tests/mode_b.rs` (round trips at 1, 64 and 256 records, the 256-record batch with
two weight-80 corner records; cross-relation verification and rewritten relation bytes
rejected; a wrong stage-2 gamma nonce rejected); unit tests of the manifest, the stage specs,
the transcript counts, the wire and the plan; `agg-pcs-config` commit and batched opening at
256 and 1,000 records; `agg-cli` mode A at 64 (Slim, one part) and 256 (two parts) and the
CLI round trip with crossed verifications.

### 10.8 Costs (local measurements, direction only)

Prediction written before measuring (estimates): prove at `2^15` on the cloud +0.75 to +1.2 s
over H-u; locally +6% to +12% of the H-u prove time at 1K and 8K.

Measured, mode B (`AGG_RELATION=... scripts/bench.sh --bench mode_b -- 1024 8192 --warmup 1
--reps 2`, default build, CPUs 0,1,3-15 of this box, medians of 2 after 1 warmup;
the PIOP part only, no Flock. Mode A adds Flock's Keccak argument, the same for every relation,
but there the `G_main` commitment overlaps the Flock commitments, so the mode-A delta may
differ from the mode-B delta in either direction; it has not been measured):

| Code | Relation | Prove 1K (s) | Prove 8K (s) | Prover peak RSS 8K (MiB) | Aggregate without public bits 8K (B) |
| --- | --- | ---: | ---: | ---: | ---: |
| first layout | H-u | 1.559 | 4.120 | 3,567 | 357,148 |
| first layout | H-exact | 1.687 | 4.378 | 3,655 | 357,105 |
| first layout | H-FIPS (sketch's `r0`, products in stage 3) | 1.883 | 4.842 | 3,897 | 357,349 |
| final layout | H-u (two runs) | 1.564, 1.676 | 4.367, 4.335 | 3,587, 3,552 | 357,148 |
| final layout | H-FIPS (affine `r0`, two runs) | 1.729, 1.760 | 4.885, 4.964 | 3,915, 3,909 | 357,309 |
| spot check | H-u | 1.658 | | | |
| spot check | H-FIPS | 1.696 | | | |

So with the final layout H-FIPS costs about +0.13 s (+8%) at 1K and +0.57 s (+13%) at 8K
over H-u in this box's mode B (means of the two interleaved runs; the run-to-run spread is about
0.1 s), +0.34 GiB of prover RSS (+9.5%) and +161 B of aggregate at 8K (0.02 B per signature:
5 more stage claims, Akita opening of 43 polynomials). The 1K delta is inside the run-to-run
noise: the spot check (1 warmup + 2 reps) measured +0.04 s, with
systematic step deltas of about +85 ms (stage 3 +47, round 1 +20, tables +12 ms) and an
opening that varied by about 90 ms. The local prediction (+6% to +12%) was low at 8K.
Stage-3 trace at 8K (`AGG_STAGE_TRACE=1`, `tests/h_fips.rs` `h_fips_stage_trace`):
the main `eq(r^(2))` group's messages take 178 ms against 93 ms for H-u (the extra terms of
`F1`, `F2`, `F45` and the reductions), the `F7` main group 55 ms, the ring's `w_A` group 129
against 103 ms (H-exact's `a_z` lane); stage 2's `F3` group 75 ms against 39 ms for H-u's
leaves.

16K, the first end-to-end use of the `d_sig = 14` rows (the `l16f` grouped row for `2^14` and
the width-16 `G_2` row at (14, 13); the bench asserts that the verifier accepts). Measured,
mode B, same box and cores, `-- 16384 --warmup 1 --reps 1`, run four times interleaved
(H-u, H-FIPS, H-u, H-FIPS):

| Relation | Prove 16K (s) | Prover peak RSS (MiB) | Aggregate without public bits (B) | Committed elements per signature |
| --- | ---: | ---: | ---: | ---: |
| H-u | 6.826, 7.230 | 6,981, 6,945 | 621,859 | 12,424.5 |
| H-FIPS | 7.858, 8.459 | 7,657, 7,638 | 621,901 | 16,521.5 |

So +1.13 s (+16%, means of two runs each; pairwise +1.03 and +1.23 s), +0.67 GiB of prover
RSS (+9.8%) and +42 B of aggregate (PIOP +80 B, Akita opening -38 B). The step deltas (means)
are stage 3 +0.47 s, tables +0.25 s, opening +0.19 s, round 1 +0.18 s and stage 2 +0.07 s; the
lookup does not change. Predictions written before this run (estimates): +0.9 to +1.3 s, +0.6
to +0.75 GiB and +100 to +250 B; the aggregate delta came out below the range.

Cloud estimate at `2^15` (weak estimate). Two calibrations of this box against
`c4d-highmem-16`. (a) H-exact's local 8K mode-B delta (+0.26 s) against its measured cloud 32K
mode-A delta (+0.21 s) gives about +0.5 s for H-FIPS. (b) The
relative overhead in mode B (+13% at 8K, +16% at 16K) applied to about 7.3 s of non-Flock
prove time at 32K on the cloud gives about +1.0 to +1.2 s. So +0.5 to +1.2 s (+3% to
+8% of 15.0 s), against the earlier +0.75 to +1.2 s (estimate). Both calibrations are weak.
Calibration (a) rests on H-exact's local 8K delta, which is smaller than the drift of H-u
itself between sessions on this box (4.120 s for the first layout, 4.335 and 4.367 s for the
final one).
Calibration (b) assumes that mode A adds no overlap effect. Only a back-to-back cloud run of
both relations at 32K can settle the number.

Prover RSS at `2^15` (estimate). The local delta grows linearly with `N` (+0.34 GiB at 8K,
+0.67 GiB at 16K), as expected: the four extra main-domain columns, the `HalfR2` group and
`u_h` all scale with `N`. Scaling by 2 gives about +1.3 to +1.4 GiB at 32K, about twice the
earlier estimate of +0.7 GiB and about +5% over H-u's 25.8 GiB on the cloud. This holds if the
mode-A peak at 32K falls while the H-FIPS tables are live. The cloud run should measure it.

## Appendix X. Spec Appendix A's `u` form (two small columns), for comparison

`u = 118,787 a_u + 2^16 B_uhi + B_ulo - 190,465` with `a_u in {0, 1}`, `B_uhi in [0, 4)`
(stored balanced `[-2, 1]`, `leaf4` and `link`), `B_ulo` looked up; `z` as in X1.

| | H | H-exact X1 (recommended) | H-exact X2 (Appendix A) |
| --- | --- | --- | --- |
| `G_main4` polynomials | 8 | 9 | 10 |
| New width-4 rows | 0 | 13 scalar + 13 grouped | 13 scalar + 13 grouped |
| Committed elements per signature at `2^15` | 12,410.5 | 13,434.5 | 14,458.5 |
| Stage degrees | (4, 5, 3) | (4, 4, 3) | (4, 3, 3) |
| Stage instances | 26, 8, 44 | 26, 7, 45 | 26, 7, 46 |
| PIOP stage bytes at `2^15` | 6,016 | 5,616 | 5,232 |
| Proof-of-work bits / bytes at `2^15` | 4,444 / 556 | 4,443 / 556 | 4,418 / 553 |
| Round-by-round ledger at `2^15` | `1,434/|K|` | `1,409/|K|` | `1,384/|K|` |
| Prove delta at `2^15` (estimate) | | about +0.1 s | about +0.2 to +0.4 s |

X2 saves 384 B of PIOP bytes against X1 (0.01 B/sig) and costs a second committed column
(+1,024 elements per signature, another lane in two stage-3 groups); X1 is cheaper where the
prover spends its time.
