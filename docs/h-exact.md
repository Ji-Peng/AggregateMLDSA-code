# H-exact: the exact FIPS 204 bounds in the PIOP and the catalogs (design)

| Field | Value |
| --- | --- |
| Status | design, docs only; nothing implemented. Implementation starts after `pow-128` is merged (AGENTS.md, decisions 2026-10-25) |
| Applies to | ML-DSA-44, lookup layout L16, modes A and B, PIOP wire version 2 of `docs/soundness.md` (branch `pow-128`) |
| Base | code `be398d8` (main); layered on branches `pow-128` (`406245a`), `piop-speed` (`7476508`), `flock-speed` (`baf9e46`) |
| Replaces | spec Appendix A, paragraph "H-exact" (which `ref/notes/msis-estimate.md` calls "spec Section 15"; there is no Section 15) |
| Last reviewed | 2026-10-25, first version |

The uppercase terms MUST, SHOULD and MAY are normative as in BCP 14. Numbers not marked
"estimate" were computed for this document with exact integer arithmetic (Section 4.4 lists
the checks) or read from the pinned code and docs; every cost figure in Section 5 is an
estimate scaled from the measured Phase 7 run (`ref/notes/bench-cloud.md`, `be398d8`).

## 0. Results in one page

1. **Relation.** H-exact is relation H of spec Section 3 with the honest bounds as the proven
   bounds: `|z|_inf <= B_z = gamma_1 - beta - 1 = 130,993` and `|u|_inf <= B_u = 2 gamma_2 + 1
   = 190,465`, everything else ((3.1), (3.3), `w_1' in [0, 44)`, the `v` range) unchanged. Its
   SelfTargetMSIS bound is `zeta' = max(130,993, 190,465 + 2^12 * 39) = 350,209 = zeta` of
   FIPS 204, so aggregate unforgeability rests on ML-DSA-44's own instance (core-SVP 123.5
   classical, `ref/notes/msis-estimate.md`).
2. **Flag.** Runtime relation id, byte `1` (H is `0`, S1+S2 keeps `2`), carried in the proof
   header's flags byte and in the event-0 manifest bytes; the verifier accepts exactly the
   relation its caller configures (no downgrade to H).
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
| H (default) | 0 | `(2^17, 2^18)` (inclusive `[-2^17, 2^17 - 1]`, `[-2^18, 2^18 - 1]`) | 421,888 | implemented |
| H-exact | 1 | `(130,993, 190,465)`, both symmetric and exact | 350,209 | this document |
| S1+S2 (`fips`) | 2 | reserved (spec Appendix A) | 350,209 | witness only |

The id equals `agg_witness::tables::Layout::id()` (0 H, 1 H-exact, 2 S1+S2) and the
`BatchTables` serialization layout byte, so one number names the relation everywhere.

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
   * the PIOP plan digest of `docs/soundness.md` 6.4 (appended by `pow-128`), which differs
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

No wire-version bump is needed beyond `pow-128`'s version 2: H proofs are byte-identical, and
H-exact proofs differ only in the flags byte, in counts the verifier derives from the manifest
(stage-2 message width and claim count, stage-3 claim count) and in the opening.

## 2. Column layout and catalogs

### 2.1 Value forms (verified, Section 4.4)

Let `w = z + 130,993 in [0, 261,986]` and `w_u = u + 190,465 in [0, 380,930]`.

* `z`: the slack lemma of `ref/notes/improvements-tables.md` 2(a) with `261,987 =
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
2026-10-24 run of all rows of both families took 2,666 s; the width-16 rows, about a third to
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
plus a constant is the same combination of the extensions plus the constant. On `piop-speed`
the linear split (`stage::split_linear`) takes only `Ints` lanes, i.e. `S_zlo, S_ulo, V_lo,
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

### 3.7 Proof-of-work schedule (on top of `pow-128`)

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

Totals (the script of `pow-128`, `docs/scripts/pow_plan.py`, run with `STAGE_DEGREES =
(4, 4, 3)` and `STAGE_GAMMAS = (26, 7, 45)`, 2026-10-25):

| `N*` | H: bits / bytes / work / ledger | H-exact: bits / bytes / work / ledger |
| ---: | --- | --- |
| `2^3` | 1,811 / 227 / 1,364 / `563/|K|` | 1,810 / 227 / 1,356 / `550/|K|` |
| `2^10` | 3,124 / 391 / 2,048 / `999/|K|` | 3,123 / 391 / 2,040 / `979/|K|` |
| `2^13` | 3,889 / 487 / 2,438 / `1,251/|K|` | 3,888 / 486 / 2,430 / `1,228/|K|` |
| `2^15` | 4,444 / 556 / 2,718 / `1,434/|K|` | 4,443 / 556 / 2,710 / `1,409/|K|` |

(`ledger` = round-by-round sum before work; nonce counts are equal, 194 to 483.) The plan
encoding of `docs/soundness.md` 6.4 needs no new field: the relation is bound by the manifest
bytes in the same event, and the stage-2 entry already makes the digests differ. Provisional
anchors (unchanged encoding, `N = N*`, default part size; to be regenerated by the script once
it takes `--relation h-exact`, and to move with any `pow-128` change):

```text
H-exact  N* = 2^3   modeB          c9dd30a002089fff68b1653faddb2ac860e093f771eb3eeaaf03e60152e54c85
H-exact  N* = 2^3   slim span on   8f69f49792afc311cf43042df023b98706f92bdaecb1c3ac837690e92995e983
H-exact  N* = 2^10  modeB          750a40c19877ba6bda8c7ddee36866239b6338f7caa84109880662a33d64a1d8
H-exact  N* = 2^10  slim span on   82ffd38abe14e463ad41cc9ad209e07d5143f28067bb3696f2c445aad2c05c8f
H-exact  N* = 2^15  modeB          e4b1dffc21351ca615ace27500406a1990609ccb6cc07e13c6da50c18a416046
H-exact  N* = 2^15  slim span on   14666dea14ede0ddea3f5cc34d50e8d746021c034941d060a6c2d08af8960d41
```

(The same computation reproduces the H anchors of `docs/soundness.md` Section 10, e.g.
`df9161a5...e2f38c` at `2^15` Slim span on.)

## 4. Knowledge soundness and the exact bounds

### 4.1 Ledger delta at `N = 2^15`

Spec 12.1 (per-instance union bound, still the text on main):

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

`docs/soundness.md` Section 5 (round-by-round, the ledger that replaces 12.1 with `pow-128`):
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
zeta (FIPS 204) = max(gamma_1 - beta, 2 gamma_2 + 1 + 2^(d-1) tau) = max(131,000, 350,209) = 350,209
```

and the reduction of spec 12.3 runs with ML-DSA-44's own instance: core-SVP 123.5 classical
(lattice-estimator rough model, block size 423; 112 quantum by security-estimates), the
published Dilithium2 figures that `ref/notes/msis-estimate.md` reproduces; the "+20.5%" sentence of spec 12.3 and
its estimator obligation do not apply to H-exact. (FIPS 204 Table 1 claims Category 2 for
ML-DSA-44; spec 12.3's "category 1" is H's.)

### 4.4 Checks run for this document

Exact integer checks (Python, `python3 -I`, `taskset -c 0,1,3-15`, scratch script, not
committed; Section 6.1 item 11 moves them into `spec_checks.py`):

* the reachable sets of both forms of 2.1 equal `[-130,993, 130,993]` and `[-190,465,
  190,465]` (full enumeration of all digit tuples), and `130,994`, `190,466` are unreachable;
  Appendix A's `u` form also gives exactly `[-190,465, 190,465]`;
* `C_z = -98,225`, `C_u = 31,540`; the canonical decomposition round-trips for every `z` and
  `u` in range with every digit in its alphabet; padded and edge encodings as in 2.1;
* `Q_6(Z (Z + 1)) = 0` iff `Z in [-3, 2]` on `[-200, 200)`;
* lifting: `|Delta^Z| < 2^54.05 < 2^63`;
* `zeta', zeta` as above; `B_z, B_u, gamma_2, beta` from FIPS 204 Table 1;
* the proof-of-work totals and digests of 3.7 with `pow-128`'s script.

## 5. Costs at `N = 2^15` (estimates, scaled from Phase 7)

Base: Phase 7, `be398d8`, `c4d-highmem-16`, `Slim`, span on, medians of 5. The deltas are
estimates from the structure of each step (which lanes and polynomials it touches); they
are not measurements. They apply in absolute terms on top of `piop-speed` and `flock-speed`
(which do not touch the parts H-exact changes, except stage 3, where `piop-speed` makes the
`Ints` lanes cheaper and leaves the narrow lanes as they are).

### 5.1 Committed elements and memory

| | H (measured layout) | H-exact |
| --- | ---: | ---: |
| Main-domain columns | 12 (4 width 16 + 8 width 4) | 13 (4 + 9) |
| Committed elements per signature | 12,410.5 | 13,434.5 (+8.25%) |
| Committed elements, total | 406.7M | 440.2M |
| Prover RSS | 28.9 GiB | +0.1 to +0.3 GiB (one `i8` column of `2^25` cells, its Akita commit data; estimate) |

### 5.2 Prover time

| Step | H, Phase 7 (ms) | H-exact (estimate, ms) | Reason |
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

So prove about 23.8 to 24.0 s at the Phase 7 base, +0.5% to +1%; against the 12-s target of
`ref/notes/plan-128bit-speed.md` the same absolute delta is +1% to +2%.

### 5.3 Proof bytes

| Part | H, Phase 7 (B) | H-exact (B) |
| --- | ---: | ---: |
| `c~` list | 1,048,576 | 1,048,576 |
| Flock proofs, commitments, final vectors | 522,458 | 522,458 |
| lookup `T_16` | 28,896 | 28,896 |
| PIOP rounds and claims | 6,016 | 5,616 (exact count, 3.5) |
| proof-of-work stream (`pow-128`, design) | 556 | 556 |
| Akita commitments | 792 | about 792 (estimate) |
| Akita opening | 62,974 | 63.0 to 64.5 K (estimate, planner) |
| **total** | **1,669,723 (51.0 B/sig)**, plus 560 with `pow-128` | **about 1,669.9 to 1,671.4 K with `pow-128` (51.0 B/sig)** |

### 5.4 Verifier

Unchanged within noise (estimate): three integer coefficients more at the stage-3 terminal,
one claim more in the Akita opening (9.1 ms today), the same lookup, public data and Flock
work. 0.92 s uncached, 0.46 s cached as measured for H.

## 6. Implementation checklist and test plan

Order: after `pow-128` is merged into main (the plan builder and the registry exist there);
rebase over `piop-speed` (conflicts expected only in `stage.rs`, `source_tables.rs`,
`lookup.rs` context lines). Two steps, each green on its own: step A (value forms, `Q_6`,
`u` exact) and step B (the `z` slack, `G_main4` count 9, catalogs). Step A alone already
yields H-u if open question 1 is answered that way.

### 6.1 Files

`crates/agg-witness`

1. `src/params.rs`: `BZ = GAMMA1 - BETA - 1`, `BU = 2 GAMMA2 + 1`, `SLACK_Z = 130,915`,
   `STEP_U = 63,079`, with tests recomputing them from FIPS 204 values (`5 STEP_U + 2^16 =
   2 BU + 1`, `SLACK_Z + 2^17 = 2 BZ + 1`).
2. No change to the base-16 H-exact layout behind the `h-exact` feature (it stays the checker
   of Section 6.3). `docs/witness.md` W2 keeps describing that layout; the PIOP encoder has its
   own padded encoding (2.1).

`crates/agg-pcs-config`

3. `src/catalogs.rs`: a `Relation` enum defined in the lowest crate both use (here or in
   `agg-witness`), re-exported by `agg_piop::manifest`; `Group::num_polynomials(self, relation)` (`Main`:
   8 or 9); `BatchLayout::new(d_sig, relation)`; `row_name` adds `x` for H-exact
   (`mldsa-44-l16x-batch-...`), H unchanged; `CATALOG_ROWS: &[(usize, Relation)]` with all 13
   sizes for both; `scalar_layouts` over both relations (dedup keeps one `G_main16` row set).
4. `src/work_plan.rs`: `WorkPlan::for_records(records, relation)`; `matches_plan` compares
   the relation's layout.
5. `src/commit.rs`: no logic change; the setup comes from the selected grouped row.
6. `tests/schedule_catalog_regen.rs`: iterate `(d_sig, relation)`; `update_catalog_rows`
   MUST NOT prune H rows when planning H-exact rows. `tests/commit_open.rs`: dummy tables in
   the H-exact alphabets (bits, `[-3, 2]`), both rounds, one opening, at 256 and 1,024.
7. `src/schedules/agg_mldsa_fp64_signed_width4.aks`: regenerated with the 26 new rows (one
   run off CPU 2, Section 2.3); width-16 artifact untouched (the byte-for-byte test proves it).

`crates/agg-piop`

8. `src/kernel/mod.rs`: `LeafPoly::Q6` (roots `[0, 2, 6]`); kernel tests for exactness and
   for the compiled leaf path (`kernel/tests.rs` `leaf_polynomials_vanish_exactly...`).
9. `src/manifest.rs`: `Relation { H = 0, HExact = 1 }`; `CommitmentManifest::for_batch(records,
   relation)` (keep a `for_batch(records)` = H shim for callers and tests); a `ValueForm {
   lanes: &[Column], weights: &[i64], constant: i64 }` per value (`z`, `u`, `v`, carries)
   replacing the mixed-radix `DigitLayout` use in the L16 layer; new columns `BzLo, BzHi, Az,
   BuLo, Du` (names `bz_lo`, `bz_hi`, `az`, `bu_lo`, `du`), alphabets `Bool` and
   `Balanced(6)`, alphabet codes 0 and 6; `group_columns(role, relation)`;
   `leaf_digits()` (H: `Z_hi` base 4, `U_hi` base 8; H-exact: `D_u` base 6) and a new
   `bool_digits()` (H: none; H-exact: `B_zhi, a_z`); `stack_columns(0, relation)`;
   `POLYNOMIAL_COUNT`/`STAGE3_POLYNOMIALS` become methods; `binding_bytes` pushes the relation
   id; `is_leaf_digit` admits base 6. Tests: the pinned event-0 prefix for H-exact
   (`[2, 0, 0x44, 1, N, N*, 1, 6]`, `G_main4` `[1, 1, 0, v, v, 9, 0, 6, 0, 0, 0, 0, 0, 0, 0]`),
   H's pinned bytes unchanged, H and H-exact bytes differ, 13,434.5 elements per signature at
   `2^15`, `matches_plan` for both.
10. `src/source_tables.rs`: `encode_record` by the relation's forms with the canonical
   choices of 2.1, `record_values` decodes by the forms, `validate_group` checks the new
   alphabets, padded rows per 2.1; `MainTables::build(manifest, ...)` unchanged otherwise.
11. `docs/scripts/spec_checks.py` (not a crate file; lands with step A): item 12, the checks of
   4.4 (block unions, constants, padding, `Q_6`), so `python3 -I` reproduces 2.1.
12. `src/virtual_polys.rs`: `compose_digits`/`compose_table`/`residual_table`/`ValueClaims`
   by `ValueForm`; `leaf(6, p)`.
13. `src/range.rs`: `digits()` returns the relation's leaf digits; base 6 -> `Q6`.
14. `src/stage2.rs`: `degree(manifest)` (5 or 4), `gamma_count(manifest)` (8 or 7), leaves from
   `range::digits`; `Stage2Claims` length from the manifest.
15. `src/stage3.rs`: `ring` terms from the forms; `link` for the leaf digits; `Term::Bool` on
   the bool digits in the `eq(r^(2))` group; gamma map of 3.4 (`gamma_link`, `gamma_bool`,
   reductions offset `1 + links + bools`); `GAMMA_COUNT` as a method (44 or 45); terminal
   check with the forms.
16. `src/stage.rs`: stage degree from the spec (already `spec.degree`); no change on
   `piop-speed` beyond what the new terms need (`split_linear` already ignores narrow lanes).
17. `src/claims.rs`: claim vectors sized by the manifest.
18. `src/transcript.rs`: `STAGE_DEGREES` and the gamma and claim counts from the manifest
   in `Registry::new(manifest, plan)`; on `pow-128`, `stage.gamma` and `stage.round` sites
   from the same source.
19. `src/pow.rs` (from `pow-128`): the plan builder takes the stage degrees and gamma counts
   from the manifest; `docs/scripts/pow_plan.py` gets `--relation h-exact` and prints the
   H-exact anchors; tests pin them (3.7).
20. `src/wire.rs`: flags byte = relation id, accept 0 and 1; message widths and claim counts
   from the manifest; size accounting (`ProofSize`).
21. `src/verifier.rs`, `src/prover.rs`, `src/mode_b.rs`, `src/binding.rs`, `src/pcs.rs`,
   `src/opening.rs`: thread the relation (prover: from the caller; verifier: from
   `RelationPolicy`, header flag checked first); `TableCommitter::new(manifest)` picks the
   relation's work plan; 39 polynomials in the opening.

`crates/agg-cli`, `crates/agg-bench`

22. `src/commands.rs`: `--relation h|h-exact` on `prove`, `verify`, `bench` (default `h`);
   verify prints the relation; `src/mode_a.rs`: manifest and plan from the relation (the
   Keccak side unchanged); `src/wire.rs`: nothing (event-2 items do not carry the relation).
23. `agg-bench` harness: a relation column in every output line.

Docs (with the implementation, not now): `docs/spec.md` (Appendix A's H-exact paragraph ->
normative text; Sections 4.1, 4.3, 7.4, 7.5, 10, 12 with H-exact rows), `docs/lookup.md`
(stack column names), `docs/pcs-config.md` (rows, regeneration), `docs/piop.md` Section 19,
`docs/soundness.md` (`STAGE_GAMMAS`, `STAGE_DEGREES` per relation, anchors),
`docs/deviations.md` (new entry: `u` by one six-valued digit instead of Appendix A's `a_u`
and base-4 digit; Booleanity at `r^(2)` instead of stage 1), `ref/notes/status.md`,
`ref/notes/msis-estimate.md` ("spec Section 15" -> the new section).

### 6.2 Honest round trips

* `agg-piop` mode B: H-exact at 64, 256, 1,024 records (`tests/mode_b.rs`), also with
  `N` not a power of two (padding) and `N = 1`; H unchanged at the same sizes (pinned digests
  of H MUST NOT move).
* `agg-cli` mode A: H-exact at 256 and 1,024 locally (`tests/mode_a.rs`, `tests/cli.rs`),
  cached and uncached verifier paths agree; 8,192 locally if time allows; 32,768 on the
  cloud VM in the next measurement run (both relations back to back, same VM, medians of 5).
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
header flag is rewritten to 0 is rejected (and vice versa); a flag byte 2 or 3 is rejected;
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
   FIPS 204 relation" statement but not for MSIS hardness. This changes the two results of the
   2026-10-25 decision record and needs the user's decision.
2. **Appendix A vs X1.** This design departs from spec Appendix A for `u` (one six-valued digit
   instead of `a_u` plus a base-4 digit; Appendix X). If Appendix A must be kept literally,
   Appendix X gives the exact deltas (+2,048 elements, `G_main4` at 10 polynomials, stage-2
   degree 3).
3. **Booleanity placement.** Stage 3 at `r^(2)` (chosen) vs stage 1 at `tau` (two more
   reductions). Both are sound; the choice is a cost question only. Confirm in review.
4. **Catalog regeneration placement.** Locally off CPU 2 (20 to 35 minutes, estimate,
   competing with other agents' builds) or on the cloud VM before the next 32K run.
5. **Plan encoding.** Keep `docs/soundness.md` 6.4 unchanged (chosen; the relation is bound by
   the manifest bytes in the same event) or add a relation byte to the plan body (moves H's
   anchors). Settle when the `pow-128` review closes.
6. **`v` exactness.** H-exact keeps `v in [-2^31, 2^31)` (the honest bound is `|v| <=
   67,068,447 < 2^26`). `v` is not part of any FIPS 204 check or of `zeta`; a tighter `v` only
   matters for (6.1), which has 9 bits of slack. No change proposed.
7. **ML-DSA-65/87.** The same construction applies with their `gamma_1, gamma_2, beta`: for
   each set, search the single-digit `u` form (`(c - 1) A = 2 B_u + 1 - 2^16`, `A <= 2^16`,
   even `c`) and the `z` slack; not done here.

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
