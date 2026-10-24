# Lookup-based range argument for the ML-DSA-44 aggregator (proposal, prototype measured)

Status: proposal. A standalone prototype (`crates/agg-piop/src/lookup.rs`) proves and
verifies the argument below on the kernel's public API; nothing is integrated into the
stages, the prover, the wire format or the PCS catalog. Spec summary: `docs/spec.md`
Appendix C. Sections 1 to 6 are the argument, its soundness, the column counts, the
costs, the prototype and the integration plan; the Handoff at the end records the state
for a resuming agent.

References: LogUp (Haboeck, ePrint 2022/1530, Lemma 5), LogUp-GKR (Papini and Haboeck,
ePrint 2023/1284, Sections 3-4, Propositions 1-2; `ref/LogUpGKR-eprint-2023-1284.pdf`),
Aerie (Section C.1 and C.4, work nonces and difficulties), Akita (Remark 10.41: no range
guarantee from a width; Section 4.6: commitments formed before their batch is known).

## 1. The argument

**Why LogUp-GKR.** Akita commits dense small-integer vectors only (full-field, small
balanced digits, one-hot shapes). Helper columns with field-sized entries (plain LogUp's
inverses `1/(X - f)`, grand-product accumulators, Plookup's sorted union) are not
committable as small digits, and Lasso/Shout need sparse one-hot commitments of
`2^16 x 2^25` cells per chunk, which dense Akita cannot afford (`ref/notes/risks.md` R5).
LogUp-GKR commits only the looked-up columns and one multiplicity column per table: the
fractional sum is proven by a GKR circuit of degree-3 sum-checks that never commits an
intermediate. Aerie-style structured range tables (`Q_16` on `P_Z`) do not reach 16 bits
(degree `2^16`).

**Layout L16.** Every range is a 16-bit lookup into the single table `T_16 = [0, 2^16)`,
except the two small top digits, which keep the cheap coded leaf path:

| Value | Decomposition | Proven range | Columns per position | Range mechanism |
| --- | --- | --- | ---: | --- |
| `z` | `z + 2^17 = z_lo + 2^16 z_hi` | `z_lo in [0, 2^16)`, `z_hi in [0, 4)`, so `z in [-2^17, 2^17)` | 2 | lookup; `leaf4` + `link` |
| `u` | `u + 2^18 = u_lo + 2^16 u_hi` | `u_hi in [0, 8)`, so `u in [-2^18, 2^18)` | 2 | lookup; `leaf8` + `link` |
| `v` | `v + 2^31 = v_lo + 2^16 v_hi` | both in `[0, 2^16)`, so `v in [-2^31, 2^31)` | 2 | two lookups |
| `b_0..b_5` | bits | Boolean, `w_1' < 44` | 6 | unchanged (`bool_b`, `range_w1`) |
| `kappa^(1)_t, kappa^(2)_t` | one column each (was 3) | `[0, 2^16)` | per lane, on `D_1, D_2` | lookup |
| `kappa^(3)` | one column (was 2) | `[0, 2^16)` | on `D_3` | lookup |

Committed values are stored balanced so that the honest producer fits a signed width-16
Akita preset: `Z_lo = z_lo - 2^15`, `U_lo = u_lo - 2^15`, `V_lo = v_lo - 2^15`,
`V_hi = v_hi - 2^15`, `Z_hi = z_hi - 2` (base 4), `U_hi = u_hi - 4` (base 8); carries are
stored as they are (`0 <= kappa <= 1023` honestly). The lookup lane is the affine shift
`f = stored + 2^15`, so a claim on `f` is a claim on the committed column. Spec (4.1)
holds with bases `(2^16, 4)`, `(2^16, 8)`, `(2^16, 2^16)`: `z = Z_lo + 2^16 Z_hi + 2^15`,
`u = U_lo + 2^16 U_hi + 2^15`, `v = V_lo + 2^16 V_hi + 2^15` (all three constants are
`C = 2^15`). Padded rows hold the digits of zero: `Z_lo = U_lo = V_lo = -2^15`,
`Z_hi = U_hi = V_hi = 0`.

Two deviations from the first design (Handoff, "superseded"), both forced by soundness:

1. *No offset-tagged shared table.* The first design merged `T_16` and `T_11` into one
   rational identity with `T_11` shifted by `2^16`. A committed value is an arbitrary
   `F_p` element, so Lemma 5 only puts a looked-up value in the *union* of the tables: a
   `T_16` column could hold `65,536..67,583` and the tagged `v_hi` any of
   `-2^16..2^11 - 1`. No integer offset separates tables (column `c` with offset `o_c`
   may take `tau - o_c` for any foreign entry `tau`). The prototype therefore proves one
   identity per table (test `two_tables_verify_and_a_cross_table_value_is_rejected`
   books a `T_8` cell valued 300 under `T_16`, which the shared identity would accept).
2. *No `T_11` at all.* `v` needs no exact range: spec (6.1) lifts `Delta = 0` to the
   integers whenever `|v| < (p/2 - 2^50)/q ~ 2^40`. With `v + 2^31 = v_lo + 2^16 v_hi`,
   both halves 16 bits, `|Delta^Z| < 2^49 + 2^29 + 2^24 + 2^18 + 2^23 2^31 < 2^54.1 < p/2`.
   The carries need no exact range either (`docs/bridge.md` Section 4: any carry range
   below `(p - W_t)/2` is sound; `y + 2 kappa <= 1 + 2^17` keeps (9.2) an integer
   equation), so they are looked up in `T_16` directly and the `T_10`, `T_8` tables and
   the shift trick of the first design are not needed. One table, one identity, one
   multiplicity column `m`.

**Stacks.** The looked-up columns of one domain form a *stack*: column `c` occupies cells
`c 2^vars ..` of a lane of `2^depth` cells, `depth = vars + log2(slots)`, `slots` the
column count rounded up to a power of two (missing slots are public zero columns; zero is
an entry, so the honest prover counts them in `m`). The L16 stacks:
`[Z_lo | U_lo | V_lo | V_hi]` on the main domain (`depth = d_main + 2`, no padding),
`kappa^(1)` (6 lanes in 8 slots on `D_1`), `kappa^(2)` (6 in 8 on `D_2`), `kappa^(3)`
(1 on `D_3`).

**Identity.** For the challenge `X in K` (sampled after the commitments of every
looked-up column and of `m`), with `L` looked-up cells `g_i` (padding included) and
`m(t)` the number of cells equal to `t`:
```
sum_i 1/(X - g_i) = sum_{t < 2^16} m(t)/(X - t).
```
Each side is the root of a binary *fraction tree* of projective pairs `(p, q)`: node `y`
of layer `k` has children `y` and `y + 2^k` in layer `k + 1` (the child bit is the top
variable, so the two halves of a stored layer are the two child lanes of the sum-check),
`p_k = p_0 q_1 + p_1 q_0`, `q_k = q_0 q_1`. Stack leaves are `(1, X - g)`, table leaves
`(m(t), X - t)`. The verifier checks, per challenge, `sum_stacks p/q = p_T/q_T` and that
every root denominator is nonzero (`q_root` is the product of all leaf denominators, so
this also excludes `X = g_i`).

**Sessions.** All trees (one per stack and one for the table, per challenge) are
top-aligned and share every challenge. Session 1: the prover sends
`p_1(0), p_1(1), q_1(0), q_1(1)` per tree (layer 1, the root's children); the verifier
forms the roots, checks the identity, samples `mu_1` and sets the layer-1 claims at
`(mu_1)` by linearity. Session `s = 2 .. D_max` (`D_max` the deepest tree): one batched
sum-check of `s - 1` rounds and degree 3, coefficients `gamma_p, gamma_q` per active tree
(independent), reducing every claim `p_{s-1}(r), q_{s-1}(r)` at the common point `r` to
layer `s`:
```
p_{s-1}(r) = sum_y eq(r, y) (P_0 Q_1 + P_1 Q_0)(y),   q_{s-1}(r) = sum_y eq(r, y) (Q_0 Q_1)(y),
```
`P_c(y) = p_s(y + c 2^(s-1))`. The groups on `kernel::Group` (eq point `r`, `s - 1`
rounds, degree 3):

- one *layer group*: lanes `P_0, P_1, Q_0, Q_1` (halves of the stored layer `s`, as
  `Source::Field`) of every tree deeper than `s`; instances `gamma_p: P_0 Q_1 + P_1 Q_0`,
  `gamma_q: Q_0 Q_1`;
- one *stack bottom group* per stack of depth `s`: lanes `v_0, v_1` (halves of the
  stacked lane, `Source::Ints`). With `q = (X - v_0)(X - v_1)`, `p = 2X - v_0 - v_1`, the
  `X`-dependence goes into the coefficients and the challenges are summed:
  `{sum (gamma_q X^2 + 2 gamma_p X): 1}`, `{-sum (gamma_q X + gamma_p): v_0 + v_1}`,
  `{sum gamma_q: v_0 v_1}`, so the bottom group of two challenges costs that of one;
- one *table bottom group* per table of width `s`: lanes `m_0, m_1` (halves of `m`) and
  the public index halves `t_0(y) = y`, `t_1 = y + 2^(s-1)`, with `q = X^2 - X(t_0 + t_1)
  + t_0 t_1` and `p = X(m_0 + m_1) - m_0 t_1 - m_1 t_0`, coefficients summed likewise.

After the rounds (point `rho`), the prover sends `P_0, P_1, Q_0, Q_1` at `rho` per
continuing tree and the two leaf-lane values per ending stack (`v_0, v_1`) or table
(`m_0, m_1`), shared by all challenges. The verifier evaluates every group with
`kernel::expected_terminal` (the index lanes itself: `t_0(rho) = sum_k 2^k rho_k`,
`t_1 = t_0 + 2^(s-1)`), checks the running claim, samples `mu_s`, and continues at
`(rho, mu_s)`. A stack ending at `s` leaves one claim `g(rho, mu_s) = (1 - mu_s) v_0 +
mu_s v_1` on its stacked lane; it splits into column claims at the low coordinates,
`g(pi) = sum_c eq(pi_top, c) f_c(pi_low)` (`StackShape::column_weights`), and a claim on
`f_c` is one on the stored column plus its `2^15` shift. The table leaves `m(pi_m)` at
`pi_m in K^16`.

Note on the stack bottom: `p` and `q` are symmetric in `(v_0, v_1)`, so a swapped leaf
pair passes its session; it changes the merged claim, which the opening rejects (test
`tampered_messages_and_claims_are_rejected`). This is the usual GKR input-layer check,
not a gap.

**Challenges.** `X` once (or `X_1, X_2`, Section 2), then per session `gamma` (a vector of
`2 * active trees`), `s - 1` round challenges and `mu_s`.

**Verifier cost.** `O(D_max^2)` `K` operations: `sum_s (s - 1)` compressed-message
evaluations, one `expected_terminal` per group, one `eq` evaluation per session and
`#trees` inversions for the roots; measured below 1 ms (Section 4).

**Proof size.** Per tree `4` layer-1 elements; per session `3(s - 1)` round elements,
`4` per continuing tree and `2` per ending stack or table; 16 bytes per element
(`LookupShape::proof_elements`). Layout L16, one challenge: 1,015 elements (16,240 B) at
`N = 1,024`, 1,246 (19,936 B) at 8,192 and 1,431 (22,896 B, 0.70 B per signature) at
`N = 2^15`; two challenges: 1,327 (21,232 B), 1,582 (25,312 B) and 1,799 (28,784 B).
The integrated protocol adds the seven digit evaluations of `m` (112 B); the stage-3
reductions join stage 3's existing rounds and add only their terminal values.

## 2. Soundness

Errors per transcript before grinding, `|K| = p^2 ~ 2^128`, union-bounded with spec
Section 12.1. `L` = looked-up cells (padding included), `M = 2^16`, `D = D_max`.

| Term | Bound | `N = 1,024` | `N = 8,192` | `N = 2^15` |
| --- | --- | ---: | ---: | ---: |
| LogUp identity, one `X` (Lemma 5 + Schwartz-Zippel, LogUp-GKR Prop. 2) | `(L + M - 1)/(|K| - M)` | `2^-105.9` | `2^-103.0` | `2^-101.0` |
| same, two independent `X_1, X_2` | square of the above | `2^-211.8` | `2^-205.9` | `2^-201.9` |
| GKR (rounds `3/|K|` each, one `gamma` vector and one `mu` per session, `mu_1`) | `(3 D(D-1)/2 + 2(D-1) + 1)/|K|` | `736/|K|` | `949/|K|` | `1,106/|K|` |
| Four new evaluation reductions in stage 3 (Section 6) | `4 d_main/|K|` | | | `100/|K|` |
| Removed: 40 `b16_prod`, 93 leaves, 53 links, 25 carry-digit reductions | `-211 d_main/|K|` | | | `-5,275/|K|` |

`L` is `4 2^d_main + 8 |D_1| + 8 |D_2| + |D_3|`: 4,395,008, 34,672,640 and 138,690,560;
`D = d_main + 2` (the main stack). Lemma 5 needs `char F_p = p > L` (by a factor of
`2^36`) and pairwise distinct table entries; it holds for *any* `m in F_p`, so the multiplicities need
no range proof and their seven Akita digits no alphabet check. The GKR term follows
Proposition 1 of LogUp-GKR with independent coefficients per claim (one `1/|K|` per
`gamma` vector: the bad set of a nonzero linear form; Aerie's conservative assigned bound
`l/|K|` for a vector of `l` coefficients gives `sum_s 2 #active_s/|K|` = `184/|K|` with
one challenge or `368/|K|` with two at `2^15`, in place of `26/|K|`; the total stays
below `2^-116`). The root check is exact; the merge at `mu_s` costs `1/|K|` per session
because one false claim suffices to carry on.

Ledger at `N = 2^15` (spec 12.1 recomputed for L16): stage sum-checks `12 d_main`
(degrees still 4, 5, 3: `range_w1` and `leaf8`), zero-checks `19 d_main` (the 19
Booleanity and range instances), leaves and links `4 d_main`, linear identities
`(13 + 1 + 6 + 1 + 41) d_main` (62 reductions minus 25 carry-digit ones plus 4 lookup
ones), gammas 3: `2,428/|K|` (was `7,603/|K|`); plus GKR `1,106/|K|`: `3,534/|K| <
2^-116.2` for every field term except the LogUp one.

**The LogUp term is the critical one** (`2^-101` at `2^15`, above every other term).
Options, all measured or bounded in Section 4:

- (i) *One `X` plus a work nonce at the `X` site.* Aerie's rule (C.4, (17)) gives the
  difficulty `g = ceil(log2(2^128 e))`: 23 bits at 1,024, 26 at 8,192, 28 at `2^15`
  (expected `2^g` sponge previews). Two obstacles: the pinned Akita transcript caps
  difficulties at `MAX_GRINDING_BITS = 25` (`akita-transcript/src/grinding.rs`) and
  searches serially, so 26 and 28 bits need our own nonce format and a parallel search;
  and this repository grinds nothing yet (`docs/deviations.md` P3).
- (ii) *Two independent challenges* `X_1, X_2` (one vector challenge): two complete
  tree sets, every session batched; error `2^-202`, no grinding. The rounds and the
  final claims are shared (the leaf lanes are the same columns), so proof size grows only
  by the second set's layer-1 and terminal values (+5.9 KB at `2^15`) and the cost by the
  second set's tree build and layer sessions (the bottom session is shared, Section 1).
- (iii) `X` in `F_{p^4}`: every GKR operation in a degree-4 extension (about 3x the
  arithmetic) and a kernel field change. Not recommended.

Recommendation: (ii), on the measurements of Section 4. It is the only option that keeps
every field term below `2^-115` without new grinding machinery, and it costs a fixed
fraction of the lookup prover rather than an `N`-dependent nonce search.

**Extraction** (spec 12.2 with L16). Akita yields `F_p` values for every committed column
(Remark 10.41). Except with the terms above: every looked-up cell is an entry of `T_16`,
so `Z_lo + 2^15, U_lo + 2^15, V_lo + 2^15, V_hi + 2^15, kappa in [0, 2^16)` as integers;
`leaf4`/`link` and `leaf8`/`link` give `Z_hi in [-2, 1]`, `U_hi in [-4, 3]`; by (4.1),
`z in [-2^17, 2^17)`, `u in [-2^18, 2^18)` (relation `H(2^17, 2^18)` unchanged, so
`zeta'` of spec (3.5) is unchanged) and `v in [-2^31, 2^31)`, enough for (6.1); every
carry is an integer in `[0, 2^16)`, enough for (9.2) (`docs/bridge.md` Section 4).
`m` is unconstrained and need not be constrained.

## 3. Column counts

Per position (main domain): 23 columns (`z` 5, `u` 5, `v` 7, `b` 6) become 12
(`Z_lo, Z_hi, U_lo, U_hi, V_lo, V_hi, b_0..b_5`). Committed elements per signature:

| | Before (spec 4.1) | L16 | Change |
| --- | ---: | ---: | ---: |
| Main domain | 23,552 | 12,288 | -47.8% |
| `D_1` carries (`N = 2^15`, `2^19` cells) | 288 (18 columns) | 96 (6) | |
| `D_2` parities and carries (`2^15` cells) | 24 (6 + 18) | 12 (6 + 6) | |
| `D_3` carries (`2^14` cells) | 1 (2) | 0.5 (1) | |
| Multiplicities `m` (7 digits x `2^16` cells) | 0 | 14 | |
| **Total at `N = 2^15`** | **23,865** | **12,410.5** | **-48.0%** |
| Total at `N = 1,024` (physical cells, 13-variable minimum) | 24,048 | 12,936 | -46.2% |

At `N = 1,024`, `m` costs 448 elements per signature (7 x `2^16` / 1,024); it is
independent of `N`. Akita inner digits per position: the width-4 columns keep one; the
four width-16 columns need one balanced base-`2^16` digit if the preset forces it
(`num_digits_inner_for_bound(16, 16) = 1`) or three base-`2^6` digits under the planner's
cost-optimal choice (`docs/pcs-config.md` Section 2: estimate 51,961 B against 37,625 B
for width 4, one polynomial at 13 variables). So 12 (forced) or 20 (planner) inner
digits per position instead of 23; which one is cheaper for the batched opening is for
the planner to decide (open question, Section 6).

Instances removed from the stages: 40 `b16_prod` (stage 1), 93 of 95 leaves (stage 2),
53 of 55 links (stage 3), 25 carry-digit evaluation reductions (stage 3); added: 4
lookup reductions (stage 3). Claims before stage 1 fall from 26 to 13, before stage 2
from 18 to 6.

## 4. Prover cost

### 4.1 Arithmetic (`K`-multiplications; estimates from the kernel's code paths)

Per leaf of a fraction tree (`2^D` leaves), counting the kernel's generic per-pair path
(3 interpolation points with the split `eq`, bilinear products with their coefficient,
one weight product per point, one fold per lane and pair):

| Part | 1 challenge | 2 challenges | Why |
| --- | ---: | ---: | --- |
| Tree build | 2 | 4 | bottom layer 1 per node, upper layers 3 per node |
| Layer sessions `s < D` | 10 | 20 | 24 per pair in round 0 (18 products, 4 folds, 2 scalings), 16 later |
| Bottom session `s = D` | 9 | 9 | 16 per pair plus 2 `Ints` folds; shared by the challenges |
| **Total per leaf** | **21** | **33** | |

Leaves per signature (main stack 4,096, carry stacks with their padding slots, table
`2^16 / N`): 4,356 at `N = 1,024`, 4,240.5 at 8,192, 4,234.5 at `2^15`. Per signature:
about 91K `K`-multiplications with one challenge and 144K with two at 1,024 (89K and 140K
at `2^15`), i.e. 270K to 430K `F_p` multiplications. The four stage-3 reductions it adds
cost about 4K per signature (4 linear lanes, degree 2).

Current range machinery, same counting: stage 2's main group (17 `P_Z` lanes, degree 5)
costs 346 products per pair from round 2 on (`17 x 5` Horner evaluations of degree 4,
weights, `eq`) plus 17 folds, about 96 per position or 98K per signature; with `def_y1`
and the carry groups, stage 2 is about 100K per signature. The lookup also removes
stage 1's `b16_prod` share (about 36K per signature: 28 of the 34 lanes of the main
`eq(tau)` group) and the stage-3 links (about 20K), so the range machinery it replaces is
about 160K per signature against 91K (one challenge) or 144K (two).

### 4.2 Measured (this machine, 7 threads, default build)

Baseline: stage 2 at `N = 1,024` on `7ed4a32`, median of 6 runs of
`mode_b_round_trip_at_1024_signatures` with `AGG_STAGE_TRACE=1` (load average 0.5 to
1.6): **464.6 ms, 454 us per signature** (stage 1 277.0 ms, stage 3 374.5 ms). Per group (message + bind): main leaves 346.9 ms,
`def_y1` 58.2, carry leaves on `D_1, D_2, D_3` 25.0, 19.5, 1.7. Stage 2 at 8,192 is not
run here (memory rule); `ref/notes/bench-cloud.md` phase 5 has 5,020.5 ms at 32,768 on
the 16-vCPU VM (another machine).

Prototype on the same kernel (`7ed4a32` exported to the session scratchpad plus
`lookup.rs`, `CARGO_TARGET_DIR=target/lookup-snap`), the full L16 layout (main stack and
three carry stacks, one table), `timing_at_1024_and_8192_signatures`, load average 3.9
to 5.3 including the run itself (a first run at 4.5 to 6.6 agreed within 5%):

| `N` | `X` | Prove | per signature | trees | layer sessions | bottom session | verify | proof | stored layers | peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 1 | 260.9 ms | 255 us | 18.2 | 159.1 | 84.0 | 0.50 ms | 16,240 B | 136 MiB | 172 MiB |
| 1,024 | 2 | 379.9 ms | 371 us | 30.7 | 259.8 | 86.9 | 0.56 ms | 21,232 B | 272 MiB | 321 MiB |
| 8,192 | 1 | 1,818.9 ms | 222 us | 140.9 | 966.6 | 711.4 | 0.62 ms | 19,936 B | 1,060 MiB | 1,336 MiB |
| 8,192 | 2 | 2,794.3 ms | 341 us | 283.3 | 1,801.3 | 707.3 | 0.69 ms | 25,312 B | 2,120 MiB | 2,292 MiB |

(layer sessions = sessions minus the bottom one; the next-largest session, `s = D - 1`,
is 53.9 / 96.9 ms at 1,024 and 452.0 / 844.8 ms at 8,192.) The second challenge costs
1.46x at 1,024 and 1.54x at 8,192, not 2x, because the bottom session is shared.

**Against stage 2 at `N = 1,024`: 255 us per signature (one challenge) or 371 us (two)
against 454 us, i.e. 0.56x or 0.82x of stage 2 alone,** while the lookup replaces more
than stage 2's range part. Lane-share estimates from the same traces: stage 1's
`b16_prod` share about 206 ms (82% of the main `eq(tau)` group, 80% of the carry
groups), stage 2's leaves 393.1 ms (measured), stage 3's links and the narrower `ring`
and `w_A z` groups about 160 ms: about 760 ms of the 1,116 ms of stages 1 to 3. Adding
the lookup with two challenges (380 ms) and the new reductions (about 20 ms, estimate),
stages 1 to 3 plus the lookup would cost about 0.76 s instead of 1.12 s at 1,024 (-32%,
estimate), before the PCS gains (46% fewer committed elements).

For information only: on the other agent's uncommitted kernel rework (main tree at
16:07, `kernel/` split into `mod, coded, field, plan, tests`), the same test gives
188.2 / 268.3 ms at 1,024 and 1,565.6 / 2,103.0 ms at 8,192 (one / two challenges); the
bottom session drops to 45 ms and 454 ms. Stage 2 was not re-measured on it.

### 4.3 Extrapolation to `N = 2^15` (estimates, this machine)

The per-signature cost is flat from 1,024 to 8,192 (the trees are linear in the leaves),
so 4x the 8,192 figures: about 7.3 s with one challenge and 11.2 s with two on 7 threads;
stored layers 4.1 and 8.3 GiB, peak RSS of the stage about 5.2 and 9.0 GiB (the trees are
dropped before stage 1, so this is a stage peak, not an addition to the process peak;
stage 1's main group holds about 4.6 GB at `2^15`, `docs/piop.md` Section 6).
Mitigations if memory matters: recompute the bottom layer from the columns in session
`D - 1` instead of storing it (halves the stored layers), and a 16-bit cell lane that
folds through two `2^16`-entry tables.

### 4.4 Option (i): the work nonce (measured)

`grinding_rate`: Akita's `preview_grinding_predicate` per candidate (clone the
Blake2b512 sponge, absorb the nonce, squeeze 32 bytes) costs 724 ns serially (Akita's
`search_grinding_nonce` is serial) and 102.9 ns per candidate on 7 threads. Expected cost
at the `X` site:

| `N` | difficulty | serial (Akita) | 7 threads | extra cost of the second challenge |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 23 bits | 6.1 s | 0.86 s | 0.12 s (measured) |
| 8,192 | 26 bits | 48.6 s | 6.9 s | 0.98 s (measured) |
| `2^15` | 28 bits | 194 s | 27.6 s | about 3.9 s (estimate) |

The nonce is about 7x dearer than the second challenge even with a parallel search, and
above 25 bits it is not expressible in the pinned Akita transcript. Hence option (ii).

## 5. Prototype (`crates/agg-piop/src/lookup.rs`)

Standalone: no stage, prover, wire or PCS file is touched; `lib.rs` gained one line.

- `Table` (`T_w`, index halves, `index_evaluation`), `StackShape` (columns, variables,
  table; `slots`, `depth`, `leaves`, `column_weights`), `LookupStack::new(&[&[i32]],
  table)` (stacks and zero-pads the looked-up values `f`, not the stored balanced ones).
- `multiplicities(stacks, tables)`: parallel histogram per table, `OutOfTable` on an honest
  out-of-range cell; `multiplicity_digits` / `compose_multiplicity`: the seven balanced
  base-16 digits of `G_m` (capacity `7 (16^7 - 1)/15 = 125,269,879`; at `2^15` the
  largest honest count is that of the entry 0 when every record is padding,
  `3 2^25 + 8 2^19 + 8 2^15 + 2^14 = 105,136,128`, 84% of the capacity; only a batch in
  which over 84% of all cells share one value would need an eighth digit, which honest
  signatures cannot produce because `z_lo, u_lo, v_lo` spread over `2^16` values; the
  encoder asserts the capacity).
- `LookupShape` (public layout, `challenges` 1 to 4, `validate`, `max_depth`, `lookups`,
  `identity_error_log2`, `proof_elements`).
- `Fs`: a local SHAKE256 transcript (framed labelled absorptions, limbs rejection-sampled
  as in `Transcript::sample_base`, every squeeze absorbed back). The PIOP transcript's
  fixed event registry is not extended by a prototype. The caller's `binding` stands for
  the commitments that must precede `X`.
- `FractionTree`: layers `0 .. depth - 1` stored as `K` pairs, built in parallel from the
  leaves (the bottom layer straight from the `i32` cells: `1` `K`-multiplication per node;
  upper layers `3`); layer `s` is freed after session `s`.
- `prove(stacks, tables, multiplicities, challenges, binding) -> LookupOutput { proof,
  claims, stats }` and `verify(shape, proof, binding) -> LookupClaims`: Section 1 exactly,
  through `kernel::{Group, GroupInstance, Lane::Source(Source::Field | Source::Ints),
  Term::{Product, Affine}, evaluate_compressed, expected_terminal}` only. The prover checks
  the root identity and every session's claim (`prove_unchecked` skips both for the
  negative tests). `verify` returns the stack and multiplicity claims; the caller checks
  them (Akita in the integration, `evaluate_mle` in the tests).
- Unit tests (`cargo test -p agg-piop --release --lib lookup`, 7 tests, 0.1 s):
  fraction-tree roots against the direct sums; digit round trip; honest main stacks at
  `d_main = 10, 13, 14` (table deeper, shallower, equal); the full L16 layout with carry
  stacks at one and two challenges (main stack deeper than the table); two tables, a
  cross-table cheat rejected; a wrong multiplicity and `v_hi = 2^16` under two cheating
  multiplicity columns rejected (identity); a tampered round message, swapped terminal
  values, a swapped table leaf pair, a changed layer-1 value, a truncated proof, another
  binding and another layout rejected; a swapped stack leaf pair caught by the claim
  (Section 1). Every honest test checks the claims against `evaluate_mle` of the stacked
  lane, of every column (through `column_weights`) and of `m`.
- Ignored timing tests: `timing_at_1024_and_8192_signatures` (Section 4; sizes from
  `AGG_LOOKUP_SIGS`, 7 threads, warmup plus five runs) and `grinding_rate`.

## 6. Integration plan

Nothing below is implemented; it is the order in which the pieces would land.

1. *PCS* (agg-pcs-config agent). A preset `agg_mldsa_fp64_signed_width16` (values in
   `[-2^15, 2^15)`; decide between the planner's three base-`2^6` inner digits and one
   forced base-`2^16` digit by planning both). Groups: `G_main16` (`Z_lo, U_lo, V_lo,
   V_hi`, width 16, `d_main` variables, round 1), `G_main4` (`Z_hi, U_hi, b_0..b_5`, width
   4, round 1), `G_1` (6 carries, width 16, round 2), `G_2` (6 parities + 6 carries; one
   width-16 group or a width-4/width-16 split, round 2), `G_3` (1 carry, round 2), `G_m`
   (7 width-4 digit polynomials on `2^16` cells, round 2: `m` counts the carries, which
   exist only after the bridge seed). Six groups instead of four; grouped catalog rows
   mixing width-16 and width-4 families under Akita 4.6's common root alphabet must be
   confirmed with the planner (open question 1).
2. *Witness* (`agg-witness`, `source_tables`). The L16 digits, the balanced storage of
   Section 1, single-column carries, the stacks and `m` (`lookup::multiplicities`), the
   digits of `m`.
3. *Transcript and wire.* New events after round 2: `X_1, X_2` (one vector challenge,
   sampled with or before `tau`); per session `gamma`, `s - 1` round messages and
   challenges, terminals and leaves, `mu_s`; the seven digit evaluations of `m` at
   `pi_m`. Wire items: the `LookupProof` (16,240 to 28,784 B, Section 1) plus 112 B. Event 0 binds the layout id (`digit layout id = 1`).
4. *Stages.* A new lookup stage before stage 1 (it depends only on the commitments and
   `X`). Stage 1: 26 instances (no `b16_prod`). Stage 2: `def_y1` (6), `leaf4[Z_hi]`,
   `leaf8[U_hi]`; the 18 carry-digit claims before stage 2 become 6. Stage 3: `ring` on 12
   lanes with (4.1) of Section 1, 2 links, 37 old reductions and 4 new ones: the main
   stack claim (one group at `eq(pi_low)` over the four width-16 columns with the
   coefficients `column_weights(pi)`, minus the `2^15` shifts), and the three carry
   stacks on `D_1, D_2, D_3`. `G_m` is opened at `pi_m` directly (its own group point).
5. *Ledger* (spec 12.1): the rows of Section 2 (spec Appendix C).
6. *Kernel* (other agent's area). The lookup uses only `Source::Field` and `Source::Ints`
   lanes and `Product`/`Affine` terms; the measured hot spots (Section 4) are the generic
   per-pair path on `K` lanes and the `Ints` bottom session. A 16-bit cell lane folding
   through two `2^16`-entry tables (`(1 - r) v_0 + r v_1 = LUT_0[v_0] + LUT_1[v_1]`) and a
   bucketed first round for the table tree would serve this stage.

## Handoff (2026-10-24)

### State

Written by the continuing agent (Opus 5.5) after steps 1 to 6 of the first agent's plan.
Submodule `code` at `7ed4a32`; nothing committed. Files of this work (uncommitted):
`crates/agg-piop/src/lookup.rs` (new), one line `pub mod lookup;` in
`crates/agg-piop/src/lib.rs`, this file, `docs/spec.md` Appendix C (new, proposal), and
`ref/LogUpGKR-eprint-2023-1284.pdf` in the top-level repo. No stage, prover, wire,
kernel or PCS file was edited. Another agent was rewriting the kernel concurrently
(`kernel.rs` split into `kernel/{mod,coded,field,plan,tests}.rs`, uncommitted at 16:14);
`lookup.rs` builds and its tests pass both on `7ed4a32` and on that in-progress tree with
no change: it uses `Group::{new, round_message, bind, claim, terminal_values}`,
`GroupInstance`, `Lane::Source(Source::Field | Source::Ints)`, `Term::{Product, Affine}`,
`evaluate_compressed`, `expected_terminal` only, so the `Lane::Lookup` change did not
touch it. Clippy pedantic: no warning in `lookup.rs`.

Reproduce (never CPU 2, never `cargo clean`):
- unit tests: `RUSTUP_TOOLCHAIN=stable taskset -c 0,1,3-15 cargo test -p agg-piop
  --release --lib lookup`;
- timings: the same with `lookup::tests::timing -- --ignored --nocapture`
  (`AGG_LOOKUP_SIGS=1024,8192`) and `lookup::tests::grinding -- --ignored --nocapture`;
- the like-for-like timings of Section 4.2 ran on `git archive 7ed4a32` plus `lookup.rs`
  and the `lib.rs` line, in the session scratchpad, with
  `CARGO_TARGET_DIR=target/lookup-snap` (a separate target dir: sharing `target/` from
  another source path overwrites the main tree's `agg-piop` artifacts, which happened
  once at 15:58 and costs the other agent one rebuild of the workspace crates).
  `target/lookup-snap` (855 MB) may be deleted.
- Raw logs: session scratchpad `bench/` only (`stage2-1024-run{1..6}.log`,
  `lookup-timing-run{1,2}.log`, `lookup-timing-rework-kernel.log`,
  `grinding-run1.log`, `lookup-grinding-run2.log`).

### Progress, in the first agent's step order

1. Done: `ref/LogUpGKR-eprint-2023-1284.pdf` (462,315 B, SHA-256 `3cdb575a...352e`).
2. Done: `lookup.rs` (Section 5), 7 unit tests and 2 ignored timing tests.
3. Done before the kernel rework landed: stage 2 at 1,024 on `7ed4a32`, median 464.6 ms
   (Section 4.2; per-group figures there).
4. Done: Sections 1 to 6.
5. Done: `docs/spec.md` Appendix C (proposal).
6. Done: the report to the caller.

### What changed against the first agent's design (and why)

- Superseded: the shared rational identity over `T_16` and `T_11` tagged by integer
  offsets. Unsound (Section 1, deviation 1). Now one identity per table.
- Superseded: `T_11` for `v_hi` (and `T_10`, `T_8` or the shift trick for the carries).
  Not needed (Section 1, deviation 2): `v + 2^31 = v_lo + 2^16 v_hi` with both halves in
  `T_16`, carries in `T_16`. Column count unchanged (12 per position); one table and one
  multiplicity column (7 x `2^16` cells instead of 7 x `(2^16 + 2^13)`).
- Changed: the final lookup claims. The first design had the prover send per-column
  values `f_c(rho_low)` at the bottom; the bottom session cannot produce them (the column
  bit is the last bound variable, so the leaf lanes end as mixtures), and a separate MLE
  evaluation would cost about `2^d` per column. Now each stack leaves one claim on its
  stacked lane at `(rho, mu)`, split into column claims by `column_weights` and reduced
  in stage 3 by one instance per stack with `K` coefficients (4 instances, not 6).
- Changed: the bottom groups merge every challenge into one instance set (the leaf lanes
  are the same columns), so the second challenge does not double the bottom session.
- Confirmed: multiplicities need no range (Lemma 5 for any `m in F_p`, `p > L`); the
  seven width-4 digits hold every honest count (Section 5, capacity 125,269,879 against
  at most 105,136,128 at `2^15`).
- Confirmed by measurement: the LogUp term needs a remedy; option (ii) (two challenges)
  is about 7x cheaper than the work nonce (Section 4.4) and the nonce exceeds Akita's
  25-bit cap from `N = 8,192` on.
- Memory of the stored layers: 1,060 MiB per challenge at 8,192 (measured), about 4.1 GiB
  at `2^15` (estimate); acceptable on the 64-128 GB VM, mitigations in Section 4.3.

### Open questions

1. PCS (agg-pcs-config agent): the signed width-16 preset (planner's three base-`2^6`
   inner digits or one forced base-`2^16` digit) and grouped catalog rows mixing the
   width-16 and width-4 families (six groups, Section 6 item 1).
2. The user's choice of the LogUp remedy: two challenges (recommended) or a work nonce
   (needs a nonce format beyond Akita's 25 bits and grinding in this repository at all,
   `docs/deviations.md` P3).
3. Whether the carries should also leave the bridge-group digits entirely (proposed:
   yes, one `T_16` column per carry; the bridge memo's "carry range below `(p - W_t)/2`"
   argument is what makes it sound).
4. Kernel support (other agent): a 16-bit cell lane for the bottom session and the
   recomputed bottom layer (Section 4.3), and whether the reworked kernel's speedup of the
   `Ints`/`Field` generic path (Section 4.2, informational figures) holds once committed.
5. Stage 2 at 8,192 and both costs at `2^15` on the cloud VM, measured end to end after
   integration (only extrapolations here).

### Next steps, in order

1. PCS: plan the width-16 preset and the six-group catalog rows at `d_sig` 8 and 10
   (`docs/pcs-config.md` Section 6), compare the opening bytes and times with today's.
2. Witness and tables: the L16 digits with balanced storage, single-column carries, the
   stacks, `m` and its digits (Section 6 item 2).
3. Transcript events, wire items and the lookup stage before stage 1, reusing
   `lookup::{prove, verify}` with the PIOP transcript instead of `Fs`; then the stage
   changes and the four stage-3 reductions (Section 6 items 3 and 4); then the spec
   ledger (Appendix C becomes normative text in Sections 4.1, 4.3, 7 and 12).
4. Measure mode B at 256 and 1,024 locally, 8,192 and `2^15` on the VM; update Section 4.

### What was read

First agent (Fable 5.1): `docs/spec.md` 1-7, 12, Appendix B; `docs/piop.md` 1-7, 13, 16,
17; `docs/pcs-config.md`; `ref/notes/improvements-tables.md`, `improvements-misc.md`,
`risks.md` R1/R5; `kernel.rs`, `stage.rs`, `stage2.rs`, `stage3.rs` head, `tables.rs`,
`range.rs`, `claims.rs` head, `manifest.rs` head, `transcript.rs` API; LogUp 2022/1530
Lemma 5 and Protocol 2; LogUp-GKR 2023/1284 Sections 3-4, Propositions 1-2; Aerie 4.4,
5.2, C.1; Akita Remark 4.1, Section 4.6, Remark 10.41.
Continuing agent: `AGENTS.md`; `docs/spec.md` 3-7, 9, 12; `docs/piop.md` 1-6, 16-17;
`docs/pcs-config.md`; `docs/bridge.md` Section 4 (carry ranges); `ref/notes/bench-cloud.md`
phases 4-5; `kernel.rs`, `stage.rs`, `tables.rs`, `transcript.rs`; Aerie C.1 and C.4
(difficulties `g_t`, eq. (17)); Akita `akita-transcript/src/grinding.rs` and
`proof_stream.rs` (`MAX_GRINDING_BITS = 25`, serial `search_grinding_nonce`); LogUp
Lemma 5 and Section 3; LogUp-GKR Section 3.2, Propositions 1 and 2. Paper texts:
`pdftotext -layout` of the PDFs in `ref/` (the first agent's scratchpad copies are in its
session directory `cccb66b5-.../scratchpad/papers/`).
