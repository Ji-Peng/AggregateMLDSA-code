# Lookup-based range argument for the ML-DSA-44 aggregator (integrated, wire version 1)

Status: integrated (branch `lookup-integration`, 2026-10-24). The lookup into `T_16` is
the range mechanism of relation H in mode A and mode B: `crates/agg-piop/src/lookup.rs`
runs inside the PIOP transcript (event 7), the six Akita groups of Section 6 are committed
under two presets with generated catalogs, and the wire format is version 1. The
normative text is `docs/spec.md` Sections 4, 7, 10 to 12 (former Appendix C); departures
are `docs/deviations.md` L1 to L7, D2. Sections 1 to 6 are the argument, its soundness
(with the comparison of remedies for the identity term, Section 2.1), the column counts, the
costs (prototype and integrated), the module and the integration as implemented; the
Handoff at the end records the state for a resuming agent.

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
`V_hi = v_hi - 2^15`, `Z_hi = z_hi - 2` (base 4), `U_hi = u_hi - 4` (base 8), and the carries
`K = kappa - 2^15` (integrated code; the first version of this section stored the carries
as they are, `docs/deviations.md` L2). Every looked-up value is the affine shift
`f = stored + 2^15`, so a claim on the stored lane is a claim on the committed column; the
tree leaves are `1 / ((X - 2^15) - stored)`. Spec (4.1) holds with bases `(2^16, 4)`,
`(2^16, 8)`, `(2^16, 2^16)` and `(2^16)` (unsigned): `z = Z_lo + 2^16 Z_hi + 2^15`,
`u = U_lo + 2^16 U_hi + 2^15`, `v = V_lo + 2^16 V_hi + 2^15`, `kappa = K + 2^15` (all four
constants are `C = 2^15`). Padded rows hold the digits of zero: `Z_lo = U_lo = V_lo = -2^15`,
`Z_hi = U_hi = V_hi = 0`; padded carries are `K = -2^15`, and so are the padding slots of
the carry stacks (looked-up value 0).

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
column count rounded up to a power of two (missing slots are public columns of looked-up
value zero, stored `-2^15`; zero is an entry, so the honest prover counts them in `m`). In
the integrated code the stacked lane is the storage itself (`G_main16` is one `i32` lane of
four columns, each carry level one lane of eight slots), so the lookup reads it without a
copy. The L16 stacks:
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

**Challenges.** `X_1, X_2` (one vector challenge; Section 2), then per session `gamma` (a
vector of `2 * active trees`), `s - 1` round challenges and `mu_s`.

**Verifier cost.** `O(D_max^2)` `K` operations: `sum_s (s - 1)` compressed-message
evaluations, one `expected_terminal` per group, one `eq` evaluation per session and
`#trees` inversions for the roots; measured below 1 ms (Section 4).

**Proof size.** Per tree `4` layer-1 elements; per session `3(s - 1)` round elements,
`4` per continuing tree and `2` per ending stack or table; 16 bytes per element
(`LookupShape::proof_elements`). Layout L16, one challenge: 1,015 elements (16,240 B) at
`N = 1,024`, 1,246 (19,936 B) at 8,192 and 1,431 (22,896 B, 0.70 B per signature) at
`N = 2^15`; two challenges: 1,327 (21,232 B), 1,582 (25,312 B) and 1,799 (28,784 B).
The integrated protocol adds the seven digit evaluations of `m` (112 B); the stage-3
reductions join stage 3's existing rounds and add no terminal value (their columns are
opened at `r^(3)` anyway).

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
`D = D_max = max(d_main + 2, 16)` (`LookupShape::max_depth`): the main stack, except at
`N* = 8` (`d_main = 13`), where the 16-variable table tree is the deepest and the GKR term
is `391/|K|` (`344/|K|` with `d_main + 2`). Lemma 5 needs `char F_p = p > L` (by a factor of
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

**The LogUp term is the critical one** (`2^-101` at `2^15` with one challenge, above every
other term). Decision (user, 2026-10-24): two independent challenges, unless a cheaper
method with a written proof is found. Section 2.1 records the bounded search; none of the
alternatives is cheaper, so the integrated protocol samples `X_1, X_2` (manifest
`LOOKUP_CHALLENGES = 2`, bound in event 0).

### 2.1 Remedies for the identity term (comparison)

Notation as above; `N(X)` is the numerator of `R(X) = sum_i 1/(X - f_i) - sum_t m(t)/(X - t)`
over the common denominator `prod_{a in S} (X - a)`, `S = {f_i} u T`, so `deg N <= |S| - 1
<= L + M - 1`.

**Lemma (identity at independent points).** If some `f_i` is not in `T`, then `R != 0`
(Lemma 5 of LogUp: the coefficient of `1/(X - f_i)` is a count in `[1, L]`, nonzero in
`F_p` because `p > L`, whatever `m` in `F_p^T`), so `N != 0`. `N` is fixed by the
commitments, which precede the challenges; a challenge `X` passes the root check only if
`X` is not in `S` (the root denominators are nonzero) and `N(X) = 0`. With `X_1, X_2`
independent and uniform in `K`,
`Pr[N(X_1) = N(X_2) = 0] <= ((L + M - 1) / |K|)^2`.
The GKR sessions are shared by both tree sets, but the event "accept although some cell is
out of `T`" splits into (a) both identities hold at their true roots, bounded above, or
(b) some tree set's true root violates its identity, so the prover's layer-1 values for it
are false and the batched sessions must accept a false claim: the GKR term (the same
`3 D (D - 1)/2 + 2 (D - 1) + 1` over `|K|`, because the second tree set adds claims to the
same batched sessions, not sessions). Union bound: `eps_X^2 + eps_GKR`.

| Remedy | Identity error at `N = 2^15` | Extra prover cost (one challenge = 1) | Proof bytes at `2^15` | Status |
| --- | ---: | --- | ---: | --- |
| one `X` | `(L + M - 1)/|K| = 2^-101.0` | 1 | 22,896 | below the 128-bit ledger |
| (i) one `X` and a 28-bit work nonce | `2^-129.0` (difficulty 28) | `+27.6 s` nonce search on 7 threads (Section 4.4), about 7x the second challenge | 22,896 + nonce | above Akita's 25-bit cap; no nonce format here (deviation P3) |
| (ii) two independent `X_1, X_2` (adopted) | `2^-201.9` | 1.46x (1,024) to 1.54x (8,192) of the lookup prover (prototype, Section 4.2); integrated figures in Section 4.5 | 28,784 | the identity term needs no grinding; the GKR rounds still need grinding to Aerie's schedule (P3): the lookup is at about `2^-117.9` before grinding (`1,106/|K|`) |
| (iii) `X` in `F_{p^4}` (trees over the degree-4 extension, challenges of the sessions in `K`) | `(L + M - 1)/p^4 = 2^-229` | every tree node and lane product in `F_{p^4}`: 3 `K`-multiplications per product (Karatsuba) plus a constant, about 3x the arithmetic of one challenge, i.e. 2x that of (ii); a kernel field change | about 2x of one `X` (every layer value and message in `F_{p^4}`) | rejected: dearer than (ii) |
| (iv) tighter counting (bound `deg N` by the distinct out-of-table values) | none: an adversary can make every `f_i` distinct, `deg N = L + M - 1` is reachable | 0 | 22,896 | no improvement in the worst case |
| (v) the conjugate `X^p` as a second point | none: `N` has coefficients in `F_p`, so `N(X) = 0` implies `N(X^p) = 0` | 0 | 22,896 | no information |
| (vi) a product denominator `1/((X_1 - f)(X_2 - f))` in one tree | `2 (L + M - 1)/|K|` (bivariate total degree) | about 1x | 22,896 | worse than one `X` |
| (vii) split the lookups into `k` parts with their own multiplicities | `(L + k M - 1)/|K|` for the union of the parts | `k` multiplicity columns | more | worse |
| (viii) a 25-bit nonce (Akita's cap) | `2^-126.0` at `2^15`; 128 bits only while `L + M < 2^25`, i.e. up to `N = 2^12` (`2^-128.9`) | about 3.4 s on 7 threads (`2^25` previews at 102.9 ns, Section 4.4) | 22,896 + nonce | not 128 bits at `2^13` and above; needs the nonce format anyway |

The only remedies that bring the identity term to 128 bits at `N = 2^15` are (i), (ii)
and (iii); (ii) is the cheapest of the three by the measurements of Section 4 and needs
neither grinding for that term nor a kernel field change. This is not 128 bits for the
lookup: its GKR sessions are `1,106/|K| ~ 2^-117.9` at `2^15` and each degree-3 round is
`3/|K| ~ 2^-126.4`, above `2^-128`, so the lookup, like the other field terms
(`3,534/|K| < 2^-116.2` in all), reaches `2^-128` per site only with grinding to Aerie's
schedule, which is not implemented (deviation P3, spec 12.1). Implementation-level savings that keep the soundness of (ii) and are
not done yet: build both tree sets in one pass over the stacked lanes (the leaves are read
once, the two `(p, q)` pairs per node stored side by side), and recompute the bottom layer
in session `D - 1` instead of storing it (Section 4.3). Both change only the prover.

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
digits per position instead of 23. Decided (deviation L3): the planner's estimated proof
is the same for one, two or three inner digits at the `G_main16` layouts, so the preset
forces one digit, 12 inner digits per position.

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

### 4.5 One against two challenges on the current kernel, and the integrated protocol

Standalone (`timing_at_1024_and_8192_signatures` with `AGG_LOOKUP_SIGS=1024,2048`, the
CLMUL build of this branch in `target/clmul`, 7 threads, warmup plus five runs, medians;
2026-10-24 20:29, load average 6.6 to 6.8 including the run and another agent's tests):

| `N` | `X` | Prove | per signature | trees | sessions | bottom session | verify | proof | stored layers | peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 1 | 221.4 ms | 216 us | 26.0 | 196.3 | 65.1 | 0.51 ms | 16,240 B | 136 MiB | 255 MiB |
| 1,024 | 2 | 312.0 ms | 305 us | 44.5 | 265.3 | 59.0 | 0.57 ms | 21,232 B | 272 MiB | 464 MiB |
| 2,048 | 1 | 401.5 ms | 196 us | 48.6 | 354.2 | 118.6 | 0.55 ms | 17,424 B | 268 MiB | 576 MiB |
| 2,048 | 2 | 617.1 ms | 301 us | 97.6 | 519.4 | 122.8 | 0.61 ms | 22,544 B | 536 MiB | 756 MiB |

The second challenge costs 1.41x at 1,024 and 1.54x at 2,048 (the bottom session is
shared), the same ratio as the prototype's on the older kernel.

Integrated, end to end (`ref/notes/bench-local.md`, "Lookup integration": CLMUL build,
interleaved with `4247f61`, medians of 5 after 1 warmup, load average 6.1 to 8.7), mode B:

| `N` | Prove before / after | Lookup | Stages 1-3 before / after | Main-domain commitments before / after | Committed cells per signature before / after | Aggregate without bits before / after |
| ---: | --- | ---: | --- | --- | --- | --- |
| 256 | 0.674 / 0.655 s | 113 ms | 175 / 77 ms | 37 / 18 ms | 24,928 / 14,672 | 76,425 / 92,630 B |
| 1,024 | 1.157 / 1.135 s | 283 ms | 404 / 218 ms | 151 / 45 ms | 24,040 / 12,932 | 102,984 / 121,508 B |
| 2,048 | 1.710 / 1.817 s | 572 ms | 740 / 386 ms | 190 / 82 ms | 23,940 / 12,658 | 135,948 / 157,057 B |

Mode A at 1,024 / 2,048: prove 2.629 -> 2.555 s and 3.957 -> 4.109 s (the Flock part is the
same code), verify 306 -> 283 and 392 -> 346 ms uncached, 700.8 -> 718.9 and 405.1 -> 415.4 B
per signature. The aggregate without the public bits grows by 16.2, 18.8, 18.5 and 21.1 KB at 256, 512,
1,024 and 2,048 (mode A: 18.5 and 21.1 KB at 1,024 and 2,048): the lookup proof minus the
shorter stage claims. The layout halves the committed cells and makes the main-domain commitments
2 to 3x faster, but with the kernel of `docs/piop.md` Section 18 the old range machinery cost
only about 0.2 ms per signature, so the lookup (0.28 ms per signature) makes the prover
break-even at 256 to 1,024 and 6% slower at 2,048 locally; the estimate of Section 4.2
(-32%) predates that kernel. Stage 3 stays at about 0.12 ms per signature: its four 16-bit
columns are generic `Ints` lanes. The prover-side levers are in the Handoff.

## 5. The module (`crates/agg-piop/src/lookup.rs`)

The prototype's construction, now driven by the PIOP transcript:

- `Table` (`T_w`, index halves, `index_evaluation`), `StackShape` (columns, variables,
  table, storage `shift`; `slots`, `depth`, `leaves`, `padding = -shift`, `column_weights`,
  `column_claim`, which removes the public padding slots from a stacked claim),
  `LookupStack::new(&[&[i32]], table)` (zero shift, a copy, the prototype tests),
  `LookupStack::with_shift` and `LookupStack::borrowed(shape, &stacked)` (the integrated
  stacks, no copy).
- `multiplicities(stacks, tables)`: parallel histogram per table of `stored + shift`,
  `OutOfTable` on an honest out-of-range cell; `multiplicity_digits` /
  `compose_multiplicity`: the seven balanced base-16 digits of `G_m` (capacity
  `7 (16^7 - 1)/15 = 125,269,879`; at `2^15` the largest honest count is that of the entry 0
  when every record is padding, `3 2^25 + 8 2^19 + 8 2^15 + 2^14 = 105,136,128`, 84% of the
  capacity; only a batch in which over 84% of all cells share one value would need an
  eighth digit, which honest signatures cannot produce because `z_lo, u_lo, v_lo` spread
  over `2^16` values; the encoder asserts the capacity).
- `LookupShape` (public layout, `challenges` 1 to 4, `validate`, `max_depth`,
  `tree_count`, `session_counts(s)` for the registry and the wire parser, `lookups`,
  `identity_error_log2`, `proof_elements`); the integrated shape is
  `CommitmentManifest::lookup_shape` (four stacks, `T_16`, two challenges, shift `2^15`).
- `LookupTranscript` (absorb, challenges at a `LookupSite`): implemented by the PIOP
  transcript (sites `agg-mldsa/v1/lookup/<site>`, `docs/deviations.md` L5) and by the
  standalone `Fs` (a SHAKE256 transcript bound to the layout and a caller binding, for the
  unit tests and timings).
- `FractionTree`: layers `0 .. depth - 1` stored as `K` pairs, built in parallel from the
  leaves (the bottom layer straight from the `i32` cells: `1` `K`-multiplication per node;
  upper layers `3`); layer `s` is freed after session `s`.
- `prove_in(stacks, tables, multiplicities, challenges, transcript, check)` and
  `verify_in(shape, proof, transcript) -> LookupClaims`; `prove`/`verify` wrap them with
  `Fs`. Section 1 exactly, through `kernel::{Group, GroupInstance, Lane::Source(Source::Field
  | Source::Ints), Term::{Product, Affine}, evaluate_compressed, expected_terminal}` only.
  `check` runs the prover's root-identity and session self-checks (tests pass `false` to
  hand a dishonest witness to the verifier). The claims go to stage 3 (stacks) and to the
  multiplicity-digit claims (table).
- Unit tests (`cargo test -p agg-piop --release --lib lookup`, 7 tests): as before (tree
  roots, digits, honest layouts at one and two challenges, two tables and a cross-table
  cheat, wrong multiplicities, tampered messages and claims, a swapped stack leaf pair
  caught by the claim). The integrated tests are in `tests/stages.rs` (Section 6).
- Ignored timing tests: `timing_at_1024_and_8192_signatures` (Section 4; sizes from
  `AGG_LOOKUP_SIGS`, 7 threads, warmup plus five runs) and `grinding_rate`.

## 6. Integration (as implemented)

1. *PCS* (`agg-pcs-config`, `docs/pcs-config.md`). Preset `AggFp64SignedWidth16` with one
   forced balanced base-`2^16` inner digit (deviation L3: the planner's estimate is the same
   for one, two or three digits). Six groups in wire order `G_main16` (width 16), `G_main4`
   (width 4), `G_1`, `G_2` (one width-16 group for the parities and the carries), `G_3`
   (width 16), `G_m` (width 4, Akita's final group; deviation L4). Two catalogs: the width-16
   one with the scalar rows of its four groups, the width-4 one with the scalar rows of
   `G_main4` and the 13 grouped rows; the planner accepted every mixed-family grouped row
   (Akita freezes the bound per group), so open question 1 is closed.
2. *Tables* (`source_tables`, in `agg-piop`; `agg-witness` and `agg-bridge` are unchanged).
   `MainTables`: `G_main16` as one stacked `i32` lane, `G_main4` as `i8` columns, encoded
   from the derived `z, u, v, w_1'` (`RecordValues`). `BridgeGroupTables`: the stacked
   balanced carries of `agg_bridge::parity::compute`, the parities, `Y`, and `m` with its
   digits over every looked-up cell of both rounds (`recount`). `Cells` reads either width.
3. *Transcript and wire.* Event 7 is the lookup (`X_1, X_2`, layer 1, `mu_1`, sessions
   `2 .. D`, the seven digit claims); `tau` follows. The registry and the wire parser
   derive every count from `N` (`session_counts`). Wire version 1 carries the six
   commitments (two at event 1, four at event 4), the `LookupProof` and the 7 digit
   claims; version 0 is refused.
4. *Stages.* Stage 1: 26 instances (no `b16_prod`). Stage 2: `def_y1` (6), `leaf4[Z_hi]`,
   `leaf8[U_hi]`; 6 `K^(1)` claims before it (was 18). Stage 3: `ring` with (4.1) of
   Section 1, 2 links, 37 reductions and the 4 stack reductions (one gamma per stack, one
   part per column with the scale `eq(pi_top, c)`, deviation L6); 44 instances, 31 terminal
   claims. The prover and verifier share `prover::prove_piop` / `verifier::verify_piop`.
5. *Ledger*: spec Section 12.1 (normative; Section 2 above).
6. *Kernel*: unchanged API; the 16-bit columns are `Source::Ints` lanes (uncoded, generic
   path). A 16-bit coded lane (two `2^16`-entry tables per fold) is the next prover-side
   step if the stage-3 groups on `Ints` lanes dominate (Section 4.5).

Integrated tests (`cargo test -p agg-piop --release`): `tests/stages.rs` (honest PIOP at
256 and 10 signatures with every claim against the table extensions; the lookup rejects
16-bit cells out of `T_16` with `z` unchanged (`Z_lo +- 2^16`, `Z_hi -+ 1`), `V_hi = 2^16`,
carries `2^16` and negative at every level, a wrong multiplicity (moved and added), a
tampered layer-1 value, GKR round message, layer terminal, table leaf pair, swapped
multiplicity-digit claims and a truncated session; swapped stage-3 column claims rejected
at stage 3; a value valid only for another range (`Z_hi = 2`, valid for `U_hi`; `U_hi = 4,
-5`; a carry stored unshifted, valid in `T_16` but meaning `kappa + 2^15`; a padding slot
holding a table entry) rejected at the leaf, the bridge identity or the stack reduction;
a low part valid only for the wider range `[0, 2^17)` (`U_lo + 2^16` with `U_hi - 1`,
`V_lo + 2^16` with `V_hi - 1`: `u`, `v` unchanged) rejected by the lookup; every old witness
tampering at its stage), `tests/mode_b.rs` (round trips at 64, 256, 1,024; end-to-end
tampering including a wrong multiplicity; a stored 16-bit value outside the width-16
interval is refused by Akita's commit, so the prover fails and no aggregate exists: the
verifier's rejection of a looked-up value outside `T_16` is tested only at the PIOP level,
in `tests/stages.rs`, without commitments and opening; an end-to-end test would need a
commit path that skips Akita's admission bound), `tests/single_thread.rs` (a whole prover
and verifier run with a one-thread rayon pool, called from a plain thread and from a pool
worker, under a timeout: the regression test of the round-1 deadlock, deviation P12),
`tests/prove_verify.rs`, `tests/bridge.rs`, `tests/source_tables.rs`,
`tests/virtual_checker.rs` (the L16 tables against the integer relation), `tests/commit_open.rs`;
`agg-cli` `tests/mode_a.rs` (mode A round trips and tampering).

## Handoff (2026-10-24, integration)

### State

Branch `lookup-integration` of the code repository (worktree
`/home/jipeng/code/aggmldsa-wt/lookup`, base `main` at `4247f61`), merged into `main` by
fast-forward after the review fixes (one-thread commit deadlock, deviation P12; test and
ledger wording); the Flock branch `flock-bytes` is not merged yet and rebases onto it.
Everything of Section 6 is implemented:
the width-16 preset and both catalogs (all 13 batch sizes), the six groups, the L16 tables,
the lookup in the PIOP transcript, the stage changes, the verifier, wire version 1, the
normative ledger (spec Section 12.1), deviations L1 to L7 and D2. The base-16 digit path is
removed (L1). `agg-witness` and `agg-bridge` are unchanged; in `agg-cli` only
`src/mode_a.rs` (one error mapping, one size line) and the tampering cases of
`tests/mode_a.rs` changed (the old cases named base-16 digits).

### Reproduce (never CPU 2, never `cargo clean`)

- tests: `RUSTUP_TOOLCHAIN=stable taskset -c 0,1,3-15 cargo test --release --workspace`;
- catalogs: `docs/pcs-config.md` Section 6 (44 minutes for both families on this machine);
- measurements: `ref/notes/bench-local.md` (lookup integration) has the commands, the
  revisions and the load averages; the baseline is `4247f61` built in
  `target/baseline-clmul` from a detached worktree.

### Open

1. Prover-side savings that keep the soundness of two challenges (measured costs, Section
   4.5: lookup 0.28 ms per signature, stage 3 0.12 ms per signature at 2,048): one pass
   building both tree sets (the trees are 98 of the 617 ms of the standalone lookup at 2,048),
   the bottom layer recomputed instead of stored (Section 4.3), a dedicated layer-session path
   (sessions `D - 1` and `D` are 179 and 123 ms there), and a 16-bit coded lane for the `Ints`
   columns of stage 3 and of the bottom session.
2. `N = 8,192` and `2^15` on the cloud VM (only extrapolated here, Section 4.5).
3. Grinding for the remaining field terms (`2^-116.2` before grinding, deviation P3).
4. The spec's ledger counts each reduction instance as a `d_main/|K|` linear identity
   (conservative, inherited from revision 2); a tighter accounting would credit the
   reductions to the stage sum-checks.
