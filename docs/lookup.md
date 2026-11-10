# Lookup-based range argument for the ML-DSA aggregator

Numbers, stacks and counts below are ML-DSA-44's unless stated; the stacks, projection digits
and lift of ML-DSA-65 and ML-DSA-87 are in the final section of `docs/spec.md`.

The lookup into `T_16` is the range mechanism of relation H in mode A and mode B: `crates/agg-piop/src/lookup.rs`
runs inside the PIOP transcript (event 7), the seven Akita groups of Section 6 are committed
under two presets with generated catalogs. The
normative text is `docs/spec.md` Sections 4, 7, 10 to 12; departures
are `docs/deviations.md` L1 to L7, D2 and K24 (the projection of `v`). Sections 1 to 6 are the argument, its soundness
(with the comparison of remedies for the identity term, Section 2.1), the column counts, the
costs, the module and the integration as implemented.

References: LogUp (Haboeck, ePrint 2022/1530, Lemma 5), LogUp-GKR (Papini and Haboeck,
ePrint 2023/1284, Sections 3-4, Propositions 1-2),
Aerie (Section C.1 and C.4, work nonces and difficulties), Akita (Remark 10.41: no range
guarantee from a width; Section 4.6: commitments formed before their batch is known).

## 1. The argument

**Why LogUp-GKR.** Akita commits dense small-integer vectors only (full-field, small
balanced digits, one-hot shapes). Helper columns with field-sized entries (plain LogUp's
inverses `1/(X - f)`, grand-product accumulators, Plookup's sorted union) are not
committable as small digits, and Lasso/Shout need sparse one-hot commitments of
`2^16 x 2^25` cells per chunk, which dense Akita cannot afford.
LogUp-GKR commits only the looked-up columns and one multiplicity column per table: the
fractional sum is proven by a GKR circuit of degree-3 sum-checks that never commits an
intermediate. Aerie-style structured range tables (`Q_16` on `P_Z`) do not reach 16 bits
(degree `2^16`).

**Layout L16.** Every range is a 16-bit lookup into the single table `T_16 = [0, 2^16)`,
except the small top digits, which keep the cheap coded leaf path. The quotient `v` has no
range check of its own: its projection outputs `k` are range-checked instead (deviation K24,
spec Section 6):

| Value | Decomposition | Proven range | Columns per position | Range mechanism |
| --- | --- | --- | ---: | --- |
| `z` | `z + 2^17 = z_lo + 2^16 z_hi` (former relation H-u, removed) | `z_lo in [0, 2^16)`, `z_hi in [0, 4)`, so `z in [-2^17, 2^17)` | 2 | lookup; `leaf4` + `link` |
| `u` | `u + 190,465 = u_lo + 63,079 u_hi` (relation H-exact, deviation L8) | `u_lo in [0, 2^16)`, `u_hi in [0, 6)`, so `u in [-190,465, 190,465]` exactly | 2 | lookup; `leaf6` + `link` |
| `v` | `v + 2^31 = v_lo + 2^16 v_hi` (honest encoding) | none; `|v|_c <= 2^38` through `k` | 2 | none (committed, not looked up) |
| `k = <Psi_o, v_i>` | `k + 2^37 = k_0 + 2^16 k_1 + 2^32 k_2 + 2^35 k_3` | `k_0, k_1 in [0, 2^16)`, `k_2, k_3 in [0, 8)`, so `k in [-2^37, 2^37)` | 4 per (row, signature), on `D_k` | two lookups; two `leaf8` + links |
| `b_0..b_5` | bits | Boolean, `w_1' < 44` | 6 | unchanged (`bool_b`, `range_w1`) |
| `kappa^(1)_t, kappa^(2)_t` | one column each | `[0, 2^16)` | per lane, on `D_1, D_2` | lookup |
| `kappa^(3)` | one column | `[0, 2^16)` | on `D_3` | lookup |

Committed values are stored balanced so that the honest producer fits a signed width-16
Akita preset: `Z_lo = z_lo - 2^15`, `U_lo = u_lo - 2^15`, `V_lo = v_lo - 2^15`,
`V_hi = v_hi - 2^15`, `Z_hi = z_hi - 2` (base 4), `U_hi = u_hi - 3` (base 6), the carries
`K = kappa - 2^15` (`docs/deviations.md` L2) and the projection digits `K_0 = k_0 - 2^15`,
`K_1 = k_1 - 2^15`, `K_2 = k_2 - 4`, `K_3 = k_3 - 4` (base 8). Every looked-up value is the affine shift
`f = stored + 2^15`, so a claim on the stored lane is a claim on the committed column; the
tree leaves are `1 / ((X - 2^15) - stored)`. Spec (4.1) holds with bases `(2^16, 4)`,
`(2^16, 2^16)` and `(2^16)` (unsigned): `z = Z_lo + 2^16 Z_hi + 2^15`,
`v = V_lo + 2^16 V_hi + 2^15`, `kappa = K + 2^15` (constant `C = 2^15` each); `u` takes the
step form (4.2), `u = U_lo + 63,079 U_hi + 31,540`; `k = K_0 + 2^16 K_1 + 2^32 K_2 + 2^35 K_3 +
2^34 + 2^31 + 2^15` (bases `(2^16, 2^16, 8, 8)`, signed). Padded rows hold the digits of zero:
`Z_lo = V_lo = -2^15`, `U_lo = -31,540` (`u_lo = 1,228`, `u_hi = 3`), `Z_hi = U_hi = V_hi = 0`;
padded carries are `K = -2^15`, and so are the padding slots of the carry stacks (looked-up
value 0); padded records project to `k = 0`, digits `(K_0, K_1, K_2, K_3) = (-2^15, -2^15, -4, 0)`.
The radix form of `z` (`Z_lo`, `Z_hi`, `leaf4`) belongs to the removed relation H-u, which
this layout was first built for; the code commits `z` in H-exact's form below, under both
H-FIPS (the default) and H-exact.

Relation H-exact (`docs/h-exact.md` step B, deviation L9) keeps this lookup unchanged: `z`
is `z + 130,993 = B_zlo + 2^16 B_zhi + 130,915 a_z` with `B_zlo` looked up in the slot of
`z_lo` (stored `S_zlo = B_zlo - 2^15`, column `bz_lo`) and the bits `B_zhi`, `a_z` proven by
two Booleanity zero-checks in stage 3, so the five stacks, `L` and every lookup site are the
same; only the honest multiplicities move (padded cells look up 65,457 instead of 0).
Relation H-FIPS (`docs/h-exact.md` Section 10.7, deviation L10) puts `l_lo` (`L_lo - 2^15`)
in the slot of `u_lo` and the hint slack `u_h` (stored `u_h - 2^15`) in the free slot 6 of the
level-2 stack, which then has 7 columns in its 8 slots: the slot count, the depths, `L` and the
sites are those of H-exact, and the stack-2 reduction (7.7) reads `u_h` as its seventh column.

Two design rules, both forced by soundness:

1. *No offset-tagged shared table.* Merging two tables into one rational identity, e.g.
   `T_16` and `T_11` with `T_11` shifted by `2^16`, is unsound. A committed value is an arbitrary
   `F_p` element, so Lemma 5 only puts a looked-up value in the *union* of the tables: a
   `T_16` column could hold `65,536..67,583` and the tagged `v_hi` any of
   `-2^16..2^11 - 1`. No integer offset separates tables (column `c` with offset `o_c`
   may take `tau - o_c` for any foreign entry `tau`). The module therefore proves one
   identity per table (test `two_tables_verify_and_a_cross_table_value_is_rejected`
   books a `T_8` cell valued 300 under `T_16`, which the shared identity would accept).
2. *No `T_11` at all.* `v` needs no exact range: spec (6.1) lifts `Delta = 0` to the
   integers whenever `|v| < (p/2 - 2^50)/q ~ 2^40`. The projection certifies `|v|_c <= 2^38`,
   so `|Delta^Z| < 2^49 + 2^29 + 2^24 + 2^18 + 2^23 2^38 < 2^61.01 < p/2`; the
   outputs `k` (at most `2^36` for every valid key) fit two `T_16` digits and two
   base-8 leaf digits in `[-2^37, 2^37)`. ML-DSA-65 and ML-DSA-87 add a base-4 leaf digit
   (`[-2^39, 2^39)`, `|v|_c <= 2^40`), and their lift holds with a margin factor of only
   1.0007 and 1.0006 (`docs/spec.md`, final section).
   The carries need no exact range either (`docs/bridge.md` Section 4: any carry range
   below `(p - W_t)/2` is sound; `y + 2 kappa <= 1 + 2^17` keeps (9.2) an integer
   equation), so they are looked up in `T_16` directly and need no `T_10`, `T_8` tables
   or shift trick. One table, one identity, one
   multiplicity column `m`.

**Stacks.** The looked-up columns of one domain form a *stack*: column `c` occupies cells
`c 2^vars ..` of a lane of `2^depth` cells, `depth = vars + log2(slots)`, `slots` the
column count rounded up to a power of two (missing slots are public columns of looked-up
value zero, stored `-2^15`; zero is an entry, so the honest prover counts them in `m`). In
the integrated code the stacked lane is the storage itself (`G_main16` is one `i32` lane of
four columns, each carry level one lane of eight slots), so the lookup reads it without a
copy. The L16 stacks:
`[Z_lo | U_lo]` on the main domain (the first half of the `G_main16` lane, `depth = d_main +
1`, no padding), `kappa^(1)` (6 lanes in 8 slots on `D_1`), `kappa^(2)` (6 in 8 on `D_2`),
`kappa^(3)` (1 on `D_3`) and `[K_0 | K_1]` on the projection domain `D_k` (`depth = d_k + 1`,
`d_k = 7 + d_sig`).

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
(`LookupShape::proof_elements`). Layout L16 with the projection (K24), two challenges:
1,394 elements (22,304 B) at `N = 1,024`, 1,664 (26,624 B) at 8,192 and 1,891 (30,256 B,
0.92 B per signature) at `N = 2^15` (before K24: 1,327, 1,582 and 1,799 elements; the fifth
stack adds two trees and the main stack's depth falls by one).
The integrated protocol adds the seven digit evaluations of `m` (112 B); the stage-3
reductions join stage 3's existing rounds and add no terminal value (their columns are
opened at `r^(3)` anyway).

## 2. Soundness

Errors per transcript before grinding, `|K| = p^2 ~ 2^128`, union-bounded with spec
Section 12.1. `L` = looked-up cells (padding included), `M = 2^16`, `D = D_max`.

| Term | Bound | `N = 1,024` | `N = 8,192` | `N = 2^15` |
| --- | --- | ---: | ---: | ---: |
| LogUp identity, one `X` (Lemma 5 + Schwartz-Zippel, LogUp-GKR Prop. 2) | `(L + M - 1)/(|K| - M)` | `2^-106.7` | `2^-103.8` | `2^-101.8` |
| same, two independent `X_1, X_2` | square of the above | `2^-213.4` | `2^-207.5` | `2^-203.5` |
| GKR (rounds `3/|K|` each, one `gamma` vector and one `mu` per session, `mu_1`) | `(3 D(D-1)/2 + 2(D-1) + 1)/|K|` | `671/|K|` | `875/|K|` | `1,026/|K|` |
| Four new evaluation reductions in stage 3 (Section 6) | `4 d_main/|K|` | | | `100/|K|` |

`L` is `2 2^d_main + 8 |D_1| + 8 |D_2| + |D_3| + 2 |D_k|` (K24; before it `4 2^d_main + ...`
without `D_k`): 2,560,000, 19,992,576 and 79,970,304;
`D = D_max = max(d_main + 1, 16)` (`LookupShape::max_depth`): the main stack, except up to
`N* = 32` (`d_main <= 15`), where the 16-variable table tree is the deepest and the GKR term
is `391/|K|`. Lemma 5 needs `char F_p = p > L` (by a factor of
`2^36`) and pairwise distinct table entries; it holds for *any* `m in F_p`, so the multiplicities need
no range proof and their seven Akita digits no alphabet check. The GKR term follows
Proposition 1 of LogUp-GKR with independent coefficients per claim (one `1/|K|` per
`gamma` vector: the bad set of a nonzero linear form; Aerie's conservative assigned bound
`l/|K|` for a vector of `l` coefficients gives `sum_s 2 #active_s/|K|` = `184/|K|` with
one challenge or `368/|K|` with two at `2^15`, in place of `26/|K|`; the total stays
below `2^-116`). The root check is exact; the merge at `mu_s` costs `1/|K|` per session
because one false claim suffices to carry on.

Ledger at `N = 2^15` (round by round, spec 12.1): stage sum-checks `11 d_main` (degrees
3, 5, 3, set by the bridge Booleanity in stage 1 and `leaf8[K_2]`, `leaf8[K_3]` in stage 2;
before deviation K26 `range_w1` gave stage 1 degree 4), the zero-check point `tau` `d_main`,
the three stage batching vectors 3: `303/|K|`; plus GKR `1,026/|K|`: `1,329/|K| = 2^-117.6`
for every field term except the LogUp one.

**The LogUp term is the critical one** (`2^-101` at `2^15` with one challenge, above every
other term). Decision: two independent challenges. Section 2.1 records the bounded search;
none of the alternatives is cheaper, so the protocol samples `X_1, X_2` (manifest
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

The error and byte figures of this comparison predate the projection stack (deviation K24), except
the adopted row's `2^-203.5`; with the projection stack one `X` gives `2^-101.8` (table above) and
the adopted layout takes 1,891 elements, 30,256 B (Proof size above).

| Remedy | Identity error at `N = 2^15` | Extra prover cost (one challenge = 1) | Proof bytes at `2^15` | Status |
| --- | ---: | --- | ---: | --- |
| one `X` | `(L + M - 1)/|K| = 2^-101.0` | 1 | 22,896 | below the 128-bit ledger |
| (i) one `X` and a 28-bit work nonce | `2^-129.0` (difficulty 28) | `+27.6 s` nonce search on 7 threads (Section 4.4), about 7x the second challenge | 22,896 + nonce | above Akita's 25-bit cap (the nonce format of wire version 2 caps targets at 25 bits) |
| (ii) two independent `X_1, X_2` (adopted) | `2^-203.5` | 1.41x (1,024) to 1.54x (2,048) of the lookup prover (Section 4.5) | 28,784 | the identity term needs no grinding; the GKR rounds are ground to Aerie's schedule in wire version 2 (`docs/soundness.md`); before grinding the lookup is at about `2^-117.9` (`1,026/|K|`) |
| (iii) `X` in `F_{p^4}` (trees over the degree-4 extension, challenges of the sessions in `K`) | `(L + M - 1)/p^4 = 2^-229` | every tree node and lane product in `F_{p^4}`: 3 `K`-multiplications per product (Karatsuba) plus a constant, about 3x the arithmetic of one challenge, i.e. 2x that of (ii); a kernel field change | about 2x of one `X` (every layer value and message in `F_{p^4}`) | rejected: dearer than (ii) |
| (iv) tighter counting (bound `deg N` by the distinct out-of-table values) | none: an adversary can make every `f_i` distinct, `deg N = L + M - 1` is reachable | 0 | 22,896 | no improvement in the worst case |
| (v) the conjugate `X^p` as a second point | none: `N` has coefficients in `F_p`, so `N(X) = 0` implies `N(X^p) = 0` | 0 | 22,896 | no information |
| (vi) a product denominator `1/((X_1 - f)(X_2 - f))` in one tree | `2 (L + M - 1)/|K|` (bivariate total degree) | about 1x | 22,896 | worse than one `X` |
| (vii) split the lookups into `k` parts with their own multiplicities | `(L + k M - 1)/|K|` for the union of the parts | `k` multiplicity columns | more | worse |
| (viii) a 25-bit nonce (Akita's cap) | `2^-126.0` at `2^15`; 128 bits only while `L + M < 2^25`, i.e. up to `N = 2^12` (`2^-128.9`) | about 3.4 s on 7 threads (`2^25` previews at 102.9 ns, Section 4.4) | 22,896 + nonce | not 128 bits at `2^13` and above; needs the nonce format anyway |

The only remedies that bring the identity term to 128 bits at `N = 2^15` are (i), (ii)
and (iii); (ii) is the cheapest of the three by the measurements of Section 4 and needs
neither grinding for that term nor a kernel field change. This is not 128 bits for the
lookup: its GKR sessions are `1,026/|K| ~ 2^-118.0` at `2^15` and each degree-3 round is
`3/|K| ~ 2^-126.4`, above `2^-128`, so the lookup, like the other field terms
(`1,354/|K| = 2^-117.6` in all), reaches `2^-128` per site only with grinding to Aerie's
schedule (`docs/soundness.md`: 2 bits per GKR round, a merge 1
bit, a batching vector `bit_length(2 active)`; deviation P3). Implementation-level savings that keep the soundness of (ii),
both implemented in Section 4.6: build both tree sets in one pass over the stacked lanes (the leaves are read
once, the two `(p, q)` pairs per node stored side by side), and recompute the bottom layer
in session `D - 1` instead of storing it (Section 4.6 item 1). Both change only the prover.

**Extraction** (spec 12.2 with L16). Akita yields `F_p` values for every committed column
(Remark 10.41). Except with the terms above: every looked-up cell is an entry of `T_16`,
so `Z_lo + 2^15, U_lo + 2^15, K_0 + 2^15, K_1 + 2^15, kappa in [0, 2^16)` as integers;
`leaf4`/`link`, `leaf6`/`link` and `leaf8`/`link` give `Z_hi in [-2, 1]`, `U_hi in [-3, 2]`, `K_2, K_3 in [-4, 3]`;
by (4.1) and spec (4.2), `|z| <= 130,993` (H-exact and H-FIPS; `z in [-2^17, 2^17)` under
the removed H-u), `u in [-190,465, 190,465]` (H-exact; H-FIPS bounds `L` instead,
`docs/h-exact.md` 10) and `k in [-2^37, 2^37)`, so `|v|_c <= 2^38` by the projection
(spec Section 6, Lemma P; `V_lo`, `V_hi` are not looked up), enough for (6.1); every
carry is an integer in `[0, 2^16)`, enough for (9.2) (`docs/bridge.md` Section 4).
`m` is unconstrained and need not be constrained.

## 3. Column counts

Per position (main domain): 12 columns (`Z_lo, Z_hi, U_lo, U_hi, V_lo, V_hi, b_0..b_5`).
Committed elements per signature:

| | L16 |
| --- | ---: |
| Main domain | 12,288 |
| `D_1` carries (`N = 2^15`, `2^19` cells) | 96 (6 columns) |
| `D_2` parities and carries (`2^15` cells) | 12 (6 + 6) |
| `D_3` carries (`2^14` cells) | 0.5 (1) |
| `G_3` zero polynomials (11 x `2^14` cells, K24) | 5.5 |
| Projection digits `K_0, K_1, K_2, K_3` (`D_k`, 128 rows) | 512 |
| Multiplicities `m` (7 digits x `2^16` cells) | 14 |
| **Total at `N = 2^15`** | **12,928** |
| Total at `N = 1,024` (physical cells, 13-variable minimum) | 13,536 |

At `N = 1,024`, `m` costs 448 elements per signature (7 x `2^16` / 1,024); it is
independent of `N`. Akita inner digits per position: one per width-4 column, and one forced
balanced base-`2^16` digit per width-16 column, so 12 per position (deviation L3: the
planner's estimated proof is the same for one, two or three inner digits at the `G_main16`
layouts; its cost-optimal choice would be three base-`2^6` digits, `docs/pcs-config.md`
Section 2).

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

Leaves per signature (main stack 2,048, carry stacks with their padding slots, table
`2^16 / N`): 2,308 at `N = 1,024`, 2,192.5 at 8,192, 2,186.5 at `2^15`. Per signature:
about 48K `K`-multiplications with one challenge and 76K with two at 1,024 (46K and 72K
at `2^15`), i.e. 145K to 229K `F_p` multiplications. The five stage-3 stack reductions
cost about 5K per signature (5 linear lanes, degree 2).

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

### 4.5 One against two challenges, and the integrated protocol

Standalone (`timing_at_1024_and_8192_signatures` with `AGG_LOOKUP_SIGS=1024,2048`, the
CLMUL build, 7 threads, warmup plus five runs, medians;
load average 6.6 to 6.8 including the run and another agent's tests):

| `N` | `X` | Prove | per signature | trees | sessions | bottom session | verify | proof | stored layers | peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 1 | 221.4 ms | 216 us | 26.0 | 196.3 | 65.1 | 0.51 ms | 16,240 B | 136 MiB | 255 MiB |
| 1,024 | 2 | 312.0 ms | 305 us | 44.5 | 265.3 | 59.0 | 0.57 ms | 21,232 B | 272 MiB | 464 MiB |
| 2,048 | 1 | 401.5 ms | 196 us | 48.6 | 354.2 | 118.6 | 0.55 ms | 17,424 B | 268 MiB | 576 MiB |
| 2,048 | 2 | 617.1 ms | 301 us | 97.6 | 519.4 | 122.8 | 0.61 ms | 22,544 B | 536 MiB | 756 MiB |

The second challenge costs 1.41x at 1,024 and 1.54x at 2,048, not 2x, because the bottom
session is shared.

Integrated (mode B): committed cells per signature 14,800, 13,060 and 12,786 at 256, 1,024
and 2,048 signatures. Stage 3 takes about 0.12 ms per signature locally (CLMUL build): its
four 16-bit columns are generic `Ints` lanes.

### 4.6 Dedicated prover paths

Prover only: the proof bytes are unchanged (the round polynomial of a session is unique, so
any exact computation of it gives the same messages). The generic kernel stays as the
fallback and as the reference: `lookup::tests::dedicated_paths_match_the_generic_kernel`
proves every layout twice (`prove_with(.., dedicated)`) and compares proofs and claims byte
for byte, and `kernel::tests::dedicated_layer_groups_match_the_generic_group` compares each
dedicated group with `kernel::Group` message by message.

1. *Session `D - 1` on integer lanes, shared by both challenges.* Node `w` of layer `D - 1`
   of a stack has the leaves `a = v(w) - c`, `b = v(w + 2^(D-1)) - c` (`c` a centre of the
   stack's stored values, `|a|, |b| <= 2^15`), so with `X' = X - shift - c`, `p = 2X' - S`
   and `q = X'^2 - X'S + M` for `S = a + b`, `M = ab`. The session's lanes `P_c, Q_c` are
   affine in the four `i32` lanes `S_0, S_1, M_0, M_1` (`c` the top bit of the node), which
   do not depend on `X`. The summand of both trees of the stack,
   `sum_X gamma_p (P_0 Q_1 + P_1 Q_0) + gamma_q Q_0 Q_1`, is the bipartite quadratic form
   `c + c_s (S_0 + S_1) + c_m (M_0 + M_1 + S_0 S_1) + c_k (S_0 M_1 + S_1 M_0) + c_mm M_0 M_1`
   with coefficients from the gammas and the powers of `X'` (`SymLanes::coefficients`). One
   group of four integer lanes replaces two trees of four field lanes, and layer `D - 1` is
   never stored: layer `D - 2` is built for every challenge in one pass over the leaves from
   the elementary symmetric polynomials `e_1 .. e_4` of its four leaves (`q = X'^4 - e_1
   X'^3 + e_2 X'^2 - e_3 X' + e_4`, `p = dq/dX'`, exact in `i64`). Stored layers halve
   (8,192 signatures: 2,120 -> 1,062 MiB; `2^15`: about 8.3 -> 4.2 GiB, estimate). A
   stack whose values spread over more than `2^16` (a dishonest witness) or of depth below 3
   keeps the stored-layer path.
2. *Bottom session and session `D - 1` on `kernel::IntPairGroup`.* The first round sums
   exact `i64` products of the integer lanes weighted by the equality table without
   reduction (`mad_int`: two `64 x 64` products per term); the table from the first fold on
   holds `S_0 (, M_0)` and `W = C (1, S_1, M_1)`, so the summand is `W_0 + S_0 W_1 + M_0 W_2`
   (one or two products per point, no coefficient).
3. *Layer sessions on `kernel::LayerGroup`.* Per tree the lanes `L_1 = gamma_p P_0`,
   `L_2 = Q_1`, `L_3 = gamma_p Q_0`, `L_4 = P_1 + (gamma_q / gamma_p) Q_1` give the summand
   `L_1 L_2 + L_3 L_4`; one pass per round folds four rows into two and accumulates `q(0)`
   and the leading coefficient (the linear one from the claim, Gruen's factorisation); the
   first round reads the stored halves and computes `q(1)` too, so the group's claim comes
   from its tables. About 12 `K`-products per pair and tree in later rounds against 16 on the
   generic kernel, 18 against 24 in the first. A zero `gamma_p` (probability `2^-128`) falls
   back to the generic group.
4. *Tree build.* The branch-free product of the kernel (`fm`) and blocks filled in place
   after a parallel first touch, instead of a parallel `unzip` of the generic product.

Measured (lookup timing test, 2 challenges, interleaved runs of the generic and the
dedicated paths in one process; local, shared machine, direction only):

| `N` | threads | load average | generic | dedicated | trees | `s = D` | `s = D - 1` | `s = D - 2` |
| ---: | ---: | --- | ---: | ---: | --- | --- | --- | --- |
| 1,024 | 1 | 5.4 to 7.7 | 1,261.9 ms | 909.9 ms | 207.9 -> 160.0 | 259.0 -> 245.7 | 373.0 -> 185.8 | 178.9 -> 135.9 |
| 8,192 | 3 | 7.6 to 8.3 | 4,519.4 ms | 2,823.4 ms | 798.9 -> 564.5 | 840.2 -> 593.3 | 1,275.0 -> 475.9 | 667.6 -> 555.9 |

(The first row is before the integer first round of item 2, which brought the bottom session
at 1,024 from 246 to 203 ms in a later pair of runs.) In mode A at 8,192 signatures (CLMUL
build, 3 threads, `agg bench --warmup 1 --reps 3`, load average 4.6 to 5.7, revisions
`be398d8` and `9a53721`), event 7 went from 3,822.5 to 2,270.0 ms. The remaining cost is
close to the arithmetic of the protocol: the tree build (about 0.5 s at 8,192 on 3 threads),
the bottom session (one pass per round over `2^(D-1)` integer pairs) and the layer sessions
`D - 2, D - 3` (two trees each). Levers not taken, with their estimates: session `D - 2`
on the integer lanes `e_1 .. e_4` (a 5 x 5 bilinear form; the first round gets cheaper but
the first fold dearer, net about 15% of that session plus 1 GiB of stored layers at 8,192,
estimate); the small-value rounds 1 and 2 of the bottom session from the integers (about
20% of that session, estimate); multi-way SIMD for the folds.

## 5. The module (`crates/agg-piop/src/lookup.rs`)

The construction, driven by the PIOP transcript:

- `Table` (`T_w`, index halves, `index_evaluation`), `StackShape` (columns, variables,
  table, storage `shift`; `slots`, `depth`, `leaves`, `padding = -shift`, `column_weights`,
  `column_claim`, which removes the public padding slots from a stacked claim),
  `LookupStack::new(&[&[i32]], table)` (zero shift, a copy; unit tests),
  `LookupStack::with_shift` and `LookupStack::borrowed(shape, &stacked)` (the integrated
  stacks, no copy).
- `multiplicities(stacks, tables)`: parallel histogram per table of `stored + shift`,
  `OutOfTable` on an honest out-of-range cell; `multiplicity_digits` /
  `compose_multiplicity`: the seven balanced base-16 digits of `G_m` (capacity
  `7 (16^7 - 1)/15 = 125,269,879`). No count can exceed it for ML-DSA-44 and ML-DSA-65,
  whose `L` at `2^15` under H-FIPS (79,970,304 and 105,136,128; 105,005,056 for 65 under
  H-exact) is below the capacity. ML-DSA-87 has `L = 147,079,168` at `2^15` under H-FIPS
  (146,948,096 under H-exact), so the bound is per value: the largest honest count is that
  of the entry 0, which receives at most the padded cells (fewer than half of the records,
  since `N* = next_pow2(N)`), the public stack slots and the live cells of value zero; the
  worst case, `N = 2^14 + 1`, gives about 88.2 million (estimate), 1.42x below the capacity.
  The test `manifest::tests::multiplicity_digits_hold_the_largest_honest_count` checks this
  bound for every set and relation at `N = 2^14 + 1`. It is a typical-case bound, not a worst
  case: it counts the public padding slots, every bridge-domain and projection cell and the
  main-domain cells of the padding records, and leaves out the live main-domain cells,
  assuming that honest signatures spread the low 16-bit digits of `z` and `u` over the `2^16`
  entries. One degenerate valid batch is covered explicitly: for a public key with `t1 = 0`,
  FIPS 204 verification accepts `z = 0`, `h = 0`, so every live record can equal the padding
  witness. The zero coefficient lands on different entries in the two looked-up main families
  (for ML-DSA-87: `z_lo` on 65,415, `L_lo` on 65,535, H-exact's `u_lo` on 65,025; derived from
  `agg_witness::digits`), so the fullest entry holds at most one family (`8 * 2^23 = 67,108,864`
  cells) plus every non-main and padding cell (about 21.3 million), at most about 88.4 million
  (estimate), 1.4x below the capacity. No bound is claimed for arbitrary adversarially chosen
  valid batches. `multiplicity_digits` returns `LookupError::Shape` for a count above
  `MULTIPLICITY_CAPACITY` (no panic), and `BridgeGroupTables::recount` turns that into an
  honest abort of the prover.
- `LookupShape` (public layout, `challenges` 1 to 4, `validate`, `max_depth`,
  `tree_count`, `session_counts(s)` for the registry and the wire parser, `lookups`,
  `identity_error_log2`, `proof_elements`); the integrated shape is
  `CommitmentManifest::lookup_shape` (five stacks, `T_16`, two challenges, shift `2^15`).
- `LookupTranscript` (absorb, challenges at a `LookupSite`): implemented by the PIOP
  transcript (sites `agg-mldsa/v1/lookup/<site>`, `docs/deviations.md` L5); the unit tests
  and timings use a SHAKE256 test transcript bound to the layout.
- `FractionTree`: layers `0 .. depth - 1` stored as `K` pairs, built in parallel from the
  leaves (the bottom layer straight from the `i32` cells: `1` `K`-multiplication per node;
  upper layers `3`); layer `s` is freed after session `s`. For a stack on the integer path
  (Section 4.6, `SymLanes`) the layers stop at `depth - 2`, built for every challenge in one
  pass (`stack_upper_bottoms`).
- `prove_in(stacks, tables, multiplicities, challenges, transcript, check)` and
  `verify_in(shape, proof, transcript) -> LookupClaims`, Section 1 exactly; the verifier
  through `kernel::{Group, GroupInstance, Lane::Source(Source::Field | Source::Ints),
  Term::{Product, Affine}, evaluate_compressed, expected_terminal}` only, the prover on the dedicated groups of Section 4.6
  (`kernel::{LayerGroup, IntPairGroup}`), whose split of the instances into groups differs
  from the verifier's plan (`session_plan`) but whose messages, sums over the groups, do not.
  `check` runs the prover's root-identity and session self-checks (tests pass `false` to
  hand a dishonest witness to the verifier). The claims go to stage 3 (stacks) and to the
  multiplicity-digit claims (table).
- Unit tests (`cargo test -p agg-piop --release --lib lookup`, 8 tests): tree
  roots, digits, honest layouts at one and two challenges, two tables and a cross-table
  cheat, wrong multiplicities, tampered messages and claims, a swapped stack leaf pair
  caught by the claim, and `dedicated_paths_match_the_generic_kernel` (Section 4.6). The
  integrated tests are in `tests/all/stages.rs` (Section 6).
- Ignored timing tests: `timing_at_1024_and_8192_signatures` (Section 4; sizes from
  `AGG_LOOKUP_SIGS`, threads from `RAYON_NUM_THREADS` (default 7), challenge counts from
  `AGG_LOOKUP_CHALLENGES` (default `1,2`), the dedicated path only; warmup plus five runs)
  and `grinding_rate`.

## 6. Integration (as implemented)

1. *PCS* (`agg-pcs-config`, `docs/pcs-config.md`). Preset `AggFp64SignedWidth16` with one
   forced balanced base-`2^16` inner digit (deviation L3: the planner's estimate is the same
   for one, two or three digits). Six groups in wire order (seven since K24, which adds the
   width-16 projection group `G_k` after `G_3` and gives `G_3` the layout of `G_2`) `G_main16` (width 16), `G_main4`
   (width 4), `G_1`, `G_2` (one width-16 group for the parities and the carries), `G_3`
   (width 16), `G_m` (width 4, Akita's final group; deviation L4). Two catalogs: the width-16
   one with the scalar rows of its four groups, the width-4 one with the scalar rows of
   `G_main4` and the 13 grouped rows; the planner accepted every mixed-family grouped row
   (Akita freezes the bound per group).
2. *Tables* (`source_tables`, in `agg-piop`).
   `MainTables`: `G_main16` as one stacked `i32` lane, `G_main4` as `i8` columns, encoded
   from the derived `z, u, v, w_1'` (`RecordValues`). `BridgeGroupTables`: the stacked
   balanced carries of `agg_bridge::parity::compute`, the parities, `Y`, and `m` with its
   digits over every looked-up cell of both rounds (`recount`). `Cells` reads either width.
3. *Transcript and wire.* Event 7 is the lookup (`X_1, X_2`, layer 1, `mu_1`, sessions
   `2 .. D`, the seven digit claims); `tau` follows. The registry and the wire parser
   derive every count from `N` (`session_counts`). Wire version 1 carries the six
   commitments (two at event 1, four at event 4), the `LookupProof` and the 7 digit
   claims; version 0 is refused.
4. *Stages.* Stage 1: 26 instances. Stage 2: `def_y1` (6), `leaf4[Z_hi]`, `leaf6[U_hi]`,
   `leaf8[K_2]`, `leaf8[K_3]`; 6 `K^(1)` claims before it, the projection claim `c` after it. Stage 3:
   `ring` with (4.1) of Section 1, 4 links, 37 reductions, the 5 stack reductions (one gamma
   per stack, one part per column with the scale `eq(pi_top, c)`, deviation L6) and the two
   sides of the projection identity; 49 instances, 35 terminal claims. The prover and verifier share `prover::prove_piop` / `verifier::verify_piop`.
5. *Ledger*: spec Section 12.1 (normative; Section 2 above).
6. *Kernel*: unchanged API; the 16-bit columns are `Source::Ints` lanes (uncoded, generic
   path). A 16-bit coded lane (two `2^16`-entry tables per fold) is the next prover-side
   step if the stage-3 groups on `Ints` lanes dominate (Section 4.5).

Integrated tests (`cargo test -p agg-piop --release`): `tests/all/stages.rs` (honest PIOP at
256 and 10 signatures with every claim against the table extensions; the lookup rejects
16-bit cells out of `T_16` with `z` unchanged (`Z_lo +- 2^16`, `Z_hi -+ 1`),
carries `2^16` and negative at every level, while `V_hi = 2^16` (no longer looked up) is
rejected by the ring identity at stage 3, a wrong multiplicity (moved and added), a
tampered layer-1 value, GKR round message, layer terminal, table leaf pair, swapped
multiplicity-digit claims and a truncated session; swapped stage-3 column claims rejected
at stage 3; a value valid only for another range (`Z_hi = 2`, valid for `U_hi`; `U_hi = 4,
-5`; a carry stored unshifted, valid in `T_16` but meaning `kappa + 2^15`; a padding slot
holding a table entry) rejected at the leaf, the bridge identity or the stack reduction;
a low part valid only for the wider range `[0, 2^17)` (`U_lo + 63,079` with `U_hi - 1`: `u`
unchanged) rejected by the lookup, and `V_lo + 2^16` with `V_hi - 1` (`v` unchanged)
accepted; every old witness tampering at its stage), `tests/all/projection.rs` (deviation
K24: the unbounded-quotient attack with `Delta = 0` in `F_p` rejected by the lookup of
`K_1`, `k` unbound to `v` at stage 3, `K_2 = 4` and `K_3 = 4` at stage 2, a changed claim `c`, the honest
abort on `|k| >= 2^37`), `tests/all/mode_b.rs` (round trips at 64, 256, 1,024; end-to-end
tampering including a wrong multiplicity; a stored 16-bit value outside the width-16
interval is refused by Akita's commit, so the prover fails and no aggregate exists: the
verifier's rejection of a looked-up value outside `T_16` is tested only at the PIOP level,
in `tests/all/stages.rs`, without commitments and opening; an end-to-end test would need a
commit path that skips Akita's admission bound), `tests/single_thread.rs` (a whole prover
and verifier run with a one-thread rayon pool, called from a plain thread and from a pool
worker, under a timeout: the regression test of the round-1 deadlock, deviation P12),
`tests/all/prove_verify.rs`, `tests/all/bridge.rs`, `tests/all/source_tables.rs`,
`tests/all/virtual_checker.rs` (the L16 tables against the integer relation), `tests/all/commit_open.rs`;
`agg-cli` `tests/all/mode_a.rs` (mode A round trips and tampering).
