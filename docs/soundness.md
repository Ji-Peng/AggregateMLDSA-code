# Soundness: site census and the proof-of-work schedule (normative)

| Field | Value |
| --- | --- |
| Status | normative design for the 128-bit schedule (plan `ref/notes/plan-128bit-speed.md` Part 1, step 1); implemented in step 2 (branch `pow-128`): PIOP wire version 2 (`agg_piop::pow`, `agg_piop::transcript`), tied-part proof version 4 (`agg_keccak::tied`, deviation K20), the list-aware Flock outer sites (`vendor/flock-core/PATCH.md` Section 5, deviation K19), the Akita plan pinned (`agg-pcs-config/tests/akita_grinding.rs`) |
| Applies to | mode A and mode B, ML-DSA-44, relations H-u (the default since 2026-10-25, `docs/h-exact.md` step A; it replaced relaxed H) and H-exact (relation byte 1, step B; Section 10.1), lookup layout L16; PIOP wire version 3 (version 2 of this document with the H-u stage-2 degree 4) and tied-part proof version 4 (introduced by this document) |
| Base | code `be398d8`; Aerie `49b6c21`; Akita `e023b6a46143f7b23199ccecb2684192baf7b640`; Flock `83f963e` with the vendored `flock-core` (K12, K18) |
| Script | `docs/scripts/pow_plan.py` (run with `python3 -I`); its output summary is Section 10; `docs/scripts/fold_support.py` (Section 4.6) |
| Replaces | the per-instance union bound of `docs/spec.md` 12.1 (Section 5), deviation P3 once implemented |
| Last reviewed | 2026-10-26: Section 12.8 (aggregate unforgeability under SUF-CMA of the relaxed scheme, spec 12.3 steps 1' and 2'; branch `suf-proof`); 2026-10-25, first version; implementation notes added (Sections 4.1, 4.3, 11); review fixes: at most two parts per segment and the `sd` union (3.5), the tie union over list pairs (3.6), the site-label argument (6.1), the completeness count (6.5), hypotheses H1/H2 (7); 2026-10-25 (branch `pow-close`): H1 discharged (4.6), the composition proof (Section 12, H2 under assumptions (F), (K)), canonical Flock nonce width and the tie-search error (6.3, 6.5); review of `pow-close`: part-site bad sets from the address alone (12.4), mode-B Corollary 12.5, spec 12.3 in two steps |

The uppercase terms MUST, MUST NOT, SHOULD and MAY are normative as in BCP 14. Every
number below that is not marked "estimate" is computed exactly by the script or read from
the pinned code; estimates are marked.

## 0. Results in one page

1. **PIOP sites.** At `N* = 2^15` the transcript draws 484 field challenges in `K`
   (`|K| = p^2`, `p = 2^64 - 59`): the lookup vector `X = (X_1, X_2)`, 27 merges `mu_s`,
   26 GKR batching vectors, 351 GKR rounds of degree 3, `tau`, `gamma^(1..3)` and
   `3 x 25` stage rounds of degrees 4, 4, 3 (relation H-u; relaxed H had 4, 5, 3 and the
   same difficulties, since `bit_length(4) = bit_length(5)`). With the assigned bounds of Section 3 every
   site except `X` needs work: 483 nonces of 8 to 13 bits, 4,444 bits = **556 bytes** in
   one packed stream, about 2,718 predicate evaluations of expected honest work
   (about 2 ms serial at the measured 724 ns per Blake2b predicate, estimate).
   `X` needs none: its joint bad set has mass `((L + M - 1)/|K|)^2 = 2^-201.9`.
2. **Tie site.** The Flock tie challenges `rho` of a tied part are drawn while the two
   Ligerito commitments are bound only to their Johnson lists, so the tie bound carries
   the list sizes: `2 L_s L_u (d + span_log + 5) + 2 L_u (log2 blocks + span_log + 9) +
   L_u (5 + span_log) + L_s (nu_s + 3)` over `2^128`. At four full segments under `Slim`
   (`L_s = L_u = 50`) this is `102,750 / 2^128`: **difficulty 17 bits, nonce 24 bits**
   (not the 7 bits of `docs/bridge.md` 5c, which omitted the lists). About `2^17`
   BLAKE3 evaluations (about 0.5 ms on 7 threads, estimate interpolated from the measured
   16- and 20-bit grinds of `docs/keccak.md` 12.3) and 4 proof bytes.
3. **Akita.** The pinned Akita `e023b6a` already contains PR #460 (commit `4505404b`, an
   ancestor) and its follow-ups (exact-cardinality pricing #104, #130, #209). Transcript
   proof-of-work is unconditional in `batched_prove`/`batched_verify`; our six-group
   catalogs already run it: 459 protected sites at `2^15`, targets up to 12 bits
   (ring-switch `alpha`, loss 4,095), plan digest bound in Akita's instance descriptor.
   Nothing needs to be turned on; `status.md`'s former "no grinding plan" was wrong for
   the opening and right only for the PIOP (status.md corrected on 2026-10-25).
4. **Flock.** Flock's strict `Fast`/`Slim` level is not a query-only claim: every outer
   challenge (zerocheck, lincheck, ring switch, transport batch, merged sum-check) has a
   per-challenge nonce and every Ligerito component is validated strictly above 128 bits.
   **But** the outer challenges are drawn before the level-0 two-point binding, and the
   Flock paper's own accounting (Appendix A "Security", Remark 11) multiplies each of them
   by the list size `L_max`; the vendored schedule charges `L_max` only at the transport
   claim batch. Under that accounting the outer sites reach 122.4 to 123.9 bits (`Fast`,
   `Slim`) and 121.9 to 122.9 bits (`Slim8`), and `Slim8`'s transport claim batch reaches
   127.86 bits. Minimal fix: list-aware nonce widths at those sites in the vendored core
   (Section 4.3). This changes the Flock verifier, so the K12 patch would no longer be
   prover-only.
5. **Binding gap in the tied part.** The two instance challengers of a tied part do not
   absorb the data the tie points are derived from (the union's cap and the final-vector
   digest for the sponge instance; the final-vector digest for the union), although
   Flock's packed-direct claims absorb values only and assume transcript-derived points.
   Tied-part proof version 4 MUST bind a tie digest into both challengers (Section 4.5).
6. **Tighter accounting.** Round-by-round (first false claim, Aerie Lemma C.1) gives the
   union-bound ledger `1,409/|K| = 2^-117.5` at `2^15` instead of spec 12.1's
   `3,509/|K| = 2^-116.2` (relaxed H: `1,434` and `3,534`): reductions, zero-checks and linear identities ride on the stage
   sum-checks and on one `tau` site (Section 5). With the schedule every address weighs
   at most `2^-128`, and the theorem of Section 7 follows Aerie Eq. (18).
7. **Composition (Section 12) and fold supports (4.6).** The theorem's two hypotheses are
   settled: every fold-challenge family of both catalogs has at least `2^128.0881` accepted
   challenges (H1, test `akita_fold_support.rs`), and the composition with the Flock
   commitments is proved (H2) with checkpoints at `sd` and `X`, no second extractor, under
   two stated assumptions: Flock's documented round-by-round analysis (F) and the
   correctness of the Keccak and bridge circuits (K).
8. **Unforgeability (Section 12.8).** Spec 12.3's open obligation (signing queries asked
   only in rewound runs) is closed by lazy forwarding to a SUF-CMA challenger of the
   relaxed scheme (Lemmas 12.6, 12.7) and a three-case SUF-CMA lemma (12.8): SelfTargetMSIS
   at `zeta' = 350,209`, a SHAKE256 collision, or Module-SIS on `[A | I]` at `l_inf`
   380,930. Theorem 12.9: `Adv_agg <= Adv_SUF-CMA + eps_ext + E[R] 2^-H_inf` (`+ Adv_PRF +
   Adv_coll(mu)` for the deterministic signer), no `q_H` or `e(Q_S + 1)` factor. The
   lattice basis moves from core-SVP `2^123.5` to `2^121.8` (block size 417, measured).

## 1. Model and method

**Oracles.** Classical random-oracle model as in Aerie Section 5.6 and Akita Section 10.1:
the PIOP Spongefish `Blake2b512` duplex (`agg_piop::transcript`), the tie challenger and
the two Flock instance challengers (Flock's `FsChallenger`, BLAKE3), Akita's native stream
(seeded by the 32-byte PIOP session digest), the setup oracle, and the hashed-entry
oracles (instance digest, public-record digest, PCS session digest, Flock seeds and span
seed, tie digests). Every challenge input MUST injectively encode the complete preceding
transcript, directly or through a hashed entry (Aerie C.1); Section 4.5 records the one
place where the code does not.

**Sites and bad sets.** A *site* is one challenge query: a scalar or a vector drawn from
one oracle stream (Aerie C.1: "we treat each vector as one field challenge"). For a site
`t`, fix the objects the extractor will return (the 38 Akita polynomials `f`, 39 under H-exact, the Flock
word vectors `z = (z_s, z_u)` up to their Johnson lists) and the transcript prefix; the
*bad set* `Bad_t` is the set of answers after which a doomed transcript (one with a failed
check or a false pending claim) may stop being doomed. `nu_t` is the numerator of its
mass: `Pr[Bad_t] <= nu_t / |F_t|` with `|F_t| = |K| = p^2` (PIOP) or `2^128` (GF(2^128)).

**Assigned bound and difficulty.** Each site gets an assigned bound `e_t >= nu_t / |F_t|`
(Aerie C.4: "the analysis requires only `e_t >= nu_t/|K|`") and the difficulty

```text
g_t = min{ g >= 0 : 2^128 e_t <= 2^g }                       (Aerie Eq. (17))
```

computed with the exact field size: in `K`, for an assigned bound `n/|K|` with
`n < 2^57`, `g = bit_length(n)` (Aerie `grind_bits`: even `1/|K|` needs one bit because
`p^2 < 2^128`); in GF(2^128), `g = ceil(log2 n)` for `n/2^128` (so `1/2^128` needs none). A site with `g_t = 0` has no
nonce; otherwise the nonce has `g_t + 7` bits.

**Per-address weight.** By Akita Proposition 10.3 (adaptive search at a protected
challenge), an address of site `t` contributes at most `2^-g_t nu_t / |F_t| <=
2^-g_t e_t <= 2^-128` to the weighted sum of Akita Theorem 10.45, and the expected total
weight read by a prover making `Q` queries and by the verifier is
`rho (Q + R_f + R_coord)` with `rho = max_a weight(a) <= 2^-128` (Aerie Eq. (18), (19)).
No factor counting all sites enters the coefficient of `Q`.

**Assigned-bound policy (normative).** `e_t` follows Aerie Section C.4 so that the
schedule does not rest on the sharper arguments of Section 5: a uniform point of `w`
coordinates `w/|F|`; a sum-check round `deg/|F|`; a vector of `l` independent batching
coefficients `l/|F|` (its true numerator is 1); a line merge `1/|F|`. The tie site is
assigned the union of its identities with list factors (Section 3.6), its true numerator
being the maximum.

## 2. Transcript map

Events of spec Section 10, with the sites they contain (mode A; mode B has no Flock
sub-proofs and therefore no tie site):

| Event | Oracle | Sites (in order) |
| --- | --- | --- |
| 0 | PIOP | none; binds the instance record, which in wire version 2 ends with the plan digest (Section 6.4) |
| 1, 2 | PIOP | none (absorbs: round-1 Akita commitments; Flock sponge caps) |
| 3 | PIOP | seed `sd` (32 bytes, no field sampling) |
| 4 | PIOP | none (round-2 Akita commitments) |
| 5 | PIOP | seeds `fs_sigma` (32 bytes each); the span seed is a hash of them |
| 6 | PIOP absorbs `Y_sigma`, then the Flock proof bytes; inside each tied part (own oracles, after the bridge union's cap): the **tie site** (tie challenger), then the sponge instance's Flock sites (zerocheck, lincheck, opening), then the union's |
| 7 | PIOP | `X` (vector of 2); `mu_1`; per session `s = 2..D`: `gamma_s` (vector), `s - 1` rounds, `mu_s` |
| 8, 9 | PIOP | `tau` (vector of `d_main`) |
| 10 to 12 | PIOP | `gamma^(1)` (26), 25 stage-1 rounds; `gamma^(2)` (8), stage-2 rounds; `gamma^(3)` (44), stage-3 rounds (each stage `d_main` rounds) |
| 13 | PIOP export, then Akita | PCS session digest (no site); Akita's own sites under its plan (Section 3.7) |

The code that draws them: `agg_piop::transcript::Transcript::{challenge_extensions,
challenge_bytes, pcs_session}` (registry `Registry::new`), called from
`lookup::{prove_in, verify_in}` (`X`, `Gamma(s)`, `RoundChallenge(s, r)`, `Mu(s)`),
`prover::prove_piop` / `verifier::verify_piop` (`tau`), `stage::{prove_stage_with,
verify}` (`gamma^(s)`, round challenges), `prover::prove_with_tables` /
`verifier::verify` (`sd`, `fs`); `agg_keccak::tied::TieChallenges::derive` (tie);
`agg_keccak::flock_union::UnionGeometry::{prove, verify}` (Flock); Akita
`batched_prove` / `batched_verify` via `agg_pcs_config::commit` (Akita).

## 3. Site census (normative)

### 3.1 PIOP sites

Geometry at `N* = 2^d_sig` (spec Section 1, `agg_pcs_config::catalogs::Group`):
`d_main = 10 + d_sig`, `d_1 = 13 + max(0, d_sig - 9)`, `d_2 = 13 + max(0, d_sig - 13)`,
`d_3 = 12 + max(0, d_sig - 13)`; per lookup challenge the trees have depths
`d_main + 2, d_1 + 3, d_2 + 3, d_3` (stacks) and 16 (`T_16`), two challenges, so ten
trees; `D = max(d_main + 2, 16)`; `L = 4 2^d_main + 8 2^d_1 + 8 2^d_2 + 2^d_3`,
`M = 2^16`; `active_s` is the number of trees of depth `>= s`.

| Class (id) | Count | `nu_t` | Assigned `e_t` | `g_t` | Nonce bits | At `2^15` |
| --- | --- | --- | --- | --- | --- | --- |
| `lookup.x` (1) | 1 | joint (3.2) | `((L + M - 1)/|K|)^2` | 0 | 0 | `2^-201.9` |
| `lookup.mu` (2) | `D` (`mu_1 .. mu_D`) | 1 | `1/|K|` | 1 | 8 | 27 sites |
| `lookup.gamma` (3) | `D - 1` | 1 | `2 active_s / |K|` | `bit_length(2 active_s)` | `g + 7` | 20, 16, 12, 8, 4 coefficients: `g` = 5, 5, 4, 4, 3 (13, 2, 2, 4, 5 sites) |
| `lookup.round` (4) | `D (D - 1)/2` | 3 | `3/|K|` | 2 | 9 | 351 sites |
| `tau` (5) | 1 | `d_main` | `d_main/|K|` | `bit_length(d_main)` | `g + 7` | 25: `g = 5` |
| `stage.gamma` (6) | 3 | 1 | `26, 8, 44 / |K|` (H-exact: `26, 7, 45`) | 5, 4, 6 (H-exact: 5, 3, 6) | 12, 11, 13 (H-exact: 12, 10, 13) | |
| `stage.round` (7) | `3 d_main` | 4, 4, 3 | `deg/|K|` | 3, 3, 2 | 10, 10, 9 | 75 sites |

Totals per `N*` are in Section 10 (from 194 nonces / 227 bytes at `N* = 8` to 483 /
556 bytes at `2^15`). The largest PIOP difficulty is 6 bits (`gamma^(3)`), far below
Akita's 25-bit transition cap.

### 3.2 The lookup challenge `X = (X_1, X_2)`: one site with the joint bound

`X_1, X_2` are drawn by one vector challenge (`LookupSite::X`, two elements from one
stream). The pending claim before `X` is "every looked-up cell of `f` lies in `T_16`"
(cells and the multiplicities `m` are fixed by the commitments of events 1 and 4). If it
is false, the numerator `N(X)` of `R(X) = sum_i 1/(X - f_i) - sum_t m(t)/(X - t)` over
`prod_{a in S}(X - a)` is a nonzero polynomial of degree at most `L + M - 1` (spec 12.1,
Lemma; LogUp Lemma 5 with `p > L`). After the answer, the transcript stays doomed unless
both true root identities pass, i.e. unless `N(X_1) = N(X_2) = 0` with `X_j` outside `S`
(if some `X_j` lies in `S`, the true root denominator is zero and the verifier's nonzero
check forces a false layer-1 value). Hence `Bad_X = {N(X_1) = 0 and N(X_2) = 0}`, of mass
at most `((L + M - 1)/|K|)^2` because the two coordinates are independent and uniform
(exact per-limb rejection sampling). This is one site: its bad set is fixed before the
query and its mass is the joint one; Proposition 10.3 then bounds `N` queried candidates
by `(N + 1) ((L + M - 1)/|K|)^2`. `g_X = 0` at every `N*` (`2^-220.3` at `N* = 8` to
`2^-201.9` at `2^15`). Treating `X_1` and `X_2` as two sites would be wrong (each alone
has mass `2^-101`); the code MUST keep them in one challenge event.

### 3.3 The GKR sessions

Session `s` reduces the layer-`(s - 1)` claims of every active tree to layer `s`
(`docs/lookup.md` Section 1, `lookup::verify_in`):

* `gamma_s` (`2 active_s` independent coefficients): if some claim is false,
  `sum_i gamma_i (c_i - c_i^f)` is a nonzero linear form; `nu = 1`.
* round `r` of session `s`: the batched summand is `eq` (multilinear) times products of
  two lanes (`P_0 Q_1 + P_1 Q_0`, `Q_0 Q_1`, and the bottom groups' quadratic forms in
  `v_0, v_1` or `m_0, m_1` and the index lanes), so the round polynomial has degree 3;
  `nu = 3` (`LOOKUP_DEGREE = 3` coefficients per message).
* the terminal check is exact: `expected_terminal` recomputes the batched summand from the
  prover's values; a false running claim forces a false terminal or leaf value.
* `mu_s`: every continuing claim becomes `line(P_0, P_1, mu)`; for a false pair the
  difference with the truth is a nonzero polynomial of degree 1 in `mu`; `nu = 1`
  (the same for the leaf claims of ending stacks and the table). `mu_1` likewise after the
  layer-1 values (the root check itself is exact).

### 3.4 `tau` and the three stages

* `tau` (vector of `d_main`): before it, the pending identities tested at `tau` are the 19
  Booleanity and range zero-checks and the level-2 and level-3 linear identities (spec 7.3).
  If the first false one has defect `phi != 0` on its Boolean domain, then
  `sum_x eq(tau_t, x) phi(x)` is a nonzero multilinear polynomial in at most `d_main`
  coordinates of `tau`; `nu = d_main` (Aerie Lemma C.1, "the point tau"). The pre-stage
  claims sent after `tau` are pending evaluation claims, reduced in stage 3.
* `gamma^(s)` (26, 8, 44 coefficients; 26, 7, 45 under H-exact): `nu = 1` as in 3.3.
* stage rounds: `nu` = the round polynomial's degree, 4, 4, 3 (`STAGE_DEGREES`; the `eq`
  factor is included: `range_w1` has degree 4, `leaf6` of the H-u digit `U_hi` degree 4
  (`leaf4` degree 3), `link` degree 3).
* identities tested at a point drawn by the preceding stage (the leaves at `r^(1)`, the
  ring identity and the links at `r^(2)`): while the earlier stage draws the point, the
  identity is the pending claim "the partially substituted `sum_x eq(r, x) phi(x)` is the
  zero polynomial", which a round's answer can falsify for at most one value of its
  coordinate; that value lies inside the round's own `nu >= 1` (Aerie Lemma C.1,
  "Identities tested at a point of the preceding stage"). No extra site.
* evaluation reductions and the four lookup-stack reductions are stage-3 instances: a
  false input claim makes the instance false, which `gamma^(3)` and the stage-3 rounds
  carry (Section 5). No extra site.

### 3.5 Seed sites and exports (no work)

* `sd` (event 3): bad set "`M_3 M_2 M_1 delta = 0` for the fixed `delta = b xor b'`"
  (spec 9.4), mass `theta = 2^-144.64` per segment check. Because each part's `b'_s` is
  fixed only up to its sponge commitment's Johnson list (Section 3.6), and `Y_sigma` is
  checked once per segment against the XOR of the segment's `P_sigma` part shares, the bad
  set of segment `sigma` is a union over the product of its `P_sigma` lists, and the
  segments' bad sets add: `sum_sigma L_s^(P_sigma) theta`. A span is one part over all
  segments (one list, one `theta` per segment row, still a union over one list). Mode A
  MUST admit at most two tied parts per segment (`d_part >= d_seg - 1`, code
  `mode_a::MAX_PARTS_LOG2`; verifiers refuse a smaller `d_part` while deriving the plan of
  event 0). The resulting values, with `S <= 4` segments at `2^15`:

  | Layout | Bound | `Fast` | `Slim` | `Slim8` |
  | --- | --- | --- | --- | --- |
  | span on, or one segment with one part | `L_s theta` | `2^-139.51` | `2^-139.00` | `2^-138.51` |
  | span off, `S = 4` segments, one part each | `S L_s theta` | `2^-137.51` | `2^-137.00` | `2^-136.51` |
  | one segment, two parts | `L_s^2 theta` | `2^-134.38` | `2^-133.35` | `2^-132.38` |
  | `S = 4` segments, two parts each | `S L_s^2 theta` | `2^-132.38` | `2^-131.35` | `2^-130.38` |

  All are below `2^-128`: no nonce (Aerie: "the projection seed requires no work"). The
  composition proof (12.4) works over a fixed bridge expansion oracle, so it uses the seed
  density `theta' = 2 theta` of Lemma 12.4, and it charges only the first false segment:
  `max_sigma L_s^(P_sigma) theta' <= 2^-131.38` (`Slim8`, two parts), still no nonce. Without
  the two-part cap the bound would be `L_s^P theta` for `P` parts in one segment, which
  exceeds `2^-128` from `P = 3` under `Slim` (`2^-127.7`) and is vacuous at the
  `2^(d_seg - 3)` parts a segment could otherwise hold (review of `pow-128`, finding 1).
* `fs_sigma` (event 5), the span seed, the PCS session digest (event 13): seeds of
  sub-protocol oracles. They have no bad set of their own; the sites they seed are
  counted where they occur.

### 3.6 The tie site of a tied part

`TieChallenges::derive` draws all tie coordinates (`rho_w, rho_i, rho_t, rho_t1, rho_r,
rho_c, rho_beta, rho_y, rho_o, rho_sigma`) from one tie-challenger stream after both caps
and the final-vector digest: one site in GF(2^128). The pending identities
(`docs/bridge.md` 5b, 5c), with `blocks = 2^max(0, d - 9)`, `span_log = ceil(log2 S)`,
`nu_s = capacity_log(n, 19)`:

| Identity | Objects | Numerator per object pair |
| --- | --- | --- |
| tie 1, two lane groups | sponge inputs (`z_s`) vs level-1 inputs (`z_u`) | `d + span_log + 5` |
| tie 2, two lane groups | level-1 parities vs their copies (both in `z_u`) | `log2 blocks + span_log + 9` |
| outputs | output columns (`z_u`) vs the public table `Yhat` | `5 + span_log` |
| public words | sponge public region (`z_s`) vs the verifier's table | `nu_s + 3` |

**List factor.** At the time of `rho`, each Ligerito commitment (sponge cap of event 2,
union cap) is bound only to its Johnson list: the level-0 two-point binding of each
instance (its ring-switched opening point and one explicit OOD sample, `vendor/flock-core
/src/pcs/ligerito.rs`, `SoundnessRegime::JohnsonOod`) happens inside its opening, after
`rho`. A bad `rho` is one at which some pair of list elements with a nonzero defect
evaluates to zero, so the bad set is the union over the lists: `L_s L_u` pairs for tie 1,
`L_u` elements for tie 2 and the outputs, `L_s` for the public words. With
`L = floor(1/(2 eta sqrt(rho_0)))` (Flock Eq. (10), `eta = 0.02`, level-0 rate
`rho_0 = 2^-r`): `L = 35, 50, 70` for `Fast, Slim, Slim8` (`r = 1, 2, 3`).

* `nu_tie` (round-by-round, first false identity, unioned over the list pairs) `=
  L_s L_u (d + span_log + 5) + L_u max(log2 blocks + span_log + 9, 5 + span_log) +
  L_s (nu_s + 3)`. A single maximum over the four identities is not a bad-set bound:
  different list pairs `(f, g)` can have different first false identities; tie 2 and the
  outputs depend only on the union element `g`, the public words only on the sponge element
  `f`, so the union over pairs adds one term per object type.
* Assigned (normative) `e_tie = [2 L_s L_u (d + span_log + 5) + 2 L_u (log2 blocks +
  span_log + 9) + L_u (5 + span_log) + L_s (nu_s + 3)] / 2^128`.
* `g_tie = ceil(log2(2^128 e_tie))`; nonce `g_tie + 7` bits.

At four full segments (`d = 13`, `S = 4`, `nu_s = 15`): `Fast` 50,925 (`g = 16`), `Slim`
102,750 (`g = 17`), `Slim8` 199,850 (`g = 18`); a single full segment 45,745 / 92,350 /
179,690 (`g = 16, 17, 18`). Without lists the sum is `bridge.md`'s 95 (`g = 7`). Every
`(d, S)` is tabulated in Section 10. The tie difficulty is a function of the part's public
geometry `(profile, d, S, n)`, all known to the verifier before it replays event 0.

### 3.7 Akita opening sites (inherited, already protected)

Akita's plan (`akita_params::derive_transcript_grinding_plan_from_public_shape`) assigns
every field site of the opening a loss factor and a target by Akita Eq. (146) priced over
the exact cardinality `p^2` (`ChallengeFieldOrder::from_field`, PR #104): evaluation
batch (loss = claims, 38), extension-opening point and batch, every sum-check round
(stage 1, 2, physical L2, extension-opening reduction; loss = degree), ring-switch
`alpha` (loss `2d' - 1`, Akita Theorem 10.36), `tau0`, `tau1` points, interstage, L2
merge and virtual batches, compression and stage-2 batches. Sparse fold challenges carry
no nonce: each coordinate stream has weight `1/C_{j,g}`, and Akita refuses any fold
family below its 128-bit entropy floor (`SparseChallengeConfig::validate_for_ring_dim`,
`MIN_FOLD_CHALLENGE_ENTROPY_BITS`). Dumped for our catalogs (Section 4.1):

| `d_sig` | PoW sites | `sum g` | max `g` | expected work `sum 2^g` | nonce bits | max nonce bytes (LEB128, inline) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 to 5 | 401 | 984 | 9 | 3,888 | 3,863 | 819 |
| 6 to 8 | 401 | 985 | 10 | 4,400 | 3,864 | 819 |
| 9 | 406 | 980 | 10 | 5,120 | 3,894 | 829 |
| 10 | 388 | 967 | 12 | 7,692 | 3,755 | 793 |
| 11 | 430 | 1,053 | 12 | 7,868 | 4,135 | 877 |
| 12 | 452 | 1,121 | 12 | 8,304 | 4,369 | 924 |
| 13 | 455 | 1,129 | 12 | 8,832 | 4,398 | 930 |
| 14 | 457 | 1,132 | 12 | 8,076 | 4,415 | 934 |
| 15 | 459 | 1,137 | 12 | 9,872 | 4,434 | 938 |

At `2^15` (7 fold levels, 7 fold-response nonces of 12 bits, 12 sparse fold groups):
ring-switch `alpha` loss 4,095 / 2,047 / 255 (x4) with `g` = 12 / 11 / 8; `tau0` loss 17
to 25 (`g` = 5); evaluation batch loss 38 (`g` = 6); 119 stage-1 rounds of loss 5 and 34
of loss 3, 119 stage-2 rounds of loss 3, 52 physical-L2 rounds of loss 5, 89
extension-opening rounds of loss 2; the rest loss 1 to 7. These nonces are already in the
measured opening (62,974 B at 32K, `ref/notes/bench-cloud.md` Phase 7).

The 13 H-exact grouped rows (`mldsa-44-l16x-batch-<N*>-mult-final`, `G_main4` at 9
polynomials, 39 claims in the evaluation batch, `docs/h-exact.md` 2.3) have the same plan
summaries as the table above at `d_sig` 3 to 14; at `d_sig` 15: 459 sites, `sum g` 1,135,
max `g` 12, expected work 8,336, nonce bits 4,432. Their digests are pinned next to H-u's
(`agg-pcs-config/tests/akita_grinding.rs`, `akita_plans_of_the_h_exact_rows_are_pinned`).

### 3.8 Flock internal sites (inherited; audit in 4.2 to 4.4)

Per Flock instance (two per tied part): zerocheck (initial point, `bits_for(m)`; skip point,
7 bits; `m - 6` rounds, 2 bits each), lincheck (`alpha`, `beta` per pinned slot, 1 bit;
rounds, 2 bits; skip, `k` bits), opening transport (ring switch, 3 bits per ring-switched
claim; claim batch, 6 bits; merged rounds, 2 bits each), Ligerito (per level: fold
challenges in F256 with 0 bits, claim batch and consistency batch with list-unioned bits,
OOD samples, queries with a 16-bit query PoW). Nonces are `u64` in Flock's bincode and
already counted in the measured Flock bytes. This document adds no site inside Flock; the
fix of Section 4.3 only widens existing ones.

## 4. Audits

### 4.1 Akita: PR #460 is in the pin and always on

* `git merge-base --is-ancestor 4505404b e023b6a` holds: "feat(protocol)!: bind
  transcript grinding end to end (#460)" is in the pin, with "fix(grinding)!: price
  challenges by exact field cardinality (#104)", "Price EvaluationBatch grinding at its
  coordinate-wise tree charge (#130)" and "fix(grinding)!: price ring-switch alpha by the
  native inner ring (#209)".
* There is no switch. `akita_config::transcript_instance_descriptor` derives "the only
  accepted grinding plan" from the selected schedule and the opening layout and binds its
  digest (`TranscriptGrindingBinding`, 32 bytes) into the instance descriptor; both
  `akita_prover` (`prove/root.rs`) and `AkitaVerifier::batched_verify`
  (`fold/verify.rs`) call it and replay the plan (`ProverGrinding`, `VerifierGrinding`,
  `grinding.finish()`); nonces are canonical LEB128 atoms inline in the Akita stream,
  `GRINDING_NONCE_SLACK_BITS = 7`, `MAX_GRINDING_BITS = 25`. Zero-bit plans exist only in
  Akita's direct tests and benchmarks.
* Aerie `49b6c21` does nothing to enable it: it calls `batched_verify` like us and adds
  its own PIOP plan (`aerie_falcon::protocol::grinding::PiopGrindingPlan`), which is the
  part our code lacks (deviation P3).
* Our presets: `agg_pcs_config::commit` calls `AkitaCommitmentScheme::batched_prove` and
  `AkitaVerifier::batched_verify` with the width-4 catalog (`Cfg = AggFp64SignedWidth4`),
  so our openings already grind. Evidence: a scratch binary (not committed) calling
  `akita_config::derive_transcript_grinding_plan::<AggFp64SignedWidth4>` on every grouped
  row (`WorkPlan::for_records(2^d_sig).schedule_key()`) produced the table of Section 3.7;
  plan digests begin `22 57 08 cb` (`d_sig` 3) through `4f 5a 60 26` (`d_sig` 15) (hex).
* Consequence for step 2: no change in `agg-pcs-config`; the implementation SHOULD add a
  test that pins the per-row Akita plan digests (regression guard against a catalog or
  Akita change), and the docs MUST stop saying the opening has no grinding plan.
* Implemented: `crates/agg-pcs-config/tests/akita_grinding.rs` derives the plan of every
  grouped row for the opening layout of its six root groups
  (`OpeningClaimsLayout::from_root_groups`), checks the table of Section 3.7 (sites, `sum g`,
  max `g`, `sum 2^g`, nonce bits of all runs) and pins the 13 full digests (`225708cb...` at
  `d_sig` 3 to `4f5a6026...` at 15). `ref/notes/status.md` (top-level repository) was
  corrected on 2026-10-25.

### 4.2 Flock: what the documented 128-bit level covers

Read from `vendor/flock-core` (identical to `83f963e` except K12/K18) and Flock's
`docs/128-bit-grinding-audit.md`: under `Fast`, `Slim` and `Slim8` the policies
`zerocheck_grinding`, `lincheck_grinding`, `opening_grinding`
(`OpeningGrinding::per_challenge_128`) are enabled, and `flock_union::UnionGeometry::
{prove, verify}` uses exactly these policies, so every outer challenge on our code path
is preceded by a PoW nonce (Flock's legacy BLAKE3 composition: the predicate is BLAKE3 of
the 64-byte block `state_digest || LE64(nonce) || 0^24` with leading zero bits, MSB first;
then the nonce is absorbed and the challenge squeezed). Inside
Ligerito, `LigeritoSecurityConfig::validate` (analysis
`f256_split_johnson_two_point_ood_query128_c3_algebraic_row_union_over_bchks25_thm_4_6`)
requires strictly more than 128 bits for the query term (with 16-bit query PoW), the F256
MCA term, the two-point OOD binding, the claim batch and the consistency batch (both
list-unioned), and the F256 fold sum-check (list-unioned, above 246 bits). So the level
is **not** only the query phase. The bounds are Flock's: we do not re-derive the MCA and
query analysis.

### 4.3 Flock: the list factor at the outer sites (finding)

The level-0 list of a Ligerito commitment is bound only by two post-commit evaluations,
the ring-switched opening and one explicit OOD sample, both inside the opening. Every
outer challenge (zerocheck, lincheck, ring switch, transport claim batch, merged
sum-check rounds, multipoint) is drawn before that binding. The Flock paper charges each
of them the list size: Appendix A, "Security": "Zerocheck: the initial round's soundness
error is `L_max (m - k_s - 7)/|F|`, the univariate skip round's `L_max 2(2^k_s - 1)/|F|`,
and the remaining rounds `L_max 2/|F|`; Lincheck: ... likewise", and Remark 11 (dropping
the initial OOD sample costs a factor `L` in the outer protocol). The vendored
`per_challenge_128` numerators omit `L_max` at every outer site except the transport
claim batch, whose comment says exactly this ("sampled before Ligerito's OOD points bind
one candidate from the Johnson list ... six PoW bits make `L/2^(128+6) < 2^-128`").

Work-normalized bits under the paper's accounting, `128 + bits - log2(nu L_max)` (script,
Section 10): `Fast` (`log2 L_max = 5.14`) 122.9 to 123.9; `Slim` (5.64) 122.4 to 123.4;
`Slim8` (6.14) 121.9 to 122.9; transport claim batch 128.86 / 128.36 / **127.86**. The
smallest values are the zerocheck skip point (`nu = 127`, 7 bits), the lincheck skip, the
ring-switch point and the zerocheck initial point.

**Minimal fix (vendored core, prover and verifier).** Charge `L_max` of the instance's
level-0 rate at every outer site with Flock's strict rule:
`bits = floor(log2(nu L_max)) + 1`. Per profile (`Fast / Slim / Slim8`): zerocheck rounds
and lincheck rounds 7 / 7 / 8; lincheck `alpha`, `beta` and the transport claim batch
6 / 6 / 7; ring-switch point 8 / 9 / 9; merged rounds 7 / 7 / 8; zerocheck skip 13 / 13 /
14; lincheck skip `k + 6` / `k + 6` / `k + 7`; zerocheck initial point
`floor(log2(m L_max)) + 1` (11 / 11 / 12 at `m = 30` or 34). Cost: about 100 outer sites
per instance at `2^7` to `2^14` BLAKE3 evaluations, a few hundred thousand hashes per
tied part at most (estimate, milliseconds with Flock's parallel grind); no byte change
(Flock nonces are 8-byte words already). Consequences: the transcript validity rule
changes, so the K12 patch is no longer prover-only and our Flock proofs stop verifying
with upstream `83f963e` (and vice versa); the profile ids SHOULD change (or a Flock
proof version byte SHOULD be added) so that old proofs are rejected cleanly. Before
patching, the finding SHOULD be raised with the Flock authors (they may have an argument
for the outer sites that their audit document does not state; Section 11, open 1). Until
then, the paper MUST NOT claim 128 bits for the Keccak side.

Implemented in step 2 (`vendor/flock-core/PATCH.md` Section 5, deviation K19): the policies
take `list = ceil(L_max)` = 36, 50, 71 (an integer upper bound of `L_max` = 35.36, 50, 70.71),
`bits = floor(log2(nu list)) + 1`; for every site above this gives exactly the bits listed
(rounds 7 / 7 / 8, `alpha`/`beta`/claim batch 6 / 6 / 7, ring switch 8 / 9 / 9, zerocheck skip
13 / 13 / 14, lincheck skip `k + 6 / k + 6 / k + 7`, zerocheck initial 11 / 11 / 12 at `m = 30`),
checked by `agg_keccak::prover::tests::flock_outer_sites_carry_the_level0_list`. The
multipoint batch and rounds get the same factor; the AG-basis skip points (not on our path)
keep upstream's bits. Raising the finding upstream is still open.

### 4.4 Flock: `Slim8`'s transport claim batch

`OpeningGrinding::per_challenge_128` fixes `claim_batch_bits = 6` for every profile, on
the premise "`log2 L < 6` in every embedded strict profile". `Slim8` (our K18, level-0
rate 1/8) has `log2 L_max = 6.14`, and its own Ligerito level-0 claim batch already uses 7
bits (`m*_slim8.toml`). The transport batch is short by 0.14 bits. Fix: 7 bits for
`Slim8` (subsumed by 4.3). `Slim8` is opt-in, not the default.

### 4.5 The tied part: points not bound into the instance challengers (finding)

`TieChallenges::derive` absorbs `fs`, `d`, `nu_s`, `log2 blocks`, `span_log`, the
final-vector digest, the sponge cap and the union cap. The sponge instance's challenger
(`TiedPart::sponge_challenger`) absorbs `fs`, its label, the gate digest, the public
digest and the tie-geometry digest, then Flock binds its own cap; it never sees the
union's cap or the final-vector digest. The union's challenger absorbs neither the
final-vector digest nor (except through `fs`) the sponge cap. Flock's merged opening
absorbs packed-direct claim values only, "the points are transcript-derived and
verifier-recomputed" (`pcs.rs`, `verify_batch_merged_core`); here the points come from a
different transcript. So the sponge opening's claim-batch coefficients are drawn from an
input that does not encode the tie points, contrary to the requirement of Section 1, and
Proposition 10.3's hypothesis (bad set fixed before the query) fails formally: a prover
may query the coefficients first and re-derive `rho` by committing a different union
afterwards. No concrete attack below `2^128` is known to us (the tie checks tie the values
to the union's own, list-bound evaluations), but the theorem needs the binding.
**Normative fix (tied-part proof version 4):** after drawing the tie coordinates, the tie
challenger squeezes a 32-byte tie digest (two more `F128` words), and both instance
challengers MUST absorb it (framed, label `agg-keccak/v4/tie-binding`) right after their
construction and before `bind_statement`. This binds the union cap, the final-vector
digest, the sponge cap, the geometry and the tie nonce into both instances.

### 4.6 Akita: accepted support of the sparse fold challenges (hypothesis H1, discharged)

**Requirement.** A fold coordinate of group `g` at level `h` is an indexed address of weight
`1/|C^acc_{h,g}|` (Akita Definition 10.16, (159); Theorem 10.46, (211)), and Aerie Eq. (19)
needs every `|C^acc| >= 2^128`. Akita's runtime floor (`SparseChallengeConfig::
validate_for_ring_dim`, `MIN_FOLD_CHALLENGE_ENTROPY_BITS = 128`, the floors of
`PRODUCTION_FOLD_CHALLENGE_LADDER`) bounds the raw family `C_0` only. Where the sampler rejects
by operator norm the raw size is not a valid denominator (Aerie C.3); the accepted size is.

**Dispatch at `e023b6a`** (`akita_types::draw_group_fold_challenges` for the nonterminal
levels, `akita_verifier::fold::terminal` for the terminal):

* evaluation trace: drawn in the inner ring, `D = d_A`, with operator-norm rejection iff the
  inner route is L2 and the family is a certified one (`selective_l2_operator_norm_rejection`);
* subring coefficient packing: drawn in the challenge subring, `D` = the challenge subring
  dimension, never rejected (the route must be L-infinity); the stride embedding into the
  ambient ring is injective, so the support is that of the subring draw;
* terminal: `D = d_A`, rejection iff the terminal has an L2 response cap.

Each coordinate reads its own stream and makes at most 4,096 draws from `C_0`: `w = a + b`
distinct positions by a partial Fisher-Yates shuffle over exactly rejection-sampled integers,
then independent signs, magnitude 1 on the first `a` positions; every element of `C_0` has
exactly `a! b!` orderings, so a draw is uniform on `C_0` in the ideal-XOF model (Akita 10.1).
Conditioned on success the output is uniform on `C^acc` (Akita Lemma 10.1); exhaustion is a
completeness event.

**Check (test `crates/agg-pcs-config/tests/akita_fold_support.rs`).** Every row of both
catalogs: width 4, 52 rows (per relation H-u and H-exact, the 13 grouped rows and the 13
scalar rows of `G_main4`, 26 grouped in all); width 16, 25 rows (shared by both relations) (the scalar rows of `G_main16`, `G_1`, `G_2`, `G_3` at every size, equal layouts
sharing a row). For every level, every group and the terminal the test runs the pinned
dispatch with a recording `FoldDraw`, which reads `D`, the family, the packing domain and the
rejection policy from the draw's absorb payload and answers a fixed seed, so that the real
sampler (rejection loop included) draws the challenges. It checks the record against the
schedule and every sampled challenge against the family, and requires an accepted support of
at least `2^128`: exactly, in integers, for an unrestricted family; by the certified accepted
count `ceil(N_raw p0)` for a rejected one (`p0` the certificate's exact lower bound on the
accepted fraction of the strict runtime predicate). `docs/scripts/fold_support.py` recomputes
the family sizes and the certified counts from the certificate files of the pinned tree, and
Akita's exact checkers (`scripts/operator_norm/d64/check_cert.py`, `d128/check_cert.py`) print
`ALL CHECKS PASS` for both certificates (run 2026-10-25, 3.5 s and 0.6 s).

| Draw | `D` | `(a, b)` (signs on `+-1`, `+-2`) | Rejection | `log2 |C_0|` | `log2 |C^acc|` | Draws (width 4 / width 16) |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| packing | 64 | (31, 10) | none | 128.0881 | 128.0881 | 151 / 24 |
| packing | 128 | (31, 0) | none | 129.6214 | 129.6214 | 79 / 23 |
| packing | 256 | (23, 0) | none | 131.0778 | 131.0778 | 4 / 2 |
| trace | 128 | (31, 0) | none | 129.6214 | 129.6214 | 64 / 26 |
| trace and terminal | 128 | (31, 0) | operator norm, threshold 13 | 129.6214 | 128.5633 (certified) | 142 / 60 |
| terminal | 64 | (31, 10) | none | 128.0881 | 128.0881 | 0 / 1 |

The certified count of the threshold-13 family is `ceil(N_raw p0) =
502822610611069237129947393810277364817` (`cert_d128_w31_gamma13.json`, `p0 = 0.48027`), the
value of Aerie C.3. Akita's other certified family, `D = 64`, `(31, 11)`, threshold 18
(`2^128.0624`), does not occur in our catalogs. Per grouped row (both relations): 11 draws at
`d_sig = 3` to 11, 12 at 12 to 15. Under H-u the smallest support is `2^128.0881` at `d_sig =
3` to 14 (a root group or level 1 drawing the `D = 64` packing family) and `2^128.5633` at
`d_sig = 15`; under H-exact it is `2^128.0881` at every `d_sig` from 3 to 15 (the `2^15` row's
level 1 draws the `D = 64` packing family). The width-4 draw counts above cover both
relations' rows (the merge of `h-exact` doubled them from 76, 39, 2, 32 and 71).

**Result.** H1 holds for every row of both catalogs. The smallest accepted support is
`2^128.0881` (Akita's production `D = 64` shell, margin 0.088 bits), so every fold coordinate
weighs at most `2^-128.088` and needs no work. Nothing is below the floor and nothing needs to
change. The test pins the per-row minima, so a catalog regeneration or an Akita change that
alters a family fails it.

## 5. Tighter accounting (round-by-round), with the argument

**Lemma S (checks of fixed objects).** Fix the 38 Akita polynomials `f` (39 under H-exact) (values in `F_p`
on their Boolean domains) and a statement. For every challenge input `u` of a PIOP site from
`X` (event 7) to the PCS export, let
`D(u) = 1` iff a verifier check determined by `u` fails or a pending claim is false,
where the pending claims are: at `X`, the relation identities of `f` (Sections 4 to 6, (9.2)
with the absorbed `Y_sigma`; `R_PIOP` of Section 12.2); after each site, the claims listed
in Section 3 (layer claims, stack and table leaf claims, stage input claims, terminal
evaluation claims). (The sites before `X`, i.e. `sd`, `fs`, the tie sites and Flock's, are
treated in Section 12.4 with the round-1 polynomials and the Flock words' Johnson lists.) Then for every site `t` of Section 3 and every input `u` with
`D(u) = 1`, there is a set `Bad_t(u)`, determined by `u` and the fixed objects, such that
an answer outside `Bad_t(u)` leaves `D = 1` at every next input, and
`Pr[Bad_t(u)] <= 2^-g_t nu_t / |F_t|` including the predicate (Akita Remark 10.18).

*Proof sketch.* Choose the first false pending claim and define `Bad_t` by it alone;
Sections 3.2 to 3.6 give the polynomial whose zero set it is and its degree. Every claim
not acted on by the site stays false. A false claim never needs two sites at once: a
batching vector turns any false input claim of a stage into one false batched claim, the
stage rounds then act on that single claim, and the terminal check is exact. This is
Aerie Lemma C.1 with our instances; at `X` the bad set is Section 3.2's. (At `sd` and at
the tie sites the bad sets are those of Section 12.4, built on Sections 3.5 and 3.6.)

**Consequences.**

1. *Reductions are credited to the sum-checks* (`docs/lookup.md` Open 4). An evaluation
   reduction `f(pi) = sum_x eq(pi, x) f(x)` (the 37 reductions and the four stack
   reductions (7.7)) has the earlier claim as its input claim; a false input claim makes
   the instance false and costs nothing beyond `gamma^(3)` (`nu = 1`) and the stage-3
   rounds (`nu = 3` each), which every instance shares. Spec 12.1 charged each of them
   `d_main/|K|` ("linear identities at a point", 62 instances), as if each had its own
   sum-check.
2. *One site for all zero-checks and linear identities at `tau`*: `nu_tau = d_main`, not
   spec 12.1's `19 d_main` for the zero-checks plus `d_main` per linear identity.
3. *Leaves at `r^(1)` and links at `r^(2)`*: inside the rounds (3.4), not spec 12.1's
   `4 d_main`; the ring identity at `r^(2)` likewise (spec 12.1 counted it among the 62).
4. *GKR batching vectors*: `nu = 1` each; spec 12.1 already used this (`docs/lookup.md`
   Section 2); the assigned bound keeps Aerie's `l/|K|`.

Before work, the round-by-round union bound (sum of `nu_t/|F_t|`, which is also the
interactive soundness error of the PIOP, plus the `X` and `sd` terms `2^-201.9` and about
`2^-139`) is `(11 d_main + d_main + 3 + GKR)/|K|` (the stage degrees sum to 11 in H-u):
`979/|K| = 2^-118.1` at `2^10`, `1,228/|K| = 2^-117.7` at `2^13`, `1,409/|K| = 2^-117.5`
at `2^15`, against spec 12.1's `2,659`, `3,160`, `3,509` over `|K|` (relaxed H, stage degrees
summing to 12: `999`, `1,251`, `1,434` and `2,679`, `3,183`, `3,534`). This is still not
128 bits; the schedule makes every site `<= 2^-128` per address, which is the statement of
Section 7.

**Where the conservative bound stays.** The schedule (`e_t`) keeps Aerie's assigned bounds
(`l/|K|` for batching vectors) and the union over the tie identities; `nu_t` (the sharper
value) is what the analysis needs. Spec 12.1 SHOULD be replaced by Sections 3 and 5 once
step 2 is merged.

## 6. Nonce format (normative)

### 6.1 Predicate and challenge at a PIOP site

At a PIOP challenge event with `g > 0` (prover and verifier run the same state machine):

1. The nonce `v` (`0 <= v < 2^(g + 7)`) is absorbed as one public message `LE32(v)` into
   the Spongefish sponge, immediately after the last absorbed payload before the site.
2. The 32-byte predicate is squeezed from the sponge; the site passes iff its low `g` bits
   are zero, bit order low-bit-first from byte 0 (Akita `grinding_predicate_accepts`).
3. The challenge (all `16 * count` bytes of the site, per-limb rejection sampling) is
   squeezed from the same sponge afterwards: predicate and challenge are disjoint,
   consecutive parts of one output stream at one address (Aerie C.1: "the stream's first
   `g_t` bits must be zero; disjoint later bits generate the challenge"), which realizes
   the joint experiment of Akita Proposition 10.3 without a separate address. No site
   label is absorbed (Aerie C.1 does include labels in the hash input; we do not need
   them): the contexts stay distinct because (i) the registry, hence the event sequence,
   is fixed by the manifest bound in event 0 together with the plan digest, (ii) every
   payload absorbed right before a nonce has a length fixed by the registry, and (iii) the
   variable-length items (event-2 and event-6 Flock items, the `c~` list) are framed with
   their lengths. The absorbed prefixes are therefore injective in the transcript, which is
   the hypothesis of Akita Proposition 10.3.
4. Sites with `g = 0` (only `X`) absorb nothing and squeeze no predicate.

### 6.2 The packed nonce stream

One byte string per proof, `work`, carries the PIOP nonces in plan order: nonce `i`
occupies bits `[o_i, o_i + g_i + 7)`, `o_i = sum_{j < i} (g_j + 7)`, little-endian bit order
within the stream (bit `k` is bit `k mod 8` of byte `k / 8`). Its length MUST be exactly
`ceil(sum_i (g_i + 7) / 8)` bytes and its padding bits MUST be zero; the verifier rejects
any other length or nonzero padding before replaying event 7. In the wire format it
follows the stage-3 terminal claims and precedes the Akita opening (wire version 2). The
tie nonces are not in this stream (6.3); Flock's and Akita's nonces stay in their own
proofs.

### 6.3 The tie nonce

In `TieChallenges::derive`, after the last absorb (the union cap) and before the first
coordinate: `nonce = grind_pow(g_tie)` on the tie challenger (Flock's
`Challenger::grind_pow` / `verify_pow`: predicate BLAKE3 of `state_digest || LE64(nonce)
|| 0^24` with `g_tie` leading zero bits, nonce absorbed as `LE64`), then the coordinates, then the
tie digest of 4.5. The honest search MUST stop at `2^(g_tie + 7)` candidates (Flock's
`grind_pow` searches upward from 0 and returns the smallest passing nonce; if that nonce is
`>= 2^(g_tie + 7)`, the prover stops with `KeccakError::Work`, which mode A returns as
`ProveError::Keccak`). The nonce travels in the tied-part proof header as `LE32`; the verifier
rejects a value `>= 2^(g_tie + 7)`. Flock's own nonces get the same width rule in the
vendored verifier (`vendor/flock-core/PATCH.md` Section 6, deviation K19).

### 6.4 The plan and its digest

The plan of a proof is the ordered list of its tie sites (one per tied part, in segment and
part order) followed by the PIOP field-challenge sites in registry order (`X` included
with `g = 0`). It is a function of public data only: `N` (hence `N*`), the Flock profile,
and the tied-part geometry (span byte, part size) of the event-2 items, which the verifier
parses before replaying event 0. Canonical encoding:

```text
body  = LE64(N*) || u8(profile id: 1 Fast, 4 Slim, 5 Slim8, 3 mode B)
        || u8(S, the span length; 0 in mode B) || u8(d_part; 0 in mode B)
        || LE32(site count) || entries
entry = u8(class id, Section 3.1; 8 = tie) || LE16(i) || LE16(j) || u8(g) || u8(nonce bits)
        (i, j) = (s, 0) for lookup.mu and lookup.gamma, (s, r) for lookup.round,
                 (stage, 0) / (stage, r) for stage sites, (part index, 0) for tie sites,
                 (0, 0) for lookup.x and tau
digest = SHA3-256("agg-mldsa/v2/pow-plan" || LE64(len(body)) || body)
```

The 32-byte digest MUST be appended to the instance record of event 0 (after the Flock
profile byte, `binding::instance_record`), so it is absorbed before the first site; the
manifest's wire version becomes 2. The script prints test vectors for `N = N*` (Section
10); the implementation MUST reproduce them. (The plan uses the actual `n` of each tied
part in `nu_s`; the vectors use full batches.)

### 6.5 Honest search allowance and failure probability

The prover tries `v = 0, 1, ...` and takes the first passing value, at most `2^(g + 7)`
candidates (deterministic proofs). A site fails with probability
`(1 - 2^-g)^(2^(g + 7)) <= exp(-2^7) = 2^-184.7`. The capped searches of a proof are the
PIOP sites (at most 484), the tie sites (one per tied part: at most `2 S <= 8` with the
two-part cap of 3.5), Akita's capped transcript searches (459 at `2^15`) and, since the
vendored Flock verifier rejects nonces of `g + 7` or more bits (`vendor/flock-core/PATCH.md`
Section 6), every Flock site with work: Flock's prover still searches upward without a cap,
so a smallest passing nonce of `2^(g + 7)` or more yields a proof that the verifier refuses,
with the same probability per site. Flock has fewer than `2^10` sites with work per tied part
(estimate from the round counts: two instances, each a zerocheck, a lincheck, the opening's
ring switch, claim batch and merged rounds, and at most six Ligerito levels with two batches
and one query nonce each, at `m <= 45` variables), so at most `2^13` for eight parts. The
completeness error is therefore at most `(484 + 8 + 459 + 2^13) 2^-184.7 < 2^-171.5`
(the Flock count an estimate; without it `< 2^-174.8`). A tie search that exhausts its
allowance returns `KeccakError::Work` (no panic). Expected work per site `2^g` predicate
evaluations.

## 7. Theorem (128-bit knowledge soundness, classical ROM)

**Theorem.** Fix `N <= 2^15`, mode A with Flock profile `Fast` or `Slim` with the fix of
4.3 (or `Slim8` with 4.3 and 4.4), tied-part proof version 4 (4.5), PIOP wire version 3
(relation H-u or H-exact, the caller's relation, bound by byte 3 of event 0 and checked by the
verifier before event 0) with the plan of Section 6 (Section 10.1 for H-exact), at most two
tied parts per segment (3.5), and Akita's plan for the selected catalog row. The two former
hypotheses are settled:

* **(H1, discharged in 4.6)** every fold group of every row of both catalogs draws its
  sparse fold challenges from a family whose accepted support is at least `2^128` after
  rejection (smallest `2^128.0881`; test `agg-pcs-config/tests/akita_fold_support.rs`);
* **(H2, proved in Section 12 under (F) and (K))** the composition of Akita's extractor with
  the Flock commitments: no second extractor is needed (the extractor returns Akita's
  round-1 sources; the Ligerito words are fixed straight-line by the Merkle expansion and
  enter only the round-by-round bits through their Johnson lists), with a checkpoint at `sd`
  (round-1 candidate) and one at `X` (all six groups), and Lemma 12.3 for the parallel
  Flock transcripts.

The theorem is conditional on the assumptions of Section 12.5: **(F)** Flock's documented
round-by-round analysis of its instances (per-site bounds as validated by its configs, the
list factor at the outer sites by K19, its work composition) and **(K)** the correctness of
the sponge and bridge-union circuits. The paper MUST state (F) and (K) as assumptions and
MUST NOT claim the theorem without them. For every
prover `P` making at most `Q` oracle queries (all oracles of Section 12.1, every nonce probe
included) there is an extractor of expected cost polynomial in that of `P` (at most
`(Q_wrap + 2) 4 Tree0_ck(sch)` runs, Section 12.6) that returns the original execution and,
whenever the verifier accepts, a witness of the caller's relation (H-u or H-exact) for every
live record, except with
probability at most

```text
eps_ext <= (Q + R_f + R_coord + R_F) 2^-128 + eps_bind + eps_pk + eps_dig + eps_fs
           + eps_merkle + delta_X + delta_samp + sum_I Adv^MSIS_I(T_I)
```

where `R_f` is the number of field sites of the PIOP and of Akita read by the verifier
(484 + 459 at `2^15`, plus the seeds and the tie sites), `R_coord` Akita's sparse fold coordinates read by
the verifier, `R_F` the Flock sites read by the verifier (both instances, outer and
Ligerito, queries included), `eps_bind, eps_pk, eps_dig` the hashed-entry terms of Aerie
Definition 5.8 (instance digest, public-record digest, PCS session digest, Flock part and
span seeds, final-vector digests, tie bindings, Flock statement digests, the BLAKE3 chaining
of Flock's challenger), `eps_fs` the Flock-seed term of Lemma 12.1, `eps_merkle` the
Ligerito Merkle term of Lemma 12.2, `delta_X < 2^-(2^110)` the seed-density term of Lemma
12.4, `delta_samp = 0` for the exact samplers, and the Module-SIS terms those of Akita
Lemma 10.20 for the schedule's occurrences.

*Proof (Section 12).* (1) Hashed entries, the Flock seeds and the Ligerito caps are expanded
(Akita Lemma 10.47, Lemmas 12.1, 12.2): `eps_bind + eps_pk + eps_dig + eps_fs + eps_merkle`.
(2) A checkpoint structure for the expanded protocol (12.2): checkpoints at `sd` (candidate
the round-1 sources, test the caller's relation H-u or H-exact) and at `X` (candidate all six groups, test the PIOP
relation), Akita's opening suffix (Proposition 10.44) after the root fold, and Lemma 12.3 for
the tied parts' own challengers. (3) Item (vii) of Akita Definition 10.16: Lemma S and the
evaluation batch on the segment of `X` (12.3); the bridge seed with the sponge lists, the
tie sites with the list factors and Flock's per-site bounds under (F) on the segment of `sd`
(12.4), with Lemma 12.4 for the seed. (4) Every address weighs at most `2^-128` (Section 3,
the script's assertion `max weight <= 2^-128`; `sd` at most `2^-131.38`; Flock with 4.3/4.4
under (F); Akita by its plan and by the accepted sparse-challenge supports of 4.6), so
Akita Theorem 10.17 with Aerie Eq. (18) gives `rho (Q + R_f + R_coord + R_F)` with `rho <=
2^-128`. (5) The witness map returns the round-1 tables, which satisfy the caller's relation by the
test at `sd` (spec 12.2: lookup, leaves, the Booleanity zero-checks of `B_zhi` and `a_z`
under H-exact, the bridge with `b = b'`).

*What is not proved here* (Section 12.7): assumption (F), which the Flock paper presents
as a sketch; assumption (K); Lemma 12.3 extends Akita's Theorem 10.17 and should be checked
by a reviewer of Akita's proof; `R_F` is not counted exactly; Akita's Module-SIS estimates
exclude the extractor's run count as in Aerie. As in Aerie, the term
`(R_f + R_coord + R_F) 2^-128` is not negligible at `Q = 0` (about `2^-117` to `2^-118`,
estimate); the claim is 128 bits per query, Aerie's notion.

*Mode B* (public bits, no Flock): Corollary 12.5 gives the same statement without (F).

## 8. Costs at `N = 2^15` (default: `Slim`, span on)

| Item | Proof bytes | Prover hashes (expected) |
| --- | ---: | ---: |
| PIOP packed stream (483 nonces, 4,444 bits) | 556 | 2,718 Blake2b predicates (about 2 ms serial, estimate) |
| Tie nonce (`g = 17`) | 4 | `2^17` BLAKE3 (about 0.5 ms on 7 threads, estimate from the measured 0.22 ms at 16 bits and 2.9 ms at 20 bits) |
| Akita (already in the 62,974-B opening) | 0 added (938 max today) | 0 added (9,872 today) |
| Flock fix of 4.3 | 0 (nonces are 8-byte words already) | about 100 sites per instance at `2^7` to `2^14` (estimate, milliseconds) |
| Plan digest in event 0 | 0 (absorbed, not sent) | one SHA3-256 |
| **Total added** | **560 B = 0.017 B/sig** (51.0 B/sig unchanged at one decimal) | well under 0.1 s (estimate) against 23.7 s |

**Measured (local, direction only, 2026-10-25).** `agg bench --count 1024 --keys 16
--warmup 1 --reps 5`, CLMUL build (`RUSTFLAGS=-C target-feature=+pclmulqdq`, separate target
dir), `RAYON_NUM_THREADS=3`, `taskset -c 0,1,3-15`, shared machine (load average 5.2 to 5.8),
the three binaries run back to back, `Slim`, one tied part:

| Binary | Commit | Prove (median of 5) | Flock parts | Aggregate | PIOP nonce stream |
| --- | --- | ---: | ---: | ---: | ---: |
| before | `7e635a6` (code `be398d8`) | 4,035.1 ms | 2,189.2 ms | 522,076 B | none |
| PIOP and tie work, no Flock fix | `a5f23da` | 4,134.4 ms | 2,203.9 ms | 522,435 B | 391 B |
| all of step 2 (with K19) | `ed966a6` | 4,081.7 ms | 2,184.7 ms | 522,465 B | 391 B |

Bytes: +391 B nonce stream and +4 B tie nonce, as Section 10 predicts (395 B at `2^10`); the
Akita opening varies by a few bytes between runs (inline LEB128 nonces). Prover time: the
differences (+1.2% and +2.5%) are inside the run-to-run spread on this shared machine (the
minimum of each set: 3,898, 4,025, 4,000 ms), consistent with the estimated few milliseconds
of hashing; the Flock fix adds no measurable time. Verify medians 442 / 434 / 425 ms
(uncached keys).

Span off at `2^15`: four tie nonces (572 B in all). The plan's earlier estimate (about
500 nonces, about 1 KB, under 10 ms) holds for the PIOP; the tie site is 10 bits harder
than the plan assumed because of the list factor.

## 9. Implementation checklist (step 2)

PIOP (`crates/agg-piop`):

1. `agg_bridge` (shared by `agg-piop` and `agg-keccak`) gets a `pow` module:
   `grind_bits_k(nu)`, `grind_bits_f128(num)`, `list_size(profile)` (35, 50, 70 with a test
   against `floor(1/(2 eta sqrt(rho_0)))` of the shipped configs), `tie_difficulty(profile,
   d, span_log, blocks_log, nu_s)`, the site-class enum, the plan builder and its canonical
   encoding/digest (Section 6.4), with tests reproducing every digest of Section 10.
2. `transcript.rs`: `RegistryEvent` gets the difficulty `g` of each `CHALLENGE_KIND`
   event from the plan; `Registry::new(manifest, plan)`; a work stream (prover: writer that
   searches and records; verifier: reader over `work`); `challenge_extensions` performs
   6.1 before sampling when `g > 0`; `ensure_complete` also checks that the work reader is
   exhausted. `challenge_bytes` (`sd`, `fs`) and `pcs_session` stay unprotected.
3. Call sites need no change beyond passing the plan: `lookup::{prove_in, verify_in}`
   (through `LookupTranscript::challenges`), `stage::{prove_stage_with, verify}`,
   `prover::prove_piop`/`verifier::verify_piop` (`tau`).
4. `manifest.rs`: `WIRE_VERSION = 2`; `binding::instance_record` appends the plan digest.
5. `wire.rs`: `AggregateProof.work: Vec<u8>` after `stage3`, before `opening`; exact
   length and zero padding (6.2); the header version 2; old version rejected.
6. Deviation P3 resolved (spec Section 10 text, `docs/piop.md` Section 9, status.md).

Keccak side (`crates/agg-keccak`, `crates/agg-cli`):

7. `tied.rs`: `VERSION = 4`; header `AGGKCT || 4 || profile || LE32(n) || d || S ||
   LE32(tie nonce) || ...`; `TieChallenges::derive` takes the difficulty and does 6.3 (the
   prover returns the nonce, the verifier checks it and the range); the tie digest of 4.5
   is absorbed by `sponge_challenger` and `bridge_challenger` in prover and verifier.
8. `agg_cli::mode_a`: compute the plan from `N`, the profile and the item geometry before
   event 0 on both sides; segment item version bump if the item framing changes.
9. Vendored `flock-core` (after the decision of Section 11, open 1): list-aware bits at
   the outer sites (4.3) and `Slim8`'s claim batch (4.4); update `vendor/flock-core/
   PATCH.md` and deviation K12 (no longer prover-only).

Akita (`crates/agg-pcs-config`): nothing to enable; add a test pinning the plan digests
of the 13 grouped rows (4.1).

Negative tests (one per class, each MUST be rejected): a wrong nonce at `mu_s`, `gamma_s`,
a GKR round, `tau`, `gamma^(s)` and a round of each stage; a nonce `>= 2^(g + 7)`; a
`work` stream one byte short, one byte long, or with a nonzero padding bit; a plan-digest
mismatch in event 0 (another profile, another span byte); a wrong tie nonce; a tie digest
not absorbed (old version 3 proof); a proof whose `X` site carries a nonce. Positive
tests: plan digests equal the script's; honest proofs round-trip at 256, 1,024 and 8,192
records in mode A and mode B.

## 10. Script output summary (`python3 -I docs/scripts/pow_plan.py`, 2026-10-25)

(Relation H-exact: Section 10.1.)

Relation H-u (stage degrees 4, 4, 3). Against relaxed H (4, 5, 3) only the ledger column and
the stage-2 round class move: every difficulty, nonce width, total, byte count and plan
digest below is unchanged (`bit_length(4) = bit_length(5) = 3`; the script asserts it), and
the relation is bound by the event-0 manifest bytes (wire version 3, `G_main4` alphabet code
6 for `U_hi`), not by the plan.

PIOP sites per `N*` (`bits` = packed-stream bits, `work` = `sum 2^g`, ledger = round-by-round
sum before work):

```text
d_sig  N*      d_main  D   L            X-site log2   sites  nonces  bits  bytes  work(sum 2^g)  est.ms  max weight  ledger(before work)
    3       8      13  16       167936        -220.3    195     194  1811    227           1364    0.99  2^-128.30  550/|K| = 2^-118.9
    4      16      14  16       200704        -220.0    198     197  1841    231           1400    1.01  2^-128.19  562/|K| = 2^-118.9
    5      32      15  17       266240        -219.3    219     218  2032    254           1494    1.08  2^-128.09  624/|K| = 2^-118.7
    6      64      16  18       397312        -218.4    241     240  2233    280           1608    1.16  2^-128.42  689/|K| = 2^-118.6
    7     128      17  19       659456        -217.1    264     263  2442    306           1710    1.24  2^-128.42  757/|K| = 2^-118.4
    8     256      18  20      1183744        -215.5    288     287  2660    333           1816    1.31  2^-128.42  828/|K| = 2^-118.3
    9     512      19  21      2232320        -213.7    313     312  2887    361           1926    1.39  2^-128.42  902/|K| = 2^-118.2
   10    1024      20  22      4395008        -211.8    339     338  3124    391           2048    1.48  2^-128.42  979/|K| = 2^-118.1
   11    2048      21  23      8720384        -209.9    366     365  3370    422           2174    1.57  2^-128.42  1059/|K| = 2^-118.0
   12    4096      22  24     17371136        -207.9    394     393  3625    454           2304    1.67  2^-128.42  1142/|K| = 2^-117.8
   13    8192      23  25     34672640        -205.9    423     422  3889    487           2438    1.77  2^-128.42  1228/|K| = 2^-117.7
   14   16384      24  26     69345280        -203.9    453     452  4162    521           2576    1.87  2^-128.42  1317/|K| = 2^-117.6
   15   32768      25  27    138690560        -201.9    484     483  4444    556           2718    1.97  2^-128.36  1409/|K| = 2^-117.5
```

Tie site, full parts (`n = 2^d S`), selected rows (the script prints every `d = 3..13`):

```text
profile  d   S  nu_s  terms(tie1,tie2,out,pub)  plain  g_plain  list  listed    g   nonce  tight(RBR)  g_tight
fast     3   1     4  (8, 9, 5, 7)                 46        6    35    20650   15     22       10360       14
fast    13   1    13  (18, 13, 5, 16)              83        7    35    45745   16     23       23065       15
fast    13   4    15  (20, 15, 7, 18)              95        7    35    50925   16     23       25655       15
slim     3   1     4  (8, 9, 5, 7)                 46        6    50    41500   16     23       20800       15
slim    10   1    11  (15, 10, 5, 14)              69        7    50    76950   17     24       38700       16
slim    13   1    13  (18, 13, 5, 16)              83        7    50    92350   17     24       46450       16
slim    13   2    14  (19, 14, 6, 17)              89        7    50    97550   17     24       49050       16
slim    13   4    15  (20, 15, 7, 18)              95        7    50   102750   17     24       51650       16
slim8   13   4    15  (20, 15, 7, 18)              95        7    70   199850   18     25      100310       17
```

Flock outer sites under the paper's list accounting (bits = `128 + bits - log2(nu L_max)`):

```text
                            Fast     Slim     Slim8   (log2 L_max 5.14 / 5.64 / 6.14)
zerocheck initial (m=30)   122.95   122.45   121.95
zerocheck skip             122.87   122.37   121.87
zerocheck / lincheck round 123.86   123.36   122.86
lincheck alpha/beta        123.86   123.36   122.86
lincheck skip (k=6)        122.88   122.38   121.88
ring-switch point          123.05   122.55   122.05
transport claim batch      128.86   128.36   127.86
merged-sumcheck round      123.86   123.36   122.86
```

Seed `sd` with the sponge lists (Section 3.5, `Fast / Slim / Slim8`): span on `2^-139.51 /
2^-139.00 / 2^-138.51`; span off at four segments `2^-137.51 / 2^-137.00 / 2^-136.51`; two
parts in each of four segments `2^-132.38 / 2^-131.35 / 2^-130.38`; all `g = 0` (three
parts in one segment, refused by mode A, would give `2^-127.71` under `Slim`). Old ledger vs
round-by-round at `2^15`: `3,509/|K| = 2^-116.2` vs `1,409/|K| = 2^-117.5` (stages 275,
`tau` 25, gammas 3, GKR 1,106). Honest failure per site `<= 2^-184.7`, per proof
`<= 2^-174.8` (951 capped searches: PIOP, 8 tie, 459 Akita; with the Flock sites of the
canonical nonce width `< 2^-171.5`, Section 6.5).

Totals with the tie nonce (`N = N*`): `2^10` 395 B (0.386 B/sig), `2^13` 491 B
(0.060 B/sig), `2^15` 560 B (0.017 B/sig) with the span, 572 B with span off.

Plan digests (Section 6.4 encoding, `N = N*`, default part size; the script prints all 13
sizes and every profile, these are the anchors the implementation tests MUST reproduce):

```text
N* = 2^3  modeB          7f1274590f56e00bad2aabff22f105f789aba6af1e6cbcbcc3f16f826dc0890a
N* = 2^3  slim   span on  1d9b37d94b1b32c19fae1312ac52d5da937824e6b8253585c00a3ccd67d1fd32
N* = 2^10 modeB          8e8eb62946890f626d31f6d135108a2c221bd6057b15aa961fcdb7663edc6356
N* = 2^10 slim   span on  2d289df145671419fe780abaa2542b28870ce0349ac8cd82ccd9da6a5b091a5b
N* = 2^13 slim   span on  86581fc1f62e7dfa20d85d03868821a6cc83296e78c9d3e2fcdedaa6d2449e73
N* = 2^15 modeB          829bd67dd0e236a142d62e83cc4d3bfbfebb54d9ccf035333e7e7d6db7c89787
N* = 2^15 fast   span on  57f65044067580b0234487193ebfd49e29093d40423f528600b804db6d90e156
N* = 2^15 slim   span on  df9161a54a441de0e4065c8129aa9f956f4146427bd3336e59639e0cfdf2e38c
N* = 2^15 slim   span off 5f69f20374997a0b0a025cf853549d98152025a4ebf946c84a55235c03c70ac0
N* = 2^15 slim8  span on  902578ff518bcb5adfb672be263e13286e32226d2cdfb8853066b89250f64795
```

The digests depend on the tie difficulties of Section 3.6 and the Flock profile; if the
decision of Section 11 (open 1 or 2) changes a difficulty, the script and these anchors
change with it.

### 10.1 Relation H-exact (`python3 -I docs/scripts/pow_plan.py --relation h-exact`)

Relation H-exact (`docs/h-exact.md` step B, relation byte 1) has the same stage degrees
(4, 4, 3) and the batching vectors `gamma^(1..3)` of 26, 7, 45 coefficients: stage 2 holds
`def_y1[t]` and one `leaf6[D_u]` (the Boolean `B_zhi` needs no leaf), stage 3 one link and
the two Booleanity zero-checks `bool[B_zhi]`, `bool[a_z]` at `r^(2)`. The zero-checks add no
site: they are identities tested at a point drawn by the preceding stage (3.4), pending
during the stage-2 rounds, and enter through `gamma^(3)`. So the only difficulty that moves
is the stage-2 batching site, `g = bit_length(7) = 3` (10-bit nonce) instead of 4 (11 bits);
`gamma^(3)` keeps `g = 6` (45 < 64), and the round-by-round ledger is H-u's (it depends on
the stage degrees only). The plan encoding (6.4) has no relation field: the relation is
bound by the manifest bytes of the same event 0 (byte 3 and the `G_main4` descriptor), and
the stage-2 entry already makes every H-exact digest differ from H-u's.

```text
d_sig  N*      sites  nonces  bits  bytes  work(sum 2^g)  ledger(before work)
    3       8    195     194  1810    227           1356  550/|K| = 2^-118.9
   10    1024    339     338  3123    391           2040  979/|K| = 2^-118.1
   13    8192    423     422  3888    486           2430  1228/|K| = 2^-117.7
   15   32768    484     483  4443    556           2710  1409/|K| = 2^-117.5
```

The maximum site weight is the same as H-u's (`2^-128.36` at `2^15`, `2^-128.42` from `2^6`
to `2^14`). Plan digests (`N = N*`, default part size; pinned by
`agg_piop::pow::tests::h_exact_plans_reproduce_docs_h_exact_section_3_7`):

```text
H-exact  N* = 2^3   modeB          c9dd30a002089fff68b1653faddb2ac860e093f771eb3eeaaf03e60152e54c85
H-exact  N* = 2^3   slim span on   8f69f49792afc311cf43042df023b98706f92bdaecb1c3ac837690e92995e983
H-exact  N* = 2^10  modeB          750a40c19877ba6bda8c7ddee36866239b6338f7caa84109880662a33d64a1d8
H-exact  N* = 2^10  slim span on   82ffd38abe14e463ad41cc9ad209e07d5143f28067bb3696f2c445aad2c05c8f
H-exact  N* = 2^15  modeB          e4b1dffc21351ca615ace27500406a1990609ccb6cc07e13c6da50c18a416046
H-exact  N* = 2^15  slim span on   14666dea14ede0ddea3f5cc34d50e8d746021c034941d060a6c2d08af8960d41
```

The theorem of Section 7 holds verbatim for H-exact with "relation H-u" replaced by
"relation H-exact" and Akita's plan of the H-exact grouped row (4.1); spec 12.2 gives the
extraction of the exact bounds.

## 11. Open questions

1. **Flock outer sites (4.3).** Patch the vendored Flock verifier now (K12 no longer
   prover-only, upstream incompatibility) or first ask the Flock authors whether the
   outer sites have an argument that avoids the list factor? Recommendation: raise it
   upstream and patch our vendored core in step 2 under a new profile/proof version; the
   cost is negligible. Step 2: patched (K19; old mode-A proofs are refused by the
   tied-part proof version 4); raising it upstream remains open.
2. **Tie difficulty.** 15 to 18 bits with the list factors (Section 3.6) instead of 7.
   An alternative that avoids the lists would bind each instance's level-0 list before the
   tie: two explicit OOD samples per instance (one alone binds only to about 113 bits at
   `L = 50`, by Flock's `paper_ood_bits`), i.e. extra MLE evaluations of both committed
   vectors and a Flock transcript change. Grinding costs about a millisecond and is
   recommended.
3. **Composition proof (written, Section 12).** No second extractor is needed; the proof
   uses checkpoints at `sd` and `X`, the Merkle expansion and Lemma 12.3 for the parallel
   Flock transcripts. It rests on assumptions (F) (Flock's round-by-round analysis, a sketch
   in the Flock paper) and (K) (circuit correctness), and Lemma 12.3 extends Akita's Theorem
   10.17; these three items remain for the review and the paper.
4. **`R_F`** (Flock sites read by the verifier) is not counted exactly here; it enters only
   the `Q = 0` term.
5. **Mode B** uses the same PIOP plan with profile id 3 and no tie site; its Keccak side is
   the public bits, so no Flock term. Its soundness statement is Corollary 12.5 (no (F),
   `sd` at most `theta' = 2^-143.64` per charged segment).
6. **Akita sparse folds (resolved, 4.6).** The accepted (post-rejection) supports of every
   row of both catalogs are checked by `agg-pcs-config/tests/akita_fold_support.rs`: all at
   least `2^128.0881`. Hypothesis H1 of Section 7 is discharged.

## 12. Composition proof (hypothesis H2)

This section proves the theorem of Section 7 from Akita's checkpoint extraction (Akita
Definition 10.16, Theorem 10.17), its opening suffix (Proposition 10.44) and the arguments
of its Theorem 10.45, in the style of Aerie Section C. Akita's Theorem 10.45 cannot be applied
verbatim, for three reasons: the round-2 root commitments (event 4) are sent after the first
application challenge `sd`; the Flock words are committed by Merkle caps, not by Akita; and
the tied parts run on their own challengers, in parallel, so their challenge inputs do not
encode the complete preceding transcript. Sections 12.1 to 12.4 build a checkpoint
structure for the expanded protocol that handles all three, 12.5 lists exactly what is
assumed, 12.6 states the bound, and 12.7 lists what is not proved. Section 12.8 uses the
theorem for aggregate unforgeability (spec 12.3): signing queries asked only in rewound
runs, by strong unforgeability of the relaxed scheme.

The key observation is that no second extractor runs. The relation (H-u or H-exact, written
`H` below for the caller's relation) constrains the round-1
polynomials alone ((3.3) is a SHAKE256 computation on their bits `b`), so the extractor
returns Akita's round-1 sources and never decodes a Flock word. The Ligerito words enter only
the analysis: they are fixed straight-line by the Merkle expansion (Lemma 12.2), and the
round-by-round bits quantify over their Johnson lists. Every test of the structure is a
polynomial-time function of Akita candidates.

### 12.1 Oracles and the expanded protocol

Oracles (classical ROM, Section 1): the PIOP duplex `H_P` (Spongefish `Blake2b512`), the Flock
challengers `H_F` (BLAKE3 `FsChallenger`: per tied part the tie challenger and the two instance
challengers, distinct labels), Akita's native stream and coordinate streams `H_A` (SHAKE, ideal
XOF as Akita 10.1), the bridge expansion `H_X` (cSHAKE256 `agg-mldsa/v1/bridge`, spec 9.1), the
setup oracle, the hashed-entry oracles (public-records tree, plan digest, PCS session digest,
part and span seeds, final-vector digests `y_digest`, tie bindings, Flock statement digests,
the BLAKE3 chaining inside `FsChallenger`), and the Ligerito Merkle hash `H_M` (BLAKE3, 256-bit
nodes, caps). `Pi_H` is the real protocol. The expanded protocol `Pi` differs in three ways:

1. every hashed entry is sent in full (Akita Lemma 10.47; error `eps_hash = eps_bind + eps_pk
   + eps_dig`, Aerie Definition 5.8, with the entries listed above);
2. every `H_F` input that contains a Flock seed `fs_sigma` (through the part or span seed)
   contains instead the PIOP input of the `fs` site, i.e. the complete PIOP prefix through
   event 5 (Lemma 12.1);
3. every Ligerito Merkle cap (level 0 and every recursive level of every instance) is replaced
   by the committed word, and every authentication path by a read of that word (Lemma 12.2).

In `Pi` every challenge address of a part encodes the PIOP prefix through `fs`, and every
committed Flock word, hence its Johnson list `Lambda(C) = Lambda_gamma(C)` at the profile's
level-0 radius (`|Lambda| <= L = 35, 50, 70`, Flock Eq. (10)), is a function of the
transcript.

**Lemma 12.1 (the Flock seed as a compressed prefix).** A prover for `Pi_H` (after item 1)
converts into a prover for the protocol with item 2 that runs it once, outputs the same
statement and makes no more challenge queries, such that the first accepts while the second
rejects with probability at most `eps_fs = (S Q_5)^2 / 2^257 + S^2 Q_F Q_5 / 2^256`, where `Q_5`
counts the PIOP queries at `fs`-site inputs, `Q_F` the `H_F` queries and `S <= 4` the seeds per
answer (a span seed contains `S` of them).

*Proof.* The proof of Akita Lemma 10.47 with the 32-byte `fs` chunks of the `fs`-site answers
in the role of digests of their PIOP input: the converted prover forwards an `H_F` query with
the chunk replaced by the PIOP input that produced it, and answers a query whose chunk was
never produced by a fresh local answer. Its bad events are two `fs`-site answers with a common
chunk and an `H_F` query (or the output) containing a chunk that an `fs`-site answer produces
only later; without them the forwarded queries are injective and the verifier of the second
protocol reads the same answers. The count gives `eps_fs`.

**Lemma 12.2 (Merkle expansion).** A prover `P'` for the protocol with items 1 and 2
converts into a prover `P` for `Pi` that runs `P'` once, outputs the same statement and makes
no more challenge queries, such that `P'` is accepted while `P` is rejected with probability
at most `eps_merkle = (Q_M choose 2) / 2^256 + Q_M V_M / 2^256`, where `Q_M` counts the `H_M`
queries of `P'` and of the verifier on its output and `V_M` the node positions they contain.

*Proof.* The straight-line Merkle extraction of BCS16 in the form of Akita Lemma 10.47. `P`
records the `H_M` queries of `P'`; for each cap it rebuilds the tree top-down from the
recorded preimages and sets every leaf it cannot reach to zero; it forwards each challenge
query with every cap replaced by the rebuilt word, and answers a query containing a cap it
cannot rebuild by a fresh local answer. Without a collision of `H_M` and without a fresh `H_M`
output equal to a node already used, every authentication path that the verifier of `P'`
accepts consists of recorded queries, so each opened leaf equals the rebuilt word at that
position, and the verifier of `Pi`, which reads the word, sees the same values and the same
challenges. A zero-filled position is never opened without a bad event. The two bad events
have the stated probabilities.

### 12.2 The checkpoint structure

**Stages** of `Pi` in transcript order, with their types in the sense of Akita Definition
10.16 (`E` extraction, `H` checkpoint, `N` plain):

| Stages | Oracle | Type |
| --- | --- | --- |
| `sd` (event 3) | `H_P` | `H`: checkpoint `l_1` |
| `fs` (event 5) | `H_P` | `N` |
| per tied part `p`, in segment and part order: the tie site; the sponge instance's Flock sites; the union's Flock sites | `H_F` | `N` |
| `X` (event 7) | `H_P` | `H`: checkpoint `l_2` |
| the PIOP sites after `X` (Section 3.1), then Akita's evaluation batch of the 38 claims (39 under H-exact; the only Akita site before the root fold, `GrindingSite::EvaluationBatch { level: 0 }`) | `H_P`, `H_A` | `N` |
| the opening suffix from the root fold challenge (Akita Proposition 10.44): fold challenges, ring-switch `alpha` of every nonterminal level, the rest | `H_A` | `E` (indexed), `H`, `N` |

The segment of `l_1` runs from `sd` to the last stage before `X`; the segment of `l_2` from `X`
to the last stage before the root fold challenge, whose input is `xi_f`. Every plain stage lies
in one segment.

**Projected addresses.** In this linear order the stages of part `p` follow those of part
`p - 1`, but the input of a part's site (its address) encodes the PIOP prefix through `fs`
(item 2) and part `p`'s own messages only, and the two instances of a part encode the tie
binding, not each other's messages. Akita's framework assumes that a stage input encodes the
complete preceding transcript. Lemma 12.3 removes that assumption for plain stages.

**Lemma 12.3 (projected addresses).** Take a checkpoint structure whose plain stages `j` read
their answer at an address `pi_j(xi)` of the stage input `xi` (the complete linear prefix)
such that (a) `pi_j(xi)` encodes the input of the checkpoint whose segment contains `j`, and
(b) `Bad_j(xi; w)` depends on `xi` only through `pi_j(xi)`; items (i) to (viii) of Definition
10.16 hold with `D` evaluated on stage inputs. Then Theorem 10.17 holds, each queried address
of stage `j` weighing `eps_j`.

*Proof.* Akita's proof (Appendix E.2) touches the inputs of plain stages only in Claims 1 and
2. Claim 1's path argument is a statement about `D` on the stage inputs of the output
transcript and does not change: some stage `j` of the segment has `D(xi_j; w) = 1` and the
answer `K(pi_j(xi_j))` in `Bad_j(xi_j; w)`. For the probability, sample the table lazily by
address: when `A` first queries an address `a` of stage `j` in `Sub(i)`, the history, `c` and
the fixed `w` determine `a` and, by (b), the set `Bad_j(a; w)`, while `K(a)` is fresh; a union
bound over the queried addresses gives Claim 1 with `N_{i,j}` counting addresses (at most the
count of inputs). By (a) every address lies in exactly one `Sub(i)`, which is Claim 2.
Extraction stages and checkpoints keep inputs that encode the complete prefix (the Akita
stream is seeded by the PCS session digest, which binds every Flock byte absorbed at event 6;
`sd` and `X` are PIOP sites), so the rest of the proof is unchanged.

The parts' sites satisfy (a) by item 2, and (b) by the bad sets of 12.4, which are defined
from the address alone (the part's own data and the PIOP prefix through `fs`).

**Candidates, tests, binding, witness.** Every certificate below the root fold is a
first-stage certificate `y` of Proposition 10.44: a prefix through `xi_f` passing every check
it determines, with a root weak opening `omega_y` of the six groups; all stages between `l_1`,
`l_2` and the root fold are plain, so the certificate sets there are this set.

* `l_1 = sd`: `Cand(y) = sigma_1(y)`, the source projections of the round-1 groups `G_main16`,
  `G_main4` in `omega_y`, decoding to `f_1 = Dec(sigma_1)`; `Good_{l_1}(y) = H(f_1)`, the caller's
  relation `H` bound by byte 3 of event 0, H-u or H-exact ((3.1) to (3.3) with that
  relation's proven bounds, spec 3.1 and 12.2), on every live record. Binding: two
  certificates with the same input at `sd` have the same round-1 commitments, so different
  projections give a kernel vector (Akita Corollary 10.25).
* `l_2 = X`: `Cand(y) = sigma(y) = Src(omega_y)`, all six groups, `f = Dec(sigma)`;
  `Good_{l_2}(y) = R_PIOP(f; xi_X(y))`, the conjunction of the Boolean-domain identities the
  PIOP proves from `X` on: lookup membership of every looked-up cell and the leaves (4.3), the
  digit reconstructions (4.1), (4.2) (and the bit-slack form (4.3) under H-exact), (6.1) with
  the range of `v`, Booleanity and (3.4), under H-exact the Booleanity zero-checks of `B_zhi`
  and `a_z` at `r^(2)` (Section 10.1), and (9.2) with the `M_t` of `sd` and the `Y_sigma` of
  `xi_X`. Binding: Corollary 10.25 (both rounds'
  commitments are in `xi_X`).
* `Wit(y) = f_1` on the original domains, i.e. `(z_i, u_i, v_i, b_i)` by (4.1) to (4.3).

All maps are polynomial time. Spec 12.2 gives `R_PIOP(f; xi_X) => A_1(f_1)` and `M^(sigma)
b_sigma(f_1) = Y_sigma` over GF(2) for every segment, where `A_1(f_1)` is the relation `H` without
(3.3) ((3.1), (3.2), `w_1' in [0, 44)`, Boolean `b`) and `M^(sigma) = M_3 M_2 M_1` (spec 9.2).
The tree size is `K = 2^2 Tree0_ck(sch)` (two application checkpoints; Aerie has one). The
structure depends on the batch geometry bound in event 0 (`N`, the profile, the tied-part
geometry through the plan digest, 6.4), as Akita Remark 10.19 permits; every admitted geometry
has `rho <= 2^-128`, so the bound holds with `R_f`, `R_coord`, `R_F` and the tree size
maximized over them.

### 12.3 The segment of `X` (PIOP and the evaluation batch)

For a fixed `sigma` put `D(xi; sigma) = 1` iff a check determined by `xi` fails or a pending
claim is false, with the pending claims of Lemma S (Section 5), starting from `R_PIOP(f;
xi_X)` at `X`. Item (vii)(a) holds by definition. Item (vii)(b) is Lemma S for the PIOP
sites; at Akita's evaluation batch a false claim `v_k != f_k(z_k)` makes the batched equation
false outside one hyperplane of the 38 or 39 independent coefficients (H-u or H-exact; mass
`1/|K|`, Akita Remark 10.43; assigned `38/|K|` or `39/|K|`, `g = 6` either way). Item (vii)(c): at `xi_f` the only pending claim is the root
row `sum_k eta_k v_k = sum_k eta_k f_k(z_k)` (Akita (202)), which a certificate's weak opening
satisfies (Proposition 10.44, witness step), as in Aerie Lemma C.1 (ii).

### 12.4 The segment of `sd` (the bridge seed, the tie sites and Flock)

Fix `sigma_1` and `f_1 = Dec(sigma_1)`. For a tied part `p` with records in the segments
`P^-1(p)`, let `C_s^p` be its sponge word (event 2, in the input of `sd`) and `C_u^p` its union
word (in its proof). For codewords `z in Lambda(C_s^p)`, `z' in Lambda(C_u^p)` (decoded to
witnesses):

* `Spg_p(z)`: the sponge registry's rows and Flock's witness contract hold (dummy rows zero,
  the constant pin); `Pub_p(z)`: the public region equals the verifier's table `(mu_i, c~_i,
  1, 0)` on live rows. By assumption (K) of 12.5, `Spg_p(z) and Pub_p(z)` give `c~_i =
  SHAKE256(mu_i || w1Encode(b'_i(z)))` for every live record `i` of `p`, `b'(z)` the input bits;
* `Uni_p(z')`: the union's rows hold (level-1 rows with `M_1` of `sd`, levels-2-3 rows with
  `M_3 M_2` and zero coefficient), so by (K) its outputs are `M_3 M_2 M_1` of its level-1
  inputs, per segment;
* `Tie_p(z, z')`: the level-1 inputs of `z'` are `b'(z)` (zero on padded positions), the
  parity copies equal the parities, the outputs equal the part's final-vector table `Y^_p`
  (its share, or the `S` vectors of a span);
* `Val_p(z, z') = Spg_p(z) and Pub_p(z) and Uni_p(z') and Tie_p(z, z')`; it implies
  `M^(sigma) b'_{p,sigma}(z) = Y^_{p,sigma}` for every segment `sigma` of `p`.

**Pending claims** at a stage input `xi` of the segment (each present once the data it names
is in `xi`):

* `C1`: `A_1(f_1)`;
* `C2_sigma` (between `sd` and the messages of event 6): there are `z_p in Lambda(C_s^p)`
  for the parts `p` of segment `sigma` with `Spg_p(z_p) and Pub_p(z_p)` for all `p` and
  `M^(sigma) (b_sigma(f_1) xor b'_sigma(z)) = 0`, `b'_sigma(z)` the segment's bits assembled
  from the parts;
* after `Y_sigma` and the shares (event 6, first in the linear order): `C3_sigma`:
  `M^(sigma) b_sigma(f_1) = Y_sigma`; and per part `C4_p`: some `z in Lambda(C_s^p)` and some
  union witness `z'` satisfy `Val_p(z, z')`, where `z'` ranges over `Lambda(C_u^p)` once the
  union word is in `xi` (sending it can only turn a false `C4_p` into a false one: every
  codeword of the list decodes to a union witness);
* after the tie site of `p`: `C4_p` is replaced by the pair claim `C5_p`: some `(z, z') in
  Lambda(C_s^p) x Lambda(C_u^p)` satisfies all tested identities at `rho` (tie 1 and tie 2 for
  both lane groups, the outputs against `Y^_p(rho)`, the public words against the table;
  write `V_p(z, z'; rho)`), and neither `z` fails the sponge instance nor `z'` the union in
  the sense of (F): the instance's registry rows and witness contract, its internal pending
  claims so far, and its packed-direct claims (`z(P_pub) = v_pub`, `z(P_s_h) = v_s_h`; the
  union's seven) once their values are in `xi`. The points are fixed in each instance's
  transcript by the tie binding (4.5).

`D(xi; sigma_1) = 1` iff a check determined by `xi` fails or a pending claim is false; at the
input of `sd` the one pending claim is `H(f_1)`. At `xi_X`, `D = 1` iff a check fails, `C1` is
false or some `C3_sigma` is false (every Flock part is verified before `X`, so its outcome is a
check determined by `xi_X`).

**Item (vii)(a)** holds by definition. **Item (vii)(c):** a certificate in `Y_X` passes the
test `R_PIOP(f; xi_X)`, which implies `C1` and every `C3_sigma` (12.2), and its prefix passes
every check, so `D(xi_X; sigma_1) = 0`. **Item (vii)(b)**, stage by stage. The two PIOP sites `sd`
and `fs` read the complete linear prefix, so their bad sets may use all of it. The part
sites (tie and Flock) read only their projected address, so their bad sets are defined from
the address alone, for every input (whether or not the part's claim is the false one), as
Lemma 12.3 (b) requires; the argument then shows that every other false claim persists for
any answer:

* `sd`. If `C1` is false it stays false. Otherwise (3.3) fails for a live record `i*` of a
  segment `sigma*`. Let `Bad_sd` be the set of seeds for which some tuple `(z_p)_p` over the
  lists of the parts of `sigma*` has `Spg_p(z_p) and Pub_p(z_p)` for all `p` and
  `M^(sigma*) delta(z) = 0`, `delta(z) = b_sigma*(f_1) xor b'_sigma*(z)`. For such a tuple,
  (K) gives `c~_i* = SHAKE256(mu_i* || w1Encode(b'_i*(z)))` while (3.3) fails for
  `b_i*(f_1)`, and w1Encode is the bit packing of spec 8, so `delta(z) != 0`. On the event of
  Lemma 12.4 at most a fraction `theta' = 2 theta` of the seeds kill a fixed nonzero `delta`,
  so `Pr[Bad_sd] <= L^(P_sigma*) theta'`: `L^2 theta' = 2^-133.38 / 2^-132.35 / 2^-131.38`
  (`Fast` / `Slim` / `Slim8`, two parts in `sigma*`), `L theta' = 2^-138.51 / 2^-138.00 /
  2^-137.51` with one part or a span. Only the first false segment is charged; the sums over
  segments of Section 3.5 are a conservative superset. Outside `Bad_sd`, `C2_sigma*` is false.
* `fs`. No bad set: `D` stays 1 for every answer and every message of event 6. If `C1` or a
  check fails, it persists. Otherwise `C2_sigma*` is false. If `M^(sigma*) b_sigma*(f_1) !=
  Y_sigma*`, `C3_sigma*` is false. Otherwise, if every part `p` of `sigma*` had `C4_p` with
  witnesses `(z_p, z'_p)`, then `M^(sigma*) b'_{p,sigma*}(z_p) = Y^_{p,sigma*}` for each `p`;
  the shares XOR to `Y_sigma*` (the verifier derives the last share from `Y_sigma*`; a span's
  table is `Y` itself), the parts' records partition the segment and `M^(sigma*)` is GF(2)
  linear, so `M^(sigma*) b'_sigma*(z) = Y_sigma* = M^(sigma*) b_sigma*(f_1)`, which makes
  `C2_sigma*` true. Hence some `C4_p` is false.
* The tie site of part `p` (address: the part's tie-challenger input, with `C_s^p`, `C_u^p`,
  `y_digest`, the geometry and the nonce). Its bad set is defined from the address alone:
  `Bad_tie_p` is the set of `rho` for which some pair `(z, z') in Lambda(C_s^p) x
  Lambda(C_u^p)` with `Spg_p(z) and Uni_p(z')` but not `(Pub_p(z) and Tie_p(z, z'))`
  satisfies all tested identities at `rho` (tie 1 for both lane groups, tie 2 for both lane
  groups, the outputs against `Y^_p(rho)`, the public words against the table). It is a
  function of `C_s^p`, `C_u^p` (hence the two lists), the part's table `Y^_p` (from
  `y_digest`, expanded by item 1), the part's public records `(mu_i, c~_i)` and the matrices
  `M_t` of `sd` (both in the PIOP prefix through `fs`, which the address encodes by item 2);
  it does not depend on `f_1`, on `C1`, on `Y_sigma`, on `C3_sigma` or on other parts. For
  such a pair the first false identity is a nonzero multilinear polynomial in its free
  coordinates, and all identities holding at `rho` forces it to vanish there; the union over
  the pairs is the numerator `nu_tie` of Section 3.6, and with the work bits (Akita Remark
  10.18, Proposition 10.3 under (F)(iv)) the address weighs at most `2^-g_tie nu_tie / 2^128
  <= 2^-128`. Persistence for an answer outside `Bad_tie_p`: a failed check, a false `C1`, a
  false `C2_sigma`, a false `C3_sigma` and a false claim of another part `q != p` (`C4_q` or
  `C5_q`) do not depend on `rho` and stay false for every answer. If `C4_p` is false, a pair
  with `Spg_p(z) and Uni_p(z')` and `V_p(z, z'; rho)` would have `Pub_p(z) and Tie_p(z,
  z')`, hence `Val_p`, which `C4_p` excludes; so every pair fails `Spg_p` or `Uni_p` (and
  then its codeword fails its instance from the first site on) or fails `V_p`, and `C5_p` is
  false. If `C4_p` is true, some other claim is false (`D = 1`) and persists.
* A Flock site `t` of the sponge instance of part `p` (the union's likewise): its bad set is
  Flock's set `B_t` of answers after which some codeword of `Lambda(C_s^p)` that fails the
  instance at the address stops failing ((F)(ii); a function of the address alone, for every
  input; weight at most `2^-128` with the scheduled work, K19). Every false claim other than
  `C5_p` (a check, `C1`, `C2_sigma`, `C3_sigma`, a claim of another part, whose instances
  this answer does not touch) persists for every answer. If `C5_p` is false and the answer avoids `B_t`,
  every pair still fails: a failed `V_p` depends only on `(z, z', rho)`, a failing `z` keeps
  failing, the union's codewords are untouched, and a message never revives a codeword
  ((F)(ii)). If both instances' verifiers accept, (F)(iii) gives codewords `z*`, `z'*` that do
  not fail, so their packed values are the sent ones, and the verifier's value checks
  (`v_in_h = eq v_s_h`, `v_top_h = v_y1_h`, `v_out = Y^_p(rho)`, `v_pub` = table) then give
  `V_p(z*, z'*; rho)`, i.e. `C5_p`. So a false `C5_p` at the end of the part makes a Flock
  verifier or a value check reject, which is a check determined by every later input.

This gives item (vii) for `l_1`, with Lemma 12.3's (a) and (b) for the part sites. The
masses are those of Sections 3.5 and 3.6; defining the part sites' bad sets from the address
alone changes none of them.

**Lemma 12.4 (uniform seed density).** Except with probability `delta_X < 2^-(2^110)` over
`H_X`, every nonzero `delta` in the level-1 input space of a segment (`J = 6 * 16 * 2^19 =
3 * 2^24` bits; smaller segments embed with zeros) has `M^(sigma)(sd) delta = 0` for at most a
fraction `theta' = 2 theta = 2^-143.64` of the 256-bit seeds.

*Proof.* As Aerie Lemma C.2. The expansion inputs `(sd, t, zeta)` are injectively framed, so
distinct seeds give independent matrix families, each with `Pr[M_3 M_2 M_1 delta = 0] <= theta =
2^-144.64` (spec 9.4; columns uniform `k_t`-subsets, which the derivation of spec 9.1 produces
exactly). For a fixed `delta` the number of killing seeds is a sum of `2^256` independent
indicators of mean at most `mu = 2^256 theta = 2^111.36`, so it exceeds `2 mu` with probability
at most `exp(-mu / 3)` (Chernoff). A union over the `2^(3 * 2^24)` vectors gives `delta_X <=
2^(3 * 2^24) exp(-2^111.36 / 3) < 2^-(2^110)`. The event holds for all `delta` at once,
including those that depend on `H_X`.

### 12.5 What is assumed (exactly)

* **(F) Flock's documented analysis, per list element.** For each instance of the vendored
  `flock-core` under `Fast`, `Slim` (or `Slim8`) with K19, in `Pi`, with level-0 word `C`:
  (i) there is, for every codeword `z in Lambda(C)` and every instance-local input, a bit "`z`
  fails" (a pending claim of the instance is false for `z`), which at the instance's first
  site is set iff `z` violates the registry rows or the witness contract, and which a
  packed-direct claim sets once its value is sent and differs from `z`'s evaluation; (ii)
  every Flock site `t` has a set `B_t`, determined by its address, of mass at most `eps_t`
  with `2^-g_t eps_t <= 2^-128`, such that outside `B_t` every failing codeword still fails at
  the next input, and no prover message unsets the bit:
  the outer sites with the level-0 list factor (Flock Appendix A, "Security", as patched by
  K19), the Ligerito sites (MCA, two-point OOD binding, claim and consistency batches, fold
  sum-checks, recursive commitments, query consistency with 16-bit query work) as validated by
  `LigeritoSecurityConfig::validate` for the shipped configs (Flock Appendix C.3, Section 4.2);
  (iii) if the instance's verifier accepts, some codeword of `Lambda(C)` does not fail at the
  end; (iv) Flock's work composition (predicate BLAKE3 of `state || LE64(nonce) || 0^24`, then the
  nonce absorbed and the challenge squeezed) meets Proposition 10.3: distinct contexts give
  distinct, independent predicate/challenge outputs. This is Flock's own 128-bit claim in the
  per-element form that its union bounds over the list prove (Flock Appendix C.3 states the
  invariant for every list element and multiplies each step by `L_max`); the Flock paper
  presents the analysis as a sketch, and we do not re-derive it.
* **(K) Circuit correctness.** Every assignment satisfying the sponge gate's rows (`docs/
  keccak.md` Sections 1, 2: each chi output is forced by its own row, the linear layers, the
  absorption and the chain are XOR expressions) computes SHAKE256 on its input bits; every
  assignment satisfying the bridge union's rows computes `M_1` per block and `M_3 M_2` on the
  copies with the part's column offsets, i.e. `M^(sigma)` of the part's bits zero-extended to
  its segment (`docs/bridge.md` 5b). A property of fixed circuits, established by
  construction and by the crates' tests, not proved here.
* **(R) Random oracles** as listed in 12.1 (Aerie's modeling of Spongefish, Flock's of BLAKE3,
  Akita's ideal XOF; the BLAKE3 uses for Merkle nodes, challenger chaining and work
  predicates are independent oracles by their input formats, as Flock assumes), the setup as
  in Aerie C.1 (`Delta_setup = 0` for the exact rejection-
  sampled setup stream of the pinned Akita revision, which Aerie `49b6c21` shares with us).
* **(A) Akita** at `e023b6a`: Proposition 10.44, Lemma 10.42, Corollaries 10.25 and 10.32,
  Lemmas 10.2 and 10.20, the extraction-admissibility of our schedules (Definition 9.2 (i) to
  (iii); the rows pass Akita's schedule audit when the catalogs load, the tree-size condition
  is not tabulated here, see 12.7), and Theorem 10.17, here with Lemma 12.3.

Discharged in this repository: the PIOP round-by-round bad sets (Lemma S), the `sd`, tie and
evaluation-batch bad sets above, the work schedule (Sections 3, 6, 10), the fold-challenge
supports (4.6), Lemmas 12.1 to 12.4.

### 12.6 The bound

Let `P` make `Q` queries (every oracle of 12.1, nonce probes included; `Q_wrap = Q + V_ver`
after the conversions of Lemmas 10.47, 12.1, 12.2, as Aerie C.5). Theorem 10.17 with Lemma
12.3, the structure of 12.2 to 12.4, `Good` the event of Lemma 12.4, and the setup,
extraction and Module-SIS steps of the proof of Akita Theorem 10.45 give an extractor that runs
`P` at most `(Q_wrap + 2) 4 Tree0_ck(sch)` times in expectation and fails on an accepting
execution with probability at most

```text
eps_hash + eps_fs + eps_merkle + delta_X + E[ sum_{a in Addr} eps(a) ] + sum_I Adv^MSIS_I(B_I)
```

with `E[sum eps(a)] <= rho (Q + R_f + R_coord + R_F)` (Aerie Eq. (18)) and `rho = max eps(a)
<= 2^-128`: PIOP sites `<= 2^-128.09` (Sections 10 and 10.1, max weight, both relations), tie sites `<= 2^-128` (3.6), `sd <=
2^-131.38`, `fs` 0, Flock sites `<= 2^-128` under (F), Akita field sites by its plan, fold
coordinates `<= 2^-128.088` (4.6). This is the bound of Section 7.

**Corollary 12.5 (mode B).** Fix `N <= 2^15`, mode B (Flock profile id 3, deviation P9:
the `w1Encode` strings of each segment's live records are the event-2 items, absorbed before
`sd`; the event-6 proof items are empty), PIOP wire version 3 with the plan of Section 6
(profile id 3, no tie site) and Akita's plan for the selected row. Then the statement of
Section 7 holds for mode B without assumption (F) and without the terms `eps_fs`,
`eps_merkle` and `R_F`, with the same extractor bound otherwise. *Proof.* The structure of
12.2 with the same checkpoints at `sd` and `X`, and no tie or Flock sites (items 2 and 3 of
12.1 are void). The public bits `b'` are fixed before `sd`, so the "lists" have one
element. The verifier checks natively that every 6-bit value is below 44, that `c~_i =
SHAKE256(mu_i || w1Encode(b'_i))` for every live record and that `Y_sigma = M^(sigma) b'_sigma`
(P9); these are checks determined by the input of `X`. At `sd`, if `C1` holds but (3.3) fails
for a live record `i*` of segment `sigma*` under `b(f_1)`, then `delta = b_sigma*(f_1) xor
b'_sigma*` is nonzero (the native check holds for `b'`), and `Bad_sd = {sd : M^(sigma*)(sd)
delta = 0}` has mass at most `theta' = 2^-143.64` on the event of Lemma 12.4 (one segment
charged, the first false one). Outside it, `M^(sigma*) b_sigma*(f_1) != Y_sigma*` once
`Y_sigma*` is sent, or the native projection check fails, so `C3_sigma*` is false or a check
fails, and `D(xi_X; sigma_1) = 1` with no site in between (`fs` has no bad set). The segment
of `X` is 12.3 unchanged; the bit check `b_t(r^(3)) = b'_t(r^(3))` after stage 3 (P9) is an
additional verifier check and is not used in this bound (the seeded matrices bind on their
own; the check is P9's matrix-independent second binding).
Assumption (K) is not needed for the sponge (SHAKE256 is computed natively) or for the
union. Every address weighs at most `2^-128` (PIOP sites as in Section 10, `sd` at most
`2^-143.64`, Akita by its plan and 4.6), so the bound of 12.6 holds with `R_F = 0`.

### 12.7 What is not proved here

1. Assumption (F): the round-by-round soundness of list-decoded Ligerito with Flock's merged
   opening and packed-direct claims, with the per-site errors that Flock's configs validate.
   The Flock paper sketches it (Appendices A and C.3); a complete proof is not available to us.
   The paper MUST present the theorem as conditional on (F).
2. Assumption (K), the correctness of the two fixed circuits.
3. Lemma 12.3 extends Akita's Theorem 10.17; its proof above modifies Claims 1 and 2 of
   Akita Appendix E.2 and SHOULD be checked by a reviewer of Akita's proof.
4. `R_F` is not counted exactly (open 4); the `Q = 0` term stays about `2^-117` (estimate).
5. The extractor's run count `(Q_wrap + 2) 4 Tree0_ck(sch)` is not tabulated per row (Aerie
   C.4 tabulates its `K`); it bounds the running time only, not the error.
6. The classical ROM only, as Aerie and Akita.
7. The CMA-to-NMA simulation step for FIPS 204's exact algorithms (12.8, Lemma 12.8 case
   (i)): the published corrected proofs cover round-3 Dilithium (UF-CMA) and generic
   Fiat-Shamir-with-aborts signatures (sUF-CMA), not ML-DSA as standardized (spec 12.3 (b)).
8. The min-entropy `H_inf` of a fresh signature's `c~` (Lemma 12.7) is evaluated only
   heuristically, in the ROM for the signer's hash calls (about 253.9 bits, estimate).

### 12.8 Aggregate unforgeability: signing queries in rewound runs (spec 12.3)

Spec 12.3 (revision 6) reduced an aggregate forger to a UF-CMA forger against the relaxed
scheme and left one obligation open: a rewound run may ask the signing oracle for
`M_i*`, and the output is then no UF-CMA forgery. ML-DSA has no salt (`mu = H(tr || M')`,
FIPS 204 Algorithm 7 line 6), so Aerie's presampled signature/target lists (Aerie C.6) do
not apply. This section closes the obligation with strong unforgeability (SUF-CMA) of the
relaxed scheme, lazy forwarding and no presampling (spec 12.3 steps 1' and 2'). Notation
as in spec 12.3: one honest key `pk*` (chosen-key model, Aerie Definition 5.10), the
relaxed scheme `Sigma_H` of the caller's relation (H-u or H-exact) whose signing returns
the FIPS 204 signature `sigma = (c~, z, h)` and its relaxed image `phi(sigma) = (c~, z,
w1', u, v)` (spec 3.2), validity by (3.1) to (3.3) with the relation's proven bounds, and
the facts (U) (`(u, v)` is a function of `(pk, M, c~, z, w1')`, since `B_u < (q - 1)/2`)
and (E) (a relaxed signature has no encoding). `Q_S` bounds the signing queries of the
aggregate adversary `A` per run.

**Lemma 12.6 (lazy forwarding is a fixed tape).** Let `B` answer the signing queries of
`A` from a table `T[(M, k)]`, where `k` is the occurrence index of `M` among the signing
queries of the current run (a counter that is part of the prover state and is restored
with it on every rewind); `T` is never rolled back, and an entry is filled on first use
by one query to `B`'s SUF-CMA challenger. Then (a) in every run, `A`'s view of the
signing oracle has the law of the real game; (b) for each value of `T` the composed
prover `P_T` (`A` with these answers, followed by the aggregate verifier) is a fixed
prover with auxiliary input `T` in the sense of Akita 10.3, and `T` is independent of
the setup and of the proof oracles of 12.1; so the theorem of Section 7 applies to `P_T`
with the same `eps_ext` (`Q` counts `P_T`'s proof-oracle queries), and its extractor
returns the original execution.

*Proof.* Sample `T` up front: independent signatures `T[(M, k)] <- Sign(sk*, M)` for the
hedged signer (FIPS 204 Algorithm 2 line 5, a fresh `rnd` per entry), and `T[(M, k)] =
Sign(sk*, M; {0}^32)` for every `k` for the deterministic signer. Filling entries on first
use gives the same joint law, because each entry is read only after it is set and the
entries are independent of each other and of the proof oracles: the challenger's signer
uses `sk*` and its own randomness, and its SHAKE256/SHAKE128 calls are not proof-oracle
queries (assumption (R) of 12.5; the transcript, bridge and record-binding hashes use their
own framings and cSHAKE customizations). (a) Within one run the `k`-th query on `M`
receives `T[(M, k)]`: a fresh independent signature for each `k` (hedged) or one fixed
signature (deterministic), as the real signer. (b) With `T` fixed, `P_T` is determined by
its random tape and its oracle answers; a rerun from a saved state restores the counters,
asks the same signing queries and reads the same entries while its oracle answers agree.
This is the case Akita's extraction experiment names (Section 10.3, the paragraph on
auxiliary environments before Definition 10.15): "an auxiliary algorithm whose answers
depend on its query history, such as a signing simulator, is part of the prover: its
initial state is fixed with the prover's tape and restored at every rerun"; here the tape
includes `T`. It is the shape of Aerie Lemma C.5 (i) and (iii). Sections 7 and 12 use no
property of the prover other than its query count, so the bound holds for every `T` and
hence on average.

**Lemma 12.7 (rerun-only answers miss the extracted signature).** In the setting of
Lemma 12.6, let the original execution output an accepting aggregate with a record `i*`,
`pk_i* = pk*`, `M_i*` not queried in the original run, and let the extractor return the
witness of 12.2, read as the relaxed signature `sigma* = (c~_i*, z_i*, w1'_i*, u_i*,
v_i*)`. Let `R` be the number of entries `T[(M_i*, k)]` created during the extraction
(the *rerun-only* queries on `M_i*`). Then:

1. every such entry is created after the original execution has ended, hence after
   `c~_i*` is fixed;
2. `sigma*` is fixed by events 0 and 1 of the original transcript before the first fork:
   `c~_i*` is a public input bound in event 0, and `z, u, v, b` are the round-1 tables of
   `G_main16`, `G_main4` (spec 4.1), fixed by the round-1 commitments up to Akita binding;
3. `Pr[sigma* = phi(T[(M_i*, k)]) for some k] <= E[R] 2^-H_inf` for the hedged signer,
   with `2^-H_inf = max_{sk, M, x} Pr_rnd[c~(Sign(sk, M; rnd)) = x]`, and at most
   `Adv_PRF + Adv_coll(mu) + E[R] 2^-H'_inf` for the deterministic signer, with `H'_inf`
   the same min-entropy over a uniform 64-byte `rho''`;
4. `E[R] <= Q_S (Q_wrap + 2) 4 Tree0_ck(sch)`.

*Proof.* (1) The original run asks no signing query on `M_i*` (freshness of its output), so
it creates no entry `(M_i*, k)`; the extractor starts from the handle of the completed
original execution (Akita Definition 10.15: the handle fixes the tape, the initial state
and the oracle transcript) and only then reruns `P_T`. (2) Every rerun restarts `P_T` from
its initial state with a challenge table that agrees with the original one (or with its
parent's, which agrees with the original up to the parent's fork) up to the fork query, the
first query at the input of an extraction or checkpoint stage of 12.2: `sd`, `X` or a stage
of Akita's opening suffix. Every such input encodes event 0 (the public records through the
public-record digest, expanded by item 1 of 12.1) and event 1. The certificates combined at
`sd` share the round-1 commitments, so their round-1 projections agree except with an Akita
binding break (Corollary 10.25, inside `eps_ext`), and `Wit(y) = f_1` (12.2) is these
projections. (3) `sigma* = phi(sigma)` implies equal `c~`, so it suffices that no
rerun-only answer has commitment hash `c~_i*`. *Hedged:* index the rerun-only entries in
creation order. The decision to create entry `j` is made before its `rnd_j` is drawn, and
`c~_i*` is fixed by (1), so `Pr[c~(entry j) = c~_i* | j <= R, history] <= 2^-H_inf`;
summing, `sum_j Pr[j <= R] 2^-H_inf = E[R] 2^-H_inf`. *Deterministic:* every entry on
`M_i*` is `Sign(sk*, M_i*; {0}^32)`, a function of `sk*` and `rho'' = H(K || {0}^32 ||
mu_i*, 64)` (Algorithm 7 line 7), where `K` is the 32-byte secret seed of `sk*` (Algorithm
6 line 1) and is used nowhere else. Assume `x -> SHAKE256(K || {0}^32 || x, 64)` is a PRF
under `K` (in the ROM its advantage is at most about `Q_H 2^-256`). A distinguisher runs
the whole experiment with its oracle in place of this function and outputs the event; it
costs `Adv_PRF`. With a random function `F` in its place, the original run sees `F` only at
the `mu_j` of queried messages; `mu_j = mu_i*` with `M_j != M_i*` is a collision of the
64-byte SHAKE256 output at Algorithm 7 line 6 (`Adv_coll(mu)`). Otherwise `F(mu_i*)` is
uniform and independent of `c~_i*` and of the event `R >= 1`, which gives `Pr[R >= 1]
2^-H'_inf <= E[R] 2^-H'_inf`. (4) Each run asks at most `Q_S` signing queries, and the
extractor runs `P_T` at most `(Q_wrap + 2) 4 Tree0_ck(sch)` times in expectation (12.6).

*Evaluation (estimate, not a proof).* `H_inf <= 256`, since `rnd` and `c~` have 32 bytes.
In the ROM for the signer's hash calls (Algorithm 7 lines 7, 11 and 15), an answer's `c~`
is a fresh oracle output unless a hash query of `A` starts with `K` (at most about `Q_H
2^-256` over the whole game) or an iteration's `w1` equals the `w1` of an earlier hash
query on `mu_i*` (the commitment min-entropy term of the FSwA analyses, KLS18 and Devevey
et al.); otherwise each of the expected 4.25 iterations (FIPS 204 Table 1, "Repetitions")
hits `c~_i*` with probability `2^-256`, about `2^-253.9` per answer. The term `E[R]
2^-H_inf` grows with the extractor's run count, as `B`'s running time and the SUF-CMA
advantage it is measured against do (the same accounting as the Module-SIS terms of
`eps_ext`, whose estimates exclude the run count as in Aerie, Section 7): it is about
`2^-253.9` per signature the reduction obtains, far below `2^-128` per unit of its work.

**Lemma 12.8 (SUF-CMA of the relaxed scheme).** Model H (SHAKE256 at FIPS 204 Algorithm 7
lines 6 and 15 and Algorithm 8 line 12) and ExpandA as random oracles. For every SUF-CMA
adversary `B'` against `Sigma_H` there are reductions with

```text
Adv^SUF-CMA_{Sigma_H}(B') <= Adv^MLWE + Adv^SelfTargetMSIS_{zeta'} + eps_sim
                             + Adv^coll_{SHAKE256/256} + Adv^MSIS_{[A | I], 380,930},
```

where `zeta' = 350,209` (spec (3.5), both relations), `eps_sim` collects the simulation
terms of the CMA-to-NMA step (below) and `Adv^coll_{SHAKE256/256} <= q_H^2 / 2^257` in the
ROM; the Module-SIS instance is `[A | I]` over `Z_q`, `256 k = 1,024` rows, `256 (k + l) =
2,048` columns, `l_inf` bound 380,930.

*Proof.* Let `(M*, sigma*)` win, `sigma* = (c~*, z*, w1'*, u*, v*)`, and let `S` be the
answers on `M*`. One of three cases holds.

* (i) `c~*` differs from the `c~` of every answer in `S` (always when `M*` was never
  queried). Then the hash input `mu* || w1Encode(w1'*)` is not the input of any returned
  answer: for an answer in `S` the outputs differ, and an answer on `M_j != M*` has
  `mu_j != mu*` unless SHAKE256 collides at line 6. The UF-CMA argument of the Dilithium
  and FIPS 204 analyses applies with the relaxed verification: replace `t` by uniform
  (MLWE), simulate signatures by programming H at `mu || w1Encode(w1)`, and output
  `(z*, -c*, c* t0 - u*)`, a SelfTargetMSIS solution with bound `zeta'` (spec 3.3). The
  CMA-to-NMA step is the one corrected by Barbosa et al. (CRYPTO 2023), with the
  strong-unforgeability bookkeeping of Devevey et al. (CRYPTO 2023; ePrint 2023/245,
  Theorems 9 and 11): the forgery's input can equal that of a rejected, never revealed
  iteration only with the commitment min-entropy term. These are `eps_sim`.
* (ii) `c~* = c~_j` for some answer `j` in `S` and `w1'* != w1'_j`. Then
  `H(mu* || w1Encode(w1'*), 32) = H(mu* || w1Encode(w1'_j), 32)` on distinct inputs
  (w1Encode, Algorithm 28, packs each coefficient of `[0, 44)` in 6 bits and is injective):
  a SHAKE256 collision at the 256-bit output whose collision strength FIPS 204 Table 1
  gives as `lambda = 128`.
* (iii) `c~* = c~_j` and `w1'* = w1'_j` for some answer `j` in `S`. Then `c* = c_j` and
  `T* = T_j` (same key), and (3.1) for both gives `sum_s A[., s] (*) dz[s] - du - q dv = 0`
  over `Z` for the differences. If `dz = 0`, then `du = -q dv` with `|du| <= 2 B_u < q`,
  so `du = dv = 0` and `sigma* = phi(sigma_j)` (this is (U)), which is excluded. So
  `dz != 0` and `[A | -I] (dz, du) = 0 mod q` with `|dz|_inf <= 2^17 + B_z = 262,065`
  (H-u: `z* in [-2^17, 2^17)`, honest `|z_j|_inf <= B_z`; H-exact: `2 B_z = 261,986`) and
  `|du|_inf <= 2 B_u = 380,930`: a nonzero Module-SIS solution for `[A | I]` with `A` the
  challenge matrix (programmed into ExpandA at `rho`). The reduction samples `(s1, s2)`,
  publishes `t1`, signs honestly and needs no MLWE step. This is the computational
  unique-response property of the relaxed identification scheme (Devevey et al.,
  Definition 6); case (i) is its no-message part.

Under H-exact every bound is the honest one or smaller, so the same instance applies.

**Theorem 12.9 (aggregate unforgeability, classical ROM).** Under the hypotheses of the
theorem of Section 7 (mode A, assumptions (F) and (K)), for every aggregate adversary `A`
in the chosen-key game of spec 12.3 that makes `Q` proof-oracle queries and `Q_S` signing
queries, there is a SUF-CMA adversary `B` against `Sigma_H` whose running time and
signing-query count are those of the extractor (at most `(Q_wrap + 2) 4 Tree0_ck(sch)`
runs of `A` in expectation, each with `Q_S` signing queries) such that

```text
Adv^agg(A) <= Adv^SUF-CMA_{Sigma_H}(B) + eps_ext + E[R] 2^-H_inf          (hedged signer)
Adv^agg(A) <= Adv^SUF-CMA_{Sigma_H}(B) + eps_ext + E[R] 2^-H'_inf
              + Adv^PRF + Adv^coll(mu)                                   (deterministic signer)
```

with `eps_ext` the bound of Section 7 and `R` as in Lemma 12.7; Lemma 12.8 bounds the
first term. There is no `q_H` factor, no `e(Q_S + 1)` factor, no presampling and no
programmed hash in step 1'. *Proof.* Lemma 12.6 makes the theorem of Section 7 apply to
`P_T`: except with `eps_ext` the extractor returns the original execution and, since the
verifier accepted, a witness of the relation for every live record; `B` outputs
`(M_i*, sigma*)` for the first fresh record `i*` with `pk_i* = pk*`. The witness satisfies
(3.1) to (3.3) with the verifier's `c~_i*`, `T_i*` and `mu_i*` computed from `pk*` and
`M_i*` (spec 2), so `sigma*` is a valid relaxed signature on `M_i*`. It is a strong
forgery unless it equals `phi` of a stored answer on `M_i*`, and Lemma 12.7 bounds that
event.

*Hardness of the two lattice instances* (`ref/notes/msis-estimate.md`, measured with
lattice-estimator `f31b86f` and pq-crystals security-estimates `75c2694`):
SelfTargetMSIS at `zeta = 350,209`, block size 423, core-SVP `2^123.5` (rough) /
`2^152.2` (default model), 123 / 112 (classical / quantum); the case-(iii) instance,
block size 417, core-SVP `2^121.8` / `2^149.3`, 121 / 110. The second is the instance of
the Dilithium round-3 strong-unforgeability row (that script's `max(2 gamma1, 4 gamma2 +
1) = 380,929` gives the same block size). Unforgeability therefore rests on `2^121.8`
core-SVP, the basis of ML-DSA-44's own SUF claim; MLWE stays at 124 / 123.

*Mode C* (private `c~_i`, not specified here). Lemma 12.7 (1) and (2) need every candidate
value of `c~_i*` fixed before the first fork, so a mode-C design MUST commit `c~` in event
1 or 2. With a Ligerito commitment the candidates are the values in the codewords of its
Johnson list (at most `L`, fixed by the Merkle expansion in the original run, Lemma 12.2),
and the term becomes `L E[R] 2^-H_inf`.

*Why not presampling* (for the record; do not re-propose). One presampled signature per
hash entry cannot answer a message signed twice under the hedged signer; a rerun branch
may receive a signature the adversary never saw; guessing the deterministic signer's
`q_H`-th hash query multiplies `eps_ext` by `q_H`; `tr = H(pk)` is public, so keyed-sponge
results do not make `mu` pseudorandom; and SHAKE256 serves `tr`, `mu`, `rho''` and `c~`
without domain separation (Aerie C.7 notes the same overlap). The salt-free case appears
in Aardal et al. (Lemma C.3) and Aerie C.7 through a partition argument with loss
`e(Q_S + 1)`; SUF removes that loss and the cached-signer restriction, at the price of the
`[A | I]` instance above (`2^121.8` instead of `2^123.5` core-SVP).
