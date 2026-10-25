# Soundness: site census and the proof-of-work schedule (normative)

| Field | Value |
| --- | --- |
| Status | normative design for the 128-bit schedule (plan `ref/notes/plan-128bit-speed.md` Part 1, step 1); implemented in step 2 (branch `pow-128`): PIOP wire version 2 (`agg_piop::pow`, `agg_piop::transcript`), tied-part proof version 4 (`agg_keccak::tied`, deviation K20), the list-aware Flock outer sites (`vendor/flock-core/PATCH.md` Section 3, deviation K19), the Akita plan pinned (`agg-pcs-config/tests/akita_grinding.rs`) |
| Applies to | mode A and mode B, ML-DSA-44, relation H, lookup layout L16; PIOP wire version 2 and tied-part proof version 4 (both introduced by this document) |
| Base | code `be398d8`; Aerie `49b6c21`; Akita `e023b6a46143f7b23199ccecb2684192baf7b640`; Flock `83f963e` with the vendored `flock-core` (K12, K18) |
| Script | `docs/scripts/pow_plan.py` (run with `python3 -I`); its output summary is Section 10 |
| Replaces | the per-instance union bound of `docs/spec.md` 12.1 (Section 5), deviation P3 once implemented |
| Last reviewed | 2026-10-25, first version; implementation notes added (Sections 4.1, 4.3, 11) |

The uppercase terms MUST, MUST NOT, SHOULD and MAY are normative as in BCP 14. Every
number below that is not marked "estimate" is computed exactly by the script or read from
the pinned code; estimates are marked.

## 0. Results in one page

1. **PIOP sites.** At `N* = 2^15` the transcript draws 484 field challenges in `K`
   (`|K| = p^2`, `p = 2^64 - 59`): the lookup vector `X = (X_1, X_2)`, 27 merges `mu_s`,
   26 GKR batching vectors, 351 GKR rounds of degree 3, `tau`, `gamma^(1..3)` and
   `3 x 25` stage rounds of degrees 4, 5, 3. With the assigned bounds of Section 3 every
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
   Nothing needs to be turned on; `status.md`'s "no grinding plan" is wrong for the
   opening and right only for the PIOP.
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
   union-bound ledger `1,434/|K| = 2^-117.5` at `2^15` instead of spec 12.1's
   `3,534/|K| = 2^-116.2`: reductions, zero-checks and linear identities ride on the stage
   sum-checks and on one `tau` site (Section 5). With the schedule every address weighs
   at most `2^-128`, and the theorem of Section 7 follows Aerie Eq. (18).

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
`t`, fix the objects the extractor will return (the 38 Akita polynomials `f`, the Flock
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
| `stage.gamma` (6) | 3 | 1 | `26, 8, 44 / |K|` | 5, 4, 6 | 12, 11, 13 | |
| `stage.round` (7) | `3 d_main` | 4, 5, 3 | `deg/|K|` | 3, 3, 2 | 10, 10, 9 | 75 sites |

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
* `gamma^(s)` (26, 8, 44 coefficients): `nu = 1` as in 3.3.
* stage rounds: `nu` = the round polynomial's degree, 4, 5, 3 (`STAGE_DEGREES`; the `eq`
  factor is included: `range_w1` has degree 4, `leaf8` degree 5, `link` degree 3).
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
  (spec 9.4), mass `theta = 2^-144.64`. Because `b'` is fixed only up to the sponge
  commitment's Johnson list (Section 3.6), the bad set is the union over that list:
  `L_s theta = 2^-139.51 / 2^-139.00 / 2^-138.51` (`Fast` / `Slim` / `Slim8`). Below
  `2^-128`: no nonce (Aerie: "the projection seed requires no work").
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

* `nu_tie` (round-by-round, first false identity) `= max(L_s L_u (d + span_log + 5),
  L_u (log2 blocks + span_log + 9), L_u (5 + span_log), L_s (nu_s + 3))`.
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
  `d_sig` 3 to `4f5a6026...` at 15). `ref/notes/status.md` (top-level repository) still says
  the opening has no grinding plan; it is outside the code repository.

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

Implemented in step 2 (`vendor/flock-core/PATCH.md` Section 3, deviation K19): the policies
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

## 5. Tighter accounting (round-by-round), with the argument

**Lemma S (checks of fixed objects).** Fix the 38 Akita polynomials `f` (values in `F_p`
on their Boolean domains) and the Flock word vectors up to their level-0 lists, and a
statement. For every challenge input `u` of a PIOP site from `sd` to the PCS export or of
a tie site, let
`D(u) = 1` iff a verifier check determined by `u` fails or a pending claim is false,
where the pending claims are: at `sd`, "`(f, z)` is a witness of relation H for the
statement" (with `b = b'`); after the Keccak side, the relation identities of `f`
(Sections 4 to 6, (9.2) with the absorbed `Y_sigma`); after each site, the claims listed
in Section 3 (layer claims, stack and table leaf claims, stage input claims, terminal
evaluation claims). Then for every site `t` of Section 3 and every input `u` with
`D(u) = 1`, there is a set `Bad_t(u)`, determined by `u` and the fixed objects, such that
an answer outside `Bad_t(u)` leaves `D = 1` at every next input, and
`Pr[Bad_t(u)] <= 2^-g_t nu_t / |F_t|` including the predicate (Akita Remark 10.18).

*Proof sketch.* Choose the first false pending claim and define `Bad_t` by it alone;
Sections 3.2 to 3.6 give the polynomial whose zero set it is and its degree. Every claim
not acted on by the site stays false. A false claim never needs two sites at once: a
batching vector turns any false input claim of a stage into one false batched claim, the
stage rounds then act on that single claim, and the terminal check is exact. This is
Aerie Lemma C.1 with our instances; at `sd` the bad set is Section 3.5's, at the tie site
Section 3.6's (with the list factors), at `X` Section 3.2's.

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
`2^-139`) is `(12 d_main + d_main + 3 + GKR)/|K|`:
`999/|K| = 2^-118.0` at `2^10`, `1,251/|K| = 2^-117.7` at `2^13`, `1,434/|K| = 2^-117.5`
at `2^15`, against spec 12.1's `2,679`, `3,183`, `3,534` over `|K|`. This is still not
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
   the joint experiment of Akita Proposition 10.3 without a separate address. The address
   is unique per site because the registry fixes the event sequence and event 0 binds the
   plan digest; no extra label is absorbed (as Aerie, registry metadata is not absorbed).
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
`>= 2^(g_tie + 7)`, the prover aborts). The nonce travels in the tied-part proof header as
`LE32`; the verifier rejects a value `>= 2^(g_tie + 7)`.

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
`(1 - 2^-g)^(2^(g + 7)) <= exp(-2^7) = 2^-184.7`; with at most 488 sites per proof the
completeness error is below `2^-175.7`. Expected work per site `2^g` predicate
evaluations.

## 7. Theorem (128-bit knowledge soundness, classical ROM)

**Theorem.** Fix `N <= 2^15`, mode A with Flock profile `Fast` or `Slim` with the fix of
4.3 (or `Slim8` with 4.3 and 4.4), tied-part proof version 4 (4.5), PIOP wire version 2
with the plan of Section 6, and Akita's plan for the selected catalog row. For every
prover `P` making at most `Q` oracle queries (all oracles of Section 1, every nonce probe
included) there is an extractor of expected cost polynomial in that of `P` (Aerie
Theorem 5.9: at most `(Q_wrap + 2) K_hat` runs) that returns the original execution and,
whenever the verifier accepts, a witness of relation H for every live record, except with
probability at most

```text
eps_ext <= (Q + R_f + R_coord + R_F) 2^-128 + eps_bind + eps_pk + eps_dig + delta_samp
           + eps_merkle + sum_I Adv^MSIS_I(T_I)
```

where `R_f` is the number of field sites of the PIOP and of Akita read by the verifier
(484 + 459 at `2^15`, plus the seeds and the tie sites), `R_coord` Akita's sparse fold coordinates read by
the verifier, `R_F` the Flock sites read by the verifier (both instances, outer and
Ligerito, queries included), `eps_bind, eps_pk, eps_dig` the hashed-entry terms of Aerie
Definition 5.8 (instance digest, public-record digest, PCS session digest, Flock seeds,
span seed, tie digests), `delta_samp = 0` for the exact samplers, `eps_merkle` the
collision term of the Ligerito Merkle trees and BLAKE3 transcript hashing, and the
Module-SIS terms those of Akita Lemma 10.20 for the schedule's occurrences.

*Proof sketch.* (1) Hashed entries replaced by their preimages (Akita Lemma 10.47):
`eps_bind + eps_pk + eps_dig`. (2) Ligerito commitments are extracted straight-line from
the Merkle queries (up to `eps_merkle`); the level-0 lists are fixed when the caps are
output (events 2 and 6), before every site that refers to them. (3) Akita's Theorem 10.45
with the application segment starting at the checkpoint `sd`, whose candidate is the
round-1 polynomials and the sponge list, with Lemma S supplying the bits and bad sets of
Definition 10.16 (vii) at every PIOP and tie site; Flock's validated per-site bounds
supply them inside the Keccak side; Akita's plan inside the opening. (4) Every address
weighs at most `2^-128` (Section 3, the script's assertion `max weight <= 2^-128`; Flock
with 4.3/4.4; Akita by its plan and its 128-bit sparse-challenge floor), so Aerie
Eq. (18) gives `rho (Q + R_f + R_coord + R_F)` with `rho <= 2^-128`. (5) The extracted
objects satisfy relation H by spec 12.2 (lookup, leaves, bridge with `b = b'`, Keccak by
Flock's relation).

*What is not proved here.* Step (3)'s composition with a second, straight-line extractor
inside the application segment (the Flock vectors) is the obligation of step 3 (review)
and of the paper; Flock's own MCA, OOD and query bounds are taken from Flock; Akita's
Module-SIS estimates exclude the extractor's run count as in Aerie. As in Aerie, the term
`(R_f + R_coord + R_F) 2^-128` is not negligible at `Q = 0` (about `2^-117` to `2^-118`,
estimate); the claim is 128 bits per query, Aerie's notion.

## 8. Costs at `N = 2^15` (default: `Slim`, span on)

| Item | Proof bytes | Prover hashes (expected) |
| --- | ---: | ---: |
| PIOP packed stream (483 nonces, 4,444 bits) | 556 | 2,718 Blake2b predicates (about 2 ms serial, estimate) |
| Tie nonce (`g = 17`) | 4 | `2^17` BLAKE3 (about 0.5 ms on 7 threads, estimate from the measured 0.22 ms at 16 bits and 2.9 ms at 20 bits) |
| Akita (already in the 62,974-B opening) | 0 added (938 max today) | 0 added (9,872 today) |
| Flock fix of 4.3 | 0 (nonces are 8-byte words already) | about 100 sites per instance at `2^7` to `2^14` (estimate, milliseconds) |
| Plan digest in event 0 | 0 (absorbed, not sent) | one SHA3-256 |
| **Total added** | **560 B = 0.017 B/sig** (51.0 B/sig unchanged at one decimal) | well under 0.1 s (estimate) against 23.7 s |

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

PIOP sites per `N*` (`bits` = packed-stream bits, `work` = `sum 2^g`, ledger = round-by-round
sum before work):

```text
d_sig  N*      d_main  D   L            X-site log2   sites  nonces  bits  bytes  work(sum 2^g)  est.ms  max weight  ledger(before work)
    3       8      13  16       167936        -220.3    195     194  1811    227           1364    0.99  2^-128.30  563/|K| = 2^-118.9
    4      16      14  16       200704        -220.0    198     197  1841    231           1400    1.01  2^-128.19  576/|K| = 2^-118.8
    5      32      15  17       266240        -219.3    219     218  2032    254           1494    1.08  2^-128.09  639/|K| = 2^-118.7
    6      64      16  18       397312        -218.4    241     240  2233    280           1608    1.16  2^-128.42  705/|K| = 2^-118.5
    7     128      17  19       659456        -217.1    264     263  2442    306           1710    1.24  2^-128.42  774/|K| = 2^-118.4
    8     256      18  20      1183744        -215.5    288     287  2660    333           1816    1.31  2^-128.42  846/|K| = 2^-118.3
    9     512      19  21      2232320        -213.7    313     312  2887    361           1926    1.39  2^-128.42  921/|K| = 2^-118.2
   10    1024      20  22      4395008        -211.8    339     338  3124    391           2048    1.48  2^-128.42  999/|K| = 2^-118.0
   11    2048      21  23      8720384        -209.9    366     365  3370    422           2174    1.57  2^-128.42  1080/|K| = 2^-117.9
   12    4096      22  24     17371136        -207.9    394     393  3625    454           2304    1.67  2^-128.42  1164/|K| = 2^-117.8
   13    8192      23  25     34672640        -205.9    423     422  3889    487           2438    1.77  2^-128.42  1251/|K| = 2^-117.7
   14   16384      24  26     69345280        -203.9    453     452  4162    521           2576    1.87  2^-128.42  1341/|K| = 2^-117.6
   15   32768      25  27    138690560        -201.9    484     483  4444    556           2718    1.97  2^-128.36  1434/|K| = 2^-117.5
```

Tie site, full parts (`n = 2^d S`), selected rows (the script prints every `d = 3..13`):

```text
profile  d   S  nu_s  terms(tie1,tie2,out,pub)  plain  g_plain  list  listed    g   nonce  tight(RBR)  g_tight
fast     3   1     4  (8, 9, 5, 7)                 46        6    35    20650   15     22        9800       14
fast    13   1    13  (18, 13, 5, 16)              83        7    35    45745   16     23       22050       15
fast    13   4    15  (20, 15, 7, 18)              95        7    35    50925   16     23       24500       15
slim     3   1     4  (8, 9, 5, 7)                 46        6    50    41500   16     23       20000       15
slim    10   1    11  (15, 10, 5, 14)              69        7    50    76950   17     24       37500       16
slim    13   1    13  (18, 13, 5, 16)              83        7    50    92350   17     24       45000       16
slim    13   2    14  (19, 14, 6, 17)              89        7    50    97550   17     24       47500       16
slim    13   4    15  (20, 15, 7, 18)              95        7    50   102750   17     24       50000       16
slim8   13   4    15  (20, 15, 7, 18)              95        7    70   199850   18     25       98000       17
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

Seed `sd` with the sponge list: `2^-139.51 / 2^-139.00 / 2^-138.51`, `g = 0`. Old ledger vs
round-by-round at `2^15`: `3,534/|K| = 2^-116.2` vs `1,434/|K| = 2^-117.5` (stages 300,
`tau` 25, gammas 3, GKR 1,106). Honest failure per site `<= 2^-184.7`, per proof
`<= 2^-175.7`.

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
3. **Composition proof.** The theorem's step (3), Akita Theorem 10.45 with a second,
   straight-line (list-decoding) extractor inside the application segment, needs a written
   proof (step 3 review, paper appendix).
4. **`R_F`** (Flock sites read by the verifier) is not counted exactly here; it enters only
   the `Q = 0` term.
5. **Mode B** uses the same PIOP plan with profile id 3 and no tie site; its Keccak side is
   the public bits, so no Flock term.
6. **Akita sparse folds.** Akita enforces a 128-bit floor on every fold-challenge family
   (`MIN_FOLD_CHALLENGE_ENTROPY_BITS = 128`); Aerie's spec script additionally checks the
   accepted (post-rejection) supports of its rows. The same check for our 13 grouped rows
   is not done here and SHOULD be added with the Akita plan-digest test (4.1).
