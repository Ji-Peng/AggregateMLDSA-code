# Soundness: site census and the proof-of-work schedule (normative)

| Field | Value |
| --- | --- |
| Implemented in | PIOP wire version 4 (`agg_piop::pow`, `agg_piop::transcript`), tied-part proof version 5 (`agg_keccak::tied`, deviations K20, K21, `docs/parity-chain.md`), the list-aware Flock outer sites (`vendor/flock-core/PATCH.md` Section 5, deviation K19), the Akita plan pinned (`agg-pcs-config/tests/all/akita_grinding.rs`) |
| Applies to | mode A and mode B, ML-DSA-44, relations H-u (the default, `docs/h-exact.md` step A), H-exact (relation byte 1, step B; Section 10.1) and H-FIPS (relation byte 2, `docs/h-exact.md` Section 10; Section 10.2), lookup layout L16 with the projection of `v` (K24); PIOP wire version 4 (the parity chain's sites in the plan; the late join of K25; the forked PIOP transcript of K23) and tied-part proof version 5 |
| Base | code `a30ed97`; Aerie `49b6c21`; Akita `e023b6a46143f7b23199ccecb2684192baf7b640`; Flock `83f963e` with the vendored `flock-core` (K12, K18) |
| Script | `docs/scripts/pow_plan.py` (run with `python3 -I`); its output summary is Section 10; `docs/scripts/fold_support.py` (Section 4.6) |
| Replaces | the per-instance union bound (Section 5); resolves deviation P3 |

The uppercase terms MUST, MUST NOT, SHOULD and MAY are normative as in BCP 14. Every
number below that is not marked "estimate" is computed exactly by the script or read from
the pinned code; estimates are marked.

## 0. Results in one page

1. **PIOP sites.** At `N* = 2^15` the transcript draws 456 field challenges in `K`
   (`|K| = p^2`, `p = 2^64 - 59`): the lookup vector `X = (X_1, X_2)`, 26 merges `mu_s`,
   25 GKR batching vectors, 325 GKR rounds of degree 3, `tau`, `gamma^(1..3)` and
   `3 x 25` stage rounds of degrees 4, 4, 3 (every relation). With the assigned bounds of Section 3 every
   field site except `X` needs work: 455 nonces of 8 to 13 bits, 4,195 bits of the packed
   stream. The byte site `proj.psi` precedes them (deviation K24): its answer is the
   projection matrix `Psi` itself (16 KiB), its bad set has mass at most `2^-128` exactly
   (Lemma P, Section 3.5), so it grinds one bit and adds one 8-bit nonce. The packed stream
   of the 456 nonces is 4,203 bits = **526 bytes**, about 2,646 predicate evaluations of
   expected honest work (about 2 ms serial at the measured 724 ns per Blake2b predicate,
   estimate). `X` needs none: its joint bad set has mass `((L + M - 1)/|K|)^2 = 2^-203.5`.
2. **Tied-part sites (version 5).** A tied part is one Flock instance (the sponge) plus the
   virtual parity chain on its tie challenger (`docs/parity-chain.md`). Its challenges are
   drawn while the sponge commitment is bound only to its Johnson list, so each bound
   carries the list size `L_s`: the tie site (the public claim's point) `L_s (nu_s + 3)`,
   the `eq_Y` site `L_s (12 + span_log)` and each of the `R = 13 + d + span_log` chain rounds
   `2 L_s`, over `2^128`. At four full segments under `Slim` (`L_s = 50`): **10, 10 and 7
   bits** (nonces of 17, 17 and 14 bits, 28 rounds), 120 proof bytes of nonces and about
   5,600 BLAKE3 evaluations (Section 8).
3. **Akita.** The pinned Akita `e023b6a` already contains PR #460 (commit `4505404b`, an
   ancestor) and its follow-ups (exact-cardinality pricing #104, #130, #209). Transcript
   proof-of-work is unconditional in `batched_prove`/`batched_verify`; our seven-group
   catalogs already run it: 461 protected sites at `2^15`, targets up to 12 bits
   (ring-switch `alpha`, loss 4,095), plan digest bound in Akita's instance descriptor.
   Nothing needs to be turned on.
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
5. **Binding gap in the tied part.** The instance challengers of a tied part must absorb
   the data the tie points are derived from, because Flock's packed-direct claims absorb
   values only and assume transcript-derived points. Tied-part proof version 5 binds a tie
   digest into the one (sponge) challenger, after the chain's rounds and its bit value
   (Section 4.5).
6. **Tighter accounting.** Round-by-round (first false claim, Aerie Lemma C.1) gives the
   union-bound ledger `1,329/|K| = 2^-117.6` at `2^15` instead of the per-instance
   `3,429/|K| = 2^-116.3`: reductions, zero-checks and linear identities ride on the stage
   sum-checks and on one `tau` site (Section 5). With the schedule every address weighs
   at most `2^-128`, and the theorem of Section 7 follows Aerie Eq. (18).
7. **Composition (Section 12) and fold supports (4.6).** The theorem's two hypotheses are
   settled: every fold-challenge family of both catalogs has at least `2^128.0881` accepted
   challenges (H1, test `akita_fold_support.rs`), and the composition with the Flock
   commitments is proved (H2) with checkpoints at `sd` and `X`, no second extractor, under
   two stated assumptions: Flock's documented round-by-round analysis (F) and the
   correctness of the Keccak circuit (K) (the bridge has no Flock circuit:
   its GF(2) side is a sum-check whose weight the verifier evaluates).
8. **Unforgeability (Section 12.8).** Signing queries asked only in rewound runs are
   handled by lazy forwarding to a SUF-CMA challenger of the
   relaxed scheme (Lemmas 12.6, 12.7) and a three-case SUF-CMA lemma (12.8): SelfTargetMSIS
   at `zeta' = 350,209`, a SHAKE256 collision, or Module-SIS on `[A | I]` at `l_inf`
   380,930. Theorem 12.9: `Adv_agg <= Adv_SUF-CMA + eps_ext + E[R] 2^-H_inf` (`+ Adv_PRF +
   Adv_coll(mu)` for the deterministic signer), no `q_H` or `e(Q_S + 1)` factor. The
   lattice basis is core-SVP `2^121.8` (block size 417, measured; SelfTargetMSIS alone is
   at `2^123.5`).

## 1. Model and method

**Oracles.** Classical random-oracle model as in Aerie Section 5.6 and Akita Section 10.1:
the PIOP Spongefish `Blake2b512` duplex (`agg_piop::transcript`), the tie challenger and
the two Flock instance challengers (Flock's `FsChallenger`, BLAKE3), Akita's native stream
(seeded by the 32-byte PIOP session digest), the setup oracle, and the hashed-entry
oracles (instance digest, public-record digest, PCS session digest, Flock seeds and span
seed with the event-6 record digest they bind, tie digests). The branch digests `d_L` and
`d_S` of the forked PIOP transcript (deviation K23) are `H_P` answers, not entries of these
oracles: the trunk absorbs them as compressed prefixes of the two branches (Lemma 12.1b). Every challenge input MUST injectively encode the complete preceding
transcript, directly, through a hashed entry or through a branch digest (Aerie C.1); Section 4.5 records the one
place where the code does not. The tied parts' sites and the sites of branch S (K23)
encode the input of `X` and their own messages, not the complete linear prefix, and Akita's
sites omit the Flock proofs (K25); Lemma 12.3 covers them.

**Sites and bad sets.** A *site* is one challenge query: a scalar or a vector drawn from
one oracle stream (Aerie C.1: "we treat each vector as one field challenge"). For a site
`t`, fix the objects the extractor will return (the 53 Akita polynomials `f`, 54 under H-exact and 59 under H-FIPS, the Flock
word vector `z_s` of each tied part up to its Johnson list) and the transcript prefix; the
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
coefficients `l/|F|` (its true numerator is 1); a line merge `1/|F|`. The tied parts' sites
are assigned their identities with the list factor (Section 3.6).

## 2. Transcript map

Events of spec Section 10, with the sites they contain (mode A; mode B has no Flock
sub-proofs and therefore no tied-part site):

| Event | Oracle | Sites (in order) |
| --- | --- | --- |
| 0 | PIOP | none; binds the instance record, which ends with the plan digest (Section 6.4) |
| 1, 2 | PIOP | none (absorbs: round-1 Akita commitments; Flock sponge caps) |
| 3 | PIOP | seed `sd` (32 bytes, no field sampling); then `proj.psi`, the projection matrix `Psi` (16 KiB, its own query, no field sampling, one bit of work; K24) |
| 4 | PIOP | none (round-2 Akita commitments, `G_k` among them) |
| 5 | PIOP | seeds `fs_sigma` (32 bytes each); the span seed is a hash of them |
| 6 | PIOP absorbs `Y_sigma`, then the share items (deviation K22: no Flock proof bytes here); no site |
| fork | PIOP (K23) | none; the sponge is cloned into branch L (event 7) and branch S (events 8 to 11), each absorbing its own label |
| 7 | PIOP, branch L | `X` (vector of 2); `mu_1`; per session `s = 2..D`: `gamma_s` (vector), `s - 1` rounds, `mu_s` |
| 8, 9 | PIOP, branch S | `tau` (vector of `d_main`) |
| 10, 11 | PIOP, branch S | `gamma^(1)` (26), 25 stage-1 rounds; `gamma^(2)` (9), stage-2 rounds (each stage `d_main` rounds); then the absorbed projection claim `c` |
| branch join | PIOP (K23) | none; the trunk absorbs the 32-byte digests of L and S, in this order |
| 12 | PIOP | `gamma^(3)` (48), `d_main` stage-3 rounds |
| join | no PIOP absorption (K25); inside each tied part (own oracles, seeded by the part or span seed, which binds `fs_sigma` and the event-6 record digest `rec`; the prover runs them concurrently with events 7 to 13): on the tie challenger the **tie site**, the **`eq_Y` site** and the **chain rounds** (version 5), then the sponge instance's Flock sites (zerocheck, lincheck, opening with the bit claim) |
| 13 | PIOP export, then Akita | PCS session digest (no site); Akita's own sites under its plan (Section 3.7) |

The code that draws them: `agg_piop::transcript::Transcript::{challenge_extensions,
challenge_bytes, pcs_session}` (registry `Registry::new`), called from
`lookup::{prove_in, verify_in}` (`X`, `Gamma(s)`, `RoundChallenge(s, r)`, `Mu(s)`),
`prover::prove_piop` / `verifier::verify_piop` (`tau`; the fork and join,
`Transcript::{fork, join}`), `stage::{prove_stage_with,
verify}` (`gamma^(s)`, round challenges), `prover::prove_with_tables` /
`verifier::verify` (`sd`, `proj.psi`, `fs`); `agg_keccak::tied::TiedPart::{prove_with_final_vectors,
verify}` with `agg_bridge::parity_chain` (tie, `eq_Y`, chain rounds);
`agg_keccak::flock_union::UnionGeometry::{prove, verify}` (Flock); Akita
`batched_prove` / `batched_verify` via `agg_pcs_config::commit` (Akita).

## 3. Site census (normative)

### 3.1 PIOP sites

Geometry at `N* = 2^d_sig` (spec Section 1, `agg_pcs_config::catalogs::Group`):
`d_main = 10 + d_sig`, `d_1 = 13 + max(0, d_sig - 9)`, `d_2 = 13 + max(0, d_sig - 13)`,
`d_3 = 12 + max(0, d_sig - 13)`, `d_k = 7 + d_sig`; per lookup challenge the trees have depths
`d_main + 1, d_1 + 3, d_2 + 3, d_3, d_k + 1` (stacks) and 16 (`T_16`), two challenges, so twelve
trees; `D = max(d_main + 1, 16)`; `L = 2 2^d_main + 8 2^d_1 + 8 2^d_2 + 2^d_3 + 2 2^d_k`,
`M = 2^16`; `active_s` is the number of trees of depth `>= s`.

| Class (id) | Count | `nu_t` | Assigned `e_t` | `g_t` | Nonce bits | At `2^15` |
| --- | --- | --- | --- | --- | --- | --- |
| `proj.psi` (11) | 1 | byte site | `2^-128` (exact, Lemma P) | 1 | 8 | `2^-129` |
| `lookup.x` (1) | 1 | joint (3.2) | `((L + M - 1)/|K|)^2` | 0 | 0 | `2^-203.5` |
| `lookup.mu` (2) | `D` (`mu_1 .. mu_D`) | 1 | `1/|K|` | 1 | 8 | 26 sites |
| `lookup.gamma` (3) | `D - 1` | 1 | `2 active_s / |K|` | `bit_length(2 active_s)` | `g + 7` | 24, 20, 16, 12, 8, 4 coefficients: `g` = 5, 5, 5, 4, 4, 3 (13, 2, 2, 4, 1, 3 sites) |
| `lookup.round` (4) | `D (D - 1)/2` | 3 | `3/|K|` | 2 | 9 | 325 sites |
| `tau` (5) | 1 | `d_main` | `d_main/|K|` | `bit_length(d_main)` | `g + 7` | 25: `g = 5` |
| `stage.gamma` (6) | 3 | 1 | `26, 10, 49 / |K|` (H-exact: `26, 9, 50`; H-FIPS: `26, 9, 56`) | 5, 4, 6 (every relation) | 12, 11, 13 | |
| `stage.round` (7) | `3 d_main` | 4, 5, 3 | `deg/|K|` | 3, 3, 2 | 10, 10, 9 | 75 sites |

Totals per `N*` are in Section 10 (from 195 nonces / 228 bytes at `N* = 8` to 456 /
526 bytes at `2^15`). The byte site `proj.psi` (Section 3.5) is the first entry of the plan
(Section 6.4) with `g = 1`; every other entry is a field challenge. The largest field
difficulty is 6 bits (`gamma^(3)`), far below
Akita's 25-bit transition cap.

### 3.2 The lookup challenge `X = (X_1, X_2)`: one site with the joint bound

`X_1, X_2` are drawn by one vector challenge (`LookupSite::X`, two elements from one
stream). The pending claim before `X` is "every looked-up cell of `f` lies in `T_16`"
(cells and the multiplicities `m` are fixed by the commitments of events 1 and 4; the
projection digits `K_0, K_1` are round-2 cells). If it
is false, the numerator `N(X)` of `R(X) = sum_i 1/(X - f_i) - sum_t m(t)/(X - t)` over
`prod_{a in S}(X - a)` is a nonzero polynomial of degree at most `L + M - 1` (spec 12.1,
Lemma; LogUp Lemma 5 with `p > L`). After the answer, the transcript stays doomed unless
both true root identities pass, i.e. unless `N(X_1) = N(X_2) = 0` with `X_j` outside `S`
(if some `X_j` lies in `S`, the true root denominator is zero and the verifier's nonzero
check forces a false layer-1 value). Hence `Bad_X = {N(X_1) = 0 and N(X_2) = 0}`, of mass
at most `((L + M - 1)/|K|)^2` because the two coordinates are independent and uniform
(exact per-limb rejection sampling). This is one site: its bad set is fixed before the
query and its mass is the joint one; Proposition 10.3 then bounds `N` queried candidates
by `(N + 1) ((L + M - 1)/|K|)^2`. `g_X = 0` at every `N*` (`2^-220.5` at `N* = 8` to
`2^-203.5` at `2^15`). Treating `X_1` and `X_2` as two sites would be wrong (each alone
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
* `gamma^(s)` (26, 10, 49 coefficients; 26, 9, 50 under H-exact; 26, 9, 56 under H-FIPS):
  `nu = 1` as in 3.3.
* stage rounds: `nu` = the round polynomial's degree, 4, 5, 3 (`STAGE_DEGREES`; the `eq`
  factor is included: `range_w1` has degree 4, `leaf8` of the projection digits `K_2, K_3` degree 5
  (`leaf4` of `Z_hi` degree 3 and `leaf6` of `U_hi` degree 4), `link` degree 3).
* identities tested at a point drawn by the preceding stage (the leaves at `r^(1)`, the
  ring identity, the links and the projection identity (spec 6.3) at `r^(2)` or its `D_k`
  suffix `r'`): while the earlier stage draws the point, the
  identity is the pending claim "the partially substituted `sum_x eq(r, x) phi(x)` is the
  zero polynomial", which a round's answer can falsify for at most one value of its
  coordinate; that value lies inside the round's own `nu >= 1` (Aerie Lemma C.1,
  "Identities tested at a point of the preceding stage"). No extra site.
* the projection identity: `c = sum_x W(x) v(x) = sum_y eq(r', y) k(y)`, two stage-3
  instances with the one sent value `c`. Its left side is the multilinear extension of
  `(o, i) -> <Psi_o, v_i>` at `r'`, so before `r^(2)` the pending identity is `D~(r') = 0` for
  the table `D = k - <Psi, v>` on `D_k` (fixed by events 1, 3 and 4, all before the fork);
  each of the 22 stage-2 rounds that binds a coordinate of `r'` can falsify a nonzero partial
  polynomial for one value only, inside the round's `nu = 5`. No extra site. With two
  separately sent claims nothing would bind `k` to `v`; the one claim `c` makes a false
  `D~(r') = 0` a false stage-3 input claim, which `gamma^(3)` and the rounds carry.
* evaluation reductions and the five lookup-stack reductions are stage-3 instances: a
  false input claim makes the instance false, which `gamma^(3)` and the stage-3 rounds
  carry (Section 5). No extra site.

### 3.5 Seed sites, the projection site and exports

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
  `2^(d_seg - 3)` parts a segment could otherwise hold.
* `proj.psi` (event 3, after `sd`, its own `H_P` query; deviation K24): the answer is the
  projection matrix `Psi in {0,1}^(128 x 1024)` itself, squeezed as 512 blocks of 32 bytes
  from the PIOP duplex (16 KiB, `Transcript::challenge_psi`) right after the sponge absorbs
  the raw label `agg-mldsa/v1/proj/psi`, so its input differs from `sd`'s (a squeeze directly
  after `sd` would continue `sd`'s output stream, one query); `Psi` is uniform at its
  address and no expansion oracle intervenes. Bad set (12.4):
  `Bad_proj = {Psi : for all o < 128, |<Psi_o, v_{i*}>|_c <= 2^37}` for the first record
  `i*` with a coordinate `|v_{i*}(x_0)|_c > 2^38` (a function of the round-1 candidate
  `f_1`, fixed before the query), mass `<= 2^-128` by Lemma P, exactly (no density loss):
  `e = 2^-128` exactly. The site is a byte challenge, so its difficulty is set by policy
  rather than by a field numerator: it grinds one bit (`g = 1`, the 8-bit nonce of
  Section 6.1) and weighs `2^-129` per address. An expanded 32-byte seed would need
  Lemma P' below with `theta_Psi = 2^-127` and `g = 2` for the same `2^-129`.

  **Lemma P (one-coordinate anti-concentration).** Let `R, n >= 1`, `B >= 0` with `4B < p`,
  and let `Psi in {0,1}^(R x n)` have iid uniform entries. For every `v in F_p^n` with
  `|v(x_0)|_c > 2B` for some `x_0`: `Pr_Psi[for all o < R: |<Psi_o, v>|_c <= B] <= 2^-R`.
  For `v_0, ..., v_(N*-1)` fixed before `Psi`, if some `v_(i*)` has such a coordinate, the
  probability that every `(o, i)` passes is at most `2^-R`.

  *Proof.* Row `o`: condition on `Psi_o(x)` for `x != x_0`; the output takes two values that
  differ by `v(x_0)`. Both within `B` of 0 would give `v(x_0) = a - b` in `F_p` with integers
  `|a|, |b| <= B`, so `|v(x_0)|_c <= 2B` since `2B < p/2`. So the row passes with probability
  at most 1/2. The bits `Psi_o(x_0)` are independent across `o`, which gives `2^-R` for
  record `i*`; that record alone bounds the batch, so no union over records enters. With
  `R = 128`, `B = 2^37` the certified bound is `|v|_c <= 2^38`, below the `(p/2 - 2^50)/q ~
  2^40` that (6.1) needs (2 bits of margin). The bound is tight: `v = (v_0, 0, ..., 0)` with
  `|v_0|_c > 2B` passes a row iff its bit is 0. Rows over `{-1, 0, 1}` pass with probability
  up to 2/3 per row (219 rows for `2^-128`); rows over `[-2, 2]` admit `v(x_0)` near `p/2`.

  **Completeness for every key.** A valid signature has `|z|_inf <= B_z = 130,993` and
  `|u|_inf <= B_u = 190,465`, so the quotient definition (3.4) gives
  `|v|_inf <= B_v = floor((2^10 ((q-1)/2) B_z + 2^d tau (2^10 - 1) + 2 gamma_2 43 + B_u)/q)
  = 67,068,447 < 2^26`. Hence `||v_i||_2 <= 32 B_v < 2^31`, and every row of `Psi` has at
  most `2^10` ones, so `|k_{o,i}| <= ||Psi_o||_2 ||v_i||_2 <= 32 * 2^31 = 2^36`. Every honest
  output therefore lies inside the proven `[-2^37, 2^37)`: the prover never aborts at this
  check, for any key and any valid signature. This replaces the earlier simulation estimate
  (`|k| <= 2^31.4` for uniform `A`), which did not cover an adversarially chosen key.

  **Lemma P' (density; only if `Psi = Expand(seed)` with a fixed oracle `H_Psi`; not used).**
  Except with probability `delta_Psi < 2^-(2^125)` over `H_Psi`, every `v in F_p^1024` with a
  coordinate above `2^38` in absolute value is passed by at most a fraction
  `theta_Psi = 2^-127` of the 256-bit seeds (as Lemma 12.4: for a fixed `v` the number of
  passing seeds is dominated by `Bin(2^256, 2^-128)`, which exceeds `2^129` with probability
  at most `exp(-2^128/3)`, and a union over the `p^1024 < 2^65536` vectors and the batch
  sizes gives the bound). It is why the protocol squeezes `Psi` directly.
* `fs_sigma` (event 5), the part and span seeds (which also bind the digest `rec` of the
  event-6 record, deviation K22), the PCS session digest (event 13): seeds of sub-protocol
  oracles. They have no bad set of their own; the sites they seed are counted where they
  occur.

### 3.6 The sites of a tied part (version 5: tie, `eq_Y`, parity-chain rounds)

Tied-part proof version 5 (deviation K21, `docs/parity-chain.md`) has one Flock instance per
tied part, the sponge instance; the GF(2) side of the bridge is a sum-check run on the tie
challenger. The tie
challenger draws, after the sponge cap, the part's geometry and the final-vector digest,
in order: the public claim's point `(rho_r, rho_c)`; `eq_Y = (r_o, r_y)`; one challenge
`r_k` per chain round after absorbing that round's message; then it absorbs the claimed bit
value `v` and squeezes the tie binding (4.5). Each of these is a site in GF(2^128) with its
own work. With `span_log = ceil(log2 S)`, `nu_s = capacity_log(n, 19)` and `R = 13 + d +
span_log` rounds (`min(10 + d, 19)` in-block positions, `max(0, d - 9)` block, 3 lane and
`span_log` segment coordinates):

| Site | Identity at stake | Objects | Numerator |
| --- | --- | --- | --- |
| tie (`rho_r, rho_c`) | public words: the sponge's public region vs the verifier's table | sponge (`z_s`) | `nu_s + 3` |
| `eq_Y` (`r_o, r_y`) | `<eq_Y, M b'> = <eq_Y, Y>`: two multilinear tables in `log2 m_3 + span_log` variables | sponge | `12 + span_log` |
| round `k < R` | the sum-check's round polynomial (degree 2) | sponge | 2 |

**List factor.** When these challenges are drawn the sponge commitment is bound only to its
Johnson list (the level-0 two-point binding happens inside its opening, after them), so each
bad set is a union over the list, `L_s` elements. With `L = floor(1/(2 eta sqrt(rho_0)))`
(Flock Eq. (10), `eta = 0.02`, level-0 rate `rho_0 = 2^-r`): `L = 35, 50, 70` for `Fast,
Slim, Slim8` (`r = 1, 2, 3`). Every identity concerns one list element, so there are no pairs.

* Assigned (normative): `e_tie = L_s (nu_s + 3) / 2^128`, `e_eq = L_s (12 + span_log) / 2^128`,
  `e_round = 2 L_s / 2^128` (`agg_bridge::pow::part_work`). ML-DSA-87: `e_tie = L_s (nu_s + 4) /
  2^128` (its public region of mu, a 64-byte `c~` and the constant spans 16 words, so the
  public claim has four word coordinates).
* `g = ceil(log2(2^128 e))` per site; nonce `g + 7` bits.

At four full segments (`d = 13`, `S = 4`, `nu_s = 15`, `R = 28`): tie `630 / 900 / 1,260`
(`g = 10, 10, 11` for `Fast / Slim / Slim8`), `eq_Y` `490 / 700 / 980` (`g = 9, 10, 10`),
rounds `70 / 100 / 140` (`g = 7, 7, 8`). Every `(d, S)` is tabulated in Section 10. The
difficulties are functions of the part's public geometry `(profile, d, S, n)`, all known to
the verifier before it replays event 0. The bridge lemma (`2^-144.64`, `docs/bridge.md` 7)
and the seed site (3.5) are untouched: `b'` is fixed by the sponge cap before `sd`, and `Y`
by its digest before `eq_Y`.

### 3.7 Akita opening sites (inherited, already protected)

Akita's plan (`akita_params::derive_transcript_grinding_plan_from_public_shape`) assigns
every field site of the opening a loss factor and a target by Akita Eq. (146) priced over
the exact cardinality `p^2` (`ChallengeFieldOrder::from_field`, PR #104): evaluation
batch (loss = claims, 53), extension-opening point and batch, every sum-check round
(stage 1, 2, physical L2, extension-opening reduction; loss = degree), ring-switch
`alpha` (loss `2d' - 1`, Akita Theorem 10.36), `tau0`, `tau1` points, interstage, L2
merge and virtual batches, compression and stage-2 batches. Sparse fold challenges carry
no nonce: each coordinate stream has weight `1/C_{j,g}`, and Akita refuses any fold
family below its 128-bit entropy floor (`SparseChallengeConfig::validate_for_ring_dim`,
`MIN_FOLD_CHALLENGE_ENTROPY_BITS`). Dumped for our catalogs (Section 4.1):

| `d_sig` | PoW sites | `sum g` | max `g` | expected work `sum 2^g` | nonce bits |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 to 5 | 401 | 984 | 9 | 3,888 | 3,863 |
| 6 | 401 | 985 | 10 | 4,400 | 3,864 |
| 7, 8 | 406 | 980 | 10 | 5,120 | 3,894 |
| 9 | 388 | 966 | 10 | 5,132 | 3,754 |
| 10 | 388 | 966 | 12 | 7,436 | 3,754 |
| 11 | 452 | 1,121 | 12 | 8,304 | 4,369 |
| 12 | 452 | 1,120 | 12 | 8,048 | 4,368 |
| 13 | 457 | 1,133 | 12 | 8,332 | 4,416 |
| 14 | 459 | 1,137 | 12 | 9,872 | 4,434 |
| 15 | 461 | 1,142 | 12 | 9,884 | 4,453 |

These are the seven-group rows of deviation K24 (`G_k` and `G_3` in `G_2`'s layout,
`docs/pcs-config.md` 4.1); before K24 the table read 401 sites at `d_sig` 3 to 8, 406 at 9,
388 at 10, 430 at 11, 452 at 12, 455 at 13, 457 at 14 and 459 at `2^15` (`sum g` 1,137, work
9,872, 4,434 nonce bits). The evaluation batch now has 53 claims (H-u; 54 under H-exact, 59
under H-FIPS, the zero claims of `G_3` included), still `g = 6`. These nonces are already in
the measured opening (65,282 B planner estimate at 32K before K24, 65,398 B after).

The 13 H-exact grouped rows (`mldsa-44-p16x-batch-<N*>-mult-final`, `G_main4` at 9
polynomials) have the same plan summaries as the H-u rows at every size; their digests are
pinned next to H-u's (`agg-pcs-config/tests/all/akita_grinding.rs`,
`akita_plans_of_the_h_exact_rows_are_pinned`). The H-FIPS rows reach each step one size
earlier (`PINNED_H_FIPS`; at `2^15`: 461 sites, `sum g` 1,141, work 8,860, 4,452 nonce bits).

### 3.8 Flock internal sites (inherited; audit in 4.2 to 4.4)

Per Flock instance (one per tied part, 3.6): zerocheck (initial point, `bits_for(m)`; skip point,
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
  its own PIOP plan (`aerie_falcon::protocol::grinding::PiopGrindingPlan`), the counterpart of
  ours (deviation P3).
* Our presets: `agg_pcs_config::commit` calls `AkitaCommitmentScheme::batched_prove` and
  `AkitaVerifier::batched_verify` with the width-4 catalog (`Cfg = AggFp64SignedWidth4`),
  so our openings already grind. Evidence: a scratch binary (not committed) calling
  `akita_config::derive_transcript_grinding_plan::<AggFp64SignedWidth4>` on every grouped
  row (`WorkPlan::for_records(2^d_sig).schedule_key()`) produced the table of Section 3.7;
  plan digests begin `22 57 08 cb` (`d_sig` 3) through `4f 5a 60 26` (`d_sig` 15) (hex).
* No change in `agg-pcs-config` is needed; `crates/agg-pcs-config/tests/all/akita_grinding.rs`
  (a regression guard against a catalog or Akita change) derives the plan of every
  grouped row for the opening layout of its six root groups
  (`OpeningClaimsLayout::from_root_groups`), checks the table of Section 3.7 (sites, `sum g`,
  max `g`, `sum 2^g`, nonce bits of all runs) and pins the 13 full digests (`225708cb...` at
  `d_sig` 3 to `4f5a6026...` at 15).

### 4.2 Flock: what the documented 128-bit level covers

Read from `vendor/flock-core` (identical to `83f963e` except K12/K18) and Flock's
`docs/128-bit-grinding-audit.md` at that upstream revision: under `Fast`, `Slim` and `Slim8` the policies
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
proof version byte SHOULD be added) so that old proofs are rejected cleanly. The finding
SHOULD be raised with the Flock authors (they may have an argument for the outer sites
that their audit document does not state; Section 11, open 1).

Implemented (`vendor/flock-core/PATCH.md` Section 5, deviation K19): the policies
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

**Version 5.** The bridge union is gone; the tie challenger absorbs `fs`, the part's
geometry digest, the final-vector digest and the sponge cap, runs the tie site, `eq_Y` and
the chain rounds, absorbs the claimed bit value `v` and then squeezes the tie digest, which
the sponge challenger absorbs before `bind_statement`. Both claim points of the sponge
opening (the public region's and the chain's bit point) and the value `v` are thereby fixed
before any Flock challenge of the instance.

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

**Check (test `crates/agg-pcs-config/tests/all/akita_fold_support.rs`).** Every row of both
catalogs: width 4, 78 rows (per relation H-u, H-exact and H-FIPS, the 13 grouped rows and
the 13 scalar rows of `G_main4`); width 16, 26 rows (shared by the three relations) (the scalar rows of `G_main16`, `G_1`, `G_2`, `G_3` at every size, equal layouts
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
`ALL CHECKS PASS` for both certificates.

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
relations' rows.

**Result.** H1 holds for every row of both catalogs. The smallest accepted support is
`2^128.0881` (Akita's production `D = 64` shell, margin 0.088 bits), so every fold coordinate
weighs at most `2^-128.088` and needs no work. Nothing is below the floor and nothing needs to
change. The test pins the per-row minima, so a catalog regeneration or an Akita change that
alters a family fails it.

## 5. Tighter accounting (round-by-round), with the argument

**Lemma S (checks of fixed objects).** Fix the 53 Akita polynomials `f` (54 under H-exact, 59 under H-FIPS) (values in `F_p`
on their Boolean domains) and a statement. For every challenge input `u` of a PIOP site from
`X` (event 7) to the PCS export, let
`D(u) = 1` iff a verifier check determined by `u` fails or a pending claim is false,
where the pending claims are: at `X`, the relation identities of `f` (Sections 4 to 6, (9.2)
with the absorbed `Y_sigma`; `R_PIOP` of Section 12.2); after each site, the claims listed
in Section 3 (layer claims, stack and table leaf claims, stage input claims, terminal
evaluation claims). (The sites before `X`, i.e. `sd` and `fs`, and the tied parts' sites,
the tie sites and Flock's, which the analysis places before `X` (12.2), are treated in Section 12.4 with the round-1 polynomials and the Flock words'
Johnson lists; a false tied-part claim is not acted on by any PIOP site. A site of branch S
(deviation K23) defines `Bad_t` from the first false claim among those its address
determines, the identities of `f` and branch S's own claims; a false claim of branch L is
not acted on by any site of S and stays false, Section 12.2.) Then for every site `t` of Section 3 and every input `u` with
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

1. *Reductions are credited to the sum-checks*. An evaluation
   reduction `f(pi) = sum_x eq(pi, x) f(x)` (the 37 reductions and the four stack
   reductions (7.7)) has the earlier claim as its input claim; a false input claim makes
   the instance false and costs nothing beyond `gamma^(3)` (`nu = 1`) and the stage-3
   rounds (`nu = 3` each), which every instance shares. The per-instance ledger charged each of them
   `d_main/|K|` ("linear identities at a point", 62 instances), as if each had its own
   sum-check.
2. *One site for all zero-checks and linear identities at `tau`*: `nu_tau = d_main`, not
   the per-instance ledger's `19 d_main` for the zero-checks plus `d_main` per linear identity.
3. *Leaves at `r^(1)` and links at `r^(2)`*: inside the rounds (3.4), not the per-instance
   ledger's `4 d_main`; the ring identity at `r^(2)` likewise (that ledger counted it among the 62).
4. *GKR batching vectors*: `nu = 1` each; the per-instance ledger already used this (`docs/lookup.md`
   Section 2); the assigned bound keeps Aerie's `l/|K|`.

Before work, the round-by-round union bound (sum of `nu_t/|F_t|`, which is also the
interactive soundness error of the PIOP, plus the `X` term `2^-203.5`, the `sd` term (`<=
2^-131.4`) and the `proj.psi` term `2^-128`) is `(11 d_main + d_main + 3 + GKR)/|K|` (the
stage degrees sum to 11 in H-u):
`914/|K| = 2^-118.2` at `2^10`, `1,154/|K| = 2^-117.8` at `2^13`, `1,329/|K| = 2^-117.6`
at `2^15`, against the per-instance ledger's `2,594`, `3,086`, `3,429` over `|K|`. This is still not
128 bits; the schedule makes every site `<= 2^-128` per address, which is the statement of
Section 7.

**Where the conservative bound stays.** The schedule (`e_t`) keeps Aerie's assigned bounds
(`l/|K|` for batching vectors) and the list factor at the tied parts' sites; `nu_t` (the sharper
value) is what the analysis needs.

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
   variable-length items (the event-2 Flock items, the event-6 share items, the `c~` list) are framed with their lengths. The absorbed prefixes are therefore injective in the transcript, which is
   the hypothesis of Akita Proposition 10.3.
4. Sites with `g = 0` (only `X`) absorb nothing and squeeze no predicate.

### 6.2 The packed nonce stream

One byte string per proof, `work`, carries the PIOP nonces in plan order: nonce `i`
occupies bits `[o_i, o_i + g_i + 7)`, `o_i = sum_{j < i} (g_j + 7)`, little-endian bit order
within the stream (bit `k` is bit `k mod 8` of byte `k / 8`). Its length MUST be exactly
`ceil(sum_i (g_i + 7) / 8)` bytes and its padding bits MUST be zero; the verifier rejects
any other length or nonzero padding before replaying event 7. In the wire format it
follows the stage-3 terminal claims and precedes the Akita opening. The
tie nonces are not in this stream (6.3); Flock's and Akita's nonces stay in their own
proofs.

### 6.3 The tied part's nonces (tie, `eq_Y`, chain rounds)

On the tie challenger (`agg_keccak::tied`, version 5), immediately before the challenge of
each of its sites (the tie site after the sponge cap, the `eq_Y` site after the public
point, each chain round after absorbing the round's message `(q(1), q(inf))`): `nonce =
grind_pow(g)` (Flock's `Challenger::grind_pow` / `verify_pow`: predicate BLAKE3 of
`state_digest || LE64(nonce) || 0^24` with `g` leading zero bits, nonce absorbed as `LE64`),
then the site's coordinates. After the last round the challenger absorbs the bit value and
squeezes the tie digest of 4.5. The honest search MUST stop at `2^(g + 7)` candidates (Flock's
`grind_pow` searches upward from 0 and returns the smallest passing nonce; if that nonce is
`>= 2^(g + 7)`, the prover stops with `KeccakError::Work`, which mode A returns as
`ProveError::Keccak`). The nonces travel in the tied-part proof as `LE32`: the tie nonce in
the header, the `eq_Y` nonce after it, each round's nonce after its message; the verifier
rejects a value `>= 2^(g + 7)` or one failing the predicate (`KeccakError::Tie` with `"tie
nonce"`, `"eq_Y nonce"`, `"chain round nonce"`). Flock's own nonces get the same width rule
in the vendored verifier (`vendor/flock-core/PATCH.md` Section 6, deviation K19).

### 6.4 The plan and its digest

The plan of a proof is the ordered list of its tied parts' sites (per tied part, in segment
and part order: the tie site, the `eq_Y` site, then the `R` chain rounds) followed by the
PIOP sites in registry order (the byte site `proj.psi` with `g = 1`, then the field-challenge
sites, `X` included with `g = 0`). The tied parts come
first by convention of the encoding: the transcript sponge never absorbs their messages, and
the analysis places their sites before `X` (12.2). It is a function of public data only: `N` (hence `N*`), the Flock profile,
and the tied-part geometry (span byte, part size) of the event-2 items, which the verifier
parses before replaying event 0. Canonical encoding:

```text
body  = LE64(N*) || u8(profile id: 1 Fast, 4 Slim, 5 Slim8, 3 mode B)
        || u8(S, the span length; 0 in mode B) || u8(d_part; 0 in mode B)
        || LE32(site count) || entries
entry = u8(class id, Section 3.1; 8 = tie, 9 = chain eq_Y, 10 = chain round,
        11 = proj.psi) || LE16(i)
        || LE16(j) || u8(g) || u8(nonce bits)
        (i, j) = (s, 0) for lookup.mu and lookup.gamma, (s, r) for lookup.round,
                 (stage, 0) / (stage, r) for stage sites, (part index, 0) for the tie and
                 eq_Y sites, (part index, k) for chain round k, (0, 0) for proj.psi,
                 lookup.x and tau
digest = SHA3-256("agg-mldsa/v2/pow-plan" || LE64(len(body)) || body)
```

The 32-byte digest MUST be appended to the instance record of event 0 (after the Flock
profile byte, `binding::instance_record`), so it is absorbed before the first site; the
manifest's wire version is 4. The script prints test vectors for `N = N*` (Section
10); the implementation MUST reproduce them. (The plan uses the actual `n` of each tied
part in `nu_s`; the vectors use full batches.)

### 6.5 Honest search allowance and failure probability

The prover tries `v = 0, 1, ...` and takes the first passing value, at most `2^(g + 7)`
candidates (deterministic proofs). A site fails with probability
`(1 - 2^-g)^(2^(g + 7)) <= exp(-2^7) = 2^-184.66`. The capped searches of a proof are the
PIOP sites (at most 457, the byte site `proj.psi` included), the tied parts' sites (per part a tie, an `eq_Y` and at most 28
round sites: at most `8 x 30 = 240` with the two-part cap of 3.5), Akita's capped
transcript searches (461 at `2^15`) and, since the
vendored Flock verifier rejects nonces of `g + 7` or more bits (`vendor/flock-core/PATCH.md`
Section 6), every Flock site with work: Flock's prover still searches upward without a cap,
so a smallest passing nonce of `2^(g + 7)` or more yields a proof that the verifier refuses,
with the same probability per site. Flock has fewer than `2^10` sites with work per tied part
(estimate from the round counts: one instance, a zerocheck, a lincheck, the
opening's three ring switches, claim batch and merged rounds, and at most six Ligerito
levels with two batches and one query nonce each, at `m <= 45` variables), so at most
`2^13` for eight parts. The completeness error is therefore at most `(457 + 240 + 461 +
2^13) 2^-184.66 < 2^-171.5` (the Flock count an estimate; without it `< 2^-174.5`). A search
that exhausts its allowance returns `KeccakError::Work` (no panic). Expected work per site `2^g` predicate
evaluations.

## 7. Theorem (128-bit knowledge soundness, classical ROM)

**Theorem.** Fix `N <= 2^15`, mode A with Flock profile `Fast` or `Slim` with the fix of
4.3 (or `Slim8` with 4.3 and 4.4), tied-part proof version 5 (3.6, 4.5), PIOP wire version 4
(relation H-u, H-exact or H-FIPS, the caller's relation, bound by byte 3 of event 0 and checked
by the verifier before event 0) with the plan of Section 6 (Section 10.1 for H-exact, 10.2 for
H-FIPS), at most two tied parts per segment (3.5), and Akita's plan for the selected catalog
row. Its two hypotheses
are settled:

* **(H1, discharged in 4.6)** every fold group of every row of both catalogs draws its
  sparse fold challenges from a family whose accepted support is at least `2^128` after
  rejection (smallest `2^128.0881`; test `agg-pcs-config/tests/all/akita_fold_support.rs`);
* **(H2, proved in Section 12 under (F) and (K))** the composition of Akita's extractor with
  the Flock commitments: no second extractor is needed (the extractor returns Akita's
  round-1 sources; the Ligerito words are fixed straight-line by the Merkle expansion and
  enter the round-by-round bits through their Johnson lists), with a checkpoint at `sd`
  (round-1 candidate) and one at `X` (all seven groups), and Lemma 12.3 for the parallel Flock transcripts and for branch S of
  the forked PIOP transcript (K23). With K23 the lemma also carries 53 core PIOP sites at
  `2^15` (`tau`, `gamma^(1)`, 25 stage-1 rounds, `gamma^(2)`, 25 stage-2 rounds); it has had
  internal review only (12.7 item 3).

The theorem is conditional on the assumptions of Section 12.5: **(F)** Flock's documented
round-by-round analysis of its instances (per-site bounds as validated by its configs, the
list factor at the outer sites by K19, its work composition, its ring-switched claims with the
multilinear skip basis of K21) and **(K)** the correctness of the sponge circuit. The paper MUST state (F) and (K) as assumptions and
MUST NOT claim the theorem without them. For every
prover `P` making at most `Q` oracle queries (all oracles of Section 12.1, every nonce probe
included) there is an extractor of expected cost at most `(Q_wrap + 2) 4 Tree0_ck(sch) T_run`
(Section 12.6). Here `T_run` bounds one run of `P` and the verifier with the extractor's
work attached to it. The cost is thus polynomial in that of `P` and in the code parameters.
The extractor returns the original execution and, whenever the verifier accepts, a witness
of the caller's relation (H-u, H-exact or H-FIPS) for every live record, except with probability at
most

```text
eps_ext <= (Q + R_f + R_coord + R_F) 2^-128 + eps_bind + eps_pk + eps_dig + eps_fs
           + eps_br + eps_merkle + delta_X + delta_samp + sum_I Adv^MSIS_I(T_I)
```

where `R_f` is the number of field sites of the PIOP and of Akita read by the verifier
(457 + 461 at `2^15`, the byte site `proj.psi` and the 456 field challenges, plus the seeds
and the tied parts' sites), `R_coord` Akita's sparse fold coordinates read by
the verifier, `R_F` the Flock sites read by the verifier (the sponge instance of every tied
part, outer and Ligerito, queries included), `eps_bind, eps_pk, eps_dig` the hashed-entry terms of Aerie
Definition 5.8 (instance digest, public-record digest, PCS session digest, Flock part and
span seeds, the event-6 record digest `rec` that they bind (K22), final-vector digests, tie
bindings, Flock statement digests, the BLAKE3 chaining of Flock's challenger), `eps_fs` the
Flock-seed term of Lemma 12.1, `eps_br <= 5 Q_P^2 / 2^257 + Q_P / 2^191` the term of the branch digests
`d_L`, `d_S` of K23 (Lemma 12.1b; `Q_P` the `H_P` queries), `eps_merkle` the
Ligerito Merkle term of Lemma 12.2, `delta_X < 2^-(2^110)` the seed-density term of Lemma
12.4, `delta_samp = 0` for the exact samplers, and the Module-SIS terms those of Akita
Lemma 10.20 for the schedule's occurrences.

*Proof (Section 12).* (1) Hashed entries, the Flock seeds, the branch digests and the
Ligerito caps are expanded (Akita Lemma 10.47, Lemmas 12.1, 12.1b, 12.2): `eps_bind + eps_pk
+ eps_dig + eps_fs + eps_br + eps_merkle`.
(2) A checkpoint structure for the expanded protocol (12.2): checkpoints at `sd` (candidate
the round-1 sources, test the caller's relation H-u, H-exact or H-FIPS) and at `X` (candidate all seven groups, test the PIOP
relation), Akita's opening suffix (Proposition 10.44) after the root fold, and Lemma 12.3 for
the tied parts' own challengers, which precede `X` with addresses that omit it, for the
sites of branch S (K23) and for the fold coordinates, whose addresses omit the parts (K25). (3) Item (vii) of Akita Definition 10.16: the bridge seed
with the sponge lists and the tied parts' sites (tie, `eq_Y`, chain rounds) with the list
factor and Flock's per-site bounds under (F) on the segment of `sd` (12.4), with Lemma 12.4
for the seed; Lemma S and the evaluation batch on the segment of `X` (12.3). (4) Every address weighs at most `2^-128` (Section 3,
the script's assertion `max weight <= 2^-128`; `sd` at most `2^-131.38`; `proj.psi`
`2^-129` by Lemma P and its one-bit grind (K24), below the field sites; the largest PIOP
plan weight is `2^-128.0931` (`tau`, `15/(16|K|)`) at `N* = 32` and `2^-128.36` at `2^15`;
so every PIOP site is at most `2^-128.09`; the heaviest quantified site is the tied-part
tie site, `2^-128.034` under `Slim` (`nu_s = 7`) and `2^-128.063` under `Fast`; Akita's
fold-coordinate weight is `2^-128.0881` (4.6); Flock with 4.3/4.4
under (F); Akita by its plan and by the accepted sparse-challenge supports of 4.6), so
Akita Theorem 10.17 with Aerie Eq. (18) gives `rho (Q + R_f + R_coord + R_F)` with `rho <=
2^-128`, and no site sits at exactly `2^-128` except where (F) is the bound. (5) The witness map returns the round-1 tables, which satisfy the caller's relation by the
test at `sd` (spec 12.2: lookup, leaves, the Booleanity zero-checks of `B_zhi` and `a_z`
under H-exact, the projection bound on `v` through Lemma P, the bridge with `b = b'`).

*What is not proved here* (Section 12.7): assumption (F), which the Flock paper presents
as a sketch; assumption (K); Lemma 12.3 extends Akita's Theorem 10.17, now carries PIOP
stages 1 and 2 as well (branch S, K23), has had internal review only and should be checked
by a reviewer of Akita's proof; `R_F` is not counted exactly;
Akita's Module-SIS estimates exclude the extractor's run count and the cost of one run as in
Aerie, although the Module-SIS algorithms built from the collisions inherit `T_run` (12.6).
As in Aerie, the term
`(R_f + R_coord + R_F) 2^-128` is not negligible at `Q = 0` (about `2^-117` to `2^-118`,
estimate); the claim is 128 bits per query, Aerie's notion.

*Mode B* (public bits, no Flock): Corollary 12.5 gives the same statement without (F).

## 8. Costs at `N = 2^15` (profile `Slim`, span on)

| Item | Proof bytes | Prover hashes (expected) |
| --- | ---: | ---: |
| PIOP packed stream (456 nonces, 4,203 bits) | 526 | 2,646 Blake2b predicates (about 2 ms serial, estimate) |
| Tied-part nonces (version 5: tie `g = 10`, `eq_Y` `g = 10`, 28 rounds `g = 7`) | 120 | `2^10 + 2^10 + 28 x 2^7 = 5,632` BLAKE3 (well under 1 ms, estimate) |
| Akita (already in the opening) | 0 added | 0 added (9,884 at `2^15` with K24) |
| Flock fix of 4.3 | 0 (nonces are 8-byte words already) | about 100 sites per instance at `2^7` to `2^14` (estimate, milliseconds) |
| Plan digest in event 0 | 0 (absorbed, not sent) | one SHA3-256 |
| **Total added** | **645 B = 0.020 B/sig** | well under 0.1 s (estimate) against 23.7 s |

**Measured (local, direction only).** `agg bench --count 1024 --keys 16
--warmup 1 --reps 5`, CLMUL build (`RUSTFLAGS=-C target-feature=+pclmulqdq`, separate target
dir), `RAYON_NUM_THREADS=3`, `taskset -c 0,1,3-15`, shared machine (load average 5.2 to 5.8),
the three binaries run back to back, `Slim`, one tied part:

| Binary | Commit | Prove (median of 5) | Flock parts | Aggregate | PIOP nonce stream |
| --- | --- | ---: | ---: | ---: | ---: |
| before | `7e635a6` (code `be398d8`) | 4,035.1 ms | 2,189.2 ms | 522,076 B | none |
| PIOP and tie work, no Flock fix | `a5f23da` | 4,134.4 ms | 2,203.9 ms | 522,435 B | 391 B |
| PIOP and tie work with K19 | `ed966a6` | 4,081.7 ms | 2,184.7 ms | 522,465 B | 391 B |

Bytes: +391 B nonce stream and +4 B tie nonce, as Section 10 predicts (395 B at `2^10`); the
Akita opening varies by a few bytes between runs (inline LEB128 nonces). Prover time: the
differences (+1.2% and +2.5%) are inside the run-to-run spread on this shared machine (the
minimum of each set: 3,898, 4,025, 4,000 ms), consistent with the estimated few milliseconds
of hashing; the Flock fix adds no measurable time. Verify medians 442 / 434 / 425 ms
(uncached keys).

Span off at `2^15`: four tied parts of 26 rounds each, `526 + 4 x 112 = 974` B in all.

## 10. Script output summary (`python3 -I docs/scripts/pow_plan.py`)

(Relation H-exact: Section 10.1.)

Relation H-u (stage degrees 4, 4, 3; batching vectors 26, 9, 48). The relation is bound by
the event-0 manifest bytes (`G_main4` alphabet code 6 for `U_hi`), not by the plan; with the
projection of `v` (K24) the plans of H-u, H-exact and H-FIPS are equal at every size.

PIOP sites per `N*` (`bits` = packed-stream bits, `work` = `sum 2^g`, ledger = round-by-round
sum before work):

```text
d_sig  N*      d_main  D   L            X-site log2   sites  nonces  bits  bytes  work(sum 2^g)  est.ms  max weight  ledger(before work)
    3       8      13  16       153600        -220.5    196     195  1818    228           1350    0.98  2^-128.30  550/|K| = 2^-118.9
    4      16      14  16       172032        -220.3    199     198  1848    231           1386    1.00  2^-128.19  562/|K| = 2^-118.9
    5      32      15  16       208896        -219.9    202     201  1878    235           1422    1.03  2^-128.09  574/|K| = 2^-118.8
    6      64      16  17       282624        -219.2    223     222  2070    259           1532    1.11  2^-128.42  636/|K| = 2^-118.7
    7     128      17  18       430080        -218.2    245     244  2270    284           1630    1.18  2^-128.42  701/|K| = 2^-118.5
    8     256      18  19       724992        -216.8    268     267  2479    310           1732    1.25  2^-128.42  769/|K| = 2^-118.4
    9     512      19  20      1314816        -215.2    292     291  2698    338           1846    1.34  2^-128.42  840/|K| = 2^-118.3
   10    1024      20  21      2560000        -213.4    317     316  2926    366           1964    1.42  2^-128.42  914/|K| = 2^-118.2
   11    2048      21  22      5050368        -211.4    343     342  3163    396           2086    1.51  2^-128.42  991/|K| = 2^-118.0
   12    4096      22  23     10031104        -209.5    370     369  3409    427           2212    1.60  2^-128.42  1071/|K| = 2^-117.9
   13    8192      23  24     19992576        -207.5    398     397  3664    458           2342    1.70  2^-128.42  1154/|K| = 2^-117.8
   14   16384      24  25     39985152        -205.5    427     426  3929    492           2492    1.80  2^-128.42  1240/|K| = 2^-117.7
   15   32768      25  26     79970304        -203.5    457     456  4203    526           2646    1.92  2^-128.36  1329/|K| = 2^-117.6
```

Sites of one tied part, version 5, full parts (`n = 2^d S`), selected rows (the script prints
every `d = 3..13`; numerators over `2^128`):

```text
profile  d   S  nu_s  list  tie=L(nu_s+3)  g   eq=L(12+s)  g_eq  round=2L  g_rd  rounds  nonce bytes  work(sum 2^g)
fast     3   1     4    35            245   8         420     9        70     7      16           72           2816
fast    13   1    13    35            560  10         420     9        70     7      26          112           4864
fast    13   4    15    35            630  10         490     9        70     7      28          120           5120
slim     3   1     4    50            350   9         600    10       100     7      16           72           3584
slim    10   1    11    50            700  10         600    10       100     7      23          100           4992
slim    13   1    13    50            800  10         600    10       100     7      26          112           5376
slim    13   2    14    50            850  10         650    10       100     7      27          116           5504
slim    13   4    15    50            900  10         700    10       100     7      28          120           5632
slim8   13   4    15    70           1260  11         980    10       140     8      28          120          10240
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
parts in one segment, refused by mode A, would give `2^-127.71` under `Slim`). Per-instance ledger vs
round-by-round at `2^15`: `3,429/|K| = 2^-116.3` vs `1,329/|K| = 2^-117.6` (stages 275,
`tau` 25, gammas 3, GKR 1,026). Honest failure per site `< 2^-184.6`, per proof
`< 2^-174.5` (1,158 capped searches: PIOP, 240 tied-part, 461 Akita; with the Flock sites of
the canonical nonce width `< 2^-171.5`, Section 6.5).

Totals with the tied parts' nonces (`N = N*`, `Slim`): `2^10` 466 B (0.455 B/sig), `2^13`
570 B (0.070 B/sig), `2^15` 646 B (0.020 B/sig) with the span, 974 B with span off.
Expected tied-part work at `2^15`, span on: 5,120 /
5,632 / 10,240 BLAKE3 evaluations (`Fast / Slim / Slim8`).

Plan digests (Section 6.4 encoding, `N = N*`, default part size; the script prints all 13
sizes and every profile, these are the anchors the implementation tests MUST reproduce):

```text
N* = 2^3  modeB          c2c88a319a289cce20146a0e8fb87d8203bb9f02a2b0743974290ef94c229ce5
N* = 2^3  slim   span on  9757c063ae3b3a704330367ea2552dcfd41c9f20841abccce47a9a720cf69437
N* = 2^10 modeB          aa465458415c67707b10cc7e91ddb1eac2f8a4182edf90db1d14f3e74107ecbd
N* = 2^10 slim   span on  0aa41fd0c4c2a9bd0df0b63d86e57787cb31c6d7cedec0ac9edd626cd2fb3eea
N* = 2^13 slim   span on  aa72651e90b32c908db52ec832790d72e685c4493c7fd3e80bebe60cd506b5c9
N* = 2^15 modeB          1634c0b863ed46dd376f873caf5dafaa838fb4a3cda274e4be7c3eca1f8665e2
N* = 2^15 fast   span on  deb090a48d8057c6aed064b95bf6fb8a1499207935005fc0d98b40ab5de3864c
N* = 2^15 slim   span on  ef9710959f30286f422e170d9112bca67e7526e17382660675061256ef3e3303
N* = 2^15 slim   span off fe42263b92bff80e6c367e2b44d7673d274f454841e1784332f5971a03e136c8
N* = 2^15 slim8  span on  0ba687fbd3b201e24c3776d9ee689dfcfb81ea841de12aa7cbbb3157199ccad8
```

The digests depend on the tied parts' difficulties and round counts of Section 3.6 and the
Flock profile (the mode-B digests have no tied part); if a
difficulty changes, the script and these anchors change with it.

### 10.1 Relation H-exact (`python3 -I docs/scripts/pow_plan.py --relation h-exact`)

Relation H-exact (`docs/h-exact.md` step B, relation byte 1) has the same stage degrees
(4, 5, 3) and the batching vectors `gamma^(1..3)` of 26, 9, 50 coefficients: stage 2 holds
`def_y1[t]`, one `leaf6[D_u]` (the Boolean `B_zhi` needs no leaf) and `leaf8[K_2]`,
`leaf8[K_3]` (K24), stage 3 three links and the two Booleanity zero-checks `bool[B_zhi]`,
`bool[a_z]` at `r^(2)`.
The zero-checks add no site: they are identities tested at a point drawn by the preceding
stage (3.4), pending during the stage-2 rounds, and enter through `gamma^(3)`.
`bit_length(9) = bit_length(10) = 4` and `bit_length(49) = bit_length(50) = 6`, so every
difficulty, the stream, the work and the plan digest are H-u's (before the projection the
stage-2 site had 7 coefficients and `g = 3`, so the plans differed); the round-by-round
ledger is H-u's (it depends on the stage degrees only). The plan encoding (6.4) has no
relation field: the relation is bound by the manifest bytes of the same event 0 (byte 3 and
the `G_main4` descriptor); equal digests are benign, as for Akita's rows of `d_sig` 3 and 4
(4.1).

```text
d_sig  N*      sites  nonces  bits  bytes  work(sum 2^g)  ledger(before work)
    3       8    196     195  1818    228           1350  550/|K| = 2^-118.9
   10    1024    317     316  2926    366           1964  914/|K| = 2^-118.2
   13    8192    398     397  3664    458           2342  1154/|K| = 2^-117.8
   15   32768    457     456  4203    526           2646  1329/|K| = 2^-117.6
```

The largest PIOP site weight is H-u's (`2^-128.09` at `N* = 32`, `2^-128.36` at `2^15`,
`2^-128.42` from `2^6` to `2^14`); exactly, `tau` weighs `15/(16|K|) = 2^-128.0931` at
`N* = 32`, so every PIOP site is at most `2^-128.09`. The heaviest quantified site is the
tied-part tie site, `2^-128.034` under `Slim` (`nu_s = 7`) and `2^-128.063` under `Fast`;
Akita's fold coordinate weighs `2^-128.0881` (4.6). No site sits at exactly `2^-128`
except where (F) is the bound.
Plan digests (`N = N*`, default part size; pinned by
`agg_piop::pow::tests::h_exact_plans_reproduce_docs_h_exact_section_3_7`, which also checks
that H-u's equal them): those of Section 10.

The theorem of Section 7 holds verbatim for H-exact with "relation H-u" replaced by
"relation H-exact" and Akita's plan of the H-exact grouped row (4.1); spec 12.2 gives the
extraction of the exact bounds.

### 10.2 Relation H-FIPS (`python3 -I docs/scripts/pow_plan.py --relation h-fips`)

Relation H-FIPS (`docs/h-exact.md` Section 10, relation byte 2, deviation L10) keeps the stage
degrees (4, 5, 3) and has batching vectors of 26, 9, 56 coefficients: stage 2 holds
`def_y1[t]`, `leaf8[K_2]`, `leaf8[K_3]` (K24) and the degree-3 zero-check
`F3 = e (e - 1)(e - delta)` at `r^(1)` (degree 4 with `eq`), stage 3 the ring identity,
`link[K_2]`, `link[K_3]`, the two sides of the
projection identity, the Booleanity of `bz_hi`, `l_hi`, `a_z` and the sign bit
`s`, `F2 = delta (delta - 2 s + 1)`, `F45 = 2 e w1' + 43 e^2 - 43 e` and the hint-weight
identity `F7` at `r^(2)` (`F7` at the point `(1/2, ..., 1/2, r^(2)_sig)`, two parts on the main
domain and `D_2`), the 37 reductions, the reductions of stage 2's `delta(r^(2))`, `e(r^(2))`
and the five stack reductions. As for H-exact (10.1), the zero-checks add no site: each tests
tables fixed before its point (`s, delta, e, l_lo, l_hi` in round 1, `u_h` in round 2, both
before `X`, `tau` and the stages; 3.4), `F3` is pending during the stage-1 rounds that draw
`r^(1)`, the others during the stage-2 rounds that draw `r^(2)` (`F7`'s identity
`sum_i eq(r^(2)_sig, i) G(i) = 0` is multilinear in `r^(2)_sig`; its `1/2` coordinates only
sum over positions and carry no randomness). `bit_length(9) = 4` and `bit_length(56) =
bit_length(50) = 6`, so every difficulty, the stream (4,203 bits, 526 bytes, 2,646 units of
work at `2^15`), the round-by-round ledger (`1,329/|K| = 2^-117.6` at `2^15`) and the plan
digest are H-exact's (and H-u's) (`agg_piop::pow::tests::h_fips_plans_equal_h_exact_plans`; the
`pow_plan.py` self-check). Equal digests are benign, as for Akita's rows of `d_sig` 3 and 4
(4.1): the relation is bound by the manifest bytes of the same event 0 (byte 3, the `G_main4`
descriptor at 12 polynomials with the ternary codes of `delta` and `e`, the `G_2` descriptor
at 13 polynomials, the `p16f` catalog row, the lookup layout with 7 columns in stack 2).

Extraction (spec 12.2 with `docs/h-exact.md` 10.3, 10.4): the lookup gives `l_lo, u_h in
[0, 2^16)` and `bz_lo` as for H-exact; the stage-3 Booleanity checks give `s, l_hi, bz_hi, a_z`
bits; `F2` gives `delta in {0, 2 s - 1}`; `F3` gives `e in {0, 1, delta}`, hence `F45` is
`e (e + 1) w1' = 0` and `e (e - 1)(w1' - 43) = 0`; `F7` gives `sum_p delta_p^2 + sum u_h = 80`
over the integers (every term is a small non-negative integer). With (3.1) for
`u = gamma_2 s + L + 1 - gamma_2 - 2 gamma_2 delta - e` and (3.3), the round-1 values of every
live record are an H-FIPS witness, and the exhaustive lemma of `docs/h-exact.md` 10.3 turns them
into a FIPS 204 signature (10.4). The extracted signature is a function of the round-1 values
alone (`u_h` only certifies the weight), so the checkpoint argument of Section 12 (round-1
tables fixed before the first fork) applies unchanged. The theorem of Section 7 holds verbatim
for H-FIPS with "relation H-u" replaced by "relation H-FIPS" and Akita's plan of the H-FIPS
grouped row (`agg-pcs-config` `tests/all/akita_grinding.rs`, `PINNED_H_FIPS`); 58 Akita polynomials
(39 claimed at `r^(3)`, 12 zero polynomials of `G_3`, 7 at `pi_m`) in place of 52.

### 10.3 ML-DSA-65 and ML-DSA-87 (`python3 -I docs/scripts/pow_plan.py --set 65|87`)

The lane sets (relation H-u only) follow the same per-site rule (`2^-128` per site after work,
Aerie's convention). Their registry differs from ML-DSA-44's in four places: `d_main = 8 +
d_sig`; the lookup has seven (65) or eight (87) stacks: the `l + k` looked-up `Z_lo`/`U_lo`
lanes split as `[8, 2, 1]` or `[8, 4, 2, 1]`, the carries of levels 1 and 2 with four columns
each (`d_1 = 13 + max(0, d_sig - 8)`), `kappa^(3)` and `[K_0 | K_1]` on `D_k` (87: two row
blocks, `d_k = 8 + d_sig`); stage 1 has degree 3 (no six-bit range identity); and the batching
vectors have 37, 18, 105 (65) and 45, 22, 132 (87) coefficients (stage 3 counts the `k` ring
instances, the links and the Booleanity checks of the slack bits per lane), so `stage.gamma`
grinds 6, 5, 7 (65) and 6, 5, 8 (87) bits. The `Psi` site keeps its one bit at 24 and 32 KiB.
The per-lane ring identities add no site: they are batched by `gamma^(3)` like any other
stage-3 instance. Totals of the PIOP part and the mode-B digests
(`agg_piop::pow::tests::ml_dsa_65_and_87_plans_reproduce_the_script`):

| Set | `N*` | sites | nonces | bits | bytes | `sum 2^g` | plan digest (mode B) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 65 | `2^3` | 190 | 189 | 1,761 | 221 | 1,690 | `ce924fd9...` |
| 65 | `2^10` | 311 | 310 | 2,867 | 359 | 2,356 | `31e84bb3...` |
| 65 | `2^13` | 392 | 391 | 3,605 | 451 | 2,770 | `97bd51e1...` |
| 65 | `2^15` | 451 | 450 | 4,144 | 518 | 3,130 | `6ff0953b...` |
| 87 | `2^3` | 190 | 189 | 1,763 | 221 | 1,850 | `7882abb9...` |
| 87 | `2^10` | 311 | 310 | 2,873 | 360 | 2,604 | `c536004a...` |
| 87 | `2^13` | 392 | 391 | 3,611 | 452 | 3,018 | `f7778510...` |
| 87 | `2^15` | 451 | 450 | 4,149 | 519 | 3,346 | `b4385bc4...` |

The mode-A digests the script prints for 65 and 87 assume the bridge of the lane sets (four
chain lanes, 256 records per level-1 block, `P_pad = 2^11`, so a full part has
`19 + max(0, d - 8) + 2 + log2 S` chain rounds); they are pinned once `agg_bridge::pow` takes
the set's shape.

## 11. Open questions

1. **Flock outer sites (4.3).** The vendored core is patched (K19); whether the outer
   sites have an argument that avoids the list factor remains to be raised with the Flock
   authors.
2. **Tied-part difficulty.** 8 to 11 bits at the tie and `eq_Y` sites and 7 or 8 bits per
   chain round with the list factor (Section 3.6). An alternative that avoids the list would bind the sponge's
   level-0 list before the tie challenger: two explicit OOD samples (one alone binds only to
   about 113 bits at `L = 50`, by Flock's `paper_ood_bits`), i.e. extra MLE evaluations of the
   committed vector and a Flock transcript change. Grinding costs well under a millisecond
   and is recommended.
3. **Composition proof (written, Section 12).** No second extractor is needed; the proof
   uses checkpoints at `sd` and `X`, the Merkle and branch-digest expansions (Lemmas 12.2,
   12.1b) and Lemma 12.3 for the parallel Flock transcripts and for branch S of the forked
   PIOP transcript (K23: 53 core PIOP sites at `2^15`, `tau` through the stage-2 rounds). It
   rests on assumptions (F) (Flock's round-by-round analysis, a sketch
   in the Flock paper) and (K) (circuit correctness), and Lemma 12.3 extends Akita's Theorem
   10.17 and has had internal review only; these three items remain for the review and the
   paper.
4. **`R_F`** (Flock sites read by the verifier) is not counted exactly here; it enters only
   the `Q = 0` term.
5. **Mode B** uses the same PIOP plan with profile id 3 and no tied-part site; its Keccak side is
   the public bits, so no Flock term. Its soundness statement is Corollary 12.5 (no (F),
   `sd` at most `theta' = 2^-143.64` per charged segment).

## 12. Composition proof (hypothesis H2)

This section proves the theorem of Section 7 from Akita's checkpoint extraction (Akita
Definition 10.16, Theorem 10.17), its opening suffix (Proposition 10.44) and the arguments
of its Theorem 10.45, in the style of Aerie Section C. Akita's Theorem 10.45 cannot be applied
verbatim, for three reasons: the round-2 root commitments (event 4) are sent after the first
application challenge `sd`; the Flock words are committed by Merkle caps, not by Akita; and
the tied parts run on their own challengers, in parallel, so their challenge inputs do not
encode the complete preceding transcript. The late join (deviation K25) adds a fourth:
neither the PIOP stream nor Akita's stream absorbs the Flock proofs, so `X`, every later
PIOP challenge and every Akita challenge are drawn without them, and the parts' stages
precede `X` in the analysis although `X`'s address omits them. The forked PIOP
transcript (deviation K23) adds a fifth: after event 6 the PIOP stream forks into branch L
(the lookup) and branch S (`tau`, stages 1 and 2), so the inputs of the 53 sites of S at
`2^15` omit branch L, and the trunk encodes both branches only through two 32-byte digests
`d_L`, `d_S`. Sections 12.1 to
12.4 build a checkpoint structure for the expanded protocol that handles all five: the
tied parts' stages are plain stages of the segment of `sd`, with projected addresses, and
their claims are settled by their own verifiers before `X`; the sites of S are plain stages of the same segment with
projected addresses (Lemma 12.3), and the branch digests are expanded (Lemma 12.1b). Section 12.5 lists exactly what is assumed, 12.6 states
the bound, and 12.7 lists what is not proved. Section 12.8 uses the theorem for aggregate
unforgeability (spec 12.3): signing queries asked only in rewound runs, by strong
unforgeability of the relaxed scheme.

The key observation is that no second extractor runs. The relation (H-u, H-exact or H-FIPS,
written `H` below for the caller's relation; for H-FIPS the weight condition is read as
`sum_p delta_p^2 <= 80` on the round-1 values, the round-2 `u_h` being only its certificate,
Section 10.2) constrains the round-1
polynomials alone ((3.3) is a SHAKE256 computation on their bits `b`), so the extractor
returns Akita's round-1 sources and never extracts a Flock witness. The Ligerito words are
fixed straight-line by the Merkle expansion (Lemma 12.2), and the round-by-round bits
quantify over their Johnson lists. They enter no test: a part's claim becomes its Flock
verifier's verdict inside the segment of `sd` (12.4). Every test is a polynomial-time
function of Akita candidates and the checkpoint's address; the extractor runs no decoder.

### 12.1 Oracles and the expanded protocol

Oracles (classical ROM, Section 1): the PIOP duplex `H_P` (Spongefish `Blake2b512`), the Flock
challengers `H_F` (BLAKE3 `FsChallenger`: per tied part the tie challenger, which also runs the
parity chain, and the sponge instance's challenger, distinct labels), Akita's native stream and coordinate streams `H_A` (SHAKE, ideal
XOF as Akita 10.1), the bridge expansion `H_X` (cSHAKE256 `agg-mldsa/v1/bridge`, spec 9.1), the
setup oracle, the hashed-entry oracles (public-records tree, plan digest, PCS session digest,
part and span seeds, the event-6 record digest `rec` that they bind (K22), final-vector
digests `y_digest`, tie bindings, Flock statement digests, the BLAKE3 chaining inside
`FsChallenger`), and the Ligerito Merkle hash `H_M` (BLAKE3, 256-bit nodes, caps). The
branch digests `d_L`, `d_S` of the forked PIOP transcript (K23) are `H_P` answers that the
trunk absorbs as compressed prefixes, like the `fs` chunks in `H_F` inputs. `Pi_H` is the real protocol. The expanded protocol `Pi` differs in four ways, and item 5
records a property it shares with `Pi_H`:

1. every hashed entry is sent in full (Akita Lemma 10.47; error `eps_hash = eps_bind + eps_pk
   + eps_dig`, Aerie Definition 5.8, with the entries listed above);
2. every `H_F` input that contains a Flock seed `fs_sigma` (through the part or span seed)
   contains instead the PIOP input of the `fs` site, i.e. the complete PIOP prefix through
   event 5 (Lemma 12.1);
3. every Ligerito Merkle cap (level 0 and every recursive level of every instance) is replaced
   by the committed word, and every authentication path by a read of that word (Lemma 12.2);
4. every `H_P` input from the branch join on (the trunk) contains, in place of `d_L` and
   `d_S`, the inputs of the branch-final queries that produced them, i.e. the complete inputs
   of branches L and S (Lemma 12.1b);
5. no `H_P` input and no `H_A` input contains a Flock proof byte (K25): the parts' messages
   enter addresses of `H_F` only.

In `Pi` every trunk input from `gamma^(3)` on, and through it the PCS session digest,
encodes both branches in full (item 4) and omits the Flock proofs (item 5). Every challenge address of a part encodes the PIOP prefix through `fs` and, through
`rec` in its part or span seed, the event-6 record (every `Y_sigma` and every share item, as
absorbed): together, the input of `X`. Every committed Flock word, hence its Johnson list
`Lambda(C) = Lambda_gamma(C)` at the profile's level-0 radius (`|Lambda| <= L = 35, 50, 70`,
Flock Eq. (10)), is a function of the transcript.

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

**Lemma 12.1b (the branch digests as compressed prefixes).** A prover for the protocol with
items 1 and 2 converts into a prover for the protocol with items 1, 2 and 4 that runs it
once, outputs the same statement and makes no more challenge queries, such that the first
accepts while the second rejects with probability at most

```text
eps_br = Q_B^2 / 2^257 + Q_J Q_B / 2^255 + Q_B / 2^191  <=  5 Q_P^2 / 2^257 + Q_P / 2^191,
```

where `Q_B` counts the `H_P` queries at branch-final inputs (the complete input of branch L
or of branch S, whose first 32 output bytes are `d_L` or `d_S`), `Q_J` the `H_P` queries
whose input contains a join payload `d_L || d_S` (the trunk from the join on, the PCS
session digest's input included), both counting the verifier's queries on the output, and
`Q_P >= max(Q_B, Q_J)` all `H_P` queries.

*Proof.* The proof of Lemma 12.1 with the 32-byte branch digests in the role of the `fs`
chunks: the converted prover forwards a trunk query with each digest replaced by the
branch-final input that produced it, and answers a query that contains a digest never
produced by a fresh local answer. A query that continues the prefix through event 6 with
one of the two 24-byte branch labels is treated as a branch query and forwarded unchanged.
Its bad events are (i) two distinct branch-final inputs
whose answers have the same first 32 bytes, at most `(Q_B choose 2) / 2^256 <= Q_B^2 / 2^257`;
(ii) a trunk query (or the output) that contains, at one of its two digest positions, a
32-byte value that a branch-final query outputs only later: each fresh branch-final answer
hits one of at most `2 Q_J` such values with probability at most `2 Q_J / 2^256`, so at most
`Q_J Q_B / 2^255` in all; and (iii) a branch-final answer whose first 24 bytes equal one of
the two labels, so that a trunk input carrying it as `d_L` would read as a branch input
(the join absorbs no label of its own, 12.2): at most `2 Q_B / 2^192 = Q_B / 2^191`. Without
them the forwarded queries are injective, trunk and branch queries are told apart, and the
verifier of the second protocol reads the same answers. The sum is `eps_br`, and
`Q_B, Q_J <= Q_P` gives `(1 + 4) Q_P^2 / 2^257 + Q_P / 2^191`. Its quadratic part has the
form of `eps_fs` (one chunk per answer, two per trunk input). Equal digests `d_L = d_S` on one transcript are a case of (i),
since the two branch-final inputs differ in their labels.

**Lemma 12.2 (Merkle expansion).** A prover `P'` for the protocol with items 1, 2 and 4
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
| `proj.psi` (event 3, after `sd`; K24) | `H_P` | `N` |
| `fs` (event 5) | `H_P` | `N` |
| per tied part `p`, after event 6 and before `X`, in segment and part order: the tie site, the `eq_Y` site and the `R` chain rounds (tie challenger); the sponge instance's Flock sites | `H_F` | `N` |
| `X` (event 7) | `H_P` | `H`: checkpoint `l_2` |
| the PIOP sites after `X` (Section 3.1), through the stage-3 rounds: in linear order the rest of branch L, branch S, then stage 3 after the branch join (K23) | `H_P` | `N` |
| Akita's evaluation batch of the 52 claims (53 under H-exact, 58 under H-FIPS, the zero claims of `G_3`'s zero polynomials included; the only Akita site before the root fold, `GrindingSite::EvaluationBatch { level: 0 }`), after stage 3 and the PCS session digest | `H_A` | `N` |
| the opening suffix from the root fold challenge (Akita Proposition 10.44): fold challenges, ring-switch `alpha` of every nonterminal level, the rest | `H_A` | `E` (indexed), `H`, `N` |

The segment of `l_1` runs from `sd` to the last stage before `X`: `sd`, `proj.psi`, `fs` and
the tied parts' stages, with the messages of events 4 and 6 and the parts' messages. The segment of `l_2` runs from `X` to the last stage before the
root fold challenge, whose input is `xi_f`: the PIOP sites and the
evaluation batch. Every plain stage lies in one segment. The tied parts' stages precede `X`
although `X`'s address omits them: Lemma 12.3 (iii) lets the test at `X` and `D` on its
segment read `X`'s address `a_X` (the prefix through event 6 followed by L's label) and
the candidate only. They cannot follow `X`: under K25 no
address after `X` (the branches, the trunk, the session digest, Akita's stream) encodes
them, so a part stage after `X` would violate (i).

With projected addresses (Lemma 12.3) the tied parts' stages
precede `X`, analysed after event 6, where the verifier also checks the Flock
sub-proofs (spec Section 11, step 3). The prover may choose the part's messages after seeing `X`,
so a rerun from `X` may change them; the proof therefore lets the checkpoint test and `D` read the
address only. The Flock verdict is a check of the segment of `sd`, which ends at `X`, so `D` at
`X` reads `a_X` and the candidate only. The test at `X` then needs no list decoder. Two
statements are made explicit in the paper's proof and apply here:
the extraction stages satisfy Akita's Lemma 10.14 conditional-answer sampling hypothesis
(Proposition 10.44 for the opening suffix; the application segment has no extraction stage), and
the verifier's `V_l` stage-`l` addresses enter only Claim 2's count and the success recurrence,
which charges the at most `Q+R` queried addresses once; the run-count recurrence is Akita's.
Together these give `(Q+R+2) K_hat` runs and `(Q+R) 2^-128`. The union of Lemma 12.4 over a
batch's `S <= 4` segments adds at most two bits to the exponent and keeps
`delta_X < 2^-(2^110)`.

**Projected addresses.** In this linear order the stages of part `p` follow event 6 and
those of part `p - 1`, and precede `X`, but the input of a part's site (its address) encodes the
PIOP prefix through `fs` (item 2), the event-6 record (through `rec`, item 1) and part `p`'s
own messages only (the sponge instance encodes the tie challenger's messages only through the
tie binding); it encodes neither `X` nor any later site nor other parts' messages. The
sites of branch S of the forked PIOP transcript (deviation K23) are the second case: in the
linear order they follow the whole of branch L, but their input is branch S's sponge, which
encodes the prefix through event 6, the label of S and S's own messages and answers, and not
branch L. Akita's framework assumes that a stage input encodes the complete preceding
transcript. Lemma 12.3 removes that assumption for plain stages and checkpoints and weakens
it for extraction stages.

**Lemma 12.3 (projected addresses).** Take a checkpoint structure whose stages `j` read
their answer at an address `pi_j(xi)` of the stage input `xi` (the complete linear prefix),
with items (i) to (viii) of Definition 10.16, such that (i) for every checkpoint or
extraction stage `l` and stage `j >= l` the map from a stage-`j` address to a stage-`l`
address is well defined; (ii) the address of an extraction stage `r` encodes every message
and answer read by `Comb_r`, by the checks of stages `r` and later, and by the maps `Cand`,
`Good`, `Coll` of every checkpoint at or after `r`; a stage it omits is a plain stage of an
earlier segment, and `Comb_r` returns the prefix of its first child; (iii) at a checkpoint
`l` the test and `D(xi_l; .)` depend on `xi_l` through its address `a_l` only, and `Coll_l`
applies to certificates with equal addresses; (iv) at a plain stage `j`, `Bad_j(xi; w)`
depends on `xi` through its address only; (v) the extraction stages satisfy Lemma 10.14's
conditional-answer sampling hypothesis. Then Theorem 10.17 holds, each queried address of
stage `j` weighing `eps_j`, for the extractor that reruns a failed checkpoint with a fresh
table on the addresses that encode its address and keeps an attempt with the same address.

*Proof.* Akita's proof (Appendix E.2) touches the inputs of plain stages only in Claims 1 and
2. Claim 1's path argument is a statement about `D` on the stage inputs of the output
transcript and does not change: some stage `j` of the segment has `D(xi_j; w) = 1` and the
answer `K(pi_j(xi_j))` in `Bad_j(xi_j; w)`. For the probability, sample the table lazily by
address: when `A` first queries an address `a` of stage `j` in `Sub(i)`, the history, `c` and
the fixed `w` determine `a` and, by (iv), the set `Bad_j(a; w)`, while `K(a)` is fresh; a union
bound over the queried addresses gives Claim 1 with `N_{i,j}` counting addresses (at most the
count of inputs). By (i) every address lies in exactly one `Sub(i)`, which is Claim 2.
Checkpoints: `sd` reads the prefix through event 2; `X` reads the prefix through event 6
followed by L's constant label, which omits the parts' stages; by (iii) its test and `D`
read `a_X` and the candidate, and Corollary 10.25 binds two certificates with equal `a_X`
(both rounds' commitments are in it). Extraction stages: a fold coordinate's address is
Akita's stream input, seeded by the session digest, which in `Pi` encodes events 0 to 6,
both branches (item 4) and stage 3, and Akita's own messages; it omits the parts' stages
(item 5), which precede it in the order and are plain stages of the segment of `sd`.
Akita's game at an indexed stage (Appendix E.2) redraws the one entry `(i, u)` with every
other entry fixed; the children share `i` and may carry different part messages, which
`Comb` (Proposition 10.44), the suffix's checks, the ring-switch maps and the tests at `X`
and `sd` do not read. `Comb` returns the first child's prefix with the combined weak
opening, a certificate of Proposition 10.44; the first child's first run is the node's, so
invariant (2) of Akita's proof holds. The rest of the proof is unchanged. A rerun from `X`
keeps the `H_F` table; the prover may then send other part messages at new addresses,
which Claim 1 charges like any other queried address.

The parts' sites lie in the segment of `sd` and satisfy (i): their address encodes the PIOP
prefix through `fs` (item 2), hence `a_sd`, the prefix through event 2. (iv) holds by the
bad sets of 12.4, defined from the address alone. The part and span seeds also bind `rec`
(`agg_cli::mode_a::{part_seed, span_seed}`), so a part's address encodes `a_X` as well; (i)
does not need it, and it binds the part's statement into its challenger at no cost.

**Branch S of the forked PIOP transcript (deviation K23).** The sites `tau`, `gamma^(1)`,
the stage-1 rounds, `gamma^(2)` and the stage-2 rounds are plain stages of the segment of
`X`, and Lemma 12.3 applies to them with its statement unchanged. No new site and no new
difficulty arise, but two things change. The lemma's scope grows: before K23 every PIOP site
after `X` had a complete-prefix input, and now the 53 sites of S at `2^15` (`tau`,
`gamma^(1)`, 25 stage-1 rounds, `gamma^(2)`, 25 stage-2 rounds) rest on it; the lemma has
had internal review only (12.7 item 3). And the branch digests add the term `eps_br` of
Lemma 12.1b to the bound (12.6).

* (i): the address of a site of S is branch S's sponge input. It encodes events 0 to 6
  (the clone is taken there) and S's label, hence `xi_X`, which is that prefix followed by
  L's constant label. So every address of S lies in exactly one `Sub(i)` of `X`.
* (iv): the claims a site of S acts on are the zero-checks and level-2/3 identities on the
  tables committed in events 1 and 4, with weights from `sd` and the `Y_sigma` of event 6
  (`tau`, Section 3.4), and the claims made by S's own earlier messages (the rounds and
  `gamma^(2)`). All of them are determined by the address and the fixed `w`, and so is
  `Bad_t` of Lemma S when defined from the first false claim among them. The path argument
  of Claim 1 needs nothing more: if `D` drops from 1 to 0 at a site of S, every claim of L
  (which no answer of S touches) was already true, so the false claim was one of S's and the
  answer lies in `Bad_t`.
* Masses and the work schedule: each site keeps its bad set, its `nu_t` and its assigned
  bound, and therefore its difficulty `g_t`; the 457 PIOP sites, the ledger
  (`1,329/|K|` before grinding at `2^15`) and the per-address weight `<= 2^-128` of every
  site are those of the linear transcript. The nonce stream lists the sites in the same
  order (L, S, stage 3), so the plan of Section 6.4 and its digest do not change.
* Branch L keeps complete-prefix addresses (it precedes S). Stage 3 and everything after
  it read the trunk, which absorbed both branch digests: the pending claims at the join are
  the union of L's (the five stack claims and the seven digit claims) and S's (the stage-2
  terminal claims, the projection claim `c` and the reductions of 7.5), carried by `gamma^(3)` and the stage-3 rounds
  as in Section 5. A digest is the first 32 bytes of the `H_P` answer at a branch's final
  input. In `Pi_H` the trunk encodes L and S only through `d_L` and `d_S` (compressed prefixes in the sense of Aerie C.1); in `Pi` they are
  replaced by the two branch-final inputs (12.1 item 4), so `gamma^(3)`, the stage-3 rounds,
  the PCS session digest and the Akita stages read inputs that encode both branches in
  full and no Flock byte. The conversion costs `eps_br <= 5 Q_P^2 / 2^257 + Q_P / 2^191` (Lemma 12.1b). Its bad events are
  two distinct branch-final inputs with equal digests, a trunk query that contains a digest
  before the branch-final query that outputs it, and a digest `d_L` that begins with a branch
  label. `R_coord` (Akita's sparse fold coordinates) is unrelated to them.
* The join absorbs no label of its own: `agg-mldsa/v1/fork/join` is a registry name, and
  neither registry names nor event kinds are absorbed. The trunk stays apart from both
  branches without one. Every trunk input from the join on is the prefix through event 6
  followed by `d_L || d_S` (64 bytes) and stage-3 data. Every branch input is the same
  prefix followed by a 24-byte label and that branch's events, whose lengths the registry
  fixes. The only branch challenge inputs shorter than the trunk's first one (prefix plus 64
  bytes, plus 4 with `gamma^(3)`'s nonce) are those of `X` and `tau` (prefix plus 24 bytes,
  plus 4 with a nonce). Every later branch input follows a first absorb of at least 64 bytes
  (L: the layer-1 values, 4 elements of 16 bytes per tree; S: the 13 pre-stage claims, 208
  bytes), so it exceeds the prefix by more than 68 bytes. Branch inputs therefore never have
  the length of `gamma^(3)`'s input. A later trunk input could equal a branch input only if
  the first 24 bytes of its `d_L` equalled a label; Lemma 12.1b charges that case as event
  (iii) (`Q_B / 2^191`).

The argument is the one for the tied parts (K22) with the projected stages inside the PIOP.
It places L before S in the linear order: with S first, `X`'s input would omit S's messages
and would not be a complete-prefix checkpoint input. This order is a device of the analysis
and does not depend on the byte order of the join. Each digest sits at a fixed position of
the trunk input and item 4 expands each in place, so any fixed order of `d_L` and `d_S` is
sound. The code fixes `d_L || d_S` so that prover and verifier compute the same trunk
input; a proof joined in the other order gets other stage-3 challenges and is rejected. Note that the linear
transcript also made a retried lookup site re-run every stage of S (their inputs encoded
L); the fork removes that coupling. The bound above never used it: Claim 1 is a union over
queried addresses with the weight of each address, which the fork leaves unchanged.

**Candidates, tests, binding, witness.** Every certificate below the root fold is a
first-stage certificate `y` of Proposition 10.44: a prefix through `xi_f` passing every check
it determines, with a root weak opening `omega_y` of the seven groups; all stages between `l_1`,
`l_2` and the root fold are plain, so the certificate sets there are this set.

* `l_1 = sd`: `Cand(y) = sigma_1(y)`, the source projections of the round-1 groups `G_main16`,
  `G_main4` in `omega_y`, decoding to `f_1 = Dec(sigma_1)`; `Good_{l_1}(y) = H(f_1)`, the caller's
  relation `H` bound by byte 3 of event 0, H-u, H-exact or H-FIPS ((3.1) to (3.3) with that
  relation's proven bounds, spec 3.1 and 12.2; for H-FIPS also the hint constraints of
  `docs/h-exact.md` 10.3 with the weight read as `sum_p delta_p^2 <= 80`, Section 10.2), on
  every live record. Binding: two
  certificates with the same input at `sd` have the same round-1 commitments, so different
  projections give a kernel vector (Akita Corollary 10.25).
* `l_2 = X`: `Cand(y) = sigma(y) = Src(omega_y)`, all seven groups, `f = Dec(sigma)`;
  `Good_{l_2}(y) = R_PIOP(f; a_X(y))`.
  `R_PIOP` is the conjunction of the Boolean-domain identities the
  PIOP proves from `X` on: lookup membership of every looked-up cell and the leaves (4.3), the
  digit reconstructions (4.1), (4.2) (and the bit-slack form (4.3) under H-exact), "`K_0, K_1`
  in `T_16 - 2^15`, `K_2, K_3 in [-4, 3]`, `k` by (4.1) with `C_k = 2^34 + 2^31 + 2^15`, and
  `<Psi_o, v_i> = k_{o,i}` on the projection domain" (with the `Psi` of `xi_X`; K24) in place
  of the former "(6.1) with the range of `v`", `Delta = 0` in `F_p`, Booleanity and (3.4), under H-exact the Booleanity zero-checks of `B_zhi`
  and `a_z` at `r^(2)` (Section 10.1), and (9.2) with the `M_t` of `sd` and the `Y_sigma` of
  `xi_X`. Binding: Corollary 10.25 (both rounds' commitments are in `a_X`).
* `Wit(y) = f_1` on the original domains, i.e. `(z_i, u_i, v_i, b_i)` by (4.1) to (4.3).

All maps are polynomial time. Spec 12.2 gives `R_PIOP(f; xi_X) => C1*(f_1; Psi)` and `M^(sigma)
b_sigma(f_1) = Y_sigma` over GF(2) for every segment, where `A_1(f_1)` is the relation `H` without
(3.3) ((3.1), (3.2), `w_1' in [0, 44)`, Boolean `b`), `A_1^-(f_1)` is `A_1(f_1)` without (3.1),
`C1*(f_1; Psi) = A_1^-(f_1) and Delta(f_1) = 0 in F_p and Prj(Psi, f_1)` with
`Prj(Psi, f_1)`: `|<Psi_o, v_i>|_c <= 2^37` for all `o < 128`, `i < N*` (12.4), and
`M^(sigma) = M_3 M_2 M_1` (spec 9.2). `R_PIOP` no longer implies `A_1(f_1)` deterministically:
(3.1) follows from `C1*` only outside the bad set of `proj.psi` (Lemma P), which the segment of
`sd` charges (12.4).
The tree size is `K = 2^2 Tree0_ck(sch)` (two application checkpoints; Aerie has one). The
structure depends on the batch geometry bound in event 0 (`N`, the profile, the tied-part
geometry through the plan digest, 6.4), as Akita Remark 10.19 permits; every admitted geometry
has `rho <= 2^-128`, so the bound holds with `R_f`, `R_coord`, `R_F` and the tree size
maximized over them.

### 12.3 The segment of `X` (PIOP and the evaluation batch)

For a fixed `sigma` put `D(xi; sigma) = 1` iff a check determined by `xi` fails or a pending
claim is false, with the pending claims of Lemma S (Section 5), starting from `R_PIOP(f;
xi_X)` at `X`. Item (vii)(a) holds by definition: `Good_{l_2}` is the conjunction of exactly these
claims at `xi_X`. Item (vii)(b): Lemma S for the PIOP sites (for branch S with its projected
addresses, 12.2); at Akita's evaluation batch a false claim `v_k != f_k(z_k)` makes the batched
equation false outside one hyperplane of the 52, 53 or 58 independent coefficients (H-u,
H-exact or H-FIPS; mass `1/|K|`, Akita Remark 10.43; assigned `52/|K|`, `53/|K|` or `58/|K|`;
the bit lengths of 52, 53 and 58 are all 6, so `g = 6` for every relation; a zero polynomial's
claim zero is a claim like any other). Item (vii)(c): the tied parts' claims belong to the segment of `sd` (12.4), so at `xi_f`
the only pending claim is the root row `sum_k eta_k v_k = sum_k eta_k f_k(z_k)` (Akita
(202)), which a certificate's weak opening satisfies (Proposition 10.44, witness step), as in
Aerie Lemma C.1 (ii).

### 12.4 The segment of `sd`: the bridge seed and the tied parts

(Tied-part proof version 5. The parts' stages are plain stages of this segment, after event 6
and before `X`, with projected addresses; the PIOP sponge and Akita's stream never absorb the
Flock proofs, K25.)

Fix `sigma_1` and `f_1 = Dec(sigma_1)`. For a tied part `p` with records in the segments
`P^-1(p)`, let `C_s^p` be its sponge word (event 2, in the input of `sd`). For a codeword `z in
Lambda(C_s^p)` (decoded to a witness):

* `Spg_p(z)`: the sponge registry's rows and Flock's witness contract hold (dummy rows zero,
  the constant pin); `Pub_p(z)`: the public region equals the verifier's table `(mu_i, c~_i,
  1, 0)` on live rows. By assumption (K) of 12.5, `Spg_p(z) and Pub_p(z)` give `c~_i =
  SHAKE256(mu_i || w1Encode(b'_i(z)))` for every live record `i` of `p`, `b'(z)` the input bits;
* `Chn_p(z)`: `M^(sigma) b'_{p,sigma}(z) = Y^_{p,sigma}` for every segment `sigma` of `p`, with
  `Y^_p` the part's final-vector table (its share, or the `S` vectors of a span, zero for the
  segments `S..2^span_log`) and `M^(sigma)` the composite of `sd` restricted to the part
  (`docs/bridge.md` 2); no circuit is involved: this is an identity of fixed GF(2) vectors;
* `Val_p(z) = Spg_p(z) and Pub_p(z) and Chn_p(z)`.

The chain (`docs/parity-chain.md` 3, 4): for `eq_Y = eq(r_o, .) eq(r_y, .)`, `<eq_Y, Y^_p> =
sum_x W(x) b''(z)(x)` holds iff `<eq_Y, M b'(z)> = <eq_Y, Y^_p>`, since `W = M^T eq_Y` in
characteristic 2 and `W = 0` on the lanes 6, 7 of `b''` (the sponge's word columns 48..63).
Write `T_0 = <eq_Y, Y^_p>`, and for the round challenges `r_1..r_k` let `s_k^z(X)` be the true
round polynomial of `W~ b''~(z)` (degree 2) and `T_k = q_k(r_k)` the running claim from the
sent `q_k`. The chain's invariant for `z` after round `k` is `Inv_k(z)`: `T_k = sum over the
remaining Boolean variables of W~ b''~(z)` at `(r_1..r_k)`; `Inv_0(z)` is `T_0 = sum_x W b''(z)`.

**Pending claims** at a stage input `xi` (each present once the data it names is in `xi`).
In the segment of `sd`, for its fixed `sigma_1`:

* `C1`: `A_1(f_1)` (from `sd` up to `proj.psi`); from the answer of `proj.psi` on, `C1*`:
  `A_1^-(f_1) and Delta(f_1) = 0 in F_p and Prj(Psi, f_1)` (12.2);
* `C2_sigma` (between `sd` and the messages of event 6): there are `z_p in Lambda(C_s^p)`
  for the parts `p` of segment `sigma` with `Spg_p(z_p) and Pub_p(z_p)` for all `p` and
  `M^(sigma) (b_sigma(f_1) xor b'_sigma(z)) = 0`, `b'_sigma(z)` the segment's bits assembled
  from the parts;
* after `Y_sigma` and the shares (event 6): `C3_sigma`: `M^(sigma) b_sigma(f_1) = Y_sigma`;
  and per part `C4_p`: some `z in Lambda(C_s^p)` satisfies `Val_p(z)`.

Per part `p`, along its own stages (after event 6, before `X`), for the fixed `sigma_1`:

* on the part's tie challenger, for `z in Lambda(C_s^p)` with `Spg_p(z)`: after the tie site
  `C5_p`: some such `z` has `Chn_p(z)` and `z(P_pub) = table(P_pub)` (the public identity at
  `(rho_r, rho_c)`); after the `eq_Y` site `C6_p^0`: some such `z` has the public identity
  and `Inv_0(z)`; after round `k`, `C6_p^k`: some such `z` has the public identity and
  `Inv_k(z)`; after the bit value `v` is absorbed, `C7_p`: some such `z` has the public
  identity and `z_bit(P_bit) = v`, where `P_bit` is the chain's point mapped to the sponge's
  bit address (`docs/parity-chain.md` 4) and the verifier's check `T_R = W~(r) eq(r_rec[nu_s..],
  0) v` holds (a check determined by the input). From the first Flock site of the sponge
  instance on, `C7_p` reads: some such `z` does not fail the instance in the sense of (F),
  with its two claims `z(P_pub) = v_pub` (packed-direct) and `z_bit(P_bit) = v`
  (ring-switched, `SkipPoint::Eq`) among its pending claims once their values are in `xi`.

In the segment of `sd`, `D(xi; sigma_1) = 1` iff a check determined by `xi` fails or a
pending claim is false; at the input of `sd` the one pending claim is `H(f_1)`. At `xi_X`,
`D = 1` iff a check determined by `xi_X` fails (the Flock verifiers' verdicts and the parts'
value checks among them), or `C1*` or some `C3_sigma` is false; no part claim is pending at
`X`, each having turned into its verifier's verdict.

**Item (vii)(a)** holds by definition. **Item (vii)(c):** a certificate in `Y_X` passes the
test `R_PIOP(f; a_X)`, which implies `C1*` (the
lookup and leaf of `K`, (4.1) for `k`, the projection identity and `Delta = 0`; `Prj` follows
from `k in [-2^37, 2^37)`) and every `C3_sigma` (12.2), and its prefix passes every check it
determines, the Flock verifiers' included, so
`D(xi_X; sigma_1) = 0`. **Item (vii)(b)**, stage by stage. The two PIOP sites `sd` and `fs`
read the complete linear prefix, so their bad sets may use all of it:

* `sd`. If `C1` is false only through (3.1), `sd` does not act on it: `Bad_sd` is defined from
  a failure of (3.3) as below, and `C1` stays pending until `proj.psi`. If `C1` is false
  otherwise it stays false. If `C1` holds, (3.3) fails for a live record `i*` of a
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
* `proj.psi` (a plain stage of this segment; its input encodes the candidate's round-1
  commitments, so `f_1` and its `v` are fixed before the answer). From its answer on, `C1` is
  replaced by `C1*`. If `A_1^-(f_1)` fails or `Delta(f_1) != 0` in `F_p`, `C1*` is false for
  every answer (empty bad set). Otherwise (3.1) fails over `Z` for a live record while
  `Delta = 0` in `F_p`, so by (6.1) the first such record `i*` (a function of `f_1`) has a
  coordinate `|v_{i*}(x_0)|_c >= (p/2 - 2^50)/q > 2^38`. `Bad_proj = {Psi : for all o,
  |<Psi_o, v_{i*}>|_c <= 2^37}` has mass at most `2^-128` (Lemma P, Section 3.5; exact for the
  squeezed `Psi`, so one bit of work gives `2^-129` without a density loss; deviation K24).
  Outside `Bad_proj`, `Prj(Psi, f_1)` is false, hence `C1*` is false. If (3.1) holds, `C1*`
  holds with `C1` for every answer, and the false claim (if any) is `C2_sigma*`, which this
  answer does not touch: it stays false. The site is the first entry of the plan and its
  nonce is checked before `Psi` is squeezed; its mass is charged once, for the first bad
  record, not per record.
* `fs`. No bad set: `D` stays 1 for every answer and every message of event 6. If `C1*` or a
  check fails, it persists. Otherwise `C2_sigma*` is false. If `M^(sigma*) b_sigma*(f_1) !=
  Y_sigma*`, `C3_sigma*` is false. Otherwise, if every part `p` of `sigma*` had `C4_p` with a
  witness `z_p`, then `M^(sigma*) b'_{p,sigma*}(z_p) = Y^_{p,sigma*}` for each `p`; the shares
  XOR to `Y_sigma*` (the verifier derives the last share from `Y_sigma*`; a span's table is `Y`
  itself), the parts' records partition the segment and `M^(sigma*)` is GF(2) linear, so
  `M^(sigma*) b'_sigma*(z) = Y_sigma* = M^(sigma*) b_sigma*(f_1)`, which makes `C2_sigma*` true.
  Hence some `C4_p` is false; it reaches that part's first site unchanged (event 6 is a
  message, and no site lies between `fs` and the parts in this order).

This gives item (vii) for `l_1` up to the parts' stages. The masses are those of Section 3.5.

**The tied parts (plain stages of this segment).** Their stages read only their projected address, so
their bad sets are defined from the address alone, for every input, as Lemma 12.3 (iv)
requires; the argument then shows that every other false claim persists for any answer. The
argument is the same whether the prover computes the part's messages before or after `X` and
the Akita opening: its addresses, and hence its bad sets, are the same, and the answers are
fresh.

* The sites of part `p`'s tie challenger (address: the challenger's input, which encodes
  `C_s^p`, `y_digest`, the geometry, the earlier nonces, coordinates and messages, through
  item 2 the PIOP prefix through `fs` with the matrices of `sd` and the part's records, and
  through `rec` in the part or span seed the event-6 record, so the input of `sd`).
  Every bad set below is a union over `z in Lambda(C_s^p)` with `Spg_p(z)` of an event in `z`
  and the address alone; it does not depend on the candidate `f`, on `R_PIOP` or the PIOP
  claims of Lemma S, on `C3_sigma` or on other parts. For persistence: a failed check, a false
  `C1*` or `C3_sigma` and a false claim of another part do not depend on these answers and
  stay false.
  * *Tie site.* `Bad_tie_p`: the `(rho_r, rho_c)` at which some `z` with `not Pub_p(z)` has
    `z(P_pub) = table(P_pub)` (two distinct multilinear polynomials in `nu_s + 3` free
    coordinates agree): mass `<= L_s (nu_s + 3) / 2^128`. Outside it, if `C4_p` is false then
    every `z` with `Spg_p(z)` fails `Pub_p` (then the public identity fails) or `Chn_p`, so
    `C5_p` is false.
  * *`eq_Y` site.* `Bad_eq_p`: the `eq_Y` at which some `z` with `not Chn_p(z)` has `<eq_Y, M
    b'(z)> = <eq_Y, Y^_p>` (two distinct tables of `12 + span_log` variables, multilinear
    extensions agreeing): mass `<= L_s (12 + span_log) / 2^128`. Outside it, a false `C5_p`
    gives a false `C6_p^0` (for every `z` the public identity or `Inv_0(z)` fails).
  * *Round `k`.* `Bad_k`: the `r_k` at which some `z` with `not Inv_{k-1}(z)` (so `q_k !=
    s_k^z`, since `q_k(0) + q_k(1) = T_{k-1}` by construction) has `q_k(r_k) = s_k^z(r_k)`:
    mass `<= 2 L_s / 2^128` (degree 2). Outside it, a false `C6_p^{k-1}` gives a false
    `C6_p^k`.
  * *The bit value* `v` (a prover message, no site): if `C6_p^R` is false and the verifier's
    final check holds, then for every `z` either the public identity fails or `T_R != W~(r)
    b''~(z)(r) = W~(r) eq(r_rec[nu_s..], 0) z_bit(P_bit)` (the sponge's rows beyond `nu_s` do
    not exist and the chain's input is zero there), hence `z_bit(P_bit) != v`: `C7_p` is
    false. `W~(r)` is the verifier's own evaluation of the multilinear extension of `M^T eq_Y`
    (`docs/parity-chain.md` 5).
  With the work bits (Akita Remark 10.18, Proposition 10.3 under (F)(iv) for the BLAKE3
  predicate) every address weighs at most `2^-g e <= 2^-128` (3.6).
* A Flock site `t` of the sponge instance of part `p`: its bad set is Flock's set `B_t` of
  answers after which some codeword of `Lambda(C_s^p)` that fails the instance at the address
  stops failing ((F)(ii); a function of the address alone, for every input; weight at most
  `2^-128` with the scheduled work, K19). Every false claim other than `C7_p` persists for
  every answer. If `C7_p` is false and the answer avoids `B_t`, every `z` with `Spg_p(z)` still
  fails: a failed public identity or a wrong bit value is a false claim of the instance
  ((F)(i), its packed-direct and ring-switched claims) that no message revives ((F)(ii)). If
  the instance's verifier accepts, (F)(iii) gives a codeword `z*` that does not fail, so its
  values are the sent ones, and the verifier's value check (`v_pub` = table) and the chain's
  final check then give `C7_p`. So a false `C7_p` at the end of the part makes the Flock
  verifier or a value check reject, which is a check determined by every later input; from
  there on the part has no pending claim. In particular `D(xi_X; sigma_1) = 1` whenever a
  part's `C4_p` was false and no bad answer occurred.

This completes item (vii) for `l_1`, with Lemma 12.3 (i) and (iv) for the part sites.
The masses are those of Section 3.6.

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
  caller's claim sets once its value is sent and differs from `z`'s evaluation: a
  packed-direct claim on the packed polynomial, or a ring-switched claim on the bit
  polynomial (Flock's own ab and c claims are of this kind; the caller's bit claim of K21
  uses the multilinear skip basis `SkipPoint::Eq`, whose claim check is exact for any linear
  functional, `vendor/flock-core/PATCH.md` Section 7); (ii)
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
  keccak.md` Sections 1, 2: each committed chi output is forced by its own row, round 23 of
  the last permutation commits only the four lanes that `c~` reads, the linear layers, the
  absorption and the chain are XOR expressions) computes SHAKE256 on its input bits. A property
  of a fixed circuit, established by construction and by the crates' tests, not proved here.
  In tied-part proof version 5 the GF(2) side of the bridge has no circuit: `Chn_p` is an
  identity of fixed vectors and the chain's weight `W~(r)` is the verifier's own evaluation
  (tested against dense evaluations, `agg_bridge::parity_chain`).
* **(R) Random oracles** as listed in 12.1 (Aerie's modeling of Spongefish, Flock's of BLAKE3,
  Akita's ideal XOF; the BLAKE3 uses for Merkle nodes, challenger chaining and work
  predicates are independent oracles by their input formats, as Flock assumes), the setup as
  in Aerie C.1 (`Delta_setup = 0` for the exact rejection-
  sampled setup stream of the pinned Akita revision, which Aerie `49b6c21` shares with us).
* **(A) Akita** at `e023b6a`: Proposition 10.44, Lemma 10.42, Corollaries 10.25 and 10.32,
  Lemmas 10.2 and 10.20, the extraction-admissibility of our schedules (Definition 9.2 (i) to
  (iii); the rows pass Akita's schedule audit when the catalogs load, the tree-size condition
  is not tabulated here, see 12.7), and Theorem 10.17, here with Lemma 12.3.

Discharged in this repository: the PIOP round-by-round bad sets (Lemma S), the `sd`, tied-part
(tie, `eq_Y`, chain rounds) and evaluation-batch bad sets above, the work schedule (Sections 3, 6, 10), the fold-challenge
supports (4.6), Lemmas 12.1 to 12.4.

### 12.6 The bound

Let `P` make `Q` queries (every oracle of 12.1, nonce probes included; `Q_wrap = Q + V_ver`
after the conversions of Lemmas 10.47, 12.1, 12.1b, 12.2, as Aerie C.5). Theorem 10.17 with Lemma
12.3, the structure of 12.2 to 12.4, `Good` the event of Lemma 12.4, and the setup,
extraction and Module-SIS steps of the proof of Akita Theorem 10.45 give an extractor that runs
`P` at most `(Q_wrap + 2) 4 Tree0_ck(sch)` times in expectation and fails on an accepting
execution with probability at most

```text
eps_hash + eps_fs + eps_br + eps_merkle + delta_X + E[ sum_{a in Addr} eps(a) ] + sum_I Adv^MSIS_I(B_I)
```

with `E[sum eps(a)] <= rho (Q + R_f + R_coord + R_F)` (Aerie Eq. (18)) and `rho = max eps(a)
<= 2^-128`: PIOP sites `<= 2^-128.09` (Sections 10 and 10.1, max weight, every relation),
`proj.psi` `2^-129` (Lemma P and one bit of work, K24), tied-part sites (tie, `eq_Y`, chain rounds) `<= 2^-128` (3.6; the tie site is the heaviest quantified site, `2^-128.034` under `Slim` and `2^-128.063` under `Fast`), `sd <=
2^-131.38`, `fs` 0, Flock sites `<= 2^-128` under (F), Akita field sites by its plan, fold
coordinates `<= 2^-128.088` (4.6). No site sits at exactly `2^-128` except where (F) is the bound. `eps_hash` covers the hashed entries of 12.1 item 1;
the branch digests `d_L`, `d_S` of K23 are not among them and enter through `eps_br <= 5
Q_P^2 / 2^257 + Q_P / 2^191` (Lemma 12.1b). This is the bound of Section 7.

*Cost.* Akita Theorem 10.17 bounds the extractor's expected cost by `(Q_wrap + 2) 4
Tree0_ck(sch) T_run`, with `T_run` the expected cost of one run of `P` and the verifier
together with the extractor's work attached to that run. The work includes the maps of 12.2:
`Cand`, `Coll`, `Wit` and the two tests. Akita's procedure `E_l` (Appendix E.2) evaluates
`Good_l` in step 2 on the certificate returned by its first continuation, once per
invocation that receives a certificate outside `T`; every invocation contains at least one
run of `P`, so each test is attached to a run at most once. `Good_{l_1}` is the relation `H`
on `f_1` (one SHAKE256 per live record). `Good_{l_2}` is `R_PIOP` on `f` at `a_X`,
polynomial in the instance; `T_run` is one run of `P` and the verifier plus these two
tests. The Module-SIS algorithms `B_I` inherit `T_run` (Akita Lemma 10.20, Remark 10.50).

**Corollary 12.5 (mode B).** Fix `N <= 2^15`, mode B (Flock profile id 3, deviation P9:
the `w1Encode` strings of each segment's live records are the event-2 items, absorbed before
`sd`; the event-6 share items and the Flock proof items are empty), PIOP wire version 4 with the plan of Section 6
(profile id 3, no tied-part site) and Akita's plan for the selected row. Then the statement of
Section 7 holds for mode B without assumption (F) and without the terms `eps_fs`,
`eps_merkle` and `R_F`, with the same extractor bound otherwise. *Proof.* The structure of
12.2 with the same checkpoints at `sd` and `X`, and no tied-part or Flock sites (items 2 and 3 of
12.1 are void, and `Good_{l_2}` is `R_PIOP` alone: there is no `C4_p`). The public bits `b'` are fixed before `sd`, so the "lists" have one
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
Assumption (K) is not needed (SHAKE256 is computed natively). Every address weighs at most
`2^-128` (PIOP sites as in Section 10, `proj.psi` `2^-129` as in 12.4, `sd` at most
`2^-143.64`, Akita by its plan and 4.6), so the bound of 12.6 holds with `R_F = 0`. The
`proj.psi` stage of 12.4 is unchanged in mode B, since `v` is committed in round 1 there too.

### 12.7 What is not proved here

1. Assumption (F): the round-by-round soundness of list-decoded Ligerito with Flock's merged
   opening and packed-direct claims, with the per-site errors that Flock's configs validate.
   The Flock paper sketches it (Appendices A and C.3); a complete proof is not available to us.
   The paper MUST present the theorem as conditional on (F).
2. Assumption (K), the correctness of the sponge circuit (the only fixed circuit of the
   Keccak side).
3. Lemma 12.3 extends Akita's Theorem 10.17; its proof above modifies Claims 1 and 2 of
   Akita Appendix E.2 and SHOULD be checked by a reviewer of Akita's proof. It has had
   internal review only. Since deviation K23 it carries the PIOP core as
   well as the Flock parts: PIOP stages 1 and 2 rest on it, i.e. the 53 sites of branch S at
   `2^15` (`tau`, `gamma^(1)`, 25 stage-1 rounds, `gamma^(2)`, 25 stage-2 rounds), where
   before K23 every PIOP site after `X` had a complete-prefix input. The paper MUST state
   Lemma 12.3 as a new extension of Akita Theorem 10.17. Since deviation K25 its condition
   (ii) also lets the fold coordinates omit the parts' stages; that step SHOULD be read
   beside invariant (2) of Akita Appendix E.2.
4. `R_F` is not counted exactly (open 4); the `Q = 0` term stays about `2^-117` (estimate).
5. The extractor's run count `(Q_wrap + 2) 4 Tree0_ck(sch)` is not tabulated per row (Aerie
   C.4 tabulates its `K`); it bounds the running time only, not the error. Nor is the cost
   `T_run` of one run (12.6).
6. The classical ROM only, as Aerie and Akita.
7. The CMA-to-NMA simulation step for FIPS 204's exact algorithms (12.8, Lemma 12.8 case
   (i)): the published corrected proofs cover round-3 Dilithium (UF-CMA) and generic
   Fiat-Shamir-with-aborts signatures (sUF-CMA), not ML-DSA as standardized (spec 12.3 (b)).
8. The min-entropy `H_inf` of a fresh signature's `c~` (Lemma 12.7) is evaluated only
   heuristically, in the ROM for the signer's hash calls (about 253.9 bits, estimate).

### 12.8 Aggregate unforgeability: signing queries in rewound runs (spec 12.3)

A reduction of an aggregate forger to a UF-CMA forger against the relaxed scheme does not
suffice: a rewound run may ask the signing oracle for `M_i*`, and the output is then no
UF-CMA forgery. ML-DSA has no salt (`mu = H(tr || M')`,
FIPS 204 Algorithm 7 line 6), so Aerie's presampled signature/target lists (Aerie C.6) do
not apply. This section handles such queries with strong unforgeability (SUF-CMA) of the
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

*Hardness of the two lattice instances* (measured with
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
