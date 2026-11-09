# The `F_p` hash field (`--hash-field fp`): constraints, layout, transcript, soundness (normative)

| Field | Value |
| --- | --- |
| Implemented in | `crates/agg-fpkeccak` (layout, packed trace, reference constraint evaluator), `crates/agg-piop/src/keccak_fp/` (masks, stage specs, virtual lanes, stage-3 weights), `agg_pcs_config::catalogs::{HashField, Group::ALL_FP, BatchLayout::new_fp}`, `agg_piop::{prover::prove_core_fp, verifier::run_fp}`, `agg-cli` `--hash-field fp` |
| Applies to | mode A, ML-DSA-44, ML-DSA-65 and ML-DSA-87, relations H-FIPS (byte 2, the default) and H-exact (byte 1); PIOP wire version 4 with the flags bit `0x08`; Akita width-4 catalog rows `mldsa-<set>-p16[x\|f]-fp-batch-<N*>-mult-final` for `d_sig` 3 to 12 (`CATALOG_D_SIG_FP`) |
| Selected by | `CommitmentManifest::hash_field() == HashField::Fp`; the default (`HashField::Gf2`, Flock and the bridge) is unchanged byte for byte |
| Script | `python3 -I docs/scripts/pow_plan.py --hash-field fp [--set 44\|65\|87] [--relation h-fips\|h-exact]`, pinned by `agg_piop::pow::tests::fp_plans_reproduce_the_script` |
| Related | `docs/spec.md` 10.1 (event table and wire layout), `docs/piop.md` 7.6 (stage counts), `docs/h-exact.md` 10.9 (`u_h` under `F_p`), `docs/soundness.md` (the default's census and theorem) |

The uppercase terms MUST, MUST NOT, SHOULD and MAY are normative as in BCP 14. Every number
below is computed by the script, read from the code or its tests, or measured where stated;
estimates are marked.

## 1. Statement

Under the `F_p` hash field the aggregator proves the same statement as the default (mode A,
`docs/spec.md` Section 3, relation H-FIPS or H-exact), but the hash relation
`c~_i = SHAKE256(mu_i || w1Encode(w1'_i))` is arithmetized over `F_p` (`p = 2^64 - 59`) inside the
three-stage sum-check PIOP instead of over GF(2) with Flock. The Keccak round states are committed
with the same Akita preset as the lattice witness (`AggFp64SignedWidth4`) and opened in the same
batched opening. There is no Flock proof, no bridge seed `sd`, no parity matrices, no carries and
no final vectors. The Keccak constraints read the committed `w1Encode` bits `b_t` of `G_main4`
directly, so the hash input is bound to the lattice witness by being the same committed cells.

## 2. Keccak-f[1600] in `F_p`: constraints

Notation: lanes `(x, y)`, `x, y in [0, 5)`, lane index `x + 5 y`, bit `z in [0, 64)`; `rot(x, y)` the
rho offset (FIPS 202 Table 2); `pi(x, y) = (y, 2x + 3y mod 5)`; `RC_r` the round constant of round
`r`. On bits, `xor(a, b) = a + b - 2ab` (degree 2) and `xor3(a, b, c) = a + b + c - 2ab - 2ac -
2bc + 4abc` (degree 3); `chi(b0, b1, b2) = xor(b0, (1 - b1) b2) = b0 + b2 - b1 b2 - 2 b0 b2 + 2 b0 b1
b2` (degree 3). A record has `P` permutations (`P = 7` for ML-DSA-44 and ML-DSA-65, `P = 9` for
ML-DSA-87) and time index `T = 24 m + r` for permutation `m in [0, P)` and round `r in [0, 24)`.

The sponge input is `mu_i` (64 bytes, lanes 0 to 7 of block 0) followed by `w1Encode(w1'_i)` (768,
768 and 1,024 bytes) and the SHAKE256 padding (`0x1F`, `0x80`); the rate is 136 bytes. Block 0
holds the eight `mu` lanes and nine whole `w1Encode` lanes; capacity lanes are zero. For
ML-DSA-44 and ML-DSA-65 the padding starts at byte 16 of block 6; for ML-DSA-87 (64 + 1,024 =
8 x 136) block 8 holds only padding. The output `c~_i` is the first `L = 4, 6, 8` lanes of the last
permutation's output (32, 48, 64 bytes). The wiring of a `w1Encode` bit to its block position is
`agg_witness::keccak::w1_bit_slot_for::<P>` (`agg_fpkeccak::KeccakLayout`).

Committed per record and round `T`:

- `A'_T[x, y, z]`: the state after theta (before rho).
- `C_T[x, z]`: the column parity of the round input `A_T`.
- `C'_T[x, z]`: the column parity of `A'_T`.

Virtual tables (re-indexings of committed cells or public data, never committed):

- `Rot(X)`: a lane read rotated in `z` (by `rot(x, y)` for an `A'` lane, by 1 for a `C` lane).
- `B_T[x, y] = Rot(A'_T[pi^-1(x, y)])` (rho then pi).
- `Prev(X)` at slot `T` is `Rot(X)` at `T - 1`; `PrevIota` is `Prev` of lane `(0, 0)` with the round
  constant of round `T - 1` applied (an affine offset the verifier evaluates). With these,
  `chi` of three `Prev` lanes is the previous round's output `out_{T-1}` (iota included).
- `Blk[(x, y)]` at the first slot of permutation `m`: the wired `b_t` cell or the padding constant
  of that position (the padding bits are an affine offset the verifier evaluates), zero elsewhere.
- `Mu[lane]`, `CTilde[lane]`: the public `mu_i` and `c~_i` lanes. Padded records (`i >= N`) have
  `mu = 0`, `b_t = 0` and `c~ = SHAKE256(0^{64 + |w1Encode|})`, the honest trace of the zero message.

Constraints, per record, on the Boolean hypercube of the live slots (`T < 24P`):

| Name | Constraint | Slots | Degree |
| --- | --- | --- | ---: |
| `k_bool_a` | `A'(A' - 1) = 0` | live | 2 |
| `k_bool_c` | `C(C - 1) = 0` | live | 2 |
| `k_theta_c` | `C'[x] - xor3(C[x], C[x - 1], Rot(C[x + 1])) = 0` | live | 3 |
| `k_parity` | `d (d - 2)(d - 4) = 0`, `d = sum_y A'[x, y] - C'[x]` | live | 3 |
| `k_link` (inner) | `xor3(A', C, C') - chi(Prev, Prev, Prev) = 0` | `T >= 1`, `T != 24 m` | 3 |
| `k_link` (`T = 0`) | `xor3(A', C, C') - In = 0`, `In` = `Mu`, `Blk` or 0 per lane | `T = 0` | 3 |
| `k_link_bnd` | `xor3(A', C, C') - xor(chi(Prev, Prev, Prev), Blk) = 0` | `T = 24 m`, `1 <= m < P` | 4 |
| `k_out` | `chi(B, B, B) - CTilde = 0` for the `L` output lanes | `T = 24P - 1` | 3 |

`C'` needs no Booleanity check: `k_theta_c` makes it a `xor3` of Booleans.

**Lemma K (exactness).** Fix a record. If every constraint above holds on its slots, then for
every `T < 24P` the committed `(A'_T, C_T, C'_T)` are the theta output and the two column
parities of the honest Keccak-f round applied to the honest round input `A_T`, and `c~_i` is the
honest SHAKE256 output of `mu_i || w1Encode(w1'_i)` for the committed bits `b_t`.

*Proof.* Induction on `T`. `k_bool_a` and `k_bool_c` make `A'_T` and `C_T` Boolean; `k_theta_c`
then makes `C'_T = C_T xor D_T` with `D_T[x] = C_T[x - 1] xor Rot(C_T[x + 1])` exactly, and the
link gives `A'_T = A_T xor C_T xor C'_T = A_T xor D_T`, where `A_T` is Boolean by induction (for
`T = 0` it is `mu`, a `b_t` bit or 0; `b_t` is Boolean by `bool_b`; at `T = 24 m` it is the previous
output XOR a Boolean block). XORing the link over `y` gives `xor_y A'_T = xor_y A_T xor C_T xor
C'_T` (five equal terms), and `k_parity` says the left side is `C'_T` (`sum_y A' - C' in {0, 2, 4}`
iff the parities agree, for Booleans); hence `C_T = xor_y A_T`, the true column parity, `D_T` is the
true theta effect and `A'_T` the true theta output. Rho, pi, chi and iota are exact polynomial
images of Boolean inputs, so `chi(Prev, Prev, Prev)` at `T + 1` is the honest round output, which
is `A_{T+1}` (XOR the absorbed block at a permutation boundary). `k_out` pins the first `64 L` bits
of the last output to `c~_i`. The absorbed blocks are the committed `b_t` cells that the ring
identity reads, so the lemma binds `c~_i` to the `w1'_i` of the relation with no copy. `□`

The reference evaluator `agg_fpkeccak::constraints::evaluate_all` checks every row in integer
arithmetic; its tests flip one bit and expect exactly the predicted violations, and real records of
every set evaluate clean against the signed `c~`.

## 3. Witness layout, groups and Akita layout

Domain of one Keccak polynomial: `(z, s, i)`, cell `z + 64 s + 2^10 i`, `d_K = 10 + d_sig`
variables: 16 time slots per polynomial. The time index is `T = 16 q + s` with slot group
`q in [0, Q)`, `Q = ceil(24P / 16)`: 11 for ML-DSA-44 and ML-DSA-65 (176 slots for 168 rounds),
14 for ML-DSA-87 (224 for 216). Slots `T >= 24P` hold zeros and carry no constraint. A polynomial
is one lane (or one column `x`) over one slot group:

| Group (wire order after `G_main4`) | Polynomials | Honest values | Round |
| --- | ---: | --- | ---: |
| `G_KA[y]`, `y = 0..4` | `A'(x, y)` for `x < 5`, times `Q`: `5Q` | `{0, 1}` | 1 |
| `G_KC` | `C[x]`, times `Q`: `5Q` | `{0, 1}` | 1 |
| `G_KC'` | `C'[x]`, times `Q`: `5Q` | `{0, 1}` | 1 |

The seven Keccak groups have one Akita layout `(d_K, 5Q)`, alphabet code 0 (Boolean) for every
polynomial, and are round-1 groups (nothing in them depends on a challenge). Committed Keccak
cells per record: `35 Q x 1024` = 394,240 (ML-DSA-44, ML-DSA-65) and 501,760 (ML-DSA-87), of which
`24 P x 2,240` = 376,320 and 483,840 carry rounds.

The batch has eleven groups (`Group::ALL_FP`, `GROUP_COUNT_FP = 11`): `G_main16`, `G_main4`,
`G_KA[0..4]`, `G_KC`, `G_KC'`, `G_k`, `G_m`. The bridge levels `G_1`, `G_2`, `G_3` do not exist.
Polynomial counts (`Group::num_polynomials_in`; variables at `N* = 2^10` in parentheses):

| Set, relation | `G_main16` | `G_main4` | Keccak (7 groups) | `G_k` | `G_m` | Total | Opened at `r^(3)` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 44 H-FIPS | 4 (20) | 12 (20) | 7 x 55 (20) | 5 (17) | 7 (16) | 413 | 406 |
| 44 H-exact | 4 | 9 | 7 x 55 | 4 | 7 | 409 | 402 |
| 65 H-FIPS | 23 (18) | 64 (18) | 7 x 55 (20) | 6 (17) | 7 | 485 | 478 |
| 65 H-exact | 23 | 46 | 7 x 55 | 5 | 7 | 466 | 459 |
| 87 H-FIPS | 31 (18) | 86 (18) | 7 x 70 (20) | 6 (18) | 7 | 620 | 613 |
| 87 H-exact | 31 | 62 | 7 x 70 | 5 | 7 | 595 | 588 |

The multiplicity digits of `G_m` are opened at `pi_m`, every other polynomial at the suffix of
`r^(3)` of its domain. The catalog rows are planned for `d_sig` 3 to 12 (60 grouped `F_p` rows in
the width-4 artifact, 220 rows in all); larger batches need rows that are not yet planned.

**`u_h` under H-FIPS.** With no `G_2`, the per-signature hint slack `u_h` is the last polynomial of
`G_k` (`Column::HINT_SLACK_PROJ`) and the third column of the projection stack
`[K_0 | K_1 | u_h]` (four slots). `D_k` ends in the `d_sig` signature variables, so `u_h(i)` sits at
`D_k` index `2^(d_k - d_sig) i`, and the second part of the hint-weight identity `F7` runs on `D_k`
with that scale (`docs/h-exact.md` 10.9). This adds no group and no scalar catalog row.

**Lookup.** One table `T_16`, two challenges, no carry stacks. ML-DSA-44: `[Z_lo | U_lo]` on the
main domain and `[K_0 | K_1]` (H-exact) or `[K_0 | K_1 | u_h]` (H-FIPS) on `D_k`. ML-DSA-65 and
ML-DSA-87: the looked-up `Z_lo`/`U_lo` lanes in stacks of `[8, 2, 1]` and `[8, 4, 2, 1]` on the
lane domain and the projection stack on `D_k`. Trees per challenge at `N* = 2^15` (H-FIPS, from the
script): 44 `[26, 24, 16]`, 65 `[26, 24, 24, 23, 16]`, 87 `[26, 25, 25, 24, 23, 16]`.

## 4. Stage placement

**Rounds.** The stages run `d_stage = max(d_main, d_K)` rounds (`CommitmentManifest::d_stage`).
For ML-DSA-44 `d_K = d_main = 10 + d_sig`. For ML-DSA-65 and ML-DSA-87 (`d_main = 8 + d_sig`) the
Keccak domain is two variables longer, so `d_stage = d_K`; the main-domain and `D_k` instances
are inactive in the leading `d_stage - d_loc` rounds and lifted by `2^-(d_stage - d_loc)` (the
kernel's lift, `docs/spec.md` 7.1), and the public lattice tables (`T~_rho`, `w_A`, the projection
weights) are read at the last `d_main` coordinates of the stage points.

**Masks.** A zero-check with weight `eq(tau, x) 1_M(s)` for a slot set `M` is multilinear in `x`
(the indicator is a Boolean function of the slot variables only), so the mask costs no degree, and
`sum_x eq(tau, x) 1_M(s) F(x) = 0` for random `tau` forces `F = 0` on `{x : s in M}`. The kernel
realizes it as `EqPoint::TauMasked(mask)` (stage 1) and `EqPoint::R1Masked(mask)` (stage 2), a
16-entry slot factor `eq(tau_s, s) 1_M(s)` in the split `eq`. Masks per slot group `q`: live
(`16 q + s < 24P`), inner after `T = 0`, `T = 0`, boundary (`16 q + s = 24 m`, `1 <= m < P`; in six
slot groups for `P = 7`, eight for `P = 9`) and output (`T = 24P - 1`).

**Instances** (`keccak_fp::specs`). Stage 1 (degree 4 with `eq`) holds only the Keccak
zero-checks, in this order: `k_bool_a` per `(x, y, q)` (`25 Q`), `k_bool_c`, `k_theta_c` and
`k_parity` per `(x, q)` (`5 Q` each), `k_link` per `(x, y, q)` (`25 Q`; at `q = 0` with a second part
on the `T = 0` mask under the same coefficient) and `k_out` per output lane (`L`):
`gamma^(1)` has `65 Q + L` coefficients. Stage 2 (degree 5) keeps the lattice instances of the
relation (the digit leaves and `range_w1`) except `def_y1[t]`, the bridge's level-1 identity
(ML-DSA-44's `range_w1` reads all six bits, so the bits keep their claims at `r^(2)`), and appends
`k_link_bnd` per lane and boundary slot group: `25 (P - 1)` instances. Stage 3 (degree 3) keeps the
lattice instances and reduces every Keccak claim of stages 1 and 2 to `r^(3)`.

| Set, relation | `gamma^(1)` | `gamma^(2)` | `gamma^(3)` | Keccak claims (stage 1 + 2) | Stage-3 Keccak instances |
| --- | ---: | ---: | ---: | ---: | ---: |
| 44 H-FIPS | 719 | 154 | 1,284 | 740 + 516 | 385 + 310 + 12 |
| 44 H-exact | 719 | 154 | 1,278 | 740 + 516 | 385 + 310 + 12 |
| 65 H-FIPS | 721 | 164 | 1,351 | 743 + 516 | 385 + 310 + 28 |
| 65 H-exact | 721 | 164 | 1,320 | 743 + 516 | 385 + 310 + 28 |
| 87 H-FIPS | 918 | 218 | 1,752 | 943 + 688 | 490 + 405 + 36 |
| 87 H-exact | 918 | 218 | 1,711 | 943 + 688 | 490 + 405 + 36 |

The stage-1 Keccak claims are `66 Q + 9 + |out_rot|` (`|out_rot|` = 5, 8, 10 rotated `A'` lanes read
by `k_out`), the stage-2 claims `86` per boundary slot group (`pow_plan.py`, `fp_gammas`). The
stage-3 vector is the default's minus the bridge instances, the carry stacks and (lane sets) the
bit reductions at `r^(2)`, plus one coefficient per Keccak claim.

**Rotated and shifted reads.** A `Rot`, `Prev` or `PrevIota` terminal claim is a claim on the
multilinear extension of a re-indexed table, which a point opening does not give (a rotation is
not a permutation of the six `z` variables, and `Prev` at `s = 0` reads slot 15 of `q - 1`). Stage 3
reduces them with factored weights: at `R = (r_z, r_s, r_i)`,

```
Prev_{a,q}(R) = sum_{z', s' <= 14} e_a(z') eq(r_s, s' + 1) X_{a,q}(z', s', r_i)
              + eq(r_s, 0) sum_{z'} e_a(z') X_{a,q-1}(z', 15, r_i),
e_a(z') = eq(r_z, (z' + rot(a)) mod 64),
```

so every term is a 64-entry `z` weight times a 16-entry `s` weight times `eq(r_i, i)` on one
polynomial. All claims on one polynomial at one point (its own evaluation, the `Rot` and `Prev`
terms of its consumers, the `s = 0` term from `q + 1`) are summed into one instance per
`(polynomial, point)` with coefficients from `gamma^(3)`: every Keccak polynomial at `R1` (the
`d_K`-suffix of `r^(1)`), and at `R2` every polynomial with a stage-2 claim (310 for `P = 7`, 405
for `P = 9`). The verifier evaluates
each weight in `O(64 x 16)` from the `eq(r_z, .)` and `eq(r_s, .)` tables
(`keccak_fp::weights`); the round-constant and padding offsets of `PrevIota`, `Rot[(0,0), Q-1]` and
`Blk` are public and evaluated by the verifier.

**Absorbed bits.** The `Blk` claims are reduced on the domain of the bits:
`sum_k alpha_k Blk_k(R) = sum_{j, i} u_{t,rho}(j) eq(r_i, i) b_t(j, rho, i)` with
`u_{t,rho}(j) = sum_{k wired to (t, j, rho)} alpha_k eq(r_z, z_k) eq(r_s, s_k)`, one public table per
bit lane and point, instance `k_absorb[t, rho]` on `GroupRole::Main`. The record coordinates of the
Keccak suffix and of the main suffix are the same coordinates of the stage point (both domains end
in the `d_sig` record variables), so `eq(r_i, i)` is shared and `b_t` is opened once, at `r^(3)`.
ML-DSA-44 (six bits per coefficient, main domain): `k_absorb[t]` for `t < 6` at both points, 12
instances. ML-DSA-65 and ML-DSA-87 (four bits per coefficient, lane domain, `u` tables of 256
entries per row lane `rho`): at `R1` only row lane 0 is wired (block 0 holds 144 coefficients of
row 0), at `R2` every `(t, rho)` with `t < 4`, `rho < k`: `4 + 4k` = 28 and 36 instances. No all-zero
weight instance is emitted.

## 5. Transcript, wire and event 0

The `F_p` mode is a separate branch of `transcript::Registry::new`; nothing is inserted into the
default's bytes (the event table is `docs/spec.md` 10.1):

- Event 0: `CommitmentManifest::binding_bytes` with group count 11, the Keccak group descriptors
  (alphabet code 0), the `-fp-` catalog row id, no bridge parameters, the lookup layout without
  carry stacks; then the public-record digest, the Keccak-side profile id **6**
  (`FlockProfile::FpKeccak`, accepted only under `HashField::Fp`) and the digest of the `F_p`
  proof-of-work plan.
- Events 1 (round-1 commitments `G_main16`, `G_main4`, `G_KA[0..4]`, `G_KC`, `G_KC'`), 3 (`proj/psi`
  with no `sd` before it), 4 (`G_k`, `G_m`), the fork, 7 (lookup), 8 (`tau in K^{d_stage}`), 10 to 12
  (stages with the vectors of Section 4), the join without Flock items, 13 (one Akita opening of the
  eleven groups). Events 2, 5, 6, 9 and `Y_sigma` do not exist. `fp_registry_has_no_bridge_or_flock_events`
  pins the order.
- Wire: bit 3 of the flags byte (`wire::HASH_FIELD_FP_FLAG = 0x08`) marks the `F_p` field; the
  relation id stays in bits 0 to 2 and the set in the high nibble. Flags bytes: `0x0A` (44 H-FIPS),
  `0x09` (44 H-exact), `0x1A` (65 H-FIPS), `0x2A` (87 H-FIPS). After the header come the `c~` list
  (32, 48 or 64 bytes per record), the framed commitments in group order, the lookup, the three
  stages, the packed nonce stream and the framed Akita opening. A verifier of the other field MUST
  reject the header before parsing (`VerifyError::HashField`); `agg verify` without
  `--hash-field fp` rejects an `F_p` aggregate.

## 6. Soundness: census and ledger

The `F_p` plan keeps the site classes of the default's PIOP (`proj.psi`, `lookup.x`, `lookup.mu`,
`lookup.gamma`, `lookup.round`, `tau`, `stage.gamma`, `stage.round`) with the per-site rule of
`docs/soundness.md` Section 3.1 (difficulty `g_t = min{g : 2^128 e_t <= 2^g}`, nonce of `g_t + 7` bits).
It has no Keccak-side site. Census from `python3 -I docs/scripts/pow_plan.py --hash-field fp --set S`
(relation H-FIPS; H-exact differs only in `gamma^(3)` and gives the same sites, nonces and ledger;
`est.ms` is the script's estimate):

| Set | `N*` | `d_stage` | `D` | `L` | `X` site | sites | nonces | bits | bytes | `sum 2^g` | est. ms | max weight | ledger before work |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 44 | `2^10` | 20 | 21 | 2,621,440 | `2^-213.3` | 317 | 316 | 2,926 | 366 | 4,948 | 3.58 | `2^-128.42` | `934/|K| = 2^-118.1` |
| 44 | `2^13` | 23 | 24 | 20,971,520 | `2^-207.3` | 398 | 397 | 3,664 | 458 | 5,326 | 3.86 | `2^-128.42` | `1,177/|K| = 2^-117.8` |
| 44 | `2^15` | 25 | 26 | 83,886,080 | `2^-203.4` | 457 | 456 | 4,201 | 526 | 5,598 | 4.05 | `2^-128.36` | `1,354/|K| = 2^-117.6` |
| 65 | `2^10` | 20 | 21 | 3,407,872 | `2^-212.5` | 317 | 316 | 2,943 | 368 | 5,220 | 3.78 | `2^-128.42` | `934/|K| = 2^-118.1` |
| 65 | `2^13` | 23 | 24 | 27,262,976 | `2^-206.6` | 398 | 397 | 3,684 | 461 | 5,646 | 4.09 | `2^-128.42` | `1,177/|K| = 2^-117.8` |
| 65 | `2^15` | 25 | 26 | 109,051,904 | `2^-202.6` | 457 | 456 | 4,223 | 528 | 5,950 | 4.31 | `2^-128.36` | `1,354/|K| = 2^-117.6` |
| 87 | `2^10` | 20 | 21 | 4,980,736 | `2^-211.5` | 317 | 316 | 2,945 | 369 | 5,244 | 3.80 | `2^-128.42` | `934/|K| = 2^-118.1` |
| 87 | `2^13` | 23 | 24 | 39,845,888 | `2^-205.5` | 398 | 397 | 3,686 | 461 | 5,670 | 4.11 | `2^-128.42` | `1,177/|K| = 2^-117.8` |
| 87 | `2^15` | 25 | 26 | 159,383,552 | `2^-201.5` | 457 | 456 | 4,225 | 529 | 5,974 | 4.33 | `2^-128.36` | `1,354/|K| = 2^-117.6` |

Batching sites at `N* = 2^15`: `stage.gamma` 719, 154, 1,284 over `|K|` (44; 10, 8, 11 bits;
nonces 17, 15, 18 bits), 721, 164, 1,351 (65) and 918, 218, 1,752 (87), both lane sets 10, 8, 11
bits; stage rounds of degree 4, 5, 3 (3, 3, 2 bits), `d_stage` each; `tau` 5 bits. The plan digest
binds profile id 6; it is equal for two relations of one set whose `gamma` vectors have equal bit
lengths, and the relation is then bound by the manifest bytes of event 0, as in the default.

**Ledger against `docs/soundness.md` Section 0, item 6** (round-by-round union bound before work,
`(sum of stage degrees) d_stage + d_stage + 3 + GKR` over `|K|`):

- *Sites removed*: the bridge seed `sd` (`<= 2^-131.4` in the composition, no nonce), the seeds
  `fs_sigma`, every tied-part site (tie, `eq_Y`, the parity-chain rounds) and every Flock internal
  site; the GKR trees of the carry stacks (the session count `D` is unchanged; the `lookup.gamma`
  bounds shrink, at most 12, 20 and 24 over `|K|` for 44, 65 and 87 at `2^15`, against 24, 32 and 36).
- *Counts changed*: `stage.gamma` from 19, 10, 56 (44 H-FIPS; 13, 18, 136 for 65 and 13, 22, 173 for
  87) to the vectors above (5, 4, 6 bits to 10, 8, 11); the stage-1 degree from 3 to 4 (its rounds
  from 2 to 3 bits); for 65 and 87 the stage rounds and `tau` from `d_main` to `d_stage = d_main + 2`
  (six more stage-round sites).
- *Resulting unprotected sum*: ML-DSA-44 `934, 1,177, 1,354` over `|K|` at `2^10, 2^13, 2^15`
  against the default's `914, 1,154, 1,329` (`+d_stage`, the stage-1 degree); ML-DSA-65 and
  ML-DSA-87 the same `934, 1,177, 1,354` against the default's `890, 1,130, 1,305` (`+44, +47, +49`:
  the degree and two more rounds per stage). Plus the `X` term of the table and `proj.psi`
  (`2^-128`), with no `sd` term. Batching vectors count `nu = 1` round by round, so their length
  enters only through the assigned bound `e_t` (and the difficulty).
- *After work*: every site weighs at most `2^-128` per address (largest `2^-128.36` at `2^15`,
  `2^-128.42` at `2^10` and `2^13`).

**Theorem (scope).** With these sites the theorem of `docs/soundness.md` Section 7 holds for the
`F_p` mode with the same proof, less the composition segment of `sd` (12.4), less Flock's
round-by-round analysis (F) and less the bridge lemma: the checkpoint structure has `X` only, and
the extractor reads the committed Keccak cells from the one Akita opening, as it reads the
lattice groups ((H1) holds for the 48 grouped `F_p` rows,
which `akita_fold_support.rs` audits with every other width-4 row). Correctness of the hash
argument is Lemma K applied to the extracted cells once the zero-checks are accepted; the binding
of the hash input to the lattice witness is syntactic (same cells). What remains conditional:
Module-SIS (Akita), the ROM, and for unforgeability the SUF-CMA reduction of `docs/soundness.md`
12.8, which never used the field of the hash argument. `Psi` is drawn after round 1 and before
round 2, as in the default; Lemma P is unchanged. The Akita opening plan of each `F_p` row is
derived by Akita's planner from the row's shape; it is not dumped in this document.

## 7. Measured at `N = 1,024` (informative)

Records `keygen-batch --count 1024 --seed 42`, `sign-batch --count 1024 --seed 43`; release binary;
one run each on a 16-thread WSL box (i7-11700K, turbo and SMT on), so the times are indicative only.
The byte hashes are pinned in `artifact/expected/hashes-fp.json`.

| Set, relation | Fp bytes (B/sig) | Gf2 bytes (B/sig) | Fp prove wall | Fp peak RSS | Fp verify (in-process, uncached) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 44 H-FIPS | 145,354 (141.9) | 428,155 (418.1) | 31.4 s | 6.35 GiB | 115 ms |
| 44 H-exact | 145,147 (141.7) | 428,049 (418.0) | 34.3 s | 6.17 GiB | 124 ms |
| 65 H-FIPS | 167,643 (163.7) | 450,126 (439.6) | 34.4 s | 6.26 GiB | 132 ms |
| 87 H-FIPS | 194,834 (190.3) | 489,408 (477.9) | 43.6 s | 7.54 GiB | 138 ms |

Each `F_p` aggregate is byte-identical under the default tier, `AGG_ISA=avx512` and `AGG_ISA=avx2`, and
`agg verify --hash-field fp` accepts it. Of the 1K bytes, the `c~` list is 32, 48 or 64 B/sig and the
Akita opening about 61.6 B/sig. Stage 1 takes most of the prover time (19.7 s of 31.4 s for 44
H-FIPS, 28.1 s of 43.5 s for 87).

## 8. Open items

- The prover is far above the cost model of the design (`F_p` 1K 31 to 44 s on the WSL box against
  1.1 to 1.4 s for the default, Section 7); before any 8K or 32K run the Keccak instances
  must share lanes per slot group, the Boolean Keccak columns must go through the coded-column
  path without `i8` or `K` expansion, and the signed `+-1` encoding of the bits should be adopted.
- Catalog rows for `d_sig` 11 to 15.
- The Akita grinding plans of the `F_p` rows are not pinned by a test (`akita_grinding.rs` pins the
  default rows).
