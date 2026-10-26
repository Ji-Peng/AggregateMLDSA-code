# The virtual parity chain: the GF(2) side of the bridge without the bridge union

| Field | Value |
| --- | --- |
| Status | design note for tied-part proof version 5 (deviation K21); replaces the bridge union of `docs/bridge.md` Section 5b as the GF(2) side of a tied part |
| Source | `ref/notes/brainstorm-20261026.md` Section 3.2 (idea and skeptic's corrections); `docs/bridge.md` Sections 4, 5, 5b, 5c, 10.4; `docs/keccak.md` Sections 9 to 12 |
| Code | `crates/agg-bridge/src/parity_chain.rs` (the sum-check, generic over `Shape`), `crates/agg-keccak/src/tied.rs` (transcript, proof bytes), `crates/agg-keccak/src/flock_union.rs` (extra ring-switched claims), `vendor/flock-core` (`SkipPoint::Eq`) |
| Numbers | "est." marks an estimate; sizes are exact products of the shape constants |

## 1. What changes

Today (proof version 4) a tied part proves `Y_sigma = M_3 M_2 M_1 b'_sigma` over GF(2) with a
second Flock instance, the bridge union (level-1 rows `y1 = M_1 x` and a levels-2-3 row),
committed after `sd`, tied to the sponge instance by two lane-group claims (tie 1), to
itself by two parity-copy claims (tie 2), and to the public final vectors by an output
claim. Its proof is 203,052 B at `8@13/4` with `Slim` plus a 2,048 B cap and 8 B of framing
(measured, `docs/keccak.md` 12.3).

Version 5 deletes the union. In characteristic 2 the composite `M = M_3 M_2 M_1` is
`F_{2^128}`-linear, so for a random `eq_Y` the verifier's value `<eq_Y, Y>` equals
`<M^T eq_Y, b'>`, an inner product of a public weight `W = M^T eq_Y` and the committed bits.
One degree-2 sum-check over the variables of `b'` reduces it to one evaluation of the bit
polynomial of `b'`, which the sponge instance's own Flock opening proves as a third
ring-switched claim. The GF(2) side then commits nothing beyond the hash witness. The F_p
side (`docs/bridge.md` Section 4) is unchanged: parity is not `F_p`-linear, so its
committed carries stay.

## 2. Indices

A tied part has `2^d` record positions per segment and `S` segments (`span_log =
ceil(log2 S)`), the part's input layout `SegmentLayout::part(PRODUCTION, 2^(10 + d), index)`
(`lane_len = 2^(10 + d)`, `offset`, `first_block`, `zeta1_base`). The chain runs over
`2^lb` lanes with `lb = ceil(log2 lanes)` (production: 3, so 8 lanes; lanes 6, 7 carry the
sponge words 48..63, see below):

| Coordinates (least significant first) | Count (production, full segment) | Meaning |
| --- | --- | --- |
| `zeta` | `Z = log2(block_len)`, `block_len = min(lane_len, J_1)`: 19 | in-block position `zeta = p + 1024 i_lo` (`p = j + 256 rho < 1024`, record low bits `i_lo`) |
| `i_mid` | `B = log2(blocks)`: 4 | level-1 block of the part |
| `t` | `lb`: 3 | lane |
| `sigma` | `span_log`: 2 at `S = 4` | segment of the span |

`R = Z + B + lb + span_log` rounds: 28 at four full segments, 26 at one, `10 + d + 3` for
`d <= 9` and one segment.

The bits `b''(sigma, t, i_mid, zeta)` are the sponge's committed words: lane `t` of record
`i = i_lo + 2^9 i_mid` of segment `sigma` is sponge row `i + 2^d sigma`, word columns `8 t +
w` (`w = p >> 7`), bit `p & 127`. Lanes 0..5 are `b'`; lanes 6, 7 are the word columns
48..63 (mu, `c~`, the constant, a zero word, the first chi words). They enter `b''` because
the final claim is one evaluation of the sponge polynomial over its word columns 0..63 (one
claim instead of two lane groups); they get weight zero.

## 3. The weight and the claim

The verifier draws `eq_Y = eq(r_o, rho_3) eq(r_Y, sigma)` (`r_o`: `log2 m_3 = 12`
coordinates, `r_Y`: `span_log`), factored. With the public final vectors `Y_sigma`
(`Y_sigma = 0` for `S <= sigma < 2^span_log`):

```text
claim  = sum_sigma eq(r_Y, sigma) sum_{rho_3} eq(r_o, rho_3) Y_sigma[rho_3]
       = sum_{sigma, t, i_mid, zeta} W(sigma, t, i_mid, zeta) b''(sigma, t, i_mid, zeta)
W      = eq(r_Y, sigma) W'(t, i_mid, zeta),  W' = 0 for t >= lanes
v2[t + 6 rho_2]     = sum_{rho_3 in S_3(t + 6 rho_2)} eq(r_o, rho_3)                (18 x 49,152)
G'(t, i_mid, rho_1) = sum_{rho_2 in S_2(zeta2_base + rho_1 + m_1 i_mid)} v2[t + 6 rho_2]  (14 x 6 x 2^17)
W'(t, i_mid, zeta)  = sum_{rho_1 in S_1(zeta1_base + zeta)} G'(t, i_mid, rho_1)         (never materialized)
```

`G'` is built once by the prover; the verifier never builds it (Section 5).

## 4. The two-phase sum-check

The summand `W~(x) b''~(x)` is a product of two multilinear polynomials in the `R`
variables; the round polynomial `q_k(X)` has degree 2. The prover sends `(q_k(1), q_k(inf))`
(`q_k(0) = T_k + q_k(1)`, as Flock's lincheck), the verifier draws `r_k` and sets `T_{k+1} =
q_k(r_k)`.

**Block phase** (`B + lb + span_log` rounds: 9 at `8@13/4`; variables `i_mid`, then `t`,
then `sigma`). The level map `I (x) M_1` acts on in-block coordinates only, so folding the
block variables commutes with it: for any block point `beta*`,
`sum_zeta W~(beta*, zeta) b''~(beta*, zeta) = sum_{rho_1} G~(beta*, rho_1) y1~(beta*, rho_1)`
with `G(beta, rho_1) = eq(r_Y, sigma) G'(t, i_mid, rho_1)` and `y1(beta) = M_1 b''(beta)`,
the honest level-1 parities (XOR, embedded in `F_{2^128}`). So the block phase is an ordinary
product sum-check of the two tables `G`, `y1` of `2^(B + lb + span_log) x m_1` entries (512 x
8,192 at `8@13/4`), bound on the block variables only. `y1` is the prover's state, not a
committed object and not checked: the messages are the true round polynomials if it is
honest, and soundness rests on the sum-check alone.

**In-block phase** (`Z` rounds over `zeta`). After the block phase the claim is
`sum_zeta w(zeta) bf(zeta) = T` with `g = G~(r_block, .)` (the folded table, `m_1` entries),
`w = M_1^T g` restricted to the part (`w(zeta) = sum_{rho_1 in S_1(zeta1_base + zeta)}
g(rho_1)`, `14 x 2^19` additions) and `bf(zeta) = sum_beta eq(r_block, beta) b''(beta, zeta)`
(`2^19 x 512` masked additions, est. 1.3e8 at `8@13/4`). An ordinary product sum-check of
`Z` rounds ends at `T_R = w~(r_zeta) bf~(r_zeta)`.

**Final claim.** The prover sends `v = b''~(r)`, `r = (r_zeta, r_block)`; the verifier
checks `T_R = W~(r) v` with its own `W~(r)` (Section 5). `v` is an evaluation of the BIT
polynomial: `r_zeta` holds the 7 in-word bit coordinates, so it is not a packed-direct claim
(the skeptic's correction). It becomes a ring-switched claim on the sponge commitment:

```text
skip coordinates (6)  = r_zeta[0..6]                       (bits 0..5 of the word)
x_full[0]             = r_zeta[6]                          (bit 6)
rows (nu_s)           = (r_zeta[10..10+min(d,9)], r_i_mid, r_sigma) truncated or zero-padded to nu_s
columns (k_log - 7)   = (r_zeta[7], r_zeta[8], r_zeta[9], r_t[0], r_t[1], r_t[2], 0, ..)
```

When `nu_s < d + span_log` (a part with few records) the sponge has no rows for the high
record coordinates and `b''` is zero there: `b''~(r) = eq(r_rec[nu_s..], 0) v'` with `v'`
the sponge polynomial's value at the truncated point (the same factor as tie 1 of version
4); the proof carries `v'`.

## 5. The verifier's weight `W~(r)`

`W~(r) = eq(r_sigma, r_Y) sum_{t, i_mid} eq(r_t, t) eq(r_i_mid, i_mid) sum_zeta eq(r_zeta,
zeta) W'(t, i_mid, zeta)`, computed right to left without `G'`:

```text
e3       = eq(r_o, .)                                   4,096
u2[rho_2] = sum_{t < lanes} eq(r_t, t) sum_{rho_3 in S_3(t + 6 rho_2)} e3[rho_3]    18 x 49,152 adds
g[rho_1]  = sum_{i_mid} eq(r_i_mid, i_mid) sum_{rho_2 in S_2(..)} u2[rho_2]           14 x 2^17 adds
W~(r)     = eq(r_sigma, r_Y) sum_zeta eq(r_zeta, zeta) sum_{rho_1 in S_1(zeta1_base + zeta)} g[rho_1]   14 x 2^19 adds
```

About `1.0 x 10^7` additions and `1.3 x 10^6` multiplications per part (derived), est. 10 to
30 ms on one core, against the union's walkers (`1.9 x 10^7` entries), its `2^20`-entry
tables, zerocheck, lincheck and Ligerito verification (est. 70 ms locally at `8@13`,
measured 83.2 ms for the whole part in `docs/keccak.md` 11).

## 6. Transcript (tie challenger, version 5)

`TieChallenges` (domain `agg-mldsa/v3/tie`) absorbs `fs`, `d`, `nu_s`, `log2 blocks`,
`span_log`, the final-vector digest and the framed sponge cap (no union cap any more), then:

1. tie work (`g_tie` bits), then `rho_r` (`nu_s`), `rho_c` (3): the public claim's point;
2. `eq_Y` work (`g_eq` bits), then `r_o` (12) and `r_Y` (`span_log`);
3. per round `k < R`: absorb `q_k(1), q_k(inf)`; work (`g_rd` bits); draw `r_k`;
4. absorb `v'`; squeeze the 32-byte tie binding.

The sponge challenger absorbs the binding (label `agg-keccak/v4/tie-binding`) before Flock's
statement binding, as in version 4 (`docs/soundness.md` 4.5): both claim points (public
region, bit point) and `v'` are fixed before any Flock challenge.

## 7. Proof bytes (version 5)

`AGGKCT || 5 || profile || LE32(n) || d || S || LE32(tie nonce) || LE32(eq nonce) || R x
(q(1) || q(inf) || LE32(round nonce)) || v' || LE32(len) || sponge proof`, the sponge proof a
`flock_union::UnionProof` with one packed-direct value (`v_pub`) and three ring-switched
claims in its merged opening (ab, c, and the bit claim). Est. added bytes at `8@13/4`:
`4 + 28 x 36 + 16 = 1,028` B of chain, about 2,056 B of ring switch (`s_hat_v`, 128 values,
and its nonce) and 2,048 B of multipoint dual values (128 per ring-switched claim in Flock's
twisted assist; the memo's 3.0-3.6 KB did not count these), minus 32 B of the two dropped
lane-group values; total est. 5.1 KB against 205,108 B removed.

## 8. Sites and work (`docs/soundness.md` 3.6, 10)

The sponge commitment is bound only to its Johnson list (`L_s` = 35, 50, 70 for `Fast`,
`Slim`, `Slim8`) when the chain's challenges are drawn:

| Site | Numerator over `2^128` | `Slim`, `8@13/4` (est.) |
| --- | --- | --- |
| tie (public words) | `L_s (nu_s + 3)` | 900, `g = 10` |
| `eq_Y` | `L_s (log2 m_3 + span_log)` | 700, `g = 10` |
| round `k` (degree 2) | `2 L_s` | 100, `g = 7` (28 sites) |

The bridge lemma (`2^-144.64`, `docs/bridge.md` 7) and the `sd` site (`L_s theta =
2^-139.0`) are untouched: `b''` is fixed by the event-2 cap before `sd`, and `Y` before
`eq_Y`. The bridge-union extractor `z_u`, ties 1 and 2 and the output claim disappear; the
union leaves assumption (K) (its circuit) and (F) (its Flock instance).

## 9. The SkipPoint route: a multilinear skip basis

Flock's ring-switched claims carry a `SkipPoint` for in-word bits 0..5 (`Phi8`: a
univariate-skip point; `Ag`), and the verifier uses it only through `SkipPoint::weights(6)`,
the 64-entry functional that `ring_switch::build_claim_weights_from_skip` tensors with bit 6
(`pcs.rs`, `verify_batch_merged_core`); the prover's ring switch never sees it (`s_hat_v`
depends only on the multilinear suffix). Two routes:

* **Multilinear variant (chosen).** `SkipPoint::Eq([F128; 6])` with `weights = eq` over the
  six coordinates (index `sum_k b_k 2^k`, Flock's in-word bit order). One `match` arm in
  `weights`; the arms that sample fresh skip points or grind lincheck skip challenges are
  unreachable for it (it only enters openings) and panic. The ring-switch identity
  `claim = sum_i w_i s_hat_v[i]` is exact for any linear functional, and the reduction after
  it (`r''`, the packed claim) does not depend on the basis, so Flock's ring-switch analysis
  carries over unchanged. Validation: one unit test that the claim check with the eq weights
  recovers the bit-MLE (the analogue of `claim_check_recovers_zhat_skip`).
* **Univariate-skip round.** The in-block phase's first six rounds replaced by one round over
  Flock's 64-point `phi_8` domain: the round polynomial has degree `2 x 63`, so 127
  coefficients (2 KB) instead of 6 x 32 B, a new prover kernel for the skip round (the
  product of two degree-63 extensions), and a verifier evaluation in the `phi_8` basis. More
  code, more bytes, no verifier-visible saving.

The multilinear variant is cheaper to validate and smaller; it is a verifier-visible change
of the vendored core (a new skip basis), recorded in K21.

## 10. Costs (estimates before measurement)

Prover at `8@13/4` (four full-segment geometries): `G'` 1.1e7 additions, `G` 4.2e6
multiplications, block rounds about 1.2e7 multiplications, `bf` 1.3e8 masked additions,
`w` 7.3e6 additions, in-block rounds about 3e6 multiplications, level-1 parities of 8 lanes
(`Level1Layout::parities` work plus two lanes); one more ring-switched claim in the sponge
opening (its `s_hat_v` and block-first bank over the 64 word columns, the per-claim block
statistics and the twisted assist). Est. 0.1 to 0.3 s against the union's measured share
(at `8@13/4` the part proved in 1,994.1 ms with `Slim`, the sponge instance of 8 records
being small, `docs/keccak.md` 12.3). Verifier: est. 10 to 30 ms for `W~(r)` plus one ring
switch, against about 70 ms for the union. Memory: `G` and `y1` tables of 64 MiB each at
`8@13/4` (est.), released after the block phase.

## 11. Tests

`agg-bridge` (unit, toy shape and a three-lane toy variant so that lanes beyond `lanes`
exist): the claim equals `sum b'' W` by brute force (`gf2_walker::transpose`); every block
round message equals the brute-force round polynomial of `W~ b''~` on random `b''` (cross
terms included), and every in-block message too; the final value and `W~(r)` agree with
dense evaluations; a wrong `y1` (flipped parity) changes the first message. `agg-keccak`
(`tests/tied.rs`, release): round trips as in version 4; negative cases: a flipped `b'` bit
with stale `Y`, a flipped `Y` bit, a tampered round message, a wrong `s_hat_v` of the bit
claim, another `sd`; the existing statement, seed, segment-count and byte-flip cases.
Added in review (`tied::tests`): a false bit value consistent with the chain (a chain copy of
`b'` with one bit flipped, its `Y`, the copy's value) passes every round and the final check
and fails Flock's ring-switch claim check (`ClaimMismatch`), end to end at 5 and 64 records
and, ignored by default (about 83 s), at 2,048; the full-height sponge test also rejects a
false bit value directly, so the block-first path is pinned in every suite run.

## 12. Measurements and corrections (2026-10-26, local, direction only)

Numbers in `docs/keccak.md` 12.7 (measured). Against the estimates of Sections 7 and 10:

* Bytes as predicted: at `8@13/4` the proof is 150,758 B (predicted 150.8 KB), 200,008 B less
  than version 4: 205,108 B removed, 5,100 B added.
* The chain's prover work as estimated: 178.5 ms at the 32K span on dense random bits (7
  threads, loaded machine), the verifier's `W~(r)` and claim 7 ms.
* Not foreseen: Flock's block-first opening (taken for a full-height sponge, `n = 2^nu_s`, so
  for every power-of-two batch from 2,048 records on) folds each ring-switched claim's column
  bit-bank in its block rounds, so the bit claim needs the FULL bank of the sponge stack (all
  2,156 word columns), not just its 64 columns (a partial bank suffices for `s_hat_v` and
  gave failing proofs at 2,048 records; fixed in `3ae1a79`). The full bank costs one pass over
  the stack (174 to 222 ms at 8,192 records, measured), and the opening's block statistics
  and `W'` fold grow by one claim. At one full segment these costs about cancel the union's;
  at 32K the prover work changes by about 0.0 to -0.8 s locally (estimate, `docs/keccak.md`
  12.7). Against version 4 with the union proven beside the sponge instance
  (`AGG_TIED_CONCURRENT=1`, off by default) the critical-path sign is unknown, since version
  5 has no such overlap; the cloud A/B decides.
* Verifier: the Flock part's verification drops from about 80 to 110 ms to 25 to 45 ms per
  part (measured, all sizes).
* Memory (fresh process, measured at 2,048 and 8,192 records, `docs/keccak.md` 12.7): the
  verifier's peak goes down; the prover's rises by about 7% with glibc's default allocator
  but by under 1% with a fixed mmap threshold, so the rise is heap retention after the
  Flock part, not live data (the memo's -0.3 to -0.6 GiB at 32K is not borne out locally;
  the cloud run decides).
