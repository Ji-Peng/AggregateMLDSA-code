# agg-witness: witness derivation, tables and checker

Crate `crates/agg-witness` (milestone M1 of `docs/spec.md` Section 14). This note records the
exact conventions, the bounds the code enforces, and the column layouts. The spec is the
source of truth; deviations are in `docs/deviations.md` (entries W1 to W3).

## 1. Pipeline and modules

| Module | Content | Spec |
| --- | --- | --- |
| `params` | FIPS 204 Table 1/2 values (ML-DSA-44) and the derived constants, recomputed in tests | 1 |
| `ntt` | FIPS 204 Algorithms 41 to 43, generic over an NTT-friendly prime (zetas computed, checked against Appendix B) | 2 |
| `native` | ctr_q, mod+-, Decompose/HighBits/LowBits/MakeHint/UseHint (Alg. 36 to 40), SimpleBitPack/BitPack/unpack (16 to 19), HintBitPack/Unpack (20, 21), pk/sig encode/decode (22, 23, 26, 27), w1Encode (28), tr, mu, SampleInBall (29), Verify_internal (8) with a trace | 2, 3 |
| `expand` | CoeffFromThreeBytes (14), RejNTTPoly (30) with SHAKE128 block count, ExpandA (32), `ExpandedKey`, `public_t` | 2 |
| `crt` | (A z) over Z by CRT modulo q and P2 = 15 2^27 + 1; schoolbook reference | 3, B.9 |
| `instance` | `PublicInstance`: per-record verifier precomputation (keys expanded once per distinct pk) | 2 |
| `derive` | `derive` (strict), `derive_unchecked`, re-encoding to signature bytes, UseHint residual lemma | 3, 6, A |
| `digits` | (4.1) layouts, slack forms of Appendix A, leaf predicates Q_2, Q_4, Q_8, Q_16 | 4.3, A |
| `tables` | `Layout`, column maps, `BatchTables` (G_main), padding, serialization | 4 |
| `bridge` | bridge geometry, honest parities and carries for given matrices, G_1/G_2/G_3 tables, GF(2) composite | 1, 9 |
| `keccak` | Keccak-f[1600], the 7-permutation sponge trace, bit wiring | 8 |
| `check` | integer checker of every identity (Section 7, (3.1) to (3.3), (9.1), Appendix A) | 7 |
| `gen` | deterministic keys, messages and signatures through the `ml-dsa` crate | 13 |

The `ml-dsa` crate (`=0.1.1`) is used only for key generation, signing and as a test oracle;
nothing in the relation depends on its internals. Features: `h-exact` (H-exact layout),
`fips` (S1+S2 layout, `relation = fips`). Default: relation H only.

## 2. Representatives (spec Section 3.2, Appendix B.9)

Per live record, with z from sigDecode (integers in [-gamma1 + 1, gamma1]):

* A[rho][s] = ctr_q(NTT^{-1}(A_hat[rho][s])), centered in [-(q-1)/2, (q-1)/2].
* T[rho] = 2^13 (c * t1[rho]) over Z[X]/(X^256 + 1), unreduced, |T| <= B_T = 326,836,224.
* (A z) over Z, |A z| < 2^49 for |z| <= B_z; computed by CRT (exact for |z| <= 2^20,
  since l 256 (q-1)/2 2^20 < q P2 / 2), schoolbook i128 otherwise.
* w' = ctr_q(A z - T); the code asserts w' = w'_Approx of Algorithm 8 line 9 (NTT path) mod q.
* w1' = UseHint(h, w') in [0, 44); b_t = bit t of w1'.
* u = ctr_q(w' - 2 gamma2 w1').
* v = (A z - T - 2 gamma2 w1' - u) / q, asserted exact over Z.
* S1+S2 only: (r1, r0) = Decompose(w'); s = 1 iff r0 > 0 else -1; e = [r0 = -gamma2];
  v_r = r0 - 1 (s = 1), 0 (e = 1), -r0 otherwise; g = h s; quotient
  v = (A z - T - 2 gamma2 r1 - r0) / q; u_h = omega - sum h.

## 3. Bounds enforced

| Where | Bound | Value | How |
| --- | --- | --- | --- |
| Strict derivation (`derive`) | Algorithm 8 accepts | max \|z\| < gamma1 - beta, c~ = c~' | error `ZNorm`, `CommitmentMismatch` |
| Strict derivation | \|z\| <= B_z | 130,993 | invariant (implied by Alg. 8) |
| Strict derivation | \|u\| <= B_u | 190,465 | invariant (UseHint lemma; `usehint_residual_max()` = 190,465 exhaustively) |
| Strict derivation | \|v\| <= B_v | 67,068,447 | invariant ((3.4'): DELTA0_MAX = 562,061,561,757,697) |
| Strict derivation | Delta_0 = 0 mod q; CRT = NTT path mod q; S1+S2: w1' = (r1 + g) mod 44 | | invariants |
| `derive_unchecked` | sigDecode succeeds (length, HintBitUnpack) | | only check |
| Relation H tables (proven ranges) | z in [-2^17, 2^17 - 1] | digits 16,16,16,16,4 signed | `b16_prod`, `leaf4` |
| | u in [-2^18, 2^18 - 1] | digits 16,16,16,16,8 signed | `b16_prod`, `leaf8` |
| | v in [-2^26, 2^26 - 1] | digits 16^6,8 signed | `b16_prod`, `leaf8` |
| | b_t in {0, 1}; w1' in [0, 44) | (7.4) b5 (b4 + b3 b2) = 0 | `bool_b`, `range_w1` |
| H-exact tables | z in [-130,993, 130,993] exactly | z = 130,915 a_z + b_z - 130,993, b_z digits 16,16,16,16,2 unsigned | `bool_slack`, digits |
| | u in [-190,465, 190,465] exactly | u = 118,787 a_u + b_u - 190,465, b_u digits 16,16,16,16,4 unsigned | same |
| S1+S2 tables | z exact as H-exact; r1 in [0, 43] | r1 = 4 A + B + 24, A in [-4, 3], B in [-8, 7] | digits |
| | v_r in [0, gamma2 - 1] = [0, 95,231] | v_r = 29,696 a_v + b_v, b_v digits 16^4 unsigned | digits |
| | s in {-1, 1}; g in {-1, 0, 1}; e in {0, 1} | s^2 = 1; g (g-1)(g+1) = 0; e^2 = e | layout identities |
| | u_h in [0, 80] | u_h = 17 a_h + b_h, b_h digits 8,8 unsigned (signature domain) | digits |
| Bridge | kappa1, kappa2 in [0, 1023]; kappa3 in [0, 255]; y1, y2 Boolean | digits 16,16,4 and 16,16 unsigned | `b16_prod`, `leaf4`, `bool_y1`, `bool_y2` |
| Bridge (honest abort) | row weight of M_1, M_2 <= 2,047; of M_3 <= 511 | | `BridgeError::RowWeight` |
| Re-encoding | z in (-gamma1, gamma1]; sum h <= 80; UseHint(h, w') = w1' reachable | h = [HighBits(w') != w1'] (H), h = g^2 (S1+S2) | `ReencodeError` |
| Checker completeness | live: \|z\| <= B_z, \|u\| <= B_u, \|v\| <= B_v; padded rows = zero encoding | | `honest_bound`, `padding_canonical` |
| Geometry | 1 <= N <= 2^15; N* = max(next_pow2(N), 8) | | `GeometryError` |
| Columns | every committed value fits i8 | | `BuildError::Encode` |

Proven ranges and digit constants (checked by enumeration in `digits` tests, as
`spec_checks.py` item 8): C_z = C_u = 34,952, C_v = 8,947,848, C_kappa12 = 648,
C_kappa3 = 136, C_bz = 100,488, C_bu = 166,024. Q_b(Z (Z + 1)) vanishes exactly on the
balanced alphabet for b in {2, 4, 8, 16} (Q_2(P) = P).

## 4. Column layouts (G_main)

Main index x = j + 256 (rho or s) + 1024 i; z is indexed by s (column of A), all other
main-domain columns by rho. Columns are `i8`. Base-k digit alphabets are [-k/2, k/2 - 1].

| Layout | Columns in table order | Count |
| --- | --- | ---: |
| H (default) | z_0..z_4 (16,16,16,16,4), u_0..u_4 (16,16,16,16,8), v_0..v_6 (16^6, 8), b_0..b_5 | 23 |
| H-exact (`h-exact`) | bz_0..bz_4 (16,16,16,16,2), az, bu_0..bu_4 (16,16,16,16,4), au, v_0..v_6, b_0..b_5 | 25 |
| S1+S2 (`fips`) | bz_0..bz_4, az, r1_a (8), r1_b (16), bvr_0..bvr_3 (16^4), avr, s, g, e, v_0..v_6, b_0..b_5; signature domain: bh_0, bh_1 (8, 8), ah | 29 + 3 |

Canonical decompositions: (4.1) digits are the mixed-radix digits of X + eps B/2 shifted by
-b_k/2; slack forms take a = [x + offset >= 2^j], b = x + offset - w a; r1 takes
a = min(floor(r1 / 4), 7). Padded rows (i >= N) hold the encoding of the zero witness:
for H the digits of zero (-8, -8, -8, -8, 0 for z and u, -8 x6, 0 for v, bits 0); see W2
for H-exact and S1+S2 (s = -1, u_h = 80).

Physical tables shorter than 2^13 cells (the S1+S2 signature columns) are zero-extended by
the PCS layer (`padding::physical_len`).

## 5. Bridge tables (G_1, G_2, G_3)

The matrices are inputs (`BridgeMatrices`; seed derivation is `agg-bridge`'s, Section 9.1;
tests use `random_for_tests`). Index maps: zeta_1 = x mod 2^19, beta_1 = x >> 19, D_1 index
rho_1 + 2^13 beta_1; zeta_2 = (D_1 index) mod 2^17, sigma = (D_1 index) >> 17, D_2 index
rho_2 + 2^13 sigma; zeta_3 = t + 6 rho_2; D_3 index rho_3 + 2^12 sigma. Columns: G_1 =
kappa1_{t,k} at 3 t + k (18); G_2 = y2_t at t, then kappa2_{t,k} at 6 + 3 t + k (24);
G_3 = kappa3_0, kappa3_1. Y is public (D_3).

## 6. Keccak sponge (Section 8)

`keccak::sponge_trace(mu, w1Encode)` returns the 7 absorbed blocks, the state before and
after each Keccak-f[1600], and c~ (the low 256 bits of the P_7 output). Message bit
512 + 1536 rho + 6 j + t carries b_t(j, rho); it lands in permutation floor(bit / 1088)
at state bit (bit mod 1088). Tests compare against `sha3::Shake256` and the signature c~.

## 7. Checker

`check::check(instance, tables, bridge, options)` evaluates pointwise on the Boolean
domain, over the integers:

| Stage | Identities |
| --- | --- |
| 1 | `bool_b`, `range_w1` (7.4), `b16_prod` (every base-16 digit), `bool_y1` (7.1) with y1 by (7.5), `level2` (7.2), `level3` (7.3), `bool_y2` |
| 2 | `leaf16` (A^lo, A^hi = Q_16^b(P_Z)), `leaf8`, `leaf4`, `leaf2` |
| 3 | `ring` (7.6) = (3.1) at every position (A = T = 0 on padded rows), `link` (P_Z = Z (Z + 1)) |
| Keccak | (3.3) c~ = SHAKE256(mu, w1Encode(bits)) through the sponge; (9.1) Y = M3 M2 M1 b over GF(2) |
| Layout | H-exact/S1+S2 slack Booleans and e^2 = e; s^2 = 1; e (1 + s) = e v_r = e r1 = 0; g (g-1)(g+1) = 0; g (s - g) = 0; sum g^2 + u_h = omega; x (x - 44)(x + 44) = 0 with x = r1 + g - w1' |
| Completeness | padded rows canonical; honest bounds |

The evaluation reductions of stage 3 are f = f pointwise and need no check. The virtual
digit tables (P_Z, A^lo, A^hi) are materialized as the prover would (`digit_virtual`);
`check_digit_column` also accepts prover-supplied tables, so `link` and `leaf16` catch a
tampered virtual table. A failure is reported with its identity and stage.

## 8. Serialization of `BatchTables`

Little-endian: magic `AGGWTBL\0`, version u16 = 1, layout id u8 (0 H, 1 H-exact, 2 S1+S2),
d_sig u8, N u64, column count u32, then per column: domain u8 (0 main, 1 signature),
log2 length u8, cells as i8. `from_bytes` checks every header field against the layout.

## 9. Tests

`cargo test -p agg-witness --all-features --release` (also passes in debug and for every
feature subset). Integration tests: `verify` (own verifier vs crate on 160 signatures and
1,024 mutations), `witness` (128 signatures per layout: checker, re-encoding to identical
bytes accepted by the crate, CRT vs schoolbook, bridge identities, padding, serialization),
`negative` (bit flips of z, h, c~ rejected by the checker; one mutation per identity
failing at its stage; the F_p quotient cheat of Section 6), `edge` (UseHint lemma and the
S1 encoding exhaustively over Z_q; pinned signatures hitting the Decompose residue -gamma2
and |z| = B_z), `keccak` (sponge and wiring for 128 signatures). Pinned edge records:
`gen::record_hedged(0xED6E, 0, 783)` (w' coefficient = q - gamma2) and
`gen::record_hedged(0x2A4A, 0, 290)` (|z| = B_z), found by the ignored test
`edge_search`.
