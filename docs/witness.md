# agg-witness: witness derivation and reference checks

Crate `crates/agg-witness` (`docs/spec.md` Section 14). This note records the
exact conventions and the bounds the code enforces; the committed column layouts are
`agg-piop`'s (`docs/lookup.md`, `docs/h-exact.md`). The spec is the
source of truth; deviations are in `docs/deviations.md` (entries W1 to W3).

## 1. Pipeline and modules

| Module | Content | Spec |
| --- | --- | --- |
| `params` | FIPS 204 Table 1/2 values (ML-DSA-44, 65 and 87) and the derived constants, recomputed in tests | 1 |
| `ntt` | FIPS 204 Algorithms 41 to 43, generic over an NTT-friendly prime (zetas computed, checked against Appendix B) | 2 |
| `native` | ctr_q, mod+-, Decompose/HighBits/LowBits/MakeHint/UseHint (Alg. 36 to 40), SimpleBitPack/BitPack/unpack (16 to 19), HintBitPack/Unpack (20, 21), pk/sig encode/decode (22, 23, 26, 27), w1Encode (28), tr, mu, SampleInBall (29), Verify_internal (8) with a trace | 2, 3 |
| `expand` | CoeffFromThreeBytes (14), RejNTTPoly (30) with SHAKE128 block count, ExpandA (32), `ExpandedKey`, `public_t` | 2 |
| `crt` | (A z) over Z by CRT modulo q and P2 = 15 2^27 + 1; schoolbook reference | 3, B.9 |
| `instance` | `PublicInstance`: per-record verifier precomputation (keys expanded once per distinct pk) | 2 |
| `derive` | `derive` (strict), `derive_unchecked`, re-encoding to signature bytes, UseHint residual lemma | 3, 6, A |
| `digits` | (4.1) layouts, the step form (4.2), the bit-slack form (4.3) and the slack form of H-FIPS's L, leaf predicates Q_2, Q_4, Q_6, Q_8, Q_16 | 4.3, A |
| `bridge` | bridge geometry, honest parities and carries for given matrices, G_1/G_2/G_3 tables, GF(2) composite | 1, 9 |
| `keccak` | Keccak-f[1600], the 7-permutation sponge trace, bit wiring | 8 |
| `hfips` | relation H-FIPS: per-coefficient witness, derivation, integer checker of every identity, signature extraction | A |
| `check` | failure reports of the H-FIPS checker (`Identity`, `CheckReport`) | A |
| `gen` | deterministic keys, messages and signatures through the `ml-dsa` crate | 13 |

The `ml-dsa` crate (`=0.1.1`) is used only for key generation, signing and as a test oracle;
nothing in the relation depends on its internals. The crate has no features.

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

## 3. Bounds enforced

| Where | Bound | Value | How |
| --- | --- | --- | --- |
| Strict derivation (`derive`) | Algorithm 8 accepts | max \|z\| < gamma1 - beta, c~ = c~' | error `ZNorm`, `CommitmentMismatch` |
| Strict derivation | \|z\| <= B_z | 130,993 | invariant (implied by Alg. 8) |
| Strict derivation | \|u\| <= B_u | 190,465 | invariant (UseHint lemma; `usehint_residual_max()` = 190,465 exhaustively) |
| Strict derivation | \|v\| <= B_v | 67,068,447 | invariant ((3.4'): DELTA0_MAX = 562,061,561,757,697) |
| Strict derivation | Delta_0 = 0 mod q; CRT = NTT path mod q | | invariants |
| `derive_unchecked` | sigDecode succeeds (length, HintBitUnpack) | | only check |
| Bridge (honest abort) | row weight of M_1, M_2 <= 2,047; of M_3 <= 511 | | `BridgeError::RowWeight` |
| Re-encoding | z in (-gamma1, gamma1]; sum h <= 80; UseHint(h, w') = w1' reachable | h = [HighBits(w') != w1'] | `ReencodeError` |
| H-FIPS witness (`HFipsRecord::from_witness`) | \|z\| <= B_z (bit-slack digits in their alphabets); every coefficient's (F1) to (F6) | | `HFipsError` |
| Geometry | 1 <= N <= 2^15; N* = max(next_pow2(N), 8) | | `GeometryError` |

Digit constants and ranges (checked by enumeration in `digits` tests, as `spec_checks.py`
item 8). The H-u step form of u with one 16-bit low digit,
u = U_lo + 63,079 U_hi + 31,540, covers exactly [-190,465, 190,465] because
5 * 63,079 + 2^16 = 2 B_u + 1 and 63,079 <= 2^16 (exhaustive test
`u_step_form_is_exact_at_the_slack_boundaries`; +-190,466 have no in-alphabet digits). The
bit-slack form of z, z = S_zlo + 2^16 B_zhi + 130,915 a_z - 98,225, covers exactly
[-B_z, B_z] (test `z_bit_slack_form_is_exact_at_the_fips_bound`), and H-FIPS's
L = S_lo + 29,696 L_hi + 2^15 exactly [0, gamma2 - 1]. Q_b(Z (Z + 1)) vanishes exactly on the
balanced alphabet for b in {2, 4, 6, 8, 16} (Q_2(P) = P, Q_6(P) = P (P - 2)(P - 6)).

## 4. Canonical decompositions

(4.1) digits are the mixed-radix digits of X + eps B/2 shifted by -b_k/2 (base-k alphabets
are [-k/2, k/2 - 1]); slack forms take a = [x + offset >= 2^j], b = x + offset - w a; the
H-u step form takes H_u = min(floor((u + B_u) / 63,079), 5), B_ulo = u + B_u - 63,079 H_u;
the bit-slack form takes a_z = [z + B_z >= 2^17], then B_zhi and B_zlo from the rest. Outside
the proven range the top digit absorbs the excess and leaves its alphabet.

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

## 9. Tests

`cargo test -p agg-witness --release` (also passes in debug). Integration tests: `verify`
(own verifier vs crate on 160 signatures and 1,024 mutations), `witness` (128 signatures: the
derived values satisfy (3.1), the bounds and UseHint, re-encoding gives identical bytes
accepted by the crate, CRT vs schoolbook; padded records have T = 0), `negative` (bit flips
of z, h, c~ rejected by the H-FIPS checker, at derivation or by |z| > B_z; the F_p quotient
cheat of Section 6; an unreachable w1' has no hint), `edge` (UseHint lemma exhaustively
over Z_q; pinned signatures hitting the Decompose residue -gamma2 and |z| = B_z, checked
under H-FIPS and re-encoded), `hfips` (`docs/h-exact.md` Section 10.5), `keccak` (sponge and
wiring for 128 signatures). Pinned edge records: `gen::record_hedged(0xED6E, 0, 783)`
(w' coefficient = q - gamma2) and `gen::record_hedged(0x2A4A, 0, 290)` (|z| = B_z), found by
the ignored test `edge_search`.

## ML-DSA-65 and ML-DSA-87 witnesses

The witness uses exact `(k, l)` dimensions `(6, 5)` and `(8, 7)` respectively, with per-polynomial lanes. Both use the H-u digit-slack forms in `docs/spec.md`, four w1 bits per coefficient, and 48-byte or 64-byte c~ respectively. The `ParamSet` constants and digit encoders in `crates/agg-witness/src/params.rs` and `digits.rs` define these layouts.
