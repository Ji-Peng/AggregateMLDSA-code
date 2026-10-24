# agg-keccak: the Keccak argument (spec Section 8)

`crates/agg-keccak` proves, for a batch of signatures, the witness-dependent hash of
ML-DSA verification, `c~_i = SHAKE256(mu_i || w1Encode(w1'_i), 32)`, with Flock HEAD
(`succinctlabs/flock` `83f963e`) and its Ligerito PCS. Deviations from the spec are
listed in `docs/deviations.md` (entries K1 to K9).

## 1. The chi-only Keccak-f gate

Module `gate` (`ChainGate`, `ChainSpec`). A gate instance proves a chain of `P`
Keccak-f[1600] permutations whose input states are absorbed from input columns and
constants, `state^(0) = absorb_0`, `state^(m) = out^(m-1) XOR absorb_m`. Committed
columns of one instance (one row of the Flock union, one `A_0` block):

| Columns | Count | Row (`C = I`, `A z * B z = z`) |
| --- | ---: | --- |
| constant one | 1 | `1 * 1 = 1`, pinned to 1 by Flock's lincheck (`const_pin_col`) |
| input bits | `n_in` | `z_i * 1 = z_i` |
| output bits | `n_out` | `out[s] * 1 = z_o` (linear row on the implicit final state) |
| chi AND outputs `t_r`, 1,600 per round | `24 * 1,600 * P` | `(B_r[x+1,y,z] + 1) * B_r[x+2,y,z] = t_r[x,y,z]` |

`B_r = phi(state_r)` with `phi = pi o rho o theta`, and
`state_{r+1} = B_r + t_r + RC_r` (chi is `B + t`, iota adds the round constant through
the constant column). theta, rho, pi, iota, the absorptions and the chaining are virtual
XOR expressions: no state is committed, so a permutation commits exactly its 38,400 chi
bits (the retired `c2d0c24` encoder also committed `state_0` and `state_24`). All rows
have degree 2.

Lincheck. Substituting the implicit states makes `A_0` and `B_0` dense (table below), so
they are never materialized. `ChainGate` implements Flock's `LincheckCircuit` by one
backward walk: with `K` the combined row weight `alpha * eq^T A + eq^T B` viewed as a
functional on the current state,

```text
comb[t_r] += K;  const += K . RC_r;  G = K + W_r (chi rows: alpha*e on B_r[x+1], e on B_r[x+2]);  K = phi^T(G)
```

round by round and permutation by permutation (at a boundary `K` lands on the absorbed
input columns and constants and carries on to the previous output). This is the retired
encoder's transpose recurrence with the A and B sides merged into one pass. Cost per fold:
`P * 24 * (1,600 + 1,600)` multiplications by `alpha` and about `P * 24 * 9,600` additions
in GF(2^128), plus reading `eq` (`2^k_log` entries).

Registry type: `k_log = ceil(log2(useful_bits))`, empty matrix stubs (Flock's walker
convention), `const_pin = Some(const column)`, IO schema = the public input fields' 128-bit
words (In) then the output words (Out). Because the registry digest only covers the
stubs, the gate digest (`ChainGate::digest`: label, rounds, layout, absorptions, outputs)
is absorbed into the Flock challenger before proving.

## 2. The SHAKE256 sponge gate

Module `sponge`. One gate instance per signature: 7 chained permutations, rate 136 bytes,
capacity 512 bits, `k_log = 19`. The gate's input fields are w1Encode (private, first) then
`mu` (public), so that the w1Encode words are the block's first 48 words (K11).

| Permutation | Absorbed into state bits | Source |
| --- | --- | --- |
| P1 | 0..511, 512..1087 | `mu` (public input field), w1Encode bytes 0..71 |
| P2..P6 | 0..1087 | w1Encode bytes `72 + 136 (m - 2)` .. (136 bytes each) |
| P7 | 0..127, 128..132, 1087 | w1Encode bytes 752..767, padding `0x1F` and `0x80` (constant column) |

State bit `s = 64 L + z` is bit `z` of lane `L = x + 5y` (FIPS 202 byte order: block byte
`b` XORs into bits `8b..8b+7`). Output: state bits 0..255 of P7's output, `c~`. Column
layout (128-bit words, `sponge::*_WORD_BASE`): w1Encode words 0..47, `mu` words 48..51, `c~`
words 52..53, the constant at bit 6,912 (word 54), word 55 zero, the chi bits from u64 112
on (`t_{m,r}` lane `L` is u64 `112 + 25 (24 m + r) + L`; `GateLayout` starts the chi words
at the next 8-word boundary after the constant). Useful bits 275,968 of 2^19 (2,156 words
per record). The w1Encode bits are in `(t, j, rho)` order, lane `t` outermost: column
`1024 t + 256 rho + j` holds stream bit `1536 rho + 6 j + t` (FIPS 204 Algorithms 16 and
28), so lane `t` is the 8 aligned words `8 t .. 8 t + 8` (`sponge::w1_column`), lanes 0..3
are words 0..31 and lanes 4, 5 words 32..47, and words 48..55 are the public region (K11).

## 3. Gate cost

Measured by `gate::tests::print_gate_costs` (`--ignored --nocapture`); structural
nonzeros count one entry per row on C plus one per XOR operand reference (the authored
DAG); flattened nonzeros are the exact entries of `A_0`, `B_0` after substituting the
implicit states (computed by GF(2) support propagation, never materialized).

| | Keccak-f gate (1 permutation) | Sponge gate (1 signature) | Sponge, per permutation |
| --- | ---: | ---: | ---: |
| block width | 2^16 | 2^19 | |
| useful bits (committed, dense stack) | 42,496 | 275,968 | 39,424 |
| rows = committed data bits | 41,601 | 275,713 | 39,388 |
| of which chi AND bits | 38,400 | 268,800 | 38,400 |
| inputs / outputs | 1,600 / 1,600 | 512 + 6,144 / 256 | |
| structural nonzeros (A + B + C) | 370,735 | 2,559,811 | 365,687 |
| flattened nonzeros A / B | 3.51e8 / 3.23e8 | 1.81e10 / 1.81e10 | 2.6e9 / 2.6e9 |

Flock's dense-stack commitment holds `ceil(useful_bits / 128)` words per live row (2,156
for the sponge), so the block padding up to 2^19 is never committed. (The rows before K11
had 275,776 and 41,792 useful bits; the chi words moved from u64 109 to u64 112.) Per signature the
spec counts 6,144 + 7 * 38,400 = 274,944 bits; the gate adds 512 (`mu`), 256 (`c~`) and the
constant (K1).

## 4. Flock integration and API

Module `prover`. `GateBatch` proves `n` instances of a gate in one union-circuit proof:

- Registry: the one gate type, capacity `2^nu`,
  `nu = max(ceil(log2 n), 3, 22 - k_log)`, plus one when `n = 2^nu < 2^11` (the full-height
  stack would select Flock's block-first opening, whose cost is about `128^2` field
  operations per used column, 2,155 columns here, independent of `n`). Dummy rows
  `[n, 2^nu)` are identically zero; Flock's count-derived constant-pin target accepts them.
- Circuit: per live instance `j` and schema word `w`, one wire class
  `{cell(w, j), public word 6 j + w}` (sponge: 4 `mu` words, then 2 `c~` words).
- Commit, then prove (spec Section 8 driver, Section 10 events 2 to 6; deviation K7):
  `GateBatch::commit(buffers)` / `commit_inputs(inputs) -> CommittedBatch` computes Flock's
  commitment before any seed exists; its `commitment()` is the event-2 payload.
  `prove_committed(committed, public, fs)` runs Flock's prover with `fs = fs_sigma` and
  asserts that Flock commits to the same cap (Flock 83f963e recommits internally; it has no
  entry that takes a precomputed commitment). `verify(public, commitment, proof, fs)` hands
  the event-2 commitment, with the verifier's own PCS parameters, to Flock's verifier.
  `prove`/`prove_buffers` return `(commitment, proof)` in one call for callers whose seed
  does not depend on the commitment (tests, benches).
- Fiat-Shamir: `FsChallenger::new("agg-mldsa/v1/keccak")`, absorb the 32-byte seed `fs`
  (spec event 5) and the gate digest, then `prove_fast_ligerito_union_circuit` binds
  registry, counts, commitment, circuit digest and public words.
- PCS: Ligerito `Fast` (rate 1/2) or `Secure`, `Profile::{Fast, Secure}`.
- Commitment bytes (`KeccakCommitment::as_bytes`): the Ligerito level-0 Merkle cap, `2^c`
  nodes of 32 bytes (`c = PcsParams::level_zero_cap_depth`; 128 nodes, 4,096 bytes, for
  every sub-batch size tested), the bytes Flock's `bind_statement` absorbs.
- Proof bytes (format version 2): `AGGKCK`, version, profile id, `n` (LE32), then bincode
  of `R1csProofCircuitMerged`; the commitment is not repeated.

`SpongeProver` (sponge front end): `commit_batch(records) -> SpongeCommitted` (commitment,
statement words and published `c~`), `prove_committed(committed, fs)`,
`verify_batch(mu, c~, commitment, proof, fs)`, and `prove_batch(records, fs) ->
SpongeSubProof { commitment, proof, c_tilde }` in one call. `commit_all(records)` splits
into sub-batches of `KeccakParams::records_per_proof()` records, the largest power of two
`n` with `7 n <= target_permutations`, and `prove_all(committed, seeds)` takes one seed
per sub-batch; the default target 2^16 gives 2^13 records (57,344 permutations), the PIOP's
segment. A single sub-batch may hold any `n` up to that (the bench proves 1,170 records,
8,190 permutations, in one proof).

## 5. Tests

- `keccak`: Keccak-f against the `keccak` crate (RustCrypto, used by `sha3`), `phi^T`
  adjoint to `phi`.
- `gate`: rows of the permutation gate on random states satisfy `a & b = z` and give
  Keccak-f (against the `keccak` crate); the walker is the transpose of the row evaluator
  for the full 24-round permutation and sponge gates (random committed vectors); the walker
  equals Flock's `SparseMatrixCircuit` fold on explicitly built matrices (2 rounds).
- `dsl_ref`: the same relation authored in Flock HEAD's Boolean DSL (virtual `xor`,
  `define_and`, `define_linear`) for 3 rounds of Keccak-f and a 2-permutation chain: our
  witness, row values and lincheck fold equal Flock's walk adapter and sparse lowering
  position by position.
- `sponge`: the gate's output equals `sha3::Shake256` for random `(mu, w1Encode)`, the
  committed `c~` words are the digest, the absorb map covers the rate and the padding.
- `witness`: the packed slot equals Flock's canonical `pack_z_lincheck` and BatchMajor
  layout, dummy rows zero.
- `tests/proofs.rs` (`--release`): Keccak-f gate round trip at 64 permutations with a wrong
  output word and a wrong seed rejected; sponge round trips at 64 and 256 signatures with a
  wrong `c~` bit, a wrong `mu` bit and a flipped proof byte rejected; a flipped w1Encode bit
  with the stale chi witness rejected; a flipped w1Encode bit with a consistent witness
  (different `c~`) against the original `c~` rejected; the `Secure` profile; the
  commit-then-prove order: two sub-batches committed, seeds derived from a transcript of
  the commitments, each proof accepted only with its absorbed commitment (a flipped cap
  bit, another witness's commitment, the other sub-batch's commitment and seeds from a
  different event-2 payload rejected; wrong-length commitments `Malformed`, not a panic).
- `prover` (unit, `--release`): `GateBatch::commit` equals the commitment Flock's own
  prover binds for 5, 64, 100 and 2,048 permutations (identity and gathered compaction,
  lane-major and plain commit); `prove_committed` panics on an altered recorded
  commitment and on a batch of another shape.

## 6. Measurements

Bench `benches/keccak_sponge.rs` (harness-free; `AGG_KECCAK_N`, `AGG_KECCAK_PROFILE`,
`AGG_KECCAK_WARMUP` (default 1), `AGG_KECCAK_REPS` (default 5)): random `(mu, w1Encode)` records, one proof,
medians of 5 timed repetitions after 1 discarded warmup run.
"prove" includes witness generation and packing; peak heap is Flock's counting allocator
(the `PeakAlloc` of Flock's `blake3_proof`/`sha2_proof` benches) over one prove; VmHWM is
the process peak RSS. Machine: i7-11700K, 7 threads (`taskset -c 0,1,3-15`), turbo and
hyper-threading off, shared with other jobs (load average 5 to 8 during the runs, so
times are upper bounds). Revision: `2c2fea6` plus the uncommitted agg-keccak changes of
this stage (2026-10-24).

| Build | Profile | Records (permutations) | Prove | Verify | Peak heap | VmHWM | Proof |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| default (no CLMUL) | Fast | 1,170 (8,190) | 89.3 s | 1,253 ms | 1,050 MiB | 1,347 MiB | 320,591 B (313.1 KiB) |
| `avx512` + `pclmulqdq` | Fast | 1,170 (8,190) | 0.92 s (8,892 perm/s) | 12.7 ms | 1,002 MiB (125 KiB/perm) | 1,381 MiB | 320,591 B (313.1 KiB) |
| `avx512` + `pclmulqdq` | Secure | 1,170 (8,190) | 0.93 s | 13.4 ms | 1,034 MiB | 1,251 MiB | 577,455 B (563.9 KiB) |
| `avx512` + `pclmulqdq` | Fast | 256 (1,792) | 0.53 s | 14.9 ms | 612 MiB | 1,021 MiB | 316,687 B |
| `avx512` + `pclmulqdq` | Fast | 64 (448) | 0.30 s | 11.4 ms | 545 MiB | 811 MiB | 288,439 B |

The 256- and 64-record rows are from a run before the last code changes of this stage
(same prover path). Witness generation and packing take 0.05 s at 1,170 records.

The rows above predate the commit-then-prove split (K7), which adds one Ligerito commit
per sub-proof. Local calibration after the split (same machine and threads, revision
`2c2fea6` plus uncommitted changes, 3 repetitions at 1,170 records, Fast, not a CPU-2
benchmark): `avx512` + `pclmulqdq` build, commit 0.046 s, prove including witness and
both commits 0.94 s, verify 13.0 ms, peak heap 990 MiB, proof 316,446 B plus a 4,096 B
commitment; default build, commit 42.0 s of 120.8 s.

Commands:

```sh
taskset -c 0,1,3-15 cargo bench -p agg-keccak --bench keccak_sponge
AGG_KECCAK_PROFILE=secure taskset -c 0,1,3-15 cargo bench -p agg-keccak --bench keccak_sponge
# CLMUL/AVX-512 build (as scripts/bench.sh AVX512=1):
RUSTFLAGS="-C target-cpu=x86-64-v4 -C target-feature=+vpclmulqdq,+pclmulqdq" CARGO_TARGET_DIR=target/avx512 \
  taskset -c 0,1,3-15 cargo bench -p agg-keccak --features avx512 --bench keccak_sponge
taskset -c 0,1,3-15 cargo test --release -p agg-keccak
taskset -c 0,1,3-15 cargo test --release -p agg-keccak --lib print_gate_costs -- --ignored --nocapture
```

Observations:

- Flock's GF(2^128) multiplication is compile-time dispatched
  (`target_feature = "pclmulqdq"`); the default build falls back to software
  multiplication and is about 100 times slower end to end. Every x86-64 CPU since 2010
  has PCLMULQDQ; enabling `+pclmulqdq` in the default build is a project decision
  (AGENTS.md: the default build must not rely on SIMD).
- Proof size is almost flat in `n` (313 KiB at 8,190 permutations, Fast); the spec's
  estimate of about 500 KiB per 2^16-permutation segment is consistent with it.
- For comparison only (different code, thread count and size): the retired `c2d0c24`
  encoder measured 0.58 s and 2.4 GB for 2^14 permutations with 15 threads
  (`ref/notes/risks.md`, R3).
- Full segments (2^13 records, 57,344 permutations, 2.3e9 committed bits) are for
  the cloud machine; extrapolation from the table, not measured: about 7 s with CLMUL
  and 7 GB peak.

## 7. Bridge slot and bridged segment (docs/bridge.md Section 5)

Module `bridge_slot` (`SlotLayout`, `BridgeSlot`) and `segment` (`BridgeBatch`,
`BridgedSegment`); the GF(2) walks are `agg_bridge::gf2_walker` (generic over
`agg_bridge::shape::Shape`, so the toy shape `L = 2^12` and the production shape share the
code). Deviations: K6 (the 48 w1Encode words in the sponge schema), K8 (`Y_sigma` as a
coefficient of the rows), K9 (full-segment scaling).

- Layout of the bridge type (one live row): input columns `c = p + Lambda t` (`Lambda =
  2^{10 + d_seg}`, `p = j + 256 rho + 1024 i_seg`), the constant at `6 Lambda`, the composite
  rows at `6 Lambda + 128 + o`, `o < 4,096` (committed zeros), `k_log = 13 + d_seg`. Rows:
  live input `z_c * 1 = z_c`; padded input and every other column empty (forced zero);
  composite row `(sum_c M[o, c] z_c + Y_o z_one) * z_one = z_{y_o}`.
- `LincheckCircuit`: `comb = alpha xi_A + xi_B` with `xi_A` on the inputs the walker
  `M_1^T M_2^T M_3^T eq_Y` (`gf2_walker::transpose_into`, parallel over lanes and level-1
  blocks) plus the identity rows, `xi_A[one] = eq[one] + sum_o Y_o eq[y_o]`,
  `xi_B[one] = eq[one] + sum_live eq[c] + sum_o eq[y_o]`; `fold_split` from the same walk;
  `const_pin_col = 6 Lambda`. The matrices are derived once from `sd`
  (`agg_bridge::seed::derive_matrices_for`) and shared (`Arc<BridgeMatrices>`).
- `BridgedSegment`: the two-type union of `n` sponge rows (bridged schema, K6) and one bridge
  row; `48 n` copy wires plus the `mu`, `c~` public wires. `commit(records)` (event 2, no
  seed), `final_vector(committed, matrices)` (`Y_sigma`, forward XOR walk),
  `prove(committed, sd, matrices, fs)` (challenger: domain, `fs`, sponge gate digest,
  `BridgeSlot::digest`), `verify(mu, c~, commitment, sd, matrices, Y, fs, proof)`. Proof
  format `AGGKCB` version 1 (kind byte 2, `n`, `d_seg`). `BridgeBatch` proves a bridge slot
  alone (kind 1), used for the toy shape.
- Cost (`BridgedSegment::cost`, exact counts): the bridge adds the factored nonzeros
  `14 * 6 Lambda + 14 * 6 * 8192 * ceil(Lambda / 2^19) + 18 * 49,152` that the walker
  touches, plus `3 * 6 * 1024 n + 3 + 2 * 4096 + |Y|` identity, constant and composite-row
  `B`, `C` entries; the flattened composite (never built) is reported by
  `gf2_walker::composite_nonzeros`. Measurements: below.

Tests (`--release`): `agg_bridge::gf2_walker` (toy composite and transpose against the
dense product `M_3 M_2 M_1`, 16 by 4,096; the transpose is the adjoint of the forward walk
on toy and production layouts; single flips change `Y` exactly when their composite column
is nonzero; the slot layout at `d_seg = 6` agrees with `agg_witness::bridge::gf2_composite`
on the main domain); `bridge_slot` (layout constants, honest rows satisfy `a & b = z` and
tampered blocks or `Y` break a row, the walker is the transpose of the row evaluator, equals
Flock's `SparseMatrixCircuit` fold on the explicitly lowered rows of the toy and of a
one-record production slot, `fold_split` agrees with two alpha folds, the digest binds `sd`,
the layout and `Y`); `tests/bridge.rs` (toy slot proven alone against a direct GF(2)
product, with a flipped `Y` bit, another seed, a malformed `Y` and a flipped committed bit
with the stale `Y` rejected; bridged segments at 5 records (`d_seg = 3`, padded positions)
and 64 records (`d_seg = 6`) round-trip with `Y` equal to the F_p side's walk, and reject a
flipped `Y` bit, another seed, a wrong `c~`, a bit flipped in the slot copy only (copy
constraint) and a bit flipped in both copies with the stale `c~` (sponge)).

Measurements at 64 records (`tests/bridge.rs`, `sponge_with_bridge_rows_round_trip_64`,
`--nocapture`; default build without CLMUL, 7 threads on `taskset -c 0,1,3-15`, revision
`3b56965` plus the uncommitted changes of this stage, 2026-10-24, single runs, not a CPU-2
benchmark):

| | Sponge sub-proof alone | Bridged segment (sponges + bridge slot) |
| --- | ---: | ---: |
| union `M` / dense `m` | 26 / 25 | 27 / 25 |
| committed 128-bit words | 137,920 | 137,920 + 3,105 |
| structural nonzeros | 163,827,904 (2,559,811 per record) | + 7,077,888 factored bridge rows + 1,189,955 identity/constant/`B`,`C` (+5.05%) |
| flattened bridge composite (never built) | | 657,057,514 entries of `A_0` (4.0 times the sponge's structural count) |
| copy wires | 384 public | 384 public + 3,072 slot copies |
| prove (both commits included) | 6.71 s | 1.62 s commit + 9.15 s prove |
| verify | 0.52 s | 1.64 s |
| proof bytes (plus the 4,096-byte commitment) | 284,294 | 343,584 |

Matrix derivation from `sd` took 0.14 s (production shape, 7 threads). At 5 records
(`d_seg = 3`) the bridge adds 2,363,427 nonzeros (18.5% of the 12.8M sponge nonzeros); the
factored count grows with `Lambda`, the identity count with the live records. Full segments
are blocked by K9 in this form.

## 8. Profile of a mode-A Flock part (2026-10-24, before the tied-part redesign)

Setup: the CLMUL build (`RUSTFLAGS="-C target-feature=+pclmulqdq"`, `target/clmul`),
`agg bench --count 512 --reps 1` with Flock's `PCS_TRACE=1 LINCHECK_TRACE=1`, 7 threads
(`taskset -c 0,1,3-15`), tree `b0d2de0`. The machine was heavily shared during the run (the
process got 134% of one CPU; the whole prove took 18.7 s against 8.2 s in
`ref/notes/bench-local.md`), so the absolute times below are about twice the quiet-machine
times; the proportions are what matters. One part of 512 records (`d_part = 9`): the union
has `nu = 10`, sponge slot `2^29` bits, bridge slot `2^32` bits (`k_log = 22`), `M = 33`,
dense stack `2^21` words (`m = 28`); 7,605 ms inside Flock's prover (commit 1,031 ms and
prove 7,778 ms in the tool's own accounting, which includes witness generation).

| Phase (Flock trace) | Time | Size | Notes |
| --- | ---: | ---: | --- |
| witness assembly (`PooledZeroed`) | 355 ms | 3 x 2^26 words = 3 GiB | the padded union buffers `z`, `a`, `b`; the bridge slot is 2^25 words of each with one live row |
| dense compaction + Ligerito commit | 86 ms | 2^21 words | count-proportional; the same as a sponge-only commit |
| zerocheck + lincheck (`M_bool = 33`) | 1,548 ms | | the wiring GKR runs concurrently on a forked transcript |
| of which union comb build | 506 ms | 2^23-word comb | the bridge walker over 2^22 columns plus the sponge walk |
| of which union partial folds | 202 ms | | count-proportional stripe folds |
| of which lincheck sumcheck | 205 ms | 2^23 entries | |
| claim assembly | 0.2 ms | 2 ring-switched + 24,630 packed-direct claims | 24,576 bridge copy words + 54 sponge schema words, one wiring gather claim each |
| merged opening | 5,573 ms | | 73% of the part; `W build + round-0 prime (2^21 words)` 4,804 ms, the per-claim weights of the 24,630 `EqPoint` claims; Ligerito inner open 630 ms |
| proof | | 676,264 B | 24,630 gather values x 16 B = 394,080 B (58%); the rest, 282 KB, is the size of a sponge-only proof (316 KB at 1,170 records) |
| verifier, this part | 678 ms | | context setup 178 ms (registry, a circuit of 24,630 wires), Flock verify 439 ms |

Memory: the tool reports 11.5 GiB prover peak RSS. The structural allocations above account
for about 6 GiB (3 GiB union buffers, the bridge slot's own prebuilt buffers of 3 x 512 MiB
plus a 512 MiB stripe, the union stripes 576 MiB, the 2^23-word lincheck vectors, the
wiring's 2^25-word cell vector); the remainder is the opening's per-claim work over 24,630
claims and Flock's pooled scratch (not separately measured).

Findings. The sponge rows are not the problem: witness, commit, partial folds and the
Ligerito inner open together cost what the sponge-only bench costs. Three costs come from
the bridge slot as designed in Section 7:

1. **Per-copied-word wiring claims.** Flock's copy constraints run over cell-slots, one per
   schema word of a type across its `2^nu` rows. The bridge type has one live row, so each of
   its `48 n` copied words is its own cell-slot with its own packed-direct gather claim: 768
   proof bytes per record (58% of the proof), 73% of the prover time in the opening, and the
   verifier's circuit setup. This cost is linear in the records with a constant that rules
   out the byte target (25 MB of gather values at `N = 2^15`) and cannot be reduced within
   the wiring argument: a cross-record linear relation needs its inputs in one block, and a
   one-row block has one cell-slot per word.
2. **Uniform row capacity (K9, K10).** The bridge slot occupies `2^(k_b + nu)` of the union
   address space, `k_b = 13 + d_part`, `nu >= d_part`: the padded buffers, the stripes and the
   zerocheck domain grow with `2^(2 d_part + 14)`, which is what forced parts of `2^9`
   records (3 GiB of union buffers; `2^13` records would need 3 x 128 GiB).
3. **The bridge walker itself**: 506 ms of comb build at `k_log = 22` (with the sponge walk),
   `7 x 10^8` additions at a full segment. This is the cost of the per-segment composite and
   is independent of 1 and 2; it also appears in the verifier (the lincheck's matrix
   discharge calls the same fold).

Decision (Section 9 and docs/bridge.md Section 5a): remove 1 and 2 by splitting every part
into two single-type Flock instances, the sponge instance (its own row capacity, no bridge
slot) and the bridge instance (`nu = 3`, one live row), and tying the bridge instance's bit
copies to the sponge instance's input words with evaluation claims on the two committed
polynomials at a random point derived from `fs_sigma`, instead of word copies through the
wiring argument. The public words `mu_i`, `c~_i` are tied by one evaluation claim against the
verifier's own table as well, so neither instance has a circuit or a wiring argument. Cost 3
stays and is now the only non-succinct verifier cost of the Keccak side.

## 9. The tied part (docs/bridge.md Section 5a, deviation K11)

Modules `flock_union` and `tied`. A part of a segment is proven by two single-type Flock
instances tied by evaluation claims instead of the two-type union of Section 7:

- `flock_union::UnionGeometry` (registry of one boolean type, declared count, PCS parameters
  derived as in Section 4): `commit(z)` (the event-2 cap of a padded slot buffer),
  `prove(buffers, circuit, claims, challenger) -> (UnionProof, cap)` and
  `verify(circuit, cap, proof, points, challenger) -> values`. The pipeline is Flock's
  boolean-only union prover (`bind_statement`, the run-list gated zerocheck over `M`, the
  union-column lincheck, the merged Ligerito opening) composed from Flock's public building
  blocks, with the caller's `(point, value)` claims added to the opening as packed-direct
  claims (the intake Flock uses for its wiring gather claims); Flock itself is unmodified.
  `eval_packed(z, point, nu)` evaluates the committed word polynomial at a point, skipping
  the columns whose eq weight is exactly zero. A `UnionProof` is Flock's
  `R1csProofMergedLigerito` plus the claim values (16 bytes each).
- `tied::TiedPart::new_part(gate, n, d, index, profile)`: the sponge instance (`n` rows of
  the sponge gate, `nu_s = capacity_log(n, 19)`) and the bridge instance (the bridge slot of
  Section 7 at `k_b = 13 + d`, one live row of `2^max(3, 22 - k_b)`). `commit(records)` /
  `commit_parts` (event 2, both caps as `TiedCommitment`), `reattach` (rebuild after the caps
  were kept), `final_vector` (the part's share), `prove` / `prove_with_final_vector(committed,
  sd, matrices, share, fs)` and `verify(mu, c~, caps, sd, matrices, share, fs, proof)`.
  `tie(fs)` returns the challenges and the five claim points of docs/bridge.md Section 5a;
  the sponge proof carries the public-region claim and the two lane-group claims, the bridge
  proof the two lane-group claims, and `verify` checks the public value against its table of
  `(mu_i, c~_i, 1, 0)` and the lane values against each other (times
  `eq(rho_i[nu_s..d], 0)`). Proof bytes: `AGGKCT`, version 1, profile, `n`, `d`, then the two
  framed `UnionProof`s. Challengers: `fs`, a label per instance, the gate digest and a digest
  of the statement words (sponge) or the slot digest (bridge), and a digest of the tie
  geometry, then Flock's statement binding.

Tests: `flock_union` (the eq-table index order and `eval_packed` against the plain inner
product; proof byte framing), `tied` (unit: the honest region values satisfy the tie
identities at `(n, d) = (5, 3), (9, 6), (3, 5)` and a flipped bridge bit breaks exactly its
lane group; the points' shape; commitment bytes), `tests/tied.rs` (release: round trips at
64 and 5 records with `Y` against the F_p walk; a flipped share, another seed, a wrong `c~`
or `mu`, another `fs`, a flipped proof byte and swapped caps rejected; a bit flipped in the
bridge copy only fails the lane tie, in both copies with the stale `c~` the public claim, a
sponge input bit with the stale chi witness the sponge rows; a 20-record segment in three
parts of 8 whose shares XOR to the segment's `Y`, with the K10 rejections). The legacy
two-type union keeps its tests (`tests/bridge.rs`).

Measured cost of the bridge instance (`ref/notes/bench-local.md`, CLMUL build, 1,024
records, `k_b = 23`): 4.2 s and 7.8 GiB, of which 3.4 s and about 6.4 GiB in one step of
Flock's merged opening, `build_merged_weight_and_prime` (trace line `W build + round-0
prime`). For every ring-switched claim (the two bit-level claims of the boolean PIOP) Flock
"bakes" a 16 x 256-entry fold table (64 KB) per used word column of the dense stack,
independent of the column's height; the bridge type has one row of `48 x 2^d + 33` used
columns (49,185 at `d = 10`), so the bake costs `2 x 49,185 x 64 KB` and `2 x 49,185 x
4,096` fold operations, and at a full segment it would cost 51 GiB. The sponge type has
2,156 columns of `n` rows, for which the bake is the right trade (the same step took 0.4 s
for the 1,024-record sponge instance). Flock's other transport (block-first, taken when
every used column has full height) costs about `128^2` field operations per column and is
no better for a wide one-row stack. The fix is a height-aware bake inside Flock (evaluate
`fold_one_slot(eq_col[c] r, table)` per slot for short columns instead of baking a table
per column; `ref/notes/bench-local.md`, "Mode A with tied parts"); without it the bridge
instance must stay at `d <= 10`, which the byte target excludes.

Sizes (exact, from `TiedPart::cost`): at 64 records (`d = 6`) the sponge instance has
`nu_s = 7`, `M = 26`, dense `m = 25` (137,984 committed words) and the bridge instance
`k_b = 19`, `M = 22`, dense `m = 22` (3,105 words); at a full segment (`d = 13`) the sponge
instance has `M = 32` (`nu_s = 13`) and the bridge instance `k_b = 26`, `M = 29`, dense
`m = 26` (393,249 words).

Measurements (`ref/notes/bench-local.md`, CLMUL build with the vendored Flock of K12, 7
threads, shared machine, one tied part per batch): the Flock part of mode A takes 0.82 /
1.37 / 2.16 / 3.73 s to commit and prove at 256 / 512 / 1,024 / 2,048 records (against 2.6
/ 5.4 / 10.8 / 21.1 s for the two-type unions in parts of 512), verifies in 108 / 164 /
288 / 527 ms, and its proof is 501 / 555 / 599 / 678 KB (sponge plus bridge proof; the
old parts cost 1.32 KB per signature). The prover's peak RSS is 1.1 / 1.3 / 1.9 / 3.1 GiB
for the whole mode-A prover (PIOP included) against 5.4 / 11.6 / 12.0 / 12.8 GiB. At
1,024 records the sponge instance takes 0.96 s and about 1 GiB (the Keccak-only bench's
cost) and the bridge instance 0.86 s, of which the walker's comb build is 0.29 s; the
bridge instance's field vectors of `2^(13 + d)` entries are the cost that grows with the
part (1 GiB each at a full segment, in prover and verifier), to be measured on the cloud
machine.
