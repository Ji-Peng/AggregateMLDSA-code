# agg-keccak: the Keccak argument (spec Section 8)

`crates/agg-keccak` proves, for a batch of signatures, the witness-dependent hash of
ML-DSA verification, `c~_i = SHAKE256(mu_i || w1Encode(w1'_i), 32)`, with Flock HEAD
(`succinctlabs/flock` `83f963e`, vendored `flock-core`, K12) and its Ligerito PCS. Deviations
from the spec are listed in `docs/deviations.md` (entries K1 to K20). Sections 1 to 6
describe the sponge gate and the standalone sponge proof, Section 7 the bridge slot and the
first (two-type union) segment, Sections 8 to 11 the tied part that mode A uses: Section 9
its API, Section 11 the current bridge instance (the two-type bridge union) and its
measurements, Section 12 the Flock bytes: the `Slim` profile in mode A, spans over all
segments (tied-part proof version 3), the level-0 cap, and the options measured or estimated.

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
  entry that takes a precomputed commitment). The tied part of the mode-A driver does not go
  through Flock's prover entry and opens the event-2 commitment without recommitting (K14,
  Sections 9 and 11); this standalone path still commits twice. `verify(public, commitment, proof, fs)` hands
  the event-2 commitment, with the verifier's own PCS parameters, to Flock's verifier.
  `prove`/`prove_buffers` return `(commitment, proof)` in one call for callers whose seed
  does not depend on the commitment (tests, benches).
- Fiat-Shamir: `FsChallenger::new("agg-mldsa/v1/keccak")`, absorb the 32-byte seed `fs`
  (spec event 5) and the gate digest, then `prove_fast_ligerito_union_circuit` binds
  registry, counts, commitment, circuit digest and public words.
- PCS: Ligerito `Fast` (rate 1/2), `Secure` (rate 1/2, unique decoding), `Slim` (rate 1/4), or
  `Slim8` (rate 1/8, K18), `Profile::{Fast, Secure, Slim, Slim8}` with header bytes 0, 1, 3, 4
  (2 is reserved for LaBinius); Flock's documented accounting: `Fast`, `Slim` and `Slim8` 128
  bits work-normalized with 16-bit query proof of work, `Secure` 120 bits. Bytes and times on a
  tied part: Section 11; mode A binds `Fast`, `Slim` or `Slim8` (K15, K18, Section 12).
- Commitment bytes (`KeccakCommitment::as_bytes`): the Ligerito level-0 Merkle cap, `2^c`
  nodes of 32 bytes (`c = PcsParams::level_zero_cap_depth`; Fast: 128 nodes, 4,096 bytes;
  Slim and Slim8: 64 nodes, 2,048 bytes), the bytes Flock's `bind_statement` absorbs.
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

## 9. The tied part (docs/bridge.md Sections 5a and 5b; deviations K11, K13, K14)

Modules `flock_union`, `bridge_union` and `tied`. A part of a segment is proven by two Flock
instances tied by evaluation claims instead of the two-type union of Section 7: the sponge
instance (one type) and, since proof version 2, the bridge union (two types, Section 11;
version 1 had the one-row bridge slot of Section 7 as a one-type instance, described at the
end of this section).

- `flock_union::UnionGeometry` (a registry of one or more boolean types in the caller's order,
  declared counts, PCS parameters derived as in Section 4; `new` for one type, `new_multi`
  for several, `slot(i)` maps a caller type to Flock's slot, which Flock sorts by `k_log`):
  `commit_data(slot_zs) -> CommitData` (the event-2 cap of the padded union buffer with
  Flock's prover data, its codeword and Merkle tree), `commit(slot_zs)` (the cap alone),
  `prove(buffers, circuits, points, challenger, pre) -> (UnionProof, cap)` and
  `verify(circuits, cap, proof, points, challenger) -> values`. The pipeline is Flock's
  boolean-only union prover (`bind_statement`, the run-list gated zerocheck over `M`, the
  union-column lincheck over the slots in registry order, the merged Ligerito opening)
  composed from Flock's public building blocks, with the caller's claim points added to the
  opening as packed-direct claims (the intake Flock uses for its wiring gather claims); the
  prover computes the claim values from the assembled union buffer. With `pre` the event-2
  commitment is opened as is (K14); without it Flock commits again (K7). `region_point(i,
  rows, cols)` builds a claim point on type `i`'s region (rows, word columns, slot prefix);
  `eval_packed(z, point, nu)` evaluates the committed word polynomial at a point, skipping
  the columns whose eq weight is exactly zero. A `UnionProof` is Flock's
  `R1csProofMergedLigerito` plus the claim values (16 bytes each); `size_breakdown()` splits
  its bytes into zerocheck, lincheck, opening and values.
- `tied::TiedPart::new_part(gate, n, d, index, profile)`: the sponge instance (`n` rows of the
  sponge gate, `nu_s = capacity_log(n, 19)`) and the bridge union of the part
  (`bridge_union::BridgeUnion::production_part(d, index)`, docs/bridge.md Section 5b).
  `honest_parts(records)` returns the sponge slot buffers and the part's lane-major input bits
  (`6 x 2^(10 + d)` bits, the bridge union's only `sd`-free source; the mode-A hooks tamper
  this vector). `commit(records)` / `commit_parts(sponge, bits, records)` (event 2: the sponge
  cap as `TiedCommitment`, the commit data kept in `TiedCommitted`, `take_commit_data()` to
  keep it while the buffers are dropped), `reattach(sponge, bits, records, commitment, data)`,
  `honest_blocks(committed, matrices)`, `final_vector` (the part's share), `prove`,
  `prove_with_final_vector(committed, sd, matrices, share, fs)` and `prove_with_blocks(...,
  blocks, share, fs)` (event 6: build and commit the bridge union, derive the tie, prove the
  sponge instance with its three claims and the union with its seven), and `verify(mu, c~,
  commitment, sd, matrices, share, fs, proof)`. `tie(fs, sponge cap, union cap)` returns the
  challenges and points of docs/bridge.md Sections 5a and 5b; `verify` checks the public value
  against its table of `(mu_i, c~_i, 1, 0)`, the level-1 inputs against the sponge's lane
  groups (times `eq(rho_i[nu_s..d], 0)`), the level-1 parities against their copies, and the
  levels-2-3 output columns against zero. Proof bytes (version 2): `AGGKCT`, version,
  profile, `n`, `d`, then the framed bridge-union cap, sponge proof and union proof. Since
  version 3 (K16, docs/bridge.md Section 5c) a tied part may span several segments
  (`TiedPart::new_span`), the output columns hold `M_3 M_2 y1` and are checked against the
  table of final vectors (`final_vectors`, `prove_with_final_vectors`, `verify(.., ys, ..)`
  take one vector per segment of the span), the header has a segment count, and the inner
  proofs omit the Ligerito level-0 cap (K17). Since version 4 (K20, docs/bridge.md Section
  5c, docs/soundness.md 3.6 and 4.5) the tie challenger grinds `TiedPart::tie_bits` (16 to
  18 bits at a full segment, the Johnson lists of both commitments priced in), the header
  carries the tie nonce (`LE32`, after the segment count), and both instance challengers
  absorb a 32-byte tie binding squeezed after the tie coordinates. The vendored Flock core
  prices the level-0 list at its own outer sites (K19).
  Challengers: `fs`, a label per instance (`agg-keccak/v2/tied-sponge`,
  `agg-keccak/v2/tied-bridge-union`), the gate digest and a digest of the statement words
  (sponge) or the two relation digests (union), and a digest of the tie geometry, then
  Flock's statement binding.

Tests: `flock_union` (the eq-table index order and `eval_packed` against the plain inner
product; proof byte framing), `bridge_union` and `tied` unit tests and `tests/tied.rs`
(docs/bridge.md Section 5b, Tests): layouts, rows, walkers against their row evaluators and
against Flock's `SparseMatrixCircuit`, a toy two-type union proven by Flock, the tie
identities on honest regions, round trips at 5 to 1,024 records with every tampering case
rejected under its error name, the kept commit data giving byte-identical proofs, and a
20-record segment in three parts whose shares XOR to the segment's `Y`. The legacy two-type
union keeps its tests (`tests/bridge.rs`).

Version 1 (the one-row bridge slot as the bridge instance; history, superseded by Section
11). The bridge instance was the bridge slot of Section 7 at `k_b = 13 + d`, one live row of
`2^max(3, 22 - k_b)`, committed at event 2 with the sponge instance (`TiedCommitment` carried
both caps), tied to the sponge by two lane-group claims at points derived from `fs` alone.
Measured cost of that bridge instance before the opening patch K12 (`ref/notes/bench-local.md`,
CLMUL build, 1,024 records, `k_b = 23`): 4.2 s and 7.8 GiB, of which 3.4 s and about 6.4 GiB in one step of
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

Version-1 sizes (exact): at 64 records (`d = 6`) the sponge instance has
`nu_s = 7`, `M = 26`, dense `m = 25` (137,984 committed words) and the bridge instance
`k_b = 19`, `M = 22`, dense `m = 22` (3,105 words); at a full segment (`d = 13`) the sponge
instance has `M = 32` (`nu_s = 13`) and the bridge instance `k_b = 26`, `M = 29`, dense
`m = 26` (393,249 words).

Version-1 measurements (`ref/notes/bench-local.md`, CLMUL build with the vendored Flock of
K12, 7 threads, shared machine, one tied part per batch): the Flock part of mode A takes 0.82 /
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

## 10. Profile of the tied part at 2,048 records (2026-10-24) and the two-type bridge union

Setup: CLMUL build (`RUSTFLAGS="-C target-feature=+pclmulqdq"`, `target/clmul`, tree
`50fc538`), `agg bench --count 2048 --warmup 0 --reps 1` with `AGG_TIED_TRACE=1 PCS_TRACE=1
LINCHECK_TRACE=1 VERIFY_TRACE=1`, 7 threads (`taskset -c 0,1,3-15`, `RAYON_NUM_THREADS=7`),
load average 1.2 to 2.0 from other jobs (the process got 409% of a CPU). One tied part of
2,048 records (`d = 11`): sponge instance `nu_s = 11`, `M = 30`; bridge instance `k_b = 24`,
`nu_b = 3`, `M = 27`, 98,337 committed words in one row.

| Phase | Sponge instance | Bridge instance |
| --- | ---: | ---: |
| event 2: commit (Ligerito cap) | 132 ms | 68 ms |
| lincheck: comb build (walker) / partial folds / sumcheck | 16 / 59 / 4 ms | 573 / 208 / 85 ms |
| zerocheck (by difference) | about 550 ms | about 500 ms |
| merged opening | 627 ms (block statistics 318, block rounds 71, merged sumcheck 390, inner open 201, overlapping) | 293 ms (claim values over 98,337 one-row columns 248 ms, inner open 11 ms) |
| instance total (`AGG_TIED_TRACE`, commit inside the prover included) | 1,394 ms | 1,725 ms |
| verifier (`VERIFY_TRACE`), of the part's 503 ms | about 20 ms | about 480 ms: the `2^24` eq table, the walker and the in-lockstep fold of `comb_vec` through 18 rounds |

The whole mode-A prover took 6.82 s (Flock part 3.80 s = commit 0.41 + prove 3.37; stages
1-3 1.86 s), the verifier 0.77 s uncached (Flock part 0.53 s), proof 821,955 B (Flock
677,749 B: sponge about 320 KB, bridge about 355 KB). Raw log: session scratchpad
`profile/bench-2048-trace.log` (not committed).

Findings:

1. **The bridge instance costs more than the sponge instance from `d = 11` on**, and all
   its costs scale with its column count `2^(13 + d)`: the walker's comb (`7 x 10^8` field
   additions and two 1 GiB vectors at a full segment), the lincheck's partial folds and
   sumcheck over `2^k_b` columns, the zerocheck over `2^(k_b + 3)` bits, and the opening's
   per-column claim values (the bridge type has one row, so every committed word is its
   own column). Extrapolated to `d = 13` (x4 per doubling of `d`... x4 in total): about 7 s
   locally; the cloud run measured 7.0 s of Flock prove per segment at 8,192 records of
   which the sponge instance is about 2.4 s (Keccak-only bench, 9,362 records: 3.1 s with
   its commit).
2. **The verifier's Flock cost is the bridge instance's lincheck discharge** and is linear
   in `2^k_b` (measured 0.26 s at `d = 10`, 0.50 s at `d = 11`, 1.43 s at `d = 13` on the
   cloud machine): Flock's lincheck verifier builds the `2^k_b` eq table, runs the walker,
   and folds the resulting `comb_vec` through every sumcheck round. Splitting the bridge
   into smaller parts (`d_b < d_s`) does not change the sum of these costs (the walker's
   nonzeros and the vector lengths add up to the same total) and would add one bridge
   proof (about 300 KB) per extra part, so it cannot reach the target of well under 1 s at
   32K; what does is a bridge layout whose per-block matrices are small.
3. **The duplicate Ligerito commit (K7)** costs the commit of event 2 again inside Flock's
   prover: 132 + 68 ms at 2,048 records (10% of the Flock part), about 0.9 s per segment
   on the cloud machine (3.5 s of the 33 s at 32K).

Decision (docs/bridge.md Section 5b, deviation K13): factor the bridge instance the way the
composite is factored. A **two-type bridge union** replaces the one-row bridge slot: a
level-1 type with one row per (lane, level-1 block) of the part whose `A_0` is `M_1` itself
(`I (x) M_1` is exactly Flock's per-block lincheck), and a one-row levels-2-3 type whose
rows prove the share `Y = M_3 M_2 y1`; the level-1 parities `y1` are committed in both
types and tied by evaluation claims, like the sponge inputs. At a full segment the walker
touches `14 x 2^19 + 14 x 6 x 2^17 + 18 x 49,152 = 1.9 x 10^7` entries instead of
`7.2 x 10^8`, the field vectors are `2^20` entries (16 MiB) instead of `2^26`, and the
address space `2^28` bits instead of `2^29` with 97 live rows instead of one. The
duplicate commit is removed by keeping Flock's `ProverData` from event 2 to event 6
(deviation K14). Measurements: Section 11 and `ref/notes/bench-local.md`.

## 11. The two-type bridge union (proof version 2) and its measurements (2026-10-24)

Implementation of the decision of Section 10 (docs/bridge.md Section 5b, normative;
deviations K13, K14): the bridge instance of a tied part is a Flock union of two boolean
types, `bridge_union::Level1Slot` (one row per lane and level-1 block, `A_0 = M_1`, the
level-1 parities committed) and `bridge_union::TopSlot` (one row whose inputs are the parities
of all level-1 rows, its composite rows `M_3 M_2` with `Y` as a coefficient), proven by the
multi-type `flock_union::UnionGeometry`; the sponge instance opens its event-2 commitment
without recommitting (K14). Profile `Slim` (header byte 3) was added for the byte comparison
below.

Two points of the Section 10 design were wrong and changed when implemented: (1) the level-1
parities depend on `sd`, so the bridge union cannot be committed at event 2; it is committed
at event 6 inside the part's proof (its cap travels there; the event-2 item of a part is the
sponge cap alone, wire item version 3), and the tie challenges are derived from `fs` and both
caps; (2) the composite rows only define the committed output columns, so a union committed
after `sd` needs the extra claim `z_y = 0` (seven claims instead of six; the output region
moved to the aligned base `6 G + 4096`). docs/bridge.md Section 5b gives both arguments.

Exact sizes (`TiedPart::cost`, `BridgeUnion::cost`) are tabulated in docs/bridge.md Section
5b; at a full segment the union has `M = 28`, 96 + 1 live rows, 411,712 committed words
(dense `m = 26`), and its walkers read 19,234,816 matrix entries per fold over `2^20`-entry
tables, against `M = 29`, 393,249 words and 716,537,856 entries over `2^26` entries for the
one-row slot.

Measurements. Machine and build as in Section 10 (CLMUL build `RUSTFLAGS="-C
target-feature=+pclmulqdq"`, `target/clmul`, `--features agg-keccak/avx512,agg-piop/avx512`,
7 threads `taskset -c 0,1,3-15`, `RAYON_NUM_THREADS=7`, not CPU 2, `crypto_bench off` not run;
load average 4.3 to 6.0 during the runs, partly the runs themselves, so times are upper
bounds). Revisions: "before" is the `50fc538` binary built at 13:12, "after" is `50fc538` plus
the uncommitted changes of this stage (the tree's other uncommitted changes were documents
only); both binaries ran interleaved, size by size. Medians of 5 runs after 1 warmup
(`agg bench --count N --warmup 1 --reps 5`); RSS is the maximum `VmHWM` over all runs.

| N | Prove before / after | Flock part (events 2, 6) | Verify uncached / cached | Flock part verify | Prover / verifier RSS | Aggregate (B/sig) | Flock proof + commitment |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 256 | 1,632.5 / 1,439.5 ms | 727.6 / 539.2 ms | 236.3 / 229.3 -> 203.9 / 194.7 ms | 74.6 -> 38.6 ms | 1,375 / 752 -> 1,196 / 668 MiB | 586,121 (2,289.5) -> 594,007 (2,320.3) | 501,453 + 8,218 -> 513,433 + 4,118 |
| 512 | 2,580.9 / 2,222.6 ms | 1,295.1 / 940.3 ms | 361.8 / 352.5 -> 243.9 / 229.2 ms | 140.6 -> 65.6 ms | 1,468 / 943 -> 1,374 / 728 MiB | 647,715 (1,265.1) -> 655,589 (1,280.4) | 554,669 + 8,218 -> 566,649 + 4,118 |
| 1,024 | 4,050.6 / 3,209.2 ms | 2,127.8 / 1,324.2 ms | 481.1 / 463.6 -> 255.4 / 227.4 ms | 259.8 -> 57.6 ms | 1,873 / 1,296 -> 1,700 / 879 MiB | 709,758 (693.1) -> 717,641 (700.8) | 598,533 + 8,218 -> 610,505 + 4,118 |
| 2,048 | 6,586.8 / 4,803.3 ms | 3,704.7 / 1,901.1 ms | 753.4 / 695.6 -> 304.4 / 251.1 ms | 504.7 -> 62.6 ms | 3,103 / 2,144 -> 1,942 / 1,304 MiB | 821,955 (401.3) -> 829,736 (405.1) | 677,749 + 8,218 -> 689,681 + 4,118 |

The Flock part's proof grows by about 12 KB (the bridge union's cap, 4,096 bytes, moved from
the event-2 item into the proof, and its opening is a few KB larger: Flock's lane-major
commitment opens rows in proportion to the committed words, which grow from 98,337 to
102,976 at 2,048), so the aggregate is 0.9 to 1.3% larger. Everything else of the prover is
unchanged code (the "everything else" column of `agg bench` is within noise: 2,882 ->
2,902 ms at 2,048).

Phase trace at 2,048 records (`AGG_TIED_TRACE=1 PCS_TRACE=1 LINCHECK_TRACE=1`, one run, same
build; compare the table of Section 10):

| Phase | Sponge instance | Bridge union (before: one-row bridge instance) |
| --- | ---: | ---: |
| event 2: commit (Ligerito cap) | 137 ms (kept, K14: no second commit) | none (committed at event 6) |
| event 6: blocks and parities; rows and commitment | | 4 ms; 16 ms |
| lincheck: comb build / partial folds / sumcheck | 19 / 59 / 3 ms | 58 / 22 / 11 ms (573 / 208 / 85 ms) |
| merged opening | 630 ms | 49 ms (293 ms) |
| instance total in the prover | about 1,230 ms | about 208 ms plus 20 ms of blocks and commitment (1,725 ms) |
| verifier (`AGG_TIED_TRACE` in `TiedPart::verify`) | 19 ms | 43 ms (about 480 ms) |

The full-segment bridge measured locally: a tied part of 8 live records at `d = 13` has the
full-segment bridge union (16 blocks, `M = 28`) and a small sponge instance, so it isolates
the bridge cost within this machine's batch limit (`tests/tied.rs`,
`tied_part_profiles_bytes_and_times` with `AGG_TIED_SIZES=8@13`, medians of 3 after 1
warmup; the "before" figure is the same measurement on an export of `50fc538` with its
one-row slot):

| Part `8@13` (full-segment bridge) | Prove | Verify | Peak RSS of the test process | Proof |
| --- | ---: | ---: | ---: | ---: |
| one-row bridge slot (`50fc538`) | 6,773.8 ms | 1,865.5 ms | 6,215 MiB | 581,677 B |
| bridge union (this stage) | 445.3 ms (bridge union about 345 ms plus 55 ms of rows and commitment) | 83.2 ms (bridge union about 70 ms) | 806 MiB | 593,529 B (union proof 361,060 B) |

At a full segment the bridge part of the Flock prover therefore drops from about 6.7 s to
about 0.4 s on this machine and the verifier's Flock cost per segment from 1.87 s to about
0.08 s (the cloud run of `50fc538` measured 1.43 s per segment); with four segments at
`N = 2^15` that is about 0.3 s of Flock verification instead of 6 s (estimate from the local
part figures; the cloud machine is to measure it). Per full segment the prover is then the
sponge instance (about 2.4 s on the cloud machine, Section 10).

Precomputed commitment (K14), at 2,048 records: keeping the sponge commitment's data from
event 2 to event 6 saves 217 ms of the Flock part (2,117.9 -> 1,901.1 ms; prove 5,063.1 ->
4,803.3 ms, `AGG_FLOCK_RECOMMIT=1` against the default) and costs 152 MiB of kept data (84
MiB at 1,024, 19 MiB at 256), which did not raise the prover's peak (1,959 and 1,942 MiB): the
kept data is the codeword and tree the sponge proof would otherwise rebuild at event 6, where
the peak is reached, so it only adds memory between events 2 and 6. Extrapolated, not
measured: about 0.6 GiB per full segment with `Fast` (twice that with `Slim`), 2.4 GiB at
`N = 2^15` if all four segments keep theirs; a driver short of memory can drop it
(`Aggregator::keep_commit_data(false)`).

Ligerito profiles on one tied part (`tied_part_profiles_bytes_and_times`, medians of 3 after
1 warmup; commit is event 2, prove includes the bridge union's blocks and commitment; the
proof includes the union's cap; Flock's documented security: `Fast` and `Slim` 128 bits
work-normalized in the Johnson regime with out-of-domain binding and 16-bit query proof of
work, `Secure` 120 bits in the unique-decoding regime):

| Records (`d`) | Profile | Commit | Prove | Verify | Proof (cap / sponge / union) | Event-2 cap | Kept data |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| 256 (8) | Fast | 45.0 ms | 481.3 ms | 38.5 ms | 513,425 B (4,096 / 308,124 / 201,180) | 4,100 B | 19 MiB |
| 256 (8) | Slim | 54.7 ms | 500.5 ms | 37.7 ms | 339,449 B (2,048 / 202,860 / 134,516) | 2,052 B | 38 MiB |
| 256 (8) | Secure | 43.4 ms | 450.3 ms | 39.8 ms | 733,033 B (8,192 / 446,980 / 277,836) | 8,196 B | 19 MiB |
| 512 (9) | Fast | 87.5 ms | 755.7 ms | 54.6 ms | 566,641 B (4,096 / 267,564 / 294,956) | 4,100 B | 52 MiB |
| 512 (9) | Slim | 127.8 ms | 816.8 ms | 53.7 ms | 379,577 B (2,048 / 197,212 / 180,292) | 2,052 B | 104 MiB |
| 512 (9) | Secure | 80.3 ms | 679.2 ms | 54.8 ms | 913,369 B (8,192 / 512,804 / 392,348) | 8,196 B | 38 MiB |
| 1,024 (10) | Fast | 180.3 ms | 992.9 ms | 57.4 ms | 610,497 B (4,096 / 299,060 / 307,316) | 4,100 B | 84 MiB |
| 1,024 (10) | Slim | 237.5 ms | 1,092.3 ms | 56.1 ms | 402,601 B (2,048 / 212,708 / 187,820) | 2,052 B | 168 MiB |
| 1,024 (10) | Secure | 161.9 ms | 970.7 ms | 58.3 ms | 953,369 B (8,192 / 540,340 / 404,812) | 8,196 B | 76 MiB |
| 2,048 (11) | Fast | 305.1 ms | 1,411.8 ms | 62.4 ms | 689,673 B (4,096 / 365,652 / 319,900) | 4,100 B | 152 MiB |
| 2,048 (11) | Slim | 427.7 ms | 1,459.8 ms | 61.4 ms | 442,945 B (2,048 / 245,300 / 195,572) | 2,052 B | 304 MiB |
| 2,048 (11) | Secure | 303.9 ms | 1,427.8 ms | 64.6 ms | 1,000,705 B (8,192 / 570,212 / 422,276) | 8,196 B | 152 MiB |
| 8 (13) | Fast | 4.9 ms | 476.3 ms | 87.1 ms | 593,529 B (4,096 / 228,348 / 361,060) | 4,100 B | 0.6 MiB |
| 8 (13) | Slim | 6.7 ms | 444.0 ms | 81.9 ms | 376,481 B (2,048 / 147,684 / 226,724) | 2,052 B | 1.2 MiB |
| 8 (13) | Secure | 4.9 ms | 397.1 ms | 81.5 ms | 822,001 B (8,192 / 311,052 / 502,732) | 8,196 B | 0.6 MiB |

`Slim` makes the part's proof and cap 33 to 37% smaller than `Fast` at every size (the
Ligerito opening dominates both inner proofs: at 256 records the openings are 303 KB of the
308 KB sponge proof and 197 KB of the 201 KB union proof), for 22 to 46% more commit time
(its rate-1/4 encoding), within noise on prove and verify, and twice the kept commit data.
At a full segment (estimate): the cloud run of `50fc538` measured 763,165 B of Flock proof
per segment; at `8@13` the union's proof is about 8 KB larger than the one-row slot's (593,529
against 581,677 B for the part, of which 4,096 B is the union's cap moved out of the event-2
item), so about 0.78 MB of proof per segment with `Fast` (about 95 bytes per signature at `N =
2^15`, against 93.2; the event-2 commitments halve from 1.0 to 0.5 bytes per signature) and,
by the 0.63 ratio measured at `8@13`, about 0.49 MB with `Slim` (about 60 bytes per
signature).
Recommendation: `Slim`, at the same documented 128-bit level as `Fast`, if the extra commit
time (about 0.1 s at 2,048 records) and the doubled kept data are acceptable; mode A binds the
Flock profile at event 0 through `agg_piop::binding::FlockProfile`, which has `Fast` only, so
a `Slim` mode-A run needs a `FlockProfile` variant in `agg-piop` (not part of this stage). The
`Secure` profile costs 43 to 61% more bytes than `Fast` for its more conservative accounting.
LaBinius (not integrated) would replace the two Ligerito openings (about 300 KB each per
part) by lattice commitments of about 86 KB per `2^18`-element witness with recursion
(`ref/notes/bench-cloud.md`), with the relation and the ties unchanged.

Raw logs: session scratchpad `bench/`, `bench2/` (not committed).

## 12. Flock bytes: `Slim` in mode A, spans, the level-0 cap, `Slim8` (2026-10-24)

Goal: the fewest Flock bytes per signature at Flock's documented 128-bit level (Ligerito in
the Johnson regime with two-point out-of-domain binding, every component of
`LigeritoSecurityConfig::validate` strictly above 128 bits, the query term work-normalized
with its proof of work), at a stated prover and verifier cost. Deviations K15 to K18.

### 12.1 Where the bytes are

A tied part's proof is the bridge union's cap, the sponge instance's proof and the bridge
union's proof; each instance proof is a zerocheck (about 3 KB), a lincheck (1.7 KB), the
claim values and Flock's merged opening, which is a ring-switch and Frobenius layer (13 to 15
KB) around one Ligerito proof. The Ligerito proof is almost everything: per level, `Q`
queried rows of the interleaved codeword (`lanes x 16` bytes at level 0, `2^4 x 16` above) and
their Merkle paths (`log2(codeword) - cap` siblings of 32 bytes), plus the caps, the final
block and the sum-check messages. Breakdown (`tests/tied.rs`, `AGG_TIED_BREAKDOWN=1`; bytes are
deterministic):

| Instance (records, `m`) | Profile | Level-0 rows | Level-0 paths | Levels 1+ rows / paths | Ligerito | Instance proof |
| --- | --- | --- | --- | --- | ---: | ---: |
| sponge (2,048, `m` 30) | Fast | 244 x 34 lanes: 129.6 KB | 83.9 KB (11 siblings) | 47.8 / 63.7 KB | 340,912 B | 361,556 B |
| sponge (2,048, `m` 30) | Slim | 119 x 34: 63.2 KB | 48.3 KB (13) | 40.0 / 60.1 KB | 223,856 B | 243,252 B |
| bridge union (2,048, `m` 24) | Fast | 244 x 51: 194.4 KB | 38.1 KB (5) | 31.8 / 19.8 KB | 296,608 B | 315,804 B |
| bridge union (2,048, `m` 24) | Slim | 119 x 51: 94.8 KB | 26.0 KB (7) | 25.2 / 20.8 KB | 175,536 B | 193,524 B |
| bridge union of four segments (`8@13/4`, `m` 28, `initial_k` 4) | Slim8 | 79 x 13: 16.0 KB | 34.6 KB (14) | 34.2 / 56.4 KB | 151,168 B | 169,972 B |

(The Ligerito column is the proof's `size_bytes` without the level-0 cap, which the
serialized proof no longer carries, K17.) So the bytes are set by the level-0 query count (rate and the 16-bit
query PoW), the level-0 row width (`lanes = ceil(words / 2^(m - 7 - initial_k))`), the path
depths, and by the NUMBER of openings: two per tied part, whatever its size.

### 12.2 What was implemented

1. **`Slim` in mode A** (K15): `FlockProfile::Slim = 4`, bound at event 0, named in the
   event-2 item (item version 4), `--flock-profile`. Default of mode A since this stage.
2. **Spans** (K16, docs/bridge.md Section 5c, tied-part proof version 3): a batch of `S >= 2`
   segments is ONE tied part (one sponge instance of all records, one bridge union of all
   segments), so it pays two openings instead of `2 S`. Every tied part now checks the output
   columns against the public table of final vectors (the coefficient form cannot carry a
   different `Y` per row). `--flock-span on|off`, default on. At `N <= 2^13` (one segment) a
   span is the part of version 2 with the output claim moved.
3. **The level-0 cap is not sent twice** (K17): 4,096 B (`Fast`) or 2,048 B (`Slim`) per
   instance.
4. **`Slim8`** (K18): a rate-1/8 Ligerito parameter set derived by Flock's own
   `derive_profile` and checked by its `validate()` (79 level-0 queries instead of 119),
   in the vendored core; `--flock-profile slim8`. Not the default: it is not a profile Flock
   ships (a project decision, below).

### 12.3 Measurements

(Tied-part proof versions 3 and 4, with the bridge union. Version 5 removed the union: 12.7.)

Machine: i7-11700K, 7 threads (`taskset -c 0,1,3-15`, `RAYON_NUM_THREADS=7`), CPU 2 not used,
`crypto_bench off` not run, shared with another agent's jobs. CLMUL build
(`RUSTFLAGS="-C target-feature=+pclmulqdq"`, `--features agg-keccak/avx512,agg-piop/avx512`,
`target/clmul`) of `e1e26e0` plus the then-uncommitted changes of this stage (spans, K17,
`Slim8`), committed as `710bd6b` together with the default switch to `Slim` and formatting,
which came after the runs and do not change the measured code. Medians of 5 runs after 1 warmup; RSS is `VmHWM`, the maximum over all runs.

Mode A, `agg bench --count N --warmup 1 --reps 5 --flock-profile P` (second series, profiles
in the order `slim8, slim, fast`, load average 4.1 to 7.4 including the runs' own 7 threads;
a first series at load 6.9 to 14.9 gave the same bytes and noisier times):

| N | Profile | Prove | Flock part: commit + prove | Verify uncached / cached | Prover / verifier RSS | Aggregate (B/sig) | Flock proof + commitment (B/sig) |
| ---: | --- | ---: | --- | --- | --- | --- | --- |
| 1,024 | Fast | 2,624.4 ms | 275.7 + 1,228.8 ms | 291.4 / 270.8 ms | 1,689 / 955 MiB | 709,395 (692.8) | 602,314 + 4,120 (592.2) |
| 1,024 | Slim | 2,801.2 ms | 415.7 + 1,297.1 ms | 296.6 / 264.5 ms | 1,767 / 1,091 MiB | 503,588 (491.8) | 398,514 + 2,072 (391.2) |
| 1,024 | Slim8 | 3,081.9 ms | 506.6 + 1,439.1 ms | 318.5 / 286.9 ms | 2,015 / 1,233 MiB | 424,441 (414.5) | 319,402 + 2,072 (313.9) |
| 2,048 | Fast | 4,016.0 ms | 539.2 + 1,746.3 ms | 337.0 / 290.4 ms | 1,966 / 1,247 MiB | 821,539 (401.1) | 681,490 + 4,120 (334.8) |
| 2,048 | Slim | 4,135.0 ms | 663.5 + 1,816.8 ms | 390.7 / 313.7 ms | 2,171 / 1,298 MiB | 576,894 (281.7) | 438,858 + 2,072 (215.3) |
| 2,048 | Slim8 | 4,546.1 ms | 916.4 + 1,920.0 ms | 339.9 / 276.8 ms | 2,474 / 1,649 MiB | 485,073 (236.9) | 347,010 + 2,072 (170.4) |

Against the tree before this stage (`4247f61`, Fast): 717,641 B at 1,024 records (cloud and
local), 829,736 B at 2,048; the `Fast` rows are 8,246 and 8,197 B smaller: 8,189 B in the
Flock items (the two level-0 caps, K17, less three new header bytes), the rest the Akita
opening, whose size varies by a few bytes with the transcript. Kept commit data (K14): 84 / 168 / 336 MiB at
1,024 and 152 / 304 / 608 MiB at 2,048 (Fast / Slim / Slim8).

Tied parts alone (`tests/tied.rs`, `tied_part_profiles_bytes_and_times`, `AGG_TIED_REPS=5`,
medians of 5 after 1 warmup, load 8.6 to 10.5; prove includes the bridge union's blocks and
commitment; peak RSS of the whole test process 3.5 GiB); `8@13` is a part of 8 live records at
`d = 13` (the full-segment bridge union, a small sponge instance) and `8@13/4` a span of four
full-segment geometries with the 8 records in segment 0 (the bridge union of a 32K batch):

| Part | Profile | Commit | Prove | Verify | Proof (cap / sponge / union) | Event-2 cap | Kept data |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| 2,048 (`d` 11) | Fast | 338.3 ms | 1,569.5 ms | 62.6 ms | 681,482 B (4,096 / 361,556 / 315,804) | 4,100 B | 152 MiB |
| 2,048 (`d` 11) | Slim | 460.4 ms | 1,557.4 ms | 61.3 ms | 438,850 B (2,048 / 243,252 / 193,524) | 2,052 B | 304 MiB |
| 2,048 (`d` 11) | Slim8 | 778.7 ms | 1,812.6 ms | 64.0 ms | 347,002 B (2,048 / 196,732 / 148,196) | 2,052 B | 608 MiB |
| `8@13` | Fast | 5.1 ms | 471.4 ms | 87.7 ms | 585,338 B (4,096 / 224,252 / 356,964) | 4,100 B | 0.6 MiB |
| `8@13` | Slim | 6.7 ms | 456.9 ms | 81.3 ms | 372,386 B (2,048 / 145,636 / 224,676) | 2,052 B | 1.2 MiB |
| `8@13` | Slim8 | 5.3 ms | 530.4 ms | 81.9 ms | 292,490 B (2,048 / 114,900 / 175,516) | 2,052 B | 2.4 MiB |
| `8@13/4` | Fast | 5.4 ms | 1,823.9 ms | 83.0 ms | 507,730 B (4,096 / 224,252 / 279,356) | 4,100 B | 0.6 MiB |
| `8@13/4` | Slim | 7.1 ms | 1,994.1 ms | 81.6 ms | 350,762 B (2,048 / 145,636 / 203,052) | 2,052 B | 1.2 MiB |
| `8@13/4` | Slim8 | 5.7 ms | 2,179.5 ms | 82.8 ms | 286,946 B (2,048 / 114,900 / 169,972) | 2,052 B | 2.4 MiB |

The four-segment bridge union proves in about four times the single segment's time but
verifies in the same 80 ms (its walkers run once per type over the block, whatever the row
count) and its proof is SMALLER than one segment's (`m = 28`, where Flock's configs use
`initial_k = 4`: 13 level-0 lanes instead of 51). Query PoW (`ligerito_query_grinding_cost`,
Flock's fused BLAKE3 grind on the 7-thread pool, medians of 5 grinds): 16 bits 0.22 ms, 20
bits 2.9 ms, 22 bits 5.8 ms, 24 bits 12.6 ms.

### 12.4 Options, measured or estimated

Estimates below come from a proof-size model of Ligerito (query counts from Flock's formulas,
the level shapes of its configs, cap depth `floor(log2 Q)`, rows, paths, caps, final block and
sum-check messages, plus the measured 16 to 19 KB per instance outside Ligerito); it is within
1.2% of every measured instance above (for the pre-K17 full-segment `Fast` part with level-0 caps,
the model predicts 783.2 KB against the cloud's measured 775,017 B, +1.06%; with caps stripped
post-K17 it predicts 775,046 B against ~766.8 KB). Session scratchpad `est/` (not committed).

- **(1) `Slim`**: measured above, 36% fewer Flock bytes than `Fast` at 2,048 records (30% fewer
  aggregate bytes) for 3% more prover time (119 ms: commit time 1.23x in mode A, 1.36x in the
  standalone tied part), the same verifier time and twice the kept commit data. Adopted as the default.
- **(2) One batched opening across the sponge instance and the bridge union** of a part: not
  possible within Flock's accounting. The two are committed at different events (the union
  depends on `sd`), as polynomials of different sizes; Flock's Ligerito opens one commitment,
  and a batched opening of two codewords would need its own proximity analysis. What it could
  save is one tail (levels 1+ and the merged-opening layer, about 45 KB of the union at 2,048
  records with `Slim`). Across the parts of all segments the schedule does allow merging (all
  sponge instances are committed at event 2, all unions at event 6): that is the span, which
  merges them into one commitment each and needs no new analysis.
- **(3) Query PoW and the rate.** Raising the query PoW from 16 to 20 / 22 / 24 bits removes
  about 3.5 / 5 / 7% of the queries per level (each bit is one bit of the work-normalized query
  term): estimated 13 / 20 / 27 KB less per 2,048-record `Slim` part (3 to 6%) and 16 / 24 /
  32 KB less for a 32K span (0.5 / 0.7 / 1.0 bytes per signature), for 8 to 12 grinds of
  expected ~4.2 / ~8.4 / ~18 ms each (measured medians of 5 samples: 2.9 / 5.8 / 12.6 ms; the
  geometric mean is ~1.44x the median). Not implemented (it is a new parameter set like `Slim8`,
  for a smaller gain). The rate is the larger lever: `Slim8` (implemented) cuts a part by another 21% for
  10% more prover time at 2,048 records (commit 1.4x `Slim`'s, 2x the kept data). The row
  width is another: Flock's `initial_k = 4` at `m = 28` shows what narrow level-0 rows give
  (the union of four segments is smaller than one segment's); a parameter set with smaller
  `initial_k` everywhere would trade larger level-1 commitments (prover time) for narrower
  rows, not explored further.
- **(4) Part size**: the span (implemented) makes the part the whole batch. Prover and
  verifier work are the stacked instances' (measured locally for the union: 4x prove, 1x
  verify); the prover's peak memory is one sponge instance of all `N` records, about four
  times a segment's Flock buffers and kept data (estimate at 32K with `Slim`: about 5 GiB of
  kept commit data, 6 GiB of union buffers for the sponge instance, on top of the PIOP's
  tables; to be measured on the cloud machine; the 128 GB VM is far from it).
- **(5) LaBinius** (estimate only, not integrated): it would replace the two Ligerito
  openings of a span (about 400 KB at 32K with `Slim`) by two lattice commitment openings.
  The only LaBinius proof size we measured with recursion is the whole binius64 Keccak proof
  of 3,495 permutations, 86.3 KB (bench-cloud.md); if an opening of our `2^33`-bit sponge stack
  stays near that size, a 32K span would be about 2 x 86 KB plus the 40 KB of zerocheck,
  lincheck and ring-switch layers, about 6.6 bytes per signature (an unverified
  extrapolation). It changes the PCS layer only (commitment at event 2, the opening at event
  6, the ring switch to LaBinius's field, K14's kept data), not the relation, the ties or the
  bridge.

### 12.5 Flock bytes per signature at `N = 2^15` (estimates from the model, not measured)

| Option | Per-segment parts (4 x 8,192) | One span |
| --- | ---: | ---: |
| Fast (pre-K17 cloud run 94.6, post-K17 ~93.6) | 94.6 (pre-K17) | 22.7 |
| Slim | 62.3 | 16.0 |
| Slim8 | 50.2 | 13.3 |
| Slim with 24-bit query PoW | 58.4 | 15.0 |
| LaBinius openings (extrapolation, Section 12.4) | - | about 6.6 |

With the default (`Slim`, span) the Flock side would be about 16 bytes per signature at 32K
instead of 94.6 (pre-K17 cloud run), the aggregate about 51 instead of 129.4 bytes per signature
(the other components as measured on the cloud). Prover time and memory: in a local sample at
`S = 2` (`count = 8193`, 7 threads), the span path took ~7% more prove time (Flock parts 9,513
ms vs 8,881 ms) and ~15% more peak memory (+1.2 GiB, 9,325 MiB vs 8,093 MiB) than
`--flock-span off`, while cutting proof bytes by 37% (513 KB vs 821 KB) and halving verify time
(87 ms vs 175 ms); at 32K the exact prove time and memory remain to be measured on the cloud.
Verifier time: one sponge-instance check plus one union discharge (about 80 ms locally) instead
of four parts (4 x 69 ms on the cloud).

### 12.6 Decisions and open points

- Default `Slim`, span on. `Slim8` is ready behind `--flock-profile slim8` (another 17% at
  32K, estimate); making it the default is a project decision: it is a parameter set derived
  and validated with Flock's analysis but not one Flock ships, so its proofs verify only with
  the vendored core (or upstream given the same configs).
- The span path of mode A runs only above `2^13` records: tested at the tied-part level and
  on the item level, to be run on the cloud machine (32K: bytes, time, peak memory).

### 12.7 Tied-part proof version 5: the virtual parity chain (2026-10-26)

The bridge union is gone (deviation K21, `docs/parity-chain.md`, `docs/bridge.md` 5d): a tied
part is the sponge instance plus a sum-check on its tie challenger, whose final claim the
sponge's opening proves as a third ring-switched claim. The table rows of 12.3 that split a
proof into "cap / sponge / union" therefore become "header and chain / sponge proof".

Machine and build as in 12.3 (i7-11700K, CLMUL build `RUSTFLAGS="-C
target-feature=+pclmulqdq"` in `target/clmul`, 7 threads `taskset -c 0,1,3-15`,
`RAYON_NUM_THREADS=7`, CPU 2 unused, `crypto_bench off` not run), shared with other agents'
jobs (load average 6 to 18 during the runs, so times are direction only). "v4" is code
`6803ce5` (main), "v5" is branch `parity-chain` (`3ae1a79` for the first two pairs of the
`8@13/4` and 2,048 rows, `5ef97f4` for the rest; they differ by one 20 to 30 ms pass of the
bit value); both binaries ran interleaved, the v4 test
binary with only its measurement printout fixed (it read the v3 header offsets). Bytes are
deterministic; times are medians of 5 runs after 1 warmup (`8192`: of 3), each pair run back
to back.

Tied part alone (`tests/tied.rs`, `tied_part_profiles_bytes_and_times`, `Slim`):

| Part | Proof v4 (cap / sponge / union) | Proof v5 (header and chain / sponge) | Prove v4 / v5 | Verify v4 / v5 |
| --- | --- | --- | --- | --- |
| `8@13/4` | 350,766 B (2,048 / 145,636 / 203,052) | 150,758 B (1,046 / 149,712) | 1,412.0 / 153.4 ms; 1,217.9 / 150.0 ms; 2,002.5 / 196.4 ms | 98.0 / 26.5; 97.9 / 24.6; 113.6 / 32.3 ms |
| 2,048 (`d` 11) | 438,854 B (2,048 / 243,252 / 193,524) | 248,230 B (902 / 247,328) | 863.0 / 802.9 ms; 804.6 / 752.5 ms; 1,340.3 / 1,034.2 ms | 79.6 / 28.4; 82.1 / 29.7; 90.4 / 39.9 ms |
| 8,192 (`d` 13) | 504,358 B (2,048 / 277,604 / 224,676) | 282,654 B (974 / 281,680) | 1,897.6 / 1,959.8 ms; 3,400.2 / 3,295.9 ms | 109.4 / 33.8; 138.8 / 44.0 ms |

Bytes: at `8@13/4` (the GF(2) side of a 32K batch) the proof shrinks by 200,008 B: the
union's 205,108 B (its proof, cap and framing) go and 5,100 B come (the chain's 1,028 B and,
in the sponge opening, the bit claim's `s_hat_v` and nonce, 2,056 B, and its 128 multipoint
dual values, 2,048 B, less the two lane-group values, 32 B). The memo's 3.0 to 3.6 KB
(estimate) missed the dual values.

Prover: the chain itself is cheap (the timing test `parity_chain::tests::
chain_timing_at_production_shapes` on dense random bits: 32.8 / 59.8 / 178.5 ms at `d` 11 /
13 / the 32K span `13/2`; the verifier's weight and claim 7 ms), but the third ring-switched
claim is not: at a full-height sponge (every power-of-two batch from 2,048 records on) Flock's
block-first opening needs the claim's full column bit-bank (all 2,156 word columns, Flock's
`union_bitbank_fold`, measured 174 to 222 ms at 8,192 records under load), and the opening's
block statistics and `W'` fold grow by one claim (about +60 ms at 8,192). At 8,192 records
these costs about cancel the full-segment union (v5 within the noise of v4); at `8@13/4` and
2,048 the union dominates and v5 is faster. At 32K (estimate): the union of four
full-segment geometries cost 1.2 to 2.0 s locally (`8@13/4`), the v5 additions about 0.18 s
(chain) + 4 x 0.2 s (bank, scaling with the stack) + 0.25 s (opening), so the local net is
roughly -0.3 to -0.8 s, and on the cloud machine smaller; the 32K A/B on the cloud decides it.

Mode A end to end (`agg bench --keys 16 --warmup 1 --reps 3 --flock-profile slim`):

| N | Aggregate v4 -> v5 (B/sig) | Flock proofs v4 -> v5 | Prove v4 / v5 (Flock parts) | Verify uncached / cached v4 -> v5 | Flock part verify |
| ---: | --- | --- | --- | --- | --- |
| 1,024 | 522,155 (509.9) -> 339,262 (331.3) | 398,518 -> 215,610 B | 2,308.1 (1,029.1) / 2,280.1 (1,019.8) ms | 181.3 / 193.2 -> 142.7 / 327.7 ms | 78.3 -> 36.4 ms |
| 2,048 | 598,031 (292.0) -> 407,457 (199.0) | 438,862 -> 248,238 B | 3,350.9 (1,209.3) / 2,839.9 (1,114.2) ms | 258.0 / 201.1 -> 147.5 / 146.4 ms | 79.2 -> 29.9 ms |

(The cached verify of v5 at 1,024 is an outlier of the shared machine; the prover's and
verifier's peak RSS, `VmHWM` maxima over the runs of one process, were 1,780 / 891 -> 2,109 /
978 MiB at 1,024 and 1,791 / 1,087 -> 2,094 / 1,612 MiB at 2,048: no reduction, cause not
analyzed; the per-phase trace of the tied part shows the same peak, 1,257 vs 1,259 MiB at
2,048.) CLI round trips (`agg aggregate` then `agg verify`, `keygen-batch --count 64`) at 256
and 1,024 records accept under H-u and H-exact (300,425 / 300,424 B and 339,252 / 339,258 B);
each aggregate verified under the other relation, and one with a flipped middle byte, is
rejected.

## Handoff (2026-10-24)

Written by the Fable agent at quota exhaustion; an Opus agent continues from this tree.

Status (same day, Opus agent): steps 1 to 8 below are done (Section 11, docs/bridge.md
Section 5b, deviations K13 and K14, `ref/notes/bench-local.md`, `ref/notes/status.md`); nothing
committed. Two parts of the design below were wrong and were changed: the bridge union cannot
be committed at event 2 (its parities depend on `sd`; it is committed at event 6 and its cap
is part of the proof, the tie challenges derive from `fs` and both caps), and the composite
rows need a seventh claim pinning their output columns to zero (Section 11). The tie names,
the `honest_parts` contract and the K14 memory trade-off are as planned. Open: a
`FlockProfile` variant for `Slim` in `agg-piop` (other agent), the cloud measurement of the
full segment, and the items of `ref/notes/status.md`. The text below is the original plan.

### State of the working tree

- Tree: `50fc538` plus ONE change: this file, Sections 10 and this handoff (appended). No
  code was changed; the workspace builds and every test passes as at `50fc538`. The CLMUL
  build `target/clmul/release/agg` (built 13:12 today from `50fc538`) is current.
- Profile run done (Section 10; raw log in the session scratchpad only). Load average
  during the run 1.2 to 2.0. No other measurement was taken yet.
- Nothing committed, nothing pushed (as instructed).

### What was established (read Section 10 first)

- Prover at `d = 11` (2,048 records, 7 threads): sponge instance 1.39 s, bridge instance
  1.73 s (walker comb 0.57 s, partial folds 0.21 s, zerocheck about 0.5 s, opening 0.29 s of
  which 0.25 s is the per-column claim values of a one-row stack), event-2 commits 132 +
  68 ms, repeated inside Flock's prover (K7).
- Verifier at `d = 11`: the Flock part is 503 ms, about 480 ms of it the bridge instance's
  lincheck discharge: `flock_core::lincheck::union::MatrixAssertion::check` builds a
  `2^k_b` eq table per type (`build_quirky_eq_table_from_weights`), calls
  `fold_alpha_batched` (the walker: `7 x 10^8` additions at `k_b = 26`), then
  `union_comb_partial` over the `2^k_b` comb. Cost is linear in `2^k_b`; measured 0.26 /
  0.50 / 1.43 s at `d = 10 / 11 / 13`. Hence `d_b < d_s` (smaller bridge parts) does NOT
  reduce the verifier total (same sum of nonzeros and vector lengths) and adds about 300 KB
  of proof per extra part; it was rejected on that ground, with these numbers.
- Flock's union lincheck (`vendor/flock-core/src/lincheck/union.rs`) builds each boolean
  type's comb with the TYPE's own `2^k_log` eq table, scaled by the slot prefix weight;
  the registry sorts boolean types by `k_log` descending (stable), slots are aligned at
  `offset`, `prefix`, `prefix_bits` (`flock_core::schedule::Slot`, all pub). A two-type
  union with `k_1 >= k_2` has `M = nu + k_1 + 1`; the dense floor needs `M >= 22`.
- Flock's Ligerito profiles (`vendor/flock-core/src/pcs/ligerito.rs`): `Fast` = rate 1/2,
  Johnson regime, 16-bit query PoW, 128-bit work-normalized; `Slim` = rate 1/4, same
  accounting, about half the proof, 2x the L0 encoding; `Secure` = rate 1/2 unique
  decoding, 120-bit. `agg_keccak::prover::Profile` has only `Fast` (id 0) and `Secure` (id
  1); `agg_piop::binding::FlockProfile` (not ours to edit) has `Fast = 1` only for Flock,
  so a `Slim` mode-A run needs a new `FlockProfile` variant (other agent) or a measurement
  inside `agg-keccak` only.
- `flock_core::pcs::ProverData { codeword: Vec<F128>, merkle_tree: Vec<Hash> }` is pub;
  about 1.1 GiB per full-segment sponge instance (rate 1/2 codeword of a `2^32`-bit stack).

### Design decided: the two-type bridge union (replaces the one-row bridge slot in `tied.rs`)

Notation: part of `2^d` record positions, `n` live records, `SegmentLayout` `seg`
(`lane_len = 2^(10 + d)`, `offset`, `first_block()`, `zeta1_base()`, `blocks() = 2^max(0,
d - 9)`, `block_len() L = min(2^(10 + d), 2^19)`), production shape `m_1 = m_2 = 2^13`,
`m_3 = 2^12`, `k_1 = k_2 = 14`, `k_3 = 18`, lanes 6, `J_1 = 2^19`.

Type B1 ("level 1"): one row per `(t, beta)`, row index `j = beta + blocks * t`, live rows
`6 * blocks` (rows for `t = 6, 7` are dummy and zero), row capacity `nu_rows = 3 +
log2(blocks)`. Columns: `[0, L)` the block's input bits (local `zeta`, `M_1` column
`zeta1_base + zeta`); constant at `L`; `y1` at `[L + 2^13, L + 2^14)` (`y1_col(rho_1) = L +
8192 + rho_1`, a 64-word aligned region); `useful_bits = L + 2^14`; `k_log = 11 + d` for
`4 <= d <= 9`, 15 at `d = 3`, 20 for `d >= 9`. Rows (`C = I`): inputs `z_c * one = z_c`
(ALL `L` columns, uniform across rows: padded positions are not forced zero by rows, the
tie forces them equal to the sponge's zeros); constant `one * one = one`; `y1` rows
`(sum_zeta M_1[rho_1, zeta1_base + zeta] z_zeta) * one = z_{y1(rho_1)}`; others empty.
Dummy rows have `one = 0`, so their inputs and `y1` are forced zero. Walker
(`fold_split`): `v[rho_1] = eq[y1_col(rho_1)]`; `xi_A[c] = eq[c] + sum_{rho_1 in
M_1.col(zeta1_base + c)} v[rho_1]` for `c < L`; `xi_A[one] = eq[one]`; `xi_B[one] =
eq[one] + sum_{c < L} eq[c] + sum_rho1 eq[y1_col(rho_1)]`. Cost `14 L = 7.3 x 10^6`.

Type B2 ("levels 2-3"): one live row. `G = blocks * 2^13` `y1` bits per lane; columns
`c(t, beta, rho_1) = rho_1 + 2^13 beta + G t` for `t < 6` (= `2^13 j + rho_1`, so B2's input
vector is the concatenation of the B1 rows' `y1` in row order); constant at `6 G`; `Y` rows
at `6 G + 128 + o`, `o < 4096` (committed zeros, K8); `useful = 6 G + 4224`; `k_log = 16`
for `d <= 9`, 20 at `d = 13`. Rows: inputs identity (all live), constant, `Y` rows `(sum_c
(M_3 M_2)[o, c] z_c + Y_o one) * one = z_{y_o}` with `(M_3 M_2)[o, (t, beta, rho_1)] =
sum_{rho_2} M_3[o, t + 6 rho_2] M_2[rho_2, rho_1 + 2^13 (first_block + beta)]`. Walker:
`v3[o] = eq[y_col(o)]`; `v2[t + 6 rho_2] = sum_{o in M_3.col(t + 6 rho_2)} v3[o]`;
`xi_A[c] = eq[c] + sum_{rho_2 in M_2.col(rho_1 + 2^13 (first_block + beta))} v2[t + 6
rho_2]`; `xi_A[one] = eq[one] + sum_o Y_o eq[y_o]`; `xi_B[one] = eq[one] + sum_{c < 6G}
eq[c] + sum_o eq[y_o]`. Cost `8.8 x 10^5 + 14 x 6 G = 1.2 x 10^7` at `d = 13`.

Union: `Registry::new(vec![B1, B2], nu)` with `nu = max(nu_rows, 3, 21 - max(k_1, k_2))`
(so `M = nu + max(k) + 1 >= 22`); counts `[6 blocks, 1]`; slot order is the registry's
(`k_log` descending, stable; at `d <= 4` B2 sorts first: look the slot indices up, do not
assume). B1 is never full height (6 of 8 rows per block group), so Flock's block-first
opening is never selected. The event-2 cap of the part is `(sponge cap, bridge-union cap)`
as `TiedCommitment` today. Honest witness from the part's lane-major input bits `bits`
(the current `bridge_z` input, `input_bits / 64` words): B1 row `(t, beta)` inputs = words
`[(beta L + lane_len t) / 64, + L / 64)` of `bits`, `y1 = M_1` (block bits) by XOR-gathering
`M_1.column(zeta1_base + zeta)` over set bits (parallel over the 96 rows), constant 1; B2 =
the `y1` concatenation, constant 1; share `Y = M_3 M_2 y1` (levels 2-3 walk as in
`gf2_walker::composite`).

Claims (packed-direct claims in the bridge union's merged opening; word coordinates
`M - 7`, rows low, then the slot's `k_log - 7` column coordinates, then the slot's
`prefix_bits` as 0/1 constants; `lane_bits(h)` = `[t0, t1, 0]` for lanes 0..3 and `[t', 0,
1]` for lanes 4, 5 as in `tied::TiePoints`; challenges from `TieChallenges::derive` plus
fresh `rho_beta` (`log2(blocks)` coords) and `rho_y` (6 coords)):

- `P_b1_h` (B1 inputs): rows `[rho_i[9..d] (beta) | lane_bits(h) | 0 ...]`, columns
  `[rho_w (3) | rho_i[0..min(d, 9)] (i_lo) | 0 ...]`, B1 prefix. Check `v_b1_h =
  eq(rho_i[nu_s..d], 0) * v_s_h` against the sponge's lane claims (unchanged `P_s0`,
  `P_s1`).
- `P_y1_h` (B1 `y1` region): rows `[rho_beta | lane_bits(h) | 0 ...]`, columns `[rho_y (6) |
  bits of (L / 8192 + 1) as constants]`, B1 prefix.
- `P_b2_h` (B2 inputs): rows all zero, columns `[rho_y (6) | rho_beta | lane_bits(h) | 0
  ...]`, B2 prefix. Check `v_b2_h = v_y1_h`.
- The sponge instance keeps its three claims and the public-table check.

Soundness (to be written into docs/bridge.md Section 5b): Flock's knowledge soundness gives
committed `z_s` and `z_u` (the union: B1 rows and the B2 row). B1's rows prove, per live
row, `y1 = M_1 b'(t, beta)` over the part's columns and pin the constant (1 on live, 0 on
dummy rows, so dummy inputs and `y1` are zero); B2's rows prove `Y_share = M_3 M_2 y1''`
for its input copies `y1''`, with `Y_share` bound as a coefficient (K8) and in the digest.
Tie 1: `f_b1(rho)` and `f_s(rho)` are the multilinear extensions, in the same `d + 5` free
coordinates `(i, w, t)`, of the B1 inputs and of the sponge inputs over the part's `2^d`
record positions (sponge dummy rows zero by Flock's witness contract; B1 positions beyond
the live records are compared against those zeros), so a differing bit is caught except
with probability `(d + 5) / 2^128` per claim (Schwartz-Zippel; `rho` is derived from `fs`,
after both caps were absorbed at event 2). Tie 2: `f_y1(rho)` and `f_b2(rho)` are the MLEs,
in the `log2(blocks) + 3 + 6` free coordinates `(beta, t, rho_1-word)`, of B1's `y1` and of
B2's copies; error `(log2(blocks) + 9) / 2^128` per claim. Hence `Y_share = M_3 M_2 M_1
b'_part` over GF(2) with the segment's matrices; the shares XOR to `Y_sigma` (K10); bound
(9.3) of docs/bridge.md Section 7 unchanged; the tie error `2 (d + 5) / 2^128 + 2 (log2
(blocks) + 9) / 2^128` per part plus the public claim, inside `epsilon_Flock`. Replay
protection: distinct challenger labels, the B1 and B2 digests (`sd`, shape, `lane_len`,
`offset`, `blocks`, `k_log`, `useful`, constant and `y` bases, and `Y` for B2) and the tie
geometry digest, then Flock's `bind_statement` (registry digest, counts, cap).

Expected cost at a full segment (estimate): bridge union prover well under 0.5 s (walker
`1.9 x 10^7`, address space `2^28` bits with 97 live rows, dense stack `2^19` words, eq
and comb vectors 16 MiB); verifier about 20 to 40 ms per segment (two folds at `2^20`
tables); proof about the size of today's bridge proof or smaller (`M = 28` instead of 29;
six claim values instead of two).

### Next steps, in order

1. `crates/agg-keccak/src/flock_union.rs`: generalize `UnionGeometry` to a vector of boolean
   types and counts (`new_multi(types, counts, nu, profile)`; keep `new(ty, n, nu, profile)`
   as the one-type wrapper). `commit(slot_zs: &[&[F128]])` assembles the padded union
   buffer (single full slot: pass through). `prove(buffers: Vec<SlotBuffers>, circuits:
   &[&dyn LincheckCircuit], points: &[Vec<F128>], ch, pre: Option<CommitData>)`: assemble
   with `union.assemble_witness`, compute the claim VALUES itself with `eval_packed` on the
   assembled `z` (the padded union buffer), one `UnionLincheckSlot` per slot in registry
   order; skip `commit_stack` when `pre` is given (`CommitData { commitment, prover_data }`
   from a new `commit_data`); `verify(circuits, ...)` likewise. `k_log()` becomes per type.
   The one-type sponge path must stay byte-identical (tests/proofs.rs, tests/tied.rs).
2. New `crates/agg-keccak/src/bridge_union.rs` (add to `lib.rs`): `Level1Layout`,
   `Level1Slot` (`LincheckCircuit` + `apply_ab` + `digest`), `TopLayout`, `TopSlot`
   (`LincheckCircuit` + `apply_ab` + `parities` + `digest`), `BridgeBlocks { rows, y1, top }`
   with `honest_blocks(bits)`, and `BridgeUnion { l1, top, geometry, l1_slot, top_slot }`
   with the tie points above. Unit tests as in `bridge_slot.rs`: layouts, honest rows
   satisfy `a & b = z` and tampering breaks a row, walker adjoint of the row evaluator (toy
   shape `Shape::TOY` with `lanes = 1`, `m_1 = 64`: keep the layouts generic over `Shape`,
   require `m_1 % 128 == 0` and 6 lanes only for the tie points; and a production part at
   `d = 3`), equality with `flock_core::lincheck::SparseMatrixCircuit` on lowered rows (B1
   at `d = 3`, B2 on the toy), digests bind `sd`, layout and `Y`.
3. `tied.rs`: replace the bridge instance by `BridgeUnion`; `honest_parts` keeps returning
   `(SlotBuffers, Vec<u64>)` with the lane-major bits (the mode-A hook `PartWitness` and
   `crates/agg-cli/tests/mode_a.rs` lines 691-703 flip a bit in that vector and expect the
   tie error for the bridge copy); `commit_parts` builds `BridgeBlocks` from the bits, plus
   a lower-level `commit_blocks(sponge, blocks, records)` for `y1`-tampering tests;
   `TiedCommitted` keeps `Option<CommitData>` for both instances; proof `VERSION = 2`,
   `TiedCost` updated; `verify` checks `v_b1_h = factor * v_s_h` and `v_b2_h = v_y1_h`.
   Error names: `Tie("bridge copy of lanes 0..3")`, `Tie("bridge copy of lanes 4, 5")`
   (keep: mode-A tests match them), `Tie("level-1 parities of lanes 0..3")`, `..."4, 5"`.
4. `crates/agg-cli/src/mode_a.rs`: keep each part's `CommitData` from event 2 to event 6
   (`PartState.commit_data`, config `keep_commit_data: bool` default true; document the
   memory: about 1.1 GiB per full segment, 4.4 GiB at 32K, against the K7 double commit of
   about 0.9 s per segment on the cloud); otherwise rebuild as today. `wire.rs`: item
   version 3 if the item bytes change (they do not have to: the cap framing is unchanged).
5. Tests: `tests/tied.rs` round trips at 5 (`d = 3`), 64 (`d = 6`), 256 (`d = 8`), 512
   (`d = 9`, exactly one level-1 block), 1,024 (`d = 10`, two blocks per lane: the first
   multi-block case, essential) and the 20-in-parts-of-8 share test; tampering: B1 copy
   bit only (tie 1), B2 `y1` copy bit only (tie 2), B1 `y1` bit only (Flock rejects the
   B1 rows), flipped share, another `sd`, wrong `c~`/`mu`, another `fs`, flipped proof byte,
   swapped caps. Keep `taskset -c 0,1,3-15 RUSTUP_TOOLCHAIN=stable cargo test --release
   --workspace` green (the legacy `segment.rs`/`tests/bridge.rs` stay untouched).
6. Measure mode A at 256, 512, 1,024, 2,048 (`--warmup 1 --reps 5`, CLMUL build,
   `RAYON_NUM_THREADS=7`, state the load average), plus one `AGG_TIED_TRACE=1 PCS_TRACE=1
   LINCHECK_TRACE=1 VERIFY_TRACE=1` run at 2,048 for the before/after phase table; append
   to `ref/notes/bench-local.md` and update the Flock items of `ref/notes/status.md`
   (open problems 2, 3, 3a, 5, 7 and the measured table).
7. Goal 3 (bytes): add `Profile::Slim` (id 3; id 2 is reserved for LaBinius) to
   `prover.rs` and measure Fast / Slim / Secure proof bytes and times of a tied part at 256
   to 2,048 with an ignored test or the bench; report with Flock's documented security
   (Fast and Slim 128-bit work-normalized with PoW, Secure 120-bit UDR); recommend Slim if
   the commit cost (2x L0 encoding, about 10% of the sponge instance) is acceptable; note
   that mode A needs an `agg_piop::binding::FlockProfile` variant (other agent) to bind it
   at event 0. LaBinius note: it would replace the Ligerito opening (about 300 KB per
   instance) by a lattice commitment of about 86 KB per `2^18`-element witness with
   recursion (bench-cloud.md), unchanged relation and ties; not integrated.
8. Docs: docs/bridge.md Section 5b (the design above, normative), docs/deviations.md K13
   (two-type bridge union) and K14 (precomputed commitment, updating K7), docs/keccak.md
   Section 11 (measurements) and a refresh of Sections 4 and 9 where they describe the
   one-row bridge instance as current.
