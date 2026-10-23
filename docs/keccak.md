# agg-keccak: the Keccak argument (spec Section 8)

`crates/agg-keccak` proves, for a batch of signatures, the witness-dependent hash of
ML-DSA verification, `c~_i = SHAKE256(mu_i || w1Encode(w1'_i), 32)`, with Flock HEAD
(`succinctlabs/flock` `83f963e`) and its Ligerito PCS. Deviations from the spec are
listed in `docs/deviations.md` (entries K1 to K6).

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
capacity 512 bits, `k_log = 19`.

| Permutation | Absorbed into state bits | Source |
| --- | --- | --- |
| P1 | 0..511, 512..1087 | `mu` (public input field), w1Encode bytes 0..71 |
| P2..P6 | 0..1087 | w1Encode bytes `72 + 136 (m - 2)` .. (136 bytes each) |
| P7 | 0..127, 128..132, 1087 | w1Encode bytes 752..767, padding `0x1F` and `0x80` (constant column) |

State bit `s = 64 L + z` is bit `z` of lane `L = x + 5y` (FIPS 202 byte order: block byte
`b` XORs into bits `8b..8b+7`). Output: state bits 0..255 of P7's output, `c~`. Column
layout (128-bit words): `mu` words 0..3, w1Encode words 4..51, `c~` words 52..53, the
constant at bit 6,912, the chi bits from u64 109 on (`t_{m,r}` lane `L` is u64
`109 + 25 (24 m + r) + L`). Useful bits 275,776 of 2^19. The w1Encode bits are in
`(t, j, rho)` order, lane `t` outermost: column `1024 t + 256 rho + j` holds stream bit
`1536 rho + 6 j + t` (FIPS 204 Algorithms 16 and 28), so lane `t` is 8 aligned words
(`sponge::w1_column`).

## 3. Gate cost

Measured by `gate::tests::print_gate_costs` (`--ignored --nocapture`); structural
nonzeros count one entry per row on C plus one per XOR operand reference (the authored
DAG); flattened nonzeros are the exact entries of `A_0`, `B_0` after substituting the
implicit states (computed by GF(2) support propagation, never materialized).

| | Keccak-f gate (1 permutation) | Sponge gate (1 signature) | Sponge, per permutation |
| --- | ---: | ---: | ---: |
| block width | 2^16 | 2^19 | |
| useful bits (committed, dense stack) | 41,792 | 275,776 | 39,397 |
| rows = committed data bits | 41,601 | 275,713 | 39,388 |
| of which chi AND bits | 38,400 | 268,800 | 38,400 |
| inputs / outputs | 1,600 / 1,600 | 512 + 6,144 / 256 | |
| structural nonzeros (A + B + C) | 370,735 | 2,559,811 | 365,687 |
| flattened nonzeros A / B | 3.51e8 / 3.23e8 | 1.81e10 / 1.81e10 | 2.6e9 / 2.6e9 |

Flock's dense-stack commitment holds `ceil(useful_bits / 128)` words per live row (2,155
for the sponge), so the block padding up to 2^19 is never committed. Per signature the
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
`AGG_KECCAK_REPS`): random `(mu, w1Encode)` records, one proof, medians of 3 repetitions.
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
