# agg-keccak: the Keccak argument (spec Section 8)

`crates/agg-keccak` proves, for a batch of signatures, the witness-dependent hash of
ML-DSA verification, `c~_i = SHAKE256(mu_i || w1Encode(w1'_i), 32)`, with Flock HEAD
(`succinctlabs/flock` `83f963e`, vendored `flock-core`, K12) and its Ligerito PCS. Deviations
from the spec are listed in `docs/deviations.md` (the K entries). Sections 1 to 6 describe
the sponge gate and the standalone sponge proof, Section 9 the tied part that mode A uses,
Section 12 the Flock bytes: the mode-A profile (`Fast` by default, or `Slim`), spans over all
segments, the level-0 cap, the options measured or estimated, and (12.7) tied-part proof
version 5, the virtual parity chain.

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
| chi AND outputs `t_r`, 1,600 per round | `(24 P - 1) * 1,600 + 64 L` | `(B_r[x+1,y,z] + 1) * B_r[x+2,y,z] = t_r[x,y,z]` |

`B_r = phi(state_r)` with `phi = pi o rho o theta`, and
`state_{r+1} = B_r + t_r + RC_r` (chi is `B + t`, iota adds the round constant through
the constant column). theta, rho, pi, iota, the absorptions and the chaining are virtual
XOR expressions: no state is committed, so a permutation commits exactly its 38,400 chi
bits. All rows have degree 2.

Pruned last round. The final state is read only by the output rows, so the last round of
the last permutation commits only the lanes `0 .. L` that hold an output bit,
`L = 1 + max_s floor(s / 64)` over the output bits `s` (`GateLayout::last_lanes`,
`ChainGate::round_lanes`): its other `25 - L` lanes have no column and no row, and the
walker, the row evaluator and the witness writers skip them. Round 22 cannot be pruned:
every bit of `B_23` depends, through the theta column parities, on all of `state_23`. The
removed AND bits are unconstrained variables that nothing reads, so (K) holds by
construction. The Keccak-f gate outputs all 1,600 bits (`L = 25`, nothing pruned); the
sponge outputs `c~`, lanes `(0..3, 0)`, so `L = 4` and round 23 of P7 commits 256 of its
1,600 AND bits.

Lincheck. Substituting the implicit states makes `A_0` and `B_0` dense (table below), so
they are never materialized. `ChainGate` implements Flock's `LincheckCircuit` by one
backward walk: with `K` the combined row weight `alpha * eq^T A + eq^T B` viewed as a
functional on the current state,

```text
comb[t_r] += K;  const += K . RC_r;  G = K + W_r (chi rows: alpha*e on B_r[x+1], e on B_r[x+2]);  K = phi^T(G)
```

round by round and permutation by permutation (at a boundary `K` lands on the absorbed
input columns and constants and carries on to the previous output). Cost per fold:
`(24 P - 1) * 3,200 + 128 L` multiplications by `alpha` and about `P * 24 * 9,600` additions
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
at the next 8-word boundary after the constant). Round 23 of P7 keeps lanes 0..3 (the
lanes of `c~`, Section 1), so the chi words end at u64 `112 + 25 * 167 + 4 = 4,291`.
Useful bits 274,624 of 2^19 (2,146 128-bit words per record, the last one half used). The w1Encode bits are in `(t, j, rho)` order, lane `t` outermost: column
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
| useful bits (committed, dense stack) | 42,496 | 274,624 | 39,232 |
| rows = committed data bits | 41,601 | 274,369 | 39,196 |
| of which chi AND bits | 38,400 | 267,456 | 38,208 |
| inputs / outputs | 1,600 / 1,600 | 512 + 6,144 / 256 | |
| structural nonzeros (A + B + C) | 370,735 | 2,551,747 | 364,535 |
| flattened nonzeros A / B | 3.51e8 / 3.23e8 | 1.79e10 / 1.79e10 | 2.6e9 / 2.6e9 |

The sponge's per-permutation column is the mean over its 7 permutations (P7 commits
38,400 - 1,344 = 37,056 chi bits, Section 1). Flock's dense-stack commitment holds
`ceil(useful_bits / 128)` words per live row (2,146 for the sponge), so the block padding
up to 2^19 is never committed. Per signature the spec counts 6,144 + 7 * 38,400 = 274,944
bits; the gate prunes 1,344 of round 23 of P7 and adds 512 (`mu`), 256 (`c~`) and the
constant (K1).

## 4. Flock integration and API

Module `prover`. `GateBatch` proves `n` instances of a gate in one union-circuit proof:

- Registry: the one gate type, capacity `2^nu`,
  `nu = max(ceil(log2 n), 3, 22 - k_log)`, plus one when `n = 2^nu < 2^11` (the full-height
  stack would select Flock's block-first opening, whose cost is about `128^2` field
  operations per used column, 2,146 columns here, independent of `n`). Dummy rows
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
  Section 9); this standalone path still commits twice. `verify(public, commitment, proof, fs)` hands
  the event-2 commitment, with the verifier's own PCS parameters, to Flock's verifier.
  `prove`/`prove_buffers` return `(commitment, proof)` in one call for callers whose seed
  does not depend on the commitment (tests, benches).
- Fiat-Shamir: `FsChallenger::new("agg-mldsa/v1/keccak")`, absorb the 32-byte seed `fs`
  (spec event 5) and the gate digest, then `prove_fast_ligerito_union_circuit` binds
  registry, counts, commitment, circuit digest and public words.
- PCS: Ligerito `Fast` (rate 1/2), `Secure` (rate 1/2, unique decoding), `Slim` (rate 1/4), or
  `Slim8` (rate 1/8, K18), `Profile::{Fast, Secure, Slim, Slim8}` with header bytes 0, 1, 3, 4
  (2 is unassigned); Flock's documented accounting: `Fast`, `Slim` and `Slim8` 128
  bits work-normalized with 16-bit query proof of work, `Secure` 120 bits. Bytes and times on a
  tied part: Section 12; mode A binds `Fast`, `Slim` or `Slim8` (K15, K18, Section 12).
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
- `tests/all/proofs.rs` (`--release`): Keccak-f gate round trip at 64 permutations with a wrong
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
times are upper bounds).

| Build | Profile | Records (permutations) | Prove | Verify | Peak heap | VmHWM | Proof |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| default (no CLMUL) | Fast | 1,170 (8,190) | 89.3 s | 1,253 ms | 1,050 MiB | 1,347 MiB | 320,591 B (313.1 KiB) |
| `avx512` + `pclmulqdq` | Fast | 1,170 (8,190) | 0.92 s (8,892 perm/s) | 12.7 ms | 1,002 MiB (125 KiB/perm) | 1,381 MiB | 320,591 B (313.1 KiB) |
| `avx512` + `pclmulqdq` | Secure | 1,170 (8,190) | 0.93 s | 13.4 ms | 1,034 MiB | 1,251 MiB | 577,455 B (563.9 KiB) |
| `avx512` + `pclmulqdq` | Fast | 256 (1,792) | 0.53 s | 14.9 ms | 612 MiB | 1,021 MiB | 316,687 B |
| `avx512` + `pclmulqdq` | Fast | 64 (448) | 0.30 s | 11.4 ms | 545 MiB | 811 MiB | 288,439 B |

Witness generation and packing take 0.05 s at 1,170 records.

The rows above do not include the commit-then-prove split (K7), which adds one Ligerito
commit per sub-proof. With the split (same machine and threads, 3 repetitions at 1,170
records, Fast, not a CPU-2 benchmark): `avx512` + `pclmulqdq` build, commit 0.046 s, prove
including witness and
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
  (the default build must not rely on SIMD).
- Proof size is almost flat in `n` (313 KiB at 8,190 permutations, Fast); the spec's
  estimate of about 500 KiB per 2^16-permutation segment is consistent with it.
- Full segments (2^13 records, 57,344 permutations, 2.3e9 committed bits) are for
  the cloud machine; extrapolation from the table, not measured: about 7 s with CLMUL
  and 7 GB peak.

## 9. The tied part (docs/bridge.md Sections 5a, 5c, 5d; deviations K11, K14, K16, K21)

Modules `flock_union` and `tied`. A part of a segment (or a span of segments, deviation K16)
is proven by one Flock instance, the sponge instance, plus the virtual parity chain on its
tie challenger (tied-part proof version 5, deviation K21, Section 12.7,
`docs/parity-chain.md`): the part's bits are tied to the public final vectors by evaluation
claims, not by Flock's word copies.

- `flock_union::UnionGeometry` (a registry of one or more boolean types in the caller's order,
  declared counts, PCS parameters derived as in Section 4; `new` builds one of a single type,
  `slot(i)` maps a caller type to Flock's slot, which Flock sorts by `k_log`):
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
- `tied::TiedPart::new_part(gate, n, d, index, profile)` and `new_span` (several segments,
  K16): the sponge instance (`n` rows of the sponge gate, `nu_s = capacity_log(n, 19)`) and
  the part's parity chain (`agg_bridge::parity_chain`). `honest_parts(records)` returns the
  sponge slot buffers and the part's lane-major input bits (`6 x 2^(10 + d)` bits per segment;
  the mode-A hooks tamper this vector). `commit(records)` / `commit_parts(sponge, bits,
  records)` (event 2: the sponge cap as `TiedCommitment`, the commit data kept in
  `TiedCommitted`, `take_commit_data()` to keep it while the buffers are dropped),
  `reattach(sponge, bits, records, commitment, data)`, `final_vectors` and
  `final_vectors_of_bits` (one share per segment, computed at event 6 from the kept input bits,
  deviation K22), `prove` and `prove_with_final_vectors(committed, sd, matrices, ys, fs)`
  (after event 6, absorbed at the join after stage 3: the tie, the chain, then the sponge
  instance with its public claim and the bit claim), and `verify(mu, c~, commitment, sd,
  matrices, ys, fs, proof)`: Flock's verification, the public value against its table of
  `(mu_i, c~_i, 1, 0)`, and the chain's rounds and last claim against the verifier's own
  weight. Proof bytes (version 5): `docs/parity-chain.md` Section 7. The sponge challenger
  absorbs `fs`, its label `agg-keccak/v5/tied-sponge`, the gate digest, a digest of the
  statement words, a digest of the tie geometry and the 32-byte tie binding (label
  `agg-keccak/v4/tie-binding`), then Flock's statement binding.

Memory: the sponge commit data of every part is kept from event 2 until its proof (K14), and
a batch of exactly one tied part keeps its whole committed witness until the join (lever W1,
`docs/perf.md` Section 9); the proof is the same as with a rebuilt witness.

Tests: `flock_union` (the eq-table index order and `eval_packed` against the plain inner
product; proof byte framing); `tied` unit tests and `tests/all/tied.rs` (docs/bridge.md Sections
5c, 5d, `docs/parity-chain.md` Section 11), including a segment in parts whose shares XOR to
the segment's `Y`.

## 12. Flock bytes: `Slim` in mode A, spans, the level-0 cap, `Slim8`

Goal: the fewest Flock bytes per signature at Flock's documented 128-bit level (Ligerito in
the Johnson regime with two-point out-of-domain binding, every component of
`LigeritoSecurityConfig::validate` strictly above 128 bits, the query term work-normalized
with its proof of work), at a stated prover and verifier cost. Deviations K15 to K18.

### 12.1 Where the bytes are

A tied part's proof is a header, the parity chain's rounds and the sponge instance's proof
(12.7); the instance proof is a zerocheck (about 3 KB), a lincheck (1.7 KB), the
claim values and Flock's merged opening, which is a ring-switch and Frobenius layer (13 to 15
KB) around one Ligerito proof. The Ligerito proof is almost everything: per level, `Q`
queried rows of the interleaved codeword (`lanes x 16` bytes at level 0, `2^4 x 16` above) and
their Merkle paths (`log2(codeword) - cap` siblings of 32 bytes), plus the caps, the final
block and the sum-check messages. Breakdown (`tests/all/tied.rs`, `AGG_TIED_BREAKDOWN=1`; bytes are
deterministic):

| Instance (records, `m`) | Profile | Level-0 rows | Level-0 paths | Levels 1+ rows / paths | Ligerito | Instance proof |
| --- | --- | --- | --- | --- | ---: | ---: |
| sponge (2,048, `m` 30) | Fast | 244 x 34 lanes: 129.6 KB | 83.9 KB (11 siblings) | 47.8 / 63.7 KB | 340,912 B | 361,556 B |
| sponge (2,048, `m` 30) | Slim | 119 x 34: 63.2 KB | 48.3 KB (13) | 40.0 / 60.1 KB | 223,856 B | 243,252 B |

(The Ligerito column is the proof's `size_bytes` without the level-0 cap, which the
serialized proof no longer carries, K17.) So the bytes are set by the level-0 query count (rate and the 16-bit
query PoW), the level-0 row width (`lanes = ceil(words / 2^(m - 7 - initial_k))`), the path
depths, and by the NUMBER of openings: one per tied part, whatever its size.

### 12.2 What was implemented

1. **`Slim` in mode A** (K15): `FlockProfile::Slim = 4`, bound at event 0, named in the
   event-2 item (item version 4), `--flock-profile`. Default of mode A is `Fast`, the
   measured best at 32K; `Slim` is the smaller-profile option.
2. **Spans** (K16, docs/bridge.md Section 5c): a batch of `S >= 2` segments is ONE tied part
   (one sponge instance of all records, one parity chain over all segments), so it pays one
   opening instead of `S`. `--flock-span on|off`, default on.
3. **The level-0 cap is not sent twice** (K17): 4,096 B (`Fast`) or 2,048 B (`Slim`) per
   instance.
4. **`Slim8`** (K18): a rate-1/8 Ligerito parameter set derived by Flock's own
   `derive_profile` and checked by its `validate()` (79 level-0 queries instead of 119),
   in the vendored core; `--flock-profile slim8`. Not the default: it is not a profile Flock
   ships (a project decision, below).

### 12.3 Measurements

(Tied-part proof versions 3 and 4, with the bridge union. Version 5 removed the union: 12.7.)

Machine: i7-11700K, 7 threads (`taskset -c 0,1,3-15`, `RAYON_NUM_THREADS=7`), CPU 2 not used,
`crypto_bench off` not run, shared with other jobs. CLMUL build
(`RUSTFLAGS="-C target-feature=+pclmulqdq"`, `--features agg-keccak/avx512,agg-piop/avx512`,
`target/clmul`), with spans, K17 and `Slim8`. Medians of 5 runs after 1 warmup; RSS is
`VmHWM`, the maximum over all runs.

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

Against the code without spans and K17 (Fast): 717,641 B at 1,024 records (cloud and
local), 829,736 B at 2,048; the `Fast` rows are 8,246 and 8,197 B smaller: 8,189 B in the
Flock items (the two level-0 caps, K17, less three new header bytes), the rest the Akita
opening, whose size varies by a few bytes with the transcript. Kept commit data (K14): 84 / 168 / 336 MiB at
1,024 and 152 / 304 / 608 MiB at 2,048 (Fast / Slim / Slim8).

Tied parts alone (`tests/all/tied.rs`, `tied_part_profiles_bytes_and_times`, `AGG_TIED_REPS=5`,
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
post-K17 it predicts 775,046 B against ~766.8 KB).

- **(1) `Slim`**: measured above, 36% fewer Flock bytes than `Fast` at 2,048 records (30% fewer
  aggregate bytes) for 3% more prover time (119 ms: commit time 1.23x in mode A, 1.36x in the
  standalone tied part), the same verifier time and twice the kept commit data. Adopted as the default.
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
- **(5) LaBinius**: not adopted. Its small proofs need LaBRADOR recursion, which costs
  about 1.5 times the prover time and about 0.4 to 0.7 s of verifier time.

### 12.5 Flock bytes per signature at `N = 2^15` (estimates from the model, not measured)

| Option | Per-segment parts (4 x 8,192) | One span |
| --- | ---: | ---: |
| Fast (pre-K17 cloud run 94.6, post-K17 ~93.6) | 94.6 (pre-K17) | 22.7 |
| Slim | 62.3 | 16.0 |
| Slim8 | 50.2 | 13.3 |
| Slim with 24-bit query PoW | 58.4 | 15.0 |

With `Slim` and spans the Flock side would be about 16 bytes per signature at 32K
instead of 94.6 (pre-K17 cloud run), the aggregate about 51 instead of 129.4 bytes per signature
(the other components as measured on the cloud). Prover time and memory: in a local sample at
`S = 2` (`count = 8193`, 7 threads), the span path took ~7% more prove time (Flock parts 9,513
ms vs 8,881 ms) and ~15% more peak memory (+1.2 GiB, 9,325 MiB vs 8,093 MiB) than
`--flock-span off`, while cutting proof bytes by 37% (513 KB vs 821 KB) and halving verify time
(87 ms vs 175 ms); at 32K the exact prove time and memory remain to be measured on the cloud.
Verifier time: one sponge-instance check plus one union discharge (about 80 ms locally) instead
of four parts (4 x 69 ms on the cloud).

### 12.6 Decisions and open points

- Default `Fast`, span on (`Fast` is the measured best at 32K; `Slim` saves the bytes above).
  `Slim8` is ready behind `--flock-profile slim8` (another 17% at
  32K, estimate); making it the default is a project decision: it is a parameter set derived
  and validated with Flock's analysis but not one Flock ships, so its proofs verify only with
  the vendored core (or upstream given the same configs).
- The span path of mode A runs only above `2^13` records; it is tested at the tied-part level
  and on the item level.

### 12.7 Tied-part proof version 5: the virtual parity chain

The bridge union is gone (deviation K21, `docs/parity-chain.md`, `docs/bridge.md` 5d): a tied
part is the sponge instance plus a sum-check on its tie challenger, whose final claim the
sponge's opening proves as a third ring-switched claim. The table rows of 12.3 that split a
proof into "cap / sponge / union" therefore become "header and chain / sponge proof".

Machine and build as in 12.3 (i7-11700K, CLMUL build `RUSTFLAGS="-C
target-feature=+pclmulqdq"` in `target/clmul`, 7 threads `taskset -c 0,1,3-15`,
`RAYON_NUM_THREADS=7`, CPU 2 unused, `crypto_bench off` not run), shared with other agents'
jobs (load average 6 to 18 during the runs, so times are direction only). "v4" is the code
with the bridge union, "v5" the code with the parity chain; both binaries ran interleaved.
Bytes are
deterministic; times are medians of 5 runs after 1 warmup (`8192`: of 3), each pair run back
to back.

Tied part alone (`tests/all/tied.rs`, `tied_part_profiles_bytes_and_times`, `Slim`):

| Part | Proof v4 (cap / sponge / union) | Proof v5 (header and chain / sponge) | Prove v4 / v5 | Verify v4 / v5 |
| --- | --- | --- | --- | --- |
| `8@13/4` | 350,766 B (2,048 / 145,636 / 203,052) | 150,758 B (1,046 / 149,712) | 1,412.0 / 153.4 ms; 1,217.9 / 150.0 ms; 2,002.5 / 196.4 ms | 98.0 / 26.5; 97.9 / 24.6; 113.6 / 32.3 ms |
| 2,048 (`d` 11) | 438,854 B (2,048 / 243,252 / 193,524) | 248,230 B (902 / 247,328) | 863.0 / 802.9 ms; 804.6 / 752.5 ms; 1,340.3 / 1,034.2 ms | 79.6 / 28.4; 82.1 / 29.7; 90.4 / 39.9 ms |
| 8,192 (`d` 13) | 504,358 B (2,048 / 277,604 / 224,676) | 282,654 B (974 / 281,680) | 1,897.6 / 1,959.8 ms; 3,400.2 / 3,295.9 ms | 109.4 / 33.8; 138.8 / 44.0 ms |

Bytes: at `8@13/4` (the GF(2) side of a 32K batch) the proof shrinks by 200,008 B: the
union's 205,108 B (its proof, cap and framing) go and 5,100 B come (the chain's 1,028 B and,
in the sponge opening, the bit claim's `s_hat_v` and nonce, 2,056 B, and its 128 multipoint
dual values, 2,048 B, less the two lane-group values, 32 B).

Prover: the chain itself is cheap (the timing test `parity_chain::tests::
chain_timing_at_production_shapes` on dense random bits: 32.8 / 59.8 / 178.5 ms at `d` 11 /
13 / the 32K span `13/2`; the verifier's weight and claim 7 ms), but the third ring-switched
claim is not: at a full-height sponge (every power-of-two batch from 2,048 records on) Flock's
block-first opening needs the claim's full column bit-bank (all 2,146 word columns, Flock's
`union_bitbank_fold`, measured 174 to 222 ms at 8,192 records under load), and the opening's
block statistics and `W'` fold grow by one claim (about +60 ms at 8,192). At 8,192 records
these costs about cancel the full-segment union (v5 within the noise of v4); at `8@13/4` and
2,048 the union dominates and v5 is faster. At 32K (estimate): the union of four
full-segment geometries cost 1.2 to 2.0 s locally (`8@13/4`), the v5 additions about 0.18 s
(chain) + 4 x 0.2 s (bank, scaling with the stack) + 0.25 s (opening) = 1.23 s, so the local
net is +0.03 to -0.77 s of prover work, about 0.0 to -0.8 s. v4 proved the union after the
sponge instance (the union took about 1.14 s of a 32K cloud run, measured), so the
critical-path effect equals the prover-work estimate; v5's chain runs strictly before the
sponge instance, whose opening holds the bank. The bank's fold scaled poorly with threads
locally (`docs/perf.md` F5), so the cloud may not shrink it.

Mode A end to end (`agg bench --keys 16 --warmup 1 --reps 3 --flock-profile slim`):

| N | Aggregate v4 -> v5 (B/sig) | Flock proofs v4 -> v5 | Prove v4 / v5 (Flock parts) | Verify uncached / cached v4 -> v5 | Flock part verify |
| ---: | --- | --- | --- | --- | --- |
| 1,024 | 522,155 (509.9) -> 339,262 (331.3) | 398,518 -> 215,610 B | 2,308.1 (1,029.1) / 2,280.1 (1,019.8) ms | 181.3 / 193.2 -> 142.7 / 327.7 ms | 78.3 -> 36.4 ms |
| 2,048 | 598,031 (292.0) -> 407,457 (199.0) | 438,862 -> 248,238 B | 3,350.9 (1,209.3) / 2,839.9 (1,114.2) ms | 258.0 / 201.1 -> 147.5 / 146.4 ms | 79.2 -> 29.9 ms |

(The cached verify of v5 at 1,024 is an outlier of the shared machine; the prover's and
verifier's peak RSS, `VmHWM` maxima over the runs of one process, were 1,780 / 891 -> 2,109 /
978 MiB at 1,024 and 1,791 / 1,087 -> 2,094 / 1,612 MiB at 2,048; the per-phase trace of the
tied part shows the same peak, 1,257 vs 1,259 MiB at 2,048; see below.) CLI round trips (`agg aggregate` then `agg verify`, `keygen-batch --count 64`) at 256
and 1,024 records accept under H-u and H-exact (300,425 / 300,424 B and 339,252 / 339,258 B);
each aggregate verified under the other relation, and one with a flipped middle byte, is
rejected.

Peak RSS in a fresh process (`/usr/bin/time -v`, `agg aggregate` and `agg verify` of the
same records, v4 and v5 CLMUL builds run interleaved, 2 runs each, load 0.9 to 4.5, measured;
times direction only):

| N | Aggregate max RSS v4 -> v5 | Same, `MALLOC_MMAP_THRESHOLD_=1048576` | Verify max RSS v4 -> v5 | Aggregate wall v4 -> v5 |
| ---: | --- | --- | --- | --- |
| 2,048 | 1,882 / 1,880 -> 2,011 / 2,007 MiB | - | 126 / 125 -> 117 / 116 MiB | 3.70 / 3.47 -> 3.29 / 3.27 s |
| 8,192 | 6,088 / 6,077 -> 6,542 / 6,484 MiB | 5,725 / 5,710 -> 5,764 / 5,761 MiB | 196 / 196 -> 143 / 144 MiB | 7.86 / 7.89 -> 7.84 / 8.02 s |

The verifier's peak goes down (the in-process `VmHWM` of `agg bench` above is an artefact of
the prover's memory still held by the allocator). The prover's peak rises by about 7% with
glibc's default allocator but only by 39 to 51 MiB (under 1%) with a fixed 1 MiB mmap threshold,
so most of the rise is heap retention, not live data: a VmRSS trace sampled every 50 ms puts
both peaks in the lookup after the Flock part (v4 6,019 MiB at 6.3 s, v5 6,540 MiB at 5.5
s), while the tied part's own peak is lower in v5 (5,812 vs 5,878 MiB at the end of its
proof, `AGG_TIED_TRACE`). glibc raises its mmap threshold dynamically (up to 32 MiB) after
large frees, so the different allocation pattern of v5 leaves more freed heap resident when
the lookup allocates. Whether this carries to 32K (25.8 GiB baseline) is for the cloud run;
a fixed mmap threshold (or `mallopt` at start-up) also lowered v4's own peak by about 370 MiB
at 8,192 and is a prover-only memory lever to measure there (its time cost is unmeasured).
