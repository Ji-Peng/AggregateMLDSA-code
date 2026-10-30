# agg-keccak: the Keccak argument (spec Section 8)

`crates/agg-keccak` proves, for a batch of signatures, the witness-dependent hash of
ML-DSA verification, `c~_i = SHAKE256(mu_i || w1Encode(w1'_i), 32)`, with Flock HEAD
(`succinctlabs/flock` `83f963e`, vendored `flock-core`, K12) and its Ligerito PCS. Deviations
from the spec are listed in `docs/deviations.md` (the K entries). Sections 1 to 6 describe
the sponge gate and the standalone sponge proof, Section 9 the tied part that mode A uses,
Section 12 the Flock bytes: the `Slim` profile in mode A, spans over all segments, the
level-0 cap, the options measured or estimated, and (12.7) tied-part proof version 5, the
virtual parity chain.

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
bits. All rows have degree 2.

Lincheck. Substituting the implicit states makes `A_0` and `B_0` dense (table below), so
they are never materialized. `ChainGate` implements Flock's `LincheckCircuit` by one
backward walk: with `K` the combined row weight `alpha * eq^T A + eq^T B` viewed as a
functional on the current state,

```text
comb[t_r] += K;  const += K . RC_r;  G = K + W_r (chi rows: alpha*e on B_r[x+1], e on B_r[x+2]);  K = phi^T(G)
```

round by round and permutation by permutation (at a boundary `K` lands on the absorbed
input columns and constants and carries on to the previous output). Cost per fold:
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

See [perf.md](perf.md), Section 19. Build instructions are in [README](../README.md).

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
serialized proof omits, K17.) So the bytes are set by the level-0 query count (rate and the 16-bit
query PoW), the level-0 row width (`lanes = ceil(words / 2^(m - 7 - initial_k))`), the path
depths, and by the NUMBER of openings: one per tied part, whatever its size.

### 12.2 Profiles and spans

1. **`Slim` in mode A** (K15): `FlockProfile::Slim = 4`, bound at event 0, named in the
   event-2 item (item version 4), `--flock-profile`. Default of mode A.
2. **Spans** (K16, docs/bridge.md Section 5c): a batch of `S >= 2` segments is ONE tied part
   (one sponge instance of all records, one parity chain over all segments), so it pays one
   opening instead of `S`. `--flock-span on|off`, default on.
3. **The level-0 cap is not sent twice** (K17): 4,096 B (`Fast`) or 2,048 B (`Slim`) per
   instance.
4. **`Slim8`** (K18): a rate-1/8 Ligerito parameter set derived by Flock's own
   `derive_profile` and checked by its `validate()` (79 level-0 queries instead of 119),
   in the vendored core; `--flock-profile slim8`. Not the default: it is not a profile Flock
   ships (a project decision, below).

### 12.6 Decisions and open points

- Default `Slim`, span on. `Slim8` is ready behind `--flock-profile slim8` (another 17% at
  32K, estimate); making it the default is a project decision: it is a parameter set derived
  and validated with Flock's analysis but not one Flock ships, so its proofs verify only with
  the vendored core (or upstream given the same configs).
- The span path of mode A runs only above `2^13` records; it is tested at the tied-part level
  and on the item level.

### 12.7 Tied-part proof version 5: the virtual parity chain

A tied part is the sponge instance plus a sum-check on its tie challenger. The
sponge opening proves the final claim as a third ring-switched claim
([parity-chain.md](parity-chain.md), [bridge.md](bridge.md), Section 5d).

A full-height sponge (every power-of-two batch from 2,048 records) uses Flock's
block-first opening. The bit claim requires the full column bit-bank, all 2,156 word
columns. A partial bank suffices for `s_hat_v` but does not prove the block rounds.
The opening's block statistics and `W'` fold include this claim.

Measurements are in [perf.md](perf.md), Section 18.
