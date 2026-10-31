# Mode A end to end and the `agg` tool (spec Sections 10, 11, 14)

Crate `agg-cli`: library `agg_cli` (aggregator, verifier, wire items, key cache, batch
files) and binary `agg`. Deviations: K10 (Flock parts within a segment), K15, K18 (the Flock
profiles `Fast`, `Slim` and `Slim8`), K16 (one span over all segments), K17 and A1 (driver choices) in
`docs/deviations.md`.

## 1. Aggregator and verifier

`agg_cli::mode_a::Aggregator::aggregate(&[(pk, M, signature)])` returns an `Aggregate`:
the wire bytes of `agg_piop::wire::AggregateProof` (header, the `c~_i` list, then the proof
blob: the seven Akita commitments, the mode-A segment items, `Y_sigma`, the PIOP rounds and
claims, the Akita opening) and a `ProveReport` (step times, per Flock part times and sizes,
the byte size of every part of the aggregate). Steps:

1. Lengths checked; `c~_i` is the first 32 signature bytes; the public instance
   (`ExpandA`, `tr`, `mu`, `SampleInBall`, `T`) is built.
2. Witnesses derived with strict FIPS 204 verification (an invalid signature is refused with
   its index) and the round-1 tables `G_main16` (`i32` cells) and `G_main4` (`i8` cells) built.
3. The PIOP prover `agg_piop::prover::prove_with_tables` runs the one transcript of spec
   Section 10 with the seed-derived matrices (`SeededMatrices`) and the Flock side
   `FlockProver` (Flock profile `AggregatorConfig::profile`: `Fast` by default, `Slim` or
   `Slim8`; bound at event 0, K15, K18): event 1 `G_main16, G_main4`, event 2 the parts' Flock
   commitments, event 3 `sd` and the projection matrix `Psi` (K24; the site grinds one bit),
   event 4 `G_1, G_2, G_3, G_k, G_m`, event 5 `fs_sigma`, event 6 `Y_sigma`
   and the parts' shares, events 7 to 12 the lookup and the stages, the join the parts'
   proofs (deviation K22: proven on their own thread pool, `AGG_FLOCK_THREADS` threads,
   default 3/8 of the global pool rounded up, `0` for the serial order, at most the logical CPUs or the
   global pool, a non-numeric value warned about on stderr and replaced by the default,
   while the prover runs events 7 to 12), event 13 the opening. There is no divisibility projection seed
   (deviation P2); the `{0,1}` projection of `v` (K24) bounds the quotient.

The stages (branch S of the transcript fork) run on their own pool when
`AGG_PIOP_S_THREADS` is set, and by default while the Flock fork runs and the PIOP pool has
four threads or more: half at 32 threads or more, 3/8 of the pool rounded to nearest and at
least 2 below. The defaults are 16 threads on a 32-thread pool and 6 on a 16-thread pool; there
the three pools add up to 60 and 28 threads. `AGG_PIOP_S_THREADS=0` keeps the shared pool.

`agg_cli::mode_a::AggregateVerifier::verify(&[(pk, M)], aggregate, cache)` returns a
`VerifyCostReport`: accept or the first failing check, the 15 step times of
`agg_piop::verifier::VerifyTimings`, per Flock part the setup and verification time, and the
keys expanded in the run. The verifier computes `tr_i, mu_i, c_i, T_i` and the `ExpandA`
data itself; `cache: Option<&mut KeyCache>` skips the expansion of cached keys and adds
new ones, so the uncached and cached numbers of spec Section 11 are two calls. The Flock
profile is the mode-A profile the aggregate's event-2 items name (a parse of the aggregate,
step `profile` of the report), or the one fixed by `AggregateVerifier::require`; the
transcript binds it at event 0 (K15).

Flock parts (K10, K11): segment `sigma` is split into parts of `2^{d_part}` record positions,
`d_part = clamp(part_log2, max(3, d_seg - 1), d_seg)` (at most two parts per segment,
`docs/soundness.md` 3.5), `part_log2` from `AggregatorConfig` (default
`DEFAULT_PART_LOG2 = 13`: the whole segment is one part). Each part
is a tied part (`agg_keccak::tied`, proof version 5): a Flock sponge instance of its records
and the virtual parity chain of its share (`docs/parity-chain.md`, `docs/keccak.md` 12.7): the sponge cap at
event 2, the share at event 6 (computed from the part's input bits, kept since event 2), and
at the join the chain's rounds and the sponge proof, whose opening carries the chain's bit
claim; the part and span seeds bind the digest of the event-6 record (K22). At most one
part's Flock buffers are alive: parts are committed one by one at event 2 (the cap and the
sponge commit data are kept, K14) and rebuilt, proven and dropped one by one for the join.
A batch of exactly one part (a span, or one part at `N <= 2^13`) keeps its committed witness
from event 2 instead and proves it without a rebuild (lever W1, `docs/perf.md` Section 9).

Span (K16, `AggregatorConfig::span`, default on): a batch of more than one segment whose
parts are whole segments is one tied part over all segments (one sponge instance of all `N`
records, one parity chain over every `Y_sigma`); segment 0's items carry its cap and proof,
the others are covered (`agg_cli::wire`, span byte 0), and the verifier checks the span at the
last segment. Batches of at most `2^13` records (every local batch) have one segment and are
unchanged by it.

Test hooks (`#[doc(hidden)]`, `AggregateHooks`): tampered round-1 or round-2 tables, the
Flock side's records taken from the tampered tables, tampered Flock records, a custom
Flock witness per part, and stale round-2 tables (the honest bits' parities and `Y_sigma`
over tampered round-1 bits).

## 2. Files and the `agg` binary

| File | Format (`agg_cli::record`, `agg_cli::cache`) |
| --- | --- |
| keys | `"AGGKEYS1" \|\| count LE32 \|\| (seed[32] \|\| LE32 len \|\| pk)*` (test keys: holds the `KeyGen` seeds) |
| records | `"AGGRECS1" \|\| count LE32 \|\| (LE32 len \|\| pk, LE32 len \|\| M, LE32 len \|\| signature)*` |
| aggregate | the wire bytes of `agg_piop::wire` (mode-A items of `agg_cli::wire`) |
| key cache | `"AGGKCACH" \|\| version \|\| count LE32 \|\| (SHA3-256(pk) \|\| pk \|\| tr \|\| A as LE32)*`, entries checked on load |

```sh
A=target/release/agg
taskset -c 0,1,3-15 $A keygen-batch --count 16 --seed 1 --out keys.bin
taskset -c 0,1,3-15 $A sign-batch --keys keys.bin --count 256 --seed 1 --out records.bin
taskset -c 0,1,3-15 $A aggregate --records records.bin --out aggregate.bin [--part-log2 9] [--flock-profile fast|slim|slim8] [--flock-span on|off] [--relation h-u|h-exact|h-fips]
taskset -c 0,1,3-15 $A verify --records records.bin --aggregate aggregate.bin [--cache keys.cache] [--flock-profile fast|slim|slim8] [--relation h-u|h-exact|h-fips]
AGG_GIT_REV=$(git rev-parse HEAD) taskset -c 0,1,3-15 $A bench --count 256 [--keys K] [--warmup 1] [--reps 5] [--part-log2 9] [--flock-profile fast|slim|slim8] [--flock-span on|off] [--relation h-u|h-exact|h-fips]
```

Environment knobs (all optional; none changes the proof bytes):

| Variable | Effect |
| --- | --- |
| `AGG_FLOCK_THREADS` | threads of the Flock pool beside the PIOP; default `ceil(3/8)` of the rayon pool, `0` serial |
| `AGG_PIOP_S_THREADS` | threads of the stage branch's own pool; default as above, `0` shares the rayon pool |
| `AGG_PIOP_THREADS` | threads of the PIOP pool beside the Flock fork; PIOP + Flock capped at the logical CPUs |
| `AGG_PIOP_FORK` | `0`: run the two transcript branches one after the other |
| `AGG_OPENING_THREADS` | threads of the Akita opening pool; default one per physical core |
| `AGG_HUGEPAGE` | `0`: turn the huge-page advice off |
| `AGG_ALLOC_CACHE` | cap the cache of freed large blocks in GiB (default 6, `0` disables it) |
| `AGG_ALLOC_POISON` | `1`: fill reused blocks with `0xA5` (test mode for uninitialised reads) |
| `AGG_W1_SPLIT` | `0`: build the level-1 weight table instead of its two factors |
| `AGG_PHASE_STAMPS` | any value: print a `phase-stamp` line per prover step on stderr |

The vendored Flock core's remaining `FLOCK_*` knobs are described in
`vendor/flock-core/PATCH.md` and `docs/perf.md`.

`--relation` selects the relation (`docs/h-exact.md`, spec Section 3.1 and Appendix A):
`h-u` (default, relation byte 0: `z in [-2^17, 2^17)`, `|u| <= 190,465`) or `h-exact`
(relation byte 1: the FIPS 204 bounds `|z| <= 130,993`, `|u| <= 190,465`, one more
committed column). `aggregate` and `bench` prove the chosen relation
(`AggregatorConfig::relation`); `verify` verifies under the chosen relation
(`AggregateVerifier::relation`) and rejects an aggregate whose header names another one
("the proof claims relation H-exact, the verifier expects H-u"), so verifying an H-exact
aggregate needs `--relation h-exact`. The prover and verifier reports and the `bench`
header print the relation. The Keccak side does not depend on it.

`--flock-profile` selects the Ligerito profile of the Flock parts (`fast`: rate 1/2, `slim`:
rate 1/4, `slim8`: rate 1/8, the derived parameter set of K18; all 128 bits work-normalized in
Flock's accounting, with the level-0 list priced at the outer sites (K19) and at the tie
(K20); default `fast`, `agg_cli::mode_a::FLOCK_PROFILE`, `docs/keccak.md` Section
12); `verify` reads the profile from the aggregate unless `--flock-profile` pins one. A profile
outside the default set (`fast`, `slim`) is rejected with a message naming
`--flock-profile slim8`, not verified under another profile. `--flock-span off` proves every
segment on its own (K10 parts) instead of one span over all segments (K16).

`verify` reads only `(pk, M)` from the record file and exits with status 0 on accept, 1 on
reject, 2 on a usage or file error. `bench` generates the records in memory and follows
the benchmark protocol: `--warmup W` runs (default 1, discarded), then `--reps R` timed runs
(default 5), for each of prove, verify uncached and verify with every key cached (after one
extra run that fills the cache); all three verify the aggregate of a timed prover run. It
prints the step report of the median prover run (with its Flock-parts / everything-else
split) and of the last verifier runs, the median and the min prove time, the verify
medians, all marked "(median of R, W warmup)", the prover's and the verifier's peak RSS
over all runs (`VmHWM`, reset in
between through `/proc/self/clear_refs`, which affects only the process's own accounting)
and the aggregate bytes per signature with the size of every part. Benchmarks proper run on
CPU 2 after `crypto_bench off`; the numbers below are not benchmarks.

## 3. Tests

`taskset -c 0,1,3-15 cargo test --release -p agg-cli` (unit tests and `tests/all/{mode_a,cli}.rs`):

- unit: part tiling of every segment (also with an all-padded trailing segment), part seeds
  and span seeds, an empty segment (no part, `Y_sigma = 0`, malformed items rejected), the
  span glue of the verifier on a four-segment geometry without proofs (segment 0 opens the
  span, segments 1 to 3 are covered, the last one verifies it; spans over fewer segments,
  covered items with commitments or proofs, per-segment or second span items inside a span
  rejected), the segment items (profile and span bytes), the options `--flock-profile`,
  `--flock-span` and `--relation`, the Flock records of honest tables equal `w1Encode`, the batch files, the
  key cache file;
- round trips at 64 (one part), 256 (two parts of 128) and 512 (one part) signatures with
  own and shared keys under each profile (`Fast`, `Slim`, `Slim8`): the `c~_i` list, the size report,
  the uncached and cached verifier (in memory and through the cache file) agree, the
  verifier reads the profile from the aggregate, and a verifier pinned to the other profile
  rejects the item's profile byte before any Flock work;
- relation H-exact: round trips at 64 (one part) and 256 (two parts) signatures under
  `Slim` with `AggregatorConfig::relation = HExact`; a verifier of the other relation
  rejects the aggregate before event 0 (`VerifyError::Relation`); the binary aggregates and
  verifies with `--relation h-exact` and rejects the cross-relation cases;
- corrupted aggregates (20 records, parts of 8, 8, 4): `c~_i`, `Y_sigma`, a share, swapped
  shares, swapped part proofs, a part-proof byte, a part-commitment byte, the part size
  (too large, too small), a dropped part, a truncated item, a pre-stage claim, stage-1, -2
  and -3 rounds, a stage-3 claim, the opening, the `G_main4` commitment, reordered round-2
  commitments, swapped records, another message, another key, a missing record, truncated
  and trailing bytes, the wire version, the item's profile byte (another mode-A profile; the
  mode-B id 3), a Flock verifier under another profile, the mode-B checks under profile 3;
- tampered witnesses through the aggregator (12 records): a `Z_lo` and a `V_lo` cell changed
  inside `T_16` (stage 3), a committed bit flipped (Flock part) and the same with stale
  round-2 tables (stage 1), a bit of value 2 (stage 1), `w_1' = 44` in the committed bits
  (Flock part) and in both copies (Flock part; stage 1 with the Flock verdict ignored), a
  stored `U_lo = 2^15` outside `T_16` (the Akita commitment refuses it), base-4 and base-6
  digits outside their alphabets (stage 2), a level-1 carry and a level-2 parity (stage 1),
  a final-vector bit (Flock part), and on the Flock side a bit
  flipped in both copies, in the parity chain's copy only, and in the sponge input with the stale
  chi witness (Flock part);
- invalid signatures refused by the aggregator: `c~`, `z` and hint bits, another message,
  wrong lengths;
- the binary: `keygen-batch`, `sign-batch`, `aggregate`, `verify` (uncached, with a new and
  a filled cache file, on other messages), usage errors, `bench`.

## 4. Local measurements

Relation sanity runs (revision `f526097`, CLMUL build `RUSTFLAGS="-C
target-feature=+pclmulqdq"`, `agg bench --count 1024 --warmup 1 --reps 1 --flock-profile slim`,
span on, CPUs 0,1,3-15 with `RAYON_NUM_THREADS=3` while a catalog generation ran with 3
workers: single noisy runs, not benchmarks):

| Relation | Aggregate | PIOP rounds and claims | Work nonces | Akita opening | Prove | Stage 2 / stage 3 | Verify uncached / cached |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H-u | 522,138 B (509.9 B/sig) | 4,736 B | 391 B | 60,994 B | 2,901 ms | 67.5 / 181.1 ms | 352 / 296 ms |
| H-exact | 522,154 B (509.9 B/sig) | 4,736 B | 391 B | 61,010 B | 2,981 ms | 60.9 / 198.0 ms | 316 / 290 ms |

Every other part is equal (lookup 21,344 B, Flock proofs 398,518 B). The PIOP bytes agree
with `16 (81 + 11 d_main)` at `d_main = 20` for both relations (9 + 1 + 34 claims).
