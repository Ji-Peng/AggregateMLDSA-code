# PCS configuration (`agg-pcs-config`)

Akita commitment preset, generated fold-schedule catalog, work plan and two-round
commit driver for the four witness groups of spec Section 4.1. Akita and Aerie pins
as in `README.md` (Akita `e023b6a`, the revision Aerie `49b6c21` pins, with the
`dev-protocol` feature the catalogs are priced for).

## 1. Preset

Every committed table uses one preset, `preset::AggFp64SignedWidth4` (signed width 4,
values in `[-8, 7]`). Its Akita parameters are Aerie's `Fp64SignedWidth4`, checked
field by field in `preset::tests::preset_matches_aerie_width4_parameters`; it is a
separate type only because the catalog is keyed by the configuration type and family
name (`docs/deviations.md` D1).

| Parameter | Value |
| --- | --- |
| Field / challenges | `Prime64Offset59` (`p = 2^64 - 59`) / `Ext2` |
| SIS modulus profile | `Q64Offset59` |
| Ring dimensions, challenge policy | fp64 `Dense` (adaptive A/B/D search) |
| `DecompositionParams` | `log_basis 3`, `log_commit_bound 4`, `log_open_bound Some(64)` |
| Opening (B/D) basis search | log basis 3..=6 (22 opening digits at basis 3) |
| Inner (A/source) basis | exactly 4: one balanced base-16 digit per coefficient |
| Source class | `CommittedSourceClass::BalancedSignedDigit` |
| Family name | `agg_mldsa_fp64_signed_width4` |

The width is a producer admission bound, not a range proof (Akita Remark 10.41); all
ranges are proven by the PIOP digits of spec Section 4.3. Every digit alphabet of the
spec (bases 4, 8, 16), the Booleans and the padded digits of zero (`-b/2`, never
below `-8`) lie in `[-8, 7]` (`width4_admits_every_digit_alphabet_and_padding`). The spec
needs no other width: Booleans use the width-4 preset (spec Appendix B.7), and the
feature-flagged layouts of Appendix A decompose every wide value into the same digits.

## 2. Which signed widths Akita admits, and with how many inner digits

`akita-params` decides representability: a raw signed width `w` with balanced base
`2^b` digits needs `num_digits_inner_for_bound(w, b)` digits, and the fp64 `Dense`
policy searches `b` in `3..=min(16, w)` (`PROOF_OPTIMIZED_INNER_LOG_BASIS_MAX = 16`).
`preset::admissible_inner_digits` and `min_inner_digits` evaluate exactly this
(test `signed_width_admissibility_table`):

| Signed width `w` | Fewest balanced inner digits | Example |
| ---: | ---: | --- |
| 1..=16 | 1 | width 4: base 16, 1 digit (or base 8, 2 digits) |
| 17..=31 | 2 | widths 18-20: base `2^11`..`2^16`, 2 digits; base `2^10` needs 3 at width 20 |
| 32..=47 | 3 | two base-`2^16` digits reach only `2^31 - 2^15 - 1` |

Planner probe (`width_admissibility_probe`): a planning-only preset per width, equal to
the width-4 preset except `log_commit_bound = w` and Dense's inner search 3..=16, planned
for one polynomial at 13 variables (2026-10-24, this machine). Every width is
admissible; the cost-optimal DP does not pick the fewest digits but small bases:

| Width | Selected inner log basis | Inner digits | Estimated proof (B) |
| ---: | ---: | ---: | ---: |
| 4 | 4 | 1 | 37,625 |
| 14 | 5 | 3 | 47,865 |
| 16 | 6 | 3 | 51,961 |
| 17 | 6 | 3 | 51,961 |
| 18 | 5 | 4 | 51,961 |
| 19 | 5 | 4 | 51,961 |
| 20 | 5 | 5 | 56,207 |

So widths 18 to 20 take at least two balanced digits (risks.md R1), and the planner
prefers four or five base-32 digits at this layout. None of these presets is needed by
the spec, so no catalog exists for them.

## 3. Group layouts

Physical layouts `(num_vars, num_polynomials)` in wire and catalog order; `G_3` is the
final group, the others are precommitted. Logical domains below `2^13` cells are
zero-extended with trailing pad variables and opened at the zero-padded point.

| `d_sig` | `N*` | `G_main` | `G_1` | `G_2` | `G_3` (final) |
| ---: | ---: | --- | --- | --- | --- |
| 3 | 8 | (13, 23) | (13, 18) | (13, 24) | (13, 2) |
| 4 | 16 | (14, 23) | (13, 18) | (13, 24) | (13, 2) |
| 5 | 32 | (15, 23) | (13, 18) | (13, 24) | (13, 2) |
| 6 | 64 | (16, 23) | (13, 18) | (13, 24) | (13, 2) |
| 7 | 128 | (17, 23) | (13, 18) | (13, 24) | (13, 2) |
| 8 | 256 | (18, 23) | (13, 18) | (13, 24) | (13, 2) |
| 9 | 512 | (19, 23) | (13, 18) | (13, 24) | (13, 2) |
| 10 | 1,024 | (20, 23) | (14, 18) | (13, 24) | (13, 2) |
| 11 | 2,048 | (21, 23) | (15, 18) | (13, 24) | (13, 2) |
| 12 | 4,096 | (22, 23) | (16, 18) | (13, 24) | (13, 2) |
| 13 | 8,192 | (23, 23) | (17, 18) | (13, 24) | (13, 2) |
| 14 | 16,384 | (24, 23) | (18, 18) | (14, 24) | (13, 2) |
| 15 | 32,768 | (25, 23) | (19, 18) | (15, 24) | (14, 2) |

Polynomial counts: `G_main` 23 (`z` 5, `u` 5, `v` 7, `b` 6), `G_1` 18, `G_2` 24
(6 parities + 18 carry digits), `G_3` 2; 67 in total.

## 4. Catalog status

All thirteen batch sizes of the wire version, `N* = 2^3` to `2^15` (`CATALOG_D_SIG`),
are ready; none is pending. The artifact
`crates/agg-pcs-config/src/schedules/agg_mldsa_fp64_signed_width4.aks` (388,645 bytes)
holds 36 rows: 23 scalar producer rows (`G_main` at 13..=25 variables, `G_1` at
13..=19, `G_2` at 13..=15) and 13 grouped rows, one per batch size, named
`mldsa-44-mode-a-batch-<N*>-level3-final`. Every group of every row uses one balanced
base-16 inner digit, and no row needs a setup prefix (checked by
`checked_in_catalog_holds_exactly_the_requested_rows`).

Planning runs (2026-10-24, `update_catalog_rows`, one process per batch size, cap
20 minutes each, `taskset -c 0,1,3-15`, 3 scalar workers, other agents' builds sharing
the machine). The sizes were planned in the order of the third column; each run planned
only the scalar rows not already in the artifact, then its grouped row (a single DP
search, 18 to 75 s). Wall times are for this machine and indicative only. The proof and
setup figures are the planner's estimates for the grouped row.

| `d_sig` | Status | Run | New scalar rows `(vars, polys)` | Wall time (s) | Est. opening proof (B) | Setup (field elements) |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 3 | ready | 5 | (13,23) | 72.1 | 61,707 | 754,688 |
| 4 | ready | 6 | (14,23) | 85.6 | 61,707 | 754,688 |
| 5 | ready | 7 | (15,23) | 77.8 | 61,707 | 754,688 |
| 6 | ready | 8 | (16,23) | 76.6 | 61,707 | 754,688 |
| 7 | ready | 9 | (17,23) | 81.3 | 61,707 | 754,688 |
| 8 | ready | 1 | (13,18), (13,24), (18,23) | 71.6 | 61,707 | 754,688 |
| 9 | ready | 10 | (19,23) | 54.9 | 61,707 | 1,013,760 |
| 10 | ready | 2 | (14,18), (20,23) | 84.9 | 63,265 | 754,688 |
| 11 | ready | 11 | (15,18), (21,23) | 74.3 | 63,273 | 743,424 |
| 12 | ready | 12 | (16,18), (22,23) | 33.0 | 64,573 | 1,002,496 |
| 13 | ready | 3 | (17,18), (23,23) | 54.1 | 64,692 | 1,048,576 |
| 14 | ready | 13 | (14,24), (18,18), (24,23) | 87.0 | 64,958 | 1,520,640 |
| 15 | ready | 4 | (15,24), (19,18), (25,23) | 69.2 | 64,958 | 2,097,152 |

Total planning time 922 s (15.4 minutes); peak resident memory per run 0.43 to 1.3 GB.
Drift check: `verify_rows` for `d_sig` 8 and 10 re-planned all seven of their rows from
scratch and matched the artifact byte for byte (244 s).

## 5. Commit driver and test

`work_plan::WorkPlan::for_records(N)` derives `N*`, the four layouts, the producer
profiles of `G_main, G_1, G_2` (their scalar rows), the final profile and the schedule
selection from the catalog alone. `commit::PcsProver` commits round 1 (`G_main`) and
round 2 (`G_1`, `G_2`, then `G_3` against the precommitted profiles), each call
transcript-free, and proves one batched opening of all four groups
(`BasisMode::Lagrange`, 32-byte session). `commit::PcsVerifier::new` rebuilds the
setup from the public seed alone (as Aerie's `FalconVerifierSetup::new`) and verifies
with profiles and selection taken from the work plan, never from the proof.

Test `tests/commit_open.rs` (release, 7 cores): dummy tables in each group's honest
alphabet (balanced digits of the listed bases, Booleans for `b_t` and `y^(2)_t`; padded
signatures carry the digits of zero), random claims in `K` at logical points, round 1,
round 2, one batched opening, verification by the seed-only and the prover-derived
verifier, and rejection of a changed claim value, a changed session and swapped
commitments.

| Records `N` | `N*` | Opening proof (bytes) | Planner estimate (B) |
| ---: | ---: | ---: | ---: |
| 256 | 256 | 59,426 | 61,707 |
| 1,000 | 1,024 | 61,003 | 63,265 |

Both batches, including setup, commit, opening and three verifications each, run in
1.4 to 1.9 s together. Larger batches are not committed locally (memory rule); the
catalog rows for them are verified structurally only.

## 6. Regenerating the catalog

The harness is `crates/agg-pcs-config/tests/schedule_catalog_regen.rs`, adapted from
Aerie's `tests/schedule_catalog_regen.rs`: same `akita_planner` entry points
(`find_schedule`, `EmitSpec`, `render_schedule_artifact_outputs_with_validation`), same
estimate cross-check against `akita_schedules`, key lists from `catalogs.rs`. Run it
with the `schedules` feature off, off CPU 2:

```sh
# plan only missing rows of one or more batch sizes (keeps the others; saves after each row)
AGG_CATALOG_D_SIG=15 taskset -c 0,1,3-15 cargo test -p agg-pcs-config --release \
    --no-default-features --test schedule_catalog_regen -- --ignored --exact update_catalog_rows --nocapture
# full regeneration of every row of CATALOG_D_SIG from scratch (Aerie's `regenerate`)
taskset -c 0,1,3-15 cargo test -p agg-pcs-config --release --no-default-features \
    --test schedule_catalog_regen -- --ignored --exact regenerate --nocapture
# drift check: re-plan the rows of the listed sizes and compare bytes with the artifact
AGG_CATALOG_D_SIG=8,10 taskset -c 0,1,3-15 cargo test -p agg-pcs-config --release \
    --no-default-features --test schedule_catalog_regen -- --ignored --exact verify_rows
# which inner digits the planner picks for raw widths 4..20
taskset -c 0,1,3-15 cargo test -p agg-pcs-config --release --no-default-features \
    --test schedule_catalog_regen -- --ignored --exact width_admissibility_probe --nocapture
```

`AKITA_SCHEDULE_GEN_JOBS` bounds the parallel scalar planners (default 3; the search
is memory bound; a run peaked at 1.3 GB here). To add or change a layout: edit
`Group`/`CATALOG_D_SIG` in `src/catalogs.rs`, run `update_catalog_rows` for the new
sizes, then `cargo test -p agg-pcs-config` (the default-feature test
`checked_in_catalog_holds_exactly_the_requested_rows` fails on any missing or extra
row and re-encodes the artifact byte for byte). A change of the Akita pin requires a
full `regenerate`.
