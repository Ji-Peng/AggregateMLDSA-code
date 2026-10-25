# PCS configuration (`agg-pcs-config`)

Akita commitment presets, generated fold-schedule catalogs, work plan and two-round
commit driver for the six witness groups of spec Section 4.1 (lookup layout L16,
`docs/lookup.md`; wire version 1). Akita and Aerie pins as in `README.md` (Akita
`e023b6a`, the revision Aerie `49b6c21` pins, with the `dev-protocol` feature the
catalogs are priced for).

## 1. Presets

Two presets, one per family:

| Parameter | `AggFp64SignedWidth4` | `AggFp64SignedWidth16` |
| --- | --- | --- |
| Field / challenges | `Prime64Offset59` (`p = 2^64 - 59`) / `Ext2` | same |
| SIS modulus profile | `Q64Offset59` | same |
| Ring dimensions, challenge policy | fp64 `Dense` (adaptive A/B/D search) | same |
| `DecompositionParams` | `log_basis 3`, `log_commit_bound 4`, `log_open_bound Some(64)` | `log_basis 3`, `log_commit_bound 16`, `log_open_bound Some(64)` |
| Opening (B/D) basis search | log basis 3..=6 | same |
| Inner (A/source) basis | exactly 4: one balanced base-16 digit | exactly 16: one balanced base-`2^16` digit (deviation L3) |
| Admitted values | `[-8, 7]` | `[-2^15, 2^15 - 1]` |
| Source class | `CommittedSourceClass::BalancedSignedDigit` | same |
| Family name | `agg_mldsa_fp64_signed_width4` | `agg_mldsa_fp64_signed_width16` |
| Groups | `G_main4` (`Z_hi, U_hi, b_0..b_5`), `G_m` (7 multiplicity digits) | `G_main16` (`Z_lo, U_lo, V_lo, V_hi`), `G_1`, `G_2`, `G_3` |

The width-4 parameters are Aerie's `Fp64SignedWidth4`, checked field by field in
`preset::tests::preset_matches_aerie_width4_parameters` (deviation D1); the width-16
preset differs only in the commit bound, the inner range and the family name
(`width16_preset_certifies_the_balanced_sixteen_bit_interval`, which also checks that
Akita's accepted interval of the one-digit geometry is exactly `[-2^15, 2^15 - 1]`,
deviation D2).

A width is a producer admission bound, not a range proof (Akita Remark 10.41): every range
is proven by the PIOP (the lookup into `T_16` for the 16-bit columns and the carries, the
base-4 and base-6 leaves for `Z_hi` and `U_hi`). The stored balanced lookup columns
(`f - 2^15`, deviation L2) make the width-16 admission interval coincide with `T_16`, so
every value the honest producer can commit is an entry of the table; the extractor still
yields arbitrary `F_p` values, which the lookup bounds. The multiplicity digits need no
alphabet check (LogUp Lemma 5 holds for any `m` in `F_p`); width 4 only sizes the honest
producer.

## 2. Which signed widths Akita admits, and with how many inner digits

`akita-params` decides representability: a raw signed width `w` with balanced base
`2^b` digits needs `num_digits_inner_for_bound(w, b)` digits, and the fp64 `Dense`
policy searches `b` in `3..=min(16, w)` (`PROOF_OPTIMIZED_INNER_LOG_BASIS_MAX = 16`).
`preset::admissible_inner_digits` and `min_inner_digits` evaluate exactly this
(test `signed_width_admissibility_table`):

| Signed width `w` | Fewest balanced inner digits | Example |
| ---: | ---: | --- |
| 1..=16 | 1 | width 4: base 16, 1 digit (or base 8, 2 digits); width 16: base `2^16`, 1 digit |
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

Width-16 inner digits for the lookup columns (`width16_inner_probe`, 2026-10-24, the
`G_main16` layouts of 256 and 1,024 signatures):

| Layout | Inner range | Selected | Estimated proof (B) | Planning time |
| --- | --- | --- | ---: | ---: |
| (18, 4) | Dense 3..=16 | base `2^9`, 2 digits | 60,877 | 27.5 s |
| (18, 4) | forced 16 | base `2^16`, 1 digit | 60,877 | 27.2 s |
| (18, 4) | forced 8 | base `2^8`, 3 digits | 60,877 | 12.2 s |
| (20, 4) | Dense 3..=16 | base `2^9`, 2 digits | 61,707 | 44.7 s |
| (20, 4) | forced 16 | base `2^16`, 1 digit | 61,707 | 22.2 s |
| (20, 4) | forced 8 | base `2^8`, 3 digits | 61,707 | 17.0 s |

The estimate is the same for every choice, so the preset forces one digit (the fewest
inner digits, hence the least commit work: 12 inner digits per position instead of 23 in
the base-16 layout; deviation L3).

## 3. Group layouts

Physical layouts `(num_vars, num_polynomials)` in wire and catalog order; `G_m` is the
final group, the others are precommitted. Logical domains below `2^13` cells are
zero-extended with trailing pad variables and opened at the zero-padded point. Both main
groups are opened at `r^(3)`, the bridge groups at its suffixes, `G_m` at the lookup's
table point.

| `d_sig` | `N*` | `G_main16` | `G_main4` | `G_1` | `G_2` | `G_3` | `G_m` (final) |
| ---: | ---: | --- | --- | --- | --- | --- | --- |
| 3 | 8 | (13, 4) | (13, 8) | (13, 6) | (13, 12) | (13, 1) | (16, 7) |
| 8 | 256 | (18, 4) | (18, 8) | (13, 6) | (13, 12) | (13, 1) | (16, 7) |
| 9 | 512 | (19, 4) | (19, 8) | (13, 6) | (13, 12) | (13, 1) | (16, 7) |
| 10 | 1,024 | (20, 4) | (20, 8) | (14, 6) | (13, 12) | (13, 1) | (16, 7) |
| 11 | 2,048 | (21, 4) | (21, 8) | (15, 6) | (13, 12) | (13, 1) | (16, 7) |
| 13 | 8,192 | (23, 4) | (23, 8) | (17, 6) | (13, 12) | (13, 1) | (16, 7) |
| 14 | 16,384 | (24, 4) | (24, 8) | (18, 6) | (14, 12) | (13, 1) | (16, 7) |
| 15 | 32,768 | (25, 4) | (25, 8) | (19, 6) | (15, 12) | (14, 1) | (16, 7) |

(`d_sig` 4 to 7 and 12 follow the same rules: `G_main` at `10 + d_sig` variables, `G_1` at
`13 + max(0, d_sig - 9)`.) Polynomial counts: `G_main16` 4, `G_main4` 8, `G_1` 6, `G_2` 12
(6 parities + 6 carries), `G_3` 1, `G_m` 7; 38 in total (67 in the base-16 layout).

## 4. Catalog status

All thirteen batch sizes of the wire version, `N* = 2^3` to `2^15` (`CATALOG_D_SIG`),
are ready; none is pending. Two artifacts:

- `crates/agg-pcs-config/src/schedules/agg_mldsa_fp64_signed_width16.aks` (222,320 bytes):
  25 scalar producer rows, `G_main16` at 13..=25 variables, `G_1` at 13..=19, `G_2` at
  13..=15, `G_3` at 13..=14;
- `crates/agg-pcs-config/src/schedules/agg_mldsa_fp64_signed_width4.aks` (331,178 bytes):
  13 scalar producer rows (`G_main4` at 13..=25 variables) and 13 grouped rows, one per
  batch size, named `mldsa-44-l16-batch-<N*>-mult-final`, each keyed by `G_m = (16, 7)` and
  the five precommitted profiles in wire order (two families).

Every width-4 group uses one balanced base-16 inner digit and every width-16 group one
balanced base-`2^16` digit; no row needs a setup prefix (checked by
`checked_in_catalog_holds_exactly_the_requested_rows`, which also re-encodes both artifacts
byte for byte).

Planning run (2026-10-24, `update_catalog_rows`, one process for all sizes in the order
8, 10, 3, 6, 9, 11, 5, 4, 7, 12, 13, 14, 15, 3 scalar workers, `taskset -c 0,1,3-15`, load
average 6 to 17 from another agent's builds and this agent's compiles). Wall time per size
(scalar rows not yet planned, then the grouped row); the proof and setup figures are the
planner's estimates for the grouped row.

| `d_sig` | Wall time (s) | Est. opening proof (B) | Setup (field elements) | Base-16 layout est. (B) |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 86.8 | 61,707 | 327,680 | 61,707 |
| 4 | 189.2 | 61,707 | 327,680 | 61,707 |
| 5 | 151.3 | 61,707 | 327,680 | 61,707 |
| 6 | 85.4 | 61,707 | 327,680 | 61,707 |
| 7 | 75.2 | 61,707 | 655,360 | 61,707 |
| 8 | 103.6 | 61,707 | 655,360 | 61,707 |
| 9 | 204.0 | 63,197 | 540,672 | 61,707 |
| 10 | 218.3 | 63,273 | 540,672 | 63,265 |
| 11 | 393.0 | 64,573 | 540,672 | 63,273 |
| 12 | 63.3 | 64,692 | 786,432 | 64,573 |
| 13 | 130.7 | 64,842 | 867,328 | 64,692 |
| 14 | 172.8 | 64,958 | 1,802,240 | 64,958 |
| 15 | 792.5 | 65,282 | 2,658,304 | 64,958 |

Total 2,666 s (44 minutes). The estimated opening proof is at most 2.4% larger than the
base-16 layout's (`d_sig` 9: +2.4%, 11: +2.1%, otherwise below 0.6%): the opening is
dominated by the folding schedule, not by the number of committed polynomials, and the
sixth group adds a little. The measured openings are in `ref/notes/bench-local.md`.

## 5. Commit driver and test

`work_plan::WorkPlan::for_records(N)` derives `N*`, the six layouts, the producer
profiles of the five precommitted groups (each from the scalar row of its family's
catalog, `work_plan::commitment_profile`), the final profile and the schedule selection
from the width-4 catalog's grouped row alone. `commit::PcsProver::commit_group` and
`commit_polynomials` commit one group under its family's catalog (the final group `G_m`
against the precommitted profiles), each call transcript-free, and `open` proves one
batched opening of all six groups under the width-4 catalog (`BasisMode::Lagrange`,
32-byte session). The setup comes from the width-4 catalog, whose grouped rows carry the
width-16 precommitted profiles (Akita's mixed-bound batches). `commit::PcsVerifier::new`
rebuilds the setup from the public seed alone (as Aerie's `FalconVerifierSetup::new`) and
verifies with profiles and selection taken from the work plan, never from the proof.

The PIOP prover drives these calls through `agg_piop::pcs::TableCommitter` (`docs/piop.md`
Section 3): it commits `G_main16` and `G_main4` concurrently (round 1, itself concurrent
with the Keccak side's commitments), `G_1`, `G_2`, `G_3`, `G_m` concurrently (round 2), and
opens with the claims it already holds (the stage-3 terminal values and the multiplicity
digit claims).

Test `tests/commit_open.rs` (release, 7 cores): dummy tables in each group's honest
alphabet (16-bit balanced values, the base-4 and base-8 digits (relaxed H; H-u's base 6 lies inside), Booleans, multiplicity
digits; padded signatures carry the digits of zero), random claims in `K` at logical points
(both main groups at one point), both rounds, one batched opening, verification by the
seed-only and the prover-derived verifier, and rejection of a changed claim value, a
changed session and swapped commitments; and the admission bounds of both families
(`2^15` refused, `2^15 - 1` accepted at width 16; 8 refused at width 4).

## 6. Regenerating the catalogs

The harness is `crates/agg-pcs-config/tests/schedule_catalog_regen.rs`, adapted from
Aerie's `tests/schedule_catalog_regen.rs`: same `akita_planner` entry points
(`find_schedule`, `EmitSpec`, `render_schedule_artifact_outputs_with_validation`), same
estimate cross-check against `akita_schedules`, key lists from `catalogs.rs`; it plans the
scalar rows of each family with its own preset and the grouped rows (in the width-4
catalog) with each precommitted producer taken from its family
(`PrecommittedProducer::from_config::<W16>` or `::<W4>`). Run it with the `schedules`
feature off, off CPU 2:

```sh
# plan only missing rows of one or more batch sizes (keeps the others; saves after each row)
AGG_CATALOG_D_SIG=15 taskset -c 0,1,3-15 cargo test -p agg-pcs-config --release \
    --no-default-features --test schedule_catalog_regen -- --ignored --exact update_catalog_rows --nocapture
# full regeneration of every row of CATALOG_D_SIG from scratch (Aerie's `regenerate`, two emit specs)
taskset -c 0,1,3-15 cargo test -p agg-pcs-config --release --no-default-features \
    --test schedule_catalog_regen -- --ignored --exact regenerate --nocapture
# drift check: re-plan the rows of the listed sizes and compare bytes with the artifacts
AGG_CATALOG_D_SIG=8,10 taskset -c 0,1,3-15 cargo test -p agg-pcs-config --release \
    --no-default-features --test schedule_catalog_regen -- --ignored --exact verify_rows
# which inner digits the planner picks for raw widths 4..20, and for the width-16 columns
taskset -c 0,1,3-15 cargo test -p agg-pcs-config --release --no-default-features \
    --test schedule_catalog_regen -- --ignored --exact width_admissibility_probe --nocapture
taskset -c 0,1,3-15 cargo test -p agg-pcs-config --release --no-default-features \
    --test schedule_catalog_regen -- --ignored --exact width16_inner_probe --nocapture
```

`AKITA_SCHEDULE_GEN_JOBS` bounds the parallel scalar planners (default 3; the search
is memory bound). To add or change a layout: edit `Group`/`CATALOG_D_SIG` in
`src/catalogs.rs`, run `update_catalog_rows` for the new sizes (it prunes the rows of
layouts no longer requested), then `cargo test -p agg-pcs-config` (the default-feature
test `checked_in_catalog_holds_exactly_the_requested_rows` fails on any missing or extra
row and re-encodes both artifacts byte for byte). A change of the Akita pin requires a
full `regenerate`.
