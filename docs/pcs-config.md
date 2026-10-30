# PCS configuration (`agg-pcs-config`)

Akita commitment presets, generated fold-schedule catalogs, work plan and two-round
commit driver for the seven witness groups of spec Section 4.1 (lookup layout L16 with the
projection of `v`, `docs/lookup.md`, deviation K24; H-u rows `mldsa-44-p16-*`, H-exact rows
`mldsa-44-p16x-*`, H-FIPS rows `mldsa-44-p16f-*`; the `l16*` rows of Section 4 below are the
six-group layout before K24). Akita and Aerie pins as in `README.md` (Akita
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
| Groups | `G_main4` (`Z_hi, U_hi, b_0..b_5`), `G_m` (7 multiplicity digits) | `G_main16` (`Z_lo, U_lo, V_lo, V_hi`), `G_1`, `G_2`, `G_3` (in `G_2`'s layout), `G_k` (`K_0, K_1, K_2`) |

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

Planner probe (measured once; the question is settled): a planning-only preset per width, equal to
the width-4 preset except `log_commit_bound = w` and Dense's inner search 3..=16, planned
for one polynomial at 13 variables (this machine). Every width is
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

Width-16 inner digits for the lookup columns (planner probe, the `G_main16` layouts of 256
and 1,024 signatures; settled by deviation L3):

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

All thirteen batch sizes of the wire version, `N* = 2^3` to `2^15`, are ready for the three
relations (`CATALOG_D_SIG` for H-u, `CATALOG_D_SIG_H_EXACT` for H-exact,
`CATALOG_D_SIG_H_FIPS` for H-FIPS; rows are keyed by `(d_sig, relation)`,
`catalogs::catalog_batches`); none is pending. Two artifacts (width 16: 246,510 B with 28
rows; width 4: 996,253 B with 78 rows; the H-FIPS rows are listed at the end of this section):

- `crates/agg-pcs-config/src/schedules/agg_mldsa_fp64_signed_width16.aks`:
  25 scalar producer rows, `G_main16` at 13..=25 variables, `G_1` at 13..=19, `G_2` at
  13..=15, `G_3` at 13..=14;
- `crates/agg-pcs-config/src/schedules/agg_mldsa_fp64_signed_width4.aks`: per relation 13
  scalar producer rows (`G_main4` at 13..=25
  variables with 8 polynomials for H-u, 9 for H-exact) and 13 grouped rows, one per batch
  size, named `mldsa-44-l16-batch-<N*>-mult-final` (H-u) and
  `mldsa-44-l16x-batch-<N*>-mult-final` (H-exact), each keyed by `G_m = (16, 7)` and the
  five precommitted profiles in wire order (two families). The width-16 rows are shared:
  H-exact changes only `G_main4` (`docs/h-exact.md` 2.2).

Every width-4 group uses one balanced base-16 inner digit and every width-16 group one
balanced base-`2^16` digit; no row needs a setup prefix (checked by
`checked_in_catalog_holds_exactly_the_requested_rows`, which also re-encodes both artifacts
byte for byte).

Planning run (`update_catalog_rows`, one process for all sizes in the order 8, 10, 3, 6, 9,
11, 5, 4, 7, 12, 13, 14, 15, 3 scalar workers, `taskset -c 0,1,3-15`, a shared machine). Wall
time per size
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

Total 2,666 s (44 minutes).

H-exact rows (`update_catalog_rows` with `AGG_CATALOG_RELATION=h-exact`, 3 scalar workers,
`taskset -c 0,1,3-15`, a shared machine): `d_sig` 3 to 10 in 732 s, 11 to 15 in 885 s, 1,617 s
(27
minutes) for the 26 rows, within the 20 to 35 minutes estimated in `docs/h-exact.md` 2.3.
The planner's opening estimate of every H-exact grouped row equals the H-u row's (61,707 B
at `d_sig` 3 to 8, 63,197, 63,273, 64,573, 64,692, 64,842, 64,958 and 65,282 B at 9 to 15);
the setup sizes differ slightly (563,200 field elements at `d_sig` 9 to 11, 794,112,
912,384, 1,892,352 and 2,838,528 at 12 to 15). Measured at `N = 1,024`: H-exact's Akita
opening is 16 B longer than H-u's (61,010 against 60,994 B, `docs/cli.md` Section 4). The
Akita grinding plans of the H-exact rows are pinned in `tests/all/akita_grinding.rs`. The rows of `d_sig` 3 and 4 have equal
plan digests (`e9ec8b52...`). This is benign: Akita's
`GrindingPlan::canonical_bytes` covers only the policy, the challenge order and the runs, not
the layout; the two rows draw identical site lists; the layout is bound separately through the
schedule key, the row name and the instance descriptor, and all 52 width-4 catalog schedules
hash differently. The estimated opening proof is at most 2.4% larger than the
base-16 layout's (`d_sig` 9: +2.4%, 11: +2.1%, otherwise below 0.6%): the opening is
dominated by the folding schedule, not by the number of committed polynomials, and the
sixth group adds a little.

H-FIPS rows (`update_catalog_rows` with `AGG_CATALOG_RELATION=h-fips` and
`AGG_CATALOG_D_SIG=3,...,15`, 3 scalar workers, `taskset -c 0,1,3-15`): 1,105.6 s (18
minutes, measured) for 13 width-4 scalar rows (`G_main4` at `(13..25, 12)`: `bz_hi`, `l_hi`,
six bits, `a_z`, `s`, `delta`, `e`), 3 width-16 scalar rows (`G_2` at `(13|14|15, 13)`: the
hint slack `u_h` is its thirteenth polynomial, deviation L10) and 13 grouped rows
`mldsa-44-l16f-batch-<N*>-mult-final`. The planner's opening estimates equal H-u's except at
`2^15` (65,398 B against 65,282 B, estimate); setups grow (3,390,464 field elements at `2^15`).
Every H-FIPS grouped row keeps at least `2^128.0881` of accepted fold support (`2^128.5633` at
`d_sig` 13 and 15; `tests/all/akita_fold_support.rs`), and the grinding plans are pinned in
`tests/all/akita_grinding.rs` (`PINNED_H_FIPS`, all distinct from the H-u and H-exact rows').

### 4.1 The seven-group layout of deviation K24 (projection of `v`)

K24 adds the width-16 group `G_k = (K_0, K_1, K_2)` on `D_k` (`7 + d_sig` variables, 3
polynomials) after `G_3`, and gives `G_3` the layout of `G_2` (its carry, then zero
polynomials up to `G_2`'s count, `G_2`'s physical variables). The reason is the planner's
root opening search: the coefficient-packing openings of the precommitted groups are
enumerated as a Cartesian product over their distinct profile classes, capped at
`MAX_PRECOMMIT_OPENING_PRODUCTS = 256` assignments per opening dimension; a larger product
drops the dimension. With `G_k` as a sixth distinct class the grouped row of `N* = 2^15`
had no schedule (`UnsupportedSchedule("no multi-group schedule in the audited fold domain
for num_vars=16; 6 distinct precommitted classes can exceed the 256-assignment root opening
bound ...")`); with `G_3` in `G_2`'s class (equal layouts give equal producer profiles) the
row plans. Rows `mldsa-44-p16{,x,f}-batch-<N*>-mult-final`, regenerated with
`update_catalog_rows` (3 scalar workers, all relations, a shared machine): new width-16
scalar rows for `G_k` at `(10..=22, 3)` (physical `max(13, 7 + d_sig)`), the `G_3` rows
coincide with `G_2`'s, the width-4 scalar rows are unchanged, and 39 new grouped rows. The
planner's opening estimates (H-u): 61,707 B at `d_sig` 3 to 6, 63,197 at 7 and 8, 63,265 at
9, 63,273 at 10, 64,692 at 11 and 12, 64,958 at 13, 65,282 at 14 and 65,398 at 15 (before
K24: 61,707 at 3 to 8, 63,197 at 9, 63,273 at 10, 64,573 at 11, 64,692 at 12, 64,842 at 13,
64,958 at 14 and 65,282 at 15; estimates). The H-exact rows have H-u's estimates; the
H-FIPS rows reach each step one size earlier (63,197 at 7, 63,265 at 8, 63,273 at 9, 64,573
at 10, 64,692 at 11, 64,842 at 12, 64,958 at 13, 65,282 at 14, 65,398 at 15). Planning took
4,658 s (78 minutes) for all relations, plus 286 s for the H-u row of `2^15`.

## 5. Commit driver and test

`work_plan::WorkPlan::for_records(N)` derives `N*`, the seven layouts, the producer
profiles of the six precommitted groups (each from the scalar row of its family's
catalog, `work_plan::commitment_profile`), the final profile and the schedule selection
from the width-4 catalog's grouped row alone. `commit::PcsProver::commit_group` and
`commit_polynomials` commit one group under its family's catalog (the final group `G_m`
against the precommitted profiles), each call transcript-free, and `open` proves one
batched opening of all seven groups under the width-4 catalog (`BasisMode::Lagrange`,
32-byte session). The setup comes from the width-4 catalog, whose grouped rows carry the
width-16 precommitted profiles (Akita's mixed-bound batches). `commit::PcsVerifier::new`
rebuilds the setup from the public seed alone (as Aerie's `FalconVerifierSetup::new`) and
verifies with profiles and selection taken from the work plan, never from the proof.

The PIOP prover drives these calls through `agg_piop::pcs::TableCommitter` (`docs/piop.md`
Section 3): it commits `G_main16` and `G_main4` concurrently (round 1, itself concurrent
with the Keccak side's commitments), `G_1`, `G_2`, `G_3`, `G_m` concurrently (round 2), and
opens with the claims it already holds (the stage-3 terminal values and the multiplicity
digit claims).

Test `tests/all/commit_open.rs` (release, 7 cores): dummy tables in each group's honest
alphabet (16-bit balanced values, the base-4 and base-8 digits (relaxed H; H-u's base 6 lies inside), Booleans, multiplicity
digits; padded signatures carry the digits of zero), random claims in `K` at logical points
(both main groups at one point), both rounds, one batched opening, verification by the
seed-only and the prover-derived verifier, and rejection of a changed claim value, a
changed session and swapped commitments; and the admission bounds of both families
(`2^15` refused, `2^15 - 1` accepted at width 16; 8 refused at width 4).

## 6. Regenerating the catalogs

The harness is `crates/agg-pcs-config/tests/all/schedule_catalog_regen.rs`, adapted from
Aerie's `tests/schedule_catalog_regen.rs`: same `akita_planner` entry points
(`find_schedule`, `EmitSpec`, `render_schedule_artifact_outputs_with_validation`), same
estimate cross-check against `akita_schedules`, key lists from `catalogs.rs`; it plans the
scalar rows of each family with its own preset and the grouped rows (in the width-4
catalog) with each precommitted producer taken from its family
(`PrecommittedProducer::from_config::<W16>` or `::<W4>`). Run it with the `schedules`
feature off, off CPU 2:

```sh
# plan only missing rows of one or more batch sizes (keeps the others, of both relations;
# saves after each row); AGG_CATALOG_RELATION = h-u, h-exact or all (default)
AGG_CATALOG_D_SIG=15 AGG_CATALOG_RELATION=h-exact taskset -c 0,1,3-15 cargo test \
    -p agg-pcs-config --release --no-default-features --test all \
    -- --ignored --exact schedule_catalog_regen::update_catalog_rows --nocapture
# full regeneration of every row of catalog_batches() from scratch (Aerie's `regenerate`, two emit specs)
taskset -c 0,1,3-15 cargo test -p agg-pcs-config --release --no-default-features \
    --test all -- --ignored --exact schedule_catalog_regen::regenerate --nocapture
# drift check: re-plan the rows of the listed sizes and compare bytes with the artifacts
AGG_CATALOG_D_SIG=8,10 taskset -c 0,1,3-15 cargo test -p agg-pcs-config --release \
    --no-default-features --test all -- --ignored --exact schedule_catalog_regen::verify_rows
```

`AKITA_SCHEDULE_GEN_JOBS` bounds the parallel scalar planners (default 3; the search
is memory bound). To add or change a layout: edit `Group`/`Relation`/`CATALOG_D_SIG`/
`CATALOG_D_SIG_H_EXACT` in `src/catalogs.rs`, run `update_catalog_rows` for the new batches
(it keeps every row of `catalog_batches()` and of the requested batches and prunes the
rest, so a new batch size is requested through the environment first and added to the
constant once its rows exist), then `cargo test -p agg-pcs-config` (the default-feature
test `checked_in_catalog_holds_exactly_the_requested_rows` fails on any missing or extra
row and re-encodes both artifacts byte for byte). A change of the Akita pin requires a
full `regenerate`.
