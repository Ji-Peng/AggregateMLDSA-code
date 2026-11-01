//! Offline generation of the `agg_mldsa_fp64_signed_width4` and
//! `agg_mldsa_fp64_signed_width16` schedule catalogs.
//!
//! Adapted from Aerie's `aerie-pcs-config/tests/schedule_catalog_regen.rs`:
//! the same `akita_planner` entry points (`find_schedule`,
//! `EmitSpec`, `render_schedule_artifact_outputs_with_validation`) and the same
//! row construction (a scalar row per precommitted layout, whose root profile is
//! the group's producer profile, and one grouped row per batch size keyed by
//! the final group and the ordered precommitted profiles). The key lists come from
//! `agg_pcs_config::catalogs`. Two families: the width-16 catalog holds
//! the scalar rows of `G_main16, G_1, G_2, G_3`; the width-4 catalog holds the scalar
//! rows of `G_main4` and the grouped rows (final group `G_m`), whose precommitted
//! producers are taken from the scalar row of their own family.
//!
//! Rows are keyed by `(d_sig, relation)` (`catalogs::catalog_batches`): H-u and H-exact
//! share every width-16 row and differ in `G_main4` (8 or 9 polynomials), so the width-4
//! catalog holds both relations' scalar and grouped rows.
//!
//! Incremental update, one batch size at a time (keeps every checked-in row of both
//! relations, plans only missing ones, saves after every planned row):
//!
//! ```text
//! AGG_CATALOG_D_SIG=8 AGG_CATALOG_RELATION=h-exact taskset -c 0,1,3-15 cargo test \
//!     -p agg-pcs-config --release --no-default-features --test all \
//!     -- --ignored --exact schedule_catalog_regen::update_catalog_rows
//! ```
//!
//! `AGG_CATALOG_RELATION` is `h-u`, `h-exact` or `all` (the default). Rows of a batch outside
//! `catalog_batches()` and the requested ones are pruned, so a batch size is first added to
//! the requested list, then to `CATALOG_D_SIG` or `CATALOG_D_SIG_H_EXACT` once its rows exist.
//!
//! Full regeneration of every row of `catalog_batches()` from scratch:
//!
//! ```text
//! taskset -c 0,1,3-15 cargo test -p agg-pcs-config --release --no-default-features \
//!     --test all -- --ignored --exact schedule_catalog_regen::regenerate
//! ```
//!
//! Drift check (re-plans the rows of `AGG_CATALOG_D_SIG` and compares bytes):
//! `-- --ignored --exact schedule_catalog_regen::verify_rows`. `AKITA_SCHEDULE_GEN_JOBS` bounds the
//! planner workers (default 3; the search is memory bound).

use agg_pcs_config::catalogs::{
    catalog_batches, scalar_layouts, BatchLayout, Family, Relation, Set,
};
use agg_pcs_config::preset::{
    AggFp64SignedWidth16 as W16, AggFp64SignedWidth4 as Cfg, WIDTH16_FAMILY, WIDTH4_FAMILY,
};
use akita_config::{policy_of, CommitmentConfig, TrustedScheduleCatalog, ValidatedScheduleCatalog};
use akita_params::{
    CommittedGroupBatchProfile, FoldSchedule, FoldScheduleEstimate, GroupCommitPhaseParams,
    PolynomialGroupLayout, ScheduleLookupKey,
};
use akita_planner::emit::{
    bounded_parallel_filter_map, offline_planning_worker_count, publish_artifact_outputs,
    render_schedule_artifact_outputs_with_validation, GroupedGenerationRequest,
    MaterializationDiagnostics, PrecommittedProducer,
};
use akita_planner::{find_schedule, EmitSpec};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

type Rows = Vec<(CommittedGroupBatchProfile, FoldSchedule)>;

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("schedules")
}

const fn family_name(family: Family) -> &'static str {
    match family {
        Family::Width4 => WIDTH4_FAMILY,
        Family::Width16 => WIDTH16_FAMILY,
    }
}

fn artifact_path(family: Family) -> PathBuf {
    output_dir().join(format!("{}.aks", family_name(family)))
}

/// Relations requested by the environment (`AGG_CATALOG_RELATION=h-exact`), default both.
fn requested_relations() -> Vec<Relation> {
    match std::env::var("AGG_CATALOG_RELATION") {
        Ok(value) if !value.trim().is_empty() && value.trim() != "all" => vec![
            Relation::parse(&value)
                .unwrap_or_else(|| panic!("AGG_CATALOG_RELATION {value:?} is not a relation")),
        ],
        _ => Relation::ALL.to_vec(),
    }
}

/// Parameter sets requested by the environment (`AGG_CATALOG_SET=65`), default all.
fn requested_sets() -> Vec<Set> {
    match std::env::var("AGG_CATALOG_SET") {
        Ok(value) if !value.trim().is_empty() && value.trim() != "all" => {
            let set = match value.trim().to_ascii_lowercase().as_str() {
                "44" | "s44" | "mldsa44" | "mldsa-44" => Set::S44,
                "65" | "s65" | "mldsa65" | "mldsa-65" => Set::S65,
                "87" | "s87" | "mldsa87" | "mldsa-87" => Set::S87,
                _ => panic!("AGG_CATALOG_SET {value:?} is not a supported set"),
            };
            vec![set]
        }
        _ => Set::ALL.to_vec(),
    }
}

/// Batches requested by the environment (`AGG_CATALOG_D_SIG=8,10` under the relations of
/// `AGG_CATALOG_RELATION` and sets of `AGG_CATALOG_SET`), defaulting to every checked-in batch.
fn requested_batches() -> Vec<BatchLayout> {
    match std::env::var("AGG_CATALOG_D_SIG") {
        Ok(value) if !value.trim().is_empty() => {
            let d_sigs: Vec<usize> = value
                .split(',')
                .map(|item| {
                    item.trim().parse::<usize>().unwrap_or_else(|_| {
                        panic!("AGG_CATALOG_D_SIG item {item:?} is not a number")
                    })
                })
                .collect();
            let mut batches = Vec::new();
            for set in requested_sets() {
                for relation in requested_relations() {
                    for &d_sig in &d_sigs {
                        if let Some(layout) = BatchLayout::new(set, d_sig, relation) {
                            batches.push(layout);
                        }
                    }
                }
            }
            batches
        }
        _ => {
            if std::env::var("AGG_CATALOG_SET").is_ok() {
                let mut batches = Vec::new();
                for set in requested_sets() {
                    for layout in agg_pcs_config::catalogs::catalog_batches_for_set(set) {
                        if requested_relations().contains(&layout.relation) {
                            batches.push(layout);
                        }
                    }
                }
                batches
            } else {
                catalog_batches()
                    .into_iter()
                    .filter(|layout| {
                        requested_sets().contains(&layout.set)
                            && requested_relations().contains(&layout.relation)
                    })
                    .collect()
            }
        }
    }
}

#[allow(dead_code)]
fn batch(set: Set, d_sig: usize, relation: Relation) -> BatchLayout {
    BatchLayout::new(set, d_sig, relation)
        .unwrap_or_else(|| panic!("{set:?} d_sig {d_sig} {relation:?} is outside the wire range"))
}

// ---------------------------------------------------------------------------
// Planner entry points (copied from Aerie's harness, generic over the family).

/// Pure DP regeneration of one scalar row of the family `C`.
fn regen_scalar<C: CommitmentConfig>(
    key: PolynomialGroupLayout,
) -> Result<FoldSchedule, akita_error::AkitaError> {
    let key = ScheduleLookupKey::single(key);
    let planned = find_schedule(
        &key,
        C::committed_source_contract()?,
        &[],
        &policy_of::<C>(),
        C::ring_challenge_config,
    )?;
    planned.schedule.validate_structure()?;
    assert_eq!(
        estimate_json(&planned.estimate),
        expanded_estimate_json::<C>(&key, &planned.schedule),
        "planner estimate must match expanded schedule sizing for {key:?}",
    );
    Ok(planned.schedule)
}

fn regen(
    family: Family,
    key: PolynomialGroupLayout,
) -> Result<FoldSchedule, akita_error::AkitaError> {
    match family {
        Family::Width4 => regen_scalar::<Cfg>(key),
        Family::Width16 => regen_scalar::<W16>(key),
    }
}

/// Pure DP regeneration of one grouped row (final group `G_m`, width-4 family).
#[allow(clippy::needless_pass_by_value)]
fn regen_group_batch(
    request: GroupedGenerationRequest,
) -> Result<FoldSchedule, akita_error::AkitaError> {
    let key = request.key();
    let planned = find_schedule(
        &key,
        Cfg::committed_source_contract()?,
        &request.source_contracts(),
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
    )?;
    planned.schedule.validate_structure()?;
    assert_eq!(
        estimate_json(&planned.estimate),
        expanded_estimate_json::<Cfg>(&key, &planned.schedule),
        "planner estimate must match expanded schedule sizing for {key:?}",
    );
    eprintln!(
        "planned grouped final {:?} + {} precommits: estimated proof payload {} B, setup {} field elements",
        key.final_group,
        key.precommitteds.len(),
        planned.estimate.estimated_proof_payload_bytes()?,
        planned.estimate.estimated_num_setup_field_elements,
    );
    Ok(planned.schedule)
}

fn estimate_json(estimate: &FoldScheduleEstimate) -> serde_json::Value {
    serde_json::json!({
        "estimated_proof_payload_bytes": estimate.estimated_proof_payload_bytes().expect("proof byte estimate"),
        "estimated_num_setup_field_elements": estimate.estimated_num_setup_field_elements,
    })
}

/// Reprice a schedule with Akita's own byte and setup sizing rules.
fn expanded_estimate_json<C: CommitmentConfig>(
    key: &ScheduleLookupKey,
    schedule: &FoldSchedule,
) -> serde_json::Value {
    serde_json::json!({
        "estimated_proof_payload_bytes": akita_schedules::expanded_schedule_proof_estimate_bytes(
            key, schedule, &policy_of::<C>(),
        ).expect("expanded proof estimate"),
        "estimated_num_setup_field_elements": akita_params::setup_matrix_capacity_for_schedule(schedule)
            .expect("expanded setup capacity").num_field_elements,
    })
}

/// Session cache of exact scalar DP results (Aerie's `SCALAR_PREPLANS`), per family.
static SCALAR_PREPLANS: Mutex<Vec<(Family, PolynomialGroupLayout, FoldSchedule)>> =
    Mutex::new(Vec::new());

fn cached_scalar(family: Family, key: PolynomialGroupLayout) -> Option<FoldSchedule> {
    SCALAR_PREPLANS
        .lock()
        .expect("scalar preplan cache poisoned")
        .iter()
        .find(|(f, cached, _)| *f == family && *cached == key)
        .map(|(_, _, schedule)| schedule.clone())
}

fn cache_scalar(family: Family, key: PolynomialGroupLayout, schedule: FoldSchedule) {
    let mut cache = SCALAR_PREPLANS
        .lock()
        .expect("scalar preplan cache poisoned");
    if !cache.iter().any(|(f, cached, _)| *f == family && *cached == key) {
        cache.push((family, key, schedule));
    }
}

fn preplanned_scalar(
    family: Family,
    key: PolynomialGroupLayout,
) -> Result<FoldSchedule, akita_error::AkitaError> {
    if let Some(schedule) = cached_scalar(family, key) {
        return Ok(schedule);
    }
    let started = Instant::now();
    eprintln!("preplanning scalar {family:?} {key:?}");
    let schedule = regen(family, key)?;
    eprintln!("planned scalar {family:?} {key:?} in {:.1?}", started.elapsed());
    cache_scalar(family, key, schedule.clone());
    Ok(schedule)
}

fn preplanned_scalar_for(
    family: Family,
    keys: &[PolynomialGroupLayout],
) -> Vec<(PolynomialGroupLayout, FoldSchedule)> {
    keys.iter()
        .filter_map(|&key| cached_scalar(family, key).map(|schedule| (key, schedule)))
        .collect()
}

/// A precommitted producer derived from its complete scalar row in its own family.
fn precommitted_producer(family: Family, layout: PolynomialGroupLayout) -> PrecommittedProducer {
    let schedule = preplanned_scalar(family, layout).unwrap_or_else(|error| {
        panic!("precommit profile planning failed for {family:?} {layout:?}: {error}")
    });
    let profile = GroupCommitPhaseParams::try_from_params(layout, &schedule.root.params)
        .expect("scalar derived precommit profile");
    match family {
        Family::Width4 => PrecommittedProducer::from_config::<Cfg>(profile),
        Family::Width16 => PrecommittedProducer::from_config::<W16>(profile),
    }
    .expect("precommitted producer")
}

fn grouped_request(layout: &BatchLayout) -> GroupedGenerationRequest {
    GroupedGenerationRequest::new(
        layout.final_group(),
        layout
            .precommitted_groups()
            .into_iter()
            .map(|(group, shape)| precommitted_producer(group.family(), shape))
            .collect(),
    )
}

/// The profiles stored beside a planned schedule (Aerie `rendered_profiles`).
fn rendered_profiles(
    key: &ScheduleLookupKey,
    schedule: &FoldSchedule,
) -> Result<CommittedGroupBatchProfile, akita_error::AkitaError> {
    Ok(CommittedGroupBatchProfile {
        final_group: GroupCommitPhaseParams::try_from_params(
            key.final_group,
            &schedule.root.params,
        )?,
        precommitteds: key.precommitteds.clone(),
    })
}

fn write_catalog<C: CommitmentConfig>(rows: Rows, destination: &Path) {
    let catalog = ValidatedScheduleCatalog::try_new(
        C::schedule_family_name(),
        rows,
        &policy_of::<C>(),
        C::ring_challenge_config,
    )
    .expect("validate catalog");
    std::fs::write(
        destination,
        catalog.to_artifact_bytes().expect("encode catalog"),
    )
    .expect("write catalog");
    eprintln!("wrote {} ({} rows)", destination.display(), catalog.len());
}

fn rows_of<C: CommitmentConfig>(catalog: &TrustedScheduleCatalog<C>) -> Rows {
    catalog
        .rows()
        .map(|row| (row.profiles().clone(), row.schedule().clone()))
        .collect()
}

/// The artifact of `C` on disk, if any (read directly, so this works with the
/// `schedules` feature off).
fn on_disk_catalog<C: CommitmentConfig>(family: Family) -> Option<TrustedScheduleCatalog<C>> {
    let bytes = std::fs::read(artifact_path(family)).ok()?;
    Some(TrustedScheduleCatalog::from_artifact_bytes(&bytes).expect("load on-disk catalog"))
}

fn on_disk_rows(family: Family) -> Option<Rows> {
    match family {
        Family::Width4 => on_disk_catalog::<Cfg>(family).map(|c| rows_of(&c)),
        Family::Width16 => on_disk_catalog::<W16>(family).map(|c| rows_of(&c)),
    }
}

fn seed_scalar_preplans(family: Family, rows: &Rows) {
    for (profiles, schedule) in rows {
        if profiles.precommitteds.is_empty() {
            cache_scalar(family, profiles.final_group.group, schedule.clone());
        }
    }
}

fn row_key(profiles: &CommittedGroupBatchProfile) -> ScheduleLookupKey {
    ScheduleLookupKey {
        final_group: profiles.final_group.group,
        precommitteds: profiles.precommitteds.clone(),
    }
}

/// Whether a catalog row of `family` belongs to one of `batches`.
fn row_is_requested(
    family: Family,
    profiles: &CommittedGroupBatchProfile,
    batches: &[BatchLayout],
) -> bool {
    if profiles.precommitteds.is_empty() {
        return scalar_layouts(batches, family).contains(&profiles.final_group.group);
    }
    family == Family::Width4
        && batches.iter().any(|layout| {
            profiles.final_group.group == layout.final_group()
                && profiles
                    .precommitteds
                    .iter()
                    .map(|profile| profile.group)
                    .eq(layout.precommitted().iter().copied())
        })
}

fn save(family: Family, rows: &Rows) {
    match family {
        Family::Width4 => write_catalog::<Cfg>(rows.clone(), &artifact_path(family)),
        Family::Width16 => write_catalog::<W16>(rows.clone(), &artifact_path(family)),
    }
}

// ---------------------------------------------------------------------------
// Offline tests (ignored by default).

/// Plan the rows of the requested batches that the on-disk catalogs lack, keep every
/// on-disk row of `catalog_batches()` (both relations) or of a requested batch, drop the
/// rest, and save after every planned row (so a run killed by a time cap keeps what it
/// finished).
#[test]
#[ignore = "offline incremental catalog update; rewrites src/schedules/"]
fn update_catalog_rows() {
    let requested = requested_batches();
    let mut keep = catalog_batches();
    for layout in &requested {
        if !keep.contains(layout) {
            keep.push(*layout);
        }
    }
    let mut rows: [Rows; 2] = [Vec::new(), Vec::new()];
    for (slot, family) in rows.iter_mut().zip(Family::ALL) {
        if let Some(on_disk) = on_disk_rows(family) {
            seed_scalar_preplans(family, &on_disk);
            for row in on_disk {
                if row_is_requested(family, &row.0, &keep) {
                    slot.push(row);
                } else {
                    eprintln!("pruned {family:?} row: final {:?}", row.0.final_group.group);
                }
            }
        }
    }
    for layout in requested {
        let (d_sig, relation) = (layout.d_sig, layout.relation);
        let started = Instant::now();
        for (slot, family) in rows.iter_mut().zip(Family::ALL) {
            let mut missing: Vec<PolynomialGroupLayout> = layout
                .precommitted_groups()
                .into_iter()
                .filter(|(group, _)| group.family() == family)
                .map(|(_, shape)| shape)
                .filter(|&shape| {
                    !slot.iter().any(|(profiles, _)| {
                        profiles.precommitteds.is_empty() && profiles.final_group.group == shape
                    })
                })
                .collect();
            missing.sort_by_key(|shape| (shape.num_vars(), shape.num_polynomials()));
            missing.dedup();
            let workers = offline_planning_worker_count(missing.len());
            let planned = bounded_parallel_filter_map(&missing, workers, |&shape| {
                let schedule = preplanned_scalar(family, shape)
                    .map_err(|error| format!("scalar planning failed for {shape:?}: {error}"))?;
                Ok(Some((shape, schedule)))
            })
            .expect("plan scalar rows");
            for (shape, schedule) in planned {
                let key = ScheduleLookupKey::single(shape);
                let profiles = rendered_profiles(&key, &schedule).expect("scalar profile");
                slot.push((profiles, schedule));
                save(family, slot);
            }
        }
        let request = grouped_request(&layout);
        let key = request.key();
        if !rows[0].iter().any(|(profiles, _)| row_key(profiles) == key) {
            let grouped_started = Instant::now();
            let schedule = regen_group_batch(request).expect("grouped planning");
            eprintln!(
                "planned grouped {} in {:.1?}",
                layout.row_name(),
                grouped_started.elapsed()
            );
            let profiles = rendered_profiles(&key, &schedule).expect("grouped profile");
            rows[0].push((profiles, schedule));
            save(Family::Width4, &rows[0]);
        }
        eprintln!(
            "d_sig {d_sig}, {relation} ({}): all rows ready in {:.1?}",
            layout.row_name(),
            started.elapsed()
        );
    }
    for (slot, family) in rows.iter().zip(Family::ALL) {
        save(family, slot);
    }
}

/// Full regeneration of `catalog_batches()` through Akita's emit pipeline, as in
/// Aerie's `regenerate`: one emit spec per family, the grouped rows in the width-4 one.
#[test]
#[ignore = "offline DP regeneration; rewrites src/schedules/"]
fn regenerate() {
    let batches = catalog_batches();
    let keys4 = scalar_layouts(&batches, Family::Width4);
    let keys16 = scalar_layouts(&batches, Family::Width16);
    let all: Vec<(Family, PolynomialGroupLayout)> = keys4
        .iter()
        .map(|&key| (Family::Width4, key))
        .chain(keys16.iter().map(|&key| (Family::Width16, key)))
        .collect();
    let workers = offline_planning_worker_count(all.len());
    eprintln!("preplanning {} scalar rows with {workers} workers", all.len());
    bounded_parallel_filter_map(&all, workers, |&(family, key)| {
        preplanned_scalar(family, key)
            .map(|_| Some(()))
            .map_err(|error| format!("scalar preplanning failed for {key:?}: {error}"))
    })
    .expect("prewarm scalar schedule plans");
    let grouped_requests = batches.iter().map(grouped_request).collect();
    let spec4 = EmitSpec {
        family_name: WIDTH4_FAMILY,
        policy: policy_of::<Cfg>(),
        source_contract: Cfg::committed_source_contract().expect("width-4 source contract"),
        preplanned_scalar: preplanned_scalar_for(Family::Width4, &keys4),
        keys: keys4,
        grouped_requests,
        output_dir: output_dir(),
        regen: regen_scalar::<Cfg>,
        regen_group_batch,
        ring_challenge_config: <Cfg as CommitmentConfig>::ring_challenge_config,
    };
    let spec16 = EmitSpec {
        family_name: WIDTH16_FAMILY,
        policy: policy_of::<W16>(),
        source_contract: W16::committed_source_contract().expect("width-16 source contract"),
        preplanned_scalar: preplanned_scalar_for(Family::Width16, &keys16),
        keys: keys16,
        grouped_requests: Vec::new(),
        output_dir: output_dir(),
        regen: regen_scalar::<W16>,
        regen_group_batch,
        ring_challenge_config: <W16 as CommitmentConfig>::ring_challenge_config,
    };
    let outputs = render_schedule_artifact_outputs_with_validation(
        &[spec4, spec16],
        MaterializationDiagnostics { row_progress: true },
        |_, _| Ok(()),
    )
    .expect("render generated outputs");
    for destination in publish_artifact_outputs(outputs).expect("publish generated outputs") {
        eprintln!("wrote {}", destination.display());
    }
}

/// Drift check: re-plan the rows of the requested batches from scratch and
/// compare them byte-exactly with the on-disk rows. Grouped producers are
/// seeded from the on-disk scalar rows, each of which is itself re-planned.
#[test]
#[ignore = "offline DP drift check; set AGG_CATALOG_D_SIG"]
fn verify_rows() {
    let batches = requested_batches();
    let mut drift = Vec::new();
    let on_disk: Vec<Rows> = Family::ALL
        .iter()
        .map(|&family| on_disk_rows(family).expect("on-disk catalog"))
        .collect();
    for (family, rows) in Family::ALL.into_iter().zip(&on_disk) {
        for group in scalar_layouts(&batches, family) {
            let fresh = regen(family, group).expect("fresh scalar planning");
            let row = rows
                .iter()
                .find(|(profiles, _)| {
                    profiles.precommitteds.is_empty() && profiles.final_group.group == group
                })
                .expect("checked-in scalar row");
            if serde_json::to_vec(&row.1).unwrap() != serde_json::to_vec(&fresh).unwrap() {
                drift.push(format!("scalar {family:?} {group:?}"));
            }
        }
        seed_scalar_preplans(family, rows);
    }
    for layout in &batches {
        let request = grouped_request(layout);
        let key = request.key();
        let fresh = regen_group_batch(request).expect("fresh grouped planning");
        let row = on_disk[0]
            .iter()
            .find(|(profiles, _)| row_key(profiles) == key)
            .expect("checked-in grouped row");
        let profiles = rendered_profiles(&key, &fresh).expect("profiles");
        if serde_json::to_vec(&row.1).unwrap() != serde_json::to_vec(&fresh).unwrap()
            || serde_json::to_vec(&row.0).unwrap() != serde_json::to_vec(&profiles).unwrap()
        {
            drift.push(format!("grouped {}", layout.row_name()));
        }
    }
    assert!(drift.is_empty(), "catalog drift: {drift:?}");
}

// ---------------------------------------------------------------------------
// Cheap checks of the checked-in artifact (run by `cargo test`).

#[cfg(feature = "schedules")]
fn requested_keys() -> (Vec<ScheduleLookupKey>, Vec<ScheduleLookupKey>) {
    let batches = catalog_batches();
    let mut keys4: Vec<ScheduleLookupKey> = scalar_layouts(&batches, Family::Width4)
        .into_iter()
        .map(ScheduleLookupKey::single)
        .collect();
    let keys16: Vec<ScheduleLookupKey> = scalar_layouts(&batches, Family::Width16)
        .into_iter()
        .map(ScheduleLookupKey::single)
        .collect();
    for layout in batches {
        keys4.push(ScheduleLookupKey {
            final_group: layout.final_group(),
            precommitteds: layout
                .precommitted_groups()
                .into_iter()
                .map(|(group, shape)| {
                    agg_pcs_config::work_plan::commitment_profile(group.family(), shape)
                        .expect("scalar producer row")
                })
                .collect(),
        });
    }
    (keys4, keys16)
}

#[cfg(feature = "schedules")]
fn check_catalog<C: CommitmentConfig>(
    catalog: &TrustedScheduleCatalog<C>,
    family: Family,
    keys: &[ScheduleLookupKey],
) {
    for (index, key) in keys.iter().enumerate() {
        assert!(
            !keys[..index].contains(key),
            "duplicate requested key {key:?}"
        );
        catalog
            .resolve_key(key)
            .unwrap_or_else(|error| panic!("requested key {key:?} has no row: {error}"));
    }
    assert_eq!(
        keys.len(),
        catalog.len(),
        "{family:?} catalog contains unrequested rows"
    );
    let expected = match family {
        Family::Width4 => (4, 1),
        Family::Width16 => (16, 1),
    };
    for row in catalog.rows() {
        // The row's own (final or scalar) group commits with one balanced inner digit of
        // its family; the precommitted profiles of grouped rows are checked by the caller.
        let profile = &row.profiles().final_group;
        assert_eq!(
            (
                profile.inner.digits.log_basis,
                profile.inner.digits.num_digits
            ),
            expected,
            "{family:?} group {:?}",
            profile.group
        );
        let schedule = row.schedule();
        for fold in std::iter::once(&schedule.root).chain(&schedule.recursive_folds) {
            assert!(fold.params.setup_prefix().is_none());
            for group in fold.params.groups() {
                assert!(group.setup_natural_len.is_none());
                assert!(group.slot_id().is_none());
            }
        }
    }
    let audited = ValidatedScheduleCatalog::try_new(
        C::schedule_family_name(),
        rows_of(catalog),
        &policy_of::<C>(),
        C::ring_challenge_config,
    )
    .expect("re-audit rows");
    let on_disk = std::fs::read(artifact_path(family)).expect("artifact");
    assert_eq!(audited.to_artifact_bytes().expect("encode"), on_disk);
}

/// Both artifacts hold exactly the requested rows, every group commits with one
/// balanced inner digit (base 16 at width 4, base `2^16` at width 16), every grouped
/// row carries the five precommitted profiles in wire order, no row needs a setup
/// prefix, and the re-audited rows re-encode to the checked-in bytes.
#[cfg(feature = "schedules")]
#[test]
fn checked_in_catalog_holds_exactly_the_requested_rows() {
    let (keys4, keys16) = requested_keys();
    check_catalog(Cfg::catalog().expect("width-4 catalog"), Family::Width4, &keys4);
    check_catalog(W16::catalog().expect("width-16 catalog"), Family::Width16, &keys16);
    let batches = catalog_batches();
    let grouped = &keys4[keys4.len() - batches.len()..];
    for (layout, key) in batches.iter().zip(grouped) {
        for ((group, _), profile) in layout.precommitted_groups().into_iter().zip(&key.precommitteds)
        {
            let digits = (profile.inner.digits.log_basis, profile.inner.digits.num_digits);
            let at = layout.row_name();
            match group.family() {
                Family::Width4 => assert_eq!(digits, (4, 1), "{group:?} in {at}"),
                Family::Width16 => assert_eq!(digits, (16, 1), "{group:?} in {at}"),
            }
        }
    }
}
