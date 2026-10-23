//! Offline generation of the `agg_mldsa_fp64_signed_width4` schedule catalog.
//!
//! Adapted from Aerie's `aerie-pcs-config/tests/schedule_catalog_regen.rs`
//! (rev 49b6c21): the same `akita_planner` entry points (`find_schedule`,
//! `EmitSpec`, `render_schedule_artifact_outputs_with_validation`) and the same
//! row construction (a scalar row per precommitted layout, whose root profile is
//! the group's producer profile, and one grouped row per batch size keyed by
//! the final group and the ordered precommitted profiles). Only the key lists
//! differ: they come from `agg_pcs_config::catalogs` (spec Section 4.1).
//!
//! Incremental update, one batch size at a time (keeps checked-in rows,
//! plans only missing ones, saves after every planned row):
//!
//! ```text
//! AGG_CATALOG_D_SIG=8 taskset -c 0,1,3-15 cargo test -p agg-pcs-config --release \
//!     --no-default-features --test schedule_catalog_regen -- --ignored --exact update_catalog_rows
//! ```
//!
//! Full regeneration of every row of `CATALOG_D_SIG` from scratch:
//!
//! ```text
//! taskset -c 0,1,3-15 cargo test -p agg-pcs-config --release --no-default-features \
//!     --test schedule_catalog_regen -- --ignored --exact regenerate
//! ```
//!
//! Drift check (re-plans the rows of `AGG_CATALOG_D_SIG` and compares bytes):
//! `-- --ignored --exact verify_rows`. `AKITA_SCHEDULE_GEN_JOBS` bounds the
//! planner workers (default 3; the search is memory bound).

use agg_pcs_config::catalogs::{scalar_layouts, BatchLayout, CATALOG_D_SIG};
use agg_pcs_config::preset::{AggFp64SignedWidth4 as Cfg, WIDTH4_FAMILY};
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

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("schedules")
}

fn artifact_path() -> PathBuf {
    output_dir().join(format!("{WIDTH4_FAMILY}.aks"))
}

/// Batch sizes requested by the environment (`AGG_CATALOG_D_SIG=8,10`),
/// defaulting to every checked-in batch size.
fn requested_d_sigs() -> Vec<usize> {
    match std::env::var("AGG_CATALOG_D_SIG") {
        Ok(value) if !value.trim().is_empty() => value
            .split(',')
            .map(|item| {
                item.trim()
                    .parse::<usize>()
                    .unwrap_or_else(|_| panic!("AGG_CATALOG_D_SIG item {item:?} is not a number"))
            })
            .collect(),
        _ => CATALOG_D_SIG.to_vec(),
    }
}

fn batch(d_sig: usize) -> BatchLayout {
    BatchLayout::new(d_sig).unwrap_or_else(|| panic!("d_sig {d_sig} is outside the wire range"))
}

// ---------------------------------------------------------------------------
// Planner entry points (copied from Aerie's harness).

/// Pure DP regeneration of one scalar row.
fn regen(key: PolynomialGroupLayout) -> Result<FoldSchedule, akita_error::AkitaError> {
    let key = ScheduleLookupKey::single(key);
    let planned = find_schedule(
        &key,
        Cfg::committed_source_contract()?,
        &[],
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
    )?;
    planned.schedule.validate_structure()?;
    assert_eq!(
        estimate_json(&planned.estimate),
        expanded_estimate_json(&key, &planned.schedule),
        "planner estimate must match expanded schedule sizing for {key:?}",
    );
    Ok(planned.schedule)
}

/// Pure DP regeneration of one grouped row.
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
        expanded_estimate_json(&key, &planned.schedule),
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
fn expanded_estimate_json(key: &ScheduleLookupKey, schedule: &FoldSchedule) -> serde_json::Value {
    serde_json::json!({
        "estimated_proof_payload_bytes": akita_schedules::expanded_schedule_proof_estimate_bytes(
            key, schedule, &policy_of::<Cfg>(),
        ).expect("expanded proof estimate"),
        "estimated_num_setup_field_elements": akita_params::setup_matrix_capacity_for_schedule(schedule)
            .expect("expanded setup capacity").num_field_elements,
    })
}

/// Session cache of exact scalar DP results (Aerie's `SCALAR_PREPLANS`).
static SCALAR_PREPLANS: Mutex<Vec<(PolynomialGroupLayout, FoldSchedule)>> = Mutex::new(Vec::new());

fn cached_scalar(key: PolynomialGroupLayout) -> Option<FoldSchedule> {
    SCALAR_PREPLANS
        .lock()
        .expect("scalar preplan cache poisoned")
        .iter()
        .find(|(cached, _)| *cached == key)
        .map(|(_, schedule)| schedule.clone())
}

fn cache_scalar(key: PolynomialGroupLayout, schedule: FoldSchedule) {
    let mut cache = SCALAR_PREPLANS
        .lock()
        .expect("scalar preplan cache poisoned");
    if !cache.iter().any(|(cached, _)| *cached == key) {
        cache.push((key, schedule));
    }
}

fn preplanned_scalar(key: PolynomialGroupLayout) -> Result<FoldSchedule, akita_error::AkitaError> {
    if let Some(schedule) = cached_scalar(key) {
        return Ok(schedule);
    }
    let started = Instant::now();
    eprintln!("preplanning scalar {key:?}");
    let schedule = regen(key)?;
    eprintln!("planned scalar {key:?} in {:.1?}", started.elapsed());
    cache_scalar(key, schedule.clone());
    Ok(schedule)
}

fn preplanned_scalar_for(
    keys: &[PolynomialGroupLayout],
) -> Vec<(PolynomialGroupLayout, FoldSchedule)> {
    keys.iter()
        .filter_map(|&key| cached_scalar(key).map(|schedule| (key, schedule)))
        .collect()
}

/// A precommitted producer derived from its complete scalar row.
fn precommitted_producer(layout: PolynomialGroupLayout) -> PrecommittedProducer {
    let schedule = preplanned_scalar(layout).unwrap_or_else(|error| {
        panic!("precommit profile planning failed for {layout:?}: {error}")
    });
    let profile = GroupCommitPhaseParams::try_from_params(layout, &schedule.root.params)
        .expect("scalar derived precommit profile");
    PrecommittedProducer::from_config::<Cfg>(profile).expect("precommitted producer")
}

fn grouped_request(layout: &BatchLayout) -> GroupedGenerationRequest {
    GroupedGenerationRequest::new(
        layout.final_group(),
        layout
            .precommitted()
            .iter()
            .map(|&group| precommitted_producer(group))
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

fn write_catalog(rows: Vec<(CommittedGroupBatchProfile, FoldSchedule)>, destination: &Path) {
    let catalog = ValidatedScheduleCatalog::try_new(
        Cfg::schedule_family_name(),
        rows,
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
    )
    .expect("validate catalog");
    std::fs::write(
        destination,
        catalog.to_artifact_bytes().expect("encode catalog"),
    )
    .expect("write catalog");
    eprintln!("wrote {} ({} rows)", destination.display(), catalog.len());
}

fn rows_of(catalog: &ValidatedScheduleCatalog) -> Vec<(CommittedGroupBatchProfile, FoldSchedule)> {
    catalog
        .rows()
        .map(|row| (row.profiles().clone(), row.schedule().clone()))
        .collect()
}

/// The artifact on disk, if any (read directly, so this works with the
/// `schedules` feature off).
fn on_disk_catalog() -> Option<TrustedScheduleCatalog<Cfg>> {
    let bytes = std::fs::read(artifact_path()).ok()?;
    Some(TrustedScheduleCatalog::from_artifact_bytes(&bytes).expect("load on-disk catalog"))
}

fn seed_scalar_preplans(catalog: &ValidatedScheduleCatalog) {
    for row in catalog.rows() {
        if row.profiles().precommitteds.is_empty() {
            cache_scalar(row.profiles().final_group.group, row.schedule().clone());
        }
    }
}

fn row_key(profiles: &CommittedGroupBatchProfile) -> ScheduleLookupKey {
    ScheduleLookupKey {
        final_group: profiles.final_group.group,
        precommitteds: profiles.precommitteds.clone(),
    }
}

/// Whether a catalog row belongs to one of the batch sizes `d_sigs`.
fn row_is_requested(profiles: &CommittedGroupBatchProfile, d_sigs: &[usize]) -> bool {
    if profiles.precommitteds.is_empty() {
        return scalar_layouts(d_sigs).contains(&profiles.final_group.group);
    }
    d_sigs.iter().any(|&d_sig| {
        let layout = batch(d_sig);
        profiles.final_group.group == layout.final_group()
            && profiles
                .precommitteds
                .iter()
                .map(|profile| profile.group)
                .eq(layout.precommitted().iter().copied())
    })
}

fn save(rows: &[(CommittedGroupBatchProfile, FoldSchedule)]) {
    write_catalog(rows.to_vec(), &artifact_path());
}

// ---------------------------------------------------------------------------
// Offline tests (ignored by default).

/// Plan the rows of `AGG_CATALOG_D_SIG` that the on-disk catalog lacks, keep
/// every on-disk row of `CATALOG_D_SIG` or `AGG_CATALOG_D_SIG`, drop the rest,
/// and save after every planned row (so a run killed by a time cap keeps
/// what it finished).
#[test]
#[ignore = "offline incremental catalog update; rewrites src/schedules/"]
fn update_catalog_rows() {
    let d_sigs = requested_d_sigs();
    let mut keep: Vec<usize> = CATALOG_D_SIG
        .iter()
        .copied()
        .chain(d_sigs.iter().copied())
        .collect();
    keep.sort_unstable();
    keep.dedup();
    let mut rows = Vec::new();
    if let Some(catalog) = on_disk_catalog() {
        seed_scalar_preplans(&catalog);
        for row in rows_of(&catalog) {
            if row_is_requested(&row.0, &keep) {
                rows.push(row);
            } else {
                eprintln!("pruned row: final {:?}", row.0.final_group.group);
            }
        }
    }
    for d_sig in d_sigs {
        let layout = batch(d_sig);
        let started = Instant::now();
        let missing: Vec<PolynomialGroupLayout> = layout
            .precommitted()
            .iter()
            .copied()
            .filter(|&group| {
                !rows.iter().any(|(profiles, _)| {
                    profiles.precommitteds.is_empty() && profiles.final_group.group == group
                })
            })
            .collect();
        let workers = offline_planning_worker_count(missing.len());
        let planned = bounded_parallel_filter_map(&missing, workers, |&group| {
            let schedule = preplanned_scalar(group)
                .map_err(|error| format!("scalar planning failed for {group:?}: {error}"))?;
            Ok(Some((group, schedule)))
        })
        .expect("plan scalar rows");
        for (group, schedule) in planned {
            let key = ScheduleLookupKey::single(group);
            let profiles = rendered_profiles(&key, &schedule).expect("scalar profile");
            rows.push((profiles, schedule));
            save(&rows);
        }
        let request = grouped_request(&layout);
        let key = request.key();
        if !rows.iter().any(|(profiles, _)| row_key(profiles) == key) {
            let grouped_started = Instant::now();
            let schedule = regen_group_batch(request).expect("grouped planning");
            eprintln!(
                "planned grouped {} in {:.1?}",
                layout.row_name(),
                grouped_started.elapsed()
            );
            let profiles = rendered_profiles(&key, &schedule).expect("grouped profile");
            rows.push((profiles, schedule));
            save(&rows);
        }
        eprintln!(
            "d_sig {d_sig} ({}): all rows ready in {:.1?}",
            layout.row_name(),
            started.elapsed()
        );
    }
    save(&rows);
}

/// Full regeneration of `CATALOG_D_SIG` through Akita's emit pipeline, as in
/// Aerie's `regenerate`.
#[test]
#[ignore = "offline DP regeneration; rewrites src/schedules/"]
fn regenerate() {
    let keys = scalar_layouts(CATALOG_D_SIG);
    let workers = offline_planning_worker_count(keys.len());
    eprintln!(
        "preplanning {} scalar rows with {workers} workers",
        keys.len()
    );
    bounded_parallel_filter_map(&keys, workers, |&key| {
        preplanned_scalar(key)
            .map(|_| Some(()))
            .map_err(|error| format!("scalar preplanning failed for {key:?}: {error}"))
    })
    .expect("prewarm scalar schedule plans");
    let grouped_requests = CATALOG_D_SIG
        .iter()
        .map(|&d_sig| grouped_request(&batch(d_sig)))
        .collect();
    let spec = EmitSpec {
        family_name: WIDTH4_FAMILY,
        policy: policy_of::<Cfg>(),
        source_contract: Cfg::committed_source_contract().expect("width-4 source contract"),
        preplanned_scalar: preplanned_scalar_for(&keys),
        keys,
        grouped_requests,
        output_dir: output_dir(),
        regen,
        regen_group_batch,
        ring_challenge_config: <Cfg as CommitmentConfig>::ring_challenge_config,
    };
    let outputs = render_schedule_artifact_outputs_with_validation(
        &[spec],
        MaterializationDiagnostics { row_progress: true },
        |_, _| Ok(()),
    )
    .expect("render generated outputs");
    for destination in publish_artifact_outputs(outputs).expect("publish generated outputs") {
        eprintln!("wrote {}", destination.display());
    }
}

/// Drift check: re-plan the rows of `AGG_CATALOG_D_SIG` from scratch and
/// compare them byte-exactly with the on-disk rows. Grouped producers are
/// seeded from the on-disk scalar rows, each of which is itself re-planned.
#[test]
#[ignore = "offline DP drift check; set AGG_CATALOG_D_SIG"]
fn verify_rows() {
    let catalog = on_disk_catalog().expect("on-disk catalog");
    let d_sigs = requested_d_sigs();
    let mut drift = Vec::new();
    for group in scalar_layouts(&d_sigs) {
        let key = ScheduleLookupKey::single(group);
        let fresh = regen(group).expect("fresh scalar planning");
        let row = catalog.resolve_key(&key).expect("checked-in scalar row");
        if serde_json::to_vec(row.schedule()).unwrap() != serde_json::to_vec(&fresh).unwrap() {
            drift.push(format!("scalar {group:?}"));
        }
    }
    seed_scalar_preplans(&catalog);
    for d_sig in d_sigs {
        let request = grouped_request(&batch(d_sig));
        let key = request.key();
        let fresh = regen_group_batch(request).expect("fresh grouped planning");
        let row = catalog.resolve_key(&key).expect("checked-in grouped row");
        let profiles = rendered_profiles(&key, &fresh).expect("profiles");
        if serde_json::to_vec(row.schedule()).unwrap() != serde_json::to_vec(&fresh).unwrap()
            || serde_json::to_vec(row.profiles()).unwrap() != serde_json::to_vec(&profiles).unwrap()
        {
            drift.push(format!("grouped d_sig {d_sig}"));
        }
    }
    assert!(drift.is_empty(), "catalog drift: {drift:?}");
}

/// A planning-only preset of signed width `W` with Dense's inner-basis search
/// (log basis 3 to 16), otherwise identical to the width-4 preset. It has no
/// catalog; it exists to ask the planner which inner digits it selects.
#[derive(Clone, Copy, Debug, Default)]
struct WidthProbe<const W: u32>;

impl<const W: u32> CommitmentConfig for WidthProbe<W> {
    type Field = <Cfg as CommitmentConfig>::Field;
    type ExtField = <Cfg as CommitmentConfig>::ExtField;
    const RING_DIMENSION_SCHEDULE: akita_config::RingDimensionSchedule =
        <Cfg as CommitmentConfig>::RING_DIMENSION_SCHEDULE;
    fn decomposition() -> akita_params::DecompositionParams {
        akita_params::DecompositionParams {
            log_commit_bound: W,
            ..Cfg::decomposition()
        }
    }
    fn ring_challenge_config(
        d: usize,
    ) -> Result<akita_challenges::SparseChallengeConfig, akita_error::AkitaError> {
        Cfg::ring_challenge_config(d)
    }
    fn sis_modulus_profile() -> akita_params::SisModulusProfileId {
        Cfg::sis_modulus_profile()
    }
    fn opening_basis_range() -> (u32, u32) {
        Cfg::opening_basis_range()
    }
    fn inner_basis_range() -> (u32, u32) {
        akita_config::proof_optimized::fp64::Dense::inner_basis_range()
    }
    fn committed_source_class() -> akita_params::sis::CommittedSourceClass {
        Cfg::committed_source_class()
    }
    fn schedule_family_name() -> &'static str {
        "agg_mldsa_width_probe"
    }
}

fn probe<const W: u32>(layout: PolynomialGroupLayout) -> Result<(u32, usize, u64), String> {
    let key = ScheduleLookupKey::single(layout);
    let started = Instant::now();
    let planned = find_schedule(
        &key,
        WidthProbe::<W>::committed_source_contract().map_err(|error| error.to_string())?,
        &[],
        &policy_of::<WidthProbe<W>>(),
        WidthProbe::<W>::ring_challenge_config,
    )
    .map_err(|error| error.to_string())?;
    let profile = GroupCommitPhaseParams::try_from_params(layout, &planned.schedule.root.params)
        .map_err(|error| error.to_string())?;
    let bytes = planned
        .estimate
        .estimated_proof_payload_bytes()
        .map_err(|error| error.to_string())?;
    eprintln!(
        "width {W:>2} at {layout:?}: inner log basis {}, {} digit(s), est. proof {bytes} B, planned in {:.1?}",
        profile.inner.digits.log_basis,
        profile.inner.digits.num_digits,
        started.elapsed()
    );
    Ok((
        profile.inner.digits.log_basis,
        profile.inner.digits.num_digits,
        bytes as u64,
    ))
}

/// Ask the planner which inner digits a raw signed width selects (risks.md R1).
/// Widths up to 16 fit one digit; 17 and more need at least two.
#[test]
#[ignore = "offline planner probe (minutes)"]
fn width_admissibility_probe() {
    let layout = PolynomialGroupLayout::new(13, 1);
    let check = |width: u32, result: Result<(u32, usize, u64), String>| {
        let (log_basis, digits, _) =
            result.unwrap_or_else(|error| panic!("width {width}: {error}"));
        let minimum = agg_pcs_config::preset::min_inner_digits(width)
            .expect("valid width")
            .expect("admissible width");
        assert!(digits >= minimum, "width {width}: {digits} < {minimum}");
        assert!(u64::from(log_basis) * digits as u64 >= u64::from(width));
        digits
    };
    check(4, probe::<4>(layout));
    check(14, probe::<14>(layout));
    check(16, probe::<16>(layout));
    check(17, probe::<17>(layout));
    for (width, digits) in [
        (18, check(18, probe::<18>(layout))),
        (19, check(19, probe::<19>(layout))),
        (20, check(20, probe::<20>(layout))),
    ] {
        assert!(digits >= 2, "width {width} selected {digits} digit(s)");
    }
}

// ---------------------------------------------------------------------------
// Cheap checks of the checked-in artifact (run by `cargo test`).

#[cfg(feature = "schedules")]
fn requested_keys() -> Vec<ScheduleLookupKey> {
    let mut keys: Vec<ScheduleLookupKey> = scalar_layouts(CATALOG_D_SIG)
        .into_iter()
        .map(ScheduleLookupKey::single)
        .collect();
    for &d_sig in CATALOG_D_SIG {
        let layout = batch(d_sig);
        keys.push(ScheduleLookupKey {
            final_group: layout.final_group(),
            precommitteds: layout
                .precommitted()
                .iter()
                .map(|&group| Cfg::commitment_profile(group).expect("scalar producer row"))
                .collect(),
        });
    }
    keys
}

/// The artifact holds exactly the requested rows, every group commits with
/// one balanced base-16 inner digit, no row needs a setup prefix, and the
/// re-audited rows re-encode to the checked-in bytes.
#[cfg(feature = "schedules")]
#[test]
fn checked_in_catalog_holds_exactly_the_requested_rows() {
    let catalog = Cfg::catalog().expect("checked-in catalog");
    let keys = requested_keys();
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
        "catalog contains unrequested rows"
    );
    for row in catalog.rows() {
        let profiles = row.profiles();
        for profile in std::iter::once(&profiles.final_group).chain(&profiles.precommitteds) {
            assert_eq!(
                (
                    profile.inner.digits.log_basis,
                    profile.inner.digits.num_digits
                ),
                (4, 1),
                "group {:?}",
                profile.group
            );
        }
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
        Cfg::schedule_family_name(),
        rows_of(catalog),
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
    )
    .expect("re-audit rows");
    let on_disk = std::fs::read(artifact_path()).expect("artifact");
    assert_eq!(audited.to_artifact_bytes().expect("encode"), on_disk);
}
