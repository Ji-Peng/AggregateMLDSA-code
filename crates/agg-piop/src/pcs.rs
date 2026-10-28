//! Akita commitments of the six groups and their batched opening (events 1, 4 and 13).
//!
//! Mirrors `aerie-prover/src/protocol/pcs/group.rs`: narrow signed tables stay `i8` (or
//! `i32` for the 16-bit lookup columns) until commitment, and each polynomial is
//! materialized polynomial-major, zero-extended to the physical (PCS floor) length.
//! Commitments go through `agg_pcs_config::commit::PcsProver` (round 1 `G_main16` and
//! `G_main4`; round 2 `G_1`, `G_2`, `G_3`, then the final group `G_m`), each call
//! transcript-free and under its group's family; the caller absorbs
//! [`GroupCommitment::serialized`] in event order. One batched opening covers the 38
//! claims: 31 at the suffixes of `r^(3)` and the 7 multiplicity digits at the lookup's
//! table point.

use std::sync::OnceLock;

use agg_pcs_config::commit::{
    evaluate_signed_table, CommitError, GroupClaim, GroupCommitment, PcsProver, PcsVerifier,
};
use agg_pcs_config::work_plan::{PlanError, WorkPlan};
use akita_error::AkitaError;
use akita_pcs::DensePoly;
use jolt_field::{Ring, Zero};
use rayon::prelude::*;

use crate::manifest::{CommitmentManifest, GroupRole, GROUP_COUNT};
use crate::source_tables::{BridgeGroupTables, Cells, MainTables, SourceTables};
use crate::{BaseField, OpeningField};

/// Cells converted per parallel task.
const FIELD_CONVERSION_CHUNK: usize = 1 << 12;

/// A commitment or opening failure.
#[derive(Debug)]
pub enum PcsError {
    /// No work plan for the batch size.
    Plan(PlanError),
    /// The work plan's shapes differ from the manifest's.
    ManifestMismatch,
    /// A point has the wrong number of coordinates for its group.
    PointShape(GroupRole),
    /// The commit driver or Akita failed.
    Commit(CommitError),
}

impl std::fmt::Display for PcsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Plan(error) => write!(f, "work plan: {error}"),
            Self::ManifestMismatch => write!(f, "work plan and manifest disagree"),
            Self::PointShape(group) => write!(f, "{group:?}: point of the wrong length"),
            Self::Commit(error) => write!(f, "commit: {error}"),
        }
    }
}

impl std::error::Error for PcsError {}

impl From<CommitError> for PcsError {
    fn from(error: CommitError) -> Self {
        Self::Commit(error)
    }
}

impl From<AkitaError> for PcsError {
    fn from(error: AkitaError) -> Self {
        Self::Commit(CommitError::Akita(error))
    }
}

impl From<PlanError> for PcsError {
    fn from(error: PlanError) -> Self {
        Self::Plan(error)
    }
}

static I8_FIELD_VALUES: OnceLock<[BaseField; 256]> = OnceLock::new();

fn i8_field_values() -> &'static [BaseField; 256] {
    I8_FIELD_VALUES.get_or_init(|| {
        std::array::from_fn(|byte| BaseField::from_i64(i64::from(byte as u8 as i8)))
    })
}

/// Materialize one Akita polynomial per logical table (`i8` or `i32` cells), in group
/// order, each zero-extended to `2^physical_vars` evaluations (index bits least significant
/// first, so the pad variables are trailing).
pub fn group_polynomials(
    columns: &[Cells<'_>],
    physical_vars: usize,
) -> Result<Vec<DensePoly<BaseField>>, AkitaError> {
    let physical_len = 1_usize << physical_vars;
    let values = i8_field_values();
    columns
        .par_iter()
        .map(|cells| {
            if cells.len() > physical_len {
                return Err(AkitaError::InvalidSize {
                    expected: physical_len,
                    actual: cells.len(),
                });
            }
            let mut evaluations = vec![BaseField::zero(); physical_len];
            match cells {
                Cells::Narrow(cells) => evaluations[..cells.len()]
                    .par_chunks_mut(FIELD_CONVERSION_CHUNK)
                    .zip(cells.par_chunks(FIELD_CONVERSION_CHUNK))
                    .for_each(|(output, input)| {
                        for (value, &cell) in output.iter_mut().zip(input) {
                            *value = values[usize::from(cell as u8)];
                        }
                    }),
                Cells::Wide(cells) => evaluations[..cells.len()]
                    .par_chunks_mut(FIELD_CONVERSION_CHUNK)
                    .zip(cells.par_chunks(FIELD_CONVERSION_CHUNK))
                    .for_each(|(output, input)| {
                        for (value, &cell) in output.iter_mut().zip(input) {
                            *value = BaseField::from_i64(i64::from(cell));
                        }
                    }),
            }
            DensePoly::from_field_evals(physical_vars, evaluations)
        })
        .collect()
}

/// Multilinear extension of a logical `i8` table at `point` (least significant variable
/// first): the value an opening claim on that table carries.
#[must_use]
pub fn evaluate_column(cells: &[i8], point: &[OpeningField]) -> OpeningField {
    evaluate_signed_table(cells, point)
}

/// The group points: every group but `G_m` at the suffix of the main-domain point `r3`,
/// `G_m` at the lookup's table point `pi_m`.
pub fn group_points(
    manifest: &CommitmentManifest,
    r3: &[OpeningField],
    pi_m: &[OpeningField],
) -> Result<[Vec<OpeningField>; GROUP_COUNT], PcsError> {
    if r3.len() != manifest.d_main() {
        return Err(PcsError::PointShape(GroupRole::Main));
    }
    if pi_m.len() != manifest.group(GroupRole::Mult).logical_vars {
        return Err(PcsError::PointShape(GroupRole::Mult));
    }
    Ok(manifest.groups().clone().map(|group| {
        if group.role == GroupRole::Mult {
            pi_m.to_vec()
        } else {
            r3[r3.len() - group.logical_vars..].to_vec()
        }
    }))
}

/// The opening claims of every committed polynomial at its group's logical point.
pub fn opening_claims(
    tables: SourceTables<'_>,
    points: &[Vec<OpeningField>; GROUP_COUNT],
) -> Result<[GroupClaim; GROUP_COUNT], PcsError> {
    for group in tables.manifest.groups() {
        if points[group.role.index()].len() != group.logical_vars {
            return Err(PcsError::PointShape(group.role));
        }
    }
    Ok(GroupRole::ALL.map(|role| {
        let point = &points[role.index()];
        let values = tables
            .group(role)
            .par_iter()
            .map(|cells| cells.evaluate(point))
            .collect();
        GroupClaim {
            point: point.clone(),
            values,
        }
    }))
}

/// Stack size of a commit thread (Akita's commit runs on it, its parallel loops on the
/// rayon pool) and of the thread [`overlap`] starts.
pub(crate) const COMMIT_THREAD_STACK: usize = 64 << 20;

/// Whether concurrent work must run on the calling thread instead of extra OS threads.
///
/// A rayon worker must never block on an OS thread: work that thread hands to the pool may
/// need the very worker that waits for it, so with one thread (`taskset -c 2`,
/// `RAYON_NUM_THREADS=1`) the pool deadlocks, and with more it loses a worker for the whole
/// wait. A caller outside the pool whose pool has one thread gains nothing from extra
/// threads either.
pub(crate) fn on_calling_thread() -> bool {
    rayon::current_thread_index().is_some() || rayon::current_num_threads() <= 1
}

/// Join an OS thread, re-raising its panic on the caller.
pub(crate) fn join_os_thread<T>(handle: std::thread::ScopedJoinHandle<'_, T>) -> T {
    handle
        .join()
        .unwrap_or_else(|payload| std::panic::resume_unwind(payload))
}

/// Run Akita commits concurrently, each on its own scoped OS thread, never as rayon jobs.
///
/// Akita's CPU backend builds its NTT slots lazily inside `OnceLock::get_or_init` with
/// rayon-parallel loops (`akita-cpu-backend` `arithmetic/prepared.rs`). Two commits joined
/// on the rayon pool can deadlock there: the worker that initializes a slot steals the other
/// commit while it waits for its own loop, and that commit re-enters the same `OnceLock` on
/// the same thread. A commit on a plain OS thread blocks on the `OnceLock` instead, and its
/// parallel loops still run on the pool. On a rayon worker, or with a one-thread pool, the
/// commits run one after another on the calling thread ([`on_calling_thread`]); one commit
/// at a time cannot re-enter a `OnceLock`.
fn concurrent_commits<const K: usize>(
    commits: [&(dyn Fn() -> Result<GroupCommitment, PcsError> + Sync); K],
) -> [Result<GroupCommitment, PcsError>; K] {
    if on_calling_thread() {
        return commits.map(|commit| commit());
    }
    std::thread::scope(|scope| {
        let handles = commits.map(|commit| {
            std::thread::Builder::new()
                .name("akita-commit".into())
                .stack_size(COMMIT_THREAD_STACK)
                .spawn_scoped(scope, commit)
                .expect("spawn a commit thread")
        });
        handles.map(join_os_thread)
    })
}

/// Run `first` on the calling thread and `second` concurrently with it; return both
/// results.
///
/// The prover overlaps its round-1 Akita commits with the Keccak side's commitments. As a
/// rayon job the commit side would block a worker on its commit threads
/// ([`concurrent_commits`]), so outside the pool `second` gets a scoped OS thread and
/// `first` (with its commit threads) stays on the caller. On a rayon worker this is
/// `rayon::join` (the commits inside then run one after another); with a one-thread pool
/// the two run one after another on the caller.
pub(crate) fn overlap<A, B>(
    first: impl FnOnce() -> A + Send,
    second: impl FnOnce() -> B + Send,
) -> (A, B)
where
    A: Send,
    B: Send,
{
    if rayon::current_thread_index().is_some() {
        return rayon::join(first, second);
    }
    if on_calling_thread() {
        return (first(), second());
    }
    std::thread::scope(|scope| {
        let handle = std::thread::Builder::new()
            .name("agg-overlap".into())
            .stack_size(COMMIT_THREAD_STACK)
            .spawn_scoped(scope, second)
            .expect("spawn an overlap thread");
        let a = first();
        (a, join_os_thread(handle))
    })
}

/// The prover side: the manifest, its work plan and the Akita prover setup.
pub struct TableCommitter {
    manifest: CommitmentManifest,
    prover: PcsProver,
}

impl TableCommitter {
    /// Plan the batch of `manifest` and allocate the seed-derived setup.
    pub fn new(manifest: CommitmentManifest) -> Result<Self, PcsError> {
        let plan = WorkPlan::for_records(manifest.records(), manifest.relation())?;
        if !manifest.matches_plan(&plan) {
            return Err(PcsError::ManifestMismatch);
        }
        Ok(Self {
            manifest,
            prover: PcsProver::new(plan)?,
        })
    }

    /// The manifest.
    #[must_use]
    pub const fn manifest(&self) -> &CommitmentManifest {
        &self.manifest
    }

    /// The underlying commit driver.
    #[must_use]
    pub const fn prover(&self) -> &PcsProver {
        &self.prover
    }

    fn commit(&self, role: GroupRole, columns: &[Cells<'_>]) -> Result<GroupCommitment, PcsError> {
        let descriptor = self.manifest.group(role);
        if columns.len() != descriptor.polynomial_count() {
            return Err(CommitError::PolynomialCount {
                group: role,
                expected: descriptor.polynomial_count(),
                actual: columns.len(),
            }
            .into());
        }
        if let Some(cells) = columns
            .iter()
            .find(|cells| cells.len() != descriptor.logical_cells())
        {
            return Err(CommitError::TableLength {
                group: role,
                expected: descriptor.logical_cells(),
                actual: cells.len(),
            }
            .into());
        }
        let polynomials = group_polynomials(columns, descriptor.physical_vars)?;
        Ok(self.prover.commit_polynomials(role, polynomials)?)
    }

    /// Round 1 (event 1, before the bridge seed): `G_main16` and `G_main4`, concurrently
    /// (each commit is transcript-free; see [`concurrent_commits`]). Call it from outside
    /// the rayon pool to get the concurrency; on a worker the two commit in turn.
    pub fn commit_round1(&self, main: &MainTables) -> Result<[GroupCommitment; 2], PcsError> {
        let narrow: Vec<Cells<'_>> = main.columns().iter().map(|c| Cells::Narrow(c)).collect();
        let wide = main.wide_columns();
        let [wide, narrow] = concurrent_commits([
            &|| self.commit(GroupRole::Main16, &wide),
            &|| self.commit(GroupRole::Main, &narrow),
        ]);
        Ok([wide?, narrow?])
    }

    /// Round 2 (event 4, after the bridge seed): `G_1`, `G_2`, `G_3` and the final group
    /// `G_m`. The four commits are independent (the final group is committed against the
    /// precommitted profiles of the work plan, not against the other commitments), so they
    /// run in parallel; an Akita commit is memory bound and does not use every thread on
    /// its own. The caller absorbs them in wire order.
    pub fn commit_round2(&self, bridge: &BridgeGroupTables) -> Result<[GroupCommitment; 4], PcsError> {
        let commit = |role: GroupRole| self.commit(role, &bridge.group(&self.manifest, role));
        let [level1, level2, level3, mult] = concurrent_commits([
            &|| commit(GroupRole::Level1),
            &|| commit(GroupRole::Level2),
            &|| commit(GroupRole::Level3),
            &|| commit(GroupRole::Mult),
        ]);
        Ok([level1?, level2?, level3?, mult?])
    }

    /// The batched opening of all six groups at `points` under the 32-byte PIOP session
    /// digest; returns the proof and the claims it authenticates (evaluated from the
    /// tables, one streaming pass per polynomial).
    pub fn open(
        &self,
        commitments: [GroupCommitment; GROUP_COUNT],
        tables: SourceTables<'_>,
        points: &[Vec<OpeningField>; GROUP_COUNT],
        session: &[u8; 32],
    ) -> Result<(Vec<u8>, [GroupClaim; GROUP_COUNT]), PcsError> {
        let claims = opening_claims(tables, points)?;
        let proof = self.open_claims(commitments, &claims, session)?;
        Ok((proof, claims))
    }

    /// The batched opening of all six groups for claims the caller already holds (the
    /// prover's stage-3 terminal values and multiplicity-digit claims, which equal the
    /// table evaluations).
    pub fn open_claims(
        &self,
        commitments: [GroupCommitment; GROUP_COUNT],
        claims: &[GroupClaim; GROUP_COUNT],
        session: &[u8; 32],
    ) -> Result<Vec<u8>, PcsError> {
        for (group, claim) in self.manifest.groups().iter().zip(claims) {
            if claim.point.len() != group.logical_vars {
                return Err(PcsError::PointShape(group.role));
            }
        }
        Ok(self.prover.open(commitments, claims, session)?)
    }

    /// The verifier derived from this prover's setup.
    pub fn verifier(&self) -> Result<PcsVerifier, PcsError> {
        Ok(self.prover.verifier()?)
    }
}

/// The verifier of the batched opening, built from the public setup seed alone.
pub fn verifier_for(manifest: &CommitmentManifest) -> Result<PcsVerifier, PcsError> {
    let plan = WorkPlan::for_records(manifest.records(), manifest.relation())?;
    if !manifest.matches_plan(&plan) {
        return Err(PcsError::ManifestMismatch);
    }
    Ok(PcsVerifier::new(plan)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::ExtField;

    #[test]
    fn group_polynomials_zero_extend_and_convert_exactly() {
        let narrow: Vec<i8> = (0..4096).map(|i| (i % 16) as i8 - 8).collect();
        let wide: Vec<i32> = (0..4096).map(|i| (i * 17) % 65_536 - 32_768).collect();
        let ones = vec![1_i8; 4096];
        let columns = [Cells::Narrow(&narrow), Cells::Wide(&wide), Cells::Narrow(&ones)];
        let polynomials = group_polynomials(&columns, 13).unwrap();
        assert_eq!(polynomials.len(), 3);
        for (poly, cells) in polynomials.iter().zip(&columns) {
            let coeffs = poly.field_coeffs();
            assert!(coeffs.len() >= 1 << 13);
            for (index, coeff) in coeffs.iter().enumerate() {
                let expected = if index < cells.len() { cells.value(index) } else { 0 };
                assert_eq!(*coeff, BaseField::from_i64(expected), "cell {index}");
            }
        }
        let long = vec![0_i8; 1 << 14];
        assert!(group_polynomials(&[Cells::Narrow(&long)], 13).is_err());
    }

    #[test]
    fn group_points_follow_the_group_domains() {
        let manifest = CommitmentManifest::for_batch(1 << 15).unwrap();
        let point: Vec<OpeningField> = (0..25)
            .map(|i| OpeningField::lift_base(BaseField::from_u64(i)))
            .collect();
        let pi_m: Vec<OpeningField> = (100..116)
            .map(|i| OpeningField::lift_base(BaseField::from_u64(i)))
            .collect();
        let points = group_points(&manifest, &point, &pi_m).unwrap();
        let lengths = points.each_ref().map(Vec::len);
        assert_eq!(lengths, [25, 25, 19, 15, 14, 16]);
        assert_eq!(points[4][0], point[11]);
        assert_eq!(points[5], pi_m);
        assert!(group_points(&manifest, &point[1..], &pi_m).is_err());
        assert!(group_points(&manifest, &point, &pi_m[1..]).is_err());
    }
}
