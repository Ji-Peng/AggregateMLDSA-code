//! Akita commitments of the seven groups and their batched opening (the round-1 and round-2 commitments and the Akita opening).
//!
//! Mirrors `aerie-prover/src/protocol/pcs/group.rs`: narrow signed tables stay `i8` (or
//! `i32` for the 16-bit lookup columns) until commitment, and each polynomial is
//! materialized polynomial-major, zero-extended to the physical (PCS floor) length.
//! Commitments go through `agg_pcs_config::commit::PcsProver` (round 1 `G_main16` and
//! `G_main4`; round 2 `G_1`, `G_2`, `G_3`, `G_k`, then the final group `G_m`), each call
//! transcript-free and under its group's family; the caller absorbs
//! [`GroupCommitment::serialized`] in event order. One batched opening covers the stage-3
//! claims at the suffixes of `r^(3)` (40 for ML-DSA-44 under H-FIPS), the 12 zero polynomials
//! of `G_3` (claim zero) and the 7 multiplicity digits at the lookup's table point: 59 claims.

use agg_pcs_config::commit::{
    CommitError, GroupClaim, GroupCommitment, PcsProver, PcsVerifier, PreparedOpening,
};
use agg_pcs_config::work_plan::{PlanError, WorkPlan};
use akita_error::AkitaError;
use akita_pcs::DensePoly;
use rayon::prelude::*;

use crate::manifest::{
    CommitmentManifest, GroupRole, GROUP_COUNT, GROUP_COUNT_FP, KECCAK_GROUP_COUNT,
};
use crate::source_tables::{BridgeGroupTables, Cells, MainTables, SourceTables};
use crate::{BaseField, OpeningField};

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

/// Ring dimension and inner decomposition of one dense group's commit.
#[derive(Clone, Copy, Debug)]
pub struct DenseShape {
    /// Commitment ring dimension.
    pub ring_dimension: usize,
    /// Balanced digits per coefficient in the A-side decomposition.
    pub num_digits: usize,
    /// Logarithm of the balanced digit basis.
    pub log_basis: u32,
}

/// Materialize one Akita polynomial per logical table (`i8` or `i32` cells), in group
/// order, each zero-extended to `2^physical_vars` evaluations (index bits least significant
/// first, so the pad variables are trailing).
///
/// With `shape`, each polynomial also receives the digit cache of the commit's
/// inner decomposition. Both cell widths keep only their signed mirror; the
/// canonical field plane is expanded on the first reader that asks for it.
pub fn group_polynomials(
    columns: &[Cells<'_>],
    physical_vars: usize,
    shape: Option<DenseShape>,
) -> Result<Vec<DensePoly<BaseField>>, AkitaError> {
    let physical_len = 1_usize << physical_vars;
    columns
        .par_iter()
        .map(|cells| {
            if cells.len() > physical_len {
                return Err(AkitaError::InvalidSize {
                    expected: physical_len,
                    actual: cells.len(),
                });
            }
            let polynomial = match cells {
                Cells::Narrow(cells) => DensePoly::from_small_i8_cells(physical_vars, cells)?,
                Cells::Wide(cells) => DensePoly::from_small_i16_cells(physical_vars, cells)?,
            };
            if let Some(shape) = shape {
                polynomial.prime_digit_cache(
                    shape.ring_dimension,
                    shape.num_digits,
                    shape.log_basis,
                )?;
            }
            Ok(polynomial)
        })
        .collect()
}

/// The group points under `HashField::Gf2`: every group but `G_m` at the suffix of the
/// main-domain point `r3`, `G_m` at the lookup's table point `pi_m`.
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
    Ok(GroupRole::ALL.map(|role| {
        let group = manifest.group(role);
        if role == GroupRole::Mult {
            pi_m.to_vec()
        } else {
            r3[r3.len() - group.logical_vars..].to_vec()
        }
    }))
}

/// The group points under `HashField::Fp`: every group but `G_m` at the suffix of the
/// stage-3 point `r3` (`d_stage = max(d_main, d_K)` coordinates), `G_m` at the lookup's
/// table point `pi_m`.
pub fn group_points_fp(
    manifest: &CommitmentManifest,
    r3: &[OpeningField],
    pi_m: &[OpeningField],
) -> Result<[Vec<OpeningField>; GROUP_COUNT_FP], PcsError> {
    let d_stage = manifest.d_main().max(manifest.d_keccak());
    if r3.len() != d_stage {
        return Err(PcsError::PointShape(GroupRole::Main));
    }
    if pi_m.len() != manifest.group(GroupRole::Mult).logical_vars {
        return Err(PcsError::PointShape(GroupRole::Mult));
    }
    Ok(GroupRole::ALL_FP.map(|role| {
        let group = manifest.group(role);
        if role == GroupRole::Mult {
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

/// Default concurrency cap for committing the seven `HashField::Fp` Keccak groups.
pub const DEFAULT_COMMIT_PARALLEL: usize = 2;

/// Maximum number of Keccak groups committed concurrently (`AGG_COMMIT_PARALLEL`,
/// default [`DEFAULT_COMMIT_PARALLEL`], at least 1).
#[must_use]
pub fn commit_parallel_limit() -> usize {
    std::env::var("AGG_COMMIT_PARALLEL")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_COMMIT_PARALLEL)
        .max(1)
}

/// Run up to `limit` Akita commits concurrently on scoped OS threads, never as rayon jobs.
fn bounded_concurrent_commits<const K: usize>(
    limit: usize,
    commits: [&(dyn Fn() -> Result<GroupCommitment, PcsError> + Sync); K],
) -> [Result<GroupCommitment, PcsError>; K] {
    if on_calling_thread() || limit <= 1 || K <= 1 {
        return commits.map(|commit| commit());
    }
    let workers = limit.min(K);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let slots: [std::sync::OnceLock<Result<GroupCommitment, PcsError>>; K] =
        std::array::from_fn(|_| std::sync::OnceLock::new());
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            let next_ref = &next;
            let slots_ref = &slots;
            let commits_ref = &commits;
            handles.push(
                std::thread::Builder::new()
                    .name("akita-commit".into())
                    .stack_size(COMMIT_THREAD_STACK)
                    .spawn_scoped(scope, move || loop {
                        let idx = next_ref.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if idx >= K {
                            break;
                        }
                        let _ = slots_ref[idx].set(commits_ref[idx]());
                    })
                    .expect("spawn a commit thread"),
            );
        }
        for handle in handles {
            join_os_thread(handle);
        }
    });
    slots.map(| mut slot| slot.take().expect("every commit slot filled"))
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
    bounded_concurrent_commits(K, commits)
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
        let plan = WorkPlan::for_hash_field(
            manifest.set(),
            manifest.records(),
            manifest.relation(),
            manifest.hash_field(),
        )?;
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

    /// Warm every group's shared NTT slots on the calling thread.
    ///
    /// The slots are otherwise built lazily by the first commit that requests
    /// them; a caller with spare time before the commit stages (the setup
    /// thread) can run those builds there.
    pub fn warm(&self) -> Result<(), PcsError> {
        self.prover.prewarm_groups()?;
        Ok(())
    }

    /// Commit one group's logical `Cells` columns under its work-plan profile.
    pub fn commit_group_cells(
        &self,
        role: GroupRole,
        columns: &[Cells<'_>],
    ) -> Result<GroupCommitment, PcsError> {
        self.commit(role, columns)
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
        let polynomials =
            group_polynomials(columns, descriptor.physical_vars, Some(self.dense_shape(role)))?;
        Ok(self.prover.commit_polynomials(role, polynomials)?)
    }

    /// The inner decomposition of `role`'s work-plan profile.
    fn dense_shape(&self, role: GroupRole) -> DenseShape {
        let profile = self.prover.plan().profile(role);
        DenseShape {
            ring_dimension: profile.inner.matrix.ring_dimension(),
            num_digits: profile.inner.digits.num_digits,
            log_basis: profile.inner.digits.log_basis,
        }
    }

    /// Round 1 (before the bridge seed): `G_main16` and `G_main4`, concurrently
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

    /// Commit the seven `HashField::Fp` Keccak groups (`KeccakA(0..4)`, `KeccakC`,
    /// `KeccakCPrime`) from the packed [`agg_fpkeccak::KeccakTables`], at most
    /// `AGG_COMMIT_PARALLEL` (default 2) groups at a time.
    pub fn commit_keccak(
        &self,
        keccak: &agg_fpkeccak::KeccakTables,
    ) -> Result<[GroupCommitment; KECCAK_GROUP_COUNT], PcsError> {
        self.commit_keccak_bounded(keccak, commit_parallel_limit())
    }

    /// [`Self::commit_keccak`] with at most `limit` groups in flight.
    ///
    /// Each polynomial is expanded from the packed words straight into the `i8` buffer of its
    /// zero-extended length, and that buffer becomes the polynomial's signed mirror (no copy):
    /// a group's only per-cell image is the one the commitment handle keeps for the opening
    /// (one byte per cell). The commitments equal those of [`Self::commit_group_cells`] over
    /// `expand_i8` columns.
    pub fn commit_keccak_bounded(
        &self,
        keccak: &agg_fpkeccak::KeccakTables,
        limit: usize,
    ) -> Result<[GroupCommitment; KECCAK_GROUP_COUNT], PcsError> {
        let commit_one = |group_idx: usize| -> Result<GroupCommitment, PcsError> {
            let role = GroupRole::KECCAK[group_idx];
            let descriptor = self.manifest.group(role);
            let polys = keccak.layout().group_poly_range(group_idx);
            if polys.len() != descriptor.polynomial_count() {
                return Err(CommitError::PolynomialCount {
                    group: role,
                    expected: descriptor.polynomial_count(),
                    actual: polys.len(),
                }
                .into());
            }
            let cells = agg_fpkeccak::RECORD_CELLS_PER_POLY * keccak.n_star();
            let physical_vars = descriptor.physical_vars;
            if cells != descriptor.logical_cells() || cells > 1 << physical_vars {
                return Err(CommitError::TableLength {
                    group: role,
                    expected: descriptor.logical_cells(),
                    actual: cells,
                }
                .into());
            }
            let shape = self.dense_shape(role);
            let polynomials = polys
                .into_par_iter()
                .map(|poly| {
                    // Zeroed: fresh zero pages, faulted in by this task's writes.
                    let mut mirror = vec![0_i8; 1 << physical_vars];
                    keccak.expand_poly_i8_into(poly, &mut mirror);
                    let polynomial = DensePoly::from_small_i8(physical_vars, mirror)?;
                    polynomial.prime_digit_cache(
                        shape.ring_dimension,
                        shape.num_digits,
                        shape.log_basis,
                    )?;
                    Ok(polynomial)
                })
                .collect::<Result<Vec<_>, AkitaError>>()?;
            Ok(self.prover.commit_polynomials(role, polynomials)?)
        };
        let [ka0, ka1, ka2, ka3, ka4, kc, kcp] = bounded_concurrent_commits(
            limit,
            [
                &|| commit_one(0),
                &|| commit_one(1),
                &|| commit_one(2),
                &|| commit_one(3),
                &|| commit_one(4),
                &|| commit_one(5),
                &|| commit_one(6),
            ],
        );
        Ok([ka0?, ka1?, ka2?, ka3?, ka4?, kc?, kcp?])
    }

    /// Round 2 under `HashField::Fp`: `G_k` (`GroupRole::Proj`) and `G_m` (`GroupRole::Mult`),
    /// committed concurrently.
    pub fn commit_round2_fp(
        &self,
        proj: &[Cells<'_>],
        mult: &[Cells<'_>],
    ) -> Result<[GroupCommitment; 2], PcsError> {
        let [proj, mult] = concurrent_commits([
            &|| self.commit(GroupRole::Proj, proj),
            &|| self.commit(GroupRole::Mult, mult),
        ]);
        Ok([proj?, mult?])
    }

    /// Round 2 (after the bridge seed): `G_1`, `G_2`, `G_3`, `G_k` and the final
    /// group `G_m`. The five commits are independent (the final group is committed against the
    /// precommitted profiles of the work plan, not against the other commitments), so they
    /// run in parallel; an Akita commit is memory bound and does not use every thread on
    /// its own. The caller absorbs them in wire order.
    pub fn commit_round2(&self, bridge: &BridgeGroupTables) -> Result<[GroupCommitment; 5], PcsError> {
        let commit = |role: GroupRole| self.commit(role, &bridge.group(&self.manifest, role));
        let [level1, level2, level3, proj, mult] = concurrent_commits([
            &|| commit(GroupRole::Level1),
            &|| commit(GroupRole::Level2),
            &|| commit(GroupRole::Level3),
            &|| commit(GroupRole::Proj),
            &|| commit(GroupRole::Mult),
        ]);
        Ok([level1?, level2?, level3?, proj?, mult?])
    }

    /// The batched opening of all seven groups at `points` under the 32-byte PIOP session
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

    /// The batched opening of all groups of the manifest for claims the caller already
    /// holds (the prover's stage-3 terminal values and multiplicity-digit claims, which
    /// equal the table evaluations).
    pub fn open_claims(
        &self,
        commitments: impl Into<Vec<GroupCommitment>>,
        claims: &[GroupClaim],
        session: &[u8; 32],
    ) -> Result<Vec<u8>, PcsError> {
        let prepared = self.prepare_open_claims(commitments, claims)?;
        self.finish_open_claims(prepared, session)
    }

    /// Compute the part of the opening that depends only on the commitments and
    /// the opening points. It holds no transcript session.
    pub fn prepare_open_claims(
        &self,
        commitments: impl Into<Vec<GroupCommitment>>,
        claims: &[GroupClaim],
    ) -> Result<PreparedOpening, PcsError> {
        if claims.len() != self.manifest.groups().len() {
            return Err(PcsError::PointShape(GroupRole::Mult));
        }
        for (group, claim) in self.manifest.groups().iter().zip(claims) {
            if claim.point.len() != group.logical_vars {
                return Err(PcsError::PointShape(group.role));
            }
        }
        Ok(self.prover.prepare_open(commitments, claims)?)
    }

    /// Bind a prepared opening to the 32-byte PIOP session digest.
    pub fn finish_open_claims(
        &self,
        prepared: PreparedOpening,
        session: &[u8; 32],
    ) -> Result<Vec<u8>, PcsError> {
        Ok(self.prover.finish_open(prepared, session)?)
    }

    /// The verifier derived from this prover's setup.
    pub fn verifier(&self) -> Result<PcsVerifier, PcsError> {
        Ok(self.prover.verifier()?)
    }
}

/// The verifier of the batched opening, built from the public setup seed alone.
pub fn verifier_for(manifest: &CommitmentManifest) -> Result<PcsVerifier, PcsError> {
    let plan = WorkPlan::for_hash_field(
        manifest.set(),
        manifest.records(),
        manifest.relation(),
        manifest.hash_field(),
    )?;
    if !manifest.matches_plan(&plan) {
        return Err(PcsError::ManifestMismatch);
    }
    Ok(PcsVerifier::new(plan)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::ExtField;
    use jolt_field::Ring;

    #[test]
    fn group_polynomials_zero_extend_and_convert_exactly() {
        let narrow: Vec<i8> = (0..4096).map(|i| (i % 16) as i8 - 8).collect();
        let wide: Vec<i32> = (0..4096).map(|i| (i * 17) % 65_536 - 32_768).collect();
        let ones = vec![1_i8; 4096];
        let columns = [Cells::Narrow(&narrow), Cells::Wide(&wide), Cells::Narrow(&ones)];
        let polynomials = group_polynomials(&columns, 13, None).unwrap();
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
        assert!(group_polynomials(&[Cells::Narrow(&long)], 13, None).is_err());
    }

    #[test]
    fn dense_shape_attaches_the_width16_mirror() {
        let narrow: Vec<i8> = (0..4096).map(|i| (i % 16) as i8 - 8).collect();
        let wide: Vec<i32> = (0..4096).map(|i| (i % 65_536) as i32 - 32_768).collect();
        let columns = [Cells::Narrow(&narrow), Cells::Wide(&wide)];
        let shape = DenseShape {
            ring_dimension: 64,
            num_digits: 1,
            log_basis: 16,
        };
        let polynomials = group_polynomials(&columns, 12, Some(shape)).unwrap();
        let mirror = polynomials[1].small_i16_mirror().expect("width-16 mirror");
        assert_eq!(mirror.len(), 4096);
        assert_eq!(
            mirror[..4096]
                .iter()
                .copied()
                .map(i32::from)
                .collect::<Vec<_>>(),
            wide
        );
        assert!(polynomials[0].small_i16_mirror().is_none());
    }

    #[test]
    fn wide_constructor_matches_field_evals() {
        let wide: Vec<i32> = (0..4096).map(|i| (i % 65_536) as i32 - 32_768).collect();
        let evals: Vec<BaseField> = wide
            .iter()
            .map(|&value| BaseField::from_i64(i64::from(value)))
            .collect();
        let direct = DensePoly::from_wide_field_evals(12, evals.clone(), &wide).unwrap();
        let old = DensePoly::from_field_evals(12, evals)
            .unwrap()
            .attach_small_i16(&wide)
            .unwrap();
        assert_eq!(direct, old);
        assert_eq!(direct.small_mirror_bounds(), old.small_mirror_bounds());
        assert!(direct.small_mirror_bounds().is_some());
    }

    #[test]
    fn wide_constructor_spans_parallel_mirror_chunks() {
        const N: usize = 1 << 16;
        let encode = |cells: &[i32]| -> Vec<BaseField> {
            cells
                .iter()
                .map(|&value| BaseField::from_i64(i64::from(value)))
                .collect()
        };
        let mut wide = vec![0_i32; N];
        wide[1] = -32_768;
        wide[N / 2] = 17;
        wide[N - 2] = 32_767;
        let polynomial = DensePoly::from_wide_field_evals(16, encode(&wide), &wide).unwrap();
        let mirror = polynomial.small_i16_mirror().unwrap();
        assert_eq!(mirror.len(), N);
        assert_eq!(mirror[1], -32_768);
        assert_eq!(mirror[N / 2], 17);
        assert_eq!(mirror[N - 2], 32_767);
        assert!(mirror
            .iter()
            .enumerate()
            .all(|(index, &value)| matches!(index, 1 | 32_768 | 65_534) || value == 0));
        assert_eq!(polynomial.small_mirror_bounds(), Some((-32_768, 32_767)));

        let mut bad = wide;
        bad[7] = 32_768;
        // A second bad cell in the next mirror chunk: the fold reports the earlier one.
        bad[N / 2 + 1] = -32_769;
        let error = DensePoly::from_wide_field_evals(16, encode(&bad), &bad).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("32768"), "{message}");
        assert!(!message.contains("32769"), "the first bad cell must win: {message}");
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
        assert_eq!(lengths, [25, 25, 19, 15, 14, 22, 16]);
        assert_eq!(points[4][0], point[11]);
        assert_eq!(points[5][0], point[3]);
        assert_eq!(points[6], pi_m);
        assert!(group_points(&manifest, &point[1..], &pi_m).is_err());
        assert!(group_points(&manifest, &point, &pi_m[1..]).is_err());
    }
}
