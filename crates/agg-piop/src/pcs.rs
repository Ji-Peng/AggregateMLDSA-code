//! Akita commitments of the four groups and their batched opening (spec Sections 4.1, 7.5,
//! 10 events 1, 4 and 12).
//!
//! Mirrors `aerie-prover/src/protocol/pcs/group.rs`: narrow signed tables stay `i8` until
//! commitment, and each polynomial is materialized polynomial-major, zero-extended to the
//! physical (PCS floor) length, through a 256-entry field lookup table. Commitments go
//! through `agg_pcs_config::commit::PcsProver` (round 1 `G_main`; round 2 `G_1`, `G_2`, then
//! the final group `G_3`), each call transcript-free; the caller absorbs
//! [`GroupCommitment::serialized`] in event order. One batched opening covers the 67 claims
//! at the four group points.

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
use crate::source_tables::{BridgeGroupTables, MainTables, SourceTables};
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

/// Materialize one Akita polynomial per logical `i8` table, in group order, each
/// zero-extended to `2^physical_vars` evaluations (index bits least significant first, so
/// the pad variables are trailing).
pub fn group_polynomials(
    columns: &[Vec<i8>],
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
            evaluations[..cells.len()]
                .par_chunks_mut(FIELD_CONVERSION_CHUNK)
                .zip(cells.par_chunks(FIELD_CONVERSION_CHUNK))
                .for_each(|(output, input)| {
                    for (value, &cell) in output.iter_mut().zip(input) {
                        *value = values[usize::from(cell as u8)];
                    }
                });
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

/// The group points induced by one main-domain point: `G_main` at the full point, `G_t` at
/// its last `d_t` coordinates (spec Section 7.1, suffixes).
pub fn suffix_points(
    manifest: &CommitmentManifest,
    point: &[OpeningField],
) -> Result<[Vec<OpeningField>; GROUP_COUNT], PcsError> {
    if point.len() != manifest.d_main() {
        return Err(PcsError::PointShape(GroupRole::Main));
    }
    Ok(manifest
        .groups()
        .clone()
        .map(|group| point[point.len() - group.logical_vars..].to_vec()))
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
            .map(|cells| evaluate_column(cells, point))
            .collect();
        GroupClaim {
            point: point.clone(),
            values,
        }
    }))
}

/// The prover side: the manifest, its work plan and the Akita prover setup.
pub struct TableCommitter {
    manifest: CommitmentManifest,
    prover: PcsProver,
}

impl TableCommitter {
    /// Plan the batch of `manifest` and allocate the seed-derived setup.
    pub fn new(manifest: CommitmentManifest) -> Result<Self, PcsError> {
        let plan = WorkPlan::for_records(manifest.records())?;
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

    fn commit(&self, role: GroupRole, columns: &[Vec<i8>]) -> Result<GroupCommitment, PcsError> {
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

    /// Round 1 (event 1, before the bridge seed): `G_main`.
    pub fn commit_round1(&self, main: &MainTables) -> Result<GroupCommitment, PcsError> {
        self.commit(GroupRole::Main, main.columns())
    }

    /// Round 2 (event 4, after the bridge seed): `G_1`, `G_2` and the final group `G_3`.
    /// The three commits are independent (the final group is committed against the
    /// precommitted profiles of the work plan, not against the other two commitments), so
    /// they run in parallel; an Akita commit is memory bound and does not use every thread
    /// on its own. The caller absorbs them in wire order.
    pub fn commit_round2(
        &self,
        bridge: &BridgeGroupTables,
    ) -> Result<[GroupCommitment; 3], PcsError> {
        let commit = |role: GroupRole| self.commit(role, bridge.group(role));
        let (level1, (level2, level3)) = rayon::join(
            || commit(GroupRole::Level1),
            || rayon::join(|| commit(GroupRole::Level2), || commit(GroupRole::Level3)),
        );
        Ok([level1?, level2?, level3?])
    }

    /// The batched opening of all four groups at `points` under the 32-byte PIOP session
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

    /// The batched opening of all four groups for claims the caller already holds (the
    /// prover's stage-3 terminal values, which equal the table evaluations).
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
    let plan = WorkPlan::for_records(manifest.records())?;
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
        let columns = vec![
            (0..4096).map(|i| (i % 16) as i8 - 8).collect::<Vec<i8>>(),
            vec![1; 4096],
        ];
        let polynomials = group_polynomials(&columns, 13).unwrap();
        assert_eq!(polynomials.len(), 2);
        for (poly, cells) in polynomials.iter().zip(&columns) {
            let coeffs = poly.field_coeffs();
            assert!(coeffs.len() >= 1 << 13);
            for (index, coeff) in coeffs.iter().enumerate() {
                let expected = cells.get(index).map_or(0, |&c| i64::from(c));
                assert_eq!(*coeff, BaseField::from_i64(expected), "cell {index}");
            }
        }
        assert!(group_polynomials(&[vec![0; 1 << 14]], 13).is_err());
    }

    #[test]
    fn suffix_points_follow_the_group_domains() {
        let manifest = CommitmentManifest::for_batch(1 << 15).unwrap();
        let point: Vec<OpeningField> = (0..25)
            .map(|i| OpeningField::lift_base(BaseField::from_u64(i)))
            .collect();
        let points = suffix_points(&manifest, &point).unwrap();
        let lengths = points.each_ref().map(Vec::len);
        assert_eq!(lengths, [25, 19, 15, 14]);
        assert_eq!(points[3][0], point[11]);
        assert!(suffix_points(&manifest, &point[1..]).is_err());
    }
}
