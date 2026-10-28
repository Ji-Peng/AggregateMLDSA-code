//! Two-round commit driver and the batched opening of the six groups
//! (`docs/spec.md`).
//!
//! Round 1 commits `G_main16` and `G_main4` before the bridge seed; round 2 commits
//! `G_1`, `G_2`, `G_3` and, last, the final group `G_m` against the precommitted profiles
//! of the work plan. Every commit is transcript-free (the caller absorbs the
//! serialized commitments in event order), so the two rounds share one
//! commitment contract and one batched Akita opening. Each group is
//! committed under its own family's catalog (width 4 or width 16); the opening and its
//! verifier run under the final group's width-4 catalog, whose grouped row carries the
//! precommitted width-16 profiles (Akita freezes the source bound per group).
//!
//! Tables are logical: `2^logical_vars` cells per polynomial with values in the family's
//! interval (`[-8, 7]` or `[-2^15, 2^15 - 1]`), index bits least significant first. A
//! table below the PCS floor is zero-extended with trailing pad variables, and its opening
//! point is extended with zeros, so the physical opening equals the logical claim.

use std::sync::Arc;

use akita_error::AkitaError;
use akita_params::BasisMode;
use akita_pcs::{
    new_prover_setup, AkitaCommitmentScheme, AkitaProverSetup, CommitmentHandle, CpuBackend,
    DensePoly, GroupContext, PrecommittedGroupProfiles, SelectedProverOpeningData,
};
use akita_serialization::{AkitaSerialize, Compress, Valid};
use akita_types::{
    derive_public_matrix_prefix, sample_akita_setup_seed, AkitaExpandedSetup, AkitaSetupDescriptor,
    AkitaVerifierSetup, Commitment, CommittedGroup, GroupBatchStatement, OpeningClaims,
    PolynomialGroupClaims, SetupPrefixVerifierRegistry,
};
use akita_verifier::AkitaVerifier;
use jolt_field::{ExtField, Ring, Zero};

use crate::catalogs::{Family, Group, GROUP_COUNT};
use crate::preset::{AggFp64SignedWidth16, AggFp64SignedWidth4 as Cfg, BaseField, OpeningField};
use crate::work_plan::WorkPlan;

/// Commit or opening failure.
#[derive(Debug)]
pub enum CommitError {
    /// A group received the wrong number of tables.
    PolynomialCount {
        /// The group.
        group: Group,
        /// Expected count.
        expected: usize,
        /// Received count.
        actual: usize,
    },
    /// A table has the wrong length.
    TableLength {
        /// The group.
        group: Group,
        /// Expected logical cells.
        expected: usize,
        /// Received cells.
        actual: usize,
    },
    /// A value lies outside the group's family interval.
    ValueOutOfRange {
        /// The group.
        group: Group,
        /// Polynomial index within the group.
        polynomial: usize,
        /// Cell index.
        cell: usize,
        /// The value.
        value: i64,
    },
    /// An opening claim has the wrong point length or value count.
    ClaimShape(Group),
    /// Akita failed.
    Akita(AkitaError),
}

impl std::fmt::Display for CommitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PolynomialCount {
                group,
                expected,
                actual,
            } => {
                write!(f, "{group:?}: {actual} tables, expected {expected}")
            }
            Self::TableLength {
                group,
                expected,
                actual,
            } => {
                write!(f, "{group:?}: table of {actual} cells, expected {expected}")
            }
            Self::ValueOutOfRange {
                group,
                polynomial,
                cell,
                value,
            } => write!(
                f,
                "{group:?}: polynomial {polynomial} cell {cell} holds {value} outside [{}, {}]",
                group.family().min_value(),
                group.family().max_value()
            ),
            Self::ClaimShape(group) => write!(f, "{group:?}: opening claim has the wrong shape"),
            Self::Akita(error) => write!(f, "akita: {error}"),
        }
    }
}

impl std::error::Error for CommitError {}

impl From<AkitaError> for CommitError {
    fn from(error: AkitaError) -> Self {
        Self::Akita(error)
    }
}

/// One committed group: the public commitment and the prover's handle.
pub struct GroupCommitment {
    /// Which group.
    pub group: Group,
    /// Profile and commitment (public).
    pub committed: CommittedGroup<BaseField>,
    handle: CommitmentHandle<BaseField, OpeningField>,
}

impl GroupCommitment {
    /// The public commitment.
    #[must_use]
    pub fn commitment(&self) -> &Commitment<BaseField> {
        self.committed.commitment()
    }

    /// Akita's native compressed encoding, the transcript payload of
    /// events 1 and 4.
    pub fn serialized(&self) -> Result<Vec<u8>, CommitError> {
        let mut bytes = Vec::new();
        self.commitment()
            .serialize_with_mode(&mut bytes, Compress::Yes)
            .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
        Ok(bytes)
    }
}

/// Claims on one group: the logical opening point and one value per polynomial.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupClaim {
    /// Logical point (`logical_vars` coordinates, least significant first).
    pub point: Vec<OpeningField>,
    /// Claimed evaluations, in the group's polynomial order.
    pub values: Vec<OpeningField>,
}

fn physical_point(
    plan: &WorkPlan,
    group: Group,
    claim: &GroupClaim,
) -> Result<Vec<OpeningField>, CommitError> {
    if claim.point.len() != plan.logical_vars(group)
        || claim.values.len() != plan.num_polynomials(group)
    {
        return Err(CommitError::ClaimShape(group));
    }
    let mut point = claim.point.clone();
    point.resize(plan.physical_vars(group), OpeningField::zero());
    Ok(point)
}

/// The prover side: setup, backend and the commit and opening calls.
pub struct PcsProver {
    plan: WorkPlan,
    precommitted: PrecommittedGroupProfiles,
    setup: AkitaProverSetup<BaseField>,
    backend: CpuBackend<BaseField, OpeningField>,
}

impl PcsProver {
    /// Allocate the seed-derived setup sized for `plan`.
    pub fn new(plan: WorkPlan) -> Result<Self, CommitError> {
        let setup = new_prover_setup(&plan.setup_requirements()?)?;
        let backend = CpuBackend::new(Arc::clone(&setup.expanded))?;
        let precommitted = PrecommittedGroupProfiles::from_profiles(plan.precommitted.clone())?;
        Ok(Self {
            plan,
            precommitted,
            setup,
            backend,
        })
    }

    /// The work plan.
    #[must_use]
    pub const fn plan(&self) -> &WorkPlan {
        &self.plan
    }

    /// Commit one group's logical tables (transcript-free).
    pub fn commit_group<T: Copy + Into<i64>>(
        &self,
        group: Group,
        tables: &[Vec<T>],
    ) -> Result<GroupCommitment, CommitError> {
        if tables.len() != self.plan.num_polynomials(group) {
            return Err(CommitError::PolynomialCount {
                group,
                expected: self.plan.num_polynomials(group),
                actual: tables.len(),
            });
        }
        let logical = self.plan.logical_cells(group);
        let physical_vars = self.plan.physical_vars(group);
        let mut polynomials = Vec::with_capacity(tables.len());
        for (polynomial, table) in tables.iter().enumerate() {
            if table.len() != logical {
                return Err(CommitError::TableLength {
                    group,
                    expected: logical,
                    actual: table.len(),
                });
            }
            let (min, max) = (group.family().min_value(), group.family().max_value());
            if let Some(cell) = table
                .iter()
                .position(|&v| !(min..=max).contains(&v.into()))
            {
                return Err(CommitError::ValueOutOfRange {
                    group,
                    polynomial,
                    cell,
                    value: table[cell].into(),
                });
            }
            let mut evaluations: Vec<BaseField> = table
                .iter()
                .map(|&v| BaseField::from_i64(v.into()))
                .collect();
            evaluations.resize(1 << physical_vars, BaseField::zero());
            polynomials.push(DensePoly::from_field_evals(physical_vars, evaluations)?);
        }
        self.commit_polynomials(group, polynomials)
    }

    /// Commit one group from already materialized physical polynomials
    /// (`2^physical_vars` evaluations each, zero-extended, in the group's
    /// polynomial order), transcript-free. The caller is responsible for the
    /// value range; Akita enforces the preset's admission bound and the
    /// profile check below rejects a wrong shape.
    pub fn commit_polynomials(
        &self,
        group: Group,
        polynomials: Vec<DensePoly<BaseField>>,
    ) -> Result<GroupCommitment, CommitError> {
        if polynomials.len() != self.plan.num_polynomials(group) {
            return Err(CommitError::PolynomialCount {
                group,
                expected: self.plan.num_polynomials(group),
                actual: polynomials.len(),
            });
        }
        let source = self.backend.import_source(polynomials)?;
        let context = if group.is_final() {
            GroupContext::scheduler_with_precommitted_groups(&self.precommitted)
        } else {
            GroupContext::scheduler_without_precommitted_groups()
        };
        let output = match group.family() {
            Family::Width4 => self.backend.commit(Cfg::catalog()?, &source, context)?,
            Family::Width16 => self
                .backend
                .commit(AggFp64SignedWidth16::catalog()?, &source, context)?,
        };
        if output.committed_group.profile != *self.plan.profile(group) {
            return Err(AkitaError::InvalidSetup(format!(
                "{group:?} committed under a profile other than the work plan's"
            ))
            .into());
        }
        Ok(GroupCommitment {
            group,
            committed: output.committed_group,
            handle: output.private_handle,
        })
    }

    /// Prove the batched opening of all six groups, in wire order, under the
    /// 32-byte PIOP session digest.
    pub fn open(
        &self,
        commitments: [GroupCommitment; GROUP_COUNT],
        claims: &[GroupClaim; GROUP_COUNT],
        session: &[u8; 32],
    ) -> Result<Vec<u8>, CommitError> {
        let mut groups = Vec::with_capacity(GROUP_COUNT);
        let mut handles = Vec::with_capacity(GROUP_COUNT);
        for (commitment, (group, claim)) in
            commitments.into_iter().zip(Group::ALL.iter().zip(claims))
        {
            if commitment.group != *group {
                return Err(CommitError::ClaimShape(*group));
            }
            let point = physical_point(&self.plan, *group, claim)?;
            groups.push(PolynomialGroupClaims::new(
                point,
                claim.values.clone(),
                commitment.committed,
            )?);
            handles.push(commitment.handle);
        }
        let catalog = Cfg::catalog()?;
        // Hold a reference to every group's committed source, so that the last one is not
        // released inside the opening (a one-thread unmap of several GiB at 32K); released
        // after it, on a detached thread. Akita shares
        // the source by `Arc` inside the opening anyway, so the clones copy nothing.
        let keep = handles.clone();
        let opening = SelectedProverOpeningData::from_committed_claims::<Cfg>(
            OpeningClaims::from_groups(groups)?,
            handles,
            catalog,
        )?;
        if opening.selection() != self.plan.selection {
            return Err(AkitaError::InvalidSetup(
                "opening selected a catalog row other than the work plan's".to_owned(),
            )
            .into());
        }
        let proof = AkitaCommitmentScheme::new(catalog.clone()).batched_prove(
            &self.setup,
            opening,
            &self.backend,
            session,
            BasisMode::Lagrange,
        );
        crate::sched::drop_in_background("opening sources", keep);
        Ok(proof?)
    }

    /// The verifier derived from this prover's setup (shares the matrix).
    pub fn verifier(&self) -> Result<PcsVerifier, CommitError> {
        let scheme = AkitaCommitmentScheme::new(Cfg::catalog()?.clone());
        Ok(PcsVerifier {
            plan: self.plan.clone(),
            akita: scheme.verifier(scheme.setup_verifier(&self.setup)?)?,
        })
    }
}

/// The verifier side of the batched opening.
pub struct PcsVerifier {
    plan: WorkPlan,
    akita: AkitaVerifier<Cfg>,
}

impl PcsVerifier {
    /// Build the verifier from the public setup seed alone (no prover state),
    /// as Aerie's `FalconVerifierSetup::new`.
    pub fn new(plan: WorkPlan) -> Result<Self, CommitError> {
        let requirements = plan.setup_requirements()?;
        let descriptor = AkitaSetupDescriptor {
            max_num_vars: requirements.max_num_vars(),
            max_num_batched_polys: requirements.max_num_batched_polys(),
            num_field_elements: requirements.matrix_capacity().num_field_elements,
            setup_seed: sample_akita_setup_seed(),
        };
        descriptor.check().map_err(|error| {
            AkitaError::InvalidSetup(format!("setup seed validation failed: {error}"))
        })?;
        let matrix = derive_public_matrix_prefix::<BaseField>(
            descriptor.num_field_elements,
            &descriptor.setup_seed,
        );
        let registry = SetupPrefixVerifierRegistry::new(descriptor.setup_seed.clone());
        let expanded = Arc::new(
            AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(descriptor, matrix),
        );
        let setup = AkitaVerifierSetup::from_parts(expanded, registry)?;
        Ok(Self {
            plan,
            akita: AkitaVerifier::<Cfg>::new(setup, Cfg::catalog()?.clone())?,
        })
    }

    /// The work plan.
    #[must_use]
    pub const fn plan(&self) -> &WorkPlan {
        &self.plan
    }

    /// Verify the batched opening of the six groups. Group profiles and the
    /// schedule selection come from the work plan, never from the proof.
    pub fn verify(
        &self,
        commitments: &[Commitment<BaseField>; GROUP_COUNT],
        claims: &[GroupClaim; GROUP_COUNT],
        proof: &[u8],
        session: &[u8; 32],
    ) -> Result<(), CommitError> {
        let committed: Vec<CommittedGroup<BaseField>> = Group::ALL
            .iter()
            .zip(commitments)
            .map(|(&group, commitment)| {
                CommittedGroup::new(*self.plan.profile(group), commitment.clone())
            })
            .collect();
        let mut points = Vec::with_capacity(GROUP_COUNT);
        for (&group, claim) in Group::ALL.iter().zip(claims) {
            points.push(physical_point(&self.plan, group, claim)?);
        }
        let groups = committed
            .iter()
            .zip(&points)
            .zip(claims)
            .map(|((group, point), claim)| {
                PolynomialGroupClaims::new(point.as_slice(), claim.values.clone(), group)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let statement =
            GroupBatchStatement::new(self.plan.selection, OpeningClaims::from_groups(groups)?)?;
        Ok(self
            .akita
            .batched_verify(proof, session, statement, BasisMode::Lagrange)?)
    }
}

/// Multilinear extension of a signed table at `point` (least significant
/// variable first), the value a claim on that table must carry.
#[must_use]
pub fn evaluate_signed_table<T: Copy + Into<i64>>(table: &[T], point: &[OpeningField]) -> OpeningField {
    assert_eq!(
        table.len(),
        1 << point.len(),
        "table length must be 2^|point|"
    );
    let Some((&first, rest)) = point.split_first() else {
        return OpeningField::lift_base(BaseField::from_i64(table[0].into()));
    };
    let mut layer: Vec<OpeningField> = table
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&[low, high]| {
            let (low, high): (i64, i64) = (low.into(), high.into());
            let lifted = OpeningField::lift_base(BaseField::from_i64(low));
            lifted + first.mul_base(BaseField::from_i64(high - low))
        })
        .collect();
    for &coordinate in rest {
        let half = layer.len() / 2;
        for j in 0..half {
            let (low, high) = (layer[2 * j], layer[2 * j + 1]);
            layer[j] = low + (high - low) * coordinate;
        }
        layer.truncate(half);
    }
    layer[0]
}
