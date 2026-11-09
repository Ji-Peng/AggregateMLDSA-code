//! Work plan of one batch: padded size, group shapes and the catalog
//! selection, derived from the record count and the generated catalog alone
//! ("verifier-derivable; never prover-supplied", as in Aerie).

use akita_config::SetupRequirements;
use akita_error::AkitaError;
use akita_params::{GroupCommitPhaseParams, OpeningScheduleSelection, ScheduleLookupKey};

use akita_params::PolynomialGroupLayout;

use crate::catalogs::{
    BatchLayout, Family, Group, Relation, Set, D_SEG_MAX, MAX_D_SIG, MIN_D_SIG, PCS_MIN_CELLS_LOG2,
};
use crate::preset::{AggFp64SignedWidth16, AggFp64SignedWidth4, BaseField};

/// The scalar producer profile of `layout` in the catalog of `family`.
pub fn commitment_profile(
    family: Family,
    layout: PolynomialGroupLayout,
) -> Result<GroupCommitPhaseParams, AkitaError> {
    match family {
        Family::Width4 => AggFp64SignedWidth4::commitment_profile(layout),
        Family::Width16 => AggFp64SignedWidth16::commitment_profile(layout),
    }
}

/// Largest record count of this wire version (`N <= 2^15`).
pub const MAX_RECORDS: usize = 1 << MAX_D_SIG;

/// Why a record count has no work plan.
#[derive(Debug)]
pub enum PlanError {
    /// `N = 0`.
    NoRecords,
    /// `N > 2^15` (the verifier MUST reject).
    TooManyRecords(usize),
    /// The padded size `2^d_sig` has no checked-in catalog rows for the relation.
    NotInCatalog {
        /// `log2 N*`.
        d_sig: usize,
        /// The requested relation.
        relation: Relation,
    },
    /// Akita rejected the catalog lookup or setup sizing.
    Akita(AkitaError),
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoRecords => write!(f, "a batch needs at least one record"),
            Self::TooManyRecords(n) => write!(f, "{n} records exceed the maximum {MAX_RECORDS}"),
            Self::NotInCatalog { d_sig, relation } => {
                write!(f, "batch size 2^{d_sig} has no generated {relation} catalog rows")
            }
            Self::Akita(error) => write!(f, "akita: {error}"),
        }
    }
}

impl std::error::Error for PlanError {}

impl From<AkitaError> for PlanError {
    fn from(error: AkitaError) -> Self {
        Self::Akita(error)
    }
}

/// `N* = max(next_power_of_two(N), 8)` and `d_sig = log2 N*`.
pub fn padded_size(records: usize) -> Result<(usize, usize), PlanError> {
    if records == 0 {
        return Err(PlanError::NoRecords);
    }
    if records > MAX_RECORDS {
        return Err(PlanError::TooManyRecords(records));
    }
    let padded = records.next_power_of_two().max(1 << MIN_D_SIG);
    Ok((padded, padded.trailing_zeros() as usize))
}

/// Shapes and catalog profiles of one batch.
#[derive(Clone, Debug)]
pub struct WorkPlan {
    /// Live records `N`.
    pub records: usize,
    /// `N*`.
    pub padded: usize,
    /// The ordered seven-group layout (it carries the relation).
    pub layout: BatchLayout,
    /// Producer profiles of the precommitted groups `G_main16, G_main4, G_1, G_2, G_3,
    /// G_k`, each from the scalar row of its family's catalog.
    pub precommitted: Vec<GroupCommitPhaseParams>,
    /// Profile of the final group `G_m`, from the grouped row of the width-4 catalog.
    pub final_profile: GroupCommitPhaseParams,
    /// The public schedule selection of the batched opening.
    pub selection: OpeningScheduleSelection,
}

impl WorkPlan {
    /// The plan for ML-DSA-44 with `records` live records under `relation`.
    pub fn for_records(records: usize, relation: Relation) -> Result<Self, PlanError> {
        Self::for_set(Set::S44, records, relation)
    }

    /// The plan for `set` with `records` live records under `relation`.
    pub fn for_set(set: Set, records: usize, relation: Relation) -> Result<Self, PlanError> {
        let (padded, d_sig) = padded_size(records)?;
        let layout = BatchLayout::new(set, d_sig, relation)
            .ok_or(PlanError::NotInCatalog { d_sig, relation })?;
        if !layout.in_catalog() {
            return Err(PlanError::NotInCatalog { d_sig, relation });
        }
        let precommitted = layout
            .precommitted_groups()
            .into_iter()
            .map(|(group, shape)| commitment_profile(group.family(), shape))
            .collect::<Result<Vec<_>, _>>()?;
        let key = ScheduleLookupKey {
            final_group: layout.final_group(),
            precommitteds: precommitted.clone(),
        };
        let row = AggFp64SignedWidth4::catalog()?.resolve_key(&key)?;
        Ok(Self {
            records,
            padded,
            layout,
            precommitted,
            final_profile: row.profiles().final_group,
            selection: row.selection(),
        })
    }

    /// The parameter set of the plan.
    #[must_use]
    pub const fn set(&self) -> Set {
        self.layout.set
    }

    /// `log2 N*`.
    #[must_use]
    pub const fn d_sig(&self) -> usize {
        self.layout.d_sig
    }

    /// The relation whose layout this plan commits.
    #[must_use]
    pub const fn relation(&self) -> Relation {
        self.layout.relation
    }

    /// Committed polynomials of `group` (its layout's count under the plan's parameter set and relation).
    #[must_use]
    pub const fn num_polynomials(&self, group: Group) -> usize {
        self.layout.groups[group.index()].num_polynomials()
    }

    /// Number of `2^13`-signature segments `S`.
    #[must_use]
    pub const fn segments(&self) -> usize {
        1 << self.d_sig().saturating_sub(D_SEG_MAX)
    }

    /// Logical variables of `group` (its domain).
    #[must_use]
    pub const fn logical_vars(&self, group: Group) -> usize {
        group.logical_vars(self.set(), self.d_sig())
    }

    /// Logical cells per polynomial of `group`.
    #[must_use]
    pub const fn logical_cells(&self, group: Group) -> usize {
        1 << self.logical_vars(group)
    }

    /// Committed (physical) variables of `group`.
    #[must_use]
    pub const fn physical_vars(&self, group: Group) -> usize {
        group.physical_vars(self.set(), self.d_sig())
    }

    /// Trailing zero pad variables between the logical and physical tables.
    #[must_use]
    pub const fn pad_vars(&self, group: Group) -> usize {
        self.physical_vars(group) - self.logical_vars(group)
    }

    /// The commitment profile of `group`.
    #[must_use]
    pub fn profile(&self, group: Group) -> &GroupCommitPhaseParams {
        if group.is_final() {
            &self.final_profile
        } else {
            &self.precommitted[group.index()]
        }
    }

    /// The catalog lookup key of the grouped row.
    #[must_use]
    pub fn schedule_key(&self) -> ScheduleLookupKey {
        ScheduleLookupKey {
            final_group: self.layout.final_group(),
            precommitteds: self.precommitted.clone(),
        }
    }

    /// Akita setup sized for this batch (largest group, every committed polynomial: 53 to 59
    /// for ML-DSA-44, 101 to 121 for ML-DSA-65, 125 to 151 for ML-DSA-87). The final
    /// family's catalog carries every precommitted profile of its grouped rows, so its
    /// requirements cover the width-16 commits too (Akita's mixed-bound batches).
    pub fn setup_requirements(&self) -> Result<SetupRequirements<BaseField>, AkitaError> {
        SetupRequirements::from_catalog(
            AggFp64SignedWidth4::catalog()?,
            self.layout.max_physical_vars().max(PCS_MIN_CELLS_LOG2),
            self.layout.total_polynomials(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padded_sizes() {
        assert_eq!(padded_size(1).unwrap(), (8, 3));
        assert_eq!(padded_size(8).unwrap(), (8, 3));
        assert_eq!(padded_size(9).unwrap(), (16, 4));
        assert_eq!(padded_size(1000).unwrap(), (1024, 10));
        assert_eq!(padded_size(MAX_RECORDS).unwrap(), (MAX_RECORDS, 15));
        assert!(matches!(padded_size(0), Err(PlanError::NoRecords)));
        assert!(matches!(
            padded_size(MAX_RECORDS + 1),
            Err(PlanError::TooManyRecords(_))
        ));
    }

    #[cfg(feature = "schedules")]
    #[test]
    fn plans_exist_for_every_catalog_size() {
        for set in Set::ALL {
            for layout in crate::catalogs::catalog_batches_for_set(set) {
                let (d_sig, relation) = (layout.d_sig, layout.relation);
                for records in [(1 << d_sig) / 2 + 1, 1 << d_sig] {
                    let plan = WorkPlan::for_set(set, records, relation).expect("catalog plan");
                    assert_eq!((plan.d_sig(), plan.relation()), (d_sig, relation));
                    assert_eq!(plan.set(), set);
                    for group in Group::ALL {
                        assert_eq!(plan.profile(group).group, group.layout(set, d_sig, relation));
                        assert_eq!(plan.num_polynomials(group), group.num_polynomials(set, relation));
                        assert_eq!(
                            plan.logical_vars(group) + plan.pad_vars(group),
                            plan.physical_vars(group)
                        );
                    }
                    assert_eq!(plan.segments(), 1 << d_sig.saturating_sub(13));
                }
            }
        }
        // Every padded size of the wire version has rows.
        let plan = WorkPlan::for_records(1000, Relation::HFips).expect("plan");
        assert_eq!(plan.precommitted.len(), crate::catalogs::GROUP_COUNT - 1);
        assert_eq!(
            crate::catalogs::CATALOG_D_SIG_H_FIPS,
            (MIN_D_SIG..=MAX_D_SIG).collect::<Vec<_>>().as_slice()
        );
        assert!(matches!(
            WorkPlan::for_records(0, Relation::HFips),
            Err(PlanError::NoRecords)
        ));
        assert!(matches!(
            WorkPlan::for_records(MAX_RECORDS + 1, Relation::HExact),
            Err(PlanError::TooManyRecords(_))
        ));
    }
}
