//! Application-facing cache controls and public resource diagnostics.
use super::{CpuBackend, PreparedCrtNttProfile, PreparedNttCacheMetric};
use crate::opaque::NttExecutionRequirements;
use akita_error::AkitaError;
use akita_params::{FoldSchedule, GroupCommitPhaseParams};
use jolt_field::{CanonicalEncoding, Field, Unreduced, WithCommitAccumulator};

impl<F: jolt_field::Field + jolt_field::CanonicalEncoding, E> CpuBackend<F, E> {
    /// Prepare the CPU cache union for a complete commit-and-prove workload.
    pub fn prewarm(&self, schedule: &FoldSchedule) -> Result<(), AkitaError> {
        let prepared = self.prepared()?;
        let planned = NttExecutionRequirements::from_commit_and_prove_schedule(schedule)?;
        super::requirements::warm_joined_ntt_requirements(self, prepared, &planned)
    }

    /// Initialized setup transforms for application profiling.
    pub fn shared_ntt_cache_metrics(&self) -> Result<Vec<PreparedNttCacheMetric>, AkitaError> {
        self.prepared()?.shared_ntt_cache_metrics()
    }

    /// Arithmetic capacity profile for an initialized ring degree.
    pub fn shared_ntt_profile(&self, ring_d: usize) -> Result<PreparedCrtNttProfile, AkitaError> {
        self.prepared()?.shared_ntt_profile(ring_d)
    }
}

impl<F, E> CpuBackend<F, E>
where
    F: Field + CanonicalEncoding + Unreduced + WithCommitAccumulator + 'static,
{
    /// Prepare the shared NTT slots of one group's root commitment.
    ///
    /// Akita builds a slot lazily inside the first commit that needs it, with
    /// the slot's parallel build running inside that commit's parallel loops.
    /// Warming it on the caller's thread moves the build ahead of the commit
    /// stages; a slot that is already present is left untouched.
    pub fn prewarm_commit(&self, profile: GroupCommitPhaseParams) -> Result<(), AkitaError> {
        let prepared = self.prepared()?;
        let executor = crate::commitment::CommitmentExecutor::cpu(
            self,
            prepared,
            &prepared.expanded,
            Vec::new(),
            crate::commitment::PortableStatePolicy,
        )?;
        let plan = crate::commitment::CommitmentExecutionPlan::for_root(&profile)?;
        executor.prewarm_plan_resources(&plan)
    }
}
