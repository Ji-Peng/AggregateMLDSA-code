//! Per-fold instance selection; typed executors retain all kernel state.
use super::registry::*;
pub use super::registry::{BackendId, BackendRegistry, FixedFoldRoute, FoldExecutionPolicy};
use super::{root, suffix, FoldPublicState, SuffixProverState};
use crate::backend::*;
use crate::SelectedProverOpeningData;
use crate::SetupPrefixProverRegistry;
use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_error::AkitaError;
use akita_params::{BasisMode, OpeningClaimsLayout};
use akita_serialization::AkitaSerialize;
use akita_types::{AkitaSetupDescriptor, Commitment, FpExtEncoding, OpeningClaims, ProverGrinding};
use jolt_field::{
    AdditiveGroup, CanonicalEncoding, ExtField, Fold, MulBaseUnreduced, PseudoMersenne, Ring,
    Unreduced,
};
use std::any::{Any, TypeId};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_PROOF: AtomicU64 = AtomicU64::new(1);

impl<Cfg, B> Executor<Cfg> for TypedExecutor<'_, Cfg, B>
where
    Cfg: CommitmentConfig,
    Cfg::Field: CanonicalEncoding + AkitaSerialize + Unreduced + PseudoMersenne + Ring + 'static,
    <Cfg::Field as Unreduced>::Wide: From<Cfg::Field> + AdditiveGroup,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>
        + ExtField<Cfg::Field>
        + Unreduced
        + Fold
        + Ring
        + MulBaseUnreduced<Cfg::Field>
        + AkitaSerialize
        + 'static,
    B: ProverBackend<Cfg::Field, Cfg::ExtField> + 'static,
{
    fn identity(&self) -> BackendInstanceIdentity {
        self.backend.instance_identity()
    }
    fn backend_type_id(&self) -> TypeId {
        TypeId::of::<B>()
    }
    fn root(
        &self,
        handoff: &mut FoldHandoff<'_, '_, '_, Cfg>,
        grinding: &mut ProverGrinding<'_>,
        claims: &OpeningClaims<'_, Cfg::ExtField, Commitment<Cfg::Field>>,
        handles: &dyn Any,
        basis: BasisMode,
    ) -> Result<ExecutionContinuation<Cfg>, AkitaError> {
        let handles = handles
            .downcast_ref::<Vec<B::CommitmentHandle>>()
            .ok_or_else(|| {
                AkitaError::InvalidInput(
                    "root commitment handle family differs from selected backend".into(),
                )
            })?;
        let session = self
            .session
            .try_borrow_mut()
            .map_err(|_| AkitaError::InvalidInput("root executor re-entry".into()))?;
        let request = handoff.request;
        let state = root::prove_root::<Cfg, B>(
            request.setup,
            &self.prefixes,
            self.backend,
            claims,
            request.layout,
            handles,
            request.schedule,
            session
                .as_ref()
                .ok_or_else(|| AkitaError::Internal("root executor has no session".into()))?
                .session(),
            grinding,
            basis,
            handoff,
        )?
        .next_state;
        handoff.complete(self.backend, state)
    }
    fn prepare_root(
        &self,
        request: &ProofRequest<'_, Cfg>,
        claims: &OpeningClaims<'_, Cfg::ExtField, Commitment<Cfg::Field>>,
        handles: &dyn Any,
        basis: BasisMode,
    ) -> Result<Box<dyn Any + Send>, AkitaError> {
        let handles = handles
            .downcast_ref::<Vec<B::CommitmentHandle>>()
            .ok_or_else(|| {
                AkitaError::InvalidInput(
                    "root commitment handle family differs from selected backend".into(),
                )
            })?;
        let session = self
            .session
            .try_borrow()
            .map_err(|_| AkitaError::InvalidInput("root executor re-entry".into()))?;
        let prepared = root::prepare_root::<Cfg, B>(
            self.backend,
            claims,
            request.layout,
            handles,
            request.schedule,
            session
                .as_ref()
                .ok_or_else(|| AkitaError::Internal("root executor has no session".into()))?
                .session(),
            basis,
        )?;
        Ok(Box::new(prepared))
    }
    fn finish_root(
        &self,
        request: &ProofRequest<'_, Cfg>,
        handoff: &mut FoldHandoff<'_, '_, '_, Cfg>,
        grinding: &mut ProverGrinding<'_>,
        claims: &OpeningClaims<'_, Cfg::ExtField, Commitment<Cfg::Field>>,
        prepared: Box<dyn Any + Send>,
    ) -> Result<ExecutionContinuation<Cfg>, AkitaError> {
        let prepared = prepared
            .downcast::<root::PreparedRootState<
                Cfg::Field,
                Cfg::ExtField,
                B::PreparedOpeningHandle,
                B::WitnessBuildHandle,
            >>()
            .map_err(|_| {
                AkitaError::InvalidInput("prepared root belongs to another backend".into())
            })?;
        let session = self
            .session
            .try_borrow()
            .map_err(|_| AkitaError::InvalidInput("root executor re-entry".into()))?;
        let state = root::finish_root::<Cfg, B>(
            request.setup,
            &self.prefixes,
            self.backend,
            claims,
            request.layout,
            request.schedule,
            session
                .as_ref()
                .ok_or_else(|| AkitaError::Internal("root executor has no session".into()))?
                .session(),
            grinding,
            *prepared,
            handoff,
        )?
        .next_state;
        handoff.complete(self.backend, state)
    }
    fn prepare_executor(&self, request: &ProofRequest<'_, Cfg>) -> Result<(), AkitaError> {
        let mut session = self
            .session
            .try_borrow_mut()
            .map_err(|_| AkitaError::InvalidInput("backend executor re-entry".into()))?;
        let prepared = self.backend.prepare_executor(
            request.setup,
            request.schedules,
            request.schedule,
            request.layout,
        )?;
        *session = Some(ProofScope::admitted(self.backend, prepared));
        Ok(())
    }
    fn begin_fold(&self, request: &ProofRequest<'_, Cfg>, level: usize) -> Result<(), AkitaError> {
        let requirements = FoldExecutionRequirements::new(request.schedule, request.layout, level)?;
        let schedule = requirements.schedule();
        let current = if requirements.level() == 0 {
            Some(&schedule.root.params)
        } else {
            schedule
                .recursive_folds
                .get(requirements.level() - 1)
                .map(|f| &f.params)
        };
        let successor = schedule
            .recursive_folds
            .get(requirements.level())
            .map(|f| &f.params);
        for params in current.into_iter().chain(successor) {
            if let Some(id) = params.setup_prefix().and_then(|p| p.slot_id()) {
                self.prefixes.get(&id).ok_or_else(|| {
                    AkitaError::InvalidSetup("executor lacks its setup-prefix handle".into())
                })?;
            }
        }
        let session = self
            .session
            .try_borrow_mut()
            .map_err(|_| AkitaError::InvalidInput("backend executor re-entry".into()))?;
        self.backend.begin_fold(
            session
                .as_ref()
                .ok_or_else(|| AkitaError::Internal("executor has not been prepared".into()))?
                .session(),
            &requirements,
        )
    }
    fn take_session(&self) -> Result<Box<dyn Any + Send>, AkitaError> {
        let mut session = self
            .session
            .try_borrow_mut()
            .map_err(|_| AkitaError::InvalidInput("backend executor re-entry".into()))?;
        let scope = session
            .take()
            .ok_or_else(|| AkitaError::Internal("executor was not admitted".into()))?;
        Ok(Box::new(scope.into_session()))
    }
    fn adopt_session(&self, session: Box<dyn Any + Send>) -> Result<(), AkitaError> {
        let session = session
            .downcast::<B::ProofSessionHandle>()
            .map_err(|_| {
                AkitaError::InvalidInput("resumed proof session belongs to another backend".into())
            })?;
        let mut slot = self
            .session
            .try_borrow_mut()
            .map_err(|_| AkitaError::InvalidInput("backend executor re-entry".into()))?;
        if slot.is_some() {
            return Err(AkitaError::InvalidInput(
                "executor already owns a proof session".into(),
            ));
        }
        *slot = Some(ProofScope::admitted(self.backend, *session));
        Ok(())
    }
    fn import_packet(
        &self,
        id: BackendId,
        plan: &ValidatedSuccessorHandoffPlan<'_, Cfg::Field>,
        packet: Box<dyn Any + Send>,
    ) -> Result<ResidentSuccessor, AkitaError> {
        let session = self.session.try_borrow_mut().map_err(|_| {
            AkitaError::InvalidInput("backend executor re-entry during import".into())
        })?;
        let packet = packet.downcast::<B::ImportPacket>().map_err(|_| {
            AkitaError::InvalidInput("destination import packet family mismatch".into())
        })?;
        let imported = self.backend.import_successor(
            session
                .as_ref()
                .ok_or_else(|| AkitaError::InvalidInput("destination was not admitted".into()))?
                .session(),
            plan,
            *packet,
        )?;
        let (witness, material) = imported.into_parts();
        Ok(ResidentSuccessor::new(id, witness, material))
    }
    fn recursive(
        &self,
        handoff: &mut FoldHandoff<'_, '_, '_, Cfg>,
        grinding: &mut ProverGrinding<'_>,
        public: FoldPublicState<Cfg::Field, Cfg::ExtField>,
        resident: ResidentSuccessor,
        index: usize,
    ) -> Result<ExecutionContinuation<Cfg>, AkitaError> {
        let session = self.session.try_borrow_mut().map_err(|_| {
            AkitaError::InvalidInput("backend executor re-entry during fold".into())
        })?;
        let (witness_handle, commitment_material) =
            resident.into_pair::<B::WitnessHandle, B::CommitmentMaterialHandle>(handoff.current)?;
        let request = handoff.request;
        let state = suffix::prove_recursive_step::<Cfg, B>(
            request.setup,
            &self.prefixes,
            self.backend,
            grinding,
            SuffixProverState {
                public,
                witness_handle,
                commitment_material,
            },
            request.schedule,
            session
                .as_ref()
                .ok_or_else(|| AkitaError::InvalidInput("executor was not admitted".into()))?
                .session(),
            index,
            handoff,
        )?;
        handoff.complete(self.backend, state)
    }
    fn terminal(
        &self,
        id: BackendId,
        request: &ProofRequest<'_, Cfg>,
        grinding: &mut ProverGrinding<'_>,
        public: FoldPublicState<Cfg::Field, Cfg::ExtField>,
        resident: ResidentSuccessor,
    ) -> Result<(), AkitaError> {
        let session = self.session.try_borrow_mut().map_err(|_| {
            AkitaError::InvalidInput("backend executor re-entry during terminal fold".into())
        })?;
        let (witness_handle, commitment_material) =
            resident.into_pair::<B::WitnessHandle, B::CommitmentMaterialHandle>(id)?;
        suffix::prove_terminal_suffix::<Cfg::Field, Cfg::ExtField, B>(
            self.backend,
            grinding,
            public.level,
            SuffixProverState {
                public,
                witness_handle,
                commitment_material,
            },
            &request.schedule.terminal,
            session
                .as_ref()
                .ok_or_else(|| {
                    AkitaError::InvalidInput("terminal executor was not admitted".into())
                })?
                .session(),
        )
    }
    fn finish(&self) -> Result<(), AkitaError> {
        let mut session = self.session.try_borrow_mut().map_err(|_| {
            AkitaError::InvalidInput("backend executor re-entry during completion".into())
        })?;
        if let Some(scope) = session.take() {
            scope.finish()?;
        }
        Ok(())
    }
    fn abort(&self) {
        if let Ok(mut session) = self.session.try_borrow_mut() {
            session.take();
        }
    }
}

/// Prove with one transcript and route-selected registry entries for every fold.
pub fn batched_prove<'a, Cfg, H>(
    expanded: &AkitaSetupDescriptor,
    schedules: &TrustedScheduleCatalog<Cfg>,
    registry: &BackendRegistry<'_, Cfg>,
    opening: SelectedProverOpeningData<'a, Cfg::ExtField, H, Cfg::Field>,
    transcript_session: &[u8],
    basis: BasisMode,
    policy: &mut dyn FoldExecutionPolicy,
) -> Result<Vec<u8>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: CanonicalEncoding + AkitaSerialize + Unreduced + PseudoMersenne + Ring + 'static,
    <Cfg::Field as Unreduced>::Wide: From<Cfg::Field> + AdditiveGroup,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>
        + ExtField<Cfg::Field>
        + Unreduced
        + Fold
        + Ring
        + MulBaseUnreduced<Cfg::Field>
        + AkitaSerialize
        + 'static,
    H: CommitmentHandleMetadata + 'static,
{
    let (resolved, claims, grinding_plan, descriptor_bytes) =
        root::resolve_root::<Cfg, H>(expanded, schedules, opening, basis)?;
    if registry.active.replace(true) {
        return Err(AkitaError::InvalidInput(
            "backend registry already has an active proof".into(),
        ));
    }
    let _guard = RegistryProofGuard(registry);
    let mut owners = Vec::with_capacity(resolved.schedule().num_fold_levels());
    let mut selected = vec![false; registry.slots.len()];
    for level in 0..resolved.schedule().num_fold_levels() {
        let requirements =
            FoldExecutionRequirements::new(resolved.schedule(), claims.opening_layout(), level)?;
        let current = owners.last().copied();
        let owner = policy.choose_backend(current, &requirements)?;
        registry.slot(owner)?;
        if let Some(previous) = current.filter(|previous| *previous != owner) {
            registry.bridge(previous, owner)?;
        }
        selected[owner.slot] = true;
        owners.push(owner);
    }
    let root_id = *owners
        .first()
        .ok_or_else(|| AkitaError::Internal("proof schedule has no fold levels".into()))?;
    let root_executor = registry.slot(root_id)?;
    let identity = NEXT_PROOF
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .map_err(|_| AkitaError::Internal("proof identity exhausted".into()))?;
    let request = ProofRequest {
        setup: expanded,
        schedules,
        schedule: resolved.schedule(),
        layout: claims.opening_layout(),
        identity,
    };
    for (executor, selected) in registry.slots.iter().zip(&selected) {
        if *selected {
            executor.prepare_executor(&request)?;
        }
    }
    for (level, owner) in owners.iter().enumerate() {
        registry.slot(*owner)?.begin_fold(&request, level)?;
    }
    let mut handoff = FoldHandoff {
        registry,
        request: &request,
        current: root_id,
        next: *owners
            .get(1)
            .ok_or_else(|| AkitaError::Internal("proof schedule has no successor fold".into()))?,
        adopted: None,
    };
    let channel = akita_transcript::new_prover_channel(transcript_session, &descriptor_bytes)?;
    let mut grinding = ProverGrinding::new(channel, &grinding_plan);
    let (mut public, mut resident) = root_executor.root(
        &mut handoff,
        &mut grinding,
        claims.opening_claims(),
        &claims.groups,
        basis,
    )?;
    for index in 0..request.schedule.recursive_folds.len() {
        let owner = resident.owner;
        handoff.current = owner;
        handoff.next = *owners
            .get(index + 2)
            .ok_or_else(|| AkitaError::Internal("resolved route has no successor fold".into()))?;
        (public, resident) = registry.slot(owner)?.recursive(
            &mut handoff,
            &mut grinding,
            public,
            resident,
            index,
        )?;
    }
    let owner = resident.owner;
    let levels = public.level + 1;
    registry
        .slot(owner)?
        .terminal(owner, &request, &mut grinding, public, resident)?;
    if levels != request.schedule.num_fold_levels() {
        return Err(AkitaError::Internal(
            "proved fold count differs from schedule".into(),
        ));
    }
    let proof = grinding.finish()?;
    for (slot, selected) in registry.slots.iter().zip(selected) {
        if selected {
            slot.finish()?;
        }
    }
    Ok(proof)
}

/// A batched opening whose transcript-free work is already complete.
///
/// The state holds no borrow of the opening claims: it owns the prepared
/// backend handles, the admitted proof lifetime and the public claims needed
/// to replay the transcript messages in order.
pub struct PreparedBatchedProve<
    'a,
    Cfg: CommitmentConfig,
    B: ProverBackend<Cfg::Field, Cfg::ExtField>,
> {
    descriptor: AkitaSetupDescriptor,
    schedule: akita_params::FoldSchedule,
    layout: OpeningClaimsLayout,
    claims: OpeningClaims<'a, Cfg::ExtField, Commitment<Cfg::Field>>,
    prefixes: Arc<SetupPrefixProverRegistry<Cfg::Field, B::CommitmentHandle>>,
    session: Box<dyn Any + Send>,
    grinding_plan: akita_params::GrindingPlan,
    descriptor_bytes: Vec<u8>,
    root_state: Box<dyn Any + Send>,
    identity: u64,
    levels: usize,
}

/// Resolve the schedule, admit the proof, and compute the root fold's
/// transcript-free work.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn prepare_batched_prove<'a, Cfg, B>(
    expanded: &AkitaSetupDescriptor,
    schedules: &TrustedScheduleCatalog<Cfg>,
    backend: &B,
    prefix_slots: SetupPrefixProverRegistry<Cfg::Field, B::CommitmentHandle>,
    opening: SelectedProverOpeningData<'a, Cfg::ExtField, B::CommitmentHandle, Cfg::Field>,
    basis: BasisMode,
) -> Result<PreparedBatchedProve<'a, Cfg, B>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: CanonicalEncoding + AkitaSerialize + Unreduced + PseudoMersenne + Ring + 'static,
    <Cfg::Field as Unreduced>::Wide: From<Cfg::Field> + AdditiveGroup,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>
        + ExtField<Cfg::Field>
        + Unreduced
        + Fold
        + Ring
        + MulBaseUnreduced<Cfg::Field>
        + AkitaSerialize
        + 'static,
    B: ProverBackend<Cfg::Field, Cfg::ExtField> + 'static,
{
    let (resolved, claims, grinding_plan, descriptor_bytes) =
        root::resolve_root::<Cfg, B::CommitmentHandle>(expanded, schedules, opening, basis)?;
    let schedule = resolved.schedule().clone();
    let layout = claims.opening_layout().clone();
    let public_claims = claims.opening_claims().clone();
    let prefixes = Arc::new(prefix_slots);
    let mut registry = BackendRegistry::<Cfg>::new()?;
    if registry.active.replace(true) {
        return Err(AkitaError::InvalidInput(
            "backend registry already has an active proof".into(),
        ));
    }
    let root = registry.register_shared(backend, Arc::clone(&prefixes))?;
    let guard = RegistryProofGuard(&registry);
    let levels = schedule.num_fold_levels();
    let identity = NEXT_PROOF
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .map_err(|_| AkitaError::Internal("proof identity exhausted".into()))?;
    let request = ProofRequest {
        setup: expanded,
        schedules,
        schedule: &schedule,
        layout: &layout,
        identity,
    };
    let root_executor = registry.slot(root)?;
    root_executor.prepare_executor(&request)?;
    for level in 0..levels {
        root_executor.begin_fold(&request, level)?;
    }
    let root_state = root_executor.prepare_root(&request, &public_claims, &claims.groups, basis)?;
    let session = root_executor.take_session()?;
    drop(guard);
    Ok(PreparedBatchedProve {
        descriptor: expanded.clone(),
        schedule,
        layout,
        claims: public_claims,
        prefixes,
        session,
        grinding_plan,
        descriptor_bytes,
        root_state,
        identity,
        levels,
    })
}

/// Finish a batched opening on the transcript session that was unavailable
/// when its transcript-free work was prepared.
#[allow(clippy::too_many_arguments)]
pub fn finish_batched_prove<Cfg, B>(
    prepared: PreparedBatchedProve<'_, Cfg, B>,
    schedules: &TrustedScheduleCatalog<Cfg>,
    backend: &B,
    transcript_session: &[u8],
) -> Result<Vec<u8>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: CanonicalEncoding + AkitaSerialize + Unreduced + PseudoMersenne + Ring + 'static,
    <Cfg::Field as Unreduced>::Wide: From<Cfg::Field> + AdditiveGroup,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>
        + ExtField<Cfg::Field>
        + Unreduced
        + Fold
        + Ring
        + MulBaseUnreduced<Cfg::Field>
        + AkitaSerialize
        + 'static,
    B: ProverBackend<Cfg::Field, Cfg::ExtField> + 'static,
{
    let PreparedBatchedProve {
        descriptor,
        schedule,
        layout,
        claims,
        prefixes,
        session,
        grinding_plan,
        descriptor_bytes,
        root_state,
        identity,
        levels,
    } = prepared;
    let mut registry = BackendRegistry::<Cfg>::new()?;
    if registry.active.replace(true) {
        return Err(AkitaError::InvalidInput(
            "backend registry already has an active proof".into(),
        ));
    }
    let root = registry.register_shared(backend, prefixes)?;
    let guard = RegistryProofGuard(&registry);
    let root_executor = registry.slot(root)?;
    root_executor.adopt_session(session)?;
    let request = ProofRequest {
        setup: &descriptor,
        schedules,
        schedule: &schedule,
        layout: &layout,
        identity,
    };
    let channel = akita_transcript::new_prover_channel(transcript_session, &descriptor_bytes)?;
    let mut grinding = ProverGrinding::new(channel, &grinding_plan);
    let mut handoff = FoldHandoff {
        registry: &registry,
        request: &request,
        current: root,
        next: root,
        adopted: None,
    };
    let (mut public, mut resident) = root_executor.finish_root(
        &request,
        &mut handoff,
        &mut grinding,
        &claims,
        root_state,
    )?;
    for index in 0..request.schedule.recursive_folds.len() {
        let owner = resident.owner;
        handoff.current = owner;
        handoff.next = root;
        (public, resident) = registry.slot(owner)?.recursive(
            &mut handoff,
            &mut grinding,
            public,
            resident,
            index,
        )?;
    }
    let owner = resident.owner;
    let proved_levels = public.level + 1;
    registry
        .slot(owner)?
        .terminal(owner, &request, &mut grinding, public, resident)?;
    if proved_levels != levels {
        return Err(AkitaError::Internal(
            "proved fold count differs from schedule".into(),
        ));
    }
    let proof = grinding.finish()?;
    for slot in &registry.slots {
        slot.finish()?;
    }
    drop(guard);
    Ok(proof)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_dispatch_checks_both_instance_and_handle_family() {
        let owner = BackendId {
            registry: 1,
            slot: 0,
        };
        let foreign = BackendId {
            registry: 1,
            slot: 1,
        };
        assert!(ResidentSuccessor::new(owner, 1u8, 2u8)
            .into_pair::<u8, u8>(foreign)
            .is_err());
        assert!(ResidentSuccessor::new(owner, 1u8, 2u8)
            .into_pair::<u16, u16>(owner)
            .is_err());
        assert_eq!(
            ResidentSuccessor::new(owner, 1u8, 2u8)
                .into_pair::<u8, u8>(owner)
                .unwrap(),
            (1, 2)
        );
    }
}
