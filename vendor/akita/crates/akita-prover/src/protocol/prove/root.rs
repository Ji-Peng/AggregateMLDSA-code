use super::*;
use crate::backend::OperationCtx;
use crate::SelectedProverOpeningData;
use akita_config::TrustedScheduleCatalog;
use jolt_field::AdditiveGroup;

#[allow(clippy::type_complexity)]
pub(super) fn resolve_root<'a, 's, Cfg, H>(
    expanded: &akita_types::AkitaSetupDescriptor,
    schedules: &'s TrustedScheduleCatalog<Cfg>,
    opening: SelectedProverOpeningData<'a, Cfg::ExtField, H, Cfg::Field>,
    basis: BasisMode,
) -> Result<
    (
        &'s akita_config::ResolvedScheduleRow,
        ProverOpeningData<'a, Cfg::ExtField, H, Cfg::Field>,
        akita_params::GrindingPlan,
        Vec<u8>,
    ),
    AkitaError,
>
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
    H: crate::CommitmentHandleMetadata,
{
    let (selection, claims) = opening.into_low_level_parts();
    let resolved = schedules.resolve_selection(selection)?;
    resolved.validate_opening_layout(claims.opening_layout())?;
    let (grinding_plan, descriptor_bytes) =
        akita_config::transcript_instance_descriptor::<Cfg::Field, Cfg>(
            expanded,
            claims.opening_layout(),
            selection,
            resolved.schedule(),
            basis,
        )?;
    Ok((resolved, claims, grinding_plan, descriptor_bytes))
}

/// Root state that depends only on the opening point and the commitments.
pub(super) struct PreparedRootState<
    F: Field + CanonicalEncoding,
    E: Field,
    OpeningHandle,
    BuildHandle,
> {
    part: PreparedRootFold<F, E, OpeningHandle, BuildHandle>,
}

/// Admit commitments and compute the transcript-free part of the root fold.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub(super) fn prepare_root<Cfg, B>(
    backend: &B,
    claims: &akita_types::OpeningClaims<'_, Cfg::ExtField, akita_types::Commitment<Cfg::Field>>,
    layout: &OpeningClaimsLayout,
    handles: &[B::CommitmentHandle],
    schedule: &FoldSchedule,
    proof_session: &B::ProofSessionHandle,
    basis: BasisMode,
) -> Result<
    PreparedRootState<
        Cfg::Field,
        Cfg::ExtField,
        B::PreparedOpeningHandle,
        B::WitnessBuildHandle,
    >,
    AkitaError,
>
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
    let root_claims = ProverOpeningData::from_parts(
        claims.clone(),
        layout.clone(),
        handles.iter().map(OpeningSource::Commitment).collect(),
    )?;
    let context = backend.proof_context(proof_session, 0)?;
    let mut material = Vec::with_capacity(layout.num_groups());
    for (index, handle) in handles.iter().enumerate() {
        let group = schedule.root.params.group_params(layout, index)?;
        material.push(backend.validate_commitment(
            proof_session,
            &context.for_group(index),
            handle,
            &group.profile,
            claims.group_commitment(index)?,
        )?);
    }
    let protocol_points = claims
        .groups()
        .iter()
        .map(|group| group.point().to_vec())
        .collect();
    let opening_ctx = OperationCtx::new(backend, proof_session, context);
    let part = prepare_fold_part::<Cfg::Field, Cfg::ExtField, B>(
        &opening_ctx,
        root_claims,
        material,
        protocol_points,
        None,
        false,
        0,
        &schedule.root.params,
        basis,
    )?;
    Ok(PreparedRootState { part })
}

/// Bind the prepared root to the live transcript and prove its fold.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub(super) fn finish_root<Cfg, B>(
    expanded: &akita_types::AkitaSetupDescriptor,
    prefix_slots: &SetupPrefixProverRegistry<Cfg::Field, B::CommitmentHandle>,
    backend: &B,
    claims: &akita_types::OpeningClaims<'_, Cfg::ExtField, akita_types::Commitment<Cfg::Field>>,
    layout: &OpeningClaimsLayout,
    schedule: &FoldSchedule,
    proof_session: &B::ProofSessionHandle,
    grinding: &mut akita_types::ProverGrinding<'_>,
    prepared: PreparedRootState<
        Cfg::Field,
        Cfg::ExtField,
        B::PreparedOpeningHandle,
        B::WitnessBuildHandle,
    >,
    handoff: &mut super::registry::FoldHandoff<'_, '_, '_, Cfg>,
) -> Result<
    ProveLevelOutput<Cfg::Field, Cfg::ExtField, B::CommitmentMaterialHandle, B::WitnessHandle>,
    AkitaError,
>
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
    crate::opening::append_opening_messages(
        claims,
        layout,
        &schedule.root.params,
        grinding,
    )?;
    let (next_params, next_binding) = schedule.recursive_folds.first().map_or(
        (
            fold::FoldSuccessorParams::Terminal(&schedule.terminal),
            akita_params::NextWitnessBindingPolicy::TerminalInnerState,
        ),
        |next| {
            (
                fold::FoldSuccessorParams::Recursive(next),
                akita_params::NextWitnessBindingPolicy::OuterPayload,
            )
        },
    );
    let context = backend.proof_context(proof_session, 0)?;
    let opening_ctx = OperationCtx::new(backend, proof_session, context);
    let prepared_fold = finish_fold_part::<Cfg::Field, Cfg::ExtField, B>(
        prepared.part,
        &opening_ctx,
        grinding,
        schedule.root.output_witness_len,
        next_params.inner_ring_dimension(),
    )?;
    prove_fold::<Cfg::Field, Cfg::ExtField, B>(
        expanded,
        prefix_slots,
        backend,
        proof_session,
        grinding,
        0,
        &schedule.root.params,
        next_params,
        schedule.root.output_witness_len,
        next_binding,
        prepared_fold,
        |relation, commitment, binding, witness, material| {
            handoff.handoff_successor(
                backend,
                proof_session,
                0,
                &schedule.root.params,
                relation,
                commitment,
                binding,
                witness,
                material,
            )
        },
    )
}

/// Prove a root fold in one call, for callers that do not split preparation.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub(super) fn prove_root<Cfg, B>(
    expanded: &akita_types::AkitaSetupDescriptor,
    prefix_slots: &SetupPrefixProverRegistry<Cfg::Field, B::CommitmentHandle>,
    backend: &B,
    claims: &akita_types::OpeningClaims<'_, Cfg::ExtField, akita_types::Commitment<Cfg::Field>>,
    layout: &OpeningClaimsLayout,
    handles: &[B::CommitmentHandle],
    schedule: &FoldSchedule,
    proof_session: &B::ProofSessionHandle,
    grinding: &mut akita_types::ProverGrinding<'_>,
    basis: BasisMode,
    handoff: &mut super::registry::FoldHandoff<'_, '_, '_, Cfg>,
) -> Result<
    ProveLevelOutput<Cfg::Field, Cfg::ExtField, B::CommitmentMaterialHandle, B::WitnessHandle>,
    AkitaError,
>
where
    Cfg: CommitmentConfig,
    Cfg::Field: CanonicalEncoding + AkitaSerialize + Unreduced + PseudoMersenne + Ring + 'static,
    <Cfg as CommitmentConfig>::ExtField: FpExtEncoding<Cfg::Field>
        + ExtField<Cfg::Field>
        + Unreduced
        + Fold
        + Ring
        + MulBaseUnreduced<Cfg::Field>
        + AkitaSerialize
        + 'static,
    B: ProverBackend<Cfg::Field, Cfg::ExtField> + 'static,
{
    let prepared = prepare_root::<Cfg, B>(
        backend,
        claims,
        layout,
        handles,
        schedule,
        proof_session,
        basis,
    )?;
    finish_root::<Cfg, B>(
        expanded,
        prefix_slots,
        backend,
        claims,
        layout,
        schedule,
        proof_session,
        grinding,
        prepared,
        handoff,
    )
}
