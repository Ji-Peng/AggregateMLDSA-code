//! The Akita opening of the seven groups (event 13): six groups at
//! `r^(3)` and its suffixes, `G_m` at the lookup's table point; and the parsing of the
//! commitments absorbed at events 1 and 4.

use agg_pcs_config::commit::GroupClaim;
use agg_pcs_config::work_plan::WorkPlan;
use akita_params::CompressionChainPlan;
use akita_serialization::{AkitaDeserialize, Compress, Validate};
use akita_types::Commitment;

use crate::claims::{MultiplicityClaims, Stage3Claims};
use crate::manifest::{CommitmentManifest, GroupRole, GROUP_COUNT};
use crate::pcs::{group_points, PcsError};
use crate::{BaseField, OpeningField as E};
use jolt_field::Zero;

/// The opening claims of the 52 polynomials (H-u): the 34 stage-3 claims at the group points
/// induced by `r3`, zero for the 11 zero polynomials of `G_3`, the multiplicity digits at
/// the table point `pi_m`.
pub fn opening_claims(
    manifest: &CommitmentManifest,
    r3: &[E],
    claims: &Stage3Claims,
    pi_m: &[E],
    multiplicities: &MultiplicityClaims,
) -> Result<[GroupClaim; GROUP_COUNT], PcsError> {
    let points = group_points(manifest, r3, pi_m)?;
    Ok(GroupRole::ALL.map(|role| GroupClaim {
        point: points[role.index()].clone(),
        values: if role == GroupRole::Mult {
            multiplicities.0.to_vec()
        } else {
            // The claimed polynomials come first in their group; G_3's zero polynomials
            // follow with the claim zero.
            let mut values = claims.group(manifest, role).to_vec();
            values.resize(manifest.group(role).polynomial_count(), E::zero());
            values
        },
    }))
}

/// Parse a group commitment from its native encoding; the coefficient count comes from
/// the work plan's profile of the group, never from the proof.
pub fn parse_commitment(
    plan: &WorkPlan,
    role: GroupRole,
    bytes: &[u8],
) -> Result<Commitment<BaseField>, PcsError> {
    let profile = plan.profile(role);
    let source = profile.outer_slice_count.complete_source_coefficients(
        profile.outer.matrix.output_rank(),
        profile.outer.matrix.ring_dimension(),
    )?;
    let expected = CompressionChainPlan::for_complete_source(
        profile.outer.matrix.sis_table_key().modulus_profile,
        source,
    )?
    .terminal_coefficients();
    let mut reader = bytes;
    let commitment = Commitment::<BaseField>::deserialize_with_mode(
        &mut reader,
        Compress::Yes,
        Validate::Yes,
        &expected,
    )
    .map_err(|e| akita_error::AkitaError::InvalidInput(e.to_string()))?;
    if !reader.is_empty() {
        return Err(akita_error::AkitaError::InvalidInput("trailing commitment bytes".into()).into());
    }
    Ok(commitment)
}
