//! The Akita opening of the seven groups (eleven under `HashField::Fp`): every group but
//! `G_m` at `r^(3)` and its suffixes, `G_m` at the lookup's table point; and the parsing of the
//! commitments absorbed in rounds 1 and 2.

use agg_pcs_config::commit::GroupClaim;
use agg_pcs_config::work_plan::WorkPlan;
use akita_params::CompressionChainPlan;
use akita_serialization::{AkitaDeserialize, Compress, Validate};
use akita_types::Commitment;

use crate::claims::{MultiplicityClaims, Stage3Claims};
use crate::manifest::{CommitmentManifest, GroupRole, HashField, GROUP_COUNT};
use crate::pcs::{group_points, group_points_fp, PcsError};
use crate::{BaseField, OpeningField as E};
use jolt_field::Zero;

/// The opening claims of the committed polynomials (59 for ML-DSA-44 under H-FIPS): the
/// stage-3 claims (40 there) at the group points induced by `r3`, zero for the 12 zero
/// polynomials of `G_3`, the multiplicity digits at the table point `pi_m`.
pub fn opening_claims(
    manifest: &CommitmentManifest,
    r3: &[E],
    claims: &Stage3Claims,
    pi_m: &[E],
    multiplicities: &MultiplicityClaims,
) -> Result<[GroupClaim; GROUP_COUNT], PcsError> {
    if manifest.hash_field() != HashField::Gf2 {
        return Err(PcsError::ManifestMismatch);
    }
    let claims = opening_claims_for(manifest, r3, claims, pi_m, multiplicities)?;
    Ok(claims.try_into().expect("seven groups under HashField::Gf2"))
}

/// [`opening_claims`] for either hash field, one claim per group in manifest order: under
/// `HashField::Fp` the seven Keccak groups sit at the `d_K` suffix of `r3` beside the
/// lattice groups (385 + 17 claims for ML-DSA-44 under H-exact), `G_m` at `pi_m`.
pub fn opening_claims_for(
    manifest: &CommitmentManifest,
    r3: &[E],
    claims: &Stage3Claims,
    pi_m: &[E],
    multiplicities: &MultiplicityClaims,
) -> Result<Vec<GroupClaim>, PcsError> {
    let points: Vec<Vec<E>> = match manifest.hash_field() {
        HashField::Gf2 => group_points(manifest, r3, pi_m)?.into(),
        HashField::Fp => group_points_fp(manifest, r3, pi_m)?.into(),
    };
    Ok(manifest
        .groups()
        .iter()
        .zip(points)
        .map(|(group, point)| GroupClaim {
            point,
            values: if group.role == GroupRole::Mult {
                multiplicities.0.to_vec()
            } else {
                // The claimed polynomials come first in their group; G_3's zero polynomials
                // follow with the claim zero.
                let mut values = claims.group(manifest, group.role).to_vec();
                values.resize(group.polynomial_count(), E::zero());
                values
            },
        })
        .collect())
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
