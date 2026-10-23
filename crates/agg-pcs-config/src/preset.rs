//! The width-4 Akita commitment preset of every committed table (spec Section 4.1).
//!
//! [`AggFp64SignedWidth4`] repeats Aerie's `Fp64SignedWidth4` parameters
//! exactly: fp64 field `Prime64Offset59` with `Ext2` challenges, SIS profile
//! `Q64Offset59`, opening decomposition log basis 3 (search 3..=6) with a
//! 64-bit open bound, signed commit bound 4 (`[-8, 7]`), source class
//! `BalancedSignedDigit`, and exactly one balanced base-16 inner digit. It is
//! a separate type only because Akita keys a generated schedule catalog by the
//! configuration type and its family name, and our group layouts are not in
//! Aerie's catalog (see `docs/deviations.md`).
//!
//! Akita enforces the declared interval at commitment admission only. It is
//! not a range proof (Akita Remark 10.41); every range in the protocol is
//! proven inside the PIOP by the digits of spec Section 4.3.

use akita_config::proof_optimized::fp64::{Dense, ExtensionField, Field};
use akita_config::{CommitmentConfig, RingDimensionSchedule, TrustedScheduleCatalog};
use akita_error::AkitaError;
use akita_params::sis::{CommittedSourceClass, CommittedSourceContract};
use akita_params::{
    DecompositionParams, GroupCommitPhaseParams, PolynomialGroupLayout, ScheduleLookupKey,
    SisModulusProfileId,
};

const _: () = assert!(
    akita_params::DEV_PROTOCOL,
    "the catalogs are generated with Akita's dev protocol (workspace feature `dev-protocol`)"
);

/// The base field `F_p`, `p = 2^64 - 59`.
pub type BaseField = Field;
/// The challenge and opening field `K = F_{p^2}`.
pub type OpeningField = ExtensionField;

/// Signed commit bound of every committed table, in bits (`[-8, 7]`).
pub const LOG_COMMIT_BOUND: u32 = 4;
/// Inner (A/source) log basis of the width-4 preset: one balanced base-16 digit.
pub const INNER_LOG_BASIS: u32 = 4;
/// Opening (B/D) decomposition: nominal log basis.
pub const OPENING_LOG_BASIS: u32 = 3;
/// Opening (B/D) decomposition: inclusive search range of the planner.
pub const OPENING_BASIS_RANGE: (u32, u32) = (3, 6);
/// Opening bound in bits (field width).
pub const LOG_OPEN_BOUND: u32 = 64;
/// Schedule family name; it is part of the catalog digest bound in event 0.
pub const WIDTH4_FAMILY: &str = "agg_mldsa_fp64_signed_width4";

/// Fp64 preset for signed width-4 sources in `[-8, 7]`, with its own catalog.
#[derive(Clone, Copy, Debug, Default)]
pub struct AggFp64SignedWidth4;

impl AggFp64SignedWidth4 {
    /// Load and audit the generated schedule catalog once.
    pub fn catalog() -> Result<&'static TrustedScheduleCatalog<Self>, AkitaError> {
        #[cfg(feature = "schedules")]
        {
            static CATALOG: std::sync::OnceLock<
                Result<TrustedScheduleCatalog<AggFp64SignedWidth4>, AkitaError>,
            > = std::sync::OnceLock::new();
            CATALOG
                .get_or_init(|| {
                    TrustedScheduleCatalog::from_artifact_bytes(crate::catalogs::WIDTH4_ARTIFACT)
                })
                .as_ref()
                .map_err(|error| AkitaError::InvalidSetup(error.to_string()))
        }
        #[cfg(not(feature = "schedules"))]
        {
            Err(AkitaError::InvalidSetup(
                "agg-pcs-config schedule catalogs are disabled; enable the `schedules` \
                 feature outside offline regeneration"
                    .to_owned(),
            ))
        }
    }

    /// The scalar (single-group) producer profile of `layout`, as used for a
    /// precommitted group.
    pub fn commitment_profile(
        layout: PolynomialGroupLayout,
    ) -> Result<GroupCommitPhaseParams, AkitaError> {
        Ok(Self::catalog()?
            .resolve_key(&ScheduleLookupKey::single(layout))?
            .profiles()
            .final_group)
    }
}

impl CommitmentConfig for AggFp64SignedWidth4 {
    type Field = Field;
    type ExtField = ExtensionField;

    const RING_DIMENSION_SCHEDULE: RingDimensionSchedule = Dense::RING_DIMENSION_SCHEDULE;

    fn decomposition() -> DecompositionParams {
        DecompositionParams {
            log_basis: OPENING_LOG_BASIS,
            log_commit_bound: LOG_COMMIT_BOUND,
            log_open_bound: Some(LOG_OPEN_BOUND),
        }
    }

    fn ring_challenge_config(
        d: usize,
    ) -> Result<akita_challenges::SparseChallengeConfig, AkitaError> {
        Dense::ring_challenge_config(d)
    }

    fn sis_modulus_profile() -> SisModulusProfileId {
        SisModulusProfileId::Q64Offset59
    }

    fn opening_basis_range() -> (u32, u32) {
        OPENING_BASIS_RANGE
    }

    fn inner_basis_range() -> (u32, u32) {
        (INNER_LOG_BASIS, INNER_LOG_BASIS)
    }

    fn committed_source_class() -> CommittedSourceClass {
        CommittedSourceClass::BalancedSignedDigit
    }

    fn schedule_family_name() -> &'static str {
        WIDTH4_FAMILY
    }
}

/// Inner-digit geometry Akita's fp64 `Dense` source policy admits for a raw
/// balanced signed source of `width` bits (`[-2^(w-1), 2^(w-1)-1]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InnerDigitOption {
    /// Inner log basis (balanced base `2^log_basis` digits).
    pub log_basis: u32,
    /// Number of inner digits per coefficient.
    pub num_digits: usize,
}

/// The inclusive A/source log-basis range Akita's fp64 `Dense` policy
/// searches for a raw source of `width` bits: `(3, min(16, max(width, 3)))`
/// (`akita-planner` `InnerBasisSource::RawCoefficients`).
#[must_use]
pub fn dense_inner_basis_search_range(width: u32) -> (u32, u32) {
    let (min, max) = Dense::inner_basis_range();
    (min, max.min(width.max(min)))
}

/// Every inner-digit option of a raw signed `width`-bit source under the fp64
/// `Dense` policy whose accepted interval is exactly the declared one, i.e.
/// the digits represent every value of `[-2^(w-1), 2^(w-1)-1]`. Returns an
/// error for an invalid width (0 or above the field width).
pub fn admissible_inner_digits(width: u32) -> Result<Vec<InnerDigitOption>, AkitaError> {
    let decomposition = DecompositionParams {
        log_basis: OPENING_LOG_BASIS,
        log_commit_bound: width,
        log_open_bound: Some(LOG_OPEN_BOUND),
    };
    let contract =
        CommittedSourceContract::try_new(CommittedSourceClass::BalancedSignedDigit, decomposition)?;
    let declared = contract.declared_bounds();
    let (lo, hi) = dense_inner_basis_search_range(width);
    let mut options = Vec::new();
    for log_basis in lo..=hi {
        let num_digits = akita_params::sis::num_digits_inner_for_bound(
            DecompositionParams {
                log_basis,
                ..decomposition
            },
            width,
        );
        if contract.accepted_bounds(log_basis, num_digits) == declared {
            options.push(InnerDigitOption {
                log_basis,
                num_digits,
            });
        }
    }
    Ok(options)
}

/// The fewest balanced inner digits Akita's fp64 `Dense` policy can use for a
/// raw signed source of `width` bits (`None` if no option is admissible).
pub fn min_inner_digits(width: u32) -> Result<Option<usize>, AkitaError> {
    Ok(admissible_inner_digits(width)?
        .iter()
        .map(|option| option.num_digits)
        .min())
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_params::sis::num_digits_open;

    #[test]
    fn preset_matches_aerie_width4_parameters() {
        use aerie_pcs_config::configs::Fp64SignedWidth4 as Aerie;
        assert_eq!(
            AggFp64SignedWidth4::decomposition(),
            <Aerie as CommitmentConfig>::decomposition()
        );
        assert_eq!(
            AggFp64SignedWidth4::inner_basis_range(),
            Aerie::inner_basis_range()
        );
        assert_eq!(
            AggFp64SignedWidth4::opening_basis_range(),
            Aerie::opening_basis_range()
        );
        assert_eq!(
            AggFp64SignedWidth4::sis_modulus_profile(),
            Aerie::sis_modulus_profile()
        );
        assert_eq!(
            AggFp64SignedWidth4::committed_source_class(),
            Aerie::committed_source_class()
        );
        assert_eq!(
            AggFp64SignedWidth4::RING_DIMENSION_SCHEDULE,
            Aerie::RING_DIMENSION_SCHEDULE
        );
        assert_ne!(
            AggFp64SignedWidth4::schedule_family_name(),
            Aerie::schedule_family_name()
        );
        let params = AggFp64SignedWidth4::decomposition();
        params.validate().expect("valid decomposition");
        assert_eq!(num_digits_open(params), 22);
        assert!(!AggFp64SignedWidth4::recursive_setup_planning());
    }

    /// Every digit alphabet of spec Section 4.3 (bases 4, 8, 16, balanced), the
    /// Booleans and the padded digits of zero (`-b/2`) lie in `[-8, 7]`.
    #[test]
    fn width4_admits_every_digit_alphabet_and_padding() {
        let contract = AggFp64SignedWidth4::committed_source_contract().expect("contract");
        let (negative, positive) = contract.accepted_bounds(INNER_LOG_BASIS, 1);
        let (negative, positive) = (
            -i128::try_from(negative.expect("bounded")).unwrap(),
            i128::try_from(positive.expect("bounded")).unwrap(),
        );
        assert_eq!((negative, positive), (-8, 7));
        for base in [2_i128, 4, 8, 16] {
            let (lo, hi) = (-base / 2, base / 2 - 1);
            assert!(negative <= lo && hi <= positive, "base {base}");
        }
        for bit in [0, 1] {
            assert!(negative <= bit && bit <= positive);
        }
        // A base-32 digit would not fit (spec Section 4.3: not adopted).
        assert!(-16 < negative);
        assert_eq!(
            admissible_inner_digits(LOG_COMMIT_BOUND).unwrap(),
            vec![
                InnerDigitOption {
                    log_basis: 3,
                    num_digits: 2
                },
                InnerDigitOption {
                    log_basis: 4,
                    num_digits: 1
                },
            ]
        );
    }

    /// Akita admissibility of raw signed widths (risks.md R1): one inner digit
    /// up to 16 bits (`PROOF_OPTIMIZED_INNER_LOG_BASIS_MAX`), two balanced
    /// digits for 17 to 32 bits, so widths 18 to 20 take two digits.
    #[test]
    fn signed_width_admissibility_table() {
        for width in 1..=16 {
            assert_eq!(min_inner_digits(width).unwrap(), Some(1), "width {width}");
        }
        for width in 17..=31 {
            assert_eq!(min_inner_digits(width).unwrap(), Some(2), "width {width}");
        }
        // Two balanced base-2^16 digits reach only 2^31 - 2^15 - 1.
        for width in 32..=47 {
            assert_eq!(min_inner_digits(width).unwrap(), Some(3), "width {width}");
        }
        for width in [18, 19, 20] {
            // Balanced base-2^10 needs three digits at width 20 (511 * 1025 < 2^19 - 1).
            let options = admissible_inner_digits(width).unwrap();
            assert!(options.iter().all(|option| option.num_digits >= 2));
            assert!(options.contains(&InnerDigitOption {
                log_basis: 11,
                num_digits: 2
            }));
        }
        assert_eq!(dense_inner_basis_search_range(4), (3, 4));
        assert_eq!(dense_inner_basis_search_range(20), (3, 16));
        assert!(admissible_inner_digits(0).is_err());
    }
}
