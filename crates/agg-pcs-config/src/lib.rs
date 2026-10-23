//! Akita commitment presets and catalogs for the four witness groups
//! (docs/spec.md Sections 4 and 14, milestone M1; docs/pcs-config.md).

pub mod catalogs;
pub mod commit;
pub mod preset;
pub mod work_plan;

/// Re-export of Aerie's preset crate, whose `Fp64SignedWidth4` parameters
/// [`preset::AggFp64SignedWidth4`] repeats.
pub use aerie_pcs_config;
