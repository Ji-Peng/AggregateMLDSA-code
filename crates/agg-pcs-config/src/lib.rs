//! Akita commitment presets and catalogs for the seven witness groups
//! (`docs/spec.md`, `docs/pcs-config.md`).

pub mod catalogs;
pub mod commit;
pub mod preset;
pub mod sched;
pub mod work_plan;

/// Re-export of Aerie's preset crate, whose `Fp64SignedWidth4` parameters
/// [`preset::AggFp64SignedWidth4`] repeats.
pub use aerie_pcs_config;
