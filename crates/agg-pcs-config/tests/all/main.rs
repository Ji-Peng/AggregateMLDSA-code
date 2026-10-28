//! Integration tests of `agg-pcs-config`, one module per topic, in one test binary.

#[cfg(feature = "schedules")]
mod dummy;
mod akita_fold_support;
mod akita_grinding;
mod commit_open;
mod schedule_catalog_regen;
