//! Akita commitment presets and catalogs for the four witness groups
//! (docs/spec.md Section 14, milestone M1).

pub mod catalogs;
pub mod commit;
pub mod preset;
pub mod work_plan;

/// Re-export of Aerie's preset crate, whose `Fp64SignedWidth4` preset we build on.
pub use aerie_pcs_config;

#[cfg(test)]
mod tests {
    #[test]
    fn aerie_pcs_config_is_importable() {
        // Touch a public module so the dependency is linked, not just declared.
        let name = std::any::type_name::<aerie_pcs_config::configs::Fp64SignedWidth4>();
        assert!(name.contains("Fp64SignedWidth4"));
    }
}
