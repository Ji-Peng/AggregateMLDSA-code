//! Regenerate the `Slim8` Ligerito security configs of the vendored Flock core
//! (`vendor/flock-core/configs/ligerito/m{22..35}_slim8.toml`, deviation K18) with Flock's
//! own derivation and validator, and check that the same derivation reproduces the shipped
//! `Slim` files byte for byte.
//!
//! `cargo run --release -p agg-keccak --example gen_slim8_configs -- <configs dir>`

use flock_core::pcs::ligerito::{LigeritoProfile, LigeritoSecurityConfig};

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "vendor/flock-core/configs/ligerito".to_string());
    for m in 22..=35 {
        let slim = LigeritoSecurityConfig::derive_profile(m, LigeritoProfile::Slim)
            .and_then(|c| c.to_toml_string())
            .expect("derive Slim");
        let shipped = std::fs::read_to_string(format!("{dir}/m{m}_slim.toml")).expect("read");
        assert_eq!(
            slim, shipped,
            "m{m}: the derivation reproduces the shipped Slim config"
        );
        let cfg = LigeritoSecurityConfig::derive_profile(m, LigeritoProfile::Slim8)
            .expect("derive Slim8");
        cfg.validate().expect("Slim8 validates");
        let toml = cfg.to_toml_string().expect("serialize");
        std::fs::write(format!("{dir}/m{m}_slim8.toml"), &toml).expect("write");
        let queries: Vec<usize> = cfg.levels.iter().map(|l| l.queries).collect();
        let rates: Vec<usize> = cfg.levels.iter().map(|l| l.log_inv_rate).collect();
        println!(
            "m{m}: initial_k {} rates {rates:?} queries {queries:?}",
            cfg.initial_k
        );
    }
}
