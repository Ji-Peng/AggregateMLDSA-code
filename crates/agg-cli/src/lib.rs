//! Library half of the `agg` command-line tool: wire format, record binding,
//! verifier caches and the prove/verify commands (docs/spec.md Section 14).

pub mod cache;
pub mod commands;
pub mod record;
pub mod wire;

/// Usage text printed by the `agg` binary.
pub const USAGE: &str = "usage: agg <prove|verify> [options]  (not implemented yet; see docs/spec.md Section 14)";

/// Parses the subcommand name. Returns `None` for anything unknown.
#[must_use]
pub fn parse_command(arg: Option<&str>) -> Option<&'static str> {
    match arg {
        Some("prove") => Some("prove"),
        Some("verify") => Some("verify"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_known_commands_only() {
        assert_eq!(parse_command(Some("prove")), Some("prove"));
        assert_eq!(parse_command(Some("verify")), Some("verify"));
        assert_eq!(parse_command(Some("x")), None);
        assert_eq!(parse_command(None), None);
        // All library crates link into the binary.
        let _ = agg_piop::avx512_enabled();
        let _ = agg_keccak::avx512_enabled();
        let _ = std::any::type_name::<agg_pcs_config::aerie_pcs_config::configs::Fp64SignedWidth4>();
        let _ = agg_bridge::seed::NAME;
    }
}
