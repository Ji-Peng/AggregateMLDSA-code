//! `agg`: prove and verify aggregate ML-DSA-44 proofs (skeleton).

use std::process::ExitCode;

fn main() -> ExitCode {
    let arg = std::env::args().nth(1);
    match agg_cli::parse_command(arg.as_deref()) {
        Some(cmd) => {
            eprintln!("agg {cmd}: not implemented yet");
            ExitCode::from(2)
        }
        None => {
            eprintln!("{}", agg_cli::USAGE);
            ExitCode::from(1)
        }
    }
}
