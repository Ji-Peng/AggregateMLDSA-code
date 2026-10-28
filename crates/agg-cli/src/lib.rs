//! Library half of the `agg` command-line tool: the mode-A aggregator and verifier
//! ([`mode_a`]), the mode-A segment items of the wire format ([`wire`]), the persistent key
//! cache ([`cache`]), the batch files ([`record`]), peak-memory readings ([`rss`]) and the
//! subcommands ([`commands`]).

pub mod cache;
pub mod commands;
pub mod mode_a;
pub mod record;
pub mod rss;
pub mod wire;

pub use commands::{parse_command, Command, USAGE};
