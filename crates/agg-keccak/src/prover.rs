//! Shared types and helpers for the tied Keccak proof.

use crate::sponge::{self, MU_BYTES};
use bincode::Options as _;
use flock_core::{
    pcs::{ligerito::LigeritoProfile, Commitment},
    union::MIN_DENSE_M,
    verifier::FlockVerifyError,
};
use flock_field::F128;
use std::fmt;

/// Transcript domain of every Keccak sub-proof.
pub const DOMAIN: &[u8] = b"agg-mldsa/v1/keccak";
/// Bytes of one Merkle cap node.
pub(crate) const CAP_NODE_BYTES: usize = 32;
const MAX_PROOF_BYTES: u64 = 64 * 1024 * 1024;

/// Fixed-width little-endian bincode, bounded, no trailing bytes.
pub(crate) fn codec() -> impl bincode::Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(MAX_PROOF_BYTES)
        .reject_trailing_bytes()
}

/// Ligerito profile of the Flock sub-proofs (Flock's own accounting,
/// `flock_core::pcs::ligerito::LigeritoProfile`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Profile {
    /// Flock's `Fast`: rate 1/2, Johnson list-decoding regime with out-of-domain binding,
    /// 16-bit query proof of work, 128 bits work-normalized. The engineering baseline.
    #[default]
    Fast,
    /// Flock's `Secure`: rate 1/2, unique-decoding regime, 120-bit soundness.
    Secure,
    /// Flock's `Slim`: rate 1/4 with the accounting of `Fast` (128 bits work-normalized);
    /// about half the opening bytes for about twice the level-0 encoding work.
    Slim,
    /// `Slim8` of the vendored Flock core: rate 1/8 at level 0 with `Slim`'s
    /// accounting, derived and validated by Flock's own `derive_profile`/`validate`; about
    /// two thirds of `Slim`'s level-0 queries for twice its level-0 encoding work.
    Slim8,
}

impl Profile {
    pub(crate) fn ligerito(self) -> LigeritoProfile {
        match self {
            Self::Fast => LigeritoProfile::Fast,
            Self::Secure => LigeritoProfile::Secure,
            Self::Slim => LigeritoProfile::Slim,
            Self::Slim8 => LigeritoProfile::Slim8,
        }
    }

    /// `floor(L_max)`, the Johnson-list size of a level-0 Ligerito commitment
    /// (the Flock paper's Johnson bound at `eta = 0.02`): 35, 50, 70 for
    /// `Fast`, `Slim`, `Slim8` (rates 1/2, 1/4, 1/8); 1 for the unique-decoding `Secure`.
    #[must_use]
    pub fn list_size(self) -> u64 {
        match self {
            Self::Secure => 1,
            _ => agg_bridge::pow::list_size(self.ligerito().log_inv_rate() as u32),
        }
    }

    /// The profile byte of the proof headers: 0 `Fast`, 1 `Secure`, 3 `Slim`, 4 `Slim8` (2 is
    /// unassigned).
    #[must_use]
    pub fn id(self) -> u8 {
        match self {
            Self::Fast => 0,
            Self::Secure => 1,
            Self::Slim => 3,
            Self::Slim8 => 4,
        }
    }

    /// The profile of a header byte.
    #[must_use]
    pub fn from_id(id: u8) -> Option<Self> {
        match id {
            0 => Some(Self::Fast),
            1 => Some(Self::Secure),
            3 => Some(Self::Slim),
            4 => Some(Self::Slim8),
            _ => None,
        }
    }
}

/// Errors of the verifier and of proof parsing.
#[derive(Debug)]
pub enum KeccakError {
    /// Bytes are not a proof of this format, profile or instance count.
    Malformed(&'static str),
    /// The statement does not have the shape the proof claims.
    Statement(&'static str),
    /// Flock rejected the proof.
    Flock(FlockVerifyError),
    /// A tie between two instances failed (`crate::tied`): the named claim's value does
    /// not match the other instance's or the verifier's table.
    Tie(&'static str),
    /// The prover's proof-of-work search exhausted its allowance of `2^(g + 7)` candidates
    /// (probability below `2^-184.6` per site).
    Work(&'static str),
}

impl fmt::Display for KeccakError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Malformed(m) => write!(f, "malformed proof: {m}"),
            Self::Statement(m) => write!(f, "bad statement: {m}"),
            Self::Flock(e) => write!(f, "flock verifier rejected: {e:?}"),
            Self::Tie(m) => write!(f, "tie claim mismatch: {m}"),
            Self::Work(m) => write!(f, "proof-of-work allowance exhausted: {m}"),
        }
    }
}

impl std::error::Error for KeccakError {}

/// A serialized Flock union-circuit proof (header, then the proof body). It
/// does not contain the commitment: the verifier takes that from the PIOP
/// transcript (the commitment phase).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeccakProof {
    pub bytes: Vec<u8>,
}

/// The Flock commitment of one sub-proof: the Ligerito level-0 Merkle cap,
/// `2^c` nodes of 32 bytes (`c = PcsParams::level_zero_cap_depth`), which is
/// exactly what Flock's `bind_statement` absorbs (the PCS parameters are not
/// transmitted; both sides derive them from `n` and the profile).
/// [`Self::as_bytes`] is the commitment-phase payload of the PIOP transcript.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KeccakCommitment {
    bytes: Vec<u8>,
}

impl KeccakCommitment {
    /// Wrap received bytes (the commitment-phase payload). The length is checked
    /// against the sub-proof's parameters by the verifier.
    #[must_use]
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    /// The canonical encoding: the cap nodes in order, 32 bytes each.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub(crate) fn from_flock(c: &Commitment) -> Self {
        Self {
            bytes: c.cap.as_flattened().to_vec(),
        }
    }
}

/// Smallest full-height count for which the opening's block-first transport is kept.
const RECT_MIN_ROWS_LOG: usize = 11;

/// Row-capacity `log2` for `n` instances: `ceil(log2 n)`, at least 3 (Flock's
/// lincheck stripe) and at least `MIN_DENSE_M - k_log` (smallest Ligerito
/// config). A full-height stack (`n = 2^nu`) makes Flock's merged opening take
/// its block-first transport, whose cost is about `128^2` field operations per
/// used chunk-column (2,146 for the ML-DSA-44 sponge gate) regardless of `n`; below
/// `2^11` rows that fixed cost exceeds the dense path's, so `nu` is raised by
/// one. A function of `n` only, so prover and verifier agree.
#[must_use]
pub fn capacity_log(n: usize, k_log: usize) -> usize {
    let mut nu = (n.next_power_of_two().trailing_zeros() as usize)
        .max(3)
        .max(MIN_DENSE_M.saturating_sub(k_log));
    if n == 1 << nu && nu < RECT_MIN_ROWS_LOG {
        nu += 1;
    }
    nu
}

/// Public words of one sponge record: `mu` (4 words) then `c~` (`c~` bytes / 16 words: 2, 3
/// or 4).
#[must_use]
pub fn sponge_public_words(mu: &[u8; MU_BYTES], c_tilde: &[u8]) -> Vec<F128> {
    let mut w = sponge::bytes_to_words(mu);
    w.extend(sponge::bytes_to_words(c_tilde));
    w
}

#[cfg(test)]
mod tests {
    use super::*;
    /// The vendored Flock core's outer sites carry the level-0 Johnson list.
    #[test]
    fn flock_outer_sites_carry_the_level0_list() {
        use flock_core::{
            lincheck::LincheckGrinding, pcs::ligerito::embedded_security_config,
            pcs::OpeningGrinding, zerocheck::ZerocheckGrinding,
        };
        for (profile, list) in [
            (Profile::Fast, 36),
            (Profile::Slim, 50),
            (Profile::Slim8, 71),
        ] {
            let ligerito = profile.ligerito();
            assert_eq!(ligerito.l0_list_factor(), list, "{profile:?}");
            assert_eq!(profile.list_size(), list as u64 - u64::from(list != 50));
            // The list factor assumes the shipped level-0 slack eta = 0.02.
            let mut seen = 0;
            for m in 22..=35 {
                if let Some(toml) = embedded_security_config(m, ligerito) {
                    let eta = toml
                        .lines()
                        .find(|l| l.trim_start().starts_with("eta"))
                        .unwrap();
                    assert_eq!(
                        eta.split('=').nth(1).unwrap().trim(),
                        "0.02",
                        "m{m} {profile:?}"
                    );
                    seen += 1;
                }
            }
            assert!(seen >= 10, "{profile:?}: {seen} configs");
        }
        // The pinned per-challenge grinding of Fast / Slim / Slim8.
        let z = ZerocheckGrinding::per_challenge_128_listed;
        assert_eq!(
            [36, 50, 71].map(|l| z(l).multilinear_round_bits()),
            [Some(7), Some(7), Some(8)]
        );
        assert_eq!(
            [36, 50, 71].map(|l| z(l).skip_bits()),
            [Some(13), Some(13), Some(14)]
        );
        assert_eq!(
            [36, 50, 71].map(|l| z(l).initial_bits(30)),
            [Some(11), Some(11), Some(12)]
        );
        let lc = LincheckGrinding::per_challenge_128_listed;
        assert_eq!(
            [36, 50, 71].map(|l| lc(l).alpha_bits()),
            [Some(6), Some(6), Some(7)]
        );
        assert_eq!(
            [36, 50, 71].map(|l| lc(l).multilinear_round_bits()),
            [Some(7), Some(7), Some(8)]
        );
        assert_eq!(
            [36, 50, 71].map(|l| lc(l).skip_bits(6)),
            [Some(12), Some(12), Some(13)]
        );
        let op = OpeningGrinding::per_challenge_128_listed;
        assert_eq!([36, 50, 71].map(|l| op(l).ring_switch_bits), [8, 9, 9]);
        assert_eq!([36, 50, 71].map(|l| op(l).claim_batch_bits), [6, 6, 7]);
        assert_eq!([36, 50, 71].map(|l| op(l).merged_round_bits), [7, 7, 8]);
        // List 1 is upstream's schedule.
        assert_eq!(
            ZerocheckGrinding::per_challenge_128_listed(1),
            ZerocheckGrinding::per_challenge_128()
        );
        assert_eq!(LincheckGrinding::per_challenge_128().skip_bits(6), Some(6));
        assert_eq!(
            OpeningGrinding::per_challenge_128_listed(1).ring_switch_bits,
            3
        );
    }
}
