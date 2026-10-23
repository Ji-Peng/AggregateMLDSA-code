//! Benchmark harness: batch generation, timing with medians, git revision stamp.

use std::time::{Duration, Instant};

use agg_witness::gen::{self, Record};
use agg_witness::params::C_TILDE_BYTES;

/// A batch of honest ML-DSA-44 records with their prover inputs.
pub struct Batch {
    /// The records (own keys, then records under a few shared keys).
    pub records: Vec<Record>,
}

impl Batch {
    /// `own` records under their own keys and `shared` records under three shared keys
    /// (`agg_witness::gen`, deterministic in `seed`).
    #[must_use]
    pub fn generate(seed: u64, own: usize, shared: usize) -> Self {
        let mut records = gen::records(seed, own);
        records.extend(gen::records_shared_keys(seed + 1, shared, 3));
        Self { records }
    }

    /// Live records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// True when empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// `(pk, M, c~)` per record.
    #[must_use]
    pub fn inputs(&self) -> Vec<(&[u8], &[u8], [u8; C_TILDE_BYTES])> {
        self.records
            .iter()
            .map(|r| {
                let c_tilde: [u8; C_TILDE_BYTES] =
                    r.sig[..C_TILDE_BYTES].try_into().expect("signature prefix");
                (r.pk.as_slice(), r.msg.as_slice(), c_tilde)
            })
            .collect()
    }

    /// The signatures.
    #[must_use]
    pub fn signatures(&self) -> Vec<&[u8]> {
        self.records.iter().map(|r| r.sig.as_slice()).collect()
    }

    /// `(pk, M)` per record (what the verifier is given).
    #[must_use]
    pub fn public(&self) -> Vec<(&[u8], &[u8])> {
        self.records
            .iter()
            .map(|r| (r.pk.as_slice(), r.msg.as_slice()))
            .collect()
    }
}

/// Runs `f` and returns its result with the wall time.
pub fn timed<T>(f: impl FnOnce() -> T) -> (T, Duration) {
    let start = Instant::now();
    let out = f();
    (out, start.elapsed())
}

/// Median of durations (mean of the middle two for an even count).
#[must_use]
pub fn median_duration(samples: &[Duration]) -> Option<Duration> {
    let mut seconds: Vec<f64> = samples.iter().map(Duration::as_secs_f64).collect();
    crate::median(&mut seconds).map(Duration::from_secs_f64)
}

/// `git rev-parse --short=12 HEAD` of the working directory, with ` (dirty)` when the
/// tree has local changes; `unknown` outside a repository.
#[must_use]
pub fn git_revision() -> String {
    let run = |args: &[&str]| {
        std::process::Command::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
    };
    let Some(head) = run(&["rev-parse", "--short=12", "HEAD"]) else {
        return "unknown".into();
    };
    let mut revision = String::from_utf8_lossy(&head.stdout).trim().to_string();
    if run(&["diff", "--quiet", "HEAD"]).is_none() {
        revision.push_str(" (dirty)");
    }
    revision
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_inputs_and_medians() {
        let batch = Batch::generate(3, 2, 2);
        assert_eq!(batch.len(), 4);
        assert!(!batch.is_empty());
        let inputs = batch.inputs();
        assert_eq!(inputs[1].2[..], batch.records[1].sig[..C_TILDE_BYTES]);
        assert_eq!(batch.public()[3].1, batch.records[3].msg.as_slice());
        assert_eq!(batch.signatures().len(), 4);
        let ms = Duration::from_millis;
        assert_eq!(median_duration(&[ms(5), ms(1), ms(3)]), Some(ms(3)));
        assert_eq!(median_duration(&[]), None);
        let (value, elapsed) = timed(|| 7);
        assert_eq!(value, 7);
        assert!(elapsed < Duration::from_secs(1));
        assert!(!git_revision().is_empty());
    }
}
