//! The all-prime-field aggregate (`HashField::Fp`, ML-DSA-44, H-u) through the agg-cli
//! library: round trips at 17 and 100 signatures, the header's hash-field flag, rejection of
//! each hash field's aggregate by a verifier of the other, and the configurations the
//! aggregator and the verifier refuse under `Fp`.
//!
//! Run with `--release`.

use std::time::Instant;

use agg_cli::mode_a::{AggregateError, AggregateVerifier, Aggregator, AggregatorConfig};
use agg_piop::binding::FlockProfile;
use agg_piop::manifest::{HashField, Relation};
use agg_piop::verifier::{ConfigurationError, VerifyError};
use agg_piop::wire::{WireError, HASH_FIELD_FP_FLAG};
use agg_witness::gen::{self, Record};
use agg_witness::params::Set;

fn on_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("test thread");
}

/// Real records: `own` with their own key, `shared` over three shared keys.
fn records(seed: u64, own: usize, shared: usize) -> Vec<Record> {
    let mut records = gen::records(seed, own);
    records.extend(gen::records_shared_keys(seed + 1, shared, 3));
    records
}

fn signed(records: &[Record]) -> Vec<(&[u8], &[u8], &[u8])> {
    records
        .iter()
        .map(|r| (r.pk.as_slice(), r.msg.as_slice(), r.sig.as_slice()))
        .collect()
}

fn statements(records: &[Record]) -> Vec<(&[u8], &[u8])> {
    records
        .iter()
        .map(|r| (r.pk.as_slice(), r.msg.as_slice()))
        .collect()
}

fn fp_config() -> AggregatorConfig {
    AggregatorConfig {
        hash_field: HashField::Fp,
        ..AggregatorConfig::default()
    }
}

/// A rejection for the wrong hash field: the PIOP verifier's header check or the parser's
/// flags error.
fn assert_hash_field_rejection(
    outcome: &Result<agg_piop::verifier::VerifiedAggregate, VerifyError>,
) {
    match outcome {
        Err(VerifyError::HashField { .. } | VerifyError::Wire(WireError::Flags(_))) => {}
        other => panic!("expected a hash-field rejection, got {other:?}"),
    }
}

fn round_trip(seed: u64, own: usize, shared: usize) {
    let records = records(seed, own, shared);
    let n = records.len();
    let start = Instant::now();
    let aggregate = Aggregator::new(fp_config())
        .aggregate(&signed(&records))
        .expect("Fp aggregate");
    let prove = start.elapsed();
    let report = &aggregate.report;
    assert_eq!(report.hash_field, HashField::Fp);
    assert_eq!(report.profile, FlockProfile::FpKeccak);
    assert!(report.parts.is_empty() && report.flock_records.is_zero());
    assert_eq!(aggregate.bytes[2] & HASH_FIELD_FP_FLAG, HASH_FIELD_FP_FLAG);
    let verifier = AggregateVerifier::new().hash_field(HashField::Fp);
    assert_eq!(verifier.hash_field_of(), HashField::Fp);
    let verdict = verifier.verify(&statements(&records), &aggregate.bytes, None);
    assert!(verdict.accepted(), "Fp verifier: {verdict}");
    assert!(verdict.parts.is_empty());
    eprintln!(
        "Fp N = {n}: {} B ({:.1} B/sig), prove {:.2} s, verify {:.1} ms",
        aggregate.bytes.len(),
        report.bytes_per_signature(),
        prove.as_secs_f64(),
        verdict.total().as_secs_f64() * 1e3
    );
}

#[test]
fn fp_round_trip_44_hu() {
    on_large_stack(|| {
        round_trip(42, 13, 4);
        round_trip(43, 80, 20);
    });
}

#[test]
fn fp_rejected_by_gf2_and_gf2_by_fp() {
    on_large_stack(|| {
        let records = records(0xf9, 13, 4);
        let statements = statements(&records);
        let fp = Aggregator::new(fp_config())
            .aggregate(&signed(&records))
            .expect("Fp aggregate");
        let gf2 = Aggregator::new(AggregatorConfig::default())
            .aggregate(&signed(&records))
            .expect("Gf2 aggregate");
        assert_eq!(gf2.bytes[2] & HASH_FIELD_FP_FLAG, 0);
        let by_gf2 = AggregateVerifier::new().verify(&statements, &fp.bytes, None);
        assert_hash_field_rejection(&by_gf2.report.outcome);
        let by_fp = AggregateVerifier::new().hash_field(HashField::Fp).verify(
            &statements,
            &gf2.bytes,
            None,
        );
        assert_hash_field_rejection(&by_fp.report.outcome);
        // The flag alone decides: flipping it on either aggregate is still refused.
        for (bytes, field) in [(&fp.bytes, HashField::Gf2), (&gf2.bytes, HashField::Fp)] {
            let mut flipped = bytes.clone();
            flipped[2] ^= HASH_FIELD_FP_FLAG;
            let verdict =
                AggregateVerifier::new()
                    .hash_field(field)
                    .verify(&statements, &flipped, None);
            assert!(!verdict.accepted(), "flipped flag accepted under {field:?}");
        }
    });
}

#[test]
fn fp_refuses_other_sets_and_relations() {
    on_large_stack(|| {
        let records = records(0xfa, 8, 0);
        let signed = signed(&records);
        let config = AggregatorConfig {
            set: Set::S65,
            ..fp_config()
        };
        assert!(matches!(
            Aggregator::new(config).aggregate(&signed),
            Err(AggregateError::HashFieldSet(Set::S65))
        ));
        let config = AggregatorConfig {
            relation: Relation::HExact,
            ..fp_config()
        };
        assert!(matches!(
            Aggregator::new(config).aggregate(&signed),
            Err(AggregateError::HashFieldRelation(Relation::HExact))
        ));
        let verdict = AggregateVerifier::for_set(Set::S65)
            .hash_field(HashField::Fp)
            .verify_with(
                &statements(&records),
                &[0; 64],
                agg_cli::mode_a::KeyCacheMut::None,
            );
        assert!(matches!(
            verdict.report.outcome,
            Err(VerifyError::Configuration(ConfigurationError::FpKeccakSet(
                Set::S65
            )))
        ));
    });
}
