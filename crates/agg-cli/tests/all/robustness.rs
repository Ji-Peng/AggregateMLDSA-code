//! The verifier on attacker-controlled bytes, and the span path of the Flock side.
//!
//! - `truncated_and_flipped_aggregates_are_rejected_without_panics`: an honest aggregate of 64
//!   records (ML-DSA-44 and ML-DSA-65) cut at every multiple of 97 bytes, and one byte
//!   flipped in each region of the wire format (header, `c~` list, Akita commitments, Flock
//!   commitment item, Flock proof item, PIOP work nonces, Akita opening). The cuts mostly
//!   fail at the outer frame, so each Akita and Flock item is also cut inside its own frame
//!   (the aggregate re-serialised with the shorter item), which hands the vendored Flock and
//!   Akita deserialisers a short buffer. Every variant must be rejected, and none may panic.
//! - `span_path_round_trips_above_one_segment` (ignored: 8,193 records): the default Flock
//!   layout above one segment, one tied part over all segments (`commit_span` and
//!   `prove_span`), verifies, and a flipped byte of the span's proof item is rejected. Run it
//!   on the benchmark machine at each freeze.
//!
//! Run with `--release`.

use std::panic::{catch_unwind, AssertUnwindSafe};

use agg_cli::mode_a::{AggregateVerifier, Aggregator, AggregatorConfig};
use agg_cli::wire::SegmentCommitments;
use agg_witness::gen::{self, Record};
use agg_witness::params::{ParamSet, Set, Set44, Set65};

fn on_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("test thread");
}

fn records_of<P: ParamSet>(seed: u64, n: usize) -> Vec<Record> {
    if P::SET == Set::S44 {
        gen::records_shared_keys(seed, n, n / 4)
    } else {
        gen::records_shared_keys_for::<P>(seed, n, n / 4)
    }
}

/// Bytes of `items` with their `LE32` frames.
fn framed(items: &[Vec<u8>]) -> usize {
    items.iter().map(|item| 4 + item.len()).sum()
}

fn sweep<P: ParamSet>(seed: u64, n: usize) {
    let records = records_of::<P>(seed, n);
    let signed: Vec<(&[u8], &[u8], &[u8])> =
        records.iter().map(|r| (r.pk.as_slice(), r.msg.as_slice(), r.sig.as_slice())).collect();
    let statements: Vec<(&[u8], &[u8])> = records.iter().map(|r| (r.pk.as_slice(), r.msg.as_slice())).collect();
    let aggregate = Aggregator::new(AggregatorConfig { set: P::SET, ..AggregatorConfig::default() })
        .aggregate(&signed)
        .expect("aggregate");
    let bytes = &aggregate.bytes;
    let proof = &aggregate.proof;
    let verifier = AggregateVerifier::for_set(P::SET);
    let honest = verifier.verify(&statements, bytes, None);
    assert!(honest.accepted(), "{honest}");

    let check = |label: &str, variant: &[u8]| {
        match catch_unwind(AssertUnwindSafe(|| verifier.verify(&statements, variant, None))) {
            Ok(report) => assert!(!report.accepted(), "{} {label}: accepted", P::NAME),
            Err(_) => panic!("{} {label}: the verifier panicked", P::NAME),
        }
    };

    let mut cuts = 0;
    for len in (0..bytes.len()).step_by(97) {
        check(&format!("truncated to {len} B"), &bytes[..len]);
        cuts += 1;
    }

    // One offset inside each region, in wire order.
    let c_tilde = 11 + P::C_TILDE_BYTES * n;
    let akita = c_tilde + 4 + proof.main16_commitment.len() / 2;
    let flock_commitments = c_tilde + 8 + proof.main16_commitment.len() + proof.main_commitment.len() + 4;
    let opening = bytes.len() - 4 - proof.opening.len();
    let work = opening - proof.work.len();
    let flock_proofs = work - framed(&proof.flock_proofs);
    let mut regions = vec![
        ("wire version", 0),
        ("flags", 2),
        ("record count", 3),
        ("c~ list", 11 + P::C_TILDE_BYTES * (n / 2) + 3),
        ("Akita commitment", akita),
        ("Flock commitment item", flock_commitments + 4 + framed(&proof.flock_commitments[..1]) / 2),
        ("Flock proof item", flock_proofs + framed(&proof.flock_proofs) / 2),
        ("Akita opening", opening + 4 + proof.opening.len() / 2),
    ];
    if !proof.work.is_empty() {
        regions.push(("work nonces", work));
    }
    for (label, at) in regions {
        let mut variant = bytes.clone();
        variant[at] ^= 1;
        check(&format!("{label} flipped at {at}"), &variant);
    }

    // Cuts inside one item, the frames kept consistent: the item's own parser sees the short
    // buffer.
    let items: [(&str, fn(&mut agg_piop::wire::AggregateProof) -> &mut Vec<u8>); 4] = [
        ("Akita commitment", |p| &mut p.main16_commitment),
        ("Flock commitment item", |p| &mut p.flock_commitments[0]),
        ("Flock proof item", |p| &mut p.flock_proofs[0]),
        ("Akita opening", |p| &mut p.opening),
    ];
    let mut inner = 0;
    for (label, item) in items {
        let full = item(&mut proof.clone()).len();
        assert!(full > 1, "{label} is not empty");
        for len in [0, 1, full / 2, full - 1] {
            let mut short = proof.clone();
            item(&mut short).truncate(len);
            check(&format!("{label} cut to {len} of {full} B"), &short.to_bytes());
            inner += 1;
        }
    }
    eprintln!(
        "{}: {} B aggregate, {cuts} truncations, {inner} in-item cuts and one flip per region rejected",
        P::NAME,
        bytes.len()
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore = "Flock and Akita proofs: run with --release")]
fn truncated_and_flipped_aggregates_are_rejected_without_panics() {
    on_large_stack(|| {
        sweep::<Set44>(0xf044, 64);
        sweep::<Set65>(0xf065, 64);
    });
}

fn span_round_trip<P: ParamSet>(seed: u64, n: usize) {
    let records = records_of::<P>(seed, n);
    let signed: Vec<(&[u8], &[u8], &[u8])> =
        records.iter().map(|r| (r.pk.as_slice(), r.msg.as_slice(), r.sig.as_slice())).collect();
    let statements: Vec<(&[u8], &[u8])> = records.iter().map(|r| (r.pk.as_slice(), r.msg.as_slice())).collect();
    let aggregate = Aggregator::new(AggregatorConfig { set: P::SET, ..AggregatorConfig::default() })
        .aggregate(&signed)
        .expect("aggregate");
    let proof = &aggregate.proof;
    assert!(proof.flock_commitments.len() >= 2, "more than one segment");
    let item = SegmentCommitments::from_bytes(&proof.flock_commitments[0]).expect("segment 0's item");
    assert_eq!(usize::from(item.span), proof.flock_commitments.len(), "one span over every segment");
    let verifier = AggregateVerifier::for_set(P::SET);
    let report = verifier.verify(&statements, &aggregate.bytes, None);
    assert!(report.accepted(), "{report}");
    assert_eq!(report.parts.len(), 1, "one tied part");

    let mut flipped = proof.clone();
    let middle = flipped.flock_proofs[0].len() / 2;
    flipped.flock_proofs[0][middle] ^= 1;
    let report = verifier.verify(&statements, &flipped.to_bytes(), None);
    assert!(!report.accepted(), "a flipped byte of the span's proof item: {report}");
    assert!(report.rejection.is_some(), "the Flock side rejects: {report}");
}

#[test]
#[ignore = "8,193 records: run with --release --ignored on the benchmark machine"]
fn span_path_round_trips_above_one_segment() {
    on_large_stack(|| {
        span_round_trip::<Set44>(0x5044, 8193);
        span_round_trip::<Set65>(0x5065, 8193);
    });
}
