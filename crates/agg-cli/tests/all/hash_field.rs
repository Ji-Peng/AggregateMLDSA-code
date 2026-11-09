//! The all-prime-field aggregate (`HashField::Fp`, ML-DSA-44, H-exact and H-FIPS) through the
//! agg-cli library: round trips at 17, 100 and 1,024 signatures (H-exact) and at 32 signatures
//! and on the weight-omega corner records (H-FIPS), the header's hash-field flag,
//! rejection of each hash field's aggregate by a verifier of the other, the configurations
//! the aggregator and the verifier refuse under `Fp`, the wire round trip, rejection of a
//! flipped byte in each region of the wire format, and byte identity of the `agg` binary's
//! output across thread counts and ISA tiers.
//!
//! Run with `--release`.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::OnceLock;
use std::time::Instant;

use agg_cli::mode_a::{AggregateError, AggregateVerifier, Aggregator, AggregatorConfig};
use agg_piop::binding::FlockProfile;
use agg_piop::manifest::{HashField, Relation};
use agg_piop::verifier::VerifyError;
use agg_piop::wire::{AggregateProof, WireError, HASH_FIELD_FP_FLAG};
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

/// The `Fp` aggregator configuration under `relation`.
fn fp_config_for(relation: Relation) -> AggregatorConfig {
    AggregatorConfig {
        hash_field: HashField::Fp,
        relation,
        ..AggregatorConfig::default()
    }
}

fn fp_config() -> AggregatorConfig {
    fp_config_for(Relation::HExact)
}

/// The `Fp` verifier of `relation`.
fn fp_verifier_for(relation: Relation) -> AggregateVerifier {
    AggregateVerifier::new()
        .relation(relation)
        .hash_field(HashField::Fp)
}

/// The `Fp` verifier of [`fp_config`]'s relation.
fn fp_verifier() -> AggregateVerifier {
    fp_verifier_for(Relation::HExact)
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
    round_trip_records(Relation::HExact, &records(seed, own, shared));
}

/// Aggregate `records` under `Fp` and `relation`, check the report and the header, verify the
/// aggregate with the verifier of `relation`, and return its bytes.
fn round_trip_records(relation: Relation, records: &[Record]) -> Vec<u8> {
    let n = records.len();
    let start = Instant::now();
    let aggregate = Aggregator::new(fp_config_for(relation))
        .aggregate(&signed(records))
        .expect("Fp aggregate");
    let prove = start.elapsed();
    let report = &aggregate.report;
    assert_eq!(report.hash_field, HashField::Fp);
    assert_eq!(report.profile, FlockProfile::FpKeccak);
    assert!(report.parts.is_empty() && report.flock_records.is_zero());
    assert_eq!(aggregate.bytes[2] & HASH_FIELD_FP_FLAG, HASH_FIELD_FP_FLAG);
    assert_eq!(aggregate.bytes[2] & 0x07, relation.id());
    let verifier = fp_verifier_for(relation);
    assert_eq!(verifier.hash_field_of(), HashField::Fp);
    let verdict = verifier.verify(&statements(records), &aggregate.bytes, None);
    assert!(verdict.accepted(), "Fp verifier: {verdict}");
    assert!(verdict.parts.is_empty());
    eprintln!(
        "Fp {relation} N = {n}: {} B ({:.1} B/sig), prove {:.2} s, verify {:.1} ms",
        aggregate.bytes.len(),
        report.bytes_per_signature(),
        prove.as_secs_f64(),
        verdict.total().as_secs_f64() * 1e3
    );
    aggregate.bytes
}

/// The verifier of the other relation rejects an `Fp` aggregate before event 0.
fn assert_other_relation_rejects(relation: Relation, records: &[Record], bytes: &[u8]) {
    let other = match relation {
        Relation::HExact => Relation::HFips,
        Relation::HFips => Relation::HExact,
    };
    let verdict = fp_verifier_for(other).verify(&statements(records), bytes, None);
    assert!(
        matches!(
            verdict.report.outcome,
            Err(VerifyError::Relation { expected, found }) if expected == other && found == relation
        ),
        "{:?}",
        verdict.report.outcome
    );
}

#[test]
fn fp_round_trip_44_hexact() {
    on_large_stack(|| {
        round_trip(42, 13, 4);
        round_trip(43, 80, 20);
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore = "1,024 records: run with --release")]
fn fp_round_trip_44_hexact_1024() {
    on_large_stack(|| round_trip(42, 768, 256));
}

/// H-FIPS (the default relation) under `Fp` at 32 signatures: the default configuration with
/// only the hash field set proves H-FIPS, the H-FIPS verifier accepts, and the H-exact
/// verifier rejects the aggregate on its relation byte.
#[test]
fn fp_round_trip_44_hfips() {
    on_large_stack(|| {
        assert_eq!(
            AggregatorConfig { hash_field: HashField::Fp, ..AggregatorConfig::default() }.relation,
            Relation::HFips
        );
        let records = records(0x44f, 24, 8);
        let bytes = round_trip_records(Relation::HFips, &records);
        assert_other_relation_rejects(Relation::HFips, &records, &bytes);
    });
}

/// H-FIPS under `Fp` on the two weight-80 records of `agg_witness::gen::hint_corner_record_for`
/// alone (`N = 2`, padded to the smallest catalog batch `N* = 8`): each has the coefficient
/// `w' = q - gamma2, h = 0` and the `e = -1` wrap to `w1' = 43`, so its hint slack is 0. The
/// aggregate verifies, and the H-exact verifier rejects it.
#[test]
fn fp_hint_corner_44_hfips() {
    use agg_witness::params::Set44;
    on_large_stack(|| {
        let records: Vec<Record> = (1..=2).map(gen::hint_corner_record_for::<Set44>).collect();
        let bytes = round_trip_records(Relation::HFips, &records);
        assert_other_relation_rejects(Relation::HFips, &records, &bytes);
    });
}

/// The 17-record `Fp` aggregate shared by the wire and flipped-byte tests: the records and
/// the aggregate bytes.
fn fp_17() -> &'static (Vec<Record>, Vec<u8>) {
    static AGGREGATE: OnceLock<(Vec<Record>, Vec<u8>)> = OnceLock::new();
    AGGREGATE.get_or_init(|| {
        let records = records(0xfb, 13, 4);
        let aggregate = Aggregator::new(fp_config())
            .aggregate(&signed(&records))
            .expect("Fp aggregate");
        (records, aggregate.bytes)
    })
}

#[test]
fn fp_wire_round_trip() {
    on_large_stack(|| {
        let (_, bytes) = fp_17();
        let proof = AggregateProof::from_bytes_for(bytes, HashField::Fp).expect("parse Fp");
        assert_eq!(proof.hash_field, HashField::Fp);
        assert_eq!(&proof.to_bytes(), bytes);
        assert_eq!(proof.size().total(), bytes.len());
        assert_eq!(
            AggregateProof::from_bytes(bytes),
            Err(WireError::Flags(bytes[2]))
        );
    });
}

#[test]
fn fp_flipped_byte_rejected() {
    on_large_stack(|| {
        let (records, bytes) = fp_17();
        let statements = statements(records);
        let proof = AggregateProof::from_bytes_for(bytes, HashField::Fp).expect("parse Fp");
        let size = proof.size();
        let verifier = fp_verifier();
        assert!(verifier.verify(&statements, bytes, None).accepted());
        let check = |label: &str, variant: &[u8]| match catch_unwind(AssertUnwindSafe(|| {
            verifier.verify(&statements, variant, None)
        })) {
            Ok(report) => {
                assert!(!report.accepted(), "{label}: accepted");
                eprintln!("{label}: {:?}", report.report.outcome.as_ref().err());
            }
            Err(_) => panic!("{label}: the verifier panicked"),
        };
        // One offset inside each region, in wire order. The Akita commitments open with the
        // two framed lattice commitments, then the seven framed Keccak commitments; the
        // stage-1 rounds follow the lookup (an `Fp` proof has no pre-stage claims).
        let c_tilde = size.header;
        let keccak = c_tilde
            + size.c_tilde
            + 8
            + proof.main16_commitment.len()
            + proof.main_commitment.len()
            + 4;
        let stage1 = c_tilde + size.c_tilde + size.akita_commitments + size.lookup;
        let work = bytes.len() - size.opening - size.work;
        let first = &proof.keccak_commitments[0];
        assert_eq!(&bytes[keccak..keccak + first.len()], first.as_slice());
        assert_eq!(
            stage1 + size.piop,
            work,
            "the stages end where the nonces start"
        );
        assert_eq!(&bytes[work..work + size.work], proof.work.as_slice());
        let regions = [
            ("c~ byte", c_tilde + 5),
            (
                "round-1 Keccak commitment",
                keccak + proof.keccak_commitments[0].len() / 2,
            ),
            ("stage-1 round element", stage1),
            ("work nonce", work),
            ("last opening byte", bytes.len() - 1),
        ];
        for (label, at) in regions {
            for mask in [0x01, 0x80] {
                let mut variant = bytes.clone();
                variant[at] ^= mask;
                check(
                    &format!("{label} (offset {at}, mask {mask:#04x})"),
                    &variant,
                );
            }
        }
        for eighth in 0..8 {
            let len = bytes.len() * eighth / 8;
            check(&format!("truncated to {len} B"), &bytes[..len]);
        }
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
        let gf2 = Aggregator::new(AggregatorConfig {
            relation: Relation::HExact,
            ..AggregatorConfig::default()
        })
        .aggregate(&signed(&records))
        .expect("Gf2 aggregate");
        assert_eq!(gf2.bytes[2] & HASH_FIELD_FP_FLAG, 0);
        let by_gf2 = AggregateVerifier::new()
            .relation(Relation::HExact)
            .verify(&statements, &fp.bytes, None);
        assert_hash_field_rejection(&by_gf2.report.outcome);
        let by_fp = fp_verifier().verify(&statements, &gf2.bytes, None);
        assert_hash_field_rejection(&by_fp.report.outcome);
        // The flag alone decides: flipping it on either aggregate is still refused.
        for (bytes, field) in [(&fp.bytes, HashField::Gf2), (&gf2.bytes, HashField::Fp)] {
            let mut flipped = bytes.clone();
            flipped[2] ^= HASH_FIELD_FP_FLAG;
            let verdict = AggregateVerifier::new()
                .relation(Relation::HExact)
                .hash_field(field)
                .verify(&statements, &flipped, None);
            assert!(!verdict.accepted(), "flipped flag accepted under {field:?}");
        }
    });
}

/// Under `Fp`, as under `Gf2`, an ML-DSA-65 or ML-DSA-87 batch holds at least
/// `LANE_MIN_RECORDS` records; the count is checked before the records are read.
#[test]
fn fp_lane_sets_refuse_short_batches() {
    on_large_stack(|| {
        let records = records(0xfa, 8, 0);
        let signed = signed(&records);
        for set in [Set::S65, Set::S87] {
            let config = AggregatorConfig {
                set,
                ..fp_config()
            };
            assert!(matches!(
                Aggregator::new(config).aggregate(&signed),
                Err(AggregateError::Count(8))
            ));
        }
    });
}

fn agg(args: &[&str]) -> (i32, String) {
    agg_env(args, &[])
}

fn agg_env(args: &[&str], env: &[(&str, &str)]) -> (i32, String) {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_agg"))
        .args(args)
        .envs(env.iter().copied())
        .output()
        .expect("run agg");
    (
        out.status.code().unwrap_or(-1),
        format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ),
    )
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Flock and Akita proofs: run with --release"
)]
fn cli_hash_field_fp_round_trip() {
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("agg-cli-hash-field");
    std::fs::create_dir_all(&dir).unwrap();
    let keys = dir.join("keys.bin");
    let records = dir.join("records.bin");
    let aggregate = dir.join("aggregate.bin");
    let json = dir.join("bench.json");
    let path = |p: &std::path::Path| p.to_str().unwrap().to_owned();
    let (keys, records, aggregate, json) =
        (path(&keys), path(&records), path(&aggregate), path(&json));
    let run = |args: &[&str]| {
        let (code, out) = agg(args);
        assert_eq!(code, 0, "{out}");
    };
    run(&[
        "keygen-batch",
        "--params",
        "44",
        "--count",
        "17",
        "--seed",
        "42",
        "--out",
        &keys,
    ]);
    run(&[
        "sign-batch",
        "--params",
        "44",
        "--keys",
        &keys,
        "--count",
        "17",
        "--seed",
        "43",
        "--out",
        &records,
    ]);
    run(&[
        "aggregate",
        "--records",
        &records,
        "--out",
        &aggregate,
        "--hash-field",
        "fp",
        "--relation",
        "h-exact",
    ]);
    run(&[
        "verify",
        "--records",
        &records,
        "--aggregate",
        &aggregate,
        "--hash-field",
        "fp",
        "--relation",
        "h-exact",
    ]);
    assert_ne!(
        agg(&["verify", "--records", &records, "--aggregate", &aggregate, "--relation", "h-exact"]).0,
        0
    );
    // Without --relation, fp proves the default relation H-FIPS; its verifier needs no
    // --relation either, and the H-exact verifier rejects the aggregate.
    run(&["aggregate", "--records", &records, "--out", &aggregate, "--hash-field", "fp"]);
    run(&["verify", "--records", &records, "--aggregate", &aggregate, "--hash-field", "fp"]);
    let (code, out) = agg(&[
        "verify", "--records", &records, "--aggregate", &aggregate, "--hash-field", "fp", "--relation", "h-exact",
    ]);
    assert_ne!(code, 0, "{out}");
    for command in ["aggregate", "verify", "bench"] {
        for (option, value) in [
            ("--flock-profile", "slim"),
            ("--flock-span", "on"),
            ("--part-log2", "13"),
        ] {
            let (code, out) = agg(&[command, "--hash-field", "fp", option, value]);
            assert_ne!(code, 0, "{out}");
            assert!(
                out.contains(option) && out.contains("--hash-field"),
                "{out}"
            );
        }
    }
    let (code, out) = agg_env(
        &[
            "bench",
            "--count",
            "17",
            "--warmup",
            "0",
            "--reps",
            "1",
            "--params",
            "44",
            "--hash-field",
            "fp",
            "--relation",
            "h-exact",
            "--json",
            &json,
        ],
        &[("AGG_FLOCK_THREADS", "1")],
    );
    assert_eq!(code, 0, "{out}");
    assert!(
        out.contains("hash field fp") && out.contains("AGG_FLOCK_THREADS is ignored"),
        "{out}"
    );
    assert!(std::fs::read_to_string(json)
        .unwrap()
        .contains("\"hash_field\": \"fp\""));
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Flock and Akita proofs: run with --release"
)]
fn fp_single_thread_byte_identical() {
    let dir =
        std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("agg-cli-hash-field-threads");
    std::fs::create_dir_all(&dir).unwrap();
    let path = |name: &str| dir.join(name).to_str().unwrap().to_owned();
    let (keys, records) = (path("keys.bin"), path("records.bin"));
    let run = |args: &[&str], env: &[(&str, &str)]| {
        let (code, out) = agg_env(args, env);
        assert_eq!(code, 0, "{out}");
    };
    run(
        &[
            "keygen-batch",
            "--params",
            "44",
            "--count",
            "25",
            "--seed",
            "42",
            "--out",
            &keys,
        ],
        &[],
    );
    run(
        &[
            "sign-batch",
            "--params",
            "44",
            "--keys",
            &keys,
            "--count",
            "100",
            "--seed",
            "43",
            "--out",
            &records,
        ],
        &[],
    );
    let aggregate = |name: &str, env: &[(&str, &str)]| {
        let out = path(name);
        let start = Instant::now();
        run(
            &[
                "aggregate",
                "--records",
                &records,
                "--out",
                &out,
                "--hash-field",
                "fp",
                "--relation",
                "h-exact",
            ],
            env,
        );
        eprintln!(
            "Fp N = 100, {env:?}: aggregate {:.2} s",
            start.elapsed().as_secs_f64()
        );
        std::fs::read(out).unwrap()
    };
    let pooled = aggregate("pooled.bin", &[]);
    assert_eq!(
        aggregate("single.bin", &[("RAYON_NUM_THREADS", "1")]),
        pooled
    );
    // A capped run picks other kernels on a CPU with AVX-512; the bytes must not change.
    if std::arch::is_x86_feature_detected!("avx2") {
        assert_eq!(aggregate("avx2.bin", &[("AGG_ISA", "avx2")]), pooled);
    }
    run(
        &[
            "verify",
            "--records",
            &records,
            "--aggregate",
            &path("pooled.bin"),
            "--hash-field",
            "fp",
            "--relation",
            "h-exact",
        ],
        &[],
    );
}
