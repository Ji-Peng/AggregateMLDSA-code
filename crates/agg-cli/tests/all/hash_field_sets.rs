//! The all-prime-field aggregate (`HashField::Fp`) for every parameter set and relation through
//! the agg-cli library and the `agg` binary: round trips of ML-DSA-44, -65 and -87 under H-exact
//! and H-FIPS at 32 signatures and of ML-DSA-87 under H-FIPS at 100 signatures (padded to 128),
//! the weight-omega corner records of each set under H-FIPS at the smallest batch the set
//! admits, the header (hash-field flag, relation, set) and the `c~` list of 32, 48 or 64 bytes
//! per record, and rejection of an `Fp` aggregate by a verifier of another set, relation or
//! hash field. Run with `--release`.

use std::time::Instant;

use agg_cli::mode_a::{AggregateVerifier, Aggregator, AggregatorConfig, KeyCacheMut};
use agg_piop::binding::FlockProfile;
use agg_piop::lane_public::LaneKeyCache;
use agg_piop::manifest::{HashField, Relation};
use agg_piop::verifier::VerifyError;
use agg_piop::wire::{c_tilde_bytes, set_of_nibble, HASH_FIELD_FP_FLAG};
use agg_witness::gen::{self, Record};
use agg_witness::params::{ParamSet, Set, Set44, Set65, Set87};

fn on_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("test thread");
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

/// The `Fp` verifier of `set` and `relation`.
fn fp_verifier(set: Set, relation: Relation) -> AggregateVerifier {
    AggregateVerifier::for_set(set)
        .relation(relation)
        .hash_field(HashField::Fp)
}

const fn other_relation(relation: Relation) -> Relation {
    match relation {
        Relation::HExact => Relation::HFips,
        Relation::HFips => Relation::HExact,
    }
}

/// Aggregate `records` of `P` under `Fp` and `relation`; check the report, the header and the
/// `c~` list; verify with the set's `Fp` verifier of `relation` (for ML-DSA-65 and ML-DSA-87
/// also with a key cache, filled and then reused); check that the verifiers of the other
/// relation, of each other set and of the other hash field reject the aggregate.
fn round_trip<P: ParamSet>(relation: Relation, records: &[Record]) {
    let n = records.len();
    let config = AggregatorConfig {
        set: P::SET,
        relation,
        hash_field: HashField::Fp,
        ..AggregatorConfig::default()
    };
    let start = Instant::now();
    let aggregate = Aggregator::new(config)
        .aggregate(&signed(records))
        .expect("Fp aggregate");
    let prove = start.elapsed();
    let report = &aggregate.report;
    assert_eq!(report.hash_field, HashField::Fp);
    assert_eq!(report.profile, FlockProfile::FpKeccak);
    assert_eq!(report.relation, relation);
    assert!(report.parts.is_empty() && report.flock_records.is_zero());
    let bytes = &aggregate.bytes;
    assert_eq!(bytes[2] & HASH_FIELD_FP_FLAG, HASH_FIELD_FP_FLAG);
    assert_eq!(bytes[2] & 0x07, relation.id());
    assert_eq!(set_of_nibble(bytes[2] >> 4), Some(P::SET));
    // The c~ list follows the header: the signatures' first `C_TILDE_BYTES` bytes, in order.
    let width = c_tilde_bytes(P::SET);
    assert_eq!(width, P::C_TILDE_BYTES);
    assert_eq!(report.size.c_tilde, n * width);
    let list = &bytes[report.size.header..report.size.header + n * width];
    for (i, r) in records.iter().enumerate() {
        assert_eq!(&list[i * width..(i + 1) * width], &r.sig[..width], "c~ of record {i}");
    }

    let statements = statements(records);
    let verifier = fp_verifier(P::SET, relation);
    let start = Instant::now();
    let verdict = verifier.verify_with(&statements, bytes, KeyCacheMut::None);
    let verify = start.elapsed();
    assert!(verdict.accepted(), "Fp verifier: {verdict}");
    assert!(verdict.parts.is_empty() && verdict.rejection.is_none());
    assert_eq!(verdict.profile, FlockProfile::FpKeccak);
    if P::SET != Set::S44 {
        let mut cache = LaneKeyCache::new(P::SET);
        for pass in ["filling", "cached"] {
            let verdict = verifier.verify_with(&statements, bytes, KeyCacheMut::Lanes(&mut cache));
            assert!(verdict.accepted() && verdict.cached, "{pass}: {verdict}");
        }
    }

    let outcome = fp_verifier(P::SET, other_relation(relation))
        .verify_with(&statements, bytes, KeyCacheMut::None)
        .report
        .outcome;
    assert!(
        matches!(outcome, Err(VerifyError::Relation { expected, found })
            if expected == other_relation(relation) && found == relation),
        "other relation: {outcome:?}"
    );
    for set in [Set::S44, Set::S65, Set::S87] {
        if set == P::SET {
            continue;
        }
        let outcome = fp_verifier(set, relation)
            .verify_with(&statements, bytes, KeyCacheMut::None)
            .report
            .outcome;
        assert!(
            matches!(outcome, Err(VerifyError::ParamSet { expected, found })
                if expected == set && found == P::SET),
            "verifier of ML-DSA-{}: {outcome:?}",
            set.name()
        );
    }
    let outcome = AggregateVerifier::for_set(P::SET)
        .relation(relation)
        .verify_with(&statements, bytes, KeyCacheMut::None)
        .report
        .outcome;
    assert!(
        matches!(outcome, Err(VerifyError::HashField { expected: HashField::Gf2, found: HashField::Fp })),
        "Gf2 verifier: {outcome:?}"
    );
    eprintln!(
        "Fp {} {relation} N = {n}: {} B ({:.1} B/sig), prove {:.2} s, verify {:.1} ms",
        P::NAME,
        bytes.len(),
        report.bytes_per_signature(),
        prove.as_secs_f64(),
        verify.as_secs_f64() * 1e3
    );
}

/// `n` real records of `P`: half with their own key, half over three shared keys.
fn records_of<P: ParamSet>(seed: u64, n: usize) -> Vec<Record> {
    let mut records = gen::records_for::<P>(seed, n - n / 2);
    records.extend(gen::records_shared_keys_for::<P>(seed + 1, n / 2, 3));
    records
}

/// H-FIPS under `Fp` on the weight-omega records of seeds 1 and 2
/// (`agg_witness::gen::hint_corner_record_for`): each has the coefficient
/// `w' = q - gamma2, h = 0` and the `e = -1` wrap to `w1' = m - 1`, so its hint slack is 0.
/// The batch is the smallest the set admits: the two records alone for ML-DSA-44 (padded to
/// 8), with 15 honest records for ML-DSA-65 and ML-DSA-87 (17, padded to 32).
fn corner<P: ParamSet>(seed: u64) {
    let mut records = if P::SET == Set::S44 {
        Vec::new()
    } else {
        gen::records_for::<P>(seed, agg_piop::prover::LANE_MIN_RECORDS - 2)
    };
    records.push(gen::hint_corner_record_for::<P>(1));
    records.push(gen::hint_corner_record_for::<P>(2));
    round_trip::<P>(Relation::HFips, &records);
}

#[test]
#[cfg_attr(debug_assertions, ignore = "Akita proofs: run with --release")]
fn fp_44_hexact_32() {
    on_large_stack(|| round_trip::<Set44>(Relation::HExact, &records_of::<Set44>(0x4401, 32)));
}

#[test]
#[cfg_attr(debug_assertions, ignore = "Akita proofs: run with --release")]
fn fp_44_hfips_32() {
    on_large_stack(|| round_trip::<Set44>(Relation::HFips, &records_of::<Set44>(0x4402, 32)));
}

#[test]
#[cfg_attr(debug_assertions, ignore = "Akita proofs: run with --release")]
fn fp_65_hexact_32() {
    on_large_stack(|| round_trip::<Set65>(Relation::HExact, &records_of::<Set65>(0x6501, 32)));
}

#[test]
#[cfg_attr(debug_assertions, ignore = "Akita proofs: run with --release")]
fn fp_65_hfips_32() {
    on_large_stack(|| round_trip::<Set65>(Relation::HFips, &records_of::<Set65>(0x6502, 32)));
}

#[test]
#[cfg_attr(debug_assertions, ignore = "Akita proofs: run with --release")]
fn fp_87_hexact_32() {
    on_large_stack(|| round_trip::<Set87>(Relation::HExact, &records_of::<Set87>(0x8701, 32)));
}

#[test]
#[cfg_attr(debug_assertions, ignore = "Akita proofs: run with --release")]
fn fp_87_hfips_32() {
    on_large_stack(|| round_trip::<Set87>(Relation::HFips, &records_of::<Set87>(0x8702, 32)));
}

#[test]
#[cfg_attr(debug_assertions, ignore = "Akita proofs: run with --release")]
fn fp_87_hfips_100() {
    on_large_stack(|| round_trip::<Set87>(Relation::HFips, &records_of::<Set87>(0x8703, 100)));
}

#[test]
#[cfg_attr(debug_assertions, ignore = "Akita proofs: run with --release")]
fn fp_44_hfips_corner() {
    on_large_stack(|| corner::<Set44>(0x44c));
}

#[test]
#[cfg_attr(debug_assertions, ignore = "Akita proofs: run with --release")]
fn fp_65_hfips_corner() {
    on_large_stack(|| corner::<Set65>(0x65c));
}

#[test]
#[cfg_attr(debug_assertions, ignore = "Akita proofs: run with --release")]
fn fp_87_hfips_corner() {
    on_large_stack(|| corner::<Set87>(0x87c));
}

/// `agg aggregate` and `agg verify` with `--params 65 --hash-field fp` at 17 records: the
/// default relation H-FIPS round-trips, and the verifiers of H-exact, of ML-DSA-87 and of
/// the GF(2) hash field exit with status 1.
#[test]
#[cfg_attr(debug_assertions, ignore = "Akita proofs: run with --release")]
fn cli_fp_65_round_trip() {
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("agg-cli-hash-field-sets");
    std::fs::create_dir_all(&dir).unwrap();
    let path = |name: &str| dir.join(name).to_str().unwrap().to_owned();
    let (keys, records, aggregate) = (path("keys.bin"), path("records.bin"), path("aggregate.bin"));
    let agg = |args: &[&str]| {
        let out = std::process::Command::new(env!("CARGO_BIN_EXE_agg"))
            .args(args)
            .output()
            .expect("run agg");
        (
            out.status.code().unwrap_or(-1),
            format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)),
        )
    };
    let run = |args: &[&str]| {
        let (code, out) = agg(args);
        assert_eq!(code, 0, "{out}");
        out
    };
    run(&["keygen-batch", "--params", "65", "--count", "5", "--seed", "65", "--out", &keys]);
    run(&["sign-batch", "--params", "65", "--keys", &keys, "--count", "17", "--seed", "66", "--out", &records]);
    run(&["aggregate", "--params", "65", "--records", &records, "--out", &aggregate, "--hash-field", "fp"]);
    let out = run(&["verify", "--params", "65", "--records", &records, "--aggregate", &aggregate, "--hash-field", "fp"]);
    assert!(out.contains("accept"), "{out}");
    for extra in [&["--params", "65", "--hash-field", "fp", "--relation", "h-exact"][..], &["--params", "65"][..]] {
        let mut args = vec!["verify", "--records", &records, "--aggregate", &aggregate];
        args.extend_from_slice(extra);
        let (code, out) = agg(&args);
        assert_eq!(code, 1, "{extra:?}: {out}");
    }
    // A verifier of another set refuses the ML-DSA-65 record file before verifying.
    let (code, out) = agg(&["verify", "--params", "87", "--records", &records, "--aggregate", &aggregate, "--hash-field", "fp"]);
    assert_ne!(code, 0, "{out}");
}
