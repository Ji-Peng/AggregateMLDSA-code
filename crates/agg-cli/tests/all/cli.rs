//! The `agg` binary end to end: `keygen-batch`, `sign-batch`, `aggregate`, `verify` (with
//! and without the key cache, on the right and on other messages) and `bench`, through the
//! files of `agg_cli::record`. Run with `--release`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn agg(args: &[&str]) -> (i32, String) {
    agg_env(args, &[])
}

fn agg_env(args: &[&str], env: &[(&str, &str)]) -> (i32, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_agg"))
        .args(args)
        .envs(env.iter().copied())
        .output()
        .expect("run agg");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.code().unwrap_or(-1), text)
}

fn s(p: &Path) -> &str {
    p.to_str().expect("utf-8 path")
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Flock and Akita proofs: run with --release"
)]
fn keygen_sign_aggregate_verify_and_bench() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("agg-cli-test");
    std::fs::create_dir_all(&dir).unwrap();
    let (keys, recs, other, aggregate, key_cache) = (
        dir.join("keys.bin"),
        dir.join("records.bin"),
        dir.join("other.bin"),
        dir.join("aggregate.bin"),
        dir.join("cache.bin"),
    );
    let _ = std::fs::remove_file(&key_cache);

    let (code, out) = agg(&[
        "keygen-batch",
        "--count",
        "3",
        "--seed",
        "5",
        "--out",
        s(&keys),
    ]);
    assert_eq!(code, 0, "{out}");
    let (code, out) = agg(&[
        "sign-batch",
        "--keys",
        s(&keys),
        "--count",
        "10",
        "--seed",
        "5",
        "--out",
        s(&recs),
    ]);
    assert_eq!(code, 0, "{out}");
    let (code, out) = agg(&[
        "sign-batch",
        "--keys",
        s(&keys),
        "--count",
        "10",
        "--seed",
        "6",
        "--msg-len",
        "33",
        "--out",
        s(&other),
    ]);
    assert_eq!(code, 0, "{out}");
    let (code, out) = agg(&[
        "aggregate",
        "--records",
        s(&recs),
        "--out",
        s(&aggregate),
        "--part-log2",
        "3",
    ]);
    assert_eq!(code, 0, "{out}");
    eprintln!("{out}");
    assert!(out.contains("Flock part (0, 1): 2 records"), "{out}");
    assert!(out.contains("B/sig"), "{out}");
    let size = std::fs::metadata(&aggregate).unwrap().len();
    assert!(out.contains(&format!("wrote {size} B")), "{out}");

    // Two parts in the segment are not the default prover's layout: refused without
    // --flock-layout any, verified with it.
    let (code, out) = agg(&[
        "verify",
        "--records",
        s(&recs),
        "--aggregate",
        s(&aggregate),
    ]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("is not accepted by default") && out.contains("--flock-layout any"), "{out}");
    assert_eq!(
        agg(&["verify", "--records", s(&recs), "--aggregate", s(&aggregate), "--flock-layout", "all"]).0,
        2
    );
    let (code, out) = agg(&[
        "verify",
        "--records",
        s(&recs),
        "--aggregate",
        s(&aggregate),
        "--flock-layout",
        "any",
    ]);
    assert_eq!(code, 0, "{out}");
    eprintln!("{out}");
    assert!(out.contains(": accept"), "{out}");
    assert!(out.contains("keys expanded in this run: 3"), "{out}");
    let (code, out) = agg(&[
        "verify",
        "--records",
        s(&recs),
        "--aggregate",
        s(&aggregate),
        "--cache",
        s(&key_cache),
        "--flock-layout",
        "any",
    ]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("3 keys (3 new)"), "{out}");
    let (code, out) = agg(&[
        "verify",
        "--records",
        s(&recs),
        "--aggregate",
        s(&aggregate),
        "--cache",
        s(&key_cache),
        "--flock-layout",
        "any",
    ]);
    assert_eq!(code, 0, "{out}");
    assert!(
        out.contains("cached keys") && out.contains("keys expanded in this run: 0"),
        "{out}"
    );
    // Other messages under the same keys: rejected, exit status 1.
    let (code, out) = agg(&[
        "verify",
        "--records",
        s(&other),
        "--aggregate",
        s(&aggregate),
        "--flock-layout",
        "any",
    ]);
    assert_eq!(code, 1, "{out}");
    assert!(!out.contains("--flock-layout any"), "rejected past the layout check: {out}");
    assert!(out.contains(": reject"), "{out}");

    // Relation H-exact: aggregate and verify with --relation h-exact; the default (H-FIPS)
    // verifier and an H-FIPS aggregate verified as H-exact are rejected.
    assert!(out.contains("relation H-FIPS"), "{out}");
    let exact = dir.join("exact.agg");
    let (code, out) = agg(&[
        "aggregate",
        "--records",
        s(&recs),
        "--out",
        s(&exact),
        "--relation",
        "h-exact",
    ]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("relation H-exact"), "{out}");
    let verify_as = |file: &std::path::Path, relation: Option<&str>| {
        let mut args = vec!["verify", "--records", s(&recs), "--aggregate", s(file)];
        // The two-part aggregate needs the layout opt-in; the others are default layouts.
        if file == aggregate.as_path() {
            args.extend(["--flock-layout", "any"]);
        }
        if let Some(r) = relation {
            args.extend(["--relation", r]);
        }
        agg(&args)
    };
    let (code, out) = verify_as(&exact, Some("h-exact"));
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("relation H-exact: accept"), "{out}");
    let (code, out) = verify_as(&exact, None);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("the proof claims relation H-exact, the verifier expects H-FIPS"), "{out}");
    let (code, out) = verify_as(&aggregate, Some("h-exact"));
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("the proof claims relation H-FIPS, the verifier expects H-exact"), "{out}");
    assert_eq!(verify_as(&aggregate, Some("fips")).0, 2);
    assert_eq!(verify_as(&aggregate, Some("h-u")).0, 2);
    // Relation H-FIPS: --relation h-fips round trip, crossed with H-exact.
    let fips = dir.join("fips.agg");
    let (code, out) = agg(&["aggregate", "--records", s(&recs), "--out", s(&fips), "--relation", "h-fips"]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("relation H-FIPS"), "{out}");
    let (code, out) = verify_as(&fips, Some("h-fips"));
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("relation H-FIPS: accept"), "{out}");
    let (code, out) = verify_as(&fips, Some("h-exact"));
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("the proof claims relation H-FIPS, the verifier expects H-exact"), "{out}");
    let (code, out) = verify_as(&exact, Some("h-fips"));
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("the proof claims relation H-exact, the verifier expects H-FIPS"), "{out}");

    // Usage errors.
    assert_eq!(agg(&["prove"]).0, 2);
    assert_eq!(agg(&[]).0, 2);
    assert_eq!(agg(&["verify", "--records", s(&recs)]).0, 2);
    assert_eq!(
        agg(&["keygen-batch", "--count", "x", "--out", s(&keys)]).0,
        2
    );
    assert_eq!(
        agg(&["aggregate", "--records", s(&keys), "--out", s(&aggregate)]).0,
        2
    );

    let (code, out) = agg(&["bench", "--count", "8", "--keys", "2", "--reps", "1"]);
    assert_eq!(code, 0, "{out}");
    eprintln!("{out}");
    for needle in [
        "summary:",
        "verify, uncached keys",
        "verify, cached keys",
        "prover peak RSS",
        "verifier peak RSS",
        "B per signature",
    ] {
        assert!(out.contains(needle), "{needle}: {out}");
    }
}


/// `--params 65` and `--params 87`: keys, records, aggregate, verify (with the per-set key
/// cache), tamper rejection, and the refusals across sets and relations.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Flock and Akita proofs: run with --release"
)]
fn ml_dsa_65_and_87_through_the_binary() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("agg-cli-sets");
    std::fs::create_dir_all(&dir).unwrap();
    let file = |name: &str| dir.join(name);
    for (params, other) in [("65", "87"), ("87", "65")] {
        let (keys, recs, aggregate, key_cache) = (
            file(&format!("keys{params}.bin")),
            file(&format!("records{params}.bin")),
            file(&format!("aggregate{params}.bin")),
            file(&format!("cache{params}.bin")),
        );
        let _ = std::fs::remove_file(&key_cache);
        let run = |args: &[&str]| {
            let mut all = args.to_vec();
            all.extend(["--params", params]);
            agg(&all)
        };
        let (code, out) = run(&["keygen-batch", "--count", "3", "--seed", "5", "--out", s(&keys)]);
        assert_eq!(code, 0, "{out}");
        assert!(out.contains(&format!("ML-DSA-{params} key pairs")), "{out}");
        let (code, out) = run(&["sign-batch", "--keys", s(&keys), "--count", "20", "--seed", "5", "--out", s(&recs)]);
        assert_eq!(code, 0, "{out}");
        let (code, out) = run(&["aggregate", "--records", s(&recs), "--out", s(&aggregate)]);
        assert_eq!(code, 0, "{out}");
        let c_tilde = if params == "65" { 960 } else { 1280 };
        assert!(out.contains(&format!("{:<26} {c_tilde:>10} B", "c~ list")), "{out}");
        // Batches of the lane sets hold at least 17 records.
        let small = file(&format!("small{params}.bin"));
        let (code, out) = run(&["sign-batch", "--keys", s(&keys), "--count", "16", "--out", s(&small)]);
        assert_eq!(code, 0, "{out}");
        let (code, out) = run(&["aggregate", "--records", s(&small), "--out", s(&file("x.agg"))]);
        assert_eq!(code, 2, "{out}");
        assert!(out.contains("17 to 2^15"), "{out}");
        let verify = |args: &[&str]| {
            let mut all = vec!["verify", "--records", s(&recs)];
            all.extend_from_slice(args);
            run(&all)
        };
        let (code, out) = verify(&["--aggregate", s(&aggregate)]);
        assert_eq!(code, 0, "{out}");
        assert!(out.contains(": accept") && out.contains("keys expanded in this run: 3"), "{out}");
        let (code, out) = verify(&["--aggregate", s(&aggregate), "--cache", s(&key_cache)]);
        assert_eq!(code, 0, "{out}");
        assert!(out.contains("3 keys (3 new)"), "{out}");
        let (code, out) = verify(&["--aggregate", s(&aggregate), "--cache", s(&key_cache)]);
        assert_eq!(code, 0, "{out}");
        assert!(out.contains("cached keys") && out.contains("keys expanded in this run: 0"), "{out}");
        // A flipped byte in the middle of the aggregate is rejected.
        let mut bytes = std::fs::read(&aggregate).unwrap();
        let middle = bytes.len() / 2;
        bytes[middle] ^= 1;
        let bad = file(&format!("bad{params}.bin"));
        std::fs::write(&bad, &bytes).unwrap();
        let (code, out) = verify(&["--aggregate", s(&bad)]);
        assert_eq!(code, 1, "{out}");
        assert!(out.contains(": reject"), "{out}");
        // Refusals: the records under another set, another relation, a cache of another set.
        let (code, out) = agg(&["verify", "--records", s(&recs), "--aggregate", s(&aggregate), "--params", other]);
        assert_eq!(code, 2, "{out}");
        assert!(out.contains(&format!("--params {params}")), "{out}");
        let (code, out) = agg(&["aggregate", "--records", s(&recs), "--out", s(&bad)]);
        assert_eq!(code, 2, "{out}");
        let (code, out) = verify(&["--aggregate", s(&aggregate), "--relation", "h-exact"]);
        assert_ne!(code, 0, "{out}");
        // H-exact through the binary: accepted under h-exact, rejected under the default.
        let exact = file(&format!("exact{params}.bin"));
        let (code, out) = run(&["aggregate", "--records", s(&recs), "--out", s(&exact), "--relation", "h-exact"]);
        assert_eq!(code, 0, "{out}");
        let (code, out) = verify(&["--aggregate", s(&exact), "--relation", "h-exact"]);
        assert_eq!(code, 0, "{out}");
        assert!(out.contains(": accept"), "{out}");
        let (code, out) = verify(&["--aggregate", s(&exact)]);
        assert_ne!(code, 0, "{out}");
        // H-FIPS through the binary: accepted under h-fips, rejected under h-exact.
        let fips = file(&format!("fips{params}.bin"));
        let (code, out) = run(&["aggregate", "--records", s(&recs), "--out", s(&fips), "--relation", "h-fips"]);
        assert_eq!(code, 0, "{out}");
        let (code, out) = verify(&["--aggregate", s(&fips), "--relation", "h-fips"]);
        assert_eq!(code, 0, "{out}");
        assert!(out.contains(": accept"), "{out}");
        let (code, out) = verify(&["--aggregate", s(&fips), "--relation", "h-exact"]);
        assert_ne!(code, 0, "{out}");
        // An (empty) ML-DSA-44 cache file.
        let cross = file("cache44.bin");
        std::fs::write(&cross, [b"AGGKCACH".as_slice(), &[2, 0x44, 0, 0, 0, 0]].concat()).unwrap();
        let (code, out) = verify(&["--aggregate", s(&aggregate), "--cache", s(&cross)]);
        assert_eq!(code, 2, "{out}");
        assert!(out.contains(&format!("ML-DSA-44 keys, not ML-DSA-{params}")), "{out}");
    }
    // An ML-DSA-65 aggregate verified as ML-DSA-87 with records whose keys pass the length
    // check is impossible (the key lengths differ), so the header check is covered by the
    // library tests; an unknown set name is a usage error.
    assert_eq!(agg(&["keygen-batch", "--count", "1", "--params", "66", "--out", s(&file("x.bin"))]).0, 2);
}

/// `AGG_W1_SPLIT=0` builds the level-1 weight table instead of its two factors. It changes
/// memory use only: the aggregate of 1,024 records is byte-identical to the default one, for
/// ML-DSA-44 (one weight) and ML-DSA-65 (one weight per polynomial lane).
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Flock and Akita proofs: run with --release"
)]
fn w1_split_off_gives_the_same_aggregate() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("agg-cli-w1-split");
    std::fs::create_dir_all(&dir).unwrap();
    for params in ["44", "65"] {
        let file = |name: &str| dir.join(format!("{name}{params}.bin"));
        let (keys, recs) = (file("keys"), file("records"));
        let (code, out) = agg(&["keygen-batch", "--count", "64", "--seed", "11", "--out", s(&keys), "--params", params]);
        assert_eq!(code, 0, "{out}");
        let (code, out) = agg(&[
            "sign-batch", "--keys", s(&keys), "--count", "1024", "--seed", "11", "--out", s(&recs), "--params", params,
        ]);
        assert_eq!(code, 0, "{out}");
        let aggregate = |split: Option<&str>, out_file: &Path| {
            let mut cmd = Command::new(env!("CARGO_BIN_EXE_agg"));
            cmd.args(["aggregate", "--records", s(&recs), "--out", s(out_file), "--params", params]);
            match split {
                Some(v) => cmd.env("AGG_W1_SPLIT", v),
                None => cmd.env_remove("AGG_W1_SPLIT"),
            };
            let out = cmd.output().expect("run agg");
            assert!(
                out.status.success(),
                "ML-DSA-{params} AGG_W1_SPLIT={split:?}: {}{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            std::fs::read(out_file).unwrap()
        };
        let default = aggregate(None, &file("split"));
        let table = aggregate(Some("0"), &file("table"));
        assert!(default == table, "ML-DSA-{params}: AGG_W1_SPLIT=0 changed the aggregate");
    }
}
