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

    let (code, out) = agg(&[
        "verify",
        "--records",
        s(&recs),
        "--aggregate",
        s(&aggregate),
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
    ]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains(": reject"), "{out}");

    // Relation H-exact: aggregate and verify with --relation h-exact; the default (H-u)
    // verifier and an H-u aggregate verified as H-exact are rejected.
    assert!(out.contains("relation H-u"), "{out}");
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
    assert!(out.contains("the proof claims relation H-exact, the verifier expects H-u"), "{out}");
    let (code, out) = verify_as(&aggregate, Some("h-exact"));
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("the proof claims relation H-u, the verifier expects H-exact"), "{out}");
    assert_eq!(verify_as(&aggregate, Some("fips")).0, 2);
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

