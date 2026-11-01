//! End-to-end multi-backend equality and verification test across all supported SIMD tiers.
//!
//! Evaluates `keygen-batch`, `sign-batch`, `aggregate`, and `verify` across:
//! - 1024 records over 64 keys (H-u)
//! - 1024 records over 1024 keys (H-u)
//! - 256 records over 64 keys (H-exact)
//! - 256 records over 64 keys (H-FIPS)
//!
//! For each supported SIMD backend (scalar, avx2, avx512) and thread configurations (1 and 16):
//! - Asserts bit-identical aggregate files across all backends and thread counts
//! - Asserts verification passes uncached
//! - Asserts verification passes with cache creation and subsequent cache hit
//!
//! Run with:
//! ```sh
//! cargo test --release -p agg-cli --test all simd_backends -- --ignored
//! ```

use std::path::Path;
use std::process::Command;

fn run_agg(args: &[&str], envs: &[(&str, &str)]) -> (i32, String) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_agg"));
    cmd.args(args);
    cmd.env("AGG_MLDSA_SIMD", "scalar")
        .env("RAYON_NUM_THREADS", "1");
    for (k, v) in envs {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("run agg binary");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.code().unwrap_or(-1), text)
}

fn s(p: &Path) -> &str {
    p.to_str().expect("valid utf-8 path")
}

struct Scenario {
    name: &'static str,
    records: usize,
    keys: usize,
    relation: &'static str,
}

#[test]
#[ignore = "Multi-backend e2e test can exceed two minutes; run with: cargo test --release -p agg-cli --test all simd_backends -- --ignored"]
fn test_simd_backends_e2e_equality_and_verification() {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let base_dir =
        std::env::temp_dir().join(format!("agg_simd_backends_{}_{nonce}", std::process::id()));
    std::fs::create_dir_all(&base_dir).unwrap();

    let mut backends = vec!["scalar"];
    if agg_mldsa_simd::available(agg_mldsa_simd::Backend::Avx2) {
        backends.push("avx2");
    }
    if agg_mldsa_simd::available(agg_mldsa_simd::Backend::Avx512) {
        backends.push("avx512");
    }

    let thread_counts = ["1", "16"];

    let scenarios = [
        Scenario {
            name: "1024_rec_64_keys_hu",
            records: 1024,
            keys: 64,
            relation: "h-u",
        },
        Scenario {
            name: "1024_rec_1024_keys_hu",
            records: 1024,
            keys: 1024,
            relation: "h-u",
        },
        Scenario {
            name: "256_rec_64_keys_hexact",
            records: 256,
            keys: 64,
            relation: "h-exact",
        },
        Scenario {
            name: "256_rec_64_keys_hfips",
            records: 256,
            keys: 64,
            relation: "h-fips",
        },
    ];

    for sc in &scenarios {
        let sc_dir = base_dir.join(sc.name);
        std::fs::create_dir_all(&sc_dir).unwrap();

        let keys_file = sc_dir.join("keys.bin");
        let recs_file = sc_dir.join("records.bin");

        // 1. keygen-batch
        let count_k = sc.keys.to_string();
        let (code, out) = run_agg(
            &[
                "keygen-batch",
                "--count",
                &count_k,
                "--seed",
                "123",
                "--out",
                s(&keys_file),
            ],
            &[],
        );
        assert_eq!(code, 0, "keygen-batch failed for {}: {out}", sc.name);

        // 2. sign-batch
        let count_n = sc.records.to_string();
        let (code, out) = run_agg(
            &[
                "sign-batch",
                "--keys",
                s(&keys_file),
                "--count",
                &count_n,
                "--seed",
                "123",
                "--out",
                s(&recs_file),
            ],
            &[],
        );
        assert_eq!(code, 0, "sign-batch failed for {}: {out}", sc.name);

        let mut baseline_aggregate: Option<Vec<u8>> = None;
        let mut baseline_cache: Option<Vec<u8>> = None;

        for &backend in &backends {
            for &threads in &thread_counts {
                let cache_file = sc_dir.join(format!("cache_{backend}_{threads}.bin"));
                let agg_file = sc_dir.join(format!("agg_{backend}_{threads}.bin"));
                let envs = [("AGG_MLDSA_SIMD", backend), ("RAYON_NUM_THREADS", threads)];

                // 3. aggregate
                let (code, out) = run_agg(
                    &[
                        "aggregate",
                        "--records",
                        s(&recs_file),
                        "--out",
                        s(&agg_file),
                        "--relation",
                        sc.relation,
                    ],
                    &envs,
                );
                assert_eq!(
                    code, 0,
                    "aggregate failed in {} with backend {} and {} threads:\n{out}",
                    sc.name, backend, threads
                );

                let agg_bytes = std::fs::read(&agg_file).unwrap_or_else(|e| {
                    panic!("failed to read aggregate {}: {e}", agg_file.display())
                });

                if let Some(ref baseline) = baseline_aggregate {
                    assert!(
                        &agg_bytes == baseline,
                        "aggregate output mismatch in {} for backend {} with {} threads vs baseline",
                        sc.name, backend, threads
                    );
                } else {
                    baseline_aggregate = Some(agg_bytes);
                }

                // 4. verify uncached
                let (code, out) = run_agg(
                    &[
                        "verify",
                        "--records",
                        s(&recs_file),
                        "--aggregate",
                        s(&agg_file),
                        "--relation",
                        sc.relation,
                    ],
                    &envs,
                );
                assert_eq!(
                    code, 0,
                    "verify uncached failed in {} with backend {} and {} threads:\n{out}",
                    sc.name, backend, threads
                );

                // 5. verify cached
                let (code, out) = run_agg(
                    &[
                        "verify",
                        "--records",
                        s(&recs_file),
                        "--aggregate",
                        s(&agg_file),
                        "--cache",
                        s(&cache_file),
                        "--relation",
                        sc.relation,
                    ],
                    &envs,
                );
                assert_eq!(
                    code, 0,
                    "verify cached (pass 1) failed in {} with backend {} and {} threads:\n{out}",
                    sc.name, backend, threads
                );

                assert!(cache_file.is_file(), "verifier did not create a cache");
                let cache_bytes = std::fs::read(&cache_file).unwrap();
                if let Some(expected) = &baseline_cache {
                    assert!(
                        &cache_bytes == expected,
                        "cache differs across backend/thread settings"
                    );
                } else {
                    baseline_cache = Some(cache_bytes.clone());
                }

                // Re-verify cached to hit populated cache
                let (code, out) = run_agg(
                    &[
                        "verify",
                        "--records",
                        s(&recs_file),
                        "--aggregate",
                        s(&agg_file),
                        "--cache",
                        s(&cache_file),
                        "--relation",
                        sc.relation,
                    ],
                    &envs,
                );
                assert_eq!(
                    code, 0,
                    "verify cached (pass 2 hit) failed in {} with backend {} and {} threads:\n{out}",
                    sc.name, backend, threads
                );
                assert_eq!(
                    std::fs::read(&cache_file).unwrap(),
                    cache_bytes,
                    "cache changed on hit"
                );
            }
        }
    }

    let _ = std::fs::remove_dir_all(&base_dir);
}
