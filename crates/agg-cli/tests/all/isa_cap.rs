//! `AGG_ISA` caps every runtime-dispatched kernel family of the `agg` binary.
//!
//! The startup report prints the level each crate selected; a capped process
//! must report no AVX-512 arm and must leave the AVX-512IFMA NTT off.

use std::process::Command;

fn report(isa: Option<&str>) -> String {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_agg"));
    cmd.env("AGG_SIMD_REPORT", "1");
    match isa {
        Some(value) => {
            cmd.env("AGG_ISA", value);
        }
        None => {
            cmd.env_remove("AGG_ISA");
        }
    }
    // No subcommand: the binary prints the report, then the usage text.
    let out = cmd.output().expect("run agg binary");
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

#[test]
fn avx2_cap_selects_no_avx512_arm() {
    if !(std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("vpclmulqdq"))
    {
        eprintln!("skipped: no avx2+vpclmulqdq on this CPU");
        return;
    }
    let text = report(Some("avx2"));
    assert!(text.contains("agg-mldsa-simd: backend=Avx2"), "{text}");
    assert!(
        text.contains("flock-core: cap=avx2 (avx2) avx2=true vpclmul256=true"),
        "{text}"
    );
    for off in [
        "avx512f=false",
        "avx512clmul=false",
        "avx512gfni=false",
        "avx512vbmi=false",
        "gf2-width=Vpclmul256",
    ] {
        assert!(text.contains(off), "missing {off} in {text}");
    }
    assert!(
        text.contains("akita-algebra: cap=avx2 (avx2) crt-ntt-ifma=false crt-ntt-avx512=false"),
        "{text}"
    );
    assert!(text.contains("akita-challenges: cap=avx2 (avx2) avx2=true"), "{text}");
    assert!(text.contains("aerie-simd: backend=avx2"), "{text}");
}

#[test]
fn scalar_cap_selects_portable_kernels() {
    let text = report(Some("scalar"));
    assert!(text.contains("agg-mldsa-simd: backend=Scalar"), "{text}");
    assert!(text.contains("flock-core: cap=scalar (scalar) avx2=false"), "{text}");
    assert!(text.contains("gf2-width=Portable"), "{text}");
    assert!(text.contains("aerie-simd: backend=scalar"), "{text}");
}

#[test]
fn avx512ifma_cap_keeps_ifma() {
    let text = report(Some("avx512ifma"));
    assert!(text.contains("akita-algebra: cap=avx512ifma (avx512ifma)"));
    if std::arch::is_x86_feature_detected!("avx512ifma") {
        assert!(text.contains("crt-ntt-ifma=true"), "{text}");
    }
}

#[test]
fn unset_cap_detects() {
    let text = report(None);
    assert!(text.contains("cap=avx512ifma (detected)"), "{text}");
}

#[test]
fn invalid_cap_is_an_error() {
    let text = report(Some("avx4"));
    assert!(text.contains("invalid AGG_ISA 'avx4'"), "{text}");
}
