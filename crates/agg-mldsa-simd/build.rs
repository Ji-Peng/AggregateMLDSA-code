//! Exposes `cfg(cref)` to this crate's tests and benches exactly when
//! `agg-mldsa-cref` builds its C references, so the C differential tests and
//! gates run whenever `AGG_CREF_SRC` points at the fetched sources. The
//! condition mirrors the one in the `agg-mldsa-cref` build script.
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=AGG_CREF_SRC");
    println!("cargo:rustc-check-cfg=cfg(cref)");
    let Ok(src) = std::env::var("AGG_CREF_SRC") else {
        return;
    };
    let src = PathBuf::from(src.trim());
    if src.as_os_str().is_empty() {
        return;
    }
    if src.join("dilithium").is_dir() && src.join("Hybrid-SHA-3").is_dir() {
        println!("cargo:rustc-cfg=cref");
    }
}
