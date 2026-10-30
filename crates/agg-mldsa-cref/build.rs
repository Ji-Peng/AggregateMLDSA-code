use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=AGG_CREF_SRC");
    println!("cargo:rustc-check-cfg=cfg(cref)");

    let cref_src = match std::env::var("AGG_CREF_SRC") {
        Ok(v) if !v.trim().is_empty() => PathBuf::from(v.trim()),
        _ => return,
    };

    let dilithium_dir = cref_src.join("dilithium");
    let hybrid_dir = cref_src.join("Hybrid-SHA-3");

    if !dilithium_dir.is_dir() || !hybrid_dir.is_dir() {
        println!(
            "cargo:warning=AGG_CREF_SRC does not contain dilithium/ and Hybrid-SHA-3/ directories"
        );
        return;
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR set by cargo"));

    // 1. Generate rename headers for hyb512_ and hyb2_
    let hyb512_rename_path = out_dir.join("hyb512_rename.h");
    let hyb512_content = r#"#ifndef HYB512_RENAME_H
#define HYB512_RENAME_H

#define ntt_so_avx512 hyb512_ntt_so_avx512
#define ntt_bo_avx512 hyb512_ntt_bo_avx512
#define intt_so_avx512 hyb512_intt_so_avx512
#define intt_bo_avx512 hyb512_intt_bo_avx512
#define shuffle hyb512_shuffle
#define pointwise_avx512 hyb512_pointwise_avx512
#define XURQ_keccak8x_permute hyb512_XURQ_keccak8x_permute
#define KeccakP1600times8_PermuteAll_24rounds hyb512_KeccakP1600times8_PermuteAll_24rounds
#define KeccakP1600times8_PermuteAll_12rounds hyb512_KeccakP1600times8_PermuteAll_12rounds
#define KeccakP1600times8_PermuteAll_6rounds hyb512_KeccakP1600times8_PermuteAll_6rounds
#define KeccakP1600times8_PermuteAll_4rounds hyb512_KeccakP1600times8_PermuteAll_4rounds
#define XURQ_AVX512_shake128x8_absorb hyb512_XURQ_AVX512_shake128x8_absorb
#define XURQ_AVX512_shake128x8_squeezeblocks hyb512_XURQ_AVX512_shake128x8_squeezeblocks
#define XURQ_AVX512_shake256x8_absorb hyb512_XURQ_AVX512_shake256x8_absorb
#define XURQ_AVX512_shake256x8_squeezeblocks hyb512_XURQ_AVX512_shake256x8_squeezeblocks
#define rej_uniform_avx512 hyb512_rej_uniform_avx512
#define rej_uniform_avx512_SHAKE128RATE hyb512_rej_uniform_avx512_SHAKE128RATE
#define poly_uniform_8x hyb512_poly_uniform_8x
#define ExpandA_shuffled hyb512_ExpandA_shuffled
#define ExpandA hyb512_ExpandA
#define ExpandA_row hyb512_ExpandA_row
#define poly_emulate_ct hyb512_poly_emulate_ct
#define poly_emulate_cs hyb512_poly_emulate_cs
#define emulate_ct hyb512_emulate_ct
#define emulate_cs1 hyb512_emulate_cs1
#define emulate_cs2 hyb512_emulate_cs2
#define decompose_avx512 hyb512_decompose_avx512
#define decompose_avx hyb512_decompose_avx
#define use_hint_avx hyb512_use_hint_avx
#define make_hint_avx hyb512_make_hint_avx
#define power2round_avx hyb512_power2round_avx
#define power2round_avx512 hyb512_power2round_avx512
#define idxlut hyb512_idxlut
#define qdata hyb512_qdata
#define qdata2 hyb512_qdata2
#define psi hyb512_psi
#define ipsi hyb512_ipsi
#define ipsi2 hyb512_ipsi2

#endif
"#;
    std::fs::write(&hyb512_rename_path, hyb512_content).expect("write hyb512_rename.h");

    let hyb2_rename_path = out_dir.join("hyb2_rename.h");
    let hyb2_content = r#"#ifndef HYB2_RENAME_H
#define HYB2_RENAME_H

#define poly_emulate_ct hyb2_poly_emulate_ct
#define poly_emulate_cs hyb2_poly_emulate_cs
#define emulate_ct hyb2_emulate_ct
#define emulate_cs1 hyb2_emulate_cs1
#define emulate_cs2 hyb2_emulate_cs2

#endif
"#;
    std::fs::write(&hyb2_rename_path, hyb2_content).expect("write hyb2_rename.h");

    // 2. Build pq-crystals avx2 static library. The ISA is pinned to the Rust backend it is
    // compared with (x86-64-v3 = AVX2, no AVX-512); `-march=native` would let GCC use
    // AVX-512 encodings on AVX-512 hosts. Tuning stays native.
    let pq_avx2 = dilithium_dir.join("avx2");
    let pq_ref = dilithium_dir.join("ref");
    let mut pq_build = cc::Build::new();
    pq_build
        .opt_level(3)
        .flag("-march=x86-64-v3")
        .flag("-mtune=native")
        .flag("-mavx2")
        .flag("-mpopcnt")
        .define("DILITHIUM_MODE", "2")
        .include(&pq_avx2)
        .include(&pq_ref)
        .files([
            pq_avx2.join("packing.c"),
            pq_avx2.join("polyvec.c"),
            pq_avx2.join("poly.c"),
            pq_avx2.join("ntt.S"),
            pq_avx2.join("invntt.S"),
            pq_avx2.join("pointwise.S"),
            pq_avx2.join("shuffle.S"),
            pq_avx2.join("consts.c"),
            pq_avx2.join("rejsample.c"),
            pq_avx2.join("rounding.c"),
            pq_avx2.join("fips202.c"),
            pq_avx2.join("fips202x4.c"),
            pq_avx2.join("f1600x4.S"),
            pq_avx2.join("symmetric-shake.c"),
        ]);
    pq_build.compile("pqcrystals_dilithium2_avx2");

    // Restore the original XKCP theta schedule in the build copy. The pinned
    // Hybrid modification overwrites column parities before later uses.
    // Keep external checkouts untouched and reject unexpected source changes.
    let xkcp_path = hybrid_dir.join("MLDSA_AVX512/keccak/KeccakP-1600-times8-SIMD512.c");
    let xkcp = std::fs::read_to_string(&xkcp_path).expect("read pinned XKCP");
    let start = xkcp
        .find("\n#define KeccakP_ThetaRhoPiChiIota0(")
        .expect("theta macro");
    let end = start
        + xkcp[start..]
            .find("#define KeccakP_ThetaRhoPiChi1(")
            .expect("next macro");
    assert!(
        xkcp[start..end].contains("_Be = XOR3(_Ba, _L2, _De);"),
        "unexpected pinned XKCP theta macro"
    );
    let restored = include_str!("c/xkcp-theta.h");
    let mut corrected = xkcp[..start].to_owned();
    corrected.push('\n');
    corrected.push_str(restored);
    corrected.push_str(&xkcp[end..]);
    let corrected_xkcp = out_dir.join("KeccakP-1600-times8-corrected.c");
    std::fs::write(&corrected_xkcp, corrected).expect("write restored XKCP");
    println!("cargo:rerun-if-changed={}", xkcp_path.display());
    println!("cargo:rerun-if-changed=c/xkcp-theta.h");
    println!("cargo:rerun-if-changed=c/shake128x8.c");

    // 3. Build Hybrid MLDSA_AVX512 static library
    let hyb512_base = hybrid_dir.join("MLDSA_AVX512");
    let mut hyb512_build = cc::Build::new();
    hyb512_build
        .opt_level(3)
        .flag("-march=x86-64-v4")
        .flag("-mtune=native")
        .flag("-mavx2")
        .flag("-mpopcnt")
        .flag("-mbmi2")
        .flag("-mavx512f")
        .flag("-mavx512vl")
        .flag("-mavx512cd")
        .flag("-mavx512bw")
        .flag("-mavx512dq")
        .flag("-mavx512vbmi2")
        .flag("-mavx512vbmi")
        .define("DILITHIUM_MODE", "2")
        .flag(&format!("-include{}", hyb512_rename_path.display()))
        .flag("-Wno-int-conversion")
        .flag("-Wno-incompatible-pointer-types")
        .flag("-Wno-unused-function")
        .include(&hyb512_base)
        .include(hyb512_base.join("ntt"))
        .include(hyb512_base.join("keccak"))
        .include(hyb512_base.join("test"))
        .files([
            hyb512_base.join("ntt/ntt512.c"),
            hyb512_base.join("ntt/inv_ntt512.c"),
            hyb512_base.join("ntt/pointwise512.c"),
            hyb512_base.join("ntt/consts.c"),
            hyb512_base.join("keccak/keccak8x.c"),
            corrected_xkcp,
            PathBuf::from("c/shake128x8.c"),
            hyb512_base.join("keccak/fips202x8.c"),
            hyb512_base.join("rejsample.c"),
            hyb512_base.join("pspm.c"),
            hyb512_base.join("rounding.c"),
        ]);
    hyb512_build.compile("hybrid_avx512");

    // 4. Build Hybrid MLDSA_AVX2 static library
    let hyb2_base = hybrid_dir.join("MLDSA_AVX2");
    let mut hyb2_build = cc::Build::new();
    hyb2_build
        .opt_level(3)
        .flag("-march=x86-64-v3")
        .flag("-mtune=native")
        .flag("-mavx2")
        .flag("-mpopcnt")
        .define("DILITHIUM_MODE", "2")
        .flag(&format!("-include{}", hyb2_rename_path.display()))
        .flag("-Wno-int-conversion")
        .flag("-Wno-incompatible-pointer-types")
        .flag("-Wno-unused-function")
        .include(&hyb2_base)
        .include(hyb2_base.join("ntt"))
        .include(hyb2_base.join("keccak4x"))
        .files([hyb2_base.join("pspm.c")]);
    hyb2_build.compile("hybrid_avx2");

    println!("cargo:rustc-cfg=cref");
}
