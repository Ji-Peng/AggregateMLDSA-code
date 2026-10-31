#!/usr/bin/env python3
"""Build every binary of the package and run the static AVX-512 checks."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    AERIE_TOOLCHAIN,
    ARTIFACT,
    BIN,
    BUILD,
    REPO,
    RUST_TOOLCHAIN,
    VENDOR,
    git_rev,
    load_configs,
    output,
    run,
    sha256_file,
    utc_stamp,
    write_json,
)

AVX512_PATTERNS = {
    "zmm": re.compile(r"%zmm[0-9]+"),
    "opmask": re.compile(r"%k[0-7]\b|\{%k[0-7]\}"),
    "high_ymm_xmm": re.compile(r"%[xy]mm(1[6-9]|2[0-9]|3[01])\b"),
    "evex_mnemonic": re.compile(
        r"\bv(pmadd52|pternlog|permt2|permi2|pcompress|pexpand|pro[lr]v?[dq]|"
        r"movdqu(8|16|32|64)|movdqa(32|64)|broadcast[if](32x|64x)|"
        r"extract[if](32x|64x)|insert[if](32x|64x)|pmov[us]?[dqw][bdw])"
    ),
}


def objdump_counts(path):
    text = output(["objdump", "-d", "--no-show-raw-insn", str(path)], default="")
    counts = {}
    for name, pattern in AVX512_PATTERNS.items():
        counts[name] = sum(1 for line in text.splitlines() if pattern.search(line))
    counts["total"] = sum(counts[name] for name in AVX512_PATTERNS)
    return counts


def features():
    env = {"RUSTUP_TOOLCHAIN": RUST_TOOLCHAIN}
    proc = run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1", "--locked"],
        cwd=REPO,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        proc = run(
            ["cargo", "metadata", "--no-deps", "--format-version", "1"],
            cwd=REPO,
            env=env,
            check=False,
        )
    if proc.returncode != 0:
        return set(), proc.stderr
    data = json.loads(proc.stdout)
    for package in data.get("packages", []):
        if package.get("name") == "agg-cli":
            return set(package.get("features", {})), None
    return set(), "agg-cli not found in cargo metadata"


def build_ours(binaries, declared, offline):
    built = {}
    aliases = {}
    env = {"RUSTUP_TOOLCHAIN": RUST_TOOLCHAIN}
    for name, spec in binaries.items():
        required = spec.get("requires_feature")
        if required and required not in declared:
            aliases[name] = "agg-v3"
            print(f"{name}: feature {required} is not declared; using agg-v3")
            continue
        target = BUILD / f"target-{name}"
        cmd = [
            "cargo",
            "build",
            "--release",
            "--locked",
            "-p",
            "agg-cli",
        ]
        if offline:
            cmd.append("--offline")
        feats = spec.get("features") or []
        if feats:
            cmd += ["--features", ",".join(feats)]
        print(f"building {name}: {spec['rustflags']}")
        run(
            cmd,
            cwd=REPO,
            env={
                **env,
                "RUSTFLAGS": spec["rustflags"],
                "CARGO_TARGET_DIR": str(target),
                "CARGO_BUILD_JOBS": str(spec.get("jobs", 6)),
            },
        )
        src = target / "release" / "agg"
        dst = BIN / name
        BIN.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        built[name] = {
            "path": str(dst),
            "sha256": sha256_file(dst),
            "rustflags": spec["rustflags"],
            "features": feats,
            "target_dir": str(target),
        }
    for alias, base in aliases.items():
        dst = BIN / alias
        # Keep the alias in step with a rebuilt base binary.
        shutil.copy2(BIN / base, dst)
        built[alias] = {
            "path": str(dst),
            "sha256": sha256_file(dst),
            "alias_of": base,
            "feature_missing": "no-avx512",
        }
    return built


def find_bench(target):
    candidates = [
        path
        for path in (target / "release" / "deps").glob("falcon-*")
        if path.is_file() and os.access(path, os.X_OK) and not path.name.endswith(".d")
    ]
    if not candidates:
        raise SystemExit(f"aerie-bench binary not found under {target}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_aerie(offline, capped):
    aerie = VENDOR / "aerie"
    if not (aerie / "Cargo.toml").exists():
        raise SystemExit("artifact/vendor/aerie is missing; run ./run.sh fetch")
    target = BUILD / "target-aerie"
    cmd = [
        "cargo",
        "build",
        "--release",
        "--locked",
        "-p",
        "aerie-bench",
        "--features",
        "falcon-bench",
        "--bench",
        "falcon",
    ]
    if offline:
        cmd.append("--offline")
    run(
        cmd,
        cwd=aerie,
        env={
            "RUSTUP_TOOLCHAIN": AERIE_TOOLCHAIN,
            "RUSTFLAGS": "",
            "CARGO_TARGET_DIR": str(target),
        },
    )
    BIN.mkdir(parents=True, exist_ok=True)
    plain = BIN / "aerie-falcon-bench"
    shutil.copy2(find_bench(target), plain)
    result = {
        "path": str(plain),
        "sha256": sha256_file(plain),
        "capped": False,
        "avx512_counts": objdump_counts(plain),
        "note": "Aerie dispatches at run time; its binary contains AVX-512 code even when capped.",
    }
    if capped:
        target_capped = BUILD / "target-aerie-capped"
        env = {
            "RUSTUP_TOOLCHAIN": AERIE_TOOLCHAIN,
            "RUSTFLAGS": "",
            "CARGO_TARGET_DIR": str(target_capped),
        }
        base = [
            "cargo",
            "build",
            "--release",
            "--locked",
            "-p",
            "aerie-bench",
            "--bench",
            "falcon",
        ]
        proc = run(
            base
            + [
                "--features",
                "falcon-bench,aerie-simd/backend-override,aerie-keccak/backend-override",
            ],
            cwd=aerie,
            env=env,
            check=False,
        )
        patched = False
        if proc.returncode != 0:
            print("aerie-simd/backend-override is not selectable; applying the cap patch")
            patch = ARTIFACT / "patches" / "aerie-simd-cap.patch"
            applied = run(["git", "apply", str(patch)], cwd=aerie, check=False)
            if applied.returncode != 0:
                result["capped_error"] = (applied.stderr or "").strip()[-2000:]
            else:
                patched = True
                proc = run(
                    base + ["--features", "falcon-bench"],
                    cwd=aerie,
                    env=env,
                    check=False,
                )
                run(["git", "apply", "-R", str(patch)], cwd=aerie, check=False)
        if proc.returncode == 0:
            capped_bin = BIN / "aerie-falcon-bench-avx2"
            shutil.copy2(find_bench(target_capped), capped_bin)
            result["capped"] = {
                "path": str(capped_bin),
                "sha256": sha256_file(capped_bin),
                "env": "AERIE_SIMD_BACKEND=avx2 AERIE_KECCAK_BACKEND=avx2",
                "patch": "patches/aerie-simd-cap.patch" if patched else None,
                "avx512_counts": objdump_counts(capped_bin),
                "note": "The cap is a run-time ceiling; the binary still contains AVX-512 code.",
            }
        else:
            result["capped_error"] = (proc.stderr or "").strip()[-2000:]
    return result


def build_natver():
    natver = ARTIFACT / "natver"
    run(["make", "-j", "6"], cwd=natver)
    BIN.mkdir(parents=True, exist_ok=True)
    dst = BIN / "natver"
    shutil.copy2(natver / "natver", dst)
    return {"path": str(dst), "sha256": sha256_file(dst)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="pass --offline to cargo")
    parser.add_argument(
        "--with-aerie-capped",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also build Aerie with AERIE_SIMD_BACKEND (default: yes)",
    )
    parser.add_argument("--skip-aerie", action="store_true")
    args = parser.parse_args()
    offline = args.offline or os.environ.get("AGG_OFFLINE") == "1"
    configs = load_configs()
    build_cfg = configs["build"]

    declared, error = features()
    if error:
        print(f"warning: cargo metadata: {error}", file=sys.stderr)
    feature_present = "no-avx512" in declared
    binaries = {}
    for name, spec in build_cfg["binaries"].items():
        binaries[name] = {**spec, "jobs": build_cfg.get("jobs", 6)}
    built = build_ours(binaries, declared, offline)

    checks = {}
    warnings = []
    for name, entry in built.items():
        spec = build_cfg["binaries"][name]
        if not spec.get("avx512_check"):
            continue
        counts = objdump_counts(Path(entry["path"]))
        hard = "alias_of" not in entry
        checks[name] = {**counts, "hard_failure": hard}
        if counts["total"] and hard:
            raise SystemExit(
                f"{name}: AVX-512 instructions in a no-avx512 build: {counts}"
            )
        if counts["total"]:
            warnings.append(
                f"{name}: built without the no-avx512 feature; "
                f"{counts['total']} AVX-512-looking lines recorded, not a failure"
            )

    natver = build_natver()
    natver_counts = objdump_counts(Path(natver["path"]))
    checks["natver"] = {**natver_counts, "hard_failure": True}
    if natver_counts["total"]:
        raise SystemExit(f"natver contains AVX-512 instructions: {natver_counts}")

    aerie = None
    if not args.skip_aerie:
        aerie = build_aerie(offline, args.with_aerie_capped)

    result = {
        "timestamp_utc": utc_stamp(),
        "commit": git_rev(REPO),
        "toolchains": {
            "rustc": output(["rustc", "-V"], env={"RUSTUP_TOOLCHAIN": RUST_TOOLCHAIN}),
            "cargo": output(["cargo", "-V"], env={"RUSTUP_TOOLCHAIN": RUST_TOOLCHAIN}),
            "aerie_rustc": output(
                ["rustc", "-V"], env={"RUSTUP_TOOLCHAIN": AERIE_TOOLCHAIN}
            ),
            "gcc": output(["gcc", "--version"]).splitlines()[0],
        },
        "no_avx512_feature": feature_present,
        "declared_features": sorted(declared),
        "binaries": built,
        "aerie": aerie,
        "natural_verifier": natver,
        "avx512_checks": checks,
        "warnings": warnings,
    }
    write_json(BUILD / "build.json", result)
    print("build: OK")
    if not feature_present:
        print(
            "warning: the workspace does not declare the no-avx512 feature; "
            "the avx2 and scalar rows use the agg-v3 binary"
        )
    for warning in warnings:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
