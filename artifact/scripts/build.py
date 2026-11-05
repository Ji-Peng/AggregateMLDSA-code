#!/usr/bin/env python3
"""Build the one release binary, Aerie's baseline and natver.

Every tier of ours is a run-time cap of the same binary, so only the C
baseline gets a static AVX-512 check: our binary legitimately contains all
the AVX-512 and IFMA kernels. The instruction set each tier selects is
checked at run time instead (`AGG_SIMD_REPORT=1`, see `simd_report.py`).
"""

import argparse
import os
import re
import shutil
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
    cpu_flags,
    load_configs,
    load_toml,
    output,
    pinned_commit,
    run,
    sha256_file,
    source_revision,
    utc_stamp,
    write_json,
)
from simd_report import capture, failures, summarise  # noqa: E402

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


def release_flags():
    """The baseline the repository compiles with, read from `.cargo/config.toml`."""
    path = REPO / ".cargo" / "config.toml"
    if not path.exists():
        return {"path": str(path), "rustflags": [], "present": False}
    data = load_toml(path)
    flags = data.get("build", {}).get("rustflags", [])
    return {
        "path": str(path),
        "rustflags": flags,
        "present": True,
        "sha256": sha256_file(path),
    }


def build_ours(offline):
    configs = load_configs()
    build_cfg = configs["build"]
    name = build_cfg["bin"]
    target = BUILD / "target-agg"
    rev = source_revision()
    cmd = [
        "cargo",
        "build",
        "--release",
        "--locked",
        "-p",
        build_cfg["package"],
    ]
    if offline:
        cmd.append("--offline")
    print("building agg with the baseline in .cargo/config.toml")
    env = {
        "RUSTUP_TOOLCHAIN": RUST_TOOLCHAIN,
        "CARGO_TARGET_DIR": str(target),
        "CARGO_BUILD_JOBS": str(build_cfg.get("jobs", 6)),
    }
    if rev:
        env["AGG_GIT_REV"] = rev
    run(
        cmd,
        cwd=REPO,
        env=env,
    )
    src = target / "release" / build_cfg["bin"]
    dst = BIN / build_cfg["bin"]
    BIN.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        name: {
            "path": str(dst),
            "sha256": sha256_file(dst),
            "target_dir": str(target),
            "release_flags": release_flags(),
            "revision": rev,
        }
    }


def collect_reports(configs):
    """The self-report of every tier this CPU can run."""
    flags = cpu_flags()
    reports = {}
    for name, spec in configs["rows"].items():
        missing = [flag for flag in spec.get("requires", []) if flag not in flags]
        if missing:
            reports[name] = {
                "available": False,
                "reason": "CPU lacks " + ", ".join(missing),
            }
            continue
        report, proc = capture(BIN / spec["binary"], spec.get("env", {}))
        problems = failures(report, name)
        reports[name] = {
            "available": True,
            "returncode": proc.returncode,
            "summary": summarise(report),
            "failures": problems,
        }
    return reports


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
    env = {
        "RUSTUP_TOOLCHAIN": AERIE_TOOLCHAIN,
        "CARGO_TARGET_DIR": str(target),
        "CARGO_BUILD_JOBS": str(6),
    }
    print("building Aerie's Falcon baseline (Falcon-512 and Falcon-1024)")
    run(cmd, cwd=aerie, env=env)
    dst = BIN / "aerie-falcon-bench"
    shutil.copy2(find_bench(target), dst)
    result = {
        "path": str(dst),
        "sha256": sha256_file(dst),
        "target_dir": str(target),
    }
    if not capped:
        return result

    # The AVX2-capped baseline needs Aerie's run-time backend override. Use
    # the feature when the pinned tree has it; otherwise apply the small
    # patch in patches/ and revert it afterwards.
    target_capped = BUILD / "target-aerie-capped"
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
    if offline:
        base.append("--offline")
    env_capped = {**env, "CARGO_TARGET_DIR": str(target_capped)}
    proc = run(
        base
        + [
            "--features",
            "falcon-bench,aerie-simd/backend-override,"
            "aerie-keccak/backend-override",
        ],
        cwd=aerie,
        env=env_capped,
        check=False,
    )
    patched = False
    if proc.returncode != 0:
        print("aerie-simd/backend-override is not selectable; applying the cap patch")
        patch = ARTIFACT / "patches" / "aerie-simd-cap.patch"
        applied = run(["git", "apply", str(patch)], cwd=aerie, check=False)
        if applied.returncode != 0:
            result["capped_error"] = (applied.stderr or "").strip()[-2000:]
            return result
        patched = True
        proc = run(
            base + ["--features", "falcon-bench"],
            cwd=aerie,
            env=env_capped,
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


# The natural verifier per ML-DSA parameter set (artifact/natver/Makefile).
NATVER_BINARIES = {"44": "natver", "65": "natver65", "87": "natver87"}


def build_natver():
    natver = ARTIFACT / "natver"
    run(["make", "-j", "6"], cwd=natver)
    BIN.mkdir(parents=True, exist_ok=True)
    built = {}
    for params, name in NATVER_BINARIES.items():
        dst = BIN / name
        shutil.copy2(natver / name, dst)
        built[params] = {"path": str(dst), "sha256": sha256_file(dst)}
    return built


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
    parser.add_argument(
        "--skip-natver",
        action="store_true",
        help="do not build natver (it needs the fetched pq-crystals tree)",
    )
    args = parser.parse_args()
    offline = args.offline or os.environ.get("AGG_OFFLINE") == "1"
    configs = load_configs()
    # The release baseline is `.cargo/config.toml`. A RUSTFLAGS in the
    # environment would override it silently, so drop it and say so.
    inherited = [
        name
        for name in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTDOCFLAGS")
        if os.environ.pop(name, None) is not None
    ]
    if inherited:
        print("ignoring inherited " + ", ".join(inherited) + "; using .cargo/config.toml")

    built = build_ours(offline)
    reports = collect_reports(configs)

    natver = {} if args.skip_natver else build_natver()
    natver_counts = {}
    for params, entry in natver.items():
        counts = objdump_counts(Path(entry["path"]))
        if counts["total"]:
            raise SystemExit(f"natver ({params}) contains AVX-512 instructions: {counts}")
        natver_counts[params] = counts

    aerie = None
    if not args.skip_aerie:
        aerie = build_aerie(offline, args.with_aerie_capped)

    warnings = []
    for name, entry in reports.items():
        if entry.get("available") and entry.get("failures"):
            warnings.append(
                f"{name}: self-report mismatch: " + "; ".join(entry["failures"])
            )

    result = {
        "timestamp_utc": utc_stamp(),
        "commit": built[configs["build"]["bin"]]["revision"],
        "source_commit": pinned_commit(),
        "toolchains": {
            "rustc": output(["rustc", "-V"], env={"RUSTUP_TOOLCHAIN": RUST_TOOLCHAIN}),
            "cargo": output(["cargo", "-V"], env={"RUSTUP_TOOLCHAIN": RUST_TOOLCHAIN}),
            "aerie_rustc": output(
                ["rustc", "-V"], env={"RUSTUP_TOOLCHAIN": AERIE_TOOLCHAIN}
            ),
            "gcc": output(["gcc", "--version"]).splitlines()[0],
        },
        "binaries": built,
        "simd_reports": reports,
        "aerie": aerie,
        "natural_verifier": natver,
        "avx512_checks": {
            "natver": {params: {**counts, "hard_failure": True} for params, counts in natver_counts.items()}
        },
        "warnings": warnings,
    }
    write_json(BUILD / "build.json", result)
    print("build: OK")
    for name, entry in reports.items():
        if entry.get("available"):
            print(
                f"  {name:<7} cap={entry['summary']['cap']} "
                f"mldsa={entry['summary']['mldsa_backend']} "
                f"aerie={entry['summary']['aerie_backend']}"
            )
        else:
            print(f"  {name:<7} n/a ({entry['reason']})")
    for warning in warnings:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
