#!/usr/bin/env python3
"""Correctness gates at N=1024 for every kernel configuration."""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ARTIFACT,
    BIN,
    BUILD,
    EXPECTED,
    REPO,
    cpu_flags,
    git_rev,
    load_configs,
    new_result_dir,
    read_json,
    run,
    sha256_file,
    utc_stamp,
    write_json,
)
from runner import sh  # noqa: E402


def topology(result_dir):
    path = result_dir / "cpusets.json"
    if not path.exists():
        run(["python3", ARTIFACT / "scripts" / "topo.py", "--out", path], quiet=True)
    return read_json(path)


def machine(result_dir):
    path = result_dir / "machine.json"
    if not path.exists():
        run(
            ["python3", ARTIFACT / "scripts" / "probe.py", "--out", path],
            quiet=True,
        )
    return read_json(path)


def pin(topo, state):
    allocation = topo["configs"]["all"]
    prefix = []
    if (
        shutil.which("numactl")
        and topo.get("numa_nodes")
        and "No NUMA" not in state.get("numactl_H", "")
    ):
        node = allocation["numa_node"]
        prefix += ["numactl", f"--cpunodebind={node}", f"--membind={node}"]
    prefix += ["taskset", "-c", allocation["cpuset"]]
    return prefix, allocation


def make_records(binary, data_dir, env):
    keys = data_dir / "gates-keys-1024.bin"
    records = data_dir / "gates-records-1024.bin"
    if records.exists():
        return records
    data_dir.mkdir(parents=True, exist_ok=True)
    for cmd in [
        [str(binary), "keygen-batch", "--count", "1024", "--seed", "1", "--out", str(keys)],
        [
            str(binary),
            "sign-batch",
            "--keys",
            str(keys),
            "--count",
            "1024",
            "--seed",
            "1",
            "--out",
            str(records),
        ],
    ]:
        proc = sh(cmd, env)
        if proc.returncode != 0:
            raise SystemExit(f"record generation failed: {proc.stderr[-2000:]}")
    return records


def aggregate_all(configs, flags, prefix, allocation, records, out_dir):
    results = {}
    for name, spec in configs["rows"].items():
        missing = [f for f in spec.get("requires", []) if f not in flags]
        if missing:
            results[name] = {"available": False, "reason": "CPU lacks " + ",".join(missing)}
            continue
        binary = BIN / spec["binary"]
        if not binary.exists():
            raise SystemExit(f"{binary} is missing; run ./run.sh build")
        env = {}
        env.update(spec.get("env", {}))
        env["RAYON_NUM_THREADS"] = str(allocation["threads"])
        env["OMP_NUM_THREADS"] = str(allocation["threads"])
        env["AGG_GIT_REV"] = git_rev(REPO)
        out = out_dir / f"gates-{name}.bin"
        proc = sh(
            prefix
            + [
                str(binary),
                "aggregate",
                "--records",
                str(records),
                "--relation",
                "h-u",
                "--out",
                str(out),
            ],
            env,
        )
        if proc.returncode != 0:
            raise SystemExit(f"{name}: aggregate failed: {proc.stderr[-2000:]}")
        results[name] = {
            "available": True,
            "binary": spec["binary"],
            "bytes": out.stat().st_size,
            "sha256": sha256_file(out),
            "path": str(out),
        }
    return results


def verify_all(configs, flags, prefix, allocation, records, out_dir, results):
    verdicts = {}
    for name, spec in configs["rows"].items():
        if not results[name].get("available"):
            continue
        binary = BIN / spec["binary"]
        env = {}
        env.update(spec.get("env", {}))
        env["RAYON_NUM_THREADS"] = str(allocation["threads"])
        env["OMP_NUM_THREADS"] = str(allocation["threads"])
        proc = sh(
            prefix
            + [
                str(binary),
                "verify",
                "--records",
                str(records),
                "--aggregate",
                results[name]["path"],
            ],
            env,
        )
        verdicts[name] = proc.returncode == 0
    return verdicts


def negative_check(configs, prefix, allocation, records, out_dir):
    sample = out_dir / "gates-ifma.bin"
    if not sample.exists():
        sample = next(out_dir.glob("gates-*.bin"))
    broken = out_dir / "gates-broken.bin"
    data = bytearray(sample.read_bytes())
    data[len(data) // 2] ^= 1
    broken.write_bytes(data)
    spec = configs["rows"]["ifma"]
    binary = BIN / spec["binary"]
    env = {}
    env.update(spec.get("env", {}))
    env["RAYON_NUM_THREADS"] = str(allocation["threads"])
    proc = sh(
        prefix
        + [
            str(binary),
            "verify",
            "--records",
            str(records),
            "--aggregate",
            str(broken),
        ],
        env,
    )
    return proc.returncode == 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", help="results directory")
    parser.add_argument("--write-expected", action="store_true")
    parser.add_argument("--reference", help="compare the aggregate to this file")
    parser.add_argument("--with-baselines", action="store_true")
    args = parser.parse_args()
    result_dir = new_result_dir() if not args.results else Path(args.results)
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "data").mkdir(exist_ok=True)
    configs = load_configs()
    flags = cpu_flags()
    topo = topology(result_dir)
    state = machine(result_dir)
    prefix, allocation = pin(topo, state)
    env = {"RAYON_NUM_THREADS": str(allocation["threads"])}
    binary = BIN / "agg-v3"
    if not binary.exists():
        raise SystemExit(f"{binary} is missing; run ./run.sh build")
    records = make_records(binary, result_dir / "data", env)
    out_dir = result_dir / "gates"
    out_dir.mkdir(exist_ok=True)
    aggregates = aggregate_all(configs, flags, prefix, allocation, records, out_dir)
    verdicts = verify_all(
        configs, flags, prefix, allocation, records, out_dir, aggregates
    )
    negative = negative_check(configs, prefix, allocation, records, out_dir)

    expected_path = EXPECTED / "hashes.json"
    expected = read_json(expected_path) if expected_path.exists() else {}
    expected_entry = (
        expected.get("aggregates", {}).get("1024", {}).get("h-u")
        if expected
        else None
    )
    hashes = {
        name: entry["sha256"]
        for name, entry in aggregates.items()
        if entry.get("available")
    }
    sizes = {
        name: entry["bytes"]
        for name, entry in aggregates.items()
        if entry.get("available")
    }
    identical = len(set(hashes.values())) <= 1 and len(set(sizes.values())) <= 1
    expected_ok = (
        expected_entry is None
        or all(
            entry["sha256"] == expected_entry.get("sha256")
            and entry["bytes"] == expected_entry.get("bytes")
            for entry in aggregates.values()
            if entry.get("available")
        )
    )
    reference = None
    if args.reference:
        ref = Path(args.reference)
        reference = {
            "path": str(ref),
            "bytes": ref.stat().st_size,
            "sha256": sha256_file(ref),
            "matches": ref.stat().st_size in sizes.values()
            and sha256_file(ref) in hashes.values(),
        }

    native = None
    natver = BIN / "natver"
    if natver.exists():
        out = result_dir / "gates" / "natver-1024.json"
        proc = sh(
            [
                str(natver),
                "1024",
                "--warmup",
                "1",
                "--reps",
                "2",
                "--seed",
                "1",
                "--json",
                str(out),
            ],
            {
                "OMP_NUM_THREADS": str(allocation["threads"]),
                "OMP_PROC_BIND": "close",
                "OMP_PLACES": "threads",
            },
        )
        if proc.returncode == 0 and out.exists():
            native = read_json(out)
            native["ok"] = (
                native["rejected_uncached"] == 0 and native["rejected_cached"] == 0
            )
        else:
            native = {"ok": False, "error": (proc.stderr or "")[-2000:]}

    aerie = None
    aerie_bin = BIN / "aerie-falcon-bench"
    if args.with_baselines and aerie_bin.exists():
        proc = sh(
            prefix + [str(aerie_bin), "--bench", "1024"],
            {"RAYON_NUM_THREADS": str(allocation["threads"])},
        )
        aerie = {"ok": proc.returncode == 0, "row": proc.stdout.strip().splitlines()[-1:]}

    report = {
        "timestamp_utc": utc_stamp(),
        "commit": git_rev(REPO),
        "machine": state.get("cpu_model"),
        "n": 1024,
        "relation": "h-u",
        "threads": allocation["threads"],
        "cpuset": allocation["cpuset"],
        "aggregates": aggregates,
        "verify": verdicts,
        "negative_rejected": negative,
        "all_configurations_identical": identical,
        "expected": expected_entry,
        "expected_ok": expected_ok,
        "reference": reference,
        "native": native,
        "aerie": aerie,
        "machine_warnings": state.get("warnings", []),
    }
    write_json(result_dir / "gates.json", report)

    if args.write_expected:
        first = next(entry for entry in aggregates.values() if entry.get("available"))
        document = {
            "commit": git_rev(REPO),
            "relation": "h-u",
            "aggregates": {
                "1024": {
                    "h-u": {"bytes": first["bytes"], "sha256": first["sha256"]}
                }
            },
            "per_configuration": {
                name: {"bytes": entry["bytes"], "sha256": entry["sha256"]}
                for name, entry in aggregates.items()
                if entry.get("available")
            },
            "note": (
                "Generated from the pinned commit by ./run.sh gates --write-expected. "
                "Every configuration must reproduce these bytes."
            ),
        }
        write_json(expected_path, document)
        print(f"wrote {expected_path}")

    print("gates at N = 1024, H-u")
    for name, entry in aggregates.items():
        if entry.get("available"):
            print(
                f"  {name:<8} {entry['bytes']} B  {entry['sha256']}  "
                f"verify={'accept' if verdicts.get(name) else 'REJECT'}"
            )
        else:
            print(f"  {name:<8} n/a ({entry.get('reason')})")
    print(f"  negative check: {'rejected as expected' if negative else 'FAILED'}")
    print(f"  all configurations identical: {identical}")
    print(f"  expected file: {expected_entry if expected_entry else 'not present'}")
    if reference:
        print(
            f"  reference {reference['path']}: {reference['bytes']} B "
            f"{(reference['sha256'])} match={reference['matches']}"
        )
    if native:
        print(
            f"  natver: rejected {native.get('rejected_uncached')} uncached / "
            f"{native.get('rejected_cached')} cached"
        )

    ok = (
        identical
        and all(verdicts.values())
        and negative
        and expected_ok
        and (native is None or native.get("ok"))
        and (reference is None or reference["matches"])
        and (aerie is None or aerie["ok"])
    )
    print("gates: PASS" if ok else "gates: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
