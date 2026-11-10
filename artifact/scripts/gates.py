#!/usr/bin/env python3
"""Correctness gates at N=1024 for every kernel tier, ML-DSA parameter set and relation.

Per set (`--params`, default 44,65,87) and relation (`--relations`, default h-fips; the
reference relation H-exact with `--relations h-fips,h-exact`): the aggregates of every
tier must be byte-identical and equal to `expected/hashes.json` (keyed by set and
relation), every tier's verifier must accept, and a flipped byte must be rejected. Per
set, the aggregate of the first selected relation must not depend on the thread count
(RAYON_NUM_THREADS=1 against the pinned allocation's count).

With `--hash-field fp` the same gates run on the all-prime-field aggregates (the Keccak
chain proven over F_p) against `expected/hashes-fp.json`, for every selected relation
(default: H-FIPS and H-exact for ML-DSA-44, H-FIPS for 65 and 87, plus every relation the
file pins). Each aggregate must also be rejected by `verify --hash-field gf2`, and the
one-thread check runs for every relation. `hashes.json` is not read or written there.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ARTIFACT,
    BIN,
    DEFAULT_RELATIONS,
    EXPECTED,
    RELATIONS,
    built_revision,
    cpu_flags,
    load_configs,
    new_result_dir,
    parse_relations,
    pinned_commit,
    read_json,
    run,
    sha256_file,
    utc_stamp,
    write_json,
)
from runner import sh  # noqa: E402
from simd_report import capture, failures, summarise  # noqa: E402


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


def available_rows(configs, flags):
    rows = {}
    for name, spec in configs["rows"].items():
        missing = [flag for flag in spec.get("requires", []) if flag not in flags]
        rows[name] = {"available": not missing, "missing": missing, "spec": spec}
    return rows


PARAMS = ("44", "65", "87")
HASH_FIELDS = ("gf2", "fp")
# The relations of the default `fp` gate per set; the pinned file may add more.
FP_DEFAULT_RELATIONS = {"44": ("h-fips", "h-exact"), "65": ("h-fips",), "87": ("h-fips",)}


def field_args(field):
    """The `--hash-field` option; the default field adds nothing, so its commands are unchanged."""
    return [] if field == "gf2" else ["--hash-field", field]


def stem(params, relation, field):
    """The file-name stem of a set's aggregates; `fp` files carry the field."""
    return f"gates-{params}-{relation}" if field == "gf2" else f"gates-fp-{params}-{relation}"


def aggregate_cmd(binary, params, records, relation, out, field):
    return [
        str(binary),
        "aggregate",
        "--params",
        params,
        "--records",
        str(records),
        "--relation",
        relation,
        "--out",
        str(out),
    ] + field_args(field)


def verify_cmd(binary, params, records, aggregate, relation, field):
    return [
        str(binary),
        "verify",
        "--params",
        params,
        "--records",
        str(records),
        "--aggregate",
        str(aggregate),
        "--relation",
        relation,
    ] + field_args(field)
# The records of the reference aggregates: 1,024 keys from seed 42, records signed with seed 43.
KEY_SEED = "42"
RECORD_SEED = "43"


def make_records(binary, data_dir, env, params):
    keys = data_dir / f"gates-keys-{params}-1024.bin"
    records = data_dir / f"gates-records-{params}-1024.bin"
    if records.exists():
        return records
    data_dir.mkdir(parents=True, exist_ok=True)
    for cmd in [
        [
            str(binary),
            "keygen-batch",
            "--params",
            params,
            "--count",
            "1024",
            "--seed",
            KEY_SEED,
            "--out",
            str(keys),
        ],
        [
            str(binary),
            "sign-batch",
            "--params",
            params,
            "--keys",
            str(keys),
            "--count",
            "1024",
            "--seed",
            RECORD_SEED,
            "--out",
            str(records),
        ],
    ]:
        proc = sh(cmd, env)
        if proc.returncode != 0:
            raise SystemExit(f"record generation failed: {proc.stderr[-2000:]}")
    return records


def check_reports(configs, rows):
    reports = {}
    for name, entry in rows.items():
        spec = entry["spec"]
        if not entry["available"]:
            reports[name] = {
                "available": False,
                "reason": "CPU lacks " + ", ".join(entry["missing"]),
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
        if problems:
            raise SystemExit(
                f"{name}: the self-report does not match the tier: " + "; ".join(problems)
            )
    return reports


def aggregate_all(
    configs, rows, prefix, allocation, records, out_dir, params, relation, revision, field
):
    results = {}
    for name, spec in configs["rows"].items():
        if not rows[name]["available"]:
            results[name] = {
                "available": False,
                "reason": "CPU lacks " + ", ".join(rows[name]["missing"]),
            }
            continue
        binary = BIN / spec["binary"]
        if not binary.exists():
            raise SystemExit(f"{binary} is missing; run ./run.sh build")
        env = {}
        env.update(spec.get("env", {}))
        env["RAYON_NUM_THREADS"] = str(allocation["threads"])
        env["OMP_NUM_THREADS"] = str(allocation["threads"])
        env["AGG_GIT_REV"] = revision
        out = out_dir / f"{stem(params, relation, field)}-{name}.bin"
        proc = sh(prefix + aggregate_cmd(binary, params, records, relation, out, field), env)
        if proc.returncode != 0:
            raise SystemExit(
                f"{name}, ML-DSA-{params}, {relation}: aggregate failed: {proc.stderr[-2000:]}"
            )
        results[name] = {
            "available": True,
            "binary": spec["binary"],
            "bytes": out.stat().st_size,
            "sha256": sha256_file(out),
            "path": str(out),
        }
    return results


def verify_all(configs, rows, prefix, allocation, records, results, params, relation, field):
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
            + verify_cmd(binary, params, records, results[name]["path"], relation, field),
            env,
        )
        verdicts[name] = proc.returncode == 0
    return verdicts


def first_tier(rows):
    order = ["ifma", "avx512", "avx2", "scalar"]
    return next(
        (candidate for candidate in order if rows.get(candidate, {}).get("available")),
        None,
    )


def threads_check(
    configs, rows, prefix, allocation, records, out_dir, params, relation, results, field
):
    """The aggregate of the first available tier on one thread equals the pinned run's."""
    name = first_tier(rows)
    if name is None or allocation["threads"] == 1:
        return None
    spec = configs["rows"][name]
    env = {}
    env.update(spec.get("env", {}))
    env["RAYON_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    out = out_dir / f"{stem(params, relation, field)}-{name}-t1.bin"
    proc = sh(
        prefix + aggregate_cmd(BIN / spec["binary"], params, records, relation, out, field), env
    )
    if proc.returncode != 0:
        return {"tier": name, "ok": False, "error": (proc.stderr or "")[-2000:]}
    return {
        "tier": name,
        "relation": relation,
        "threads": [1, allocation["threads"]],
        "ok": sha256_file(out) == results[name]["sha256"],
    }


def negative_check(configs, rows, prefix, allocation, records, out_dir, params, relation, field):
    name = first_tier(rows)
    if name is None:
        return False, None
    sample = out_dir / f"{stem(params, relation, field)}-{name}.bin"
    broken = out_dir / f"{stem(params, relation, field)}-broken.bin"
    data = bytearray(sample.read_bytes())
    data[len(data) // 2] ^= 1
    broken.write_bytes(data)
    spec = configs["rows"][name]
    binary = BIN / spec["binary"]
    env = {}
    env.update(spec.get("env", {}))
    env["RAYON_NUM_THREADS"] = str(allocation["threads"])
    proc = sh(prefix + verify_cmd(binary, params, records, broken, relation, field), env)
    return proc.returncode == 1, name


def cross_field_check(configs, rows, prefix, allocation, records, out_dir, params, relation):
    """An `fp` aggregate is rejected (exit status 1) by the verifier of the default field."""
    name = first_tier(rows)
    if name is None:
        return False
    spec = configs["rows"][name]
    env = dict(spec.get("env", {}))
    env["RAYON_NUM_THREADS"] = str(allocation["threads"])
    sample = out_dir / f"{stem(params, relation, 'fp')}-{name}.bin"
    proc = sh(
        prefix + verify_cmd(BIN / spec["binary"], params, records, sample, relation, "gf2"), env
    )
    return proc.returncode == 1


def sde_check(configs, rows, timeout=1800):
    """Emulate an AVX2-only and an AVX-512-without-IFMA CPU, if SDE is installed."""
    exe = shutil.which("sde64") or shutil.which("sde")
    if exe is None:
        return {"available": False, "reason": "sde64 is not installed"}
    version = run([exe, "--version"], check=False, capture=True, quiet=True)
    help_proc = run([exe, "-help"], check=False, capture=True, quiet=True)
    help_text = (help_proc.stdout or "") + (help_proc.stderr or "")
    chips = {}
    for tier, chip in (("avx2", "-hsw"), ("avx512", "-skx")):
        if re.search(rf"(^|\s){re.escape(chip)}(\s|$)", help_text):
            chips[tier] = chip
    if not chips:
        return {
            "available": False,
            "reason": "sde64 -help lists neither -hsw nor -skx",
            "version": ((version.stdout or "") + (version.stderr or "")).strip(),
        }
    runs = {}
    for tier, chip in chips.items():
        if not rows[tier]["available"]:
            runs[tier] = {"ok": True, "skipped": "CPU lacks the tier"}
            continue
        spec = configs["rows"][tier]
        env = os.environ.copy()
        env.update({k: str(v) for k, v in spec.get("env", {}).items()})
        env["RAYON_NUM_THREADS"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        cmd = [
            exe,
            chip,
            "--",
            str(BIN / spec["binary"]),
            "bench",
            "--count",
            "256",
            "--warmup",
            "1",
            "--reps",
            "1",
        ]
        try:
            proc = subprocess.run(
                cmd,
                env=env,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
            runs[tier] = {
                "chip": chip,
                "returncode": proc.returncode,
                "ok": proc.returncode == 0,
                "stderr_tail": (proc.stderr or "")[-2000:],
            }
        except subprocess.TimeoutExpired:
            runs[tier] = {"chip": chip, "ok": False, "error": "timed out"}
    return {
        "available": True,
        "executable": exe,
        "chips": chips,
        "runs": runs,
    }


NOTES = {
    "gf2": (
        "Generated by ./run.sh gates --write-expected, keyed by ML-DSA parameter set, "
        "batch size and relation. Every tier must reproduce these bytes."
    ),
    "fp": (
        "F_p hash field (--hash-field fp) aggregates, generated by ./run.sh gates "
        "--hash-field fp --write-expected, keyed by ML-DSA parameter set, batch size and "
        "relation. Every tier must reproduce these bytes."
    ),
}
EXPECTED_FILES = {"gf2": "hashes.json", "fp": "hashes-fp.json"}
NATVER_BINARIES = {"44": "natver", "65": "natver65", "87": "natver87"}


def expected_for(expected, params, relation):
    """The expected 1K entry of a set and relation (a file without set keys holds ML-DSA-44's)."""
    aggregates = expected.get("aggregates", {}) if expected else {}
    if params in aggregates:
        return aggregates[params].get("1024", {}).get(relation)
    if params == "44":
        return aggregates.get("1024", {}).get(relation)
    return None


def natver_check(params, allocation, result_dir):
    """The pq-crystals AVX2 batch verifier of the set accepts its own 1K dataset."""
    natver = BIN / NATVER_BINARIES[params]
    if not natver.exists():
        return None
    out = result_dir / "gates" / f"natver-{params}-1024.json"
    proc = sh(
        [str(natver), "1024", "--warmup", "1", "--reps", "2", "--seed", "1", "--json", str(out)],
        {
            "OMP_NUM_THREADS": str(allocation["threads"]),
            "OMP_PROC_BIND": "close",
            "OMP_PLACES": "threads",
        },
    )
    if proc.returncode == 0 and out.exists():
        native = read_json(out)
        native["ok"] = native["rejected_uncached"] == 0 and native["rejected_cached"] == 0
        return native
    return {"ok": False, "error": (proc.stderr or "")[-2000:]}


def fp_relations(expected, params):
    """The default `fp` relations of a set: the fixed ones, then those the pinned file adds."""
    pinned = expected.get("aggregates", {}).get(params, {}).get("1024", {}) if expected else {}
    chosen = set(FP_DEFAULT_RELATIONS[params]) | {r for r in pinned if r in RELATIONS}
    return [relation for relation in RELATIONS if relation in chosen]


def write_expected(path, expected, per_set, revision, field):
    """Re-pin the 1K entries that ran; the entries of the sets and relations not run are kept."""
    # A file without set keys holds ML-DSA-44's. Every tier must match the one entry per
    # set and relation, so per-tier copies are not stored.
    document = dict(expected)
    if "1024" in document.get("aggregates", {}):
        document["aggregates"] = {"44": document["aggregates"]}
    document.pop("per_configuration", None)
    document.pop("relation", None)
    document["commit"] = revision
    document["records"] = (
        f"keygen-batch --params P --count 1024 --seed {KEY_SEED}; "
        f"sign-batch --params P --count 1024 --seed {RECORD_SEED}"
    )
    document.setdefault("aggregates", {})
    # Entries of relations this artifact no longer accepts are dropped.
    for sizes in document["aggregates"].values():
        for entries in sizes.values():
            for relation in [r for r in entries if r not in RELATIONS]:
                del entries[relation]
    for params, result in per_set.items():
        entries = document["aggregates"].setdefault(params, {}).setdefault("1024", {})
        for relation, outcome in result["relations"].items():
            available = [
                entry for entry in outcome["aggregates"].values() if entry.get("available")
            ]
            if not outcome["all_configurations_identical"]:
                raise SystemExit(
                    f"ML-DSA-{params}, {relation}: the tiers disagree; not writing {path}"
                )
            first = available[0]
            entries[relation] = {"bytes": first["bytes"], "sha256": first["sha256"]}
    document["note"] = NOTES[field]
    write_json(path, document)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", help="results directory")
    parser.add_argument("--write-expected", action="store_true")
    parser.add_argument("--reference", help="compare the aggregate to this file")
    parser.add_argument("--with-baselines", action="store_true")
    parser.add_argument(
        "--params",
        default=",".join(PARAMS),
        type=lambda v: [p.strip() for p in v.split(",") if p.strip()],
        help="comma-separated ML-DSA parameter sets (default 44,65,87)",
    )
    parser.add_argument(
        "--relations",
        default=None,
        help="comma-separated relations (default h-fips; the reference relation H-exact "
        "with --relations h-fips,h-exact; under --hash-field fp the default is h-fips,h-exact "
        "for 44 and h-fips for 65 and 87, plus the relations hashes-fp.json pins)",
    )
    parser.add_argument(
        "--hash-field",
        choices=HASH_FIELDS,
        default="gf2",
        help="gf2 (default): the mixed-field aggregates against hashes.json; fp: the "
        "all-prime-field aggregates against hashes-fp.json",
    )
    parser.add_argument(
        "--sde",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the optional Intel SDE check when sde64 is installed",
    )
    args = parser.parse_args()
    unknown = [params for params in args.params if params not in PARAMS]
    if unknown or not args.params:
        parser.error(f"--params: a comma-separated subset of {','.join(PARAMS)}, not {unknown}")
    field = args.hash_field
    explicit_relations = args.relations is not None
    try:
        args.relations = parse_relations(
            args.relations if explicit_relations else ",".join(DEFAULT_RELATIONS)
        )
    except ValueError as error:
        parser.error(f"--{error}")
    result_dir = new_result_dir() if not args.results else Path(args.results)
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "data").mkdir(exist_ok=True)
    configs = load_configs()
    flags = cpu_flags()
    rows = available_rows(configs, flags)
    topo = topology(result_dir)
    state = machine(result_dir)
    prefix, allocation = pin(topo, state)
    env = {"RAYON_NUM_THREADS": str(allocation["threads"])}
    binary = BIN / "agg"
    if not binary.exists():
        raise SystemExit(f"{binary} is missing; run ./run.sh build")

    reports = check_reports(configs, rows)
    out_dir = result_dir / "gates"
    out_dir.mkdir(exist_ok=True)
    # The SDE run checks the tier caps, which do not depend on the hash field.
    if not args.sde:
        sde = {"available": False, "reason": "disabled with --no-sde"}
    elif field != "gf2":
        sde = {"available": False, "reason": "runs in the gf2 gate only"}
    else:
        sde = sde_check(configs, rows)
    expected_path = EXPECTED / EXPECTED_FILES[field]
    expected = read_json(expected_path) if expected_path.exists() else {}

    revision = built_revision()
    per_set = {}
    for params in args.params:
        records = make_records(binary, result_dir / "data", env, params)
        per_relation = {}
        relations = (
            args.relations if field == "gf2" or explicit_relations
            else fp_relations(expected, params)
        )
        for relation in relations:
            aggregates = aggregate_all(
                configs, rows, prefix, allocation, records, out_dir, params, relation, revision,
                field,
            )
            verdicts = verify_all(
                configs, rows, prefix, allocation, records, aggregates, params, relation, field
            )
            negative, negative_tier = negative_check(
                configs, rows, prefix, allocation, records, out_dir, params, relation, field
            )
            cross_field = None
            threads = None
            if field != "gf2":
                cross_field = cross_field_check(
                    configs, rows, prefix, allocation, records, out_dir, params, relation
                )
                threads = threads_check(
                    configs, rows, prefix, allocation, records, out_dir, params, relation,
                    aggregates, field,
                )
            expected_entry = expected_for(expected, params, relation)
            available = [entry for entry in aggregates.values() if entry.get("available")]
            hashes = {entry["sha256"] for entry in available}
            sizes = {entry["bytes"] for entry in available}
            identical = len(hashes) <= 1 and len(sizes) <= 1
            # --write-expected re-pins the bytes, so the previous pin does not gate the run.
            # The fp gate also fails on a relation its file does not pin.
            expected_ok = (
                args.write_expected
                or (expected_entry is None and field == "gf2")
                or (
                    expected_entry is not None
                    and all(
                        entry["sha256"] == expected_entry.get("sha256")
                        and entry["bytes"] == expected_entry.get("bytes")
                        for entry in available
                    )
                )
            )
            reference = None
            if args.reference and params == args.params[0] and relation == relations[0]:
                ref = Path(args.reference)
                reference = {
                    "path": str(ref),
                    "bytes": ref.stat().st_size,
                    "sha256": sha256_file(ref),
                    "matches": ref.stat().st_size in sizes and sha256_file(ref) in hashes,
                }
            per_relation[relation] = {
                "aggregates": aggregates,
                "verify": verdicts,
                "negative_rejected": negative,
                "negative_tier": negative_tier,
                "all_configurations_identical": identical,
                "expected": expected_entry,
                "expected_ok": expected_ok,
                "reference": reference,
            }
            if field != "gf2":
                per_relation[relation]["cross_field_rejected"] = cross_field
                per_relation[relation]["threads"] = threads
        threads = None
        native = None
        if field == "gf2":
            first_relation = relations[0]
            threads = threads_check(
                configs,
                rows,
                prefix,
                allocation,
                records,
                out_dir,
                params,
                first_relation,
                per_relation[first_relation]["aggregates"],
                field,
            )
            native = natver_check(params, allocation, result_dir)
        per_set[params] = {
            "relations": per_relation,
            "threads": threads,
            "native": native,
        }

    aerie = None
    aerie_bin = BIN / "aerie-falcon-bench"
    if args.with_baselines and aerie_bin.exists() and field == "gf2":
        proc = sh(
            prefix + [str(aerie_bin), "--bench", "1024"],
            {"RAYON_NUM_THREADS": str(allocation["threads"])},
        )
        aerie = {"ok": proc.returncode == 0, "row": proc.stdout.strip().splitlines()[-1:]}

    report = {
        "timestamp_utc": utc_stamp(),
        "commit": revision,
        "source_commit": pinned_commit(),
        "machine": state.get("cpu_model"),
        "n": 1024,
        "hash_field": field,
        "relations": {params: list(result["relations"]) for params, result in per_set.items()}
        if field != "gf2"
        else args.relations,
        "threads": allocation["threads"],
        "cpuset": allocation["cpuset"],
        "tiers": [name for name, entry in rows.items() if entry["available"]],
        "simd_reports": reports,
        "params": per_set,
        "sde": sde,
        "aerie": aerie,
        "machine_warnings": state.get("warnings", []),
    }
    write_json(result_dir / ("gates.json" if field == "gf2" else f"gates-{field}.json"), report)

    if args.write_expected:
        write_expected(expected_path, expected, per_set, revision, field)

    ok = True
    for params, result in per_set.items():
        for relation, outcome in result["relations"].items():
            suffix = "" if field == "gf2" else f", hash field {field}"
            print(f"gates at N = 1024, ML-DSA-{params}, {relation}{suffix}")
            for name, entry in outcome["aggregates"].items():
                if entry.get("available"):
                    suffix = " (gate only)" if configs["rows"][name].get("gates_only") else ""
                    verdict = "accept" if outcome["verify"].get(name) else "REJECT"
                    print(
                        f"  {name:<7} {entry['bytes']} B  {entry['sha256']}  "
                        f"verify={verdict}{suffix}"
                    )
                else:
                    print(f"  {name:<7} n/a ({entry.get('reason')})")
            negative = outcome["negative_rejected"]
            print(
                f"  negative check: {'rejected as expected' if negative else 'FAILED'} "
                f"({outcome['negative_tier']})"
            )
            if field != "gf2":
                cross = outcome["cross_field_rejected"]
                print(
                    "  cross-field check: "
                    + ("rejected under gf2 as expected" if cross else "FAILED")
                )
                threads = outcome["threads"]
                if threads:
                    print(
                        f"  threads {threads.get('threads')}: "
                        f"{'identical' if threads['ok'] else 'DIFFER'} ({threads['tier']})"
                    )
                ok = ok and cross and (threads is None or threads["ok"])
            print(f"  all configurations identical: {outcome['all_configurations_identical']}")
            entry = outcome["expected"]
            print(f"  expected file: {entry if entry else 'not present'}")
            reference = outcome["reference"]
            if reference:
                print(
                    f"  reference {reference['path']}: {reference['bytes']} B "
                    f"{reference['sha256']} match={reference['matches']}"
                )
            ok = ok and (
                outcome["all_configurations_identical"]
                and all(outcome["verify"].values())
                and negative
                and outcome["expected_ok"]
                and (reference is None or reference["matches"])
            )
        threads = result["threads"]
        if threads:
            print(
                f"  ML-DSA-{params} threads {threads.get('threads')}: "
                f"{'identical' if threads['ok'] else 'DIFFER'} "
                f"({threads['tier']}, {threads.get('relation')})"
            )
        native = result["native"]
        if native:
            print(
                f"  ML-DSA-{params} natver: rejected {native.get('rejected_uncached')} uncached / "
                f"{native.get('rejected_cached')} cached"
            )
        ok = ok and (threads is None or threads["ok"]) and (native is None or native.get("ok"))
    for name, entry in reports.items():
        if entry.get("available"):
            summary = entry["summary"]
            print(
                f"  report {name:<7} cap={summary['cap']} "
                f"mldsa={summary['mldsa_backend']} aerie={summary['aerie_backend']}"
            )
    if sde.get("available"):
        parts = []
        for tier, run_result in sde["runs"].items():
            if run_result.get("skipped"):
                parts.append(f"{tier} skipped")
            else:
                parts.append(
                    f"{tier} {run_result['chip']} "
                    + ("ok" if run_result.get("ok") else "FAILED")
                )
        print("  sde: " + ", ".join(parts))
    else:
        print(f"  sde: skipped ({sde.get('reason')})")

    sde_ok = not sde.get("available") or all(
        run_result.get("ok") for run_result in sde.get("runs", {}).values()
    )
    ok = ok and sde_ok and (aerie is None or aerie["ok"])
    print("gates: PASS" if ok else "gates: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
