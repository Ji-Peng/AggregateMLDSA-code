#!/usr/bin/env python3
"""Run the quick or full experiment matrix on one machine."""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ARTIFACT,
    BIN,
    BUILD,
    REPO,
    TABLES,
    cpu_flags,
    git_rev,
    load_configs,
    load_matrix,
    median,
    new_result_dir,
    pinned_commit,
    read_json,
    run,
    utc_stamp,
    write_json,
)
from stats import describe, flags as cell_flags  # noqa: E402
from simd_report import capture, failures, summarise  # noqa: E402


def sh(cmd, env, cwd=None, timeout=None):
    full_env = os.environ.copy()
    full_env.update({k: str(v) for k, v in env.items()})
    proc = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
        env=full_env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return proc


def rotate(items, r):
    if not items:
        return items
    k = r % len(items)
    return items[k:] + items[:k]


def process_chunks(reps, processes):
    """Split `reps` timed runs into `processes` positive, balanced chunks."""
    processes = max(1, min(int(processes or 1), int(reps)))
    base, extra = divmod(int(reps), processes)
    return [base + (1 if index < extra else 0) for index in range(processes)]


def read_stat():
    stats = {}
    try:
        with open("/proc/stat", encoding="utf-8") as f:
            for line in f:
                if not line.startswith("cpu") or line.startswith("cpu "):
                    continue
                parts = line.split()
                cpu = int(parts[0][3:])
                values = [int(v) for v in parts[1:]]
                busy = sum(values[:8]) - values[3] - values[4]
                stats[cpu] = (busy, sum(values))
    except OSError:
        pass
    return stats


def foreign_fraction(before, after, our_cpus):
    busy = total = 0
    for cpu, (b0, t0) in before.items():
        if cpu in our_cpus or cpu not in after:
            continue
        b1, t1 = after[cpu]
        busy += b1 - b0
        total += t1 - t0
    return busy / total if total else 0.0


def start_sampler(result_dir, allocation):
    path = result_dir / "logs" / f"freq-{allocation['cpuset'].replace(',', '_')}.log"
    cpus = [str(c) for c in allocation["cpus"]]
    proc = subprocess.Popen(
        [str(ARTIFACT / "scripts" / "sampler.sh"), str(path), "100"] + cpus,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc, path


def stop_sampler(proc):
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def frequency(path, cpus):
    values = []
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 3 and int(parts[1]) in cpus:
                values.append(float(parts[2]) / 1000.0)
    except OSError:
        pass
    if not values:
        return {"median_mhz": None, "min_mhz": None}
    return {"median_mhz": median(values), "min_mhz": min(values)}


def allocation_for(topo, requested, fallbacks):
    entry = topo["configs"].get(requested)
    if entry and entry.get("available"):
        return requested, entry, None
    seen = set()
    candidate = requested
    while candidate in fallbacks and candidate not in seen:
        seen.add(candidate)
        candidate = fallbacks[candidate]
        entry = topo["configs"].get(candidate)
        if entry and entry.get("available"):
            return candidate, entry, f"fallback from {requested}"
    return None, None, f"{requested} not available on this machine"


def pin_prefix(topo, allocation):
    prefix = []
    if allocation["numa_node"] is not None and "No NUMA" not in topo.get("_numactl", ""):
        if shutil_which("numactl"):
            prefix += [
                "numactl",
                f"--cpunodebind={allocation['numa_node']}",
                f"--membind={allocation['numa_node']}",
            ]
    prefix += ["taskset", "-c", allocation["cpuset"]]
    return prefix


def shutil_which(tool):
    from shutil import which
    return which(tool)


class Runner:
    def __init__(self, args):
        self.args = args
        self.configs = load_configs()
        self.matrix = load_matrix(args.mode)
        self.flags = cpu_flags()
        self.result_dir = new_result_dir() if not args.results else Path(args.results)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        (self.result_dir / "raw").mkdir(exist_ok=True)
        (self.result_dir / "logs").mkdir(exist_ok=True)
        (self.result_dir / "data").mkdir(exist_ok=True)
        # The archive build has no .git, so prefer the pin in sources.lock and
        # fall back to the tree revision in a development checkout.
        self.code_commit = pinned_commit() or git_rev(REPO)
        self.cells = []
        self.skipped = []
        self.reports = {}
        self.machine = self.load_machine()
        self.topo = self.load_topology()
        self.topo["_numactl"] = self.machine.get("numactl_H", "")
        (self.result_dir / "machine.json").write_text(
            json.dumps(self.machine, indent=2) + "\n", encoding="utf-8"
        )
        (self.result_dir / "cpusets.json").write_text(
            json.dumps({k: v for k, v in self.topo.items() if k != "_numactl"}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        warnings = self.machine.get("warnings", [])
        (self.result_dir / "warnings.txt").write_text(
            "".join(f"{w}\n" for w in warnings), encoding="utf-8"
        )
        for warning in warnings:
            print(f"warning: {warning}")

    def load_machine(self):
        path = self.result_dir / "machine.json"
        if path.exists():
            return read_json(path)
        proc = run(
            ["python3", ARTIFACT / "scripts" / "probe.py", "--out", path],
            check=False,
            quiet=True,
        )
        if not path.exists():
            raise SystemExit("probe.py did not write machine.json")
        return read_json(path)

    def load_topology(self):
        path = self.result_dir / "cpusets.json"
        run(
            ["python3", ARTIFACT / "scripts" / "topo.py", "--out", path],
            quiet=True,
        )
        return read_json(path)

    def rows(self, requested):
        requested = requested or sorted(self.configs["rows"])
        if self.args.configs:
            requested = [name for name in requested if name in self.args.configs]
        selected = []
        for name in requested:
            spec = self.configs["rows"].get(name)
            if spec is None:
                raise SystemExit(f"unknown configuration {name}")
            if spec.get("gates_only"):
                continue
            missing = [flag for flag in spec.get("requires", []) if flag not in self.flags]
            if missing:
                self.skipped.append(
                    {
                        "kind": "configuration",
                        "name": name,
                        "reason": "CPU lacks " + ", ".join(missing),
                    }
                )
                print(f"skip {name}: CPU lacks {', '.join(missing)}")
                continue
            binary = BIN / spec["binary"]
            if not binary.exists():
                raise SystemExit(f"{binary} is missing; run ./run.sh build")
            if name not in self.reports:
                report, _ = capture(binary, spec.get("env", {}))
                problems = failures(report, name)
                self.reports[name] = {
                    "summary": summarise(report),
                    "failures": problems,
                }
                if problems:
                    raise SystemExit(
                        f"{name}: the self-report does not match the tier: "
                        + "; ".join(problems)
                    )
            selected.append(name)
        return selected

    def allocations(self, group, defaults):
        if self.args.allocations:
            return self.args.allocations
        return group.get("allocations") or defaults

    def allocation_plan(self, group, defaults):
        """Resolve requested allocations, dropping duplicates on small machines.

        A machine with eight cores resolves both 16C32T and 8C16T to the same
        CPU set. Run that set once, under the name the machine can fill.
        """
        requested = list(self.allocations(group, defaults))
        requested.sort(
            key=lambda name: 0
            if self.topo["configs"].get(name, {}).get("available")
            else 1
        )
        plan = []
        used = set()
        for name in requested:
            actual, allocation, note = allocation_for(
                self.topo, name, self.configs["topology"].get("fallbacks", {})
            )
            if allocation is None:
                self.skipped.append(
                    {"kind": "allocation", "name": name, "reason": note}
                )
                print(f"skip allocation {name}: {note}")
                continue
            if allocation["cpuset"] in used:
                self.skipped.append(
                    {
                        "kind": "allocation",
                        "name": name,
                        "reason": f"same CPU set as {actual}",
                    }
                )
                print(f"skip allocation {name}: same CPU set as {actual}")
                continue
            used.add(allocation["cpuset"])
            plan.append((name, actual, allocation, note))
        return plan

    def env_for(self, name, allocation, extra=None):
        spec = self.configs["rows"][name]
        env = {}
        env.update(self.configs.get("runtime", {}).get("extra_omp", {}))
        env.update(spec.get("env", {}))
        if extra:
            env.update(extra)
        env["RAYON_NUM_THREADS"] = str(allocation["threads"])
        env["OMP_NUM_THREADS"] = str(allocation["threads"])
        env["AGG_GIT_REV"] = self.code_commit
        return env

    def invoke_ours(self, name, allocation, n, relation, extra, warmup, reps, trace=False):
        spec = self.configs["rows"][name]
        binary = BIN / spec["binary"]
        out = self.result_dir / "raw" / "parts"
        out.mkdir(exist_ok=True)
        tag = f"{name}-{allocation['cpuset'].replace(',', '_')}-{n}-{relation}-{time.time_ns()}"
        json_path = out / f"{tag}.json"
        log_path = self.result_dir / "logs" / f"{tag}.log"
        env = self.env_for(name, allocation, extra)
        if trace:
            env["AGG_PHASE_STAMPS"] = "1"
        cmd = pin_prefix(self.topo, allocation) + [
            str(binary),
            "bench",
            "--count",
            str(n),
            "--warmup",
            str(warmup),
            "--reps",
            str(reps),
            "--relation",
            relation,
            "--json",
            str(json_path),
        ]
        proc = sh(cmd, env)
        log_path.write_text((proc.stdout or "") + (proc.stderr or ""), encoding="utf-8")
        if proc.returncode != 0:
            raise SystemExit(
                f"bench failed for {name} at N={n}: rc={proc.returncode}\n"
                + (proc.stderr or "")[-4000:]
            )
        if not json_path.exists():
            raise SystemExit(f"bench wrote no JSON for {name} at N={n}")
        part = read_json(json_path)
        part["_cmd"] = " ".join(str(c) for c in cmd)
        part["_env"] = {k: v for k, v in env.items() if k not in ("PATH",)}
        part["_json"] = str(json_path)
        part["_log"] = str(log_path)
        return part

    def process_count(self, group, reps):
        requested = (
            self.args.processes
            if self.args.processes is not None
            else group.get("processes", 2)
        )
        return max(1, min(int(requested), int(reps)))

    def run_ours_group(self, group, rows, sizes, relations, reps, warmup):
        for requested, actual, allocation, note in self.allocation_plan(
            group, ["8C16T"]
        ):
            for n in sizes:
                for relation in relations:
                    self.run_ours_cell(
                        group,
                        rows,
                        requested,
                        actual,
                        allocation,
                        note,
                        n,
                        relation,
                        reps,
                        warmup,
                    )

    def run_ours_cell(self, group, rows, requested, actual, allocation, note, n, relation, reps, warmup):
        tag = group["tag"]
        trace = bool(group.get("traced"))
        chunks = process_chunks(reps, self.process_count(group, reps))
        cells = {}
        for name in rows:
            cell = self.base_cell(
                "ours", name, tag, requested, actual, allocation, note, n, relation
            )
            cell["warmup"] = warmup
            cell["reps"] = reps
            cell["protocol"] = "warm"
            cell["process_count"] = len(chunks)
            cells[name] = cell
        before = read_stat()
        sampler, freq_path = start_sampler(self.result_dir, allocation)
        started = utc_stamp()
        try:
            # One process per chunk, round-robin across tiers. Each process
            # pays one warm-up and then contributes its timed runs.
            for r in range(len(chunks)):
                for name in rotate(rows, r):
                    part = self.invoke_ours(
                        name, allocation, n, relation, None, warmup, chunks[r], trace=trace
                    )
                    cell = cells[name]
                    cell["times_ms"].extend(part["prove_ms"])
                    cell["verify_uncached_ms"].extend(part["verify_uncached_ms"])
                    cell["verify_cached_ms"].extend(part["verify_cached_ms"])
                    cell["parts"].append(part)
                    cell["processes"].append(
                        {
                            "json": part["_json"],
                            "warmup": warmup,
                            "reps": chunks[r],
                            "prove_ms": part["prove_ms"],
                        }
                    )
                    cell["aggregate_bytes"] = part["aggregate_bytes"]
                    cell["aggregate_sha256"] = part["aggregate_sha256"]
                    cell["kernels"] = part["kernels"]
                    peak = part.get("prove_peak_kib")
                    if peak:
                        cell["peak_rss_kib"] = max(cell.get("peak_rss_kib") or 0, peak)
        finally:
            stop_sampler(sampler)
        after = read_stat()
        for name, cell in cells.items():
            cell["start_utc"] = started
            cell["end_utc"] = utc_stamp()
            cell["freq_mhz"] = frequency(freq_path, set(allocation["cpus"]))
            cell["foreign_busy"] = foreign_fraction(before, after, set(allocation["cpus"]))
            cell["loadavg"] = self.load_before_after(before)
            self.finish_cell(cell)
            self.cells.append(cell)
        self.check_hashes(cells, tag, n, relation)

    def load_before_after(self, before):
        return {"start": self.machine.get("loadavg"), "end": list(os.getloadavg())}

    def base_cell(self, system, config, tag, requested, actual, allocation, note, n, relation):
        spec = self.configs["rows"].get(config, {})
        binary = BIN / spec.get("binary", "")
        alloc_tag = actual if requested == actual else f"{requested}-to-{actual}"
        return {
            "run_id": f"{tag}-{config}-{alloc_tag}-n{n}-{relation}",
            "system": system,
            "tag": tag,
            "config": config,
            "label": spec.get("label", config),
            "binary": spec.get("binary"),
            "binary_sha256": sha256(binary) if binary.exists() else None,
            "threads": allocation["threads"],
            "allocation": actual,
            "allocation_requested": requested,
            "allocation_note": note,
            "cpuset": allocation["cpuset"],
            "numa_node": allocation["numa_node"],
            "n": n,
            "relation": relation,
            "knobs": spec.get("env", {}),
            "warmup": 0,
            "reps": 0,
            "protocol": None,
            "process_count": 0,
            "processes": [],
            "times_ms": [],
            "verify_uncached_ms": [],
            "verify_cached_ms": [],
            "peak_rss_kib": None,
            "aggregate_bytes": None,
            "aggregate_sha256": None,
            "accepted": True,
            "kernels": {},
            "simd_report": self.reports.get(config),
            "freq_mhz": {},
            "foreign_busy": None,
            "loadavg": None,
            "commit": self.code_commit,
            "machine": self.machine.get("cpu_model"),
            "parts": [],
        }

    def finish_cell(self, cell):
        cell["reps"] = len(cell["times_ms"])
        cell["process_count"] = len(cell.get("processes", []))
        # Keep the per-invocation documents out of the raw cell; they are large.
        cell["parts"] = len(cell.get("parts", []))
        path = self.result_dir / "raw" / f"{cell['run_id']}.json"
        write_json(path, cell)

    def check_hashes(self, cells, tag, n, relation):
        hashes = {
            cell.get("aggregate_sha256")
            for cell in cells.values()
            if cell.get("aggregate_sha256")
        }
        sizes = {
            cell.get("aggregate_bytes")
            for cell in cells.values()
            if cell.get("aggregate_bytes")
        }
        if len(hashes) > 1 or len(sizes) > 1:
            print(
                f"WARNING: aggregates differ across configurations at {tag} N={n} {relation}: "
                f"{sizes} {hashes}"
            )

    def run_aerie_group(self, group, rows, sizes, reps, warmup):
        backend = group.get("backend", self.args.aerie_backend)
        binary = BIN / "aerie-falcon-bench"
        if backend == "avx2":
            binary = BIN / "aerie-falcon-bench-avx2"
            if not binary.exists():
                raise SystemExit(
                    "the AVX2-capped Aerie binary is missing; rerun ./run.sh build"
                )
        if not binary.exists():
            raise SystemExit(f"{binary} is missing; run ./run.sh build")
        for requested, actual, allocation, note in self.allocation_plan(
            group, ["8C16T"]
        ):
            for n in sizes:
                self.run_aerie_cell(
                    group, binary, backend, requested, actual, allocation, note, n, reps, warmup
                )

    def run_aerie_cell(self, group, binary, backend, requested, actual, allocation, note, n, reps, warmup):
        env = {
            "RAYON_NUM_THREADS": str(allocation["threads"]),
            "AERIE_SIMD_BACKEND": backend,
        }
        if backend == "auto":
            env.pop("AERIE_SIMD_BACKEND")
        else:
            # Aerie's Keccak dispatcher has its own environment variable and
            # does not follow AERIE_SIMD_BACKEND.
            env["AERIE_KECCAK_BACKEND"] = backend
        cmd = pin_prefix(self.topo, allocation) + [str(binary), "--bench", str(n)]
        # One fixture run, then the timed runs.
        for _ in range(max(warmup, 0)):
            proc = sh(cmd, env)
            if proc.returncode != 0:
                raise SystemExit(
                    f"Aerie fixture run failed: {(proc.stderr or '')[-2000:]}"
                )
        series = {"prove": [], "verify_parallel": [], "verify_single": [], "wire": None}
        logs = []
        for _ in range(reps):
            proc = sh(cmd, env)
            logs.append((proc.stdout or "") + (proc.stderr or ""))
            if proc.returncode != 0:
                raise SystemExit(f"Aerie run failed: {(proc.stderr or '')[-2000:]}")
            row = parse_aerie(proc.stdout or "", n)
            if row is None:
                raise SystemExit("Aerie printed no result row:\n" + (proc.stdout or "")[-2000:])
            series["prove"].append(row["prove_ms"])
            series["verify_parallel"].append(row["verify_parallel_ms"])
            series["verify_single"].append(row["verify_single_ms"])
            series["wire"] = row["wire_bytes"]
        cell = {
            "run_id": f"{group['tag']}-aerie-{actual if requested == actual else f'{requested}-to-{actual}'}-n{n}",
            "system": "aerie",
            "tag": group["tag"],
            "config": "aerie",
            "label": "Aerie v0.2.0 Falcon-512",
            "threads": allocation["threads"],
            "allocation": actual,
            "allocation_requested": requested,
            "allocation_note": note,
            "cpuset": allocation["cpuset"],
            "numa_node": allocation["numa_node"],
            "n": n,
            "relation": "falcon",
            "knobs": {"AERIE_SIMD_BACKEND": backend, "AERIE_KECCAK_BACKEND": backend},
            "warmup": warmup,
            "reps": reps,
            "times_ms": series["prove"],
            "verify_uncached_ms": series["verify_parallel"],
            "verify_cached_ms": [],
            "verify_single_ms": series["verify_single"],
            "peak_rss_kib": None,
            "aggregate_bytes": series["wire"],
            "aggregate_sha256": None,
            "accepted": True,
            "kernels": {"aerie_simd_backend": backend},
            "freq_mhz": {},
            "foreign_busy": None,
            "loadavg": {"start": self.machine.get("loadavg"), "end": list(os.getloadavg())},
            "commit": "49b6c212c209286c536196f53618823ffdf0e899",
            "machine": self.machine.get("cpu_model"),
        }
        (self.result_dir / "logs" / f"{cell['run_id']}.log").write_text(
            "\n".join(logs), encoding="utf-8"
        )
        self.finish_cell(cell)
        self.cells.append(cell)

    def run_native_group(self, group, rows, sizes, reps, warmup):
        binary = BIN / "natver"
        if not binary.exists():
            raise SystemExit(f"{binary} is missing; run ./run.sh build")
        for requested, actual, allocation, note in self.allocation_plan(
            group, ["8C16T"]
        ):
            for n in sizes:
                out = self.result_dir / "raw" / "parts" / f"native-{actual}-{n}-{time.time_ns()}.json"
                out.parent.mkdir(exist_ok=True)
                env = {
                    "OMP_NUM_THREADS": str(allocation["threads"]),
                    "OMP_PROC_BIND": "close",
                    "OMP_PLACES": "threads",
                }
                cmd = pin_prefix(self.topo, allocation) + [
                    str(binary),
                    str(n),
                    "--warmup",
                    str(warmup),
                    "--reps",
                    str(reps),
                    "--seed",
                    "1",
                    "--json",
                    str(out),
                ]
                proc = sh(cmd, env)
                if proc.returncode != 0 or not out.exists():
                    raise SystemExit(
                        f"natver failed at N={n}: {(proc.stderr or '')[-2000:]}"
                    )
                data = read_json(out)
                cell = {
                    "run_id": f"{group['tag']}-native-{actual if requested == actual else f'{requested}-to-{actual}'}-n{n}",
                    "system": "native",
                    "tag": group["tag"],
                    "config": "native",
                    "label": "pq-crystals ML-DSA-44 AVX2",
                    "threads": allocation["threads"],
                    "allocation": actual,
                    "allocation_requested": requested,
                    "allocation_note": note,
                    "cpuset": allocation["cpuset"],
                    "numa_node": allocation["numa_node"],
                    "n": n,
                    "relation": "none",
                    "knobs": {},
                    "warmup": warmup,
                    "reps": reps,
                    "times_ms": [],
                    "verify_uncached_ms": data["uncached_ms"],
                    "verify_cached_ms": data["cached_ms"],
                    "verify_single_ms": [],
                    "peak_rss_kib": None,
                    "aggregate_bytes": None,
                    "aggregate_sha256": None,
                    "accepted": data["rejected_uncached"] == 0
                    and data["rejected_cached"] == 0,
                    "rejected_uncached": data["rejected_uncached"],
                    "rejected_cached": data["rejected_cached"],
                    "dataset_sha256": data.get("dataset_sha256"),
                    "kernels": {"c_isa": "avx2,bmi2,popcnt,aes"},
                    "freq_mhz": {},
                    "foreign_busy": None,
                    "loadavg": {"start": self.machine.get("loadavg"), "end": list(os.getloadavg())},
                    "commit": "d35ba3fe5449bee3e6d43e1f296c3ca818bd36be",
                    "machine": self.machine.get("cpu_model"),
                }
                self.finish_cell(cell)
                self.cells.append(cell)

    def ensure_records(self, n):
        records = self.result_dir / "data" / f"records-{n}.bin"
        keys = self.result_dir / "data" / f"keys-{n}.bin"
        binary = BIN / "agg"
        if not records.exists():
            env = self.env_for("ifma", self.topo["configs"]["all"], None)
            proc = sh(
                [str(binary), "keygen-batch", "--count", str(n), "--seed", "1", "--out", str(keys)],
                env,
            )
            if proc.returncode != 0:
                raise SystemExit("keygen-batch failed")
            proc = sh(
                [
                    str(binary),
                    "sign-batch",
                    "--keys",
                    str(keys),
                    "--count",
                    str(n),
                    "--seed",
                    "1",
                    "--out",
                    str(records),
                ],
                env,
            )
            if proc.returncode != 0:
                raise SystemExit("sign-batch failed")
        return records

    def run_fresh_group(self, group, rows, sizes, reps):
        for requested, actual, allocation, note in self.allocation_plan(
            group, ["8C16T"]
        ):
            for n in sizes:
                records = self.ensure_records(n)
                for name in rows:
                    if group.get("mode") == "aggregate":
                        self.fresh_aggregate(group, name, requested, actual, allocation, note, n, records, reps)
                    else:
                        self.fresh_verify(group, name, requested, actual, allocation, note, n, records, reps)

    def time_v(self, cmd, env, log):
        time_file = log.with_suffix(".time")
        proc = sh(["/usr/bin/time", "-v", "-o", str(time_file)] + cmd, env)
        text = time_file.read_text(encoding="utf-8") if time_file.exists() else ""
        rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
        wall = re.search(r"Elapsed \(wall clock\) time.*?: (\S+)", text)
        seconds = parse_elapsed(wall.group(1)) if wall else None
        return proc, int(rss.group(1)) if rss else None, seconds

    def fresh_aggregate(self, group, name, requested, actual, allocation, note, n, records, reps, record=True):
        binary = BIN / self.configs["rows"][name]["binary"]
        out = self.result_dir / "data" / f"fresh-{name}-{actual}-{n}.bin"
        env = self.env_for(name, allocation, None)
        peaks, walls, hashes = [], [], set()
        for i in range(reps):
            log = self.result_dir / "logs" / f"{group['tag']}-{name}-{actual}-{n}-{i}.log"
            proc, rss, seconds = self.time_v(
                [str(binary), "aggregate", "--records", str(records), "--relation", "h-u", "--out", str(out)],
                env,
                log,
            )
            if proc.returncode != 0:
                raise SystemExit(f"fresh aggregate failed: {(proc.stderr or '')[-2000:]}")
            peaks.append(rss)
            walls.append(seconds * 1000 if seconds else None)
            hashes.add(sha256(out))
        cell = {
            "run_id": f"{group['tag']}-{name}-{actual if requested == actual else f'{requested}-to-{actual}'}-n{n}",
            "system": "ours",
            "tag": group["tag"],
            "config": name,
            "label": self.configs["rows"][name].get("label", name),
            "threads": allocation["threads"],
            "allocation": actual,
            "allocation_requested": requested,
            "allocation_note": note,
            "cpuset": allocation["cpuset"],
            "numa_node": allocation["numa_node"],
            "n": n,
            "relation": "h-u",
            "knobs": self.configs["rows"][name].get("env", {}),
            "warmup": 0,
            "reps": reps,
            "times_ms": [w for w in walls if w],
            "verify_uncached_ms": [],
            "verify_cached_ms": [],
            "peak_rss_kib": max([p for p in peaks if p] or [0]) or None,
            "aggregate_bytes": out.stat().st_size,
            "aggregate_sha256": next(iter(hashes)) if len(hashes) == 1 else None,
            "accepted": True,
            "kernels": {},
            "freq_mhz": {},
            "foreign_busy": None,
            "loadavg": {"start": self.machine.get("loadavg"), "end": list(os.getloadavg())},
            "commit": self.code_commit,
            "machine": self.machine.get("cpu_model"),
            "fresh_process": True,
        }
        if record:
            self.finish_cell(cell)
            self.cells.append(cell)

    def fresh_verify(self, group, name, requested, actual, allocation, note, n, records, reps):
        binary = BIN / self.configs["rows"][name]["binary"]
        aggregate = self.result_dir / "data" / f"fresh-{name}-{actual}-{n}.bin"
        if not aggregate.exists():
            self.fresh_aggregate(
                group, name, requested, actual, allocation, note, n, records, 1, record=False
            )
        env = self.env_for(name, allocation, None)
        cache = self.result_dir / "data" / f"cache-{name}-{actual}-{n}.bin"
        results = {}
        for mode in ["uncached", "cached"]:
            peaks, walls = [], []
            for i in range(reps):
                log = self.result_dir / "logs" / f"{group['tag']}-{mode}-{name}-{actual}-{n}-{i}.log"
                cmd = [str(binary), "verify", "--records", str(records), "--aggregate", str(aggregate)]
                if mode == "cached":
                    cmd += ["--cache", str(cache)]
                proc, rss, seconds = self.time_v(cmd, env, log)
                if proc.returncode != 0:
                    raise SystemExit(f"fresh verify {mode} failed: {(proc.stderr or '')[-2000:]}")
                peaks.append(rss)
                walls.append(seconds * 1000 if seconds else None)
            results[mode] = {
                "peak": max([p for p in peaks if p] or [0]) or None,
                "times": [w for w in walls if w],
            }
        cell = {
            "run_id": f"{group['tag']}-{name}-{actual if requested == actual else f'{requested}-to-{actual}'}-n{n}",
            "system": "ours",
            "tag": group["tag"],
            "config": name,
            "label": self.configs["rows"][name].get("label", name),
            "threads": allocation["threads"],
            "allocation": actual,
            "allocation_requested": requested,
            "allocation_note": note,
            "cpuset": allocation["cpuset"],
            "numa_node": allocation["numa_node"],
            "n": n,
            "relation": "h-u",
            "knobs": self.configs["rows"][name].get("env", {}),
            "warmup": 0,
            "reps": reps,
            "times_ms": [],
            "verify_uncached_ms": results["uncached"]["times"],
            "verify_cached_ms": results["cached"]["times"],
            "peak_rss_kib": results["uncached"]["peak"],
            "peak_rss_cached_kib": results["cached"]["peak"],
            "aggregate_bytes": aggregate.stat().st_size,
            "aggregate_sha256": sha256(aggregate),
            "accepted": True,
            "kernels": {},
            "freq_mhz": {},
            "foreign_busy": None,
            "loadavg": {"start": self.machine.get("loadavg"), "end": list(os.getloadavg())},
            "commit": self.code_commit,
            "machine": self.machine.get("cpu_model"),
            "fresh_process": True,
        }
        self.finish_cell(cell)
        self.cells.append(cell)

    def run_ablations(self, group, rows, sizes, relation, reps, warmup):
        arms = group["arms"]
        chunks = process_chunks(reps, self.process_count(group, reps))
        for requested, actual, allocation, note in self.allocation_plan(
            group, ["8C16T"]
        ):
            for n in sizes:
                pairs = [(name, arm) for name in rows for arm in arms]
                cells = {}
                for name, arm in pairs:
                    key = (name, arm["tag"])
                    cell = self.base_cell(
                        "ours", name, f"{group['tag']}-{arm['tag']}", requested, actual, allocation, note, n, relation
                    )
                    cell["arm"] = arm["tag"]
                    cell["warmup"] = warmup
                    cell["reps"] = reps
                    cell["protocol"] = "warm"
                    cell["process_count"] = len(chunks)
                    cells[key] = cell
                before = read_stat()
                sampler, freq_path = start_sampler(self.result_dir, allocation)
                started = utc_stamp()
                try:
                    for r in range(len(chunks)):
                        for name, arm in rotate(pairs, r):
                            part = self.invoke_ours(
                                name,
                                allocation,
                                n,
                                relation,
                                arm.get("env"),
                                warmup,
                                chunks[r],
                            )
                            cell = cells[(name, arm["tag"])]
                            cell["times_ms"].extend(part["prove_ms"])
                            cell["verify_uncached_ms"].extend(part["verify_uncached_ms"])
                            cell["verify_cached_ms"].extend(part["verify_cached_ms"])
                            cell["parts"].append(part)
                            cell["processes"].append(
                                {
                                    "json": part["_json"],
                                    "warmup": warmup,
                                    "reps": chunks[r],
                                    "prove_ms": part["prove_ms"],
                                }
                            )
                            cell["aggregate_bytes"] = part["aggregate_bytes"]
                            cell["aggregate_sha256"] = part["aggregate_sha256"]
                            cell["kernels"] = part["kernels"]
                            peak = part.get("prove_peak_kib")
                            if peak:
                                cell["peak_rss_kib"] = max(
                                    cell.get("peak_rss_kib") or 0, peak
                                )
                finally:
                    stop_sampler(sampler)
                after = read_stat()
                for cell in cells.values():
                    cell["start_utc"] = started
                    cell["end_utc"] = utc_stamp()
                    cell["freq_mhz"] = frequency(freq_path, set(allocation["cpus"]))
                    cell["foreign_busy"] = foreign_fraction(
                        before, after, set(allocation["cpus"])
                    )
                    cell["loadavg"] = self.load_before_after(before)
                    self.finish_cell(cell)
                    self.cells.append(cell)

    def summary(self):
        rows = []
        for cell in self.cells:
            prove = describe(cell.get("times_ms") or [])
            uncached = describe(cell.get("verify_uncached_ms") or [])
            cached = describe(cell.get("verify_cached_ms") or [])
            rows.append(
                {
                    "run_id": cell["run_id"],
                    "system": cell["system"],
                    "config": cell["config"],
                    "tag": cell["tag"],
                    "allocation": cell["allocation"],
                    "threads": cell["threads"],
                    "n": cell["n"],
                    "relation": cell["relation"],
                    "protocol": cell.get("protocol") or "",
                    "prove_median_ms": prove["median"] if prove else "",
                    "prove_min_ms": prove["min"] if prove else "",
                    "prove_max_ms": prove["max"] if prove else "",
                    "prove_interval_lo": prove["interval"][0] if prove else "",
                    "prove_interval_hi": prove["interval"][1] if prove else "",
                    "prove_runs": prove["n"] if prove else 0,
                    "verify_uncached_median_ms": uncached["median"] if uncached else "",
                    "verify_cached_median_ms": cached["median"] if cached else "",
                    "peak_rss_kib": cell.get("peak_rss_kib") or "",
                    "aggregate_bytes": cell.get("aggregate_bytes") or "",
                    "aggregate_sha256": cell.get("aggregate_sha256") or "",
                    "flags": ";".join(cell_flags(cell)),
                }
            )
        path = self.result_dir / "summary.csv"
        if rows:
            import csv

            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        if self.skipped:
            write_json(self.result_dir / "skipped.json", self.skipped)

    def memory_filter(self, tag, sizes):
        """Drop batch sizes whose recorded prover peak does not fit with headroom."""
        available = (self.machine.get("memory") or {}).get("available_bytes")
        if not available:
            return sizes
        need = {1024: 2, 2048: 3, 4096: 5, 8192: 8, 16384: 14, 32768: 28}
        kept = []
        for n in sizes:
            required = need.get(n)
            if required is None or available >= required * (1 << 30):
                kept.append(n)
            else:
                self.skipped.append(
                    {
                        "kind": "batch-size",
                        "name": str(n),
                        "tag": tag,
                        "reason": f"MemAvailable {available / (1 << 30):.1f} GiB, "
                        f"{required} GiB needed",
                    }
                )
                print(
                    f"skip N={n} in {tag}: MemAvailable "
                    f"{available / (1 << 30):.1f} GiB < {required} GiB"
                )
        return kept

    def run(self):
        sizes = self.args.sizes
        requested_reps = self.args.reps
        for group in self.matrix.get("runs", []):
            tag = group.get("tag", group["kind"])
            if self.args.only and self.args.only not in tag:
                continue
            kind = group["kind"]
            rows = self.rows(group.get("configs")) if kind != "native" else []
            group_sizes = sizes or group.get("sizes") or [1024]
            if self.args.smoke:
                group_sizes = [s for s in group_sizes if s <= 1024] or [1024]
            if kind in ("ours", "ablations", "fresh"):
                group_sizes = self.memory_filter(tag, group_sizes)
                if not group_sizes:
                    print(f"skip {tag}: no batch size fits in memory")
                    continue
            reps = requested_reps or group.get("reps") or group.get("rounds") or 3
            warmup = (
                self.args.warmup
                if self.args.warmup is not None
                else group.get("warmup", 1)
            )
            processes = self.process_count(group, reps)
            if self.args.smoke:
                reps = 1
                warmup = 1
                processes = 1
            print(
                f"== {tag} ({kind}) sizes={group_sizes} reps={reps} "
                f"warmup={warmup} processes={processes}"
            )
            if kind == "ours":
                relations = group.get("relations") or [group.get("relation", "h-u")]
                self.run_ours_group(group, rows, group_sizes, relations, reps, warmup)
            elif kind == "ablations":
                self.run_ablations(
                    group,
                    rows,
                    group_sizes,
                    group.get("relation", "h-u"),
                    reps,
                    warmup,
                )
            elif kind == "aerie":
                self.run_aerie_group(group, rows, group_sizes, reps, warmup)
            elif kind == "native":
                self.run_native_group(group, rows, group_sizes, reps, warmup)
            elif kind == "fresh":
                self.run_fresh_group(group, rows, group_sizes, reps)
            else:
                raise SystemExit(f"unknown run kind {kind}")
        self.summary()
        print(f"results: {self.result_dir}")


def parse_aerie(stdout, n):
    for line in stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 33:
            continue
        try:
            if int(parts[0]) != n:
                continue
            return {
                "prove_ms": float(parts[2]),
                "verify_parallel_ms": float(parts[18]),
                "verify_single_ms": float(parts[24]),
                "wire_bytes": int(parts[32]),
            }
        except ValueError:
            continue
    return None


def parse_elapsed(text):
    parts = text.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except ValueError:
        return None


def sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["quick", "full"])
    parser.add_argument("--smoke", action="store_true", help="N=1024 and 1+1 runs")
    parser.add_argument("--results")
    parser.add_argument("--only", help="run only groups whose tag contains this")
    parser.add_argument("--configs")
    parser.add_argument("--allocations")
    parser.add_argument("--sizes")
    parser.add_argument("--reps", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument(
        "--processes",
        type=int,
        help="processes per warm cell (default: matrix value or 2)",
    )
    parser.add_argument("--aerie-backend", default="auto")
    args = parser.parse_args()
    if args.configs:
        args.configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    if args.sizes:
        args.sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    if args.allocations:
        args.allocations = [a.strip() for a in args.allocations.split(",") if a.strip()]
    runner = Runner(args)
    runner.run()


if __name__ == "__main__":
    main()
