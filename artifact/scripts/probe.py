#!/usr/bin/env python3
"""Record machine state; never change it.

`probe.py` writes machine.json and the warnings that the table notes repeat.
`probe.py --check` reports missing tools, sources and binaries.
"""

import argparse
import glob
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    AERIE_TOOLCHAIN,
    ARTIFACT,
    BIN,
    PARAM_SETS,
    RUST_TOOLCHAIN,
    VENDOR,
    cpu_flags,
    cpu_model,
    hostname,
    memory_floor_gib,
    output,
    supported_tiers,
    utc_stamp,
    write_json,
)


def read(path, default=None):
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return default


def first_line(text, default="unknown"):
    return text.splitlines()[0].strip() if text else default


def cpuinfo_value(field):
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith(field):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def memory():
    values = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                key, rest = line.split(":", 1)
                values[key] = int(rest.split()[0]) * 1024
    except OSError:
        pass
    return {
        "total_bytes": values.get("MemTotal"),
        "available_bytes": values.get("MemAvailable"),
        "swap_total_bytes": values.get("SwapTotal"),
        "swap_free_bytes": values.get("SwapFree"),
    }


def governors():
    values = {}
    for path in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor"):
        cpu = path.split("/")[-3]
        values[cpu] = read(path)
    return values


def boost():
    candidates = [
        ("/sys/devices/system/cpu/cpufreq/boost", "cpufreq/boost"),
        ("/sys/devices/system/cpu/intel_pstate/no_turbo", "intel_pstate/no_turbo"),
        ("/sys/devices/system/cpu/amd_pstate/status", "amd_pstate/status"),
    ]
    for path, label in candidates:
        value = read(path)
        if value is not None:
            return {"source": label, "value": value}
    return {"source": None, "value": None}


def energy_preference():
    values = {}
    for path in glob.glob(
        "/sys/devices/system/cpu/cpu[0-9]*/cpufreq/energy_performance_preference"
    ):
        cpu = path.split("/")[-3]
        values[cpu] = read(path)
    return values


def thp():
    base = "/sys/kernel/mm/transparent_hugepage"
    return {
        "enabled": read(f"{base}/enabled"),
        "defrag": read(f"{base}/defrag"),
        "shmem_enabled": read(f"{base}/shmem_enabled"),
    }


def top_processes():
    proc = subprocess.run(
        ["ps", "-eo", "pid,comm,pcpu,pmem", "--sort=-pcpu"],
        text=True,
        capture_output=True,
    )
    rows = []
    for line in (proc.stdout or "").splitlines()[1:11]:
        parts = line.split(None, 3)
        if len(parts) == 4:
            rows.append(
                {
                    "pid": int(parts[0]),
                    "comm": parts[1],
                    "cpu_percent": float(parts[2]),
                    "mem_percent": float(parts[3]),
                }
            )
    return rows


def container_state():
    markers = []
    if Path("/.dockerenv").exists():
        markers.append("/.dockerenv")
    if Path("/run/.containerenv").exists():
        markers.append("/run/.containerenv")
    cgroup = read("/proc/1/cgroup") or ""
    if "docker" in cgroup or "kubepods" in cgroup:
        markers.append("cgroup")
    return {"container": bool(markers), "markers": markers}


def tool_versions():
    return {
        "rustc": first_line(output(["rustc", "-V"])),
        "cargo": first_line(output(["cargo", "-V"])),
        "rustc_pinned": first_line(output(["rustc", "-V"], env={"RUSTUP_TOOLCHAIN": RUST_TOOLCHAIN})),
        "aerie_rustc": first_line(output(["rustc", "-V"], env={"RUSTUP_TOOLCHAIN": AERIE_TOOLCHAIN})),
        "gcc": first_line(output(["gcc", "--version"])),
        "python": sys.version.split()[0],
        "make": first_line(output(["make", "--version"])),
        "objdump": first_line(output(["objdump", "--version"])),
    }


def build_warnings(state):
    warnings = []
    values = set(state["governor"].values())
    if values and values != {"performance"}:
        warnings.append(
            "frequency governor "
            + ", ".join(sorted(v or "unknown" for v in values))
            + " (not performance)"
        )
    if not state["governor"]:
        warnings.append("no cpufreq governor is exposed (virtual machine or WSL)")
    if state["thp"]["enabled"] and "always" not in state["thp"]["enabled"]:
        warnings.append("transparent huge pages: " + state["thp"]["enabled"])
    if state["container"]["container"]:
        warnings.append("running inside a container: " + ", ".join(state["container"]["markers"]))
    if "hypervisor" in state["flags"]:
        warnings.append("virtual machine (hypervisor CPU flag)")
    available = state["memory"]["available_bytes"]
    if available is not None:
        short = [
            f"ML-DSA-{params} ({memory_floor_gib(params, 32768)} GiB)"
            for params in PARAM_SETS
            if available < memory_floor_gib(params, 32768) * (1 << 30)
        ]
        if short:
            warnings.append(
                f"MemAvailable {available / (1 << 30):.1f} GiB: the 32K cells of "
                + ", ".join(short)
                + " will be skipped"
            )
    load = state["loadavg"]
    nproc = state["nproc"]
    if load[0] > 0.5 * nproc:
        warnings.append(f"load average {load[0]:.2f} on {nproc} CPUs")
    return warnings


def probe():
    lscpu = output(["lscpu"], default="")
    numa = output(["numactl", "-H"], default="numactl not installed")
    governor = governors()
    state = {
        "timestamp_utc": utc_stamp(),
        "hostname": hostname(),
        "kernel": os.uname().release,
        "uname": " ".join(os.uname()),
        "os_release": read("/etc/os-release", ""),
        "architecture": platform.machine(),
        "nproc": os.cpu_count() or 0,
        "cpu_model": cpu_model(),
        "cpu_vendor": cpuinfo_value("vendor_id"),
        "cpu_family": cpuinfo_value("cpu family"),
        "cpu_flags": sorted(cpu_flags()),
        "hypervisor": "hypervisor" in cpu_flags(),
        "lscpu": lscpu,
        "numactl_H": numa,
        "memory": memory(),
        "loadavg": list(os.getloadavg()),
        "uptime_seconds": float(read("/proc/uptime", "0 0").split()[0]),
        "governor": governor,
        "boost": boost(),
        "energy_performance_preference": energy_preference(),
        "thp": thp(),
        "vm_overcommit_memory": read("/proc/sys/vm/overcommit_memory"),
        "processes": top_processes(),
        "container": container_state(),
        "toolchains": tool_versions(),
        "flags": sorted(cpu_flags()),
        "tiers": supported_tiers(),
    }
    state["warnings"] = build_warnings(state)
    return state


def check():
    problems = []
    notes = []
    flags = cpu_flags()
    if platform.machine() != "x86_64":
        problems.append(f"architecture {platform.machine()}, x86-64 required")
    required = ["bash", "python3", "git", "gcc", "make", "rustc", "cargo", "objdump"]
    for tool in required:
        if shutil.which(tool) is None:
            problems.append(f"missing tool: {tool}")
    if "pclmulqdq" not in flags:
        problems.append("CPU lacks pclmulqdq: the prover requires CLMUL")
    if "avx2" not in flags:
        problems.append(
            "CPU lacks avx2: the release binary is built for x86-64-v3 and does not start"
        )
    if sys.version_info < (3, 11):
        notes.append("Python older than 3.11: using the bundled TOML fallback")
    for tool in ["taskset", "numactl", "/usr/bin/time", "shellcheck", "docker"]:
        if shutil.which(tool) is None:
            notes.append(f"optional tool missing: {tool}")
    for name, path in [
        ("workspace Cargo.toml", ARTIFACT.parent / "Cargo.toml"),
        ("natver source", ARTIFACT / "natver" / "natver.c"),
    ]:
        if not path.exists():
            problems.append(f"missing source: {name} at {path}")
    for name, path in [
        ("aerie", VENDOR / "aerie"),
        ("dilithium", VENDOR / "dilithium"),
    ]:
        if not path.exists():
            notes.append(f"source not fetched yet: {name} at {path} (run ./run.sh fetch)")
    for name in ["agg", "mode_b", "aerie-falcon-bench", "natver", "natver65", "natver87"]:
        if not (BIN / name).exists():
            notes.append(f"binary not built yet: {BIN / name}")

    print("artifact check")
    print(f"  host           {hostname()} ({cpu_model()})")
    print(f"  kernel         {os.uname().release}")
    print(f"  CPUs           {os.cpu_count()}")
    print(
        "  features       "
        + " ".join(
            f"{f}={'yes' if f in flags else 'no'}"
            for f in ["pclmulqdq", "avx2", "avx512f", "avx512bw", "avx512vl", "avx512ifma", "gfni"]
        )
    )
    tiers = supported_tiers(flags)
    print("  tiers          " + (" ".join(tiers) if tiers else "none"))
    print(f"  Python         {sys.version.split()[0]}")
    for note in notes:
        print(f"  note           {note}")
    if problems:
        for problem in problems:
            print(f"  MISSING        {problem}")
        print("check: FAIL")
        return 1
    print("check: OK")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report missing requirements")
    parser.add_argument("--out", help="write machine.json here")
    args = parser.parse_args()
    if args.check:
        return check()
    state = probe()
    if args.out:
        write_json(args.out, state)
    else:
        print(state)
    for warning in state["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
