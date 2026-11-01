"""Capture and check the instruction set the dispatchers actually selected.

`agg` prints one line per crate on stderr when `AGG_SIMD_REPORT=1` is set.
The lines come from the same detectors the prover uses, so they state which
instruction set every backend chose after applying `AGG_ISA`.
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path

CRATES = (
    "agg-mldsa-simd",
    "agg-simd",
    "flock-core",
    "akita-algebra",
    "akita-challenges",
    "akita-cpu-backend",
    "aerie-simd",
)

PAIR = re.compile(r"([A-Za-z][\w-]*)=([^\s,]+)")


def parse(stderr):
    """The `key=value` pairs of every report line, keyed by crate."""
    lines = {}
    for raw in stderr.splitlines():
        line = raw.strip()
        if line.startswith("agg: "):
            line = line[len("agg: "):]
        crate, sep, rest = line.partition(":")
        if not sep or crate.strip() not in CRATES:
            continue
        lines[crate.strip()] = dict(PAIR.findall(rest))
    return lines


def summarise(report):
    """The cap and the AVX-512 and IFMA choices of every dispatcher."""
    agg = report.get("agg-simd", {})
    mldsa = report.get("agg-mldsa-simd", {})
    flock = report.get("flock-core", {})
    akita = report.get("akita-algebra", {})
    aerie = report.get("aerie-simd", {})

    def flag(values, name):
        return values.get(name) == "true"

    cap = agg.get("cap")
    if cap is None:
        for values in report.values():
            if "cap" in values:
                cap = values["cap"]
                break
    avx512 = {
        "agg-simd": agg.get("backend") == "Avx512",
        "mldsa": str(mldsa.get("backend", "")).startswith("Avx512"),
        "aerie-simd": aerie.get("backend") == "avx512",
        "flock-avx512f": flag(flock, "avx512f"),
        "flock-avx512clmul": flag(flock, "avx512clmul"),
        "flock-avx512gfni": flag(flock, "avx512gfni"),
        "flock-avx512vbmi": flag(flock, "avx512vbmi"),
        "flock-avx512bw": flag(flock, "avx512bw"),
        "akita-crt-ntt-avx512": flag(akita, "crt-ntt-avx512"),
    }
    ifma = {
        "agg-simd": agg.get("backend") == "Avx512",
        "akita-crt-ntt-ifma": flag(akita, "crt-ntt-ifma"),
    }
    avx2 = {
        "mldsa": str(mldsa.get("backend", "")).startswith("Avx2"),
        "flock": flag(flock, "avx2"),
        "akita-challenges": flag(report.get("akita-challenges", {}), "avx2"),
        "akita-cpu-backend": flag(report.get("akita-cpu-backend", {}), "avx2"),
    }
    return {
        "cap": cap,
        "mldsa_backend": mldsa.get("backend"),
        "aerie_backend": aerie.get("backend"),
        "avx512": avx512,
        "ifma": ifma,
        "avx2": avx2,
        "lines": report,
    }


def expected_cap(tier):
    return "avx512ifma" if tier == "ifma" else tier


def failures(report, tier):
    """Why the report does not match `tier`; an empty list means it matches."""
    summary = summarise(report)
    problems = []
    if not report:
        return ["no AGG_SIMD_REPORT line was printed"]
    cap = summary["cap"]
    want = expected_cap(tier)
    if cap != want:
        problems.append(f"cap={cap}, expected {want}")
    if tier == "avx512":
        for name, selected in summary["ifma"].items():
            if selected:
                problems.append(f"{name} selected IFMA under AGG_ISA=avx512")
    if tier in ("avx2", "scalar"):
        for name, selected in summary["avx512"].items():
            if selected:
                problems.append(f"{name} selected AVX-512 under AGG_ISA={tier}")
        for name, selected in summary["ifma"].items():
            if selected:
                problems.append(f"{name} selected IFMA under AGG_ISA={tier}")
    return problems


def capture(binary, env, timeout=600):
    """Run a cheap command with the report enabled and return the parsed lines."""
    full_env = os.environ.copy()
    full_env.update({k: str(v) for k, v in env.items()})
    full_env["AGG_SIMD_REPORT"] = "1"
    with tempfile.TemporaryDirectory(prefix="agg-report-") as tmp:
        cmd = [
            str(binary),
            "keygen-batch",
            "--count",
            "1",
            "--seed",
            "1",
            "--out",
            str(Path(tmp) / "keys.bin"),
        ]
        proc = subprocess.run(
            cmd,
            env=full_env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    return parse(proc.stderr or ""), proc
