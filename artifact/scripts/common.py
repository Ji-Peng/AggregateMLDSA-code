"""Paths, configuration loading and process helpers shared by the scripts."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import tomllib as _toml
except ModuleNotFoundError:  # Python 3.10 and older
    import toml_compat as _toml

ARTIFACT = Path(__file__).resolve().parent.parent
REPO = ARTIFACT.parent
CONFIG = ARTIFACT / "config"
EXPECTED = ARTIFACT / "expected"
BUILD = ARTIFACT / "build"
BIN = BUILD / "bin"
RESULTS = ARTIFACT / "results"
VENDOR = ARTIFACT / "vendor"
TABLES = ARTIFACT / "paper-tables"

RUST_TOOLCHAIN = "1.99.1"
AERIE_TOOLCHAIN = "1.95.0"

# Relations the scripts accept. H-FIPS is the relation of every default run;
# H-exact is an opt-in reference (`--relations h-fips,h-exact`).
RELATIONS = ("h-fips", "h-exact")
DEFAULT_RELATIONS = ("h-fips",)
DEFAULT_RELATION = "h-fips"


def parse_relations(text):
    """A comma-separated relation list (`h-fips,h-exact`); raises ValueError on an unknown one."""
    items = [item.strip().lower() for item in str(text).split(",") if item.strip()]
    unknown = [item for item in items if item not in RELATIONS]
    if unknown or not items:
        raise ValueError(
            f"relations: a comma-separated subset of {','.join(RELATIONS)}, not {text!r}"
        )
    return list(dict.fromkeys(items))

def load_toml(path):
    with open(path, "rb") as f:
        data = f.read()
    # tomllib (Python 3.11+) and the bundled fallback both accept a string.
    return _toml.loads(data.decode("utf-8"))


def load_configs():
    return load_toml(CONFIG / "configs.toml")


def load_matrix(mode):
    return load_toml(CONFIG / f"matrix-{mode}.toml")


def pinned_commit():
    """The in-repo source commit named by `sources.lock`."""
    for source in load_toml(ARTIFACT / "sources.lock").get("source", []):
        if source.get("in_repo"):
            return source["commit"]
    return ""


def source_revision():
    """The revision the binaries are built from.

    In a git checkout this is `HEAD`, with `-dirty` when the tree has changes,
    because the pin in `sources.lock` lags behind development commits. Only an
    archive without `.git` falls back to the pin.
    """
    if (REPO / ".git").exists():
        head = git_rev(REPO)
        if head:
            return head + ("-dirty" if git_dirty(REPO) else "")
    return pinned_commit()


def built_revision():
    """The revision `build.py` stamped into the binaries, or the current one if unknown."""
    path = BUILD / "build.json"
    if path.exists():
        try:
            commit = json.loads(path.read_text(encoding="utf-8")).get("commit")
        except (OSError, ValueError):
            commit = None
        if commit:
            return commit
    return source_revision()


# Per-set constants of FIPS 204 (Tables 1 and 2): NIST security category,
# signature bytes, bytes of the commitment hash c~ (lambda/4), and the natural
# verifier binary that artifact/natver/Makefile builds for the set.
PARAM_SETS = {
    "44": {"category": 2, "sig_bytes": 2420, "ctilde_bytes": 32, "natver": "natver"},
    "65": {"category": 3, "sig_bytes": 3309, "ctilde_bytes": 48, "natver": "natver65"},
    "87": {"category": 5, "sig_bytes": 4627, "ctilde_bytes": 64, "natver": "natver87"},
}


# Prover peak resident set (VmHWM, MiB) of the paper's measurements on an
# EPYC 9355 (Zen 5): per set and batch size, the maximum over the three tiers,
# the allocations and the relations measured there. It includes the
# allocator's block cache.
PROVER_PEAK_MIB = {
    "44": {1024: 1978, 8192: 4635, 32768: 16678},
    "65": {1024: 2075, 8192: 4559, 32768: 15486},
    "87": {1024: 2254, 2048: 1872, 4096: 3285, 8192: 5837, 16384: 10802, 32768: 20483},
}
# Free memory kept beyond the recorded peak, for run-to-run variation of the
# peak, the other ablation arms and the harness itself.
MEMORY_HEADROOM_GIB = 4


def memory_floor_gib(params, n):
    """`MemAvailable` (whole GiB) a prover cell of set `params` at batch size `n` needs.

    The floor is the recorded peak plus the headroom, rounded up. A size without
    a record for the set uses the largest peak recorded at that size for any set
    (ML-DSA-87 has the largest peak at every size it shares with the others),
    and a size without any record uses the next larger recorded size. Above
    the largest recorded size there is no floor (None).
    """
    peak = PROVER_PEAK_MIB.get(str(params), {}).get(n)
    if peak is None:
        sizes = sorted({m for peaks in PROVER_PEAK_MIB.values() for m in peaks if m >= n})
        if not sizes:
            return None
        peak = max(peaks.get(sizes[0], 0) for peaks in PROVER_PEAK_MIB.values())
    return -(-(peak + MEMORY_HEADROOM_GIB * 1024) // 1024)


def cell_params(cell):
    """The ML-DSA parameter set of a result cell (cells written before the key are ML-DSA-44)."""
    return str(cell.get("params") or "44")


def run(cmd, *, env=None, cwd=None, check=True, capture=True, quiet=False):
    """Run a command with an optional environment overlay."""
    full_env = os.environ.copy()
    if env:
        full_env.update({k: str(v) for k, v in env.items()})
    if not quiet:
        print("+", " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
        env=full_env,
        text=True,
        capture_output=capture,
    )
    if check and proc.returncode != 0:
        if proc.stdout:
            sys.stderr.write(proc.stdout)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc


def output(cmd, *, cwd=None, env=None, default=""):
    proc = run(cmd, cwd=cwd, env=env, check=False, capture=True, quiet=True)
    return (proc.stdout or default).strip()


def have(tool):
    return shutil_which(tool) is not None


def shutil_which(tool):
    from shutil import which
    return which(tool)


def cpu_flags():
    flags = set()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("flags"):
                    flags.update(line.split(":", 1)[1].split())
                    break
    except OSError:
        pass
    return flags


def supported_tiers(flags=None):
    """The reported tiers this CPU can run, from `ifma` down to `avx2`."""
    flags = set(cpu_flags() if flags is None else flags)
    tiers = []
    if {"avx512f", "avx512dq", "avx512bw", "avx512vl", "avx512ifma"} <= flags:
        tiers.append("ifma")
    if "avx512f" in flags:
        tiers.append("avx512")
    if {"avx2", "pclmulqdq"} <= flags:
        tiers.append("avx2")
    return tiers


def cpu_model():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def git_rev(cwd, rev="HEAD"):
    return output(["git", "-C", str(cwd), "rev-parse", rev])


def git_dirty(cwd):
    return bool(output(["git", "-C", str(cwd), "status", "--porcelain"]))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_stamp():
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def hostname():
    return os.uname().nodename


def result_dir(path=None):
    """The selected results directory: --results, $AGG_RESULTS, or a new one."""
    if path:
        directory = Path(path)
    elif os.environ.get("AGG_RESULTS"):
        directory = Path(os.environ["AGG_RESULTS"])
    else:
        latest = RESULTS / "LATEST"
        if latest.is_file():
            directory = RESULTS / latest.read_text(encoding="utf-8").strip()
        else:
            directory = RESULTS / utc_stamp()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def new_result_dir():
    # Named by the UTC time only: the directory name is copied into the
    # generated tables, which must not carry the host name.
    directory = RESULTS / utc_stamp()
    directory.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "LATEST").write_text(directory.name + "\n", encoding="utf-8")
    (directory / "raw").mkdir(exist_ok=True)
    (directory / "logs").mkdir(exist_ok=True)
    return directory


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def median(values):
    values = sorted(values)
    if not values:
        return float("nan")
    n = len(values)
    return values[n // 2] if n % 2 else 0.5 * (values[n // 2 - 1] + values[n // 2])


def order_statistic_interval(values, target=0.95):
    """The widest central order-statistic interval with coverage >= target.

    Returns (low, high, coverage, k); with n = 10 this is [x(2), x(9)]
    (97.9%) and with n = 5 [x(1), x(5)] (93.75%).
    """
    from math import comb

    n = len(values)
    ordered = sorted(values)
    if n == 0:
        return (float("nan"), float("nan"), 0.0, 0)
    best = (1, 1.0 - 2 * 0.5 ** n)
    for k in range(1, n // 2 + 1):
        coverage = 1.0 - 2.0 * sum(comb(n, i) for i in range(k)) / (2.0 ** n)
        if coverage >= target:
            best = (k, coverage)
    k, coverage = best
    return (ordered[k - 1], ordered[n - k], coverage, k)
