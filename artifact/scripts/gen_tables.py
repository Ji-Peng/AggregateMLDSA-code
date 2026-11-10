#!/usr/bin/env python3
"""Turn raw measurement JSON into the paper's LaTeX tables and in-text macros."""

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    DEFAULT_RELATION,
    PARAM_SETS,
    TABLES,
    cell_params,
    median,
    read_json,
    result_dir,
    utc_stamp,
)
from runner import parse_size_breakdown  # noqa: E402
from stats import describe  # noqa: E402

# The paper reports these three tiers on one machine. The scalar cap is a
# gates-only check and does not appear in a table.
ORDER = ["ifma", "avx512", "avx2"]
MACRO = {"ifma": "Ifma", "avx512": "AvxVn", "avx2": "AvxTwo"}
SET_MACRO = {"44": "FortyFour", "65": "SixtyFive", "87": "EightySeven"}
LABEL = {"ifma": "IFMA", "avx512": "AVX-512F", "avx2": "AVX2"}
RELATION = {"h-fips": "$\\Relfips$", "h-exact": "$\\Relex$"}
# H-FIPS is the relation of every table; H-exact rows appear in tab:relations when the
# run opted in (`--relations h-fips,h-exact`).
RELATION_ORDER = ["h-fips", "h-exact"]
ALLOC_ORDER = ["16C32T", "8C16T", "1T", "all"]
NA = "n/a"

ABLATION_LABELS = {
    "late-join": "late join (\\texttt{AGG\\_LATE\\_JOIN=0})",
    "no-hugepage": "no huge-page advice (\\texttt{AGG\\_HUGEPAGE=0})",
    "no-alloc-cache": "no allocator cache (\\texttt{AGG\\_ALLOC\\_CACHE=0})",
    "stage-branch-main-pool": "stage branch on main pool (\\texttt{AGG\\_PIOP\\_S\\_THREADS=0})",
    "piop-row": "row layout (\\texttt{AGG\\_PIOP\\_LAYOUT=row})",
    "flock-serial": "Flock after the PIOP (\\texttt{AGG\\_FLOCK\\_THREADS=0})",
}


def param_sets(cells):
    """The ML-DSA parameter sets present in `cells`, in FIPS 204 order."""
    present = {cell_params(c) for c in cells if c.get("system") in ("ours", "native")}
    return [params for params in PARAM_SETS if params in present] or list(PARAM_SETS)


def preferred_allocation(cells):
    for allocation in ALLOC_ORDER:
        if allocation not in ("all", "1T") and any(c.get("allocation") == allocation for c in cells):
            return allocation
    for allocation in ALLOC_ORDER:
        if allocation != "all" and any(c.get("allocation") == allocation for c in cells):
            return allocation
    return cells[0]["allocation"] if cells else "16C32T"


def preferred_tier(cells):
    for config in ORDER:
        if any(c.get("config") == config for c in cells):
            return config
    return "ifma"


def load_cells(directory):
    cells = []
    for path in sorted((directory / "raw").glob("*.json")):
        try:
            cell = read_json(path)
        except (OSError, ValueError):
            continue
        if cell.get("times_ms") or cell.get("verify_uncached_ms") or cell.get("aggregate_bytes"):
            cells.append(cell)
    return cells


def skipped_configs(directory):
    """Tiers the CPU cannot run, with a short reason."""
    path = directory / "skipped.json"
    if not path.exists():
        return {}
    try:
        entries = read_json(path)
    except (OSError, ValueError):
        return {}
    result = {}
    for entry in entries:
        if entry.get("kind") != "configuration":
            continue
        reason = entry.get("reason", "")
        if "avx512ifma" in reason:
            result[entry["name"]] = "n/a (CPU without AVX-512IFMA)"
        elif "avx512f" in reason:
            result[entry["name"]] = "n/a (CPU without AVX-512)"
        elif "avx2" in reason:
            result[entry["name"]] = "n/a (CPU below x86-64-v3)"
        else:
            result[entry["name"]] = "n/a (" + reason + ")"
    return result


def is_standard_ours(cell):
    """True for standard Mode-A benchmark cells (not fresh-process, traced or non-default ablation)."""
    if cell.get("system") != "ours":
        return False
    if cell.get("fresh_process") or cell.get("traced"):
        return False
    arm = cell.get("arm")
    if arm and arm != "defaults":
        return False
    return True


def cell_priority(cell, prefer_tag="headline"):
    tag = str(cell.get("tag", ""))
    is_1t_tag = "1t" in tag
    if prefer_tag == "headline":
        if tag.startswith("headline") and not is_1t_tag:
            score = 5
        elif tag.startswith("headline"):
            score = 4
        elif tag.startswith("quick-ours") and not is_1t_tag:
            score = 3
        elif tag.startswith("scaling"):
            score = 2
        elif tag.startswith("relations"):
            score = 1
        else:
            score = 0
    elif prefer_tag == "scaling":
        if tag.startswith("scaling"):
            score = 5
        elif tag.startswith("headline") and not is_1t_tag:
            score = 4
        elif tag.startswith("quick-ours") and not is_1t_tag:
            score = 3
        elif tag.startswith("relations"):
            score = 1
        else:
            score = 0
    elif prefer_tag == "relations":
        if tag.startswith("relations"):
            score = 5
        elif tag.startswith("headline") and not is_1t_tag:
            score = 4
        elif tag.startswith("quick-ours") and not is_1t_tag:
            score = 3
        elif tag.startswith("scaling"):
            score = 2
        else:
            score = 0
    else:
        score = 1 if tag.startswith(prefer_tag) else 0
    return (
        score,
        1 if cell.get("size_breakdown") else 0,
        1 if cell.get("peak_rss_kib") else 0,
        1 if cell.get("allocation") == "16C32T" else 0,
        1 if cell.get("config") == "ifma" else 0,
        cell.get("reps", 0),
    )


def pick_ours_cell(
    cells, *, params, config, allocation, n, relation=DEFAULT_RELATION, prefer_tag="headline"
):
    matches = [
        c
        for c in cells
        if is_standard_ours(c)
        and cell_params(c) == params
        and c.get("config") == config
        and c.get("n") == n
        and c.get("relation") == relation
        and c.get("allocation") == allocation
    ]
    if not matches:
        return None
    return max(matches, key=lambda c: cell_priority(c, prefer_tag=prefer_tag))


def fmt_n_tex(n):
    return f"{n:,}".replace(",", "{,}")


def fmt_pow2_or_n(n):
    if n > 0 and (n & (n - 1)) == 0:
        return f"$2^{{{n.bit_length() - 1}}}$"
    return fmt_n_tex(n)


def fmt_seconds(ms):
    """Seconds with two decimals, or one decimal from 100 s on."""
    if ms is None:
        return "--"
    seconds = ms / 1000
    return f"{seconds:,.1f}" if seconds >= 100 else f"{seconds:.2f}"


def fmt_ms(ms, allow_decimal=False):
    """Whole milliseconds with a thousands comma; one decimal when `allow_decimal` is set."""
    if ms is None:
        return "--"
    if allow_decimal:
        return f"{ms:,.1f}"
    # Halves round up, as in the paper.
    return f"{math.floor(ms + 0.5):,}"


def bytes_per_sig(cell):
    """Aggregate bytes per signature of `cell`, or None."""
    if not cell or not cell.get("aggregate_bytes") or not cell.get("n"):
        return None
    return cell["aggregate_bytes"] / cell["n"]


def fmt_ratio(sig_bytes, per_sig):
    """Compression factor: signature bytes over aggregate bytes per signature, one decimal."""
    if not per_sig:
        return "--"
    return f"{sig_bytes / per_sig:.1f}$\\times$"


def fmt_bytes_per_sig(cell):
    if not cell or not cell.get("aggregate_bytes") or not cell.get("n"):
        return "--"
    return f"{cell['aggregate_bytes'] / cell['n']:.1f}"


def fmt_gib(kib):
    if not kib:
        return "--"
    return f"{kib / (1024.0 * 1024.0):.1f}"


def header(directory, commit, machine):
    return (
        f"% Generated by artifact/scripts/gen_tables.py\n"
        f"% Results: {Path(directory).name}\n"
        f"% Commit: {commit}\n"
        f"% Machine: {machine}\n"
        f"% Date: {utc_stamp()}\n"
    )


def table_begin(
    label,
    caption,
    note,
    colspec,
    header_rows=1,
    extra_hlines=None,
    env="table",
    colsep="4pt",
    rowsep=None,
):
    hline_block = f"  hline{{1,{header_rows + 1},Z}} = {{1pt}},\n"
    if extra_hlines:
        rows_str = ",".join(str(r) for r in extra_hlines)
        hline_block += f"  hline{{{rows_str}}} = {{0.5pt}},\n"
    rowsep_line = f"  rowsep = {rowsep},\n" if rowsep else ""
    header_set = ",".join(str(r) for r in range(1, header_rows + 1))
    return (
        f"\\begin{{{env}}}[t]\n\\centering\n"
        "\\begin{talltblr}[\n"
        f"  caption = {{{caption}}},\n"
        f"  label = {{{label}}},\n"
        f"  remark{{Note}} = {{{note}}},\n"
        "]{\n"
        f"  width = \\linewidth,\n  colspec = {{{colspec}}},\n"
        f"  colsep = {colsep},\n"
        f"{rowsep_line}"
        f"  row{{{header_set}}} = {{font=\\bfseries}},\n"
        "  rows = {valign=m},\n"
        f"{hline_block}"
        "}\n"
    )


def table_end(env="table"):
    return f"\\end{{talltblr}}\n\\end{{{env}}}\n"


def relations_table(cells, directory, commit, machine, skipped):
    """Generate `tab-relations.tex` (`\\label{tab:relations}`): H-exact against H-FIPS across sets.

    H-exact is an opt-in reference relation; without its cells the file holds a comment.
    """
    standard = [c for c in cells if is_standard_ours(c) and c.get("times_ms")]
    if not any(c.get("relation") == "h-exact" for c in standard):
        return "% no H-exact data (opt in with --relations h-fips,h-exact)\n"
    n = max(c["n"] for c in standard if c.get("relation") == "h-exact")
    at_n = [c for c in standard if c["n"] == n]
    alloc = preferred_allocation([c for c in at_n if c.get("relation") == "h-exact"])
    sets = [
        p
        for p in PARAM_SETS
        if any(cell_params(c) == p and c.get("relation") == "h-exact" for c in at_n)
    ] or list(PARAM_SETS)
    per_set = len(RELATION_ORDER)
    extra_hlines = [2 + per_set * i for i in range(1, len(sets))]
    out = header(directory, commit, machine)
    out += table_begin(
        "tab:relations",
        f"Relations $\\Relfips$ and $\\Relex$ at $N={fmt_n_tex(n)}$ on {alloc}.",
        "Tier columns: prove time in s (change against $\\Relfips$ in parentheses). Verify: IFMA, uncached / cached keys.",
        "Q[l] Q[l] X[r] X[r] X[r] X[1.1,r] X[1.2,r]",
        header_rows=1,
        extra_hlines=extra_hlines,
    )
    out += "Set & Rel. & " + " & ".join(LABEL[c] for c in ORDER) + " & Verify (ms) & Bytes \\\\\n"
    for params in sets:
        base_cells = {
            config: pick_ours_cell(
                at_n,
                params=params,
                config=config,
                allocation=alloc,
                n=n,
                relation=DEFAULT_RELATION,
                prefer_tag="relations",
            )
            for config in ORDER
        }
        for idx, relation in enumerate(RELATION_ORDER):
            set_col = f"\\SetCell[r={per_set}]{{l}} {params}" if idx == 0 else ""
            row = [set_col, RELATION[relation]]
            verify_str = "--"
            bytes_str = "--"
            for config in ORDER:
                cell = pick_ours_cell(
                    at_n,
                    params=params,
                    config=config,
                    allocation=alloc,
                    n=n,
                    relation=relation,
                    prefer_tag="relations",
                )
                if cell and cell.get("times_ms"):
                    med = median(cell["times_ms"])
                    val = fmt_seconds(med)
                    base = base_cells.get(config)
                    if relation != DEFAULT_RELATION and base and base.get("times_ms"):
                        base_med = median(base["times_ms"])
                        if base_med > 0:
                            pct = (med - base_med) / base_med * 100.0
                            val += f" ({pct:+.1f}\\%)"
                    row.append(val)
                    if verify_str == "--" and cell.get("verify_uncached_ms"):
                        unc = fmt_ms(median(cell["verify_uncached_ms"]))
                        cac = (
                            fmt_ms(median(cell["verify_cached_ms"]))
                            if cell.get("verify_cached_ms")
                            else "--"
                        )
                        verify_str = f"{unc} / {cac}"
                    if bytes_str == "--" and cell.get("aggregate_bytes"):
                        b = cell["aggregate_bytes"]
                        bytes_str = fmt_n_tex(b)
                        base_b = (
                            base["aggregate_bytes"] if base and base.get("aggregate_bytes") else None
                        )
                        if relation != DEFAULT_RELATION and base_b is not None:
                            diff = b - base_b
                            bytes_str += f" ({diff:+d})"
                elif config in skipped:
                    row.append(NA)
                else:
                    row.append("--")
            row.append(verify_str)
            row.append(bytes_str)
            out += " & ".join(row) + " \\\\\n"
    out += "\\end{talltblr}\n\\end{table}\n"
    return out


def cell_logs(cell, directory):
    """Text of the run logs of `cell` (looked up in `<results>/logs` when the recorded path moved)."""
    texts = []
    for proc_entry in cell.get("processes", []) if cell else []:
        log = proc_entry.get("log")
        if not log:
            continue
        for path in (Path(log), directory / "logs" / Path(log).name):
            if path.exists():
                try:
                    texts.append(path.read_text(encoding="utf-8"))
                except OSError:
                    continue
                break
    return texts


def trace_values(text, pattern):
    return [float(m) for m in re.findall(pattern, text, re.MULTILINE)]


def timed_values(values, warmup):
    """Drop the warm-up runs' values when the timed runs also printed one."""
    return values[warmup:] if len(values) > warmup else values


def serial_trace(cell, directory):
    """Values of the traced Flock-after-the-PIOP run that the `tab:ablation` note quotes, in ms.

    `flock` is the Flock proofs alone on all threads (median run), `chain` the prover's
    parity chain (its trace stamp is cumulative and includes the chain input), `matrices` the verifier's bridge matrices and
    `vchain` the verifier's parity chain; the verifier bridge is `matrices + vchain`.
    """
    flock, chain, matrices, vchain = [], [], [], []
    warmup = int(cell.get("warmup") or 0) if cell else 0
    for text in cell_logs(cell, directory):
        flock += trace_values(text, r"median run: Flock parts ([\d.]+) ms")
        parity = timed_values(trace_values(text, r"^\s*\[tied\] parity chain\s+([\d.]+) ms"), warmup)
        chain += parity
        matrices += trace_values(text, r"^\s*matrices\s+bridge matrices from sd\s+([\d.]+) ms")
        vchain += trace_values(text, r"^\s*\[tied\] verify: parity chain\s+([\d.]+) ms")
    result = {}
    for key, values in (("flock", flock), ("chain", chain), ("matrices", matrices), ("vchain", vchain)):
        if values:
            result[key] = median(values)
    return result


# Verifier steps that the paper quotes per set.
VERIFY_STEPS = ("public_data", "public_t", "keccak", "matrices", "public_a")


def verifier_steps(cell, directory):
    """Step times (ms) of the last uncached and the last cached verifier run of a traced cell."""
    steps = {}
    for text in cell_logs(cell, directory):
        for block, label in (("uncached", "last uncached verifier run:"), ("cached", "last cached verifier run:")):
            start = text.rfind(label)
            if start < 0:
                continue
            end = text.find("total", start)
            body = text[start : end if end > 0 else len(text)]
            for key in VERIFY_STEPS:
                match = re.search(rf"^\s*{key}\s+.*?([\d.]+) ms\s*$", body, re.MULTILINE)
                if match:
                    steps[(block, key)] = float(match.group(1))
    return steps


def ablation_table(cells, directory, commit, machine, skipped):
    """Generate `tab-ablation.tex` (`\\label{tab:ablation}`): the GF(2)-side ablation.

    The rows are those of the paper on the AVX2 tier when the results have it: the
    untraced full aggregate, Flock after the PIOP (`AGG_FLOCK_THREADS=0`) and the
    mode-B public-bits aggregate. The traced run feeds the note only.
    """
    ours_44 = [c for c in cells if c.get("system") == "ours" and cell_params(c) == "44"]
    mode_b = [c for c in cells if c.get("system") == "mode_b" and cell_params(c) == "44"]
    if not ours_44 and not mode_b:
        return "% no ablation data\n"
    all_44 = ours_44 + mode_b
    # The batch size of the ablation and mode-B runs, not of the largest scaling cell.
    arm_cells = [c for c in all_44 if c.get("arm") or c.get("system") == "mode_b"]
    n = max(c["n"] for c in (arm_cells or all_44))
    at_n = [c for c in all_44 if c["n"] == n]
    alloc = preferred_allocation(at_n)
    # The paper reports the ablation on the AVX2 tier, as tab:scaling.
    tier = "avx2" if any(c.get("config") == "avx2" for c in at_n) else preferred_tier(at_n)

    # 1. Full aggregate: the untraced headline cell, else the ablation defaults arm.
    abl_defaults = [
        c
        for c in ours_44
        if c["n"] == n
        and c.get("config") == tier
        and c.get("allocation") == alloc
        and not c.get("traced")
        and c.get("arm") == "defaults"
        and c.get("times_ms")
    ]
    full_cell = pick_ours_cell(
        ours_44, params="44", config=tier, allocation=alloc, n=n, relation=DEFAULT_RELATION, prefer_tag="headline"
    ) or (abl_defaults[0] if abl_defaults else None)

    # 2. Traced default and traced Flock-after-the-PIOP (`timeline`).
    traced_defaults = [
        c
        for c in ours_44
        if c["n"] == n
        and c.get("config") == tier
        and c.get("allocation") == alloc
        and c.get("traced")
        and c.get("arm") in (None, "defaults")
        and c.get("times_ms")
    ]
    traced_serial = [
        c
        for c in ours_44
        if c["n"] == n
        and c.get("config") == tier
        and c.get("allocation") == alloc
        and c.get("traced")
        and c.get("arm") == "flock-serial"
        and c.get("times_ms")
    ]
    untraced_serial = [
        c
        for c in ours_44
        if c["n"] == n
        and c.get("config") == tier
        and c.get("allocation") == alloc
        and not c.get("traced")
        and c.get("arm") == "flock-serial"
        and c.get("times_ms")
    ]
    # The untraced arm gives the row; a traced run stands in only when it is all there is.
    serial_cell = untraced_serial[0] if untraced_serial else (traced_serial[0] if traced_serial else None)
    serial_is_traced = not untraced_serial and bool(traced_serial)

    # 3. Public bits (mode B).
    mode_b_matches = [
        c
        for c in mode_b
        if c["n"] == n and c.get("config") == tier and c.get("allocation") == alloc and c.get("times_ms")
    ]
    if not mode_b_matches:
        mode_b_matches = [c for c in mode_b if c["n"] == n and c.get("times_ms")]
    mode_b_cell = mode_b_matches[0] if mode_b_matches else None

    # Note: the traced Flock-after-the-PIOP run, as in the paper.
    trace = serial_trace(traced_serial[0], directory) if traced_serial else {}
    parts = []
    if "flock" in trace:
        part = f"Traced $\\GFt$ side alone: {fmt_seconds(trace['flock'])}\\,s"
        if "chain" in trace:
            part += f" (parity chain {fmt_ms(trace['chain'])}\\,ms)"
        parts.append(part)
    if "matrices" in trace and "vchain" in trace:
        parts.append(
            f"verifier bridge: {fmt_ms(trace['matrices'] + trace['vchain'])}\\,ms "
            f"(matrices {trace['matrices']:.1f}\\,ms)"
        )
    note = [f"$N={fmt_n_tex(n)}$, {LABEL[tier]}, {alloc}, uncached keys."]
    if parts:
        note.append("; ".join(parts) + ".")
    if serial_is_traced:
        note.append("Flock after the PIOP: traced run.")
    if traced_defaults and not parts:
        note.append(f"Full aggregate traced: {fmt_seconds(median(traced_defaults[0]['times_ms']))}\\,s.")

    out = header(directory, commit, machine)
    out += table_begin(
        "tab:ablation",
        "Ablation of the $\\GFt$ side for ML-DSA-44.",
        " ".join(note),
        "X[l] Q[r] Q[r] Q[r]",
        header_rows=1,
        colsep="6pt",
    )
    out += "Variant & Prove (s) & Verify (ms) & B/sig \\\\\n"

    if full_cell and full_cell.get("times_ms"):
        ver = fmt_ms(median(full_cell["verify_uncached_ms"])) if full_cell.get("verify_uncached_ms") else "--"
        out += (
            f"full aggregate & {fmt_seconds(median(full_cell['times_ms']))} & "
            f"{ver} & {fmt_bytes_per_sig(full_cell)} \\\\\n"
        )
    else:
        out += "full aggregate & -- & -- & -- \\\\\n"

    if serial_cell and serial_cell.get("times_ms"):
        bps = fmt_bytes_per_sig(serial_cell)
        if bps == "--" and full_cell:
            bps = fmt_bytes_per_sig(full_cell)
        out += f"Flock after the PIOP & {fmt_seconds(median(serial_cell['times_ms']))} & -- & {bps} \\\\\n"
    else:
        out += "Flock after the PIOP & -- & -- & -- \\\\\n"

    if mode_b_cell and mode_b_cell.get("times_ms"):
        ver = (
            fmt_ms(median(mode_b_cell["verify_uncached_ms"]))
            if mode_b_cell.get("verify_uncached_ms")
            else "--"
        )
        out += (
            f"public bits & {fmt_seconds(median(mode_b_cell['times_ms']))} & "
            f"{ver} & {fmt_bytes_per_sig(mode_b_cell)} \\\\\n"
        )
    else:
        out += "public bits & -- & -- & -- \\\\\n"

    out += "\\end{talltblr}\n\\end{table}\n"
    return out


def ablation_controls_table(cells, directory, commit, machine):
    """Generate `tab-ablation-controls.tex`: the control arms of the ablation group (not in the paper)."""
    arms = [
        c
        for c in cells
        if c.get("system") == "ours"
        and cell_params(c) == "44"
        and c.get("arm")
        and not c.get("traced")
        and c.get("times_ms")
    ]
    if not arms:
        return "% no control ablation data\n"
    n = max(c["n"] for c in arms)
    at_n = [c for c in arms if c["n"] == n]
    alloc = preferred_allocation(at_n)
    tier = "avx2" if any(c.get("config") == "avx2" for c in at_n) else preferred_tier(at_n)
    chosen = [c for c in at_n if c.get("config") == tier and c.get("allocation") == alloc]
    by_arm = {c["arm"]: c for c in chosen}
    out = header(directory, commit, machine)
    out += table_begin(
        "tab:ablation-controls",
        "Control ablations for ML-DSA-44.",
        f"$N={fmt_n_tex(n)}$, {LABEL[tier]}, {alloc}, uncached keys, untraced.",
        "X[l] Q[r] Q[r] Q[r]",
        header_rows=1,
        colsep="6pt",
    )
    out += "Variant & Prove (s) & Verify (ms) & B/sig \\\\\n"
    labels = {"defaults": "defaults", **ABLATION_LABELS}
    for arm_key, arm_label in labels.items():
        cell = by_arm.get(arm_key)
        if not cell:
            continue
        ver = fmt_ms(median(cell["verify_uncached_ms"])) if cell.get("verify_uncached_ms") else "--"
        out += (
            f"{arm_label} & {fmt_seconds(median(cell['times_ms']))} & "
            f"{ver} & {fmt_bytes_per_sig(cell)} \\\\\n"
        )
    out += "\\end{talltblr}\n\\end{table}\n"
    return out


def scaling_table(cells, directory, commit, machine, skipped):
    """Generate `tab-scaling.tex` (`\\label{tab:scaling}`): scaling with the batch size across sets."""
    standard = [c for c in cells if is_standard_ours(c) and c.get("relation") == DEFAULT_RELATION and c.get("times_ms")]
    if not standard:
        return "% no scaling data\n"
    alloc = preferred_allocation(standard)
    # The paper reports scaling on the AVX2 tier, which every x86-64-v3 CPU runs.
    tier = "avx2" if any(c.get("config") == "avx2" for c in standard) else preferred_tier(standard)
    sets = [p for p in PARAM_SETS if any(cell_params(c) == p for c in standard)] or list(PARAM_SETS)
    paper_sizes = [1024, 8192, 32768]

    set_rows = []
    for params in sets:
        available_sizes = sorted(
            {
                c["n"]
                for c in standard
                if cell_params(c) == params and c.get("config") == tier and c.get("allocation") == alloc
            }
        )
        if set(paper_sizes) <= set(available_sizes):
            chosen_sizes = paper_sizes
        elif available_sizes:
            chosen_sizes = available_sizes
        else:
            chosen_sizes = sorted({c["n"] for c in standard})
        set_rows.append((params, chosen_sizes))

    extra_hlines = []
    row_cursor = 2
    for idx, (_, chosen_sizes) in enumerate(set_rows):
        row_cursor += len(chosen_sizes)
        if idx + 1 < len(set_rows):
            extra_hlines.append(row_cursor)

    out = header(directory, commit, machine)
    out += table_begin(
        "tab:scaling",
        "Scaling with the batch size.",
        f"{LABEL[tier]} on {alloc}. Verify: uncached / cached keys. Memory: prover peak. "
        "Compression: signature bytes over B/sig.",
        "Q[l] Q[r] X[r] Q[r] Q[r] Q[r] Q[r]",
        header_rows=1,
        extra_hlines=extra_hlines,
        colsep="2pt",
    )
    out += "Set & $N$ & B/sig & Compression & Prove (s) & Verify (ms) & Memory (GiB) \\\\\n"
    for params, chosen_sizes in set_rows:
        nrows = len(chosen_sizes)
        for idx, n in enumerate(chosen_sizes):
            set_col = f"\\SetCell[r={nrows}]{{l}} {params}" if idx == 0 else ""
            cell = pick_ours_cell(
                standard,
                params=params,
                config=tier,
                allocation=alloc,
                n=n,
                relation=DEFAULT_RELATION,
                prefer_tag="scaling",
            )
            if cell:
                bps = fmt_bytes_per_sig(cell)
                ratio = fmt_ratio(PARAM_SETS[params]["sig_bytes"], bytes_per_sig(cell))
                prove = fmt_seconds(median(cell["times_ms"])) if cell.get("times_ms") else "--"
                if cell.get("verify_uncached_ms"):
                    unc = fmt_ms(median(cell["verify_uncached_ms"]))
                    cac = (
                        fmt_ms(median(cell["verify_cached_ms"]))
                        if cell.get("verify_cached_ms")
                        else "--"
                    )
                    verify = f"{unc} / {cac}"
                else:
                    verify = "--"
                peak = fmt_gib(cell.get("peak_rss_kib"))
            else:
                bps = ratio = prove = verify = peak = "--"
            out += f"{set_col} & {fmt_pow2_or_n(n)} & {bps} & {ratio} & {prove} & {verify} & {peak} \\\\\n"
    out += "\\end{talltblr}\n\\end{table}\n"
    return out


def extract_breakdown(cell, directory):
    """Return the component byte dict for `cell` (from the cell JSON or its log file)."""
    if not cell:
        return None
    if cell.get("size_breakdown"):
        return cell["size_breakdown"]
    for proc_entry in cell.get("processes", []):
        log_path = proc_entry.get("log")
        if log_path and Path(log_path).exists():
            parsed = parse_size_breakdown(Path(log_path).read_text(encoding="utf-8"))
            if parsed:
                return parsed
    logs_dir = directory / "logs"
    if logs_dir.exists():
        params = cell_params(cell)
        n = cell.get("n")
        relation = cell.get("relation", DEFAULT_RELATION)
        pattern = f"*-{n}-{relation}-p{params}-*.log"
        for log_file in sorted(logs_dir.glob(pattern)):
            try:
                parsed = parse_size_breakdown(log_file.read_text(encoding="utf-8"))
            except OSError:
                continue
            if parsed:
                return parsed
    return None


def bytes_table(cells, directory, commit, machine):
    """Generate `tab-bytes.tex` (`\\label{tab:bytes}`): composition of the aggregate across sets."""
    standard = [
        c for c in cells if is_standard_ours(c) and c.get("relation") == DEFAULT_RELATION and c.get("aggregate_bytes")
    ]
    if not standard:
        return "% no byte data\n"
    sets = [p for p in PARAM_SETS if any(cell_params(c) == p for c in standard)] or list(PARAM_SETS)
    # The largest N that every set has, so that the caption holds for each column.
    common = set.intersection(*({c["n"] for c in standard if cell_params(c) == p} for p in sets))
    n = max(common) if common else max(c["n"] for c in standard)
    chosen = {}
    breakdowns = {}
    for params in sets:
        matches = [c for c in standard if cell_params(c) == params and c["n"] == n]
        if not matches:
            matches = [c for c in standard if cell_params(c) == params]
        if matches:
            cell = max(matches, key=lambda c: cell_priority(c, prefer_tag="headline"))
            chosen[params] = cell
            bd = extract_breakdown(cell, directory)
            if not bd:
                for candidate in sorted(
                    matches, key=lambda c: cell_priority(c, prefer_tag="headline"), reverse=True
                ):
                    bd = extract_breakdown(candidate, directory)
                    if bd:
                        break
            breakdowns[params] = bd

    out = header(directory, commit, machine)
    colspec = "X[2.4,l] " + " ".join("X[r]" for _ in sets)
    out += table_begin(
        "tab:bytes",
        f"Composition of the aggregate at $N={fmt_n_tex(n)}$.",
        "Bytes per signature. Commitments: Akita and Flock; other: bridge images, nonces, header.",
        colspec,
        header_rows=1,
        extra_hlines=[9],
    )
    out += "Component & " + " & ".join(sets) + " \\\\\n"

    def comp_val(params, fn):
        cell = chosen.get(params)
        bd = breakdowns.get(params)
        if not cell or not cell.get("n"):
            return "--"
        val = fn(cell, bd)
        if val is None:
            return "--"
        return f"{val / cell['n']:.1f}"

    components = [
        (
            "hashes $\\ct_i$",
            lambda cell, bd: bd["c_tilde"]
            if bd and "c_tilde" in bd
            else PARAM_SETS[cell_params(cell)]["ctilde_bytes"] * cell["n"],
        ),
        ("Flock proof", lambda cell, bd: bd.get("flock_proofs") if bd else None),
        ("Akita opening", lambda cell, bd: bd.get("opening") if bd else None),
        ("lookup", lambda cell, bd: bd.get("lookup") if bd else None),
        ("PIOP rounds and claims", lambda cell, bd: bd.get("piop") if bd else None),
        (
            "commitments",
            lambda cell, bd: (bd["akita_commitments"] + bd["flock_commitments"])
            if bd and "akita_commitments" in bd and "flock_commitments" in bd
            else None,
        ),
        (
            "other",
            lambda cell, bd: (bd["final_vectors"] + bd["work"] + bd["header"])
            if bd and "final_vectors" in bd and "work" in bd and "header" in bd
            else None,
        ),
        ("total", lambda cell, bd: cell.get("aggregate_bytes")),
    ]

    for label, fn in components:
        row = [label] + [comp_val(params, fn) for params in sets]
        out += " & ".join(row) + " \\\\\n"

    out += "\\end{talltblr}\n\\end{table}\n"
    return out


# Falcon signature bytes (Falcon specification), the numerator of Aerie's compression factor.
FALCON_SIG_BYTES = {512: 666, 1024: 1280}
# Thread count of each allocation, used to place Alloul26 runs, which record
# only their thread count.
ALLOC_THREADS = {32: "16C32T", 16: "8C16T", 1: "1T"}
# Rows of tab:baselines per system: Aerie and Alloul26 have no IFMA build.
BASELINE_ISA = [("avx2", "AVX2"), ("avx512", "AVX-512")]
OURS_ISA = [("avx2", "AVX2"), ("avx512", "AVX-512"), ("ifma", "IFMA")]
# The Alloul26 family the paper reports (Johnson).
ALLOUL_FAMILY = "J"
ALLOUL_PARAMS = "87"


def load_alloul26(directory, extra_paths=()):
    """Alloul26 runs: the JSON lines `scripts/alloul26.sh` prints, collected from
    `<results>/alloul26*.jsonl`, `<results>/alloul26/*.json*` and `extra_paths`."""
    paths = sorted(directory.glob("alloul26*.jsonl")) + sorted((directory / "alloul26").glob("*.json*"))
    paths += [Path(p) for p in extra_paths]
    runs = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                run = json.loads(line)
            except ValueError:
                continue
            if run.get("system") == "Alloul26" and run.get("prove_s") is not None:
                runs.append(run)
    return runs


def falcon_degree(cell):
    if cell.get("degree"):
        try:
            return int(cell["degree"])
        except ValueError:
            pass
    for key in ("tag", "label", "relation"):
        val = str(cell.get(key, ""))
        if "1024" in val:
            return 1024
        if "512" in val:
            return 512
    return 512


def falcon_category(deg):
    return "1" if deg == 512 else ("5" if deg == 1024 else str(deg))


def is_aerie_avx2(cell):
    return (
        "avx2" in str(cell.get("tag", "")).lower()
        or cell.get("knobs", {}).get("AERIE_SIMD_BACKEND") == "avx2"
    )


def median_of(cell, key):
    return median(cell[key]) if cell and cell.get(key) else None


def baselines_table(cells, directory, commit, machine, skipped, alloul=()):
    """Generate `tab-baselines.tex` (`\\label{tab:baselines}`): Aerie, ours for every set
    and tier, and Alloul26, with prove and verify times on 16C32T, 8C16T and 1T. Native verification times go into the table note."""
    standard = [c for c in cells if is_standard_ours(c) and c.get("relation") == DEFAULT_RELATION and c.get("times_ms")]
    aerie = [c for c in cells if c.get("system") == "aerie"]
    native = [c for c in cells if c.get("system") == "native"]
    if not standard and not aerie and not native and not alloul:
        return "% no baseline data\n"

    # Compare at the largest N that has a baseline cell, so that a run whose own
    # cells go further (quick: ML-DSA-44 at 4K, baselines at 1K) still fills the rows.
    baseline_sizes = [c["n"] for c in aerie + native] + [int(r["N"]) for r in alloul]
    all_cells = standard + aerie + native
    if baseline_sizes:
        n = max(baseline_sizes)
    else:
        n = max(c["n"] for c in all_cells)
    sets = [p for p in PARAM_SETS if any(cell_params(c) == p for c in standard)] or list(PARAM_SETS)
    prove_allocs = ["16C32T", "8C16T", "1T"]
    verify_allocs = ["16C32T", "8C16T", "1T"]

    def pick_native(params, alloc):
        matches = [
            c for c in native if cell_params(c) == params and c.get("allocation") == alloc and c.get("n") == n
        ]
        return max(matches, key=lambda c: c.get("reps", 0)) if matches else None

    def pick_aerie(deg, isa, alloc):
        matches = [
            c
            for c in aerie
            if falcon_degree(c) == deg
            and is_aerie_avx2(c) == (isa == "avx2")
            and c.get("allocation") == alloc
            and c.get("n") == n
        ]
        return max(matches, key=lambda c: c.get("reps", 0)) if matches else None

    def alloul_runs(isa, alloc):
        return [
            r
            for r in alloul
            if r.get("family") == ALLOUL_FAMILY
            and int(r.get("N", 0)) == n
            and r.get("isa") == isa
            and ALLOC_THREADS.get(int(r.get("threads", 0))) == alloc
        ]

    def row(cols):
        return re.sub(r" {2,}", " ", " & ".join(cols) + " \\\\") + "\n"

    body = []
    hlines = []
    cursor = 3  # first body row: two header rows above it

    # Aerie: Falcon-512 and Falcon-1024, one row per ISA; B/sig and ratio span the rows.
    for deg in (512, 1024):
        per_isa = {isa: {a: pick_aerie(deg, isa, a) for a in prove_allocs} for isa, _ in BASELINE_ISA}
        ref = next((c for isa in per_isa for c in per_isa[isa].values() if bytes_per_sig(c)), None)
        per_sig = bytes_per_sig(ref)
        rows = len(BASELINE_ISA)
        for idx, (isa, label) in enumerate(BASELINE_ISA):
            picked = per_isa[isa]
            prove = [fmt_seconds(median_of(picked[a], "times_ms")) for a in prove_allocs]
            # Aerie reports a parallel and a single-thread verifier time in every run.
            v_1t = median_of(picked["1T"], "verify_single_ms") or median_of(picked["1T"], "verify_uncached_ms")
            v_all = [median_of(picked[a], "verify_uncached_ms") for a in verify_allocs if a != "1T"] + [v_1t]
            verify = [fmt_ms(v, allow_decimal=True) for v in v_all]
            if idx == 0:
                lead = [
                    f"\\SetCell[r={rows}]{{l}} Aerie-{deg}~\\cite{{CDBT26}}",
                    f"\\SetCell[r={rows}]{{c}} {falcon_category(deg)}",
                ]
                tail = [
                    f"\\SetCell[r={rows}]{{r}} " + (f"{per_sig:.1f}" if per_sig else "--"),
                    f"\\SetCell[r={rows}]{{r}} " + fmt_ratio(FALCON_SIG_BYTES[deg], per_sig),
                ]
            else:
                lead, tail = ["", ""], ["", ""]
            body.append(row(lead + [label] + prove + verify + tail))
        cursor += rows

    # Ours: every set, one row per tier.
    for params in sets:
        hlines.append(cursor)
        constants = PARAM_SETS[params]
        rows = len(OURS_ISA)
        picked_all = {
            config: {
                a: pick_ours_cell(
                    standard, params=params, config=config, allocation=a, n=n, relation=DEFAULT_RELATION, prefer_tag="headline"
                )
                for a in prove_allocs
            }
            for config, _ in OURS_ISA
        }
        ref = next((c for config in picked_all for c in picked_all[config].values() if bytes_per_sig(c)), None)
        per_sig = bytes_per_sig(ref)
        for idx, (config, label) in enumerate(OURS_ISA):
            picked = picked_all[config]
            prove = [fmt_seconds(median_of(picked[a], "times_ms")) for a in prove_allocs]
            verify = [fmt_ms(median_of(picked[a], "verify_uncached_ms")) for a in verify_allocs]
            if idx == 0:
                lead = [
                    f"\\SetCell[r={rows}]{{l}} Ours-{params}",
                    f"\\SetCell[r={rows}]{{c}} {constants['category']}",
                ]
                tail = [
                    f"\\SetCell[r={rows}]{{r}} " + (f"{per_sig:.1f}" if per_sig else "--"),
                    f"\\SetCell[r={rows}]{{r}} " + fmt_ratio(constants["sig_bytes"], per_sig),
                ]
            else:
                lead, tail = ["", ""], ["", ""]
            body.append(row(lead + [label] + prove + verify + tail))
        cursor += rows

    # Alloul26: ML-DSA-87, family J, one row per ISA build.
    hlines.append(cursor)
    constants = PARAM_SETS[ALLOUL_PARAMS]
    rows = len(BASELINE_ISA)
    per_sig_runs = [r["bytes_per_sig"] for r in alloul if r.get("family") == ALLOUL_FAMILY and int(r.get("N", 0)) == n and r.get("bytes_per_sig")]
    per_sig = median(per_sig_runs) if per_sig_runs else None
    for idx, (isa, label) in enumerate(BASELINE_ISA):
        prove = []
        for a in prove_allocs:
            runs = alloul_runs(isa, a)
            prove.append(fmt_seconds(median([r["prove_s"] * 1000 for r in runs])) if runs else "--")
        verify = []
        for a in verify_allocs:
            runs = [r for r in alloul_runs(isa, a) if r.get("verify_s") is not None]
            verify.append(fmt_ms(median([r["verify_s"] * 1000 for r in runs])) if runs else "--")
        if idx == 0:
            lead = [
                f"\\SetCell[r={rows}]{{l}} Alloul26-{ALLOUL_PARAMS}~\\cite{{Alloul26}}",
                f"\\SetCell[r={rows}]{{c}} {constants['category']}",
            ]
            tail = [
                f"\\SetCell[r={rows}]{{r}} " + (f"{per_sig:.1f}" if per_sig else "--"),
                f"\\SetCell[r={rows}]{{r}} " + fmt_ratio(constants["sig_bytes"], per_sig),
            ]
        else:
            lead, tail = ["", ""], ["", ""]
        body.append(row(lead + [label] + prove + verify + tail))

    # Native AVX2 verification (reference code) sits in the note, not in a row.
    native_par = [fmt_ms(median_of(pick_native(p, "16C32T"), "verify_uncached_ms")) for p in sets]
    native_1t = [fmt_ms(median_of(pick_native(p, "1T"), "verify_uncached_ms")) for p in sets]

    def series(values):
        return values[0] if len(values) == 1 else ", ".join(values[:-1]) + " and " + values[-1]

    set_names = series([f"-{p}" for p in sets]).replace("-", "ML-DSA-", 1)
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}

    def medians_of(counts):
        counts = sorted(set(counts))
        if len(counts) != 1:
            return ""
        if counts[0] == 1:
            return "one run, "
        return f"medians of {words.get(counts[0], str(counts[0]))} runs, "

    aerie_runs = medians_of(len(c["times_ms"]) for c in aerie if c.get("n") == n and c.get("times_ms"))
    alloul_counts = {}
    for r in alloul:
        if r.get("family") == ALLOUL_FAMILY and int(r.get("N", 0)) == n:
            key = (r.get("isa"), r.get("threads"))
            alloul_counts[key] = alloul_counts.get(key, 0) + 1
    alloul_runs = medians_of(alloul_counts.values()).rstrip(", ")
    note = (
        "Cat.: NIST category. Uncached keys; keys and messages excluded. "
        "AVX-512: F, BW and VBMI, no IFMA. Compression: signature bytes over B/sig. "
        f"Aerie: {aerie_runs}prove without setup, B/sig with 40\\,B salt. "
        "Alloul26: Johnson family" + (f", {alloul_runs}." if alloul_runs else ".")
    )
    if any(v != "--" for v in native_par + native_1t):
        note += (
            f" Native AVX2 verification (reference code): {series(native_par)}\\,ms on 16C32T and "
            f"{series(native_1t)}\\,ms on 1T for {set_names}."
        )
    if skipped:
        note += " Tiers this CPU lacks: " + ", ".join(sorted(skipped)) + "."

    out = header(directory, commit, machine)
    out += table_begin(
        "tab:baselines",
        f"Baselines at $N={fmt_n_tex(n)}$.",
        note,
        "Q[l] Q[c] Q[l] X[r] X[r] X[r] X[r] X[r] X[r] X[r] Q[r]",
        header_rows=2,
        extra_hlines=hlines,
        env="table*",
        rowsep="0.6pt",
    )
    out += (
        "\\SetCell[r=2]{l} System & \\SetCell[r=2]{c} Cat. & \\SetCell[r=2]{l} ISA & "
        "\\SetCell[c=3]{c} Prove (s) & & & \\SetCell[c=3]{c} Verify (ms) & & & "
        "\\SetCell[r=2]{r} B/sig & \\SetCell[r=2]{r} Compression \\\\\n"
    )
    out += " & & & " + " & ".join(prove_allocs + verify_allocs) + " & & \\\\\n"
    out += "".join(body)
    out += table_end("table*")
    return out


def numbers(cells, directory, commit, machine):
    """Generate `numbers.tex` with macros for all sets, relations and mode B."""
    standard = [c for c in cells if is_standard_ours(c) and c.get("times_ms")]
    lines = [header(directory, commit, machine)]
    if not standard:
        return "".join(lines) + "% no headline data\n"
    n = max(c["n"] for c in standard)
    at_n = [c for c in standard if c["n"] == n]
    alloc = preferred_allocation(at_n)
    values = {}

    for params, set_suffix in SET_MACRO.items():
        for config in ORDER:
            cell = pick_ours_cell(
                at_n,
                params=params,
                config=config,
                allocation=alloc,
                n=n,
                relation=DEFAULT_RELATION,
                prefer_tag="headline",
            )
            if not cell:
                continue
            m_tier = MACRO[config]
            prove_s = fmt_seconds(median(cell["times_ms"]))
            values[f"Prove{m_tier}{set_suffix}"] = prove_s
            if params == "44":
                values[f"Prove{m_tier}"] = prove_s
            if cell.get("verify_uncached_ms"):
                unc = fmt_ms(median(cell["verify_uncached_ms"]))
                values[f"Verify{m_tier}Uncached{set_suffix}"] = unc
                if params == "44":
                    values[f"Verify{m_tier}Uncached"] = unc
            if cell.get("verify_cached_ms"):
                cac = fmt_ms(median(cell["verify_cached_ms"]))
                values[f"Verify{m_tier}Cached{set_suffix}"] = cac
                if params == "44":
                    values[f"Verify{m_tier}Cached"] = cac
            if cell.get("aggregate_bytes"):
                bps = fmt_bytes_per_sig(cell)
                agg_b = fmt_n_tex(cell["aggregate_bytes"])
                values[f"BytesPerSig{m_tier}{set_suffix}"] = bps
                values[f"AggregateBytes{set_suffix}"] = agg_b
                if params == "44":
                    values[f"BytesPerSig{m_tier}"] = bps
                    values["AggregateBytes"] = agg_b

        # The reference relation H-exact against H-FIPS on IFMA for each set (opt-in runs only).
        base_cell = pick_ours_cell(
            at_n,
            params=params,
            config="ifma",
            allocation=alloc,
            n=n,
            relation=DEFAULT_RELATION,
            prefer_tag="relations",
        )
        exact_cell = pick_ours_cell(
            at_n,
            params=params,
            config="ifma",
            allocation=alloc,
            n=n,
            relation="h-exact",
            prefer_tag="relations",
        )
        if base_cell and base_cell.get("times_ms") and exact_cell and exact_cell.get("times_ms"):
            base_ms = median(base_cell["times_ms"])
            if base_ms > 0:
                pct = (median(exact_cell["times_ms"]) - base_ms) / base_ms * 100.0
                values[f"RelExactVsFipsIfma{set_suffix}"] = f"{pct:+.1f}\\%"

    if "ProveIfma" in values:
        try:
            prove = float(values["ProveIfma"])
            for config in ("avx512", "avx2"):
                name = f"Prove{MACRO[config]}"
                if name in values and prove > 0:
                    values[f"GainIfmaOver{MACRO[config]}"] = f"{float(values[name]) / prove:.2f}"
        except ValueError:
            pass

    mode_b = [c for c in cells if c.get("system") == "mode_b" and c.get("times_ms")]
    if mode_b:
        mb_n = max(c["n"] for c in mode_b)
        mb_cells = [c for c in mode_b if c["n"] == mb_n]
        # The AVX2 tier, as in tab:ablation.
        mb_cell = next((c for c in mb_cells if c.get("config") == "avx2"), mb_cells[0])
        values["ProveModeB"] = fmt_seconds(median(mb_cell["times_ms"]))
        if mb_cell.get("verify_uncached_ms"):
            values["VerifyModeBUncached"] = fmt_ms(median(mb_cell["verify_uncached_ms"]))
        if mb_cell.get("aggregate_bytes"):
            values["BytesPerSigModeB"] = fmt_bytes_per_sig(mb_cell)

    # Traced runs: the tab:ablation note and the per-set verifier steps. AVX2 when present.
    traced = [c for c in cells if c.get("system") == "ours" and c.get("traced") and c.get("times_ms")]
    if traced:
        tr_n = max(c["n"] for c in traced)
        tr_cells = [c for c in traced if c["n"] == tr_n]
        tr_alloc = preferred_allocation(tr_cells)
        tr_tier = "avx2" if any(c.get("config") == "avx2" for c in tr_cells) else preferred_tier(tr_cells)
        tr_cells = [c for c in tr_cells if c.get("config") == tr_tier and c.get("allocation") == tr_alloc]
        serial = next((c for c in tr_cells if cell_params(c) == "44" and c.get("arm") == "flock-serial"), None)
        trace = serial_trace(serial, directory) if serial else {}
        if "flock" in trace:
            values["TraceFlockAlone"] = fmt_seconds(trace["flock"])
        if "chain" in trace:
            values["TraceParityChain"] = fmt_ms(trace["chain"])
        if "matrices" in trace and "vchain" in trace:
            values["TraceVerifyMatrices"] = f"{trace['matrices']:.1f}"
            values["TraceVerifyBridge"] = fmt_ms(trace["matrices"] + trace["vchain"])
        for params, set_suffix in SET_MACRO.items():
            cell = next(
                (c for c in tr_cells if cell_params(c) == params and c.get("arm") in (None, "defaults")),
                None,
            )
            steps = verifier_steps(cell, directory) if cell else {}
            if ("uncached", "keccak") in steps:
                values[f"TraceVerifyFlock{set_suffix}"] = fmt_ms(steps[("uncached", "keccak")])
            if ("uncached", "public_data") in steps and ("uncached", "public_t") in steps:
                keys = steps[("uncached", "public_data")] + steps[("uncached", "public_t")]
                values[f"TraceVerifyKeys{set_suffix}"] = fmt_ms(keys)
            if ("uncached", "public_a") in steps:
                values[f"TraceVerifyExpandA{set_suffix}"] = fmt_ms(steps[("uncached", "public_a")])
            if ("cached", "public_a") in steps:
                values[f"TraceVerifyExpandACached{set_suffix}"] = fmt_ms(steps[("cached", "public_a")])

    for name, value in values.items():
        if value:
            lines.append(f"\\newcommand{{\\{name}}}{{{value}}}\n")
    return "".join(lines)


def claims_csv(cells, directory):
    rows = []
    for cell in cells:
        if cell.get("system") not in ("ours", "mode_b", "native", "aerie"):
            continue
        prove = describe(cell.get("times_ms") or [])
        uncached = describe(cell.get("verify_uncached_ms") or [])
        cached = describe(cell.get("verify_cached_ms") or [])
        rows.append(
            {
                "run_id": cell["run_id"],
                "system": cell.get("system", ""),
                "params": cell_params(cell) if cell.get("system") != "aerie" else "",
                "config": cell["config"],
                "allocation": cell["allocation"],
                "n": cell["n"],
                "relation": cell["relation"],
                "prove_median_ms": prove["median"] if prove else "",
                "verify_uncached_median_ms": uncached["median"] if uncached else "",
                "verify_cached_median_ms": cached["median"] if cached else "",
                "peak_rss_kib": cell.get("peak_rss_kib", ""),
                "aggregate_bytes": cell.get("aggregate_bytes", ""),
                "aggregate_sha256": cell.get("aggregate_sha256", ""),
            }
        )
    path = directory / "claims-measured.csv"
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", help="results directory (default: LATEST)")
    parser.add_argument("--out", help="output directory for generated tables (default: <results>/paper-tables)")
    parser.add_argument(
        "--alloul26",
        action="append",
        default=[],
        metavar="FILE",
        help="JSON lines printed by `./run.sh alloul26` (in addition to <results>/alloul26*.jsonl)",
    )
    args = parser.parse_args()
    directory = result_dir(args.results)
    cells = load_cells(directory)
    skipped = skipped_configs(directory)
    alloul = load_alloul26(directory, args.alloul26)
    commit = next(
        (c.get("commit") for c in cells if c.get("system") == "ours" and c.get("commit")),
        "unknown",
    )
    machine = next((c.get("machine") for c in cells if c.get("machine")), "unknown")
    outputs = {
        "tab-ablation.tex": ablation_table(cells, directory, commit, machine, skipped),
        "tab-ablation-controls.tex": ablation_controls_table(cells, directory, commit, machine),
        "tab-scaling.tex": scaling_table(cells, directory, commit, machine, skipped),
        "tab-bytes.tex": bytes_table(cells, directory, commit, machine),
        "tab-baselines.tex": baselines_table(cells, directory, commit, machine, skipped, alloul),
        "tab-relations.tex": relations_table(cells, directory, commit, machine, skipped),
        "numbers.tex": numbers(cells, directory, commit, machine),
    }
    out_dir = Path(args.out) if args.out else (directory / "paper-tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [out_dir] if args.out else [out_dir, TABLES]
    for target in targets:
        target.mkdir(parents=True, exist_ok=True)
        # The prover table was merged into tab-baselines.tex; drop a copy left by an older run.
        (target / "tab-prove.tex").unlink(missing_ok=True)
        for name, text in outputs.items():
            (target / name).write_text(text, encoding="utf-8")
    claims_csv(cells, directory)
    print(f"tables: {out_dir}")
    for name in outputs:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
