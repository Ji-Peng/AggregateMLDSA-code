#!/usr/bin/env python3
"""Turn raw measurement JSON into the paper's LaTeX tables and in-text macros."""

import argparse
import csv
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
    "flock-serial": "Flock after the PIOP, untraced (\\texttt{AGG\\_FLOCK\\_THREADS=0})",
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
    if ms is None:
        return "--"
    return f"{ms / 1000:.2f}"


def fmt_ms(ms, allow_decimal=False):
    if ms is None:
        return "--"
    if allow_decimal and ms < 100:
        return f"{ms:.1f}"
    return f"{int(round(ms)):,}".replace(",", "{,}")


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


def table_begin(label, caption, note, colspec, header_rows=1, extra_hlines=None):
    hline_block = f"  hline{{1,{header_rows + 1},Z}} = {{1pt}},\n"
    if extra_hlines:
        rows_str = ",".join(str(r) for r in extra_hlines)
        hline_block += f"  hline{{{rows_str}}} = {{0.5pt}},\n"
    return (
        "\\begin{table}[t]\n\\centering\n"
        "\\begin{talltblr}[\n"
        f"  caption = {{{caption}}},\n"
        f"  label = {{{label}}},\n"
        f"  remark{{Note}} = {{{note}}},\n"
        "]{\n"
        f"  width = \\linewidth,\n  colspec = {{{colspec}}},\n"
        "  colsep = 4pt,\n"
        f"  row{{1{(',' + str(header_rows)) if header_rows > 1 else ''}}} = {{font=\\bfseries}},\n"
        "  rows = {valign=m},\n"
        f"{hline_block}"
        "}\n"
    )


def prove_table(cells, directory, commit, machine, skipped):
    """Generate `tab-prove.tex` (`\\label{tab:prove}`): prover time by set, tier and allocation."""
    standard = [c for c in cells if is_standard_ours(c) and c.get("relation") == DEFAULT_RELATION and c.get("times_ms")]
    if not standard:
        return "% no prover data\n"
    n = max(c["n"] for c in standard)
    sets = [p for p in PARAM_SETS if any(cell_params(c) == p for c in standard)] or list(PARAM_SETS)
    allocs = ["16C32T", "8C16T", "1T"]
    extra_hlines = [2 + 3 * i for i in range(1, len(sets))]
    out = header(directory, commit, machine)
    out += table_begin(
        "tab:prove",
        f"Prover time at $N={fmt_n_tex(n)}$ in seconds.",
        "$\\Relfips$; medians of the measured runs. The aggregate bytes do not depend on the tier or the allocation.",
        "Q[l] X[l] X[r] X[r] X[r]",
        header_rows=1,
        extra_hlines=extra_hlines,
    )
    out += "Set & Tier & " + " & ".join(allocs) + " \\\\\n"
    for params in sets:
        for idx, config in enumerate(ORDER):
            set_col = f"\\SetCell[r=3]{{l}} {params}" if idx == 0 else ""
            row = [set_col, LABEL[config]]
            for allocation in allocs:
                cell = pick_ours_cell(
                    standard,
                    params=params,
                    config=config,
                    allocation=allocation,
                    n=n,
                    relation=DEFAULT_RELATION,
                    prefer_tag="headline",
                )
                if cell and cell.get("times_ms"):
                    row.append(fmt_seconds(median(cell["times_ms"])))
                elif config in skipped:
                    row.append(NA)
                else:
                    row.append("--")
            out += " & ".join(row) + " \\\\\n"
    out += "\\end{talltblr}\n\\end{table}\n"
    return out


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
        "Prove in seconds (change against $\\Relfips$ in parentheses); verify in ms on IFMA, uncached / cached keys.",
        "Q[l] Q[l] X[r] X[r] X[r] X[1.1,r] X[1.2,r]",
        header_rows=1,
        extra_hlines=extra_hlines,
    )
    out += "Set & Rel. & " + " & ".join(LABEL[c] for c in ORDER) + " & Verify & Bytes \\\\\n"
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


def ablation_table(cells, directory, commit, machine, skipped):
    """Generate `tab-ablation.tex` (`\\label{tab:ablation}`): GF(2)-side and control ablations."""
    ours_44 = [c for c in cells if c.get("system") == "ours" and cell_params(c) == "44"]
    mode_b = [c for c in cells if c.get("system") == "mode_b" and cell_params(c) == "44"]
    if not ours_44 and not mode_b:
        return "% no ablation data\n"
    all_44 = ours_44 + mode_b
    n = max(c["n"] for c in all_44)
    at_n = [c for c in all_44 if c["n"] == n]
    alloc = preferred_allocation(at_n)
    tier = preferred_tier(at_n)

    # 1. Full aggregate (untraced default): prefer ablation defaults arm, then headline.
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
    full_cell = (
        abl_defaults[0]
        if abl_defaults
        else pick_ours_cell(
            ours_44, params="44", config=tier, allocation=alloc, n=n, relation=DEFAULT_RELATION, prefer_tag="headline"
        )
    )

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
    serial_cell = traced_serial[0] if traced_serial else (untraced_serial[0] if untraced_serial else None)

    # 3. Public bits (mode B).
    mode_b_matches = [
        c
        for c in mode_b
        if c["n"] == n and c.get("config") == tier and c.get("allocation") == alloc and c.get("times_ms")
    ]
    if not mode_b_matches:
        mode_b_matches = [c for c in mode_b if c["n"] == n and c.get("times_ms")]
    mode_b_cell = mode_b_matches[0] if mode_b_matches else None

    # Additional control ablation arms (untraced).
    extra_arms = []
    for arm_key, arm_label in ABLATION_LABELS.items():
        if arm_key == "flock-serial" and not traced_serial:
            continue
        matches = [
            c
            for c in ours_44
            if c["n"] == n
            and c.get("config") == tier
            and c.get("allocation") == alloc
            and not c.get("traced")
            and c.get("arm") == arm_key
            and c.get("times_ms")
        ]
        if matches:
            extra_arms.append((arm_label, matches[0]))

    traced_note = "Flock after the PIOP: traced run"
    if traced_defaults:
        traced_s = fmt_seconds(median(traced_defaults[0]["times_ms"]))
        traced_note += f" (full aggregate traced: {traced_s}\\,s)"
    traced_note += "."

    extra_hlines = [5] if extra_arms else None
    out = header(directory, commit, machine)
    out += table_begin(
        "tab:ablation",
        "Ablation of the $\\GFt$ side for ML-DSA-44.",
        f"$N={fmt_n_tex(n)}$, {LABEL[tier]}, {alloc}, one job. Prove in s, verify in ms, keys not cached. "
        f"{traced_note} Public bits: with the public instance.",
        "X[2.6,l] X[r] X[r] X[r]",
        header_rows=1,
        extra_hlines=extra_hlines,
    )
    out += "Variant & Prove & Verify & B/sig \\\\\n"

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

    for arm_label, cell in extra_arms:
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
    tier = preferred_tier(standard)
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
        f"{LABEL[tier]} tier on {alloc}. Prove in seconds; verify in ms, uncached / cached keys; peak in GiB.",
        "Q[l] Q[r] X[r] X[r] X[1.3,r] X[r]",
        header_rows=1,
        extra_hlines=extra_hlines,
    )
    out += "Set & $N$ & B/sig & Prove & Verify & Peak \\\\\n"
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
                bps = prove = verify = peak = "--"
            out += f"{set_col} & {fmt_pow2_or_n(n)} & {bps} & {prove} & {verify} & {peak} \\\\\n"
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
    n = max(c["n"] for c in standard)
    sets = [p for p in PARAM_SETS if any(cell_params(c) == p for c in standard)] or list(PARAM_SETS)
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


def baselines_table(cells, directory, commit, machine, skipped):
    """Generate `tab-baselines.tex` (`\\label{tab:baselines}`): native, Aerie and ours at 16C32T and 1T."""
    standard = [c for c in cells if is_standard_ours(c) and c.get("relation") == DEFAULT_RELATION and c.get("times_ms")]
    aerie = [c for c in cells if c.get("system") == "aerie"]
    native = [c for c in cells if c.get("system") == "native"]
    if not standard and not aerie and not native:
        return "% no baseline data\n"

    all_cells = standard + aerie + native
    n = max(c["n"] for c in all_cells)
    par_alloc = "16C32T" if any(c.get("allocation") == "16C32T" for c in all_cells) else "8C16T"
    tier = preferred_tier(standard)
    sets = [p for p in PARAM_SETS if any(cell_params(c) == p for c in standard + native)] or list(PARAM_SETS)

    def pick_native(params, alloc):
        matches = [
            c
            for c in native
            if cell_params(c) == params and c.get("allocation") == alloc and c.get("n") == n
        ]
        if not matches:
            matches = [c for c in native if cell_params(c) == params and c.get("allocation") == alloc]
        return max(matches, key=lambda c: c.get("reps", 0)) if matches else None

    def pick_aerie(deg, alloc, avx2=False):
        matches = [
            c
            for c in aerie
            if falcon_degree(c) == deg
            and is_aerie_avx2(c) == avx2
            and c.get("allocation") == alloc
            and c.get("n") == n
        ]
        if not matches:
            matches = [
                c
                for c in aerie
                if falcon_degree(c) == deg and is_aerie_avx2(c) == avx2 and c.get("allocation") == alloc
            ]
        return max(matches, key=lambda c: c.get("reps", 0)) if matches else None

    aerie_variants = [(512, False), (1024, False)]
    for deg in (512, 1024):
        if any(falcon_degree(c) == deg and is_aerie_avx2(c) for c in aerie):
            aerie_variants.append((deg, True))

    native_end = 3 + len(sets)
    aerie_end = native_end + len(aerie_variants)
    out = header(directory, commit, machine)
    out += table_begin(
        "tab:baselines",
        f"Baselines at $N={fmt_n_tex(n)}$.",
        "Cat.: NIST category. Verify without cached keys. "
        "Native: AVX2 reference code, parallel over signatures. "
        "Aerie: Falcon-512/1024, prove without setup, bytes with salts. Keys and messages excluded.",
        "X[1.9,l] Q[c] X[1.2,r] X[1.15,r] X[1.25,r] X[1.1,r] X[1.1,r]",
        header_rows=2,
        extra_hlines=[native_end, aerie_end],
    )
    out += (
        "\\SetCell[r=2]{l} System & \\SetCell[r=2]{c} Cat. & "
        "\\SetCell[c=2]{c} Prove (s) & & \\SetCell[c=2]{c} Verify (ms) & & "
        "\\SetCell[r=2]{r} B/sig \\\\\n"
    )
    out += f" & & {par_alloc} & 1T & {par_alloc} & 1T & \\\\\n"

    for params in sets:
        constants = PARAM_SETS[params]
        c_par = pick_native(params, par_alloc)
        c_1t = pick_native(params, "1T")
        v_par = fmt_ms(median(c_par["verify_uncached_ms"])) if c_par and c_par.get("verify_uncached_ms") else "--"
        v_1t = fmt_ms(median(c_1t["verify_uncached_ms"])) if c_1t and c_1t.get("verify_uncached_ms") else "--"
        sig_b = fmt_n_tex(constants["sig_bytes"])
        out += f"native {params} & {constants['category']} & -- & -- & {v_par} & {v_1t} & {sig_b} \\\\\n"

    for deg, avx2 in aerie_variants:
        cat = falcon_category(deg)
        suffix = " (AVX2)" if avx2 else ""
        c_par = pick_aerie(deg, par_alloc, avx2=avx2)
        c_1t = pick_aerie(deg, "1T", avx2=avx2)
        p_par = fmt_seconds(median(c_par["times_ms"])) if c_par and c_par.get("times_ms") else "--"
        p_1t = fmt_seconds(median(c_1t["times_ms"])) if c_1t and c_1t.get("times_ms") else "--"
        v_par = (
            fmt_ms(median(c_par["verify_uncached_ms"]), allow_decimal=True)
            if c_par and c_par.get("verify_uncached_ms")
            else "--"
        )
        v_1t_samples = (
            (c_1t.get("verify_single_ms") or c_1t.get("verify_uncached_ms")) if c_1t else None
        )
        v_1t = fmt_ms(median(v_1t_samples), allow_decimal=True) if v_1t_samples else "--"
        ref = c_par or c_1t
        bps = f"{ref['aggregate_bytes'] / ref['n']:.1f}" if ref and ref.get("aggregate_bytes") else "--"
        out += f"Aerie-{deg}{suffix} & {cat} & {p_par} & {p_1t} & {v_par} & {v_1t} & {bps} \\\\\n"

    for params in sets:
        constants = PARAM_SETS[params]
        c_par = pick_ours_cell(
            standard, params=params, config=tier, allocation=par_alloc, n=n, relation=DEFAULT_RELATION, prefer_tag="headline"
        )
        c_1t = pick_ours_cell(
            standard, params=params, config=tier, allocation="1T", n=n, relation=DEFAULT_RELATION, prefer_tag="headline"
        )
        p_par = fmt_seconds(median(c_par["times_ms"])) if c_par and c_par.get("times_ms") else "--"
        p_1t = fmt_seconds(median(c_1t["times_ms"])) if c_1t and c_1t.get("times_ms") else "--"
        v_par = fmt_ms(median(c_par["verify_uncached_ms"])) if c_par and c_par.get("verify_uncached_ms") else "--"
        v_1t = fmt_ms(median(c_1t["verify_uncached_ms"])) if c_1t and c_1t.get("verify_uncached_ms") else "--"
        ref = c_par or c_1t
        bps = fmt_bytes_per_sig(ref)
        out += f"ours, {params} & {constants['category']} & {p_par} & {p_1t} & {v_par} & {v_1t} & {bps} \\\\\n"

    out += "\\end{talltblr}\n\\end{table}\n"
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
        mb_cell = next((c for c in mb_cells if c.get("config") == "ifma"), mb_cells[0])
        values["ProveModeB"] = fmt_seconds(median(mb_cell["times_ms"]))
        if mb_cell.get("verify_uncached_ms"):
            values["VerifyModeBUncached"] = fmt_ms(median(mb_cell["verify_uncached_ms"]))
        if mb_cell.get("aggregate_bytes"):
            values["BytesPerSigModeB"] = fmt_bytes_per_sig(mb_cell)

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
    args = parser.parse_args()
    directory = result_dir(args.results)
    cells = load_cells(directory)
    skipped = skipped_configs(directory)
    commit = next(
        (c.get("commit") for c in cells if c.get("system") == "ours" and c.get("commit")),
        "unknown",
    )
    machine = next((c.get("machine") for c in cells if c.get("machine")), "unknown")
    outputs = {
        "tab-prove.tex": prove_table(cells, directory, commit, machine, skipped),
        "tab-ablation.tex": ablation_table(cells, directory, commit, machine, skipped),
        "tab-scaling.tex": scaling_table(cells, directory, commit, machine, skipped),
        "tab-bytes.tex": bytes_table(cells, directory, commit, machine),
        "tab-baselines.tex": baselines_table(cells, directory, commit, machine, skipped),
        "tab-relations.tex": relations_table(cells, directory, commit, machine, skipped),
        "numbers.tex": numbers(cells, directory, commit, machine),
    }
    out_dir = Path(args.out) if args.out else (directory / "paper-tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in outputs.items():
        (out_dir / name).write_text(text, encoding="utf-8")
    if not args.out:
        TABLES.mkdir(parents=True, exist_ok=True)
        for name, text in outputs.items():
            (TABLES / name).write_text(text, encoding="utf-8")
    claims_csv(cells, directory)
    print(f"tables: {out_dir}")
    for name in outputs:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
