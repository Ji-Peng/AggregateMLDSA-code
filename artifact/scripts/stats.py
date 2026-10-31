#!/usr/bin/env python3
"""Medians, extremes and order-statistic intervals from raw cells."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import median, order_statistic_interval, read_json  # noqa: E402


def describe(values):
    if not values:
        return None
    low, high, coverage, k = order_statistic_interval(values)
    med = median(values)
    return {
        "n": len(values),
        "median": med,
        "min": min(values),
        "max": max(values),
        "interval": [low, high],
        "interval_coverage": coverage,
        "interval_k": k,
        "spread": (max(values) - min(values)) / med if med else None,
    }


def flags(cell):
    out = []
    times = cell.get("times_ms") or []
    if times:
        info = describe(times)
        if info["spread"] is not None and info["spread"] > 0.05:
            out.append("noisy")
    if cell.get("foreign_busy") is not None and cell["foreign_busy"] > 0.05:
        out.append("co-tenant")
    if cell.get("rejected"):
        out.append("rejected")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    for path in args.paths:
        cell = read_json(path)
        print(
            json.dumps(
                {
                    "run_id": cell.get("run_id"),
                    "prove": describe(cell.get("times_ms") or []),
                    "verify_uncached": describe(cell.get("verify_uncached_ms") or []),
                    "verify_cached": describe(cell.get("verify_cached_ms") or []),
                    "flags": flags(cell),
                }
            )
        )


if __name__ == "__main__":
    main()
