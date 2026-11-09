#!/usr/bin/env bash
# Run the agg-bench benchmarks, pinned with taskset to $AGG_CPUS when it is set.
# Usage: scripts/bench.sh [extra cargo bench args...]
#   The build takes its flags from `.cargo/config.toml` (x86-64-v3 plus
#   PCLMULQDQ); higher ISA levels are selected at run time and capped by `AGG_ISA`.
# Benchmarks report medians; record `git rev-parse HEAD` with every result.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "git revision: $(git rev-parse HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo ' (dirty)')"
pin=()
[[ -n "${AGG_CPUS:-}" ]] && pin=(taskset -c "$AGG_CPUS")
exec "${pin[@]}" cargo bench -p agg-bench "$@"
