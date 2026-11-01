#!/usr/bin/env bash
# Run benchmarks on CPUs 0,1,3-15 (local calibration only; CPU 2 is reserved
# for another task, and large runs go to xcni / c4d).
# Usage: scripts/bench.sh [extra cargo bench args...]
#   The build takes its flags from `.cargo/config.toml` (x86-64-v3 plus
#   PCLMULQDQ); higher ISA levels are selected at run time and capped by `AGG_ISA`.
# Benchmarks report medians; record `git rev-parse HEAD` with every result.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "git revision: $(git rev-parse HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo ' (dirty)')"
exec taskset -c 0,1,3-15 cargo bench -p agg-bench "$@"
