#!/usr/bin/env bash
# Run one configuration of ours through the quick matrix.
#
# usage: run_ours.sh CONFIG ALLOCATION N [REPS]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
config=${1:?tier: ifma, avx512 or avx2}
allocation=${2:?allocation: 16C32T, 8C16T or 1T}
n=${3:?batch size}
reps=${4:-3}
exec python3 "$here/runner.py" quick \
  --only quick-ours \
  --configs "$config" \
  --allocations "$allocation" \
  --sizes "$n" \
  --reps "$reps"
