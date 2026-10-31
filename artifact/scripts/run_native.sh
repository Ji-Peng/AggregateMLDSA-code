#!/usr/bin/env bash
# Run the pq-crystals ML-DSA-44 AVX2 baseline in the quick matrix.
#
# usage: run_native.sh N [REPS] [ALLOCATION]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
n=${1:?batch size}
reps=${2:-2}
allocation=${3:-8C16T}
exec python3 "$here/runner.py" quick \
  --only quick-native \
  --allocations "$allocation" \
  --sizes "$n" \
  --reps "$reps"
