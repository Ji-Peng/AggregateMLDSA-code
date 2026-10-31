#!/usr/bin/env bash
# Run Aerie's Falcon-512 baseline in the quick matrix.
#
# usage: run_aerie.sh N [REPS] [ALLOCATION] [auto|avx2]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
n=${1:?batch size}
reps=${2:-2}
allocation=${3:-8C16T}
backend=${4:-auto}
exec python3 "$here/runner.py" quick \
  --only quick-aerie \
  --allocations "$allocation" \
  --sizes "$n" \
  --reps "$reps" \
  --aerie-backend "$backend"
