#!/usr/bin/env bash
# Run Aerie's Falcon-512 or Falcon-1024 baseline in the quick matrix.
#
# usage: run_aerie.sh N [REPS] [ALLOCATION] [auto|avx2] [512|1024]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
n=${1:?batch size}
reps=${2:-2}
allocation=${3:-8C16T}
backend=${4:-auto}
degree=${5:-512}
exec python3 "$here/runner.py" quick \
  --only quick-aerie \
  --allocations "$allocation" \
  --sizes "$n" \
  --reps "$reps" \
  --aerie-backend "$backend" \
  --aerie-degree "$degree"
