#!/usr/bin/env bash
# Entry point of the reproduction package.
#
#   ./run.sh check    report missing tools, sources and binaries
#   ./run.sh fetch    clone and verify the pinned sources
#   ./run.sh build    build the release binary, Aerie and natver
#   ./run.sh gates    byte identity, acceptance and the negative check at 1K
#   ./run.sh quick    small matrix; add --smoke for N=1024 and 1+1 runs
#   ./run.sh full     the experiment matrix of the evaluation
#   ./run.sh tables   raw JSON -> LaTeX tables
#   ./run.sh all      fetch, build, gates, quick (or full with --full), tables
#   ./run.sh alloul26  run one Alloul26 measurement (full --alloul26: the table's cells)
#   ./run.sh freeze   maintainers: re-pin hashes.json and sources.lock to HEAD
#
# Every run uses relation H-FIPS. gates, quick, full and all take
# --relations h-fips,h-exact to add the reference relation H-exact.
set -uo pipefail

here=$(cd "$(dirname "$0")" && pwd)
cd "$here" || exit 1

export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-6}"
export RUSTUP_TOOLCHAIN="${RUSTUP_TOOLCHAIN:-1.99.1}"

usage() {
  sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}

# Run the Alloul26 comparison and keep its JSON lines in the results
# directory, where gen_tables.py reads them for tab:baselines. `quick` runs the
# 128-signature smoke case. `full` runs the cells of the table: 32,768
# signatures, family J, the AVX2 and AVX-512 builds, on the CPUs of the
# 16C32T, 8C16T and 1T allocations that the runner recorded in cpusets.json,
# ALLOUL26_RUNS (default 3) fresh processes per cell.
run_alloul26() {
  local dir=${AGG_RESULTS:-results/$(cat results/LATEST 2>/dev/null)}
  mkdir -p "$dir"
  if [ "$1" != full ]; then
    bash scripts/alloul26.sh | tee -a "$dir/alloul26.jsonl"
    return
  fi
  local isas=(avx2)
  # The AVX-512 build targets x86-64-v4.
  if grep -q avx512vl /proc/cpuinfo && grep -q avx512bw /proc/cpuinfo; then isas+=(avx512); fi
  local threads cpuset isa
  while read -r threads cpuset; do
    for isa in "${isas[@]}"; do
      bash scripts/alloul26.sh --n 32768 --family J --isa "$isa" --threads "$threads" \
        --cpus "$cpuset" --runs "${ALLOUL26_RUNS:-3}" | tee -a "$dir/alloul26.jsonl"
    done
  done < <(python3 - "$dir/cpusets.json" <<'PY'
import json, sys
configs = json.load(open(sys.argv[1], encoding="utf-8"))["configs"]
for name, threads in (("16C32T", 32), ("8C16T", 16), ("1T", 1)):
    alloc = configs.get(name, {})
    # Skip an allocation that this machine cannot provide in full.
    if alloc.get("available") and alloc.get("threads") == threads and alloc.get("cpuset"):
        print(threads, alloc["cpuset"])
PY
  )
}

mode=${1:-}
if [ -z "$mode" ]; then
  usage
fi
shift

case "$mode" in
  check)
    exec python3 scripts/probe.py --check "$@"
    ;;
  fetch)
    exec bash scripts/fetch.sh "$@"
    ;;
  build)
    exec bash scripts/build.sh "$@"
    ;;
  gates)
    exec python3 scripts/gates.py "$@"
    ;;
  quick | full)
    extra=()
    alloul26=0
    for arg in "$@"; do
      case "$arg" in --alloul26) alloul26=1 ;; *) extra+=("$arg") ;; esac
    done
    python3 scripts/runner.py "$mode" "${extra[@]}" || exit 1
    for arg in "$@"; do
      case "$arg" in
        --dry-run | -n) exit 0 ;;
      esac
    done
    if (( alloul26 )); then run_alloul26 "$mode"; fi
    python3 scripts/gen_tables.py || exit 1
    ;;
  alloul26)
    exec bash scripts/alloul26.sh "$@"
    ;;
  tables)
    exec python3 scripts/gen_tables.py "$@"
    ;;
  freeze)
    exec python3 scripts/freeze.py "$@"
    ;;
  all)
    run_mode=quick
    alloul26=0
    extra=()
    relations=()
    while [ $# -gt 0 ]; do
      case "$1" in
        --full) run_mode=full ;;
        --alloul26) alloul26=1 ;;
        --relations)
          relations=(--relations "${2:-}")
          extra+=("$1" "${2:-}")
          shift
          ;;
        --relations=*)
          relations=("$1")
          extra+=("$1")
          ;;
        *) extra+=("$1") ;;
      esac
      shift
    done
    bash scripts/fetch.sh || exit 1
    bash scripts/build.sh || exit 1
    python3 scripts/gates.py ${relations[@]+"${relations[@]}"} || exit 1
    python3 scripts/runner.py "$run_mode" "${extra[@]}" || exit 1
    if (( alloul26 )); then run_alloul26 "$run_mode"; fi
    python3 scripts/gen_tables.py || exit 1
    ;;
  -h | --help | help)
    usage
    ;;
  *)
    echo "unknown mode: $mode" >&2
    usage
    ;;
esac
