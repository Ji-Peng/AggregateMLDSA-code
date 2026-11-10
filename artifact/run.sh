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
# directory, where gen_tables.py reads them for tab:baselines. `quick` runs
# 1,024 signatures, family J, the AVX2 build, one process on each of the 8C16T
# and 1T allocations that quick measures. `full` runs the cells of the table:
# 32,768 signatures, family J, the AVX2 and AVX-512 builds, on the 16C32T,
# 8C16T and 1T allocations, ALLOUL26_RUNS (default 3) fresh processes per
# cell. CPU sets come from cpusets.json. A cell is skipped (and recorded in
# skipped.json) when MemAvailable is below its peak plus headroom; a failed
# cell stops the run with a non-zero status.
run_alloul26() {
  local dir=${AGG_RESULTS:-results/$(cat results/LATEST 2>/dev/null)}
  mkdir -p "$dir"
  local n=32768 runs=${ALLOUL26_RUNS:-3} need_gib=44
  local allocs=(16C32T:32 8C16T:16 1T:1)
  local isas=(avx2)
  if [ "$1" = full ]; then
    # The AVX-512 build targets x86-64-v4.
    if grep -q avx512vl /proc/cpuinfo && grep -q avx512bw /proc/cpuinfo; then isas+=(avx512); fi
  else
    # Family J peaks at about 4.7 GiB at 1,024 signatures.
    n=1024 runs=${ALLOUL26_RUNS:-1} need_gib=8 allocs=(8C16T:16 1T:1)
  fi
  local threads cpuset isa avail_kib
  while read -r threads cpuset; do
    for isa in "${isas[@]}"; do
      avail_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
      if (( ${avail_kib:-0} < need_gib * 1024 * 1024 )); then
        echo "alloul26: skip N=$n $isa on $threads threads: MemAvailable $(( ${avail_kib:-0} / 1048576 )) GiB < $need_gib GiB" >&2
        python3 - "$dir/skipped.json" "$n" "$isa" "$threads" "$need_gib" "${avail_kib:-0}" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
try:
    entries = json.loads(path.read_text(encoding="utf-8"))
except (OSError, ValueError):
    entries = []
n, isa, threads, need, avail = sys.argv[2:7]
entries.append({"kind": "alloul26", "name": f"J-{n}-{isa}-t{threads}", "tag": "alloul26",
                "reason": f"MemAvailable {int(avail) / 1048576:.1f} GiB, {need} GiB needed"})
path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
PY
        continue
      fi
      bash scripts/alloul26.sh --n "$n" --family J --isa "$isa" --threads "$threads" \
        --cpus "$cpuset" --runs "$runs" | tee -a "$dir/alloul26.jsonl" || {
        echo "alloul26: N=$n $isa on $threads threads failed" >&2
        return 1
      }
    done
  done < <(python3 - "$dir/cpusets.json" "${allocs[@]}" <<'PY'
import json, sys
configs = json.load(open(sys.argv[1], encoding="utf-8"))["configs"]
for item in sys.argv[2:]:
    name, threads = item.split(":")
    alloc = configs.get(name, {})
    # Skip an allocation that this machine cannot provide in full.
    if alloc.get("available") and alloc.get("threads") == int(threads) and alloc.get("cpuset"):
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
    if (( alloul26 )); then run_alloul26 "$mode" || exit 1; fi
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
    fetch_args=()
    if (( alloul26 )); then fetch_args=(--alloul26); fi
    bash scripts/fetch.sh ${fetch_args[@]+"${fetch_args[@]}"} || exit 1
    bash scripts/build.sh || exit 1
    python3 scripts/gates.py ${relations[@]+"${relations[@]}"} || exit 1
    python3 scripts/runner.py "$run_mode" "${extra[@]}" || exit 1
    if (( alloul26 )); then run_alloul26 "$run_mode" || exit 1; fi
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
