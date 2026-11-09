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
#   ./run.sh alloul26  run the optional Alloul26 comparison
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
  sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
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
    python3 scripts/gen_tables.py || exit 1
    if (( alloul26 )); then bash scripts/alloul26.sh; fi
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
    python3 scripts/gen_tables.py || exit 1
    if (( alloul26 )); then bash scripts/alloul26.sh; fi
    ;;
  -h | --help | help)
    usage
    ;;
  *)
    echo "unknown mode: $mode" >&2
    usage
    ;;
esac
