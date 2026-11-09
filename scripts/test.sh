#!/usr/bin/env bash
# Run the workspace tests (pinned with taskset to $AGG_CPUS when it is set), then the lib
# tests of the vendored flock-core, which the workspace excludes (Cargo.toml `exclude`) and
# whose patched verifier the Keccak side depends on.
# Usage: scripts/test.sh [extra cargo test args...]   e.g. scripts/test.sh --release -p agg-witness
# The vendored lib tests run in their own target directory ($FLOCK_CORE_TARGET_DIR, default
# target/flock-core), with --release if the arguments contain it; they are skipped when the
# arguments select packages (-p/--package) or when SKIP_FLOCK_CORE is set. Flock's three
# m = 20 proof-size printouts are skipped (`estimator_matches_actual_m20` proves and verifies
# one of their shapes).
set -euo pipefail
cd "$(dirname "$0")/.."
# The toolchain comes from rust-toolchain.toml.

release=()
vendored=1
for arg in "$@"; do
  case "$arg" in
    --release) release=(--release) ;;
    -p | -p* | --package | --package=*) vendored=0 ;;
  esac
done
[[ -n "${SKIP_FLOCK_CORE:-}" ]] && vendored=0
pin=()
[[ -n "${AGG_CPUS:-}" ]] && pin=(taskset -c "$AGG_CPUS")

"${pin[@]}" cargo test --workspace "$@"

if [[ "$vendored" == 1 ]]; then
  CARGO_TARGET_DIR="${FLOCK_CORE_TARGET_DIR:-target/flock-core}" \
    "${pin[@]}" cargo test "${release[@]}" \
    --manifest-path vendor/flock-core/Cargo.toml --lib -- --skip ligerito_size_breakdown_m20
fi
