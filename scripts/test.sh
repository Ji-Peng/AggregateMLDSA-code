#!/usr/bin/env bash
# Run the workspace tests on CPUs 0,1,3-15 (CPU 2 is reserved), then the lib
# tests of the vendored flock-core, which the workspace excludes (Cargo.toml `exclude`) and
# whose patched verifier the Keccak side depends on (vendor/flock-core/PATCH.md).
# Usage: scripts/test.sh [extra cargo test args...]   e.g. scripts/test.sh --release -p agg-witness
# The vendored lib tests run in their own target directory ($FLOCK_CORE_TARGET_DIR, default
# target/flock-core), with --release if the arguments contain it; they are skipped when the
# arguments select packages (-p/--package) or when SKIP_FLOCK_CORE is set.
set -euo pipefail
cd "$(dirname "$0")/.."
export RUSTUP_TOOLCHAIN="${RUSTUP_TOOLCHAIN:-stable}"

release=()
vendored=1
for arg in "$@"; do
  case "$arg" in
    --release) release=(--release) ;;
    -p | -p* | --package | --package=*) vendored=0 ;;
  esac
done
[[ -n "${SKIP_FLOCK_CORE:-}" ]] && vendored=0

taskset -c 0,1,3-15 cargo test --workspace "$@"

if [[ "$vendored" == 1 ]]; then
  CARGO_TARGET_DIR="${FLOCK_CORE_TARGET_DIR:-target/flock-core}" \
    taskset -c 0,1,3-15 cargo test "${release[@]}" \
    --manifest-path vendor/flock-core/Cargo.toml --lib
fi
