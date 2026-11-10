#!/usr/bin/env bash
# Run the workspace tests (pinned with taskset to $AGG_CPUS when it is set), then the lib
# tests of the vendored flock-core, which the workspace excludes (Cargo.toml `exclude`) and
# whose patched verifier the Keccak side depends on.
# Usage: scripts/test.sh [extra cargo test args...]   e.g. scripts/test.sh --release -p agg-witness
# On a full run the tests of agg-simd, agg-mldsa-simd and agg-keccak are then repeated under
# AGG_ISA=avx512, avx2 and scalar (about half a minute in release). The vendored lib tests
# run in their own target directory ($FLOCK_CORE_TARGET_DIR, default target/flock-core),
# with --release if the arguments contain it; they are skipped when the arguments select
# packages (-p/--package) or when SKIP_FLOCK_CORE is set. Flock's three m = 20 proof-size
# printouts are skipped (`estimator_matches_actual_m20` proves and verifies
# one of their shapes).
set -euo pipefail
cd "$(dirname "$0")/.."
# The toolchain comes from rust-toolchain.toml.

release=()
vendored=1
ws=(--workspace)
for arg in "$@"; do
  case "$arg" in
    --release) release=(--release) ;;
    -p | -p* | --package | --package=*) vendored=0; ws=() ;;
  esac
done
[[ -n "${SKIP_FLOCK_CORE:-}" ]] && vendored=0
pin=()
[[ -n "${AGG_CPUS:-}" ]] && pin=(taskset -c "$AGG_CPUS")

"${pin[@]}" cargo test "${ws[@]}" "$@"

# The kernel crates pick their SIMD tier once per process, so the workspace run above tests
# only the highest tier of this CPU. Rerun their tests under each lower AGG_ISA cap (a cap
# above the hardware falls back to what the CPU has). Skipped when the arguments select
# packages or when SKIP_ISA_TIERS is set.
if [[ ${#ws[@]} -gt 0 && -z "${SKIP_ISA_TIERS:-}" ]]; then
  for isa in avx512 avx2 scalar; do
    echo "== AGG_ISA=$isa: agg-simd, agg-mldsa-simd, agg-keccak"
    AGG_ISA="$isa" "${pin[@]}" cargo test "${release[@]}" -p agg-simd -p agg-mldsa-simd -p agg-keccak
  done
fi

if [[ "$vendored" == 1 ]]; then
  CARGO_TARGET_DIR="${FLOCK_CORE_TARGET_DIR:-target/flock-core}" \
    "${pin[@]}" cargo test "${release[@]}" \
    --manifest-path vendor/flock-core/Cargo.toml --lib -- --skip ligerito_size_breakdown_m20
fi
