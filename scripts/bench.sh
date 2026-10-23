#!/usr/bin/env bash
# Run benchmarks on CPUs 0,1,3-15 (local calibration only; CPU 2 is reserved
# for another task, and large runs go to the cloud machine).
# Usage: scripts/bench.sh [extra cargo bench args...]
#   AVX512=1 scripts/bench.sh   builds with the `avx512` feature and Flock's
#   compile-time AVX-512/VPCLMULQDQ kernels in a separate target dir.
# Benchmarks report medians; record `git rev-parse HEAD` with every result.
set -euo pipefail
cd "$(dirname "$0")/.."
export RUSTUP_TOOLCHAIN="${RUSTUP_TOOLCHAIN:-stable}"
echo "git revision: $(git rev-parse HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo ' (dirty)')"
if [[ "${AVX512:-0}" == "1" ]]; then
  # x86-64-v4 = AVX-512 F/BW/CD/DQ/VL. A bare `+avx512f,...` feature list made
  # rustc 1.99.1 (LLVM 23) crash with SIGILL while compiling aerie-prover.
  export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=x86-64-v4 -C target-feature=+vpclmulqdq,+pclmulqdq"
  export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target/avx512}"
  exec taskset -c 0,1,3-15 cargo bench -p agg-bench --features avx512 "$@"
fi
exec taskset -c 0,1,3-15 cargo bench -p agg-bench "$@"
