#!/usr/bin/env bash
# Run benchmarks on CPUs 0,1,3-15 (local calibration only; CPU 2 is reserved
# for another task, and large runs go to the cloud machine).
# Usage: scripts/bench.sh [extra cargo bench args...]
#   CLMUL=1 scripts/bench.sh    builds with `-C target-feature=+pclmulqdq,+sse4.1` (Flock's
#   carry-less GF(2^128) kernels) and the thin-LTO cargo profile `clmul` (outputs in
#   target/clmul); LTO + sse4.1 inline Flock's scalar multiply (docs/perf.md Section 10).
#   Akita, Jolt and Aerie gain no target features but are compiled under thin LTO.
#   AVX512=1 scripts/bench.sh   adds the `avx512` feature and x86-64-v4 (VPCLMULQDQ);
#   BROKEN for the Akita side as of 2026-10-24, see README "SIMD policy".
# Benchmarks report medians; record `git rev-parse HEAD` with every result.
set -euo pipefail
cd "$(dirname "$0")/.."
export RUSTUP_TOOLCHAIN="${RUSTUP_TOOLCHAIN:-stable}"
echo "git revision: $(git rev-parse HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo ' (dirty)')"
if [[ "${CLMUL:-0}" == "1" ]]; then
  export RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+pclmulqdq,+sse4.1"
  exec taskset -c 0,1,3-15 cargo bench --profile clmul -p agg-bench "$@"
fi
if [[ "${AVX512:-0}" == "1" ]]; then
  # x86-64-v4 = AVX-512 F/BW/CD/DQ/VL. A bare `+avx512f,...` feature list made
  # rustc 1.99.1 (LLVM 23) crash with SIGILL while compiling aerie-prover.
  # With x86-64-v4, Akita's compile-time avx512bw path (windowed_accum.rs) makes its
  # prover self-check fail ("relation session final claim disagrees with its folded
  # oracle"); mode A and the Akita bench do not run. Kept for Keccak-only measurements.
  export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=x86-64-v4 -C target-feature=+vpclmulqdq,+pclmulqdq"
  export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target/avx512}"
  exec taskset -c 0,1,3-15 cargo bench -p agg-bench --features avx512 "$@"
fi
exec taskset -c 0,1,3-15 cargo bench -p agg-bench "$@"
