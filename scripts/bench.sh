#!/usr/bin/env bash
# Run benchmarks on CPUs 0,1,3-15 (local calibration only; CPU 2 is reserved
# for another task, and large runs go to the cloud machine).
# Usage: scripts/bench.sh [extra cargo bench args...]
#   CLMUL=1 scripts/bench.sh    builds with `-C target-feature=+pclmulqdq` (Flock's
#   carry-less GF(2^128) kernels) in target/clmul; Akita stays at baseline codegen.
#   AVX512=1 scripts/bench.sh   adds the `avx512` feature and x86-64-v4 (VPCLMULQDQ);
#   BROKEN for the Akita side, see README "SIMD policy".
# Benchmarks report medians; record `git rev-parse HEAD` with every result.
set -euo pipefail
cd "$(dirname "$0")/.."
export RUSTUP_TOOLCHAIN="${RUSTUP_TOOLCHAIN:-stable}"
echo "git revision: $(git rev-parse HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo ' (dirty)')"
if [[ "${CLMUL:-0}" == "1" ]]; then
  export RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+pclmulqdq"
  export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target/clmul}"
  exec taskset -c 0,1,3-15 cargo bench -p agg-bench "$@"
fi
if [[ "${AVX512:-0}" == "1" ]]; then
  # x86-64-v4 = AVX-512 F/BW/CD/DQ/VL. A bare `+avx512f,...` feature list makes
  # rustc 1.99.1 (LLVM 23) crash with SIGILL while compiling aerie-prover.
  # This is a separate, non-shipped target: shipped builds use the one baseline in
  # `.cargo/config.toml` and select every higher level at run time. Kept for
  # codegen A/B measurements and the Keccak-only bench.
  export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=x86-64-v4 -C target-feature=+vpclmulqdq,+pclmulqdq"
  export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target/avx512}"
  exec taskset -c 0,1,3-15 cargo bench -p agg-bench --features avx512 "$@"
fi
exec taskset -c 0,1,3-15 cargo bench -p agg-bench "$@"
