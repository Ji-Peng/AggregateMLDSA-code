#!/usr/bin/env bash
# Run the workspace tests on CPUs 0,1,3-15 (CPU 2 is reserved; see AGENTS.md).
# Usage: scripts/test.sh [extra cargo test args...]   e.g. scripts/test.sh --release -p agg-witness
set -euo pipefail
cd "$(dirname "$0")/.."
export RUSTUP_TOOLCHAIN="${RUSTUP_TOOLCHAIN:-stable}"
exec taskset -c 0,1,3-15 cargo test --workspace "$@"
