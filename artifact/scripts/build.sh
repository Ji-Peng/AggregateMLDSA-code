#!/usr/bin/env bash
# Build the three binaries of ours, Aerie's baseline and natver.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-6}"
export RUSTUP_TOOLCHAIN="${RUSTUP_TOOLCHAIN:-1.99.1}"
exec python3 "$here/build.py" "$@"
