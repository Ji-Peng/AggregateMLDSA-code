#!/usr/bin/env bash
# Build the release binary of ours, Aerie's baseline and natver.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-6}"
export RUSTUP_TOOLCHAIN="${RUSTUP_TOOLCHAIN:-1.99.1}"
if [ -z "${AGG_GIT_REV:-}" ]; then
  AGG_GIT_REV=$(grep -A 3 'name = "ours"' "$here/../sources.lock" 2>/dev/null | grep 'commit =' | head -n1 | cut -d'"' -f2 || true)
  if [ -n "$AGG_GIT_REV" ]; then
    export AGG_GIT_REV
  fi
fi
exec python3 "$here/build.py" "$@"
