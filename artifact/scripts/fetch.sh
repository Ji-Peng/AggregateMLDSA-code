#!/usr/bin/env bash
# Fetch the pinned external sources and warm the cargo caches.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
exec python3 "$here/fetch.py" "$@"
