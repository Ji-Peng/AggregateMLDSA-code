#!/usr/bin/env bash
# Correctness gate for the unified build: 1K aggregates under H-u and H-exact must
# verify, a tampered aggregate must be rejected, and a 3-record batch must verify
# under H-u and H-FIPS. Reports bytes per signature. Proof-byte identity against a
# reference is checked separately (the per-step gate against wide-ref.bin).
# Usage: gate.sh            build + 1K round trips + tamper check
#        gate.sh time N     also time `agg bench --count N --warmup 1 --reps 2` (N <= 4096 here)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
O="${GATE_DIR:-$ROOT/target/gate}"
mkdir -p "$O"
cd "$ROOT"
export RUSTUP_TOOLCHAIN="${RUSTUP_TOOLCHAIN:-stable}"
export RUSTFLAGS="-C target-cpu=x86-64-v3 -C target-feature=+pclmulqdq"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target/gate}"
echo "RUSTFLAGS=$RUSTFLAGS"
cargo build --release -p agg-cli
A="$CARGO_TARGET_DIR/release/agg"
if [[ ! -f "$O/rec1k.bin" ]]; then
    "$A" keygen-batch --count 1024 --seed 7 --out "$O/keys1k.bin" > /dev/null
    "$A" sign-batch --keys "$O/keys1k.bin" --count 1024 --seed 7 --out "$O/rec1k.bin" > /dev/null
fi
for rel in h-u h-exact; do
    out="$O/agg1k.$rel.bin"
    "$A" aggregate --records "$O/rec1k.bin" --out "$out" --relation "$rel"
    "$A" verify --records "$O/rec1k.bin" --aggregate "$out" --relation "$rel" > /dev/null
    sz=$(stat -c %s "$out")
    echo "GATE verify $rel $sz B = $(awk -v b="$sz" 'BEGIN{printf "%.1f", b/1024}') B/sig at 1K"
    python3 - "$out" "$out.bad" <<'PY'
import sys
b = bytearray(open(sys.argv[1], 'rb').read()); b[len(b) // 2] ^= 1
open(sys.argv[2], 'wb').write(b)
PY
    if "$A" verify --records "$O/rec1k.bin" --aggregate "$out.bad" --relation "$rel" > /dev/null 2>&1; then
        echo "GATE FAIL tampered aggregate accepted ($rel)"; exit 1
    fi
    echo "GATE tamper-rejected $rel"
done
# A batch of three records (d_main = 13: stage 3's level groups on the full domain).
"$A" keygen-batch --count 4 --seed 11 --out "$O/keys4.bin" > /dev/null
"$A" sign-batch --keys "$O/keys4.bin" --count 3 --seed 11 --out "$O/rec3.bin" > /dev/null
for rel in h-u h-fips; do
    "$A" aggregate --records "$O/rec3.bin" --out "$O/agg3.$rel.bin" --relation "$rel" > /dev/null
    "$A" verify --records "$O/rec3.bin" --aggregate "$O/agg3.$rel.bin" --relation "$rel" > /dev/null
    echo "GATE verify $rel N=3 $(stat -c %s "$O/agg3.$rel.bin") B"
done
if [[ "${1:-}" == time ]]; then
    AGG_GIT_REV="$(git rev-parse HEAD)$(git diff --quiet || echo -dirty)" \
        "$A" bench --count "${2:-1024}" --keys 16 --warmup 1 --reps 2
fi
echo "GATE PASS $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (dirty)')"
