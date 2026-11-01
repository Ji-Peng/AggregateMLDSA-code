#!/usr/bin/env bash
# Correctness gate for the unified build: 1K ML-DSA-44 aggregates under H-u and H-exact
# must verify, a tampered aggregate must be rejected, and a 3-record batch must verify
# under H-u and H-FIPS; for ML-DSA-65 and ML-DSA-87 (H-u only) the 1K aggregate must
# verify, a tampered one must be rejected, and a 20-record batch must verify (lane-set
# batches hold at least 17 records). Reports bytes per signature. Proof-byte identity
# against a reference is checked separately (artifact/run.sh gates, keyed by set).
# Usage: gate.sh            build + 1K round trips + tamper checks for GATE_PARAMS
#                           (default "44 65 87")
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
for p in ${GATE_PARAMS:-44 65 87}; do
    # Record files of ML-DSA-44 keep their historical names.
    tag=$([[ "$p" == 44 ]] && echo "" || echo ".$p")
    rec="$O/rec1k$tag.bin"
    if [[ ! -f "$rec" ]]; then
        "$A" keygen-batch --params "$p" --count 1024 --seed 7 --out "$O/keys1k$tag.bin" > /dev/null
        "$A" sign-batch --params "$p" --keys "$O/keys1k$tag.bin" --count 1024 --seed 7 --out "$rec" > /dev/null
    fi
    relations=$([[ "$p" == 44 ]] && echo "h-u h-exact" || echo "h-u")
    for rel in $relations; do
        out="$O/agg1k$tag.$rel.bin"
        "$A" aggregate --params "$p" --records "$rec" --out "$out" --relation "$rel"
        "$A" verify --params "$p" --records "$rec" --aggregate "$out" --relation "$rel" > /dev/null
        sz=$(stat -c %s "$out")
        echo "GATE ML-DSA-$p verify $rel $sz B = $(awk -v b="$sz" 'BEGIN{printf "%.1f", b/1024}') B/sig at 1K"
        python3 - "$out" "$out.bad" <<'PY'
import sys
b = bytearray(open(sys.argv[1], 'rb').read()); b[len(b) // 2] ^= 1
open(sys.argv[2], 'wb').write(b)
PY
        if "$A" verify --params "$p" --records "$rec" --aggregate "$out.bad" --relation "$rel" > /dev/null 2>&1; then
            echo "GATE FAIL tampered aggregate accepted (ML-DSA-$p, $rel)"; exit 1
        fi
        echo "GATE ML-DSA-$p tamper-rejected $rel"
    done
    if [[ "$p" == 44 ]]; then
        # A batch of three records (d_main = 13: stage 3's level groups on the full domain).
        "$A" keygen-batch --count 4 --seed 11 --out "$O/keys4.bin" > /dev/null
        "$A" sign-batch --keys "$O/keys4.bin" --count 3 --seed 11 --out "$O/rec3.bin" > /dev/null
        for rel in h-u h-fips; do
            "$A" aggregate --records "$O/rec3.bin" --out "$O/agg3.$rel.bin" --relation "$rel" > /dev/null
            "$A" verify --records "$O/rec3.bin" --aggregate "$O/agg3.$rel.bin" --relation "$rel" > /dev/null
            echo "GATE verify $rel N=3 $(stat -c %s "$O/agg3.$rel.bin") B"
        done
    else
        # The smallest lane-set batch class (17 to 32 records: d_main = 13).
        "$A" keygen-batch --params "$p" --count 4 --seed 11 --out "$O/keys4$tag.bin" > /dev/null
        "$A" sign-batch --params "$p" --keys "$O/keys4$tag.bin" --count 20 --seed 11 --out "$O/rec20$tag.bin" > /dev/null
        "$A" aggregate --params "$p" --records "$O/rec20$tag.bin" --out "$O/agg20$tag.bin" > /dev/null
        "$A" verify --params "$p" --records "$O/rec20$tag.bin" --aggregate "$O/agg20$tag.bin" > /dev/null
        echo "GATE ML-DSA-$p verify h-u N=20 $(stat -c %s "$O/agg20$tag.bin") B"
    fi
done
if [[ "${1:-}" == time ]]; then
    AGG_GIT_REV="$(git rev-parse HEAD)$(git diff --quiet || echo -dirty)" \
        "$A" bench --count "${2:-1024}" --keys 16 --warmup 1 --reps 2
fi
echo "GATE PASS $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (dirty)')"
