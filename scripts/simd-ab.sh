#!/usr/bin/env bash
# Local CLMUL A/B against main. Logs and detached baseline stay under target/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CARGO_BUILD_JOBS=3 RUSTFLAGS='-C target-feature=+pclmulqdq'
unset RAYON_NUM_THREADS AGG_MLDSA_SIMD
MAIN_WT="$ROOT/target/main-wt"
export LOG_DIR="$ROOT/target/simd-ab-logs"
mkdir -p "$LOG_DIR"
TMP_DIR="$(mktemp -d "$ROOT/target/simd-ab.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT
if [[ ! -d "$MAIN_WT" ]]; then
    git worktree add --detach "$MAIN_WT" main
fi
[[ "$(git -C "$MAIN_WT" rev-parse HEAD)" == "$(git rev-parse main)" ]] || {
    echo 'Baseline worktree differs from main; inspect target/main-wt before rerunning.' >&2
    exit 1
}
(cd "$MAIN_WT" && CARGO_TARGET_DIR="$MAIN_WT/target" cargo build --release -p agg-cli --bin agg)
cargo build --release -p agg-cli --bin agg
MAIN_BIN="$MAIN_WT/target/release/agg"
CURRENT_BIN="$ROOT/target/release/agg"
git rev-parse main HEAD > "$LOG_DIR/revisions.txt"
# Detect the same feature conjunctions as Engine::new, including OS-enabled flags.
BACKENDS=(scalar)
flags=" $(awk '/^flags/{print; exit}' /proc/cpuinfo) "
if [[ "$flags" == *' avx2 '* && "$flags" == *' popcnt '* ]]; then
    BACKENDS+=(avx2)
    if [[ "$flags" == *' avx512f '* && "$flags" == *' avx512bw '* && "$flags" == *' avx512vl '* ]]; then
        BACKENDS+=(avx512)
    fi
fi
export AB_BACKENDS="${BACKENDS[*]}"
# Every runtime gate shares the machine-wide benchmark lock and CPU affinity.
gate() { flock /home/jipengzhang/agy_out/mldsa-simd/bench.lock taskset -c 3 "$@"; }
gate "$CURRENT_BIN" keygen-batch --count 64 --seed 42 --out "$TMP_DIR/keys.bin"
gate "$CURRENT_BIN" sign-batch --keys "$TMP_DIR/keys.bin" --count 1024 --seed 42 --out "$TMP_DIR/records.bin"
gate "$MAIN_BIN" aggregate --records "$TMP_DIR/records.bin" --out "$TMP_DIR/main.bin" --relation h-u
for b in "${BACKENDS[@]}"; do
    for threads in 1 16; do
        gate env AGG_MLDSA_SIMD="$b" RAYON_NUM_THREADS="$threads" "$CURRENT_BIN" aggregate --records "$TMP_DIR/records.bin" --out "$TMP_DIR/current.bin" --relation h-u
        cmp "$TMP_DIR/main.bin" "$TMP_DIR/current.bin"
        echo "IDENTICAL main/$b threads=$threads" | tee -a "$LOG_DIR/equality.log"
    done
done
for keys in 4096 16; do
    gate env AGG_GIT_REV="$(git rev-parse main)" "$MAIN_BIN" bench --count 4096 --keys "$keys" --warmup 1 --reps 2 --relation h-u > "$LOG_DIR/main_${keys}.log" 2>&1
    for b in "${BACKENDS[@]}"; do
        gate env AGG_GIT_REV="$(git rev-parse HEAD)" AGG_MLDSA_SIMD="$b" "$CURRENT_BIN" bench --count 4096 --keys "$keys" --warmup 1 --reps 2 --relation h-u > "$LOG_DIR/${b}_${keys}.log" 2>&1
    done
done
python3 - <<'PY' | tee "$LOG_DIR/table.md"
import os, re
from pathlib import Path
root = Path(os.environ['LOG_DIR'])
backends = ['main'] + os.environ['AB_BACKENDS'].split()
metrics = [
    ('instance', r'instance \(ExpandA, mu, c, T\)'),
    ('witness and tables', r'witness and G_main tables'),
    ('Flock records', r'Flock records'),
    ('prove total', r'prove'),
    ('verify uncached', r'verify, uncached keys'),
    ('verify cached', r'verify, cached keys'),
    ('public_data', r'public_data'), ('public_t', r'public_t'), ('public_a', r'public_a'),
    ('prover RSS MiB', r'prover peak RSS'), ('verifier RSS MiB', r'verifier peak RSS'),
]
for keys in (4096, 16):
    print(f'\nLocal direction only: N=4096, K={keys}, H-u, CLMUL, CPU 3, default threads. Times in ms.')
    print('| Metric | ' + ' | '.join(backends) + ' |')
    print('|---|' + '---:|' * len(backends))
    texts = [(root / f'{b}_{keys}.log').read_text() for b in backends]
    for label, pattern in metrics:
        values = []
        for text in texts:
            prefix = r'\b' if label in ('public_data', 'public_t', 'public_a') else r'^\s*'
            match = re.search(prefix + pattern + r'\s+([0-9.,]+)\s+(?:ms|MiB)', text, re.M)
            if not match:
                raise SystemExit(f'Missing metric {label}; inspect logs')
            values.append(float(match[1].replace(',', '')))
        cells = [f'{values[0]:.1f}']
        for value in values[1:]:
            ratio = value / values[0] if values[0] else 1
            flag = ' FLAG >5%' if ratio > 1.05 else ''
            cells.append(f'{value:.1f} ({ratio:.3f}x){flag}')
        print('| ' + label + ' | ' + ' | '.join(cells) + ' |')
PY
printf '%s\n' "Cloud only (do not run locally; c4d-highmem-16, each supported backend):"
for b in "${BACKENDS[@]}"; do
    for keys in 32768 16; do
        printf "RUSTFLAGS='-C target-feature=+pclmulqdq' RAYON_NUM_THREADS=16 AGG_MLDSA_SIMD=%s cargo run --release -p agg-cli --bin agg -- bench --count 32768 --keys %s --warmup 1 --reps 5 --relation h-u\n" "$b" "$keys"
    done
done
