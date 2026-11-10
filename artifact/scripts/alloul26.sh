#!/usr/bin/env bash
# Reproduce a Geode hidden-u certificate; each measurement is a fresh process.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
usage() {
  echo 'Usage: alloul26.sh [--n N] [--family J|U] [--isa avx2|avx512] [--soundness default] [--threads T] [--cpus LIST] [--runs R]' >&2
}
n=128 family=J isa=avx2 soundness=default threads=1 runs=1 cpus=
while (($#)); do
  case "$1" in
    --n|--family|--isa|--soundness|--threads|--cpus|--runs)
      (($# >= 2)) || { usage; exit 2; }
      case "$1" in
        --n) n=$2 ;; --family) family=$2 ;; --isa) isa=$2 ;;
        --soundness) soundness=$2 ;; --threads) threads=$2 ;; --cpus) cpus=$2 ;; --runs) runs=$2 ;;
      esac
      shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done
[[ $n =~ ^[0-9]+$ && $threads =~ ^[0-9]+$ && $runs =~ ^[0-9]+$ ]] || { usage; exit 2; }
(( n >= 32 && n <= 65536 && (n & (n-1)) == 0 && threads > 0 && runs > 0 )) || { usage; exit 2; }
case "$family" in J) h9_family=johnson ;; U) h9_family=ud ;; *) usage; exit 2 ;; esac
case "$isa" in avx2) cpu=x86-64-v3 ;; avx512) cpu=x86-64-v4 ;; *) usage; exit 2 ;; esac
[[ $soundness == default ]] || { echo 'Geode has no configurable soundness profile; use default.' >&2; exit 2; }
readarray -t pin < <(python3 - "$here/sources.lock" <<'PY'
import sys
try:
    import tomllib
except ImportError:
    sys.path.insert(0, str(__import__('pathlib').Path(sys.argv[1]).parent / 'scripts'))
    import toml_compat as tomllib
with open(sys.argv[1], encoding='utf-8') as f:
    source = next(s for s in tomllib.loads(f.read())['source'] if s['name'] == 'alloul26')
print(source['url']); print(source['tag']); print(source['commit'])
PY
)
((${#pin[@]} == 3)) || { echo 'Cannot read Geode pin' >&2; exit 1; }
source_dir="$here/build/geode-${pin[1]}"
mkdir -p "$here/build"
if [[ ! -d $source_dir/.git ]]; then
  [[ ! -e $source_dir ]] || { echo "Existing non-repository path: $source_dir" >&2; exit 1; }
  git clone --quiet --branch "${pin[1]}" --single-branch "${pin[0]}" "$source_dir" >&2
fi
[[ $(git -C "$source_dir" rev-parse HEAD) == "${pin[2]}" && $(git -C "$source_dir" rev-parse "${pin[1]}^{}") == "${pin[2]}" ]] || { echo 'Geode commit does not match sources.lock' >&2; exit 1; }
[[ -z $(git -C "$source_dir" status --porcelain --untracked-files=no) ]] || { echo 'Geode source has local modifications' >&2; exit 1; }
# Geode pins Rust 1.96.0 (sources.lock); install it when rustup lacks it.
rustup which --toolchain 1.96.0 cargo >/dev/null 2>&1 || rustup toolchain install --profile minimal 1.96.0 >&2
export RUSTUP_TOOLCHAIN=1.96.0
export CARGO_TARGET_DIR="$here/build/geode-target-$isa"
export RUSTFLAGS="-C target-cpu=$cpu"
export RAYON_NUM_THREADS=$threads H9_FAMILY=$h9_family
cargo build --manifest-path "$source_dir/aggregate/r3-cert/Cargo.toml" --release --locked >&2
tmp=$(mktemp -d "$here/build/alloul26-run.XXXXXXXX")
trap 'rm -rf "$tmp"' EXIT
export R3_OUT=$tmp
# With --cpus the prover and verifier are pinned to that CPU list, so a run
# uses the same cores as the allocation it is compared with.
pin_cmd=()
[[ -z $cpus ]] || pin_cmd=(taskset -c "$cpus")
for ((run=1; run<=runs; run++)); do
  /usr/bin/time -v -o "$tmp/time" ${pin_cmd[@]+"${pin_cmd[@]}"} "$CARGO_TARGET_DIR/release/r3-cert" h9 "$n" >"$tmp/log"
  if [[ -n "${ALLOUL_LOG_DIR:-}" ]]; then
    mkdir -p "$ALLOUL_LOG_DIR"
    cp "$tmp/log" "$ALLOUL_LOG_DIR/alloul-$family-$n-$isa-run$run.log"
    cp "$tmp/time" "$ALLOUL_LOG_DIR/alloul-$family-$n-$isa-run$run.time"
    [[ -f "$tmp/h9-cert-$n.bin" ]] && cp "$tmp/h9-cert-$n.bin" "$ALLOUL_LOG_DIR/alloul-$family-$n-$isa-run$run.bin"
  fi
  python3 - "$tmp/log" "$tmp/time" "$tmp/h9-cert-$n.bin" "$family" "$n" "$threads" "$isa" "$run" "$cpus" <<'PY'
import json, pathlib, re, sys
log = pathlib.Path(sys.argv[1]).read_text()
timing = pathlib.Path(sys.argv[2]).read_text()
def match(pattern, text):
    m = re.search(pattern, text, re.MULTILINE)
    if not m:
        raise SystemExit('Missing expected Geode output: ' + pattern)
    return m.group(1)
if 'VERIFY: ACCEPT' not in log or 'VERIFY: REJECT' in log:
    raise SystemExit('Geode verification did not accept')
n = int(sys.argv[5])
size = pathlib.Path(sys.argv[3]).stat().st_size
disk_m = re.search(r'storage traffic over all prover phases: read ([\d.]+) GiB, written ([\d.]+) GiB; SSD-backed peak ([\d.]+) GiB', log)
disk_rd_gib = float(disk_m.group(1)) if disk_m else 0.0
disk_wr_gib = float(disk_m.group(2)) if disk_m else 0.0
disk_spill_gib = float(disk_m.group(3)) if disk_m else 0.0
print(json.dumps(dict(system='Alloul26', family=sys.argv[4], N=n, threads=int(sys.argv[6]),
    isa=sys.argv[7], cpus=sys.argv[9], soundness='default', run=int(sys.argv[8]),
    prove_s=float(match(r'^\s*([\d.]+)\s+TOTAL prover', log))/1000,
    verify_s=float(match(r'^\s*([\d.]+)\s+TOTAL verifier', log))/1000,
    bytes=size, bytes_per_sig=size/n,
    peak_rss_kib=int(match(r'Maximum resident set size \(kbytes\):\s*(\d+)', timing)),
    disk_read_gib=disk_rd_gib, disk_write_gib=disk_wr_gib, disk_spill_peak_gib=disk_spill_gib)))
PY
done
