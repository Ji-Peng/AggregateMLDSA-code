#!/usr/bin/env bash
# Mixed-field against all-prime-field run of one parameter set, batch size and tier.
#
# usage: allprime.sh SET N TIER OUTDIR
#   SET     44, 65 or 87
#   N       batch size (records)
#   TIER    ifma, avx512, avx2 or scalar (the AGG_ISA cap of both runs)
#   OUTDIR  output directory (records are reused when present)
#   REPS    environment: timed runs after one warm-up (default 5)
#
# Builds agg once, generates the artifact records (keygen-batch --seed 42,
# sign-batch --seed 43), then per hash field (gf2, fp) runs `agg bench` under
# /usr/bin/time -v, and one `agg aggregate` plus `agg verify` of the records
# for the byte fingerprint. Writes OUTDIR/<set>-<N>-<tier>-<field>.{json,time,log,agg}
# and prints one table row per field. `agg bench` signs its own records from
# `--seed 42`; the aggregate on the seed-42/43 records is what `gates.py` pins at 1K.
set -euo pipefail

set_=${1:?SET: 44, 65 or 87}
n=${2:?N: batch size}
tier=${3:?TIER: ifma, avx512, avx2 or scalar}
outdir=${4:?OUTDIR}
reps=${REPS:-5}

case "$set_" in 44 | 65 | 87) ;; *) echo "SET must be 44, 65 or 87" >&2; exit 2 ;; esac
case "$n" in '' | *[!0-9]*) echo "N must be a positive integer" >&2; exit 2 ;; esac
case "$reps" in '' | *[!0-9]* | 0) echo "REPS must be a positive integer" >&2; exit 2 ;; esac
# A cap never raises the level, so a tier the CPU lacks would silently run lower.
flags=$(grep -m1 '^flags' /proc/cpuinfo || true)
case "$tier" in
  ifma | avx512ifma) isa=avx512ifma; need=avx512ifma ;;
  avx512) isa=avx512; need=avx512f ;;
  avx2) isa=avx2; need=avx2 ;;
  scalar) isa=scalar; need=avx2 ;;
  *) echo "TIER must be ifma, avx512, avx2 or scalar" >&2; exit 2 ;;
esac
if ! grep -qw "$need" <<<"$flags"; then
  echo "this CPU lacks $need: tier $tier is not available" >&2
  exit 2
fi
[ -x /usr/bin/time ] || { echo "/usr/bin/time is required (GNU time)" >&2; exit 2; }

here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
mkdir -p "$outdir"
outdir=$(cd "$outdir" && pwd)

# The release baseline is .cargo/config.toml; an exported RUSTFLAGS would replace it.
unset RUSTFLAGS
echo "building agg-cli (release)" >&2
(cd "$repo" && cargo build --release -p agg-cli >&2)
agg="${CARGO_TARGET_DIR:-$repo/target}/release/agg"
[ -x "$agg" ] || { echo "$agg is missing after the build" >&2; exit 1; }
export AGG_GIT_REV="${AGG_GIT_REV:-$(git -C "$repo" rev-parse HEAD 2>/dev/null || echo unknown)}"

keys="$outdir/keys-$set_-$n.bin"
records="$outdir/records-$set_-$n.bin"
if [ ! -s "$records" ]; then
  echo "generating $n records of ML-DSA-$set_" >&2
  "$agg" keygen-batch --params "$set_" --count "$n" --seed 42 --out "$keys" >&2
  "$agg" sign-batch --params "$set_" --keys "$keys" --count "$n" --seed 43 --out "$records" >&2
fi

export AGG_ISA="$isa"
for field in gf2 fp; do
  stem="$outdir/$set_-$n-$tier-$field"
  echo "ML-DSA-$set_, N = $n, $tier, $field: bench (1 warm-up + $reps)" >&2
  if ! /usr/bin/time -v "$agg" bench --count "$n" --warmup 1 --reps "$reps" --params "$set_" \
    --seed 42 --hash-field "$field" --json "$stem.json" >"$stem.log" 2>"$stem.time"; then
    tail -n 25 "$stem.time" >&2
    exit 1
  fi
  echo "ML-DSA-$set_, N = $n, $tier, $field: aggregate and verify the records" >&2
  if ! "$agg" aggregate --params "$set_" --records "$records" --hash-field "$field" \
    --out "$stem.agg" >>"$stem.log" 2>&1; then
    tail -n 25 "$stem.log" >&2
    exit 1
  fi
  if "$agg" verify --params "$set_" --records "$records" --aggregate "$stem.agg" \
    --hash-field "$field" >>"$stem.log" 2>&1; then
    echo accept >"$stem.verdict"
  else
    echo REJECT >"$stem.verdict"
  fi
done

python3 - "$outdir" "$set_" "$n" "$tier" "$reps" <<'EOF'
import hashlib, json, re, sys
from pathlib import Path

outdir, set_, n, tier, reps = sys.argv[1:]
outdir = Path(outdir)

def median(values):
    values = sorted(values)
    k = len(values)
    return values[k // 2] if k % 2 else 0.5 * (values[k // 2 - 1] + values[k // 2])

def time_rss_kib(path):
    match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", path.read_text())
    return int(match.group(1)) if match else None

rows = []
for field in ("gf2", "fp"):
    stem = outdir / f"{set_}-{n}-{tier}-{field}"
    doc = json.loads(stem.with_suffix(".json").read_text())
    # Prover peak from the bench JSON; GNU time's maximum over the whole process otherwise.
    peak = doc.get("prove_peak_kib") or time_rss_kib(stem.with_suffix(".time"))
    agg = stem.with_suffix(".agg")
    rows.append([
        field,
        f"{median(doc['prove_ms']):.1f}",
        f"{median(doc['verify_uncached_ms']):.1f}",
        f"{median(doc['verify_cached_ms']):.1f}",
        f"{doc['bytes_per_signature']:.1f}",
        f"{peak / 2**20:.2f}" if peak else "n/a",
        hashlib.sha256(agg.read_bytes()).hexdigest()[:16],
        stem.with_suffix(".verdict").read_text().strip(),
    ])
first = json.loads((outdir / f"{set_}-{n}-{tier}-gf2.json").read_text())
print(f"ML-DSA-{set_}, N = {n}, tier {tier}, 1 warm-up + {reps} runs (medians), "
      f"revision {first.get('revision')}, backend {first['kernels']['agg_simd_backend']}")
head = ["field", "prove ms", "verify ms", "cached ms", "B/sig", "peak GiB",
        "records agg sha256", "verify"]
widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(head)]
print("  ".join(h.rjust(w) for h, w in zip(head, widths)))
for row in rows:
    print("  ".join(c.rjust(w) for c, w in zip(row, widths)))
EOF
grep -qx accept "$outdir/$set_-$n-$tier-gf2.verdict" && grep -qx accept "$outdir/$set_-$n-$tier-fp.verdict"
