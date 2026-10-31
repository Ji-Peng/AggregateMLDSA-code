#!/usr/bin/env bash
# Sample the frequency of the CPUs in use at 10 Hz.
#
# usage: sampler.sh OUTPUT INTERVAL_MS CPU [CPU ...]
# Each line is: <unix nanoseconds> <cpu> <scaling_cur_freq in kHz>
set -u

out=$1
interval_ms=$2
shift 2

: >"$out"
interval_s=$(awk -v ms="$interval_ms" 'BEGIN { printf "%.3f", ms / 1000 }')

while :; do
  now=$(date +%s%N)
  for cpu in "$@"; do
    freq_file="/sys/devices/system/cpu/cpu${cpu}/cpufreq/scaling_cur_freq"
    if [ -r "$freq_file" ]; then
      printf '%s %s %s\n' "$now" "$cpu" "$(cat "$freq_file")" >>"$out"
    fi
  done
  sleep "$interval_s"
done
