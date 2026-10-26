#!/usr/bin/env python3
"""Sample /proc/stat every 100 ms until the stop file exists; print
<unix_ns> <busy_frac_all> <user> <system> <iowait> per line (deltas over the interval)."""
import sys, time, os
stop = sys.argv[1]
period = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
def read():
    with open('/proc/stat') as f:
        p = f.readline().split()[1:]
    v = list(map(int, p))
    # user nice system idle iowait irq softirq steal
    idle = v[3] + v[4]
    total = sum(v[:8])
    return total, idle, v[0] + v[1], v[2] + v[5] + v[6], v[4]
prev = read()
nxt = time.monotonic() + period
while not os.path.exists(stop):
    time.sleep(max(0.0, nxt - time.monotonic()))
    nxt += period
    cur = read()
    dt = cur[0] - prev[0]
    if dt > 0:
        busy = 1.0 - (cur[1] - prev[1]) / dt
        print(time.time_ns(), f"{busy:.4f}", f"{(cur[2]-prev[2])/dt:.4f}", f"{(cur[3]-prev[3])/dt:.4f}", f"{(cur[4]-prev[4])/dt:.4f}", flush=True)
    prev = cur
