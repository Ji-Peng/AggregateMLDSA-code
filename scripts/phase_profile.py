#!/usr/bin/env python3
"""Per-step wall time and CPU busy share of stamped proves (docs/perf.md).
usage: phase_profile.py RUN [RUN ...]  where RUN.cpu is scripts/cpusample.py output and RUN.err
the stderr of `AGG_PHASE_STAMPS=1 agg aggregate`; prints the median over the runs."""
import sys, statistics as S
runs=sys.argv[1:]
def load(f):
    samples=[(int(l.split()[0]),float(l.split()[1])) for l in open(f+'.cpu')]
    st=dict((l.split()[1],int(l.split()[2])) for l in open(f+'.err') if l.startswith('phase-stamp '))
    return samples,st
def busy(samples,a,b):
    tot=w=0.0
    for i,(t,x) in enumerate(samples):
        lo=samples[i-1][0] if i else t-100_000_000
        ov=min(b,t)-max(a,lo)
        if ov>0: tot+=x*ov; w+=ov
    return tot/w if w else float('nan')
wins=[('start','instance','instance (ExpandA, mu, c, T)'),('instance','tables','witness and G_main tables'),('tables','flock-records','Flock records'),
('flock-records','setup','FlockProver::new + Akita setup'),('setup','round1-commit','event 1 G_main commits beside the Flock event 2'),
('round1-commit','flock-commit','Flock event 2 alone (witness + commit)'),('flock-commit','round2','events 3-5: matrices, round 2'),('round2','shares','event 6 shares'),
('shares','parity-tables','fork: parity tables (PIOP side)'),('parity-tables','lookup','fork: lookup'),('lookup','stage1','fork: tau, pre-stage, stage 1'),('stage1','stage2','fork: stage 2'),('stage2','stage3','fork: stage 3'),
('stage3','join','fork: Flock-only tail after stage 3'),('shares','join','fork-join section'),('join','opening','event 13 Akita opening'),('opening','end','serialize'),('start','end','TOTAL')]
L=[load(f) for f in runs]
print(f"{'window':<50}{'wall ms (median)':>18}{'busy %':>8}{'idle vCPU-s':>12}")
for a,b,name in wins:
    ws=[(st[b]-st[a])/1e6 for _,st in L]; bs=[busy(s,st[a],st[b]) for s,st in L]
    w=S.median(ws); u=S.median(bs)
    print(f"{name:<50}{w:>12.1f} ({min(ws):.0f}-{max(ws):.0f}){100*u:>8.1f}{(1-u)*16*w/1e3:>12.2f}")
