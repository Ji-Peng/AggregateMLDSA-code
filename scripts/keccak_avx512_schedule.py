#!/usr/bin/env python3
"""Generates the loop body of the AVX-512 Keccak-f[1600] x8 permutation in
crates/agg-mldsa-simd/src/keccak/avx512.rs (`permute`).

The body is four rounds in SSA form (25 words and 5 column parities in and
out). It is list-scheduled for a model of two vector ports, where `vprolq`
runs only on port 0 and `vpternlogq`/`vpxorq` on either, with latency 1 and a
limit on live vectors; each emitted `tern!`/`rol!` is one non-`pure` `asm!`
statement, so LLVM keeps this order. A ternlog's destructive operand is one
that dies there when possible (with the truth table permuted to match).

Usage: scripts/keccak_avx512_schedule.py [LIMIT MODE loop]; the committed body
is `31 height loop`. Prints the body with `$a` word names; the permutation
uses `a`.
"""
import sys, itertools

RHO = {}
PLANES = [
 ([0,6,12,18,24],[0,44,43,21,14]),
 ([3,9,10,16,22],[28,20,3,45,61]),
 ([1,7,13,19,20],[1,6,25,8,18]),
 ([4,5,11,17,23],[27,36,10,15,56]),
 ([2,8,14,15,21],[62,55,39,41,2]),
]
for sl, rr in PLANES:
    for i, r in zip(sl, rr): RHO[i] = r

ops = []  # dict(kind, imm/n, ins, out)
def op(kind, ins, arg=None, name=None):
    v = len(ops)
    ops.append(dict(kind=kind, ins=list(ins), arg=arg, v=v))
    return ('v', v)

def build(rounds=24, carried=False):
    a = [('in', i) for i in range(25)]
    c = [('c', k) for k in range(5)] if carried else None
    for r in range(rounds):
        if c is None:
            c = [op('tern', [op('tern', [a[x], a[x+5], a[x+10]], 0x96), a[x+15], a[x+20]], 0x96) for x in range(5)]
        rr = [op('rot', [c[(x+1) % 5]], 1) for x in range(5)]
        t = [op('tern', [a[i], c[(i % 5 - 1) % 5], rr[i % 5]], 0x96) for i in range(25)]
        out = [None]*25
        for pi, (sl, rh) in enumerate(PLANES):
            e = [t[i] if RHO[i] == 0 else op('rot', [t[i]], RHO[i]) for i in sl]
            for k, i in enumerate(sl):
                o = op('tern', [e[k], e[(k+1) % 5], e[(k+2) % 5]], 0xd2)
                if pi == 0 and k == 0:
                    o = op('rc', [o], r)
                out[i] = o
        # Output lane x + 5 y is slot x of plane y.
        a = [out[PLANES[i // 5][0][i % 5]] for i in range(25)]
        if r < rounds - 1 or carried:
            cols = [[a[x + 5 * y] for y in range(5)] for x in range(5)]
            c = [op('tern', [op('tern', [col[0], col[1], col[2]], 0x96), col[3], col[4]], 0x96) for col in cols]
    return (a + c) if carried else a

maxlive = [0]
def schedule(outs, limit):
    n = len(ops)
    users = [[] for _ in range(n)]
    for o in ops:
        for x in o['ins']:
            if x[0] == 'v': users[x[1]].append(o['v'])
    # Height: longest path to any sink (latency 1).
    h = [0]*n
    for o in reversed(ops):
        h[o['v']] = 1 + max([h[u] for u in users[o['v']]], default=0)
    done_cycle = {}
    remaining_users = [len(users[v]) for v in range(n)]
    out_set = set(x[1] for x in outs if x[0] == 'v')
    live = set(('in', i) for i in range(25)) | set(('c', k) for k in range(5))
    in_users = {}
    for o in ops:
        for x in o['ins']:
            if x[0] != 'v': in_users[x] = in_users.get(x, 0) + 1
    order = []
    scheduled = [False]*n
    cycle = 0
    pending = set(range(n))
    def ready(v):
        return all(x[0] != 'v' or (x[1] in done_cycle and done_cycle[x[1]] < cycle) for x in ops[v]['ins'])
    def delta(v):
        # Change of live registers if v executes now.
        d = 1
        for x in set(map(tuple, ops[v]['ins'])):
            if x[0] == 'v' and remaining_users[x[1]] == 1 and x[1] not in out_set: d -= 1
            if x[0] != 'v' and in_users[x] == 1: d -= 1
        return d
    while pending:
        cands = [v for v in pending if ready(v)]
        slots = []
        # Port 0: a rotation if any is ready (it has no other port), else logic.
        for port in (0, 5):
            pool = [v for v in cands if v not in slots and (ops[v]['kind'] != 'rot' or port == 0)]
            if not pool: continue
            if len(live) >= limit:
                # Hard register limit: only operations that do not add a live value.
                low = [v for v in pool if delta(v) <= 0]
                if not low and (slots or port == 0):
                    continue
                pool = low or [min(pool, key=lambda v: (delta(v), v))]
            rot_first = (lambda v: -(ops[v]['kind'] == 'rot')) if port == 0 else (lambda v: 0)
            if len(live) >= limit:
                key = lambda v: (delta(v), rot_first(v), v if MODE == 'old' else -h[v], v)
            elif MODE == 'old':
                key = lambda v: (rot_first(v), v)
            else:
                key = lambda v: (rot_first(v), -h[v], v)
            v = min(pool, key=key)
            slots.append(v)
            for x in set(map(tuple, ops[v]['ins'])):
                if x[0] == 'v':
                    remaining_users[x[1]] -= 1
                    if remaining_users[x[1]] == 0 and x[1] not in out_set: live.discard(x)
                else:
                    in_users[x] -= 1
                    if in_users[x] == 0: live.discard(x)
            live.add(('v', v))
        for v in slots:
            done_cycle[v] = cycle
            pending.discard(v)
            order.append(v)
        cycle += 1
        maxlive[0] = max(maxlive[0], len(live))
    return order, cycle

def perm_tt(imm, perm):
    # Truth table of f(x[perm[0]], x[perm[1]], x[perm[2]]) as a function of x.
    out = 0
    for idx in range(8):
        bits = [(idx >> 2) & 1, (idx >> 1) & 1, idx & 1]  # a, b, c
        src = [bits[perm[0]], bits[perm[1]], bits[perm[2]]]
        if (imm >> (src[0]*4 + src[1]*2 + src[2])) & 1: out |= 1 << idx
    return out

def emit(order, outs):
    name = lambda x: f'$a{x[1]}' if x[0] == 'in' else (f'c{x[1]}' if x[0] == 'c' else f'v{x[1]}')
    n = len(ops)
    last = {}
    pos = {v: k for k, v in enumerate(order)}
    for v in order:
        for x in ops[v]['ins']:
            last[tuple(x)] = pos[v]
    for x in outs: last[tuple(x)] = 10**9
    lines = []
    for v in order:
        o = ops[v]
        ins = [tuple(x) for x in o['ins']]
        if o['kind'] == 'rot':
            lines.append(f'let v{v} = rol!({o["arg"]}, {name(ins[0])});')
        elif o['kind'] == 'rc':
            lines.append(f'let v{v} = _mm512_xor_si512({name(ins[0])}, _mm512_set1_epi64({RCN}[{o["arg"]}] as i64));')
        else:
            # Put an operand that dies here first (destructive destination).
            best = None
            for p in itertools.permutations(range(3)):
                dies = last[ins[p[0]]] == pos[v] and ins.count(ins[p[0]]) == 1
                if best is None or (dies and not best[0]):
                    best = (dies, p)
                if dies: break
            p = best[1]
            # New argument order (ins[p0], ins[p1], ins[p2]); g(y) = f(x) where x[p[j]] = y[j].
            inv = [p.index(j) for j in range(3)]
            imm = perm_tt(o['arg'], inv)
            q = [ins[p[0]], ins[p[1]], ins[p[2]]]
            lines.append(f'let mut v{v} = {name(q[0])}; tern!({hex(imm)}; v{v}, {name(q[1])}, {name(q[2])});')
    for i, x in enumerate(outs):
        lines.append(f'$a{i} = {name(tuple(x))};' if i < 25 else f'c{i - 25} = {name(tuple(x))};')
    return lines

limit = int(sys.argv[1]) if len(sys.argv) > 1 else 31
MODE = sys.argv[2] if len(sys.argv) > 2 else 'height'
LOOP = len(sys.argv) <= 3 or sys.argv[3] == 'loop'
RCN = 'rc' if LOOP else 'RC'
outs = build(4, True) if LOOP else build()
order, cycles = schedule(outs, limit)
sys.stderr.write(f'{len(ops)} ops, {cycles} cycles on the model, max live {maxlive[0]}\n')
print('\n'.join('        ' + l for l in emit(order, outs)))
