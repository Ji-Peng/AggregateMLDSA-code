#!/usr/bin/env python3
"""Proof-of-work schedule of docs/soundness.md (normative), recomputed from first principles.

Run (isolated mode, standard library only):

    python3 -I docs/scripts/pow_plan.py              # summary tables (committed in the doc)
    python3 -I docs/scripts/pow_plan.py --details 15 # every site of N* = 2^15, default config
    python3 -I docs/scripts/pow_plan.py --relation h-exact  # relation H-exact (docs/h-exact.md)
    python3 -I docs/scripts/pow_plan.py --set 65            # ML-DSA-65, relation H-FIPS
    python3 -I docs/scripts/pow_plan.py --set 65 --relation h-exact
    python3 -I docs/scripts/pow_plan.py --set 87 --details 15

What it computes, with exact integer arithmetic:

* the ordered PIOP field-challenge sites of the mode-A/mode-B transcript
  (crates/agg-piop/src/transcript.rs Registry::new) for every N* = 2^3 .. 2^15, each with its
  round-by-round numerator nu_t, its assigned bound e_t (>= nu_t / |F|), the difficulty
  g_t = min{g >= 0 : 2^128 e_t <= 2^g} (Aerie) and the nonce width g_t + 7
  (0 when g_t = 0: no nonce); the projection site proj.psi (a 16 KiB byte challenge with an
  exact 2^-128 bound) leads them with one bit of assigned work (2^-129);
* the sites of every tied part (tied-part proof version 5, docs/parity-chain.md:
  the tie site of the public claim, the eq_Y site and the parity-chain rounds) for part
  sizes d = 3 .. 13 and span lengths S = 1, 2, 4, with the Johnson-list factor of the
  sponge commitment;
* the packed nonce stream size, the expected honest work sum 2^g, the honest failure bound,
  the union-bound ledger before work, and the plan digest test vectors (encoding of
  docs/soundness.md);
* the audit of Flock's outer (pre-binding) challenge sites against the Flock paper's
  list-size accounting (docs/soundness.md).
"""

from __future__ import annotations

import argparse
import hashlib
import math
from fractions import Fraction

P = 2**64 - 59
K = P * P  # |K| = p^2, the PIOP challenge field
F2_128 = 2**128  # |GF(2^128)|, the Flock / tie challenge field
TARGET = 128
SLACK = 7  # nonce width = g + 7 (Aerie, Akita)
M_TABLE = 2**16  # |T_16|
CHALLENGES = 2  # LOOKUP_CHALLENGES
# Relation H-FIPS (the default, docs/h-exact.md): stage 2 holds def_y1, leaf8[K_2], leaf8[K_3]
# of the projection digits of v and F3 (9 coefficients, g = 4; its rounds have degree 5),
# stage 3 56 coefficients (g = 6).
# Relation H-exact (an optional reference relation): stage 2 holds leaf6[U_hi], leaf8[K_2],
# leaf8[K_3] (9 coefficients, g = 4), stage 3 the ring, three links, two Booleanity
# zero-checks, 37 reductions, the projection group and the k reduction and five stack
# reductions (50 coefficients, g = 6); the degrees are the same. Every difficulty, total and
# plan digest of H-FIPS equals H-exact's (the relation is bound by the manifest bytes of
# event 0, not by the plan encoding).
STAGE_DEGREES = (4, 5, 3)
RELATION_GAMMAS = {"h-exact": (26, 9, 50), "h-fips": (26, 9, 56)}
STAGE_GAMMAS = RELATION_GAMMAS["h-fips"]
SPONGE_K_LOG = 19
MIN_DENSE_M = 22
RECT_MIN_ROWS_LOG = 11
D_SEG_MAX = 13
OUT_VARS = 12  # log2 m_3, the output-row coordinates of eq_Y
ZETA_VARS = 19  # log2 J_1
LANE_VARS = 3  # ceil(log2 6): the chain's eight lanes
BLOCK_RECORDS_LOG2 = 9  # records per level-1 block of the bridge
P_PAD_BITS = 10  # log2 of the padded positions P_pad of one record on the bridge input

# Parameter sets (FIPS 204 ML-DSA-44, -65, -87). The main domain is (j, rho, i) for 44
# (d_main = 10 + d_sig) and (j, i) with polynomial lanes for 65 and 87 (d_main = 8 + d_sig).
# The main-domain lookup stacks hold the l + k looked-up Z_lo/U_lo polynomials split by the
# binary decomposition of l + k (44: one stack [Z_lo | U_lo] of two columns); the level-1 and
# level-2 carry stacks have one column per w1 bit (6 in 8 slots for 44, 4 for 65/87); the
# projection domain D_k has 128 rows per record and block (87: two row blocks).
# Stage 1 has degree 4 with the six-bit range identity of 44 and 3 otherwise; the batching
# vectors count the instances of crates/agg-piop/src/stage{1,2,3}.rs gamma_count (65 and 87:
# one Booleanity check per bit and lane, k ring instances, l + k links and Booleanity
# checks). The bridge constants of 65/87 (four chain lanes, 256 records per level-1 block,
# P_pad = 2^11) set the tied-part rounds of mode A; the sponge's public region (mu, c~, the
# constant) spans 8 words for 44 and 65 and 16 words for 87 (c~ of 64 bytes), so the tie
# site's public claim has 3 or 4 word coordinates beyond the nu_s rows.
SETS = {
    "44": {"d_pos": 10, "main_stacks": (2,), "carry_slots": 8, "d1_shift": 9, "dk_rows": 7,
           "degrees": (4, 5, 3), "gammas": None, "lane_vars": 3, "block_log": 9, "p_pad_bits": 10, "public_vars": 3},
    "65": {"d_pos": 8, "main_stacks": (8, 2, 1), "carry_slots": 4, "d1_shift": 8, "dk_rows": 7,
           "degrees": (3, 5, 3), "gammas": (37, 18, 105), "lane_vars": 2, "block_log": 8, "p_pad_bits": 11, "public_vars": 3},
    "87": {"d_pos": 8, "main_stacks": (8, 4, 2, 1), "carry_slots": 4, "d1_shift": 8, "dk_rows": 8,
           "degrees": (3, 5, 3), "gammas": (45, 22, 132), "lane_vars": 2, "block_log": 8, "p_pad_bits": 11, "public_vars": 4},
}
SET = SETS["44"]
# H-FIPS of ML-DSA-65 and ML-DSA-87: stage 2 trades leaf8[U_mid] for F_e per row lane (same length),
# stage 3 gains 5 k + 1 coefficients (Booleanity of l_mid, F_s, F_delta, F_w1 per row, F_omega, the
# reductions of delta and e; minus link[U_mid]). The hint slack u_h is a fifth column of the
# level-2 carry stack, padded from 4 to 8 slots. H-exact takes the counts of SETS.
LANE_FIPS = {"65": {"gammas": (37, 18, 136), "carry2_slots": 8},
             "87": {"gammas": (45, 22, 173), "carry2_slots": 8}}
CARRY2_SLOTS = None  # level-2 carry-stack slots when they differ from SET["carry_slots"]
PROFILE_ID = {"fast": 1, "slim": 4, "slim8": 5, "modeB": 3}
# Level-0 Ligerito rate exponent r (rho = 2^-r) per profile; eta = 0.02 at level 0 in every
# shipped and derived config (vendor/flock-core/configs/ligerito/m*_{fast,slim,slim8}.toml).
L0_RATE = {"fast": 1, "slim": 2, "slim8": 3}
# Measured cost of one Blake2b predicate preview (Akita preview_grinding_predicate,
# docs/lookup.md): 724 ns serial on this machine.
PREVIEW_NS = 724.0

CLASS_ID = {
    "proj.psi": 11,
    "lookup.x": 1,
    "lookup.mu": 2,
    "lookup.gamma": 3,
    "lookup.round": 4,
    "tau": 5,
    "stage.gamma": 6,
    "stage.round": 7,
    "tie": 8,
    "chain.eq": 9,
    "chain.round": 10,
}

# proj.psi is a 16 KiB byte challenge (Psi squeezed directly, no seed expansion) whose bad
# set has mass at most 2^-128, so the schedule assigns it one bit: e = 2^-128, g = 1,
# 2^-129 per query.
PSI_BITS = 1


# Akita's capped transcript proof-of-work searches at 2^15 (docs/soundness.md).
AKITA_SITES_2_15 = 461


def grind_bits(num: int, den: int) -> int:
    """Least g >= 0 with 2^128 * num / den <= 2^g (Aerie Eq. (17), exact)."""
    if num <= 0:
        return 0
    g = 0
    while (num << TARGET) > (den << g):
        g += 1
    return g


def site_bits(cls: str, num: int, den: int) -> int:
    """Difficulty of a site: the Aerie bound for a field challenge, PSI_BITS for proj.psi."""
    return PSI_BITS if cls == "proj.psi" else grind_bits(num, den)


def width(g: int) -> int:
    return g + SLACK if g > 0 else 0


def list_size(rate_exp: int) -> int:
    """floor(L_max), L_max = 1/(2 eta sqrt(rho)) = 25 * 2^(r/2) at eta = 1/50 (Flock Eq. (10))."""
    return math.isqrt(625 * 2**rate_exp)


def capacity_log(n: int, k_log: int = SPONGE_K_LOG) -> int:
    """agg_keccak::prover::capacity_log: rows of the sponge instance."""
    nu = max((n - 1).bit_length() if n > 1 else 0, 3, MIN_DENSE_M - k_log)
    if n == 1 << nu and nu < RECT_MIN_ROWS_LOG:
        nu += 1
    return nu


def geometry(d_sig: int) -> dict:
    d_main = SET["d_pos"] + d_sig
    d1 = 13 + max(0, d_sig - SET["d1_shift"])
    d2 = 13 + max(0, d_sig - 13)
    d3 = 12 + max(0, d_sig - 13)
    d_k = SET["dk_rows"] + d_sig  # the projection domain (o, i): 128 rows (times blocks) times N*
    slots = SET["carry_slots"]
    slots2 = CARRY2_SLOTS or slots
    # The main-domain stacks, kappa^(1) and kappa^(2) (one column per bit in `slots` slots),
    # kappa^(3), [K_0 | K_1].
    main = [d_main + (c - 1).bit_length() for c in SET["main_stacks"]]
    stacks = main + [d1 + (slots - 1).bit_length(), d2 + (slots2 - 1).bit_length(), d3, d_k + 1]
    trees = (stacks + [16]) * CHALLENGES
    lookups = (sum(SET["main_stacks"]) * 2**d_main + slots * 2**d1 + slots2 * 2**d2 + 2**d3
               + 2 * 2**d_k)
    return {
        "d_sig": d_sig,
        "d_main": d_main,
        "d": (d1, d2, d3),
        "trees": trees,
        "D": max(trees),
        "L": lookups,
        "segments": 1 << max(0, d_sig - D_SEG_MAX),
        "d_seg": min(d_sig, D_SEG_MAX),
    }


def piop_sites(geo: dict) -> list:
    """The PIOP sites in registry order: (class, i, j, nu, e_num, e_den).

    `proj.psi` is a byte challenge (nu = None, e = 2^-128) with one assigned bit of work; it
    precedes the field-challenge sites, which begin with the joint-bound site `X` (nu = None).
    """
    sites = [("proj.psi", 0, 0, None, 1, F2_128)]
    L, D, trees, d_main = geo["L"], geo["D"], geo["trees"], geo["d_main"]
    x_num = (L + M_TABLE - 1) ** 2
    sites.append(("lookup.x", 0, 0, None, x_num, K * K))
    sites.append(("lookup.mu", 1, 0, 1, 1, K))
    for s in range(2, D + 1):
        active = sum(1 for t in trees if t >= s)
        sites.append(("lookup.gamma", s, 0, 1, 2 * active, K))
        for r in range(s - 1):
            sites.append(("lookup.round", s, r, 3, 3, K))
        sites.append(("lookup.mu", s, 0, 1, 1, K))
    sites.append(("tau", 0, 0, d_main, d_main, K))
    for stage in (1, 2, 3):
        sites.append(("stage.gamma", stage, 0, 1, STAGE_GAMMAS[stage - 1], K))
        deg = STAGE_DEGREES[stage - 1]
        for r in range(d_main):
            sites.append(("stage.round", stage, r, deg, deg, K))
    return sites


def chain_rounds(d: int, span_log: int) -> int:
    """Rounds of the parity chain of a production part (agg_bridge::pow::chain_rounds)."""
    block = SET["block_log"]
    return min(SET["p_pad_bits"] + min(d, block), ZETA_VARS) + max(0, d - block) + SET["lane_vars"] + span_log


def tie_terms(d: int, span_log: int, nu_s: int, profile: str) -> dict:
    """Work sites of one tied part, version 5 (docs/parity-chain.md, docs/soundness.md).

    The sponge commitment is bound only to its Johnson list of L_s codewords when the tie
    challenger draws, so every numerator carries L_s: the public claim L_s (nu_s + c) with c
    the public region's word coordinates (3, or 4 for ML-DSA-87), eq_Y
    L_s (log2 m_3 + span_log), each degree-2 round 2 L_s.
    """
    ls = list_size(L0_RATE[profile])
    tie = ls * (nu_s + SET["public_vars"])
    eq = ls * (OUT_VARS + span_log)
    rnd = 2 * ls
    rounds = chain_rounds(d, span_log)
    g, g_eq, g_rd = grind_bits(tie, F2_128), grind_bits(eq, F2_128), grind_bits(rnd, F2_128)
    return {
        "d": d,
        "span_log": span_log,
        "nu_s": nu_s,
        "list": ls,
        "tie": tie,
        "eq": eq,
        "round": rnd,
        "rounds": rounds,
        "g": g,
        "w": width(g),
        "g_eq": g_eq,
        "g_round": g_rd,
        # bytes of nonces in the proof (LE32 each): tie, eq_Y, every round
        "nonce_bytes": 4 * (2 + rounds),
        "work": (1 << g) + (1 << g_eq) + rounds * (1 << g_rd),
        "sites": 2 + rounds,
    }


def tie_parts(geo: dict, profile: str, span: bool, n: int) -> list:
    """Tied parts of a mode-A batch of n records with default part size (d_part = d_seg)."""
    segs = geo["segments"]
    if segs == 1:
        return [tie_terms(geo["d_seg"], 0, capacity_log(n), profile)]
    if span:
        return [tie_terms(D_SEG_MAX, (segs - 1).bit_length(), capacity_log(n), profile)]
    parts = []
    for sigma in range(segs):
        live = min(max(n - (sigma << D_SEG_MAX), 0), 1 << D_SEG_MAX)
        if live:
            parts.append(tie_terms(D_SEG_MAX, 0, capacity_log(live), profile))
    return parts


def plan_digest(geo: dict, sites: list, ties: list, profile: str, span_count: int, d_part: int) -> str:
    """SHA3-256 test vector of the canonical plan encoding (docs/soundness.md)."""
    entries = []
    # Tied parts first: a fixed convention of the plan encoding (the transcript absorbs them
    # at the join after stage 3).
    for p, t in enumerate(ties):
        entries.append((CLASS_ID["tie"], p, 0, t["g"], t["w"]))
        entries.append((CLASS_ID["chain.eq"], p, 0, t["g_eq"], width(t["g_eq"])))
        for k in range(t["rounds"]):
            entries.append((CLASS_ID["chain.round"], p, k, t["g_round"], width(t["g_round"])))
    for cls, i, j, nu, num, den in sites:
        g = site_bits(cls, num, den)
        entries.append((CLASS_ID[cls], i, j, g, width(g)))
    body = bytearray()
    body += (1 << geo["d_sig"]).to_bytes(8, "little")
    body += bytes([PROFILE_ID[profile], span_count, d_part])
    body += len(entries).to_bytes(4, "little")
    for cls, i, j, g, w in entries:
        body += bytes([cls]) + i.to_bytes(2, "little") + j.to_bytes(2, "little") + bytes([g, w])
    framed = b"agg-mldsa/v2/pow-plan" + len(body).to_bytes(8, "little") + bytes(body)
    return hashlib.sha3_256(framed).hexdigest()


def summarize(sites: list) -> dict:
    total_bits = 0
    nonces = 0
    work = 0
    weights = []
    union_nu = 0
    for cls, i, j, nu, num, den in sites:
        g = site_bits(cls, num, den)
        w = width(g)
        total_bits += w
        nonces += 1 if g else 0
        work += (1 << g) if g else 0
        # per-address weight after work: 2^-g * nu / |F| (nu replaced by e for the joint X site)
        weights.append(Fraction(num if nu is None else nu, den) / (1 << g))
        if nu is not None:
            union_nu += nu
    return {
        "bits": total_bits,
        "bytes": (total_bits + 7) // 8,
        "nonces": nonces,
        "work": work,
        "max_weight_log2": math.log2(max(weights)),
        "union_nu": union_nu,
    }


def class_table(sites: list) -> list:
    rows = {}
    order = []
    for cls, i, j, nu, num, den in sites:
        g = site_bits(cls, num, den)
        key = (cls, nu, num if cls != "lookup.x" else None, g)
        if key not in rows:
            rows[key] = 0
            order.append(key)
        rows[key] += 1
    return [(k, rows[k]) for k in order]


def fmt_e(cls, nu, num) -> str:
    if cls == "proj.psi":
        return "2^-128"
    if cls == "lookup.x":
        return "((L+M-1)/|K|)^2"
    return f"{num}/|K|"


def print_summary(args) -> None:
    print(f"== ML-DSA-{args.set}, relation {args.relation}: STAGE_DEGREES = {STAGE_DEGREES}, STAGE_GAMMAS = {STAGE_GAMMAS} ==")
    print()
    print("== PIOP sites per N* (mode A and mode B; registry order) ==")
    print("d_sig  N*      d_main  D   L            X-site log2   sites  nonces  bits  bytes  work(sum 2^g)  est.ms  max weight  ledger(before work)")
    for d_sig in range(3, 16):
        geo = geometry(d_sig)
        sites = piop_sites(geo)
        s = summarize(sites)
        x_site = next(site for site in sites if site[0] == "lookup.x")
        x_log2 = math.log2(Fraction(x_site[4], x_site[5]))
        assert s["max_weight_log2"] <= -TARGET, "a site exceeds 2^-128"
        print(
            f"{d_sig:>5}  {1 << d_sig:>6}  {geo['d_main']:>6}  {geo['D']:>2}  {geo['L']:>11}  {x_log2:>12.1f}  {len(sites):>5}  {s['nonces']:>6}  {s['bits']:>4}  {s['bytes']:>5}  {s['work']:>13}  {s['work'] * PREVIEW_NS / 1e6:>6.2f}  2^{s['max_weight_log2']:.2f}  {s['union_nu']}/|K| = 2^{math.log2(Fraction(s['union_nu'], K)):.1f}"
        )
    print()
    geo = geometry(15)
    print(f"== PIOP site classes at N* = 2^15 (d_main = {geo['d_main']}, D = {geo['D']}) ==")
    print("class         count  nu      e_t               g   nonce bits")
    for (cls, nu, num, g), count in class_table(piop_sites(geo)):
        label = "byte" if cls == "proj.psi" else ("joint" if cls == "lookup.x" else str(nu))
        print(f"{cls:<12}  {count:>5}  {label:<6}  {fmt_e(cls, nu, num):<16}  {g:>2}  {width(g):>3}")
    print()
    # The spec 12.1 ledger counted the instances of ML-DSA-44; other sets print the
    # round-by-round sums only.
    print("== Old ledger (spec 12.1, union bound per instance) vs round-by-round numerators, before work ==")
    for d_sig in (10, 13, 15) if args.set == "44" else ():
        geo = geometry(d_sig)
        dm, D = geo["d_main"], geo["D"]
        gkr = 3 * D * (D - 1) // 2 + 2 * (D - 1) + 1
        deg = sum(STAGE_DEGREES)
        old = deg * dm + 19 * dm + 4 * dm + 62 * dm + 3 + gkr
        new = deg * dm + dm + 3 + gkr
        print(
            f"N* = 2^{d_sig}: spec 12.1 {old}/|K| = 2^{math.log2(Fraction(old, K)):.1f};"
            f" round-by-round sum {new}/|K| = 2^{math.log2(Fraction(new, K)):.1f}"
            f" (stages {deg * dm}, tau {dm}, gammas 3, GKR {gkr})"
        )
    print()
    theta = Fraction(1, math.comb(8192, 14)) * 2 + Fraction(1, math.comb(4096, 18))
    print("== Seed site sd (event 3), no nonce ==")
    for prof in ("fast", "slim", "slim8"):
        ls = list_size(L0_RATE[prof])
        e = theta * ls
        print(f"{prof:<6} theta = 2^{math.log2(theta):.2f}, times L_s = {ls}: 2^{math.log2(e):.2f}, g = {grind_bits(e.numerator, e.denominator)}")
        # sum over segment checks of L_s^(parts in the segment) theta (docs/soundness.md);
        # mode A admits at most two parts per segment.
        rows = (("span off, 4 segments x 1 part", 4 * ls), ("1 segment x 2 parts", ls * ls),
                ("4 segments x 2 parts", 4 * ls * ls), ("1 segment x 3 parts (refused)", ls ** 3))
        for label, mult in rows:
            e = theta * mult
            print(f"       {label:<31} 2^{math.log2(e):.2f}, g = {grind_bits(e.numerator, e.denominator)}")
    print()
    print("== Sites of one tied part, version 5 (GF(2^128), docs/parity-chain.md 8), full parts n = 2^d S ==")
    print("profile  d   S  nu_s  list  tie=L(nu_s+c)   g   eq=L(12+s)  g_eq  round=2L  g_rd  rounds  nonce bytes  work(sum 2^g)")
    for prof in ("fast", "slim", "slim8"):
        for d in range(3, 14):
            for S in (1, 2, 4):
                if S > 1 and d != D_SEG_MAX:
                    continue
                span_log = (S - 1).bit_length()
                t = tie_terms(d, span_log, capacity_log((1 << d) * S), prof)
                print(
                    f"{prof:<7} {d:>2}  {S:>2}  {t['nu_s']:>4}  {t['list']:>4}  {t['tie']:>13}  {t['g']:>2}  {t['eq']:>10}  {t['g_eq']:>4}  {t['round']:>8}  {t['g_round']:>4}  {t['rounds']:>6}  {t['nonce_bytes']:>11}  {t['work']:>13}"
                )
    print()
    print("== Flock outer sites before the level-0 two-point binding (audit, Section 4.3) ==")
    print("Under the Flock paper's accounting (Appendix A 'Security', Remark 11) each outer challenge")
    print("carries the list factor L_max; the vendored per_challenge_128 numerators do not (except the")
    print("transport claim batch). Work-normalized bits = 128 + bits - log2(nu * L_max).")
    outer = [
        ("zerocheck initial (m=30)", 30, (30).bit_length()),
        ("zerocheck initial (m=34)", 34, (34).bit_length()),
        ("zerocheck skip (K_SKIP=6)", 127, 7),
        ("zerocheck round", 2, 2),
        ("lincheck alpha/beta", 1, 1),
        ("lincheck round", 2, 2),
        ("lincheck skip (k=6)", 63, 6),
        ("ring-switch point", 7, 3),
        ("transport claim batch", 1, 6),  # numerator L itself; the 6 bits cover the list
        ("merged-sumcheck round", 2, 2),
    ]
    for prof in ("fast", "slim", "slim8"):
        lmax = 25 * 2 ** (L0_RATE[prof] / 2)
        print(f"{prof}: L_max = {lmax:.2f} (log2 {math.log2(lmax):.2f}), floor {list_size(L0_RATE[prof])}")
        for name, nu, bits in outer:
            sec = TARGET + bits - math.log2(nu * lmax)
            fix = math.floor(math.log2(nu * lmax)) + 1
            print(f"   {name:<26} nu {nu:>4}  bits {bits:>2}  -> {sec:7.2f} bits; strict bits with L_max: {fix}")
    print()
    print("== Totals at N = N* (bytes of the PIOP packed stream + tied-part nonces; expected work) ==")
    print("d_sig  profile  span  parts  tie g     PIOP bytes  part bytes  total bytes  B/sig    PIOP work  part work")
    for d_sig in range(3, 16):
        geo = geometry(d_sig)
        s = summarize(piop_sites(geo))
        for prof in ("fast", "slim", "slim8"):
            spans = (True, False) if geo["segments"] > 1 else (True,)
            for span in spans:
                ties = tie_parts(geo, prof, span, 1 << d_sig)
                tie_bytes = sum(t["nonce_bytes"] for t in ties)
                tie_work = sum(t["work"] for t in ties)
                tot = s["bytes"] + tie_bytes
                print(
                    f"{d_sig:>5}  {prof:<7}  {'on ' if span else 'off'}   {len(ties):>5}  {','.join(str(t['g']) for t in ties):<8}  {s['bytes']:>10}  {tie_bytes:>10}  {tot:>11}  {tot / (1 << d_sig):>6.3f}  {s['work']:>9}  {tie_work:>9}"
                )
    print()
    print("== Plan digests (SHA3-256, Section 6.4 encoding; N = N*, default part size) ==")
    for d_sig in range(3, 16):
        geo = geometry(d_sig)
        sites = piop_sites(geo)
        print(f"N* = 2^{d_sig:<2} modeB          {plan_digest(geo, sites, [], 'modeB', 0, 0)}")
        for prof in ("fast", "slim", "slim8"):
            spans = (True, False) if geo["segments"] > 1 else (True,)
            for span in spans:
                ties = tie_parts(geo, prof, span, 1 << d_sig)
                S = geo["segments"] if span else 1
                print(f"N* = 2^{d_sig:<2} {prof:<6} span {'on ' if span else 'off'} {plan_digest(geo, sites, ties, prof, S, geo['d_seg'])}")
    print()
    # Honest completeness: per site (1 - 2^-g)^(2^(g+7)) <= exp(-2^7).
    # Capped searches: PIOP sites, tied-part sites (at most 2 parts x 4 segments, each a tie, an
    # eq_Y and at most 28 rounds), Akita's 461.
    part_sites = 8 * (2 + chain_rounds(D_SEG_MAX, 2))
    worst = max(len(piop_sites(geometry(d))) for d in range(3, 16)) + part_sites + AKITA_SITES_2_15
    print(
        f"Honest failure: per site <= exp(-128) = 2^{-128 / math.log(2):.1f};"
        f" at most {worst} capped searches per proof (PIOP + {part_sites} tied-part + {AKITA_SITES_2_15} Akita):"
        f" <= 2^{math.log2(worst) - 128 / math.log(2):.1f}"
    )


def print_details(d_sig: int) -> None:
    geo = geometry(d_sig)
    print(f"== Every site at N* = 2^{d_sig} (default: Slim, span on) ==")
    print("#    class         i   j   nu      e_t numerator     g  width")
    ties = tie_parts(geo, "slim", True, 1 << d_sig)
    idx = 0
    for p, t in enumerate(ties):
        print(f"{idx:<4} tie          {p:>2}  0   {t['nu_s'] + SET['public_vars']:<6}  {t['tie']}/2^128  {t['g']:>2}  {t['w']:>2}")
        print(f"{idx + 1:<4} chain.eq     {p:>2}  0   {OUT_VARS + t['span_log']:<6}  {t['eq']}/2^128  {t['g_eq']:>2}  {width(t['g_eq']):>2}")
        idx += 2
        for k in range(t["rounds"]):
            print(f"{idx:<4} chain.round  {p:>2}  {k:>2}  2       {t['round']}/2^128  {t['g_round']:>2}  {width(t['g_round']):>2}")
            idx += 1
    for cls, i, j, nu, num, den in piop_sites(geo):
        g = site_bits(cls, num, den)
        e = "2^-128" if cls == "proj.psi" else (f"{num}/|K|" if cls != "lookup.x" else "((L+M-1)/|K|)^2")
        print(f"{idx:<4} {cls:<12} {i:>2}  {j:>2}  {str(nu) if nu is not None else '-':<6}  {e:<16}  {g:>2}  {width(g):>2}")
        idx += 1


def with_gammas(gammas: tuple, f):
    """Evaluate f() with STAGE_GAMMAS set to gammas."""
    global STAGE_GAMMAS
    saved = STAGE_GAMMAS
    STAGE_GAMMAS = gammas
    try:
        return f()
    finally:
        STAGE_GAMMAS = saved


def self_check() -> None:
    # bit_length(4) = bit_length(5): a degree-4 and a degree-5 round grind the same bits.
    assert grind_bits(4, K) == grind_bits(5, K) == 3
    # Aerie's grind_bits(L) = bit_length(L) for |K| sites (crates/aerie-falcon/src/protocol/grinding.rs).
    for nu, want in ((1, 1), (3, 2), (8, 4), (16, 5), (25, 5), (44, 6)):
        assert grind_bits(nu, K) == want, (nu, grind_bits(nu, K))
    assert all(grind_bits(nu, K) == nu.bit_length() for nu in range(1, 5000))
    # GF(2^128): ceil(log2 nu), 1/2^128 needs no work.
    assert grind_bits(1, F2_128) == 0 and grind_bits(95, F2_128) == 7
    # proj.psi is a byte challenge with the exact bound 2^-128; the schedule assigns it one
    # bit, so its weight is 2^-129 and the |K| ledger is unchanged.
    assert site_bits("proj.psi", 1, F2_128) == 1
    assert site_bits("lookup.x", 1, F2_128) == 0
    # Version 5 (docs/parity-chain.md): a span of four full segments under Slim.
    t = tie_terms(13, 2, 15, "slim")
    assert (t["tie"], t["g"], t["eq"], t["g_eq"], t["round"], t["g_round"], t["rounds"]) == (
        900, 10, 700, 10, 100, 7, 28), t
    assert chain_rounds(13, 0) == 26 and chain_rounds(3, 0) == 16 and chain_rounds(8, 0) == 21
    # The spec's ledger: GKR 1,026/|K| at 2^15 and 391/|K| at N* = 8; L at 2^15 (the
    # projection of v: V_lo, V_hi leave the main stack, [K_0 | K_1] is a fifth stack).
    g15 = geometry(15)
    assert g15["L"] == 79_970_304 and g15["D"] == 26
    assert sorted(set(g15["trees"]), reverse=True) == [26, 23, 22, 18, 16, 14]
    for d_sig, want in ((15, 1026), (3, 391)):
        geo = geometry(d_sig)
        gkr = sum(nu for (cls, _i, _j, nu, _n, _d) in piop_sites(geo) if cls.startswith("lookup") and nu is not None)
        assert gkr == want, (d_sig, gkr)
    # list sizes: floor(1/(2 * 0.02 * sqrt(rho))).
    assert [list_size(r) for r in (1, 2, 3)] == [35, 50, 70]
    # sponge rows (agg_keccak::prover::capacity_log).
    assert capacity_log(1 << 15) == 15 and capacity_log(1 << 10) == 11 and capacity_log(8) == 4
    # Relation H-exact (docs/h-exact.md): the stage-2 batching site has 9 coefficients
    # (leaf6[U_hi], leaf8[K_2], leaf8[K_3]), g = 4; gamma^(3) has g = 6 (50, 56 < 64). The
    # totals and the Slim/span-on and mode-B digests at 2^15.
    assert grind_bits(7, K) == 3 and grind_bits(8, K) == grind_bits(9, K) == 4
    assert grind_bits(48, K) == grind_bits(49, K) == grind_bits(55, K) == 6
    exact = RELATION_GAMMAS["h-exact"]
    for d_sig, bits, work in ((3, 1818, 1350), (10, 2926, 1964), (13, 3664, 2342), (15, 4203, 2646)):
        s = with_gammas(exact, lambda: summarize(piop_sites(geometry(d_sig))))
        assert (s["bits"], s["work"]) == (bits, work), (d_sig, s)
    g15 = geometry(15)
    digest = with_gammas(exact, lambda: plan_digest(g15, piop_sites(g15), [], "modeB", 0, 0))
    assert digest.startswith("1634c0b8"), digest
    digest = with_gammas(exact, lambda: plan_digest(
        g15, piop_sites(g15), tie_parts(g15, "slim", True, 1 << 15), "slim", g15["segments"], g15["d_seg"]))
    assert digest.startswith("ef971095"), digest
    # Relation H-FIPS: the same difficulties at every site, so its plans equal H-exact's at
    # every size.
    for gammas in (RELATION_GAMMAS["h-fips"],):
        for d_sig in (3, 10, 13, 15):
            geo = geometry(d_sig)
            a = with_gammas(gammas, lambda: summarize(piop_sites(geo)))
            b = with_gammas(exact, lambda: summarize(piop_sites(geo)))
            assert (a["bits"], a["work"]) == (b["bits"], b["work"]), (d_sig, a, b)


def configure(set_name: str, relation: str) -> None:
    """Select the parameter set and the relation."""
    global SET, STAGE_DEGREES, STAGE_GAMMAS, CARRY2_SLOTS
    SET = SETS[set_name]
    STAGE_DEGREES = SET["degrees"]
    CARRY2_SLOTS = None
    if SET["gammas"] is None:
        STAGE_GAMMAS = RELATION_GAMMAS[relation]
    elif relation == "h-fips":
        STAGE_GAMMAS = LANE_FIPS[set_name]["gammas"]
        CARRY2_SLOTS = LANE_FIPS[set_name]["carry2_slots"]
    else:
        STAGE_GAMMAS = SET["gammas"]


def self_check_sets() -> None:
    """The per-set plans of ML-DSA-65 and ML-DSA-87 (pinned in crates/agg-piop/src/pow.rs)."""
    try:
        configure("65", "h-exact")
        g = geometry(15)
        assert (g["d_main"], g["d"], g["D"]) == (23, (20, 15, 14), 26), g
        assert sorted(set(g["trees"]), reverse=True) == [26, 24, 23, 22, 17, 16, 14], g["trees"]
        assert [grind_bits(n, K) for n in STAGE_GAMMAS] == [6, 5, 7]
        assert chain_rounds(13, 0) == 19 + 5 + 2
        configure("87", "h-exact")
        g = geometry(15)
        assert (g["d_main"], g["d"], g["D"]) == (23, (20, 15, 14), 26), g
        assert sorted(set(g["trees"]), reverse=True) == [26, 25, 24, 23, 22, 17, 16, 14], g["trees"]
        assert [grind_bits(n, K) for n in STAGE_GAMMAS] == [6, 5, 8]
        # H-FIPS (pinned in crates/agg-piop/src/pow.rs, docs/h-exact.md): the same sites as
        # H-exact; only gamma^(3) of 65 and one lookup batching site grind one more bit.
        for name, want in (("65", {3: (190, 1763), 10: (311, 2868), 15: (451, 4145)}),
                           ("87", {3: (190, 1764), 10: (311, 2874), 15: (451, 4149)})):
            for d_sig, (n_sites, n_bits) in want.items():
                configure(name, "h-exact")
                hx = piop_sites(geometry(d_sig))
                configure(name, "h-fips")
                hf = piop_sites(geometry(d_sig))
                total = summarize(hf)["bits"]
                assert (len(hf), total) == (n_sites, n_bits), (name, d_sig, len(hf), total)
                assert len(hx) == len(hf)
                for a, b in zip(hf, hx):
                    assert a[:3] == b[:3], (a, b)
                    ga, gb = site_bits(a[0], a[4], a[5]), site_bits(b[0], b[4], b[5])
                    if ga != gb:  # the lookup's X site keeps g = 0 although L grows
                        assert a[0] in ("stage.gamma", "lookup.gamma") and ga == gb + 1, (a, b)
    finally:
        configure("44", "h-fips")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--details", type=int, metavar="D_SIG", help="print every site of N* = 2^D_SIG")
    parser.add_argument("--relation", choices=sorted(RELATION_GAMMAS), default="h-fips",
                        help="the relation whose stage batching vectors to use (default h-fips)")
    parser.add_argument("--set", choices=sorted(SETS), default="44",
                        help="the ML-DSA parameter set (default 44)")
    args = parser.parse_args()
    self_check()
    self_check_sets()
    configure(args.set, args.relation)
    if args.details is not None:
        print_details(args.details)
    else:
        print_summary(args)


if __name__ == "__main__":
    main()
