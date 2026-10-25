#!/usr/bin/env python3
"""Proof-of-work schedule of docs/soundness.md (normative), recomputed from first principles.

Run (isolated mode, standard library only):

    python3 -I docs/scripts/pow_plan.py              # summary tables (committed in the doc)
    python3 -I docs/scripts/pow_plan.py --details 15 # every site of N* = 2^15, default config

What it computes, with exact integer arithmetic:

* the ordered PIOP field-challenge sites of the mode-A/mode-B transcript (spec Section 10,
  crates/agg-piop/src/transcript.rs Registry::new) for every N* = 2^3 .. 2^15, each with its
  round-by-round numerator nu_t, its assigned bound e_t (>= nu_t / |F|), the difficulty
  g_t = min{g >= 0 : 2^128 e_t <= 2^g} (Aerie Eq. (17)) and the nonce width g_t + 7
  (0 when g_t = 0: no nonce);
* the Flock tie site of every tied part (docs/bridge.md 5c) for part sizes d = 3 .. 13 and
  span lengths S = 1, 2, 4, with the Johnson-list factors of the two Ligerito commitments;
* the packed nonce stream size, the expected honest work sum 2^g, the honest failure bound,
  the union-bound ledger before work, and the plan digest test vectors (encoding of
  docs/soundness.md Section 6.4);
* the audit of Flock's outer (pre-binding) challenge sites against the Flock paper's
  list-size accounting (docs/soundness.md Section 4.3).
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
SLACK = 7  # nonce width = g + 7 (Aerie C.1, Akita (146))
M_TABLE = 2**16  # |T_16|
CHALLENGES = 2  # LOOKUP_CHALLENGES
STAGE_DEGREES = (4, 5, 3)
STAGE_GAMMAS = (26, 8, 44)
SPONGE_K_LOG = 19
MIN_DENSE_M = 22
RECT_MIN_ROWS_LOG = 11
D_SEG_MAX = 13
OUTPUT_WORDS_LOG2 = 5
PROFILE_ID = {"fast": 1, "slim": 4, "slim8": 5, "modeB": 3}
# Level-0 Ligerito rate exponent r (rho = 2^-r) per profile; eta = 0.02 at level 0 in every
# shipped and derived config (vendor/flock-core/configs/ligerito/m*_{fast,slim,slim8}.toml).
L0_RATE = {"fast": 1, "slim": 2, "slim8": 3}
# Measured cost of one Blake2b predicate preview (Akita preview_grinding_predicate,
# docs/lookup.md Section 4.4): 724 ns serial on this machine.
PREVIEW_NS = 724.0

CLASS_ID = {
    "lookup.x": 1,
    "lookup.mu": 2,
    "lookup.gamma": 3,
    "lookup.round": 4,
    "tau": 5,
    "stage.gamma": 6,
    "stage.round": 7,
    "tie": 8,
}


# Akita's capped transcript proof-of-work searches at 2^15 (soundness.md 3.7).
AKITA_SITES_2_15 = 459


def grind_bits(num: int, den: int) -> int:
    """Least g >= 0 with 2^128 * num / den <= 2^g (Aerie Eq. (17), exact)."""
    if num <= 0:
        return 0
    g = 0
    while (num << TARGET) > (den << g):
        g += 1
    return g


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
    d_main = 10 + d_sig
    d1 = 13 + max(0, d_sig - 9)
    d2 = 13 + max(0, d_sig - 13)
    d3 = 12 + max(0, d_sig - 13)
    stacks = [d_main + 2, d1 + 3, d2 + 3, d3]
    trees = (stacks + [16]) * CHALLENGES
    lookups = 4 * 2**d_main + 8 * 2**d1 + 8 * 2**d2 + 2**d3
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
    """The PIOP field-challenge sites in registry order: (class, i, j, nu, e_num, e_den)."""
    sites = []
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


def tie_terms(d: int, span_log: int, nu_s: int, profile: str) -> dict:
    """Tie identities of one tied part (docs/bridge.md 5c) and their list-weighted bounds."""
    lst = list_size(L0_RATE[profile])
    ls = lu = lst
    blocks_log = max(0, d - 9)
    tie1 = d + span_log + 5  # sponge inputs vs level-1 inputs (two lane groups): pairs of lists
    tie2 = blocks_log + span_log + 9  # parities vs copies inside the union (two lane groups)
    out = OUTPUT_WORDS_LOG2 + span_log  # output columns vs the public table of Y
    pub = nu_s + 3  # sponge public region vs the verifier's table
    plain = 2 * tie1 + 2 * tie2 + out + pub
    listed = 2 * ls * lu * tie1 + 2 * lu * tie2 + lu * out + ls * pub
    # union over list pairs: tie2 and out depend only on g, pub only on f (soundness.md 3.6)
    tight = ls * lu * tie1 + lu * max(tie2, out) + ls * pub
    g = grind_bits(listed, F2_128)
    return {
        "d": d,
        "span_log": span_log,
        "nu_s": nu_s,
        "list": lst,
        "terms": (tie1, tie2, out, pub),
        "plain": plain,
        "listed": listed,
        "tight": tight,
        "g": g,
        "g_plain": grind_bits(plain, F2_128),
        "g_tight": grind_bits(tight, F2_128),
        "w": width(g),
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
    """SHA3-256 test vector of the canonical plan encoding (docs/soundness.md Section 6.4)."""
    entries = []
    for t in ties:  # tie sites first: they precede the lookup (event 6)
        entries.append((CLASS_ID["tie"], len(entries), 0, t["g"], t["w"]))
    for cls, i, j, nu, num, den in sites:
        g = grind_bits(num, den)
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
        g = grind_bits(num, den)
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
        g = grind_bits(num, den)
        key = (cls, nu, num if cls != "lookup.x" else None, g)
        if key not in rows:
            rows[key] = 0
            order.append(key)
        rows[key] += 1
    return [(k, rows[k]) for k in order]


def fmt_e(cls, nu, num) -> str:
    if cls == "lookup.x":
        return "((L+M-1)/|K|)^2"
    return f"{num}/|K|"


def print_summary(args) -> None:
    print("== PIOP sites per N* (mode A and mode B; registry order) ==")
    print("d_sig  N*      d_main  D   L            X-site log2   sites  nonces  bits  bytes  work(sum 2^g)  est.ms  max weight  ledger(before work)")
    for d_sig in range(3, 16):
        geo = geometry(d_sig)
        sites = piop_sites(geo)
        s = summarize(sites)
        x_log2 = math.log2(Fraction(sites[0][4], sites[0][5]))
        assert s["max_weight_log2"] <= -TARGET, "a site exceeds 2^-128"
        print(
            f"{d_sig:>5}  {1 << d_sig:>6}  {geo['d_main']:>6}  {geo['D']:>2}  {geo['L']:>11}  {x_log2:>12.1f}  {len(sites):>5}  {s['nonces']:>6}  {s['bits']:>4}  {s['bytes']:>5}  {s['work']:>13}  {s['work'] * PREVIEW_NS / 1e6:>6.2f}  2^{s['max_weight_log2']:.2f}  {s['union_nu']}/|K| = 2^{math.log2(Fraction(s['union_nu'], K)):.1f}"
        )
    print()
    print("== PIOP site classes at N* = 2^15 (d_main = 25, D = 27) ==")
    print("class         count  nu      e_t               g   nonce bits")
    geo = geometry(15)
    for (cls, nu, num, g), count in class_table(piop_sites(geo)):
        print(f"{cls:<12}  {count:>5}  {str(nu) if nu is not None else 'joint':<6}  {fmt_e(cls, nu, num):<16}  {g:>2}  {width(g):>3}")
    print()
    print("== Old ledger (spec 12.1, union bound per instance) vs round-by-round numerators, before work ==")
    for d_sig in (10, 13, 15):
        geo = geometry(d_sig)
        dm, D = geo["d_main"], geo["D"]
        gkr = 3 * D * (D - 1) // 2 + 2 * (D - 1) + 1
        old = 12 * dm + 19 * dm + 4 * dm + 62 * dm + 3 + gkr
        new = 12 * dm + dm + 3 + gkr
        print(
            f"N* = 2^{d_sig}: spec 12.1 {old}/|K| = 2^{math.log2(Fraction(old, K)):.1f};"
            f" round-by-round sum {new}/|K| = 2^{math.log2(Fraction(new, K)):.1f}"
            f" (stages {12 * dm}, tau {dm}, gammas 3, GKR {gkr})"
        )
    print()
    theta = Fraction(1, math.comb(8192, 14)) * 2 + Fraction(1, math.comb(4096, 18))
    print("== Seed site sd (event 3), no nonce ==")
    for prof in ("fast", "slim", "slim8"):
        ls = list_size(L0_RATE[prof])
        e = theta * ls
        print(f"{prof:<6} theta = 2^{math.log2(theta):.2f}, times L_s = {ls}: 2^{math.log2(e):.2f}, g = {grind_bits(e.numerator, e.denominator)}")
        # sum over segment checks of L_s^(parts in the segment) theta (soundness.md 3.5);
        # mode A admits at most two parts per segment.
        rows = (("span off, 4 segments x 1 part", 4 * ls), ("1 segment x 2 parts", ls * ls),
                ("4 segments x 2 parts", 4 * ls * ls), ("1 segment x 3 parts (refused)", ls ** 3))
        for label, mult in rows:
            e = theta * mult
            print(f"       {label:<31} 2^{math.log2(e):.2f}, g = {grind_bits(e.numerator, e.denominator)}")
    print()
    print("== Tie site of one tied part (GF(2^128), docs/bridge.md 5c), full parts n = 2^d S ==")
    print("profile  d   S  nu_s  terms(tie1,tie2,out,pub)  plain  g_plain  list  listed    g   nonce  tight(RBR)  g_tight")
    for prof in ("fast", "slim", "slim8"):
        for d in range(3, 14):
            for S in (1, 2, 4):
                if S > 1 and d != D_SEG_MAX:
                    continue
                span_log = (S - 1).bit_length()
                t = tie_terms(d, span_log, capacity_log((1 << d) * S), prof)
                print(
                    f"{prof:<7} {d:>2}  {S:>2}  {t['nu_s']:>4}  {str(t['terms']):<24}  {t['plain']:>5}  {t['g_plain']:>7}  {t['list']:>4}  {t['listed']:>7}  {t['g']:>3}  {t['w']:>5}  {t['tight']:>10}  {t['g_tight']:>7}"
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
    print("== Totals at N = N* (bytes of the PIOP packed stream + tie nonces; expected work) ==")
    print("d_sig  profile  span  parts  tie g     PIOP bytes  tie bytes  total bytes  B/sig    PIOP work  tie work")
    for d_sig in range(3, 16):
        geo = geometry(d_sig)
        s = summarize(piop_sites(geo))
        for prof in ("fast", "slim", "slim8"):
            spans = (True, False) if geo["segments"] > 1 else (True,)
            for span in spans:
                ties = tie_parts(geo, prof, span, 1 << d_sig)
                tie_bytes = 4 * len(ties)
                tie_work = sum(1 << t["g"] for t in ties)
                tot = s["bytes"] + tie_bytes
                print(
                    f"{d_sig:>5}  {prof:<7}  {'on ' if span else 'off'}   {len(ties):>5}  {','.join(str(t['g']) for t in ties):<8}  {s['bytes']:>10}  {tie_bytes:>9}  {tot:>11}  {tot / (1 << d_sig):>6.3f}  {s['work']:>9}  {tie_work:>8}"
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
    # Capped searches: PIOP sites, tie sites (at most 2 parts x 4 segments), Akita's 459.
    worst = max(len(piop_sites(geometry(d))) for d in range(3, 16)) + 8 + AKITA_SITES_2_15
    print(
        f"Honest failure: per site <= exp(-128) = 2^{-128 / math.log(2):.1f};"
        f" at most {worst} capped searches per proof (PIOP + 8 tie + {AKITA_SITES_2_15} Akita):"
        f" <= 2^{math.log2(worst) - 128 / math.log(2):.1f}"
    )


def print_details(d_sig: int) -> None:
    geo = geometry(d_sig)
    print(f"== Every site at N* = 2^{d_sig} (default: Slim, span on) ==")
    print("#    class         i   j   nu      e_t numerator     g  width")
    ties = tie_parts(geo, "slim", True, 1 << d_sig)
    idx = 0
    for t in ties:
        print(f"{idx:<4} tie          {idx:>2}  0   {t['tight']:<6}  {t['listed']}/2^128  {t['g']:>2}  {t['w']:>2}")
        idx += 1
    for cls, i, j, nu, num, den in piop_sites(geo):
        g = grind_bits(num, den)
        e = f"{num}/|K|" if cls != "lookup.x" else "((L+M-1)/|K|)^2"
        print(f"{idx:<4} {cls:<12} {i:>2}  {j:>2}  {str(nu) if nu is not None else '-':<6}  {e:<16}  {g:>2}  {width(g):>2}")
        idx += 1


def self_check() -> None:
    # Aerie's grind_bits(L) = bit_length(L) for |K| sites (crates/aerie-falcon/src/protocol/grinding.rs).
    for nu, want in ((1, 1), (3, 2), (8, 4), (16, 5), (25, 5), (44, 6)):
        assert grind_bits(nu, K) == want, (nu, grind_bits(nu, K))
    assert all(grind_bits(nu, K) == nu.bit_length() for nu in range(1, 5000))
    # GF(2^128): ceil(log2 nu), 1/2^128 needs no work.
    assert grind_bits(1, F2_128) == 0 and grind_bits(95, F2_128) == 7
    # docs/bridge.md 5c: 95 at four full segments without lists.
    assert tie_terms(13, 2, 15, "slim")["plain"] == 95
    # docs/bridge.md 5b: 83 at a full segment.
    assert tie_terms(13, 0, 13, "slim")["plain"] == 83
    # spec 12.1: GKR 1,106/|K| at 2^15 and 391/|K| at N* = 8; L at 2^15.
    g15 = geometry(15)
    assert g15["L"] == 138_690_560 and g15["D"] == 27
    assert sorted(set(g15["trees"]), reverse=True) == [27, 22, 18, 16, 14]
    for d_sig, want in ((15, 1106), (3, 391)):
        geo = geometry(d_sig)
        gkr = sum(nu for (cls, _i, _j, nu, _n, _d) in piop_sites(geo) if cls.startswith("lookup") and nu is not None)
        assert gkr == want, (d_sig, gkr)
    # list sizes: floor(1/(2 * 0.02 * sqrt(rho))).
    assert [list_size(r) for r in (1, 2, 3)] == [35, 50, 70]
    # sponge rows (agg_keccak::prover::capacity_log).
    assert capacity_log(1 << 15) == 15 and capacity_log(1 << 10) == 11 and capacity_log(8) == 4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--details", type=int, metavar="D_SIG", help="print every site of N* = 2^D_SIG")
    args = parser.parse_args()
    self_check()
    if args.details is not None:
        print_details(args.details)
    else:
        print_summary(args)


if __name__ == "__main__":
    main()
