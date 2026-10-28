#!/usr/bin/env python3
"""Accepted support of Akita's sparse fold-challenge families (docs/soundness.md).

Run (isolated mode, standard library only):

    python3 -I docs/scripts/fold_support.py [AKITA_CHECKOUT]

AKITA_CHECKOUT defaults to the cargo git checkout of the pinned revision,
~/.cargo/git/checkouts/akita-*/e023b6a. The script reads only the two operator-norm
certificates (JSON) of that tree; it does not run code from it.

What it computes, exactly (integers and fractions):

* the size C(D, w) C(w, a) 2^w of every family (D, a, b), w = a + b, that the catalog test
  crates/agg-pcs-config/tests/akita_fold_support.rs finds in our rows, and of Akita's
  production ladder, against 2^128;
* for the two certified operator-norm policies, N_raw (checked against the family size)
  and the accepted floor ceil(N_raw p0), where p0 is the certificate's exact lower bound on
  the accepted fraction of the strict runtime predicate (Akita: conditioned on
  success the sampler's output is uniform on the accepted set, so this is the support).

The certificates' own proofs (moments, dual polynomial, Bernstein or Sturm positivity) are
checked by Akita's exact checkers, scripts/operator_norm/d64/check_cert.py and
scripts/operator_norm/d128/check_cert.py of the same tree (both print ALL CHECKS PASS at
e023b6a).
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
from fractions import Fraction

TARGET = 128
REVISION = "e023b6a"

# (D, count_pm1, count_pm2): the families of our catalogs (akita_fold_support.rs) and the
# rest of Akita's production ladder (akita-challenges config.rs).
FAMILIES = [
    (64, 31, 10),
    (128, 31, 0),
    (256, 23, 0),
    (512, 19, 0),
    (1024, 16, 0),
    (2048, 14, 0),
    (64, 31, 11),
]

CERTIFICATES = [
    # (file, D, a, b, runtime threshold, rounding margin)
    ("scripts/operator_norm/d128/cert_d128_w31_gamma13.json", 128, 31, 0, 13, 351),
    ("scripts/operator_norm/d64/cert_d64_a31_b11_gamma18.json", 64, 31, 11, 18, 600),
]


def family_size(d: int, a: int, b: int) -> int:
    w = a + b
    return math.comb(d, w) * math.comb(w, a) * 2**w


def log2(x: int | Fraction) -> float:
    x = Fraction(x)
    return math.log2(x.numerator) - math.log2(x.denominator)


def akita_root(argv: list[str]) -> str:
    if len(argv) > 1:
        return argv[1]
    pattern = os.path.expanduser(f"~/.cargo/git/checkouts/akita-*/{REVISION}")
    hits = sorted(glob.glob(pattern))
    if not hits:
        sys.exit(f"no Akita checkout matches {pattern}; pass the path")
    return hits[0]


def main() -> None:
    print("family (D, a, b)      log2 |C|    >= 2^128")
    for d, a, b in FAMILIES:
        s = family_size(d, a, b)
        print(f"({d:4}, {a:2}, {b:2})     {log2(s):10.4f}    {s >= 2**TARGET}")
        assert s >= 2**TARGET
    # The ladder's floor(log2) values and the edges: one coefficient fewer is below.
    assert family_size(64, 30, 10) < 2**TARGET
    assert family_size(128, 30, 0) < 2**TARGET

    root = akita_root(sys.argv)
    print(f"\ncertified operator-norm floors ({root})")
    for rel, d, a, b, thr, margin in CERTIFICATES:
        with open(os.path.join(root, rel), encoding="ascii") as fh:
            cert = json.load(fh)
        p = cert["params"]
        fp = cert["fixed_point_containment"]
        assert (p["d"], p["a_mag1"], p["b_mag2"]) == (d, a, b), rel
        assert (fp["runtime_strict_threshold"], fp["rounding_margin_units"]) == (thr, margin)
        assert (fp["fractional_bits"], fp["root_coordinate_error_units"]) == (48, 4)
        n_raw = int(cert["N_raw"])
        assert n_raw == family_size(d, a, b), rel
        p0 = Fraction(cert["claimed"]["p0"])
        acc = n_raw * p0
        floor = -((-acc.numerator) // acc.denominator)
        print(
            f"D {d:3} ({a}, {b}) threshold {thr}: p0 = {float(p0):.9f}, "
            f"ceil(N_raw p0) = {floor}, log2 = {log2(acc):.6f}, raw {log2(n_raw):.4f}"
        )
        assert acc >= 2**TARGET
    print("\nevery family clears 2^128 after rejection")


if __name__ == "__main__":
    main()
