#!/usr/bin/env python3
"""Accepted support of Akita's sparse fold-challenge families (docs/soundness.md).

Run (isolated mode, standard library only):

    python3 -I docs/scripts/fold_support.py [AKITA_CHECKOUT]

The script reads only the two operator-norm certificates (JSON) of Akita revision e023b6a;
it does not run code from Akita. Without an argument it reads the copies in
docs/scripts/akita-certs/ (Akita is MIT/Apache-2.0 licensed); with AKITA_CHECKOUT it reads
them from that tree. Either way each file must match the SHA-256 pinned below.

What it computes, exactly (integers and fractions):

* the size C(D, w) C(w, a) 2^w of every family (D, a, b), w = a + b, that the catalog test
  crates/agg-pcs-config/tests/all/akita_fold_support.rs finds in our rows, and of Akita's
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

import hashlib
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
    # (file in the Akita tree, SHA-256, D, a, b, runtime threshold, rounding margin)
    (
        "scripts/operator_norm/d128/cert_d128_w31_gamma13.json",
        "e9f255359549bf3b7726eb5b71e80b6d8aae29160832ed06f920194fa34d41f7",
        128, 31, 0, 13, 351,
    ),
    (
        "scripts/operator_norm/d64/cert_d64_a31_b11_gamma18.json",
        "cd967cd643a21f89543b5340d54830837ce73beb4c91bb154baf6e5942e9e230",
        64, 31, 11, 18, 600,
    ),
]
VENDORED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "akita-certs")


def family_size(d: int, a: int, b: int) -> int:
    w = a + b
    return math.comb(d, w) * math.comb(w, a) * 2**w


def log2(x: int | Fraction) -> float:
    x = Fraction(x)
    return math.log2(x.numerator) - math.log2(x.denominator)


def certificate_path(argv: list[str], rel: str) -> str:
    """The certificate in the given Akita checkout, else the vendored copy."""
    if len(argv) > 1:
        return os.path.join(argv[1], rel)
    return os.path.join(VENDORED, os.path.basename(rel))


def main() -> None:
    print("family (D, a, b)      log2 |C|    >= 2^128")
    for d, a, b in FAMILIES:
        s = family_size(d, a, b)
        print(f"({d:4}, {a:2}, {b:2})     {log2(s):10.4f}    {s >= 2**TARGET}")
        assert s >= 2**TARGET
    # The ladder's floor(log2) values and the edges: one coefficient fewer is below.
    assert family_size(64, 30, 10) < 2**TARGET
    assert family_size(128, 30, 0) < 2**TARGET

    source = sys.argv[1] if len(sys.argv) > 1 else "vendored copies, docs/scripts/akita-certs"
    print(f"\ncertified operator-norm floors (Akita {REVISION}, {source})")
    for rel, digest, d, a, b, thr, margin in CERTIFICATES:
        with open(certificate_path(sys.argv, rel), "rb") as fh:
            raw = fh.read()
        assert hashlib.sha256(raw).hexdigest() == digest, f"{rel} differs from Akita {REVISION}"
        cert = json.loads(raw.decode("ascii"))
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
