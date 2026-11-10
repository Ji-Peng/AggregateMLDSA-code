#!/usr/bin/env python3
"""Pin the artifact to the current revision in one step.

Builds the release binary at the checked-out revision, runs the 1K gates for every
parameter set and relation (H-FIPS and the opt-in H-exact) with `--write-expected` (which rewrites
`expected/hashes.json`), then the `--hash-field fp` gates with `--write-expected` (which rewrite
`expected/hashes-fp.json` for its default relations; `--skip-fp` leaves that file alone), and
sets the in-repo commit of `sources.lock` to `HEAD`. Both files get the same commit and records
metadata. Run it on a clean tree at the submission revision, then commit the changed files.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ARTIFACT,
    EXPECTED,
    RELATIONS,
    REPO,
    git_dirty,
    git_rev,
    read_json,
    run,
)

LOCK = ARTIFACT / "sources.lock"
HASHES = EXPECTED / "hashes.json"
HASHES_FP = EXPECTED / "hashes-fp.json"


def pinned_entries(document):
    """{(set, relation): (bytes, sha256)} of the 1K entries of a hashes.json document."""
    entries = {}
    for params, sizes in document.get("aggregates", {}).items():
        for relation, entry in sizes.get("1024", {}).items():
            entries[(params, relation)] = (entry.get("bytes"), entry.get("sha256"))
    return entries


def show(path, before):
    """Print each pinned entry of a hashes file with its state against `before`."""
    after = pinned_entries(read_json(path))
    print(f"{path.name}:")
    for key in sorted(after):
        old = before.get(key)
        state = "new" if old is None else ("unchanged" if old == after[key] else "CHANGED")
        print(f"  ML-DSA-{key[0]} {key[1]:<8} {after[key][0]} B {after[key][1][:16]}  {state}")


def pin_lock(commit):
    """Rewrite the in-repo source block of sources.lock; comments and order are kept."""
    text = LOCK.read_text(encoding="utf-8")
    blocks = re.split(r"(?m)^(?=\[\[source\]\])", text)
    hits = 0
    for index, block in enumerate(blocks):
        if not re.search(r"(?m)^in_repo\s*=\s*true\s*$", block):
            continue
        hits += 1
        block = re.sub(r'(?m)^commit\s*=\s*"[^"]*"', f'commit = "{commit}"', block)
        blocks[index] = block
    if hits != 1:
        raise SystemExit(f"{LOCK}: expected one in_repo source, found {hits}")
    LOCK.write_text("".join(blocks), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-dirty", action="store_true", help="run on a dirty tree")
    parser.add_argument("--no-build", action="store_true", help="reuse artifact/build/bin/agg")
    parser.add_argument(
        "--skip-fp", action="store_true", help="leave expected/hashes-fp.json unchanged"
    )
    args = parser.parse_args()
    head = git_rev(REPO)
    if not head:
        raise SystemExit("freeze needs a git checkout")
    if git_dirty(REPO) and not args.allow_dirty:
        raise SystemExit("the tree has uncommitted changes; commit them or pass --allow-dirty")
    scripts = ARTIFACT / "scripts"
    before = pinned_entries(read_json(HASHES)) if HASHES.exists() else {}
    before_fp = pinned_entries(read_json(HASHES_FP)) if HASHES_FP.exists() else {}
    if not args.no_build:
        run(
            ["python3", scripts / "build.py", "--skip-aerie", "--skip-natver"],
            capture=False,
        )
    gates = run(
        [
            "python3",
            scripts / "gates.py",
            "--write-expected",
            "--no-sde",
            "--relations",
            ",".join(RELATIONS),
        ],
        check=False,
        capture=False,
    )
    if gates.returncode != 0:
        raise SystemExit("gates failed; sources.lock is unchanged (inspect expected/hashes.json)")
    pinned = [HASHES]
    if not args.skip_fp:
        gates_fp = run(
            [
                "python3",
                scripts / "gates.py",
                "--write-expected",
                "--hash-field",
                "fp",
            ],
            check=False,
            capture=False,
        )
        if gates_fp.returncode != 0:
            raise SystemExit(
                "fp gates failed; sources.lock is unchanged (inspect expected/hashes-fp.json)"
            )
        pinned.append(HASHES_FP)
    show(HASHES, before)
    if not args.skip_fp:
        show(HASHES_FP, before_fp)
    pin_lock(head)
    names = " and ".join(f"expected/{path.name}" for path in pinned)
    print(f"pinned sources.lock and {names} to {head}")
    print("commit artifact/sources.lock and artifact/" + " and artifact/".join(
        f"expected/{path.name}" for path in pinned
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
