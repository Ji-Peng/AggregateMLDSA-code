#!/usr/bin/env python3
"""Pin the artifact to the current revision in one step.

Builds the release binary at the checked-out revision, runs the 1K gates for every
parameter set and relation with `--write-expected` (which rewrites
`expected/hashes.json`), and sets the in-repo commit of `sources.lock` to `HEAD`.
Run it on a clean tree at the submission revision, then commit the two files.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ARTIFACT, EXPECTED, REPO, git_dirty, git_rev, read_json, run  # noqa: E402

LOCK = ARTIFACT / "sources.lock"
HASHES = EXPECTED / "hashes.json"


def pinned_entries(document):
    """{(set, relation): (bytes, sha256)} of the 1K entries of a hashes.json document."""
    entries = {}
    for params, sizes in document.get("aggregates", {}).items():
        for relation, entry in sizes.get("1024", {}).items():
            entries[(params, relation)] = (entry.get("bytes"), entry.get("sha256"))
    return entries


def pin_lock(commit, archive_url):
    """Rewrite the in-repo source block of sources.lock; comments and order are kept."""
    text = LOCK.read_text(encoding="utf-8")
    blocks = re.split(r"(?m)^(?=\[\[source\]\])", text)
    hits = 0
    for index, block in enumerate(blocks):
        if not re.search(r"(?m)^in_repo\s*=\s*true\s*$", block):
            continue
        hits += 1
        block = re.sub(r'(?m)^commit\s*=\s*"[^"]*"', f'commit = "{commit}"', block)
        if archive_url:
            block = re.sub(r'(?m)^url\s*=\s*"[^"]*"', f'url = "{archive_url}"', block)
            block = re.sub(
                r'(?m)^archive_url\s*=\s*"[^"]*"', f'archive_url = "{archive_url}"', block
            )
        blocks[index] = block
    if hits != 1:
        raise SystemExit(f"{LOCK}: expected one in_repo source, found {hits}")
    LOCK.write_text("".join(blocks), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-dirty", action="store_true", help="run on a dirty tree")
    parser.add_argument("--no-build", action="store_true", help="reuse artifact/build/bin/agg")
    parser.add_argument(
        "--archive-url",
        help="also replace the in-repo source URL by this anonymous archive URL",
    )
    args = parser.parse_args()
    head = git_rev(REPO)
    if not head:
        raise SystemExit("freeze needs a git checkout")
    if git_dirty(REPO) and not args.allow_dirty:
        raise SystemExit("the tree has uncommitted changes; commit them or pass --allow-dirty")
    scripts = ARTIFACT / "scripts"
    before = pinned_entries(read_json(HASHES)) if HASHES.exists() else {}
    if not args.no_build:
        run(
            ["python3", scripts / "build.py", "--skip-aerie", "--skip-natver"],
            capture=False,
        )
    gates = run(
        ["python3", scripts / "gates.py", "--write-expected", "--no-sde"],
        check=False,
        capture=False,
    )
    if gates.returncode != 0:
        raise SystemExit("gates failed; sources.lock is unchanged (inspect expected/hashes.json)")
    after = pinned_entries(read_json(HASHES))
    for key in sorted(after):
        old = before.get(key)
        state = "new" if old is None else ("unchanged" if old == after[key] else "CHANGED")
        print(f"  ML-DSA-{key[0]} {key[1]:<8} {after[key][0]} B {after[key][1][:16]}  {state}")
    pin_lock(head, args.archive_url)
    print(f"pinned sources.lock and expected/hashes.json to {head}")
    print("commit artifact/sources.lock and artifact/expected/hashes.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
