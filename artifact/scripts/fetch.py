#!/usr/bin/env python3
"""Fetch and verify the pinned sources."""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ARTIFACT,
    AERIE_TOOLCHAIN,
    BUILD,
    REPO,
    RUST_TOOLCHAIN,
    VENDOR,
    git_rev,
    load_toml,
    output,
    run,
    write_json,
)


def archive_sha(path, commit):
    proc = subprocess.Popen(
        ["git", "-C", str(path), "archive", "--format=tar.gz", commit],
        stdout=subprocess.PIPE,
    )
    digest = hashlib.sha256()
    assert proc.stdout is not None
    for chunk in iter(lambda: proc.stdout.read(1 << 20), b""):
        digest.update(chunk)
    proc.wait()
    return digest.hexdigest()


def clone(url, path, commit):
    if (path / ".git").exists():
        head = output(["git", "-C", str(path), "rev-parse", "HEAD"], default="")
        if head == commit:
            return
        dirty = output(["git", "-C", str(path), "status", "--porcelain"], default="")
        if dirty:
            raise RuntimeError(
                f"{path} is at {head[:12]} with local changes; refusing to overwrite"
            )
        run(["git", "-C", str(path), "fetch", "--quiet", "origin", commit], check=False)
        run(["git", "-C", str(path), "checkout", "--quiet", commit])
        return
    print(f"cloning {url}", flush=True)
    run(["git", "clone", "--quiet", url, str(path)])
    run(["git", "-C", str(path), "checkout", "--quiet", commit])


def check_cargo_lock(sources):
    # Path-patched crates (Akita, Flock) disappear from Cargo.lock, so also
    # read the workspace manifest, which carries the `rev` of every git
    # dependency.
    lock = (REPO / "Cargo.lock").read_text(encoding="utf-8")
    lock += (REPO / "Cargo.toml").read_text(encoding="utf-8")
    missing = []
    for source in sources:
        if source.get("in_repo"):
            continue
        if "dependency" in source.get("role", ""):
            needle = source["commit"][:40]
            if needle not in lock:
                missing.append(source["name"])
    return missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-cargo", action="store_true", help="skip cargo fetch")
    parser.add_argument(
        "--alloul26",
        action="store_true",
        help="also install the toolchain of the optional Alloul26 comparison",
    )
    args = parser.parse_args()
    lock = load_toml(ARTIFACT / "sources.lock")
    sources = lock["source"]
    resolved = []
    for source in sources:
        name = source["name"]
        commit = source["commit"]
        if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
            raise SystemExit(f"{name}: commit must be a full 40-character hex SHA")
        if source.get("in_repo"):
            head = git_rev(REPO)
            status = output(["git", "-C", str(REPO), "status", "--porcelain"], default="")
            print(f"{name}: tree HEAD {head[:12]}, base {commit[:12]}")
            resolved.append(
                {
                    "name": name,
                    "kind": "in-repo",
                    "base_commit": commit,
                    "head_commit": head,
                    "dirty": bool(status),
                    # An anonymized download carries no git history to archive.
                    "tarball_sha256": archive_sha(REPO, commit) if head else None,
                }
            )
            continue
        if source.get("role") == "optional comparison":
            print(f"{name}: optional; fetched by its comparison script")
            resolved.append({"name": name, "kind": "optional", "commit": commit})
            continue
        if source.get("path", "").startswith("artifact/vendor/"):
            path = REPO / source["path"]
            clone(source["url"], path, commit)
            head = git_rev(path)
            if head != commit:
                raise SystemExit(f"{name}: HEAD {head} != pinned {commit}")
            print(f"{name}: {commit[:12]} ok")
            resolved.append(
                {
                    "name": name,
                    "kind": "clone",
                    "path": str(path),
                    "commit": head,
                    "tarball_sha256": archive_sha(path, commit),
                }
            )
            continue
        print(f"{name}: cargo-managed at {commit[:12]}")
        resolved.append({"name": name, "kind": "cargo", "commit": commit})

    missing = check_cargo_lock(sources)
    if missing:
        raise SystemExit(
            "Cargo.lock does not name the pinned revision of: " + ", ".join(missing)
        )

    if not args.no_cargo:
        # Install the pinned toolchains if rustup lacks them (a no-op when present).
        toolchains = [RUST_TOOLCHAIN, AERIE_TOOLCHAIN]
        if args.alloul26:
            toolchains += [s["toolchain"] for s in sources if s["name"] == "alloul26" and s.get("toolchain")]
        for toolchain in toolchains:
            run(["rustup", "toolchain", "install", "--profile", "minimal", toolchain])
        run(
            ["cargo", "fetch", "--locked"],
            cwd=REPO,
            env={"RUSTUP_TOOLCHAIN": RUST_TOOLCHAIN},
        )
        aerie = VENDOR / "aerie"
        if aerie.exists():
            run(
                ["cargo", "fetch", "--locked"],
                cwd=aerie,
                env={"RUSTUP_TOOLCHAIN": AERIE_TOOLCHAIN},
            )

    write_json(BUILD / "sources.json", {"sources": resolved})
    print("fetch: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
