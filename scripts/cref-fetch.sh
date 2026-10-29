#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}/cref-src"
mkdir -p "$TARGET_DIR"

DILITHIUM_REV="444cdcc84eb36b66fe27b3a2529ee48f6d8150c2"
HYBRID_REV="65e060ff3108a7daf7b8002bc62e025d2b8b3784"

# 1. dilithium
if [[ -d "/home/jipengzhang/code/dilithium" ]] && [[ "$(git -C /home/jipengzhang/code/dilithium rev-parse HEAD 2>/dev/null || true)" == "$DILITHIUM_REV" ]]; then
    ln -sfn /home/jipengzhang/code/dilithium "$TARGET_DIR/dilithium"
elif [[ ! -d "$TARGET_DIR/dilithium" ]]; then
    git clone https://github.com/pq-crystals/dilithium.git "$TARGET_DIR/dilithium"
    git -C "$TARGET_DIR/dilithium" checkout "$DILITHIUM_REV"
fi

# 2. Hybrid-SHA-3
if [[ -d "/home/jipengzhang/code/Hybrid-SHA-3" ]] && [[ "$(git -C /home/jipengzhang/code/Hybrid-SHA-3 rev-parse HEAD 2>/dev/null || true)" == "$HYBRID_REV" ]]; then
    ln -sfn /home/jipengzhang/code/Hybrid-SHA-3 "$TARGET_DIR/Hybrid-SHA-3"
elif [[ ! -d "$TARGET_DIR/Hybrid-SHA-3" ]]; then
    git clone https://github.com/xurq5953/Hybrid-SHA-3-and-its-application-to-FSwA-signatures.git "$TARGET_DIR/Hybrid-SHA-3"
    git -C "$TARGET_DIR/Hybrid-SHA-3" checkout "$HYBRID_REV"
fi

echo "AGG_CREF_SRC prepared at $TARGET_DIR"
echo "export AGG_CREF_SRC=\"$TARGET_DIR\""
