#!/usr/bin/env bash
# Fetch the JITServe artifact and copy its traces into data/traces/.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$REPO_ROOT/data/traces"
if [ ! -d "$REPO_ROOT/data/jitserve" ]; then
  git clone --depth 1 https://github.com/UIUC-MLSys/JITServe.git "$REPO_ROOT/data/jitserve"
fi
cp "$REPO_ROOT/data/jitserve/traces/lmsys.json" "$REPO_ROOT/data/traces/" 2>/dev/null || true
cp "$REPO_ROOT/data/jitserve/traces/deepresearch_filter.jsonl" "$REPO_ROOT/data/traces/" 2>/dev/null || true
cp -r "$REPO_ROOT/data/jitserve/traces/burst" "$REPO_ROOT/data/traces/" 2>/dev/null || true
echo "traces staged under $REPO_ROOT/data/traces"
