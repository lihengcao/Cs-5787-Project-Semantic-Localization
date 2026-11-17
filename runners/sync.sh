#!/usr/bin/env bash
# Usage:
#   ./runners/sync.sh [options] "message"   Strip colab.ipynb outputs (default), commit @, push liheng
#   ./runners/sync.sh [options] -s          Strip colab.ipynb outputs (default), squash @ into @-, push liheng
#   ./runners/sync.sh [options]             Strip colab.ipynb outputs (default), push liheng
#
# Options:
#   -k, --keep-outputs    Keep runners/colab.ipynb cell outputs & execution counts
#   --check-only          Run notebook output hygiene (unless -k) without committing or pushing
#   --repo-root DIR       Operate on DIR instead of the parent of runners/
#   -s                    Squash @ into @- (refuses to squash into trunk())
#   -h, --help            Show this help message
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
KEEP_OUTPUTS=0
CHECK_ONLY=0
SQUASH=0
MSG_PARTS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -k|--keep-outputs)
            KEEP_OUTPUTS=1
            shift
            ;;
        --check-only)
            CHECK_ONLY=1
            shift
            ;;
        --repo-root)
            [[ $# -ge 2 ]] || { echo "error: --repo-root requires a directory argument" >&2; exit 2; }
            REPO_ROOT="$(cd "$2" && pwd)"
            shift 2
            ;;
        --repo-root=*)
            REPO_ROOT="$(cd "${1#*=}" && pwd)"
            shift
            ;;
        -s)
            SQUASH=1
            shift
            ;;
        -h|--help)
            sed -n '2,12p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        --)
            shift
            MSG_PARTS+=("$@")
            break
            ;;
        -*)
            echo "error: unknown option: $1" >&2
            exit 2
            ;;
        *)
            MSG_PARTS+=("$1")
            shift
            ;;
    esac
done

if [[ "$SQUASH" -eq 1 && ${#MSG_PARTS[@]} -gt 0 ]]; then
    echo "error: cannot pass both -s and a commit message" >&2
    exit 2
fi

cd "$REPO_ROOT"

if [[ "$KEEP_OUTPUTS" -eq 0 && -f "$REPO_ROOT/runners/colab.ipynb" ]]; then
    if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
        PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
    else
        PYTHON_BIN="python3"
    fi

    "$PYTHON_BIN" - "$REPO_ROOT" <<'PY'
from __future__ import annotations

import json
import os
import sys

repo_root = sys.argv[1]
nb_path = os.path.join(repo_root, "runners", "colab.ipynb")

with open(nb_path, "r", encoding="utf-8") as f:
    raw = f.read()
nb = json.loads(raw)
dirty = False
for cell in nb.get("cells", []):
    if cell.get("cell_type") != "code":
        continue
    if cell.get("outputs"):
        cell["outputs"] = []
        dirty = True
    if cell.get("execution_count") is not None:
        cell["execution_count"] = None
        dirty = True
    meta = cell.get("metadata")
    if isinstance(meta, dict):
        for key in ("execution", "executionInfo", "outputId"):
            if key in meta:
                del meta[key]
                dirty = True
if dirty:
    indent = 2
    for line in raw.splitlines()[1:10]:
        stripped = line.lstrip(" ")
        if stripped and len(line) > len(stripped):
            indent = len(line) - len(stripped)
            break
    with open(nb_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(nb, indent=indent, ensure_ascii=False) + "\n")
    print(f"sync.sh: stripped cell outputs from {os.path.relpath(nb_path, repo_root)}")
PY
fi

if [[ "$CHECK_ONLY" -eq 1 ]]; then
    exit 0
fi

if [[ "$SQUASH" -eq 1 ]]; then
    [[ -z "$(jj log -r '@- & ::trunk()' --no-graph -T 'commit_id')" ]] || { echo "error: refusing to squash into trunk()" >&2; exit 1; }
    if [[ "$(jj log -r @ --no-graph -T 'empty')" == "false" ]]; then
        jj squash -u
    fi
    jj bookmark set liheng -r @-
elif [[ ${#MSG_PARTS[@]} -gt 0 ]]; then
    jj commit -m "${MSG_PARTS[*]}"
    jj bookmark set liheng -r @-
elif [[ "$(jj log -r @ --no-graph -T 'empty')" == "false" ]]; then
    echo 'error: @ has uncommitted changes; pass "./runners/sync.sh \"msg\"" or "./runners/sync.sh -s"' >&2
    exit 1
elif [[ -n "$(jj log -r 'liheng..@-' --no-graph -T 'commit_id' 2>/dev/null || true)" ]]; then
    jj bookmark set liheng -r @-
fi

jj git push --bookmark liheng
