#!/usr/bin/env bash
# Idempotent fresh-VM bootstrap for SceneAligner (Linux GPU VM or local Unix devbox).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
    FORCE=1
fi

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

requirements_hash() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum requirements.txt | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 requirements.txt | awk '{print $1}'
    else
        python3 -c 'import hashlib; print(hashlib.sha256(open("requirements.txt", "rb").read()).hexdigest())'
    fi
}

has_7z() {
    command -v 7z >/dev/null 2>&1 || command -v 7zz >/dev/null 2>&1 || command -v 7za >/dev/null 2>&1
}

has_libgl() {
    if [[ "$(uname -s)" != "Linux" ]]; then
        return 0
    fi
    if command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q 'libGL\.so\.1'; then
        return 0
    fi
    [[ -f /usr/lib/x86_64-linux-gnu/libGL.so.1 || -f /usr/lib/aarch64-linux-gnu/libGL.so.1 ]]
}

REQ_HASH="$(requirements_hash)"
STAMP_FILE=".venv/.scenealigner_ready"

if [[ "$FORCE" -eq 0 ]] \
    && [[ -x .venv/bin/python ]] \
    && [[ -f "$STAMP_FILE" ]] \
    && [[ "$(cat "$STAMP_FILE")" == "$REQ_HASH" ]] \
    && command -v aria2c >/dev/null 2>&1 \
    && has_7z; then
    exit 0
fi

SUDO=""
if [[ "$(id -u)" -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
fi

run_apt() {
    if [[ -n "$SUDO" ]]; then
        "$SUDO" env DEBIAN_FRONTEND=noninteractive apt-get "$@"
    else
        env DEBIAN_FRONTEND=noninteractive apt-get "$@"
    fi
}

if command -v apt-get >/dev/null 2>&1; then
    if ! command -v aria2c >/dev/null 2>&1 \
        || ! has_7z \
        || ! command -v curl >/dev/null 2>&1 \
        || ! command -v git >/dev/null 2>&1 \
        || ! has_libgl; then
        echo "[setup_vm] Installing system packages (aria2, p7zip-full, curl, git, libgl1)..."
        run_apt update -qq
        run_apt install -y -qq p7zip-full aria2 curl ca-certificates git libgl1
    fi
fi

if ! command -v uv >/dev/null 2>&1 && command -v curl >/dev/null 2>&1; then
    echo "[setup_vm] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

need_venv=0
if [[ ! -x .venv/bin/python ]]; then
    need_venv=1
elif ! .venv/bin/python -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
    need_venv=1
fi

if [[ "$need_venv" -eq 1 ]]; then
    echo "[setup_vm] Creating Python 3.11 virtual environment in .venv..."
    if command -v uv >/dev/null 2>&1; then
        uv venv --clear --python 3.11 --seed .venv
    else
        PY_BIN=""
        for cand in python3.11 python3.12 python3.13 python3; do
            if command -v "$cand" >/dev/null 2>&1 \
                && "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
                PY_BIN="$cand"
                break
            fi
        done
        if [[ -z "$PY_BIN" ]]; then
            echo "[setup_vm] error: Python >= 3.11 (or curl for uv) is required." >&2
            exit 1
        fi
        if ! "$PY_BIN" -c 'import ensurepip' >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
            run_apt update -qq
            run_apt install -y -qq python3-venv
        fi
        "$PY_BIN" -m venv --clear .venv
    fi
fi

pip_install() {
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python .venv/bin/python "$@" || .venv/bin/python -m pip install "$@"
    else
        .venv/bin/python -m pip install "$@"
    fi
}

HAS_GPU=0
if [[ "${FORCE_CUDA:-0}" == "1" ]]; then
    HAS_GPU=1
elif [[ "$(uname -s)" == "Linux" ]] \
    && [[ "$(uname -m)" == "x86_64" ]] \
    && command -v nvidia-smi >/dev/null 2>&1 \
    && nvidia-smi >/dev/null 2>&1; then
    HAS_GPU=1
fi

need_torch=0
if ! .venv/bin/python -c 'import torch, torchvision' >/dev/null 2>&1; then
    need_torch=1
elif [[ "$HAS_GPU" -eq 1 ]] \
    && ! .venv/bin/python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
    need_torch=1
fi

if [[ "$need_torch" -eq 1 ]]; then
    if [[ "$HAS_GPU" -eq 1 ]]; then
        TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
        echo "[setup_vm] Installing PyTorch 2.7.1 + CUDA (index: $TORCH_INDEX_URL)..."
        pip_install "torch==2.7.1" "torchvision==0.22.1" --index-url "$TORCH_INDEX_URL"
    else
        echo "[setup_vm] Installing PyTorch..."
        pip_install torch torchvision
    fi
fi

echo "[setup_vm] Installing project dependencies from requirements.txt..."
pip_install -r requirements.txt pytest

.venv/bin/python -c "import torch, cv2, peft, transformers, shapely, scenealigner; print(f'[setup_vm] Ready: torch={torch.__version__} cuda={torch.cuda.is_available()}')"

if [[ "$HAS_GPU" -eq 1 ]] \
    && ! .venv/bin/python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
    echo "[setup_vm] WARNING: nvidia-smi succeeded, but torch.cuda.is_available() is False!" >&2
fi

printf '%s\n' "$REQ_HASH" > "$STAMP_FILE"
