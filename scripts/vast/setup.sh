#!/usr/bin/env bash
set -euo pipefail

# Minimal dependency setup for Vast.ai CUDA instances.
# Assumes you're using a PyTorch CUDA base image (recommended), so torch is already installed.

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "/venv/main/bin/python" ]]; then
    PYTHON_BIN="/venv/main/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Error: neither 'python' nor 'python3' found on PATH." >&2
    exit 1
  fi
fi

"$PYTHON_BIN" - <<'PY'
import sys
print("python:", sys.version.replace("\n"," "))
try:
    import torch
    print("torch:", torch.__version__, "| cuda:", torch.version.cuda, "| cuda_available:", torch.cuda.is_available())
except Exception as e:
    print("torch: MISSING/ERROR:", e)
    print("Tip: Use a PyTorch CUDA image, e.g. pytorch/pytorch:*cuda*")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

"$PYTHON_BIN" -m pip install -U pip wheel setuptools

# Core runtime deps for onyx.train (torch is expected from the base image).
"$PYTHON_BIN" -m pip install -U transformers psutil

# Optional: experiment tracking
if [[ "${INSTALL_WANDB:-0}" == "1" ]]; then
  "$PYTHON_BIN" -m pip install -U wandb
fi

echo "Done."
