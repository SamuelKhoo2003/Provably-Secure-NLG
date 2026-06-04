#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${USER:-}" ]]; then
  echo "ERROR: USER is not set." >&2
  exit 1
fi

FYP_BITBUCKET_ROOT="${FYP_BITBUCKET_ROOT:-/vol/bitbucket/$USER/Provably-Secure-NLG}"
FYP_LARGE_OUTPUT_ROOT="${FYP_LARGE_OUTPUT_ROOT:-$FYP_BITBUCKET_ROOT/outputs/large_experiments}"
VENV_DIR="${FYP_LARGE_VENV_DIR:-$FYP_BITBUCKET_ROOT/venvs/large-experiments}"

PIP_CACHE_DIR="${PIP_CACHE_DIR:-$FYP_BITBUCKET_ROOT/pip-cache}"
TORCH_HOME="${TORCH_HOME:-$FYP_BITBUCKET_ROOT/torch-cache}"
HF_HOME="${HF_HOME:-$FYP_BITBUCKET_ROOT/hf-cache}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$FYP_BITBUCKET_ROOT/cache}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$FYP_BITBUCKET_ROOT/model-cache}"

mkdir -p \
  "$FYP_BITBUCKET_ROOT/venvs" \
  "$FYP_LARGE_OUTPUT_ROOT" \
  "$PIP_CACHE_DIR" \
  "$TORCH_HOME" \
  "$HF_HOME" \
  "$XDG_CACHE_HOME" \
  "$MODEL_CACHE_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Creating virtualenv: $VENV_DIR"
  python3 -m virtualenv "$VENV_DIR"
else
  echo "Virtualenv already exists: $VENV_DIR"
fi

cat <<EOF

Ada large-experiment storage is ready.

Activate it with:

  source "$VENV_DIR/bin/activate"
  source "$(pwd)/scripts/activate_ada_large_experiments.sh"

Environment for this setup:

  export FYP_BITBUCKET_ROOT="$FYP_BITBUCKET_ROOT"
  export FYP_LARGE_OUTPUT_ROOT="$FYP_LARGE_OUTPUT_ROOT"
  export FYP_LARGE_VENV_DIR="$VENV_DIR"
  export PIP_CACHE_DIR="$PIP_CACHE_DIR"
  export TORCH_HOME="$TORCH_HOME"
  export HF_HOME="$HF_HOME"
  export TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
  export XDG_CACHE_HOME="$XDG_CACHE_HOME"
  export MODEL_CACHE_DIR="$MODEL_CACHE_DIR"

Install lightweight non-Torch dependencies with:

  pip install numpy pandas matplotlib pyyaml scipy gurobipy

PyTorch is not installed automatically. Install it only if you need real
model/adapter inference.
EOF
