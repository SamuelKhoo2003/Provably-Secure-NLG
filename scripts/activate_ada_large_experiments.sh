#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "This script should be sourced, not executed:" >&2
  echo "  source scripts/activate_ada_large_experiments.sh" >&2
  exit 1
fi

if [[ -z "${USER:-}" ]]; then
  echo "ERROR: USER is not set." >&2
  return 1
fi

export FYP_BITBUCKET_ROOT="${FYP_BITBUCKET_ROOT:-/vol/bitbucket/$USER/Provably-Secure-NLG}"
export FYP_LARGE_OUTPUT_ROOT="${FYP_LARGE_OUTPUT_ROOT:-$FYP_BITBUCKET_ROOT/outputs/large_experiments}"
export FYP_LARGE_VENV_DIR="${FYP_LARGE_VENV_DIR:-$FYP_BITBUCKET_ROOT/venvs/large-experiments}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$FYP_BITBUCKET_ROOT/pip-cache}"
export TORCH_HOME="${TORCH_HOME:-$FYP_BITBUCKET_ROOT/torch-cache}"
export HF_HOME="${HF_HOME:-$FYP_BITBUCKET_ROOT/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$FYP_BITBUCKET_ROOT/cache}"
export MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$FYP_BITBUCKET_ROOT/model-cache}"

echo "Ada large-experiment environment configured:"
echo "  FYP_BITBUCKET_ROOT=$FYP_BITBUCKET_ROOT"
echo "  FYP_LARGE_OUTPUT_ROOT=$FYP_LARGE_OUTPUT_ROOT"
echo "  FYP_LARGE_VENV_DIR=$FYP_LARGE_VENV_DIR"
