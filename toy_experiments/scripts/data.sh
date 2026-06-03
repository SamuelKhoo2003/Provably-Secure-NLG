#!/usr/bin/env bash
set -euo pipefail

TOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$TOY_DIR/.." && pwd)"
cd "$REPO_ROOT"

. "${TOY_DIR}/scripts/_python.sh"
PYTHON_BIN="$(resolve_python_bin)"

if [[ -z "${CONFIG:-}" ]]; then
  echo "ERROR: CONFIG is required." >&2
  echo "Example: CONFIG=toy_experiments/configs/medium.yaml ./toy_experiments/scripts/data.sh" >&2
  exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config file not found: $CONFIG" >&2
  exit 1
fi

args=(benchmark --config "$CONFIG")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  args+=(--dry-run)
fi

if [[ "${VERBOSE:-0}" == "1" ]]; then
  args+=(--verbose)
fi

echo "Config: $CONFIG"
"$PYTHON_BIN" -m toy_experiments.experiments "${args[@]}"
