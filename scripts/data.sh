#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

. "${ROOT_DIR}/scripts/_python.sh"
PYTHON_BIN="$(resolve_python_bin)"

if [[ -z "${CONFIG:-}" ]]; then
  echo "ERROR: CONFIG is required." >&2
  echo "Example: CONFIG=configs/medium.yaml ./scripts/data.sh" >&2
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
"$PYTHON_BIN" -m toy_certificate.experiments "${args[@]}"
