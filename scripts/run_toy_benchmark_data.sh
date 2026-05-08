#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
OUT_DIR="${OUT_DIR:-toy_results/benchmark_large}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found at $PYTHON_BIN" >&2
  echo "Create the environment first: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

echo "Writing benchmark data to: $OUT_DIR"

"$PYTHON_BIN" -m toy_certificate.experiments benchmark \
  --Ks "${KS:-3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}" \
  --Ns "${NS:-2,3,4,5,6,7,8,9,10,11,12}" \
  --lengths "${LENGTHS:-2,3,4,5,6,7,8,9,10}" \
  --Ts "${TS:-3,4,5,6,7,8,9,10,11,12}" \
  --deltas "${DELTAS:-0.0,0.1,0.2,0.3,0.4}" \
  --target-bias "${TARGET_BIAS:-0.2}" \
  --influence-mode "${INFLUENCE_MODE:-dense}" \
  --seed "${SEED:-0}" \
  --save-dir "$OUT_DIR"

echo
echo "CSV: $OUT_DIR/benchmark_results.csv"
