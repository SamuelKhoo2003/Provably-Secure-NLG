#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
OUT_DIR="${OUT_DIR:-toy_results/benchmark_large}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found at $PYTHON_BIN" >&2
  echo "Create the environment first: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

echo "Writing outputs to: $OUT_DIR"

"$PYTHON_BIN" -m toy_certificate.experiments visualize \
  --K "${VIS_K:-7}" \
  --N "${VIS_N:-3}" \
  --L "${VIS_L:-4}" \
  --T "${VIS_T:-5}" \
  --delta-stab "${VIS_DELTA_STAB:-${VIS_DELTA:-0.2}}" \
  --delta-val "${VIS_DELTA_VAL:-${VIS_DELTA:-0.2}}" \
  --target-bias "${TARGET_BIAS:-0.2}" \
  --seed "${SEED:-0}" \
  --save-dir "${VIS_OUT_DIR:-toy_results/default_instance}"

"$PYTHON_BIN" -m toy_certificate.experiments benchmark \
  --Ks "${KS:-5,7,9,11,15}" \
  --Ns "${NS:-2,3,5,8,12}" \
  --lengths "${LENGTHS:-2,4,8}" \
  --Ts "${TS:-3,5,8,12}" \
  --deltas "${DELTAS:-0.0,0.1,0.2,0.3,0.4}" \
  --target-bias "${TARGET_BIAS:-0.2}" \
  --seed "${SEED:-0}" \
  --save-dir "$OUT_DIR"

echo
echo "CSV:   $OUT_DIR/benchmark_results.csv"
echo "Plots: $OUT_DIR/focused_*.svg"
