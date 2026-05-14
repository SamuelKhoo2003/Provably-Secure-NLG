#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
VIS_OUT_DIR="${VIS_OUT_DIR:-toy_results/check_run/instance}"
OUT_DIR="${OUT_DIR:-$VIS_OUT_DIR}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found at $PYTHON_BIN" >&2
  echo "Create the environment first: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

echo "== Toy certificate check run =="
echo "Python:       $PYTHON_BIN"
echo "Instance dir: $VIS_OUT_DIR"
echo

echo "== 1/3 Compile check =="
"$PYTHON_BIN" -m compileall toy_certificate tests
echo

echo "== 2/3 Test bench =="
"$PYTHON_BIN" -m unittest discover
echo

echo "== 3/3 Instance visualization =="
"$PYTHON_BIN" -m toy_certificate.experiments visualize \
  --K "${VIS_K:-20}" \
  --N "${VIS_N:-12}" \
  --L "${VIS_L:-10}" \
  --T "${VIS_T:-12}" \
  --delta-stab "${VIS_DELTA_STAB:-${VIS_DELTA:-0.2}}" \
  --delta-val "${VIS_DELTA_VAL:-${VIS_DELTA:-0.2}}" \
  --target-bias "${TARGET_BIAS:-0.2}" \
  --influence-mode "${INFLUENCE_MODE:-dense}" \
  --stability-competitor-mode "${STABILITY_COMPETITOR_MODE:-all}" \
  --seed "${SEED:-0}" \
  --save-dir "$OUT_DIR"
echo

echo "Done."
echo "Instance plots: $VIS_OUT_DIR/*.svg"
