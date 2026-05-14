#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

. "${ROOT_DIR}/scripts/_python.sh"
PYTHON_BIN="$(resolve_python_bin)"
OUT_DIR="${OUT_DIR:-toy_results/default_instance}"

echo "Writing visualization outputs to: $OUT_DIR"

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
echo "Instance plots: $OUT_DIR/*.svg"
