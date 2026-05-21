#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

. "${ROOT_DIR}/scripts/_python.sh"
PYTHON_BIN="$(resolve_python_bin)"
OUT_DIR="${OUT_DIR:-toy_results/smoke/instance}"
STABILITY_COMPETITOR_MODE="${STABILITY_COMPETITOR_MODE:-runner_up}"

echo "Writing visualization outputs to: $OUT_DIR"

"$PYTHON_BIN" -m toy_certificate.experiments visualize \
  --K "${VIS_K:-4}" \
  --N "${VIS_N:-2}" \
  --L "${VIS_L:-3}" \
  --T "${VIS_T:-4}" \
  --delta-stab "${VIS_DELTA_STAB:-${VIS_DELTA:-0.2}}" \
  --delta-val "${VIS_DELTA_VAL:-${VIS_DELTA:-0.2}}" \
  --target-bias "${TARGET_BIAS:-0.3}" \
  --influence-mode "${INFLUENCE_MODE:-dense}" \
  --stability-competitor-mode "$STABILITY_COMPETITOR_MODE" \
  --seed "${SEED:-0}" \
  --save-dir "$OUT_DIR"

echo
echo "Instance plots: $OUT_DIR/*.svg"
