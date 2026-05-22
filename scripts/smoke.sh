#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

. "${ROOT_DIR}/scripts/_python.sh"
. "${ROOT_DIR}/scripts/_config.sh"
PYTHON_BIN="$(resolve_python_bin)"
CONFIG="${CONFIG:-configs/smoke.yaml}"
VIS_OUT_DIR="${VIS_OUT_DIR:-${OUT_DIR:-$(config_value output_dir outputs/smoke)}}"
OUT_DIR="${OUT_DIR:-$VIS_OUT_DIR}"
STABILITY_COMPETITOR_MODE="${STABILITY_COMPETITOR_MODE:-$(config_value stability_competitor_mode all)}"

echo "== Toy certificate smoke run =="
echo "Python:       $PYTHON_BIN"
echo "Config:       $CONFIG"
echo "Instance dir: $VIS_OUT_DIR"
echo "Stability:    $STABILITY_COMPETITOR_MODE"
echo

if [[ "${RUN_TESTS:-0}" == "1" ]]; then
  echo "== Optional compile check =="
  "$PYTHON_BIN" -m compileall toy_certificate tests
  echo

  echo "== Optional unit tests =="
  "$PYTHON_BIN" -m unittest discover
  echo
fi

echo "== Smoke instance visualization =="
"$PYTHON_BIN" -m toy_certificate.experiments visualize \
  --K "${VIS_K:-$(config_first_value K 4)}" \
  --N "${VIS_N:-$(config_first_value N 2)}" \
  --L "${VIS_L:-$(config_first_value L 2)}" \
  --T "${VIS_T:-$(config_first_value T 4)}" \
  --delta-stab "${VIS_DELTA_STAB:-${VIS_DELTA:-$(config_value delta_stab 0.2)}}" \
  --delta-val "${VIS_DELTA_VAL:-${VIS_DELTA:-$(config_value delta_val 0.2)}}" \
  --target-bias "${TARGET_BIAS:-$(config_value target_bias 0.3)}" \
  --influence-mode "${INFLUENCE_MODE:-$(config_value influence_mode dense)}" \
  --stability-competitor-mode "$STABILITY_COMPETITOR_MODE" \
  --seed "${SEED:-$(config_value seed 0)}" \
  --save-dir "$OUT_DIR"
echo

mkdir -p "$OUT_DIR" outputs/logs
{
  echo "smoke_status=passed"
  echo "config=$CONFIG"
  echo "output_dir=$OUT_DIR"
  echo "stability_competitor_mode=$STABILITY_COMPETITOR_MODE"
} > "$OUT_DIR/smoke_summary.txt"

echo "Done."
echo "Instance plots: $VIS_OUT_DIR/*.svg"
echo "Summary: $VIS_OUT_DIR/smoke_summary.txt"
echo "Run optional checks with: RUN_TESTS=1 ./scripts/check.sh"
