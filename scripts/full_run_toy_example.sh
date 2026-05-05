#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
OUT_DIR="${OUT_DIR:-toy_results/full_run/benchmark}"
VIS_OUT_DIR="${VIS_OUT_DIR:-toy_results/full_run/instance}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found at $PYTHON_BIN" >&2
  echo "Create the environment first: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

echo "== Toy certificate full run =="
echo "Python:       $PYTHON_BIN"
echo "Instance dir: $VIS_OUT_DIR"
echo "Benchmark:    $OUT_DIR"
echo

echo "== 1/5 Compile check =="
"$PYTHON_BIN" -m compileall toy_certificate tests
echo

echo "== 2/5 Test bench =="
"$PYTHON_BIN" -m unittest discover
echo

echo "== 3/5 Instance visualization =="
"$PYTHON_BIN" -m toy_certificate.experiments visualize \
  --K "${VIS_K:-7}" \
  --N "${VIS_N:-3}" \
  --L "${VIS_L:-4}" \
  --T "${VIS_T:-5}" \
  --delta-stab "${VIS_DELTA_STAB:-${VIS_DELTA:-0.2}}" \
  --delta-val "${VIS_DELTA_VAL:-${VIS_DELTA:-0.2}}" \
  --target-bias "${TARGET_BIAS:-0.2}" \
  --influence-mode "${INFLUENCE_MODE:-dense}" \
  --seed "${SEED:-0}" \
  --save-dir "$VIS_OUT_DIR"
echo

echo "== 4/5 Full benchmark =="
"$PYTHON_BIN" -m toy_certificate.experiments benchmark \
  --Ks "${KS:-5,7,9,11,15}" \
  --Ns "${NS:-2,3,5,8,12}" \
  --lengths "${LENGTHS:-2,4,8}" \
  --Ts "${TS:-3,5,8,12}" \
  --deltas "${DELTAS:-0.0,0.1,0.2,0.3,0.4}" \
  --target-bias "${TARGET_BIAS:-0.2}" \
  --influence-mode "${INFLUENCE_MODE:-dense}" \
  --seed "${SEED:-0}" \
  --save-dir "$OUT_DIR"
echo

echo "== 5/5 Replot generated CSV =="
"$PYTHON_BIN" -m toy_certificate.experiments plot-csv \
  --csv "$OUT_DIR/benchmark_results.csv" \
  --save-dir "$OUT_DIR"
echo

echo "Done."
echo "Instance plots:  $VIS_OUT_DIR/*.svg"
echo "Benchmark CSV:   $OUT_DIR/benchmark_results.csv"
echo "Benchmark plots: $OUT_DIR/*.svg"
