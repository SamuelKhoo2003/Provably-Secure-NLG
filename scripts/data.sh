#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

. "${ROOT_DIR}/scripts/_python.sh"
PYTHON_BIN="$(resolve_python_bin)"
OUT_DIR="${OUT_DIR:-toy_results/benchmark_large}"

echo "Writing benchmark data to: $OUT_DIR"

"$PYTHON_BIN" -m toy_certificate.experiments benchmark \
  --Ks "${KS:-5,10,15,20,25}" \
  --Ns "${NS:-3,5,7,9,11}" \
  --lengths "${LENGTHS:-3,6,9,12}" \
  --Ts "${TS:-3,6,9,12}" \
  --deltas "${DELTAS:-0.0,0.25,0.5}" \
  --target-bias "${TARGET_BIAS:-0.2}" \
  --influence-mode "${INFLUENCE_MODE:-dense}" \
  --stability-competitor-mode "${STABILITY_COMPETITOR_MODE:-all}" \
  --seed "${SEED:-0}" \
  --save-dir "$OUT_DIR"

echo
echo "CSV: $OUT_DIR/benchmark_results.csv"
