#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

. "${ROOT_DIR}/scripts/_python.sh"
PYTHON_BIN="$(resolve_python_bin)"
OUT_DIR="${OUT_DIR:-toy_results/benchmark_large}"
CSV_PATH="${CSV_PATH:-$OUT_DIR/benchmark_results.csv}"

if [[ ! -f "$CSV_PATH" ]]; then
  echo "Benchmark CSV not found at $CSV_PATH" >&2
  echo "Run ./scripts/data.sh first, or set CSV_PATH to an existing CSV." >&2
  exit 1
fi

echo "Reading benchmark data from: $CSV_PATH"
echo "Writing plots to: $OUT_DIR"

"$PYTHON_BIN" -m toy_certificate.experiments plot-csv \
  --csv "$CSV_PATH" \
  --save-dir "$OUT_DIR"

echo
echo "Plots: $OUT_DIR/*.svg"
